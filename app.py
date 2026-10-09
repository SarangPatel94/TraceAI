import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from query_engine import CodebaseQAEngine, DENSE_MODEL_NAME
from ingest import BatchCodebasePipeline, SUPPORTED_EXTENSIONS

# Populated at startup by the lifespan handler below; kept out of global scope so nothing
# tries to use the engine/pipeline before the (slow) embedding model has actually loaded.
_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"⏳ Loading shared embedding model [{DENSE_MODEL_NAME}] once for both engine and pipeline...")
    shared_embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
    _state["engine"] = CodebaseQAEngine(embedding_model=shared_embedding_model)
    _state["pipeline"] = BatchCodebasePipeline(embedding_model=shared_embedding_model)
    print("✅ Codebase QA service ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Codebase QA Service",
    description=(
        "Thin API wrapper around the ingestion pipeline and the Confidence-Router / "
        "agent-loop query engine. See /ingest to (re)index a repo, folder, or file, "
        "and /ask to query it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schemas -------------------------------------------------------------------
class IngestRequest(BaseModel):
    path: str = Field(..., description="Local path to a repo root, a folder, or a single file to ingest.")
    recreate_collection: Optional[bool] = Field(
        None,
        description=(
            "Force whether to wipe+recreate the Qdrant collection first. Defaults to True only "
            "when `path` resolves to the configured REPO_BASE_PATH root; otherwise defaults to "
            "False, so ingesting one file/folder doesn't wipe everything else already indexed."
        ),
    )


class IngestResponse(BaseModel):
    target_path: str
    files_processed: int
    chunks_indexed: int
    files_skipped_unsupported: int
    collection_recreated: bool
    warnings: List[str] = []


class Question(BaseModel):
    id: str
    type: str
    question: str


class AskRequest(BaseModel):
    questions: List[Question]


class AnswerResult(BaseModel):
    id: str
    type: str
    question: str
    status: str  # "answered" | "escalated" | "error"
    answer: str
    citations: List[str]
    confidence: Dict[str, Any]
    agent_trace: List[Dict[str, Any]] = []
    contact: Optional[str] = None
    partial_context: Optional[List[str]] = None


class AskResponse(BaseModel):
    answers: List[AnswerResult]


# --- Endpoints -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Ingests a local repo root, folder, or single file. A single-file target has its
    extension validated up front (400 if unsupported, matching SUPPORTED_EXTENSIONS);
    inside a folder, unsupported files are silently skipped and counted instead, matching
    the pipeline's existing directory-walk behavior.
    """
    resolved_path = os.path.abspath(request.path)

    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {resolved_path}")

    if os.path.isfile(resolved_path):
        ext = os.path.splitext(resolved_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
            )

    pipeline: BatchCodebasePipeline = _state["pipeline"]
    try:
        summary = pipeline.run_ingestion(target_path=resolved_path, recreate_collection=request.recreate_collection)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(**summary)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Runs each question through the query engine (retrieval -> generation -> Confidence
    Router -> agent loop if needed). Questions are answered sequentially, in request order.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="`questions` must contain at least one question.")

    engine: CodebaseQAEngine = _state["engine"]
    answers = []
    for q in request.questions:
        result = engine.ask(q.question, silent=True)
        answers.append(AnswerResult(
            id=q.id,
            type=q.type,
            question=q.question,
            status=result["status"],
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            agent_trace=result.get("agent_trace", []),
            contact=result.get("contact"),
            partial_context=result.get("partial_context"),
        ))

    return AskResponse(answers=answers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
