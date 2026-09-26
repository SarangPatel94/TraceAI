import os
import gc
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load operational environmental parameters
load_dotenv()

# Extract and validate environment configurations
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_LLM_MODEL = "qwen/qwen3.8-27b"

# --- Confidence Router configuration -----------------------------------------
# Retrieval-side thresholds: tuned against your own fused (RRF) score distribution.
# Raise RETRIEVAL_MIN_TOP_SCORE if you see high-confidence answers on weak matches;
# lower RETRIEVAL_SCORE_GAP_THRESHOLD if strong queries are being escalated too often.
RETRIEVAL_MIN_TOP_SCORE = float(os.getenv("RETRIEVAL_MIN_TOP_SCORE", "0.35"))
RETRIEVAL_SCORE_GAP_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_GAP_THRESHOLD", "0.15"))

# Whether to spend a second, cheap Groq call asking the model to self-audit its own
# answer against the retrieved context. Adds latency + cost per query; set to
# "false" to route on the retrieval signal alone.
ENABLE_LLM_CONFIDENCE_CHECK = os.getenv("ENABLE_LLM_CONFIDENCE_CHECK", "true").strip().lower() in ("1", "true", "yes")

# Who/where to point people when the router escalates.
TECH_LEAD_CONTACT = os.getenv("TECH_LEAD_CONTACT", "#eng-tech-leads on repo")


class CodebaseQAEngine:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ Missing 'GROQ_API_KEY' inside environment variables.")

        print(f"⏳ Syncing local query embedding engine [{DENSE_MODEL_NAME}]...")
        self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    def search_codebase(self, query: str, limit: int = 5):
        """Executes full hybrid dense + sparse vector lookups against Qdrant."""

        # 1. Compute dense embedding matching the prefix strategy from ingestion
        instructional_query = "Represent this sentence for searching relevant code snippets: " + query
        dense_vector = self.embedding_model.encode(instructional_query).tolist()

        # 2. Query Qdrant with parallel prefetch arrays using built-in Document parsers
        response = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # Dense lookup configuration
                models.Prefetch(
                    query=dense_vector,
                    using="text-dense",
                    limit=limit * 2
                ),
                # Sparse structural text matching configuration
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25"
                    ),
                    using="text-sparse",
                    limit=limit * 2
                )
            ],
            # Use Reciprocal Rank Fusion (RRF) to merge structural text vs dense semantic scores
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        return response.points

    # --- Confidence Router: signal #1 (retrieval) ----------------------------
    def _score_retrieval(self, points):
        """
        Derives a confidence signal purely from the fused Qdrant scores.

        Two failure modes get flagged LOW:
          - the best match is a weak match in absolute terms (top_score below floor)
          - the ranking is flat, i.e. the top chunk isn't meaningfully better than the
            weakest chunk pulled in (score_gap below threshold), which usually means
            retrieval didn't find anything clearly relevant and just returned "closest of
            a bad bunch."
        """
        if not points:
            return "LOW", {"top_score": 0.0, "score_gap": 0.0, "reason": "No points retrieved."}

        scores = [p.score for p in points]
        top_score = scores[0]
        score_gap = (top_score - scores[-1]) if len(scores) > 1 else top_score

        if top_score < RETRIEVAL_MIN_TOP_SCORE:
            return "LOW", {
                "top_score": top_score,
                "score_gap": score_gap,
                "reason": f"Top fused score {top_score:.3f} is below the {RETRIEVAL_MIN_TOP_SCORE} floor."
            }
        if score_gap < RETRIEVAL_SCORE_GAP_THRESHOLD:
            return "LOW", {
                "top_score": top_score,
                "score_gap": score_gap,
                "reason": f"Score gap {score_gap:.3f} between best and weakest chunk is too small to trust the ranking."
            }
        return "HIGH", {
            "top_score": top_score,
            "score_gap": score_gap,
            "reason": "Retrieval ranking is well separated from a strong top match."
        }

    # --- Confidence Router: signal #2 (LLM self-assessment) ------------------
    def _assess_answer_confidence(self, query: str, answer: str, context_str: str):
        """
        Second, cheap Groq call: audits whether the retrieved context actually
        contains enough information to support the generated answer, rather than
        the model padding gaps with general/parametric knowledge.
        """
        judge_prompt = (
            "You are auditing an AI-generated answer for an internal codebase Q&A tool.\n"
            "Given the question, the retrieved code context, and the generated answer, decide whether "
            "the context ACTUALLY contains enough information to fully and correctly answer the question.\n"
            "Mark LOW if the answer relies on guessing, on general knowledge not present in the context, "
            "or if the context is incomplete, contradictory, or ambiguous with respect to the question.\n"
            "Respond with ONLY a JSON object, no other text: "
            "{\"confidence\": \"HIGH\"|\"LOW\", \"reason\": \"<one short sentence>\"}\n\n"
            f"Question:\n{query}\n\nRetrieved Context:\n{context_str}\n\nGenerated Answer:\n{answer}"
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            verdict = json.loads(completion.choices[0].message.content)
            confidence = str(verdict.get("confidence", "LOW")).strip().upper()
            if confidence not in ("HIGH", "LOW"):
                confidence = "LOW"
            return confidence, verdict.get("reason", "")
        except Exception as e:
            # Fail closed: if the auditor call itself breaks, don't silently assume high confidence.
            return "LOW", f"Self-assessment call failed, failing closed: {str(e)}"

    @staticmethod
    def _route_confidence(retrieval_confidence: str, llm_confidence: str) -> str:
        """Conservative combination: escalate unless every available signal says HIGH."""
        signals = [retrieval_confidence]
        if llm_confidence != "N/A":
            signals.append(llm_confidence)
        return "HIGH" if all(s == "HIGH" for s in signals) else "LOW"

    def ask(self, query: str, silent: bool = False):
        """
        Retrieves codebase context chunks, generates an answer with Groq, then runs the
        Confidence Router over both the retrieval quality and an LLM self-assessment of
        whether the context actually supports the answer.

        Always returns a dict so this method works both interactively (silent=False,
        prints progress + result) and programmatically from batch/evaluation flows
        (silent=True). Shape:

          {
            "status": "answered" | "escalated" | "error",
            "answer": str,              # the answer to show the user (or the escalation message)
            "citations": [str, ...],    # file paths referenced
            "confidence": {
                "overall": "HIGH" | "LOW",
                "retrieval": {"level": ..., "top_score": ..., "score_gap": ..., "reason": ...},
                "llm_self_assessment": {"level": ..., "reason": ...},
            },
            "contact": str,             # only present when escalated/error
            "partial_context": [str],   # only present when escalated/error
            "draft_answer": str | None, # the raw LLM answer, kept for audit even when escalated
          }
        """
        if not silent:
            print(f"\n🔍 Searching vector space for: '{query}'...")

        points = self.search_codebase(query, limit=6)

        if not points:
            if not silent:
                print("⚠️ No matching code blocks surfaced from the database vector indexes.")
            return {
                "status": "escalated",
                "answer": (
                    "⚠️ No relevant code was found for this question — please verify with a Tech Lead "
                    f"({TECH_LEAD_CONTACT}) rather than relying on a guess."
                ),
                "citations": [],
                "confidence": {
                    "overall": "LOW",
                    "retrieval": {"level": "LOW", "top_score": 0.0, "score_gap": 0.0, "reason": "No points retrieved."},
                    "llm_self_assessment": {"level": "N/A", "reason": "Skipped — no context to assess."},
                },
                "contact": TECH_LEAD_CONTACT,
                "partial_context": [],
                "draft_answer": None,
            }

        # Build context metadata payload string
        context_blocks, referenced_files = [], set()
        for point in points:
            path = point.payload["metadata"].get("file_path", "Unknown File")
            text = point.payload.get("text", "")
            referenced_files.add(path)
            context_blocks.append(f"--- START FILE SEGMENT: {path} ---\n{text}\n--- END FILE SEGMENT ---")

        context_str = "\n\n".join(context_blocks)

        # Retrieval-side confidence signal can be computed regardless of whether generation succeeds.
        retrieval_confidence, retrieval_meta = self._score_retrieval(points)

        system_instructions = (
            "You are a Principal Software Engineering Assistant specializing in codebase reasoning.\n"
            "Analyze the absolute file paths, dependencies, and code patterns provided below.\n"
            "Answer the user's inquiry accurately and concisely, based ONLY on the structural code blocks "
            "supplied, citing source paths inline using bracket notation [file_path].\n"
            "If you do not see the implementation details or answers in the context, clearly explain what is missing.\n"
            "Always target clean output, formatting answers using Markdown formatting."
        )
        user_prompt = f"Codebase Context:\n{context_str}\n\nUser Question:\n{query}"

        if not silent:
            print("⚡ Dispatching optimized payload context block to Groq architecture...")

        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature prevents creative logic modifications
                max_tokens=4096
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            error_msg = f"❌ Groq Completion Engine failure sequence triggered: {str(e)}"
            if not silent:
                print(error_msg)
            return {
                "status": "error",
                "answer": error_msg,
                "citations": sorted(referenced_files),
                "confidence": {
                    "overall": "LOW",
                    "retrieval": {"level": retrieval_confidence, **retrieval_meta},
                    "llm_self_assessment": {"level": "N/A", "reason": "Skipped — generation call failed."},
                },
                "contact": TECH_LEAD_CONTACT,
                "partial_context": sorted(referenced_files),
                "draft_answer": None,
            }
        finally:
            gc.collect()

        # --- Confidence Router: combine signals and decide how to respond ---
        if ENABLE_LLM_CONFIDENCE_CHECK:
            llm_confidence, llm_reason = self._assess_answer_confidence(query, answer, context_str)
        else:
            llm_confidence, llm_reason = "N/A", "LLM self-check disabled (ENABLE_LLM_CONFIDENCE_CHECK=false)."

        overall_confidence = self._route_confidence(retrieval_confidence, llm_confidence)
        confidence_report = {
            "overall": overall_confidence,
            "retrieval": {"level": retrieval_confidence, **retrieval_meta},
            "llm_self_assessment": {"level": llm_confidence, "reason": llm_reason},
        }

        if overall_confidence == "HIGH":
            result = {
                "status": "answered",
                "answer": answer,
                "citations": sorted(referenced_files),
                "confidence": confidence_report,
            }
            if not silent:
                print("\n🤖 Groq Engineering Response:")
                print("======================================================================")
                print(answer)
                print(f"\n✅ Confidence: HIGH  (retrieval={retrieval_confidence}, llm_self_check={llm_confidence})")
                print("======================================================================\n")
            return result

        # Low confidence: do not assert the draft answer as fact — escalate instead.
        escalation_answer = (
            "⚠️ Confidence too low to trust this answer automatically — please verify with a Tech Lead "
            f"({TECH_LEAD_CONTACT}) before relying on it.\n\n"
            "Partial context retrieved (unverified):\n"
            + "\n".join(f"- {p}" for p in sorted(referenced_files))
        )
        result = {
            "status": "escalated",
            "answer": escalation_answer,
            "citations": sorted(referenced_files),
            "confidence": confidence_report,
            "contact": TECH_LEAD_CONTACT,
            "partial_context": sorted(referenced_files),
            "draft_answer": answer,
        }
        if not silent:
            print("\n🚨 LOW CONFIDENCE — escalating instead of asserting an answer")
            print("======================================================================")
            print(escalation_answer)
            print(f"\nRetrieval confidence: {retrieval_confidence} ({retrieval_meta.get('reason')})")
            print(f"LLM self-assessment: {llm_confidence} ({llm_reason})")
            print("======================================================================\n")
        return result

    def evaluate_and_score(self, question: dict, answer: str, context_files: list):
        """Asks Groq to judge whether `answer` satisfactorily answers `question` (offline report judge)."""
        judge_prompt = (
            "Evaluate the response below. If the answer does not successfully satisfy the query, "
            "mark confidence as 'LOW', otherwise 'HIGH'. The question is a JSON object; return the "
            "evaluation in the given format.\n"
            f"Question: {question}\nAnswer: {answer}\nFiles: {', '.join(context_files)}\n"
            "Return JSON: {\"id\": <question_id>, \"type\": <question_type>, \"question\": <question>, "
            "\"answer\": <answer>, \"citations\": [<list of files>]"
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"❌ Judge completion failed: {str(e)}")
            return {
                "id": question.get("id"),
                "type": question.get("type"),
                "question": question.get("question"),
                "citations": [],
                "confidence": "None"
            }

    def run_evaluation(self, eval_dataset: dict, output_file: str = "evaluation_results.md"):
        """
        Runs every question in eval_dataset through ask() + evaluate_and_score() and writes a
        markdown report. Each row includes both the offline judge's confidence verdict
        (evaluate_and_score) and the live Confidence Router's own verdict (from ask()), so you
        can sanity-check that the two agree.
        """
        dataset = eval_dataset.get("evaluation_dataset", {})
        md_content = "# Evaluation Results\n\n"

        for category in dataset.get("categories", []):
            for q_obj in category.get("questions", []):
                print(f"File Data for eval : {q_obj}")
                q_obj.pop("answer", None)
                result = self.ask(q_obj.get("question"), silent=True)
                ans = result["answer"]
                files = result["citations"]
                response_data = self.evaluate_and_score(q_obj, ans, files)
                response_data["router_status"] = result["status"]
                response_data["router_confidence"] = result["confidence"]["overall"]
                md_content += str(response_data) + "\n"
                print(f"The content for the MD file {md_content}")
                time.sleep(75)

        if os.path.exists(output_file):
            base_name, extension = os.path.splitext(output_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_{timestamp}{extension}"
            print(f"🔄 File already exists! Renaming active evaluation summary output to: {output_file}")
        else:
            print(f"📝 Writing brand new evaluation results log to: {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)


if __name__ == "__main__":
    # Initialize implementation pipeline interface
    engine = CodebaseQAEngine()

    while True:
        sample_query = input("\n📊 Enter your BA query (or type 'exit'): ")
        if sample_query.lower() == 'exit':
            print("Thank you for using the bot see you again!!")
            break
        engine.ask(sample_query)
