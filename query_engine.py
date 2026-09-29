import os
import gc
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq

import git_utils

# Load operational environmental parameters
load_dotenv()

# Extract and validate environment configurations
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
REPO_BASE_PATH = os.getenv("REPO_BASE_PATH", "../TestRepo")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_LLM_MODEL = "qwen/qwen3.8-27b"

# --- Confidence Router configuration -----------------------------------------
# Retrieval-side thresholds: tuned against your own fused (RRF) score distribution.
RETRIEVAL_MIN_TOP_SCORE = float(os.getenv("RETRIEVAL_MIN_TOP_SCORE", "0.35"))
RETRIEVAL_SCORE_GAP_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_GAP_THRESHOLD", "0.15"))

# Whether to spend a second, cheap Groq call asking the model to self-audit its own
# answer against the retrieved context.
ENABLE_LLM_CONFIDENCE_CHECK = os.getenv("ENABLE_LLM_CONFIDENCE_CHECK", "true").strip().lower() in ("1", "true", "yes")

# Who/where to point people when the router escalates.
TECH_LEAD_CONTACT = os.getenv("TECH_LEAD_CONTACT", "#eng-tech-leads on Teams")

# --- Agent loop configuration --------------------------------------------------
# When a first-pass answer comes back LOW confidence, instead of escalating immediately
# we hand the model real tools and let it try to close the gap itself.
ENABLE_AGENT_LOOP = os.getenv("ENABLE_AGENT_LOOP", "true").strip().lower() in ("1", "true", "yes")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "3"))

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": (
                "Re-run hybrid dense+sparse semantic search against the codebase with a new, "
                "reformulated query. Use this when the currently retrieved context doesn't actually "
                "contain the answer — try being more specific, using different terminology, or "
                "targeting a narrower part of the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The new search query to run against the codebase.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_history",
            "description": (
                "Look up the recent git commit history (hash, author, date, commit message) for a "
                "specific file already surfaced in the retrieved context. Use this when the question "
                "is about who changed something, when, or why — information that isn't visible in a "
                "code snippet alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Repo-relative file path to look up, e.g. 'src/Pyz/Client/Foo/FooClient.php'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many recent commits to return.",
                        "default": 5,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]


class CodebaseQAEngine:
    def __init__(self, embedding_model: SentenceTransformer = None):
        if not GROQ_API_KEY:
            raise ValueError("❌ Missing 'GROQ_API_KEY' inside environment variables.")

        if embedding_model is not None:
            # Reuse an already-loaded model (e.g. shared with BatchCodebasePipeline by app.py)
            # instead of loading a second copy into memory.
            self.embedding_model = embedding_model
        else:
            print(f"⏳ Syncing local query embedding engine [{DENSE_MODEL_NAME}]...")
            self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        self.git_root = git_utils.detect_git_root(os.path.abspath(REPO_BASE_PATH)) if ENABLE_AGENT_LOOP else None
        if ENABLE_AGENT_LOOP and not self.git_root:
            print(f"⚠️ The agent's get_git_history tool will return empty results — "
                  f"'{REPO_BASE_PATH}' isn't inside a readable git repo (or git is unavailable).")

    def search_codebase(self, query: str, limit: int = 5):
        """Executes full hybrid dense + sparse vector lookups against Qdrant."""
        instructional_query = "Represent this sentence for searching relevant code snippets: " + query
        dense_vector = self.embedding_model.encode(instructional_query).tolist()

        response = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vector, using="text-dense", limit=limit * 2),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="text-sparse",
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        return response.points

    @staticmethod
    def _build_context(points):
        context_blocks, referenced_files = [], set()
        for point in points:
            path = point.payload["metadata"].get("file_path", "Unknown File")
            text = point.payload.get("text", "")
            referenced_files.add(path)
            context_blocks.append(f"--- START FILE SEGMENT: {path} ---\n{text}\n--- END FILE SEGMENT ---")
        return "\n\n".join(context_blocks), referenced_files

    def _generate_answer(self, query: str, context_str: str):
        system_instructions = (
            "You are a Principal Software Engineering Assistant specializing in codebase reasoning.\n"
            "Analyze the absolute file paths, dependencies, and code patterns provided below.\n"
            "Answer the user's inquiry accurately and concisely, based ONLY on the structural code blocks "
            "supplied, citing source paths inline using bracket notation [file_path].\n"
            "If you do not see the implementation details or answers in the context, clearly explain what is missing.\n"
            "Always target clean output, formatting answers using Markdown formatting."
        )
        user_prompt = f"Codebase Context:\n{context_str}\n\nUser Question:\n{query}"
        completion = self.groq_client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        return completion.choices[0].message.content

    # --- Confidence Router: signal #1 (retrieval) ----------------------------
    def _score_retrieval(self, points):
        """
        Flags LOW when either the best match is weak in absolute terms (top_score below
        floor), or the ranking is flat (top chunk isn't meaningfully better than the
        weakest chunk pulled in) — usually a sign retrieval didn't find anything clearly
        relevant and just returned "closest of a bad bunch."
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
        """Second, cheap Groq call: audits whether the retrieved context actually supports the answer."""
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

    @staticmethod
    def _confidence_report(attempt):
        return {
            "overall": attempt["overall"],
            "retrieval": {"level": attempt["retrieval_confidence"], **attempt["retrieval_meta"]},
            "llm_self_assessment": {"level": attempt["llm_confidence"], "reason": attempt.get("llm_reason")},
        }

    # --- First pass: plain retrieval + generation + confidence scoring -------
    def _first_pass(self, query: str):
        points = self.search_codebase(query, limit=6)

        if not points:
            return {
                "points": [], "context_str": "", "referenced_files": set(),
                "answer": None, "error": None,
                "retrieval_confidence": "LOW",
                "retrieval_meta": {"top_score": 0.0, "score_gap": 0.0, "reason": "No points retrieved."},
                "llm_confidence": "N/A", "llm_reason": "Skipped — no context to assess.",
                "overall": "LOW",
            }

        context_str, referenced_files = self._build_context(points)
        retrieval_confidence, retrieval_meta = self._score_retrieval(points)

        try:
            answer = self._generate_answer(query, context_str)
        except Exception as e:
            return {
                "points": points, "context_str": context_str, "referenced_files": referenced_files,
                "answer": None, "error": f"❌ Groq Completion Engine failure sequence triggered: {str(e)}",
                "retrieval_confidence": retrieval_confidence, "retrieval_meta": retrieval_meta,
                "llm_confidence": "N/A", "llm_reason": "Skipped — generation call failed.",
                "overall": "LOW",
            }
        finally:
            gc.collect()

        if ENABLE_LLM_CONFIDENCE_CHECK:
            llm_confidence, llm_reason = self._assess_answer_confidence(query, answer, context_str)
        else:
            llm_confidence, llm_reason = "N/A", "LLM self-check disabled (ENABLE_LLM_CONFIDENCE_CHECK=false)."

        overall = self._route_confidence(retrieval_confidence, llm_confidence)

        return {
            "points": points, "context_str": context_str, "referenced_files": referenced_files,
            "answer": answer, "error": None,
            "retrieval_confidence": retrieval_confidence, "retrieval_meta": retrieval_meta,
            "llm_confidence": llm_confidence, "llm_reason": llm_reason,
            "overall": overall,
        }

    # --- Agent tools -----------------------------------------------------------
    def _execute_tool_call(self, name: str, arguments: dict, state: dict):
        """
        Executes one agent tool call and returns a JSON-serializable result. `state` is a
        mutable dict the agent loop uses to track the latest retrieval points/context/files
        across iterations, so the final confidence check scores whatever the agent actually
        ended up looking at.
        """
        if name == "search_codebase":
            query = (arguments.get("query") or "").strip()
            if not query:
                return {"error": "Missing 'query' argument."}

            points = self.search_codebase(query, limit=6)
            if not points:
                return {"query": query, "results": [], "note": "No matches found for this query either."}

            context_str, referenced_files = self._build_context(points)
            state["points"] = points
            state["context_str"] = context_str
            state["referenced_files"] = referenced_files

            return {
                "query": query,
                "results": [
                    {
                        "file_path": p.payload["metadata"].get("file_path", "Unknown File"),
                        "score": p.score,
                        "snippet": p.payload.get("text", "")[:600],
                    }
                    for p in points
                ],
            }

        if name == "get_git_history":
            file_path = (arguments.get("file_path") or "").strip()
            if not file_path:
                return {"error": "Missing 'file_path' argument."}
            try:
                limit = int(arguments.get("limit") or 5)
            except (TypeError, ValueError):
                limit = 5

            if not self.git_root:
                return {"file_path": file_path, "commits": [],
                        "note": "Git history unavailable — repo isn't a readable git checkout."}

            full_path = os.path.join(REPO_BASE_PATH, file_path)
            commits = git_utils.file_log(self.git_root, full_path, limit=limit)
            if not commits:
                return {"file_path": file_path, "commits": [],
                        "note": "No git history found (untracked file, path typo, or git unavailable)."}
            return {"file_path": file_path, "commits": commits}

        return {"error": f"Unknown tool '{name}'."}

    # --- The agent loop itself ---------------------------------------------
    def _run_agent_loop(self, query: str, attempt: dict, silent: bool):
        """
        Confidence-triggered ReAct loop: gives the model real tools (re-search with a
        reformulated query, pull live git history) and lets IT decide how to close the
        evidence gap, instead of escalating on the very first low-confidence pass.

        Returns (refined_attempt_dict, agent_trace), where agent_trace lists every tool
        call the agent made — surfaced later so a human reviewing an escalation can see
        what was already tried.
        """
        state = {
            "points": attempt["points"],
            "context_str": attempt["context_str"],
            "referenced_files": set(attempt["referenced_files"]),
        }
        agent_trace = []

        system_prompt = (
            "You are a Principal Software Engineering Assistant with access to two tools over a real "
            "codebase: `search_codebase` (re-run semantic search with a better query) and "
            "`get_git_history` (look up recent commits for a specific file). Your first answer attempt "
            "was judged LOW confidence — the retrieved context may not actually support it. You MUST "
            "call at least one tool to investigate before giving a final answer. Reformulate the search "
            "if the context seems off-target; pull git history if the question is about who/when/why "
            "something changed. When you have gathered enough evidence, respond with your final answer "
            "as plain text with no further tool calls, citing source paths inline using bracket notation "
            "[file_path]. If, after investigating, you still cannot find enough evidence, say so plainly "
            "rather than guessing."
        )
        diagnostic = (
            f"Original question: {query}\n\n"
            f"Initial retrieval confidence: {attempt['retrieval_confidence']} "
            f"({attempt['retrieval_meta'].get('reason')})\n"
            f"Initial LLM self-assessment: {attempt['llm_confidence']} ({attempt.get('llm_reason')})\n\n"
            f"Initial retrieved context:\n{attempt['context_str'] or '(none — search returned nothing)'}\n\n"
            f"Initial draft answer:\n{attempt.get('answer') or '(none — generation failed or was skipped)'}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diagnostic},
        ]

        final_answer = None

        for step in range(AGENT_MAX_STEPS):
            try:
                completion = self.groq_client.chat.completions.create(
                    model=GROQ_LLM_MODEL,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=2048,
                )
            except Exception as e:
                if not silent:
                    print(f"❌ Agent loop step {step + 1} failed: {str(e)}")
                break

            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                final_answer = msg.content
                break

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if not silent:
                    print(f"🛠️  Agent step {step + 1}: calling {tc.function.name}({args})")

                result = self._execute_tool_call(tc.function.name, args, state)
                agent_trace.append({"step": step + 1, "tool": tc.function.name, "arguments": args})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        if final_answer is None:
            # Budget exhausted without the model volunteering a final plain-text answer —
            # force a concluding response from whatever it has gathered so far.
            try:
                wrap_up = self.groq_client.chat.completions.create(
                    model=GROQ_LLM_MODEL,
                    messages=messages + [{
                        "role": "user",
                        "content": "Give your best final answer now, in plain text, based on everything gathered so far."
                    }],
                    temperature=0.1,
                    max_tokens=2048,
                )
                final_answer = wrap_up.choices[0].message.content
            except Exception as e:
                final_answer = attempt.get("answer") or f"❌ Agent loop could not produce a final answer: {str(e)}"

        # Re-score confidence against whatever context the agent ended up using.
        final_points = state["points"]
        final_context = state["context_str"]
        final_files = state["referenced_files"]

        if final_points:
            retrieval_confidence, retrieval_meta = self._score_retrieval(final_points)
        else:
            retrieval_confidence, retrieval_meta = "LOW", {"top_score": 0.0, "score_gap": 0.0, "reason": "No points retrieved."}

        if ENABLE_LLM_CONFIDENCE_CHECK and final_answer:
            llm_confidence, llm_reason = self._assess_answer_confidence(query, final_answer, final_context)
        else:
            llm_confidence, llm_reason = "N/A", "Skipped."

        overall = self._route_confidence(retrieval_confidence, llm_confidence)

        refined = {
            "points": final_points, "context_str": final_context, "referenced_files": final_files,
            "answer": final_answer, "error": None,
            "retrieval_confidence": retrieval_confidence, "retrieval_meta": retrieval_meta,
            "llm_confidence": llm_confidence, "llm_reason": llm_reason,
            "overall": overall,
        }
        return refined, agent_trace

    def ask(self, query: str, silent: bool = False):
        """
        Retrieves context, generates an answer, and runs the Confidence Router. If the
        first pass comes back LOW confidence, hands off to a bounded ReAct-style agent
        loop (re-search / pull git history) before making the final answer-or-escalate
        call. Always returns a dict; see the "answered"/"escalated"/"error" branches
        below for the exact shape, including the "agent_trace" list of tool calls made
        (empty if the fast path never needed the agent loop).
        """
        if not silent:
            print(f"\n🔍 Searching vector space for: '{query}'...")

        attempt = self._first_pass(query)
        agent_trace = []

        if attempt["overall"] != "HIGH" and ENABLE_AGENT_LOOP:
            if not silent:
                print(f"🤖 First pass confidence LOW (retrieval={attempt['retrieval_confidence']}, "
                      f"llm={attempt['llm_confidence']}) — handing off to the agent loop "
                      f"(up to {AGENT_MAX_STEPS} tool-assisted steps)...")
            attempt, agent_trace = self._run_agent_loop(query, attempt, silent)

        citations = sorted(attempt["referenced_files"])
        retrieved_at_5 = self._top_retrieved_files(attempt.get("points", []), k=5)
        confidence_report = self._confidence_report(attempt)

        if attempt.get("error") and attempt["overall"] != "HIGH":
            if not silent:
                print(attempt["error"])
            return {
                "status": "error",
                "answer": attempt["error"],
                "citations": citations,
                "confidence": confidence_report,
                "contact": TECH_LEAD_CONTACT,
                "partial_context": citations,
                "draft_answer": attempt.get("answer"),
                "agent_trace": agent_trace,
                "retrieved_at_5": retrieved_at_5,
            }

        if attempt["overall"] == "HIGH":
            if not silent:
                print("\n🤖 Response:")
                print("======================================================================")
                print(attempt["answer"])
                suffix = " (resolved via agent loop)" if agent_trace else ""
                print(f"\n✅ Confidence: HIGH (retrieval={attempt['retrieval_confidence']}, "
                      f"llm_self_check={attempt['llm_confidence']}){suffix}")
                print("======================================================================\n")
            return {
                "status": "answered",
                "answer": attempt["answer"],
                "citations": citations,
                "confidence": confidence_report,
                "agent_trace": agent_trace,
                "retrieved_at_5": retrieved_at_5,
            }

        # Still LOW after the (optional) agent loop -> escalate.
        escalation_answer = (
            "⚠️ Confidence too low to trust this answer automatically — even after investigating "
            f"further — please verify with a Tech Lead ({TECH_LEAD_CONTACT}) before relying on it.\n\n"
            "Partial context retrieved (unverified):\n"
            + ("\n".join(f"- {p}" for p in citations) if citations else "(none)")
        )
        if not silent:
            print("\n🚨 LOW CONFIDENCE — escalating instead of asserting an answer")
            print("======================================================================")
            print(escalation_answer)
            if agent_trace:
                print("\nAgent trace:")
                for entry in agent_trace:
                    print(f"  step {entry['step']}: {entry['tool']}({entry['arguments']})")
            print(f"\nRetrieval confidence: {attempt['retrieval_confidence']} ({attempt['retrieval_meta'].get('reason')})")
            print(f"LLM self-assessment: {attempt['llm_confidence']} ({attempt.get('llm_reason')})")
            print("======================================================================\n")
        return {
            "status": "escalated",
            "answer": escalation_answer,
            "citations": citations,
            "confidence": confidence_report,
            "contact": TECH_LEAD_CONTACT,
            "partial_context": citations,
            "draft_answer": attempt.get("answer"),
            "agent_trace": agent_trace,
            "retrieved_at_5": retrieved_at_5,
        }

    @staticmethod
    def _top_retrieved_files(points, k: int = 5) -> list:
        """Return ranked, de-duplicated file paths for retrieval metrics such as precision/recall@k."""
        ranked_files = []
        seen = set()
        for point in points:
            path = point.payload.get("metadata", {}).get("file_path")
            if path and path not in seen:
                ranked_files.append(path)
                seen.add(path)
            if len(ranked_files) >= k:
                break
        return ranked_files

    @staticmethod
    def _expected_relevant_files(question: dict):
        """Read optional gold retrieval labels from the eval-set question."""
        for key in (
            "relevant_files",
            "expected_citations",
            "ground_truth_files",
            "reference_files",
            "gold_files",
            "citations",
        ):
            value = question.get(key)
            if isinstance(value, (list, tuple, set)) and value:
                return {str(path) for path in value if path}
        return None

    @staticmethod
    def _retrieval_metrics_at_5(retrieved_files: list, relevant_files):
        """Calculate per-question precision@5 and recall@5 when gold file labels exist."""
        if not relevant_files:
            return None

        retrieved = set(retrieved_files[:5])
        relevant = set(relevant_files)
        hits = len(retrieved & relevant)
        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(relevant) if relevant else 0.0
        return {"precision": precision, "recall": recall, "hits": hits}

    @staticmethod
    def _category_name(category: dict) -> str:
        return str(category.get("name") or category.get("type") or category.get("id") or "Uncategorized")

    @staticmethod
    def _format_percent(value):
        return "N/A" if value is None else f"{value * 100:.1f}%"

    def evaluate_and_score(self, question: dict, answer: str, context_files: list, silent: bool = True):
        """Judge only whether the generated answer passes; do not return or print the raw answer."""
        judge_prompt = (
            "You are evaluating an internal codebase Q&A response. "
            "Mark HIGH only when the generated answer satisfactorily answers the question using the supplied context; "
            "otherwise mark LOW. Respond ONLY with JSON in this exact compact shape: "
            "{\"confidence\": \"HIGH\"|\"LOW\", \"reason\": \"one short sentence\"}.\n\n"
            f"Question: {question.get('question', '')}\n"
            f"Answer: {answer or '(no answer)'}\n"
            f"Retrieved files: {', '.join(context_files)}"
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=120,
                response_format={"type": "json_object"}
            )
            verdict = json.loads(completion.choices[0].message.content)
            confidence = str(verdict.get("confidence", "LOW")).strip().upper()
            if confidence not in ("HIGH", "LOW"):
                confidence = "LOW"
            return {
                "confidence": confidence,
                "reason": str(verdict.get("reason", "")).strip(),
                "error": None,
            }
        except Exception as e:
            if not silent:
                print(f"Judge completion failed: {str(e)}")
            return {
                "confidence": "LOW",
                "reason": "Judge evaluation failed.",
                "error": str(e),
            }

    def run_evaluation(self, eval_dataset: dict, output_file: str = "evaluation_results.md"):
        """Run the eval set and write an aggregate Markdown summary, not raw per-question JSON."""
        dataset = eval_dataset.get("evaluation_dataset", {})
        categories = dataset.get("categories", [])

        total = 0
        passed = 0
        judge_high = 0
        judge_low = 0
        router_high = 0
        router_low = 0
        escalated = 0
        judge_errors = 0
        agent_steps_total = 0
        category_stats = {}
        precision_values = []
        recall_values = []

        for category in categories:
            category_name = self._category_name(category)
            stats = category_stats.setdefault(category_name, {"total": 0, "passed": 0, "high": 0, "low": 0})

            for raw_question in category.get("questions", []):
                # Do not mutate the source eval dataset while removing the gold answer from the judge input.
                question = dict(raw_question)
                question.pop("answer", None)

                total += 1
                stats["total"] += 1

                result = self.ask(question.get("question", ""), silent=True)
                judge = self.evaluate_and_score(
                    question,
                    result.get("answer"),
                    result.get("citations", []),
                    silent=True,
                )

                confidence = judge["confidence"]
                if confidence == "HIGH":
                    passed += 1
                    judge_high += 1
                    stats["passed"] += 1
                    stats["high"] += 1
                else:
                    judge_low += 1
                    stats["low"] += 1

                router_confidence = result.get("confidence", {}).get("overall", "LOW")
                if router_confidence == "HIGH":
                    router_high += 1
                else:
                    router_low += 1
                if result.get("status") == "escalated":
                    escalated += 1
                if judge.get("error"):
                    judge_errors += 1
                agent_steps_total += len(result.get("agent_trace", []))

                relevant_files = self._expected_relevant_files(question)
                retrieval_metrics = self._retrieval_metrics_at_5(
                    result.get("retrieved_at_5", []),
                    relevant_files,
                )
                if retrieval_metrics is not None:
                    precision_values.append(retrieval_metrics["precision"])
                    recall_values.append(retrieval_metrics["recall"])

        precision_at_5 = sum(precision_values) / len(precision_values) if precision_values else None
        recall_at_5 = sum(recall_values) / len(recall_values) if recall_values else None
        pass_rate = passed / total if total else 0.0

        lines = [
            "# Evaluation Results",
            "",
            f"**Overall pass rate:** {self._format_percent(pass_rate)} ({passed}/{total})",
            "",
            "## Pass rate by category",
            "",
            "| Category | Pass rate | Passed | Total |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, stats in category_stats.items():
            rate = stats["passed"] / stats["total"] if stats["total"] else 0.0
            lines.append(f"| {name} | {self._format_percent(rate)} | {stats['passed']} | {stats['total']} |")

        lines.extend([
            "",
            "## Confidence",
            "",
            "| Source | HIGH | LOW |",
            "| --- | ---: | ---: |",
            f"| Judge | {judge_high} | {judge_low} |",
            f"| Router | {router_high} | {router_low} |",
            "",
            "## Retrieval quality",
            "",
            f"- **Precision@5:** {self._format_percent(precision_at_5)}",
            f"- **Recall@5:** {self._format_percent(recall_at_5)}",
        ])

        if precision_at_5 is None or recall_at_5 is None:
            lines.append("- Add `relevant_files` (or `expected_citations`) to each eval question to enable gold retrieval precision/recall@5.")

        lines.extend([
            "",
            "## Evaluation health",
            "",
            f"- Router escalations: {escalated}",
            f"- Judge failures: {judge_errors}",
            f"- Agent-loop steps: {agent_steps_total}",
            "",
            "> Pass = judge confidence HIGH. Only aggregate metrics are written here; per-question answers, citations, and judge JSON are intentionally omitted.",
            "",
        ])
        md_content = "\n".join(lines)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_content = md_content.replace(
            "# Evaluation Results\n\n",
            f"# Evaluation Results\n\n_Generated: {timestamp}_\n\n",
            1,
        )
        result_path = output_file if output_file.endswith(".md") else f"{output_file}.md"
        print(f"Writing evaluation summary to: {result_path}")

        with open(result_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "output_file": result_path,
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "judge_high": judge_high,
            "judge_low": judge_low,
            "router_high": router_high,
            "router_low": router_low,
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5,
            "judge_errors": judge_errors,
        }


if __name__ == "__main__":
    engine = CodebaseQAEngine()

    while True:
        sample_query = input("\n📊 Enter your BA query (or type 'exit'): ")
        if sample_query.lower() == 'exit':
            print("Thank you for using the bot see you again!!")
            break
        engine.ask(sample_query)
