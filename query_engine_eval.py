import os
import json
import gc
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = "qwen/qwen3.8-27b" 

class CodebaseQAEngine:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ Missing 'GROQ_API_KEY' inside environment variables.")
        self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    def search_codebase(self, query: str, limit: int = 5):
        instructional_query = "Represent this sentence for searching relevant code snippets: " + query
        dense_vector = self.embedding_model.encode(instructional_query).tolist()
        response = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vector, using="text-dense", limit=limit * 2),
                models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"), using="text-sparse", limit=limit * 2)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        return response.points

    def ask(self, query: str, silent: bool = False):
        points = self.search_codebase(query, limit=6)
        if not points:
            return "⚠️ No matching code blocks surfaced.", []
        context_blocks, referenced_files = [], set()
        for point in points:
            path = point.payload["metadata"].get("file_path", "Unknown File")
            text = point.payload.get("text", "")
            referenced_files.add(path)
            context_blocks.append(f"--- START FILE SEGMENT: {path} ---\n{text}\n--- END FILE SEGMENT ---")
        context_str = "\n\n".join(context_blocks)
        system_instructions = (
            "You are a Principal Software Engineering Assistant. Analyze the code blocks "
            "and cite source paths inline using bracket notation [file_path]."
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"Codebase Context:\n{context_str}\n\nUser Question:\n{query}"}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            return completion.choices[0].message.content, list(referenced_files)
        except Exception as e:
            return f"❌ Failure: {str(e)}", []
        finally:
            gc.collect()

    def evaluate_and_score(self, question: str, answer: str, context_files: list):
        judge_prompt = (
            "Evaluate response on 1-5 scale for Grounding, Coverage, and Citation Accuracy.\n"
            f"Q: {question}\nA: {answer}\nFiles: {', '.join(context_files)}\n"
            "Return JSON: {\"grounding_score\": 5, \"coverage_score\": 4, \"citation_score\": 5, \"justification\": \"string\"}"
        )
        try:
            completion = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices.message.content)
        except Exception as e:
            return {"grounding_score": 0, "coverage_score": 0, "citation_score": 0, "justification": f"Error: {str(e)}"}

    def run_evaluation(self, eval_dataset: dict, output_file: str = "evaluation_results.md"):
        dataset = eval_dataset.get("evaluation_dataset", {})
        md_content = f"# Evaluation Results\n\n"
        total_q, sum_g, sum_c, sum_cit = 0, 0, 0, 0

        for category in dataset.get("categories", []):
            for q_obj in category.get("questions", []):
                print(f"File Data for eval : {q_obj}")
                total_q += 1
                ans, files = self.ask(q_obj.get("question"), silent=True)
                scores = self.evaluate_and_score(q_obj.get("question"), ans, files)
                sum_g += scores.get("grounding_score", 0)
                sum_c += scores.get("coverage_score", 0)
                sum_cit += scores.get("citation_score", 0)
                md_content += f"### Q: {q_obj.get('question')}\n**Answer:** {ans}\n\n"
                print(f"The content for the MD file {md_content}")
                time.sleep(40)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

if __name__ == "__main__":
    engine = CodebaseQAEngine()
    if os.path.exists("eval_set.json"):
        with open("eval_set.json", "r", encoding="utf-8") as f:
            engine.run_evaluation(json.load(f))
    else:
        print(f"No file is present called eval_set.json Please add the same and run the eval engine")