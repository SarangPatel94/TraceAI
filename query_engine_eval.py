import os
import json
import gc
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime

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
            "You are a Principal Software Engineering Assistant make sure to answer the query in less tokens. Analyze the code blocks "
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

    def evaluate_and_score(self, question: dict, answer: str, context_files: list):
        judge_prompt = (
            "Evaluate response if the answer block does not successfully satisfy the query please mark the confidence as 'LOW' otherwise 'HIGH' question is an json object please also return the evaluation in given format\n"
            f"Question: {question}\nA: {answer}\nFiles: {', '.join(context_files)}\n"
            "Return JSON: {\"id\": <question_id>, \"type\": <question_type>, \"question\": <question>, \"answer\": <answer> \"citations\": [<list of files>], \"confidence\": <confidence>}"
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
            return {"id": question.get("id"), "type": question.get("type"), "question": question.get("question"), "citations": [], "confidence": "None"}

    def run_evaluation(self, eval_dataset: dict, output_file: str = "evaluation_results.md"):
        dataset = eval_dataset.get("evaluation_dataset", {})
        md_content = f"# Evaluation Results\n\n"

        for category in dataset.get("categories", []):
            for q_obj in category.get("questions", []):
                print(f"File Data for eval : {q_obj}")
                q_obj.pop("answer", None)
                ans, files = self.ask(q_obj.get("question"), silent=True)
                responseData = self.evaluate_and_score(q_obj, ans, files)
                md_content += str (responseData) + "\n"
                print(f"The content for the MD file {md_content}")
                time.sleep(75)
                # Check if the target evaluation results file already exists
        if os.path.exists(output_file):
            # Split the filename into name and extension (e.g., 'evaluation_results' and '.md')
            base_name, extension = os.path.splitext(output_file)
            
            # Generate a unique timestamp pattern (YYYYMMDD_HHMMSS)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Construct the new non-conflicting filename
            output_file = f"{base_name}_{timestamp}{extension}"
            print(f"🔄 File already exists! Renaming active evaluation summary output to: {output_file}")
        else:
            print(f"📝 Writing brand new evaluation results log to: {output_file}")

        # Safely flush the complete generated markdown summary dataset to storage
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

if __name__ == "__main__":
    engine = CodebaseQAEngine()
    if os.path.exists("eval_set.json"):
        with open("eval_set.json", "r", encoding="utf-8") as f:
            engine.run_evaluation(json.load(f))
    else:
        print(f"No file is present called eval_set.json Please add the same and run the eval engine")