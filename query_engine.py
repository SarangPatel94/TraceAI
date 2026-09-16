import os
import json
import re
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
# 🚀 IMPORT LOCAL FASTEMBED FOR CLIENT-SIDE SPARSE VECTOR GENERATION
from fastembed import SparseTextEmbedding

from sentence_transformers import SentenceTransformer

# Initialize environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

class LocalCodebaseQueryEngine:
    def __init__(self):
        print(f"⏳ Loading local dense embedding engine [{DENSE_MODEL_NAME}]...")
        self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        
        print("⏳ Initializing local client-side BM25 tokenizer...")
        # 🚀 Forces tokenizing to execute inside Python instead of the database container
        self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        
        # Extended network request buffer safety threshold
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=90)

    def hybrid_search(self, query_text: str, top_k: int = 4) -> list:
        """Runs accelerated reciprocal rank fusion search using client-side generated vectors."""
        # 1. Compute Dense Vector (1024 dimensions)
        query_dense = self.embedding_model.encode(
            "Represent this sentence for searching relevant code snippets: " + query_text
        ).tolist()
        
        # 2. Compute Sparse Vector locally using unified FastEmbed engine
        # query_embed yields a generator of sparse structures; take the first index item
        sparse_embeddings_raw = list(self.sparse_embedding_model.query_embed(query_text))[0]
        
        query_sparse = models.SparseVector(
            indices=sparse_embeddings_raw.indices.tolist(),
            values=sparse_embeddings_raw.values.tolist()
        )
        
        # 3. Query Qdrant with pre-computed mathematical matrices
        search_results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="text-dense", limit=top_k * 2),
                models.Prefetch(query=query_sparse, using="text-sparse", limit=top_k * 2),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )
        return [p.payload for p in search_results.points]

    def _extract_and_parse_json(self, raw_text: str) -> dict:
        """Cleans conversations or markdown boundaries to return structural JSON records."""
        clean_text = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            return {
                "confidence": "LOW",
                "answer": "Error parsing structural layout boundaries from local model text stream.",
                "citations": [],
                "tech_lead_escalation": f"Raw local text dump requires inspection: {raw_text}"
            }

    def ask(self, ba_query: str) -> dict:
        """Performs RAG search extraction queries and renders answers via Ollama endpoints."""
        contexts = self.hybrid_search(ba_query)
        if not contexts:
            return {
                "confidence": "LOW", 
                "answer": "No relevant codebase contexts found.", 
                "citations": [], 
                "tech_lead_escalation": "Verify folder paths inside the local .env settings configuration."
            }

        formatted_context = ""
        for idx, ctx in enumerate(contexts):
            formatted_context += (
                f"--- CONTEXT BLOCK {idx+1} ---\n"
                f"File: {ctx['metadata']['file_path']}\n"
                f"Code Snippet:\n{ctx['text']}\n\n"
            )

        system_prompt = (
            "You are an expert technical AI assistant. You answer system functionality questions from Business Analysts "
            "using provided repository context snippets.\n\n"
            "CRITICAL RULES:\n"
            "1. Evaluate your confidence honestly based ONLY on the context blocks provided. If the context does not explicitly "
            "contain the logic or answers required to resolve the query, you MUST set 'confidence' to 'LOW'.\n"
            "2. Provide definitive answers ONLY when confidence is 'HIGH'.\n"
            "3. Every assertion made in a 'HIGH' confidence answer must point to specific files inside your response text.\n"
            "4. You MUST output your final response STRICTLY inside a valid JSON object format."
        )

        user_prompt = f"""
        Context Data from Codebase:
        {formatted_context}
        
        Business Analyst Query: {ba_query}
        
        Respond strictly in this exact JSON format layout structure:
        {{
          "confidence": "HIGH" | "LOW",
          "answer": "Clear, plain-language business analysis description explicitly answering the query.",
          "citations": [
            {{ "file_path": "path/to/file" }}
          ],
          "tech_lead_escalation": "If confidence is LOW, provide a concise question framework pointing out what is ambiguous for the Tech Lead to verify. If confidence is HIGH, leave this field empty."
        }}
        """

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {"temperature": 0.1},
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            raw_content = response.json()["message"]["content"]
            return self._extract_and_parse_json(raw_content)
        except Exception as e:
            return {
                "confidence": "LOW",
                "answer": "Network connection error communicating with local Ollama service host.",
                "citations": [],
                "tech_lead_escalation": f"Error details: {str(e)}"
            }

if __name__ == "__main__":
    engine = LocalCodebaseQueryEngine()
    while True:
        query = input("\n📊 Enter your BA query (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        output = engine.ask(query)
        print("\n🤖 AI Response:")
        print(json.dumps(output, indent=2))