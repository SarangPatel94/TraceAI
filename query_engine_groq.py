import os
import gc
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
                # Correct Sparse structural text matching configuration
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25"
                    ),
                    using="text-sparse",
                    limit=limit * 2
                )
            ],
            # Use Relative Score Fusion (RRF) to merge structural text vs dense semantic scores
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        return response.points

    def ask(self, query: str):
        """Retrieves codebase context chunks and generates a precise answer using Groq."""
        print(f"\n🔍 Searching vector space for: '{query}'...")
        points = self.search_codebase(query, limit=6)
        
        if not points:
            print("⚠️ No matching code blocks surfaced from the database vector indexes.")
            return

        # Build context metadata payload string
        context_blocks = []
        for point in points:
            path = point.payload["metadata"].get("file_path", "Unknown File")
            text = point.payload.get("text", "")
            context_blocks.append(f"--- START FILE SEGMENT: {path} ---\n{text}\n--- END FILE SEGMENT ---")
            
        context_str = "\n\n".join(context_blocks)

        # Build execution payload instructions for Groq
        system_instructions = (
            "You are a Principal Software Engineering Assistant specializing in codebase reasoning.\n"
            "Analyze the absolute file paths, dependencies, and code patterns provided below.\n"
            "Answer the user's inquiry accurately based ONLY on the structural code blocks supplied.\n"
            "If you do not see the implementation details or answers in the context, clearly explain what is missing.\n"
            "Always target clean output, formatting answers using Markdown formatting."
        )

        user_prompt = f"Codebase Context:\n{context_str}\n\nUser Question:\n{query}"

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
            
            print("\n🤖 Groq Engineering Response:")
            print("======================================================================")
            print(completion.choices[0].message.content)
            print("======================================================================\n")
            
        except Exception as e:
            print(f"❌ Groq Completion Engine failure sequence triggered: {str(e)}")
            
        finally:
            gc.collect()

if __name__ == "__main__":
    # Initialize implementation pipeline interface
    engine = CodebaseQAEngine()

    while True:
        # Target search sample 
        sample_query = input("\n📊 Enter your BA query (or type 'exit'): ")
        if (sample_query.lower() == 'exit'):
            print(f"Thank you for using the bot see you again!!")
            break
        engine.ask(sample_query)