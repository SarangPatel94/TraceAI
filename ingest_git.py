import os
import uuid
import gc
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sentence_transformers import SentenceTransformer

# Load environment entries from local storage parameters
load_dotenv()

# Extract and validate environment configurations
REPO_BASE_PATH = os.getenv("REPO_BASE_PATH", "../TestRepo")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

# Core runtime constants
SUPPORTED_EXTENSIONS = ('.php', '.twig', '.scss', '.js', '.py', '.ts', '.json', '.md', '.xml')
LANG_MAP = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.ts': Language.TS,
    '.php': Language.PHP,
    '.twig': Language.HTML
}

class BatchCodebasePipeline:
    def __init__(self):
        print(f"⏳ Loading local dense embedding engine [{DENSE_MODEL_NAME}]...")
        self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        # Network timeout=60 safeguards against first-run BM25 setup latencies
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        
    def init_qdrant_collection(self):
        """Prepares database schema layout inside local Qdrant container."""
        if self.qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            self.qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )
        print(f"✨ Clean hybrid storage table '{COLLECTION_NAME}' created successfully!")

    def run_ingestion(self):
        """Parses repository target trees and flushes points in memory-safe batches."""
        self.init_qdrant_collection()
        search_path = os.path.abspath(REPO_BASE_PATH)
        
        current_batch = []
        total_indexed_points = 0
        total_processed_files = 0

        TARGET_PATHS = [
            os.path.normpath("vendor/spryker"),
            os.path.normpath("src/Pyz")
        ]

        print(f"🕵️ Target-scanning directories under path: {search_path}")
        print(f"🎯 Allowed scopes: {', '.join(TARGET_PATHS)}")
        print(f"⚡ Memory Guard Active: Batching execution at {BATCH_SIZE} points per flush.\n")

        for root, dirs, files in os.walk(search_path):
            rel_root_path = os.path.normpath(os.path.relpath(root, search_path))
            
            # --- Dynamic Tree Pruning Rule ---
            valid_dirs = []
            for d in dirs:
                potential_rel_path = os.path.normpath(os.path.join(rel_root_path, d)) if rel_root_path != "." else d
                is_valid = any(
                    potential_rel_path.startswith(target) or target.startswith(potential_rel_path)
                    for target in TARGET_PATHS
                )
                if is_valid:
                    valid_dirs.append(d)
            dirs[:] = valid_dirs

            # --- File Extraction Scope Matcher ---
            in_allowed_scope = any(
                rel_root_path == target or rel_root_path.startswith(target + os.sep)
                for target in TARGET_PATHS
            )
            
            if not in_allowed_scope:
                continue

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, REPO_BASE_PATH)
                    
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read().strip()
                        if not content:
                            continue
                            
                        total_processed_files += 1
                        
                        lang = LANG_MAP.get(ext)
                        splitter = RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=1200, chunk_overlap=200) if lang else RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                        
                        chunks = splitter.split_text(content)
                        
                        for chunk in chunks:
                            dense_vector = self.embedding_model.encode(
                                "Represent this sentence for searching relevant code snippets: " + chunk
                            ).tolist()

                            current_batch.append(PointStruct(
                                id=str(uuid.uuid4()),
                                payload={
                                    "text": chunk,
                                    "metadata": {
                                        "file_path": rel_path,
                                        "file_extension": ext
                                    }
                                },
                                vector={
                                    "text-dense": dense_vector,
                                    "text-sparse": Document(text=chunk, model="Qdrant/bm25")
                                }
                            ))

                            if len(current_batch) >= BATCH_SIZE:
                                self.qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=current_batch)
                                total_indexed_points += len(current_batch)
                                print(f"💾 [Batch Flush] Upserted {len(current_batch)} points | Current file: {rel_path}")
                                current_batch.clear()
                                gc.collect()

                    except Exception as e:
                        print(f"❌ Failed to parse file {rel_path}: {str(e)}")

        if current_batch:
            self.qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=current_batch)
            total_indexed_points += len(current_batch)
            print(f"💾 [Final Flush] Upserted remaining {len(current_batch)} trailing points.")
            current_batch.clear()
            gc.collect()

        print(f"\n✅ Ingestion complete! Scanned {total_processed_files} files and safely indexed {total_indexed_points} total codebase vectors.")

if __name__ == "__main__":
    pipeline = BatchCodebasePipeline()
    pipeline.run_ingestion()