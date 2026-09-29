import os
import uuid
import gc
import glob
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter, Language
from sentence_transformers import SentenceTransformer

import git_utils

# Load environment entries from local storage parameters
load_dotenv()

# Extract and validate environment configurations
REPO_BASE_PATH = os.getenv("REPO_BASE_PATH", "../TestRepo")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_repo_chunks")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

# Set to "false" to skip git entirely (e.g. running against a plain directory that
# isn't a git checkout, or to speed up ingestion when attribution isn't needed).
ENABLE_GIT_METADATA = os.getenv("ENABLE_GIT_METADATA", "true").strip().lower() in ("1", "true", "yes")

# Core runtime constants
SUPPORTED_EXTENSIONS = ('.php', '.twig', '.scss', '.js', '.py', '.ts', '.json', '.md', '.xml')
LANG_MAP = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.ts': Language.TS,
    '.php': Language.PHP,
    '.twig': Language.HTML,
    '.md': Language.MARKDOWN
}

_NO_GIT_META = {"commit_hash": None, "author": None, "date": None, "subject": None}


class BatchCodebasePipeline:
    def __init__(self, embedding_model: SentenceTransformer = None):
        if embedding_model is not None:
            # Reuse an already-loaded model (e.g. shared with CodebaseQAEngine by app.py)
            # instead of loading a second copy into memory.
            self.embedding_model = embedding_model
        else:
            print(f"⏳ Loading local dense embedding engine [{DENSE_MODEL_NAME}]...")
            self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        # Network timeout=60 safeguards against first-run BM25 setup latencies
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        self.git_root = None  # resolved per-run in run_ingestion(), once we know the target path

    def init_qdrant_collection(self):
        """Wipes and recreates the collection from scratch — used for a full repo rebuild."""
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

    def ensure_qdrant_collection(self):
        """Creates the collection only if missing — non-destructive, used for partial/incremental ingests."""
        if not self.qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            print(f"✨ Collection '{COLLECTION_NAME}' didn't exist yet — created it fresh.")

    def _relative_file_path(self, full_path: str) -> str:
        """Stores paths relative to REPO_BASE_PATH (matching existing citation format) when
        the file is actually inside it; falls back to the absolute path otherwise, since an
        /ingest call can point at a file or folder outside the configured repo root."""
        repo_abs = os.path.abspath(REPO_BASE_PATH)
        if full_path == repo_abs or full_path.startswith(repo_abs + os.sep):
            return os.path.relpath(full_path, REPO_BASE_PATH)
        return full_path

    def _process_file(self, full_path: str, rel_path: str, ext: str):
        """Chunks one file, attaches git metadata + dense/sparse vectors. Returns list[PointStruct]
        (empty if the file is blank). Shared by both the directory walk and single-file ingestion."""
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
        content = raw_content.strip()
        if not content:
            return []

        # --- Git metadata: one git log + one git blame call per file ---
        if self.git_root:
            log_entries = git_utils.file_log(self.git_root, full_path, limit=1)
            file_fallback = log_entries[0] if log_entries else _NO_GIT_META
            blame_data = git_utils.blame_lines(self.git_root, full_path)
        else:
            file_fallback = _NO_GIT_META
            blame_data = None

        # Map offsets in the stripped `content` back to real line numbers in `raw_content`,
        # since leading blank lines would otherwise throw off the blame line lookup below.
        content_offset_in_raw = raw_content.find(content)
        if content_offset_in_raw < 0:
            content_offset_in_raw = 0
        base_line_offset = raw_content.count("\n", 0, content_offset_in_raw)

        if ext == ".json":
            json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
            chunks = json_splitter.split_text(content)
        else:
            lang = LANG_MAP.get(ext)
            splitter = RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=1200, chunk_overlap=200) if lang else RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = splitter.split_text(content)

        points = []
        search_cursor = 0
        for chunk in chunks:
            chunk_offset = content.find(chunk, search_cursor)
            if chunk_offset == -1:
                chunk_offset = search_cursor
            search_cursor = chunk_offset

            start_line = base_line_offset + content.count("\n", 0, chunk_offset)
            end_line = base_line_offset + content.count("\n", 0, chunk_offset + len(chunk))
            git_meta = git_utils.dominant_commit_for_range(blame_data, start_line, end_line, file_fallback)

            dense_vector = self.embedding_model.encode(
                "Represent this sentence for searching relevant code snippets: " + chunk
            ).tolist()

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                payload={
                    "text": chunk,
                    "metadata": {
                        "file_path": rel_path,
                        "file_extension": ext,
                        "author": git_meta.get("author"),
                        "commit_hash": git_meta.get("commit_hash"),
                        "date": git_meta.get("date"),
                    }
                },
                vector={
                    "text-dense": dense_vector,
                    "text-sparse": Document(text=chunk, model="Qdrant/bm25")
                }
            ))
        return points

    def run_ingestion(self, target_path: str = None, recreate_collection: bool = None):
        """
        Ingests a repo root, a folder, or a single file.

          - target_path=None -> the full default repo (REPO_BASE_PATH), using the original
            TARGET_PATHS scoping (vendor/spryker, src/Pyz, config). Recreates the collection
            (full rebuild) unless recreate_collection is explicitly False.
          - target_path=<folder> -> every supported file under it, no TARGET_PATHS scoping.
            Upserts into the existing collection (no wipe) unless recreate_collection is
            explicitly True.
          - target_path=<file> -> just that one file. Same upsert-only default as folders.

        Raises FileNotFoundError if target_path doesn't exist, ValueError if a single-file
        target has an unsupported extension. Returns a summary dict.
        """
        is_full_default_repo = target_path is None
        resolved_target = os.path.abspath(target_path) if target_path else os.path.abspath(REPO_BASE_PATH)

        if not os.path.exists(resolved_target):
            raise FileNotFoundError(f"Path not found: {resolved_target}")

        if recreate_collection is None:
            recreate_collection = is_full_default_repo

        if recreate_collection:
            self.init_qdrant_collection()
        else:
            self.ensure_qdrant_collection()

        warnings = []
        if ENABLE_GIT_METADATA:
            probe_dir = resolved_target if os.path.isdir(resolved_target) else os.path.dirname(resolved_target)
            self.git_root = git_utils.detect_git_root(probe_dir)
            if self.git_root:
                print(f"🔗 Git metadata enabled — resolved repo root: {self.git_root}")
            else:
                msg = f"'{resolved_target}' isn't inside a readable git repo — indexing without author/commit_hash/date."
                print(f"⚠️ {msg}")
                warnings.append(msg)
        else:
            self.git_root = None

        current_batch = []
        total_indexed_points = 0
        total_processed_files = 0
        skipped_unsupported = 0

        print(f"🕵️ Ingesting target: {resolved_target}")
        print(f"⚡ Memory Guard Active: Batching execution at {BATCH_SIZE} points per flush.\n")

        if os.path.isfile(resolved_target):
            ext = os.path.splitext(resolved_target)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file extension '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

            rel_path = self._relative_file_path(resolved_target)
            try:
                points = self._process_file(resolved_target, rel_path, ext)
                total_processed_files += 1
                current_batch.extend(points)
            except Exception as e:
                print(f"❌ Failed to parse file {rel_path}: {str(e)}")

        else:
            TARGET_PATHS = [
                os.path.normpath("vendor"),
                os.path.normpath("src"),
                os.path.normpath("config"),
            ] if is_full_default_repo else None

            if TARGET_PATHS:
                print(f"🎯 Allowed scopes: {', '.join(TARGET_PATHS)}")

            for root, dirs, files in os.walk(resolved_target):
                if TARGET_PATHS is not None:
                    rel_root_path = os.path.normpath(
                        os.path.relpath(root, resolved_target)
                    )

                    # --- Dynamic Tree Pruning Rule ---
                    # Skip any directory whose name contains "test" (case-insensitive).
                    valid_dirs = []
                    for d in dirs:
                        if "test" in d.lower():
                            continue
                        potential_rel_path = (
                            os.path.normpath(os.path.join(rel_root_path, d))
                            if rel_root_path != "."
                            else d
                        )
                        is_valid = any(
                            potential_rel_path.startswith(target)
                            or target.startswith(potential_rel_path)
                            for target in TARGET_PATHS
                        )
                        if is_valid:
                            valid_dirs.append(d)
                    dirs[:] = valid_dirs

                    # --- File Extraction Scope Matcher ---
                    in_allowed_scope = any(
                        rel_root_path == target
                        or rel_root_path.startswith(target + os.sep)
                        for target in TARGET_PATHS
                    )
                    if not in_allowed_scope:
                        continue

                # --- Skip test files ---
                # Applies regardless of extension and is case-insensitive.
                filtered_files = [
                    filename
                    for filename in files
                    if "test" not in filename.lower()
                ]

                for file in filtered_files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        skipped_unsupported += 1
                        continue

                    full_path = os.path.join(root, file)
                    rel_path = self._relative_file_path(full_path)

                    try:
                        points = self._process_file(full_path, rel_path, ext)
                        total_processed_files += 1
                        current_batch.extend(points)

                        while len(current_batch) >= BATCH_SIZE:
                            flush_batch, current_batch = current_batch[:BATCH_SIZE], current_batch[BATCH_SIZE:]
                            self.qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=flush_batch)
                            total_indexed_points += len(flush_batch)
                            print(f"💾 [Batch Flush] Upserted {len(flush_batch)} points | Current file: {rel_path}")
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

        return {
            "target_path": resolved_target,
            "files_processed": total_processed_files,
            "chunks_indexed": total_indexed_points,
            "files_skipped_unsupported": skipped_unsupported,
            "collection_recreated": recreate_collection,
            "warnings": warnings,
        }


if __name__ == "__main__":
    pipeline = BatchCodebasePipeline()
    start_time = time.perf_counter()
    pipeline.run_ingestion()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.4f} seconds")
