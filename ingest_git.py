import os
import re
import uuid
import gc
import time
import glob
import subprocess
from collections import Counter
from datetime import datetime, timezone
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
    '.twig': Language.HTML
}

# git blame --line-porcelain prefixes each line-group with "<40-hex-sha> <orig> <final> [<count>]"
_BLAME_SHA_RE = re.compile(r'^([0-9a-f]{40})\s+\d+\s+\d+')


class BatchCodebasePipeline:
    def __init__(self):
        print(f"⏳ Loading local dense embedding engine [{DENSE_MODEL_NAME}]...")
        self.embedding_model = SentenceTransformer(DENSE_MODEL_NAME)
        # Network timeout=60 safeguards against first-run BM25 setup latencies
        self.qdrant_client = QdrantClient(QDRANT_URL, timeout=60)
        self.git_root = None  # resolved per-run in run_ingestion(), once we know the target path

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

    # --- Git metadata: helpers ------------------------------------------------
    def _run_git(self, args, cwd, timeout=30):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    def _detect_git_root(self, path):
        out = self._run_git(["rev-parse", "--show-toplevel"], cwd=path)
        return out.strip() if out else None

    def _git_file_log(self, full_path):
        """File-level fallback: hash/author/date of the last commit that touched this file."""
        out = self._run_git(
            ["log", "-1", "--format=%H\x1f%an\x1f%ad", "--date=short", "--", full_path],
            cwd=self.git_root,
        )
        if not out or not out.strip():
            return {"commit_hash": None, "author": None, "date": None}
        parts = out.strip().split("\x1f")
        if len(parts) != 3:
            return {"commit_hash": None, "author": None, "date": None}
        commit_hash, author, date = parts
        return {"commit_hash": commit_hash, "author": author, "date": date}

    def _git_blame_lines(self, full_path):
        """
        Runs `git blame --line-porcelain` ONCE per file (not per chunk, which would be far
        too slow) and returns a list of (commit_hash, author, date) tuples, one per source
        line (0-indexed). Returns None if blame isn't available (untracked file, binary,
        not a git repo, etc.) so callers can fall back to file-level git log metadata.
        """
        out = self._run_git(["blame", "--line-porcelain", "--", full_path], cwd=self.git_root)
        if out is None:
            return None

        lines_meta = []
        current_hash = None
        current_author = None
        current_date = None

        for raw_line in out.splitlines():
            sha_match = _BLAME_SHA_RE.match(raw_line)
            if sha_match:
                current_hash = sha_match.group(1)
                continue
            if raw_line.startswith("author "):
                current_author = raw_line[len("author "):]
                continue
            if raw_line.startswith("author-time "):
                try:
                    ts = int(raw_line.split(" ", 1)[1])
                    current_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, IndexError):
                    current_date = None
                continue
            if raw_line.startswith("\t"):
                # One "\t<content>" line closes out each line-group's metadata block.
                lines_meta.append((current_hash, current_author, current_date))

        return lines_meta

    def _chunk_git_metadata(self, blame_lines, start_line, end_line, fallback):
        """
        Majority-vote the commit across a chunk's line range using the file's precomputed
        blame data, so a chunk is attributed to whichever commit actually wrote most of it
        rather than just the first or last line touched. Falls back to file-level git log
        metadata when blame is unavailable or the range has nothing usable.
        """
        if not blame_lines:
            return fallback

        start = max(0, start_line)
        end = min(len(blame_lines), end_line + 1)
        segment = [entry for entry in blame_lines[start:end] if entry[0]]

        if not segment:
            return fallback

        dominant_hash, _ = Counter(entry[0] for entry in segment).most_common(1)[0]
        for commit_hash, author, date in segment:
            if commit_hash == dominant_hash:
                return {"commit_hash": commit_hash, "author": author, "date": date}

        return fallback

    def run_ingestion(self):
        """Parses repository target trees and flushes points in memory-safe batches."""
        self.init_qdrant_collection()
        search_path = os.path.abspath(REPO_BASE_PATH)

        if ENABLE_GIT_METADATA:
            self.git_root = self._detect_git_root(search_path)
            if self.git_root:
                print(f"🔗 Git metadata enabled — resolved repo root: {self.git_root}")
            else:
                print(f"⚠️ Git metadata requested but '{search_path}' isn't inside a readable git "
                      f"repo (or git isn't installed) — chunks will be indexed with author/commit_hash/date = None.")
        else:
            print("🔕 Git metadata disabled (ENABLE_GIT_METADATA=false).")

        current_batch = []
        total_indexed_points = 0
        total_processed_files = 0

        # 1. Join search_path with the pattern to locate the vendor directory correctly
        glob_pattern = os.path.join(REPO_BASE_PATH, "vendor", "spryker*")

        # 2. Find paths and convert them back to relative paths matching your search_path context
        spryker_paths = [
            os.path.normpath(os.path.relpath(p, search_path))
            for p in glob.glob(glob_pattern)
        ]

        TARGET_PATHS = spryker_paths + [
            os.path.normpath("src/Pyz"),
            os.path.normpath("config")
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
                            raw_content = f.read()
                        content = raw_content.strip()
                        if not content:
                            continue

                        total_processed_files += 1

                        # --- Git metadata: one git log + one git blame call per file ---
                        if self.git_root:
                            file_fallback = self._git_file_log(full_path)
                            blame_lines = self._git_blame_lines(full_path)
                        else:
                            file_fallback = {"commit_hash": None, "author": None, "date": None}
                            blame_lines = None

                        # Map offsets in the stripped `content` back to real line numbers in
                        # `raw_content`, since leading blank lines would otherwise throw off
                        # the blame line lookup below.
                        content_offset_in_raw = raw_content.find(content)
                        if content_offset_in_raw < 0:
                            content_offset_in_raw = 0
                        base_line_offset = raw_content.count("\n", 0, content_offset_in_raw)

                        lang = LANG_MAP.get(ext)
                        splitter = RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=1200, chunk_overlap=200) if lang else RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

                        chunks = splitter.split_text(content)

                        search_cursor = 0
                        for chunk in chunks:
                            # Locate this chunk's line range so we can attribute it to the
                            # commit that actually wrote those lines (git blame), instead of
                            # just stamping every chunk with the file's last-touched commit.
                            chunk_offset = content.find(chunk, search_cursor)
                            if chunk_offset == -1:
                                chunk_offset = search_cursor
                            search_cursor = chunk_offset

                            start_line = base_line_offset + content.count("\n", 0, chunk_offset)
                            end_line = base_line_offset + content.count("\n", 0, chunk_offset + len(chunk))

                            git_meta = self._chunk_git_metadata(blame_lines, start_line, end_line, file_fallback)

                            dense_vector = self.embedding_model.encode(
                                "Represent this sentence for searching relevant code snippets: " + chunk
                            ).tolist()

                            current_batch.append(PointStruct(
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
    start_time = time.perf_counter()
    pipeline.run_ingestion()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.4f} seconds")