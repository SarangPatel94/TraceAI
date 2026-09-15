import os
import subprocess
from datetime import datetime
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load the high-performance embedding model locally (automatically uses GPU if available)
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def generate_local_embedding(text):
    """Generates localized BGE embeddings with search instructions."""
    instruction = "Represent this sentence for searching relevant code snippets: "
    return embedding_model.encode(instruction + text).tolist()

def get_complete_file_context(repo_path, file_path):
    """Executes porcelain git blame to pair every line of code with its metadata."""
    try:
        cmd = ["git", "blame", "-p", file_path]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None, None
    
    lines = result.stdout.split('\n')
    commit_pool = {}
    complete_codebase_lines = []
    raw_file_content_builder = []
    current_commit = None
    
    for line in lines:
        if not line: 
            continue
        parts = line.split()
        if not parts:
            continue
            
        # Catch the commit header line (40-char SHA)
        if len(parts[0]) == 40:
            current_commit = parts[0]
            if current_commit not in commit_pool:
                commit_pool[current_commit] = {
                    "commit_hash": current_commit,
                    "author": "Unknown",
                    "date": "Unknown",
                    "summary": "No Summary"
                }
        elif line.startswith("author "):
            # Safe Guard: If metadata occurs before a 40-char SHA line
            if current_commit is None:
                continue
            commit_pool[current_commit]["author"] = line[7:].strip()
        elif line.startswith("author-time "):
            if current_commit is None:
                continue
            timestamp = int(line[12:].strip())
            commit_pool[current_commit]["date"] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        elif line.startswith("summary "):
            if current_commit is None:
                continue
            commit_pool[current_commit]["summary"] = line[8:].strip()
        elif line.startswith("\t"):
            # Safe Guard: Ensure we have a commit tracking object initialized
            if current_commit is None:
                fallback_sha = "unknown_commit"
                if fallback_sha not in commit_pool:
                    commit_pool[fallback_sha] = {"commit_hash": fallback_sha, "author": "Unknown", "date": "Unknown", "summary": "No Summary"}
                current_commit = fallback_sha
                
            code_content = line[1:] 
            raw_file_content_builder.append(code_content)
            
            line_context = {
                "line_content": code_content,
                "commit_hash": commit_pool[current_commit]["commit_hash"],
                "author": commit_pool[current_commit]["author"],
                "date": commit_pool[current_commit]["date"],
                "summary": commit_pool[current_commit]["summary"]
            }
            complete_codebase_lines.append(line_context)
            
    # Combine back to reconstruct full unadulterated source code text
    full_file_text = "\n".join(raw_file_content_builder)
    return complete_codebase_lines, full_file_text

def chunk_code_file_with_blame(file_path, file_content, line_metadata, file_extension):
    """Chunks code using LangChain language structures and extracts metadata mappings."""
    # Mappings for specialized structural splitting using LangChain's built-in enums
    lang_map = {
        '.py': Language.PYTHON, 
        '.js': Language.JS, 
        '.ts': Language.TS,
        '.php': Language.PHP,
        '.twig': Language.HTML # Twig shares layout structures closely mapped by HTML hooks
    }
    
    lang = lang_map.get(file_extension)
    
    # Choose splitter type based on whether a LangChain Language preset exists
    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1200, chunk_overlap=200
        )
    else:
        # Fallback for .scss and other formats: use standard text rules
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
    
    # Generate LangChain text splits
    chunks = splitter.split_text(file_content)
    processed_chunks = []
    
    for chunk in chunks:
        # Resolve which line numbers this chunk spans by searching string content matches
        start_line = 1
        end_line = len(line_metadata)
        
        # Simple window tracking to discover matching lines for this snippet block
        chunk_lines = chunk.split('\n')
        first_line_clean = chunk_lines[0].strip() if chunk_lines else ""
        
        for i, line_meta in enumerate(line_metadata):
            if first_line_clean and line_meta["line_content"].strip() == first_line_clean:
                start_line = i + 1
                end_line = min(start_line + len(chunk_lines), len(line_metadata))
                break
                
        # Slice target metadata array to aggregate details
        meta_slice = line_metadata[start_line-1:end_line]
        
        authors = set()
        commits = set()
        latest_date = "0000-00-00 00:00:00"
        
        for m in meta_slice:
            authors.add(m["author"])
            commits.add(m["commit_hash"])
            if m["date"] > latest_date:
                latest_date = m["date"]
                
        # Generate the vector embedding array using the configured BGE model
        vector_embeddings = generate_local_embedding(chunk)
        
        processed_chunks.append({
            "text": chunk,
            "embeddings": vector_embeddings,
            "metadata": {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "authors": list(authors),
                "commit_hashes": list(commits),
                "last_modified": latest_date
            }
        })
        
    return processed_chunks

def ingest_local_repository(repo_root, target_subfolder):
    """Recursively scans repository, builds LangChain blocks, and embeds payload profiles."""
    all_final_vectors = []
    absolute_search_path = os.path.abspath(os.path.join(repo_root, target_subfolder))
    
    # Configured to look for .php, .twig, .scss, .js, and .py files
    valid_extensions = ('.js', '.ts', '.php', '.twig', '.scss')
    
    print(f"🔍 Initializing Scan: {absolute_search_path}")
    
    for root, _, files in os.walk(absolute_search_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, file)
                rel_path_from_repo = os.path.relpath(full_path, repo_root)
                
                print(f"📦 Indexing and Embedding: {rel_path_from_repo}")
                
                line_metadata, full_file_text = get_complete_file_context(repo_root, rel_path_from_repo)
                
                if not line_metadata or not full_file_text:
                    continue
                    
                file_chunks = chunk_code_file_with_blame(
                    rel_path_from_repo, full_file_text, line_metadata, ext
                )
                all_final_vectors.extend(file_chunks)
                
    return all_final_vectors

# --- Pipeline Tester Execution Block ---
if __name__ == "__main__":
    REPO_BASE_PATH = "../TestRepo" 
    TARGET_CODEBASE = "src/Pyz"
    
    ready_to_upsert_payloads = ingest_local_repository(REPO_BASE_PATH, TARGET_CODEBASE)
    print(f"\n✅ Finished processing {len(ready_to_upsert_payloads)} high-fidelity chunks with vector embeddings!")