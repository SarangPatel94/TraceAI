"""
Shared git helpers used by both the ingestion pipeline (ingest.py, for per-chunk
author/commit/date attribution) and the live query engine (query_engine.py, for the
agent's `get_git_history` tool). Keeping this in one place means the two never drift
out of sync on how they talk to git.
"""
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone

# git blame --line-porcelain prefixes each line-group with "<40-hex-sha> <orig> <final> [<count>]"
_BLAME_SHA_RE = re.compile(r'^([0-9a-f]{40})\s+\d+\s+\d+')


def run_git(args, cwd, timeout=30):
    """Runs a git subcommand and returns stdout, or None on any failure (incl. no cwd)."""
    if not cwd:
        return None
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


def detect_git_root(path):
    """Resolves the top-level directory of the git repo containing `path`, or None."""
    out = run_git(["rev-parse", "--show-toplevel"], cwd=path)
    return out.strip() if out else None


def file_log(git_root, full_path, limit=1):
    """
    Returns up to `limit` most recent commits that touched `full_path`, newest first,
    each as {"commit_hash", "author", "date", "subject"}. Empty list if untracked,
    no git root, or git is unavailable.
    """
    out = run_git(
        ["log", f"-{int(limit)}", "--format=%H\x1f%an\x1f%ad\x1f%s", "--date=short", "--", full_path],
        cwd=git_root,
    )
    if not out or not out.strip():
        return []
    commits = []
    for line in out.strip().splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({"commit_hash": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
    return commits


def blame_lines(git_root, full_path):
    """
    Runs `git blame --line-porcelain` ONCE for `full_path` and returns a list of
    (commit_hash, author, date) tuples, one per source line (0-indexed). Returns None
    if blame isn't available (untracked file, binary, no git root, etc) so callers can
    fall back to file_log() instead.
    """
    out = run_git(["blame", "--line-porcelain", "--", full_path], cwd=git_root)
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


def dominant_commit_for_range(blame_data, start_line, end_line, fallback):
    """
    Majority-vote the commit across [start_line, end_line] (0-indexed, inclusive) using
    precomputed blame_lines() output, so a chunk is attributed to whichever commit wrote
    most of it. Falls back to `fallback` (a dict shaped like a file_log() entry) if blame
    is unavailable or the range has nothing usable.
    """
    if not blame_data:
        return fallback

    start = max(0, start_line)
    end = min(len(blame_data), end_line + 1)
    segment = [entry for entry in blame_data[start:end] if entry[0]]

    if not segment:
        return fallback

    dominant_hash, _ = Counter(entry[0] for entry in segment).most_common(1)[0]
    for commit_hash, author, date in segment:
        if commit_hash == dominant_hash:
            return {"commit_hash": commit_hash, "author": author, "date": date, "subject": fallback.get("subject")}

    return fallback
