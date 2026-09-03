"""Reproducibility and provenance tracking utilities."""
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python built-in random, NumPy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit_hash(repo_dir: Optional[Path] = None) -> str:
    """Return current git commit hash or 'unknown' if not in a git repo."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "git_commit_unavailable"


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for provenance verification."""
    if not file_path.exists():
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_provenance_metadata(config: Dict[str, Any], extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble structured provenance record for experiment artifacts."""
    meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "git_commit": get_git_commit_hash(),
        "random_seed": config.get("random_seed", 42),
        "config_snapshot": config,
    }
    if extra_info:
        meta.update(extra_info)
    return meta
