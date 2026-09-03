"""Cross-platform path resolution utilities using pathlib."""
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is at src/shift_ami/utils/paths.py -> 3 levels up to project root
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path, None] = None) -> Path:
    """
    Resolve a relative or absolute path against a base directory (defaulting to project root).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(base_dir) if base_dir is not None else get_project_root()
    return (base / p).resolve()
