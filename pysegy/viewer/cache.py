"""Persistent metadata cache for the local viewer."""

import hashlib
import json
from pathlib import Path
import pickle
from typing import Optional

from platformdirs import user_cache_path

from ..scan import SegyScan, load_scan, save_scan


def default_cache_dir() -> Path:
    """Return the platform-appropriate viewer cache directory."""

    return user_cache_path("pysegy", appauthor=False) / "viewer"


def dataset_fingerprint(
    path: str,
    *,
    pattern: Optional[str],
    by_receiver: bool,
) -> str:
    """Build a stable cache key from dataset files and scan settings."""

    source = Path(path).expanduser().resolve()
    files = sorted(source.glob(pattern or "*.segy")) if source.is_dir() else [source]
    payload = {
        "path": str(source),
        "pattern": pattern,
        "by_receiver": by_receiver,
        "files": [
            (str(item), item.stat().st_size, item.stat().st_mtime_ns)
            for item in files
            if item.is_file()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(
    path: str,
    *,
    pattern: Optional[str],
    by_receiver: bool,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Return the cache file belonging to a dataset and scan settings."""

    key = dataset_fingerprint(path, pattern=pattern, by_receiver=by_receiver)
    return (cache_dir or default_cache_dir()) / f"{key}.scan"


def load_cached_scan(path: Path) -> Optional[SegyScan]:
    """Load a cached scan, removing an unreadable cache entry."""

    if not path.exists():
        return None
    try:
        return load_scan(str(path))
    except (EOFError, OSError, ValueError, TypeError, pickle.UnpicklingError):
        path.unlink(missing_ok=True)
        return None


def store_cached_scan(path: Path, scan: SegyScan) -> None:
    """Atomically store scan metadata in the viewer cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_scan(str(temporary), scan)
    temporary.replace(path)


def clear_cache(cache_dir: Optional[Path] = None) -> int:
    """Delete viewer scan entries and return the number removed."""

    directory = cache_dir or default_cache_dir()
    if not directory.exists():
        return 0
    entries = list(directory.glob("*.scan")) + list(directory.glob("*.tmp"))
    for entry in entries:
        entry.unlink(missing_ok=True)
    return len(entries)
