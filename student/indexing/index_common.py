from pathlib import Path
from ..config import Config


def _metadata_path(index_path: str) -> Path:
    """Return the metadata file path for an index.

    Args:
        index_path: The repository-relative or absolute path to the
            index folder.

    Returns:
        A `Path` pointing to the `metadata.json` file inside the
        resolved index directory.
    """

    return resolve_repo_path(index_path) / "metadata.json"


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root.

    If `path` is absolute, it is returned as-is. Otherwise the path is
    resolved relative to `Config.REPO_ROOT` and returned as an absolute
    `Path`.

    Args:
        path: A path string or `Path` which may be relative to the repo
            root.

    Returns:
        An absolute `Path` instance for the provided `path`.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Config.REPO_ROOT / candidate).resolve()
