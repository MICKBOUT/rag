import json
from pathlib import Path
from typing import Any, cast

import bm25s

from .ast_cleaninig import get_ready_to_index_data
from .config import Config


def _metadata_path(index_path: str) -> Path:
    """Return the metadata file path for an index.

    Args:
        index_path: The repository-relative or absolute path to the
            index folder.

    Returns:
        A `Path` pointing to the `metadata.json` file inside the
        resolved index directory.
    """

    return _resolve_repo_path(index_path) / "metadata.json"


def _resolve_repo_path(path: str | Path) -> Path:
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


def _entry_text(entry: dict[str, Any]) -> str:
    """Return the text content for a corpus entry.

    Args:
        entry: A corpus entry mapping expected to contain a "text"
            key.

    Returns:
        The `text` value coerced to `str`, or an empty string if
        missing.
    """

    return str(entry.get("text", ""))


def _source_span_length(entry: dict[str, Any]) -> int:
    """Return the span length of the source text for an entry.

    The BM25-processed `text` may include prefixes (e.g. "FILE:/HEADING:")
    which make `len(text)` larger than the actual source span. This
    function computes the length using the `first_character_index` and
    `last_character_index` fields to obtain the true span length.

    Args:
        entry: A corpus entry expected to contain
            `first_character_index` and `last_character_index` keys.

    Returns:
        The non-negative integer span length (last - first).
    """

    first = int(entry.get("first_character_index", 0))
    last = int(entry.get("last_character_index", 0))
    return max(0, last - first)


def _limit_entry_text(
        entry: dict[str, Any], max_chunk_size: int) -> dict[str, Any]:
    """Return a copy of `entry` whose text/span is clipped to `max_chunk_size`.

    This helper ensures that both the BM25 `text` and the source span
    (`last_character_index`) do not exceed `max_chunk_size`. If clipping
    is necessary a shallow copy of `entry` is returned with the
    adjusted fields; otherwise the original `entry` is returned.

    Args:
        entry: The corpus entry to potentially limit.
        max_chunk_size: Maximum allowed characters for text/span.

    Returns:
        A dict representing the possibly-limited entry.
    """

    text = _entry_text(entry)
    span_length = _source_span_length(entry)

    if len(text) <= max_chunk_size and span_length <= max_chunk_size:
        return entry

    limited_entry = dict(entry)

    if len(text) > max_chunk_size:
        limited_entry["text"] = text[:max_chunk_size].rstrip()

    if span_length > max_chunk_size:
        first = int(entry.get("first_character_index", 0))
        limited_entry["last_character_index"] = first + max_chunk_size

    return limited_entry


def _write_index_metadata(
        index_path: str, *, max_chunk_size: int, folder_path: str) -> None:
    """Write index metadata to the index folder.

    Stores information about how the index was created so subsequent
    loads can validate compatibility (e.g. `max_chunk_size`).

    Args:
        index_path: Path to the index directory.
        max_chunk_size: Maximum chunk size used when building the
            index.
        folder_path: The source folder path that was indexed.
    """

    metadata = {
        "max_chunk_size": max_chunk_size,
        "folder_path": folder_path,
    }
    _metadata_path(index_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_chunks(corpus: list[dict[str, Any]]) -> None:
    """Write the serialized corpus to the configured chunks file.

    Each entry is written as a single JSON line to `Config.CHUNKS_PATH`.

    Args:
        corpus: A list of corpus entry dicts to persist.
    """

    path = _resolve_repo_path(Config.CHUNKS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in corpus:
            handle.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )


def load_chunks(
        chunks_path: str = Config.CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load chunked corpus entries from a newline-delimited JSON file.

    Args:
        chunks_path: Path to the chunks file (defaults to
            `Config.CHUNKS_PATH`).

    Returns:
        A list of corpus entry dicts.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
    """

    path = _resolve_repo_path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run `student index` first."
        )

    chunks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunks.append(cast(dict[str, Any], json.loads(line)))
    return chunks


def _read_index_metadata(index_path: str) -> dict[str, Any] | None:
    """Read metadata.json from an index path if present.

    Args:
        index_path: Path to the index folder.

    Returns:
        The parsed metadata dict, or `None` if the metadata file does
        not exist.
    """

    path = _metadata_path(index_path)
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], metadata)


def _corpus_has_absolute_paths(corpus: list[dict[str, Any]]) -> bool:
    """Return True if any corpus entry contains an absolute file path.

    This is used as a safety check to detect indexes created on another
    machine or with absolute paths that would not be portable.

    Args:
        corpus: List of corpus entries, each possibly containing a
            `file_path` key.

    Returns:
        `True` if any `file_path` is absolute, otherwise `False`.
    """

    for entry in corpus:
        file_path = str(entry.get("file_path", ""))
        if Path(file_path).is_absolute():
            return True
    return False


def build_and_save_index(
        folder_path: str = Config.RAW_ROOT,
        index_path: str = Config.INDEX_PATH,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Build a BM25 index from source files and persist it.

    This function collects ready-to-index data from `folder_path`,
    applies chunk size limiting, tokenizes the corpus for BM25,
    builds and saves the retriever and also persists the chunk file
    and index metadata.

    Args:
        folder_path: Source folder to index.
        index_path: Path where the BM25 index will be saved.
        max_chunk_size: Maximum chunk size used when generating
            corpus entries.

    Returns:
        A tuple containing the constructed `bm25s.BM25` retriever and
        the list of corpus entry dicts.

    Raises:
        ValueError: If no corpus entries were produced from the source
            folder.
    """

    resolved_index_path = _resolve_repo_path(index_path)
    corpus = [
        _limit_entry_text(entry, max_chunk_size)
        for entry in get_ready_to_index_data(folder_path)
    ]
    if not corpus:
        raise ValueError(
            f"No source files were indexed from {folder_path}. "
            "Check that the corpus path is correct."
        )
    corpus_texts = [_entry_text(entry) for entry in corpus]

    corpus_tokens = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(resolved_index_path), corpus=corpus)
    _write_chunks(corpus)
    _write_index_metadata(
        str(resolved_index_path),
        max_chunk_size=max_chunk_size,
        folder_path=folder_path,
    )

    print(f"Index built and saved ({len(corpus)} documents).")
    return retriever, corpus


def load_index(
        index_path: str = Config.INDEX_PATH,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Load a saved BM25 index and validate compatibility.

    This attempts to load the index from `index_path` and performs a
    series of checks to ensure the corpus exists, uses relative paths,
    and that the stored `max_chunk_size` matches the requested value.

    Args:
        index_path: Path to the saved BM25 index.
        max_chunk_size: Expected max chunk size for compatibility.

    Returns:
        A tuple of the loaded `bm25s.BM25` retriever and the corpus list.

    Raises:
        ValueError: When the index is missing required data or appears
            incompatible and should be rebuilt.
    """

    resolved_index_path = _resolve_repo_path(index_path)
    retriever = bm25s.BM25.load(str(resolved_index_path), load_corpus=True)
    corpus = retriever.corpus

    if corpus is None:
        raise ValueError("Corpus is None — rebuilding.")

    normalized_corpus = list(corpus)
    if _corpus_has_absolute_paths(normalized_corpus):
        raise ValueError("Corpus paths are absolute — rebuilding.")

    metadata = _read_index_metadata(str(resolved_index_path))
    if metadata is None:
        raise ValueError("Index metadata missing — rebuilding.")
    if int(metadata.get("max_chunk_size", -1)) != max_chunk_size:
        raise ValueError("Index chunk size changed — rebuilding.")

    return retriever, normalized_corpus


def load_or_build_index(
        folder_path: str = Config.RAW_ROOT,
        index_path: str = Config.INDEX_PATH,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Load an existing index or build a new one if loading fails.

    This convenience wrapper first attempts to load the index using
    `load_index`. If that fails for any reason it falls back to
    `build_and_save_index` to create a fresh index from `folder_path`.

    Args:
        folder_path: Source folder to index if building is required.
        index_path: Path to the index to load or create.
        max_chunk_size: Max chunk size parameter for load/build.

    Returns:
        A tuple with the retriever and the normalized corpus list.
    """

    try:
        return load_index(index_path, max_chunk_size=max_chunk_size)
    except Exception:
        print("Index not loaded, building from scratch")
        return build_and_save_index(folder_path, index_path, max_chunk_size)
