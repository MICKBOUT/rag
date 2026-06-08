import json
from pathlib import Path
from typing import Any, cast

import bm25s

from ast_cleaninig import get_ready_to_index_data


INDEX_PATH = "data/processed/bm25_index"
CHUNKS_PATH = "data/processed/chunks/chunks.jsonl"
RAW_ROOT = "data/raw/vllm-0.10.1"
DEFAULT_MAX_CHUNK_SIZE = 2000
REPO_ROOT = Path(__file__).resolve().parents[1]


def _metadata_path(index_path: str) -> Path:
    return _resolve_repo_path(index_path) / "metadata.json"


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("text", ""))


def _limit_entry_text(
        entry: dict[str, Any], max_chunk_size: int) -> dict[str, Any]:
    text = _entry_text(entry)
    if len(text) <= max_chunk_size:
        first = int(entry.get("first_character_index", 0))
        last = int(entry.get("last_character_index", 0))
        if (last - first) <= max_chunk_size:
            return entry
        limited_entry = dict(entry)
        limited_entry["last_character_index"] = first + max_chunk_size
        return limited_entry

    limited_entry = dict(entry)
    limited_entry["text"] = text[:max_chunk_size].rstrip()
    first = int(entry.get("first_character_index", 0))
    limited_entry["last_character_index"] = first + max_chunk_size
    return limited_entry


def _write_index_metadata(
        index_path: str, *, max_chunk_size: int, folder_path: str) -> None:
    metadata = {
        "max_chunk_size": max_chunk_size,
        "folder_path": folder_path,
    }
    _metadata_path(index_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_chunks(corpus: list[dict[str, Any]]) -> None:
    path = _resolve_repo_path(CHUNKS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in corpus:
            handle.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )


def load_chunks(
        chunks_path: str = CHUNKS_PATH) -> list[dict[str, Any]]:
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
    path = _metadata_path(index_path)
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], metadata)


def _corpus_has_absolute_paths(corpus: list[dict[str, Any]]) -> bool:
    for entry in corpus:
        file_path = str(entry.get("file_path", ""))
        if Path(file_path).is_absolute():
            return True
    return False


def build_and_save_index(
        folder_path: str = RAW_ROOT,
        index_path: str = INDEX_PATH,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
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
        index_path: str = INDEX_PATH,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
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
        folder_path: str = RAW_ROOT,
        index_path: str = INDEX_PATH,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    try:
        return load_index(index_path, max_chunk_size=max_chunk_size)
    except Exception:
        print("Index not loaded, building from scratch")
        return build_and_save_index(folder_path, index_path, max_chunk_size)
