from typing import Any
from pathlib import Path

import bm25s

from ast_cleaninig import get_ready_to_index_data


INDEX_PATH = "data/processed/bm25_index"
RAW_ROOT = "data/raw/vllm-0.10.1"


def _entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("text", ""))


def _corpus_has_absolute_paths(corpus: list[dict[str, Any]]) -> bool:
    for entry in corpus:
        file_path = str(entry.get("file_path", ""))
        if Path(file_path).is_absolute():
            return True
    return False


def build_and_save_index(
        folder_path: str = RAW_ROOT,
        index_path: str = INDEX_PATH
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    corpus = get_ready_to_index_data(folder_path)
    corpus_texts = [_entry_text(entry) for entry in corpus]

    corpus_tokens = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(index_path, corpus=corpus)

    print(f"Index built and saved ({len(corpus)} documents).")
    return retriever, corpus


def load_index(
        index_path: str = INDEX_PATH
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    retriever = bm25s.BM25.load(index_path, load_corpus=True)
    corpus = retriever.corpus

    if corpus is None:
        raise ValueError("Corpus is None — rebuilding.")

    normalized_corpus = list(corpus)
    if _corpus_has_absolute_paths(normalized_corpus):
        raise ValueError("Corpus paths are absolute — rebuilding.")

    return retriever, normalized_corpus


def load_or_build_index(
        folder_path: str = RAW_ROOT,
        index_path: str = INDEX_PATH
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    try:
        return load_index(index_path)
    except Exception:
        print("Index not loaded, building from scratch")
        return build_and_save_index(folder_path, index_path)
