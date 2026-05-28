from typing import Any

import bm25s

from ast_proto import get_ready_to_index_data


INDEX_PATH = "data/processed/bm25_index"


def _entry_text(entry: dict[str, Any] | str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("text", ""))
    return entry


def _entry_index(entry: Any) -> int:
    return int(entry)


def build_and_save_index() -> tuple[bm25s.BM25, list[Any]]:
    corpus = get_ready_to_index_data()
    corpus_texts = [_entry_text(entry) for entry in corpus]

    corpus_tokens = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(INDEX_PATH, corpus=corpus)

    print(f"Index built and saved ({len(corpus)} documents).")
    return retriever, corpus


def load_index() -> tuple[bm25s.BM25, list[Any]]:
    retriever = bm25s.BM25.load(INDEX_PATH, load_corpus=True)
    corpus = retriever.corpus

    if corpus is None:
        raise ValueError("Corpus is None — rebuilding.")

    print("Index loaded from disk")
    return retriever, list(corpus)


def main() -> None:
    try:
        retriever, corpus = load_index()
    except Exception:
        print("Folder not loaded, building from scratch")
        retriever, corpus = build_and_save_index()

    query = "What does the usage property return in vLLM's KV cache manager?"
    # query = (
    #     "vLLM supports large-scale deployment combining Data Parallel "
    #     "attention with Expert or Tensor Parallel MoE layers for "
    #     "distributed "serving of Mixture of Experts models."
    # )
    query_tokens = bm25s.tokenize(query)

    docs, scores = retriever.retrieve(query_tokens, k=1)

    raw = docs[0, 0]
    entry = raw if isinstance(raw, dict) else corpus[_entry_index(raw)]
    text = _entry_text(entry)
    file_path = (
        entry.get("file_path", "unknown")
        if isinstance(entry, dict)
        else "unknown"
    )
    kind = (
        entry.get("kind", "unknown")
        if isinstance(entry, dict)
        else "unknown"
    )

    print(f"Best result (score: {scores[0, 0]:.2f})")
    print(f"Source: {file_path} [{kind}]")
    print(text)


if __name__ == "__main__":
    main()
