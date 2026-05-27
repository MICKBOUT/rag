from typing import Any

import bm25s

from ast_proto import get_ready_to_index_data


INDEX_PATH = "bm25s_index_llm"


def build_and_save_index() -> tuple[bm25s.BM25, list[Any]]:
    corpus = get_ready_to_index_data()

    corpus_tokens = bm25s.tokenize(corpus)

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

    # normalize: bm25s may deserialize as list of dicts
    corpus = [
        doc.get('text', str(doc)) if isinstance(doc, dict) else doc
        for doc in corpus
    ]

    print("Index loaded from disk")
    return retriever, corpus


def main() -> None:
    try:
        retriever, corpus = load_index()
    except Exception:
        print("Folder not loaded, building from scratch")
        retriever, corpus = build_and_save_index()

    query = "What does the usage property return in vLLM's KV cache manager?"
    query_tokens = bm25s.tokenize(query)

    docs, scores = retriever.retrieve(query_tokens, k=1)

    raw = docs[0, 0]
    if isinstance(raw, dict):  # to handle when the bm25 is loaded or created
        text = raw.get('text', str(raw))
    else:
        text = corpus[raw]   # int index into our normalized list

    print(f"Best result (score: {scores[0, 0]:.2f}):\n{text}")


if __name__ == "__main__":
    main()
