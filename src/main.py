from indexing import load_or_build_index
from retrieval import search


def main() -> None:
    retriever, corpus = load_or_build_index()

    query = "What does the usage property return in vLLM's KV cache manager?"
    results = search(query, retriever, corpus, k=3)

    for result in results:
        print(f"Rank {result.rank} | score={result.score:.2f}")
        print(f"Source: {result.file_path} [{result.kind}]")
        print(
            f"Span: {result.first_character_index}-"
            f"{result.last_character_index}"
        )
        print(result.text)
        print()


if __name__ == "__main__":
    main()
