import json
from pathlib import Path
from typing import Any

import fire

from generation import answer_dataset_to_file, answer_question
from indexing import build_and_save_index, load_or_build_index
from pipeline import evaluate_search_results, search_dataset_to_file
from retrieval import search


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class RAGCLI:
    def index(
            self,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        retriever, corpus = build_and_save_index(
            folder_path,
            index_path,
            max_chunk_size=max_chunk_size,
        )
        return {
            "index_path": index_path,
            "documents_indexed": len(corpus),
            "retriever_type": type(retriever).__name__,
        }

    def build_index(
            self,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        return self.index(folder_path, index_path, max_chunk_size)

    def search(
            self,
            query: str,
            k: int = 10,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        retriever, corpus = load_or_build_index(
            folder_path,
            index_path,
            max_chunk_size=max_chunk_size,
        )
        results = search(query, retriever, corpus, k=k)
        return {
            "query": query,
            "k": k,
            "results": [result.to_dict() for result in results],
        }

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results",
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> str:
        retriever, corpus = load_or_build_index(
            folder_path,
            index_path,
            max_chunk_size=max_chunk_size,
        )
        output_path = search_dataset_to_file(
            dataset_path,
            k=k,
            output_dir=save_directory,
            retriever=retriever,
            corpus=corpus,
            max_chunk_size=max_chunk_size
        )
        return str(output_path)

    def answer(
            self,
            question: str,
            k: int = 10,
            model: str = DEFAULT_MODEL,
            base_url: str = "http://localhost:8000/v1",
            top_context_chunks: int = 3,
            max_context_chars: int = 12_000,
            temperature: float = 0.0,
            max_tokens: int = 256,
            timeout_seconds: float = 60.0,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        retriever, corpus = load_or_build_index(
            folder_path,
            index_path,
            max_chunk_size=max_chunk_size,
        )
        generated = answer_question(
            question,
            model=model,
            base_url=base_url,
            search_k=k,
            top_context_chunks=top_context_chunks,
            max_context_chars=max_context_chars,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retriever=retriever,
            corpus=corpus,
        )
        return generated.to_dict()

    def answer_dataset(
            self,
            student_search_results_path: str,
            model: str = DEFAULT_MODEL,
            base_url: str = "http://localhost:8000/v1",
            top_context_chunks: int = 3,
            max_context_chars: int = 12_000,
            temperature: float = 0.0,
            max_tokens: int = 256,
            timeout_seconds: float = 600.0,
            concurrency: int = 1,
            checkpoint_interval: int = 1,
            save_directory: str = "data/output/search_results_and_answer",
    ) -> str:
        output_path = answer_dataset_to_file(
            student_search_results_path,
            output_dir=save_directory,
            model=model,
            base_url=base_url,
            top_context_chunks=top_context_chunks,
            max_context_chars=max_context_chars,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            checkpoint_interval=checkpoint_interval,
        )
        return str(output_path)

    def evaluate(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
    ) -> dict[str, Any]:
        summary = evaluate_search_results(
            student_results_path,
            dataset_path,
            minimal_iou_threshold=minimal_iou_threshold,
        )
        return summary.to_dict()

    def evaluate_search_results(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
    ) -> dict[str, Any]:
        return self.evaluate(
            student_results_path,
            dataset_path,
            minimal_iou_threshold=minimal_iou_threshold,
        )

    def show_config(self) -> str:
        return json.dumps(
            {
                "default_model": DEFAULT_MODEL,
                "default_base_url": "http://localhost:8000/v1",
                "index_path": "data/processed/bm25_index",
            },
            indent=2,
        )

    def datasets(self, root: str = "data/datasets") -> dict[str, Any]:
        root_path = Path(root)
        return {
            "root": str(root_path),
            "datasets": [
                str(path)
                for path in sorted(root_path.rglob("*.json"))
            ],
        }


def main() -> None:
    try:
        fire.Fire(RAGCLI)
    except ValueError as e:
        print(f"\033[91mError\033[0m, {e}")
    except FileNotFoundError as e:
        print("\033[91mERROR\033[0m: file needed to run the programe ->", e)


if __name__ == "__main__":
    main()
