import json
from pathlib import Path
from typing import Any

import fire
from pydantic import ValidationError

from generation import answer_dataset_to_file, answer_question, DEFAULT_MODEL
from indexing import build_and_save_index, load_or_build_index
from pipeline import evaluate_search_results, search_dataset_to_file
from retrieval import search

# Import the validation models from your validation module
from validation import (
    IndexParams,
    SearchParams,
    SearchDatasetParams,
    AnswerParams,
    AnswerDatasetParams,
    EvaluateParams,
    DatasetsParams,
)


class RAGCLI:
    def index(
            self,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        # Validate using Pydantic
        args = IndexParams(
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = build_and_save_index(
            args.folder_path,
            args.index_path,
            max_chunk_size=args.max_chunk_size,
        )
        return {
            "index_path": args.index_path,
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
        # Validate using Pydantic
        args = SearchParams(
            query=query,
            k=k,
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = load_or_build_index(
            args.folder_path,
            args.index_path,
            max_chunk_size=args.max_chunk_size,
        )
        results = search(args.query, retriever, corpus, k=args.k)
        return {
            "query": args.query,
            "k": args.k,
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
        # Validate using Pydantic
        args = SearchDatasetParams(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory,
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = load_or_build_index(
            args.folder_path,
            args.index_path,
            max_chunk_size=args.max_chunk_size,
        )
        output_path = search_dataset_to_file(
            args.dataset_path,
            k=args.k,
            output_dir=args.save_directory,
            retriever=retriever,
            corpus=corpus,
            max_chunk_size=args.max_chunk_size
        )
        return str(output_path)

    def answer(
            self,
            question: str,
            k: int = 10,
            model: str = DEFAULT_MODEL,
            base_url: str = "http://localhost:8000/v1",
            top_context_chunks: int = 3,
            max_tokens: int = 256,
            timeout_seconds: float = 60.0,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any]:
        # Validate using Pydantic
        args = AnswerParams(
            question=question,
            k=k,
            model=model,
            base_url=base_url,
            top_context_chunks=top_context_chunks,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = load_or_build_index(
            args.folder_path,
            args.index_path,
            max_chunk_size=args.max_chunk_size,
        )
        generated = answer_question(
            args.question,
            model=args.model,
            base_url=args.base_url,
            search_k=args.k,
            top_context_chunks=args.top_context_chunks,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
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
            max_tokens: int = 256,
            timeout_seconds: float = 600.0,
            concurrency: int = 1,
            checkpoint_interval: int = 1,
            save_directory: str = "data/output/search_results_and_answer",
    ) -> str:
        # Validate using Pydantic
        args = AnswerDatasetParams(
            student_search_results_path=student_search_results_path,
            model=model,
            base_url=base_url,
            top_context_chunks=top_context_chunks,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            checkpoint_interval=checkpoint_interval,
            save_directory=save_directory,
        )

        output_path = answer_dataset_to_file(
            args.student_search_results_path,
            output_dir=args.save_directory,
            model=args.model,
            base_url=args.base_url,
            top_context_chunks=args.top_context_chunks,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
        )
        return str(output_path)

    def evaluate(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
            threshold: float | None = None,
    ) -> dict[str, Any]:
        # Validate using Pydantic
        args = EvaluateParams(
            student_results_path=student_results_path,
            dataset_path=dataset_path,
            minimal_iou_threshold=minimal_iou_threshold,
            threshold=threshold,
        )

        summary = evaluate_search_results(
            args.student_results_path,
            args.dataset_path,
            minimal_iou_threshold=args.minimal_iou_threshold,
            threshold=args.threshold
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
        # Validate using Pydantic
        args = DatasetsParams(root=root)
        root_path = Path(args.root)
        
        return {
            "root": str(root_path),
            "datasets": [
                str(path)
                for path in sorted(root_path.rglob("*.json"))
            ],
        }



def main() -> None:
    error_str = "\033[91mError\033[0m:"
    validation_error = "\033[91mVALIDATION ERROR\033[0m:"
    try:
        fire.Fire(RAGCLI)
    except ValidationError as e:
        print(f"{validation_error} Invalid arguments provided to CLI command.")
        print("==========")
        for error in e.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            print(
                f"  \033[93m{location}\033[0m: {error['msg']} "
                f"(Provided input: '{error.get('input')}')"
            )
        print("==========")
    except (ValueError, RuntimeError) as e:
        print(error_str, e)
    except TypeError as e:
        print(f"{error_str} Invalid argument combinations.\nDetail: {e}")
    except FileNotFoundError as e:
        print(f"{error_str} File needed to run the program ->", e)


if __name__ == "__main__":
    main()