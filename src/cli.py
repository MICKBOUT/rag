import json
from pathlib import Path
from typing import Any

import fire
from pydantic import ValidationError

from generation import answer_dataset_to_file, answer_question
from indexing import build_and_save_index, load_or_build_index
from pipeline import evaluate_search_results, search_dataset_to_file
from retrieval import search
from validation import (
    AnswerDatasetParams,
    AnswerParams,
    EvaluateParams,
    IndexParams,
    SearchDatasetParams,
    SearchParams,
    _fmt_errors,
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def _validation_error(exc: ValidationError) -> None:
    """Print a clean validation error and return None (no crash)."""
    print(f"\033[91mInvalid input:\033[0m\n{_fmt_errors(exc)}")


class RAGCLI:
    def index(
            self,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any] | None:
        try:
            params = IndexParams(
                folder_path=folder_path,
                index_path=index_path,
                max_chunk_size=max_chunk_size,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        retriever, corpus = build_and_save_index(
            params.folder_path,
            params.index_path,
            max_chunk_size=params.max_chunk_size,
        )
        return {
            "index_path": params.index_path,
            "documents_indexed": len(corpus),
            "retriever_type": type(retriever).__name__,
        }

    def build_index(
            self,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any] | None:
        return self.index(folder_path, index_path, max_chunk_size)

    def search(
            self,
            query: str,
            k: int = 10,
            folder_path: str = "data/raw/vllm-0.10.1",
            index_path: str = "data/processed/bm25_index",
            max_chunk_size: int = 2000,
    ) -> dict[str, Any] | None:
        try:
            params = SearchParams(
                query=query,
                k=k,
                folder_path=folder_path,
                index_path=index_path,
                max_chunk_size=max_chunk_size,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        retriever, corpus = load_or_build_index(
            params.folder_path,
            params.index_path,
            max_chunk_size=params.max_chunk_size,
        )
        results = search(params.query, retriever, corpus, k=params.k)
        return {
            "query": params.query,
            "k": params.k,
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
    ) -> str | None:
        try:
            params = SearchDatasetParams(
                dataset_path=dataset_path,
                k=k,
                save_directory=save_directory,
                folder_path=folder_path,
                index_path=index_path,
                max_chunk_size=max_chunk_size,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        retriever, corpus = load_or_build_index(
            params.folder_path,
            params.index_path,
            max_chunk_size=params.max_chunk_size,
        )
        output_path = search_dataset_to_file(
            params.dataset_path,
            k=params.k,
            output_dir=params.save_directory,
            retriever=retriever,
            corpus=corpus,
            max_chunk_size=params.max_chunk_size,
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
    ) -> dict[str, Any] | None:
        try:
            params = AnswerParams(
                question=question,
                k=k,
                model=model,
                base_url=base_url,
                top_context_chunks=top_context_chunks,
                max_context_chars=max_context_chars,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                folder_path=folder_path,
                index_path=index_path,
                max_chunk_size=max_chunk_size,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        retriever, corpus = load_or_build_index(
            params.folder_path,
            params.index_path,
            max_chunk_size=params.max_chunk_size,
        )
        generated = answer_question(
            params.question,
            model=params.model,
            base_url=params.base_url,
            search_k=params.k,
            top_context_chunks=params.top_context_chunks,
            max_context_chars=params.max_context_chars,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout_seconds=params.timeout_seconds,
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
    ) -> str | None:
        try:
            params = AnswerDatasetParams(
                student_search_results_path=student_search_results_path,
                model=model,
                base_url=base_url,
                top_context_chunks=top_context_chunks,
                max_context_chars=max_context_chars,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
                checkpoint_interval=checkpoint_interval,
                save_directory=save_directory,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        output_path = answer_dataset_to_file(
            params.student_search_results_path,
            output_dir=params.save_directory,
            model=params.model,
            base_url=params.base_url,
            top_context_chunks=params.top_context_chunks,
            max_context_chars=params.max_context_chars,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout_seconds=params.timeout_seconds,
            concurrency=params.concurrency,
            checkpoint_interval=params.checkpoint_interval,
        )
        return str(output_path)

    def evaluate(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
            threshold: float | None = None,
    ) -> dict[str, Any] | None:
        try:
            params = EvaluateParams(
                student_results_path=student_results_path,
                dataset_path=dataset_path,
                minimal_iou_threshold=minimal_iou_threshold,
                threshold=threshold,
            )
        except ValidationError as exc:
            _validation_error(exc)
            return None

        summary = evaluate_search_results(
            params.student_results_path,
            params.dataset_path,
            minimal_iou_threshold=params.minimal_iou_threshold,
            threshold=params.threshold,
        )
        return summary.to_dict()

    def evaluate_search_results(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
    ) -> dict[str, Any] | None:
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
    except RuntimeError:
        print(
            "\033[91mError\033[0m: "
            "the script has timed out, "
            "try with a smaller input or increase the timeout"
        )
    except ValueError as e:
        print(f"\033[91mError\033[0m: {e}")
    except FileNotFoundError as e:
        print("\033[91mERROR\033[0m: file needed to run the program ->", e)


if __name__ == "__main__":
    main()
