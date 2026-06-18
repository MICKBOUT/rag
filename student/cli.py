import json
from typing import Any

import fire
from pydantic import ValidationError
from litellm.exceptions import InternalServerError

from .generation import answer_dataset_to_file, answer_question
from .indexing import build_and_save_index, load_or_build_index
from .pipeline import evaluate_search_results, search_dataset_to_file
from .retrieval import search
from .config import Config
from .validation import (
    IndexParams,
    SearchParams,
    SearchDatasetParams,
    AnswerParams,
    AnswerDatasetParams,
    EvaluateParams,
    MinimalAnswer,
    AnswerDatasetOutput,
    MinimalSearchResults,
    FileStudentSearchResults,
)


class RAGCLI:
    """
    RAGCLI
    ======
    Command-line interface wrapper for a Retrieval-Augmented Generation (RAG)
    pipeline. Provides methods to build and load vector indexes, run searches,
    generate answers via an LLM, run batch operations over datasets, and
    evaluate search/answer results.

    Each method performs necessary index loading/creation as needed and may
    read from or write to disk (indexes, output files). The class methods are
    thin adapters that convert simple parameter inputs into the underlying
    pipeline functions and return plain Python serializable results
    (dicts or string paths).

    Methods
    -------
    index(folder_path: str = Config.RAW_ROOT,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> dict[str, Any]
        Build and save an index from documents found under `folder_path`. If
        successful, returns a dictionary with:
        - "index_path": path to the saved index
        - "documents_indexed_count": number of documents indexed
        - "retriever_type": name of the retriever class used

    search(question: str,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> dict[str, Any]
        Ensure an index is available (load or build), run a nearest-neighbor
        search for `question` returning the top `k` results. Returns a dict
        with:
          - "question": the input question
          - "k": number of results requested
          - "results": list of result objects serialized to dict (one per hit)

    answer(question: str,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> dict[str, Any]
        Load or build the index, retrieve relevant context chunks for
        `question` (using `k` as the search width), and invoke the configured
        LLM endpoint (`model` and `base_url`) to generate a final answer.
        `top_context_chunks` controls how many retrieved chunks are provided
        to the model. Returns the model's generated object serialized to a
        dict (including answer text and metadata).

    answer_dataset(student_search_results_path: str,
        save_directory: str | Path = Config.DEFAULT_OUTPUT_DIR_ANSWER) -> str
        Batch-generate answers for a collection of search results previously
        produced (e.g., from `search_dataset`). Supports concurrency, periodic
        checkpointing, and writes outputs to `save_directory`. Returns the
        path to the output file.

    evaluate(student_results_path: str,
        threshold: float | None = None) -> dict[str, Any]
        Evaluate student (predicted) search results against a ground-truth
        dataset. `minimal_iou_threshold` controls the minimal overlap IoU
        required to consider a prediction matching a ground-truth item;
        `threshold` may be used to filter predictions by score if supported.
        Returns a summary object as a dict containing metrics and aggregate
        results.

    evaluate_search_results(student_results_path: str,
        minimal_iou_threshold: float = 0.05) -> dict[str, Any]
        Convenience wrapper delegating to `evaluate`. Same return structure.

    show_config() -> str
        Return a JSON-formatted string exposing key configuration values (e.g.,
        default model, base URL, index path). Useful for quick diagnostics.

    Notes
    -----
    - Methods may perform I/O (reading/writing index files, output files).
    - Returned dicts are intended to be JSON-serializable for CLI or API usage.
    - Exceptions from underlying pipeline functions (indexing, searching, model
      calls, or file I/O) are propagated to the caller; callers should catch
      and handle them as appropriate.
    """
    def index(
            self,
            folder_path: str = Config.RAW_ROOT,
            index_path: str = Config.INDEX_PATH,
            max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, Any]:

        index_params = IndexParams(
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = build_and_save_index(index_params)
        return {
            "index_path": index_params.index_path,
            "documents_indexed_count": len(corpus),
            "retriever_type": type(retriever).__name__,
        }

    def search(
            self,
            question: str,
            k: int = Config.DEFAULT_SEARCH_K,
            folder_path: str = Config.INDEX_PATH,
            index_path: str = Config.INDEX_PATH,
            max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, Any]:
        search_params = SearchParams(
            question=question,
            k=k,
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = load_or_build_index(index_params=IndexParams(
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        ))

        search_result = search(
            question=search_params.question,
            retriever=retriever,
            corpus=corpus,
            k=search_params.k
        )
        retrieved_sources = [result.to_dict() for result in search_result]

        result = MinimalSearchResults(
            question_id="Single question (No id)",
            question=search_params.question,
            retrieved_sources=retrieved_sources
        )

        return result.to_dict()

    def search_dataset(
            self,
            dataset_path: str,
            k: int = Config.DEFAULT_SEARCH_K,
            save_directory: str = Config.DEFAULT_OUTPUT_DIR,
            folder_path: str = Config.RAW_ROOT,
            index_path: str = Config.INDEX_PATH,
            max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
    ) -> str:
        search_dataset_params = SearchDatasetParams(
            dataset_path=dataset_path,
            k=k,
            save_directory=str(save_directory),
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        )

        retriever, corpus = load_or_build_index(index_params=IndexParams(
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        ))

        output_path = search_dataset_to_file(
            search_dataset_params.dataset_path,
            k=search_dataset_params.k,
            output_dir=search_dataset_params.save_directory,
            retriever=retriever,
            corpus=corpus,
            max_chunk_size=search_dataset_params.max_chunk_size
        )
        FileStudentSearchResults(
            file_path=output_path
        )

        return output_path

    def answer(
            self,
            question: str,
            k: int = Config.DEFAULT_SEARCH_K,
            model: str = Config.DEFAULT_MODEL,
            base_url: str = Config.DEFAULT_BASE_URL,
            top_context_chunks: int | None = Config.DEFAULT_TOP_CONTEXT_CHUNKS,
            max_tokens: int = Config.DEFAULT_MAX_TOKENS,
            timeout_seconds: float = Config.TIMEOUT_SECONDS_SINGLE_QUESTION,
            folder_path: str = Config.RAW_ROOT,
            index_path: str = Config.INDEX_PATH,
            max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE,
    ) -> dict[str, Any]:
        answer_params = AnswerParams(
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

        retriever, corpus = load_or_build_index(IndexParams(
            folder_path=folder_path,
            index_path=index_path,
            max_chunk_size=max_chunk_size,
        ))

        generated_answer = answer_question(
            question=answer_params.question,
            corpus=corpus,
            model=answer_params.model,
            base_url=answer_params.base_url,
            search_k=answer_params.k,
            top_context_chunks=answer_params.top_context_chunks,
            max_tokens=answer_params.max_tokens,
            timeout_seconds=answer_params.timeout_seconds,
            retriever=retriever,
        )
        MinimalAnswer(
                answer=generated_answer.answer,
                question_id=generated_answer.question_id,
                question=generated_answer.question,
                retrieved_sources=generated_answer.retrieved_sources,
            )

        return generated_answer.to_dict()

    def answer_dataset(
            self,
            student_search_results_path: str,
            model: str = Config.DEFAULT_MODEL,
            base_url: str = Config.DEFAULT_BASE_URL,
            top_context_chunks: int | None = Config.DEFAULT_TOP_CONTEXT_CHUNKS,
            max_tokens: int = Config.DEFAULT_MAX_TOKENS,
            timeout_seconds: float = Config.TIMEOUT_SECONDS_MULTIPLE_QUESTION,
            concurrency: int = 1,
            checkpoint_interval: int = 1,
            save_directory: str = Config.DEFAULT_OUTPUT_DIR_ANSWER,
    ) -> str:
        args = AnswerDatasetParams(
            student_search_results_path=student_search_results_path,
            model=model,
            base_url=base_url,
            top_context_chunks=top_context_chunks,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            checkpoint_interval=checkpoint_interval,
            save_directory=str(save_directory),
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
        AnswerDatasetOutput(file_path=str(output_path))
        return str(output_path)

    def evaluate(
            self,
            student_results_path: str,
            dataset_path: str,
            minimal_iou_threshold: float = 0.05,
            threshold: float | None = None,
    ) -> dict[str, Any]:
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

    def show_config(self) -> str:
        return json.dumps({
                "default_model": Config.DEFAULT_MODEL,
                "default_base_url": Config.DEFAULT_BASE_URL,
                "index_path": Config.INDEX_PATH,
            },
            indent=2
        )


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
                f"  \033[93m{location}\033[0m: {error['msg']} \n"
                f"(Provided input: '{error.get('input')}')"
            )
        print("==========")
    except (ValueError, RuntimeError) as e:
        print(error_str, e)
    except TypeError as e:
        print(f"{error_str} Invalid argument combinations.\nDetail: {e}")
    except FileNotFoundError as e:
        print(f"{error_str} File needed to run the program ->", e)
    except InternalServerError:
        print(f"{error_str} Please make sure your local vLLM/inference "
              "server is active and listening on the configured port "
              f"(default: {Config.DEFAULT_BASE_URL}).")


if __name__ == "__main__":
    main()
