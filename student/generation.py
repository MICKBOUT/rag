import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from typing import Sequence, Annotated

import dspy
from tqdm import tqdm

from .models import GeneratedAnswer
from .retrieval import search
from .config import Config


class RAGSignature(dspy.Signature):
    """Signature used for RAG generation.

    This signature describes the inputs and outputs used by the
    chain-of-thought predictor for retrieval-augmented generation.

    Attributes:
        context: Relevant source text chunks from documentation.
        question: The question that needs to be answered.
        answer: A precise, factually accurate answer based strictly on
            the provided context.
    """
    context: Annotated[str, dspy.InputField(
        desc="Relevant source text chunks from documentation.")]
    question: Annotated[str, dspy.InputField(
        desc="The question that needs to be answered.")]
    answer: Annotated[str, dspy.OutputField(
        desc="A precise, "
        "factually accurate answer based strictly on the context.")
    ]


def _get_source_text(source: dict[str, Any]) -> str:
    """Extract text for a retrieved source entry.

    The function attempts to return the text in the following order:
    1. If the `source` dict contains a non-empty "text" field, return it.
    2. If there is a `file_path` that exists on disk, read the file and
       return the substring between `first_character_index` and
       `last_character_index` (inclusive).
    3. Otherwise return an empty string.

    Args:
        source: A mapping describing a retrieved source. Expected keys
            include "text", "file_path", "first_character_index",
            and "last_character_index".

    Returns:
        The extracted text for the source or an empty string if no text
        could be obtained.
    """

    if "text" in source and source["text"]:
        return str(source["text"])

    file_path = Path(source.get("file_path", ""))
    if file_path.exists():
        try:
            content = file_path.read_text(encoding="utf-8")
            start = int(source.get("first_character_index", 0))
            end = int(source.get("last_character_index", len(content)))
            return content[start:end + 1]
        except Exception:
            return ""
    return ""


def answer_question(
    question: str,
    corpus: Sequence[dict[str, Any]],
    *,
    model: str = Config.DEFAULT_MODEL,
    base_url: str = Config.DEFAULT_BASE_URL,
    search_k: int = Config.DEFAULT_SEARCH_K,
    top_context_chunks: int | None = Config.DEFAULT_TOP_CONTEXT_CHUNKS,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    timeout_seconds: float = Config.DEFAULT_TIMEOUT_SECONDS,
    retriever: Any = None,
) -> GeneratedAnswer:
    """Answer a single question using retrieval-augmented generation.

    This function runs a retrieval step against the provided `corpus`,
    constructs a context from the top returned chunks, and invokes the
    language model to generate a concise answer.

    Args:
        question: The question string to answer.
        corpus: Sequence of documents/chunks available to the retriever.
        model: Model name to use for generation.
        base_url: Base API URL for the LM provider.
        search_k: Number of results to retrieve from the retriever.
        top_context_chunks: Maximum number of retrieved chunks to include
            in the context passed to the model.
        max_tokens: Maximum number of tokens to generate.
        timeout_seconds: Timeout (in seconds) for LM calls.
        retriever: Optional retriever instance to use for search. If
            `None`, a default retriever is used by `search`.

    Returns:
        A `GeneratedAnswer` object containing the question id, generated
        answer text, metadata about retrieved sources, and generation
        parameters.
    """

    results = search(question, retriever, corpus, k=search_k)

    context_pieces = [res.text for res in results[:top_context_chunks]]

    while context_pieces and sum(len(p) for p in context_pieces) > 7000:
        context_pieces.pop()

    context_str = "\n---\n".join(context_pieces)

    lm = dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key="none",
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )

    with dspy.context(lm=lm):
        predictor = dspy.ChainOfThought(RAGSignature)
        prediction = predictor(context=context_str, question=question)

    retrieved_sources = [res.to_MinimalSource() for res in results]
    return GeneratedAnswer(
        question_id="single_question",
        question_str=question,
        retrieved_sources=retrieved_sources,
        answer=prediction.answer,
    )


def answer_dataset_to_file(
    student_search_results_path: str | Path,
    *,
    output_dir: str = Config.DEFAULT_OUTPUT_DIR,
    model: str = Config.DEFAULT_MODEL,
    base_url: str = Config.DEFAULT_BASE_URL,
    top_context_chunks: int | None = Config.DEFAULT_TOP_CONTEXT_CHUNKS,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    timeout_seconds: float = Config.DEFAULT_TIMEOUT_SECONDS,
    concurrency: int = Config.DEFAULT_CONCURRENCY,
    checkpoint_interval: int = Config.DEFAULT_CHECKPOINT_INTERVAL,
) -> Path:
    """Process a search-results dataset and write generated answers to a file.

    This function reads a search-results JSON file (as produced by the
    retrieval step), generates answers for each question concurrently,
    and writes the outputs to `output_dir` using the same filename.
    Progress is periodically checkpointed to avoid redoing work.

    Args:
        student_search_results_path: Path to the JSON file containing
            search results. The file should contain a top-level key
            "search_results" which is a list of items with
            `question_id`, `question`, and `retrieved_sources`.
        output_dir: Directory where generated output will be written.
        model: LM model name to use for generation.
        base_url: Base API URL for the LM provider.
        top_context_chunks: Max number of retrieved chunks to include per
            question when building the context.
        max_tokens: Maximum number of tokens to generate per answer.
        timeout_seconds: Timeout (in seconds) for LM calls.
        concurrency: Number of worker threads to use for generation.
        checkpoint_interval: Number of answers between writing checkpoint
            files to disk.

    Returns:
        The `Path` to the output JSON file containing generated answers
        under the top-level key "answers".
    """
    path_dir = Path(output_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    output_path = path_dir / Path(student_search_results_path).name

    with open(student_search_results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    search_results = payload.get("search_results", [])

    answers = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                answers = existing_data.get("answers", [])
        except Exception:
            answers = []

    completed_ids = {str(ans["question_id"]) for ans in answers}
    remaining_items = [
        item for item in search_results
        if str(item.get("question_id")) not in completed_ids
    ]

    shared_lm = dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key="none",
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )

    def _worker(item: dict[str, Any]) -> dict[str, Any]:
        """Worker that generates an answer for a single search-result item.

        This function prepares the context from `retrieved_sources`,
        invokes the shared language model context to generate an answer,
        and returns a serializable dict representing a `GeneratedAnswer`.

        Args:
            item: A single element from the input search results. Expected
                to contain `question_id`, `question`, and
                `retrieved_sources`.

        Returns:
            A dict representation of the generated answer suitable for
            JSON serialization and writing into the output file.
        """

        question = item.get("question", "")
        question_id = item.get("question_id", "")
        retrieved_sources = item.get("retrieved_sources", [])

        context_pieces = []
        for src in retrieved_sources[:top_context_chunks]:
            txt = _get_source_text(src)
            if txt:
                context_pieces.append(txt)

        current_pieces = list(context_pieces)
        while current_pieces and sum(len(p) for p in current_pieces) > 7000:
            current_pieces.pop()

        context_str = "\n---\n".join(current_pieces)

        with dspy.context(lm=shared_lm):
            predictor = dspy.ChainOfThought(RAGSignature)
            try:
                prediction = predictor(
                    context=context_str, question=question)
                answer_text = prediction.answer
            except Exception as e:
                if "context window" in str(e).lower() and current_pieces:
                    try:
                        current_pieces.pop()
                        context_str = "\n---\n".join(current_pieces)
                        prediction = predictor(
                            context=context_str, question=question)
                        answer_text = prediction.answer
                    except Exception as retry_e:
                        answer_text = f"Error generating answer: {
                            str(retry_e)}"
                else:
                    answer_text = f"Error generating answer: {str(e)}"

        gen_answer = GeneratedAnswer(
            question_id=str(question_id),
            question_str=question,
            answer=answer_text,
            retrieved_sources=retrieved_sources,
        )
        return gen_answer.to_dict()

    if remaining_items:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results_generator = executor.map(_worker, remaining_items)

            for idx, ans_dict in enumerate(
                tqdm(
                    results_generator,
                    total=len(remaining_items),
                    desc="Generating answers")
            ):
                answers.append(ans_dict)
                if (idx + 1) % checkpoint_interval == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"answers": answers},
                            f,
                            indent=2,
                            ensure_ascii=False
                        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"answers": answers}, f, indent=2, ensure_ascii=False)
    return output_path
