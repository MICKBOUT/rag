import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from typing import Sequence, Annotated

import dspy
from tqdm import tqdm

from models import GeneratedAnswer
from retrieval import search
from config import Config


class RAGSignature(dspy.Signature):
    """
    Answer the question concisely and accurately
    using only the provided context chunks.
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

    retrieved_sources = [res.to_source_dict() for res in results]
    return GeneratedAnswer(
        question_id="single_query",
        question_str=question,
        answer=prediction.answer,
        retrieved_sources=retrieved_sources,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        search_k=search_k,
        top_context_chunks=top_context_chunks,
    )


def answer_dataset_to_file(
    student_search_results_path: str | Path,
    *,
    output_dir: str | Path = Config.DEFAULT_OUTPUT_DIR,
    model: str = Config.DEFAULT_MODEL,
    base_url: str = Config.DEFAULT_BASE_URL,
    top_context_chunks: int | None = Config.DEFAULT_TOP_CONTEXT_CHUNKS,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    timeout_seconds: float = Config.DEFAULT_TIMEOUT_SECONDS,
    concurrency: int = Config.DEFAULT_CONCURRENCY,
    checkpoint_interval: int = Config.DEFAULT_CHECKPOINT_INTERVAL,
) -> Path:
    """
    Processes an entire search result dataset concurrently,
    running generation and tracking progress via checkpoints.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(student_search_results_path).name

    with open(student_search_results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    search_results = payload.get("search_results", [])
    search_k = payload.get("k", Config.DEFAULT_SEARCH_K)

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
        question_str = item.get("question_str", "")
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
                    context=context_str, question=question_str)
                answer_text = prediction.answer
            except Exception as e:
                if "context window" in str(e).lower() and current_pieces:
                    try:
                        current_pieces.pop()
                        context_str = "\n---\n".join(current_pieces)
                        prediction = predictor(context=context_str, question=question_str)
                        answer_text = prediction.answer
                    except Exception as retry_e:
                        answer_text = f"Error generating answer: {str(retry_e)}"
                else:
                    answer_text = f"Error generating answer: {str(e)}"

        gen_answer = GeneratedAnswer(
            question_id=str(question_id),
            question_str=question_str,
            answer=answer_text,
            retrieved_sources=retrieved_sources,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            search_k=search_k,
            top_context_chunks=top_context_chunks,
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