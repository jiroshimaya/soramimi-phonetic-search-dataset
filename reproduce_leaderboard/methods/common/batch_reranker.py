import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel
from reranker import (
    OPENAI_BATCH_DISCOUNT_FACTOR,
    calculate_token_cost,
    get_last_token_usage,
    get_rerank_response_format,
    retrieve_openai_batch_rerank_job,
    submit_openai_batch_rerank_job,
)
from soramimi_phonetic_search_dataset.evaluate import calculate_recall
from soramimi_phonetic_search_dataset.schemas import (
    PhoneticSearchMetrics,
    PhoneticSearchParameters,
    PhoneticSearchResult,
    PhoneticSearchResults,
)


def prepare_rerank_candidates(
    base_ranked_wordlists: list[list[str]],
    positive_texts: list[list[str]],
    rerank_input_size: int,
) -> list[list[str]]:
    topk_ranked_wordlists = []
    for wordlist, positive_text in zip(base_ranked_wordlists, positive_texts):
        topk = wordlist[:rerank_input_size]
        missing_positive_count = sum(1 for text in positive_text if text not in topk)
        if missing_positive_count > 0:
            topk = topk[:-missing_positive_count]
            for text in positive_text:
                if text not in topk:
                    topk.append(text)
        topk_ranked_wordlists.append(sorted(topk))
    return topk_ranked_wordlists


def get_default_batch_state_path(output_path: str) -> str:
    output_path_lib = Path(output_path)
    return str(
        output_path_lib.with_name(f"{output_path_lib.stem}_openai_batch_state.json")
    )


def _build_results_from_ranked_wordlists(
    query_texts: list[str],
    positive_texts: list[list[str]],
    ranked_wordlists: list[list[str]],
    structured_outputs: list[dict[str, Any]] | None = None,
    *,
    topn: int,
    execution_time: float,
) -> PhoneticSearchResults:
    recall = calculate_recall(ranked_wordlists, positive_texts, topn=topn)
    if structured_outputs is None:
        structured_outputs = [{} for _ in ranked_wordlists]
    results = [
        PhoneticSearchResult(
            query=query,
            ranked_words=wordlist[:topn],
            positive_words=positive_text,
            thoughts=structured_output.get("thoughts"),
        )
        for query, wordlist, positive_text, structured_output in zip(
            query_texts,
            ranked_wordlists,
            positive_texts,
            structured_outputs,
        )
    ]
    return PhoneticSearchResults(
        parameters=PhoneticSearchParameters(
            topn=topn,
            rank_func="unknown",
            execution_timestamp=datetime.now().isoformat(),
        ),
        metrics=PhoneticSearchMetrics(
            recall=recall,
            execution_time=execution_time,
        ),
        results=results,
    )


def submit_openai_batch_evaluation(
    *,
    base_rank_func: Callable[..., list[list[str]]],
    query_texts: list[str],
    word_texts: list[str],
    positive_texts: list[list[str]],
    rank_kwargs: dict[str, Any],
    rerank_input_size: int,
    topn: int,
    model_name: str,
    prompt_template: str,
    response_format: type[BaseModel] | None = None,
    input_transform: str = "none",
    state_path: str,
    output_file_path: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    base_ranked_wordlists = base_rank_func(query_texts, word_texts, **rank_kwargs)
    topk_ranked_wordlists = prepare_rerank_candidates(
        base_ranked_wordlists,
        positive_texts,
        rerank_input_size,
    )
    if response_format is None:
        response_format = get_rerank_response_format(include_thoughts=False)
    return submit_openai_batch_rerank_job(
        query_texts=query_texts,
        wordlist_texts=topk_ranked_wordlists,
        positive_texts=positive_texts,
        topn=topn,
        model_name=model_name,
        prompt_template=prompt_template,
        input_transform=input_transform,
        response_format=response_format,
        state_path=state_path,
        output_file_path=output_file_path,
        reasoning_effort=reasoning_effort,
    )


def retrieve_openai_batch_evaluation_results(
    *,
    state_path: str,
    query_texts: list[str],
    positive_texts: list[list[str]],
    response_format: type[BaseModel] | None = None,
    rank_func: str,
    vowel_ratio: float,
    topn: int,
    rerank_input_size: int,
    model_name: str,
    reasoning_effort: str | None,
    prompt_template: str,
    rerank_include_thoughts: bool = False,
    input_transform: str = "none",
    backend: str,
) -> PhoneticSearchResults:
    if response_format is None:
        response_format = get_rerank_response_format(include_thoughts=False)
    retrieved = retrieve_openai_batch_rerank_job(
        state_path=state_path,
        response_format=response_format,
    )
    with open(state_path, encoding="utf-8") as f:
        batch_state = json.load(f)

    results = _build_results_from_ranked_wordlists(
        query_texts=query_texts,
        positive_texts=positive_texts,
        ranked_wordlists=retrieved.reranked_wordlists,
        structured_outputs=retrieved.structured_outputs,
        topn=topn,
        execution_time=retrieved.execution_time,
    )
    results.parameters.rank_func = rank_func
    results.parameters.vowel_ratio = (
        vowel_ratio if rank_func in ["kanasim", "vowel_consonant"] else None
    )
    results.parameters.rerank = True
    results.parameters.rerank_model_name = model_name
    results.parameters.rerank_reasoning_effort = reasoning_effort
    results.parameters.rerank_prompt_template = prompt_template
    results.parameters.rerank_include_thoughts = rerank_include_thoughts
    results.parameters.rerank_input_transform = input_transform
    results.parameters.rerank_input_size = rerank_input_size
    results.parameters.rerank_backend = backend
    results.parameters.rerank_batch_id = batch_state["batch_id"]

    token_usage = get_last_token_usage()
    token_cost = calculate_token_cost(
        model_name,
        token_usage,
        discount_factor=OPENAI_BATCH_DISCOUNT_FACTOR,
    )
    results.metrics.rerank_input_tokens = token_usage.input_tokens
    results.metrics.rerank_output_tokens = token_usage.output_tokens
    results.metrics.rerank_reasoning_tokens = token_usage.reasoning_tokens
    results.metrics.rerank_total_tokens = token_usage.total_tokens
    results.metrics.rerank_input_cost = token_cost.input_cost
    results.metrics.rerank_output_cost = token_cost.output_cost
    results.metrics.rerank_reasoning_cost = token_cost.reasoning_cost
    results.metrics.rerank_total_cost = token_cost.total_cost
    return results
