import time
from dataclasses import dataclass
from typing import Any, Type

from litellm import batch_completion, completion, cost_per_token
from pydantic import BaseModel
from soramimi_phonetic_search_dataset.evaluate import RankingFunctionOutput
from tqdm import tqdm

PROMPT_INSTRUCTIONS = """
You are a phonetic search assistant.
You are given a query and a list of words.
You need to rerank the words based on phonetic similarity to the query.
When estimating phonetic similarity, please consider the following:
1. Prioritize matching vowels
2. Substitution, insertion, or deletion of nasal sounds, geminate consonants, and long vowels is acceptable
3. For other cases, words with similar mora counts are preferred
You need to return only the reranked list of index numbers of the words, no other text.
You need to return only topn index numbers.
"""
PROMPT_EXAMPLE_SUFFIX = """
Example:
Query: タロウ
Wordlist:
0. アオ
1. アオウヅ
2. アノウ
3. タキョウ
4. タド
5. タノ
6. タロウ
7. タンノ
Top N: 5
Reranked: 6, 4, 5, 7, 2
"""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @property
    def output_tokens(self) -> int:
        return max(self.completion_tokens - self.reasoning_tokens, 0)


@dataclass
class TokenCost:
    input_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class StructuredOutputsResult:
    parsed_responses: list[BaseModel | dict[str, Any]]
    structured_outputs: list[dict[str, Any]]
    token_usage: TokenUsage


class RerankedWordlist(BaseModel):
    reranked: list[int]


def calculate_token_cost(
    model_name: str,
    token_usage: TokenUsage,
    *,
    discount_factor: float = 1.0,
) -> TokenCost:
    input_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=token_usage.input_tokens,
        completion_tokens=token_usage.completion_tokens,
    )
    input_cost *= discount_factor
    completion_cost *= discount_factor
    if token_usage.completion_tokens == 0:
        reasoning_cost = 0.0
    else:
        reasoning_cost = completion_cost * (
            token_usage.reasoning_tokens / token_usage.completion_tokens
        )
    output_cost = completion_cost - reasoning_cost
    return TokenCost(
        input_cost=input_cost,
        output_cost=output_cost,
        reasoning_cost=reasoning_cost,
        total_cost=input_cost + completion_cost,
    )


def build_rerank_metrics_metadata(
    *,
    model_name: str,
    token_usage: TokenUsage,
    token_cost: TokenCost,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "token_usage": {
            "input_tokens": token_usage.input_tokens,
            "output_tokens": token_usage.output_tokens,
            "reasoning_tokens": token_usage.reasoning_tokens,
            "total_tokens": token_usage.total_tokens,
        },
        "cost": {
            "input_cost": token_cost.input_cost,
            "output_cost": token_cost.output_cost,
            "reasoning_cost": token_cost.reasoning_cost,
            "total_cost": token_cost.total_cost,
        },
    }


def merge_token_usage(target: TokenUsage, source: TokenUsage) -> None:
    target.input_tokens += source.input_tokens
    target.completion_tokens += source.completion_tokens
    target.reasoning_tokens += source.reasoning_tokens
    target.total_tokens += source.total_tokens


def accumulate_token_usage(response: Any, token_usage: TokenUsage) -> None:
    usage = _get_value(response, "usage")
    if usage is None:
        return

    completion_details = _get_value(usage, "completion_tokens_details")
    if completion_details is None:
        completion_details = _get_value(usage, "output_tokens_details")
    reasoning_tokens = _get_value(completion_details, "reasoning_tokens", 0) or 0

    input_tokens = _get_value(usage, "prompt_tokens", 0) or _get_value(
        usage, "input_tokens", 0
    )
    completion_tokens = _get_value(usage, "completion_tokens", 0) or _get_value(
        usage, "output_tokens", 0
    )
    total_tokens = _get_value(usage, "total_tokens", 0) or (
        input_tokens + completion_tokens
    )

    token_usage.input_tokens += input_tokens
    token_usage.completion_tokens += completion_tokens
    token_usage.reasoning_tokens += reasoning_tokens
    token_usage.total_tokens += total_tokens

def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _extract_response_content(response: Any) -> str:
    choices = _get_value(response, "choices")
    if not choices:
        raise TypeError(f"Unexpected response without choices: {response!r}")

    first_choice = choices[0]
    message = _get_value(first_choice, "message")
    if message is None:
        raise TypeError(f"Unexpected response without message: {response!r}")

    content = _get_value(message, "content")
    if not content:
        raise ValueError(f"Empty content: {response!r}")
    if isinstance(content, list):
        raise TypeError(f"Unsupported content format: {response!r}")
    return content


def _extract_structured_output(
    response: BaseModel | dict[str, Any],
) -> dict[str, Any]:
    return response.model_dump() if isinstance(response, BaseModel) else dict(response)


def build_system_prompt() -> str:
    return f"{PROMPT_INSTRUCTIONS.strip()}\n\n{PROMPT_EXAMPLE_SUFFIX.strip()}"


def build_rerank_messages(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int,
) -> list[list[dict[str, str]]]:
    prompt = build_system_prompt()
    user_prompt = """
    Query: {query}
    Wordlist:
    {wordlist}
    Top N: {topn}
    Reranked:
    """

    messages = []
    for query, wordlist in zip(query_texts, wordlist_texts):
        wordlist_str = "\n".join(
            [f"{i}. {word}" for i, word in enumerate(wordlist)]
        )
        messages.append(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": user_prompt.format(
                        query=query, wordlist=wordlist_str, topn=topn
                    ),
                },
            ]
        )
    return messages


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> StructuredOutputsResult:
    token_usage = TokenUsage()

    raw_responses = batch_completion(
        model=model_name,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    parsed_responses = []
    for message, response in zip(messages, raw_responses):
        try:
            accumulate_token_usage(response, token_usage)
            parsed_responses.append(
                response_format.model_validate_json(_extract_response_content(response))
            )
        except (TypeError, ValueError):
            fallback_response = completion(
                model=model_name,
                messages=message,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            accumulate_token_usage(fallback_response, token_usage)
            parsed_responses.append(
                response_format.model_validate_json(
                    _extract_response_content(fallback_response)
                )
            )

    structured_outputs = [
        _extract_structured_output(response) for response in parsed_responses
    ]
    return StructuredOutputsResult(
        parsed_responses=parsed_responses,
        structured_outputs=structured_outputs,
        token_usage=token_usage,
    )


def rank_by_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-4o-mini",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> RankingFunctionOutput:
    messages = build_rerank_messages(
        query_texts,
        wordlist_texts,
        topn=topn,
    )

    reranked_wordlists = []
    structured_outputs = []
    total_token_usage = TokenUsage()
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i : i + batch_size]
        batch_result = get_structured_outputs(
            model_name=model_name,
            messages=batch_messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=RerankedWordlist,
        )
        merge_token_usage(total_token_usage, batch_result.token_usage)
        structured_outputs.extend(batch_result.structured_outputs)
        for wordlist, response in zip(
            wordlist_texts[i : i + batch_size], batch_result.parsed_responses
        ):
            response_dict = _extract_structured_output(response)

            reranked = response_dict.get("reranked")
            if not isinstance(reranked, list):
                raise TypeError(f"Unexpected reranked payload: {response!r}")

            reranked_wordlists.append(
                [
                    wordlist[int(index)] if 0 <= int(index) < len(wordlist) else "NA"
                    for index in reranked
                ]
            )

        time.sleep(rerank_interval)

    token_cost = calculate_token_cost(model_name, total_token_usage)
    return RankingFunctionOutput(
        ranked_wordlists=reranked_wordlists,
        metrics_metadata=build_rerank_metrics_metadata(
            model_name=model_name,
            token_usage=total_token_usage,
            token_cost=token_cost,
        ),
    )