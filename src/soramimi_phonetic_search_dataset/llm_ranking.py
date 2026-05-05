import time
from dataclasses import dataclass
from typing import Any, Type

from litellm import batch_completion, completion, cost_per_token
from pydantic import BaseModel
from tqdm import tqdm

PROMPT_INSTRUCTIONS = {
    "default": """
    You are a phonetic search assistant.
    You are given a query and a list of words.
    You need to rerank the words based on phonetic similarity to the query.
    When estimating phonetic similarity, please consider the following:
    1. Prioritize matching vowels
    2. Substitution, insertion, or deletion of nasal sounds, geminate consonants, and long vowels is acceptable
    3. For other cases, words with similar mora counts are preferred
    You need to return only the reranked list of index numbers of the words, no other text.
    You need to return only topn index numbers.
    """,
}
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


_last_token_usage = TokenUsage()
_last_structured_outputs: list[dict[str, Any]] = []


class RerankedWordlist(BaseModel):
    reranked: list[int]


class ThoughtfulRerankedWordlist(BaseModel):
    thoughts: list[str]
    reranked: list[int]


def get_rerank_response_format(*, include_thoughts: bool) -> Type[BaseModel]:
    if include_thoughts:
        return ThoughtfulRerankedWordlist
    return RerankedWordlist


def reset_token_usage() -> None:
    global _last_token_usage
    _last_token_usage = TokenUsage()


def reset_last_structured_outputs() -> None:
    global _last_structured_outputs
    _last_structured_outputs = []


def set_last_structured_outputs(outputs: list[dict[str, Any]]) -> None:
    global _last_structured_outputs
    _last_structured_outputs = [dict(output) for output in outputs]


def get_last_structured_outputs() -> list[dict[str, Any]]:
    return [dict(output) for output in _last_structured_outputs]


def set_last_token_usage(token_usage: TokenUsage) -> None:
    global _last_token_usage
    _last_token_usage = TokenUsage(
        input_tokens=token_usage.input_tokens,
        completion_tokens=token_usage.completion_tokens,
        reasoning_tokens=token_usage.reasoning_tokens,
        total_tokens=token_usage.total_tokens,
    )


def get_last_token_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=_last_token_usage.input_tokens,
        completion_tokens=_last_token_usage.completion_tokens,
        reasoning_tokens=_last_token_usage.reasoning_tokens,
        total_tokens=_last_token_usage.total_tokens,
    )


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


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def accumulate_token_usage(response: Any) -> None:
    usage = _get_value(response, "usage")
    if usage is None:
        return

    completion_details = _get_value(usage, "completion_tokens_details")
    if completion_details is None:
        completion_details = _get_value(usage, "output_tokens_details")
    reasoning_tokens = _get_value(completion_details, "reasoning_tokens", 0) or 0

    _last_token_usage.input_tokens += _get_value(
        usage, "prompt_tokens", 0
    ) or _get_value(usage, "input_tokens", 0)
    _last_token_usage.completion_tokens += _get_value(
        usage, "completion_tokens", 0
    ) or _get_value(usage, "output_tokens", 0)
    _last_token_usage.reasoning_tokens += reasoning_tokens
    _last_token_usage.total_tokens += _get_value(usage, "total_tokens", 0) or (
        _last_token_usage.input_tokens + _last_token_usage.completion_tokens
    )


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


def _parse_response(response: Any, response_format: Type[BaseModel]) -> BaseModel:
    return response_format.model_validate_json(_extract_response_content(response))


def _extract_reranked_indices(response: BaseModel | dict[str, Any]) -> list[int]:
    response_dict = (
        response.model_dump() if isinstance(response, BaseModel) else response
    )
    reranked = response_dict.get("reranked")
    if not isinstance(reranked, list):
        raise TypeError(f"Unexpected reranked payload: {response!r}")
    return [int(index) for index in reranked]


def _extract_structured_output(
    response: BaseModel | dict[str, Any],
) -> dict[str, Any]:
    return response.model_dump() if isinstance(response, BaseModel) else dict(response)


def _build_reranked_wordlist(
    wordlist: list[str], reranked_indices: list[int]
) -> list[str]:
    reranked_wordlist = []
    for index in reranked_indices:
        if 0 <= index < len(wordlist):
            reranked_wordlist.append(wordlist[index])
        else:
            reranked_wordlist.append("NA")
    return reranked_wordlist


def build_system_prompt(prompt_template: str = "default") -> str:
    try:
        prompt_instructions = PROMPT_INSTRUCTIONS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt_template: {prompt_template}") from exc
    return f"{prompt_instructions.strip()}\n\n{PROMPT_EXAMPLE_SUFFIX.strip()}"


def prompt_template_requires_thoughts(prompt_template: str) -> bool:
    return False


def build_rerank_messages(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int,
    prompt_template: str,
) -> list[list[dict[str, str]]]:
    prompt = build_system_prompt(prompt_template)
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
) -> list[BaseModel]:
    reset_token_usage()
    reset_last_structured_outputs()

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
            accumulate_token_usage(response)
            parsed_responses.append(_parse_response(response, response_format))
        except (TypeError, ValueError):
            fallback_response = completion(
                model=model_name,
                messages=message,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            accumulate_token_usage(fallback_response)
            parsed_responses.append(_parse_response(fallback_response, response_format))

    set_last_structured_outputs(
        [_extract_structured_output(response) for response in parsed_responses]
    )
    return parsed_responses


def rank_by_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-4o-mini",
    prompt_template: str = "default",
    include_thoughts: bool = False,
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> list[list[str]]:
    messages = build_rerank_messages(
        query_texts,
        wordlist_texts,
        topn=topn,
        prompt_template=prompt_template,
    )
    response_format = get_rerank_response_format(
        include_thoughts=include_thoughts
        or prompt_template_requires_thoughts(prompt_template)
    )

    reranked_wordlists = []
    structured_outputs = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i : i + batch_size]
        responses = get_structured_outputs(
            model_name=model_name,
            messages=batch_messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=response_format,
        )
        for wordlist, response in zip(wordlist_texts[i : i + batch_size], responses):
            structured_outputs.append(_extract_structured_output(response))
            reranked_wordlists.append(
                _build_reranked_wordlist(wordlist, _extract_reranked_indices(response))
            )

        time.sleep(rerank_interval)

    set_last_structured_outputs(structured_outputs)
    return reranked_wordlists