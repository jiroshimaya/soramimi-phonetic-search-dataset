import time
from typing import Any, Type

from litellm import batch_completion, completion, cost_per_token
from pydantic import BaseModel
import pyopenjtalk
from soramimi_phonetic_search_dataset import reasoning_llm_ranking as _core_reasoning
from tqdm import tqdm

from rerank_prompts import (
    DEFAULT_USER_PROMPT_TEMPLATE,
    RerankPromptConfig,
    get_prompt_config,
)

OPENAI_MODEL_PREFIXES = _core_reasoning.OPENAI_MODEL_PREFIXES
TokenUsage = _core_reasoning.TokenUsage
TokenCost = _core_reasoning.TokenCost
RerankedWordlist = _core_reasoning.RerankedWordlist
ThoughtfulRerankedWordlist = _core_reasoning.ThoughtfulRerankedWordlist

_last_token_usage = TokenUsage()
_last_structured_outputs: list[dict[str, Any]] = []


def transform_text_for_rerank(text: str, input_transform: str = "none") -> str:
    if input_transform == "none":
        return text
    if input_transform == "pyopenjtalk_romaji":
        phonemes = pyopenjtalk.g2p(text)
        phoneme_text = phonemes if isinstance(phonemes, str) else " ".join(phonemes)
        return " ".join(phoneme_text.lower().split())
    if input_transform == "kana_and_pyopenjtalk_romaji":
        romaji = transform_text_for_rerank(text, "pyopenjtalk_romaji")
        return f"{text}（{romaji}）"
    raise ValueError(f"Unknown input_transform: {input_transform}")


def _resolve_prompt_config(
    prompt_template: str = "default",
    *,
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
) -> RerankPromptConfig:
    prompt_config = get_prompt_config(prompt_template)
    return RerankPromptConfig(
        prompt_instructions=prompt_instructions or prompt_config.prompt_instructions,
        prompt_example_suffix=(
            prompt_example_suffix or prompt_config.prompt_example_suffix
        ),
        user_prompt_template=user_prompt_template or prompt_config.user_prompt_template,
        requires_thoughts=prompt_config.requires_thoughts,
    )


def build_system_prompt(
    prompt_template: str = "default",
    *,
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
) -> str:
    prompt_config = _resolve_prompt_config(
        prompt_template,
        prompt_instructions=prompt_instructions,
        prompt_example_suffix=prompt_example_suffix,
    )
    return (
        f"{prompt_config.prompt_instructions.strip()}\n\n"
        f"{prompt_config.prompt_example_suffix.strip()}"
    )


def prompt_template_requires_thoughts(prompt_template: str) -> bool:
    return get_prompt_config(prompt_template).requires_thoughts


def get_rerank_response_format(*, include_thoughts: bool) -> Type[BaseModel]:
    if include_thoughts:
        return ThoughtfulRerankedWordlist
    return RerankedWordlist


def build_rerank_messages(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int,
    prompt_template: str = "default",
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
    input_transform: str = "none",
) -> list[list[dict[str, str]]]:
    prompt_config = _resolve_prompt_config(
        prompt_template,
        prompt_instructions=prompt_instructions,
        prompt_example_suffix=prompt_example_suffix,
        user_prompt_template=user_prompt_template,
    )
    prompt = build_system_prompt(
        prompt_template,
        prompt_instructions=prompt_config.prompt_instructions,
        prompt_example_suffix=prompt_config.prompt_example_suffix,
    )
    user_prompt = prompt_config.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE

    messages = []
    for query, wordlist in zip(query_texts, wordlist_texts):
        transformed_query = transform_text_for_rerank(query, input_transform)
        transformed_wordlist = [
            transform_text_for_rerank(word, input_transform) for word in wordlist
        ]
        wordlist_str = "\n".join(
            [f"{i}. {word}" for i, word in enumerate(transformed_wordlist)]
        )
        messages.append(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": user_prompt.format(
                        query=transformed_query, wordlist=wordlist_str, topn=topn
                    ),
                },
            ]
        )
    return messages


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


def _normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    return None if reasoning_effort in (None, "none") else reasoning_effort


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


def get_gpt5_max_completion_tokens(
    max_tokens: int,
    reasoning_effort: str | None,
    *,
    is_fallback: bool = False,
) -> int:
    if reasoning_effort == "medium":
        return max(max_tokens, 32000 if is_fallback else 24000)
    if reasoning_effort == "high":
        return max(max_tokens, 40000 if is_fallback else 32000)
    return max(max_tokens, 4000) if is_fallback else max_tokens


def _build_litellm_completion_kwargs(
    model_name: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
    *,
    is_fallback: bool = False,
) -> dict[str, Any]:
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    is_gpt5 = model_name.startswith("gpt-5")

    completion_kwargs: dict[str, Any] = {}
    if is_gpt5:
        completion_kwargs["max_completion_tokens"] = get_gpt5_max_completion_tokens(
            max_tokens,
            normalized_reasoning_effort,
            is_fallback=is_fallback,
        )
        if normalized_reasoning_effort is not None:
            completion_kwargs["extra_body"] = {
                "reasoning_effort": normalized_reasoning_effort
            }
    else:
        completion_kwargs["temperature"] = temperature
        completion_kwargs["max_tokens"] = max_tokens
        if normalized_reasoning_effort is not None:
            completion_kwargs["reasoning_effort"] = normalized_reasoning_effort
    return completion_kwargs


def is_openai_model(model_name: str) -> bool:
    return model_name.startswith(OPENAI_MODEL_PREFIXES)


def _build_openai_chat_completion_body(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
    response_format: Type[BaseModel] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    if model_name.startswith("gpt-5"):
        body["max_completion_tokens"] = get_gpt5_max_completion_tokens(
            max_tokens,
            normalized_reasoning_effort,
        )
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
    if normalized_reasoning_effort is not None:
        body["reasoning_effort"] = normalized_reasoning_effort
    if response_format is not None:
        body["response_format"] = _build_openai_json_schema_response_format(
            response_format
        )
    return body


def _build_openai_json_schema_response_format(
    response_format: Type[BaseModel],
) -> dict[str, Any]:
    schema = _normalize_openai_json_schema(response_format.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def _normalize_openai_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            normalized[key] = _normalize_openai_json_schema(value)
        elif isinstance(value, list):
            normalized[key] = [
                _normalize_openai_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            normalized[key] = value

    if normalized.get("type") == "object" and "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False

    return normalized


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


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
) -> list[BaseModel]:
    reset_token_usage()
    reset_last_structured_outputs()
    completion_kwargs = _build_litellm_completion_kwargs(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

    raw_responses = batch_completion(
        model=model_name,
        messages=messages,
        response_format=response_format,
        **completion_kwargs,
    )

    parsed_responses = []
    for message, response in zip(messages, raw_responses):
        try:
            accumulate_token_usage(response)
            parsed_responses.append(_parse_response(response, response_format))
        except (TypeError, ValueError):
            fallback_kwargs = _build_litellm_completion_kwargs(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                is_fallback=True,
            )
            fallback_response = completion(
                model=model_name,
                messages=message,
                response_format=response_format,
                **fallback_kwargs,
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
    reasoning_effort: str | None = None,
    prompt_template: str = "default",
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
    include_thoughts: bool = False,
    input_transform: str = "none",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> list[list[str]]:
    messages = build_rerank_messages(
        query_texts,
        wordlist_texts,
        topn=topn,
        prompt_template=prompt_template,
        prompt_instructions=prompt_instructions,
        prompt_example_suffix=prompt_example_suffix,
        user_prompt_template=user_prompt_template,
        input_transform=input_transform,
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
            reasoning_effort=reasoning_effort,
        )
        for wordlist, response in zip(wordlist_texts[i : i + batch_size], responses):
            structured_outputs.append(_extract_structured_output(response))
            reranked_wordlists.append(
                _build_reranked_wordlist(wordlist, _extract_reranked_indices(response))
            )

        time.sleep(rerank_interval)

    set_last_structured_outputs(structured_outputs)
    return reranked_wordlists
