import time
from dataclasses import asdict, dataclass
from typing import Any, Type

from litellm import batch_completion, completion, cost_per_token
from pydantic import BaseModel
import pyopenjtalk
from tqdm import tqdm

from soramimi_phonetic_search_dataset.evaluate import RankingFunctionOutput

PROMPT_INSTRUCTIONS = """
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
出力は上位Top N件のインデックスのみ返してください。
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
USER_PROMPT_TEMPLATE = """
Query: {query}
Wordlist:
{wordlist}
Top N: {topn}
Reranked:
"""
DEFAULT_PROMPT_EXAMPLE_SUFFIX = PROMPT_EXAMPLE_SUFFIX
DEFAULT_USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE

OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


@dataclass(frozen=True)
class RerankPromptConfig:
    prompt_instructions: str
    prompt_example_suffix: str = DEFAULT_PROMPT_EXAMPLE_SUFFIX
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    requires_thoughts: bool = False


PROMPT_CONFIGS = {
    "default": RerankPromptConfig(
        prompt_instructions="""
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
    ),
    "simple": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "detailed": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- 子音より母音の一致を優先してください
- クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "step_by_step": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
以下の手順で判断してください。
- 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
- 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
- 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
- 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "detailed_romaji_explicit": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- Query と Wordlist は、元のカタカナ表記をローマ字変換したものです
- 子音より母音の一致を優先してください
- クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
出力は上位Top N件のインデックスのみ返してください。
""",
    ),
    "nonreasoning_cot": RerankPromptConfig(
        prompt_instructions="""
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
以下の手順で判断してください。
- 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
- 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
- 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
- 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
構造化出力の thoughts フィールドには、最終順位に効いた判断要点だけを短い箇条書きで入れてください。
構造化出力の reranked フィールドには、上位Top N件のインデックスのみを入れてください。
""",
        requires_thoughts=True,
    ),
}


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


class ThoughtfulRerankedWordlist(BaseModel):
    thoughts: list[str]
    reranked: list[int]


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


def get_prompt_config(prompt_template: str = "default") -> RerankPromptConfig:
    try:
        return PROMPT_CONFIGS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt_template: {prompt_template}") from exc


def _resolve_prompt_config(
    prompt_template: str | None = None,
    *,
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
) -> RerankPromptConfig:
    if prompt_template is None:
        return RerankPromptConfig(
            prompt_instructions=prompt_instructions or PROMPT_INSTRUCTIONS,
            prompt_example_suffix=prompt_example_suffix or PROMPT_EXAMPLE_SUFFIX,
            user_prompt_template=user_prompt_template or USER_PROMPT_TEMPLATE,
        )

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
    prompt_template: str | None = None,
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


def prompt_template_requires_thoughts(prompt_template: str | None) -> bool:
    if prompt_template is None:
        return False
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
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
    prompt_template: str | None = None,
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


def accumulate_token_usage(response: Any, token_usage: TokenUsage) -> None:
    usage = _get_value(response, "usage")
    if usage is None:
        return

    completion_details = _get_value(usage, "completion_tokens_details")
    if completion_details is None:
        completion_details = _get_value(usage, "output_tokens_details")
    reasoning_tokens = _get_value(completion_details, "reasoning_tokens", 0) or 0

    token_usage.input_tokens += _get_value(usage, "prompt_tokens", 0) or _get_value(
        usage, "input_tokens", 0
    )
    token_usage.completion_tokens += _get_value(
        usage, "completion_tokens", 0
    ) or _get_value(usage, "output_tokens", 0)
    token_usage.reasoning_tokens += reasoning_tokens
    token_usage.total_tokens += _get_value(usage, "total_tokens", 0) or (
        token_usage.input_tokens + token_usage.completion_tokens
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


def _merge_token_usage(total: TokenUsage, partial: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=total.input_tokens + partial.input_tokens,
        completion_tokens=total.completion_tokens + partial.completion_tokens,
        reasoning_tokens=total.reasoning_tokens + partial.reasoning_tokens,
        total_tokens=total.total_tokens + partial.total_tokens,
    )


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
) -> StructuredOutputsResult:
    token_usage = TokenUsage()
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
            accumulate_token_usage(response, token_usage)
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
            accumulate_token_usage(fallback_response, token_usage)
            parsed_responses.append(_parse_response(fallback_response, response_format))
    structured_outputs = [
        _extract_structured_output(response) for response in parsed_responses
    ]
    return StructuredOutputsResult(
        parsed_responses=parsed_responses,
        structured_outputs=structured_outputs,
        token_usage=token_usage,
    )


def rank_by_reasoning_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-5.1-mini",
    reasoning_effort: str | None = None,
    prompt_template: str | None = None,
    prompt_instructions: str | None = None,
    prompt_example_suffix: str | None = None,
    user_prompt_template: str | None = None,
    include_thoughts: bool = False,
    input_transform: str = "none",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> RankingFunctionOutput:
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
    total_token_usage = TokenUsage()
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i : i + batch_size]
        batch_result = get_structured_outputs(
            model_name=model_name,
            messages=batch_messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        structured_outputs.extend(batch_result.structured_outputs)
        total_token_usage = _merge_token_usage(
            total_token_usage, batch_result.token_usage
        )
        for wordlist, response in zip(
            wordlist_texts[i : i + batch_size], batch_result.parsed_responses
        ):
            reranked_wordlists.append(
                _build_reranked_wordlist(wordlist, _extract_reranked_indices(response))
            )

        time.sleep(rerank_interval)

    token_cost = calculate_token_cost(model_name, total_token_usage)
    return RankingFunctionOutput(
        ranked_wordlists=reranked_wordlists,
        result_metadata=structured_outputs,
        metrics_metadata={
            "model_name": model_name,
            "token_usage": {
                "input_tokens": total_token_usage.input_tokens,
                "output_tokens": total_token_usage.output_tokens,
                "reasoning_tokens": total_token_usage.reasoning_tokens,
                "total_tokens": total_token_usage.total_tokens,
            },
            "cost": asdict(token_cost),
        },
    )
