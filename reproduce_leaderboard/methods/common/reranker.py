import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from litellm import batch_completion, completion, cost_per_token
from openai import OpenAI
from pydantic import BaseModel
import pyopenjtalk
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
    "simple": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "detailed": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    - 子音より母音の一致を優先してください
    - クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "step_by_step": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    以下の手順で判断してください。
    - 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
    - 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
    - 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
    - 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "detailed_romaji_explicit": """
    クエリ（Query）と単語一覧（Wordlist）が与えられます。
    クエリと発音が似ている順に、単語一覧を並び替えてください。
    - Query と Wordlist は、元のカタカナ表記をローマ字変換したものです
    - 子音より母音の一致を優先してください
    - クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
    出力は上位Top N件のインデックスのみ返してください。
    """,
    "nonreasoning_cot": """
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

OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_BATCH_DISCOUNT_FACTOR = 0.5
OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


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
class OpenAIBatchRerankResult:
    reranked_wordlists: list[list[str]]
    batch_id: str
    batch_status: str
    execution_time: float
    output_file_path: str | None
    error_file_path: str | None


_last_token_usage = TokenUsage()


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


def build_system_prompt(prompt_template: str = "default") -> str:
    try:
        prompt_instructions = PROMPT_INSTRUCTIONS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt_template: {prompt_template}") from exc
    return f"{prompt_instructions.strip()}\n\n{PROMPT_EXAMPLE_SUFFIX.strip()}"


def prompt_template_requires_thoughts(prompt_template: str) -> bool:
    return prompt_template == "nonreasoning_cot"


def get_rerank_response_format(*, include_thoughts: bool) -> Type[BaseModel]:
    if include_thoughts:
        return ThoughtfulRerankedWordlist
    return RerankedWordlist


def build_rerank_messages(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int,
    prompt_template: str,
    input_transform: str = "none",
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


def _token_usage_from_batch_usage(batch_usage: Any) -> TokenUsage:
    if batch_usage is None:
        return TokenUsage()

    output_details = _get_value(batch_usage, "output_tokens_details")
    reasoning_tokens = _get_value(output_details, "reasoning_tokens", 0) or 0
    input_tokens = _get_value(batch_usage, "input_tokens", 0) or _get_value(
        batch_usage, "prompt_tokens", 0
    )
    completion_tokens = _get_value(batch_usage, "output_tokens", 0) or _get_value(
        batch_usage, "completion_tokens", 0
    )
    total_tokens = _get_value(batch_usage, "total_tokens", 0) or (
        input_tokens + completion_tokens
    )
    return TokenUsage(
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
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


def build_openai_batch_requests(
    *,
    model_name: str,
    messages: list[list[dict[str, str]]],
    custom_ids: list[str],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    if len(messages) != len(custom_ids):
        raise ValueError("messages and custom_ids must have the same length")
    if not is_openai_model(model_name):
        raise ValueError(f"OpenAI batch backend does not support model: {model_name}")

    requests = []
    for custom_id, message in zip(custom_ids, messages):
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": OPENAI_BATCH_ENDPOINT,
                "body": _build_openai_chat_completion_body(
                    model_name=model_name,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                ),
            }
        )
    return requests


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_openai_client(client: Any | None = None) -> Any:
    return client if client is not None else OpenAI()


def submit_openai_batch_rerank_job(
    *,
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    positive_texts: list[list[str]],
    topn: int,
    model_name: str,
    prompt_template: str,
    input_transform: str = "none",
    response_format: Type[BaseModel],
    state_path: str,
    output_file_path: str,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    client: Any | None = None,
) -> dict[str, Any]:
    state_file_path = Path(state_path)
    request_file_path = state_file_path.with_suffix(".requests.jsonl")
    request_items: list[dict[str, Any]] = [
        {
            "custom_id": f"rerank-{index:04d}",
            "query": query,
            "candidate_words": wordlist,
            "positive_words": positive,
        }
        for index, (query, wordlist, positive) in enumerate(
            zip(query_texts, wordlist_texts, positive_texts)
        )
    ]
    messages = build_rerank_messages(
        query_texts,
        wordlist_texts,
        topn=topn,
        prompt_template=prompt_template,
        input_transform=input_transform,
    )
    custom_ids = [str(item["custom_id"]) for item in request_items]
    requests = build_openai_batch_requests(
        model_name=model_name,
        messages=messages,
        custom_ids=custom_ids,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    _write_jsonl(request_file_path, requests)

    openai_client = _get_openai_client(client)
    with open(request_file_path, "rb") as f:
        input_file = openai_client.files.create(file=f, purpose="batch")
    batch = openai_client.batches.create(
        input_file_id=input_file.id,
        endpoint=OPENAI_BATCH_ENDPOINT,
        completion_window="24h",
    )

    state = {
        "schema_version": 1,
        "backend": "openai_batch",
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "batch_id": batch.id,
        "batch_status": _get_value(batch, "status"),
        "input_file_id": input_file.id,
        "request_file_path": str(request_file_path),
        "output_file_path": output_file_path,
        "result_file_path": None,
        "error_file_path": None,
        "submitted_at": datetime.now().isoformat(),
        "parameters": {
            "topn": topn,
            "rerank_model_name": model_name,
            "rerank_reasoning_effort": reasoning_effort,
            "rerank_prompt_template": prompt_template,
            "rerank_input_transform": input_transform,
        },
        "items": request_items,
    }
    _write_json(state_file_path, state)
    return state


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


def _get_batch_execution_time(batch: Any) -> float:
    started_at = _get_value(batch, "in_progress_at") or _get_value(batch, "created_at")
    completed_at = _get_value(batch, "completed_at")
    if started_at is None or completed_at is None:
        return 0.0
    return max(float(completed_at) - float(started_at), 0.0)


def _jsonify_openai_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: _jsonify_openai_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonify_openai_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonify_openai_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonify_openai_value(vars(value))
    return value


def _summarize_batch_errors(
    error_file_path: str | None, *, limit: int = 3
) -> str | None:
    if error_file_path is None:
        return None

    rows = _read_jsonl(Path(error_file_path))
    summaries = []
    for row in rows[:limit]:
        custom_id = row.get("custom_id", "unknown")
        response = row.get("response")
        response_error = _get_value(_get_value(response, "body"), "error")
        error = row.get("error")
        message = _get_value(response_error, "message") or _get_value(error, "message")
        if message is not None:
            summaries.append(f"{custom_id}: {message}")

    if not summaries:
        return None
    return "; ".join(summaries)


def retrieve_openai_batch_rerank_job(
    *,
    state_path: str,
    response_format: Type[BaseModel],
    client: Any | None = None,
) -> OpenAIBatchRerankResult:
    state_file_path = Path(state_path)
    with open(state_file_path, encoding="utf-8") as f:
        state = json.load(f)

    openai_client = _get_openai_client(client)
    batch = openai_client.batches.retrieve(state["batch_id"])
    state["batch_status"] = _get_value(batch, "status")
    state["output_file_id"] = _get_value(batch, "output_file_id")
    state["error_file_id"] = _get_value(batch, "error_file_id")
    state["request_counts"] = _jsonify_openai_value(_get_value(batch, "request_counts"))

    if state["output_file_id"]:
        result_file_path = state_file_path.with_suffix(".results.jsonl")
        result_bytes = openai_client.files.content(state["output_file_id"]).content
        result_file_path.write_bytes(result_bytes)
        state["result_file_path"] = str(result_file_path)

    if state["error_file_id"]:
        error_file_path = state_file_path.with_suffix(".errors.jsonl")
        error_bytes = openai_client.files.content(state["error_file_id"]).content
        error_file_path.write_bytes(error_bytes)
        state["error_file_path"] = str(error_file_path)

    batch_status = state["batch_status"]
    if batch_status != "completed":
        _write_json(state_file_path, state)
        raise RuntimeError(
            "OpenAI batch is not completed yet: "
            f"batch_id={state['batch_id']} status={batch_status}"
        )

    if not state.get("result_file_path"):
        error_summary = _summarize_batch_errors(state.get("error_file_path"))
        error_context = (
            f" error_file_path={state.get('error_file_path')}"
            if state.get("error_file_path")
            else ""
        )
        if error_summary:
            error_context += f" sample_errors={error_summary}"
        raise RuntimeError(
            f"OpenAI batch completed without result file: {state['batch_id']}{error_context}"
        )

    result_rows = _read_jsonl(Path(state["result_file_path"]))
    row_by_custom_id = {row["custom_id"]: row for row in result_rows}

    reset_token_usage()
    reranked_wordlists = []
    for item in state["items"]:
        row = row_by_custom_id.get(item["custom_id"])
        if row is None:
            raise RuntimeError(
                f"Missing batch result for custom_id={item['custom_id']}"
            )
        response = row.get("response")
        if response is None:
            raise RuntimeError(
                f"Batch result missing response for custom_id={item['custom_id']}"
            )
        if response.get("status_code") != 200:
            raise RuntimeError(
                "Batch request failed for "
                f"custom_id={item['custom_id']}: {json.dumps(row, ensure_ascii=False)}"
            )
        body = response["body"]
        accumulate_token_usage(body)
        parsed = _parse_response(body, response_format)
        reranked_wordlists.append(
            _build_reranked_wordlist(
                item["candidate_words"], _extract_reranked_indices(parsed)
            )
        )

    token_usage = _token_usage_from_batch_usage(_get_value(batch, "usage"))
    if token_usage.total_tokens == 0:
        token_usage = get_last_token_usage()
    else:
        set_last_token_usage(token_usage)

    state["retrieved_at"] = datetime.now().isoformat()
    state["usage"] = asdict(token_usage)
    _write_json(state_file_path, state)

    return OpenAIBatchRerankResult(
        reranked_wordlists=reranked_wordlists,
        batch_id=state["batch_id"],
        batch_status=batch_status,
        execution_time=_get_batch_execution_time(batch),
        output_file_path=state.get("result_file_path"),
        error_file_path=state.get("error_file_path"),
    )


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
) -> list[BaseModel]:
    reset_token_usage()
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
    return parsed_responses


def rerank_by_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-4o-mini",
    reasoning_effort: str | None = None,
    prompt_template: str = "default",
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
        input_transform=input_transform,
    )
    response_format = get_rerank_response_format(
        include_thoughts=include_thoughts
        or prompt_template_requires_thoughts(prompt_template)
    )

    reranked_wordlists = []
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
            reranked_wordlists.append(
                _build_reranked_wordlist(wordlist, _extract_reranked_indices(response))
            )

        time.sleep(rerank_interval)

    return reranked_wordlists
