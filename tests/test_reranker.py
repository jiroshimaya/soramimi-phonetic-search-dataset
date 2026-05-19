import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soramimi_phonetic_search_dataset import reasoning_llm_ranking as reranker


class SampleResponse(BaseModel):
    reranked: list[int]


def _mock_completion_response(payload: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )


def test_get_structured_outputs_passes_reasoning_effort_for_gpt5(monkeypatch):
    captured_kwargs = {}

    def fake_batch_completion(**kwargs):
        captured_kwargs.update(kwargs)
        response = _mock_completion_response('{"reranked": [0]}')
        response.usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        )
        return [response]

    monkeypatch.setattr(reranker, "batch_completion", fake_batch_completion)

    results = reranker.get_structured_outputs(
        model_name="gpt-5.4",
        messages=[[{"role": "user", "content": "hello"}]],
        response_format=SampleResponse,
        reasoning_effort="medium",
    )

    assert captured_kwargs["max_completion_tokens"] == 24000
    assert captured_kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert "max_tokens" not in captured_kwargs
    assert "temperature" not in captured_kwargs
    assert results.parsed_responses == [SampleResponse(reranked=[0])]
    assert results.structured_outputs == [{"reranked": [0]}]
    assert results.token_usage == reranker.TokenUsage(
        input_tokens=10,
        completion_tokens=20,
        reasoning_tokens=7,
        total_tokens=30,
    )
    assert reranker.calculate_token_cost("gpt-5.4", results.token_usage).total_cost > 0


def test_get_structured_outputs_omits_reasoning_effort_when_unspecified(monkeypatch):
    captured_kwargs = {}

    def fake_batch_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return [_mock_completion_response('{"reranked": [1]}')]

    monkeypatch.setattr(reranker, "batch_completion", fake_batch_completion)

    results = reranker.get_structured_outputs(
        model_name="gpt-4o-mini",
        messages=[[{"role": "user", "content": "hello"}]],
        response_format=SampleResponse,
    )

    assert "reasoning_effort" not in captured_kwargs
    assert captured_kwargs["max_tokens"] == 1000
    assert captured_kwargs["temperature"] == 0.0
    assert "extra_body" not in captured_kwargs
    assert results.parsed_responses == [SampleResponse(reranked=[1])]


def test_get_structured_outputs_omits_reasoning_effort_when_none(monkeypatch):
    captured_kwargs = {}

    def fake_batch_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return [_mock_completion_response('{"reranked": [1]}')]

    monkeypatch.setattr(reranker, "batch_completion", fake_batch_completion)

    results = reranker.get_structured_outputs(
        model_name="gpt-5.4",
        messages=[[{"role": "user", "content": "hello"}]],
        response_format=SampleResponse,
        reasoning_effort="none",
    )

    assert captured_kwargs["max_completion_tokens"] == 1000
    assert "extra_body" not in captured_kwargs
    assert results.parsed_responses == [SampleResponse(reranked=[1])]


def test_get_structured_outputs_falls_back_to_single_completion(monkeypatch):
    batch_kwargs = {}
    completion_kwargs = {}

    def fake_batch_completion(**kwargs):
        batch_kwargs.update(kwargs)
        return [ValueError("empty response")]

    def fake_completion(**kwargs):
        completion_kwargs.update(kwargs)
        response = _mock_completion_response('{"reranked": [2]}')
        response.usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=22,
            total_tokens=33,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=9),
        )
        return response

    monkeypatch.setattr(reranker, "batch_completion", fake_batch_completion)
    monkeypatch.setattr(reranker, "completion", fake_completion)

    results = reranker.get_structured_outputs(
        model_name="gpt-5.4",
        messages=[[{"role": "user", "content": "hello"}]],
        response_format=SampleResponse,
        reasoning_effort="medium",
    )

    assert batch_kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert completion_kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert batch_kwargs["max_completion_tokens"] == 24000
    assert completion_kwargs["max_completion_tokens"] == 32000
    assert results.parsed_responses == [SampleResponse(reranked=[2])]
    assert results.token_usage == reranker.TokenUsage(
        input_tokens=11,
        completion_tokens=22,
        reasoning_tokens=9,
        total_tokens=33,
    )


def test_get_gpt5_max_completion_tokens_scales_with_reasoning_effort():
    assert reranker.get_gpt5_max_completion_tokens(1000, None) == 1000
    assert reranker.get_gpt5_max_completion_tokens(1000, "medium") == 24000
    assert (
        reranker.get_gpt5_max_completion_tokens(1000, "medium", is_fallback=True)
        == 32000
    )


def test_token_usage_exposes_output_tokens():
    usage = reranker.TokenUsage(
        input_tokens=10,
        completion_tokens=20,
        reasoning_tokens=7,
        total_tokens=30,
    )

    assert usage.output_tokens == 13


def test_build_system_prompt_reuses_example_suffix():
    prompt = reranker.build_system_prompt()

    assert "クエリと発音が似ている順に、単語一覧を並び替えてください。" in prompt
    assert "Example:" in prompt
    assert "Reranked: 6, 4, 5, 7, 2" in prompt


def test_build_system_prompt_accepts_overrides():
    prompt = reranker.build_system_prompt(
        prompt_instructions="Custom instructions",
        prompt_example_suffix="Custom example",
    )

    assert prompt == "Custom instructions\n\nCustom example"


def test_transform_text_for_rerank_rejects_non_default_transform():
    with pytest.raises(ValueError, match="Unknown input_transform"):
        reranker.transform_text_for_rerank("タロウ", input_transform="unknown")


def test_rank_by_reasoning_llm_accepts_prompt_instructions(monkeypatch):
    captured_messages = []

    def fake_get_structured_outputs(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return reranker.StructuredOutputsResult(
            parsed_responses=[{"reranked": [1, 0]}],
            structured_outputs=[{"reranked": [1, 0]}],
            token_usage=reranker.TokenUsage(),
        )

    monkeypatch.setattr(reranker, "get_structured_outputs", fake_get_structured_outputs)

    reranked = reranker.rank_by_reasoning_llm(
        query_texts=["アケ"],
        wordlist_texts=[["アベ", "カケイ"]],
        model_name="gpt-5.4",
        rerank_interval=0,
        prompt_instructions="以下の手順で判断してください。",
    )

    assert "以下の手順で判断してください。" in captured_messages[0][0]["content"]
    assert reranked.ranked_wordlists == [["カケイ", "アベ"]]


def test_get_rerank_response_format_uses_thoughtful_schema_when_requested():
    response_format = reranker.get_rerank_response_format(include_thoughts=True)

    assert response_format is reranker.ThoughtfulRerankedWordlist


def test_rank_by_reasoning_llm_accepts_thoughtful_structured_output(monkeypatch):
    captured_response_format = None

    def fake_get_structured_outputs(**kwargs):
        nonlocal captured_response_format
        captured_response_format = kwargs["response_format"]
        return reranker.StructuredOutputsResult(
            parsed_responses=[
                {"thoughts": ["母音列が一致", "子音差を比較"], "reranked": [1, 0]}
            ],
            structured_outputs=[
                {"thoughts": ["母音列が一致", "子音差を比較"], "reranked": [1, 0]}
            ],
            token_usage=reranker.TokenUsage(),
        )

    monkeypatch.setattr(reranker, "get_structured_outputs", fake_get_structured_outputs)

    reranked = reranker.rank_by_reasoning_llm(
        query_texts=["アケ"],
        wordlist_texts=[["アベ", "カケイ"]],
        model_name="gpt-5.4",
        include_thoughts=True,
        rerank_interval=0,
    )

    assert captured_response_format is reranker.ThoughtfulRerankedWordlist
    assert reranked.ranked_wordlists == [["カケイ", "アベ"]]
    assert reranked.result_metadata == [
        {"thoughts": ["母音列が一致", "子音差を比較"], "reranked": [1, 0]}
    ]


def test_build_openai_json_schema_response_format_uses_pydantic_schema():
    expected_schema = reranker._normalize_openai_json_schema(
        SampleResponse.model_json_schema()
    )
    assert reranker._build_openai_json_schema_response_format(SampleResponse) == {
        "type": "json_schema",
        "json_schema": {
            "name": "SampleResponse",
            "strict": True,
            "schema": expected_schema,
        },
    }


def test_normalize_openai_json_schema_adds_additional_properties_false():
    schema = {
        "type": "object",
        "properties": {
            "reranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                },
            }
        },
    }

    normalized = reranker._normalize_openai_json_schema(schema)

    assert normalized["additionalProperties"] is False
    assert (
        normalized["properties"]["reranked"]["items"]["additionalProperties"] is False
    )


def test_build_system_prompt_supports_custom_romaji_instructions():
    prompt = reranker.build_system_prompt(
        prompt_instructions=(
            "Query と Wordlist は、元のカタカナ表記をローマ字変換したものです"
        )
    )

    assert "元のカタカナ表記をローマ字変換したものです" in prompt
    assert "Example:" in prompt
