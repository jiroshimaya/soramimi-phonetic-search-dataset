import sys
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduce_leaderboard.methods.common import reranker


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

    assert captured_kwargs["max_completion_tokens"] == 16000
    assert captured_kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert "max_tokens" not in captured_kwargs
    assert "temperature" not in captured_kwargs
    assert results == [SampleResponse(reranked=[0])]
    assert reranker.get_last_token_usage() == reranker.TokenUsage(
        input_tokens=10,
        completion_tokens=20,
        reasoning_tokens=7,
        total_tokens=30,
    )
    assert (
        reranker.calculate_token_cost(
            "gpt-5.4", reranker.get_last_token_usage()
        ).total_cost
        > 0
    )


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
    assert results == [SampleResponse(reranked=[1])]


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
    assert results == [SampleResponse(reranked=[1])]


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
    assert batch_kwargs["max_completion_tokens"] == 16000
    assert completion_kwargs["max_completion_tokens"] == 24000
    assert results == [SampleResponse(reranked=[2])]
    assert reranker.get_last_token_usage() == reranker.TokenUsage(
        input_tokens=11,
        completion_tokens=22,
        reasoning_tokens=9,
        total_tokens=33,
    )


def test_get_gpt5_max_completion_tokens_scales_with_reasoning_effort():
    assert reranker.get_gpt5_max_completion_tokens(1000, None) == 1000
    assert reranker.get_gpt5_max_completion_tokens(1000, "medium") == 16000
    assert (
        reranker.get_gpt5_max_completion_tokens(1000, "medium", is_fallback=True)
        == 24000
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
    prompt = reranker.build_system_prompt("008_02_detailed")

    assert "子音より母音の一致を優先してください" in prompt
    assert "Example:" in prompt
    assert "Reranked: 6, 4, 5, 7, 2" in prompt


def test_transform_text_for_rerank_supports_pyopenjtalk_romaji():
    transformed = reranker.transform_text_for_rerank(
        "タロウ", input_transform="pyopenjtalk_romaji"
    )

    assert transformed == "t a r o o"


def test_rerank_by_llm_uses_selected_prompt_template(monkeypatch):
    captured_messages = []

    def fake_get_structured_outputs(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return [{"reranked": [1, 0]}]

    monkeypatch.setattr(reranker, "get_structured_outputs", fake_get_structured_outputs)

    reranked = reranker.rerank_by_llm(
        query_texts=["アケ"],
        wordlist_texts=[["アベ", "カケイ"]],
        model_name="gpt-5.4",
        prompt_template="008_03_step_by_step",
        rerank_interval=0,
    )

    assert "以下の手順で判断してください。" in captured_messages[0][0]["content"]
    assert reranked == [["カケイ", "アベ"]]


def test_rerank_by_llm_transforms_query_and_candidates_before_prompt(monkeypatch):
    captured_messages = []

    def fake_get_structured_outputs(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return [{"reranked": [1, 0]}]

    monkeypatch.setattr(reranker, "get_structured_outputs", fake_get_structured_outputs)
    monkeypatch.setattr(
        reranker,
        "transform_text_for_rerank",
        lambda text, input_transform="none": f"roma:{text}",
    )

    reranked = reranker.rerank_by_llm(
        query_texts=["アケ"],
        wordlist_texts=[["アベ", "カケイ"]],
        model_name="gpt-5.4",
        prompt_template="008_02_detailed",
        input_transform="pyopenjtalk_romaji",
        rerank_interval=0,
    )

    assert "Query: roma:アケ" in captured_messages[0][1]["content"]
    assert "0. roma:アベ" in captured_messages[0][1]["content"]
    assert "1. roma:カケイ" in captured_messages[0][1]["content"]
    assert reranked == [["カケイ", "アベ"]]
