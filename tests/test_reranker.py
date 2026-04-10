from types import SimpleNamespace

from pydantic import BaseModel

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
        return [_mock_completion_response('{"reranked": [0]}')]

    monkeypatch.setattr(reranker, "batch_completion", fake_batch_completion)

    results = reranker.get_structured_outputs(
        model_name="gpt-5.4",
        messages=[[{"role": "user", "content": "hello"}]],
        response_format=SampleResponse,
        reasoning_effort="medium",
    )

    assert captured_kwargs["max_completion_tokens"] == 1000
    assert captured_kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert "max_tokens" not in captured_kwargs
    assert "temperature" not in captured_kwargs
    assert results == [SampleResponse(reranked=[0])]


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


def test_get_structured_outputs_falls_back_to_single_completion(monkeypatch):
    batch_kwargs = {}
    completion_kwargs = {}

    def fake_batch_completion(**kwargs):
        batch_kwargs.update(kwargs)
        return [ValueError("empty response")]

    def fake_completion(**kwargs):
        completion_kwargs.update(kwargs)
        return _mock_completion_response('{"reranked": [2]}')

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
    assert batch_kwargs["max_completion_tokens"] == 1000
    assert completion_kwargs["max_completion_tokens"] == 4000
    assert results == [SampleResponse(reranked=[2])]
