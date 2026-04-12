import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_build_openai_batch_requests_uses_json_mode_and_reasoning_effort():
    requests = reranker.build_openai_batch_requests(
        model_name="gpt-5.4",
        messages=[[{"role": "user", "content": "hello"}]],
        custom_ids=["rerank-0000"],
        response_format=SampleResponse,
        reasoning_effort="medium",
    )

    assert requests[0]["custom_id"] == "rerank-0000"
    assert requests[0]["method"] == "POST"
    assert requests[0]["url"] == "/v1/chat/completions"
    assert requests[0]["body"]["model"] == "gpt-5.4"
    assert requests[0]["body"]["max_completion_tokens"] == 16000
    assert requests[0]["body"]["reasoning_effort"] == "medium"
    assert requests[0]["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert requests[0]["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "SampleResponse",
            "strict": True,
            "schema": SampleResponse.model_json_schema(),
        },
    }


def test_build_openai_json_schema_response_format_uses_pydantic_schema():
    assert reranker._build_openai_json_schema_response_format(SampleResponse) == {
        "type": "json_schema",
        "json_schema": {
            "name": "SampleResponse",
            "strict": True,
            "schema": SampleResponse.model_json_schema(),
        },
    }


def test_submit_openai_batch_rerank_job_writes_state_and_requests(tmp_path):
    class FakeFiles:
        def create(self, *, file, purpose):
            assert purpose == "batch"
            assert file.read()
            file.seek(0)
            return SimpleNamespace(id="file-input")

    class FakeBatches:
        def create(self, *, input_file_id, endpoint, completion_window):
            assert input_file_id == "file-input"
            assert endpoint == "/v1/chat/completions"
            assert completion_window == "24h"
            return SimpleNamespace(id="batch-123", status="validating")

    fake_client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    state_path = tmp_path / "rerank_state.json"

    state = reranker.submit_openai_batch_rerank_job(
        query_texts=["アケ"],
        wordlist_texts=[["アベ", "カケイ"]],
        positive_texts=[["アベ"]],
        topn=10,
        model_name="gpt-5.4",
        prompt_template="008_03_step_by_step",
        response_format=SampleResponse,
        state_path=str(state_path),
        output_file_path=str(tmp_path / "results.json"),
        reasoning_effort="medium",
        client=fake_client,
    )

    assert state["batch_id"] == "batch-123"
    assert state_path.exists()
    request_path = state_path.with_suffix(".requests.jsonl")
    assert request_path.exists()
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["parameters"]["rerank_model_name"] == "gpt-5.4"
    assert saved_state["items"][0]["candidate_words"] == ["アベ", "カケイ"]
    request_row = json.loads(request_path.read_text(encoding="utf-8").strip())
    assert request_row["custom_id"] == "rerank-0000"


def test_retrieve_openai_batch_rerank_job_restores_results_and_token_usage(tmp_path):
    state_path = tmp_path / "rerank_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "openai_batch",
                "endpoint": "/v1/chat/completions",
                "batch_id": "batch-123",
                "batch_status": "validating",
                "input_file_id": "file-input",
                "request_file_path": str(tmp_path / "requests.jsonl"),
                "output_file_path": str(tmp_path / "results.json"),
                "result_file_path": None,
                "error_file_path": None,
                "submitted_at": "2026-04-12T00:00:00",
                "parameters": {
                    "topn": 10,
                    "rerank_model_name": "gpt-5.4",
                    "rerank_reasoning_effort": "medium",
                    "rerank_prompt_template": "008_03_step_by_step",
                },
                "items": [
                    {
                        "custom_id": "rerank-0000",
                        "query": "アケ",
                        "candidate_words": ["アベ", "カケイ"],
                        "positive_words": ["アベ"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response_row = {
        "custom_id": "rerank-0000",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"reranked": [1, 0]}',
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            },
        },
    }

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file-output"
            return SimpleNamespace(
                content=(json.dumps(response_row, ensure_ascii=False) + "\n").encode("utf-8")
            )

    class FakeBatches:
        def retrieve(self, batch_id):
            assert batch_id == "batch-123"
            return SimpleNamespace(
                id=batch_id,
                status="completed",
                output_file_id="file-output",
                error_file_id=None,
                created_at=100,
                in_progress_at=110,
                completed_at=170,
                request_counts={"total": 1, "completed": 1, "failed": 0},
                usage=SimpleNamespace(
                    input_tokens=11,
                    output_tokens=22,
                    total_tokens=33,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=9),
                ),
            )

    fake_client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())

    result = reranker.retrieve_openai_batch_rerank_job(
        state_path=str(state_path),
        response_format=SampleResponse,
        client=fake_client,
    )

    assert result.reranked_wordlists == [["カケイ", "アベ"]]
    assert result.batch_status == "completed"
    assert result.execution_time == 60.0


def test_retrieve_openai_batch_rerank_job_surfaces_error_file_details(tmp_path):
    state_path = tmp_path / "rerank_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "openai_batch",
                "endpoint": "/v1/chat/completions",
                "batch_id": "batch-123",
                "batch_status": "validating",
                "input_file_id": "file-input",
                "request_file_path": str(tmp_path / "requests.jsonl"),
                "output_file_path": str(tmp_path / "results.json"),
                "result_file_path": None,
                "error_file_path": None,
                "submitted_at": "2026-04-12T00:00:00",
                "parameters": {
                    "topn": 10,
                    "rerank_model_name": "gpt-5.4",
                    "rerank_reasoning_effort": "medium",
                    "rerank_prompt_template": "008_03_step_by_step",
                },
                "items": [
                    {
                        "custom_id": "rerank-0000",
                        "query": "アケ",
                        "candidate_words": ["アベ", "カケイ"],
                        "positive_words": ["アベ"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    error_row = {
        "custom_id": "rerank-0000",
        "response": {
            "status_code": 400,
            "body": {
                "error": {
                    "message": "schema validation failed for reranked",
                }
            },
        },
        "error": None,
    }

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file-error"
            return SimpleNamespace(
                content=(json.dumps(error_row, ensure_ascii=False) + "\n").encode("utf-8")
            )

    class FakeBatches:
        def retrieve(self, batch_id):
            assert batch_id == "batch-123"
            return SimpleNamespace(
                id=batch_id,
                status="completed",
                output_file_id=None,
                error_file_id="file-error",
                created_at=100,
                in_progress_at=110,
                completed_at=170,
                request_counts={"total": 1, "completed": 0, "failed": 1},
                usage=None,
            )

    fake_client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())

    with pytest.raises(RuntimeError, match="sample_errors=.*schema validation failed"):
        reranker.retrieve_openai_batch_rerank_job(
            state_path=str(state_path),
            response_format=SampleResponse,
            client=fake_client,
        )
    assert reranker.get_last_token_usage() == reranker.TokenUsage(
        input_tokens=11,
        completion_tokens=22,
        reasoning_tokens=9,
        total_tokens=33,
    )
