# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "reproduce_leaderboard" / "methods" / "common"))

from reproduce_leaderboard.methods.common import batch_reranker, reranker


def test_prepare_rerank_candidates_keeps_positive_words():
    candidates = batch_reranker.prepare_rerank_candidates(
        base_ranked_wordlists=[["オウ", "アベ", "カケイ"]],
        positive_texts=[["カケイ"]],
        rerank_input_size=2,
    )

    assert candidates == [["オウ", "カケイ"]]


def test_submit_openai_batch_evaluation_prepares_candidates(monkeypatch):
    captured = {}

    def fake_submit_openai_batch_rerank_job(**kwargs):
        captured.update(kwargs)
        return {"batch_id": "batch-123"}

    monkeypatch.setattr(
        batch_reranker,
        "submit_openai_batch_rerank_job",
        fake_submit_openai_batch_rerank_job,
    )

    state = batch_reranker.submit_openai_batch_evaluation(
        base_rank_func=lambda query_texts, word_texts, **_: [
            ["オウ", "アベ", "カケイ"]
        ],
        query_texts=["アケ"],
        word_texts=["オウ", "アベ", "カケイ"],
        positive_texts=[["カケイ"]],
        rank_kwargs={},
        rerank_input_size=2,
        topn=10,
        model_name="gpt-5.4",
        prompt_template="default",
        state_path="state.json",
        output_file_path="output.json",
        reasoning_effort="medium",
    )

    assert state == {"batch_id": "batch-123"}
    assert captured["wordlist_texts"] == [["オウ", "カケイ"]]


def test_retrieve_openai_batch_evaluation_results_sets_metadata(tmp_path, monkeypatch):
    state_path = tmp_path / "rerank_state.json"
    state_path.write_text('{"batch_id":"batch-123"}', encoding="utf-8")

    monkeypatch.setattr(
        batch_reranker,
        "retrieve_openai_batch_rerank_job",
        lambda **_: SimpleNamespace(
            reranked_wordlists=[["カケイ", "アベ"]],
            execution_time=12.5,
        ),
    )
    monkeypatch.setattr(
        batch_reranker,
        "get_last_token_usage",
        lambda: reranker.TokenUsage(
            input_tokens=11,
            completion_tokens=22,
            reasoning_tokens=9,
            total_tokens=33,
        ),
    )
    monkeypatch.setattr(
        batch_reranker,
        "calculate_token_cost",
        lambda model_name, token_usage, *, discount_factor: reranker.TokenCost(
            input_cost=0.1 * discount_factor,
            output_cost=0.2 * discount_factor,
            reasoning_cost=0.05 * discount_factor,
            total_cost=0.3 * discount_factor,
        ),
    )

    results = batch_reranker.retrieve_openai_batch_evaluation_results(
        state_path=str(state_path),
        query_texts=["アケ"],
        positive_texts=[["アベ"]],
        rank_func="vowel_consonant",
        vowel_ratio=0.5,
        topn=10,
        rerank_input_size=100,
        model_name="gpt-5.4",
        reasoning_effort="medium",
        prompt_template="default",
        backend="openai_batch",
    )

    assert results.parameters.rank_func == "vowel_consonant"
    assert results.parameters.rerank_batch_id == "batch-123"
    assert results.parameters.rerank_backend == "openai_batch"
    assert results.metrics.execution_time == 12.5
    assert results.metrics.rerank_total_tokens == 33
    assert results.metrics.rerank_total_cost == 0.15
