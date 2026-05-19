# ruff: noqa: E402

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "reproduce_leaderboard" / "methods" / "common"))

from reproduce_leaderboard.methods.common import evaluate_ranking
from soramimi_phonetic_search_dataset.schemas import (
    PhoneticSearchDataset,
    PhoneticSearchMetrics,
    PhoneticSearchParameters,
    PhoneticSearchQuery,
    PhoneticSearchResults,
)


def make_sample_dataset() -> PhoneticSearchDataset:
    return PhoneticSearchDataset(
        queries=[PhoneticSearchQuery(query="アケ", positive=["アベ"])],
        words=["アベ", "カケイ"],
    )


def make_sample_results() -> PhoneticSearchResults:
    return PhoneticSearchResults(
        parameters=PhoneticSearchParameters(topn=10, rank_func="unknown"),
        metrics=PhoneticSearchMetrics(recall=1.0, execution_time=0.0),
        results=[],
    )


def test_prepare_rerank_candidates_keeps_positive_words():
    candidates = evaluate_ranking.prepare_rerank_candidates(
        base_ranked_wordlists=[["オウ", "アベ", "カケイ"]],
        positive_texts=[["カケイ"]],
        rerank_input_size=2,
    )

    assert candidates == [["オウ", "カケイ"]]


def test_正常系_query_limit付きの出力ファイル名を生成できる():
    output_path = evaluate_ranking.get_default_output_path(
        "vowel_consonant",
        10,
        query_limit=100,
        rerank=True,
        rerank_topn=100,
        rerank_model_name="gpt-5.4",
        rerank_reasoning_effort="none",
        rerank_prompt_template="step_by_step",
        rerank_input_transform="none",
    )

    assert output_path.endswith("_promptstep_by_step_querylimit100.json")


def test_正常系_query_offset付きの出力ファイル名を生成できる():
    output_path = evaluate_ranking.get_default_output_path(
        "vowel_consonant",
        10,
        query_limit=50,
        query_offset=100,
        rerank=True,
        rerank_topn=100,
        rerank_model_name="gpt-5.4",
        rerank_reasoning_effort="none",
        rerank_prompt_template="step_by_step",
        rerank_input_transform="none",
    )

    assert output_path.endswith("_promptstep_by_step_querylimit50_queryoffset100.json")


def test_正常系_query_limit指定でdefault_datasetを絞って評価できる(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, int | None] = {}
    sample_results = make_sample_results()

    def fake_load_default_dataset(query_limit=None, query_offset=0):
        captured["query_limit"] = query_limit
        captured["query_offset"] = query_offset
        return make_sample_dataset()

    monkeypatch.setattr(
        evaluate_ranking, "load_default_dataset", fake_load_default_dataset
    )
    monkeypatch.setattr(
        evaluate_ranking,
        "evaluate_ranking_function",
        lambda ranking_func, topn, dataset: sample_results,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_ranking.py", "--query_limit", "100", "--no_save"],
    )

    evaluate_ranking.main()

    assert captured["query_limit"] == 100
    assert captured["query_offset"] == 0
    assert sample_results.parameters.metadata["query_limit"] == 100
    assert sample_results.parameters.metadata["query_offset"] == 0


def test_正常系_query_offset指定でdefault_datasetをずらして評価できる(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, int | None] = {}
    sample_results = make_sample_results()

    def fake_load_default_dataset(query_limit=None, query_offset=0):
        captured["query_limit"] = query_limit
        captured["query_offset"] = query_offset
        return make_sample_dataset()

    monkeypatch.setattr(
        evaluate_ranking, "load_default_dataset", fake_load_default_dataset
    )
    monkeypatch.setattr(
        evaluate_ranking,
        "evaluate_ranking_function",
        lambda ranking_func, topn, dataset: sample_results,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ranking.py",
            "--query_limit",
            "50",
            "--query_offset",
            "100",
            "--no_save",
        ],
    )

    evaluate_ranking.main()

    assert captured["query_limit"] == 50
    assert captured["query_offset"] == 100
    assert sample_results.parameters.metadata["query_limit"] == 50
    assert sample_results.parameters.metadata["query_offset"] == 100


def test_異常系_small_datasetとquery_limitの併用はエラーになる(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ranking.py",
            "--dataset_size",
            "small",
            "--query_limit",
            "5",
            "--no_save",
        ],
    )

    with pytest.raises(SystemExit):
        evaluate_ranking.main()


def test_異常系_small_datasetとquery_offsetの併用はエラーになる(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ranking.py",
            "--dataset_size",
            "small",
            "--query_offset",
            "1",
            "--no_save",
        ],
    )

    with pytest.raises(SystemExit):
        evaluate_ranking.main()
