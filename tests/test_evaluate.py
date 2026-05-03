import json

import pytest

from soramimi_phonetic_search_dataset import (
    PhoneticSearchDataset,
    PhoneticSearchQuery,
    build_wordlist_dataset,
    evaluate_ranking_function,
    load_default_dataset,
    load_default_dataset_for_llm,
    load_phonetic_search_dataset,
    load_small_dataset,
)
from soramimi_phonetic_search_dataset.evaluate import (
    RankingFunctionOutput,
    calculate_recall,
)


@pytest.fixture
def sample_dataset():
    """サンプルデータセットを作成"""
    return PhoneticSearchDataset(
        queries=[
            PhoneticSearchQuery(query="タロウ", positive=["タロー", "タロ"]),
            PhoneticSearchQuery(query="ハナコ", positive=["ハナ", "ハナゴ"]),
        ],
        words=["タロウ", "タロー", "タロ", "ハナコ", "ハナ", "ハナゴ"],
    )


@pytest.fixture
def sample_dataset_for_llm():
    """LLM向け候補生成を検証するサンプルデータセット"""
    return PhoneticSearchDataset(
        queries=[
            PhoneticSearchQuery(
                query="アケ",
                positive=["アベ", "イケ"],
                hard_negatives=["ウエ", "オノ", "カコ", "キク"],
            )
        ],
        words=["アベ", "イケ", "ウエ", "オノ", "カコ", "キク"],
    )


@pytest.fixture
def sample_wordlist_dataset(sample_dataset):
    return build_wordlist_dataset(sample_dataset)


@pytest.fixture
def sample_dataset_file(sample_dataset, tmp_path):
    """サンプルデータセットをファイルに保存"""
    dataset_path = tmp_path / "test_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(
            {
                "queries": [
                    {"query": q.query, "positive": q.positive}
                    for q in sample_dataset.queries
                ],
                "words": sample_dataset.words,
            },
            f,
        )
    return str(dataset_path)


def test_load_phonetic_search_dataset(sample_dataset, sample_dataset_file):
    """データセット読み込みのテスト"""
    loaded_dataset = load_phonetic_search_dataset(sample_dataset_file)
    assert len(loaded_dataset.queries) == len(sample_dataset.queries)
    assert len(loaded_dataset.words) == len(sample_dataset.words)
    for loaded_query, original_query in zip(
        loaded_dataset.queries, sample_dataset.queries
    ):
        assert loaded_query.query == original_query.query
        assert loaded_query.positive == original_query.positive


def test_load_phonetic_search_dataset_with_hard_negatives(tmp_path):
    """hard_negatives を含むデータセットを読み込める"""

    dataset_path = tmp_path / "test_dataset_with_hard_negatives.json"
    with open(dataset_path, "w") as f:
        json.dump(
            {
                "queries": [
                    {
                        "query": "タロウ",
                        "positive": ["タロー", "タロ"],
                        "hard_negatives": ["ハナコ", "サブロウ"],
                    }
                ],
                "words": ["タロウ", "タロー", "タロ", "ハナコ", "サブロウ"],
            },
            f,
        )

    loaded_dataset = load_phonetic_search_dataset(str(dataset_path))

    assert loaded_dataset.queries[0].query == "タロウ"
    assert loaded_dataset.queries[0].positive == ["タロー", "タロ"]
    assert loaded_dataset.queries[0].hard_negatives == ["ハナコ", "サブロウ"]


def test_load_default_dataset_with_query_limit(monkeypatch, sample_dataset):
    """クエリ数を絞ってデータセットを読み込める"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    limited_dataset = load_default_dataset(query_limit=1)
    assert len(limited_dataset.queries) == 1
    assert limited_dataset.queries[0].wordlist == sample_dataset.words
    assert limited_dataset.queries[0].positive_words == sample_dataset.queries[0].positive
    assert limited_dataset.metadata["query_limit"] == 1
    assert limited_dataset.metadata["query_offset"] == 0
    assert limited_dataset.metadata["subset"] == "queries_1_to_1"
    assert limited_dataset.metadata["format"] == "query_with_wordlist"


def test_load_default_dataset_with_invalid_query_limit(monkeypatch, sample_dataset):
    """query_limitは正の整数のみ受け付ける"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    with pytest.raises(ValueError, match="query_limit must be a positive integer"):
        load_default_dataset(query_limit=0)


def test_load_default_dataset_with_query_offset(monkeypatch, sample_dataset):
    """query_offset付きでデータセットを読み込める"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    sliced_dataset = load_default_dataset(query_limit=1, query_offset=1)
    assert len(sliced_dataset.queries) == 1
    assert sliced_dataset.queries[0].query == "ハナコ"
    assert sliced_dataset.queries[0].wordlist == sample_dataset.words
    assert sliced_dataset.queries[0].positive_words == ["ハナ", "ハナゴ"]
    assert sliced_dataset.metadata["query_limit"] == 1
    assert sliced_dataset.metadata["query_offset"] == 1
    assert sliced_dataset.metadata["subset"] == "queries_2_to_2"


def test_load_default_dataset_with_invalid_query_offset(monkeypatch, sample_dataset):
    """query_offsetは0以上のみ受け付ける"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    with pytest.raises(ValueError, match="query_offset must be a non-negative integer"):
        load_default_dataset(query_offset=-1)


def test_load_small_dataset(monkeypatch, sample_dataset):
    """小データセットは query ごとの wordlist を持つ形式で返る"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    loaded_dataset = load_small_dataset()
    assert len(loaded_dataset.queries) == len(sample_dataset.queries)
    assert loaded_dataset.queries[0].wordlist == sample_dataset.words
    assert loaded_dataset.queries[0].positive_words == sample_dataset.queries[0].positive
    assert loaded_dataset.metadata["format"] == "query_with_wordlist"


def test_load_default_dataset_for_llm_builds_query_wordlists(
    monkeypatch, sample_dataset_for_llm
):
    """LLM向けローダーは query ごとの候補語リストを返す"""

    def mock_load_dataset(path):
        return sample_dataset_for_llm

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    loaded_dataset = load_default_dataset_for_llm(wordlist_size=4)

    assert len(loaded_dataset.queries) == 1
    assert loaded_dataset.queries[0].query == "アケ"
    assert loaded_dataset.queries[0].wordlist == ["アベ", "イケ", "ウエ", "オノ"]
    assert loaded_dataset.queries[0].positive_words == ["アベ", "イケ"]
    assert loaded_dataset.metadata["wordlist_size"] == 4
    assert loaded_dataset.metadata["format"] == "query_with_wordlist"


def test_load_default_dataset_for_llm_requires_enough_hard_negatives(
    monkeypatch, sample_dataset
):
    """LLM向けローダーは不足した hard_negatives を弾く"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    with pytest.raises(
        ValueError, match="hard_negatives are required to build an LLM wordlist"
    ):
        load_default_dataset_for_llm(wordlist_size=4)


def test_calculate_recall():
    """リコール計算のテスト"""
    ranked_wordlists = [
        ["タロー", "タロウ", "タロ", "ハナコ"],  # 2/2 = 1.0
        ["ハナ", "ハナゴ", "ハナコ", "タロウ"],  # 2/2 = 1.0
    ]
    positive_texts = [
        ["タロー", "タロ"],
        ["ハナ", "ハナゴ"],
    ]
    recall = calculate_recall(ranked_wordlists, positive_texts, topn=3)
    assert recall == 1.0  # 両方のクエリで正解を含む

    # 一部のみ正解を含むケース
    ranked_wordlists = [
        ["タロー", "タロウ", "タロ", "ハナコ"],  # 1/2 = 0.5
        ["ハナコ", "タロウ", "タロ", "ハナ"],  # 0/2 = 0.0
    ]
    recall = calculate_recall(ranked_wordlists, positive_texts, topn=2)
    assert recall == 0.25  # (0.5 + 0.0) / 2


def test_evaluate_ranking_function(monkeypatch, sample_dataset):
    """評価関数のテスト"""

    # デフォルトのデータセットをモック
    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    # 完全一致するランキング関数
    def perfect_ranking(query_inputs):
        results = []
        for query_input in query_inputs:
            if query_input.query == "タロウ":
                results.append(["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"])
            else:  # ハナコ
                results.append(["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"])
        return results

    results = evaluate_ranking_function(ranking_func=perfect_ranking, topn=2)
    assert results.metrics.recall == 1.0  # 全てのクエリで正解を含む
    assert results.parameters.rank_func == "perfect_ranking"


def test_evaluate_ranking_function_with_explicit_dataset(
    sample_dataset, sample_wordlist_dataset
):
    """明示的に渡したデータセットで評価できる"""

    def perfect_ranking(query_inputs):
        assert [query_input.query for query_input in query_inputs] == ["タロウ", "ハナコ"]
        assert query_inputs[0].wordlist == sample_dataset.words
        assert query_inputs[0].wordlist is query_inputs[1].wordlist
        return [
            ["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"],
            ["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"],
        ]

    results = evaluate_ranking_function(
        ranking_func=perfect_ranking,
        topn=2,
        dataset=sample_wordlist_dataset,
    )
    assert results.metrics.recall == 1.0


def test_evaluate_ranking_function_with_metadata(sample_dataset, sample_wordlist_dataset):
    """ランキング関数がmetadataを返しても評価できる"""

    def ranking_with_metadata(query_inputs):
        assert [query_input.query for query_input in query_inputs] == ["タロウ", "ハナコ"]
        assert query_inputs[0].wordlist == sample_dataset.words
        return RankingFunctionOutput(
            ranked_wordlists=[
                ["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"],
                ["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"],
            ],
            result_metadata=[
                {"source": "exact", "score": 1.0, "thoughts": ["母音列が近い"]},
                {"source": "fuzzy", "score": 0.8, "thoughts": ["子音差を確認"]},
            ],
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_with_metadata,
        topn=2,
        dataset=sample_wordlist_dataset,
    )

    assert results.metrics.recall == 1.0
    assert results.results[0].metadata == {
        "source": "exact",
        "score": 1.0,
        "thoughts": ["母音列が近い"],
    }
    assert results.results[1].metadata == {
        "source": "fuzzy",
        "score": 0.8,
        "thoughts": ["子音差を確認"],
    }


def test_evaluate_ranking_function_with_metrics_metadata(
    sample_dataset, sample_wordlist_dataset
):
    """ランキング関数が全体メトリクスmetadataを返しても評価できる"""

    def ranking_with_metrics_metadata(query_inputs):
        assert [query_input.query for query_input in query_inputs] == ["タロウ", "ハナコ"]
        assert query_inputs[0].wordlist == sample_dataset.words
        return RankingFunctionOutput(
            ranked_wordlists=[
                ["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"],
                ["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"],
            ],
            metrics_metadata={
                "model_name": "gpt-5.4",
                "token_usage": {"total_tokens": 123},
                "cost": {"total_cost": 0.42},
            },
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_with_metrics_metadata,
        topn=2,
        dataset=sample_wordlist_dataset,
    )

    assert results.metrics.recall == 1.0
    assert results.metrics.metadata == {
        "model_name": "gpt-5.4",
        "token_usage": {"total_tokens": 123},
        "cost": {"total_cost": 0.42},
    }
