import json

import pytest

from soramimi_phonetic_search_dataset import (
    PhoneticSearchDataset,
    PhoneticSearchQuery,
    evaluate_ranking_function,
    load_default_dataset,
    load_phonetic_search_dataset,
    load_small_dataset,
)
from soramimi_phonetic_search_dataset.evaluate import calculate_recall


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
    assert limited_dataset.words == sample_dataset.words
    assert limited_dataset.metadata["query_limit"] == 1
    assert limited_dataset.metadata["subset"] == "first_1_queries"


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


def test_load_small_dataset(monkeypatch, sample_dataset):
    """小データセットは十分小さい入力では元データをそのまま返す"""

    def mock_load_dataset(path):
        return sample_dataset

    monkeypatch.setattr(
        "soramimi_phonetic_search_dataset.dataset.load_phonetic_search_dataset",
        mock_load_dataset,
    )

    loaded_dataset = load_small_dataset()
    assert loaded_dataset is sample_dataset


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
    def perfect_ranking(query_texts, wordlist_texts):
        results = []
        for query in query_texts:
            if query == "タロウ":
                results.append(["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"])
            else:  # ハナコ
                results.append(["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"])
        return results

    recall = evaluate_ranking_function(ranking_func=perfect_ranking, topn=2)
    assert recall == 1.0  # 全てのクエリで正解を含む


def test_evaluate_ranking_function_with_explicit_dataset(sample_dataset):
    """明示的に渡したデータセットで評価できる"""

    def perfect_ranking(query_texts, wordlist_texts):
        assert query_texts == ["タロウ", "ハナコ"]
        assert wordlist_texts == sample_dataset.words
        return [
            ["タロー", "タロ", "タロウ", "ハナコ", "ハナ", "ハナゴ"],
            ["ハナ", "ハナゴ", "ハナコ", "タロウ", "タロー", "タロ"],
        ]

    recall = evaluate_ranking_function(
        ranking_func=perfect_ranking,
        topn=2,
        dataset=sample_dataset,
    )
    assert recall == 1.0
