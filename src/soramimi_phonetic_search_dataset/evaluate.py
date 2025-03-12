import time  # 追加
from pathlib import Path
from typing import Callable

from soramimi_phonetic_search_dataset.schemas import (
    PhoneticSearchDataset,
    PhoneticSearchMetrics,
    PhoneticSearchParameters,
    PhoneticSearchResult,
    PhoneticSearchResults,
)


def load_phonetic_search_dataset(path: str) -> PhoneticSearchDataset:
    """データセットを読み込む"""
    import json

    with open(path, "r") as f:
        dataset = json.load(f)
    return PhoneticSearchDataset.from_dict(dataset)


def calculate_recall(
    ranked_wordlists: list[list[str]],
    positive_texts: list[list[str]],
    topn: int = 10,
) -> float:
    """
    ランキング結果のRecall@Nを計算する

    Args:
        ranked_wordlists: 各クエリに対するランキング結果
        positive_texts: 各クエリに対する正解リスト
        topn: 評価に使用する上位n件

    Returns:
        float: Recall@N
    """
    recalls = []
    for wordlist, positive_text in zip(ranked_wordlists, positive_texts):
        topn_wordlist = wordlist[:topn]
        positive_text_count = len(positive_text)
        hit_count = len(set(topn_wordlist) & set(positive_text))
        recall = hit_count / positive_text_count if positive_text_count > 0 else 0.0
        recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0.0


def evaluate_ranking_function_with_details(
    ranking_func: Callable[[list[str], list[str]], list[list[str]]],
    topn: int = 10,
) -> PhoneticSearchResults:
    """
    ランキング関数の評価を行う

    Args:
        ranking_func: ランキング関数。query_textsとwordlist_textsを受け取り、
                     各クエリに対する単語リストのランキングを返す
        topn: 評価に使用する上位n件

    Returns:
        PhoneticSearchResults: 評価結果
    """
    # デフォルトのデータセットを読み込む
    dataset_path = Path(__file__).parent / "data" / "baseball.json"
    dataset = load_phonetic_search_dataset(str(dataset_path))

    # クエリと正解を取得
    query_texts = [query.query for query in dataset.queries]
    positive_texts = [query.positive for query in dataset.queries]

    # ランキングを実行（実行時間を計測）
    start_time = time.time()
    ranked_wordlists = ranking_func(query_texts, dataset.words)
    execution_time = time.time() - start_time

    # Recallを計算
    recall = calculate_recall(ranked_wordlists, positive_texts, topn=topn)

    # 結果を作成
    results = [
        PhoneticSearchResult(
            query=query.query,
            ranked_words=wordlist[:topn],
            positive_words=positive_text,
        )
        for query, wordlist, positive_text in zip(
            dataset.queries, ranked_wordlists, positive_texts
        )
    ]

    # パラメータは最小限の情報のみ
    parameters = PhoneticSearchParameters(
        topn=topn,
        rank_func="unknown",  # basic_usage.py側で設定する
    )

    metrics = PhoneticSearchMetrics(
        recall=recall,
        execution_time=execution_time,  # 実行時間を追加
    )

    return PhoneticSearchResults(
        parameters=parameters,
        metrics=metrics,
        results=results,
    )


def evaluate_ranking_function(
    ranking_func: Callable[[list[str], list[str]], list[list[str]]],
    topn: int = 10,
) -> float:
    return evaluate_ranking_function_with_details(ranking_func, topn).metrics.recall
