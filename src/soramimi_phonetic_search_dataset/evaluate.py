import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TypeAlias

from soramimi_phonetic_search_dataset.dataset import load_default_dataset
from soramimi_phonetic_search_dataset.schemas import (
    PhoneticSearchDataset,
    PhoneticSearchMetrics,
    PhoneticSearchParameters,
    PhoneticSearchResult,
    PhoneticSearchResults,
)


@dataclass
class RankingFunctionOutput:
    ranked_wordlists: list[list[str]]
    result_metadata: list[dict[str, Any]] | None = None
    metrics_metadata: dict[str, Any] | None = None


RankingFunc: TypeAlias = Callable[
    [list[str], list[str]],
    list[list[str]] | RankingFunctionOutput,
]
"""\
評価対象のランキング関数のシグネチャ。

入力として query_texts と wordlist_texts を受け取り、各クエリに対する ranked_words の
リストを返す。必要に応じて RankingFunctionOutput を返し、各クエリに対応する
metadata のリストや、評価全体に紐づく metrics metadata を渡してもよい。
"""


def _normalize_ranking_output(
    ranking_output: list[list[str]] | RankingFunctionOutput,
) -> tuple[list[list[str]], list[dict[str, Any]] | None, dict[str, Any] | None]:
    if isinstance(ranking_output, list):
        return ranking_output, None, None

    return (
        ranking_output.ranked_wordlists,
        ranking_output.result_metadata,
        ranking_output.metrics_metadata,
    )


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


def evaluate_ranking_function(
    ranking_func: RankingFunc,
    topn: int = 10,
    dataset: PhoneticSearchDataset | None = None,
) -> PhoneticSearchResults:
    """
    ランキング関数の評価を行う

    Args:
        ranking_func: ランキング関数。query_textsとwordlist_textsを受け取り、
                 各クエリに対する単語リストのランキング、またはランキングとmetadataを返す
        topn: 評価に使用する上位n件

    Returns:
        PhoneticSearchResults: 評価結果
    """
    if dataset is None:
        dataset = load_default_dataset()

    # クエリと正解を取得
    query_texts = [query.query for query in dataset.queries]
    positive_texts = [query.positive for query in dataset.queries]

    # ランキングを実行（実行時間を計測）
    start_time = time.time()
    ranking_output = ranking_func(query_texts, dataset.words)
    execution_time = time.time() - start_time
    ranked_wordlists, metadatas, metrics_metadata = _normalize_ranking_output(
        ranking_output
    )

    # Recallを計算
    recall = calculate_recall(ranked_wordlists, positive_texts, topn=topn)

    # 結果を作成
    results = [
        PhoneticSearchResult(
            query=query.query,
            ranked_words=wordlist[:topn],
            positive_words=positive_text,
            metadata=metadatas[index] if metadatas is not None else None,
        )
        for index, (query, wordlist, positive_text) in enumerate(
            zip(dataset.queries, ranked_wordlists, positive_texts)
        )
    ]

    # パラメータは最小限の情報のみ
    parameters = PhoneticSearchParameters(
        topn=topn,
        rank_func="unknown",  # basic_usage.py側で設定する
        execution_timestamp=datetime.now().isoformat(),  # 実行日時を追加
    )

    metrics = PhoneticSearchMetrics(
        recall=recall,
        execution_time=execution_time,  # 実行時間を追加
        metadata=metrics_metadata or {},
    )

    return PhoneticSearchResults(
        parameters=parameters,
        metrics=metrics,
        results=results,
    )
