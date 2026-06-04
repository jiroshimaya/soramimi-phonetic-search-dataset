"""
データセット関連の処理を提供するモジュール
"""

import json
from pathlib import Path

from .schemas import (
    PhoneticSearchDataset,
    PhoneticSearchQueryWithWordlist,
    PhoneticSearchWordlistDataset,
)

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "baseball.json"
SMALL_DATASET_QUERY_COUNT = 10
DEFAULT_LLM_WORDLIST_SIZE = 100


def load_phonetic_search_dataset(path: str) -> PhoneticSearchDataset:
    """データセットを読み込む"""
    with open(path, "r") as f:
        dataset = json.load(f)
    return PhoneticSearchDataset.from_dict(dataset)


def _slice_dataset(
    dataset: PhoneticSearchDataset,
    *,
    query_limit: int | None = None,
    query_offset: int = 0,
) -> PhoneticSearchDataset:
    if query_offset < 0:
        raise ValueError("query_offset must be a non-negative integer")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be a positive integer")
    if query_offset == 0 and query_limit is None:
        return dataset
    if query_offset >= len(dataset.queries):
        raise ValueError("query_offset must be smaller than the number of queries")

    end_index = len(dataset.queries)
    if query_limit is not None:
        end_index = min(end_index, query_offset + query_limit)
    if query_offset == 0 and end_index >= len(dataset.queries):
        return dataset

    metadata = {
        **dataset.metadata,
        "query_offset": query_offset,
        "source_dataset": DEFAULT_DATASET_PATH.name,
        "subset": f"queries_{query_offset + 1}_to_{end_index}",
    }
    if query_limit is not None:
        metadata["query_limit"] = query_limit
    return PhoneticSearchDataset(
        queries=dataset.queries[query_offset:end_index],
        words=dataset.words,
        metadata=metadata,
    )


def load_default_dataset(
    query_limit: int | None = None,
    query_offset: int = 0,
) -> PhoneticSearchWordlistDataset:
    """デフォルトのデータセットを読み込む"""
    dataset = load_phonetic_search_dataset(str(DEFAULT_DATASET_PATH))
    return build_wordlist_dataset(
        _slice_dataset(dataset, query_limit=query_limit, query_offset=query_offset)
    )


def build_wordlist_dataset(
    dataset: PhoneticSearchDataset,
) -> PhoneticSearchWordlistDataset:
    """query ごとに wordlist を持つ入力形式へ変換する"""
    return PhoneticSearchWordlistDataset(
        queries=[
            PhoneticSearchQueryWithWordlist(
                query=query.query,
                wordlist=dataset.words,
                positive_words=query.positive,
                subset=query.subset,
            )
            for query in dataset.queries
        ],
        metadata={
            **dataset.metadata,
            "format": "query_with_wordlist",
        },
    )


def build_wordlist_dataset_for_llm(
    dataset: PhoneticSearchDataset,
    *,
    wordlist_size: int = DEFAULT_LLM_WORDLIST_SIZE,
) -> PhoneticSearchWordlistDataset:
    """LLM rerank 向けに query ごとの候補語リストへ変換する"""
    return PhoneticSearchWordlistDataset(
        queries=[
            PhoneticSearchQueryWithWordlist(
                query=query.query,
                wordlist=query.build_wordlist_for_llm(wordlist_size=wordlist_size),
                positive_words=query.positive,
                subset=query.subset,
            )
            for query in dataset.queries
        ],
        metadata={
            **dataset.metadata,
            "wordlist_size": wordlist_size,
            "format": "query_with_wordlist",
        },
    )


def load_default_dataset_for_llm(
    query_limit: int | None = None,
    query_offset: int = 0,
    *,
    wordlist_size: int = DEFAULT_LLM_WORDLIST_SIZE,
) -> PhoneticSearchWordlistDataset:
    """LLM rerank 向けのデフォルトデータセットを読み込む"""
    dataset = _slice_dataset(
        load_phonetic_search_dataset(str(DEFAULT_DATASET_PATH)),
        query_limit=query_limit,
        query_offset=query_offset,
    )
    return build_wordlist_dataset_for_llm(dataset, wordlist_size=wordlist_size)


def load_small_dataset() -> PhoneticSearchWordlistDataset:
    """LLMの試行用に先頭10件へ絞った小データセットを読み込む"""
    return load_default_dataset(query_limit=SMALL_DATASET_QUERY_COUNT)
