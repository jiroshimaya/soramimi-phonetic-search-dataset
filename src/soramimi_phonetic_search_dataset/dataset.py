"""
データセット関連の処理を提供するモジュール
"""

import json
from pathlib import Path

from .schemas import PhoneticSearchDataset

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "baseball.json"
SMALL_DATASET_QUERY_COUNT = 10


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
) -> PhoneticSearchDataset:
    """デフォルトのデータセットを読み込む"""
    dataset = load_phonetic_search_dataset(str(DEFAULT_DATASET_PATH))
    return _slice_dataset(dataset, query_limit=query_limit, query_offset=query_offset)


def load_small_dataset() -> PhoneticSearchDataset:
    """LLMの試行用に先頭10件へ絞った小データセットを読み込む"""
    return load_default_dataset(query_limit=SMALL_DATASET_QUERY_COUNT)
