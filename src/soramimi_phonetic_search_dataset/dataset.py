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


def _subset_dataset(
    dataset: PhoneticSearchDataset, *, query_limit: int
) -> PhoneticSearchDataset:
    if query_limit <= 0:
        raise ValueError("query_limit must be a positive integer")
    if query_limit >= len(dataset.queries):
        return dataset

    metadata = {
        **dataset.metadata,
        "query_limit": query_limit,
        "subset": f"first_{query_limit}_queries",
        "source_dataset": DEFAULT_DATASET_PATH.name,
    }
    return PhoneticSearchDataset(
        queries=dataset.queries[:query_limit],
        words=dataset.words,
        metadata=metadata,
    )


def load_default_dataset(query_limit: int | None = None) -> PhoneticSearchDataset:
    """デフォルトのデータセットを読み込む"""
    dataset = load_phonetic_search_dataset(str(DEFAULT_DATASET_PATH))
    if query_limit is None:
        return dataset
    return _subset_dataset(dataset, query_limit=query_limit)


def load_small_dataset() -> PhoneticSearchDataset:
    """LLMの試行用に先頭10件へ絞った小データセットを読み込む"""
    return load_default_dataset(query_limit=SMALL_DATASET_QUERY_COUNT)
