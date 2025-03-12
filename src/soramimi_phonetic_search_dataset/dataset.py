"""
データセット関連の処理を提供するモジュール
"""

import json
from pathlib import Path

from .schemas import PhoneticSearchDataset

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "baseball.json"


def load_phonetic_search_dataset(path: str) -> PhoneticSearchDataset:
    """データセットを読み込む"""
    with open(path, "r") as f:
        dataset = json.load(f)
    return PhoneticSearchDataset.from_dict(dataset)


def load_default_dataset() -> PhoneticSearchDataset:
    """デフォルトのデータセットを読み込む"""
    return load_phonetic_search_dataset(str(DEFAULT_DATASET_PATH))
