"""
Soramimi Phonetic Search Dataset package
"""

from .dataset import (
    DEFAULT_DATASET_PATH,
    SMALL_DATASET_QUERY_COUNT,
    load_default_dataset,
    load_phonetic_search_dataset,
    load_small_dataset,
)
from .evaluate import RankingFunctionOutput, evaluate_ranking_function
from .ranking import (
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from .schemas import PhoneticSearchDataset, PhoneticSearchQuery

__all__ = [
    "evaluate_ranking_function",
    "RankingFunctionOutput",
    "rank_by_mora_editdistance",
    "rank_by_vowel_consonant_editdistance",
    "rank_by_phoneme_editdistance",
    "rank_by_kanasim",
    "PhoneticSearchDataset",
    "PhoneticSearchQuery",
    "load_phonetic_search_dataset",
    "load_default_dataset",
    "load_small_dataset",
    "DEFAULT_DATASET_PATH",
    "SMALL_DATASET_QUERY_COUNT",
]
