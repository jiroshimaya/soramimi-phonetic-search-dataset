"""
Soramimi Phonetic Search Dataset package
"""

from .dataset import (
    DEFAULT_DATASET_PATH,
    load_default_dataset,
    load_phonetic_search_dataset,
)
from .distinctive_feature import rank_by_distinctive_feature_distance
from .evaluate import evaluate_ranking_function, evaluate_ranking_function_with_details
from .ranking import (
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from .schemas import PhoneticSearchDataset, PhoneticSearchQuery

__all__ = [
    "evaluate_ranking_function",
    "evaluate_ranking_function_with_details",
    "rank_by_mora_editdistance",
    "rank_by_vowel_consonant_editdistance",
    "rank_by_phoneme_editdistance",
    "rank_by_distinctive_feature_distance",
    "rank_by_kanasim",
    "PhoneticSearchDataset",
    "PhoneticSearchQuery",
    "load_phonetic_search_dataset",
    "load_default_dataset",
    "DEFAULT_DATASET_PATH",
]
