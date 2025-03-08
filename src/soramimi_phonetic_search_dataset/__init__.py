from .evaluate import evaluate_ranking_function
from .ranking import (
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from .schemas import PhoneticSearchDataset, PhoneticSearchQuery

__all__ = [
    "evaluate_ranking_function",
    "rank_by_mora_editdistance",
    "rank_by_vowel_consonant_editdistance",
    "rank_by_phoneme_editdistance",
    "rank_by_kanasim",
    "PhoneticSearchDataset",
    "PhoneticSearchQuery",
]


def hello() -> str:
    return "Hello from soramimi-phonetic-search-dataset!"
