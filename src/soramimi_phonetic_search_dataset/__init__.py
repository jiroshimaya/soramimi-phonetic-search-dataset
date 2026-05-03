"""
Soramimi Phonetic Search Dataset package
"""

from .dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LLM_WORDLIST_SIZE,
    SMALL_DATASET_QUERY_COUNT,
    build_wordlist_dataset,
    build_wordlist_dataset_for_llm,
    load_default_dataset,
    load_default_dataset_for_llm,
    load_phonetic_search_dataset,
    load_small_dataset,
)
from .evaluate import RankingFunctionOutput, evaluate_ranking_function
from .base_ranking import (
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from .llm_ranking import rank_by_llm
from .schemas import (
    PhoneticSearchDataset,
    PhoneticSearchQuery,
    PhoneticSearchQueryWithWordlist,
    PhoneticSearchWordlistDataset,
)

__all__ = [
    "evaluate_ranking_function",
    "RankingFunctionOutput",
    "rank_by_llm",
    "rank_by_mora_editdistance",
    "rank_by_vowel_consonant_editdistance",
    "rank_by_phoneme_editdistance",
    "rank_by_kanasim",
    "PhoneticSearchDataset",
    "PhoneticSearchQuery",
    "PhoneticSearchQueryWithWordlist",
    "PhoneticSearchWordlistDataset",
    "build_wordlist_dataset",
    "load_phonetic_search_dataset",
    "load_default_dataset",
    "load_default_dataset_for_llm",
    "load_small_dataset",
    "build_wordlist_dataset_for_llm",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_LLM_WORDLIST_SIZE",
    "SMALL_DATASET_QUERY_COUNT",
]
