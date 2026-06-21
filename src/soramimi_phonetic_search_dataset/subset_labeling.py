from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import editdistance as ed
import jamorasep

from .base_ranking import (
    rank_by_mora_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from .dataset import build_wordlist_dataset
from .schemas import (
    PhoneticSearchDataset,
    PhoneticSearchQueryWithWordlist,
    PhoneticSearchWordlistDataset,
)

DEFAULT_SUBSET_TOPN = 10
DEFAULT_SUBSET_VOWEL_RATIO = 0.8
SubsetLabel = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class SubsetLabelingDiagnostic:
    query: str
    subset: SubsetLabel
    current_mora_hit: bool
    stable_mora_hit: bool
    current_vowel_hit: bool
    stable_vowel_hit: bool


def _score_by_mora_editdistance(
    query_text: str,
    wordlist: list[str],
) -> dict[str, int]:
    query_mora = jamorasep.parse(query_text)
    return {
        word: ed.eval(query_mora, jamorasep.parse(word))
        for word in wordlist
    }


def _score_by_vowel_consonant_editdistance(
    query_text: str,
    wordlist: list[str],
    *,
    vowel_ratio: float,
) -> dict[str, float]:
    def _parse_word(text: str) -> tuple[list[str], list[str]]:
        mora = jamorasep.parse(text, output_format="simple-ipa")
        vowels = [m[-1] for m in mora]
        consonants = [m[:-1] if m[:-1] else "sp" for m in mora]
        return vowels, consonants

    query_vowels, query_consonants = _parse_word(query_text)
    scores = {}
    for word in wordlist:
        word_vowels, word_consonants = _parse_word(word)
        vowel_distance = ed.eval(query_vowels, word_vowels)
        consonant_distance = ed.eval(query_consonants, word_consonants)
        scores[word] = vowel_distance * vowel_ratio + consonant_distance * (
            1 - vowel_ratio
        )
    return scores


def _build_pessimistic_ranking(
    wordlist: list[str],
    *,
    score_by_word: Mapping[str, int | float],
    positive_words: list[str],
) -> list[str]:
    positive_word_set = set(positive_words)
    return sorted(
        wordlist,
        key=lambda word: (score_by_word[word], word in positive_word_set),
    )


def _has_perfect_recall_at_k(
    ranked_wordlist: list[str],
    positive_words: list[str],
    *,
    topn: int,
) -> bool:
    return set(ranked_wordlist[:topn]) >= set(positive_words)


def label_wordlist_dataset_subsets(
    dataset: PhoneticSearchWordlistDataset,
    *,
    topn: int = DEFAULT_SUBSET_TOPN,
    vowel_ratio: float = DEFAULT_SUBSET_VOWEL_RATIO,
) -> list[SubsetLabelingDiagnostic]:
    query_texts = [query.query for query in dataset.queries]
    wordlists = [query.wordlist for query in dataset.queries]

    mora_rankings = rank_by_mora_editdistance(query_texts, wordlists)
    vowel_rankings = rank_by_vowel_consonant_editdistance(
        query_texts,
        wordlists,
        vowel_ratio=vowel_ratio,
    )

    diagnostics = []
    for query, mora_ranking, vowel_ranking in zip(
        dataset.queries,
        mora_rankings,
        vowel_rankings,
    ):
        mora_scores = _score_by_mora_editdistance(query.query, query.wordlist)
        vowel_scores = _score_by_vowel_consonant_editdistance(
            query.query,
            query.wordlist,
            vowel_ratio=vowel_ratio,
        )

        current_mora_hit = _has_perfect_recall_at_k(
            mora_ranking,
            query.positive_words,
            topn=topn,
        )
        stable_mora_hit = _has_perfect_recall_at_k(
            _build_pessimistic_ranking(
                query.wordlist,
                score_by_word=mora_scores,
                positive_words=query.positive_words,
            ),
            query.positive_words,
            topn=topn,
        )
        current_vowel_hit = _has_perfect_recall_at_k(
            vowel_ranking,
            query.positive_words,
            topn=topn,
        )
        stable_vowel_hit = _has_perfect_recall_at_k(
            _build_pessimistic_ranking(
                query.wordlist,
                score_by_word=vowel_scores,
                positive_words=query.positive_words,
            ),
            query.positive_words,
            topn=topn,
        )

        if stable_mora_hit:
            subset = "easy"
        elif stable_vowel_hit:
            subset = "medium"
        else:
            subset = "hard"

        diagnostics.append(
            SubsetLabelingDiagnostic(
                query=query.query,
                subset=subset,
                current_mora_hit=current_mora_hit,
                stable_mora_hit=stable_mora_hit,
                current_vowel_hit=current_vowel_hit,
                stable_vowel_hit=stable_vowel_hit,
            )
        )

    return diagnostics


def build_subset_definition_metadata(
    *,
    topn: int = DEFAULT_SUBSET_TOPN,
    vowel_ratio: float = DEFAULT_SUBSET_VOWEL_RATIO,
) -> dict[str, object]:
    vowel_method = f"vowel_consonant_editdistance(vowel_ratio={vowel_ratio})"
    return {
        "labels": ["easy", "medium", "hard"],
        "k": topn,
        "priority": ["mora_editdistance", vowel_method],
        "decision_style": "staged",
        "tie_break_policy": (
            "same-score ties are resolved pessimistically by ordering negatives "
            "before positives"
        ),
        "rule": (
            "moraで安定してrecall@k=1ならeasy、そうでなければ"
            f"{vowel_method}で安定して1ならmedium、どちらでも安定して"
            "満たさなければhard"
        ),
    }


def relabel_dataset_subsets(
    dataset: PhoneticSearchDataset,
    *,
    topn: int = DEFAULT_SUBSET_TOPN,
    vowel_ratio: float = DEFAULT_SUBSET_VOWEL_RATIO,
) -> PhoneticSearchDataset:
    wordlist_dataset = build_wordlist_dataset(dataset)
    diagnostics = label_wordlist_dataset_subsets(
        wordlist_dataset,
        topn=topn,
        vowel_ratio=vowel_ratio,
    )
    subset_by_query: dict[str, SubsetLabel] = {
        diagnostic.query: diagnostic.subset for diagnostic in diagnostics
    }
    relabeled_queries = [
        replace(query, subset=subset_by_query[query.query]) for query in dataset.queries
    ]
    metadata = {
        **dataset.metadata,
        "subset_definition": build_subset_definition_metadata(
            topn=topn,
            vowel_ratio=vowel_ratio,
        ),
    }
    return PhoneticSearchDataset(
        queries=relabeled_queries,
        words=dataset.words,
        metadata=metadata,
    )


def attach_subset_labels_to_wordlist_dataset(
    dataset: PhoneticSearchWordlistDataset,
    *,
    topn: int = DEFAULT_SUBSET_TOPN,
    vowel_ratio: float = DEFAULT_SUBSET_VOWEL_RATIO,
) -> PhoneticSearchWordlistDataset:
    diagnostics = label_wordlist_dataset_subsets(
        dataset,
        topn=topn,
        vowel_ratio=vowel_ratio,
    )
    subset_by_query: dict[str, SubsetLabel] = {
        diagnostic.query: diagnostic.subset for diagnostic in diagnostics
    }
    return PhoneticSearchWordlistDataset(
        queries=[
            PhoneticSearchQueryWithWordlist(
                query=query.query,
                wordlist=query.wordlist,
                positive_words=query.positive_words,
                subset=subset_by_query[query.query],
            )
            for query in dataset.queries
        ],
        metadata={
            **dataset.metadata,
            "subset_definition": build_subset_definition_metadata(
                topn=topn,
                vowel_ratio=vowel_ratio,
            ),
        },
    )
