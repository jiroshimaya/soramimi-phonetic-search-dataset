from soramimi_phonetic_search_dataset.schemas import (
    PhoneticSearchDataset,
    PhoneticSearchQuery,
    PhoneticSearchQueryWithWordlist,
    PhoneticSearchWordlistDataset,
)
from soramimi_phonetic_search_dataset import subset_labeling


def test_label_wordlist_dataset_subsets_uses_staged_tie_aware_rule(monkeypatch):
    dataset = PhoneticSearchWordlistDataset(
        queries=[
            PhoneticSearchQueryWithWordlist(
                query="q_easy",
                wordlist=["p_easy", "n_easy"],
                positive_words=["p_easy"],
            ),
            PhoneticSearchQueryWithWordlist(
                query="q_medium",
                wordlist=["p_medium", "n_medium"],
                positive_words=["p_medium"],
            ),
            PhoneticSearchQueryWithWordlist(
                query="q_hard",
                wordlist=["p_hard", "n_hard"],
                positive_words=["p_hard"],
            ),
        ]
    )

    monkeypatch.setattr(
        subset_labeling,
        "rank_by_mora_editdistance",
        lambda query_texts, wordlists: [
            ["p_easy", "n_easy"],
            ["p_medium", "n_medium"],
            ["n_hard", "p_hard"],
        ],
    )
    monkeypatch.setattr(
        subset_labeling,
        "rank_by_vowel_consonant_editdistance",
        lambda query_texts, wordlists, vowel_ratio: [
            ["p_easy", "n_easy"],
            ["p_medium", "n_medium"],
            ["p_hard", "n_hard"],
        ],
    )
    monkeypatch.setattr(
        subset_labeling,
        "_score_by_mora_editdistance",
        lambda query_text, wordlist: {
            "q_easy": {"p_easy": 0, "n_easy": 1},
            "q_medium": {"p_medium": 0, "n_medium": 0},
            "q_hard": {"p_hard": 1, "n_hard": 0},
        }[query_text],
    )
    monkeypatch.setattr(
        subset_labeling,
        "_score_by_vowel_consonant_editdistance",
        lambda query_text, wordlist, vowel_ratio: {
            "q_easy": {"p_easy": 0, "n_easy": 1},
            "q_medium": {"p_medium": 0, "n_medium": 1},
            "q_hard": {"p_hard": 0, "n_hard": 0},
        }[query_text],
    )

    diagnostics = subset_labeling.label_wordlist_dataset_subsets(dataset, topn=1)

    assert [diagnostic.subset for diagnostic in diagnostics] == [
        "easy",
        "medium",
        "hard",
    ]
    assert diagnostics[1].current_mora_hit is True
    assert diagnostics[1].stable_mora_hit is False
    assert diagnostics[1].stable_vowel_hit is True
    assert diagnostics[2].current_vowel_hit is True
    assert diagnostics[2].stable_vowel_hit is False


def test_relabel_dataset_subsets_updates_queries_and_metadata(monkeypatch):
    dataset = PhoneticSearchDataset(
        queries=[
            PhoneticSearchQuery(query="q_easy", positive=["p_easy"]),
            PhoneticSearchQuery(query="q_medium", positive=["p_medium"]),
            PhoneticSearchQuery(query="q_hard", positive=["p_hard"]),
        ],
        words=[
            "p_easy",
            "n_easy",
            "p_medium",
            "n_medium",
            "p_hard",
            "n_hard",
        ],
        metadata={"query_count": 3},
    )

    monkeypatch.setattr(
        subset_labeling,
        "label_wordlist_dataset_subsets",
        lambda dataset, topn, vowel_ratio: [
            subset_labeling.SubsetLabelingDiagnostic(
                query="q_easy",
                subset="easy",
                current_mora_hit=True,
                stable_mora_hit=True,
                current_vowel_hit=True,
                stable_vowel_hit=True,
            ),
            subset_labeling.SubsetLabelingDiagnostic(
                query="q_medium",
                subset="medium",
                current_mora_hit=True,
                stable_mora_hit=False,
                current_vowel_hit=True,
                stable_vowel_hit=True,
            ),
            subset_labeling.SubsetLabelingDiagnostic(
                query="q_hard",
                subset="hard",
                current_mora_hit=False,
                stable_mora_hit=False,
                current_vowel_hit=True,
                stable_vowel_hit=False,
            ),
        ],
    )

    relabeled = subset_labeling.relabel_dataset_subsets(dataset)

    assert [query.subset for query in relabeled.queries] == ["easy", "medium", "hard"]
    assert relabeled.metadata["query_count"] == 3
    assert relabeled.metadata["subset_definition"]["decision_style"] == "staged"
    assert "tie_break_policy" in relabeled.metadata["subset_definition"]
