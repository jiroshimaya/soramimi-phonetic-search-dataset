import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from soramimi_phonetic_search_dataset.dataset import (
    build_wordlist_dataset,
    load_phonetic_search_dataset,
)
from soramimi_phonetic_search_dataset.subset_labeling import (
    DEFAULT_SUBSET_TOPN,
    DEFAULT_SUBSET_VOWEL_RATIO,
    relabel_dataset_subsets,
)

SubsetLabel = Literal["easy", "medium", "hard"]


class ResultEntry(TypedDict):
    query: str
    positive_words: list[str]
    ranked_words: list[str]


def _calculate_recall_by_subset(
    result_entries: list[ResultEntry],
    subset_by_query: dict[str, SubsetLabel],
    *,
    topn: int,
) -> dict[str, float]:
    hits_by_subset: dict[str, list[float]] = {}
    for result in result_entries:
        query = result["query"]
        subset = subset_by_query[query]
        positive_words = set(result["positive_words"])
        ranked_words = result["ranked_words"][:topn]
        hit = 1.0 if set(ranked_words) >= positive_words else 0.0
        hits_by_subset.setdefault(subset, []).append(hit)
    return {
        subset: sum(hits) / len(hits)
        for subset, hits in sorted(hits_by_subset.items())
    }


def _update_result_file(
    result_path: Path,
    subset_by_query: dict[str, SubsetLabel],
    *,
    topn: int,
) -> None:
    result_data = cast(dict[str, Any], json.loads(result_path.read_text()))
    metrics_metadata = result_data.setdefault("metrics", {}).setdefault("metadata", {})
    metrics_metadata["recall_by_subset"] = _calculate_recall_by_subset(
        cast(list[ResultEntry], result_data["results"]),
        subset_by_query,
        topn=topn,
    )
    result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged tie-aware subset labels and recall_by_subset metadata updater."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("src/soramimi_phonetic_search_dataset/data/baseball.json"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=[
            Path("reproduce_leaderboard/results"),
            Path("reproduce_leaderboard/results_small"),
        ],
    )
    parser.add_argument("--topn", type=int, default=DEFAULT_SUBSET_TOPN)
    parser.add_argument(
        "--vowel-ratio",
        type=float,
        default=DEFAULT_SUBSET_VOWEL_RATIO,
    )
    args = parser.parse_args()

    dataset = load_phonetic_search_dataset(str(args.dataset))
    relabeled_dataset = relabel_dataset_subsets(
        dataset,
        topn=args.topn,
        vowel_ratio=args.vowel_ratio,
    )
    args.dataset.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": query.query,
                        "positive": query.positive,
                        "hard_negatives": query.hard_negatives,
                        "subset": query.subset,
                    }
                    for query in relabeled_dataset.queries
                ],
                "words": relabeled_dataset.words,
                "metadata": relabeled_dataset.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    wordlist_dataset = build_wordlist_dataset(relabeled_dataset)
    subset_by_query: dict[str, SubsetLabel] = {
        query.query: cast(SubsetLabel, query.subset) for query in wordlist_dataset.queries
    }
    for results_dir in args.results_dir:
        if not results_dir.exists():
            continue
        for result_path in sorted(results_dir.glob("*.json")):
            _update_result_file(
                result_path,
                subset_by_query,
                topn=args.topn,
            )

    counts = Counter(query.subset for query in relabeled_dataset.queries)
    print(json.dumps({"subset_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
