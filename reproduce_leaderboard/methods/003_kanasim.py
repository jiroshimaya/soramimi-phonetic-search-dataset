"""
KanaSim編集距離による評価を実行するスクリプト
"""

import json
from pathlib import Path

from soramimi_phonetic_search_dataset import (
    evaluate_ranking_function,
    load_default_dataset,
    rank_by_kanasim,
)


def main():
    # 結果の出力先を作成
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "003_kanasim.json"

    def ranking_func(
        query_texts: list[str], wordlists: list[list[str]]
    ) -> list[list[str]]:
        return rank_by_kanasim(
            query_texts,
            wordlists,
            vowel_ratio=0.8,
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_func,
        topn=10,
        dataset=load_default_dataset(),
    )

    results.parameters.rank_func = "kanasim"
    results.parameters.metadata["vowel_ratio"] = 0.8

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.__dict__,
        )

    print("Recall: ", results.metrics.recall)
    print("Execution time: ", results.metrics.execution_time)


if __name__ == "__main__":
    main()
