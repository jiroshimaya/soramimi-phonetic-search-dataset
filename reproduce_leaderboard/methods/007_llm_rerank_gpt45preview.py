"""LLMリランク (gpt-4.5-preview) による評価を実行するスクリプト"""

import json
from pathlib import Path

from soramimi_phonetic_search_dataset import (
    RankingFunctionOutput,
    evaluate_ranking_function,
    load_default_dataset_for_llm,
    rank_by_llm,
)


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "007_llm_rerank_gpt45preview.json"

    dataset = load_default_dataset_for_llm(wordlist_size=100)

    def ranking_func(
        query_texts: list[str], wordlists: list[list[str]]
    ) -> RankingFunctionOutput:
        return rank_by_llm(
            query_texts,
            wordlists,
            topn=10,
            model_name="gpt-4.5-preview",
            batch_size=2,
            rerank_interval=1,
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_func,
        topn=10,
        dataset=dataset,
    )

    results.parameters.rank_func = "llm_rerank"
    results.parameters.metadata.update(
        {
            "rerank_model_name": "gpt-4.5-preview",
            "rerank_input_size": 100,
            "rerank_batch_size": 2,
            "rerank_interval": 1,
        }
    )

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
