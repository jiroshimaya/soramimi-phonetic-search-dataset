"""LLMリランク (gpt-5.4) による評価を実行するスクリプト"""

import json
from pathlib import Path

from soramimi_phonetic_search_dataset import (
    RankingFunctionOutput,
    evaluate_ranking_function,
    load_default_dataset_for_llm,
    reasoning_llm_ranking,
)


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "008_llm_rerank_gpt54.json"

    dataset = load_default_dataset_for_llm(wordlist_size=100)

    def ranking_func(
        query_texts: list[str], wordlists: list[list[str]]
    ) -> RankingFunctionOutput:
        return reasoning_llm_ranking.rank_by_reasoning_llm(
            query_texts,
            wordlists,
            topn=10,
            model_name="gpt-5.4",
            reasoning_effort="none",
            batch_size=10,
            rerank_interval=0,
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_func,
        topn=10,
        dataset=dataset,
    )

    results.parameters.rank_func = "llm_rerank"
    results.parameters.metadata.update(
        {
            "rerank_model_name": "gpt-5.4",
            "rerank_reasoning_effort": "none",
            "rerank_input_size": 100,
            "rerank_batch_size": 10,
            "rerank_interval": 0,
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
