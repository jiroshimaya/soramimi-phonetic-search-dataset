"""
LLMリランク (gpt-5.4, step-by-step prompt) による評価を実行するスクリプト
"""

import json
from pathlib import Path

from soramimi_phonetic_search_dataset import (
    RankingFunctionOutput,
    evaluate_ranking_function,
    load_default_dataset_for_llm,
    reasoning_llm_ranking,
)

TOPN = 10
RERANK_INPUT_SIZE = 100
RERANK_BATCH_SIZE = 10
RERANK_INTERVAL = 0
MODEL_NAME = "gpt-5.4"
PROMPT_INSTRUCTIONS = """
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
以下の手順で判断してください。
- 1. クエリと比較対象単語から促音（ッ）、撥音（ン）、長音（ー）を削除
- 2. クエリと比較対象単語をそれぞれ小文字ローマ字に直す
- 3. 同じ母音が連続していれば2文字目以降を削除する。例えば「k a a」は「k a」にする。「カア」は実質「カー」であるため長音の削除に相当。同様に「ei」「ou」についてはそれぞれ「e」「o」にする。これも「エイ」「オウ」は実質「エー」「オー」であるため長音の削除に対応する
- 4. 母音（aiueo）の並びが一致していることを優先し、母音の一致が同程度であればなるべく子音が似ているものを、より発音が似ているとする。
出力は上位Top N件のインデックスのみ返してください。
"""
PROMPT_EXAMPLE_SUFFIX = """
Example:
Query: タロウ
Wordlist:
0. アオ
1. アオウヅ
2. アノウ
3. タキョウ
4. タド
5. タノ
6. タロウ
7. タンノ
Top N: 5
Reranked: 6, 4, 5, 7, 2
"""


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "008_03_llm_rerank_gpt54_step_by_step.json"

    dataset = load_default_dataset_for_llm(wordlist_size=RERANK_INPUT_SIZE)

    def ranking_func(
        query_texts: list[str], wordlists: list[list[str]]
    ) -> RankingFunctionOutput:
        return reasoning_llm_ranking.rank_by_reasoning_llm(
            query_texts,
            wordlists,
            topn=TOPN,
            model_name=MODEL_NAME,
            reasoning_effort="none",
            batch_size=RERANK_BATCH_SIZE,
            rerank_interval=RERANK_INTERVAL,
            prompt_instructions=PROMPT_INSTRUCTIONS,
            prompt_example_suffix=PROMPT_EXAMPLE_SUFFIX,
        )

    results = evaluate_ranking_function(
        ranking_func=ranking_func,
        topn=TOPN,
        dataset=dataset,
    )

    results.parameters.rank_func = "llm_rerank"
    results.parameters.metadata.update(
        {
            "rerank": True,
            "rerank_model_name": MODEL_NAME,
            "rerank_reasoning_effort": "none",
            "rerank_prompt_template": "step_by_step",
            "rerank_prompt_instructions": PROMPT_INSTRUCTIONS.strip(),
            "rerank_prompt_example_suffix": PROMPT_EXAMPLE_SUFFIX.strip(),
            "rerank_user_prompt_template": None,
            "rerank_include_thoughts": False,
            "rerank_input_transform": "none",
            "rerank_input_size": RERANK_INPUT_SIZE,
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "rerank_interval": RERANK_INTERVAL,
            "rerank_backend": "litellm",
            "rerank_batch_id": None,
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
