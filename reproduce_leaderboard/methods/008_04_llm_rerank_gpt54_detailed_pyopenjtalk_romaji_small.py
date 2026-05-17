"""LLMリランク (gpt-5.4, detailed prompt, pyopenjtalk romaji input, small dataset) による評価を実行するスクリプト"""

import json
from pathlib import Path

import pyopenjtalk

from soramimi_phonetic_search_dataset import (
    SMALL_DATASET_QUERY_COUNT,
    RankingFunctionOutput,
    load_default_dataset_for_llm,
    evaluate_ranking_function,
    reasoning_llm_ranking,
)

TOPN = 10
RERANK_INPUT_SIZE = 100
RERANK_BATCH_SIZE = 10
RERANK_INTERVAL = 0
MODEL_NAME = "gpt-5.4"
VOWEL_RATIO = 0.5
PROMPT_TEMPLATE = "detailed"
INPUT_TRANSFORM = "pyopenjtalk_romaji"
PROMPT_INSTRUCTIONS = """
クエリ（Query）と単語一覧（Wordlist）が与えられます。
クエリと発音が似ている順に、単語一覧を並び替えてください。
- 子音より母音の一致を優先してください
- クエリとモウラ数が同じであることを優先してください。ただし促音（ッ）、撥音（ン）、長音（「ー」や直前のカナの母音と同じ単母音モウラ、エ段のカナの直後のイ、オ段のカナの直後のウ、など）の挿入や削除は許容されます。
出力は上位Top N件のインデックスのみ返してください。
"""
EXAMPLE_QUERY = "タロウ"
EXAMPLE_WORDLIST = ["アオ", "アオウヅ", "アノウ", "タキョウ", "タド", "タノ", "タロウ", "タンノ"]
EXAMPLE_TOPN = 5
EXAMPLE_RERANKED = [6, 4, 5, 7, 2]


def transform_text_for_prompt(text: str) -> str:
    phonemes = pyopenjtalk.g2p(text)
    phoneme_text = phonemes if isinstance(phonemes, str) else " ".join(phonemes)
    return " ".join(phoneme_text.lower().split())


def build_prompt_example_suffix() -> str:
    example_wordlist = "\n".join(
        f"{index}. {transform_text_for_prompt(word)}"
        for index, word in enumerate(EXAMPLE_WORDLIST)
    )
    reranked = ", ".join(str(index) for index in EXAMPLE_RERANKED)
    return f"""
Example:
Query: {transform_text_for_prompt(EXAMPLE_QUERY)}
Wordlist:
{example_wordlist}
Top N: {EXAMPLE_TOPN}
Reranked: {reranked}
"""


def restore_original_wordlist(
    reranked_transformed_words: list[str],
    transformed_wordlist: list[str],
    original_wordlist: list[str],
) -> list[str]:
    transformed_to_originals: dict[str, list[str]] = {}
    for transformed, original in zip(transformed_wordlist, original_wordlist):
        transformed_to_originals.setdefault(transformed, []).append(original)

    restored_wordlist = []
    for transformed in reranked_transformed_words:
        originals = transformed_to_originals.get(transformed)
        if not originals:
            raise ValueError(
                f"Unknown transformed word returned by rank_by_reasoning_llm: {transformed}"
            )
        restored_wordlist.append(originals.pop(0))
    return restored_wordlist


def build_ranking_function(dataset) -> callable:
    prompt_example_suffix = build_prompt_example_suffix()

    def ranking_func(
        query_texts: list[str],
        wordlists: list[list[str]],
    ) -> RankingFunctionOutput:
        transformed_query_texts = [
            transform_text_for_prompt(query) for query in query_texts
        ]
        transformed_wordlists = [
            [transform_text_for_prompt(word) for word in wordlist]
            for wordlist in wordlists
        ]

        reranked = reasoning_llm_ranking.rank_by_reasoning_llm(
            query_texts=transformed_query_texts,
            wordlist_texts=transformed_wordlists,
            topn=TOPN,
            model_name=MODEL_NAME,
            reasoning_effort="none",
            batch_size=RERANK_BATCH_SIZE,
            rerank_interval=RERANK_INTERVAL,
            prompt_instructions=PROMPT_INSTRUCTIONS,
            prompt_example_suffix=prompt_example_suffix,
        )
        restored_wordlists = [
            restore_original_wordlist(reranked_wordlist, transformed_wordlist, original_wordlist)
            for reranked_wordlist, transformed_wordlist, original_wordlist in zip(
                reranked.ranked_wordlists,
                transformed_wordlists,
                wordlists,
            )
        ]
        return RankingFunctionOutput(
            ranked_wordlists=restored_wordlists,
            metrics_metadata=reranked.metrics_metadata,
        )

    return ranking_func


def main():
    output_dir = Path(__file__).parent.parent / "results_small"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "008_04_llm_rerank_gpt54_detailed_pyopenjtalk_romaji.json"

    dataset = load_default_dataset_for_llm(
        query_limit=SMALL_DATASET_QUERY_COUNT,
        wordlist_size=RERANK_INPUT_SIZE,
    )
    results = evaluate_ranking_function(
        ranking_func=build_ranking_function(dataset),
        topn=TOPN,
        dataset=dataset,
    )

    results.parameters.rank_func = "vowel_consonant"
    results.parameters.metadata.update(
        {
            "vowel_ratio": VOWEL_RATIO,
            "rerank": True,
            "rerank_input_size": RERANK_INPUT_SIZE,
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "rerank_interval": RERANK_INTERVAL,
            "rerank_model_name": MODEL_NAME,
            "rerank_reasoning_effort": "none",
            "rerank_prompt_template": PROMPT_TEMPLATE,
            "rerank_prompt_instructions": PROMPT_INSTRUCTIONS.strip(),
            "rerank_prompt_example_suffix": build_prompt_example_suffix().strip(),
            "rerank_input_transform": INPUT_TRANSFORM,
            "dataset_size": "small",
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
