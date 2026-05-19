import argparse
import json
from typing import Callable

from soramimi_phonetic_search_dataset import (
    evaluate_ranking_function,
    load_default_dataset,
    load_small_dataset,
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    reasoning_llm_ranking,
    rank_by_vowel_consonant_editdistance,
)
from soramimi_phonetic_search_dataset.evaluate import RankingFunctionOutput


def prepare_rerank_candidates(
    base_ranked_wordlists: list[list[str]],
    positive_texts: list[list[str]],
    rerank_input_size: int,
) -> list[list[str]]:
    topk_ranked_wordlists = []
    for wordlist, positive_text in zip(base_ranked_wordlists, positive_texts):
        topk = wordlist[:rerank_input_size]
        missing_positive_count = sum(1 for text in positive_text if text not in topk)
        if missing_positive_count > 0:
            topk = topk[:-missing_positive_count]
            for text in positive_text:
                if text not in topk:
                    topk.append(text)
        topk_ranked_wordlists.append(sorted(topk))
    return topk_ranked_wordlists


def create_reranking_function(
    base_rank_func: Callable[[list[str], list[list[str]]], list[list[str]]],
    rerank_input_size: int,
    rerank_model_name: str,
    rerank_reasoning_effort: str | None,
    rerank_prompt_instructions: str | None,
    rerank_prompt_example_suffix: str | None,
    rerank_user_prompt_template: str | None,
    rerank_include_thoughts: bool,
    rerank_input_transform: str,
    rerank_batch_size: int,
    rerank_interval: int,
    topn: int,
    positive_texts: list[list[str]],
    **base_rank_kwargs,
) -> Callable[[list[str], list[list[str]]], list[list[str]]]:
    """
    ベースのランキング関数とLLMによるリランクを組み合わせた関数を作成する

    Args:
        base_rank_func: ベースのランキング関数
        rerank_input_size: リランクに使用する候補数
        rerank_model_name: リランクに使用するモデル名
        rerank_reasoning_effort: リランクに使用するreasoning effort
        rerank_input_transform: リランク前に query / candidate に適用する入力変換
        rerank_batch_size: リランクのバッチサイズ
        rerank_interval: リランクのインターバル
        topn: 最終的な出力数
        positive_texts: 各クエリに対する正解リスト
        **base_rank_kwargs: ベースのランキング関数に渡す追加の引数

    Returns:
        組み合わせたランキング関数
    """

    def combined_rank_func(query_texts: list[str], wordlists: list[list[str]]):
        # ベースのランキングを実行
        base_ranked_wordlists = base_rank_func(
            query_texts, wordlists, **base_rank_kwargs
        )

        # 上位N件を取得してリランク
        topk_ranked_wordlists = prepare_rerank_candidates(
            base_ranked_wordlists,
            positive_texts,
            rerank_input_size,
        )

        rerank_output = reasoning_llm_ranking.rank_by_reasoning_llm(
            query_texts,
            topk_ranked_wordlists,
            topn=topn,
            model_name=rerank_model_name,
            reasoning_effort=rerank_reasoning_effort,
            prompt_instructions=rerank_prompt_instructions,
            prompt_example_suffix=rerank_prompt_example_suffix,
            user_prompt_template=rerank_user_prompt_template,
            include_thoughts=rerank_include_thoughts,
            input_transform=rerank_input_transform,
            batch_size=rerank_batch_size,
            rerank_interval=rerank_interval,
        )
        return RankingFunctionOutput(
            ranked_wordlists=rerank_output.ranked_wordlists,
            result_metadata=(
                rerank_output.result_metadata if rerank_include_thoughts else None
            ),
            metrics_metadata=rerank_output.metrics_metadata,
        )

    return combined_rank_func


def get_default_output_path(
    rank_func: str,
    topn: int,
    dataset_size: str = "default",
    query_limit: int | None = None,
    query_offset: int = 0,
    rerank: bool = False,
    rerank_topn: int = 10,
    rerank_model_name: str = "gpt-4o-mini",
    rerank_reasoning_effort: str | None = None,
    rerank_include_thoughts: bool = False,
    rerank_input_transform: str = "none",
) -> str:
    suffix = f"_{rank_func}_top{topn}"
    if rerank:
        # スラッシュを含む場合はハイフンに変換
        model_name_safe = rerank_model_name.replace("/", "-")
        suffix += f"_reranked_top{rerank_topn}_model{model_name_safe}"
        if rerank_reasoning_effort:
            suffix += f"_reasoning{rerank_reasoning_effort}"
        if rerank_include_thoughts:
            suffix += "_withthoughts"
        if rerank_input_transform != "none":
            suffix += f"_transform{rerank_input_transform}"
    if query_limit is not None:
        suffix += f"_querylimit{query_limit}"
    if query_offset != 0:
        suffix += f"_queryoffset{query_offset}"
    if dataset_size != "default":
        suffix += f"_dataset{dataset_size}"
    return f"output{suffix}.json"


def load_dataset_for_evaluation(
    dataset_size: str,
    query_limit: int | None,
    query_offset: int,
) -> tuple:
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be a positive integer")
    if query_offset < 0:
        raise ValueError("query_offset must be a non-negative integer")
    if dataset_size == "small":
        if query_limit is not None or query_offset != 0:
            raise ValueError(
                "query_limit/query_offset cannot be used with dataset_size=small"
            )
        return load_small_dataset(), None, None
    return (
        load_default_dataset(query_limit=query_limit, query_offset=query_offset),
        query_limit,
        query_offset,
    )


def _read_optional_text_file(path: str | None) -> str | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Evaluate phonetic search dataset.")
    parser.add_argument(
        "-r",
        "--rank_func",
        type=str,
        choices=["kanasim", "vowel_consonant", "phoneme", "mora"],
        default="vowel_consonant",
        help="Rank function: kanasim, vowel_consonant, phoneme, mora",
    )
    parser.add_argument(
        "-n",
        "--topn",
        type=int,
        default=10,
        help="Top N",
    )
    parser.add_argument(
        "-vr",
        "--vowel_ratio",
        type=float,
        default=0.5,
        help="Vowel ratio, which is used only when rank_func is vowel_consonant",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["default", "small"],
        default="default",
        help="Dataset size: default (150 queries) or small (10 queries)",
    )
    parser.add_argument(
        "--query_limit",
        type=int,
        help="Use only the first N queries from the default dataset",
    )
    parser.add_argument(
        "--query_offset",
        type=int,
        default=0,
        help="Skip the first N queries from the default dataset",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the wordlists by LLM",
    )
    parser.add_argument(
        "--rerank_input_size",
        type=int,
        default=100,
        help="Number of top candidates to consider for reranking",
    )
    parser.add_argument(
        "--rerank_batch_size",
        type=int,
        default=10,
        help="Batch size for reranking",
    )
    parser.add_argument(
        "--rerank_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name for reranking",
    )
    parser.add_argument(
        "--rerank_reasoning_effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for reranking models that support it",
    )
    parser.add_argument(
        "--rerank_prompt_instructions_path",
        type=str,
        help="Path to a text file containing prompt instructions for LLM reranking",
    )
    parser.add_argument(
        "--rerank_prompt_example_suffix_path",
        type=str,
        help="Path to a text file containing prompt example suffix for LLM reranking",
    )
    parser.add_argument(
        "--rerank_user_prompt_template_path",
        type=str,
        help="Path to a text file containing user prompt template for LLM reranking",
    )
    parser.add_argument(
        "--rerank_include_thoughts",
        action="store_true",
        help="Require structured outputs to include thoughts in addition to reranked",
    )
    parser.add_argument(
        "--rerank_input_transform",
        type=str,
        choices=["none"],
        default="none",
        help="Transform query/candidates before reranking",
    )
    parser.add_argument(
        "--rerank_interval",
        type=int,
        default=0,
        help="Sleep interval in seconds between reranking batches",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save results to file",
    )
    args = parser.parse_args()
    # ベースのランキング関数を選択
    if args.rank_func == "kanasim":
        base_rank_func = rank_by_kanasim
        rank_kwargs = {"vowel_ratio": args.vowel_ratio}
    elif args.rank_func == "mora":
        base_rank_func = rank_by_mora_editdistance
        rank_kwargs = {}
    elif args.rank_func == "vowel_consonant":
        base_rank_func = rank_by_vowel_consonant_editdistance
        rank_kwargs = {"vowel_ratio": args.vowel_ratio}
    elif args.rank_func == "phoneme":
        base_rank_func = rank_by_phoneme_editdistance
        rank_kwargs = {}

    if args.output_file_path:
        output_path = args.output_file_path
    else:
        output_path = get_default_output_path(
            args.rank_func,
            args.topn,
            args.dataset_size,
            args.query_limit,
            args.query_offset,
            args.rerank,
            args.rerank_input_size,
            args.rerank_model_name,
            args.rerank_reasoning_effort,
            args.rerank_include_thoughts,
            args.rerank_input_transform,
        )
    try:
        dataset, effective_query_limit, effective_query_offset = (
            load_dataset_for_evaluation(
                args.dataset_size,
                args.query_limit,
                args.query_offset,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    rerank_prompt_instructions = _read_optional_text_file(
        args.rerank_prompt_instructions_path
    )
    rerank_prompt_example_suffix = _read_optional_text_file(
        args.rerank_prompt_example_suffix_path
    )
    rerank_user_prompt_template = _read_optional_text_file(
        args.rerank_user_prompt_template_path
    )
    # リランクが必要な場合は組み合わせた関数を作成
    if args.rerank:
        positive_texts = [query.positive_words for query in dataset.queries]

        _rank_func = create_reranking_function(
            base_rank_func=base_rank_func,
            rerank_input_size=args.rerank_input_size,
            rerank_model_name=args.rerank_model_name,
            rerank_reasoning_effort=args.rerank_reasoning_effort,
            rerank_prompt_instructions=rerank_prompt_instructions,
            rerank_prompt_example_suffix=rerank_prompt_example_suffix,
            rerank_user_prompt_template=rerank_user_prompt_template,
            rerank_include_thoughts=args.rerank_include_thoughts,
            rerank_input_transform=args.rerank_input_transform,
            rerank_batch_size=args.rerank_batch_size,
            rerank_interval=args.rerank_interval,
            topn=args.topn,
            positive_texts=positive_texts,
            **rank_kwargs,
        )

        # 警告を回避するためdefでラップ
        def rank_func(query_texts, wordlists):
            return _rank_func(query_texts, wordlists)
    else:
        # 警告を回避するためdefでラップ
        def rank_func(query_texts, wordlists):
            return base_rank_func(query_texts, wordlists, **rank_kwargs)

    # 評価を実行
    results = evaluate_ranking_function(
        ranking_func=rank_func,
        topn=args.topn,
        dataset=dataset,
    )

    # パラメータを設定
    results.parameters.rank_func = args.rank_func
    results.parameters.metadata.update(
        {
            "query_limit": effective_query_limit,
            "query_offset": effective_query_offset,
            "vowel_ratio": (
                args.vowel_ratio
                if args.rank_func in ["kanasim", "vowel_consonant"]
                else None
            ),
            "rerank": args.rerank,
            "rerank_model_name": args.rerank_model_name if args.rerank else None,
            "rerank_reasoning_effort": (
                args.rerank_reasoning_effort if args.rerank else None
            ),
            "rerank_prompt_instructions": (
                rerank_prompt_instructions.strip()
                if args.rerank and rerank_prompt_instructions
                else None
            ),
            "rerank_prompt_example_suffix": (
                rerank_prompt_example_suffix.strip()
                if args.rerank and rerank_prompt_example_suffix
                else None
            ),
            "rerank_user_prompt_template": (
                rerank_user_prompt_template.strip()
                if args.rerank and rerank_user_prompt_template
                else None
            ),
            "rerank_include_thoughts": (
                args.rerank_include_thoughts if args.rerank else None
            ),
            "rerank_input_transform": (
                args.rerank_input_transform if args.rerank else None
            ),
            "rerank_input_size": args.rerank_input_size if args.rerank else None,
            "rerank_backend": "litellm" if args.rerank else None,
            "rerank_batch_id": None,
        }
    )
    print("Recall: ", results.metrics.recall)
    print("Execution time: ", results.metrics.execution_time)

    if args.output_file_path:
        output_path = args.output_file_path
    else:
        output_path = get_default_output_path(
            args.rank_func,
            args.topn,
            args.dataset_size,
            args.query_limit,
            args.query_offset,
            args.rerank,
            args.rerank_input_size,
            args.rerank_model_name,
            args.rerank_reasoning_effort,
            args.rerank_include_thoughts,
            args.rerank_input_transform,
        )
    if not args.no_save:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results, f, ensure_ascii=False, indent=2, default=lambda x: x.__dict__
            )


if __name__ == "__main__":
    main()
