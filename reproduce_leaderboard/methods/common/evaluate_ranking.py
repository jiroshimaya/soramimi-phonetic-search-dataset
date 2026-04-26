import argparse
import json
from typing import Callable

from batch_reranker import (
    get_default_batch_state_path,
    prepare_rerank_candidates,
    retrieve_openai_batch_evaluation_results,
    submit_openai_batch_evaluation,
)
from reranker import (
    calculate_token_cost,
    get_last_token_usage,
    get_rerank_response_format,
    get_last_structured_outputs,
    rerank_by_llm,
)
from soramimi_phonetic_search_dataset import (
    evaluate_ranking_function_with_details,
    load_default_dataset,
    load_small_dataset,
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)


def create_reranking_function(
    base_rank_func: Callable[[list[str], list[str]], list[list[str]]],
    rerank_input_size: int,
    rerank_model_name: str,
    rerank_reasoning_effort: str | None,
    rerank_prompt_template: str,
    rerank_include_thoughts: bool,
    rerank_input_transform: str,
    rerank_batch_size: int,
    rerank_interval: int,
    topn: int,
    positive_texts: list[list[str]],
    **base_rank_kwargs,
) -> Callable[[list[str], list[str]], list[list[str]]]:
    """
    ベースのランキング関数とLLMによるリランクを組み合わせた関数を作成する

    Args:
        base_rank_func: ベースのランキング関数
        rerank_input_size: リランクに使用する候補数
        rerank_model_name: リランクに使用するモデル名
        rerank_reasoning_effort: リランクに使用するreasoning effort
        rerank_prompt_template: リランクに使用するプロンプトテンプレート
        rerank_input_transform: リランク前に query / candidate に適用する入力変換
        rerank_batch_size: リランクのバッチサイズ
        rerank_interval: リランクのインターバル
        topn: 最終的な出力数
        positive_texts: 各クエリに対する正解リスト
        **base_rank_kwargs: ベースのランキング関数に渡す追加の引数

    Returns:
        組み合わせたランキング関数
    """

    def combined_rank_func(
        query_texts: list[str], wordlist_texts: list[str]
    ) -> list[list[str]]:
        # ベースのランキングを実行
        base_ranked_wordlists = base_rank_func(
            query_texts, wordlist_texts, **base_rank_kwargs
        )

        # 上位N件を取得してリランク
        topk_ranked_wordlists = prepare_rerank_candidates(
            base_ranked_wordlists,
            positive_texts,
            rerank_input_size,
        )

        reranked_wordlists = rerank_by_llm(
            query_texts,
            topk_ranked_wordlists,
            topn=topn,
            model_name=rerank_model_name,
            reasoning_effort=rerank_reasoning_effort,
            prompt_template=rerank_prompt_template,
            include_thoughts=rerank_include_thoughts,
            input_transform=rerank_input_transform,
            batch_size=rerank_batch_size,
            rerank_interval=rerank_interval,
        )
        return reranked_wordlists

    return combined_rank_func


def get_default_output_path(
    rank_func: str,
    topn: int,
    dataset_size: str = "default",
    rerank: bool = False,
    rerank_topn: int = 10,
    rerank_model_name: str = "gpt-4o-mini",
    rerank_reasoning_effort: str | None = None,
    rerank_prompt_template: str = "default",
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
        if rerank_prompt_template != "default":
            suffix += f"_prompt{rerank_prompt_template}"
        if rerank_include_thoughts:
            suffix += "_withthoughts"
        if rerank_input_transform != "none":
            suffix += f"_transform{rerank_input_transform}"
    if dataset_size != "default":
        suffix += f"_dataset{dataset_size}"
    return f"output{suffix}.json"


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
        "--rerank_prompt_template",
        type=str,
        choices=[
            "default",
            "simple",
            "detailed",
            "step_by_step",
            "detailed_romaji_explicit",
            "nonreasoning_cot",
        ],
        default="default",
        help="System prompt template for LLM reranking",
    )
    parser.add_argument(
        "--rerank_include_thoughts",
        action="store_true",
        help="Require structured outputs to include thoughts in addition to reranked",
    )
    parser.add_argument(
        "--rerank_backend",
        type=str,
        choices=["litellm", "openai_batch"],
        default="litellm",
        help="Backend for LLM reranking",
    )
    parser.add_argument(
        "--rerank_batch_action",
        type=str,
        choices=["submit", "retrieve"],
        default="submit",
        help="Action for OpenAI Batch reranking",
    )
    parser.add_argument(
        "--rerank_batch_state_path",
        type=str,
        help="Path to the OpenAI Batch state JSON file",
    )
    parser.add_argument(
        "--rerank_input_transform",
        type=str,
        choices=["none", "pyopenjtalk_romaji", "kana_and_pyopenjtalk_romaji"],
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
            args.rerank,
            args.rerank_input_size,
            args.rerank_model_name,
            args.rerank_reasoning_effort,
            args.rerank_prompt_template,
            args.rerank_include_thoughts,
            args.rerank_input_transform,
        )
    batch_state_path = args.rerank_batch_state_path or get_default_batch_state_path(
        output_path
    )

    dataset = (
        load_small_dataset() if args.dataset_size == "small" else load_default_dataset()
    )

    if args.rerank and args.rerank_backend == "openai_batch":
        query_texts = [query.query for query in dataset.queries]
        positive_texts = [query.positive for query in dataset.queries]
        response_format = get_rerank_response_format(
            include_thoughts=args.rerank_include_thoughts
        )

        if args.rerank_batch_action == "submit":
            batch_state = submit_openai_batch_evaluation(
                base_rank_func=base_rank_func,
                query_texts=query_texts,
                word_texts=dataset.words,
                positive_texts=positive_texts,
                rank_kwargs=rank_kwargs,
                rerank_input_size=args.rerank_input_size,
                topn=args.topn,
                model_name=args.rerank_model_name,
                prompt_template=args.rerank_prompt_template,
                response_format=response_format,
                input_transform=args.rerank_input_transform,
                state_path=batch_state_path,
                output_file_path=output_path,
                reasoning_effort=args.rerank_reasoning_effort,
            )
            print("OpenAI batch submitted: ", batch_state["batch_id"])
            print("Batch state path: ", batch_state_path)
            return

        results = retrieve_openai_batch_evaluation_results(
            state_path=batch_state_path,
            query_texts=query_texts,
            positive_texts=positive_texts,
            response_format=response_format,
            rank_func=args.rank_func,
            vowel_ratio=args.vowel_ratio,
            topn=args.topn,
            rerank_input_size=args.rerank_input_size,
            model_name=args.rerank_model_name,
            reasoning_effort=args.rerank_reasoning_effort,
            prompt_template=args.rerank_prompt_template,
            rerank_include_thoughts=args.rerank_include_thoughts,
            input_transform=args.rerank_input_transform,
            backend=args.rerank_backend,
        )

        print("Recall: ", results.metrics.recall)
        print("Execution time: ", results.metrics.execution_time)
        if not args.no_save:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda x: x.__dict__,
                )
        return
    # リランクが必要な場合は組み合わせた関数を作成
    if args.rerank:
        positive_texts = [query.positive for query in dataset.queries]

        _rank_func = create_reranking_function(
            base_rank_func=base_rank_func,
            rerank_input_size=args.rerank_input_size,
            rerank_model_name=args.rerank_model_name,
            rerank_reasoning_effort=args.rerank_reasoning_effort,
            rerank_prompt_template=args.rerank_prompt_template,
            rerank_include_thoughts=args.rerank_include_thoughts,
            rerank_input_transform=args.rerank_input_transform,
            rerank_batch_size=args.rerank_batch_size,
            rerank_interval=args.rerank_interval,
            topn=args.topn,
            positive_texts=positive_texts,
            **rank_kwargs,
        )

        # 警告を回避するためdefでラップ
        def rank_func(q, w):
            return _rank_func(q, w)
    else:
        # 警告を回避するためdefでラップ
        def rank_func(q, w):
            return base_rank_func(q, w, **rank_kwargs)

    # 評価を実行
    results = evaluate_ranking_function_with_details(
        ranking_func=rank_func,
        topn=args.topn,
        dataset=dataset,
    )

    # パラメータを設定
    results.parameters.rank_func = args.rank_func
    results.parameters.vowel_ratio = (
        args.vowel_ratio if args.rank_func in ["kanasim", "vowel_consonant"] else None
    )
    results.parameters.rerank = args.rerank
    results.parameters.rerank_model_name = (
        args.rerank_model_name if args.rerank else None
    )
    results.parameters.rerank_reasoning_effort = (
        args.rerank_reasoning_effort if args.rerank else None
    )
    results.parameters.rerank_prompt_template = (
        args.rerank_prompt_template if args.rerank else None
    )
    results.parameters.rerank_include_thoughts = (
        args.rerank_include_thoughts if args.rerank else None
    )
    results.parameters.rerank_input_transform = (
        args.rerank_input_transform if args.rerank else None
    )
    results.parameters.rerank_input_size = (
        args.rerank_input_size if args.rerank else None
    )
    results.parameters.rerank_backend = args.rerank_backend if args.rerank else None
    results.parameters.rerank_batch_id = None
    if args.rerank and args.rerank_include_thoughts:
        structured_outputs = get_last_structured_outputs()
        for result, structured_output in zip(results.results, structured_outputs):
            result.thoughts = structured_output.get("thoughts")
    if args.rerank:
        token_usage = get_last_token_usage()
        token_cost = calculate_token_cost(args.rerank_model_name, token_usage)
        results.metrics.rerank_input_tokens = token_usage.input_tokens
        results.metrics.rerank_output_tokens = token_usage.output_tokens
        results.metrics.rerank_reasoning_tokens = token_usage.reasoning_tokens
        results.metrics.rerank_total_tokens = token_usage.total_tokens
        results.metrics.rerank_input_cost = token_cost.input_cost
        results.metrics.rerank_output_cost = token_cost.output_cost
        results.metrics.rerank_reasoning_cost = token_cost.reasoning_cost
        results.metrics.rerank_total_cost = token_cost.total_cost

    print("Recall: ", results.metrics.recall)
    print("Execution time: ", results.metrics.execution_time)

    if args.output_file_path:
        output_path = args.output_file_path
    else:
        output_path = get_default_output_path(
            args.rank_func,
            args.topn,
            args.dataset_size,
            args.rerank,
            args.rerank_input_size,
            args.rerank_model_name,
            args.rerank_reasoning_effort,
            args.rerank_prompt_template,
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
