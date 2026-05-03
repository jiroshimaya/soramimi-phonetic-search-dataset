from .base_ranking import rank_by_vowel_consonant_editdistance
from .evaluate import RankingFunc, RankingFunctionOutput
from .reranker import rerank_by_llm as _rerank_by_llm_impl


def rank_by_llm(
    query_texts: list[str],
    wordlists: list[list[str]],
    *,
    topn: int = 10,
    rerank_input_size: int = 100,
    base_rank_func: RankingFunc = rank_by_vowel_consonant_editdistance,
    model_name: str = "gpt-4o-mini",
    reasoning_effort: str | None = None,
    prompt_template: str = "default",
    include_thoughts: bool = False,
    input_transform: str = "none",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
    **base_rank_kwargs,
) -> list[list[str]]:
    """
    ベースランキングで候補を絞ってから LLM による再ランキングを実行する

    まず編集距離ベースなどの軽量なランキング関数で候補を並べ、上位候補だけを
    パッケージ内の LLM reranker に渡す。
    """

    base_ranking_output = base_rank_func(query_texts, wordlists, **base_rank_kwargs)
    if isinstance(base_ranking_output, RankingFunctionOutput):
        base_ranked_wordlists = base_ranking_output.ranked_wordlists
    else:
        base_ranked_wordlists = base_ranking_output

    topk_ranked_wordlists = [
        ranked_wordlist[:rerank_input_size] for ranked_wordlist in base_ranked_wordlists
    ]

    return _rerank_by_llm_impl(
        query_texts,
        topk_ranked_wordlists,
        topn=topn,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        prompt_template=prompt_template,
        include_thoughts=include_thoughts,
        input_transform=input_transform,
        batch_size=batch_size,
        temperature=temperature,
        rerank_interval=rerank_interval,
    )