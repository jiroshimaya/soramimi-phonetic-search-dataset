from soramimi_phonetic_search_dataset import (
    rank_by_kanasim,
    rank_by_llm,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from soramimi_phonetic_search_dataset import llm_ranking
from soramimi_phonetic_search_dataset import reasoning_llm_ranking
from soramimi_phonetic_search_dataset.evaluate import RankingFunctionOutput


def test_rank_by_mora_editdistance():
    """モーラ編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_mora_editdistance(query_texts, [wordlist_texts])

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]  # 最も類似度が高い
    assert ranked_wordlists[0][-1] == "ハナコ"  # 最も類似度が低い


def test_rank_by_mora_editdistance_reuses_shared_words(monkeypatch):
    """異なる単語リストでも同じ単語の前処理を再利用する"""
    from soramimi_phonetic_search_dataset import base_ranking

    original_parse = base_ranking.jamorasep.parse
    parse_call_count = 0

    def counting_parse(text, *args, **kwargs):
        nonlocal parse_call_count
        parse_call_count += 1
        return original_parse(text, *args, **kwargs)

    monkeypatch.setattr(base_ranking.jamorasep, "parse", counting_parse)

    query_texts = ["タロウ", "ジロウ"]
    wordlist_a = ["タロー", "タロ", "ハナコ"]
    wordlist_b = ["タロー", "ジロ", "ハナコ"]

    ranked_wordlists = rank_by_mora_editdistance(query_texts, [wordlist_a, wordlist_b])

    assert len(ranked_wordlists) == 2
    assert parse_call_count == len(query_texts) + 4


def test_rank_by_vowel_consonant_editdistance():
    """母音子音編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]

    # デフォルトの重み（母音:子音 = 0.5:0.5）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(query_texts, [wordlist_texts])
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # 母音重視（母音:子音 = 0.8:0.2）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(
        query_texts, [wordlist_texts], vowel_ratio=0.8
    )
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_phoneme_editdistance():
    """音素編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_phoneme_editdistance(query_texts, [wordlist_texts])

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_kanasim():
    """KanaSimによるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_kanasim(query_texts, [wordlist_texts])

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # カスタムパラメータでのテスト
    ranked_wordlists = rank_by_kanasim(query_texts, [wordlist_texts], vowel_ratio=0.5)
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_llm_reranks_candidates(monkeypatch):
    """LLM ランキングは候補 wordlist をそのまま再ランキングする"""

    captured_messages = []

    def fake_get_structured_outputs(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return [{"reranked": [1, 0]}]

    monkeypatch.setattr(
        reasoning_llm_ranking, "get_structured_outputs", fake_get_structured_outputs
    )

    query_texts = ["タロウ"]
    wordlists = [["タロー", "タロ", "ハナコ", "サブロウ"]]

    ranked_wordlists = reasoning_llm_ranking.rank_by_llm(
        query_texts,
        wordlists,
        topn=2,
        model_name="gpt-5.4",
        prompt_template="detailed",
        rerank_interval=0,
    )

    assert ranked_wordlists == [["タロ", "タロー"]]
    assert "Query: タロウ" in captured_messages[0][1]["content"]
    assert "0. タロー" in captured_messages[0][1]["content"]
    assert "1. タロ" in captured_messages[0][1]["content"]
    assert "Top N: 2" in captured_messages[0][1]["content"]


def test_rank_by_llm_returns_ranking_function_output_with_metrics_metadata(
    monkeypatch,
):
    def fake_get_structured_outputs(**kwargs):
        return llm_ranking.StructuredOutputsResult(
            parsed_responses=[{"reranked": [1, 0]}],
            structured_outputs=[{"reranked": [1, 0]}],
            token_usage=llm_ranking.TokenUsage(
                input_tokens=12,
                completion_tokens=8,
                reasoning_tokens=3,
                total_tokens=20,
            ),
        )

    monkeypatch.setattr(llm_ranking, "get_structured_outputs", fake_get_structured_outputs)
    monkeypatch.setattr(
        llm_ranking,
        "calculate_token_cost",
        lambda model_name, token_usage: llm_ranking.TokenCost(
            input_cost=0.1,
            output_cost=0.2,
            reasoning_cost=0.05,
            total_cost=0.3,
        ),
    )

    reranked = rank_by_llm(
        query_texts=["タロウ"],
        wordlist_texts=[["タロー", "タロ"]],
        model_name="gpt-5.4",
        rerank_interval=0,
    )

    assert isinstance(reranked, RankingFunctionOutput)
    assert reranked.ranked_wordlists == [["タロ", "タロー"]]
    assert reranked.metrics_metadata == {
        "model_name": "gpt-5.4",
        "token_usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "reasoning_tokens": 3,
            "total_tokens": 20,
        },
        "cost": {
            "input_cost": 0.1,
            "output_cost": 0.2,
            "reasoning_cost": 0.05,
            "total_cost": 0.3,
        },
    }


def test_rank_by_llm_aggregates_token_usage_across_batches(monkeypatch):
    token_usages = iter(
        [
            llm_ranking.TokenUsage(
                input_tokens=10,
                completion_tokens=6,
                reasoning_tokens=2,
                total_tokens=16,
            ),
            llm_ranking.TokenUsage(
                input_tokens=20,
                completion_tokens=9,
                reasoning_tokens=4,
                total_tokens=29,
            ),
        ]
    )

    def fake_get_structured_outputs(**kwargs):
        reranked = [{"reranked": [0]} for _ in kwargs["messages"]]
        return llm_ranking.StructuredOutputsResult(
            parsed_responses=reranked,
            structured_outputs=reranked,
            token_usage=next(token_usages),
        )

    monkeypatch.setattr(llm_ranking, "get_structured_outputs", fake_get_structured_outputs)
    monkeypatch.setattr(
        llm_ranking,
        "calculate_token_cost",
        lambda model_name, token_usage: llm_ranking.TokenCost(
            input_cost=float(token_usage.input_tokens),
            output_cost=float(token_usage.output_tokens),
            reasoning_cost=float(token_usage.reasoning_tokens),
            total_cost=float(token_usage.total_tokens),
        ),
    )

    reranked = rank_by_llm(
        query_texts=["タロウ", "ハナコ"],
        wordlist_texts=[["タロー"], ["ハナ"]],
        model_name="gpt-5.4",
        batch_size=1,
        rerank_interval=0,
    )

    assert reranked.metrics_metadata == {
        "model_name": "gpt-5.4",
        "token_usage": {
            "input_tokens": 30,
            "output_tokens": 9,
            "reasoning_tokens": 6,
            "total_tokens": 45,
        },
        "cost": {
            "input_cost": 30.0,
            "output_cost": 9.0,
            "reasoning_cost": 6.0,
            "total_cost": 45.0,
        },
    }
