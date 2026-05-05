from soramimi_phonetic_search_dataset import (
    rank_by_kanasim,
    rank_by_llm,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from soramimi_phonetic_search_dataset import llm_ranking
from soramimi_phonetic_search_dataset import reasoning_llm_ranking


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


def test_llm_ranking_build_system_prompt_only_supports_default():
    import pytest

    prompt = llm_ranking.build_system_prompt("default")

    assert "You are a phonetic search assistant." in prompt
    with pytest.raises(ValueError, match="Unknown prompt_template: detailed"):
        llm_ranking.build_system_prompt("detailed")
