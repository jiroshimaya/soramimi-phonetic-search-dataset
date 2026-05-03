from soramimi_phonetic_search_dataset import (
    PhoneticSearchQueryWithWordlist,
    rank_by_kanasim,
    rank_by_llm,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from soramimi_phonetic_search_dataset import llm_ranking


def _build_query_inputs(
    query_texts: list[str],
    wordlist_texts: list[str],
) -> list[PhoneticSearchQueryWithWordlist]:
    return [
        PhoneticSearchQueryWithWordlist(
            query=query_text,
            wordlist=wordlist_texts,
            positive_words=[],
        )
        for query_text in query_texts
    ]


def test_rank_by_mora_editdistance():
    """モーラ編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_mora_editdistance(
        _build_query_inputs(query_texts, wordlist_texts)
    )

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]  # 最も類似度が高い
    assert ranked_wordlists[0][-1] == "ハナコ"  # 最も類似度が低い


def test_rank_by_vowel_consonant_editdistance():
    """母音子音編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    query_inputs = _build_query_inputs(query_texts, wordlist_texts)

    # デフォルトの重み（母音:子音 = 0.5:0.5）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(query_inputs)
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # 母音重視（母音:子音 = 0.8:0.2）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(
        query_inputs, vowel_ratio=0.8
    )
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_phoneme_editdistance():
    """音素編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_phoneme_editdistance(
        _build_query_inputs(query_texts, wordlist_texts)
    )

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_kanasim():
    """KanaSimによるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    query_inputs = _build_query_inputs(query_texts, wordlist_texts)
    ranked_wordlists = rank_by_kanasim(query_inputs)

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # カスタムパラメータでのテスト
    ranked_wordlists = rank_by_kanasim(query_inputs, vowel_ratio=0.5)
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_llm_delegates_to_reranker(monkeypatch):
    """LLM ランキング前にベースランキングで候補を絞り込む"""

    captured = {}

    def fake_base_rank_func(query_inputs, **kwargs):
        captured["base_query_inputs"] = query_inputs
        captured["base_kwargs"] = kwargs
        return [["ハナコ", "タロー", "タロ", "サブロウ"]]

    def fake_rerank_by_llm(query_texts, wordlist_texts, **kwargs):
        captured["query_texts"] = query_texts
        captured["wordlist_texts"] = wordlist_texts
        captured["kwargs"] = kwargs
        return [["タロー", "タロ"]]

    monkeypatch.setattr(llm_ranking, "_rerank_by_llm_impl", fake_rerank_by_llm)

    query_inputs = _build_query_inputs(
        ["タロウ"],
        ["タロー", "タロ", "ハナコ", "サブロウ"],
    )

    ranked_wordlists = rank_by_llm(
        query_inputs,
        topn=2,
        rerank_input_size=3,
        base_rank_func=fake_base_rank_func,
        vowel_ratio=0.7,
        model_name="gpt-5.4",
        prompt_template="detailed",
        rerank_interval=0,
    )

    assert ranked_wordlists == [["タロー", "タロ"]]
    assert [query.query for query in captured["base_query_inputs"]] == ["タロウ"]
    assert captured["base_query_inputs"][0].wordlist == ["タロー", "タロ", "ハナコ", "サブロウ"]
    assert captured["base_kwargs"]["vowel_ratio"] == 0.7
    assert captured["query_texts"] == ["タロウ"]
    assert captured["wordlist_texts"] == [["ハナコ", "タロー", "タロ"]]
    assert captured["kwargs"]["topn"] == 2
    assert captured["kwargs"]["model_name"] == "gpt-5.4"
    assert captured["kwargs"]["prompt_template"] == "detailed"
