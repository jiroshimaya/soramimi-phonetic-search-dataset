from soramimi_phonetic_search_dataset import (
    rank_by_kanasim,
    rank_by_mora_editdistance,
    rank_by_phoneme_editdistance,
    rank_by_vowel_consonant_editdistance,
)


def test_rank_by_mora_editdistance():
    """モーラ編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_mora_editdistance(query_texts, wordlist_texts)

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]  # 最も類似度が高い
    assert ranked_wordlists[0][-1] == "ハナコ"  # 最も類似度が低い


def test_rank_by_vowel_consonant_editdistance():
    """母音子音編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]

    # デフォルトの重み（母音:子音 = 0.5:0.5）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(query_texts, wordlist_texts)
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # 母音重視（母音:子音 = 0.8:0.2）
    ranked_wordlists = rank_by_vowel_consonant_editdistance(
        query_texts, wordlist_texts, vowel_ratio=0.8
    )
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_phoneme_editdistance():
    """音素編集距離によるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_phoneme_editdistance(query_texts, wordlist_texts)

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"


def test_rank_by_kanasim():
    """KanaSimによるランキングのテスト"""
    query_texts = ["タロウ"]
    wordlist_texts = ["タロー", "タロ", "ハナコ"]
    ranked_wordlists = rank_by_kanasim(query_texts, wordlist_texts)

    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"

    # カスタムパラメータでのテスト
    ranked_wordlists = rank_by_kanasim(query_texts, wordlist_texts, vowel_ratio=0.5)
    assert len(ranked_wordlists) == 1
    assert ranked_wordlists[0][0] in ["タロー", "タロ"]
    assert ranked_wordlists[0][-1] == "ハナコ"
