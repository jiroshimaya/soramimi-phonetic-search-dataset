import editdistance as ed
import jamorasep
import pyopenjtalk
from kanasim import create_kana_distance_calculator


def rank_by_mora_editdistance(
    query_texts: list[str], wordlist_texts: list[str]
) -> list[list[str]]:
    """
    モーラ編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    query_moras = [jamorasep.parse(text) for text in query_texts]
    wordlist_moras = [jamorasep.parse(text) for text in wordlist_texts]

    final_results = []
    for query_mora in query_moras:
        scores = []
        for wordlist_mora in wordlist_moras:
            distance = ed.eval(query_mora, wordlist_mora)
            scores.append(distance)

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_vowel_consonant_editdistance(
    query_texts: list[str], wordlist_texts: list[str], vowel_ratio: float = 0.5
) -> list[list[str]]:
    """
    母音と子音の編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト
        vowel_ratio: 母音の重み（0.0-1.0）

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    query_moras = [
        jamorasep.parse(text, output_format="simple-ipa") for text in query_texts
    ]
    query_vowels = [[m[-1] for m in mora] for mora in query_moras]
    query_consonants = [
        [m[:-1] if m[:-1] else "sp" for m in mora] for mora in query_moras
    ]
    wordlist_moras = [
        jamorasep.parse(text, output_format="simple-ipa") for text in wordlist_texts
    ]
    wordlist_vowels = [[m[-1] for m in mora] for mora in wordlist_moras]
    wordlist_consonants = [
        [m[:-1] if m[:-1] else "sp" for m in mora] for mora in wordlist_moras
    ]

    final_results = []
    for query_vowel, query_consonant in zip(query_vowels, query_consonants):
        scores = []
        for wordlist_vowel, wordlist_consonant in zip(
            wordlist_vowels, wordlist_consonants
        ):
            vowel_distance = ed.eval(query_vowel, wordlist_vowel)
            consonant_distance = ed.eval(query_consonant, wordlist_consonant)
            distance = vowel_distance * vowel_ratio + consonant_distance * (
                1 - vowel_ratio
            )
            scores.append(distance)

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_phoneme_editdistance(
    query_texts: list[str], wordlist_texts: list[str]
) -> list[list[str]]:
    """
    音素編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    query_phonemes = [pyopenjtalk.g2p(text).split() for text in query_texts]
    wordlist_phonemes = [pyopenjtalk.g2p(text).split() for text in wordlist_texts]

    final_results = []
    for query_phoneme in query_phonemes:
        scores = []
        for wordlist_phoneme in wordlist_phonemes:
            distance = ed.eval(query_phoneme, wordlist_phoneme)
            scores.append(distance)

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_kanasim(
    query_texts: list[str], wordlist_texts: list[str], **kwargs
) -> list[list[str]]:
    """
    KanaSimに基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト
        **kwargs: KanaSimのパラメータ

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    kana_distance_calculator = create_kana_distance_calculator(**kwargs)
    all_scores = kana_distance_calculator.calculate_batch(query_texts, wordlist_texts)

    ranked_wordlists = []
    for scores in all_scores:
        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists
