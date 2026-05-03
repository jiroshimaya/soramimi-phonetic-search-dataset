import editdistance as ed
import jamorasep
import pyopenjtalk
from kanasim import create_kana_distance_calculator
from typing import Callable, TypeVar


ConvertedT = TypeVar("ConvertedT")


def _split_phonemes(text: str) -> list[str]:
    phonemes = pyopenjtalk.g2p(text)
    return phonemes if isinstance(phonemes, list) else phonemes.split()


def _get_or_convert(
    memo: dict[str, ConvertedT], text: str, convert: Callable[[str], ConvertedT]
) -> ConvertedT:
    converted = memo.get(text)
    if converted is None:
        converted = convert(text)
        memo[text] = converted
    return converted


def rank_by_mora_editdistance(
    query_texts: list[str],
    wordlists: list[list[str]],
) -> list[list[str]]:
    """
    モーラ編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlists: 各クエリに対応する単語リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    word_mora_memo: dict[str, list[str]] = {}
    final_results = []
    for query_text, wordlist in zip(query_texts, wordlists):
        query_mora = _get_or_convert(word_mora_memo, query_text, jamorasep.parse)
        wordlist_moras = [
            _get_or_convert(word_mora_memo, text, jamorasep.parse) for text in wordlist
        ]
        scores = [ed.eval(query_mora, wordlist_mora) for wordlist_mora in wordlist_moras]
        final_results.append(
            [word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])]
        )
    return final_results


def rank_by_vowel_consonant_editdistance(
    query_texts: list[str],
    wordlists: list[list[str]],
    vowel_ratio: float = 0.5,
) -> list[list[str]]:
    """
    母音と子音の編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlists: 各クエリに対応する単語リスト
        vowel_ratio: 母音の重み（0.0-1.0）

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    def _parse_word(text: str) -> tuple[list[str], list[str]]:
        mora = jamorasep.parse(text, output_format="simple-ipa")
        vowels = [m[-1] for m in mora]
        consonants = [m[:-1] if m[:-1] else "sp" for m in mora]
        return vowels, consonants

    word_feature_memo: dict[str, tuple[list[str], list[str]]] = {}
    final_results = []
    for query_text, wordlist in zip(query_texts, wordlists):
        query_vowel, query_consonant = _get_or_convert(
            word_feature_memo, query_text, _parse_word
        )
        wordlist_features = [
            _get_or_convert(word_feature_memo, text, _parse_word) for text in wordlist
        ]
        scores = []
        for wordlist_vowel, wordlist_consonant in wordlist_features:
            vowel_distance = ed.eval(query_vowel, wordlist_vowel)
            consonant_distance = ed.eval(query_consonant, wordlist_consonant)
            distance = vowel_distance * vowel_ratio + consonant_distance * (
                1 - vowel_ratio
            )
            scores.append(distance)
        final_results.append(
            [word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])]
        )
    return final_results


def rank_by_phoneme_editdistance(
    query_texts: list[str],
    wordlists: list[list[str]],
) -> list[list[str]]:
    """
    音素編集距離に基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlists: 各クエリに対応する単語リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    word_phoneme_memo: dict[str, list[str]] = {}
    final_results = []
    for query_text, wordlist in zip(query_texts, wordlists):
        query_phoneme = _get_or_convert(word_phoneme_memo, query_text, _split_phonemes)
        wordlist_phonemes = [
            _get_or_convert(word_phoneme_memo, text, _split_phonemes) for text in wordlist
        ]
        scores = [
            ed.eval(query_phoneme, wordlist_phoneme)
            for wordlist_phoneme in wordlist_phonemes
        ]
        final_results.append(
            [word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])]
        )
    return final_results


def rank_by_kanasim(
    query_texts: list[str], wordlists: list[list[str]], **kwargs
) -> list[list[str]]:
    """
    KanaSimに基づくランキング関数

    Args:
        query_texts: クエリのリスト
        wordlists: 各クエリに対応する単語リスト
        **kwargs: KanaSimのパラメータ

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    kana_distance_calculator = create_kana_distance_calculator(**kwargs)
    ranked_wordlists = []
    for query_text, wordlist in zip(query_texts, wordlists):
        scores = kana_distance_calculator.calculate_batch([query_text], wordlist)[0]
        ranked_wordlists.append(
            [word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])]
        )

    return ranked_wordlists