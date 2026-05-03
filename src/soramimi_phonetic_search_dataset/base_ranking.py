import editdistance as ed
import jamorasep
import pyopenjtalk
from kanasim import create_kana_distance_calculator

from .schemas import PhoneticSearchQueryWithWordlist


def _split_phonemes(text: str) -> list[str]:
    phonemes = pyopenjtalk.g2p(text)
    return phonemes if isinstance(phonemes, list) else phonemes.split()


def _has_shared_wordlist(
    query_inputs: list[PhoneticSearchQueryWithWordlist],
) -> bool:
    return bool(query_inputs) and all(
        query_input.wordlist is query_inputs[0].wordlist for query_input in query_inputs
    )


def rank_by_mora_editdistance(
    query_inputs: list[PhoneticSearchQueryWithWordlist],
) -> list[list[str]]:
    """
    モーラ編集距離に基づくランキング関数

    Args:
        query_inputs: クエリごとの入力リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    if _has_shared_wordlist(query_inputs):
        shared_wordlist = query_inputs[0].wordlist
        wordlist_moras = [jamorasep.parse(text) for text in shared_wordlist]
        final_results = []
        for query_input in query_inputs:
            query_mora = jamorasep.parse(query_input.query)
            scores = [ed.eval(query_mora, wordlist_mora) for wordlist_mora in wordlist_moras]
            ranked_wordlist = [
                word
                for word, _ in sorted(zip(shared_wordlist, scores), key=lambda x: x[1])
            ]
            final_results.append(ranked_wordlist)
        return final_results

    final_results = []
    for query_input in query_inputs:
        query_mora = jamorasep.parse(query_input.query)
        wordlist_moras = [jamorasep.parse(text) for text in query_input.wordlist]
        scores = [ed.eval(query_mora, wordlist_mora) for wordlist_mora in wordlist_moras]
        ranked_wordlist = [
            word
            for word, _ in sorted(zip(query_input.wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_vowel_consonant_editdistance(
    query_inputs: list[PhoneticSearchQueryWithWordlist],
    vowel_ratio: float = 0.5,
) -> list[list[str]]:
    """
    母音と子音の編集距離に基づくランキング関数

    Args:
        query_inputs: クエリごとの入力リスト
        vowel_ratio: 母音の重み（0.0-1.0）

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    def _parse_wordlist(wordlist_texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        wordlist_moras = [
            jamorasep.parse(text, output_format="simple-ipa") for text in wordlist_texts
        ]
        wordlist_vowels = [[m[-1] for m in mora] for mora in wordlist_moras]
        wordlist_consonants = [
            [m[:-1] if m[:-1] else "sp" for m in mora] for mora in wordlist_moras
        ]
        return wordlist_vowels, wordlist_consonants

    if _has_shared_wordlist(query_inputs):
        shared_wordlist = query_inputs[0].wordlist
        wordlist_vowels, wordlist_consonants = _parse_wordlist(shared_wordlist)
        final_results = []
        for query_input in query_inputs:
            query_mora = jamorasep.parse(query_input.query, output_format="simple-ipa")
            query_vowel = [m[-1] for m in query_mora]
            query_consonant = [m[:-1] if m[:-1] else "sp" for m in query_mora]
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
                word
                for word, _ in sorted(zip(shared_wordlist, scores), key=lambda x: x[1])
            ]
            final_results.append(ranked_wordlist)
        return final_results

    final_results = []
    for query_input in query_inputs:
        query_mora = jamorasep.parse(query_input.query, output_format="simple-ipa")
        query_vowel = [m[-1] for m in query_mora]
        query_consonant = [m[:-1] if m[:-1] else "sp" for m in query_mora]
        wordlist_vowels, wordlist_consonants = _parse_wordlist(query_input.wordlist)
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
            word
            for word, _ in sorted(zip(query_input.wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_phoneme_editdistance(
    query_inputs: list[PhoneticSearchQueryWithWordlist],
) -> list[list[str]]:
    """
    音素編集距離に基づくランキング関数

    Args:
        query_inputs: クエリごとの入力リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    if _has_shared_wordlist(query_inputs):
        shared_wordlist = query_inputs[0].wordlist
        wordlist_phonemes = [_split_phonemes(text) for text in shared_wordlist]
        final_results = []
        for query_input in query_inputs:
            query_phoneme = _split_phonemes(query_input.query)
            scores = [
                ed.eval(query_phoneme, wordlist_phoneme)
                for wordlist_phoneme in wordlist_phonemes
            ]
            ranked_wordlist = [
                word
                for word, _ in sorted(zip(shared_wordlist, scores), key=lambda x: x[1])
            ]
            final_results.append(ranked_wordlist)
        return final_results

    final_results = []
    for query_input in query_inputs:
        query_phoneme = _split_phonemes(query_input.query)
        wordlist_phonemes = [_split_phonemes(text) for text in query_input.wordlist]
        scores = [
            ed.eval(query_phoneme, wordlist_phoneme)
            for wordlist_phoneme in wordlist_phonemes
        ]
        ranked_wordlist = [
            word
            for word, _ in sorted(zip(query_input.wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
    return final_results


def rank_by_kanasim(
    query_inputs: list[PhoneticSearchQueryWithWordlist], **kwargs
) -> list[list[str]]:
    """
    KanaSimに基づくランキング関数

    Args:
        query_inputs: クエリごとの入力リスト
        **kwargs: KanaSimのパラメータ

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    kana_distance_calculator = create_kana_distance_calculator(**kwargs)
    if _has_shared_wordlist(query_inputs):
        query_texts = [query_input.query for query_input in query_inputs]
        shared_wordlist = query_inputs[0].wordlist
        all_scores = kana_distance_calculator.calculate_batch(query_texts, shared_wordlist)

        ranked_wordlists = []
        for scores in all_scores:
            ranked_wordlist = [
                word
                for word, _ in sorted(zip(shared_wordlist, scores), key=lambda x: x[1])
            ]
            ranked_wordlists.append(ranked_wordlist)
        return ranked_wordlists

    ranked_wordlists = []
    for query_input in query_inputs:
        scores = kana_distance_calculator.calculate_batch(
            [query_input.query], query_input.wordlist
        )[0]
        ranked_wordlist = [
            word
            for word, _ in sorted(zip(query_input.wordlist, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists