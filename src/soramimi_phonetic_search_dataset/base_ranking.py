import editdistance as ed
import jamorasep
import pyopenjtalk
from kanasim import create_kana_distance_calculator


def _split_phonemes(text: str) -> list[str]:
    phonemes = pyopenjtalk.g2p(text)
    return phonemes if isinstance(phonemes, list) else phonemes.split()


def _has_shared_wordlist(wordlists: list[list[str]]) -> bool:
    return bool(wordlists) and all(wordlist is wordlists[0] for wordlist in wordlists)


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
    if _has_shared_wordlist(wordlists):
        shared_wordlist = wordlists[0]
        wordlist_moras = [jamorasep.parse(text) for text in shared_wordlist]
        final_results = []
        for query_text in query_texts:
            query_mora = jamorasep.parse(query_text)
            scores = [ed.eval(query_mora, wordlist_mora) for wordlist_mora in wordlist_moras]
            ranked_wordlist = [
                word
                for word, _ in sorted(zip(shared_wordlist, scores), key=lambda x: x[1])
            ]
            final_results.append(ranked_wordlist)
        return final_results

    final_results = []
    for query_text, wordlist in zip(query_texts, wordlists):
        query_mora = jamorasep.parse(query_text)
        wordlist_moras = [jamorasep.parse(text) for text in wordlist]
        scores = [ed.eval(query_mora, wordlist_mora) for wordlist_mora in wordlist_moras]
        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
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
    def _parse_wordlist(wordlist_texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        wordlist_moras = [
            jamorasep.parse(text, output_format="simple-ipa") for text in wordlist_texts
        ]
        wordlist_vowels = [[m[-1] for m in mora] for mora in wordlist_moras]
        wordlist_consonants = [
            [m[:-1] if m[:-1] else "sp" for m in mora] for mora in wordlist_moras
        ]
        return wordlist_vowels, wordlist_consonants

    if _has_shared_wordlist(wordlists):
        shared_wordlist = wordlists[0]
        wordlist_vowels, wordlist_consonants = _parse_wordlist(shared_wordlist)
        final_results = []
        for query_text in query_texts:
            query_mora = jamorasep.parse(query_text, output_format="simple-ipa")
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
    for query_text, wordlist in zip(query_texts, wordlists):
        query_mora = jamorasep.parse(query_text, output_format="simple-ipa")
        query_vowel = [m[-1] for m in query_mora]
        query_consonant = [m[:-1] if m[:-1] else "sp" for m in query_mora]
        wordlist_vowels, wordlist_consonants = _parse_wordlist(wordlist)
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
            word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
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
    if _has_shared_wordlist(wordlists):
        shared_wordlist = wordlists[0]
        wordlist_phonemes = [_split_phonemes(text) for text in shared_wordlist]
        final_results = []
        for query_text in query_texts:
            query_phoneme = _split_phonemes(query_text)
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
    for query_text, wordlist in zip(query_texts, wordlists):
        query_phoneme = _split_phonemes(query_text)
        wordlist_phonemes = [_split_phonemes(text) for text in wordlist]
        scores = [
            ed.eval(query_phoneme, wordlist_phoneme)
            for wordlist_phoneme in wordlist_phonemes
        ]
        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])
        ]
        final_results.append(ranked_wordlist)
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
    if _has_shared_wordlist(wordlists):
        shared_wordlist = wordlists[0]
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
    for query_text, wordlist in zip(query_texts, wordlists):
        scores = kana_distance_calculator.calculate_batch([query_text], wordlist)[0]
        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists