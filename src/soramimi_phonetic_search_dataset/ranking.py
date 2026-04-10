from functools import lru_cache

import editdistance as ed
import jamorasep
import pyphone
import pyopenjtalk
from kanasim import create_kana_distance_calculator

PYOPENJTALK_TO_PYPHONE_PHONEMES: dict[str, tuple[str, ...]] = {
    "a": ("a",),
    "b": ("b",),
    "by": ("b", "j"),
    "ch": ("tʃ",),
    "d": ("d",),
    "dy": ("d", "j"),
    "e": ("ɛ",),
    "f": ("f",),
    "g": ("ɡ",),
    "gw": ("ɡ", "w"),
    "gy": ("ɡ", "j"),
    "h": ("h",),
    "hy": ("h", "j"),
    "i": ("i",),
    "I": ("i",),
    "j": ("dʒ",),
    "k": ("k",),
    "kw": ("k", "w"),
    "ky": ("k", "j"),
    "m": ("m",),
    "my": ("m", "j"),
    "n": ("n",),
    "ny": ("n", "j"),
    "o": ("o",),
    "p": ("p",),
    "r": ("r",),
    "ry": ("r", "j"),
    "s": ("s",),
    "sh": ("ʃ",),
    "t": ("t",),
    "ts": ("ts",),
    "u": ("u",),
    "U": ("u",),
    "w": ("w",),
    "y": ("j",),
    "z": ("z",),
}
_BILABIAL_PYPHONE_PHONEMES = {"b", "m", "p", "f"}
_VELAR_PYPHONE_PHONEMES = {"k", "ɡ", "ŋ"}


def _next_pyphone_phoneme(phonemes: list[str], start_index: int) -> str | None:
    for phoneme in phonemes[start_index:]:
        if phoneme in {"N", "cl"}:
            continue
        mapped = PYOPENJTALK_TO_PYPHONE_PHONEMES.get(phoneme)
        if mapped:
            return mapped[0]
    return None


def _expand_japanese_phonemes_for_pyphone(phonemes: list[str]) -> list[str]:
    expanded = []
    for index, phoneme in enumerate(phonemes):
        if phoneme == "N":
            next_phoneme = _next_pyphone_phoneme(phonemes, index + 1)
            if next_phoneme in _BILABIAL_PYPHONE_PHONEMES:
                expanded.append("m")
            elif next_phoneme in _VELAR_PYPHONE_PHONEMES:
                expanded.append("ŋ")
            else:
                expanded.append("n")
            continue

        if phoneme == "cl":
            next_phoneme = _next_pyphone_phoneme(phonemes, index + 1)
            expanded.append(next_phoneme if next_phoneme is not None else " ")
            continue

        try:
            expanded.extend(PYOPENJTALK_TO_PYPHONE_PHONEMES[phoneme])
        except KeyError as error:
            raise ValueError(
                f"Unsupported phoneme for pyphone mapping: {phoneme}"
            ) from error
    return expanded


def _to_pyphone_phonemes(text: str) -> list[str]:
    return _expand_japanese_phonemes_for_pyphone(pyopenjtalk.g2p(text).split())


@lru_cache(maxsize=None)
def _feature_substitution_cost(phoneme_a: str, phoneme_b: str) -> float:
    return float(pyphone.distance(phoneme_a, phoneme_b))


def _weighted_levenshtein_distance(
    phonemes_a: list[str], phonemes_b: list[str]
) -> float:
    previous_row = [float(index) for index in range(len(phonemes_b) + 1)]

    for index_a, phoneme_a in enumerate(phonemes_a, start=1):
        current_row = [float(index_a)]
        for index_b, phoneme_b in enumerate(phonemes_b, start=1):
            current_row.append(
                min(
                    previous_row[index_b] + 1.0,
                    current_row[index_b - 1] + 1.0,
                    previous_row[index_b - 1]
                    + _feature_substitution_cost(phoneme_a, phoneme_b),
                )
            )
        previous_row = current_row

    return previous_row[-1]


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


def rank_by_distinctive_feature_distance(
    query_texts: list[str], wordlist_texts: list[str]
) -> list[list[str]]:
    """
    音素の弁別素性に基づく距離でランキングする関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    query_phonemes = [_to_pyphone_phonemes(text) for text in query_texts]
    wordlist_phonemes = [_to_pyphone_phonemes(text) for text in wordlist_texts]

    ranked_wordlists = []
    for query_phoneme in query_phonemes:
        scores = []
        for wordlist_phoneme in wordlist_phonemes:
            scores.append(
                _weighted_levenshtein_distance(query_phoneme, wordlist_phoneme)
            )

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists
