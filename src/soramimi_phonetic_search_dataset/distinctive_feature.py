from functools import lru_cache

import pyphone
import pyopenjtalk

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
            if next_phoneme is not None:
                expanded.append(next_phoneme)
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


def _split_vowels_and_consonants(phonemes: list[str]) -> tuple[list[str], list[str]]:
    vowels = [phoneme for phoneme in phonemes if pyphone.is_vowel(phoneme)]
    consonants = [phoneme for phoneme in phonemes if not pyphone.is_vowel(phoneme)]
    return vowels, consonants


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


def rank_by_distinctive_feature_distance(
    query_texts: list[str], wordlist_texts: list[str], vowel_ratio: float = 0.5
) -> list[list[str]]:
    """
    音素の弁別素性に基づく距離でランキングする関数

    Args:
        query_texts: クエリのリスト
        wordlist_texts: 単語リスト
        vowel_ratio: 母音距離の重み（0.0-1.0）

    Returns:
        list[list[str]]: 各クエリに対する単語のランキング結果
    """
    query_phonemes = [_to_pyphone_phonemes(text) for text in query_texts]
    wordlist_phonemes = [_to_pyphone_phonemes(text) for text in wordlist_texts]

    query_features = [
        _split_vowels_and_consonants(phonemes) for phonemes in query_phonemes
    ]
    wordlist_features = [
        _split_vowels_and_consonants(phonemes) for phonemes in wordlist_phonemes
    ]

    ranked_wordlists = []
    for query_vowels, query_consonants in query_features:
        scores = []
        for word_vowels, word_consonants in wordlist_features:
            vowel_distance = _weighted_levenshtein_distance(query_vowels, word_vowels)
            consonant_distance = _weighted_levenshtein_distance(
                query_consonants, word_consonants
            )
            scores.append(
                vowel_distance * vowel_ratio + consonant_distance * (1 - vowel_ratio)
            )

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists
