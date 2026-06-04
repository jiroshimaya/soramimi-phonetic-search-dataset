import json
from pathlib import Path

from soramimi_phonetic_search_dataset.base_ranking import (
    rank_by_mora_editdistance,
    rank_by_vowel_consonant_editdistance,
)
from soramimi_phonetic_search_dataset.dataset import load_phonetic_search_dataset

K = 10
VOWEL_RATIO = 0.8
HARD_NEGATIVE_COUNT = 100

GOJUON_ORDER = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヰヱヲンー"
GOJUON_INDEX = {char: index for index, char in enumerate(GOJUON_ORDER)}
SMALL_TO_LARGE = {
    "ァ": "ア",
    "ィ": "イ",
    "ゥ": "ウ",
    "ェ": "エ",
    "ォ": "オ",
    "ャ": "ヤ",
    "ュ": "ユ",
    "ョ": "ヨ",
    "ッ": "ツ",
    "ヮ": "ワ",
    "ヵ": "カ",
    "ヶ": "ケ",
}
VOICED_TO_BASE = {
    "ガ": "カ",
    "ギ": "キ",
    "グ": "ク",
    "ゲ": "ケ",
    "ゴ": "コ",
    "ザ": "サ",
    "ジ": "シ",
    "ズ": "ス",
    "ゼ": "セ",
    "ゾ": "ソ",
    "ダ": "タ",
    "ヂ": "チ",
    "ヅ": "ツ",
    "デ": "テ",
    "ド": "ト",
    "バ": "ハ",
    "ビ": "ヒ",
    "ブ": "フ",
    "ベ": "ヘ",
    "ボ": "ホ",
    "ヴ": "ウ",
}
SEMI_VOICED_TO_BASE = {
    "パ": "ハ",
    "ピ": "ヒ",
    "プ": "フ",
    "ペ": "ヘ",
    "ポ": "ホ",
}


def _to_katakana(text: str) -> str:
    normalized = text.strip()
    chars: list[str] = []
    for char in normalized:
        codepoint = ord(char)
        if 0x3041 <= codepoint <= 0x3096:
            chars.append(chr(codepoint + 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def _gojuon_sort_key(text: str) -> tuple[tuple[int, int, int, int], ...]:
    key: list[tuple[int, int, int, int]] = []
    for char in _to_katakana(text):
        voiced_rank = 0
        if char in VOICED_TO_BASE:
            voiced_rank = 1
            base_char = VOICED_TO_BASE[char]
        elif char in SEMI_VOICED_TO_BASE:
            voiced_rank = 2
            base_char = SEMI_VOICED_TO_BASE[char]
        else:
            base_char = char

        is_small = int(base_char in SMALL_TO_LARGE)
        normalized_base = SMALL_TO_LARGE.get(base_char, base_char)
        primary_order = GOJUON_INDEX.get(normalized_base, len(GOJUON_INDEX) + ord(char))
        key.append((primary_order, voiced_rank, is_small, ord(char)))
    return tuple(key)


def _recall_is_one_at_k(
    ranked_words: list[str], positive_words: list[str], k: int
) -> bool:
    return all(word in set(ranked_words[:k]) for word in positive_words)


def _build_difficulty(
    mora_ranked_words: list[str],
    vowel_consonant_ranked_words: list[str],
    positive_words: list[str],
) -> str:
    if _recall_is_one_at_k(mora_ranked_words, positive_words, K):
        return "easy"
    if _recall_is_one_at_k(vowel_consonant_ranked_words, positive_words, K):
        return "medium"
    return "hard"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "src/soramimi_phonetic_search_dataset/data/baseball.json"
    dataset = load_phonetic_search_dataset(str(dataset_path))

    query_texts = [query.query for query in dataset.queries]
    wordlists = [dataset.words] * len(query_texts)

    mora_ranked_words = rank_by_mora_editdistance(query_texts, wordlists)
    vowel_consonant_ranked_words = rank_by_vowel_consonant_editdistance(
        query_texts,
        wordlists,
        vowel_ratio=VOWEL_RATIO,
    )

    with open(dataset_path, "r", encoding="utf-8") as file:
        raw_dataset = json.load(file)

    for index, raw_query in enumerate(raw_dataset["queries"]):
        positives = raw_query["positive"]
        raw_query["difficulty"] = _build_difficulty(
            mora_ranked_words[index], vowel_consonant_ranked_words[index], positives
        )
        raw_query["hard_negatives"] = sorted(
            raw_query["hard_negatives"][:HARD_NEGATIVE_COUNT],
            key=_gojuon_sort_key,
        )

    raw_dataset["metadata"] = {
        **raw_dataset.get("metadata", {}),
        "difficulty_definition": {
            "labels": ["easy", "medium", "hard"],
            "k": K,
            "priority": [
                "mora_editdistance",
                f"vowel_consonant_editdistance(vowel_ratio={VOWEL_RATIO})",
            ],
            "rule": "moraでrecall@k=1ならeasy、次にvowel/consonantで1ならmedium、どちらも満たさなければhard",
        },
        "hard_negatives_definition": {
            "source": "rank_by_vowel_consonant_editdistance(vowel_ratio=0.5)",
            "take_topn_after_positive_exclusion": HARD_NEGATIVE_COUNT,
            "display_order": "gojuon",
        },
    }

    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(raw_dataset, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
