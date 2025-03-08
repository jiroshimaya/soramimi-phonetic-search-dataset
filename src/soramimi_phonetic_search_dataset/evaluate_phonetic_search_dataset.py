import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Type

import dotenv
import editdistance as ed
import jamorasep
import pyopenjtalk
from kanasim import create_kana_distance_calculator
from litellm import batch_completion
from pydantic import BaseModel
from tqdm import tqdm

from soramimi_phonetic_search_dataset.schemas import PhoneticSearchDataset

dotenv.load_dotenv()


def load_phonetic_search_dataset(path: str) -> PhoneticSearchDataset:
    with open(path, "r") as f:
        dataset = json.load(f)
    return PhoneticSearchDataset.from_dict(dataset)


def rank_by_mora_editdistance(
    query_texts: list[str], wordlist_texts: list[str]
) -> list[list[str]]:
    query_moras = [jamorasep.parse(text) for text in query_texts]
    wordlist_moras = [jamorasep.parse(text) for text in wordlist_texts]

    filnal_results = []
    for query_mora in query_moras:
        scores = []
        for wordlist_mora in wordlist_moras:
            distance = ed.eval(query_mora, wordlist_mora)
            scores.append(distance)

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        filnal_results.append(ranked_wordlist)
    return filnal_results


def rank_by_vowel_consonant_editdistance(
    query_texts: list[str], wordlist_texts: list[str], vowel_ratio: float = 0.5
) -> list[list[str]]:
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

    filnal_results = []
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
        filnal_results.append(ranked_wordlist)
    return filnal_results


def rank_by_phoneme_editdistance(
    query_texts: list[str], wordlist_texts: list[str]
) -> list[list[str]]:
    query_phonemes = [pyopenjtalk.g2p(text).split() for text in query_texts]
    wordlist_phonemes = [pyopenjtalk.g2p(text).split() for text in wordlist_texts]

    filnal_results = []
    for query_phoneme in query_phonemes:
        scores = []
        for wordlist_phoneme in wordlist_phonemes:
            distance = ed.eval(query_phoneme, wordlist_phoneme)
            scores.append(distance)

        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        filnal_results.append(ranked_wordlist)
    return filnal_results


def rank_by_kanasim(
    query_texts: list[str], wordlist_texts: list[str], **kwargs
) -> list[list[str]]:
    kana_distance_calculator = create_kana_distance_calculator(**kwargs)

    all_scores = kana_distance_calculator.calculate_batch(query_texts, wordlist_texts)

    ranked_wordlists = []
    for scores in all_scores:
        ranked_wordlist = [
            word for word, _ in sorted(zip(wordlist_texts, scores), key=lambda x: x[1])
        ]
        ranked_wordlists.append(ranked_wordlist)

    return ranked_wordlists


def get_structured_outputs(
    model_name: str,
    messages: list[list[dict[str, Any]]],
    response_format: Type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> list[BaseModel]:
    raw_responses = batch_completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return [
        response_format.model_validate_json(response.choices[0].message.content)
        for response in raw_responses
    ]


def rerank_by_llm(
    query_texts: list[str],
    wordlist_texts: list[list[str]],
    *,
    topn: int = 10,
    model_name: str = "gpt-4o-mini",
    batch_size: int = 10,
    temperature: float = 0.0,
    rerank_interval: int = 60,
) -> list[list[str]]:
    from pydantic import BaseModel

    class RerankedWordlist(BaseModel):
        reranked: list[int]

    prompt = """
    You are a phonetic search assistant.
    You are given a query and a list of words.
    You need to rerank the words based on phonetic similarity to the query.
    When estimating phonetic similarity, please consider the following:
    1. Prioritize matching vowels
    2. Substitution, insertion, or deletion of nasal sounds, geminate consonants, and long vowels is acceptable
    3. For other cases, words with similar mora counts are preferred
    You need to return only the reranked list of index numbers of the words, no other text.
    You need to return only topn index numbers.

    Example:
    Query: タロウ
    Wordlist: 
    0. アオ
    1. アオウヅ
    2. アノウ
    3. タキョウ
    4. タド
    5. タノ
    6. タロウ
    7. タンノ
    Top N: 5
    Reranked: 6, 4, 5, 7, 2
    """

    user_prompt = """
    Query: {query}
    Wordlist:
    {wordlist}
    Top N: {topn}
    Reranked:
    """

    messages = []
    for query, wordlist in zip(query_texts, wordlist_texts):
        wordlist_str = "\n".join([f"{i}. {word}" for i, word in enumerate(wordlist)])
        message = []
        message.append({"role": "system", "content": prompt})
        message.append(
            {
                "role": "user",
                "content": user_prompt.format(
                    query=query, wordlist=wordlist_str, topn=topn
                ),
            }
        )
        messages.append(message)

    reranked_wordlists = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i : i + batch_size]
        responses = get_structured_outputs(
            model_name=model_name,
            messages=batch_messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=RerankedWordlist,
        )
        for wordlist, response in zip(wordlist_texts[i : i + batch_size], responses):
            reranked_wordlist = []

            response_typed = RerankedWordlist.model_validate(response)
            for i in response_typed.reranked:
                if 0 <= i < len(wordlist):
                    reranked_wordlist.append(wordlist[i])
                else:
                    reranked_wordlist.append("NA")
            reranked_wordlists.append(reranked_wordlist)

        time.sleep(rerank_interval)

    return reranked_wordlists


def rank_dataset(
    phonetic_search_dataset: PhoneticSearchDataset,
    rank_func: Callable[[list[str], list[str]], list[list[str]]],
    rank_func_kwargs: dict[str, Any] = {},
) -> list[list[str]]:
    query_texts = [query.query for query in phonetic_search_dataset.queries]
    wordlist_texts = phonetic_search_dataset.words

    ranked_wordlists = rank_func(query_texts, wordlist_texts, **rank_func_kwargs)

    return ranked_wordlists


def calculate_recall(
    ranked_wordlists: list[list[str]],
    positive_texts: list[list[str]],
    topn: int = 10,
) -> float:
    recalls = []
    for wordlist, positive_text in zip(ranked_wordlists, positive_texts):
        topn_wordlist = wordlist[:topn]
        positive_text_count = len(positive_text)
        hit_count = len(set(topn_wordlist) & set(positive_text))
        recall = hit_count / positive_text_count
        recalls.append(recall)

    return sum(recalls) / len(recalls)


def get_default_output_path(
    input_path: str,
    rank_func: str,
    topn: int,
    rerank: bool = False,
    rerank_topn: int = 10,
    rerank_model_name: str = "gpt-4o-mini",
) -> str:
    input_path_lib = Path(input_path)
    suffix = f"_{rank_func}_top{topn}"
    if rerank:
        suffix += f"_reranked_top{rerank_topn}_model{rerank_model_name}"
    return str(input_path_lib.parent / f"{input_path_lib.stem}{suffix}.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate phonetic search dataset.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "-r",
        "--rank_func",
        type=str,
        choices=["kanasim", "vowel_consonant", "phoneme", "mora"],
        default="vowel_consonant",
        help="Rank function: kanasim, vowel_consonant, phoneme, mora",
    )
    parser.add_argument(
        "-n",
        "--topn",
        type=int,
        default=10,
        help="Top N",
    )
    parser.add_argument(
        "-vr",
        "--vowel_ratio",
        type=float,
        default=0.5,
        help="Vowel ratio, which is used only when rank_func is vowel_consonant",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the wordlists by LLM",
    )
    parser.add_argument(
        "--rerank_input_size",
        type=int,
        default=100,
        help="Number of top candidates to consider for reranking",
    )
    parser.add_argument(
        "--rerank_batch_size",
        type=int,
        default=10,
        help="Batch size for reranking",
    )
    parser.add_argument(
        "--rerank_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name for reranking",
    )
    parser.add_argument(
        "--rerank_interval",
        type=int,
        default=0,
        help="Sleep interval in seconds between reranking batches",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save results to file",
    )
    args = parser.parse_args()

    dataset = load_phonetic_search_dataset(args.input_path)
    if args.rank_func == "kanasim":
        ranked_wordlists = rank_dataset(
            dataset,
            rank_by_kanasim,
            {"vowel_ratio": args.vowel_ratio},
        )
    elif args.rank_func == "mora":
        ranked_wordlists = rank_dataset(
            dataset,
            rank_by_mora_editdistance,
        )
    elif args.rank_func == "vowel_consonant":
        ranked_wordlists = rank_dataset(
            dataset,
            rank_by_vowel_consonant_editdistance,
            {"vowel_ratio": args.vowel_ratio},
        )
    elif args.rank_func == "phoneme":
        ranked_wordlists = rank_dataset(dataset, rank_by_phoneme_editdistance)

    if args.rerank:
        query_texts = [query.query for query in dataset.queries]
        topk_ranked_wordlists = [
            wordlist[: args.rerank_input_size] for wordlist in ranked_wordlists
        ]
        reranked_wordlists = rerank_by_llm(
            query_texts,
            topk_ranked_wordlists,
            topn=args.topn,
            model_name=args.rerank_model_name,
            batch_size=args.rerank_batch_size,
            rerank_interval=args.rerank_interval,
        )
        ranked_wordlists = reranked_wordlists
    positive_texts = [query.positive for query in dataset.queries]
    recall = calculate_recall(ranked_wordlists, positive_texts, args.topn)
    print("Recall: ", recall)

    if args.output_file_path:
        output_path = args.output_file_path
    else:
        output_path = get_default_output_path(
            args.input_path,
            args.rank_func,
            args.topn,
            args.rerank,
            args.rerank_input_size,
            args.rerank_model_name,
        )

    if not args.no_save:
        results = {
            "parameters": {
                "input_path": args.input_path,
                "rank_func": args.rank_func,
                "topn": args.topn,
                "vowel_ratio": args.vowel_ratio
                if args.rank_func in ["kanasim", "vowel_consonant"]
                else None,
                "rerank": args.rerank,
                "rerank_input_size": args.rerank_input_size if args.rerank else None,
                "rerank_model_name": args.rerank_model_name if args.rerank else None,
            },
            "metrics": {
                "recall": recall,
            },
            "results": [
                {
                    "query": query.query,
                    "ranked_words": wordlist[: args.topn],
                    "positive_words": positive_text,
                }
                for query, wordlist, positive_text in zip(
                    dataset.queries, ranked_wordlists, positive_texts
                )
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
