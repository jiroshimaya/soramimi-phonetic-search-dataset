import time
from typing import Any, Type

from litellm import batch_completion
from pydantic import BaseModel
from tqdm import tqdm


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
