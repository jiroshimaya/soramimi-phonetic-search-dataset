from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PhoneticSearchQuery:
    query: str
    positive: list[str]
    hard_negatives: list[str] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None

    def build_wordlist_for_llm(self, *, wordlist_size: int = 100) -> list[str]:
        if wordlist_size <= 0:
            raise ValueError("wordlist_size must be a positive integer")
        if self.hard_negatives is None:
            raise ValueError("hard_negatives are required to build an LLM wordlist")

        positive_count = len(self.positive)
        if positive_count > wordlist_size:
            raise ValueError(
                "wordlist_size must be greater than or equal to the number of positive words"
            )

        required_hard_negative_count = wordlist_size - positive_count
        if len(self.hard_negatives) < required_hard_negative_count:
            raise ValueError(
                "hard_negatives must contain enough words to fill the LLM wordlist"
            )

        return sorted(
            [
                *self.hard_negatives[:required_hard_negative_count],
                *self.positive,
            ]
        )


@dataclass
class PhoneticSearchDataset:
    queries: list[PhoneticSearchQuery]
    words: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchDataset":
        queries = [PhoneticSearchQuery(**query) for query in data["queries"]]
        words = data["words"]
        metadata = data.get("metadata", {})
        return cls(queries=queries, words=words, metadata=metadata)


@dataclass
class PhoneticSearchQueryWithWordlist:
    query: str
    wordlist: list[str]
    positive_words: list[str]


@dataclass
class PhoneticSearchWordlistDataset:
    queries: list[PhoneticSearchQueryWithWordlist]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhoneticSearchResult:
    query: str
    ranked_words: list[str]
    positive_words: list[str]
    metadata: dict[str, Any] | None = None


@dataclass
class PhoneticSearchMetrics:
    recall: float
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchMetrics":
        metadata = data.get("metadata", {})
        extra_metadata = {
            key: value
            for key, value in data.items()
            if key not in {"recall", "execution_time", "metadata"}
        }
        return cls(
            recall=data["recall"],
            execution_time=data["execution_time"],
            metadata={**extra_metadata, **metadata},
        )


@dataclass
class PhoneticSearchParameters:
    topn: int
    rank_func: str
    execution_timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def attach_metadata(self, metadata: dict[str, Any]) -> "PhoneticSearchParameters":
        self.metadata.update(metadata)
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchParameters":
        metadata = data.get("metadata", {})
        extra_metadata = {
            key: value
            for key, value in data.items()
            if key not in {"topn", "rank_func", "execution_timestamp", "metadata"}
        }
        return cls(
            topn=data["topn"],
            rank_func=data["rank_func"],
            execution_timestamp=data.get("execution_timestamp"),
            metadata={**extra_metadata, **metadata},
        )


@dataclass
class PhoneticSearchResults:
    parameters: PhoneticSearchParameters
    metrics: PhoneticSearchMetrics
    results: list[PhoneticSearchResult]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneticSearchResults":
        results = [PhoneticSearchResult(**result) for result in data["results"]]
        return cls(
            parameters=PhoneticSearchParameters.from_dict(data["parameters"]),
            metrics=PhoneticSearchMetrics.from_dict(data["metrics"]),
            results=results,
        )
