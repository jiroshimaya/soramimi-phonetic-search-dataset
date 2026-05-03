from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhoneticSearchQuery:
    query: str
    positive: list[str]


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
