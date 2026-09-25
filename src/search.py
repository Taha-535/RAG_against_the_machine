from src.models import (
    Index,
    UnansweredQuestion,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    MinimalSearchResults,
)
from src.llm import LLModel, Embedder
from pydantic import ValidationError
from heapq import heappush, heappop
from tqdm import tqdm
import numpy as np
import pickle
import sys


class _BM25:
    def __init__(self, N: int, n: dict[int, int], avgdl: float) -> None:
        self.N = N
        self.n = n
        self.avgdl = avgdl

    def _idf(self, term: int) -> float:
        return np.log(
            (self.N - self.n.get(term, 0) + 0.5) / (self.n.get(term, 0) + 0.5)
        )

    @staticmethod
    def _rtf(term: str, document: dict[str, int], len_norm: float) -> float:
        k = 1.5
        freq = document.get(term, 0)

        return (freq * (k + 1)) / (freq + k * len_norm)

    def _len_norm(self, doc_len: int) -> float:
        b = 0.75

        return 1 - b + (b * doc_len / self.avgdl)

    def bm25(
        self, query: list[str], doc: dict[str, int], doc_len: int
    ) -> float:
        return sum(
            self._idf(term) * self._rtf(term, doc, self._len_norm(doc_len))
            for term in query
        )


class Retriever:
    def __init__(self, split_terms: callable) -> None:
        try:
            with open("data/processed/index", "rb") as f:
                self.index = Index.model_validate(pickle.load(f))
        except (IOError, pickle.PickleError, ValidationError) as e:
            print("Error loading Index!!", file=sys.stderr)
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            exit(2)

        self._split_terms = split_terms

        self._bm25 = _BM25(
            self.index.documents_number,
            self.index.term_appearances,
            self.index.avg_doc_len,
        )

        self._embedder = Embedder()

    def score(self, query: str, k: int) -> list[tuple[int, MinimalSource]]:
        query_token_ids = self._split_terms(query)
        score_heap = []

        for chunk in (
            chunk
            for file in self.index.files.values()
            for chunk in file.chunks
        ):
            query_embed = self._embedder.embed(query)
            heappush(
                score_heap,
                (
                    -self._bm25.bm25(
                        query_token_ids, chunk.terms, chunk.document_length
                    ) * self._embedder.similarity(query_embed, chunk.embedding),
                    chunk,
                ),
            )

        return [
            heappop(score_heap)[1]
            for _ in range(k if len(score_heap) >= k else len(score_heap))
        ]

    def search_dataset(
        self, dataset: RagDataset, k: int
    ) -> StudentSearchResults:
        search_results = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching",
            ascii=True,
            unit="question",
        ):
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=self.score(question.question, k),
                )
            )

        return StudentSearchResults(search_results=search_results, k=k)
