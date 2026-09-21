from src.models import Index, UnansweredQuestion, MinimalSource
from pydantic import ValidationError
from heapq import heappush, heappop
from src.llm import LLModel
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
    def _rtf(term: int, document: dict[int, int]) -> float:
        k = 1.5
        freq = document.get(term, 0)

        return (freq * (k + 1)) / (freq + k)

    def _len_norm(self, doc_len: int) -> float:
        b = 0.75

        return 1 - b + (b * doc_len / self.avgdl)

    def bm25(
        self, query: list[int], doc: dict[int, int], doc_len: int
    ) -> float:
        return sum(
            self._idf(term) * self._rtf(term, doc) / self._len_norm(doc_len)
            for term in query
        )


class Retriever:
    def __init__(self, model: LLModel) -> None:
        try:
            with open("data/processed/index", 'rb') as f:
                self.index = Index.model_validate(pickle.load(f))
        except (IOError, pickle.PickleError, ValidationError) as e:
            print("Error loading Index!!", file=sys.stderr)
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            exit(2)

        self._model = model

        self._bm25 = _BM25(
            self.index.documents_number,
            self.index.term_appearances,
            self.index.avg_doc_len,
        )

    def score(self, query: str, k: int) -> list[tuple[int, MinimalSource]]:
        query_token_ids = self._model.encode(query)
        score_heap = []

        for chunk in tqdm(
            (
                chunk
                for file in self.index.files.values()
                for chunk in file.chunks
            ),
            total=self.index.documents_number,
            unit="document",
            ascii=True,
        ):
            heappush(
                score_heap,
                (
                    self._bm25.bm25(
                        query_token_ids, chunk.terms, chunk.document_length
                    ),
                    chunk,
                ),
            )

        return [
            heappop(score_heap)[1]
            for _ in range(k if len(score_heap) >= k else len(score_heap))
        ]
