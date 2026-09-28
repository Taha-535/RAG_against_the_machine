"""BM25 retrieval over the persisted index."""

from src.models import (
    Index,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    MinimalSearchResults,
)
from src.helper import term_splitter
from src.llm import Embedder
from numpy.typing import NDArray
from pydantic import ValidationError
from heapq import heappush, heappop
from typing import Any, Optional
from tqdm import tqdm
import numpy as np
import pickle
import sys


class _BM25:
    """Okapi BM25 scorer (``k1 = 1.5``, ``b = 0.75``)."""

    def __init__(self, N: int, n: dict[str, int], avgdl: float) -> None:
        """Store the collection statistics.

        Args:
            N: Total number of chunks in the collection.
            n: Number of chunks containing each term.
            avgdl: Average chunk length, in characters.
        """
        self.N = N
        self.n = n
        self.avgdl = avgdl

    def _idf(self, term: str) -> float:
        """Compute the inverse document frequency of a term.

        Args:
            term: The term.

        Returns:
            ``ln((N - n + 0.5) / (n + 0.5))``, where ``n`` is the number of
            chunks containing the term.
        """
        return float(
            np.log(
                (self.N - self.n.get(term, 0) + 0.5)
                / (self.n.get(term, 0) + 0.5)
            )
        )

    @staticmethod
    def _rtf(term: str, document: dict[str, int], len_norm: float) -> float:
        """Compute the saturated term frequency of a term in a chunk.

        Args:
            term: The term.
            document: Term frequencies of the chunk.
            len_norm: Length normalisation factor of the chunk.

        Returns:
            ``f * (k + 1) / (f + k * len_norm)``.
        """
        k = 1.5
        freq = document.get(term, 0)

        return (freq * (k + 1)) / (freq + k * len_norm)

    def _len_norm(self, doc_len: int) -> float:
        """Compute the length normalisation factor of a chunk.

        Args:
            doc_len: Length of the chunk, in characters.

        Returns:
            ``1 - b + b * doc_len / avgdl``.
        """
        b = 0.75

        return 1 - b + (b * doc_len / self.avgdl)

    def bm25(
        self, query: list[str], doc: dict[str, int], doc_len: int
    ) -> float:
        """Score a chunk against a query.

        Args:
            query: The query terms.
            doc: Term frequencies of the chunk.
            doc_len: Length of the chunk, in characters.

        Returns:
            The BM25 score of the chunk.
        """
        return sum(
            self._idf(term) * self._rtf(term, doc, self._len_norm(doc_len))
            for term in query
        )


class Retriever:
    """Retrieve the most relevant chunks of the index for a query."""

    def __init__(self) -> None:
        """Load the index from ``data/processed/index.pkl``.

        The program exits with status 2 if the index cannot be loaded.
        """
        try:
            with open("data/processed/index.pkl", "rb") as f:
                self.index = Index.model_validate(pickle.load(f))
        except (IOError, pickle.PickleError, ValidationError) as e:
            print("Error loading Index!!", file=sys.stderr)
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            exit(2)

        self._bm25 = _BM25(
            self.index.documents_number,
            self.index.term_appearances,
            self.index.avg_doc_len,
        )

        self._embedder: Optional[Embedder] = None

    def score(self, query: str, k: int, embed: bool) -> list[MinimalSource]:
        """Return the top-k chunks for a query.

        Every chunk is scored with BM25. When ``embed`` is true, the score
        is multiplied by the cosine similarity between the query and the
        chunk embeddings.

        Args:
            query: The query text.
            k: Number of chunks to return.
            embed: Whether to weight the BM25 score with the semantic
                similarity (the index must have been built with
                embeddings).

        Returns:
            At most ``k`` chunks, best first.
        """
        if embed and self._embedder is None:
            self._embedder = Embedder()

        query_terms = term_splitter(query)
        query_embed: Optional[NDArray[Any]] = (
            self._embedder.embed(query)
            if embed and self._embedder is not None
            else None
        )
        score_heap: list[tuple[float, MinimalSource]] = []

        for chunk in (
            chunk
            for file in self.index.files.values()
            for chunk in file.chunks
        ):
            score = self._bm25.bm25(
                query_terms, chunk.terms or {}, chunk.document_length or 0
            )

            if query_embed is not None and self._embedder is not None:
                score *= self._embedder.similarity(
                    query_embed, chunk.embedding
                )

            heappush(
                score_heap,
                (
                    -score,
                    chunk,
                ),
            )

        return [
            heappop(score_heap)[1]
            for _ in range(k if len(score_heap) >= k else len(score_heap))
        ]

    def search_dataset(
        self, dataset: RagDataset, k: int, embed: bool
    ) -> StudentSearchResults:
        """Retrieve the top-k chunks for every question of a dataset.

        Args:
            dataset: The questions to search.
            k: Number of chunks to retrieve per question.
            embed: Whether to use the semantic similarity (see ``score``).

        Returns:
            The search results of every question.
        """
        if embed and self._embedder is None:
            self._embedder = Embedder()

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
                    retrieved_sources=self.score(question.question, k, embed),
                )
            )

        return StudentSearchResults(search_results=search_results, k=k)
