"""Pydantic data models exchanged between the stages of the RAG pipeline.

The models named in the subject (``MinimalSource``, ``UnansweredQuestion``,
``AnsweredQuestion``, ``RagDataset``, ``MinimalSearchResults``,
``MinimalAnswer``, ``StudentSearchResults`` and
``StudentSearchResultsAndAnswer``) are extended with a few optional fields
used internally by the index (term counts, embeddings, ...).
"""

from pydantic import BaseModel, Field
from typing import Optional
from functools import total_ordering
from typing import List, Dict
import uuid


@total_ordering
class MinimalSource(BaseModel):
    """A chunk of a source file, identified by its character range.

    Attributes:
        file_path: Path of the file, written exactly as in the corpus.
        first_character_index: Index of the first character of the chunk.
        last_character_index: Index of the last character (inclusive).
        document_length: Length of the chunk in characters (index only).
        embedding: Sentence embedding of the chunk (index only, optional).
        terms: Term frequencies of the chunk (index only).
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    document_length: Optional[int] = None
    embedding: Optional[list[float]] = None
    terms: Optional[Dict[str, int]] = None

    def _key(self) -> tuple[str, int, int, int]:
        """Build the tuple used for equality and ordering.

        A missing ``document_length`` is mapped to ``-1`` so that keys are
        always comparable.

        Returns:
            A ``(file_path, first, last, document_length)`` tuple.
        """
        return (
            self.file_path,
            self.first_character_index,
            self.last_character_index,
            -1 if self.document_length is None else self.document_length,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two sources on their location and length."""
        if not isinstance(other, MinimalSource):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: object) -> bool:
        """Order two sources by path, then by character range."""
        if not isinstance(other, MinimalSource):
            return NotImplemented
        return self._key() < other._key()

    def __str__(self) -> str:
        """Return the ``path[first:last]`` representation."""
        return (
            f"{self.file_path}[{self.first_character_index}:"
            f"{self.last_character_index}]"
        )


class FileIndex(BaseModel):
    """Index entry of one file.

    Attributes:
        last_index: Timestamp (``time.time()``) of the last indexing.
        chunks: The chunks the file was split into.
    """

    last_index: float
    chunks: List[MinimalSource]


class Index(BaseModel):
    """The whole persisted lexical index.

    Attributes:
        files: Index entries, keyed by file path.
        term_appearances: Number of chunks containing each term.
        documents_number: Total number of chunks.
        avg_doc_len: Average chunk length, in characters.
    """

    files: Dict[str, FileIndex]
    term_appearances: Dict[str, int]
    documents_number: int
    avg_doc_len: float
    max_chunk_size: int


class UnansweredQuestion(BaseModel):
    """A question without ground truth.

    Attributes:
        question_id: Unique identifier (a UUID4 by default).
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer.

    Attributes:
        sources: Ground-truth source locations.
        answer: Ground-truth answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not.

    Attributes:
        rag_questions: The questions of the dataset.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieval result for one question.

    Attributes:
        question_id: Identifier of the question.
        question: The question text.
        retrieved_sources: Ranked sources, best first.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval result for one question, plus the generated answer.

    Attributes:
        answer: The answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``.

    Attributes:
        search_results: One result per question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``.

    Attributes:
        search_results: One result and answer per question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int


class QueryAnswer(BaseModel):
    """Answer to a single query (used by the HTTP API).

    Attributes:
        question: The query text.
        retrieved_sources: Ranked sources, best first.
        answer: The generated answer.
    """

    question: str
    retrieved_sources: List[MinimalSource]
    answer: str
