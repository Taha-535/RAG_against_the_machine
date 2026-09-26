from pydantic import BaseModel, Field
from typing import Optional
from functools import total_ordering
from typing import List, Dict
import uuid


@total_ordering
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    document_length: Optional[int] = None
    embedding: Optional[list[float]] = None
    terms: Optional[Dict[str, int]] = None

    def _key(self) -> tuple[str, int, int, int]:
        return (
            self.file_path,
            self.first_character_index,
            self.last_character_index,
            self.document_length,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MinimalSource):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MinimalSource):
            return NotImplemented
        return self._key() < other._key()

    def __str__(self) -> str:
        return f"{self.file_path}[{self.first_character_index}:{self.last_character_index}]"


class FileIndex(BaseModel):
    last_index: float
    chunks: List[MinimalSource]


class Index(BaseModel):
    files: Dict[str, FileIndex]
    term_appearances: Dict[str, int]
    documents_number: int
    avg_doc_len: float


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    # question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # question: str
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
