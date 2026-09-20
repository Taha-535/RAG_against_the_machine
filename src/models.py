from pydantic import BaseModel
import uuid


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    document_length: int
    terms: dict[str, int]

    def __str__(self) -> str:
        return (
            f"[{self.last_character_index - self.first_character_index}]"
            f" chars: {self.file_path}"
        )


class FileIndex(BaseModel):
    file_path: str
    last_index: int
    chunks: list[MinimalSource]


class Index(BaseModel):
    files: list[FileIndex]
    term_appearances: dict[str, int] # to change
    documents_number: int
    avg_doc_len: float


class UnansweredQuestion(BaseModel): 
    question_id: str = Field(default_factory(lambda: str(uuid.uuid4())))
    question: str
