from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    document_length: int
    terms: dict[int, int]

    def __str__(self) -> str:
        return (
            f"[{self.last_character_index - self.first_character_index}]"
            f" chars: {self.file_path}"
        )


class FileIndex(BaseModel):
    last_index: float
    chunks: list[MinimalSource]


class Index(BaseModel):
    files: dict[str, FileIndex]
    term_appearances: dict[int, int]
    documents_number: int
    avg_doc_len: float
