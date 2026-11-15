from abc import ABC, abstractmethod
from pathlib import Path


class Chunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, path: Path) -> list[tuple[int, int]]: ...
