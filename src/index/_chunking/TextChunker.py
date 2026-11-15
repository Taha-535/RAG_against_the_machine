from .Chunker import Chunker
from pathlib import Path


class TextChunker(Chunker):
    def chunk(self, path: Path) -> list[tuple[int, int]]:
        n = self.max_chunk_size

        with open(path) as f:
            txt_len = len(f.read())

        if not txt_len:
            return []

        return [
            (fst, min(fst + n, txt_len) - 1) for fst in range(0, txt_len, n)
        ]
