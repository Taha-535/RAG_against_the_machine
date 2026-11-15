"""Indexer: builds, updates and persists the lexical index."""

from ._chunking import Chunker, PythonChunker, TextChunker
from ._update import update_index, update_embedding
from ._tokenize import tokenize, tokenize_chunk
from pydantic import ValidationError
from typing import Any, Optional
from src.llm import Embedder
from src.models import Index
from pathlib import Path
from tqdm import tqdm
from time import time
import pickle
import sys


class Indexer:
    """Chunk a corpus, tokenize the chunks and persist the index.

    The index is saved as a pickle in ``data/processed/index.pkl``. If it
    already exists and is valid, it is updated instead of rebuilt.
    """

    def __init__(
        self,
        data_collection_path: str,
        max_chunk_size: int,
        embed: bool,
    ) -> None:
        """Configure the indexer.

        Args:
            data_collection_path: Name of the corpus folder in
                ``data/raw/``.
            max_chunk_size: Maximum chunk size, in characters.
            embed: Whether to compute a sentence embedding per chunk.
        """
        self.data_collection_path = Path(f"{data_collection_path}")

        if not self.data_collection_path.is_dir():
            raise IOError(
                f"Couldn't find directory '{str(self.data_collection_path)}'"
            )

        self.max_chunk_size = max_chunk_size

        self._chunkers: dict[str, Chunker] = {
            ".py": PythonChunker(max_chunk_size),
            ".txt": TextChunker(max_chunk_size),
            ".md": TextChunker(max_chunk_size),
        }

        self._index_file = "data/processed/index.pkl"

        self._embedder: Optional[Embedder] = Embedder() if embed else None

        self._result: dict[str, Any] = {}

    def index(self) -> None:
        """Build the index, or update it if a valid one already exists.

        The index is rebuilt from scratch when the file is missing, empty,
        invalid, or older than the indexing code.
        """
        try:
            try:
                if not Path(self._index_file).is_file():
                    raise IOError

                with open(self._index_file, "rb") as f:
                    loaded = Index.model_validate(pickle.load(f))

                if not loaded.files:
                    raise ValueError("Found Empty index file")

                if loaded.max_chunk_size != self.max_chunk_size:
                    raise ValueError("max_chunk_size Changed")
            except FileNotFoundError:
                print("\n=== Creating Index ===")
                raise
            except ValueError as e:
                print(f"\n=== {e}. Creating new index ===")
                raise
            except (IOError, ValidationError):
                print(
                    "\n=== Found Invalid or non-existing index file.",
                    "Creating new index ===",
                )
                raise
        except (FileNotFoundError, ValueError, IOError, ValidationError):
            self._init_index()
        else:
            print("\n=== Index File Already Exists! Updating Old Index ===")

            self._result = loaded.model_dump()

            self._update_index()

        try:
            Path(self._index_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self._index_file, "wb") as f:
                pickle.dump(self._result, f)
        except IOError as e:
            raise type(e)(f"Error writing index: {e}")

    def _init_index(self) -> None:
        """Build a brand new index: chunk, then tokenize."""
        self._result = {
            "files": {},
            "term_appearances": {},
            "documents_number": 0,
            "avg_doc_len": 0,
            "max_chunk_size": self.max_chunk_size,
        }

        self._chunk()

        self._tokenize()

    def _chunk_file(self, path: Path) -> None:
        """Chunk one file and register its chunks in the index being built.

        Markdown files use ``chunk_md``; every other file is treated as Python.
        The entry ``self._result["files"][str(path)]`` must already exist.

        Args:
            self: The indexer.
            path: Path of the file.
            n: Maximum chunk size, in characters.
        """
        try:
            chunks = self._chunkers[path.suffix].chunk(path)
        except (SyntaxError, IOError) as e:
            print(
                f"Error reading file: {type(e).__name__}: {e}", file=sys.stderr
            )
            exit(1)

        if not chunks:
            return

        for fst, lst in chunks:
            self._result["files"][str(path)]["chunks"].append(
                {
                    "file_path": str(path),
                    "first_character_index": fst,
                    "last_character_index": lst,
                    "document_length": lst - fst + 1,
                    "terms": {},
                    "embedding": None,
                }
            )

            self._result["documents_number"] += 1
            self._result["avg_doc_len"] += lst - fst + 1

    def _chunk(self) -> None:
        """Chunk every Markdown and Python file of the corpus.

        Args:
            self: The indexer.
        """
        for path in tqdm(
            list(self.data_collection_path.glob("**/*.md"))
            + list(self.data_collection_path.glob("**/*.txt"))
            + list(self.data_collection_path.glob("**/*.py")),
            ascii=True,
            desc="Chunking",
            unit="file",
        ):
            self._result["files"][str(path)] = {
                "last_index": time(),
                "chunks": [],
            }

            self._chunk_file(path)

        self._result["avg_doc_len"] /= self._result["documents_number"]

    _update_embedding = update_embedding

    _update_index = update_index

    _tokenize_chunk = tokenize_chunk

    _tokenize = tokenize
