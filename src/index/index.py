"""Indexer: builds, updates and persists the lexical index."""

from ._chunk import chunk_md, get_chunk_size, chunk_py, chunk_file, chunk
from ._tokenize import tokenize, tokenize_chunk
from ._update import update_index, update_embedding
from src.llm import Embedder
from pydantic import ValidationError
from src.models import Index
from typing import Any, Optional
from pathlib import Path
import pickle


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
        self.data_collection_path = Path(f"data/raw/{data_collection_path}")
        self.max_chunk_size = max_chunk_size

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
                with open(self._index_file, "rb") as f:
                    loaded = Index.model_validate(pickle.load(f))

                if not loaded.files:
                    raise ValueError("Found Empty index file")

                if loaded.max_chunk_size != self.max_chunk_size:
                    raise ValueError("max_chunk_size Changed")

                file_name = list(loaded.files.keys())[0]
                if (
                    Path(__file__).stat().st_mtime
                    > loaded.files[file_name].last_index
                ):
                    raise ValueError("Indexing code was changed")

                self._result = loaded.model_dump()
            except FileNotFoundError:
                print("\n=== Creating Index ===")
                raise
            except ValueError as e:
                print(f"\n=== {e}. Creating new index ===")
                raise
            except (IOError, ValidationError):
                print("\n=== Found Invalid index file. Creating new index ===")
                raise
        except (FileNotFoundError, ValueError, IOError, ValidationError):
            self._init_index()
        else:
            print("\n=== Index File Already Exists! Updating Old Index ===")
            self._update_index()

        with open(self._index_file, "wb") as f:
            pickle.dump(self._result, f)

    def _init_index(self) -> None:
        """Build a brand new index: chunk, then tokenize."""
        self._result = {
            "files": {},
            "term_appearances": {},
            "documents_number": 0,
            "avg_doc_len": 0,
            "max_chunk_size": self.max_chunk_size
        }

        self._chunk()

        self._tokenize()

    _update_embedding = update_embedding

    _update_index = update_index

    _chunk_md = staticmethod(chunk_md)

    _get_chunk_size = staticmethod(get_chunk_size)

    _chunk_py = chunk_py

    _chunk_file = chunk_file

    _chunk = chunk

    _tokenize_chunk = tokenize_chunk

    _tokenize = tokenize
