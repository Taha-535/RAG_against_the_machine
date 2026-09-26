from ._chunk import chunk_md, get_chunk_size, chunk_py, chunk_file, chunk
from ._tokenize import tokenize, tokenize_chunk
from ._update import update_index, update_embedding
from src.llm import Embedder
from pydantic import ValidationError
from src.models import Index
from typing import Optional
from pathlib import Path
import pickle


class Indexer:
    def __init__(
        self,
        data_collection_path: str,
        max_chunk_size: int,
        embed: bool,
    ) -> None:
        self.data_collection_path = Path(f"data/raw/{data_collection_path}")
        self.max_chunk_size = max_chunk_size

        self._index_file = f"data/processed/index.pkl"

        self._embedder: Optional[Embedder] = Embedder() if embed else None

    def index(self):
        try:
            try:
                with open(self._index_file, "rb") as f:
                    loaded = Index.model_validate(pickle.load(f))

                if not loaded.files:
                    raise ValueError("Found Empty index file")

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
        self._result = {
            "files": {},
            "term_appearances": {},
            "documents_number": 0,
            "avg_doc_len": 0,
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
