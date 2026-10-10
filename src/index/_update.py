"""Incremental update of an existing index."""

from __future__ import annotations

from tqdm import tqdm
from time import time
from ._helper import get_chunk_content
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .index import Indexer


def update_embedding(self: Indexer) -> Callable[[dict[str, Any]], None]:
    """Build the function that (re)computes the embedding of one chunk.

    Args:
        self: The indexer.

    Returns:
        A function filling the ``embedding`` of a chunk.
    """
    curr_file = ("", "")

    def main_function(chunk: dict[str, Any]) -> None:
        """Compute and store the embedding of a chunk.

        Args:
            chunk: The chunk, as a dictionary.

        Raises:
            RuntimeError: If the indexer has no embedder.
        """
        nonlocal curr_file

        if self._embedder is None:
            raise RuntimeError("No embedder available")

        curr_file, chunk_txt = get_chunk_content(curr_file, chunk)

        chunk["embedding"] = self._embedder.embed(chunk_txt).tolist()

    return main_function


def update_index(self: Indexer) -> None:
    """Update the loaded index: only new or modified files are re-indexed.

    Files that disappeared are dropped. The collection statistics (number
    of chunks, average length, term document frequencies) are recomputed
    from the chunks.

    Args:
        self: The indexer.
    """
    self._result["documents_number"] = 0
    self._result["avg_doc_len"] = 0
    self._result["term_appearances"] = {}

    tokenize_chunk = self._tokenize_chunk()

    paths = list(self.data_collection_path.glob("**/*.md")) + list(
        self.data_collection_path.glob("**/*.py")) + list(
        self.data_collection_path.glob("**/*.txt")
    )

    for stale in set(self._result["files"].keys()).difference(
        {str(path) for path in paths}
    ):
        del self._result["files"][stale]

    update_embedding = self._update_embedding()

    for path in tqdm(
        paths,
        ascii=True,
        desc="Updating",
        unit="file",
    ):
        file = self._result["files"].get(str(path))

        if file is None or file["last_index"] < path.stat().st_mtime:
            self._result["files"][str(path)] = {
                "last_index": time(),
                "chunks": [],
            }
            self._chunk_file(path, self.max_chunk_size)

            file = self._result["files"][str(path)]

            for chunk in file["chunks"]:
                tokenize_chunk(chunk)
        else:
            for chunk in file["chunks"]:
                if self._embedder is None:
                    chunk["embedding"] = None
                elif chunk["embedding"] is None:
                    update_embedding(chunk)

                self._result["documents_number"] += 1
                self._result["avg_doc_len"] += chunk["document_length"]
                for term in chunk["terms"].keys():
                    self._result["term_appearances"][term] = (
                        self._result["term_appearances"].get(term, 0) + 1
                    )

    self._result["avg_doc_len"] /= self._result["documents_number"]
