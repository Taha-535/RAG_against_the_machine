"""Tokenization step: term counts and optional embeddings of each chunk."""

from __future__ import annotations

from tqdm import tqdm
from ._helper import get_chunk_content
from src.helper import term_splitter
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .index import Indexer


def tokenize_chunk(self: Indexer) -> Callable[[dict[str, Any]], None]:
    """Build the function that tokenizes one chunk.

    Args:
        self: The indexer.

    Returns:
        A function filling the ``terms`` (and ``embedding``) of a chunk and
        updating the document frequencies of the index being built.
    """
    curr_file = ("", "")

    def perform_tokenization(chunk: dict[str, Any]) -> None:
        """Fill the terms and the embedding of a chunk.

        Args:
            chunk: The chunk, as a dictionary.
        """
        nonlocal curr_file

        curr_file, chunk_txt = get_chunk_content(curr_file, chunk)

        chunk["embedding"] = (
            self._embedder.embed(chunk_txt).tolist()
            if self._embedder is not None
            else None
        )

        for token_id in term_splitter(chunk_txt):
            chunk["terms"][token_id] = chunk["terms"].get(token_id, 0) + 1
            if chunk["terms"][token_id] == 1:
                self._result["term_appearances"][token_id] = (
                    self._result["term_appearances"].get(token_id, 0) + 1
                )

    return perform_tokenization


def tokenize(self: Indexer) -> None:
    """Tokenize every chunk of the index being built.

    Args:
        self: The indexer.
    """
    tokenize_chunk = self._tokenize_chunk()

    for chunk in tqdm(
        (
            chunk
            for file in self._result["files"].values()
            for chunk in file["chunks"]
        ),
        ascii=True,
        desc="Tokenizing",
        unit="chunk",
        total=self._result["documents_number"],
    ):
        tokenize_chunk(chunk)
