"""Chunking strategies: fixed-size for Markdown, AST-based for Python.

The functions taking ``self`` are attached as methods of ``Indexer``.
"""

from __future__ import annotations

from ast import parse, stmt
from tqdm import tqdm
from pathlib import Path
from time import time
from collections import deque
from typing import Optional, TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from .index import Indexer


def chunk_txt(path: Path, n: int) -> list[tuple[int, int]]:
    """Split a Markdown/text file into consecutive fixed-size windows.

    Args:
        n: Maximum chunk size, in characters.
        txt_len: Size of the file.

    Returns:
        ``(first, last)`` inclusive character ranges covering the file.
    """
    with open(path) as f:
        txt_len = len(f.read())

    if not txt_len:
        return []

    return [(fst, min(fst + n, txt_len) - 1) for fst in range(0, txt_len, n)]


def get_chunk_size(
    new_lines: list[int], node1: stmt, node2: Optional[stmt] = None
) -> tuple[int, int, int]:
    """Compute the character span of one AST node, or of a node range.

    Args:
        new_lines: Character index of every ``\\n`` of the file.
        node1: First node of the span.
        node2: Last node of the span. When ``None``, the span is the one
            of ``node1`` alone.

    Returns:
        A ``(first, last, length)`` tuple, indices being inclusive.
    """
    end = node1 if node2 is None else node2

    beg_line, beg_col = node1.lineno, node1.col_offset
    end_line = end.end_lineno if end.end_lineno is not None else end.lineno
    end_col = (
        end.end_col_offset
        if end.end_col_offset is not None
        else end.col_offset
    )

    beg_idx = (
        new_lines[beg_line - 2] + beg_col + 1 if beg_line > 1 else beg_col
    )
    end_idx = new_lines[end_line - 2] + end_col if end_line > 1 else end_col

    return (beg_idx, end_idx, end_idx - beg_idx + 1)


def chunk_py(self: Indexer, path: Path, n: int) -> list[tuple[int, int]]:
    """Split a Python file into chunks following its syntax tree.

    Top-level statements are merged greedily while the merged span fits in
    ``n`` characters. A statement that is too long is replaced by its
    children if it has a body, or cut into slices otherwise.

    Args:
        self: The indexer.
        path: Path of the Python file.
        n: Maximum chunk size, in characters.

    Returns:
        ``(first, last)`` inclusive character ranges.
    """
    with open(path) as f:
        content = f.read()

    if not content:
        return []

    new_lines = []
    for i, c in enumerate(content):
        if c == "\n":
            new_lines.append(i)

    groups: deque[stmt] = deque(parse(content).body)

    result: list[tuple[int, int]] = []

    while groups:
        curr = groups.popleft()

        if self._get_chunk_size(new_lines, curr)[2] >= n:
            if hasattr(curr, "body"):
                for node in curr.body[::-1]:
                    groups.appendleft(node)
            else:
                bi, ei, chunk_len = self._get_chunk_size(new_lines, curr)
                i = 2
                while chunk_len / i > n:
                    i += 1

                for j in range(i + 1):
                    fst = bi + j * (chunk_len // i)
                    lst = min(ei, fst + (chunk_len // i))

                    result.append((fst, lst))

        else:
            merge_idx = -1
            while (
                merge_idx + 1 < len(groups)
                and self._get_chunk_size(
                    new_lines, curr, groups[merge_idx + 1]
                )[2]
                <= n
            ):
                merge_idx += 1

            merge = groups[merge_idx] if merge_idx != -1 else curr

            bi, ei, _ = self._get_chunk_size(new_lines, curr, merge)

            result.append((bi, ei))

            for _ in range(0, merge_idx + 1):
                groups.popleft()

    return result


def chunk_file(self: Indexer, path: Path, n: int) -> None:
    """Chunk one file and register its chunks in the index being built.

    Markdown files use ``chunk_md``; every other file is treated as Python.
    The entry ``self._result["files"][str(path)]`` must already exist.

    Args:
        self: The indexer.
        path: Path of the file.
        n: Maximum chunk size, in characters.
    """
    try:
        if path.name.endswith(".py"):
            chunks = self._chunk_py(path, n)
        else:
            chunks = self._chunk_txt(path, n)
    except (SyntaxError, IOError) as e:
        print(f"Error reading file: {type(e).__name__}: {e}", file=sys.stderr)
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


def chunk(self: Indexer) -> None:
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
        self._result["files"][str(path)] = {"last_index": time(), "chunks": []}

        self._chunk_file(
            path,
            self.max_chunk_size,
        )

    self._result["avg_doc_len"] /= self._result["documents_number"]
