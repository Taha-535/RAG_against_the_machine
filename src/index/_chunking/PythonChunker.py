from collections import deque
from .Chunker import Chunker
from ast import parse, stmt
from typing import Optional
from pathlib import Path


class PythonChunker(Chunker):
    def chunk(self, path: Path) -> list[tuple[int, int]]:
        with open(path) as f:
            content = f.read()

        if not content:
            return []

        line_ends = self._get_line_ends(content)

        groups: deque[stmt] = deque(parse(content).body)

        result: list[tuple[int, int]] = []

        n = self.max_chunk_size

        while groups:
            curr = groups.popleft()

            if self._get_chunk_size(line_ends, curr)[2] > n:
                children = self._get_node_children(curr)

                if children:
                    for node in reversed(children):
                        groups.appendleft(node)
                else:
                    result.extend(
                        self._character_based_fullback(
                            *self._get_chunk_size(line_ends, curr)
                        )
                    )
            else:
                beg_idx, end_idx, merge_idx = self._merge_small_chunks(
                    groups, curr, line_ends
                )

                result.append((beg_idx, end_idx))

                for _ in range(0, merge_idx + 1):
                    groups.popleft()

        return result

    @staticmethod
    def _get_chunk_size(
        new_lines: list[int], node1: stmt, node2: Optional[stmt] = None
    ) -> tuple[int, int, int]:
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
        end_idx = (
            new_lines[end_line - 2] + end_col if end_line > 1 else end_col
        )

        return (beg_idx, end_idx, end_idx - beg_idx + 1)

    @staticmethod
    def _get_node_children(node: stmt) -> list[stmt]:
        children = []

        for attr in ("body", "handlers", "orelse", "finalbody", "cases"):
            if hasattr(node, attr):
                val = getattr(node, attr)
                if isinstance(val, list):
                    children.extend(val)

        return children

    def _character_based_fullback(
        self, beg_idx: int, end_idx: int, chunk_len: int
    ) -> list[tuple[int, int]]:

        n = self.max_chunk_size

        subchunks_num = 2

        while chunk_len / subchunks_num > n:
            subchunks_num += 1

        step = (chunk_len + subchunks_num - 1) // subchunks_num

        return [
            (fst, min(end_idx, fst + step - 1))
            for fst in range(beg_idx, end_idx + 1, step)
        ]

    def _merge_small_chunks(
        self, groups: deque[stmt], curr: stmt, line_ends: list[int]
    ) -> tuple[int, int, int]:
        n = self.max_chunk_size

        merge_idx = -1

        while (
            merge_idx + 1 < len(groups)
            and self._get_chunk_size(line_ends, curr, groups[merge_idx + 1])[2]
            <= n
        ):
            merge_idx += 1

        merge = groups[merge_idx] if merge_idx != -1 else curr

        beg_idx, end_idx, _ = self._get_chunk_size(line_ends, curr, merge)

        return (beg_idx, end_idx, merge_idx)

    @staticmethod
    def _get_line_ends(text: str) -> list[int]:
        return [i for i, c in enumerate(text) if c == '\n']
