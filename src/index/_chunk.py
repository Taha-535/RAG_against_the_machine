from ast import AST, parse
from tqdm import tqdm
from pathlib import Path
from time import time
from collections import deque


def chunk_md(n: int, txt_len: int) -> list[tuple[int, int]]:
    result = []

    for k in range(txt_len // n + 1):
        fst = k * n
        lst = fst + n - 1 if fst + n < txt_len else txt_len - 1

        result.append((fst, lst))

    return result


def get_chunk_size(
    new_lines: list[int], node1: AST, node2: AST = None
) -> tuple[int, int, int]:
    bl, bc = node1.lineno, node1.col_offset

    if node2 is None:
        el, ec = node1.end_lineno, node1.end_col_offset
    else:
        el, ec = node2.end_lineno, node2.end_col_offset

    bi = new_lines[bl - 2] + bc + 1 if bl > 1 else bc
    ei = new_lines[el - 2] + ec if el > 1 else ec

    return (bi, ei, ei - bi + 1)


def chunk_py(self, path: Path, n: int) -> list[tuple[int, int]]:
    with open(path) as f:
        content = f.read()

    new_lines = []
    for i, c in enumerate(content):
        if c == "\n":
            new_lines.append(i)

    groups = deque(parse(content).body)

    result = []

    while groups:
        curr = groups.popleft()

        if self._get_chunk_size(new_lines, curr)[2] > n:
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


def chunk_file(self, path: Path, n: int):
    if str(path).endswith(".md"):
        chunks = self._chunk_md(n, path.stat().st_size)
    else:
        chunks = self._chunk_py(path, n)

    if chunks is None:
        print(str(path))
        exit()
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


def chunk(self):
    for path in tqdm(
        list(self.data_collection_path.glob("**/*.md"))
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
