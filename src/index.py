from src.llm import LLModel, Embedder
from pydantic import ValidationError
from collections import deque
from src.models import Index
from typing import Optional
from ast import parse, AST
from pathlib import Path
from time import time
from tqdm import tqdm
import pickle
import sys
import os


class Indexer:
    def __init__(
        self,
        split_terms: callable,
        data_collection_path: str,
        index_path: str,
        max_chunk_size: int,
        embed: bool,
    ) -> None:
        self.data_collection_path = Path(f"data/raw/{data_collection_path}")
        self.max_chunk_size = max_chunk_size

        self._index_file = f"data/processed/{index_path}"

        self._split_terms = split_terms

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

    def _update_embedding(self) -> callable:
        curr_file = ("", "")

        def main_function(chunk) -> list[float]:
            nonlocal curr_file
            nonlocal self

            if curr_file[0] != chunk["file_path"]:
                if curr_file[1] is None:
                    return

                try:
                    with open(chunk["file_path"]) as f:
                        curr_file = (chunk["file_path"], f.read())
                except IOError as e:
                    print(
                        f"[WARNING] {type(e)}: {e}: Couldn't read file {chunk['file_path']}",
                    )
                    curr_file = (chunk["file_path"], None)
                    return

            chunk_txt = curr_file[1][
                chunk["first_character_index"] : chunk["last_character_index"]
                + 1
            ]

            chunk["embedding"] = self._embedder.embed(chunk_txt).tolist()

        return main_function

    def _update_index(self) -> None:
        self._result["documents_number"] = 0
        self._result["avg_doc_len"] = 0
        self._result["term_appearances"] = {}

        tokenize_chunk = self._tokenize_chunk()

        paths = list(self.data_collection_path.glob("**/*.md")) + list(
            self.data_collection_path.glob("**/*.py")
        )

        for file in set(self._result["files"].keys()).difference(
            {str(path) for path in paths}
        ):
            del self._result["files"][file]

        update_embedding = self._update_embedding()

        for path in tqdm(
            paths,
            ascii=True,
            desc="Updating",
            unit="file",
        ):
            file = self._result["files"].get(str(path))

            if file is None or file["last_index"] < path.stat().st_mtime:
                self._chunk_file(
                    path,
                    self.max_chunk_size,
                    path.stat().st_size,
                )

                if file is None:
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

    @staticmethod
    def _chunk_md(n: int, txt_len: int) -> list[tuple[int, int]]:
        result = []

        for k in range(txt_len // n + 1):
            fst = k * n
            lst = fst + n - 1 if fst + n < txt_len else txt_len - 1

            result.append((fst, lst))

        return result

    @staticmethod
    def _get_chunk_size(new_lines: list[int], node1: AST, node2: AST = None) -> tuple[int, int, int]:
        bl, bc = node1.lineno, node1.col_offset

        if node2 is None:
            el, ec = node1.end_lineno, node1.end_col_offset
        else:
            el, ec = node2.end_lineno, node2.end_col_offset

        bi = new_lines[bl - 2] + bc + 1 if bl > 1 else bc
        ei = new_lines[el - 2] + ec if el > 1 else ec

        return (bi, ei, ei - bi + 1)


    def _chunk_py(self, path: Path, n: int) -> list[tuple[int, int]]:
        with open(path) as f:
            content = f.read()

        new_lines = []
        for i, c in enumerate(content):
            if c == '\n':
                new_lines.append(i)

        groups = deque(parse(content).body)

        result = []

        while groups:
            curr = groups.popleft()

            if self._get_chunk_size(new_lines, curr)[2] > n:
                if hasattr(curr, 'body'):
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
                    and self._get_chunk_size(new_lines, curr, groups[merge_idx + 1])[2]
                    <= n
                ):
                    merge_idx += 1

                merge = groups[merge_idx] if merge_idx != -1 else curr

                bi, ei, _ = self._get_chunk_size(new_lines, curr, merge)

                result.append((bi, ei))

                for _ in range(0, merge_idx + 1):
                    groups.popleft()

        return result

    def _chunk_file(self, path: Path, n: int):
        if str(path).endswith('.md'):
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

    def _chunk(self):
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

    def _tokenize_chunk(self):
        curr_file = ("", "")

        def perform_tokenization(chunk):
            nonlocal curr_file
            nonlocal self

            if curr_file[0] != chunk["file_path"]:
                if curr_file[1] is None:
                    return

                try:
                    with open(chunk["file_path"]) as f:
                        curr_file = (chunk["file_path"], f.read())
                except IOError as e:
                    print(
                        f"[WARNING] {type(e)}: {e}: Couldn't read file {chunk['file_path']}",
                    )
                    curr_file = (chunk["file_path"], None)
                    return

            chunk_txt = curr_file[1][
                chunk["first_character_index"] : chunk["last_character_index"]
                + 1
            ]

            chunk["embedding"] = (
                self._embedder.embed(chunk_txt).tolist()
                if self._embedder is not None
                else None
            )

            for token_id in self._split_terms(chunk_txt):
                chunk["terms"][token_id] = chunk["terms"].get(token_id, 0) + 1
                if chunk["terms"][token_id] == 1:
                    self._result["term_appearances"][token_id] = (
                        self._result["term_appearances"].get(token_id, 0) + 1
                    )

        return perform_tokenization

    def _tokenize(self):
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
