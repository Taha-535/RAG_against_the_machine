from pydantic import ValidationError
from src.models import Index
from src.llm import LLModel
from pathlib import Path
from time import time
from tqdm import tqdm
import pickle
import sys
import os


class Indexer:
    def __init__(
        self, model: LLModel, data_collection_path: str, max_chunk_size: int
    ) -> None:
        self.data_collection_path = Path(data_collection_path)
        self.max_chunk_size = max_chunk_size
        self.model = model

        self._index_file = "data/processed/index"

    def index(self):
        try:
            try:
                with open(self._index_file, 'rb') as f:
                    loaded = Index.model_validate(pickle.load(f))

                if not loaded.files:
                    raise ValueError

                self._result = loaded.model_dump()
            except FileNotFoundError:
                print('\n=== Creating Index ===')
                raise
            except ValueError:
                print('\n=== Found Empty index file. Creating new index ===')
                raise
            except (IOError, ValidationError):
                print('\n=== Found Invalid index file. Creating new index ===')
                raise
        except (FileNotFoundError, ValueError, IOError, ValidationError):
            self._init_index()
        else:
            print('\n=== Index File Already Exists! Updating Old Index ===')
            self._update_index()

        with open(self._index_file, 'wb') as f:
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

    def _update_index(self) -> None:
        self._result["documents_number"] = 0
        self._result["avg_doc_len"] = 0
        self._result["term_appearances"] = {}

        tokenize_chunk = self._tokenize_chunk()

        for file_path, file in self._result['files'].items():
            path = Path(file_path)
            
            if file['last_index'] < path.stat().st_mtime:
                self._chunk_file(
                    path,
                    self.max_chunk_size,
                    20 if self.max_chunk_size >= 100 else 0,
                    path.stat().st_size
                )
                
                for chunk in file['chunks']:
                    tokenize_chunk(chunk)

                print(f"[LOG] Updated '{file_path}' file index")
            else:
                for chunk in file['chunks']:
                    self._result["documents_number"] += 1
                    self._result["avg_doc_len"] += chunk['document_length']
                    for term in chunk['terms'].keys():
                        self._result["term_appearances"][term] = self._result["term_appearances"].get(term, 0) + 1

        self._result["avg_doc_len"] /= self._result["documents_number"]

    def _chunk_file(self, path: Path, n: int, o: int, txt_len: int):

        self._result["files"][str(path)] = {"last_index": time(), "chunks": []}

        # Character Based Chunking

        for k in range(txt_len // n + 1):
            fst = k * (n - o)
            lst = fst + n - 1 if fst + n < txt_len else txt_len - 1

            self._result["files"][str(path)]["chunks"].append(
                {
                    "file_path": str(path),
                    "first_character_index": fst,
                    "last_character_index": lst,
                    "document_length": lst - fst + 1,
                    "terms": {},
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
            self._chunk_file(
                path,
                self.max_chunk_size,
                20 if self.max_chunk_size >= 100 else 0,
                path.stat().st_size
            )

            break

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

            for token_id in self.model.encode(chunk_txt):
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
            total=self._result["documents_number"]
        ):
            tokenize_chunk(chunk)

