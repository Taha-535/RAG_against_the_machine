from src.models import Index
from src.llm import LLModel
from pathlib import Path
from time import time
from tqdm import tqdm
import json
import sys
import os


class Indexer:
    def __init__(
        self, model: LLModel, data_collection_path: str, max_chunk_size: int
    ) -> None:
        self.data_collection_path = Path(data_collection_path)
        self.max_chunk_size = max_chunk_size
        self.model = model

        self._index_file = "data/processed/index.json"

    def index(self):
        # try:
        #     with open(self._index_file) as f:
        #         self._result = Index(**json.load(f))
        # except (IOError, ValidationError):
        #     print((
        #         "Error Reading Index File!\n"
        #         "Creating a new indexing..."
        #     ))
        self._result = {
            "files": [],
            "term_appearances": {},
            "documents_number": 0,
            "avg_doc_len": 0,
        }

        self._chunk()

        self._tokenize()

        with open(self._index_file, 'w') as f:
            json.dump(self._result, f, indent=4)


    def _chunk(self):
        for path in tqdm(
            list(self.data_collection_path.glob("**/*.md"))
            + list(self.data_collection_path.glob("**/*.py")),
            ascii=True,
            desc="Chunking",
            unit="file",
        ):
            self._result["files"].append(
                {"file": path.name, "last_index": time(), "chunks": []}
            )

            # Character Based Chunking

            n = self.max_chunk_size
            o = 20 if n >= 100 else 0
            txt_len = path.stat().st_size
            for k in range(txt_len // n + 1):
                fst = k * (n - o)
                lst = fst + n - 1 if fst + n < txt_len else txt_len - 1

                self._result["files"][-1]["chunks"].append(
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

        self._result["avg_doc_len"] /= self._result["documents_number"]

    def _tokenize(self):
        curr_file = ("", "")

        for chunk in tqdm(
            (
                chunk
                for file in self._result["files"]
                for chunk in file["chunks"]
            ),
            ascii=True,
            desc="Tokenizing",
            unit="chunk",
            total=self._result["documents_number"]
        ):
            if curr_file[0] != chunk["file_path"]:
                if curr_file[1] is None:
                    continue

                try:
                    with open(chunk["file_path"]) as f:
                        curr_file = (chunk["file_path"], f.read())
                except IOError as e:
                    print(
                        f"[WARNING] {type(e)}: {e}: Couldn't read file {chunk['file_path']}",
                    )
                    curr_file = (chunk["file_path"], None)
                    continue

            chunk_txt = curr_file[1][
                chunk["first_character_index"] : chunk["last_character_index"]
                + 1
            ]

            for token_id in self.model.encode(chunk_txt):
                chunk["terms"][str(token_id)] = chunk["terms"].get(str(token_id), 0) + 1
                if chunk["terms"][str(token_id)] == 1:
                    self._result["term_appearances"][str(token_id)] = (
                        self._result["term_appearances"].get(str(token_id), 0) + 1
                    )

