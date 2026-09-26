from tqdm import tqdm
from ._helper import retrieve_chunk_content


def update_embedding(self) -> callable:
    curr_file = ("", "")

    def main_function(chunk) -> list[float]:
        chunk_txt = retrieve_chunk_content(curr_file, chunk)

        chunk["embedding"] = self._embedder.embed(chunk_txt).tolist()

    return main_function


def update_index(self) -> None:
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
