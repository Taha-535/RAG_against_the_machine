from tqdm import tqdm
from ._helper import retrieve_chunk_content
from src.helper import term_splitter


def tokenize_chunk(self):
    curr_file = ("", "")

    def perform_tokenization(chunk):
        chunk_txt = retrieve_chunk_content(curr_file, chunk)

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


def tokenize(self):
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
