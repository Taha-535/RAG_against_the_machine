from pydantic import ValidationError
from src.models import RagDataset
from src.search import Retriever
from src.index import Indexer
from src.llm import LLModel
from pathlib import Path
import fire
import json
import sys


class App:
    def __init__(self) -> None:
        self.model = LLModel()

    def index(self, max_chunk_size: int = 2000):
        Indexer(self.model, "data/raw/vllm-0.10.1", max_chunk_size).index()

    def search(self, query: str, k: int):
        print(
            "\n".join(
                (
                    f"{src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
                for src in Retriever(self.model).score(query, k)
            )
        )

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        try:
            with open(dataset_path) as f:
                dataset = RagDataset.model_validate_json(f.read())
        except (IOError, ValidationError) as e:
            print(
                "Error loading dataset!",
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

        try:
            dir_path = Path(save_directory)
            dir_path.mkdir(parents=True, exist_ok=True)
        except FileExistsError as e:
            print(e, file=sys.stderr)

        output_path = Path(str(dir_path) + "/" + Path(dataset_path).name)
        try:
            with open(output_path, "w") as f:
                json.dump(
                    Retriever(self.model)
                    .search_dataset(dataset, k)
                    .model_dump(
                        exclude={
                            "search_results": {
                                "__all__": {
                                    "retrieved_sources": {
                                        "__all__": {"document_length", "terms"}
                                    }
                                }
                            }
                        }
                    ),
                    f,
                    indent=4,
                )
        except IOError as e:
            print(e, file=sys.stderr)

    def answer(self, query: str, k: int): ...

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ): ...

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ): ...


if __name__ == "__main__":
    fire.Fire(App)
