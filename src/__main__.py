from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from pydantic import BaseModel, ValidationError
from src.search import Retriever
from src.index import Indexer
from src.llm import LLModel
from pathlib import Path
import fire
import json
import sys


def _get_data_and_output_path(
    model: type, data_path: str, save_directory: str
):
    try:
        with open(data_path) as f:
            dataset = model.model_validate_json(f.read())
    except (IOError, ValidationError) as e:
        print(
            "Error loading dataset!",
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        raise

    try:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        raise

    return dataset, Path(str(dir_path) + "/" + Path(data_path).name)


def _dump_results_model(model: BaseModel) -> dict:
    return model.model_dump(
        exclude={
            "search_results": {
                "__all__": {
                    "retrieved_sources": {
                        "__all__": {"document_length", "terms"}
                    }
                }
            }
        }
    )


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
            dataset, output_path = _get_data_and_output_path(
                RagDataset, dataset_path, save_directory
            )
        except (IOError, ValidationError, FileExistsError):
            exit(1)

        try:
            with open(output_path, "w") as f:
                json.dump(
                    _dump_results_model(
                        Retriever(self.model).search_dataset(dataset, k)
                    ),
                    f,
                    indent=4,
                )
        except IOError as e:
            print(e, file=sys.stderr)

    def answer(self, query: str, k: int):
        print(self.model.answer(query, Retriever(self.model).score(query, k)))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        try:
            dataset, output_path = _get_data_and_output_path(
                StudentSearchResults, student_search_results_path, save_directory
            )
        except (IOError, ValidationError, FileExistsError):
            exit(1)

        try:
            with open(output_path, "w") as f:
                json.dump(
                    _dump_results_model(
                        StudentSearchResultsAndAnswer(
                            search_results=[
                                MinimalAnswer(
                                    question_id=search_result.question_id,
                                    question=search_result.question,
                                    retrieved_sources=search_result.retrieved_sources,
                                    answer=self.model.answer(
                                        search_result.question,
                                        search_result.retrieved_sources,
                                    ),
                                )
                                for search_result in dataset.search_results[:1]
                            ],
                            k=dataset.k,
                        )
                    ),
                    f,
                    indent=4,
                )
        except IOError as e:
            print(e, file=sys.stderr)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ): ...


if __name__ == "__main__":
    fire.Fire(App)
