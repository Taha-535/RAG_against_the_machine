from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from pydantic import BaseModel, ValidationError
from src.evaluate import Evaluator
from src.search import Retriever
from src.index import Indexer
from src.llm import LLModel
from pathlib import Path
import fire
import json
import sys
import re


def _get_data(model: type, data_path: str):
    try:
        with open(data_path) as f:
            dataset = model.model_validate_json(f.read())
    except IOError as e:
        print(
            "Error loading dataset!",
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        exit(1)
    except ValidationError as e:
        print((f"Invalid data in {data_path}\n{e}"), file=sys.stderr)
        exit(1)

    return dataset


def _get_output_path(data_path: str, save_directory: str):
    try:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        raise

    return Path(str(dir_path) + "/" + Path(data_path).name)


def _get_data_and_output_path(
    model: type, data_path: str, save_directory: str
):

    return _get_data(model, data_path), _get_output_path(
        data_path, save_directory
    )


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


def _term_splitter(text: str) -> list[str]:
    terms = []

    for match in re.finditer(r"([a-zA-Z1-9]+)(?:[-_']([a-zA-Z1-9]+))?", text):
        composed, fst, sec = match.group(0, 1, 2)

        if sec is not None:
            terms.extend([composed, fst, sec])
        else:
            terms.append(composed)

    return terms


class App:
    @staticmethod
    def index(max_chunk_size: int = 2000):
        Indexer(_term_splitter, "data/raw/vllm-0.10.1", max_chunk_size).index()

    @staticmethod
    def search(query: str, k: int):
        print(
            "\n".join(
                (
                    f"{src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
                for src in Retriever(_term_splitter).score(query, k)
            )
        )

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str):
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
                        Retriever(_term_splitter).search_dataset(dataset, k)
                    ),
                    f,
                    indent=4,
                )
        except IOError as e:
            print(e, file=sys.stderr)

    @staticmethod
    def answer(query: str, k: int):
        print(
            LLModel().answer(query, Retriever(_term_splitter).score(query, k))
        )

    @staticmethod
    def answer_dataset(student_search_results_path: str, save_directory: str):
        try:
            dataset, output_path = _get_data_and_output_path(
                StudentSearchResults,
                student_search_results_path,
                save_directory,
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
                                    answer=LLModel().answer(
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
            exit(1)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        max_context_length: int = 2000,
    ):
        results = _get_data(StudentSearchResults, student_search_results_path)
        dataset = _get_data(RagDataset, dataset_path)

        Evaluator(results, dataset, max_context_length).evaluate()


if __name__ == "__main__":
    fire.Fire(App)
