from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from src.helper import (
    get_data,
    get_data_and_output_path,
    dump_results_model,
)
from pydantic import ValidationError
from src.evaluate import Evaluator
from src.search import Retriever
from src.index import Indexer
from src.llm import LLModel
import fire
import json
import sys


class App:
    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        raw_data: str = "vllm-0.10.1",
        embed: bool = False,
    ) -> None:
        Indexer(raw_data, max_chunk_size, embed).index()

    @staticmethod
    def search(query: str, k: int, embed: bool = False) -> None:
        print(
            "\n".join(
                (
                    f"{src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
                for src in Retriever().score(query, k, embed)
            )
        )

    @staticmethod
    def search_dataset(
        dataset_path: str, k: int, save_directory: str, embed: bool = False
    ) -> None:
        try:
            dataset, output_path = get_data_and_output_path(
                RagDataset, dataset_path, save_directory
            )
        except (IOError, ValidationError, FileExistsError):
            exit(1)

        try:
            with open(output_path, "w") as f:
                json.dump(
                    dump_results_model(
                        Retriever().search_dataset(dataset, k, embed)
                    ),
                    f,
                    indent=4,
                )
        except IOError as e:
            print(e, file=sys.stderr)

    @staticmethod
    def answer(query: str, k: int, embed: bool = False) -> None:
        print(LLModel().answer(query, Retriever().score(query, k, embed)))

    @staticmethod
    def answer_dataset(
        student_search_results_path: str, save_directory: str
    ) -> None:
        try:
            dataset, output_path = get_data_and_output_path(
                StudentSearchResults,
                student_search_results_path,
                save_directory,
            )
        except (IOError, ValidationError, FileExistsError):
            exit(1)

        try:
            with open(output_path, "w") as f:
                json.dump(
                    dump_results_model(
                        StudentSearchResultsAndAnswer(
                            search_results=[
                                MinimalAnswer(
                                    question_id=search_result.question_id,
                                    question=search_result.question,
                                    retrieved_sources=(
                                        search_result.retrieved_sources
                                    ),
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
    ) -> None:
        results = get_data(StudentSearchResults, student_search_results_path)
        dataset = get_data(RagDataset, dataset_path)

        Evaluator(results, dataset, max_context_length).evaluate()


if __name__ == "__main__":
    fire.Fire(App)
