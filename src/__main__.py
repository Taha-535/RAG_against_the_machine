"""Command-line interface (Python Fire) and HTTP API of the RAG system."""

from src.models import (
    Index,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    QueryAnswer,
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
from fastapi import FastAPI
from tqdm import tqdm
import pickle
import fire
import json
import sys


class App:
    """Commands exposed by the CLI: ``uv run python -m src <command>``."""

    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        raw_data: str = "vllm-0.10.1",
        embed: bool = False,
    ) -> None:
        """Ingest ``data/raw/<raw_data>`` and build the index.

        Args:
            max_chunk_size: Maximum chunk size, in characters.
            raw_data: Name of the corpus folder in ``data/raw/``.
            embed: Whether to also compute semantic embeddings.
        """
        Indexer(raw_data, max_chunk_size, embed).index()

    @staticmethod
    def search(query: str, k: int, embed: bool = False) -> None:
        """Print the top-k sources of a single query.

        Args:
            query: The question.
            k: Number of sources to return.
            embed: Whether to weight BM25 with semantic similarity.
        """
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
        """Search a whole dataset and save a ``StudentSearchResults`` file.

        Args:
            dataset_path: JSON dataset of questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory receiving the JSON output (same file
                name as the dataset).
            embed: Whether to weight BM25 with semantic similarity.
        """
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
        """Retrieve sources for a query and print the generated answer.

        Args:
            query: The question.
            k: Number of sources given to the model.
            embed: Whether to weight BM25 with semantic similarity.
        """
        print(LLModel().answer(query, Retriever().score(query, k, embed)))

    @staticmethod
    def answer_dataset(
        student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers from saved search results.

        Args:
            student_search_results_path: ``StudentSearchResults`` JSON file
                produced by ``search_dataset``.
            save_directory: Directory receiving the
                ``StudentSearchResultsAndAnswer`` JSON output.
        """
        try:
            dataset, output_path = get_data_and_output_path(
                StudentSearchResults,
                student_search_results_path,
                save_directory,
            )
        except (IOError, ValidationError, FileExistsError):
            exit(1)

        search_results = dataset.search_results

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
                                for search_result in tqdm(search_results)
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
        """Print recall@k of search results against a ground truth.

        Args:
            student_search_results_path: ``StudentSearchResults`` JSON file.
            dataset_path: Ground-truth ``AnsweredQuestions`` JSON dataset.
            max_context_length: Maximum allowed source length, in
                characters.
        """
        results = get_data(StudentSearchResults, student_search_results_path)
        dataset = get_data(RagDataset, dataset_path)

        Evaluator(results, dataset, max_context_length).evaluate()


app = FastAPI()


@app.get("/index")
def index() -> Index:
    """Build (or update) the index and return it."""
    App.index()

    with open("data/processed/index.pkl", "rb") as f:
        return Index.model_validate(pickle.load(f))


@app.get("/answer")
def answer(query: str, k: int) -> QueryAnswer:
    """Retrieve the top-k sources of a query and answer it.

    Args:
        query: The question.
        k: Number of sources given to the model.

    Returns:
        The question, its retrieved sources and the generated answer.
    """
    sources = Retriever().score(query, k, False)
    answer = LLModel().answer(query, sources)

    return QueryAnswer(
        question=query, retrieved_sources=sources, answer=answer
    )


if __name__ == "__main__":
    fire.Fire(App)
