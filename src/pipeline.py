"""Core RAG operations shared by the CLI and the HTTP API.

This module holds the actual logic (index, search, answer, evaluate). It is
deliberately independent from ``python-fire`` and ``FastAPI``: the CLI
(``src/__main__.py``) and the API (``src/api.py``) are thin front ends on
top of it, so neither depends on the other.
"""

from src.models import (
    MinimalAnswer,
    MinimalSource,
    QueryAnswer,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
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
from tqdm import tqdm
import json
import sys


def build_index(
    max_chunk_size: int = 2000,
    raw_data: str = "vllm-0.10.1",
    embed: bool = False,
) -> None:
    """Ingest ``data/raw/<raw_data>`` and build (or update) the index.

    The cached retriever is dropped so that the next search sees the new
    index.

    Args:
        max_chunk_size: Maximum chunk size, in characters.
        raw_data: Name of the corpus folder in ``data/raw/``.
        embed: Whether to also compute semantic embeddings.

    Raises:
        IOError: If the corpus cannot be read or the index written.
    """
    Indexer(raw_data, max_chunk_size, embed).index()


def search_query(
    query: str, k: int, embed: bool = False
) -> list[MinimalSource]:
    """Return the top-k sources of a single query.

    Args:
        query: The question.
        k: Number of sources to return.
        embed: Whether to weight BM25 with semantic similarity.

    Returns:
        At most ``k`` sources, best first.
    """
    return Retriever().score(query, k, embed)


def search_questions(
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
        sys.exit(1)

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
        sys.exit(1)

    print(f"Saved student_search_results to {output_path}")


def answer_query(query: str, k: int, embed: bool = False) -> QueryAnswer:
    """Retrieve the top-k sources of a query and answer it.

    Args:
        query: The question.
        k: Number of sources given to the model.
        embed: Whether to weight BM25 with semantic similarity.

    Returns:
        The question, its retrieved sources and the generated answer.
    """
    sources = search_query(query, k, embed)

    return QueryAnswer(
        question=query,
        retrieved_sources=sources,
        answer=LLModel().answer(query, sources),
    )


def answer_questions(
    student_search_results_path: str, save_directory: str
) -> None:
    """Generate answers from saved search results.

    Args:
        student_search_results_path: ``StudentSearchResults`` JSON file
            produced by ``search_questions``.
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
        sys.exit(1)

    llm = LLModel()

    output = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=llm.answer(result.question, result.retrieved_sources),
            )
            for result in tqdm(dataset.search_results)
        ],
        k=dataset.k,
    )

    try:
        with open(output_path, "w") as f:
            json.dump(dump_results_model(output), f, indent=4)
    except IOError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Saved student_search_results_and_answer to {output_path}")


def evaluate_results(
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
