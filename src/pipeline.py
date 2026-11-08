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
from pydantic import ValidationError, validate_call, Field, ConfigDict
from src.evaluate import Evaluator
from src.search import Retriever
from src.index import Indexer
from typing import Annotated
from src.llm import LLModel
from tqdm import tqdm
import json
import sys


strict_conf = ConfigDict(strict=True)


@validate_call(config=strict_conf)
def index(
    max_chunk_size: Annotated[int, Field(gt=0, le=2000)] = 2000,
    raw_data: Annotated[str, Field(min_length=1)] = "data/raw/vllm-0.10.1",
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


@validate_call(config=strict_conf)
def search(
    query: Annotated[str, Field(min_length=1)],
    k: Annotated[int, Field(gt=0)],
    embed: bool = False,
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


@validate_call(config=strict_conf)
def search_dataset(
    dataset_path: Annotated[str, Field(min_length=1)],
    k: Annotated[int, Field(gt=0)],
    save_directory: Annotated[str, Field(min_length=1)],
    embed: bool = False,
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


@validate_call(config=strict_conf)
def answer(
    query: Annotated[str, Field(min_length=1)],
    k: Annotated[int, Field(gt=0)],
    embed: bool = False,
) -> QueryAnswer:
    """Retrieve the top-k sources of a query and answer it.

    Args:
        query: The question.
        k: Number of sources given to the model.
        embed: Whether to weight BM25 with semantic similarity.

    Returns:
        The question, its retrieved sources and the generated answer.
    """
    sources = search(query, k, embed)

    return QueryAnswer(
        question=query,
        retrieved_sources=sources,
        answer=LLModel().answer(query, sources),
    )


@validate_call(config=strict_conf)
def answer_dataset(
    student_search_results_path: Annotated[str, Field(min_length=1)],
    save_directory: Annotated[str, Field(min_length=1)],
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


@validate_call(config=strict_conf)
def evaluate(
    student_search_results_path: Annotated[str, Field(min_length=1)],
    dataset_path: Annotated[str, Field(min_length=1)],
    max_context_length: Annotated[int, Field(gt=0)] = 2000,
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
