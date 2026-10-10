"""Command-line interface of the RAG pipeline (built with Python Fire).

Only ``index``, ``search``, ``search_dataset``, ``answer``,
``answer_dataset`` and ``evaluate`` are reachable from the command line.

``fire.Fire`` is never given a container (module, dict, class or instance):
whatever the container, Fire falls back on ``dir()`` when a name is not one
of its keys, which would let ``__new__``, ``__class__``, ``keys``, imported
modules, etc. be reached. Instead, the command name is looked up here in a
private table and Fire is handed that single function. Any other name is
rejected before Fire runs. Every command returns ``None`` so Fire has no
returned object to navigate into either.

Usage: ``uv run python -m src <command> [options]``
"""

from typing import Callable, Any
import sys

try:
    import fire
    from src.models import MinimalSource
    from src import pipeline
except ImportError as e:
    print(
        f"ImportError: {e}. Run `uv sync` or `make install` first!",
        file=sys.stderr,
    )
    sys.exit(1)


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
    if (
        not isinstance(max_chunk_size, int)
        or isinstance(max_chunk_size, bool)
        or not 0 < max_chunk_size <= 2000
    ):
        print(
            (
                "Invalid 'max_chunk_size' argument! "
                "Expected an integer value between 1 and 2000, "
                f"got '{max_chunk_size}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(raw_data, str):
        print(
            (
                "Invalid 'raw_data' argument! "
                f"Expected a string, got '{raw_data}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(embed, bool):
        print(
            f"Invalid 'embed' argument! Expected a boolean, got '{embed}'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        pipeline.build_index(max_chunk_size, raw_data, embed)
    except Exception as e:
        print(
            f"Error Indexing collection: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def search(query: str, k: int, embed: bool = False) -> list[MinimalSource]:
    """Print the top-k sources of a single query.

    Args:
        query: The question.
        k: Number of sources to return.
        embed: Whether to weight BM25 with semantic similarity.
    """
    if not isinstance(query, str) or not query.strip():
        print(
            (
                "Invalid 'query' argument! "
                f"Expected a non-empty string, got '{query}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(k, int) or k < 0:
        print(
            (
                "Invalid 'k' argument! "
                f"Expected a non-negative integer, got '{k}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(embed, bool):
        print(
            f"Invalid 'embed' argument! Expected a boolean, got '{embed}'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        results = pipeline.search_query(str(query), k, embed)

        print(
            "\n".join(
                f"{src.file_path} "
                f"[{src.first_character_index}:{src.last_character_index}]"
                for src in results
            )
        )

        return results
    except Exception as e:
        print(
            f"Error executing search: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


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
    if not isinstance(dataset_path, str):
        print(
            (
                "Invalid 'dataset_path' argument! "
                f"Expected a string, got '{dataset_path}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(k, int) or k < 0:
        print(
            (
                "Invalid 'k' argument! "
                f"Expected a non-negative integer, got '{k}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(save_directory, str):
        print(
            (
                "Invalid 'save_directory' argument! "
                f"Expected a string, got '{save_directory}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(embed, bool):
        print(
            f"Invalid 'embed' argument! Expected a boolean, got '{embed}'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        pipeline.search_questions(dataset_path, k, save_directory, embed)
    except Exception as e:
        print(
            f"Error searching dataset: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def answer(query: str, k: int, embed: bool = False) -> None:
    """Retrieve sources for a query and print the generated answer.

    Args:
        query: The question.
        k: Number of sources given to the model.
        embed: Whether to weight BM25 with semantic similarity.
    """
    if not isinstance(query, str) or not query.strip():
        print(
            (
                "Invalid 'query' argument! "
                f"Expected a non-empty string, got '{query}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(k, int) or k < 0:
        print(
            (
                "Invalid 'k' argument! "
                f"Expected a non-negative integer, got '{k}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(embed, bool):
        print(
            f"Invalid 'embed' argument! Expected a boolean, got '{embed}'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        res = pipeline.answer_query(str(query), k, embed)
        print(res.answer)
    except Exception as e:
        print(
            f"Error generating answer: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


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
    if not isinstance(student_search_results_path, str):
        print(
            (
                "Invalid 'student_search_results_path' argument! "
                f"Expected a string, got '{student_search_results_path}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(save_directory, str):
        print(
            (
                "Invalid 'save_directory' argument! "
                f"Expected a string, got '{save_directory}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        pipeline.answer_questions(student_search_results_path, save_directory)
    except Exception as e:
        print(
            f"Error generating answers for dataset: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def evaluate(
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
    if not isinstance(student_search_results_path, str):
        print(
            (
                "Invalid 'student_search_results_path' argument! "
                f"Expected a string, got '{student_search_results_path}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(dataset_path, str):
        print(
            (
                "Invalid 'dataset_path' argument! "
                f"Expected a string, got '{dataset_path}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if (
        not isinstance(max_context_length, int)
        or isinstance(max_context_length, bool)
        or max_context_length <= 0
    ):
        print(
            (
                "Invalid 'max_context_length' argument! "
                "Expected a positive integer, "
                f"got '{max_context_length}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        pipeline.evaluate_results(
            student_search_results_path, dataset_path, max_context_length
        )
    except Exception as e:
        print(
            f"Error evaluating results: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def _usage() -> str:
    """Build the list of available commands.

    Returns:
        A multi-line usage message.
    """
    lines = ["Usage: python -m src <command> [options]", "", "Commands:"]

    for name, func in _COMMANDS.items():
        summary = (func.__doc__ or "").strip().splitlines()[0]
        lines.append(f"  {name:<16}{summary}")

    lines += ["", "Run `python -m src <command> --help` for its options."]

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Run the command named by the first argument through Fire.

    Args:
        argv: Command-line arguments, without the program name.

    Returns:
        The process exit code.
    """
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 2

    command = _COMMANDS.get(argv[0])

    if command is None:
        print(f"Unknown command: {argv[0]}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    try:
        fire.Fire(command, command=argv[1:], name=argv[0])
    except KeyboardInterrupt:
        print("Program has been terminated!", file=sys.stderr)
        return 1
    except Exception as e:
        print(
            f"Unexpected error has occurred: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


_COMMANDS: dict[str, Callable[..., Any]] = {
    "index": index,
    "search": search,
    "search_dataset": search_dataset,
    "answer": answer,
    "answer_dataset": answer_dataset,
    "evaluate": evaluate,
}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
