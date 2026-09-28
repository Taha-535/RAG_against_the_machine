"""Small helpers shared by the CLI, the retriever and the indexer."""

from pydantic import ValidationError, BaseModel
from pathlib import Path
from typing import Any, TypeVar
import sys
import re

ModelT = TypeVar("ModelT", bound=BaseModel)


def get_data(model: type[ModelT], data_path: str) -> ModelT:
    """Load and validate a JSON file against a pydantic model.

    On failure an error is printed on stderr and the program exits with
    status 1.

    Args:
        model: The pydantic model class the file must conform to.
        data_path: Path of the JSON file.

    Returns:
        The validated model instance.
    """
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


def get_output_path(data_path: str, save_directory: str) -> Path:
    """Build the output file path and create the output directory.

    The output file keeps the name of the input file.

    Args:
        data_path: Path of the input file.
        save_directory: Directory in which the output must be saved.

    Returns:
        The path of the output file.

    Raises:
        FileExistsError: If ``save_directory`` exists and is not a
            directory.
    """
    try:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        raise

    return Path(str(dir_path) + "/" + Path(data_path).name)


def get_data_and_output_path(
    model: type[ModelT], data_path: str, save_directory: str
) -> tuple[ModelT, Path]:
    """Load an input file and compute where its output must be written.

    Args:
        model: The pydantic model class the input file must conform to.
        data_path: Path of the input JSON file.
        save_directory: Directory in which the output must be saved.

    Returns:
        A ``(validated_data, output_path)`` tuple.
    """
    return get_data(model, data_path), get_output_path(
        data_path, save_directory
    )


def dump_results_model(model: BaseModel) -> dict[str, Any]:
    """Serialise a results model, dropping the index-only source fields.

    ``document_length``, ``terms`` and ``embedding`` are removed from every
    retrieved source so the output only holds the fields of the subject.

    Args:
        model: A ``StudentSearchResults`` or
            ``StudentSearchResultsAndAnswer`` instance.

    Returns:
        A JSON-serialisable dictionary.
    """
    return model.model_dump(
        exclude={
            "search_results": {
                "__all__": {
                    "retrieved_sources": {
                        "__all__": {"document_length", "terms", "embedding"}
                    }
                }
            }
        }
    )


def term_splitter(text: str) -> list[str]:
    """Split a text into search terms.

    A word optionally joined to a second word by ``-``, ``_`` or ``'``
    yields three terms: the composed form and each of its two parts (e.g.
    ``max_chunk`` gives ``max_chunk``, ``max`` and ``chunk``). Any other
    word yields itself. Terms are case sensitive.

    Args:
        text: The text to split.

    Returns:
        The list of terms, with repetitions.
    """
    terms = []

    for match in re.finditer(r"([a-zA-Z1-9]+)(?:[-_']([a-zA-Z1-9]+))?", text):
        composed, fst, sec = match.group(0, 1, 2)

        if sec is not None:
            terms.extend([composed, fst, sec])
        else:
            terms.append(composed)

    return terms
