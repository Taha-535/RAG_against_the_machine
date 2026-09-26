from pydantic import ValidationError, BaseModel
from pathlib import Path
import sys
import re


def get_data(model: type, data_path: str):
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


def get_output_path(data_path: str, save_directory: str):
    try:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        raise

    return Path(str(dir_path) + "/" + Path(data_path).name)


def get_data_and_output_path(model: type, data_path: str, save_directory: str):

    return get_data(model, data_path), get_output_path(
        data_path, save_directory
    )


def dump_results_model(model: BaseModel) -> dict:
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
    terms = []

    for match in re.finditer(r"([a-zA-Z1-9]+)(?:[-_']([a-zA-Z1-9]+))?", text):
        composed, fst, sec = match.group(0, 1, 2)

        if sec is not None:
            terms.extend([composed, fst, sec])
        else:
            terms.append(composed)

    return terms
