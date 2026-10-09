"""Local HTTP API (FastAPI) on top of the RAG pipeline.

Run with ``uv run uvicorn src.api:app``. It only depends on
``src.pipeline``, never on the Fire CLI in ``src/__main__.py``, so it works
independently of which commands the CLI exposes.
"""

from fastapi import FastAPI, HTTPException
from src.models import Index, QueryAnswer
from src import pipeline
import pickle


app = FastAPI()


@app.get("/")
def root():
    return {
        "name": "RAG against the machine",
        "docs": "/docs",
        "endpoints": ["/index", "/answer"],
    }


@app.get("/index")
def index(max_chunk_size: int = 2000, embed: bool = False) -> Index:
    """Build (or update) the index and return it."""

    if (
        not isinstance(max_chunk_size, int)
        or isinstance(max_chunk_size, bool)
        or not 0 < max_chunk_size <= 2000
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid 'max_chunk_size' argument! "
                "Expected an integer value between 1 and 2000, "
                f"got '{max_chunk_size}'"
            ),
        )

    if not isinstance(embed, bool):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid 'embed' argument! Expected a boolean, got '{embed}'"
            )
        )

    try:
        pipeline.build_index(max_chunk_size, embed=embed)

        with open("data/processed/index.pkl", "rb") as f:
            loaded: Index = Index.model_validate(pickle.load(f))

        return loaded
    except (IOError, pickle.PickleError) as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    except SystemExit:
        raise HTTPException(status_code=500, detail="Indexing failed")


@app.get("/answer")
def answer(query: str, k: int, embed: bool = False) -> QueryAnswer:
    """Retrieve the top-k sources of a query and answer it.

    Args:
        query: The question.
        k: Number of sources given to the model.

    Returns:
        The question, its retrieved sources and the generated answer.
    """

    if not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid 'query' argument! "
                f"Expected a non-empty string, got '{query}'"
            )
        )

    if not isinstance(k, int) or k < 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid 'k' argument! "
                f"Expected a non-negative integer, got '{k}'"
            ),
        )

    if not isinstance(embed, bool):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid 'embed' argument! Expected a boolean, got '{embed}'"
            )
        )

    try:
        return pipeline.answer_query(query, k)
    except SystemExit:
        raise HTTPException(
            status_code=503, detail="Index or model unavailable"
        )
