"""Local HTTP API (FastAPI) on top of the RAG pipeline.

Run with ``uv run uvicorn src.api:app``. It only depends on
``src.pipeline``, never on the Fire CLI in ``src/__main__.py``, so it works
independently of which commands the CLI exposes.
"""

try:
    from src.helper import format_pydantic_error
    from fastapi import FastAPI, HTTPException
    from src.models import Index, QueryAnswer
    from pydantic import ValidationError
    from src import pipeline
    import pickle
except ImportError as e:
    print(
        f"ImportError: {e}. Run `uv sync` or `make install` first!",
        file=sys.stderr,
    )
    sys.exit(1)


app = FastAPI()


@app.get("/")
def root() -> dict[str, str | list[str]]:
    return {
        "name": "RAG against the machine",
        "docs": "/docs",
        "endpoints": ["/index", "/answer"],
    }


@app.get("/index")
def index(max_chunk_size: int = 2000, embed: bool = False) -> Index:
    """Build (or update) the index and return it."""

    try:
        pipeline.index(max_chunk_size, embed=embed)

        with open("data/processed/index.pkl", "rb") as f:
            loaded: Index = Index.model_validate(pickle.load(f))

        return loaded
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=format_pydantic_error(e, pipeline.index)
        )
    except (IOError, pickle.PickleError) as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    except (Exception, SystemExit):
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

    try:
        return pipeline.answer(query, k, embed)
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=format_pydantic_error(e, pipeline.answer)
        )
    except (Exception, SystemExit):
        raise HTTPException(
            status_code=503, detail="Index or model unavailable"
        )
