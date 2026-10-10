from sentence_transformers import SentenceTransformer
from typing import Any, Optional
from numpy.typing import NDArray
import numpy as np
import sys


class Embedder:
    """Sentence embedder based on ``all-MiniLM-L6-v2`` (CPU friendly)."""

    def __init__(self) -> None:
        """Load the sentence-transformers model."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        try:
            self._model = SentenceTransformer(
                self.model_name
            )
        except Exception as e:
            print(
                (
                    f"Couldn't load model '{self.model_name}': "
                    f"{type(e).__name__}: {e}"
                ),
                file=sys.stderr
            )
            exit(1)

    def embed(self, text: str) -> NDArray[np.float64]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        try:
            return np.asarray(self._model.encode(text), dtype=np.float64)
        except Exception as e:
            print(
                (
                    f"Couldn't embed text: "
                    f"{type(e).__name__}: {e}"
                ),
                file=sys.stderr
            )
            exit(1)

    def similarity(
        self, embed1: NDArray[Any], embed2: Optional[list[float]]
    ) -> float:
        """Compute the similarity between two embeddings.

        Args:
            embed1: First embedding (typically the query).
            embed2: Second embedding (typically a chunk). When it is
                ``None`` (chunk indexed without embedding), the similarity
                is neutral and ``1.0`` is returned.

        Returns:
            The similarity score.
        """
        if embed2 is None:
            return 1.0

        return float(self._model.similarity(embed1, np.asarray(embed2)))
