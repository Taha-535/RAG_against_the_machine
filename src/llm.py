"""Language model and sentence embedding wrappers."""

from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from huggingface_hub.utils import disable_progress_bars, logging as hf_logging
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from src.models import MinimalSource
from typing import Any, Optional
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()
hf_logging.set_verbosity_error()
disable_progress_bars()


class Embedder:
    """Sentence embedder based on ``all-MiniLM-L6-v2`` (CPU friendly)."""

    def __init__(self) -> None:
        """Load the sentence-transformers model."""
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, text: str) -> NDArray[Any]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        return np.asarray(self._model.encode(text))

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


class LLModel:
    """Answer generator based on ``Qwen/Qwen3-0.6B``."""

    def __init__(self) -> None:
        """Load the tokenizer and the model."""
        self.model_name = "Qwen/Qwen3-0.6B"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self._model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

    def encode(self, txt: str) -> list[int]:
        """Convert a text into token ids.

        Args:
            txt: The text to tokenize.

        Returns:
            The token ids.
        """
        ids: list[int] = list(self._tokenizer.encode(txt))
        return ids

    def decode(self, txt_ids: list[int]) -> str:
        """Convert token ids back into a text.

        Special tokens are dropped and surrounding newlines are stripped.

        Args:
            txt_ids: The token ids.

        Returns:
            The decoded text.
        """
        text = str(self._tokenizer.decode(txt_ids, skip_special_tokens=True))
        return text.strip("\n")

    def answer(self, query: str, sources: list[MinimalSource]) -> str:
        """Generate an answer to a query from retrieved sources.

        The text of every source is read from disk, concatenated into the
        prompt, and the model is asked to answer the question from it.

        Args:
            query: The question to answer.
            sources: The retrieved sources used as context.

        Returns:
            The generated answer.
        """
        chunks_txt: list[str] = []

        for source in sources:
            fst = source.first_character_index
            lst = source.last_character_index

            with open(source.file_path) as f:
                chunks_txt.append(f.read()[fst:lst + 1])

        prompt = (
            "Your task is to answer the asked question from these documents."
            + '\nDocuments:\n"'
            + '"\n"'.join(chunks_txt)
            + '"\n'
            + "Question: "
            + query
        )

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(
            self._model.device
        )
        generated_ids = self._model.generate(
            **model_inputs
        )

        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        return self.decode(output_ids)
