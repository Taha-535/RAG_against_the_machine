"""Language model wrapper."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.utils import logging
from src.models import MinimalSource
from typing import Any
import sys


class LLModel:
    """Answer generator based on ``Qwen/Qwen3-0.6B``."""

    def __init__(self) -> None:
        """Load the tokenizer and the model."""
        logging.set_verbosity_error()

        self.model_name = "Qwen/Qwen3-0.6B"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self._model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_name,
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

            try:
                with open(source.file_path) as f:
                    chunks_txt.append(f.read()[fst:lst + 1])
            except (IOError, Exception) as e:
                print(
                    (
                        f"Error getting chunk from '{source.file_path}': "
                        f"{type(e).__name__}: {e}"
                    ),
                    file=sys.stderr
                )
                exit(1)

        prompt = (
            "Answer the question using only the documents below. "
            "If they do not contain the answer, say so.\n"
            'Documents:\n"' + '"\n"'.join(chunks_txt) + '"\n'
            "Question: " + query
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

        try:
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=256
            )
        except Exception as e:
            print(
                (
                    f"Error generating prompt: "
                    f"{type(e).__name__}: {e}"
                ),
                file=sys.stderr
            )
            exit(1)

        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        return self.decode(output_ids)
