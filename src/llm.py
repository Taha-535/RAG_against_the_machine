from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

class LLModel:
    def __init__(self) -> None:
        self.model_name = 'Qwen/Qwen3-0.6B'
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

        self.vocab_file = hf_hub_download(
            repo_id=self.model_name,
            filename='vocab.json'
        )

    def encode(self, txt: str) -> list[int]:
        return self._tokenizer.encode(txt)

