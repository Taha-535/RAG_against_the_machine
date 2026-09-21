from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from src.models import MinimalSource


class LLModel:
    def __init__(self) -> None:
        self.model_name = "Qwen/Qwen3-0.6B"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

    def encode(self, txt: str) -> list[int]:
        return self._tokenizer.encode(txt)

    def decode(self, txt_ids: list[int]) -> str:
        return self._tokenizer.decode(txt_ids, skip_special_tokens=True).strip('\n')

    def ask(self, query: str, sources: list[MinimalSource]) -> str:
        chunks_txt = []

        for source in sources:
            fst = source.first_character_index
            lst = source.last_character_index

            with open(source.file_path) as f:
                chunks_txt.append(f.read()[fst : lst + 1])

        print('\n'.join(chunks_txt))

        prompt = (
            "Your task is to answer the asked question from the given documents."
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
            enable_thinking=True
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=32768
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        index = len(output_ids) - output_ids[::-1].index(151668)
    
        return self.decode(output_ids[index:])
