import fire
import json
from src.index import Indexer
from src.llm import LLModel


class App:
    def __init__(self) -> None:
        self.model = LLModel()

    def index(self, max_chunk_size: int = 2000):
        Indexer(
            self.model,
            'data/raw/vllm-0.10.1',
            max_chunk_size
        ).index()

    @staticmethod
    def search(query: str, k: int = 50): ...

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str): ...

    @staticmethod
    def answer(query: str, k: int): ...

    @staticmethod
    def answer_dataset(
        student_search_results_path: str, save_directory: str
    ): ...

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str): ...


if __name__ == "__main__":
    fire.Fire(App)
