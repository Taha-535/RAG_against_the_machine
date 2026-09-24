from src.models import StudentSearchResults, RagDataset, MinimalSource
import sys


class Evaluator:
    def __init__(
        self,
        search_results: StudentSearchResults,
        dataset: RagDataset,
        max_length: int,
    ) -> None:
        self._dataset_questions = {
            model.question_id: model for model in dataset.rag_questions
        }
        self._result_questions = {
            model.question_id: model for model in search_results.search_results
        }

        self._k = search_results.k

        self._max_len = max_length

    def _check_valid(self) -> bool:
        if self._k > 10:
            print("Student data has more than 10 sources")
            return False

        if self._k <= 0:
            print("k is not strictly positive")
            return False

        invalid_sources = []

        for source in (
            source
            for sources in self._result_questions.values()
            for source in sources.retrieved_sources
        ):
            src_len = (
                source.last_character_index - source.first_character_index + 1
            )

            if src_len > self._max_len:
                invalid_sources.append((src_len, source))

        if invalid_sources:
            print(
                "\n".join(
                    (
                        f"Source {source} has a length of {src_len} "
                        "which is more than the limit of "
                        f"{self._max_len} characters"
                    )
                    for src_len, source in invalid_sources
                )
            )
            return False

        return True

    def _print_general(self) -> None:

        is_valid = self._check_valid()

        print("Student Data is valid:", is_valid)

        if is_valid:
            print("Total number of questions:", len(self._dataset_questions))
            self._total = sum(
                1
                for question in self._dataset_questions.values()
                if question.sources
            )
            print(
                "Total number of questions with sources:",
                self._total,
            )
            print(
                "Total number of questions with student sources:",
                sum(
                    1
                    for question in self._result_questions.values()
                    if question.retrieved_sources
                ),
            )
            print()
        else:
            print("Student search results are not valid")
            print("False\n")
            exit(1)

    @staticmethod
    def _IoU(chunk1: MinimalSource, chunk2: MinimalSource) -> float:
        inter = (
            min(chunk1.last_character_index, chunk2.last_character_index)
            - max(chunk1.first_character_index, chunk2.first_character_index)
            + 1
        )

        if inter <= 0:
            return 0

        union = (
            max(chunk1.last_character_index, chunk2.last_character_index)
            - min(chunk1.first_character_index, chunk2.first_character_index)
            + 1
        )

        return inter / union

    def _evaluate_questions(self):
        self._recall = {
            "recall@1": 0,
            "recall@3": 0,
            "recall@5": 0,
            "recall@10": 0,
        }
        k_s = (10, 5, 3, 1)

        for qui, question in self._result_questions.items():
            try:
                answered_question = self._dataset_questions[qui]
            except KeyError:
                print(
                    f"Question with question_id {qui} doesn't exist in the dataset",
                    file=sys.stderr,
                )
                exit(1)

            expected_num = len(answered_question.sources)
            if expected_num == 0:
                continue

            for expected in answered_question.sources:
                valid = [
                    src
                    for src in enumerate(question.retrieved_sources, 1)
                    if (
                        src[1].file_path == expected.file_path
                        and self._IoU(src[1], expected) >= 0.05
                    )
                ]

                if valid:
                    fst = valid[0][0]

                    for k in k_s:
                        if fst > k:
                            break
                        self._recall[f"recall@{k}"] += 1 / expected_num

        for k in self._recall.keys():
            self._recall[k] /= self._total

    def _print_result(self):
        print("Questions evaluated:", self._total)

        for k, v in self._recall.items():
            print(f"{k}: {v:.3f} ({v * 100:.1f}%)")

        print(self._recall)

    def evaluate(self):
        self._print_general()

        print("Evaluation Results")
        print(" ".join("==" for _ in range(10)))

        self._evaluate_questions()

        self._print_result()
