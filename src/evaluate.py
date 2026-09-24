from src.models import StudentSearchResults, RagDataset


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
            print(
                "Total number of questions with sources:",
                sum(
                    1
                    for question in self._dataset_questions.values()
                    if question.sources
                ),
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

    def evaluate(self):
        self._print_general()
