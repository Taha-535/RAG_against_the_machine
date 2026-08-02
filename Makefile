RUN = uv run

install:
	uv sync

run:
	$(RUN) python -m src

debug:
	$(RUN) python -m pdb -m src

clean:
	rm -rf $$(find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \))

lint:
	 $(RUN) -m flake8 src
	 $(RUN) -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	 $(RUN) -m flake8 src
	 $(RUN) -m mypy src --strict

.PHONY: install run clean lint lint-strict
