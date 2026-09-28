*This project has been created as part of the 42 curriculum by tel-moat.*
 
# RAG against the machine
 
## Description
 
**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system that answers questions about a codebase, here the [vLLM](https://github.com/vllm-project/vllm) `0.10.1` repository.
 
A language model only knows what it was trained on. Instead of retraining it, the system lets a small local model (`Qwen/Qwen3-0.6B`) *look things up* at answer time:
 
1. **Index** the Python and Markdown files of the repository into searchable chunks.
2. **Retrieve** the top-k chunks that best match a question (BM25).
3. **Augment** the prompt with the text of those chunks.
4. **Generate** a grounded answer and save it as JSON.
The system is judged on whether it retrieves the right *source locations* (file path + character range) and on the answers grounded in them. Retrieval quality is measured with **recall@k**: a ground-truth source counts as found when a retrieved chunk is in the same file and overlaps its character range (IoU ≥ 0.05). Targets: **≥ 80 % recall@5 on docs questions** and **≥ 50 % on code questions**.
 
## Instructions
 
### Requirements
 
- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) (the only tool needed to install the dependencies)
- Several GB of free disk space (PyTorch and the model weights)

### Installation
 
```bash
make install        # runs `uv sync`
```
 
Expected layout at the repository root:
 
```
src/                                   # the Python module
pyproject.toml, uv.lock, Makefile, README.md
data/raw/vllm-0.10.1/                  # the corpus to index
data/processed/                        # the index (generated)
data/datasets/UnansweredQuestions/     # questions without answers
data/datasets/AnsweredQuestions/       # ground truth
data/output/search_results/<scope>/            # generated
data/output/search_results_and_answer/<scope>/ # generated
```
 
### Makefile rules
 
| Rule | Purpose |
| --- | --- |
| `make install` | Install the dependencies with `uv` |
| `make run` | Run the main script |
| `make debug` | Run the main script under `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, ... |
| `make lint` | `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` (optional) |
 
### Commands
 
Every command is run as `uv run python -m src <command> [options]`.
 
| Command | Description |
| --- | --- |
| `index --max_chunk_size 2000` | Ingest `data/raw/` and build the index in `data/processed/` |
| `search "<query>" --k 5` | Print the top-k sources of one query |
| `search_dataset --dataset_path P --k 10 --save_directory D` | Search a whole dataset and write a `StudentSearchResults` JSON file |
| `answer "<query>" --k 5` | Retrieve, then generate an answer for one query |
| `answer_dataset --student_search_results_path P --save_directory D` | Generate answers and write a `StudentSearchResultsAndAnswer` JSON file |
| `evaluate --student_search_results_path P --dataset_path Q` | Print recall@1/3/5/10 (local check; the moulinette is the official scorer) |
 
Optional flags: `--embed` (on `index`, `search`, `search_dataset`, `answer`) enables the semantic embeddings described below, and `--raw_data` selects another folder of `data/raw/` (default `vllm-0.10.1`).
 
## Example usage
 
```bash
# 1. Build the index (chunks of at most 2000 characters)
uv run python -m src index --max_chunk_size 2000
 
# 2. Single query
uv run python -m src search "How to configure the OpenAI server?" --k 5
# (illustrative output)
# data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1936:3170]
# ...
 
# 3. Search a whole dataset (scope the output directory by dataset)
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
 
# 4. Score with the moulinette (official) or with our own command
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
 
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
 
# 5. Generate the answers
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
 
Semantic embeddings (bonus), which must be enabled at both indexing and search time:
 
```bash
uv run python -m src index --max_chunk_size 2000 --embed
uv run python -m src search "How to enable LoRA?" --k 5 --embed
```
 
## System architecture
 
```
 data/raw/vllm-0.10.1
        │  *.py, *.md
        ▼
 ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐
 │  Chunking    │──▶│ Tokenizing   │──▶│ data/processed/index.pkl│
 │ (py │ md)    │   │ (terms, tf)  │   │ (chunks + statistics)   │
 └──────────────┘   └──────────────┘   └───────────┬────────────┘
                                                   ▼
 question ──────────────────────────────▶  Retriever (BM25)  ──────▶  top-k sources
                                                                           │
                                              read chunk text from disk    │
                                                                           ▼
                                    prompt ──▶ Qwen3-0.6B ────────▶  answer (JSON)
```
 
| Module | Role |
| --- | --- |
| `src/__main__.py` | Python Fire CLI (`App`) and the optional FastAPI app |
| `src/index/` | `Indexer`: chunking (`_chunk.py`), tokenization (`_tokenize.py`), incremental update (`_update.py`), persistence (`index.py`) |
| `src/search.py` | `Retriever` and the `_BM25` scorer |
| `src/llm.py` | `LLModel` (Qwen3-0.6B answer generation) and `Embedder` (all-MiniLM-L6-v2) |
| `src/models.py` | Pydantic models exchanged between stages (`MinimalSource`, `StudentSearchResults`, ...) |
| `src/evaluate.py` | Local recall@k computation |
| `src/helper.py` | JSON loading/validation, output paths, term splitting |
 
All the data exchanged between stages is validated with pydantic: datasets, search results and the index itself are re-validated when loaded.
 
## Chunking strategy
 
Chunk size is set by `--max_chunk_size` (default **2000 characters**, the maximum accepted by the moulinette). Chunks are described by an inclusive character range `[first_character_index, last_character_index]` in their file, so the file is never duplicated in the index. Two strategies are used, depending on the file type:
 
- **Markdown / text (`*.md`)**: fixed-size consecutive windows of `max_chunk_size` characters.
- **Python (`*.py`)**: syntax-aware chunking based on the `ast` module.
  1. The module is parsed and its top-level statements are taken in order.
  2. Consecutive statements are **merged greedily** as long as the merged span fits in `max_chunk_size`, so small functions and imports stay together.
  3. A statement that is **too large and has a body** (class, function, `if`, `with`, ...) is replaced by its children, which go through the same process, so chunks tend to follow function and method boundaries.
  4. A **too-large statement with no body** (for example a huge literal) is cut into equal slices.

## Retrieval method
 
- **Tokenization**: each chunk is split into terms with a regular expression (`term_splitter`). Identifiers joined by `_`, `-` or `'` produce three terms: the compound and both parts (`max_chunk` → `max_chunk`, `max`, `chunk`). This lets a question quoting an identifier verbatim *and* a question paraphrasing its words both find the chunk. Terms are case sensitive.
- **Index**: for each chunk, the term frequencies and length (in characters) are stored, plus collection statistics (number of chunks, average length, number of chunks containing each term). It is pickled to `data/processed/index.pkl`.
- **Ranking**: **BM25** (Okapi) with `k1 = 1.5` and `b = 0.75`:
  - `idf(t) = ln((N - n_t + 0.5) / (n_t + 0.5))`
  - `score(q, d) = Σ idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))`
  - Every chunk is scored, pushed on a heap, and the `k` best are returned.
- **Semantic option (`--embed`)**: chunks are also embedded with `all-MiniLM-L6-v2` (CPU friendly) and stored in the index. At search time the BM25 score of each chunk is **multiplied by the cosine similarity** between the query and the chunk embeddings.
- **Answer generation**: the text of the retrieved chunks is read from disk and placed in a prompt asking `Qwen/Qwen3-0.6B` (thinking mode disabled) to answer the question from these documents.

## Performance analysis
 
| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target (recall@5) |
| --- | --- | --- | --- | --- | --- |
| Docs | 54.0% | 70.0% | 80.0% | 83.0% | ≥ 80 % |
| Code | 34.3% | 45.5% | 50.5% | 63.6% | ≥ 50 % |
 
| Measure | Result | Limit |
| --- | --- | --- |
| Indexing time (whole corpus) | < 10 s | ≤ 5 min |
| Retrieval time (200 questions) | < 1 min | ≤ 90 s |
 
## Design decisions
 
- **BM25 rather than TF-IDF**: it handles term-frequency saturation and chunk-length normalisation, which matters because chunks have different lengths (a merged group of small functions vs. a full 2000-character window).
- **Two chunkers**: source code has structure that a fixed window would cut in the middle of a function; Markdown has none that is worth parsing.
- **Identifier-aware tokenization**: compound terms are kept *and* split, to serve both verbatim-identifier and paraphrased questions.
- **Persisted, validated index**: saved as a pickle and validated with pydantic on load; a missing or invalid index makes the CLI exit with a clear message instead of a traceback.
- **Index freshness**: if the index exists it is *updated* (only new or modified files, detected by modification time, are re-chunked and re-tokenized; deleted files are removed), and it is rebuilt from scratch when the indexing code is newer than the index.
- **Optional semantic scoring**: the embedding model is only loaded when `--embed` is used, so the mandatory path stays lightweight.
- **Optional HTTP API**: `src/__main__.py` also exposes a FastAPI app (`GET /answer?query=...&k=...` and `GET /index`), for example served with `uvicorn src.__main__:app`.

## Challenges faced
 
- **Character offsets from the AST**: `ast` gives line/column positions while the output needs character offsets. The file's newline positions are precomputed so that node positions can be turned into absolute character ranges.
- **Nodes larger than a chunk**: giant classes or literals cannot be one chunk. Nodes with a body are descended into and body-less ones are sliced, so that chunks stay within `max_chunk_size` (a single longer source makes the moulinette reject the whole output; check with `evaluate` or the moulinette).
- **Vocabulary mismatch between questions and code**: questions paraphrase ideas or quote identifiers. Emitting both the compound and split terms addresses both cases.
- **Keeping the index up to date**: the index records a timestamp per file and compares it to the file's modification time, and it detects a change of the indexing code itself, to avoid searching a stale index.
- **Robustness of the CLI**: invalid JSON, missing files, or a missing index are reported with an error message rather than a traceback.

## Resources
 
- **General RAG architecture**:
> ["RAG Architecture: Components, Timing & Design Patterns - Language AI Handbook"](https://mbrenndoerfer.com/writing/rag-architecture-retriever-generator-design-patterns)
> ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks - Research Paper"](https://arxiv.org/pdf/2005.11401)

- **Chunking**:
> ["Document Chunking: Optimizing RAG Retrieval Pipelines - Language AI Handbook"](https://mbrenndoerfer.com/writing/document-chunking-rag-strategies-retrieval#structural-chunking)
> ["CAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree - Research Paper"](https://arxiv.org/pdf/2506.15655)

- **Retrieval**:
> ["BM25: the Search Algorithm Behind Elasticsearch - Language AI Handbook"](https://mbrenndoerfer.com/writing/bm25-search-algorithm-elasticsearch-implementation#problem-3-how-do-we-handle-document-length)
> ["The Probabilistic Relevance Framework: BM25 and Beyond - Research Paper"](www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
> ["TF-IDF for Text Representation - Language AI Handbook"](https://mbrenndoerfer.com/writing/tf-idf-term-frequency-inverse-document-frequency-text-representation)
> ["Probabilistic relevance model - Article"](https://en.wikipedia.org/wiki/Probabilistic_relevance_model)

- **Documentations**:
> ["Qwen/Qwen3-0.6B Model - HuggingFace"](https://huggingface.co/Qwen/Qwen3-0.6B)
> ["pathlib - Python"](https://docs.python.org/3/library/pathlib.html)
> ["Pydantic serialization - Pydantic"](https://pydantic.dev/docs/validation/latest/concepts/serialization/)
> ["pickle - Python"](https://docs.python.org/3/library/pickle.html)
> ["re (regext) - Python"](https://docs.python.org/3/library/re.html)
> ["ast (Abstract Syntax Tree) - Python"](https://docs.python.org/3/library/ast.html)
> ["Python fire module"](https://python-fire.readthedocs.io/en/latest/)
> ["FastAPI starting guide"](https://realpython.com/get-started-with-fastapi/)

### How AI was used
 
AI assistance was used for the following tasks:
 
- Finding infomrative resources
- Fixing `mypy` errors
- Adding Docstrings
- writing README file

AI was never used for writing the code and solving the project's problems!
