# RepoLens

RepoLens helps developers navigate unfamiliar Python repositories with free-text search. Ask **“Where is a failed payment retried?”** or **“Which function calculates delivery charges?”** and inspect the matching code with its original path and line range.

The app supports a bundled order-service project and bounded repository ZIP uploads. Uploaded code is parsed in memory and is never imported or executed.

## Features

- Extract functions, classes, methods, docstrings, source, and line ranges with Python AST.
- Rank symbols with BM25 keyword search or pretrained Sentence Transformer embeddings.
- Combine keyword and semantic rankings with Reciprocal Rank Fusion.
- Compare the top five keyword, semantic, and hybrid results side by side.
- Browse complete files or filter by function, class, and method.
- Resolve unique direct calls and label uncertain receiver relationships as ambiguous.
- Cache parsed files by SHA-256 content hash so unchanged files are reused.
- Validate uploaded ZIP size, paths, symlinks, file count, encodings, and expansion size.
- Expose the bundled sample through a validated FastAPI search endpoint.
- Evaluate retrieval with Recall@5, MRR@5, latency, and documented no-answer searches.

## Stack

Python, AST, Sentence Transformers, BM25, NumPy, Streamlit, FastAPI, Pydantic, pytest, and Git.

## Project structure

```text
repolens/
├── .streamlit/config.toml     # Dark theme and upload limit
├── app.py                     # Streamlit interface
├── core.py                    # Parsing, cache, calls, BM25, semantic and RRF
├── ingest.py                  # Safe in-memory ZIP validation
├── api.py                     # FastAPI extension
├── evaluate.py                # Recall@5, MRR@5, latency and failure report
├── evaluation_queries.json    # 15 answerable + 5 no-answer questions
├── evaluation_report.json     # Reproducible results from the bundled dataset
├── sample_repo/               # Bundled order-service demo
└── tests/                     # Core and API tests
```

## Run on Windows

Install Python 3.12 and Git. Open PowerShell inside this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501. The first semantic or hybrid search downloads `all-MiniLM-L6-v2`; keyword search works immediately.

Run the API separately:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Interactive API documentation is at http://127.0.0.1:8000/docs.

## Test and evaluate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluate.py --output evaluation_report.json
```

The evaluation runs all three retrieval methods with the real embedding model. It measures answerable questions and records what each method returns for questions whose functionality does not exist. The bundled question set is small and written for the sample repository; its results are a demo check, not a general benchmark.

Measured on the bundled 15-question set:

| Method | Recall@5 | MRR@5 | Median query latency* |
|---|---:|---:|---:|
| Keyword | 1.000 | 0.922 | 0.04 ms |
| Semantic | 1.000 | 1.000 | 16.97 ms |
| Hybrid | 1.000 | 0.967 | 37.18 ms |

\*Latency is machine-dependent and excludes one-time model/document loading. Semantic happened to rank best on this deliberately small sample; hybrid is not claimed to always win.

## How search works

1. AST finds top-level functions, classes, and direct methods without executing repository code.
2. Each symbol keeps its qualified name, file path, line range, docstring, and source.
3. BM25 scores exact words from identifiers, paths, docstrings, and code.
4. MiniLM creates normalized vectors for a short summary of each symbol. Dot product gives cosine similarity.
5. Hybrid search adds `1 / (60 + rank)` from the BM25 and semantic lists.

Raw BM25 and cosine scores use different scales, so Reciprocal Rank Fusion combines ranks instead of adding the scores directly.

Call resolution is deliberately conservative. A unique local `helper()` or `self.total()` target is marked resolved. A call such as `order.total()` is ambiguous because AST alone does not prove the runtime type of `order`. Unmatched calls are treated as external.

## Limits

- ZIP: 10 MB upload, 30 MB expanded, 3,000 entries, 200 Python files, and 200 KB per file.
- Parses top-level functions, classes, and direct methods. Nested functions are not indexed separately.
- The cache lasts for the Streamlit session; it is not a shared production cache.
- Semantic quality depends on a pretrained general sentence model; the model was not trained in this project.
- Retrieval suggests relevant code. It does not prove program behaviour or correctness.
- The FastAPI extension searches the bundled project. Uploaded repositories remain inside the Streamlit session.

