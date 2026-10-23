"""Small FastAPI extension for RepoLens search."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core import SearchIndex, embedding_text, index_sources, load_directory


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REPOSITORY = index_sources(load_directory(ROOT / "sample_repo"))


@lru_cache(maxsize=1)
def semantic_components():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    documents = [embedding_text(symbol) for symbol in REPOSITORY.symbols]
    vectors = np.asarray(model.encode(documents, normalize_embeddings=True), dtype=np.float32)
    return SearchIndex(REPOSITORY.symbols, vectors), model


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    limit: int = Field(default=5, ge=1, le=10)


app = FastAPI(title="RepoLens API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "symbols": len(REPOSITORY.symbols)}


@app.post("/search")
def search(request: SearchRequest):
    if request.mode == "keyword":
        index = SearchIndex(REPOSITORY.symbols)
        query_vector = None
    else:
        index, model = semantic_components()
        query_vector = np.asarray(model.encode([request.query], normalize_embeddings=True)[0], dtype=np.float32)
    results = index.search(request.query, request.mode, request.limit, query_vector)
    return {
        "query": request.query,
        "mode": request.mode,
        "results": [
            {
                "name": result.symbol.qualified_name,
                "kind": result.symbol.kind,
                "path": result.symbol.path,
                "start_line": result.symbol.start_line,
                "end_line": result.symbol.end_line,
                "score": round(result.score, 6),
            }
            for result in results
        ],
    }
