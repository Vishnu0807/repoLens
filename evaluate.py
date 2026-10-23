"""Compare RepoLens retrieval modes on a small manually checked question set."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from core import SearchIndex, embedding_text, index_sources, load_directory


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def reciprocal_rank(results, expected):
    return next((1 / rank for rank, item in enumerate(results, 1) if item.symbol.qualified_name == expected), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()

    questions = json.loads((ROOT / "evaluation_queries.json").read_text(encoding="utf-8"))
    positives = [item for item in questions if item["expected"]]
    negatives = [item for item in questions if not item["expected"]]
    repository = index_sources(load_directory(ROOT / "sample_repo"))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    documents = [embedding_text(symbol) for symbol in repository.symbols]
    vectors = np.asarray(model.encode(documents, normalize_embeddings=True), dtype=np.float32)
    index = SearchIndex(repository.symbols, vectors)

    report = {
        "dataset": "bundled-order-service-v1",
        "positive_queries": len(positives),
        "no_answer_queries": len(negatives),
        "symbol_count": len(repository.symbols),
        "model": MODEL_NAME,
        "limitations": (
            "The questions were manually written for this small repository. This checks the demo, "
            "not retrieval quality on unrelated repositories. Latency is machine-dependent."
        ),
        "modes": {},
        "unsuccessful_searches": [],
    }

    for mode in ("keyword", "semantic", "hybrid"):
        rows = []
        latencies = []
        for item in positives:
            start = time.perf_counter()
            query_vector = (
                np.asarray(model.encode([item["query"]], normalize_embeddings=True)[0], dtype=np.float32)
                if mode != "keyword"
                else None
            )
            results = index.search(item["query"], mode, 5, query_vector)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            rr = reciprocal_rank(results, item["expected"])
            rows.append({"query": item["query"], "expected": item["expected"], "rank": round(1 / rr) if rr else None})
        recall = sum(row["rank"] is not None for row in rows) / len(rows)
        mrr = sum((1 / row["rank"]) if row["rank"] else 0 for row in rows) / len(rows)
        report["modes"][mode] = {
            "recall_at_5": round(recall, 4),
            "mrr_at_5": round(mrr, 4),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "queries": rows,
        }

    for item in negatives:
        rows = {"query": item["query"], "cause": item["cause"], "top_results": {}}
        for mode in ("keyword", "semantic", "hybrid"):
            query_vector = (
                np.asarray(model.encode([item["query"]], normalize_embeddings=True)[0], dtype=np.float32)
                if mode != "keyword"
                else None
            )
            results = index.search(item["query"], mode, 3, query_vector)
            rows["top_results"][mode] = [result.symbol.qualified_name for result in results]
        report["unsuccessful_searches"].append(rows)

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for mode, metrics in report["modes"].items():
        print(f"{mode:8} Recall@5={metrics['recall_at_5']:.3f} MRR@5={metrics['mrr_at_5']:.3f} median={metrics['median_latency_ms']:.2f} ms")
    print(f"Saved {args.output}; review {len(negatives)} no-answer searches and their causes.")


if __name__ == "__main__":
    main()
