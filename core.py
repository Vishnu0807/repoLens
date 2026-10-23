"""Repository parsing, call resolution, caching, and retrieval for RepoLens."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RawCall:
    expression: str
    name: str
    line: int
    receiver: str = ""


@dataclass(frozen=True)
class CallLink:
    expression: str
    line: int
    status: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Symbol:
    name: str
    qualified_name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    docstring: str
    source: str
    raw_calls: tuple[RawCall, ...] = ()
    calls: tuple[CallLink, ...] = ()


@dataclass(frozen=True)
class Repository:
    symbols: tuple[Symbol, ...]
    sources: dict[str, str]
    warnings: tuple[str, ...]
    fingerprint: str
    cache_hits: int
    parsed_files: int


@dataclass(frozen=True)
class SearchResult:
    symbol: Symbol
    score: float
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    matched_terms: tuple[str, ...] = ()


class _CallVisitor(ast.NodeVisitor):
    """Collect calls while skipping nested definitions."""

    def __init__(self):
        self.calls: list[RawCall] = []

    def visit_Call(self, node: ast.Call):
        expression = ast.unparse(node.func)
        if isinstance(node.func, ast.Name):
            self.calls.append(RawCall(expression, node.func.id, node.lineno))
        elif isinstance(node.func, ast.Attribute):
            receiver = ast.unparse(node.func.value)
            self.calls.append(RawCall(expression, node.func.attr, node.lineno, receiver))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return None

    def visit_ClassDef(self, node: ast.ClassDef):
        return None


def _calls_in(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[RawCall, ...]:
    visitor = _CallVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return tuple(visitor.calls)


def _make_symbol(node, path: str, lines: list[str], parent: str = "") -> Symbol:
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    end = node.end_lineno or node.lineno
    kind = "class" if isinstance(node, ast.ClassDef) else "method" if parent else "function"
    raw_calls = () if isinstance(node, ast.ClassDef) else _calls_in(node)
    return Symbol(
        name=node.name,
        qualified_name=f"{parent}.{node.name}" if parent else node.name,
        kind=kind,
        path=path,
        start_line=start,
        end_line=end,
        docstring=ast.get_docstring(node) or "",
        source="".join(lines[start - 1 : end]),
        raw_calls=raw_calls,
    )


def parse_python(text: str, path: str) -> tuple[Symbol, ...]:
    """Extract top-level functions/classes and direct methods without executing code."""
    tree = ast.parse(text, filename=path)
    lines = text.splitlines(keepends=True)
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(_make_symbol(node, path, lines))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(_make_symbol(child, path, lines, node.name))
    return tuple(symbols)


class ParseCache:
    """Reuse parsed files when their SHA-256 digest is unchanged."""

    def __init__(self):
        self._entries: dict[str, tuple[str, tuple[Symbol, ...]]] = {}

    def parse(self, path: str, source: str) -> tuple[tuple[Symbol, ...], bool, str]:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        cached = self._entries.get(path)
        if cached and cached[0] == digest:
            return cached[1], True, digest
        symbols = parse_python(source, path)
        self._entries[path] = (digest, symbols)
        return symbols, False, digest

    def keep_only(self, paths: Iterable[str]):
        allowed = set(paths)
        self._entries = {path: value for path, value in self._entries.items() if path in allowed}


def _reference(symbol: Symbol) -> str:
    return f"{symbol.qualified_name} · {symbol.path}:{symbol.start_line}"


def resolve_calls(symbols: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Resolve direct names confidently; label uncertain receiver calls as ambiguous."""
    by_name: dict[str, list[Symbol]] = {}
    by_qualified: dict[str, list[Symbol]] = {}
    for symbol in symbols:
        by_name.setdefault(symbol.name, []).append(symbol)
        by_qualified.setdefault(symbol.qualified_name, []).append(symbol)

    resolved_symbols = []
    for symbol in symbols:
        links = []
        parent = symbol.qualified_name.rsplit(".", 1)[0] if symbol.kind == "method" else ""
        for call in symbol.raw_calls:
            candidates: list[Symbol] = []
            status = "external"
            if not call.receiver:
                candidates = by_name.get(call.name, [])
                status = "resolved" if len(candidates) == 1 else "ambiguous" if candidates else "external"
            elif call.receiver in {"self", "cls"} and parent:
                candidates = by_qualified.get(f"{parent}.{call.name}", [])
                status = "resolved" if len(candidates) == 1 else "ambiguous" if candidates else "external"
            else:
                candidates = by_name.get(call.name, [])
                status = "ambiguous" if candidates else "external"
            links.append(CallLink(call.expression, call.line, status, tuple(_reference(item) for item in candidates)))
        resolved_symbols.append(replace(symbol, calls=tuple(links)))
    return tuple(resolved_symbols)


def index_sources(sources: dict[str, str], cache: ParseCache | None = None) -> Repository:
    """Parse source strings, reuse unchanged files, and resolve local call relationships."""
    cache = cache or ParseCache()
    cache.keep_only(sources)
    symbols: list[Symbol] = []
    warnings: list[str] = []
    digests = []
    hits = 0
    parsed = 0
    for path in sorted(sources):
        try:
            file_symbols, reused, digest = cache.parse(path, sources[path])
            symbols.extend(file_symbols)
            digests.append(f"{path}:{digest}")
            hits += int(reused)
            parsed += int(not reused)
        except (SyntaxError, UnicodeError, RecursionError) as error:
            warnings.append(f"Skipped {path}: {type(error).__name__}")
    fingerprint = hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
    return Repository(
        symbols=resolve_calls(tuple(symbols)),
        sources=dict(sources),
        warnings=tuple(warnings),
        fingerprint=fingerprint,
        cache_hits=hits,
        parsed_files=parsed,
    )


def load_directory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if not any(part.startswith(".") or part in {"__pycache__", "venv", "node_modules"} for part in path.parts)
    }


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


def embedding_text(symbol: Symbol) -> str:
    return f"{symbol.qualified_name} in {symbol.path}. {symbol.docstring}\n{symbol.source[:1200]}"


class SearchIndex:
    def __init__(self, symbols: Iterable[Symbol], vectors: np.ndarray | None = None):
        self.symbols = tuple(symbols)
        self.vectors = vectors
        documents = [tokenize(f"{s.qualified_name} {s.path} {s.docstring} {s.source}") for s in self.symbols]
        self.counts = [Counter(document) for document in documents]
        self.lengths = [len(document) for document in documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency = Counter(term for counts in self.counts for term in counts)

    def _keyword(self, query: str) -> list[tuple[int, float]]:
        terms = set(tokenize(query))
        ranked = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1 + (len(self.symbols) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = self.lengths[i] / self.average_length
                score += inverse_frequency * frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * length_ratio)
                )
            if score > 0:
                ranked.append((i, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def _semantic(self, query_vector: np.ndarray) -> list[tuple[int, float]]:
        if self.vectors is None:
            raise ValueError("Semantic vectors have not been built")
        scores = self.vectors @ query_vector
        return sorted(enumerate(scores.tolist()), key=lambda item: (-item[1], item[0]))

    def search(
        self,
        query: str,
        mode: str = "keyword",
        limit: int = 5,
        query_vector: np.ndarray | None = None,
    ) -> list[SearchResult]:
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("Mode must be keyword, semantic, or hybrid")
        if not query.strip() or not self.symbols:
            return []
        keyword = self._keyword(query)
        semantic = self._semantic(query_vector) if mode != "keyword" and query_vector is not None else []
        keyword_ranks = {i: rank for rank, (i, _) in enumerate(keyword, 1)}
        semantic_ranks = {i: rank for rank, (i, _) in enumerate(semantic, 1)}
        if mode == "keyword":
            ranked = keyword
        elif mode == "semantic":
            if query_vector is None:
                raise ValueError("Semantic search needs a query vector")
            ranked = semantic
        else:
            if query_vector is None:
                raise ValueError("Hybrid search needs a query vector")
            fused = {
                i: (1 / (60 + keyword_ranks[i]) if i in keyword_ranks else 0)
                + (1 / (60 + semantic_ranks[i]) if i in semantic_ranks else 0)
                for i in keyword_ranks.keys() | semantic_ranks.keys()
            }
            ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        terms = set(tokenize(query))
        return [
            SearchResult(
                symbol=self.symbols[i],
                score=float(score),
                keyword_rank=keyword_ranks.get(i),
                semantic_rank=semantic_ranks.get(i),
                matched_terms=tuple(sorted(terms & self.counts[i].keys())),
            )
            for i, score in ranked[:limit]
        ]
