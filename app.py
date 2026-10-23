"""Streamlit interface for RepoLens."""

from pathlib import Path
import time

import numpy as np
import streamlit as st

from core import ParseCache, SearchIndex, embedding_text, index_sources, load_directory
from ingest import read_repository_zip


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading the embedding model for the first time…")
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_data(show_spinner="Encoding repository symbols…")
def encode_documents(fingerprint: str, documents: tuple[str, ...]) -> np.ndarray:
    del fingerprint  # The fingerprint is the cache key for repository content.
    return np.asarray(load_model().encode(list(documents), normalize_embeddings=True), dtype=np.float32)


def semantic_index(repository):
    documents = tuple(embedding_text(symbol) for symbol in repository.symbols)
    vectors = encode_documents(repository.fingerprint, documents)
    return SearchIndex(repository.symbols, vectors), load_model()


def call_rows(symbol):
    visible = [call for call in symbol.calls if call.status != "external"]
    return [
        {
            "Call": call.expression,
            "Line": call.line,
            "Status": call.status,
            "Target": ", ".join(call.candidates) if call.candidates else "—",
        }
        for call in visible
    ]


st.set_page_config(page_title="RepoLens", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.2rem;}
    [data-testid="stMetric"] {background:#121923; border:1px solid #273142; padding:14px; border-radius:8px;}
    [data-testid="stCode"] {border:1px solid #273142; border-radius:8px;}
    .rl-kicker {color:#6ee7b7; font:600 12px ui-monospace; letter-spacing:.12em; text-transform:uppercase;}
    .rl-title {font-size:42px; line-height:1.1; font-weight:650; margin:.45rem 0 .5rem; letter-spacing:-.04em;}
    .rl-subtitle {color:#94a3b8; max-width:720px; margin-bottom:1.4rem;}
    .rl-source {font:12px ui-monospace; margin:-.75rem 0 1.4rem;}
    .rl-source a {color:#6ee7b7; text-decoration:none;}
    .rl-source a:hover {text-decoration:underline;}
    .rl-path {font:12px ui-monospace; color:#6ee7b7; margin-bottom:.5rem;}
    </style>
    <div class="rl-kicker">Python repository search</div>
    <div class="rl-title">RepoLens</div>
    <div class="rl-subtitle">Search unfamiliar Python code by identifier or behaviour, then inspect the original source and call relationships.</div>
    <div class="rl-source"><a href="https://github.com/vishnu0807/repolens" target="_blank">github.com/vishnu0807/repolens ↗</a></div>
    """,
    unsafe_allow_html=True,
)

if "parse_cache" not in st.session_state:
    st.session_state.parse_cache = ParseCache()

with st.sidebar:
    st.header("Repository")
    source_choice = st.radio("Source", ["Bundled sample", "Upload ZIP"], label_visibility="collapsed")
    uploaded = None
    upload_warnings = []
    if source_choice == "Upload ZIP":
        uploaded = st.file_uploader("Python repository ZIP", type=["zip"], help="10 MB, 200 Python files maximum")
        st.caption("Files are parsed in memory. Uploaded code is never imported or executed.")

try:
    if source_choice == "Upload ZIP":
        if uploaded is None:
            st.info("Upload a repository ZIP to begin, or select the bundled sample.")
            st.stop()
        sources, upload_warnings = read_repository_zip(uploaded.getvalue())
        repository_name = Path(uploaded.name).stem
    else:
        sources = load_directory(ROOT / "sample_repo")
        repository_name = "order-service"
    repository = index_sources(sources, st.session_state.parse_cache)
except ValueError as error:
    st.error(str(error))
    st.stop()

warnings = [*upload_warnings, *repository.warnings]
for warning in warnings[:5]:
    st.warning(warning)

files = sorted(repository.sources)
classes = sum(symbol.kind == "class" for symbol in repository.symbols)
functions = sum(symbol.kind in {"function", "method"} for symbol in repository.symbols)
resolved_calls = sum(call.status == "resolved" for symbol in repository.symbols for call in symbol.calls)

st.markdown(f"<div class='rl-path'>{repository_name} / {len(files)} Python files</div>", unsafe_allow_html=True)
metric_columns = st.columns(4)
metric_columns[0].metric("Symbols", len(repository.symbols))
metric_columns[1].metric("Functions & methods", functions)
metric_columns[2].metric("Classes", classes)
metric_columns[3].metric("Resolved calls", resolved_calls)

search_tab, browse_tab, method_tab = st.tabs(["Search", "Browse source", "How it works"])

with search_tab:
    query = st.text_input("Ask about the repository", value="Where is a failed payment retried?", max_chars=300)
    option_columns = st.columns([2, 2, 5])
    mode = option_columns[0].radio("Retrieval", ["Keyword", "Semantic", "Hybrid"], horizontal=True)
    compare = option_columns[1].checkbox("Compare all modes")
    run_search = st.button("Search repository", type="primary", use_container_width=False)

    if run_search or query:
        if not query.strip():
            st.info("Enter a question or identifier.")
        else:
            try:
                started = time.perf_counter()
                query_vector = None
                if mode != "Keyword" or compare:
                    index, model = semantic_index(repository)
                    query_vector = np.asarray(model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
                else:
                    index = SearchIndex(repository.symbols)
                results = index.search(query, mode.lower(), 5, query_vector)
                elapsed_ms = (time.perf_counter() - started) * 1000
                st.caption(f"{len(results)} results · {mode.lower()} retrieval · {elapsed_ms:.1f} ms")

                if compare:
                    comparison = {}
                    for comparison_mode in ("keyword", "semantic", "hybrid"):
                        compared = index.search(query, comparison_mode, 5, query_vector if comparison_mode != "keyword" else None)
                        names = [item.symbol.qualified_name for item in compared]
                        comparison[comparison_mode.title()] = names + ["—"] * (5 - len(names))
                    st.dataframe(comparison, use_container_width=True, hide_index=True)

                if not results:
                    st.info("No lexical match was found. Try Semantic search or use different terms.")
                else:
                    selected_name = st.selectbox(
                        "Results",
                        range(len(results)),
                        format_func=lambda i: f"{i + 1}. {results[i].symbol.qualified_name}  ·  {results[i].symbol.path}:{results[i].symbol.start_line}",
                    )
                    selected = results[selected_name]
                    symbol = selected.symbol
                    left, right = st.columns([3, 2])
                    with left:
                        st.subheader(symbol.qualified_name)
                        st.caption(f"{symbol.path}:{symbol.start_line}–{symbol.end_line} · {symbol.kind}")
                        if symbol.docstring:
                            st.write(symbol.docstring.splitlines()[0])
                        st.code(symbol.source, language="python", line_numbers=True)
                    with right:
                        st.subheader("Why it ranked")
                        st.write(f"Score: `{selected.score:.4f}`")
                        st.write("Matched terms: " + (", ".join(selected.matched_terms) or "semantic similarity"))
                        if selected.keyword_rank:
                            st.write(f"Keyword rank: **{selected.keyword_rank}**")
                        if selected.semantic_rank:
                            st.write(f"Semantic rank: **{selected.semantic_rank}**")
                        st.subheader("Direct calls")
                        rows = call_rows(symbol)
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No local direct calls were resolved from this symbol.")
            except Exception as error:
                st.error(f"Search could not run: {error}")
                if mode != "Keyword" or compare:
                    st.info("Keyword retrieval remains available without the embedding model.")

with browse_tab:
    chosen_file = st.selectbox("File", files)
    kind_filter = st.multiselect("Symbol type", ["function", "class", "method"], default=["function", "class", "method"])
    file_symbols = [symbol for symbol in repository.symbols if symbol.path == chosen_file and symbol.kind in kind_filter]
    if file_symbols:
        chosen_symbol_name = st.selectbox("Function or class", range(len(file_symbols)), format_func=lambda i: file_symbols[i].qualified_name)
        chosen_symbol = file_symbols[chosen_symbol_name]
        st.caption(f"{chosen_symbol.path}:{chosen_symbol.start_line}–{chosen_symbol.end_line}")
        st.code(chosen_symbol.source, language="python", line_numbers=True)
        rows = call_rows(chosen_symbol)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No symbols match this filter.")
    with st.expander("View complete file"):
        st.code(repository.sources[chosen_file], language="python", line_numbers=True)

with method_tab:
    st.markdown(
        """
        **AST extraction.** Python's parser identifies functions, classes, methods, docstrings, calls, and exact line ranges without executing the repository.

        **Keyword retrieval.** BM25 rewards matching identifiers and rare terms while normalizing for symbol length.

        **Semantic retrieval.** A pretrained MiniLM Sentence Transformer maps the query and code summaries to normalized vectors; their dot product is cosine similarity.

        **Hybrid retrieval.** Reciprocal Rank Fusion adds `1 / (60 + rank)` from the keyword and semantic result lists. It combines positions because their raw scores use different scales.

        **Call resolution.** A unique direct name such as `retry_payment()` is resolved. Calls through an uncertain receiver such as `order.total()` are labelled ambiguous. Imported or unmatched calls are treated as external.
        """
    )
    st.caption(
        f"Parse cache this run: {repository.cache_hits} reused, {repository.parsed_files} parsed. "
        "A content hash invalidates only changed files. Rankings are suggestions, not confidence probabilities."
    )
