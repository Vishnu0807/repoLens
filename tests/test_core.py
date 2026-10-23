import io
import stat
import zipfile

import numpy as np
import pytest

from core import ParseCache, SearchIndex, index_sources, parse_python
from ingest import read_repository_zip


def make_zip(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, source in files.items():
            archive.writestr(path, source)
    return output.getvalue()


def test_ast_extracts_symbols_docstrings_and_lines():
    source = '''@staticmethod
async def load_order():
    """Load one order."""
    return 1

class Shop:
    def checkout(self):
        return load_order()
'''
    symbols = parse_python(source, "shop.py")
    assert [symbol.qualified_name for symbol in symbols] == ["load_order", "Shop", "Shop.checkout"]
    assert symbols[0].start_line == 1
    assert symbols[0].docstring == "Load one order."
    assert symbols[2].raw_calls[0].name == "load_order"


def test_call_resolution_marks_confident_and_ambiguous_links():
    sources = {
        "orders.py": '''def helper():
    return 1

class Order:
    def total(self):
        return helper()

    def checkout(self, other):
        return self.total() + other.total()
'''
    }
    repository = index_sources(sources)
    total = next(symbol for symbol in repository.symbols if symbol.qualified_name == "Order.total")
    checkout = next(symbol for symbol in repository.symbols if symbol.qualified_name == "Order.checkout")
    assert total.calls[0].status == "resolved"
    assert checkout.calls[0].status == "resolved"
    assert checkout.calls[1].status == "ambiguous"


def test_content_cache_only_reparses_changed_files():
    cache = ParseCache()
    sources = {"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"}
    first = index_sources(sources, cache)
    second = index_sources(sources, cache)
    changed = index_sources({**sources, "b.py": "def b():\n    return 3\n"}, cache)
    assert (first.parsed_files, first.cache_hits) == (2, 0)
    assert (second.parsed_files, second.cache_hits) == (0, 2)
    assert (changed.parsed_files, changed.cache_hits) == (1, 1)


def test_keyword_semantic_and_hybrid_search():
    repository = index_sources({"a.py": "def retry_payment():\n    return True\n\ndef reserve_stock():\n    return True\n"})
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = SearchIndex(repository.symbols, vectors)
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    assert index.search("retry payment", "keyword")[0].symbol.name == "retry_payment"
    assert index.search("charge again", "semantic", query_vector=query_vector)[0].symbol.name == "retry_payment"
    hybrid = index.search("retry payment", "hybrid", query_vector=query_vector)
    assert hybrid[0].symbol.name == "retry_payment"
    assert hybrid[0].keyword_rank == hybrid[0].semantic_rank == 1


@pytest.mark.parametrize("path", ["../evil.py", "/tmp/evil.py", "a/../../evil.py", "C:/evil.py", "a\\evil.py"])
def test_zip_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="unsafe"):
        read_repository_zip(make_zip({path: "def bad(): pass"}))


def test_zip_strips_github_wrapper_and_skips_cache():
    sources, warnings = read_repository_zip(
        make_zip({"project/a.py": "def okay(): pass", "project/__pycache__/b.py": "def ignored(): pass"})
    )
    assert sources == {"a.py": "def okay(): pass"}
    assert warnings == []


def test_zip_rejects_symlinks():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        entry = zipfile.ZipInfo("link.py")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "/etc/passwd")
    with pytest.raises(ValueError, match="Symbolic"):
        read_repository_zip(output.getvalue())
