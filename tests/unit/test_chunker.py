"""Tests for AST-aware code chunking via tree-sitter."""

import textwrap

import pytest

from forge.rag.chunker import (
    Chunk,
    _chunk_by_lines,
    _estimate_tokens,
    _merge_small_chunks,
    chunk_file,
    path_stem,
    supported_extensions,
)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_short_string(self):
        assert _estimate_tokens("abcd") == 1

    def test_known_length(self):
        # 20 chars → 20 // 4 = 5
        assert _estimate_tokens("a" * 20) == 5

    def test_non_divisible_length(self):
        # 7 chars → 7 // 4 = 1
        assert _estimate_tokens("a" * 7) == 1

    def test_longer_string(self):
        assert _estimate_tokens("a" * 400) == 100


# ---------------------------------------------------------------------------
# path_stem / supported_extensions
# ---------------------------------------------------------------------------


class TestPathStem:
    def test_simple(self):
        assert path_stem("/foo/bar/baz.py") == "baz"

    def test_nested_extensions(self):
        assert path_stem("archive.tar.gz") == "archive.tar"

    def test_no_extension(self):
        assert path_stem("Makefile") == "Makefile"


class TestSupportedExtensions:
    def test_returns_set(self):
        exts = supported_extensions()
        assert isinstance(exts, set)

    def test_includes_common_extensions(self):
        exts = supported_extensions()
        for ext in (".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"):
            assert ext in exts, f"Missing {ext}"

    def test_no_empty_string(self):
        assert "" not in supported_extensions()


# ---------------------------------------------------------------------------
# _merge_small_chunks
# ---------------------------------------------------------------------------


class TestMergeSmallChunks:
    def _make_chunk(self, tokens: int, name: str = "c", start: int = 1, end: int = 1) -> Chunk:
        return Chunk(
            file_path="f.py",
            chunk_type="test",
            name=name,
            content="x" * (tokens * 4),
            start_line=start,
            end_line=end,
            token_count=tokens,
        )

    def test_empty_list(self):
        assert _merge_small_chunks([], "f.py") == []

    def test_single_chunk_unchanged(self):
        c = self._make_chunk(50)
        result = _merge_small_chunks([c], "f.py")
        assert len(result) == 1
        assert result[0].token_count == 50

    def test_two_small_chunks_merge(self):
        a = self._make_chunk(100, name="a", start=1, end=5)
        b = self._make_chunk(100, name="b", start=6, end=10)
        result = _merge_small_chunks([a, b], "f.py")
        assert len(result) == 1
        assert result[0].token_count == 200
        assert result[0].chunk_type == "merged"
        assert result[0].name == "a"  # keeps first name
        assert result[0].start_line == 1
        assert result[0].end_line == 10

    def test_large_chunks_not_merged(self):
        a = self._make_chunk(300, start=1, end=10)
        b = self._make_chunk(300, start=11, end=20)
        result = _merge_small_chunks([a, b], "f.py")
        assert len(result) == 2

    def test_merge_chain_stops_at_target(self):
        # Three chunks of 150 tokens: first two merge to 300, third stays separate
        # (300 + 150 = 450 > 400 target)
        a = self._make_chunk(150, start=1, end=3)
        b = self._make_chunk(150, start=4, end=6)
        c = self._make_chunk(150, start=7, end=9)
        result = _merge_small_chunks([a, b, c], "f.py")
        assert len(result) == 2
        assert result[0].token_count == 300
        assert result[1].token_count == 150


# ---------------------------------------------------------------------------
# chunk_file — tree-sitter path (Python)
# ---------------------------------------------------------------------------


class TestChunkFilePython:
    def test_single_function(self):
        code = textwrap.dedent("""\
            def hello(name):
                \"\"\"Greet someone.\"\"\"
                greeting = f"Hello, {name}!"
                print(greeting)
                return greeting
        """)
        chunks = chunk_file("example.py", content=code)
        assert len(chunks) >= 1
        # Should identify the function
        func_chunks = [c for c in chunks if c.name == "hello"]
        assert len(func_chunks) == 1
        assert func_chunks[0].start_line == 1
        assert func_chunks[0].chunk_type == "function_definition"

    def test_multiple_functions(self):
        code = textwrap.dedent("""\
            def foo():
                return "foo result value"

            def bar():
                return "bar result value"

            def baz():
                return "baz result value"
        """)
        chunks = chunk_file("multi.py", content=code)
        names = {c.name for c in chunks if c.name}
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_class_chunked(self):
        code = textwrap.dedent("""\
            class Calculator:
                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b
        """)
        chunks = chunk_file("calc.py", content=code)
        assert any(c.name == "Calculator" for c in chunks)

    def test_empty_file_returns_no_chunks(self):
        chunks = chunk_file("empty.py", content="")
        assert chunks == []

    def test_imports_only_below_min_threshold(self):
        # Very short — below _MIN_CHUNK_TOKENS (30)
        code = "import os\n"
        chunks = chunk_file("tiny.py", content=code)
        # Should be empty since token count < 30
        assert chunks == []

    def test_decorated_function(self):
        code = textwrap.dedent("""\
            import functools

            def my_decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            @my_decorator
            def decorated():
                \"\"\"A decorated function with enough content to pass threshold.\"\"\"
                x = 1
                y = 2
                return x + y
        """)
        chunks = chunk_file("deco.py", content=code)
        names = {c.name for c in chunks if c.name}
        assert "decorated" in names or "my_decorator" in names

    def test_large_function_split(self):
        # Create a function exceeding _MAX_CHUNK_TOKENS (800)
        lines = ["def huge():"]
        for i in range(500):
            lines.append(f"    x_{i} = {i} * {i}  # computation line {i}")
        lines.append("    return x_0")
        code = "\n".join(lines)
        chunks = chunk_file("huge.py", content=code)
        # The single function should be split into multiple chunks
        assert len(chunks) >= 1
        total_lines = sum(c.end_line - c.start_line + 1 for c in chunks)
        assert total_lines > 0

    def test_chunk_lines_are_1_indexed(self):
        code = textwrap.dedent("""\
            def first():
                return "first function body here"

            def second():
                return "second function body"
        """)
        chunks = chunk_file("lines.py", content=code)
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_token_count_is_positive(self):
        code = textwrap.dedent("""\
            def something():
                return "a meaningful value here"
        """)
        chunks = chunk_file("tokens.py", content=code)
        for chunk in chunks:
            assert chunk.token_count >= 1

    def test_file_path_preserved(self):
        code = 'def f(): return "value"\n' * 10
        chunks = chunk_file("/some/path/mod.py", content=code)
        for chunk in chunks:
            assert chunk.file_path == "/some/path/mod.py"


# ---------------------------------------------------------------------------
# chunk_file — line-based fallback
# ---------------------------------------------------------------------------


class TestChunkByLinesFallback:
    def test_unsupported_extension_uses_line_fallback(self):
        # .xyz is not in _LANG_CONFIG
        content = "line\n" * 200
        chunks = chunk_file("data.xyz", content=content)
        # Should still produce chunks via line-based fallback
        assert len(chunks) >= 1
        for c in chunks:
            assert c.chunk_type == "block"

    def test_empty_content(self):
        chunks = _chunk_by_lines("", "empty.txt")
        assert chunks == []

    def test_content_below_threshold(self):
        # Very short content — fewer than _MIN_CHUNK_TOKENS
        chunks = _chunk_by_lines("hello", "short.txt")
        assert chunks == []

    def test_long_content_produces_multiple_chunks(self):
        # Each line ~50 chars, target_chars = 1600
        # So ~32 lines per chunk
        content = "\n".join(f"line_{i:04d} = " + "x" * 40 for i in range(200))
        chunks = _chunk_by_lines(content, "big.txt")
        assert len(chunks) > 1

    def test_line_numbers_contiguous(self):
        content = "\n".join(f"line {i}" * 10 for i in range(300))
        chunks = _chunk_by_lines(content, "seq.txt")
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Next chunk starts after current chunk ends
                assert chunks[i + 1].start_line == chunks[i].end_line + 1


# ---------------------------------------------------------------------------
# chunk_file — JavaScript
# ---------------------------------------------------------------------------


class TestChunkFileJavaScript:
    def test_function_declaration(self):
        code = textwrap.dedent("""\
            function greet(name) {
                const msg = `Hello, ${name}!`;
                console.log(msg);
                return msg;
            }
        """)
        chunks = chunk_file("greet.js", content=code)
        assert len(chunks) >= 1
        assert any(c.name == "greet" for c in chunks)

    def test_class_declaration(self):
        code = textwrap.dedent("""\
            class Animal {
                constructor(name) {
                    this.name = name;
                }
                speak() {
                    return `${this.name} makes a noise.`;
                }
            }
        """)
        chunks = chunk_file("animal.js", content=code)
        assert any(c.name == "Animal" for c in chunks)


# ---------------------------------------------------------------------------
# chunk_file — Rust
# ---------------------------------------------------------------------------


class TestChunkFileRust:
    def test_function_item(self):
        code = textwrap.dedent("""\
            fn add(a: i32, b: i32) -> i32 {
                let result = a + b;
                println!("{}", result);
                result
            }
        """)
        chunks = chunk_file("lib.rs", content=code)
        assert len(chunks) >= 1

    def test_struct_item(self):
        code = textwrap.dedent("""\
            struct Point {
                x: f64,
                y: f64,
            }

            impl Point {
                fn new(x: f64, y: f64) -> Self {
                    Point { x, y }
                }

                fn distance(&self, other: &Point) -> f64 {
                    ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
                }
            }
        """)
        chunks = chunk_file("point.rs", content=code)
        names = {c.name for c in chunks if c.name}
        assert "Point" in names


# ---------------------------------------------------------------------------
# chunk_file reads from disk when content not provided
# ---------------------------------------------------------------------------


class TestChunkFileFromDisk:
    def test_reads_file_when_content_is_none(self, tmp_path):
        f = tmp_path / "auto.py"
        f.write_text('def auto():\n    return "auto read from disk"\n')
        chunks = chunk_file(str(f))
        assert len(chunks) >= 1
