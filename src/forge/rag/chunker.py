"""AST-aware code chunking using tree-sitter with cAST-style recursive split/merge.

Strategy:
1. Parse file with tree-sitter to get AST
2. Extract top-level "semantic" nodes (functions, classes, etc.)
3. If a node is within token budget → emit as one chunk
4. If too large → recursively split into children
5. Merge small adjacent nodes to avoid tiny chunks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser

# Mapping of file extensions to (language module, top-level node types to extract)
_LANG_CONFIG: dict[str, tuple[str, set[str]]] = {
    ".py": ("tree_sitter_python", {"function_definition", "class_definition", "decorated_definition"}),
    ".js": ("tree_sitter_javascript", {"function_declaration", "class_declaration", "export_statement", "lexical_declaration"}),
    ".jsx": ("tree_sitter_javascript", {"function_declaration", "class_declaration", "export_statement", "lexical_declaration"}),
    ".ts": ("tree_sitter_typescript", {"function_declaration", "class_declaration", "export_statement", "lexical_declaration", "interface_declaration", "type_alias_declaration"}),
    ".tsx": ("tree_sitter_typescript", {"function_declaration", "class_declaration", "export_statement", "lexical_declaration", "interface_declaration", "type_alias_declaration"}),
    ".rs": ("tree_sitter_rust", {"function_item", "struct_item", "enum_item", "impl_item", "trait_item", "mod_item"}),
    ".go": ("tree_sitter_go", {"function_declaration", "method_declaration", "type_declaration"}),
    ".java": ("tree_sitter_java", {"class_declaration", "interface_declaration", "method_declaration", "enum_declaration"}),
    ".c": ("tree_sitter_c", {"function_definition", "struct_specifier", "enum_specifier", "type_definition"}),
    ".cpp": ("tree_sitter_cpp", {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"}),
    ".h": ("tree_sitter_c", {"function_definition", "struct_specifier", "enum_specifier", "type_definition"}),
    ".hpp": ("tree_sitter_cpp", {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"}),
    ".sh": ("tree_sitter_bash", {"function_definition"}),
    ".bash": ("tree_sitter_bash", {"function_definition"}),
    ".json": ("tree_sitter_json", {"object", "array"}),
    ".yaml": ("tree_sitter_yaml", {"block_mapping", "block_sequence"}),
    ".yml": ("tree_sitter_yaml", {"block_mapping", "block_sequence"}),
    ".toml": ("tree_sitter_toml", {"table", "pair"}),
    ".html": ("tree_sitter_html", {"element"}),
    ".css": ("tree_sitter_css", {"rule_set", "media_statement"}),
    ".md": ("tree_sitter_markdown", {"section", "fenced_code_block"}),
}

# Approximate tokens ≈ chars / 4 (conservative for code)
_CHARS_PER_TOKEN = 4
_MIN_CHUNK_TOKENS = 30
_MAX_CHUNK_TOKENS = 800
_TARGET_CHUNK_TOKENS = 400


@dataclass
class Chunk:
    file_path: str
    chunk_type: str
    name: str | None
    content: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    token_count: int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _get_parser(ext: str) -> tuple[Parser, set[str]] | None:
    """Get a tree-sitter parser for the given file extension."""
    config = _LANG_CONFIG.get(ext)
    if config is None:
        return None

    module_name, top_types = config
    try:
        import importlib
        mod = importlib.import_module(module_name)
        lang = Language(mod.language())
        parser = Parser(lang)
        return parser, top_types
    except (ImportError, AttributeError):
        return None


def _node_name(node: Node) -> str | None:
    """Extract the name of a function/class/struct from its AST node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
            return child.text.decode() if child.text else None
        # Python decorated_definition wraps the actual definition
        if child.type in ("function_definition", "class_definition"):
            return _node_name(child)
    return None


def _node_to_chunk(node: Node, file_path: str) -> Chunk:
    """Convert a tree-sitter node to a Chunk."""
    text = node.text.decode() if node.text else ""
    return Chunk(
        file_path=file_path,
        chunk_type=node.type,
        name=_node_name(node),
        content=text,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        token_count=_estimate_tokens(text),
    )


def _split_node(node: Node, file_path: str, max_tokens: int) -> list[Chunk]:
    """Recursively split a large node into smaller chunks.

    cAST-style: try children first, merge small ones together.
    """
    text = node.text.decode() if node.text else ""
    tokens = _estimate_tokens(text)

    # Fits in budget → return as single chunk
    if tokens <= max_tokens:
        return [_node_to_chunk(node, file_path)]

    # Try splitting into named children (functions inside a class, etc.)
    named_children = [c for c in node.children if c.is_named]

    if not named_children or len(named_children) <= 1:
        # Can't split further — return as-is even if oversized
        return [_node_to_chunk(node, file_path)]

    # Recursively split children, then merge small adjacent ones
    child_chunks: list[Chunk] = []
    for child in named_children:
        child_chunks.extend(_split_node(child, file_path, max_tokens))

    return _merge_small_chunks(child_chunks, file_path)


def _merge_small_chunks(chunks: list[Chunk], file_path: str) -> list[Chunk]:
    """Merge adjacent small chunks to avoid fragments below the minimum size."""
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        combined_tokens = buffer.token_count + chunk.token_count
        # Merge if both are small and combined fits the target
        if combined_tokens <= _TARGET_CHUNK_TOKENS:
            buffer = Chunk(
                file_path=file_path,
                chunk_type="merged",
                name=buffer.name,  # keep first name
                content=buffer.content + "\n" + chunk.content,
                start_line=buffer.start_line,
                end_line=chunk.end_line,
                token_count=combined_tokens,
            )
        else:
            merged.append(buffer)
            buffer = chunk

    merged.append(buffer)
    return merged


def chunk_file(file_path: str, content: str | None = None) -> list[Chunk]:
    """Chunk a source file using tree-sitter AST parsing.

    Falls back to line-based chunking if tree-sitter doesn't support the language.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if content is None:
        content = path.read_text(errors="replace")

    # Try tree-sitter parsing
    result = _get_parser(ext)
    if result is not None:
        parser, top_types = result
        return _chunk_with_treesitter(parser, top_types, content, file_path)

    # Fallback: line-based chunking for unsupported languages
    return _chunk_by_lines(content, file_path)


def _chunk_with_treesitter(
    parser: Parser,
    top_types: set[str],
    content: str,
    file_path: str,
) -> list[Chunk]:
    """Parse with tree-sitter and extract semantic chunks."""
    tree = parser.parse(content.encode())
    root = tree.root_node

    chunks: list[Chunk] = []
    other_lines: list[str] = []
    other_start = 1

    for child in root.children:
        if child.type in top_types:
            # Flush accumulated non-semantic lines as a block
            if other_lines:
                block_text = "\n".join(other_lines)
                tokens = _estimate_tokens(block_text)
                if tokens >= _MIN_CHUNK_TOKENS:
                    chunks.append(Chunk(
                        file_path=file_path,
                        chunk_type="block",
                        name=None,
                        content=block_text,
                        start_line=other_start,
                        end_line=other_start + len(other_lines) - 1,
                        token_count=tokens,
                    ))
                other_lines = []

            # Process the semantic node
            node_chunks = _split_node(child, file_path, _MAX_CHUNK_TOKENS)
            chunks.extend(node_chunks)
        else:
            # Accumulate imports, comments, etc.
            text = child.text.decode() if child.text else ""
            if not other_lines:
                other_start = child.start_point[0] + 1
            other_lines.append(text)

    # Flush remaining
    if other_lines:
        block_text = "\n".join(other_lines)
        tokens = _estimate_tokens(block_text)
        if tokens >= _MIN_CHUNK_TOKENS:
            chunks.append(Chunk(
                file_path=file_path,
                chunk_type="block",
                name=None,
                content=block_text,
                start_line=other_start,
                end_line=other_start + len(other_lines) - 1,
                token_count=tokens,
            ))

    # If no chunks were produced (e.g., very small file), chunk the whole file
    if not chunks:
        tokens = _estimate_tokens(content)
        if tokens >= _MIN_CHUNK_TOKENS:
            chunks.append(Chunk(
                file_path=file_path,
                chunk_type="file",
                name=path_stem(file_path),
                content=content,
                start_line=1,
                end_line=content.count("\n") + 1,
                token_count=tokens,
            ))

    return chunks


def _chunk_by_lines(content: str, file_path: str) -> list[Chunk]:
    """Fallback: chunk by lines with a target size."""
    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    target_chars = _TARGET_CHUNK_TOKENS * _CHARS_PER_TOKEN

    current_lines: list[str] = []
    current_chars = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        current_lines.append(line)
        current_chars += len(line) + 1  # +1 for newline

        if current_chars >= target_chars:
            text = "\n".join(current_lines)
            tokens = _estimate_tokens(text)
            if tokens >= _MIN_CHUNK_TOKENS:
                chunks.append(Chunk(
                    file_path=file_path,
                    chunk_type="block",
                    name=None,
                    content=text,
                    start_line=start_line,
                    end_line=i,
                    token_count=tokens,
                ))
            current_lines = []
            current_chars = 0
            start_line = i + 1

    # Remaining lines
    if current_lines:
        text = "\n".join(current_lines)
        tokens = _estimate_tokens(text)
        if tokens >= _MIN_CHUNK_TOKENS:
            chunks.append(Chunk(
                file_path=file_path,
                chunk_type="block",
                name=None,
                content=text,
                start_line=start_line,
                end_line=len(lines),
                token_count=tokens,
            ))

    return chunks


def path_stem(file_path: str) -> str:
    return Path(file_path).stem


def supported_extensions() -> set[str]:
    """Return the set of file extensions supported by tree-sitter."""
    return set(_LANG_CONFIG.keys())
