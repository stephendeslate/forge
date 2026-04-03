"""Integration tests — multimodal input parsing."""

import struct
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent

from forge.agent.multimodal import parse_multimodal_input


def _make_1x1_png(path: Path) -> None:
    """Write a minimal valid 1x1 red PNG."""
    import zlib

    # Minimal PNG: 8-byte signature + IHDR + IDAT + IEND
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8bit RGB
    raw_row = b"\x00\xff\x00\x00"  # filter=None, R=255 G=0 B=0
    idat_data = zlib.compress(raw_row)

    path.write_bytes(sig + _chunk(b"IHDR", ihdr_data) + _chunk(b"IDAT", idat_data) + _chunk(b"IEND", b""))


class TestMultimodalParsing:
    def test_plain_text_passthrough(self, tmp_path):
        result = parse_multimodal_input("just plain text", tmp_path)
        assert result == "just plain text"

    def test_image_extraction(self, tmp_path):
        img = tmp_path / "test.png"
        _make_1x1_png(img)
        result = parse_multimodal_input("look at @test.png please", tmp_path)
        assert isinstance(result, list)
        binary_parts = [p for p in result if isinstance(p, BinaryContent)]
        assert len(binary_parts) == 1

    def test_email_not_matched(self, tmp_path):
        result = parse_multimodal_input("email user@test.png about it", tmp_path)
        assert isinstance(result, str)
        assert result == "email user@test.png about it"

    def test_multiple_images(self, tmp_path):
        _make_1x1_png(tmp_path / "a.png")
        _make_1x1_png(tmp_path / "b.png")
        result = parse_multimodal_input("compare @a.png and @b.png", tmp_path)
        assert isinstance(result, list)
        binary_parts = [p for p in result if isinstance(p, BinaryContent)]
        assert len(binary_parts) == 2

    def test_missing_image_returns_original(self, tmp_path):
        result = parse_multimodal_input("see @missing.png here", tmp_path)
        # Missing file → no BinaryContent loaded → return original string
        assert isinstance(result, str)

    def test_non_image_extension_ignored(self, tmp_path):
        (tmp_path / "file.py").write_text("print('hi')")
        result = parse_multimodal_input("look at @file.py", tmp_path)
        assert isinstance(result, str)
