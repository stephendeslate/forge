"""Multimodal input parsing — extract @path/to/image references from user input."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic_ai.messages import BinaryContent, UserContent

from forge.log import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})

# Match @/absolute/path.ext or @relative/path.ext (image extensions only)
# Negative lookbehind avoids matching email addresses
_IMAGE_REF_RE = re.compile(
    r"(?<!\w)@((?:/|\.{0,2}/)?[\w./_-]+\.(?:" + "|".join(
        ext.lstrip(".") for ext in IMAGE_EXTENSIONS
    ) + r"))\b",
    re.IGNORECASE,
)


def parse_multimodal_input(
    text: str, cwd: Path,
) -> str | list[UserContent]:
    """Extract @file.png references from user input.

    Returns a mixed content list if images are found, or the original
    text string if no images are referenced.
    """
    matches = list(_IMAGE_REF_RE.finditer(text))
    if not matches:
        return text

    # Resolve paths and load images
    images: list[BinaryContent] = []
    clean_text = text
    for match in reversed(matches):  # reverse to preserve offsets
        raw_path = match.group(1)
        p = Path(raw_path)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        else:
            p = p.resolve()

        if not p.is_file():
            logger.warning("Image not found: %s", p)
            continue

        try:
            bc = BinaryContent.from_path(p)
            images.append(bc)
            # Remove the @path from text
            clean_text = clean_text[:match.start()] + clean_text[match.end():]
        except Exception as e:
            logger.warning("Failed to load image %s: %s", p, e)

    if not images:
        return text

    clean_text = clean_text.strip()
    parts: list[UserContent] = []
    if clean_text:
        parts.append(clean_text)
    parts.extend(images)
    return parts
