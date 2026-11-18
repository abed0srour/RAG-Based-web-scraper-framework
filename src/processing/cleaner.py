"""Stage 1 of the data-processing module: HTML and text cleaning.

The scraping module (Task 2) deliberately stores the HTTP response *verbatim*
in ``raw_pages``. That raw payload is unusable as it stands: roughly two thirds
of it is markup, inline JavaScript, CSS, navigation chrome, cookie banners and
tracking pixels. This module turns it into plain, readable text.

The cleaning is done in four passes, each removing a different class of noise:

1. **Executable and non-content elements** - ``script``, ``style``,
   ``noscript``, ``template``, ``svg``, ``iframe`` and HTML comments. These are
   dropped first, because their *text* content (JavaScript source, CSS rules)
   would otherwise be mistaken for page text by any text extractor.
2. **Boilerplate layout elements** - ``nav``, ``header``, ``footer``,
   ``aside``, ``form``, ``button``. These contain real words, but the same
   words on every page of the site, which is exactly what a retrieval system
   must not be filled with.
3. **Character-level normalisation** - HTML entities are unescaped, the text
   is put into Unicode NFKC form, and invisible characters (zero-width spaces,
   soft hyphens, BOMs, control codes) are removed. Without this step the same
   word can occur in several byte representations and no lookup matches.
4. **Whitespace and line-level normalisation** - runs of spaces and tabs are
   collapsed, lines are stripped, empty lines are capped, and consecutive
   duplicate lines (a very common artefact of menu markup) are folded into one.

Everything here is pure text processing: no network, no database. That is what
makes the stage trivially unit-testable and cheap to re-run over stored HTML
when the rules change.
"""

from __future__ import annotations

import html as html_module
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# Version of the cleaning rules. It is written into every document so that a
# re-processing run can be told apart from the one before it.
CLEANER_VERSION = "1.0"

# Pass 1: elements whose textual content is code or decoration, never prose.
CODE_TAGS = (
    "script", "style", "noscript", "template", "svg", "canvas",
    "iframe", "object", "embed", "applet", "map", "audio", "video",
)

# Pass 2: elements that hold the repeated site furniture.
BOILERPLATE_TAGS = (
    "nav", "header", "footer", "aside", "form", "button",
    "select", "option", "label", "input", "textarea",
)

# Attributes carrying tracking/jQuery payloads. Removing them shrinks the
# document before it is walked, and keeps `data-*` blobs out of the text.
NOISY_ATTRIBUTE_PREFIXES = ("on", "data-")

# Block-level tags: a line break is forced around them so that two paragraphs
# do not get glued into one sentence when the markup has no whitespace.
BLOCK_TAGS = (
    "p", "div", "section", "article", "li", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "figcaption",
)

# Invisible characters that survive NFKC normalisation and must go explicitly.
_INVISIBLE = dict.fromkeys(
    [
        0x00AD,  # soft hyphen
        0x200B, 0x200C, 0x200D,  # zero-width space / non-joiner / joiner
        0x200E, 0x200F,  # left-to-right / right-to-left marks
        0x2028, 0x2029,  # line / paragraph separators
        0xFEFF,  # byte-order mark
    ]
)

_HORIZONTAL_WS = re.compile(r"[ \t  -   　]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A tag-looking fragment in text that is supposed to be plain already.
_STRAY_TAG = re.compile(r"<[^<>]{1,2000}?>", re.DOTALL)
_SCRIPT_BLOCK = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
# Lines that are navigation furniture rather than content.
_NAV_LINE = re.compile(
    r"^(home|menu|search|login|log in|sign in|sign up|register|cart|basket|"
    r"next|previous|prev|back|top|skip to content|accept( all)?( cookies)?|"
    r"cookie[s]? policy|privacy policy|terms( of use| and conditions)?|"
    r"share|tweet|follow us|subscribe|newsletter|all rights reserved)$",
    re.IGNORECASE,
)


@dataclass
class CleanedText:
    """Result of cleaning one HTML document.

    Besides the text itself this carries the measurements that make the
    cleaning *visible* in the report: how many bytes went in, how many came
    out, and how many elements each pass removed.
    """

    text: str
    lines: list[str] = field(default_factory=list)
    raw_length: int = 0
    removed_elements: dict[str, int] = field(default_factory=dict)
    removed_comments: int = 0
    dropped_lines: int = 0
    duration_ms: float = 0.0

    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def reduction_ratio(self) -> float:
        """Fraction of the raw payload that was noise (0.0 - 1.0)."""
        if not self.raw_length:
            return 0.0
        return round(1.0 - (self.length / self.raw_length), 4)

    def summary(self) -> dict[str, float | int]:
        """Compact, JSON-serialisable view used in the document metadata."""
        return {
            "raw_length": self.raw_length,
            "clean_length": self.length,
            "reduction_ratio": self.reduction_ratio,
            "removed_elements": sum(self.removed_elements.values()),
            "removed_comments": self.removed_comments,
            "dropped_lines": self.dropped_lines,
            "clean_ms": round(self.duration_ms, 3),
        }


# ---------------------------------------------------------------------------
# Character- and line-level helpers (usable on their own, and unit-tested)
# ---------------------------------------------------------------------------


def strip_tags(text: str) -> str:
    """Remove markup from a string that is supposed to be plain text already.

    Some scraped fields (a product description, a JSON-embedded snippet) carry
    a second layer of HTML inside what the page presented as text. Script and
    style blocks are removed *with their contents* first, so that stripping the
    tags cannot promote JavaScript source into the text.
    """
    if not text:
        return ""
    without_code = _SCRIPT_BLOCK.sub(" ", text)
    return _STRAY_TAG.sub(" ", without_code)


def normalise_characters(text: str) -> str:
    """Unescape entities, apply Unicode NFKC and delete invisible codepoints."""
    if not text:
        return ""
    # Entities first: "&lt;b&gt;" becomes "<b>", which strip_tags can then see.
    text = html_module.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    return _CONTROL_CHARS.sub("", text)


def normalise_whitespace(text: str, *, max_blank_lines: int = 1) -> str:
    """Collapse horizontal whitespace and cap runs of empty lines."""
    if not text:
        return ""
    lines = (_HORIZONTAL_WS.sub(" ", line).strip() for line in text.splitlines())
    text = "\n".join(lines)
    separator = "\n" * (max_blank_lines + 1)
    return _BLANK_LINES.sub(separator, text).strip()


def filter_lines(
    lines: Iterable[str],
    *,
    min_length: int = 2,
    drop_navigation: bool = True,
    drop_consecutive_duplicates: bool = True,
) -> tuple[list[str], int]:
    """Drop junk lines; return the kept lines and how many were discarded.

    Three kinds of line are removed: ones too short to carry meaning, ones
    that match a known navigation label, and a line identical to the one
    before it (menus repeated in a mobile and a desktop container produce
    exactly that).
    """
    kept: list[str] = []
    dropped = 0
    previous: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < min_length:
            dropped += 1
            continue
        if drop_navigation and _NAV_LINE.match(line):
            dropped += 1
            continue
        if drop_consecutive_duplicates and line == previous:
            dropped += 1
            continue
        kept.append(line)
        previous = line

    return kept, dropped


# ---------------------------------------------------------------------------
# The full HTML cleaning pass
# ---------------------------------------------------------------------------


def _remove_elements(soup: BeautifulSoup, tags: Iterable[str]) -> dict[str, int]:
    """Decompose every element of the given tags, counting what was removed."""
    removed: dict[str, int] = {}
    for tag_name in tags:
        found = soup.find_all(tag_name)
        if not found:
            continue
        removed[tag_name] = len(found)
        for element in found:
            element.decompose()
    return removed


def _remove_comments(soup: BeautifulSoup) -> int:
    comments = soup.find_all(string=lambda node: isinstance(node, Comment))
    for comment in comments:
        comment.extract()
    return len(comments)


def _strip_noisy_attributes(soup: BeautifulSoup) -> None:
    """Delete event handlers and ``data-*`` payload attributes."""
    for element in soup.find_all(True):
        if not element.attrs:
            continue
        element.attrs = {
            name: value
            for name, value in element.attrs.items()
            if not name.lower().startswith(NOISY_ATTRIBUTE_PREFIXES)
        }


def _force_block_breaks(soup: BeautifulSoup) -> None:
    """Append a newline inside block elements so paragraphs stay separated."""
    for element in soup.find_all(BLOCK_TAGS):
        element.append("\n")


def clean_html(
    raw_html: str,
    *,
    remove_boilerplate: bool = True,
    drop_navigation_lines: bool = True,
) -> CleanedText:
    """Turn a raw HTML document into clean, readable text.

    ``remove_boilerplate`` keeps the layout containers when set to False, which
    is useful for pages whose entire content sits inside a ``<header>`` - the
    benchmark uses it to show the cost and the benefit of that pass.
    """
    started = time.perf_counter()
    raw_html = raw_html or ""
    raw_length = len(raw_html)

    if not raw_html.strip():
        return CleanedText(text="", lines=[], raw_length=raw_length)

    # lxml as the tree builder: the parsing itself happens in C, and it
    # tolerates the broken markup that real pages are full of.
    soup = BeautifulSoup(raw_html, "lxml")

    removed = _remove_elements(soup, CODE_TAGS)
    removed_comments = _remove_comments(soup)
    if remove_boilerplate:
        removed.update(_remove_elements(soup, BOILERPLATE_TAGS))
    _strip_noisy_attributes(soup)
    _force_block_breaks(soup)

    text = soup.get_text(separator="\n")
    text = normalise_characters(text)
    # Any markup that was nested inside a text node (double-encoded HTML)
    # becomes visible only after unescaping, so strip tags again here.
    text = strip_tags(text)
    text = normalise_whitespace(text)

    lines, dropped = filter_lines(
        text.splitlines(), drop_navigation=drop_navigation_lines
    )
    text = "\n".join(lines)

    return CleanedText(
        text=text,
        lines=lines,
        raw_length=raw_length,
        removed_elements=removed,
        removed_comments=removed_comments,
        dropped_lines=dropped,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def clean_value(value: object) -> object:
    """Recursively clean the strings inside a scraped structured field.

    The per-site extractors of Task 2 return nested dictionaries and lists
    (a product information table, a list of quotes). Their values are page
    text and therefore carry the same entities and stray markup as the body,
    so the whole structure is walked and every string is cleaned in place.
    """
    if isinstance(value, str):
        return normalise_whitespace(strip_tags(normalise_characters(value)))
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
    return value
