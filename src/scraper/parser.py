"""HTML parsing and content extraction.

Scrapy hands over the raw HTML; this module turns it into structured data.
Two parsing technologies are used deliberately, each where it is strongest:

* **BeautifulSoup with the lxml tree builder** for navigating the document,
  stripping boilerplate (``script``, ``style``, ``nav``, ``footer``, ...) and
  reading metadata. BeautifulSoup tolerates the malformed markup that real
  web pages are full of, and the ``lxml`` backend keeps it fast because the
  parsing itself happens in C.
* **lxml with XPath** for the targeted, per-site field extraction, where an
  XPath expression is more precise and considerably faster than walking the
  soup tree.

The output of :func:`parse_html` is a :class:`ParsedContent`, which maps
directly onto the ``parsed_pages`` table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
from lxml import html as lxml_html

logger = logging.getLogger(__name__)

# Elements that never contain article content.
BOILERPLATE_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside", "form", "button",
)

# Only follow links that look like documents.
SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z", ".exe", ".dmg",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".webm",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class ParsedContent:
    """Structured result of parsing one HTML document."""

    url: str
    site: str
    title: str | None = None
    description: str | None = None
    language: str | None = None
    headings: list[str] = field(default_factory=list)
    clean_text: str = ""
    structured_data: dict[str, Any] = field(default_factory=dict)
    links: list[dict[str, Any]] = field(default_factory=list)
    parser_used: str = "beautifulsoup+lxml"

    @property
    def word_count(self) -> int:
        return len(self.clean_text.split())


def site_of(url: str) -> str:
    """Registrable host of a URL, used to group and filter by site."""
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


def normalise_url(base_url: str, href: str) -> str | None:
    """Resolve ``href`` against ``base_url`` and drop anything unusable.

    Normalising before enqueueing is important: without it the frontier fills
    up with the same page reached through different fragments or trailing
    slashes, and the crawler does redundant work.
    """
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    absolute = urljoin(base_url, href)
    absolute, _fragment = urldefrag(absolute)       # drop #anchors
    parsed = urlparse(absolute)

    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        return None

    # Normalise away a trailing slash on non-root paths so "/a" and "/a/" are
    # recognised as the same page.
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    rebuilt = f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
    if parsed.query:
        rebuilt += f"?{parsed.query}"
    return rebuilt


def _clean_text(soup: BeautifulSoup) -> str:
    """Strip boilerplate and return readable text."""
    for tag_name in BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator="\n")
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _extract_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    """Title, description and language from the document head."""
    meta: dict[str, Any] = {"title": None, "description": None, "language": None}

    if soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()
    if not meta["title"]:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            meta["title"] = og_title["content"].strip()
    if not meta["title"]:
        h1 = soup.find("h1")
        if h1:
            meta["title"] = h1.get_text(strip=True)

    for finder in (
        lambda: soup.find("meta", attrs={"name": "description"}),
        lambda: soup.find("meta", property="og:description"),
    ):
        tag = finder()
        if tag and tag.get("content"):
            meta["description"] = tag["content"].strip()
            break

    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        meta["language"] = html_tag["lang"].strip()[:16]

    return meta


def _extract_headings(soup: BeautifulSoup, limit: int = 30) -> list[str]:
    headings: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text and text not in headings:
            headings.append(text)
        if len(headings) >= limit:
            break
    return headings


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    """Every in-document link, normalised and de-duplicated."""
    base_site = site_of(base_url)
    seen: set[str] = set()
    links: list[dict[str, Any]] = []

    for anchor in soup.find_all("a", href=True):
        target = normalise_url(base_url, anchor["href"])
        if target is None or target in seen:
            continue
        seen.add(target)
        links.append(
            {
                "url": target,
                "anchor_text": (anchor.get_text(strip=True) or None),
                "internal": site_of(target) == base_site,
            }
        )
    return links


# ---------------------------------------------------------------------------
# Per-site structured extraction, using lxml XPath.
#
# This is the part that differs per target website. Each extractor receives
# the lxml tree and returns the site-specific fields that the generic text
# extraction cannot know about (a price, an author, a star rating...).
# ---------------------------------------------------------------------------


def _xpath_first(tree: Any, expression: str) -> str | None:
    found = tree.xpath(expression)
    if not found:
        return None
    value = found[0]
    text = value if isinstance(value, str) else value.text_content()
    text = (text or "").strip()
    return text or None


def _extract_books_toscrape(tree: Any) -> dict[str, Any]:
    """Structured fields for books.toscrape.com (an e-commerce layout)."""
    data: dict[str, Any] = {}

    data["product_title"] = _xpath_first(tree, "//div[contains(@class,'product_main')]/h1")
    data["price"] = _xpath_first(tree, "//p[@class='price_color']")
    data["availability"] = _xpath_first(
        tree, "//p[contains(@class,'availability')]"
    )

    rating = tree.xpath("//p[contains(@class,'star-rating')]/@class")
    if rating:
        words = str(rating[0]).replace("star-rating", "").strip()
        data["rating"] = words or None

    # The product information table (UPC, product type, taxes, reviews...).
    table: dict[str, str] = {}
    for row in tree.xpath("//table[contains(@class,'table-striped')]//tr"):
        header = row.xpath("./th")
        cell = row.xpath("./td")
        if header and cell:
            key = (header[0].text_content() or "").strip()
            value = (cell[0].text_content() or "").strip()
            if key:
                table[key] = value
    if table:
        data["product_information"] = table

    breadcrumb = [
        (node.text_content() or "").strip()
        for node in tree.xpath("//ul[@class='breadcrumb']//a")
    ]
    if breadcrumb:
        data["breadcrumb"] = breadcrumb
        if len(breadcrumb) > 2:
            data["category"] = breadcrumb[-1]

    # Listing pages: capture every product teaser on the page.
    products = []
    for pod in tree.xpath("//article[contains(@class,'product_pod')]"):
        title = pod.xpath(".//h3/a/@title")
        price = pod.xpath(".//p[@class='price_color']/text()")
        products.append(
            {
                "title": (title[0].strip() if title else None),
                "price": (price[0].strip() if price else None),
            }
        )
    if products:
        data["products_on_page"] = products
        data["product_count"] = len(products)

    return {k: v for k, v in data.items() if v}


def _extract_quotes_toscrape(tree: Any) -> dict[str, Any]:
    """Structured fields for quotes.toscrape.com (a content/listing layout)."""
    data: dict[str, Any] = {}

    quotes = []
    for node in tree.xpath("//div[@class='quote']"):
        text = _xpath_first(node, ".//span[@class='text']")
        author = _xpath_first(node, ".//small[@class='author']")
        tags = [
            (t.text_content() or "").strip()
            for t in node.xpath(".//a[@class='tag']")
        ]
        if text:
            quotes.append(
                {"text": text, "author": author, "tags": [t for t in tags if t]}
            )
    if quotes:
        data["quotes"] = quotes
        data["quote_count"] = len(quotes)
        data["authors"] = sorted({q["author"] for q in quotes if q["author"]})
        all_tags = sorted({tag for q in quotes for tag in q["tags"]})
        if all_tags:
            data["tags"] = all_tags

    # Author detail pages.
    author_name = _xpath_first(tree, "//h3[@class='author-title']")
    if author_name:
        data["author_name"] = author_name
        data["author_born_date"] = _xpath_first(
            tree, "//span[@class='author-born-date']"
        )
        data["author_born_location"] = _xpath_first(
            tree, "//span[@class='author-born-location']"
        )
        data["author_description"] = _xpath_first(
            tree, "//div[@class='author-description']"
        )

    top_tags = [
        (t.text_content() or "").strip()
        for t in tree.xpath("//span[@class='tag-item']//a")
    ]
    if top_tags:
        data["top_tags"] = top_tags

    return {k: v for k, v in data.items() if v}


# Registry: site -> extractor. A site with no entry still gets the full
# generic extraction (title, description, headings, clean text, links).
SITE_EXTRACTORS: dict[str, Any] = {
    "books.toscrape.com": _extract_books_toscrape,
    "quotes.toscrape.com": _extract_quotes_toscrape,
}


def parse_html(html: str, url: str) -> ParsedContent:
    """Parse raw HTML into a :class:`ParsedContent`.

    The document is parsed twice on purpose: once into a BeautifulSoup tree
    (whose copy is mutated to strip boilerplate before the text is taken) and
    once into an lxml tree used for the XPath-based structured extraction, so
    the two stages cannot interfere with each other.
    """
    site = site_of(url)

    # --- BeautifulSoup over the lxml tree builder ---
    soup = BeautifulSoup(html, "lxml")
    metadata = _extract_metadata(soup)
    headings = _extract_headings(soup)
    links = _extract_links(soup, url)
    # _clean_text decomposes nodes, so it runs last on this tree.
    clean_text = _clean_text(soup)

    # --- raw lxml tree for XPath field extraction ---
    structured: dict[str, Any] = {}
    extractor = SITE_EXTRACTORS.get(site)
    if extractor is not None:
        try:
            tree = lxml_html.fromstring(html)
            structured = extractor(tree)
        except Exception as exc:  # noqa: BLE001 - a bad page must not kill the crawl
            logger.warning("Structured extraction failed for %s: %s", url, exc)

    structured.setdefault("link_count", len(links))
    structured.setdefault(
        "internal_link_count", sum(1 for link in links if link["internal"])
    )

    return ParsedContent(
        url=url,
        site=site,
        title=metadata["title"],
        description=metadata["description"],
        language=metadata["language"],
        headings=headings,
        clean_text=clean_text,
        structured_data=structured,
        links=links,
        parser_used="beautifulsoup+lxml" if extractor is None else "beautifulsoup+lxml-xpath",
    )
