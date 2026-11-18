"""Tests for the HTML parsing / content-extraction layer (Task 2).

These run entirely on HTML strings, with no network, no broker and no
database, which is exactly why the parsing stage was kept separate from the
pipeline that stores its output.
"""

from __future__ import annotations

import pytest

from src.scraper.parser import (
    SITE_EXTRACTORS,
    normalise_url,
    parse_html,
    site_of,
)

SAMPLE_PAGE = """
<!DOCTYPE html>
<html lang="en-GB">
  <head>
    <title>  Sample Document  </title>
    <meta name="description" content="A page used by the test suite.">
  </head>
  <body>
    <nav><a href="/menu">Menu noise</a></nav>
    <header>Header noise</header>
    <h1>Main Heading</h1>
    <h2>Second Heading</h2>
    <h2>Second Heading</h2>
    <p>The quick brown fox jumps over the lazy dog.</p>
    <a href="/relative/page">Relative</a>
    <a href="https://external.example.com/page">External</a>
    <a href="mailto:nobody@example.com">Mail</a>
    <a href="/file.pdf">A PDF</a>
    <script>var tracking = 1;</script>
    <style>body { color: red; }</style>
    <footer>Footer noise</footer>
  </body>
</html>
"""


class TestSiteOf:
    def test_extracts_host(self) -> None:
        assert site_of("https://books.toscrape.com/index.html") == "books.toscrape.com"

    def test_strips_www_prefix(self) -> None:
        assert site_of("https://www.example.com/a") == "example.com"

    def test_lowercases_host(self) -> None:
        assert site_of("https://Example.COM/a") == "example.com"


class TestNormaliseUrl:
    def test_resolves_relative_path(self) -> None:
        assert (
            normalise_url("https://a.com/dir/page.html", "other.html")
            == "https://a.com/dir/other.html"
        )

    def test_drops_fragment(self) -> None:
        assert normalise_url("https://a.com/", "/p#section") == "https://a.com/p"

    def test_keeps_query_string(self) -> None:
        assert normalise_url("https://a.com/", "/p?x=1") == "https://a.com/p?x=1"

    def test_strips_trailing_slash_on_subpath(self) -> None:
        assert normalise_url("https://a.com/", "/p/") == "https://a.com/p"

    @pytest.mark.parametrize(
        "href",
        [
            "mailto:a@b.com",
            "tel:+123456",
            "javascript:void(0)",
            "#anchor-only",
            "data:text/plain;base64,AAA",
            "",
        ],
    )
    def test_rejects_non_http_schemes(self, href: str) -> None:
        assert normalise_url("https://a.com/", href) is None

    @pytest.mark.parametrize(
        "href", ["/doc.pdf", "/img.PNG", "/archive.zip", "/style.css", "/app.js"]
    )
    def test_rejects_non_document_extensions(self, href: str) -> None:
        assert normalise_url("https://a.com/", href) is None

    def test_rejects_unsupported_scheme(self) -> None:
        assert normalise_url("https://a.com/", "ftp://a.com/x") is None


class TestParseHtml:
    @pytest.fixture(scope="class")
    def parsed(self):  # type: ignore[no-untyped-def]
        return parse_html(SAMPLE_PAGE, "https://example.com/dir/page.html")

    def test_title_is_extracted_and_stripped(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.title == "Sample Document"

    def test_description_is_extracted(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.description == "A page used by the test suite."

    def test_language_is_extracted(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.language == "en-GB"

    def test_headings_are_deduplicated_in_order(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.headings == ["Main Heading", "Second Heading"]

    def test_body_text_is_kept(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert "quick brown fox" in parsed.clean_text

    @pytest.mark.parametrize(
        "noise", ["var tracking", "color: red", "Menu noise", "Header noise", "Footer noise"]
    )
    def test_boilerplate_is_stripped(self, parsed, noise: str) -> None:  # type: ignore[no-untyped-def]
        assert noise not in parsed.clean_text

    def test_word_count_matches_clean_text(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.word_count == len(parsed.clean_text.split())
        assert parsed.word_count > 0

    def test_links_are_normalised_and_filtered(self, parsed) -> None:  # type: ignore[no-untyped-def]
        urls = {link["url"] for link in parsed.links}
        assert "https://example.com/relative/page" in urls
        assert "https://external.example.com/page" in urls
        # mailto: and .pdf must have been dropped
        assert not any("mailto" in u or u.endswith(".pdf") for u in urls)

    def test_internal_flag_is_set_per_link(self, parsed) -> None:  # type: ignore[no-untyped-def]
        by_url = {link["url"]: link for link in parsed.links}
        assert by_url["https://example.com/relative/page"]["internal"] is True
        assert by_url["https://external.example.com/page"]["internal"] is False

    def test_link_counts_are_reported(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.structured_data["link_count"] == len(parsed.links)
        assert parsed.structured_data["internal_link_count"] >= 1

    def test_site_is_derived_from_url(self, parsed) -> None:  # type: ignore[no-untyped-def]
        assert parsed.site == "example.com"

    def test_malformed_html_does_not_raise(self) -> None:
        parsed = parse_html("<html><body><p>unclosed", "https://example.com/")
        assert "unclosed" in parsed.clean_text

    def test_empty_html_yields_empty_text(self) -> None:
        parsed = parse_html("", "https://example.com/")
        assert parsed.clean_text == ""
        assert parsed.word_count == 0


class TestSiteSpecificExtractors:
    """The per-site XPath extractors used on the two target websites."""

    def test_both_target_sites_are_registered(self) -> None:
        assert "books.toscrape.com" in SITE_EXTRACTORS
        assert "quotes.toscrape.com" in SITE_EXTRACTORS

    def test_books_product_fields(self) -> None:
        html = """
        <html><body>
          <ul class="breadcrumb"><li><a href="/">Home</a></li>
              <li><a href="/b">Books</a></li><li><a href="/p">Poetry</a></li></ul>
          <div class="col-sm-6 product_main">
            <h1>A Light in the Attic</h1>
            <p class="price_color">&pound;51.77</p>
            <p class="instock availability">In stock (22 available)</p>
            <p class="star-rating Three"></p>
          </div>
          <table class="table table-striped">
            <tr><th>UPC</th><td>a897fe39b1053632</td></tr>
            <tr><th>Price (excl. tax)</th><td>&pound;51.77</td></tr>
          </table>
        </body></html>
        """
        data = parse_html(html, "https://books.toscrape.com/catalogue/x.html").structured_data
        assert data["product_title"] == "A Light in the Attic"
        assert "51.77" in data["price"]
        assert "In stock" in data["availability"]
        assert data["rating"] == "Three"
        assert data["product_information"]["UPC"] == "a897fe39b1053632"
        assert data["breadcrumb"] == ["Home", "Books", "Poetry"]
        assert data["category"] == "Poetry"

    def test_books_listing_page_products(self) -> None:
        html = """
        <html><body>
          <article class="product_pod">
            <h3><a href="/1" title="Book One">Book One</a></h3>
            <p class="price_color">&pound;10.00</p>
          </article>
          <article class="product_pod">
            <h3><a href="/2" title="Book Two">Book Two</a></h3>
            <p class="price_color">&pound;20.00</p>
          </article>
        </body></html>
        """
        data = parse_html(html, "https://books.toscrape.com/").structured_data
        assert data["product_count"] == 2
        assert data["products_on_page"][0]["title"] == "Book One"
        assert "20.00" in data["products_on_page"][1]["price"]

    def test_quotes_listing_page(self) -> None:
        html = """
        <html><body>
          <div class="quote">
            <span class="text">&ldquo;Be yourself.&rdquo;</span>
            <small class="author">Oscar Wilde</small>
            <a class="tag">be-yourself</a><a class="tag">honesty</a>
          </div>
          <div class="quote">
            <span class="text">&ldquo;Think different.&rdquo;</span>
            <small class="author">Anonymous</small>
            <a class="tag">thinking</a>
          </div>
        </body></html>
        """
        data = parse_html(html, "https://quotes.toscrape.com/").structured_data
        assert data["quote_count"] == 2
        assert data["authors"] == ["Anonymous", "Oscar Wilde"]
        assert "Be yourself" in data["quotes"][0]["text"]
        assert data["quotes"][0]["tags"] == ["be-yourself", "honesty"]
        assert set(data["tags"]) == {"be-yourself", "honesty", "thinking"}

    def test_quotes_author_detail_page(self) -> None:
        html = """
        <html><body>
          <h3 class="author-title">Albert Einstein</h3>
          <span class="author-born-date">March 14, 1879</span>
          <span class="author-born-location">in Ulm, Germany</span>
          <div class="author-description">A theoretical physicist.</div>
        </body></html>
        """
        data = parse_html(
            html, "https://quotes.toscrape.com/author/Albert-Einstein/"
        ).structured_data
        assert data["author_name"] == "Albert Einstein"
        assert data["author_born_date"] == "March 14, 1879"
        assert "Ulm" in data["author_born_location"]
        assert "physicist" in data["author_description"]

    def test_unknown_site_still_gets_generic_extraction(self) -> None:
        parsed = parse_html(SAMPLE_PAGE, "https://unknown-site.test/page")
        assert parsed.title == "Sample Document"
        assert parsed.parser_used == "beautifulsoup+lxml"
        assert "link_count" in parsed.structured_data

    def test_extractor_failure_is_contained(self) -> None:
        """A page that breaks the XPath extractor must not break the crawl."""
        parsed = parse_html("<html><body><div class='quote'></div>", "https://quotes.toscrape.com/")
        # No exception, and the generic fields are still present.
        assert "link_count" in parsed.structured_data
