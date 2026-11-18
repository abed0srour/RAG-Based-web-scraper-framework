"""Tests for the PostgreSQL storage layer (Task 2).

Skipped automatically when the database container is not running.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from src.storage.database import (
    ScrapeRepository,
    get_engine,
    init_schema,
    node_id,
    sha256,
)
from src.storage.models import Base


def _database_available() -> bool:
    try:
        with get_engine().connect() as c:
            c.execute(text("select 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


database_required = pytest.mark.skipif(
    not _database_available(), reason="PostgreSQL is not reachable"
)


class TestHelpers:
    def test_sha256_is_stable(self) -> None:
        assert sha256("abc") == sha256("abc")
        assert len(sha256("abc")) == 64

    def test_sha256_differs_for_different_input(self) -> None:
        assert sha256("abc") != sha256("abd")

    def test_node_id_has_host_and_pid(self) -> None:
        assert ":" in node_id()


@database_required
class TestSchema:
    def test_every_table_exists(self) -> None:
        init_schema()
        with get_engine().connect() as c:
            found = set(
                c.execute(
                    text(
                        "select table_name from information_schema.tables "
                        "where table_schema = 'public'"
                    )
                ).scalars()
            )
        assert {"crawl_jobs", "raw_pages", "parsed_pages", "discovered_links"} <= found

    def test_models_declare_four_tables(self) -> None:
        assert len(Base.metadata.tables) == 4


@database_required
class TestRepository:
    @pytest.fixture()
    def job(self):  # type: ignore[no-untyped-def]
        init_schema()
        site = f"test-{uuid.uuid4().hex[:8]}.local"
        job_id = ScrapeRepository.create_job(
            job_name="pytest", site=site, seed_urls=[f"https://{site}/"]
        )
        yield job_id, site
        # Removing the job cascades to its pages, keeping the database clean.
        with get_engine().begin() as c:
            c.execute(text("delete from crawl_jobs where id = :i"), {"i": job_id})
            c.execute(text("delete from discovered_links where site = :s"), {"s": site})

    def test_create_job(self, job) -> None:  # type: ignore[no-untyped-def]
        job_id, site = job
        with get_engine().connect() as c:
            row = c.execute(
                text("select job_name, site, status from crawl_jobs where id = :i"),
                {"i": job_id},
            ).one()
        assert row == ("pytest", site, "running")

    def test_save_raw_page(self, job) -> None:  # type: ignore[no-untyped-def]
        job_id, site = job
        raw_id, _ = ScrapeRepository.save_raw_page(
            job_id=job_id,
            url=f"https://{site}/a",
            site=site,
            depth=0,
            http_status=200,
            content_type="text/html",
            headers={"Server": "pytest"},
            raw_html="<html><body>hello</body></html>",
        )
        assert raw_id > 0

        with get_engine().connect() as c:
            row = c.execute(
                text(
                    "select http_status, content_length, content_hash, fetched_by "
                    "from raw_pages where id = :i"
                ),
                {"i": raw_id},
            ).one()
        assert row[0] == 200
        assert row[1] == len("<html><body>hello</body></html>")
        assert row[2] == sha256("<html><body>hello</body></html>")
        assert ":" in row[3]

    def test_duplicate_url_updates_instead_of_duplicating(self, job) -> None:  # type: ignore[no-untyped-def]
        """At-least-once delivery means the same URL can arrive twice."""
        job_id, site = job
        url = f"https://{site}/dup"
        common = dict(
            job_id=job_id, url=url, site=site, depth=0, http_status=200,
            content_type="text/html", headers={},
        )

        first_id, _ = ScrapeRepository.save_raw_page(**common, raw_html="<p>v1</p>")
        second_id, _ = ScrapeRepository.save_raw_page(**common, raw_html="<p>v2</p>")

        assert first_id == second_id      # same row, not a second one
        with get_engine().connect() as c:
            count = c.execute(
                text("select count(*) from raw_pages where url = :u"), {"u": url}
            ).scalar_one()
            html = c.execute(
                text("select raw_html from raw_pages where id = :i"), {"i": first_id}
            ).scalar_one()
        assert count == 1
        assert html == "<p>v2</p>"        # the newer fetch won

    def test_save_parsed_page(self, job) -> None:  # type: ignore[no-untyped-def]
        job_id, site = job
        raw_id, _ = ScrapeRepository.save_raw_page(
            job_id=job_id, url=f"https://{site}/p", site=site, depth=0,
            http_status=200, content_type="text/html", headers={},
            raw_html="<html><body>words here</body></html>",
        )
        parsed_id = ScrapeRepository.save_parsed_page(
            raw_page_id=raw_id,
            url=f"https://{site}/p",
            site=site,
            title="A Title",
            description="A description",
            language="en",
            headings=["H1", "H2"],
            clean_text="one two three four",
            structured_data={"price": "10.00"},
            parser_used="beautifulsoup+lxml",
        )
        assert parsed_id > 0

        with get_engine().connect() as c:
            row = c.execute(
                text(
                    "select title, word_count, headings, structured_data->>'price' "
                    "from parsed_pages where id = :i"
                ),
                {"i": parsed_id},
            ).one()
        assert row[0] == "A Title"
        assert row[1] == 4                # word count computed on write
        assert row[2] == ["H1", "H2"]     # JSONB round-trip
        assert row[3] == "10.00"

    def test_reparsing_updates_the_same_row(self, job) -> None:  # type: ignore[no-untyped-def]
        job_id, site = job
        raw_id, _ = ScrapeRepository.save_raw_page(
            job_id=job_id, url=f"https://{site}/r", site=site, depth=0,
            http_status=200, content_type="text/html", headers={}, raw_html="<p>x</p>",
        )
        common = dict(
            raw_page_id=raw_id, url=f"https://{site}/r", site=site,
            description=None, language=None, headings=[], structured_data={},
            parser_used="beautifulsoup+lxml",
        )
        first = ScrapeRepository.save_parsed_page(**common, title="old", clean_text="a")
        second = ScrapeRepository.save_parsed_page(**common, title="new", clean_text="a b")

        assert first == second
        with get_engine().connect() as c:
            row = c.execute(
                text("select title, word_count from parsed_pages where id = :i"),
                {"i": first},
            ).one()
        assert row == ("new", 2)

    def test_save_links_ignores_duplicates(self, job) -> None:  # type: ignore[no-untyped-def]
        _job_id, site = job
        links = [
            {
                "source_url": f"https://{site}/",
                "target_url": f"https://{site}/a",
                "site": site,
                "anchor_text": "A",
                "depth": 1,
                "enqueued": True,
            }
        ]
        assert ScrapeRepository.save_links(links) == 1
        assert ScrapeRepository.save_links(links) == 0   # conflict ignored

    def test_save_links_handles_empty_list(self) -> None:
        assert ScrapeRepository.save_links([]) == 0

    def test_bump_and_finish_job(self, job) -> None:  # type: ignore[no-untyped-def]
        job_id, _site = job
        ScrapeRepository.bump_job_counters(job_id, scraped=3, failed=1)
        ScrapeRepository.finish_job(job_id, status="completed")

        with get_engine().connect() as c:
            row = c.execute(
                text(
                    "select pages_scraped, pages_failed, status, finished_at "
                    "from crawl_jobs where id = :i"
                ),
                {"i": job_id},
            ).one()
        assert row[0] == 3
        assert row[1] == 1
        assert row[2] == "completed"
        assert row[3] is not None

    def test_stats_structure(self) -> None:
        stats = ScrapeRepository.stats()
        assert "sites" in stats
        assert "totals" in stats
        for key in ("raw_pages", "parsed_pages", "discovered_links", "crawl_jobs"):
            assert key in stats["totals"]
