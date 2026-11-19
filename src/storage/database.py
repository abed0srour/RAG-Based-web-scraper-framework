"""PostgreSQL connection management and the repository used by the pipelines.

Every worker process gets its own SQLAlchemy engine with a small connection
pool, and `pool_pre_ping` is enabled so that a connection dropped by a
restarted database container is detected and replaced instead of raising on
first use - the first piece of fault tolerance in the storage layer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import settings
from src.storage.models import Base, CrawlJob, DiscoveredLink, ParsedPage, RawPage

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def database_url(driver: str = "psycopg") -> str:
    """Build the SQLAlchemy URL from the central settings."""
    return (
        f"postgresql+{driver}://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url(),
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,     # drop and replace stale connections
            pool_recycle=1800,
            future=True,
        )
        logger.info(
            "PostgreSQL engine created for %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema(drop_first: bool = False) -> None:
    """Create (or recreate) every table in the schema."""
    engine = get_engine()
    if drop_first:
        logger.warning("Dropping all existing tables")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Schema ready: %s", ", ".join(sorted(Base.metadata.tables)))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def node_id() -> str:
    """Identify the worker that performed an action (hostname:pid)."""
    return f"{socket.gethostname()}:{os.getpid()}"


class ScrapeRepository:
    """All database writes and reads used by the scraping module."""

    # ---------------- crawl jobs ----------------

    @staticmethod
    def create_job(job_name: str, site: str, seed_urls: list[str]) -> int:
        with session_scope() as s:
            job = CrawlJob(
                job_name=job_name, site=site, seed_urls=seed_urls, status="running"
            )
            s.add(job)
            s.flush()
            return job.id

    @staticmethod
    def finish_job(job_id: int, status: str = "completed") -> None:
        with session_scope() as s:
            job = s.get(CrawlJob, job_id)
            if job is None:
                return
            job.status = status
            job.finished_at = func.now()

    @staticmethod
    def bump_job_counters(job_id: int, scraped: int = 0, failed: int = 0) -> None:
        with session_scope() as s:
            job = s.get(CrawlJob, job_id)
            if job is None:
                return
            job.pages_scraped += scraped
            job.pages_failed += failed

    # ---------------- raw pages ----------------

    @staticmethod
    def save_raw_page(
        *,
        job_id: int | None,
        url: str,
        site: str,
        depth: int,
        http_status: int | None,
        content_type: str | None,
        headers: dict[str, Any],
        raw_html: str,
    ) -> tuple[int, bool]:
        """Insert a raw page.

        Returns ``(raw_page_id, inserted)``. ``inserted`` is False when the URL
        was already stored for this job, which is how the at-least-once
        delivery guarantee of RabbitMQ is made idempotent: a duplicate message
        updates the existing row instead of creating a second one.
        """
        values = {
            "job_id": job_id,
            "url": url,
            "site": site,
            "depth": depth,
            "http_status": http_status,
            "content_type": content_type,
            "headers": headers,
            "raw_html": raw_html,
            "content_length": len(raw_html),
            "content_hash": sha256(raw_html),
            "fetched_by": node_id(),
        }
        with session_scope() as s:
            stmt = (
                pg_insert(RawPage)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_raw_pages_job_url",
                    set_={
                        "raw_html": values["raw_html"],
                        "content_hash": values["content_hash"],
                        "content_length": values["content_length"],
                        "http_status": values["http_status"],
                        "fetched_by": values["fetched_by"],
                    },
                )
                .returning(RawPage.id)
            )
            raw_id = s.execute(stmt).scalar_one()

            # Was this an insert or a conflict update? Compare the row count of
            # matching URLs before/after is unnecessary - checking whether a
            # parsed row already exists is enough for the caller's purposes.
            existing_parsed = s.execute(
                select(func.count()).select_from(ParsedPage).where(
                    ParsedPage.raw_page_id == raw_id
                )
            ).scalar_one()
            return raw_id, existing_parsed == 0

    # ---------------- parsed pages ----------------

    @staticmethod
    def save_parsed_page(
        *,
        raw_page_id: int,
        url: str,
        site: str,
        title: str | None,
        description: str | None,
        language: str | None,
        headings: list[str],
        clean_text: str,
        structured_data: dict[str, Any],
        parser_used: str,
    ) -> int:
        values = {
            "raw_page_id": raw_page_id,
            "url": url,
            "site": site,
            "title": title,
            "description": description,
            "language": language,
            "headings": headings,
            "clean_text": clean_text,
            "word_count": len(clean_text.split()),
            "structured_data": structured_data,
            "parser_used": parser_used,
        }
        with session_scope() as s:
            stmt = (
                pg_insert(ParsedPage)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["raw_page_id"],
                    set_={
                        k: v
                        for k, v in values.items()
                        if k not in ("raw_page_id",)
                    },
                )
                .returning(ParsedPage.id)
            )
            return s.execute(stmt).scalar_one()

    # ---------------- discovered links ----------------

    @staticmethod
    def save_links(
        links: list[dict[str, Any]],
    ) -> int:
        """Bulk-insert discovered links, ignoring ones already recorded."""
        if not links:
            return 0
        with session_scope() as s:
            # RETURNING is required to count the rows that were actually
            # inserted: with ON CONFLICT DO NOTHING the driver reports a
            # rowcount of -1, so the skipped duplicates could not be told
            # apart from the new rows.
            stmt = (
                pg_insert(DiscoveredLink)
                .values(links)
                .on_conflict_do_nothing(constraint="uq_link_source_target")
                .returning(DiscoveredLink.id)
            )
            return len(s.execute(stmt).scalars().all())

    # ---------------- statistics ----------------

    @staticmethod
    def stats() -> dict[str, Any]:
        """Per-site counters, used by the CLI report and later by the API."""
        with session_scope() as s:
            per_site = s.execute(
                select(
                    RawPage.site,
                    func.count(RawPage.id),
                    func.sum(RawPage.content_length),
                )
                .group_by(RawPage.site)
                .order_by(RawPage.site)
            ).all()

            parsed_per_site = dict(
                s.execute(
                    select(ParsedPage.site, func.count(ParsedPage.id)).group_by(
                        ParsedPage.site
                    )
                ).all()
            )
            words_per_site = dict(
                s.execute(
                    select(ParsedPage.site, func.sum(ParsedPage.word_count)).group_by(
                        ParsedPage.site
                    )
                ).all()
            )
            links_per_site = dict(
                s.execute(
                    select(DiscoveredLink.site, func.count(DiscoveredLink.id)).group_by(
                        DiscoveredLink.site
                    )
                ).all()
            )

            return {
                "sites": [
                    {
                        "site": site,
                        "raw_pages": count,
                        "raw_bytes": int(total or 0),
                        "parsed_pages": parsed_per_site.get(site, 0),
                        "total_words": int(words_per_site.get(site, 0) or 0),
                        "discovered_links": links_per_site.get(site, 0),
                    }
                    for site, count, total in per_site
                ],
                "totals": {
                    "raw_pages": s.execute(
                        select(func.count()).select_from(RawPage)
                    ).scalar_one(),
                    "parsed_pages": s.execute(
                        select(func.count()).select_from(ParsedPage)
                    ).scalar_one(),
                    "discovered_links": s.execute(
                        select(func.count()).select_from(DiscoveredLink)
                    ).scalar_one(),
                    "crawl_jobs": s.execute(
                        select(func.count()).select_from(CrawlJob)
                    ).scalar_one(),
                },
            }

    # ---------------- read side (used by the Task 5 API) ----------------
    #
    # Everything below is query-only. It lives in the storage layer rather
    # than in the API routers for the same reason the writes do: the SQL
    # belongs next to the schema that defines it, so a column rename is a
    # one-file change, and the HTTP layer stays a thin translation of
    # requests into calls on this repository.
    #
    # Every listing takes limit/offset and returns a total count alongside
    # the rows: an endpoint that cannot tell a client how much data exists
    # cannot be paged through safely.

    @staticmethod
    def list_jobs(
        *,
        site: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Crawl runs, newest first, with the total number of matches."""
        with session_scope() as s:
            filters = []
            if site:
                filters.append(CrawlJob.site == site)
            if status:
                filters.append(CrawlJob.status == status)

            total = s.execute(
                select(func.count()).select_from(CrawlJob).where(*filters)
            ).scalar_one()

            rows = (
                s.execute(
                    select(CrawlJob)
                    .where(*filters)
                    .order_by(CrawlJob.started_at.desc(), CrawlJob.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )

            return [
                {
                    "id": job.id,
                    "job_name": job.job_name,
                    "site": job.site,
                    "seed_urls": list(job.seed_urls or []),
                    "status": job.status,
                    "pages_scraped": job.pages_scraped,
                    "pages_failed": job.pages_failed,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
                for job in rows
            ], total

    @staticmethod
    def list_raw_pages(
        *,
        site: str | None = None,
        job_id: int | None = None,
        http_status: int | None = None,
        url_contains: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Raw-page metadata, newest first, without the HTML body.

        The body of a single page can be hundreds of kilobytes, so a listing
        returns only the metadata and the content hash; the full HTML is
        fetched one page at a time from the detail endpoint.
        """
        with session_scope() as s:
            filters = []
            if site:
                filters.append(RawPage.site == site)
            if job_id is not None:
                filters.append(RawPage.job_id == job_id)
            if http_status is not None:
                filters.append(RawPage.http_status == http_status)
            if url_contains:
                filters.append(RawPage.url.ilike(f"%{url_contains}%"))

            total = s.execute(
                select(func.count()).select_from(RawPage).where(*filters)
            ).scalar_one()

            rows = (
                s.execute(
                    select(
                        RawPage.id,
                        RawPage.job_id,
                        RawPage.url,
                        RawPage.site,
                        RawPage.depth,
                        RawPage.http_status,
                        RawPage.content_type,
                        RawPage.content_length,
                        RawPage.content_hash,
                        RawPage.fetched_at,
                        RawPage.fetched_by,
                    )
                    .where(*filters)
                    .order_by(RawPage.fetched_at.desc(), RawPage.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )

            return [dict(row) for row in rows], total

    @staticmethod
    def get_raw_page(
        page_id: int,
        *,
        include_html: bool = True,
        max_html_chars: int | None = None,
    ) -> dict[str, Any] | None:
        """One raw page, optionally with its (possibly truncated) HTML body."""
        with session_scope() as s:
            page = s.get(RawPage, page_id)
            if page is None:
                return None

            record: dict[str, Any] = {
                "id": page.id,
                "job_id": page.job_id,
                "url": page.url,
                "site": page.site,
                "depth": page.depth,
                "http_status": page.http_status,
                "content_type": page.content_type,
                "headers": dict(page.headers or {}),
                "content_length": page.content_length,
                "content_hash": page.content_hash,
                "fetched_at": page.fetched_at,
                "fetched_by": page.fetched_by,
                "raw_html": None,
                "html_truncated": False,
            }

            if include_html:
                html = page.raw_html or ""
                if max_html_chars is not None and len(html) > max_html_chars:
                    record["raw_html"] = html[:max_html_chars]
                    record["html_truncated"] = True
                else:
                    record["raw_html"] = html
            return record

    @staticmethod
    def get_raw_page_html(page_id: int) -> tuple[str, str] | None:
        """(html, content_type) for one page, for the raw-HTML endpoint."""
        from sqlalchemy import text as sql_text  # noqa: F401  (kept local)

        with session_scope() as s:
            row = s.execute(
                select(RawPage.raw_html, RawPage.content_type).where(
                    RawPage.id == page_id
                )
            ).first()
            if row is None:
                return None
            return row[0] or "", row[1] or "text/html"

    @staticmethod
    def list_parsed_pages(
        *,
        site: str | None = None,
        min_words: int | None = None,
        limit: int = 20,
        offset: int = 0,
        include_text: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """The parsed (extracted) view of the crawl."""
        with session_scope() as s:
            filters = []
            if site:
                filters.append(ParsedPage.site == site)
            if min_words is not None:
                filters.append(ParsedPage.word_count >= min_words)

            total = s.execute(
                select(func.count()).select_from(ParsedPage).where(*filters)
            ).scalar_one()

            columns = [
                ParsedPage.id,
                ParsedPage.raw_page_id,
                ParsedPage.url,
                ParsedPage.site,
                ParsedPage.title,
                ParsedPage.description,
                ParsedPage.language,
                ParsedPage.headings,
                ParsedPage.word_count,
                ParsedPage.structured_data,
                ParsedPage.parser_used,
                ParsedPage.parsed_at,
            ]
            if include_text:
                columns.append(ParsedPage.clean_text)

            rows = (
                s.execute(
                    select(*columns)
                    .where(*filters)
                    .order_by(ParsedPage.parsed_at.desc(), ParsedPage.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )

            return [dict(row) for row in rows], total

    @staticmethod
    def get_parsed_page(raw_page_id: int) -> dict[str, Any] | None:
        """The parsed view of one raw page, including its clean text."""
        with session_scope() as s:
            row = s.execute(
                select(ParsedPage).where(ParsedPage.raw_page_id == raw_page_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "raw_page_id": row.raw_page_id,
                "url": row.url,
                "site": row.site,
                "title": row.title,
                "description": row.description,
                "language": row.language,
                "headings": list(row.headings or []),
                "clean_text": row.clean_text,
                "word_count": row.word_count,
                "structured_data": dict(row.structured_data or {}),
                "parser_used": row.parser_used,
                "parsed_at": row.parsed_at,
            }

    @staticmethod
    def list_links(
        *,
        site: str | None = None,
        source_url: str | None = None,
        enqueued: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """The frontier audit trail: which page produced which URL."""
        with session_scope() as s:
            filters = []
            if site:
                filters.append(DiscoveredLink.site == site)
            if source_url:
                filters.append(DiscoveredLink.source_url == source_url)
            if enqueued is not None:
                filters.append(DiscoveredLink.enqueued == enqueued)

            total = s.execute(
                select(func.count()).select_from(DiscoveredLink).where(*filters)
            ).scalar_one()

            rows = (
                s.execute(
                    select(
                        DiscoveredLink.id,
                        DiscoveredLink.source_url,
                        DiscoveredLink.target_url,
                        DiscoveredLink.site,
                        DiscoveredLink.anchor_text,
                        DiscoveredLink.depth,
                        DiscoveredLink.enqueued,
                        DiscoveredLink.discovered_at,
                    )
                    .where(*filters)
                    .order_by(
                        DiscoveredLink.discovered_at.desc(), DiscoveredLink.id.desc()
                    )
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )

            return [dict(row) for row in rows], total

    @staticmethod
    def ping() -> bool:
        """True when PostgreSQL answers - used by the API readiness probe."""
        from sqlalchemy import text as sql_text

        try:
            with get_engine().connect() as connection:
                connection.execute(sql_text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - any driver error means "down"
            logger.warning("PostgreSQL ping failed: %s", exc)
            return False
