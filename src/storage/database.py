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
