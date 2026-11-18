"""PostgreSQL schema for the scraped data.

Four tables model the crawl:

``crawl_jobs``
    One row per crawl run, so every stored page can be traced back to the
    run that produced it.

``raw_pages``
    The *raw* HTTP response exactly as fetched: status code, headers and the
    unmodified HTML body. Keeping the raw payload is deliberate - the parsing
    rules will change as the project evolves, and re-parsing stored HTML is
    far cheaper (and kinder to the target site) than re-crawling it.

``parsed_pages``
    The result of the BeautifulSoup/lxml parsing stage: title, cleaned main
    text, metadata and word count. One row per raw page.

``discovered_links``
    Every link found on a page. This is the audit trail of the dynamic URL
    frontier: it records which page a URL came from and whether it was
    published back to RabbitMQ.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every table in the scraped-data schema."""


class CrawlJob(Base):
    """A single crawl run."""

    __tablename__ = "crawl_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    site: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    seed_urls: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    pages_scraped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    raw_pages: Mapped[list["RawPage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<CrawlJob id={self.id} site={self.site!r} status={self.status!r}>"


class RawPage(Base):
    """The unmodified HTTP response for one URL."""

    __tablename__ = "raw_pages"
    __table_args__ = (
        # The same URL must not be stored twice for the same job; this is the
        # database-level guarantee behind at-least-once message delivery,
        # which can hand the same URL to a worker more than once.
        UniqueConstraint("job_id", "url", name="uq_raw_pages_job_url"),
        Index("ix_raw_pages_site_fetched", "site", "fetched_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("crawl_jobs.id", ondelete="CASCADE"), nullable=True
    )

    url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    site: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    headers: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    raw_html: Mapped[str] = mapped_column(Text, nullable=False)
    content_length: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # SHA-256 of the body: lets us detect unchanged pages without comparing
    # megabytes of HTML.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    # Which worker/node fetched it - the evidence that the crawl was distributed.
    fetched_by: Mapped[str | None] = mapped_column(String(255), nullable=True)

    job: Mapped["CrawlJob | None"] = relationship(back_populates="raw_pages")
    parsed: Mapped["ParsedPage | None"] = relationship(
        back_populates="raw_page", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RawPage id={self.id} url={self.url!r} status={self.http_status}>"


class ParsedPage(Base):
    """Structured content extracted from a raw page."""

    __tablename__ = "parsed_pages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    raw_page_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("raw_pages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    site: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    headings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    clean_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Site-specific structured fields (price, author, rating, ...) extracted by
    # the per-site extractor. JSONB keeps the schema flexible across sites.
    structured_data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    parser_used: Mapped[str] = mapped_column(String(64), nullable=False, default="lxml")

    parsed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    raw_page: Mapped["RawPage"] = relationship(back_populates="parsed")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ParsedPage id={self.id} title={self.title!r} words={self.word_count}>"


class DiscoveredLink(Base):
    """A link found on a crawled page (the frontier audit trail)."""

    __tablename__ = "discovered_links"
    __table_args__ = (
        UniqueConstraint("source_url", "target_url", name="uq_link_source_target"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    target_url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    site: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    anchor_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # True once the URL has been published to the RabbitMQ frontier queue.
    enqueued: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DiscoveredLink {self.source_url!r} -> {self.target_url!r}>"
