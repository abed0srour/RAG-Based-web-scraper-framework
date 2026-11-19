"""The LLM layer: two providers behind one interface.

The augmentation stage needs a language model to turn retrieved passages into
a contextual summary. This module provides two implementations of that
operation and picks between them at runtime:

:class:`ExtractiveLLM` — **the free, offline provider, and the default.**
    No API key, no network, no cost. It scores the sentences of the retrieved
    passages against the query (term overlap, position, length, and the rank
    of the passage the sentence came from) and returns the best ones in their
    original order. This is *extractive* summarisation: every word of the
    output is a word that appears in the corpus, so the summary cannot state
    anything the sources do not. That property is the reason it is not merely
    a stand-in - it is the baseline the generated summaries are compared
    against, and it is what keeps the whole pipeline runnable, testable and
    reproducible on a machine with no credentials.

:class:`AnthropicLLM` — **Claude, used when a key is configured.**
    Sends the cited context block to the Messages API and asks for a short
    abstractive summary that stays inside the context and marks each claim
    with the ``[n]`` of the passage it came from. This reads better, resolves
    pronouns and merges facts spread across several pages - things an
    extractive summariser cannot do - at the cost of an API call per document.

``LLM_PROVIDER`` selects between them: ``extractive`` (never spend money),
``anthropic`` (always use Claude, fail loudly without a key), or ``auto`` (the
default - Claude when ``ANTHROPIC_API_KEY`` is set, extractive otherwise).

**Grounding is enforced on both paths.** The prompt tells Claude to use only
the supplied context and to say so when the context does not answer the
question; the extractive path cannot do otherwise by construction. Every
response records which provider produced it, so a summary stored in MongoDB
always says where it came from.

**SDK compatibility.** The pinned ``anthropic`` release predates the newer
request fields (``output_config.effort``), so those are sent through
``extra_body``, which the SDK forwards verbatim, and a request rejected
because of them is retried once without them. That keeps one code path
working across SDK versions instead of pinning the project to one.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.common.config import settings

logger = logging.getLogger(__name__)

PROVIDERS = ("auto", "anthropic", "extractive")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have he her his
    how i in into is it its of on or she that the their them there these they
    this to was were what when where which who why will with you your about
    """.split()
)

# How strongly a sentence's own passage rank discounts its score: a sentence
# from the top-ranked passage keeps its full score, one from the fifth is
# worth about two thirds.
_RANK_DECAY = 0.12

SYSTEM_PROMPT = (
    "You summarise web pages for a search index. You are given numbered "
    "passages that were retrieved from a crawled corpus, and you answer only "
    "from them.\n"
    "Rules:\n"
    "1. Use only facts stated in the passages. Never add outside knowledge.\n"
    "2. Mark every claim with the number of the passage it came from, like "
    "[2].\n"
    "3. If the passages do not contain the answer, say exactly what is "
    "missing instead of guessing.\n"
    "4. Write plain, specific prose. No preamble, no bullet padding, no "
    "repetition of the question."
)


class LLMError(RuntimeError):
    """Raised when the configured provider cannot produce a response."""


@dataclass
class LLMResponse:
    """One generated (or extracted) piece of text, plus its provenance."""

    text: str
    provider: str
    model: str
    duration_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    sentences: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True when the output can only contain source text.

        The extractive provider copies sentences, so its output is grounded by
        construction. Claude's output is instructed to be grounded, which is
        a weaker guarantee and is reported as such.
        """
        return self.provider == "extractive"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "duration_ms": round(self.duration_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "sentences": self.sentences,
            "grounded_by_construction": self.grounded,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences, conservatively.

    Scraped text is full of abbreviations and prices, so the split is made
    only after terminal punctuation followed by whitespace and a capital or a
    digit. Over-splitting would produce fragments; under-splitting only costs
    a slightly longer summary sentence.
    """
    sentences: list[str] = []
    for block in (text or "").split("\n"):
        block = block.strip()
        if not block:
            continue
        for sentence in _SENTENCE_RE.split(block):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def content_words(text: str) -> list[str]:
    return [
        word
        for word in _WORD_RE.findall((text or "").lower())
        if word not in _STOPWORDS and len(word) > 1
    ]


class BaseLLM:
    """The interface the augmentation stage programs against."""

    name = "base"

    @property
    def model(self) -> str:  # pragma: no cover - overridden
        return "unknown"

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def summarise(
        self,
        query: str,
        context: str,
        *,
        passages: Sequence[Any] = (),
        max_sentences: int = 4,
        instruction: str | None = None,
        max_passage_rank: int | None = None,
    ) -> LLMResponse:
        """Summarise ``context`` with respect to ``query``.

        ``max_passage_rank`` restricts which passages may be *quoted from*
        (the first ``n`` of the context block); later passages stay as
        background. The extractive provider enforces it mechanically, since
        it selects sentences; Claude is told about it in the instruction.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": self.available()}


# ---------------------------------------------------------------------------
# The free provider
# ---------------------------------------------------------------------------


class ExtractiveLLM(BaseLLM):
    """Query-focused extractive summarisation. Local, free, deterministic."""

    name = "extractive"

    @property
    def model(self) -> str:
        return "extractive-sentence-ranker/1.0"

    def available(self) -> bool:
        return True

    def summarise(
        self,
        query: str,
        context: str,
        *,
        passages: Sequence[Any] = (),
        max_sentences: int = 4,
        instruction: str | None = None,
        max_passage_rank: int | None = None,
    ) -> LLMResponse:
        """Return the sentences of ``context`` that best answer ``query``.

        ``passages`` is accepted (and used for the per-passage rank discount)
        so that the two providers are interchangeable at the call site.
        ``max_passage_rank`` is enforced by simply not considering sentences
        from the later passages.
        """
        started = time.perf_counter()
        terms = set(content_words(query))

        # Sentences are collected per passage so that a sentence can be
        # discounted by the rank of the passage it came from.
        candidates: list[tuple[float, int, int, str]] = []
        blocks = _split_context_blocks(context) if context else []
        if not blocks and passages:
            blocks = [str(getattr(chunk, "text", "")) for chunk in passages]

        for passage_rank, block in enumerate(blocks, start=1):
            if max_passage_rank is not None and passage_rank > max_passage_rank:
                break
            sentences = split_sentences(block)
            for position, sentence in enumerate(sentences):
                words = content_words(sentence)
                if len(words) < 3:
                    continue

                overlap = sum(1 for word in set(words) if word in terms)
                coverage = (overlap / len(terms)) if terms else 0.0
                density = overlap / len(words)
                # A lead sentence usually carries the passage's topic.
                lead = 1.0 / (1.0 + position)
                # Very short and very long sentences both summarise badly.
                length_fit = min(len(words), 40) / 40.0 if len(words) >= 6 else 0.3

                score = (
                    (2.5 * coverage)
                    + (1.5 * density)
                    + (0.8 * lead)
                    + (0.4 * length_fit)
                ) * (1.0 / (1.0 + _RANK_DECAY * (passage_rank - 1)))

                candidates.append((score, passage_rank, position, sentence))

        if not candidates:
            return LLMResponse(
                text="",
                provider=self.name,
                model=self.model,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                notes=["no usable sentence in the retrieved context"],
            )

        best = sorted(candidates, key=lambda row: row[0], reverse=True)

        chosen: list[tuple[int, int, str]] = []
        seen: list[set[str]] = []
        for _score, passage_rank, position, sentence in best:
            words = set(content_words(sentence))
            # Scraped corpora repeat boilerplate across pages, so a sentence
            # that mostly duplicates one already chosen is dropped.
            if any(_overlap_ratio(words, earlier) > 0.6 for earlier in seen):
                continue
            chosen.append((passage_rank, position, sentence))
            seen.append(words)
            if len(chosen) >= max_sentences:
                break

        # Restore reading order: passage order, then position in the passage.
        chosen.sort(key=lambda row: (row[0], row[1]))
        text = " ".join(f"{sentence} [{rank}]" for rank, _position, sentence in chosen)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            sentences=len(chosen),
        )


def _split_context_blocks(context: str) -> list[str]:
    """Recover the individual passages from a rendered context block.

    :meth:`src.rag.retriever.RetrievalResult.context` writes each passage as
    ``[n] title`` / ``URL: ...`` / body, so the headers are stripped back off
    here and only the bodies are summarised.
    """
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n(?=\[\d+\]|\S)", context):
        lines = [line for line in raw.splitlines() if line.strip()]
        body = [
            line
            for line in lines
            if not re.match(r"^\[\d+\]\s", line) and not line.startswith("URL: ")
        ]
        if body:
            blocks.append("\n".join(body))
    return blocks


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


# ---------------------------------------------------------------------------
# The Claude provider
# ---------------------------------------------------------------------------


class AnthropicLLM(BaseLLM):
    """Abstractive summarisation with Claude through the Messages API."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self._model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.effort = effort or settings.llm_effort
        self._client: Any | None = None

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise LLMError(
                    "the anthropic package is not installed; run "
                    "'pip install -r requirements.txt'"
                ) from exc
            if not self.api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Set it in .env, or use "
                    "LLM_PROVIDER=extractive to run the free offline summariser."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _create(self, system: str, prompt: str, *, max_tokens: int) -> Any:
        """One Messages API call, tolerant of the installed SDK's age.

        ``output_config.effort`` is a newer request field than the pinned SDK
        knows about, so it is passed through ``extra_body``; if the API
        rejects it the call is retried once without it rather than failing.
        """
        import anthropic

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        extras = {"output_config": {"effort": self.effort}} if self.effort else {}

        try:
            return self.client.messages.create(**request, extra_body=extras)
        except anthropic.BadRequestError:
            if not extras:
                raise
            logger.debug("Retrying without output_config (not accepted by this API)")
            return self.client.messages.create(**request)

    def summarise(
        self,
        query: str,
        context: str,
        *,
        passages: Sequence[Any] = (),
        max_sentences: int = 4,
        instruction: str | None = None,
        max_passage_rank: int | None = None,
    ) -> LLMResponse:
        """Ask Claude for a grounded summary of the retrieved passages."""
        import anthropic

        task = instruction or (
            f"Write at most {max_sentences} sentences that answer the request "
            "using only the passages below."
        )
        if max_passage_rank:
            task = (
                f"{task}\n\nOnly passages [1] to [{max_passage_rank}] may be "
                "quoted or described as the subject. Any later passage is "
                "background context about neighbouring pages and must not be "
                "presented as part of the subject."
            )
        prompt = (
            f"{task}\n\n"
            f"Request: {query}\n\n"
            f"Retrieved passages:\n{context}\n"
        )

        started = time.perf_counter()
        try:
            response = self._create(SYSTEM_PROMPT, prompt, max_tokens=self.max_tokens)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude request failed ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API: {exc}") from exc
        elapsed = (time.perf_counter() - started) * 1000.0

        notes: list[str] = []
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            notes.append(f"refused: {getattr(details, 'category', 'unknown')}")

        text = "\n".join(
            block.text for block in getattr(response, "content", []) if block.type == "text"
        ).strip()
        if getattr(response, "stop_reason", None) == "max_tokens":
            notes.append("output truncated at max_tokens")

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=getattr(response, "model", self.model),
            duration_ms=elapsed,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            sentences=len(split_sentences(text)),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class FallbackLLM(BaseLLM):
    """Claude with the extractive summariser behind it.

    This is what ``LLM_PROVIDER=auto`` builds when a key is present. A
    transient API failure then degrades the *quality* of one summary instead
    of failing the run, and the response says which provider produced it, so
    a degraded summary is never mistaken for a generated one.
    """

    name = "auto"

    def __init__(self, primary: BaseLLM, fallback: BaseLLM) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def model(self) -> str:
        return self.primary.model

    def available(self) -> bool:
        return self.primary.available() or self.fallback.available()

    def summarise(self, query: str, context: str, **kwargs: Any) -> LLMResponse:
        try:
            return self.primary.summarise(query, context, **kwargs)
        except LLMError as exc:
            logger.warning("Falling back to the extractive summariser: %s", exc)
            response = self.fallback.summarise(query, context, **kwargs)
            response.notes.append(f"{self.primary.name} unavailable: {exc}")
            return response

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.primary.model,
            "primary": self.primary.describe(),
            "fallback": self.fallback.describe(),
            "available": self.available(),
        }


def get_llm(provider: str | None = None) -> BaseLLM:
    """Build the configured provider.

    ``auto`` resolves to Claude-with-extractive-fallback when a key is
    configured, and to the extractive summariser alone when it is not, so the
    pipeline has a working LLM stage in both cases.
    """
    choice = (provider or settings.llm_provider or "auto").strip().lower()
    if choice not in PROVIDERS:
        raise LLMError(f"unknown LLM_PROVIDER {choice!r}; expected one of {PROVIDERS}")

    if choice == "extractive":
        return ExtractiveLLM()
    if choice == "anthropic":
        return AnthropicLLM()

    claude = AnthropicLLM()
    if claude.available():
        return FallbackLLM(claude, ExtractiveLLM())

    logger.info("No ANTHROPIC_API_KEY set - using the free extractive summariser.")
    return ExtractiveLLM()
