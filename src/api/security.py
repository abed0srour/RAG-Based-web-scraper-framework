"""API-key authentication for the REST layer.

Why API keys and not user accounts: the clients of this service are other
programs - the dashboard, a notebook, another node of the system - not people
with browsers. There is no login form, no session and no cookie to protect,
so a shared secret sent on every request is the whole of what is needed, and
it keeps the service stateless, which is what lets several API replicas sit
behind one load balancer without sharing a session store.

What the mechanism provides:

* **Identity.** Every key has a name, so the access log and the rate-limit
  counters can say *which* client made a request rather than only that one
  was made.
* **Scopes.** A key carries a set of scopes (``raw``, ``documents``,
  ``search``, ``ask``, or ``*`` for all of them) and every endpoint declares
  the one it requires. A key issued for reading the crawl therefore cannot
  spend money on LLM generation, which is the main reason the scopes exist at
  all: ``/v1/ask`` is the one endpoint whose cost per call is unbounded.
* **Constant-time comparison.** The presented key is compared with
  :func:`hmac.compare_digest` against every configured key instead of being
  looked up in a dictionary, so the time taken does not depend on how many
  leading characters were guessed correctly.
* **No secrets in logs.** Only a short SHA-256 fingerprint of a key is ever
  logged, so a leaked log file does not leak credentials.

Keys are configured in one environment variable, ``API_KEYS``::

    API_KEYS=reader:dev-reader-key:raw,documents;admin:secret:*

Entries are separated by ``;``, the three fields by ``:`` and the scopes by
``,``. ``API_AUTH_ENABLED=false`` turns the whole check off for local
development; the service then reports every caller as ``anonymous`` and the
rate limiter falls back to identifying clients by IP address.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from fastapi import Depends, HTTPException, Request, status

from src.common.config import settings

logger = logging.getLogger(__name__)

#: Every scope the service knows about. ``*`` means "all of them".
SCOPES: tuple[str, ...] = ("raw", "documents", "search", "ask", "admin")

WILDCARD = "*"

#: Sent with a 401 so a client knows *how* to authenticate, not only that it
#: failed. Two schemes are accepted: the X-API-Key header and a bearer token.
API_KEY_HEADER = "X-API-Key"


def fingerprint(key: str) -> str:
    """A short, non-reversible identifier for a key, safe to log."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ApiKey:
    """One configured credential."""

    name: str
    secret: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    @property
    def key_id(self) -> str:
        return fingerprint(self.secret)

    def allows(self, scope: str) -> bool:
        return WILDCARD in self.scopes or scope in self.scopes

    def describe(self) -> dict[str, object]:
        """Public description of a key - never includes the secret."""
        return {
            "name": self.name,
            "key_id": self.key_id,
            "scopes": sorted(self.scopes),
        }


@dataclass(frozen=True)
class Principal:
    """The authenticated caller behind one request."""

    name: str
    scopes: frozenset[str]
    key_id: str = ""
    authenticated: bool = True

    def allows(self, scope: str) -> bool:
        return WILDCARD in self.scopes or scope in self.scopes

    @property
    def identity(self) -> str:
        """The rate-limiting identity: the key, not the client address.

        Keying the limiter on the credential rather than on the IP is what
        makes the limit meaningful behind a load balancer or a NAT, where
        every request can arrive from the same address.
        """
        if self.authenticated:
            return f"key:{self.key_id}"
        # With authentication disabled there is no credential to key on, so
        # the client address is the best identity available.
        return f"ip:{self.key_id or 'unknown'}"

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "key_id": self.key_id,
            "scopes": sorted(self.scopes),
            "authenticated": self.authenticated,
        }


def parse_api_keys(spec: str) -> list[ApiKey]:
    """Parse the ``API_KEYS`` specification into credentials.

    Malformed entries are skipped with a warning rather than raising: one
    typo in an environment variable must not stop the whole service from
    starting, and the startup log makes the loaded keys visible by name.
    """
    keys: list[ApiKey] = []
    for raw_entry in (spec or "").split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue

        parts = [part.strip() for part in entry.split(":")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            logger.warning("Ignoring malformed API_KEYS entry %r", entry)
            continue

        name, secret = parts[0], parts[1]
        scope_spec = parts[2] if len(parts) > 2 else ""
        scopes = {
            scope.strip()
            for scope in scope_spec.split(",")
            if scope.strip()
        } or {"raw", "documents", "search"}

        unknown = scopes - set(SCOPES) - {WILDCARD}
        if unknown:
            logger.warning(
                "API key %r declares unknown scope(s): %s",
                name, ", ".join(sorted(unknown)),
            )

        keys.append(ApiKey(name=name, secret=secret, scopes=frozenset(scopes)))
    return keys


@lru_cache(maxsize=4)
def _registry_for(spec: str) -> tuple[ApiKey, ...]:
    return tuple(parse_api_keys(spec))


def registry() -> tuple[ApiKey, ...]:
    """The configured keys, re-parsed whenever the specification changes."""
    return _registry_for(settings.api_keys)


def describe_keys() -> list[dict[str, object]]:
    """Names and scopes of the configured keys, for the startup log."""
    return [key.describe() for key in registry()]


def lookup(presented: str) -> ApiKey | None:
    """Find the key matching ``presented`` in constant time."""
    match: ApiKey | None = None
    for candidate in registry():
        # compare_digest over every candidate, and no early return: the loop
        # always runs to the end so a near-miss is not faster than a miss.
        if hmac.compare_digest(candidate.secret, presented):
            match = candidate
    return match


def presented_key(request: Request) -> str | None:
    """Extract the credential from the request headers.

    Two transports are accepted because clients differ: ``X-API-Key`` is what
    curl and the demo script use, and ``Authorization: Bearer <key>`` is what
    most HTTP libraries and API consoles send by default.
    """
    header = request.headers.get(API_KEY_HEADER)
    if header and header.strip():
        return header.strip()

    authorization = request.headers.get("Authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() in ("bearer", "apikey") and value.strip():
        return value.strip()
    return None


ANONYMOUS_SCOPES = frozenset({WILDCARD})


def authenticate(request: Request) -> Principal:
    """Resolve the caller, or refuse the request.

    Used as a FastAPI dependency; the result is cached per request, so the
    scope check and the rate limiter see the same principal without the
    headers being parsed twice.
    """
    if not settings.api_auth_enabled:
        # Authentication disabled (local development): every scope is granted
        # and the client address stands in for the credential.
        client = request.client.host if request.client else "local"
        return Principal(
            name="anonymous",
            scopes=ANONYMOUS_SCOPES,
            key_id=client,
            authenticated=False,
        )

    credential = presented_key(request)
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "missing API key: send it in the X-API-Key header or as "
                "'Authorization: Bearer <key>'"
            ),
            headers={"WWW-Authenticate": f'ApiKey header="{API_KEY_HEADER}"'},
        )

    key = lookup(credential)
    if key is None:
        logger.warning(
            "Rejected request to %s: unknown API key (fingerprint %s)",
            request.url.path, fingerprint(credential),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": f'ApiKey header="{API_KEY_HEADER}"'},
        )

    return Principal(name=key.name, scopes=key.scopes, key_id=key.key_id)


def anonymous_identity(request: Request) -> str:
    """Rate-limit identity for an unauthenticated caller (health endpoints)."""
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


def require_scope(scope: str):
    """Build a dependency that admits only keys holding ``scope``.

    The returned dependency is what every protected route declares, so the
    permission a route needs is visible in the route definition itself - and
    in the generated OpenAPI description.
    """

    def dependency(
        request: Request, principal: Principal = Depends(authenticate)
    ) -> Principal:
        if not principal.allows(scope):
            logger.warning(
                "Key %r (%s) denied access to %s: missing scope %r",
                principal.name, principal.key_id or "-", request.url.path, scope,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"this API key is not allowed to use {request.url.path}: "
                    f"the {scope!r} scope is required"
                ),
            )
        return principal

    dependency.__name__ = f"require_scope_{scope}"
    return dependency


def scope_summary(scopes: Iterable[str]) -> str:
    return ", ".join(sorted(scopes))
