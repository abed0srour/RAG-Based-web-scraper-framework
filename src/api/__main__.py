"""``python -m src.api`` - run the REST service.

Equivalent to ``uvicorn src.api.main:app``, but it reads the host and port
from the central settings, so the service is started the same way as every
other component of the system.
"""

from __future__ import annotations

from src.api.main import main

if __name__ == "__main__":
    raise SystemExit(main())
