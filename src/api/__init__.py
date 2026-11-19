"""The REST API layer (Task 5).

The module is split by concern, one file per responsibility:

``main.py``        the FastAPI application: composition, middleware, errors
``security.py``    API-key authentication and per-endpoint scopes
``rate_limit.py``  per-key fixed-window limits, shared across replicas via Redis
``schemas.py``     the request/response contract (and the OpenAPI schema)
``deps.py``        paging, and the injected handles to the data layer
``metrics.py``     Prometheus series for requests, latency and refusals
``routers/``       the endpoints themselves, grouped by what they expose

Start it with ``python -m src.api`` or
``uvicorn src.api.main:app --port 8080``.
"""
