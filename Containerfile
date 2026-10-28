FROM docker.io/library/postgres:16.4-alpine3.20 AS database

RUN apk add --no-cache --virtual .pgvector-package postgresql-pgvector=0.6.2-r0 \
    && cp /usr/lib/postgresql16/vector.so "$(pg_config --pkglibdir)/vector.so" \
    && cp /usr/share/postgresql16/extension/vector* "$(pg_config --sharedir)/extension/" \
    && apk del .pgvector-package

FROM docker.io/library/python:3.12.11-slim-bookworm AS system

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /bin/uv

FROM system AS builder
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
RUN uv sync --frozen --no-dev --no-editable

FROM system AS runtime
RUN groupadd --gid 10001 cliptrace \
    && useradd --uid 10001 --gid 10001 --create-home cliptrace
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=cliptrace:cliptrace app ./app
COPY --chown=cliptrace:cliptrace alembic ./alembic
COPY --chown=cliptrace:cliptrace alembic.ini pyproject.toml ./
RUN mkdir -p /app/data/uploads /app/data/models && chown -R cliptrace:cliptrace /app/data
ENV PATH="/opt/venv/bin:$PATH"
USER cliptrace
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM system AS test
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project
COPY app ./app
RUN uv sync --frozen --no-editable
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini pyproject.toml ./
COPY tests ./tests
CMD ["pytest"]
