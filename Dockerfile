FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS builder

WORKDIR /app

ENV \
    DEBIAN_FRONTEND=noninteractive \
    MYPY_CACHE_DIR='/tmp/.mypy_cache' \
    PYTHONUNBUFFERED=TRUE \
    PYTHONFAULTHANDLER=1 \
    UV_LINK_MODE=copy

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get -qq update \
    && apt-get -qq install -y build-essential

ADD . /app

# Install dependencies using the lockfile and settings
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --dev

# Build the final image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

COPY --from=builder /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

