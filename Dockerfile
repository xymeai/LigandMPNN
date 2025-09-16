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
    && apt-get -qq install -y build-essential python3 python3-pip wget

RUN pip install uv

# Install dependencies using the lockfile and settings
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project --dev

# Build the final image
FROM builder

WORKDIR /app
COPY . /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

