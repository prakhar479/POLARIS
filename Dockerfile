# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system polaris && adduser --system --ingroup polaris --home /app polaris

COPY pyproject.toml setup.py README.md /app/
COPY requirements /app/requirements
COPY polaris /app/polaris
COPY polaris_cli /app/polaris_cli
COPY config /app/config

RUN python -m pip install --upgrade pip

RUN mkdir -p /app/logs /app/metrics /app/data && chown -R polaris:polaris /app

FROM runtime-base AS core
RUN python -m pip install -c requirements/constraints.txt .

USER polaris

ENV POLARIS_CONFIG=/app/config/default.yaml

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD sh -c "polaris doctor --config ${POLARIS_CONFIG:-/app/config/default.yaml} || exit 1"

ENTRYPOINT ["polaris"]
CMD ["--config", "/app/config/default.yaml"]

FROM runtime-base AS full-feature
RUN python -m pip install -c requirements/constraints.txt .[all]

USER polaris

ENV POLARIS_CONFIG=/app/config/default.yaml

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD sh -c "polaris doctor --config ${POLARIS_CONFIG:-/app/config/default.yaml} || exit 1"

ENTRYPOINT ["polaris"]
CMD ["--config", "/app/config/default.yaml"]

FROM runtime-base AS ci
COPY tests /app/tests
COPY Makefile /app/Makefile
COPY run_tests.py /app/run_tests.py

RUN python -m pip install -c requirements/constraints.txt .[dev]

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["tests/", "-v", "--tb=short"]
