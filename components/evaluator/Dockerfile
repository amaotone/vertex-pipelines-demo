FROM python:3.8-slim as builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install -U pip poetry
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-dev --no-interaction

FROM python:3.8-slim as app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH
COPY src /app/src
ENTRYPOINT [ "python", "src/evaluator.py" ]