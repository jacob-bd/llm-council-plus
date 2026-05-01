FROM nikolaik/python-nodejs:python3.12-nodejs22-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    FRONTEND_DIST_DIR=/app/frontend/dist

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci

COPY . .
RUN cd frontend && npm run build

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/.venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]
