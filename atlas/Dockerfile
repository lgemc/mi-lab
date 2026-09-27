# The atlas as one image: the React build is copied into the API image and served by it, so the
# deployment is one Deployment and one Service. Build from this directory:
#   docker build -t mi-lab-atlas:latest atlas/
# Deployed from ~/m/projects/k8s/atlas (see README.md).

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app/api
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --frozen --no-dev
COPY api/atlas atlas
COPY --from=web /web/dist /app/web/dist
ENV ATLAS_STATIC=/app/web/dist
# Unprivileged: the server writes nothing, so it needs no home and no rights.
RUN useradd --system --uid 10001 atlas
USER 10001
EXPOSE 8000
# Two workers: every trainer is lru-cached per process, and a slow first training in one worker
# should not stall a stranger's slider in the other.
CMD ["uvicorn", "atlas.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
