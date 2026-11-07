# GENEIE v2 — Dockerfile
# Build context MUST be the app folder (geneie-v2/), after copying this
# file plus requirements-deploy.txt into it. Build with:
#   docker build -t geneie-v2 .
#   docker run -p 8050:8050 -e PORT=8050 geneie-v2
# (Docker is NOT installed on this VM, so no image was built here;
# the file was validated statically — see DEPLOY.md "Validation".)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Only the two files the app needs. geneie_core.py holds all the analysis.
COPY app.py geneie_core.py ./

EXPOSE 8050

# $PORT is set by Render/Railway/Fly.io; default 8050 for local `docker run`.
# gthread + 1 worker + 4 threads: see Procfile comment for why.
CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:${PORT:-8050} --worker-class gthread --workers 1 --threads 4 --timeout 300"]
