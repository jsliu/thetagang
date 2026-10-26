# thetagang-nt: Python 3.12 + NautilusTrader 1.231 (needs >= 3.12; the trade env is 3.11).
# Code is NOT copied in: docker-compose mounts the repo at /app, so only a change to
# deploy/requirements.txt needs a rebuild (`docker compose build thetagang-nt`).
FROM python:3.12.15-slim

# NautilusTrader first, on its own: the large layer, rebuilt only when this pin changes.
RUN pip install --no-cache-dir "nautilus_trader[ib]==1.231.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
