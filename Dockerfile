FROM python:3.11-slim

WORKDIR /app

# Slimmer runtime deps (no Locust / notebook stack) → faster builds, less RAM.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY models/ ./models/
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh && \
    # Seed copies used on first boot when a Render disk is empty.
    mkdir -p /app/seed/models /app/seed/data && \
    cp -a /app/models/. /app/seed/models/ && \
    cp -a /app/data/. /app/seed/data/

WORKDIR /app/api

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
# Default local paths; overridden on Render when /var/data exists (see entrypoint).
ENV BEAN_DATA_ROOT=/app
ENV BEAN_MODEL_DIR=/app/models
ENV BEAN_TRAIN_DIR=/app/data/train

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
