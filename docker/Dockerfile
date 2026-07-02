# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install --no-cache-dir -r /app/docker/requirements.txt

COPY tracks/ /app/tracks/
COPY tools/validate_submission.py /app/tools/validate_submission.py
COPY onnx/models/ /app/onnx/models/
COPY models/cross_encoder/ /app/models/cross_encoder/
COPY config.yaml /app/config.yaml
COPY docker/ /app/docker/

ENV PYTHONPATH=/app \
    REDROB_CPU_ONLY=1 \
    REDROB_IN_CONTAINER=1

RUN chmod +x /app/docker/docker-entrypoint.sh

ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]
