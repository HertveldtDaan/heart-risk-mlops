# Build from the repository root:  docker build -f api/Dockerfile -t heart-risk-api .
# (the root is the build context because the API needs the shared src/heartrisk package)
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer stays cached when only code or model change
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared feature contract (same code as the Azure ML components) + API code
COPY src/heartrisk ./heartrisk
COPY api/app ./app
# MLflow model bundle downloaded from the Azure ML registry by the pipeline
COPY model ./model

# Lineage info, shown on GET /version
ARG GIT_COMMIT=local
ARG MODEL_VERSION=dev
ENV GIT_COMMIT=${GIT_COMMIT} \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_DIR=/app/model \
    SQLITE_PATH=/tmp/predictions.db \
    PYTHONUNBUFFERED=1

# Don't run as root inside the container
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
