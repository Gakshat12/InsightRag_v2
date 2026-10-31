FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY frontend/ ./frontend/

EXPOSE 8000 8501

# Default entrypoint starts the FastAPI server
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

