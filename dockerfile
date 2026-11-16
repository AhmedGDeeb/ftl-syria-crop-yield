FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Keep the original source layout
COPY src/ ./src/

COPY config.yaml .
COPY models/ /models/
COPY data/interim/train.csv ./data/interim/train.csv

# Make /app importable so `src.deploy.predict` resolves
ENV PYTHONPATH=/app
ENV MODELS_DIR=/models

EXPOSE 9696
CMD ["uvicorn", "src.deploy.predict:app", "--host", "0.0.0.0", "--port", "9696"]