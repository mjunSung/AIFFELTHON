FROM python:3.12.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY config/ ./config/
COPY core/ ./core/
COPY models/ ./models/
COPY pipelines/ ./pipelines/
COPY utils/ ./utils/
COPY data/ ./data/
COPY ui/ ./ui/

EXPOSE 8000

CMD ["sh", "-c", "python make_db.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]