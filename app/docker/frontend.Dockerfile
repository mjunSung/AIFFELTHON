FROM python:3.12.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web/ ./web/

EXPOSE 8501

CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]