
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/raw/ ./data/raw/

CMD ["python", "scripts/train.py"]
