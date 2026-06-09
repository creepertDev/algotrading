FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/       bot/
COPY main.py    .
COPY config/    config/

RUN mkdir -p logs

CMD ["python", "main.py", "--config", "config/config.yaml"]
