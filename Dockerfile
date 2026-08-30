FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


ENV HF_HOME=/app/huggingface_cache


ENV PYTHONPATH=/app/src

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

t
COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . .


RUN python -c "\
from transformers import AutoProcessor, AutoModelForTokenClassification; \
AutoProcessor.from_pretrained('SamuelParrales/key-value-ner'); \
AutoModelForTokenClassification.from_pretrained('SamuelParrales/key-value-ner') \
"

EXPOSE 9000


CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "9000"]