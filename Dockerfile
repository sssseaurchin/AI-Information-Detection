ARG BASE_IMAGE=tensorflow/tensorflow:2.20.0
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/cnn

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade --ignore-installed -r /app/requirements.txt

COPY . /app

CMD ["python3", "-m", "src.cnn.main"]
