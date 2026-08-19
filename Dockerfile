ARG PYTHON_VERSION=3.12-slim-bullseye
FROM python:${PYTHON_VERSION}

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y bash ca-certificates curl git libexpat1 openssh-client ripgrep && \
    rm -rf /var/lib/apt/lists/*
RUN git config --system --add safe.directory '*'
RUN git config --system core.quotepath false

COPY . /src
RUN pip install --no-cache-dir /src