FROM python:3.9-slim

WORKDIR /app

COPY kubeflow /app/kubeflow

COPY kubeflow/requirements.txt .
RUN pip install -r requirements.txt
