FROM apache/superset:latest

USER root
COPY requirements-local.txt /app/docker/requirements-local.txt
RUN pip install --no-cache-dir -r /app/docker/requirements-local.txt
USER superset
