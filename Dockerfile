# syntax=docker/dockerfile:1
FROM python:3.12-slim

# set display port to avoid crash
ENV DISPLAY=:99

# upgrade pip
RUN python -m pip install pip==26.2.1

# Install from the lockfile: every version is pinned, so pip has nothing to
# resolve. requirements.txt has open pins (mlflow, evidently, langchain…) that
# made pip backtrack for ~40 min. After changing requirements.txt, regenerate
# the lock from a working container:
#   docker compose exec web pip freeze > requirements.lock.txt
# The cache mount (needs BuildKit/buildx) keeps downloaded wheels between builds.
COPY requirements.lock.txt requirements.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.lock.txt

COPY . .

CMD ["/bin/bash", "+x", "/entrypoint.sh"]
