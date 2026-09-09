# HARVEST application image
# ==========================
# Runs the dashboard/server, the farm-device simulator and the FIWARE sync
# daemon (one image, three services in docker-compose.yml).
#
# WISEPACK pattern: the image carries *dependencies only*; the repository is
# bind-mounted at /harvest at run time, so code changes never require an image
# rebuild and the image stays generic.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /harvest

# Dependency layers only -- copied individually so the cache survives source edits.
COPY requirements.txt requirements-integrations.txt ./
RUN pip install -r requirements.txt -r requirements-integrations.txt

CMD ["python", "server.py"]
