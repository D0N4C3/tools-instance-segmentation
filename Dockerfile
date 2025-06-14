# Use official Python slim image
FROM python:3.10-slim

# Stop Python buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code & the model
COPY app ./app
COPY best.pt .

# Expose (for documentation only—Fly will map the real port)
EXPOSE 8080

# Entrypoint: bind to 0.0.0.0, pick up $PORT at runtime (default to 8080 if unset).
# No --workers so uvicorn runs in-process and actually binds.
ENTRYPOINT ["sh","-c","exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${PORT:-8080} \
  --log-level info"]
