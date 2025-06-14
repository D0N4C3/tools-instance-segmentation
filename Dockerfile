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

# Expose the port Fly expects
EXPOSE 8080

# Use exec form so signals are forwarded correctly.
ENTRYPOINT ["sh","-c","exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${PORT} \
  --workers 4 \
  --log-level info"]
