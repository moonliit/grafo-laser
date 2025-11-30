# backend/Dockerfile
FROM python:3.12-slim

# Avoid interactive prompts, ensure pip uses latest resolver
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build dependencies for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to use Docker layer cache for dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the port your app uses
EXPOSE 8000

# Default command: run the server
CMD ["python3", "server.py"]
