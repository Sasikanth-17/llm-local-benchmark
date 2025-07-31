# Use official Python slim image for smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY benchmark.py .
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV TRANSFORMERS_NO_SYMLINKS_WARNING=1

# Run benchmark script
# HF_TOKEN must be passed via docker run -e HF_TOKEN=your_token
CMD ["python", "benchmark.py", "--keep-cache"]
