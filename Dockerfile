FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (tzdata for ib_insync)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Command to run (overridden by docker-compose)
CMD ["python", "main.py"]
