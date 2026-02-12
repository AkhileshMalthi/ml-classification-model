FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Set environment variables for MLflow tracking if needed
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]