# Dockerfile

# 1. Base Image: Use a specific Python version for reproducibility
FROM python:3.10-slim-buster

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    APP_ENV=docker \
    # Ensure Python can find modules in the /app directory
    PYTHONPATH=/app

# 3. Create and Set Workdir
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt and Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Create directories that config.py might try to validate/create,
# ensuring they exist within the image.
# These are placeholders if not using a volume for these initial paths.
# If a volume is mounted at /data, then KNOWLEDGE_BASE_DIR=/data/kb etc. will use the volume.
RUN mkdir -p ./knowledge_base_docs ./vector_store
COPY . .

# 7. Expose Port that the application runs on
EXPOSE ${PORT}

# 8. CMD (Command to run the application)
# Run backend.api_server as a module from the /app directory
CMD ["python", "-m", "backend.api_server"]