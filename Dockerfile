# Dockerfile

# 1. Base Image: Use a specific Python version for reproducibility
FROM python:3.10-slim-buster

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # PORT for the Uvicorn server; api_server.py already respects this
    PORT=8000 \
    # HOST for Uvicorn, ensuring it listens on all interfaces within the container
    # This is not strictly needed if using the direct uvicorn CMD below, but good for clarity
    HOST=0.0.0.0 \
    # Indicate running environment for any conditional logic (optional)
    APP_ENV=docker

# 3. Create and Set Workdir
WORKDIR /app

# 4. Install system dependencies
# libgomp1 is often needed by libraries like faiss-cpu or numpy for OpenMP
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt and Install Python Dependencies
# Copying requirements first leverages Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Create directories that config.py might try to validate/create,
# ensuring they exist within the image.
# These will be in /app/knowledge_base_docs and /app/vector_store
RUN mkdir -p ./knowledge_base_docs ./vector_store
COPY . .

# 7. Expose Port that the application runs on
# This should match the PORT environment variable Uvicorn will use
EXPOSE ${PORT}

# 8. CMD (Command to run the application)
# Runs the FastAPI application using Uvicorn.
# The host 0.0.0.0 makes the server accessible from outside the container.
# The port is taken from the ENV PORT variable.
# backend.api_server:app refers to the 'app' FastAPI instance in 'backend/api_server.py'
CMD ["uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "8000"]