# Stage 1: Build dependencies
FROM python:3.10 AS builder

# Set working directory
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    mv /root/.local/bin/poetry /usr/local/bin/ && \
    poetry self add poetry-plugin-export

# Copy pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* ./

# Update the lock file to match pyproject.toml
RUN poetry lock --no-update

# Export dependencies to requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Application
FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ARG RENDER_EXTERNAL_HOSTNAME
RUN grep -rl "your-app-url.com" . | xargs sed -i "s/your-app-url.com/${RENDER_EXTERNAL_HOSTNAME}/g"

ARG WEAVIATE_HOSTNAME
ENV WEAVIATE_HOST=http://${WEAVIATE_HOSTNAME}

# Expose port and define entrypoint
EXPOSE 8080
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}"]