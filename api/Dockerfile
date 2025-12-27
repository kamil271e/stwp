FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -u 1000 app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# CDS API credentials: Mount at runtime or provide via environment
# Option 1: Mount file at runtime: -v ~/.cdsapirc:/home/app/.cdsapirc:ro
# Option 2: Set environment variables CDS_API_URL and CDS_API_KEY
# See api/.cdsapirc.example for format

# Set timezone and user
ENV TZ=Europe/Warsaw
USER app

# Expose port
EXPOSE 8888

# Run the API
CMD ["uv", "run", "python", "-m", "stwp.api.main"]
