FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy CDS API credentials
COPY api/.cdsapirc /root/

# Set timezone
ENV TZ=Europe/Warsaw

# Expose port
EXPOSE 8888

# Run the API
CMD ["uv", "run", "python", "-m", "stwp.api.main"]
