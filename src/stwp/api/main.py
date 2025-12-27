"""FastAPI application for weather prediction API."""

import logging

from fastapi import FastAPI

from stwp.api.routes import maps, weather
from stwp.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="STWP Weather Prediction API",
    description="Short-term weather prediction API for Poland",
    version="1.0.0",
)

# Include routers
app.include_router(weather.router)
app.include_router(maps.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on startup."""
    logging.info("Starting STWP Weather API...")


def main() -> None:
    """Run the API server."""
    import uvicorn

    config = Config()
    uvicorn.run(
        "stwp.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
