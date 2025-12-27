"""Weather API routes."""

from fastapi import APIRouter, Query

from stwp.api.services.prediction import get_prediction_service

router = APIRouter(tags=["weather"])


@router.get("/weather")
async def get_weather(
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
) -> dict:
    """Get weather prediction for a specific location.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Weather prediction data with timestamps
    """
    service = get_prediction_service()
    return service.get_weather_by_location(latitude, longitude)


@router.get("/info")
async def get_info() -> dict:
    """Get API status information.

    Returns:
        Dictionary with current data timestamp and status
    """
    service = get_prediction_service()
    last_update = service.get_last_update()

    return {
        "current_data_from": last_update.isoformat() if last_update else None,
        "status": "operational",
    }
