"""Weather API routes."""

from fastapi import APIRouter, HTTPException, Query

from stwp.api.services.prediction import get_prediction_service
from stwp.constants import POLAND_LAT_MAX, POLAND_LAT_MIN, POLAND_LNG_MAX, POLAND_LNG_MIN

router = APIRouter(tags=["weather"])


@router.get("/weather")
async def get_weather(
    latitude: float = Query(
        ...,
        description="Latitude of the location",
        ge=POLAND_LAT_MIN,
        le=POLAND_LAT_MAX,
    ),
    longitude: float = Query(
        ...,
        description="Longitude of the location",
        ge=POLAND_LNG_MIN,
        le=POLAND_LNG_MAX,
    ),
) -> dict:
    """Get weather prediction for a specific location.

    Args:
        latitude: Latitude coordinate (must be within Poland bounds)
        longitude: Longitude coordinate (must be within Poland bounds)

    Returns:
        Weather prediction data with timestamps

    Raises:
        HTTPException: If coordinates are outside Poland bounds
    """
    # Additional validation in case Query constraints are bypassed
    if not (POLAND_LAT_MIN <= latitude <= POLAND_LAT_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"Latitude must be between {POLAND_LAT_MIN} and {POLAND_LAT_MAX}",
        )
    if not (POLAND_LNG_MIN <= longitude <= POLAND_LNG_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"Longitude must be between {POLAND_LNG_MIN} and {POLAND_LNG_MAX}",
        )

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
