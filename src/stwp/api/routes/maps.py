"""Maps API routes."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from stwp.api.services.map_generator import get_map_generator
from stwp.api.services.prediction import get_prediction_service

router = APIRouter(tags=["maps"])


@router.get("/maps")
async def get_maps() -> FileResponse:
    """Get weather maps as a ZIP archive.

    Returns:
        ZIP file containing all weather maps
    """
    # Get prediction data (will refresh if needed)
    prediction_service = get_prediction_service()
    json_data = prediction_service.get_prediction_data()

    # Generate maps
    map_generator = get_map_generator()
    map_generator.create_all_maps(json_data)

    # Create and return ZIP
    zip_path = map_generator.create_zip()

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename="maps.zip",
    )
