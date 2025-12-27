"""Data download utilities for ERA5 reanalysis data."""

from pathlib import Path
from typing import Any

import cdsapi

# Geographic area definitions
BIG_AREA: list[float] = [55.75, 13.25, 48.0, 25.0]  # for NN models
SMALL_AREA: list[float] = [55.0, 14.0, 49.0, 25.0]  # for data processing

# Default query configuration
DEFAULT_DATASET = "reanalysis-era5-single-levels"

DEFAULT_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential",
    "land_sea_mask",
    "surface_pressure",
    "total_cloud_cover",
    "total_precipitation",
]

DEFAULT_YEARS = ["2019", "2020", "2021"]

DEFAULT_MONTHS = [f"{i:02d}" for i in range(1, 13)]

DEFAULT_DAYS = [f"{i:02d}" for i in range(1, 32)]

DEFAULT_TIMES = ["06:00", "18:00"]


class DataImporter:
    """Import ERA5 reanalysis data from Copernicus Climate Data Store."""

    def __init__(self, client: cdsapi.Client | None = None):
        """Initialize the data importer.

        Args:
            client: Optional CDS API client. If not provided, creates a new one.
        """
        self.client = client or cdsapi.Client()

    def download(
        self,
        output_path: str | Path,
        *,
        dataset: str = DEFAULT_DATASET,
        variables: list[str] | None = None,
        years: list[str] | None = None,
        months: list[str] | None = None,
        days: list[str] | None = None,
        times: list[str] | None = None,
        area: list[float] | None = None,
        data_format: str = "grib",
    ) -> Path:
        """Download ERA5 data from CDS.

        Args:
            output_path: Path to save the downloaded file
            dataset: CDS dataset name
            variables: List of variables to download
            years: List of years
            months: List of months
            days: List of days
            times: List of times
            area: Geographic area [north, west, south, east]
            data_format: Output format (grib or netcdf)

        Returns:
            Path to the downloaded file
        """
        output_path = Path(output_path)

        query: dict[str, Any] = {
            "product_type": "reanalysis",
            "format": data_format,
            "variable": variables or DEFAULT_VARIABLES,
            "year": years or DEFAULT_YEARS,
            "month": months or DEFAULT_MONTHS,
            "day": days or DEFAULT_DAYS,
            "time": times or DEFAULT_TIMES,
            "area": area or SMALL_AREA,
        }

        self.client.retrieve(dataset, query, str(output_path))
        return output_path


def download_data(
    output_path: str | Path = "data.grib",
    area: list[float] | None = None,
) -> Path:
    """Convenience function to download ERA5 data.

    Args:
        output_path: Path to save the downloaded file
        area: Geographic area [north, west, south, east]

    Returns:
        Path to the downloaded file
    """
    importer = DataImporter()
    return importer.download(output_path, area=area or SMALL_AREA)


if __name__ == "__main__":
    download_data("../data2019-2021_SMALL.grib")
