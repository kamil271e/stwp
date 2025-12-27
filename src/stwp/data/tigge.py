"""TIGGE (THORPEX Interactive Grand Global Ensemble) data download utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecmwfapi import ECMWFDataServer

# Default TIGGE configuration
DEFAULT_AREA = "55/14/49/25"  # N/W/S/E
DEFAULT_GRID = "0.25/0.25"
DEFAULT_STEPS = "0/to/12/by/6"
DEFAULT_TIMES = "00:00:00/12:00:00"

# Parameter codes for surface variables
# sp=134, u10=165, v10=166, t2m=167, tcc=228164, tp=228228
DEFAULT_PARAMS = "134/165/166/167/228164/228228"


class TIGGEDownloader:
    """Download TIGGE ensemble forecast data from ECMWF."""

    def __init__(self, server: ECMWFDataServer | None = None):
        """Initialize the TIGGE downloader.

        Args:
            server: Optional ECMWF data server. If not provided, creates a new one.
        """
        self.server = server or ECMWFDataServer()

    def download(
        self,
        output_path: str | Path,
        date_range: str,
        *,
        origin: str = "ecmf",
        area: str = DEFAULT_AREA,
        grid: str = DEFAULT_GRID,
        params: str = DEFAULT_PARAMS,
        steps: str = DEFAULT_STEPS,
        times: str = DEFAULT_TIMES,
        forecast_type: str = "fc",
    ) -> Path:
        """Download TIGGE data.

        Args:
            output_path: Path to save the downloaded file
            date_range: Date range in format "YYYY-MM-DD/to/YYYY-MM-DD"
            origin: Data origin (ecmf for ECMWF)
            area: Geographic area "N/W/S/E"
            grid: Grid resolution "lat/lon"
            params: Parameter codes to download
            steps: Forecast steps
            times: Analysis times
            forecast_type: Forecast type (fc=forecast, cf=control forecast)

        Returns:
            Path to the downloaded file
        """
        output_path = Path(output_path)

        request: dict[str, Any] = {
            "class": "ti",
            "dataset": "tigge",
            "date": date_range,
            "expver": "prod",
            "grid": grid,
            "levtype": "sfc",
            "origin": origin,
            "param": params,
            "step": steps,
            "time": times,
            "area": area,
            "type": forecast_type,
            "target": str(output_path),
        }

        self.server.retrieve(request)
        return output_path


def download_tigge(
    output_path: str | Path | None = None,
    date_range: str = "2021-01-01/to/2021-12-31",
) -> Path:
    """Convenience function to download TIGGE data.

    Args:
        output_path: Path to save the downloaded file. If None, uses default naming.
        date_range: Date range in format "YYYY-MM-DD/to/YYYY-MM-DD"

    Returns:
        Path to the downloaded file
    """
    if output_path is None:
        output_path = f"{date_range.replace('/', '-')}_tigge.grib"

    downloader = TIGGEDownloader()
    return downloader.download(output_path, date_range)


if __name__ == "__main__":
    download_tigge()
