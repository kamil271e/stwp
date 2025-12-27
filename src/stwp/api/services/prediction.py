"""Thread-safe prediction service for weather API."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from stwp.config import Config
from stwp.data.download import DataImporter
from stwp.models.gnn.trainer import Trainer

logger = logging.getLogger(__name__)


class PredictionService:
    """Thread-safe service for managing weather predictions."""

    def __init__(
        self,
        model_path: str | Path = "data/gnn_fh5.pt",
        config: Config | None = None,
    ):
        """Initialize the prediction service.

        Args:
            model_path: Path to the trained model checkpoint
            config: Configuration object (uses default if None)
        """
        self._lock = Lock()
        self._json_data: dict[str, Any] | None = None
        self._last_update: datetime | None = None
        self._model_path = Path(model_path)
        self._config = config or Config
        self._data_importer = DataImporter()

        # Coordinate settings
        self._coord_acc = 0.25
        self._lat_min: float | None = None
        self._lat_max: float | None = None
        self._lng_min: float | None = None
        self._lng_max: float | None = None

    def _should_update(self) -> bool:
        """Check if data should be refreshed.

        Returns:
            True if data is stale or not yet loaded
        """
        if self._json_data is None or self._last_update is None:
            return True

        # Data from 7 days ago due to ERA5 availability lag
        current_date = datetime.now() - timedelta(days=7)
        elapsed = (current_date - self._last_update).total_seconds()

        # Refresh every 6 hours
        return elapsed > 21600

    def _download_data(self) -> None:
        """Download current weather data from ERA5."""
        current_date = datetime.now() - timedelta(days=7)
        previous_day = current_date - timedelta(days=1)

        # Set year range
        if previous_day.year != current_date.year:
            years = [str(previous_day.year), str(current_date.year)]
        else:
            years = [str(current_date.year)]

        # Set month range
        if previous_day.month != current_date.month:
            months = [f"{previous_day.month:02d}", f"{current_date.month:02d}"]
        else:
            months = [f"{current_date.month:02d}"]

        # Set day range
        if previous_day.day > current_date.day:
            days = [f"{current_date.day:02d}", f"{previous_day.day:02d}"]
        else:
            days = [f"{previous_day.day:02d}", f"{current_date.day:02d}"]

        self._data_importer.download(
            output_path=self._config.data_path,
            years=years,
            months=months,
            days=days,
            times=["00:00", "06:00", "12:00", "18:00"],
        )

    def _refresh_data(self) -> None:
        """Download new data and generate predictions."""
        logger.info("Refreshing prediction data...")

        # Download latest data
        self._download_data()

        # Create trainer and load model
        trainer = Trainer(architecture="trans", hidden_dim=32)
        trainer.load_model(str(self._model_path))

        # Determine which sequence to use based on current hour
        current_date = datetime.now() - timedelta(days=7)
        most_recent_hour = (current_date.hour // 6) * 6
        sequence_map = {0: 0, 6: 1, 12: 2, 18: 3}
        which_sequence = sequence_map.get(most_recent_hour, 0)

        # Generate predictions
        self._json_data = trainer.predict_to_json(which_sequence=which_sequence)

        # Update coordinate bounds
        self._update_bounds()

        # Update timestamp
        self._last_update = datetime(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day,
            hour=most_recent_hour,
        )

        logger.info(f"Data refreshed at {self._last_update}")

    def _update_bounds(self) -> None:
        """Update coordinate bounds from loaded data."""
        if self._json_data is None:
            return

        lats = [float(k) for k in self._json_data]
        self._lat_min = min(lats)
        self._lat_max = max(lats)

        first_lat = str(self._lat_min)
        lngs = [float(k) for k in self._json_data[first_lat]]
        self._lng_min = min(lngs)
        self._lng_max = max(lngs)

    def get_prediction_data(self) -> dict[str, Any]:
        """Get current prediction data, refreshing if needed.

        Returns:
            Dictionary with predictions keyed by lat/lon/feature/timestamp
        """
        with self._lock:
            if self._should_update():
                self._refresh_data()
            return self._json_data

    def get_last_update(self) -> datetime | None:
        """Get timestamp of last data update.

        Returns:
            Datetime of last update or None if never updated
        """
        with self._lock:
            return self._last_update

    def _get_fractions(self, lat: float, lng: float) -> tuple[float, float]:
        """Calculate interpolation fractions for a coordinate.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            Tuple of (lat_fraction, lng_fraction)
        """
        latitudes = np.arange(self._lat_max, self._lat_min - self._coord_acc, -self._coord_acc)
        longitudes = np.arange(self._lng_min, self._lng_max + self._coord_acc, self._coord_acc)

        lat_index = int(np.floor((self._lat_max - lat) / self._coord_acc))
        lng_index = int(np.floor((lng - self._lng_min) / self._coord_acc))

        lat_center = latitudes[lat_index]
        lng_center = longitudes[lng_index]

        lat_distance = (lat_center - lat) / self._coord_acc
        lng_distance = (lng - lng_center) / self._coord_acc

        return lat_distance, lng_distance

    def _interpolate_value(
        self,
        array: np.ndarray,
        lat: float,
        lng: float,
    ) -> float:
        """Interpolate a value at a given coordinate.

        Args:
            array: 2D array of values
            lat: Latitude
            lng: Longitude

        Returns:
            Interpolated value
        """
        lat_index = int(np.floor((self._lat_max - lat) / self._coord_acc))
        lng_index = int(np.floor((lng - self._lng_min) / self._coord_acc))

        lat_frac, lng_frac = self._get_fractions(lat, lng)

        # Handle edge cases
        if lat == self._lat_max and lng == self._lng_max:
            return float(array[lat_index, lng_index])
        elif lat == self._lat_max:
            return float(
                (1 - lng_frac) * array[lat_index, lng_index]
                + lng_frac * array[lat_index, lng_index + 1]
            )
        elif lng == self._lng_max:
            return float(
                (1 - lat_frac) * array[lat_index, lng_index]
                + lat_frac * array[lat_index + 1, lng_index]
            )
        else:
            return float(
                (1 - lat_frac) * (1 - lng_frac) * array[lat_index, lng_index]
                + lat_frac * (1 - lng_frac) * array[lat_index, lng_index + 1]
                + (1 - lat_frac) * lng_frac * array[lat_index + 1, lng_index]
                + lat_frac * lng_frac * array[lat_index + 1, lng_index + 1]
            )

    def get_weather_by_location(
        self,
        lat: float,
        lng: float,
    ) -> dict[str, Any]:
        """Get interpolated weather data for a specific location.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            Dictionary with weather data for the location
        """
        with self._lock:
            if self._should_update():
                self._refresh_data()

            json_data = self._json_data

            # Get all timestamps from the data
            first_lat = str(self._lat_min)
            first_lng = str(self._lng_min)
            timestamps = list(json_data[first_lat][first_lng]["t2m"].keys())

            # Build feature arrays
            features = self._build_feature_arrays(json_data)

            # Build response
            response: dict[str, Any] = {"lat": lat, "lng": lng, "timestamps": []}

            for i, timestamp in enumerate(timestamps):
                data: dict[str, Any] = {"timestamp": timestamp, "values": {}}
                for feature_name, feature_array in features.items():
                    value = self._interpolate_value(feature_array[:, :, i], lat, lng)
                    data["values"][feature_name] = value
                response["timestamps"].append(data)

            return response

    def _build_feature_arrays(
        self,
        json_data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Build numpy arrays from JSON data for interpolation.

        Args:
            json_data: Raw prediction data

        Returns:
            Dictionary of feature arrays
        """
        features = {
            "sp": np.array(
                [
                    [
                        [json_data[lat][lng]["sp"][ts] for ts in json_data[lat][lng]["sp"]]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            "tcc": np.array(
                [
                    [
                        [
                            max(0.0, min(1.0, json_data[lat][lng]["tcc"][ts]))
                            for ts in json_data[lat][lng]["tcc"]
                        ]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            "tp": np.array(
                [
                    [
                        [json_data[lat][lng]["tp"][ts] for ts in json_data[lat][lng]["tp"]]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            "u10": np.array(
                [
                    [
                        [json_data[lat][lng]["u10"][ts] * 3.6 for ts in json_data[lat][lng]["u10"]]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            "v10": np.array(
                [
                    [
                        [json_data[lat][lng]["v10"][ts] * 3.6 for ts in json_data[lat][lng]["v10"]]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            "t2m": np.array(
                [
                    [
                        [json_data[lat][lng]["t2m"][ts] for ts in json_data[lat][lng]["t2m"]]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
        }
        return features


# Singleton instance for the API
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create the singleton prediction service instance.

    Returns:
        PredictionService instance
    """
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
