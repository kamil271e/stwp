"""Tests for API endpoints."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestWeatherEndpoint:
    """Tests for /weather endpoint."""

    @pytest.fixture
    def mock_prediction_service(self, sample_weather_data: dict[str, Any]) -> MagicMock:
        """Create a mock prediction service."""
        service = MagicMock()
        service.get_weather_by_location.return_value = {
            "lat": 52.23,
            "lng": 21.01,
            "timestamps": [
                {
                    "timestamp": "2024-01-01T06:00:00",
                    "values": {
                        "t2m": 280.5,
                        "sp": 101325.0,
                        "tcc": 0.5,
                        "tp": 0.0,
                        "u10": 2.5,
                        "v10": 1.5,
                    },
                }
            ],
        }
        service.get_last_update.return_value = datetime(2024, 1, 1, 0, 0, 0)
        return service

    def test_weather_endpoint(self, mock_prediction_service: MagicMock) -> None:
        """Test weather endpoint returns correct structure."""
        with patch(
            "stwp.api.routes.weather.get_prediction_service",
            return_value=mock_prediction_service,
        ):
            from stwp.api.main import app

            client = TestClient(app)
            response = client.get("/weather?latitude=52.23&longitude=21.01")

            assert response.status_code == 200
            data = response.json()

            assert "lat" in data
            assert "lng" in data
            assert "timestamps" in data
            assert len(data["timestamps"]) > 0

    def test_weather_endpoint_requires_params(self) -> None:
        """Test weather endpoint requires latitude and longitude."""
        from stwp.api.main import app

        with patch(
            "stwp.api.routes.weather.get_prediction_service",
            return_value=MagicMock(),
        ):
            client = TestClient(app)
            response = client.get("/weather")

            assert response.status_code == 422  # Validation error

    def test_info_endpoint(self, mock_prediction_service: MagicMock) -> None:
        """Test info endpoint returns status."""
        with patch(
            "stwp.api.routes.weather.get_prediction_service",
            return_value=mock_prediction_service,
        ):
            from stwp.api.main import app

            client = TestClient(app)
            response = client.get("/info")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert data["status"] == "operational"
            assert "current_data_from" in data


class TestMapsEndpoint:
    """Tests for /maps endpoint."""

    @pytest.fixture
    def mock_services(self, sample_weather_data: dict[str, Any]) -> tuple[MagicMock, MagicMock]:
        """Create mock prediction and map generator services."""
        prediction_service = MagicMock()
        prediction_service.get_prediction_data.return_value = sample_weather_data

        map_generator = MagicMock()
        map_generator.create_all_maps.return_value = "/tmp/maps"
        map_generator.create_zip.return_value = "/tmp/maps.zip"

        return prediction_service, map_generator

    def test_maps_endpoint_calls_services(self, mock_services: tuple[MagicMock, MagicMock]) -> None:
        """Test maps endpoint calls the services correctly."""
        prediction_service, map_generator = mock_services

        # Create a temporary zip file for the response
        import tempfile
        import zipfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr("test.txt", "test content")
            temp_zip_path = f.name

        map_generator.create_zip.return_value = temp_zip_path

        with (
            patch(
                "stwp.api.routes.maps.get_prediction_service",
                return_value=prediction_service,
            ),
            patch(
                "stwp.api.routes.maps.get_map_generator",
                return_value=map_generator,
            ),
        ):
            from stwp.api.main import app

            client = TestClient(app)
            response = client.get("/maps")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"

            prediction_service.get_prediction_data.assert_called_once()
            map_generator.create_all_maps.assert_called_once()
            map_generator.create_zip.assert_called_once()

        # Cleanup
        import os

        os.unlink(temp_zip_path)
