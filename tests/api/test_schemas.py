"""Tests for API schemas."""

import pytest

from stwp.api.schemas import InfoResponse, TimestampData, WeatherResponse, WeatherValue
from stwp.features import Features


class TestWeatherValueSchema:
    """Tests for WeatherValue schema."""

    def test_all_features_present(self) -> None:
        """Test that WeatherValue has all feature fields."""
        # Get field names from the schema
        field_names = set(WeatherValue.model_fields.keys())

        # All features should be present as fields
        for feature in Features.ALL:
            assert feature in field_names, f"Feature {feature} not in WeatherValue schema"

    def test_field_descriptions_from_features(self) -> None:
        """Test that field descriptions come from Features metadata."""
        schema = WeatherValue.model_json_schema()
        properties = schema.get("properties", {})

        for feature in Features.ALL:
            if feature in properties:
                field_desc = properties[feature].get("description", "")
                expected_desc = Features.get_metadata(feature).description
                assert field_desc == expected_desc

    def test_create_weather_value(self) -> None:
        """Test creating a WeatherValue instance."""
        value = WeatherValue(
            t2m=280.0,
            sp=101325.0,
            tcc=0.5,
            tp=0.0,
            u10=5.0,
            v10=3.0,
        )
        assert value.t2m == 280.0
        assert value.sp == 101325.0
        assert value.tcc == 0.5

    def test_weather_value_validation(self) -> None:
        """Test WeatherValue validation for required fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WeatherValue(t2m=280.0)  # Missing required fields


class TestTimestampDataSchema:
    """Tests for TimestampData schema."""

    def test_create_timestamp_data(self) -> None:
        """Test creating TimestampData instance."""
        value = WeatherValue(
            t2m=280.0,
            sp=101325.0,
            tcc=0.5,
            tp=0.0,
            u10=5.0,
            v10=3.0,
        )
        data = TimestampData(timestamp="2024-01-01T06:00:00", values=value)
        assert data.timestamp == "2024-01-01T06:00:00"
        assert data.values.t2m == 280.0


class TestWeatherResponseSchema:
    """Tests for WeatherResponse schema."""

    def test_create_weather_response(self) -> None:
        """Test creating WeatherResponse instance."""
        value = WeatherValue(
            t2m=280.0,
            sp=101325.0,
            tcc=0.5,
            tp=0.0,
            u10=5.0,
            v10=3.0,
        )
        data = TimestampData(timestamp="2024-01-01T06:00:00", values=value)
        response = WeatherResponse(lat=55.0, lng=14.0, timestamps=[data])

        assert response.lat == 55.0
        assert response.lng == 14.0
        assert len(response.timestamps) == 1


class TestInfoResponseSchema:
    """Tests for InfoResponse schema."""

    def test_create_info_response(self) -> None:
        """Test creating InfoResponse instance."""
        response = InfoResponse(
            current_data_from="2024-01-01T00:00:00",
            status="ok",
        )
        assert response.status == "ok"
        assert response.current_data_from == "2024-01-01T00:00:00"
