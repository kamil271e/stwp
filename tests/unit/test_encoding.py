"""Tests for encoding utilities."""

from datetime import datetime

import numpy as np
import pytest

from stwp.utils.encoding import (
    datetime64_to_datetime,
    get_day_of_year,
    trig_decode,
    trig_encode,
)


class TestTrigEncode:
    """Tests for trig_encode function."""

    def test_sin_encode_zero(self) -> None:
        """Test sin encoding at zero."""
        result = trig_encode(0, 24, "sin")
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_sin_encode_quarter(self) -> None:
        """Test sin encoding at quarter period."""
        result = trig_encode(6, 24, "sin")
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cos_encode_zero(self) -> None:
        """Test cos encoding at zero."""
        result = trig_encode(0, 24, "cos")
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_cos_encode_quarter(self) -> None:
        """Test cos encoding at quarter period."""
        result = trig_encode(6, 24, "cos")
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_invalid_trig_func(self) -> None:
        """Test error on invalid trig function."""
        with pytest.raises(ValueError, match="Unknown trig function"):
            trig_encode(0, 24, "tan")

    def test_cyclical_property(self) -> None:
        """Test that encoding is cyclical."""
        norm_v = 365
        for v in [0, 365, 730]:
            sin_result = trig_encode(v, norm_v, "sin")
            cos_result = trig_encode(v, norm_v, "cos")
            assert sin_result == pytest.approx(0.0, abs=1e-10)
            assert cos_result == pytest.approx(1.0, abs=1e-10)


class TestTrigDecode:
    """Tests for trig_decode function."""

    def test_decode_zero(self) -> None:
        """Test decoding at zero."""
        vsin = trig_encode(0, 24, "sin")
        vcos = trig_encode(0, 24, "cos")
        result = trig_decode(vsin, vcos, 24)
        assert result == 0

    def test_decode_quarter(self) -> None:
        """Test decoding at quarter period."""
        vsin = trig_encode(6, 24, "sin")
        vcos = trig_encode(6, 24, "cos")
        result = trig_decode(vsin, vcos, 24)
        assert result == 6

    def test_roundtrip_hours(self) -> None:
        """Test encode/decode roundtrip for hours."""
        norm_v = 24
        for hour in range(24):
            vsin = trig_encode(hour, norm_v, "sin")
            vcos = trig_encode(hour, norm_v, "cos")
            decoded = trig_decode(vsin, vcos, norm_v)
            assert decoded == hour, f"Failed for hour {hour}"

    def test_roundtrip_days(self) -> None:
        """Test encode/decode roundtrip for days of year."""
        norm_v = 365
        for day in [1, 50, 100, 182, 250, 300, 364]:
            vsin = trig_encode(day, norm_v, "sin")
            vcos = trig_encode(day, norm_v, "cos")
            decoded = trig_decode(vsin, vcos, norm_v)
            assert decoded == day, f"Failed for day {day}"


class TestDatetimeConversion:
    """Tests for datetime conversion utilities."""

    def test_datetime64_to_datetime(self) -> None:
        """Test numpy datetime64 to Python datetime conversion."""
        dt64 = np.datetime64("2024-06-15T12:30:00")
        result = datetime64_to_datetime(dt64)

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 30

    def test_get_day_of_year_jan1(self) -> None:
        """Test day of year for January 1st."""
        dt = datetime(2024, 1, 1)
        assert get_day_of_year(dt) == 1

    def test_get_day_of_year_dec31(self) -> None:
        """Test day of year for December 31st (leap year)."""
        dt = datetime(2024, 12, 31)  # 2024 is a leap year
        assert get_day_of_year(dt) == 366

    def test_get_day_of_year_mid_year(self) -> None:
        """Test day of year for mid-year date."""
        dt = datetime(2024, 7, 1)
        # Jan(31) + Feb(29) + Mar(31) + Apr(30) + May(31) + Jun(30) + 1 = 183
        assert get_day_of_year(dt) == 183
