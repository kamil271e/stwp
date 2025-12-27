"""Trigonometric and temporal encoding utilities."""

from datetime import datetime
from enum import StrEnum

import numpy as np


class TrigFunc(StrEnum):
    SIN = "sin"
    COS = "cos"


def trig_encode(v: float, norm_v: float, trig_func: TrigFunc = TrigFunc.SIN) -> float:
    """Encode a value using trigonometric function for cyclical features.

    Args:
        v: Value to encode
        norm_v: Normalization value (period of the cycle)
        trig_func: Either "sin" or "cos"

    Returns:
        Encoded value in range [-1, 1]
    """
    if trig_func == TrigFunc.SIN:
        return float(np.sin(2 * np.pi * v / norm_v))
    elif trig_func == TrigFunc.COS:
        return float(np.cos(2 * np.pi * v / norm_v))
    else:
        raise ValueError(f"Unknown trig function: {trig_func}. Use 'sin' or 'cos'.")


def trig_decode(vsin: float, vcos: float, norm_v: float) -> int:
    """Decode trigonometric-encoded values back to original value.

    Args:
        vsin: Sin-encoded value
        vcos: Cos-encoded value
        norm_v: Normalization value (period of the cycle)

    Returns:
        Decoded integer value
    """
    varcsin = np.arcsin(vsin)
    if varcsin < 0:
        va = np.array([np.pi - varcsin, 2 * np.pi + varcsin])
    else:
        va = np.array([varcsin, np.pi - varcsin])
    varccos = np.arccos(vcos)
    vb = np.array([varccos, 2 * np.pi - varccos])
    va = np.round(va, 5)
    vb = np.round(vb, 5)
    v = np.intersect1d(va, vb)[0]
    return int(np.round(v * norm_v / (2 * np.pi), 0))


def datetime64_to_datetime(datetime64: np.datetime64) -> datetime:
    """Convert numpy datetime64 to Python datetime.

    Args:
        datetime64: NumPy datetime64 object

    Returns:
        Python datetime object
    """
    timestamp = (datetime64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(timestamp)


def get_day_of_year(dt: datetime) -> int:
    """Get day of year from datetime.

    Args:
        dt: Python datetime object

    Returns:
        Day of year (1-366)
    """
    return dt.timetuple().tm_yday
