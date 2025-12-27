"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field

from stwp.features import Features


class WeatherValue(BaseModel):
    """Weather values for a single timestamp."""

    sp: float = Field(..., description=Features.get_metadata(Features.SP).description)
    tcc: float = Field(..., description=Features.get_metadata(Features.TCC).description)
    tp: float = Field(..., description=Features.get_metadata(Features.TP).description)
    u10: float = Field(..., description=Features.get_metadata(Features.U10).description)
    v10: float = Field(..., description=Features.get_metadata(Features.V10).description)
    t2m: float = Field(..., description=Features.get_metadata(Features.T2M).description)


class TimestampData(BaseModel):
    """Weather data for a single timestamp."""

    timestamp: str
    values: WeatherValue


class WeatherResponse(BaseModel):
    """Response for weather endpoint."""

    lat: float
    lng: float
    timestamps: list[TimestampData]


class InfoResponse(BaseModel):
    """Response for info endpoint."""

    current_data_from: str
    status: str
