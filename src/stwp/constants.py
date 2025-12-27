"""Centralized constants to eliminate magic numbers throughout the codebase."""

from typing import Final

# =============================================================================
# Geographic Constants
# =============================================================================

# Geographic area bounds for Poland (north, west, south, east)
POLAND_LARGE_AREA: Final[tuple[float, float, float, float]] = (55.75, 13.25, 48.0, 25.0)
POLAND_SMALL_AREA: Final[tuple[float, float, float, float]] = (55.0, 14.0, 49.0, 25.0)

# Coordinate resolution in degrees
COORD_RESOLUTION: Final[float] = 0.25

# Poland coordinate bounds for API validation
POLAND_LAT_MIN: Final[float] = 49.0
POLAND_LAT_MAX: Final[float] = 55.0
POLAND_LNG_MIN: Final[float] = 14.0
POLAND_LNG_MAX: Final[float] = 25.0

# =============================================================================
# Time Constants
# =============================================================================

# Data update interval in seconds (6 hours)
DATA_UPDATE_INTERVAL_SECONDS: Final[int] = 21600

# Hours per data interval
HOURS_PER_INTERVAL: Final[int] = 6

# ERA5 data availability lag in days
ERA5_DATA_LAG_DAYS: Final[int] = 7

# Sequence mapping: hour -> sequence index
HOUR_TO_SEQUENCE_MAP: Final[dict[int, int]] = {0: 0, 6: 1, 12: 2, 18: 3}

# Standard forecast times
FORECAST_TIMES: Final[tuple[str, ...]] = ("00:00", "06:00", "12:00", "18:00")

# =============================================================================
# Unit Conversions
# =============================================================================

# Temperature: Kelvin to Celsius offset
KELVIN_TO_CELSIUS: Final[float] = 273.15

# Pressure: Pascal to hectoPascal divisor
PA_TO_HPA: Final[float] = 100.0

# Precipitation: meters to millimeters multiplier
M_TO_MM: Final[float] = 1000.0

# Wind: m/s to km/h multiplier
MS_TO_KMH: Final[float] = 3.6

# =============================================================================
# Model Architecture Constants
# =============================================================================

# Default input sequence length
DEFAULT_INPUT_SIZE: Final[int] = 5

# Default forecast horizon
DEFAULT_FORECAST_HORIZON: Final[int] = 1

# Default batch size
DEFAULT_BATCH_SIZE: Final[int] = 8

# Default neighborhood radius
DEFAULT_RADIUS: Final[int] = 2

# Default graph cells
DEFAULT_GRAPH_CELLS: Final[int] = 9

# =============================================================================
# API Constants
# =============================================================================

# Default API port
DEFAULT_API_PORT: Final[int] = 8888

# Default API host
DEFAULT_API_HOST: Final[str] = "0.0.0.0"
