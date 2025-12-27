"""Centralized feature definitions to avoid magic strings throughout the codebase."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class FeatureName(str, Enum):
    """Weather feature identifiers."""

    T2M = "t2m"  # Temperature at 2m above ground
    SP = "sp"  # Surface pressure
    TCC = "tcc"  # Total cloud cover
    U10 = "u10"  # 10m U wind component
    V10 = "v10"  # 10m V wind component
    TP = "tp"  # Total precipitation


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata for a weather feature."""

    name: FeatureName
    description: str
    unit: str
    display_unit: str  # Unit used for display (after conversion)


@dataclass(frozen=True)
class Features:
    """Central registry of weather features and their metadata.

    This class provides a single source of truth for feature names,
    avoiding magic strings like 't2m' scattered throughout the code.

    Usage:
        from stwp.features import Features

        # Get feature list as strings
        feature_list = Features.as_list()  # ['t2m', 'sp', 'tcc', 'u10', 'v10', 'tp']

        # Access individual features
        temp_feature = Features.T2M  # 't2m'

        # Get metadata
        temp_meta = Features.get_metadata(Features.T2M)
        print(temp_meta.description)  # 'Temperature at 2m above ground'
    """

    # Feature name constants
    T2M: ClassVar[str] = FeatureName.T2M.value
    SP: ClassVar[str] = FeatureName.SP.value
    TCC: ClassVar[str] = FeatureName.TCC.value
    U10: ClassVar[str] = FeatureName.U10.value
    V10: ClassVar[str] = FeatureName.V10.value
    TP: ClassVar[str] = FeatureName.TP.value

    T2M_IDX: ClassVar[int] = 0
    SP_IDX: ClassVar[int] = 1
    TCC_IDX: ClassVar[int] = 2
    U10_IDX: ClassVar[int] = 3
    V10_IDX: ClassVar[int] = 4
    TP_IDX: ClassVar[int] = 5

    # Ordered list of all features (matches data array ordering)
    ALL: ClassVar[tuple[str, ...]] = (
        FeatureName.T2M.value,
        FeatureName.SP.value,
        FeatureName.TCC.value,
        FeatureName.U10.value,
        FeatureName.V10.value,
        FeatureName.TP.value,
    )

    # Number of features
    COUNT: ClassVar[int] = 6

    # Feature metadata registry
    METADATA: ClassVar[dict[str, FeatureMetadata]] = {
        FeatureName.T2M.value: FeatureMetadata(
            name=FeatureName.T2M,
            description="Temperature at 2m above ground",
            unit="K",
            display_unit="C",
        ),
        FeatureName.SP.value: FeatureMetadata(
            name=FeatureName.SP,
            description="Surface pressure",
            unit="Pa",
            display_unit="hPa",
        ),
        FeatureName.TCC.value: FeatureMetadata(
            name=FeatureName.TCC,
            description="Total cloud cover",
            unit="fraction",
            display_unit="%",
        ),
        FeatureName.U10.value: FeatureMetadata(
            name=FeatureName.U10,
            description="10m U wind component",
            unit="m/s",
            display_unit="km/h",
        ),
        FeatureName.V10.value: FeatureMetadata(
            name=FeatureName.V10,
            description="10m V wind component",
            unit="m/s",
            display_unit="km/h",
        ),
        FeatureName.TP.value: FeatureMetadata(
            name=FeatureName.TP,
            description="Total precipitation",
            unit="m",
            display_unit="mm",
        ),
    }

    # Visualization settings
    COLORMAPS: ClassVar[dict[str, str]] = {
        FeatureName.TCC.value: "Greens",
        FeatureName.T2M.value: "coolwarm",
    }

    LABELS: ClassVar[dict[str, str]] = {
        FeatureName.TP.value: "[mm]",
        FeatureName.TCC.value: "[%]",
        FeatureName.T2M.value: "[C]",
    }

    @classmethod
    def as_list(cls) -> list[str]:
        """Return feature names as a list (for compatibility with existing code)."""
        return list(cls.ALL)

    @classmethod
    def get_metadata(cls, feature: str) -> FeatureMetadata:
        """Get metadata for a specific feature."""
        return cls.METADATA[feature]

    @classmethod
    def get_index(cls, feature: str) -> int:
        """Get the index of a feature in the standard ordering."""
        return cls.ALL.index(feature)

    @classmethod
    def validate(cls, feature: str) -> bool:
        """Check if a string is a valid feature name."""
        return feature in cls.ALL
