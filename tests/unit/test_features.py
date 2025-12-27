"""Tests for the centralized Features class."""

import pytest

from stwp.features import FeatureMetadata, FeatureName, Features


class TestFeatureName:
    """Tests for FeatureName enum."""

    def test_enum_values(self) -> None:
        """Test that enum values are correct."""
        assert FeatureName.T2M.value == "t2m"
        assert FeatureName.SP.value == "sp"
        assert FeatureName.TCC.value == "tcc"
        assert FeatureName.U10.value == "u10"
        assert FeatureName.V10.value == "v10"
        assert FeatureName.TP.value == "tp"

    def test_enum_is_string(self) -> None:
        """Test that FeatureName values work as strings."""
        assert FeatureName.T2M == "t2m"
        assert FeatureName.SP == "sp"


class TestFeatures:
    """Tests for Features class."""

    def test_feature_constants(self) -> None:
        """Test feature name constants."""
        assert Features.T2M == "t2m"
        assert Features.SP == "sp"
        assert Features.TCC == "tcc"
        assert Features.U10 == "u10"
        assert Features.V10 == "v10"
        assert Features.TP == "tp"

    def test_all_features_tuple(self) -> None:
        """Test ALL tuple contains all features in correct order."""
        expected = ("t2m", "sp", "tcc", "u10", "v10", "tp")
        assert expected == Features.ALL

    def test_feature_count(self) -> None:
        """Test COUNT is correct."""
        assert Features.COUNT == 6
        assert len(Features.ALL) == Features.COUNT

    def test_as_list(self) -> None:
        """Test as_list returns a list of feature names."""
        feature_list = Features.as_list()
        assert isinstance(feature_list, list)
        assert feature_list == ["t2m", "sp", "tcc", "u10", "v10", "tp"]

    def test_get_metadata(self) -> None:
        """Test get_metadata returns correct metadata."""
        t2m_meta = Features.get_metadata(Features.T2M)
        assert isinstance(t2m_meta, FeatureMetadata)
        assert t2m_meta.name == FeatureName.T2M
        assert t2m_meta.description == "Temperature at 2m above ground"
        assert t2m_meta.unit == "K"
        assert t2m_meta.display_unit == "C"

    def test_get_metadata_all_features(self) -> None:
        """Test metadata exists for all features."""
        for feature in Features.ALL:
            meta = Features.get_metadata(feature)
            assert meta is not None
            assert meta.description != ""
            assert meta.unit != ""

    def test_get_index(self) -> None:
        """Test get_index returns correct indices."""
        assert Features.get_index(Features.T2M) == 0
        assert Features.get_index(Features.SP) == 1
        assert Features.get_index(Features.TCC) == 2
        assert Features.get_index(Features.U10) == 3
        assert Features.get_index(Features.V10) == 4
        assert Features.get_index(Features.TP) == 5

    def test_get_index_invalid(self) -> None:
        """Test get_index raises error for invalid feature."""
        with pytest.raises(ValueError):
            Features.get_index("invalid_feature")

    def test_validate(self) -> None:
        """Test validate returns correct values."""
        assert Features.validate(Features.T2M) is True
        assert Features.validate(Features.SP) is True
        assert Features.validate("invalid") is False
        assert Features.validate("") is False

    def test_colormaps(self) -> None:
        """Test COLORMAPS is correctly defined."""
        assert Features.TCC in Features.COLORMAPS
        assert Features.T2M in Features.COLORMAPS
        assert Features.COLORMAPS[Features.TCC] == "Greens"
        assert Features.COLORMAPS[Features.T2M] == "coolwarm"

    def test_labels(self) -> None:
        """Test LABELS is correctly defined."""
        assert Features.TP in Features.LABELS
        assert Features.TCC in Features.LABELS
        assert Features.T2M in Features.LABELS


class TestFeatureMetadata:
    """Tests for FeatureMetadata dataclass."""

    def test_metadata_is_frozen(self) -> None:
        """Test that FeatureMetadata is immutable."""
        meta = FeatureMetadata(
            name=FeatureName.T2M,
            description="Test",
            unit="K",
            display_unit="C",
        )
        with pytest.raises(AttributeError):
            meta.description = "Modified"  # type: ignore

    def test_metadata_fields(self) -> None:
        """Test FeatureMetadata has correct fields."""
        meta = FeatureMetadata(
            name=FeatureName.SP,
            description="Surface pressure",
            unit="Pa",
            display_unit="hPa",
        )
        assert meta.name == FeatureName.SP
        assert meta.description == "Surface pressure"
        assert meta.unit == "Pa"
        assert meta.display_unit == "hPa"
