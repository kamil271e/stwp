"""Tests for visualization utilities."""

from stwp.features import Features


class TestMapGeneratorConfig:
    """Tests for MapGenerator configuration."""

    def test_feature_colormaps_use_features(self) -> None:
        """Test that MapGenerator uses centralized feature colormaps."""
        from stwp.api.services.map_generator import MapGenerator

        # Check that colormaps reference the Features class
        assert MapGenerator.FEATURE_COLORMAPS == Features.COLORMAPS
        assert Features.TCC in MapGenerator.FEATURE_COLORMAPS
        assert Features.T2M in MapGenerator.FEATURE_COLORMAPS

    def test_feature_labels_use_features(self) -> None:
        """Test that MapGenerator uses centralized feature labels."""
        from stwp.api.services.map_generator import MapGenerator

        assert MapGenerator.FEATURE_LABELS == Features.LABELS

    def test_precipitation_colors_defined(self) -> None:
        """Test that precipitation colors are defined."""
        from stwp.api.services.map_generator import MapGenerator

        assert len(MapGenerator.PRECIPITATION_COLORS) > 0
        # Each color should be RGB tuple
        for color in MapGenerator.PRECIPITATION_COLORS:
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 1

    def test_precipitation_levels_defined(self) -> None:
        """Test that precipitation levels are defined."""
        from stwp.api.services.map_generator import MapGenerator

        assert len(MapGenerator.PRECIPITATION_LEVELS) > 0
        # Levels should be sorted ascending
        levels = MapGenerator.PRECIPITATION_LEVELS
        assert levels == sorted(levels)


class TestMapGeneratorInstance:
    """Tests for MapGenerator instance methods."""

    def test_init_default(self) -> None:
        """Test MapGenerator initialization with defaults."""
        from stwp.api.services.map_generator import MapGenerator

        generator = MapGenerator()
        assert generator.output_dir.name == "maps"
        assert generator.coord_acc == 0.25

    def test_init_custom(self) -> None:
        """Test MapGenerator initialization with custom values."""
        from stwp.api.services.map_generator import MapGenerator

        generator = MapGenerator(output_dir="/tmp/custom_maps", coord_acc=0.5)
        assert str(generator.output_dir) == "/tmp/custom_maps"
        assert generator.coord_acc == 0.5

    def test_singleton_get_map_generator(self) -> None:
        """Test that get_map_generator returns singleton."""
        from stwp.api.services.map_generator import get_map_generator

        gen1 = get_map_generator()
        gen2 = get_map_generator()
        assert gen1 is gen2
