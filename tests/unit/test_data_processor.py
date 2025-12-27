"""Tests for DataProcessor utilities."""

import numpy as np


class TestDataProcessorUtilities:
    """Tests for DataProcessor static methods."""

    def test_count_neighbours_radius_0(self) -> None:
        """Test count_neighbours with radius 0."""
        from stwp.data.processor import DataProcessor

        count, indices = DataProcessor.count_neighbours(0)
        assert count == 0
        assert indices == []

    def test_count_neighbours_radius_1(self) -> None:
        """Test count_neighbours with radius 1."""
        from stwp.data.processor import DataProcessor

        count, indices = DataProcessor.count_neighbours(1)
        # 8 neighbours for radius 1 (including diagonals)
        assert count == 4
        # Check that (1,0), (-1,0), (0,1), (0,-1) are included
        expected_indices = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        assert sorted(indices) == sorted(expected_indices)

    def test_count_neighbours_radius_2(self) -> None:
        """Test count_neighbours with radius 2."""
        from stwp.data.processor import DataProcessor

        count, indices = DataProcessor.count_neighbours(2)
        # Should have more neighbours
        assert count > 4
        # Verify no duplicates
        assert len(indices) == len(set(indices))

    def test_count_neighbours_negative_radius(self) -> None:
        """Test count_neighbours with negative radius."""
        from stwp.data.processor import DataProcessor

        count, indices = DataProcessor.count_neighbours(-1)
        assert count == 0
        assert indices == []

    def test_get_spatial_info_default(self) -> None:
        """Test get_spatial_info with default area."""
        from stwp.data.processor import DataProcessor

        lat_span, lon_span, limits = DataProcessor.get_spatial_info()

        # Check shapes are 2D
        assert lat_span.ndim == 2
        assert lon_span.ndim == 2
        assert len(limits) == 4

    def test_get_spatial_info_custom_area(self) -> None:
        """Test get_spatial_info with custom area."""
        from stwp.data.processor import DataProcessor

        custom_area = (55.0, 14.0, 49.0, 25.0)  # north, west, south, east
        lat_span, lon_span, limits = DataProcessor.get_spatial_info(area=custom_area)

        # Check limits match input
        assert limits == [14.0, 25.0, 49.0, 55.0]  # west, east, south, north


class TestTrainTestSplit:
    """Tests for train/test split functionality."""

    def test_split_ratio(self) -> None:
        """Test that split respects ratio."""
        from stwp.data.processor import DataProcessor

        X = np.random.rand(100, 10)
        y = np.random.rand(100, 5)

        X_train, X_test, y_train, y_test = DataProcessor.train_test_split(
            X, y, split_ratio=0.5, split_type=3
        )

        assert len(X_train) == 50
        assert len(y_train) == 50

    def test_split_type_1(self) -> None:
        """Test split type 1 (train: first third, test: second third)."""
        from stwp.data.processor import DataProcessor

        X = np.arange(90).reshape(30, 3)
        y = np.arange(60).reshape(30, 2)

        X_train, X_test, y_train, y_test = DataProcessor.train_test_split(
            X, y, split_ratio=1 / 3, split_type=1
        )

        assert len(X_train) == 10
        assert len(X_test) == 10

    def test_split_type_2(self) -> None:
        """Test split type 2 (train: first third, test: last third)."""
        from stwp.data.processor import DataProcessor

        X = np.arange(90).reshape(30, 3)
        y = np.arange(60).reshape(30, 2)

        X_train, X_test, y_train, y_test = DataProcessor.train_test_split(
            X, y, split_ratio=1 / 3, split_type=2
        )

        assert len(X_train) == 10
        # Last third should be from index 20 onwards
        assert len(X_test) == 10

    def test_shapes_preserved(self) -> None:
        """Test that array shapes are preserved correctly."""
        from stwp.data.processor import DataProcessor

        X = np.random.rand(100, 25, 45, 6)
        y = np.random.rand(100, 25, 45, 6)

        X_train, X_test, y_train, y_test = DataProcessor.train_test_split(
            X, y, split_ratio=0.5, split_type=3
        )

        # Check all dimensions except the first are preserved
        assert X_train.shape[1:] == (25, 45, 6)
        assert y_train.shape[1:] == (25, 45, 6)
