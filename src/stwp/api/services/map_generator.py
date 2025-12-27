"""Map generation service for weather visualization."""

from __future__ import annotations

import logging
import math
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

from stwp.features import Features

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.contour import QuadContourSet

logger = logging.getLogger(__name__)


class MapGenerator:
    """Service for generating weather maps."""

    # Feature configurations - use centralized definitions
    FEATURE_COLORMAPS = Features.COLORMAPS

    FEATURE_LABELS = Features.LABELS

    # Custom colors for precipitation
    PRECIPITATION_COLORS = [
        (0.8431372549, 0.91764705882, 0.97647058823),
        (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
        (0.0, 1.0, 1.0),
        (0.0, 0.8784313797950745, 0.501960813999176),
        (0.0, 0.7529411911964417, 0.0),
        (0.501960813999176, 0.8784313797950745, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.6274510025978088, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.125490203499794, 0.501960813999176),
        (0.9411764740943909, 0.250980406999588, 1.0),
        (0.501960813999176, 0.125490203499794, 1.0),
        (0.250980406999588, 0.250980406999588, 1.0),
    ]

    PRECIPITATION_LEVELS = [0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 100]

    def __init__(
        self,
        output_dir: str | Path = "./maps",
        coord_acc: float = 0.25,
    ):
        """Initialize the map generator.

        Args:
            output_dir: Directory to save generated maps
            coord_acc: Coordinate accuracy in degrees
        """
        self.output_dir = Path(output_dir)
        self.coord_acc = coord_acc
        self._poland_path: mpath.Path | None = None

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_poland_boundary(self) -> mpath.Path:
        """Get the Poland boundary path for map clipping.

        Returns:
            Matplotlib Path for Poland boundary
        """
        if self._poland_path is not None:
            return self._poland_path

        reader = shpreader.Reader(
            shpreader.natural_earth(
                resolution="10m",
                category="cultural",
                name="admin_0_countries",
            )
        )

        for country in reader.records():
            if country.attributes["NAME_LONG"] == "Poland":
                poland_geom = country.geometry
                break
        else:
            raise ValueError("Poland not found in shapefile")

        poland_vertices = list(poland_geom.exterior.coords)
        self._poland_path = mpath.Path(poland_vertices)

        return self._poland_path

    def _extract_features(
        self,
        json_data: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Extract feature arrays from JSON prediction data.

        Args:
            json_data: Prediction data keyed by lat/lon/feature/timestamp

        Returns:
            Tuple of (features dict, latitudes array, longitudes array)
        """
        lats = sorted([float(k) for k in json_data], reverse=True)
        first_lat = str(lats[0])
        lngs = sorted([float(k) for k in json_data[first_lat]])

        lat_min, lat_max = min(lats), max(lats)
        lng_min, lng_max = min(lngs), max(lngs)

        lat_arr = np.arange(lat_max, lat_min - self.coord_acc, -self.coord_acc)
        lng_arr = np.arange(lng_min, lng_max + self.coord_acc, self.coord_acc)

        features = {
            Features.TCC: np.array(
                [
                    [
                        [
                            max(0.0, min(1.0, json_data[lat][lng][Features.TCC][ts]))
                            for ts in json_data[lat][lng][Features.TCC]
                        ]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            Features.TP: np.array(
                [
                    [
                        [
                            max(0.0, min(100.0, json_data[lat][lng][Features.TP][ts]))
                            if json_data[lat][lng][Features.TP][ts] >= 0.05
                            else 0.0
                            for ts in json_data[lat][lng][Features.TP]
                        ]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
            Features.T2M: np.array(
                [
                    [
                        [
                            json_data[lat][lng][Features.T2M][ts]
                            for ts in json_data[lat][lng][Features.T2M]
                        ]
                        for lng in json_data[lat]
                    ]
                    for lat in json_data
                ]
            ),
        }

        return features, lat_arr, lng_arr

    def _get_ranges(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Calculate value ranges for colormaps.

        Args:
            features: Dictionary of feature arrays

        Returns:
            Dictionary of range arrays for each feature
        """
        t2m_min = math.ceil(np.min(features[Features.T2M]))
        t2m_max = math.floor(np.max(features[Features.T2M]))

        return {
            Features.TP: np.array(self.PRECIPITATION_LEVELS),
            Features.TCC: np.arange(0.01, 1.01, 0.01),
            Features.T2M: np.arange(t2m_min - 1, t2m_max + 2, 1),
        }

    def _create_feature_map(
        self,
        feature_name: str,
        data: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        levels: np.ndarray,
        timestep: int,
    ) -> QuadContourSet:
        """Create a single feature map.

        Args:
            feature_name: Name of the feature
            data: 2D array of values
            lats: Latitude array
            lons: Longitude array
            levels: Contour levels
            timestep: Timestep index

        Returns:
            Contourf object for legend creation
        """
        map_crs = ccrs.Mercator(central_longitude=40)
        data_crs = ccrs.PlateCarree()

        _ = plt.figure(1, figsize=(14, 12))
        ax: GeoAxes = plt.subplot(1, 1, 1, projection=map_crs)
        ax.set_extent([14, 25, 49, 55])

        # Create contour plot with appropriate colormap
        cmap_obj: Any
        cf: QuadContourSet
        if feature_name == Features.TP:
            cmap_obj = mcolors.ListedColormap(self.PRECIPITATION_COLORS, "custom_cmap")
            norm = mcolors.BoundaryNorm(self.PRECIPITATION_LEVELS, cmap_obj.N)
            cf = ax.contourf(
                lons,
                lats,
                data,
                levels=levels,
                cmap=cmap_obj,
                transform=data_crs,
                norm=norm,
            )
        else:
            cmap_obj = plt.colormaps[self.FEATURE_COLORMAPS.get(feature_name, "viridis")]
            cf = ax.contourf(lons, lats, data, levels=levels, cmap=cmap_obj, transform=data_crs)

        # Clip to Poland boundary
        poland_path = self._get_poland_boundary()
        patch = patches.PathPatch(
            poland_path,
            transform=ccrs.PlateCarree(),
            facecolor="none",
        )
        ax.add_patch(patch)
        cf.set_clip_path(patch)

        ax.set_axis_off()

        # Save map
        output_path = self.output_dir / f"{feature_name}{timestep}.png"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.clf()

        return cf

    def _create_legend(
        self,
        feature_name: str,
        cf: QuadContourSet,
        ranges: np.ndarray,
    ) -> None:
        """Create a colorbar legend for a feature.

        Args:
            feature_name: Name of the feature
            cf: Contourf return object
            ranges: Value range array
        """
        fig, ax = plt.subplots(figsize=(24, 3))
        cbar = plt.colorbar(cf, cax=ax, orientation="horizontal")

        min_val = float(np.min(ranges))
        max_val = float(np.max(ranges))

        ticks: list[float]
        if feature_name == Features.T2M:
            ticks = list(np.linspace(min_val, max_val, num=len(ranges)))
            ticklabels = [f"{int(tick)}" for tick in ticks]
        elif feature_name == Features.TCC:
            ticks = list(np.linspace(min_val, max_val, num=11))
            ticklabels = [f"{int(tick * 100)}" for tick in ticks]
        else:
            ticks = list(self.PRECIPITATION_LEVELS)
            ticklabels = [f"{tick}" for tick in ticks]
            ticklabels[-1] = f"≥{int(ticks[-1])}"

        cbar.set_ticks(ticks)
        cbar.set_label(self.FEATURE_LABELS[feature_name], fontsize=32, color="white")
        cbar.set_ticklabels(ticklabels, fontsize=32, color="white")

        plt.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.4)

        output_path = self.output_dir / f"{feature_name}_legend.png"
        plt.savefig(output_path, transparent=True)
        plt.clf()

    def create_all_maps(self, json_data: dict[str, Any]) -> Path:
        """Create all weather maps from prediction data.

        Args:
            json_data: Prediction data keyed by lat/lon/feature/timestamp

        Returns:
            Path to the output directory
        """
        logger.info("Generating weather maps...")
        self._ensure_output_dir()

        # Extract features and coordinate arrays
        features, lats, lons = self._extract_features(json_data)
        ranges = self._get_ranges(features)

        # Generate maps for each feature and timestep
        for feature_name, feature_data in features.items():
            num_timesteps = feature_data.shape[2]
            cf = None

            for t in range(num_timesteps):
                cf = self._create_feature_map(
                    feature_name=feature_name,
                    data=feature_data[:, :, t],
                    lats=lats,
                    lons=lons,
                    levels=ranges[feature_name],
                    timestep=t,
                )

            # Create legend using the last contourf object
            if cf is not None:
                self._create_legend(feature_name, cf, ranges[feature_name])

        logger.info(f"Maps generated in {self.output_dir}")
        return self.output_dir

    def create_zip(self, zip_path: str | Path = "./maps.zip") -> Path:
        """Create a ZIP archive of all generated maps.

        Args:
            zip_path: Path for the ZIP file

        Returns:
            Path to the created ZIP file
        """
        zip_path = Path(zip_path)

        with open(zip_path, "wb") as f_out, zipfile.ZipFile(f_out, mode="w") as archive:
            for image_file in self.output_dir.iterdir():
                if image_file.suffix == ".png":
                    archive.write(
                        image_file,
                        arcname=f"maps/{image_file.name}",
                    )

        return zip_path


# Singleton instance
_map_generator: MapGenerator | None = None


def get_map_generator() -> MapGenerator:
    """Get or create the singleton map generator instance.

    Returns:
        MapGenerator instance
    """
    global _map_generator
    if _map_generator is None:
        _map_generator = MapGenerator()
    return _map_generator
