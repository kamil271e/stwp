"""Visualization utilities for weather data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.contour import QuadContourSet
    from numpy.typing import NDArray

coolwarm = plt.cm.coolwarm  # type: ignore[attr-defined]


def draw_poland(
    ax: GeoAxes,
    data: NDArray[np.floating[Any]],
    title: str,
    cmap: str | Any,
    norm: Normalize | None = None,
    method: int = 1,
    **spatial: Any,
) -> QuadContourSet:
    """Draw a contour map of Poland with weather data.

    Args:
        ax: Matplotlib axes with Mercator projection
        data: 2D array of data to plot
        title: Plot title
        cmap: Colormap to use
        norm: Optional normalization for colormap
        method: 0 for animation frame, 1 for static plot
        **spatial: Spatial parameters (lat_span, lon_span, spatial_limits)

    Returns:
        Contour plot object
    """
    lat_span = spatial["lat_span"]
    lon_span = spatial["lon_span"]
    spatial_limits = spatial["spatial_limits"]

    if method == 1:
        ax.clear()
        ax.set_extent(spatial_limits)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(title, fontsize=25)
    else:
        ax.set_extent(spatial_limits)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(title)

    contour_plot = ax.contourf(
        lon_span, lat_span, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm
    )
    if method == 1:
        plt.colorbar(contour_plot)

    return contour_plot


def draw_poland_animation(
    ax: GeoAxes,
    X: NDArray[np.floating[Any]],
    title: str,
    cmap: str | Any,
    norm: Normalize | None = None,
    method: int = 0,
    **spatial: Any,
) -> QuadContourSet:
    """Draw Poland map variant for animations.

    This is an alias for draw_poland with different defaults suitable for animations.

    Args:
        ax: Matplotlib axes with Mercator projection
        X: 2D array of data to plot
        title: Plot title
        cmap: Colormap to use
        norm: Optional normalization for colormap
        method: 0 for animation frame, 1 for static plot
        **spatial: Spatial parameters (lat_span, lon_span, spatial_limits)

    Returns:
        Contour plot object
    """
    return draw_poland(ax, X, title, cmap, norm=norm, method=method, **spatial)


def get_spatial_info(
    res: float = 0.25,
    area: tuple[float, float, float, float] | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[float]]:
    """Get spatial information for plotting.

    Args:
        res: Resolution in degrees
        area: Geographic area (north, west, south, east). Uses SMALL_AREA by default.

    Returns:
        Tuple of (lat_span, lon_span, spatial_limits)
    """
    from stwp.data.download import SMALL_AREA

    if area is None:
        north, west, south, east = SMALL_AREA
    else:
        north, west, south, east = area

    spatial_limits = [west, east, south, north]
    we_span_1d = np.arange(west, east + res, res)
    ns_span_1d = np.arange(north, south - res, -res)
    lon_span = np.array([we_span_1d for _ in range(len(ns_span_1d))])
    lat_span = np.array([ns_span_1d for _ in range(len(we_span_1d))]).T
    return lat_span, lon_span, spatial_limits


def create_pred_animation(
    y: NDArray[np.floating[Any]],
    y_hat: NDArray[np.floating[Any]],
    num_samples: int = 20,
) -> animation.FuncAnimation:
    """Create an animation comparing predictions with ground truth.

    Args:
        y: Ground truth data
        y_hat: Predicted data
        num_samples: Number of frames in the animation

    Returns:
        Matplotlib animation object
    """
    vmin = min(np.min(y[:num_samples]), np.min(y_hat[:num_samples]))
    vmax = max(np.max(y[:num_samples]), np.max(y_hat[:num_samples]))
    norm = Normalize(vmin=vmin, vmax=vmax)

    lat_span, lon_span, spatial_limits = get_spatial_info()
    spatial = {
        "lat_span": lat_span,
        "lon_span": lon_span,
        "spatial_limits": spatial_limits,
    }

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05])

    ax0 = fig.add_subplot(gs[0], projection=ccrs.Mercator(central_longitude=40))
    ax1 = fig.add_subplot(gs[2], projection=ccrs.Mercator(central_longitude=40))

    cax0 = fig.add_subplot(gs[1])
    cbar0 = ColorbarBase(cax0, cmap=coolwarm, norm=norm, orientation="vertical")
    cbar0.ax.tick_params(labelsize=20)

    cax1 = fig.add_subplot(gs[3])
    cbar1 = ColorbarBase(cax1, cmap=coolwarm, norm=norm, orientation="vertical")
    cbar1.ax.tick_params(labelsize=20)

    ax0.axis("off")
    ax1.axis("off")

    def update(frame: int) -> list[Any]:
        ax0.clear()
        im0 = draw_poland_animation(
            ax=ax0,
            X=y[frame],
            title=r"$Y_{t2m}$",
            cmap=coolwarm,
            norm=norm,
            method=0,
            **spatial,
        )

        ax1.clear()
        im1 = draw_poland_animation(
            ax=ax1,
            X=y_hat[frame],
            title=r"$\hat{Y}_{t2m}$",
            cmap=coolwarm,
            norm=norm,
            method=0,
            **spatial,
        )

        plt.tight_layout()
        return [im0, im1]

    ani = animation.FuncAnimation(fig, update, frames=num_samples, interval=200, blit=True)
    return ani
