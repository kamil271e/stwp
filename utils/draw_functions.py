import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from matplotlib.cm import coolwarm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import cartopy.crs as ccrs
from models.data_processor import DataProcessor


def draw_poland(ax, X, title, cmap, norm=None, method=1, **spatial):
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
        lon_span, lat_span, X, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm
    )
    if method == 1:
        plt.colorbar(contour_plot)

    return contour_plot


def create_pred_animation(y, y_hat, num_samples=20):
    vmin = min(np.min(y[:num_samples]), np.min(y_hat[:num_samples]))
    vmax = max(np.max(y[:num_samples]), np.max(y_hat[:num_samples]))
    norm = Normalize(vmin=vmin, vmax=vmax)

    lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
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
    cbar0 = ColorbarBase(cax0, cmap=coolwarm, norm=norm, orientation='vertical')
    cbar0.ax.tick_params(labelsize=20)

    cax1 = fig.add_subplot(gs[3])
    cbar1 = ColorbarBase(cax1, cmap=coolwarm, norm=norm, orientation='vertical')
    cbar1.ax.tick_params(labelsize=20)

    ax0.axis('off')
    ax1.axis('off')

    def update(frame):
        ax0.clear()
        im0 = draw_poland2(ax=ax0, X=y[frame], title=r'$Y_{t2m}$', cmap=coolwarm, norm=norm, method=0, **spatial)

        ax1.clear()
        im1 = draw_poland2(ax=ax1, X=y_hat[frame], title=r'$\hat{Y}_{t2m}$', cmap=coolwarm, norm=norm, method=0, **spatial)

        plt.tight_layout()
        return [im0, im1]

    ani = animation.FuncAnimation(fig, update, frames=num_samples, interval=200, blit=True)
    return ani
