"""TIGGE (THORPEX Interactive Grand Global Ensemble) integration."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

import cfgrib
import numpy as np

from stwp.features import Features
from stwp.models.base import BaselineRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def step_split(feature: NDArray[Any], n_steps: int = 3) -> NDArray[Any]:
    """Split feature array by forecast steps.

    Args:
        feature: Feature array with steps dimension
        n_steps: Number of steps to split into

    Returns:
        Array of split features
    """
    step_arrays = np.split(feature, n_steps, axis=1)
    step_arrays = [np.squeeze(arr, axis=1) for arr in step_arrays]
    return np.array(step_arrays)


def load_tigge_0_to_12_by_6(
    grib_file: str,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Load TIGGE data with 0, 6, and 12 hour forecast steps.

    Args:
        grib_file: Path to TIGGE GRIB file

    Returns:
        Tuple of (data_step_0, data_step_6, data_step_12) arrays
    """
    grib_data = cfgrib.open_datasets(grib_file)

    tcc_tigge = grib_data[0].tcc.to_numpy()
    tcc_step_0, tcc_step_6, tcc_step_12 = step_split(tcc_tigge) / 100

    u10_tigge = grib_data[1].u10.to_numpy()
    u10_step_0, u10_step_6, u10_step_12 = step_split(u10_tigge)

    v10_tigge = grib_data[1].v10.to_numpy()
    v10_step_0, v10_step_6, v10_step_12 = step_split(v10_tigge)

    t2m_tigge = grib_data[2].t2m.to_numpy()
    t2m_step_0, t2m_step_6, t2m_step_12 = step_split(t2m_tigge) - 273.15

    sp_tigge = grib_data[3].sp.to_numpy()
    sp_step_0, sp_step_6, sp_step_12 = step_split(sp_tigge) / 100

    tp_tigge = grib_data[3].tp.to_numpy()
    tp_step_0, tp_step_6, tp_step_12 = step_split(tp_tigge)

    data_step_0 = np.stack(
        (t2m_step_0, sp_step_0, tcc_step_0, u10_step_0, v10_step_0, tp_step_0), axis=-1
    )
    data_step_6 = np.stack(
        (t2m_step_6, sp_step_6, tcc_step_6, u10_step_6, v10_step_6, tp_step_6), axis=-1
    )
    data_step_12 = np.stack(
        (t2m_step_12, sp_step_12, tcc_step_12, u10_step_12, v10_step_12, tp_step_12),
        axis=-1,
    )

    return data_step_0, data_step_6, data_step_12


def evaluate_and_compare(
    data1: NDArray[Any],
    data2: NDArray[Any],
    max_samples: int = 1,
) -> None:
    """Evaluate and compare two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        max_samples: Maximum number of samples to plot
    """
    feature_list = Features.as_list()

    X, Y = data1[..., np.newaxis, :], data2[..., np.newaxis, :]
    reg = BaselineRegressor(X.shape, 1, feature_list)
    reg.plot_predictions(X, Y, max_samples=max_samples)
    rmse_scores, mae_scores = reg.evaluate(X, Y)

    for i, feature in enumerate(feature_list):
        print(f"{feature} => RMSE: {rmse_scores[i]};  MAE: {mae_scores[i]};")


def save_prediction_tensor(
    y_hat: NDArray[Any],
    path: str | None = None,
) -> None:
    """Save predictions to file.

    Args:
        y_hat: Predictions to save
        path: Output path
    """
    if path is None:
        t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        path = f"../data/pred/tigge_{t}.npy"
    np.save(path, y_hat)
