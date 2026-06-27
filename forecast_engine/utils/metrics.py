"""
forecast_engine.utils.metrics
=============================
Forecasting metrics done correctly.

The defaults here are deliberate and opinionated:
- MASE over MAPE (scale-free, defined at zero, symmetric)
- Empirical interval coverage as a first-class metric (calibration)
- Pinball / quantile loss for probabilistic evaluation
"""
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(actual: np.ndarray, predicted: np.ndarray,
         train: np.ndarray, seasonal_period: int = 1) -> float:
    """
    Mean Absolute Scaled Error.

    Scaled by the in-sample MAE of a (seasonal) naive forecast. MASE < 1
    means the model beats the naive baseline. Scale-free and comparable
    across series — the right default for a heterogeneous model zoo.

    Hyndman & Koehler (2006).
    """
    if len(train) <= seasonal_period:
        scale = np.mean(np.abs(np.diff(train))) + 1e-8
    else:
        naive_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])
        scale = np.mean(naive_errors) + 1e-8
    return float(np.mean(np.abs(actual - predicted)) / scale)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — bounded [0, 200], still imperfect near zero."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + 1e-8
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)


def coverage(actual: np.ndarray, lower: np.ndarray,
             upper: np.ndarray) -> float:
    """
    Empirical coverage of a prediction interval. A well-calibrated 80%
    interval should contain the truth ~80% of the time. The check almost
    everyone skips and that matters more than point accuracy for decisions.
    """
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def pinball_loss(actual: np.ndarray, predicted: np.ndarray,
                 quantile: float) -> float:
    """
    Pinball (quantile) loss. The proper scoring rule for a single quantile
    forecast. Averaging across quantiles gives a proper probabilistic score.
    """
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def weighted_quantile_loss(actual: np.ndarray,
                           quantile_forecasts: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a set of quantile forecasts."""
    losses = [
        pinball_loss(actual, fc, q)
        for q, fc in quantile_forecasts.items()
    ]
    return float(np.mean(losses))


def asymmetric_loss(actual: np.ndarray, predicted: np.ndarray,
                    over_penalty: float = 1.0,
                    under_penalty: float = 1.0) -> float:
    """
    Asymmetric absolute loss. Encodes that over- and under-forecasting
    carry different costs — the realistic case for most business decisions.

    over_penalty applies when predicted > actual (over-forecast).
    under_penalty applies when predicted < actual (under-forecast).
    """
    error = predicted - actual
    weighted = np.where(error > 0, error * over_penalty, -error * under_penalty)
    return float(np.mean(weighted))
