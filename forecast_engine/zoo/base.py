"""
forecast_engine.zoo.base
========================
The common interface every model in the zoo implements.

A BaseForecaster wraps any underlying model (StatsForecast, NeuralForecast,
LightGBM, a foundation model) behind a uniform fit/predict contract that
always produces a probabilistic Forecast. This uniformity is what lets the
selection and backtesting layers treat the entire zoo interchangeably.
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from forecast_engine.types import Forecast


class BaseForecaster(abc.ABC):
    """
    Abstract base for all forecasters in the zoo.

    Subclasses implement _fit and _predict. The base class handles the
    uniform Forecast packaging and the cost accounting.
    """

    name: str = "base"
    cost: float = 1.0          # relative compute cost (training + inference)
    supports_covariates: bool = False
    supports_future_covariates: bool = False
    is_probabilistic: bool = True

    def __init__(self, **kwargs):
        self._fitted = False
        self._train_series: pd.Series | None = None
        self.config = kwargs

    # ── Public API ────────────────────────────────────────────────────

    def fit(self, series: pd.Series,
            covariates: pd.DataFrame | None = None) -> "BaseForecaster":
        """Fit the model to a series (and optional covariates)."""
        self._train_series = series
        self._fit(series, covariates)
        self._fitted = True
        return self

    def predict(self, horizon: int,
                quantile_levels: list[int] | None = None,
                future_covariates: pd.DataFrame | None = None,
                timestamps: pd.DatetimeIndex | None = None) -> Forecast:
        """Produce a probabilistic forecast over the horizon."""
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fit before predict")
        quantile_levels = quantile_levels or [80, 95]
        return self._predict(horizon, quantile_levels, future_covariates, timestamps)

    def cost_estimate(self) -> float:
        return self.cost

    # ── To implement ──────────────────────────────────────────────────

    @abc.abstractmethod
    def _fit(self, series: pd.Series,
             covariates: pd.DataFrame | None) -> None:
        ...

    @abc.abstractmethod
    def _predict(self, horizon: int, quantile_levels: list[int],
                 future_covariates: pd.DataFrame | None,
                 timestamps: pd.DatetimeIndex | None) -> Forecast:
        ...

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_timestamps(self, horizon: int,
                         timestamps: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
        """Generate forecast timestamps if not provided."""
        if timestamps is not None:
            return timestamps
        if self._train_series is not None and \
           isinstance(self._train_series.index, pd.DatetimeIndex):
            last = self._train_series.index[-1]
            freq = pd.infer_freq(self._train_series.index) or "D"
            return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]
        return pd.RangeIndex(horizon)

    def _intervals_from_std(self, point: np.ndarray, std: np.ndarray,
                            quantile_levels: list[int]) -> dict:
        """
        Build Gaussian prediction intervals from a point forecast and a
        standard deviation estimate. Fallback for models that don't emit
        native quantiles.
        """
        from scipy import stats as scipy_stats
        intervals = {}
        for level in quantile_levels:
            alpha = 1 - level / 100
            z = scipy_stats.norm.ppf(1 - alpha / 2)
            intervals[level] = (point - z * std, point + z * std)
        return intervals

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', cost={self.cost})"
