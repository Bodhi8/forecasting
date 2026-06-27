"""
forecast_engine.zoo.classical
==============================
Classical statistical forecasters: ARIMA, ETS, Theta, Croston.

These wrap Nixtla's StatsForecast where available, with a dependency-free
fallback implementation so the engine runs even without StatsForecast
installed. Classical models are not legacy — they win on short, regular
series and are the correct default for a large fraction of real series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.types import Forecast
from forecast_engine.zoo.base import BaseForecaster

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, CrostonOptimized
    _HAS_STATSFORECAST = True
except ImportError:
    _HAS_STATSFORECAST = False


class ClassicalForecaster(BaseForecaster):
    """
    Classical statistical forecaster. Wraps StatsForecast AutoARIMA/AutoETS/
    AutoTheta when available; otherwise falls back to a seasonal-naive +
    drift model with empirical residual intervals.
    """

    name = "classical"
    cost = 0.1                  # very cheap
    is_probabilistic = True

    def __init__(self, model: str = "auto_ets", season_length: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.season_length = season_length
        self._sf = None
        self._residual_std = None
        self._last_values = None

    def _fit(self, series: pd.Series, covariates=None) -> None:
        self._last_values = series.dropna().values

        if _HAS_STATSFORECAST:
            model_map = {
                "auto_arima": AutoARIMA(season_length=self.season_length),
                "auto_ets": AutoETS(season_length=self.season_length),
                "auto_theta": AutoTheta(season_length=self.season_length),
                "croston": CrostonOptimized(),
            }
            model = model_map.get(self.model, AutoETS(season_length=self.season_length))
            df = pd.DataFrame({
                "unique_id": "series",
                "ds": series.index,
                "y": series.values,
            })
            self._sf = StatsForecast(models=[model], freq=pd.infer_freq(series.index) or "D")
            self._sf.fit(df)
        else:
            # Fallback: estimate residual std from DETRENDED seasonal-naive errors.
            # Including the trend inflates the std and over-widens intervals,
            # breaking calibration. Detrend first for honest interval widths.
            v = self._last_values
            if len(v) > self.season_length:
                # Seasonal differencing removes both trend and seasonality,
                # leaving the irreducible noise the intervals should reflect.
                seasonal_diff = v[self.season_length:] - v[:-self.season_length]
                detrended = seasonal_diff - np.mean(seasonal_diff)
                # Differencing two independent noise terms inflates variance by
                # a factor of 2 (std by sqrt(2)). Divide it back out to recover
                # the true one-step innovation std for honest interval widths.
                self._residual_std = float(np.std(detrended) / np.sqrt(2))
            else:
                self._residual_std = float(np.std(np.diff(v)) if len(v) > 1 else 1.0)

    def _predict(self, horizon, quantile_levels, future_covariates, timestamps) -> Forecast:
        ts = self._make_timestamps(horizon, timestamps)

        if _HAS_STATSFORECAST and self._sf is not None:
            levels = quantile_levels
            fc = self._sf.predict(h=horizon, level=levels)
            # StatsForecast returns columns named by model + level
            model_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
            point_col = [c for c in model_cols if "-lo-" not in c and "-hi-" not in c][0]
            point = fc[point_col].values
            intervals = {}
            for level in levels:
                lo_col = [c for c in model_cols if f"-lo-{level}" in c]
                hi_col = [c for c in model_cols if f"-hi-{level}" in c]
                if lo_col and hi_col:
                    intervals[level] = (fc[lo_col[0]].values, fc[hi_col[0]].values)
            if not intervals:
                std = np.full(horizon, np.std(np.diff(self._last_values)))
                intervals = self._intervals_from_std(point, std, quantile_levels)
        else:
            # Fallback: seasonal-naive + drift
            point = self._seasonal_naive_forecast(horizon)
            # Intervals widen with horizon but more slowly than full sqrt growth,
            # which over-widens for seasonal series and breaks calibration.
            # Use a damped growth: std grows with horizon^0.4, calibrated so
            # near-term intervals are tight and far-term appropriately wider.
            horizon_scaling = np.power(np.arange(1, horizon + 1), 0.4)
            std = self._residual_std * horizon_scaling
            intervals = self._intervals_from_std(point, std, quantile_levels)

        return Forecast(
            series_id="series",
            model_name=f"{self.name}:{self.model}",
            horizon=horizon,
            timestamps=ts,
            point=point,
            intervals=intervals,
        )

    def _seasonal_naive_forecast(self, horizon: int) -> np.ndarray:
        v = self._last_values
        sp = self.season_length
        if len(v) >= sp:
            season = v[-sp:]
            reps = int(np.ceil(horizon / sp))
            forecast = np.tile(season, reps)[:horizon]
            # add drift
            if len(v) > 1:
                drift = (v[-1] - v[0]) / len(v)
                forecast = forecast + drift * np.arange(1, horizon + 1)
            return forecast
        return np.full(horizon, v[-1] if len(v) else 0.0)


class IntermittentForecaster(ClassicalForecaster):
    """Croston-family forecaster for intermittent/sparse demand series."""
    name = "intermittent"
    cost = 0.1

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(model="croston", season_length=season_length, **kwargs)
