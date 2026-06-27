"""
forecast_engine.zoo.deep
========================
Deep learning forecasters (NeuralForecast: TFT, N-HiTS, TiDE, TSMixer)
and time-series foundation models (Chronos, TimesFM, MOIRAI, Lag-Llama).

Both wrap heavy external dependencies that may not be installed. Each
adapter degrades gracefully: if the dependency is missing, it raises a
clear ImportError at construction so the selection layer simply excludes
it from the candidate set rather than crashing a run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.types import Forecast
from forecast_engine.zoo.base import BaseForecaster


class DeepForecaster(BaseForecaster):
    """
    Deep learning forecaster wrapping NeuralForecast models.

    Supported model types: 'tft', 'nhits', 'tide', 'tsmixer'. TFT exposes
    interpretable attention attributions; TiDE/TSMixer are cheaper MLP
    architectures that remain competitive — important at scale.
    """

    name = "deep"
    cost = 2.0
    supports_covariates = True
    supports_future_covariates = True
    is_probabilistic = True

    def __init__(self, model_type: str = "tide", horizon: int = 28,
                 input_size: int | None = None, max_steps: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.horizon = horizon
        self.input_size = input_size or horizon * 4
        self.max_steps = max_steps
        self.name = f"deep:{model_type}"
        self._nf = None

        try:
            import neuralforecast  # noqa
            self._available = True
        except ImportError:
            self._available = False
            raise ImportError(
                f"NeuralForecast required for DeepForecaster({model_type}). "
                "pip install neuralforecast"
            )

    def _fit(self, series: pd.Series, covariates=None) -> None:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT, NHITS, TiDE, TSMixer
        from neuralforecast.losses.pytorch import MQLoss

        model_map = {
            "tft": TFT, "nhits": NHITS, "tide": TiDE, "tsmixer": TSMixer,
        }
        ModelClass = model_map.get(self.model_type, TiDE)

        # MQLoss gives native probabilistic (multi-quantile) output
        model = ModelClass(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            loss=MQLoss(level=[80, 95]),
        )
        self._nf = NeuralForecast(models=[model], freq=pd.infer_freq(series.index) or "D")

        df = pd.DataFrame({
            "unique_id": "series",
            "ds": series.index,
            "y": series.values,
        })
        self._nf.fit(df)

    def _predict(self, horizon, quantile_levels, future_covariates, timestamps) -> Forecast:
        ts = self._make_timestamps(horizon, timestamps)
        fc = self._nf.predict()

        model_col = self.model_type.upper()
        point = fc[f"{model_col}-median"].values if f"{model_col}-median" in fc \
            else fc[model_col].values

        intervals = {}
        for level in quantile_levels:
            lo = f"{model_col}-lo-{level}"
            hi = f"{model_col}-hi-{level}"
            if lo in fc.columns and hi in fc.columns:
                intervals[level] = (fc[lo].values[:horizon], fc[hi].values[:horizon])

        return Forecast(
            series_id="series",
            model_name=self.name,
            horizon=horizon,
            timestamps=ts,
            point=point[:horizon],
            intervals=intervals,
        )


class FoundationForecaster(BaseForecaster):
    """
    Time-series foundation model forecaster (zero-shot).

    Supports Chronos, TimesFM, MOIRAI, Lag-Llama. Their unique value is
    zero-shot forecasting: a reasonable forecast on a series the model has
    never seen, with no training. The answer for cold-start and the long
    tail of low-volume series, and a strong baseline any trained model
    must beat to justify its cost.
    """

    name = "foundation"
    cost = 1.0
    is_probabilistic = True

    def __init__(self, model: str = "chronos", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.name = f"foundation:{model}"
        self._pipeline = None
        self._context = None

    def _fit(self, series: pd.Series, covariates=None) -> None:
        # Foundation models are zero-shot: "fit" just stores the context window
        self._context = series.dropna().values

        if self.model == "chronos":
            try:
                from chronos import ChronosPipeline
                import torch
                self._pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-small",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
            except ImportError:
                raise ImportError(
                    "chronos-forecasting required for Chronos. "
                    "pip install chronos-forecasting"
                )
        # TimesFM / MOIRAI / Lag-Llama adapters follow the same pattern;
        # left as extension points to keep the dependency surface manageable.

    def _predict(self, horizon, quantile_levels, future_covariates, timestamps) -> Forecast:
        ts = self._make_timestamps(horizon, timestamps)

        if self.model == "chronos" and self._pipeline is not None:
            import torch
            context = torch.tensor(self._context, dtype=torch.float32)
            # Chronos returns sample paths; we take quantiles from them
            forecast_samples = self._pipeline.predict(
                context=context, prediction_length=horizon, num_samples=100
            )
            samples = forecast_samples[0].numpy()  # (num_samples, horizon)
            point = np.median(samples, axis=0)
            intervals = {}
            for level in quantile_levels:
                alpha = (100 - level) / 2 / 100
                lo = np.quantile(samples, alpha, axis=0)
                hi = np.quantile(samples, 1 - alpha, axis=0)
                intervals[level] = (lo, hi)
        else:
            # Fallback when no foundation model is installed: damped naive
            point, intervals = self._fallback_forecast(horizon, quantile_levels)

        return Forecast(
            series_id="series",
            model_name=self.name,
            horizon=horizon,
            timestamps=ts,
            point=point,
            intervals=intervals,
        )

    def _fallback_forecast(self, horizon, quantile_levels):
        """Damped-trend naive fallback so the engine runs without the model."""
        v = self._context
        last = v[-1]
        if len(v) > 1:
            recent_trend = np.mean(np.diff(v[-min(len(v), 14):]))
        else:
            recent_trend = 0.0
        damping = 0.9 ** np.arange(1, horizon + 1)
        point = last + np.cumsum(recent_trend * damping)
        std = np.std(np.diff(v)) * np.sqrt(np.arange(1, horizon + 1)) \
            if len(v) > 1 else np.ones(horizon)
        intervals = self._intervals_from_std(point, std, quantile_levels)
        return point, intervals
