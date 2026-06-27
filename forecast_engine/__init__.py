"""
forecast_engine
===============
A hybrid production forecasting engine.

Classical statistical models, gradient-boosted trees, deep learning, and
time-series foundation models coexist in a governed model zoo, competing
and ensembling per series. An optional LLM operations layer handles framing,
configuration, drift narration, and executive communication — but never
produces the forecast itself.

Quick start:
    from forecast_engine import ForecastEngine
    from forecast_engine.types import ForecastSpec, OutputForm

    spec = ForecastSpec(name="mau", horizon=28, granularity="D",
                        series_value="high")
    engine = ForecastEngine(spec)
    result = engine.fit_select(series)         # backtests the zoo, picks winner
    forecast = engine.forecast(series, horizon=28)
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Brian Curry"

import pandas as pd

from forecast_engine.types import (
    ForecastSpec, OutputForm, LossAsymmetry, SeriesProfile,
    Forecast, BacktestResult, SelectionResult,
)
from forecast_engine.zoo import ModelZoo
from forecast_engine.selection.backtest import Backtester
from forecast_engine.selection import ModelSelector
from forecast_engine.utils.profiling import profile_series

__all__ = [
    "ForecastEngine",
    "ForecastSpec",
    "OutputForm",
    "LossAsymmetry",
    "Forecast",
    "ModelZoo",
]


class ForecastEngine:
    """
    Top-level orchestrator tying the lifecycle together:
    profile → candidate selection → backtest → select → forecast.

    This is the object you actually use in production. It holds the spec,
    the zoo, and the selection decision, and exposes a clean forecast API.
    """

    def __init__(self, spec: ForecastSpec, zoo: ModelZoo | None = None,
                 n_backtest_windows: int = 5):
        self.spec = spec
        self.zoo = zoo or ModelZoo()
        self.backtester = Backtester(spec, n_windows=n_backtest_windows)
        self.selector = ModelSelector(spec)

        self._profile: SeriesProfile | None = None
        self._backtest_results: list[BacktestResult] = []
        self._selection: SelectionResult | None = None
        self._fitted_model = None

    def fit_select(self, series: pd.Series,
                   covariates: pd.DataFrame | None = None) -> SelectionResult:
        """
        Profile the series, backtest the appropriate candidate models, and
        select the winner (or ensemble). Returns the selection decision.
        """
        # 1. Profile
        self._profile = profile_series(
            series, str(series.name or "series"),
            self.spec.granularity, covariates,
            future_covariates=None,
        )

        # 2. Candidate models for this series profile
        candidate_names = self.zoo.candidates_for(self._profile)
        seasonal_period = (self._profile.seasonal_periods[0]
                           if self._profile.seasonal_periods else 1)

        candidates = []
        for name in candidate_names:
            try:
                candidates.append(self.zoo.build(name))
            except ImportError:
                # Optional dependency not installed — skip this family
                continue

        # 3. Backtest the candidates
        self._backtest_results = self.backtester.evaluate_zoo(
            candidates, series, covariates, seasonal_period,
        )

        # 4. Select
        self._selection = self.selector.select(self._backtest_results)
        return self._selection

    def forecast(self, series: pd.Series, horizon: int | None = None,
                 covariates: pd.DataFrame | None = None,
                 future_covariates: pd.DataFrame | None = None) -> Forecast:
        """
        Produce a forecast using the selected model (or ensemble).
        Must call fit_select first.
        """
        if self._selection is None:
            raise RuntimeError("Call fit_select before forecast")

        horizon = horizon or self.spec.horizon

        if self._selection.strategy == "single":
            return self._forecast_single(
                self._selection.model, series, horizon, covariates, future_covariates
            )
        else:
            return self._forecast_ensemble(
                series, horizon, covariates, future_covariates
            )

    def _forecast_single(self, model_name, series, horizon,
                         covariates, future_covariates) -> Forecast:
        family = model_name.split(":")[0]
        model = self.zoo.build(family)
        model.fit(series, covariates)
        return model.predict(
            horizon, quantile_levels=self.spec.quantile_levels,
            future_covariates=future_covariates,
        )

    def _forecast_ensemble(self, series, horizon,
                           covariates, future_covariates) -> Forecast:
        """Weighted combination of the selected ensemble members."""
        import numpy as np

        forecasts = []
        weights = []
        for model_name in self._selection.ensemble_models:
            family = model_name.split(":")[0]
            try:
                model = self.zoo.build(family)
                model.fit(series, covariates)
                fc = model.predict(
                    horizon, quantile_levels=self.spec.quantile_levels,
                    future_covariates=future_covariates,
                )
                forecasts.append(fc)
                weights.append(self._selection.ensemble_weights.get(model_name, 1.0))
            except ImportError:
                continue

        if not forecasts:
            raise RuntimeError("No ensemble members could be built")

        weights = np.array(weights) / sum(weights)

        # Weighted point forecast
        point = sum(w * fc.point[:horizon] for w, fc in zip(weights, forecasts))

        # Weighted interval bounds (approximate but well-behaved)
        intervals = {}
        for level in self.spec.quantile_levels:
            members = [fc for fc in forecasts if level in fc.intervals]
            if members:
                lo = sum(w * fc.intervals[level][0][:horizon]
                         for w, fc in zip(weights, forecasts) if level in fc.intervals)
                hi = sum(w * fc.intervals[level][1][:horizon]
                         for w, fc in zip(weights, forecasts) if level in fc.intervals)
                intervals[level] = (lo, hi)

        return Forecast(
            series_id=str(series.name or "series"),
            model_name="ensemble:" + "+".join(
                m.split(":")[0] for m in self._selection.ensemble_models
            ),
            horizon=horizon,
            timestamps=forecasts[0].timestamps,
            point=point,
            intervals=intervals,
        )

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def profile(self) -> SeriesProfile | None:
        return self._profile

    @property
    def backtest_results(self) -> list[BacktestResult]:
        return self._backtest_results

    @property
    def selection(self) -> SelectionResult | None:
        return self._selection

    def leaderboard(self) -> pd.DataFrame:
        """Backtest leaderboard across the candidate zoo, sorted by decision loss."""
        rows = []
        for r in self._backtest_results:
            rows.append({
                "model": r.model_name,
                "mase": r.mean_metrics.get("mase"),
                "decision_loss": r.mean_metrics.get("decision_loss"),
                "loss_std": r.mean_metrics.get("loss_std"),
                "coverage_80": r.calibration.get("mean_coverage_80"),
                "coverage_95": r.calibration.get("mean_coverage_95"),
                "cost": r.cost_estimate,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("decision_loss") if not df.empty else df
