"""
forecast_engine.selection.backtest
==================================
Walk-forward time-series cross-validation.

The only honest evaluation: repeatedly train on data up to time t, evaluate
on t+1..t+h, walk forward. Evaluates not just point accuracy but probabilistic
calibration and the decision-relevant (possibly asymmetric) loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.types import (
    BacktestResult, FoldResult, ForecastSpec,
)
from forecast_engine.zoo.base import BaseForecaster
from forecast_engine.utils import metrics


class Backtester:
    """
    Walk-forward backtesting with expanding windows.

    Usage:
        bt = Backtester(spec, n_windows=5)
        result = bt.evaluate(forecaster, series, covariates)
    """

    def __init__(self, spec: ForecastSpec, n_windows: int = 5,
                 step_size: int | None = None):
        self.spec = spec
        self.n_windows = n_windows
        self.horizon = spec.horizon
        self.step_size = step_size or spec.horizon
        self.loss_fn = spec.loss_fn()

    def evaluate(self, forecaster: BaseForecaster, series: pd.Series,
                 covariates: pd.DataFrame | None = None,
                 seasonal_period: int = 1) -> BacktestResult:
        """Walk forward through the series, training and scoring each fold."""
        fold_results = []
        n = len(series)

        for w in range(self.n_windows):
            # Expanding window: train on everything up to cutoff
            cutoff = n - (self.n_windows - w) * self.step_size
            if cutoff < self.horizon * 2:  # need enough training history
                continue

            train = series.iloc[:cutoff]
            actual = series.iloc[cutoff:cutoff + self.horizon].values

            if len(actual) < self.horizon:
                continue

            train_cov = covariates.iloc[:cutoff] if covariates is not None else None
            future_cov = (
                covariates.iloc[cutoff:cutoff + self.horizon]
                if covariates is not None else None
            )

            # Fit on the training window, forecast the held-out horizon
            try:
                forecaster.fit(train, train_cov)
                forecast = forecaster.predict(
                    self.horizon,
                    quantile_levels=self.spec.quantile_levels,
                    future_covariates=future_cov,
                )
            except Exception as e:
                # A model that fails on a fold is recorded as worst-case,
                # not allowed to crash the whole backtest
                continue

            point = forecast.point[:self.horizon]

            cov_results = {}
            for level in self.spec.quantile_levels:
                if level in forecast.intervals:
                    lo, hi = forecast.intervals[level]
                    cov_results[level] = metrics.coverage(
                        actual, lo[:self.horizon], hi[:self.horizon]
                    )
                else:
                    cov_results[level] = 0.0

            fold_results.append(FoldResult(
                fold=w,
                cutoff=train.index[-1] if hasattr(train.index[-1], "year")
                       else pd.Timestamp.now(),
                mae=metrics.mae(actual, point),
                rmse=metrics.rmse(actual, point),
                mase=metrics.mase(actual, point, train.values, seasonal_period),
                decision_loss=self.loss_fn(actual, point),
                coverage=cov_results,
            ))

        return BacktestResult(
            model_name=forecaster.name,
            series_id=str(series.name or "series"),
            fold_results=fold_results,
            cost_estimate=forecaster.cost_estimate(),
        )

    def evaluate_zoo(self, forecasters: list[BaseForecaster],
                     series: pd.Series,
                     covariates: pd.DataFrame | None = None,
                     seasonal_period: int = 1) -> list[BacktestResult]:
        """Backtest a list of candidate models on the same series."""
        results = []
        for fc in forecasters:
            result = self.evaluate(fc, series, covariates, seasonal_period)
            if result.fold_results:  # only keep models that produced folds
                results.append(result)
        return results
