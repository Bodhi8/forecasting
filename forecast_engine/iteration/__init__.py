"""
forecast_engine.iteration
==========================
Higher-order capabilities that the same forecasting infrastructure produces:
anomaly detection, scenario analysis, and early-warning signals.

These are often where the system delivers its greatest business value —
point forecasts are the beginning, not the end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_engine.types import Forecast
from forecast_engine.zoo.base import BaseForecaster


@dataclass
class Anomaly:
    timestamp: pd.Timestamp
    actual: float
    forecast: float
    lower: float
    upper: float
    severity: str


class AnomalyDetector:
    """
    A forecast + prediction interval is an anomaly detector for free:
    an actual outside the interval is, by definition, anomalous.
    """

    def detect(self, actuals: pd.Series, forecast: Forecast,
               level: int = 99) -> list[Anomaly]:
        if level not in forecast.intervals:
            # Use the widest available interval
            level = max(forecast.intervals.keys())
        lower, upper = forecast.intervals[level]

        anomalies = []
        for i, (ts, actual) in enumerate(actuals.items()):
            if i >= len(forecast.point):
                break
            lo, hi, pt = lower[i], upper[i], forecast.point[i]
            if actual < lo or actual > hi:
                # Severity by how far outside the interval
                width = (hi - lo) + 1e-8
                distance = max(lo - actual, actual - hi) / width
                severity = ("high" if distance > 1.0 else
                            "moderate" if distance > 0.5 else "low")
                anomalies.append(Anomaly(
                    timestamp=ts, actual=float(actual), forecast=float(pt),
                    lower=float(lo), upper=float(hi), severity=severity,
                ))
        return anomalies


class ScenarioEngine:
    """
    Generate forecast scenarios under different assumptions about controllable
    future covariates. Native for covariate-aware models — vary the known
    future covariates, re-forecast, present the distribution of outcomes.
    """

    def __init__(self, forecaster: BaseForecaster):
        if not forecaster.supports_future_covariates:
            raise ValueError(
                f"{forecaster.name} does not support future covariates; "
                "scenario analysis requires a covariate-aware model "
                "(gbdt, deep:tft, deep:tide)."
            )
        self.forecaster = forecaster

    def run(self, horizon: int, base_covariates: pd.DataFrame,
            scenarios: list[dict], quantile_levels: list[int] | None = None) -> dict:
        """
        scenarios: list of {name, covariate_overrides} where overrides is a
        dict of {column: value_or_series} applied to the base covariates.
        """
        quantile_levels = quantile_levels or [80, 95]
        results = {}

        for scenario in scenarios:
            modified = base_covariates.copy()
            for col, val in scenario["covariate_overrides"].items():
                modified[col] = val

            forecast = self.forecaster.predict(
                horizon, quantile_levels=quantile_levels,
                future_covariates=modified,
            )
            lo80 = forecast.intervals.get(80, (forecast.point, forecast.point))[0]
            hi80 = forecast.intervals.get(80, (forecast.point, forecast.point))[1]

            results[scenario["name"]] = {
                "expected_value": float(np.sum(forecast.point)),
                "downside_80": float(np.sum(lo80)),
                "upside_80": float(np.sum(hi80)),
                "forecast": forecast,
            }
        return results
