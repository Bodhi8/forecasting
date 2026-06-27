"""
forecast_engine.types
=====================
Core data structures shared across the entire forecasting engine.

These are the contracts every layer agrees on: the series representation,
the forecast output (always probabilistic), the backtest result, and the
model selection decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class OutputForm(Enum):
    """The required form of a forecast output, set by the decision layer."""
    POINT = "point"
    DISTRIBUTION = "distribution"  # full predictive intervals
    SCENARIO = "scenario"          # multiple covariate-conditioned paths


class LossAsymmetry(Enum):
    """Direction of the asymmetric cost of error."""
    SYMMETRIC = "symmetric"
    OVER_COSTLY = "over_costly"    # over-forecasting is worse (e.g. inventory)
    UNDER_COSTLY = "under_costly"  # under-forecasting is worse (e.g. capacity)


@dataclass
class ForecastSpec:
    """
    The contract produced by the decision layer. Everything downstream is
    built to satisfy this spec and judged against it.
    """
    name: str
    horizon: int                       # decision-relevant horizon (periods)
    granularity: str                   # 'D', 'W', 'M', 'H', etc.
    output_form: OutputForm = OutputForm.DISTRIBUTION
    loss_asymmetry: LossAsymmetry = LossAsymmetry.SYMMETRIC
    asymmetry_ratio: float = 1.0       # how much worse the costly direction is
    quantile_levels: list[int] = field(default_factory=lambda: [80, 95])
    requires_explainability: bool = True
    series_value: str = "medium"       # 'low' | 'medium' | 'high' — drives cost budget
    cost_budget: float = 1.0           # relative compute budget per series
    notes: str = ""

    def loss_fn(self):
        """Return the decision-relevant loss function implied by the spec."""
        if self.loss_asymmetry == LossAsymmetry.SYMMETRIC:
            return lambda a, p: float(np.mean(np.abs(a - p)))
        elif self.loss_asymmetry == LossAsymmetry.UNDER_COSTLY:
            # Under-forecasting (p < a) penalized by asymmetry_ratio
            def asym(a, p):
                err = a - p
                weighted = np.where(err > 0, err * self.asymmetry_ratio, -err)
                return float(np.mean(weighted))
            return asym
        else:  # OVER_COSTLY
            def asym(a, p):
                err = a - p
                weighted = np.where(err < 0, -err * self.asymmetry_ratio, err)
                return float(np.mean(weighted))
            return asym


@dataclass
class SeriesProfile:
    """
    A statistical profile of a single series, used by the selection layer
    to decide which model families are appropriate candidates.
    """
    series_id: str
    history_length: int
    frequency: str
    has_seasonality: bool = False
    seasonal_periods: list[int] = field(default_factory=list)
    is_intermittent: bool = False      # many zeros — Croston territory
    signal_to_noise: float = 1.0
    trend_strength: float = 0.0
    n_covariates: int = 0
    n_future_covariates: int = 0
    missing_pct: float = 0.0


@dataclass
class Forecast:
    """
    A forecast output. Always carries a point estimate AND predictive
    intervals — point-only forecasts are a degenerate special case, not
    the default.
    """
    series_id: str
    model_name: str
    horizon: int
    timestamps: pd.DatetimeIndex
    point: np.ndarray
    # quantile -> array of predictions; e.g. intervals[80] = (lower, upper)
    intervals: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    attributions: Optional[dict[str, np.ndarray]] = None  # driver attributions
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def lower(self, level: int) -> np.ndarray:
        return self.intervals[level][0]

    def upper(self, level: int) -> np.ndarray:
        return self.intervals[level][1]

    def to_frame(self) -> pd.DataFrame:
        data = {"timestamp": self.timestamps, "point": self.point}
        for level, (lo, hi) in self.intervals.items():
            data[f"lower_{level}"] = lo
            data[f"upper_{level}"] = hi
        return pd.DataFrame(data).set_index("timestamp")


@dataclass
class FoldResult:
    """Metrics from a single backtest fold."""
    fold: int
    cutoff: datetime
    mae: float
    rmse: float
    mase: float
    decision_loss: float
    coverage: dict[int, float]  # level -> empirical coverage


@dataclass
class BacktestResult:
    """Aggregated backtest result for one model on one series."""
    model_name: str
    series_id: str
    fold_results: list[FoldResult]
    mean_metrics: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    cost_estimate: float = 0.0

    def __post_init__(self):
        if self.fold_results and not self.mean_metrics:
            self.mean_metrics = self._aggregate()
            self.calibration = self._calibration_summary()

    def _aggregate(self) -> dict[str, float]:
        if not self.fold_results:
            return {}
        return {
            "mae": float(np.mean([f.mae for f in self.fold_results])),
            "rmse": float(np.mean([f.rmse for f in self.fold_results])),
            "mase": float(np.mean([f.mase for f in self.fold_results])),
            "decision_loss": float(np.mean([f.decision_loss for f in self.fold_results])),
            "loss_std": float(np.std([f.decision_loss for f in self.fold_results])),
        }

    def _calibration_summary(self) -> dict[str, float]:
        summary = {}
        if not self.fold_results:
            return summary
        levels = self.fold_results[0].coverage.keys()
        for level in levels:
            covs = [f.coverage[level] for f in self.fold_results]
            summary[f"mean_coverage_{level}"] = float(np.mean(covs))
        return summary


@dataclass
class SelectionResult:
    """The model selection decision for a series."""
    series_id: str
    strategy: str                       # 'single' | 'ensemble'
    model: Optional[str] = None
    ensemble_models: list[str] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    flagged_for_review: bool = False
    runner_up: Optional[str] = None
