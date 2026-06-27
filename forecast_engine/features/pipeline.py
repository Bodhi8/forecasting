"""
forecast_engine.features.pipeline
=================================
Feature engineering that serves the entire model zoo.

Different model families consume features in different shapes:
- Classical models: just the target (+ optional exogenous regressors)
- Tree models: a wide tabular matrix (lags, rolling stats, calendar, Fourier)
- Deep learning / foundation models: tensor windows with covariate channels

This module produces all of those from the same underlying data, with
strict no-leakage guarantees (all rolling/lag features use only past values).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """Declares the feature surface available to the zoo."""
    lags: list[int] = field(default_factory=lambda: [1, 7, 28, 365])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 28, 90])
    fourier_terms: dict[float, int] = field(
        default_factory=lambda: {7.0: 3, 365.25: 5}
    )
    calendar: bool = True
    future_covariates: list[str] = field(default_factory=list)
    past_covariates: list[str] = field(default_factory=list)
    static_covariates: list[str] = field(default_factory=list)


class FeaturePipeline:
    """
    Builds features in every format the zoo requires.

    Usage:
        pipe = FeaturePipeline(spec)
        tabular = pipe.build_tabular(df, "target")       # for tree models
        ctx, cov, tgt = pipe.build_windows(df, "target", 90, 28)  # for deep
    """

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()

    # ── Tabular (tree models) ─────────────────────────────────────────

    def build_tabular(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Wide tabular feature matrix for gradient-boosted models.
        All features computed from PAST values only — no leakage.
        """
        features = pd.DataFrame(index=df.index)
        features[target_col] = df[target_col]

        # Lag features
        for lag in self.spec.lags:
            features[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        # Rolling statistics — shift(1) first to guarantee no leakage
        shifted = df[target_col].shift(1)
        for window in self.spec.rolling_windows:
            features[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            features[f"roll_std_{window}"] = shifted.rolling(window).std()
            features[f"roll_min_{window}"] = shifted.rolling(window).min()
            features[f"roll_max_{window}"] = shifted.rolling(window).max()

        # Calendar features
        if self.spec.calendar and isinstance(df.index, pd.DatetimeIndex):
            features["day_of_week"] = df.index.dayofweek
            features["day_of_month"] = df.index.day
            features["week_of_year"] = df.index.isocalendar().week.astype(int)
            features["month"] = df.index.month
            features["quarter"] = df.index.quarter
            features["is_month_start"] = df.index.is_month_start.astype(int)
            features["is_month_end"] = df.index.is_month_end.astype(int)

        # Fourier seasonality terms
        t = np.arange(len(df))
        for period, order in self.spec.fourier_terms.items():
            for k in range(1, order + 1):
                features[f"sin_{int(period)}_{k}"] = np.sin(2 * np.pi * k * t / period)
                features[f"cos_{int(period)}_{k}"] = np.cos(2 * np.pi * k * t / period)

        # Covariates (past + future — both available at training)
        for col in self.spec.past_covariates + self.spec.future_covariates:
            if col in df.columns:
                features[col] = df[col]

        return features

    # ── Tensor windows (deep + foundation models) ─────────────────────

    def build_windows(self, df: pd.DataFrame, target_col: str,
                      context_length: int, horizon: int):
        """
        Sliding tensor windows for deep learning and foundation models.

        Returns:
            context: (n_windows, context_length) target history
            future_cov: (n_windows, horizon, n_future_cov) known-future covariates
            target: (n_windows, horizon) values to predict
        """
        values = df[target_col].values
        n = len(values)
        n_windows = n - context_length - horizon + 1

        if n_windows <= 0:
            raise ValueError(
                f"Series too short: need {context_length + horizon} points, "
                f"have {n}"
            )

        context = np.zeros((n_windows, context_length))
        target = np.zeros((n_windows, horizon))

        future_cov_cols = [c for c in self.spec.future_covariates if c in df.columns]
        future_cov = np.zeros((n_windows, horizon, len(future_cov_cols)))

        for i in range(n_windows):
            context[i] = values[i:i + context_length]
            target[i] = values[i + context_length:i + context_length + horizon]
            for j, col in enumerate(future_cov_cols):
                future_cov[i, :, j] = df[col].values[
                    i + context_length:i + context_length + horizon
                ]

        return context, future_cov, target

    def future_covariate_frame(self, df: pd.DataFrame,
                               horizon: int) -> pd.DataFrame | None:
        """
        Extract the known-future covariate values for the forecast horizon.
        Used by covariate-aware models (TFT, TiDE, MOIRAI) at inference.
        """
        cols = [c for c in self.spec.future_covariates if c in df.columns]
        if not cols:
            return None
        return df[cols].iloc[-horizon:]
