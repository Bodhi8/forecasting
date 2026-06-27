"""
forecast_engine.utils.profiling
================================
Builds a SeriesProfile from raw series data. The profile drives which
model families the selection layer considers as candidates — so this is
the first gate that keeps the system from running every model on every
series at web scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from forecast_engine.types import SeriesProfile


def profile_series(series: pd.Series, series_id: str,
                   frequency: str,
                   covariates: pd.DataFrame | None = None,
                   future_covariates: list[str] | None = None) -> SeriesProfile:
    """
    Compute a statistical profile of a series.

    Detects: history length, seasonality (via autocorrelation peaks),
    intermittency (zero fraction), signal-to-noise, trend strength,
    and covariate richness.
    """
    values = series.dropna().values
    n = len(values)

    # Intermittency: high fraction of zeros → Croston territory
    zero_frac = float(np.mean(values == 0)) if n > 0 else 0.0
    is_intermittent = zero_frac > 0.30

    # Seasonality detection via autocorrelation
    seasonal_periods, has_seasonality = _detect_seasonality(values, frequency)

    # Signal-to-noise: variance of smoothed signal vs residual variance
    snr = _signal_to_noise(values)

    # Trend strength: R^2 of a linear fit
    trend_strength = _trend_strength(values)

    # Missing fraction
    missing_pct = float(series.isna().mean())

    n_cov = covariates.shape[1] if covariates is not None else 0
    n_future_cov = len(future_covariates) if future_covariates else 0

    return SeriesProfile(
        series_id=series_id,
        history_length=n,
        frequency=frequency,
        has_seasonality=has_seasonality,
        seasonal_periods=seasonal_periods,
        is_intermittent=is_intermittent,
        signal_to_noise=snr,
        trend_strength=trend_strength,
        n_covariates=n_cov,
        n_future_covariates=n_future_cov,
        missing_pct=missing_pct,
    )


def _detect_seasonality(values: np.ndarray,
                        frequency: str) -> tuple[list[int], bool]:
    """Detect seasonal periods via autocorrelation peaks at known lags."""
    candidate_periods = {
        "H": [24, 168],         # daily, weekly (hourly data)
        "D": [7, 30, 365],      # weekly, monthly, yearly (daily data)
        "W": [52],              # yearly (weekly data)
        "M": [12],              # yearly (monthly data)
    }.get(frequency, [])

    if len(values) < 2 * max(candidate_periods, default=1):
        return [], False

    detected = []
    for period in candidate_periods:
        if len(values) > 2 * period:
            acf = _autocorr(values, period)
            if acf > 0.3:  # meaningful seasonal autocorrelation
                detected.append(period)

    return detected, len(detected) > 0


def _autocorr(values: np.ndarray, lag: int) -> float:
    if lag >= len(values):
        return 0.0
    v = values - values.mean()
    denom = np.sum(v ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(v[lag:] * v[:-lag]) / denom)


def _signal_to_noise(values: np.ndarray) -> float:
    """Ratio of smoothed-signal variance to residual variance."""
    if len(values) < 5:
        return 1.0
    window = min(7, len(values) // 2)
    smoothed = pd.Series(values).rolling(window, center=True).mean().values
    residual = values - np.nan_to_num(smoothed, nan=np.nanmean(values))
    signal_var = np.nanvar(smoothed)
    noise_var = np.var(residual) + 1e-8
    return float(signal_var / noise_var)


def _trend_strength(values: np.ndarray) -> float:
    """R^2 of a linear trend fit — proxy for trend strength."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values))
    slope, intercept, r_value, _, _ = stats.linregress(x, values)
    return float(r_value ** 2)
