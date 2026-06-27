"""
examples/quickstart.py
======================
End-to-end forecasting example: profile a series, backtest the model zoo,
select the best model (or ensemble), forecast with calibrated intervals,
and detect anomalies.

Run:
    python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from forecast_engine import ForecastEngine, ForecastSpec
from forecast_engine.types import OutputForm, LossAsymmetry
from forecast_engine.iteration import AnomalyDetector


def make_demo_series(n: int = 730) -> pd.Series:
    """A synthetic daily series with trend, weekly + yearly seasonality, noise."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(100, 200, n)
    weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.randn(n) * 5
    return pd.Series(trend + weekly + yearly + noise, index=dates, name="mau")


def main():
    series = make_demo_series()

    # 1. Define the decision-relevant spec.
    #    Here: capacity-style forecast where under-forecasting is 3x costlier.
    spec = ForecastSpec(
        name="monthly_active_users",
        horizon=28,
        granularity="D",
        output_form=OutputForm.DISTRIBUTION,
        loss_asymmetry=LossAsymmetry.UNDER_COSTLY,
        asymmetry_ratio=3.0,
        series_value="high",   # high value → ensemble if families are diverse
    )

    # 2. Fit & select: profile the series, backtest the candidate zoo, pick winner.
    engine = ForecastEngine(spec, n_backtest_windows=4)
    selection = engine.fit_select(series)

    print("=" * 64)
    print("SERIES PROFILE")
    print("=" * 64)
    p = engine.profile
    print(f"  History length:    {p.history_length}")
    print(f"  Seasonal periods:  {p.seasonal_periods}")
    print(f"  Signal/noise:      {p.signal_to_noise:.2f}")
    print(f"  Trend strength:    {p.trend_strength:.2f}")

    print("\n" + "=" * 64)
    print("BACKTEST LEADERBOARD")
    print("=" * 64)
    print(engine.leaderboard().to_string(index=False))

    print("\n" + "=" * 64)
    print("SELECTION")
    print("=" * 64)
    print(f"  Strategy:  {selection.strategy}")
    print(f"  Model:     {selection.model or selection.ensemble_models}")
    print(f"  Rationale: {selection.rationale}")
    print(f"  Flagged for review: {selection.flagged_for_review}")

    # 3. Forecast with calibrated intervals.
    forecast = engine.forecast(series, horizon=28)

    print("\n" + "=" * 64)
    print("FORECAST (first 7 days)")
    print("=" * 64)
    print(forecast.to_frame().head(7).round(1).to_string())

    # 4. Anomaly detection on a held-out window using the forecast intervals.
    detector = AnomalyDetector()
    recent_actuals = series.iloc[-28:]
    # Refit on data excluding the last 28 days, forecast that window
    engine_holdout = ForecastEngine(spec, n_backtest_windows=3)
    engine_holdout.fit_select(series.iloc[:-28])
    holdout_forecast = engine_holdout.forecast(series.iloc[:-28], horizon=28)
    holdout_forecast.timestamps = recent_actuals.index
    anomalies = detector.detect(recent_actuals, holdout_forecast, level=95)

    print("\n" + "=" * 64)
    print(f"ANOMALY DETECTION ({len(anomalies)} found in last 28 days)")
    print("=" * 64)
    for a in anomalies[:5]:
        print(f"  {a.timestamp.date()}  actual={a.actual:.1f} "
              f"forecast={a.forecast:.1f} [{a.lower:.1f}, {a.upper:.1f}] "
              f"severity={a.severity}")
    if not anomalies:
        print("  No anomalies — actuals fell within forecast intervals.")


if __name__ == "__main__":
    main()
