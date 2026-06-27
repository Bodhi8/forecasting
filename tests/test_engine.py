"""
tests/test_engine.py
End-to-end and unit tests for the forecast engine.
"""
import numpy as np
import pandas as pd
import pytest

from forecast_engine import ForecastEngine, ForecastSpec
from forecast_engine.types import OutputForm, LossAsymmetry, Forecast
from forecast_engine.zoo import ModelZoo
from forecast_engine.zoo.classical import ClassicalForecaster
from forecast_engine.utils import metrics
from forecast_engine.utils.profiling import profile_series
from forecast_engine.monitoring import DriftMonitor


@pytest.fixture
def seasonal_series():
    np.random.seed(42)
    n = 365 * 2
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = (np.linspace(100, 200, n)
              + 10 * np.sin(2 * np.pi * np.arange(n) / 7)
              + 20 * np.sin(2 * np.pi * np.arange(n) / 365)
              + np.random.randn(n) * 5)
    return pd.Series(values, index=dates, name="mau")


@pytest.fixture
def spec():
    return ForecastSpec(name="mau", horizon=28, granularity="D",
                        output_form=OutputForm.DISTRIBUTION, series_value="high")


# ── Metrics ───────────────────────────────────────────────────────────

class TestMetrics:
    def test_mase_beats_naive_below_one(self):
        train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        actual = np.array([11, 12, 13])
        good_pred = np.array([11, 12, 13])  # perfect
        assert metrics.mase(actual, good_pred, train) < 0.01

    def test_coverage_perfect(self):
        actual = np.array([5.0, 5.0, 5.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        assert metrics.coverage(actual, lower, upper) == 1.0

    def test_coverage_none(self):
        actual = np.array([15.0, 15.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        assert metrics.coverage(actual, lower, upper) == 0.0

    def test_asymmetric_loss_penalizes_correctly(self):
        actual = np.array([10.0])
        under = np.array([8.0])   # under-forecast
        over = np.array([12.0])   # over-forecast
        # With under_penalty high, under should cost more
        l_under = metrics.asymmetric_loss(actual, under, over_penalty=1, under_penalty=5)
        l_over = metrics.asymmetric_loss(actual, over, over_penalty=1, under_penalty=5)
        assert l_under > l_over

    def test_pinball_loss_nonneg(self):
        actual = np.array([5.0, 6.0])
        pred = np.array([4.0, 7.0])
        assert metrics.pinball_loss(actual, pred, 0.5) >= 0


# ── Profiling ─────────────────────────────────────────────────────────

class TestProfiling:
    def test_detects_seasonality(self, seasonal_series):
        profile = profile_series(seasonal_series, "mau", "D")
        assert profile.has_seasonality
        assert 7 in profile.seasonal_periods

    def test_detects_history_length(self, seasonal_series):
        profile = profile_series(seasonal_series, "mau", "D")
        assert profile.history_length == len(seasonal_series)

    def test_intermittent_detection(self):
        sparse = pd.Series([0, 0, 0, 5, 0, 0, 3, 0, 0, 0] * 10, name="sparse")
        profile = profile_series(sparse, "sparse", "D")
        assert profile.is_intermittent


# ── Zoo ───────────────────────────────────────────────────────────────

class TestModelZoo:
    def test_default_families_registered(self):
        zoo = ModelZoo()
        assert "classical" in zoo.registry
        assert "gbdt" in zoo.registry
        assert "foundation" in zoo.registry

    def test_candidates_respect_history_gate(self, seasonal_series):
        zoo = ModelZoo()
        profile = profile_series(seasonal_series, "mau", "D")
        candidates = zoo.candidates_for(profile)
        # 730 days of history — classical, gbdt, foundation should be candidates
        assert "classical" in candidates
        assert "gbdt" in candidates

    def test_short_series_excludes_deep(self):
        zoo = ModelZoo()
        short = pd.Series(np.random.randn(50), name="short")
        profile = profile_series(short, "short", "D")
        candidates = zoo.candidates_for(profile)
        assert "deep" not in candidates  # needs 1000+ history

    def test_build_classical(self):
        zoo = ModelZoo()
        model = zoo.build("classical")
        assert isinstance(model, ClassicalForecaster)


# ── Forecaster ────────────────────────────────────────────────────────

class TestForecaster:
    def test_classical_fit_predict(self, seasonal_series):
        model = ClassicalForecaster(model="auto_ets", season_length=7)
        model.fit(seasonal_series)
        forecast = model.predict(28, quantile_levels=[80, 95])
        assert isinstance(forecast, Forecast)
        assert len(forecast.point) == 28
        assert 80 in forecast.intervals
        # Intervals should bracket the point forecast
        lo, hi = forecast.intervals[80]
        assert np.all(lo <= forecast.point)
        assert np.all(forecast.point <= hi)

    def test_predict_before_fit_raises(self):
        model = ClassicalForecaster()
        with pytest.raises(RuntimeError):
            model.predict(10)

    def test_forecast_to_frame(self, seasonal_series):
        model = ClassicalForecaster()
        model.fit(seasonal_series)
        forecast = model.predict(10)
        df = forecast.to_frame()
        assert "point" in df.columns
        assert len(df) == 10


# ── End to end ────────────────────────────────────────────────────────

class TestEngine:
    def test_full_lifecycle(self, seasonal_series, spec):
        engine = ForecastEngine(spec, n_backtest_windows=3)
        selection = engine.fit_select(seasonal_series)
        assert selection.strategy in ("single", "ensemble")
        assert engine.profile is not None
        assert not engine.leaderboard().empty

        forecast = engine.forecast(seasonal_series, horizon=28)
        assert len(forecast.point) == 28
        assert 80 in forecast.intervals

    def test_leaderboard_sorted(self, seasonal_series, spec):
        engine = ForecastEngine(spec, n_backtest_windows=3)
        engine.fit_select(seasonal_series)
        lb = engine.leaderboard()
        losses = lb["decision_loss"].values
        assert np.all(np.diff(losses) >= 0)  # ascending

    def test_asymmetric_spec(self, seasonal_series):
        spec = ForecastSpec(
            name="capacity", horizon=14, granularity="D",
            loss_asymmetry=LossAsymmetry.UNDER_COSTLY, asymmetry_ratio=3.0,
        )
        engine = ForecastEngine(spec, n_backtest_windows=3)
        selection = engine.fit_select(seasonal_series)
        assert selection.model is not None


# ── Drift monitoring ──────────────────────────────────────────────────

class TestDriftMonitor:
    def test_no_drift_same_distribution(self):
        np.random.seed(0)
        ref = np.random.randn(1000)
        prod = np.random.randn(1000)
        monitor = DriftMonitor()
        result = monitor.detect_data_drift(ref, prod)
        assert not result.detected

    def test_drift_shifted_distribution(self):
        np.random.seed(0)
        ref = np.random.randn(1000)
        prod = np.random.randn(1000) + 3.0  # big shift
        monitor = DriftMonitor()
        result = monitor.detect_data_drift(ref, prod)
        assert result.detected
        assert result.severity == "high"

    def test_concept_drift_error_inflation(self):
        np.random.seed(0)
        baseline = np.abs(np.random.randn(500)) * 1.0
        recent = np.abs(np.random.randn(500)) * 2.0  # errors doubled
        monitor = DriftMonitor()
        result = monitor.detect_concept_drift(recent, baseline)
        assert result.detected
        assert result.error_inflation_ratio > 1.15

    def test_calibration_breach(self):
        actual = np.array([15.0] * 100)  # all outside
        lower = np.array([0.0] * 100)
        upper = np.array([10.0] * 100)
        monitor = DriftMonitor()
        result = monitor.check_calibration(actual, lower, upper, level=80)
        assert result.breach
