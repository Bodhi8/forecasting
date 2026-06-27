"""
forecast_engine.monitoring.drift
================================
Production drift detection.

Distinguishes the two phenomena that are routinely (and expensively) conflated:
- Data drift: the input distribution changed (PSI, KS test)
- Concept drift: the input-output relationship changed (error inflation)

They require different responses, so the monitor reports them separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DataDriftResult:
    psi: float
    ks_statistic: float
    ks_p_value: float
    detected: bool
    severity: str


@dataclass
class ConceptDriftResult:
    error_inflation_ratio: float
    significance_p: float
    detected: bool
    interpretation: str


@dataclass
class CalibrationDriftResult:
    expected_coverage: float
    observed_coverage: float
    breach: bool


class DriftMonitor:
    """
    Multi-signal drift detection for a deployed forecaster.

    Usage:
        monitor = DriftMonitor()
        data = monitor.detect_data_drift(reference_features, live_features)
        concept = monitor.detect_concept_drift(recent_errors, baseline_errors)
        calib = monitor.check_calibration(actuals, lowers, uppers, level=80)
    """

    def __init__(self, psi_threshold: float = 0.2,
                 error_inflation_threshold: float = 1.15,
                 calibration_tolerance: float = 0.10):
        self.psi_threshold = psi_threshold
        self.error_inflation_threshold = error_inflation_threshold
        self.calibration_tolerance = calibration_tolerance

    def detect_data_drift(self, reference: np.ndarray,
                          production: np.ndarray) -> DataDriftResult:
        """Population Stability Index + Kolmogorov-Smirnov test."""
        psi = self._psi(reference, production)
        ks_stat, ks_p = stats.ks_2samp(reference, production)

        detected = psi > self.psi_threshold or ks_p < 0.05
        severity = ("high" if psi > 0.25 else
                    "moderate" if psi > 0.1 else "low")

        return DataDriftResult(
            psi=float(psi),
            ks_statistic=float(ks_stat),
            ks_p_value=float(ks_p),
            detected=detected,
            severity=severity,
        )

    def detect_concept_drift(self, recent_errors: np.ndarray,
                             baseline_errors: np.ndarray) -> ConceptDriftResult:
        """
        Concept drift: error distribution worsens even when inputs look stable.
        One-sided t-test for upward shift in errors, plus inflation ratio.
        """
        recent = np.abs(recent_errors)
        baseline = np.abs(baseline_errors)

        t_stat, t_p = stats.ttest_ind(recent, baseline, alternative="greater")
        inflation = float(recent.mean() / (baseline.mean() + 1e-8))

        detected = t_p < 0.05 and inflation > self.error_inflation_threshold

        return ConceptDriftResult(
            error_inflation_ratio=inflation,
            significance_p=float(t_p),
            detected=detected,
            interpretation=(
                "Input-output relationship has degraded — inputs may look "
                "normal but the model no longer maps them correctly. "
                "Retraining on recent data may not fix this; the model or "
                "features may need rethinking."
                if detected else
                "No significant concept drift detected."
            ),
        )

    def check_calibration(self, actual: np.ndarray, lower: np.ndarray,
                          upper: np.ndarray, level: int) -> CalibrationDriftResult:
        """Has the model's interval calibration drifted out of tolerance?"""
        inside = (actual >= lower) & (actual <= upper)
        observed = float(np.mean(inside))
        expected = level / 100
        breach = abs(observed - expected) > self.calibration_tolerance

        return CalibrationDriftResult(
            expected_coverage=expected,
            observed_coverage=observed,
            breach=breach,
        )

    @staticmethod
    def _psi(reference: np.ndarray, production: np.ndarray,
             bins: int = 10) -> float:
        """Population Stability Index — the production-standard drift metric."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf

        ref_counts = np.histogram(reference, breakpoints)[0]
        prod_counts = np.histogram(production, breakpoints)[0]

        ref_pct = ref_counts / max(len(reference), 1) + 1e-6
        prod_pct = prod_counts / max(len(production), 1) + 1e-6

        return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
