"""
forecast_engine.selection.selector
==================================
Per-series, cost-aware model selection and diversity-driven ensembling.

Selection is multi-objective: accuracy on the decision loss, calibration,
cost, and robustness. Ensembling, when chosen, requires family diversity —
because the value of an ensemble comes from decorrelated failure modes, not
from averaging strong models.
"""
from __future__ import annotations

import numpy as np

from forecast_engine.types import BacktestResult, ForecastSpec, SelectionResult


class ModelSelector:
    """
    Selects the deployed model (or ensemble) per series from backtest results.

    Usage:
        selector = ModelSelector(spec)
        decision = selector.select(backtest_results)
    """

    def __init__(self, spec: ForecastSpec,
                 calibration_tolerance: float = 0.125,
                 close_call_margin: float = 0.02):
        self.spec = spec
        self.calibration_tolerance = calibration_tolerance
        self.close_call_margin = close_call_margin

    def select(self, results: list[BacktestResult]) -> SelectionResult:
        if not results:
            return SelectionResult(
                series_id="unknown", strategy="single",
                rationale="no viable models", flagged_for_review=True,
            )

        series_id = results[0].series_id
        scored = []

        for r in results:
            # Hard gate: reject badly miscalibrated models if calibration required
            if self.spec.requires_explainability or \
               self.spec.output_form.value == "distribution":
                cov_80 = r.calibration.get("mean_coverage_80", 0.0)
                if abs(cov_80 - 0.80) > self.calibration_tolerance:
                    continue

            score = self._composite_score(r)
            scored.append((r, score))

        if not scored:
            # Everything failed calibration — fall back to best by loss, flag it
            best = min(results, key=lambda r: r.mean_metrics.get("decision_loss", 1e9))
            return SelectionResult(
                series_id=series_id, strategy="single", model=best.model_name,
                rationale="all models failed calibration gate; selected best loss",
                flagged_for_review=True,
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        return self._decide(scored, series_id)

    def _composite_score(self, r: BacktestResult) -> float:
        """Accuracy (decision loss) + robustness − cost."""
        decision_loss = r.mean_metrics.get("decision_loss", 1e9)
        loss_std = r.mean_metrics.get("loss_std", 0.0)

        accuracy = 1.0 / (1.0 + decision_loss)
        robustness = 1.0 / (1.0 + loss_std)
        cost_penalty = r.cost_estimate / max(self.spec.cost_budget, 1e-6)

        return 0.60 * accuracy + 0.25 * robustness - 0.15 * cost_penalty

    def _decide(self, scored: list[tuple[BacktestResult, float]],
                series_id: str) -> SelectionResult:
        top_result, top_score = scored[0]
        runner_up = scored[1][0].model_name if len(scored) > 1 else None

        # Close call detection: flag for human review if top two are within margin
        flagged = False
        if len(scored) > 1:
            top_loss = top_result.mean_metrics["decision_loss"]
            second_loss = scored[1][0].mean_metrics["decision_loss"]
            rel_gap = abs(top_loss - second_loss) / (top_loss + 1e-8)
            cost_gap = abs(top_result.cost_estimate - scored[1][0].cost_estimate)
            if rel_gap < self.close_call_margin and cost_gap > 0.5:
                flagged = True  # nearly tied accuracy but big cost difference

        # High-value series: ensemble top 3 IF they are family-diverse
        if self.spec.series_value == "high" and len(scored) >= 3:
            top_3 = [s[0] for s in scored[:3]]
            families = {self._family(r.model_name) for r in top_3}
            if len(families) >= 2:
                weights = self._ensemble_weights(top_3)
                return SelectionResult(
                    series_id=series_id,
                    strategy="ensemble",
                    ensemble_models=[r.model_name for r in top_3],
                    ensemble_weights=weights,
                    rationale=(
                        f"high-value series; diverse-family ensemble of "
                        f"{sorted(families)} for decorrelated errors"
                    ),
                    flagged_for_review=flagged,
                    runner_up=runner_up,
                )

        return SelectionResult(
            series_id=series_id,
            strategy="single",
            model=top_result.model_name,
            rationale=(
                f"best composite score; MASE="
                f"{top_result.mean_metrics.get('mase', float('nan')):.3f}, "
                f"cost={top_result.cost_estimate:.2f}"
            ),
            flagged_for_review=flagged,
            runner_up=runner_up,
        )

    def _ensemble_weights(self, results: list[BacktestResult]) -> dict[str, float]:
        """
        Inverse-loss weighting, normalized. Models with lower decision loss
        get more weight, but every diverse member contributes.
        """
        losses = np.array([r.mean_metrics["decision_loss"] for r in results])
        inv = 1.0 / (losses + 1e-8)
        weights = inv / inv.sum()
        return {r.model_name: float(w) for r, w in zip(results, weights)}

    @staticmethod
    def _family(model_name: str) -> str:
        """Extract the family prefix from a model name."""
        return model_name.split(":")[0].split("_")[0]
