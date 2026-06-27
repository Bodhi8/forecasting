"""
forecast_engine.llmops
======================
The LLM operations layer — connective tissue, not forecasting engine.

LLMs do everything here EXCEPT produce the forecast:
- Build:   problem framing, feature discovery suggestions
- Deploy:  monitoring config generation (schema-validated)
- Monitor: drift narration, root-cause hypotheses
- Decide:  executive narrative generation

Every method takes a pluggable `llm` client with a `.complete(prompt) -> str`
interface, so this works with any provider. The LLM never produces a number
that drives a decision, and never acts without validation.
"""
from __future__ import annotations

import json
from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal interface any LLM provider must satisfy."""
    def complete(self, prompt: str) -> str: ...


class ForecastFraming:
    """Build-time: turn a vague business request into a precise ForecastSpec."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def frame_problem(self, business_request: str,
                      available_history: dict) -> dict:
        prompt = f"""
A stakeholder made this forecasting request:
"{business_request}"

Available data summary: {json.dumps(available_history, default=str)}

Produce a structured forecasting specification as JSON. Reason about:
1. The actual decision this forecast drives
2. Whether over- vs under-forecasting costs are symmetric (if not, which
   direction is costlier and roughly the ratio)
3. The decision-relevant horizon and granularity
4. Whether a point forecast suffices or a full predictive distribution is needed
5. The explainability requirement based on who consumes the forecast

Return JSON with keys: horizon, granularity, loss_asymmetry
('symmetric'|'over_costly'|'under_costly'), asymmetry_ratio, output_form
('point'|'distribution'|'scenario'), requires_explainability (bool),
and a 'clarifications_needed' list flagging anything underspecified.
"""
        return self._safe_json(self.llm.complete(prompt))

    @staticmethod
    def _safe_json(text: str) -> dict:
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"error": "could not parse LLM response", "raw": text}


class FeatureAdvisor:
    """Build-time: propose candidate features. LLM proposes; human disposes."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def suggest_features(self, series_description: str,
                         current_features: list[str]) -> dict:
        prompt = f"""
We are forecasting: {series_description}
Current features: {current_features}

Propose additional candidate features or exogenous drivers that domain
knowledge suggests would improve this forecast. For each, give the
hypothesized causal mechanism and how to source the data.

Prioritize known-future covariates (things knowable in advance over the
forecast horizon). Flag any proposed feature that risks target leakage.

Return JSON: a list of {{name, mechanism, data_source, is_future_known,
leakage_risk}}.
"""
        return ForecastFraming._safe_json(self.llm.complete(prompt))


class SelectionExplainer:
    """Build-time: make the quantitative selection decision human-auditable."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def explain(self, backtest_summary: list[dict], selection: dict) -> str:
        prompt = f"""
Forecasting model selection.

Backtest results across the zoo: {json.dumps(backtest_summary, default=str)}
Selected: {json.dumps(selection, default=str)}

In 3-4 sentences, explain why this selection is reasonable given the evidence.
Note: Was it close? Is there a cheaper model nearly as good? Any sign the
winner is overfit to the backtest folds? Recommend human review if the top
two are within 2% on decision loss but differ materially in cost or
interpretability.
"""
        return self.llm.complete(prompt)


class DriftAnalyst:
    """Monitor-time: turn statistical drift signals into actionable diagnosis."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, data_drift: dict, concept_drift: dict,
                forecast_errors: dict, known_events: list) -> str:
        prompt = f"""
A forecasting model triggered drift alerts.

Data drift: {json.dumps(data_drift, default=str)}
Concept drift: {json.dumps(concept_drift, default=str)}
Recent forecast errors: {json.dumps(forecast_errors, default=str)}
Known business events in the window: {known_events}

Produce:
1. Ranked root-cause hypotheses (data-quality issue? genuine regime change?
   known event the model didn't account for? upstream pipeline break?)
2. Evidence for and against each
3. A recommended action (retrain / hotfix / wait and monitor / escalate)
4. A one-paragraph summary suitable for an on-call engineer at 3am

Clearly distinguish data drift (input distribution changed) from concept
drift (input-output relationship changed) — they need different responses.
"""
        return self.llm.complete(prompt)


class ForecastNarrator:
    """Decide-time: turn a forecast + attributions into an executive narrative."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def narrate(self, forecast_summary: dict, attributions: dict,
                prior_forecast_summary: dict, audience: str = "executive") -> str:
        prompt = f"""
Current forecast (point + 80%/95% intervals): {json.dumps(forecast_summary, default=str)}
Change vs last forecast: {json.dumps(prior_forecast_summary, default=str)}
Model driver attributions: {json.dumps(attributions, default=str)}
Audience: {audience}

Write a concise narrative (3-4 sentences for an executive audience):
- The headline number and the uncertainty around it
- What changed versus the prior forecast and why (use the attributions)
- The key risk and the scenario that would break the forecast

Report ONLY what the numbers and attributions support. Do not speculate
beyond the provided evidence. State uncertainty plainly.
"""
        return self.llm.complete(prompt)
