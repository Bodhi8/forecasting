"""
forecast_engine.decision
========================
The decision layer. Forecasting starts with the decision, not the model.

This module holds helpers for constructing a ForecastSpec — the contract
that the rest of the engine is built to satisfy. The LLM-assisted framing
lives in forecast_engine.llmops.ForecastFraming; this module provides the
deterministic, code-first path.
"""
from forecast_engine.types import ForecastSpec, OutputForm, LossAsymmetry

__all__ = ["ForecastSpec", "OutputForm", "LossAsymmetry"]
