"""
forecast_engine.serving
=======================
Serving layer: batch and on-demand inference over the selected models.

In production this wraps the fitted models behind a service interface with
caching, batching, and the scenario engine. The reference implementation
here keeps it minimal; production deployments add the orchestration,
feature-store wiring, and serving infrastructure described in the article.
"""
from forecast_engine.iteration import ScenarioEngine, AnomalyDetector
__all__ = ["ScenarioEngine", "AnomalyDetector"]
