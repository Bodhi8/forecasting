"""
forecast_engine.zoo
===================
The governed model zoo.

Five families, each with a job:
1. Classical statistical (ClassicalForecaster) — short, regular series
2. Gradient-boosted trees (GBDTForecaster)     — rich covariates, nonlinear
3. Deep learning (DeepForecaster)               — long, complex, multivariate
4. Foundation models (FoundationForecaster)     — cold-start, zero-shot, long tail
5. LLM-adapter (TimeLLMForecaster)              — text-context-driven series

The ModelZoo registry stores each family with metadata describing where it
is appropriate, so the selection layer can choose candidates per series
rather than running every model on every series — essential at scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from forecast_engine.types import SeriesProfile
from forecast_engine.zoo.base import BaseForecaster
from forecast_engine.zoo.classical import ClassicalForecaster, IntermittentForecaster

__all__ = [
    "BaseForecaster",
    "ClassicalForecaster",
    "IntermittentForecaster",
    "ModelZoo",
    "ZooEntry",
]


@dataclass
class ZooEntry:
    """A registered model family with its applicability metadata."""
    name: str
    builder: Callable[..., BaseForecaster]
    min_history: int = 0
    max_history: int | None = None
    strengths: list[str] = field(default_factory=list)
    cost: str = "medium"            # 'very_low' | 'low' | 'medium' | 'high'
    governance: str = "stable"      # 'stable' | 'experimental'
    requires_covariates: bool = False
    best_for_intermittent: bool = False


class ModelZoo:
    """
    Registry of forecasting model families.

    Usage:
        zoo = ModelZoo()                          # registers defaults
        candidates = zoo.candidates_for(profile)  # which families to try
        model = zoo.build("classical")            # instantiate one
    """

    def __init__(self, register_defaults: bool = True):
        self.registry: dict[str, ZooEntry] = {}
        if register_defaults:
            self._register_defaults()

    def register(self, entry: ZooEntry) -> None:
        self.registry[entry.name] = entry

    def build(self, name: str, **kwargs) -> BaseForecaster:
        if name not in self.registry:
            raise KeyError(f"Unknown model family '{name}'. "
                           f"Available: {list(self.registry)}")
        return self.registry[name].builder(**kwargs)

    def candidates_for(self, profile: SeriesProfile) -> list[str]:
        """
        Return the model families appropriate for a series, based on its
        profile. This is the cost-control gate: it prevents running every
        model on every series at web scale.
        """
        candidates = []
        for name, entry in self.registry.items():
            # History length gate
            if profile.history_length < entry.min_history:
                continue
            if entry.max_history is not None and \
               profile.history_length > entry.max_history:
                continue
            # Intermittency routing
            if profile.is_intermittent and not entry.best_for_intermittent:
                # Skip non-intermittent models for intermittent series
                # unless they're explicitly broad (foundation/gbdt)
                if entry.name not in ("foundation", "gbdt"):
                    continue
            # Covariate requirement
            if entry.requires_covariates and profile.n_covariates == 0:
                continue
            candidates.append(name)
        return candidates

    def _register_defaults(self) -> None:
        # Family 1: Classical statistical
        self.register(ZooEntry(
            name="classical",
            builder=lambda **kw: ClassicalForecaster(model="auto_ets", **kw),
            min_history=20,
            strengths=["short_series", "regular_seasonality", "interpretable",
                       "well_calibrated", "cheap"],
            cost="very_low",
        ))

        # Family 1b: Intermittent demand
        self.register(ZooEntry(
            name="intermittent",
            builder=lambda **kw: IntermittentForecaster(**kw),
            min_history=20,
            strengths=["intermittent_demand", "sparse_series"],
            cost="very_low",
            best_for_intermittent=True,
        ))

        # Family 2: Gradient-boosted trees
        self.register(ZooEntry(
            name="gbdt",
            builder=self._build_gbdt,
            min_history=100,
            strengths=["rich_covariates", "nonlinear", "many_series",
                       "cheap_to_serve", "explainable"],
            cost="low",
        ))

        # Family 3: Deep learning
        self.register(ZooEntry(
            name="deep",
            builder=self._build_deep,
            min_history=1000,
            strengths=["long_complex_series", "multivariate",
                       "covariate_interactions"],
            cost="high",
        ))

        # Family 4: Foundation models (zero-shot)
        self.register(ZooEntry(
            name="foundation",
            builder=self._build_foundation,
            min_history=0,    # zero-shot
            strengths=["cold_start", "zero_shot", "long_tail", "strong_baseline"],
            cost="medium",
        ))

    # Lazy builders so missing optional deps don't break import
    @staticmethod
    def _build_gbdt(**kw):
        from forecast_engine.zoo.gbdt import GBDTForecaster
        return GBDTForecaster(**kw)

    @staticmethod
    def _build_deep(**kw):
        from forecast_engine.zoo.deep import DeepForecaster
        return DeepForecaster(**kw)

    @staticmethod
    def _build_foundation(**kw):
        from forecast_engine.zoo.deep import FoundationForecaster
        return FoundationForecaster(**kw)
