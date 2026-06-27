# Forecast Engine

**A hybrid production forecasting engine — classical statistics, gradient boosting, deep learning, and time-series foundation models in one governed zoo, with an LLM operations layer that never touches the forecast.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Vector1%20Private-black.svg)](#license)

---

> Private reference implementation maintained by [Vector1 Research](https://vector1.ai) for production forecasting engagements. Not for redistribution.

This is the working toolkit behind the architecture described in *"Forecasting at Scale: A Hybrid Architecture Where Classical Models, Deep Learning, Foundation Models, and LLMs Each Do What They Do Best."* It runs the full lifecycle: use-case framing → feature engineering → model zoo → backtesting → selection/ensembling → serving → monitoring → iteration.

The central design principle: **forecasts come from forecasting models; LLMs reason about the system.** Classical, tree-based, deep, and foundation models compete and ensemble in a governed zoo, selected by honest walk-forward backtesting. The LLM layer handles framing, configuration, drift narration, and executive communication — but never produces a number that drives a decision.

---

## Why This Exists

No model family dominates across all series. The factor that decides the winner is a property of the individual series — history length, signal-to-noise, covariate richness, cold-start. A system that picks one model leaves accuracy on the table for most of its series. This engine maintains the zoo and lets honest backtesting decide, per series, which model wins or which diverse ensemble to combine.

```python
from forecast_engine import ForecastEngine, ForecastSpec
from forecast_engine.types import OutputForm, LossAsymmetry

# Define the decision, not the model.
spec = ForecastSpec(
    name="monthly_active_users",
    horizon=28,
    granularity="D",
    output_form=OutputForm.DISTRIBUTION,
    loss_asymmetry=LossAsymmetry.UNDER_COSTLY,  # under-forecasting is costlier
    asymmetry_ratio=3.0,
    series_value="high",                         # high value → consider ensembling
)

engine = ForecastEngine(spec)
selection = engine.fit_select(series)   # profiles, backtests the zoo, selects
print(engine.leaderboard())             # see every candidate's honest performance

forecast = engine.forecast(series, horizon=28)
print(forecast.to_frame())              # point + calibrated 80%/95% intervals
```

---

## Architecture

```
DECISION    → ForecastSpec: horizon, granularity, loss asymmetry, output form
FEATURES    → lags, rolling stats, calendar, Fourier, covariates (serves whole zoo)
MODEL ZOO   → classical · gbdt · deep · foundation · (llm-adapter)
SELECTION   → walk-forward backtest · calibration gate · cost-aware · diversity ensemble
SERVING     → probabilistic forecast · scenario engine
MONITORING  → data drift (PSI/KS) · concept drift · calibration decay
ITERATION   → anomaly detection · scenarios · early warning
LLM OPS      → framing · config · drift narration · executive narrative (never the forecast)
```

---

## The Model Zoo

| Family | Module | Best for | Cost |
|---|---|---|---|
| **Classical** | `zoo.classical` | Short, regular series; interpretable; well-calibrated | very low |
| **Intermittent** | `zoo.classical` | Sparse / intermittent demand (Croston) | very low |
| **GBDT** | `zoo.gbdt` | Rich covariates, nonlinear, many series, explainable | low |
| **Deep** | `zoo.deep` | Long, complex, multivariate, covariate interactions | high |
| **Foundation** | `zoo.deep` | Cold-start, zero-shot, the long tail, strong baselines | medium |

Each family wraps an optional heavy dependency. If the dependency isn't installed, the family is simply excluded from the candidate set — the engine still runs. Classical and foundation families have dependency-free fallbacks so the engine works out of the box.

---

## Key Design Decisions

**MASE over MAPE.** MAPE is undefined at zero, explodes near it, and asymmetrically penalizes over- vs under-forecasting. MASE is scale-free, interpretable against a naive baseline, and comparable across series.

**Calibration is a first-class selection criterion.** A forecast whose 80% intervals only contain the truth 60% of the time is lying about its own uncertainty. The selection layer rejects badly miscalibrated models outright.

**Ensembles need family diversity.** The value of an ensemble comes from decorrelated failure modes, not from averaging strong models. The selector only ensembles when the top candidates come from different families.

**Cost-aware, per-series selection.** At scale, a 0.5% accuracy gain is not worth 10x the compute. Selection trades accuracy against cost, and the series-profiling gate prevents running every model on every series.

**LLMs wrap the system, never produce the forecast.** The `llmops` layer handles framing, config generation, drift narration, and executive communication. An LLM hallucinating a narrative is an annoyance; an LLM hallucinating a forecast that drives a capacity decision is an outage.

---

## Installation

```bash
# Core (classical + foundation fallbacks work with just these)
pip install numpy pandas scipy scikit-learn

# Full zoo (optional, heavy)
pip install statsforecast        # classical models
pip install lightgbm             # gradient boosting
pip install neuralforecast       # deep learning (TFT, N-HiTS, TiDE, TSMixer)
pip install chronos-forecasting  # foundation models

# Then install this package
pip install -e .
```

---

## Module Reference

```
forecast_engine/
├── types.py          Core dataclasses: ForecastSpec, Forecast, BacktestResult, ...
├── decision/         Decision-layer spec construction
├── features/         Feature pipeline (tabular + tensor windows)
├── zoo/              The model zoo (classical, gbdt, deep, foundation)
├── selection/        Walk-forward backtesting + cost-aware selection/ensembling
├── serving/          Inference + scenario engine
├── monitoring/       Data drift, concept drift, calibration drift
├── iteration/        Anomaly detection, scenario analysis
├── llmops/           LLM operations layer (framing, config, narration)
└── utils/            Metrics (MASE, coverage, pinball), series profiling
```

---

## Running

```bash
# End-to-end example
PYTHONPATH=. python examples/quickstart.py

# Tests
PYTHONPATH=. python -m pytest tests/ -q
```

---

## LLM Operations Layer

The `llmops` module takes any client with a `.complete(prompt) -> str` method, so it works with any provider. Four applications, none of which produce the forecast:

```python
from forecast_engine.llmops import (
    ForecastFraming,     # turn a vague request into a precise ForecastSpec
    FeatureAdvisor,      # propose candidate features (human reviews)
    DriftAnalyst,        # turn drift signals into root-cause hypotheses
    ForecastNarrator,    # turn a forecast + attributions into an exec narrative
)
```

---

## License

Proprietary. This is a private Vector1 Research reference implementation maintained for production engagements and internal use. Not licensed for redistribution.

---

*Built by [Brian Curry](https://vector1.ai) at Vector1 Research · Kansas City, MO*
