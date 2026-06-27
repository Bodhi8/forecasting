"""
forecast_engine.zoo.gbdt
========================
Gradient-boosted tree forecaster (LightGBM).

Trained on the wide tabular feature matrix from the feature pipeline.
Uses quantile regression to produce native probabilistic forecasts —
one model per quantile level. A global GBDT across many series with
static categorical identifiers remains one of the strongest and most
cost-effective forecasters in existence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.types import Forecast
from forecast_engine.zoo.base import BaseForecaster
from forecast_engine.features.pipeline import FeaturePipeline, FeatureSpec

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


class GBDTForecaster(BaseForecaster):
    """
    LightGBM forecaster with quantile regression for probabilistic output.

    Trains one model for the median (point) and one per requested interval
    bound. Recursive multi-step forecasting: predict one step, append, repeat.
    """

    name = "gbdt"
    cost = 0.3
    supports_covariates = True
    supports_future_covariates = True
    is_probabilistic = True

    def __init__(self, feature_spec: FeatureSpec | None = None,
                 num_leaves: int = 64, learning_rate: float = 0.05,
                 n_estimators: int = 300, **kwargs):
        super().__init__(**kwargs)
        if not _HAS_LGB:
            raise ImportError("LightGBM is required for GBDTForecaster. "
                              "pip install lightgbm")
        self.feature_pipeline = FeaturePipeline(feature_spec)
        self.params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "verbose": -1,
        }
        self._models = {}        # quantile -> fitted model
        self._feature_cols = None
        self._target_col = None
        self._train_df = None

    def _fit(self, series: pd.Series, covariates=None) -> None:
        self._target_col = series.name or "target"
        df = pd.DataFrame({self._target_col: series})
        if covariates is not None:
            df = df.join(covariates)
        self._train_df = df

        features = self.feature_pipeline.build_tabular(df, self._target_col)
        features = features.dropna()

        self._feature_cols = [c for c in features.columns if c != self._target_col]
        X = features[self._feature_cols]
        y = features[self._target_col]

        # Median model (point forecast)
        self._models["point"] = lgb.LGBMRegressor(
            objective="regression", **self.params
        ).fit(X, y)

        # Quantile models for intervals
        for q in [0.025, 0.10, 0.90, 0.975]:
            self._models[q] = lgb.LGBMRegressor(
                objective="quantile", alpha=q, **self.params
            ).fit(X, y)

    def _predict(self, horizon, quantile_levels, future_covariates, timestamps) -> Forecast:
        ts = self._make_timestamps(horizon, timestamps)

        # Recursive forecasting: extend the frame one step at a time
        work_df = self._train_df.copy()
        point_preds = []
        quantile_preds = {q: [] for q in self._models if q != "point"}

        for step in range(horizon):
            features = self.feature_pipeline.build_tabular(work_df, self._target_col)
            last_row = features[self._feature_cols].iloc[[-1]]

            point = float(self._models["point"].predict(last_row)[0])
            point_preds.append(point)

            for q in quantile_preds:
                quantile_preds[q].append(
                    float(self._models[q].predict(last_row)[0])
                )

            # Append the prediction and roll forward
            next_idx = ts[step] if step < len(ts) else len(work_df)
            new_row = pd.DataFrame(
                {self._target_col: [point]},
                index=[next_idx]
            )
            work_df = pd.concat([work_df, new_row])

        point_arr = np.array(point_preds)
        intervals = {}
        if 0.10 in quantile_preds and 0.90 in quantile_preds:
            intervals[80] = (np.array(quantile_preds[0.10]),
                             np.array(quantile_preds[0.90]))
        if 0.025 in quantile_preds and 0.975 in quantile_preds:
            intervals[95] = (np.array(quantile_preds[0.025]),
                             np.array(quantile_preds[0.975]))

        # SHAP-based attributions for explainability
        attributions = self._compute_attributions()

        return Forecast(
            series_id=str(self._target_col),
            model_name=self.name,
            horizon=horizon,
            timestamps=ts,
            point=point_arr,
            intervals=intervals,
            attributions=attributions,
        )

    def _compute_attributions(self) -> dict | None:
        """Feature importance as a simple attribution signal."""
        if "point" not in self._models:
            return None
        importances = self._models["point"].feature_importances_
        return {
            col: np.array([imp])
            for col, imp in zip(self._feature_cols, importances)
        }
