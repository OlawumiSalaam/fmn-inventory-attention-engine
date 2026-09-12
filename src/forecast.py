"""Forecasting utilities for FMN Project 1.

The module implements the leakage-controlled pooled LightGBM formulation
used in the baseline/model comparison notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_14",
    "roll_std_28",
    "trend_ratio_14_28",
    "day_of_week",
    "lead_time_days",
    "category_code",
]


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the pooled LightGBM demand-ratio model."""

    n_estimators: int = 200
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 20
    random_state: int = 42


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate weighted absolute percentage error."""
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator)


def build_supervised_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-controlled supervised rows for lead-time demand ratios.

    Only established SKUs with a fixed lead time are used. A training row
    requires 28 complete historical demand observations and complete demand
    over the subsequent lead-time horizon.
    """
    rows: list[dict] = []

    for sku_id, group in data.groupby("sku_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        valid_lead_time = group["lead_time_days"].dropna()
        if valid_lead_time.empty:
            continue

        lead_time = int(round(valid_lead_time.mode().iloc[0]))

        for origin_idx in range(28, len(group) - lead_time):
            history = group.loc[:origin_idx, "units_sold"].astype(float)
            future = group.loc[
                origin_idx + 1 : origin_idx + lead_time, "units_sold"
            ].astype(float)

            if history.tail(28).isna().any() or future.isna().any():
                continue

            baseline = history.tail(28).mean()
            if pd.isna(baseline) or baseline <= 0:
                continue

            rows.append(
                {
                    "sku_id": sku_id,
                    "origin_date": group.loc[origin_idx, "date"],
                    "category": group.loc[origin_idx, "category"],
                    "lag_1": history.iloc[-1],
                    "lag_7": history.iloc[-7],
                    "lag_14": history.iloc[-14],
                    "roll_mean_7": history.tail(7).mean(),
                    "roll_mean_14": history.tail(14).mean(),
                    "roll_mean_28": baseline,
                    "roll_std_7": history.tail(7).std(),
                    "roll_std_14": history.tail(14).std(),
                    "roll_std_28": history.tail(28).std(),
                    "trend_ratio_14_28": history.tail(14).mean() / baseline,
                    "day_of_week": group.loc[origin_idx, "date"].dayofweek,
                    "lead_time_days": lead_time,
                    "target_ratio": future.sum() / (baseline * lead_time),
                    "actual_lead_demand": future.sum(),
                    "baseline_28": baseline * lead_time,
                    "lead_time": lead_time,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["category_code"] = result["category"].astype("category").cat.codes
    return result


def fit_model(training_data: pd.DataFrame, config: ForecastConfig | None = None) -> LGBMRegressor:
    """Fit the pooled LightGBM demand-ratio model."""
    config = config or ForecastConfig()
    model = LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(training_data[FEATURES], training_data["target_ratio"])
    return model


def predict_lead_time_demand(
    model: LGBMRegressor, features: pd.DataFrame
) -> pd.Series:
    """Predict non-negative demand over each row's SKU lead-time horizon."""
    ratio = np.clip(model.predict(features[FEATURES]), 0, None)
    return pd.Series(ratio * features["baseline_28"].to_numpy(), index=features.index)
