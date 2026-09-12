# FMN Project 1 — Inventory Risk Engine

## Purpose

Convert demand forecasts and replenishment behaviour into a deterministic inventory risk and planner attention signal.

## Current MVP configuration

- As of date: 2026-06-29
- Forecast: pooled LightGBM lead-time demand forecast
- Review window: 3 days
- Uncertainty multiplier: 0.5
- Overstock multiplier: 1.5x replenishment-cycle demand
- Expected delivery: inferred from historical receipt cadence and recent receipt quantities
- No autonomous purchase quantity

## Risk states

- Critical: zero stock or projected stockout before expected replenishment
- Watch: projected stock falls below safety buffer within review window
- Healthy: projected stock is sufficiently protected
- Overstock: projected stock materially exceeds one replenishment cycle plus buffer

## Validation result

Development backtest comparison:

| Method | Recall | Precision | False alert rate | Attention volume |
|---|---:|---:|---:|---:|
| Baseline | 99.2% | 29.8% | 29.2% | 37.0% |
| Candidate, 3 day / 0.5 | 86.2% | 37.5% | 18.0% | 25.6% |

The candidate materially reduces alert burden and false alerts but does not fully meet the approximately 90% recall aspiration. This is intentionally disclosed and should be revisited during final calibration rather than hidden.

## Key runtime behaviour

The engine considers expected delivery timing. A SKU with low current coverage is not automatically Critical when historical replenishment is expected to arrive before projected stockout.

## Next step

Create the shared SKU assessment contract and attention ranking layer, then connect this deterministic output to Streamlit and the grounded AI explanation layer.
