"""Build a complete current SKU assessment for all 28 FMN assessment SKUs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.assess import rank_attention, attention_summary, _driver_objects
from src.risk import RiskConfig, assess_sku, expected_replenishment, robust_daily_uncertainty, project_inventory, classify_risk
from src.schemas import RiskDriver, SkuAssessment

ROOT = Path(__file__).resolve().parent
DF = pd.read_csv(ROOT / "data/processed/clean_daily_data.csv", parse_dates=["date"]).sort_values(["sku_id", "date"])
RISK = pd.read_csv(ROOT / "artifacts/evaluation/risk_current_assessment.csv")
ORIGIN = DF["date"].max()

rows = []
for sku_id, g in DF.groupby("sku_id", sort=True):
    hist = g[g.date <= ORIGIN].sort_values("date")
    last = hist.iloc[-1]
    status = str(last["sku_status"])
    lead_values = sorted(hist["lead_time_days"].dropna().astype(int).unique().tolist())
    lead_range = f"{min(lead_values)}–{max(lead_values)} days" if lead_values else None

    existing = RISK[RISK.sku_id == sku_id]
    if not existing.empty:
        rec = existing.iloc[0].to_dict()
        rec["category"] = last["category"]
        rec["sku_status"] = status
        rec["planning_horizon_days"] = int(rec["lead_time_days"])
        rec["lead_time_range_days"] = lead_range
        rows.append(rec)
        continue

    demand = hist["units_sold"].dropna().astype(float)
    if status == "new":
        planning_horizon = 14
        lead_time = int(round(float(last["lead_time_days"])))
        daily = float(demand.tail(12).mean())
        uncertainty = float(demand.tail(12).std(ddof=1)) if len(demand.tail(12)) >= 3 else 0.0
        expected_date, expected_qty, cadence = expected_replenishment(hist, ORIGIN)
        stock = float(last["closing_stock"])
        safety = 0.5 * uncertainty * np.sqrt(planning_horizon)
        # New SKUs have no reliable replenishment lead time, so use the 14 day planning horizon.
        projection = project_inventory(stock, daily, planning_horizon, expected_date, expected_qty, uncertainty, 0.5, ORIGIN)
        state = "Critical" if stock <= 0 or (projection["projected_stock"] <= 0).any() else "Watch"
        positive = projection.loc[projection["projected_stock"] <= 0]
        days_stockout = float((positive.iloc[0]["date"] - ORIGIN).days) if not positive.empty else np.nan
        min_projected = float(projection["projected_stock"].min())
        unmet = max(0.0, -min_projected)
        coverage = stock / daily if daily > 0 else np.inf
        drivers = ["Limited SKU history", "Lead time inconsistency"]
        if stock <= 0:
            drivers.insert(0, "Low stock")
        if coverage < planning_horizon:
            drivers.append("Short stock coverage")
        drivers.append("Forecast uncertainty")
        recommendation = "Review replenishment immediately; new SKU has limited history and unreliable lead time." if state == "Critical" else "Monitor closely and confirm replenishment timing."
        rows.append({
            "sku_id": sku_id, "date": ORIGIN, "category": last["category"], "sku_status": status,
            "current_stock": stock, "lead_time_days": lead_time, "planning_horizon_days": planning_horizon,
            "lead_time_range_days": lead_range, "forecast_daily_demand": daily,
            "forecast_lead_time_demand": daily * planning_horizon, "uncertainty_daily": uncertainty,
            "safety_buffer": safety, "coverage_days": coverage, "expected_delivery_date": expected_date,
            "expected_receipt_qty": expected_qty, "cadence_days": cadence, "risk_state": state,
            "days_to_projected_stockout": days_stockout, "projected_unmet_units": unmet,
            "minimum_projected_stock": min_projected, "drivers": drivers, "recommendation": recommendation,
            "forecast_method": "12 day cold start baseline",
        })

# 1017 has one unresolved demand value in its recent 28 day history, so use an explicit fallback
# based on available recent observations instead of silently fabricating the missing demand.
if "SKU-1017" not in RISK.sku_id.values:
    sku_id = "SKU-1017"
    g = DF[DF.sku_id == sku_id].sort_values("date")
    hist = g[g.date <= ORIGIN]
    d = hist.units_sold.dropna().astype(float).tail(28)
    daily = float(d.mean())
    uncertainty = float(d.std(ddof=1))
    lead = int(hist.iloc[-1].lead_time_days)
    stock = float(hist.iloc[-1].closing_stock)
    expected_date, expected_qty, cadence = expected_replenishment(hist, ORIGIN)
    safety = 0.5 * uncertainty * np.sqrt(lead)
    projection = project_inventory(stock, daily, lead + 3, expected_date, expected_qty, uncertainty, 0.5, ORIGIN)
    state = classify_risk(
        current_stock=stock,
        projected=projection,
        safety_buffer=safety,
        cycle_demand=daily * lead,
        overstock_multiplier=1.5,
        review_window_days=3,
    )
    pos = projection.loc[projection.projected_stock <= 0]
    days_stockout = float((pos.iloc[0].date - ORIGIN).days) if not pos.empty else np.nan
    minp = float(projection.projected_stock.min())
    rows.append({
        "sku_id": sku_id, "date": ORIGIN, "category": hist.iloc[-1].category, "sku_status": "established",
        "current_stock": stock, "lead_time_days": lead, "planning_horizon_days": lead,
        "lead_time_range_days": "3 days", "forecast_daily_demand": daily,
        "forecast_lead_time_demand": daily * lead, "uncertainty_daily": uncertainty,
        "safety_buffer": safety, "coverage_days": stock / daily, "expected_delivery_date": expected_date,
        "expected_receipt_qty": expected_qty, "cadence_days": cadence, "risk_state": state,
        "days_to_projected_stockout": days_stockout, "projected_unmet_units": max(0,-minp),
        "minimum_projected_stock": minp, "drivers": ["Forecast uncertainty"],
        "recommendation": "Monitor closely." if state == "Healthy" else "Review replenishment and consider expediting.",
        "forecast_method": "28 day available history fallback",
    })

base = pd.DataFrame(rows)
# Build deterministic, evidence carrying drivers before serialising the canonical contract.
base["history_days"] = base["sku_id"].map(DF.groupby("sku_id")["date"].nunique()).fillna(0).astype(int)
base["data_quality_flags"] = base.apply(lambda r: [x for x in [
    "lead_time_inconsistent" if "–" in str(r.get("lead_time_range_days", "")) else None,
    "limited_history" if r["sku_status"] == "new" else None,
] if x], axis=1)

def _derive_driver_labels(row: pd.Series) -> list[str]:
    """Derive planner facing risk drivers from deterministic assessment fields."""
    labels = list(row["drivers"]) if isinstance(row.get("drivers"), list) else []
    if float(row["current_stock"]) <= 0 and "Low stock" not in labels:
        labels.insert(0, "Low stock")
    if float(row.get("projected_unmet_units", 0)) > 0 and "Projected shortage" not in labels:
        labels.append("Projected shortage")
    coverage = float(row.get("coverage_days", np.inf))
    lead = float(row.get("lead_time_days", 0))
    if coverage < lead and "Short stock coverage" not in labels:
        labels.append("Short stock coverage")
    if float(row.get("uncertainty_daily", 0)) > 0 and "Forecast uncertainty" not in labels:
        labels.append("Forecast uncertainty")
    lead_range = str(row.get("lead_time_range_days", ""))
    if "–" in lead_range:
        parts = lead_range.split("–", 1)
        try:
            lead_min = float(parts[0].strip())
            lead_max = float(parts[1].split()[0].strip())
        except (ValueError, IndexError):
            lead_min = lead_max = float("nan")
        if pd.notna(lead_min) and pd.notna(lead_max) and lead_min != lead_max and "Lead time inconsistency" not in labels:
            labels.append("Lead time inconsistency")
    if row.get("sku_status") == "new" and "Limited SKU history" not in labels:
        labels.append("Limited SKU history")
    expected = row.get("expected_delivery_date")
    days_stockout = row.get("days_to_projected_stockout")
    if pd.notna(expected) and pd.notna(days_stockout):
        delivery_days = (pd.Timestamp(expected) - ORIGIN).days
        if float(days_stockout) < delivery_days and "Expected delivery timing" not in labels:
            labels.append("Expected delivery timing")
    if row.get("risk_state") == "Overstock" and "Excess projected stock" not in labels:
        labels.append("Excess projected stock")
    return labels

base["drivers"] = base.apply(_derive_driver_labels, axis=1)

# Direct construction keeps the canonical contract independent of the earlier risk CSV schema.
contracts=[]
for _,r in base.iterrows():
    drv=_driver_objects(r)
    contracts.append(SkuAssessment(
        sku_id=str(r.sku_id), date=str(r.date), category=str(r.category), sku_type=str(r.sku_status),
        current_stock=float(r.current_stock), lead_time_days=int(r.lead_time_days), planning_horizon_days=int(r.planning_horizon_days),
        lead_time_range_days=None if pd.isna(r.lead_time_range_days) else str(r.lead_time_range_days),
        forecast_daily_demand=float(r.forecast_daily_demand), forecast_lead_time_demand=float(r.forecast_lead_time_demand),
        uncertainty_daily=float(r.uncertainty_daily), safety_buffer=float(r.safety_buffer), coverage_days=float(r.coverage_days),
        expected_delivery_date=None if pd.isna(r.expected_delivery_date) else str(r.expected_delivery_date),
        expected_receipt_qty=float(r.expected_receipt_qty), cadence_days=None if pd.isna(r.cadence_days) else float(r.cadence_days),
        risk_state=str(r.risk_state), days_to_projected_stockout=None if pd.isna(r.days_to_projected_stockout) else float(r.days_to_projected_stockout),
        projected_unmet_units=float(r.projected_unmet_units), minimum_projected_stock=float(r.minimum_projected_stock), drivers=drv,
        recommendation=str(r.recommendation), attention_priority="P4", attention_score=0.0, data_quality_flags=r.data_quality_flags,
        forecast_method=str(r.forecast_method)).to_dict())
final=pd.DataFrame(contracts)
final["attention_priority"]=final.risk_state.map({"Critical":"P1","Watch":"P2","Overstock":"P3","Healthy":"P4"})
# Score using the same business ordering logic
from src.assess import _priority_score
final["attention_score"]=final.apply(_priority_score,axis=1)
final=rank_attention(final)
for c in ["drivers","data_quality_flags"]:
    final[c]=final[c].apply(lambda xs: json.dumps(xs))
out=ROOT/"artifacts/evaluation"
final.to_csv(out/"sku_assessments.csv",index=False)
final[final.risk_state!="Healthy"].to_csv(out/"attention_queue.csv",index=False)
(out/"attention_summary.json").write_text(json.dumps(attention_summary(final),indent=2))
print(attention_summary(final))
print(final[["attention_rank","sku_id","sku_type","risk_state","attention_priority","current_stock","planning_horizon_days","lead_time_range_days"]].to_string(index=False))
