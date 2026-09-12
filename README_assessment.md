# Project 1 — SKU Assessment and Attention Ranking

This stage creates the canonical `SkuAssessment` contract and deterministic planner attention queue for all 28 assessment SKUs.

## Outputs
- `artifacts/evaluation/sku_assessments.csv`
- `artifacts/evaluation/attention_queue.csv`
- `artifacts/evaluation/attention_summary.json`
- `notebooks/05_sku_assessment_attention.ipynb`
- `src/schemas.py`
- `src/assess.py`

## Current summary
- 28 SKUs covered
- 10 Critical
- 6 Overstock
- 12 Healthy
- 0 Watch in the current as of 29 June 2026 snapshot

## Design rule
Risk and attention ranking are deterministic. The LLM is not allowed to calculate risk, change ranking, or invent evidence.

## Cold start
New SKUs use observed history with a 14 day planning horizon and explicitly surface limited history and lead time inconsistency. SKU 1017 uses an available history fallback because one unresolved demand observation prevents the complete LightGBM feature row.

## Next stage
Build the Streamlit Attention Center and SKU Drilldown using this contract. Then add grounded AI explanations and structured Q&A.
