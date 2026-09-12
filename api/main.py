"""FastAPI application for the FMN Inventory Attention Engine.

The API exposes deterministic inventory analytics and optional grounded LLM
capabilities. Forecasting and risk decisions remain deterministic. The LLM
only explains analytical evidence and answers questions through bounded tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query

from api.llm_tools import build_tool_functions, build_tool_schemas
from api.models import (
    AIExplanationResponse,
    AskRequest,
    AskResponse,
    AttentionResponse,
    ForecastValidationResponse,
    HealthResponse,
    RiskValidationResponse,
    SkuAssessmentResponse,
    SkuProjectionResponse,
    ToolCallResponse,
    ValidationResponse,
)
from api.services import ArtifactStore, get_attention, get_projection, get_sku
from src.llm.client import LLMClient, LLMUnavailableError
from src.llm.explain import explain_sku
from src.llm.qa import answer_question


app = FastAPI(
    title="FMN Inventory Attention Engine API",
    description=(
        "Deterministic inventory assessment services for the FMN Project 1 "
        "decision support application, with optional grounded AI explanation "
        "and structured data question answering."
    ),
    version="0.1.0",
)

store = ArtifactStore()


def _load_llm_client() -> LLMClient | None:
    """Create the configured LLM client when a provider is available."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config: dict[str, Any] = yaml.safe_load(handle) or {}

        llm_config = config.get("llm")

        if not llm_config:
            return None

        return LLMClient.from_config(llm_config)

    except (FileNotFoundError, KeyError, LLMUnavailableError):
        return None


llm_client = _load_llm_client()
tool_schemas = build_tool_schemas()
tool_functions = build_tool_functions(store)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return service health and the assessment data date."""
    assessments = store.assessments()
    data_as_of = str(assessments["date"].iloc[0]).split(" ")[0]

    return HealthResponse(
        status="ok",
        service="fmn-inventory-attention-api",
        data_as_of=data_as_of,
    )


@app.get("/api/v1/summary", response_model=dict, tags=["attention"])
def summary() -> dict:
    """Return current SKU risk counts."""
    return store.summary()


@app.get("/api/v1/attention", response_model=AttentionResponse, tags=["attention"])
def attention(
    risk_state: Literal["Critical", "Watch", "Healthy", "Overstock"] | None = Query(
        default=None
    ),
    category: str | None = Query(default=None),
    sku_type: str | None = Query(default=None),
) -> AttentionResponse:
    """Return the ranked attention queue with optional filters."""
    as_of_date, summary_data, items = get_attention(
        store,
        risk_state,
        category,
        sku_type,
    )

    return AttentionResponse(
        as_of_date=as_of_date,
        summary=summary_data,
        items=items,
    )


@app.get(
    "/api/v1/skus/{sku_id}",
    response_model=SkuAssessmentResponse,
    tags=["skus"],
)
def sku_details(sku_id: str) -> SkuAssessmentResponse:
    """Return the current deterministic assessment for one SKU."""
    assessment = get_sku(store, sku_id)

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail=f"SKU {sku_id} was not found.",
        )

    return SkuAssessmentResponse.model_validate(assessment)


@app.get(
    "/api/v1/skus/{sku_id}/projection",
    response_model=SkuProjectionResponse,
    tags=["skus"],
)
def sku_projection(sku_id: str) -> SkuProjectionResponse:
    """Return the projected inventory trajectory for one SKU."""
    projection = get_projection(store, sku_id)

    if projection is None:
        raise HTTPException(
            status_code=404,
            detail=f"SKU {sku_id} was not found.",
        )

    return SkuProjectionResponse.model_validate(projection)


@app.get(
    "/api/v1/skus/{sku_id}/explanation",
    response_model=AIExplanationResponse,
    tags=["ai"],
)
def sku_explanation(sku_id: str) -> AIExplanationResponse:
    """Generate a grounded natural language explanation for one SKU."""
    assessment = get_sku(store, sku_id)

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail=f"SKU {sku_id} was not found.",
        )

    if llm_client is None:
        return AIExplanationResponse(
            sku_id=sku_id,
            grounding_passed=False,
            attempts=0,
            error=(
                "AI explanation is unavailable because no configured LLM "
                "provider has an API key."
            ),
        )

    result = explain_sku(
        client=llm_client,
        sku_id=sku_id,
        facts=assessment,
    )

    return AIExplanationResponse(
        sku_id=sku_id,
        explanation=result.text,
        grounding_passed=result.passed_check,
        attempts=result.attempts,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        error=result.error,
    )


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    tags=["ai"],
)
def ask_data(request: AskRequest) -> AskResponse:
    """Answer a natural language question using bounded analytical tools."""
    if llm_client is None:
        return AskResponse(
            answer=None,
            grounding_passed=False,
            error=(
                "Ask the Data is unavailable because no configured LLM "
                "provider has an API key."
            ),
        )

    result = answer_question(
        client=llm_client,
        question=request.question,
        tool_schemas=tool_schemas,
        functions=tool_functions,
    )

    tool_calls = [
        ToolCallResponse(
            name=call.name,
            arguments=call.arguments,
            result=call.result,
        )
        for call in result.tool_calls
    ]

    return AskResponse(
        answer=result.answer,
        grounding_passed=result.passed_check,
        tool_calls=tool_calls,
        provider=result.provider,
        model=result.model,
        error=result.error,
    )


@app.get("/api/v1/validation", response_model=ValidationResponse, tags=["validation"])
def validation() -> ValidationResponse:
    """Return forecast and risk validation evidence."""
    forecast = store.forecast_validation().where(
        lambda frame: frame.notna(),
        None,
    )
    risk = store.risk_validation().where(
        lambda frame: frame.notna(),
        None,
    )

    return ValidationResponse(
        forecast=ForecastValidationResponse(
            results=forecast.to_dict(orient="records"),
        ),
        risk=RiskValidationResponse(
            results=risk.to_dict(orient="records"),
        ),
    )