"""Generate plain language, grounded explanations for SKU inventory risk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.llm.client import LLMClient, LLMUnavailableError
from src.llm.grounding import check_numbers, collect_numbers


SYSTEM_PROMPT = """You explain inventory risk to a Supply Chain planner.

The planner is not expected to understand machine learning, forecasting
algorithms, or statistical terminology. Your job is to translate the
supplied inventory assessment into clear, practical business language.

The deterministic inventory assessment has already decided the risk state,
priority, projected impact, and recommended action. Do not change or
reinterpret those decisions.

Rules:
- Use only the facts explicitly provided.
- Never invent facts, causes, predictions, quantities, dates, or actions.
- Never calculate a new number from the supplied facts.
- Never introduce a number simply because it can be calculated from other
  numbers.
- Use only numbers explicitly present in the supplied facts.
- Do not expose internal field names, model names, algorithms, feature names,
  WAPE, statistical measures, internal scores, or implementation details.
- Do not describe forecast uncertainty using technical metrics such as
  standard deviation or demand variability. If the facts indicate limited
  history or another source of uncertainty, explain it in simple business
  language.
- Refer to the SKU by its ID, for example SKU-1010.
- Use natural business language that a Supply Chain planner can understand
  without technical knowledge.
- Use natural dates where dates are provided.
- Use presentation-ready numbers exactly as supplied. Do not calculate or
  round numbers yourself.
- Keep the explanation to 3 or 4 short sentences.
- Be direct, practical, and decision oriented.
- Do not use headings, bullets, bold text, or markdown formatting.

Use the risk state to determine the explanation:

Critical:
Clearly explain the immediate or projected problem, when it may occur,
the operational impact if provided, and what the planner should investigate
or do next.

Watch:
Explain that the SKU requires attention because conditions are moving
toward a potential problem. Do not describe it as an immediate stockout
unless the supplied facts explicitly say so. State what the planner should
monitor or investigate.

Overstock:
Explain that inventory is higher than needed based on the assessment and
why this may require attention. Focus on excess inventory and the next
replenishment decision. Do not invent financial values because no unit cost
may be available.

Healthy:
Do not manufacture a problem or create an unnecessary reason for attention.
Clearly state that no immediate inventory action is required when the facts
support that conclusion. If a material data quality issue is present, it may
be mentioned separately without implying that the SKU has an inventory risk.

New SKU:
If the facts identify the SKU as new or indicate limited history, explain
that the limited history makes the forecast less certain. Keep this
explanation simple and operational.

The explanation should help the planner answer four questions:
1. What is happening?
2. Why does this SKU need attention, if it does?
3. When could the problem occur or what is the projected impact?
4. What should the planner investigate or do next?
"""


MAX_ATTEMPTS = 2


@dataclass
class ExplanationResult:
    """Result of an LLM explanation and its grounding check."""

    text: str | None
    passed_check: bool
    unmatched_numbers: list[float] = field(default_factory=list)
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    error: str | None = None


def _prepare_value(value: Any) -> Any:
    """Convert raw assessment values into planner friendly display values."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, float):
        if value != value:
            return None

        if abs(value - round(value)) < 1e-9:
            return int(round(value))

        return round(value, 1)

    if isinstance(value, dict):
        return {
            key: _prepare_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_prepare_value(item) for item in value]

    if isinstance(value, tuple):
        return [_prepare_value(item) for item in value]

    return value


def _prepare_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Prepare structured assessment facts for plain language generation."""
    return {
        key: _prepare_value(value)
        for key, value in facts.items()
    }


def explain_sku(
    client: LLMClient,
    sku_id: str,
    facts: dict[str, Any],
) -> ExplanationResult:
    """Generate a grounded plain language explanation for one SKU assessment."""
    prepared_facts = _prepare_facts(facts)
    allowed = collect_numbers(prepared_facts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"SKU: {sku_id}\n"
                "Assessment facts:\n"
                f"{json.dumps(prepared_facts, indent=2)}"
            ),
        },
    ]

    last_reply = None
    last_unmatched: list[float] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            reply = client.chat(messages)
        except LLMUnavailableError as exc:
            return ExplanationResult(
                text=None,
                passed_check=False,
                attempts=attempt,
                error=str(exc),
            )

        last_reply = reply
        passed, unmatched = check_numbers(reply.content, allowed)

        if passed:
            return ExplanationResult(
                text=reply.content,
                passed_check=True,
                attempts=attempt,
                provider=reply.provider,
                model=reply.model,
                latency_ms=reply.latency_ms,
            )

        last_unmatched = unmatched

        messages.extend(
            [
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": (
                        f"These numbers are not explicitly present in the "
                        f"supplied facts: {unmatched}. Rewrite the explanation "
                        "without calculating, estimating, or introducing any "
                        "new numbers. Keep it in plain language for a Supply "
                        "Chain planner."
                    ),
                },
            ]
        )

    return ExplanationResult(
        text=None,
        passed_check=False,
        unmatched_numbers=last_unmatched,
        attempts=MAX_ATTEMPTS,
        provider=last_reply.provider if last_reply else None,
        model=last_reply.model if last_reply else None,
        latency_ms=last_reply.latency_ms if last_reply else None,
        error="LLM response failed numerical grounding after retry.",
    )