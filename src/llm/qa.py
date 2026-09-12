"""Grounded question answering over structured inventory tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm.client import ChatReply, LLMClient, LLMUnavailableError
from src.llm.grounding import check_numbers, collect_numbers

SYSTEM_PROMPT = """You answer questions about SKU inventory for a supply chain planner.

Rules:
- Always call the available tools to obtain data. Never answer from memory or guess.
- Quote numbers exactly as the tools return them, without thousands separators.
- Refer to SKUs by their ID, for example SKU-1010.
- Use plain business language and never expose internal field names.
- Mention relevant notes such as limited history or data quality issues.
- If a tool says a SKU was not found, say so clearly.
- If the tools cannot answer the question, explain what the available data covers.
- Answer in 1 to 4 short sentences."""

MAX_ROUNDS = 3


@dataclass
class ToolCallRecord:
    """Record of one deterministic tool call used to answer a question."""

    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class QAResult:
    """Grounded answer plus the evidence used to produce it."""

    answer: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    passed_check: bool = False
    unmatched_numbers: list[float] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    error: str | None = None


def answer_question(
    client: LLMClient,
    question: str,
    tool_schemas: list[dict[str, Any]],
    functions: dict[str, Callable[..., Any]],
) -> QAResult:
    """Answer a question through deterministic tools and verify numeric grounding."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    calls: list[ToolCallRecord] = []
    last_reply: ChatReply | None = None

    for _ in range(MAX_ROUNDS):
        try:
            reply = client.chat(messages, tools=tool_schemas)
        except LLMUnavailableError as exc:
            return QAResult(None, calls, error=str(exc))

        last_reply = reply
        if not reply.tool_calls:
            allowed = collect_numbers(
                [question] + [[call.arguments, call.result] for call in calls]
            )
            passed, unmatched = check_numbers(reply.content, allowed)
            return QAResult(
                answer=reply.content if passed else None,
                tool_calls=calls,
                passed_check=passed,
                unmatched_numbers=unmatched,
                provider=reply.provider,
                model=reply.model,
                error=None if passed else "Answer failed numerical grounding.",
            )

        messages.append(assistant_message(reply))
        for call in reply.tool_calls:
            arguments, result = run_tool(call, functions)
            calls.append(
                ToolCallRecord(call.function.name, arguments, result)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    return QAResult(
        answer=None,
        tool_calls=calls,
        provider=last_reply.provider if last_reply else None,
        model=last_reply.model if last_reply else None,
        error=f"No answer after {MAX_ROUNDS} tool rounds.",
    )


def assistant_message(reply: ChatReply) -> dict[str, Any]:
    """Convert an SDK response into the assistant message required for tool calls."""
    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [
            call.model_dump(exclude_none=True) for call in reply.tool_calls
        ],
    }


def run_tool(
    call: Any,
    functions: dict[str, Callable[..., Any]],
) -> tuple[dict[str, Any], Any]:
    """Safely execute a requested deterministic tool and return its evidence."""
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}, {"error": "Arguments were not valid JSON."}

    name = call.function.name
    if name not in functions:
        return arguments, {"error": f"Unknown tool: {name}"}

    try:
        return arguments, functions[name](**arguments)
    except TypeError as exc:
        return arguments, {"error": str(exc)}
    except Exception as exc:  # tool errors become model-visible evidence
        return arguments, {"error": f"Tool failed: {exc}"}
