"""The agent loop shared by the orchestrator and both specialists.

An "agent" here is nothing more than this loop:

  1. Send Claude the conversation plus a list of tool descriptions.
  2. If Claude answers in text, we are done.
  3. If Claude asks for tools, run them, append the results, and go to 1.

Everything else in this file is about making that loop safe to run unattended:
a turn limit, tool errors that go back to Claude instead of crashing, every
stop reason handled, and a trace of what happened.
"""

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

import anthropic

from agents.client import get_client

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
MAX_TOKENS = 16000

# A loop that never ends is the classic agent failure. Each pass through the
# loop is one model call, so this caps both runtime and the visitor's spend.
MAX_TURNS = 12

# How much of each tool result to keep in the trace shown to the user. Claude
# always gets the full result; this only shortens the copy we display.
TRACE_PREVIEW_CHARS = 300


# --- Trace ---
# One list per chat request, shared by every agent that runs inside it, so the
# orchestrator's steps and a specialist's steps land in one timeline. A
# ContextVar keeps concurrent requests from writing into each other's trace.
_trace: ContextVar[list | None] = ContextVar("agent_trace", default=None)


@contextmanager
def collect_trace():
    events: list = []
    token = _trace.set(events)
    try:
        yield events
    finally:
        _trace.reset(token)


def _record(event: dict) -> None:
    events = _trace.get()
    if events is not None:
        events.append(event)


def _text(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def run_agent(
    name: str,
    system: str,
    tools: list,
    execute_tool: Callable[[str, dict], str],
    messages: list,
) -> str:
    """Run one agent to completion and return its final text.

    `messages` is the conversation so far and must end with a user turn.
    `execute_tool(tool_name, tool_input)` runs one tool and returns a string.
    """
    messages = list(messages)  # the loop appends; don't mutate the caller's list

    for _ in range(MAX_TURNS):
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            messages=messages,
        )
        _record({
            "type": "model_call",
            "agent": name,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        if response.stop_reason == "tool_use":
            # The whole assistant turn goes back unchanged, not just the tool
            # calls: it can also hold text and thinking blocks, and Claude
            # needs to see its own turn exactly as it wrote it.
            messages.append({"role": "assistant", "content": response.content})

            # Claude may ask for several tools at once. Every request needs a
            # result, and all results go back together in one user turn.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                # Recorded before the tool runs, then filled in, so a delegation
                # appears in the trace ahead of the steps its specialist takes.
                event = {"type": "tool_call", "agent": name, "tool": block.name, "input": block.input}
                _record(event)
                started = time.perf_counter()
                is_error = False
                try:
                    result = str(execute_tool(block.name, block.input))
                except anthropic.APIError:
                    # A delegated agent's own API failure (bad key, rate limit)
                    # is not something Claude can work around, so let it surface.
                    raise
                except Exception as e:
                    # A failed tool is information, not a crash. Claude sees
                    # the error and can retry with different input or explain.
                    result = f"Error running {block.name}: {type(e).__name__}: {e}"
                    is_error = True
                event.update(
                    is_error=is_error,
                    ms=round((time.perf_counter() - started) * 1000),
                    result_preview=result[:TRACE_PREVIEW_CHARS],
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        text = _text(response)
        if response.stop_reason == "end_turn":
            return text or "No response generated."
        if response.stop_reason == "max_tokens":
            return (text + "\n\n" if text else "") + "(This answer was cut off at the length limit.)"
        if response.stop_reason == "refusal":
            return text or "Claude declined this request."
        return text or f"Stopped unexpectedly ({response.stop_reason})."

    return f"The {name} agent stopped after {MAX_TURNS} steps without finishing. Try a narrower request."
