"""Tests for the agent loop, run against a scripted fake of the Claude client.

No API key is needed and nothing is billed: each test hands the loop a fixed
sequence of responses and checks what the loop does with them.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from agents import loop, orchestrator
from agents.client import use_client
from agents.loop import collect_trace, run_agent


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(name, tool_input, id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


def response(stop_reason, *content):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


class FakeClient:
    """Returns the scripted responses in order and records each request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot the messages: the loop keeps appending to the same list
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def run(client, execute_tool=lambda name, tool_input: "ok", messages=None):
    with use_client(client), collect_trace() as trace:
        answer = run_agent(
            name="test",
            system="You are a test agent.",
            tools=[],
            execute_tool=execute_tool,
            messages=messages or [{"role": "user", "content": "hi"}],
        )
    return answer, trace


def test_answers_directly_when_no_tool_is_needed():
    client = FakeClient(response("end_turn", text("Hello.")))
    answer, trace = run(client)
    assert answer == "Hello."
    assert len(client.requests) == 1
    assert [e["type"] for e in trace] == ["model_call"]


def test_runs_a_tool_and_feeds_the_result_back():
    client = FakeClient(
        response("tool_use", text("Let me check."), tool_use("add", {"a": 2, "b": 3})),
        response("end_turn", text("It is 5.")),
    )
    answer, trace = run(client, execute_tool=lambda name, i: i["a"] + i["b"])

    assert answer == "It is 5."
    # The second request carries the assistant turn and the tool result
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "5", "is_error": False}
    ]
    tool_event = next(e for e in trace if e["type"] == "tool_call")
    assert (tool_event["tool"], tool_event["input"], tool_event["is_error"]) == ("add", {"a": 2, "b": 3}, False)


def test_parallel_tool_calls_return_together_in_one_turn():
    client = FakeClient(
        response("tool_use", tool_use("f", {}, id="a"), tool_use("f", {}, id="b")),
        response("end_turn", text("Done.")),
    )
    run(client)
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_a_failing_tool_is_reported_to_claude_instead_of_crashing():
    def broken(name, tool_input):
        raise ValueError("no such ticker")

    client = FakeClient(
        response("tool_use", tool_use("get_beta", {"ticker": "??"})),
        response("end_turn", text("I could not find that ticker.")),
    )
    answer, trace = run(client, execute_tool=broken)

    assert answer == "I could not find that ticker."
    result = client.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "no such ticker" in result["content"]
    assert next(e for e in trace if e["type"] == "tool_call")["is_error"] is True


def test_stops_after_the_turn_limit(monkeypatch):
    monkeypatch.setattr(loop, "MAX_TURNS", 3)
    client = FakeClient(*[response("tool_use", tool_use("f", {})) for _ in range(5)])
    answer, _ = run(client)
    assert len(client.requests) == 3
    assert "stopped after 3 steps" in answer


def test_truncated_and_refused_answers_end_the_loop():
    answer, _ = run(FakeClient(response("max_tokens", text("Partial"))))
    assert answer.startswith("Partial") and "cut off" in answer

    answer, _ = run(FakeClient(response("refusal")))
    assert answer == "Claude declined this request."


def test_orchestrator_delegates_and_specialist_starts_with_a_blank_conversation():
    client = FakeClient(
        # orchestrator decides to delegate
        response("tool_use", tool_use("delegate_to_finance", {"request": "Beta of AAPL over 3y"})),
        # finance agent answers (no tools needed in this script)
        response("end_turn", text("AAPL beta is 1.2.")),
        # orchestrator relays the result
        response("end_turn", text("Apple's beta is 1.2.")),
    )
    history = [
        {"role": "user", "content": "I'm looking at Apple."},
        {"role": "assistant", "content": "Sure, what would you like to know?"},
    ]
    with use_client(client), collect_trace() as trace:
        answer = orchestrator.run_orchestrator("What's its beta?", history)

    assert answer == "Apple's beta is 1.2."
    orchestrator_call, finance_call, _ = client.requests
    # The orchestrator sees the chat history; the specialist sees only the request
    assert len(orchestrator_call["messages"]) == 3
    assert finance_call["messages"] == [{"role": "user", "content": "Beta of AAPL over 3y"}]
    assert [e["agent"] for e in trace if e["type"] == "model_call"] == ["orchestrator", "finance", "orchestrator"]


def test_chat_endpoint_returns_the_trace_and_requires_a_key(monkeypatch):
    client = FakeClient(response("end_turn", text("Hi there.")))
    monkeypatch.setattr(main, "use_api_key", lambda key: use_client(client))
    http = TestClient(main.app)

    assert http.post("/chat", json={"message": "hi"}).status_code == 401

    # A history that opens with an assistant turn is trimmed to start on a user turn
    body = {"message": "hi", "history": [{"role": "assistant", "content": "stray"}]}
    res = http.post("/chat", json=body, headers={"X-Anthropic-Key": "sk-ant-test"})
    assert res.status_code == 200
    assert res.json()["response"] == "Hi there."
    assert res.json()["trace"][0]["agent"] == "orchestrator"
    assert client.requests[0]["messages"] == [{"role": "user", "content": "hi"}]
