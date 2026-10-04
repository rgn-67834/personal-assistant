import os
from dotenv import load_dotenv
import anthropic
from agents.finance import run_finance_agent
from agents.lawyer import run_lawyer_agent

load_dotenv()

from agents.client import get_client

SYSTEM_PROMPT = """You are a personal assistant with access to two specialized agents:
a finance agent and a lawyer/patent agent.

Route requests as follows:
- Finance agent: statistical analysis, financial modeling, corporate finance, stock prices,
  financial statements, risk metrics (beta, alpha, Sharpe ratio, WACC, CAPM), market data.
- Lawyer agent: patent searches, intellectual property, patent portfolios, USPTO, EPO,
  Japan/Korea/Taiwan patent databases, IP strategy, patent details.

For everything else, answer directly and conversationally.

Always delegate to the appropriate agent — do not try to answer finance or patent
questions yourself."""

tools = [
    {
        "name": "delegate_to_finance",
        "description": "Send a request to the finance agent for financial analysis, market data, risk metrics, or corporate finance tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "The full request to pass to the finance agent."}
            },
            "required": ["request"]
        }
    },
    {
        "name": "delegate_to_lawyer",
        "description": "Send a request to the lawyer/patent agent for patent searches, IP analysis, or patent portfolio queries across USPTO, EPO, Japan, Korea, and Taiwan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "The full request to pass to the lawyer agent."}
            },
            "required": ["request"]
        }
    }
]

def run_orchestrator(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = get_client().messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Return the final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "delegate_to_finance":
                        result = run_finance_agent(block.input["request"])
                    elif block.name == "delegate_to_lawyer":
                        result = run_lawyer_agent(block.input["request"])
                    else:
                        result = "Unknown tool."

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})
