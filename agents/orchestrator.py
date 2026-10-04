from agents.finance import run_finance_agent
from agents.lawyer import run_lawyer_agent
from agents.loop import run_agent

SYSTEM_PROMPT = """You are a personal assistant with access to two specialized agents:
a finance agent and a lawyer/patent agent.

Route requests as follows:
- Finance agent: statistical analysis, financial modeling, corporate finance, stock prices,
  financial statements, risk metrics (beta, alpha, Sharpe ratio, WACC, CAPM), market data.
- Lawyer agent: patent searches, intellectual property, patent portfolios, USPTO, EPO,
  Japan/Korea/Taiwan patent databases, IP strategy, patent details.

For everything else, answer directly and conversationally.

Always delegate to the appropriate agent — do not try to answer finance or patent
questions yourself.

The agents cannot see this conversation. Each one starts fresh and reads only the
request you write for it, so make every request self-contained: restate the company,
ticker, time period and any numbers the user gave earlier instead of referring back
to them ("the same company", "that patent")."""

# Delegation is just tool use. To the orchestrator, a whole specialist agent
# looks like one tool that takes a request and returns text.
tools = [
    {
        "name": "delegate_to_finance",
        "description": "Send a request to the finance agent for financial analysis, market data, risk metrics, or corporate finance tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "The full, self-contained request to pass to the finance agent."}
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
                "request": {"type": "string", "description": "The full, self-contained request to pass to the lawyer agent."}
            },
            "required": ["request"]
        }
    }
]


def execute_tool(tool_name, tool_input):
    if tool_name == "delegate_to_finance":
        return run_finance_agent(tool_input["request"])
    if tool_name == "delegate_to_lawyer":
        return run_lawyer_agent(tool_input["request"])
    return "Unknown tool."


def run_orchestrator(user_message: str, history: list | None = None) -> str:
    # `history` is the earlier user/assistant text of this chat. Only the
    # orchestrator gets it; the specialists stay stateless.
    messages = list(history or []) + [{"role": "user", "content": user_message}]
    return run_agent(
        name="orchestrator",
        system=SYSTEM_PROMPT,
        tools=tools,
        execute_tool=execute_tool,
        messages=messages,
    )
