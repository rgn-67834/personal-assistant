# Personal Assistant

A small multi-agent chat app built on the Claude API. An orchestrator reads each
message and either answers directly or hands it to a specialist:

- **Finance agent**: market data via yfinance, financial statements, risk
  metrics (beta, alpha, Sharpe, CAPM, WACC), NPV, IRR, CAGR.
- **Lawyer agent**: patent searches across USPTO and Google Patents, patent
  details, portfolio comparisons.

Backend is FastAPI; the frontend is one static HTML page.

**Live demo:** <https://claude-assistant.up.railway.app>
(you need your own Anthropic API key, see below)

## Try asking

- "Compare Apple's and Microsoft's beta, Sharpe ratio and max drawdown over the
  last three years."
- "What is the NPV of -1000, 300, 400, 500, 600 at a 9% discount rate?"
- "Find recent Tesla patents on battery thermal management."

## How it works

There is no agent framework here. All three agents run the same loop, written
out in [`agents/loop.py`](agents/loop.py):

1. Send Claude the conversation and a list of tool descriptions.
2. If Claude answers in text, stop.
3. If Claude asks for tools, run them, append the results, and repeat.

What differs between agents is only the system prompt and the tools:

- `agents/orchestrator.py` has two tools, `delegate_to_finance` and
  `delegate_to_lawyer`. Claude decides whether to answer or hand off.
- `agents/finance.py` gives Claude 21 tools for statements, prices, risk
  metrics and valuation math.
- `agents/lawyer.py` gives Claude 6 tools for patent search, details, portfolio
  comparison and CSV export.

```
you ── orchestrator ──┬── delegate_to_finance ── finance agent ── 21 tools
                      └── delegate_to_lawyer ─── lawyer agent ─── 6 tools
```

Under each answer, the page shows the trace: which agent ran, each tool call
with its input and timing, and the tokens used.

## Design decisions

- **Delegation is tool use.** To the orchestrator, a specialist is one tool
  that takes a request and returns text. Multi-agent needs no extra machinery.
- **Specialists get a blank conversation.** Only the orchestrator sees the chat
  history. Each specialist reads just the request written for it, which keeps
  its context small and focused, so the orchestrator is prompted to write
  self-contained requests.
- **Tool errors go back to Claude.** A tool that raises returns a result marked
  `is_error` instead of crashing the request, so Claude can retry with
  different input or explain what went wrong.
- **The loop always ends.** There is a turn limit, and every stop reason is
  handled (finished, wants tools, cut off at the length limit, declined).
- **Parallel tool calls are answered together.** When Claude asks for several
  tools in one turn, all the results go back in a single message.
- **The server keeps no state.** The browser holds the chat history and sends
  the most recent turns with each message, which also suits bring-your-own-key.

## Tests

The loop is tested against a scripted fake of the Claude client, so the tests
need no API key and cost nothing:

```bash
pip install pytest
pytest
```

They cover the tool round trip, parallel calls, a failing tool, the turn limit,
truncated and declined answers, delegation, and the chat endpoint.

Built with [Claude Code](https://claude.com/claude-code). It grew out of a
one-tool calculator agent I wrote first to learn the loop.

## Bring your own key

The server holds no Anthropic API key. Each visitor pastes their own key into
the page. It is kept in that browser tab's `sessionStorage`, sent with each
request in the `X-Anthropic-Key` header, used for that request, and never
stored or logged. There is no fallback to a server-side key, so hosting this
publicly cannot spend the host's credits.

Get a key at <https://console.anthropic.com/settings/keys>.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://localhost:8000> and paste your key.

## Deploy on Railway

Create a project from this repo. The `Procfile` supplies the start command.
Do not set `ANTHROPIC_API_KEY`; the app does not read it.

`SERPAPI_KEY` is optional and turns on Google Patents search. It is a
server-side key, so on a public deployment visitors' searches use the host's
SerpAPI quota. Leave it unset to keep the deployment free of any shared
credentials; USPTO search still works without it.
