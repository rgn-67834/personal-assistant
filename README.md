# Personal Assistant

A small multi-agent chat app built on the Claude API. An orchestrator reads each
message and either answers directly or hands it to a specialist:

- **Finance agent**: market data via yfinance, financial statements, risk
  metrics (beta, alpha, Sharpe, CAPM, WACC), NPV, IRR, CAGR.
- **Lawyer agent**: patent searches across USPTO and Google Patents, patent
  details, portfolio comparisons.

Backend is FastAPI; the frontend is one static HTML page.

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
