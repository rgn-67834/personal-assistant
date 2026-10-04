import os
import math
from dotenv import load_dotenv
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np

load_dotenv()

from agents.client import get_client

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a corporate finance and statistical analysis assistant.
You have tools to fetch real financial data from public markets and run calculations.

When returning an income statement, balance sheet, or cash flow statement:
- Always present the full data as a clean markdown table with all available years as columns (up to 5 years), with the most recent periods on the right of the chart and the most distant periods to the left of the chart.
- Format all dollar values in billions (e.g. $12.3B) or millions (e.g. $450M) as appropriate
- Include all rows returned by the tool - do not summarize or truncate
- After the table, provide a brief plain-English commentary on the key trends

When returning risk metrics, present results in a clean markdown table where applicable.
Always explain what each metric means and how to interpret it.
When you save a file, tell the user the filename so they can find it."""

tools = [
    {
        "name": "get_income_statement",
        "description": "Fetch the annual income statement for a publicly traded company. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, MSFT."},
                "save_csv": {"type": "boolean", "description": "Whether to save to CSV."}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_balance_sheet",
        "description": "Fetch the annual balance sheet for a publicly traded company. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "save_csv": {"type": "boolean"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_cash_flow",
        "description": "Fetch the annual cash flow statement for a publicly traded company. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "save_csv": {"type": "boolean"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_price_history",
        "description": "Fetch historical stock price data for a company. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
                    "description": "Time period to fetch."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_company_info",
        "description": "Fetch key company information and financial ratios.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_financials",
        "description": "Compare a key financial metric across multiple companies. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string", "description": "e.g. 'Total Revenue', 'Net Income'."},
                "save_csv": {"type": "boolean"}
            },
            "required": ["tickers", "metric"]
        }
    },
    {
        "name": "get_risk_free_rate",
        "description": "Fetch the current risk-free rate from US Treasury yields (3-month and 10-year).",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_beta",
        "description": "Calculate the beta of a stock relative to a benchmark (default S&P 500) over a given period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["3mo", "6mo", "1y", "2y", "3y", "5y"],
                    "description": "Lookback period."
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark ticker, defaults to ^GSPC (S&P 500)."
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_alpha",
        "description": "Calculate Jensen's alpha for a stock vs a benchmark over a given period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["3mo", "6mo", "1y", "2y", "3y", "5y"]
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark ticker, defaults to ^GSPC."
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_volatility",
        "description": "Calculate annualized volatility and variance for a stock over a given period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_sharpe_ratio",
        "description": "Calculate the Sharpe ratio (risk-adjusted return) for a stock over a given period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["3mo", "6mo", "1y", "2y", "3y", "5y"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_capm_expected_return",
        "description": "Calculate the CAPM expected return for a stock using the risk-free rate, beta, and market premium.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["1y", "2y", "3y", "5y"],
                    "description": "Period for beta calculation."
                },
                "market_risk_premium": {
                    "type": "number",
                    "description": "Expected market risk premium as a decimal. Defaults to 0.055 (5.5%) if not provided."
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_correlation_matrix",
        "description": "Calculate the return correlation matrix across multiple stocks over a given period. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "period": {
                    "type": "string",
                    "enum": ["3mo", "6mo", "1y", "2y", "3y", "5y"]
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["tickers", "period"]
        }
    },
    {
        "name": "get_total_return",
        "description": "Calculate the total return for a stock over a given period including price appreciation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_max_drawdown",
        "description": "Calculate the maximum drawdown (peak to trough decline) for a stock over a given period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y"]
                }
            },
            "required": ["ticker", "period"]
        }
    },
    {
        "name": "get_rolling_volatility",
        "description": "Calculate rolling annualized volatility for a stock over a period with a given window. Optionally save to CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["6mo", "1y", "2y", "3y", "5y"]
                },
                "window_days": {
                    "type": "integer",
                    "description": "Rolling window in trading days, e.g. 30, 60, 90."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["ticker", "period", "window_days"]
        }
    },
    {
        "name": "calculate_wacc",
        "description": "Calculate Weighted Average Cost of Capital (WACC) for a company.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Used to fetch market cap, debt, and beta automatically."},
                "cost_of_debt": {"type": "number", "description": "Pre-tax cost of debt as a decimal, e.g. 0.04 for 4%."},
                "tax_rate": {"type": "number", "description": "Corporate tax rate as a decimal, e.g. 0.21 for 21%."},
                "market_risk_premium": {"type": "number", "description": "Market risk premium as a decimal. Defaults to 0.055."},
                "period": {"type": "string", "enum": ["1y", "2y", "3y", "5y"], "description": "Period for beta calculation."}
            },
            "required": ["ticker", "cost_of_debt", "tax_rate"]
        }
    },
    {
        "name": "calculate_cagr",
        "description": "Calculate Compound Annual Growth Rate (CAGR).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_value": {"type": "number"},
                "end_value": {"type": "number"},
                "years": {"type": "number"}
            },
            "required": ["start_value", "end_value", "years"]
        }
    },
    {
        "name": "calculate_npv",
        "description": "Calculate Net Present Value (NPV).",
        "input_schema": {
            "type": "object",
            "properties": {
                "discount_rate": {"type": "number"},
                "cash_flows": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["discount_rate", "cash_flows"]
        }
    },
    {
        "name": "calculate_irr",
        "description": "Estimate Internal Rate of Return (IRR).",
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_investment": {"type": "number"},
                "cash_flows": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["initial_investment", "cash_flows"]
        }
    },
    {
        "name": "calculate_statistics",
        "description": "Calculate descriptive statistics for a list of numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["values"]
        }
    }
]


# --- Helpers ---

def save_df_to_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    return path

def fetch_returns(ticker, period, interval="1d"):
    df = yf.Ticker(ticker.upper()).history(period=period, interval=interval)
    return df["Close"].pct_change().dropna()

def df_to_markdown(df, ticker, label):
    df = df.iloc[:, :5]
    df.columns = [str(col.year) for col in df.columns]
    years = list(df.columns)
    header = "| Metric | " + " | ".join(years) + " |"
    separator = "|---|" + "|".join(["---"] * len(years)) + "|"
    rows = []
    for metric, values in df.iterrows():
        formatted = []
        for v in values:
            if pd.isna(v):
                formatted.append("N/A")
            elif abs(v) >= 1e9:
                formatted.append(f"${v/1e9:.1f}B")
            elif abs(v) >= 1e6:
                formatted.append(f"${v/1e6:.1f}M")
            else:
                formatted.append(f"{v:,.0f}")
        rows.append(f"| {metric} | " + " | ".join(formatted) + " |")
    table = "\n".join([header, separator] + rows)
    return f"**{label} - {ticker.upper()}**\n\n{table}"


# --- Financial statement tools ---

def get_income_statement(ticker, save_csv=False):
    try:
        df = yf.Ticker(ticker.upper()).financials
        if df.empty:
            return f"No income statement data found for {ticker}."
        result = df_to_markdown(df, ticker, "Income Statement")
        if save_csv:
            path = save_df_to_csv(df.iloc[:, :5], f"{ticker.upper()}_income_statement.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_balance_sheet(ticker, save_csv=False):
    try:
        df = yf.Ticker(ticker.upper()).balance_sheet
        if df.empty:
            return f"No balance sheet data found for {ticker}."
        result = df_to_markdown(df, ticker, "Balance Sheet")
        if save_csv:
            path = save_df_to_csv(df.iloc[:, :5], f"{ticker.upper()}_balance_sheet.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_cash_flow(ticker, save_csv=False):
    try:
        df = yf.Ticker(ticker.upper()).cashflow
        if df.empty:
            return f"No cash flow data found for {ticker}."
        result = df_to_markdown(df, ticker, "Cash Flow Statement")
        if save_csv:
            path = save_df_to_csv(df.iloc[:, :5], f"{ticker.upper()}_cash_flow.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_stock_price_history(ticker, period, save_csv=False):
    try:
        df = yf.Ticker(ticker.upper()).history(period=period)
        if df.empty:
            return f"No price history found for {ticker}."
        summary = df[["Open", "High", "Low", "Close", "Volume"]].tail(10).to_string()
        result = f"Stock price history for {ticker.upper()} ({period}), last 10 rows:\n{summary}"
        if save_csv:
            path = save_df_to_csv(df, f"{ticker.upper()}_price_history_{period}.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker.upper()).info
        keys = [
            "longName", "sector", "industry", "country", "marketCap",
            "trailingPE", "forwardPE", "priceToBook", "returnOnEquity",
            "returnOnAssets", "debtToEquity", "revenueGrowth", "earningsGrowth",
            "grossMargins", "operatingMargins", "profitMargins", "currentRatio",
            "totalRevenue", "netIncomeToCommon", "totalDebt"
        ]
        lines = [f"**Company Info - {ticker.upper()}**\n"]
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for key in keys:
            val = info.get(key)
            if val is not None:
                lines.append(f"| {key} | {val} |")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

def compare_financials(tickers, metric, save_csv=False):
    try:
        results = {}
        for ticker in tickers:
            df = yf.Ticker(ticker.upper()).financials
            if metric in df.index:
                results[ticker.upper()] = df.loc[metric]
        comparison_df = pd.DataFrame(results)
        summary = comparison_df.to_string()
        result = f"Comparison of '{metric}':\n{summary}"
        if save_csv:
            path = save_df_to_csv(comparison_df, f"comparison_{'_'.join(t.upper() for t in tickers)}_{metric.replace(' ', '_')}.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# --- Market & rate tools ---

def get_risk_free_rate():
    try:
        t3m = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
        t10y = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1] / 100
        return (
            f"**Current US Treasury Yields (Risk-Free Rates)**\n\n"
            f"| Tenor | Yield |\n|---|---|\n"
            f"| 3-Month T-Bill | {t3m:.3%} |\n"
            f"| 10-Year T-Note | {t10y:.3%} |"
        )
    except Exception as e:
        return f"Error fetching risk-free rate: {str(e)}"


# --- Risk metric tools ---

def get_beta(ticker, period, benchmark="^GSPC"):
    try:
        stock_ret = fetch_returns(ticker, period)
        bench_ret = fetch_returns(benchmark, period)
        combined = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()
        cov = combined.cov().iloc[0, 1]
        var = combined["bench"].var()
        beta = cov / var
        return (
            f"**Beta - {ticker.upper()} vs {benchmark} ({period})**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Beta | {beta:.4f} |\n"
            f"| Benchmark Variance | {var:.6f} |\n"
            f"| Covariance | {cov:.6f} |"
        )
    except Exception as e:
        return f"Error calculating beta: {str(e)}"

def get_alpha(ticker, period, benchmark="^GSPC"):
    try:
        stock_ret = fetch_returns(ticker, period)
        bench_ret = fetch_returns(benchmark, period)
        combined = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()

        # Beta
        cov = combined.cov().iloc[0, 1]
        var = combined["bench"].var()
        beta = cov / var

        # Risk-free rate (annualized 3-month T-bill / 252)
        rf_annual = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
        rf_daily = rf_annual / 252

        # Jensen's alpha (annualized)
        avg_stock = combined["stock"].mean()
        avg_bench = combined["bench"].mean()
        alpha_daily = avg_stock - (rf_daily + beta * (avg_bench - rf_daily))
        alpha_annual = alpha_daily * 252

        return (
            f"**Jensen's Alpha - {ticker.upper()} vs {benchmark} ({period})**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Alpha (annualized) | {alpha_annual:.4%} |\n"
            f"| Beta | {beta:.4f} |\n"
            f"| Avg Daily Return (stock) | {avg_stock:.4%} |\n"
            f"| Avg Daily Return (benchmark) | {avg_bench:.4%} |\n"
            f"| Risk-Free Rate (annual) | {rf_annual:.3%} |"
        )
    except Exception as e:
        return f"Error calculating alpha: {str(e)}"

def get_volatility(ticker, period):
    try:
        returns = fetch_returns(ticker, period)
        daily_vol = returns.std()
        annual_vol = daily_vol * math.sqrt(252)
        daily_var = returns.var()
        annual_var = daily_var * 252
        return (
            f"**Volatility & Variance - {ticker.upper()} ({period})**\n\n"
            f"| Metric | Daily | Annualized |\n|---|---|---|\n"
            f"| Volatility (Std Dev) | {daily_vol:.4%} | {annual_vol:.4%} |\n"
            f"| Variance | {daily_var:.6f} | {annual_var:.6f} |"
        )
    except Exception as e:
        return f"Error calculating volatility: {str(e)}"

def get_sharpe_ratio(ticker, period):
    try:
        returns = fetch_returns(ticker, period)
        rf_annual = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1] / 100
        rf_daily = rf_annual / 252
        excess = returns - rf_daily
        sharpe = (excess.mean() / excess.std()) * math.sqrt(252)
        return (
            f"**Sharpe Ratio - {ticker.upper()} ({period})**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Sharpe Ratio (annualized) | {sharpe:.4f} |\n"
            f"| Annualized Return | {returns.mean() * 252:.4%} |\n"
            f"| Annualized Volatility | {returns.std() * math.sqrt(252):.4%} |\n"
            f"| Risk-Free Rate | {rf_annual:.3%} |"
        )
    except Exception as e:
        return f"Error calculating Sharpe ratio: {str(e)}"

def get_capm_expected_return(ticker, period, market_risk_premium=0.055):
    try:
        # Beta
        stock_ret = fetch_returns(ticker, period)
        bench_ret = fetch_returns("^GSPC", period)
        combined = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()
        beta = combined.cov().iloc[0, 1] / combined["bench"].var()

        # Risk-free rate
        rf = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1] / 100

        expected_return = rf + beta * market_risk_premium

        return (
            f"**CAPM Expected Return - {ticker.upper()} ({period})**\n\n"
            f"| Input | Value |\n|---|---|\n"
            f"| Risk-Free Rate (10Y Treasury) | {rf:.3%} |\n"
            f"| Beta | {beta:.4f} |\n"
            f"| Market Risk Premium | {market_risk_premium:.3%} |\n\n"
            f"**Expected Return = {rf:.3%} + {beta:.4f} x {market_risk_premium:.3%} = {expected_return:.4%}**"
        )
    except Exception as e:
        return f"Error calculating CAPM: {str(e)}"

def get_correlation_matrix(tickers, period, save_csv=False):
    try:
        returns = {}
        for ticker in tickers:
            returns[ticker.upper()] = fetch_returns(ticker, period)
        df = pd.DataFrame(returns).dropna()
        corr = df.corr()

        # Format as markdown table
        cols = list(corr.columns)
        header = "| | " + " | ".join(cols) + " |"
        separator = "|---|" + "|".join(["---"] * len(cols)) + "|"
        rows = []
        for idx, row in corr.iterrows():
            rows.append(f"| **{idx}** | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        table = "\n".join([header, separator] + rows)
        result = f"**Correlation Matrix ({period})**\n\n{table}"

        if save_csv:
            path = save_df_to_csv(corr, f"correlation_{'_'.join(t.upper() for t in tickers)}_{period}.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"

def get_total_return(ticker, period):
    try:
        df = yf.Ticker(ticker.upper()).history(period=period)
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        total_return = (end_price - start_price) / start_price
        return (
            f"**Total Return - {ticker.upper()} ({period})**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Start Price | ${start_price:.2f} |\n"
            f"| End Price | ${end_price:.2f} |\n"
            f"| Total Return | {total_return:.4%} |"
        )
    except Exception as e:
        return f"Error calculating total return: {str(e)}"

def get_max_drawdown(ticker, period):
    try:
        df = yf.Ticker(ticker.upper()).history(period=period)
        prices = df["Close"]
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin().strftime("%Y-%m-%d")
        peak_date = prices[:drawdown.idxmin()].idxmax().strftime("%Y-%m-%d")
        return (
            f"**Maximum Drawdown - {ticker.upper()} ({period})**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Max Drawdown | {max_dd:.4%} |\n"
            f"| Peak Date | {peak_date} |\n"
            f"| Trough Date | {max_dd_date} |"
        )
    except Exception as e:
        return f"Error calculating drawdown: {str(e)}"

def get_rolling_volatility(ticker, period, window_days, save_csv=False):
    try:
        returns = fetch_returns(ticker, period)
        rolling_vol = returns.rolling(window=window_days).std() * math.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        summary = rolling_vol.tail(10)
        lines = [f"**Rolling {window_days}-Day Volatility (Annualized) - {ticker.upper()} ({period})**\n"]
        lines.append("| Date | Volatility |")
        lines.append("|---|---|")
        for date, val in summary.items():
            lines.append(f"| {date.strftime('%Y-%m-%d')} | {val:.4%} |")
        lines.append(f"\n**Current:** {rolling_vol.iloc[-1]:.4%} | **Min:** {rolling_vol.min():.4%} | **Max:** {rolling_vol.max():.4%}")
        result = "\n".join(lines)
        if save_csv:
            path = save_df_to_csv(rolling_vol.to_frame("rolling_volatility"), f"{ticker.upper()}_rolling_vol_{period}.csv")
            result += f"\n\nSaved to: {path}"
        return result
    except Exception as e:
        return f"Error calculating rolling volatility: {str(e)}"

def calculate_wacc(ticker, cost_of_debt, tax_rate, market_risk_premium=0.055, period="2y"):
    try:
        info = yf.Ticker(ticker.upper()).info
        market_cap = info.get("marketCap", 0)
        total_debt = info.get("totalDebt", 0)

        if not market_cap or not total_debt:
            return "Could not fetch market cap or total debt for this ticker."

        total_capital = market_cap + total_debt
        weight_equity = market_cap / total_capital
        weight_debt = total_debt / total_capital

        # Beta for cost of equity
        stock_ret = fetch_returns(ticker, period)
        bench_ret = fetch_returns("^GSPC", period)
        combined = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()
        beta = combined.cov().iloc[0, 1] / combined["bench"].var()

        # Risk-free rate
        rf = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1] / 100

        cost_of_equity = rf + beta * market_risk_premium
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

        return (
            f"**WACC - {ticker.upper()}**\n\n"
            f"| Component | Value |\n|---|---|\n"
            f"| Market Cap (Equity) | ${market_cap/1e9:.1f}B |\n"
            f"| Total Debt | ${total_debt/1e9:.1f}B |\n"
            f"| Weight of Equity | {weight_equity:.4%} |\n"
            f"| Weight of Debt | {weight_debt:.4%} |\n"
            f"| Risk-Free Rate | {rf:.3%} |\n"
            f"| Beta | {beta:.4f} |\n"
            f"| Market Risk Premium | {market_risk_premium:.3%} |\n"
            f"| Cost of Equity (CAPM) | {cost_of_equity:.4%} |\n"
            f"| Pre-Tax Cost of Debt | {cost_of_debt:.3%} |\n"
            f"| Tax Rate | {tax_rate:.3%} |\n"
            f"| After-Tax Cost of Debt | {after_tax_cost_of_debt:.4%} |\n\n"
            f"**WACC = {wacc:.4%}**"
        )
    except Exception as e:
        return f"Error calculating WACC: {str(e)}"


# --- Legacy calculation tools ---

def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return "Error: start value and years must be positive."
    cagr = (end_value / start_value) ** (1 / years) - 1
    return f"{cagr:.4f} ({cagr * 100:.2f}%)"

def calculate_npv(discount_rate, cash_flows):
    npv = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows))
    return f"{npv:,.2f}"

def calculate_irr(initial_investment, cash_flows):
    flows = [-initial_investment] + cash_flows
    rate = 0.1
    for _ in range(1000):
        npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(flows))
        dnpv = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(flows))
        if dnpv == 0:
            return "Could not converge."
        rate -= npv / dnpv
        if abs(npv) < 1e-6:
            return f"{rate:.4f} ({rate * 100:.2f}%)"
    return "Could not converge."

def calculate_statistics(values):
    n = len(values)
    if n == 0:
        return "Error: no values provided."
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    return (
        f"Mean: {mean:,.4f} | Median: {median:,.4f} | "
        f"Std Dev: {std_dev:,.4f} | Min: {min(values):,.4f} | Max: {max(values):,.4f}"
    )


# --- Dispatcher ---

def execute_tool(tool_name, tool_input):
    dispatch = {
        "get_income_statement": lambda i: get_income_statement(i["ticker"], i.get("save_csv", False)),
        "get_balance_sheet": lambda i: get_balance_sheet(i["ticker"], i.get("save_csv", False)),
        "get_cash_flow": lambda i: get_cash_flow(i["ticker"], i.get("save_csv", False)),
        "get_stock_price_history": lambda i: get_stock_price_history(i["ticker"], i["period"], i.get("save_csv", False)),
        "get_company_info": lambda i: get_company_info(i["ticker"]),
        "compare_financials": lambda i: compare_financials(i["tickers"], i["metric"], i.get("save_csv", False)),
        "get_risk_free_rate": lambda i: get_risk_free_rate(),
        "get_beta": lambda i: get_beta(i["ticker"], i["period"], i.get("benchmark", "^GSPC")),
        "get_alpha": lambda i: get_alpha(i["ticker"], i["period"], i.get("benchmark", "^GSPC")),
        "get_volatility": lambda i: get_volatility(i["ticker"], i["period"]),
        "get_sharpe_ratio": lambda i: get_sharpe_ratio(i["ticker"], i["period"]),
        "get_capm_expected_return": lambda i: get_capm_expected_return(i["ticker"], i["period"], i.get("market_risk_premium", 0.055)),
        "get_correlation_matrix": lambda i: get_correlation_matrix(i["tickers"], i["period"], i.get("save_csv", False)),
        "get_total_return": lambda i: get_total_return(i["ticker"], i["period"]),
        "get_max_drawdown": lambda i: get_max_drawdown(i["ticker"], i["period"]),
        "get_rolling_volatility": lambda i: get_rolling_volatility(i["ticker"], i["period"], i["window_days"], i.get("save_csv", False)),
        "calculate_wacc": lambda i: calculate_wacc(i["ticker"], i["cost_of_debt"], i["tax_rate"], i.get("market_risk_premium", 0.055), i.get("period", "2y")),
        "calculate_cagr": lambda i: calculate_cagr(i["start_value"], i["end_value"], i["years"]),
        "calculate_npv": lambda i: calculate_npv(i["discount_rate"], i["cash_flows"]),
        "calculate_irr": lambda i: calculate_irr(i["initial_investment"], i["cash_flows"]),
        "calculate_statistics": lambda i: calculate_statistics(i["values"]),
    }
    fn = dispatch.get(tool_name)
    return fn(tool_input) if fn else "Unknown tool."


def run_finance_agent(user_request: str) -> str:
    messages = [{"role": "user", "content": user_request}]

    while True:
        response = get_client().messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
