import os
import csv
import requests
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"
SERPAPI_URL = "https://serpapi.com/search"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert intellectual property attorney and patent analyst.
You have tools to search global patent databases.

Primary sources:
- USPTO (PatentsView): authoritative for US grants, free, always available
- Google Patents (via SerpApi): covers USPTO, EPO, JPO (Japan), KIPO (Korea),
  TIPO (Taiwan), CNIPA (China), WIPO, and 100+ other offices

When searching patents by company:
- Always present results as a clean markdown table: Patent Number | Title | Date | Office/Country
- Report the total count found and how many are shown
- Offer to export to CSV if the user wants the full dataset

When asked about a specific patent number, provide full details:
title, abstract, claims summary, assignee, inventors, filing date, grant date, jurisdiction.

When comparing portfolios, show a side-by-side table and highlight differences in volume,
technology focus, and key jurisdictions.

Always note which database(s) results came from. Remind the user that pending or secret
applications may not appear in public databases."""


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
tools = [
    {
        "name": "search_patents_by_company_uspto",
        "description": (
            "Search USPTO (PatentsView) for all granted US patents assigned to a company. "
            "Free, no API key required. Best for comprehensive US patent counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Exact legal assignee name, e.g. 'Apple Inc.'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Patents to return. Defaults to 50."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "search_patents_google",
        "description": (
            "Search Google Patents via SerpApi. Covers all jurisdictions: US (USPTO), "
            "Europe (EPO), Japan (JPO), Korea (KIPO), Taiwan (TIPO), China (CNIPA), WIPO, and more. "
            "Requires SERPAPI_KEY in .env. Free tier: 100 searches/month at serpapi.com."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use 'assignee:\"Company Name\"' to find patents by a company. "
                        "Can also search by keyword, inventor, patent number, etc."
                    )
                },
                "country": {
                    "type": "string",
                    "description": (
                        "Filter by country/office code. Examples: 'US', 'EP', 'JP', 'KR', 'TW', 'CN', 'WO'. "
                        "Leave blank to search all jurisdictions."
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Results to return. Defaults to 20."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_patents_global",
        "description": (
            "Search both USPTO (PatentsView) and Google Patents (SerpApi) simultaneously "
            "for a comprehensive global patent portfolio view of a company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company/assignee name to search across all databases."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "get_patent_details",
        "description": (
            "Retrieve full details for a specific US patent number from USPTO. "
            "Returns title, abstract, assignee, inventors, filing date, and grant date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patent_number": {
                    "type": "string",
                    "description": "US patent number, e.g. '11234567' or 'US11234567'."
                }
            },
            "required": ["patent_number"]
        }
    },
    {
        "name": "compare_patent_portfolios",
        "description": (
            "Compare patent portfolio sizes across multiple companies using USPTO and Google Patents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of company names to compare."
                },
                "save_csv": {"type": "boolean"}
            },
            "required": ["companies"]
        }
    },
    {
        "name": "export_patents_csv",
        "description": "Fetch patents for a company from USPTO and export the full list to a CSV file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of records to export. Defaults to 500."
                }
            },
            "required": ["company_name"]
        }
    }
]


# ---------------------------------------------------------------------------
# Helper: save list of dicts to CSV
# ---------------------------------------------------------------------------
def _save_csv(records: list, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    if not records:
        return "No records to save."
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Helper: format patent dicts as a markdown table
# ---------------------------------------------------------------------------
def _to_markdown(records: list, title: str) -> str:
    if not records:
        return "No patents found."
    cols = list(records[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(r.get(c, "N/A")) for c in cols) + " |" for r in records]
    return f"**{title}**\n\n" + "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# USPTO PatentsView  (free, no key)
# ---------------------------------------------------------------------------
def _fetch_uspto(company_name: str, max_results: int = 50) -> tuple[list, int]:
    """Returns (records list, total_count)."""
    per_page = min(max_results, 100)
    payload = {
        "q": {"assignee_organization": company_name},
        "f": ["patent_number", "patent_title", "patent_date",
              "assignee_organization", "inventor_last_name"],
        "o": {"per_page": per_page, "page": 1},
        "s": [{"patent_date": "desc"}]
    }
    resp = requests.post(PATENTSVIEW_URL, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    total = data.get("total_patent_count", 0)
    records = []
    for p in (data.get("patents") or []):
        assignee = ""
        if p.get("assignees"):
            assignee = p["assignees"][0].get("assignee_organization", "")
        inventor = ""
        if p.get("inventors"):
            inventor = p["inventors"][0].get("inventor_last_name", "")
        records.append({
            "Patent Number": f"US{p.get('patent_number', '')}",
            "Title": p.get("patent_title", ""),
            "Grant Date": p.get("patent_date", ""),
            "Assignee": assignee,
            "Lead Inventor": inventor,
            "Office": "USPTO"
        })
    return records[:max_results], total


# ---------------------------------------------------------------------------
# Google Patents via SerpApi
# ---------------------------------------------------------------------------
def _fetch_google_patents(query: str, country: str = None, max_results: int = 20) -> list:
    """Returns list of patent dicts. Requires SERPAPI_KEY in .env."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return []

    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": api_key,
        "num": min(max_results, 10)   # SerpApi returns up to 10 per page
    }
    if country:
        params["country"] = country.upper()

    records = []
    # Fetch up to 3 pages (30 results max per call before hitting free tier fast)
    for start in range(0, max_results, 10):
        if start > 0:
            params["start"] = start
        resp = requests.get(SERPAPI_URL, params=params, timeout=15)
        if resp.status_code != 200:
            break
        data = resp.json()
        results = data.get("organic_results", [])
        if not results:
            break
        for r in results:
            records.append({
                "Patent Number": r.get("patent_id", "N/A"),
                "Title": r.get("title", "N/A"),
                "Grant Date": r.get("grant_date") or r.get("filing_date", "N/A"),
                "Assignee": r.get("assignee", "N/A"),
                "Inventor": r.get("inventor", "N/A"),
                "Office": r.get("country_status", {}).get("country", "N/A")
                          if isinstance(r.get("country_status"), dict)
                          else r.get("country_status", "N/A")
            })
        if len(results) < 10:
            break

    return records[:max_results]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_patents_by_company_uspto(company_name: str, max_results: int = 50, save_csv: bool = False) -> str:
    try:
        records, total = _fetch_uspto(company_name, max_results)
        if not records:
            return (
                f"No USPTO patents found for '{company_name}'.\n"
                "Tip: use the exact legal name (e.g. 'Apple Inc.' not 'Apple')."
            )
        result = f"**USPTO - {company_name}**\n\nTotal on record: **{total:,}** | Showing: {len(records)}\n\n"
        result += _to_markdown(records, f"USPTO Patents - {company_name}")
        if save_csv:
            path = _save_csv(records, f"{company_name.replace(' ', '_')}_uspto.csv")
            result += f"\n\nSaved to: `{path}`"
        return result
    except Exception as e:
        return f"Error searching USPTO: {str(e)}"


def search_patents_google(query: str, country: str = None, max_results: int = 20, save_csv: bool = False) -> str:
    try:
        if not os.getenv("SERPAPI_KEY"):
            return (
                "Google Patents search requires a SerpApi key.\n\n"
                "**Setup (free tier = 100 searches/month):**\n"
                "1. Sign up at https://serpapi.com/\n"
                "2. Copy your API key from the dashboard\n"
                "3. Add to your `.env` file:\n"
                "   ```\n"
                "   SERPAPI_KEY=your_key_here\n"
                "   ```\n"
                "4. Restart the server\n\n"
                "In the meantime, use `search_patents_by_company_uspto` for US patents."
            )
        records = _fetch_google_patents(query, country, max_results)
        if not records:
            return f"No results found on Google Patents for: `{query}`"
        result = f"**Google Patents search:** `{query}`"
        if country:
            result += f" | Country filter: `{country}`"
        result += f"\n\nShowing: {len(records)}\n\n"
        result += _to_markdown(records, "Google Patents Results")
        if save_csv:
            safe = query.replace('"', '').replace(':', '_').replace(' ', '_')[:50]
            path = _save_csv(records, f"google_patents_{safe}.csv")
            result += f"\n\nSaved to: `{path}`"
        return result
    except Exception as e:
        return f"Error searching Google Patents: {str(e)}"


def search_patents_global(company_name: str, save_csv: bool = False) -> str:
    lines = [f"**Global Patent Search - {company_name}**\n"]
    all_records = []

    # USPTO
    try:
        records, total = _fetch_uspto(company_name, max_results=50)
        lines.append(f"- **USPTO (US):** {total:,} total grants | showing {len(records)} most recent")
        all_records.extend(records)
    except Exception as e:
        lines.append(f"- **USPTO:** Error - {str(e)}")

    # Google Patents
    if not os.getenv("SERPAPI_KEY"):
        lines.append(
            "- **Google Patents (JP, KR, TW, EP, CN, WO, +more):** Not configured — "
            "add SERPAPI_KEY to .env (free tier at serpapi.com)"
        )
    else:
        try:
            gp_records = _fetch_google_patents(f'assignee:"{company_name}"', max_results=30)
            lines.append(f"- **Google Patents (all jurisdictions):** {len(gp_records)} results shown")
            all_records.extend(gp_records)
        except Exception as e:
            lines.append(f"- **Google Patents:** Error - {str(e)}")

    result = "\n".join(lines) + "\n\n"
    if all_records:
        result += _to_markdown(all_records, f"Combined Results - {company_name}")
        if save_csv:
            path = _save_csv(all_records, f"{company_name.replace(' ', '_')}_global_patents.csv")
            result += f"\n\nFull dataset saved to: `{path}`"
    else:
        result += "No records retrieved. Check the company name spelling."

    result += (
        "\n\n---\n"
        "**Coverage:** USPTO = US grants only. Google Patents indexes US, EP, JP, KR, TW, CN, WO, and 100+ offices. "
        "Pending and unpublished applications may not appear."
    )
    return result


def get_patent_details(patent_number: str) -> str:
    try:
        clean = patent_number.upper().replace("US", "").replace(",", "").strip()
        payload = {
            "q": {"patent_number": clean},
            "f": [
                "patent_number", "patent_title", "patent_abstract",
                "patent_date", "patent_type", "assignee_organization",
                "inventor_first_name", "inventor_last_name",
                "app_date", "app_number"
            ]
        }
        resp = requests.post(PATENTSVIEW_URL, json=payload, timeout=15)
        resp.raise_for_status()
        patents = resp.json().get("patents") or []
        if not patents:
            return f"Patent '{patent_number}' not found in USPTO database."

        p = patents[0]
        assignees = ", ".join(
            a.get("assignee_organization", "Unknown") for a in (p.get("assignees") or [])
        )
        inventors = ", ".join(
            f"{i.get('inventor_first_name','')} {i.get('inventor_last_name','')}".strip()
            for i in (p.get("inventors") or [])
        )
        abstract = p.get("patent_abstract", "N/A")
        if abstract and len(abstract) > 800:
            abstract = abstract[:800] + "..."

        return "\n".join([
            f"**Patent US{p.get('patent_number', '')}**\n",
            "| Field | Detail |",
            "|---|---|",
            f"| Title | {p.get('patent_title', 'N/A')} |",
            f"| Grant Date | {p.get('patent_date', 'N/A')} |",
            f"| Application Date | {p.get('app_date', 'N/A')} |",
            f"| Application Number | {p.get('app_number', 'N/A')} |",
            f"| Type | {p.get('patent_type', 'N/A')} |",
            f"| Assignee(s) | {assignees or 'N/A'} |",
            f"| Inventors | {inventors or 'N/A'} |",
            f"\n**Abstract:**\n\n{abstract}"
        ])
    except Exception as e:
        return f"Error fetching patent details: {str(e)}"


def compare_patent_portfolios(companies: list, save_csv: bool = False) -> str:
    try:
        rows = []
        for company in companies:
            row = {"Company": company}
            try:
                _, total = _fetch_uspto(company, max_results=1)
                row["USPTO Total"] = f"{total:,}"
            except Exception:
                row["USPTO Total"] = "Error"

            if os.getenv("SERPAPI_KEY"):
                try:
                    gp = _fetch_google_patents(f'assignee:"{company}"', max_results=10)
                    row["Google Patents (sample)"] = str(len(gp))
                except Exception:
                    row["Google Patents (sample)"] = "Error"
            else:
                row["Google Patents (sample)"] = "No SERPAPI_KEY"

            rows.append(row)

        result = _to_markdown(rows, "Patent Portfolio Comparison")
        if save_csv:
            path = _save_csv(rows, "patent_portfolio_comparison.csv")
            result += f"\n\nSaved to: `{path}`"
        return result
    except Exception as e:
        return f"Error comparing portfolios: {str(e)}"


def export_patents_csv(company_name: str, max_results: int = 500) -> str:
    try:
        records, total = _fetch_uspto(company_name, max_results=max_results)
        if not records:
            return f"No USPTO patents found for '{company_name}'."
        filename = f"{company_name.replace(' ', '_').replace('.', '')}_patents_export.csv"
        path = _save_csv(records, filename)
        return (
            f"Exported {len(records)} of {total:,} USPTO patents for **{company_name}**.\n\n"
            f"Saved to: `{path}`"
        )
    except Exception as e:
        return f"Error exporting patents: {str(e)}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
TOOL_DISPATCH = {
    "search_patents_by_company_uspto": lambda i: search_patents_by_company_uspto(
        i["company_name"], i.get("max_results", 50), i.get("save_csv", False)
    ),
    "search_patents_google": lambda i: search_patents_google(
        i["query"], i.get("country"), i.get("max_results", 20), i.get("save_csv", False)
    ),
    "search_patents_global": lambda i: search_patents_global(
        i["company_name"], i.get("save_csv", False)
    ),
    "get_patent_details": lambda i: get_patent_details(i["patent_number"]),
    "compare_patent_portfolios": lambda i: compare_patent_portfolios(
        i["companies"], i.get("save_csv", False)
    ),
    "export_patents_csv": lambda i: export_patents_csv(
        i["company_name"], i.get("max_results", 500)
    ),
}


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------
def run_lawyer_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    while True:
        response = client.messages.create(
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
                    handler = TOOL_DISPATCH.get(block.name)
                    result = handler(block.input) if handler else f"Unknown tool: {block.name}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
