"""
Financial Slide Deck Generator — Deep Agent with HTML slides.

Generates polished financial HTML presentations using Claude,
converts slides to PNG via Playwright, and attaches them to LangSmith traces.
"""

import asyncio
import base64
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.sandbox import SandboxClient
from langsmith.schemas import Attachment

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# Import database tools
from db import list_tables, describe_table, query_financials

# ============================================================================
# SOURCE DOCUMENT RETRIEVER
# ============================================================================

SOURCES_DIR = os.path.join(os.path.dirname(__file__), "sources")

# Module-level storage for source docs read during the current invocation
_last_source_docs: dict[str, str] = {}


def _convert_docs(files_content: list[tuple[str, str, str]]) -> list[dict]:
    """Convert file data to LangSmith Document format (plain dicts)."""
    return [
        {
            "page_content": content,
            "type": "Document",
            "metadata": {"source": filename, "path": path},
        }
        for filename, path, content in files_content
    ]


@traceable(run_type="retriever", name="source_document_retriever")
def _retrieve_docs(query: str) -> list[dict]:
    """Retriever that loads all .txt source documents. Traced as a retriever
    in LangSmith so documents render with the dedicated document viewer."""
    files = sorted(f for f in os.listdir(SOURCES_DIR) if f.endswith(".txt"))
    files_content = []
    for filename in files:
        path = os.path.join(SOURCES_DIR, filename)
        with open(path, "r") as f:
            content = f.read()
        _last_source_docs[filename] = content
        files_content.append((filename, path, content))
    return _convert_docs(files_content)


@tool
def retrieve_source_documents(query: str) -> str:
    """Retrieve all .txt source documents from the sources directory.
    These are primary reference materials — earnings reports, risk committee
    minutes, treasury strategy memos — that provide authoritative context
    for building financial slides.

    ALWAYS call this tool before querying other data sources.

    Args:
        query: A brief description of what you're looking for (e.g. 'Q4 2024 financial data')
    """
    docs = _retrieve_docs(query)
    if not docs:
        return "No source documents found."
    parts = []
    for doc in docs:
        parts.append(f"=== {doc['metadata']['source']} ===\n{doc['page_content']}")
    return "\n\n".join(parts)


# ============================================================================
# HTML → PNG CONVERSION
# ============================================================================

async def html_to_pngs(html: str) -> list[bytes]:
    """Render HTML slides to PNG images using Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.set_content(html, wait_until="networkidle")
        slides = await page.query_selector_all(".slide")
        pngs = []
        for slide in slides:
            png = await slide.screenshot()
            pngs.append(png)
        await browser.close()
        return pngs


# ============================================================================
# SLIDE GENERATION TOOL
# ============================================================================

# Module-level storage for PNGs from the last generate_slides call
_last_slide_pngs: list[bytes] = []


@tool
async def generate_slides(html: str) -> str:
    """Render an HTML slide deck to PNG images. Pass the complete self-contained HTML string.
    Returns confirmation with slide count. The slides are stored for attachment to the trace."""
    global _last_slide_pngs
    try:
        pngs = await html_to_pngs(html)
        _last_slide_pngs = pngs
        return f"Successfully rendered {len(pngs)} slide(s) to PNG."
    except Exception as e:
        _last_slide_pngs = []
        return f"Error rendering slides: {e}"


# ============================================================================
# LANGSMITH SANDBOX FOR CALCULATIONS
# ============================================================================

_sandbox_client: SandboxClient | None = None
_sandbox_template_ready = False


def _ensure_sandbox():
    """Lazy-initialize the sandbox client and template on first use."""
    global _sandbox_client, _sandbox_template_ready
    if _sandbox_client is None:
        _sandbox_client = SandboxClient()
    if not _sandbox_template_ready:
        try:
            _sandbox_client.create_template(
                name="financial-compute",
                image="python:3.12-slim",
            )
        except Exception:
            pass  # Template already exists
        _sandbox_template_ready = True


@tool
def run_financial_calculation(python_code: str) -> str:
    """Execute Python code in a secure LangSmith sandbox for financial calculations,
    projections, and statistical modeling.

    The sandbox has pandas and numpy available. Use print() to output results.
    Use for: DCF models, compound growth projections, amortization schedules,
    Monte Carlo simulations, variance analysis, and financial ratio computations.
    """
    _ensure_sandbox()
    with _sandbox_client.sandbox(template_name="financial-compute") as sb:
        sb.run("pip install -q pandas numpy 2>/dev/null")
        result = sb.run(f"python3 << 'CALCEOF'\n{python_code}\nCALCEOF")
        if result.success:
            return result.stdout.strip() if result.stdout else "Calculation completed (no output). Use print() to display results."
        return f"Calculation error:\n{result.stderr.strip() if result.stderr else 'Unknown error'}"


# ============================================================================
# ENTERPRISE DATA SOURCE CONNECTORS
# ============================================================================

@tool
def query_core_banking_ledger(account_type: str = "all", as_of_date: str = "2024-12-31") -> str:
    """Query the core banking general ledger (Oracle Flexcube).
    Returns account balances, transaction volumes, and loan portfolio metrics.

    Args:
        account_type: 'assets', 'liabilities', 'equity', or 'all'
        as_of_date: Balance sheet date (YYYY-MM-DD)
    """
    ledger = {
        "assets": [
            ("Cash & Cash Equivalents", 124_500_000, "+8.2%"),
            ("Trading Securities", 287_300_000, "+3.1%"),
            ("Loans & Advances - Commercial", 412_800_000, "+12.4%"),
            ("Loans & Advances - Retail", 198_600_000, "+6.7%"),
            ("Fixed Assets", 45_200_000, "-1.2%"),
            ("Intangible Assets", 23_100_000, "+0.5%"),
        ],
        "liabilities": [
            ("Customer Deposits - Demand", 389_200_000, "+9.8%"),
            ("Customer Deposits - Term", 245_700_000, "+4.3%"),
            ("Interbank Borrowings", 87_400_000, "-15.2%"),
            ("Subordinated Debt", 45_000_000, "0.0%"),
            ("Accrued Liabilities", 12_800_000, "+2.1%"),
        ],
        "equity": [
            ("Common Stock", 150_000_000, "0.0%"),
            ("Retained Earnings", 89_400_000, "+18.5%"),
            ("Other Comprehensive Income", -3_200_000, "n/a"),
            ("Regulatory Reserves", 22_800_000, "+5.0%"),
        ],
    }

    sections = [account_type] if account_type != "all" else ["assets", "liabilities", "equity"]
    lines = [f"=== Core Banking Ledger — {as_of_date} ===", "Source: Oracle Flexcube GL | Connection: PROD-GL-PRIMARY", ""]

    for section in sections:
        if section in ledger:
            total = sum(row[1] for row in ledger[section])
            lines.append(f"--- {section.upper()} (Total: ${total:,.0f}) ---")
            lines.append(f"{'Account':<40} {'Balance':>15} {'QoQ Change':>12}")
            lines.append("-" * 70)
            for name, balance, change in ledger[section]:
                lines.append(f"{name:<40} ${balance:>14,.0f} {change:>11}")
            lines.append("")

    lines.append("Transaction Volume (MTD): 1,247,832 transactions")
    lines.append("Loan Default Rate: 0.42% | NPL Ratio: 1.8%")

    return "\n".join(lines)


@tool
def fetch_risk_exposure_report(portfolio_id: str = "FIRM_WIDE", risk_type: str = "all") -> str:
    """Query the enterprise risk management platform (Murex MX.3).
    Returns Value-at-Risk, credit exposure, stress test results, and concentration metrics.

    Args:
        portfolio_id: Portfolio identifier — 'FIRM_WIDE', 'TRADING', 'BANKING', 'WEALTH_MGMT'
        risk_type: 'market', 'credit', 'operational', 'liquidity', or 'all'
    """
    lines = [
        f"=== Risk Exposure Report — Portfolio: {portfolio_id} ===",
        "Source: Murex MX.3 | Risk Engine: Monte Carlo (10K scenarios) | COB: 2024-12-31",
        "",
    ]

    if risk_type in ("market", "all"):
        lines.extend([
            "--- MARKET RISK ---",
            f"{'Metric':<35} {'1-Day':>12} {'10-Day':>12} {'Limit':>12} {'Utilization':>12}",
            "-" * 85,
            f"{'VaR (99% confidence)':<35} {'$4.2M':>12} {'$13.3M':>12} {'$20.0M':>12} {'66.5%':>12}",
            f"{'Expected Shortfall (CVaR)':<35} {'$6.8M':>12} {'$21.5M':>12} {'$30.0M':>12} {'71.7%':>12}",
            f"{'Stressed VaR':<35} {'$9.1M':>12} {'$28.8M':>12} {'$40.0M':>12} {'72.0%':>12}",
            f"{'Interest Rate DV01':<35} {'$285K':>12} {'—':>12} {'$500K':>12} {'57.0%':>12}",
            f"{'FX Delta':<35} {'$1.8M':>12} {'—':>12} {'$5.0M':>12} {'36.0%':>12}",
            f"{'Equity Beta Exposure':<35} {'$12.4M':>12} {'—':>12} {'$25.0M':>12} {'49.6%':>12}",
            "",
        ])

    if risk_type in ("credit", "all"):
        lines.extend([
            "--- CREDIT RISK ---",
            f"{'Counterparty Tier':<25} {'Exposure':>14} {'Collateral':>14} {'Net Exposure':>14} {'PD (1Y)':>10}",
            "-" * 80,
            f"{'Investment Grade (A+)':<25} {'$245.0M':>14} {'$198.2M':>14} {'$46.8M':>14} {'0.08%':>10}",
            f"{'Investment Grade (BBB)':<25} {'$187.3M':>14} {'$142.1M':>14} {'$45.2M':>14} {'0.24%':>10}",
            f"{'High Yield (BB)':<25} {'$62.4M':>14} {'$38.7M':>14} {'$23.7M':>14} {'1.12%':>10}",
            f"{'High Yield (B)':<25} {'$28.1M':>14} {'$15.9M':>14} {'$12.2M':>14} {'3.45%':>10}",
            f"{'Distressed (CCC-)':<25} {'$8.7M':>14} {'$4.2M':>14} {'$4.5M':>14} {'12.80%':>10}",
            "",
            "Expected Credit Loss (ECL): $3.42M | Loss Given Default (LGD): 42.5%",
            "Largest Single-Name Exposure: Meridian Capital Corp — $18.4M (3.5% of portfolio)",
            "",
        ])

    if risk_type in ("operational", "all"):
        lines.extend([
            "--- OPERATIONAL RISK ---",
            "Operational VaR (99.9%): $8.5M",
            "Incident Count (QTD): 23 | Severity: 2 High, 8 Medium, 13 Low",
            "Key Risk Indicators:",
            "  System Downtime: 0.12% (limit: 0.5%) — GREEN",
            "  Failed Trades: 0.03% (limit: 0.1%) — GREEN",
            "  Manual Overrides: 847 (limit: 1000) — AMBER",
            "  Data Quality Exceptions: 156 (limit: 200) — AMBER",
            "",
        ])

    if risk_type in ("liquidity", "all"):
        lines.extend([
            "--- LIQUIDITY RISK ---",
            "Liquidity Coverage Ratio (LCR): 142.8% (regulatory min: 100%)",
            "Net Stable Funding Ratio (NSFR): 118.5% (regulatory min: 100%)",
            "High-Quality Liquid Assets (HQLA): $187.3M",
            "Net Cash Outflows (30-day stress): $131.1M",
            "Contingent Funding Capacity: $95.0M",
            "",
        ])

    return "\n".join(lines)


@tool
def pull_treasury_positions(currency: str = "all", position_type: str = "all") -> str:
    """Query the treasury management system (Kyriba TMS).
    Returns cash positions, FX exposure, investment portfolio, and funding status.

    Args:
        currency: 'USD', 'EUR', 'GBP', 'JPY', 'all'
        position_type: 'cash', 'fx', 'investments', 'funding', or 'all'
    """
    lines = [
        "=== Treasury Position Report ===",
        "Source: Kyriba TMS | As-of: 2024-12-31 16:00 EST | Next sweep: 2025-01-02",
        "",
    ]

    if position_type in ("cash", "all"):
        lines.extend([
            "--- CASH POSITIONS ---",
            f"{'Entity / Account':<35} {'Currency':>8} {'Balance':>16} {'Available':>16}",
            "-" * 78,
            f"{'HQ Operating - JPMorgan':<35} {'USD':>8} {'$47,250,000':>16} {'$45,800,000':>16}",
            f"{'HQ Payroll - Bank of America':<35} {'USD':>8} {'$8,120,000':>16} {'$8,120,000':>16}",
            f"{'EMEA Ops - Barclays':<35} {'EUR':>8} {'€12,340,000':>16} {'€11,900,000':>16}",
            f"{'EMEA Ops - Deutsche Bank':<35} {'EUR':>8} {'€5,670,000':>16} {'€5,670,000':>16}",
            f"{'UK Entity - HSBC':<35} {'GBP':>8} {'£4,890,000':>16} {'£4,200,000':>16}",
            f"{'APAC Ops - MUFG':<35} {'JPY':>8} {'¥1,245,000,000':>16} {'¥1,180,000,000':>16}",
            f"{'APAC Ops - DBS Singapore':<35} {'USD':>8} {'$3,450,000':>16} {'$3,450,000':>16}",
            "",
            "Total USD Equivalent: $89,420,000 | Unencumbered: $84,150,000",
            "",
        ])

    if position_type in ("fx", "all"):
        lines.extend([
            "--- FX EXPOSURE (Unhedged) ---",
            f"{'Currency Pair':<15} {'Notional':>14} {'Mark-to-Market':>16} {'Hedge Ratio':>12} {'Maturity':>12}",
            "-" * 72,
            f"{'EUR/USD':<15} {'€18,200,000':>14} {'$19,474,000':>16} {'78.5%':>12} {'Various':>12}",
            f"{'GBP/USD':<15} {'£7,340,000':>14} {'$9,342,200':>16} {'65.0%':>12} {'Various':>12}",
            f"{'USD/JPY':<15} {'¥2,100,000,000':>14} {'$14,084,507':>16} {'82.3%':>12} {'Various':>12}",
            f"{'USD/CHF':<15} {'CHF 3,200,000':>14} {'$3,657,143':>16} {'45.0%':>12} {'Q1 2025':>12}",
            "",
            "Net Unhedged FX Exposure: $12.8M | FX VaR (95%, 1-day): $420K",
            "",
        ])

    if position_type in ("investments", "all"):
        lines.extend([
            "--- SHORT-TERM INVESTMENT PORTFOLIO ---",
            f"{'Instrument':<30} {'Par Value':>14} {'Yield':>8} {'Maturity':>12} {'Rating':>8}",
            "-" * 75,
            f"{'US Treasury Bills':<30} {'$25,000,000':>14} {'5.28%':>8} {'2025-03-15':>12} {'AAA':>8}",
            f"{'Commercial Paper - MSFT':<30} {'$10,000,000':>14} {'5.35%':>8} {'2025-02-28':>12} {'A-1+':>8}",
            f"{'Commercial Paper - JNJ':<30} {'$7,500,000':>14} {'5.31%':>8} {'2025-01-31':>12} {'A-1+':>8}",
            f"{'Bank CDs - Citi':<30} {'$15,000,000':>14} {'5.45%':>8} {'2025-06-30':>12} {'A':>8}",
            f"{'Money Market Fund':<30} {'$18,500,000':>14} {'5.22%':>8} {'Overnight':>12} {'AAA':>8}",
            f"{'Repo Agreements':<30} {'$12,000,000':>14} {'5.30%':>8} {'2025-01-15':>12} {'N/A':>8}",
            "",
            "Total Investment Portfolio: $88,000,000 | Weighted Avg Yield: 5.32%",
            "Duration: 0.34 years | WAL: 98 days",
            "",
        ])

    if position_type in ("funding", "all"):
        lines.extend([
            "--- FUNDING & DEBT STRUCTURE ---",
            f"{'Facility':<30} {'Committed':>14} {'Drawn':>14} {'Available':>14} {'Rate':>8}",
            "-" * 83,
            f"{'Revolving Credit - Syndicate':<30} {'$100,000,000':>14} {'$35,000,000':>14} {'$65,000,000':>14} {'SOFR+125':>8}",
            f"{'Term Loan A':<30} {'$50,000,000':>14} {'$50,000,000':>14} {'$0':>14} {'SOFR+150':>8}",
            f"{'Term Loan B':<30} {'$75,000,000':>14} {'$75,000,000':>14} {'$0':>14} {'SOFR+200':>8}",
            f"{'Letter of Credit Facility':<30} {'$20,000,000':>14} {'$8,400,000':>14} {'$11,600,000':>14} {'1.25%':>8}",
            "",
            "Total Debt Outstanding: $168,400,000 | Weighted Avg Cost: SOFR+168bps",
            "Next Maturity: Term Loan A ($50M) — 2026-06-30",
            "Covenants: Leverage 2.8x (limit 4.0x) | Interest Coverage 8.2x (min 3.0x)",
            "",
        ])

    return "\n".join(lines)


@tool
def get_regulatory_capital_metrics(reporting_period: str = "Q4 2024") -> str:
    """Query the regulatory reporting system (AxiomSL / Moody's Analytics).
    Returns Basel III capital adequacy, liquidity ratios, and stress test results.

    Args:
        reporting_period: e.g. 'Q4 2024', 'Q3 2024'
    """
    return "\n".join([
        f"=== Regulatory Capital & Compliance Report — {reporting_period} ===",
        "Source: AxiomSL RegReporter | Submission Status: Filed | Examiner: OCC",
        "",
        "--- CAPITAL ADEQUACY (BASEL III) ---",
        f"{'Metric':<40} {'Actual':>10} {'Minimum':>10} {'Buffer':>10} {'Status':>10}",
        "-" * 83,
        f"{'CET1 Capital Ratio':<40} {'14.2%':>10} {'4.5%':>10} {'9.7%':>10} {'PASS':>10}",
        f"{'Tier 1 Capital Ratio':<40} {'15.8%':>10} {'6.0%':>10} {'9.8%':>10} {'PASS':>10}",
        f"{'Total Capital Ratio':<40} {'18.1%':>10} {'8.0%':>10} {'10.1%':>10} {'PASS':>10}",
        f"{'Leverage Ratio':<40} {'6.9%':>10} {'3.0%':>10} {'3.9%':>10} {'PASS':>10}",
        f"{'Countercyclical Buffer':<40} {'2.5%':>10} {'0.0%':>10} {'2.5%':>10} {'PASS':>10}",
        "",
        "--- RISK-WEIGHTED ASSETS ---",
        f"{'Category':<35} {'RWA':>16} {'% of Total':>12}",
        "-" * 66,
        f"{'Credit Risk - Standardized':<35} {'$312,400,000':>16} {'52.8%':>12}",
        f"{'Credit Risk - IRB':<35} {'$145,600,000':>16} {'24.6%':>12}",
        f"{'Market Risk':<35} {'$78,200,000':>16} {'13.2%':>12}",
        f"{'Operational Risk':<35} {'$55,800,000':>16} {'9.4%':>12}",
        f"{'Total RWA':<35} {'$592,000,000':>16} {'100.0%':>12}",
        "",
        "--- STRESS TEST RESULTS (DFAST/CCAR) ---",
        f"{'Scenario':<25} {'CET1 Min':>10} {'Tier1 Min':>10} {'Total Cap Min':>12} {'Leverage Min':>12}",
        "-" * 72,
        f"{'Baseline':<25} {'13.8%':>10} {'15.4%':>10} {'17.7%':>12} {'6.7%':>12}",
        f"{'Adverse':<25} {'11.2%':>10} {'12.8%':>10} {'15.1%':>12} {'5.8%':>12}",
        f"{'Severely Adverse':<25} {'8.9%':>10} {'10.5%':>10} {'12.8%':>12} {'4.9%':>12}",
        "",
        "Stress Test Result: PASS — CET1 remains above 4.5% minimum in all scenarios",
        "",
        "--- LIQUIDITY METRICS ---",
        "Liquidity Coverage Ratio (LCR): 142.8% (min 100%)",
        "Net Stable Funding Ratio (NSFR): 118.5% (min 100%)",
        "HQLA Composition: Level 1: $142M (76%) | Level 2A: $32M (17%) | Level 2B: $13M (7%)",
        "",
        "--- COMPLIANCE SUMMARY ---",
        "Open Regulatory Findings: 3 (0 Critical, 1 High, 2 Medium)",
        "Remediation On-Track: 2 of 3 | Overdue: 1 (Medium — data lineage documentation)",
        "Next Exam: 2025-Q2 (OCC Targeted Review — Credit Risk)",
    ])


# ============================================================================
# AGENT SETUP
# ============================================================================

SYSTEM_PROMPT = """You are a financial analyst that creates comprehensive HTML slide decks.

IMPORTANT: At the start of every request, use the write_todos tool to create a plan before doing any work.

WORKFLOW:
1. Create a plan using write_todos with the steps you'll take
2. Load the "langchain-brand-slides" skill to get the design system and templates
3. ALWAYS call retrieve_source_documents first — this retrieves all .txt source documents (earnings reports, risk committee minutes, strategy memos) that provide authoritative context for the slides.
4. Query financial data sources — use BOTH the SQL database AND the enterprise system connectors:
   - SQL Database: use list_tables, describe_table, query_financials for internal metrics
   - Core Banking: use query_core_banking_ledger for GL balances and loan portfolio data
   - Risk Platform: use fetch_risk_exposure_report for VaR, credit exposure, stress tests
   - Treasury: use pull_treasury_positions for cash positions, FX, investments, and funding
   - Regulatory: use get_regulatory_capital_metrics for capital ratios and compliance
5. For any calculations, projections, or financial modeling, use run_financial_calculation to execute Python code in a secure sandbox. Examples:
   - Revenue growth projections and CAGR calculations
   - DCF models and NPV analysis
   - Monte Carlo simulations for risk scenarios
   - Variance analysis and budget vs. actuals
   - Ratio analysis (debt/equity, current ratio, ROE, etc.)
   - Amortization schedules and interest calculations
   Do NOT do complex math in your head — always use the sandbox.
6. Build a complete self-contained HTML slide deck following the skill's design rules
7. Call generate_slides with the full HTML string to render it to PNG images
8. Write a 2-3 sentence summary of the key financial highlights shown in the deck

Mark each todo as completed as you finish it.

RULES:
- Always create a plan with write_todos FIRST
- Always load the skill first to get the latest design templates
- ALWAYS call retrieve_source_documents before querying other data sources — this is mandatory
- Always query the database AND enterprise systems — do not make up financial numbers
- Use run_financial_calculation for any non-trivial math — projections, compound growth, scenario modeling
- Be thorough: pull data from ALL relevant tables and enterprise systems
- When multiple sources have related data, include ALL of them — show every angle
- Create 4-6 slides with dense metric grids. Pack each slide with as many metrics as fit.
- Use cols-4 grids wherever possible. More data = better.
- Include data from overlapping sources on the SAME slide when they cover the same topic
- Format currency as $3.4M, percentages as 24.6%
- Show trends (up/down/flat) when comparing time periods
- When in doubt, add MORE data, not less
"""

agent = create_deep_agent(
    name="financial-slide-agent",
    model="claude-sonnet-4-5-20250929",
    tools=[
        retrieve_source_documents,
        list_tables, describe_table, query_financials, generate_slides,
        run_financial_calculation,
        query_core_banking_ledger, fetch_risk_exposure_report,
        pull_treasury_positions, get_regulatory_capital_metrics,
    ],
    system_prompt=SYSTEM_PROMPT,
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    skills=["./skills/"],
    checkpointer=MemorySaver(),
)


# ============================================================================
# TRACEABLE WRAPPER
# ============================================================================

@traceable(name="financial_slide_agent")
async def invoke_agent(message: str, thread_id: str = "default") -> dict:
    """
    Invoke the Deep Agent, attach PNGs and source docs to the LangSmith trace.

    Datadog context is passed via langsmith_extra from the server.
    """
    global _last_slide_pngs
    _last_slide_pngs = []
    _last_source_docs.clear()

    run_tree = get_current_run_tree()

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract text response from last AI message.
    # Use isinstance(AIMessage) — hasattr(msg, "name") is True for AIMessage too,
    # so the old check skipped real AI content and always returned the default.
    messages = result.get("messages", [])
    text_summary = "Slide deck generated successfully."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            text_summary = msg.content
            break

    # Convert PNGs to base64 for the frontend
    slide_pngs_base64 = [base64.b64encode(png).decode() for png in _last_slide_pngs]

    # Attach PNGs and source documents to the LangSmith run
    if run_tree:
        attachments = {}
        for i, png in enumerate(_last_slide_pngs):
            attachments[f"slide_{i+1}"] = Attachment(mime_type="image/png", data=png)
        for filename, content in _last_source_docs.items():
            safe_name = filename.replace(".", "_")
            attachments[f"source_{safe_name}"] = Attachment(
                mime_type="text/plain", data=content.encode()
            )
        if attachments:
            run_tree.attachments = attachments

    run_id = str(run_tree.id) if run_tree else None

    # Set trace output to the graph's native result (full messages
    # array with tool_calls, usage_metadata, etc.)
    if run_tree:
        run_tree.outputs = result

    return {
        "response": text_summary,
        "slide_pngs_base64": slide_pngs_base64,
        "run_id": run_id,
    }
