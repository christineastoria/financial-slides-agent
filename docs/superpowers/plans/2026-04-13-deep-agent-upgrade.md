# Deep Agent Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PPTX-based LangChain agent with a Deep Agent that generates HTML slides, queries SQLite, supports multi-turn threads, and preserves all tracing.

**Architecture:** Deep Agent with SQLite query tools + HTML slide generation tool + LangChain brand skill. FastAPI server wraps invocation with @traceable for PNG attachments and Datadog metadata. Frontend gets thread sidebar and multi-turn chat.

**Tech Stack:** deepagents, playwright, sqlite3, FastAPI, ddtrace, langsmith, LangChain brand design system (light mode)

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Replace the dependencies list:

```toml
[project]
name = "financial-slides-demo"
version = "0.2.0"
description = "Financial slide generator with Datadog APM/LLM Observability and LangSmith tracing"
readme = "../README.md"
requires-python = ">=3.11"
dependencies = [
    "ddtrace>=4.2.0",
    "deepagents>=0.1.0",
    "fastapi>=0.128.0",
    "httpx>=0.28.1",
    "langchain>=1.2.3",
    "langchain-anthropic>=1.3.1",
    "langgraph-cli[inmem]>=0.4.11",
    "langsmith>=0.6.2",
    "pandas>=2.0.0",
    "playwright>=1.52.0",
    "pydantic>=2.12.5",
    "python-dotenv>=1.2.1",
    "pytz>=2025.2",
    "uvicorn>=0.40.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]
```

Removed: `python-pptx`, `pdf2image`, `Pillow`
Added: `deepagents`, `playwright`

- [ ] **Step 2: Install dependencies and Playwright browser**

```bash
uv sync
uv run playwright install chromium
```

Expected: dependencies resolve, Chromium downloads successfully.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: swap pptx/pdf2image for deepagents/playwright"
```

---

### Task 2: Create SQLite database module

**Files:**
- Create: `db.py`

- [ ] **Step 1: Create db.py with schema, seed data, and query tools**

```python
"""
SQLite database for financial data.

Seeds three tables from the original demo datasets at import time.
Exposes read-only query tools for the Deep Agent.
"""

import sqlite3
import os

import pandas as pd
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(__file__), "financial_data.db")

# ---------------------------------------------------------------------------
# Seed data (same datasets as the original PPTX demo)
# ---------------------------------------------------------------------------

company_financials = pd.DataFrame({
    "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "Revenue": [2500000, 2750000, 3100000, 3450000],
    "COGS": [1500000, 1600000, 1750000, 1850000],
    "Gross_Profit": [1000000, 1150000, 1350000, 1600000],
    "Operating_Expenses": [600000, 650000, 700000, 750000],
    "Net_Income": [400000, 500000, 650000, 850000],
    "Gross_Margin_Pct": [40.0, 41.8, 43.5, 46.4],
    "Net_Margin_Pct": [16.0, 18.2, 21.0, 24.6],
    "Customers": [1200, 1350, 1520, 1750],
    "Employees": [45, 52, 58, 65],
})

saas_metrics = pd.DataFrame({
    "Month": ["Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024"],
    "MRR": [95000, 112000, 135000, 165000],
    "ARR": [1140000, 1344000, 1620000, 1980000],
    "New_Customers": [45, 62, 78, 95],
    "Churned_Customers": [8, 6, 7, 5],
    "Net_New_MRR": [12000, 17000, 23000, 30000],
    "Churn_Rate_Pct": [3.5, 2.8, 2.4, 1.9],
    "CAC": [520, 485, 450, 410],
    "LTV": [4800, 5200, 5800, 6500],
    "LTV_CAC_Ratio": [9.2, 10.7, 12.9, 15.9],
})

ecommerce_data = pd.DataFrame({
    "Category": ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty"],
    "Revenue": [1250000, 890000, 650000, 420000, 380000],
    "Orders": [8500, 12000, 5200, 3800, 6500],
    "Avg_Order_Value": [147, 74, 125, 111, 58],
    "Return_Rate_Pct": [8.5, 15.2, 4.3, 6.8, 3.2],
    "Profit_Margin_Pct": [18.5, 42.0, 35.0, 28.0, 55.0],
    "YoY_Growth_Pct": [12.0, 25.0, 8.0, 35.0, 45.0],
})


def _init_db():
    """Create and seed the database if it doesn't exist or is empty."""
    conn = sqlite3.connect(DB_PATH)
    company_financials.to_sql("company_financials", conn, if_exists="replace", index=False)
    saas_metrics.to_sql("saas_metrics", conn, if_exists="replace", index=False)
    ecommerce_data.to_sql("ecommerce_data", conn, if_exists="replace", index=False)
    conn.close()


_init_db()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH"}


@tool
def list_tables() -> str:
    """List all available tables in the financial database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "Available tables: " + ", ".join(tables)


@tool
def describe_table(table_name: str) -> str:
    """Describe a table's columns and show the first 3 rows as a sample."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        if not columns:
            return f"Table '{table_name}' not found."
        col_info = "\n".join(f"  {c[1]} ({c[2]})" for c in columns)
        sample = conn.execute(f"SELECT * FROM '{table_name}' LIMIT 3")
        col_names = [desc[0] for desc in sample.description]
        rows = sample.fetchall()
        sample_str = "\n".join(
            "  " + " | ".join(str(v) for v in row) for row in rows
        )
        return f"Table: {table_name}\n\nColumns:\n{col_info}\n\nSample rows ({', '.join(col_names)}):\n{sample_str}"
    finally:
        conn.close()


@tool
def query_financials(sql: str) -> str:
    """Execute a read-only SQL query against the financial database. Returns formatted results."""
    upper = sql.upper().strip()
    for kw in FORBIDDEN:
        if kw in upper.split():
            return f"Error: {kw} statements are not allowed. Read-only queries only."
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(sql)
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        if not rows:
            return "Query returned no results."
        header = " | ".join(col_names)
        divider = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{divider}\n{body}"
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()
```

- [ ] **Step 2: Verify the module loads and seeds correctly**

```bash
uv run python -c "from db import list_tables, describe_table, query_financials; print(list_tables.invoke({}))"
```

Expected: `Available tables: company_financials, ecommerce_data, saas_metrics`

- [ ] **Step 3: Commit**

```bash
git add db.py
git commit -m "feat: add SQLite database with financial data and query tools"
```

---

### Task 3: Create the LangChain brand slides skill

**Files:**
- Create: `skills/langchain-brand-slides/SKILL.md`

- [ ] **Step 1: Create the skill directory and SKILL.md**

```bash
mkdir -p skills/langchain-brand-slides
```

Write `skills/langchain-brand-slides/SKILL.md`:

````markdown
---
name: langchain-brand-slides
description: Generate HTML slide decks styled to the LangChain brand design system in light mode. Use when building financial presentations, reports, or any multi-slide HTML output.
---

# LangChain Brand Slides — Light Mode

Generate self-contained HTML slide decks using the LangChain design system adapted for light backgrounds.

## Slide HTML Structure

Every slide deck is a single HTML string. Each slide is a `<section class="slide">`. The HTML must be fully self-contained with an inline `<style>` block.

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --lc-surface: #F2FAFF;
  --lc-card-alt: #E5F4FF;
  --lc-dark-text: #030710;
  --lc-muted: #6B8299;
  --lc-blue: #7FC8FF;
  --lc-blue-hover: #99D4FF;
  --lc-blue-bg: #E5F4FF;
  --lc-border: #B8DFFF;
  --lc-lime: #4ade80;
  --lc-rose: #f87171;
  --lc-white: #FFFFFF;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

.slide {
  width: 1280px;
  height: 720px;
  background: var(--lc-surface);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  padding: 48px;
  position: relative;
  overflow: hidden;
  page-break-after: always;
}

/* Title slide variant */
.slide.title-slide {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

h1 {
  font-size: 48px;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  color: var(--lc-dark-text);
  margin-bottom: 16px;
}

h2 {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.02em;
  color: var(--lc-dark-text);
  margin-bottom: 24px;
}

h3 {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--lc-dark-text);
}

.subtitle {
  font-size: 18px;
  font-weight: 400;
  color: var(--lc-muted);
  line-height: 1.65;
}

.overline {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-blue);
  margin-bottom: 12px;
}

/* Metric card */
.metric-card {
  background: var(--lc-white);
  border: 1px solid var(--lc-border);
  border-radius: 16px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-muted);
}

.metric-value {
  font-size: 36px;
  font-weight: 800;
  color: var(--lc-dark-text);
  letter-spacing: -0.02em;
}

.metric-trend {
  font-size: 14px;
  font-weight: 600;
}

.metric-trend.up { color: var(--lc-lime); }
.metric-trend.down { color: var(--lc-rose); }
.metric-trend.flat { color: var(--lc-muted); }

/* Grid layouts */
.metrics-grid {
  display: grid;
  gap: 20px;
  margin-top: 24px;
}

.metrics-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.metrics-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.metrics-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }

/* Content card */
.content-card {
  background: var(--lc-card-alt);
  border: 1px solid var(--lc-border);
  border-radius: 16px;
  padding: 32px;
}

/* Slide number */
.slide-number {
  position: absolute;
  bottom: 24px;
  right: 48px;
  font-size: 12px;
  font-weight: 500;
  color: var(--lc-muted);
}

/* Tag/pill */
.tag {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-blue);
  background: var(--lc-blue-bg);
  padding: 4px 12px;
  border-radius: 100px;
}
</style>
</head>
<body>

<!-- SLIDES GO HERE -->

</body>
</html>
```

## Slide Types

### Title Slide
```html
<section class="slide title-slide">
  <div class="overline">QUARTERLY REPORT</div>
  <h1>Q4 2024 Business Review</h1>
  <p class="subtitle">Key financial metrics and growth highlights</p>
  <span class="slide-number">1</span>
</section>
```

### Metrics Slide (3 columns)
```html
<section class="slide">
  <div class="overline">REVENUE METRICS</div>
  <h2>Financial Performance</h2>
  <div class="metrics-grid cols-3">
    <div class="metric-card">
      <span class="metric-label">Revenue</span>
      <span class="metric-value">$3.4M</span>
      <span class="metric-trend up">+11.3% vs Q3</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Gross Profit</span>
      <span class="metric-value">$1.6M</span>
      <span class="metric-trend up">+18.5% vs Q3</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Net Income</span>
      <span class="metric-value">$850K</span>
      <span class="metric-trend up">+30.8% vs Q3</span>
    </div>
  </div>
  <span class="slide-number">2</span>
</section>
```

## Rules

1. Always produce a complete self-contained HTML document with the full `<style>` block
2. Each slide is a `<section class="slide">` at exactly 1280x720px
3. Use 2-3 slides total: title slide + 1-2 content slides
4. Format currency as $3.4M not $3400000
5. Format percentages with one decimal: 24.6%
6. Use `.up` (green) for positive trends, `.down` (red) for negative, `.flat` (gray) for neutral
7. Use `.cols-2` for 2 metrics, `.cols-3` for 3, `.cols-4` for 4
8. Always include slide numbers
9. Never use external images or resources beyond Google Fonts
10. After generating, call the `generate_slides` tool with the full HTML string
````

- [ ] **Step 2: Commit**

```bash
git add skills/
git commit -m "feat: add langchain-brand-slides skill for HTML slide generation"
```

---

### Task 4: Rewrite agent.py with Deep Agent

**Files:**
- Rewrite: `agent.py`

- [ ] **Step 1: Write the new agent.py**

```python
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
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.schemas import Attachment

# Load environment from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# Import database tools
from db import list_tables, describe_table, query_financials


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
# AGENT SETUP
# ============================================================================

SYSTEM_PROMPT = """You are a financial analyst that creates polished HTML slide decks.

WORKFLOW:
1. Load the "langchain-brand-slides" skill to get the design system and templates
2. Query the financial database to get the data you need (use list_tables, describe_table, query_financials)
3. Build a complete self-contained HTML slide deck following the skill's design rules
4. Call generate_slides with the full HTML string to render it to PNG images
5. Write a 2-3 sentence summary of the key financial highlights shown in the deck

RULES:
- Always load the skill first to get the latest design templates
- Always query the database — do not make up financial numbers
- Create 2-3 slides: title slide + 1-2 metric/content slides
- Format currency as $3.4M, percentages as 24.6%
- Show trends (up/down/flat) when comparing time periods
"""

agent = create_deep_agent(
    name="financial-slide-agent",
    model="claude-sonnet-4-5-20250929",
    tools=[list_tables, describe_table, query_financials, generate_slides],
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
    Invoke the Deep Agent, attach PNGs to the LangSmith trace.

    Datadog context is passed via langsmith_extra from the server.
    """
    global _last_slide_pngs
    _last_slide_pngs = []

    run_tree = get_current_run_tree()

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract text response from last AI message
    messages = result.get("messages", [])
    text_summary = "Slide deck generated successfully."
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            # Skip tool results — find the last AI text
            if not hasattr(msg, "name"):
                text_summary = msg.content
                break

    # Convert PNGs to base64 for the frontend
    slide_pngs_base64 = [base64.b64encode(png).decode() for png in _last_slide_pngs]

    # Attach PNGs to the LangSmith run
    if run_tree and _last_slide_pngs:
        run_tree.attachments = {
            f"slide_{i+1}": Attachment(mime_type="image/png", data=png)
            for i, png in enumerate(_last_slide_pngs)
        }

    run_id = str(run_tree.id) if run_tree else None

    if run_tree:
        run_tree.outputs = {
            "response": text_summary,
            "slide_pngs_base64": slide_pngs_base64,
        }

    return {
        "response": text_summary,
        "slide_pngs_base64": slide_pngs_base64,
        "run_id": run_id,
    }
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
uv run python -c "from agent import agent; print('Agent loaded:', agent.name)"
```

Expected: `Agent loaded: financial-slide-agent`

- [ ] **Step 3: Commit**

```bash
git add agent.py
git commit -m "feat: rewrite agent as Deep Agent with HTML slides and SQLite"
```

---

### Task 5: Update server.py for thread support

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Update ChatRequest to include thread_id and update the /chat endpoint**

In `server.py`, change the `ChatRequest` model:

```python
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
```

Update the `/chat` endpoint — change the agent import and invocation:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from agent import invoke_agent

    current_span = tracer.current_span() if DATADOG_ENABLED and tracer else None
    dd_context = get_datadog_trace_context()

    print(f"\n--- Slide Request ---")
    print(f"Message: {request.message[:100]}")
    print(f"Thread: {request.thread_id}")
    print(f"Datadog Trace ID: {dd_context['trace_id']}")

    result = await invoke_agent(
        message=request.message,
        thread_id=request.thread_id,
        langsmith_extra={
            "metadata": {
                "datadog_trace_id": dd_context["trace_id"],
                "datadog_span_id": dd_context["span_id"],
                "datadog_trace_url": dd_context["trace_url"],
            }
        },
    )

    # Build LangSmith URL from run_id
    run_id = result.get("run_id")
    ws_id = os.getenv("LS_WORKSPACE_ID", "default")
    proj_id = os.getenv("LS_PROJECT_ID", "default")
    ls_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    ls_ui_base = ls_endpoint.replace("api.", "").replace("/api", "")
    langsmith_url = (
        f"{ls_ui_base}/o/{ws_id}/projects/p/{proj_id}/r/{run_id}"
        if run_id else None
    )

    if current_span:
        current_span.set_tag("langsmith.run_id", run_id)
        current_span.set_tag("langsmith.url", langsmith_url)
        current_span.set_tag("langsmith.project", os.getenv("LANGSMITH_PROJECT", "default"))
        current_span.set_tag("slides.count", len(result.get("slide_pngs_base64", [])))

    print(f"LangSmith Run ID: {run_id}")
    print(f"Slides generated: {len(result.get('slide_pngs_base64', []))}")
    print(f"--- End Request ---\n")

    return ChatResponse(
        response=result["response"],
        slide_pngs_base64=result.get("slide_pngs_base64", []),
        langsmith_run_id=run_id,
        langsmith_url=langsmith_url,
        datadog_trace_id=dd_context["trace_id"],
        datadog_trace_url=dd_context["trace_url"],
    )
```

The key change is: `from agent import invoke_agent` (was `invoke_slide_agent`), and passing `thread_id=request.thread_id`.

- [ ] **Step 2: Commit**

```bash
git add server.py
git commit -m "feat: add thread_id support to /chat endpoint"
```

---

### Task 6: Rewrite frontend with thread sidebar and multi-turn chat

**Files:**
- Rewrite: `static/index.html`

- [ ] **Step 1: Write the new index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Slide Generator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            height: 100vh;
            display: flex;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background: #161b22;
            border-right: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            padding: 1rem;
        }
        .sidebar h2 {
            font-size: 0.9rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }
        .new-thread-btn {
            width: 100%;
            padding: 0.6rem;
            border-radius: 8px;
            border: 1px solid #30363d;
            background: #21262d;
            color: #c9d1d9;
            font-size: 0.85rem;
            cursor: pointer;
            margin-bottom: 0.75rem;
        }
        .new-thread-btn:hover { background: #30363d; }
        .thread-list {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .thread-item {
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            color: #8b949e;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .thread-item:hover { background: #21262d; color: #c9d1d9; }
        .thread-item.active { background: #1f6feb22; color: #58a6ff; }

        /* Main */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 1.5rem 2rem;
        }
        .header {
            margin-bottom: 1rem;
        }
        .header h1 { font-size: 1.3rem; color: #f0f6fc; }
        .header .subtitle { color: #8b949e; font-size: 0.85rem; }
        .status { font-size: 0.8rem; color: #8b949e; margin-bottom: 0.75rem; }
        .status .dot {
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; margin-right: 4px;
        }
        .status .dot.green { background: #3fb950; }
        .status .dot.red { background: #f85149; }

        .examples {
            margin-bottom: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .example-btn {
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            border: 1px solid #30363d;
            background: #161b22;
            color: #8b949e;
            font-size: 0.8rem;
            cursor: pointer;
        }
        .example-btn:hover { border-color: #58a6ff; color: #c9d1d9; }

        .messages {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .message {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            max-width: 90%;
            line-height: 1.5;
        }
        .message.user {
            background: #1f6feb;
            color: #fff;
            align-self: flex-end;
        }
        .message.assistant {
            background: #161b22;
            border: 1px solid #30363d;
            align-self: flex-start;
        }
        .message.assistant.loading {
            color: #8b949e;
            font-style: italic;
        }
        .slide-gallery {
            margin-top: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .slide-img {
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        .slide-label {
            font-size: 0.75rem;
            color: #8b949e;
            margin-bottom: 0.25rem;
        }
        .trace-links {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #30363d;
            font-size: 0.8rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .trace-links a { color: #58a6ff; text-decoration: none; }
        .trace-links a:hover { text-decoration: underline; }
        .input-row { display: flex; gap: 0.5rem; }
        input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid #30363d;
            background: #161b22;
            color: #c9d1d9;
            font-size: 1rem;
            outline: none;
        }
        input[type="text"]:focus { border-color: #1f6feb; }
        button.send-btn {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: none;
            background: #238636;
            color: #fff;
            font-size: 1rem;
            cursor: pointer;
        }
        button.send-btn:hover { background: #2ea043; }
        button.send-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading-dots { animation: pulse 1.5s infinite; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Threads</h2>
        <button class="new-thread-btn" onclick="createThread()">+ New Thread</button>
        <div class="thread-list" id="thread-list"></div>
    </div>

    <div class="main">
        <div class="header">
            <h1>Financial Slide Generator</h1>
            <p class="subtitle">AI slide decks — traced across Datadog APM + LangSmith</p>
        </div>
        <div class="status" id="status"></div>
        <div class="examples">
            <button class="example-btn" onclick="sendExample(this)">Q4 2024 Business Review</button>
            <button class="example-btn" onclick="sendExample(this)">SaaS Investor Update</button>
            <button class="example-btn" onclick="sendExample(this)">E-commerce Category Review</button>
        </div>
        <div class="messages" id="messages"></div>
        <form class="input-row" id="chat-form">
            <input type="text" id="input" placeholder="Describe the financial deck you want..." autocomplete="off" />
            <button type="submit" class="send-btn" id="send-btn">Send</button>
        </form>
    </div>

    <script>
        // ── State ──
        const threads = {};       // { threadId: { name, messages: [{role, content, opts}] } }
        let activeThreadId = null;

        // ── RUM ──
        (async function initRUM() {
            try {
                const res = await fetch("/config");
                const config = await res.json();
                if (config.dd_rum_enabled && config.dd_rum_client_token) {
                    const script = document.createElement("script");
                    script.src = "https://www.datadoghq-browser-agent.com/us1/v5/datadog-rum.js";
                    script.onload = function () {
                        window.DD_RUM && window.DD_RUM.init({
                            clientToken: config.dd_rum_client_token,
                            applicationId: config.dd_rum_application_id,
                            site: config.dd_site,
                            service: config.dd_service,
                            env: config.dd_env,
                            sessionSampleRate: 100,
                            sessionReplaySampleRate: 100,
                            trackUserInteractions: true,
                            trackResources: true,
                            trackLongTasks: true,
                            allowedTracingUrls: [window.location.origin],
                        });
                        setStatus(true, "Datadog RUM active");
                    };
                    document.head.appendChild(script);
                } else {
                    setStatus(true, "Server connected (RUM not configured)");
                }
            } catch {
                setStatus(false, "Could not reach server");
            }
        })();

        function setStatus(ok, text) {
            document.getElementById("status").innerHTML =
                `<span class="dot ${ok ? "green" : "red"}"></span>${text}`;
        }

        // ── Thread Management ──
        function genId() { return crypto.randomUUID(); }

        function createThread() {
            const id = genId();
            threads[id] = { name: "New thread", messages: [] };
            switchThread(id);
            renderThreadList();
            return id;
        }

        function switchThread(id) {
            activeThreadId = id;
            renderThreadList();
            renderMessages();
            document.getElementById("input").focus();
        }

        function renderThreadList() {
            const list = document.getElementById("thread-list");
            list.innerHTML = "";
            for (const [id, thread] of Object.entries(threads)) {
                const el = document.createElement("div");
                el.className = "thread-item" + (id === activeThreadId ? " active" : "");
                el.textContent = thread.name;
                el.onclick = () => switchThread(id);
                list.appendChild(el);
            }
        }

        // ── Messages ──
        const messagesEl = document.getElementById("messages");
        const form = document.getElementById("chat-form");
        const inputEl = document.getElementById("input");
        const sendBtn = document.getElementById("send-btn");

        function renderMessages() {
            messagesEl.innerHTML = "";
            if (!activeThreadId || !threads[activeThreadId]) return;
            for (const msg of threads[activeThreadId].messages) {
                appendMessageEl(msg.role, msg.content, msg.opts || {});
            }
        }

        function appendMessageEl(role, content, opts) {
            const div = document.createElement("div");
            div.className = `message ${role}`;
            if (opts.loading) div.classList.add("loading");

            const textP = document.createElement("p");
            textP.textContent = content;
            div.appendChild(textP);

            if (opts.slidePngs && opts.slidePngs.length > 0) {
                const gallery = document.createElement("div");
                gallery.className = "slide-gallery";
                opts.slidePngs.forEach((b64, i) => {
                    const label = document.createElement("div");
                    label.className = "slide-label";
                    label.textContent = `Slide ${i + 1}`;
                    gallery.appendChild(label);
                    const img = document.createElement("img");
                    img.src = `data:image/png;base64,${b64}`;
                    img.alt = `Slide ${i + 1}`;
                    img.className = "slide-img";
                    gallery.appendChild(img);
                });
                div.appendChild(gallery);
            }

            if (opts.traceLinks) {
                const links = document.createElement("div");
                links.className = "trace-links";
                links.innerHTML = opts.traceLinks;
                div.appendChild(links);
            }

            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            return div;
        }

        function storeMessage(role, content, opts) {
            if (!activeThreadId) return;
            threads[activeThreadId].messages.push({ role, content, opts });
        }

        // ── Chat ──
        async function sendChat(msg) {
            if (!activeThreadId) createThread();

            // Name the thread after the first message
            if (threads[activeThreadId].messages.length === 0) {
                threads[activeThreadId].name = msg.substring(0, 40) + (msg.length > 40 ? "..." : "");
                renderThreadList();
            }

            storeMessage("user", msg);
            appendMessageEl("user", msg, {});
            inputEl.value = "";
            sendBtn.disabled = true;
            sendBtn.textContent = "...";

            const loadingEl = appendMessageEl("assistant",
                "Generating slides... this may take 15-30 seconds",
                { loading: true }
            );

            try {
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: msg, thread_id: activeThreadId }),
                });
                const data = await res.json();
                loadingEl.remove();

                let links = "";
                if (data.datadog_trace_url)
                    links += `<a href="${data.datadog_trace_url}" target="_blank">Datadog Trace</a>`;
                if (data.langsmith_url)
                    links += `<a href="${data.langsmith_url}" target="_blank">LangSmith Trace</a>`;

                const opts = { slidePngs: data.slide_pngs_base64, traceLinks: links || null };
                storeMessage("assistant", data.response, opts);
                appendMessageEl("assistant", data.response, opts);
            } catch (err) {
                loadingEl.remove();
                storeMessage("assistant", "Error: " + err.message);
                appendMessageEl("assistant", "Error: " + err.message, {});
            }

            sendBtn.disabled = false;
            sendBtn.textContent = "Send";
            inputEl.focus();
        }

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const msg = inputEl.value.trim();
            if (msg) sendChat(msg);
        });

        function sendExample(btn) {
            const text = btn.textContent;
            const prompts = {
                "Q4 2024 Business Review": "Create a Q4 2024 Business Review with 3 slides: title slide, revenue metrics (Revenue, Gross Profit, Net Income), and growth metrics (Customers, Employees, Net Margin %). Use Q4 2024 data, show trends vs Q3.",
                "SaaS Investor Update": "Create a SaaS Investor Update for Dec 2024 with 3 slides: title, revenue metrics (MRR, ARR, Net New MRR), and unit economics (CAC, LTV, LTV/CAC Ratio, Churn Rate). Show trends vs Nov.",
                "E-commerce Category Review": "Create an E-commerce Category Review with 2 slides: title slide and top 3 categories by revenue with YoY Growth %.",
            };
            createThread();
            sendChat(prompts[text] || `Create a ${text}`);
        }

        // Start with one thread
        createThread();
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: rewrite frontend with thread sidebar and multi-turn chat"
```

---

### Task 7: Update langgraph.json

**Files:**
- Modify: `langgraph.json`

- [ ] **Step 1: Update the graph export**

```json
{
  "dependencies": ".",
  "graphs": {
    "financial_slide_agent": "./agent.py:agent"
  },
  "env": "../.env"
}
```

No change needed — the export name and path stay the same since we named the new agent variable `agent` in agent.py.

- [ ] **Step 2: Commit (only if changed)**

---

### Task 8: Update run.sh

**Files:**
- Modify: `run.sh`

- [ ] **Step 1: Replace LibreOffice check with Playwright check, add playwright install**

Replace the entire LibreOffice check block (lines 38-54) with:

```bash
# Ensure Playwright browsers are installed
echo -e "${YELLOW}Checking Playwright browsers...${NC}"
uv run playwright install chromium --with-deps 2>/dev/null || {
    echo -e "${YELLOW}Installing Playwright Chromium...${NC}"
    uv run playwright install chromium
}
```

- [ ] **Step 2: Commit**

```bash
git add run.sh
git commit -m "chore: replace LibreOffice with Playwright in run.sh"
```

---

### Task 9: Smoke test the full stack

- [ ] **Step 1: Start the Datadog agent (if not running)**

```bash
docker start dd-agent 2>/dev/null || docker run -d --name dd-agent \
  -e DD_API_KEY=1c01ed64e7a9f91cf36a74b4713bd2b6 \
  -e DD_APM_ENABLED=true \
  -e DD_APM_NON_LOCAL_TRAFFIC=true \
  -e DD_SITE=us5.datadoghq.com \
  -e DD_HOSTNAME=langsmith-demo-local \
  -p 8126:8126 \
  gcr.io/datadoghq/agent:latest
```

- [ ] **Step 2: Start the server**

```bash
bash run.sh
```

Expected: Server starts on port 8001, no import errors, Playwright check passes.

- [ ] **Step 3: Test in browser**

Open http://localhost:8001. Verify:
- Thread sidebar appears with one default thread
- Example buttons work and create new threads
- Slides render as PNGs inline in the chat
- Second message in same thread gets conversation context
- Creating a new thread works
- Switching between threads shows correct history

- [ ] **Step 4: Verify LangSmith traces**

Check https://smith.langchain.com in the `financial-slides-agent` project:
- Root trace is `financial_slide_agent` (from @traceable)
- Child runs show: Deep Agent planning, skill loading, SQL queries, LLM calls, generate_slides tool
- PNG attachments visible on the root run

- [ ] **Step 5: Verify Datadog traces**

Check https://app.us5.datadoghq.com/apm/traces filtered by `service:financial-slide-agent-demo`:
- Trace appears with `langsmith.run_id` and `langsmith.url` tags in span metadata

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete Deep Agent upgrade with HTML slides, SQLite, and threads"
```
