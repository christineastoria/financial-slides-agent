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

SYSTEM_PROMPT = """You are a financial analyst that creates comprehensive HTML slide decks.

IMPORTANT: At the start of every request, use the write_todos tool to create a plan before doing any work.

WORKFLOW:
1. Create a plan using write_todos with the steps you'll take
2. Load the "langchain-brand-slides" skill to get the design system and templates
3. Query the financial database to get ALL relevant data — use list_tables first, then describe and query EVERY table that could be relevant. Always query at least 4-5 tables.
4. Build a complete self-contained HTML slide deck following the skill's design rules
5. Call generate_slides with the full HTML string to render it to PNG images
6. Write a 2-3 sentence summary of the key financial highlights shown in the deck

Mark each todo as completed as you finish it.

RULES:
- Always create a plan with write_todos FIRST
- Always load the skill first to get the latest design templates
- Always query the database — do not make up financial numbers
- Be thorough: pull data from ALL relevant tables, even when they cover similar ground
- When multiple tables have related data (e.g. company_financials AND regional_revenue AND monthly_financials all have revenue figures), include ALL of them — show every angle
- Create 4-6 slides with dense metric grids. Pack each slide with as many metrics as fit.
- Use cols-4 grids wherever possible. More data = better.
- Include data from overlapping sources on the SAME slide when they cover the same topic (e.g. show company-level revenue alongside regional revenue alongside monthly revenue on one slide)
- Format currency as $3.4M, percentages as 24.6%
- Show trends (up/down/flat) when comparing time periods
- When in doubt, add MORE data, not less
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

    # Set trace output to the graph's native result (full messages
    # array with tool_calls, usage_metadata, etc.)
    if run_tree:
        run_tree.outputs = result

    return {
        "response": text_summary,
        "slide_pngs_base64": slide_pngs_base64,
        "run_id": run_id,
    }
