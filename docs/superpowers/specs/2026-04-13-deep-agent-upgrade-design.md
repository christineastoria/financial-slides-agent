# Deep Agent Upgrade: Financial Slide Generator

## Summary

Replace the current LangChain `create_agent` with a Deep Agent (`create_deep_agent`) that generates HTML slides styled to the LangChain brand (light mode), queries financial data from SQLite, supports multi-turn conversation with threads, and preserves all existing Datadog + LangSmith tracing including PNG attachments.

## Architecture

```
Browser (chat UI with thread sidebar)
    |  POST /chat { message, thread_id }
    v
FastAPI server (ddtrace instrumented)
    |  @traceable wrapper for attachments + DD metadata
    v
Deep Agent (create_deep_agent)
    |-- Tools: query_financials, list_tables, describe_table, generate_slides
    |-- Skill: langchain-brand-slides (light-mode design system, loaded on demand)
    |-- Checkpointer: MemorySaver (thread persistence)
    |-- Backend: FilesystemBackend (virtual_mode=True)
    v
LangSmith traces (PNG attachments) + Datadog APM spans
```

## Components

### 1. SQLite Database — `db.py`

New file. Seeds the three existing datasets into `financial_data.db` at import time.

**Tables:**
- `company_financials` — Q1-Q4 2024 revenue, profit, margins, headcount
- `saas_metrics` — Sep-Dec 2024 MRR, ARR, churn, unit economics
- `ecommerce_data` — 5 categories with revenue, orders, margins, growth

**Tools exposed to the agent:**
- `list_tables()` — returns table names
- `describe_table(table_name: str)` — returns column names, types, and sample rows
- `query_financials(sql: str)` — executes read-only SQL, returns formatted results. Rejects writes (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE).

### 2. Deep Agent — `agent.py`

Replace `create_agent` + `@tool`-decorated PPTX functions with:

```python
agent = create_deep_agent(
    name="financial-slide-agent",
    model="claude-sonnet-4-5-20250929",
    tools=[query_financials, list_tables, describe_table, generate_slides],
    system_prompt="...",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    skills=["./skills/"],
    checkpointer=MemorySaver(),
)
```

**System prompt** instructs the agent to:
1. Load the `langchain-brand-slides` skill when creating slides
2. Query the database to get relevant data before building slides
3. Generate a single self-contained HTML string with all slides
4. Call `generate_slides(html)` to render and return PNGs
5. Provide a 2-3 sentence summary after generating

**`generate_slides(html: str)` tool:**
- Takes the agent's HTML string
- Renders it with Playwright (headless Chromium)
- Screenshots each `.slide` element as a PNG
- Returns list of PNG bytes + confirmation message
- This is a custom tool, not a Deep Agent built-in

**`@traceable` wrapper** (`invoke_agent`):
- Wraps the `agent.ainvoke()` call
- Passes `thread_id` in config for conversation persistence
- Attaches PNGs to `run_tree.attachments`
- Sets `run_tree.outputs` with response text + base64 PNGs (for annotation queue custom rendering). Full message trajectory is captured automatically by LangGraph's auto-tracing as child runs — online evaluators access it through the trace hierarchy out of the box.
- Accepts `langsmith_extra` for Datadog metadata injection
- Returns `response`, `slide_pngs_base64`, `run_id`

**Removed:**
- All PPTX tools (create_presentation, add_slide, add_title_text, etc.)
- PresentationBuilder class
- python-pptx dependency
- LibreOffice dependency
- pandas DataFrames (moved to SQLite)

### 3. Skill — `skills/langchain-brand-slides/SKILL.md`

LangChain brand design system adapted to light mode for HTML slides.

**Design tokens (light mode):**
- Page background: `#F2FAFF` (--lc-surface)
- Card background: `#E5F4FF` (--lc-card-alt)
- Heading text: `#030710` (--lc-dark-text)
- Body text: `#6B8299` (--lc-muted)
- Primary accent: `#7FC8FF` (--lc-blue)
- Borders: `#B8DFFF` (--lc-border)
- Positive trend: `#E3FF8F` (--lc-lime)
- Negative trend: `#B27D75` (--lc-rose)

**Slide structure:**
- Each slide is a `<section class="slide">` at 1280x720px (16:9)
- Self-contained HTML with inline `<style>` block
- Google Fonts: Inter (400, 500, 600, 700, 800)
- Geometric line SVG icons (stroke only, no fills)
- Components: metric cards, title slides, content slides, trend indicators

**Content in SKILL.md:**
- Full CSS with all design tokens as custom properties
- HTML templates for each slide type (title, metrics, content)
- Rules for metric card layout and positioning
- Typography scale matching brand guidelines
- Example complete slide deck HTML

### 4. Server — `server.py`

**Preserved (no changes):**
- ddtrace + LLMObs initialization with `patch_all(langchain=False)`
- `get_datadog_trace_context()` function
- Datadog span tagging (langsmith.run_id, langsmith.url, langsmith.project)
- LangSmith URL construction from LANGSMITH_ENDPOINT
- CORS middleware, static file serving
- `/health` and `/config` endpoints

**Modified:**
- `ChatRequest` adds `thread_id: str = "default"` field
- `/chat` endpoint passes `thread_id` to `invoke_agent`
- Response still includes `slide_pngs_base64`, `langsmith_run_id`, `langsmith_url`, `datadog_trace_id`, `datadog_trace_url`

### 5. Frontend — `static/index.html`

**New: Thread sidebar**
- Left panel with thread list
- "New Thread" button creates a UUID thread_id
- Click to switch between threads
- Active thread highlighted
- Thread names auto-generated from first message (truncated)

**New: Multi-turn chat**
- Message history displayed per thread
- User messages and agent responses in chat bubbles
- Slide PNGs rendered inline below agent responses
- Scroll to bottom on new message

**Preserved:**
- Example prompt buttons (start new thread on click)
- Datadog RUM initialization from `/config`
- Trace links display (Datadog + LangSmith)
- Overall visual style

**Modified:**
- POST `/chat` now sends `{ message, thread_id }`
- Messages stored client-side per thread_id (simple JS object, no persistence needed)

### 6. HTML to PNG — Playwright

Replace LibreOffice PPTX-to-PNG pipeline with Playwright HTML-to-PNG:

```python
async def html_to_pngs(html: str) -> list[bytes]:
    browser = await playwright.chromium.launch()
    page = await browser.new_page(viewport={"width": 1280, "height": 720})
    await page.set_content(html)
    slides = await page.query_selector_all(".slide")
    pngs = []
    for slide in slides:
        pngs.append(await slide.screenshot())
    await browser.close()
    return pngs
```

**Dependencies:**
- Add `playwright` to pyproject.toml
- Remove `python-pptx`, `pdf2image`, `Pillow`
- Setup: `playwright install chromium`

### 7. langgraph.json

Update the graph export to point to the new Deep Agent:

```json
{
  "dependencies": ".",
  "graphs": {
    "financial_slide_agent": "./agent.py:agent"
  },
  "env": "../.env"
}
```

## Tracing Flow

1. Frontend sends `{ message, thread_id }` to `/chat`
2. Server captures Datadog trace context
3. Server calls `invoke_agent(message, thread_id, langsmith_extra={metadata: dd_context})`
4. `@traceable` decorator creates LangSmith run, injects DD metadata
5. Deep Agent invokes with thread_id config — LangGraph auto-traces all internal steps as child runs
6. Agent loads skill, queries SQLite, generates HTML, calls generate_slides tool
7. PNGs attached to `run_tree.attachments`
8. `run_tree.outputs` set to `{"response": text, "slide_pngs_base64": [...]}` — full message trajectory (LLM calls, tool calls, tool results) is captured automatically as child runs by LangGraph auto-tracing
9. Server sets Datadog span tags with LangSmith URL
10. Response returned with PNGs + trace links

## Dependencies

**Add:**
- `deepagents`
- `playwright`

**Remove:**
- `python-pptx`
- `pdf2image`
- `Pillow`

**Keep:**
- `ddtrace`, `fastapi`, `langchain`, `langchain-anthropic`, `langsmith`
- `langgraph-cli[inmem]`, `pydantic`, `python-dotenv`, `uvicorn`

**Note:** `pandas` stays for SQLite seeding convenience, or replace with raw SQL inserts.

## Setup Changes

`run.sh` needs one addition after dependency install:
```bash
uv run playwright install chromium
```

## File Changes Summary

| File | Action |
|------|--------|
| `agent.py` | Rewrite: Deep Agent + generate_slides tool + @traceable wrapper |
| `db.py` | New: SQLite setup + seed data + query tools |
| `server.py` | Modify: add thread_id to request, update import |
| `static/index.html` | Rewrite: thread sidebar + multi-turn chat |
| `skills/langchain-brand-slides/SKILL.md` | New: light-mode brand design system + templates |
| `langgraph.json` | Modify: update graph export |
| `pyproject.toml` | Modify: swap dependencies |
| `run.sh` | Modify: add playwright install |
