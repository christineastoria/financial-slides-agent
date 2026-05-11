# Financial Slide Agent — Evaluation Suite

An AI agent that generates polished financial presentation slides (HTML → PNG), backed by a comprehensive evaluation suite designed for **banking and financial services** use cases.

The agent queries multiple data sources (SQL database, core banking, risk platform, treasury, regulatory), builds branded HTML slide decks, and renders them to PNG. The evaluation suite validates everything from tool call trajectories to visual quality of the rendered slides.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Financial Slide Agent (Deep Agent + Claude)            │
│                                                         │
│  Tools:                                                 │
│  ├── retrieve_source_documents  (earnings reports, etc) │
│  ├── list_tables / describe_table / query_financials    │
│  ├── query_core_banking_ledger  (Oracle Flexcube)       │
│  ├── fetch_risk_exposure_report (Murex MX.3)            │
│  ├── pull_treasury_positions    (Kyriba TMS)            │
│  ├── get_regulatory_capital_metrics (AxiomSL)           │
│  ├── run_financial_calculation  (LangSmith Sandbox)     │
│  └── generate_slides            (Playwright HTML→PNG)   │
│                                                         │
│  Output: PNG slide images + LangSmith trace             │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Evaluation Suite (9 evaluators)                        │
│                                                         │
│  Behavioral:                                            │
│  ├── 1. Slide Quality      (generic visual catch-all)   │
│  ├── 2. Trajectory         (tool call sequence)         │
│  └── 3. Assertions         (per-example goal checks)    │
│                                                         │
│  Visual — Banking-Grade:                                │
│  ├── 4. Overflow/Overlap   (clipping, collisions)       │
│  ├── 5. Chart Accuracy     (proportions, labels)        │
│  ├── 6. Data Integrity     (numbers add up, no fakes)   │
│  ├── 7. Readability        (contrast, hierarchy, a11y)  │
│  └── 8. Regulatory         (dates, sources, formatting) │
│                                                         │
│  Golden Reference:                                      │
│  └── 9. Golden Slide Compare (dataset attachment imgs)  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and enter the project
cd financial-slides

# Copy environment template and fill in your keys
cp env.example .env

# Install dependencies
uv sync
uv run playwright install chromium

# Run the full eval suite (5 examples × 4 models × 9 evaluators)
uv run python eval.py

# Run a quick smoke test (1 example × 1 model × 9 evaluators)
uv run python test_eval.py

# Start the web server (for interactive use)
bash run.sh
```

## Environment Variables

Copy `env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key — powers the agent |
| `OPENAI_API_KEY` | Yes | GPT-4.1 API key — powers the LLM judge evaluators |
| `LANGSMITH_API_KEY` | Yes | LangSmith API key — tracing + eval results |
| `LANGSMITH_PROJECT` | Yes | LangSmith project name |
| `DD_API_KEY` | No | Datadog API key (for APM tracing) |

## Evaluation Suite

The eval suite runs the agent end-to-end for each test case, then applies **9 evaluators** to the outputs. Results are uploaded to LangSmith for comparison across models and runs.

### Dataset

5 examples covering different banking presentation types:

| # | Example | Key Data Sources | Golden Slides |
|---|---------|-------------------|---------------|
| 1 | Q4 Executive Board Pack | SQL + all enterprise systems | — |
| 2 | Risk & Capital Adequacy Dashboard | Risk platform + regulatory | Yes (6 slides) |
| 3 | Treasury & Liquidity Overview | Treasury + regulatory + banking | — |
| 4 | Investor Update (SaaS + E-commerce) | SQL (multiple tables) + sandbox | — |
| 5 | Regulatory Compliance & Stress Test | Regulatory + risk + SQL + sandbox | — |

### Evaluators

#### Behavioral Evaluators

**1. Slide Quality** (generic catch-all)
- First-pass visual check — real data present? Text legible? Professional look?
- LLM judge (GPT-4.1) examines slide PNGs
- Good as a baseline; the specialized evaluators below catch what this misses

**2. Trajectory**
- Validates the agent called the right tools in the right order
- Uses LCS (longest common subsequence) for order scoring
- Checks: tool set coverage, call order, trajectory length ratio

**3. Assertions**
- Per-example goal checking — "Should display VaR metrics", "Should include RAROC calculation"
- LLM judge cross-references slide images + text summary against assertion list
- Returns fraction of assertions met (0.0–1.0)

#### Visual Evaluators (Banking-Grade)

These are the evaluators a bank cares about. Each uses a multimodal LLM judge examining the rendered slide PNGs.

**4. Overflow/Overlap**
- Detects text clipped at card/slide edges (especially bottom and right)
- Finds element collisions in metric grids
- Checks for content pushed outside the 1280×720 slide bounds
- Flags unbalanced whitespace (cramped vs. empty regions)
- Scores on 4 boolean criteria → 0.0–1.0

**5. Chart Accuracy**
- Verifies chart proportions match data (a 40% pie slice should be ~40% of the circle)
- Checks bar heights are proportional to their values
- Validates label-to-visual consistency (no bar labeled $1M taller than one labeled $3M)
- Checks axis correctness (direction, scale, units)
- Falls back to trend indicator checks when no charts are present

**6. Data Integrity**
- Checks internal consistency (Revenue − COGS = Gross Profit)
- Detects fabricated/placeholder numbers (suspiciously round values)
- Validates units and magnitudes ($3.4M not $3.4)
- Confirms trend direction matches the sign of the change (green = positive)

**7. Readability**
- Font sizes adequate for projection
- Text contrast against backgrounds
- Visual hierarchy clear (key numbers stand out)
- Color-blind accessibility (trends have text indicators, not just red/green)

**8. Regulatory**
- Every data slide has a reporting period or as-of date
- Data sources are attributed
- No misleading presentation (truncated axes, unlabeled projections)
- Consistent number formatting across all slides

#### Golden Reference Evaluator

**9. Golden Slide Compare** (dataset attachments)
- Compares generated slides against golden reference PNGs stored as **LangSmith dataset attachments**
- Uses the `attachments` parameter in the evaluator function — LangSmith automatically downloads and provides the golden images
- Checks: layout similarity, data coverage match, design system consistency
- Only runs on examples that have golden slides attached (example 2); returns score 1.0 and skips for others

This evaluator demonstrates how to use **dataset attachments** for image-based evaluation — you upload reference images when creating the dataset, and they're automatically available to evaluators at runtime.

### Models Compared

The full eval runs each example against 4 model configurations:

| Model | Temperature | Purpose |
|---|---|---|
| `claude-sonnet-4-5-20250929` | 0 | Primary — best quality |
| `claude-haiku-4-5-20251001` | 0 | Cost comparison |
| `claude-sonnet-4-20250514` | 0 | Previous generation |
| `claude-sonnet-4-5-20250929` | 0.5 | Temperature sensitivity |

### Running Evaluations

```bash
# Full suite: 5 examples × 4 models = 20 agent runs, each scored by 9 evaluators
uv run python eval.py

# Smoke test: 1 example (Risk & Capital with golden slides) × 1 model
uv run python test_eval.py
```

Results appear in LangSmith under the dataset "Financial Slide Agent Evals v2". Each experiment shows:
- Per-evaluator scores with reasoning
- Slide PNG attachments on each trace
- Model/tool metadata for filtering and comparison

## Project Structure

```
financial-slides/
├── agent.py              # Deep Agent — tools, system prompt, traceable wrapper
├── db.py                 # SQLite database with 8 financial tables (seeded on import)
├── server.py             # FastAPI server with Datadog APM + LangSmith tracing
├── eval.py               # Evaluation suite — 9 evaluators, 5 examples, 4 models
├── test_eval.py          # Quick smoke test (1 example, 1 model)
├── run.sh                # Startup script (deps, Playwright, Datadog agent, server)
├── env.example           # Environment variable template
├── langgraph.json        # LangGraph deployment config
├── pyproject.toml        # Dependencies
├── skills/
│   └── langchain-brand-slides/
│       └── SKILL.md      # Design system — colors, typography, slide templates
├── sources/              # Source documents (earnings reports, committee minutes)
│   ├── q4_2024_earnings_report.txt
│   ├── risk_committee_minutes.txt
│   └── treasury_strategy_memo.txt
└── tmp/
    ├── golden_slide_*.png  # Golden reference slides (attached to dataset)
    └── *.html              # Previously generated slide decks
```

## How the Golden Slide Attachments Work

This is the key pattern for image-based evaluation with LangSmith datasets:

**1. Create golden reference images** (one-time setup):
```python
# Render a known-good HTML deck to PNGs
pngs = await html_to_pngs(golden_html)
for i, png in enumerate(pngs):
    with open(f"tmp/golden_slide_{i+1}.png", "wb") as f:
        f.write(png)
```

**2. Upload as dataset attachments** (in `ensure_dataset`):
```python
# Load golden PNGs as (mime_type, bytes) tuples
attachments = {
    "golden_slide_1": ("image/png", open("tmp/golden_slide_1.png", "rb").read()),
    "golden_slide_2": ("image/png", open("tmp/golden_slide_2.png", "rb").read()),
}

# Attach to the relevant dataset example
client.create_example(
    inputs=example["inputs"],
    outputs=example["outputs"],
    dataset_id=ds.id,
    attachments=attachments,  # <-- golden images stored with the example
)
```

**3. Access in evaluator** via the `attachments` parameter:
```python
def golden_slide_evaluator(outputs: dict, attachments: dict, reference_outputs: dict) -> dict:
    # LangSmith auto-downloads attachments when the evaluator signature
    # includes an `attachments` parameter
    for name in sorted(attachments.keys()):
        reader = attachments[name]["reader"]   # io.BytesIO
        image_bytes = reader.read()
        mime_type = attachments[name]["mime_type"]  # "image/png"
        # ... compare against generated slides ...
    return {"score": 0.9, "comment": "Layout matches golden reference"}
```

The readers are automatically reset (`seek(0)`) between evaluators, so multiple evaluators can read the same attachments.

## Observability

The agent traces to both **LangSmith** and **Datadog APM**:

- **LangSmith**: Full agent trace with tool calls, LLM interactions, slide PNG attachments, source document attachments
- **Datadog**: APM trace with `langsmith.run_id` and `langsmith.url` tags for cross-linking
- **LLM Observability**: Token usage, model info, prompt/completion pairs via `ddtrace`

Start the server with `bash run.sh` (with Datadog) or `bash run.sh --no-dd` (LangSmith only).
