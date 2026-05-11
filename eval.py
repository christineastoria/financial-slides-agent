"""
Offline evaluations for the Financial Slide Agent.

Nine evaluator types:
1. Slide quality        — generic LLM judge on data presence, readability, polish (first-pass catch-all)
2. Trajectory           — validates tool call sequence and completeness
3. Assertions           — LLM judge checks per-example goals are met
4. Overflow/overlap     — detects text clipping, element collisions, content outside slide bounds
5. Chart accuracy       — verifies chart proportions match data (pie slices, bar heights, trend arrows)
6. Data integrity       — checks numbers are consistent, correctly formatted, and not fabricated
7. Readability          — font sizes, contrast, visual hierarchy, color-blind accessibility
8. Regulatory           — date labeling, source attribution, no misleading presentation, consistent formatting
9. Golden slide compare — compares generated slides against golden reference images stored as dataset attachments

Runs experiments across 4 models with structured metadata for the LangSmith UI.

Usage:
    cd financial-slides
    uv run python eval.py
"""

import asyncio
import base64
import os
import uuid

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate, traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.schemas import Attachment

client = Client()

# ---------------------------------------------------------------------------
# Dataset — updated for sandbox + enterprise system connectors
# ---------------------------------------------------------------------------

DATASET_NAME = "Financial Slide Agent Evals v2"

EXAMPLES = [
    {
        "inputs": {
            "message": (
                "Create a Q4 2024 Executive Board Pack with P&L summary, "
                "revenue trends across regions, SaaS metrics, and 2025 "
                "quarterly revenue projections calculated using Q1-Q4 growth rates."
            ),
        },
        "outputs": {
            "assertions": [
                "Slides should contain Q4 2024 revenue and net income from the database",
                "Should show regional revenue breakdown (North America, EMEA, APAC)",
                "Should include SaaS metrics (MRR, ARR, churn rate)",
                "Should include 2025 projected revenue figures computed in the sandbox",
                "Title slide should reference Q4 2024 Executive Board Pack",
                "Should pull data from enterprise systems (core banking or treasury)",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "retrieve_source_documents",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_core_banking_ledger",
                "fetch_risk_exposure_report",
                "pull_treasury_positions",
                "get_regulatory_capital_metrics",
                "run_financial_calculation",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Build a comprehensive Risk & Capital Adequacy dashboard showing "
                "firm-wide VaR, credit exposure by counterparty tier, Basel III "
                "capital ratios, and stress test results. Calculate risk-adjusted "
                "return on capital (RAROC) using the sandbox."
            ),
        },
        "outputs": {
            "assertions": [
                "Should display VaR metrics (1-day, 10-day) from the risk platform",
                "Should show credit exposure broken down by counterparty rating tier",
                "Should include Basel III capital ratios (CET1, Tier 1, Total Capital)",
                "Should show DFAST/CCAR stress test scenario results",
                "Should include a RAROC calculation computed in the sandbox",
                "Should include risk-weighted asset breakdown",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "retrieve_source_documents",
                "fetch_risk_exposure_report",
                "get_regulatory_capital_metrics",
                "list_tables",
                "describe_table",
                "query_financials",
                "run_financial_calculation",
                "generate_slides",
            ],
            "has_golden_slides": True,
        },
    },
    {
        "inputs": {
            "message": (
                "Generate a Treasury & Liquidity Overview deck showing global cash "
                "positions by currency, FX exposure and hedge ratios, investment "
                "portfolio yields, debt structure with covenant compliance, and "
                "regulatory liquidity ratios (LCR, NSFR)."
            ),
        },
        "outputs": {
            "assertions": [
                "Should show cash positions across multiple currencies (USD, EUR, GBP, JPY)",
                "Should display FX exposure with hedge ratios",
                "Should include investment portfolio with yields and maturities",
                "Should show debt/funding facilities with utilization rates",
                "Should include LCR and NSFR regulatory liquidity metrics",
                "Should show covenant compliance status",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "retrieve_source_documents",
                "pull_treasury_positions",
                "get_regulatory_capital_metrics",
                "query_core_banking_ledger",
                "run_financial_calculation",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Create an Investor Update with SaaS unit economics (CAC, LTV, "
                "LTV/CAC ratio, churn), customer cohort retention analysis, "
                "e-commerce category performance, and a 12-month forward revenue "
                "projection using compound monthly growth computed in the sandbox."
            ),
        },
        "outputs": {
            "assertions": [
                "Should display CAC, LTV, and LTV/CAC ratio from SaaS metrics",
                "Should show customer cohort retention rates",
                "Should include e-commerce category revenue and margins",
                "Should include a 12-month forward revenue projection from sandbox",
                "Should show churn rate trends",
                "Projection should use compound growth methodology",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "retrieve_source_documents",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_financials",
                "query_financials",
                "run_financial_calculation",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Build a Regulatory Compliance & Stress Test deck pulling capital "
                "metrics from the regulatory system, risk exposures from the risk "
                "platform, and headcount/burn rate from the database. Use the sandbox "
                "to model a 200bps interest rate shock impact on our DV01 exposure "
                "and show resulting capital ratio degradation."
            ),
        },
        "outputs": {
            "assertions": [
                "Should show Basel III capital ratios from regulatory system",
                "Should include risk exposure data (VaR, credit exposure)",
                "Should display headcount and burn rate from database",
                "Should include a 200bps rate shock scenario calculated in sandbox",
                "Should show post-shock capital ratio impact",
                "Should include stress test pass/fail assessment",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "retrieve_source_documents",
                "get_regulatory_capital_metrics",
                "fetch_risk_exposure_report",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_financials",
                "run_financial_calculation",
                "generate_slides",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Models — 4 models to compare
# ---------------------------------------------------------------------------

MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    ("claude-sonnet-4-5-20250929", 0.5),  # same model, higher temperature
]

# All tools the agent exposes — used in experiment metadata
TOOL_DEFINITIONS = [
    {
        "name": "retrieve_source_documents",
        "description": "Retrieve all .txt source documents from the sources directory. Traced as a retriever in LangSmith.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_tables",
        "description": "List all available tables in the financial database",
    },
    {
        "name": "describe_table",
        "description": "Describe a table's columns and show sample rows",
        "parameters": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
        },
    },
    {
        "name": "query_financials",
        "description": "Execute read-only SQL query against the financial database",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "run_financial_calculation",
        "description": "Execute Python code in a secure LangSmith sandbox for financial calculations",
        "parameters": {
            "type": "object",
            "properties": {"python_code": {"type": "string"}},
            "required": ["python_code"],
        },
    },
    {
        "name": "query_core_banking_ledger",
        "description": "Query Oracle Flexcube core banking GL for account balances and loan metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "account_type": {"type": "string", "enum": ["assets", "liabilities", "equity", "all"]},
                "as_of_date": {"type": "string"},
            },
        },
    },
    {
        "name": "fetch_risk_exposure_report",
        "description": "Query Murex MX.3 risk platform for VaR, credit exposure, and stress tests",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_id": {"type": "string"},
                "risk_type": {"type": "string", "enum": ["market", "credit", "operational", "liquidity", "all"]},
            },
        },
    },
    {
        "name": "pull_treasury_positions",
        "description": "Query Kyriba TMS for cash positions, FX exposure, investments, and funding",
        "parameters": {
            "type": "object",
            "properties": {
                "currency": {"type": "string"},
                "position_type": {"type": "string", "enum": ["cash", "fx", "investments", "funding", "all"]},
            },
        },
    },
    {
        "name": "get_regulatory_capital_metrics",
        "description": "Query AxiomSL for Basel III capital adequacy, liquidity ratios, and stress tests",
        "parameters": {
            "type": "object",
            "properties": {
                "reporting_period": {"type": "string"},
            },
        },
    },
    {
        "name": "generate_slides",
        "description": "Render HTML slide deck to PNG images via Playwright",
        "parameters": {
            "type": "object",
            "properties": {"html": {"type": "string"}},
            "required": ["html"],
        },
    },
]


GOLDEN_SLIDES_DIR = os.path.join(os.path.dirname(__file__), "tmp")


def _load_golden_slides() -> dict[str, tuple[str, bytes]]:
    """Load golden reference PNGs from tmp/ as LangSmith attachments."""
    attachments = {}
    for f in sorted(os.listdir(GOLDEN_SLIDES_DIR)):
        if f.startswith("golden_slide_") and f.endswith(".png"):
            path = os.path.join(GOLDEN_SLIDES_DIR, f)
            with open(path, "rb") as fh:
                attachments[f.replace(".png", "")] = ("image/png", fh.read())
    return attachments


def ensure_dataset():
    """Create or update the evaluation dataset in LangSmith."""
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        client.delete_dataset(dataset_id=datasets[0].id)
        print(f"Deleted existing dataset '{DATASET_NAME}'")

    ds = client.create_dataset(
        DATASET_NAME,
        description="Financial slide agent eval — sandbox, enterprise connectors, SQL database",
    )

    golden_attachments = _load_golden_slides()
    print(f"Loaded {len(golden_attachments)} golden reference slides as attachments")

    for e in EXAMPLES:
        attachments = None
        if e["outputs"].get("has_golden_slides") and golden_attachments:
            attachments = golden_attachments

        client.create_example(
            inputs=e["inputs"],
            outputs=e["outputs"],
            dataset_id=ds.id,
            attachments=attachments,
        )

    print(f"Dataset '{DATASET_NAME}' created with {len(EXAMPLES)} examples")
    return ds


# ---------------------------------------------------------------------------
# Run function (parameterised by model)
# ---------------------------------------------------------------------------

_current_slide_pngs_b64: list[str] = []


def make_target(model: str, temperature: float = 0):
    """Return a run function that invokes the agent with the given model."""

    async def _ainvoke(inputs: dict) -> dict:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        from db import list_tables, describe_table, query_financials
        from agent import (
            retrieve_source_documents,
            generate_slides,
            run_financial_calculation,
            query_core_banking_ledger,
            fetch_risk_exposure_report,
            pull_treasury_positions,
            get_regulatory_capital_metrics,
            SYSTEM_PROMPT,
        )
        import agent as agent_module

        agent_module._last_slide_pngs = []

        llm = ChatAnthropic(model=model, temperature=temperature) if temperature > 0 else model

        eval_agent = create_deep_agent(
            name="financial-slide-agent-eval",
            model=llm,
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

        thread_id = f"eval-{uuid.uuid4()}"
        result = await eval_agent.ainvoke(
            {"messages": [HumanMessage(content=inputs["message"])]},
            config={"configurable": {"thread_id": thread_id}},
        )

        messages = result.get("messages", [])

        # Extract tool-call trajectory
        trajectory = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    trajectory.append(tc["name"])

        # Extract final text response
        text_response = ""
        for msg in reversed(messages):
            if hasattr(msg, "name"):
                continue
            if not hasattr(msg, "content"):
                continue
            content = msg.content
            if isinstance(content, str) and content.strip():
                text_response = content
                break
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                joined = "\n".join(text_parts).strip()
                if joined:
                    text_response = joined
                    break

        raw_pngs = agent_module._last_slide_pngs
        slide_pngs_b64 = [base64.b64encode(png).decode() for png in raw_pngs]

        return {
            "response": text_response,
            "slide_pngs_base64": slide_pngs_b64,
            "trajectory": trajectory,
            "num_slides": len(slide_pngs_b64),
        }

    @traceable(name="slide_agent_eval")
    def target(inputs: dict, **kwargs) -> dict:
        global _current_slide_pngs_b64
        result = asyncio.run(_ainvoke(inputs))

        _current_slide_pngs_b64 = result.pop("slide_pngs_base64", [])

        rt = get_current_run_tree()
        if rt and _current_slide_pngs_b64:
            rt.attachments = {
                f"slide_{i + 1}": Attachment(
                    mime_type="image/png",
                    data=base64.b64decode(b64),
                )
                for i, b64 in enumerate(_current_slide_pngs_b64)
            }

        return result

    return target


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_judge = ChatOpenAI(model="gpt-4.1", max_tokens=1024)


# -- 1. Multi-modal: LLM judge on slide images --------------------------------

class SlideQualityGrade(BaseModel):
    reasoning: str = Field(description="What you observe in the slides")
    data_present: bool = Field(
        description="True if slides contain real financial data, not placeholders"
    )
    visually_readable: bool = Field(
        description="True if text is legible and the layout is clean"
    )
    professional: bool = Field(
        description="True if the slides look polished and well-designed"
    )


_slide_judge = _judge.with_structured_output(SlideQualityGrade)


def slide_quality_evaluator(run, example):
    """Multi-modal eval: send rendered slide PNGs to an LLM vision judge."""
    pngs_b64 = _current_slide_pngs_b64

    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "Evaluate these financial presentation slides. Check:\n"
                "1. Do they contain real financial data (numbers, metrics), not lorem ipsum?\n"
                "2. Is the text legible and the layout clean?\n"
                "3. Do they look professional and well-designed?"
            ),
        },
    ]
    for b64 in pngs_b64[:4]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    grade = _slide_judge.invoke([{"role": "user", "content": content}])
    checks = [grade.data_present, grade.visually_readable, grade.professional]
    score = sum(checks) / len(checks)
    return {"score": score, "comment": grade.reasoning}


# -- 2. Trajectory: validate tool call sequence --------------------------------

def trajectory_evaluator(run, example):
    """Compare actual tool call trajectory against reference trajectory."""
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    ex_outputs = (
        example.outputs
        if hasattr(example, "outputs")
        else example.get("outputs", {}) or {}
    )

    actual = outputs.get("trajectory", [])
    expected = ex_outputs.get("expected_trajectory", [])

    if not actual:
        return {"score": 0, "comment": "No tool calls recorded"}
    if not expected:
        return {"score": 0, "comment": "No reference trajectory in dataset"}

    # Tool set coverage
    expected_set = set(expected)
    actual_set = set(actual)
    missing = expected_set - actual_set
    coverage = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    # Order score via LCS
    def lcs_len(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = lcs_len(actual, expected)
    order_score = lcs / max(len(expected), len(actual)) if (expected or actual) else 1.0

    # Length ratio
    len_ratio = min(len(actual), len(expected)) / max(len(actual), len(expected)) if (expected or actual) else 1.0

    score = (coverage + order_score + len_ratio) / 3

    extra = actual_set - expected_set
    comment = (
        f"Expected: {' -> '.join(expected)}. "
        f"Actual: {' -> '.join(actual)}. "
        f"Coverage: {coverage:.0%}, Order (LCS): {order_score:.0%}, "
        f"Length ratio: {len_ratio:.0%}."
        + (f" Missing: {missing}." if missing else "")
        + (f" Extra: {extra}." if extra else "")
    )
    return {"score": score, "comment": comment}


# -- 3. Assertions: LLM judge checks per-example goals ------------------------

class AssertionGrade(BaseModel):
    reasoning: str = Field(
        description="For each assertion, explain whether it was satisfied"
    )
    fraction_met: float = Field(
        description="Fraction of assertions met, from 0.0 to 1.0"
    )


_assertion_judge = _judge.with_structured_output(AssertionGrade)


def assertion_evaluator(run, example):
    """Multi-modal LLM judge: checks assertions against slide images + text."""
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    ex_outputs = (
        example.outputs
        if hasattr(example, "outputs")
        else example.get("outputs", {}) or {}
    )

    response = outputs.get("response", "")
    num_slides = outputs.get("num_slides", 0)
    pngs_b64 = _current_slide_pngs_b64
    assertions = ex_outputs.get("assertions", [])

    if not assertions:
        return {"score": 1.0, "comment": "No assertions defined"}

    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(assertions))

    content = [
        {
            "type": "text",
            "text": (
                f"An AI agent was asked to create financial slides. "
                f"It produced {num_slides} slide(s) and returned this summary:\n\n"
                f"---\n{response}\n---\n\n"
                f"The rendered slides are attached as images. "
                f"Look at BOTH the text summary AND the slide images to determine "
                f"what fraction of these assertions are satisfied:\n\n"
                f"{numbered}\n\n"
                f"Return fraction_met as a float between 0.0 and 1.0."
            ),
        },
    ]
    for b64 in pngs_b64[:4]:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    grade = _assertion_judge.invoke([{"role": "user", "content": content}])
    return {"score": grade.fraction_met, "comment": grade.reasoning}


# -- 4. Visual: content overlap / overflow ------------------------------------

class OverflowOverlapGrade(BaseModel):
    reasoning: str = Field(description="Describe every overlap or overflow issue found, or state that none were found")
    text_clipped: bool = Field(
        description="True if any text is visually cut off at slide edges or card boundaries"
    )
    elements_overlapping: bool = Field(
        description="True if any elements (cards, text, charts) overlap each other in a way that obscures content"
    )
    content_outside_bounds: bool = Field(
        description="True if any content extends beyond the visible slide area (1280x720)"
    )
    whitespace_balanced: bool = Field(
        description="True if spacing between elements is roughly even — no cramped or empty regions"
    )


_overflow_judge = _judge.with_structured_output(OverflowOverlapGrade)


def overflow_overlap_evaluator(run, example):
    """Visual eval: detect text clipping, element overlap, and content outside slide bounds."""
    pngs_b64 = _current_slide_pngs_b64
    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "You are a QA engineer reviewing financial presentation slides for a bank.\n"
                "Each slide is exactly 1280x720 pixels. Carefully inspect EVERY slide for:\n\n"
                "1. TEXT CLIPPING: Is any text cut off at the edge of a slide or card? "
                "Look at the bottom and right edges especially — long metric values, titles, "
                "or trend labels that extend past their container.\n"
                "2. ELEMENT OVERLAP: Do any cards, text blocks, or chart elements overlap each other "
                "in a way that makes content unreadable? Check metric grids where cards might collide.\n"
                "3. CONTENT OUTSIDE BOUNDS: Is any content partially or fully outside the visible slide area? "
                "Check for elements that appear to be pushed off-screen.\n"
                "4. WHITESPACE BALANCE: Is spacing roughly even? Flag slides where elements are crammed "
                "together in one area but large empty gaps exist elsewhere.\n\n"
                "Be strict — in banking presentations, even minor visual defects undermine credibility."
            ),
        },
    ]
    for b64 in pngs_b64[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _overflow_judge.invoke([{"role": "user", "content": content}])
    defects = sum([
        grade.text_clipped,
        grade.elements_overlapping,
        grade.content_outside_bounds,
        not grade.whitespace_balanced,
    ])
    score = 1.0 - (defects / 4)
    return {"score": score, "comment": grade.reasoning}


# -- 5. Visual: chart proportionality / accuracy -------------------------------

class ChartAccuracyGrade(BaseModel):
    reasoning: str = Field(
        description="For each chart or visual data representation found, describe whether "
        "its visual proportions match the numbers shown. If no charts are present, state that."
    )
    charts_found: int = Field(description="Number of charts or visual data representations found across all slides")
    proportions_accurate: bool = Field(
        description="True if all chart segments, bars, or visual areas are proportional to the data values they represent. "
        "For example, a pie chart with 40% should take up roughly 40% of the circle; "
        "a bar at $3.4M should be roughly 2x a bar at $1.7M."
    )
    labels_match_visuals: bool = Field(
        description="True if every data label/number next to a chart element matches what the visual shows. "
        "No bar labeled $1M that is taller than a bar labeled $3M."
    )
    axes_correct: bool = Field(
        description="True if chart axes (where present) have correct scale, direction, and labels. "
        "Y-axis should increase upward, time axes left-to-right, no missing units."
    )


_chart_judge = _judge.with_structured_output(ChartAccuracyGrade)


def chart_accuracy_evaluator(run, example):
    """Visual eval: check chart proportionality and data-to-visual accuracy."""
    pngs_b64 = _current_slide_pngs_b64
    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "You are a data visualization auditor for a bank. Examine these financial slides.\n\n"
                "For EVERY chart, graph, pie chart, bar chart, or visual data representation:\n\n"
                "1. PROPORTIONALITY: Do the visual sizes match the data? A segment labeled 40% should "
                "take up ~40% of the total area. A bar for $3.4M should be ~2x taller than one for $1.7M. "
                "CSS-styled metric cards with colored bars or progress indicators count too.\n"
                "2. LABEL-VISUAL MATCH: Does every number next to a visual element match what the visual shows? "
                "A green up-arrow next to a negative number is wrong. A bar labeled $1M that appears taller "
                "than a bar labeled $3M is wrong.\n"
                "3. AXES: If axes are present, are they correctly scaled? Y-axis increasing upward, time going "
                "left-to-right, units labeled, no misleading truncated axes.\n\n"
                "If no charts/graphs are present (only metric cards with text), note that and check that "
                "trend indicators (up/down arrows, green/red colors) correctly match the sign of the "
                "percentage changes shown.\n\n"
                "Be precise — in financial presentations, a misleading chart can cause regulatory issues."
            ),
        },
    ]
    for b64 in pngs_b64[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _chart_judge.invoke([{"role": "user", "content": content}])

    if grade.charts_found == 0:
        checks = [grade.labels_match_visuals]
    else:
        checks = [grade.proportions_accurate, grade.labels_match_visuals, grade.axes_correct]
    score = sum(checks) / len(checks)
    return {
        "score": score,
        "comment": f"Charts found: {grade.charts_found}. {grade.reasoning}",
    }


# -- 6. Financial data integrity: numbers match source data --------------------

class DataIntegrityGrade(BaseModel):
    reasoning: str = Field(
        description="For each financial figure on the slides, state whether it appears "
        "consistent with the agent's text summary and whether any numbers look fabricated, "
        "transposed, or contradictory."
    )
    numbers_consistent: bool = Field(
        description="True if the financial figures shown on slides are internally consistent "
        "(e.g., Gross Profit = Revenue - COGS, percentages add to 100% where expected)"
    )
    no_fabricated_data: bool = Field(
        description="True if all numbers appear to be real query results, not round placeholder values "
        "like $1,000,000 or $100K that look made up"
    )
    units_correct: bool = Field(
        description="True if all values have correct units ($ for currency, % for percentages, "
        "bps for basis points) and magnitudes make sense (no $3.4 when $3.4M is meant)"
    )
    trends_directionally_correct: bool = Field(
        description="True if trend indicators (up/down, green/red, +/-) match the direction "
        "of the actual change. A positive change should never show a red down arrow."
    )


_data_integrity_judge = _judge.with_structured_output(DataIntegrityGrade)


def data_integrity_evaluator(run, example):
    """Visual eval: verify financial numbers are consistent, correctly formatted, and not fabricated."""
    pngs_b64 = _current_slide_pngs_b64
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    response = outputs.get("response", "")

    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "You are a financial controller auditing presentation slides before they go to the board.\n\n"
                f"The agent's text summary was:\n---\n{response[:2000]}\n---\n\n"
                "Now examine the slide images and check:\n\n"
                "1. INTERNAL CONSISTENCY: Do the numbers add up? If Revenue is $3.4M and COGS is $1.85M, "
                "then Gross Profit should be ~$1.55M. If a pie chart has 5 segments, they should sum to 100%. "
                "If Q3 Revenue was $3.1M and Q4 is $3.45M, the QoQ growth should be ~11.3%, not some other number.\n"
                "2. NO FABRICATED DATA: Are any numbers suspiciously round or generic-looking? "
                "Real financial data rarely has values like exactly $1,000,000 or $500K. "
                "Look for placeholder-like figures that suggest the agent made up numbers.\n"
                "3. CORRECT UNITS: Every currency value should have $ (or appropriate symbol), "
                "every percentage should have %, basis points should use bps. "
                "Check that magnitudes are sensible — $3.4M and $3,400,000 are fine, $3.4 alone is suspicious.\n"
                "4. TREND DIRECTION: If a metric went up, the trend indicator should be green/up/positive. "
                "If it went down, red/down/negative. Check every single trend indicator against its number."
            ),
        },
    ]
    for b64 in pngs_b64[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _data_integrity_judge.invoke([{"role": "user", "content": content}])
    checks = [
        grade.numbers_consistent,
        grade.no_fabricated_data,
        grade.units_correct,
        grade.trends_directionally_correct,
    ]
    score = sum(checks) / len(checks)
    return {"score": score, "comment": grade.reasoning}


# -- 7. Readability & accessibility for financial content ----------------------

class ReadabilityGrade(BaseModel):
    reasoning: str = Field(description="Describe any readability or accessibility issues found")
    font_sizes_adequate: bool = Field(
        description="True if all text is large enough to read comfortably when projected. "
        "No text smaller than ~10px equivalent. Metric values should be prominent."
    )
    contrast_sufficient: bool = Field(
        description="True if all text has sufficient contrast against its background. "
        "No light gray text on white, no light blue on light blue."
    )
    hierarchy_clear: bool = Field(
        description="True if visual hierarchy is clear — titles are largest, "
        "section headers next, body text smaller, labels smallest. "
        "The most important numbers should visually stand out."
    )
    color_not_sole_indicator: bool = Field(
        description="True if color is NOT the only way to convey meaning. "
        "Trends should have text (+/-/up/down) in addition to red/green coloring "
        "so the information is accessible to color-blind readers."
    )


_readability_judge = _judge.with_structured_output(ReadabilityGrade)


def readability_evaluator(run, example):
    """Visual eval: check font sizes, contrast, hierarchy, and color-blind accessibility."""
    pngs_b64 = _current_slide_pngs_b64
    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "You are reviewing financial presentation slides for readability and accessibility.\n"
                "These slides will be shown to bank executives, projected on screens, and printed.\n\n"
                "Check each slide for:\n\n"
                "1. FONT SIZE: Is all text readable? Metric values should be large and bold. "
                "Labels can be smaller but still legible. Flag any text that looks too small to read.\n"
                "2. CONTRAST: Can you read all text clearly against its background? "
                "Check especially: muted/gray text on light backgrounds, colored text on colored cards.\n"
                "3. VISUAL HIERARCHY: Is it immediately obvious what the most important information is on each slide? "
                "The eye should go to the key numbers first, then supporting details.\n"
                "4. COLOR ACCESSIBILITY: For trend indicators (up/down/positive/negative), "
                "is the meaning conveyed through text AND color, not color alone? "
                "A color-blind person should be able to tell if a trend is positive or negative "
                "from the text (+11.3%, -2.1%) or symbols (arrows) without relying on green/red."
            ),
        },
    ]
    for b64 in pngs_b64[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _readability_judge.invoke([{"role": "user", "content": content}])
    checks = [
        grade.font_sizes_adequate,
        grade.contrast_sufficient,
        grade.hierarchy_clear,
        grade.color_not_sole_indicator,
    ]
    score = sum(checks) / len(checks)
    return {"score": score, "comment": grade.reasoning}


# -- 8. Regulatory & compliance visual checks ----------------------------------

class RegulatoryVisualsGrade(BaseModel):
    reasoning: str = Field(
        description="Describe whether regulatory presentation standards are met"
    )
    dates_labeled: bool = Field(
        description="True if every slide with financial data clearly shows the reporting period "
        "or as-of date (e.g., 'Q4 2024', 'As of Dec 31, 2024')"
    )
    sources_attributed: bool = Field(
        description="True if data sources are identified — either on individual slides "
        "or on a sources/disclaimer slide"
    )
    no_misleading_presentation: bool = Field(
        description="True if no data is presented in a misleading way — "
        "no cherry-picked timeframes, no truncated axes that exaggerate changes, "
        "no mixing of actual and projected data without labeling"
    )
    consistent_formatting: bool = Field(
        description="True if number formatting is consistent across all slides — "
        "same decimal places for percentages, same abbreviation style for currency "
        "(all use $3.4M or all use $3,400,000, not a mix)"
    )


_regulatory_judge = _judge.with_structured_output(RegulatoryVisualsGrade)


def regulatory_visuals_evaluator(run, example):
    """Visual eval: check for date labeling, source attribution, and presentation standards
    that banks require for board/regulatory presentations."""
    pngs_b64 = _current_slide_pngs_b64
    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated"}

    content = [
        {
            "type": "text",
            "text": (
                "You are a compliance officer reviewing financial slides before they are presented "
                "to the Board of Directors or submitted to regulators.\n\n"
                "Check ALL slides for:\n\n"
                "1. DATE LABELING: Every slide showing financial data MUST clearly indicate the reporting "
                "period or as-of date. 'Q4 2024', 'FY2024', 'As of Dec 31, 2024' are all acceptable. "
                "A title slide with just the date is sufficient if content slides reference it, "
                "but undated data slides are a fail.\n"
                "2. SOURCE ATTRIBUTION: Is it clear where the data came from? This can be footnotes, "
                "a dedicated sources slide, or inline labels like 'Source: Core Banking GL'. "
                "Not required on title slides, but any slide with data should have it or reference a source.\n"
                "3. NO MISLEADING PRESENTATION: Look for truncated axes, cherry-picked date ranges, "
                "mixing of actual and projected data without clear labeling, or any other presentation "
                "technique that could mislead the reader about financial performance.\n"
                "4. CONSISTENT FORMATTING: Are numbers formatted the same way across all slides? "
                "Percentages should all use the same precision (24.6% everywhere, not 24.6% on one slide "
                "and 25% on another). Currency should use the same abbreviation style throughout."
            ),
        },
    ]
    for b64 in pngs_b64[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _regulatory_judge.invoke([{"role": "user", "content": content}])
    checks = [
        grade.dates_labeled,
        grade.sources_attributed,
        grade.no_misleading_presentation,
        grade.consistent_formatting,
    ]
    score = sum(checks) / len(checks)
    return {"score": score, "comment": grade.reasoning}


# -- 9. Golden slide comparison (dataset attachments) --------------------------

class GoldenSlideGrade(BaseModel):
    reasoning: str = Field(
        description="For each golden reference slide, compare it to the closest generated slide. "
        "Describe what matches and what differs."
    )
    layout_similar: bool = Field(
        description="True if the generated slides use a similar layout structure to the golden reference — "
        "same general grid arrangement, similar card placement, comparable slide count"
    )
    data_coverage_matches: bool = Field(
        description="True if the generated slides cover the same key metrics shown in the golden reference. "
        "The exact numbers may differ but the same categories of data should be present."
    )
    design_consistent: bool = Field(
        description="True if the generated slides follow the same design system as the golden reference — "
        "same color scheme, font style, card styling, and brand treatment"
    )


_golden_judge = _judge.with_structured_output(GoldenSlideGrade)


def golden_slide_evaluator(outputs: dict, attachments: dict, reference_outputs: dict) -> dict:
    """Compare generated slides against golden reference images stored as dataset attachments.

    Uses the LangSmith dataset attachments feature — golden PNGs are uploaded
    with the dataset example and passed to the evaluator automatically via the
    `attachments` parameter.
    """
    pngs_b64 = _current_slide_pngs_b64

    if not attachments:
        return {"score": 1.0, "comment": "No golden reference attached to this example — skipped"}
    if not pngs_b64:
        return {"score": 0, "comment": "No slides were generated to compare"}

    golden_images = []
    for name in sorted(attachments.keys()):
        if attachments[name].get("mime_type", "").startswith("image/"):
            reader = attachments[name]["reader"]
            golden_b64 = base64.b64encode(reader.read()).decode()
            golden_images.append((name, golden_b64))

    if not golden_images:
        return {"score": 1.0, "comment": "No image attachments found — skipped"}

    content = [
        {
            "type": "text",
            "text": (
                "You are comparing GENERATED financial slides against GOLDEN REFERENCE slides.\n\n"
                f"There are {len(golden_images)} golden reference slide(s) and "
                f"{len(pngs_b64)} generated slide(s).\n\n"
                "The images below are labeled. Compare them and evaluate:\n"
                "1. LAYOUT SIMILARITY: Do the generated slides use a similar grid layout, card placement, "
                "and overall structure? They don't need to be pixel-identical, but the approach should match.\n"
                "2. DATA COVERAGE: Do the generated slides cover the same key metrics and data categories "
                "as the golden reference? Exact numbers may differ across runs.\n"
                "3. DESIGN CONSISTENCY: Do both follow the same design system — color palette, typography, "
                "card styling, trend indicator style?\n\n"
                "--- GOLDEN REFERENCE SLIDES ---"
            ),
        },
    ]

    for name, b64 in golden_images[:4]:
        content.append({"type": "text", "text": f"[Golden: {name}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    content.append({"type": "text", "text": "--- GENERATED SLIDES ---"})
    for i, b64 in enumerate(pngs_b64[:4]):
        content.append({"type": "text", "text": f"[Generated: slide {i+1}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    grade = _golden_judge.invoke([{"role": "user", "content": content}])
    checks = [grade.layout_similar, grade.data_coverage_matches, grade.design_consistent]
    score = sum(checks) / len(checks)
    return {
        "score": score,
        "comment": f"Compared {len(golden_images)} golden vs {len(pngs_b64)} generated. {grade.reasoning}",
    }


# ---------------------------------------------------------------------------
# Main — run experiments across 4 models with structured metadata
# ---------------------------------------------------------------------------

def main():
    ensure_dataset()

    for entry in MODELS:
        if isinstance(entry, tuple):
            model, temperature = entry
        else:
            model, temperature = entry, 0

        short = model.replace("claude-", "").replace("-20250929", "").replace("-20250514", "").replace("-20251001", "")
        if temperature > 0:
            short += f"-t{temperature}"
        prefix = f"slide-agent-{short}"

        experiment_metadata = {
            "models": [
                {
                    "id": ["langchain", "chat_models", "anthropic", "ChatAnthropic"],
                    "lc": 1,
                    "type": "constructor",
                    "kwargs": {"model_name": model, "temperature": temperature},
                },
            ],
            "prompts": ["financial-slide-agent/system-prompt:v2"],
            "tools": TOOL_DEFINITIONS,
        }

        print(f"\n{'=' * 60}")
        print(f"Experiment: {prefix}  (model={model}, temp={temperature})")
        print(f"{'=' * 60}\n")

        evaluate(
            make_target(model, temperature),
            data=DATASET_NAME,
            evaluators=[
                slide_quality_evaluator,
                trajectory_evaluator,
                assertion_evaluator,
                overflow_overlap_evaluator,
                chart_accuracy_evaluator,
                data_integrity_evaluator,
                readability_evaluator,
                regulatory_visuals_evaluator,
                golden_slide_evaluator,
            ],
            experiment_prefix=prefix,
            description=f"Financial slide agent eval — {model} (temp={temperature})",
            metadata=experiment_metadata,
            max_concurrency=1,
        )

    print("\nAll experiments complete. View results in LangSmith.")


if __name__ == "__main__":
    main()
