"""
Offline evaluations for the Financial Slide Agent.

Three evaluator types:
1. Multi-modal — LLM judge evaluates rendered slide PNGs (via attachments)
2. Trajectory — validates tool call sequence and completeness
3. Assertions — LLM judge checks per-example goals are met

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

from langchain_anthropic import ChatAnthropic
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
                "fetch_risk_exposure_report",
                "get_regulatory_capital_metrics",
                "list_tables",
                "describe_table",
                "query_financials",
                "run_financial_calculation",
                "generate_slides",
            ],
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

    for e in EXAMPLES:
        client.create_example(
            inputs=e["inputs"],
            outputs=e["outputs"],
            dataset_id=ds.id,
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

_judge = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=1024)


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
            ],
            experiment_prefix=prefix,
            description=f"Financial slide agent eval — {model} (temp={temperature})",
            metadata=experiment_metadata,
            max_concurrency=1,
        )

    print("\nAll experiments complete. View results in LangSmith.")


if __name__ == "__main__":
    main()
