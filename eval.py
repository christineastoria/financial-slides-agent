"""
Offline evaluations for the Financial Slide Agent.

Three evaluator types:
1. Multi-modal — LLM judge evaluates rendered slide PNGs (via attachments)
2. Trajectory — validates tool call sequence and completeness
3. Assertions — LLM judge checks per-example goals are met

Runs experiments in a loop, swapping out the agent model each time.
Uses LangSmith attachments so slide images are viewable in the UI.

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

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_anthropic import ChatAnthropic
from langsmith import Client, evaluate, traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.schemas import Attachment

client = Client()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "Financial Slide Agent Evals"

EXAMPLES = [
    {
        "inputs": {
            "message": (
                "Create a Q4 2024 Business Review with revenue, gross profit, "
                "net income metrics and growth trends vs Q3"
            ),
        },
        "outputs": {
            "assertions": [
                "Slides should contain Q4 2024 revenue figures from the database",
                "Should show quarter-over-quarter growth trend vs Q3 2024",
                "Should include a Net Income metric",
                "Title slide should clearly reference Q4 2024 Business Review",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_financials",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Create a SaaS Investor Update for Dec 2024 showing MRR, ARR, "
                "churn rate, and unit economics (CAC, LTV)"
            ),
        },
        "outputs": {
            "assertions": [
                "Should display MRR for December 2024",
                "Should include ARR calculation",
                "Should show CAC and LTV metrics with LTV/CAC ratio",
                "Should include churn rate trend",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_financials",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Create an E-commerce Category Review showing all categories "
                "by revenue, profit margins, and YoY growth"
            ),
        },
        "outputs": {
            "assertions": [
                "Should show Electronics as the highest revenue category",
                "Should include profit margin percentages per category",
                "Should display Year-over-Year growth for each category",
                "Should cover at least 4 product categories",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "list_tables",
                "describe_table",
                "query_financials",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Create a Headcount & Burn Rate analysis showing team growth, "
                "revenue per employee, monthly burn, and runway"
            ),
        },
        "outputs": {
            "assertions": [
                "Should show headcount breakdown by department",
                "Should include revenue per employee metric",
                "Should display monthly net burn figures",
                "Should mention cash runway",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "list_tables",
                "describe_table",
                "describe_table",
                "query_financials",
                "query_financials",
                "generate_slides",
            ],
        },
    },
    {
        "inputs": {
            "message": (
                "Create a Regional Revenue Breakdown showing performance across "
                "North America, EMEA, and APAC with customer counts"
            ),
        },
        "outputs": {
            "assertions": [
                "Should break down revenue by region (NA, EMEA, APAC)",
                "Should include customer counts per region",
                "Should show growth percentages",
                "North America should show the highest revenue",
            ],
            "expected_trajectory": [
                "write_todos",
                "load_skill",
                "list_tables",
                "describe_table",
                "query_financials",
                "query_financials",
                "query_financials",
                "generate_slides",
            ],
        },
    },
]

# Models to compare
MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-3-5-haiku-latest",
]


def ensure_dataset():
    """Create or update the evaluation dataset in LangSmith."""
    # Delete existing dataset entirely so outputs are fresh
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        client.delete_dataset(dataset_id=datasets[0].id)
        print(f"Deleted existing dataset '{DATASET_NAME}'")

    ds = client.create_dataset(
        DATASET_NAME,
        description="Financial slide agent eval examples with per-example assertions",
    )

    for e in EXAMPLES:
        client.create_example(
            inputs=e["inputs"],
            outputs=e["outputs"],
            dataset_id=ds.id,
        )

    print(f"Dataset '{DATASET_NAME}' created with {len(EXAMPLES)} examples (with reference outputs)")
    return ds


# ---------------------------------------------------------------------------
# Run function (parameterised by model)
# ---------------------------------------------------------------------------


# Module-level storage for PNGs between target and evaluators.
# Safe because max_concurrency=1 guarantees target→evaluators run sequentially.
_current_slide_pngs_b64: list[str] = []


def make_target(model: str):
    """Return a run function that invokes the agent with the given model."""

    async def _ainvoke(inputs: dict) -> dict:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        from db import list_tables, describe_table, query_financials
        from agent import generate_slides, SYSTEM_PROMPT
        import agent as agent_module

        # Reset global PNG state before each run
        agent_module._last_slide_pngs = []

        eval_agent = create_deep_agent(
            name="financial-slide-agent-eval",
            model=model,
            tools=[list_tables, describe_table, query_financials, generate_slides],
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

        # Extract final text response (last AI message that isn't a tool result)
        text_response = ""
        for msg in reversed(messages):
            if hasattr(msg, "name"):
                continue  # skip tool results
            if not hasattr(msg, "content"):
                continue
            content = msg.content
            # Handle string content
            if isinstance(content, str) and content.strip():
                text_response = content
                break
            # Handle list-of-blocks content (Claude format)
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

        # Stash PNGs for evaluators, remove from outputs to keep them clean
        _current_slide_pngs_b64 = result.pop("slide_pngs_base64", [])

        # Attach PNGs to the LangSmith run so they're viewable in the UI
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

# Shared judge model (vision-capable for slide evaluation)
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

    # Build multi-modal message with slide images for vision evaluation
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
    for b64 in pngs_b64[:4]:  # cap at 4 to control cost
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

    # 1. Tool set coverage: did we use all the expected tool types?
    expected_set = set(expected)
    actual_set = set(actual)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    coverage = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0

    # 2. Order score: longest common subsequence / max length
    #    Measures whether tools appeared in the right relative order
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

    # 3. Length penalty: penalise trajectories that are way longer or shorter
    len_ratio = min(len(actual), len(expected)) / max(len(actual), len(expected)) if (expected or actual) else 1.0

    score = (coverage + order_score + len_ratio) / 3

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

    # Build multi-modal message: text prompt + slide images
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
# Main — run experiments across models
# ---------------------------------------------------------------------------


def main():
    ensure_dataset()

    for model in MODELS:
        short = model.split("-")[1]  # "sonnet" or "haiku"
        prefix = f"slide-agent-{short}"

        print(f"\n{'=' * 60}")
        print(f"Experiment: {prefix}  (model={model})")
        print(f"{'=' * 60}\n")

        evaluate(
            make_target(model),
            data=DATASET_NAME,
            evaluators=[
                slide_quality_evaluator,
                trajectory_evaluator,
                assertion_evaluator,
            ],
            experiment_prefix=prefix,
            metadata={"model": model},
            max_concurrency=1,  # sequential — avoids global PNG state conflicts
        )

    print("\nAll experiments complete. View results in LangSmith.")


if __name__ == "__main__":
    main()
