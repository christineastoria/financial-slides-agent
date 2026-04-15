"""
FastAPI server for the Financial Slide Generator with Datadog APM + LangSmith tracing.

Run with: ddtrace-run uvicorn server:app --port 8001 --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Datadog APM + LLM Observability
try:
    from ddtrace import tracer, patch_all
    from ddtrace.llmobs import LLMObs
    patch_all(langchain=False)
    LLMObs.enable(
        ml_app=os.getenv("DD_LLMOBS_ML_APP", "financial-slide-agent-demo"),
        agentless_enabled=os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "0") == "1",
    )
    DATADOG_ENABLED = True
except ImportError:
    DATADOG_ENABLED = False
    print("Warning: ddtrace not installed. Datadog tracing disabled.")
    tracer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Financial Slide Generator - Datadog + LangSmith Demo")
    print("=" * 60)
    print(f"Datadog APM: {'Enabled' if DATADOG_ENABLED else 'Disabled'}")
    print(f"LangSmith Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"DD Service: {os.getenv('DD_SERVICE', 'not set')}")
    print("=" * 60)
    yield
    print("Shutting down...")


app = FastAPI(
    title="Financial Slide Generator",
    description="Generate financial slide decks with Datadog + LangSmith tracing",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    slide_pngs_base64: list[str] = []
    langsmith_run_id: Optional[str] = None
    langsmith_url: Optional[str] = None
    datadog_trace_id: Optional[str] = None
    datadog_trace_url: Optional[str] = None


def get_datadog_trace_context() -> dict:
    if not DATADOG_ENABLED or not tracer:
        return {"trace_id": None, "span_id": None, "trace_url": None}
    span = tracer.current_span()
    if not span:
        return {"trace_id": None, "span_id": None, "trace_url": None}
    trace_id = span.trace_id
    dd_site = os.getenv("DD_SITE", "us5.datadoghq.com")
    trace_url = f"https://app.{dd_site}/apm/traces?query=trace_id%3A{trace_id}"
    return {
        "trace_id": str(trace_id),
        "span_id": str(span.span_id),
        "trace_url": trace_url,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    static_index = os.path.join(static_path, "index.html")
    if os.path.exists(static_index):
        with open(static_index, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Financial Slide Generator</h1><p>Frontend not found.</p>")


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
                "thread_id": request.thread_id,
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
    # Derive the UI URL from the API endpoint so dev/prod stay in sync
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


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "datadog_enabled": DATADOG_ENABLED,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "default"),
    }


@app.get("/config")
async def get_config():
    return {
        "dd_rum_enabled": bool(os.getenv("DD_RUM_APPLICATION_ID")),
        "dd_rum_application_id": os.getenv("DD_RUM_APPLICATION_ID", ""),
        "dd_rum_client_token": os.getenv("DD_RUM_CLIENT_TOKEN", ""),
        "dd_site": os.getenv("DD_SITE", "datadoghq.com"),
        "dd_service": os.getenv("DD_SERVICE", "financial-slide-agent-demo"),
        "dd_env": os.getenv("DD_ENV", "development"),
    }
