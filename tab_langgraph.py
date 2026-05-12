# =========================================
# TAB 4: LANGGRAPH + LANGSMITH
# =========================================
#
# LangGraph enables building stateful, multi-step agentic workflows with cycles.
# LangSmith provides monitoring, debugging, and tracing for LLM applications.
# This module demonstrates a multi-agent pipeline using LangGraph nodes and edges.

import json
import os
import time as _time
from typing import Any, Dict, List, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, START, END
from langsmith import traceable


# ========== State Definition ==========
class WorkflowState(TypedDict):
    """State maintained across LangGraph workflow execution."""
    user_input: str
    selected_metrics: list
    explanations: str
    report: dict
    step_tracker: list  # Track each step's token usage


# ========== Utility Functions ==========

def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _invoke_and_track(llm: BaseLanguageModel, messages: list, step_name: str) -> tuple[str, dict]:
    """
    Invoke the LLM with a list of messages and return (content, usage_dict).
    usage_dict keys: step, input_tokens, output_tokens, total_tokens
    """
    response = llm.invoke(messages)
    usage = getattr(response, "usage_metadata", None) or {}
    return response.content, {
        "step": step_name,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ========== LangGraph Nodes ==========

@traceable(name="selector_node")
def selector_node(state: WorkflowState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Node 1: Select the 5 most relevant metrics based on user input.
    Decorated with @traceable for LangSmith monitoring.
    """
    selector_messages = [
        SystemMessage(content=(
            "You are an expert in Machine Learning and Deep Learning evaluation. "
            "Return ONLY a valid JSON array — no markdown, no extra text."
        )),
        HumanMessage(content=(
            f"User request:\n{state['user_input']}\n\n"
            "Select the 5 most important evaluation metrics for this request. "
            'Return a JSON array where each element has: '
            '{"name": "...", "why_best": "...", "when_to_use": "..."}'
        )),
    ]
    metrics_raw, usage = _invoke_and_track(llm, selector_messages, "Selector")
    
    try:
        metrics = json.loads(_strip_fences(metrics_raw))
    except (json.JSONDecodeError, IndexError):
        metrics = [{"name": "Parse error", "why_best": metrics_raw[:200], "when_to_use": "N/A"}]
    
    return {
        "selected_metrics": metrics,
        "step_tracker": state.get("step_tracker", []) + [usage],
    }


@traceable(name="explainer_node")
def explainer_node(state: WorkflowState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Node 2: Provide detailed explanations for each selected metric.
    """
    explainer_messages = [
        SystemMessage(content=(
            "You are an expert technical educator. "
            "Provide clear, practical explanations for ML/DL evaluation metrics."
        )),
        HumanMessage(content=(
            f"Explain each of the following metrics in detail:\n"
            f"{json.dumps(state['selected_metrics'], indent=2)}\n\n"
            "For each metric include:\n"
            "1. A clear explanation\n"
            "2. A concrete practical example\n"
            "3. Key considerations / pitfalls"
        )),
    ]
    explanation, usage = _invoke_and_track(llm, explainer_messages, "Explainer")
    
    return {
        "explanations": explanation,
        "step_tracker": state.get("step_tracker", []) + [usage],
    }


@traceable(name="formatter_node")
def formatter_node(state: WorkflowState, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Node 3: Format results as a structured JSON report.
    """
    formatter_messages = [
        SystemMessage(content=(
            "You are a technical report writer. "
            "Return ONLY valid JSON — no markdown, no extra text."
        )),
        HumanMessage(content=(
            "Create a structured report from the following information.\n\n"
            f"User request: {state['user_input']}\n\n"
            f"Metrics selected:\n{json.dumps(state['selected_metrics'], indent=2)}\n\n"
            f"Detailed explanation:\n{state['explanations']}\n\n"
            'Output JSON with this exact structure:\n'
            '{\n'
            '  "title": "...",\n'
            '  "sections": [\n'
            '    {"heading": "...", "content": "..."},\n'
            '    ...\n'
            '  ]\n'
            '}'
        )),
    ]
    report_raw, usage = _invoke_and_track(llm, formatter_messages, "Formatter")
    
    try:
        report = json.loads(_strip_fences(report_raw))
    except (json.JSONDecodeError, IndexError):
        report = {
            "title": "LangGraph Report",
            "sections": [
                {"heading": "Metrics", "content": json.dumps(state['selected_metrics'], indent=2)},
                {"heading": "Explanation", "content": state['explanations']},
            ],
        }
    
    return {
        "report": report,
        "step_tracker": state.get("step_tracker", []) + [usage],
    }


# ========== Workflow Builder ==========

def build_langgraph_workflow(llm: BaseLanguageModel) -> StateGraph:
    """
    Build the LangGraph workflow graph: Selector → Explainer → Formatter
    """
    graph = StateGraph(WorkflowState)
    
    # Create node closures that capture the llm instance
    def selector_wrapper(state: WorkflowState) -> Dict[str, Any]:
        return selector_node(state, llm)
    
    def explainer_wrapper(state: WorkflowState) -> Dict[str, Any]:
        return explainer_node(state, llm)
    
    def formatter_wrapper(state: WorkflowState) -> Dict[str, Any]:
        return formatter_node(state, llm)
    
    # Add nodes
    graph.add_node("selector", selector_wrapper)
    graph.add_node("explainer", explainer_wrapper)
    graph.add_node("formatter", formatter_wrapper)
    
    # Add edges (pipeline flow)
    graph.add_edge(START, "selector")
    graph.add_edge("selector", "explainer")
    graph.add_edge("explainer", "formatter")
    graph.add_edge("formatter", END)
    
    return graph.compile()


# ========== Main Execution Function ==========

@traceable(name="run_langgraph_workflow")
def run_langgraph_workflow(user_input: str, llm) -> tuple[dict, list, str, dict]:
    """
    Run a complete LangGraph workflow with multi-agent pipeline.
    
    Pipeline:
      1. Selector agent  – picks the 5 most relevant ML/DL evaluation metrics.
      2. Explainer agent – expands each metric with explanation, example, and notes.
      3. Formatter agent – structures the output as a JSON report.
    
    LangSmith Integration:
      - Workflow is decorated with @traceable for automatic tracing.
      - Each node is traced individually for granular debugging.
    
    Args:
        user_input: User's task or query description.
        llm: LLM instance to use (e.g., ChatOpenAI, ChatAnthropic, etc.)
    
    Returns:
        (report, metrics, explanation, token_summary)
        token_summary mirrors the LM Studio tab format:
          { "per_call": [...], "totals": {...}, "performance": {...} }
    """
    pipeline_start = _time.time()
    
    # Build and execute the workflow
    workflow = build_langgraph_workflow(llm)
    
    initial_state: WorkflowState = {
        "user_input": user_input,
        "selected_metrics": [],
        "explanations": "",
        "report": {},
        "step_tracker": [],
    }
    
    # Execute the compiled workflow
    final_state = workflow.invoke(initial_state)
    
    # Extract results from final state
    report = final_state.get("report", {})
    metrics = final_state.get("selected_metrics", [])
    explanation = final_state.get("explanations", "")
    per_call = final_state.get("step_tracker", [])
    
    # ========== Build Token Summary ==========
    elapsed = _time.time() - pipeline_start
    total_prompt = sum(c.get("input_tokens", 0) for c in per_call)
    total_completion = sum(c.get("output_tokens", 0) for c in per_call)
    total_tokens = sum(c.get("total_tokens", 0) for c in per_call)
    
    token_summary = {
        "per_call": per_call,
        "totals": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        },
        "performance": {
            "elapsed_seconds": round(elapsed, 2),
            "completion_tokens_per_sec": round(total_completion / elapsed, 2) if elapsed > 0 and total_completion > 0 else 0.0,
            "total_tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 and total_tokens > 0 else 0.0,
        },
    }
    
    return report, metrics, explanation, token_summary
