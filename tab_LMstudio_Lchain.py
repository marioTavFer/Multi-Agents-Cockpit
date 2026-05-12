# =========================================
# TAB6 LangChain + LM Studio (Local Server)
# =========================================
#
# LM Studio exposes an OpenAI-compatible REST API at:
#   http://localhost:1234/v1
# Make sure LM Studio is running and a model is loaded before using this tab.

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _invoke_and_track(llm, messages: list, step_name: str) -> tuple[str, dict]:
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


def run_lmstudio_lchain(user_input: str, llm_studio):
    """
    Run a LangChain pipeline against LM Studio.

    Pipeline:
      1. Selector agent  – picks the 5 most relevant topics and explains why.
      2. Explainer agent – expands each topic with explanation, example, and notes.
      3. Formatter agent – structures the output as a JSON report.

    Returns:
        (report, topics, explanation, token_summary)
        token_summary mirrors the Tab 5 format:
          { "per_call": [...], "totals": {...}, "performance": {...} }
    """
    import time as _time

    llm = llm_studio
    per_call: list[dict] = []
    pipeline_start = _time.time()

    # ------------------------------------------------------------------
    # Step 1: Select topics
    # ------------------------------------------------------------------
    selector_messages = [
        SystemMessage(content=(
            "You are an expert in AI. "
            "Return ONLY a valid JSON array — no markdown, no extra text."
        )),
        HumanMessage(content=(
            f"User request:\n{user_input}\n\n"
            "Select the 5 most important topics for this request. "
            'Return a JSON array where each element has: '
            '{"name": "...", "why_best": "...", "when_to_use": "..."}'
        )),
    ]
    metrics_raw, usage1 = _invoke_and_track(llm, selector_messages, "Selector")
    per_call.append(usage1)

    try:
        metrics = json.loads(_strip_fences(metrics_raw))
    except (json.JSONDecodeError, IndexError):
        metrics = [{"name": "Parse error", "why_best": metrics_raw[:200], "when_to_use": "N/A"}]

    # ------------------------------------------------------------------
    # Step 2: Explain topics
    # ------------------------------------------------------------------
    explainer_messages = [
        SystemMessage(content=(
            "You are an expert technical educator. "
            "Provide clear, practical explanations for the topics."
        )),
        HumanMessage(content=(
            f"Explain each of the following topics in detail:\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "For each topic include:\n"
            "1. A clear explanation\n"
            "2. A concrete practical example\n"
            "3. Key considerations / pitfalls"
        )),
    ]
    explanation, usage2 = _invoke_and_track(llm, explainer_messages, "Explainer")
    per_call.append(usage2)

    # ------------------------------------------------------------------
    # Step 3: Format as structured report JSON
    # ------------------------------------------------------------------
    formatter_messages = [
        SystemMessage(content=(
            "You are a technical report writer. "
            "Return ONLY valid JSON — no markdown, no extra text."
        )),
        HumanMessage(content=(
            "Create a structured report from the following information.\n\n"
            f"User request: {user_input}\n\n"
            f"Metrics selected:\n{json.dumps(metrics, indent=2)}\n\n"
            f"Detailed explanation:\n{explanation}\n\n"
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
    report_raw, usage3 = _invoke_and_track(llm, formatter_messages, "Formatter")
    per_call.append(usage3)

    try:
        report = json.loads(_strip_fences(report_raw))
    except (json.JSONDecodeError, IndexError):
        report = {
            "title": "LM Studio Report",
            "sections": [
                {"heading": "Metrics", "content": json.dumps(metrics, indent=2)},
                {"heading": "Explanation", "content": explanation},
            ],
        }

    # ------------------------------------------------------------------
    # Build token summary (mirrors Tab 5 format)
    # ------------------------------------------------------------------
    elapsed = _time.time() - pipeline_start
    total_prompt = sum(c["input_tokens"] for c in per_call)
    total_completion = sum(c["output_tokens"] for c in per_call)
    total_tokens = sum(c["total_tokens"] for c in per_call)

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
