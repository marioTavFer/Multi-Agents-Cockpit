# =========================================
# TAB - LM Studio Token Test
# =========================================
# Mirrors tab_crewTokenTest.py but uses LM Studio (LangChain + OpenAI-compatible API)
# instead of CrewAI. Runs a single task and reports token usage.

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def run_lmstudio_token_test(llm_studio):
    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes AI topics."), 
        HumanMessage(content="Summarize the latest AI trends in 5 paragraphs."),
    ]

    start = time.time()
    response = llm_studio.invoke(messages)
    elapsed = time.time() - start

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    per_call = [
        {
            "step": "Researcher",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    ]

    completion_tps = round(output_tokens / elapsed, 2) if elapsed > 0 and output_tokens > 0 else 0.0
    total_tps = round(total_tokens / elapsed, 2) if elapsed > 0 and total_tokens > 0 else 0.0

    token_summary = {
        "per_call": per_call,
        "totals": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": 0,
            "successful_requests": 1,
        },
        "performance": {
            "elapsed_seconds": round(elapsed, 2),
            "completion_tokens_per_sec": completion_tps,
            "total_tokens_per_sec": total_tps,
        },
    }

    return response.content, token_summary
