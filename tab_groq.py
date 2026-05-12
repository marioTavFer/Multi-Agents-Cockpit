# ===================================
# TAB2 Groq + TAVILY DIRECT (NO CREW)
# ===================================

import json
import time
import litellm
from tavily import TavilyClient


def _completion_fail_fast(model, messages, api_key, temperature):
    return litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        timeout=20,
        max_retries=0,
    )


def run_groq_direct(user_input, groq_model, groq_api_key, tavily_api_key):
    """Run Groq + Tavily directly without CrewAI agents."""
    
    # Initialize clients
    tavily_client = TavilyClient(api_key=tavily_api_key)
    
    # Step 1: Search the web for evaluation metrics
    search_query = f"Select the 5 best topics for {user_input}"
    search_results = tavily_client.search(query=search_query, max_results=5)
    search_context = "\n".join([f"- {result['title']}: {result['content']}" for result in search_results.get('results', [])])
    
    # Step 2: Groq selects key metrics
    selector_prompt = f"""
You are an expert in AI.

{user_input}

Based on this request and the following web search results:
{search_context}

Select the 5 most important topics. Return a JSON array with each topic having:
{{"name": "...", "why_best": "...", "when_to_use": "..."}}

Return ONLY the JSON array, no other text.
"""
    
    kickoff_start = time.time()
    _t0 = time.time()
    metrics_response = _completion_fail_fast(
        model=groq_model,
        messages=[{"role": "user", "content": selector_prompt}],
        api_key=groq_api_key,
        temperature=0.7,
    )
    _t1 = time.time()
    _token_log = [{
        "agent": "Selector",
        "iteration": round(_t1 - _t0, 2),
    }]
    
    # Extract and parse metrics
    try:
        metrics_text = metrics_response['choices'][0]['message']['content'].strip()
        # Remove markdown code blocks if present
        if metrics_text.startswith('```'):
            metrics_text = metrics_text.split('```')[1]
            if metrics_text.startswith('json'):
                metrics_text = metrics_text[4:]
        metrics = json.loads(metrics_text)
    except json.JSONDecodeError:
        metrics = [{"name": "Unable to parse", "why_best": "JSON parse error", "when_to_use": "N/A"}]
    
    # Step 3: Groq validates and explains metrics
    explainer_prompt = f"""
You are an expert technical educator. Review these evaluation metrics:
{json.dumps(metrics, indent=2)}

For each metric, provide:
1. A clear explanation
2. A practical example
3. Key considerations

Format as a detailed, well-structured explanation.
"""
    
    _t0 = time.time()
    explanation_response = _completion_fail_fast(
        model=groq_model,
        messages=[{"role": "user", "content": explainer_prompt}],
        api_key=groq_api_key,
        temperature=0.7,
    )
    _t1 = time.time()
    _token_log.append({"agent": "Explainer", "iteration": round(_t1 - _t0, 2)})

    explanation = explanation_response['choices'][0]['message']['content'].strip()
    
    # Step 4: Groq generates the final report
    report_prompt = f"""
Create a structured technical report as JSON about evaluation metrics.

User request: {user_input}

Key metrics identified:
{json.dumps(metrics, indent=2)}

Detailed explanation:
{explanation}

Return a JSON object with this exact structure:
{{
  "title": "ML/DL Evaluation Metrics Report",
  "sections": [
    {{"heading": "Overview", "content": "..."}},
    {{"heading": "Selected Metrics", "content": "..."}},
    {{"heading": "Detailed Explanations", "content": "..."}},
    {{"heading": "Best Practices", "content": "..."}}
  ]
}}

Return ONLY the JSON, no other text.
"""
    
    _t0 = time.time()
    report_response = _completion_fail_fast(
        model=groq_model,
        messages=[{"role": "user", "content": report_prompt}],
        api_key=groq_api_key,
        temperature=0.5,
    )
    _t1 = time.time()
    _token_log.append({"agent": "Reporter", "iteration": round(_t1 - _t0, 2)})
    
    try:
        report_text = report_response['choices'][0]['message']['content'].strip()
        # Remove markdown code blocks if present
        if report_text.startswith('```'):
            report_text = report_text.split('```')[1]
            if report_text.startswith('json'):
                report_text = report_text[4:]
        report = json.loads(report_text)
    except json.JSONDecodeError:
        report = {
            "title": "Evaluation Metrics Report",
            "sections": [{
                "heading": "Summary",
                "content": explanation
            }]
        }
    
    kickoff_elapsed = time.time() - kickoff_start

    # Collect token usage from each LiteLLM response
    def _usage(resp):
        u = resp.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }

    usages = [_usage(metrics_response), _usage(explanation_response), _usage(report_response)]
    names = ["Selector", "Explainer", "Reporter"]

    per_call = [
        {"agent": names[i], "iteration": _token_log[i]["iteration"]}
        for i in range(len(names))
    ]

    total_prompt = sum(u["prompt_tokens"] for u in usages)
    total_completion = sum(u["completion_tokens"] for u in usages)
    total_tokens = sum(u["total_tokens"] for u in usages)

    completion_tps = round(total_completion / kickoff_elapsed, 2) if kickoff_elapsed > 0 and total_completion > 0 else 0.0
    total_tps = round(total_tokens / kickoff_elapsed, 2) if kickoff_elapsed > 0 and total_tokens > 0 else 0.0

    token_summary = {
        "per_call": per_call,
        "totals": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": 0,
            "successful_requests": len(names),
        },
        "performance": {
            "elapsed_seconds": round(kickoff_elapsed, 2),
            "completion_tokens_per_sec": completion_tps,
            "total_tokens_per_sec": total_tps,
        }
    }

    # Create a simple result object to match the interface
    class DirectResult:
        def __init__(self, data):
            self.pydantic = None
            self.json = data

    return DirectResult(report), token_summary
