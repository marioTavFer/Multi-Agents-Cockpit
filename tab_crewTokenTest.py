from crewai import Agent, Task, Crew
from crewai.hooks.llm_hooks import register_after_llm_call_hook, unregister_after_llm_call_hook, LLMCallHookContext
import os
import time

# Accumulator for per-call token data
_token_log: list[dict] = []


def _track_token_usage(context: LLMCallHookContext) -> None:
    """Called after every LLM call. Stores token usage from the crew result's task outputs."""
    agent_role = context.agent.role if context.agent else "unknown"
    # Per-call token data is not directly on LLMCallHookContext; log the call for counting
    _token_log.append({"agent": agent_role, "iteration": context.iterations})


def run_testToken(llm):
    global _token_log
    _token_log = []

    register_after_llm_call_hook(_track_token_usage)

    try:
        # 1. Define Agents
        researcher = Agent(
            role='Researcher',
            goal='Research AI advancements',
            backstory='An expert researcher',
            verbose=True,
            llm=llm
        )

        # 2. Define Tasks
        task1 = Task(
            description='Summarize the latest AI trends in a report.',
            expected_output='A 5-paragraph summary',
            agent=researcher
        )

        # 3. Create and run the Crew
        crew = Crew(
            agents=[researcher],
            tasks=[task1],
            verbose=True
        )

        kickoff_start = time.time()
        result = crew.kickoff()
        kickoff_elapsed = time.time() - kickoff_start

    finally:
        unregister_after_llm_call_hook(_track_token_usage)

    # result.token_usage is a UsageMetrics object populated by CrewAI
    usage = result.token_usage

    completion_tps = (
        round(usage.completion_tokens / kickoff_elapsed, 2)
        if kickoff_elapsed > 0 and usage.completion_tokens > 0
        else 0.0
    )
    total_tps = (
        round(usage.total_tokens / kickoff_elapsed, 2)
        if kickoff_elapsed > 0 and usage.total_tokens > 0
        else 0.0
    )

    token_summary = {
        "per_call": list(_token_log),
        "totals": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cached_prompt_tokens": usage.cached_prompt_tokens,
            "successful_requests": usage.successful_requests,
        },
        "performance": {
            "elapsed_seconds": round(kickoff_elapsed, 2),
            "completion_tokens_per_sec": completion_tps,
            "total_tokens_per_sec": total_tps,
        }
    }

    return result, token_summary
