# =============================
# TAB1 crewAI - CREW BUILDER
# =============================

from crewai import Agent, Task, Crew, LLM
from crewai.hooks.llm_hooks import register_after_llm_call_hook, unregister_after_llm_call_hook, LLMCallHookContext
from pydantic import BaseModel
import time

# Accumulator for per-call token data
_token_log: list[dict] = []


def _track_token_usage(context: LLMCallHookContext) -> None:
    """Called after every LLM call. Stores agent role and iteration count."""
    agent_role = context.agent.role if context.agent else "unknown"
    _token_log.append({"agent": agent_role, "iteration": context.iterations})


class Section(BaseModel):
    heading: str
    content: str


class ReportModel(BaseModel):
    title: str
    sections: list[Section]


def run_crew(user_input, llm):
    """Run CrewAI agents to generate evaluation metrics report."""
    global _token_log
    _token_log = []

    register_after_llm_call_hook(_track_token_usage)

    try:
        # Agents
        selector = Agent(
            role="AI Evaluation Specialist",
            goal="select five best topics of machine learning and deep learning algorithms",
            backstory="Expert in Machine learning and deep learningtesting.",
            llm=llm,
            verbose=True
        )
        explainer = Agent(
            role="Technical Educator",
            goal="Explain the selected topics clearly with examples and equations",
            backstory="Expert in simplifying Machine learning and deep learning concepts.",
            llm=llm,
            verbose=True
        )

        reporter = Agent(
            role="Technical Report Writer",
            goal="Generate structured report",
            backstory="Expert in technical documentation.",
            llm=llm,
            verbose=True
        )

        # Tasks
        task1 = Task(
            description=f"""
{user_input}

Select most used topics.
Return JSON with name, why_best, when_to_use.
""",
            expected_output="JSON with name, why_best, when_to_use.",
            agent=selector
        )

        task2 = Task(
            description=""" 
Explain the final methods clearly with examples.
Return JSON with explanations.
""",
            expected_output="JSON with explanations.",
            agent=explainer,
            context=[task1]
        )

        task3 = Task(
            description="""
Create a structured report.

Return JSON:
{
  "title": "",
  "sections": [
    {"heading": "", "content": ""}
  ]
}

Return ONLY JSON.
""",
            expected_output="JSON with title and sections.",
            output_pydantic=ReportModel,
            agent=reporter,
            context=[task1, task2]
        )

        crew = Crew(
            agents=[selector, explainer, reporter],
            tasks=[task1, task2, task3],
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
