from job_agent.config import get_settings
from job_agent.graph.prompts.reformulate import REFORMULATE_PROMPT
from job_agent.graph.state import AgentState
from job_agent.llm import ensure_budget, get_chat_model


def reformulate_query(state: AgentState) -> dict:
    """Ask the LLM for a broader search query and bump the loop counter."""
    settings = get_settings()
    calls = state.get("llm_calls", 0)
    ensure_budget(calls, 1, settings.max_llm_calls_per_run)
    profile = state["profile"]

    prompt = REFORMULATE_PROMPT.format(
        profile=", ".join(profile.primary_roles + profile.skills[:10]),
        previous_query=state.get("search_query") or "",
    )
    new_query = get_chat_model(settings.scout_model, temperature=0.0).invoke(prompt).content.strip()

    return {
        "search_query": new_query,
        "reformulation_count": state.get("reformulation_count", 0) + 1,
        "llm_calls": calls + 1,
    }
