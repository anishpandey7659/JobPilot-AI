from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from job_agent.graph.state import AgentState
from job_agent.graph.nodes import tailor, reformulate_query, fetch_jobs, rank_jobs,validate_tailoring


GOOD_FIT_THRESHOLD = 60
MIN_GOOD_JOBS = 5
MAX_REFORMULATIONS = 2

# Our Pydantic state models, explicitly allow-listed for checkpoint
# (de)serialization — LangGraph warns on (and will eventually block)
# unregistered types coming back out of a checkpoint.
_CHECKPOINT_TYPES = [
    ("job_agent.graph.schemas", name)
    for name in (
        "Profile",
        "JobPosting",
        "RankedJob",
        "TailoredBullet",
        "ExperienceEntry",
        "CVContent",
        "TailoringPack",
        "FlaggedClaim",
        "FabricationReport",
    )
]


def _checkpointer() -> MemorySaver:
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES))


def route_entry(state: AgentState) -> str:
    """According to the user query and the current state, decide which node to run next."""
    return "tailor" if state.get("selected_job_id") else "fetch_jobs"

def should_reformulate(state: AgentState) -> str:
    """Route after ranking: loop to broaden the search, or finish.

    Loops only if there are fewer than ``MIN_GOOD_JOBS`` jobs scoring at least
    ``GOOD_FIT_THRESHOLD`` and we are under the reformulation cap. Reaching the
    cap with thin results is expected, not an error. The cap defaults to
    ``MAX_REFORMULATIONS`` but SCOUT_MAX_REFORMULATIONS can lower it (0 = one
    pass, no loops — demo/latency mode); the conditional edge itself is the
    lesson and stays.
    """
    from job_agent.config import get_settings

    cap = min(MAX_REFORMULATIONS, get_settings().scout_max_reformulations)
    ranked = state.get("ranked_jobs", [])
    good = sum(1 for r in ranked if r.fit_score >= GOOD_FIT_THRESHOLD)
    if good < MIN_GOOD_JOBS and state.get("reformulation_count", 0) < cap:
        return "reformulate_query"
    return END



def build_graph(checkpointer: MemorySaver | None = None):
    """Build and compile the job-finding graph (starts from the profile input)."""
    builder = StateGraph(AgentState)
    builder.add_node("fetch_jobs", fetch_jobs)
    builder.add_node("rank_jobs", rank_jobs)
    builder.add_node("reformulate_query", reformulate_query)
    builder.add_node("tailor", tailor)
    builder.add_node("validate_tailoring", validate_tailoring)

    builder.add_conditional_edges(START, route_entry, ["fetch_jobs", "tailor"])
    builder.add_edge("fetch_jobs", "rank_jobs")
    builder.add_conditional_edges("rank_jobs", should_reformulate, ["reformulate_query", END])
    builder.add_edge("reformulate_query", "fetch_jobs")
    builder.add_edge("tailor", "validate_tailoring")
    builder.add_edge("validate_tailoring", END)

    return builder.compile(checkpointer=checkpointer or _checkpointer())


@lru_cache(maxsize=1)
def get_compiled_graph():
    """One compiled graph + one ``MemorySaver`` for the process lifetime.

    ``MemorySaver`` is in-process, so the sharing scope is one process: the
    Gradio app, a batch script, or a notebook each get exactly one instance,
    which is also the widest scope ``MemorySaver`` can honor. Tests that need
    isolation keep calling ``build_graph()`` directly.
    """
    return build_graph()
