from job_agent.tools.jobs_api import run_search, search_jobs
from job_agent.graph.state import AgentState
from job_agent.llm import ensure_budget, get_chat_model
from job_agent.config import get_settings
from job_agent.graph.schemas import JobPosting
from langchain_core.messages import HumanMessage, SystemMessage


# The merged ceiling bounds growth across reformulation loops, so a broadened
# search can still ADD jobs beyond the per-fetch limit without ballooning.
MERGED_CEILING = 25

MAX_QUERY_WORDS = 6
_SYSTEM = (
    "You are a job search assistant. Call the search_jobs tool exactly once.\n"
    "The query goes verbatim to job boards, which match it against job TITLES. "
    "So write it as a job title someone would actually post: the candidate's "
    "current role at the right seniority, and NOTHING else. Two to four words.\n"
    "Never append skills, technologies, tools or synonyms. Every extra term "
    "narrows the match, and a long query returns nothing at all.\n"
    "Good: 'senior data scientist' · 'machine learning engineer' · 'staff backend engineer'\n"
    "Bad:  'senior data scientist AI engineer deep learning LLMs RAG vector databases'\n"
    "Pick a country code from their location and set the remote flag from their preference."
)


def _build_prompt(state: AgentState) -> str:
    """Describe the candidate to the LLM, adding reformulation guidance if looping."""
    profile = state["profile"]
    lines = [
        f"Seniority: {profile.seniority}",
        f"Recent / primary roles: {', '.join(profile.primary_roles) or 'unknown'}",
        f"Key skills: {', '.join(profile.skills[:15])}",
        f"Summary: {profile.raw_summary or 'n/a'}",
        f"Locations: {', '.join(profile.locations) or 'unknown'}",
        f"Open to remote: {profile.remote_ok}",
    ]
    reformulated = state.get("search_query")
    if state.get("reformulation_count", 0) and reformulated:
        lines.append(
            f"\nThe previous search returned too few good matches. Use this broader query and search again: {reformulated!r}"
        )
    return "\n".join(lines)


def fetch_jobs(state: AgentState) -> dict:
    """Run the job search with LLM-chosen arguments and merge results into state."""
    settings = get_settings()
    calls = state.get("llm_calls", 0)
    ensure_budget(calls, 1, settings.max_llm_calls_per_run)
    profile = state["profile"]
    errors = list(state.get("errors", []))


    model_name = settings.scout_fetch_model or settings.scout_model
    model = get_chat_model(model_name, temperature=0.0).bind_tools([search_jobs])
    message = model.invoke([SystemMessage(_SYSTEM), HumanMessage(_build_prompt(state))])
    calls += 1

    location = profile.locations[0] if profile.locations else None
    if message.tool_calls:
        args = message.tool_calls[0]["args"]
        query = args.get("query") or " ".join(profile.primary_roles[:2])
        query, dropped = _trim_query(query)
        if dropped:
            errors.append(f"fetch_jobs: query trimmed to {MAX_QUERY_WORDS} words, dropped {dropped!r}")
        country = args.get("country")
        remote = bool(args.get("remote", profile.remote_ok))
    else:
        errors.append("fetch_jobs: LLM issued no tool call; used profile-derived query")
        query = " ".join(profile.primary_roles[:2]) or " ".join(profile.skills[:3])
        country = None
        remote = profile.remote_ok

    jobs, sources = run_search(query=query, location=location, country=country, remote=remote, limit=settings.scout_max_jobs)
    jobs = [j.model_copy(update={"title": clean_text(j.title),"company": clean_text(j.company),
            "location": clean_text(j.location),
            "description": clean_text(j.description),})
            for j in jobs]
    jobs = _dedupe_with_existing(state.get("jobs", []), jobs)[:MERGED_CEILING]

    return {
        "jobs": jobs,
        "search_query": query,
        "jobs_sources": sources,
        "errors": errors,
        "llm_calls": calls,
    }

def clean_text(s):
    if not isinstance(s, str):
        return s
    return s.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")


def _trim_query(query: str) -> tuple[str, str]:
    """Cut a query back to a title-length phrase.

    Returns the kept phrase and whatever was dropped (empty when nothing was).
    Keeping the *first* words is deliberate: both models we tested lead with the
    role title and then trail off into skills, so the front of the string is the
    part worth searching for.
    """
    words = query.split()
    if len(words) <= MAX_QUERY_WORDS:
        return query.strip(), ""
    return " ".join(words[:MAX_QUERY_WORDS]), " ".join(words[MAX_QUERY_WORDS:])


def _dedupe_with_existing(existing: list[JobPosting], new: list[JobPosting]) -> list[JobPosting]:
    """On a reformulation loop, merge new results with prior ones, deduped."""
    seen = {(j.title.strip().lower(), j.company.strip().lower()) for j in existing}
    merged = list(existing)
    for job in new:
        key = (job.title.strip().lower(), job.company.strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(job)
    return merged
