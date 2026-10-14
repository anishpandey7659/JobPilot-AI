from langfuse import observe
from job_agent.llm import get_chat_model
from job_agent.config import get_settings
from job_agent.graph.schemas import Profile
from langchain_core.runnables import RunnableConfig
from job_agent.tracing import get_tracer,run_config

EXTRACT_PROFILE_PROMPT_NAME = "extract_profile"

EXTRACT_PROFILE_PROMPT = """You are a recruiting assistant. Read the CV text below and extract a structured candidate profile.

Fill in every field:
- name: the candidate's name, or null if not present.
- seniority: one of junior, mid, senior, lead, or unknown.
- primary_roles: the job titles/roles this person is a fit for, ordered with their current or most recent role first.
- skills: a list of their skills, lowercased.
- years_experience: total years of professional experience as a number, or null.
- locations: locations where they could work.
- languages:  spoken/written languages explicitly mentioned in the CV. Write English as default if none are mentioned.
- remote_ok: true if they are open to remote work.
- raw_summary: a 3-4 sentence summary, starting with their most recent experience.

CV text:
{cv_text}
"""


@observe(name="extract_profile")
def extract_profile(
    cv_text: str,
    *,
    thread_id: str,
    user_id: str = "Test_user",
    tags: list[str] | None = None,
    model: str | None = None,
    remote: bool | None = True,
) -> Profile:
    """Extract a structured candidate profile from CV text."""

    settings = get_settings()
    model_name = model or settings.scout_model

    llm = get_chat_model(model_name,temperature=0.0,).with_structured_output(Profile)

    tracer, usage_cb = get_tracer(thread_id=thread_id,user_id=user_id,tags=tags,)

    config: RunnableConfig = run_config(thread_id=thread_id,run_name="extract_profile",tracer=tracer,usage_cb=usage_cb,)

    config.setdefault("metadata", {}).update({
        "operation": "extract_profile",
        "langfuse_tags": ["extract-profile", *(tags or [])],
        "remote": remote,
        "model":model_name,
    })

    profile: Profile = llm.invoke(EXTRACT_PROFILE_PROMPT.format(cv_text=cv_text),config=config,)

    if remote is not None:
        profile.remote_ok = remote

    return profile
