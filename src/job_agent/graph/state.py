from pydantic import BaseModel, Field
from typing import TypedDict
from job_agent.graph.schemas import JobPosting, Profile,RankedJob, TailoringPack, FabricationReport


class AgentState(TypedDict, total=False):
    """The state of the job agent, including the candidate profile, job postings, and ranked jobs."""


    cv_text: str
    profile: Profile | None
    search_query: str | None
    jobs: list[JobPosting]
    ranked_jobs: list[RankedJob]
    reformulation_count: int
    llm_calls: int
    errors: list[str]
    jobs_sources: list[str]
    tailoring: TailoringPack | None
    selected_job_id: str | None
    linkedin_zip_path: str | None
    research_notes: str | None
    fabrication_flags: int
    fabrication_report: FabricationReport | None

