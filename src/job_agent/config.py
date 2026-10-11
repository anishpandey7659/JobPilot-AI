from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime configuration, read once from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(_REPO_ENV, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    scout_model: str = Field(default="openai:gpt-4o-mini", alias="SCOUT_MODEL")
    scout_tailor_model: str = Field(default="openai:gpt-4o-mini", alias="SCOUT_TAILOR_MODEL")

    fallback_model :str = Field(default="groq:openai/gpt-oss-120b", alias="FALLBACK_MODEL")
    
    openai_api_key :SecretStr =Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    groq_api_key :SecretStr = Field(default=SecretStr(""), alias="GROQ_API_KEY")

    jsearch_api_key: SecretStr = Field(default=SecretStr(""), alias="JSEARCH_API_KEY")
    adzuna_app_id: SecretStr = Field(default=SecretStr(""), alias="ADZUNA_APP_ID")
    adzuna_app_key: SecretStr = Field(default=SecretStr(""), alias="ADZUNA_APP_KEY")

    tavily_api_key: SecretStr = Field(default=SecretStr(""), alias="TAVILY_API_KEY")

    langfuse_secret_key: SecretStr = Field(default=SecretStr(""), alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: SecretStr = Field(default=SecretStr(""), alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    max_llm_calls_per_run: int = Field(default=25, alias="MAX_LLM_CALLS_PER_RUN")
    scout_max_jobs: int = Field(default=10, alias="SCOUT_MAX_JOBS")
    scout_max_reformulations: int = Field(default=2, alias="SCOUT_MAX_REFORMULATIONS")
    scout_fetch_model: str = Field(default="", alias="SCOUT_FETCH_MODEL")
    scout_rank_batch: int = Field(default=4, alias="SCOUT_RANK_BATCH")
    scout_concurrent_sources: bool = Field(default=True, alias="SCOUT_CONCURRENT_SOURCES")

    scout_source_soft_deadline: float = Field(
        default=1.0,
        alias="SCOUT_SOURCE_SOFT_DEADLINE",
        description="Seconds to wait for the first concurrent source before falling through to faster ones.",
    )


    fab_bullet_ratio: float = Field(default=0.65, alias="SCOUT_FAB_BULLET_RATIO")
    fab_skill_ratio: float = Field(default=0.85, alias="SCOUT_FAB_SKILL_RATIO")
    fab_letter_ratio: float = Field(default=0.55, alias="SCOUT_FAB_LETTER_RATIO")

    environment : str = "developement"


    @property
    def has_jsearch(self) -> bool:
        """Whether a JSearch API key is configured."""
        return bool(self.jsearch_api_key.get_secret_value())

    
    @property
    def has_adzuna(self) -> bool:
        """Whether both Adzuna credentials are configured."""
        return bool(self.adzuna_app_id.get_secret_value() and self.adzuna_app_key.get_secret_value())

    @property
    def has_tavily(self) -> bool:
        """Whether a Tavily API key is configured (optional company research)."""
        return bool(self.tavily_api_key.get_secret_value())

    @property
    def has_langfuse(self) -> bool:
        """Whether Langfuse is configured for LLM observability and analytics."""
        return bool(self.langfuse_secret_key.get_secret_value() and self.langfuse_public_key.get_secret_value())



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
