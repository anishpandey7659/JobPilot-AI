import os
from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from job_agent.config import get_settings

class LLMBudgetExceededError(RuntimeError):
    """Raised when a run would exceed ``MAX_LLM_CALLS_PER_RUN``."""



DEFAULT_FALLBACK_MODEL = "groq:llama-3.3-70b-versatile"


def _export_api_keys() -> None:
    """Export provider keys to env vars (each independently) so LangChain can find them."""
    settings = get_settings()

    openai_key = settings.openai_api_key.get_secret_value()
    groq_key = settings.groq_api_key.get_secret_value()

    if openai_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_key
    if groq_key and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = groq_key


@lru_cache(maxsize=8)
def get_chat_model(
    model: str = get_settings().scout_model,
    temperature: float = 0.0,
    fallback_model: str | None = get_settings().fallback_model,
):
    """Return a cached chat model (e.g. ``openai:gpt-4o-mini``) with an optional Groq fallback.

    If the primary call fails (rate limit, outage, timeout, ...), the fallback
    model is tried automatically.
    """
    _export_api_keys()

    primary = init_chat_model(model, temperature=temperature)

    # Only attach a fallback if one is requested, differs from the primary,
    # and its provider key is actually available.
    if fallback_model and fallback_model != model:
        needs_groq = fallback_model.startswith("groq:")
        if not needs_groq or os.environ.get("GROQ_API_KEY"):
            fallback = init_chat_model(fallback_model, temperature=temperature)
            return primary.with_fallbacks([fallback])

    return primary


def ensure_budget(current_calls: int, planned: int, max_calls: int) -> None:
    """Raise ``LLMBudgetExceededError`` if ``planned`` more calls would exceed ``max_calls``."""
    if current_calls + planned > max_calls:
        raise LLMBudgetExceededError(
            f"Run would make {current_calls + planned} LLM calls, exceeding MAX_LLM_CALLS_PER_RUN={max_calls}."
        )