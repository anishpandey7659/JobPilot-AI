# tracing.py
import os
import logging
from job_agent.config import get_settings
from dataclasses import dataclass, field
from langfuse import Langfuse,get_client
from typing import Any, Callable, TypeVar
from functools import lru_cache
from langfuse.langchain import CallbackHandler 
from langchain_core.callbacks import UsageMetadataCallbackHandler

logger = logging.getLogger(__name__)
T = TypeVar("T")

settings = get_settings()

def configure_langfuse(environment: str = "development", release: str | None = None) -> Langfuse:
    """Call once at startup, before any @observe function runs."""
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_base_url,
        environment=environment,
        release=release,
    )
    if not client.auth_check():
        raise ConnectionError("Langfuse auth failed: check keys and host")
    return client


@lru_cache(maxsize=1)
def _langfuse_client() -> Langfuse | None:
    """Run configure_langfuse once per process. None means tracing is off."""
    try:
        return configure_langfuse(
            environment=settings.environment,      # adjust to your settings
            release=getattr(settings, "release", None),
        )
    except Exception as exc:  # noqa: BLE001 - missing keys, bad auth, network: never fatal
        logger.warning("Langfuse disabled: %s", exc)
        return None

def traced_call(
    name: str,
    fn: Callable[[], T],
    metadata: dict[str, Any] | None = None,
) -> Callable[[], T]:
    """Wrap ``fn`` so each call runs inside its own Langfuse span."""
    def wrapped() -> T:
        with get_client().start_as_current_observation(
            name=name, as_type="span", metadata=metadata
        ) as span:
            try:
                result = fn()
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise
            span.update(output={"count": len(result) if hasattr(result, "__len__") else None})
            return result

    return wrapped


@dataclass
class Tracer:
    """Langfuse-only state. Token counting lives outside it."""
    thread_id: str
    handler: CallbackHandler                      # Langfuse CallbackHandler
    user_id: str | None = None
    tags: list[str] = field(default_factory=lambda: ["job-finder"])

    @property
    def metadata(self) -> dict:
        return {
            "langfuse_session_id": self.thread_id,
            "langfuse_user_id": self.user_id,
            "langfuse_tags": self.tags,
        }

    @property
    def trace_id(self) -> str | None:
        return getattr(self.handler, "last_trace_id", None)

    def flush(self) -> None:
        """Send any queued spans to Langfuse. Never raises."""
        client = _langfuse_client()
        if client is None:
            return
        try:
            client.flush()
        except Exception:  # noqa: BLE001 - flushing must never break a run
            pass

def get_tracer(
    thread_id: str,
    user_id: str,
    tags: list[str] | None = None,
) -> tuple[Tracer | None, UsageMetadataCallbackHandler]:
    """Fresh usage counter every run; Tracer is None if Langfuse is off."""
    usage_cb = UsageMetadataCallbackHandler()    # always created, fresh per run

    if _langfuse_client() is None:
        return None, usage_cb


    tracer = Tracer(
        thread_id=thread_id,
        handler=CallbackHandler(),
        user_id=user_id,
        tags=tags or ["job-finder"],
    )
    return tracer, usage_cb

from langchain_core.runnables import RunnableConfig

def run_config(
    thread_id: str,
    tracer: Tracer | None,
    usage_cb: UsageMetadataCallbackHandler,
) -> RunnableConfig:

    config: RunnableConfig = {
        "callbacks": [usage_cb],
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "thread_id": thread_id,
            "feature": "job-finder",
            "environment": "development",
        },
    }

    if tracer:
        config["callbacks"] = [tracer.handler, usage_cb]
        config["run_name"] = "job-finder"
        config["recursion_limit"] = 25
        config["metadata"].update(tracer.metadata)

    return config


def langfuse_url(tracer: Tracer | None) -> str:
    """Link to this run's trace in the Langfuse UI, or "" if tracing is off."""
    if tracer is None or not tracer.trace_id:
        return ""
    client = _langfuse_client()
    if client is None:
        return ""
    try:
        return client.get_trace_url(trace_id=tracer.trace_id) or ""
    except Exception:  # noqa: BLE001 - a missing link must never break a run
        return ""


from pathlib import Path
from langfuse.api import MediaContentType 
def attach_cv(tracer: Tracer | None, pdf_path: str | Path) -> None:
    """Attach the uploaded CV PDF to the run's trace (best effort, post-run)."""
    if tracer is None or not tracer.trace_id:
        return
    try:
        from langfuse.media import LangfuseMedia

        client = _langfuse_client()
        if client is None:
            return

        cv = LangfuseMedia(file_path=str(pdf_path), content_type=MediaContentType.APPLICATION_PDF)
        client.create_event(
            trace_context={"trace_id": tracer.trace_id},
            name="cv-attachment",
            metadata={"cv": cv, "file_name": Path(pdf_path).name},
        )
    except Exception:  # noqa: BLE001, S110 - attachment is nice-to-have, never fatal
        pass






"""
python
@observe()
def fetch_jobs(query: str):
    langfuse = get_client()
    langfuse.update_current_span(metadata={"query": query, "source": "linkedin"})
    return ...
-----------------------------------------------------------------------------
from langfuse import get_client
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()  # uses the client from configure_langfuse()
result = graph.invoke(
    {"messages": [("user", "find python jobs")]},
    config={"callbacks": [langfuse_handler]},
)
OR Add trace-level metadata (user, session, tags) via config metadata:
result = graph.invoke(
    {"messages": [...]},
    config={
        "callbacks": [langfuse_handler],
        "run_name": "job-agent",
        "metadata": {
            "langfuse_user_id": "user_123",
            "langfuse_session_id": "session_abc",
            "langfuse_tags": ["job-agent", "prod"],
            "agent_version": "1.2",   # custom metadata
        },
    },
)
    

"""
