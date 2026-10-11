import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CorpusKind = Literal["bullet", "skill", "education", "summary"]
CorpusSource = Literal["cv"]

# Section-heading keywords → the kind of items the section yields. Includes the
# German headings used by the non-English fixture CV. Anything unmatched is
# treated as experience (bullets) — the safest default for tailoring.
_SUMMARY_HEADINGS = {"summary", "profile", "about", "objective", "profile"}
_SKILL_HEADINGS = {"skills", "kenntnisse", "technologies", "tools", "competencies", "skills & tools"}
_EDUCATION_HEADINGS = {"education","educational background","academic background","academic qualifications","education and qualifications"}
# A heading containing any of these words is a skills section even when it is
# not an exact match ("Technical Skills", "Tech Stack", "Skills & Tools").
_SKILL_HEADING_WORDS = {"skills", "skill", "kenntnisse", "technologies", "tools", "competencies", "stack"}

_BULLET_GLYPHS = ("-", "•", "*", "–", "◦")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";")
_PHONEISH = re.compile(r"\+?\d[\d\s().\-/]{6,}")
_PIPES = re.compile(r"\s*\|\s*")


class CorpusItem(BaseModel):
    """One verifiable unit of the candidate's real experience."""

    id: str
    text: str
    kind: CorpusKind
    source: CorpusSource
    section: str  # the heading (CV) 


class CandidateCorpus(BaseModel):
    """All corpus items, addressable by id for grounding checks."""

    items: list[CorpusItem] = Field(default_factory=list)

    def get(self, item_id: str) -> CorpusItem | None:
        """Look up one item by its id, or ``None``."""
        return next((item for item in self.items if item.id == item_id), None)

    def skills(self) -> list[str]:
        """Texts of every skill item (the allowed skill vocabulary)."""
        return [item.text for item in self.items if item.kind == "skill"]

    def render_for_prompt(self) -> str:
        """Render as ``[id] text`` lines the tailor LLM selects from."""
        return "\n".join(f"[{item.id}] {item.text}" for item in self.items)


def build_corpus(cv_text: str) -> CandidateCorpus:
    """Build the corpus from CV text. """
    items = _segment_cv(cv_text)
    return CandidateCorpus(items=items)


def _looks_like_heading(line: str) -> bool:
    """A short standalone word-or-two line reads as a section heading."""
    words = line.split()
    return 0 < len(words) <= 3 and not line.startswith(_BULLET_GLYPHS) and "," not in line and not line.rstrip().endswith(".")


def _section_kind(heading: str) -> CorpusKind:
    """Map a section heading to the kind of items it yields."""
    lowered = heading.lower().strip(" :")
    if lowered in _SUMMARY_HEADINGS:
        return "summary"
    if lowered in _EDUCATION_HEADINGS:
        return "education"
    if lowered in _SKILL_HEADINGS or any(word.strip("&/") in _SKILL_HEADING_WORDS for word in lowered.split()):
        return "skill"
    return "bullet"


def _logical_lines(cv_text: str) -> list[str]:
    """Re-join PDF-wrapped lines into logical ones.

    A line continues the previous one when that previous line ends mid-sentence
    (no terminal punctuation) and the new line reads as a continuation: it does
    not open a bullet, does not look like a heading, and either the previous
    line ends with a comma or the new line starts lowercase. Blank lines are
    hard barriers, so paragraphs never merge across them.
    """
    lines: list[str] = []
    for raw in cv_text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        prev = lines[-1] if lines else ""
        if (
            prev
            and not prev.endswith(_TERMINAL_PUNCT)
            and not line.startswith(_BULLET_GLYPHS)
            and not _looks_like_heading(line)
            and (prev.endswith(",") or line[:1].islower())
        ):
            lines[-1] = f"{prev} {line}"
        else:
            lines.append(line)
    return [line for line in lines if line]


def _segment_cv(cv_text: str) -> list[CorpusItem]:
    """Split raw CV text into corpus items using line/keyword heuristics."""
    items: list[CorpusItem] = []
    counters: dict[str, int] = {}
    section = ""
    kind: CorpusKind = "summary"  # leading paragraph before any heading

    def add(text: str, item_kind: CorpusKind, item_section: str) -> None:
        text = text.strip()
        if len(text) < 3:
            return
        counters[item_kind] = counters.get(item_kind, 0) + 1
        items.append(
            CorpusItem(
                id=f"cv-{item_kind}-{counters[item_kind]:03d}",
                text=text,
                kind=item_kind,
                source="cv",
                section=item_section or "(top)",
            )
        )

    for line in _logical_lines(cv_text):
        # Contact/header noise: emails, and pipe rows that read as contact
        # details (phone numbers, profile links). Pipe rows that carry real
        # content survive with the pipes turned into commas.
        if "@" in line or ("|" in line and (_PHONEISH.search(line) or "linkedin.com" in line.lower())):
            continue
        line = _PIPES.sub(", ", line)
        if _looks_like_heading(line):
            section = line
            kind = _section_kind(line)
            continue
        if kind == "skill":
            for skill in line.rstrip(".").split(","):
                add(skill, "skill", section)
        elif kind in ("summary", "education"):
            add(line, kind, section)
        else:
            add(line.lstrip("".join(_BULLET_GLYPHS)).strip(), "bullet", section)
    return items



