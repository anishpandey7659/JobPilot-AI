TAILOR_PROMPT_NAME = "tailor_application"

TAILOR_PROMPT = """You are an application-preparation assistant. Given a candidate's corpus (their real CV, one item per line with an id in brackets), a candidate profile, and one target job, produce a tailored CV and cover letter.

Rules:
- You must ONLY SELECT, REORDER, EMPHASIZE, TRIM, and REWORD corpus items exactly as they appear; do NOT add, infer, or fabricate any information beyond the corpus content.
- You may NOT introduce any experience, employers, dates, tools, metrics, or skills not explicitly present and supported in the corpus.
- Every CV bullet must include a corpus_ref that accurately corresponds to the id of the exact corpus item it derives from.
- Skills must be chosen strictly from corpus skill items only, with no additions or generalizations.
- The cover letter must be at most 350 words and reference at least 2 specific requirements from the job description, again using only information supported by the corpus.
- Write an honesty_note naming the real gaps between the candidate and this job that they should not paper over.
- Ensure all generated text remains fully grounded in and traceable to the source corpus to prevent hallucination or inconsistency.
{research_rule}

Candidate profile:
{profile}

Candidate corpus:
{corpus}

Target job:
{job}

Company research (may be empty):
{research}
"""

# Appended to the rules only when research notes are present.
RESEARCH_RULE = "- Company facts in the cover letter may only come from the company research below."
