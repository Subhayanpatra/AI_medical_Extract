from typing import Any

from .gemini_client import generate_json


RELEVANCE_PROMPT = """
You are a biomedical literature Full-Text Relevance Agent.

Your task is to determine whether a biomedical research paper is relevant to the user's clinical research query.

Evaluate relevance using the complete research meaning of the query.

Do not classify a paper as relevant only because it contains one isolated keyword.

Consider whether the paper meaningfully discusses the important concepts in the query, such as:

- disease or medical condition
- treatment, drug, intervention, or comparator
- population
- outcome or endpoint
- study design
- database or data source
- real-world evidence
- healthcare resource utilization
- costs or economic outcomes
- clinical setting

Rules:

1. Read the paper title and full text.
2. Compare the paper with the corrected user query.
3. A paper is relevant when its main topic, population, methods, treatment, outcome, or research objective meaningfully matches the query.
4. A paper is not relevant when the query concepts appear only incidentally, in references, background discussion, or unrelated sections.
5. Do not invent information.
6. If the full text is missing or too short, mark the paper as not relevant.
7. Return only valid JSON.
8. Do not use Markdown or code fences.

Return exactly:

{
  "Relevant": true,
  "Relevance_Score": 0.0,
  "Relevance_Reason": ""
}

Relevance_Score must be between 0.0 and 1.0.

Corrected user query:
__QUERY__

Paper title:
__TITLE__

Full text:
__FULL_TEXT__
"""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def relevance_agent(query: Any, title: Any, full_text: Any, max_characters: int = 60000) -> dict:
    query_text = _text(query)
    title_text = _text(title)
    full_text_value = _text(full_text)

    if not query_text:
        return {
            "Relevant": False,
            "Relevance_Score": 0.0,
            "Relevance_Reason": "The corrected query is missing.",
        }

    if not full_text_value or len(full_text_value) < 200:
        return {
            "Relevant": False,
            "Relevance_Score": 0.0,
            "Relevance_Reason": "Full text is missing or too short.",
        }

    prompt = (
        RELEVANCE_PROMPT
        .replace("__QUERY__", query_text)
        .replace("__TITLE__", title_text)
        .replace("__FULL_TEXT__", full_text_value[:max_characters])
    )

    try:
        data = generate_json(prompt)
    except Exception as exc:
        return {
            "Relevant": False,
            "Relevance_Score": 0.0,
            "Relevance_Reason": f"Agent request failed: {exc}",
        }

    if not isinstance(data, dict):
        data = {}

    relevant = data.get("Relevant", False)
    if isinstance(relevant, str):
        relevant = relevant.strip().lower() == "true"
    else:
        relevant = bool(relevant)

    try:
        score = float(data.get("Relevance_Score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    return {
        "Relevant": relevant,
        "Relevance_Score": round(max(0.0, min(score, 1.0)), 4),
        "Relevance_Reason": str(data.get("Relevance_Reason", "")).strip(),
    }
