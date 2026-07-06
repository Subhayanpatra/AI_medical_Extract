"""Normalize a user's disease input before building a PubMed query."""

import difflib

from .gemini_client import generate_json


PROMPT = """You are a biomedical Disease Normalization Agent.

Your responsibilities are:
- Detect and expand abbreviations.
- Correct spelling mistakes.
- Identify the disease mentioned by the user.
- Normalize it to the official disease name used in biomedical literature.
- If the input is already an official disease name, keep it unchanged.
- If the input is not a disease, return INVALID_DISEASE as official_disease_name.

Return only a JSON object with exactly these fields:
{{
  "input": {user_input_json},
  "official_disease_name": "",
  "is_valid_disease": true,
  "confidence": 0.0
}}

Rules:
- official_disease_name must contain exactly one disease name or INVALID_DISEASE.
- is_valid_disease must be false when official_disease_name is INVALID_DISEASE.
- confidence must be a number from 0.0 to 1.0.
- Do not add explanations, markdown, or additional fields.
"""


DISEASE_ALIASES = {
    "alzheimers": "Alzheimer disease",
    "copd": "Chronic obstructive pulmonary disease",
    "covid": "COVID-19",
    "covid19": "COVID-19",
    "diabetes": "Diabetes mellitus",
    "diabetes mellitus": "Diabetes mellitus",
    "dm": "Diabetes mellitus",
    "hbp": "Hypertension",
    "high blood pressure": "Hypertension",
    "htn": "Hypertension",
    "t1d": "Type 1 diabetes mellitus",
    "t1dm": "Type 1 diabetes mellitus",
    "type 1 diabetes": "Type 1 diabetes mellitus",
    "type 1 diabetes mellitus": "Type 1 diabetes mellitus",
    "t2d": "Type 2 diabetes mellitus",
    "t2dm": "Type 2 diabetes mellitus",
    "type 2 diabetes": "Type 2 diabetes mellitus",
    "type 2 diabetes mellitus": "Type 2 diabetes mellitus",
}


def normalize_disease(user_input: str) -> dict:
    """Return a validated disease-normalization result from Gemini."""
    cleaned_input = " ".join(str(user_input).split())
    if not cleaned_input:
        return _invalid_result("")

    # repr produces a quoted, escaped value and prevents input from changing the prompt structure.
    try:
        raw = generate_json(
            PROMPT.format(user_input_json=repr(cleaned_input)),
            model_name="gemini-2.5-pro",
        )
    except Exception as exc:
        fallback = _local_normalize(cleaned_input)
        fallback["normalization_method"] = "Local fallback"
        fallback["normalization_warning"] = str(exc)
        return fallback
    if not isinstance(raw, dict):
        raise ValueError("Disease normalization agent returned a non-object response.")

    official_name = str(raw.get("official_disease_name", "")).strip()
    is_valid = raw.get("is_valid_disease") is True
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Disease normalization agent returned an invalid confidence.") from exc

    confidence = max(0.0, min(1.0, confidence))
    if not is_valid or not official_name or official_name.upper() == "INVALID_DISEASE":
        return _invalid_result(cleaned_input, confidence)

    return {
        "input": cleaned_input,
        "official_disease_name": official_name,
        "is_valid_disease": True,
        "confidence": confidence,
        "normalization_method": "Gemini",
        "normalization_warning": "",
    }


def _invalid_result(user_input: str, confidence: float = 0.0) -> dict:
    return {
        "input": user_input,
        "official_disease_name": "INVALID_DISEASE",
        "is_valid_disease": False,
        "confidence": confidence,
    }


def _local_normalize(user_input: str) -> dict:
    normalized = " ".join(user_input.lower().replace("-", " ").split())
    if normalized in DISEASE_ALIASES:
        official_name = DISEASE_ALIASES[normalized]
        confidence = 0.95
    else:
        match = difflib.get_close_matches(normalized, DISEASE_ALIASES, n=1, cutoff=0.72)
        if not match:
            result = _invalid_result(user_input)
            result["normalization_method"] = "Local fallback"
            result["normalization_warning"] = "Gemini normalization was unavailable."
            return result
        official_name = DISEASE_ALIASES[match[0]]
        confidence = round(difflib.SequenceMatcher(None, normalized, match[0]).ratio(), 2)

    return {
        "input": user_input,
        "official_disease_name": official_name,
        "is_valid_disease": True,
        "confidence": confidence,
    }
