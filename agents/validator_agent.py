import re
from collections import Counter, defaultdict


DISEASE_KEYWORDS = [
    "diabetes",
    "type 1 diabetes",
    "type 2 diabetes",
    "diabetes mellitus",
    "prediabetes",
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "hypertension",
    "obesity",
    "depression",
    "anxiety",
    "asthma",
    "copd",
    "heart failure",
    "stroke",
    "kidney disease",
    "renal disease",
    "alzheimer",
    "parkinson",
    "arthritis",
    "covid-19",
    "tuberculosis",
    "hiv",
]


def validate_disease(record: dict, requested_disease: str) -> dict:
    """Classify primary and major secondary diseases for one paper."""
    evidence = _collect_evidence(record)
    requested = _normalize(requested_disease)
    scores = defaultdict(float)
    reasons = defaultdict(list)

    for source, text, weight in evidence:
        for disease in _find_diseases(text, requested_disease):
            scores[disease] += weight
            reasons[disease].append(source)

    if requested and requested in _normalize(" ".join(record.get("Title", "").split())):
        scores[requested_disease] += 5
        reasons[requested_disease].append("requested disease appears in title")

    if not scores:
        primary = requested_disease.strip()
        confidence = 0.2
        reason = "No clear disease evidence found; defaulted to requested disease for single primary disease requirement."
    else:
        primary = max(scores, key=scores.get)
        confidence = min(0.99, round(scores[primary] / 12, 2))
        reason = _build_reason(primary, reasons[primary], scores[primary])

    secondary = _major_secondary_diseases(primary, scores, reasons)
    accepted = _matches_requested(primary, requested_disease) or any(
        _matches_requested(item, requested_disease) for item in secondary
    )

    return {
        "Primary Disease": _display_name(primary),
        "Major Secondary Diseases": "; ".join(_display_name(item) for item in secondary),
        "Disease Confidence Score": confidence,
        "Disease Validation Reason": reason,
        "Accepted": "Yes" if accepted else "No",
        "Rejection Reason": "" if accepted else (
            "Requested disease is neither the primary disease nor a major secondary disease."
        ),
    }


def _collect_evidence(record: dict) -> list[tuple[str, str, float]]:
    return [
        ("Title", record.get("Title", ""), 5.0),
        ("MeSH Terms", record.get("MeSH Terms", ""), 4.0),
        ("Abstract", record.get("Abstract", ""), 3.0),
        ("Keywords", record.get("Keywords", ""), 2.0),
        ("Publication Types", record.get("Publication Types", ""), 0.5),
        ("Chemical List", record.get("Chemical List", ""), 0.5),
        ("Existing extracted metadata", _stringify_existing_metadata(record), 1.0),
        ("Existing NLP entities", record.get("NLP Entities", ""), 1.0),
        ("BioC title and abstract", record.get("BioC Title Abstract", ""), 2.0),
        ("LLM reasoning", record.get("LLM Disease Reasoning", ""), 1.0),
    ]


def _find_diseases(text: str, requested_disease: str) -> list[str]:
    normalized = _normalize(text)
    found = []

    requested = _normalize(requested_disease)
    if requested and requested in normalized:
        found.append(requested_disease.strip())

    for keyword in DISEASE_KEYWORDS:
        if _normalize(keyword) in normalized:
            found.append(keyword)

    patterns = [
        r"\b([a-z][a-z -]+(?:disease|syndrome|cancer|carcinoma|diabetes|hypertension|depression|asthma))\b",
    ]
    for pattern in patterns:
        found.extend(match.group(1) for match in re.finditer(pattern, normalized))

    return _dedupe(found)


def _major_secondary_diseases(primary: str, scores, reasons) -> list[str]:
    secondary = []
    primary_norm = _normalize(primary)

    for disease, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        disease_norm = _normalize(disease)
        if disease_norm == primary_norm or _same_disease_family(disease_norm, primary_norm):
            continue

        sources = set(reasons[disease])
        has_major_evidence = (
            score >= 6
            or "Title" in sources
            or ("MeSH Terms" in sources and "Abstract" in sources)
            or Counter(reasons[disease])["Abstract"] >= 2
        )
        if has_major_evidence:
            secondary.append(disease)

    return secondary


def _same_disease_family(left: str, right: str) -> bool:
    if not left or not right:
        return False

    diabetes_terms = {"diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "prediabetes"}
    if left in diabetes_terms and right in diabetes_terms:
        return True

    return left in right or right in left


def _matches_requested(disease: str, requested_disease: str) -> bool:
    disease_norm = _normalize(disease)
    requested_norm = _normalize(requested_disease)
    return bool(
        disease_norm == requested_norm
        or requested_norm in disease_norm
        or disease_norm in requested_norm
    )


def _build_reason(disease: str, sources: list[str], score: float) -> str:
    unique_sources = _dedupe(sources)
    return (
        f"Classified '{_display_name(disease)}' as primary disease based on weighted evidence "
        f"from {', '.join(unique_sources)}; weighted score={round(score, 2)}."
    )


def _stringify_existing_metadata(record: dict) -> str:
    keys = [
        "Primary_Disease",
        "Primary Disease",
        "Disease",
        "Conditions",
        "Extracted Diseases",
    ]
    return " ".join(str(record.get(key, "")) for key in keys)


def _normalize(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _display_name(value: str) -> str:
    value = str(value or "").strip()
    known = {
        "copd": "COPD",
        "hiv": "HIV",
        "covid-19": "COVID-19",
    }
    return known.get(value.lower(), value.title())


def _dedupe(values) -> list[str]:
    seen = set()
    result = []

    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value)

    return result
