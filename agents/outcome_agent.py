from .gemini_client import generate_json


def outcome_agent(sections: dict) -> dict:
    prompt = f"""
You are a senior biomedical research analyst.

Identify the PRIMARY OUTCOME or main finding explicitly reported in the article.
Focus on RESULTS, DISCUSSION, and CONCLUSION. Do not infer or invent findings.

Return ONLY valid JSON:
{{
  "Outcome": "",
  "Outcome_Evidence": ""
}}

Article Sections:
{sections}
"""
    data = generate_json(prompt)
    if "Evidence" in data and "Outcome_Evidence" not in data:
        data["Outcome_Evidence"] = data.pop("Evidence")
    return {"Outcome": "", "Outcome_Evidence": "", **data}
