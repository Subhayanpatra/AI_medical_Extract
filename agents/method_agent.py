from .gemini_client import generate_json


def method_agent(sections: dict) -> dict:
    prompt = f"""
You are a senior biomedical research analyst.

Identify the PRIMARY STUDY DESIGN explicitly reported in the article.
Focus on METHODS, ABSTRACT, and TITLE. Do not infer.

Return ONLY valid JSON:
{{
  "Method": "",
  "Method_Evidence": ""
}}

Article Sections:
{sections}
"""
    data = generate_json(prompt)
    if "Evidence" in data and "Method_Evidence" not in data:
        data["Method_Evidence"] = data.pop("Evidence")
    return {"Method": "", "Method_Evidence": "", **data}
