from .gemini_client import generate_json


def analysis_agent(sections: dict) -> dict:
    prompt = f"""
You are a senior biomedical research analyst.

Extract statistical and computational analysis methods explicitly used in the article.
Focus on METHODS and RESULTS. Do not infer or guess.

Return ONLY valid JSON:
{{
  "Analysis": [],
  "Analysis_Evidence": []
}}

Article Sections:
{sections}
"""
    data = generate_json(prompt)
    if "Evidence" in data and "Analysis_Evidence" not in data:
        data["Analysis_Evidence"] = data.pop("Evidence")
    return {"Analysis": [], "Analysis_Evidence": [], **data}
