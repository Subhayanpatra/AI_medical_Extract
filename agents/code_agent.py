from .gemini_client import generate_json


EMPTY_CODE_RESULT = {
    "Primary_Disease": "",
    "ICD_9_CM": [],
    "ICD_10_CM": [],
    "ICD_10_PCS": [],
    "CPT": [],
    "HCPCS": [],
    "NDC": [],
}


def code_extraction_agent(sections: dict) -> dict:
    prompt = f"""
You are a senior biomedical information extraction expert.

Identify the SINGLE primary disease or medical condition that is the main focus of the article.
Then extract ONLY medical codes explicitly written for that primary disease.

Extract ONLY these coding systems if explicitly present:
- ICD-9-CM
- ICD-10-CM
- ICD-10-PCS
- CPT
- HCPCS
- NDC

Rules:
1. Never infer a code from a disease name.
2. Extract only codes explicitly written in the article.
3. Preserve every code exactly as written.
4. Remove duplicates.
5. Ignore codes for comorbidities, secondary diagnoses, complications, exclusions, adverse events, and unrelated diseases.
6. Return ONLY valid JSON.

Output format:
{{
  "Primary_Disease": "",
  "ICD_9_CM": [],
  "ICD_10_CM": [],
  "ICD_10_PCS": [],
  "CPT": [],
  "HCPCS": [],
  "NDC": []
}}

Article Sections:
{sections}
"""
    return {**EMPTY_CODE_RESULT, **generate_json(prompt)}
