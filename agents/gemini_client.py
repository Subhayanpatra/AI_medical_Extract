import json
import re

import google.generativeai as genai

from config import GEMINI_API_KEY


MODEL_NAME = "gemini-2.5-flash"


def generate_json(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in .streamlit/secrets.toml")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    text = response.text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)
