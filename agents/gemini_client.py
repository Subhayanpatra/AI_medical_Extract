import json
import re

from google import genai
from google.genai import types

from config import GEMINI_API_KEY


MODEL_NAME = "gemini-2.5-flash"


def generate_json(prompt: str, model_name: str = MODEL_NAME) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in .streamlit/secrets.toml")

    client = genai.Client(api_key=GEMINI_API_KEY.strip())
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)
