import json
import re
import time

from google import genai
from google.genai import types

from config import GEMINI_API_KEY


MODEL_NAME = "gemini-2.5-flash"
RETRYABLE_ERROR_MARKERS = (
    "503",
    "UNAVAILABLE",
    "overloaded",
    "429",
    "RESOURCE_EXHAUSTED",
    "rate limit",
)


def generate_json(prompt: str, model_name: str = MODEL_NAME, max_retries: int = 4) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to the project .env file.")

    client = genai.Client(api_key=GEMINI_API_KEY.strip())
    last_error = None

    for attempt in range(max_retries + 1):
        try:
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
        except Exception as exc:
            last_error = exc
            message = str(exc)
            is_retryable = any(
                marker.lower() in message.lower()
                for marker in RETRYABLE_ERROR_MARKERS
            )

            if not is_retryable or attempt >= max_retries:
                raise

            time.sleep(min(30, 3 * (2 ** attempt)))

    raise last_error
