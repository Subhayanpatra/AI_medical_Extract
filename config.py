import os


def _streamlit_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


EMAIL = os.getenv("NCBI_EMAIL") or _streamlit_secret("NCBI_EMAIL", "your_email@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY") or _streamlit_secret("NCBI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or _streamlit_secret("GEMINI_API_KEY", "")
