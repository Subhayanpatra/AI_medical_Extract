"""Build PubMed search queries."""

from typing import Optional


def build_query(disease: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> str:
    query = f'("{disease}"[Title/Abstract]) AND free full text[sb]'

    if start_year and end_year:
        query += f' AND ("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'

    return query
