import requests
from bs4 import BeautifulSoup


def get_bioc_xml(pmcid: str):
    """Download PMC Open Access BioC XML for a PMCID."""
    clean_pmcid = str(pmcid).replace("PMC", "").strip()
    if not clean_pmcid:
        return None

    url = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        f"pmcoa.cgi/BioC_xml/PMC{clean_pmcid}/unicode"
    )
    response = requests.get(url, timeout=30)

    if response.status_code != 200 or "<collection>" not in response.text:
        return None

    return BeautifulSoup(response.text, "xml")
