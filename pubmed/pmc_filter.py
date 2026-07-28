from .query_builder import build_query
from .search import search_pubmed
from .metadata import (
    attach_pmcids_from_links,
    fetch_metadata,
    merge_pubmed_metadata,
    parse_articles,
    parse_pmc_summaries,
)
from Bio import Entrez
from config import EMAIL, NCBI_API_KEY
import time


Entrez.email = EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


def _fetch_pmc_links(pmids: list[str], chunk_size: int = 20) -> dict[str, str]:
    """Map PubMed PMIDs to PMC IDs through NCBI links."""
    if not pmids:
        return {}

    pmid_to_pmcid = {}

    for index in range(0, len(pmids), chunk_size):
        chunk = pmids[index:index + chunk_size]
        pmid_to_pmcid.update(_fetch_pmc_links_chunk(chunk))

    return pmid_to_pmcid


def _fetch_pmc_links_chunk(pmids: list[str], retries: int = 3) -> dict[str, str]:
    """Fetch PMC links for a small PMID chunk with retry fallback."""
    pmid_to_pmcid = {}

    for attempt in range(retries):
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=",".join(pmids),
                linkname="pubmed_pmc",
            )
            link_sets = Entrez.read(handle)
            handle.close()
            break
        except Exception:
            link_sets = []
            time.sleep(1 + attempt)

    if not link_sets and len(pmids) > 1:
        for pmid in pmids:
            pmid_to_pmcid.update(_fetch_pmc_links_chunk([pmid], retries=retries))
        return pmid_to_pmcid

    for link_set in link_sets:
        source_ids = [str(item) for item in link_set.get("IdList", [])]
        pmid = source_ids[0] if source_ids else ""

        for link_set_db in link_set.get("LinkSetDb", []):
            for link in link_set_db.get("Link", []):
                pmc_numeric_id = str(link.get("Id", ""))
                if pmid and pmc_numeric_id:
                    pmid_to_pmcid[pmid] = f"PMC{pmc_numeric_id}"
                    break

    return pmid_to_pmcid


def get_required_pmc_papers(
        disease,
        required_papers=20,
        batch_size=100,
        start_year=None,
        end_year=None,
        max_batches=20,
):
    """Return PMCID papers matching the PubMed search query."""

    collected = []
    retstart = 0
    batches_checked = 0
    seen_pmcids = set()

    query = build_query(disease, start_year=start_year, end_year=end_year)

    while len(collected) < required_papers and batches_checked < max_batches:

        pmids = search_pubmed(
            query=query,
            retmax=batch_size,
            retstart=retstart,
        )

        if not pmids:
            break

        articles = fetch_metadata(pmids)
        parsed_articles = parse_articles(articles)
        parsed_articles = attach_pmcids_from_links(parsed_articles, _fetch_pmc_links(pmids))

        for article in parsed_articles:
            pmcid = article.get("PMCID", "")

            if pmcid and pmcid not in seen_pmcids:
                collected.append(article)
                seen_pmcids.add(pmcid)

            if len(collected) >= required_papers:
                break

        retstart += batch_size
        batches_checked += 1

    if len(collected) < required_papers:
        needed = required_papers - len(collected)
        for article in _get_required_papers_from_pmc(
                disease=disease,
                required_papers=needed,
                batch_size=batch_size,
                start_year=start_year,
                end_year=end_year,
                max_batches=max_batches,
        ):
            pmcid = article.get("PMCID", "")
            if pmcid and pmcid not in seen_pmcids:
                collected.append(article)
                seen_pmcids.add(pmcid)
            if len(collected) >= required_papers:
                break

    return collected[:required_papers]


def _get_required_papers_from_pmc(
        disease,
        required_papers=20,
        batch_size=100,
        start_year=None,
        end_year=None,
        max_batches=20,
):
    collected = []
    seen_pmcids = set()
    retstart = 0
    batches_checked = 0
    query = _build_pmc_query(disease, start_year=start_year, end_year=end_year)

    while len(collected) < required_papers and batches_checked < max_batches:
        try:
            search_handle = Entrez.esearch(
                db="pmc",
                term=query,
                retmax=batch_size,
                retstart=retstart,
                sort="relevance",
            )
            search_result = Entrez.read(search_handle)
            search_handle.close()
            pmc_ids = search_result.get("IdList", [])
        except Exception:
            pmc_ids = []

        if not pmc_ids:
            break

        try:
            summary_handle = Entrez.esummary(
                db="pmc",
                id=",".join(pmc_ids),
            )
            summaries = Entrez.read(summary_handle)
            summary_handle.close()
            parsed_records = parse_pmc_summaries(summaries)
            pmids = [record["PMID"] for record in parsed_records if record.get("PMID")]
            parsed_records = merge_pubmed_metadata(
                parsed_records,
                parse_articles(fetch_metadata(pmids)),
            )
        except Exception:
            parsed_records = []

        for article in parsed_records:
            pmcid = article.get("PMCID", "")

            if pmcid and pmcid not in seen_pmcids:
                collected.append(article)
                seen_pmcids.add(pmcid)

            if len(collected) >= required_papers:
                break

        retstart += batch_size
        batches_checked += 1

    return collected[:required_papers]


def _build_pmc_query(disease: str, start_year=None, end_year=None) -> str:
    query = disease.strip()

    if start_year and end_year:
        query += f" AND {int(start_year)}:{int(end_year)}[pdat]"

    return query
