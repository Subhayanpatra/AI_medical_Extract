from agents.analysis_agent import analysis_agent
from agents.code_agent import EMPTY_CODE_RESULT, code_extraction_agent
from agents.method_agent import method_agent
from agents.outcome_agent import outcome_agent
from parser.section_extractor import extract_sections
from pubmed.downloader import get_bioc_xml


def process_paper(pmcid: str) -> dict:
    soup = get_bioc_xml(pmcid)
    sections = extract_sections(soup)

    if not sections:
        return {
            **EMPTY_CODE_RESULT,
            "Method": "Full text not available in PMC BioC",
            "Method_Evidence": "",
            "Analysis": [],
            "Analysis_Evidence": [],
            "Outcome": "Full text not available in PMC BioC",
            "Outcome_Evidence": "",
        }

    return {
        **code_extraction_agent(sections),
        **method_agent(sections),
        **analysis_agent(sections),
        **outcome_agent(sections),
    }
