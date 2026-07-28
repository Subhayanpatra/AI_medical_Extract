from agents.analysis_agent import analysis_agent
from agents.code_agent import EMPTY_CODE_RESULT, code_extraction_agent
from agents.outcome_agent import outcome_agent
from agents.relevance_agent import relevance_agent
from agents.slr_agent import slr_agent
from parser.supplementary import extract_pmc_supplementary_material
from parser.xml_parser import parse_pmc_xml
from pubmed.downloader import get_pmc_xml


def process_paper(
    pmcid: str,
    title: str = "",
    query: str = "",
    include_supplementary: bool = True,
    run_extended_agents: bool = True,
) -> dict:
    xml_data = get_pmc_xml(pmcid)
    parsed = parse_pmc_xml(xml_data)

    full_text = parsed.get("Full_Text", "")
    sections = parsed.get("Sections", {})
    figures = parsed.get("Figures", [])
    tables = parsed.get("Tables", [])
    if include_supplementary:
        try:
            supplementary = extract_pmc_supplementary_material(
                pmcid,
                xml_content=xml_data,
            )
        except Exception as exc:
            supplementary = {
                "Supplementary_Content": "",
                "Supplementary_Status": f"Processing failed: {exc}",
                "Supplementary_Files": [],
            }
    else:
        supplementary = {
            "Supplementary_Content": "",
            "Supplementary_Status": "Not requested",
            "Supplementary_Files": [],
        }
    supplementary_content = supplementary.get("Supplementary_Content", "")

    if not full_text:
        return {
            **{key: "" for key in EMPTY_CODE_RESULT},
            "Full_Text": "",
            "Sections": {},
            "Figures": [],
            "Tables": [],
            **supplementary,
            "Relevant": False,
            "Relevance_Score": 0.0,
            "Relevance_Reason": "Full text not available in PMC XML",
            "Is_SLR": None,
            "Study_Design": "Unclear" if run_extended_agents else "",
            "Analysis": "",
            "Outcome": "Full text not available in PMC XML",
            "Country": "",
        }

    relevance_result = relevance_agent(query, title, full_text)
    result = {
        "Full_Text": full_text,
        "Sections": sections,
        "Figures": figures,
        "Tables": tables,
        **supplementary,
        **relevance_result,
    }

    if not run_extended_agents:
        return {
            **result,
            "Is_SLR": None,
            "Study_Design": "",
            **{key: "" for key in EMPTY_CODE_RESULT},
            "Analysis": "",
            "Outcome": "",
            "Country": "",
        }

    if relevance_result["Relevant"]:
        slr_result = slr_agent(title, sections, full_text)
    else:
        slr_result = {
            "Is_SLR": False,
            "Study_Design": "Not assessed (not relevant)",
        }

    return {
        **result,
        **slr_result,
        **code_extraction_agent(
            title,
            full_text,
            tables,
            figures,
            supplementary_content,
        ),
        **analysis_agent(
            full_text,
            tables,
            figures,
            supplementary_content,
        ),
        **outcome_agent(
            full_text,
            tables,
            figures,
            supplementary_content,
        ),
    }
