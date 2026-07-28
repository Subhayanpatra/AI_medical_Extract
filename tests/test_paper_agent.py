import unittest
from unittest.mock import patch

from agents.paper_agent import process_paper


class PaperAgentPipelineTests(unittest.TestCase):
    @patch("agents.paper_agent.outcome_agent")
    @patch("agents.paper_agent.analysis_agent")
    @patch("agents.paper_agent.code_extraction_agent")
    @patch("agents.paper_agent.slr_agent")
    @patch("agents.paper_agent.relevance_agent")
    @patch("agents.paper_agent.extract_pmc_supplementary_material")
    @patch("agents.paper_agent.parse_pmc_xml")
    @patch("agents.paper_agent.get_pmc_xml")
    def test_baseline_stops_after_relevance(
        self,
        get_pmc_xml,
        parse_pmc_xml,
        extract_supplementary,
        relevance_agent,
        slr_agent,
        code_agent,
        analysis_agent,
        outcome_agent,
    ):
        get_pmc_xml.return_value = "<article/>"
        parse_pmc_xml.return_value = {
            "Full_Text": "Complete article text",
            "Sections": {"Methods": "Study methods"},
            "Figures": ["Figure caption"],
            "Tables": ["Table content"],
        }
        extract_supplementary.return_value = {
            "Supplementary_Content": "Supplement text",
            "Supplementary_Status": "Downloaded",
            "Supplementary_Files": [{"name": "supplement.pdf"}],
        }
        relevance_agent.return_value = {
            "Relevant": True,
            "Relevance_Score": 0.9,
            "Relevance_Reason": "Matches the query",
        }
        slr_agent.return_value = {
            "Is_SLR": False,
            "Study_Design": "Retrospective Cohort Study",
        }

        result = process_paper(
            "PMC123",
            title="Example",
            query="diabetes",
            run_extended_agents=False,
        )

        self.assertEqual(result["Full_Text"], "Complete article text")
        self.assertEqual(result["Supplementary_Status"], "Downloaded")
        self.assertTrue(result["Relevant"])
        self.assertIsNone(result["Is_SLR"])
        self.assertEqual(result["Study_Design"], "")
        self.assertEqual(result["ICD_10_CM"], "")
        self.assertEqual(result["Analysis"], "")
        self.assertEqual(result["Outcome"], "")
        self.assertEqual(result["Country"], "")
        slr_agent.assert_not_called()
        code_agent.assert_not_called()
        analysis_agent.assert_not_called()
        outcome_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
