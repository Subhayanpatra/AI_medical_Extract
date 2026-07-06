import importlib
import inspect
import os

import pandas as pd
import streamlit as st

import agents.disease_normalization_agent
import pubmed.metadata
import pubmed.pmc_filter
import pubmed.query_builder
import pubmed.search


st.set_page_config(
    page_title="PMCID Paper Finder",
    layout="wide",
)

st.title("PMCID Paper Finder")

with st.sidebar:
    st.header("Search")
    disease = st.text_input("Disease name", placeholder="diabetes")
    paper_count = st.number_input(
        "Number of PMCID papers",
        min_value=1,
        max_value=200,
        value=30,
        step=1,
    )
    use_year_filter = st.checkbox("Filter by publication year")
    run_agents = st.checkbox("Run AI extraction agents")
    max_agent_papers = st.number_input(
        "Papers to analyze with agents",
        min_value=1,
        max_value=200,
        value=5,
        step=1,
        disabled=not run_agents,
    )

    start_year = None
    end_year = None
    if use_year_filter:
        start_year = st.number_input("Start year", min_value=1900, max_value=2100, value=2020)
        end_year = st.number_input("End year", min_value=1900, max_value=2100, value=2026)

    search = st.button("Find papers", type="primary", use_container_width=True)

st.caption("Searches PubMed and returns papers that have a PMCID code.")

if search:
    if not disease.strip():
        st.error("Enter a disease name.")
    elif use_year_filter and int(start_year) > int(end_year):
        st.error("Start year must be less than or equal to end year.")
    else:
        normalized_disease = None
        normalization_result = None
        with st.spinner("Checking and normalizing the disease name..."):
            try:
                importlib.reload(agents.disease_normalization_agent)
                normalization_result = agents.disease_normalization_agent.normalize_disease(
                    disease.strip()
                )
                if normalization_result["is_valid_disease"]:
                    normalized_disease = normalization_result["official_disease_name"]
                else:
                    st.error(
                        f'"{disease.strip()}" was not recognized as a valid disease. '
                        "Enter a disease name and try again."
                    )
            except Exception as exc:
                st.error(f"Disease normalization failed: {exc}")

        if normalized_disease:
            if normalization_result.get("normalization_warning"):
                st.warning(
                    "Gemini normalization is unavailable because GEMINI_API_KEY is invalid. "
                    "A local disease-name fallback was used. Add a valid Google AI Studio "
                    "API key to enable the AI agent."
                )
            st.info(
                f"Searching for: {normalized_disease} "
                f"(normalization confidence: {normalization_result['confidence']:.0%}, "
                f"method: {normalization_result.get('normalization_method', 'Gemini')})"
            )

        if normalized_disease:
            with st.spinner(f"Finding {paper_count} PMCID papers for {normalized_disease}..."):
                try:
                    importlib.reload(pubmed.query_builder)
                    importlib.reload(pubmed.search)
                    importlib.reload(pubmed.metadata)
                    importlib.reload(pubmed.pmc_filter)

                    query = pubmed.query_builder.build_query(
                        normalized_disease,
                        start_year=int(start_year) if start_year else None,
                        end_year=int(end_year) if end_year else None,
                    )
                    papers = pubmed.pmc_filter.get_required_pmc_papers(
                        disease=normalized_disease,
                        required_papers=int(paper_count),
                        start_year=int(start_year) if start_year else None,
                        end_year=int(end_year) if end_year else None,
                    )
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    papers = []
                    query = ""
        else:
            papers = []
            query = ""

        if normalized_disease:
            with st.expander("Debug search details"):
                st.write(
                    {
                        "working_directory": os.getcwd(),
                        "main_file": __file__,
                        "pmc_filter_file": inspect.getfile(pubmed.pmc_filter),
                        "disease_normalization": normalization_result,
                        "query": query,
                        "requested": int(paper_count),
                        "returned": len(papers),
                    }
                )

        if papers:
            df = pd.DataFrame(papers)
            agent_view = None

            if run_agents:
                from agents.paper_agent import process_paper

                agent_results = []
                progress = st.progress(0)
                status = st.empty()
                papers_to_analyze = min(int(max_agent_papers), len(df))

                for index, row in df.head(papers_to_analyze).iterrows():
                    status.write(
                        f"Running agents for {index + 1}/{papers_to_analyze}: {row['PMCID']}"
                    )
                    try:
                        agent_results.append(process_paper(row["PMCID"]))
                    except Exception as exc:
                        agent_results.append({"Agent_Error": str(exc)})
                    progress.progress((index + 1) / papers_to_analyze)

                agent_df = pd.DataFrame(agent_results)
                agent_view = pd.concat(
                    [
                        df.head(papers_to_analyze).reset_index(drop=True),
                        agent_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
                df = pd.concat(
                    [
                        df.reset_index(drop=True),
                        agent_df.reindex(range(len(df))).reset_index(drop=True),
                    ],
                    axis=1,
                )
                status.write("Agent extraction complete.")

                if agent_results:
                    st.subheader("AI Extraction Preview")
                    st.json(agent_results[0])

            if len(df) == int(paper_count):
                st.success(f"Found all {len(df)} requested PMCID paper(s).")
            else:
                st.warning(
                    f"Requested {int(paper_count)} PMCID paper(s), but found {len(df)}. "
                    "Try a broader disease name or remove year filters."
                )

            if agent_view is not None:
                st.subheader(f"AI Agent Results ({len(agent_view)} paper)")
                preferred_agent_columns = [
                    "PMCID",
                    "PMID",
                    "Title",
                    "Primary_Disease",
                    "ICD_9_CM",
                    "ICD_10_CM",
                    "ICD_10_PCS",
                    "CPT",
                    "HCPCS",
                    "NDC",
                    "Method",
                    "Analysis",
                    "Outcome",
                    "Agent_Error",
                ]
                visible_agent_columns = [
                    column for column in preferred_agent_columns if column in agent_view.columns
                ]
                st.dataframe(
                    agent_view[visible_agent_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader(f"PMCID Search Results ({len(df)} papers)")
            base_columns = [
                "PMCID",
                "PMID",
                "Title",
                "Primary Disease",
                "Major Secondary Diseases",
                "Disease Confidence Score",
                "Disease Validation Reason",
                "Accepted",
                "Rejection Reason",
                "Journal",
                "PublicationYear",
                "Authors",
                "DOI",
                "Abstract",
                "MeSH Terms",
                "Keywords",
                "Publication Types",
                "Chemical List",
                "PubMedURL",
                "PMCURL",
            ]
            base_columns = [column for column in base_columns if column in df.columns]
            agent_columns = [
                column
                for column in df.columns
                if column not in base_columns
            ]

            if agent_columns:
                st.subheader("AI Agent Results")
                st.dataframe(
                    df[base_columns + agent_columns],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(
                    df[base_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{normalized_disease.replace(' ', '_')}_pmcid_papers.csv",
                mime="text/csv",
            )
        elif normalized_disease:
            st.warning("No PMCID papers found for this search.")
else:
    st.info("Enter a disease name and paper count in the sidebar, then click Find papers.")
