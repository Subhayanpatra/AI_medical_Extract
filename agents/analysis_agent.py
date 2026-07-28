from typing import Any

from .gemini_client import generate_json


ANALYSIS_PROMPT = """
You are a senior biomedical research analysis-method extraction expert.

Your ONLY task is to identify the statistical, epidemiological, mathematical, and computational analysis methods explicitly used by the authors in the biomedical research article.

Read all available article content, including METHODS, STATISTICAL ANALYSIS, DATA ANALYSIS, RESULTS, TABLES, TABLE FOOTNOTES, FIGURES, FIGURE CAPTIONS, SUPPLEMENTARY METHODS, DISCUSSION, CONCLUSION, and FULL TEXT.

Give highest priority to:

1. METHODS
2. STATISTICAL ANALYSIS or DATA ANALYSIS subsections
3. RESULTS
4. TABLES and table footnotes
5. FIGURE captions

Use remaining content only when analysis methods are not clearly reported in
the priority sections.

Extract only statistical, epidemiological, mathematical, machine-learning, and computational analysis methods explicitly used in the study.

Examples include, but are not limited to:

Descriptive and comparative statistical methods:

- Descriptive Statistics
- Mean and Standard Deviation
- Median and Interquartile Range
- Chi-square Test
- Fisher's Exact Test
- Student's t-test
- Paired t-test
- Welch's t-test
- ANOVA
- Repeated-Measures ANOVA
- ANCOVA
- Mann-Whitney U Test
- Wilcoxon Signed-Rank Test
- Wilcoxon Rank-Sum Test
- Kruskal-Wallis Test
- McNemar Test

Regression and multivariable methods:

- Logistic Regression
- Multivariable Logistic Regression
- Linear Regression
- Multiple Linear Regression
- Poisson Regression
- Negative Binomial Regression
- Cox Proportional Hazards Regression
- Fine-Gray Competing Risks Regression
- Generalized Linear Model
- Generalized Additive Model
- Mixed Effects Model
- Multilevel Model
- Generalized Estimating Equations

Survival and time-to-event methods:

- Kaplan-Meier Analysis
- Log-rank Test
- Cox Regression
- Competing Risks Analysis
- Cumulative Incidence Analysis

Association, diagnostic, and model-performance methods:

- Pearson Correlation
- Spearman Correlation
- Partial Correlation
- ROC Analysis
- Area Under the Curve Analysis
- Calibration Analysis
- Discrimination Analysis

Causal-inference and real-world evidence methods:

- Propensity Score Matching
- Propensity Score Weighting
- Inverse Probability of Treatment Weighting
- Stabilized Weighting
- Doubly Robust Estimation
- Difference-in-Differences
- Instrumental Variable Analysis
- Marginal Structural Model
- Interrupted Time-Series Analysis
- Regression Discontinuity
- Target Trial Emulation

Longitudinal and repeated-measures methods:

- Longitudinal Analysis
- Repeated-Measures Model
- Time-Series Analysis

Meta-analysis and evidence-synthesis methods:

- Fixed-Effect Meta-analysis
- Random-Effects Meta-analysis
- Heterogeneity Analysis
- Meta-regression
- Subgroup Analysis
- Publication Bias Analysis
- Funnel Plot Analysis
- Egger's Test

Machine-learning and computational methods:

- Random Forest
- XGBoost
- Gradient Boosting
- LightGBM
- Support Vector Machine
- Decision Tree
- Neural Network
- Deep Learning
- Convolutional Neural Network
- Recurrent Neural Network
- LSTM
- Natural Language Processing
- Clustering
- Principal Component Analysis
- Feature Selection
- Cross-validation
- Bootstrapping

Other explicitly reported analytical methods:

- Sensitivity Analysis
- Interaction Analysis
- Mediation Analysis
- Missing-Data Analysis
- Multiple Imputation
- Bonferroni Correction
- False Discovery Rate Correction

STRICT RULES

1. Extract only analysis methods explicitly stated as used by the authors.
2. Do not infer a method from statistical values or outputs.
3. Do not extract a method merely because it appears in the introduction, discussion, references, or description of another study.
4. Do not extract study designs such as randomized controlled trial, cohort study, case-control study, cross-sectional study, systematic review, observational study, case report, or case series.
5. Do not extract diseases, treatments, therapies, outcomes, endpoints, medical coding systems, databases, data sources, eligibility criteria, laboratory procedures, or imaging-acquisition procedures.
6. Do not treat software names such as SAS, R, Python, SPSS, Stata, MATLAB, TensorFlow, or PyTorch as analysis methods.
7. Do not extract statistical measures such as p-value, confidence interval, odds ratio, hazard ratio, risk ratio, standard deviation, or standard error as standalone methods.
8. Do not extract covariate adjustment as a method unless the model type is stated.
9. Preserve the method name as written whenever it is clear.
10. Standardize only obvious equivalent expressions:
    - Cox model -> Cox Proportional Hazards Regression
    - Kaplan-Meier method -> Kaplan-Meier Analysis
    - chi-squared test -> Chi-square Test
    - propensity-score matching -> Propensity Score Matching
    - inverse probability weighting -> Inverse Probability of Treatment Weighting
    - random effects model for pooled estimates -> Random-Effects Meta-analysis
11. Remove duplicate methods.
12. If the same method appears multiple times, return it only once.
13. If no analysis method is explicitly reported, return an empty list.
14. Return only valid JSON.
15. Do not provide explanations.
16. Do not use Markdown.
17. Do not wrap the JSON in code fences.

OUTPUT FORMAT

Return exactly this JSON structure:

{
    "Analysis": []
}

ARTICLE CONTENT

FULL TEXT:
__FULL_TEXT__

TABLES:
__TABLES__

FIGURES:
__FIGURES__

SUPPLEMENTARY CONTENT:
__SUPPLEMENTARY_CONTENT__
"""


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {_content_to_text(item)}"
            for key, item in value.items()
            if _content_to_text(item)
        )
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_content_to_text(item) for item in value if _content_to_text(item))
    return str(value).strip()


def _normalize_analysis(data: dict) -> list[str]:
    if not isinstance(data, dict):
        return []

    values = data.get("Analysis", [])
    if not isinstance(values, list):
        return []

    methods = []
    seen = set()
    for method in values:
        if isinstance(method, dict):
            method = method.get("Method") or method.get("Analysis") or ""
        method = str(method).strip()
        key = method.casefold()
        if method and key not in seen:
            methods.append(method)
            seen.add(key)
    return methods


def analysis_agent(
    full_text: Any,
    tables: Any = "",
    figures: Any = "",
    supplementary_content: Any = "",
) -> dict:
    full_text_value = _content_to_text(full_text)
    tables_text = _content_to_text(tables)
    figures_text = _content_to_text(figures)
    supplementary_text = _content_to_text(supplementary_content)

    if not " ".join(
        [full_text_value, tables_text, figures_text, supplementary_text]
    ).strip():
        return {"Analysis": ""}

    prompt = (
        ANALYSIS_PROMPT
        .replace("__FULL_TEXT__", full_text_value)
        .replace("__TABLES__", tables_text)
        .replace("__FIGURES__", figures_text)
        .replace("__SUPPLEMENTARY_CONTENT__", supplementary_text)
    )

    try:
        methods = _normalize_analysis(generate_json(prompt))
        return {"Analysis": "; ".join(methods)}
    except Exception as exc:
        return {"Analysis": "", "Analysis_Agent_Error": str(exc)}
