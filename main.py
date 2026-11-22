"""
HR Screening LangGraph application
Run with: streamlit run main.py

Set OPENAI_API_KEY in your environment or Streamlit secrets.
"""
from __future__ import annotations

import os
from typing import TypedDict

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. Workflow state
# -----------------------------
class State(TypedDict, total=False):
    application: str
    experience_level: str
    skill_match: str
    response: str


# -----------------------------
# 2. LLM configuration
# -----------------------------
def get_api_key() -> str | None:
    """Read the API key from environment variables or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def build_llm() -> ChatOpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in your environment "
            "or add OPENAI_API_KEY to .streamlit/secrets.toml."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )


# -----------------------------
# 3. LangGraph node functions
# -----------------------------
def build_workflow(llm: ChatOpenAI):
    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Classify the applicant's experience as exactly one of: "
            "Entry-level, Mid-level, or Senior-level. Return only the label.\\n"
            "Application: {application}"
        )
        response = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content
        experience = str(response).strip()
        return {"experience_level": experience}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Assess this application for a Python Developer position. "
            "Return exactly Match or No Match, with no explanation.\\n"
            "Application: {application}"
        )
        response = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content
        skill_match = str(response).strip()
        return {"skill_match": skill_match}

    def schedule_hr_interview(state: State) -> State:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: State) -> State:
        return {
            "response": (
                "Candidate has senior-level experience but does not match "
                "the job skills. Please review manually."
            )
        }

    def reject_application(state: State) -> State:
        return {
            "response": (
                "Candidate does not meet the job requirements. "
                "Please review before making a final hiring decision."
            )
        }

    def route_app(state: State) -> str:
        skill_match = state.get("skill_match", "").strip().lower()
        experience = state.get("experience_level", "").strip().lower()

        if skill_match == "match":
            return "schedule_hr_interview"
        if experience == "senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    graph = StateGraph(State)
    graph.add_node("categorize_experience", categorize_experience)
    graph.add_node("assess_skillset", assess_skillset)
    graph.add_node("schedule_hr_interview", schedule_hr_interview)
    graph.add_node("escalate_to_recruiter", escalate_to_recruiter)
    graph.add_node("reject_application", reject_application)

    graph.add_edge(START, "categorize_experience")
    graph.add_edge("categorize_experience", "assess_skillset")
    graph.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    graph.add_edge("schedule_hr_interview", END)
    graph.add_edge("escalate_to_recruiter", END)
    graph.add_edge("reject_application", END)

    return graph.compile()


# -----------------------------
# 4. Streamlit user interface
# -----------------------------
st.set_page_config(
    page_title="HR Screening Agent",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 HR Screening Agent")
st.caption("LangGraph + OpenAI | Python Developer candidate screening")

st.warning(
    "This is a screening-support prototype. Use human review for every "
    "candidate; do not make final hiring or rejection decisions solely "
    "from an AI-generated assessment."
)

with st.sidebar:
    st.header("Configuration")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.write(f"Model: `{model_name}`")
    st.write("API key: " + ("Configured" if get_api_key() else "Not configured"))

st.subheader("Candidate application")
application_text = st.text_area(
    "Paste the candidate's application, CV summary, or relevant experience",
    height=260,
    placeholder=(
        "Example: Python developer with 5 years of experience in Python, "
        "Django, SQL, REST APIs and cloud deployment..."
    ),
)

if st.button("Screen candidate", type="primary", disabled=not application_text.strip()):
    try:
        with st.spinner("Evaluating candidate..."):
            llm = build_llm()
            app = build_workflow(llm)
            result = app.invoke(
                {
                    "application": application_text.strip(),
                    "experience_level": "",
                    "skill_match": "",
                    "response": "",
                }
            )

        st.success("Workflow completed")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Experience category", result.get("experience_level", "N/A"))
        with col2:
            st.metric("Skill assessment", result.get("skill_match", "N/A"))

        st.subheader("Workflow outcome")
        st.info(result.get("response", "No response returned."))

        with st.expander("View workflow state"):
            st.json(result)

    except Exception as exc:
        st.error(f"Unable to run screening: {exc}")
        st.info(
            "Check that dependencies are installed and OPENAI_API_KEY is "
            "configured correctly."
        )

st.divider()
st.caption(
    "Prototype note: Validate prompts, job criteria, fairness, privacy, "
    "audit logging, and human-review controls before production use."
)
