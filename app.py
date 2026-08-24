#!/usr/bin/env python
# coding: utf-8
"""
app.py

A Streamlit UI for research_agent.py. Each click of "Start Research" runs a
fresh, independent agent loop — no state is persisted across runs or shared
via localStorage/browser sessions.

Usage:
    streamlit run app.py
"""

import re

import streamlit as st

from research_agent import MAX_STEPS, run_agent, run_evals, save_run

# Matches a leading "Findings" heading the model sometimes writes itself
# (e.g. "1. Findings:", "Findings:", "**Findings**") so it isn't duplicated
# under our own "#### Findings" header.
_LEADING_FINDINGS_HEADING_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?\**findings\**:?\s*\n+", re.IGNORECASE
)

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

st.title("🔎 Research Agent")

goal = st.text_input(
    "Research question",
    placeholder="e.g. What is the gold price today, and what moved it this past month?",
)
max_steps = st.slider("Max steps", min_value=3, max_value=15, value=MAX_STEPS)
start_clicked = st.button("Start Research", type="primary")


def summarize_step(action, decision, result):
    """Build a short, human-readable summary line for one agent step."""
    if action == "SEARCH":
        query = decision.get("query", "")
        n = len(result) if isinstance(result, list) else 0
        return f'query: "{query}" — {n} result(s) found'

    if action == "READ":
        url = decision.get("url", "")
        if isinstance(result, str) and result.startswith("[failed to read:"):
            return f"{url} — {result}"
        return f"{url} — read OK"

    if action == "FINISH":
        return "producing final report"

    if action == "ERROR":
        return decision.get("detail", "unknown error")

    return str(decision)


def action_icon(action):
    return {
        "SEARCH": "🔍",
        "READ": "📄",
        "FINISH": "✅",
        "ERROR": "⚠️",
    }.get(action, "•")


def render_report(report):
    """Split the report into findings / SOURCES READ / LINKS FOUND BUT NOT READ."""
    if "SOURCES READ:" in report:
        findings_part, rest = report.split("SOURCES READ:", 1)
    else:
        findings_part, rest = report, ""

    if "LINKS FOUND BUT NOT READ:" in rest:
        sources_part, links_part = rest.split("LINKS FOUND BUT NOT READ:", 1)
    else:
        sources_part, links_part = rest, ""

    findings_text = _LEADING_FINDINGS_HEADING_RE.sub("", findings_part.strip(), count=1).strip()

    st.markdown("#### Findings")
    st.markdown(findings_text or "_No findings._")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sources Read")
        st.markdown(sources_part.strip() or "_None._")
    with col2:
        st.markdown("#### Links Found But Not Read")
        st.markdown(links_part.strip() or "_None._")


if start_clicked:
    if not goal.strip():
        st.warning("Please enter a research question.")
    else:
        step_log = []
        steps_slot = st.empty()

        def render_steps_expander(finished):
            # Expanded while the agent is still running so progress is visible;
            # collapsed once it's done, leaving just a clickable title behind.
            label = f"Agent Steps ({len(step_log)} steps)" if finished else "Agent Steps (running…)"
            with steps_slot.container():
                with st.expander(label, expanded=not finished):
                    for line in step_log:
                        st.markdown(line)

        render_steps_expander(finished=False)

        def on_step(step_number, decision, result):
            action = decision.get("action", "UNKNOWN")
            summary = summarize_step(action, decision, result)
            step_log.append(f"**{action_icon(action)} Step {step_number}: {action}** — {summary}")
            render_steps_expander(finished=False)

        with st.spinner("Researching..."):
            state, report = run_agent(goal, on_step=on_step, max_steps=max_steps)

        render_steps_expander(finished=True)

        st.subheader("Final Report")
        render_report(report)

        with st.expander("Eval Results"):
            eval_summary = run_evals(state, report)
            for check in eval_summary["checks"]:
                icon = "✅" if check["passed"] else "❌"
                st.markdown(f"{icon} {check['name']}")
            st.markdown(f"**Score:** {eval_summary['score']}")

            mismatch = eval_summary["sources_mismatch"]
            if mismatch["missing_from_list"]:
                st.markdown(f"**Read but missing from SOURCES READ:** {mismatch['missing_from_list']}")
            if mismatch["falsely_claimed"]:
                st.markdown(f"**Claimed in SOURCES READ but never read:** {mismatch['falsely_claimed']}")

        save_run(goal, state, report, eval_summary)
