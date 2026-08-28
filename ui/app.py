"""ui/app.py — landing page and navigation entry point for the pipeline's
Streamlit multipage app.

Streamlit's multipage convention auto-discovers every script under
ui/pages/ and lists them in the sidebar; this file only needs to exist and
call st.set_page_config() once, since that call is only valid on the
FIRST script Streamlit runs in a session. Kept intentionally minimal --
this is a landing page, not a dashboard; each page under ui/pages/ owns its
own data loading and layout.
"""
import streamlit as st

st.set_page_config(page_title="Clinical Neuro-Symbolic Pipeline", page_icon="🧠", layout="wide")

st.title("Clinical Neuro-Symbolic Pipeline")
st.markdown(
    """
Use the sidebar to navigate:

- **🩺 HITL Review Queue** — review Stage 3/Objective 3 decisions before KG3 write-back.
- **🚀 Pipeline Runner** — run Stage 1→2a→2b→3 on a note.
- **🔍 Troubleshooting** — pick or type any note_id and step through Stage 1→2a→2b→3 for it, input/output at each stage, with a full gold comparison.
- **📊 Evaluation Metrics** — gold-recall and calibration reports.
"""
)
