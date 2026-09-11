"""
app.py
======
Unified Entry Point for the Insurance Business Analytics Engine Dashboard.

Run via:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

# Load .env file for local development (no-op if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root, src, and ui directories to Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for sub in ["src", "ui", "src/engine", "src/analytics", "src/retrieval", "src/llm"]:
    p = os.path.join(ROOT_DIR, sub)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

MODES = {
    "Conversation": "chat_dashboard",
    "Classic dashboard": "business_dashboard",
}

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Insurance Business Analytics Engine",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    with st.sidebar:
        mode = st.radio("Mode", list(MODES), index=0, key="app_mode",
                        help="Conversation supports follow-up questions; the classic "
                             "dashboard answers each question independently.")
        st.divider()

    import importlib
    importlib.import_module(MODES[mode]).render()
