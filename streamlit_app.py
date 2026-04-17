"""
Streamlit deployment entrypoint for the Investment Thesis Copilot UI.

Run locally:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def _read_frontend_file(filename: str) -> str:
    """Read a frontend asset file and return its UTF-8 content."""
    return (FRONTEND_DIR / filename).read_text(encoding="utf-8")


def _build_embedded_page() -> str:
    """
    Inline CSS/JS into index.html so Streamlit can serve the full app
    from a single HTML payload.
    """
    index_html = _read_frontend_file("index.html")
    css = _read_frontend_file("styles.css")
    js = _read_frontend_file("app.js")

    index_html = index_html.replace(
        '<link rel="stylesheet" href="./styles.css" />',
        f"<style>{css}</style>",
    )
    index_html = index_html.replace(
        '<script src="./app.js"></script>',
        f"<script>{js}</script>",
    )
    return index_html


st.set_page_config(
    page_title="Investment Thesis Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Investment Thesis Copilot")
st.caption("Claude-style investment thesis workflow running in Streamlit.")

if not FRONTEND_DIR.exists():
    st.error("Frontend assets not found. Expected folder: `frontend/`")
else:
    html(_build_embedded_page(), height=1150, scrolling=True)
