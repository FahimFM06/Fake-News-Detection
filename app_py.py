# app.py
# Main entry for Streamlit multipage app.
# Put this file in your project root (same level as "pages/" folder).

import streamlit as st

st.set_page_config(
    page_title="Fake News Detection (RoBERTa)",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: Small global CSS for consistent fonts/cards
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .app-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 22px 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      }
      .small-muted { color: rgba(0,0,0,0.60); font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page content (acts like a "Home" / landing)
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.title("Fake News Detection System")
st.write(
    "This Streamlit application uses a fine-tuned **RoBERTa-base** model to classify news text as **FAKE (0)** or **REAL (1)**."
)
st.markdown(
    """
    **How to use:**
    - Go to **Introduction** to see project details and workflow.
    - Go to **Input** to paste a news headline/article.
    - Go to **Result** to view the predicted label and confidence.
    """
)
st.markdown('<p class="small-muted">Tip: Use the sidebar to navigate between pages.</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
