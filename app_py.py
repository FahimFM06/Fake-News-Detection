# app.py
# ============================================================
# Fake News Detection (RoBERTa) - Streamlit Multi-Page App
# Main entry file.
#
# Folder structure:
#   app.py
#   pages/1_Introduction.py
#   pages/2_Input.py
#   pages/3_Result.py
#   assets/bg_intro.jpg
#   assets/bg_input.jpg
#   assets/bg_result.jpg
#   model/roberta_fake_news_model/  (your saved model folder)
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize shared session state keys (used across pages)
defaults = {
    "input_text": "",
    "pred_label": None,          # 0 or 1
    "probs": None,               # {"FAKE": float, "REAL": float}
    "model_name": "roberta-base",
    "last_error": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# This page can be a simple landing/redirect page.
st.title("Fake News Detection App")
st.write(
    "Use the pages in the left sidebar to navigate:\n"
    "- Introduction\n"
    "- Input\n"
    "- Result"
)

st.info(
    "Tip: Start with **Introduction**, then go to **Input** to paste text, and finally check **Result**."
)
