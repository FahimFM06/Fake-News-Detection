import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared session keys across pages
defaults = {
    "input_text": "",
    "pred_label": None,   # 0 FAKE, 1 REAL
    "probs": None,        # {"FAKE": float, "REAL": float}
    "model_id": "fmfahim6/fake-news-roberta",
    "last_error": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("Fake News Detection (RoBERTa)")
st.write("Use the sidebar to open pages: **Introduction → Input → Result**.")
st.info("Model is loaded from Hugging Face Hub: `fmfahim6/fake-news-roberta`")
