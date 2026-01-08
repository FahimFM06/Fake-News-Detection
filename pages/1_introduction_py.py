# pages/1_Introduction.py
# Page 1: Introduction

import base64
import streamlit as st

st.set_page_config(page_title="Introduction", page_icon="📘", layout="wide")

def set_background(image_path: str):
    """
    Sets a full-page background image using CSS.
    Put your background images inside: assets/
    Example: assets/bg_intro.png
    """
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
              .stApp {{
                background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
              }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        # If you don't add background images, the page will still work.
        pass

# Optional background (create and place it in assets/)
set_background("assets/bg_intro.png")

st.markdown(
    """
    <style>
      .intro-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 26px 26px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
      }
      .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(255,255,255,0.75);
        font-size: 0.9rem;
        margin-right: 8px;
      }
      .muted { color: rgba(0,0,0,0.65); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="intro-card">', unsafe_allow_html=True)

st.title("Introduction")
st.write(
    "This project detects **Fake News** using an advanced NLP model fine-tuned on a labeled dataset."
)

st.markdown(
    """
    <span class="badge">Model: RoBERTa-base</span>
    <span class="badge">Task: Binary Classification</span>
    <span class="badge">Labels: FAKE=0, REAL=1</span>
    """,
    unsafe_allow_html=True,
)

st.subheader("Project goal")
st.write(
    "Given an input news headline or full article text, the model predicts whether it is **FAKE** or **REAL**, "
    "and provides confidence scores for each class."
)

st.subheader("How it works")
st.markdown(
    """
    1. **User enters text** (headline or article).
    2. Text is **tokenized** with the RoBERTa tokenizer.
    3. The fine-tuned **RoBERTa classifier** produces logits.
    4. Logits are converted to **probabilities** for FAKE/REAL.
    5. The app displays the **final label + confidence**.
    """
)

st.subheader("Dataset and evaluation (example)")
st.markdown(
    """
    - Dataset size: ~47K samples  
    - Balanced enough for training (REAL slightly higher than FAKE)  
    - Strong baseline: TF-IDF + Logistic Regression  
    - Final model: RoBERTa-base (best among tested transformer models)
    """
)

st.markdown('<p class="muted">Go to the Input page to test the model with your own text.</p>', unsafe_allow_html=True)

# Navigation button (works if Streamlit supports switch_page)
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Go to Input"):
        try:
            st.switch_page("pages/2_Input.py")
        except Exception:
            st.info("Use the sidebar to open: Input")

st.markdown("</div>", unsafe_allow_html=True)
