# pages/2_Input.py
import os
import base64
import streamlit as st
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Input | Fake News Detection", layout="wide")

ASSETS_DIR = "assets"
BG_PATH = os.path.join(ASSETS_DIR, "bg_input.jpg")

MODEL_DIR = os.path.join("model", "roberta_fake_news_model")  # change if your folder name differs
MODEL_NAME_DISPLAY = "roberta-base"
MAX_LEN = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Styling Helpers
# -----------------------------
def set_bg(image_path: str) -> None:
    if not os.path.exists(image_path):
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(1100px 500px at 20% 0%, rgba(34,197,94,0.16), transparent 60%),
                            radial-gradient(900px 500px at 80% 10%, rgba(56,189,248,0.16), transparent 60%),
                            #0b1220;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
              linear-gradient(rgba(0,0,0,0.66), rgba(0,0,0,0.66)),
              url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }

        .glass-card {
            background: rgba(10, 18, 32, 0.74);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 14px 46px rgba(0,0,0,0.50);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 26px;
            color: #F8FAFC;
        }
        .title {
            font-size: 38px;
            font-weight: 850;
            margin: 0 0 10px 0;
            color: #F8FAFC;
            text-shadow: 0 2px 14px rgba(0,0,0,0.55);
        }
        .subtitle {
            font-size: 15px;
            line-height: 1.75;
            color: rgba(248,250,252,0.92);
        }
        .small-note {
            font-size: 13px;
            color: rgba(248,250,252,0.78);
            line-height: 1.6;
        }
        .btn-row {
            display:flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg(BG_PATH)
inject_css()

# -----------------------------
# Model Loader (cached)
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer(model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model folder not found: '{model_dir}'. "
            f"Place your saved RoBERTa model inside: {model_dir}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_text(text: str):
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    text = "" if text is None else str(text).strip()
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # Class index: 0=FAKE, 1=REAL
    pred = int(np.argmax(probs))
    return pred, float(probs[0]), float(probs[1])


# -----------------------------
# UI
# -----------------------------
col_left, col_right = st.columns([1.45, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Input Text</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtitle">
        Paste a headline or full news content. Then click <b>Predict</b>.
        </div>
        <div class="small-note">
        Label mapping: FAKE → 0, REAL → 1. Model: RoBERTa-base.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pre-fill from session state
    default_text = st.session_state.get("input_text", "")

    text = st.text_area(
        "News text",
        value=default_text,
        height=260,
        placeholder="Paste news headline or article here..."
    )

    # Simple counters
    words = len(text.split())
    chars = len(text)
    st.caption(f"Length: {words} words | {chars} characters")

    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    with b2:
        example_clicked = st.button("Use Example", use_container_width=True)

    with b3:
        clear_clicked = st.button("Clear", use_container_width=True)

    if example_clicked:
        example = (
            "WASHINGTON (Reuters) - The U.S. administration announced new measures on Monday, "
            "aiming to strengthen policy enforcement and improve transparency, officials said."
        )
        st.session_state["input_text"] = example
        st.session_state["last_error"] = ""
        st.rerun()

    if clear_clicked:
        st.session_state["input_text"] = ""
        st.session_state["pred_label"] = None
        st.session_state["probs"] = None
        st.session_state["last_error"] = ""
        st.rerun()

    if predict_clicked:
        st.session_state["input_text"] = text
        st.session_state["model_name"] = MODEL_NAME_DISPLAY
        st.session_state["last_error"] = ""

        if len(text.strip()) < 20:
            st.session_state["last_error"] = "Please enter at least 20 characters for a reliable prediction."
            st.warning(st.session_state["last_error"])
        else:
            with st.spinner("Running RoBERTa inference..."):
                try:
                    pred, p_fake, p_real = predict_text(text)
                    st.session_state["pred_label"] = pred
                    st.session_state["probs"] = {"FAKE": p_fake, "REAL": p_real}
                    st.success("Prediction saved. Open the **Result** page from the sidebar.")
                except Exception as e:
                    st.session_state["last_error"] = str(e)
                    st.error("Prediction failed. Please verify model path and dependencies.")
                    st.code(str(e))

    if st.session_state.get("last_error"):
        st.warning(st.session_state["last_error"])

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(
        """
        <div class="glass-card">
          <h3 style="margin-top:0; color:#F8FAFC;">Tips for Better Results</h3>
          <ul style="line-height:1.85; color:rgba(248,250,252,0.92);">
            <li>Prefer full article text over very short headlines.</li>
            <li>Avoid random characters or incomplete sentences.</li>
            <li>Try multiple samples and compare confidence scores.</li>
          </ul>

          <div style="height:10px;"></div>

          <h4 style="margin:10px 0 6px 0; color:#F8FAFC;">Interpretation</h4>
          <div class="small-note">
            The probabilities indicate model confidence, not factual certainty. Use credible sources to verify claims.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
