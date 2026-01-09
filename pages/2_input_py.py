import os
import base64
import numpy as np
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Input | Fake News Detection", layout="wide")

ASSETS_DIR = "assets"
BG_PATH = os.path.join(ASSETS_DIR, "bg_input.jpg")

MODEL_ID = "fmfahim6/fake-news-roberta"
MAX_LEN = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            font-weight: 900;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg(BG_PATH)
inject_css()

# Cache model + tokenizer so the app does not reload it every time
@st.cache_resource
def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def predict_text(text: str):
    tokenizer, model = load_model_and_tokenizer(MODEL_ID)
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

    pred = int(np.argmax(probs))  # 0=FAKE, 1=REAL
    return pred, float(probs[0]), float(probs[1])

left, right = st.columns([1.45, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Input Text</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="subtitle">
        Paste a headline or full news article text and click <b>Predict</b>.
        </div>
        <div class="small-note">
        Label mapping: FAKE → 0, REAL → 1
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_text = st.session_state.get("input_text", "")
    text = st.text_area(
        "News text",
        value=default_text,
        height=260,
        placeholder="Paste news headline or article here..."
    )

    st.caption(f"Words: {len(text.split())} | Characters: {len(text)}")

    c1, c2, c3 = st.columns(3)

    with c1:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    with c2:
        example_clicked = st.button("Use Example", use_container_width=True)
    with c3:
        clear_clicked = st.button("Clear", use_container_width=True)

    if example_clicked:
        example = (
            "WASHINGTON (Reuters) - Officials announced new measures on Monday to improve transparency "
            "and strengthen enforcement, according to a statement."
        )
        st.session_state["input_text"] = example
        st.session_state["pred_label"] = None
        st.session_state["probs"] = None
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
        st.session_state["model_id"] = MODEL_ID
        st.session_state["last_error"] = ""

        if len(text.strip()) < 20:
            st.warning("Please enter at least 20 characters for a reliable prediction.")
        else:
            with st.spinner("Running RoBERTa inference (Hugging Face Hub)..."):
                try:
                    pred, p_fake, p_real = predict_text(text)
                    st.session_state["pred_label"] = pred
                    st.session_state["probs"] = {"FAKE": p_fake, "REAL": p_real}
                    st.success("Prediction saved. Now open the **Result** page from the sidebar.")
                except Exception as e:
                    st.session_state["last_error"] = str(e)
                    st.error("Prediction failed.")
                    st.code(str(e))

    if st.session_state.get("last_error"):
        st.warning(st.session_state["last_error"])

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="glass-card">
          <h3 style="margin-top:0; color:#F8FAFC;">Guidelines</h3>
          <ul style="line-height:1.85; color:rgba(248,250,252,0.92);">
            <li>Use complete text for best accuracy.</li>
            <li>Short headlines may produce lower confidence.</li>
            <li>Try multiple inputs to test the model behavior.</li>
          </ul>
          <div class="small-note">
            The model outputs probabilities, not verified truth. Always fact-check.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
