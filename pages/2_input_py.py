# pages/2_Input.py
# Page 2: Input page (collect text + run prediction)

import base64
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Input", page_icon="✍️", layout="wide")

def set_background(image_path: str):
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
        pass

set_background("assets/bg_input.png")

st.markdown(
    """
    <style>
      .input-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 26px 26px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
      }
      .muted { color: rgba(0,0,0,0.65); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model loading (cached)
# -----------------------------
MODEL_DIR = "model/roberta_fake_news_model"  # put your saved model folder here
MAX_LEN = 256

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict(text: str):
    """
    Returns:
      pred_label: int (0 or 1)
      probs: [p_fake, p_real]
    """
    tokenizer, model, device = load_model()

    text = "" if text is None else str(text).strip()
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    pred_label = int(torch.argmax(torch.tensor(probs)).item())
    return pred_label, probs

# Session state init
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None  # will store dict

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.title("Input")
st.write("Paste a news headline or article text and click **Predict**.")

example_text = (
    "WASHINGTON (Reuters) - The U.S. House of Representatives voted on a new bill today "
    "after lawmakers debated policy changes and economic impacts."
)

# Input area
text = st.text_area(
    "News text",
    value=st.session_state["input_text"],
    height=220,
    placeholder="Paste your news headline/article here...",
)

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    if st.button("Use Example"):
        st.session_state["input_text"] = example_text
        st.rerun()

with col_b:
    if st.button("Clear"):
        st.session_state["input_text"] = ""
        st.session_state["prediction"] = None
        st.rerun()

with col_c:
    st.markdown('<p class="muted">Label mapping: FAKE = 0, REAL = 1</p>', unsafe_allow_html=True)

# Predict action
st.markdown("---")
if st.button("Predict", type="primary"):
    cleaned = "" if text is None else str(text).strip()

    if len(cleaned) < 20:
        st.warning("Please enter a longer text (at least ~20 characters) for a more reliable prediction.")
    else:
        st.session_state["input_text"] = cleaned

        with st.spinner("Running RoBERTa prediction..."):
            pred_label, probs = predict(cleaned)

        st.session_state["prediction"] = {
            "pred_label": pred_label,
            "p_fake": float(probs[0]),
            "p_real": float(probs[1]),
        }

        # Go to result page if possible
        try:
            st.switch_page("pages/3_Result.py")
        except Exception:
            st.success("Prediction completed. Open the Result page from the sidebar to view details.")

st.markdown("</div>", unsafe_allow_html=True)
