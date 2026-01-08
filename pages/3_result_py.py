# pages/3_Result.py
import os
import base64
import streamlit as st
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Result | Fake News Detection", layout="wide")

ASSETS_DIR = "assets"
BG_PATH = os.path.join(ASSETS_DIR, "bg_result.jpg")

# -----------------------------
# Styling Helpers
# -----------------------------
def set_bg(image_path: str) -> None:
    if not os.path.exists(image_path):
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(1100px 500px at 20% 0%, rgba(56,189,248,0.18), transparent 60%),
                            radial-gradient(900px 500px at 80% 10%, rgba(244,63,94,0.16), transparent 60%),
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
        .badge {
            display:inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 16px;
            border: 1px solid rgba(255,255,255,0.16);
            margin: 10px 0 10px 0;
        }
        .badge-fake {
            background: rgba(244,63,94,0.16);
            border-color: rgba(244,63,94,0.30);
        }
        .badge-real {
            background: rgba(34,197,94,0.16);
            border-color: rgba(34,197,94,0.30);
        }

        .prob-wrap {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-top: 8px;
        }
        .prob-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
        }
        .prob-label {
            font-size: 14px;
            color: rgba(248,250,252,0.85);
            margin-bottom: 6px;
        }
        .prob-value {
            font-size: 26px;
            font-weight: 900;
            color: #F8FAFC;
            margin-bottom: 10px;
        }
        .prob-bar-bg {
            height: 10px;
            background: rgba(255,255,255,0.10);
            border-radius: 999px;
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%;
            width: 0%;
            background: rgba(56,189,248,0.95);
            border-radius: 999px;
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


def render_probs(probs: dict) -> None:
    p_fake = float(np.clip(probs.get("FAKE", 0.0), 0.0, 1.0))
    p_real = float(np.clip(probs.get("REAL", 0.0), 0.0, 1.0))

    html = f"""
    <div class="prob-wrap">
      <div class="prob-card">
        <div class="prob-label">FAKE (0)</div>
        <div class="prob-value">{p_fake*100:.2f}%</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p_fake*100:.2f}%"></div></div>
      </div>

      <div class="prob-card">
        <div class="prob-label">REAL (1)</div>
        <div class="prob-value">{p_real*100:.2f}%</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p_real*100:.2f}%"></div></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


set_bg(BG_PATH)
inject_css()

# -----------------------------
# Read state
# -----------------------------
text = st.session_state.get("input_text", "")
pred_label = st.session_state.get("pred_label", None)
probs = st.session_state.get("probs", None)
model_name = st.session_state.get("model_name", "roberta-base")

# -----------------------------
# UI
# -----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Result</div>', unsafe_allow_html=True)

    if pred_label is None or probs is None or len(str(text).strip()) == 0:
        st.warning("No prediction found. Please go to the **Input** page and click **Predict** first.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Determine final label and confidence
        if int(pred_label) == 0:
            label = "FAKE"
            badge_class = "badge badge-fake"
            confidence = probs.get("FAKE", 0.0)
        else:
            label = "REAL"
            badge_class = "badge badge-real"
            confidence = probs.get("REAL", 0.0)

        st.markdown(
            f"""
            <div class="{badge_class}">
              Prediction: {label} &nbsp; | &nbsp; Confidence: {confidence*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="small-note">
            Model: <b>{model_name}</b> &nbsp; | &nbsp; Label mapping: FAKE → 0, REAL → 1
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("Probabilities")
        render_probs(probs)

        st.write("")
        with st.expander("Show input text"):
            st.write(text)

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="glass-card">
          <h3 style="margin-top:0; color:#F8FAFC;">How to Read This Output</h3>
          <ul style="line-height:1.85; color:rgba(248,250,252,0.92);">
            <li><b>Prediction</b> is the class with higher probability.</li>
            <li><b>Confidence</b> is the model probability for the predicted class.</li>
            <li>Probabilities reflect the model’s belief, not verified truth.</li>
          </ul>
          <div class="small-note">
            If you want, you can add Explainable AI (LIME / Integrated Gradients) under this page later.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
