# pages/1_Introduction.py
import os
import base64
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Introduction | Fake News Detection", layout="wide")

ASSETS_DIR = "assets"
BG_PATH = os.path.join(ASSETS_DIR, "bg_intro.jpg")

# -----------------------------
# Styling Helpers
# -----------------------------
def set_bg(image_path: str) -> None:
    if not os.path.exists(image_path):
        # Fallback background (no image available)
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(1100px 500px at 20% 0%, rgba(56,189,248,0.22), transparent 60%),
                            radial-gradient(900px 500px at 80% 10%, rgba(168,85,247,0.18), transparent 60%),
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
              linear-gradient(rgba(0,0,0,0.68), rgba(0,0,0,0.68)),
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
        /* Reduce top padding */
        .block-container { padding-top: 1.4rem; }

        .glass-card {
            background: rgba(10, 18, 32, 0.74);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 14px 46px rgba(0,0,0,0.50);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 28px;
            color: #F8FAFC;
        }
        .title {
            font-size: 44px;
            font-weight: 850;
            margin: 0 0 10px 0;
            color: #F8FAFC;
            text-shadow: 0 2px 14px rgba(0,0,0,0.55);
        }
        .subtitle {
            font-size: 17px;
            line-height: 1.75;
            color: rgba(248,250,252,0.92);
        }
        .kpi-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-top: 14px;
        }
        .kpi {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
        }
        .kpi-label {
            font-size: 13px;
            color: rgba(248,250,252,0.85);
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 22px;
            font-weight: 850;
            color: #F8FAFC;
        }
        .small-note {
            font-size: 13px;
            color: rgba(248,250,252,0.80);
            line-height: 1.6;
        }
        .pill {
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(56,189,248,0.14);
            border: 1px solid rgba(56,189,248,0.30);
            font-size: 13px;
            color: rgba(248,250,252,0.95);
            margin-right: 8px;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg(BG_PATH)
inject_css()

# -----------------------------
# Content
# -----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="subtitle">
        This application classifies a news text as <b>FAKE</b> or <b>REAL</b> using a fine-tuned
        <b>RoBERTa-base</b> transformer model.
        <br><br>
        <b>Label Mapping</b><br>
        <span class="pill">FAKE → 0</span>
        <span class="pill">REAL → 1</span>
        <br><br>
        <b>Workflow</b>
        <ol>
          <li>Go to <b>Input</b> page and paste a headline or full news text.</li>
          <li>Click <b>Predict</b> to run the model inference.</li>
          <li>Go to <b>Result</b> page to view prediction and confidence.</li>
        </ol>
        </div>
        <div class="small-note">
        Note: This is a research/demo system. Predictions may be incorrect; do not treat it as verified fact-checking.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="glass-card">
          <h3 style="margin-top:0; color:#F8FAFC;">Model Snapshot</h3>
          <div class="kpi-row">
            <div class="kpi">
              <div class="kpi-label">Model</div>
              <div class="kpi-value">RoBERTa-base</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Task</div>
              <div class="kpi-value">Binary NLP</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Output</div>
              <div class="kpi-value">FAKE / REAL</div>
            </div>
          </div>

          <div style="height:12px;"></div>

          <h4 style="margin:10px 0 6px 0; color:#F8FAFC;">What the Result Shows</h4>
          <ul style="line-height:1.85; color:rgba(248,250,252,0.92); margin-top:6px;">
            <li>Final predicted label (FAKE or REAL)</li>
            <li>Confidence scores (probabilities)</li>
            <li>Optional: XAI (LIME / IG) can be added later</li>
          </ul>

          <div class="small-note">
            For best results, paste complete text rather than only a very short headline.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.info("Use the sidebar to open **Input** and run a prediction.")
