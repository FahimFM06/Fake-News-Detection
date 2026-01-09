import os
import base64
import streamlit as st

st.set_page_config(page_title="Introduction | Fake News Detection", layout="wide")

ASSETS_DIR = "assets"
BG_PATH = os.path.join(ASSETS_DIR, "bg_intro.jpg")

def set_bg(image_path: str) -> None:
    if not os.path.exists(image_path):
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
            font-weight: 900;
            margin: 0 0 10px 0;
            color: #F8FAFC;
            text-shadow: 0 2px 14px rgba(0,0,0,0.55);
        }
        .subtitle {
            font-size: 16px;
            line-height: 1.75;
            color: rgba(248,250,252,0.92);
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
        .small-note {
            font-size: 13px;
            color: rgba(248,250,252,0.80);
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg(BG_PATH)
inject_css()

model_id = st.session_state.get("model_id", "fmfahim6/fake-news-roberta")

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="subtitle">
        This application classifies news text as <b>FAKE</b> or <b>REAL</b> using a fine-tuned
        <b>RoBERTa-base</b> transformer model loaded from Hugging Face Hub.
        <br><br>
        <b>Label Mapping</b><br>
        <span class="pill">FAKE → 0</span>
        <span class="pill">REAL → 1</span>
        <br><br>
        <b>How to use</b>
        <ol>
          <li>Go to <b>Input</b> page</li>
          <li>Paste headline or full news article</li>
          <li>Click <b>Predict</b></li>
          <li>Open <b>Result</b> page to see output</li>
        </ol>
        </div>
        <div class="small-note">
        Disclaimer: This is a research/demo system. Always verify claims using trusted sources.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        f"""
        <div class="glass-card">
          <h3 style="margin-top:0; color:#F8FAFC;">Model Info</h3>
          <div class="subtitle">
            <b>Model ID:</b> {model_id}<br>
            <b>Architecture:</b> RoBERTa-base<br>
            <b>Task:</b> Binary text classification<br>
          </div>
          <div style="height:10px;"></div>
          <div class="small-note">
            Tip: Full text typically gives more reliable predictions than very short headlines.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info("Next: Open the **Input** page from the sidebar.")
