# pages/3_Result.py
# Page 3: Result page (show label + probabilities + confidence visuals)

import base64
import streamlit as st

st.set_page_config(page_title="Result", page_icon="✅", layout="wide")

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

set_background("assets/bg_result.png")

st.markdown(
    """
    <style>
      .result-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 18px;
        padding: 26px 26px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
      }
      .big-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
      }
      .muted { color: rgba(0,0,0,0.65); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.title("Result")

pred = st.session_state.get("prediction", None)
text = st.session_state.get("input_text", "")

if not pred:
    st.warning("No prediction found. Please go to the Input page and run a prediction first.")
    if st.button("Go to Input"):
        try:
            st.switch_page("pages/2_Input.py")
        except Exception:
            st.info("Use the sidebar to open: Input")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

pred_label = pred["pred_label"]
p_fake = pred["p_fake"]
p_real = pred["p_real"]

final_label = "FAKE (0)" if pred_label == 0 else "REAL (1)"
confidence = max(p_fake, p_real)

st.markdown(f"<p class='big-label'>Prediction: {final_label}</p>", unsafe_allow_html=True)
st.markdown(f"<p class='muted'>Confidence (max probability): {confidence:.2%}</p>", unsafe_allow_html=True)

# Probability metrics
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("P(FAKE)", f"{p_fake:.4f}")
with col2:
    st.metric("P(REAL)", f"{p_real:.4f}")
with col3:
    st.progress(float(confidence))

st.markdown("---")

with st.expander("Show input text"):
    st.write(text)

# Actions
col_a, col_b = st.columns([1, 4])
with col_a:
    if st.button("Try Another"):
        try:
            st.switch_page("pages/2_Input.py")
        except Exception:
            st.info("Use the sidebar to open: Input")

st.markdown("</div>", unsafe_allow_html=True)
