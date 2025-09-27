import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Numerical Methods Lab", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 42px !important;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #ff6a00, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: gray;
            margin-bottom: 30px;
        }
        .card {
            background-color: #f9f9f9;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
        }
        .card:hover {
            transform: scale(1.05);
            background-color: #f0f4ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.markdown('<p class="title">📊 Numerical Methods Simulation Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Click a module below or use the sidebar to begin exploring</p>', unsafe_allow_html=True)

st.write("---")

# --- Cards as Navigation ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔢 Root Approximation", use_container_width=True):
        st.switch_page("pages/Root_Approximation.py")

with col2:
    if st.button("📈 Interpolation", use_container_width=True):
        st.switch_page("pages/Interpolation.py")  # ✅ match your filename

with col3:
    if st.button("∫ Numerical Integration", use_container_width=True):
        st.switch_page("pages/Numerical_Integration.py")

col4, col5 = st.columns(2)

with col4:
    if st.button("📐 Differentiation", use_container_width=True):
        st.switch_page("pages/Numerical_Differentiation.py")

with col5:
    if st.button("🧮 Matrix Computations", use_container_width=True):
        st.switch_page("pages/Matrix_Computations.py")

st.write("---")

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px; margin-top: 20px;">
        Made with ❤️ using Streamlit | © 2025 Numerical Methods Lab
    </div>
    """,
    unsafe_allow_html=True,
)
