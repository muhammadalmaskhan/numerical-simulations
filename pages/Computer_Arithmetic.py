import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("💻 Computer Arithmetic and Sources of Error")

st.markdown("""
This demo shows how **rounding and truncation** introduce errors in numerical computations.
""")

# --- User Inputs ---
number = st.number_input("Enter a floating-point number:", value=3.141592653589793)
decimals = st.slider("Select decimal places to keep:", 1, 10, 4)

# Rounding and truncation
rounded_val = round(number, decimals)
truncated_val = float(str(number)[:str(number).find(".")+decimals+1])

# Errors
round_error = abs(number - rounded_val)
trunc_error = abs(number - truncated_val)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔢 Values")
    st.write(f"**Original value:** {number}")
    st.write(f"**Rounded value:** {rounded_val} → Error = {round_error:.2e}")
    st.write(f"**Truncated value:** {truncated_val} → Error = {trunc_error:.2e}")

with col2:
    st.markdown("### 📘 Concept")
    st.latex(r"\text{Error} = |x - \tilde{x}|")
    st.markdown("Where $x$ is the true value and $\\tilde{x}$ is the approximation.")

# --- Error propagation demo ---
st.subheader("🔎 Error Propagation in a Function")

func_input = st.text_input("Enter function f(x):", "np.sin(x) + np.log(x+1)")
f = lambda x: eval(func_input, {"x": x, "np": np})

x_val = st.slider("Choose x value:", 0.1, 10.0, 2.0)

exact = f(x_val)
approx_rounded = f(rounded_val)
approx_truncated = f(truncated_val)

st.markdown("### Function Evaluation")
st.write(f"**f({x_val:.2f}) exact = {exact:.6f}**")
st.write(f"Using rounded input → {approx_rounded:.6f}, error = {abs(exact - approx_rounded):.2e}")
st.write(f"Using truncated input → {approx_truncated:.6f}, error = {abs(exact - approx_truncated):.2e}")

# --- Interactive Plot: error effect over range ---
x = np.linspace(0.1, 10, 200)
y_exact = [f(val) for val in x]
y_rounded = [f(round(val, decimals)) for val in x]
y_truncated = [f(float(str(val)[:str(val).find('.')+decimals+1])) for val in x]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_exact, mode="lines", name="Exact f(x)", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=x, y=y_rounded, mode="lines", name="Rounded input", line=dict(dash="dash", color="red")))
fig.add_trace(go.Scatter(x=x, y=y_truncated, mode="lines", name="Truncated input", line=dict(dash="dot", color="green")))

fig.update_layout(
    title="Effect of Rounding and Truncation on f(x)",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
