import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("📐 Numerical Differentiation")

st.markdown("""
This demo shows how **Forward, Backward, and Central Difference** formulas approximate
the derivative of a function.
""")

# --- Function Input ---
func_input = st.text_input("Enter function f(x):", "np.sin(x) + x**2")
f = lambda x: eval(func_input, {"x": x, "np": np})

x0 = st.number_input("Point of differentiation (x₀):", value=1.0)
h = st.number_input("Step size (h):", value=0.1)

# --- Difference Formulas ---
forward = (f(x0 + h) - f(x0)) / h
backward = (f(x0) - f(x0 - h)) / h
central = (f(x0 + h) - f(x0 - h)) / (2 * h)

# --- Exact Derivative ---
try:
    from sympy import symbols, diff, sympify, lambdify
    x = symbols("x")
    f_sym = sympify(func_input.replace("np.", ""))
    f_prime = diff(f_sym, x)
    f_prime_func = lambdify(x, f_prime, "numpy")
    exact = f_prime_func(x0)
except Exception:
    exact = None

# --- 1 Row, 2 Columns Layout ---
col1, col2 = st.columns([1, 2])  # left = 1 part, right = 2 parts

with col1:
    st.subheader("📊 Results")
    st.write(f"**Forward Difference:** {forward:.6f}")
    st.write(f"**Backward Difference:** {backward:.6f}")
    st.write(f"**Central Difference:** {central:.6f}")
    if exact is not None:
        st.success(f"Exact Derivative f'({x0}) = {exact:.6f}")

with col2:
    st.subheader("📈 Visualization")

    X = np.linspace(x0 - 2, x0 + 2, 400)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(5, 3))  # compact size
    ax.plot(X, Y, label="f(x)", color="blue")

    # Mark x0
    ax.scatter([x0], [f(x0)], color="red", zorder=5, label="x₀")

    # Tangent line (Central Difference)
    m = central
    tangent_line = m * (X - x0) + f(x0)
    ax.plot(X, tangent_line, "--", label="Tangent (Central Diff)", color="orange")

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Function & Tangent", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)
