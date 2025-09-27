import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔍 Root Approximation Explorer")

st.markdown("""
Choose a root-finding method and see step-by-step approximations:
- **Bisection**
- **False Position (Regula Falsi)**
- **Newton-Raphson**
- **Secant**
""")

# Function input
func_input = st.text_input("Enter function f(x):", "x**3 - x - 2")
f = lambda x: eval(func_input, {"x": x, "np": np})

a = st.number_input("Interval start (a)", -5.0)
b = st.number_input("Interval end (b)", 5.0)
x0 = st.number_input("Initial guess x₀ (for Newton/Secant)", 1.0)
x1 = st.number_input("Second guess x₁ (for Secant)", 2.0)
tol = st.number_input("Tolerance", 0.001)

# Dropdown to select method
method = st.selectbox("Select Method", ["Bisection", "False Position", "Newton-Raphson", "Secant"])

# --- Methods ---
def bisection(f, a, b, tol):
    steps = []
    if f(a) * f(b) >= 0:
        return None, steps
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        steps.append(c)
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, steps

def false_position(f, a, b, tol):
    steps = []
    if f(a) * f(b) >= 0:
        return None, steps
    while abs(b - a) > tol:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        steps.append(c)
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, steps

def newton_raphson(f, x0, tol, max_iter=50):
    steps = [x0]
    for _ in range(max_iter):
        # Derivative using finite difference
        h = 1e-6
        fprime = (f(x0 + h) - f(x0 - h)) / (2 * h)
        if fprime == 0:
            return None, steps
        x1 = x0 - f(x0) / fprime
        steps.append(x1)
        if abs(x1 - x0) < tol:
            return x1, steps
        x0 = x1
    return x0, steps

def secant(f, x0, x1, tol, max_iter=50):
    steps = [x0, x1]
    for _ in range(max_iter):
        if f(x1) - f(x0) == 0:
            return None, steps
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        steps.append(x2)
        if abs(x2 - x1) < tol:
            return x2, steps
        x0, x1 = x1, x2
    return x2, steps

# --- Run Selected Method ---
if st.button("Run Method"):
    if method == "Bisection":
        root, steps = bisection(f, a, b, tol)
    elif method == "False Position":
        root, steps = false_position(f, a, b, tol)
    elif method == "Newton-Raphson":
        root, steps = newton_raphson(f, x0, tol)
    elif method == "Secant":
        root, steps = secant(f, x0, x1, tol)

    if root is None:
        st.error(f"{method} method failed.")
    else:
        st.success(f"{method} Root found: {root:.6f}")

        # --- Visualization ---
        x = np.linspace(a, b, 400)
        y = [f(val) for val in x]

        fig, ax = plt.subplots()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(x, y, label="f(x)")
        ax.scatter(steps, [f(s) for s in steps], color="red", zorder=5, label="Approximations")
        ax.set_title(f"{method} Method Convergence")
        ax.legend()
        st.pyplot(fig)
