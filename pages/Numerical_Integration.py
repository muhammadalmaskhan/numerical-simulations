import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("∫ Numerical Integration Explorer")

st.markdown("""
This demo compares several **Numerical Integration Methods**:

- **Simple Rules:** Left Riemann, Right Riemann, Midpoint, Trapezoidal, Simpson
- **Composite Rules:** Trapezoidal, Simpson

Visualizations show how each method approximates the integral.
""")

# --- Function Input ---
func_input = st.text_input("Enter function f(x):", "np.sin(x) + x**2")
f = lambda x: eval(func_input, {"x": x, "np": np})

a = st.number_input("Lower limit (a):", value=0.0)
b = st.number_input("Upper limit (b):", value=5.0)
n = st.slider("Number of subintervals (n, even for Simpson):", 2, 50, 6, step=2)

# --- Methods ---
def left_riemann(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b-h, n)
    return h * np.sum(f(x))

def right_riemann(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a+h, b, n)
    return h * np.sum(f(x))

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    midpoints = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(midpoints))

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (h/2) * (y[0] + 2*sum(y[1:-1]) + y[-1])

def simpson(f, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (h/3) * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-2:2]) + y[-1])

# --- Run Methods ---
results = {
    "Left Riemann": left_riemann(f, a, b, n),
    "Right Riemann": right_riemann(f, a, b, n),
    "Midpoint": midpoint_rule(f, a, b, n),
    "Trapezoidal": trapezoidal(f, a, b, n),
    "Simpson": simpson(f, a, b, n),
}

# --- Display Results ---
st.subheader("📊 Results")
for method, val in results.items():
    st.write(f"**{method} Rule:** {val:.6f}")

# --- Visualization ---
st.subheader("🔎 Visualizations")

x = np.linspace(a, b, 400)
y = f(x)

# Create two columns for plots
col1, col2 = st.columns(2)

# Composite Trapezoidal Visualization
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, y, "b", label="f(x)")
    X = np.linspace(a, b, n+1)
    Y = f(X)
    for i in range(n):
        xs = [X[i], X[i], X[i+1], X[i+1]]
        ys = [0, Y[i], Y[i+1], 0]
        ax.fill(xs, ys, color="orange", alpha=0.3)
    ax.set_title("Composite Trapezoidal Rule")
    ax.legend()
    st.pyplot(fig)

# Composite Simpson Visualization
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, y, "b", label="f(x)")
    X = np.linspace(a, b, n+1)
    for i in range(0, n, 2):
        xs = np.linspace(X[i], X[i+2], 100)
        coeffs = np.polyfit([X[i], X[i+1], X[i+2]],
                            [f(X[i]), f(X[i+1]), f(X[i+2])], 2)
        ys = np.polyval(coeffs, xs)
        ax.fill_between(xs, ys, alpha=0.3, color="green")
    ax.set_title("Composite Simpson's Rule")
    ax.legend()
    st.pyplot(fig)
