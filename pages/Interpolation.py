import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Interpolation Explorer — multiple methods")

st.markdown(
    "Choose how to provide data (manual points or sample from a function), pick an interpolation method, "
    "and compare the interpolant (and true function if available)."
)

# -----------------------
# Data input mode
# -----------------------
mode = st.radio("Data input mode:", ["Manual Points", "Function-Based"], index=0)

# defaults
monster_fn = "np.sin(x) + np.cos(2*x) + 0.1*x**3"

if mode == "Function-Based":
    st.subheader("Function-based sampling")
    func_str = st.text_input("Enter function of x (NumPy syntax, use 'np'):", value=monster_fn)
    col_a, col_b = st.columns(2)
    with col_a:
        x_start = st.number_input("Interval start (a):", value=-3.0)
    with col_b:
        x_end = st.number_input("Interval end (b):", value=3.0)
    n_points = st.slider("Number of sample points (n):", 2, 20, 6)
    try:
        x_points = np.linspace(x_start, x_end, n_points)
        # evaluate safely
        y_points = eval(func_str, {"np": np, "x": x_points})
        y_points = np.array(y_points, dtype=float)
        use_true_function = True
    except Exception as e:
        st.error(f"Error evaluating function: {e}")
        x_points = np.linspace(-1, 1, 3)
        y_points = np.zeros_like(x_points)
        use_true_function = False

else:
    st.subheader("Manual data points (enter x and y)")
    n_points = st.slider("Number of data points (n):", 2, 12, 4)
    xs = []
    ys = []
    # present inputs in a compact way
    for i in range(n_points):
        c1, c2 = st.columns([1, 1])
        with c1:
            xi = st.number_input(f"x[{i}]", value=float(i), key=f"mx_{i}")
        with c2:
            yi = st.number_input(f"y[{i}]", value=float(i * i), key=f"my_{i}")
        xs.append(xi)
        ys.append(yi)
    x_points = np.array(xs, dtype=float)
    y_points = np.array(ys, dtype=float)
    use_true_function = False
    func_str = None

# -----------------------
# Basic validation & sorting
# -----------------------
# sort by x to avoid issues with interpolators that require increasing x
order = np.argsort(x_points)
x_points = x_points[order]
y_points = y_points[order]

# check duplicates
if len(np.unique(x_points)) != len(x_points):
    st.warning("Duplicate x-values detected. Many interpolation methods require distinct x. "
               "If you want Hermite with repeated nodes, use function-based sampling or provide derivatives.")
    # still proceed but some methods might fail

# -----------------------
# Method selection
# -----------------------
method = st.selectbox(
    "Select interpolation method",
    [
        "Vandermonde (polyfit)",
        "Lagrange",
        "Newton Divided Difference",
        "Hermite (Krogh)",
        "Spline (Linear)",
        "Spline (Cubic)"
    ],
)

# -----------------------
# Implementations
# -----------------------
def vandermonde_eval(x_pts, y_pts, xx):
    # fit polynomial of degree n-1 (may be ill-conditioned for large n)
    coeffs = np.polyfit(x_pts, y_pts, len(x_pts) - 1)  # highest-to-lowest
    return np.polyval(coeffs, xx)


def lagrange_eval(x_pts, y_pts, xx):
    xx = np.asarray(xx)
    def single(x):
        total = 0.0
        n = len(x_pts)
        for i in range(n):
            term = y_pts[i]
            for j in range(n):
                if j != i:
                    term *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
            total += term
        return total
    # vectorize for array xx
    if xx.ndim == 0:
        return single(xx)
    return np.array([single(x) for x in xx])


def newton_coeffs(x_pts, y_pts):
    n = len(x_pts)
    dd = np.zeros((n, n), dtype=float)
    dd[:, 0] = y_pts
    for j in range(1, n):
        for i in range(n - j):
            denom = x_pts[i + j] - x_pts[i]
            if denom == 0:
                dd[i, j] = np.nan
            else:
                dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / denom
    return dd[0, :]


def newton_eval(x_pts, y_pts, xx):
    coef = newton_coeffs(x_pts, y_pts)
    if np.any(np.isnan(coef)):
        raise ValueError("Divided difference computation failed due to duplicate x values.")
    def eval_one(x):
        n = len(coef)
        val = coef[-1]
        for k in range(n - 2, -1, -1):
            val = val * (x - x_pts[k]) + coef[k]
        return val
    xx = np.asarray(xx)
    if xx.ndim == 0:
        return eval_one(xx)
    return np.array([eval_one(x) for x in xx])


# SciPy-based methods (guard import)
scipy_ok = True
try:
    from scipy.interpolate import KroghInterpolator, interp1d, CubicSpline
except Exception as e:
    scipy_ok = False
    krogh_error = str(e)

def hermite_eval(x_pts, y_pts, xx):
    if not scipy_ok:
        raise RuntimeError("scipy is required for Hermite (Krogh) interpolation.")
    interp = KroghInterpolator(x_pts, y_pts)
    return interp(xx)

def spline_eval(x_pts, y_pts, xx, kind="linear"):
    if not scipy_ok:
        raise RuntimeError("scipy is required for spline interpolation.")
    if kind == "linear":
        f = interp1d(x_pts, y_pts, kind="linear", bounds_error=False, fill_value="extrapolate")
    else:
        f = CubicSpline(x_pts, y_pts, extrapolate=True)
    return f(xx)


# -----------------------
# Prepare x_range and compute interpolant
# -----------------------
# pick plotting range
margin = max(1.0, 0.1 * (np.max(x_points) - np.min(x_points) if len(x_points) > 1 else 1.0))
x_range = np.linspace(np.min(x_points) - margin, np.max(x_points) + margin, 800)

error_msg = None
y_interp = None

try:
    if method == "Vandermonde (polyfit)":
        y_interp = vandermonde_eval(x_points, y_points, x_range)
    elif method == "Lagrange":
        y_interp = lagrange_eval(x_points, y_points, x_range)
    elif method == "Newton Divided Difference":
        y_interp = newton_eval(x_points, y_points, x_range)
    elif method == "Hermite (Krogh)":
        if not scipy_ok:
            raise RuntimeError(f"scipy import failed: {krogh_error}")
        y_interp = hermite_eval(x_points, y_points, x_range)
    elif method == "Spline (Linear)":
        if len(x_points) < 2:
            raise ValueError("Need at least 2 points for linear spline.")
        y_interp = spline_eval(x_points, y_points, x_range, kind="linear")
    elif method == "Spline (Cubic)":
        if len(x_points) < 2:
            raise ValueError("Need at least 2 points for cubic spline.")
        y_interp = spline_eval(x_points, y_points, x_range, kind="cubic")
except Exception as e:
    error_msg = str(e)

# -----------------------
# Plotting
# -----------------------
st.subheader("Visualization")

fig = go.Figure()

# If function-based and function is given, plot the true function too
if mode == "Function-Based" and use_true_function:
    try:
        true_f = lambda xx: eval(func_str, {"np": np, "x": xx})
        y_true = true_f(x_range)
        fig.add_trace(
            go.Scatter(x=x_range, y=y_true, mode="lines", name="True function",
                       line=dict(color="green", dash="dot", width=2))
        )
    except Exception:
        # fail silently for plotting true function, interpolation will still show
        pass

if error_msg:
    st.error(f"Interpolation error: {error_msg}")
else:
    fig.add_trace(
        go.Scatter(x=x_range, y=y_interp, mode="lines", name=f"{method} interpolation",
                   line=dict(color="blue", width=2))
    )

# datapoints
fig.add_trace(
    go.Scatter(x=x_points, y=y_points, mode="markers+text", name="Data points",
               marker=dict(color="red", size=8), text=[f"({x:.2f},{y:.2f})" for x, y in zip(x_points, y_points)],
               textposition="top center")
)

fig.update_layout(
    title=f"{method} — Interpolation",
    xaxis_title="x", yaxis_title="f(x)",
    template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# -----------------------
# Evaluate at a user x
# -----------------------
st.subheader("Evaluate interpolant at x")

x_eval = st.number_input("Enter x value to evaluate:", value=float(np.mean(x_points)))
y_eval = None
eval_error = None

try:
    if error_msg:
        raise RuntimeError("Interpolation is not available due to previous error.")
    if method == "Vandermonde (polyfit)":
        y_eval = vandermonde_eval(x_points, y_points, x_eval)
    elif method == "Lagrange":
        y_eval = lagrange_eval(x_points, y_points, x_eval)
    elif method == "Newton Divided Difference":
        y_eval = newton_eval(x_points, y_points, x_eval)
    elif method == "Hermite (Krogh)":
        y_eval = hermite_eval(x_points, y_points, x_eval)
    elif method == "Spline (Linear)":
        y_eval = spline_eval(x_points, y_points, x_eval, kind="linear")
    elif method == "Spline (Cubic)":
        y_eval = spline_eval(x_points, y_points, x_eval, kind="cubic")
except Exception as e:
    eval_error = str(e)

if eval_error:
    st.error(f"Evaluation error: {eval_error}")
else:
    st.success(f"Interpolated value: f({x_eval}) ≈ {float(y_eval):.6f}")
