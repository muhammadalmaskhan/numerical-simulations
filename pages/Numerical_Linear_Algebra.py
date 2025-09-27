import streamlit as st
import numpy as np
import pandas as pd

st.title("🔢 Numerical Linear Algebra - Gaussian Elimination")

st.markdown("""
This demo shows how **Gaussian Elimination** works step by step to solve a system of linear equations:

**Ax = b**
""")

# --- Matrix Input ---
n = st.slider("Number of variables (n):", 2, 5, 3)

st.subheader("Enter Coefficient Matrix (A) and RHS Vector (b)")

# Input A matrix
A = np.zeros((n, n))
b = np.zeros((n, 1))

cols = st.columns(n+1)
for i in range(n):
    for j in range(n):
        A[i, j] = cols[j].number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0, key=f"A_{i}_{j}")
    b[i, 0] = cols[-1].number_input(f"b[{i+1}]", value=1.0 if i == 0 else 0.0, key=f"b_{i}")

# --- Gaussian Elimination Function ---
def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    steps = []

    # Forward elimination
    for i in range(n):
        # Pivot (avoid division by zero)
        if A[i, i] == 0:
            for k in range(i+1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    b[[i, k]] = b[[k, i]]
                    break

        # Normalize pivot row
        pivot = A[i, i]
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot

        # Eliminate below
        for j in range(i+1, n):
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

        steps.append((A.copy(), b.copy()))

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
    return x, steps

# --- Run ---
if st.button("Solve System"):
    try:
        solution, steps = gaussian_elimination(A, b)
        
        st.subheader("Step-by-Step Row Reduction")
        for k, (Ak, bk) in enumerate(steps):
            st.markdown(f"**Step {k+1}:**")
            df = pd.DataFrame(np.hstack((Ak, bk)), 
                              columns=[f"x{j+1}" for j in range(n)] + ["b"])
            st.dataframe(df.style.format("{:.3f}"))
        
        st.success(f"Solution: x = {solution}")
    except Exception as e:
        st.error(f"Error: {e}")
