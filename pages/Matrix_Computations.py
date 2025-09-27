import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("🧮 Matrix Computations - Eigenvalues & Eigenvectors")

st.markdown("""
This demo shows how to compute **Eigenvalues** and **Eigenvectors** of a matrix.

For a matrix $A$:

$$A v = \lambda v$$

where $\\lambda$ is the eigenvalue and $v$ is the corresponding eigenvector.
""")

# --- Matrix Input ---
n = st.slider("Matrix size (n × n):", 2, 4, 2)

st.subheader("Enter Matrix A")
A = np.zeros((n, n))

cols = st.columns(n)
for i in range(n):
    for j in range(n):
        A[i, j] = cols[j].number_input(
            f"A[{i+1},{j+1}]",
            value=1.0 if i == j else 0.0,
            key=f"A_{i}_{j}"
        )

# --- Compute Eigenvalues & Eigenvectors ---
if st.button("Compute Eigenvalues & Eigenvectors"):
    try:
        eigvals, eigvecs = np.linalg.eig(A)

        # --- Results Table ---
        st.subheader("📊 Results")
        df = pd.DataFrame({
            "Eigenvalue (λ)": np.round(eigvals, 4),
            "Eigenvector": [np.round(vec, 4) for vec in eigvecs.T]
        })
        st.dataframe(df)

        # --- Visualization (only for 2×2 case) ---
        if n == 2:
            st.subheader("🔎 Visualization (2D only)")

            # Create Plotly figure
            fig = go.Figure()

            # Plot eigenvectors
            fig.add_trace(go.Scatter(
                x=[0, eigvecs[0, 0]], y=[0, eigvecs[1, 0]],
                mode="lines+markers", name=f"Eigenvector 1 (λ={eigvals[0]:.2f})",
                line=dict(color="red", width=3), marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=[0, eigvecs[0, 1]], y=[0, eigvecs[1, 1]],
                mode="lines+markers", name=f"Eigenvector 2 (λ={eigvals[1]:.2f})",
                line=dict(color="blue", width=3), marker=dict(size=8)
            ))

            # Show unit circle + transformed ellipse
            theta = np.linspace(0, 2 * np.pi, 200)
            circle = np.array([np.cos(theta), np.sin(theta)])
            ellipse = A @ circle

            fig.add_trace(go.Scatter(
                x=circle[0], y=circle[1],
                mode="lines", name="Unit Circle", line=dict(color="gray", dash="dot")
            ))

            fig.add_trace(go.Scatter(
                x=ellipse[0], y=ellipse[1],
                mode="lines", name="Transformed Ellipse", line=dict(color="green", width=2)
            ))

            # Layout settings
            fig.update_layout(
                title="Eigenvectors as Directions of Stretching",
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[-3, 3]),
                yaxis=dict(range=[-3, 3]),
                template="plotly_white",
                legend=dict(x=0, y=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- Notes ---
        st.info("""
        **Interpretation:**
        - Eigenvectors show the directions that remain unchanged under transformation.
        - Eigenvalues show how much the vector is stretched/compressed along that direction.
        """)

    except Exception as e:
        st.error(f"Error: {e}")
