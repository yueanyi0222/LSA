import streamlit as st
import numpy as np

st.set_page_config(page_title="LSA Solver", layout="wide")

st.title("📐 Least Squares Adjustment (LSA) Step-by-Step Solver")
st.write("This interactive app demonstrates the 9-step mathematical procedures for LSA in Surveying Engineering.")

# 侧边栏：数据输入
st.sidebar.header("1. Input Data Configuration")
n_obs = st.sidebar.number_input("Number of Observations (N)", min_value=2, max_value=10, value=4)
n_unk = st.sidebar.number_input("Number of Unknowns (U)", min_value=1, max_value=5, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Matrix A (Design Matrix) & Vector L")

default_A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
default_L = np.array([1.2, 1.9, 3.1, 3.8])

A = np.zeros((n_obs, n_unk))
L = np.zeros((n_obs, 1))

cols = st.columns(2)

with cols[0]:
    st.subheader("Matrix A")
    for i in range(n_obs):
        row_cols = st.columns(n_unk)
        for j in range(n_unk):
            val = default_A[i, j] if i < 4 and j < 2 else 0.0
            A[i, j] = row_cols[j].number_input(f"A[{i+1},{j+1}]", value=val, key=f"A_{i}_{j}")

with cols[1]:
    st.subheader("Vector L")
    for i in range(n_obs):
        val = default_L[i] if i < 4 else 0.0
        L[i, 0] = st.number_input(f"L[{i+1}]", value=val, key=f"L_{i}")

st.markdown("---")
st.header("🧮 Step-by-Step Calculation Results")

# Step 1: Model
st.subheader("Step 1: Model the Observation Equation")
st.latex(r"L + V = AX \quad \text{or} \quad V = AX - L")
st.info(f"Observations (N): {n_obs} | Unknowns (U): {n_unk} | Redundancy: {n_obs - n_unk}")

# Step 2: Create matrices
st.subheader("Step 2: Create Matrix A, X, and L")
c1, c2 = st.columns(2)
c1.write("**Matrix A:**")
c1.dataframe(A)
c2.write("**Vector L:**")
c2.dataframe(L)

# Step 3: Find ATA
st.subheader("Step 3: Find Matrix A^T * A")
ATA = np.dot(A.T, A)
st.dataframe(ATA)

# Step 4: Determinant
st.subheader("Step 4: Find Determinant for A^T * A")
det_ATA = float(np.linalg.det(ATA))
st.write(f"det(A^T * A) = {det_ATA:.6f}")

if abs(det_ATA) < 1e-9:
    st.error("Error: Matrix A^T A is singular or near-singular! Unable to compute inverse.")
else:
    # Step 5: Minor matrix
    st.subheader("Step 5: Find Minor Matrix for A^T * A")
    u_size = ATA.shape[0]
    minor_ATA = np.zeros((u_size, u_size))
    
    for i in range(u_size):
        for j in range(u_size):
            sub = np.delete(np.delete(ATA, i, axis=0), j, axis=1)
            if sub.size == 1:
                minor_ATA[i, j] = sub[0, 0]
            elif sub.size == 0:
                minor_ATA[i, j] = 1.0
            else:
                minor_ATA[i, j] = np.linalg.det(sub)
                
    st.dataframe(minor_ATA)

    # Step 6: Adjoint matrix
    st.subheader("Step 6: Adjoint Matrix of A^T * A")
    cofactor_matrix = np.zeros((u_size, u_size))
    for i in range(u_size):
        for j in range(u_size):
            cofactor_matrix[i, j] = ((-1) ** (i + j)) * minor_ATA[i, j]
    adj_ATA = cofactor_matrix.T
    st.dataframe(adj_ATA)

    # Step 7: Inverse matrix
    st.subheader("Step 7: Inverse Matrix (A^T * A)^-1")
    inv_ATA = adj_ATA / det_ATA
    st.dataframe(inv_ATA)

    # Step 8: Find ATL
    st.subheader("Step 8: Find A^T * L")
    ATL = np.dot(A.T, L)
    st.dataframe(ATL)

    # Step 9: Solve X
    st.subheader("Step 9: Solve X = (A^T * A)^-1 * A^T * L")
    X = np.dot(inv_ATA, ATL)
    
    st.success("🎉 Solution Vector X:")
    for i in range(n_unk):
        st.write(f"**X[{i+1}] = {X[i, 0]:.6f}**")
