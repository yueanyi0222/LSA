import streamlit as st
import numpy as np

st.set_page_config(page_title="LSA Best Fit Line Solver", layout="wide")

st.title("📈 LSA Best Fit Line Solver (Traverse Coordinate / Line Fitting)")
st.write("Find the best fit line (y = mx + b) for coordinate points using Least Squares Adjustment (LSA).")

# Sidebar Configuration
st.sidebar.header("1. Points Configuration")
n_pts = st.sidebar.number_input("Number of Coordinate Points (N)", min_value=2, max_value=20, value=4)

# Default data
default_x = [100.000, 86.493, 85.960, 97.881]
default_y = [100.000, 97.617, 111.473, 113.764]

st.markdown("---")
st.header("📥 Input Point Coordinates (x, y)")

x_coords = []
y_coords = []

cols_per_row = 2
for i in range(n_pts):
    if i % cols_per_row == 0:
        cols = st.columns(cols_per_row)
    
    col_idx = i % cols_per_row
    with cols[col_idx]:
        st.subheader(f"Stesen {i+1}")
        def_x = default_x[i] if i < len(default_x) else float(i + 1)
        def_y = default_y[i] if i < len(default_y) else float(i + 1)
        
        px = st.number_input(f"X_{i+1}", value=def_x, format="%.4f", step=0.0001, key=f"px_{i}")
        py = st.number_input(f"Y_{i+1}", value=def_y, format="%.4f", step=0.0001, key=f"py_{i}")
        
        x_coords.append(px)
        y_coords.append(py)

# Build Matrices Automatically based on y = m*x + b
A = np.zeros((n_pts, 2))
L = np.zeros((n_pts, 1))

for i in range(n_pts):
    A[i, 0] = x_coords[i]
    A[i, 1] = 1.0
    L[i, 0] = y_coords[i]

st.markdown("---")
st.header("🧮 LSA Calculation Steps")

# STEP 1
st.subheader("STEP 1 : Form Observation Equations")
st.write("Formula: y_i = m * x_i + b - V_i  ==>  x_i * m + b = y_i + V_i")
st.write("**Observation Equations for each station:**")
for i in range(n_pts):
    st.write(f"{x_coords[i]:.3f} m + b = {y_coords[i]:.3f} + V{i+1}")

# STEP 2
st.subheader("STEP 2 : Construct Matrix A, Vector X and Vector L")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Matrix A [x , 1]:**")
    st.dataframe(A)
with c2:
    st.write("**Vector X [m, b]:**")
    st.write("X = [m, b]^T")
with c3:
    st.write("**Vector L [y]:**")
    st.dataframe(L)

# STEP 3
st.subheader("STEP 3 : Form Normal Equation (A^T * A * X = A^T * L)")
ATA = np.dot(A.T, A)
ATL = np.dot(A.T, L)

c1, c2 = st.columns(2)
with c1:
    st.write("**Matrix A^T * A:**")
    st.dataframe(ATA)
with c2:
    st.write("**Vector A^T * L:**")
    st.dataframe(ATL)

# STEP 4
st.subheader("STEP 4 : Solve X = (A^T * A)^-1 * (A^T * L)")
det_ATA = float(np.linalg.det(ATA))
st.write(f"det(A^T * A) = {det_ATA:.6f}")

if abs(det_ATA) < 1e-9:
    st.error("⚠️ Error: Matrix A^T * A is singular! Points might be vertically aligned.")
else:
    inv_ATA = np.linalg.inv(ATA)
    X = np.dot(inv_ATA, ATL)
    
    m_val = X[0, 0]
    b_val = X[1, 0]
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Inverse Matrix (A^T * A)^-1:**")
        st.dataframe(inv_ATA)
    with c2:
        st.success("🎉 **Solution Vector X:**")
        st.markdown(f"### **m (slope) = {m_val:.8f}**")
        st.markdown(f"### **b (intercept) = {b_val:.8f}**")
        st.write(f"**Best Fit Line Equation:** y = {m_val:.8f} x + {b_val:.8f}")

    # STEP 5
    st.subheader("STEP 5 : Calculate Residuals V = A * X - L")
    V = np.dot(A, X) - L
    
    st.write("**Residual Vector V:**")
    st.dataframe(V)
    
    for i in range(n_pts):
        y_calc = m_val * x_coords[i] + b_val
        st.write(f"Stesen {i+1}: Y_calc = **{y_calc:.8f}**, Residual V{i+1} = **{V[i, 0]:+.8f}**")
