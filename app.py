import streamlit as st
import numpy as np

st.set_page_config(page_title="Auto LSA Distance Baseline Solver", layout="wide")

st.title("📏 Auto-Generated LSA Distance Baseline Solver")
st.write("Select the segments for each measurement and enter the observed distance. **Matrix A will be generated automatically!**")

# Sidebar Configuration
st.sidebar.header("1. Unknown Segments Configuration")
n_unk = st.sidebar.number_input("Number of Unknown Segments (U)", min_value=1, max_value=6, value=3)

# Default segment names: AB, BC, CD...
default_segments = ["AB", "BC", "CD", "DE", "EF", "FG"]
unk_names = []
for j in range(n_unk):
    d_name = default_segments[j] if j < len(default_segments) else f"Seg_{j+1}"
    name = st.sidebar.text_input(f"Segment {j+1} Name:", value=d_name, key=f"unk_{j}")
    unk_names.append(name)

st.sidebar.markdown("---")
st.sidebar.header("2. Observations Configuration")
n_obs = st.sidebar.number_input("Total Number of Observations (N)", min_value=1, max_value=15, value=6)

# Default baseline combinations (AB, BC, CD, AC, BD, AD)
default_combos = [
    [0],        # AB
    [1],        # BC
    [2],        # CD
    [0, 1],     # AC = AB + BC
    [1, 2],     # BD = BC + CD
    [0, 1, 2]   # AD = AB + BC + CD
]
default_L_vals = [25.051, 25.047, 25.110, 50.091, 50.150, 75.200]

st.markdown("---")
st.header("📥 Observation Data Input")

A = np.zeros((n_obs, n_unk))
L = np.zeros((n_obs, 1))

# Dynamic UI construction
for i in range(n_obs):
    st.subheader(f"Observation {i+1}")
    c1, c2 = st.columns([3, 1])
    
    def_selected = default_combos[i] if i < len(default_combos) else [0]
    def_selected_names = [unk_names[idx] for idx in def_selected if idx < n_unk]
    
    with c1:
        selected = st.multiselect(
            f"Select segments included in Observation {i+1}:",
            options=unk_names,
            default=def_selected_names,
            key=f"ms_{i}"
        )
        # Populate Matrix A automatically (1 if selected, 0 if not)
        for j, name in enumerate(unk_names):
            if name in selected:
                A[i, j] = 1.0
                
    with c2:
        def_l = default_L_vals[i] if i < len(default_L_vals) else 0.0
        L[i, 0] = st.number_input(f"Distance (m)", value=def_l, format="%.3f", key=f"L_{i}")

st.markdown("---")
st.header("🧮 9-Step Least Squares Adjustment Results")

# STEP 1
st.subheader("STEP 1 : Model the observation equation")
for i in range(n_obs):
    included = [unk_names[j] for j in range(n_unk) if A[i, j] == 1]
    eq_str = " + ".join(included) if included else "0"
    st.write(f"{eq_str} = {L[i, 0]:.3f} + V{i+1}")

st.info(f"Number of observations, n = {n_obs} | Variables, U = {n_unk} | Redundancy = {n_obs - n_unk}")

if n_obs < n_unk:
    st.error("⚠️ Error: Number of observations (N) cannot be less than number of unknowns (U)!")
else:
    # STEP 2
    st.subheader("STEP 2 : Create matrix A, X and L (Matrix A is automatically generated!)")
    c1, c2 = st.columns(2)
    c1.write("**Automatically Generated Matrix A:**")
    c1.dataframe(A)
    c2.write("**Vector L:**")
    c2.dataframe(L)

    # STEP 3
    st.subheader("STEP 3 : Find matrix A^T * A")
    ATA = np.dot(A.T, A)
    st.dataframe(ATA)

    # STEP 4
    st.subheader("STEP 4 : Find Determinant for A^T * A")
    det_ATA = float(np.linalg.det(ATA))
    st.write(f"det(A^T * A) = {det_ATA:.4f}")

    if abs(det_ATA) < 1e-9:
        st.error("⚠️ Error: Matrix A^T * A is singular! Please check if all segments are measured.")
    else:
        # STEP 5
        st.subheader("STEP 5 : Find minor matrix A^T * A")
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

        # STEP 6
        st.subheader("STEP 6 : Adjoint matrix A^T * A")
        cofactor_matrix = np.zeros((u_size, u_size))
        for i in range(u_size):
            for j in range(u_size):
                cofactor_matrix[i, j] = ((-1) ** (i + j)) * minor_ATA[i, j]
        adj_ATA = cofactor_matrix.T
        st.dataframe(adj_ATA)

        # STEP 7
        st.subheader("STEP 7 : Inverse matrix (A^T * A)^-1")
        inv_ATA = adj_ATA / det_ATA
        st.dataframe(inv_ATA)

        # STEP 8
        st.subheader("STEP 8 : Find A^T * L")
        ATL = np.dot(A.T, L)
        st.dataframe(ATL)

        # STEP 9
        st.subheader("STEP 9 : Solve X = (A^T A)^-1 * A^T L")
        X = np.dot(inv_ATA, ATL)

        st.success("🎉 Adjusted Variables Result (X):")
        for j in range(n_unk):
            st.markdown(f"### **{unk_names[j]} = {X[j, 0]:.4f} m**")
