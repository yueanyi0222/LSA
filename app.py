import streamlit as st
import numpy as np

st.set_page_config(page_title="Dynamic LSA Solver", layout="wide")

st.title("📐 Least Squares Adjustment (LSA) Universal Solver")
st.write("自由设定观测值（Observations）和未知数（Unknowns）数量，程序将自动完成 9 步平差计算：")

# 侧边栏：配置参数
st.sidebar.header("1. 动态设置参数")
n_obs = st.sidebar.number_input("观测值数量 (Number of Observations, N)", min_value=1, max_value=20, value=6)
n_unk = st.sidebar.number_input("未知数数量 (Number of Unknowns, U)", min_value=1, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.header("2. 未知数标签设置")
default_names = ["AB", "BC", "CD", "DE", "EF", "FG", "GH"]
var_names = []
for j in range(n_unk):
    d_name = default_names[j] if j < len(default_names) else f"X{j+1}"
    v_name = st.sidebar.text_input(f"未知数 {j+1} 的名称:", value=d_name, key=f"vname_{j}")
    var_names.append(v_name)

# 预设例题数据
default_A = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 1]
])
default_L = np.array([25.051, 25.047, 25.110, 50.091, 50.150, 75.200])

st.markdown("---")
st.header("📥 输入观测数据与系数 (Data Input)")

A = np.zeros((n_obs, n_unk))
L = np.zeros((n_obs, 1))

cols = st.columns([n_unk + 1, 1])

with cols[0]:
    st.subheader("矩阵 A (设计矩阵系数 Matrix A)")
    for i in range(n_obs):
        row_cols = st.columns(n_unk)
        for j in range(n_unk):
            def_val = float(default_A[i, j]) if (i < 6 and j < 3) else 0.0
            A[i, j] = row_cols[j].number_input(
                f"Obs {i+1} -> {var_names[j]} 系数", 
                value=def_val, 
                key=f"A_{i}_{j}"
            )

with cols[1]:
    st.subheader("向量 L (观测值 Observation Vector)")
    for i in range(n_obs):
        def_l = float(default_L[i]) if i < 6 else 0.0
        L[i, 0] = st.number_input(f"L[{i+1}]", value=def_l, format="%.3f", key=f"L_{i}")

st.markdown("---")
st.header("🧮 9-Step Least Squares Adjustment Result")

# STEP 1
st.subheader("STEP 1 : Model the observation equation")
st.write("平差观测方程 (Linearized Error Equation): V = AX - L")
for i in range(n_obs):
    eq_terms = [f"{A[i, j]:.1f}({var_names[j]})" for j in range(n_unk) if A[i, j] != 0]
    eq_str = " + ".join(eq_terms) if eq_terms else "0"
    st.write(f"{eq_str} = {L[i, 0]:.3f} + V{i+1}")

st.info(f"观测值数量 N = {n_obs} | 未知数数量 U = {n_unk} | 多余观测数 (Redundancy) = {n_obs - n_unk}")

if n_obs < n_unk:
    st.error("⚠️ 错误：观测值数量 (N) 必须大于等于未知数数量 (U) 才能进行平差！")
else:
    # STEP 2
    st.subheader("STEP 2 : Create matrix A, X and L")
    c1, c2 = st.columns(2)
    c1.write("**Matrix A:**")
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
        st.error("⚠️ 错误：A^T * A 的行列式为 0（矩阵奇异），无法求逆！请检查输入的矩阵系数。")
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
        st.subheader("STEP 9 : Solve X = (A^T * A)^-1 * A^T * L")
        X = np.dot(inv_ATA, ATL)

        st.success("🎉 最终平差结果 (Adjusted Variables):")
        for j in range(n_unk):
            st.markdown(f"### **{var_names[j]} = {X[j, 0]:.4f}**")
