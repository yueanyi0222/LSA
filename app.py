import streamlit as st
import numpy as np

st.set_page_config(page_title="Auto LSA Distance Baseline Solver", layout="wide")

st.title("📏 Auto-Generated LSA Distance Baseline Solver")
st.write("只需选择测距路径和输入距离，系统将**自动生成矩阵 A** 并完成 9 步平差计算！")

# 1. 侧边栏配置未知数（相邻基线）
st.sidebar.header("1. 基础未知段设置")
n_unk = st.sidebar.number_input("基础未知段数量 (U)", min_value=1, max_value=6, value=3)

# 默认段名：AB, BC, CD...
default_segments = ["AB", "BC", "CD", "DE", "EF", "FG"]
unk_names = []
for j in range(n_unk):
    d_name = default_segments[j] if j < len(default_segments) else f"Seg_{j+1}"
    name = st.sidebar.text_input(f"第 {j+1} 未知段名称:", value=d_name, key=f"unk_{j}")
    unk_names.append(name)

st.sidebar.markdown("---")
st.sidebar.header("2. 观测值数量设置")
n_obs = st.sidebar.number_input("总观测距离数量 (N)", min_value=1, max_value=15, value=6)

# 默认的经典例题组合 (AB, BC, CD, AC, BD, AD)
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
st.header("📥 观测数据录入（勾选测距包含的线段）")

A = np.zeros((n_obs, n_unk))
L = np.zeros((n_obs, 1))

# 动态构建输入界面
for i in range(n_obs):
    st.subheader(f"观测值 {i+1} (Observation {i+1})")
    c1, c2 = st.columns([3, 1])
    
    # 默认选中逻辑
    def_selected = default_combos[i] if i < len(default_combos) else [0]
    def_selected_names = [unk_names[idx] for idx in def_selected if idx < n_unk]
    
    with c1:
        # 用户直接多选：这条测距包含哪些基本线段？
        selected = st.multiselect(
            f"Obs {i+1} 包含哪些段？",
            options=unk_names,
            default=def_selected_names,
            key=f"ms_{i}"
        )
        # 根据用户的选择，自动给 Matrix A 赋值 1 或 0
        for j, name in enumerate(unk_names):
            if name in selected:
                A[i, j] = 1.0
                
    with c2:
        def_l = default_L_vals[i] if i < len(default_L_vals) else 0.0
        L[i, 0] = st.number_input(f"测得距离 (m)", value=def_l, format="%.3f", key=f"L_{i}")

st.markdown("---")
st.header("🧮 9-Step Least Squares Adjustment Results")

# STEP 1
st.subheader("STEP 1 : Model the observation equation")
for i in range(n_obs):
    included = [unk_names[j] for j in range(n_unk) if A[i, j] == 1]
    eq_str = " + ".join(included) if included else "0"
    st.write(f"{eq_str} = {L[i, 0]:.3f} + V{i+1}")

st.info(f"观测数 N = {n_obs} | 未知数 U = {n_unk} | 多余观测 (Redundancy) = {n_obs - n_unk}")

if n_obs < n_unk:
    st.error("⚠️ 错误：观测数量 (N) 不能小于未知数数量 (U)！")
else:
    # STEP 2
    st.subheader("STEP 2 : Create matrix A, X and L (Matrix A 是自动算出的！)")
    c1, c2 = st.columns(2)
    c1.write("**自动生成的 Matrix A:**")
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
        st.error("⚠️ 错误：Matrix A^T * A 的行列式为 0（矩阵奇异），请检查是否所有未知段都被测量到了。")
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

        st.success("🎉 最终平差结果 (Adjusted Variables):")
        for j in range(n_unk):
            st.markdown(f"### **{unk_names[j]} = {X[j, 0]:.4f} m**")
