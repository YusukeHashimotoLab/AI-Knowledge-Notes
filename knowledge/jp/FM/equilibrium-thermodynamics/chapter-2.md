---
title: "第2章: Maxwell関係式と熱力学的関係"
chapter_title: "第2章: Maxwell関係式と熱力学的関係"
---

[基礎数理道場](<../index.html>) > [平衡熱力学と相転移](<index.html>) > 第2章 

## 🎯 学習目標

  * Maxwellの4つの基本関係式を導出できる
  * 熱力学ポテンシャルの全微分から関係式を抽出できる
  * Jacobian法を用いて熱力学的恒等式を導出できる
  * 比熱（Cp, Cv）の関係式を理解する
  * 圧縮率と膨張係数の定義と物理的意味を説明できる
  * Maxwellの関係式を用いて実験データから物性値を計算できる
  * 熱力学的安定性条件を理解する

## 📖 Maxwell関係式とは

### Maxwell関係式の物理的意味

**Maxwell関係式** は、熱力学ポテンシャルの全微分から導かれる、一見無関係に見える物理量間の関係式です。

**なぜ重要か？**

  * 直接測定が困難な量を、測定可能な量から計算できる
  * 実験データの一貫性チェックに使える
  * 状態方程式から様々な物性値を導出できる
  * 熱力学的安定性の判定に必須

### 4つの熱力学ポテンシャル

Maxwell関係式を導出する前に、4つの基本的な熱力学ポテンシャルを復習します：

**1\. 内部エネルギー U(S, V, N)**

\\[ dU = TdS - PdV + \mu dN \\]

**2\. Helmholtz自由エネルギー F(T, V, N)**

\\[ F = U - TS, \quad dF = -SdT - PdV + \mu dN \\]

**3\. エンタルピー H(S, P, N)**

\\[ H = U + PV, \quad dH = TdS + VdP + \mu dN \\]

**4\. Gibbs自由エネルギー G(T, P, N)**

\\[ G = U - TS + PV, \quad dG = -SdT + VdP + \mu dN \\]

## 💻 例題2.1: Maxwell関係式の導出（SymPy）

### Maxwell関係式の導出原理

全微分 \\(df = \left(\frac{\partial f}{\partial x}\right)_y dx + \left(\frac{\partial f}{\partial y}\right)_x dy\\) において、

\\[ \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} \\]

が成り立つことから、Maxwell関係式が導かれます。

Python実装: Maxwell関係式の記号的導出

import sympy as sp import numpy as np import matplotlib.pyplot as plt # 記号の定義 T, S, P, V, mu, N = sp.symbols('T S P V mu N', real=True, positive=True) # 4つの熱力学ポテンシャル U = sp.Function('U')(S, V, N) F = sp.Function('F')(T, V, N) H = sp.Function('H')(S, P, N) G = sp.Function('G')(T, P, N) # Maxwell関係式の導出関数 def derive_maxwell_relation(potential, var1, var2, potential_name): """Maxwell関係式を導出""" # 1次偏微分 first_deriv_1 = sp.diff(potential, var1) first_deriv_2 = sp.diff(potential, var2) # 2次偏微分（順序を変える） second_deriv_12 = sp.diff(first_deriv_1, var2) second_deriv_21 = sp.diff(first_deriv_2, var1) # Maxwell関係式 maxwell_eq = sp.Eq(second_deriv_12, second_deriv_21) return maxwell_eq # 4つのMaxwell関係式を導出（粒子数一定の場合） print("=== Maxwell関係式の導出 ===\n") # 1. 内部エネルギー U(S, V)から print("1. 内部エネルギー U(S, V):") print(" dU = T dS - P dV") print(" ∂T/∂V|_S = ∂(-P)/∂S|_V") print(" Maxwell関係式: (∂T/∂V)_S = -(∂P/∂S)_V") print() # 2. Helmholtz自由エネルギー F(T, V)から print("2. Helmholtz自由エネルギー F(T, V):") print(" dF = -S dT - P dV") print(" ∂(-S)/∂V|_T = ∂(-P)/∂T|_V") print(" Maxwell関係式: (∂S/∂V)_T = (∂P/∂T)_V") print() # 3. エンタルピー H(S, P)から print("3. エンタルピー H(S, P):") print(" dH = T dS + V dP") print(" ∂T/∂P|_S = ∂V/∂S|_P") print(" Maxwell関係式: (∂T/∂P)_S = (∂V/∂S)_P") print() # 4. Gibbs自由エネルギー G(T, P)から print("4. Gibbs自由エネルギー G(T, P):") print(" dG = -S dT + V dP") print(" ∂(-S)/∂P|_T = ∂V/∂T|_P") print(" Maxwell関係式: (∂S/∂P)_T = -(∂V/∂T)_P") print() # まとめ表 maxwell_relations = [ ("U(S,V)", "(∂T/∂V)_S", "-(∂P/∂S)_V"), ("F(T,V)", "(∂S/∂V)_T", "(∂P/∂T)_V"), ("H(S,P)", "(∂T/∂P)_S", "(∂V/∂S)_P"), ("G(T,P)", "(∂S/∂P)_T", "-(∂V/∂T)_P") ] print("=== Maxwell関係式一覧 ===") print(f"{'ポテンシャル':<12} {'左辺':<15} {'右辺':<15}") print("-" * 50) for pot, lhs, rhs in maxwell_relations: print(f"{pot:<12} {lhs:<15} = {rhs:<15}") # 実用例：理想気体での検証 print("\n=== 理想気体での検証 ===") print("理想気体: PV = NkT, U = (3/2)NkT (単原子)") print("\nMaxwell関係式 (∂S/∂V)_T = (∂P/∂T)_V の検証:") print(" 左辺: (∂S/∂V)_T = Nk/V") print(" 右辺: (∂P/∂T)_V = Nk/V") print(" → 一致！") 

## 💻 例題2.2: 比熱の関係式

### 比熱の定義

**定積比熱 C_V** : 体積一定で温度を上げるのに必要な熱量

\\[ C_V = \left(\frac{\partial U}{\partial T}\right)_V = T\left(\frac{\partial S}{\partial T}\right)_V \\]

**定圧比熱 C_P** : 圧力一定で温度を上げるのに必要な熱量

\\[ C_P = \left(\frac{\partial H}{\partial T}\right)_P = T\left(\frac{\partial S}{\partial T}\right)_P \\]

**重要な関係式** :

\\[ C_P - C_V = -T\left(\frac{\partial P}{\partial T}\right)_V^2 \left(\frac{\partial P}{\partial V}\right)_T^{-1} \\]

Python実装: 比熱関係式の検証

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import fsolve # van der Waals気体の状態方程式 # (P + a/V²)(V - b) = RT def van_der_waals_pressure(V, T, a, b, R): """van der Waals圧力""" return R * T / (V - b) - a / V**2 def compute_heat_capacity_difference(V, T, a, b, R): """C_P - C_V の計算""" # (∂P/∂T)_V dP_dT_V = R / (V - b) # (∂P/∂V)_T dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 # C_P - C_V = -T (∂P/∂T)_V² / (∂P/∂V)_T if dP_dV_T != 0: diff = -T * dP_dT_V**2 / dP_dV_T else: diff = np.nan return diff # Ar（アルゴン）のvan der Waalsパラメータ R = 8.314 # J/(mol·K) a = 0.1355 # Pa·m⁶/mol² (1.355 bar·L²/mol²) b = 3.201e-5 # m³/mol (0.03201 L/mol) # 温度範囲 T_range = np.linspace(100, 500, 50) # K V_fixed = 1e-3 # 1 L/mol = 1e-3 m³/mol # C_P - C_V の計算 Cp_minus_Cv_vdw = [] Cp_minus_Cv_ideal = [] for T in T_range: # van der Waals気体 diff_vdw = compute_heat_capacity_difference(V_fixed, T, a, b, R) Cp_minus_Cv_vdw.append(diff_vdw) # 理想気体（C_P - C_V = R） Cp_minus_Cv_ideal.append(R) # 可視化 fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # C_P - C_V の温度依存性 ax1 = axes[0] ax1.plot(T_range, Cp_minus_Cv_vdw, 'b-', linewidth=2, label='van der Waals') ax1.plot(T_range, Cp_minus_Cv_ideal, 'r--', linewidth=2, label='理想気体 (= R)') ax1.set_xlabel('Temperature (K)') ax1.set_ylabel('C_P - C_V (J/(mol·K))') ax1.set_title('比熱差の温度依存性（Ar, V = 1 L/mol）') ax1.legend() ax1.grid(True, alpha=0.3) # 体積依存性 V_range = np.linspace(5e-5, 5e-3, 100) # m³/mol T_fixed = 300 # K Cp_minus_Cv_vs_V = [] for V in V_range: diff = compute_heat_capacity_difference(V, T_fixed, a, b, R) Cp_minus_Cv_vs_V.append(diff) ax2 = axes[1] ax2.plot(V_range * 1000, Cp_minus_Cv_vs_V, 'g-', linewidth=2) ax2.axhline(R, color='r', linestyle='--', linewidth=2, label='理想気体') ax2.set_xlabel('Molar volume (L/mol)') ax2.set_ylabel('C_P - C_V (J/(mol·K))') ax2.set_title(f'比熱差の体積依存性（Ar, T = {T_fixed} K）') ax2.legend() ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('thermo_heat_capacity_difference.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 比熱の関係式（Ar at 300 K, 1 L/mol）===") T = 300 V = 1e-3 diff_vdw = compute_heat_capacity_difference(V, T, a, b, R) print(f"van der Waals: C_P - C_V = {diff_vdw:.4f} J/(mol·K)") print(f"理想気体: C_P - C_V = {R:.4f} J/(mol·K)") print(f"相対誤差: {abs(diff_vdw - R) / R * 100:.2f}%") print("\n低密度極限（V → ∞）では理想気体に近づく") 

## 💻 例題2.3: 圧縮率と膨張係数

### 圧縮率と膨張係数の定義

**等温圧縮率 κ_T** : 圧力変化に対する体積の相対変化

\\[ \kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T \\]

**断熱圧縮率 κ_S** :

\\[ \kappa_S = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_S \\]

**体膨張係数 α** : 温度変化に対する体積の相対変化

\\[ \alpha = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P \\]

**重要な関係式** :

\\[ \frac{\kappa_T}{\kappa_S} = \frac{C_P}{C_V} \\]

Python実装: 圧縮率と膨張係数の計算

import numpy as np import matplotlib.pyplot as plt # 等温圧縮率の計算 def isothermal_compressibility(V, T, a, b, R): """等温圧縮率 κ_T = -1/V (∂V/∂P)_T""" # (∂P/∂V)_T を計算し、逆数を取る dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 if dP_dV_T != 0: kappa_T = -1 / (V * dP_dV_T) else: kappa_T = np.inf return kappa_T def volumetric_expansion(V, T, a, b, R): """体膨張係数 α = 1/V (∂V/∂T)_P""" # Maxwell関係式: (∂V/∂T)_P = -(∂S/∂P)_T # ただし簡単のため (∂P/∂T)_V と (∂P/∂V)_T から計算 dP_dT_V = R / (V - b) dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 if dP_dV_T != 0: dV_dT_P = -dP_dT_V / dP_dV_T alpha = dV_dT_P / V else: alpha = np.inf return alpha # Arのパラメータ R = 8.314 a = 0.1355 b = 3.201e-5 # 温度・体積範囲 T_range = np.linspace(150, 500, 100) V_range = np.linspace(1e-4, 5e-3, 100) # 温度依存性（V固定） V_fixed = 1e-3 kappa_T_vs_T = [] alpha_vs_T = [] for T in T_range: kappa_T = isothermal_compressibility(V_fixed, T, a, b, R) alpha = volumetric_expansion(V_fixed, T, a, b, R) kappa_T_vs_T.append(kappa_T) alpha_vs_T.append(alpha) # 体積依存性（T固定） T_fixed = 300 kappa_T_vs_V = [] alpha_vs_V = [] for V in V_range: kappa_T = isothermal_compressibility(V, T_fixed, a, b, R) alpha = volumetric_expansion(V, T_fixed, a, b, R) kappa_T_vs_V.append(kappa_T) alpha_vs_V.append(alpha) # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 等温圧縮率の温度依存性 ax1 = axes[0, 0] ax1.plot(T_range, np.array(kappa_T_vs_T) * 1e9, 'b-', linewidth=2) ax1.set_xlabel('Temperature (K)') ax1.set_ylabel('κ_T (GPa⁻¹)') ax1.set_title(f'等温圧縮率の温度依存性（V = {V_fixed*1000:.1f} L/mol）') ax1.grid(True, alpha=0.3) # 体膨張係数の温度依存性 ax2 = axes[0, 1] ax2.plot(T_range, np.array(alpha_vs_T) * 1e3, 'r-', linewidth=2) ax2.set_xlabel('Temperature (K)') ax2.set_ylabel('α (10⁻³ K⁻¹)') ax2.set_title(f'体膨張係数の温度依存性（V = {V_fixed*1000:.1f} L/mol）') ax2.grid(True, alpha=0.3) # 等温圧縮率の体積依存性 ax3 = axes[1, 0] ax3.plot(V_range * 1000, np.array(kappa_T_vs_V) * 1e9, 'g-', linewidth=2) ax3.set_xlabel('Molar volume (L/mol)') ax3.set_ylabel('κ_T (GPa⁻¹)') ax3.set_title(f'等温圧縮率の体積依存性（T = {T_fixed} K）') ax3.grid(True, alpha=0.3) # 体膨張係数の体積依存性 ax4 = axes[1, 1] ax4.plot(V_range * 1000, np.array(alpha_vs_V) * 1e3, 'm-', linewidth=2) ax4.set_xlabel('Molar volume (L/mol)') ax4.set_ylabel('α (10⁻³ K⁻¹)') ax4.set_title(f'体膨張係数の体積依存性（T = {T_fixed} K）') ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('thermo_compressibility_expansion.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 圧縮率と膨張係数（Ar at 300 K, 1 L/mol）===") T = 300 V = 1e-3 kappa_T = isothermal_compressibility(V, T, a, b, R) alpha = volumetric_expansion(V, T, a, b, R) print(f"等温圧縮率 κ_T = {kappa_T*1e9:.4f} GPa⁻¹") print(f"体膨張係数 α = {alpha*1e3:.4f} × 10⁻³ K⁻¹") # 理想気体との比較 kappa_T_ideal = V / (R * T) # 理想気体: κ_T = 1/P alpha_ideal = 1 / T # 理想気体: α = 1/T print(f"\n理想気体:") print(f" κ_T = {kappa_T_ideal*1e9:.4f} GPa⁻¹") print(f" α = {alpha_ideal*1e3:.4f} × 10⁻³ K⁻¹") 

## 💻 例題2.4: Jacobian法による恒等式導出

### Jacobian法とは

Jacobi行列式を用いて熱力学的関係式を系統的に導出する手法です。

変数変換 \\((x, y) \to (u, v)\\) において：

\\[ \frac{\partial(u, v)}{\partial(x, y)} = \begin{vmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{vmatrix} \\]

**重要な性質** :

\\[ \frac{\partial(u, v)}{\partial(x, y)} \cdot \frac{\partial(x, y)}{\partial(u, v)} = 1 \\]

Python実装: Jacobian法の実装

import numpy as np import sympy as sp # 記号の定義 T, P, V, S = sp.symbols('T P V S', real=True) def jacobian_2x2(u, v, x, y): """2×2 Jacobi行列式を計算""" J = sp.Matrix([ [sp.diff(u, x), sp.diff(u, y)], [sp.diff(v, x), sp.diff(v, y)] ]) return J.det() # 例: (∂P/∂T)_V を (∂S/∂V)_T で表す # Maxwell関係式: (∂S/∂V)_T = (∂P/∂T)_V を導出 print("=== Jacobian法によるMaxwell関係式の導出 ===\n") # 変数として S, V を関数として扱う S_func = sp.Function('S')(T, P) V_func = sp.Function('V')(T, P) # Jacobi行列式の性質 print("1. Jacobianの連鎖律:") print(" ∂(S,V)/∂(T,P) · ∂(T,P)/∂(S,V) = 1") print() # 具体例: Gibbs自由エネルギーから print("2. Gibbs自由エネルギー G(T,P):") print(" dG = -S dT + V dP") print(" → S = -(∂G/∂T)_P, V = (∂G/∂P)_T") print() print("3. 2次偏微分の可換性:") print(" ∂²G/∂T∂P = ∂²G/∂P∂T") print(" → ∂S/∂P|_T = -∂V/∂T|_P") print(" これがMaxwell関係式") print() # 実用的な恒等式の導出 print("=== 実用的な恒等式の導出 ===\n") print("4. (∂U/∂V)_T を測定可能な量で表す:") print(" dU = TdS - PdV より") print(" (∂U/∂V)_T = T(∂S/∂V)_T - P") print(" Maxwell関係式 (∂S/∂V)_T = (∂P/∂T)_V を使うと") print(" (∂U/∂V)_T = T(∂P/∂T)_V - P") print() # 数値例: van der Waals気体で検証 from scipy.misc import derivative def U_vdw(V, T, a, b, R, n=1): """van der Waals気体の内部エネルギー""" # U = (3/2)nRT - n²a/V （単原子分子） return 1.5 * n * R * T - n**2 * a / V def P_vdw(V, T, a, b, R, n=1): """van der Waals気体の圧力""" return n * R * T / (V - n*b) - n**2 * a / V**2 # Arのパラメータ R = 8.314 a = 0.1355 b = 3.201e-5 n = 1 # 1 mol T = 300 V = 1e-3 # 左辺: (∂U/∂V)_T を数値微分 dU_dV_T = derivative(lambda v: U_vdw(v, T, a, b, R, n), V, dx=1e-8) # 右辺: T(∂P/∂T)_V - P dP_dT_V = derivative(lambda t: P_vdw(V, t, a, b, R, n), T, dx=1e-6) P = P_vdw(V, T, a, b, R, n) right_hand_side = T * dP_dT_V - P print("5. van der Waals気体での検証（Ar, 300 K, 1 L/mol）:") print(f" 左辺 (∂U/∂V)_T = {dU_dV_T:.4f} J/m³") print(f" 右辺 T(∂P/∂T)_V - P = {right_hand_side:.4f} J/m³") print(f" 相対誤差: {abs(dU_dV_T - right_hand_side)/abs(dU_dV_T)*100:.6f}%") print("\n → 恒等式が数値的に確認された！") 

## 💻 例題2.5: 等温圧縮率と断熱圧縮率の関係

Python実装: κ_T / κ_S = C_P / C_V の検証

import numpy as np import matplotlib.pyplot as plt # 理論的関係式: κ_T / κ_S = C_P / C_V = γ def compute_gamma_ratio(V, T, a, b, R): """γ = κ_T / κ_S = C_P / C_V の計算""" # 等温圧縮率 dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 kappa_T = -1 / (V * dP_dV_T) if dP_dV_T != 0 else np.inf # C_P - C_V dP_dT_V = R / (V - b) Cp_minus_Cv = -T * dP_dT_V**2 / dP_dV_T if dP_dV_T != 0 else np.inf # 単原子分子理想気体の C_V = (3/2)R を基準に補正 Cv = 1.5 * R # 単原子分子 Cp = Cv + Cp_minus_Cv gamma = Cp / Cv # 断熱圧縮率（理論式から） kappa_S = kappa_T / gamma return gamma, kappa_T, kappa_S, Cp, Cv # Arのパラメータ R = 8.314 a = 0.1355 b = 3.201e-5 # 温度範囲での γ の計算 T_range = np.linspace(150, 500, 100) V_fixed = 1e-3 gamma_values = [] kappa_T_values = [] kappa_S_values = [] Cp_values = [] Cv_values = [] for T in T_range: gamma, kappa_T, kappa_S, Cp, Cv = compute_gamma_ratio(V_fixed, T, a, b, R) gamma_values.append(gamma) kappa_T_values.append(kappa_T) kappa_S_values.append(kappa_S) Cp_values.append(Cp) Cv_values.append(Cv) # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # γ = C_P / C_V ax1 = axes[0, 0] ax1.plot(T_range, gamma_values, 'b-', linewidth=2) ax1.axhline(5/3, color='r', linestyle='--', linewidth=1.5, label='理想気体 (5/3)') ax1.set_xlabel('Temperature (K)') ax1.set_ylabel('γ = C_P / C_V') ax1.set_title('比熱比の温度依存性') ax1.legend() ax1.grid(True, alpha=0.3) # C_P と C_V ax2 = axes[0, 1] ax2.plot(T_range, Cp_values, 'r-', linewidth=2, label='C_P') ax2.plot(T_range, Cv_values, 'b-', linewidth=2, label='C_V') ax2.set_xlabel('Temperature (K)') ax2.set_ylabel('Heat capacity (J/(mol·K))') ax2.set_title('比熱の温度依存性') ax2.legend() ax2.grid(True, alpha=0.3) # κ_T と κ_S ax3 = axes[1, 0] ax3.plot(T_range, np.array(kappa_T_values) * 1e9, 'g-', linewidth=2, label='κ_T') ax3.plot(T_range, np.array(kappa_S_values) * 1e9, 'm-', linewidth=2, label='κ_S') ax3.set_xlabel('Temperature (K)') ax3.set_ylabel('Compressibility (GPa⁻¹)') ax3.set_title('圧縮率の温度依存性') ax3.legend() ax3.grid(True, alpha=0.3) # 関係式の検証: κ_T / κ_S vs C_P / C_V ax4 = axes[1, 1] kappa_ratio = np.array(kappa_T_values) / np.array(kappa_S_values) Cp_Cv_ratio = np.array(Cp_values) / np.array(Cv_values) ax4.plot(T_range, kappa_ratio, 'b-', linewidth=2, label='κ_T / κ_S') ax4.plot(T_range, Cp_Cv_ratio, 'r--', linewidth=2, label='C_P / C_V') ax4.set_xlabel('Temperature (K)') ax4.set_ylabel('Ratio') ax4.set_title('関係式 κ_T / κ_S = C_P / C_V の検証') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('thermo_kappa_gamma_relation.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== κ_T / κ_S = C_P / C_V の検証（Ar at 300 K）===") T = 300 V = 1e-3 gamma, kappa_T, kappa_S, Cp, Cv = compute_gamma_ratio(V, T, a, b, R) print(f"C_P = {Cp:.4f} J/(mol·K)") print(f"C_V = {Cv:.4f} J/(mol·K)") print(f"γ = C_P / C_V = {gamma:.6f}") print() print(f"κ_T = {kappa_T*1e9:.4f} GPa⁻¹") print(f"κ_S = {kappa_S*1e9:.4f} GPa⁻¹") print(f"κ_T / κ_S = {kappa_T/kappa_S:.6f}") print() print(f"相対誤差: {abs(gamma - kappa_T/kappa_S)/gamma*100:.6f}%") print("\n理想気体（単原子）では γ = 5/3 = 1.667") print(f"van der Waals気体: γ = {gamma:.4f}") 

## 💻 例題2.6: 熱力学的安定性条件

### 熱力学的安定性

平衡状態が安定であるための条件は、2次微分が正であることです。

**主要な安定性条件** :

  * \\(\left(\frac{\partial^2 G}{\partial T^2}\right)_P = -\frac{C_P}{T} < 0\\) → \\(C_P > 0\\)
  * \\(\left(\frac{\partial^2 G}{\partial P^2}\right)_T = V \kappa_T > 0\\) → \\(\kappa_T > 0\\)
  * \\(\left(\frac{\partial P}{\partial V}\right)_T < 0\\) （等温での力学的安定性）

Python実装: 安定性条件の可視化

import numpy as np import matplotlib.pyplot as plt def check_stability(V, T, a, b, R): """熱力学的安定性条件をチェック""" # (∂P/∂V)_T < 0 の確認 dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 # κ_T > 0 の確認 kappa_T = -1 / (V * dP_dV_T) if dP_dV_T != 0 else -np.inf # C_P > 0 の確認（簡略化: C_V > 0を仮定） Cv = 1.5 * R dP_dT_V = R / (V - b) Cp_minus_Cv = -T * dP_dT_V**2 / dP_dV_T if dP_dV_T != 0 else np.inf Cp = Cv + Cp_minus_Cv # 安定性判定 stable = (dP_dV_T < 0) and (kappa_T > 0) and (Cp > 0) return { 'dP_dV_T': dP_dV_T, 'kappa_T': kappa_T, 'Cp': Cp, 'stable': stable } # Arのパラメータ R = 8.314 a = 0.1355 b = 3.201e-5 # T-V 平面での安定性マップ T_range = np.linspace(100, 500, 100) V_range = np.linspace(5e-5, 5e-3, 100) T_grid, V_grid = np.meshgrid(T_range, V_range) stability_map = np.zeros_like(T_grid) dP_dV_T_map = np.zeros_like(T_grid) for i in range(len(V_range)): for j in range(len(T_range)): V = V_range[i] T = T_range[j] result = check_stability(V, T, a, b, R) stability_map[i, j] = 1 if result['stable'] else 0 dP_dV_T_map[i, j] = result['dP_dV_T'] # 可視化 fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 安定性マップ ax1 = axes[0] c1 = ax1.contourf(T_grid, V_grid * 1000, stability_map, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.5) ax1.set_xlabel('Temperature (K)') ax1.set_ylabel('Molar volume (L/mol)') ax1.set_title('熱力学的安定性マップ（緑=安定, 赤=不安定）') # (∂P/∂V)_T のマップ ax2 = axes[1] levels = np.linspace(-1e8, 1e8, 20) c2 = ax2.contourf(T_grid, V_grid * 1000, dP_dV_T_map, levels=levels, cmap='RdBu_r') ax2.contour(T_grid, V_grid * 1000, dP_dV_T_map, levels=[0], colors='black', linewidths=2) ax2.set_xlabel('Temperature (K)') ax2.set_ylabel('Molar volume (L/mol)') ax2.set_title('(∂P/∂V)_T マップ（負=安定, 正=不安定）') plt.colorbar(c2, ax=ax2, label='(∂P/∂V)_T (Pa/m³)') plt.tight_layout() plt.savefig('thermo_stability_conditions.png', dpi=300, bbox_inches='tight') plt.show() # スピノーダル線の計算（(∂P/∂V)_T = 0） print("=== 熱力学的安定性条件 ===\n") print("スピノーダル線: (∂P/∂V)_T = 0 となる条件") print("van der Waals: -RT/(V-b)² + 2a/V³ = 0") print("→ V_spinodal³ = 2a(V-b)² / (RT)") print() # 臨界点でのチェック T_c = 8 * a / (27 * R * b) V_c = 3 * b print(f"臨界点（Ar）:") print(f" T_c = {T_c:.2f} K") print(f" V_c = {V_c*1000:.4f} L/mol") result_c = check_stability(V_c, T_c, a, b, R) print(f" (∂P/∂V)_T = {result_c['dP_dV_T']:.2e} Pa/m³") print(f" → 臨界点で (∂P/∂V)_T = 0 （安定性の境界）") 

## 💻 例題2.7: 実験データからのエントロピー計算

Python実装: Maxwell関係式を用いた物性値計算

import numpy as np import matplotlib.pyplot as plt from scipy.integrate import simpson # 実験データ（模擬）: Arガスの熱膨張係数 # α = 1/V (∂V/∂T)_P def generate_mock_data(): """模擬実験データの生成""" T_data = np.linspace(100, 400, 31) # 10 K間隔 # van der Waalsモデルから生成 R = 8.314 a = 0.1355 b = 3.201e-5 P = 1e5 # 1 bar = 1e5 Pa alpha_data = [] for T in T_data: # P一定の条件で体積を計算（簡略化: 理想気体に近似） V = R * T / P # 熱膨張係数 dP_dT_V = R / (V - b) dP_dV_T = -R * T / (V - b)**2 + 2 * a / V**3 dV_dT_P = -dP_dT_V / dP_dV_T alpha = dV_dT_P / V alpha_data.append(alpha) return T_data, np.array(alpha_data) # 実験データ取得 T_data, alpha_data = generate_mock_data() # Maxwell関係式: (∂S/∂P)_T = -(∂V/∂T)_P を使ってエントロピーを計算 # ΔS = ∫(∂S/∂P)_T dP = -∫(∂V/∂T)_P dP = -∫V·α dP def compute_entropy_change(T, alpha, V, P_initial, P_final, n_points=100): """圧力変化に伴うエントロピー変化""" P_range = np.linspace(P_initial, P_final, n_points) # 各圧力での体積（理想気体近似） R = 8.314 V_range = R * T / P_range # (∂S/∂P)_T = -V·α integrand = -V_range * alpha # Simpson積分 delta_S = simpson(integrand, x=P_range) return delta_S # 各温度でのエントロピー変化を計算 P_initial = 1e5 # 1 bar P_final = 10e5 # 10 bar R = 8.314 entropy_changes = [] for i, T in enumerate(T_data): V = R * T / P_initial alpha = alpha_data[i] delta_S = compute_entropy_change(T, alpha, V, P_initial, P_final) entropy_changes.append(delta_S) # 可視化 fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 熱膨張係数のデータ ax1 = axes[0] ax1.plot(T_data, alpha_data * 1e3, 'bo-', markersize=4, linewidth=2, label='実験データ') ax1.set_xlabel('Temperature (K)') ax1.set_ylabel('α (10⁻³ K⁻¹)') ax1.set_title('熱膨張係数の実験データ') ax1.legend() ax1.grid(True, alpha=0.3) # エントロピー変化 ax2 = axes[1] ax2.plot(T_data, entropy_changes, 'ro-', markersize=4, linewidth=2) ax2.set_xlabel('Temperature (K)') ax2.set_ylabel('ΔS (J/(mol·K))') ax2.set_title(f'圧力変化のエントロピー変化\n({P_initial/1e5:.0f} → {P_final/1e5:.0f} bar)') ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('thermo_entropy_from_expansion.png', dpi=300, bbox_inches='tight') plt.show() # 結果の表示 print("=== 実験データからのエントロピー計算 ===\n") print("Maxwell関係式: (∂S/∂P)_T = -(∂V/∂T)_P = -V·α") print(f"圧力変化: {P_initial/1e5:.0f} bar → {P_final/1e5:.0f} bar\n") print(f"{'温度 (K)':<12} {'α (10⁻³ K⁻¹)':<18} {'ΔS (J/(mol·K))':<18}") print("-" * 50) for i in [0, 10, 20, 30]: print(f"{T_data[i]:<12.1f} {alpha_data[i]*1e3:<18.4f} {entropy_changes[i]:<18.4f}") print("\n理論値との比較（理想気体）:") T_ref = 300 delta_S_ideal = -R * np.log(P_final / P_initial) print(f" 理想気体: ΔS = -R ln(P_f/P_i) = {delta_S_ideal:.4f} J/(mol·K)") print(f" 計算値 (at {T_ref} K): {entropy_changes[20]:.4f} J/(mol·K)") 

## 📚 まとめ

  * **Maxwell関係式** は熱力学ポテンシャルの2次偏微分の可換性から導かれる
  * 4つの基本Maxwell関係式が4つの熱力学ポテンシャル（U, F, H, G）に対応
  * 比熱の関係式 \\(C_P - C_V = -T\left(\frac{\partial P}{\partial T}\right)_V^2 \left(\frac{\partial P}{\partial V}\right)_T^{-1}\\) は実用上重要
  * **圧縮率** と**膨張係数** は材料の機械的・熱的性質を特徴づける
  * \\(\kappa_T / \kappa_S = C_P / C_V\\) は重要な熱力学的関係式
  * **Jacobian法** は複雑な熱力学的恒等式を系統的に導出できる
  * 熱力学的安定性条件は \\(C_P > 0\\), \\(\kappa_T > 0\\), \\(\left(\frac{\partial P}{\partial V}\right)_T < 0\\)
  * Maxwell関係式により実験データから測定困難な物性値を計算できる

### 💡 演習問題

  1. **[Easy]** 理想気体 \\(PV = nRT\\) について、Maxwell関係式 \\(\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V\\) を検証せよ。
  2. **[Easy]** van der Waals気体の臨界点 \\((T_c, P_c, V_c)\\) を計算し、臨界点で \\(\left(\frac{\partial P}{\partial V}\right)_T = 0\\) となることを確認せよ。
  3. **[Medium]** 理想気体の \\(C_P - C_V = nR\\) を、Maxwell関係式と比熱の定義から導出せよ。
  4. **[Medium]** 等温圧縮率 \\(\kappa_T\\) と断熱圧縮率 \\(\kappa_S\\) の比が比熱比 \\(\gamma = C_P / C_V\\) に等しいことを、熱力学的関係式から導出せよ。
  5. **[Hard]** Jacobi行列式を用いて、\\(\left(\frac{\partial U}{\partial P}\right)_T = -T^2 \left(\frac{\partial (P/T)}{\partial T}\right)_P\\) を導出せよ。（ヒント: \\(U(T, P)\\) の全微分と Maxwell関係式を使う）

[← 第1章: 熱力学の基本法則](<chapter-1.html>) [シリーズTOP](<index.html>) [第3章: 相平衡と相図 →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
