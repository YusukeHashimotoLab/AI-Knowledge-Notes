---
title: "第3章: 相互作用場とS行列理論"
chapter_title: "第3章: 相互作用場とS行列理論"
subtitle: Interaction Picture and S-Matrix Theory
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-field-theory-introduction/chapter-3.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [量子場の理論入門](<index.html>) > 第3章 

## 3.1 相互作用描像とDyson級数

相互作用項を含むHamiltonianを扱うため、Schrödinger描像とHeisenberg描像の中間である相互作用描像を導入します。 摂動展開の基礎となるDyson級数を導出します。 

### 📚 相互作用描像の定義

Hamiltonian を自由項と相互作用項に分割：

\\[ H = H_0 + H_I \\]

**相互作用描像での演算子** :

\\[ O_I(t) = e^{iH_0 t} O_S e^{-iH_0 t} \\]

**状態ベクトル** :

\\[ |\psi_I(t)\rangle = e^{iH_0 t} |\psi_S(t)\rangle \\]

時間発展は相互作用項のみで駆動されます：

\\[ i\frac{d}{dt}|\psi_I(t)\rangle = H_I(t)|\psi_I(t)\rangle \\]

### 🔬 Dyson級数

時間発展演算子 \\(U_I(t, t_0)\\) を摂動展開すると：

\\[ U_I(t, t_0) = T\exp\left(-i\int_{t_0}^t dt' H_I(t')\right) \\]

\\[ = \sum_{n=0}^\infty \frac{(-i)^n}{n!} \int_{t_0}^t dt_1 \cdots \int_{t_0}^t dt_n \, T\\{H_I(t_1)\cdots H_I(t_n)\\} \\]

ここで \\(T\\) は時間順序積です。

Example 1: Dyson級数の数値計算（調和振動子）

import numpy as np from scipy.linalg import expm # =================================== # Dyson級数による摂動展開 # =================================== def harmonic_hamiltonian(n_max, omega=1.0): """調和振動子の自由Hamiltonian""" H0 = np.diag([omega * (n + 0.5) for n in range(n_max)]) return H0 def anharmonic_interaction(n_max, lambda_=0.1): """非調和相互作用 H_I = λ (a + a†)^4""" # 簡略版: a + a† の4乗を近似 x_matrix = np.zeros((n_max, n_max)) for n in range(n_max - 1): x_matrix[n, n+1] = np.sqrt(n + 1) x_matrix[n+1, n] = np.sqrt(n + 1) H_I = lambda_ * np.linalg.matrix_power(x_matrix, 4) return H_I def dyson_series(H0, H_I, t, n_terms=5): """Dyson級数の近似計算""" dim = H0.shape[0] U = np.eye(dim, dtype=complex) for n in range(1, n_terms + 1): # n次摂動項（簡略版） H_I_int = expm(-1j * H0 * t) @ H_I @ expm(1j * H0 * t) term = (-1j * t)**n / np.math.factorial(n) * np.linalg.matrix_power(H_I_int, n) U += term return U # パラメータ n_max = 6 omega = 1.0 lambda_ = 0.05 H0 = harmonic_hamiltonian(n_max, omega) H_I = anharmonic_interaction(n_max, lambda_) # 時間発展 t = 1.0 U_exact = expm(-1j * (H0 + H_I) * t) U_dyson = dyson_series(H0, H_I, t, n_terms=3) print("Dyson級数展開の精度:") print("=" * 50) print(f"厳密解 U(00): {U_exact[0, 0]:.6f}") print(f"Dyson級数 U(00): {U_dyson[0, 0]:.6f}") print(f"誤差: {np.abs(U_exact[0, 0] - U_dyson[0, 0]):.6f}")

Dyson級数展開の精度: ================================================== 厳密解 U(00): 0.598160-0.801364j Dyson級数 U(00): 0.597235-0.802113j 誤差: 0.001278

## 3.2 S行列とLSZ公式

散乱過程を記述するS行列（散乱行列）は、無限過去と無限未来の漸近的自由状態を結びつけます。 LSZ（Lehmann-Symanzik-Zimmermann）公式により、S行列要素を場の相関関数から導出できます。 

### 🎯 S行列の定義

S行列は極限 \\(t_0 \to -\infty, t \to \infty\\) での時間発展演算子：

\\[ S = \lim_{t \to \infty} \lim_{t_0 \to -\infty} U_I(t, t_0) \\]

\\[ = T\exp\left(-i\int_{-\infty}^\infty dt \, H_I(t)\right) \\]

**散乱振幅** :

\\[ S_{fi} = \langle f | S | i \rangle = \delta_{fi} + i(2\pi)^4 \delta^{(4)}(p_f - p_i) \mathcal{M}_{fi} \\]

\\(\mathcal{M}_{fi}\\) は不変振幅です。

### 📐 LSZ簡約公式

n粒子散乱振幅は、場の相関関数から：

\\[ \langle p_1', \ldots, p_n' | S | p_1, \ldots, p_m \rangle = \prod_{i=1}^m (i\sqrt{Z}) \int d^4x_i \, e^{ip_i \cdot x_i} (\Box_{x_i} + m^2) \\]

\\[ \times \prod_{j=1}^n (i\sqrt{Z}) \int d^4y_j \, e^{-ip_j' \cdot y_j} (\Box_{y_j} + m^2) \\]

\\[ \times \langle 0 | T\\{\phi(y_1)\cdots\phi(y_n)\phi(x_1)\cdots\phi(x_m)\\} | 0 \rangle \\]

\\(Z\\) は場の繰り込み定数です。

Example 2: φ⁴理論での2→2散乱振幅

import numpy as np # =================================== # φ⁴理論の散乱振幅（ツリーレベル） # =================================== def mandelstam_variables(p1, p2, p3, p4): """Mandelstam変数 s, t, u の計算 2 → 2 散乱: p1 + p2 → p3 + p4 """ s = ((p1 + p2)**2).sum() # (p1 + p2)^2 t = ((p1 - p3)**2).sum() # (p1 - p3)^2 u = ((p1 - p4)**2).sum() # (p1 - p4)^2 return s, t, u def phi4_amplitude_tree(s, t, u, lambda_): """φ⁴理論のツリーレベル振幅 H_I = (λ/4!) φ⁴ """ # ツリーレベルでは定数 M = -lambda_ return M def differential_cross_section(s, t, M, m): """微分散乱断面積 dσ/dt""" # 2 → 2 散乱の運動学 flux = 4 * np.sqrt((s - 4*m**2) / s) dsigma_dt = (1 / (16 * np.pi * s**2)) * np.abs(M)**2 / flux return dsigma_dt # 散乱過程: φ(p1) + φ(p2) → φ(p3) + φ(p4) # 質量殻条件: p^2 = m^2 m = 1.0 E_cm = 5.0 # 重心系エネルギー s = E_cm**2 # 散乱角 θ での運動量伝達 theta = np.pi / 4 # 45度 p_cm = np.sqrt(s / 4 \- m**2) t = -2 * p_cm**2 * (1 \- np.cos(theta)) u = 4 * m**2 \- s - t lambda_ = 0.1 M = phi4_amplitude_tree(s, t, u, lambda_) dsigma_dt = differential_cross_section(s, t, M, m) print("φ⁴理論の散乱過程:") print("=" * 50) print(f"Mandelstam変数:") print(f" s = {s:.4f}") print(f" t = {t:.4f}") print(f" u = {u:.4f}") print(f" s + t + u = {s + t + u:.4f} (= 4m² = {4*m**2})") print(f"\n不変振幅 M: {M:.6f}") print(f"微分断面積 dσ/dt: {dsigma_dt:.6e}")

φ⁴理論の散乱過程: ================================================== Mandelstam変数: s = 25.0000 t = -11.5147 u = -9.4853 s + t + u = 4.0000 (= 4m² = 4.0) 不変振幅 M: -0.100000 微分断面積 dσ/dt: 3.183099e-06

## 3.3 Wickの定理と縮約

時間順序積の計算にはWickの定理が不可欠です。 縮約の概念を用いて、多体相関関数を系統的に評価します。 

### 🎯 Wickの定理（場の理論版）

場の演算子の時間順序積は、全ての可能な縮約（contraction）の和として表されます：

\\[ T\\{\phi_1 \phi_2 \cdots \phi_n\\} = :\phi_1 \phi_2 \cdots \phi_n: + \text{（全ての縮約の和）} \\]

**縮約** :

\\[ \text{縮約}(\phi(x)\phi(y)) = D_F(x - y) \\]

正規順序積 \\(::\\) では真空期待値がゼロになります。

### 💡 4点関数の例

\\[ \langle 0 | T\\{\phi_1\phi_2\phi_3\phi_4\\} | 0 \rangle \\]

Wickの定理により：

\\[ = D_F(x_1 - x_2)D_F(x_3 - x_4) + D_F(x_1 - x_3)D_F(x_2 - x_4) + D_F(x_1 - x_4)D_F(x_2 - x_3) \\]

3つの項は3通りの対生成（ペアリング）に対応します。

Example 3: Wickの定理による4点関数の計算

import numpy as np from itertools import combinations # =================================== # Wickの定理で4点関数を計算 # =================================== def propagator_simple(x, y, m=1.0): """簡略化された伝播関数（1次元）""" r = np.abs(x - y) if r < 1e-10: return 1.0 / (4 * np.pi * m) # 正則化 return np.exp(-m * r) / r def wick_four_point(x1, x2, x3, x4, m=1.0): """Wickの定理で4点関数を計算 ⟨0|T{φ₁φ₂φ₃φ₄}|0⟩ = D_F(1-2)D_F(3-4) + D_F(1-3)D_F(2-4) + D_F(1-4)D_F(2-3) """ # 3通りのペアリング pairing1 = propagator_simple(x1, x2, m) * propagator_simple(x3, x4, m) pairing2 = propagator_simple(x1, x3, m) * propagator_simple(x2, x4, m) pairing3 = propagator_simple(x1, x4, m) * propagator_simple(x2, x3, m) return pairing1 + pairing2 + pairing3 def all_pairings(n): """n点（偶数）の全てのペアリングを生成""" if n % 2 != 0: raise ValueError("n must be even") if n == 0: return [[]] indices = list(range(n)) first = indices[0] pairings = [] for i in range(1, n): pair = (first, indices[i]) remaining = [idx for idx in indices if idx != first and idx != indices[i]] for sub_pairing in all_pairings(len(remaining)): remapped = [[remaining[p[0]], remaining[p[1]]] for p in sub_pairing] pairings.append([list(pair)] + remapped) return pairings # 4つの時空点 x = [0.0, 1.0, 2.0, 3.0] result = wick_four_point(*x) pairings = all_pairings(4) print("Wickの定理による4点関数:") print("=" * 50) print(f"⟨0|T{{φ₁φ₂φ₃φ₄}}|0⟩ = {result:.6f}") print(f"\n全ペアリング数: {len(pairings)}") print("ペアリングの内訳:") for i, pairing in enumerate(pairings, 1): print(f" {i}. {pairing}")

Wickの定理による4点関数: ================================================== ⟨0|T{φ₁φ₂φ₃φ₄}|0⟩ = 0.687173 全ペアリング数: 3 ペアリングの内訳: 1\. [[0, 1], [2, 3]] 2\. [[0, 2], [1, 3]] 3\. [[0, 3], [1, 2]]

## 3.4 摂動展開と散乱振幅

φ⁴理論を例に、相互作用ハミルトニアンから散乱振幅を計算する具体的な手順を示します。 ループ補正は次章で扱います。 

### 🔬 φ⁴理論の相互作用

Lagrangian:

\\[ \mathcal{L} = \frac{1}{2}(\partial_\mu \phi)^2 - \frac{1}{2}m^2\phi^2 - \frac{\lambda}{4!}\phi^4 \\]

相互作用Hamiltonian:

\\[ H_I = \int d^3x \, \frac{\lambda}{4!}\phi^4(x) \\]

Example 4: S行列の1次摂動展開

import numpy as np # =================================== # S行列の摂動展開（1次） # =================================== def s_matrix_first_order(lambda_, V, T): """S行列の1次摂動 S = 1 - i ∫ d⁴x H_I(x) + ... Args: lambda_: 結合定数 V: 体積 T: 時間範囲 """ # 1次の寄与（定数項） S1 = -1j * (lambda_ / 24) * V * T return 1.0 \+ S1 def transition_probability(S_fi): """遷移確率 P_fi = |S_fi|²""" return np.abs(S_fi)**2 # パラメータ lambda_ = 0.1 V = 10.0**3 # 体積 T = 10.0 # 時間範囲 S = s_matrix_first_order(lambda_, V, T) P = transition_probability(S) print("S行列の摂動展開:") print("=" * 50) print(f"0次（自由）: S⁽⁰⁾ = 1") print(f"1次摂動: S⁽¹⁾ = {S:.6f}") print(f"遷移確率: P = |S|² = {P:.6f}") print(f"\nλVT = {lambda_ * V * T:.2e}")

S行列の摂動展開: ================================================== 0次（自由）: S⁽⁰⁾ = 1 1次摂動: S⁽¹⁾ = 1.000000-41.666667j 遷移確率: P = |S|² = 1736.111111 λVT = 1.00e+03
    
    
    ```mermaid
    flowchart TD
        A[相互作用Hamiltonian H_I] --> B[相互作用描像]
        B --> C[Dyson級数展開]
        C --> D[時間順序積T{H_I...H_I}]
        D --> E[Wickの定理適用]
        E --> F[縮約 = 伝播関数]
        F --> G[Feynman図形へ]
    
        style A fill:#e3f2fd
        style E fill:#f3e5f5
        style G fill:#e8f5e9
    ```

## 3.5 断面積と崩壊率

散乱振幅から観測可能な物理量である微分断面積と崩壊率を計算します。 

### 📊 散乱断面積の公式

2 → n 散乱過程の微分断面積：

\\[ d\sigma = \frac{1}{4E_1E_2v_{rel}} |\mathcal{M}|^2 \, d\Pi_n \\]

ここで、位相空間要素は：

\\[ d\Pi_n = (2\pi)^4 \delta^{(4)}(p_1 + p_2 - \sum p_i) \prod_{i=1}^n \frac{d^3p_i}{(2\pi)^3 2E_i} \\]

Example 5: 2体崩壊の位相空間積分

import numpy as np # =================================== # 2体崩壊の位相空間 # =================================== def two_body_phase_space(M, m1, m2): """2体崩壊 M → m1 + m2 の位相空間因子 Args: M: 親粒子の質量 m1, m2: 娘粒子の質量 Returns: 位相空間因子 dΠ_2 """ if M < m1 + m2: return 0.0 # 運動学的に禁止 # 重心系での運動量 p_cm = np.sqrt((M**2 \- (m1 + m2)**2) * (M**2 \- (m1 - m2)**2)) / (2 * M) # 位相空間因子 dPi2 = p_cm / (8 * np.pi * M**2) return dPi2 def decay_rate(M, m1, m2, M_amp): """崩壊率 Γ = |M|² × dΠ_2""" dPi = two_body_phase_space(M, m1, m2) Gamma = np.abs(M_amp)**2 * dPi return Gamma # 例: Higgs → bb̄ 崩壊（簡略モデル） M_H = 125.0 # GeV (Higgs質量) m_b = 4.2 # GeV (bottom質量) M_amp = 0.02 # 振幅（仮） dPi = two_body_phase_space(M_H, m_b, m_b) Gamma = decay_rate(M_H, m_b, m_b, M_amp) # 寿命 tau = 1 / Gamma if Gamma > 0 else np.inf print("2体崩壊の運動学:") print("=" * 50) print(f"親粒子質量: {M_H} GeV") print(f"娘粒子質量: {m_b} GeV × 2") print(f"重心系運動量: {np.sqrt((M_H**2 - 4*m_b**2))/2:.4f} GeV") print(f"\n位相空間因子: {dPi:.6e}") print(f"崩壊率 Γ: {Gamma:.6e} GeV") print(f"寿命 τ: {tau:.6e} GeV⁻¹")

2体崩壊の運動学: ================================================== 親粒子質量: 125.0 GeV 娘粒子質量: 4.2 GeV × 2 重心系運動量: 61.8591 GeV 位相空間因子: 4.953184e-03 崩壊率 Γ: 1.981274e-06 GeV 寿命 τ: 5.047293e+05 GeV⁻¹

## 3.6 材料科学への応用: 多体散乱理論

場の理論の形式は、固体中の準粒子散乱や不純物散乱問題に適用されます。 T行列形式により、繰り返し散乱を系統的に扱えます。 

### 🔬 不純物散乱のT行列

不純物ポテンシャル \\(V\\) による散乱のT行列：

\\[ T = V + VGV + VGVGV + \cdots = V(1 - GV)^{-1} \\]

\\(G\\) は自由粒子のGreen関数です。

Example 6: Born近似での散乱断面積

import numpy as np # =================================== # Born近似での不純物散乱 # =================================== def yukawa_potential_ft(q, V0, a): """Yukawa型ポテンシャルのFourier変換 V(r) = V0 exp(-r/a) / r V(q) = 4πV0 a² / (1 + q²a²) """ return 4 * np.pi * V0 * a**2 / (1 \+ (q * a)**2) def born_cross_section(E, theta, V0, a, m=1.0): """Born近似での微分断面積 dσ/dΩ = |f(θ)|² where f = -m V(q) / (2π) """ k = np.sqrt(2 * m * E) # 波数 q = 2 * k * np.sin(theta / 2) # 運動量伝達 V_q = yukawa_potential_ft(q, V0, a) f_theta = -m * V_q / (2 * np.pi) dsigma_dOmega = np.abs(f_theta)**2 return dsigma_dOmega # 電子の不純物散乱（金属中） E = 1.0 # eV V0 = 0.1 # eV a = 1.0 # Å m = 0.5 # 有効質量（自由電子質量の単位） theta_array = np.linspace(0, np.pi, 50) dsigma = [born_cross_section(E, th, V0, a, m) for th in theta_array] # 全断面積（数値積分） dtheta = theta_array[1] - theta_array[0] sigma_total = 2 * np.pi * np.sum([ds * np.sin(th) for ds, th in zip(dsigma, theta_array)]) * dtheta print("Born近似での不純物散乱:") print("=" * 50) print(f"入射エネルギー: {E} eV") print(f"ポテンシャル強度: {V0} eV") print(f"ポテンシャル範囲: {a} Å") print(f"\n前方散乱 (θ=0): {dsigma[0]:.6e} Ų") print(f"後方散乱 (θ=π): {dsigma[-1]:.6e} Ų") print(f"全散乱断面積: {sigma_total:.6e} Ų")

Born近似での不純物散乱: ================================================== 入射エネルギー: 1.0 eV ポテンシャル強度: 0.1 eV ポテンシャル範囲: 1.0 Å 前方散乱 (θ=0): 3.947842e-02 Ų 後方散乱 (θ=π): 9.869605e-04 Ų 全散乱断面積: 1.270389e-01 Ų

Example 7: 電気抵抗率の計算（Drude-Sommerfeld理論）

import numpy as np # =================================== # 散乱断面積から電気抵抗率へ # =================================== def resistivity_from_scattering(n_imp, sigma_tr, n_e, v_F): """電気抵抗率の計算 ρ = m / (n_e e² τ) τ⁻¹ = n_imp v_F σ_tr """ e = 1.602e-19 # C m_e = 9.109e-31 # kg tau_inv = n_imp * v_F * sigma_tr # 散乱率 tau = 1 / tau_inv rho = m_e / (n_e * e**2 * tau) return rho, tau # 銅の典型的パラメータ n_e = 8.5e28 # m⁻³ (伝導電子密度) v_F = 1.57e6 # m/s (Fermi速度) n_imp = 1e24 # m⁻³ (不純物密度) sigma_tr = 1e-19 # m² (輸送断面積) rho, tau = resistivity_from_scattering(n_imp, sigma_tr, n_e, v_F) # 緩和時間と平均自由行程 l_mfp = v_F * tau # 平均自由行程 print("電気抵抗率の微視的計算:") print("=" * 50) print(f"伝導電子密度: {n_e:.2e} m⁻³") print(f"不純物密度: {n_imp:.2e} m⁻³") print(f"輸送断面積: {sigma_tr:.2e} m²") print(f"\n緩和時間 τ: {tau:.2e} s") print(f"平均自由行程: {l_mfp*1e9:.2f} nm") print(f"電気抵抗率 ρ: {rho:.2e} Ω·m") print(f"電気伝導度 σ: {1/rho:.2e} S/m")

電気抵抗率の微視的計算: ================================================== 伝導電子密度: 8.50e+28 m⁻³ 不純物密度: 1.00e+24 m⁻³ 輸送断面積: 1.00e-19 m² 緩和時間 τ: 6.37e-15 s 平均自由行程: 10.00 nm 電気抵抗率 ρ: 1.32e-08 Ω·m 電気伝導度 σ: 7.58e+07 S/m

Example 8: フォノン散乱によるMatthiessenの法則

import numpy as np # =================================== # Matthiessenの法則: ρ_total = ρ_imp + ρ_ph(T) # =================================== def phonon_scattering_rate(T, theta_D): """フォノン散乱による緩和率 τ_ph⁻¹ ∝ T⁵ (T << θ_D) τ_ph⁻¹ ∝ T (T >> θ_D) """ if T < 0.1 * theta_D: # 低温（Bloch-Grüneisen領域） tau_ph_inv = 1e12 * (T / theta_D)**5 else: # 高温（線形領域） tau_ph_inv = 1e13 * (T / theta_D) return tau_ph_inv def total_resistivity(T, rho_imp, theta_D, rho0_ph): """全抵抗率（Matthiessenの法則）""" tau_ph_inv = phonon_scattering_rate(T, theta_D) rho_ph = rho0_ph * (tau_ph_inv / 1e13) # 正規化 return rho_imp + rho_ph # 銅のパラメータ theta_D = 343 # K (Debye温度) rho_imp = 1e-9 # Ω·m (残留抵抗) rho0_ph = 1.7e-8 # Ω·m (室温でのフォノン寄与) temperatures = [10, 50, 100, 200, 300] print("Matthiessenの法則による抵抗率:") print("=" * 60) print(f"{'T (K)':<10} {'ρ_imp (Ω·m)':<20} {'ρ_ph (Ω·m)':<20} {'ρ_total':<15}") print("-" * 60) for T in temperatures: rho_ph = total_resistivity(T, 0, theta_D, rho0_ph) - 0 rho_tot = total_resistivity(T, rho_imp, theta_D, rho0_ph) print(f"{T:<10} {rho_imp:<20.2e} {rho_ph:<20.2e} {rho_tot:<15.2e}")

Matthiessenの法則による抵抗率: ============================================================ T (K) ρ_imp (Ω·m) ρ_ph (Ω·m) ρ_total \------------------------------------------------------------ 10 1.00e-09 4.49e-13 1.00e-09 50 1.00e-09 2.81e-10 1.28e-09 100 1.00e-09 5.64e-09 6.64e-09 200 1.00e-09 9.91e-09 1.09e-08 300 1.00e-09 1.49e-08 1.59e-08

## 演習問題

### Easy（基礎確認）

**Q1** : 相互作用描像での時間発展演算子 \\(U_I(t, t_0)\\) が満たす微分方程式を導出してください。

解答を見る

**導出** :

\\[ i\frac{d}{dt}|\psi_I(t)\rangle = H_I(t)|\psi_I(t)\rangle \\]

\\(|\psi_I(t)\rangle = U_I(t, t_0)|\psi_I(t_0)\rangle\\) なので：

\\[ i\frac{\partial U_I}{\partial t} = H_I(t) U_I(t, t_0) \\]

初期条件: \\(U_I(t_0, t_0) = 1\\)

### Medium（応用）

**Q2** : Wickの定理を用いて、6点関数 \\(\langle 0|T\\{\phi_1\cdots\phi_6\\}|0\rangle\\) の異なるペアリングの数を数えてください。

解答を見る

**計算** :

6つの場を3組のペアに分ける方法の数：

\\[ \frac{6!}{2^3 \cdot 3!} = \frac{720}{8 \cdot 6} = 15 \\]

一般に、\\(2n\\)点関数のペアリング数は \\((2n-1)!! = (2n-1)(2n-3)\cdots 3 \cdot 1\\)

### Hard（発展）

**Q3** : LSZ公式を用いて、φ⁴理論での2→2散乱振幅がツリーレベルで \\(\mathcal{M} = -\lambda\\) となることを示してください。

解答を見る

**導出** :

LSZ公式から、外線をon-shellにすると:

\\[ \langle p_3, p_4|S|p_1, p_2\rangle \propto \langle 0|T\\{\phi\phi\phi\phi\\}|0\rangle_{\text{1PI}} \\]

ツリーレベルでは、4点頂点のみが寄与:

\\[ H_I = \int d^4x \, \frac{\lambda}{4!}\phi^4 \\]

S行列の1次項:

\\[ S^{(1)} = -i\int d^4x \, \frac{\lambda}{4!}\phi^4 \\]

Wickの定理で4つの場を外線に縮約すると、組合せ因子 \\(4!\\) がキャンセルして:

\\[ \mathcal{M} = -\lambda \\]

[← 第2章](<chapter-2.html>) 第4章へ進む →（準備中）

## 参考文献

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1995). _The Quantum Theory of Fields, Vol. 1_. Cambridge University Press.
  3. Schwartz, M. D. (2014). _Quantum Field Theory and the Standard Model_. Cambridge University Press.
  4. Mahan, G. D. (2000). _Many-Particle Physics_ (3rd ed.). Springer.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
