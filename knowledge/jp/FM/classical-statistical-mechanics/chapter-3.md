---
title: "第3章: グランドカノニカル集団と化学ポテンシャル"
chapter_title: "第3章: グランドカノニカル集団と化学ポテンシャル"
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/classical-statistical-mechanics/chapter-3.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [古典統計力学入門](<index.html>) > 第3章 

## 🎯 学習目標

  * グランドカノニカル集団の概念と適用場面を理解する
  * 大分配関数と化学ポテンシャルの関係を学ぶ
  * 粒子数揺らぎの計算方法を習得する
  * Langmuir吸着等温線を導出し、材料への応用を学ぶ
  * 格子気体モデルとIsingモデルの対応を理解する
  * 量子統計（Fermi-Dirac分布、Bose-Einstein分布）の基礎を学ぶ
  * 化学反応平衡への応用を理解する
  * 材料科学における吸着現象をシミュレートする

## 📖 グランドカノニカル集団とは

### グランドカノニカル集団（Grand Canonical Ensemble）

温度 \\(T\\) と化学ポテンシャル \\(\mu\\) が固定され、粒子とエネルギーを粒子浴と交換できる系。

  * 体積 \\(V\\)、温度 \\(T\\)、化学ポテンシャル \\(\mu\\) が固定
  * 粒子数 \\(N\\) とエネルギー \\(E\\) が揺らぐ
  * 吸着、化学反応、開放系で重要

**大分配関数（Grand partition function）** :

\\[ \Xi(\mu, V, T) = \sum_{N=0}^\infty \sum_i e^{\beta(\mu N - E_i)} = \sum_{N=0}^\infty e^{\beta\mu N} Z(N, V, T) \\]

ここで \\(\beta = 1/(k_B T)\\)、\\(Z(N, V, T)\\) はカノニカル分配関数です。

### グランドポテンシャルと化学ポテンシャル

グランドポテンシャル（Grand potential）:

\\[ \Omega = -k_B T \ln \Xi = F - \mu N \\]

熱力学量の導出：

  * 平均粒子数: \\(\langle N \rangle = -\frac{\partial \Omega}{\partial \mu} = \frac{1}{\beta}\frac{\partial \ln \Xi}{\partial \mu}\\)
  * 圧力: \\(P = -\frac{\partial \Omega}{\partial V}\\)
  * 粒子数揺らぎ: \\(\langle (\Delta N)^2 \rangle = k_B T \left(\frac{\partial \langle N \rangle}{\partial \mu}\right)_{V,T}\\)

**化学ポテンシャル** \\(\mu\\) は「粒子を1つ追加するのに必要な自由エネルギー」です。

## 💻 例題3.1: 理想気体の大分配関数

### 理想気体のグランドカノニカル計算

1粒子分配関数 \\(z_1 = V/\lambda_T^3\\)（\\(\lambda_T\\): 熱的de Broglie波長）を用いて：

\\[ \Xi = \sum_{N=0}^\infty \frac{(z_1 e^{\beta\mu})^N}{N!} = \exp(z_1 e^{\beta\mu}) \\]

平均粒子数:

\\[ \langle N \rangle = z_1 e^{\beta\mu} \\]

これから化学ポテンシャルを解くと:

\\[ \mu = k_B T \ln\left(\frac{\langle N \rangle \lambda_T^3}{V}\right) \\]

Python実装: 理想気体の化学ポテンシャル

import numpy as np import matplotlib.pyplot as plt k_B = 1.380649e-23 # J/K h = 6.62607015e-34 # J·s m_Ar = 6.63e-26 # Ar原子の質量 (kg) def thermal_wavelength(T, m, h, k_B): """熱的de Broglie波長""" return h / np.sqrt(2 * np.pi * m * k_B * T) def chemical_potential_ideal_gas(N, V, T, m, h, k_B): """理想気体の化学ポテンシャル""" lambda_T = thermal_wavelength(T, m, h, k_B) return k_B * T * np.log((N / V) * lambda_T**3) def fugacity(mu, T, k_B): """フガシティ z = exp(βμ)""" return np.exp(mu / (k_B * T)) # 標準状態付近のArガス N_A = 6.022e23 N_molar = N_A V_molar = 0.0224 # m³ T_range = np.linspace(100, 1000, 100) # 密度依存性（温度固定） T_fixed = 300 # K V_range = np.linspace(0.001, 0.1, 100) # m³ n_range = N_molar / V_range # 数密度 mu_vs_V = [chemical_potential_ideal_gas(N_molar, V, T_fixed, m_Ar, h, k_B) for V in V_range] # 温度依存性（体積固定） mu_vs_T = [chemical_potential_ideal_gas(N_molar, V_molar, T, m_Ar, h, k_B) for T in T_range] # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 化学ポテンシャルの密度依存性 ax1 = axes[0, 0] ax1.plot(n_range / 1e25, np.array(mu_vs_V) / (k_B * T_fixed), 'b-', linewidth=2) ax1.set_xlabel('Number density (10²⁵ m⁻³)') ax1.set_ylabel('μ / (k_B T)') ax1.set_title(f'化学ポテンシャルの密度依存性（T = {T_fixed} K）') ax1.grid(True, alpha=0.3) # 化学ポテンシャルの温度依存性 ax2 = axes[0, 1] ax2.plot(T_range, np.array(mu_vs_T) / 1e-21, 'r-', linewidth=2) ax2.set_xlabel('Temperature (K)') ax2.set_ylabel('μ (10⁻²¹ J)') ax2.set_title('化学ポテンシャルの温度依存性（V = 22.4 L）') ax2.grid(True, alpha=0.3) # フガシティ ax3 = axes[1, 0] z_vals = [fugacity(mu, T_fixed, k_B) for mu in mu_vs_V] ax3.semilogy(n_range / 1e25, z_vals, 'g-', linewidth=2) ax3.set_xlabel('Number density (10²⁵ m⁻³)') ax3.set_ylabel('Fugacity z = exp(βμ)') ax3.set_title(f'フガシティ（T = {T_fixed} K）') ax3.grid(True, alpha=0.3, which='both') # 粒子数揺らぎ ax4 = axes[1, 1] # <(ΔN)²> = k_B T (∂/∂μ) =  (理想気体) fluctuation_ratio = 1 / np.sqrt(n_range * V_molar) # σ_N /  = 1/√ ax4.loglog(n_range / 1e25, fluctuation_ratio, 'm-', linewidth=2) ax4.set_xlabel('Number density (10²⁵ m⁻³)') ax4.set_ylabel('σ_N / ') ax4.set_title('相対揺らぎ') ax4.grid(True, alpha=0.3, which='both') plt.tight_layout() plt.savefig('stat_mech_grand_canonical_ideal_gas.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 理想気体のグランドカノニカル統計 ===\n") print(f"Arガス（1 mol, 標準状態付近）\n") T_std = 298.15 # K V_std = V_molar mu_std = chemical_potential_ideal_gas(N_molar, V_std, T_std, m_Ar, h, k_B) lambda_T_std = thermal_wavelength(T_std, m_Ar, h, k_B) print(f"T = {T_std} K, V = {V_std*1000:.1f} L") print(f"熱的de Broglie波長: λ_T = {lambda_T_std*1e9:.4f} nm") print(f"化学ポテンシャル: μ = {mu_std:.6e} J") print(f" μ/(k_B T) = {mu_std/(k_B*T_std):.4f}") print(f"フガシティ: z = {fugacity(mu_std, T_std, k_B):.6e}") print(f"\n相対揺らぎ: σ_N/ = 1/√ = {1/np.sqrt(N_molar):.6e}") print(f" = {1/np.sqrt(N_molar):.2e} （極めて小さい）") 

## 💻 例題3.2: Langmuir吸着等温線

### Langmuir吸着モデル

\\(M\\)個の吸着サイトがあり、各サイトは最大1個の粒子を吸着できる:

大分配関数（1サイトあたり）:

\\[ \xi = 1 + e^{\beta(\mu - \varepsilon)} \\]

ここで \\(\varepsilon\\) は吸着エネルギー（負の値）。

被覆率（coverage） \\(\theta\\):

\\[ \theta = \frac{\langle N \rangle}{M} = \frac{e^{\beta(\mu - \varepsilon)}}{1 + e^{\beta(\mu - \varepsilon)}} = \frac{1}{1 + e^{-\beta(\mu - \varepsilon)}} \\]

気相圧力 \\(P\\) との関係（\\(\mu = k_B T \ln(P/P_0)\\)）:

\\[ \theta = \frac{KP}{1 + KP}, \quad K = e^{-\beta\varepsilon} / P_0 \\]

これが**Langmuir吸着等温線** です。

Python実装: Langmuir吸着等温線

import numpy as np import matplotlib.pyplot as plt k_B = 1.380649e-23 # J/K def langmuir_coverage(P, K): """Langmuir吸着等温線""" return K * P / (1 + K * P) def freundlich_isotherm(P, k_F, n): """Freundlich吸着等温線（経験式、多層吸着）""" return k_F * P**(1/n) def bet_isotherm(P, P0, c): """BET吸着等温線（多層吸着）""" x = P / P0 return (c * x) / ((1 - x) * (1 + (c - 1) * x)) # 異なる吸着エネルギー T = 300 # K epsilon_values = [-0.1, -0.3, -0.5, -0.8] # eV epsilon_J = [eps * 1.602e-19 for eps in epsilon_values] # J P0 = 1e5 # Pa (標準圧力) K_values = [np.exp(-eps / (k_B * T)) / P0 for eps in epsilon_J] P_range = np.logspace(-2, 6, 200) # Pa fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Langmuir等温線（線形プロット） ax1 = axes[0, 0] colors = ['blue', 'green', 'orange', 'red'] for epsilon, K, color in zip(epsilon_values, K_values, colors): theta = langmuir_coverage(P_range, K) ax1.plot(P_range / 1e3, theta, color=color, linewidth=2, label=f'ε = {epsilon} eV') ax1.set_xlabel('Pressure (kPa)') ax1.set_ylabel('Coverage θ') ax1.set_title('Langmuir吸着等温線') ax1.legend() ax1.grid(True, alpha=0.3) ax1.set_xlim([0, 100]) # Langmuir等温線（対数プロット） ax2 = axes[0, 1] for epsilon, K, color in zip(epsilon_values, K_values, colors): theta = langmuir_coverage(P_range, K) ax2.semilogx(P_range, theta, color=color, linewidth=2, label=f'ε = {epsilon} eV') ax2.set_xlabel('Pressure (Pa)') ax2.set_ylabel('Coverage θ') ax2.set_title('Langmuir等温線（対数スケール）') ax2.legend() ax2.grid(True, alpha=0.3) # Langmuirプロット（P/θ vs P） ax3 = axes[1, 0] K_demo = K_values[2] theta_demo = langmuir_coverage(P_range, K_demo) # P/θ = 1/K + P (直線になる) P_over_theta = P_range / theta_demo ax3.plot(P_range / 1e3, P_over_theta / 1e3, 'b-', linewidth=2) ax3.set_xlabel('Pressure (kPa)') ax3.set_ylabel('P/θ (kPa)') ax3.set_title('Langmuirプロット（直線性の確認）') ax3.grid(True, alpha=0.3) ax3.set_xlim([0, 100]) # 温度依存性 ax4 = axes[1, 1] temperatures = [250, 300, 350, 400] epsilon_fixed = epsilon_J[2] P_fixed_range = np.logspace(2, 5, 100) for T_val, color in zip(temperatures, colors): K_T = np.exp(-epsilon_fixed / (k_B * T_val)) / P0 theta_T = langmuir_coverage(P_fixed_range, K_T) ax4.semilogx(P_fixed_range, theta_T, color=color, linewidth=2, label=f'T = {T_val} K') ax4.set_xlabel('Pressure (Pa)') ax4.set_ylabel('Coverage θ') ax4.set_title(f'温度依存性（ε = {epsilon_values[2]} eV）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_langmuir_adsorption.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== Langmuir吸着等温線 ===\n") print(f"温度 T = {T} K\n") for epsilon, K in zip(epsilon_values, K_values): print(f"吸着エネルギー ε = {epsilon} eV:") print(f" 吸着平衡定数 K = {K:.6e} Pa⁻¹") # 半被覆圧力（θ = 0.5） P_half = 1 / K print(f" 半被覆圧力 P₁/₂ = {P_half:.2e} Pa = {P_half/1e3:.2f} kPa\n") print("Langmuir等温線の特徴:") print(" - 低圧: θ ≈ KP（ Henry's law）") print(" - 高圧: θ → 1（飽和）") print(" - 単分子層吸着を仮定") 

## 💻 例題3.3: 格子気体モデル

### 格子気体モデル

\\(M\\)個の格子サイトがあり、各サイトは占有（粒子あり）または空（粒子なし）の2状態：

エネルギー: \\(E = -\varepsilon_0 N - \frac{1}{2}J \sum_{\langle i,j \rangle} n_i n_j\\)

ここで \\(n_i = 0, 1\\) は占有数、\\(J\\) は最隣接相互作用。

**Isingモデルとの対応** : \\(s_i = 2n_i - 1\\) と置換すると、スピン系のIsingモデルに帰着します。

平均場近似では、\\(\langle n \rangle = \theta\\)（被覆率）として:

\\[ \theta = \frac{1}{1 + e^{-\beta(\mu - \varepsilon_0 + Jz\theta)}} \\]

ここで \\(z\\) は配位数です。

Python実装: 格子気体の平均場理論

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import fsolve k_B = 1.380649e-23 # J/K def mean_field_equation(theta, mu, epsilon_0, J, z, T, k_B): """平均場方程式""" beta = 1 / (k_B * T) effective_mu = mu - epsilon_0 + J * z * theta return theta - 1 / (1 + np.exp(-beta * effective_mu)) def solve_mean_field_coverage(mu, epsilon_0, J, z, T, k_B, theta_init=0.5): """平均場方程式を解いて被覆率を求める""" sol = fsolve(mean_field_equation, theta_init, args=(mu, epsilon_0, J, z, T, k_B)) return sol[0] # パラメータ epsilon_0 = -0.5 * 1.602e-19 # 吸着エネルギー (J) z = 4 # 2次元正方格子の配位数 T = 300 # K # 相互作用強度の異なるケース J_values = [0, -0.1e-19, -0.2e-19, -0.3e-19] # J (引力的) J_eV = [J / 1.602e-19 for J in J_values] colors = ['blue', 'green', 'orange', 'red'] # 化学ポテンシャル範囲 mu_range = np.linspace(-1.0e-19, 0.5e-19, 200) fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 被覆率 vs 化学ポテンシャル ax1 = axes[0, 0] for J, J_ev, color in zip(J_values, J_eV, colors): theta_list = [] for mu in mu_range: theta = solve_mean_field_coverage(mu, epsilon_0, J, z, T, k_B) theta_list.append(theta) ax1.plot(mu_range / 1.602e-19, theta_list, color=color, linewidth=2, label=f'J = {J_ev:.2f} eV') ax1.set_xlabel('Chemical potential μ (eV)') ax1.set_ylabel('Coverage θ') ax1.set_title('被覆率（平均場理論）') ax1.legend() ax1.grid(True, alpha=0.3) # 温度依存性 ax2 = axes[0, 1] temperatures = [200, 300, 400, 500] J_fixed = J_values[2] mu_fixed = -0.3e-19 # J for T_val, color in zip(temperatures, colors): theta_T_list = [] for mu in mu_range: theta = solve_mean_field_coverage(mu, epsilon_0, J_fixed, z, T_val, k_B) theta_T_list.append(theta) ax2.plot(mu_range / 1.602e-19, theta_T_list, color=color, linewidth=2, label=f'T = {T_val} K') ax2.set_xlabel('Chemical potential μ (eV)') ax2.set_ylabel('Coverage θ') ax2.set_title(f'温度依存性（J = {J_fixed/1.602e-19:.2f} eV）') ax2.legend() ax2.grid(True, alpha=0.3) # 相互作用による相転移（低温） ax3 = axes[1, 0] T_low = 200 # K J_strong = -0.4e-19 # 強い引力 # 自己無撞着方程式の複数解 theta_init_values = [0.1, 0.5, 0.9] markers = ['o', 's', '^'] for mu in mu_range[::10]: for theta_init, marker in zip(theta_init_values, markers): try: theta_sol = solve_mean_field_coverage(mu, epsilon_0, J_strong, z, T_low, k_B, theta_init) ax3.plot(mu / 1.602e-19, theta_sol, marker, color='blue', markersize=4) except: pass ax3.set_xlabel('Chemical potential μ (eV)') ax3.set_ylabel('Coverage θ') ax3.set_title(f'相転移（T = {T_low} K, J = {J_strong/1.602e-19:.2f} eV）') ax3.grid(True, alpha=0.3) # 圧縮率 ax4 = axes[1, 1] J_demo = J_values[1] mu_demo_range = np.linspace(-0.8e-19, 0.2e-19, 100) theta_demo = [solve_mean_field_coverage(mu, epsilon_0, J_demo, z, T, k_B) for mu in mu_demo_range] # 圧縮率 κ ∝ ∂θ/∂μ dtheta_dmu = np.gradient(theta_demo, mu_demo_range) ax4.plot(mu_demo_range / 1.602e-19, dtheta_dmu * 1.602e-19, 'purple', linewidth=2) ax4.set_xlabel('Chemical potential μ (eV)') ax4.set_ylabel('∂θ/∂μ (eV⁻¹)') ax4.set_title('応答関数（圧縮率）') ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_lattice_gas.png', dpi=300, bbox_inches='tight') plt.show() print("=== 格子気体モデル（平均場理論）===\n") print(f"吸着エネルギー ε₀ = {epsilon_0/1.602e-19:.2f} eV") print(f"配位数 z = {z}") print(f"温度 T = {T} K\n") for J, J_ev in zip(J_values, J_eV): print(f"相互作用 J = {J_ev:.2f} eV:") # μ = 0 での被覆率 theta_0 = solve_mean_field_coverage(0, epsilon_0, J, z, T, k_B) print(f" μ=0 での被覆率: θ = {theta_0:.4f}\n") print("Isingモデルとの対応:") print(" n_i = 0, 1 → s_i = 2n_i - 1 = -1, +1") print(" 格子気体 ↔ スピン系") 

## 💻 例題3.4: Fermi-Dirac分布とBose-Einstein分布

### 量子統計

**Fermi-Dirac分布** （フェルミオン）:

\\[ \langle n_i \rangle = \frac{1}{e^{\beta(\varepsilon_i - \mu)} + 1} \\]

Pauli排他原理により、各状態に最大1個の粒子。

**Bose-Einstein分布** （ボゾン）:

\\[ \langle n_i \rangle = \frac{1}{e^{\beta(\varepsilon_i - \mu)} - 1} \\]

各状態に複数の粒子が占有可能。

**古典極限** （高温または低密度）:

\\[ \langle n_i \rangle \approx e^{-\beta(\varepsilon_i - \mu)} \quad (\text{Maxwell-Boltzmann分布}) \\]

Python実装: 量子統計分布の比較

import numpy as np import matplotlib.pyplot as plt k_B = 1.380649e-23 # J/K def fermi_dirac(epsilon, mu, T, k_B): """Fermi-Dirac分布""" beta = 1 / (k_B * T) x = beta * (epsilon - mu) # オーバーフロー回避 if x > 100: return 0 elif x < -100: return 1 return 1 / (np.exp(x) + 1) def bose_einstein(epsilon, mu, T, k_B): """Bose-Einstein分布""" beta = 1 / (k_B * T) x = beta * (epsilon - mu) if x <= 0: return np.inf # μ < ε が必要 if x < 0.01: return 1 / x # 近似 return 1 / (np.exp(x) - 1) def maxwell_boltzmann(epsilon, mu, T, k_B): """Maxwell-Boltzmann分布""" beta = 1 / (k_B * T) x = beta * (epsilon - mu) if x > 100: return 0 return np.exp(-x) # エネルギー範囲（化学ポテンシャルを0とする） mu = 0 epsilon_range = np.linspace(-5, 5, 500) # k_B T 単位 # 異なる温度（k_B T 単位で規格化） k_B_T = 1 # 規格化単位 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # T = 1 (k_B T 単位) での分布 ax1 = axes[0, 0] n_FD = [fermi_dirac(eps * k_B_T, mu, 1, k_B) for eps in epsilon_range] n_BE = [bose_einstein(eps * k_B_T, mu, 1, k_B) if eps > 0 else 0 for eps in epsilon_range] n_MB = [maxwell_boltzmann(eps * k_B_T, mu, 1, k_B) for eps in epsilon_range] ax1.plot(epsilon_range, n_FD, 'b-', linewidth=2, label='Fermi-Dirac') ax1.plot(epsilon_range, n_BE, 'r-', linewidth=2, label='Bose-Einstein') ax1.plot(epsilon_range, n_MB, 'g--', linewidth=2, label='Maxwell-Boltzmann') ax1.axvline(0, color='k', linestyle=':', linewidth=1, label='μ = 0') ax1.set_xlabel('(ε - μ) / (k_B T)') ax1.set_ylabel('Occupation number ⟨n⟩') ax1.set_title('量子統計分布（T = 1）') ax1.legend() ax1.grid(True, alpha=0.3) ax1.set_ylim([0, 3]) # 低温での Fermi-Dirac ax2 = axes[0, 1] temperatures_FD = [0.1, 0.5, 1.0, 2.0] colors_FD = ['blue', 'green', 'orange', 'red'] for T_val, color in zip(temperatures_FD, colors_FD): n_FD_T = [fermi_dirac(eps * k_B_T, mu, T_val, k_B) for eps in epsilon_range] ax2.plot(epsilon_range, n_FD_T, color=color, linewidth=2, label=f'k_B T = {T_val}') ax2.axvline(0, color='k', linestyle=':', linewidth=1) ax2.set_xlabel('(ε - μ) / (E_F)') ax2.set_ylabel('⟨n⟩') ax2.set_title('Fermi-Dirac分布の温度依存性') ax2.legend() ax2.grid(True, alpha=0.3) # Bose-Einstein分布とBose凝縮 ax3 = axes[1, 0] epsilon_BE = np.linspace(0.01, 5, 100) temperatures_BE = [0.5, 1.0, 2.0, 5.0] for T_val, color in zip(temperatures_BE, colors_FD): n_BE_T = [bose_einstein(eps * k_B_T, mu * 0.9, T_val, k_B) for eps in epsilon_BE] ax3.semilogy(epsilon_BE, n_BE_T, color=color, linewidth=2, label=f'k_B T = {T_val}') ax3.set_xlabel('(ε - μ) / (k_B T)') ax3.set_ylabel('⟨n⟩ (log scale)') ax3.set_title('Bose-Einstein分布') ax3.legend() ax3.grid(True, alpha=0.3, which='both') # 古典極限の検証 ax4 = axes[1, 1] epsilon_classical = np.linspace(0, 10, 100) T_high = 5 # 高温 n_FD_high = [fermi_dirac(eps * k_B_T, mu, T_high, k_B) for eps in epsilon_classical] n_BE_high = [bose_einstein(eps * k_B_T, mu * 0.5, T_high, k_B) for eps in epsilon_classical] n_MB_high = [maxwell_boltzmann(eps * k_B_T, mu * 0.5, T_high, k_B) for eps in epsilon_classical] ax4.semilogy(epsilon_classical, n_FD_high, 'b-', linewidth=2, label='Fermi-Dirac') ax4.semilogy(epsilon_classical, n_BE_high, 'r-', linewidth=2, label='Bose-Einstein') ax4.semilogy(epsilon_classical, n_MB_high, 'g--', linewidth=2, label='Maxwell-Boltzmann') ax4.set_xlabel('(ε - μ) / (k_B T)') ax4.set_ylabel('⟨n⟩ (log scale)') ax4.set_title(f'古典極限（k_B T = {T_high}）') ax4.legend() ax4.grid(True, alpha=0.3, which='both') plt.tight_layout() plt.savefig('stat_mech_quantum_statistics.png', dpi=300, bbox_inches='tight') plt.show() print("=== 量子統計 ===\n") print("Fermi-Dirac分布:") print(" - フェルミオン（電子、陽子、中性子など）") print(" - Pauli排他原理: ⟨n⟩ ≤ 1") print(" - T = 0 でステップ関数（Fermi面）\n") print("Bose-Einstein分布:") print(" - ボゾン（光子、フォノン、He-4など）") print(" - 複数占有可能: ⟨n⟩ ≥ 0") print(" - 低温でBose凝縮（μ → ε₀）\n") print("古典極限（高温）:") print(" - e^{β(ε-μ)} >> 1 のとき") print(" - FD ≈ BE ≈ MB") print(" - 量子統計 → 古典統計\n") # 数値例 epsilon_test = 2 * k_B_T T_test = 300 # K print(f"数値例（ε-μ = 2k_BT, T = {T_test} K）:") n_FD_test = fermi_dirac(epsilon_test, mu, T_test, k_B) n_BE_test = bose_einstein(epsilon_test, mu * 0.5, T_test, k_B) n_MB_test = maxwell_boltzmann(epsilon_test, mu * 0.5, T_test, k_B) print(f" Fermi-Dirac: ⟨n⟩ = {n_FD_test:.6f}") print(f" Bose-Einstein: ⟨n⟩ = {n_BE_test:.6f}") print(f" Maxwell-Boltzmann: ⟨n⟩ = {n_MB_test:.6f}") 

## 📚 まとめ

  * **グランドカノニカル集団** は粒子数が揺らぐ開放系を記述し、化学ポテンシャルが重要な変数
  * **大分配関数** \\(\Xi\\) からグランドポテンシャル \\(\Omega\\) を通じて全熱力学量が導出可能
  * **化学ポテンシャル** は粒子を追加する際の自由エネルギー変化で、平衡条件を決定
  * **Langmuir吸着等温線** は単分子層吸着を記述し、材料科学で広く応用される
  * **格子気体モデル** はIsingモデルと等価で、相互作用系の相転移を記述
  * **Fermi-Dirac分布** はフェルミオンの量子統計で、電子系・金属の記述に不可欠
  * **Bose-Einstein分布** はボゾンの量子統計で、フォノン・光子系を記述
  * 高温極限で量子統計は古典的Maxwell-Boltzmann分布に帰着する

### 💡 演習問題

  1. **粒子数揺らぎ** : 理想気体のグランドカノニカル集団で \\(\langle (\Delta N)^2 \rangle / \langle N \rangle\\) を計算し、\\(N \to \infty\\) で相対揺らぎがゼロに近づくことを示せ。
  2. **BET吸着** : 多層吸着のBET理論を実装し、Langmuir等温線との違いを可視化せよ。
  3. **格子気体の相転移** : 平均場理論で臨界温度 \\(T_c\\) を求め、\\(J\\) と \\(z\\) の依存性を調べよ。
  4. **Fermiエネルギー** : 3次元電子ガスのFermi energy \\(E_F\\) を密度の関数として計算し、金属の典型値と比較せよ。
  5. **Planck分布** : 光子のBose-Einstein分布から黒体放射のPlanck分布を導出せよ（\\(\mu = 0\\)）。

[← 第2章: カノニカル集団と分配関数](<chapter-2.html>) 第4章: 相互作用系と相転移 →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
