---
title: "第4章: 相互作用系と相転移"
chapter_title: "第4章: 相互作用系と相転移"
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/classical-statistical-mechanics/chapter-4.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [古典統計力学入門](<index.html>) > 第4章 

## 🎯 学習目標

  * Isingモデルと強磁性の統計力学を理解する
  * 平均場近似の手法を習得する
  * 1次相転移と2次相転移の違いを学ぶ
  * 臨界現象と臨界指数を理解する
  * Landau理論による秩序-無秩序転移の記述を学ぶ
  * van der Waals気体と液-気相転移を理解する
  * 秩序パラメータの概念を習得する
  * 材料科学における相転移現象を理解する

## 📖 Isingモデルと強磁性

### Isingモデル

スピン \\(s_i = \pm 1\\) が格子点に配置されたモデル：

\\[ H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i \\]

  * \\(J > 0\\): 強磁性相互作用（平行配置が有利）
  * \\(J < 0\\): 反強磁性相互作用（反平行配置が有利）
  * \\(h\\): 外部磁場
  * \\(\langle i,j \rangle\\): 最隣接ペアの和

**磁化（秩序パラメータ）** :

\\[ m = \frac{1}{N}\sum_i \langle s_i \rangle \\]

低温で \\(m \neq 0\\)（強磁性相）、高温で \\(m = 0\\)（常磁性相）。

### 平均場近似（Mean Field Approximation）

各スピンが平均磁場 \\(h_{\text{eff}} = Jz\langle s \rangle + h\\) を感じると近似：

\\[ \langle s_i \rangle = \tanh(\beta h_{\text{eff}}) = \tanh(\beta(Jzm + h)) \\]

ここで \\(z\\) は配位数、\\(m = \langle s_i \rangle\\) は磁化です。

**自己無撞着方程式** :

\\[ m = \tanh(\beta Jzm + \beta h) \\]

**臨界温度** （\\(h = 0\\)）:

\\[ T_c = \frac{Jz}{k_B} \\]

\\(T < T_c\\) で自発磁化 \\(m \neq 0\\) が出現します。

## 💻 例題4.1: Ising平均場理論と自発磁化

### 自発磁化の温度依存性

\\(h = 0\\) での磁化:

\\[ m = \tanh(\beta Jzm) \\]

臨界温度近傍で \\(m \approx \sqrt{3(1 - T/T_c)}\\) （臨界指数 \\(\beta = 1/2\\)）。

Python実装: Ising平均場理論

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import fsolve k_B = 1.380649e-23 # J/K def mean_field_ising_equation(m, T, J, z, h, k_B): """平均場方程式""" if T == 0: return m - np.sign(J*z*m + h) beta = 1 / (k_B * T) return m - np.tanh(beta * (J * z * m + h)) def solve_magnetization(T, J, z, h, k_B, m_init=0.5): """自己無撞着方程式を解く""" sol = fsolve(mean_field_ising_equation, m_init, args=(T, J, z, h, k_B)) return sol[0] # パラメータ J = 1e-21 # J (相互作用強度) z = 4 # 2次元正方格子 T_c = J * z / k_B # 臨界温度 # 温度範囲 T_range = np.linspace(0.01, 2.5 * T_c, 200) # h = 0 での自発磁化 m_positive = [] m_negative = [] for T in T_range: # 正の解 m_pos = solve_magnetization(T, J, z, 0, k_B, m_init=0.9) m_positive.append(m_pos) # 負の解 m_neg = solve_magnetization(T, J, z, 0, k_B, m_init=-0.9) m_negative.append(m_neg) # 外部磁場ありの場合 h_values = [0, 0.1e-21, 0.3e-21, 0.5e-21] h_eV = [h / 1.602e-19 for h in h_values] colors = ['blue', 'green', 'orange', 'red'] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 自発磁化の温度依存性 ax1 = axes[0, 0] ax1.plot(T_range / T_c, m_positive, 'b-', linewidth=2, label='m > 0') ax1.plot(T_range / T_c, m_negative, 'r--', linewidth=2, label='m < 0') ax1.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c') ax1.set_xlabel('T / T_c') ax1.set_ylabel('Magnetization m') ax1.set_title('自発磁化（h = 0）') ax1.legend() ax1.grid(True, alpha=0.3) # 外部磁場の効果 ax2 = axes[0, 1] for h, h_ev, color in zip(h_values, h_eV, colors): m_h = [solve_magnetization(T, J, z, h, k_B) for T in T_range] ax2.plot(T_range / T_c, m_h, color=color, linewidth=2, label=f'h = {h_ev:.2f} eV') ax2.axvline(1, color='k', linestyle=':', linewidth=1) ax2.set_xlabel('T / T_c') ax2.set_ylabel('Magnetization m') ax2.set_title('外部磁場による磁化') ax2.legend() ax2.grid(True, alpha=0.3) # 磁化率（h = 0での応答） ax3 = axes[1, 0] # χ = ∂m/∂h |_{h=0} epsilon = 1e-23 # 微小磁場 chi_values = [] for T in T_range: m0 = solve_magnetization(T, J, z, 0, k_B) m_eps = solve_magnetization(T, J, z, epsilon, k_B) chi = (m_eps - m0) / epsilon chi_values.append(chi) ax3.plot(T_range / T_c, np.array(chi_values) * 1e21, 'purple', linewidth=2) ax3.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c') ax3.set_xlabel('T / T_c') ax3.set_ylabel('χ (10⁻²¹ J⁻¹)') ax3.set_title('磁化率（T_c で発散）') ax3.legend() ax3.grid(True, alpha=0.3) # 臨界領域の拡大 ax4 = axes[1, 1] T_critical = np.linspace(0.5 * T_c, 1.5 * T_c, 100) m_critical = [solve_magnetization(T, J, z, 0, k_B, m_init=0.9) for T in T_critical] ax4.plot(T_critical / T_c, m_critical, 'b-', linewidth=2) ax4.axvline(1, color='k', linestyle=':', linewidth=2, label='T_c') ax4.set_xlabel('T / T_c') ax4.set_ylabel('Magnetization m') ax4.set_title('臨界領域の磁化') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_ising_mean_field.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== Ising平均場理論 ===\n") print(f"相互作用強度 J = {J:.2e} J = {J/1.602e-19:.4f} eV") print(f"配位数 z = {z}") print(f"臨界温度 T_c = {T_c:.2f} K\n") # 異なる温度での磁化 test_temps = [0.5 * T_c, 0.9 * T_c, 1.0 * T_c, 1.5 * T_c] for T_test in test_temps: m_test = solve_magnetization(T_test, J, z, 0, k_B, m_init=0.9) print(f"T = {T_test:.2f} K (T/T_c = {T_test/T_c:.2f}): m = {m_test:.4f}") print("\n臨界指数:") print(" 磁化: m ~ (T_c - T)^β, β = 1/2 (平均場)") print(" 磁化率: χ ~ |T - T_c|^{-γ}, γ = 1 (平均場)") 

## 💻 例題4.2: 相転移の分類と臨界現象

### 相転移の分類（Ehrenfestの分類）

**1次相転移** :

  * 自由エネルギーの1階微分（エントロピー、体積）が不連続
  * 潜熱が存在
  * 例: 液-気転移、融解、蒸発

**2次相転移** :

  * 自由エネルギーの2階微分（比熱、圧縮率）が不連続または発散
  * 潜熱なし、連続転移
  * 例: 強磁性転移、超伝導転移、超流動転移

**臨界指数** （2次相転移）:

  * \\(\beta\\): \\(m \sim |T - T_c|^\beta\\) （秩序パラメータ）
  * \\(\gamma\\): \\(\chi \sim |T - T_c|^{-\gamma}\\) （磁化率）
  * \\(\alpha\\): \\(C \sim |T - T_c|^{-\alpha}\\) （比熱）
  * \\(\delta\\): \\(m \sim h^{1/\delta}\\) at \\(T = T_c\\)

平均場理論: \\(\beta = 1/2, \gamma = 1, \alpha = 0, \delta = 3\\)

Python実装: 臨界指数の計算

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import fsolve, curve_fit k_B = 1.380649e-23 def solve_m(T, J, z, h, k_B): """平均場方程式を解く""" if T < 1e-10: return np.sign(h) if h != 0 else 1.0 beta = 1 / (k_B * T) eq = lambda m: m - np.tanh(beta * (J * z * m + h)) return fsolve(eq, 0.5 if h >= 0 else -0.5)[0] J = 1e-21 z = 4 T_c = J * z / k_B # 臨界指数 β の計算 T_below_Tc = np.linspace(0.5 * T_c, 0.999 * T_c, 50) m_beta = [solve_m(T, J, z, 0, k_B) for T in T_below_Tc] reduced_temp_beta = (T_c - T_below_Tc) / T_c # べき乗フィッティング def power_law_beta(t, beta_exp): return t**beta_exp # 対数フィットで指数を求める log_t = np.log(reduced_temp_beta) log_m = np.log(np.abs(m_beta)) poly_beta = np.polyfit(log_t, log_m, 1) beta_fitted = poly_beta[0] # 臨界指数 γ の計算（磁化率） T_chi = np.linspace(1.01 * T_c, 2 * T_c, 50) epsilon_h = 1e-23 chi_gamma = [] for T in T_chi: m0 = solve_m(T, J, z, 0, k_B) m_h = solve_m(T, J, z, epsilon_h, k_B) chi = (m_h - m0) / epsilon_h chi_gamma.append(chi) reduced_temp_gamma = (T_chi - T_c) / T_c log_t_gamma = np.log(reduced_temp_gamma) log_chi = np.log(chi_gamma) poly_gamma = np.polyfit(log_t_gamma, log_chi, 1) gamma_fitted = -poly_gamma[0] # 臨界指数 δ の計算（T = T_c） h_range = np.logspace(-24, -20, 30) m_delta = [solve_m(T_c, J, z, h, k_B) for h in h_range] log_h = np.log(h_range) log_m_delta = np.log(np.abs(m_delta)) poly_delta = np.polyfit(log_h, log_m_delta, 1) delta_fitted = 1 / poly_delta[0] # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 臨界指数 β ax1 = axes[0, 0] ax1.loglog(reduced_temp_beta, m_beta, 'bo', markersize=6, label='Data') ax1.loglog(reduced_temp_beta, reduced_temp_beta**0.5, 'r--', linewidth=2, label=f'β = 0.5 (theory)') ax1.loglog(reduced_temp_beta, reduced_temp_beta**beta_fitted, 'g-', linewidth=2, label=f'β = {beta_fitted:.3f} (fit)') ax1.set_xlabel('(T_c - T) / T_c') ax1.set_ylabel('Magnetization m') ax1.set_title('臨界指数 β') ax1.legend() ax1.grid(True, alpha=0.3, which='both') # 臨界指数 γ ax2 = axes[0, 1] ax2.loglog(reduced_temp_gamma, np.array(chi_gamma) * 1e21, 'go', markersize=6, label='Data') ax2.loglog(reduced_temp_gamma, 1e21 * (reduced_temp_gamma)**(-1), 'r--', linewidth=2, label='γ = 1 (theory)') ax2.loglog(reduced_temp_gamma, 1e21 * (reduced_temp_gamma)**(-gamma_fitted), 'b-', linewidth=2, label=f'γ = {gamma_fitted:.3f} (fit)') ax2.set_xlabel('(T - T_c) / T_c') ax2.set_ylabel('χ (10⁻²¹ J⁻¹)') ax2.set_title('臨界指数 γ') ax2.legend() ax2.grid(True, alpha=0.3, which='both') # 臨界指数 δ ax3 = axes[1, 0] ax3.loglog(h_range / 1.602e-19, np.abs(m_delta), 'mo', markersize=6, label='Data') ax3.loglog(h_range / 1.602e-19, (h_range/1e-21)**(1/3), 'r--', linewidth=2, label='δ = 3 (theory)') ax3.loglog(h_range / 1.602e-19, (h_range/1e-21)**(1/delta_fitted), 'c-', linewidth=2, label=f'δ = {delta_fitted:.3f} (fit)') ax3.set_xlabel('h (eV)') ax3.set_ylabel('m (at T = T_c)') ax3.set_title('臨界指数 δ') ax3.legend() ax3.grid(True, alpha=0.3, which='both') # 比熱（数値微分） ax4 = axes[1, 1] T_heat = np.linspace(0.5 * T_c, 1.5 * T_c, 100) dT = T_heat[1] - T_heat[0] free_energy = [] for T in T_heat: m_eq = solve_m(T, J, z, 0, k_B) # 自由エネルギー（平均場） if T > 0: beta = 1 / (k_B * T) f = -J * z * m_eq**2 / 2 - k_B * T * np.log(2 * np.cosh(beta * J * z * m_eq)) else: f = -J * z free_energy.append(f) entropy = -np.gradient(free_energy, dT) heat_capacity = T_heat * np.gradient(entropy, dT) ax4.plot(T_heat / T_c, -heat_capacity / k_B, 'r-', linewidth=2) ax4.axvline(1, color='k', linestyle=':', linewidth=2, label='T_c') ax4.set_xlabel('T / T_c') ax4.set_ylabel('C / k_B') ax4.set_title('比熱（T_c で不連続）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_critical_exponents.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 臨界指数（平均場理論）===\n") print(f"理論値（平均場）:") print(f" β = 1/2 （磁化）") print(f" γ = 1 （磁化率）") print(f" δ = 3 （臨界等温線）") print(f" α = 0 （比熱、不連続）\n") print(f"数値フィット:") print(f" β = {beta_fitted:.4f}") print(f" γ = {gamma_fitted:.4f}") print(f" δ = {delta_fitted:.4f}\n") print("実験値（3次元Ising）:") print(" β ≈ 0.325") print(" γ ≈ 1.24") print(" δ ≈ 4.8") print(" α ≈ 0.11") 

## 💻 例題4.3: Landau理論

### Landau自由エネルギー

秩序パラメータ \\(m\\) の関数として自由エネルギーを展開:

\\[ F(m, T) = F_0 + a(T) m^2 + b m^4 - hm \\]

ここで \\(a(T) = a_0(T - T_c)\\)、\\(b > 0\\)。

**平衡条件** : \\(\frac{\partial F}{\partial m} = 0\\)

\\[ 2a(T)m + 4bm^3 = h \\]

\\(h = 0\\) のとき:

  * \\(T > T_c\\): \\(m = 0\\) のみ（常磁性相）
  * \\(T < T_c\\): \\(m = \pm\sqrt{-a(T)/(2b)}\\) （強磁性相）

Python実装: Landau理論

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import minimize_scalar def landau_free_energy(m, T, T_c, a0, b, h): """Landau自由エネルギー""" a = a0 * (T - T_c) return a * m**2 + b * m**4 - h * m def equilibrium_magnetization(T, T_c, a0, b, h): """平衡磁化（自由エネルギー最小化）""" # 複数の初期値から探索 m_candidates = [] for m_init in [-1, 0, 1]: result = minimize_scalar(lambda m: landau_free_energy(m, T, T_c, a0, b, h), bounds=(-2, 2), method='bounded') m_candidates.append(result.x) # 最小の自由エネルギーを与える解を選択 F_values = [landau_free_energy(m, T, T_c, a0, b, h) for m in m_candidates] return m_candidates[np.argmin(F_values)] # パラメータ T_c = 500 # K a0 = 1e-4 b = 1e-6 h_values = [0, 0.01, 0.05, 0.1] T_range = np.linspace(300, 700, 100) colors = ['blue', 'green', 'orange', 'red'] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Landauポテンシャルの形状（異なる温度） ax1 = axes[0, 0] m_plot = np.linspace(-1.5, 1.5, 200) temperatures_plot = [0.8*T_c, 0.95*T_c, T_c, 1.2*T_c] for T_val, color in zip(temperatures_plot, colors): F_plot = [landau_free_energy(m, T_val, T_c, a0, b, 0) for m in m_plot] ax1.plot(m_plot, F_plot, color=color, linewidth=2, label=f'T/T_c = {T_val/T_c:.2f}') ax1.set_xlabel('Order parameter m') ax1.set_ylabel('Free energy F(m)') ax1.set_title('Landauポテンシャル（h = 0）') ax1.legend() ax1.grid(True, alpha=0.3) # 平衡磁化の温度依存性 ax2 = axes[0, 1] for h, color in zip(h_values, colors): m_eq = [equilibrium_magnetization(T, T_c, a0, b, h) for T in T_range] ax2.plot(T_range / T_c, m_eq, color=color, linewidth=2, label=f'h = {h:.2f}') ax2.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c') ax2.set_xlabel('T / T_c') ax2.set_ylabel('Magnetization m') ax2.set_title('平衡磁化') ax2.legend() ax2.grid(True, alpha=0.3) # 1次相転移（bを負にする場合） ax3 = axes[1, 0] # F = a m² + b m⁴ + c m⁶ (b < 0の場合、1次転移) b_first_order = -2e-6 c = 1e-7 def landau_first_order(m, T, T_c, a0, b, c, h): a = a0 * (T - T_c) return a * m**2 + b * m**4 + c * m**6 - h * m m_1st = np.linspace(-1.5, 1.5, 200) T_1st = [0.95*T_c, 0.98*T_c, T_c, 1.02*T_c] for T_val, color in zip(T_1st, colors): F_1st = [landau_first_order(m, T_val, T_c, a0, b_first_order, c, 0) for m in m_1st] ax3.plot(m_1st, F_1st, color=color, linewidth=2, label=f'T/T_c = {T_val/T_c:.2f}') ax3.set_xlabel('Order parameter m') ax3.set_ylabel('Free energy F(m)') ax3.set_title('1次相転移（二重井戸ポテンシャル）') ax3.legend() ax3.grid(True, alpha=0.3) # 磁化率 ax4 = axes[1, 1] epsilon_h = 0.001 chi_landau = [] for T in T_range: m0 = equilibrium_magnetization(T, T_c, a0, b, 0) m_h = equilibrium_magnetization(T, T_c, a0, b, epsilon_h) chi = (m_h - m0) / epsilon_h chi_landau.append(chi) ax4.plot(T_range / T_c, chi_landau, 'purple', linewidth=2) ax4.axvline(1, color='k', linestyle=':', linewidth=1, label='T_c') ax4.set_xlabel('T / T_c') ax4.set_ylabel('χ') ax4.set_title('磁化率（T_c で発散）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_landau_theory.png', dpi=300, bbox_inches='tight') plt.show() print("=== Landau理論 ===\n") print(f"パラメータ:") print(f" T_c = {T_c} K") print(f" a₀ = {a0}") print(f" b = {b}\n") print("2次相転移（b > 0）:") print(f" T > T_c: m = 0（単一井戸）") print(f" T < T_c: m = ±√(-a/2b)（二重井戸）") print(f" 臨界指数 β = 1/2\n") print("1次相転移（b < 0, c > 0）:") print(f" 二重井戸ポテンシャル") print(f" 転移で磁化が不連続にジャンプ") print(f" 潜熱が存在") 

## 💻 例題4.4: van der Waals気体と液-気相転移

### van der Waals状態方程式

\\[ \left(P + \frac{a}{V^2}\right)(V - b) = k_B T \\]

  * \\(a\\): 分子間引力（凝集力）
  * \\(b\\): 分子の排除体積

**臨界点** :

\\[ T_c = \frac{8a}{27k_B b}, \quad P_c = \frac{a}{27b^2}, \quad V_c = 3b \\]

換算変数 \\(t = T/T_c, p = P/P_c, v = V/V_c\\) で:

\\[ \left(p + \frac{3}{v^2}\right)(3v - 1) = 8t \\]

\\(T < T_c\\) で液相-気相共存（Maxwell構成則）。

Python実装: van der Waals気体

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import fsolve k_B = 1.380649e-23 def vdw_pressure(V, T, a, b, k_B): """van der Waals圧力""" return k_B * T / (V - b) - a / V**2 def vdw_equation(V, P, T, a, b, k_B): """van der Waals方程式（体積を求める）""" return P - vdw_pressure(V, T, a, b, k_B) # van der Waalsパラメータ（Arガス） a = 1.355e-49 # J·m³ b = 3.201e-29 # m³ T_c = 8 * a / (27 * k_B * b) P_c = a / (27 * b**2) V_c = 3 * b print(f"臨界定数:") print(f" T_c = {T_c:.2f} K") print(f" P_c = {P_c/1e6:.2f} MPa") print(f" V_c = {V_c*1e30:.2f} ų\n") # 等温線（換算変数） v_range = np.linspace(0.4, 5, 300) temperatures_reduced = [0.85, 0.95, 1.0, 1.1, 1.3] colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures_reduced))) fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # P-V図（換算変数） ax1 = axes[0, 0] for t_r, color in zip(temperatures_reduced, colors): p_values = [(3 / v**2) * (8*t_r / (3*v - 1) - 1) for v in v_range if 3*v > 1] v_plot = [v for v in v_range if 3*v > 1] ax1.plot(v_plot, p_values, color=color, linewidth=2, label=f't = {t_r:.2f}') ax1.axhline(1, color='k', linestyle=':', linewidth=1) ax1.axvline(1, color='k', linestyle=':', linewidth=1) ax1.plot(1, 1, 'ro', markersize=10, label='Critical point') ax1.set_xlabel('v = V/V_c') ax1.set_ylabel('p = P/P_c') ax1.set_title('van der Waals等温線（換算変数）') ax1.set_xlim([0.4, 3]) ax1.set_ylim([0, 2]) ax1.legend() ax1.grid(True, alpha=0.3) # Maxwellの等面積則 ax2 = axes[0, 1] t_coexist = 0.90 v_fine = np.linspace(0.4, 3, 500) p_fine = [(3 / v**2) * (8*t_coexist / (3*v - 1) - 1) for v in v_fine if 3*v > 1] v_plot_fine = [v for v in v_fine if 3*v > 1] ax2.plot(v_plot_fine, p_fine, 'b-', linewidth=2, label=f't = {t_coexist}') # 平衡圧力の推定（簡易版） p_equilibrium = 0.7 # 目視で調整 ax2.axhline(p_equilibrium, color='r', linestyle='--', linewidth=2, label='Maxwell construction') ax2.set_xlabel('v = V/V_c') ax2.set_ylabel('p = P/P_c') ax2.set_title('Maxwell等面積則') ax2.set_xlim([0.4, 3]) ax2.set_ylim([0.4, 1.2]) ax2.legend() ax2.grid(True, alpha=0.3) # 圧縮率 ax3 = axes[1, 0] for t_r, color in zip(temperatures_reduced, colors): # κ = -1/V (∂V/∂P)_T v_comp = np.linspace(0.5, 3, 100) dv = v_comp[1] - v_comp[0] p_comp = [(3 / v**2) * (8*t_r / (3*v - 1) - 1) for v in v_comp if 3*v > 1] v_comp_valid = [v for v in v_comp if 3*v > 1] if len(p_comp) > 2: dP_dV = np.gradient(p_comp, dv) kappa = -1 / (np.array(v_comp_valid) * np.array(dP_dV)) ax3.plot(v_comp_valid, kappa, color=color, linewidth=2, label=f't = {t_r:.2f}') ax3.set_xlabel('v = V/V_c') ax3.set_ylabel('κ (reduced)') ax3.set_title('等温圧縮率') ax3.set_ylim([0, 10]) ax3.legend() ax3.grid(True, alpha=0.3) # 相図（T-P平面） ax4 = axes[1, 1] # 蒸気圧曲線（簡易推定） T_sat = np.linspace(0.6 * T_c, T_c, 50) # Clausius-Clapeyron近似 P_sat = P_c * np.exp(-1.5 * (1 - T_sat / T_c)) ax4.plot(T_sat / T_c, P_sat / P_c, 'b-', linewidth=3, label='Coexistence curve') ax4.plot(1, 1, 'ro', markersize=12, label='Critical point') ax4.fill_between(T_sat / T_c, 0, P_sat / P_c, alpha=0.3, color='blue', label='Liquid') ax4.fill_between(T_sat / T_c, P_sat / P_c, 2, alpha=0.3, color='red', label='Gas') ax4.set_xlabel('T / T_c') ax4.set_ylabel('P / P_c') ax4.set_title('相図（液-気共存曲線）') ax4.set_xlim([0.6, 1.2]) ax4.set_ylim([0, 1.5]) ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('stat_mech_vdw_phase_transition.png', dpi=300, bbox_inches='tight') plt.show() print("van der Waals気体の特徴:") print(" - 臨界点以下で液-気共存") print(" - Maxwell等面積則で平衡圧力決定") print(" - 臨界点で等温圧縮率が発散") print(" - 格子気体モデルと数学的に等価") 

## 📚 まとめ

  * **Isingモデル** は相互作用スピン系の最も基本的なモデルで、強磁性転移を記述
  * **平均場近似** は多体問題を1体問題に還元する手法で、臨界温度と自発磁化を予測
  * **2次相転移** では秩序パラメータが連続的に変化し、臨界指数で特徴づけられる
  * **1次相転移** では秩序パラメータが不連続に変化し、潜熱が存在
  * **Landau理論** は秩序パラメータの現象論的記述で、相転移の普遍的性質を捉える
  * **臨界指数** は平均場理論で計算可能だが、実験値とは系統的にずれる（次元依存性）
  * **van der Waals気体** は液-気相転移を記述し、格子気体モデルと数学的に等価
  * 相転移理論は材料科学における強磁性、超伝導、構造相転移の理解に不可欠

### 💡 演習問題

  1. **反強磁性Ising** : \\(J < 0\\) の反強磁性Isingモデルで、2つの副格子を考えた平均場理論を展開し、Néel温度を求めよ。
  2. **3次元Ising** : 単純立方格子（\\(z = 6\\)）の強磁性Isingモデルで臨界温度を求め、実験値（\\(T_c \approx 4.51 J/k_B\\)）と比較せよ。
  3. **Widom scaling** : 臨界指数の間に成り立つスケーリング則 \\(\alpha + 2\beta + \gamma = 2\\) と \\(\gamma = \beta(\delta - 1)\\) を平均場理論で検証せよ。
  4. **Landau-Ginzburg** : 空間依存性を含むLandau-Ginzburg自由エネルギーを考え、界面エネルギーを計算せよ。
  5. **van der Waals実在気体** : CO₂のvan der Waalsパラメータを用いて、臨界点と相図を計算せよ。

[← 第3章: グランドカノニカル集団と化学ポテンシャル](<chapter-3.html>) [第5章: 統計力学シミュレーション →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
