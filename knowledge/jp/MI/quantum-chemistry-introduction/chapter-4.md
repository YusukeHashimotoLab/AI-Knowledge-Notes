---
title: "🔬 第4章: 密度汎関数理論（DFT）"
chapter_title: "🔬 第4章: 密度汎関数理論（DFT）"
---

[ナレッジベース](<../../index.html>) > [MI](<../index.html>) > [量子化学入門](<index.html>) > 第4章 

## 🎯 学習目標

  * Hohenberg-Kohn定理と電子密度の一意性を理解する
  * Kohn-Sham方程式の導出と物理的意味を学ぶ
  * 交換相関汎関数の役割と近似を理解する
  * 局所密度近似（LDA）と一般化勾配近似（GGA）を学ぶ
  * ハイブリッド汎関数（B3LYP、PBE0）の概念を習得する
  * 平面波基底とPseudopotentialを理解する
  * 固体のバンド構造と状態密度を計算する
  * 実際の材料計算への応用を学ぶ

## 📖 Hohenberg-Kohn定理

### DFTの基本原理

N電子系の基底状態は、電子密度 \\(\rho(\mathbf{r})\\) のみで決定されます。

**第1定理（一意性定理）** :

外部ポテンシャル \\(v_{ext}(\mathbf{r})\\) は電子密度 \\(\rho(\mathbf{r})\\) で一意に決まる（定数項を除く）。 したがって、基底状態の全ての性質は \\(\rho(\mathbf{r})\\) の汎関数として表現できます。

**第2定理（変分原理）** :

エネルギー汎関数 \\(E[\rho]\\) は基底状態密度で最小値をとります：

\\[ E[\rho] = T[\rho] + V_{ext}[\rho] + V_{ee}[\rho] \geq E_0 \\]

ここで \\(E_0\\) は基底状態の厳密なエネルギーです。

### Kohn-Sham方程式

実在系を非相互作用系にマッピングし、同じ電子密度を再現します：

\\[ \left[-\frac{1}{2}\nabla^2 + v_{KS}(\mathbf{r})\right]\phi_i(\mathbf{r}) = \varepsilon_i \phi_i(\mathbf{r}) \\]

**Kohn-Shamポテンシャル** :

\\[ v_{KS}(\mathbf{r}) = v_{ext}(\mathbf{r}) + v_H(\mathbf{r}) + v_{xc}(\mathbf{r}) \\]

  * \\(v_{ext}\\): 外部ポテンシャル（核-電子引力）
  * \\(v_H\\): Hartreeポテンシャル（古典的電子-電子反発）
  * \\(v_{xc}\\): 交換相関ポテンシャル（量子効果）

電子密度:

\\[ \rho(\mathbf{r}) = \sum_{i=1}^{N/2} 2|\phi_i(\mathbf{r})|^2 \\]

## 💻 例題4.1: Kohn-Sham DFT計算（1次元）

### 1次元調和井戸のKohn-Sham計算

簡単な1次元系でKohn-Sham方程式を数値的に解きます：

外部ポテンシャル: \\(v_{ext}(x) = \frac{1}{2}\omega^2 x^2\\)

LDA近似（1次元）: \\(\varepsilon_{xc}[\rho] = C_x \rho^{4/3}\\)

Python実装: 1次元Kohn-Sham DFT

import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigh from scipy.integrate import simps class KohnSham1D: """1次元Kohn-Sham DFT計算""" def __init__(self, N_electrons=2, omega=1.0, L=10.0, N_grid=200): """ N_electrons: 電子数 omega: 調和ポテンシャルの強さ L: 計算領域のサイズ N_grid: グリッド点数 """ self.N_electrons = N_electrons self.omega = omega self.L = L self.N_grid = N_grid # 空間グリッド self.x = np.linspace(-L/2, L/2, N_grid) self.dx = self.x[1] - self.x[0] # 運動エネルギー演算子（有限差分） self.T = self.kinetic_energy_matrix() def kinetic_energy_matrix(self): """運動エネルギー行列（3点有限差分）""" N = self.N_grid dx = self.dx # -1/2 d²/dx² T = np.zeros((N, N)) for i in range(1, N-1): T[i, i-1] = -1 / (2 * dx**2) T[i, i] = 1 / dx**2 T[i, i+1] = -1 / (2 * dx**2) # 境界条件（ψ=0） T[0, 0] = T[-1, -1] = 1e10 # 大きな値でψ≈0を強制 return T def external_potential(self): """外部ポテンシャル（調和井戸）""" return 0.5 * self.omega**2 * self.x**2 def hartree_potential(self, rho): """Hartreeポテンシャル（1次元）""" # 簡易的なCoulombポテンシャル（1次元） # v_H(x) = ∫ ρ(x') / |x - x'| dx' v_H = np.zeros_like(self.x) for i, xi in enumerate(self.x): # 特異点を避ける denominator = np.abs(self.x - xi) + 1e-10 v_H[i] = simps(rho / denominator, self.x) return v_H def xc_potential_lda(self, rho): """交換相関ポテンシャル（LDA、1次元）""" # 1次元LDA: v_xc = d(ε_xc ρ)/dρ # ε_xc ∝ ρ^(4/3) → v_xc ∝ ρ^(1/3) C_x = -0.5 # 交換エネルギー定数（1次元、簡略化） v_xc = (4/3) * C_x * np.sign(rho) * np.abs(rho)**(1/3) v_xc[np.abs(rho) < 1e-10] = 0 # 数値安定性 return v_xc def kohn_sham_potential(self, rho): """Kohn-Shamポテンシャル""" v_ext = self.external_potential() v_H = self.hartree_potential(rho) v_xc = self.xc_potential_lda(rho) return v_ext + v_H + v_xc def solve_kohn_sham(self, rho_init=None, max_iter=50, conv_threshold=1e-6): """Kohn-Sham方程式の自己無撞着計算""" # 初期密度 if rho_init is None: # Gaussian初期密度 rho = np.exp(-self.x**2) / np.sqrt(np.pi) rho = rho * self.N_electrons / simps(rho, self.x) energies = [] converged = False for iteration in range(max_iter): # Kohn-Shamポテンシャル v_KS = self.kohn_sham_potential(rho) # Hamiltonian行列 V_diag = np.diag(v_KS) H = self.T + V_diag # 固有値問題を解く eigenvalues, eigenvectors = eigh(H) # 占有軌道（最低N/2個） n_occupied = self.N_electrons // 2 # 新しい密度 rho_new = np.zeros_like(self.x) for i in range(n_occupied): rho_new += 2 * eigenvectors[:, i]**2 # スピン対 # 規格化 rho_new = rho_new * self.N_electrons / simps(rho_new, self.x) # エネルギー計算 E_total = self.compute_energy(rho_new, eigenvalues[:n_occupied]) energies.append(E_total) # 収束判定 if iteration > 0: delta_rho = np.max(np.abs(rho_new - rho)) if delta_rho < conv_threshold: converged = True break # 密度の混合（収束安定化） alpha_mix = 0.3 rho = alpha_mix * rho_new + (1 - alpha_mix) * rho return { 'converged': converged, 'iterations': iteration + 1, 'energy': E_total, 'density': rho_new, 'orbitals': eigenvectors[:, :n_occupied], 'orbital_energies': eigenvalues[:n_occupied], 'energy_history': energies } def compute_energy(self, rho, orbital_energies): """全エネルギー""" # Kohn-Shamエネルギー E_KS = 2 * np.sum(orbital_energies) # 二重計上補正（簡略版） v_H = self.hartree_potential(rho) E_H = 0.5 * simps(rho * v_H, self.x) v_xc = self.xc_potential_lda(rho) E_xc = simps(rho * v_xc, self.x) * (3/4) # 簡略補正 E_total = E_KS - E_H - E_xc * (1/4) return E_total # DFT計算実行 dft = KohnSham1D(N_electrons=2, omega=1.0, L=10, N_grid=200) result = dft.solve_kohn_sham() # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 電子密度 ax1 = axes[0, 0] ax1.plot(dft.x, result['density'], 'b-', linewidth=2, label='DFT density') ax1.fill_between(dft.x, 0, result['density'], alpha=0.3) ax1.set_xlabel('Position x') ax1.set_ylabel('Electron density ρ(x)') ax1.set_title('電子密度分布') ax1.legend() ax1.grid(True, alpha=0.3) # Kohn-Sham軌道 ax2 = axes[0, 1] for i, phi in enumerate(result['orbitals'].T): # 規格化確認 norm = np.sqrt(simps(phi**2, dft.x)) phi_normalized = phi / norm ax2.plot(dft.x, phi_normalized + i*0.5, linewidth=2, label=f'φ_{i+1}') ax2.set_xlabel('Position x') ax2.set_ylabel('Kohn-Sham orbitals φ_i(x)') ax2.set_title('Kohn-Sham軌道') ax2.legend() ax2.grid(True, alpha=0.3) # ポテンシャル成分 ax3 = axes[1, 0] v_ext = dft.external_potential() v_H = dft.hartree_potential(result['density']) v_xc = dft.xc_potential_lda(result['density']) v_KS = v_ext + v_H + v_xc ax3.plot(dft.x, v_ext, 'b-', linewidth=2, label='External') ax3.plot(dft.x, v_H, 'r-', linewidth=2, label='Hartree') ax3.plot(dft.x, v_xc, 'g-', linewidth=2, label='XC (LDA)') ax3.plot(dft.x, v_KS, 'k--', linewidth=2, label='Total KS') ax3.set_xlabel('Position x') ax3.set_ylabel('Potential') ax3.set_title('Kohn-Shamポテンシャル成分') ax3.legend() ax3.grid(True, alpha=0.3) ax3.set_ylim([0, 10]) # 収束履歴 ax4 = axes[1, 1] ax4.plot(result['energy_history'], 'go-', linewidth=2, markersize=6) ax4.set_xlabel('SCF iteration') ax4.set_ylabel('Total energy') ax4.set_title('DFT-SCF収束') ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_kohn_sham_dft.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== Kohn-Sham DFT計算（1次元） ===\n") print(f"電子数: {dft.N_electrons}") print(f"収束: {result['converged']} ({result['iterations']} iterations)") print(f"全エネルギー: {result['energy']:.6f}") print(f"\nKohn-Sham軌道エネルギー:") for i, eps in enumerate(result['orbital_energies']): print(f" φ_{i+1}: ε = {eps:.6f}") print(f"\n電子密度の規格化:") total_electrons = simps(result['density'], dft.x) print(f" ∫ρ(x)dx = {total_electrons:.6f} (目標: {dft.N_electrons})") 

## 💻 例題4.2: 交換相関汎関数

### 交換相関汎関数の階層

**局所密度近似（LDA）** :

\\[ E_{xc}^{LDA}[\rho] = \int \rho(\mathbf{r}) \varepsilon_{xc}(\rho(\mathbf{r})) d^3r \\]

  * 一様電子ガスのエネルギーを使用
  * 交換: \\(\varepsilon_x(\rho) = -C_x \rho^{1/3}\\)、\\(C_x = \frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\\)
  * 相関: Vosko-Wilk-Nusair（VWN）などのパラメータ化

**一般化勾配近似（GGA）** :

\\[ E_{xc}^{GGA}[\rho] = \int \rho(\mathbf{r}) \varepsilon_{xc}(\rho, |\nabla\rho|) d^3r \\]

  * 密度勾配を考慮、不均一系で改善
  * 代表例: PBE、BLYP、PW91

**ハイブリッド汎関数** :

\\[ E_{xc}^{hybrid} = aE_x^{HF} + (1-a)E_x^{DFT} + E_c^{DFT} \\]

  * Hartree-Fock交換を一部混合
  * B3LYP: \\(a=0.2\\)、PBE0: \\(a=0.25\\)
  * バンドギャップ、反応エネルギーで高精度

Python実装: 交換相関汎関数の比較

import numpy as np import matplotlib.pyplot as plt def exchange_lda(rho): """LDA交換エネルギー密度""" C_x = (3/4) * (3/np.pi)**(1/3) return -C_x * rho**(4/3) def exchange_pbe(rho, grad_rho, kappa=0.804, mu=0.2195): """PBE交換エネルギー密度（簡略版）""" # 還元密度勾配 s = np.abs(grad_rho) / (2 * (3*np.pi**2)**(1/3) * rho**(4/3)) # Enhancement factor F_x = 1 + kappa - kappa / (1 + mu * s**2 / kappa) # LDA × enhancement epsilon_x_lda = exchange_lda(rho) / rho return rho * epsilon_x_lda * F_x def correlation_vwn(rho): """VWN相関エネルギー密度（簡略版）""" # Vosko-Wilk-Nusair パラメータ（常磁性） A = 0.0310907 x0 = -0.10498 b = 3.72744 c = 12.9352 r_s = (3 / (4 * np.pi * rho))**(1/3) x = np.sqrt(r_s) X = x**2 + b*x + c X0 = x0**2 + b*x0 + c Q = np.sqrt(4*c - b**2) epsilon_c = A * ( np.log(x**2 / X) + 2*b/Q * np.arctan(Q / (2*x + b)) - b*x0/X0 * ( np.log((x - x0)**2 / X) + 2*(b + 2*x0)/Q * np.arctan(Q / (2*x + b)) ) ) return rho * epsilon_c # 電子密度範囲 rho_range = np.logspace(-2, 1, 100) # 勾配の強さ（GGA） grad_rho_weak = 0.1 * rho_range**(4/3) grad_rho_strong = 1.0 * rho_range**(4/3) # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # LDA交換エネルギー密度 ax1 = axes[0, 0] epsilon_x_lda = exchange_lda(rho_range) / rho_range ax1.plot(rho_range, epsilon_x_lda, 'b-', linewidth=2, label='LDA exchange') ax1.set_xlabel('Electron density ρ') ax1.set_ylabel('ε_x (energy per electron)') ax1.set_title('LDA交換エネルギー密度') ax1.set_xscale('log') ax1.legend() ax1.grid(True, alpha=0.3) # PBE vs LDA（弱勾配） ax2 = axes[0, 1] epsilon_x_pbe_weak = exchange_pbe(rho_range, grad_rho_weak) / rho_range ax2.plot(rho_range, epsilon_x_lda, 'b-', linewidth=2, label='LDA') ax2.plot(rho_range, epsilon_x_pbe_weak, 'r-', linewidth=2, label='PBE (weak ∇ρ)') ax2.set_xlabel('Electron density ρ') ax2.set_ylabel('ε_x') ax2.set_title('GGA効果（弱勾配）') ax2.set_xscale('log') ax2.legend() ax2.grid(True, alpha=0.3) # PBE vs LDA（強勾配） ax3 = axes[1, 0] epsilon_x_pbe_strong = exchange_pbe(rho_range, grad_rho_strong) / rho_range ax3.plot(rho_range, epsilon_x_lda, 'b-', linewidth=2, label='LDA') ax3.plot(rho_range, epsilon_x_pbe_strong, 'r-', linewidth=2, label='PBE (strong ∇ρ)') ax3.set_xlabel('Electron density ρ') ax3.set_ylabel('ε_x') ax3.set_title('GGA効果（強勾配）') ax3.set_xscale('log') ax3.legend() ax3.grid(True, alpha=0.3) # 相関エネルギー（VWN） ax4 = axes[1, 1] epsilon_c_vwn = correlation_vwn(rho_range) / rho_range ax4.plot(rho_range, epsilon_c_vwn, 'g-', linewidth=2, label='VWN correlation') ax4.set_xlabel('Electron density ρ') ax4.set_ylabel('ε_c') ax4.set_title('LDA相関エネルギー（VWN）') ax4.set_xscale('log') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_xc_functionals.png', dpi=300, bbox_inches='tight') plt.show() # 数値比較 print("\n=== 交換相関汎関数の比較 ===\n") print("代表的な密度での交換エネルギー（ρ = 0.1）:") rho_test = 0.1 print(f" LDA exchange: {exchange_lda(rho_test) / rho_test:.6f}") grad_test_weak = 0.01 grad_test_strong = 0.1 print(f" PBE (|∇ρ| = {grad_test_weak}): {exchange_pbe(rho_test, grad_test_weak) / rho_test:.6f}") print(f" PBE (|∇ρ| = {grad_test_strong}): {exchange_pbe(rho_test, grad_test_strong) / rho_test:.6f}") print(f"\n相関エネルギー:") print(f" VWN correlation: {correlation_vwn(rho_test) / rho_test:.6f}") print("\n汎関数の選択指針:") print(" LDA: 金属、高対称性系") print(" GGA (PBE): 分子、表面、一般的固体") print(" Hybrid (B3LYP, PBE0): 分子化学、バンドギャップ") print(" Meta-GGA (TPSS, SCAN): 高精度固体計算") 

## 💻 例題4.3: 平面波基底とPseudopotential

### 周期系のDFT計算

**Blochの定理** :

周期ポテンシャル中の波動関数:

\\[ \psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r}) \\]

ここで \\(u_{n\mathbf{k}}\\) は格子周期性を持ちます。

**平面波展開** :

\\[ \psi_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}} \\]

\\(\mathbf{G}\\) は逆格子ベクトルです。

**Pseudopotential近似** :

  * 内殻電子を擬ポテンシャルで置換
  * 価電子のみを明示的に扱う
  * Norm-conserving、Ultrasoft、PAW法

Python実装: 1次元周期系のバンド構造

import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigh class PlaneWave1D: """1次元周期系の平面波DFT""" def __init__(self, a=1.0, V0=1.0, N_pw=11): """ a: 格子定数 V0: ポテンシャルの強さ N_pw: 平面波の数（-N_pw/2 から N_pw/2） """ self.a = a self.V0 = V0 self.N_pw = N_pw # 逆格子ベクトル self.G = 2 * np.pi / a * np.arange(-N_pw//2, N_pw//2 + 1) def periodic_potential(self, x): """周期ポテンシャル V(x) = V0 cos(2πx/a)""" return self.V0 * np.cos(2 * np.pi * x / self.a) def hamiltonian_matrix(self, k): """k点でのHamiltonian行列""" N = len(self.G) H = np.zeros((N, N), dtype=complex) for i, Gi in enumerate(self.G): for j, Gj in enumerate(self.G): if i == j: # 運動エネルギー H[i, j] = 0.5 * (k + Gi)**2 else: # ポテンシャルのFourier成分 # V(x) = V0 cos(2πx/a) → V_G = V0/2 δ_{G,±G0} G_diff = Gi - Gj G0 = 2 * np.pi / self.a if np.abs(G_diff - G0) < 1e-10: H[i, j] = self.V0 / 2 elif np.abs(G_diff + G0) < 1e-10: H[i, j] = self.V0 / 2 return H def compute_bands(self, k_points): """バンド構造計算""" bands = [] for k in k_points: H = self.hamiltonian_matrix(k) eigenvalues = eigh(H, eigvals_only=True) bands.append(eigenvalues) return np.array(bands) # バンド構造計算 pw = PlaneWave1D(a=1.0, V0=2.0, N_pw=11) # Brillouin zone: -π/a から π/a k_points = np.linspace(-np.pi/pw.a, np.pi/pw.a, 100) bands = pw.compute_bands(k_points) # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # バンド構造 ax1 = axes[0, 0] n_bands = min(5, bands.shape[1]) # 最初の5バンド for i in range(n_bands): ax1.plot(k_points * pw.a / np.pi, bands[:, i], linewidth=2) ax1.set_xlabel('k (units of π/a)') ax1.set_ylabel('Energy') ax1.set_title(f'バンド構造（V₀ = {pw.V0}）') ax1.axvline(0, color='k', linestyle='--', linewidth=1) ax1.grid(True, alpha=0.3) # バンドギャップの V0 依存性 ax2 = axes[0, 1] V0_range = np.linspace(0, 5, 20) band_gaps = [] k_gamma = 0 # Γ点 k_edge = np.pi / pw.a # Brillouin zone境界 for V0 in V0_range: pw_temp = PlaneWave1D(a=1.0, V0=V0, N_pw=11) # Γ点のバンド H_gamma = pw_temp.hamiltonian_matrix(k_gamma) E_gamma = eigh(H_gamma, eigvals_only=True) # Zone境界のバンド H_edge = pw_temp.hamiltonian_matrix(k_edge) E_edge = eigh(H_edge, eigvals_only=True) # バンドギャップ（最低励起エネルギー） gap = E_gamma[1] - E_gamma[0] if len(E_gamma) > 1 else 0 band_gaps.append(gap) ax2.plot(V0_range, band_gaps, 'ro-', linewidth=2, markersize=6) ax2.set_xlabel('Potential strength V₀') ax2.set_ylabel('Band gap') ax2.set_title('バンドギャップのV₀依存性') ax2.grid(True, alpha=0.3) # 状態密度（DOS） ax3 = axes[1, 0] # 詳細なk点サンプリング k_dense = np.linspace(-np.pi/pw.a, np.pi/pw.a, 500) bands_dense = pw.compute_bands(k_dense) # ヒストグラム法でDOS計算 E_min, E_max = bands_dense.min(), bands_dense.max() E_bins = np.linspace(E_min, E_max, 100) dos, _ = np.histogram(bands_dense.flatten(), bins=E_bins) ax3.plot(dos, E_bins[:-1], 'b-', linewidth=2) ax3.set_xlabel('Density of States') ax3.set_ylabel('Energy') ax3.set_title('状態密度（DOS）') ax3.grid(True, alpha=0.3) # 波動関数（Γ点、最低バンド） ax4 = axes[1, 1] k_gamma = 0 H_gamma = pw.hamiltonian_matrix(k_gamma) eigenvalues, eigenvectors = eigh(H_gamma) # 実空間再構成 x = np.linspace(0, pw.a, 200) psi_0 = np.zeros_like(x, dtype=complex) for i, G in enumerate(pw.G): psi_0 += eigenvectors[i, 0] * np.exp(1j * (k_gamma + G) * x) ax4.plot(x / pw.a, np.real(psi_0), 'b-', linewidth=2, label='Re(ψ)') ax4.plot(x / pw.a, np.abs(psi_0)**2, 'r-', linewidth=2, label='|ψ|²') # ポテンシャル（規格化して表示） V_plot = pw.periodic_potential(x) V_normalized = V_plot / np.max(np.abs(V_plot)) * np.max(np.abs(psi_0)) ax4.plot(x / pw.a, V_normalized, 'g--', linewidth=1, label='V(x) (scaled)') ax4.set_xlabel('Position (x/a)') ax4.set_ylabel('Wavefunction') ax4.set_title('Γ点波動関数（最低バンド）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_plane_wave_bands.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 平面波基底とバンド構造 ===\n") print(f"格子定数: a = {pw.a}") print(f"ポテンシャル強度: V₀ = {pw.V0}") print(f"平面波数: {pw.N_pw}") print(f"\nΓ点のエネルギー（最低3バンド）:") H_gamma = pw.hamiltonian_matrix(0) E_gamma = eigh(H_gamma, eigvals_only=True) for i in range(min(3, len(E_gamma))): print(f" Band {i+1}: E = {E_gamma[i]:.6f}") print(f"\nバンドギャップ: {E_gamma[1] - E_gamma[0]:.6f}") 

## 📚 まとめ

  * **Hohenberg-Kohn定理** により基底状態が電子密度のみで決定される
  * **Kohn-Sham方程式** は非相互作用系へのマッピングで実用計算を可能にする
  * **交換相関汎関数** が量子多体効果を近似的に取り入れる
  * **LDA** は一様系で正確、GGAは不均一系で改善
  * **ハイブリッド汎関数** はHF交換を混合し分子化学で高精度
  * **平面波基底** は周期系の自然な基底で、高速FFTが利用可能
  * **Pseudopotential** により内殻電子を効率的に扱える
  * **バンド構造** と**状態密度** から固体の電子状態を理解
  * DFTは材料科学の第一原理計算の標準手法
  * 適切な汎関数選択が計算精度を決定する

### 💡 演習問題

  1. **He原子のDFT計算** : He原子（2電子）をKohn-Sham DFTで計算し、LDAとGGAでエネルギーを比較せよ。
  2. **Jacob's Ladder** : LDA、GGA、meta-GGA、Hybrid、Double Hybridの階層を調べ、それぞれの適用範囲をまとめよ。
  3. **バンドギャップ問題** : DFT（LDA/GGA）が半導体のバンドギャップを過小評価する理由を調べよ。
  4. **1次元金属** : V₀=0の平面波計算を行い、金属のDOSがFermi準位で有限であることを確認せよ。
  5. **k点サンプリング** : Brillouin zone積分のk点収束性を調べ、Monkhorst-Pack法を実装せよ。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
