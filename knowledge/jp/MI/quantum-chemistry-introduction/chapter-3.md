---
title: "🔬 第3章: 分子軌道法と電子状態計算"
chapter_title: "🔬 第3章: 分子軌道法と電子状態計算"
---

[ナレッジベース](<../../index.html>) > [MI](<../index.html>) > [量子化学入門](<index.html>) > 第3章 

## 🎯 学習目標

  * Hartree-Fock方程式の導出と物理的意味を理解する
  * Fock演算子と自己無撞着場（SCF）法を学ぶ
  * Roothaan方程式と基底関数展開を習得する
  * Gaussian型基底関数とSlater型基底関数を理解する
  * LCAO-MO（原子軌道の線形結合）法を実装する
  * Hückel分子軌道法と拡張Hückel法を学ぶ
  * 電子相関と配置間相互作用（CI）を理解する
  * 実際の分子に対するSCF計算を実装する

## 📖 Hartree-Fock方程式

### 多電子系のHamiltonian

N電子系の電子Hamiltonian（原子単位系）：

\\[ \hat{H}_{el} = \sum_{i=1}^N \left(-\frac{1}{2}\nabla_i^2 - \sum_A \frac{Z_A}{r_{iA}}\right) + \sum_{i<j} \frac{1}{r_{ij}} \\]

  * 第1項：電子の運動エネルギー
  * 第2項：電子-核間引力
  * 第3項：電子-電子間反発（多体項）

この多体問題を近似的に解くのがHartree-Fock法です。

### Hartree-Fock近似

N電子波動関数を**Slater行列式** で表現：

\\[ \Psi_{HF} = \frac{1}{\sqrt{N!}} \begin{vmatrix} \chi_1(1) & \chi_2(1) & \cdots & \chi_N(1) \\\ \chi_1(2) & \chi_2(2) & \cdots & \chi_N(2) \\\ \vdots & \vdots & \ddots & \vdots \\\ \chi_1(N) & \chi_2(N) & \cdots & \chi_N(N) \end{vmatrix} \\]

ここで \\(\chi_i\\) はスピン軌道（空間軌道×スピン）です。

**Fock方程式** :

\\[ \hat{f} \chi_i = \varepsilon_i \chi_i \\]

**Fock演算子** :

\\[ \hat{f}(1) = \hat{h}(1) + \sum_{j=1}^N \left[\hat{J}_j(1) - \hat{K}_j(1)\right] \\]

  * \\(\hat{h}\\)：1電子Hamiltonian
  * \\(\hat{J}_j\\)：Coulomb演算子（古典的電子-電子反発）
  * \\(\hat{K}_j\\)：交換演算子（量子力学的効果、Pauli排他原理）

## 💻 例題3.1: Roothaan方程式とSCF法

### Roothaan方程式

基底関数 \\(\\{\phi_\mu\\}\\) で分子軌道を展開：

\\[ \psi_i = \sum_{\mu=1}^K C_{\mu i} \phi_\mu \\]

Roothaan方程式（行列形式のHartree-Fock方程式）：

\\[ \mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\mathbf{\varepsilon} \\]

  * \\(\mathbf{F}\\)：Fock行列 \\(F_{\mu\nu} = \langle \phi_\mu | \hat{f} | \phi_\nu \rangle\\)
  * \\(\mathbf{S}\\)：重なり行列 \\(S_{\mu\nu} = \langle \phi_\mu | \phi_\nu \rangle\\)
  * \\(\mathbf{C}\\)：軌道係数行列
  * \\(\mathbf{\varepsilon}\\)：軌道エネルギー（対角行列）

**SCF（Self-Consistent Field）アルゴリズム** :

  1. 初期軌道係数 \\(\mathbf{C}^{(0)}\\) を設定
  2. 密度行列 \\(\mathbf{P}\\) を計算
  3. Fock行列 \\(\mathbf{F}\\) を構築
  4. Roothaan方程式を解いて新しい \\(\mathbf{C}\\) を得る
  5. 収束するまで2-4を繰り返す

Python実装: 最小基底H₂のSCF計算

import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigh class MinimalBasisH2: """最小基底（STO-1G）でのH₂分子のSCF計算""" def __init__(self, R=1.4): """ R: 核間距離（Bohr） """ self.R = R self.alpha = 1.0 # STO-1Gのexponent（簡略化） def overlap_matrix(self): """重なり行列 S""" R = self.R alpha = self.alpha # S_11 = S_22 = 1 (規格化) # S_12 = S_21 = ⟨φ_A|φ_B⟩（簡易近似） S_12 = np.exp(-alpha * R) * (1 + alpha * R + (alpha * R)**2 / 3) S = np.array([[1.0, S_12], [S_12, 1.0]]) return S def core_hamiltonian(self): """コアHamiltonian行列 H_core""" R = self.R alpha = self.alpha # 運動エネルギー + 核引力 # H_11 = ⟨φ_A|T + V_A + V_B|φ_A⟩ H_11 = -alpha**2 / 2 - 1/R * (1 + 1/R) * np.exp(-2*alpha*R) - 1.0 # H_12 = ⟨φ_A|T + V_A + V_B|φ_B⟩ S_12 = np.exp(-alpha * R) * (1 + alpha * R + (alpha * R)**2 / 3) H_12 = -alpha**2 * S_12 / 2 - S_12 / R - S_12 / alpha H_core = np.array([[H_11, H_12], [H_12, H_11]]) return H_core def two_electron_integrals(self): """2電子積分（簡易近似）""" R = self.R alpha = self.alpha # (μν|λσ) 積分（簡略化） g = {} g['1111'] = 5/8 * alpha # ⟨φ_A φ_A|φ_A φ_A⟩ g['1122'] = 1/R * np.exp(-alpha * R) # ⟨φ_A φ_A|φ_B φ_B⟩ g['1212'] = 0.5 * g['1122'] # ⟨φ_A φ_B|φ_A φ_B⟩ g['2222'] = g['1111'] return g def build_fock_matrix(self, P, H_core, g): """Fock行列の構築""" # F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5(μλ|νσ)] F = H_core.copy() # Coulomb項とExchange項（簡略化） F[0, 0] += P[0, 0] * g['1111'] + P[1, 1] * (g['1122'] - 0.5 * g['1212']) F[1, 1] += P[1, 1] * g['2222'] + P[0, 0] * (g['1122'] - 0.5 * g['1212']) F[0, 1] += P[0, 1] * (g['1212'] - 0.5 * g['1122']) F[1, 0] = F[0, 1] return F def scf_iteration(self, max_iter=50, conv_threshold=1e-6): """SCF反復計算""" S = self.overlap_matrix() H_core = self.core_hamiltonian() g = self.two_electron_integrals() # 初期密度行列（ゼロ） P = np.zeros((2, 2)) energies = [] converged = False for iteration in range(max_iter): # Fock行列構築 F = self.build_fock_matrix(P, H_core, g) # 一般化固有値問題を解く: FC = SCε epsilon, C = eigh(F, S) # 密度行列の更新（2電子系なので占有軌道1つ） P_new = 2 * np.outer(C[:, 0], C[:, 0]) # 2電子分 # 電子エネルギー E_elec = 0.5 * np.sum(P_new * (H_core + F)) # 核間反発 V_NN = 1 / self.R # 全エネルギー E_total = E_elec + V_NN energies.append(E_total) # 収束判定 if iteration > 0: delta_E = abs(E_total - energies[-2]) if delta_E < conv_threshold: converged = True break P = P_new return { 'converged': converged, 'iterations': iteration + 1, 'energy': E_total, 'orbital_energies': epsilon, 'coefficients': C, 'density_matrix': P, 'energy_history': energies } # 異なる核間距離でSCF計算 R_range = np.linspace(0.5, 4.0, 30) energies_scf = [] orbital_energies_bonding = [] orbital_energies_antibonding = [] for R in R_range: h2 = MinimalBasisH2(R) result = h2.scf_iteration() energies_scf.append(result['energy']) orbital_energies_bonding.append(result['orbital_energies'][0]) orbital_energies_antibonding.append(result['orbital_energies'][1]) # 平衡核間距離 E_array = np.array(energies_scf) R_eq_idx = np.argmin(E_array) R_eq = R_range[R_eq_idx] E_eq = E_array[R_eq_idx] # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # ポテンシャルエネルギー曲線 ax1 = axes[0, 0] ax1.plot(R_range, energies_scf, 'b-', linewidth=2, label='SCF Energy') ax1.plot(R_eq, E_eq, 'ro', markersize=10, label=f'R_eq = {R_eq:.2f} Bohr') ax1.axhline(0, color='k', linestyle='--', linewidth=1) ax1.set_xlabel('Internuclear distance R (Bohr)') ax1.set_ylabel('Energy (Hartree)') ax1.set_title('H₂ ポテンシャルエネルギー曲線（SCF）') ax1.legend() ax1.grid(True, alpha=0.3) # 軌道エネルギー ax2 = axes[0, 1] ax2.plot(R_range, orbital_energies_bonding, 'b-', linewidth=2, label='Bonding MO') ax2.plot(R_range, orbital_energies_antibonding, 'r-', linewidth=2, label='Antibonding MO') ax2.axhline(0, color='k', linestyle='--', linewidth=1) ax2.set_xlabel('Internuclear distance R (Bohr)') ax2.set_ylabel('Orbital energy (Hartree)') ax2.set_title('分子軌道エネルギー') ax2.legend() ax2.grid(True, alpha=0.3) # SCF収束履歴（R = R_eq） ax3 = axes[1, 0] h2_eq = MinimalBasisH2(R_eq) result_eq = h2_eq.scf_iteration() ax3.plot(result_eq['energy_history'], 'go-', linewidth=2, markersize=6) ax3.set_xlabel('SCF iteration') ax3.set_ylabel('Total energy (Hartree)') ax3.set_title(f'SCF収束（R = {R_eq:.2f} Bohr）') ax3.grid(True, alpha=0.3) # 分子軌道係数（R = R_eq） ax4 = axes[1, 1] C = result_eq['coefficients'] x = ['φ_A (H_A)', 'φ_B (H_B)'] width = 0.35 bonding_coeffs = C[:, 0] antibonding_coeffs = C[:, 1] x_pos = np.arange(len(x)) ax4.bar(x_pos - width/2, bonding_coeffs, width, label='Bonding', color='blue') ax4.bar(x_pos + width/2, antibonding_coeffs, width, label='Antibonding', color='red') ax4.set_ylabel('Coefficient') ax4.set_title(f'分子軌道係数（R = {R_eq:.2f} Bohr）') ax4.set_xticks(x_pos) ax4.set_xticklabels(x) ax4.legend() ax4.grid(True, alpha=0.3, axis='y') ax4.axhline(0, color='k', linewidth=0.5) plt.tight_layout() plt.savefig('qchem_scf_h2.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== H₂分子のSCF計算（最小基底）===\n") print(f"平衡核間距離: R_eq = {R_eq:.3f} Bohr = {R_eq * 0.529:.3f} Å") print(f"全エネルギー: E = {E_eq:.6f} Hartree = {E_eq * 27.2114:.3f} eV") print(f"\nSCF収束:") print(f" 反復回数: {result_eq['iterations']}") print(f" 収束: {result_eq['converged']}") print(f"\n軌道エネルギー（R = R_eq）:") print(f" 結合性MO: ε₁ = {result_eq['orbital_energies'][0]:.6f} Hartree") print(f" 反結合性MO: ε₂ = {result_eq['orbital_energies'][1]:.6f} Hartree") 

## 💻 例題3.2: 基底関数とGaussian型軌道

### 基底関数の種類

**Slater型軌道（STO）** :

\\[ \chi_{STO}(r) = N r^{n-1} e^{-\zeta r} Y_l^m(\theta, \phi) \\]

  * 原子軌道に近い形状
  * 2電子積分の計算が困難

**Gaussian型軌道（GTO）** :

\\[ \chi_{GTO}(r) = N r^{2n-2-l} e^{-\alpha r^2} Y_l^m(\theta, \phi) \\]

  * 2電子積分が解析的に計算可能
  * 複数のGaussianでSTOを近似（STO-nG基底）

**縮約Gaussian基底** :

\\[ \chi_{CGTO} = \sum_i d_i \chi_{GTO,i} \\]

代表的な基底セット：STO-3G、3-21G、6-31G、6-311G、cc-pVDZ、cc-pVTZなど

Python実装: Gaussian型基底関数

import numpy as np import matplotlib.pyplot as plt def gaussian_1s(r, alpha): """Gaussian型1s軌道（規格化済み）""" N = (2 * alpha / np.pi)**(3/4) return N * np.exp(-alpha * r**2) def slater_1s(r, zeta=1.0): """Slater型1s軌道（規格化済み）""" N = (zeta**3 / np.pi)**0.5 return N * np.exp(-zeta * r) def sto_3g_1s(r): """STO-3G基底（3つのGaussianでSTOを近似）""" # H原子の1s軌道（ζ=1.0）に対するSTO-3Gパラメータ alphas = np.array([0.168856, 0.623913, 3.42525]) coeffs = np.array([0.444635, 0.535328, 0.154329]) result = np.zeros_like(r) for alpha, coeff in zip(alphas, coeffs): result += coeff * gaussian_1s(r, alpha) return result # 動径座標 r = np.linspace(0, 5, 500) # 異なる基底関数 psi_slater = slater_1s(r, zeta=1.0) psi_sto3g = sto_3g_1s(r) psi_gauss_single = gaussian_1s(r, alpha=0.3) # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 波動関数の比較 ax1 = axes[0, 0] ax1.plot(r, psi_slater, 'b-', linewidth=2, label='Slater 1s (ζ=1.0)') ax1.plot(r, psi_sto3g, 'r--', linewidth=2, label='STO-3G') ax1.plot(r, psi_gauss_single, 'g:', linewidth=2, label='Single Gaussian (α=0.3)') ax1.set_xlabel('r (Bohr)') ax1.set_ylabel('ψ(r)') ax1.set_title('基底関数の比較') ax1.legend() ax1.grid(True, alpha=0.3) # 動径確率密度 ax2 = axes[0, 1] ax2.plot(r, r**2 * psi_slater**2, 'b-', linewidth=2, label='Slater') ax2.plot(r, r**2 * psi_sto3g**2, 'r--', linewidth=2, label='STO-3G') ax2.set_xlabel('r (Bohr)') ax2.set_ylabel('r² |ψ(r)|²') ax2.set_title('動径確率密度') ax2.legend() ax2.grid(True, alpha=0.3) # STO-3Gの構成Gaussian ax3 = axes[1, 0] alphas_sto3g = np.array([0.168856, 0.623913, 3.42525]) coeffs_sto3g = np.array([0.444635, 0.535328, 0.154329]) for i, (alpha, coeff) in enumerate(zip(alphas_sto3g, coeffs_sto3g)): psi_component = coeff * gaussian_1s(r, alpha) ax3.plot(r, psi_component, linewidth=2, label=f'G{i+1} (α={alpha:.2f}, c={coeff:.3f})') ax3.plot(r, psi_sto3g, 'k-', linewidth=3, label='STO-3G (sum)') ax3.set_xlabel('r (Bohr)') ax3.set_ylabel('ψ(r)') ax3.set_title('STO-3G構成Gaussian') ax3.legend() ax3.grid(True, alpha=0.3) # 基底セットサイズの効果（概念的） ax4 = axes[1, 1] basis_sets = ['Minimal\n(STO-3G)', 'Double-ζ\n(6-31G)', 'Triple-ζ\n(6-311G)', 'cc-pVDZ', 'cc-pVTZ'] n_functions = [1, 2, 3, 5, 14] # H原子での関数数（概算） relative_accuracy = [0.8, 0.92, 0.96, 0.98, 0.995] x_pos = np.arange(len(basis_sets)) bars = ax4.bar(x_pos, relative_accuracy, color=['red', 'orange', 'yellow', 'lightgreen', 'green']) ax4.set_ylabel('Relative accuracy') ax4.set_title('基底セットサイズと精度') ax4.set_xticks(x_pos) ax4.set_xticklabels(basis_sets, rotation=15, ha='right') ax4.set_ylim([0.7, 1.0]) ax4.grid(True, alpha=0.3, axis='y') for i, (bar, n_func) in enumerate(zip(bars, n_functions)): height = bar.get_height() ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{n_func} funcs', ha='center', va='bottom', fontsize=9) plt.tight_layout() plt.savefig('qchem_basis_functions.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 基底関数 ===\n") print("STO-3G パラメータ（H 1s）:") for i, (alpha, coeff) in enumerate(zip(alphas_sto3g, coeffs_sto3g)): print(f" Gaussian {i+1}: α = {alpha:.6f}, c = {coeff:.6f}") print("\n基底セットの選択指針:") print(" - Minimal（STO-3G）: 定性的理解、大規模系") print(" - Double-ζ（6-31G）: 標準計算、合理的精度") print(" - Triple-ζ（6-311G）: 高精度計算") print(" - cc-pVnZ: ベンチマーク計算、相関エネルギー") 

## 💻 例題3.3: Hückel分子軌道法

### Hückel近似

π電子系の簡易的分子軌道計算法：

  * σ電子とπ電子を分離
  * π電子のみを考慮（p_z軌道）
  * 重なり積分を無視（S = I）

**Hückel Hamiltonian行列** :

\\[ H_{ii} = \alpha \quad (\text{対角要素}) \\]

\\[ H_{ij} = \beta \quad (\text{隣接原子}), \quad H_{ij} = 0 \quad (\text{非隣接}) \\]

  * \\(\alpha\\)：Coulomb積分（p_z軌道のエネルギー）
  * \\(\beta\\)：共鳴積分（負の値、結合エネルギーに対応）

Python実装: ベンゼンのHückel計算

import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigh class HuckelMO: """Hückel分子軌道法""" def __init__(self, adjacency_matrix, alpha=0, beta=-1): """ adjacency_matrix: 隣接行列（結合の有無） alpha: Coulomb積分（エネルギーゼロ点） beta: 共鳴積分 """ self.adjacency = adjacency_matrix self.n_atoms = len(adjacency_matrix) self.alpha = alpha self.beta = beta def build_hamiltonian(self): """Hückel Hamiltonian行列""" H = self.alpha * np.eye(self.n_atoms) + self.beta * self.adjacency return H def solve(self): """固有値問題を解く""" H = self.build_hamiltonian() eigenvalues, eigenvectors = eigh(H) # エネルギー順にソート idx = np.argsort(eigenvalues) eigenvalues = eigenvalues[idx] eigenvectors = eigenvectors[:, idx] return eigenvalues, eigenvectors def pi_energy(self, n_electrons): """π電子エネルギー""" eigenvalues, _ = self.solve() # 占有軌道のエネルギー合計 n_occupied = n_electrons // 2 # 各軌道に2電子 E_pi = 2 * np.sum(eigenvalues[:n_occupied]) return E_pi # ベンゼン（C6H6）の隣接行列 benzene_adj = np.array([ [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0] ]) # ブタジエン（C4H6）の隣接行列 butadiene_adj = np.array([ [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0] ]) # エチレン（C2H4） ethylene_adj = np.array([ [0, 1], [1, 0] ]) # ベンゼンの計算 benzene = HuckelMO(benzene_adj, alpha=0, beta=-1) E_benzene, C_benzene = benzene.solve() E_pi_benzene = benzene.pi_energy(6) # 6個のπ電子 # ブタジエンの計算 butadiene = HuckelMO(butadiene_adj, alpha=0, beta=-1) E_butadiene, C_butadiene = butadiene.solve() # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # ベンゼンのエネルギー準位図 ax1 = axes[0, 0] colors_benzene = ['blue', 'green', 'green', 'red', 'red', 'red'] occupancy_benzene = [2, 2, 2, 0, 0, 0] # 各準位の電子数 for i, (E, occ, color) in enumerate(zip(E_benzene, occupancy_benzene, colors_benzene)): ax1.hlines(E, i-0.3, i+0.3, colors=color, linewidth=3) # 電子を表示 if occ > 0: ax1.plot([i-0.1, i+0.1], [E, E], 'o', color='black', markersize=8) ax1.axhline(0, color='k', linestyle='--', linewidth=1, label='α (reference)') ax1.set_xticks(range(len(E_benzene))) ax1.set_xticklabels([f'MO{i+1}' for i in range(len(E_benzene))]) ax1.set_ylabel('Energy (units of β)') ax1.set_title('ベンゼンのHückel MO準位') ax1.legend() ax1.grid(True, alpha=0.3, axis='y') # ブタジエンのエネルギー準位 ax2 = axes[0, 1] for i, E in enumerate(E_butadiene): color = 'blue' if i < 2 else 'red' ax2.hlines(E, i-0.3, i+0.3, colors=color, linewidth=3) if i < 2: # 占有軌道 ax2.plot([i-0.1, i+0.1], [E, E], 'o', color='black', markersize=8) ax2.axhline(0, color='k', linestyle='--', linewidth=1) ax2.set_xticks(range(len(E_butadiene))) ax2.set_xticklabels([f'π{i+1}' for i in range(len(E_butadiene))]) ax2.set_ylabel('Energy (units of β)') ax2.set_title('ブタジエンのHückel MO準位') ax2.grid(True, alpha=0.3, axis='y') # ベンゼンの分子軌道係数（HOMO） ax3 = axes[1, 0] homo_index = 2 # 3番目のMO（0-indexed） homo_coeffs = C_benzene[:, homo_index] theta = np.linspace(0, 2*np.pi, 7) x_pos = np.cos(theta[:6]) y_pos = np.sin(theta[:6]) # 係数の符号で色分け colors_coeff = ['red' if c > 0 else 'blue' for c in homo_coeffs] sizes = np.abs(homo_coeffs) * 500 ax3.scatter(x_pos, y_pos, s=sizes, c=colors_coeff, alpha=0.6, edgecolors='black', linewidth=2) # ベンゼン環を描画 for i in range(6): ax3.plot([x_pos[i], x_pos[(i+1)%6]], [y_pos[i], y_pos[(i+1)%6]], 'k-', linewidth=1) ax3.set_xlim([-1.5, 1.5]) ax3.set_ylim([-1.5, 1.5]) ax3.set_aspect('equal') ax3.set_title(f'ベンゼンHOMO（MO{homo_index+1}）係数') ax3.set_xticks([]) ax3.set_yticks([]) ax3.text(0, -1.8, '赤: 正, 青: 負', ha='center', fontsize=10) # π電子エネルギーの共役鎖長依存性 ax4 = axes[1, 1] chain_lengths = range(2, 11) pi_energies = [] for n in chain_lengths: # 直鎖共役系の隣接行列 adj = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1) mol = HuckelMO(adj, alpha=0, beta=-1) E_pi = mol.pi_energy(n) # n個のπ電子 pi_energies.append(E_pi) ax4.plot(chain_lengths, pi_energies, 'go-', linewidth=2, markersize=8) ax4.set_xlabel('Number of carbon atoms') ax4.set_ylabel('Total π energy (units of β)') ax4.set_title('共役鎖長とπ電子エネルギー') ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_huckel_mo.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== Hückel分子軌道法 ===\n") print("ベンゼン（C6H6）:") print(f" MOエネルギー（α + xβ）:") for i, E in enumerate(E_benzene): print(f" MO{i+1}: E = α + ({E:.4f})β") print(f" 総π電子エネルギー: E_π = {E_pi_benzene:.4f}β") print(f" 共鳴安定化エネルギー: {E_pi_benzene - 6*(-1):.4f}β") print("\nブタジエン（C4H6）:") for i, E in enumerate(E_butadiene): print(f" π{i+1}: E = α + ({E:.4f})β") 

## 💻 例題3.4: 電子相関と配置間相互作用

### 電子相関

Hartree-Fock法では、各電子は平均場中を運動すると近似します。 しかし実際には、電子間の瞬間的な相互作用（**電子相関** ）が存在します。

**相関エネルギー** :

\\[ E_{corr} = E_{exact} - E_{HF} \\]

電子相関を取り入れる方法：

  * **CI（Configuration Interaction）** : 励起配置の混合
  * **MP2（Møller-Plesset摂動論）** : 摂動展開
  * **CCSD（Coupled Cluster）** : 高精度post-HF法
  * **DFT（密度汎関数理論）** : 次章で学習

Python実装: CI計算の概念（2電子系）

import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigh class SimpleCIS: """簡易的なCIS（Configuration Interaction Singles）""" def __init__(self, n_orbitals=4): """ n_orbitals: 分子軌道の数 """ self.n_orbitals = n_orbitals # ダミーの軌道エネルギー（Hartree単位） self.orbital_energies = np.array([ -0.5, # HOMO-1 -0.3, # HOMO 0.2, # LUMO 0.4 # LUMO+1 ]) # 電子数（2電子系と仮定） self.n_electrons = 2 self.homo_index = 1 # 0-indexed def ground_state_energy(self): """基底状態エネルギー（HF）""" # 占有軌道のエネルギー合計 E_HF = 2 * np.sum(self.orbital_energies[:self.n_electrons//2]) return E_HF def single_excitations(self): """1電子励起配置""" excitations = [] # HOMO → LUMO, HOMO → LUMO+1など for occ in range(self.n_electrons // 2): for virt in range(self.n_electrons // 2, self.n_orbitals): excitation_energy = self.orbital_energies[virt] - self.orbital_energies[occ] excitations.append({ 'from': occ, 'to': virt, 'energy': excitation_energy }) return excitations def cis_matrix(self): """CIS Hamiltonian行列（簡略版）""" excitations = self.single_excitations() n_exc = len(excitations) H_CIS = np.zeros((n_exc, n_exc)) # 対角要素：励起エネルギー for i, exc in enumerate(excitations): H_CIS[i, i] = exc['energy'] # 非対角要素：配置間相互作用（簡略化：無視） # 実際にはCoulomb積分とExchange積分を計算 return H_CIS, excitations def solve_cis(self): """CIS方程式を解く""" H_CIS, excitations = self.cis_matrix() eigenvalues, eigenvectors = eigh(H_CIS) return eigenvalues, eigenvectors, excitations # CIS計算 cis = SimpleCIS(n_orbitals=4) E_ground = cis.ground_state_energy() exc_energies, exc_states, excitations = cis.solve_cis() # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 軌道エネルギー図 ax1 = axes[0, 0] orb_energies = cis.orbital_energies colors_orb = ['blue', 'blue', 'red', 'red'] labels_orb = ['HOMO-1', 'HOMO', 'LUMO', 'LUMO+1'] for i, (E, color, label) in enumerate(zip(orb_energies, colors_orb, labels_orb)): ax1.hlines(E, i-0.3, i+0.3, colors=color, linewidth=3) if i < 2: # 占有軌道 ax1.plot([i-0.1, i+0.1], [E, E], 'o', color='black', markersize=8) ax1.text(i, E - 0.15, label, ha='center', fontsize=9) ax1.axhline(0, color='k', linestyle='--', linewidth=1, label='Vacuum level') ax1.set_xlim([-0.5, 3.5]) ax1.set_ylabel('Energy (Hartree)') ax1.set_title('分子軌道エネルギー準位') ax1.set_xticks([]) ax1.legend() ax1.grid(True, alpha=0.3, axis='y') # 励起状態エネルギー ax2 = axes[0, 1] state_labels = [f"S{i+1}" for i in range(len(exc_energies))] ax2.barh(range(len(exc_energies)), exc_energies, color='orange', edgecolor='black') for i, (E, label) in enumerate(zip(exc_energies, state_labels)): ax2.text(E + 0.02, i, f'{label}: {E:.3f} Ha', va='center', fontsize=10) ax2.set_xlabel('Excitation energy (Hartree)') ax2.set_ylabel('Excited state') ax2.set_title('CIS励起状態') ax2.set_yticks(range(len(exc_energies))) ax2.set_yticklabels(state_labels) ax2.grid(True, alpha=0.3, axis='x') # 励起配置の構成（最初の励起状態） ax3 = axes[1, 0] state_idx = 0 state_vector = exc_states[:, state_idx] x_pos = np.arange(len(state_vector)) ax3.bar(x_pos, np.abs(state_vector)**2, color='green', edgecolor='black') exc_labels = [f"{exc['from']}→{exc['to']}" for exc in excitations] ax3.set_xticks(x_pos) ax3.set_xticklabels(exc_labels) ax3.set_xlabel('Excitation') ax3.set_ylabel('|Coefficient|²') ax3.set_title(f'励起状態S{state_idx+1}の構成') ax3.grid(True, alpha=0.3, axis='y') # 相関エネルギーの概念図 ax4 = axes[1, 1] methods = ['HF', 'CIS', 'CISD', 'CCSD', 'FCI'] relative_corr = [0, 0.1, 0.5, 0.8, 1.0] # 相対的な相関エネルギー回復率 colors_method = ['red', 'orange', 'yellow', 'lightgreen', 'green'] bars = ax4.barh(range(len(methods)), relative_corr, color=colors_method, edgecolor='black') ax4.set_xlabel('Correlation energy recovery') ax4.set_ylabel('Method') ax4.set_title('電子相関の取り扱い（概念図）') ax4.set_yticks(range(len(methods))) ax4.set_yticklabels(methods) ax4.set_xlim([0, 1.1]) ax4.grid(True, alpha=0.3, axis='x') for i, (bar, corr) in enumerate(zip(bars, relative_corr)): width = bar.get_width() ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{corr*100:.0f}%', va='center', fontsize=10) plt.tight_layout() plt.savefig('qchem_electron_correlation.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 電子相関とCI ===\n") print(f"基底状態エネルギー（HF）: {E_ground:.6f} Hartree") print(f"\n励起状態（CIS）:") for i, E_exc in enumerate(exc_energies): print(f" S{i+1}: ΔE = {E_exc:.6f} Hartree = {E_exc * 27.2114:.3f} eV") print("\nPost-HF法の階層:") print(" HF: 平均場近似、電子相関なし") print(" CIS: 1電子励起のみ、励起状態計算") print(" CISD: 1電子・2電子励起、相関の一部") print(" CCSD: Coupled Cluster、高精度") print(" FCI: Full CI、厳密解（小分子のみ）") 

## 📚 まとめ

  * **Hartree-Fock法** は多電子系を平均場近似で解く基本手法
  * **Fock演算子** はCoulomb項と交換項を含み、自己無撞着的に解く必要がある
  * **Roothaan方程式** により基底関数展開でHF方程式を行列形式で解ける
  * **SCF法** は反復計算により自己無撞着な解に収束させる手法
  * **Gaussian型基底関数** は計算効率が高く、実用計算で広く使われる
  * **基底セット** の選択は計算精度とコストのトレードオフ
  * **Hückel法** はπ電子系の定性的理解に有用な簡易手法
  * **電子相関** はHF法で無視される重要な量子効果
  * **CI法** などのpost-HF法で電子相関を取り入れ、高精度計算が可能
  * これらの手法は密度汎関数理論（DFT）の基礎となる

### 💡 演習問題

  1. **H₂のSTO-6G計算** : STO-6G基底（6つのGaussian）を用いてH₂のSCF計算を実装し、STO-3Gとの違いを調べよ。
  2. **HeH⁺イオン** : HeH⁺の最小基底SCF計算を実装し、ポテンシャル曲線を求めよ。
  3. **ナフタレンのHückel計算** : ナフタレン（C₁₀H₈）の隣接行列を作成し、Hückel MOエネルギーと共鳴安定化エネルギーを計算せよ。
  4. **Koopmansの定理** : HF軌道エネルギーと第一イオン化エネルギーの関係を検証せよ。
  5. **基底セット重ね合わせ誤差** : H₂計算で基底セットサイズを変えて、BSSE（Basis Set Superposition Error）の影響を調べよ。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
