---
title: "🔬 第2章: 原子・分子の量子論"
chapter_title: "🔬 第2章: 原子・分子の量子論"
---

[ナレッジベース](<../../index.html>) > [MI](<../index.html>) > [量子化学入門](<index.html>) > 第2章 

## 🎯 学習目標

  * 水素原子のSchrödinger方程式を解き、原子軌道を理解する
  * 量子数（n, l, m, s）の物理的意味を学ぶ
  * 動径波動関数とBohr半径を計算する
  * 電子スピンとPauli排他原理を理解する
  * 多電子原子のHartree近似とSlater行列式を学ぶ
  * Born-Oppenheimer近似の概念を習得する
  * 水素分子イオン（H₂⁺）の電子状態を理解する
  * 変分法の原理と応用を学ぶ

## 📖 水素原子のSchrödinger方程式

### 水素原子のHamiltonian

球座標系 \\((r, \theta, \phi)\\) でのSchrödinger方程式：

\\[ \left[-\frac{\hbar^2}{2m_e}\nabla^2 - \frac{e^2}{4\pi\epsilon_0 r}\right]\psi = E\psi \\]

変数分離 \\(\psi(r, \theta, \phi) = R(r) Y_l^m(\theta, \phi)\\) により：

**動径方程式** :

\\[ -\frac{\hbar^2}{2m_e}\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) + \left[\frac{\hbar^2 l(l+1)}{2m_e r^2} - \frac{e^2}{4\pi\epsilon_0 r}\right]R = ER \\]

**量子数** :

  * 主量子数 \\(n = 1, 2, 3, \ldots\\)
  * 軌道角運動量量子数 \\(l = 0, 1, \ldots, n-1\\)
  * 磁気量子数 \\(m = -l, -l+1, \ldots, l\\)

### 水素原子の固有値と固有関数

**エネルギー固有値** :

\\[ E_n = -\frac{m_e e^4}{32\pi^2\epsilon_0^2\hbar^2 n^2} = -\frac{13.6 \text{ eV}}{n^2} \\]

**Bohr半径** :

\\[ a_0 = \frac{4\pi\epsilon_0\hbar^2}{m_e e^2} \approx 0.529 \text{ Å} \\]

**原子軌道** \\(\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)\\)：

  * \\(n=1, l=0\\): 1s軌道
  * \\(n=2, l=0\\): 2s軌道、\\(n=2, l=1\\): 2p軌道
  * \\(n=3, l=0\\): 3s軌道、\\(l=1\\): 3p軌道、\\(l=2\\): 3d軌道

## 💻 例題2.1: 水素原子の原子軌道

### 動径波動関数

1s軌道（\\(n=1, l=0\\)）:

\\[ R_{10}(r) = 2\left(\frac{1}{a_0}\right)^{3/2} e^{-r/a_0} \\]

2s軌道（\\(n=2, l=0\\)）:

\\[ R_{20}(r) = \frac{1}{2\sqrt{2}}\left(\frac{1}{a_0}\right)^{3/2}\left(2 - \frac{r}{a_0}\right) e^{-r/(2a_0)} \\]

2p軌道（\\(n=2, l=1\\)）:

\\[ R_{21}(r) = \frac{1}{2\sqrt{6}}\left(\frac{1}{a_0}\right)^{3/2}\frac{r}{a_0} e^{-r/(2a_0)} \\]

Python実装: 水素原子の原子軌道

import numpy as np import matplotlib.pyplot as plt from scipy.special import sph_harm, genlaguerre, factorial # 物理定数（SI単位系） hbar = 1.054571817e-34 # J·s m_e = 9.1093837015e-31 # kg e = 1.602176634e-19 # C epsilon_0 = 8.8541878128e-12 # F/m a_0 = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2) # Bohr radius # 原子単位系（簡略化） a_0_au = 1.0 # Bohr Ry = 13.6 # eV (Rydberg constant) def hydrogen_energy(n): """水素原子のエネルギー準位 (eV)""" return -Ry / n**2 def radial_wavefunction(r, n, l, a_0=1.0): """動径波動関数 R_nl(r)（原子単位系）""" rho = 2 * r / (n * a_0) # Laguerre多項式 L = genlaguerre(n - l - 1, 2*l + 1) # 規格化定数 norm = np.sqrt((2/(n*a_0))**3 * factorial(n - l - 1) / (2*n*factorial(n + l))) R_nl = norm * np.exp(-rho/2) * rho**l * L(rho) return R_nl def radial_probability(r, n, l, a_0=1.0): """動径確率密度 r² |R_nl(r)|²""" R = radial_wavefunction(r, n, l, a_0) return r**2 * R**2 # 動径座標 r = np.linspace(0, 30, 500) # Bohr単位 # プロット fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 動径波動関数 ax1 = axes[0, 0] orbitals = [(1, 0, '1s'), (2, 0, '2s'), (2, 1, '2p'), (3, 0, '3s'), (3, 1, '3p'), (3, 2, '3d')] for n, l, label in orbitals[:5]: R = radial_wavefunction(r, n, l, a_0_au) ax1.plot(r, R, linewidth=2, label=label) ax1.set_xlabel('r (Bohr radii)') ax1.set_ylabel('R_nl(r)') ax1.set_title('動径波動関数') ax1.legend() ax1.grid(True, alpha=0.3) ax1.axhline(0, color='k', linewidth=0.5) # 動径確率密度 ax2 = axes[0, 1] for n, l, label in orbitals[:5]: prob = radial_probability(r, n, l, a_0_au) ax2.plot(r, prob, linewidth=2, label=label) ax2.set_xlabel('r (Bohr radii)') ax2.set_ylabel('r² |R_nl(r)|²') ax2.set_title('動径確率密度') ax2.legend() ax2.grid(True, alpha=0.3) # エネルギー準位図 ax3 = axes[1, 0] n_max = 6 energies = [hydrogen_energy(n) for n in range(1, n_max+1)] degeneracies = [n**2 for n in range(1, n_max+1)] for n, E, g in zip(range(1, n_max+1), energies, degeneracies): ax3.hlines(E, 0, 1, colors='blue', linewidth=2) ax3.text(1.1, E, f'n={n}, E={E:.2f} eV ({g} states)', fontsize=10, va='center') ax3.set_xlim([-0.1, 3]) ax3.set_ylim([min(energies) * 1.1, 0]) ax3.set_ylabel('Energy (eV)') ax3.set_title('水素原子のエネルギー準位') ax3.axhline(0, color='k', linestyle='--', linewidth=1, label='Ionization') ax3.grid(True, alpha=0.3, axis='y') ax3.set_xticks([]) ax3.legend() # 最大確率半径 ax4 = axes[1, 1] r_max_values = [] n_range = range(1, 10) for n in n_range: for l in range(n): r_fine = np.linspace(0.1, 50, 1000) prob = radial_probability(r_fine, n, l, a_0_au) r_max = r_fine[np.argmax(prob)] r_max_values.append((n, l, r_max)) # 量子数ごとに色分け colors_l = {0: 'blue', 1: 'red', 2: 'green', 3: 'purple'} markers_l = {0: 'o', 1: 's', 2: '^', 3: 'd'} for l_val in [0, 1, 2]: data = [(n, r_max) for n, l, r_max in r_max_values if l == l_val] if data: ns, r_maxs = zip(*data) label_l = ['s', 'p', 'd', 'f'][l_val] ax4.plot(ns, r_maxs, markers_l[l_val], color=colors_l[l_val], markersize=8, linewidth=2, label=f'{label_l} orbitals') ax4.set_xlabel('Principal quantum number n') ax4.set_ylabel('Most probable radius (Bohr radii)') ax4.set_title('最大確率半径') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_hydrogen_atom.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 水素原子の原子軌道 ===\n") print(f"Bohr半径 a₀ = {a_0*1e10:.4f} Å") print(f"Rydberg定数 Ry = {Ry} eV\n") print("エネルギー準位:") for n in range(1, 7): E_n = hydrogen_energy(n) print(f" n = {n}: E = {E_n:.4f} eV ({n**2} 縮退)") print("\n最大確率半径（いくつかの軌道）:") for n, l, r_max in r_max_values[:10]: orbital_name = ['s', 'p', 'd', 'f'][l] print(f" {n}{orbital_name}: r_max = {r_max:.2f} a₀") 

## 💻 例題2.2: 電子スピンとPauli排他原理

### 電子スピン

電子は固有の**スピン角運動量** \\(\mathbf{S}\\) を持ちます：

  * スピン量子数 \\(s = 1/2\\)
  * スピン磁気量子数 \\(m_s = \pm 1/2\\)（スピンアップ \\(\uparrow\\)、スピンダウン \\(\downarrow\\)）

**Pauli排他原理** :

同一の量子状態 \\((n, l, m, m_s)\\) に2つ以上のフェルミオンは入れない。

原子軌道1つに最大2個の電子（スピン対）が入る。

### 多電子原子の電子配置

**Aufbau原理** : エネルギーの低い軌道から電子を詰める

軌道のエネルギー順序（近似）:

1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < ...

**Hund則** :

  1. 同じエネルギーの軌道には、スピンを平行にして電子を配置
  2. 縮退軌道を電子で満たしてから対を作る

Python実装: 電子配置とスペクトル項

import numpy as np import matplotlib.pyplot as plt # 原子番号と電子配置の辞書 electron_configs = { 1: '1s¹', # H 2: '1s²', # He 3: '[He]2s¹', # Li 4: '[He]2s²', # Be 5: '[He]2s²2p¹', # B 6: '[He]2s²2p²', # C 7: '[He]2s²2p³', # N 8: '[He]2s²2p⁴', # O 9: '[He]2s²2p⁵', # F 10: '[He]2s²2p⁶', # Ne 11: '[Ne]3s¹', # Na 18: '[Ne]3s²3p⁶', # Ar 26: '[Ar]3d⁶4s²', # Fe 29: '[Ar]3d¹⁰4s¹', # Cu } element_names = { 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 18: 'Ar', 26: 'Fe', 29: 'Cu' } # 第一イオン化エネルギー（eV） ionization_energies = { 1: 13.6, 2: 24.6, 3: 5.4, 4: 9.3, 5: 8.3, 6: 11.3, 7: 14.5, 8: 13.6, 9: 17.4, 10: 21.6, 11: 5.1, 18: 15.8 } def orbital_diagram(config_str): """電子配置の軌道図""" # 簡易パーサー（実装は省略、概念的） pass # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 電子配置の周期性（原子番号 1-18） ax1 = axes[0, 0] Z_range = list(range(1, 11)) + [11, 18] configs = [electron_configs.get(Z, '') for Z in Z_range] names = [element_names.get(Z, '') for Z in Z_range] ax1.barh(range(len(Z_range)), Z_range, color='skyblue', edgecolor='black') for i, (Z, name, config) in enumerate(zip(Z_range, names, configs)): ax1.text(Z + 0.5, i, f'{name} ({Z}): {config}', va='center', fontsize=9) ax1.set_yticks(range(len(Z_range))) ax1.set_yticklabels(names) ax1.set_xlabel('Atomic number Z') ax1.set_title('電子配置（周期表順）') ax1.grid(True, alpha=0.3, axis='x') # イオン化エネルギーの周期性 ax2 = axes[0, 1] IE_Z = sorted(ionization_energies.keys()) IE_values = [ionization_energies[Z] for Z in IE_Z] ax2.plot(IE_Z, IE_values, 'o-', linewidth=2, markersize=8, color='red') for Z, IE in zip(IE_Z, IE_values): ax2.text(Z, IE + 0.5, element_names[Z], ha='center', fontsize=9) ax2.set_xlabel('Atomic number Z') ax2.set_ylabel('Ionization energy (eV)') ax2.set_title('第一イオン化エネルギー') ax2.grid(True, alpha=0.3) # スピン多重度（C原子を例に） ax3 = axes[1, 0] # C原子: 1s² 2s² 2p² # 2p軌道に2電子（Hund則により平行スピン） orbital_labels = ['2p_x', '2p_y', '2p_z'] spins_up = [1, 1, 0] # アップスピンの数 spins_down = [0, 0, 0] # ダウンスピンの数 x = np.arange(len(orbital_labels)) width = 0.35 bars_up = ax3.bar(x - width/2, spins_up, width, label='Spin up (↑)', color='blue') bars_down = ax3.bar(x + width/2, spins_down, width, label='Spin down (↓)', color='red') ax3.set_ylabel('Number of electrons') ax3.set_xlabel('2p orbitals') ax3.set_title('C原子の2p電子配置（Hund則）') ax3.set_xticks(x) ax3.set_xticklabels(orbital_labels) ax3.set_ylim([0, 2]) ax3.legend() ax3.grid(True, alpha=0.3, axis='y') # 軌道エネルギー図（水素型と多電子原子の比較） ax4 = axes[1, 1] # 水素型（n依存のみ） E_H = {1: -13.6, 2: -3.4, 3: -1.51} orbitals_H = {'1s': (1, -13.6), '2s': (2, -3.4), '2p': (2, -3.4), '3s': (3, -1.51), '3p': (3, -1.51), '3d': (3, -1.51)} # 多電子原子（l依存性あり、概念的） E_multi = {'1s': -20, '2s': -6, '2p': -5, '3s': -2.5, '3p': -2, '3d': -0.5} x_pos_H = [0.3, 0.8, 1.0, 1.5, 1.7, 1.9] x_pos_multi = [0.5, 1.0, 1.2, 1.7, 1.9, 2.1] orbital_order = ['1s', '2s', '2p', '3s', '3p', '3d'] for i, orb in enumerate(orbital_order): if orb in orbitals_H: ax4.hlines(orbitals_H[orb][1], x_pos_H[i]-0.1, x_pos_H[i]+0.1, colors='blue', linewidth=2) if orb in E_multi: ax4.hlines(E_multi[orb], x_pos_multi[i]-0.1, x_pos_multi[i]+0.1, colors='red', linewidth=2) ax4.plot([], [], 'b-', linewidth=2, label='Hydrogen-like') ax4.plot([], [], 'r-', linewidth=2, label='Multi-electron') for i, orb in enumerate(orbital_order): ax4.text((x_pos_H[i] + x_pos_multi[i])/2, -22, orb, ha='center', fontsize=10) ax4.set_xlim([0, 2.5]) ax4.set_ylim([-25, 0]) ax4.set_ylabel('Energy (eV)') ax4.set_title('軌道エネルギー順序（概念図）') ax4.legend() ax4.grid(True, alpha=0.3, axis='y') ax4.set_xticks([]) plt.tight_layout() plt.savefig('qchem_electron_config.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 電子配置とPauli排他原理 ===\n") print("いくつかの元素の電子配置:") for Z in [1, 2, 6, 7, 8, 10]: print(f" {element_names[Z]} (Z={Z}): {electron_configs[Z]}") print("\nPauli排他原理:") print(" - 同一量子状態 (n, l, m, m_s) に最大1電子") print(" - 1つの軌道に最大2電子（スピン対）") print("\nHund則（C原子の例）:") print(" 2p²: ↑ ↑ _ （スピン平行、S=1, ³P基底状態）") 

## 💻 例題2.3: Born-Oppenheimer近似と分子の電子状態

### Born-Oppenheimer近似

電子は核に比べて遥かに軽い（\\(m_e/m_p \approx 1/1836\\)）ため、核の運動と電子の運動を分離できます：

  * 核の位置 \\(\mathbf{R}\\) を固定して電子のSchrödinger方程式を解く
  * 電子エネルギー \\(E_{el}(\mathbf{R})\\) が核のポテンシャルとなる
  * 核の運動は \\(E_{el}(\mathbf{R})\\) 上で振動・回転する

**分子の全Hamiltonian** :

\\[ \hat{H} = \hat{T}_{nuclei} + \hat{H}_{el}(\mathbf{R}) \\]

\\(\hat{H}_{el}(\mathbf{R})\\) は電子のHamiltonianで、核座標 \\(\mathbf{R}\\) をパラメータとして含みます。

### 水素分子イオン（H₂⁺）

最も単純な分子：2つの陽子（A, B）と1つの電子

核間距離 \\(R\\) を固定した電子のHamiltonian:

\\[ \hat{H}_{el} = -\frac{\hbar^2}{2m_e}\nabla^2 - \frac{e^2}{4\pi\epsilon_0 r_A} - \frac{e^2}{4\pi\epsilon_0 r_B} \\]

LCAO（原子軌道の線形結合）近似:

\\[ \psi_\pm = N_\pm(\phi_A \pm \phi_B) \\]

  * \\(\psi_+\\): 結合性軌道（bonding）
  * \\(\psi_-\\): 反結合性軌道（antibonding）

Python実装: H₂⁺のポテンシャルエネルギー曲線

import numpy as np import matplotlib.pyplot as plt # 原子単位系 a_0 = 1.0 # Bohr E_h = 1.0 # Hartree def overlap_integral(R, a_0=1.0): """重なり積分 S(R)（簡易近似）""" S = np.exp(-R/a_0) * (1 + R/a_0 + (R/a_0)**2/3) return S def coulomb_integral(R, a_0=1.0): """Coulomb積分 H_AA（簡易近似）""" H_AA = -1/a_0**2 - (1/R) * (1 + 1/R) * np.exp(-2*R/a_0) return H_AA def exchange_integral(R, a_0=1.0): """交換積分 H_AB（簡易近似）""" S = overlap_integral(R, a_0) H_AB = (-S/R - S/(a_0) * (1 + R/a_0)) * np.exp(-R/a_0) return H_AB def h2plus_energy(R, bonding=True, a_0=1.0): """H₂⁺のエネルギー（LCAO近似）""" S = overlap_integral(R, a_0) H_AA = coulomb_integral(R, a_0) H_AB = exchange_integral(R, a_0) # 核間反発 V_NN = 1 / R if bonding: # 結合性軌道 E = (H_AA + H_AB) / (1 + S) + V_NN else: # 反結合性軌道 E = (H_AA - H_AB) / (1 - S) + V_NN return E # 核間距離 R_range = np.linspace(0.5, 10, 200) # ポテンシャルエネルギー曲線 E_bonding = [h2plus_energy(R, bonding=True) for R in R_range] E_antibonding = [h2plus_energy(R, bonding=False) for R in R_range] # 平衡核間距離と解離エネルギー E_bonding_array = np.array(E_bonding) R_eq_idx = np.argmin(E_bonding_array) R_eq = R_range[R_eq_idx] E_eq = E_bonding_array[R_eq_idx] E_dissociation = 0 # 解離極限（H + H⁺） D_e = E_dissociation - E_eq # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # ポテンシャルエネルギー曲線 ax1 = axes[0, 0] ax1.plot(R_range, E_bonding, 'b-', linewidth=2, label='Bonding (σ_g)') ax1.plot(R_range, E_antibonding, 'r-', linewidth=2, label='Antibonding (σ_u*)') ax1.axhline(0, color='k', linestyle='--', linewidth=1, label='Dissociation limit') ax1.plot(R_eq, E_eq, 'go', markersize=10, label=f'Equilibrium (R={R_eq:.2f} a₀)') ax1.set_xlabel('Internuclear distance R (Bohr)') ax1.set_ylabel('Energy (Hartree)') ax1.set_title('H₂⁺ ポテンシャルエネルギー曲線') ax1.set_ylim([-1.5, 2]) ax1.legend() ax1.grid(True, alpha=0.3) # 分子軌道の形状（1次元断面） ax2 = axes[0, 1] R_vis = 2.0 # 核間距離（可視化用） x = np.linspace(-5, 5, 200) # 原子軌道（1s） phi_A = np.exp(-np.abs(x + R_vis/2)) phi_B = np.exp(-np.abs(x - R_vis/2)) # 分子軌道 S_vis = overlap_integral(R_vis) psi_bonding = (phi_A + phi_B) / np.sqrt(2 * (1 + S_vis)) psi_antibonding = (phi_A - phi_B) / np.sqrt(2 * (1 - S_vis)) ax2.plot(x, psi_bonding, 'b-', linewidth=2, label='Bonding MO (σ_g)') ax2.plot(x, psi_antibonding, 'r-', linewidth=2, label='Antibonding MO (σ_u*)') ax2.axvline(-R_vis/2, color='k', linestyle=':', linewidth=1, label='Nuclei') ax2.axvline(R_vis/2, color='k', linestyle=':', linewidth=1) ax2.axhline(0, color='k', linewidth=0.5) ax2.set_xlabel('Position (Bohr)') ax2.set_ylabel('Wavefunction ψ(x)') ax2.set_title(f'分子軌道（R = {R_vis} a₀）') ax2.legend() ax2.grid(True, alpha=0.3) # 電子密度 ax3 = axes[1, 0] rho_bonding = psi_bonding**2 rho_antibonding = psi_antibonding**2 ax3.plot(x, rho_bonding, 'b-', linewidth=2, label='Bonding |ψ|²') ax3.plot(x, rho_antibonding, 'r-', linewidth=2, label='Antibonding |ψ|²') ax3.axvline(-R_vis/2, color='k', linestyle=':', linewidth=1) ax3.axvline(R_vis/2, color='k', linestyle=':', linewidth=1) ax3.set_xlabel('Position (Bohr)') ax3.set_ylabel('Electron density |ψ|²') ax3.set_title('電子密度分布') ax3.legend() ax3.grid(True, alpha=0.3) # エネルギー準位図 ax4 = axes[1, 1] # 原子軌道エネルギー E_1s = -0.5 # H原子の1sエネルギー（Hartree単位） # 分子軌道エネルギー（R = R_eq） E_sigma_g = h2plus_energy(R_eq, bonding=True) - 1/R_eq # 核反発を除く E_sigma_u = h2plus_energy(R_eq, bonding=False) - 1/R_eq # プロット ax4.hlines(E_1s, 0, 0.3, colors='blue', linewidth=3, label='H 1s') ax4.hlines(E_1s, 0.7, 1.0, colors='blue', linewidth=3) ax4.hlines(E_sigma_g, 0.35, 0.65, colors='green', linewidth=3, label='σ_g (bonding)') ax4.hlines(E_sigma_u, 0.35, 0.65, colors='red', linewidth=3, label='σ_u* (antibonding)') # 電子配置（↑ = 1電子） ax4.plot(0.5, E_sigma_g, 'o', color='black', markersize=10) ax4.text(0.5, E_sigma_g - 0.15, '↑', fontsize=16, ha='center') ax4.set_xlim([-0.1, 1.1]) ax4.set_ylim([-0.8, 0.2]) ax4.set_ylabel('Energy (Hartree)') ax4.set_title('H₂⁺ 分子軌道エネルギー準位') ax4.set_xticks([0.15, 0.5, 0.85]) ax4.set_xticklabels(['H(A)', 'H₂⁺', 'H(B)']) ax4.legend(loc='upper right') ax4.grid(True, alpha=0.3, axis='y') plt.tight_layout() plt.savefig('qchem_h2plus_molecule.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== H₂⁺ 分子（LCAO近似）===\n") print(f"平衡核間距離 R_eq = {R_eq:.2f} Bohr = {R_eq * 0.529:.2f} Å") print(f"平衡エネルギー E_eq = {E_eq:.4f} Hartree = {E_eq * 27.2:.2f} eV") print(f"解離エネルギー D_e = {D_e:.4f} Hartree = {D_e * 27.2:.2f} eV") print(f"\n実験値との比較:") print(f" R_eq (実験) ≈ 1.06 Å") print(f" D_e (実験) ≈ 2.79 eV") print(f"\nLCAO近似は定性的に正しいが、定量的改善には基底関数の拡張が必要") 

## 💻 例題2.4: 変分法の原理

### 変分原理

任意の規格化された試行関数 \\(\phi\\) に対し：

\\[ E[\phi] = \frac{\langle \phi | \hat{H} | \phi \rangle}{\langle \phi | \phi \rangle} \geq E_0 \\]

ここで \\(E_0\\) は基底状態の厳密なエネルギーです。

**変分法** :

  1. パラメータ \\(\alpha\\) を含む試行関数 \\(\phi(\alpha)\\) を用意
  2. エネルギー期待値 \\(E(\alpha) = \langle \phi(\alpha) | \hat{H} | \phi(\alpha) \rangle\\) を計算
  3. \\(\frac{\partial E}{\partial \alpha} = 0\\) を解いて最適パラメータを求める

最適化された \\(E(\alpha_{opt})\\) が基底状態エネルギーの上界を与えます。

Python実装: 変分法による基底状態計算

import numpy as np import matplotlib.pyplot as plt from scipy.optimize import minimize_scalar from scipy.integrate import quad # 調和振動子のHamiltonian（原子単位） hbar = 1.0 m = 1.0 omega = 1.0 def harmonic_potential(x, omega=1.0): """調和振動子ポテンシャル""" return 0.5 * omega**2 * x**2 def trial_wavefunction_gaussian(x, alpha): """試行関数（Gaussian）""" return (alpha / np.pi)**0.25 * np.exp(-alpha * x**2 / 2) def kinetic_energy_expectation(alpha): """運動エネルギーの期待値 ⟨T⟩""" # ⟨T⟩ = ∫ ψ* (-ℏ²/2m d²/dx²) ψ dx = ℏ²α/(4m) return hbar**2 * alpha / (4 * m) def potential_energy_expectation(alpha, omega=1.0): """ポテンシャルエネルギーの期待値 ⟨V⟩""" # ⟨V⟩ = ∫ ψ* (1/2 m ω² x²) ψ dx = m ω²/(4α) return m * omega**2 / (4 * alpha) def energy_expectation(alpha, omega=1.0): """全エネルギーの期待値 E(α)""" T = kinetic_energy_expectation(alpha) V = potential_energy_expectation(alpha, omega) return T + V # 変分パラメータの範囲 alpha_range = np.linspace(0.1, 3.0, 200) E_variational = [energy_expectation(alpha, omega) for alpha in alpha_range] # 最適パラメータ result = minimize_scalar(lambda a: energy_expectation(a, omega), bounds=(0.1, 10), method='bounded') alpha_opt = result.x E_opt = result.fun # 厳密解 E_exact = 0.5 * hbar * omega # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # エネルギー期待値 vs 変分パラメータ ax1 = axes[0, 0] ax1.plot(alpha_range, E_variational, 'b-', linewidth=2, label='E(α)') ax1.axhline(E_exact, color='r', linestyle='--', linewidth=2, label=f'Exact E₀ = {E_exact:.3f}') ax1.plot(alpha_opt, E_opt, 'go', markersize=10, label=f'Optimum α = {alpha_opt:.3f}') ax1.set_xlabel('Variational parameter α') ax1.set_ylabel('Energy E(α)') ax1.set_title('変分エネルギー') ax1.legend() ax1.grid(True, alpha=0.3) # 試行関数と厳密解の比較 ax2 = axes[0, 1] x = np.linspace(-4, 4, 500) psi_trial = trial_wavefunction_gaussian(x, alpha_opt) psi_exact = (omega / np.pi)**0.25 * np.exp(-omega * x**2 / 2) ax2.plot(x, psi_trial, 'b-', linewidth=2, label=f'Trial (α={alpha_opt:.3f})') ax2.plot(x, psi_exact, 'r--', linewidth=2, label='Exact') ax2.set_xlabel('Position x') ax2.set_ylabel('Wavefunction ψ(x)') ax2.set_title('波動関数の比較') ax2.legend() ax2.grid(True, alpha=0.3) # 運動エネルギーとポテンシャルエネルギー ax3 = axes[1, 0] T_values = [kinetic_energy_expectation(alpha) for alpha in alpha_range] V_values = [potential_energy_expectation(alpha, omega) for alpha in alpha_range] ax3.plot(alpha_range, T_values, 'b-', linewidth=2, label='⟨T⟩') ax3.plot(alpha_range, V_values, 'r-', linewidth=2, label='⟨V⟩') ax3.plot(alpha_range, E_variational, 'g-', linewidth=2, label='⟨E⟩ = ⟨T⟩ + ⟨V⟩') ax3.axvline(alpha_opt, color='k', linestyle=':', linewidth=1, label=f'α_opt') ax3.set_xlabel('Variational parameter α') ax3.set_ylabel('Energy') ax3.set_title('エネルギー成分') ax3.legend() ax3.grid(True, alpha=0.3) # Virial定理の検証 ax4 = axes[1, 1] # 調和振動子では ⟨T⟩ = ⟨V⟩ (Virial定理) T_at_opt = kinetic_energy_expectation(alpha_opt) V_at_opt = potential_energy_expectation(alpha_opt, omega) ratio = np.array(T_values) / np.array(V_values) ax4.plot(alpha_range, ratio, 'purple', linewidth=2, label='⟨T⟩ / ⟨V⟩') ax4.axhline(1, color='r', linestyle='--', linewidth=2, label='Virial theorem (=1)') ax4.axvline(alpha_opt, color='k', linestyle=':', linewidth=1) ax4.set_xlabel('Variational parameter α') ax4.set_ylabel('⟨T⟩ / ⟨V⟩') ax4.set_title('Virial定理') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_variational_method.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 変分法による基底状態計算 ===\n") print(f"試行関数: ψ(x; α) = (α/π)^(1/4) exp(-αx²/2)") print(f"\n最適化結果:") print(f" 最適パラメータ α_opt = {alpha_opt:.6f}") print(f" 変分エネルギー E(α_opt) = {E_opt:.6f}") print(f" 厳密エネルギー E_exact = {E_exact:.6f}") print(f" 相対誤差 = {abs(E_opt - E_exact)/E_exact * 100:.6f} %") print(f"\nVirial定理の確認:") print(f" ⟨T⟩ = {T_at_opt:.6f}") print(f" ⟨V⟩ = {V_at_opt:.6f}") print(f" ⟨T⟩ / ⟨V⟩ = {T_at_opt / V_at_opt:.6f} ≈ 1") 

## 📚 まとめ

  * **水素原子** は唯一厳密解が得られる多体問題で、原子軌道の基礎となる
  * **量子数** (n, l, m, m_s) が電子の状態を完全に指定する
  * **動径波動関数** と**球面調和関数** の積で原子軌道が記述される
  * **Pauli排他原理** により同一量子状態に2つ以上の電子は入れない
  * **電子配置** はAufbau原理、Hund則、Pauli排他原理に従う
  * **Born-Oppenheimer近似** により核と電子の運動を分離でき、分子の電子状態を計算可能
  * **LCAO近似** は分子軌道を原子軌道の線形結合で構築する基本手法
  * **変分法** は試行関数を用いて基底状態エネルギーの上界を得る強力な手法
  * これらの概念は分子軌道法やDFTの基礎となる

### 💡 演習問題

  1. **水素原子の励起状態** : n=3の全ての軌道（3s, 3p, 3d）の動径確率密度を計算し、最大確率半径を求めよ。
  2. **He⁺イオン** : Z=2の水素型イオンHe⁺のエネルギー準位とBohr半径を計算せよ。
  3. **N原子の電子配置** : N原子（Z=7）の基底状態電子配置を書き、Hund則により全スピンSと軌道角運動量Lを求めよ。
  4. **H₂分子** : H₂⁺の結果を拡張して、H₂分子（2電子）のLCAO計算を実装せよ。
  5. **変分法の応用** : 水素原子の1s軌道を試行関数 ψ(r; α) = exp(-αr) で近似し、最適αを変分法で求めよ。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
