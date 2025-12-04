---
title: "🔬 第1章: 量子力学の基礎"
chapter_title: "🔬 第1章: 量子力学の基礎"
---

[ナレッジベース](<../../index.html>) > [MI](<../index.html>) > [量子化学入門](<index.html>) > 第1章 

## 🎯 学習目標

  * 波動関数と確率解釈の概念を理解する
  * 時間依存・時間非依存Schrödinger方程式を学ぶ
  * 演算子と物理量の期待値の計算方法を習得する
  * 1次元箱の中の粒子の厳密解を導出する
  * 調和振動子の固有関数とエネルギー準位を計算する
  * 不確定性原理と交換関係を理解する
  * 角運動量演算子と球面調和関数を学ぶ
  * Pythonで量子系のシミュレーションを実装する

## 📖 波動関数とSchrödinger方程式

### 波動関数と確率解釈

量子力学では、粒子の状態を**波動関数** \\(\psi(\mathbf{r}, t)\\) で記述します：

  * \\(|\psi(\mathbf{r}, t)|^2\\) は位置 \\(\mathbf{r}\\) に時刻 \\(t\\) で粒子を見出す確率密度
  * 規格化条件: \\(\int |\psi(\mathbf{r}, t)|^2 d^3r = 1\\)
  * 波動関数は複素数値関数

**時間依存Schrödinger方程式** :

\\[ i\hbar \frac{\partial \psi}{\partial t} = \hat{H} \psi \\]

ここで \\(\hat{H}\\) はHamiltonianパラメータで、系の全エネルギーを表します。

### 時間非依存Schrödinger方程式

定常状態（エネルギー固有状態）では \\(\psi(\mathbf{r}, t) = \phi(\mathbf{r}) e^{-iEt/\hbar}\\) と変数分離できます：

\\[ \hat{H} \phi = E \phi \\]

これが**時間非依存Schrödinger方程式** （固有値問題）です。

1次元の場合:

\\[ -\frac{\hbar^2}{2m} \frac{d^2\phi}{dx^2} + V(x)\phi = E\phi \\]

## 💻 例題1.1: 1次元箱の中の粒子

### 無限ポテンシャル井戸

幅 \\(L\\) の箱、ポテンシャル \\(V(x) = 0\\) (\\(0 < x < L\\))、\\(V(x) = \infty\\) (その他)：

**固有関数** :

\\[ \phi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right), \quad n = 1, 2, 3, \ldots \\]

**エネルギー固有値** :

\\[ E_n = \frac{n^2 \pi^2 \hbar^2}{2m L^2} \\]

Python実装: 箱の中の粒子

import numpy as np import matplotlib.pyplot as plt # 物理定数（原子単位系） hbar = 1.0 m = 1.0 L = 1.0 def energy_level(n, L, m, hbar): """エネルギー準位""" return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2) def wavefunction(x, n, L): """規格化された固有関数""" return np.sqrt(2/L) * np.sin(n * np.pi * x / L) def probability_density(x, n, L): """確率密度""" psi = wavefunction(x, n, L) return np.abs(psi)**2 # 座標 x = np.linspace(0, L, 500) # 異なる量子数 quantum_numbers = [1, 2, 3, 4] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 波動関数 ax1 = axes[0, 0] for n in quantum_numbers: psi = wavefunction(x, n, L) E_n = energy_level(n, L, m, hbar) ax1.plot(x, psi, linewidth=2, label=f'n={n}, E={E_n:.2f}') ax1.set_xlabel('Position x') ax1.set_ylabel('Wavefunction ψ_n(x)') ax1.set_title('固有関数（波動関数）') ax1.legend() ax1.grid(True, alpha=0.3) ax1.axhline(0, color='k', linewidth=0.5) # 確率密度 ax2 = axes[0, 1] for n in quantum_numbers: rho = probability_density(x, n, L) ax2.plot(x, rho, linewidth=2, label=f'n={n}') ax2.set_xlabel('Position x') ax2.set_ylabel('Probability density |ψ_n(x)|²') ax2.set_title('確率密度分布') ax2.legend() ax2.grid(True, alpha=0.3) # エネルギー準位図 ax3 = axes[1, 0] n_max = 10 energies = [energy_level(n, L, m, hbar) for n in range(1, n_max+1)] for i, (n, E) in enumerate(zip(range(1, n_max+1), energies)): ax3.hlines(E, 0, 1, colors='blue', linewidth=2) ax3.text(1.1, E, f'n={n}, E={E:.2f}', fontsize=10, va='center') ax3.set_xlim([-0.1, 2]) ax3.set_ylim([0, max(energies) * 1.1]) ax3.set_ylabel('Energy E_n') ax3.set_title('エネルギー準位') ax3.grid(True, alpha=0.3, axis='y') ax3.set_xticks([]) # 期待値計算（位置の期待値） ax4 = axes[1, 1] expectation_x = [] for n in range(1, n_max+1): psi = wavefunction(x, n, L) #  = ∫ ψ*(x) x ψ(x) dx integrand = np.conj(psi) * x * psi exp_x = np.trapz(integrand, x) expectation_x.append(exp_x) ax4.plot(range(1, n_max+1), expectation_x, 'o-', linewidth=2, markersize=8) ax4.axhline(L/2, color='r', linestyle='--', linewidth=2, label='Classical (L/2)') ax4.set_xlabel('Quantum number n') ax4.set_ylabel('⟨x⟩') ax4.set_title('位置の期待値') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_particle_in_box.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("=== 1次元箱の中の粒子 ===\n") print(f"箱の幅 L = {L}") print(f"質量 m = {m}") print(f"\nエネルギー準位:") for n in range(1, 6): E_n = energy_level(n, L, m, hbar) print(f" n = {n}: E_{n} = {E_n:.4f} (原子単位)") print(f"\n位置の期待値（全て x = L/2 = {L/2}）:") print(f" 対称性により、すべての状態で ⟨x⟩ = L/2") 

## 💻 例題1.2: 調和振動子

### 量子調和振動子

ポテンシャル \\(V(x) = \frac{1}{2}m\omega^2 x^2\\) の系：

**エネルギー固有値** :

\\[ E_n = \hbar\omega \left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, \ldots \\]

**固有関数** :

\\[ \phi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n(\xi) e^{-\xi^2/2} \\]

ここで \\(\xi = \sqrt{m\omega/\hbar} \cdot x\\)、\\(H_n\\) はHermite多項式です。

Python実装: 量子調和振動子

import numpy as np import matplotlib.pyplot as plt from scipy.special import hermite, factorial # 物理定数 hbar = 1.0 m = 1.0 omega = 1.0 def energy_harmonic(n, omega, hbar): """調和振動子のエネルギー""" return hbar * omega * (n + 0.5) def harmonic_wavefunction(x, n, m, omega, hbar): """調和振動子の固有関数""" xi = np.sqrt(m * omega / hbar) * x normalization = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * factorial(n)) H_n = hermite(n) psi = normalization * H_n(xi) * np.exp(-xi**2 / 2) return psi def classical_turning_point(n, omega, m): """古典的転回点（E = V）""" E_n = energy_harmonic(n, omega, hbar) return np.sqrt(2 * E_n / (m * omega**2)) # 座標範囲 x_max = 5 x = np.linspace(-x_max, x_max, 500) # ポテンシャル V = 0.5 * m * omega**2 * x**2 # 異なる量子数 quantum_numbers = [0, 1, 2, 3, 5, 10] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 波動関数とポテンシャル ax1 = axes[0, 0] ax1.plot(x, V, 'k--', linewidth=2, label='Potential V(x)') for n in [0, 1, 2, 3, 5]: E_n = energy_harmonic(n, omega, hbar) psi = harmonic_wavefunction(x, n, m, omega, hbar) # エネルギー準位で平行移動して表示 ax1.plot(x, E_n + psi * 2, linewidth=1.5, label=f'n={n}') ax1.hlines(E_n, -x_max, x_max, colors='gray', linestyles=':', linewidth=0.8) ax1.set_xlabel('Position x') ax1.set_ylabel('Energy / Wavefunction') ax1.set_title('調和振動子の固有関数とエネルギー準位') ax1.legend() ax1.grid(True, alpha=0.3) ax1.set_ylim([0, 12]) # 確率密度（低量子数） ax2 = axes[0, 1] for n in [0, 1, 2, 3]: psi = harmonic_wavefunction(x, n, m, omega, hbar) rho = np.abs(psi)**2 ax2.plot(x, rho, linewidth=2, label=f'n={n}') ax2.set_xlabel('Position x') ax2.set_ylabel('Probability density |ψ_n(x)|²') ax2.set_title('確率密度分布（低励起状態）') ax2.legend() ax2.grid(True, alpha=0.3) # 古典極限（高量子数） ax3 = axes[1, 0] n_high = 10 psi_high = harmonic_wavefunction(x, n_high, m, omega, hbar) rho_high = np.abs(psi_high)**2 # 古典的確率密度 E_classical = energy_harmonic(n_high, omega, hbar) x_turn = classical_turning_point(n_high, omega, m) rho_classical = np.zeros_like(x) mask = np.abs(x) < x_turn # 古典的には確率 ∝ 1/v ∝ 1/√(E - V) rho_classical[mask] = 1 / np.sqrt(E_classical - 0.5 * m * omega**2 * x[mask]**2) rho_classical /= np.trapz(rho_classical, x) # 規格化 ax3.plot(x, rho_high, 'b-', linewidth=2, label=f'Quantum (n={n_high})') ax3.plot(x, rho_classical, 'r--', linewidth=2, label='Classical') ax3.axvline(-x_turn, color='k', linestyle=':', linewidth=1, label='Turning points') ax3.axvline(x_turn, color='k', linestyle=':', linewidth=1) ax3.set_xlabel('Position x') ax3.set_ylabel('Probability density') ax3.set_title(f'古典極限（n={n_high}）') ax3.legend() ax3.grid(True, alpha=0.3) # ゼロ点エネルギー ax4 = axes[1, 1] n_range = np.arange(0, 20) E_quantum = [energy_harmonic(n, omega, hbar) for n in n_range] E_classical = n_range * hbar * omega # 古典的エネルギー（基底状態 = 0） ax4.plot(n_range, E_quantum, 'bo-', linewidth=2, markersize=6, label='Quantum') ax4.plot(n_range, E_classical, 'r--', linewidth=2, label='Classical (E=nℏω)') ax4.fill_between(n_range, E_classical, E_quantum, alpha=0.3, color='yellow', label='Zero-point energy') ax4.set_xlabel('Quantum number n') ax4.set_ylabel('Energy E_n') ax4.set_title('ゼロ点エネルギー（E_0 = ℏω/2）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_harmonic_oscillator.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 量子調和振動子 ===\n") print(f"角振動数 ω = {omega}") print(f"質量 m = {m}") print(f"ゼロ点エネルギー E_0 = {energy_harmonic(0, omega, hbar):.4f}\n") print("エネルギー準位:") for n in range(6): E_n = energy_harmonic(n, omega, hbar) print(f" n = {n}: E_{n} = {E_n:.4f} = {n + 0.5:.1f}ℏω") print(f"\n古典的転回点:") for n in [0, 1, 5, 10]: x_turn = classical_turning_point(n, omega, m) print(f" n = {n}: x_turn = ±{x_turn:.4f}") 

## 💻 例題1.3: 演算子と期待値

### 量子力学の演算子

物理量は演算子で表現されます：

  * 位置演算子: \\(\hat{x} = x\\) （位置で掛ける）
  * 運動量演算子: \\(\hat{p} = -i\hbar \frac{\partial}{\partial x}\\)
  * エネルギー演算子: \\(\hat{H} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\\)

**期待値** :

\\[ \langle A \rangle = \int \psi^* \hat{A} \psi \, dx \\]

**Ehrenfestの定理** :

\\[ \frac{d\langle x \rangle}{dt} = \frac{\langle p \rangle}{m}, \quad \frac{d\langle p \rangle}{dt} = -\left\langle \frac{\partial V}{\partial x} \right\rangle \\]

Python実装: 演算子と期待値

import numpy as np import matplotlib.pyplot as plt hbar = 1.0 m = 1.0 omega = 1.0 def position_expectation(n, omega, m, hbar): """位置の期待値（対称性によりゼロ）""" return 0.0 # 調和振動子の場合、対称性により ⟨x⟩ = 0 def momentum_expectation(n, omega, m, hbar): """運動量の期待値（対称性によりゼロ）""" return 0.0 def position_uncertainty(n, omega, m, hbar): """位置の不確定性 Δx""" # ⟨x²⟩ = (n + 1/2) ℏ/(mω) x_squared = (n + 0.5) * hbar / (m * omega) return np.sqrt(x_squared) def momentum_uncertainty(n, omega, m, hbar): """運動量の不確定性 Δp""" # ⟨p²⟩ = (n + 1/2) mωℏ p_squared = (n + 0.5) * m * omega * hbar return np.sqrt(p_squared) # 不確定性関係の検証 n_range = np.arange(0, 20) Delta_x = np.array([position_uncertainty(n, omega, m, hbar) for n in n_range]) Delta_p = np.array([momentum_uncertainty(n, omega, m, hbar) for n in n_range]) product = Delta_x * Delta_p heisenberg_limit = hbar / 2 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 位置の不確定性 ax1 = axes[0, 0] ax1.plot(n_range, Delta_x, 'bo-', linewidth=2, markersize=6) ax1.set_xlabel('Quantum number n') ax1.set_ylabel('Δx') ax1.set_title('位置の不確定性') ax1.grid(True, alpha=0.3) # 運動量の不確定性 ax2 = axes[0, 1] ax2.plot(n_range, Delta_p, 'ro-', linewidth=2, markersize=6) ax2.set_xlabel('Quantum number n') ax2.set_ylabel('Δp') ax2.set_title('運動量の不確定性') ax2.grid(True, alpha=0.3) # 不確定性関係 ax3 = axes[1, 0] ax3.plot(n_range, product, 'go-', linewidth=2, markersize=6, label='Δx·Δp') ax3.axhline(heisenberg_limit, color='r', linestyle='--', linewidth=2, label=f'Heisenberg limit (ℏ/2 = {heisenberg_limit})') ax3.set_xlabel('Quantum number n') ax3.set_ylabel('Δx · Δp') ax3.set_title('Heisenberg不確定性関係') ax3.legend() ax3.grid(True, alpha=0.3) # 数値微分による運動量演算子の作用 ax4 = axes[1, 1] x = np.linspace(-5, 5, 500) dx = x[1] - x[0] n = 2 psi = harmonic_wavefunction(x, n, m, omega, hbar) # 運動量演算子 p̂ψ = -iℏ dψ/dx dpsi_dx = np.gradient(psi, dx) p_psi = -1j * hbar * dpsi_dx ax4.plot(x, np.real(psi), 'b-', linewidth=2, label='Re(ψ)') ax4.plot(x, np.real(p_psi), 'r-', linewidth=2, label='Re(p̂ψ)') ax4.plot(x, np.imag(p_psi), 'g--', linewidth=2, label='Im(p̂ψ)') ax4.set_xlabel('Position x') ax4.set_ylabel('Amplitude') ax4.set_title(f'運動量演算子の作用（n={n}）') ax4.legend() ax4.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('qchem_operators_expectation.png', dpi=300, bbox_inches='tight') plt.show() # 数値結果 print("\n=== 演算子と期待値 ===\n") print("調和振動子の期待値:\n") for n in [0, 1, 2, 5]: Dx = position_uncertainty(n, omega, m, hbar) Dp = momentum_uncertainty(n, omega, m, hbar) print(f"n = {n}:") print(f" ⟨x⟩ = {position_expectation(n, omega, m, hbar):.4f} (対称性)") print(f" ⟨p⟩ = {momentum_expectation(n, omega, m, hbar):.4f} (対称性)") print(f" Δx = {Dx:.4f}") print(f" Δp = {Dp:.4f}") print(f" Δx·Δp = {Dx * Dp:.4f} (≥ ℏ/2 = {hbar/2:.4f})\n") print("Heisenberg不確定性関係:") print(f" 基底状態（n=0）では Δx·Δp = ℏ/2 （最小不確定性状態）") 

## 💻 例題1.4: 角運動量と球面調和関数

### 角運動量演算子

3次元空間での角運動量演算子：

\\[ \hat{\mathbf{L}} = \mathbf{r} \times \hat{\mathbf{p}} = -i\hbar (\mathbf{r} \times \nabla) \\]

**交換関係** :

\\[ [\hat{L}_x, \hat{L}_y] = i\hbar \hat{L}_z, \quad [\hat{L}_y, \hat{L}_z] = i\hbar \hat{L}_x, \quad [\hat{L}_z, \hat{L}_x] = i\hbar \hat{L}_y \\]

\\([\hat{L}^2, \hat{L}_z] = 0\\) なので、\\(\hat{L}^2\\) と \\(\hat{L}_z\\) は同時固有状態を持ちます。

**固有値** :

\\[ \hat{L}^2 Y_l^m(\theta, \phi) = \hbar^2 l(l+1) Y_l^m(\theta, \phi) \\]

\\[ \hat{L}_z Y_l^m(\theta, \phi) = \hbar m Y_l^m(\theta, \phi) \\]

ここで \\(Y_l^m(\theta, \phi)\\) は**球面調和関数** です。

Python実装: 球面調和関数

import numpy as np import matplotlib.pyplot as plt from scipy.special import sph_harm from mpl_toolkits.mplot3d import Axes3D def plot_spherical_harmonic(l, m, ax, title): """球面調和関数の可視化""" # 球面座標 theta = np.linspace(0, np.pi, 100) phi = np.linspace(0, 2*np.pi, 100) Theta, Phi = np.meshgrid(theta, phi) # 球面調和関数（scipy convention: Y_l^m(phi, theta)） Y_lm = sph_harm(m, l, Phi, Theta) # 絶対値（確率密度の角度依存性） R = np.abs(Y_lm) # デカルト座標 X = R * np.sin(Theta) * np.cos(Phi) Y = R * np.sin(Theta) * np.sin(Phi) Z = R * np.cos(Theta) # プロット surface = ax.plot_surface(X, Y, Z, cmap='viridis', facecolors=plt.cm.viridis(R/R.max()), alpha=0.9, shade=True) ax.set_xlabel('X') ax.set_ylabel('Y') ax.set_zlabel('Z') ax.set_title(title) ax.set_box_aspect([1,1,1]) # 異なる (l, m) の球面調和関数 fig = plt.figure(figsize=(16, 12)) configs = [ (0, 0, 's orbital (l=0, m=0)'), (1, 0, 'p_z orbital (l=1, m=0)'), (1, 1, 'p_x orbital (l=1, m=1)'), (2, 0, 'd_{z²} orbital (l=2, m=0)'), (2, 1, 'd_{xz} orbital (l=2, m=1)'), (2, 2, 'd_{xy} orbital (l=2, m=2)'), (3, 0, 'f_{z³} orbital (l=3, m=0)'), (3, 2, 'f orbital (l=3, m=2)'), ] for idx, (l, m, title) in enumerate(configs): ax = fig.add_subplot(2, 4, idx+1, projection='3d') plot_spherical_harmonic(l, m, ax, title) plt.tight_layout() plt.savefig('qchem_spherical_harmonics.png', dpi=300, bbox_inches='tight') plt.show() # 角運動量の固有値 print("\n=== 角運動量と球面調和関数 ===\n") print("角運動量の固有値:\n") hbar = 1.0 for l in range(4): L_squared_eigenvalue = hbar**2 * l * (l + 1) print(f"l = {l}:") print(f" L² の固有値 = {L_squared_eigenvalue:.4f} = ℏ²·{l}·{l+1}") print(f" 許される m の値: {-l} から {l} ({2*l+1}個)") for m in range(-l, l+1): Lz_eigenvalue = hbar * m print(f" m = {m:2d}: L_z = {Lz_eigenvalue:+.4f} = {m:+d}ℏ") print() print("球面調和関数の直交性:") print(" ∫ Y_l^m* Y_l'^m' dΩ = δ_{ll'} δ_{mm'}") 

## 📚 まとめ

  * **波動関数** は粒子の量子状態を記述し、確率振幅を与える
  * **Schrödinger方程式** は量子力学の基本方程式で、波動関数の時間発展を支配する
  * **固有値問題** として定常状態（エネルギー固有状態）を求める
  * **箱の中の粒子** と**調和振動子** は厳密解が得られる重要なモデル
  * **量子化** により離散的なエネルギー準位が出現する
  * **ゼロ点エネルギー** は量子効果の帰結で、基底状態でも運動している
  * **演算子** は物理量を表現し、期待値は測定の平均値を与える
  * **不確定性関係** は量子力学の本質的制約で、位置と運動量は同時に精確に決まらない
  * **角運動量** は量子化され、球面調和関数で記述される
  * 球面調和関数は原子軌道（s, p, d, f軌道）の角度依存性を与える

### 💡 演習問題

  1. **粒子の規格化** : 箱の中の粒子（\\(n=1\\)）の波動関数が規格化されていることを数値的に確認せよ。
  2. **トンネル効果** : 有限ポテンシャル障壁を通り抜ける透過確率を計算し、古典力学との違いを議論せよ。
  3. **調和振動子の演算子法** : 生成・消滅演算子 \\(\hat{a}^\dagger, \hat{a}\\) を用いて固有関数を構築せよ。
  4. **Virial定理** : 調和振動子で \\(2\langle T \rangle = \langle V \rangle\\) が成り立つことを確認せよ。
  5. **球面調和関数の直交性** : 異なる \\((l, m)\\) の球面調和関数が直交することを数値積分で確認せよ。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
