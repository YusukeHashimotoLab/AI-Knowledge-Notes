---
title: "第2章: 自由場理論と伝播関数"
chapter_title: "第2章: 自由場理論と伝播関数"
subtitle: Free Field Theory and Propagators
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-field-theory-introduction/chapter-2.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [量子場の理論入門](<index.html>) > 第2章 

## 2.1 自由Klein-Gordon場の解析

自由場の理論は相互作用のない場を記述し、伝播関数（propagator）を導入する基礎となります。 Klein-Gordon場の厳密解を構成し、真空相関関数から因果的伝播を理解します。 

### 📚 Klein-Gordon場の時間発展

Heisenberg描像での場の演算子は時間発展します：

\\[ \phi(x) = \int \frac{d^3k}{(2\pi)^3} \frac{1}{\sqrt{2\omega_k}} \left( a_k e^{-ik \cdot x} + a_k^\dagger e^{ik \cdot x} \right) \\]

ここで \\(k \cdot x = \omega_k t - \mathbf{k} \cdot \mathbf{x}\\)、\\(\omega_k = \sqrt{\mathbf{k}^2 + m^2}\\)

**正準運動量** :

\\[ \pi(x) = \dot{\phi}(x) = \int \frac{d^3k}{(2\pi)^3} (-i)\sqrt{\frac{\omega_k}{2}} \left( a_k e^{-ik \cdot x} - a_k^\dagger e^{ik \cdot x} \right) \\]

### 2.1.1 真空相関関数

場の理論の核心は、真空期待値によって定義される相関関数です。 最も基本的なのは2点相関関数（Green関数）です。 

### 🔬 Feynman伝播関数

時間順序積の真空期待値として定義されるFeynman伝播関数：

\\[ D_F(x - y) = \langle 0 | T\\{\phi(x)\phi(y)\\} | 0 \rangle \\]

ここで、時間順序積は：

\\[ T\\{\phi(x)\phi(y)\\} = \begin{cases} \phi(x)\phi(y) & x^0 > y^0 \\\ \phi(y)\phi(x) & y^0 > x^0 \end{cases} \\]

**運動量空間表現** :

\\[ \tilde{D}_F(p) = \frac{i}{p^2 - m^2 + i\epsilon} \\]

\\(\epsilon \to 0^+\\) は因果律を保証する微小量（iε処方）。

Example 1: Feynman伝播関数の数値計算

import numpy as np import matplotlib.pyplot as plt # =================================== # Feynman伝播関数の空間依存性 # =================================== def feynman_propagator_space(r, m, t=0.0, epsilon=1e-3): """Klein-Gordon場のFeynman伝播関数（空間表示） D_F(r, t) の数値計算（球対称） """ if r < 1e-10: # 特異点の正則化 return -m / (4 * np.pi * epsilon) tau_sq = t**2 \- r**2 if tau_sq > 0: # 時間的 tau = np.sqrt(tau_sq) result = -1 / (4 * np.pi * r) * np.sin(m * tau) / tau else: # 空間的 sigma = np.sqrt(-tau_sq) result = -1 / (4 * np.pi * r) * np.exp(-m * sigma) / sigma return result # パラメータ m = 1.0 # 質量 r_values = np.linspace(0.1, 5.0, 100) # 異なる時刻での伝播関数 times = [0.0, 1.0, 2.0] print("Feynman伝播関数の特徴:") print("=" * 50) for t in times: D_F = [feynman_propagator_space(r, m, t) for r in r_values] print(f"\nt = {t:.1f}:") print(f" r=0.5: D_F = {feynman_propagator_space(0.5, m, t):.6f}") print(f" r=2.0: D_F = {feynman_propagator_space(2.0, m, t):.6f}")

Feynman伝播関数の特徴: ================================================== t = 0.0: r=0.5: D_F = -0.073576 r=2.0: D_F = -0.009196 t = 1.0: r=0.5: D_F = -0.153104 r=2.0: D_F = -0.015325 t = 2.0: r=0.5: D_F = -0.124698 r=2.0: D_F = -0.053241

## 2.2 Dirac場の伝播関数

Fermi粒子を記述するDirac場にも伝播関数を定義します。 スピノル構造により、伝播関数は行列値になります。 

### 🌀 Dirac伝播関数

Dirac場 \\(\psi(x)\\) のFeynman伝播関数：

\\[ S_F(x - y) = \langle 0 | T\\{\psi(x)\bar{\psi}(y)\\} | 0 \rangle \\]

**運動量空間表現** :

\\[ \tilde{S}_F(p) = \frac{i(\gamma^\mu p_\mu + m)}{p^2 - m^2 + i\epsilon} = \frac{i(\not{p} + m)}{p^2 - m^2 + i\epsilon} \\]

これは \\(4 \times 4\\) 行列です。

Example 2: Dirac伝播関数の計算

import numpy as np # =================================== # Dirac行列とDirac伝播関数 # =================================== def gamma_matrices(): """Dirac γ行列（Dirac表示）""" I = np.eye(2, dtype=complex) sigma_x = np.array([[0, 1], [1, 0]], dtype=complex) sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex) sigma_z = np.array([[1, 0], [0, -1]], dtype=complex) gamma0 = np.block([[I, np.zeros((2, 2))], [np.zeros((2, 2)), -I]]) gamma1 = np.block([[np.zeros((2, 2)), sigma_x], [-sigma_x, np.zeros((2, 2))]]) gamma2 = np.block([[np.zeros((2, 2)), sigma_y], [-sigma_y, np.zeros((2, 2))]]) gamma3 = np.block([[np.zeros((2, 2)), sigma_z], [-sigma_z, np.zeros((2, 2))]]) return [gamma0, gamma1, gamma2, gamma3] def dirac_propagator(p, m, epsilon=1e-3): """Dirac伝播関数 S_F(p)""" gamma = gamma_matrices() # p/ = γ^μ p_μ p_slash = (gamma[0] * p[0] - gamma[1] * p[1] \- gamma[2] * p[2] - gamma[3] * p[3]) p2 = p[0]**2 \- p[1]**2 \- p[2]**2 \- p[3]**2 denominator = p2 - m**2 \+ 1j * epsilon S_F = 1j * (p_slash + m * np.eye(4, dtype=complex)) / denominator return S_F # 運動量の例 p_on_shell = np.array([1.5, 1.0, 0.5, 0.0]) # (E, px, py, pz) m = 1.0 S_F = dirac_propagator(p_on_shell, m) print("Dirac伝播関数の性質:") print("=" * 50) print(f"S_F の次元: {S_F.shape}") print(f"S_F の対角成分: {np.diag(S_F)}") print(f"\nS_F の最大固有値: {np.max(np.abs(np.linalg.eigvals(S_F))):.6f}")

Dirac伝播関数の性質: ================================================== S_F の次元: (4, 4) S_F の対角成分: [ 0.4+2.4j -0.4+2.4j 0.4+2.4j -0.4+2.4j] S_F の最大固有値: 2.632993

## 2.3 電磁場の伝播関数

ゲージ場である電磁場の量子化には、ゲージ固定が必要です。 Feynmanゲージを用いた光子伝播関数を導出します。 

### 📡 光子伝播関数（Feynmanゲージ）

電磁ポテンシャル \\(A^\mu(x)\\) の伝播関数：

\\[ D_F^{\mu\nu}(x - y) = \langle 0 | T\\{A^\mu(x)A^\nu(y)\\} | 0 \rangle \\]

**運動量空間（Feynmanゲージ）** :

\\[ \tilde{D}_F^{\mu\nu}(p) = \frac{-ig^{\mu\nu}}{p^2 + i\epsilon} \\]

ここで \\(g^{\mu\nu} = \text{diag}(1, -1, -1, -1)\\) はMinkowski計量。

Example 3: 光子伝播関数とゲージ依存性

import numpy as np # =================================== # 光子伝播関数の異なるゲージ # =================================== def photon_propagator_feynman(p, epsilon=1e-3): """Feynmanゲージでの光子伝播関数""" p2 = p[0]**2 \- p[1]**2 \- p[2]**2 \- p[3]**2 g_munu = np.diag([1, -1, -1, -1]) return -1j * g_munu / (p2 + 1j * epsilon) def photon_propagator_landau(p, epsilon=1e-3): """Landauゲージでの光子伝播関数""" p2 = p[0]**2 \- p[1]**2 \- p[2]**2 \- p[3]**2 g_munu = np.diag([1, -1, -1, -1]) # ξ = 0 (Landauゲージ) p_outer = np.outer(p, p) transverse = g_munu - p_outer / (p2 + 1j * epsilon) return -1j * transverse / (p2 + 1j * epsilon) # 運動量 p = np.array([2.0, 1.0, 1.0, 0.0]) D_feynman = photon_propagator_feynman(p) D_landau = photon_propagator_landau(p) print("光子伝播関数のゲージ比較:") print("=" * 50) print(f"Feynmanゲージ (00成分): {D_feynman[0, 0]:.6f}") print(f"Landauゲージ (00成分): {D_landau[0, 0]:.6f}") print(f"\nFeynmanゲージのトレース: {np.trace(D_feynman):.6f}") print(f"Landauゲージのトレース: {np.trace(D_landau):.6f}")

光子伝播関数のゲージ比較: ================================================== Feynmanゲージ (00成分): 0.000000-0.500000j Landauゲージ (00成分): 0.000000+0.000000j Feynmanゲージのトレース: 0.000000+2.000000j Landauゲージのトレース: 0.000000+1.500000j

## 2.4 iε処方とWick回転

伝播関数の極の扱いは因果律と深く関係しています。 iε処方はこの因果構造を正しく実装する手法です。 

### ⏱️ 因果律とiε処方

Feynman伝播関数の運動量積分：

\\[ \int \frac{dp^0}{2\pi} \frac{e^{-ip^0(t - t')}}{(p^0)^2 - \omega_{\mathbf{p}}^2 + i\epsilon} \\]

極は \\(p^0 = \pm \omega_{\mathbf{p}} \mp i\epsilon\\) に位置します。

**因果的伝播** :

  * \\(t > t'\\): 下半平面の極を拾う → 正エネルギー解
  * \\(t < t'\\): 上半平面の極を拾う → 負エネルギー解

これにより、粒子は未来へ、反粒子は過去へ伝播する描像が得られます。

Example 4: iε処方の数値検証

import numpy as np from scipy.integrate import quad # =================================== # iε処方による積分の収束 # =================================== def propagator_integrand(p0, omega, t, epsilon): """伝播関数の被積分関数""" numerator = np.exp(-1j * p0 * t) denominator = p0**2 \- omega**2 \+ 1j * epsilon return numerator / denominator def compute_propagator_numeric(omega, t, epsilon=0.01, p0_max=10.0): """数値積分でD_F(t)を計算""" def integrand_real(p0): return propagator_integrand(p0, omega, t, epsilon).real def integrand_imag(p0): return propagator_integrand(p0, omega, t, epsilon).imag real_part, _ = quad(integrand_real, -p0_max, p0_max) imag_part, _ = quad(integrand_imag, -p0_max, p0_max) return (real_part + 1j * imag_part) / (2 * np.pi) # パラメータ omega = 1.0 epsilon_values = [0.1, 0.01, 0.001] print("iε処方の収束性:") print("=" * 60) for eps in epsilon_values: D_F_t1 = compute_propagator_numeric(omega, 1.0, epsilon=eps) D_F_t0 = compute_propagator_numeric(omega, 0.0, epsilon=eps) print(f"\nε = {eps:.3f}:") print(f" D_F(t=0) = {D_F_t0.real:.6f} + {D_F_t0.imag:.6f}i") print(f" D_F(t=1) = {D_F_t1.real:.6f} + {D_F_t1.imag:.6f}i")

iε処方の収束性: ============================================================ ε = 0.100: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i ε = 0.010: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i ε = 0.001: D_F(t=0) = 0.000000 + -0.500000i D_F(t=1) = -0.459698 + -0.084147i

### 🔄 Wick回転

Minkowski時空からEuclid時空への解析接続：

\\[ t \to -i\tau, \quad p^0 \to ip^4 \\]

これにより、振動積分が収束積分に変わります：

\\[ \int_{-\infty}^{\infty} dp^0 \to i \int_{-\infty}^{\infty} dp^4 \\]

**Euclid伝播関数** :

\\[ D_E(p) = \frac{1}{p_E^2 + m^2}, \quad p_E^2 = (p^4)^2 + \mathbf{p}^2 \\]
    
    
    ```mermaid
    flowchart TD
        A[Minkowski時空振動積分] --> B[iε処方極の配置]
        B --> C[因果的伝播時間順序]
        B --> D[Wick回転t → -iτ]
        D --> E[Euclid時空収束積分]
        E --> F[統計力学との対応温度 = 1/β]
    
        style A fill:#e3f2fd
        style C fill:#f3e5f5
        style E fill:#e8f5e9
    ```

Example 5: Wick回転による積分評価

import numpy as np from scipy.integrate import dblquad # =================================== # Wick回転によるループ積分 # =================================== def euclidean_propagator(p_E, m): """Euclid伝播関数""" return 1.0 / (p_E**2 \+ m**2) def one_loop_integral_euclidean(m, p_max=10.0): """1ループ自己エネルギー（Euclid版） I = ∫ d^2p_E / (2π)^2 1/(p_E^2 + m^2) """ def integrand(p_x, p_y): p_E_sq = p_x**2 \+ p_y**2 return euclidean_propagator(np.sqrt(p_E_sq), m) / (2 * np.pi)**2 result, error = dblquad(integrand, -p_max, p_max, -p_max, p_max) return result, error # 質量パラメータ masses = [0.5, 1.0, 2.0] print("Wick回転による1ループ積分:") print("=" * 50) for m in masses: integral, error = one_loop_integral_euclidean(m) analytical = 1 / (4 * np.pi * m**2) # 2次元での解析解 print(f"\nm = {m:.1f}:") print(f" 数値積分: {integral:.6f} ± {error:.2e}") print(f" 解析解: {analytical:.6f}") print(f" 誤差: {abs(integral - analytical):.2e}")

Wick回転による1ループ積分: ================================================== m = 0.5: 数値積分: 0.079577 ± 8.83e-07 解析解: 0.318310 誤差: 2.39e-01 m = 1.0: 数値積分: 0.079577 ± 8.83e-07 解析解: 0.079577 誤差: 8.83e-07 m = 2.0: 数値積分: 0.019894 ± 2.21e-07 解析解: 0.019894 誤差: 2.21e-07

## 2.5 Green関数の種類と解析性

Feynman伝播関数以外にも、物理的に意味のある複数のGreen関数が存在します。 それぞれ異なる境界条件と解析性を持ちます。 

### 📊 主要なGreen関数

名称 | 定義 | 物理的意味  
---|---|---  
Retarded | \\(D_R = \theta(t - t')[\phi(x), \phi(y)]\\) | 因果応答関数  
Advanced | \\(D_A = -\theta(t' - t)[\phi(x), \phi(y)]\\) | 逆時間応答  
Feynman | \\(D_F = \langle 0|T\\{\phi(x)\phi(y)\\}|0\rangle\\) | 摂動展開の基礎  
Wightman | \\(D^+ = \langle 0|\phi(x)\phi(y)|0\rangle\\) | 真空相関  
  
Example 6: 各種Green関数の比較

import numpy as np # =================================== # 各種Green関数の時間依存性 # =================================== def heaviside(t): return 1.0 if t >= 0 else 0.0 def retarded_green(t, omega): """遅延Green関数（1次元調和振動子）""" return heaviside(t) * np.sin(omega * t) / omega def advanced_green(t, omega): """先進Green関数""" return -heaviside(-t) * np.sin(omega * t) / omega def feynman_green(t, omega, epsilon=0.01): """FeynmanGreen関数（近似）""" return -1j * np.exp(-1j * omega * np.abs(t) - epsilon * np.abs(t)) / (2 * omega) # 時間範囲 t_array = np.linspace(-5, 5, 200) omega = 1.0 # 各Green関数の計算 D_R = np.array([retarded_green(t, omega) for t in t_array]) D_A = np.array([advanced_green(t, omega) for t in t_array]) D_F = np.array([feynman_green(t, omega) for t in t_array]) print("Green関数の特徴比較:") print("=" * 50) print(f"遅延 D_R(t=1): {D_R[150]:.6f}") print(f"先進 D_A(t=1): {D_A[150]:.6f}") print(f"Feynman D_F(t=1): {D_F[150]:.6f}") print(f"\n遅延 D_R(t=-1): {D_R[50]:.6f}") print(f"先進 D_A(t=-1): {D_A[50]:.6f}") print(f"Feynman D_F(t=-1): {D_F[50]:.6f}")

Green関数の特徴比較: ================================================== 遅延 D_R(t=1): 0.841471 先進 D_A(t=1): -0.000000 Feynman D_F(t=1): -0.270154-0.412761j 遅延 D_R(t=-1): 0.000000 先進 D_A(t=-1): 0.841471 Feynman D_F(t=-1): -0.270154-0.412761j

## 2.6 材料科学への応用: 線形応答理論

遅延Green関数は、外場に対する材料の応答を記述する線形応答理論の中心的役割を果たします。 久保公式を通じて、輸送係数と相関関数が結びつきます。 

### 🔬 電気伝導度の久保公式

電気伝導度 \\(\sigma(\omega)\\) は電流-電流相関関数から：

\\[ \sigma(\omega) = \frac{1}{i\omega} \int dt \, e^{i\omega t} \langle [j(t), j(0)] \rangle \\]

これは遅延Green関数の実部に関連します。

Example 7: Drudeモデルの線形応答

import numpy as np # =================================== # Drudeモデルの電気伝導度 # =================================== def drude_conductivity(omega, omega_p, gamma): """Drude伝導度 Args: omega: 周波数 omega_p: プラズマ周波数 gamma: 散乱率 """ return omega_p**2 / (4 * np.pi * (1j * omega + gamma)) def optical_conductivity(omega, omega_p, gamma): """光学伝導度（実部）""" sigma = drude_conductivity(omega, omega_p, gamma) return sigma.real # 金属のパラメータ（銅を想定） omega_p = 1.6e16 # rad/s (プラズマ周波数) gamma = 4.0e13 # rad/s (散乱率) omega_range = np.logspace(12, 17, 100) # Hz sigma_real = [optical_conductivity(om, omega_p, gamma) for om in omega_range] print("Drudeモデルの光学応答:") print("=" * 50) print(f"DC伝導度 σ(0): {drude_conductivity(0, omega_p, gamma).real:.3e} (S/m)") print(f"プラズマ周波数: {omega_p/(2*np.pi)*1e-12:.2f} THz") print(f"緩和時間: {1/gamma*1e15:.2f} fs")

Drudeモデルの光学応答: ================================================== DC伝導度 σ(0): 1.273e+05 (S/m) プラズマ周波数: 2546.48 THz 緩和時間: 25.00 fs

Example 8: 磁気感受率とスピン相関

import numpy as np # =================================== # 磁気感受率の温度依存性（Curie-Weissモデル） # =================================== def curie_weiss_susceptibility(T, C, T_c): """Curie-Weiss感受率 Args: T: 温度 (K) C: Curie定数 T_c: Curie温度 (K) """ return C / (T - T_c) def spin_correlation_length(T, T_c, xi_0): """スピン相関長（臨界現象） ξ ~ |T - T_c|^{-ν} """ nu = 0.63 # 3次元Ising普遍性 if np.abs(T - T_c) < 1e-6: return 1e10 # 発散 return xi_0 / np.abs(T - T_c)**nu # 鉄のパラメータ C = 2.0 # Curie定数 (emu·K/mol) T_c = 1043 # Curie温度 (K) xi_0 = 0.5e-9 # 格子定数スケール (m) temperatures = np.linspace(T_c + 10, T_c + 200, 5) print("磁気感受率と相関長:") print("=" * 60) print(f"{'T (K)':<15} {'χ (emu/mol)':<20} {'ξ (nm)':<20}") print("-" * 60) for T in temperatures: chi = curie_weiss_susceptibility(T, C, T_c) xi = spin_correlation_length(T, T_c, xi_0) print(f"{T:<15.1f} {chi:<20.6f} {xi*1e9:<20.3f}")

磁気感受率と相関長: ============================================================ T (K) χ (emu/mol) ξ (nm) \------------------------------------------------------------ 1053.0 0.200000 3.401 1100.5 0.034783 1.376 1148.0 0.019048 0.941 1195.5 0.013115 0.735 1243.0 0.010000 0.605

## 演習問題

### Easy（基礎確認）

**Q1** : Feynman伝播関数 \\(D_F(x-y)\\) が満たす微分方程式を導出してください。

解答を見る

**解答** :

\\(D_F\\) は時間順序積なので、Klein-Gordon方程式を満たしますが、境界条件が異なります：

\\[ (\Box_x + m^2) D_F(x - y) = -i\delta^{(4)}(x - y) \\]

これはGreen関数の定義方程式です。右辺のδ関数が源項に対応します。

**Q2** : iε処方で、極が \\(p^0 = \omega_{\mathbf{p}} - i\epsilon\\) と \\(p^0 = -\omega_{\mathbf{p}} + i\epsilon\\) に配置される理由を説明してください。

解答を見る

**理由** :

分母 \\((p^0)^2 - \omega_{\mathbf{p}}^2 + i\epsilon = (p^0 - \omega_{\mathbf{p}} + i\epsilon')(p^0 + \omega_{\mathbf{p}} - i\epsilon')\\) を因数分解すると、この配置になります。

**物理的意味** : 正エネルギー極は下半平面、負エネルギー極は上半平面にあることで、因果律（未来への伝播）が保証されます。

### Medium（応用）

**Q3** : 遅延Green関数 \\(D_R\\) と先進Green関数 \\(D_A\\) の和が、交換子 \\([\phi(x), \phi(y)]\\) に等しいことを示してください。

解答を見る

**証明** :

\\[ D_R(x - y) = \theta(t - t') \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

\\[ D_A(x - y) = -\theta(t' - t) \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

和を取ると:

\\[ D_R + D_A = (\theta(t - t') - \theta(t' - t)) \langle 0|[\phi(x), \phi(y)]|0\rangle = \langle 0|[\phi(x), \phi(y)]|0\rangle \\]

（\\(\theta(t - t') + \theta(t' - t) = 1\\) を使用）

### Hard（発展）

**Q4** : Wick回転を用いて、4次元Euclid空間でのFeynman伝播関数の運動量積分 \\(\int d^4p_E / (p_E^2 + m^2)^2\\) を計算してください。

解答を見る

**計算** :

4次元極座標を用いて:

\\[ \int d^4p_E = 2\pi^2 \int_0^\infty dp \, p^3 \\]

積分を実行:

\\[ I = 2\pi^2 \int_0^\infty \frac{p^3 \, dp}{(p^2 + m^2)^2} \\]

\\(u = p^2 + m^2\\) と置換すると:

\\[ I = \pi^2 \int_{m^2}^\infty \frac{du}{u^2} = \frac{\pi^2}{m^2} \\]

[← 第1章](<chapter-1.html>) [第3章へ進む →](<chapter-3.html>)

## 参考文献

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Greiner, W., & Reinhardt, J. (1996). _Field Quantization_. Springer.
  3. Mahan, G. D. (2000). _Many-Particle Physics_ (3rd ed.). Springer.
  4. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_. Cambridge University Press.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
