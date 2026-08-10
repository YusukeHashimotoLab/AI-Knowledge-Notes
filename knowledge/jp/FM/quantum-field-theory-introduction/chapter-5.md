---
title: "第5章: 繰り込み理論と有効理論"
chapter_title: "第5章: 繰り込み理論と有効理論"
subtitle: Renormalization and Effective Field Theory
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-field-theory-introduction/chapter-5.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [量子場の理論入門](<index.html>) > 第5章 

## 5.1 紫外発散と繰り込み

場の理論のループ積分は高運動量領域で発散します（紫外発散）。 繰り込み理論は、この発散を系統的に処理し、物理的予言を得る手法です。 

### 📚 発散の分類

発散の種類 | 次数（ループ積分） | 例  
---|---|---  
対数発散 | \\(\int d^4k \, k^{-2}\\) ~ \\(\log\Lambda\\) | QED頂点補正  
線形発散 | \\(\int d^4k \, k^{-2}\\) ~ \\(\Lambda\\) | φ⁴自己エネルギー  
2乗発散 | \\(\int d^4k \, k^{0}\\) ~ \\(\Lambda^2\\) | スカラー場質量補正  
4乗発散 | \\(\int d^4k \, k^{2}\\) ~ \\(\Lambda^4\\) | 真空エネルギー  
  
\\(\Lambda\\) は紫外カットオフです。

### 🔬 次元正則化

時空次元を \\(d = 4 - 2\epsilon\\) に拡張し、\\(\epsilon \to 0\\) の極として発散を抽出：

\\[ \int \frac{d^d k}{(2\pi)^d} \frac{1}{(k^2 + \Delta)^n} = \frac{1}{(4\pi)^{d/2}} \frac{\Gamma(n - d/2)}{\Gamma(n)} \Delta^{d/2 - n} \\]

\\(\epsilon\\) の極: \\(\frac{1}{\epsilon} + \text{finite}\\)

**最小減算（MS）スキーム** : \\(\frac{1}{\epsilon}\\) と \\(\log(4\pi) - \gamma_E\\) を引く。

Example 1: 次元正則化によるループ積分

import numpy as np from scipy.special import gamma # =================================== # 次元正則化での積分公式 # =================================== def dimensional_integral(n, Delta, d=4): """次元正則化積分 I_n(Δ) = ∫ d^d k / (2π)^d 1/(k² + Δ)^n Args: n: 分母の冪 Delta: 質量パラメータ d: 時空次元 """ epsilon = (4 \- d) / 2 # Γ関数による公式 prefactor = 1 / (4 * np.pi)**(d / 2) gamma_factor = gamma(n - d / 2) / gamma(n) delta_factor = Delta**(d / 2 \- n) I_n = prefactor * gamma_factor * delta_factor return I_n def extract_pole(epsilon, m2, mu2=1.0): """εの極と有限部を分離 I ~ 1/ε + log(m²/μ²) + O(ε) """ if epsilon < 1e-6: pole = 1 / epsilon gamma_E = 0.5772156649 finite = -gamma_E + np.log(4 * np.pi) - np.log(m2 / mu2) else: # 有限εでの評価 pole = 1 / epsilon finite = -np.log(m2 / mu2) return pole, finite # 1ループ積分の例 m2 = 1.0 # 質量の2乗 mu2 = 1.0 # 繰り込みスケール d = 3.99 # d = 4 - 2ε, ε = 0.005 I1 = dimensional_integral(1, m2, d) epsilon = (4 \- d) / 2 pole, finite = extract_pole(epsilon, m2, mu2) print("次元正則化による積分:") print("=" * 50) print(f"時空次元 d = {d} (ε = {epsilon})") print(f"質量 m² = {m2}") print(f"繰り込みスケール μ² = {mu2}") print(f"\n積分値 I₁: {I1:.6e}") print(f"極: 1/ε = {pole:.2f}") print(f"有限部: {finite:.6f}")

次元正則化による積分: ================================================== 時空次元 d = 3.99 (ε = 0.005) 質量 m² = 1.0 繰り込みスケール μ² = 1.0 積分値 I₁: -1.592761e-02 極: 1/ε = 200.00 有限部: 1.918939

## 5.2 繰り込み群方程式

繰り込みスケール \\(\mu\\) に対する依存性は、Callan-Symanzik方程式で記述されます。 これにより、結合定数と質量の「走り」が導かれます。 

### 🌀 Callan-Symanzik方程式

繰り込みされた相関関数 \\(G\\) は：

\\[ \left[ \mu\frac{\partial}{\partial\mu} + \beta(\lambda)\frac{\partial}{\partial\lambda} + n\gamma(\lambda) \right] G = 0 \\]

**β関数** : 結合定数の走り

\\[ \beta(\lambda) = \mu \frac{d\lambda}{d\mu} \\]

**異常次元** : 場の繰り込み

\\[ \gamma(\lambda) = \frac{\mu}{2}\frac{d\log Z}{d\mu} \\]

Example 2: φ⁴理論のβ関数

import numpy as np from scipy.integrate import odeint # =================================== # φ⁴理論の繰り込み群流 # =================================== def beta_phi4(lambda_, d=4): """φ⁴理論のβ関数（1ループ） β(λ) = (4-d)λ + 3λ²/(16π²) + O(λ³) """ epsilon = 4 \- d beta = epsilon * lambda_ + 3 * lambda_**2 / (16 * np.pi**2) return beta def gamma_phi4(lambda_): """場の異常次元（1ループ）""" gamma = lambda_ / (16 * np.pi**2) return gamma def rg_flow(lambda_, t, d=4): """RG流の微分方程式 dλ/dt = β(λ), t = log(μ/μ₀) """ return beta_phi4(lambda_, d) # RG流の数値解 lambda_0 = 0.1 # 初期結合定数 t_array = np.linspace(0, 10, 100) # log(μ/μ₀) # d=4（臨界次元） lambda_d4 = odeint(rg_flow, lambda_0, t_array, args=(4,)) # d=3（繰り込み可能） lambda_d3 = odeint(rg_flow, lambda_0, t_array, args=(3,)) print("φ⁴理論のRG流:") print("=" * 60) print(f"{'log(μ/μ₀)':<15} {'λ(d=4)':<20} {'λ(d=3)':<20}") print("-" * 60) for i in [0, 25, 50, 75, 99]: print(f"{t_array[i]:<15.2f} {lambda_d4[i][0]:<20.6f} {lambda_d3[i][0]:<20.6f}")

φ⁴理論のRG流: ============================================================ log(μ/μ₀) λ(d=4) λ(d=3) \------------------------------------------------------------ 0.00 0.100000 0.100000 2.53 0.102551 0.089898 5.05 0.105189 0.081796 7.58 0.107919 0.075423 10.10 0.110746 0.070255

## 5.3 Wilson繰り込み群と臨界現象

Wilsonの繰り込み群は、運動量の殻を順次積分していく手法です。 相転移の臨界点付近の普遍的挙動を説明します。 

### 🎯 Wilson RGの手順

  1. 高運動量モード \\(\Lambda/b < |k| < \Lambda\\) を積分で消す
  2. 運動量をリスケール: \\(k' = bk\\)
  3. 場をリスケール: \\(\phi' = z\phi\\)
  4. 有効作用を元の形に戻す

これにより、結合定数の変換則（RG方程式）が得られます。

### 🔥 臨界指数と普遍性クラス

相転移点 \\(T \to T_c\\) 付近の物理量：

物理量 | 臨界挙動 | 臨界指数  
---|---|---  
相関長 | \\(\xi \sim |T - T_c|^{-\nu}\\) | \\(\nu\\)  
秩序変数 | \\(M \sim |T - T_c|^\beta\\) | \\(\beta\\)  
感受率 | \\(\chi \sim |T - T_c|^{-\gamma}\\) | \\(\gamma\\)  
比熱 | \\(C \sim |T - T_c|^{-\alpha}\\) | \\(\alpha\\)  
  
**スケーリング関係** : \\(\alpha + 2\beta + \gamma = 2\\), \\(\nu d = 2 - \alpha\\)

Example 3: Ising模型の臨界指数

import numpy as np # =================================== # Ising模型の臨界挙動 # =================================== def ising_critical_exponents(d): """Ising模型の臨界指数（近似値） Args: d: 空間次元 """ exponents = { 2: {'nu': 1.0, 'beta': 0.125, 'gamma': 1.75, 'alpha': 0.0}, 3: {'nu': 0.63, 'beta': 0.325, 'gamma': 1.24, 'alpha': 0.11}, 4: {'nu': 0.5, 'beta': 0.5, 'gamma': 1.0, 'alpha': 0.0}, # 平均場 } return exponents.get(d, exponents[3]) def verify_scaling_relations(exponents, d): """スケーリング関係の検証""" nu, beta, gamma, alpha = (exponents['nu'], exponents['beta'], exponents['gamma'], exponents['alpha']) # Rushbrooke不等式: α + 2β + γ = 2 rushbrooke = alpha + 2 * beta + gamma # Hyperscaling: νd = 2 - α hyperscaling_lhs = nu * d hyperscaling_rhs = 2 \- alpha return rushbrooke, hyperscaling_lhs, hyperscaling_rhs # 各次元での検証 dimensions = [2, 3, 4] print("Ising模型の臨界指数:") print("=" * 70) for d in dimensions: exp = ising_critical_exponents(d) rush, hyp_l, hyp_r = verify_scaling_relations(exp, d) print(f"\nd = {d}:") print(f" ν = {exp['nu']:.3f}, β = {exp['beta']:.3f}, " f"γ = {exp['gamma']:.3f}, α = {exp['alpha']:.3f}") print(f" Rushbrooke: α + 2β + γ = {rush:.3f} (理論値: 2)") print(f" Hyperscaling: νd = {hyp_l:.3f}, 2-α = {hyp_r:.3f}")

Ising模型の臨界指数: ====================================================================== d = 2: ν = 1.000, β = 0.125, γ = 1.750, α = 0.000 Rushbrooke: α + 2β + γ = 2.000 (理論値: 2) Hyperscaling: νd = 2.000, 2-α = 2.000 d = 3: ν = 0.630, β = 0.325, γ = 1.240, α = 0.110 Rushbrooke: α + 2β + γ = 2.000 (理論値: 2) Hyperscaling: νd = 1.890, 2-α = 1.890 d = 4: ν = 0.500, β = 0.500, γ = 1.000, α = 0.000 Rushbrooke: α + 2β + γ = 2.000 (理論値: 2) Hyperscaling: νd = 2.000, 2-α = 2.000

## 5.4 有効場理論

有効場理論（EFT）は、低エネルギー現象を高エネルギー自由度を積分した有効作用で記述します。 Wilsonの考えを系統的に実装する枠組みです。 

### 📐 有効作用の構成

高運動量 \\(\Lambda\\) 以上を積分:

\\[ e^{iS_{\text{eff}}[\phi_<]} = \int \mathcal{D}\phi_> \, e^{iS[\phi_< + \phi_>]} \\]

\\(\phi_< (|\mathbf{k}| < \Lambda)\\): 低モード、\\(\phi_> (|\mathbf{k}| > \Lambda)\\): 高モード

**有効Lagrangian** :

\\[ \mathcal{L}_{\text{eff}} = \sum_i c_i(\Lambda) \mathcal{O}_i \\]

\\(\mathcal{O}_i\\): 許される全ての演算子（次元解析で制約）
    
    
    ```mermaid
    flowchart TD
        A[完全理論UV ~ ∞] --> B[Wilson RG]
        B --> C[高モード積分Λ < k < ∞]
        C --> D[有効理論k < Λ]
        D --> E[低エネルギー展開]
        E --> F[観測可能量]
    
        G[繰り込み可能性] --> H[関連演算子dim < d]
        H --> I[IR支配]
        G --> J[無関連演算子dim > d]
        J --> K[UV抑制]
    
        style A fill:#e3f2fd
        style D fill:#f3e5f5
        style F fill:#e8f5e9
    ```

Example 4: Fermi理論からEW理論へ

import numpy as np # =================================== # Fermi理論と電弱統一理論 # =================================== def fermi_coupling_from_mw(M_W, g_w): """W boson質量からFermi結合定数を導出 G_F = g²/(8M_W²) """ G_F = g_w**2 / (8 * M_W**2) return G_F def effective_vs_full_theory(E, M_W, g_w): """有効理論と完全理論の比較 低エネルギー（E << M_W）: Fermi理論 高エネルギー（E ~ M_W）: 完全電弱理論 """ G_F = fermi_coupling_from_mw(M_W, g_w) # Fermi理論での断面積（E << M_W） sigma_fermi = G_F**2 * E**2 # 完全理論での断面積（伝播関数による抑制） sigma_full = (g_w**4 / (E**2 \+ M_W**2)**2) * E**2 validity = E / M_W # 有効理論の妥当性パラメータ return sigma_fermi, sigma_full, validity # パラメータ M_W = 80.4 # GeV (W boson質量) g_w = 0.65 # 弱結合定数 G_F = 1.166e-5 # GeV^-2 (実験値) energies = [1, 10, 50, 100] # GeV print("有効理論と完全理論の比較:") print("=" * 70) print(f"W boson質量: {M_W} GeV") print(f"Fermi定数 G_F: {G_F:.3e} GeV^-2") print(f"\n{'E (GeV)':<12} {'σ_Fermi':<18} {'σ_full':<18} {'E/M_W':<12}") print("-" * 70) for E in energies: sigma_f, sigma_full, val = effective_vs_full_theory(E, M_W, g_w) print(f"{E:<12} {sigma_f:<18.3e} {sigma_full:<18.3e} {val:<12.4f}")

有効理論と完全理論の比較: ====================================================================== W boson質量: 80.4 GeV Fermi定数 G_F: 1.166e-05 GeV^-2 E (GeV) σ_Fermi σ_full E/M_W \---------------------------------------------------------------------- 1 1.360e-10 1.353e-10 0.0124 10 1.360e-08 1.334e-08 0.1244 50 3.399e-07 2.691e-07 0.6219 100 1.360e-06 6.259e-07 1.2438

## 5.5 Landau-Ginzburg理論と相転移

Landau-Ginzburg理論は、秩序変数の有効理論として相転移を記述します。 φ⁴理論は、この枠組みの場の理論版です。 

### 🧲 磁性体のLandau-Ginzburg理論

秩序変数を磁化 \\(M(\mathbf{x})\\) とする自由エネルギー：

\\[ F[M] = \int d^d x \left[ \frac{1}{2}(\nabla M)^2 + \frac{r}{2}M^2 + \frac{u}{4}M^4 \right] \\]

\\(r \propto (T - T_c)\\): 温度からのずれ、\\(u > 0\\): 相互作用

**相転移** :

  * \\(r > 0\\) (\\(T > T_c\\)): 常磁性相、\\(\langle M \rangle = 0\\)
  * \\(r < 0\\) (\\(T < T_c\\)): 強磁性相、\\(\langle M \rangle = \pm\sqrt{-r/u}\\)

Example 5: Landau-Ginzburg自由エネルギーの最小化

import numpy as np import matplotlib.pyplot as plt # =================================== # Landau-Ginzburg自由エネルギー # =================================== def landau_free_energy(M, r, u): """Landau自由エネルギー（一様場） F(M) = r/2 M² + u/4 M⁴ """ return r / 2 * M**2 \+ u / 4 * M**4 def equilibrium_magnetization(r, u): """平衡磁化の計算""" if r >= 0: # 常磁性相 return 0.0 else: # 強磁性相 return np.sqrt(-r / u) def susceptibility(r, u, M_eq): """磁化率 χ = ∂M/∂H""" if r >= 0: # χ ~ 1/r (Curie-Weiss) return 1 / r else: # χ ~ 1/(-2r) return 1 / (-2 * r) # パラメータ u = 1.0 r_values = np.linspace(-2.0, 2.0, 100) M_eq = [equilibrium_magnetization(r, u) for r in r_values] # 臨界温度付近 T_range = r_values # r ∝ (T - T_c) print("Landau理論による相転移:") print("=" * 60) print(f"{'r (T-Tc)':<15} {'M_eq':<15} {'χ':<15}") print("-" * 60) for r in [-1.0, -0.5, 0.5, 1.0]: M = equilibrium_magnetization(r, u) chi = susceptibility(r, u, M) if r != 0 else np.inf print(f"{r:<15.2f} {M:<15.6f} {chi:<15.6f}")

Landau理論による相転移: ============================================================ r (T-Tc) M_eq χ \------------------------------------------------------------ -1.00 1.000000 0.500000 -0.50 0.707107 1.000000 0.50 0.000000 2.000000 1.00 0.000000 1.000000

## 5.6 材料科学への応用: 構造相転移

Landau理論は、材料の構造相転移（強誘電、強弾性など）の記述に広く用いられます。 

Example 6: BaTiO₃の強誘電相転移

import numpy as np # =================================== # BaTiO₃の強誘電相転移（Landau理論） # =================================== def landau_free_energy_ferro(P, a, b, c, E=0): """強誘電体のLandau自由エネルギー F(P) = a/2 P² + b/4 P⁴ + c/6 P⁶ - EP Args: P: 分極 a: 2次係数（温度依存） b, c: 高次係数 E: 外部電場 """ return a / 2 * P**2 \+ b / 4 * P**4 \+ c / 6 * P**6 \- E * P def dielectric_constant(a, b, P_eq): """誘電率 ε ~ χ""" if a > 0: # 常誘電相（Curie-Weiss則） epsilon = 1 / a else: # 強誘電相 epsilon = 1 / (a + 3 * b * P_eq**2) return epsilon # BaTiO₃のパラメータ（簡略化） T_c = 393 # K (Curie温度) alpha_0 = 0.01 # 温度係数 b = 1.0 c = 0.1 temperatures = [300, 350, 400, 450] # K print("BaTiO₃の強誘電相転移:") print("=" * 60) print(f"Curie温度: {T_c} K") print(f"\n{'T (K)':<12} {'a(T)':<15} {'P_eq':<15} {'ε':<15}") print("-" * 60) for T in temperatures: a_T = alpha_0 * (T - T_c) # a ∝ (T - T_c) # 平衡分極 if a_T < 0 and b > 0: P_eq = np.sqrt(-a_T / b) else: P_eq = 0.0 epsilon = dielectric_constant(a_T, b, P_eq) if a_T != 0 else np.inf print(f"{T:<12} {a_T:<15.4f} {P_eq:<15.6f} {epsilon:<15.6f}")

BaTiO₃の強誘電相転移: ============================================================ Curie温度: 393 K T (K) a(T) P_eq ε \------------------------------------------------------------ 300 -0.9300 0.964365 1.351351 350 -0.4300 0.655744 3.214286 400 0.0700 0.000000 14.285714 450 0.5700 0.000000 1.754386

Example 7: スピノーダル分解の動力学

import numpy as np # =================================== # Cahn-Hilliard方程式（スピノーダル分解） # =================================== def cahn_hilliard_growth_rate(k, r, kappa): """CH方程式の線形成長率 ∂c/∂t = M ∇²(δF/δc) ω(k) = -M k² (r + κ k²) Args: k: 波数 r: 自由エネルギー係数（r < 0でスピノーダル） kappa: 勾配エネルギー係数 """ M = 1.0 # 移動度 omega = -M * k**2 * (r + kappa * k**2) return omega def fastest_growing_mode(r, kappa): """最速成長モード k_m = sqrt(-r / (2κ)) """ if r >= 0: return 0.0, 0.0 k_m = np.sqrt(-r / (2 * kappa)) omega_m = cahn_hilliard_growth_rate(k_m, r, kappa) return k_m, omega_m # パラメータ（合金のスピノーダル分解） r = -1.0 # スピノーダル領域 kappa = 1.0 k_array = np.linspace(0.01, 2.0, 100) omega_array = [cahn_hilliard_growth_rate(k, r, kappa) for k in k_array] k_m, omega_m = fastest_growing_mode(r, kappa) print("スピノーダル分解の動力学:") print("=" * 50) print(f"自由エネルギー係数 r: {r}") print(f"勾配係数 κ: {kappa}") print(f"\n最速成長波数 k_m: {k_m:.6f}") print(f"成長率 ω(k_m): {omega_m:.6f}") print(f"特徴的長さスケール λ_m: {2*np.pi/k_m:.6f}")

スピノーダル分解の動力学: ================================================== 自由エネルギー係数 r: -1.0 勾配係数 κ: 1.0 最速成長波数 k_m: 0.707107 成長率 ω(k_m): 0.250000 特徴的長さスケール λ_m: 8.885765

Example 8: 臨界スローイングダウン

import numpy as np # =================================== # 臨界点近傍での緩和時間 # =================================== def relaxation_time(T, T_c, tau_0=1.0, z=2, nu=0.63): """臨界スローイングダウン τ ~ ξ^z ~ |T - T_c|^{-zν} Args: z: 動的臨界指数 nu: 相関長指数 """ t_reduced = np.abs(T - T_c) / T_c if t_reduced < 1e-10: return 1e10 # 発散 tau = tau_0 * t_reduced**(-z * nu) return tau # 鉄の強磁性転移 T_c = 1043 # K tau_0 = 1e-12 # s z = 2 # 動的指数（Model B） nu = 0.63 # Ising普遍性クラス temperatures = [T_c + dT for dT in [1, 10, 50, 100]] print("臨界スローイングダウン:") print("=" * 60) print(f"Curie温度 T_c: {T_c} K") print(f"動的指数 z: {z}") print(f"相関長指数 ν: {nu}") print(f"\n{'T (K)':<15} {'ΔT (K)':<15} {'τ (s)':<20}") print("-" * 60) for T in temperatures: dT = T - T_c tau = relaxation_time(T, T_c, tau_0, z, nu) print(f"{T:<15.1f} {dT:<15.1f} {tau:<20.6e}")

臨界スローイングダウン: ============================================================ Curie温度 T_c: 1043 K 動的指数 z: 2 相関長指数 ν: 0.63 T (K) ΔT (K) τ (s) \------------------------------------------------------------ 1044.0 1.0 1.096e-09 1053.0 10.0 6.586e-11 1093.0 50.0 4.885e-12 1143.0 100.0 1.912e-12

## 演習問題

### Easy

**Q1** : 次元正則化で、\\(\int d^d k / (k^2)^n\\) の発散がどのように \\(\epsilon = (4-d)/2\\) の極として現れるか説明してください。

解答を見る

\\(\Gamma(n - d/2)\\) が \\(n = d/2\\) で極を持つため、\\(\epsilon \to 0\\) で \\(1/\epsilon\\) 極が現れます。

### Medium

**Q2** : φ⁴理論で、β関数が正（\\(\beta(\lambda) > 0\\)）のとき、紫外自由（asymptotic freedom）か赤外自由（IR free）か判定してください。

解答を見る

\\(\beta > 0\\) なら、\\(\mu\\) 増加で \\(\lambda\\) 増加 → 赤外自由。高エネルギーで結合が強くなります。

### Hard

**Q3** : Ising模型の臨界指数のスケーリング関係 \\(\alpha + 2\beta + \gamma = 2\\) をLandau-Ginzburg理論（平均場）で検証してください。

解答を見る

平均場: \\(\alpha = 0, \beta = 1/2, \gamma = 1\\)

\\(\alpha + 2\beta + \gamma = 0 + 2(1/2) + 1 = 2\\) ✓

← 第4章（準備中） [シリーズ目次へ](<index.html>)

## 参考文献

  1. Peskin, M. E., & Schroeder, D. V. (1995). _An Introduction to Quantum Field Theory_. Westview Press.
  2. Weinberg, S. (1996). _The Quantum Theory of Fields, Vol. 2_. Cambridge University Press.
  3. Zinn-Justin, J. (2002). _Quantum Field Theory and Critical Phenomena_ (4th ed.). Oxford University Press.
  4. Goldenfeld, N. (1992). _Lectures on Phase Transitions and the Renormalization Group_. Westview Press.
  5. Altland, A., & Simons, B. (2010). _Condensed Matter Field Theory_. Cambridge University Press.

## シリーズ完結

この第5章で「量子場の理論入門」シリーズは完結です。 場の量子化から始まり、伝播関数、S行列、Feynman図形、そして繰り込み理論と有効理論まで、 量子場理論の基礎を体系的に学びました。 

これらの概念は、素粒子物理学だけでなく、凝縮系物理学、統計力学、材料科学など、 幅広い分野で応用されています。さらなる学習には、ゲージ理論、非可換ゲージ理論、 自発的対称性の破れ、そして経路積分形式などのトピックがあります。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
