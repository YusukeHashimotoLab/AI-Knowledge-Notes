---
title: "第2章: 熱伝導方程式と拡散現象"
chapter_title: "第2章: 熱伝導方程式と拡散現象"
subtitle: Heat Equation and Diffusion
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/pde-boundary-value/chapter-2.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [偏微分方程式と境界値問題](<index.html>) > 第2章 

## 2.1 熱伝導方程式の導出

フーリエの熱伝導法則から拡散方程式を導出します。 

**📐 理論**  

**熱伝導方程式（拡散方程式）:**

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$ 

ここで $u(x,t)$ は温度、$\alpha = k/(\rho c_p)$ は熱拡散率

$k$: 熱伝導率、$\rho$: 密度、$c_p$: 比熱

💻 コード例 1: 熱伝導方程式の数値解（陽的スキーム）

import numpy as np import matplotlib.pyplot as plt # パラメータ L = 10.0 # 棒の長さ alpha = 0.1 # 熱拡散率 T_total = 10.0 Nx = 100 Nt = 2000 x = np.linspace(0, L, Nx) t = np.linspace(0, T_total, Nt) dx = x[1] - x[0] dt = t[1] - t[0] # 安定性条件の確認 r = alpha * dt / dx**2 print(f"安定性パラメータ r = {r:.4f} (安定条件: r ≤ 0.5)") # 初期条件: ステップ関数 u = np.zeros((Nt, Nx)) u[0, Nx//4:3*Nx//4] = 100.0 # 境界条件: u(0,t) = u(L,t) = 0 # 陽的スキーム（FTCS: Forward Time Central Space） for n in range(Nt-1): for i in range(1, Nx-1): u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) u[n+1, 0] = 0 u[n+1, -1] = 0 # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() times_idx = [0, Nt//10, Nt//3, Nt-1] for idx, n in enumerate(times_idx): ax = axes[idx] ax.plot(x, u[n], 'b-', linewidth=2) ax.axhline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) ax.set_xlabel('位置 x', fontsize=12) ax.set_ylabel('温度 u(x,t)', fontsize=12) ax.set_title(f't = {t[n]:.2f}', fontsize=12) ax.set_ylim(-10, 110) plt.suptitle('熱伝導方程式の数値解（陽的スキーム）', fontsize=14) plt.tight_layout() plt.show() # 時空間プロット plt.figure(figsize=(12, 6)) plt.contourf(x, t, u, levels=50, cmap='hot') plt.colorbar(label='温度 u(x,t)') plt.xlabel('位置 x', fontsize=12) plt.ylabel('時間 t', fontsize=12) plt.title('熱拡散の時空間図', fontsize=14) plt.tight_layout() plt.show() print("\n熱伝導方程式の性質:") print("- 熱は高温から低温へ拡散") print("- 時間とともに温度分布は滑らかになる") print("- エントロピーは増大（不可逆過程）")

## 2.2 基本解（ガウス核）

無限領域での熱伝導方程式の基本解を学びます。 

**📐 理論**  

**基本解（ガウス核）:**

$$G(x,t) = \frac{1}{\sqrt{4\pi\alpha t}} e^{-x^2/(4\alpha t)}$$ 

**畳み込みによる一般解:**

$$u(x,t) = \int_{-\infty}^{\infty} G(x-\xi, t) f(\xi) d\xi$$ 

💻 コード例 2: 基本解（ガウス核）の可視化

import numpy as np import matplotlib.pyplot as plt # パラメータ alpha = 0.1 x = np.linspace(-10, 10, 500) # 基本解 def fundamental_solution(x, t, alpha): if t <= 0: return np.zeros_like(x) return (1/np.sqrt(4*np.pi*alpha*t)) * np.exp(-x**2 / (4*alpha*t)) # 時間発展 t_values = [0.1, 0.5, 1.0, 2.0, 5.0] fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # 基本解の時間発展 for t in t_values: G = fundamental_solution(x, t, alpha) ax1.plot(x, G, linewidth=2, label=f't = {t}') ax1.grid(True, alpha=0.3) ax1.set_xlabel('位置 x', fontsize=12) ax1.set_ylabel('G(x,t)', fontsize=12) ax1.set_title('基本解（ガウス核）の時間発展', fontsize=14) ax1.legend() # 畳み込みによる一般解 # 初期条件: ステップ関数 def initial_condition(x): return np.where(np.abs(x) < 1, 1.0, 0.0) # 畳み込み def convolution_solution(x, t, alpha, f): xi = np.linspace(-20, 20, 1000) dxi = xi[1] - xi[0] u = np.zeros_like(x) for i, x_val in enumerate(x): G = fundamental_solution(x_val - xi, t, alpha) integrand = G * f(xi) u[i] = np.trapz(integrand, dx=dxi) return u # 異なる時刻での解 for t in [0.5, 1.0, 2.0, 5.0]: u = convolution_solution(x, t, alpha, initial_condition) ax2.plot(x, u, linewidth=2, label=f't = {t}') # 初期条件 ax2.plot(x, initial_condition(x), 'k--', linewidth=2, alpha=0.5, label='初期条件') ax2.grid(True, alpha=0.3) ax2.set_xlabel('位置 x', fontsize=12) ax2.set_ylabel('温度 u(x,t)', fontsize=12) ax2.set_title('畳み込みによる一般解', fontsize=14) ax2.legend() plt.tight_layout() plt.show() print("=== 基本解の性質 ===") print("- 質量保存: ∫G(x,t)dx = 1") print("- 時間とともに広がる（拡散）") print("- t→0 でデルタ関数に近づく")

## 2.3 変数分離法による境界値問題

有限領域での熱伝導方程式を変数分離法で解きます。 

💻 コード例 3: 変数分離解とフーリエ級数

import numpy as np import matplotlib.pyplot as plt # パラメータ L = 10.0 alpha = 0.1 n_modes = 20 x = np.linspace(0, L, 200) # 初期条件 def initial_temp(x): return 100 * np.sin(np.pi * x / L) + 50 * np.sin(2 * np.pi * x / L) # フーリエ係数 def compute_fourier_coeff(f, L, n_max): coeffs = [] for n in range(1, n_max+1): x_int = np.linspace(0, L, 1000) integrand = f(x_int) * np.sin(n * np.pi * x_int / L) c_n = (2/L) * np.trapz(integrand, x_int) coeffs.append(c_n) return coeffs c_n = compute_fourier_coeff(initial_temp, L, n_modes) print(f"フーリエ係数 c_n: {c_n[:5]}") # 変数分離解 def separation_solution(x, t, L, alpha, c_n): u = np.zeros_like(x) for n, c in enumerate(c_n, 1): lambda_n = (n * np.pi / L)**2 u += c * np.sin(n * np.pi * x / L) * np.exp(-alpha * lambda_n * t) return u # 可視化 fig, axes = plt.subplots(2, 3, figsize=(15, 10)) axes = axes.flatten() times = [0, 1, 2, 5, 10, 20] for idx, t in enumerate(times): ax = axes[idx] u = separation_solution(x, t, L, alpha, c_n) ax.plot(x, u, 'b-', linewidth=2) ax.axhline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) ax.set_xlabel('位置 x', fontsize=11) ax.set_ylabel('温度 u(x,t)', fontsize=11) ax.set_title(f't = {t:.1f}', fontsize=12) ax.set_ylim(-200, 200) plt.suptitle('変数分離解: フーリエ級数展開', fontsize=14) plt.tight_layout() plt.show() # 各モードの減衰 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # モードの時間減衰 t_range = np.linspace(0, 20, 200) for n in range(1, 6): lambda_n = (n * np.pi / L)**2 amplitude = np.exp(-alpha * lambda_n * t_range) ax1.plot(t_range, amplitude, linewidth=2, label=f'Mode {n}') ax1.grid(True, alpha=0.3) ax1.set_xlabel('時間 t', fontsize=12) ax1.set_ylabel('振幅（規格化）', fontsize=12) ax1.set_title('各モードの時間減衰', fontsize=14) ax1.legend() # 減衰率 modes = np.arange(1, 11) decay_rates = alpha * (modes * np.pi / L)**2 ax2.plot(modes, decay_rates, 'bo-', linewidth=2, markersize=8) ax2.grid(True, alpha=0.3) ax2.set_xlabel('モード番号 n', fontsize=12) ax2.set_ylabel('減衰率 $\\\alpha\\\lambda_n$', fontsize=12) ax2.set_title('減衰率とモード番号の関係', fontsize=14) plt.tight_layout() plt.show() print("\n変数分離解の特徴:") print("- 高次モードほど速く減衰") print("- 長時間後は基本モード（n=1）が支配的") print(f"- 基本モードの減衰時定数: τ = 1/(αλ₁) = {1/(alpha*(np.pi/L)**2):.2f}")

## 2.4 境界条件の種類

Dirichlet、Neumann、Robin境界条件を学びます。 

**📐 理論**  

**境界条件の種類:**

  * **Dirichlet:** $u(0,t) = T_0$ （温度指定）
  * **Neumann:** $\frac{\partial u}{\partial x}(0,t) = q_0$ （熱流束指定）
  * **Robin:** $\frac{\partial u}{\partial x} + hu = 0$ （対流境界）

💻 コード例 4: 異なる境界条件の比較

import numpy as np import matplotlib.pyplot as plt # パラメータ L = 10.0 alpha = 0.1 T_total = 15.0 Nx = 100 Nt = 1500 x = np.linspace(0, L, Nx) dx = x[1] - x[0] dt = T_total / Nt r = alpha * dt / dx**2 # 初期条件 def init_cond(x): return 100 * np.exp(-((x - L/2)/2)**2) # 3種類の境界条件でシミュレーション cases = { 'Dirichlet (u=0)': 'dirichlet', 'Neumann (∂u/∂x=0)': 'neumann', 'Robin (対流)': 'robin' } solutions = {} for name, bc_type in cases.items(): u = np.zeros((Nt, Nx)) u[0] = init_cond(x) for n in range(Nt-1): for i in range(1, Nx-1): u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) # 境界条件 if bc_type == 'dirichlet': # 両端で温度固定 u[n+1, 0] = 0 u[n+1, -1] = 0 elif bc_type == 'neumann': # 両端で断熱（∂u/∂x = 0） u[n+1, 0] = u[n+1, 1] u[n+1, -1] = u[n+1, -2] elif bc_type == 'robin': # 対流境界条件（簡易） h = 0.5 # 対流係数 u[n+1, 0] = (u[n+1, 1] + h*dx*0) / (1 + h*dx) u[n+1, -1] = (u[n+1, -2] + h*dx*0) / (1 + h*dx) solutions[name] = u # 可視化 fig, axes = plt.subplots(2, 3, figsize=(16, 10)) times_idx = [0, Nt//5, 2*Nt//5, 3*Nt//5, 4*Nt//5, Nt-1] t = np.linspace(0, T_total, Nt) for idx, n in enumerate(times_idx): ax = axes[idx//3, idx%3] for name, u in solutions.items(): ax.plot(x, u[n], linewidth=2, label=name, alpha=0.8) ax.axhline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) ax.set_xlabel('位置 x', fontsize=11) ax.set_ylabel('温度 u(x,t)', fontsize=11) ax.set_title(f't = {t[n]:.2f}', fontsize=12) ax.set_ylim(-10, 110) if idx == 0: ax.legend() plt.suptitle('境界条件の種類による温度分布の違い', fontsize=14) plt.tight_layout() plt.show() print("=== 境界条件の物理的意味 ===") print("Dirichlet: 一定温度に保たれた境界（例: 氷水浴）") print("Neumann: 断熱境界（例: 断熱材）") print("Robin: 対流境界（例: 空気中への放熱）")

## 📝 章末問題

**演習問題**

  1. 熱拡散率 $\alpha = 0.2$ cm²/s、長さ $L=10$ cm の棒で、基本モードの減衰時定数を求めよ。
  2. Neumann境界条件で全体が断熱されている場合、平衡状態での温度分布を求めよ。
  3. 2次元正方形領域で、1つの辺のみが高温（他は低温）の場合の定常状態を求めよ。
  4. Crank-Nicolson法で $r=2$ の場合の数値解を陽的法と比較せよ。

[← 第1章へ](<chapter-1.html>) [第3章へ →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
