---
title: "第1章: 波動方程式と振動現象"
chapter_title: "第1章: 波動方程式と振動現象"
subtitle: Wave Equation and Oscillations
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/pde-boundary-value/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [偏微分方程式と境界値問題](<index.html>) > 第1章 

## 1.1 波動方程式の導出

弦の振動から1次元波動方程式を導出します。 

**📐 理論**  

**1次元波動方程式:**

\\[\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}\\] 

ここで \\(u(x,t)\\) は変位、\\(c\\) は波の伝播速度

**導出:** 張力 \\(T\\) の弦、線密度 \\(\rho\\) のとき \\(c = \sqrt{T/\rho}\\)

💻 コード例 1: 波動方程式の数値シミュレーション（有限差分法）

import numpy as np import matplotlib.pyplot as plt from matplotlib.animation import FuncAnimation from IPython.display import HTML # パラメータ L = 10.0 # 弦の長さ c = 1.0 # 波の伝播速度 T = 20.0 # 時間範囲 Nx = 200 # 空間格子点数 Nt = 1000 # 時間ステップ数 # 格子 x = np.linspace(0, L, Nx) t = np.linspace(0, T, Nt) dx = x[1] - x[0] dt = t[1] - t[0] # CFL条件の確認 r = c * dt / dx # クーラント数 print(f"クーラント数 r = {r:.3f} (安定条件: r ≤ 1)") # 初期条件: 三角波 def initial_displacement(x): u = np.zeros_like(x) peak = L / 2 width = L / 5 mask = np.abs(x - peak) < width u[mask] = 1 - np.abs(x[mask] - peak) / width return u # 初期化 u = np.zeros((Nt, Nx)) u[0] = initial_displacement(x) u[1] = u[0].copy() # 初速度=0 # 有限差分法（陽的スキーム） for n in range(1, Nt-1): for i in range(1, Nx-1): u[n+1, i] = (2*(1-r**2)*u[n, i] - u[n-1, i] + r**2*(u[n, i+1] + u[n, i-1])) # 境界条件: u(0,t) = u(L,t) = 0 u[n+1, 0] = 0 u[n+1, -1] = 0 # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # スナップショット times = [0, Nt//4, Nt//2, 3*Nt//4] for idx, n in enumerate(times): ax = axes[idx//2, idx%2] ax.plot(x, u[n], 'b-', linewidth=2) ax.axhline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) ax.set_xlabel('位置 x', fontsize=12) ax.set_ylabel('変位 u(x,t)', fontsize=12) ax.set_title(f't = {t[n]:.2f}', fontsize=12) ax.set_ylim(-1.2, 1.2) plt.suptitle('波動方程式の数値解（有限差分法）', fontsize=14) plt.tight_layout() plt.show() # 時空間プロット plt.figure(figsize=(12, 6)) plt.contourf(x, t, u, levels=50, cmap='RdBu_r') plt.colorbar(label='変位 u(x,t)') plt.xlabel('位置 x', fontsize=12) plt.ylabel('時間 t', fontsize=12) plt.title('波動の伝播（時空間図）', fontsize=14) plt.tight_layout() plt.show() print("\n波動方程式の性質:") print("- 波は左右に伝播し、境界で反射") print("- エネルギーは保存される") print(f"- 伝播速度: c = {c} m/s")

## 1.2 ダランベールの解（進行波解）

無限領域での波動方程式の一般解はダランベールの公式で表されます。 

**📐 理論**  

**ダランベールの解:**

\\[u(x,t) = f(x-ct) + g(x+ct)\\] 

左進行波 \\(f(x-ct)\\) と右進行波 \\(g(x+ct)\\) の重ね合わせ

💻 コード例 2: ダランベールの解の可視化

import numpy as np import matplotlib.pyplot as plt # パラメータ c = 1.0 x = np.linspace(-10, 10, 500) t_values = np.linspace(0, 5, 6) # 初期波形 def f(xi): """左進行波""" return np.exp(-xi**2) def g(xi): """右進行波""" return 0.5 * np.exp(-(xi-2)**2 / 0.5) # 可視化 fig, axes = plt.subplots(3, 2, figsize=(14, 12)) axes = axes.flatten() for idx, t in enumerate(t_values): ax = axes[idx] # ダランベールの解 u_left = f(x - c*t) u_right = g(x + c*t) u_total = u_left + u_right ax.plot(x, u_left, 'b--', linewidth=1.5, alpha=0.7, label='左進行波 f(x-ct)') ax.plot(x, u_right, 'r--', linewidth=1.5, alpha=0.7, label='右進行波 g(x+ct)') ax.plot(x, u_total, 'k-', linewidth=2, label='全波形 u(x,t)') ax.axhline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) ax.set_xlabel('位置 x', fontsize=11) ax.set_ylabel('変位 u', fontsize=11) ax.set_title(f't = {t:.2f}', fontsize=12) ax.set_ylim(-0.5, 1.5) ax.legend(fontsize=9) plt.suptitle('ダランベールの解: 進行波の重ね合わせ', fontsize=14) plt.tight_layout() plt.show() print("=== ダランベールの解の性質 ===") print("- 左進行波: f(x-ct) は速度 c で右に移動") print("- 右進行波: g(x+ct) は速度 c で左に移動") print("- 総合波形: 重ね合わせの原理により u = f + g")

## まとめ

  * 波動方程式は弦の振動から導出され、波の伝播を記述する
  * ダランベールの解により、左右に進行する波の重ね合わせとして理解できる
  * 有限差分法により数値的に解くことができる（CFL条件に注意）
  * 材料科学では超音波探傷などに応用される

[← シリーズトップ](<index.html>) [第2章へ →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
