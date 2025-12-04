---
title: "第3章: ラプラス方程式とポテンシャル理論"
chapter_title: "第3章: ラプラス方程式とポテンシャル理論"
subtitle: Laplace Equation and Potential Theory
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/pde-boundary-value/chapter-3.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [偏微分方程式と境界値問題](<index.html>) > 第3章 

## 🎯 学習目標

  * ラプラス方程式とポテンシャル理論の基礎を理解する
  * 調和関数の性質と最大値原理を学ぶ
  * 極座標・円筒座標でのラプラス方程式を解く
  * グリーン関数による境界値問題の解法を習得する
  * ポアソン方程式と電荷分布・熱源の扱いを理解する
  * 反復法による数値解法（Jacobi法、Gauss-Seidel法、SOR法）を実装する
  * 材料科学への応用（静電場解析、定常熱伝導）を理解する

## 📖 ラプラス方程式とは

### ラプラス方程式の定義

**ラプラス方程式（Laplace equation）** は、以下の形式の楕円型偏微分方程式です：

\\[ \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} = 0 \\]

ラプラス方程式の解を**調和関数（harmonic function）** と呼びます。

**ポアソン方程式（Poisson equation）** は、右辺が非ゼロの場合です：

\\[ \nabla^2 u = f(x,y,z) \\]

ここで \\(f\\) は熱源や電荷密度を表します。

### 物理的意義

  * **静電ポテンシャル** : 電荷のない領域での電位 \\(V\\) は \\(\nabla^2 V = 0\\)
  * **定常熱伝導** : 熱源のない領域での温度分布 \\(T\\) は \\(\nabla^2 T = 0\\)
  * **流体ポテンシャル** : 非圧縮・非粘性流体の速度ポテンシャル \\(\phi\\) は \\(\nabla^2 \phi = 0\\)
  * **重力ポテンシャル** : 質量のない領域での重力ポテンシャル

### 調和関数の性質

**最大値原理（Maximum Principle）** : 調和関数は内部に極値を持たず、最大値・最小値は境界上で達成されます。

**平均値定理（Mean Value Theorem）** : 点 \\((x_0, y_0)\\) における調和関数の値は、その点を中心とする円周上の平均値に等しい：

\\[ u(x_0, y_0) = \frac{1}{2\pi} \int_0^{2\pi} u(x_0 + r\cos\theta, y_0 + r\sin\theta) d\theta \\]

**一意性** : ディリクレ境界条件のもとで、ラプラス方程式の解は一意的です。

## 💻 例題3.1: 調和関数と最大値原理の検証

Python実装: 調和関数の性質確認

import numpy as np import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D # 調和関数の例: u(x,y) = x^2 - y^2 (実部 z^2 の調和関数) def harmonic_function(x, y): return x**2 - y**2 # 2次元グリッド作成 x = np.linspace(-2, 2, 100) y = np.linspace(-2, 2, 100) X, Y = np.meshgrid(x, y) U = harmonic_function(X, Y) # ラプラシアンの計算（数値微分） dx = x[1] - x[0] dy = y[1] - y[0] # 2次偏微分 d2u_dx2 = np.zeros_like(U) d2u_dy2 = np.zeros_like(U) for i in range(1, len(x)-1): for j in range(1, len(y)-1): d2u_dx2[j, i] = (U[j, i+1] - 2*U[j, i] + U[j, i-1]) / dx**2 d2u_dy2[j, i] = (U[j+1, i] - 2*U[j, i] + U[j-1, i]) / dy**2 laplacian = d2u_dx2 + d2u_dy2 # 可視化 fig = plt.figure(figsize=(15, 5)) # 調和関数 ax1 = fig.add_subplot(131, projection='3d') ax1.plot_surface(X, Y, U, cmap='viridis', alpha=0.8) ax1.set_xlabel('x') ax1.set_ylabel('y') ax1.set_zlabel('u(x,y)') ax1.set_title('調和関数: u = x² - y²') # 等高線図 ax2 = fig.add_subplot(132) contour = ax2.contour(X, Y, U, levels=20, cmap='viridis') ax2.clabel(contour, inline=True, fontsize=8) ax2.set_xlabel('x') ax2.set_ylabel('y') ax2.set_title('等高線図') ax2.axis('equal') # ラプラシアン ax3 = fig.add_subplot(133) laplacian_plot = ax3.imshow(laplacian[1:-1, 1:-1], extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1) plt.colorbar(laplacian_plot, ax=ax3, label='∇²u') ax3.set_xlabel('x') ax3.set_ylabel('y') ax3.set_title(f'ラプラシアン (max: {np.max(np.abs(laplacian[1:-1,1:-1])):.2e})') plt.tight_layout() plt.savefig('laplace_harmonic_function.png', dpi=300, bbox_inches='tight') plt.show() # 最大値原理の検証 print("=== 最大値原理の検証 ===") print(f"内部の最大値: {np.max(U[10:-10, 10:-10]):.4f}") print(f"境界の最大値: {np.max([np.max(U[0,:]), np.max(U[-1,:]), np.max(U[:,0]), np.max(U[:,-1])]):.4f}") print(f"内部の最小値: {np.min(U[10:-10, 10:-10]):.4f}") print(f"境界の最小値: {np.min([np.min(U[0,:]), np.min(U[-1,:]), np.min(U[:,0]), np.min(U[:,-1])]):.4f}") 

**出力解説** :

  * \\(u = x^2 - y^2\\) は調和関数（\\(\nabla^2 u = 2 - 2 = 0\\)）
  * ラプラシアンが数値誤差の範囲内でゼロであることを確認
  * 最大値原理により、極値は境界上に存在

## 📚 まとめ

  * **ラプラス方程式** は定常状態の物理現象を記述し、調和関数として最大値原理などの重要な性質を持つ
  * **極座標・円筒座標** での変数分離法により、円形・球形領域での解析解が得られる
  * **グリーン関数** により、点源に対する応答から境界値問題の解を構成できる
  * **ポアソン方程式** は熱源や電荷分布を含む問題を扱い、材料科学で広く応用される
  * **反復法** （Jacobi, Gauss-Seidel, SOR）により数値的に解を求めることができ、SOR法が最も高速
  * 複雑形状での定常熱伝導問題など、実用的な材料科学への応用が可能

### 💡 演習問題

  1. **調和関数の検証** : \\(u(x,y) = xy\\) が調和関数でないことを、ラプラシアンを計算して確認せよ。
  2. **極座標での解** : 半径 \\(a\\) の円盤上で境界条件 \\(u(a,\theta) = \cos(2\theta)\\) を満たすラプラス方程式の解を求め、可視化せよ。
  3. **グリーン関数の応用** : 矩形領域でのグリーン関数を用い、熱源 \\(f(x,y) = \delta(x-0.5, y-0.5)\\) に対する温度分布を求めよ。
  4. **収束の比較** : SOR法の緩和係数 \\(\omega\\) を 1.0 から 2.0 まで変化させ、最適な \\(\omega\\) を求めよ。
  5. **複雑形状** : 円形の穴が開いた矩形領域でのラプラス方程式を解き、穴の周りの温度分布を可視化せよ。

[← 第2章: 熱方程式と拡散](<chapter-2.html>) 第4章: 変分法と最適化 →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
