---
title: "第4章: ベクトル場と微分演算子"
chapter_title: "第4章: ベクトル場と微分演算子"
subtitle: Vector Fields and Differential Operators
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/calculus-vector-analysis/chapter-4.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [微積分とベクトル解析入門](<index.html>) > 第4章 

## 4.1 ベクトル場の定義と可視化

**📐 定義: ベクトル場**  
空間の各点に1つのベクトルを対応させる関数をベクトル場と呼びます： $$\mathbf{F}(\mathbf{r}) = (F_x(x,y,z), F_y(x,y,z), F_z(x,y,z))$$ 例: 流体の速度場、電場、磁場など 

### 💻 コード例1: 2次元ベクトル場の可視化

import numpy as np import matplotlib.pyplot as plt # 2次元ベクトル場の定義: F(x,y) = (-y, x) (回転場) def vector_field(x, y): """回転するベクトル場""" Fx = -y Fy = x return Fx, Fy # グリッドの作成 x = np.linspace(-3, 3, 15) y = np.linspace(-3, 3, 15) X, Y = np.meshgrid(x, y) Fx, Fy = vector_field(X, Y) # quiverプロットでベクトル場を可視化 plt.figure(figsize=(10, 8)) plt.quiver(X, Y, Fx, Fy, np.sqrt(Fx**2 + Fy**2), cmap='viridis') plt.colorbar(label='ベクトルの大きさ') plt.xlabel('x') plt.ylabel('y') plt.title('回転ベクトル場 F = (-y, x)') plt.axis('equal') plt.grid(True, alpha=0.3) plt.show() # 流線（streamline）の描画 x_fine = np.linspace(-3, 3, 100) y_fine = np.linspace(-3, 3, 100) X_fine, Y_fine = np.meshgrid(x_fine, y_fine) Fx_fine, Fy_fine = vector_field(X_fine, Y_fine) plt.figure(figsize=(10, 8)) plt.streamplot(X_fine, Y_fine, Fx_fine, Fy_fine, density=1.5, color='blue', linewidth=1) plt.xlabel('x') plt.ylabel('y') plt.title('ベクトル場の流線') plt.axis('equal') plt.grid(True, alpha=0.3) plt.show()

## 4.2 勾配（gradient, grad）

**📐 定義: 勾配**  
スカラー場 φ の勾配は： $$\nabla \phi = \text{grad}\,\phi = \left(\frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z}\right)$$ 勾配ベクトルは、φ が最も急激に増加する方向を指します。 

### 💻 コード例2: 勾配ベクトル場の計算と可視化

def scalar_field(x, y): """スカラー場: φ(x,y) = x² + y²""" return x**2 + y**2 def gradient_field(x, y): """勾配: ∇φ = (2x, 2y)""" grad_x = 2*x grad_y = 2*y return grad_x, grad_y # 可視化 x = np.linspace(-2, 2, 20) y = np.linspace(-2, 2, 20) X, Y = np.meshgrid(x, y) phi = scalar_field(X, Y) grad_x, grad_y = gradient_field(X, Y) fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # 左図: スカラー場の等高線 contour = ax1.contourf(X, Y, phi, levels=20, cmap='viridis') fig.colorbar(contour, ax=ax1, label='φ(x,y)') ax1.set_xlabel('x') ax1.set_ylabel('y') ax1.set_title('スカラー場 φ = x² + y²') ax1.axis('equal') # 右図: 勾配ベクトル場 ax2.contour(X, Y, phi, levels=10, colors='gray', alpha=0.3) ax2.quiver(X, Y, grad_x, grad_y, color='red') ax2.set_xlabel('x') ax2.set_ylabel('y') ax2.set_title('勾配ベクトル場 ∇φ = (2x, 2y)') ax2.axis('equal') plt.tight_layout() plt.show()

## 4.3 発散（divergence, div）

**📐 定義: 発散**  
ベクトル場 F の発散は： $$\text{div}\,\mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$$ 発散は、その点からのベクトル場の「湧き出し」の強さを表します。 

### 💻 コード例3: 発散の計算

def divergence_numerical(Fx, Fy, x, y, h=1e-5): """発散の数値計算: div F = ∂Fx/∂x + ∂Fy/∂y""" dFx_dx = (Fx(x+h, y) - Fx(x-h, y)) / (2*h) dFy_dy = (Fy(x, y+h) - Fy(x, y-h)) / (2*h) return dFx_dx + dFy_dy # 例1: 発散が正のベクトル場（発散場） def Fx_diverging(x, y): return x def Fy_diverging(x, y): return y # 例2: 発散がゼロのベクトル場（回転場） def Fx_rotating(x, y): return -y def Fy_rotating(x, y): return x # 発散の計算 x0, y0 = 1, 1 div_diverging = divergence_numerical(Fx_diverging, Fy_diverging, x0, y0) div_rotating = divergence_numerical(Fx_rotating, Fy_rotating, x0, y0) print(f"発散場 F=(x,y) の発散: div F = {div_diverging:.6f} (解析解: 2)") print(f"回転場 F=(-y,x) の発散: div F = {div_rotating:.6f} (解析解: 0)")

## 4.4 回転（rotation, curl）

**📐 定義: 回転**  
3次元ベクトル場 F の回転は： $$\text{rot}\,\mathbf{F} = \nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\\ F_x & F_y & F_z \end{vmatrix}$$ 

### 💻 コード例4: 2次元での回転（スカラー）

# 2次元では回転はスカラー値: rot F = ∂Fy/∂x - ∂Fx/∂y def curl_2d(Fx, Fy, x, y, h=1e-5): """2次元での回転の数値計算""" dFy_dx = (Fy(x+h, y) - Fy(x-h, y)) / (2*h) dFx_dy = (Fx(x, y+h) - Fx(x, y-h)) / (2*h) return dFy_dx - dFx_dy # 回転場での回転を計算 curl_rotating = curl_2d(Fx_rotating, Fy_rotating, x0, y0) print(f"\n回転場 F=(-y,x) の回転: rot F = {curl_rotating:.6f} (解析解: 2)") # 発散場での回転を計算 curl_diverging = curl_2d(Fx_diverging, Fy_diverging, x0, y0) print(f"発散場 F=(x,y) の回転: rot F = {curl_diverging:.6f} (解析解: 0)")

## 4.5 ラプラシアン（Laplacian, Δ）

**📐 定義: ラプラシアン**  
スカラー場 φ のラプラシアンは： $$\Delta \phi = \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}$$ 熱方程式、波動方程式など多くの物理法則に現れます。 

### 💻 コード例5: ラプラシアンの計算と応用

def laplacian_2d(phi, x, y, h=1e-4): """2次元ラプラシアンの数値計算""" lap = (phi(x+h, y) + phi(x-h, y) + phi(x, y+h) + phi(x, y-h) - 4*phi(x, y)) / h**2 return lap # テスト関数: φ(x,y) = x² + y² phi = lambda x, y: x**2 + y**2 lap = laplacian_2d(phi, 1, 1) print(f"\nφ = x² + y² のラプラシアン: Δφ = {lap:.6f} (解析解: 4)") # ラプラス方程式 Δφ = 0 の解（調和関数） phi_harmonic = lambda x, y: x**2 - y**2 lap_harmonic = laplacian_2d(phi_harmonic, 1, 1) print(f"φ = x² - y² のラプラシアン: Δφ = {lap_harmonic:.6f} (解析解: 0)")

## 4.6 保存場とポテンシャル関数

**📐 定理: 保存場の条件**  
ベクトル場 F が保存場（スカラーポテンシャル φ が存在）である条件は： $$\text{rot}\,\mathbf{F} = \mathbf{0}$$ このとき F = grad φ と表せます。 

### 💻 コード例6: 保存場の判定とポテンシャル関数の計算

import sympy as sp x, y = sp.symbols('x y') # ベクトル場 F = (2xy, x² + 2y) Fx_sym = 2*x*y Fy_sym = x**2 + 2*y # 回転の計算 curl_z = sp.diff(Fy_sym, x) - sp.diff(Fx_sym, y) print("ベクトル場 F = (2xy, x² + 2y) の保存場判定:") print(f"rot F = ∂Fy/∂x - ∂Fx/∂y = {curl_z}") if curl_z == 0: print("→ rot F = 0 なので保存場です\n") # ポテンシャル関数の計算 # φ を ∂φ/∂x = Fx, ∂φ/∂y = Fy となるように求める phi = sp.integrate(Fx_sym, x) # x で積分 print(f"∫ Fx dx = {phi} + g(y)") # y の関数 g(y) を決定 dPhi_dy = sp.diff(phi, y) g_prime = Fy_sym - dPhi_dy g = sp.integrate(g_prime, y) phi_final = phi + g print(f"ポテンシャル関数: φ = {phi_final}") # 検証 grad_phi_x = sp.diff(phi_final, x) grad_phi_y = sp.diff(phi_final, y) print(f"\n検証:") print(f"∂φ/∂x = {grad_phi_x} = Fx ✓") print(f"∂φ/∂y = {grad_phi_y} = Fy ✓")

### 💻 コード例7: 材料科学への応用（拡散流束）

# Fickの第一法則: J = -D ∇C （拡散流束） # 濃度勾配により拡散流束が生じる def concentration(x, y): """濃度分布 C(x,y)""" return np.exp(-(x**2 + y**2)) # 勾配（濃度勾配）を計算 x = np.linspace(-2, 2, 20) y = np.linspace(-2, 2, 20) X, Y = np.meshgrid(x, y) # 数値微分で勾配を計算 h = x[1] - x[0] C = concentration(X, Y) dC_dx = np.gradient(C, h, axis=1) dC_dy = np.gradient(C, h, axis=0) # 拡散流束: J = -D ∇C D = 1.0 # 拡散係数 Jx = -D * dC_dx Jy = -D * dC_dy fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # 左図: 濃度分布 contour = ax1.contourf(X, Y, C, levels=15, cmap='viridis') fig.colorbar(contour, ax=ax1, label='濃度 C') ax1.set_xlabel('x') ax1.set_ylabel('y') ax1.set_title('濃度分布 C(x,y) = exp(-(x²+y²))') ax1.axis('equal') # 右図: 拡散流束ベクトル ax2.contour(X, Y, C, levels=10, colors='gray', alpha=0.3) ax2.quiver(X, Y, Jx, Jy, color='red', alpha=0.7) ax2.set_xlabel('x') ax2.set_ylabel('y') ax2.set_title('拡散流束 J = -D∇C') ax2.axis('equal') plt.tight_layout() plt.show() # 発散 div J（正味の流出入） div_J = np.gradient(Jx, h, axis=1) + np.gradient(Jy, h, axis=0) print(f"\n中心点での発散: div J = {div_J[len(y)//2, len(x)//2]:.6f}") print("(負 → 流入, 正 → 流出)")

## まとめ

  * ベクトル場は空間の各点にベクトルを対応させ、流体速度や電場などを表現
  * 勾配（grad）はスカラー場が最も急激に増加する方向を指すベクトル場
  * 発散（div）はベクトル場の「湧き出し」、回転（curl）は「渦」を表す
  * ラプラシアンは拡散や波動現象を記述する重要な演算子
  * 保存場では回転がゼロで、ポテンシャル関数が存在する

[← 第3章: 多変数関数](<chapter-3.html>) [第5章: 線積分・面積分 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
