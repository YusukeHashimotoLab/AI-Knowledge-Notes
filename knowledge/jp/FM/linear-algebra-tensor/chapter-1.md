---
title: "第1章: ベクトルと行列の基礎"
chapter_title: "第1章: ベクトルと行列の基礎"
subtitle: Vectors and Matrices Fundamentals
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/linear-algebra-tensor/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [線形代数とテンソル解析](<index.html>) > 第1章 

## 1.1 ベクトルの基礎

**📐 定義: ベクトル**  
n次元ベクトルは n 個の数の組： \\[\mathbf{v} = \begin{pmatrix} v_1 \\\ v_2 \\\ \vdots \\\ v_n \end{pmatrix}\\] 大きさと向きを持つ量を表します（位置、速度、力など）。 

### 💻 コード例1: NumPyによるベクトル演算

Python実装: NumPyによるベクトル演算

import numpy as np import matplotlib.pyplot as plt # ベクトルの定義 v1 = np.array([3, 2]) v2 = np.array([1, 4]) # ベクトルの演算 v_sum = v1 + v2 # 和 v_diff = v1 - v2 # 差 v_scalar = 2 * v1 # スカラー倍 print("ベクトル演算:") print(f"v1 = {v1}") print(f"v2 = {v2}") print(f"v1 + v2 = {v_sum}") print(f"v1 - v2 = {v_diff}") print(f"2 * v1 = {v_scalar}") # 可視化 fig, ax = plt.subplots(figsize=(8, 8)) origin = [0, 0] # ベクトルを矢印で描画 ax.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='blue', width=0.01, label='v1') ax.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='red', width=0.01, label='v2') ax.quiver(*origin, *v_sum, angles='xy', scale_units='xy', scale=1, color='green', width=0.01, label='v1+v2') # 平行四辺形の法則を表示 ax.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], 'k--', alpha=0.3) ax.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], 'k--', alpha=0.3) ax.set_xlim(-1, 6) ax.set_ylim(-1, 7) ax.set_xlabel('x') ax.set_ylabel('y') ax.set_title('ベクトルの加法') ax.legend() ax.grid(True, alpha=0.3) ax.axhline(y=0, color='k', linewidth=0.5) ax.axvline(x=0, color='k', linewidth=0.5) plt.axis('equal') plt.show()

## 1.2 内積とノルム

**📐 定義: 内積とノルム**  
ベクトル v, w の内積： \\[\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_i w_i = v_1 w_1 + v_2 w_2 + \cdots + v_n w_n\\] ノルム（長さ）： \\[\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}\\] 

### 💻 コード例2: 内積とノルムの計算

# 内積の計算 dot_product = np.dot(v1, v2) # または v1 @ v2 でも可 # ノルムの計算 norm_v1 = np.linalg.norm(v1) norm_v2 = np.linalg.norm(v2) # 角度の計算: cos θ = (v·w) / (||v|| ||w||) cos_theta = dot_product / (norm_v1 * norm_v2) theta_rad = np.arccos(cos_theta) theta_deg = np.degrees(theta_rad) print(f"\n内積と角度:") print(f"v1 · v2 = {dot_product}") print(f"||v1|| = {norm_v1:.4f}") print(f"||v2|| = {norm_v2:.4f}") print(f"v1とv2の成す角度: {theta_deg:.2f}°") # 単位ベクトル（正規化） v1_unit = v1 / norm_v1 v2_unit = v2 / norm_v2 print(f"\nv1の単位ベクトル: {v1_unit}") print(f"単位ベクトルのノルム: {np.linalg.norm(v1_unit):.10f}")

## 1.3 外積（3次元）

**📐 定義: 外積**  
3次元ベクトル v, w の外積： \\[\mathbf{v} \times \mathbf{w} = \begin{pmatrix} v_2 w_3 - v_3 w_2 \\\ v_3 w_1 - v_1 w_3 \\\ v_1 w_2 - v_2 w_1 \end{pmatrix}\\] 結果は v と w の両方に垂直なベクトルです。 

### 💻 コード例3: 外積の計算と可視化

from mpl_toolkits.mplot3d import Axes3D # 3次元ベクトル v_3d = np.array([1, 0, 0]) w_3d = np.array([0, 1, 0]) # 外積の計算 cross_product = np.cross(v_3d, w_3d) print("外積の計算:") print(f"v = {v_3d}") print(f"w = {w_3d}") print(f"v × w = {cross_product}") print(f"||v × w|| = {np.linalg.norm(cross_product):.4f}") # v × w が v, w の両方に垂直であることを確認 print(f"\n垂直性の確認:") print(f"(v × w) · v = {np.dot(cross_product, v_3d):.10f}") print(f"(v × w) · w = {np.dot(cross_product, w_3d):.10f}") # 3D可視化 fig = plt.figure(figsize=(10, 8)) ax = fig.add_subplot(111, projection='3d') origin = [0, 0, 0] ax.quiver(*origin, *v_3d, color='blue', arrow_length_ratio=0.1, linewidth=2, label='v') ax.quiver(*origin, *w_3d, color='red', arrow_length_ratio=0.1, linewidth=2, label='w') ax.quiver(*origin, *cross_product, color='green', arrow_length_ratio=0.1, linewidth=2, label='v×w') ax.set_xlim([-0.5, 1.5]) ax.set_ylim([-0.5, 1.5]) ax.set_zlim([-0.5, 1.5]) ax.set_xlabel('X') ax.set_ylabel('Y') ax.set_zlabel('Z') ax.set_title('外積: v × w') ax.legend() plt.show()

## 1.4 行列の基本演算

**📐 定義: 行列**  
m × n 行列は m 行 n 列の数の配列： \\[A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\\ a_{21} & a_{22} & \cdots & a_{2n} \\\ \vdots & \vdots & \ddots & \vdots \\\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}\\] 

### 💻 コード例4: 行列の基本演算

# 行列の定義 A = np.array([[1, 2, 3], [4, 5, 6]]) B = np.array([[7, 8], [9, 10], [11, 12]]) print("行列の演算:") print(f"A (2×3) =\n{A}\n") print(f"B (3×2) =\n{B}\n") # 転置行列 A_T = A.T print(f"A^T (転置) =\n{A_T}\n") # 行列の積 (2×3) × (3×2) = (2×2) C = A @ B # または np.dot(A, B) print(f"A × B (2×2) =\n{C}\n") # 要素ごとの演算 D = A + A # 同じ形状なら加算可能 E = 2 * A # スカラー倍 print(f"A + A =\n{D}\n") print(f"2 * A =\n{E}")

## 1.5 特殊な行列

### 💻 コード例5: 単位行列、零行列、対角行列

# 単位行列（対角成分が1） I = np.eye(3) print("単位行列 I (3×3):") print(I) # 零行列 Z = np.zeros((2, 3)) print(f"\n零行列 O (2×3):\n{Z}") # 対角行列 diag_values = [1, 2, 3] D_diag = np.diag(diag_values) print(f"\n対角行列 D:\n{D_diag}") # 行列の対角成分を取り出す A_square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) diag_A = np.diag(A_square) print(f"\nA の対角成分: {diag_A}") # トレース（対角成分の和） trace_A = np.trace(A_square) print(f"tr(A) = {trace_A}")

## 1.6 逆行列

**📐 定義: 逆行列**  
正方行列 A に対して AA⁻¹ = A⁻¹A = I となる A⁻¹ を逆行列と呼びます。 逆行列が存在する ⇔ det(A) ≠ 0（正則行列） 

### 💻 コード例6: 逆行列の計算

# 正則行列 A_inv = np.array([[1, 2], [3, 4]]) # 逆行列の計算 A_inverse = np.linalg.inv(A_inv) print("逆行列の計算:") print(f"A =\n{A_inv}\n") print(f"A^(-1) =\n{A_inverse}\n") # 検証: A × A^(-1) = I product = A_inv @ A_inverse print(f"A × A^(-1) =\n{product}\n") print(f"単位行列に近いか: {np.allclose(product, np.eye(2))}") # 行列式 det_A = np.linalg.det(A_inv) print(f"\ndet(A) = {det_A:.4f}") print(f"det(A) ≠ 0 なので逆行列が存在 ✓")

## 1.7 線形変換の幾何学的意味

### 💻 コード例7: 回転行列

# 回転行列: 反時計回りに θ 回転 def rotation_matrix(theta): """2D回転行列""" cos_t = np.cos(theta) sin_t = np.sin(theta) return np.array([[cos_t, -sin_t], [sin_t, cos_t]]) # 45度回転 theta = np.pi / 4 # 45度 R = rotation_matrix(theta) print(f"45度回転行列:") print(R) # 元のベクトル points = np.array([[1, 0], [0, 1], [1, 1], [0, 0]]) # 回転後のベクトル rotated_points = points @ R.T # 可視化 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # 元の図形 ax1.plot([0, 1], [0, 0], 'b-', linewidth=2, marker='o') ax1.plot([0, 0], [0, 1], 'r-', linewidth=2, marker='o') ax1.set_xlim(-1.5, 1.5) ax1.set_ylim(-1.5, 1.5) ax1.set_xlabel('x') ax1.set_ylabel('y') ax1.set_title('元の図形') ax1.grid(True, alpha=0.3) ax1.axhline(y=0, color='k', linewidth=0.5) ax1.axvline(x=0, color='k', linewidth=0.5) ax1.axis('equal') # 回転後の図形 ax2.plot([0, rotated_points[0,0]], [0, rotated_points[0,1]], 'b-', linewidth=2, marker='o') ax2.plot([0, rotated_points[1,0]], [0, rotated_points[1,1]], 'r-', linewidth=2, marker='o') ax2.set_xlim(-1.5, 1.5) ax2.set_ylim(-1.5, 1.5) ax2.set_xlabel('x') ax2.set_ylabel('y') ax2.set_title('45度回転後') ax2.grid(True, alpha=0.3) ax2.axhline(y=0, color='k', linewidth=0.5) ax2.axvline(x=0, color='k', linewidth=0.5) ax2.axis('equal') plt.tight_layout() plt.show()

### 💻 コード例8: スケーリングとせん断

# スケーリング行列（拡大・縮小） S = np.array([[2, 0], [0, 0.5]]) # せん断（shear）行列 Sh = np.array([[1, 0.5], [0, 1]]) # 元の正方形の頂点 square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) # 変換後 scaled = square @ S.T sheared = square @ Sh.T fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # 元の図形 axes[0].plot(square[:,0], square[:,1], 'b-', linewidth=2) axes[0].fill(square[:,0], square[:,1], alpha=0.3) axes[0].set_title('元の正方形') axes[0].set_xlim(-0.5, 2.5) axes[0].set_ylim(-0.5, 2.5) axes[0].grid(True, alpha=0.3) axes[0].axis('equal') # スケーリング axes[1].plot(scaled[:,0], scaled[:,1], 'r-', linewidth=2) axes[1].fill(scaled[:,0], scaled[:,1], alpha=0.3, color='red') axes[1].set_title('スケーリング (2倍, 0.5倍)') axes[1].set_xlim(-0.5, 2.5) axes[1].set_ylim(-0.5, 2.5) axes[1].grid(True, alpha=0.3) axes[1].axis('equal') # せん断 axes[2].plot(sheared[:,0], sheared[:,1], 'g-', linewidth=2) axes[2].fill(sheared[:,0], sheared[:,1], alpha=0.3, color='green') axes[2].set_title('せん断（shear）') axes[2].set_xlim(-0.5, 2.5) axes[2].set_ylim(-0.5, 2.5) axes[2].grid(True, alpha=0.3) axes[2].axis('equal') plt.tight_layout() plt.show() print("線形変換:") print(f"スケーリング行列:\n{S}") print(f"\nせん断行列:\n{Sh}")

## まとめ

  * ベクトルは大きさと向きを持つ量で、NumPy配列として効率的に扱える
  * 内積はベクトルの成す角度の計算、外積は垂直なベクトルの生成に使われる
  * 行列は線形変換を表現し、回転・スケーリング・せん断などを統一的に扱える
  * 逆行列は線形方程式の解法や変換の逆操作に必要不可欠
  * NumPyの線形代数関数により、複雑な演算も簡潔に実装できる

[← シリーズトップ](<index.html>) [第2章: 行列式と連立方程式 →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
