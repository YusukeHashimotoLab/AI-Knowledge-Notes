---
title: "第1章: 複素数と複素平面"
chapter_title: "第1章: 複素数と複素平面"
subtitle: Complex Numbers and Complex Plane
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/complex-special-functions/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [複素関数論と特殊関数](<index.html>) > 第1章 

## 1.1 複素数の基本演算

複素数は \\(z = x + iy\\) の形で表され、実部 \\(x\\) と虚部 \\(y\\) を持ちます。Pythonでは`complex`型やNumPyで扱えます。 

**📐 定義: 複素数**  
複素数の定義: \\[z = x + iy, \quad i = \sqrt{-1}\\] 基本演算: 

  * 加法: \\((x_1 + iy_1) + (x_2 + iy_2) = (x_1 + x_2) + i(y_1 + y_2)\\)
  * 乗法: \\((x_1 + iy_1)(x_2 + iy_2) = (x_1x_2 - y_1y_2) + i(x_1y_2 + x_2y_1)\\)
  * 共役: \\(\bar{z} = x - iy\\)
  * 絶対値: \\(|z| = \sqrt{x^2 + y^2}\\)

### 💻 コード例 1: 複素数の基本演算

Python実装: 複素数の基本演算

import numpy as np import matplotlib.pyplot as plt # 複素数の定義 z1 = 3 + 4j z2 = 1 - 2j print(f"z1 = {z1}") print(f"z2 = {z2}") print(f"z1 + z2 = {z1 + z2}") print(f"z1 * z2 = {z1 * z2}") print(f"z1 / z2 = {z1 / z2}") # 共役複素数と絶対値 print(f"\n共役: z1.conjugate() = {z1.conjugate()}") print(f"絶対値: |z1| = {np.abs(z1)}") print(f"偏角: arg(z1) = {np.angle(z1)} rad = {np.degrees(np.angle(z1)):.2f}°") # 複素平面での可視化 fig, ax = plt.subplots(figsize=(8, 8)) ax.axhline(0, color='gray', linewidth=0.5) ax.axvline(0, color='gray', linewidth=0.5) ax.grid(True, alpha=0.3) # 複素数をベクトルとして描画 def plot_complex(z, label, color): ax.arrow(0, 0, z.real, z.imag, head_width=0.3, head_length=0.2, fc=color, ec=color, linewidth=2, label=label) ax.plot(z.real, z.imag, 'o', color=color, markersize=8) ax.text(z.real + 0.3, z.imag + 0.3, label, fontsize=12, color=color) plot_complex(z1, 'z1', 'blue') plot_complex(z2, 'z2', 'red') plot_complex(z1 + z2, 'z1+z2', 'green') ax.set_xlabel('実部 (Re)', fontsize=12) ax.set_ylabel('虚部 (Im)', fontsize=12) ax.set_title('複素平面でのベクトル表示', fontsize=14) ax.legend() ax.axis('equal') ax.set_xlim(-1, 5) ax.set_ylim(-3, 5) plt.tight_layout() plt.show()

## 1.2 極形式とオイラーの公式

複素数は極形式 \\(z = r e^{i\theta}\\) でも表現でき、これはオイラーの公式 \\(e^{i\theta} = \cos\theta + i\sin\theta\\) に基づきます。 

**📐 定理: オイラーの公式**  
極形式表示: \\[z = r e^{i\theta} = r(\cos\theta + i\sin\theta)\\] ここで \\(r = |z|\\) (絶対値), \\(\theta = \arg(z)\\) (偏角)  
特殊な場合: \\[e^{i\pi} + 1 = 0 \quad \text{(オイラーの等式)}\\] 

## まとめ

  * 複素数は実部と虚部から構成され、複素平面上のベクトルとして表現できる
  * 極形式表示により、複素数の乗除算が回転と拡大縮小として理解できる
  * オイラーの公式は複素数と三角関数を結びつける重要な関係式
  * 複素数は交流回路解析や量子力学など様々な物理現象の記述に使われる

[← シリーズトップ](<index.html>) [第2章: 正則関数と複素微積分 →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
