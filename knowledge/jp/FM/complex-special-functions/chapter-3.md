---
title: "第3章: Laurent展開と留数定理"
chapter_title: "第3章: Laurent展開と留数定理"
subtitle: Laurent Series and Residue Theorem
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/complex-special-functions/chapter-3.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [複素関数論と特殊関数](<index.html>) > 第3章 

## 3.1 Taylor級数とMaclaurin展開

正則関数は収束円内でTaylor級数に展開できます。 

**📐 定義: Taylor級数展開**  
$$f(z) = \sum_{n=0}^{\infty} \frac{f^{(n)}(z_0)}{n!} (z - z_0)^n$$ **Maclaurin展開 ($z_0 = 0$):** $$f(z) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} z^n$$ 

### 💻 コード例 1: Taylor級数展開の計算

Python実装: Taylor級数による関数近似

import numpy as np import matplotlib.pyplot as plt from scipy.special import factorial import sympy as sp # SymPyで記号計算 z = sp.Symbol('z') z0 = sp.Symbol('z0') # 関数の定義 functions_sym = { 'e^z': sp.exp(z), 'sin(z)': sp.sin(z), 'cos(z)': sp.cos(z), '1/(1-z)': 1/(1-z), } print("=== Taylor級数展開 (Maclaurin展開, z0=0) ===\n") for name, f_sym in functions_sym.items(): print(f"f(z) = {name}") # Taylor展開（10次まで） taylor_series = sp.series(f_sym, z, 0, n=6).removeO() print(f"Taylor series: {taylor_series}") print() # 可視化省略（元のコード参照）

## 3.2 Laurent級数展開

特異点を含む領域では、負のべきを含むLaurent級数で展開されます。 

**📐 定義: Laurent級数展開**  
$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$ 正部（正則部）と負部（主要部）に分けると: $$f(z) = \underbrace{\sum_{n=0}^{\infty} a_n (z - z_0)^n}_{\text{正則部}} + \underbrace{\sum_{n=1}^{\infty} \frac{a_{-n}}{(z - z_0)^n}}_{\text{主要部}}$$ 

## 3.3 特異点の分類

特異点は除去可能特異点、極、真性特異点の3つに分類されます。 

**📐 定理: 特異点の分類**  

  * **除去可能特異点:** 主要部が0 → $\lim_{z \to z_0} f(z)$ が有限
  * **$m$ 位の極:** 主要部が有限項 $(z-z_0)^{-m}$ まで
  * **真性特異点:** 主要部が無限項

## 3.4 留数の計算

留数は Laurent展開の $(z-z_0)^{-1}$ の係数で、複素積分の計算に重要です。 

**📐 定義: 留数（Residue）**  
$$\text{Res}(f, z_0) = a_{-1}$$ ただし $a_{-1}$ はLaurent展開 $f(z) = \sum a_n (z-z_0)^n$ の $(z-z_0)^{-1}$ の係数  
  
**$m$ 位の極の場合:** $$\text{Res}(f, z_0) = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}} [(z-z_0)^m f(z)]$$ 

## 3.5 留数定理

留数定理により、複素積分が留数の和で計算できます。 

**📐 定理: 留数定理**  
$$\oint_C f(z) dz = 2\pi i \sum_{k} \text{Res}(f, z_k)$$ ただし $z_k$ は $C$ 内部の特異点 

## 3.6 実積分への応用 (1): 有理関数

留数定理を使うと、複雑な実積分を複素積分に変換して計算できます。 

**🔬 応用例:** 有理関数の実積分  
$$\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)} dx = 2\pi i \sum_{\text{上半平面}} \text{Res}(f, z_k)$$ ただし $\deg Q \geq \deg P + 2$ のとき収束 

## 3.7 実積分への応用 (2): 三角関数を含む積分

$z = e^{i\theta}$ の置換により、三角関数を含む積分を複素積分に変換できます。 

**📐 定義: 三角関数積分の変換**  
$$z = e^{i\theta}, \quad \cos\theta = \frac{z + z^{-1}}{2}, \quad \sin\theta = \frac{z - z^{-1}}{2i}$$ $$\int_0^{2\pi} R(\cos\theta, \sin\theta) d\theta = \oint_{|z|=1} R\left(\frac{z+z^{-1}}{2}, \frac{z-z^{-1}}{2i}\right) \frac{dz}{iz}$$ 

## 3.8 実積分への応用 (3): フーリエ型積分

$e^{iax}$ を含む積分にも留数定理が有効です。 

**📐 定理: フーリエ型積分**  
$$\int_{-\infty}^{\infty} f(x) e^{iax} dx = 2\pi i \sum_{\text{Im}(z_k)>0} \text{Res}(f(z)e^{iaz}, z_k) \quad (a > 0)$$ 

## 3.9 材料科学への応用: 格子振動とフォノン分散

固体物理学では、格子振動の分散関係を解析する際に複素関数論が使われます。 

**📝 物理的意義:**

  * グリーン関数の極 → 格子振動モード（フォノン）
  * スペクトル関数 → 状態密度
  * 複素周波数 → 振動の減衰

## 📝 章末問題

**✏️ 演習問題**

  1. $f(z) = \frac{e^z}{z^3}$ の $z=0$ 周りのLaurent展開を求めよ。
  2. $f(z) = \frac{1}{z(z-1)(z-2)}$ の留数を全ての特異点で計算せよ。
  3. 留数定理を使って $\int_{-\infty}^{\infty} \frac{dx}{1+x^4}$ を計算せよ。
  4. $\int_0^{2\pi} \frac{d\theta}{3 + 2\cos\theta}$ を留数定理で計算せよ。

## まとめ

  * Laurent級数は特異点近傍での関数表現を提供
  * 留数定理により複雑な実積分が計算可能
  * 物理学（量子力学、統計力学）での応用が広範
  * 数値計算でも留数の理解が重要

[← 第2章: 複素積分](<chapter-2.html>) 第4章: フーリエ変換 →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
