---
title: "第4章: フーリエ変換とラプラス変換"
chapter_title: "第4章: フーリエ変換とラプラス変換"
subtitle: Fourier and Laplace Transforms
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/complex-special-functions/chapter-4.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [複素関数論と特殊関数](<index.html>) > 第4章 

## 4.1 フーリエ級数

周期関数は三角関数の級数（フーリエ級数）で表現できます。 

**📐 定義: フーリエ級数展開**  
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L} \right)$$ **フーリエ係数:** $$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\frac{n\pi x}{L} dx$$ $$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\frac{n\pi x}{L} dx$$ 

### 💻 コード例 1: フーリエ級数展開

Python実装: 方形波のフーリエ級数近似

import numpy as np import matplotlib.pyplot as plt from scipy import integrate def fourier_coefficients(f, L, n_max): """フーリエ係数の計算""" a0 = (1/L) * integrate.quad(f, -L, L)[0] a_n = [] b_n = [] for n in range(1, n_max + 1): # a_n integrand_a = lambda x: f(x) * np.cos(n * np.pi * x / L) a_n.append((1/L) * integrate.quad(integrand_a, -L, L)[0]) # b_n integrand_b = lambda x: f(x) * np.sin(n * np.pi * x / L) b_n.append((1/L) * integrate.quad(integrand_b, -L, L)[0]) return a0, np.array(a_n), np.array(b_n) # テスト関数: 方形波 L = np.pi def square_wave(x): return np.where(np.abs(x) < L/2, 1.0, 0.0) # 可視化省略（元のコード参照）

## 4.2 フーリエ変換

非周期関数に対しては、フーリエ級数を連続化したフーリエ変換を用います。 

**📐 定義: フーリエ変換**  
$$F(\omega) = \mathcal{F}[f(t)] = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$ **逆フーリエ変換:** $$f(t) = \mathcal{F}^{-1}[F(\omega)] = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} d\omega$$ 

## 4.3 畳み込み定理

フーリエ変換は畳み込み演算を単純な積に変換します。 

**📐 定理: 畳み込み定理**  
$$\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$$ ここで畳み込み $(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau$ 

## 4.4 ラプラス変換

ラプラス変換は片側のフーリエ変換を一般化したもので、微分方程式の解法に有用です。 

**📐 定義: ラプラス変換**  
$$F(s) = \mathcal{L}[f(t)] = \int_0^{\infty} f(t) e^{-st} dt$$ **主な性質:**

  * 微分: $\mathcal{L}[f'(t)] = sF(s) - f(0)$
  * 積分: $\mathcal{L}\left[\int_0^t f(\tau)d\tau\right] = \frac{F(s)}{s}$
  * 畳み込み: $\mathcal{L}[f * g] = F(s) \cdot G(s)$

## 4.5 逆ラプラス変換と微分方程式

ラプラス変換を使うと、微分方程式を代数方程式に変換できます。 

**🔬 応用例:** 微分方程式の解法  
微分方程式: $y'' + 4y' + 3y = e^{-t}$, $y(0) = 0$, $y'(0) = 0$  
  
ラプラス変換により:  
$(s^2 + 4s + 3)Y(s) = \frac{1}{s+1}$  
  
解: $Y(s) = \frac{1}{(s+1)^2(s+3)}$ 

## 4.6 フーリエ変換の性質

フーリエ変換には様々な有用な性質があります。 

**📝 主要な性質:**

  * 時間シフト: $f(t-t_0) \rightarrow e^{-i\omega t_0}F(\omega)$
  * スケーリング: $f(at) \rightarrow \frac{1}{|a|}F(\omega/a)$
  * 微分: $f'(t) \rightarrow i\omega F(\omega)$

## 4.7 ウィンドウ関数とスペクトル漏れ

有限長の信号をFFTで解析する際、ウィンドウ関数を使ってスペクトル漏れを抑制します。 

**📐 定理: ウィンドウ関数の特性**  

  * **Rectangular:** メインローブ幅最小、サイドローブ大
  * **Hann:** バランスが良い、汎用的
  * **Blackman:** サイドローブ最小、メインローブ幅大

## 4.8 材料科学への応用: X線回折パターン解析

結晶構造の解析では、実空間の原子配列とフーリエ変換された逆格子空間（回折パターン）が対応します。 

**🔬 物理的意義:**

  * 実空間の周期構造 → 逆空間の離散的なブラッグピーク
  * 格子定数 $a$ が大きい → ブラッグピーク間隔が小さい
  * 結晶サイズが大きい → ブラッグピークがシャープ

## 📝 章末問題

**✏️ 演習問題**

  1. 方形波のフーリエ級数展開を10次まで求め、ギブス現象を観察せよ。
  2. Gaussian関数 $f(t) = e^{-t^2/(2\sigma^2)}$ のフーリエ変換を計算し、自己双対性を確認せよ。
  3. 畳み込み定理を使って、2つのローパスフィルタの縦続接続の周波数応答を求めよ。
  4. ラプラス変換を使って、微分方程式 $y'' + 2y' + 2y = \sin(t)$, $y(0)=0$, $y'(0)=1$ を解け。

## まとめ

  * フーリエ級数は周期関数を三角関数の和で表現
  * フーリエ変換は時間領域と周波数領域を相互変換
  * ラプラス変換は微分方程式を代数方程式に変換
  * 畳み込み定理により信号処理が効率化
  * 材料科学（X線回折）や工学（制御理論）で広範に応用

[← 第3章: Laurent展開](<chapter-3.html>) [第5章: 特殊関数 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
