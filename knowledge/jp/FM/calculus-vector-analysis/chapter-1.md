---
title: "第1章: 微分の基礎と数値微分"
chapter_title: "第1章: 微分の基礎と数値微分"
subtitle: Fundamentals of Differentiation and Numerical Differentiation
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/calculus-vector-analysis/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [微積分とベクトル解析入門](<index.html>) > 第1章 

## 1.1 微分の定義と導関数

**📐 定義: 微分**  
関数 \\(f(x)\\) の \\(x = a\\) における微分係数は、極限値 \\[ f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h} \\] で定義されます。この値は、点 \\((a, f(a))\\) における接線の傾きを表します。 

微分は、関数の「瞬間的な変化率」を表す概念です。材料科学では、温度に対する物性値の変化率、 プロセス工学では反応速度、機械学習では損失関数の勾配として頻繁に登場します。 

### 💻 コード例1: 微分係数の数値計算（前進差分法）

Python実装: 前進差分法による数値微分

import numpy as np import matplotlib.pyplot as plt # 関数の定義: f(x) = x^2 def f(x): return x**2 # 前進差分法による微分係数の近似 def forward_difference(f, x, h=1e-5): """前進差分法: f'(x) ≈ [f(x+h) - f(x)] / h""" return (f(x + h) - f(x)) / h # x = 2 における微分係数を計算 x0 = 2.0 numerical_derivative = forward_difference(f, x0) analytical_derivative = 2 * x0 # 解析解: f'(x) = 2x print(f"数値微分: f'({x0}) ≈ {numerical_derivative:.6f}") print(f"解析解: f'({x0}) = {analytical_derivative:.6f}") print(f"誤差: {abs(numerical_derivative - analytical_derivative):.2e}") # 可視化 x = np.linspace(0, 4, 100) y = f(x) tangent_y = analytical_derivative * (x - x0) + f(x0) plt.figure(figsize=(8, 6)) plt.plot(x, y, label='f(x) = x²', linewidth=2) plt.plot(x, tangent_y, '--', label=f"接線 (傾き={analytical_derivative})", linewidth=2) plt.scatter([x0], [f(x0)], color='red', s=100, zorder=5) plt.xlabel('x', fontsize=12) plt.ylabel('f(x)', fontsize=12) plt.title('微分係数と接線', fontsize=14) plt.legend() plt.grid(True, alpha=0.3) plt.show() 

数値微分: f'(2.0) ≈ 4.000010 解析解: f'(2.0) = 4.000000 誤差: 1.00e-05 

## 1.2 微分の計算法則

**📐 定理: 微分の基本公式**  

  * \\((c)' = 0\\) （定数関数）
  * \\((x^n)' = nx^{n-1}\\) （べき関数）
  * \\((e^x)' = e^x\\) （指数関数）
  * \\((\ln x)' = \frac{1}{x}\\) （対数関数）
  * \\((\sin x)' = \cos x, (\cos x)' = -\sin x\\) （三角関数）
  * \\((cf)' = cf'\\) （定数倍）
  * \\((f + g)' = f' + g'\\) （和の微分）
  * \\((fg)' = f'g + fg'\\) （積の微分）
  * \\(\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}\\) （商の微分）

### 💻 コード例2: SymPyによる記号微分

Python実装: SymPyによる記号微分

import sympy as sp # 記号変数の定義 x = sp.Symbol('x') # 様々な関数の微分 functions = [ x**3, sp.exp(x), sp.ln(x), sp.sin(x), x**2 * sp.exp(x), sp.sin(x) / x ] print("記号微分の例:") for func in functions: derivative = sp.diff(func, x) print(f"d/dx({func}) = {derivative}") 

記号微分の例: d/dx(x**3) = 3*x**2 d/dx(exp(x)) = exp(x) d/dx(log(x)) = 1/x d/dx(sin(x)) = cos(x) d/dx(x**2*exp(x)) = x**2*exp(x) + 2*x*exp(x) d/dx(sin(x)/x) = -sin(x)/x**2 + cos(x)/x 

## 1.3 数値微分法の比較

実際のデータ解析では、関数形が不明な場合が多く、数値微分が必要になります。 代表的な数値微分法には、前進差分法、後退差分法、中心差分法があります。 

### 💻 コード例3: 前進・後退・中心差分法の比較

Python実装: 数値微分法の精度比較

def forward_diff(f, x, h): """前進差分法: O(h)""" return (f(x + h) - f(x)) / h def backward_diff(f, x, h): """後退差分法: O(h)""" return (f(x) - f(x - h)) / h def central_diff(f, x, h): """中心差分法: O(h²) - より精度が高い""" return (f(x + h) - f(x - h)) / (2 * h) # テスト関数: f(x) = sin(x), f'(x) = cos(x) f = np.sin f_prime_exact = np.cos x0 = np.pi / 4 # 45度 exact = f_prime_exact(x0) # 様々な刻み幅で誤差を評価 h_values = np.logspace(-10, -1, 50) errors_forward = [] errors_backward = [] errors_central = [] for h in h_values: errors_forward.append(abs(forward_diff(f, x0, h) - exact)) errors_backward.append(abs(backward_diff(f, x0, h) - exact)) errors_central.append(abs(central_diff(f, x0, h) - exact)) # 可視化 plt.figure(figsize=(10, 6)) plt.loglog(h_values, errors_forward, label='前進差分法', marker='o', markersize=3) plt.loglog(h_values, errors_backward, label='後退差分法', marker='s', markersize=3) plt.loglog(h_values, errors_central, label='中心差分法', marker='^', markersize=3) plt.loglog(h_values, h_values, '--', label='O(h)', alpha=0.5) plt.loglog(h_values, h_values**2, '--', label='O(h²)', alpha=0.5) plt.xlabel('刻み幅 h', fontsize=12) plt.ylabel('絶対誤差', fontsize=12) plt.title('数値微分法の精度比較', fontsize=14) plt.legend() plt.grid(True, alpha=0.3) plt.show() print(f"解析解: cos(π/4) = {exact:.10f}") print(f"前進差分 (h=1e-5): {forward_diff(f, x0, 1e-5):.10f}") print(f"中心差分 (h=1e-5): {central_diff(f, x0, 1e-5):.10f}")

**📝 注意:** 中心差分法は前進・後退差分法よりも高精度（O(h²)）ですが、 計算量は2倍必要です。実用上は、精度と計算コストのバランスを考慮して選択します。 

## 1.4 高階導関数

導関数をさらに微分したものを高階導関数と呼びます。第2次導関数 f''(x) は関数の凸凹性を、 第3次以降の導関数は関数の細かい形状を特徴づけます。 

### 💻 コード例4: 高階導関数の数値計算

`def second_derivative(f, x, h=1e-5): """第2次導関数: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²""" return (f(x + h) - 2*f(x) + f(x - h)) / h**2 def third_derivative(f, x, h=1e-4): """第3次導関数 (中心差分)""" return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3) # テスト関数: f(x) = x^4 f = lambda x: x**4 x0 = 2.0 # 解析解と数値解の比較 print("f(x) = x^4 の高階導関数 (x=2):") print(f"f'(x) = 4x³ → f'(2) = {4 * x0**3:.1f} (解析解)") print(f"f'(x) → f'(2) ≈ {central_diff(f, x0, 1e-5):.6f} (数値)") print(f"f''(x) = 12x² → f''(2) = {12 * x0**2:.1f} (解析解)") print(f"f''(x) → f''(2) ≈ {second_derivative(f, x0):.6f} (数値)") print(f"f'''(x) = 24x → f'''(2) = {24 * x0:.1f} (解析解)") print(f"f'''(x) → f'''(2) ≈ {third_derivative(f, x0):.6f} (数値)")`

## 1.5 材料科学への応用: 熱膨張係数

**🔬 応用例:** 材料の熱膨張係数 α は、長さ L の温度 T による変化率として定義されます： \\[\alpha = \frac{1}{L}\frac{dL}{dT}\\] 実測データから数値微分で熱膨張係数を求めます。 

### 💻 コード例5: 熱膨張係数の数値計算

`# 実験データ: 温度 T (K) vs 長さ L (mm) temperature = np.array([300, 350, 400, 450, 500, 550, 600]) length = np.array([100.000, 100.087, 100.175, 100.265, 100.357, 100.450, 100.545]) # スプライン補間で滑らかな関数を作成 from scipy.interpolate import UnivariateSpline spline = UnivariateSpline(temperature, length, s=0, k=3) # 微分して dL/dT を求める dL_dT = spline.derivative()(temperature) # 熱膨張係数 α = (1/L) * dL/dT alpha = dL_dT / length # 結果の表示 print("熱膨張係数の計算結果:") print("T (K)\tL (mm)\tdL/dT (mm/K)\tα (1/K)") for T, L, dLdT, a in zip(temperature, length, dL_dT, alpha): print(f"{T:.0f}\t{L:.3f}\t{dLdT:.6f}\t{a:.2e}") # 可視化 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) # 左図: 長さの温度依存性 T_fine = np.linspace(300, 600, 100) ax1.plot(temperature, length, 'o', label='実験データ', markersize=8) ax1.plot(T_fine, spline(T_fine), '-', label='スプライン補間', linewidth=2) ax1.set_xlabel('温度 T (K)', fontsize=12) ax1.set_ylabel('長さ L (mm)', fontsize=12) ax1.set_title('熱膨張曲線', fontsize=14) ax1.legend() ax1.grid(True, alpha=0.3) # 右図: 熱膨張係数の温度依存性 ax2.plot(temperature, alpha * 1e6, 'o-', linewidth=2, markersize=8) ax2.set_xlabel('温度 T (K)', fontsize=12) ax2.set_ylabel('熱膨張係数 α (10⁻⁶/K)', fontsize=12) ax2.set_title('熱膨張係数の温度依存性', fontsize=14) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.show()`

## 1.6 Richardson外挿法による高精度化

Richardson外挿法は、異なる刻み幅での数値微分結果を組み合わせて、 より高精度な近似値を得る手法です。 

### 💻 コード例6: Richardson外挿法

`def richardson_extrapolation(f, x, h, order=4): """Richardson外挿法による高精度数値微分""" # 異なる刻み幅での中心差分 D1 = central_diff(f, x, h) D2 = central_diff(f, x, h/2) # 1次外挿 (O(h⁴)の精度) D_improved = (4 * D2 - D1) / 3 return D_improved # テスト: f(x) = exp(x), f'(x) = exp(x) f = np.exp x0 = 1.0 exact = np.exp(x0) h = 0.1 D_central = central_diff(f, x0, h) D_richardson = richardson_extrapolation(f, x0, h) print(f"解析解: {exact:.10f}") print(f"中心差分 (h=0.1): {D_central:.10f}, 誤差 = {abs(D_central - exact):.2e}") print(f"Richardson外挿: {D_richardson:.10f}, 誤差 = {abs(D_richardson - exact):.2e}") print(f"精度向上: {abs(D_central - exact) / abs(D_richardson - exact):.1f}倍")`

## 1.7 練習問題

**✏️ 演習1:** 関数 f(x) = x³ - 3x² + 2x + 1 について、x = 2 における微分係数を (1) 解析的に、(2) 前進差分法で、(3) 中心差分法で求めよ。 

**✏️ 演習2:** プロセス変数 y(t) が時間 t に対して y(t) = 10 + 5sin(πt/10) で与えられる。 t = 5 における変化率 dy/dt を数値微分で求め、制御の必要性を判断せよ。 

### 💻 コード例7: 演習問題の解答例

`# 演習1の解答 x = sp.Symbol('x') f_sym = x**3 - 3*x**2 + 2*x + 1 f_prime_sym = sp.diff(f_sym, x) f_prime_at_2 = f_prime_sym.subs(x, 2) f_num = lambda x: x**3 - 3*x**2 + 2*x + 1 x0 = 2.0 print("演習1の解答:") print(f"(1) 解析解: f'(2) = {f_prime_at_2}") print(f"(2) 前進差分: f'(2) ≈ {forward_diff(f_num, x0, 1e-5):.6f}") print(f"(3) 中心差分: f'(2) ≈ {central_diff(f_num, x0, 1e-5):.6f}") # 演習2の解答 def y(t): return 10 + 5 * np.sin(np.pi * t / 10) t0 = 5.0 dy_dt = central_diff(y, t0, 0.01) print(f"\n演習2の解答:") print(f"t = 5 における dy/dt = {dy_dt:.4f}") print(f"解析解: dy/dt = (5π/10)cos(π·5/10) = {5*np.pi/10 * np.cos(np.pi*5/10):.4f}") if abs(dy_dt) > 0.5: print("→ 変化率が大きいため、制御介入が必要")`

## まとめ

  * 微分は関数の瞬間的な変化率を表し、接線の傾きとして幾何学的に解釈できる
  * 数値微分では中心差分法が前進・後退差分法より高精度（O(h²)）
  * Richardson外挿法により、さらに高精度な数値微分が可能
  * 材料科学では熱膨張係数、反応速度など様々な物性値の計算に微分が使われる
  * SymPyによる記号微分とNumPyによる数値微分を使い分けることが重要

[← シリーズトップ](<index.html>) [第2章: 積分の基礎 →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
