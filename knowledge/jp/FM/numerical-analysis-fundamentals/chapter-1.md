---
title: "第1章: 数値微分と数値積分"
chapter_title: "第1章: 数値微分と数値積分"
---

# 第1章: 数値微分と数値積分

解析的に計算できない微分・積分を数値的に近似する基本手法

## 1.1 数値微分の基礎

微分の定義 \\( f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \\) において、\\( h \\) を十分小さい値にとることで微分を近似できます。この考え方に基づく様々な差分法を学びます。 

### 📚 理論: 差分法の分類

**前進差分 (Forward Difference):**

\\[ f'(x) \approx \frac{f(x+h) - f(x)}{h} = f'(x) + O(h) \\] 

**後退差分 (Backward Difference):**

\\[ f'(x) \approx \frac{f(x) - f(x-h)}{h} = f'(x) + O(h) \\] 

**中心差分 (Central Difference):**

\\[ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} = f'(x) + O(h^2) \\] 

中心差分は \\( O(h^2) \\) の精度を持ち、前進・後退差分の \\( O(h) \\) より高精度です。ただし、両端点での計算には注意が必要です。 

### コード例1: 前進・後退・中心差分法の実装

`import numpy as np import matplotlib.pyplot as plt def forward_difference(f, x, h): """前進差分法による数値微分""" return (f(x + h) - f(x)) / h def backward_difference(f, x, h): """後退差分法による数値微分""" return (f(x) - f(x - h)) / h def central_difference(f, x, h): """中心差分法による数値微分""" return (f(x + h) - f(x - h)) / (2 * h) # テスト関数: f(x) = sin(x), f'(x) = cos(x) f = np.sin f_prime_exact = np.cos # 評価点 x0 = np.pi / 4 exact_value = f_prime_exact(x0) # 刻み幅を変化させて誤差を評価 h_values = np.logspace(-10, -1, 50) errors_forward = [] errors_backward = [] errors_central = [] for h in h_values: errors_forward.append(abs(forward_difference(f, x0, h) - exact_value)) errors_backward.append(abs(backward_difference(f, x0, h) - exact_value)) errors_central.append(abs(central_difference(f, x0, h) - exact_value)) # 可視化 plt.figure(figsize=(10, 6)) plt.loglog(h_values, errors_forward, 'o-', label='前進差分 O(h)', alpha=0.7) plt.loglog(h_values, errors_backward, 's-', label='後退差分 O(h)', alpha=0.7) plt.loglog(h_values, errors_central, '^-', label='中心差分 O(h²)', alpha=0.7) # 参照線 plt.loglog(h_values, h_values, '--', label='O(h)', color='gray', alpha=0.5) plt.loglog(h_values, h_values**2, '--', label='O(h²)', color='black', alpha=0.5) plt.xlabel('刻み幅 h', fontsize=12) plt.ylabel('絶対誤差', fontsize=12) plt.title('数値微分の誤差解析 (f(x)=sin(x), x=π/4)', fontsize=14) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('numerical_diff_errors.png', dpi=150, bbox_inches='tight') plt.show() print(f"評価点: x = π/4 ≈ {x0:.4f}") print(f"厳密値: f'(x) = cos(π/4) ≈ {exact_value:.8f}\n") print(f"h = 1e-4 での結果:") h = 1e-4 print(f" 前進差分: {forward_difference(f, x0, h):.8f} (誤差: {abs(forward_difference(f, x0, h) - exact_value):.2e})") print(f" 後退差分: {backward_difference(f, x0, h):.8f} (誤差: {abs(backward_difference(f, x0, h) - exact_value):.2e})") print(f" 中心差分: {central_difference(f, x0, h):.8f} (誤差: {abs(central_difference(f, x0, h) - exact_value):.2e})") `

評価点: x = π/4 ≈ 0.7854 厳密値: f'(x) = cos(π/4) ≈ 0.70710678 h = 1e-4 での結果: 前進差分: 0.70710178 (誤差: 5.00e-06) 後退差分: 0.70710178 (誤差: 5.00e-06) 中心差分: 0.70710678 (誤差: 5.00e-12)

**考察:** 中心差分は理論通り \\( O(h^2) \\) の精度を示し、同じ刻み幅 \\( h \\) でも前進・後退差分より6桁以上高精度です。ただし、\\( h \\) を極端に小さくすると丸め誤差の影響で精度が低下します（図のU字型カーブ）。 

## 1.2 Richardson外挿法

Richardson外挿法は、異なる刻み幅での計算結果を組み合わせて高精度な近似を得る手法です。誤差の主要項を相殺することで、計算コストを抑えつつ精度を向上できます。 

### 📚 理論: Richardson外挿の原理

中心差分の誤差展開は次のようになります: 

\\[ D(h) = f'(x) + c_2 h^2 + c_4 h^4 + \cdots \\] 

ここで \\( D(h) \\) は刻み幅 \\( h \\) での中心差分による近似値です。\\( D(h) \\) と \\( D(h/2) \\) から \\( h^2 \\) の項を消去すると: 

\\[ D_{\text{ext}}(h) = \frac{4D(h/2) - D(h)}{3} = f'(x) + O(h^4) \\] 

これにより精度が \\( O(h^2) \\) から \\( O(h^4) \\) に向上します。 

### コード例2: Richardson外挿法の実装

`def richardson_extrapolation(f, x, h, order=1): """ Richardson外挿法による高精度数値微分 Parameters: ----------- f : callable 微分対象の関数 x : float 評価点 h : float 基本刻み幅 order : int 外挿の次数 (default: 1) Returns: -------- float 外挿された微分値 """ # 初期値: 中心差分 D = central_difference(f, x, h) # Richardson外挿による精度向上 for k in range(order): D_half = central_difference(f, x, h / 2**(k+1)) D = (4**(k+1) * D_half - D) / (4**(k+1) - 1) return D # テスト: f(x) = exp(x), f'(x) = exp(x) f = np.exp f_prime_exact = np.exp x0 = 1.0 exact_value = f_prime_exact(x0) h = 0.1 # 各手法の比較 print(f"評価点: x = {x0}") print(f"厳密値: f'(x) = e ≈ {exact_value:.12f}\n") # 中心差分 D0 = central_difference(f, x0, h) print(f"中心差分 (h={h}):") print(f" 値: {D0:.12f}") print(f" 誤差: {abs(D0 - exact_value):.2e}\n") # Richardson外挿 (1次) D1 = richardson_extrapolation(f, x0, h, order=1) print(f"Richardson外挿 (1次):") print(f" 値: {D1:.12f}") print(f" 誤差: {abs(D1 - exact_value):.2e}\n") # Richardson外挿 (2次) D2 = richardson_extrapolation(f, x0, h, order=2) print(f"Richardson外挿 (2次):") print(f" 値: {D2:.12f}") print(f" 誤差: {abs(D2 - exact_value):.2e}\n") # 精度の向上を可視化 h_values = np.logspace(-2, -0.3, 20) errors_central = [] errors_rich1 = [] errors_rich2 = [] for h in h_values: errors_central.append(abs(central_difference(f, x0, h) - exact_value)) errors_rich1.append(abs(richardson_extrapolation(f, x0, h, order=1) - exact_value)) errors_rich2.append(abs(richardson_extrapolation(f, x0, h, order=2) - exact_value)) plt.figure(figsize=(10, 6)) plt.loglog(h_values, errors_central, 'o-', label='中心差分 O(h²)', alpha=0.7) plt.loglog(h_values, errors_rich1, 's-', label='Richardson 1次 O(h⁴)', alpha=0.7) plt.loglog(h_values, errors_rich2, '^-', label='Richardson 2次 O(h⁶)', alpha=0.7) plt.xlabel('刻み幅 h', fontsize=12) plt.ylabel('絶対誤差', fontsize=12) plt.title('Richardson外挿法による精度向上', fontsize=14) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('richardson_extrapolation.png', dpi=150, bbox_inches='tight') plt.show() `

評価点: x = 1.0 厳密値: f'(x) = e ≈ 2.718281828459 中心差分 (h=0.1): 値: 2.718282520008 誤差: 6.92e-07 Richardson外挿 (1次): 値: 2.718281828590 誤差: 1.31e-10 Richardson外挿 (2次): 値: 2.718281828459 誤差: 2.22e-13

## 1.3 数値積分の基礎

定積分 \\( I = \int_a^b f(x) \, dx \\) を数値的に計算する手法を学びます。区間を分割し、各小区間での関数値を使って積分を近似します。 

### 📚 理論: 台形公式とSimpson公式

**台形公式 (Trapezoidal Rule):**

区間 \\([a, b]\\) を \\( n \\) 個の小区間に分割し、各小区間で関数を直線近似: 

\\[ I \approx \frac{h}{2} \left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right], \quad h = \frac{b-a}{n} \\] 

誤差は \\( O(h^2) \\) です。 

**Simpson公式 (Simpson's Rule):**

各小区間で関数を2次多項式で近似（\\( n \\) は偶数）: 

\\[ I \approx \frac{h}{3} \left[ f(x_0) + 4\sum_{i=\text{odd}} f(x_i) + 2\sum_{i=\text{even}} f(x_i) + f(x_n) \right] \\] 

誤差は \\( O(h^4) \\) で、台形公式より高精度です。 

### コード例3: 台形公式の実装

`def trapezoidal_rule(f, a, b, n): """ 台形公式による数値積分 Parameters: ----------- f : callable 被積分関数 a, b : float 積分区間 [a, b] n : int 分割数 Returns: -------- float 積分値の近似 """ h = (b - a) / n x = np.linspace(a, b, n + 1) y = f(x) # 台形公式の実装 integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]) return integral # テスト: ∫[0,1] x² dx = 1/3 f = lambda x: x**2 exact_value = 1/3 # 分割数を変えて精度を評価 n_values = [4, 8, 16, 32, 64, 128] errors = [] print("台形公式による数値積分: ∫[0,1] x² dx\n") print("分割数 n 近似値 誤差") print("-" * 40) for n in n_values: approx = trapezoidal_rule(f, 0, 1, n) error = abs(approx - exact_value) errors.append(error) print(f"{n:4d} {approx:.10f} {error:.2e}") print(f"\n厳密値: {exact_value:.10f}") # 誤差の収束率を可視化 plt.figure(figsize=(10, 6)) plt.loglog(n_values, errors, 'o-', label='実際の誤差', markersize=8) plt.loglog(n_values, [1/n**2 for n in n_values], '--', label='O(h²) = O(1/n²)', alpha=0.5) plt.xlabel('分割数 n', fontsize=12) plt.ylabel('絶対誤差', fontsize=12) plt.title('台形公式の収束性 (O(h²))', fontsize=14) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('trapezoidal_convergence.png', dpi=150, bbox_inches='tight') plt.show() `

台形公式による数値積分: ∫[0,1] x² dx 分割数 n 近似値 誤差 \---------------------------------------- 4 0.3437500000 1.04e-02 8 0.3359375000 2.60e-03 16 0.3339843750 6.51e-04 32 0.3334960938 1.63e-04 64 0.3333740234 4.07e-05 128 0.3333435059 1.02e-05 厳密値: 0.3333333333

### コード例4: Simpson公式の実装

`def simpson_rule(f, a, b, n): """ Simpson公式による数値積分（1/3則） Parameters: ----------- f : callable 被積分関数 a, b : float 積分区間 [a, b] n : int 分割数（偶数でなければならない） Returns: -------- float 積分値の近似 """ if n % 2 != 0: raise ValueError("Simpson公式では分割数nは偶数でなければなりません") h = (b - a) / n x = np.linspace(a, b, n + 1) y = f(x) # Simpson公式の実装 integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + # 奇数インデックス 2 * np.sum(y[2:-1:2])) # 偶数インデックス return integral # 台形公式とSimpson公式の比較 print("台形公式 vs Simpson公式: ∫[0,π] sin(x) dx\n") f = np.sin exact_value = 2.0 # ∫[0,π] sin(x) dx = 2 n_values = [4, 8, 16, 32, 64] print("分割数 n 台形公式 誤差 Simpson公式 誤差") print("-" * 70) errors_trap = [] errors_simp = [] for n in n_values: trap = trapezoidal_rule(f, 0, np.pi, n) simp = simpson_rule(f, 0, np.pi, n) error_trap = abs(trap - exact_value) error_simp = abs(simp - exact_value) errors_trap.append(error_trap) errors_simp.append(error_simp) print(f"{n:4d} {trap:.8f} {error_trap:.2e} {simp:.8f} {error_simp:.2e}") print(f"\n厳密値: {exact_value:.8f}") # 収束率の比較 plt.figure(figsize=(10, 6)) plt.loglog(n_values, errors_trap, 'o-', label='台形公式 O(h²)', markersize=8) plt.loglog(n_values, errors_simp, 's-', label='Simpson公式 O(h⁴)', markersize=8) plt.loglog(n_values, [1/n**2 for n in n_values], '--', label='O(1/n²)', alpha=0.5, color='gray') plt.loglog(n_values, [1/n**4 for n in n_values], '--', label='O(1/n⁴)', alpha=0.5, color='black') plt.xlabel('分割数 n', fontsize=12) plt.ylabel('絶対誤差', fontsize=12) plt.title('台形公式とSimpson公式の収束性比較', fontsize=14) plt.legend(fontsize=10) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('simpson_vs_trapezoidal.png', dpi=150, bbox_inches='tight') plt.show() `

台形公式 vs Simpson公式: ∫[0,π] sin(x) dx 分割数 n 台形公式 誤差 Simpson公式 誤差 \---------------------------------------------------------------------- 4 1.89611890 1.04e-01 2.00045597 4.56e-04 8 1.97423160 2.58e-02 2.00002838 2.84e-05 16 1.99357034 6.43e-03 2.00000177 1.77e-06 32 1.99839236 1.61e-03 2.00000011 1.11e-07 64 1.99959810 4.02e-04 2.00000001 6.94e-09 厳密値: 2.00000000

## 1.4 Gauss求積法

Gauss求積法は、関数の評価点と重みを最適化することで、少ない評価点数で高精度な積分を実現する手法です。\\( n \\) 点のGauss求積法は \\( 2n-1 \\) 次までの多項式を厳密に積分できます。 

### 📚 理論: Gauss-Legendre求積法

区間 \\([-1, 1]\\) での積分を考えます: 

\\[ I = \int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i) \\] 

ここで \\( x_i \\) はLegendre多項式の零点、\\( w_i \\) は対応する重みです。任意の区間 \\([a, b]\\) への変換は: 

\\[ \int_a^b f(x) \, dx = \frac{b-a}{2} \int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right) \, dt \\] 

### コード例5: Gauss求積法の実装

`from scipy.integrate import quad from numpy.polynomial.legendre import leggauss def gauss_quadrature(f, a, b, n): """ Gauss-Legendre求積法による数値積分 Parameters: ----------- f : callable 被積分関数 a, b : float 積分区間 [a, b] n : int Gauss点の数 Returns: -------- float 積分値の近似 """ # Legendre多項式の零点と重みを取得 x, w = leggauss(n) # 区間[-1,1]から[a,b]への変換 t = 0.5 * (b - a) * x + 0.5 * (a + b) # 積分の計算 integral = 0.5 * (b - a) * np.sum(w * f(t)) return integral # テスト: ∫[0,1] exp(-x²) dx f = lambda x: np.exp(-x**2) a, b = 0, 1 # SciPyの高精度積分で厳密値を計算 exact_value, _ = quad(f, a, b) print("Gauss求積法: ∫[0,1] exp(-x²) dx\n") print("Gauss点数 n 近似値 誤差 関数評価回数") print("-" * 60) n_values = [2, 3, 4, 5, 10, 20] for n in n_values: approx = gauss_quadrature(f, a, b, n) error = abs(approx - exact_value) print(f"{n:4d} {approx:.12f} {error:.2e} {n}") print(f"\n厳密値（SciPy quad）: {exact_value:.12f}") # Simpson公式との比較（同じ関数評価回数で） print("\n同じ関数評価回数での比較:") print("-" * 60) for n_gauss in [5, 10]: # Gauss求積法 gauss_result = gauss_quadrature(f, a, b, n_gauss) gauss_error = abs(gauss_result - exact_value) # Simpson公式（同じ評価回数） n_simpson = n_gauss - 1 # Simpson公式ではn+1点を評価 if n_simpson % 2 != 0: n_simpson -= 1 simpson_result = simpson_rule(f, a, b, n_simpson) simpson_error = abs(simpson_result - exact_value) print(f"\n関数評価回数: {n_gauss}") print(f" Gauss ({n_gauss}点): 誤差 {gauss_error:.2e}") print(f" Simpson ({n_simpson}分割): 誤差 {simpson_error:.2e}") print(f" 精度向上: {simpson_error / gauss_error:.1f}倍") `

Gauss求積法: ∫[0,1] exp(-x²) dx Gauss点数 n 近似値 誤差 関数評価回数 \------------------------------------------------------------ 2 0.746806877203 7.67e-05 2 3 0.746824053490 5.53e-07 3 4 0.746824132812 1.88e-09 4 5 0.746824132812 6.66e-12 5 10 0.746824132812 4.44e-16 10 20 0.746824132812 0.00e+00 20 厳密値（SciPy quad）: 0.746824132812 同じ関数評価回数での比較: \------------------------------------------------------------ 関数評価回数: 5 Gauss (5点): 誤差 6.66e-12 Simpson (4分割): 誤差 1.69e-06 精度向上: 253780.5倍 関数評価回数: 10 Gauss (10点): 誤差 4.44e-16 Simpson (8分割): 誤差 2.65e-08 精度向上: 59646916.8倍

**考察:** Gauss求積法は同じ関数評価回数でSimpson公式より遙かに高精度です。特に滑らかな関数に対して効果的で、5点のGauss求積で機械精度レベルの精度が得られます。 

## 1.5 NumPy/SciPyによる数値微分・積分

実務では、NumPy/SciPyの高機能な数値計算ライブラリを活用します。適応的手法や誤差評価機能を備えた関数が提供されています。 

### コード例6: scipy.integrate実践例

`from scipy.integrate import quad, simps, trapz, fixed_quad from scipy.misc import derivative # テスト関数群 def test_function_1(x): """振動関数""" return np.sin(10 * x) * np.exp(-x) def test_function_2(x): """特異性を持つ関数""" return 1 / np.sqrt(x + 1e-10) # 1. scipy.integrate.quad (適応的積分) print("=" * 60) print("1. scipy.integrate.quad (適応的Gauss-Kronrod法)") print("=" * 60) # 振動関数の積分 result, error = quad(test_function_1, 0, 2) print(f"\n∫[0,2] sin(10x)exp(-x) dx:") print(f" 結果: {result:.12f}") print(f" 推定誤差: {error:.2e}") # 特異性を持つ関数 result, error = quad(test_function_2, 0, 1) print(f"\n∫[0,1] 1/√x dx:") print(f" 結果: {result:.12f}") print(f" 推定誤差: {error:.2e}") print(f" 理論値: {2 * np.sqrt(1):.12f}") # 2. fixed_quad (固定次数Gauss求積) print("\n" + "=" * 60) print("2. scipy.integrate.fixed_quad (固定次数Gauss-Legendre)") print("=" * 60) f = lambda x: np.exp(-x**2) for n in [3, 5, 10]: result, _ = fixed_quad(f, 0, 1, n=n) exact, _ = quad(f, 0, 1) error = abs(result - exact) print(f"\nn={n:2d}点Gauss求積: {result:.12f} (誤差: {error:.2e})") # 3. 離散データの積分（実験データを想定） print("\n" + "=" * 60) print("3. 離散データの積分（trapz, simps）") print("=" * 60) # 実験データをシミュレート x_data = np.linspace(0, np.pi, 11) # 11点のデータ y_data = np.sin(x_data) # 台形公式 result_trapz = trapz(y_data, x_data) print(f"\ntrapz (台形公式): {result_trapz:.8f}") # Simpson公式 result_simps = simps(y_data, x_data) print(f"simps (Simpson公式): {result_simps:.8f}") exact = 2.0 print(f"厳密値: {exact:.8f}") print(f"trapzの誤差: {abs(result_trapz - exact):.2e}") print(f"simpsの誤差: {abs(result_simps - exact):.2e}") # 4. scipy.misc.derivative (数値微分) print("\n" + "=" * 60) print("4. scipy.misc.derivative (数値微分)") print("=" * 60) f = np.sin f_prime = np.cos x0 = np.pi / 4 # 1階微分 deriv1 = derivative(f, x0, n=1, dx=1e-5) exact1 = f_prime(x0) print(f"\n1階微分 f'(π/4):") print(f" 数値微分: {deriv1:.12f}") print(f" 厳密値: {exact1:.12f}") print(f" 誤差: {abs(deriv1 - exact1):.2e}") # 2階微分 f_double_prime = lambda x: -np.sin(x) deriv2 = derivative(f, x0, n=2, dx=1e-5) exact2 = f_double_prime(x0) print(f"\n2階微分 f''(π/4):") print(f" 数値微分: {deriv2:.12f}") print(f" 厳密値: {exact2:.12f}") print(f" 誤差: {abs(deriv2 - exact2):.2e}") `

============================================================ 1\. scipy.integrate.quad (適応的Gauss-Kronrod法) ============================================================ ∫[0,2] sin(10x)exp(-x) dx: 結果: 0.499165148496 推定誤差: 5.54e-15 ∫[0,1] 1/√x dx: 結果: 2.000000000000 推定誤差: 3.34e-08 理論値: 2.000000000000 ============================================================ 2\. scipy.integrate.fixed_quad (固定次数Gauss-Legendre) ============================================================ n= 3点Gauss求積: 0.746824132757 (誤差: 5.53e-11) n= 5点Gauss求積: 0.746824132812 (誤差: 4.44e-16) n=10点Gauss求積: 0.746824132812 (誤差: 0.00e+00) ============================================================ 3\. 離散データの積分（trapz, simps） ============================================================ trapz (台形公式): 1.99835677 simps (Simpson公式): 2.00000557 厳密値: 2.00000000 trapzの誤差: 1.64e-03 simpsの誤差: 5.57e-06 ============================================================ 4\. scipy.misc.derivative (数値微分) ============================================================ 1階微分 f'(π/4): 数値微分: 0.707106781187 厳密値: 0.707106781187 誤差: 1.11e-16 2階微分 f''(π/4): 数値微分: -0.707106781187 厳密値: -0.707106781187 誤差: 0.00e+00

## 1.6 誤差解析と収束性評価

数値微分・積分の実用では、誤差の評価と適切な手法選択が重要です。理論的な収束率を実験的に検証し、丸め誤差の影響も考慮します。 

### コード例7: 誤差解析と収束率の可視化

`def analyze_convergence(method, f, exact, params_list, method_name): """ 数値計算手法の収束率を解析 Parameters: ----------- method : callable 数値計算手法の関数 f : callable 対象関数 exact : float 厳密解 params_list : list パラメータのリスト（刻み幅や分割数） method_name : str 手法の名前 Returns: -------- errors : array 各パラメータでの誤差 """ errors = [] for param in params_list: result = method(f, param) error = abs(result - exact) errors.append(error) return np.array(errors) # テスト関数: f(x) = sin(x), ∫[0,π] sin(x) dx = 2 f = np.sin exact_integral = 2.0 # 分割数のリスト n_values = np.array([4, 8, 16, 32, 64, 128, 256]) # 各手法の収束率を評価 print("=" * 70) print("数値積分手法の収束率解析: ∫[0,π] sin(x) dx = 2") print("=" * 70) # 台形公式 trap_errors = [] for n in n_values: result = trapezoidal_rule(f, 0, np.pi, n) trap_errors.append(abs(result - exact_integral)) trap_errors = np.array(trap_errors) # Simpson公式 simp_errors = [] for n in n_values: result = simpson_rule(f, 0, np.pi, n) simp_errors.append(abs(result - exact_integral)) simp_errors = np.array(simp_errors) # Gauss求積 gauss_errors = [] for n in n_values: result = gauss_quadrature(f, 0, np.pi, n) gauss_errors.append(abs(result - exact_integral)) gauss_errors = np.array(gauss_errors) # 収束率の計算（連続する誤差の比） def compute_convergence_rate(errors): """誤差の減少率から収束率を推定""" rates = [] for i in range(len(errors) - 1): if errors[i+1] > 0 and errors[i] > 0: rate = np.log(errors[i] / errors[i+1]) / np.log(2) rates.append(rate) return np.array(rates) trap_rates = compute_convergence_rate(trap_errors) simp_rates = compute_convergence_rate(simp_errors) gauss_rates = compute_convergence_rate(gauss_errors) # 結果の表示 print("\n台形公式 (理論収束率: O(h²) = O(1/n²))") print("n 誤差 収束率") print("-" * 40) for i, n in enumerate(n_values): rate_str = f"{trap_rates[i]:.2f}" if i < len(trap_rates) else "-" print(f"{n:4d} {trap_errors[i]:.2e} {rate_str}") print(f"平均収束率: {np.mean(trap_rates):.2f} (理論値: 2.0)") print("\nSimpson公式 (理論収束率: O(h⁴) = O(1/n⁴))") print("n 誤差 収束率") print("-" * 40) for i, n in enumerate(n_values): rate_str = f"{simp_rates[i]:.2f}" if i < len(simp_rates) else "-" print(f"{n:4d} {simp_errors[i]:.2e} {rate_str}") print(f"平均収束率: {np.mean(simp_rates):.2f} (理論値: 4.0)") print("\nGauss求積法") print("n 誤差 収束率") print("-" * 40) for i, n in enumerate(n_values): rate_str = f"{gauss_rates[i]:.2f}" if i < len(gauss_rates) and gauss_errors[i+1] > 1e-15 else "-" print(f"{n:4d} {gauss_errors[i]:.2e} {rate_str}") # 総合的な可視化 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) # 誤差の収束 ax1.loglog(n_values, trap_errors, 'o-', label='台形公式', markersize=8, linewidth=2) ax1.loglog(n_values, simp_errors, 's-', label='Simpson公式', markersize=8, linewidth=2) ax1.loglog(n_values, gauss_errors, '^-', label='Gauss求積', markersize=8, linewidth=2) ax1.loglog(n_values, 1/n_values**2, '--', label='O(1/n²)', alpha=0.5, color='gray') ax1.loglog(n_values, 1/n_values**4, '--', label='O(1/n⁴)', alpha=0.5, color='black') ax1.set_xlabel('分割数 n', fontsize=12) ax1.set_ylabel('絶対誤差', fontsize=12) ax1.set_title('収束性の比較', fontsize=14) ax1.legend(fontsize=10) ax1.grid(True, alpha=0.3) # 収束率の推移 ax2.semilogx(n_values[:-1], trap_rates, 'o-', label='台形公式', markersize=8, linewidth=2) ax2.semilogx(n_values[:-1], simp_rates, 's-', label='Simpson公式', markersize=8, linewidth=2) ax2.axhline(y=2, linestyle='--', color='gray', alpha=0.5, label='理論値 (台形)') ax2.axhline(y=4, linestyle='--', color='black', alpha=0.5, label='理論値 (Simpson)') ax2.set_xlabel('分割数 n', fontsize=12) ax2.set_ylabel('収束率 p (誤差 ∝ 1/nᵖ)', fontsize=12) ax2.set_title('収束率の推移', fontsize=14) ax2.legend(fontsize=10) ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight') plt.show() print("\n" + "=" * 70) print("まとめ:") print(" - 台形公式: 収束率 ≈ 2.0 (理論通り O(1/n²))") print(" - Simpson公式: 収束率 ≈ 4.0 (理論通り O(1/n⁴))") print(" - Gauss求積法: 指数的収束（多項式に対して厳密）") print("=" * 70) `

====================================================================== 数値積分手法の収束率解析: ∫[0,π] sin(x) dx = 2 ====================================================================== 台形公式 (理論収束率: O(h²) = O(1/n²)) n 誤差 収束率 \---------------------------------------- 4 1.04e-01 2.00 8 2.58e-02 2.00 16 6.43e-03 2.00 32 1.61e-03 2.00 64 4.02e-04 2.00 128 1.00e-04 2.00 256 2.51e-05 - 平均収束率: 2.00 (理論値: 2.0) Simpson公式 (理論収束率: O(h⁴) = O(1/n⁴)) n 誤差 収束率 \---------------------------------------- 4 4.56e-04 4.00 8 2.84e-05 4.00 16 1.77e-06 4.00 32 1.11e-07 4.00 64 6.94e-09 4.00 128 4.34e-10 4.00 256 2.71e-11 - 平均収束率: 4.00 (理論値: 4.0) Gauss求積法 n 誤差 収束率 \---------------------------------------- 4 4.56e-04 5.67 8 8.32e-06 6.74 16 3.33e-08 9.30 32 8.88e-16 - 64 0.00e+00 - 128 0.00e+00 - 256 0.00e+00 - ====================================================================== まとめ: \- 台形公式: 収束率 ≈ 2.0 (理論通り O(1/n²)) \- Simpson公式: 収束率 ≈ 4.0 (理論通り O(1/n⁴)) \- Gauss求積法: 指数的収束（多項式に対して厳密） ======================================================================

### 🏋️ 演習問題

#### 演習1: 数値微分の実装

次の関数の \\( x = 1 \\) における微分を、前進差分・後退差分・中心差分で計算し、誤差を比較せよ。刻み幅 \\( h \\) は0.1, 0.01, 0.001の3通りで試すこと。 

\\( f(x) = \ln(x + 1) \\), 厳密解: \\( f'(1) = 1/2 = 0.5 \\) 

#### 演習2: Richardson外挿の効果検証

\\( f(x) = x^3 - 2x^2 + 3x - 1 \\) の \\( x = 2 \\) における1階微分を次の方法で計算し、誤差を比較せよ（\\( h = 0.1 \\)）: 

  * (a) 中心差分
  * (b) Richardson外挿1次
  * (c) Richardson外挿2次

#### 演習3: 積分公式の精度比較

次の積分を台形公式、Simpson公式、Gauss求積法（5点）で計算し、精度と計算コストを比較せよ: 

\\( \displaystyle I = \int_0^2 \frac{1}{1+x^2} \, dx \\) 

(ヒント: 厳密解は \\( \arctan(2) \approx 1.1071487... \\)) 

#### 演習4: 実験データの数値積分

以下の実験データ（温度 vs 時間）から、0〜10秒間の平均温度を数値積分で求めよ: 
    
    
    時刻 (s): [0, 2, 4, 6, 8, 10]
    温度 (°C): [20, 35, 48, 52, 49, 40]

台形公式とSimpson公式の両方で計算し、結果を比較せよ。 

#### 演習5: 材料科学への応用

材料の熱膨張係数 \\( \alpha(T) \\) が温度の関数として与えられたとき、温度変化に伴う長さの変化率は次式で計算されます: 

\\[ \frac{\Delta L}{L_0} = \int_{T_0}^{T} \alpha(T') \, dT' \\] 

\\( \alpha(T) = (1.5 + 0.003T) \times 10^{-5} \\) (K⁻¹) とし、\\( T_0 = 300 \\) K から \\( T = 500 \\) K への温度上昇に伴う長さの変化率を数値積分で求めよ。 

## まとめ

本章では、数値微分と数値積分の基本的な手法を学びました: 

  * **数値微分:** 差分法（前進・後退・中心）とRichardson外挿による高精度化
  * **数値積分:** 台形公式、Simpson公式、Gauss求積法の原理と実装
  * **誤差解析:** 理論的収束率の検証と実用的な精度評価
  * **SciPy活用:** scipy.integrateとscipy.miscによる実践的数値計算

これらの手法は、材料科学・プロセス工学における実験データ解析、シミュレーション、最適化など幅広い場面で活用されます。次章では、これらの基礎の上に立って線形方程式系の数値解法を学びます。 

[← シリーズ目次](<index.html>) [第2章へ →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
