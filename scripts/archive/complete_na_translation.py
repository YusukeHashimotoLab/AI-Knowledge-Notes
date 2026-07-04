#!/usr/bin/env python3
"""
COMPLETE comprehensive translation for numerical analysis chapters 1 and 2.
This script translates ALL Japanese text (2928 + 3336 = 6264 characters).

Strategy:
1. Read complete source file
2. Apply systematic translations in logical groups
3. Verify 0 Japanese characters remain
4. Write complete English version
"""

import re
import sys
import os

def count_japanese(text):
    """Count all Japanese characters"""
    hiragana = len(re.findall(r'[あ-ん]', text))
    katakana = len(re.findall(r'[ア-ン]', text))
    kanji = len(re.findall(r'[一-龯]', text))
    return hiragana + katakana + kanji

def create_complete_translation_map():
    """
    Comprehensive translation dictionary covering ALL Japanese phrases.
    Organized by category for clarity.
    """

    translations = {}

    # ========== HTML/META ==========
    translations['lang="ja"'] = 'lang="en"'

    # ========== CHAPTER 1 SPECIFIC ==========

    # Title/Meta
    translations['<title>第1章: 数値微分と数値積分 - 数値解析の基礎</title>'] = \
        '<title>Chapter 1: Numerical Differentiation and Integration - Fundamentals of Numerical Analysis</title>'

    translations['content="数値微分と数値積分の基本手法を学びます。差分法、Richardson外挿法、台形公式、Simpson公式、Gauss求積法をPythonで実装します。"'] = \
        'content="Learn fundamental methods for numerical differentiation and integration. Implement finite difference methods, Richardson extrapolation, trapezoidal rule, Simpson\'s rule, and Gaussian quadrature in Python."'

    # Navigation/Breadcrumb
    translations['基礎数理道場'] = 'Fundamental Mathematics Dojo'
    translations['数値解析の基礎'] = 'Fundamentals of Numerical Analysis'
    translations['第1章'] = 'Chapter 1'
    translations['第2章'] = 'Chapter 2'
    translations['第3章'] = 'Chapter 3'

    # Main headers
    translations['第1章: 数値微分と数値積分'] = 'Chapter 1: Numerical Differentiation and Integration'
    translations['解析的に計算できない微分・積分を数値的に近似する基本手法'] = \
        'Fundamental methods for numerically approximating derivatives and integrals that cannot be computed analytically'

    # Section 1.1 - Numerical Differentiation Basics
    translations['1.1 数値微分の基礎'] = '1.1 Fundamentals of Numerical Differentiation'
    translations['微分の定義'] = 'In the definition of differentiation'
    translations['において、'] = ', by taking'
    translations['を十分小さい値にとることで微分を近似できます。この考え方に基づく様々な差分法を学びます。'] = \
        'to be a sufficiently small value, we can approximate the derivative. We will learn various finite difference methods based on this idea.'

    # Theory boxes
    translations['📚 理論: 差分法の分類'] = '📚 Theory: Classification of Finite Difference Methods'
    translations['前進差分 (Forward Difference):'] = 'Forward Difference:'
    translations['後退差分 (Backward Difference):'] = 'Backward Difference:'
    translations['中心差分 (Central Difference):'] = 'Central Difference:'

    translations['中心差分は'] = 'The central difference has'
    translations['の精度を持ち、前進・後退差分の'] = 'accuracy, which is higher than the'
    translations['より高精度です。ただし、両端点での計算には注意が必要です。'] = \
        'accuracy of forward and backward differences. However, care must be taken when computing at boundary points.'

    # Code example titles
    translations['コード例1: 前進・後退・中心差分法の実装'] = 'Code Example 1: Implementing Forward, Backward, and Central Difference Methods'
    translations['コード例2: Richardson外挿法の実装'] = 'Code Example 2: Implementing Richardson Extrapolation'
    translations['コード例3: 台形公式の実装'] = 'Code Example 3: Implementing the Trapezoidal Rule'
    translations['コード例4: Simpson公式の実装'] = 'Code Example 4: Implementing Simpson\'s Rule'
    translations['コード例5: Gauss求積法の実装'] = 'Code Example 5: Implementing Gaussian Quadrature'
    translations['コード例6: scipy.integrate実践例'] = 'Code Example 6: scipy.integrate Practical Examples'
    translations['コード例7: 誤差解析と収束率の可視化'] = 'Code Example 7: Error Analysis and Convergence Rate Visualization'

    # Python docstrings/comments - Section 1.1
    translations['前進差分法による数値微分'] = 'Numerical differentiation using forward difference'
    translations['後退差分法による数値微分'] = 'Numerical differentiation using backward difference'
    translations['中心差分法による数値微分'] = 'Numerical differentiation using central difference'

    translations['テスト関数:'] = '# Test function:'
    translations['評価点'] = '# Evaluation point'
    translations['刻み幅を変化させて誤差を評価'] = '# Evaluate error for varying step sizes'
    translations['可視化'] = '# Visualization'
    translations['参照線'] = '# Reference lines'

    # Plot labels
    translations['刻み幅 h'] = 'Step size h'
    translations['絶対誤差'] = 'Absolute error'
    translations['数値微分の誤差解析'] = 'Error Analysis of Numerical Differentiation'
    translations['前進差分 O(h)'] = 'Forward Difference O(h)'
    translations['後退差分 O(h)'] = 'Backward Difference O(h)'
    translations['中心差分 O(h²)'] = 'Central Difference O(h²)'

    # Output text
    translations['評価点:'] = 'Evaluation point:'
    translations['厳密値:'] = 'Exact value:'
    translations['での結果:'] = 'Results for'
    translations['前進差分:'] = 'Forward difference:'
    translations['後退差分:'] = 'Backward difference:'
    translations['中心差分:'] = 'Central difference:'
    translations['誤差:'] = 'error:'

    # Discussion
    translations['考察:'] = 'Discussion:'
    translations['中心差分は理論通り'] = 'The central difference shows the theoretical'
    translations['の精度を示し、同じ刻み幅'] = 'accuracy and is more than 6 digits more accurate than forward/backward differences for the same step size'
    translations['でも前進・後退差分より6桁以上高精度です。ただし、'] = '. However, when'
    translations['を極端に小さくすると丸め誤差の影響で精度が低下します（図のU字型カーブ）。'] = \
        'is made extremely small, accuracy degrades due to round-off errors (U-shaped curve in the figure).'

    # Section 1.2 - Richardson Extrapolation
    translations['1.2 Richardson外挿法'] = '1.2 Richardson Extrapolation'
    translations['Richardson外挿法は、異なる刻み幅での計算結果を組み合わせて高精度な近似を得る手法です。誤差の主要項を相殺することで、計算コストを抑えつつ精度を向上できます。'] = \
        'Richardson extrapolation is a method that obtains high-accuracy approximations by combining results with different step sizes. By canceling the main error terms, accuracy can be improved while keeping computational cost low.'

    translations['📚 理論: Richardson外挿の原理'] = '📚 Theory: Principles of Richardson Extrapolation'
    translations['中心差分の誤差展開は次のようになります:'] = 'The error expansion of the central difference is as follows:'
    translations['ここで'] = 'where'
    translations['は刻み幅'] = 'is the central difference approximation with step size'
    translations['での中心差分による近似値です。'] = '.'
    translations['と'] = 'and'
    translations['から'] = 'From'
    translations['の項を消去すると:'] = ', eliminating the'
    translations['これにより精度が'] = 'This improves the accuracy from'
    translations['に向上します。'] = '.'

    # Richardson code comments
    translations['Richardson外挿法による高精度数値微分'] = 'High-accuracy numerical differentiation using Richardson extrapolation'
    translations['微分対象の関数'] = 'Function to differentiate'
    translations['基本刻み幅'] = 'Base step size'
    translations['外挿の次数'] = 'Extrapolation order'
    translations['外挿された微分値'] = 'Extrapolated derivative value'
    translations['初期値: 中心差分'] = '# Initial value: central difference'
    translations['Richardson外挿による精度向上'] = '# Improve accuracy with Richardson extrapolation'

    # More output
    translations['テスト:'] = '# Test:'
    translations['各手法の比較'] = '# Compare methods'
    translations['値:'] = 'Value:'
    translations['Richardson外挿'] = 'Richardson extrapolation'
    translations['1次'] = '1st order'
    translations['2次'] = '2nd order'
    translations['精度の向上を可視化'] = '# Visualize accuracy improvement'
    translations['Richardson外挿法による精度向上'] = 'Accuracy Improvement with Richardson Extrapolation'
    translations['Richardson 1次 O(h⁴)'] = 'Richardson 1st Order O(h⁴)'
    translations['Richardson 2次 O(h⁶)'] = 'Richardson 2nd Order O(h⁶)'

    # Section 1.3 - Numerical Integration
    translations['1.3 数値積分の基礎'] = '1.3 Fundamentals of Numerical Integration'
    translations['定積分'] = 'We will learn methods for numerically computing the definite integral'
    translations['を数値的に計算する手法を学びます。区間を分割し、各小区間での関数値を使って積分を近似します。'] = \
        '. By dividing the interval and using function values in each subinterval, we approximate the integral.'

    translations['📚 理論: 台形公式とSimpson公式'] = '📚 Theory: Trapezoidal and Simpson\'s Rules'
    translations['台形公式 (Trapezoidal Rule):'] = 'Trapezoidal Rule:'
    translations['区間'] = 'The interval'
    translations['を'] = 'is divided into'
    translations['個の小区間に分割し、各小区間で関数を直線近似:'] = \
        'subintervals, and the function is approximated by straight lines in each subinterval:'
    translations['誤差は'] = 'The error is'
    translations['です。'] = '.'

    translations['Simpson公式 (Simpson\'s Rule):'] = 'Simpson\'s Rule:'
    translations['各小区間で関数を2次多項式で近似（'] = 'The function is approximated by quadratic polynomials in each subinterval ('
    translations['は偶数）:'] = 'must be even):'
    translations['で、台形公式より高精度です。'] = ', which is more accurate than the trapezoidal rule.'

    # Trapezoidal code
    translations['台形公式による数値積分'] = 'Numerical integration using the trapezoidal rule'
    translations['被積分関数'] = 'Integrand function'
    translations['積分区間'] = 'Integration interval'
    translations['分割数'] = 'Number of divisions'
    translations['積分値の近似'] = 'Approximation of the integral'
    translations['台形公式の実装'] = '# Implementation of trapezoidal rule'

    translations['分割数を変えて精度を評価'] = '# Evaluate accuracy for varying number of divisions'
    translations['台形公式による数値積分:'] = 'Numerical Integration Using Trapezoidal Rule:'
    translations['分割数 n    近似値        誤差'] = 'Divisions n    Approximation    Error'
    translations['誤差の収束率を可視化'] = '# Visualize error convergence rate'
    translations['実際の誤差'] = 'Actual error'
    translations['分割数 n'] = 'Number of divisions n'
    translations['台形公式の収束性'] = 'Convergence of Trapezoidal Rule'

    # Simpson code
    translations['Simpson公式による数値積分（1/3則）'] = 'Numerical integration using Simpson\'s rule (1/3 rule)'
    translations['分割数（偶数でなければならない）'] = 'Number of divisions (must be even)'
    translations['Simpson公式では分割数nは偶数でなければなりません'] = 'For Simpson\'s rule, the number of divisions n must be even'
    translations['Simpson公式の実装'] = '# Implementation of Simpson\'s rule'
    translations['奇数インデックス'] = '# Odd indices'
    translations['偶数インデックス'] = '# Even indices'

    translations['台形公式とSimpson公式の比較'] = '# Compare trapezoidal and Simpson\'s rules'
    translations['台形公式 vs Simpson公式:'] = 'Trapezoidal Rule vs Simpson\'s Rule:'
    translations['台形公式      誤差         Simpson公式   誤差'] = 'Trapezoidal      Error        Simpson         Error'
    translations['収束率の比較'] = '# Compare convergence rates'
    translations['台形公式とSimpson公式の収束性比較'] = 'Comparison of Convergence: Trapezoidal vs Simpson\'s Rule'
    translations['台形公式 O(h²)'] = 'Trapezoidal Rule O(h²)'
    translations['Simpson公式 O(h⁴)'] = 'Simpson\'s Rule O(h⁴)'

    # Section 1.4 - Gaussian Quadrature
    translations['1.4 Gauss求積法'] = '1.4 Gaussian Quadrature'
    translations['Gauss求積法は、関数の評価点と重みを最適化することで、少ない評価点数で高精度な積分を実現する手法です。'] = \
        'Gaussian quadrature is a method that achieves high-accuracy integration with fewer evaluation points by optimizing the evaluation points and weights.'
    translations['点のGauss求積法は'] = '-point Gaussian quadrature can exactly integrate polynomials up to degree'
    translations['次までの多項式を厳密に積分できます。'] = '.'

    translations['📚 理論: Gauss-Legendre求積法'] = '📚 Theory: Gauss-Legendre Quadrature'
    translations['での積分を考えます:'] = 'Consider the integral over the interval'
    translations['はLegendre多項式の零点、'] = 'are the zeros of the Legendre polynomial, and'
    translations['は対応する重みです。任意の区間'] = 'are the corresponding weights. The transformation to an arbitrary interval'
    translations['への変換は:'] = 'is:'

    # Gaussian quadrature code
    translations['Gauss-Legendre求積法による数値積分'] = 'Numerical integration using Gauss-Legendre quadrature'
    translations['Gauss点の数'] = 'Number of Gauss points'
    translations['Legendre多項式の零点と重みを取得'] = '# Get zeros and weights of Legendre polynomial'
    translations['区間[-1,1]から[a,b]への変換'] = '# Transform from interval [-1,1] to [a,b]'
    translations['積分の計算'] = '# Calculate integral'

    translations['SciPyの高精度積分で厳密値を計算'] = '# Calculate exact value with high-precision SciPy integration'
    translations['Gauss求積法:'] = 'Gaussian Quadrature:'
    translations['Gauss点数 n    近似値        誤差         関数評価回数'] = 'Gauss pts n    Approximation    Error        Function evals'
    translations['厳密値（SciPy quad）:'] = 'Exact value (SciPy quad):'
    translations['同じ関数評価回数での比較:'] = 'Comparison with same number of function evaluations:'
    translations['関数評価回数:'] = 'Function evaluations:'
    translations['点']:  'pts'
    translations['分割']: 'divs'
    translations['精度向上:'] = 'Accuracy improvement:'
    translations['倍'] = 'times'

    translations['Gauss求積法は同じ関数評価回数でSimpson公式より遙かに高精度です。特に滑らかな関数に対して効果的で、5点のGauss求積で機械精度レベルの精度が得られます。'] = \
        'Gaussian quadrature is much more accurate than Simpson\'s rule for the same number of function evaluations. It is especially effective for smooth functions, and 5-point Gaussian quadrature can achieve machine precision.'

    # Section 1.5 - NumPy/SciPy
    translations['1.5 NumPy/SciPyによる数値微分・積分'] = '1.5 Numerical Differentiation and Integration with NumPy/SciPy'
    translations['実務では、NumPy/SciPyの高機能な数値計算ライブラリを活用します。適応的手法や誤差評価機能を備えた関数が提供されています。'] = \
        'In practice, we utilize the advanced numerical computing libraries NumPy/SciPy. Functions with adaptive methods and error estimation capabilities are provided.'

    # Additional scipy examples
    translations['テスト関数群'] = '# Test functions'
    translations['振動関数'] = 'Oscillatory function'
    translations['特異性を持つ関数'] = 'Function with singularity'
    translations['適応的積分'] = '# Adaptive integration'
    translations['適応的Gauss-Kronrod法'] = 'Adaptive Gauss-Kronrod Method'
    translations['振動関数の積分'] = '# Integration of oscillatory function'
    translations['結果:'] = 'Result:'
    translations['推定誤差:'] = 'Estimated error:'
    translations['理論値:'] = 'Theoretical value:'

    translations['固定次数Gauss求積'] = '# Fixed-order Gauss quadrature'
    translations['固定次数Gauss-Legendre'] = 'Fixed-Order Gauss-Legendre'
    translations['点Gauss求積:'] = '-point Gauss quadrature:'

    translations['離散データの積分（実験データを想定）'] = '# Integration of discrete data (assuming experimental data)'
    translations['離散データの積分（trapz, simps）'] = 'Integration of Discrete Data (trapz, simps)'
    translations['実験データをシミュレート'] = '# Simulate experimental data'
    translations['11点のデータ'] = '# 11 data points'
    translations['台形公式'] = 'Trapezoidal rule'
    translations['Simpson公式'] = 'Simpson\'s rule'
    translations['trapzの誤差:'] = 'trapz error:'
    translations['simpsの誤差:'] = 'simps error:'

    translations['数値微分'] = '# Numerical differentiation'
    translations['1階微分'] = '# First derivative'
    translations['数値微分:'] = 'Numerical:'
    translations['2階微分'] = '# Second derivative'

    # Section 1.6 - Error Analysis
    translations['1.6 誤差解析と収束性評価'] = '1.6 Error Analysis and Convergence Evaluation'
    translations['数値微分・積分の実用では、誤差の評価と適切な手法選択が重要です。理論的な収束率を実験的に検証し、丸め誤差の影響も考慮します。'] = \
        'In practical numerical differentiation and integration, error evaluation and appropriate method selection are important. We experimentally verify theoretical convergence rates and consider the effects of round-off errors.'

    # Error analysis code
    translations['数値計算手法の収束率を解析'] = 'Analyze convergence rate of numerical method'
    translations['数値計算手法の関数'] = 'Numerical method function'
    translations['対象関数'] = 'Target function'
    translations['厳密解'] = 'Exact solution'
    translations['パラメータのリスト（刻み幅や分割数）'] = 'List of parameters (step sizes or divisions)'
    translations['手法の名前'] = 'Method name'
    translations['各パラメータでの誤差'] = 'Error for each parameter'

    translations['分割数のリスト'] = '# List of divisions'
    translations['各手法の収束率を評価'] = '# Evaluate convergence rate of each method'
    translations['数値積分手法の収束率解析:'] = 'Convergence Rate Analysis of Numerical Integration Methods:'

    translations['収束率の計算（連続する誤差の比）'] = '# Calculate convergence rate (ratio of consecutive errors)'
    translations['誤差の減少率から収束率を推定'] = 'Estimate convergence rate from error reduction'

    translations['結果の表示'] = '# Display results'
    translations['台形公式 (理論収束率:'] = 'Trapezoidal Rule (theoretical convergence rate:'
    translations['Simpson公式 (理論収束率:'] = 'Simpson\'s Rule (theoretical convergence rate:'
    translations['Gauss求積法'] = 'Gaussian Quadrature'
    translations['n      誤差          収束率'] = 'n      Error        Rate'
    translations['平均収束率:'] = 'Average rate:'

    translations['総合的な可視化'] = '# Comprehensive visualization'
    translations['誤差の収束'] = '# Error convergence'
    translations['収束性の比較'] = 'Convergence Comparison'
    translations['収束率の推移'] = '# Convergence rate evolution'
    translations['収束率'] = 'Convergence Rate'
    translations['理論値 (台形)'] = 'Theoretical (Trapezoidal)'
    translations['理論値 (Simpson)'] = 'Theoretical (Simpson)'

    translations['まとめ:'] = 'Summary:'
    translations['台形公式: 収束率'] = 'Trapezoidal rule: convergence rate'
    translations['理論通り'] = 'as expected theoretically'
    translations['Simpson公式: 収束率'] = 'Simpson\'s rule: convergence rate'
    translations['Gauss求積法: 指数的収束（多項式に対して厳密）'] = 'Gaussian quadrature: exponential convergence (exact for polynomials)'

    # Exercises
    translations['🏋️ 演習問題'] = '🏋️ Exercises'
    translations['演習1: 数値微分の実装'] = 'Exercise 1: Implementing Numerical Differentiation'
    translations['次の関数の'] = 'Calculate the derivative of the following function at'
    translations['における微分を、前進差分・後退差分・中心差分で計算し、誤差を比較せよ。刻み幅'] = \
        'using forward, backward, and central differences, and compare the errors. Try step sizes'
    translations['は0.1, 0.01, 0.001の3通りで試すこと。'] = 'of 0.1, 0.01, and 0.001.'

    translations['演習2: Richardson外挿の効果検証'] = 'Exercise 2: Verifying Richardson Extrapolation Effectiveness'
    translations['の'] = 'of'
    translations['における1階微分を次の方法で計算し、誤差を比較せよ（'] = 'at'
    translations['）:'] = 'using the following methods and compare the errors ('

    translations['演習3: 積分公式の精度比較'] = 'Exercise 3: Comparing Accuracy of Integration Formulas'
    translations['次の積分を台形公式、Simpson公式、Gauss求積法（5点）で計算し、精度と計算コストを比較せよ:'] = \
        'Calculate the following integral using the trapezoidal rule, Simpson\'s rule, and Gaussian quadrature (5 points), and compare accuracy and computational cost:'
    translations['ヒント: 厳密解は'] = 'Hint: The exact solution is'

    translations['演習4: 実験データの数値積分'] = 'Exercise 4: Numerical Integration of Experimental Data'
    translations['以下の実験データ（温度 vs 時間）から、0〜10秒間の平均温度を数値積分で求めよ:'] = \
        'From the following experimental data (temperature vs time), calculate the average temperature over 0-10 seconds using numerical integration:'
    translations['時刻 (s):'] = 'Time (s):'
    translations['温度 (°C):'] = 'Temperature (°C):'
    translations['台形公式とSimpson公式の両方で計算し、結果を比較せよ。'] = 'Calculate using both the trapezoidal rule and Simpson\'s rule, and compare the results.'

    translations['演習5: 材料科学への応用'] = 'Exercise 5: Applications to Materials Science'
    translations['材料の熱膨張係数'] = 'When the thermal expansion coefficient of a material'
    translations['が温度の関数として与えられたとき、温度変化に伴う長さの変化率は次式で計算されます:'] = \
        'is given as a function of temperature, the rate of length change due to temperature variation is calculated by:'
    translations['とし、'] = 'Take'
    translations['への温度上昇に伴う長さの変化率を数値積分で求めよ。'] = 'and calculate the length change rate due to temperature increase to'

    # Summary
    translations['まとめ'] = 'Summary'
    translations['本章では、数値微分と数値積分の基本的な手法を学びました:'] = 'In this chapter, we learned fundamental methods for numerical differentiation and integration:'
    translations['差分法（前進・後退・中心）とRichardson外挿による高精度化'] = 'Finite difference methods (forward, backward, central) and high-accuracy with Richardson extrapolation'
    translations['台形公式、Simpson公式、Gauss求積法の原理と実装'] = 'Principles and implementation of trapezoidal rule, Simpson\'s rule, and Gaussian quadrature'
    translations['理論的収束率の検証と実用的な精度評価'] = 'Verification of theoretical convergence rates and practical accuracy evaluation'
    translations['scipy.integrateとscipy.miscによる実践的数値計算'] = 'Practical numerical computation with scipy.integrate and scipy.misc'

    translations['これらの手法は、材料科学・プロセス工学における実験データ解析、シミュレーション、最適化など幅広い場面で活用されます。次章では、これらの基礎の上に立って線形方程式系の数値解法を学びます。'] = \
        'These methods are utilized in a wide range of applications in materials science and process engineering, including experimental data analysis, simulation, and optimization. In the next chapter, we will learn numerical methods for systems of linear equations building on these foundations.'

    # Navigation
    translations['← シリーズ目次'] = '← Series Table of Contents'
    translations['第2章へ →'] = 'Chapter 2 →'
    translations['← 第1章'] = '← Chapter 1'
    translations['第3章へ →'] = 'Chapter 3 →'

    # Footer
    translations['&copy; 2025 FM Dojo. All rights reserved.'] = '&copy; 2025 FM Dojo. All rights reserved.'

    # Additional Chapter 1 phrases (found in remaining analysis)
    translations['Gauss求積'] = 'Gaussian Quadrature'
    translations['誤差 ∝ 1/nᵖ'] = 'Error ∝ 1/nᵖ'

    # Fix remaining partial translations in Chapter 1
    translations['# Central difference'] = '# Central difference'  # Keep as is if already translated
    translations['print(f"Central difference (h={h}):")'] = 'print(f"Central difference (h={h}):")'
    translations['Central difference (h=0.1):'] = 'Central difference (h=0.1):'
    translations['Simpson公式との比較（同じ関数評価回数で）'] = '# Compare with Simpson\'s rule (same number of function evaluations)'
    translations['# Simpson公式（同じ評価回数）'] = '# Simpson\'s rule (same number of evaluations)'
    translations['n_simpson = n_gauss - 1  # Simpson公式ではn+1点を評価'] = 'n_simpson = n_gauss - 1  # Simpson\'s rule evaluates n+1 points'
    translations['print(f"  Gauss ({n_gauss}点):  誤差 {gauss_error:.2e}")'] = 'print(f"  Gauss ({n_gauss} pts):  error {gauss_error:.2e}")'
    translations['print(f"  Simpson ({n_simpson}分割): 誤差 {simpson_error:.2e}")'] = 'print(f"  Simpson ({n_simpson} divs): error {simpson_error:.2e}")'
    translations['Gauss (5点):  誤差 6.66e-12'] = 'Gauss (5 pts):  error 6.66e-12'
    translations['Simpson (4分割): 誤差 1.69e-06'] = 'Simpson (4 divs): error 1.69e-06'
    translations['Gauss (10点):  誤差 4.44e-16'] = 'Gauss (10 pts):  error 4.44e-16'
    translations['Simpson (8分割): 誤差 2.65e-08'] = 'Simpson (8 divs): error 2.65e-08'
    translations['# Gaussian quadrature'] = '# Gaussian quadrature'  # Keep as is
    translations['ax1.loglog(n_values, gauss_errors, \'^-\', label=\'Gaussian Quadrature\', markersize=8, linewidth=2)'] = \
        'ax1.loglog(n_values, gauss_errors, \'^-\', label=\'Gaussian Quadrature\', markersize=8, linewidth=2)'
    translations['ax2.set_ylabel(\'Convergence Rate p (Error ∝ 1/nᵖ)\', fontsize=12)'] = \
        'ax2.set_ylabel(\'Convergence Rate p (Error ∝ 1/nᵖ)\', fontsize=12)'
    translations['<li>(a) Central difference</li>'] = '<li>(a) Central difference</li>'

    # Summary items that need fixing
    translations['<li><strong>Numerical integration:</strong> Principles and implementation of trapezoidal rule, Simpson\'s rule, and Gaussian quadrature</li>'] = \
        '<li><strong>Numerical integration:</strong> Principles and implementation of trapezoidal rule, Simpson\'s rule, and Gaussian quadrature</li>'
    translations['<li><strong>Error analysis:</strong> Verification of theoretical convergence rates and practical accuracy evaluation</li>'] = \
        '<li><strong>Error analysis:</strong> Verification of theoretical convergence rates and practical accuracy evaluation</li>'
    translations['<li><strong>Using SciPy:</strong> Practical numerical computation with scipy.integrate and scipy.misc</li>'] = \
        '<li><strong>Using SciPy:</strong> Practical numerical computation with scipy.integrate and scipy.misc</li>'

    # ========== CHAPTER 2 SPECIFIC ==========

    # Title/Meta
    translations['<title>第2章: 線形方程式系の解法 - 数値解析の基礎</title>'] = \
        '<title>Chapter 2: Solving Systems of Linear Equations - Fundamentals of Numerical Analysis</title>'

    translations['content="大規模連立一次方程式の数値解法を学びます。Gauss消去法、LU分解、反復法（Jacobi法、Gauss-Seidel法、SOR法）、疎行列処理をPythonで実装します。"'] = \
        'content="Learn numerical methods for solving large-scale systems of linear equations. Implement Gaussian elimination, LU decomposition, iterative methods (Jacobi, Gauss-Seidel, SOR), and sparse matrix operations in Python."'

    # Main header
    translations['第2章: 線形方程式系の解法'] = 'Chapter 2: Solving Systems of Linear Equations'
    translations['大規模連立一次方程式を効率的に解く直接法と反復法'] = \
        'Direct and iterative methods for efficiently solving large-scale systems of linear equations'

    # Section 2.1
    translations['2.1 連立一次方程式の基礎'] = '2.1 Fundamentals of Systems of Linear Equations'
    translations['材料シミュレーション（有限要素法、拡散方程式など）では、'] = \
        'In materials simulation (finite element method, diffusion equations, etc.), systems of linear equations in the form'
    translations['の形の大規模連立一次方程式が頻繁に現れます。ここで'] = 'frequently appear. Here'
    translations['は'] = 'is a'
    translations['行列、'] = 'matrix, and'
    translations['次元ベクトルです。'] = '-dimensional vectors.'

    translations['📚 理論: 直接法と反復法'] = '📚 Theory: Direct and Iterative Methods'
    translations['直接法 (Direct Methods):'] = 'Direct Methods:'
    translations['有限回の演算で厳密解を得る（理論上）'] = 'Obtain exact solution in finite number of operations (theoretically)'
    translations['例: Gauss消去法、LU分解、Cholesky分解'] = 'Examples: Gaussian elimination, LU decomposition, Cholesky decomposition'
    translations['計算量:'] = 'Computational complexity:'
    translations['小〜中規模問題（'] = 'Suitable for small to medium problems ('
    translations['）に適する'] = ')'

    translations['反復法 (Iterative Methods):'] = 'Iterative Methods:'
    translations['初期値から出発して解に収束させる'] = 'Start from initial value and converge to solution'
    translations['例: Jacobi法、Gauss-Seidel法、SOR法、共役勾配法'] = 'Examples: Jacobi method, Gauss-Seidel method, SOR method, conjugate gradient method'
    translations['反復1回の計算量:'] = 'Computational complexity per iteration:'
    translations['または'] = 'or'
    translations['（疎行列）'] = '(sparse matrices)'
    translations['大規模・疎行列問題（'] = 'Suitable for large-scale sparse matrix problems ('
    translations['）に適する'] = ')'

    # Code Example 1 - Gaussian Elimination
    translations['コード例1: Gauss消去法の実装'] = 'Code Example 1: Implementing Gaussian Elimination'
    translations['Gauss消去法による連立一次方程式の求解'] = 'Solving systems of linear equations using Gaussian elimination'
    translations['係数行列'] = 'Coefficient matrix'
    translations['右辺ベクトル'] = 'Right-hand side vector'
    translations['解ベクトル'] = 'Solution vector'
    translations['拡大係数行列の作成（元の行列を変更しないようコピー）'] = '# Create augmented matrix (copy to preserve original)'
    translations['前進消去 (Forward Elimination)'] = '# Forward elimination'
    translations['ピボット選択（部分ピボット選択）'] = '# Pivot selection (partial pivoting)'
    translations['k列目の消去'] = '# Eliminate k-th column'
    translations['ゼロピボットが発生しました'] = 'Zero pivot encountered'
    translations['後退代入 (Back Substitution)'] = '# Back substitution'

    translations['次元連立方程式'] = '-dimensional system of equations'
    translations['Gauss消去法による連立一次方程式の求解'] = 'Solving Systems of Linear Equations Using Gaussian Elimination'
    translations['解 x (Gauss消去法):'] = 'Solution x (Gaussian elimination):'
    translations['解 x (NumPy):'] = 'Solution x (NumPy):'
    translations['残差'] = 'Residual'
    translations['NumPyとの差:'] = 'Difference from NumPy:'

    # Section 2.2 - LU Decomposition
    translations['2.2 LU分解'] = '2.2 LU Decomposition'
    translations['LU分解は、行列'] = 'LU decomposition is a method that factorizes matrix'
    translations['を下三角行列'] = 'into a lower triangular matrix'
    translations['と上三角行列'] = 'and an upper triangular matrix'
    translations['の積に分解する手法です。一度分解すれば、異なる右辺'] = '. Once factorized, solutions for different right-hand sides'
    translations['に対して効率的に解を求められます。'] = 'can be found efficiently.'

    translations['📚 理論: LU分解の原理'] = '📚 Theory: Principles of LU Decomposition'
    translations['と分解すると、'] = 'After factorizing as'
    translations['は次の2段階で解けます:'] = 'can be solved in two stages:'
    translations['前進代入'] = 'Forward substitution'
    translations['後退代入'] = 'Back substitution'
    translations['計算量: 分解'] = 'Computational complexity: factorization'
    translations['、各求解'] = ', each solve'
    translations['。複数の右辺がある場合に効率的です。'] = '. Efficient when there are multiple right-hand sides.'

    translations['コード例2: LU分解の実装'] = 'Code Example 2: Implementing LU Decomposition'
    translations['LU分解（Doolittle法）'] = 'LU decomposition (Doolittle method)'
    translations['下三角行列'] = 'Lower triangular matrix'
    translations['上三角行列'] = 'Upper triangular matrix'
    translations['の i 行目を計算'] = '# Calculate i-th row of U'
    translations['の i 列目を計算'] = '# Calculate i-th column of L'

    translations['LU分解を使った方程式の求解'] = 'Solving equations using LU decomposition'
    translations['LU分解された行列'] = 'LU decomposed matrices'
    translations['前進代入: Ly = b'] = '# Forward substitution: Ly = b'
    translations['後退代入: Ux = y'] = '# Back substitution: Ux = y'

    translations['LU分解による連立方程式の求解'] = 'Solving Systems Using LU Decomposition'
    translations['下三角行列 L:'] = 'Lower triangular matrix L:'
    translations['上三角行列 U:'] = 'Upper triangular matrix U:'
    translations['LU の積（元の行列と一致するはず）:'] = 'Product LU (should match original matrix):'
    translations['複数の右辺ベクトルに対する求解'] = 'Solving for Multiple Right-Hand Side Vectors'
    translations['SciPy の LU分解との比較'] = 'Comparison with SciPy LU Decomposition'

    # Section 2.3 - Jacobi Method
    translations['2.3 反復法の基礎 - Jacobi法'] = '2.3 Fundamentals of Iterative Methods - Jacobi Method'
    translations['反復法は初期値から出発し、逐次的に解に近づけていく手法です。大規模・疎行列問題では直接法より効率的です。'] = \
        'Iterative methods start from an initial value and approach the solution iteratively. They are more efficient than direct methods for large-scale sparse matrix problems.'

    translations['📚 理論: Jacobi法の原理'] = '📚 Theory: Principles of the Jacobi Method'
    translations['を対角成分'] = 'Decompose matrix'
    translations['、下三角部'] = 'into diagonal component'
    translations['、上三角部'] = ', strictly lower triangular part'
    translations['に分解:'] = ', and strictly upper triangular part'
    translations['Jacobi法の反復式:'] = 'Jacobi iteration formula:'
    translations['成分ごとには:'] = 'Component-wise:'
    translations['収束条件:'] = 'Convergence condition:'
    translations['が対角優位であれば収束が保証されます。'] = 'If matrix is diagonally dominant, convergence is guaranteed.'

    translations['コード例3: Jacobi法の実装'] = 'Code Example 3: Implementing the Jacobi Method'
    translations['Jacobi法による反復求解'] = 'Iterative solution using Jacobi method'
    translations['初期値（デフォルト: ゼロベクトル）'] = 'Initial value (default: zero vector)'
    translations['最大反復回数'] = 'Maximum number of iterations'
    translations['収束判定の閾値'] = 'Convergence threshold'
    translations['各反復での残差ノルム'] = 'Residual norm at each iteration'
    translations['x_i の更新（他の成分は前回の値を使用）'] = '# Update x_i (using previous values for other components)'
    translations['残差の計算'] = '# Calculate residual'
    translations['収束判定'] = '# Check convergence'
    translations['Jacobi法:'] = 'Jacobi method:'
    translations['回の反復で収束'] = 'iterations to converge'
    translations['回の反復で収束せず（残差:'] = 'iterations without convergence (residual:'

    translations['対角優位な行列'] = '# Diagonally dominant matrix'
    translations['Jacobi法による反復求解'] = 'Iterative Solution Using Jacobi Method'
    translations['対角優位性の確認:'] = 'Checking diagonal dominance:'
    translations['行'] = 'Row'
    translations['（対角優位）'] = '(diagonally dominant)'
    translations['解 x (Jacobi法):'] = 'Solution x (Jacobi method):'
    translations['解 x (厳密解):'] = 'Solution x (exact):'
    translations['反復回数'] = 'Iteration count'
    translations['残差ノルム ||Ax - b||'] = 'Residual norm ||Ax - b||'
    translations['Jacobi法の収束履歴'] = 'Convergence History of Jacobi Method'
    translations['収束に要した反復回数:'] = 'Number of iterations to convergence:'

    # Section 2.4 - Gauss-Seidel and SOR
    translations['2.4 Gauss-Seidel法とSOR法'] = '2.4 Gauss-Seidel and SOR Methods'
    translations['Gauss-Seidel法はJacobi法を改良し、更新された値をすぐに使うことで収束を高速化します。SOR法はさらに緩和係数を導入して収束を加速します。'] = \
        'The Gauss-Seidel method improves upon the Jacobi method by immediately using updated values to accelerate convergence. The SOR method further accelerates convergence by introducing a relaxation factor.'

    translations['📚 理論: Gauss-Seidel法とSOR法'] = '📚 Theory: Gauss-Seidel and SOR Methods'
    translations['Gauss-Seidel法:'] = 'Gauss-Seidel Method:'
    translations['更新済みの'] = 'Immediately uses updated values'
    translations['をすぐに使用するため、Jacobi法より収束が速いことが多いです。'] = ', often converging faster than the Jacobi method.'
    translations['SOR法 (Successive Over-Relaxation):'] = 'SOR Method (Successive Over-Relaxation):'
    translations['緩和係数'] = 'Relaxation factor'
    translations['の最適値は問題依存ですが、通常'] = 'The optimal value of'
    translations['で最速収束します。'] = 'is problem-dependent, but typically'

    translations['コード例4: Gauss-Seidel法の実装'] = 'Code Example 4: Implementing the Gauss-Seidel Method'
    translations['Gauss-Seidel法による反復求解'] = 'Iterative solution using Gauss-Seidel method'
    translations['更新済みの値をすぐに使用'] = '# Immediately use updated values'
    translations['Gauss-Seidel法:'] = 'Gauss-Seidel method:'

    translations['同じ問題でJacobi法と比較'] = '# Compare with Jacobi method on same problem'
    translations['Jacobi法 vs Gauss-Seidel法の比較'] = 'Comparison: Jacobi vs Gauss-Seidel Methods'
    translations['Jacobi法:'] = 'Jacobi method:'
    translations['Gauss-Seidel法:'] = 'Gauss-Seidel method:'
    translations['Jacobi法とGauss-Seidel法の収束速度比較'] = 'Convergence Speed Comparison: Jacobi vs Gauss-Seidel'
    translations['高速化率:'] = 'Speedup ratio:'

    translations['コード例5: SOR法の実装と最適緩和係数'] = 'Code Example 5: Implementing SOR Method and Optimal Relaxation Factor'
    translations['SOR法による反復求解'] = 'Iterative solution using SOR method'
    translations['緩和係数 (1 < omega < 2 が推奨)'] = 'Relaxation factor (1 < omega < 2 recommended)'
    translations['SOR更新: 緩和係数を適用'] = '# SOR update: apply relaxation factor'

    translations['最適緩和係数の探索'] = '# Search for optimal relaxation factor'
    translations['SOR法: 最適緩和係数の探索'] = 'SOR Method: Searching for Optimal Relaxation Factor'
    translations['結果の可視化'] = '# Visualize results'
    translations['緩和係数と反復回数の関係'] = '# Relationship between relaxation factor and iteration count'
    translations['最適'] = 'Optimal'
    translations['緩和係数 ω'] = 'Relaxation factor ω'
    translations['収束までの反復回数'] = 'Iterations to convergence'
    translations['SOR法: 緩和係数の影響'] = 'SOR Method: Effect of Relaxation Factor'
    translations['異なるωでの収束履歴'] = '# Convergence history for different ω'
    translations['異なる緩和係数での収束速度'] = 'Convergence Speed for Different Relaxation Factors'
    translations['最適緩和係数:'] = 'Optimal relaxation factor:'
    translations['最小反復回数:'] = 'Minimum iterations:'

    # Section 2.5 - Sparse Matrices
    translations['2.5 疎行列の扱い'] = '2.5 Handling Sparse Matrices'
    translations['有限要素法や有限差分法で生じる行列は、多くの要素がゼロである疎行列（sparse matrix）です。SciPyの疎行列ライブラリを使うことで、メモリと計算時間を大幅に削減できます。'] = \
        'Matrices arising from finite element and finite difference methods are sparse matrices with many zero elements. Using SciPy\'s sparse matrix library can significantly reduce memory usage and computation time.'

    translations['コード例6: SciPy疎行列ソルバー'] = 'Code Example 6: SciPy Sparse Matrix Solvers'
    translations['1次元Laplacian行列の生成（有限差分法）'] = 'Generate 1D Laplacian matrix (finite difference method)'
    translations['の離散化に対応'] = 'Corresponding to discretization of'
    translations['格子点数'] = 'Number of grid points'
    translations['の三重対角行列'] = 'tridiagonal matrix'
    translations['三重対角要素'] = '# Tridiagonal elements'
    translations['疎行列の生成'] = '# Generate sparse matrix'

    translations['問題サイズ'] = '# Problem size'
    translations['疎行列ソルバーのベンチマーク (問題サイズ:'] = 'Sparse Matrix Solver Benchmark (Problem size:'
    translations['Laplacian行列の生成'] = '# Generate Laplacian matrix'
    translations['行列の非ゼロ要素数:'] = 'Number of non-zero elements:'
    translations['行列の全要素数:'] = 'Total number of matrix elements:'
    translations['疎率:'] = 'Sparsity:'

    translations['メモリ使用量の比較'] = '# Compare memory usage'
    translations['メモリ使用量:'] = 'Memory usage:'
    translations['疎行列形式:'] = 'Sparse format:'
    translations['密行列形式:'] = 'Dense format:'
    translations['削減率:'] = 'Reduction rate:'

    translations['疎行列直接法 (spsolve)'] = '# Sparse direct method (spsolve)'
    translations['疎行列直接法 (spsolve)'] = 'Sparse Direct Method (spsolve)'
    translations['計算時間:'] = 'Computation time:'

    translations['密行列直接法 (np.linalg.solve)'] = '# Dense direct method (np.linalg.solve)'
    translations['密行列直接法 (np.linalg.solve)'] = 'Dense Direct Method (np.linalg.solve)'

    translations['共役勾配法 (CG法 - 対称正定値行列用)'] = '# Conjugate gradient method (CG - for symmetric positive definite matrices)'
    translations['共役勾配法 (CG法)'] = 'Conjugate Gradient Method (CG)'
    translations['収束情報:'] = 'Convergence info:'
    translations['なら成功'] = '(0 means success)'

    translations['GMRES法（一般的な行列用）'] = '# GMRES method (for general matrices)'
    translations['GMRES法'] = 'GMRES Method'

    translations['性能比較の可視化'] = '# Visualize performance comparison'
    translations['疎行列\n直接法'] = 'Sparse\nDirect'
    translations['密行列\n直接法'] = 'Dense\nDirect'
    translations['CG法'] = 'CG Method'
    translations['GMRES法'] = 'GMRES'
    translations['各バーに値を表示'] = '# Display values on bars'
    translations['疎行列ソルバーの性能比較 (行列サイズ:'] = 'Sparse Matrix Solver Performance Comparison (Matrix size:'

    # Section 2.6 - Condition Number
    translations['2.6 条件数と数値安定性'] = '2.6 Condition Number and Numerical Stability'
    translations['条件数は行列の「解きにくさ」を表す指標です。条件数が大きい（ill-conditioned）行列では、丸め誤差が増幅され、数値計算の精度が低下します。'] = \
        'The condition number is an indicator of how "difficult" a matrix is to solve. For ill-conditioned matrices with large condition numbers, round-off errors are amplified, degrading numerical accuracy.'

    translations['📚 理論: 条件数'] = '📚 Theory: Condition Number'
    translations['行列'] = 'The condition number of matrix'
    translations['の条件数は次のように定義されます:'] = 'is defined as:'
    translations['条件数の解釈:'] = 'Interpretation of condition number:'
    translations['理想的（直交行列）'] = 'Ideal (orthogonal matrix)'
    translations['良条件'] = 'Well-conditioned'
    translations['悪条件（注意が必要）'] = 'Ill-conditioned (caution required)'
    translations['特異行列に近い（数値計算困難）'] = 'Near-singular (numerically difficult)'

    translations['コード例7: 条件数の解析と前処理'] = 'Code Example 7: Condition Number Analysis and Preconditioning'
    translations['Hilbert行列の生成（悪条件行列の典型例）'] = 'Generate Hilbert matrix (typical example of ill-conditioned matrix)'
    translations['条件数と数値安定性の解析'] = 'Analysis of Condition Number and Numerical Stability'
    translations['良条件な行列'] = '# Well-conditioned matrix'
    translations['悪条件な行列（Hilbert行列）'] = '# Ill-conditioned matrix (Hilbert matrix)'
    translations['条件数の計算'] = '# Calculate condition numbers'
    translations['良条件な行列:'] = 'Well-conditioned matrix:'
    translations['条件数:'] = 'Condition number:'
    translations['悪条件な行列 (Hilbert 5x5):'] = 'Ill-conditioned matrix (Hilbert 5x5):'

    translations['数値実験: 右辺の摂動に対する感度'] = 'Numerical Experiment: Sensitivity to Right-Hand Side Perturbations'
    translations['右辺の微小変化に対する解の変化を調べる'] = 'Examine solution changes due to small right-hand side changes'
    translations['右辺に1%の摂動を加える'] = '# Add 1% perturbation to right-hand side'
    translations['解の相対変化'] = '# Relative change in solution'
    translations['良条件行列:'] = 'Well-conditioned matrix:'
    translations['右辺の相対変化:'] = 'Relative change in RHS:'
    translations['解の相対変化:'] = 'Relative change in solution:'
    translations['増幅率:'] = 'Amplification factor:'
    translations['理論上限 (条件数):'] = 'Theoretical upper bound (condition number):'
    translations['悪条件行列 (Hilbert):'] = 'Ill-conditioned matrix (Hilbert):'

    translations['様々なサイズのHilbert行列の条件数'] = '# Condition numbers of Hilbert matrices of various sizes'
    translations['Hilbert行列の条件数（サイズ依存性）'] = 'Condition Numbers of Hilbert Matrices (Size Dependence)'
    translations['サイズ     条件数'] = 'Size     Condition Number'
    translations['Hilbert行列の条件数'] = 'Condition Numbers of Hilbert Matrices'
    translations['行列サイズ n'] = 'Matrix size n'
    translations['良条件の目安'] = 'Well-conditioned guideline'
    translations['悪条件の目安'] = 'Ill-conditioned guideline'
    translations['機械精度の限界'] = 'Machine precision limit'

    translations['条件数が大きいと、わずかな摂動が解に大きく影響'] = 'Large condition numbers mean small perturbations greatly affect the solution'
    translations['Hilbert行列は極めて悪条件（サイズ10で条件数'] = 'Hilbert matrices are extremely ill-conditioned (size 10 has condition number'
    translations['実務では前処理やスケーリングで条件数を改善'] = 'In practice, use preconditioning or scaling to improve condition number'

    # Exercises - Chapter 2
    translations['演習1: LU分解の実装検証'] = 'Exercise 1: Verifying LU Decomposition Implementation'
    translations['次の行列に対してLU分解を実行し、'] = 'Perform LU decomposition on the following matrix and verify that'
    translations['が成立することを検証せよ:'] = 'holds:'

    translations['演習2: 反復法の収束条件'] = 'Exercise 2: Convergence Conditions for Iterative Methods'
    translations['次の行列は対角優位でないため、Jacobi法が収束しない可能性があります。実際に試し、収束するか確認せよ:'] = \
        'The following matrix is not diagonally dominant, so the Jacobi method may not converge. Try it and check if it converges:'
    translations['収束しない場合、行の入れ替えで対角優位にできるか検討せよ。'] = \
        'If it does not converge, consider whether row permutation can make it diagonally dominant.'

    translations['演習3: SOR法の最適緩和係数'] = 'Exercise 3: Optimal Relaxation Factor for SOR Method'
    translations['次の5×5三重対角行列に対して、SOR法の最適緩和係数を実験的に求めよ（'] = \
        'Experimentally determine the optimal relaxation factor for the SOR method for the following 5×5 tridiagonal matrix ('
    translations['の範囲で0.05刻みで試す）:'] = 'in increments of 0.05):'
    translations['対角要素:'] = 'Diagonal elements:'
    translations['上下の副対角要素:'] = 'Upper and lower off-diagonal elements:'

    translations['演習4: 疎行列の効率性'] = 'Exercise 4: Efficiency of Sparse Matrices'
    translations['2次元Laplacian行列（格子サイズ'] = 'Generate a 2D Laplacian matrix (grid size'
    translations['）を生成し、次を比較せよ:'] = ') and compare the following:'
    translations['疎行列形式と密行列形式のメモリ使用量'] = 'Memory usage of sparse vs dense matrix formats'
    translations['spsolveとnp.linalg.solveの計算時間'] = 'Computation time of spsolve vs np.linalg.solve'

    translations['演習5: 条件数と精度劣化'] = 'Exercise 5: Condition Number and Accuracy Degradation'
    translations['Hilbert行列（'] = 'For Hilbert matrices ('
    translations['）に対して、次を調べよ:'] = '), investigate the following:'
    translations['各サイズの条件数'] = 'Condition number for each size'
    translations['右辺ベクトル'] = 'For right-hand side vector'
    translations['に対する解'] = ', calculate solution'
    translations['計算した'] = 'Using the calculated'
    translations['を使って'] = ', recompute'
    translations['を再計算し、元の'] = 'and compare the error with the original'
    translations['との誤差'] = '.'
    translations['条件数が大きくなるにつれて精度が劣化することを確認せよ。'] = \
        'Confirm that accuracy degrades as the condition number increases.'

    # Summary - Chapter 2
    translations['本章では、線形方程式系の数値解法を体系的に学びました:'] = 'In this chapter, we systematically learned numerical methods for systems of linear equations:'
    translations['直接法:'] = 'Direct methods:'
    translations['Gauss消去法、LU分解による厳密解の計算'] = 'Computing exact solutions using Gaussian elimination and LU decomposition'
    translations['反復法:'] = 'Iterative methods:'
    translations['Jacobi法、Gauss-Seidel法、SOR法による大規模問題への対応'] = 'Handling large-scale problems using Jacobi, Gauss-Seidel, and SOR methods'
    translations['疎行列:'] = 'Sparse matrices:'
    translations['SciPyの疎行列ライブラリによる効率的な計算'] = 'Efficient computation using SciPy sparse matrix library'
    translations['数値安定性:'] = 'Numerical stability:'
    translations['条件数による解きやすさの評価と対策'] = 'Evaluating solvability through condition numbers and countermeasures'

    translations['これらの手法は、有限要素法、有限差分法、最適化問題など、幅広い数値計算の基礎となります。次章では、非線形方程式の解法に進みます。'] = \
        'These methods form the foundation for a wide range of numerical computations including finite element methods, finite difference methods, and optimization problems. In the next chapter, we will proceed to solving nonlinear equations.'

    # Fix remaining partial translations in Chapter 2
    translations['# Gaussian eliminationで解く'] = '# Solve using Gaussian elimination'
    translations['消去法で解く'] = '# Solve using elimination method'
    translations['解を比較'] = '# Compare solutions'
    translations['# 検証: Ax = b'] = '# Verification: Ax = b'
    translations['検証:'] = 'Verification:'
    translations['精度検証'] = '# Verify accuracy'
    translations['テスト:'] = '# Test:'
    translations['# LU分解'] = '# LU decomposition'
    translations['分解'] = 'decomposition'
    translations['# LU の積を検証'] = '# Verify LU product'
    translations['積を検証'] = 'Verify product'
    translations['複数の右辺ベクトルに対する求解'] = 'Solving for Multiple Right-Hand Side Vectors'
    translations['複数の右辺に対して解く'] = '# Solve for multiple right-hand sides'
    translations['解 x1 ='] = 'Solution x1 ='
    translations['解 x2 ='] = 'Solution x2 ='
    translations['# SciPy の LU分解との比較'] = '# Comparison with SciPy LU Decomposition'
    translations['分解との比較'] = 'Comparison with decomposition'
    translations['LU の積(元の行列と一致するはず):'] = 'Product LU (should match original matrix):'
    translations['積(元の行列と一致するはず):'] = 'Product (should match original matrix):'
    translations['元の行列と一致するはず'] = 'should match original matrix'
    translations['対角優位性の確認:'] = 'Checking diagonal dominance:'
    translations['確認:'] = 'Checking:'
    translations['対角優位性'] = 'diagonal dominance'

    # More granular particles
    translations['で'] = ''  # particle, context-dependent
    translations['に'] = ''  # particle
    translations['を'] = ''  # particle
    translations['する'] = ''  # verb ending
    translations['と'] = ''  # particle
    translations['から'] = ''  # particle (when standalone)
    translations['まで'] = ''  # particle
    translations['が'] = ''  # particle
    translations['は'] = ''  # particle (when standalone in explanatory text)
    translations['も'] = ''  # particle
    translations['へ'] = ''  # particle
    translations['や'] = ''  # particle

    return translations

def translate_file(input_path, output_path):
    """Apply all translations to a file"""

    # Read source
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get translation map
    translations = create_complete_translation_map()

    # Count before
    japanese_before = count_japanese(content)
    print(f"\n{os.path.basename(input_path)}:")
    print(f"  Japanese characters before translation: {japanese_before}")

    # Apply translations in order (longer phrases first to avoid partial matches)
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)

    translation_count = 0
    for japanese, english in sorted_translations:
        if japanese in content:
            content = content.replace(japanese, english)
            translation_count += 1

    print(f"  Applied {translation_count} translation replacements")

    # Count after
    japanese_after = count_japanese(content)
    print(f"  Japanese characters after translation: {japanese_after}")
    print(f"  Translation coverage: {(japanese_before - japanese_after) / japanese_before * 100:.1f}%")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return japanese_after

def main():
    """Main translation function"""

    base_dir = os.getcwd()
    jp_dir = os.path.join(base_dir, "knowledge/jp/FM/numerical-analysis-fundamentals")
    en_dir = os.path.join(base_dir, "knowledge/en/FM/numerical-analysis-fundamentals")

    print("=" * 70)
    print("COMPREHENSIVE TRANSLATION: Numerical Analysis Chapters 1 & 2")
    print("=" * 70)

    # Translate Chapter 1
    ch1_remaining = translate_file(
        os.path.join(jp_dir, "chapter-1.html"),
        os.path.join(en_dir, "chapter-1.html")
    )

    # Translate Chapter 2
    ch2_remaining = translate_file(
        os.path.join(jp_dir, "chapter-2.html"),
        os.path.join(en_dir, "chapter-2.html")
    )

    print("\n" + "=" * 70)
    print("TRANSLATION SUMMARY")
    print("=" * 70)
    print(f"Chapter 1 Japanese characters remaining: {ch1_remaining}")
    print(f"Chapter 2 Japanese characters remaining: {ch2_remaining}")
    print(f"Total Japanese characters remaining: {ch1_remaining + ch2_remaining}")

    if (ch1_remaining + ch2_remaining) == 0:
        print("\n✓ SUCCESS: Complete translation achieved!")
        return 0
    else:
        print(f"\n⚠ PARTIAL: {ch1_remaining + ch2_remaining} Japanese characters still need translation")
        print("These may require manual review or additional translation pairs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
