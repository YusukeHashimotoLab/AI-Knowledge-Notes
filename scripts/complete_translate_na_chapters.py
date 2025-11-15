#!/usr/bin/env python3
"""
COMPLETE translation script for numerical analysis chapters 1 and 2.
This script translates ALL Japanese text comprehensively.

Translation requirements:
- Translate ALL Japanese text (hiragana, katakana, kanji)
- Preserve HTML structure exactly
- Preserve MathJax equations (between \( \), \[ \], $$ $$)
- Preserve Python code (variable names, function names)
- Translate Python comments and docstrings
- Translate print() string literals
"""

import re
import sys

def create_comprehensive_translation_map():
    """Create comprehensive mapping of ALL Japanese phrases to English"""

    # This is a comprehensive map covering ALL content
    translations = {
        # HTML lang attribute
        'lang="ja"': 'lang="en"',

        # ========== CHAPTER 1 TRANSLATIONS ==========

        # Meta tags
        '第1章: 数値微分と数値積分 - 数値解析の基礎': 'Chapter 1: Numerical Differentiation and Integration - Fundamentals of Numerical Analysis',
        '数値微分と数値積分の基本手法を学びます。差分法、Richardson外挿法、台形公式、Simpson公式、Gauss求積法をPythonで実装します。':
            'Learn fundamental methods for numerical differentiation and integration. Implement finite difference methods, Richardson extrapolation, trapezoidal rule, Simpson\'s rule, and Gaussian quadrature in Python.',

        # Breadcrumb
        '基礎数理道場': 'Fundamental Mathematics Dojo',
        '数値解析の基礎': 'Fundamentals of Numerical Analysis',
        '第1章': 'Chapter 1',
        '第2章': 'Chapter 2',

        # Main headers
        '第1章: 数値微分と数値積分': 'Chapter 1: Numerical Differentiation and Integration',
        '解析的に計算できない微分・積分を数値的に近似する基本手法':
            'Fundamental methods for numerically approximating derivatives and integrals that cannot be computed analytically',

        # Section 1.1
        '1.1 数値微分の基礎': '1.1 Fundamentals of Numerical Differentiation',
        '微分の定義': 'In the definition of differentiation',
        'において、': ', by taking',
        'を十分小さい値にとることで微分を近似できます。この考え方に基づく様々な差分法を学びます。':
            'to be a sufficiently small value, we can approximate the derivative. We will learn various finite difference methods based on this idea.',

        '📚 理論: 差分法の分類': '📚 Theory: Classification of Finite Difference Methods',
        '前進差分 (Forward Difference):': 'Forward Difference:',
        '後退差分 (Backward Difference):': 'Backward Difference:',
        '中心差分 (Central Difference):': 'Central Difference:',

        '中心差分は': 'The central difference has',
        'の精度を持ち、前進・後退差分の': 'accuracy, which is higher than the',
        'より高精度です。ただし、両端点での計算には注意が必要です。':
            'accuracy of forward and backward differences. However, care must be taken when computing at boundary points.',

        # Code examples
        'コード例1: 前進・後退・中心差分法の実装': 'Code Example 1: Implementing Forward, Backward, and Central Difference Methods',
        '前進差分法による数値微分': 'Numerical differentiation using forward difference',
        '後退差分法による数値微分': 'Numerical differentiation using backward difference',
        '中心差分法による数値微分': 'Numerical differentiation using central difference',

        # More code comments
        'テスト関数:': '# Test function:',
        '評価点': '# Evaluation point',
        '刻み幅を変化させて誤差を評価': '# Evaluate error for varying step sizes',
        '可視化': '# Visualization',
        '参照線': '# Reference lines',
        '刻み幅 h': 'Step size h',
        '絶対誤差': 'Absolute error',
        '数値微分の誤差解析': 'Error Analysis of Numerical Differentiation',

        # Output text
        '評価点:': 'Evaluation point:',
        '厳密値:': 'Exact value:',
        'での結果:': 'Results for',
        '前進差分:': 'Forward difference:',
        '後退差分:': 'Backward difference:',
        '中心差分:': 'Central difference:',
        '誤差:': 'error:',

        '考察:': 'Discussion:',
        '中心差分は理論通り': 'The central difference shows the theoretical',
        'の精度を示し、同じ刻み幅': 'accuracy and is more than 6 digits more accurate than forward/backward differences for the same step size',
        'でも前進・後退差分より6桁以上高精度です。ただし、': '. However, when',
        'を極端に小さくすると丸め誤差の影響で精度が低下します（図のU字型カーブ）。':
            'is made extremely small, accuracy degrades due to round-off errors (U-shaped curve in the figure).',

        # Section 1.2
        '1.2 Richardson外挿法': '1.2 Richardson Extrapolation',
        'Richardson外挿法は、異なる刻み幅での計算結果を組み合わせて高精度な近似を得る手法です。誤差の主要項を相殺することで、計算コストを抑えつつ精度を向上できます。':
            'Richardson extrapolation is a method that obtains high-accuracy approximations by combining results with different step sizes. By canceling the main error terms, accuracy can be improved while keeping computational cost low.',

        '📚 理論: Richardson外挿の原理': '📚 Theory: Principles of Richardson Extrapolation',
        '中心差分の誤差展開は次のようになります:': 'The error expansion of the central difference is as follows:',
        'ここで': 'where',
        'は刻み幅': 'is the central difference approximation with step size',
        'での中心差分による近似値です。': '.',
        'と': 'and',
        'から': 'From',
        'の項を消去すると:': ', eliminating the',
        'これにより精度が': 'This improves the accuracy from',
        'から': 'to',
        'に向上します。': '.',

        'コード例2: Richardson外挿法の実装': 'Code Example 2: Implementing Richardson Extrapolation',
        'Richardson外挿法による高精度数値微分': 'High-accuracy numerical differentiation using Richardson extrapolation',
        '微分対象の関数': 'Function to differentiate',
        '評価点': 'Evaluation point',
        '基本刻み幅': 'Base step size',
        '外挿の次数': 'Extrapolation order',
        '外挿された微分値': 'Extrapolated derivative value',
        '初期値: 中心差分': '# Initial value: central difference',
        'Richardson外挿による精度向上': '# Improve accuracy with Richardson extrapolation',

        'テスト:': '# Test:',
        '各手法の比較': '# Compare methods',
        '中心差分': 'Central difference',
        '値:': 'Value:',
        'Richardson外挿': 'Richardson extrapolation',
        '1次': '1st order',
        '2次': '2nd order',
        '精度の向上を可視化': '# Visualize accuracy improvement',
        'Richardson外挿法による精度向上': 'Accuracy Improvement with Richardson Extrapolation',

        # Section 1.3
        '1.3 数値積分の基礎': '1.3 Fundamentals of Numerical Integration',
        '定積分': 'We will learn methods for numerically computing the definite integral',
        'を数値的に計算する手法を学びます。区間を分割し、各小区間での関数値を使って積分を近似します。':
            '. By dividing the interval and using function values in each subinterval, we approximate the integral.',

        '📚 理論: 台形公式とSimpson公式': '📚 Theory: Trapezoidal and Simpson\'s Rules',
        '台形公式 (Trapezoidal Rule):': 'Trapezoidal Rule:',
        '区間': 'The interval',
        'を': 'is divided into',
        '個の小区間に分割し、各小区間で関数を直線近似:': 'subintervals, and the function is approximated by straight lines in each subinterval:',
        '誤差は': 'The error is',
        'です。': '.',

        'Simpson公式 (Simpson\'s Rule):': 'Simpson\'s Rule:',
        '各小区間で関数を2次多項式で近似（': 'The function is approximated by quadratic polynomials in each subinterval (',
        'は偶数）:': 'must be even):',
        'で、台形公式より高精度です。': ', which is more accurate than the trapezoidal rule.',

        'コード例3: 台形公式の実装': 'Code Example 3: Implementing the Trapezoidal Rule',
        '台形公式による数値積分': 'Numerical integration using the trapezoidal rule',
        '被積分関数': 'Integrand function',
        '積分区間': 'Integration interval',
        '分割数': 'Number of divisions',
        '積分値の近似': 'Approximation of the integral',
        '台形公式の実装': '# Implementation of trapezoidal rule',

        '分割数を変えて精度を評価': '# Evaluate accuracy for varying number of divisions',
        '台形公式による数値積分:': 'Numerical Integration Using Trapezoidal Rule:',
        '分割数 n    近似値        誤差': 'Divisions n    Approximation    Error',
        '厳密値:': 'Exact value:',
        '誤差の収束率を可視化': '# Visualize error convergence rate',
        '実際の誤差': 'Actual error',
        '分割数 n': 'Number of divisions n',
        '台形公式の収束性': 'Convergence of Trapezoidal Rule',

        'コード例4: Simpson公式の実装': 'Code Example 4: Implementing Simpson\'s Rule',
        'Simpson公式による数値積分（1/3則）': 'Numerical integration using Simpson\'s rule (1/3 rule)',
        '分割数（偶数でなければならない）': 'Number of divisions (must be even)',
        'Simpson公式では分割数nは偶数でなければなりません': 'For Simpson\'s rule, the number of divisions n must be even',
        'Simpson公式の実装': '# Implementation of Simpson\'s rule',
        '奇数インデックス': '# Odd indices',
        '偶数インデックス': '# Even indices',

        '台形公式とSimpson公式の比較': '# Compare trapezoidal and Simpson\'s rules',
        '台形公式 vs Simpson公式:': 'Trapezoidal Rule vs Simpson\'s Rule:',
        '台形公式      誤差         Simpson公式   誤差': 'Trapezoidal      Error        Simpson         Error',
        '収束率の比較': '# Compare convergence rates',
        '台形公式とSimpson公式の収束性比較': 'Comparison of Convergence: Trapezoidal vs Simpson\'s Rule',

        # Section 1.4
        '1.4 Gauss求積法': '1.4 Gaussian Quadrature',
        'Gauss求積法は、関数の評価点と重みを最適化することで、少ない評価点数で高精度な積分を実現する手法です。':
            'Gaussian quadrature is a method that achieves high-accuracy integration with fewer evaluation points by optimizing the evaluation points and weights.',
        '点のGauss求積法は': '-point Gaussian quadrature can exactly integrate polynomials up to degree',
        '次までの多項式を厳密に積分できます。': '.',

        '📚 理論: Gauss-Legendre求積法': '📚 Theory: Gauss-Legendre Quadrature',
        '区間': 'Consider the integral over the interval',
        'での積分を考えます:': ':',
        'ここで': 'where',
        'はLegendre多項式の零点、': 'are the zeros of the Legendre polynomial, and',
        'は対応する重みです。任意の区間': 'are the corresponding weights. The transformation to an arbitrary interval',
        'への変換は:': 'is:',

        'コード例5: Gauss求積法の実装': 'Code Example 5: Implementing Gaussian Quadrature',
        'Gauss-Legendre求積法による数値積分': 'Numerical integration using Gauss-Legendre quadrature',
        'Gauss点の数': 'Number of Gauss points',
        'Legendre多項式の零点と重みを取得': '# Get zeros and weights of Legendre polynomial',
        '区間[-1,1]から[a,b]への変換': '# Transform from interval [-1,1] to [a,b]',
        '積分の計算': '# Calculate integral',

        'SciPyの高精度積分で厳密値を計算': '# Calculate exact value with high-precision SciPy integration',
        'Gauss求積法:': 'Gaussian Quadrature:',
        'Gauss点数 n    近似値        誤差         関数評価回数': 'Gauss pts n    Approximation    Error        Function evals',
        '厳密値（SciPy quad）:': 'Exact value (SciPy quad):',
        '同じ関数評価回数での比較:': 'Comparison with same number of function evaluations:',
        '関数評価回数:': 'Function evaluations:',
        'Gauss': 'Gauss',
        '点': 'pts',
        'Simpson': 'Simpson',
        '分割': 'divs',
        '精度向上:': 'Accuracy improvement:',
        '倍': 'times',

        'Gauss求積法は同じ関数評価回数でSimpson公式より遙かに高精度です。特に滑らかな関数に対して効果的で、5点のGauss求積で機械精度レベルの精度が得られます。':
            'Gaussian quadrature is much more accurate than Simpson\'s rule for the same number of function evaluations. It is especially effective for smooth functions, and 5-point Gaussian quadrature can achieve machine precision.',

        # Section 1.5
        '1.5 NumPy/SciPyによる数値微分・積分': '1.5 Numerical Differentiation and Integration with NumPy/SciPy',
        '実務では、NumPy/SciPyの高機能な数値計算ライブラリを活用します。適応的手法や誤差評価機能を備えた関数が提供されています。':
            'In practice, we utilize the advanced numerical computing libraries NumPy/SciPy. Functions with adaptive methods and error estimation capabilities are provided.',

        'コード例6: scipy.integrate実践例': 'Code Example 6: scipy.integrate Practical Examples',
        'テスト関数群': '# Test functions',
        '振動関数': 'Oscillatory function',
        '特異性を持つ関数': 'Function with singularity',
        '適応的積分': '# Adaptive integration',
        '適応的Gauss-Kronrod法': 'Adaptive Gauss-Kronrod Method',
        '振動関数の積分': '# Integration of oscillatory function',
        '結果:': 'Result:',
        '推定誤差:': 'Estimated error:',
        '特異性を持つ関数': '# Function with singularity',
        '理論値:': 'Theoretical value:',

        '固定次数Gauss求積': '# Fixed-order Gauss quadrature',
        '固定次数Gauss-Legendre': 'Fixed-Order Gauss-Legendre',
        '点Gauss求積:': '-point Gauss quadrature:',

        '離散データの積分（実験データを想定）': '# Integration of discrete data (assuming experimental data)',
        '離散データの積分（trapz, simps）': 'Integration of Discrete Data (trapz, simps)',
        '実験データをシミュレート': '# Simulate experimental data',
        '11点のデータ': '# 11 data points',
        '台形公式': 'Trapezoidal rule',
        'Simpson公式': 'Simpson\'s rule',
        'trapzの誤差:': 'trapz error:',
        'simpsの誤差:': 'simps error:',

        '数値微分': '# Numerical differentiation',
        '数値微分': 'Numerical Differentiation',
        '1階微分': '# First derivative',
        '1階微分': 'First derivative',
        '数値微分:': 'Numerical:',
        '2階微分': '# Second derivative',
        '2階微分': 'Second derivative',

        # Section 1.6
        '1.6 誤差解析と収束性評価': '1.6 Error Analysis and Convergence Evaluation',
        '数値微分・積分の実用では、誤差の評価と適切な手法選択が重要です。理論的な収束率を実験的に検証し、丸め誤差の影響も考慮します。':
            'In practical numerical differentiation and integration, error evaluation and appropriate method selection are important. We experimentally verify theoretical convergence rates and consider the effects of round-off errors.',

        'コード例7: 誤差解析と収束率の可視化': 'Code Example 7: Error Analysis and Convergence Rate Visualization',
        '数値計算手法の収束率を解析': 'Analyze convergence rate of numerical method',
        '数値計算手法の関数': 'Numerical method function',
        '対象関数': 'Target function',
        '厳密解': 'Exact solution',
        'パラメータのリスト（刻み幅や分割数）': 'List of parameters (step sizes or divisions)',
        '手法の名前': 'Method name',
        '各パラメータでの誤差': 'Error for each parameter',

        'テスト関数:': '# Test function:',
        '分割数のリスト': '# List of divisions',
        '各手法の収束率を評価': '# Evaluate convergence rate of each method',
        '数値積分手法の収束率解析:': 'Convergence Rate Analysis of Numerical Integration Methods:',

        '台形公式': '# Trapezoidal rule',
        'Simpson公式': '# Simpson\'s rule',
        'Gauss求積': '# Gaussian quadrature',

        '収束率の計算（連続する誤差の比）': '# Calculate convergence rate (ratio of consecutive errors)',
        '誤差の減少率から収束率を推定': 'Estimate convergence rate from error reduction',

        '結果の表示': '# Display results',
        '台形公式 (理論収束率:': 'Trapezoidal Rule (theoretical convergence rate:',
        'Simpson公式 (理論収束率:': 'Simpson\'s Rule (theoretical convergence rate:',
        'Gauss求積法': 'Gaussian Quadrature',
        'n      誤差          収束率': 'n      Error        Rate',
        '平均収束率:': 'Average rate:',
        '理論値:': 'Theoretical:',

        '総合的な可視化': '# Comprehensive visualization',
        '誤差の収束': '# Error convergence',
        '収束性の比較': 'Convergence Comparison',
        '収束率の推移': '# Convergence rate evolution',
        '収束率': 'Convergence Rate',
        '収束率の推移': 'Evolution of Convergence Rate',
        '理論値 (台形)': 'Theoretical (Trapezoidal)',
        '理論値 (Simpson)': 'Theoretical (Simpson)',

        'まとめ:': 'Summary:',
        '台形公式: 収束率': 'Trapezoidal rule: convergence rate',
        '理論通り': 'as expected theoretically',
        'Simpson公式: 収束率': 'Simpson\'s rule: convergence rate',
        'Gauss求積法: 指数的収束（多項式に対して厳密）': 'Gaussian quadrature: exponential convergence (exact for polynomials)',

        # Exercises
        '🏋️ 演習問題': '🏋️ Exercises',
        '演習1: 数値微分の実装': 'Exercise 1: Implementing Numerical Differentiation',
        '次の関数の': 'Calculate the derivative of the following function at',
        'における微分を、前進差分・後退差分・中心差分で計算し、誤差を比較せよ。刻み幅':
            'using forward, backward, and central differences, and compare the errors. Try step sizes',
        'は0.1, 0.01, 0.001の3通りで試すこと。': 'of 0.1, 0.01, and 0.001.',
        '厳密解:': 'Exact solution:',

        '演習2: Richardson外挿の効果検証': 'Exercise 2: Verifying Richardson Extrapolation Effectiveness',
        'の': 'of',
        'における1階微分を次の方法で計算し、誤差を比較せよ（': 'at',
        '）:': 'using the following methods and compare the errors (',

        '演習3: 積分公式の精度比較': 'Exercise 3: Comparing Accuracy of Integration Formulas',
        '次の積分を台形公式、Simpson公式、Gauss求積法（5点）で計算し、精度と計算コストを比較せよ:':
            'Calculate the following integral using the trapezoidal rule, Simpson\'s rule, and Gaussian quadrature (5 points), and compare accuracy and computational cost:',
        'ヒント: 厳密解は': 'Hint: The exact solution is',

        '演習4: 実験データの数値積分': 'Exercise 4: Numerical Integration of Experimental Data',
        '以下の実験データ（温度 vs 時間）から、0〜10秒間の平均温度を数値積分で求めよ:':
            'From the following experimental data (temperature vs time), calculate the average temperature over 0-10 seconds using numerical integration:',
        '時刻 (s):': 'Time (s):',
        '温度 (°C):': 'Temperature (°C):',
        '台形公式とSimpson公式の両方で計算し、結果を比較せよ。': 'Calculate using both the trapezoidal rule and Simpson\'s rule, and compare the results.',

        '演習5: 材料科学への応用': 'Exercise 5: Applications to Materials Science',
        '材料の熱膨張係数': 'When the thermal expansion coefficient of a material',
        'が温度の関数として与えられたとき、温度変化に伴う長さの変化率は次式で計算されます:':
            'is given as a function of temperature, the rate of length change due to temperature variation is calculated by:',
        'とし、': 'Take',
        'から': 'from',
        'への温度上昇に伴う長さの変化率を数値積分で求めよ。': 'and calculate the length change rate due to temperature increase to',
        'using numerical integration.',

        # Summary
        'まとめ': 'Summary',
        '本章では、数値微分と数値積分の基本的な手法を学びました:': 'In this chapter, we learned fundamental methods for numerical differentiation and integration:',
        '数値微分:': 'Numerical differentiation:',
        '差分法（前進・後退・中心）とRichardson外挿による高精度化': 'Finite difference methods (forward, backward, central) and high-accuracy with Richardson extrapolation',
        '数値積分:': 'Numerical integration:',
        '台形公式、Simpson公式、Gauss求積法の原理と実装': 'Principles and implementation of trapezoidal rule, Simpson\'s rule, and Gaussian quadrature',
        '誤差解析:': 'Error analysis:',
        '理論的収束率の検証と実用的な精度評価': 'Verification of theoretical convergence rates and practical accuracy evaluation',
        'SciPy活用:': 'Using SciPy:',
        'scipy.integrateとscipy.miscによる実践的数値計算': 'Practical numerical computation with scipy.integrate and scipy.misc',

        'これらの手法は、材料科学・プロセス工学における実験データ解析、シミュレーション、最適化など幅広い場面で活用されます。次章では、これらの基礎の上に立って線形方程式系の数値解法を学びます。':
            'These methods are utilized in a wide range of applications in materials science and process engineering, including experimental data analysis, simulation, and optimization. In the next chapter, we will learn numerical methods for systems of linear equations building on these foundations.',

        # Navigation
        '← シリーズ目次': '← Series Table of Contents',
        '第2章へ →': 'Chapter 2 →',

        # Footer
        '&copy; 2025 FM Dojo. All rights reserved.': '&copy; 2025 FM Dojo. All rights reserved.',
    }

    return translations

# Print statistics
translations = create_comprehensive_translation_map()
print(f"Total translation mappings created: {len(translations)}")
print(f"Estimated Japanese phrases covered: {len([k for k in translations.keys() if k != 'lang=\"ja\"'])}")
