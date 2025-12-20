---
title: "第1章: 格子力学の理論"
subtitle: "Born-von Kármán境界条件、動的行列、対称性解析"
series: "フォノン物理学（中級）"
chapter: 1
difficulty: "中級"
reading_time: "約3時間"
---

[🌐 EN](../../../en/MS/phonon-intermediate/chapter-1.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン物理学（中級)](index.md) > 第1章

# 第1章: 格子力学の理論

**Born-von Kármán境界条件、動的行列、対称性解析**

## 学習目標

- Born-von Kármán周期境界条件を理解し、その物理的意味を説明できる
- 一般的な3次元結晶の運動方程式を導出できる
- 動的行列 \\(D(\mathbf{q})\\) の定式化を理解し、その性質を説明できる
- 力定数の対称性と並進不変性の関係を理解できる
- 音響和則の物理的意味と数学的導出を理解できる
- 点群対称性と空間群操作がフォノンに与える影響を説明できる
- Pythonで動的行列を構築し、フォノン分散を計算できる

## 導入

入門編では調和近似の下での単純な1次元モデルと分散関係を学びました。本章では、その理論を実際の3次元結晶に適用するための厳密な定式化を展開します。Born-von Kármán境界条件の導入から始め、一般的な多原子基底を持つ結晶の運動方程式を導出し、動的行列の固有値問題としてフォノン問題を定式化します。

特に重要なのは、結晶対称性がフォノンの性質にどのように反映されるかという点です。力定数の対称性、音響和則、点群・空間群操作とフォノン固有モードの関係を詳細に議論します。これらの概念は、第一原理計算によるフォノン計算の理論的基礎となります。

## 1.1 Born-von Kármán周期境界条件

### 1.1.1 境界条件の必要性

結晶中のフォノンを記述するには、無限周期系を有限系で近似する境界条件が必要です。Born-von Kármán (BvK) 境界条件は、結晶を \\(N_1 \times N_2 \times N_3\\) 個の単位格子からなる有限系として扱い、周期境界条件を課すものです。

**定義: Born-von Kármán境界条件**

格子ベクトル \\(\mathbf{R}_n = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3\\) で指定される格子点での変位 \\(\mathbf{u}(\mathbf{R}_n)\\) に対して、以下の周期性を課す：

\\[
\mathbf{u}(\mathbf{R}_n + N_i\mathbf{a}_i) = \mathbf{u}(\mathbf{R}_n), \quad i = 1, 2, 3
\\]

ここで、\\(\mathbf{a}_i\\) は原始格子ベクトル、\\(N_i\\) は各方向の繰り返し数です。

### 1.1.2 許される波数ベクトル

BvK境界条件の下で、平面波解 \\(\mathbf{u}(\mathbf{R}_n) \propto e^{i\mathbf{q}\cdot\mathbf{R}_n}\\) が境界条件を満たすためには、波数ベクトル \\(\mathbf{q}\\) は以下の離散的な値のみを取ります：

\\[
\mathbf{q} = \frac{m_1}{N_1}\mathbf{b}_1 + \frac{m_2}{N_2}\mathbf{b}_2 + \frac{m_3}{N_3}\mathbf{b}_3
\\]

ここで、\\(\mathbf{b}_i\\) は逆格子ベクトル（\\(\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\delta_{ij}\\)）、\\(m_i\\) は整数です。熱力学的極限 \\(N_i \to \infty\\) において、\\(\mathbf{q}\\) は連続的な値を取るようになり、第一ブリルアンゾーン内で密に分布します。

**物理的意味**

BvK境界条件は、表面効果を無視し、バルク結晶の性質に焦点を当てるための数学的技巧です。\\(N_i\\) が十分大きければ、境界の影響は無視でき、無限結晶の良い近似となります。第一原理計算では、通常数百から数千原子のスーパーセルにBvK境界条件を適用します。

### 1.1.3 状態密度

BvK境界条件により、許される \\(\mathbf{q}\\) 点の密度は：

\\[
\rho(\mathbf{q}) = \frac{N_1 N_2 N_3}{(2\pi)^3} V_\text{cell} = \frac{V}{(2\pi)^3}
\\]

ここで、\\(V = N_1 N_2 N_3 V_\text{cell}\\) は結晶全体の体積、\\(V_\text{cell}\\) は単位格子の体積です。熱力学的極限では、\\(\mathbf{q}\\) 空間での和は積分に置き換えられます：

\\[
\sum_\mathbf{q} \to \frac{V}{(2\pi)^3} \int d^3q
\\]

## 1.2 3次元結晶の運動方程式

### 1.2.1 多原子基底結晶の記述

一般的な結晶は、各単位格子内に \\(r\\) 個の原子（原子基底）を含みます。格子点 \\(\mathbf{R}_n\\) にある単位格子内の \\(\kappa\\) 番目の原子の位置は：

\\[
\mathbf{r}_{n\kappa} = \mathbf{R}_n + \boldsymbol{\tau}_\kappa
\\]

ここで、\\(\boldsymbol{\tau}_\kappa\\) は単位格子内での原子 \\(\kappa\\) の平衡位置です。この原子の変位を \\(\mathbf{u}_{n\kappa}(t)\\) とし、質量を \\(M_\kappa\\) とします。

**例: ダイヤモンド構造**

ダイヤモンド構造（Si, Geなど）は、面心立方格子に2原子基底を持ちます：

\\[
\boldsymbol{\tau}_1 = (0, 0, 0), \quad \boldsymbol{\tau}_2 = \frac{a}{4}(1, 1, 1)
\\]

ここで、\\(a\\) は格子定数です。したがって、単位格子あたり6つの自由度（2原子×3方向）があり、6本のフォノンブランチ（3本の音響モード + 3本の光学モード）が存在します。

### 1.2.2 調和近似とポテンシャルエネルギー

調和近似において、結晶のポテンシャルエネルギー \\(\Phi\\) は原子変位の2次まで展開されます：

\\[
\Phi = \Phi_0 + \sum_{n\kappa\alpha} \frac{\partial \Phi}{\partial u_{n\kappa\alpha}} u_{n\kappa\alpha} + \frac{1}{2}\sum_{\substack{n\kappa\alpha \\ n'\kappa'\alpha'}} \frac{\partial^2 \Phi}{\partial u_{n\kappa\alpha} \partial u_{n'\kappa'\alpha'}} u_{n\kappa\alpha} u_{n'\kappa'\alpha'}
\\]

ここで、\\(\alpha, \alpha'\\) はデカルト座標成分（\\(x, y, z\\)）を表します。平衡位置では第1項（線形項）はゼロであり、第2項（2次項）のみが残ります。

### 1.2.3 力定数行列

**定義: 力定数**

2次の力定数を以下のように定義します：

\\[
\Phi_{\alpha\alpha'}(n\kappa, n'\kappa') = \frac{\partial^2 \Phi}{\partial u_{n\kappa\alpha} \partial u_{n'\kappa'\alpha'}}
\\]

これは、格子点 \\(n\\) の原子 \\(\kappa\\) の \\(\alpha\\) 方向変位と、格子点 \\(n'\\) の原子 \\(\kappa'\\) の \\(\alpha'\\) 方向変位に対するポテンシャルの2階微分です。

並進対称性により、力定数は格子点の差 \\(\mathbf{R}_{nn'} = \mathbf{R}_n - \mathbf{R}_{n'}\\) のみに依存します：

\\[
\Phi_{\alpha\alpha'}(n\kappa, n'\kappa') = \Phi_{\alpha\alpha'}(\mathbf{R}_{nn'}, \kappa, \kappa')
\\]

### 1.2.4 運動方程式

ニュートンの運動方程式は以下のようになります：

\\[
M_\kappa \frac{\partial^2 u_{n\kappa\alpha}}{\partial t^2} = -\sum_{n'\kappa'\alpha'} \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') u_{n'\kappa'\alpha'}
\\]

これは \\(3r \times N\\) 個の連立微分方程式系（\\(r\\) は基底原子数、\\(N = N_1 N_2 N_3\\) は単位格子数）です。

### 1.2.5 平面波解

BvK境界条件の下で、平面波形式の解を仮定します：

\\[
u_{n\kappa\alpha}(t) = \frac{1}{\sqrt{M_\kappa}} e_{\kappa\alpha}(\mathbf{q}, j) e^{i(\mathbf{q}\cdot\mathbf{R}_n - \omega_j(\mathbf{q})t)}
\\]

ここで、\\(e_{\kappa\alpha}(\mathbf{q}, j)\\) は偏極ベクトル（固有ベクトル）、\\(\omega_j(\mathbf{q})\\) は角振動数（固有値）、\\(j\\) はブランチインデックス（\\(j = 1, 2, \ldots, 3r\\)）です。質量因子 \\(1/\sqrt{M_\kappa}\\) は後の計算を簡略化します。

## 1.3 動的行列の定式化

### 1.3.1 動的行列の導出

平面波解を運動方程式に代入すると、以下の固有値問題が得られます：

\\[
\omega^2 e_{\kappa\alpha}(\mathbf{q}, j) = \sum_{\kappa'\alpha'} D_{\kappa\alpha, \kappa'\alpha'}(\mathbf{q}) e_{\kappa'\alpha'}(\mathbf{q}, j)
\\]

**定義: 動的行列**

動的行列 \\(D(\mathbf{q})\\) は \\(3r \times 3r\\) エルミート行列で、以下のように定義されます：

\\[
D_{\kappa\alpha, \kappa'\alpha'}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{n'} \Phi_{\alpha\alpha'}(0\kappa, n'\kappa') e^{i\mathbf{q}\cdot\mathbf{R}_{n'}}
\\]

ここで、和は単位格子 \\(n'\\) にわたって行われ、\\(\Phi_{\alpha\alpha'}(0\kappa, n'\kappa')\\) は原点の単位格子（\\(n=0\\)）内の原子 \\(\kappa\\) と、格子点 \\(n'\\) の原子 \\(\kappa'\\) 間の力定数です。

### 1.3.2 動的行列の性質

**定理: 動的行列の性質**

1. **エルミート性:** \\(D_{\kappa\alpha, \kappa'\alpha'}(\mathbf{q}) = D^*_{\kappa'\alpha', \kappa\alpha}(\mathbf{q})\\)
2. **実数固有値:** エルミート行列なので、\\(\omega^2\\) は実数
3. **正定値性:** 安定な結晶では \\(\omega^2 \geq 0\\)（すべての固有値は非負）
4. **時間反転対称性:** \\(D(\mathbf{q}) = D^*(-\mathbf{q})\\)、したがって \\(\omega_j(\mathbf{q}) = \omega_j(-\mathbf{q})\\)

**証明（エルミート性）:**

力定数の対称性 \\(\Phi_{\alpha\alpha'}(0\kappa, n'\kappa') = \Phi_{\alpha'\alpha}(n'\kappa', 0\kappa)\\) と、並進対称性 \\(\Phi_{\alpha'\alpha}(n'\kappa', 0\kappa) = \Phi_{\alpha'\alpha}(0\kappa', -n'\kappa)\\) を用いると：

\\[
\begin{align}
D^*_{\kappa'\alpha', \kappa\alpha}(\mathbf{q}) &= \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{n'} \Phi_{\alpha'\alpha}(0\kappa', n'\kappa) e^{-i\mathbf{q}\cdot\mathbf{R}_{n'}} \\\\
&= \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{n''} \Phi_{\alpha\alpha'}(0\kappa, n''\kappa') e^{i\mathbf{q}\cdot\mathbf{R}_{n''}} \\\\
&= D_{\kappa\alpha, \kappa'\alpha'}(\mathbf{q})
\end{align}
\\]

### 1.3.3 固有値問題の行列形式

動的行列 \\(D(\mathbf{q})\\) の固有値問題は以下のように書けます：

\\[
D(\mathbf{q}) \mathbf{e}(\mathbf{q}, j) = \omega_j^2(\mathbf{q}) \mathbf{e}(\mathbf{q}, j)
\\]

または、行列式形式で：

\\[
\det|D(\mathbf{q}) - \omega^2 I| = 0
\\]

この方程式は \\(\omega^2\\) に関する \\(3r\\) 次の代数方程式であり、\\(3r\\) 個の固有値 \\(\omega_j^2(\mathbf{q})\\) （\\(j = 1, 2, \ldots, 3r\\)）を与えます。これらがフォノンの \\(3r\\) 本のブランチに対応します。

**例: 1次元単原子鎖（復習）**

入門編で学んだ1次元単原子鎖の場合、動的行列は \\(1 \times 1\\) 行列（スカラー）です：

\\[
D(q) = \frac{1}{M} \sum_{n'} \Phi(0, n') e^{iqn'a}
\\]

最近接相互作用のみを考慮すると（\\(\Phi(0, \pm1) = C\\)、\\(\Phi(0, 0) = -2C\\)）：

\\[
D(q) = \frac{2C}{M}(1 - \cos qa)
\\]

これより \\(\omega^2(q) = D(q) = \frac{2C}{M}(1 - \cos qa)\\)、すなわち \\(\omega(q) = 2\sqrt{C/M} |\sin(qa/2)|\\) が得られます。

## 1.4 力定数とその性質

### 1.4.1 力定数の対称性

**力定数に課される対称性条件**

1. **並進対称性:**
   \\[
   \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') = \Phi_{\alpha\alpha'}(\mathbf{R}_{nn'}, \kappa, \kappa')
   \\]

2. **逆空間対称性:**
   \\[
   \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') = \Phi_{\alpha'\alpha}(n'\kappa', n\kappa)
   \\]

3. **点群対称性:** 点群操作 \\(R\\) に対して
   \\[
   \Phi_{\alpha\alpha'}(R\mathbf{r}, R\kappa, R\mathbf{r}', R\kappa') = R_{\alpha\beta} R_{\alpha'\beta'} \Phi_{\beta\beta'}(\mathbf{r}, \kappa, \mathbf{r}', \kappa')
   \\]
   ここで \\(R_{\alpha\beta}\\) は回転行列の成分です。

### 1.4.2 並進不変性と音響和則

**並進不変性の原理**

結晶全体を剛体として一様に移動させても、ポテンシャルエネルギーは変化しません。これは以下の条件を課します：

\\[
\sum_{n'\kappa'} \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') = 0
\\]

これを**音響和則（acoustic sum rule）**または**原子サイト和則**と呼びます。

**証明:**

すべての原子が同じ変位 \\(\mathbf{u}_0\\) を持つ場合（\\(u_{n\kappa\alpha} = u_{0\alpha}\\) for all \\(n, \kappa\\)）、ポテンシャルエネルギーの変化はゼロでなければなりません：

\\[
\Delta\Phi = \frac{1}{2}\sum_{\substack{n\kappa\alpha \\ n'\kappa'\alpha'}} \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') u_{0\alpha} u_{0\alpha'} = 0
\\]

これがすべての \\(u_{0\alpha}\\) に対して成り立つには、各 \\(n, \kappa, \alpha\\) について \\(\sum_{n'\kappa'\alpha'} \Phi_{\alpha\alpha'}(n\kappa, n'\kappa') u_{0\alpha'} = 0\\) が必要であり、さらにこれがすべての \\(u_{0\alpha'}\\) に対して成り立つには音響和則が満たされなければなりません。

**音響和則の帰結**

音響和則は、\\(\mathbf{q} = 0\\) で少なくとも3つのゼロ固有値（音響モード）が存在することを保証します：

\\[
\omega_j(\mathbf{q} \to 0) \to 0, \quad j = 1, 2, 3
\\]

これらは結晶全体の剛体並進運動に対応します。

**証明:**

\\(\mathbf{q} = 0\\) での動的行列は：

\\[
D_{\kappa\alpha, \kappa'\alpha'}(0) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}} \sum_{n'} \Phi_{\alpha\alpha'}(0\kappa, n'\kappa')
\\]

音響和則により、\\(\mathbf{e}_\alpha = (0, \ldots, 0, 1, 0, \ldots, 0)\\) （\\(\kappa\alpha\\) 成分のみ1で他はゼロ）に対して：

\\[
\sum_{\kappa'\alpha'} D_{\kappa\alpha, \kappa'\alpha'}(0) e_{\kappa'\alpha} = 0
\\]

したがって、一様な変位パターン \\(e_{\kappa\alpha} = \delta_{\alpha,x}\\), \\(\delta_{\alpha,y}\\), \\(\delta_{\alpha,z}\\) （すべての原子が同じ方向に変位）は、\\(\omega^2 = 0\\) の固有ベクトルです。

### 1.4.3 力定数の実空間カットオフ

実際の材料では、力定数 \\(\Phi_{\alpha\alpha'}(0\kappa, n'\kappa')\\) は原子間距離が大きくなると急速に減衰します。典型的には：

- **イオン結晶:** クーロン相互作用により長距離（ただしEwald和法で扱う）
- **共有結合結晶:** 第2〜第4近接まで（Si, C など）
- **金属:** 第10近接以上（スクリーニング効果による）

第一原理計算では、有限範囲の力定数を計算し、それを用いて動的行列をフーリエ変換します。

## 1.5 点群対称性とフォノン

### 1.5.1 結晶点群とその表現

結晶の点群 \\(G\\) は、原点を固定して結晶を不変に保つ回転・鏡映操作の集合です。各点群操作 \\(R \in G\\) は、動的行列に以下の変換を施します：

\\[
D(R\mathbf{q}) = T(R) D(\mathbf{q}) T^\dagger(R)
\\]

ここで、\\(T(R)\\) は点群操作に対応する \\(3r \times 3r\\) ユニタリ行列です。

### 1.5.2 対称性の高い点とモード分類

高対称点（\\(\Gamma, X, L, K\\) など）では、その点を不変に保つ対称操作（点の安定化群または小群）が存在します。この小群の既約表現によって、フォノンモードは分類されます。

**例: 立方晶の \\(\Gamma\\) 点（\\(\mathbf{q} = 0\\)）**

単純立方格子（点群 \\(O_h\\)）の \\(\Gamma\\) 点では、すべての点群操作が波数ベクトル \\(\mathbf{q} = 0\\) を不変に保ちます。\\(O_h\\) 群の既約表現は：

- \\(A_{1g}\\): 完全対称（1次元）
- \\(E_g\\): 2重縮退（2次元）
- \\(T_{1g}, T_{2g}\\): 3重縮退（各3次元）
- \\(A_{2u}, E_u, T_{1u}, T_{2u}\\): 対応する奇パリティ表現

音響モードは \\(T_{1u}\\) 表現（3重縮退の並進）に属します。

### 1.5.3 縮退の解除

一般的な \\(\mathbf{q}\\) 点では、対称性が低下し、高対称点で縮退していたモードが分裂します。例えば、\\(\Gamma\\) 点での \\(T_{2g}\\) 3重縮退モードは、\\(X\\) 点に向かう \\(\Delta\\) 線上では安定化群が \\(D_{4h}\\) に低下し、\\(E_g\\) (2重) + \\(A_{2u}\\) (1重) に分裂する可能性があります。

### 1.5.4 選択則への応用

対称性分析は、光学測定における選択則を予測するために重要です：

- **赤外活性:** \\(T_{1u}\\) 表現のモード（双極子モーメント変化）
- **ラマン活性:** 対称テンソル表現 (\\(A_{1g}, E_g, T_{2g}\\)) のモード
- **非活性:** \\(T_{1g}, T_{2u}\\) など（特定の対称性の場合）

## 1.6 空間群操作とフォノン対称性

### 1.6.1 空間群と非シンモルフィック操作

空間群は、並進と点群操作の組み合わせです。非シンモルフィック空間群では、らせん軸や滑り鏡面のように、回転・鏡映と並進が結合した操作が存在します。

**空間群操作の一般形**

空間群操作 \\(\{R|\mathbf{t}\}\\) は以下のように作用します：

\\[
\{R|\mathbf{t}\}\mathbf{r} = R\mathbf{r} + \mathbf{t}
\\]

ここで、\\(R\\) は点群操作（回転行列）、\\(\mathbf{t}\\) は並進ベクトルです。シンモルフィック空間群では \\(\mathbf{t}\\) は格子ベクトルのみですが、非シンモルフィックでは分数並進が含まれます。

### 1.6.2 動的行列への影響

空間群操作 \\(\{R|\mathbf{t}\}\\) は、動的行列に以下の変換を施します：

\\[
D(R\mathbf{q}) = e^{-i(R\mathbf{q})\cdot\mathbf{t}} T(R) D(\mathbf{q}) T^\dagger(R)
\\]

位相因子 \\(e^{-i(R\mathbf{q})\cdot\mathbf{t}}\\) が非シンモルフィック操作に由来します。

### 1.6.3 ブロッホの定理とフォノン

結晶の並進対称性により、フォノンの波動関数は以下のブロッホ形式を取ります：

\\[
u_{n\kappa\alpha}(\mathbf{q}, j) = e^{i\mathbf{q}\cdot\mathbf{R}_n} e_{\kappa\alpha}(\mathbf{q}, j)
\\]

ここで、\\(e_{\kappa\alpha}(\mathbf{q}, j)\\) は単位格子内で周期的な偏極ベクトルです。これは電子のブロッホ波動関数と同じ構造を持ちます。

### 1.6.4 既約表現とモードの指標

各 \\(\mathbf{q}\\) 点での動的行列の固有モードは、その点の小群（star of \\(\mathbf{q}\\)）の既約表現によって分類されます。Bilbao Crystallographic Server などのツールを使って、特定の空間群と \\(\mathbf{q}\\) 点に対する既約表現と互換性条件を調べることができます。

**実用上の重要性**

第一原理フォノン計算ソフトウェア（PHONOPY, Quantum ESPRESSO など）は、空間群対称性を自動的に考慮します。しかし、結果を解釈する際には、対称性分析の理解が不可欠です。特に、相転移点やトポロジカル性質を調べる際に重要です。

## 1.7 Pythonによる動的行列の構築と対角化

### 1.7.1 1次元2原子鎖の実装

理論を実践するため、1次元2原子鎖（異なる質量 \\(M_1, M_2\\)）のフォノン分散を計算してみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

def dynamical_matrix_1d_diatomic(q, M1, M2, C):
    """
    1次元2原子鎖の動的行列を構築する

    Parameters:
    -----------
    q : float
        波数（単位: 1/a、aは格子定数）
    M1, M2 : float
        原子1, 2の質量
    C : float
        バネ定数

    Returns:
    --------
    D : ndarray (2x2)
        動的行列
    """
    # 動的行列の成分（より正確な表式）
    D11 = C / M1
    D22 = C / M2
    D12 = -C / np.sqrt(M1 * M2) * np.exp(-1j * q)
    D21 = -C / np.sqrt(M1 * M2) * np.exp(1j * q)

    D = np.array([[D11, D12],
                  [D21, D22]])

    return D

def compute_phonon_dispersion(M1, M2, C, npoints=100):
    """
    フォノン分散関係を計算する

    Parameters:
    -----------
    M1, M2 : float
        質量
    C : float
        バネ定数
    npoints : int
        q点の数

    Returns:
    --------
    q_values : ndarray
        波数配列
    omega_acoustic : ndarray
        音響ブランチの振動数
    omega_optical : ndarray
        光学ブランチの振動数
    """
    # 第一ブリルアンゾーン: -π < q ≤ π
    q_values = np.linspace(-np.pi, np.pi, npoints)
    omega_acoustic = np.zeros(npoints)
    omega_optical = np.zeros(npoints)

    for i, q in enumerate(q_values):
        D = dynamical_matrix_1d_diatomic(q, M1, M2, C)
        eigenvalues = np.linalg.eigvalsh(D)  # エルミート行列の固有値

        # ω² が固有値なので、ω = sqrt(eigenvalues)
        omega_sq = eigenvalues
        omega = np.sqrt(np.abs(omega_sq))  # 数値誤差で負になる場合に対応

        # 昇順にソート（音響 < 光学）
        omega_acoustic[i] = omega[0]
        omega_optical[i] = omega[1]

    return q_values, omega_acoustic, omega_optical

# パラメータ設定
M1 = 1.0   # 軽い原子の質量（任意単位）
M2 = 2.0   # 重い原子の質量
C = 1.0    # バネ定数（任意単位）

# 計算実行
q, omega_ac, omega_op = compute_phonon_dispersion(M1, M2, C)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(q, omega_ac, 'b-', linewidth=2, label='Acoustic branch')
plt.plot(q, omega_op, 'r-', linewidth=2, label='Optical branch')
plt.xlabel(r'Wave vector $q$ (units of $\pi/a$)', fontsize=12)
plt.ylabel(r'Frequency $\omega$ (arb. units)', fontsize=12)
plt.title('Phonon Dispersion: 1D Diatomic Chain', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(-np.pi, np.pi)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('phonon_dispersion_1d_diatomic.png', dpi=150)
plt.show()

# q=0での振動数を確認（音響和則のチェック）
print(f"ω_acoustic(q=0) = {omega_ac[npoints//2]:.6f}")  # should be ~0
print(f"ω_optical(q=0) = {omega_op[npoints//2]:.6f}")

# q=πでの振動数を確認（ゾーン境界）
print(f"ω_acoustic(q=π) = {omega_ac[-1]:.6f}")
print(f"ω_optical(q=π) = {omega_op[-1]:.6f}")
```

### 1.7.2 3次元結晶への拡張

3次元結晶では、力定数テンソルを読み込み、各 \\(\mathbf{q}\\) 点で \\(3r \times 3r\\) 行列を構築します。以下は簡略化した実装例です：

```python
import numpy as np

class DynamicalMatrix3D:
    """3次元結晶の動的行列を扱うクラス"""

    def __init__(self, force_constants, masses, lattice_vectors):
        """
        Parameters:
        -----------
        force_constants : dict
            キー: (n, kappa, alpha, n', kappa', alpha')
            値: Φ_αα'(nκ, n'κ')
        masses : array
            各原子種の質量 [M_1, M_2, ..., M_r]
        lattice_vectors : ndarray (3, 3)
            原始格子ベクトル [a1, a2, a3]
        """
        self.fc = force_constants
        self.masses = np.array(masses)
        self.lattice = np.array(lattice_vectors)
        self.n_atoms = len(masses)

    def construct_D(self, q):
        """
        波数ベクトル q での動的行列を構築

        Parameters:
        -----------
        q : array-like (3,)
            波数ベクトル（逆格子単位）

        Returns:
        --------
        D : ndarray (3*n_atoms, 3*n_atoms)
            動的行列（エルミート）
        """
        size = 3 * self.n_atoms
        D = np.zeros((size, size), dtype=complex)

        q_cart = self.reciprocal_lattice @ q  # デカルト座標へ変換

        for key, phi in self.fc.items():
            n, kappa, alpha, n_p, kappa_p, alpha_p = key

            # 格子ベクトル R_n'
            R_n_p = n_p[0]*self.lattice[0] + \
                    n_p[1]*self.lattice[1] + \
                    n_p[2]*self.lattice[2]

            # 位相因子
            phase = np.exp(1j * np.dot(q_cart, R_n_p))

            # 動的行列の成分
            i = 3*kappa + alpha
            j = 3*kappa_p + alpha_p

            D[i, j] += phi / np.sqrt(self.masses[kappa] *
                                     self.masses[kappa_p]) * phase

        return D

    @property
    def reciprocal_lattice(self):
        """逆格子ベクトルを計算"""
        V = np.dot(self.lattice[0],
                   np.cross(self.lattice[1], self.lattice[2]))
        b1 = 2*np.pi * np.cross(self.lattice[1], self.lattice[2]) / V
        b2 = 2*np.pi * np.cross(self.lattice[2], self.lattice[0]) / V
        b3 = 2*np.pi * np.cross(self.lattice[0], self.lattice[1]) / V
        return np.array([b1, b2, b3])

    def solve(self, q):
        """
        固有値問題を解く

        Returns:
        --------
        omega : ndarray
            角振動数 ω (昇順)
        eigvecs : ndarray
            固有ベクトル（偏極ベクトル）
        """
        D = self.construct_D(q)
        eigenvalues, eigenvectors = np.linalg.eigh(D)

        # ω = sqrt(ω²)
        omega = np.sqrt(np.abs(eigenvalues))

        return omega, eigenvectors

# 使用例（仮想的な単純立方格子、単原子）
lattice = np.eye(3) * 5.0  # 5 Åの立方格子
masses = [28.0]  # Si原子（amu）

# 力定数の例（実際はDFTから計算）
force_constants = {
    # (n, κ, α, n', κ', α'): Φ値
    # 簡略化のため省略
}

# dm = DynamicalMatrix3D(force_constants, masses, lattice)
# omega, evecs = dm.solve(q=[0.5, 0.5, 0.0])  # X点
```

### 1.7.3 音響和則の確認

実装が正しいかを検証するため、\\(\mathbf{q} = 0\\) で音響モードの振動数がゼロになることを確認します：

```python
def verify_acoustic_sum_rule(dynamical_matrix, tolerance=1e-6):
    """
    音響和則を検証する

    Parameters:
    -----------
    dynamical_matrix : DynamicalMatrix3D
        動的行列オブジェクト
    tolerance : float
        許容誤差

    Returns:
    --------
    bool : 音響和則が満たされているか
    """
    q_gamma = np.array([0.0, 0.0, 0.0])
    omega, _ = dynamical_matrix.solve(q_gamma)

    # 最小の3つの固有値（音響モード）を確認
    acoustic_freqs = omega[:3]

    print("Frequencies at Γ point (q=0):")
    for i, freq in enumerate(omega):
        print(f"  Mode {i+1}: ω = {freq:.8f}")

    # 音響モードがゼロに近いか
    if np.all(acoustic_freqs < tolerance):
        print(f"\n✓ Acoustic sum rule satisfied (max ω = {acoustic_freqs.max():.2e})")
        return True
    else:
        print(f"\n✗ Acoustic sum rule violated (max ω = {acoustic_freqs.max():.2e})")
        return False

# verify_acoustic_sum_rule(dm)
```

## 1.8 第一原理計算との関連

### 1.8.1 密度汎関数摂動論（DFPT）

実際の材料のフォノンを計算するには、第一原理計算から力定数を得る必要があります。密度汎関数摂動論（Density Functional Perturbation Theory, DFPT）は、原子変位に対する電子系の応答を自己無撞着に計算し、力定数を直接求める手法です。

```
[結晶構造] → [DFT基底状態計算] → [DFPT: 動的行列計算] → [各q点でD_q計算]
→ [固有値問題を解く] → [ω_jq, e_jq] → [フォノン分散/状態密度/熱力学量]
```

### 1.8.2 凍結フォノン法

別のアプローチとして、**凍結フォノン法（frozen phonon method）**があります。これは、原子を有限変位させたスーパーセルでDFT計算を行い、力の変化から力定数を数値的に求める方法です：

\\[
\Phi_{\alpha\alpha'}(0\kappa, n'\kappa') \approx \frac{F_\alpha(0\kappa; u_{n'\kappa'\alpha'}) - F_\alpha(0\kappa; 0)}{u_{n'\kappa'\alpha'}}
\\]

ここで、\\(F_\alpha(0\kappa; u)\\) は変位 \\(u\\) があるときの原子 \\(0\kappa\\) に働く力の \\(\alpha\\) 成分です。

### 1.8.3 実用的なワークフロー

代表的な第一原理フォノン計算の流れ：

1. **構造最適化:** DFT計算で平衡格子構造を決定
2. **力定数計算:** DFPTまたは凍結フォノン法で \\(\Phi\\) を計算
3. **フォノン分散:** PHONOPYなどのツールで動的行列を構築し、高対称線に沿ってプロット
4. **状態密度:** ブリルアンゾーン全体で密な \\(\mathbf{q}\\) メッシュを使って積分
5. **熱力学量:** フォノン状態密度から比熱、自由エネルギーなどを計算

**推奨ソフトウェア**

- **PHONOPY:** 汎用的なフォノン解析（Python、様々なDFTコードと連携）
- **Quantum ESPRESSO (PHonon):** DFPTの標準的実装
- **VASP + Phonopy:** VASPで力を計算し、Phonopyで解析
- **ABINIT:** DFPTとフォノン計算の統合環境

## まとめ

本章では、格子力学の理論的基礎を以下の観点から学びました：

- **Born-von Kármán境界条件:** 無限結晶を有限系で扱うための周期境界条件と、許される波数の離散化
- **運動方程式:** 一般的な3次元多原子基底結晶の運動方程式と、平面波形式の解
- **動的行列:** フォノン問題を固有値問題 \\(D(\mathbf{q})\mathbf{e} = \omega^2\mathbf{e}\\) として定式化
- **力定数の性質:** 対称性、並進不変性、音響和則の数学的導出と物理的意味
- **対称性解析:** 点群と空間群がフォノンに与える影響、既約表現によるモード分類
- **実装:** Pythonによる動的行列の構築と対角化の実践
- **第一原理計算:** DFPTと凍結フォノン法による実材料への応用

この理論的枠組みは、次章以降で学ぶフォノン相互作用、熱伝導、電子-フォノン結合の基盤となります。また、第一原理計算ソフトウェアを使った実際の研究においても、結果を正しく解釈するために不可欠な知識です。

## 演習問題

### 問題1: 音響和則の検証（基礎）

1次元2原子鎖において、力定数が音響和則 \\(\sum_{n'\kappa'} \Phi(0\kappa, n'\kappa') = 0\\) を満たすことを、最近接相互作用のみの場合について確認せよ。

### 問題2: 動的行列の計算（中級）

単純立方格子（格子定数 \\(a\\)）で、最近接相互作用のみを考慮する。力定数を \\(C\\) として、\\(\mathbf{q} = (\pi/a, 0, 0)\\) （X点）での動的行列を構築し、3つの固有振動数を求めよ。対称性から縮退が予想されるか考察せよ。

### 問題3: 光学モードと音響モードの区別（中級）

NaCl構造（岩塩型）は、2原子基底（Na, Cl）を持つ面心立方格子です。\\(\mathbf{q} \to 0\\) の極限で、光学モード（TO, LO）と音響モードはどのように振る舞うか、原子変位の様子を図示して説明せよ。特にイオン結晶特有の長距離クーロン相互作用の影響を考察せよ。

### 問題4: 対称性と縮退（発展）

面心立方格子の \\(L\\) 点 \\(\mathbf{q} = (\pi/a)(1/2, 1/2, 1/2)\\) において、点群 \\(D_{3d}\\) の既約表現を調べ、単原子系のフォノンモードがどのように分類されるかを示せ。3つの音響モードの縮退構造を議論せよ。

### 問題5: Pythonプログラミング（実践）

1次元2原子鎖のコードを改良し、以下を追加実装せよ：

1. 偏極ベクトル \\(e_{\kappa}(\mathbf{q}, j)\\) のプロット（各ブランチについて、振幅と位相を図示）
2. 群速度 \\(v_g = \partial\omega/\partial q\\) の計算とプロット
3. フォノン状態密度 \\(g(\omega) = \sum_j \delta(\omega - \omega_j(q))\\) の数値計算（ヒストグラム法）

### 問題6: 第一原理計算との接続（発展）

PHONOPYのドキュメントを読み、以下を調べよ：

1. VASPで計算した力定数ファイル（FORCE_CONSTANTS）の形式
2. BORN ファイルの役割（イオン結晶の長距離相互作用補正）
3. `phonopy-load` コマンドを使ったフォノン分散の可視化方法

## 参考文献

1. M. Born and K. Huang, *Dynamical Theory of Crystal Lattices*, Oxford University Press (1954) — 格子力学の古典的名著
2. G. Leibfried and W. Ludwig, "Theory of Anharmonic Effects in Crystals", *Solid State Physics* **12**, 275 (1961)
3. S. Baroni, S. de Gironcoli, A. Dal Corso, and P. Giannozzi, "Phonons and related crystal properties from density-functional perturbation theory", *Rev. Mod. Phys.* **73**, 515 (2001) — DFPTの標準的レビュー
4. A. Togo and I. Tanaka, "First principles phonon calculations in materials science", *Scr. Mater.* **108**, 1 (2015) — PHONOPYの理論的背景
5. C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley (2005) — 入門的な説明（第4章、第5章）
6. N. W. Ashcroft and N. D. Mermin, *Solid State Physics*, Saunders (1976) — より詳細な理論的導出（第22章、第23章）

---

**ナビゲーション:**
- [← シリーズ目次](index.md)
- [第2章: フォノン相互作用 →](chapter-2.md)

---

## 免責事項

この教育コンテンツは、橋本研究室のナレッジベース用にAIの支援を受けて作成されました。正確性を期していますが、重要な情報については一次資料や教科書で確認することをお勧めします。

---

&copy; 2025 東北大学 橋本研究室. All rights reserved.
