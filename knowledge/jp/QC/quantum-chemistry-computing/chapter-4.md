---
title: "第4章: ハンズオン: H2 をゼロから"
chapter_title: "第4章: ハンズオン: H2 をゼロから"
subtitle: "ガウス型基底関数からVQEのエネルギーまで、すべての数値をコードが生み出す"
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/FkSyO-F2TPI"
    title="量子コンピュータによる量子化学計算 第4章: ハンズオン: H2 をゼロから"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

---

🌐 JP | [🇬🇧 EN](<../../../en/QC/quantum-chemistry-computing/chapter-4.html>) | Last sync: 2026-08-17

[量子コンピューティング道場](<../index.html>) > [量子コンピュータによる量子化学計算](<index.html>) > 第4章

これから出会うVQEのチュートリアルは、どれも分子ハミルトニアンがどこか別の場所からやってくるところから始まります。ライブラリの呼び出し、印刷されたパウリ係数の表、あるいはファイルです。それはアルゴリズムを教える方法としては妥当ですが、計算を理解する方法としては貧しいものです。なぜなら、興味深い工学は量子ビットよりも上流にあるからです。あの係数はどこから来るのでしょうか。量子的な部分が始まる前にどんな決定が下され、そのうちどれが、最適化器がどれほど良く働こうとも精度に上限を課すのでしょうか。

本章では、水素分子を積分の水準まで分解します。公表されたSTO-3Gの基底関数パラメータだけを出発点として、それ以外は何も使わずに、重なり積分、運動エネルギー積分、核引力積分、二電子積分をガウス関数について閉じた形で計算し、ハートリー・フォックの自己無撞着場ループを収束まで走らせ、分子軌道へ変換し、第二量子化されたハミルトニアンを組み立て、Jordan-Wigner変換で4量子ビットへ写像し、それを厳密に対角化し、最後にその厳密解に対してVQEを走らせます。道具は `numpy` と `math` だけです。化学パッケージも、SciPyも使いません。とくに `scipy.special.erf` は使いません。Boys関数に必要なものはPythonの `math.erf` だけで足りるからです。**ここに印字される数値はすべて実行結果から出てきます**。VQEが照合される参照エネルギーは、数節前で同じコードが生成したものであり、それこそがこの照合を飾りではなく意味のあるものにしています。

## 4.1 計画

5つのステージがあり、それぞれが次のステージの消費する成果物を生み出します。

| ステージ | 入力 | 出力 |
|---|---|---|
| 1. 基底関数系と積分 | STO-3Gパラメータ、核配置 | \\(S\\), \\(T\\), \\(V\\), \\((\mu\nu\|\lambda\sigma)\\), \\(E_{\text{nuc}}\\) |
| 2. ハートリー・フォックSCF | それらの積分 | MO係数 \\(C\\)、RHFエネルギー |
| 3. MO変換 | \\(C\\) とAO積分 | \\(h_{pq}\\), \\((pq\|rs)\\) |
| 4. Jordan-Wigner変換とFCI | MO積分 | \\(16 \times 16\\) 行列、FCIエネルギー |
| 5. VQE | その行列 | 変分エネルギー（ステージ4と照合） |

配置は全体を通じて固定します。2個の陽子を \\(z\\) 軸に沿って **\\(R = 1.4\\) ボーア** 離して置きます。これは1つの選択であり、以降のすべての数値が再現可能であるようにここで明示しておきます。平衡結合長に近い値ですが、そこから導かれたものではなく、ポテンシャルエネルギー曲線の極小について何かを主張するものでもありません。

## 4.2 ステージ1: 基底関数系と積分

### 📚 STO-3Gのパラメータは結果ではなく定義である

**基底関数系** とは、分子軌道を展開するための固定された関数のリストです。STO-3Gはその中で最小の実用的なもので、各原子軌道は3個のガウス型プリミティブの **縮約** であり、Slater型軌道を近似するようにフィットされて一度限り選ばれ、公表されています。水素の場合、1原子あたりちょうど1個の基底関数があり、

\\[ \phi_\mu(\mathbf{r}) = \sum_{k=1}^{3} d_k \left( \frac{2\alpha_k}{\pi} \right)^{3/4} e^{-\alpha_k |\mathbf{r} - \mathbf{R}_\mu|^2} \\]

そのパラメータは次のように公表されています。

| \\(k\\) | 指数 \\(\alpha_k\\) | 係数 \\(d_k\\) |
|---|---|---|
| 1 | 3.42525091 | 0.15432897 |
| 2 | 0.62391373 | 0.53532814 |
| 3 | 0.16885540 | 0.44463454 |

この6個の数値が、本章においてコードの外から取り込まれる唯一の量です。「水素に対するSTO-3G」が何を意味するかの **定義** であって、結果ではありません。ここから2つの帰結が導かれ、そのどちらも見た目以上に重要です。1原子あたり1個の関数であれば、分子は **2個の空間軌道** を持ち、したがって **4個のスピン軌道** を持ちます。4量子ビットで足りるのはこのためです。そして最小基底は厳しい近似です。精度の天井は、どの量子アルゴリズムを選ぶよりも前に、ここで決まってしまいます。

### 📚 4種類の積分を閉じた形で

物理的にはより優れたSlater関数ではなくガウス関数が使われる理由は1つです。必要な積分がすべて閉じた形を持つからです。その原動力は **ガウス積の定理** です。\\(\mathbf{A}\\) と \\(\mathbf{B}\\) を中心とする2つのs型ガウス関数の積は、その中間を中心とする単一のガウス関数になります。

\\[ p = \alpha + \beta, \qquad \mathbf{P} = \frac{\alpha \mathbf{A} + \beta \mathbf{B}}{p}, \qquad K_{AB} = \exp\left( -\frac{\alpha\beta}{p} |\mathbf{A} - \mathbf{B}|^2 \right) \\]

プリミティブの規格化定数を \\(N_a = (2\alpha/\pi)^{3/4}\\) とすると、規格化されたs型プリミティブについての4種類の積分は次のようになります。

\\[ S_{ab} = N_a N_b \, K_{AB} \left( \frac{\pi}{p} \right)^{3/2} \\]

\\[ T_{ab} = N_a N_b \, \frac{\alpha\beta}{p} \left[ 3 - \frac{2\alpha\beta}{p} |\mathbf{A} - \mathbf{B}|^2 \right] K_{AB} \left( \frac{\pi}{p} \right)^{3/2} \\]

\\[ V_{ab}^{(C)} = -N_a N_b \, Z_C \, \frac{2\pi}{p} \, K_{AB} \, F_0\!\left( p \, |\mathbf{P} - \mathbf{C}|^2 \right) \\]

\\[ (ab|cd) = N_a N_b N_c N_d \, \frac{2\pi^{5/2}}{p\,q\sqrt{p+q}} \, K_{AB} K_{CD} \, F_0\!\left( \frac{pq}{p+q} |\mathbf{P} - \mathbf{Q}|^2 \right) \\]

ここで \\(q\\) と \\(\mathbf{Q}\\) は2番目の対についての積のパラメータであり、二電子積分は **化学者の記法** \\((ab|cd) = \iint a(1) b(1) \frac{1}{r_{12}} c(2) d(2)\\) で書かれています。

特殊関数はただ1つしか現れません。0次の **Boys関数**

\\[ F_0(x) = \int_0^1 e^{-x t^2} \, dt = \frac{1}{2}\sqrt{\frac{\pi}{x}}\,\mathrm{erf}\!\left(\sqrt{x}\right), \qquad F_0(0) = 1 \\]

は、ガウス積分を済ませたあとにクーロン特異性 \\(1/r\\) が変じたものです。\\(x = 0\\) における除去可能な特異性に注意してください。閉じた形は \\(\sqrt{x}\\) で割るため、引数が小さいときにはコードが極限展開 \\(F_0(x) \approx 1 - x/3\\) へ切り替えなければなりません。

各 **縮約** 積分は、対応するプリミティブ積分をプリミティブのすべての組み合わせについて縮約係数で重み付けして足し上げたものです。1関数あたり3個のプリミティブということは、重なり積分の1要素あたり \\(3^2 = 9\\) 個のプリミティブ項、二電子積分1個あたり \\(3^4 = 81\\) 個のプリミティブ項を意味します。

```python
import math
import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

# 水素に対するSTO-3Gパラメータ: 公表された基底関数系の「定義」。
EXP_H = np.array([3.42525091, 0.62391373, 0.16885540])
COEF_H = np.array([0.15432897, 0.53532814, 0.44463454])

R_BOHR = 1.4                                    # 選んだH-H間距離（ボーア単位）
NUC = [(1.0, np.array([0.0, 0.0, 0.0])), (1.0, np.array([0.0, 0.0, R_BOHR]))]
BASIS = [(EXP_H, COEF_H, NUC[0][1]), (EXP_H, COEF_H, NUC[1][1])]
NBF = len(BASIS)

def F0(x):                                      # Boys関数、math.erfのみを使用
    return 1.0 - x / 3.0 if x < 1e-12 else 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))

def Ns(a):                                      # s型プリミティブの規格化定数
    return (2.0 * a / math.pi) ** 0.75

def gp(a, A, b, B):                             # ガウス積の定理
    p = a + b
    return p, (a * A + b * B) / p, math.exp(-a * b / p * float(np.dot(A - B, A - B)))

def s_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    return Ns(a) * Ns(b) * K * (math.pi / p) ** 1.5

def t_prim(a, A, b, B):
    p, _, K = gp(a, A, b, B)
    mu, AB2 = a * b / p, float(np.dot(A - B, A - B))
    return Ns(a) * Ns(b) * mu * (3.0 - 2.0 * mu * AB2) * K * (math.pi / p) ** 1.5

def v_prim(a, A, b, B, Zc, C):
    p, P, K = gp(a, A, b, B)
    return -Ns(a) * Ns(b) * Zc * (2.0 * math.pi / p) * K * F0(p * float(np.dot(P - C, P - C)))

def eri_prim(a, A, b, B, c, C, d, D):
    p, P, Kab = gp(a, A, b, B)
    q, Q, Kcd = gp(c, C, d, D)
    pref = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
    return (Ns(a) * Ns(b) * Ns(c) * Ns(d) * pref * Kab * Kcd
            * F0(p * q / (p + q) * float(np.dot(P - Q, P - Q))))

def contract2(fn):                              # 2添字積分を縮約する
    M = np.zeros((NBF, NBF))
    for i, (ea, ca, A) in enumerate(BASIS):
        for j, (eb, cb, B) in enumerate(BASIS):
            M[i, j] = sum(wa * wb * fn(a, A, b, B)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb))
    return M

S, T = contract2(s_prim), contract2(t_prim)
V = sum(contract2(lambda a, A, b, B, Z=Z, C=C: v_prim(a, A, b, B, Z, C)) for Z, C in NUC)
H_core = T + V

ERI = np.zeros((NBF,) * 4)
for i, j, k, l in product(range(NBF), repeat=4):
    (ea, ca, A), (eb, cb, B) = BASIS[i], BASIS[j]
    (ec, cc, C), (ed, cd, D) = BASIS[k], BASIS[l]
    ERI[i, j, k, l] = sum(wa * wb * wc * wd * eri_prim(a, A, b, B, c, C, d, D)
                          for a, wa in zip(ea, ca) for b, wb in zip(eb, cb)
                          for c, wc in zip(ec, cc) for d, wd in zip(ed, cd))

E_nuc = NUC[0][0] * NUC[1][0] / float(np.linalg.norm(NUC[0][1] - NUC[1][1]))

for name, M in [("S", S), ("T", T), ("V", V), ("H_core = T + V", H_core)]:
    print(f"{name:16s} = [{M[0,0]:+.6f} {M[0,1]:+.6f} ; {M[1,0]:+.6f} {M[1,1]:+.6f}]")
print(f"\n(00|00) = {ERI[0,0,0,0]:.9f}   (00|11) = {ERI[0,0,1,1]:.9f}")
print(f"(01|01) = {ERI[0,1,0,1]:.9f}   (00|01) = {ERI[0,0,0,1]:.9f}")
print(f"\nS symmetric: {np.allclose(S, S.T)}   "
      f"max |S_ii - 1| = {np.max(np.abs(np.diag(S) - 1.0)):.2e}   "
      f"(ij|kl) = (kl|ij): {np.allclose(ERI, ERI.transpose(2, 3, 0, 1))}")
print(f"nuclear repulsion Z_A Z_B / R = {E_nuc:.9f} hartree")
```

**出力:**

```
S                = [+1.000000 +0.659318 ; +0.659318 +1.000000]
T                = [+0.760032 +0.236455 ; +0.236455 +0.760032]
V                = [-1.880441 -1.194835 ; -1.194835 -1.880441]
H_core = T + V   = [-1.120409 -0.958380 ; -0.958380 -1.120409]

(00|00) = 0.774605930   (00|11) = 0.569675915
(01|01) = 0.297028535   (00|01) = 0.444107650

S symmetric: True   max |S_ii - 1| = 9.11e-09   (ij|kl) = (kl|ij): True
nuclear repulsion Z_A Z_B / R = 0.714285714 hartree
```

**結果を読む。** 2つの水素1s関数の重なりは \\(0.659318\\) です。大きな値ですが、これは \\(1.4\\) ボーアでは原子どうしが結合できるほど近いからです。\\(S\\) の対角成分は \\(9 \times 10^{-9}\\) の精度で1に等しく、これは入力ではなく検算です。公表された縮約係数は規格化された関数を与えるはずであり、実際に、それらが表として与えられた精度の範囲で与えています。核引力はいたるところで負であり、正の運動エネルギーを上回るため、\\(H^{\text{core}}\\) は負になります。二電子積分の8重の置換対称性も厳密に成り立っています。

## 4.3 ステージ2: 制限ハートリー・フォック法

ハートリー・フォック法は、電子間反発を平均場で置き換えます。各電子は他の電子がつくる平均場の中を動きます。\\(2n\\) 個の電子が \\(n\\) 個の二重占有された空間軌道に入っている閉殻分子の場合、これが **制限** ハートリー・フォック法（RHF）です。H₂ では \\(n = 1\\) です。

軌道は基底で \\(\psi_i = \sum_\mu C_{\mu i}\phi_\mu\\) と展開され、変分条件は **Roothaan方程式** \\(\mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}\\) になります。これは *一般化* 固有値問題です。基底が直交していないからです。これは対称直交化行列 \\(\mathbf{X} = \mathbf{S}^{-1/2}\\)（\\(\mathbf{S}\\) の固有分解から計算します）によって通常の固有値問題に還元され、その後 \\(\mathbf{F}' = \mathbf{X}^{T}\mathbf{F}\mathbf{X}\\) を対角化して \\(\mathbf{C} = \mathbf{X}\mathbf{C}'\\) とします。この方程式は **非線形** です。Fock行列が、それ自身の定める軌道に依存するからです。したがって密度行列 \\(P_{\mu\nu} = 2\sum_{i}^{\text{occ}} C_{\mu i} C_{\nu i}\\) についての自己無撞着場ループを回すことになり、

\\[ F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma} \left[ (\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma) \right], \qquad E_{\text{elec}} = \frac{1}{2}\sum_{\mu\nu} P_{\mu\nu}\left( H^{\text{core}}_{\mu\nu} + F_{\mu\nu} \right) \\]

そして \\(E_{\text{RHF}} = E_{\text{elec}} + E_{\text{nuc}}\\) となります。Fock行列の2つの項は **クーロン** 反発と **交換** 相互作用であり、後者は反対称性の帰結であって古典的な対応物を持たず、係数 \\(\frac{1}{2}\\) を伴います。交換が同じスピンをもつ電子どうしの間にしか働かないからです。

```python
sv, sc = np.linalg.eigh(S)
X = sc @ np.diag(1.0 / np.sqrt(sv)) @ sc.T      # 対称直交化行列 S^(-1/2)
N_OCC = 1                                       # 電子2個、空間軌道1つあたり2個

def fock(P):
    F = H_core.copy()
    for m, n in product(range(NBF), repeat=2):
        F[m, n] += float(np.sum(P * (ERI[m, n] - 0.5 * ERI[m, :, n, :])))
    return F

# 意図的に偏らせた初期推定: 両方の電子を原子Aに置く。
P, E_elec = np.array([[2.0, 0.0], [0.0, 0.0]]), 0.0
print(f"X^T S X = I: {np.allclose(X.T @ S @ X, np.eye(NBF))}   "
      f"overlap eigenvalues = {sv}\n")
print("  iter          E_elec (hartree)        change     max |dP|")
for it in range(1, 51):
    eps, Co = np.linalg.eigh(X.T @ fock(P) @ X)
    C = X @ Co
    P_new = 2.0 * (C[:, :N_OCC] @ C[:, :N_OCC].T)
    E_new = 0.5 * float(np.sum(P_new * (H_core + fock(P_new))))
    dE, dP = E_new - E_elec, float(np.max(np.abs(P_new - P)))
    print(f"  {it:4d}   {E_new:+22.12f}   {dE:+11.3e}   {dP:.3e}")
    P, E_elec = P_new, E_new
    if abs(dE) < 1e-12 and dP < 1e-12:
        break

E_rhf = E_elec + E_nuc
print(f"\nFock hermitian: {np.allclose(fock(P), fock(P).T)}   "
      f"C^T S C = I: {np.allclose(C.T @ S @ C, np.eye(NBF))}   "
      f"orbital energies = {eps}")
print(f"MO coefficients C (columns are the molecular orbitals) =")
print(C)
print(f"\nelectronic energy = {E_elec:+.9f}   nuclear repulsion = {E_nuc:+.9f}")
print(f"TOTAL RHF ENERGY  = {E_rhf:+.9f} hartree")
```

**出力:**

```
X^T S X = I: True   overlap eigenvalues = [0.340682 1.659318]

  iter          E_elec (hartree)        change     max |dP|
     1          -1.827181194524    -1.827e+00   1.284e+00
     2          -1.830964837516    -3.784e-03   1.028e-01
     3          -1.830999715265    -3.488e-05   9.533e-03
     4          -1.831000036365    -3.211e-07   9.115e-04
     5          -1.831000039321    -2.956e-09   8.743e-05
     6          -1.831000039348    -2.722e-11   8.389e-06
     7          -1.831000039348    -2.507e-13   8.049e-07
     8          -1.831000039348    -1.998e-15   7.723e-08
     9          -1.831000039348    -2.220e-16   7.410e-09
    10          -1.831000039348    +0.000e+00   7.110e-10
    11          -1.831000039348    -2.220e-16   6.822e-11
    12          -1.831000039348    +2.220e-16   6.546e-12
    13          -1.831000039348    +0.000e+00   6.279e-13

Fock hermitian: True   C^T S C = I: True   orbital energies = [-0.578203  0.670268]
MO coefficients C (columns are the molecular orbitals) =
[[-0.548934 -1.211464]
 [-0.548934  1.211464]]

electronic energy = -1.831000039   nuclear repulsion = +0.714285714
TOTAL RHF ENERGY  = -1.116714325 hartree
```

**結果を読む。** 3点あります。

  * **収束は単調で、おおむね線形です。** 意図的に偏らせた初期推定——両方の電子を片方の原子に置いたもの——から出発して、エネルギー変化は1反復あたりおよそ2桁ずつ小さくなり、8回目の反復で計算機の精度に達します。密度はエネルギーより遅れて落ち着きますが、これはいつものパターンです。解においてエネルギーは停留しているので、密度の1次の誤差はエネルギーには2次でしか効かないのです。これは第3章の変分原理が、純粋に古典的な計算の中に現れたものです。
  * **対称性は課されるのではなく回復されます。** 占有軌道は両方の原子上で \\(\pm 0.548934\\) に収束します。等しく、対称で、結合性の組み合わせです。一方で空の軌道は2つの中心上で符号が逆になり、軌道エネルギーはそれぞれ \\(-0.578203\\) と \\(+0.670268\\) ハートリーです。コードの中でこれを強制したものは何もありません。偏った初期推定は単に誤っており、反復がそれを正したのです。
  * **RHFの全エネルギーは \\(-1.116714325\\) ハートリー** であり、その内訳は電子エネルギー \\(-1.831000039\\) と核間反発 \\(+0.714285714\\) です。量子計算が改善しなければならないのは、この数値です。

## 4.4 ステージ3: 分子軌道基底へ

量子ビットに載せたいハミルトニアンは第二量子化で書かれており、基底関数ではなく軌道を用いて表されます。そのためには、いま得られた係数で積分を変換する必要があります。

\\[ h_{pq} = \sum_{\mu\nu} C_{\mu p} H^{\text{core}}_{\mu\nu} C_{\nu q}, \qquad (pq|rs) = \sum_{\mu\nu\lambda\sigma} C_{\mu p} C_{\nu q} C_{\lambda r} C_{\sigma s} \, (\mu\nu|\lambda\sigma) \\]

2番目のものが悪名高い **4添字変換** です。素朴に行えば \\(O(N^8)\\) のコストがかかり、1添字ずつの4回の連続した縮約として行えば \\(O(N^5)\\) で済みます。ここでは `np.einsum` に `optimize=True` を渡すことでそれを実現しています。\\(N = 2\\) ではこの区別は形式的なものですが、「小さな」分子が小さくない理由の1つがこれです。

そして電子ハミルトニアンは次のようになります。

\\[ \hat{H} = \sum_{pq} \sum_{\sigma} h_{pq}\, a^{\dagger}_{p\sigma} a_{q\sigma} \;+\; \frac{1}{2}\sum_{pqrs} \sum_{\sigma\tau} (pq|rs)\, a^{\dagger}_{p\sigma} a^{\dagger}_{r\tau} a_{s\tau} a_{q\sigma} \;+\; E_{\text{nuc}} \\]

ここで \\(p, q, r, s\\) は2個の **空間** 軌道を走り、\\(\sigma, \tau\\) はスピンを走ります。第2章では同じ演算子を物理学者の記法の積分 \\(h_{pqrs}\\) で書きました。ここで用いる化学者の記法は、同じ対象の4つの添字を別の組み方で対にしたものであり、生成演算子と消滅演算子がラベルと同じ順序で現れないのはそのためです。

```python
h_mo = C.T @ H_core @ C
eri_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, ERI, optimize=True)

print("h_pq =")
print(h_mo)
print(f"(00|00) = {eri_mo[0,0,0,0]:.9f}   (00|11) = {eri_mo[0,0,1,1]:.9f}")
print(f"(01|01) = {eri_mo[0,1,0,1]:.9f}   (11|11) = {eri_mo[1,1,1,1]:.9f}")
print(f"(00|01) = {eri_mo[0,0,0,1]:.9f}   (vanishes by symmetry)")

E_check = 2.0 * h_mo[0, 0] + eri_mo[0, 0, 0, 0] + E_nuc
print(f"\nRHF energy rebuilt from the MO integrals = {E_check:+.9f} hartree")
print(f"agrees with the SCF total: {abs(E_check - E_rhf) < 1e-10}   "
      f"(difference {abs(E_check - E_rhf):.2e})")
```

**出力:**

```
h_pq =
[[-1.252797  0.      ]
 [ 0.       -0.475602]]
(00|00) = 0.674594084   (00|11) = 0.663563991
(01|01) = 0.181257915   (11|11) = 0.697495347
(00|01) = 0.000000000   (vanishes by symmetry)

RHF energy rebuilt from the MO integrals = -1.116714325 hartree
agrees with the SCF total: True   (difference 4.44e-16)
```

**結果を読む。** 一電子行列は対角的で、\\(h_{00} = -1.252797\\)、\\(h_{11} = -0.475602\\) ハートリーです。分子軌道がFock行列を対角化し、しかもH₂ では、それらを結合性・反結合性にしている対称性が同時にそれらをコアハミルトニアンの固有関数にもしているからです。\\((00|01)\\) が消えるのも同じ理由で、これはgerade軌道とungerade軌道を結び付ける量だからです。最後の2行は **コストに見合う相互検算** です。変換された積分だけからRHFエネルギーを再構成すると、\\(E_{\text{RHF}} = 2h_{00} + (00|00) + E_{\text{nuc}}\\) はSCFの合計を \\(4 \times 10^{-16}\\) ハートリーの精度で再現します。4添字変換の誤りは犯しやすく、しかも黙って通り過ぎるものです。この検算がそれを捕まえます。

## 4.5 ステージ4: Jordan-Wigner変換と厳密対角化

4個のスピン軌道が4個の量子ビットになります。それらを量子ビット \\(0,1,2,3\\) 上で \\(0\alpha, 0\beta, 1\alpha, 1\beta\\) の順に並べると、**Jordan-Wigner変換** は各フェルミオン消滅演算子を次のように表します。

\\[ a_p = \left( \bigotimes_{q < p} Z_q \right) \sigma^-_p, \qquad \sigma^- = |0\rangle\langle 1| \\]

物理的な仕事をするのは下降演算子で、占有されたスピン軌道を空にします。そして、それより小さい添字のすべての量子ビットに作用する \\(Z\\) 演算子の列が、フェルミオン的な負符号を供給します。これがなければ演算子はボソンのように交換してしまいますが、これがあることで反交換し、コードは何かを組み立てる前に \\(\{a_p, a_q^\dagger\} = \delta_{pq}\\) を数値的に検証します。

これらの行列を第二量子化されたハミルトニアンに代入すると、明示的な \\(16 \times 16\\) 行列が得られます。そこで、現実の計算にはできないことができます。対角化です。中性一重項セクターにおけるその最低固有値が **完全配置間相互作用（FCI）** エネルギーであり、この基底関数系の内側では厳密であって、VQEが照合される目標となります。

```python
N_SO, DIM = 4, 16
Zp = np.array([[1.0, 0.0], [0.0, -1.0]])
LOW = np.array([[0.0, 1.0], [0.0, 0.0]])            # sigma^- = |0><1|

def jw(p):                                          # a_p = Z_0 ... Z_{p-1} sigma^-_p
    op = np.array([[1.0]])
    for q in range(N_SO):
        op = np.kron(op, Zp if q < p else (LOW if q == p else np.eye(2)))
    return op

def so(p, sigma):                                   # 0a, 0b, 1a, 1b -> 量子ビット 0..3
    return 2 * p + sigma

a = [jw(p) for p in range(N_SO)]
ad = [op.T for op in a]
err = max(float(np.max(np.abs(a[p] @ ad[q] + ad[q] @ a[p]
                              - (np.eye(DIM) if p == q else 0.0))))
          for p, q in product(range(N_SO), repeat=2))
print(f"max deviation in the anticommutator (a_p, a_q^dag) = delta_pq : {err:.2e}")

H_q = E_nuc * np.eye(DIM)
for (p, q), sg in product(product(range(2), repeat=2), (0, 1)):
    H_q += h_mo[p, q] * (ad[so(p, sg)] @ a[so(q, sg)])
for (p, q, r, s), (sg, tau) in product(product(range(2), repeat=4),
                                       product((0, 1), repeat=2)):
    H_q += 0.5 * eri_mo[p, q, r, s] * (ad[so(p, sg)] @ ad[so(r, tau)]
                                       @ a[so(s, tau)] @ a[so(q, sg)])

HF, DOUBLE = 0b1100, 0b0011
print(f"H is {H_q.shape[0]}x{H_q.shape[1]}, hermitian: {np.allclose(H_q, H_q.T)}")
print(f"<1100|H|1100> = {H_q[HF, HF]:+.9f}   equals the SCF total: "
      f"{abs(H_q[HF, HF] - E_rhf) < 1e-10}")

evals = np.linalg.eigvalsh(H_q)
print("\nfull 16x16 spectrum (hartree)")
for k in range(0, DIM, 4):
    print("  " + "  ".join(f"{v:+14.9f}" for v in evals[k:k + 4]))

sectors = {}
for i in range(DIM):
    o = [(i >> (N_SO - 1 - q)) & 1 for q in range(N_SO)]
    sectors.setdefault((sum(o), o[0] + o[2] - o[1] - o[3]), []).append(i)

print("\n   N   2*Sz   dim      lowest energy (hartree)")
low, allv = {}, []
for key in sorted(sectors):
    vals = np.linalg.eigvalsh(H_q[np.ix_(sectors[key], sectors[key])])
    low[key] = float(vals[0])
    allv.extend(vals.tolist())
    print(f"  {key[0]:2d}   {key[1]:+4d}   {len(sectors[key]):3d}   {vals[0]:+22.9f}")
print(f"\nsector eigenvalues reproduce the full spectrum: "
      f"{np.allclose(np.sort(np.array(allv)), evals)}")

E_fci = low[(2, 0)]
print(f"\nTOTAL RHF ENERGY   = {E_rhf:+.9f} hartree")
print(f"FCI GROUND ENERGY  = {E_fci:+.9f} hartree")
print(f"CORRELATION ENERGY = {E_fci - E_rhf:+.9f} hartree")
print(f"FCI <= RHF: {E_fci <= E_rhf}   "
      f"FCI is the global minimum of the 16x16: {abs(E_fci - evals[0]) < 1e-12}")

dets = [0b1100, 0b1001, 0b0110, 0b0011]
print("\nthe N=2, Sz=0 block in the basis |1100>, |1001>, |0110>, |0011>")
print(H_q[np.ix_(dets, dets)])
print(f"<1100|H|1001> = {H_q[HF, 0b1001]:+.2e}      "
      f"<1100|H|0011> = {H_q[HF, DOUBLE]:+.9f}")

# このハミルトニアンは実際にいくつのパウリ文字列を含むのか。
PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
n_terms = 0
for st in product('IXYZ', repeat=N_SO):
    M = np.array([[1.0]], dtype=complex)
    for ch in st:
        M = np.kron(M, PAULI[ch])
    if abs(complex(np.trace(M.conj().T @ H_q)) / DIM) > 1e-12:
        n_terms += 1
print(f"\nnonzero Pauli strings out of the 256 four-qubit strings: {n_terms}")
```

**出力:**

```
max deviation in the anticommutator (a_p, a_q^dag) = delta_pq : 0.00e+00
H is 16x16, hermitian: True
<1100|H|1100> = -1.116714325   equals the SCF total: True

full 16x16 spectrum (hartree)
    -1.137275944    -0.538511348    -0.538511348    -0.531807570
    -0.531807570    -0.531807570    -0.446446557    -0.446446557
    -0.169291741    +0.238683415    +0.238683415    +0.353649468
    +0.353649468    +0.481138081    +0.714285714    +0.921316558

   N   2*Sz   dim      lowest energy (hartree)
   0     +0     1             +0.714285714
   1     -1     2             -0.538511348
   1     +1     2             -0.538511348
   2     -2     1             -0.531807570
   2     +0     4             -1.137275944
   2     +2     1             -0.531807570
   3     -1     2             -0.446446557
   3     +1     2             -0.446446557
   4     +0     1             +0.921316558

sector eigenvalues reproduce the full spectrum: True

TOTAL RHF ENERGY   = -1.116714325 hartree
FCI GROUND ENERGY  = -1.137275944 hartree
CORRELATION ENERGY = -0.020561619 hartree
FCI <= RHF: True   FCI is the global minimum of the 16x16: True

the N=2, Sz=0 block in the basis |1100>, |1001>, |0110>, |0011>
[[-1.116714  0.       -0.        0.181258]
 [ 0.       -0.35055  -0.181258  0.      ]
 [-0.       -0.181258 -0.35055  -0.      ]
 [ 0.181258  0.       -0.        0.460576]]
<1100|H|1001> = +2.80e-14      <1100|H|0011> = +0.181257915

nonzero Pauli strings out of the 256 four-qubit strings: 15
```

**結果を読む。** 5点あり、覚えておくべきは2点目です。

  * **ハートリー・フォック行列式は対角上にあります。** \\(\langle 1100|\hat{H}|1100\rangle = -1.116714325\\) ハートリーで、ステージ2のSCFの合計と一致します。2つの独立した経路——密度行列についてのSCFループと、量子ビット演算子の行列要素——が同じ数値を与えており、これは積分からJordan-Wigner変換までの連鎖全体を検証しています。ハミルトニアンはまた **粒子数とスピンについてブロック対角** です。各セクターの固有値を別々に計算すると、それらは厳密に全スペクトルへと組み上がります。これは \\(\hat{H}\\) が粒子数演算子および \\(S_z\\) 演算子と交換することの帰結であり、ステージ5が利用する構造的な事実でもあります。
  * **変分原理が要求するとおり、FCIはRHFの下にあります。** \\(E_{\text{FCI}} = -1.137275944\\) に対して \\(E_{\text{RHF}} = -1.116714325\\) ハートリーですから、このコードが計算した **相関エネルギー** は \\(-0.020561619\\) ハートリーです。それがここで得られる賞金のすべてであり、平均場描像が取りこぼすものです。それが何であるかは正確に述べてください。*STO-3G基底の内側での* 厳密解とハートリー・フォックの差です。水素分子の相関エネルギーではありません。基底が最小だからであり、この基底の内側で走らせるどんなアルゴリズムも、基底が表現できないものを回復することはできないからです。
  * **中性一重項は \\(16 \times 16\\) 行列の大域的最小でもあります** が、そのことは事前に何ら保証されていたわけではありません。イオン化したセクターはより高い位置にあります。電子を1個取り去るコストは、核間反発が減ることによる得より大きいのです。
  * **ハミルトニアンは15個のパウリ文字列を含みます。** \\(16 \times 16\\) 行列を256要素の4量子ビットパウリ基底で展開すると、係数が0でない文字列はちょうど15個残り、恒等演算子もそこに含まれます。第5章はこの個数を測定問題の出発点として取り上げ、それが水素固有の積分ではなく対称性から従うことを論じます。
  * **意味を持つ非対角要素は1つだけです。** 4行列式からなる \\(N=2, S_z=0\\) ブロックにおいて、ハートリー・フォック行列式は二重励起の \\(|0011\rangle\\) と強さ \\(+0.181257915\\) で結び付き、2つの一重励起行列式とは \\(10^{-14}\\) 程度の行列要素で結び付きます。数値的には0です。これはBrillouinの定理と軌道の対称性が、主張としてではなく印字された数値として現れたものです。

## 4.6 ステージ5: 簡約された問題に対するVQE

最後の観察は贈り物です。基底状態は一重励起行列式上に重みを持たないので、問題全体は \\(|1100\rangle\\) と \\(|0011\rangle\\) が張る **2次元** の部分空間に収まり、パラメータは1個で足ります。

\\[ |\psi(\theta)\rangle = \cos\frac{\theta}{2}\,|1100\rangle + \sin\frac{\theta}{2}\,|0011\rangle \\]

\\(\theta = 0\\) においてこれはちょうどハートリー・フォック行列式であり、これがVQEの標準的な出発点です。エネルギーが次の閉じた形を持つため、

\\[ E(\theta) = \frac{H_{00} + H_{11}}{2} + \frac{H_{00} - H_{11}}{2}\cos\theta + H_{01}\sin\theta \\]

第3章のパラメータシフト則は、このアンザッツに対して単に良い推定量であるだけでなく **厳密** です。

このコードは状態ベクトルを直接操作します。ハードウェア上では、同じ状態は量子ビット0と1に \\(X\\) ゲートを掛けてハートリー・フォック行列式をつくり、続いて \\(R_y(\theta)\\) を1個だけ含むもつれブロックによって振幅をコヒーレントに \\(|0011\rangle\\) へ移すことで準備されます。二重励起回転、あるいはGivens回転と呼ばれるものです。\\(\theta\\) がちょうど1個のそうした回転を通じて入るので、パラメータシフト則が適用できます。これは第3章の意味での対称性を保つアンザッツです。\\(N=2, S_z=0\\) セクターの外に出ることができないので、最適化器が別の化学種へさまよい出ることはありません。

```python
def ansatz(t):                                  # cos(t/2)|1100> + sin(t/2)|0011>
    psi = np.zeros(DIM)
    psi[HF], psi[DOUBLE] = math.cos(t / 2.0), math.sin(t / 2.0)
    return psi

def energy(t):
    psi = ansatz(t)
    return float(psi @ H_q @ psi)

def grad(t):                                    # パラメータシフト則、ここでは厳密
    return 0.5 * (energy(t + math.pi / 2) - energy(t - math.pi / 2))

print("   theta      E(theta) (hartree)")
for t in np.linspace(-1.0, 0.5, 7):
    print(f"  {t:+.4f}   {energy(t):+18.9f}")

theta = 0.0                                     # ハートリー・フォック行列式から開始
print("\n step        theta        E(theta) (hartree)     E - E_FCI     dE/dtheta")
for step in range(13):
    if step % 2 == 0:
        print(f"  {step:3d}   {theta:+11.8f}   {energy(theta):+18.9f}   "
              f"{energy(theta) - E_fci:+11.3e}   {grad(theta):+11.3e}")
    theta = theta - grad(theta)

E_vqe, psi = energy(theta), ansatz(theta)
print(f"\noptimal theta = {theta:+.9f} rad")
print(f"VQE ENERGY    = {E_vqe:+.9f} hartree      FCI ENERGY = {E_fci:+.9f} hartree")
print(f"|VQE - FCI|   = {abs(E_vqe - E_fci):.3e} hartree   "
      f"VQE >= FCI: {E_vqe - E_fci > -1e-12}")
print(f"amplitudes: |1100> = {psi[HF]:+.9f}   |0011> = {psi[DOUBLE]:+.9f}   "
      f"HF weight = {psi[HF] ** 2:.9f}")
print(f"\nRHF {E_rhf:+.9f}    FCI {E_fci:+.9f}    VQE {E_vqe:+.9f}    "
      f"correlation {E_fci - E_rhf:+.9f}")
```

**出力:**

```
   theta      E(theta) (hartree)
  -1.0000         -0.906699132
  -0.7500         -1.028664408
  -0.5000         -1.107070050
  -0.2500         -1.137041175
  +0.0000         -1.116714325
  +0.2500         -1.047353324
  +0.5000         -0.933270703

 step        theta        E(theta) (hartree)     E - E_FCI     dE/dtheta
    0   +0.00000000         -1.116714325    +2.056e-02    +1.813e-01
    2   -0.21737965         -1.137246494    +2.945e-05    +6.904e-03
    4   -0.22560061         -1.137275905    +3.903e-08    +2.513e-04
    6   -0.22589989         -1.137275944    +5.172e-11    +9.149e-06
    8   -0.22591078         -1.137275944    +6.883e-14    +3.330e-07
   10   -0.22591118         -1.137275944    +2.220e-16    +1.212e-08
   12   -0.22591119         -1.137275944    +0.000e+00    +4.413e-10

optimal theta = -0.225911191 rad
VQE ENERGY    = -1.137275944 hartree      FCI ENERGY = -1.137275944 hartree
|VQE - FCI|   = 2.220e-16 hartree   VQE >= FCI: True
amplitudes: |1100> = +0.993627297   |0011> = -0.112715549   HF weight = 0.987295205

RHF -1.116714325    FCI -1.137275944    VQE -1.137275944    correlation -0.020561619
```

**結果を読む。** 4点あります。

  * **スキャンは、ハートリー・フォック状態が最小ではないことを示します。** \\(E(0) = -1.116714325\\) ハートリーは構成上RHFエネルギーそのものであり、曲線は \\(\theta\\) が負の側でその下へ落ち込みます。相関エネルギーは、その落ち込みの深さとして目に見えるものになります。
  * **VQEは計算機精度でFCIエネルギーに到達します。** 両者とも \\(-1.137275944\\) ハートリーで、差は \\(2 \times 10^{-16}\\)、浮動小数点の丸め誤差です。勾配降下は倍精度を使い切るのにおよそ10ステップを要します。変分的な検算 \\(E_{\text{VQE}} \geq E_{\text{FCI}}\\) は成り立っており、アンザッツが全空間の部分集合である以上、そうならなければなりません。
  * **相関を取り込んだ状態も、なお98.7%はハートリー・フォックです。** 最適点は \\(\theta = -0.225911\\) ラジアンにあり、振幅は \\(|1100\rangle\\) 上で \\(+0.993627\\)、\\(|0011\rangle\\) 上で \\(-0.112716\\) です。平衡付近のH₂ は弱く相関した系であり、単一の参照行列式がこれほどうまく働くのはそのためです。結合を伸ばせばこの数値は下がります。まさにそれゆえに、結合解離が標準的な難問となっているのです。
  * **この一致が証明するのは実装の正しさであって、優位性ではありません。** 計算全体はノートPC上で1秒に満たない時間で走り、「量子的な」ステップは2次元空間上で1個の角度を最適化しただけです。実証されたのは、パイプラインが端から端まで正しいということです。

## 4.7 何が変わったのか、そして次に来るもの

本章を、それに先立つおもちゃのモデルと比べてみましょう。

  * **係数はもはや発明されたものではありません。** 第3章では \\(0.5, 0.5, 0.25, 0.3\\) という数値が、読みやすい例をつくるために選ばれていました。ここでは量子ビットハミルトニアンのすべての係数が、4添字変換、SCFループ、そしてガウス積分をさかのぼって、公表された6個の基底関数パラメータと選ばれた1つの結合長へと辿り着きます。
  * **本物の近似の階層があります。** 最小基底、次に平均場、次に基底の内側での厳密解です。各水準に数値が結び付いており、そのどれもが実行結果から出てきました。VQEエネルギーはハートリー・フォックを \\(0.0206\\) ハートリー改善し、FCIは何も改善しません。この基底の内側ではFCIが天井だからです。
  * **重労働をこなしたのは対称性であり、厳密対角化は手の届く範囲にありました。** 4量子ビットは2次元の部分空間と1個のパラメータになりましたが、それは近似によってではなく、ハミルトニアンが真に従う保存則を利用したことによります。真剣なVQEの実装はどれも、何らかの形でこれを行っています。そして、私たちが対角化したFCI空間は16次元でした。それこそが要点であり、そしてそれは一時的なことです。

最後の項目こそ、第5章が始まる場所です。ヒルベルト空間はスピン軌道の数に対して指数的に増大するので、ここでは \\(16 \times 16\\) の配列に収まったFCI行列は、化学的に興味のある分子では天文学的に巨大になります。これがこの企て全体の動機であり、同時にそのどれもが容易でない理由でもあります。第5章では、系が大きくなるにつれて実際に何が壊れるのかを検討します。パウリ項の数とそれらが要求するショット数、化学的に十分なアンザッツの深さ、最適化ランドスケープにおける不毛の台地、そして存在するデモンストレーションと有用となるであろう計算との隔たりです。それは正直な会計の章であり、最後に置かれるにふさわしいものです。

### 🎯 演習問題

  1. **核を動かす。** \\(R = 1.0\\), \\(1.4\\), \\(2.0\\), \\(3.0\\) ボーアでコードを再実行し、RHF、FCI、相関エネルギーを表にしてください。FCI基底状態におけるハートリー・フォック行列式の重みが最も大きく落ち込むのはどの距離で、なぜそれが平均場的手法にとって最も苦手な場合になるのでしょうか。
  2. **Boys関数の極限。** \\(x \to 0\\) のとき \\(F_0(x) \to 1\\) であることを示し、最初の補正項 \\(F_0(x) \approx 1 - x/3\\) を導いてください。次にコードから小さい \\(x\\) の分岐を取り除き、結果が目に見えて誤るようになる距離を見つけてください。
  3. **検算を壊す。** 4添字変換において2つの添字ラベルを意図的に入れ替えてください。印字される相互検算のどれがそれを捕まえ、どれが黙ったままでしょうか。そこから、科学計算のコードのどこにアサーションを置くべきかについて何が言えるでしょうか。
  4. **項を数える。** \\(M\\) 個の空間軌道に対して、第二量子化されたハミルトニアンは \\(O(M^4)\\) 個の二電子係数を持ちます。\\(M = 2\\), \\(10\\), \\(50\\) についてその個数を評価し、第3章のショット数のスケーリングの議論と組み合わせて、測定コストがどう増大するかを見積もってください。
  5. **変分的下界を、意図的に破る。** \\(\theta\\) を \\([0, \pi]\\) に制限して再最適化してください。エネルギーがFCI値の上にとどまることを数値的に示し、それより下の数値を誤って報告することがありえないことを保証している変分原理の性質がどれかを説明してください。

## まとめ

本章では、量子化学から量子ビットまでの完全なパイプラインをNumPyで構築し、\\(R = 1.4\\) ボーアのH₂ に対して走らせました。水素に対する **STO-3Gパラメータ**——基底の定義を成す6個の公表された数値であり、本章における唯一の外部入力です——から出発して、重なり積分、運動エネルギー積分、核引力積分、二電子積分を、縮約されたs型ガウス関数について閉じた形で評価しました。用いたのは **ガウス積の定理** と、`math.erf` だけで計算した **Boys関数** \\(F_0(x) = \frac{1}{2}\sqrt{\pi/x}\,\mathrm{erf}(\sqrt{x})\\) です。意図的に偏らせた密度から出発した **制限ハートリー・フォック** のSCFループは単調に収束し、**RHFの全エネルギー \\(-1.116714325\\) ハートリー**（電子項 \\(-1.831000039\\) に核間反発 \\(+0.714285714\\) を加えたもの）を与え、初期推定が壊していた対称な結合性軌道を回復しました。分子軌道へ変換して \\(h_{pq}\\) と \\((pq|rs)\\) を得て、そこからRHFエネルギーを再構成することで \\(4 \times 10^{-16}\\) ハートリーの精度で相互検算しました。**Jordan-Wigner変換** は4個のスピン軌道を4個の量子ビットへ、第二量子化されたハミルトニアンを明示的な \\(16 \times 16\\) 行列へと変え、コードは使用前にそのフェルミオン的反交換関係を検証しました。厳密対角化は \\(N=2, S_z=0\\) セクターにおいて **FCIエネルギー \\(-1.137275944\\) ハートリー** を与え、したがって **相関エネルギー \\(-0.020561619\\) ハートリー** を与えました。これは *この基底の内側での* 厳密解と平均場の差であり、変分原理が要求するとおりFCIはRHFの下にあります。最後に、ハミルトニアンのブロック構造を利用して、\\(|1100\rangle\\) と \\(|0011\rangle\\) が張る部分空間上の **1パラメータのVQE** を厳密なパラメータシフト勾配で収束させ、\\(-1.137275944\\) ハートリーを得て、コード自身のFCI値と \\(2 \times 10^{-16}\\) の精度で一致させました。最適化された状態は、なお性格としては98.7%がハートリー・フォックです。

次章は、本章が入念に準備してきた問いを立てます。ここでのすべては、どんな量子デバイスにも及ばないほど速く正確に、ノートPC上で走りました——では、分子が大きくなると何が変わるのか、そして量子コンピュータがこの計算の中でその居場所を得るためには、何が真でなければならないのでしょうか。

[← 第3章: VQE — アルゴリズムの全体像](<chapter-3.html>) [第5章: H2 の先へ — 誠実なフロンティア →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
