---
title: "第4章: 量子化学・材料計算への応用"
chapter_title: "第4章: 量子化学・材料計算への応用"
subtitle: ⚛️ 第二量子化、Jordan-Wigner変換、そして自分で対角化できるモデルハミルトニアン
reading_time: 40-45分
difficulty: 上級
code_examples: 6
exercises: 6
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-computing-introduction/chapter-4.html>) | Last sync: 2026-08-12

[基礎数理道場](<../index.html>) > [量子コンピューティング入門](<index.html>) > 第4章

第3章では、係数が表として与えられた2量子ビットハミルトニアンに対して変分量子固有値ソルバーを走らせました。あの表は天から降ってきたものではありません。電子、軌道、生成消滅演算子、フェルミオン-量子ビット写像という一連の鎖の終着点であり、その鎖のすべての環が、量子コンピュータが材料研究に対して実際に何をなしうるかを制約しています。本章はこの鎖を明示的に組み立てます。読み終えたとき、格子模型あるいは小さな活性空間を受け取ってPauli文字列の和として書き下し、それを手元のノートPCで厳密対角化できるようになります。量子アルゴリズムが同じ問題に対して正しい答えを出したかどうかを知る、正直な方法はそれしかありません。

本章には2つの主題が通っています。第1は**電子構造問題がなぜ難しいのか**です。方程式が未知だからではありません。厳密解が組み合わせ論的に増大する次元の空間に住んでいるからであり、そして弱相関物質に対して見事に機能する近似（何よりも密度汎関数理論）が、面白い材料のあるまさにその場所で破綻するからです。第2は**変換の各段階が支払う代償**です。Jordan-Wigner変換は反交換関係を局所性の犠牲によって買い、Trotter分解は回路を $1/r$ でしか減らない誤差の犠牲によって買い、位相推定は精度を深さの犠牲によって買います。第5章では、今日のハードウェアがその代金をどこまで払えるかに数値を与えます。本章で確立するのは「何が買われているのか」です。

4.2節以降で使う第二量子化の形式は[量子場の理論入門](<../quantum-field-theory-introduction/index.html>)コースでより深く展開されています。4.6節の背後にある変分原理は[量子力学入門](<../quantum-mechanics/index.html>)コースにあり、テンソルと固有値の道具立ては[線形代数とテンソル解析](<../linear-algebra-tensor/index.html>)コースにあります。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * 厳密な電子構造問題のスケーリングを定量化し、古典的なfull CIの壁が軌道数・行列式数・バイト数で実際にどこにあるかを述べられる
  * 密度汎関数理論が強相関材料で破綻する理由を説明し、問題が強相関であることを示す物理的兆候（準縮退、分数占有、Mott絶縁性）を同定できる
  * フェルミオンのハミルトニアンを第二量子化形式で書き、任意の量子ビット写像が再現しなければならない反交換関係を述べられる
  * Jordan-Wigner変換をゼロから実装し、正準反交換関係を数値的に検証し、パリティ文字列が局所性を壊す理由を説明できる
  * デジタル量子シミュレーションとアナログ量子シミュレーションを区別し、量子位相推定と変分量子固有値ソルバーを深さ・精度・ハードウェア要求の観点で比較できる
  * 1次および2次のTrotter積公式の誤差を測定し、目標精度が要求するゲート数を外挿できる
  * 横磁場Ising鎖と2サイトHubbard模型の量子ビットハミルトニアンを構成し、双方を厳密対角化して物理的に解釈できる（量子相転移、Mott局在、超交換相互作用）
  * 同じハミルトニアンに対してVQEを走らせ、3つの誤差源 — ansatzの表現能力、最適化器の収束、測定統計 — を切り分けて診断できる

* * *

## 4.1 電子構造問題のスケーリング

### 問題は方程式ではない

Born-Oppenheimer近似のもとで、分子あるいは固体の電子ハミルトニアンは完全に既知です：

$$ \hat{H} = -\sum_i \frac{\hbar^2}{2m_e}\nabla_i^2 - \sum_{i,A} \frac{Z_A e^2}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{R}_A|} + \sum_{i<j} \frac{e^2}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{r}_j|} $$

ここに発見すべきものは何もありません。難しさは完全に計算上のものであり、その原因は最後の項にあります。電子間反発がすべての座標を他のすべての座標に結合させるため、波動関数は因子分解しません。

多電子波動関数を $M$ 個の空間軌道から作られるSlater行列式で展開します。上向きスピン電子 $N_\alpha$ 個、下向きスピン電子 $N_\beta$ 個のとき、行列式の数は

$$ D = \binom{M}{N_\alpha}\binom{M}{N_\beta} $$

です。これが**完全配置間相互作用**（full CI）空間の次元、すなわち選んだ軌道基底の中での厳密解の次元です。二項係数であり、二項係数は容赦がありません。

### 壁はどこにあるか

量子コンピュータはスピン軌道1個あたり1量子ビットを必要とするので、$M$ 個の空間軌道は $2M$ 量子ビットになります。頭に入れておくべき比較は、$D$（古典的な厳密計算が保持しなければならない量）と $2M$（量子コンピュータが用意しなければならない量）の間の比較です。

ここで記号を1つ固定し、本章の残りでは一貫して用います。$M$ は常に**空間**軌道の数を表し、系はスピン軌道 $2M$ 個をもち $2M$ 量子ビットを必要とします。4.2節以降のようにスピン軌道について走る添字の範囲は $0 \ldots 2M-1$ です。

Code Example 1: 厳密な問題はどれだけ速く増大するか

```python
"""Chapter 4, Example 1: how fast the exact electronic-structure problem grows."""
import numpy as np
from math import comb


def fci_dimension(n_orb: int, n_alpha: int, n_beta: int) -> int:
    """Number of Slater determinants in a full-CI expansion."""
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


print("Exact diagonalization of the electronic-structure problem")
print("(closed shell, half filling, one qubit per spin orbital)\n")
print(f"{'n_orb':>6} {'n_elec':>7} {'qubits':>7} "
      f"{'FCI determinants':>20} {'2^qubits':>12} {'FCI vector':>15}")
print("-" * 72)
for n_orb in (2, 4, 8, 12, 16, 20, 30, 50, 100):
    n_alpha = n_beta = n_orb // 2            # closed shell, half filled
    dim = fci_dimension(n_orb, n_alpha, n_beta)
    n_qubits = 2 * n_orb                     # one qubit per spin orbital
    mem_gib = dim * 16 / 2 ** 30             # one complex128 amplitude each
    print(f"{n_orb:6d} {2*n_alpha:7d} {n_qubits:7d} "
          f"{dim:20.6e} {2.0**n_qubits:12.3e} {mem_gib:11.3e} GiB")

print("\nWhere the classical wall is:")
prev = None
for n_orb in (16, 18, 20, 22, 24):
    dim = fci_dimension(n_orb, n_orb // 2, n_orb // 2)
    growth = f"  x{dim/prev:5.2f}" if prev else "        "
    print(f"  n_orb = {n_orb:3d}: {dim:>18,d} determinants, "
          f"{dim * 16 / 2**40:9.3f} TiB per vector{growth}")
    prev = dim

# FeMoco: FeMo補因子全体の (113電子, 76空間軌道) 活性空間。本シリーズを通して
# この規約を用いる（第1章と同じ）。
print("\nQubit count for representative active spaces:")
for name, n_orb, n_elec in (
        ("H2 / STO-3G", 2, 2),
        ("LiH / STO-3G", 6, 4),
        ("N2 triple bond", 6, 6),
        ("FeMoco active space (literature scale)", 76, 113)):
    print(f"  {name:40s}: {n_elec:3d} e- in {n_orb:3d} orbitals"
          f" -> {2*n_orb:4d} qubits")
```

```text
Exact diagonalization of the electronic-structure problem
(closed shell, half filling, one qubit per spin orbital)

 n_orb  n_elec  qubits     FCI determinants     2^qubits      FCI vector
------------------------------------------------------------------------
     2       2       4         4.000000e+00    1.600e+01   5.960e-08 GiB
     4       4       8         3.600000e+01    2.560e+02   5.364e-07 GiB
     8       8      16         4.900000e+03    6.554e+04   7.302e-05 GiB
    12      12      24         8.537760e+05    1.678e+07   1.272e-02 GiB
    16      16      32         1.656369e+08    4.295e+09   2.468e+00 GiB
    20      20      40         3.413478e+10    1.100e+12   5.086e+02 GiB
    30      30      60         2.406145e+16    1.153e+18   3.585e+08 GiB
    50      50     100         1.597964e+28    1.268e+30   2.381e+20 GiB
   100     100     200         1.017906e+58    1.607e+60   1.517e+50 GiB

Where the classical wall is:
  n_orb =  16:        165,636,900 determinants,     0.002 TiB per vector        
  n_orb =  18:      2,363,904,400 determinants,     0.034 TiB per vector  x14.27
  n_orb =  20:     34,134,779,536 determinants,     0.497 TiB per vector  x14.44
  n_orb =  22:    497,634,306,624 determinants,     7.242 TiB per vector  x14.58
  n_orb =  24:  7,312,459,672,336 determinants,   106.410 TiB per vector  x14.69

Qubit count for representative active spaces:
  H2 / STO-3G                             :   2 e- in   2 orbitals ->    4 qubits
  LiH / STO-3G                            :   4 e- in   6 orbitals ->   12 qubits
  N2 triple bond                          :   6 e- in   6 orbitals ->   12 qubits
  FeMoco active space (literature scale)  : 113 e- in  76 orbitals ->  152 qubits
```

**着目点。** 軌道を2個追加するたびに行列式の数は約14倍になり、24軌道でもその係数はまだ増加中です。20軌道の活性空間は波動関数ベクトル1本で半テラバイトを要し、24軌道なら100テラバイト、しかも厳密固有値ソルバーはそのようなベクトルを同時に何本も必要とします。これが実運用のfull CI計算が20軌道付近で止まる理由であり、あらゆる強相関研究に「活性空間」という語が登場する理由です。少数の軌道を選んで厳密に扱い、残りは退屈であることを願うのです。

同じ表は、よくある過大な言い方も打ち砕きます。量子コンピュータの優位はここでは**メモリ的**なものです。40量子ビットは $1.1 \times 10^{12}$ 次元ベクトルと同じ情報を保持し、152量子ビットならFeMocoの活性空間を保持できます。しかし状態を保持することは基底状態を見つけることと同じではなく、しかも量子ビット数は私たちが見積もる3つの資源のうち**最も安い**ものです。深さと測定コストのほうがはるかに厳しく、4.4節と第5章がそれを定量化します。

### なぜDFTではいけないのか

密度汎関数理論は、波動関数ではなく電子密度から基底状態の性質を計算し、そのコストは指数関数的ではなくおよそ $O(M^3)$ です。計算材料科学において圧倒的に最も成功した手法であり、周期表の大部分に対しては正しい道具です。

その破綻は系統的で、よく特徴づけられています。

領域 | 物理的兆候 | DFTが苦しむ理由
---|---|---
遷移金属酸化物 | 部分的に占有された局在 $d$ 殻 | 自己相互作用誤差が電子を過剰に非局在化し、バンドギャップが潰れる
Mott絶縁体 | バンドが半充填なのに絶縁性 | 絶縁機構が相互作用起源で、単一行列式の参照状態に存在しない
結合解離 | 大きな $R$ で2つの配置が準縮退 | 静的相関は1つの行列式では捉えられない
遷移金属触媒 | 複数のスピン状態が数kcal/mol以内 | 相対スピン状態エネルギーの誤差が、関心のあるエネルギー差を上回る
ランタノイド・アクチノイド | 局在した $f$ 電子 | 強相関に加えて相対論的効果
非従来型超伝導体・スピン液体 | 長距離エンタングルメント | 基底状態がどの単一行列式にも近くない。従来型のBCS超伝導体は逆の場合で、平均場状態であり DFT + Eliashberg 理論でよく記述できる

共通する糸は**静的（強）相関**です。厳密な基底状態が同程度の重みをもつ複数の行列式の重ね合わせであり、いかなる単一参照法 — DFT、Hartree-Fock、単一参照の結合クラスタ — もそれを表現できません。これはまさに、量子コンピュータが一般の重ね合わせを保持できる能力が原理的に正しい道具となる領域です。

実務的な帰結ははっきり述べておく価値があります。量子コンピューティングの有用な目標は「DFTを置き換える」ことではありません。「DFTが間違える強相関な部分を供給する」こと、典型的には数十軌道の活性空間を、はるかに大きなDFTあるいは古典的相関の取り扱いの中に埋め込むことです。本章のすべてはその部分に向けられています。

### 古典側も立ち止まっていない

量子優位性のいかなる評価も、full CIではなく最良の古典手法と比較しなければなりません。

手法 | コスト | 得意なところ | 破綻するところ
---|---|---|---
Full CI | $O(D)$、$D$ は二項係数 | 厳密なベンチマーク、$\lesssim 20$ 軌道 | それより大きいものすべて
DMRG / MPS | ボンド次元 $\chi$ について $O(\chi^3)$ | 1次元・準1次元、低エンタングルメント | エンタングルメントの大きい2次元・3次元
量子モンテカルロ | 多項式、統計的 | ボソン、符号問題のないフェルミオン模型 | フェルミオン符号問題：指数関数的コスト
結合クラスタ（CCSD(T)） | $O(M^7)$ | 弱相関、閉殻 | 強相関、結合切断
テンソルネットワーク（PEPS） | 条件付きで多項式 | 2次元の面積則状態 | 高エンタングルメント、収縮が困難

機会についての正直な言明は狭いものです。量子コンピューティングが候補になるのは、**強相関**であり（したがってDFTとCCSD(T)が破綻し）、**テンソルネットワークには過剰にエンタングルしており**（したがってDMRGが破綻し）、かつ**符号問題を抱えている**（したがって量子モンテカルロが破綻する）問題です。その交わりは実在します — ドープされた2次元Hubbard模型といくつかの遷移金属活性空間がそこに位置します — が、「量子コンピュータが化学をシミュレートする」という言い方が示唆するよりずっと小さいのです。

* * *

## 4.2 第二量子化とフェルミオン

### 座標ではなく占有数

反対称化された波動関数を手で書くのは不快であり、一般化もできません。第二量子化は「どの電子がどの軌道にいるか」を「各軌道に電子が何個いるか」に置き換え、反対称性を行列式ではなく演算子の代数によって強制します。包括的な扱いは[量子場の理論入門](<../quantum-field-theory-introduction/index.html>)コースにあります。ここで必要なのは実用の規則だけです。

$2M$ 個のスピン軌道の組 $\lbrace \phi_0, \phi_1, \ldots, \phi_{2M-1} \rbrace$ を順序付けて固定します。基底状態は占有数の列

$$ \lvert n_0 n_1 \cdots n_{2M-1} \rangle, \qquad n_p \in \lbrace 0, 1 \rbrace $$

であり、$n_p = 1$ は軌道 $p$ が占有されていることを意味します。Pauliの排他原理はこれで自動的です。$n_p$ は1を超えられません。

生成消滅演算子の作用は

$$ \hat{c}_p^\dagger \lvert \cdots n_p = 0 \cdots \rangle = (-1)^{\sigma_p} \lvert \cdots n_p = 1 \cdots \rangle, \qquad \hat{c}_p^\dagger \lvert \cdots n_p = 1 \cdots \rangle = 0 $$

$$ \hat{c}_p \lvert \cdots n_p = 1 \cdots \rangle = (-1)^{\sigma_p} \lvert \cdots n_p = 0 \cdots \rangle, \qquad \hat{c}_p \lvert \cdots n_p = 0 \cdots \rangle = 0 $$

であり、符号は $p$ より左側の占有の**パリティ**で決まります：

$$ \sigma_p = \sum_{q<p} n_q $$

このパリティ因子がフェルミオンの反対称性の内容そのものであり、量子ビット写像を非自明にしているものです。それ以外は帳簿付けにすぎません。

### 反交換関係

定義となる代数は

$$ \lbrace \hat{c}_p, \hat{c}_q^\dagger \rbrace = \hat{c}_p \hat{c}_q^\dagger + \hat{c}_q^\dagger \hat{c}_p = \delta_{pq}, \qquad \lbrace \hat{c}_p, \hat{c}_q \rbrace = 0, \qquad \lbrace \hat{c}_p^\dagger, \hat{c}_q^\dagger \rbrace = 0 $$

です。量子ビット写像の候補が正しいのは、これらの関係を厳密に再現するとき、そしてそのときに限ります。4.3節では4モード系についてこれら $2 \times 4^2$ 個すべてを数値的に検証します。符号を1つ間違えた写像は、もっともらしく見えて静かに誤ったハミルトニアンを生み出すからです。

数演算子は $\hat{n}_p = \hat{c}_p^\dagger \hat{c}_p$ で固有値は0と1、全粒子数は $\hat{N} = \sum_p \hat{n}_p$ です。

### 電子ハミルトニアン

この言語では電子ハミルトニアンは2行で書ける対象になります：

$$ \hat{H} = \sum_{pq} h_{pq} \hat{c}_p^\dagger \hat{c}_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s $$

1電子積分と2電子積分は

$$ h_{pq} = \int d\mathbf{x}\, \phi_p^\ast(\mathbf{x}) \left(-\frac{\hbar^2}{2m_e}\nabla^2 + V_{\text{nuc}}(\mathbf{x})\right) \phi_q(\mathbf{x}) $$

$$ h_{pqrs} = \frac{e^2}{4\pi\epsilon_0}\int d\mathbf{x}_1 d\mathbf{x}_2\, \frac{\phi_p^\ast(\mathbf{x}_1)\phi_q^\ast(\mathbf{x}_2)\phi_r(\mathbf{x}_2)\phi_s(\mathbf{x}_1)}{|\mathbf{x}_1 - \mathbf{x}_2|} $$

です。以降のために重要な帰結が3つあります。

  1. **積分は古典的な入力である。** これらは古典コンピュータ上のHartree-FockあるいはDFT計算から得られます。量子アルゴリズムはこれを消費するのであって、生成するのではありません。
  2. **積分は $O(M^4)$ 個ある。** 2電子テンソルは $M^4$ 個の成分をもち（対称性で減りますがスケーリングは同じ）、各成分がPauli文字列の組になります。20個の空間軌道なら $10^5$ 個のオーダーの項です。そのすべてを測定しなければならず、これが第5章で定量化する測定ボトルネックの起源です。
  3. **形は普遍である。** 格子模型はほとんどの積分をゼロにした同じ表式です。Hubbard模型はホッピング振幅1つと在サイト反発1つだけを残し、横磁場Ising鎖はさらにスピンへ簡約した後に残るものです。だからこそ4量子ビットの模型で練習しながら、本物の機構を学ぶことができます。

材料研究は、電子をすでに積分してしまった**有効**模型にも関心をもちます。最も重要なのはHubbard模型の大 $U$ 極限で、電荷ゆらぎが凍結してスピンだけが残り、Heisenberg模型 $\hat{H} = J \sum_{\langle ij \rangle} \left(\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j - \tfrac{1}{4}\right)$（ただし $J = 4t^2/U$）を与えます。ボンドあたりの定数 $-J/4$ は飾りではありません。同じ2次過程から出てくる項であり、これを落とすと1ボンドの一重項エネルギーが $-J$ ではなく $-3J/4$ になり、$4/3$ 倍だけ誤ります。4.6節ではこの $-J = -4t^2/U$ を数値で確認します。スピン模型は量子ビットへ直接写ります — 量子ビット1個につきスピン1/2が1個、パリティ文字列もJordan-Wignerも不要 — ので、量子シミュレーションの標的としては最も安い部類です。4.6節では $U/t = 128$ で $4t^2/U$ 則が0.1%以内で現れる様子を観察します。

* * *

## 4.3 Jordan-Wigner変換

### 写像

反交換関係を満たす演算子を、自然な演算子（異なる量子ビット上のPauli行列）が**可換**である量子ビットで表現する必要があります。Jordan-Wigner変換はこれを、左側の占有のパリティを数える $Z$ 演算子の列を付けることで解決します：

$$ \hat{c}_p = \left(\prod_{q<p} Z_q\right) \frac{X_p + iY_p}{2}, \qquad \hat{c}_p^\dagger = \left(\prod_{q<p} Z_q\right) \frac{X_p - iY_p}{2} $$

局所因子は量子ビット $p$ 上の昇降演算子です：

$$ \frac{X - iY}{2} = \begin{pmatrix} 0 & 0 \\\\ 1 & 0 \end{pmatrix}, \qquad \frac{X + iY}{2} = \begin{pmatrix} 0 & 1 \\\\ 0 & 0 \end{pmatrix} $$

本シリーズの規約（量子ビット0 = 左端ビット = 最上位ビット）では、占有1が $\lvert 1 \rangle$ に対応し、数演算子は見事に簡単になります：

$$ \hat{n}_p = \hat{c}_p^\dagger \hat{c}_p = \frac{I - Z_p}{2} $$

$Z$ 文字列は飾りではありません。$Z$ が占有軌道に対して $-1$、空軌道に対して $+1$ を返すため、4.2節の $(-1)^{\sigma_p}$ の符号をちょうど供給します。

### 代償：局所性

$p < q$ の軌道間のホッピング項を考えます。$Z$ 文字列が部分的に打ち消し合い、

$$ \hat{c}_p^\dagger \hat{c}_q + \hat{c}_q^\dagger \hat{c}_p = \frac{1}{2}\left(X_p Z_{p+1}\cdots Z_{q-1} X_q + Y_p Z_{p+1} \cdots Z_{q-1} Y_q\right) $$

が残ります。フェルミオンの言語では局所的だった項 — 2つの軌道 — が、$q - p + 1$ 個の量子ビットに作用するPauli文字列になりました。**Pauli重み**が軌道間距離に比例して増大します。接続性の限られたハードウェアでは、そのような文字列は重みに比例したCNOTのはしごを要するので、写像の非局所性はそのまま回路の深さに変換されます。

これが代替案の動機です。

写像 | スピン軌道あたり量子ビット | ホッピング項の重み | 備考
---|---|---|---
Jordan-Wigner | 1 | $O(M)$ | 最も単純。$\hat{n}_p$ は局所のまま
パリティ | 1 | $O(M)$ | 相補的な構造。対称性による2量子ビット削減が可能
Bravyi-Kitaev | 1 | $O(\log M)$ | 漸近的な重みは最良。帳簿付けはより複雑
3進木・平衡木 | 1 | $O(\log M)$ | 重み最適な構成
コンパクト・局所符号化 | $> 1$ | $O(1)$ | 余分な量子ビットで格子模型の局所性を買う

本章の4モード系ではJordan-Wignerに何のコストもありません（最長の文字列はどちらの写像でも4量子ビットに及ぶ）し、実装と検証が圧倒的に明快です。100軌道の計算では $O(\log M)$ の写像が重要になります。

### 実装

以下のコードは本章の残りのための道具箱です。演算子をPauli文字列から係数への辞書として表現し、Pauli代数（位相を伴う積、打ち消しを伴う和）を実装し、$\hat{c}_p$、$\hat{c}_p^\dagger$、$\hat{n}_p$、ホッピング項のJordan-Wigner像を構成し、厳密対角化のための行列表現を提供します。

末尾の自己検査が重要な部分です。$2 \times 4^2$ 個すべての反交換関係を検証するので、この写像で計算されたエネルギーを信用する前に、写像が正しいことがわかります。

Code Example 2: Pauli文字列代数とJordan-Wigner変換

```python
"""Chapter 4, Example 2: Pauli-string algebra and the Jordan-Wigner transform.

This block is the toolbox for the rest of the chapter: run it first, then
Examples 3-6 in the same session (or paste everything into one file).
"""
import numpy as np

TOL = 1e-12

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

# ---------------------------------------------------------------------
# Pauli-string algebra.  An operator is a dict {'IXZY': coefficient, ...}
# Character j of the string acts on qubit j (qubit 0 = leftmost = MSB).
# ---------------------------------------------------------------------

# single-qubit product table:  a * b = phase * c
_MUL = {
    ('I', 'I'): (1, 'I'),   ('I', 'X'): (1, 'X'),   ('I', 'Y'): (1, 'Y'),   ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'),   ('X', 'X'): (1, 'I'),   ('X', 'Y'): (1j, 'Z'),  ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'),   ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'),   ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'),   ('Z', 'X'): (1j, 'Y'),  ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
}


def pauli_mul(a: str, b: str):
    """Product of two equal-length Pauli strings -> (phase, string)."""
    phase, out = 1.0 + 0j, []
    for ca, cb in zip(a, b):
        p, c = _MUL[(ca, cb)]
        phase *= p
        out.append(c)
    return phase, ''.join(out)


def op_add(*ops):
    """Sum of Pauli-string operators; drops numerically vanishing terms."""
    total = {}
    for op in ops:
        for s, c in op.items():
            total[s] = total.get(s, 0) + c
    return {s: c for s, c in total.items() if abs(c) > TOL}


def op_scale(op, alpha):
    return {s: alpha * c for s, c in op.items()}


def op_mul(op1, op2):
    """Operator product, distributed over Pauli strings."""
    out = {}
    for s1, c1 in op1.items():
        for s2, c2 in op2.items():
            ph, s = pauli_mul(s1, s2)
            out[s] = out.get(s, 0) + c1 * c2 * ph
    return {s: c for s, c in out.items() if abs(c) > TOL}


def op_str(op):
    """Human-readable Pauli-string linear combination."""
    if not op:
        return "0"
    parts = []
    for s, c in sorted(op.items()):
        c = complex(c)
        parts.append(f"{c.real:+.4f}*{s}" if abs(c.imag) < TOL
                     else f"({c.real:+.4f}{c.imag:+.4f}j)*{s}")
    return "  ".join(parts)


def op_weight(op):
    """Largest number of non-identity factors in any string of the operator."""
    return max(sum(1 for ch in s if ch != 'I') for s in op)


# ---------------------------------------------------------------------
# Jordan-Wigner transform
#   c_p     = (Z_0 ... Z_{p-1}) (X_p + i Y_p) / 2
#   c_p^dag = (Z_0 ... Z_{p-1}) (X_p - i Y_p) / 2
#   n_p     = c_p^dag c_p = (I - Z_p) / 2
# ---------------------------------------------------------------------

def jw_annihilate(p: int, n: int):
    """Annihilation operator for spin orbital p, mapped onto n qubits."""
    head = 'Z' * p                        # the parity (Jordan-Wigner) string
    tail = 'I' * (n - p - 1)
    return {head + 'X' + tail: 0.5 + 0j,
            head + 'Y' + tail: 0.5j}


def jw_create(p: int, n: int):
    """Creation operator for spin orbital p, mapped onto n qubits."""
    head = 'Z' * p
    tail = 'I' * (n - p - 1)
    return {head + 'X' + tail: 0.5 + 0j,
            head + 'Y' + tail: -0.5j}


def jw_number(p: int, n: int):
    """Occupation-number operator n_p = (I - Z_p) / 2."""
    return {'I' * n: 0.5 + 0j,
            'I' * p + 'Z' + 'I' * (n - p - 1): -0.5 + 0j}


def jw_hop(p: int, q: int, n: int):
    """Hermitian hopping term c_p^dag c_q + c_q^dag c_p."""
    return op_add(op_mul(jw_create(p, n), jw_annihilate(q, n)),
                  op_mul(jw_create(q, n), jw_annihilate(p, n)))


# ---------------------------------------------------------------------
# Matrix realisation, for verification and exact diagonalization
# ---------------------------------------------------------------------

def pauli_matrix(s: str) -> np.ndarray:
    """Kronecker product in big-endian order: qubit 0 is the outermost factor."""
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, PAULI[ch])
    return M


def to_matrix(op) -> np.ndarray:
    n = len(next(iter(op)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in op.items():
        M += c * pauli_matrix(s)
    return M


# =====================================================================
n = 4                                     # four spin orbitals -> four qubits

print("Jordan-Wigner images of the elementary operators (n = 4)")
print("-" * 68)
for p in range(n):
    print(f"  c_{p}      = {op_str(jw_annihilate(p, n))}")
print()
for p in range(n):
    print(f"  n_{p}      = {op_str(jw_number(p, n))}")

print()
print("Canonical anticommutation relations (the whole point of JW)")
print("-" * 68)
ident = 'I' * n
ok = True
for p in range(n):
    for q in range(n):
        ac = op_add(op_mul(jw_annihilate(p, n), jw_create(q, n)),
                    op_mul(jw_create(q, n), jw_annihilate(p, n)))
        expected = {ident: 1.0 + 0j} if p == q else {}
        match = (set(ac) == set(expected)
                 and all(abs(ac[k] - expected[k]) < 1e-12 for k in expected))
        ok &= match
        if p <= q <= p + 1:
            print(f"  {{c_{p}, c_{q}^dag}} = {op_str(ac):<28s} "
                  f"expected {'I' if p == q else '0'}   {'OK' if match else 'FAIL'}")
        ac2 = op_add(op_mul(jw_annihilate(p, n), jw_annihilate(q, n)),
                     op_mul(jw_annihilate(q, n), jw_annihilate(p, n)))
        ok &= (len(ac2) == 0)
print(f"  all 2 x {n}^2 relations verified: {ok}")

print()
print("Hopping terms: locality is destroyed by the parity string")
print("-" * 68)
for (p, q) in ((0, 1), (0, 2), (0, 3), (1, 3)):
    op = jw_hop(p, q, n)
    print(f"  c_{p}^dag c_{q} + h.c. = {op_str(op)}")
    print(f"       -> {len(op)} Pauli strings, maximum weight {op_weight(op)}")

print()
print("Pauli weight of c_p^dag c_q grows linearly with orbital distance")
print("-" * 68)
for dist in range(1, 8):
    op = jw_hop(0, dist, 12)
    print(f"  |p-q| = {dist}: weight {op_weight(op):2d}  ({len(op)} strings)")

print()
print("Consistency check: the mapped number operator on a basis state")
print("-" * 68)
# |1010> means orbitals 0 and 2 occupied (big-endian bit string)
psi = np.zeros(2 ** n, dtype=complex)
psi[int('1010', 2)] = 1.0
for p in range(n):
    occ = np.real(np.vdot(psi, to_matrix(jw_number(p, n)) @ psi))
    print(f"  <1010| n_{p} |1010> = {occ:.1f}")
N_tot = to_matrix(op_add(*[jw_number(p, n) for p in range(n)]))
print(f"  total particle number = {np.real(np.vdot(psi, N_tot @ psi)):.1f}")
```

```text
Jordan-Wigner images of the elementary operators (n = 4)
--------------------------------------------------------------------
  c_0      = +0.5000*XIII  (+0.0000+0.5000j)*YIII
  c_1      = +0.5000*ZXII  (+0.0000+0.5000j)*ZYII
  c_2      = +0.5000*ZZXI  (+0.0000+0.5000j)*ZZYI
  c_3      = +0.5000*ZZZX  (+0.0000+0.5000j)*ZZZY

  n_0      = +0.5000*IIII  -0.5000*ZIII
  n_1      = +0.5000*IIII  -0.5000*IZII
  n_2      = +0.5000*IIII  -0.5000*IIZI
  n_3      = +0.5000*IIII  -0.5000*IIIZ

Canonical anticommutation relations (the whole point of JW)
--------------------------------------------------------------------
  {c_0, c_0^dag} = +1.0000*IIII                 expected I   OK
  {c_0, c_1^dag} = 0                            expected 0   OK
  {c_1, c_1^dag} = +1.0000*IIII                 expected I   OK
  {c_1, c_2^dag} = 0                            expected 0   OK
  {c_2, c_2^dag} = +1.0000*IIII                 expected I   OK
  {c_2, c_3^dag} = 0                            expected 0   OK
  {c_3, c_3^dag} = +1.0000*IIII                 expected I   OK
  all 2 x 4^2 relations verified: True

Hopping terms: locality is destroyed by the parity string
--------------------------------------------------------------------
  c_0^dag c_1 + h.c. = +0.5000*XXII  +0.5000*YYII
       -> 2 Pauli strings, maximum weight 2
  c_0^dag c_2 + h.c. = +0.5000*XZXI  +0.5000*YZYI
       -> 2 Pauli strings, maximum weight 3
  c_0^dag c_3 + h.c. = +0.5000*XZZX  +0.5000*YZZY
       -> 2 Pauli strings, maximum weight 4
  c_1^dag c_3 + h.c. = +0.5000*IXZX  +0.5000*IYZY
       -> 2 Pauli strings, maximum weight 3

Pauli weight of c_p^dag c_q grows linearly with orbital distance
--------------------------------------------------------------------
  |p-q| = 1: weight  2  (2 strings)
  |p-q| = 2: weight  3  (2 strings)
  |p-q| = 3: weight  4  (2 strings)
  |p-q| = 4: weight  5  (2 strings)
  |p-q| = 5: weight  6  (2 strings)
  |p-q| = 6: weight  7  (2 strings)
  |p-q| = 7: weight  8  (2 strings)

Consistency check: the mapped number operator on a basis state
--------------------------------------------------------------------
  <1010| n_0 |1010> = 1.0
  <1010| n_1 |1010> = 0.0
  <1010| n_2 |1010> = 1.0
  <1010| n_3 |1010> = 0.0
  total particle number = 2.0
```

**着目点。** 4つあり、それぞれ一瞬立ち止まる価値があります。

第1に、$Z$ 文字列が出力に見えています。$\hat{c}_0$ には $Z$ がなく、$\hat{c}_1$ には1個、$\hat{c}_3$ には3個あります。これがパリティのカウンタを明示したものです。

第2に、数演算子はすべて重み1です。どの $p$ についても $\hat{n}_p = (I - Z_p)/2$ で、文字列はまったくありません。Jordan-Wignerはホッピングを非局所にする一方で、占有は局所に保ちます。だからこそHubbard模型の在サイトCoulomb項 $U \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}$ が、4.6節では単純な2量子ビット $ZZ$ 相互作用になるのです。

第3に、$2 \times 4^2 = 32$ 個すべての反交換関係が厳密に成立し、$\lbrace \hat{c}_0, \hat{c}_1^\dagger \rbrace$ は小さな数ではなく**空の演算子**として印字されます。打ち消しは数値的ではなく代数的なのです。$Z$ 文字列を忘れていたら、この項はゼロでないPauli文字列として現れ、結果のハミルトニアンは偽のスペクトルをもったはずです。

第4に、重みの表が線形増大を示しています。隣接軌道間のホッピングは重み2、7軌道離れたホッピングは重み8です。$O(M^4)$ 個の項をもち典型的距離が $O(M)$ である分子ハミルトニアンでは、総Pauli重みが回路の深さを決める量になります。

* * *

## 4.4 デジタル/アナログシミュレーション、QPEとVQE

### シミュレーションの2つの流儀

**アナログ量子シミュレーション**は、そのハミルトニアンが標的ハミルトニアンに（近似的に）等しい物理系を作り、単に時間発展させます。光格子中の冷却原子はHubbard模型を実現し、イオントラップ結晶は長距離IsingおよびHeisenberg模型を実現し、超伝導回路アレイはBose-Hubbard物理を実現します。ゲートのコンパイルもTrotter誤差もなく、コヒーレンスの要求もずっと緩いのですが、装置はたまたま持っているハミルトニアンをシミュレートするだけで、校正と検証が困難です。

**デジタル量子シミュレーション**は標的の時間発展を普遍ゲート集合にコンパイルします。任意のハミルトニアンを表現でき、しかも誤り訂正と両立します。これは長期的には決定的です。その代価は深さです。

観点 | アナログ | デジタル
---|---|---
プログラマビリティ | ハードウェアで固定 | 任意のハミルトニアン
誤り訂正 | 一般には利用不可 | 両立可能
系統誤差 | 校正、不要な項 | Trotter・コンパイル誤差、制御可能
必要なコヒーレンス | 中程度 | 大きい
規模を決めるもの | 配列サイズと校正（ゲート誤差ではない） | （ゲート数）×（ゲート誤差率）が1を十分下回ること
検証 | 困難 | 回路レベル、検査可能

### Trotter分解

$e^{-i\hat{H}t}$ のデジタルシミュレーションには、指数化できる部分への分解 $\hat{H} = \sum_j \hat{H}_j$ が必要です。各部分は可換でないので素朴な積は誤りですが、1次のLie-Trotter公式が誤差を制御します：

$$ e^{-i\hat{H}t} = \left(\prod_j e^{-i\hat{H}_j t/r}\right)^r + O\!\left(\frac{t^2}{r}\right) $$

対称化された（2次Suzuki）公式は各ステップを半分にし、前向きと後向きに掃きます：

$$ e^{-i\hat{H}t} \approx \left(\prod_j e^{-i\hat{H}_j t/2r} \prod_j^{\text{reverse}} e^{-i\hat{H}_j t/2r}\right)^r + O\!\left(\frac{t^3}{r^2}\right) $$

Pauli文字列 $P$ に対する各因子 $e^{-i\theta P}$ は、CNOTのはしご、1個の $R_z$、逆はしごにコンパイルされます。これは第2章2.5節で確立した恒等式です。したがってゲート数は（Pauli文字列の数）×（Trotterステップ数）×（重みに依存するCNOTコスト）になります。

漸近記法を信用するのではなく、誤差を測ってみましょう。

Code Example 3: Trotter誤差とデジタルシミュレーションのゲートコスト

```python
"""Chapter 4, Example 3: Trotter error and the gate cost of digital simulation.
Continues from Example 2 (same session)."""


def expm_hermitian(M, scalar):
    """exp(scalar * M) for Hermitian M, via its eigendecomposition."""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(scalar * w)) @ v.conj().T


def hubbard_dimer_bare(t, U):
    """Two-site Hubbard model without the chemical-potential term.
    Mode order: 0 = (site 0, up), 1 = (site 0, dn), 2 = (site 1, up), 3 = (site 1, dn)."""
    n = 4
    Hq = {}
    for (a, b) in ((0, 2), (1, 3)):        # up-spin and down-spin hopping
        Hq = op_add(Hq, op_scale(jw_hop(a, b, n), -t))
    for (u, d) in ((0, 1), (2, 3)):        # on-site repulsion on each site
        Hq = op_add(Hq, op_scale(op_mul(jw_number(u, n), jw_number(d, n)), U))
    return Hq


t, U, tau = 1.0, 4.0, 1.0
Hq = hubbard_dimer_bare(t, U)
terms = sorted(Hq.items())
n_terms = len(terms)
# the identity string is a global phase: it needs no gate at all
n_rot = sum(1 for s, _ in terms if set(s) != {'I'})
U_exact = expm_hermitian(to_matrix(Hq), -1j * tau)

print(f"Trotter error, two-site Hubbard (t={t}, U={U}), evolution time tau={tau}")
print("=" * 76)
print(f"The Hamiltonian is a sum of {n_terms} Pauli strings"
      f" ({n_rot} of them non-identity, i.e. needing a gate):")
for s, c in terms:
    print(f"    {complex(c).real:+.4f} * {s}")

print(f"\nFirst-order product formula, r steps of dt = tau/r")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} "
      f"{'error x r':>12} {'Pauli rotations':>16}")
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    dt = tau / r
    step = np.eye(2 ** 4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r:12.5f} {r*n_rot:16d}")

print("\nSecond-order (symmetric) product formula")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r^2':>14}")
for r in (1, 2, 4, 8, 16, 32):
    dt = tau / r
    step = np.eye(2 ** 4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    for s, c in reversed(terms):
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r*r:14.5f}")

print("\nExtrapolated gate cost of a phase-estimation run")
print("-" * 76)
dt = tau / 64
step = np.eye(2 ** 4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
C = np.linalg.norm(np.linalg.matrix_power(step, 64) - U_exact, ord=2) * 64
print(f"  first-order error ~ C / r  with  C = {C:.4f}")
for target in (1e-2, 1e-3, 1e-6):
    r_need = C / target
    print(f"  error <= {target:.0e}: r = {r_need:12.1f} steps"
          f"  ->  {r_need*n_rot:12.3e} Pauli rotations for tau = 1")
print("\n  Phase estimation to precision eps needs the evolution repeated ~1/eps")
print("  times.  Counting each repetition as an independent tau = 1 block held to")
print("  error eps is the OPTIMISTIC accounting: a single coherent evolution to")
print("  time tau = 1/eps has C ~ tau^2 and needs ~1/eps^3 steps (Exercise 4).")
print(f"  eps = 1e-3  ->  >~{C/1e-3*n_rot/1e-3:.3e} Pauli rotations in total,")
print("  for a model whose exact answer fits in a 16 x 16 matrix.")
```

```text
Trotter error, two-site Hubbard (t=1.0, U=4.0), evolution time tau=1.0
============================================================================
The Hamiltonian is a sum of 11 Pauli strings (10 of them non-identity, i.e. needing a gate):
    +2.0000 * IIII
    -1.0000 * IIIZ
    -1.0000 * IIZI
    +1.0000 * IIZZ
    -0.5000 * IXZX
    -0.5000 * IYZY
    -1.0000 * IZII
    -0.5000 * XZXI
    -0.5000 * YZYI
    -1.0000 * ZIII
    +1.0000 * ZZII

First-order product formula, r steps of dt = tau/r
 steps r        dt   spectral error    error x r  Pauli rotations
       1   1.00000     1.866470e+00      1.86647               10
       2   0.50000     1.416147e+00      2.83229               20
       4   0.25000     8.068454e-01      3.22738               40
       8   0.12500     4.163665e-01      3.33093               80
      16   0.06250     2.098203e-01      3.35713              160
      32   0.03125     1.051154e-01      3.36369              320
      64   0.01562     5.258338e-02      3.36534              640
     128   0.00781     2.629490e-02      3.36575             1280

Second-order (symmetric) product formula
 steps r        dt   spectral error    error x r^2
       1   1.00000     1.590328e+00        1.59033
       2   0.50000     4.799819e-01        1.91993
       4   0.25000     1.320533e-01        2.11285
       8   0.12500     3.368977e-02        2.15615
      16   0.06250     8.462631e-03        2.16643
      32   0.03125     2.118133e-03        2.16897

Extrapolated gate cost of a phase-estimation run
----------------------------------------------------------------------------
  first-order error ~ C / r  with  C = 3.3653
  error <= 1e-02: r =        336.5 steps  ->     3.365e+03 Pauli rotations for tau = 1
  error <= 1e-03: r =       3365.3 steps  ->     3.365e+04 Pauli rotations for tau = 1
  error <= 1e-06: r =    3365336.1 steps  ->     3.365e+07 Pauli rotations for tau = 1

  Phase estimation to precision eps needs the evolution repeated ~1/eps
  times.  Counting each repetition as an independent tau = 1 block held to
  error eps is the OPTIMISTIC accounting: a single coherent evolution to
  time tau = 1/eps has C ~ tau^2 and needs ~1/eps^3 steps (Exercise 4).
  eps = 1e-3  ->  >~3.365e+07 Pauli rotations in total,
  for a model whose exact answer fits in a 16 x 16 matrix.
```

**着目点。** スケーリングの主張は仮定ではなく数値的に確認されました。1次公式では誤差 × $r$ が3.366に収束し、対称公式では誤差 × $r^2$ が2.169に収束します。この2つの定数が、このハミルトニアンについての $O(1/r)$ と $O(1/r^2)$ という言明の内容そのものを具体化したものです。

実務的なメッセージは最後のブロックにあります。この4量子ビット模型を時間1だけ精度 $10^{-6}$ でシミュレートするには、1次公式でおよそ $3 \times 10^7$ 回のPauli回転が必要です。これに位相推定が要求する $1/\varepsilon$ 回の繰り返しを掛けると、$16 \times 16$ 行列に対して `numpy.linalg.eigvalsh` を呼べばマイクロ秒で厳密解が得られる系に対して、天文学的な回数になります。高次公式、より良い項の順序付け、qubitizationやpost-Trotter法はこれらの数を大幅に — 桁で — 減らしますが、小さくするわけではありません。

$\varepsilon = 10^{-3}$ に対して出力された $3.4 \times 10^7$ 回という数は*下限*です。その理由を明示しておく価値があります。この数は位相推定を、$\tau = 1$ のブロックを誤差 $\varepsilon$ でTrotter化したものの $1/\varepsilon$ 回の独立な繰り返しとして数えています。実際の位相推定回路は時間 $\tau \sim 1/\varepsilon$ までコヒーレントに発展し、演習4が示すように1次公式の定数は $\tau^2$ で増大するので、*全体*のユニタリ誤差を $\varepsilon$ に保つには $\sim C\tau^2/\varepsilon = 3.4/\varepsilon^3$ ステップ — $\varepsilon = 10^{-3}$ ではさらに3桁 — かかります。第5章のCode Example 5はこの $3.4 \times 10^7$ を引用しており、2つの章は同じ楽観的な数え方で一致しています。どちらも提案書に書ける資源見積りではありません。

恒等文字列は係数 $+2$ をもちますがゲートを必要としないことにも注意してください。大域位相は観測不可能です。11項のうち10項が回転を要します。資源見積りが成果物であるとき、このような細かい帳簿付けが効いてきます。

### QPEとVQE

基底状態推定では2つのアルゴリズムが支配的で、両者はハードウェアのスペクトラムの正反対の端に位置します。

**量子位相推定**は基底状態と重なり $\lvert \langle \psi_{\text{trial}} \vert \psi_0 \rangle \rvert^2 = p_0$ をもつ試行状態を用意し、$e^{-i\hat{H}t}$ の制御べきを作用させ、逆Fourier変換から位相を読み出します。回路の深さ $O(1/\varepsilon)$ と成功確率 $p_0$ で、固有値を精度 $\varepsilon$ で返します。その精度は測定を増やしても劣化しません。重なりの問題を除けば、**決定論的**な固有値抽出です。

**変分量子固有値ソルバー**（第3章）はパラメータ化された状態を用意し、$\langle \hat{H} \rangle$ を測定し、古典最適化器にそれを最小化させます。深さは要求精度ではなくansatzによって決まり、だからこそ現在のハードウェアに載ります。代価は測定回数と、ansatzが真の基底状態を表現できるかどうかで支払われます。

性質 | QPE | VQE
---|---|---
回路の深さ | $O(1/\varepsilon)$、非常に深い | 浅い、ansatz依存
精度の限界 | 系統的、制御可能 | ansatzのバイアス + ショットノイズ
補助量子ビット | 必要（$\log(1/\varepsilon)$） | 不要
測定回数 | 少数回の繰り返し | $O(M^4/\varepsilon^2)$ 回の回路実行
最適化 | 不要 | 非凸、barren plateau
誤り訂正 | 必須 | 不要
保証 | 確率 $p_0$ で固有値を精度 $\varepsilon$ | 変分的な上限のみ
時代 | FTQC | NISQ

この表の最後の行が本章で最も重要な1行です。2つのアルゴリズムは同じ機械をめぐる競争相手ではなく、2つの異なる機械に対する答えです。VQEが与えるのは**変分的な上限**であり、ansatzが基底状態を表現できないなら、答えは既知の向きに、しかし未知の量だけ誤っています。QPEは固有値を与えますが、誤り耐性機械しか供給できない深さを必要とします。第5章では、現在の誤り率がその深さからどれだけ離れているかを定量化します。

* * *

## 4.5 材料科学における標的問題

### 良い標的とは何か

材料研究にとっての量子コンピューティングの標的は、4つの条件を同時に満たすべきです。

  1. **強相関である** — さもなければDFTか結合クラスタがすでにより速く答えている。
  2. **活性空間が十分小さい** — 量子ビット数とPauli項数が機械に収まらなければならない。
  3. **古典的に難しい** — full CIにとって難しいだけでなく、DMRGにも量子モンテカルロにも難しい。
  4. **科学的に決定的である** — 答えが意思決定を変えなければならない。単にデータ点を1つ増やすだけでは足りない。

今日この4つすべてを満たす問題はごくわずかです。正直に順位づけると：

標的 | 相関 | 活性空間の大きさ | 古典的な状況 | 近未来の見通し
---|---|---|---|---
横磁場Ising鎖 | 中程度 | 極小 | 厳密可解 | ベンチマーク専用
Heisenberg鎖・ラダー | 強 | 小 | DMRGが1次元を事実上厳密に解く | ベンチマーク、アルゴリズム開発
1次元Hubbard | 強 | 小 | Bethe仮説 + DMRG | ベンチマーク
2次元Hubbard、ドープ | 強 | 中程度 | 未解決：符号問題、競合する秩序 | 真の候補、長期
遷移金属2量体 | 強 | 中程度 | 労力をかければDMRGで可能 | 中期的にありうる
FeMoco（ニトロゲナーゼ補因子） | 非常に強 | 約76軌道 | 厳密法の範囲外 | 教科書的なFTQC標的、NISQではない
電池正極の酸化還元 | 中〜強 | 中程度 | DFT+U、ハイブリッド汎関数で可能 | 優位性は不明
光触媒の励起状態 | 強 | 中程度 | 多参照法が存在 | 興味深いが未証明
高温超伝導の機構 | 非常に強 | 大 | 未解決 | 活性空間計算では扱えない

この表について2点。第1に、有名な標的（FeMoco、高温超伝導）はまさに誤り耐性を必要とするものであり、現在の機械に収まる標的はまさに古典手法がすでに解いているものです。この隔たりがこの分野の中心的事実であり、第5章が存在する理由です。第2に、ドープされた2次元Hubbard模型が「真の候補」であるというのは**問題**についての言明であり、時期についての言明ではありません。古典的に未解決で物理的に重要である、それが備えるべき正しい対象である理由です。

### モデルハミルトニアン

近未来の文献の大半は3つの模型でカバーされ、うち2つを4.6節で構成します。

**横磁場Ising鎖。**

$$ \hat{H} = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i $$

真の量子相転移をもつ最も単純な模型で、熱力学極限では $h/J = 1$ に転移点があります。Jordan-Wigner変換とBogoliubov回転によって自由フェルミオンに写るので厳密可解であり、だからこそ理想的なベンチマークになります。任意の数値結果や量子計算結果を閉じた形の答えと照合できるのです。

**Hubbard模型。**

$$ \hat{H} = -t \sum_{\langle ij \rangle, \sigma} \left(\hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma} + \hat{c}_{j\sigma}^\dagger \hat{c}_{i\sigma}\right) + U \sum_i \hat{n}_{i\uparrow}\hat{n}_{i\downarrow} - \mu \sum_{i\sigma} \hat{n}_{i\sigma} $$

相関電子の最小模型です。運動エネルギー $t$ と在サイト反発 $U$ が競合します。半充填かつ大きな $U/t$ ではMott絶縁体であり、2次元で半充填から離れると相図は真に未解決です。この模型で面白いことはすべて2つの項の争いです。

**Heisenberg模型。**

$$ \hat{H} = J\sum_{\langle ij \rangle} \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j $$

Hubbard模型の大 $U$ 有効理論で $J = 4t^2/U$ です（4.2節の定数 $-J/4$ を含めたボンドあたりの形）。スピンのみなのでJordan-Wignerのオーバーヘッドがありません。

### FeMocoの例を正直に

ニトロゲナーゼ補因子FeMocoはこの分野の標準的な例なので、公表された資源見積りが実際に何を言っているかを述べておく価値があります。この分子は工業的なHaber-Bosch法が数百度と数百気圧を要する条件で、常温常圧の大気窒素を固定します。機構の理解には真の価値があり、その約76軌道の活性空間（およそ152量子ビット）は古典的な厳密対角化の遥か彼方です。

この系に対する誤り耐性の資源見積りは、qubitization、2電子テンソルのより良い因子分解、改良されたmagic state蒸留が素朴なTrotter分解を置き換えるにつれ、この10年で数桁下がってきました。その進歩の**方向**は励みになりますが、**水準**は標準的な仮定のもとで依然として数百万の物理量子ビットと数日の実行時間です。非Cliffordゲート数が $10^{10}$ から $10^{11}$ のオーダーであり — 論理Toffoli 1個あたり約10マイクロ秒とすればそれだけで数日の実時間になります — 誤り訂正なしに達成できる誤り率ではそれを支えられないからです（第5章Code Example 5のパートC）。これらの指数は桁として受け取ってください。公表値はすでに数桁動いており、今後も動きます。したがってFeMocoは誤り耐性量子コンピュータを建設すべき優れた論拠であり、近未来の化学的成果を期待すべき貧しい論拠です。

* * *

## 4.6 実装：2つのモデルハミルトニアンを端から端まで

ここで横磁場Ising鎖と2サイトHubbard模型の量子ビットハミルトニアンを構成し、厳密対角化し、独立な解析的結果と照合し、それから同じ対象にVQEを走らせます。3つすべてを行う意味は、比較だけが情報を与えるという点にあります。厳密対角化は答えを教え、解析式は厳密対角化が正しいことを教え、VQEは量子アルゴリズムなら何を報告したかを教えます。

### 横磁場Ising鎖

IsingハミルトニアンはすでにPauli文字列の和なので、フェルミオン写像は不要です。これが考えうる最も安価な量子シミュレーション標的であり、ほとんどすべてのハードウェア実証に登場する理由です。

Code Example 4: Ising鎖を対角化して照合する

```python
"""Chapter 4, Example 4: transverse-field Ising chain -> qubit Hamiltonian.
Continues from Example 2 (same session)."""


def tfim_hamiltonian(N: int, J: float, h: float, periodic: bool = False) -> dict:
    """H = -J sum_i Z_i Z_{i+1} - h sum_i X_i, as a Pauli-string dictionary."""
    terms = {}
    bonds = list(range(N - 1)) + ([N - 1] if periodic and N > 2 else [])
    for i in bonds:
        j = (i + 1) % N
        s = ''.join('Z' if k in (i, j) else 'I' for k in range(N))
        terms[s] = terms.get(s, 0.0) - J
    for i in range(N):
        s = 'I' * i + 'X' + 'I' * (N - i - 1)
        terms[s] = terms.get(s, 0.0) - h
    return terms


N, J = 4, 1.0
print(f"Transverse-field Ising chain, N = {N}, open boundary, J = {J}")
print("=" * 74)
Hq = tfim_hamiltonian(N, J, 1.0)
print(f"qubit Hamiltonian at h = 1.0  ({len(Hq)} Pauli terms):")
print("  H =", op_str(Hq))

M = to_matrix(Hq)
print(f"\nHermitian: {np.allclose(M, M.conj().T)}")
evals, evecs = np.linalg.eigh(M)
print(f"lowest four eigenvalues: {np.round(evals[:4], 8)}")
print(f"ground-state energy  E0 = {evals[0]:.10f}")
print(f"first excited state  E1 = {evals[1]:.10f}")
print(f"spectral gap            = {evals[1] - evals[0]:.10f}")

print("\nField scan: order parameter, correlation and gap")
print("-" * 74)
print(f"{'h':>6} {'E0':>12} {'E0/N':>10} {'<X_0>':>10} "
      f"{'<Z_0 Z_1>':>11} {'gap':>10}")
for h in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
    evals, evecs = np.linalg.eigh(to_matrix(tfim_hamiltonian(N, J, h)))
    g = evecs[:, 0]
    mx = np.real(np.vdot(g, pauli_matrix('X' + 'I' * (N - 1)) @ g))
    zz = np.real(np.vdot(g, pauli_matrix('ZZ' + 'I' * (N - 2)) @ g))
    print(f"{h:6.2f} {evals[0]:12.6f} {evals[0]/N:10.6f} "
          f"{mx:10.6f} {zz:11.6f} {evals[1]-evals[0]:10.6f}")

print("\nSize dependence of the energy density at h = J = 1 (open chain)")
print("-" * 74)
for N_ in (2, 4, 6, 8, 10):
    e0 = np.linalg.eigvalsh(to_matrix(tfim_hamiltonian(N_, 1.0, 1.0)))[0]
    print(f"  N = {N_:3d}: E0 = {e0:11.6f}   E0/N = {e0/N_:10.6f}   "
          f"Hilbert dim = {2**N_:6d}")

print("\nIndependent check: periodic chain against the free-fermion solution")
print("-" * 74)
for N_ in (4, 6, 8, 10):
    e0 = np.linalg.eigvalsh(to_matrix(
        tfim_hamiltonian(N_, 1.0, 1.0, periodic=True)))[0]
    ks = (2 * np.arange(N_) + 1) * np.pi / N_          # antiperiodic sector
    exact = -np.sum(np.sqrt(1 + 1.0 ** 2 - 2 * 1.0 * np.cos(ks)))
    print(f"  N = {N_:3d}: E0 = {e0:11.6f}   free fermions = {exact:11.6f}"
          f"   difference = {abs(e0-exact):.2e}")
```

```text
Transverse-field Ising chain, N = 4, open boundary, J = 1.0
==========================================================================
qubit Hamiltonian at h = 1.0  (7 Pauli terms):
  H = -1.0000*IIIX  -1.0000*IIXI  -1.0000*IIZZ  -1.0000*IXII  -1.0000*IZZI  -1.0000*XIII  -1.0000*ZZII

Hermitian: True
lowest four eigenvalues: [-4.75877048 -4.06417777 -2.75877048 -2.06417777]
ground-state energy  E0 = -4.7587704831
first excited state  E1 = -4.0641777725
spectral gap            = 0.6945927107

Field scan: order parameter, correlation and gap
--------------------------------------------------------------------------
     h           E0       E0/N      <X_0>   <Z_0 Z_1>        gap
  0.00    -3.000000  -0.750000   0.000000    1.000000   0.000000
  0.25    -3.097889  -0.774472   0.261617    0.957928   0.007325
  0.50    -3.427034  -0.856759   0.545775    0.818662   0.094788
  0.75    -4.005816  -1.001454   0.755422    0.636728   0.335511
  1.00    -4.758770  -1.189693   0.862086    0.494818   0.694593
  1.25    -5.605986  -1.401496   0.913793    0.399018   1.115221
  1.50    -6.503892  -1.625973   0.941298    0.333158   1.566416
  2.00    -8.376799  -2.094200   0.967737    0.250022   2.510954
  3.00   -12.250561  -3.062640   0.985913    0.166677   4.461930
  4.00   -16.187740  -4.046935   0.992125    0.125003   6.439777

Size dependence of the energy density at h = J = 1 (open chain)
--------------------------------------------------------------------------
  N =   2: E0 =   -2.236068   E0/N =  -1.118034   Hilbert dim =      4
  N =   4: E0 =   -4.758770   E0/N =  -1.189693   Hilbert dim =     16
  N =   6: E0 =   -7.296230   E0/N =  -1.216038   Hilbert dim =     64
  N =   8: E0 =   -9.837951   E0/N =  -1.229744   Hilbert dim =    256
  N =  10: E0 =  -12.381490   E0/N =  -1.238149   Hilbert dim =   1024

Independent check: periodic chain against the free-fermion solution
--------------------------------------------------------------------------
  N =   4: E0 =   -5.226252   free fermions =   -5.226252   difference = 1.78e-15
  N =   6: E0 =   -7.727407   free fermions =   -7.727407   difference = 8.88e-16
  N =   8: E0 =  -10.251662   free fermions =  -10.251662   difference = 2.13e-14
  N =  10: E0 =  -12.784906   free fermions =  -12.784906   difference = 1.07e-14
```

**着目点。** 磁場スキャンは、4サイトの窓から覗いた量子相転移です。$h = 0$ では基底状態は古典的な強磁性体で、$\langle Z_0 Z_1 \rangle = 1$、$\langle X_0 \rangle = 0$、そして2つの強磁性配置が厳密に縮退しているためギャップが消えています。$h$ が増えると横磁場がそれらを混合し、ギャップが開き、$\langle X_0 \rangle$ が1に向かって上昇し、$ZZ$ 相関が下がります。$h = 4$ では状態はほぼ積状態 $\lvert +{+}{+}{+} \rangle$ で、$\langle X_0 \rangle = 0.992$、$\langle Z_0 Z_1 \rangle = 0.125$ でこれはちょうど $J/2h$ です。すなわち*1次*摂動論の結果です。$-J\sum Z_iZ_{i+1}$ を $\lvert{+}{+}{+}{+}\rangle$ への摂動として扱うと、ボンド項は両スピンを反転した状態（励起エネルギー $4h$）へ振幅 $J/4h$ を与え、$\langle Z_0Z_1\rangle \simeq J/2h$ となります。$h = 2, 3, 4$ で出力された0.250・0.167・0.125は $J/2h$ と5桁一致します。無限鎖の転移は $h/J = 1$ にありますが、4サイトではクロスオーバーににじみます。有限サイズスケーリングがそうならざるをえないと言うとおりです。

エネルギー密度の収束は遅く、$E_0/N$ は $N=2$ の $-1.118$ から $N=10$ の $-1.238$ まで動き、熱力学極限値 $-4/\pi = -1.2732$ に上から近づきます。格子模型の小さな量子シミュレーションが与えるのは小さな格子であって熱力学極限ではない、という有用な注意でもあります。

最後のブロックが最も重要です。周期鎖のエネルギーは閉じた形の自由フェルミオン表式 $E_0 = -\sum_k \sqrt{1 + h^2 - 2h\cos k}$ と $10^{-14}$ で一致します。この式はまったく別の経路（Jordan-Wignerでフェルミオン化し、運動量空間でBogoliubov対角化）で導かれたものなので、この一致はCode Example 2のPauli文字列→行列のパイプライン全体を検証しています。量子アルゴリズムの答えを信用する前に、参照値が正しいことを確かめましょう。

### 2サイトHubbard模型

次はフェルミオンの場合で、Jordan-Wigner変換が本当の仕事をします。4つのスピン軌道を（サイト0 ↑、サイト0 ↓、サイト1 ↑、サイト1 ↓）の順に並べ、$\mu = U/2$ とします。これは粒子-ホール対称点で、大域的な基底状態が半充填であることが保証されます。

Code Example 5: Jordan-WignerによるHubbard二量体

```python
"""Chapter 4, Example 5: two-site Hubbard model, built by Jordan-Wigner.
Continues from Example 2 (same session)."""

# Spin-orbital ordering (= qubit index):
#   0 = site 0 spin up    1 = site 0 spin down
#   2 = site 1 spin up    3 = site 1 spin down
N_MODES = 4
UP = {0: 0, 1: 2}
DN = {0: 1, 1: 3}


def hubbard_dimer(t: float, U: float, mu: float = None) -> dict:
    """H = -t sum_sigma (c^dag_{0 sigma} c_{1 sigma} + h.c.)
           + U sum_i n_{i up} n_{i dn} - mu sum_p n_p
    With mu = U/2 the model is particle-hole symmetric and the global
    ground state lies in the half-filled sector."""
    n = N_MODES
    if mu is None:
        mu = U / 2.0
    Hq = {}
    for spin in (UP, DN):
        Hq = op_add(Hq, op_scale(jw_hop(spin[0], spin[1], n), -t))
    for site in (0, 1):
        Hq = op_add(Hq, op_scale(op_mul(jw_number(UP[site], n),
                                        jw_number(DN[site], n)), U))
    for p in range(n):
        Hq = op_add(Hq, op_scale(jw_number(p, n), -mu))
    return Hq


def sector_indices(n_up: int, n_dn: int):
    """Basis indices with a given number of up and down electrons."""
    out = []
    for i in range(2 ** N_MODES):
        b = format(i, f'0{N_MODES}b')
        if (int(b[UP[0]]) + int(b[UP[1]]) == n_up
                and int(b[DN[0]]) + int(b[DN[1]]) == n_dn):
            out.append(i)
    return out


t = 1.0
print("Two-site Hubbard model via Jordan-Wigner (4 spin orbitals -> 4 qubits)")
print("=" * 76)

Hq = hubbard_dimer(t, 4.0)
print(f"t = 1, U = 4, mu = U/2:  {len(Hq)} Pauli terms")
print("  H =", op_str(Hq))
M = to_matrix(Hq)
print(f"\nHermitian: {np.allclose(M, M.conj().T)}")
evals = np.linalg.eigvalsh(M)
print(f"full 16-level spectrum:\n  {np.round(evals, 6)}")
print(f"global ground-state energy E0 = {evals[0]:.10f}")

print("\nHalf-filled sector (one up + one down electron), mu = 0")
print("-" * 76)
idx = sector_indices(1, 1)
print(f"  sector dimension: {len(idx)} of {2**N_MODES}")
print(f"{'U':>6} {'E0 (numeric)':>15} {'E0 (analytic)':>15} {'|diff|':>10} "
      f"{'<n_up n_dn>':>12} {'<S_0.S_1>':>11}")
D_tot = to_matrix(op_add(op_mul(jw_number(UP[0], 4), jw_number(DN[0], 4)),
                         op_mul(jw_number(UP[1], 4), jw_number(DN[1], 4))))
Sz0 = (to_matrix(jw_number(UP[0], 4)) - to_matrix(jw_number(DN[0], 4))) / 2
Sz1 = (to_matrix(jw_number(UP[1], 4)) - to_matrix(jw_number(DN[1], 4))) / 2
for U in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    sub = to_matrix(hubbard_dimer(t, U, mu=0.0))[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(sub)
    g = np.zeros(2 ** N_MODES, dtype=complex)
    g[idx] = v[:, 0]
    analytic = U / 2 - np.sqrt((U / 2) ** 2 + 4 * t ** 2)
    docc = np.real(np.vdot(g, D_tot @ g)) / 2
    spin_corr = 3 * np.real(np.vdot(g, (Sz0 @ Sz1) @ g))   # isotropic singlet
    print(f"{U:6.1f} {w[0]:15.8f} {analytic:15.8f} {abs(w[0]-analytic):10.2e} "
          f"{docc:12.6f} {spin_corr:11.6f}")

print("\nLarge-U limit: the Hubbard dimer becomes a Heisenberg dimer")
print("-" * 76)
print(f"{'U':>8} {'E0':>14} {'-4t^2/U':>12} {'ratio':>10}")
for U in (8.0, 16.0, 32.0, 64.0, 128.0):
    sub = to_matrix(hubbard_dimer(t, U, mu=0.0))[np.ix_(idx, idx)]
    e0 = np.linalg.eigvalsh(sub)[0]
    J_super = -4 * t ** 2 / U
    print(f"{U:8.1f} {e0:14.8f} {J_super:12.8f} {e0/J_super:10.6f}")

print("\nSpin gap and charge gap from sector-resolved diagonalization (mu = 0)")
print("-" * 76)


def particle_sector_energies(U, n_particles):
    Mn = to_matrix(hubbard_dimer(t, U, mu=0.0))
    sel = [i for i in range(2 ** N_MODES) if bin(i).count('1') == n_particles]
    return np.linalg.eigvalsh(Mn[np.ix_(sel, sel)])


print(f"{'U':>6} {'singlet (N=2)':>15} {'triplet (N=2)':>15} "
      f"{'spin gap':>10} {'charge gap':>11}")
for U in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
    e2 = particle_sector_energies(U, 2)
    e1 = particle_sector_energies(U, 1)
    e3 = particle_sector_energies(U, 3)
    charge_gap = e1[0] + e3[0] - 2 * e2[0]
    print(f"{U:6.1f} {e2[0]:15.8f} {e2[1]:15.8f} "
          f"{e2[1]-e2[0]:10.6f} {charge_gap:11.6f}")
print("\n  large-U asymptotics for the dimer:"
      " spin gap -> 4t^2/U,  charge gap -> U - 2t")
```

```text
Two-site Hubbard model via Jordan-Wigner (4 spin orbitals -> 4 qubits)
============================================================================
t = 1, U = 4, mu = U/2:  7 Pauli terms
  H = -2.0000*IIII  +1.0000*IIZZ  -0.5000*IXZX  -0.5000*IYZY  -0.5000*XZXI  -0.5000*YZYI  +1.0000*ZZII

Hermitian: True
full 16-level spectrum:
  [-4.828427 -4.       -4.       -4.       -3.       -3.       -3.
 -3.       -1.       -1.       -1.       -1.        0.        0.
  0.        0.828427]
global ground-state energy E0 = -4.8284271247

Half-filled sector (one up + one down electron), mu = 0
----------------------------------------------------------------------------
  sector dimension: 4 of 16
     U    E0 (numeric)   E0 (analytic)     |diff|  <n_up n_dn>   <S_0.S_1>
   0.0     -2.00000000     -2.00000000   0.00e+00     0.250000   -0.375000
   1.0     -1.56155281     -1.56155281   4.44e-16     0.189366   -0.465951
   2.0     -1.23606798     -1.23606798   4.44e-16     0.138197   -0.542705
   4.0     -0.82842712     -0.82842712   2.22e-16     0.073223   -0.640165
   8.0     -0.47213595     -0.47213595   4.44e-16     0.026393   -0.710410
  16.0     -0.24621125     -0.24621125   4.44e-16     0.007464   -0.738803
  32.0     -0.12451550     -0.12451550   1.12e-14     0.001931   -0.747104

Large-U limit: the Hubbard dimer becomes a Heisenberg dimer
----------------------------------------------------------------------------
       U             E0      -4t^2/U      ratio
     8.0    -0.47213595  -0.50000000   0.944272
    16.0    -0.24621125  -0.25000000   0.984845
    32.0    -0.12451550  -0.12500000   0.996124
    64.0    -0.06243908  -0.06250000   0.999025
   128.0    -0.03124237  -0.03125000   0.999756

Spin gap and charge gap from sector-resolved diagonalization (mu = 0)
----------------------------------------------------------------------------
     U   singlet (N=2)   triplet (N=2)   spin gap  charge gap
   0.0     -2.00000000     -0.00000000   2.000000    2.000000
   1.0     -1.56155281     -0.00000000   1.561553    2.123106
   2.0     -1.23606798     -0.00000000   1.236068    2.472136
   4.0     -0.82842712     -0.00000000   0.828427    3.656854
   8.0     -0.47213595     -0.00000000   0.472136    6.944272
  16.0     -0.24621125     -0.00000000   0.246211   14.492423

  large-U asymptotics for the dimer: spin gap -> 4t^2/U,  charge gap -> U - 2t
```

**着目点。** この出力は、16次元の行列にしては驚くほど多くの物性物理を含んでいます。

**量子ビットハミルトニアンは驚くほどコンパクトで、その構造は読み取れます。** `IIZZ` と `ZZII` は在サイト $U$ 項です。展開すると $\hat{n}_{i\uparrow}\hat{n}_{i\downarrow} = (I - Z_u)(I - Z_d)/4 = (I - Z_u - Z_d + Z_uZ_d)/4$ で、各サイトは定数・単一 $Z$ 項2つ・$ZZ$ 項を出します。粒子正孔対称点 $\mu = U/2$ では単一 $Z$ 項が化学ポテンシャル項とちょうど打ち消し合うため、$ZZ$ だけが残るのです。`XZXI`、`YZYI`、`IXZX`、`IYZY` は2つのホッピング項で、私たちの順序付けでは対になるスピン軌道が2つ離れているため、それぞれJordan-Wigner文字列由来の $Z$ を1つ担っています。係数 $-2$ の `IIII` は2つの定数の残りです。相互作用項から $+U \cdot 2/4 = +2$、化学ポテンシャル項から $-\mu \cdot 4/2 = -4$ が来ています。軌道の順序を変えれば — サイト別ではなくスピン別にまとめれば — 文字列は変わりますがスペクトルは変わりません。回路の深さは**変わる**ので、軌道の順序付けは実際の最適化対象です（演習3）。

**解析的な照合は厳密です。** Hubbard二量体の半充填基底状態は閉じた形で知られており、

$$ E_0 = \frac{U}{2} - \sqrt{\left(\frac{U}{2}\right)^2 + 4t^2} $$

すべての数値が $10^{-14}$ 以下で一致します。これは同語反復ではなくJordan-Wigner実装の本当の試験です。パリティ文字列の符号を1つ誤れば異なるスペクトルが出ます。

**二重占有率がMott局在を示します。** $U = 0$ では電子は独立で $\langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle = 0.25$、まさに無相関の値 $0.5 \times 0.5$ です。$U$ が増えるとこれは崩壊します。$U = 4$ で0.0732、$U = 32$ で0.0019 — 電子が互いのサイトを訪れなくなるのです。これがMott絶縁体の2サイト版の戯画であり、二重占有率はそれを検出するために量子コンピュータ上で測定するまさにその観測量です。

**超交換相互作用が定量的に現れます。** $\langle \hat{\mathbf{S}}_0 \cdot \hat{\mathbf{S}}_1 \rangle$ は $U = 0$ の $-0.375$ から $U = 32$ の $-0.747$ へ動き、一重項の値 $-3/4$ に収束します。そして比 $E_0 / (-4t^2/U)$ は $U/t = 128$ で0.99976まで上がります。この分母がどのHeisenbergエネルギーかに注意してください。有効ボンドハミルトニアンは $J(\hat{\mathbf{S}}_0\cdot\hat{\mathbf{S}}_1 - 1/4)$（$J = 4t^2/U$）であり、その一重項エネルギーが $-J = -4t^2/U$ です。定数を落とせば $-3J/4 = -3t^2/U$ となり、比は $4/3$ からずれたままになります。有名な $J = 4t^2/U$ の式は手を振った議論ではなく、収束していく様子を観察できる展開の主要項であり、同じ機構が銅酸化物、そして事実上すべての磁性絶縁体の交換相互作用を決めています。

**2つのギャップが分離します。** スピンギャップ（一重項-三重項）は $4t^2/U$ として**閉じ**、電荷ギャップは $U - 2t$ として**開き**ます。大 $U$ のHubbard系は低エネルギーのスピン励起をもつ電荷絶縁体であり、まさにこれが有効スピン模型が正しい記述である理由、そしてMott絶縁体の低エネルギー分光を磁気励起が支配する理由です。

### 同じハミルトニアンにVQEを

最後に量子アルゴリズムです。以下のコードは第1-2章のミニシミュレータ（APIもbig-endianの規約も不変）を再掲して本章が単体で動くようにし、hardware-efficient ansatzを構成し、厳密なparameter-shift勾配を計算し、素朴な勾配降下を走らせます。

Code Example 6: VQEと厳密対角化の比較

```python
"""Chapter 4, Example 6: VQE on both qubit Hamiltonians, against exact diagonalization.
Continues from Examples 2, 4 and 5 (same session).
The first part re-lists the Chapter 1-2 mini simulator so the chapter is
self-contained; the API and the big-endian convention are unchanged."""
import numpy as np

# =====================================================================
# Mini state-vector simulator (Chapters 1-2 API, big-endian:
# qubit 0 = leftmost bit = most significant bit, index = sum_i q_i 2^(n-1-i))
# =====================================================================
# ---- 1量子ビットゲート --------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


# ---- 状態 ---------------------------------------------------------------
def ket(bits: str) -> np.ndarray:
    """'01' -> 4次元の基底状態 |01>（ビッグエンディアン）"""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


def apply_gate(state, U, targets, n):
    """n量子ビット状態の targets に 2^k x 2^k ユニタリ U を作用させる"""
    k = len(targets)
    psi = state.reshape([2] * n)          # 1. n添字テンソルとして見る
    psi = np.moveaxis(psi, targets, range(k))   # 2. 標的軸を先頭へ
    rest = psi.shape[k:]
    psi = psi.reshape(2 ** k, -1)         # 3. 平坦化して行列積
    psi = U @ psi
    psi = psi.reshape(list((2,) * k) + list(rest))
    psi = np.moveaxis(psi, range(k), targets)   # 4. 軸を元に戻す
    return psi.reshape(-1)


CNOT4 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)


def cnot(state, control, target, n):
    """任意の量子ビット対・任意の向きのCNOT"""
    return apply_gate(state, CNOT4, [control, target], n)


def probs(state):
    """Born則による全 2^n 通りの確率"""
    return np.abs(state) ** 2


def sample(state, shots, seed=None):
    """測定のシミュレーション: {ビット列: 回数}"""
    n = int(np.log2(state.size))
    rng = np.random.default_rng(seed)
    idx = rng.choice(state.size, size=shots, p=probs(state))
    out = {}
    for i in idx:
        b = format(i, f'0{n}b')
        out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items()))


PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def expval(state, pauli, coeff_map=None):
    """'ZZ' や 'XI' のようなPauli文字列（1量子ビット1文字）の期待値。

    coeff_map を与えると結果に coeff_map[pauli] を掛けるので、ハミルトニアン
    全体が1行で書ける:  sum(expval(psi, p, terms) for p in terms)
    """
    n = len(pauli)
    phi = state.copy()
    for q, ch in enumerate(pauli):
        if ch != 'I':
            phi = apply_gate(phi, PAULI[ch], [q], n)
    val = np.vdot(state, phi).real
    if coeff_map is not None:
        val *= coeff_map.get(pauli, 1.0)
    return val


# =====================================================================
# Hardware-efficient ansatz, energy, parameter-shift gradient, VQE loop
# =====================================================================

def ansatz(theta, n, layers):
    """Ry layer, then `layers` blocks of (CNOT ladder + Ry layer)."""
    psi, k = ket('0' * n), 0
    for q in range(n):
        psi = apply_gate(psi, ry(theta[k]), [q], n)
        k += 1
    for _ in range(layers):
        for q in range(n - 1):
            psi = cnot(psi, q, q + 1, n)
        for q in range(n):
            psi = apply_gate(psi, ry(theta[k]), [q], n)
            k += 1
    return psi


def real_terms(Hq):
    """A Hermitian Pauli-string operator has real coefficients."""
    return {s: complex(c).real for s, c in Hq.items()}


def energy(theta, Hq, n, layers):
    terms = real_terms(Hq)
    psi = ansatz(theta, n, layers)
    return sum(expval(psi, s, terms) for s in terms)


def gradient(theta, Hq, n, layers):
    """Exact parameter-shift rule: dE/dtheta_i = [E(+pi/2) - E(-pi/2)] / 2."""
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy(tp, Hq, n, layers) - energy(tm, Hq, n, layers))
    return g


def vqe(Hq, n, layers=3, steps=1200, lr=0.3, seed=1, log_every=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.3, size=n * (layers + 1))
    for it in range(steps):
        if log_every and it % log_every == 0:
            print(f"      iter {it:5d}   E = {energy(theta, Hq, n, layers):+.8f}")
        theta -= lr * gradient(theta, Hq, n, layers)
    return energy(theta, Hq, n, layers), theta


# =====================================================================
print("VQE vs exact diagonalization")
print("=" * 72)

print("\n[1] Transverse-field Ising chain, N = 4, J = h = 1")
H_ising = tfim_hamiltonian(4, 1.0, 1.0)
exact_i = np.linalg.eigvalsh(to_matrix(H_ising))[0]
print(f"    exact   E0 = {exact_i:.8f}")
e_i, th_i = vqe(H_ising, 4, layers=3, steps=1200, lr=0.3, seed=1, log_every=300)
print(f"    VQE     E  = {e_i:.8f}    error = {e_i - exact_i:.3e}")

print("\n[2] Two-site Hubbard model, t = 1, U = 4, mu = U/2")
H_hub = hubbard_dimer(1.0, 4.0)
exact_h = np.linalg.eigvalsh(to_matrix(H_hub))[0]
print(f"    exact   E0 = {exact_h:.8f}")
e_h, th_h = vqe(H_hub, 4, layers=3, steps=600, lr=0.3, seed=3, log_every=150)
print(f"    VQE     E  = {e_h:.8f}    error = {e_h - exact_h:.3e}")

print("\n[3] Does the VQE state reproduce the physics, not just the energy?")
psi = ansatz(th_h, 4, 3)
docc = op_mul(jw_number(0, 4), jw_number(1, 4))
g = np.linalg.eigh(to_matrix(H_hub))[1][:, 0]
docc_r = real_terms(docc)
print(f"    <n_0up n_0dn>   VQE = "
      f"{sum(expval(psi, s, docc_r) for s in docc_r):.6f}")
print(f"    <n_0up n_0dn> exact = "
      f"{np.real(np.vdot(g, to_matrix(docc) @ g)):.6f}")
n_op_r = real_terms(op_add(*[jw_number(p, 4) for p in range(4)]))
print(f"    particle number VQE = "
      f"{sum(expval(psi, s, n_op_r) for s in n_op_r):.6f}"
      f"   (2 at half filling)")
print(f"    state overlap |<VQE|exact>|^2 = {abs(np.vdot(g, psi))**2:.6f}")

print("\n[4] Ansatz depth: how many layers are enough?")
print(f"    {'layers':>7} {'params':>7} {'Ising error':>14} {'Hubbard error':>15}")
for L in (1, 2, 3, 4):
    ei, _ = vqe(H_ising, 4, layers=L, steps=500, lr=0.3, seed=1)
    eh, _ = vqe(H_hub, 4, layers=L, steps=500, lr=0.3, seed=3)
    print(f"    {L:7d} {4*(L+1):7d} {ei-exact_i:14.3e} {eh-exact_h:15.3e}")

print("\n[5] Optimizer restarts: the landscape is not convex")
print(f"    {'seed':>5} {'Ising E':>14} {'error':>12}")
for s in range(6):
    ei, _ = vqe(H_ising, 4, layers=3, steps=500, lr=0.3, seed=s)
    print(f"    {s:5d} {ei:14.8f} {ei-exact_i:12.3e}")

print("\n[6] Measurement cost: exact expectation vs finite sampling")
print("    (20 independent shot budgets per row; <P> = 2 Pr[+1] - 1)")
print(f"    {'shots/term':>11} {'mean E':>12} {'std dev':>10} {'1/sqrt(N)':>11}")
psi_i = ansatz(th_i, 4, 3)
true_p = {s: expval(psi_i, s) for s in H_ising}
rng = np.random.default_rng(0)
for shots in (100, 1_000, 10_000, 100_000):
    ests = []
    for _ in range(20):
        est = 0.0
        for s, c in H_ising.items():
            hits = rng.binomial(shots, (1 + true_p[s]) / 2)
            est += complex(c).real * (2 * hits / shots - 1)
        ests.append(est)
    ests = np.array(ests)
    print(f"    {shots:11,d} {ests.mean():12.6f} {ests.std():10.6f} "
          f"{np.sqrt(len(H_ising)/shots):11.6f}")
```

```text
VQE vs exact diagonalization
========================================================================

[1] Transverse-field Ising chain, N = 4, J = h = 1
    exact   E0 = -4.75877048
      iter     0   E = -2.11673906
      iter   300   E = -4.74508701
      iter   600   E = -4.74696210
      iter   900   E = -4.75090684
    VQE     E  = -4.75479183    error = 3.979e-03

[2] Two-site Hubbard model, t = 1, U = 4, mu = U/2
    exact   E0 = -4.82842712
      iter     0   E = -1.41323792
      iter   150   E = -4.82841059
      iter   300   E = -4.82842704
      iter   450   E = -4.82842712
    VQE     E  = -4.82842712    error = 2.150e-12

[3] Does the VQE state reproduce the physics, not just the energy?
    <n_0up n_0dn>   VQE = 0.073223
    <n_0up n_0dn> exact = 0.073223
    particle number VQE = 2.000000   (2 at half filling)
    state overlap |<VQE|exact>|^2 = 1.000000

[4] Ansatz depth: how many layers are enough?
     layers  params    Ising error   Hubbard error
          1       8      2.466e-02       8.284e-01
          2      12      8.098e-03      -1.776e-15
          3      16      1.260e-02       7.193e-11
          4      20      1.139e-02       9.334e-06

[5] Optimizer restarts: the landscape is not convex
     seed        Ising E        error
        0    -4.75647707    2.293e-03
        1    -4.74616690    1.260e-02
        2    -4.75826562    5.049e-04
        3    -4.70883018    4.994e-02
        4    -4.74486976    1.390e-02
        5    -4.75594391    2.827e-03

[6] Measurement cost: exact expectation vs finite sampling
    (20 independent shot budgets per row; <P> = 2 Pr[+1] - 1)
     shots/term       mean E    std dev   1/sqrt(N)
            100    -4.689000   0.174066    0.264575
          1,000    -4.736500   0.050175    0.083666
         10,000    -4.749890   0.023303    0.026458
        100,000    -4.755493   0.005735    0.008367
```

**着目点。** この1つの出力が、実際のVQE実験では混ざって現れる3つの誤差源を切り分けています。

**Hubbardの結果は本質的に厳密です。** VQEは厳密値 $-4.82842712$ に対して $-4.82842712$ に到達し、誤差は $2 \times 10^{-12}$ です。ブロック[3]がそれを信じるべき理由を示しています。二重占有率が6桁一致し、ansatzが粒子数を保存しないにもかかわらず粒子数がちょうど2に出て、厳密基底状態との重なりが1.000000です。変分原理は仕事をし、エネルギーだけでなく物理が再現されました。

**Isingの結果は厳密ではなく、その理由はansatzではなく最適化器です。** ブロック[4]は2層のansatzが $8 \times 10^{-3}$ に達し、3層と4層は固定ステップ数の中でそれ以上良くならないことを示しています。ブロック[5]は同じ3層回路の6回のランダム再開始が $5 \times 10^{-4}$ から $5 \times 10^{-2}$ の間に着地することを示しています。初期パラメータだけで2桁の広がりです。非凸な地形上の素朴な勾配降下は信頼できる最小化器ではありません。実務では複数回の再開始とより良い最適化器を使い、最良値を報告します。変分的な上限なので「最良」には意味があります。

**層を増やすと悪くなることがあります。** ブロック[4]のHubbardの列が示唆的です。2層で $-2 \times 10^{-15}$（計算機精度）、3層で $7 \times 10^{-11}$、4層で $9 \times 10^{-6}$。表現力の高いansatzは地形が難しくなります — ただしここでの機構は平凡なものであって、風変わりなものではありません。500ステップという固定予算では、20個のパラメータは12個ほど遠くまで進めておらず、増えた方向は素朴な勾配降下がはまり込む局所解とほぼ平坦な谷を増やします。これはbarren plateauでは**ありません**。barren plateauは勾配分散が $n$ に対して指数的に潰れる現象であり、$n = 4$ 量子ビットで20パラメータはその領域から遠く離れています（第3章3.6節がより広い回路で本来の効果を測っています）。教訓は、収束不足がansatz誤差のように見えてしまうこと、そしてそれを見分けられるのはステップ予算のスキャンか複数回のリスタートだけだということです。

**測定ノイズが実務上支配的な誤差です。** ブロック[6]は、実験を志す者が心配すべきものです。エネルギー推定値の標準偏差は当然 $1/\sqrt{N_{\text{shots}}}$ で下がりますが、定数が容赦ありません。Pauli項あたり10,000ショット — この7項ハミルトニアンでは70,000回の回路実行 — でもばらつきは $0.023$ 残ります。$10^5$ 項のハミルトニアンで化学精度（$1.6 \times 10^{-3}$ Hartree）に達するには、ショット数が拘束条件になります。第5章はこれを実時間に換算し、答えは年単位です。

* * *

## 演習

本章のコードを手元に置いて取り組んでください。各問のあとに解答があります。

#### 演習1: 行列式の数え上げ

(a) 12軌道12電子の活性空間を使う計算を考えます。full CIには何個の行列式が必要で、量子アルゴリズムなら何量子ビットを使いますか。(b) この大きさで軌道を2個追加すると行列式の数はおよそ何倍になりますか。(c) 量子ビット数が**全**基底に対する $\log_2 D$ より遥かに小さく見えるのに、活性空間に対する $\log_2 D$ よりは大きいのはなぜですか。

<details><summary>解答</summary>
<p>(a) \(M = 12\)、\(N_\alpha = N_\beta = 6\) なので \(D = \binom{12}{6}^2 = 924^2 = 853{,}776\) 個の行列式、量子ビットは \(2M = 24\) 個です。Code Example 1 がまさにこの数を印字します。</p>
<p>(b) 増大率の表から、\(12 \to 14\) 軌道で \(D\) は13.80倍になります。この係数は \(M\) とともにゆっくり増え、半充填では漸近的に16に近づきます（新しい軌道対ごとに各二項係数がほぼ4倍になる）。</p>
<p>(c) \(\log_2 853{,}776 = 19.7\) なので24量子ビットはわずかに多いだけです。量子ビットのレジスタは<em>すべて</em>の占有列を符号化しており、粒子数やスピンが誤ったものも含みます。\(2^{24} = 1.7 \times 10^7\) 状態に対し物理的な状態は \(8.5 \times 10^5\) 個です。この20倍が対称性を課さないことの代価です。対称性による削減（パリティ写像 + 2量子ビット削減、粒子数保存ansatz）はその一部を回収します。第3章で4量子ビットの \(\mathrm{H}_2\) 問題を2量子ビットに圧縮できたのがその例です。</p>
</details>

#### 演習2: パリティ文字列は省略可能ではない

Code Example 2 の `jw_annihilate` と `jw_create` を、$Z$ 文字列を省いた版（局所的な昇降演算子だけを返す）に書き換えてください。(a) どの反交換関係が破れますか。(b) 壊れた写像で $U = 4$ の半充填Hubbard二量体の基底状態エネルギーを再計算してください。(c) このことは量子化学のパイプラインの検証について何を教えますか。

<details><summary>解答</summary>
<p>(a) 同一モードの関係 \(\{c_p, c_p^\dagger\} = I\) は局所演算子がそれを満たすので依然成立します。破れるのは<em>すべての</em>異モードの関係です。文字列がないと異なる量子ビット上の演算子は反交換ではなく可換になるので、\(\{c_0, c_1^\dagger\}\) と \(\{c_0, c_1\}\) が消えずにゼロでないPauli文字列になります。Example 2 の検証ループは <code>FAIL</code> を報告し、最後の <code>all 2 x 4^2 relations verified</code> の行が <code>False</code> になります。</p>
<p>(b) 壊れた写像では、ホッピング項は同じスピンの2つのスピン軌道上で間に \(Z\) を挟まない \((XX + YY)/2\) になります。ハミルトニアンは依然エルミートで対角化も成功しますが、返るのはフェルミオンではなく<em>ハードコアボソン</em>系のスペクトルです。2サイト二量体では数値がたまたま近いままで、それがこのバグを危険にしています。破綻が静かなのです。</p>
<p>(c) 独立な照合なしにエネルギーを信用してはいけません。代数を検証し（反交換子）、閉じた形の解があればそれと照合し（\(U/2 - \sqrt{(U/2)^2 + 4t^2}\) の式）、保存量を検証する（粒子数、全スピン）ことです。Example 2、4、5 の3つの検査はこのために存在し、誤った数値を公表するコストに比べれば安価です。</p>
</details>

#### 演習3: 軌道の順序と回路の深さ

Code Example 5 ではモードを（サイト0 ↑、サイト0 ↓、サイト1 ↑、サイト1 ↓）の順に並べたので、ホッピング項が $Z$ を1つ担いました。(a) 順序を（サイト0 ↑、サイト1 ↑、サイト0 ↓、サイト1 ↓）にして量子ビットハミルトニアンを導き直してください。(b) 2つの順序について、ホッピング項と相互作用項の最大Pauli重みを比較してください。(c) $L$ サイトの1次元鎖では、どちらの順序が総重みを最小にしますか。

<details><summary>解答</summary>
<p>(a) スピン別の順序では、上向きスピンのホッピングはモード0と1 — 隣接 — を結ぶので \((XX + YY)/2\) となり重み2です。下向きスピンのホッピングはモード2と3を結び、これも隣接で重み2です。相互作用 \(n_{0\uparrow} n_{0\downarrow}\) はモード0と2を結び2つ離れていますが、\(n_p = (I - Z_p)/2\) は \(Z\) と \(I\) しか含まないので文字列は不要で、項は依然 \(ZZ\)、重み2です。したがってこの順序ではどこでも最大重み2になります。</p>
<p>(b) サイト別：ホッピング重み3、相互作用重み2。スピン別：ホッピング重み2、相互作用重み2。二量体ではスピン別のほうが良いです。</p>
<p>(c) \(L\) サイト鎖では、スピン別順序はすべての最近接ホッピングを隣接にし（重み2）、在サイト相互作用は \(L\) モードにまたがりますが数演算子は文字列を必要としないので依然重み2です。したがってスピン別順序が1次元Hubbard模型では最適で、すべての項が \(O(1)\) 重み、総重み \(O(L)\) になります。サイト別順序はすべてのホッピングが重み3でこれも総計 \(O(L)\) なので、1次元ではどちらも許容できます。2つの順序が分岐するのは2次元で、そこではどの順序もすべての隣接を隣接にできず、ホッピング重みは最良でも \(O(\sqrt{L})\) で増大します。2次元Hubbard模型が難しく面白い場合である理由の1つです。</p>
</details>

#### 演習4: 実際の計算のためのTrotter予算

Code Example 3 の当てはめ定数 $C = 3.3653$ を使って、(a) $\tau = 1$ でユニタリ誤差 $10^{-4}$ を得るには1次Trotterステップが何回必要ですか。(b) 定数が2.169の2次公式では何回ですか。(c) 目標が $\tau = 10$ なら各答えはどう変わり、それは長時間ダイナミクスについて何を意味しますか。

<details><summary>解答</summary>
<p>(a) 1次：\(r = C/\varepsilon = 3.3653/10^{-4} = 33{,}653\) ステップ、すなわち恒等以外の10項に対しておよそ \(3.4 \times 10^5\) 回のPauli回転です。</p>
<p>(b) 2次：\(r = \sqrt{C_2/\varepsilon} = \sqrt{2.169/10^{-4}} = 147\) ステップ。1ステップあたりの回転数は2倍（前向きと後向きの掃き）なので約2,950回 — 115倍安いです。実務で1次Trotterを誰も使わない理由です。</p>
<p>(c) 誤差定数は発展時間とともに増大し、1次ではおよそ \(C \propto \tau^2\)、2次では \(\tau^3\) です（\(O(t^2/r)\) と \(O(t^3/r^2)\) の形）。\(\tau = 10\) なら1次は \(r \sim 100 \times 33{,}653 = 3.4 \times 10^6\) ステップ、2次は \(r \sim \sqrt{1000} \times 147 = 4{,}650\) ステップです。長時間ダイナミクスはどちらでも高価であり、標準的な対処は \(\tau\) を短く保って繰り返すこと、まさに位相推定がやっていることです。そして位相推定の深さが誤り耐性量子化学の拘束条件である理由でもあります。</p>
</details>

#### 演習5: 4サイトでは相転移を示せない理由

Code Example 4 を使って、(a) $N = 4$ で $\langle X_0 \rangle$ が1/2に達するのは $h/J$ がいくらのときですか。(b) 4サイトの計算が真の相転移を決して示せないのはなぜで、それは少数量子ビットのハードウェア実証について何を意味しますか。

<details><summary>解答</summary>
<p>(a) 印字された磁場スキャンを \(h = 0.25\)（\(\langle X_0 \rangle = 0.2616\)）と \(h = 0.5\)（0.5458）の間で補間すると、交点は \(h/J \approx 0.46\) 付近です。無限鎖の転移点 \(h/J = 1\) よりかなり下で、ボンドが3本しかない開放鎖の有限サイズシフトとして予期されるとおりです。</p>
<p>(b) 相転移は自由エネルギーの非解析性であり、有限系の自由エネルギーはパラメータの解析関数です。分配関数は指数関数の有限和、すなわち整関数です。非解析性は \(N \to \infty\) の極限でしか現れません。有限系は \(N\) とともに鋭くなるクロスオーバーを示すだけで、転移点の抽出には複数の \(N\) にわたる有限サイズスケーリングが必要です。だからこそ少数量子ビットでの実証はハードウェアのベンチマークであって、模型についての発見ではないのです。</p>
</details>

#### 演習6: どの誤差が支配的か

Hubbard二量体に対するVQEが厳密値 $-4.828427$ に対して $E = -4.81$ を報告したとします。Code Example 6 を使って、3つの誤差源 — ansatzの表現能力、最適化器の収束、測定統計 — のどれが原因かを判定し、それぞれの診断法を述べてください。

<details><summary>解答</summary>
<p>差は \(1.8 \times 10^{-2}\) です。コストの小さい順に診断します。</p>
<p><strong>ansatz。</strong> 厳密基底状態をansatz多様体に射影したエネルギーを計算する、あるいは実務的には多数の初期点から非常に大きなステップ予算で最適化を走らせ、最良値が厳密値より上で平坦化するかを見ます。ブロック[4]がこれをやっています。2層でHubbardの誤差は \(10^{-15}\) なので、この模型ではansatzは制約になっていません。もしそうであれば、最適化やサンプリングをいくら増やしても助けになりません。</p>
<p><strong>最適化器。</strong> 異なるシードから再開始します（ブロック[5]）。同じansatzと厳密な期待値で結果がばらつけば、最適化器が問題だと証明されます。Hubbard二量体では600ステップで \(2 \times 10^{-12}\) に収束するので、\(1.8 \times 10^{-2}\) は学習率の選択がまずいか反復が少なすぎることを示します。</p>
<p><strong>測定。</strong> <em>同じ</em>最適化済みパラメータで何度も繰り返し、ばらつきを見ます（ブロック[6]）。統計誤差は真値のまわりで対称で \(1/\sqrt{N}\) で縮みます。ansatz誤差は常に正で、まったく縮みません。ブロック[6]から、\(1.8 \times 10^{-2}\) のばらつきはPauli項あたりおよそ \(10^4\) ショットに対応します。</p>
<p>決定的な特徴：測定誤差はラン毎に揺らぎ、最適化器の誤差はシードで変わり、ansatz誤差は再現的で片側です。この場合 \(-4.81\) は厳密値より上なので3つすべての検査で切り分ける必要がありますが、最速の判別法は別のシードで再実行することです。</p>
</details>

* * *

## まとめ

### 要点

**1. 厳密な問題は組み合わせ論的に増大し、壁は近い**

  * Full CIは $\binom{M}{N_\alpha}\binom{M}{N_\beta}$ 個の行列式を要し、$M = 20$ 付近では軌道対の追加ごとに約14倍になる。
  * 20軌道の活性空間は波動関数ベクトル1本で半テラバイト、24軌道で100テラバイト。
  * 量子コンピュータは $2M$ 量子ビットで済む — 比較すれば安い。量子ビット数は3つの資源制約のうち最も軽い。

**2. DFTは面白い材料のあるところで破綻する**

  * 破綻の様式は静的相関、すなわち同程度の重みをもつ複数の行列式 — 遷移金属酸化物、Mott絶縁体、結合解離、触媒のスピン状態順序。
  * 現実的な目標はDFTの置き換えではなく、DFTが扱えない強相関の活性空間を供給すること。
  * 競争相手はfull CIではなくDMRG・量子モンテカルロ・テンソルネットワーク。量子優位性はそれらすべてに同時に勝つことを要求する。

**3. 第二量子化とJordan-Wignerが量子ビットへの橋である**

  * フェルミオンハミルトニアン $\sum h_{pq}\hat{c}_p^\dagger \hat{c}_q + \frac{1}{2}\sum h_{pqrs}\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s$ は $O(M^4)$ 個の項をもつ。
  * Jordan-Wigner：$\hat{c}_p = (\prod_{q<p} Z_q)(X_p + iY_p)/2$、$\hat{n}_p = (I - Z_p)/2$。占有は局所のまま、ホッピングの重みは軌道間距離に比例して増える。
  * Bravyi-Kitaevや木構造の写像は重みを $O(\log M)$ に減らし、数十軌道を超えると重要になる。
  * 反交換関係は必ず数値的に検証すること。パリティ文字列の欠落は静かなバグである。

**4. デジタルシミュレーションは深さを消費し、そのコストは測定できる**

  * Hubbard二量体で1次Trotter誤差 × $r \to 3.366$、2次誤差 × $r^2 \to 2.169$。漸近スケーリングが定数付きで確認された。
  * 4量子ビット模型の1時間ステップでユニタリ誤差 $10^{-6}$ に達するには、1次では $\sim 3 \times 10^7$ 回のPauli回転が必要。
  * QPEは深さ $O(1/\varepsilon)$ で固有値を与え誤り訂正を要する。VQEは浅い深さで変分的な上限を与え測定回数で支払う。両者は2つの異なる機械に属する。

**5. モデルハミルトニアンは物理が読み取れる場所である**

  * 横磁場Ising：厳密可解で理想的なベンチマーク。周期鎖のエネルギーは自由フェルミオンの式と $10^{-14}$ で一致した。
  * Hubbard二量体：7個のPauli文字列。基底状態エネルギーは $U/2 - \sqrt{(U/2)^2 + 4t^2}$ と計算機精度で一致。
  * 二重占有率は $U/t$ が0から32へ動くと0.25から0.0019へ — 2サイトでのMott局在。
  * $\langle \mathbf{S}_0 \cdot \mathbf{S}_1 \rangle \to -3/4$、$E_0 \to -4t^2/U$（$U/t = 128$ で比0.99976）：超交換相互作用が定量的に現れる。
  * スピンギャップは $4t^2/U$ で閉じ、電荷ギャップは $U - 2t$ で開く：Mott絶縁体の2つのエネルギースケール構造。

**6. VQEには切り分け可能な3つの誤差源があり、努力で縮むのは1つだけ**

  * ansatz誤差は片側で再現的。層を増やすと収束が悪くなることがある。
  * 最適化器の誤差は乱数シードで変わる。同じ回路で6回の再開始が $5\times10^{-4}$ から $5\times10^{-2}$ に広がった。
  * 測定誤差は $1/\sqrt{N_{\text{shots}}}$ で縮む。項あたり $10^4$ ショットでも7項ハミルトニアンでばらつき0.023が残った。
  * エネルギーだけでなく観測量（二重占有率、粒子数、状態の重なり）を検査することが、正しい計算と運の良い計算を分ける。

**実務上の含意**

  * テストする系については必ず厳密解または解析解の参照値を計算すること。それなしに量子計算の結果は解釈できない。
  * 軌道の順序は意図して選ぶこと。スペクトルを変えずに回路の深さを変える。
  * 深さ・幅・測定を別々に見積もること。拘束条件が量子ビット数であることはほとんどない。
  * 変分エネルギーは上限として扱い、診断（再開始、観測量、ショット数）を併記すること。
  * 主張がどの時代のものかを明確にすること。「量子コンピュータならFeMocoが解ける」は誤り耐性機械についての言明であり、そう言うことは研究計画とプレスリリースの違いである。

### 次章へ

これで電子からPauli文字列までの鎖が揃い、厳密に対角化できる2つのモデルハミルトニアンと、それを再現するVQEが手元にあります。ここまでの数値はすべて完璧な量子コンピュータを仮定していました。第5章はその仮定を外します。同じ状態ベクトルシミュレータ上に軌跡法による脱分極チャネルとしてノイズモデルを組み込み、現実的な誤り率で忠実度が回路深さとともにどれだけ速く減衰するかを測定します。それから上で計算したまさにそのVQE状態にゼロノイズ外挿を適用し、ノイズ由来のバイアスをどれだけ回収できるか、そしてサンプル数でいくら支払うかを見ます。最後に本章の3つの予算 — 深さ、幅、測定 — を誤り訂正が要求するものの隣に置き、近未来の量子コンピューティングが材料研究に対して何ができて何ができないかを率直に述べます。

[← 第3章: 変分量子固有値ソルバー](<chapter-3.html>) [第5章: NISQの現実と展望 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章に引用したモデルパラメータ、活性空間の規模、資源見積りは教育目的のための文献規模の代表値です。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
