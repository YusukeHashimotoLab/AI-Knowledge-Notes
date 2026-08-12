---
title: "第1章: 量子ビットと重ね合わせ"
chapter_title: "第1章: 量子ビットと重ね合わせ"
subtitle: 状態ベクトル、Bloch球、Born則、そして指数の壁
reading_time: 30-35分
difficulty: 初級
code_examples: 8
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-computing-introduction/chapter-1.html>) | Last sync: 2026-08-12

[基礎数理道場](<../index.html>) > [量子コンピューティング入門](<index.html>) > 第1章

本章では、本シリーズの以降すべてが作用する対象、すなわち量子ビットレジスタの状態を構築します。Schrödinger方程式を解いた経験があれば、必要な数学はほぼすでにご存じのはずです。本章の第一の役割は、新しい形式論を教えることではなく、その対応関係を明示することにあります。第二の役割は定量的なものです。材料研究者が量子コンピューティングに関心をもつべき理由は多電子波動関数のスケーリングに関する議論であり、その議論は量子ビットが登場する前に数値で示しておく価値があります。8つのコード例を通じて、章の終わりには状態ベクトルシミュレータの最初の3関数が完成し、それを第2章から第5章まで変更せずに使い続けます。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * 電子構造問題の厳密解が指数関数的に困難である理由を定量的に説明し、与えられた活性空間に対するfull CI空間の次元を述べられる
  * 量子ビット状態を規格化された複素ベクトルとして書き、ノルムと内積を計算し、大域位相が物理的情報をもたない一方で相対位相はもつ理由を説明できる
  * 任意の1量子ビット純粋状態を2つの角度でパラメータ化し、状態ベクトルとBlochベクトルを相互に変換し、Bloch球の描像が多量子ビットに一般化しない理由を説明できる
  * Born則から測定確率を求め、射影測定後の状態の収縮を記述し、Pauli観測量の期待値を計算できる
  * 多量子ビット状態をテンソル積として構成し、本シリーズのbig-endian規約を正しく使い、Schmidtランクによって積状態とエンタングル状態を区別できる
  * `ket`、`probs`、`sample` をNumPyで実装し、与えられた精度の期待値に必要な測定ショット数を見積もれる

* * *

## 1.1 なぜ材料研究者が量子計算を学ぶのか

### すでに手元にある問題

Born-Oppenheimer近似のもとでの分子や固体の非相対論的な電子ハミルトニアンは、完全に既知です。

\\[ \hat{H} = -\sum_i \frac{\hbar^2}{2m_e}\nabla_i^2 - \sum_{i,A} \frac{Z_A e^2}{4\pi\epsilon_0 r_{iA}} + \sum_{i<j} \frac{e^2}{4\pi\epsilon_0 r_{ij}} \\]

化学や固体物理が要求する精度の水準において、この式に未知の項も近似も含まれていません。困難はもっぱら解を求める段階にあります。基底状態 \\(\Psi(r_1, \ldots, r_N)\\) は \\(3N\\) 個の座標の関数であり、交換に対して反対称でなければなりません。これを \\(M\\) 個の空間軌道からなる有限基底で展開すると、次元が2つの二項係数の積で与えられる線形空間が得られます。

\\[ \dim H_\text{FCI} = \binom{M}{N_\alpha}\binom{M}{N_\beta} \\]

これが**full configuration interaction**（full CI）の次元、すなわち \\(M\\) 個の軌道に上向きスピン \\(N_\alpha\\) 個と下向きスピン \\(N_\beta\\) 個の電子を配置して作れるSlater行列式の数です。この量は \\(M\\) のいかなる多項式よりも速く増大します。控えめな6-31G基底の水分子でもすでに約 \\(1.7 \times 10^6\\)、ニトロゲナーゼの鉄モリブデン補因子について通常引用される活性空間では約 \\(10^{35}\\) に達します。

### 標準的な手法がそれにどう対処しているか

計算化学と電子構造理論の道具立て全体は、full CI問題を「解かないための」戦略の集合と見ることができます。

| 手法 | 形式的スケーリング | 前提としているもの |
| --- | --- | --- |
| 密度汎関数理論（KS-DFT） | \\(O(M^3)\\) | 未知の交換相関汎関数を近似できること、単参照的な性格 |
| Hartree-Fock | \\(O(M^4)\\) | 1つの行列式のみ。相関を完全に無視 |
| MP2 | \\(O(M^5)\\) | 良い単参照のまわりの摂動的な相関 |
| CCSD | \\(O(M^6)\\) | 指数型ansatzを二重励起で打ち切ること |
| CCSD(T) | \\(O(M^7)\\) | 「gold standard」— ただし弱相関系に限る |
| DMRG | 結合次元について多項式 | 低いエンタングルメント、実質的に1次元的な結合性 |
| full CI | 指数関数的 | 何も仮定しない。与えられた基底では厳密 |

この表で多項式スケーリングを実現している項目はいずれも、波動関数の構造に関する仮定を対価としてその速さを買っています。そして各仮定には、それを破る物質群が存在します。不都合なことに、それらは面白い物質群でもあります。

  * **遷移金属酸化物・錯体** ：部分的に占有された \\(d\\) 殻はほぼ縮退した複数の行列式を生じるため、単参照は定性的に誤りとなります。
  * **鉄硫黄クラスターと金属酵素の活性中心** ：強く結合した開殻金属中心が数十個 — FeMoco問題です。
  * **銅酸化物などの相関超伝導体** ：関心の対象となる物性が、まさに平均場理論が捨てている部分です。
  * **単分子磁石、アクチノイド化学、結合解離、多参照的な性格をもつ励起状態。**

これらすべてに与えられる呼称が**強相関**であり、その意味はどの場合も同一です。すなわち、単一の行列式もその低次補正も、良い出発点にはならないということです。

### Feynmanの指摘

1981年、Feynmanはこの困難が根本的な不可能性ではなく表現の不整合であることを指摘しました。古典計算機は量子状態の振幅を1つずつ数値として保持するので、指数関数的に多くの数値を必要とします。適切な自由度をもつ量子系は、その状態を「自分自身の状態として」保持し、表現のコストはまったくかかりません。量子系のシミュレーションは量子機械にとって自然な仕事である、というのが彼の主張でした。

具体的には、第4章のJordan-Wigner写像のもとで1つのスピン軌道が1つの量子ビットになります。すると必要資源の性質が完全に変わります。

| 量 | 古典的な厳密解法 | 量子ビットレジスタ |
| --- | --- | --- |
| 状態の記憶量 | \\(\binom{M}{N_\alpha}\binom{M}{N_\beta}\\) 個の振幅 | \\(2M\\) 量子ビット |
| 軌道数に対する増大 | 指数関数的 | 線形 |

この1行が、材料科学における量子コンピューティングの物理的な根拠のすべてです。これは速度についての主張ではなく、あらゆる計算についての主張でもありません。多体量子状態の**表現**についての主張です。

### コード例1: 指数の壁を数値で見る

「指数関数的」という抽象的な言明は、実際のバイト数よりもはるかに説得力に欠けます。この例では代表的な活性空間についてfull CIの次元を評価し、同じ問題が必要とする量子ビット数と比較します。

```python
import numpy as np
from math import comb


def fci_dimension(n_orb: int, n_elec: int) -> int:
    """full CI空間の次元：n_orb個の空間軌道にn_elec個の電子を配置する場合。"""
    n_alpha = (n_elec + 1) // 2
    n_beta = n_elec // 2
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


def human_bytes(nbytes: float) -> str:
    """バイト数を2進接頭辞つきで整形する。"""
    for unit in ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2e} YB"


# (名称, 空間軌道数, 電子数) — 代表的な活性空間
# FeMoco: FeMo補因子全体の (113電子, 76空間軌道) 活性空間。本シリーズで
# FeMocoに言及する際は一貫してこの規約を用いる。
systems = [
    ("H2 / STO-3G",          2,   2),
    ("H2O / 6-31G",         13,  10),
    ("N2 / cc-pVDZ",        28,  14),
    ("Fe2S2 active space",  20,  30),
    ("FeMoco active space", 76, 113),
]

header = (f"{'system':<22}{'orbitals':>9}{'electrons':>10}"
          f"{'qubits':>8}{'FCI dimension':>16}{'CI vector':>12}")
print(header)
print("-" * len(header))
for name, n_orb, n_elec in systems:
    dim = fci_dimension(n_orb, n_elec)
    print(f"{name:<22}{n_orb:>9}{n_elec:>10}{2 * n_orb:>8}"
          f"{dim:>16.3e}{human_bytes(dim * 8):>12}")

print()
print("n量子ビットの状態ベクトルを古典計算機で保持する場合（complex128）:")
for n in [10, 20, 30, 40, 50]:
    print(f"  n = {n:>3} qubits -> {2**n:>16,d} 個の振幅"
          f" -> {human_bytes(2**n * 16):>10}")

# 困難な問題に必要な量子ビット数と、対応する状態ベクトルの大きさ
n_qubits = 2 * 76
print()
print(f"FeMocoの活性空間はJordan-Wigner変換で{n_qubits}量子ビットを必要とする。")
print(f"同じ大きさの古典状態ベクトルは 2^{n_qubits} = "
      f"{2.0**n_qubits:.2e} 個の振幅をもつ。")
print(f"比較：観測可能な宇宙の原子数は約1e80。")
```

```
system                 orbitals electrons  qubits   FCI dimension   CI vector
-----------------------------------------------------------------------------
H2 / STO-3G                   2         2       4       4.000e+00      32.0 B
H2O / 6-31G                  13        10      26       1.656e+06     12.6 MB
N2 / cc-pVDZ                 28        14      56       1.402e+12     10.2 TB
Fe2S2 active space           20        30      40       2.404e+08      1.8 GB
FeMoco active space          76       113     152       4.169e+35 2.76e+12 YB

n量子ビットの状態ベクトルを古典計算機で保持する場合（complex128）:
  n =  10 qubits ->            1,024 個の振幅 ->    16.0 kB
  n =  20 qubits ->        1,048,576 個の振幅 ->    16.0 MB
  n =  30 qubits ->    1,073,741,824 個の振幅 ->    16.0 GB
  n =  40 qubits -> 1,099,511,627,776 個の振幅 ->    16.0 TB
  n =  50 qubits -> 1,125,899,906,842,624 個の振幅 ->    16.0 PB

FeMocoの活性空間はJordan-Wigner変換で152量子ビットを必要とする。
同じ大きさの古典状態ベクトルは 2^152 = 5.71e+45 個の振幅をもつ。
比較：観測可能な宇宙の原子数は約1e80。
```

**着目点。** 二重ゼータ基底の窒素分子 — 大学院生なら「小さい系」と呼ぶ規模 — でも、CIベクトルにすでに10テラバイトを要します。これがそのサイズでfull CIが事実上まったく実行されない理由です。FeMocoの行は「大きな数」ではなく「意味をもたない数」です。\\(10^{35}\\) 個の振幅を保持できる古典計算機は考えられません。それにもかかわらず、量子ビットの列は控えめな値にとどまり、2ずつしか増えません。2番目の表は同じ壁を反対側から見たものであり、本シリーズのシミュレーションがすべて約20量子ビット以下にとどまる理由でもあります。16 MBは快適ですが、16 TBはそうではありません。

### 誠実さを置くべき場所

上の議論が過大な主張になる前に、2つの留保をつけておく必要があります。いずれも第3章と第5章で展開します。

  1. **レジスタを持つことは状態を持つことではありません。** 量子ビットレジスタは基底状態を*保持*できますが、それを*準備*することは別の問題です。一般に局所ハミルトニアンの基底状態を求めることはQMA困難であり、量子計算機にとってさえ困難です。VQEを含むあらゆる実用的アルゴリズムは、この準備段階に対するヒューリスティクスであり、優位性の証明はありません。
  2. **答えを読み出すにはショットが必要です。** 状態は \\(2^n\\) 個の振幅を含みますが、1回の測定が返すのはビット列1つだけです。エネルギーを化学的精度で取り出すには膨大な繰り返しが必要で、コード例8がそれを実測し、第3章では設計上の制約として扱います。

いずれの点もスケーリングの議論を無効にはしません。しかし両方とも、この分野が30年を経てなお材料科学にとって実用前段階にある理由を説明します。

* * *

## 1.2 状態ベクトルとディラック記法

### 2準位系としての量子ビット

**量子ビット**とは、アクセス可能かつ制御可能な2つの準位をもつ任意の量子系です。計算に固有のものは何もありません。磁気共鳴や光学分光でおなじみの2準位系そのものです。

| 物理的実装 | 2つの準位 | 材料研究者が出会う場面 |
| --- | --- | --- |
| 磁場中のスピン1/2 | \\(m_s = +1/2, -1/2\\) | NMR、ESR、磁化ダイナミクス |
| 超伝導トランズモン | 非調和振動子の最低2準位 | 現在最も広く使われるハードウェア |
| イオントラップの超微細準位 | 長寿命な2つの原子状態 | 測定された最高忠実度のゲート |
| 光子の偏光 | 水平、垂直 | 量子光学、線形光学方式 |
| ダイヤモンド中のNVセンター | S=1基底状態の \\(m_s = 0, \pm 1\\) | 量子センシング、磁気計測 |
| 半導体量子ドットのスピン | 電子スピンの上向き・下向き | シリコン互換の量子ビット提案 |

この抽象化の要点は、以下の数学がこれらすべてに対して同一だということです。物理的な実装に立ち戻るのは、ノイズがそれに依存する第5章だけです。

### ケット、振幅、規格化

2つの準位を \\(|0\rangle\\) と \\(|1\rangle\\) と名づけ、2次元複素Hilbert空間の正規直交基底とします。

\\[ |0\rangle = \begin{pmatrix} 1 \\\\ 0 \end{pmatrix}, \qquad |1\rangle = \begin{pmatrix} 0 \\\\ 1 \end{pmatrix} \\]

一般の純粋状態はその複素線形結合、すなわち**重ね合わせ**です。

\\[ |\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\\\ \beta \end{pmatrix}, \qquad \alpha, \beta \in \mathbb{C} \\]

\\(\alpha\\) と \\(\beta\\) は**振幅**であり、確率ではありません。この違いこそが本題です。制約は規格化条件だけです。

\\[ \langle\psi|\psi\rangle = |\alpha|^2 + |\beta|^2 = 1 \\]

**ブラ** \\(\langle\psi|\\) はケットの共役転置なので、2つの状態の内積は次のようになります。

\\[ \langle\phi|\psi\rangle = \phi_0^{\ast}\psi_0 + \phi_1^{\ast}\psi_1 \\]

NumPyではこれは `np.vdot(phi, psi)` であり、*第1引数*を複素共役します — ブラ・ケットと同じ規約です。代わりに `np.dot` を使うのは典型的で、しかも見つけにくい誤りです。実ベクトルでは `vdot` と一致し、複素ベクトルでは一致しないからです。

内積が \\(\langle\phi|\psi\rangle = 0\\) となる2つの状態は**直交**します。直交する状態は適切な測定によって完全に識別でき、直交しない状態は完全には識別できません。

### 名前をつけておきたい4つの重ね合わせ

\\[ |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \qquad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}, \qquad |{\pm}i\rangle = \frac{|0\rangle \pm i|1\rangle}{\sqrt{2}} \\]

これら4つはすべて、\\(\\{|0\rangle, |1\rangle\\}\\) 基底で測ればどちらの結果も確率 \\(1/2\\) になります。それでもこれらは4つの異なる物理状態です。\\(|+\rangle\\) と \\(|-\rangle\\) は互いに直交するので、適切な測定によって確実に区別できます。

これが、量子ビットが確率的な古典ビットから乖離する最初の場所です。半分の確率で表が出るコインの記述は1通りですが、\\(P(0) = 1/2\\) をもつ量子ビットの記述は連続無限個あり、それらは振幅間の**相対位相**によって区別されます。この位相こそが干渉を可能にし、量子アルゴリズムは干渉でできています。

### 大域位相と相対位相

一般の状態を極形式で書きます。

\\[ |\psi\rangle = e^{i\gamma}\left(\cos\frac{\theta}{2}|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}|1\rangle\right) \\]

全体にかかる因子 \\(e^{i\gamma}\\) は**大域位相**です。あらゆる測定確率、あらゆる期待値から打ち消されるため物理的ではありません。\\(|\psi\rangle\\) と \\(e^{i\gamma}|\psi\rangle\\) は*同じ状態*であり、したがって1量子ビットの物理的な状態空間は3個ではなく2個の実パラメータで表されます。

内側の因子 \\(e^{i\varphi}\\) は2つの振幅の間の**相対位相**であり、これは完全に物理的です。\\(\varphi\\) を変えれば状態は別の識別可能な状態へ移ります。この区別を正しく保つことが、動くアルゴリズムとデバッグ作業の分かれ目になります。

### コード例2: 振幅、ノルム、位相

```python
import numpy as np

# 計算基底
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

# Z基底の測定統計がすべて50/50になる4つの状態
plus = (ket0 + ket1) / np.sqrt(2)          # |+>
minus = (ket0 - ket1) / np.sqrt(2)         # |->
plus_i = (ket0 + 1j * ket1) / np.sqrt(2)   # |+i>
minus_i = (ket0 - 1j * ket1) / np.sqrt(2)  # |-i>

states = {"|0>": ket0, "|1>": ket1, "|+>": plus,
          "|->": minus, "|+i>": plus_i, "|-i>": minus_i}


def norm(psi):
    """ノルム sqrt(<psi|psi>)。np.vdot は第1引数を複素共役する。"""
    return np.sqrt(np.real(np.vdot(psi, psi)))


print("state    amplitudes                          norm   P(0)    P(1)")
print("-" * 68)
for name, psi in states.items():
    p = np.abs(psi) ** 2
    amp = f"[{psi[0]:+.3f}, {psi[1]:+.3f}]"
    print(f"{name:<8} {amp:<36} {norm(psi):.3f}  {p[0]:.3f}   {p[1]:.3f}")

# 任意のベクトルを規格化する
raw = np.array([3, 4j], dtype=complex)
psi = raw / norm(raw)
print(f"\n規格化前 : {raw}, ノルム = {norm(raw):.4f}")
print(f"規格化後 : {np.round(psi, 4)}, ノルム = {norm(psi):.4f}")
print(f"確率     : P(0) = {abs(psi[0])**2:.4f}, P(1) = {abs(psi[1])**2:.4f}")

# 大域位相は観測できないが、相対位相は観測できる
gamma = 0.7
print("\n大域位相と相対位相")
print(f"  |+>              の確率: {np.round(np.abs(plus)**2, 6)}")
print(f"  e^(i*0.7)|+>     の確率: "
      f"{np.round(np.abs(np.exp(1j*gamma)*plus)**2, 6)}   <- 同一")
print(f"  |->              の確率: {np.round(np.abs(minus)**2, 6)}"
      "   <- この基底では同じく同一")
print(f"  |<+|->|          = {abs(np.vdot(plus, minus)):.3f}"
      "   <- 直交：別の基底では完全に識別できる")
print(f"  |<+|e^(i*0.7)|+>|= {abs(np.vdot(plus, np.exp(1j*gamma)*plus)):.3f}"
      "   <- 絶対値1：物理的に全く同じ状態")

# 計算基底の正規直交性
gram = np.array([[np.vdot(a, b) for b in (ket0, ket1)] for a in (ket0, ket1)])
print("\n{|0>, |1>} のGram行列:")
print(np.real_if_close(gram))
```

```
state    amplitudes                          norm   P(0)    P(1)
--------------------------------------------------------------------
|0>      [+1.000+0.000j, +0.000+0.000j]       1.000  1.000   0.000
|1>      [+0.000+0.000j, +1.000+0.000j]       1.000  0.000   1.000
|+>      [+0.707+0.000j, +0.707+0.000j]       1.000  0.500   0.500
|->      [+0.707+0.000j, -0.707+0.000j]       1.000  0.500   0.500
|+i>     [+0.707+0.000j, +0.000+0.707j]       1.000  0.500   0.500
|-i>     [+0.707+0.000j, +0.000-0.707j]       1.000  0.500   0.500

規格化前 : [3.+0.j 0.+4.j], ノルム = 5.0000
規格化後 : [0.6+0.j  0. +0.8j], ノルム = 1.0000
確率     : P(0) = 0.3600, P(1) = 0.6400

大域位相と相対位相
  |+>              の確率: [0.5 0.5]
  e^(i*0.7)|+>     の確率: [0.5 0.5]   <- 同一
  |->              の確率: [0.5 0.5]   <- この基底では同じく同一
  |<+|->|          = 0.000   <- 直交：別の基底では完全に識別できる
  |<+|e^(i*0.7)|+>|= 1.000   <- 絶対値1：物理的に全く同じ状態

{|0>, |1>} のGram行列:
[[1. 0.]
 [0. 1.]]
```

**着目点。** 4つの重ね合わせは `P(0)` と `P(1)` の列が同一で、振幅の列が異なります。「量子ビットは単なるランダムビットではない」という言葉の内容はこれに尽きます。最後の2行は2種類の位相を分離しています。\\(|+\rangle\\) と \\(|-\rangle\\) の重なりは0（\\(Z\\) 基底の統計が同一な、異なる状態）であり、\\(|+\rangle\\) と \\(e^{i\gamma}|+\rangle\\) の重なりは絶対値1（数値の書き方が違うだけの同じ状態）です。すべての配列に `dtype=complex` を指定していることにも注意してください。実数配列に代入すると虚部が警告なしに切り捨てられ、エラーメッセージなしに誤った物理が得られます。

* * *

## 1.3 Bloch球

### 2つの角度で足りる

大域位相を捨て規格化を課すと、実パラメータはちょうど2個残ります。したがって1量子ビットの純粋状態は2次元の曲面をなします。標準的なパラメータ化は

\\[ |\psi(\theta, \varphi)\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}|1\rangle, \qquad \theta \in [0, \pi], \quad \varphi \in [0, 2\pi) \\]

であり、この曲面は球面 — **Bloch球** — になります。対応を具体化するのは3つのPauli期待値で、これらが**Blochベクトル**を定義します。

\\[ \mathbf{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle) = (\sin\theta\cos\varphi,\; \sin\theta\sin\varphi,\; \cos\theta) \\]

ここでPauli行列は

\\[ X = \begin{pmatrix} 0 & 1 \\\\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\\\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\\\ 0 & -1 \end{pmatrix} \\]

です。すべての純粋状態について \\(|\mathbf{r}| = 1\\) が厳密に成り立ちます。目印となる状態は覚えておく価値があります。

| 状態 | \\(\theta\\) | \\(\varphi\\) | Blochベクトル | 位置 |
| --- | --- | --- | --- | --- |
| \\(\lvert 0\rangle\\) | 0 | — | \\((0,0,1)\\) | 北極 |
| \\(\lvert 1\rangle\\) | \\(\pi\\) | — | \\((0,0,-1)\\) | 南極 |
| \\(\lvert +\rangle\\) | \\(\pi/2\\) | 0 | \\((1,0,0)\\) | \\(+x\\) |
| \\(\lvert -\rangle\\) | \\(\pi/2\\) | \\(\pi\\) | \\((-1,0,0)\\) | \\(-x\\) |
| \\(\lvert {+}i\rangle\\) | \\(\pi/2\\) | \\(\pi/2\\) | \\((0,1,0)\\) | \\(+y\\) |
| \\(\lvert {-}i\rangle\\) | \\(\pi/2\\) | \\(3\pi/2\\) | \\((0,-1,0)\\) | \\(-y\\) |

この幾何には、驚かれることが多いので明示しておくべき特徴が2つあります。**直交する状態は垂直ではなく対極にあります。** \\(\langle 0|1\rangle = 0\\) であり、両者のBlochベクトルは正反対を向きます。また \\(\cos(\theta/2)\\) の半角は、Hilbert空間の角度がBloch角の半分であることを意味します。Bloch球上の \\(2\pi\\) 回転は状態ベクトルに \\(-1\\) の因子を与える — おなじみのスピノルの二重被覆です。

### この描像が物理的に実在する理由

スピン1/2の場合、Blochベクトルは記帳のための道具ではありません。それはg因子の分だけ換算すれば、そのまま磁気モーメントの向き*そのもの*です。印加磁場まわりのLarmor precessionはBlochベクトルの剛体回転であり、Rabiパルスは赤道面内の軸まわりの回転、\\(T_1\\) 緩和は \\(z\\) 成分を縮め、\\(T_2\\) 位相緩和は横成分を縮めます。ESRやNMRの実験を解釈したことがある人は、すでに量子ゲートの幾何を扱っていたのです。第2章では同じ回転にゲートの名前を与え、第5章ではベクトルが球の内部へ縮むことを許します。そこが混合状態と密度行列の入口です。

### この描像が通用しなくなる場所

Bloch球は1量子ビット専用の道具であり、一般化しません。\\(n\\) 量子ビットの純粋状態は規格化と大域位相を除いて \\(2 \cdot 2^n - 2\\) 個の実パラメータをもちますが、\\(n\\) 個のBloch球が担えるのは \\(3n\\) 個です。\\(n = 2\\) では6対6となりますが、それでも描像は破綻します。2量子ビットのパラメータは独立な2本のBlochベクトルとして分配されていないからです。エンタングルメントは相関のなかに住み、Bell状態の各量子ビットのBlochベクトルは \\(\mathbf{0}\\) — 球の中心 — になります。「各量子ビットは1本の矢印である」という形の直観は、面白い物理が始まるちょうどその場所で破綻します。

### コード例3: BlochベクトルとBloch球の描画

この例では2つの表現を双方向に変換し、すべての純粋状態が単位球面上にあることを数値的に検証し、目印となる状態を記した球面を描きます。図はコードが生成します。手元で実行して確認してください。本文の内容は画像に依存しません。

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """1量子ビット状態のBlochベクトル (<X>, <Y>, <Z>)。"""
    return np.array([np.real(np.vdot(psi, P @ psi)) for P in (X, Y, Z)])


def angles_from_state(psi: np.ndarray) -> tuple:
    """状態ベクトルから (theta, phi) を復元する（大域位相を除去）。"""
    a, b = psi
    theta = 2 * np.arccos(np.clip(np.abs(a), 0.0, 1.0))
    phi = 0.0 if np.isclose(np.abs(b), 0.0) else np.angle(b) - np.angle(a)
    return theta, np.mod(phi, 2 * np.pi)


named = {
    "|0>":  (0.0, 0.0),
    "|1>":  (np.pi, 0.0),
    "|+>":  (np.pi / 2, 0.0),
    "|->":  (np.pi / 2, np.pi),
    "|+i>": (np.pi / 2, np.pi / 2),
    "|-i>": (np.pi / 2, 3 * np.pi / 2),
    "(pi/2,pi/4)": (np.pi / 2, np.pi / 4),
}

print("state        theta/pi  phi/pi     <X>      <Y>      <Z>   |r|")
print("-" * 66)
for name, (theta, phi) in named.items():
    psi = state_from_angles(theta, phi)
    r = np.round(bloch_vector(psi), 12) + 0.0
    print(f"{name:<12} {theta/np.pi:>7.3f} {phi/np.pi:>7.3f}  "
          f"{r[0]:>+7.3f} {r[1]:>+7.3f} {r[2]:>+7.3f}  {np.linalg.norm(r):.3f}")

# 往復変換：状態 -> Bloch角 -> 状態
theta0, phi0 = 1.1, 2.3
psi = state_from_angles(theta0, phi0)
theta1, phi1 = angles_from_state(psi)
print(f"\n往復変換: (theta, phi) = ({theta0:.4f}, {phi0:.4f}) -> "
      f"({theta1:.4f}, {phi1:.4f})")

# 純粋状態はすべて厳密に単位球面上にある
rng = np.random.default_rng(7)
radii = []
for _ in range(2000):
    v = rng.normal(size=4)
    psi = (v[:2] + 1j * v[2:]) / np.linalg.norm(v)
    radii.append(np.linalg.norm(bloch_vector(psi)))
print(f"ランダムな純粋状態2000個: |r| は "
      f"[{min(radii):.12f}, {max(radii):.12f}]  （厳密に1のはず）")

# --- 可視化 ---------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)),
                  np.outer(np.sin(u), np.sin(v)),
                  np.outer(np.ones_like(u), np.cos(v)),
                  color="lightgray", linewidth=0.4)

for axis, label in zip(np.eye(3), ["x  (<X>)", "y  (<Y>)", "z  (<Z>)"]):
    ax.quiver(0, 0, 0, *axis, color="k", arrow_length_ratio=0.08, linewidth=1)
    ax.text(*(1.15 * axis), label, fontsize=9)

for name, (theta, phi) in named.items():
    r = bloch_vector(state_from_angles(theta, phi))
    ax.quiver(0, 0, 0, *r, color="tab:purple", arrow_length_ratio=0.12, linewidth=2)
    ax.text(*(1.08 * r), name, fontsize=10, color="tab:purple")

ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
ax.set_axis_off()
ax.set_title("Bloch sphere: single-qubit pure states", fontsize=13)
plt.tight_layout()
plt.show()
```

```
state        theta/pi  phi/pi     <X>      <Y>      <Z>   |r|
------------------------------------------------------------------
|0>            0.000   0.000   +0.000  +0.000  +1.000  1.000
|1>            1.000   0.000   +0.000  +0.000  -1.000  1.000
|+>            0.500   0.000   +1.000  +0.000  +0.000  1.000
|->            0.500   1.000   -1.000  +0.000  +0.000  1.000
|+i>           0.500   0.500   +0.000  +1.000  +0.000  1.000
|-i>           0.500   1.500   +0.000  -1.000  +0.000  1.000
(pi/2,pi/4)    0.500   0.250   +0.707  +0.707  +0.000  1.000

往復変換: (theta, phi) = (1.1000, 2.3000) -> (1.1000, 2.3000)
ランダムな純粋状態2000個: |r| は [1.000000000000, 1.000000000000]  （厳密に1のはず）
```

**着目点。** 表に出力されたBlochベクトルは目印の表を正確に再現し、往復変換は状態ベクトルが任意の位相規約で作られていても \\((\theta, \varphi)\\) を機械精度で復元します。`angles_from_state` が意図的に `np.angle(a)` を引いて大域位相を除いているためです。ランダム状態の検査は、球の*表面*が純粋状態の空間であることの数値的な表現です。独立に2000回引いたすべての状態が、12桁の精度で半径1に載ります。\\(|\mathbf{r}| < 1\\) となる状態は混合状態であり、そもそも単一の状態ベクトルでは書けません。

* * *

## 1.4 測定とBorn則

### 規則

測定は、振幅がついに観測可能になる場所です。計算基底での測定について、**Born則**は次を述べます。

\\[ P(k) = |\langle k|\psi\rangle|^2 = |\psi_k|^2 \\]

したがって1量子ビットでは \\(P(0) = |\alpha|^2\\)、\\(P(1) = |\beta|^2\\) です。規格化条件は、これらの和が1になるという要請にほかなりません。Blochパラメータで書けば

\\[ P(0) = \cos^2\frac{\theta}{2}, \qquad P(1) = \sin^2\frac{\theta}{2} \\]

となり、\\(Z\\) 基底の統計は極角のみで決まり、方位角はこの測定には見えません。別の軸に沿って測るのは別の実験であり、\\(\varphi\\) を観測可能にするのはこの軸の選択です。

### 収縮

測定の公理の後半は**射影の公理**です。結果 \\(k\\) が得られた測定の後、状態は規格化された射影になります。

\\[ |\psi\rangle \;\longrightarrow\; \frac{P_k|\psi\rangle}{\sqrt{\langle\psi|P_k|\psi\rangle}}, \qquad P_k = |k\rangle\langle k| \\]

実務上重要な帰結は3つあります。

  * **反復可能性。** 同じ量子ビットを同じ基底で2回測ると、2回目は確率1で同じ答えを返します。1回目の測定がすでに重ね合わせを壊しているからです。
  * **不可逆性。** 収縮は形式論全体のなかで唯一の非ユニタリな操作です。第2章のあらゆるゲートは可逆ですが、測定はそうではありません。
  * **1ショットあたり1つのビット列。** \\(n\\) 量子ビットレジスタの1回の測定は、\\(2^n\\) 通りのうち1つのビット列を返します。他の振幅はすべて失われます。これが、量子アルゴリズムが指数的に大きな状態を単純に「読み出す」ことができない理由であり、有用なアルゴリズムが測定の*前*に干渉によって答えへ確率を集中させる理由です。

### 期待値

関心のある量の多く — とりわけエネルギー — は単一の測定結果ではなく観測量の期待値です。エルミート演算子 \\(A\\) に対して

\\[ \langle A\rangle = \langle\psi|A|\psi\rangle = \sum_k a_k P(a_k) \\]

です。固有値が \\(\pm 1\\)、固有ベクトルが \\(|0\rangle\\) と \\(|1\rangle\\) であるPauli \\(Z\\) については

\\[ \langle Z\rangle = (+1)P(0) + (-1)P(1) = P(0) - P(1) = \cos\theta \\]

となり、これは実験が従う具体的な手順そのものです。多数回測定し、結果0に \\(+1\\)、結果1に \\(-1\\) を割り当て、平均を取ります。分散は \\(Z^2 = I\\) から得られます。

\\[ \operatorname{Var}(Z) = \langle Z^2\rangle - \langle Z\rangle^2 = 1 - \langle Z\rangle^2 \\]

したがって \\(N\\) ショットからの推定値の統計誤差は \\(\sqrt{(1 - \langle Z\rangle^2)/N}\\) です。ここには好都合な特別な場合があります。状態が固有状態なら分散は消え、1ショットで足ります。測定に費用がかかるのは、その中間にある真に重ね合わさった状態です — そして変分アルゴリズムが訪れるのは、まさにそうした状態なのです。

一般のハミルトニアンは、Pauli文字列の線形結合として書くことで扱います。

\\[ H = \sum_j c_j\, P_j, \qquad \langle H\rangle = \sum_j c_j \langle P_j\rangle \\]

各項を別々に測定し、古典的な係数を掛けて足し合わせます。これが第3章の道具立てであり、コード例5がその最小形を先取りします。

### コード例4: Born則と収縮

```python
import numpy as np


def state_from_angles(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def probs(state):
    """Born則による全 2^n 通りの確率"""
    return np.abs(state) ** 2


def measure_qubit(state: np.ndarray, qubit: int, n: int, rng) -> tuple:
    """n量子ビット状態の1量子ビットをZ基底で測定する（big-endian）。

    (測定結果, 収縮後の状態) を返す。収縮後の状態は規格化されているので、
    そのまま再測定したり時間発展させたりできる。
    """
    psi = state.reshape([2] * n)
    # 結果0への射影：該当スライスを残し、他方をゼロにする
    sl0 = [slice(None)] * n
    sl0[qubit] = 0
    branch0 = np.zeros_like(psi)
    branch0[tuple(sl0)] = psi[tuple(sl0)]
    p0 = float(np.sum(np.abs(branch0) ** 2))

    if rng.random() < p0:
        return 0, (branch0 / np.sqrt(p0)).reshape(-1)
    branch1 = psi - branch0
    return 1, (branch1 / np.sqrt(1.0 - p0)).reshape(-1)


rng = np.random.default_rng(2026)

# 偏りのある1量子ビット状態：theta = pi/3 なら P(0) = cos^2(pi/6) = 3/4
theta, phi = np.pi / 3, 0.4
psi = state_from_angles(theta, phi)
p = probs(psi)
print(f"状態 |psi(theta=pi/3, phi=0.4)>  ->  P(0) = {p[0]:.4f}, P(1) = {p[1]:.4f}")
print(f"解析値 cos^2(theta/2) = {np.cos(theta/2)**2:.4f}, "
      f"sin^2(theta/2) = {np.sin(theta/2)**2:.4f}")

print("\n有限回の測定頻度はBorn則の確率に収束する:")
print(f"{'shots':>9}{'f(0)':>10}{'f(1)':>10}{'|f(0) - P(0)|':>16}")
print("-" * 45)
for shots in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    outcomes = rng.choice(2, size=shots, p=p)
    f0 = np.mean(outcomes == 0)
    print(f"{shots:>9}{f0:>10.4f}{1-f0:>10.4f}{abs(f0 - p[0]):>16.5f}")

# 収縮：同じ量子ビットの2回目の測定は必ず1回目と一致する
print("\n同一量子ビットの繰り返し測定（収縮）:")
agree = 0
for _ in range(1000):
    first, collapsed = measure_qubit(psi, 0, 1, rng)
    second, _ = measure_qubit(collapsed, 0, 1, rng)
    agree += (first == second)
print(f"  1回目と2回目が一致した回数: {agree}/1000")

# 積状態では、片方の測定が他方に影響しない
two = np.kron(state_from_angles(np.pi / 3, 0.0),
              state_from_angles(np.pi / 2, 0.0))   # |psi> (x) |+>
print("\n2量子ビットの積状態 |psi(pi/3,0)> (x) |+>:")
print(f"  |q0 q1> 上の全確率        : {np.round(probs(two), 4)}")
out, after = measure_qubit(two, 0, 2, rng)
print(f"  量子ビット0の測定結果      : {out}")
print(f"  収縮後の状態の確率         : {np.round(probs(after), 4)}")
print("  量子ビット1は依然50/50：積状態では一方の測定は他方について何も語らない")
```

```
状態 |psi(theta=pi/3, phi=0.4)>  ->  P(0) = 0.7500, P(1) = 0.2500
解析値 cos^2(theta/2) = 0.7500, sin^2(theta/2) = 0.2500

有限回の測定頻度はBorn則の確率に収束する:
    shots      f(0)      f(1)   |f(0) - P(0)|
---------------------------------------------
       10    0.8000    0.2000         0.05000
      100    0.7700    0.2300         0.02000
     1000    0.7270    0.2730         0.02300
    10000    0.7511    0.2489         0.00110
   100000    0.7529    0.2471         0.00294
  1000000    0.7502    0.2498         0.00018

同一量子ビットの繰り返し測定（収縮）:
  1回目と2回目が一致した回数: 1000/1000

2量子ビットの積状態 |psi(pi/3,0)> (x) |+>:
  |q0 q1> 上の全確率        : [0.375 0.375 0.125 0.125]
  量子ビット0の測定結果      : 0
  収縮後の状態の確率         : [0.5 0.5 0.  0. ]
  量子ビット1は依然50/50：積状態では一方の測定は他方について何も語らない
```

**着目点。** 収束の列はおおむね \\(1/\sqrt{N}\\) で減少しますが、単調ではありません。1000ショットの実行は偶然100ショットより悪くなっています。これは不具合ではなく、標準偏差が縮んでいく確率変数の見え方そのものです。最適化が収束したかどうかを判断しようとする前に、この感覚を身につけておく価値があります。収縮の検査は厳密で、1000回中1000回一致します。1回目の測定の後、状態は固有状態になっているからです。最後のブロックでは、*積*状態の量子ビット0を測定しても量子ビット1がそのまま残ることに注目してください。第2章では同じ実験をエンタングル状態に対して行い、結果は大きく異なります。

### コード例5: 期待値と2項ハミルトニアン

```python
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def expectation(psi: np.ndarray, A: np.ndarray) -> float:
    """エルミート演算子Aの期待値 <A> = <psi|A|psi>。"""
    return float(np.real(np.vdot(psi, A @ psi)))


print("theta/pi  phi/pi     <Z>   cos(theta)     <X>  sin(t)cos(p)     <Y>  sin(t)sin(p)")
print("-" * 84)
for theta in [0.0, np.pi / 4, np.pi / 2, 2 * np.pi / 3, np.pi]:
    for phi in [0.0, np.pi / 3]:
        psi = state_from_angles(theta, phi)
        ez, ex, ey = (expectation(psi, Z), expectation(psi, X), expectation(psi, Y))
        print(f"{theta/np.pi:>8.3f}{phi/np.pi:>8.3f}"
              f"{ez:>8.3f}{np.cos(theta):>13.3f}"
              f"{ex:>8.3f}{np.sin(theta)*np.cos(phi):>14.3f}"
              f"{ey:>8.3f}{np.sin(theta)*np.sin(phi):>14.3f}")

# 期待値は測定結果の重み付き和にほかならない：
# <Z> = (+1) P(0) + (-1) P(1)
theta, phi = 1.0, 0.5
psi = state_from_angles(theta, phi)
p = np.abs(psi) ** 2
print(f"\n演算子から求めた <Z>          : {expectation(psi, Z):.6f}")
print(f"(+1)P(0) + (-1)P(1) から      : {p[0] - p[1]:.6f}")

# 有限回の測定から <Z> を推定する
rng = np.random.default_rng(11)
exact = expectation(psi, Z)
print(f"\nショット数による <Z> の推定（厳密値 {exact:.6f}）:")
print(f"{'shots':>9}{'estimate':>12}{'error':>10}{'1/sqrt(N)':>12}")
print("-" * 43)
for shots in [10, 100, 1_000, 10_000, 100_000]:
    outcomes = rng.choice([+1.0, -1.0], size=shots, p=p)
    est = outcomes.mean()
    print(f"{shots:>9}{est:>12.4f}{abs(est-exact):>10.4f}{1/np.sqrt(shots):>12.4f}")

# Pauli項の線形結合としてのハミルトニアン（第3章で使うパターン）
coeffs = {"Z": -0.8, "X": 0.3}
paulis = {"X": X, "Y": Y, "Z": Z}
energy = sum(c * expectation(psi, paulis[k]) for k, c in coeffs.items())
H = sum(c * paulis[k] for k, c in coeffs.items())
print(f"\nH = -0.8 Z + 0.3 X")
print(f"  項ごとに足した <H>    : {energy:.6f}")
print(f"  行列から求めた <H>    : {expectation(psi, H):.6f}")
print(f"  厳密な基底エネルギー  : {np.linalg.eigvalsh(H)[0]:.6f}")
```

```
theta/pi  phi/pi     <Z>   cos(theta)     <X>  sin(t)cos(p)     <Y>  sin(t)sin(p)
------------------------------------------------------------------------------------
   0.000   0.000   1.000        1.000   0.000         0.000   0.000         0.000
   0.000   0.333   1.000        1.000   0.000         0.000   0.000         0.000
   0.250   0.000   0.707        0.707   0.707         0.707   0.000         0.000
   0.250   0.333   0.707        0.707   0.354         0.354   0.612         0.612
   0.500   0.000   0.000        0.000   1.000         1.000   0.000         0.000
   0.500   0.333   0.000        0.000   0.500         0.500   0.866         0.866
   0.667   0.000  -0.500       -0.500   0.866         0.866   0.000         0.000
   0.667   0.333  -0.500       -0.500   0.433         0.433   0.750         0.750
   1.000   0.000  -1.000       -1.000   0.000         0.000   0.000         0.000
   1.000   0.333  -1.000       -1.000   0.000         0.000   0.000         0.000

演算子から求めた <Z>          : 0.540302
(+1)P(0) + (-1)P(1) から      : 0.540302

ショット数による <Z> の推定（厳密値 0.540302）:
    shots    estimate     error   1/sqrt(N)
-------------------------------------------
       10      0.6000    0.0597      0.3162
      100      0.5600    0.0197      0.1000
     1000      0.5920    0.0517      0.0316
    10000      0.5422    0.0019      0.0100
   100000      0.5390    0.0013      0.0032

H = -0.8 Z + 0.3 X
  項ごとに足した <H>    : -0.210704
  行列から求めた <H>    : -0.210704
  厳密な基底エネルギー  : -0.854400
```

**着目点。** 最初の表はBlochの恒等式を項ごとに検証しています。\\(\langle Z\rangle\\) は \\(\theta\\) のみに依存し、\\(\langle X\rangle\\) と \\(\langle Y\rangle\\) が方位角を担います。最初の2行、\\(\theta = 0\\) で \\(\varphi\\) が異なる場合に注目してください。極における方位角は意味をもたず、3つの期待値はいずれも変わりません。最後のブロックは変分計算のミニチュアです。\\((\theta, \varphi) = (1.0, 0.5)\\) の状態は \\(\langle H\rangle = -0.2107\\) を与え、これは厳密な基底エネルギー \\(-0.8544\\) より上にあります — 変分原理が保証するとおりです。第3章がやることは、手で選んだ2つの角度を数値最適化器に置き換えるだけです。

* * *

## 1.5 多量子ビットとテンソル積

### 状態空間は掛け算で増える

複合量子系は連結ではなく**テンソル積**で結合します。2量子ビットでは基底が4要素になり、

\\[ |00\rangle, \quad |01\rangle, \quad |10\rangle, \quad |11\rangle \\]

一般の状態は4つの複素振幅をもちます。

\\[ |\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle \\]

\\(n\\) 量子ビットでは次元は \\(2^n\\) であり、これは1.1節に現れたのと同じ指数です。この対比は正確に述べる価値があります。「量子ビットは0と1を同時にとれる」という言い方はその要約として不十分だからです。\\(n\\) 個の古典ビットは \\(n\\) 個の数値で記述されますが、\\(n\\) 量子ビットは（規格化と大域位相を除いて）\\(2^n - 1\\) 個の複素数を必要とします。資源となっているのは*結合した*振幅の構造であり、個々の量子ビットの「決めきれなさ」ではありません。

### 本シリーズのインデックス規約

状態ベクトルは平坦な配列なので、ビット列を配列のインデックスへどう対応させるかを固定しなければなりません。**本シリーズは全体を通してbig-endian順序を採用します。**

  * 量子ビット0は文字列の**左端**のビットであり、**最上位**ビットです。
  * 基底状態 \\(|q_0 q_1 \ldots q_{n-1}\rangle\\) のインデックスは \\(\sum_{i=0}^{n-1} q_i\, 2^{\,n-1-i}\\) です。
  * 同じことですが、\\(|q_0 q_1 \ldots q_{n-1}\rangle = |q_0\rangle \otimes |q_1\rangle \otimes \cdots \otimes |q_{n-1}\rangle\\) を `np.kron` で左から右の順に組み上げます。

| ビット列 | インデックス（\\(n=3\\)） |
| --- | --- |
| \\(\lvert 000\rangle\\) | 0 |
| \\(\lvert 001\rangle\\) | 1 |
| \\(\lvert 010\rangle\\) | 2 |
| \\(\lvert 100\rangle\\) | 4 |
| \\(\lvert 101\rangle\\) | 5 |
| \\(\lvert 111\rangle\\) | 7 |

**半日を節約するための警告。** Qiskitは逆の規約 — 量子ビット0を*右端*かつ最下位ビットとするlittle-endian — を採用しているため、同じ物理状態がビット列を反転した形で表示されます。CirqとPennyLaneは本シリーズと同じbig-endianです。2つの規約はラベルの付け替えだけの違いですが、混在させると「明らかに誤りではない形で誤った」結果を生みます。2量子ビットのCNOTが別の量子ビットに作用しているように見え、Pauli文字列から組んだハミルトニアンは項が入れ替わります。フレームワーク間でコードを移植するときは、まず `ket('01')` を確認してください。

### 積状態とエンタングル状態

2つの量子ビットが独立に準備されたなら、全体の状態はテンソル積になり、

\\[ |\psi\rangle = |a\rangle \otimes |b\rangle, \qquad \alpha_{ij} = a_i b_j \\]

このとき同時確率分布は因数分解します：\\(P(ij) = P(i)P(j)\\)。この形に書け*ない*状態が**エンタングル状態**です。標準的な例はBell状態

\\[ |\Phi^{+}\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \\]

で、その周辺分布はどちらも一様 — 各量子ビット単体は最大にランダムに見える — 一方で同時の結果は完全に相関しています。2つの量子ビットに個別の状態を割り当てる方法では、これを再現できません。

きれいな判定法が**Schmidt分解**です。\\(2^n\\) 次元ベクトルを、関心のあるカットに沿って行列に整形し、特異値分解を取ります。

\\[ |\psi\rangle = \sum_{k} \lambda_k\, |u_k\rangle_A \otimes |v_k\rangle_B, \qquad \sum_k \lambda_k^2 = 1 \\]

非ゼロの \\(\lambda_k\\) の個数が**Schmidtランク**です。ランク1なら積状態、1より大きければエンタングルしています。**エンタングルメントエントロピー**

\\[ S = -\sum_k \lambda_k^2 \log_2 \lambda_k^2 \\]

がその量を測り、積状態の0からBell状態の1ビットまでの範囲をとります。

### 多体物理への橋渡し

以上のどれも、物性物理の読者にとって新しいものではありません。新しいのは記法だけです。第二量子化では、フェルミオンの状態は占有数基底で書かれ、

\\[ |n_1 n_2 \ldots n_{2M}\rangle, \qquad n_p \in \\{0, 1\\} \\]

一般の相関した状態はそのような文字列の重ね合わせ — 配置間相互作用（CI）展開 — になります。これは*文字通り*量子ビットレジスタです。各スピン軌道が1つの量子ビットで、占有か空、そしてCI係数が振幅です。第4章のJordan-Wigner変換は、この同一視をフェルミオンの反対称性が保たれるだけ丁寧に行うこと以上のものではありません。

Schmidtの描像は、古典手法が時に勝つ理由も説明します。1次元のギャップのある局所ハミルトニアンの基底状態は**面積則**に従い、カットを横切るエンタングルメントエントロピーは系のサイズとともに増大せず飽和します。したがって重要なSchmidt値はわずかで、控えめな結合次元の行列積状態で足ります。これがDMRGが鎖状系できわめて成功し、2次元系や長距離相互作用をもつ系ではそれほど成功しない理由です。量子計算機が面白いのは、まさにこの古典的な近道が使えない場所 — つまり強くエンタングルした状態に対して — です。したがって「この状態は弱くエンタングルしているか」は、量子優位性の候補問題に対して問うべき最も切れ味のよい単一の問いです。

### コード例6: テンソル積、規約、Schmidtランク

```python
import numpy as np

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def ket(bits: str) -> np.ndarray:
    """'01' -> 4次元の基底状態 |01>（ビッグエンディアン）"""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


# インデックス規約を明示的に確認する
print("n = 3 量子ビットのbig-endianインデックス規約")
print(f"{'|q0 q1 q2>':>12}{'index':>8}{'= 4 q0 + 2 q1 + q2':>22}")
print("-" * 42)
for i in range(8):
    bits = format(i, "03b")
    q0, q1, q2 = (int(b) for b in bits)
    print(f"{'|' + bits + '>':>12}{i:>8}{4*q0 + 2*q1 + q2:>22}")

# kron も同じ順序で同じベクトルを作る
print("\nket('01') == kron(|0>, |1>):",
      np.allclose(ket("01"), np.kron(ket0, ket1)))
print("ket('10') == kron(|1>, |0>):",
      np.allclose(ket("10"), np.kron(ket1, ket0)))
print("ket('01') の非ゼロ成分の位置:", int(np.argmax(np.abs(ket("01")))))
print("ket('10') の非ゼロ成分の位置:", int(np.argmax(np.abs(ket("10")))),
      " <- テンソル積の順序が意味をもつ")

# 積状態：確率が因数分解される
a = np.array([np.cos(0.3), np.sin(0.3)], dtype=complex)          # 量子ビット0
b = np.array([np.cos(1.1), np.exp(0.7j) * np.sin(1.1)], dtype=complex)  # 量子ビット1
prod = np.kron(a, b)
print("\n積状態 |a> (x) |b>")
print(f"  同時確率                  : {np.round(np.abs(prod)**2, 5)}")
print(f"  P(q0) x P(q1) の直積      : "
      f"{np.round(np.outer(np.abs(a)**2, np.abs(b)**2).ravel(), 5)}")

# Bell状態：確率は因数分解されない
bell = (ket("00") + ket("11")) / np.sqrt(2)
print("\nBell状態 (|00> + |11>)/sqrt(2)")
print(f"  同時確率                  : {np.round(np.abs(bell)**2, 5)}")
print("  量子ビット0の周辺分布      :",
      np.round(np.abs(bell.reshape(2, 2)) ** 2 @ np.ones(2), 5))
print("  量子ビット1の周辺分布      :",
      np.round(np.ones(2) @ np.abs(bell.reshape(2, 2)) ** 2, 5))
print("  周辺分布の直積             : [0.25 0.25 0.25 0.25]  <- 同時分布と一致しない")

# Schmidtランク：行列に整形して特異値を数える
print("\n量子ビット0 | 量子ビット1 のカットでのSchmidt分解")
for name, psi in [("product", prod), ("Bell", bell)]:
    s = np.linalg.svd(psi.reshape(2, 2), compute_uv=False)
    rank = int(np.sum(s > 1e-12))
    entropy = max(0.0, -sum(x * np.log2(x) for x in s ** 2 if x > 1e-12))
    print(f"  {name:<8} 特異値 {np.round(s, 4)}  "
          f"Schmidtランク {rank}  エントロピー {entropy:.4f} bit")

# 状態ベクトル側から見た指数の壁
print("\nn量子ビット状態がもつ複素振幅の個数:")
print(f"{'n':>4}{'2^n':>26}{'complex128 memory':>22}")
print("-" * 52)
for n in [1, 2, 10, 20, 30, 40, 50, 60]:
    nbytes = 2.0 ** n * 16
    unit = "B"
    for u in ["kB", "MB", "GB", "TB", "PB", "EB"]:
        if nbytes >= 1024:
            nbytes /= 1024
            unit = u
    print(f"{n:>4}{2**n:>26,d}{f'{nbytes:.1f} {unit}':>22}")
```

```
n = 3 量子ビットのbig-endianインデックス規約
  |q0 q1 q2>   index    = 4 q0 + 2 q1 + q2
------------------------------------------
       |000>       0                     0
       |001>       1                     1
       |010>       2                     2
       |011>       3                     3
       |100>       4                     4
       |101>       5                     5
       |110>       6                     6
       |111>       7                     7

ket('01') == kron(|0>, |1>): True
ket('10') == kron(|1>, |0>): True
ket('01') の非ゼロ成分の位置: 1
ket('10') の非ゼロ成分の位置: 2  <- テンソル積の順序が意味をもつ

積状態 |a> (x) |b>
  同時確率                  : [0.18778 0.72489 0.01797 0.06936]
  P(q0) x P(q1) の直積      : [0.18778 0.72489 0.01797 0.06936]

Bell状態 (|00> + |11>)/sqrt(2)
  同時確率                  : [0.5 0.  0.  0.5]
  量子ビット0の周辺分布      : [0.5 0.5]
  量子ビット1の周辺分布      : [0.5 0.5]
  周辺分布の直積             : [0.25 0.25 0.25 0.25]  <- 同時分布と一致しない

量子ビット0 | 量子ビット1 のカットでのSchmidt分解
  product  特異値 [1. 0.]  Schmidtランク 1  エントロピー 0.0000 bit
  Bell     特異値 [0.7071 0.7071]  Schmidtランク 2  エントロピー 1.0000 bit

n量子ビット状態がもつ複素振幅の個数:
   n                       2^n     complex128 memory
----------------------------------------------------
   1                         2                32.0 B
   2                         4                64.0 B
  10                     1,024               16.0 kB
  20                 1,048,576               16.0 MB
  30             1,073,741,824               16.0 GB
  40         1,099,511,627,776               16.0 TB
  50     1,125,899,906,842,624               16.0 PB
  60 1,152,921,504,606,846,976               16.0 EB
```

**着目点。** 最初の表は飾りではなく規約の検査です。インデックスの列と算術の列が行ごとに一致しており、これは `index = 4*q0 + 2*q1 + q2` という主張そのものです。積状態のブロックでは同時分布が周辺分布の直積と印字されたすべての桁で一致し、Bellのブロックでは — 2つの周辺分布が個別には何の特徴もないにもかかわらず — 明確に一致しません。Schmidtの行はこの区別を定量的かつ基底に依存しない形にします。ランク1でエントロピー0に対し、ランク2でちょうど1ビットです。`reshape(2, 2)` と `svd` の組み合わせは覚えておいてください。3行で書け、任意の量子ビット数の任意の2分割に一般化し、シミュレートした状態がどれだけエンタングルしたかを見る標準的な診断法です。

* * *

## 1.6 NumPyでミニシミュレータを作る

### 設計上の決定

本シリーズで用いるシミュレータは意図的に小さく — 完成形でも99行（第2章のコード例2）です — 以下の4つの決定は今後変更しません。

| 決定事項 | 選択 | 理由 |
| --- | --- | --- |
| 表現 | `complex128` の密な `numpy` 配列 | 厳密で見通しがよく、20量子ビット程度までは十分 |
| 量子ビット順序 | big-endian（量子ビット0が左端・最上位） | 本文および1.5節の記法と一致する |
| インタフェース | クラスを使わない素の関数 | 各関数を独立に試験でき、そのままコピーできる |
| 純粋性 | 新しい状態を返し、引数を書き換えない | 最適化ループの中で安全に再利用できる |

最後の点は強調に値します。第3章ではこれらの関数が最適化器の内部で数千回呼ばれます。引数を黙って書き換える関数は、数回の反復を経てから初めて現れる不具合を生み、それは最も発見コストの高い種類の不具合です。

ここで作るのは3つの関数です。残り — ゲート行列、`apply_gate`、`cnot`、`expval` — は第2章で登場します。ゲートの定数は下記にすでに含めてあるので、第2章に至った時点でモジュールが完成します。

### コード例7: `ket`、`probs`、`sample`

```python
"""ミニ状態ベクトルシミュレータ その1: ket / probs / sample。

規約（本シリーズの全章で変更しない）:
    量子ビット0 = 左端のビット = 最上位ビット
    |q0 q1 ... q_{n-1}> のインデックス = sum_i q_i 2^(n-1-i)
"""
import numpy as np

# --- 2x2 の構成要素（第2章以降で使用） -------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def ket(bits: str) -> np.ndarray:
    """'01' -> 4次元の基底状態 |01>（ビッグエンディアン）"""
    n = len(bits)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


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


if __name__ == "__main__":
    # 1. 基底状態
    print("ket('0')   =", ket("0"))
    print("ket('01')  =", ket("01"))
    print("ket('101') の振幅の位置は",
          int(np.argmax(np.abs(ket("101")))), "= 0b101")

    # 2. 重ね合わせはケットを足して規格化するだけ
    bell = (ket("00") + ket("11")) / np.sqrt(2)
    w3 = (ket("100") + ket("010") + ket("001")) / np.sqrt(3)
    print(f"\nBell状態のノルム : {np.linalg.norm(bell):.6f}")
    print(f"W状態のノルム    : {np.linalg.norm(w3):.6f}")

    # 3. probs
    print("\nprobs((|00> + |11>)/sqrt(2)):")
    for i, p in enumerate(probs(bell)):
        print(f"  |{format(i, '02b')}>  {p:.4f}")
    print(f"確率の総和 = {probs(bell).sum():.6f}")

    # 4. sample - 有限統計。seedを与えれば再現可能
    counts = sample(bell, shots=4000, seed=42)
    print("\nsample(bell, shots=4000, seed=42):", counts)
    print("相対頻度:",
          {k: round(v / 4000, 4) for k, v in counts.items()})

    counts_w = sample(w3, shots=6000, seed=7)
    print("\nsample(W状態, shots=6000, seed=7):", counts_w)
    print("期待値は各1/3 ->",
          {k: round(v / 6000, 4) for k, v in counts_w.items()})

    # 5. 不均等な重ね合わせ：振幅は確率ではない
    psi = (0.6 * ket("00") + 0.8j * ket("11"))
    print(f"\npsi = 0.6|00> + 0.8i|11>,  ノルム = {np.linalg.norm(psi):.4f}")
    print("probs:", np.round(probs(psi), 4))
    print("sample(psi, 10000, seed=1):", sample(psi, 10000, seed=1))

    # 6. 自分のコードにも残しておきたい健全性チェック
    assert np.isclose(probs(bell).sum(), 1.0)
    assert set(sample(bell, 100, seed=0)) <= {"00", "01", "10", "11"}
    print("\nすべてのアサーションを通過しました")
```

```
ket('0')   = [1.+0.j 0.+0.j]
ket('01')  = [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
ket('101') の振幅の位置は 5 = 0b101

Bell状態のノルム : 1.000000
W状態のノルム    : 1.000000

probs((|00> + |11>)/sqrt(2)):
  |00>  0.5000
  |01>  0.0000
  |10>  0.0000
  |11>  0.5000
確率の総和 = 1.000000

sample(bell, shots=4000, seed=42): {'00': 2023, '11': 1977}
相対頻度: {'00': 0.5058, '11': 0.4943}

sample(W状態, shots=6000, seed=7): {'001': 2068, '010': 1933, '100': 1999}
期待値は各1/3 -> {'001': 0.3447, '010': 0.3222, '100': 0.3332}

psi = 0.6|00> + 0.8i|11>,  ノルム = 1.0000
probs: [0.36 0.   0.   0.64]
sample(psi, 10000, seed=1): {'00': 3563, '11': 6437}

すべてのアサーションを通過しました
```

**着目点。** `ket` は4行で、big-endian規約のすべてが `int(bits, 2)` という1つの呼び出しに収まっています — Python自身による、2進文字列の最上位ビット先頭としての読み方です。`sample` は任意の `seed` を受け取るので、本シリーズに印字されたすべての数値は再現可能です。実機にはseedはなく、再実行すれば別のカウントが得られます。最後の例は覚えておくべきものです。振幅 \\(0.6\\) と \\(0.8i\\) は確率 \\(0.36\\) と \\(0.64\\) を与え、10 000ショットは3563と6437を返します。2番目の振幅にかかる虚数単位はカウントにはまったく現れません。相対位相は同じ基底での測定には決して現れず、さらにゲートを作用させた後の干渉にのみ現れます。

### コード例8: 期待値1つにショットは何回必要か

状態ベクトルは厳密な振幅を保持しますが、実機が与えるのはサンプルだけです。この例では統計誤差のスケーリングを実測し、それを第3章を支配する数値へ換算します。

```python
import numpy as np
import matplotlib.pyplot as plt

Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_from_angles(theta, phi=0.0):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def estimate_Z(psi, shots, rng):
    """shots回の射影測定から <Z> を推定する。"""
    p = np.abs(psi) ** 2
    outcomes = rng.choice([+1.0, -1.0], size=shots, p=p)
    return outcomes.mean()


rng = np.random.default_rng(2026)
theta = 1.0
psi = state_from_angles(theta)
exact = float(np.real(np.vdot(psi, Z @ psi)))
variance = 1.0 - exact ** 2          # Var(Z) = <Z^2> - <Z>^2 = 1 - <Z>^2

print(f"状態: theta = 1.0 rad,  <Z> = {exact:.6f},  Var(Z) = {variance:.6f}")
print(f"\n{'shots':>9}{'RMS error':>12}{'sqrt(Var/N)':>14}{'ratio':>8}")
print("-" * 43)

shot_list = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000]
rms = []
for shots in shot_list:
    errs = [estimate_Z(psi, shots, rng) - exact for _ in range(400)]
    r = float(np.sqrt(np.mean(np.square(errs))))
    rms.append(r)
    predicted = np.sqrt(variance / shots)
    print(f"{shots:>9}{r:>12.5f}{predicted:>14.5f}{r/predicted:>8.3f}")

slope, intercept = np.polyfit(np.log(shot_list), np.log(rms), 1)
print(f"\n両対数プロットの傾き = {slope:.3f}   （理論値: -0.5）")

# 1つのPauli項を化学的精度で測るには何ショット必要か
target = 1.6e-3        # Hartree、いわゆる化学的精度（1 kcal/mol）
needed = variance / target ** 2
print(f"\n1つのPauli項の標準誤差を {target:.1e} にするのに必要なショット数:"
      f" {needed:,.0f}")
print("分子ハミルトニアンは数百項をもち、VQEの最適化は数百回の反復を要する。")
print("この1つの数値が、第3章で測定コストを設計上の第一級の制約として扱う理由である。")

# --- 可視化 ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(shot_list, rms, "o-", color="tab:purple", label="measured RMS error")
ax.loglog(shot_list, np.sqrt(variance / np.array(shot_list)), "k--",
          label=r"$\sqrt{\mathrm{Var}(Z)/N}$")
ax.set_xlabel("number of shots $N$")
ax.set_ylabel(r"RMS error of $\langle Z \rangle$")
ax.set_title("Shot noise scales as $1/\\sqrt{N}$")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()
```

```
状態: theta = 1.0 rad,  <Z> = 0.540302,  Var(Z) = 0.708073

    shots   RMS error   sqrt(Var/N)   ratio
-------------------------------------------
       10     0.26891       0.26610   1.011
       30     0.16542       0.15363   1.077
      100     0.08422       0.08415   1.001
      300     0.04961       0.04858   1.021
     1000     0.02626       0.02661   0.987
     3000     0.01614       0.01536   1.050
    10000     0.00840       0.00841   0.998
    30000     0.00493       0.00486   1.016

両対数プロットの傾き = -0.503   （理論値: -0.5）

1つのPauli項の標準誤差を 1.6e-03 にするのに必要なショット数: 276,591
分子ハミルトニアンは数百項をもち、VQEの最適化は数百回の反復を要する。
この1つの数値が、第3章で測定コストを設計上の第一級の制約として扱う理由である。
```

**着目点。** `ratio` の列は3桁半にわたって1から数パーセント以内にとどまり、フィットした傾きは予測 \\(-1/2\\) に対して \\(-0.503\\) です。誤差は隠れたバイアスのない純粋なショットノイズです。その帰結は容赦がありません。誤差を半分にするには測定回数が4倍必要なので、*1つの*Pauli項を化学的精度で測るための \\(2.8 \times 10^5\\) ショットは、現実的なハミルトニアンと現実的な最適化を考慮すると \\(10^{10}\\) のオーダーになります。アルゴリズムに誤りがあるわけではありません。\\(1/\sqrt{N}\\) は単にBorn則の代価です。第3章では可換な項をまとめるなど定数を小さくする方法を論じ、第5章ではノイズがこの定数をさらに悪化させる理由を説明します。

### いま何ができ、次に何が来るか

| 構成要素 | 本章終了時点の状態 |
| --- | --- |
| `ket(bits)` | ✅ 実装済み |
| `probs(state)` | ✅ 実装済み |
| `sample(state, shots, seed=None)` | ✅ 実装済み |
| `I2, X, Y, Z, H, S, T` | ✅ 定義済み（第2章から使用） |
| `rx, ry, rz(theta)` | 第2章 |
| `apply_gate(state, U, targets, n)` | 第2章 |
| `cnot(state, control, target, n)` | 第2章 |
| `expval(state, pauli, coeff_map=None)` | 第2章 |

ここまではすべて*状態の準備と読み出し*です。まったく欠けているのは**ダイナミクス**です。ある状態を別の状態に変える手段が、まだありません。それが次章の主題であり、必要なNumPyコードは約60行です。

* * *

## 演習

#### 演習1: 規格化、位相、Blochベクトル

次の状態を考えます。

\\[ |\psi\rangle = \frac{1-i}{2}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle \\]

  1. この状態が規格化されていることを確認しなさい。
  2. \\(P(0)\\) と \\(P(1)\\) を求めなさい。
  3. 2つの振幅の間の相対位相と、Bloch角 \\((\theta, \varphi)\\) を求めなさい。
  4. Blochベクトルを与え、球面上のどの名前つき状態に最も近いかを述べなさい。

<details>
<summary>解答</summary>
<p><strong>1.</strong> \(\left|\frac{1-i}{2}\right|^2 = \frac{1^2+1^2}{4} = \frac{1}{2}\) かつ \(\left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}\)。和が1なので規格化されています。</p>
<p><strong>2.</strong> \(P(0) = P(1) = 0.5\)。この状態はBloch球の赤道上にあります。</p>
<p><strong>3.</strong> 相対位相は \(\arg(\beta) - \arg(\alpha) = 0 - (-\pi/4) = \pi/4\) なので \(\varphi = \pi/4\)。また \(|\alpha| = \cos(\theta/2) = 1/\sqrt{2}\) より \(\theta = \pi/2\) です。</p>
<p><strong>4.</strong> \(\mathbf{r} = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta) = (0.7071, 0.7071, 0)\) — 赤道上、\(|+\rangle\)（\(+x\)）と \(|{+}i\rangle\)（\(+y\)）のちょうど中間です。数値で確認します。</p>

```python
import numpy as np
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
psi = np.array([(1 - 1j) / 2, 1 / np.sqrt(2)], dtype=complex)
print(round(np.real(np.vdot(psi, psi)), 12))           # 1.0
print(np.abs(psi) ** 2)                                # [0.5 0.5]
print((np.angle(psi[1]) - np.angle(psi[0])) / np.pi)   # 0.25  -> phi = pi/4
print(np.round([np.real(np.vdot(psi, P @ psi)) for P in (X, Y, Z)], 6))
# [0.707107 0.707107 0.      ]
```

<p>与えられた振幅の大域位相はゼロではありませんが、Blochベクトルはそれに影響されないことに注意してください。入ってきたのは位相の<em>差</em>だけです。</p>

</details>

#### 演習2: 角度から期待値へ

\\(\theta = 2\pi/3\\)、\\(\varphi = \pi/6\\) の状態 \\(|\psi(\theta,\varphi)\rangle\\) をとります。

  1. 2つの振幅を数値で書きなさい。
  2. \\(P(0)\\) と \\(P(1)\\) を求めなさい。
  3. \\(\langle X\rangle\\)、\\(\langle Y\rangle\\)、\\(\langle Z\rangle\\) を求め、Blochの公式と照合しなさい。
  4. \\(|\mathbf{r}| = 1\\) を確認し、この状態に対する \\(Z\\) 測定の分散を述べなさい。

<details>
<summary>解答</summary>
<p><strong>1.</strong> \(\cos(\theta/2) = \cos(\pi/3) = 0.5\)、\(e^{i\pi/6}\sin(\pi/3) = 0.8660\,e^{i\pi/6} = 0.75 + 0.4330i\) なので \(|\psi\rangle = 0.5|0\rangle + (0.75 + 0.4330i)|1\rangle\) です。</p>
<p><strong>2.</strong> \(P(0) = 0.25\)、\(P(1) = 0.75\)。</p>
<p><strong>3.</strong> \(\langle Z\rangle = \cos\theta = -0.5\)、\(\langle X\rangle = \sin\theta\cos\varphi = 0.8660 \times 0.8660 = 0.75\)、\(\langle Y\rangle = \sin\theta\sin\varphi = 0.8660 \times 0.5 = 0.4330\)。</p>
<p><strong>4.</strong> \(|\mathbf{r}|^2 = 0.5625 + 0.1875 + 0.25 = 1\)。分散は \(1 - \langle Z\rangle^2 = 1 - 0.25 = 0.75\) なので、\(\langle Z\rangle\) を \(\pm 0.01\) の精度で推定するには約 \(0.75/10^{-4} = 7500\) ショットが必要です。</p>

```python
import numpy as np
theta, phi = 2 * np.pi / 3, np.pi / 6
psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
print(np.round(psi, 4))              # [0.5 +0.j    0.75+0.433j]
print(np.round(np.abs(psi)**2, 6))   # [0.25 0.75]
# <X>, <Y>, <Z> = 0.75, 0.433013, -0.5 ; |r| = 1.0
```

</details>

#### 演習3: インデックスの記帳

本シリーズのbig-endian規約を用いて答えなさい。

  1. \\(n = 4\\) のとき、\\(|1011\rangle\\) はどの配列インデックスを占めますか。
  2. \\(n = 4\\) のとき、インデックス6に対応するビット列は何ですか。
  3. \\(\frac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle)\\) で非ゼロになるインデックスはどれですか。また4つの量子ビットが4つのスピン軌道だとすると、この状態は何を記述していますか。
  4. 同じビット列 \\(1011\\) をQiskitのようなlittle-endianのフレームワークに読み込ませた場合、どのインデックスに載りますか。またそれはどのような不具合を生みますか。

<details>
<summary>解答</summary>
<p><strong>1.</strong> \(1011_2 = 8 + 0 + 2 + 1 = 11\)。</p>
<p><strong>2.</strong> \(6 = 4 + 2\) なのでビット列は \(0110\) です。</p>
<p><strong>3.</strong> \(0011_2 = 3\) と \(1100_2 = 12\)。量子ビットをスピン軌道の占有数として読むと、これは2つの配置の等重率の重ね合わせです。一方は後ろ2つの軌道が占有され、他方は前2つが占有されています — まさに2配置（多参照）波動関数の構造であり、単一のSlater行列式が破綻する状況そのものです。</p>
<p><strong>4.</strong> little-endianは同じ文字列を逆順に読むので、\(1011\) は11ではなく \(1101_2 = 13\) になります。厄介なのはその失敗の仕方です。エラーは出ず、状態は規格化されたままで、確率の和も1のままです。変わるのは各振幅が<em>どの物理的な量子ビット</em>を指すかであり、その結果CNOTが別のターゲットに作用しているように見え、Pauli文字列で組んだハミルトニアンは静かに項が入れ替わります。コードがフレームワークの境界を越えるときは、まず <code>ket('01')</code> を確認してください。</p>

```python
print(int("1011", 2))          # 11   big-endian
print(format(6, "04b"))        # 0110
print(int("1011"[::-1], 2))    # 13   little-endianとして読んだ場合
```

</details>

#### 演習4: 周辺分布、収縮、エンタングルメント

次の2量子ビット状態を考えます。

\\[ |\psi\rangle = \frac{1}{\sqrt{6}}\left(|00\rangle + |01\rangle + 2|11\rangle\right) \\]

  1. 規格化を確認し、4つの同時確率を列挙しなさい。
  2. 各量子ビットの周辺分布を求めなさい。
  3. 量子ビット0を測定して1が得られた後の状態は何ですか。それはどの確率で起こりますか。
  4. この状態はエンタングルしていますか。カットを横切るSchmidt係数とエンタングルメントエントロピーを求めなさい。

<details>
<summary>解答</summary>
<p><strong>1.</strong> 振幅は \((1, 1, 0, 2)/\sqrt{6}\) で、\((1 + 1 + 0 + 4)/6 = 1\)。同時確率は \(P(00) = P(01) = 1/6\)、\(P(10) = 0\)、\(P(11) = 4/6 = 2/3\) です。</p>
<p><strong>2.</strong> 量子ビット0：\(P(q_0 = 0) = 1/6 + 1/6 = 1/3\)、\(P(q_0 = 1) = 0 + 2/3 = 2/3\)。量子ビット1：\(P(q_1 = 0) = 1/6 + 0 = 1/6\)、\(P(q_1 = 1) = 1/6 + 2/3 = 5/6\)。</p>
<p><strong>3.</strong> 射影を生き残るのは \(|11\rangle\) だけなので、収縮後の状態はちょうど \(|11\rangle\) であり、その結果が起こる確率は \(2/3\) です。この測定が量子ビット1まで決めてしまったことに注意してください。それが相関であり、エンタングルメントの徴です。</p>
<p><strong>4.</strong> 行列 \(\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & 1 \\ 0 & 2\end{pmatrix}\) に整形すると特異値は \(0.9342\) と \(0.3568\) です。どちらも非ゼロなのでSchmidtランクは2、すなわちエンタングルしています。ただし部分的です。係数の2乗は \(0.8727\) と \(0.1273\) で、\(S = 0.5500\) ビット、Bell状態の1ビットよりかなり小さい値です。</p>

```python
import numpy as np
psi = np.array([1, 1, 0, 2], dtype=complex) / np.sqrt(6)
M = psi.reshape(2, 2)
print(np.round(np.sum(np.abs(M)**2, axis=1), 6))   # [0.333333 0.666667] 量子ビット0
print(np.round(np.sum(np.abs(M)**2, axis=0), 6))   # [0.166667 0.833333] 量子ビット1
s = np.linalg.svd(M, compute_uv=False)
print(np.round(s, 6))                              # [0.934172 0.356822]
print(round(-sum(x * np.log2(x) for x in s**2), 6))  # 0.550048 bit
```

</details>

#### 演習5: 資源の見積もり

6-31G基底の水分子（空間軌道13個、電子10個）の計算に量子計算機が役立つかどうかを問われたとします。

  1. スピン軌道1つを量子ビット1つに割り当てる写像で、この問題には何量子ビット必要ですか。またfull CIの次元はいくらですか。
  2. CIベクトルはノートPCのメモリに収まりますか。40量子ビットの状態ベクトルはどうですか。
  3. 量子ビットハミルトニアンがPauli項200個をもち、各項に \\(10^{-3}\\) Hartreeの標準誤差が必要で、最適化に300反復かかるとします。\\(\operatorname{Var}(P_j) \le 1\\) として総ショット数を見積もりなさい。
  4. 繰り返しレートが毎秒 \\(10^4\\) ショットのとき、それにはどれだけの時間がかかりますか。この結果から、研究の努力をどこに向けるべきだと言えますか。

<details>
<summary>解答</summary>
<p><strong>1.</strong> 空間軌道13個はスピン軌道26個なので<strong>26量子ビット</strong>です。full CIの次元は \(\binom{13}{5}^2 = 1287^2 = 1\,656\,369 \approx 1.7 \times 10^6\)。</p>
<p><strong>2.</strong> CIベクトルは \(1.66 \times 10^6 \times 8\) バイト \(\approx 12.6\) MB — ごく小さく、だからこそこの計算は古典的に日常的に行われており、実演の対象としては不適切です。40量子ビットの状態ベクトルは \(2^{40} \times 16\) バイト = <strong>16 TB</strong> で、これはノートPCではなくスーパーコンピュータの話です。</p>
<p><strong>3.</strong> \(\operatorname{Var} \le 1\) とすると、1項を \(\sigma = 10^{-3}\) で測るのに \(1/\sigma^2 = 10^6\) ショット。したがって \(10^6 \times 200 \times 300 = 6 \times 10^{10}\) ショットです。</p>
<p><strong>4.</strong> \(6\times10^{10}/10^4 = 6 \times 10^6\) 秒 \(\approx\) <strong>69日</strong>の連続測定 — ノートPCが数秒で扱う分子に対して、です。レートを毎秒 \(10^6\) ショットに上げれば約0.7日になります。だからこそ繰り返しレート、項のグルーピング、ショット配分の戦略は、工学的な細部ではなく活発な研究領域なのです。誠実な読み方はこうです。近未来の実用性を制限しているのは、量子ビット数よりもはるかに測定コストとノイズである。</p>

```python
from math import comb
dim = comb(13, 5) ** 2
print(dim, round(dim * 8 / 1024**2, 2))      # 1656369 12.64   (MB)
print(2**40 * 16 / 1024**4)                  # 16.0           (TB)
total = (1 / 1e-3**2) * 200 * 300
print(f"{total:.2e}", round(total / 1e4 / 86400, 1))   # 6.00e+10 69.4  （日）
```

</details>

* * *

## まとめ

### 要点

**1\. 動機はスケーリングの議論である**

  * 厳密な多電子波動関数は次元 \\(\binom{M}{N_\alpha}\binom{M}{N_\beta}\\) の空間に住み、最も重要な強相関系については古典的な記憶容量を超えます。
  * 多項式スケーリングの古典手法はいずれも構造に関する仮定を対価としており、強相関はその仮定が破れる場所です。
  * 量子ビットレジスタは同じ状態を、軌道数に*線形*な量子ビット数で保持します。それが — 一般的な意味での速度ではなく — 材料科学における量子コンピューティングの物理的根拠です。

**2\. 量子ビットは規格化された複素ベクトルである**

  * \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\)、\\(|\alpha|^2 + |\beta|^2 = 1\\)。振幅は確率ではありません。
  * 大域位相は非物理的、相対位相は物理的であり、後者が干渉を可能にします。
  * 内積は `np.dot` ではなく `np.vdot` で実装され、配列はすべて `dtype=complex` にすべきです。

**3\. Bloch球は1量子ビットの完全な描像である**

  * 角度2つで足ります：\\(|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\varphi}\sin(\theta/2)|1\rangle\\)、そして \\(\mathbf{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)\\) はすべての純粋状態で長さ1です。
  * 直交する状態は対極にあり、スピン1/2のBlochベクトルは物理的な磁気モーメントそのものです。
  * この描像は多量子ビットに拡張されません。Bell状態の各量子ビットは自分の球の中心に座ります。

**4\. 測定は振幅が数値になる場所である**

  * Born則 \\(P(k) = |\psi_k|^2\\)、収縮後の状態を与える射影の公理、1ショットあたり1つのビット列。
  * \\(\langle Z\rangle = P(0) - P(1)\\) であり、一般のハミルトニアンはPauli分解の後に項ごとに測定します。
  * \\(\operatorname{Var}(Z) = 1 - \langle Z\rangle^2\\) なので、統計誤差は \\(1/\sqrt{N}\\) でしか減りません。

**5\. 多量子ビットはテンソル積である**

  * \\(\dim = 2^n\\)。本シリーズ全体でインデックス規約 \\(\sum_i q_i 2^{\,n-1-i}\\)（big-endian）を用います — Qiskitとは逆です。
  * 積状態は因数分解し、エンタングル状態はしません。カットを横切るSchmidtランクがその判定法です。
  * 第二量子化の占有数基底は*そのまま*量子ビットレジスタであり、それが第4章の写像を可能にします。面積則的なエンタングルメントは、DMRGが古典的に機能する理由です。

**6\. シミュレータは小さく、そして自分のものである**

  * `ket`、`probs`、`sample` は約20行で、任意の状態の準備と読み出しに十分です。
  * 再現性は明示的なseedから、厳密さは `complex128` から、安全性は引数を書き換えないことから得られます。
  * まだ状態を発展させる手段はありません。ダイナミクスが欠けている要素です。

**実務上の含意**

  * コードがフレームワークの境界を越えるときは、まずインデックス規約を検査してください。
  * 変分計算の結果を信じる前にショット予算を見積もってください。ショットが足りないまま収束した最適化は、ノイズに収束しています。
  * 量子優位性の主張に対しては次を問うてください。対象の状態は強くエンタングルしているか、そして答えを得るには何回の測定が必要か。

次章ではダイナミクスを追加します。ユニタリ発展はSchrödinger方程式を有限個の可逆なゲートに変え、1量子ビットゲートはBlochベクトルの回転になり、CNOTは本章で手書きするしかなかったエンタングルメントを生成します。そしてミニシミュレータは `apply_gate`、`cnot`、`expval` を獲得し、その時点で本シリーズの残りのどの回路も実行できるようになります。

[← シリーズトップ](<index.html>) [第2章: 量子ゲートと回路 →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
