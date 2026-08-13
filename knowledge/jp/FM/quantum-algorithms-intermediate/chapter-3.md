---
title: "第3章: Shorのアルゴリズム"
chapter_title: "第3章: Shorのアルゴリズム"
subtitle: 周期発見としての因数分解、2つの整数のエンドツーエンド実行、そして同じ回路が暗号規模で要するもの
reading_time: 45-50分
difficulty: 中級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-algorithms-intermediate/chapter-3.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子アルゴリズム（中級）](<index.html>) > 第3章

この分野を有名にしたアルゴリズムであり、本コースで高速化が超多項式的かつ誰も異論を唱えない唯一の場所です。$n$ ビット整数の因数分解は、知られている最良の古典手法で $\exp\left(O(n^{1/3}(\log n)^{2/3})\right)$ 演算、Shorのアルゴリズムで $O(n^3)$ ゲートです。この差は定数倍ではなく、条件付きの優位でもなく、データ書き込みの仮定に依存するものでもありません。第1章の二次高速化には1ページぶんの留保が付きましたが、こちらには付きません。

その代わりに付くのはサイズです。本章で15を因数分解する回路は13量子ビットを使い、2048ビットのRSA法を因数分解する回路は、標準的な誤り訂正の仮定の下で $10^7$ 台の物理量子ビットと $10^9$ 台のToffoliゲートを必要とします。3.4節では明示した前提からその見積りを通します。誠実な立場は2つのよく見る誤りの間にあります。1つは「量子コンピュータはRSAを破る」と言うことです。破りません。どの装置もこの回路をそのサイズで走らせられず、距離は工学的な改良ではなく桁で測られます。もう1つは、その距離からアルゴリズムに意味がないと結論することで、こちらも同じくらい誤りです。数学は決着しており、格子暗号への移行はまさにこの理由で既に仕様化され進行中であり、そして因数分解回路は、量子計算が同じ複雑性クラスの別の書き方に過ぎないのではないことを示す最も清潔な存在証明です。

技術的な内容は短くて済みます。第2章がほとんどの仕事を済ませたからです。因数分解は $N$ を法とする無作為な整数の乗法的位数を求めることに還元され、位数発見は特定のユニタリに対する位相推定であり、位相推定は制御べき乗の後の逆QFTです。ここで新しいのは量子回路の両側にある数論 — 入る側の還元と、出る側の連分数 — と、回路の高価な部分がフーリエ変換ではまったくなく剰余算術であるという観察だけです。

## 学習目標

本章を終えると、次のことができるようになります。

  * 因数分解から位数発見への還元を、偶数 $N$ と完全べき乗を片付ける古典的な近道も含めて実行し、無作為に選んだ底が失敗する2通りを述べられる
  * 相異なる奇素因数を2個持つ $N$ に対して底ごとの失敗確率が高々 $1/2$ である理由を説明し、$N = 15$ と $N = 21$ で列挙によって限界を検証できる
  * 位数発見ユニタリ $U_a\lvert y \rangle = \lvert ay \bmod N \rangle$ を構成し、その固有値が $e^{2\pi i s/r}$ であることを同定し、$\lvert 1 \rangle$ が正しい入力である理由を説明できる
  * 回路のコストを正しく見積れる。モジュラー指数演算が $O(n^3)$ ゲート、QFTが $O(n^2)$ なので、フーリエ変換が安い側である
  * 測定された計数レジスタ値から連分数で位数を回復し、近似分数が $r$ ではなく $r/\gcd(s,r)$ を返しうる理由を説明できる
  * $N = 15$ と $N = 21$ でアルゴリズム全体をシミュレータ上で走らせ、1回あたりの厳密な成功確率を計算し、標本抽出でそれを再現できる
  * 法のサイズを、明示した仮定から論理量子ビット数・Toffoli数・符号距離・物理量子ビット数に変換し、格子暗号が標準的な対応である理由と、Grover対策には対称鍵長を倍にすれば足りる理由を説明できる

* * *

## 3.1 因数分解から位数発見へ

### 還元

$N$ を完全べき乗でない奇合成数とし、$1 < a < N$ かつ $\gcd(a, N) = 1$ の整数 $a$ を選びます。$a$ の $N$ を法とする**乗法的位数**は

$$ a^r \equiv 1 \pmod N $$

を満たす最小の $r > 0$ です。$r$ が既知で偶数だとします。すると $x = a^{r/2}$ は $x^2 \equiv 1 \pmod N$ を満たすので

$$ (x-1)(x+1) \equiv 0 \pmod N $$

つまり $N$ がこの積を割ります。どちらの因子も $N$ で割り切れないなら — すなわち $x \not\equiv \pm 1 \pmod N$ なら — $N$ は各因子と自明でない因数を共有せねばならず、

$$ \gcd\left(a^{r/2} - 1,\, N\right), \qquad \gcd\left(a^{r/2} + 1,\, N\right) $$

が $N$ の真の約数となり、Euclidの互除法でマイクロ秒で計算できます。これが還元の全部です。因数分解はどの時点でも直接攻撃されていません。量子コンピュータが供給するのは整数1個、$r$ です。

この構成が失敗するのは正確に2通りで、どちらも導出から見えます。$r$ が**奇数**なら $a^{r/2}$ は整数でなく計算すべきものがありません。$x \equiv -1 \pmod N$ なら2番目のgcdは $N$、1番目は1なので、どちらの約数も自明です。$x \equiv +1$ は起こりえないことにも注意してください。それは $r/2$ が $r$ より小さい位数であることを意味してしまいます。

### 無作為な底がふつう機能する理由

標準的な定理はこうです。$N$ が相異なる奇素因数を $k \ge 2$ 個もち、$a$ を $N$ と互いに素な整数から一様に引くとき、

$$ \Pr\left[r \text{ が偶数かつ } a^{r/2} \not\equiv -1 \pmod N\right] \; \ge \; 1 - \frac{1}{2^{k-1}} \; \ge \; \frac{1}{2} $$

証明は各素数べきを法とする位数の2進付値についての中国剰余定理的な帳簿付けで、どの教科書にもあります。ここで重要なのは結論の形です。失敗は誤った答ではなく検出された失敗であり — $\gcd$ が1か $N$ を返し、これは古典的に直ちに検査されます — だからアルゴリズムはただ新しい底を引きます。繰り返しは底ごとの確率 $1/2$ を $m$ 個の底の後に失敗確率 $2^{-m}$ に変え、$m = 30$ で既に無視できます。

$N$ の2つのクラスは量子回路に到達しません。**偶数の $N$** は一目で因数分解できます。**完全べき乗** $N = c^b$ は $\lfloor \log_2 N \rfloor$ 個の整数根を試せば見つかり、だから定理の「相異なる素因数2個」という仮定は制限になりません。除外される場合は易しい場合です。入口にはもう一つ棚ぼたがあります。無作為に引いた $a$ が $\gcd(a, N) > 1$ を満たしてしまえば、そのgcdが既に因数で、回路は一度も走りません。本章の極小の $N$ ではこれが無視できない頻度で起き、例6がそれを定量化します。小規模な実演の読み方に対する警告でもあります。

### 連分数

量子サブルーチンは $r$ を返しません。$t$ ビットレジスタから整数 $k$ を返し、確率 $4/\pi^2 \approx 0.405$ 以上でその $k$ は $s/r$ に最も近い格子点であり、そのとき未知の $s \in \lbrace 0, 1, \ldots, r-1 \rbrace$ に対して

$$ \left\lvert \frac{k}{2^t} - \frac{s}{r} \right\rvert \le \frac{1}{2^{t+1}} $$

を満たします。この限界は最良の結果についての言明で、どの測定にも当てはまるわけではありません。もっと離れた $k$ も十分起こりえます。そのときは連分数が古典的検査 $a^{q} \equiv 1$ を通らない分母を返し、回路をもう一度回すだけです。演習2がこの限界を条件付きの形で述べています。さて、その精度で分かっている実数から $s/r$ を既約形で回復するのは、古典的な答をもつ古典的な問題です。$2^t \ge N^2$ なら $r < N$ のすべてに対して $1/2^{t+1} < 1/(2r^2)$ となり、$k/2^t$ にそれだけ近い分母 $N$ 未満の有理数は一意で、**連分数の近似分数**の中に現れます。近似分数は漸化式

$$ \frac{h_i}{q_i}, \qquad h_i = c_i h_{i-1} + h_{i-2}, \qquad q_i = c_i q_{i-1} + q_{i-2} $$

で生成されます。$c_i$ は $k$ と $2^t$ に対するEuclidの互除法の部分商です。全体が整数演算で、ビット演算 $O(n^3)$ で済みます。これが計数レジスタが $n$ ではなく $t = 2n + 1$ 量子ビットである理由です。追加の2倍が有理数復元を一意にします。

微妙な点が1つ残り、それが以下のコードの後処理が候補を複数試す理由です。近似分数は $s/r$ を*既約形で*返すので、出てくるのは $r/\gcd(s,r)$ です。$\gcd(s,r) > 1$ のとき分母は $r$ の真の約数であり、古典的な検査 $a^{q} \equiv 1 \pmod N$ が直ちにこれを検出します。候補を2倍・3倍すれば剰余べき乗数回の代価でよくある場合が回復でき、回路を2回走らせて2つの分母の最小公倍数を取るのがもう一つの標準的な対処です。

### Code Example 1: 道具箱の再掲

本章は[量子コンピューティング入門 第2章](<../quantum-computing-introduction/chapter-2.html>)の状態ベクトルシミュレータと、本コース第2章のFourier機構を必要とします。どちらもここに再掲します — 本章が使う関数のみを逐語で、ただしモジュールのdocstringだけは2つの出典を1ファイルにまとめるために書き直したもので、他章と同様に原文（英語）のまま置いています — ので、以下のどこも自分で復元しなければならないファイルに依存しません。`shorlib.py` として保存してください。規約は入門コースから受け継いだまま変えていません。**ビッグエンディアン**で、量子ビット0がケットの左端かつ振幅添字の最上位ビットなので、$t$ 量子ビットの計数レジスタから読み出される整数は $k = \sum_j q_j 2^{t-1-j}$ です。

```python
"""Chapter 3 toolbox. Save as shorlib.py and start every example with
`from shorlib import *`.

Part 1 is the state-vector simulator of Introduction to Quantum Computing,
Chapter 2 -- the functions this chapter needs, verbatim. Part 2 is the Fourier
machinery of Chapter 2 of this course, also verbatim.
"""
import numpy as np

# ---- 第1部: ミニシミュレータ（逐語） ----------------------------------------
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


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


def probs(state):
    """Born則による全 2^n 通りの確率"""
    return np.abs(state) ** 2


# ---- 第2部: 第2章のFourier機構（逐語） --------------------------------------
SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def cphase(theta):
    """量子ビット対に作用する制御位相ゲート diag(1, 1, 1, exp(i theta)) です。"""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def iqft(state, qubits, n):
    """逆QFTです。同じゲートを逆順に、位相を複素共役にして掛けます。"""
    m = len(qubits)
    for j in range(m // 2):
        state = apply_gate(state, SWAP4, [qubits[j], qubits[m - 1 - j]], n)
    for j in reversed(range(m)):
        for k in reversed(range(j + 1, m)):
            state = apply_gate(state, cphase(-np.pi / 2 ** (k - j)),
                               [qubits[k], qubits[j]], n)
        state = apply_gate(state, H, [qubits[j]], n)
    return state


def controlled(U):
    """U のブロック対角な制御版です。制御量子ビットは先頭の1個です。"""
    d = U.shape[0]
    C = np.eye(2 * d, dtype=complex)
    C[d:, d:] = U
    return C
```

シミュレータの9個の関数のうち必要なのは4個だけで、第2章の変換の対も逆変換だけです。位数発見は順方向QFTを一度も掛けないからです。残りの仕事は第2章 Code Example 4 の `controlled` ヘルパがします。任意の $2^k \times 2^k$ ユニタリを $2^{k+1} \times 2^{k+1}$ にし、それを `apply_gate` がレジスタ内のどこにある制御量子ビット1個と標的 $k$ 個にも作用させます。

### Code Example 2: 古典側半分

3.1節の内容をすべて計算したものです。数個を標本抽出するのではなく互いに素な底を全部表にする意味は、それによって失敗の様式が数え上げ可能になり、定理の $1/2$ の限界が受け入れる主張ではなく検査できる数値になることです。

```python
"""第3章 Code Example 2: Shorのアルゴリズムの古典側半分です。
Code Example 1 の続き（同一セッション）です。"""

import numpy as np
from math import gcd


def order(a, N):
    """a^r = 1 (mod N) を満たす最小の r > 0 を総当たりで求めます。古典 O(N) です。"""
    x, r = a % N, 1
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def factors_from_order(a, r, N):
    """還元の最後の段 gcd(a^(r/2) -+ 1, N) です。"""
    if r % 2 != 0:
        return None, "r is odd"
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None, "a^(r/2) = -1 mod N"
    f1, f2 = gcd(x - 1, N), gcd(x + 1, N)
    for f in (f1, f2):
        if 1 < f < N:
            return (f, N // f), "ok"
    return None, "only trivial gcds"


for N in (15, 21):
    coprime = [a for a in range(2, N) if gcd(a, N) == 1]
    print(f"N = {N}: every base coprime to N, and what it yields")
    print("-" * 74)
    print(f"  {'a':>4}{'r = ord_N(a)':>14}{'r even':>8}"
          f"{'a^(r/2) mod N':>15}{'factors':>12}{'verdict':>20}")
    good = 0
    for a in coprime:
        r = order(a, N)
        fac, why = factors_from_order(a, r, N)
        if fac:
            good += 1
        half = pow(a, r // 2, N) if r % 2 == 0 else None
        print(f"  {a:>4}{r:>14}{'yes' if r % 2 == 0 else 'no':>8}"
              f"{('-' if half is None else str(half)):>15}"
              f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>12}"
              f"{why:>20}")
    print(f"\n  {good} of {len(coprime)} bases succeed "
          f"({good/len(coprime):.3f}); the theorem guarantees at least 1/2 "
          f"for\n  an N with two distinct odd prime factors.\n")

print("The cases the quantum part never sees")
print("-" * 74)


def classical_shortcuts(N):
    """Shorのアルゴリズムが測定前に古典処理で片付けてしまう場合の全部です。"""
    if N % 2 == 0:
        return f"even: 2 x {N//2}"
    for b in range(2, int(np.log2(N)) + 1):
        root = round(N ** (1.0 / b))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** b == N:
                return f"perfect power: {cand}^{b}"
    return None


for N in (15, 21, 16, 27, 35, 2 ** 10, 91):
    sc = classical_shortcuts(N)
    print(f"  N = {N:>5}: " + (sc if sc else "needs order finding"))
print("  A random base can also be lucky: if gcd(a, N) > 1 the factor is free.")
print(f"  For N = 15 that happens for a in "
      f"{[a for a in range(2, 15) if 1 < gcd(a, 15) < 15]}, "
      f"{len([a for a in range(2,15) if 1 < gcd(a,15) < 15])}/13 of the bases.")

print("\nContinued fractions: recovering r from a noisy s/r")
print("-" * 74)


def convergents(num, den, max_den):
    """分母が max_den 以下となる num/den の連分数近似分数です。"""
    out = []
    h2, h1, k2, k1 = 0, 1, 1, 0
    n, d = num, den
    while d:
        q = n // d
        h, k = q * h1 + h2, q * k1 + k2
        if k > max_den:
            break
        out.append((h, k))
        h2, h1, k2, k1 = h1, h, k1, k
        n, d = d, n - q * d
    return out


for num, den, max_den, label in [(1365, 4096, 21, "k = 1365, t = 12, N = 21"),
                                 (683, 2048, 21, "k = 683, t = 11, N = 21"),
                                 (192, 256, 15, "k = 192, t = 8, N = 15"),
                                 (3, 8, 15, "k = 3, t = 3, N = 15")]:
    cs = convergents(num, den, max_den)
    print(f"  {label:<26} {num}/{den} = {num/den:.6f}")
    print("     convergents: " + "  ".join(f"{h}/{k}" for h, k in cs))
print("  The last convergent with denominator <= N is the candidate for r; "
      "the check\n  a^r = 1 mod N is classical and cheap, so a wrong guess "
      "costs nothing.")
```

```text
N = 15: every base coprime to N, and what it yields
--------------------------------------------------------------------------
     a  r = ord_N(a)  r even  a^(r/2) mod N     factors             verdict
     2             4     yes              4         3x5                  ok
     4             2     yes              4         3x5                  ok
     7             4     yes              4         3x5                  ok
     8             4     yes              4         3x5                  ok
    11             2     yes             11         5x3                  ok
    13             4     yes              4         3x5                  ok
    14             2     yes             14           -  a^(r/2) = -1 mod N

  6 of 7 bases succeed (0.857); the theorem guarantees at least 1/2 for
  an N with two distinct odd prime factors.

N = 21: every base coprime to N, and what it yields
--------------------------------------------------------------------------
     a  r = ord_N(a)  r even  a^(r/2) mod N     factors             verdict
     2             6     yes              8         7x3                  ok
     4             3      no              -           -            r is odd
     5             6     yes             20           -  a^(r/2) = -1 mod N
     8             2     yes              8         7x3                  ok
    10             6     yes             13         3x7                  ok
    11             6     yes              8         7x3                  ok
    13             2     yes             13         3x7                  ok
    16             3      no              -           -            r is odd
    17             6     yes             20           -  a^(r/2) = -1 mod N
    19             6     yes             13         3x7                  ok
    20             2     yes             20           -  a^(r/2) = -1 mod N

  6 of 11 bases succeed (0.545); the theorem guarantees at least 1/2 for
  an N with two distinct odd prime factors.

The cases the quantum part never sees
--------------------------------------------------------------------------
  N =    15: needs order finding
  N =    21: needs order finding
  N =    16: even: 2 x 8
  N =    27: perfect power: 3^3
  N =    35: needs order finding
  N =  1024: even: 2 x 512
  N =    91: needs order finding
  A random base can also be lucky: if gcd(a, N) > 1 the factor is free.
  For N = 15 that happens for a in [3, 5, 6, 9, 10, 12], 6/13 of the bases.

Continued fractions: recovering r from a noisy s/r
--------------------------------------------------------------------------
  k = 1365, t = 12, N = 21   1365/4096 = 0.333252
     convergents: 0/1  1/3
  k = 683, t = 11, N = 21    683/2048 = 0.333496
     convergents: 0/1  1/2  1/3
  k = 192, t = 8, N = 15     192/256 = 0.750000
     convergents: 0/1  1/1  3/4
  k = 3, t = 3, N = 15       3/8 = 0.375000
     convergents: 0/1  1/2  1/3  3/8
  The last convergent with denominator <= N is the candidate for r; the check
  a^r = 1 mod N is classical and cheap, so a wrong guess costs nothing.
```

**注目すべき点。** $N = 15$ では互いに素な7個の底のうち6個、$N = 21$ では11個のうち6個が成功し、どちらも保証された $1/2$ を上回ります。ただしどちらの数え上げも $1 < a < N$ の範囲で、定理の標本空間が含み必ず失敗する $a = 1$ を除いています。$\mathbb{Z}_N^\ast$ 上では $N = 21$ の成功率はちょうど $6/12$ で、限界は上回られるのではなく達成されます。演習1が $N = 33$ について同じことを計算します。そして $N = 15$ の唯一の失敗様式は $a = 14 \equiv -1$ で、位数が2、$a^{r/2}$ が構成上 $-1$ です。$N = 21$ では両方の失敗様式が現れます。$a = 4$ と $a = 16$ は位数3、$a = 5, 17, 20$ は $a^{r/2} \equiv -1$ に当たります。表全体として読むと、どの失敗も*底*の性質であり、古典算術2行で検出され、どれもアルゴリズムの性質ではありません。

連分数のブロックは後処理の予告です。$k = 1365$、$t = 12$ の行が研究に値します。$1365/4096 = 0.333252$ で近似分数は $1/3$ で止まりますが、3は21を法とする2の位数ではありません。位数は6です。真の $s/r$ は $2/6$ で、これが $1/3$ に約されて $\gcd(s,r) = 2$ が失われました。対処は上で述べたものです。$k = 683$ の行は別経路を示します。ここでは近似分数が $1/2$ を通って $1/3$ に至るので、同じ2倍で6が回復します。

* * *

## 3.2 位数発見の回路

### ユニタリとその固有値

$\gcd(a, N) = 1$ に対して、$n = \lceil \log_2 N \rceil$ 量子ビットレジスタ上の演算子

$$ U_a \lvert y \rangle = \lvert ay \bmod N \rangle $$

を定義し、$y \ge N$ の $2^n - N$ 個の基底状態では恒等写像として延長します。$a$ 倍写像が $N$ と互いに素な剰余を置換するのでユニタリです。実際 $U_a$ は置換行列そのものであり、$U_a^r = I$ です。

その固有ベクトルは1の軌道上のFourierモードです。$s \in \lbrace 0, \ldots, r-1 \rbrace$ に対して

$$ \lvert u_s \rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1} e^{-2\pi i sj/r}\, \lvert a^j \bmod N \rangle, \qquad U_a \lvert u_s \rangle = e^{2\pi i s/r}\, \lvert u_s \rangle $$

なので $U_a$ の固有位相はすべて $1/r$ の倍数です。したがって $U_a$ に対する位相推定は $s/r$ の推定値を返し、これが着想の全部です。残る問題は $\lvert u_s \rangle$ を用意するには $r$ を知る必要があることです。その解決がこのアルゴリズムが優雅である理由です。計算基底状態 $\lvert 1 \rangle$ はそれら全部の一様重ね合わせであり、

$$ \lvert 1 \rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1} \lvert u_s \rangle $$

なので $\lvert 1 \rangle$ を位相推定に入れると一様乱数の $s \in \lbrace 0, \ldots, r-1 \rbrace$ が返ります。第2章 Code Example 4 が示したとおり、固有ベクトルの重ね合わせである入力は各固有位相をその重みで返すからです。結果 $s = 0$ は無用で確率 $1/r$ で起こり、それ以外の結果はすべて $r$ についての情報を担います。

### コストの所在

回路は $j = 0, \ldots, t-1$ に対して制御 $U_a^{2^j}$ を必要としますが、素朴な読み — $U_a$ の作用 $2^t - 1$ 回 — は実際の作り方ではありません。$a^{2^j} \bmod N$ が*古典*計算だからです。各制御べき乗は、事前計算した定数による制御剰余乗算1回です。したがってそれが $t = 2n+1$ 回あり、各々が教科書的算術で $O(n^2)$ 基本ゲートの可逆剰余乗算器なので、合計 $O(n^3)$ です。それに対し $t$ 量子ビットの逆QFTは $t(t+1)/2 = O(n^2)$ です。

この比が要点で、標語として述べる価値があります。**Shorのアルゴリズムは、末尾にフーリエ変換をホチキス留めしたモジュラー指数演算回路です。** 名前をもらった部分が漸近的に無料な部分なのです。因数分解に関するまともな資源見積りはすべて算術の見積り — どの加算器、どの乗算器、補助量子ビット何個、どれだけウィンドウ化や事前計算ができるか — であり、QFTは主要項に現れません。これはまた*近似*QFTが普遍的に使われる理由でもあります。その誤差は他のすべてに比べて無視できます。第2章で既に代価を払った実装上の注意を1つ。ここで $t$ 量子ビットの計数レジスタをコヒーレントに保つ必要はありません。2.3節の補助1量子ビットの反復的位相推定が $U_a$ にそのまま適用でき、$t = 2n+1$ 本の計数量子ビットを1本の補助量子ビットと $t$ ラウンドの測定・フィードバックで置き換えられます。文献にあるコンパクトな因数分解回路が $3n$ ではなく $2n$ 程度の量子ビット数を挙げるのは、この道筋によります。

| | モジュラー指数演算 | 逆QFT |
| --- | --- | --- |
| 役割 | $a^k \bmod N$ を作業レジスタに書く | 計数レジスタから周期を読む |
| 個数 | 制御剰余乗算 $2n+1$ 回 | 回転 $t(t+1)/2$ 個、SWAP $\lfloor t/2 \rfloor$ 個 |
| ゲート数 | $O(n^3)$ | $O(n^2)$ |
| 非Clifford成分 | 加算器中のToffoli — 支配的コスト | 小角回転、ほぼ落とせる |
| 補助量子ビット | $O(n)$ の作業レジスタ | なし |

### Code Example 3: 位数発見の回路

```python
"""第3章 Code Example 3: 位数発見の回路と、そのコストの所在です。
Code Example 2 の続き（同一セッション）です。"""

import numpy as np


def modmul_unitary(a, N, n_work):
    """|y> -> |a y mod N> の置換行列です。y >= N では恒等写像です。

    gcd(a, N) = 1 のとき a 倍写像は Z_N の全単射なので、置換になります。
    実機では行列ではなく、加算器から組んだ可逆な剰余乗算器です。
    """
    d = 2 ** n_work
    U = np.zeros((d, d), dtype=complex)
    for y in range(d):
        U[(a * y) % N if y < N else y, y] = 1.0
    return U


def order_finding_state(a, N, t, n_work):
    """制御モジュラー指数演算と逆QFTを終えた後の全状態です。

    計数量子ビットは 0..t-1（qubit 0 が最上位）、作業レジスタは
    t..t+n_work-1 で |1> に初期化します。a^(2^j) mod N は事前に古典計算
    できるので、必要な制御剰余乗算は t 回だけです。
    """
    n = t + n_work
    state = np.kron(ket('0' * t), ket(format(1, f'0{n_work}b')))
    for j in range(t):
        state = apply_gate(state, H, [j], n)
    for j in range(t):
        a_pow = pow(a, 2 ** (t - 1 - j), N)
        state = apply_gate(state, controlled(modmul_unitary(a_pow, N, n_work)),
                           [j] + list(range(t, n)), n)
    return iqft(state, list(range(t)), n)


print("The modular multiplication operator is a permutation")
print("-" * 74)
for a, N, n_work in [(7, 15, 4), (2, 21, 5)]:
    U = modmul_unitary(a, N, n_work)
    r = order(a, N)
    Ur = np.linalg.matrix_power(U, r)
    print(f"  a = {a}, N = {N}, {n_work} work qubits, dimension {2**n_work}")
    print(f"    unitary: max |U^dag U - I| = "
          f"{np.max(np.abs(U.conj().T @ U - np.eye(2**n_work))):.1e}")
    print(f"    order r = {r}:  max |U^r - I| = "
          f"{np.max(np.abs(Ur - np.eye(2**n_work))):.1e}")
    orbit = [1]
    while True:
        nxt = (orbit[-1] * a) % N
        if nxt == 1:
            break
        orbit.append(nxt)
    print(f"    orbit of |1>: " + " -> ".join(str(y) for y in orbit)
          + " -> 1")
    print(f"    basis states with y >= N: {2**n_work - N} left untouched, "
          f"which is what keeps U unitary")

print("\nWhere the gates actually go")
print("-" * 74)
print("  Counting register t = 2n+1 bits, work register n = ceil(log2 N) bits.")
print(f"  {'N':>10}{'n':>6}{'t':>7}{'qubits':>9}{'ctrl mod-mults':>16}"
      f"{'~gates in them':>17}{'~QFT gates':>12}")
for N in [15, 21, 2 ** 16 + 1, 2 ** 64 + 1, 2 ** 1024 + 1, 2 ** 2048 + 1]:
    n_bits = N.bit_length()
    t = 2 * n_bits + 1
    label = f"{N}" if N < 1000 else f"~2^{n_bits-1}"
    print(f"  {label:>10}{n_bits:>6}{t:>7}{t + n_bits:>9}{t:>16}"
          f"{t * n_bits ** 2:>17d}{t * (t + 1) // 2:>12d}")
print("  The modular arithmetic outweighs the QFT by a factor ~n at every "
      "size:\n  Shor's algorithm is a modular-exponentiation circuit with a "
      "Fourier transform\n  stapled to the end, not the other way round.")

print("\nThe state just before the inverse QFT, N = 15, a = 7, t = 4")
print("-" * 74)
a, N, n_work, t = 7, 15, 4, 4
n = t + n_work
st = np.kron(ket('0' * t), ket(format(1, f'0{n_work}b')))
for j in range(t):
    st = apply_gate(st, H, [j], n)
for j in range(t):
    st = apply_gate(st, controlled(modmul_unitary(pow(a, 2 ** (t-1-j), N),
                                                 N, n_work)),
                    [j] + list(range(t, n)), n)
work_probs = probs(st).reshape(2 ** t, -1).sum(axis=0)
print("  work-register marginal (only the orbit of 1 is populated):")
print("   " + "  ".join(f"|{y}>: {work_probs[y]:.4f}"
                        for y in np.flatnonzero(work_probs > 1e-9)))
print(f"  each of the {int(round(1/work_probs.max()))} orbit states carries "
      f"probability 1/r, and the counting register\n  conditioned on any one "
      "of them is periodic with period r -- which is what the\n  inverse QFT "
      "then reads.")
```

```text
The modular multiplication operator is a permutation
--------------------------------------------------------------------------
  a = 7, N = 15, 4 work qubits, dimension 16
    unitary: max |U^dag U - I| = 0.0e+00
    order r = 4:  max |U^r - I| = 0.0e+00
    orbit of |1>: 1 -> 7 -> 4 -> 13 -> 1
    basis states with y >= N: 1 left untouched, which is what keeps U unitary
  a = 2, N = 21, 5 work qubits, dimension 32
    unitary: max |U^dag U - I| = 0.0e+00
    order r = 6:  max |U^r - I| = 0.0e+00
    orbit of |1>: 1 -> 2 -> 4 -> 8 -> 16 -> 11 -> 1
    basis states with y >= N: 11 left untouched, which is what keeps U unitary

Where the gates actually go
--------------------------------------------------------------------------
  Counting register t = 2n+1 bits, work register n = ceil(log2 N) bits.
           N     n      t   qubits  ctrl mod-mults   ~gates in them  ~QFT gates
          15     4      9       13               9              144          45
          21     5     11       16              11              275          66
       ~2^16    17     35       52              35            10115         630
       ~2^64    65    131      196             131           553475        8646
     ~2^1024  1025   2051     3076            2051       2154831875     2104326
     ~2^2048  2049   4099     6148            4099      17209245699     8402950
  The modular arithmetic outweighs the QFT by a factor ~n at every size:
  Shor's algorithm is a modular-exponentiation circuit with a Fourier transform
  stapled to the end, not the other way round.

The state just before the inverse QFT, N = 15, a = 7, t = 4
--------------------------------------------------------------------------
  work-register marginal (only the orbit of 1 is populated):
   |1>: 0.2500  |4>: 0.2500  |7>: 0.2500  |13>: 0.2500
  each of the 4 orbit states carries probability 1/r, and the counting register
  conditioned on any one of them is periodic with period r -- which is what the
  inverse QFT then reads.
```

**注目すべき点。** $U_a$ は誤差厳密ゼロで置換行列であり、$U_a^r = I$ も厳密に成り立ちます。浮動小数点で計算した整数の恒等式に残差がないのは、置換行列の積が丸めを生まないからです。印字された軌道は $N$ を法とする列 $1, a, a^2, \ldots$ で長さはちょうど $r$、$\lbrace 0, \ldots, N-1 \rbrace$ の外の状態は不動点です。これが次元が $N$ でないレジスタ上で演算子がユニタリであり続ける仕組みです。

計上の表は3.2節の主張を数値にしたものです。$n = 2049$ ビットで剰余算術が $1.7\times10^{10}$ ゲート、QFTが $8.4\times10^6$ ゲート、比は2000です。「量子フーリエ変換がRSAを破る」という語り方はすべて強調点が間違っています。

最後のブロックは位相推定が実際に作用する状態を示します。制御モジュラー指数演算の後、作業レジスタはちょうど軌道の $r$ 個の要素に台をもち各々確率 $1/r$ であり、そのどれか1つに条件付けると計数レジスタは周期 $r$ の周期的重ね合わせを保持します。まさに第2章 Code Example 3 の状態です。次に逆QFTが周期を読み、オフセット — 軌道のどの要素が測られたか — は例3が予測したとおり厳密に捨てられます。

* * *

## 3.3 15と21をエンドツーエンドで

### Code Example 4: $N = 15$、周期がレジスタを割る場合

$N = 15$ は伝統的な実演対象で、代表性を欠くほど親切です。どの位数も2か4でどちらも $2^t$ を割るので、出力分布は漏れなしで厳密にピークが立ちます。だからこそ、答が目で見える場合に対して後処理を検査する正しい場所になります。

```python
"""第3章 Code Example 4: N = 15 のエンドツーエンドの因数分解です。
Code Example 3 の続き（同一セッション）です。"""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd


def postprocess(k, t, a, N, max_multiple=3):
    """測定された計数レジスタ値 k を (r, 因数) か None に変換します。

    連分数は k/2^t ~ s/r の分母の候補を与えます。近似分数が返すのは r では
    なく r/gcd(s, r) なので、各候補の小さな倍数も試します。これで頻出する
    gcd = 2 や 3 を回復できます。回数固定の古典的な剰余べき乗であって探索
    ではありません。s = 0 は r の情報を持たないので近似分数 0/1 は飛ばします。
    """
    for _, r_cand in convergents(k, 2 ** t, N):
        if r_cand < 2:
            continue
        for m in range(1, max_multiple + 1):
            r = m * r_cand
            if r <= N and pow(a, r, N) == 1:
                return r, factors_from_order(a, r, N)[0]
    return None, None


a, N, n_work, t = 7, 15, 4, 8
print(f"Order finding for a = {a}, N = {N}: t = {t} counting qubits, "
      f"{t + n_work} qubits total")
print("-" * 76)
state = order_finding_state(a, N, t, n_work)
p = probs(state).reshape(2 ** t, -1).sum(axis=1)
r_true = order(a, N)
print(f"  true order r = {r_true}, so the ideal peaks sit at "
      f"k = j * 2^t/r = j * {2**t // r_true}")
print(f"  {'k':>6}{'p(k)':>10}{'k/2^t':>10}{'CF convergents':>26}"
      f"{'r found':>9}{'factors':>10}")
for k in np.flatnonzero(p > 1e-9):
    r, fac = postprocess(int(k), t, a, N)
    cs = "  ".join(f"{h}/{q}" for h, q in convergents(int(k), 2 ** t, N))
    print(f"  {k:>6}{p[k]:>10.4f}{k/2**t:>10.4f}{cs:>26}"
          f"{('-' if r is None else r):>9}"
          f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>10}")
print(f"  total probability on those {int((p > 1e-9).sum())} outcomes: "
      f"{p[p > 1e-9].sum():.6f}")

print("\nExact success probability, summed over the whole distribution")
print("-" * 76)
succ = sum(p[k] for k in range(2 ** t)
           if postprocess(k, t, a, N)[1] is not None)
print(f"  P(a nontrivial factor of {N} on one run with a = {a}) = {succ:.6f}")
print(f"  P(failure)  = {1 - succ:.6f}, and p(k = 0) alone is {p[0]:.6f}")

print("\nSampling the circuit, 2000 shots")
print("-" * 76)
rng = np.random.default_rng(20260813)
counts = rng.multinomial(2000, p)
found = {}
for k in np.flatnonzero(counts):
    r, fac = postprocess(int(k), t, a, N)
    key = "no factor" if fac is None else f"{fac[0]} x {fac[1]}"
    found[key] = found.get(key, 0) + int(counts[k])
    print(f"  k = {k:>3} seen {counts[k]:>5} times  ->  "
          f"r = {r if r else '-':>2}, {key}")
print("  tally: " + ",  ".join(f"{k}: {v}" for k, v in sorted(found.items())))

print("\nEvery usable base for N = 15, at t = 8")
print("-" * 76)
print(f"  {'a':>4}{'r':>4}{'peaks':>8}{'P(success)':>13}{'factors found':>16}")
for a_ in [a_ for a_ in range(2, N) if gcd(a_, N) == 1]:
    st = order_finding_state(a_, N, t, n_work)
    pp = probs(st).reshape(2 ** t, -1).sum(axis=1)
    s = 0.0
    facs = set()
    for k in range(2 ** t):
        r, fac = postprocess(k, t, a_, N)
        if fac:
            s += pp[k]
            facs.add(tuple(sorted(fac)))
    print(f"  {a_:>4}{order(a_, N):>4}{int((pp > 1e-9).sum()):>8}{s:>13.6f}"
          f"{(', '.join(f'{f[0]}x{f[1]}' for f in facs) or '-'):>16}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(np.arange(2 ** t), p, width=1.0, color="tab:blue")
for j in range(r_true):
    ax[0].axvline(j * 2 ** t / r_true, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("measured k"); ax[0].set_ylabel("probability")
ax[0].set_title(f"N = 15, a = 7, t = 8: r = {r_true} divides $2^t$")
p14 = probs(order_finding_state(14, N, t, n_work)).reshape(2 ** t, -1).sum(axis=1)
ax[1].bar(np.arange(2 ** t), p14, width=1.0, color="tab:red")
ax[1].set_xlabel("measured k"); ax[1].set_ylabel("probability")
ax[1].set_title("a = 14: r = 2, clean peaks, no factor")
plt.tight_layout()
plt.show()
```

```text
Order finding for a = 7, N = 15: t = 8 counting qubits, 12 qubits total
----------------------------------------------------------------------------
  true order r = 4, so the ideal peaks sit at k = j * 2^t/r = j * 64
       k      p(k)     k/2^t            CF convergents  r found   factors
       0    0.2500    0.0000                       0/1        -         -
      64    0.2500    0.2500                  0/1  1/4        4       3x5
     128    0.2500    0.5000                  0/1  1/2        4       3x5
     192    0.2500    0.7500             0/1  1/1  3/4        4       3x5
  total probability on those 4 outcomes: 1.000000

Exact success probability, summed over the whole distribution
----------------------------------------------------------------------------
  P(a nontrivial factor of 15 on one run with a = 7) = 0.750000
  P(failure)  = 0.250000, and p(k = 0) alone is 0.250000

Sampling the circuit, 2000 shots
----------------------------------------------------------------------------
  k =   0 seen   483 times  ->  r =  -, no factor
  k =  64 seen   530 times  ->  r =  4, 3 x 5
  k = 128 seen   505 times  ->  r =  4, 3 x 5
  k = 192 seen   482 times  ->  r =  4, 3 x 5
  tally: 3 x 5: 1517,  no factor: 483

Every usable base for N = 15, at t = 8
----------------------------------------------------------------------------
     a   r   peaks   P(success)   factors found
     2   4       4     0.750000             3x5
     4   2       2     0.500000             3x5
     7   4       4     0.750000             3x5
     8   4       4     0.750000             3x5
    11   2       2     0.500000             3x5
    13   4       4     0.750000             3x5
    14   2       2     0.000000               -
```

**注目すべき点。** 結果は4通り、各々厳密に確率 $1/4$、$k = 0, 64, 128, 192$ — $2^t/r = 64$ の倍数です。うち3つが因数分解を与え、1つ $k = 0$ は与えません。3.2節の議論で確率 $1/r$ の無用な $s = 0$ の結果です。したがって1回あたりの厳密な成功確率は $3/4$ で、2000ショットの標本は $1517/2000 = 0.759$ を返します。

中間の行は連分数が本当の仕事をしているところです。$k = 128$ での近似分数は $1/4$ ではなく $1/2$ です。真の $s/r$ が $2/4$ で約分されました。候補2は検査 $7^2 \equiv 4 \pmod{15}$ で落ち、倍にした候補4が通り、因数が出ます。3.1節の $\gcd(s,r) > 1$ の場合で、この底では全実行の4分の1で起きます。

底の表が例2との輪を閉じます。$r = 4$ の底はすべて成功確率 $3/4$、$r = 2$ の底はすべて $1/2$ です。ピークが2個しかなくその一方が $k = 0$ だからです。そして $a = 14$ は厳密にゼロです。位数は偶数ですが $a^{r/2}$ が $-1$ なので、測定されたどの $k$ も役に立ちません。ここで回路が失敗しているのではありません — 完璧に $r = 2$ を返しており、*還元*にそれの使い道がないのです。

### Code Example 5: $N = 21$、割らない場合

21は一般の振る舞いを示す最小の実例です。21を法とする2の位数は6、6はどの2のべきも割らず、出力分布は漏れます。

```python
"""第3章 Code Example 5: r が 2^t を割らない場合の N = 21 の因数分解です。
Code Example 4 の続き（同一セッション）です。"""

import numpy as np
import matplotlib.pyplot as plt
from math import gcd

a, N, n_work = 2, 21, 5
r_true = order(a, N)
print(f"N = {N}, a = {a}: true order r = {r_true}, and 2^t/r is never an "
      f"integer")
print("-" * 78)
print(f"  {'t':>4}{'qubits':>8}{'peak k':>9}{'k/2^t':>10}{'s/r ideal':>11}"
      f"{'p(peak)':>10}{'P(success)':>12}{'P(k=0)':>9}")
dists = {}
for t in range(6, 14):
    st = order_finding_state(a, N, t, n_work)
    p = probs(st).reshape(2 ** t, -1).sum(axis=1)
    dists[t] = p
    k = int(np.argmax(p[1:]) + 1)
    s_over_r = round(k / 2 ** t * r_true) / r_true
    succ = sum(p[j] for j in range(2 ** t)
               if postprocess(j, t, a, N)[1] is not None)
    print(f"  {t:>4}{t + n_work:>8}{k:>9}{k/2**t:>10.6f}{s_over_r:>11.6f}"
          f"{p[k]:>10.4f}{succ:>12.6f}{p[0]:>9.4f}")

t = 11
p = dists[t]
print(f"\nThe six peaks at t = {t}, and what each one postprocesses to")
print("-" * 78)
top = np.sort(np.argsort(p)[::-1][:6])
print(f"  {'k':>6}{'p(k)':>9}{'k/2^t':>10}{'nearest s/r':>13}"
      f"{'CF convergents':>26}{'r':>4}{'factors':>10}")
for k in top:
    r, fac = postprocess(int(k), t, a, N)
    cs = "  ".join(f"{h}/{q}" for h, q in convergents(int(k), 2 ** t, N))
    s = round(k / 2 ** t * r_true)
    print(f"  {k:>6}{p[k]:>9.4f}{k/2**t:>10.6f}{f'{s}/{r_true}':>13}"
          f"{cs:>26}{('-' if r is None else r):>4}"
          f"{('-' if fac is None else f'{fac[0]}x{fac[1]}'):>10}")
print(f"  the six peaks hold {p[top].sum():.4f} of the probability; the rest "
      f"leaks into\n  the {2**t - 6} remaining bins because r = {r_true} does "
      f"not divide 2^{t} = {2**t}")

print(f"\nSampling the t = {t} circuit, 2000 shots")
print("-" * 78)
rng = np.random.default_rng(20260813)
counts = rng.multinomial(2000, p)
tally = {}
for k in np.flatnonzero(counts):
    _, fac = postprocess(int(k), t, a, N)
    key = "no factor" if fac is None else f"{fac[0]} x {fac[1]}"
    tally[key] = tally.get(key, 0) + int(counts[k])
print(f"  distinct k values observed: {int((counts > 0).sum())}")
print("  " + ",  ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
print(f"  measured success rate {tally.get('7 x 3', 0)/2000:.4f} against the "
      f"exact "
      f"{sum(p[j] for j in range(2**t) if postprocess(j, t, a, N)[1]):.4f}")

print("\nAll usable bases for N = 21 at t = 11")
print("-" * 78)
print(f"  {'a':>4}{'r':>4}{'P(success)':>13}{'factors found':>16}"
      f"{'why it can fail':>26}")
for a_ in [x for x in range(2, N) if gcd(x, N) == 1]:
    st = order_finding_state(a_, N, t, n_work)
    pp = probs(st).reshape(2 ** t, -1).sum(axis=1)
    s = 0.0
    facs = set()
    for k in range(2 ** t):
        rr, fac = postprocess(k, t, a_, N)
        if fac:
            s += pp[k]
            facs.add(tuple(sorted(fac)))
    r_ = order(a_, N)
    why = ("r odd" if r_ % 2 else
           ("a^(r/2) = -1" if pow(a_, r_ // 2, N) == N - 1 else
            "only k = 0 and CF misses"))
    print(f"  {a_:>4}{r_:>4}{s:>13.6f}"
          f"{(', '.join(f'{f[0]}x{f[1]}' for f in facs) or '-'):>16}{why:>26}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t_, style in [(8, "-"), (11, "-")]:
    ax[0].plot(np.arange(2 ** t_) / 2 ** t_, dists[t_], style, lw=1,
               label=f"t = {t_}")
for s in range(r_true):
    ax[0].axvline(s / r_true, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("$k/2^t$"); ax[0].set_ylabel("probability")
ax[0].set_title("N = 21, a = 2: peaks near $s/6$")
ax[0].legend(fontsize=8)
ax[1].plot(list(dists), [sum(dists[t_][j] for j in range(2 ** t_)
                             if postprocess(j, t_, a, N)[1] is not None)
                         for t_ in dists], "o-", color="tab:green")
ax[1].set_xlabel("counting qubits $t$"); ax[1].set_ylabel("P(success)")
ax[1].set_ylim(0, 1)
ax[1].set_title("Success probability of one run")
plt.tight_layout()
plt.show()
```

```text
N = 21, a = 2: true order r = 6, and 2^t/r is never an integer
------------------------------------------------------------------------------
     t  qubits   peak k     k/2^t  s/r ideal   p(peak)  P(success)   P(k=0)
     6      11       32  0.500000   0.500000    0.1670    0.787013   0.1670
     7      12       64  0.500000   0.500000    0.1667    0.815174   0.1667
     8      13      128  0.500000   0.500000    0.1667    0.823615   0.1667
     9      14      256  0.500000   0.500000    0.1667    0.828564   0.1667
    10      15      512  0.500000   0.500000    0.1667    0.830918   0.1667
    11      16     1024  0.500000   0.500000    0.1667    0.832118   0.1667
    12      17     2048  0.500000   0.500000    0.1667    0.832721   0.1667
    13      18     4096  0.500000   0.500000    0.1667    0.833029   0.1667

The six peaks at t = 11, and what each one postprocesses to
------------------------------------------------------------------------------
       k     p(k)     k/2^t  nearest s/r            CF convergents   r   factors
       0   0.1667  0.000000          0/6                       0/1   -         -
     341   0.1140  0.166504          1/6                  0/1  1/6   6       7x3
     683   0.1140  0.333496          2/6             0/1  1/2  1/3   6       7x3
    1024   0.1667  0.500000          3/6                  0/1  1/2   6       7x3
    1365   0.1140  0.666504          4/6        0/1  1/1  1/2  2/3   6       7x3
    1707   0.1140  0.833496          5/6             0/1  1/1  5/6   6       7x3
  the six peaks hold 0.7893 of the probability; the rest leaks into
  the 2042 remaining bins because r = 6 does not divide 2^11 = 2048

Sampling the t = 11 circuit, 2000 shots
------------------------------------------------------------------------------
  distinct k values observed: 78
  7 x 3: 1677,  no factor: 323
  measured success rate 0.8385 against the exact 0.8321

All usable bases for N = 21 at t = 11
------------------------------------------------------------------------------
     a   r   P(success)   factors found           why it can fail
     2   6     0.832118             3x7  only k = 0 and CF misses
     4   3     0.000000               -                     r odd
     5   6     0.000000               -              a^(r/2) = -1
     8   2     0.500000             3x7  only k = 0 and CF misses
    10   6     0.832118             3x7  only k = 0 and CF misses
    11   6     0.832118             3x7  only k = 0 and CF misses
    13   2     0.500000             3x7  only k = 0 and CF misses
    16   3     0.000000               -                     r odd
    17   6     0.000000               -              a^(r/2) = -1
    19   6     0.832118             3x7  only k = 0 and CF misses
    20   2     0.000000               -              a^(r/2) = -1
```

**注目すべき点。** ピークは $k/2^t = 0.166504$ などに来ます。$s/6$ の近くで決してその上ではなく、周期がレジスタ長を割らない場合について第2章 Code Example 3 が予測したとおりです。$t = 11$ で最良6ビンに乗る確率は $78.9\%$ しかなく、標本実行は6個ではなく78個の異なる $k$ を見ました。

それでも成功確率は $0.8321$ で、ピークに乗る確率の割合より高いです。これは矛盾ではなく、本章で最も教育的な数値です。ピークの*肩*にある結果も同じ近似分数に丸まるので、連分数はそこからも $r$ を回復します。後処理は単なる後片付けではありません — 幅のある分布を鋭い答に変えているのが後処理です。$t$ の走査が収束を示します。$t = 6$ で $0.787$、$t = 13$ で $0.833$ と上がり、無用な $s = 0$ の結果が定める上限 $1 - 1/r = 5/6 = 0.8333$ に近づきます。

底ごとの表は例2の分類を、今回は回路を組み込んだ形で繰り返します。$a^{r/2} \equiv -1$ の3個と奇位数の2個は決定論的に厳密ゼロ、使える6個は $r$ が2か6かに応じて $0.5$ か $0.83$ です。*どの*底が機能するかについて確率的なところは何もありません。機能する底の与えられた1回の実行が有用な $k$ に落ちるかどうかだけが確率的です。

### Code Example 6: アルゴリズム全体と、その成功頻度

両方の半分を合わせ、古典的な近道、無作為な底、再試行ループを含めて、600回の完全実行の統計を取ります。

```python
"""第3章 Code Example 6: アルゴリズム全体と、その成功頻度です。
Code Example 5 の続き（同一セッション）です。"""

import numpy as np
from math import gcd

_CACHE = {}


def measure_k(a, N, t, n_work, rng):
    """キャッシュした厳密分布から、量子サブルーチンを1ショット引きます。"""
    key = (a, N, t, n_work)
    if key not in _CACHE:
        st = order_finding_state(a, N, t, n_work)
        _CACHE[key] = probs(st).reshape(2 ** t, -1).sum(axis=1)
    p = _CACHE[key]
    return int(rng.choice(p.size, p=p / p.sum()))


def shor(N, rng, max_rounds=10):
    """仕様どおりのShorのアルゴリズムです。古典側の段も全部含みます。

    (因数, 量子呼び出し回数, 経路) を返します。1ラウンドは無作為な底1個と
    位数発見回路の呼び出し1回です。
    """
    if N % 2 == 0:
        return tuple(sorted((2, N // 2))), 0, "even"
    for b in range(2, N.bit_length() + 1):
        root = round(N ** (1.0 / b))
        for cand in (root - 1, root, root + 1):
            if cand > 1 and cand ** b == N:
                return tuple(sorted((cand, N // cand))), 0, "perfect power"
    n_work = N.bit_length()
    t = 2 * n_work + 1
    calls = 0
    for _ in range(max_rounds):
        a = int(rng.integers(2, N))
        g = gcd(a, N)
        if g > 1:
            return tuple(sorted((g, N // g))), calls, "lucky gcd"
        calls += 1
        k = measure_k(a, N, t, n_work, rng)
        _, fac = postprocess(k, t, a, N)
        if fac:
            return tuple(sorted(fac)), calls, "order finding"
    return None, calls, "gave up"


for N in (15, 21):
    rng = np.random.default_rng(1234)
    n_runs = 600
    routes, calls_hist, results = {}, [], {}
    for _ in range(n_runs):
        fac, calls, route = shor(N, rng)
        routes[route] = routes.get(route, 0) + 1
        calls_hist.append(calls)
        key = "failed" if fac is None else f"{fac[0]} x {fac[1]}"
        results[key] = results.get(key, 0) + 1
    calls_hist = np.array(calls_hist)
    print(f"N = {N}: {n_runs} complete runs "
          f"(t = {2*N.bit_length()+1} counting qubits)")
    print("-" * 78)
    for key in sorted(results):
        print(f"  result {key:<12} {results[key]:>5} runs "
              f"({results[key]/n_runs:.4f})")
    for key in sorted(routes):
        print(f"  route  {key:<12} {routes[key]:>5} runs "
              f"({routes[key]/n_runs:.4f})")
    print(f"  quantum calls per run: mean {calls_hist.mean():.3f}, "
          f"median {int(np.median(calls_hist))}, max {calls_hist.max()}")
    hist = np.bincount(calls_hist)
    print("  calls histogram: " + "  ".join(
        f"{i}: {c}" for i, c in enumerate(hist) if c))
    print()

print("Where the failures come from, counted exactly rather than sampled")
print("-" * 78)
print(f"  {'N':>4}{'bases a':>9}{'lucky gcd':>11}{'r odd':>8}"
      f"{'a^(r/2)=-1':>12}{'usable':>8}{'mean P(succ | usable)':>23}")
for N in (15, 21):
    n_work = N.bit_length()
    t = 2 * n_work + 1
    lucky = [a for a in range(2, N) if gcd(a, N) > 1]
    cop = [a for a in range(2, N) if gcd(a, N) == 1]
    odd_r = [a for a in cop if order(a, N) % 2]
    minus1 = [a for a in cop if order(a, N) % 2 == 0
              and pow(a, order(a, N) // 2, N) == N - 1]
    usable = [a for a in cop if a not in odd_r and a not in minus1]
    ps = []
    for a in usable:
        st = order_finding_state(a, N, t, n_work)
        p = probs(st).reshape(2 ** t, -1).sum(axis=1)
        ps.append(sum(p[k] for k in range(2 ** t)
                      if postprocess(k, t, a, N)[1] is not None))
    print(f"  {N:>4}{len(cop) + len(lucky):>9}{len(lucky):>11}{len(odd_r):>8}"
          f"{len(minus1):>12}{len(usable):>8}{np.mean(ps):>23.6f}")
print("  Overall probability of success per round = "
      "P(lucky gcd) + P(usable base) * P(circuit succeeds).")
print("  Every failure is detected classically in microseconds, so the "
      "algorithm simply\n  draws a new base. Repetition turns a per-round "
      "probability of about one half\n  into a failure probability of 2^-m "
      "after m rounds, at negligible cost.")
```

```text
N = 15: 600 complete runs (t = 9 counting qubits)
------------------------------------------------------------------------------
  result 3 x 5          600 runs (1.0000)
  route  lucky gcd      352 runs (0.5867)
  route  order finding   248 runs (0.4133)
  quantum calls per run: mean 0.747, median 1, max 5
  calls histogram: 0: 262  1: 254  2: 63  3: 17  4: 3  5: 1

N = 21: 600 complete runs (t = 11 counting qubits)
------------------------------------------------------------------------------
  result 3 x 7          600 runs (1.0000)
  route  lucky gcd      404 runs (0.6733)
  route  order finding   196 runs (0.3267)
  quantum calls per run: mean 0.873, median 1, max 7
  calls histogram: 0: 262  1: 215  2: 85  3: 22  4: 10  5: 4  6: 1  7: 1

Where the failures come from, counted exactly rather than sampled
------------------------------------------------------------------------------
     N  bases a  lucky gcd   r odd  a^(r/2)=-1  usable  mean P(succ | usable)
    15       13          6       0           1       6               0.666667
    21       19          8       2           3       6               0.721412
  Overall probability of success per round = P(lucky gcd) + P(usable base) * P(circuit succeeds).
  Every failure is detected classically in microseconds, so the algorithm simply
  draws a new base. Repetition turns a per-round probability of about one half
  into a failure probability of 2^-m after m rounds, at negligible cost.
```

**注目すべき点。** 両方の $N$ について600回すべてが成功し、量子回路の呼び出し回数は平均でそれぞれ $0.75$ 回と $0.87$ 回、最悪で5回と7回です。定理が言うとおりに再試行ループが働いています。

さてここからが警告で、この例が存在する理由です。$N = 15$ では*実行*の $59\%$、$N = 21$ では $67\%$ が量子回路に一度も触れていません。無作為に引いた $a$ が $\gcd(a, N) > 1$ で、因数がgcdから出てしまったのです。この2つの数値は完了した実行に対する経路の割合であり、1回の引きで幸運なgcdに当たる確率よりも大きくなります。後者は $1 - (\phi(N)-1)/(N-2)$ で、$N = 15$ では $6/13 = 0.4615$、$N = 21$ では $8/19 = 0.4211$ です（底は $2 \le a < N$ から一様に引くので、どちらの数え上げからも $a = 1$ は除かれます）。経路の割合のほうが大きいのは、幸運な引きは必ずその場で実行を終わらせるのに対し、互いに素な引きは2つの古典的検査に落ちてもう一度引かせることがあるからです。いずれにせよこの割合は $N$ が大きくなればゼロに向かいますが、これほど小さい $N$ では*古典的*近道が仕事の大半をしています。したがって2桁の数に対するShorのアルゴリズムの実演は、後処理と回路機構の実演であって、因数分解についての証拠ではありません。末尾の厳密計上の表が寄与を明確に分離するので、何も信用に頼る必要はありません。

* * *

## 3.4 これが意味すること、しないこと

### 高速化は本物で、指数的ではない

知られている最良の古典因数分解アルゴリズムである一般数体篩法は

$$ \exp\left(\left(\tfrac{64}{9}\right)^{1/3} (\ln N)^{1/3} (\ln \ln N)^{2/3}\right) $$

演算で走ります。$\ln N$ について劣指数的、超多項式的です。Shorのそれは $n = \log_2 N$ として $O(n^3)$、すなわち多項式です。したがって差は指数的ではなく*超多項式的*であり、この区別は些事ではありません。素朴な $2^n$ 対 $n^3$ の比較が示唆するよりも交叉点がはるかに大きいことを意味し、また量子側が勝つために古典側が指数的に悪くある必要はないことを意味します。

そしてこれは、因数分解が古典的に難しいことの証明が存在しないことも意味します。Shorのアルゴリズムは因数分解が量子多項式クラスに属することを示しますが、古典多項式クラスの外にあることは誰も示していません。古典多項式時間の因数分解アルゴリズムは衝撃でしょうが、どの定理にも矛盾しません。Shorが証明したことの正しい言明は量子側についてのみです。

### 明示した仮定からのサイズ

$n^3$ を機械に変えるには入力が3つ必要で、それを述べずに資源見積りを引用しても意味がありません。

  * **論理回路。** $n$ ビットのモジュラー指数演算を論理量子ビット $\sim 3n$ 個、Toffoli数 $n^3$ 台で。係数は算術に依存し、10年の最適化がそれを大幅に動かしましたが指数は変えていません。
  * **誤り訂正の仮定。** 物理誤り率、符号、閾値。符号距離 $d$ は $0.1(p/p_{\text{th}})^{(d+1)/2} < 1/(\text{演算回数})$ を満たさねばならず、表面符号は論理1個あたり物理 $\approx 2d^2$ 個、加えてToffoliのためのmagic state工場を要します。
  * **クロック。** シンドローム抽出のサイクル時間と、回路のどれだけを並列化できるかの仮定。

例7は明示した3つの選択 — $p = 10^{-3}$、閾値 $10^{-2}$、サイクル時間 $1\ \mu$s — を通し、$n = 2048$ で物理量子ビット $10^7$ 台と数時間の実行時間に到達します。これらは公表されている見積りと同じ範囲にあり、それは算術については安心材料ですが前提については違います。3つの入力はどれも仮定であり、公表された数値は仮定が改善するにつれて桁で動いてきました。仮数ではなく指数を読んでください。

### 2つの誤りを並べて

| 過大評価 | 過小評価 |
| --- | --- |
| 「量子コンピュータはRSAを破る」 | 「Shorのアルゴリズムは好奇心の対象だ」 |
| どの装置もこの回路を走らせられず、量子ビット数の差は $10^6$ 以上 | 数学は決着しており指数は多項式 |
| 2桁の実演を617桁の法についての証拠として扱う | 保存された暗号文が後で復号されうることを無視 |
| 小規模実演の大部分が古典算術であることを無視 | 標準の移行がその直接の帰結であることを無視 |

どちらに対しても正しい応答は同じです。回路サイズを述べ、仮定を述べ、証明されたことと工学的なことを分けることです。

### 何が生き残り、なぜか

Shorのアルゴリズムは汎用の攻撃ではありません。**有限アーベル群における隠れ部分群問題**を解くもので、因数分解、素数を法とする離散対数、楕円曲線離散対数はすべてその実例です。3つに共通するのは、見つけるべき周期をもつ群構造です。QFTはその群の指標変換であり、周期を測定可能にしているのがそれです。

暗号系の難しさがアーベル群から来ていないところでは、Shorは何も言えません。対称暗号とハッシュ関数は構造なし探索に依拠し、そこではGroverのみが適用されます。二次高速化であり、鍵長やダイジェスト長を倍にすれば対処でき、性能数%の代価で再設計は不要です。格子問題 — Learning With Errors の基礎にある最短ベクトル問題・最近ベクトル問題 — はどちらの箱にも入りません。利用できるアーベル隠れ部分群はなく、既知の最良の量子攻撃が改善するのは最良の古典攻撃の*指数の定数*だけです。ヒューリスティックな格子篩は古典で約 $2^{0.292n}$、量子で約 $2^{0.265n}$ であり、この改善は次元を控えめに増やせば吸収できます。「既知の高速化がない」は言い方として不正確で、「費用の形を変える既知の高速化がない」が正確です。どんな時期予測でもなくそれが、格子ベースの鍵交換と署名が標準的な耐量子代替である理由であり、この移行が実演への反応ではなく通常の標準化工学として実行されている理由です。

機械がいつ現れるかに関わらず移行が急を要する理由となる非対称性が1つあります。**暗号化された通信はいま記録して後で復号できます。** 長い秘匿期間を要するものの機密性は、量子コンピュータが存在する時点で使われているアルゴリズムではなく、いま使われているアルゴリズムに依存します。署名にはこの問題がありません。署名は信頼されている間だけ攻撃に耐えればよいからです。この区別が、予測ではなく、現実の移行計画の優先順位を決めるものです。

### Code Example 7: 暗号規模までの距離

```python
"""第3章 Code Example 7: N = 21 から暗号規模の法までの距離です。
Code Example 6 の続き（同一セッション）です。"""

import numpy as np

# 以下の定数はすべて明示したモデル上の仮定であり、測定値ではありません。
P_PHYS = 1e-3          # 仮定した物理2量子ビット誤り率
P_THRESH = 1e-2        # 仮定した表面符号の閾値
CYCLE = 1e-6           # 仮定した表面符号の測定サイクル、秒
TOFFOLI_COEF = 0.3     # モジュラー指数演算中のToffoli数 ~ 0.3 n^3


def gnfs_ops(n_bits):
    """一般数体篩法の漸近コストのヒューリスティックな評価です。"""
    lnN = n_bits * np.log(2.0)
    return np.exp((64.0 / 9.0) ** (1 / 3) * lnN ** (1 / 3)
                  * np.log(lnN) ** (2 / 3))


def surface_code_distance(n_logical_ops):
    """0.1 (p/p_th)^((d+1)/2) < 1/(10 n_logical_ops) を満たす最小の奇数 d です。"""
    for d in range(3, 60, 2):
        if 0.1 * (P_PHYS / P_THRESH) ** ((d + 1) / 2) < 0.1 / n_logical_ops:
            return d
    return None


print("Circuit size against modulus size (this chapter's own accounting)")
print("-" * 78)
print(f"  {'n bits':>8}{'logical qubits':>16}{'Toffolis ~0.3n^3':>19}"
      f"{'GNFS ops':>13}{'quantum/GNFS':>15}")
for n_bits in [5, 32, 256, 1024, 2048, 4096]:
    logical = 3 * n_bits
    toff = TOFFOLI_COEF * n_bits ** 3
    g = gnfs_ops(n_bits)
    print(f"  {n_bits:>8}{logical:>16d}{toff:>19.2e}{g:>13.2e}"
          f"{toff/g:>15.2e}")
print("  The crossover is not in doubt -- polynomial beats subexponential -- "
      "but it\n  arrives at a circuit size, not at a date.")

print("\nWhat error correction does to those numbers")
print("-" * 78)
print(f"  assumptions: physical error rate {P_PHYS:.0e}, threshold "
      f"{P_THRESH:.0e}, cycle time {CYCLE:.0e} s")
print(f"  {'n bits':>8}{'Toffolis':>12}{'code distance':>15}"
      f"{'physical qubits':>18}   wall clock")
for n_bits in [(21).bit_length(), 256, 1024, 2048]:
    logical = 3 * n_bits
    toff = TOFFOLI_COEF * n_bits ** 3
    d = surface_code_distance(max(toff, 10))
    phys = 2 * logical * d ** 2
    phys_total = 2.5 * phys          # magic state工場のぶん +150%
    seconds = toff * d * CYCLE       # Toffoli 1個 ~ d サイクル、直列実行
    unit = ("s" if seconds < 3600 else
            "hours" if seconds < 3 * 86400 else "days")
    val = (seconds if unit == "s" else
           seconds / 3600 if unit == "hours" else seconds / 86400)
    print(f"  {n_bits:>8}{toff:>12.2e}{d:>15}{phys_total:>18.2e}"
          f"{val:>10.1f} {unit}")
print("  Read only the exponents. The number of physical qubits is millions "
      "and the\n  runtime is hours to days -- both several orders of magnitude "
      "beyond anything\n  that has been built, and both sensitive to every "
      "assumption listed above.")

print("\nThe gap, stated as a ratio")
print("-" * 78)
n_demo, n_rsa = 5, 2048
print(f"  N = 21 needed {2*n_demo+1 + n_demo} simulated qubits and "
      f"{2*n_demo+1} controlled modular multiplications.")
print(f"  ratio of Toffoli counts, {n_rsa} bits to {n_demo} bits: "
      f"{(n_rsa/n_demo)**3:.2e}")
print(f"  ratio of logical qubit counts: {n_rsa/n_demo:.0f}")
print(f"  ratio of state-vector memory if you tried to simulate it: "
      f"2^{3*n_rsa} amplitudes")
print("  A demonstration on 21 is a demonstration of the postprocessing, not "
      "of the\n  cryptanalysis. Nothing about it is evidence that the "
      "cryptanalysis is near --\n  and nothing about it is evidence that the "
      "algorithm is wrong, either.")

print("\nWhat a quantum computer does and does not break")
print("-" * 78)
rows = [
    ("RSA", "factoring", "Shor, polynomial", "broken in principle"),
    ("Diffie-Hellman, DSA", "discrete log mod p", "Shor, polynomial",
     "broken in principle"),
    ("Elliptic-curve DH/DSA", "discrete log on a curve", "Shor, polynomial",
     "broken in principle; smaller circuits than RSA"),
    ("AES-128 / AES-256", "unstructured key search", "Grover, quadratic",
     "key length doubles, then safe"),
    ("SHA-2 / SHA-3 preimage", "unstructured search", "Grover, quadratic",
     "output length doubles, then safe"),
    ("Lattice problems (LWE)", "shortest/closest vector", "sieving: 2^0.265n",
     "the basis of the standard replacements"),
    ("Hash-based signatures", "preimage resistance", "Grover, quadratic",
     "parameter bump, then safe"),
]
print(f"  {'primitive':<24}{'hard problem':<26}{'quantum attack':<22}"
      f"status")
for a, b, c, d_ in rows:
    print(f"  {a:<24}{b:<26}{c:<22}{d_}")
print("\n  Two rows need their fine print. Collision resistance is NOT "
      "quadratically\n  sped up: the classical birthday attack already costs "
      "2^(n/2), and the best\n  known quantum collision finder (BHT) reaches "
      "2^(n/3) while needing 2^(n/3)\n  quantum memory, so hash-based "
      "signature parameters are set by preimage and\n  second-preimage "
      "resistance, where Grover does apply. And lattices are not "
      "'no\n  known speedup': heuristic sieving improves from 2^(0.292n) "
      "classically to about\n  2^(0.265n) quantumly. That is a change in the "
      "exponent's constant, absorbed by a\n  modest increase in dimension, "
      "not the polynomial-versus-exponential change Shor\n  gives for "
      "factoring.")
print("\n  The pattern: Shor needs an abelian group with a hidden period. "
      "Where the\n  hardness is unstructured search instead, only Grover "
      "applies, and a quadratic\n  speedup is answered by doubling a key "
      "length. Lattice problems fall in neither\n  box, which is why "
      "lattice-based key exchange and signatures are the standard\n  "
      "post-quantum replacements -- a migration that is engineering, "
      "already specified,\n  and independent of when or whether a "
      "cryptographically useful quantum computer\n  is built.")
```

```text
Circuit size against modulus size (this chapter's own accounting)
------------------------------------------------------------------------------
    n bits  logical qubits   Toffolis ~0.3n^3     GNFS ops   quantum/GNFS
         5              15           3.75e+01     2.89e+01       1.30e+00
        32              96           9.83e+03     9.73e+04       1.01e-01
       256             768           5.03e+06     1.12e+14       4.51e-08
      1024            3072           3.22e+08     1.32e+26       2.45e-18
      2048            6144           2.58e+09     1.53e+35       1.68e-26
      4096           12288           2.06e+10     1.29e+47       1.60e-37
  The crossover is not in doubt -- polynomial beats subexponential -- but it
  arrives at a circuit size, not at a date.

What error correction does to those numbers
------------------------------------------------------------------------------
  assumptions: physical error rate 1e-03, threshold 1e-02, cycle time 1e-06 s
    n bits    Toffolis  code distance   physical qubits   wall clock
         5    3.75e+01              3          6.75e+02       0.0 s
       256    5.03e+06             13          6.49e+05      65.4 s
      1024    3.22e+08             17          4.44e+06       1.5 hours
      2048    2.58e+09             19          1.11e+07      13.6 hours
  Read only the exponents. The number of physical qubits is millions and the
  runtime is hours to days -- both several orders of magnitude beyond anything
  that has been built, and both sensitive to every assumption listed above.

The gap, stated as a ratio
------------------------------------------------------------------------------
  N = 21 needed 16 simulated qubits and 11 controlled modular multiplications.
  ratio of Toffoli counts, 2048 bits to 5 bits: 6.87e+07
  ratio of logical qubit counts: 410
  ratio of state-vector memory if you tried to simulate it: 2^6144 amplitudes
  A demonstration on 21 is a demonstration of the postprocessing, not of the
  cryptanalysis. Nothing about it is evidence that the cryptanalysis is near --
  and nothing about it is evidence that the algorithm is wrong, either.

What a quantum computer does and does not break
------------------------------------------------------------------------------
  primitive               hard problem              quantum attack        status
  RSA                     factoring                 Shor, polynomial      broken in principle
  Diffie-Hellman, DSA     discrete log mod p        Shor, polynomial      broken in principle
  Elliptic-curve DH/DSA   discrete log on a curve   Shor, polynomial      broken in principle; smaller circuits than RSA
  AES-128 / AES-256       unstructured key search   Grover, quadratic     key length doubles, then safe
  SHA-2 / SHA-3 preimage  unstructured search       Grover, quadratic     output length doubles, then safe
  Lattice problems (LWE)  shortest/closest vector   sieving: 2^0.265n     the basis of the standard replacements
  Hash-based signatures   preimage resistance       Grover, quadratic     parameter bump, then safe

  Two rows need their fine print. Collision resistance is NOT quadratically
  sped up: the classical birthday attack already costs 2^(n/2), and the best
  known quantum collision finder (BHT) reaches 2^(n/3) while needing 2^(n/3)
  quantum memory, so hash-based signature parameters are set by preimage and
  second-preimage resistance, where Grover does apply. And lattices are not 'no
  known speedup': heuristic sieving improves from 2^(0.292n) classically to about
  2^(0.265n) quantumly. That is a change in the exponent's constant, absorbed by a
  modest increase in dimension, not the polynomial-versus-exponential change Shor
  gives for factoring.

  The pattern: Shor needs an abelian group with a hidden period. Where the
  hardness is unstructured search instead, only Grover applies, and a quadratic
  speedup is answered by doubling a key length. Lattice problems fall in neither
  box, which is why lattice-based key exchange and signatures are the standard
  post-quantum replacements -- a migration that is engineering, already specified,
  and independent of when or whether a cryptographically useful quantum computer
  is built.
```

**注目すべき点。** 最初の表が示すのは2つの費用の*形*であり、最後の列を交叉点の位置と読んではいけません。量子側は最適化された定数 — $0.3n^3$ Toffoli、算術回路の10年の成果 — を担いでいる一方、古典側は篩法の裸の漸近形で定数を1に置いたものであり、篩法の真の定数は1ではありません。この表が確かに示すのは、比が32ビットから2048ビットの間で25桁下がるということです。一方は多項式、他方は劣指数関数であり、定数をどう選んでもそれは変わりません。2本の曲線が実際にどこで交わるかはどちらの列も押さえていない定数の問題であり、どちらの側がいつ実行できるようになるかについても何も言っていません。

2番目の表が誤り訂正の効果です。物理誤り率 $10^{-3}$ で $2.6\times10^9$ Toffoliに到達するには符号距離19が必要で、したがって物理量子ビット $10^7$ 個と、述べた仮定の下で約半日の実行時間です。その行のどの数値もファイル冒頭で宣言した4つの定数の帰結であり、物理誤り率を1桁動かすと $d$ がおよそ2倍動きます — $p = 10^{-3}$ で19、$p = 10^{-4}$ で9です — したがって $d^2$ に比例する物理量子ビット数はおよそ4.5倍動きます。「どれだけ遠いのか」がアルゴリズムの問題ではなく材料と工学の問題であるとはこの意味です。アルゴリズムは何十年も前に完成しています。

3番目のブロックは差を比として述べ、4番目は読者が持ち帰るべき要約です。Shorは難しさがアーベル隠れ部分群問題である3つの基本要素を破ります。Groverは難しさが構造なし探索であるすべての実効鍵長を半分にしますが、パラメータを倍にすれば修復できます。格子問題はどちらの機構にも、指数を少し削る以上には攻撃されません — 量子篩の $2^{0.265n}$ 対古典の $2^{0.292n}$ です — だから代替になります。この3つの言明はいずれも年号に依存しません。

* * *

## 演習

#### 演習1: 還元を手で

$N = 33$、$a = 5$ とします。

  1. 33を法とする5の位数 $r$ を手計算か逐次乗算で求めてください。
  2. 還元を適用してください。$r$ は偶数か、$a^{r/2} \bmod N$ は何か、2つのgcdは何を与えますか。
  3. $a = 10$ と $a = 32$ で繰り返してください。3.1節の2つの失敗様式で各結果を分類してください。
  4. $33 = 3 \times 11$ は相異なる奇素因数を $k = 2$ 個もつので、定理は成功確率 $1 - 2^{-1} = 1/2$ 以上を約束します。互いに素な底の個数と成功する個数を数え、比べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(5^1 = 5\)、\(5^2 = 25\)、\(5^3 = 125 = 26\)、\(5^4 = 130 = 31\)、\(5^5 = 155 = 23\)、\(5^6 = 115 = 16\)、\(5^7 = 80 = 14\)、\(5^8 = 70 = 4\)、\(5^9 = 20\)、\(5^{10} = 100 = 1\)。よって \(r = 10\) です。</p>

<p><strong>2.</strong> \(r\) は偶数、\(5^5 = 23 \bmod 33\) で1でも32でもありません。\(\gcd(22, 33) = 11\)、\(\gcd(24, 33) = 3\) でどちらも自明でなく、\(33 = 3 \times 11\) です。</p>

<p><strong>3.</strong> \(a = 10\): \(10^2 = 100 = 1\) なので \(r = 2\)、\(10^1 = 10\)、\(\gcd(9,33) = 3\)、\(\gcd(11,33) = 11\) — 成功です。\(a = 32 \equiv -1\): \(r = 2\) で \(a^{r/2} = 32 \equiv -1\)、2番目の失敗様式でどちらのgcdも自明です。この2つはどちらも第1の失敗様式（位数が奇数）を示していません。\(N = 33\) でその様式が最初に現れるのは位数5の \(a = 4\) であり、両方の様式をすべて数え上げるのが(4)です。</p>

<p><strong>4.</strong> 互いに素な剰余は \(\phi(33) = 20\) 個で、うち \(1 < a < 33\) にあるのは19個です。列挙すると10個が成功、9個が失敗し、失敗様式は両方とも現れます。成功する10個は \(a = 5, 7, 10, 13, 14, 19, 20, 23, 26, 28\) で、位数は10、ただし \(a = 10\) と \(a = 23\) は位数2です。\(a^{r/2} \equiv -1\) で失敗するのは5個、位数10の \(a = 2, 8, 17, 29\) と位数2の \(a = 32\) です。位数が奇数で失敗するのは4個、\(a = 4, 16, 25, 31\) でいずれも位数 \(r = 5\) です。</p>

<p>したがって \(1 < a < 33\) 上の成功率は \(10/19 = 0.526\) です。定理そのものの標本空間は \(N\) と互いに素な整数から一様に引く \(a\) であり、\(a = 1\) を含みます。\(a = 1\) は位数1、すなわち奇数なので失敗し、\(\mathbb{Z}_{33}^{\ast}\) 上の成功率はちょうど \(10/20 = 0.5000\) です。<strong>この限界はここでは保守的ではなく、ぴったり達成されています</strong> — \(N = 21\) でも同じで、数え上げは \(6/12 = 0.5000\) です。\(k = 2\) で定理が約束するのは \(\ge 1 - 2^{-1}\) であり、この2つの法はそれを厳密に達成します。「少なくとも半分」から期待される余裕は、素因数がもっと多い法の性質であって、この限界の性質ではありません。系統的に確認するには Code Example 2 を \(N = 33\) で再利用すればよく、コードの変更は不要です。</p>

</details>

#### 演習2: なぜ $t = 2n + 1$ か

計数レジスタは作業レジスタの2倍プラス1ビットです。

  1. $2^t \ge N^2$ なら $r < N$ のすべてに対して $1/2^{t+1} < 1/(2r^2)$ であることを示してください。
  2. 連分数の一意性定理は、$\gcd(s,r) = 1$ で $\lvert x - s/r \rvert < 1/(2r^2)$ なら $s/r$ は $x$ の近似分数である、と言います。これを1と組み合わせて $t = 2n+1$ を正当化してください。
  3. $t = n$ だと何が壊れますか。分母が $N$ 未満の異なる2つの分数が互いに $2^{-n-1}$ 以内にある確率を見積ってください。
  4. Code Example 5 は $N = 21$ で $t = 6$ でも成功しますが、定理は $t = 11$ を求めます。これが矛盾でないのはなぜですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(r < N\) では \(2^t \ge N^2 > r^2\) なので \(1/2^{t+1} \le 1/(2N^2) < 1/(2r^2)\) です。</p>

<p><strong>2.</strong> 位相推定は最良の結果について \(\lvert k/2^t - s/r \rvert \le 2^{-(t+1)}\) を保証し、1によりこれは \(1/(2r^2)\) 未満です。すると定理が \(s/r\) を \(k/2^t\) の近似分数の中に置き、Euclidの互除法が \(O(n)\) 段でそれを列挙します。\(t = 2n+1\) は、すべての \(n\) ビット \(N\) について \(2^t \ge N^2\) を保証する最小のレジスタ長です。</p>

<p><strong>3.</strong> \(t = n\) だと保証は \(\lvert k/2^n - s/r\rvert \le 2^{-(n+1)} \approx 1/(2N)\) だけです。\([0,1)\) には分母が \(N\) 未満の分数が \(\Theta(N^2)\) 個あるので典型的な間隔は \(\Theta(1/N^2)\) — 誤差棒よりはるかに小さいです。多数の候補分数が当てはまり、復元は一意でありません。それでも時々は機能し、それがまさに4です。</p>

<p><strong>4.</strong> 定理は \(r < N\) のすべてにわたる最悪ケースの保証です。特定の小さい \(r\) — ここでは \(r = 6\) — に対しては分数 \(s/6\) が広く離れているので、はるかに粗いレジスタで分解できます。\(t = 2n+1\) は \(r\) が未知で \(N\) ほど大きくなりうるときに選ばねばならない値で、特定の実例が必要とする値ではありません。</p>

</details>

#### 演習3: $N = 21$ の分布を読む

Code Example 5 の $t = 11$ の出力を使ってください。

  1. 6個のピークは確率の $0.7893$ を占めるのに、成功確率は $0.8321$ です。余分な $0.043$ はどこから来ますか。
  2. $t \to \infty$ での上限 $5/6$ を導いてください。
  3. $a = 8$（位数2）では成功確率がどの $t$ でも厳密に $0.5$ です。おおよそではなく厳密に2分の1である理由を説明してください。
  4. 回路を2回走らせて結果を組み合わせてよいとします。1実験あたりの成功確率を $5/6$ より上げる後処理規則を述べ、$a = 2$、$N = 21$ での新しい値を見積ってください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 肩からです。ピークから1〜2ビン離れた結果 \(k\) でも、\(k/2^t\) は分母21未満の他のどの分数よりも同じ \(s/r\) に近いので、その連分数展開は同じ近似分数を返します。後処理の捕獲半径は1ビンより広く、その半径に余分な確率が住んでいます。</p>

<p><strong>2.</strong> 入力 \(\lvert 1 \rangle\) は \(r\) 個の固有ベクトル \(\lvert u_s \rangle\) の一様重ね合わせなので \(s\) は \(\lbrace 0,\ldots,r-1\rbrace\) 上一様です。\(s = 0\) はピークを \(k = 0\) に置き \(r\) についての情報を与えないので、上限は \(r = 6\) に対して \(1 - 1/r = 5/6\) です。\(t\) は何も変えません。</p>

<p><strong>3.</strong> \(r = 2\) では固有位相が \(0\) と \(1/2\) で、どちらも任意の \(t \ge 1\) ビットで厳密に表現できます。したがって分布は \(k = 0\) と \(k = 2^{t-1}\) の厳密に2本のデルタピークで各々厳密に確率 \(1/2\) です。漏れがないので \(t\) 依存性もなく、上限 \(1 - 1/r\) が達成されます。</p>

<p><strong>4.</strong> 測定された2つの分母 \(q_1, q_2\) について、各々に加えて \(\mathrm{lcm}(q_1, q_2)\) も検査します。失敗するには両方の実行が無情報、すなわち両方が \(s = 0\) になるか、lcmでも \(r\) を外す必要があります。1回あたり \(P(s = 0) = 1/6\) なので両方がそうなる確率は \(1/36\) で、2回で成功確率は約 \(0.97\) — 回路コスト2倍の代価です。これが標準的な対処であり、「アルゴリズムは確率2分の1以上で成功する」が実務上の制限でない理由です。</p>

</details>

#### 演習4: コストは算術である

Code Example 3 のスケーリングモデルを使ってください。

  1. $n = 1025$ — Code Example 3 の $\sim 2^{1024}$ の行で、$t = 2n+1 = 2051$ です — について、$n^3$ 対 $t^2/2$ のモデルでモジュラー指数演算のゲート数と逆QFTのゲート数の比を計算し、それが $n$ とともにどうスケールするか述べてください。
  2. 誰かがQFTを100倍改善したとします。$n = 1024$ で回路全体はどれだけ縮みますか。
  3. 代わりに誰かが剰余乗算器を $n^2$ から $n^{1.6}$ ゲートに改善したとします（Karatsuba的）。全体はどれだけ縮みますか。
  4. これは因数分解回路のどの部分が最適化の労力に値するか、そして「効率的なQFT」という題の論文をどう読むべきかについて、何を含意しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 算術 \(\approx t n^2 = 2051 \times 1025^2 = 2.15\times10^9\)、QFT \(\approx t(t+1)/2 = 2.1\times10^6\)。比は \(\approx 1020\) で、\(n^3/n^2 = n\) として増えるのでサイズとともに差が開きます。</p>

<p><strong>2.</strong> QFTが \(2.1\times10^6\) から \(2.1\times10^4\) に落ち、合計は \(2.152\times10^9\) から \(2.150\times10^9\) — 節約は千分の一です。</p>

<p><strong>3.</strong> \(t n^{1.6} = 2051 \times 1025^{1.6} = 2051 \times 6.56\times10^4 = 1.35\times10^8\) で、合計に対して16分の1です。QFTを2桁改善するより乗算器を控えめに改善するほうが価値があります。</p>

<p><strong>4.</strong> 労力はすべて算術に属します — 実際にそこへ向かってきました。ウィンドウ化算術、より良い加算器、Toffoli数の削減です。より効率的なQFTについての論文は、因数分解回路の無視できる項についての論文です。QFTに \(n^3\) の算術が伴わない他のアルゴリズムでは重要かもしれませんが、このアルゴリズムでは重要ではありません。</p>

</details>

#### 演習5: 主張を監査する

プレスリリースにこうあります。「当社のプロセッサはShorのアルゴリズムで48ビットの数を因数分解しました。記録です。外挿すればRSA-2048も射程内です。」

  1. Code Example 3 のモデルを使うと、本物の48ビット因数分解に必要な回路サイズは量子ビット数と剰余乗算回数でいくらですか。
  2. その回路を走らせずに正しい因数を出せる方法を3つ挙げ、それぞれを排除できる公表すべき細目を述べてください。
  3. $n^3$ スケーリングを額面どおり取ると、48ビットと2048ビットの回路サイズ比はいくらで、外挿が成り立つには「射程内」が何を意味しなければなりませんか。
  4. 代わりにリリースに入っていてほしい2つの文を書いてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(n = 48\)、\(t = 97\) なので \(t + n = 145\) 論理量子ビット、制御剰余乗算97回、算術部分に \(\sim t n^2 = 2.2\times10^5\) ゲートです。例7の仮定で誤り訂正すると最低でも物理数千量子ビットであり、どのゲートもコヒーレントに成功しなければなりません。</p>

<p><strong>2.</strong> (i) 底 \(a\) を \(\gcd(a,N) > 1\) となるように、あるいは \(r\) が極小で既知になるように選んだ場合。回路が潰れます — 底、位数、測定分布全体を公表すれば排除できます。(ii) モジュラー指数演算を既知の因数分解を<em>使って</em>コンパイルした場合。小規模Shor実演に対する標準的な批判です — \(N\) と \(a\) のみに依存する回路を公表すれば排除できます。(iii) 後処理が仕事をした場合。\(N\) が小さければ候補 \(r\) をいくつか古典的に試すだけで量子的情報なしに因数が出ます — Code Example 4 と 5 のように1回あたりの成功確率を理論値と比べて報告すれば排除できます。</p>

<p><strong>3.</strong> 誤り訂正前でゲート数比 \((2048/48)^3 = 7.8\times10^4\)、量子ビット数比 \(2048/48 = 43\) です。誤り訂正後は符号距離が回路サイズとともに増えるので物理量子ビット比はさらに大きくなります。「射程内」は、コヒーレントなゲート数で5桁、物理量子ビットで4桁が漸進的だという意味でなければならず、ハードウェアのどんな読み方もそれを支持しません。</p>

<p><strong>4.</strong> たとえばこうです。「48ビット整数を、\(N\) と無作為に選んだ底のみを入力とする回路で因数分解し、測定された1回あたりの成功確率は理論値 \(1 - 1/r\) と標本誤差の範囲で一致した。」そして「同じ回路を2048ビットへスケールすると \(\sim 10^9\) 個のToffoliゲートと、標準的な表面符号の仮定の下で \(\sim 10^7\) 個の物理量子ビットを要する。本実演はその領域についての証拠ではない。」最初の文が結果を検査可能にし、2番目の文が誠実にします。</p>

</details>

* * *

## まとめ

### 要点

**1\. 因数分解は古典的に位数発見へ還元される**

  * $r$ を $N$ を法とする $a$ の位数とし $r$ が偶数なら、$a^{r/2}\equiv-1$ でない限り $\gcd(a^{r/2}\pm1, N)$ は真の約数です。
  * 失敗はマイクロ秒で検出されるのでアルゴリズムは底を引き直します。相異なる奇素因数2個の $N$ に対する底ごとの成功確率は $1/2$ 以上で、列挙では $N = 15$ で $6/7$、$N = 21$ で $6/11$ でした。
  * 偶数の $N$ と完全べき乗は回路に到達せず、幸運な $\gcd(a,N) > 1$ は回路を丸ごと飛ばします。2桁の $N$ では実行の大半でそうなります。

**2\. 位数発見は置換に対する位相推定である**

  * $U_a\lvert y\rangle = \lvert ay \bmod N\rangle$ の固有位相は $s/r$ であり、計算基底状態 $\lvert 1 \rangle$ は $r$ 個の固有ベクトル全部の一様重ね合わせです。だから1回の実行が一様乱数の $s$ を返し、$s = 0$ が確率 $1/r$ の無用な結果です。
  * したがって*位数発見回路*の1回あたり成功確率の上限は $1 - 1/r$ であり、$r$ が $2^t$ を割るとき厳密に達成されます。*因数分解アルゴリズム*の1ラウンドあたり成功確率はこれとは別の量で、Code Example 6 がこれに幸運なgcdと悪い底の場合を足し合わせて組み立てています。
  * 連分数が測定された $k/2^t$ を既約形の $s/r$ に変換し、レジスタ長 $t = 2n+1$ がその復元を一意にします。

**3\. フーリエ変換は安い部分である**

  * 制御剰余乗算 $2n+1$ 回、各 $O(n^2)$ ゲートで $O(n^3)$。逆QFTは $O(n^2)$。$n = 2049$ で比は2000対1です。
  * したがって近似QFTは自由に使えて、最適化の労力はすべて算術に属します。

**4\. 両方の整数が因数分解でき、統計は理論と合う**

  * $N = 15$、$a = 7$: 厳密なピーク4本、1回あたり成功確率 $3/4$ に対し2000ショットの測定値 $0.759$。$a = 14$ は $a^{r/2} \equiv -1$ のため厳密ゼロです。
  * $N = 21$、$a = 2$: ピークは $s/6$ の近く、そこに乗る確率は $78.9\%$ しかないのに成功確率 $0.8321$ — 肩も正しく後処理され、上限 $5/6$ に $t$ とともに下から近づきます。
  * アルゴリズム全体の600回の完全実行はすべて成功し、量子呼び出しは平均 $0.75$ 回と $0.87$ 回でした。

**5\. 暗号についての結論、両方向に**

  * 分離は超多項式的で決着済みです。それでも因数分解が古典的に難しいことの証明はありません。
  * $p = 10^{-3}$、閾値 $10^{-2}$、サイクル時間 $1\ \mu$s から、$n = 2048$ はToffoli $2.6\times10^9$ 個、符号距離19、物理量子ビット $10^7$ 台、実行時間は数時間です。指数を読んでください。
  * Shorはアーベル隠れ部分群問題を解くので、RSA、有限体上の離散対数、楕円曲線離散対数が原理的に落ちます。対称基本要素に対するGroverの二次高速化はパラメータを倍にすれば対処できます。格子問題はどちらにも指数の定数以上には攻撃されないので標準的な代替であり、記録された暗号文が移行を実演まで待てない理由です。

**実務上の含意**

  * Shorの実演を読むときは、底、測定分布、$1 - 1/r$ と比べた1回あたり成功確率を求めてください。この3つで検査可能になります。
  * 資源見積りを読むときは、Toffoli数、仮定した物理誤り率、符号距離を求めてください。それらのない物理量子ビット数は見積りではありません。
  * 耐量子移行を計画するときは機密性と真正性を分けてください。遡って脅かされるのは前者だけです。

### 次章へ

第3章は本コースにおける証明可能な量子優位の最高水位点であり、それを可能にしたものに注意する価値があります。隠れた*群*構造をもつ問題、厳密な周期性、そして1回の測定を答に変える古典後処理です。第4章は、物理的動機が最も強くそのような構造をもたない応用 — ハミルトニアンのシミュレーション — に移り、そこでは研究対象が $e^{-iHt}$ そのものです。そこでの技法（ブロック符号化、qubitization、ランダム化コンパイル）は入門コースのTrotter分解の現代的な代替であり、同時に第2章の位相推定が無料と仮定した制御 $e^{-iH\tau}$ を供給するものです。

[← 第2章: 量子フーリエ変換と位相推定](<chapter-2.html>) [第4章: ハミルトニアンシミュレーションの現代的手法 →](<chapter-4.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の資源見積り — ゲート数、符号距離、物理量子ビット数、実行時間 — はコード中に明記したモデル上の仮定から従う桁レベルの教育用概算であり、予測でも測定値でもありません。本章は暗号に関する助言ではありません。セキュリティ上の判断を行う前に最新の標準および一次資料を参照してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
