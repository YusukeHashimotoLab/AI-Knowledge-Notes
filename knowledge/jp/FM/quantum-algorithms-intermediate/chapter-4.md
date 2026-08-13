---
title: "第4章: ハミルトニアンシミュレーションの現代的手法"
chapter_title: "第4章: ハミルトニアンシミュレーションの現代的手法"
subtitle: ブロック符号化、qubitization、ランダム化コンパイル、そしてToffoli数で語るということ
reading_time: 45-50分
difficulty: 上級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-algorithms-intermediate/chapter-4.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子アルゴリズム（中級）](<index.html>) > 第4章

[第2章](<chapter-2.html>)で構築した位相推定は、ユニタリを固有値に変える装置です。化学と材料でほしいユニタリは $e^{-iHt}$ であり、姉妹コースはそれを素直な方法で作りました。指数関数をPauli回転の積に刻み、誤差を受け入れるのです。[量子コンピューティング入門 第4章](<../quantum-computing-introduction/chapter-4.html>)はその誤差を漸近形で引用せずに実測しました。そして測定結果は落胆させるものでした。4量子ビットのHubbard二量体の時間発展1単位を $10^{-6}$ に抑えるのに約 $3 \times 10^{7}$ 回のPauli回転が必要で、しかもその模型の厳密解は `numpy.linalg.eigvalsh` がマイクロ秒で返してきます。本章は、この状況に対して分野が何をしたかを扱います。

考え方は3つあり、Trotter分解からの距離が近い順に並べます。**ブロック符号化**は $H$ の指数関数ではなく $H$ そのものを、より大きなユニタリの内部に埋め込みます。代価は補助レジスタ1つと規格化因子1つです。**qubitization**はそのユニタリを量子ウォークに変え、そのウォークの固有位相が規格化ハミルトニアンの固有値の $\arccos$ になるようにします。この性質のおかげで、ウォーク上の位相推定はクエリモデルで最適になります。**qDRIFT**は逆方向に進み、決定性そのものを放棄してハミルトニアンの項をランダムに抽出します。4.4節は、これらすべてがどんな言語で報告されるかを扱います。誤り耐性アルゴリズムの論文は実行時間を提出しません。提出するのはToffoli数、論理量子ビット数、そして前提条件の一覧です。それを正しく読むことは、意識して身につける価値のある技能です。

## 学習目標

本章を読み終えると、以下ができるようになります：

  * 1次と2次の積公式を思い出し、固有値問題にとって重要な形 — どんな有限次数でも消えない $1/\varepsilon^{1/2k}$ 依存性 — で誤差スケーリングを述べられる
  * ハミルトニアンの $(\alpha, m, 0)$ ブロック符号化を定義し、規格化 $\alpha$ がスペクトルノルムではなく1ノルムである理由を説明できる
  * ユニタリの線形結合（LCU）によるブロック符号化をPREPAREとSELECTから明示的に構成し、2量子ビットの例で両方をゲートレベルまで下ろし、得られた $16 \times 16$ ユニタリの左上ブロックが $H/\alpha$ そのものであることを検証できる
  * LCU 1ラウンドの成功確率を計算し、$\lVert H \lvert \psi \rangle \rVert / \alpha$ と結びつけ、ほぼ確実な成功に必要な振幅増幅のラウンド数を数えられる
  * qubitizationウォーク $W$ を構成し、その固有位相が $\cos\theta_k = E_k/\alpha$ を満たすことを数値的に検証し、$W$ 上で位相推定を走らせて基底状態エネルギーを取り出せる
  * qDRIFTチャネルをスーパー作用素として厳密に評価し、その $O(\lambda^2 t^2/N)$ 誤差と項数 $L$ への非依存性を確認し、2次Trotterとの交差点を突き止められる
  * 誤り耐性のリソース見積りをToffoli数・論理量子ビット数・表面符号の距離で読み書きし、答えがどの入力に実際に敏感かを見分けられる

### 引き継ぐもの

以下のすべては[量子コンピューティング入門 第2章](<../quantum-computing-introduction/chapter-2.html>)のミニシミュレータの上で走ります。本章が自己完結するようCode Example 1に再掲します。規約は**ビッグエンディアン**で、量子ビット0がケットの左端かつ振幅添字の最上位ビットです。この規約は4.2節で直ちに元が取れます。補助レジスタが状態ベクトルの*先頭*添字を占めるので、「ハミルトニアンが住むユニタリのブロック」が、印字した行列の文字どおり左上隅になるのです。

本章を通して使う記号を2つ、いま区別しておきます。$\alpha$ はブロック符号化の規格化で、LCU構成では $\lambda = \sum_l \lvert c_l \rvert$、すなわちハミルトニアンのPauli分解の**1ノルム**に等しくなります。$\lVert H \rVert$ はスペクトルノルムです。常に $\alpha \ge \lVert H \rVert$ で、通常その比は系のサイズとともに増大します。そして本章のあらゆるコスト式には $\lVert H \rVert$ ではなく $\alpha$ が入ります。この2つを混同することは、リソース要求を1桁過小評価する最短経路です。

* * *

## 4.1 Trotter分解の復習と、その限界

### 積公式の復習

$e^{-iHt}$ のデジタルシミュレーションは、厳密に指数化できる部分への分解 $H = \sum_{j=1}^{L} H_j$ から始まります。本章では実係数のPauli文字列です。部分は互いに可換ではないので素朴な積は誤りであり、1次のLie-Trotter公式がその誤りの大きさを定量化します：

$$ e^{-iHt} = \left(\prod_{j=1}^{L} e^{-iH_j t/r}\right)^{r} + O\left(\frac{t^2}{r}\right) $$

対称化された2次公式は各ステップを半分にし、前向きと後ろ向きに掃くことで主要誤差項を打ち消します：

$$ e^{-iHt} \approx \left(\prod_{j=1}^{L} e^{-iH_j t/2r} \prod_{j=L}^{1} e^{-iH_j t/2r}\right)^{r} + O\left(\frac{t^3}{r^2}\right) $$

Pauli文字列 $P$ に対する各因子 $e^{-i\theta P}$ はCNOTのはしご、$R_z$ 1個、そして逆順のはしごにコンパイルされるので、ゲート数は（非恒等文字列の個数）$\times$（ステップ数）$\times$（重みに依存するCNOTコスト）です。恒等文字列は大域位相であり、ゲートを一切必要としません。

### 誤差が実際にどうスケールするか

$O$ 記法は定数を隠しますが、実務上の困難はすべてその定数に宿ります。$2k$ 次の積公式に対する標準的な限界は*費用*についての限界であり、誤差を $\varepsilon$ に抑えるために必要なゲート数は

$$ \text{ゲート数} \; = \; O\left(\frac{L\,(\alpha t)^{1 + 1/2k}}{\varepsilon^{1/2k}}\right) \quad \text{（目標誤差 } \varepsilon \text{ に対して）} $$

であり、交換子に依存するより厳しい版も知られています。この式については、厳密な形よりも次の2つの特徴のほうが重要です。

**$1/\varepsilon$ への依存は多項式であり、決して対数にはなりません。** 次数 $k$ を上げると指数は $1/2k$ に改善しますが、Suzuki再帰で1ステップあたりのコストがおよそ $5^{k-1}$ 段に増えるので、$(t, \varepsilon)$ ごとに最適な $k$ が存在し、それでも多項式は消えません。$\varepsilon = 10^{-3}$ で十分なことが多い*ダイナミクス*の計算ならこれは許容できます。*固有値*の計算では許容できません。精度 $\varepsilon$ の位相推定は時間 $t \sim 1/\varepsilon$ までのコヒーレントな発展を要求し、$t^{1+1/2k}$ の因子が $\varepsilon^{-1/2k}$ の因子と掛け算になるからです。

**$L$ への依存は線形で、しかも $L$ は大きいのです。** $M$ 個の空間軌道の電子構造ハミルトニアンは、Jordan-Wigner変換の後で $O(M^4)$ 個のPauli文字列をもちます。姉妹コースのFeMoco活性空間である $M = 76$ では、切り捨て前で $10^{7}$ 項の程度になり、そのすべてがすべてのTrotterステップに現れます。

姉妹コースがHubbard二量体で行ったのと同じことを、本章が通して使うハミルトニアンについて実行し、両方の定数を測りましょう。後で出てくるqubitizationの数値が比較すべき具体的な相手をもつようにするためです。

### Code Example 1: ミニシミュレータの再掲

```python
"""Minimal state-vector simulator (big-endian: qubit 0 = leftmost = most significant).

Save this file as qcsim.py; every later example does `from qcsim import *`.
"""
import numpy as np

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
```

99行、NumPy以外の依存はゼロです。`qcsim.py` として保存してください。本章の以降のすべての例は `from qcsim import *` で始まります。

### Code Example 2: 本章の実行例ハミルトニアンとTrotterの基準値

```python
"""第4章 Code Example 2: 本章の実行例ハミルトニアンとTrotterの基準値。
Code Example 1 の qcsim.py の上で走ります。"""
import numpy as np
from functools import reduce
from qcsim import *

def pauli_matrix(s):
    """'ZZ' のようなPauli文字列の密行列（量子ビット0が左端）"""
    return reduce(np.kron, [PAULI[c] for c in s])


def to_matrix(terms):
    """{'ZZ': 0.8, ...} の密行列"""
    n = len(next(iter(terms)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in terms.items():
        M += c * pauli_matrix(s)
    return M


def expm_hermitian(M, scalar):
    """エルミート行列 M に対する exp(scalar * M) を固有分解で計算します"""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(scalar * w)) @ v.conj().T


# 本章の実行例: 4項からなる2量子ビットハミルトニアン。
# LCUの係数レジスタがちょうど2量子ビットになるように選びました。
HAM = {'IX': 0.3, 'XI': 0.5, 'YY': 0.2, 'ZZ': 0.8}
STRINGS = sorted(HAM)
COEF = np.array([HAM[s] for s in STRINGS])
ALPHA = COEF.sum()                      # 1ノルム  alpha = sum_l |c_l|
Hmat = to_matrix(HAM)

print("本章の実行例ハミルトニアン")
print("=" * 68)
for s, c in zip(STRINGS, COEF):
    print(f"    {c:+.4f} * {s}")
print(f"  1ノルム  alpha = sum_l |c_l| = {ALPHA:.4f}")
print(f"  スペクトルノルム ||H|| = "
      f"{np.linalg.norm(Hmat, ord=2):.6f}")
print(f"  固有値 = "
      f"{np.round(np.linalg.eigvalsh(Hmat), 6)}")
print(f"  alpha / ||H|| = "
      f"{ALPHA/np.linalg.norm(Hmat, ord=2):.4f}"
      "   (LCU表現の代価)")

tau = 1.0
U_exact = expm_hermitian(Hmat, -1j * tau)
terms = [(s, HAM[s]) for s in STRINGS]

print(f"\n1次のLie-Trotter分解、dt = tau/r の r ステップ、tau = {tau}")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r':>11}"
      f" {'rotations':>10}")
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    dt = tau / r
    step = np.eye(4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r:11.5f} {r*len(terms):10d}")

print("\n2次（対称）Suzuki公式")
print(f"{'steps r':>8} {'dt':>9} {'spectral error':>16} {'error x r^2':>13}"
      f" {'rotations':>10}")
for r in (1, 2, 4, 8, 16, 32):
    dt = tau / r
    step = np.eye(4, dtype=complex)
    for s, c in terms:
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    for s, c in reversed(terms):
        step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    err = np.linalg.norm(np.linalg.matrix_power(step, r) - U_exact, ord=2)
    print(f"{r:8d} {dt:9.5f} {err:16.6e} {err*r*r:13.5f} "
          f"{2*r*len(terms):10d}")

# 2つの漸近定数を実測します
dt = tau / 128
step = np.eye(4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
C1 = np.linalg.norm(np.linalg.matrix_power(step, 128) - U_exact, ord=2) * 128
dt = tau / 32
step = np.eye(4, dtype=complex)
for s, c in terms:
    step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
for s, c in reversed(terms):
    step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
C2 = np.linalg.norm(np.linalg.matrix_power(step, 32) - U_exact, ord=2) * 32 ** 2

print("\n実測した定数と tau = 1 での外挿コスト")
print("-" * 68)
print(f"  1次:  誤差 ~ C1 / r    C1 = {C1:.4f}")
print(f"  2次:  誤差 ~ C2 / r^2  C2 = {C2:.4f}")
print(f"{'target error':>13} {'r (1st)':>12} {'rotations':>12}"
      f" {'r (2nd)':>10} {'rotations':>12}")
for eps in (1e-3, 1e-6, 1e-9):
    r1 = C1 / eps
    r2 = np.sqrt(C2 / eps)
    print(f"{eps:13.0e} {r1:12.3e} {r1*4:12.3e} {r2:10.1f} {r2*8:12.3e}")
```

```text
本章の実行例ハミルトニアン
====================================================================
    +0.3000 * IX
    +0.5000 * XI
    +0.2000 * YY
    +0.8000 * ZZ
  1ノルム  alpha = sum_l |c_l| = 1.8000
  スペクトルノルム ||H|| = 1.019804
  固有値 = [-1.019804 -1.        1.        1.019804]
  alpha / ||H|| = 1.7650   (LCU表現の代価)

1次のLie-Trotter分解、dt = tau/r の r ステップ、tau = 1.0
 steps r        dt   spectral error   error x r  rotations
       1   1.00000     4.292796e-01     0.42928          4
       2   0.50000     2.051686e-01     0.41034          8
       4   0.25000     1.013788e-01     0.40552         16
       8   0.12500     5.053855e-02     0.40431         32
      16   0.06250     2.525042e-02     0.40401         64
      32   0.03125     1.262285e-02     0.40393        128
      64   0.01562     6.311131e-03     0.40391        256
     128   0.00781     3.155528e-03     0.40391        512

2次（対称）Suzuki公式
 steps r        dt   spectral error   error x r^2  rotations
       1   1.00000     1.065662e-01       0.10657          8
       2   0.50000     2.528742e-02       0.10115         16
       4   0.25000     6.234274e-03       0.09975         32
       8   0.12500     1.553070e-03       0.09940         64
      16   0.06250     3.879236e-04       0.09931        128
      32   0.03125     9.695940e-05       0.09929        256

実測した定数と tau = 1 での外挿コスト
--------------------------------------------------------------------
  1次:  誤差 ~ C1 / r    C1 = 0.4039
  2次:  誤差 ~ C2 / r^2  C2 = 0.0993
 target error      r (1st)    rotations    r (2nd)    rotations
        1e-03    4.039e+02    1.616e+03       10.0    7.971e+01
        1e-06    4.039e+05    1.616e+06      315.1    2.521e+03
        1e-09    4.039e+08    1.616e+09     9964.3    7.971e+04
```

**注目すべき点。** 2つの定数は $C_1 = 0.4039$ と $C_2 = 0.0993$ であり、これがこのハミルトニアンについての $O(1/r)$ と $O(1/r^2)$ という言明の内容のすべてです。本章の以降のすべては、この2つを基準にして測られます。

最後のブロックは、積公式で固有値計算を行う場合の正直なコストです。$10^{-9}$ に到達するには — 2つのほぼ縮退したスピン状態間のエネルギー差を求めるときには不合理な目標ではありません — 1次で $1.6 \times 10^{9}$ 回、2次で $8.0 \times 10^{4}$ 回のPauli回転が必要で、しかもHilbert空間は*4次元*です。2次への改善は2万倍ですが、それでもなお $1/\varepsilon$ の多項式です。qubitizationが閉じるのはこの隔たりです。

ヘッダのブロックにある $\alpha/\lVert H \rVert = 1.765$ にも注目してください。4項しかないハミルトニアンで、1ノルムがスペクトルノルムを76%上回っています。系が大きくなるとこの比がどうなるかは、Code Example 4が示します。

* * *

## 4.2 ブロック符号化、LCU、qubitization

### 発想: $e^{-iHt}$ ではなく $H$ を符号化する

ハミルトニアンはユニタリではないので、量子コンピュータはそれを直接作用させられません。Trotter分解は指数関数だけを作用させることでこれを回避します。ブロック符号化は逆の道を取ります。ユニタリでない $H$ を、より大きな空間に作用するユニタリの1つのブロックとして埋め込み、そのユニタリを操作するのです。

形式的には、$m$ 個の補助量子ビットと系に作用するユニタリ $U_A$ が

$$ \left(\langle 0^{m} \rvert \otimes I\right) U_A \left(\lvert 0^{m} \rangle \otimes I\right) = \frac{H}{\alpha} $$

を満たすとき、$U_A$ を $H$ の **$(\alpha, m, 0)$ ブロック符号化**と呼びます。補助を先頭に置くビッグエンディアン順では、これは行列 $U_A$ の左上 $2^n \times 2^n$ 隅がちょうど $H/\alpha$ に等しいという意味になります：

$$ U_A = \begin{pmatrix} H/\alpha & \ast \cr \ast & \ast \end{pmatrix} $$

$\ast$ のブロックは自由なパラメータではありません — ユニタリ性がそのノルムを固定します — が、我々の関心事でもありません。ブロック符号化の上に立つアルゴリズムはどれも、補助を $\lvert 0^m \rangle$ に事後選択するか、その部分空間について鏡映するかのどちらかだからです。規格化 $\alpha$ は避けられません。ユニタリの左上ブロックのスペクトルノルムは高々1なので、常に $\alpha \ge \lVert H \rVert$ です。

### LCU: PREPAREとSELECT

標準的な構成は $H$ をユニタリの線形結合として書きます：

$$ H = \sum_{l=0}^{L-1} c_l\, U_l, \qquad c_l > 0, \qquad \alpha = \sum_l c_l $$

Pauli分解ならこれは無料です。符号を $U_l = \pm P_l$ に吸収させれば、すべての $c_l$ が正になります。あとは2つの回路で足ります。

**PREPARE** は $m = \lceil \log_2 L \rceil$ 個の補助量子ビットに作用し、規格化した係数の平方根を読み込みます：

$$ \mathrm{PREP} \lvert 0^{m} \rangle = \sum_{l} \sqrt{\frac{c_l}{\alpha}} \lvert l \rangle $$

**SELECT** は補助レジスタが $l$ を読んでいることを条件に、系に $U_l$ を作用させます：

$$ \mathrm{SELECT} = \sum_{l} \lvert l \rangle\langle l \rvert \otimes U_l $$

一方を他方で挟むとブロック符号化が得られます：

$$ U_A = \left(\mathrm{PREP}^{\dagger} \otimes I\right) \mathrm{SELECT} \left(\mathrm{PREP} \otimes I\right) $$

$\langle 0^m \rvert \mathrm{PREP}^\dagger \lvert l \rangle = \sqrt{c_l/\alpha}$ で、右側にも同じ因子が現れるので、定義的性質は1行で従います：

$$ \left(\langle 0^{m} \rvert \otimes I\right) U_A \left(\lvert 0^{m} \rangle \otimes I\right) = \sum_l \sqrt{\frac{c_l}{\alpha}} \sqrt{\frac{c_l}{\alpha}}\, U_l = \frac{1}{\alpha}\sum_l c_l U_l = \frac{H}{\alpha} $$

### 両方をゲートレベルまで

上の式はまだ回路ではありません。そしてこの手法のコストのすべては、それらを回路に変えるところに宿っています。

**実非負振幅のPREPARE** は $R_y$ 回転の二分木です。目標振幅 $v = (v_{00}, v_{01}, v_{10}, v_{11})$ をもつ2補助量子ビットの場合、第1量子ビットの $R_y(\theta_a)$ が確率を2つの半分に分け、第2量子ビットの*多重化*された $R_y$ — 第1量子ビットが $\lvert 0 \rangle$ なら角度 $\theta_{b0}$、$\lvert 1 \rangle$ なら $\theta_{b1}$ — が各半分の内部を分けます。多重化器は新しい基本要素ではありません。$\theta_{\pm} = (\theta_{b0} \pm \theta_{b1})/2$ とおくと

$$ \lvert 0 \rangle\langle 0 \rvert \otimes R_y(\theta_{b0}) + \lvert 1 \rangle\langle 1 \rvert \otimes R_y(\theta_{b1}) = \mathrm{CNOT}\, \left[I \otimes R_y(\theta_{-})\right]\, \mathrm{CNOT}\, \left[I \otimes R_y(\theta_{+})\right] $$

が成り立ちます。$X R_y(\theta) X = R_y(-\theta)$ だからです。合計5ゲート — $R_y$ が3個、CNOTが2個 — で厳密な2量子ビットPREPAREになります。

**SELECT** は多重制御演算 $L$ 個の積であり、Toffoliが宿るのはここです。素朴に書くと、各 $\lvert l \rangle\langle l \rvert \otimes U_l$ は深さ $m$ のToffoli木を要します。リソース見積りで実際に使われる構成は**ユナリ反復**です。$L$ 個の制御条件を補助量子ビットのはしごに沿って逐次計算し、部分積を再利用することで、SELECT全体を $O(L \log L)$ ではなく $L - 1$ Toffoliに収めます。小さな組合せ的工夫ですが公表される数値への効果は大きく、SELECTのコストがハミルトニアン1項あたりToffoli *1個*になる理由です。

### Code Example 3: 2量子ビットハミルトニアンの明示的ブロック符号化

```python
"""第4章 Code Example 3: 2量子ビットハミルトニアンのブロック符号化を明示構成します。
Code Example 2 の続き（同一セッション）。"""
I4 = np.eye(4, dtype=complex)
E4 = np.eye(4, dtype=complex)


def prepare_angles(coef):
    """実非負振幅に対する2量子ビットPREPAREのRy角"""
    v = np.sqrt(coef / coef.sum())
    theta_a = 2 * np.arctan2(np.linalg.norm(v[2:]), np.linalg.norm(v[:2]))
    theta_b0 = 2 * np.arctan2(v[1], v[0])
    theta_b1 = 2 * np.arctan2(v[3], v[2])
    return v, theta_a, theta_b0, theta_b1


AMP, TH_A, TH_B0, TH_B1 = prepare_angles(COEF)
TH_P, TH_M = (TH_B0 + TH_B1) / 2, (TH_B0 - TH_B1) / 2

# PREPAREを5ゲート回路として: Ry(0), Ry(1), CNOT, Ry(1), CNOT
PREP = CNOT4 @ np.kron(I2, ry(TH_M)) @ CNOT4 @ np.kron(I2, ry(TH_P)) \
       @ np.kron(ry(TH_A), I2)

print("PREPARE: 4つのRy角と、得られる振幅")
print("=" * 68)
print(f"  目標振幅 sqrt(c_l/alpha) = {np.round(AMP, 6)}")
print(f"  theta_a  = {TH_A:.6f} rad   (第1補助量子ビットへのRy)")
print(f"  theta_b0 = {TH_B0:.6f} rad,  theta_b1 = {TH_B1:.6f} rad"
      "   (多重化Ry)")
print(f"  第2補助量子ビット上で Ry({TH_P:.6f}) . CNOT . Ry({TH_M:.6f})"
      " . CNOT にコンパイルされます")
print(f"  PREP|00> = {np.round(PREP[:, 0].real, 6)}")
print(f"  目標からの最大偏差 = "
      f"{np.max(np.abs(PREP[:, 0] - AMP)):.2e}")
print(f"  PREPはユニタリか? {np.allclose(PREP.conj().T @ PREP, I4)}"
      f"    実行列か? {np.allclose(PREP.imag, 0)}")

# SELECTを4つの2重制御Pauliの積として構成します
CC = []
for l, s in enumerate(STRINGS):
    proj = np.outer(E4[l], E4[l])
    CC.append(np.eye(16, dtype=complex)
              - np.kron(proj, I4) + np.kron(proj, pauli_matrix(s)))
SELECT = reduce(lambda a, b: a @ b, CC)
SELECT_direct = sum(np.kron(np.outer(E4[l], E4[l]), pauli_matrix(STRINGS[l]))
                    for l in range(4))

print("\nSELECT: 4つの2重制御Pauliの積")
print("-" * 68)
for l, s in enumerate(STRINGS):
    print(f"  補助 |{l//2}{l%2}>  ->  系に {s} を作用")
print(f"  4つのCC-Pauliの積 == sum_l |l><l| (x) P_l か? "
      f"{np.allclose(SELECT, SELECT_direct)}")
print(f"  SELECTはエルミートか? {np.allclose(SELECT, SELECT.conj().T)}"
      f"   SELECT^2 = I か? {np.allclose(SELECT @ SELECT, np.eye(16))}")

U_A = np.kron(PREP.conj().T, I4) @ SELECT @ np.kron(PREP, I4)

print("\nブロック符号化 U_A = (PREP^dag (x) I) SELECT (PREP (x) I)")
print("-" * 68)
print(f"  U_A は 16 x 16、ユニタリか? "
      f"{np.allclose(U_A.conj().T @ U_A, np.eye(16))}")
print(f"  U_A はエルミートか（したがって鏡映か）? "
      f"{np.allclose(U_A, U_A.conj().T)}")
print("\n  U_A の左上 4 x 4 ブロック（実部）:")
for row in np.round(U_A[:4, :4].real, 6):
    print("     ", "  ".join(f"{x:+.6f}" for x in row))
print("\n  H / alpha（実部）:")
for row in np.round((Hmat / ALPHA).real, 6):
    print("     ", "  ".join(f"{x:+.6f}" for x in row))
print(f"\n  max |U_A[:4,:4] - H/alpha| = "
      f"{np.max(np.abs(U_A[:4, :4] - Hmat/ALPHA)):.3e}"
      "   <- これが定義そのものです")
print(f"  左上ブロックのスペクトルノルム = "
      f"{np.linalg.norm(U_A[:4, :4], ord=2):.6f}"
      f"  (= ||H||/alpha = {np.linalg.norm(Hmat, ord=2)/ALPHA:.6f})")
print("  他のブロックは小さくありません。ユニタリ性の残りの予算を担っています。")
print(f"  例: ||U_A[:4,4:]|| = {np.linalg.norm(U_A[:4, 4:], ord=2):.6f}")

# シミュレータ上での事後選択
rng = np.random.default_rng(11)
psi_sys = rng.normal(size=4) + 1j * rng.normal(size=4)
psi_sys /= np.linalg.norm(psi_sys)
full = np.kron(ket('00'), psi_sys)
out = apply_gate(full, U_A, [0, 1, 2, 3], 4)
branch = out[:4]
p00 = float(np.vdot(branch, branch).real)
target = Hmat @ psi_sys / ALPHA

print("\nシミュレータで実行: 補助量子ビットを |00> に事後選択します")
print("-" * 68)
print(f"  P(補助 = 00)                 = {p00:.8f}")
print(f"  ||H|psi>||^2 / alpha^2       = "
      f"{np.linalg.norm(Hmat @ psi_sys)**2 / ALPHA**2:.8f}")
print(f"  max |分枝 - H|psi>/alpha|    = "
      f"{np.max(np.abs(branch - target)):.3e}")
print("  つまり生き残る分枝はちょうど H|psi>/alpha（未規格化）です。")

print("\nこのSELECTのToffoli計算")
print("-" * 68)
L = len(STRINGS)
print(f"  項数 L = {L}、係数レジスタ = "
      f"{int(np.ceil(np.log2(L)))} 量子ビット")
print(f"  素朴な構成: 1項あたり2重制御Pauli 1個、各2 Toffoli"
      f"  -> {2*L} Toffoli")
print(f"  ユナリ反復（標準的な構成）: SELECT全体で L - 1 = {L-1}"
      " Toffoli")
print("  L個の振幅のPREPAREはQROMで O(L) Toffoli、QROAMで O(sqrt(L))"
      " です。")
```

```text
PREPARE: 4つのRy角と、得られる振幅
====================================================================
  目標振幅 sqrt(c_l/alpha) = [0.408248 0.527046 0.333333 0.666667]
  theta_a  = 1.682137 rad   (第1補助量子ビットへのRy)
  theta_b0 = 1.823477 rad,  theta_b1 = 2.214297 rad   (多重化Ry)
  第2補助量子ビット上で Ry(2.018887) . CNOT . Ry(-0.195410) . CNOT にコンパイルされます
  PREP|00> = [0.408248 0.527046 0.333333 0.666667]
  目標からの最大偏差 = 1.11e-16
  PREPはユニタリか? True    実行列か? True

SELECT: 4つの2重制御Pauliの積
--------------------------------------------------------------------
  補助 |00>  ->  系に IX を作用
  補助 |01>  ->  系に XI を作用
  補助 |10>  ->  系に YY を作用
  補助 |11>  ->  系に ZZ を作用
  4つのCC-Pauliの積 == sum_l |l><l| (x) P_l か? True
  SELECTはエルミートか? True   SELECT^2 = I か? True

ブロック符号化 U_A = (PREP^dag (x) I) SELECT (PREP (x) I)
--------------------------------------------------------------------
  U_A は 16 x 16、ユニタリか? True
  U_A はエルミートか（したがって鏡映か）? True

  U_A の左上 4 x 4 ブロック（実部）:
      +0.444444  +0.166667  +0.277778  -0.111111
      +0.166667  -0.444444  +0.111111  +0.277778
      +0.277778  +0.111111  -0.444444  +0.166667
      -0.111111  +0.277778  +0.166667  +0.444444

  H / alpha（実部）:
      +0.444444  +0.166667  +0.277778  -0.111111
      +0.166667  -0.444444  +0.111111  +0.277778
      +0.277778  +0.111111  -0.444444  +0.166667
      -0.111111  +0.277778  +0.166667  +0.444444

  max |U_A[:4,:4] - H/alpha| = 8.327e-17   <- これが定義そのものです
  左上ブロックのスペクトルノルム = 0.566558  (= ||H||/alpha = 0.566558)
  他のブロックは小さくありません。ユニタリ性の残りの予算を担っています。
  例: ||U_A[:4,4:]|| = 0.831479

シミュレータで実行: 補助量子ビットを |00> に事後選択します
--------------------------------------------------------------------
  P(補助 = 00)                 = 0.31090298
  ||H|psi>||^2 / alpha^2       = 0.31090298
  max |分枝 - H|psi>/alpha|    = 1.130e-16
  つまり生き残る分枝はちょうど H|psi>/alpha（未規格化）です。

このSELECTのToffoli計算
--------------------------------------------------------------------
  項数 L = 4、係数レジスタ = 2 量子ビット
  素朴な構成: 1項あたり2重制御Pauli 1個、各2 Toffoli  -> 8 Toffoli
  ユナリ反復（標準的な構成）: SELECT全体で L - 1 = 3 Toffoli
  L個の振幅のPREPAREはQROMで O(L) Toffoli、QROAMで O(sqrt(L)) です。
```

**注目すべき点。** $U_A$ の左上 $4 \times 4$ ブロックと行列 $H/\alpha$ が上下に印字されており、成分ごとに $8 \times 10^{-17}$ で一致しています。この構成のどこにも近似はありません。ブロック符号化は厳密であり、その代価は補助レジスタ1つと因子 $\alpha$ です。

逆に取り違えやすい細部が2つあり、立ち止まる価値があります。第一に、非対角ブロックのスペクトルノルムは $0.831$ で、小さくはありません。ブロック符号化は $U_A$ の不要な部分を無視できるほど小さくするのではなく、事後選択や鏡映で*扱える*ようにするのです。第二に、$U_A$ はエルミートに、したがって $U_A^2 = I$ になりました。これは鏡映です。この例に特有の偶然ではありません。SELECTが自己逆（Pauliの積はそうです）でPREPAREが実行列であれば必ず起こり、次に出てくるウォーク作用素はこの性質に依存します。

最後のブロックは、行列代数ではなくシミュレータの上で符号化を走らせます。$\lvert 00 \rangle \otimes \lvert \psi \rangle$ を用意し、$16 \times 16$ のゲートを `apply_gate` で4量子ビットすべてに作用させ、先頭4振幅を見ます。それはちょうど $H \lvert \psi \rangle / \alpha$（未規格化）であり、その二乗ノルムは補助の測定が $00$ を返す確率です。この一文が、ブロック符号化の操作的な意味のすべてです。

### 成功確率、そして $\alpha$ こそ恐れるべき数である理由

事後選択の成功確率は

$$ P(0^{m}) = \frac{\lVert H \lvert \psi \rangle \rVert^{2}}{\alpha^{2}} $$

であり、規格化された状態に対しては高々 $\lVert H \rVert^2/\alpha^2$ です。1ノルムが噛みつくのはここです。振幅増幅（第1章）は $O(1/\sqrt{P}) = O(\alpha / \lVert H \lvert \psi \rangle \rVert)$ ラウンドの代価で確率を固定するので、$\alpha$ はあらゆるコスト式に線形に入ります。そして $\alpha$ は項についての和なのです。

### Code Example 4: LCUの成功確率と、$\alpha$ が要求する代価

```python
"""第4章 Code Example 4: LCUの成功確率と、alpha が要求する代価。
Code Example 3 の続き（同一セッション）。"""
print("LCU 1ラウンドの成功確率: P = ||H|psi>||^2 / alpha^2")
print("=" * 68)


def lcu_success(psi):
    full = np.kron(ket('00'), psi)
    out = apply_gate(full, U_A, [0, 1, 2, 3], 4)
    return float(np.vdot(out[:4], out[:4]).real)


w_H, v_H = np.linalg.eigh(Hmat)
print(f"{'state':>26} {'P(00) simulated':>17} {'||H psi||^2/a^2':>17}")
cases = [('|00>', ket('00')), ('|01>', ket('01')),
         ('|++>', apply_gate(apply_gate(ket('00'), H, [0], 2), H, [1], 2))]
for k in range(4):
    cases.append((f'eigenstate E = {w_H[k]:+.4f}', v_H[:, k].astype(complex)))
for name, st in cases:
    print(f"{name:>26} {lcu_success(st):17.8f} "
          f"{np.linalg.norm(Hmat@st)**2/ALPHA**2:17.8f}")

rng = np.random.default_rng(4)
ps = []
for _ in range(20000):
    z = rng.normal(size=4) + 1j * rng.normal(size=4)
    z /= np.linalg.norm(z)
    ps.append(np.linalg.norm(Hmat @ z) ** 2 / ALPHA ** 2)
ps = np.array(ps)
print(f"\n  Haarランダム状態20000回: 平均 P = {ps.mean():.6f},"
      f" 最小 {ps.min():.6f}, 最大 {ps.max():.6f}")
print(f"  解析的な平均 Tr(H^2)/(d alpha^2) = "
      f"{np.trace(Hmat@Hmat).real/(4*ALPHA**2):.6f}")

print("\n振幅増幅: ほぼ確実な成功に必要なラウンド数")
print("-" * 68)
print(f"{'p (one round)':>14} {'rounds ~ pi/(4 arcsin sqrt p)':>31}"
      f" {'P after':>9}")
for p in (0.5, 0.31, 0.1, 0.03, 0.01, 1e-3):
    th = np.arcsin(np.sqrt(p))
    k = int(np.round((np.pi / 2 - th) / (2 * th)))
    print(f"{p:14.4f} {k:31d} {np.sin((2*k+1)*th)**2:9.5f}")
print("  ラウンド数は 1/sqrt(p) = alpha/||H|psi>|| として増えます。")
print("  これがqubitizationのあらゆるコスト式に現れる alpha であり、"
      "スペクトルノルムではなく1ノルムです。")

print("\nalpha を心配すべき理由: Heisenberg鎖")
print("-" * 68)
print(f"{'sites n':>8} {'terms L':>8} {'alpha (1-norm)':>15}"
      f" {'||H||':>10} {'alpha/||H||':>12}")
for n in range(2, 11):
    ch = {}
    for i in range(n - 1):
        for P in 'XYZ':
            s = ''.join(P if q in (i, i + 1) else 'I' for q in range(n))
            ch[s] = 1.0
    a = sum(abs(c) for c in ch.values())
    Hc = to_matrix(ch)
    nrm = np.linalg.norm(Hc, ord=2)
    print(f"{n:8d} {len(ch):8d} {a:15.2f} {nrm:10.4f} {a/nrm:12.4f}")
print("  alpha は項数に比例して増え、||H|| はそれより遅く増えるので、")
print("  比は上へ流れていきます。LCUのオーバーヘッドは実在し、"
      "しかも示量的です。")
```

```text
LCU 1ラウンドの成功確率: P = ||H|psi>||^2 / alpha^2
====================================================================
                     state   P(00) simulated   ||H psi||^2/a^2
                      |00>        0.31481481        0.31481481
                      |01>        0.31481481        0.31481481
                      |++>        0.30864198        0.30864198
    eigenstate E = -1.0198        0.32098765        0.32098765
    eigenstate E = -1.0000        0.30864198        0.30864198
    eigenstate E = +1.0000        0.30864198        0.30864198
    eigenstate E = +1.0198        0.32098765        0.32098765

  Haarランダム状態20000回: 平均 P = 0.314835, 最小 0.308670, 最大 0.320936
  解析的な平均 Tr(H^2)/(d alpha^2) = 0.314815

振幅増幅: ほぼ確実な成功に必要なラウンド数
--------------------------------------------------------------------
 p (one round)   rounds ~ pi/(4 arcsin sqrt p)   P after
        0.5000                               0   0.50000
        0.3100                               1   0.96026
        0.1000                               2   0.99856
        0.0300                               4   0.99998
        0.0100                               7   0.99534
        0.0010                              24   0.99956
  ラウンド数は 1/sqrt(p) = alpha/||H|psi>|| として増えます。
  これがqubitizationのあらゆるコスト式に現れる alpha であり、スペクトルノルムではなく1ノルムです。

alpha を心配すべき理由: Heisenberg鎖
--------------------------------------------------------------------
 sites n  terms L  alpha (1-norm)      ||H||  alpha/||H||
       2        3            3.00     3.0000       1.0000
       3        6            6.00     4.0000       1.5000
       4        9            9.00     6.4641       1.3923
       5       12           12.00     7.7115       1.5561
       6       15           15.00     9.9743       1.5039
       7       18           18.00    11.3450       1.5866
       8       21           21.00    13.4997       1.5556
       9       24           24.00    14.9453       1.6059
      10       27           27.00    17.0321       1.5852
  alpha は項数に比例して増え、||H|| はそれより遅く増えるので、
  比は上へ流れていきます。LCUのオーバーヘッドは実在し、しかも示量的です。
```

**注目すべき点。** シミュレーションした確率と閉形式 $\lVert H \lvert \psi \rangle \rVert^2/\alpha^2$ は、試したすべての状態で印字桁数すべてにわたり一致します。固有状態では確率が $(E_k/\alpha)^2$ に帰着することも含めてです。20 000個のランダム状態にわたるHaar平均は、当然そうならなければならないとおり $\mathrm{Tr}(H^2)/(d\,\alpha^2)$ に乗ります。

覚えておくべきはHeisenberg鎖の表です。1ノルムは項数にちょうど比例して増え（結合あたり3項なので $3(n-1)$）、スペクトルノルムはそれより遅く増えるので、比 $\alpha/\lVert H \rVert$ は2サイトの1.0から10サイトの約1.6へ流れていきます。分子ハミルトニアンではこの比はもっと悪くなります。$\alpha$ は $O(M^4)$ 個の積分の、打ち消しのない和である一方、$\lVert H \rVert$ には打ち消しがたっぷりあるからです。したがって2電子テンソルを再因子分解して $\alpha$ を下げる作業は化粧ではありません。誤り耐性量子化学のコストを動かす*その*てこであり、4.4節がその算術を示します。

### qubitization: 鏡映からウォークへ

事後選択と振幅増幅は $H$ を1回作用させるには十分です。$e^{-iHt}$ をシミュレートするにはもっと良いものが必要で、それを与えてくれるのが次の観察です。$\Pi = \lvert 0^m \rangle\langle 0^m \rvert \otimes I$ を良い補助部分空間への射影とし、**量子ウォーク作用素**

$$ W = \left(2\Pi - I\right) U_A $$

を定義します。$U_A$ 自身が鏡映であるとき、これは2つの鏡映の積です。2つの鏡映の積は回転であり、空間は2次元の不変部分空間 — $H$ の固有ベクトル $\lvert E_k \rangle$ ごとに1つ — に分解し、その内部で $W$ は重なりが決める角度だけ回転します。結果がqubitizationの中心的な恒等式です：

$$ \text{関係する部分空間上の } W \text{ の固有値} = e^{\pm i \theta_k}, \qquad \cos\theta_k = \frac{E_k}{\alpha} $$

つまり $U_A$ への1回の呼び出し — PREPARE 1回、SELECT 1回、PREPARE$^\dagger$ 1回、それに $m$ 補助量子ビット上の鏡映 — が、$H$ のスペクトルの既知かつ可逆な関数をスペクトルとしてもつウォークを1歩進めます。したがって $W$ 上の位相推定は $\theta_k$ を返し、$E_k = \alpha \cos\theta_k$ がエネルギーを復元します。$\arccos$ はバンド端付近での誤差伝播にとって厄介ですが、それ以上のものではありません。

これは見た目より強い主張です。固有値を精度 $\varepsilon$ で推定するには $U_A$ への $O(\alpha/\varepsilon)$ 回の呼び出しが必要で、これは*最適*です。$U_A$ をブラックボックスとするモデルで、これより良いアルゴリズムは存在しません。同じウォークから量子信号処理で組み立てた時間発展は $O(\alpha t + \log(1/\varepsilon))$ クエリで、これも最適です。そしてこの対数こそ要点で、正しい桁数を倍にするのに要するのは因子ではなく定数の追加です。

### Code Example 5: qubitizationウォークと、その上での位相推定

```python
"""第4章 Code Example 5: qubitizationウォークと、その上での位相推定。
Code Example 4 の続き（同一セッション）。"""
PI0 = np.kron(np.outer(E4[0], E4[0]), I4)
REFL = 2 * PI0 - np.eye(16)
WALK = REFL @ U_A

print("qubitizationウォーク W = (2|00><00| (x) I - I) . U_A")
print("=" * 68)
print(f"  W はユニタリか? {np.allclose(WALK.conj().T @ WALK, np.eye(16))}")
ev = np.linalg.eigvals(WALK)
phases = np.angle(ev)
order = np.argsort(phases)
print(f"\n{'eigenphase theta':>18} {'cos(theta)':>12} {'alpha cos(theta)':>18}")
for k in order:
    print(f"{phases[k]:18.9f} {np.cos(phases[k]):12.6f} "
          f"{ALPHA*np.cos(phases[k]):18.6f}")
inner = np.sort(np.unique(np.round(np.cos(phases[np.abs(np.cos(phases)) < 0.999]), 9)))
print(f"\n  +-1 以外の相異なる cos(theta) : {inner}")
print(f"  H/alpha の固有値              : "
      f"{np.round(np.linalg.eigvalsh(Hmat/ALPHA), 9)}")
print(f"  最大の食い違い                : "
      f"{np.max(np.abs(inner - np.linalg.eigvalsh(Hmat/ALPHA))):.3e}")
print("  +-1 の固有値は直交する「ジャンク」部分空間に属します。そこでは W は")
print("  単なる鏡映として働き、スペクトルの情報を運びません。")

# --- ウォーク作用素の上での位相推定 -----------------------------------
def qft_matrix(m, inverse=False):
    """2^m x 2^m の密なQFT（ビッグエンディアン、最上位量子ビットが先頭）"""
    N = 2 ** m
    j = np.arange(N)
    sign = 1.0 if not inverse else -1.0
    F = np.exp(sign * 2j * np.pi * np.outer(j, j) / N) / np.sqrt(N)
    return F


def qpe_on_walk(m, sys_state):
    """4量子ビットのウォーク作用素 W に対する、補助 m 量子ビットの標準QPE"""
    n = m + 4
    psi = np.kron(ket('0' * m), np.kron(ket('00'), sys_state))
    for j in range(m):
        psi = apply_gate(psi, H, [j], n)
    for j in range(m):
        power = 2 ** (m - 1 - j)
        Wp = np.linalg.matrix_power(WALK, power)
        cW = np.eye(32, dtype=complex)
        cW[16:, 16:] = Wp
        psi = apply_gate(psi, cW, [j, m, m + 1, m + 2, m + 3], n)
    psi = apply_gate(psi, qft_matrix(m, inverse=True), list(range(m)), n)
    pr = probs(psi).reshape(2 ** m, 16).sum(axis=1)
    return pr


gs = v_H[:, 0].astype(complex)
print("\nW 上の位相推定、系レジスタは基底状態")
print("-" * 68)
print(f"{'m (ancillas)':>13} {'peak readout':>14} {'prob':>8}"
      f" {'theta':>10} {'alpha cos theta':>16} {'error':>10}")
for m in (6, 8, 10):
    pr = qpe_on_walk(m, gs)
    k = int(np.argmax(pr))
    th = 2 * np.pi * k / 2 ** m
    est = ALPHA * np.cos(th)
    print(f"{m:13d} {format(k, f'0{m}b'):>14} {pr[k]:8.4f} {th:10.6f}"
          f" {est:16.6f} {abs(est - w_H[0]):10.2e}")
print(f"  厳密な基底状態エネルギー = {w_H[0]:.6f}")
pr = qpe_on_walk(8, gs)
top = np.argsort(pr)[::-1][:4]
print("\n  m = 8 での上位4ピーク。2ピーク構造が見えます:")
for k in sorted(top):
    th = 2 * np.pi * k / 2 ** 8
    print(f"    {format(k, '08b')}  p = {pr[k]:.6f}"
          f"  theta = {th:.6f}  alpha cos theta = {ALPHA*np.cos(th):+.6f}")
print("  |00>|E_k> は位相 +theta_k と -theta_k をもつ2つの固有ベクトルの")
print("  等重率混合であり、cos は偶関数なので両ピークが同じエネルギーを与えます。")
print("  小さい m での残差は、真の位相のどちら側に最近接格子点が落ちるかで決まり、")
print("  ウォーク作用素のせいではありません。")

print("\nクエリ計算量: 1クエリで何が買えるか")
print("-" * 68)
print(f"{'target error':>13} {'Trotter-1 rot.':>15} {'Trotter-2 rot.':>15}"
      f" {'qubitization queries':>21}")
for eps in (1e-3, 1e-6, 1e-9):
    r1, r2 = C1 / eps, np.sqrt(C2 / eps)
    q = np.pi * ALPHA / eps / 2
    print(f"{eps:13.0e} {r1*4:15.3e} {r2*8:15.3e} {q:21.3e}")
print("  Trotterは次数で決まる指数の poly(1/eps) を要します。")
print("  qubitizationは時間発展に O(alpha t + log(1/eps))、固有値そのものに")
print("  O(alpha/eps) を要し、クエリモデルで最適です。")
```

```text
qubitizationウォーク W = (2|00><00| (x) I - I) . U_A
====================================================================
  W はユニタリか? True

  eigenphase theta   cos(theta)   alpha cos(theta)
      -2.173118745    -0.566558          -1.019804
      -2.159827297    -0.555556          -1.000000
      -0.981765357     0.555556           1.000000
      -0.968473908     0.566558           1.019804
      -0.000000000     1.000000           1.800000
      -0.000000000     1.000000           1.800000
       0.000000000     1.000000           1.800000
       0.000000000     1.000000           1.800000
       0.968473908     0.566558           1.019804
       0.981765357     0.555556           1.000000
       2.159827297    -0.555556          -1.000000
       2.173118745    -0.566558          -1.019804
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000
       3.141592654    -1.000000          -1.800000

  +-1 以外の相異なる cos(theta) : [-0.56655772 -0.55555556  0.55555556  0.56655772]
  H/alpha の固有値              : [-0.56655772 -0.55555556  0.55555556  0.56655772]
  最大の食い違い                : 4.444e-10
  +-1 の固有値は直交する「ジャンク」部分空間に属します。そこでは W は
  単なる鏡映として働き、スペクトルの情報を運びません。

W 上の位相推定、系レジスタは基底状態
--------------------------------------------------------------------
 m (ancillas)   peak readout     prob      theta  alpha cos theta      error
            6         101010   0.4707   4.123340        -1.000026   1.98e-02
            8       01011001   0.2364   2.184389        -1.036455   1.67e-02
           10     1010011110   0.4576   4.111069        -1.018317   1.49e-03
  厳密な基底状態エネルギー = -1.019804

  m = 8 での上位4ピーク。2ピーク構造が見えます:
    01011000  p = 0.170385  theta = 2.159845  alpha cos theta = -1.000026
    01011001  p = 0.236359  theta = 2.184389  alpha cos theta = -1.036455
    10100111  p = 0.236359  theta = 4.098797  alpha cos theta = -1.036455
    10101000  p = 0.170385  theta = 4.123340  alpha cos theta = -1.000026
  |00>|E_k> は位相 +theta_k と -theta_k をもつ2つの固有ベクトルの
  等重率混合であり、cos は偶関数なので両ピークが同じエネルギーを与えます。
  小さい m での残差は、真の位相のどちら側に最近接格子点が落ちるかで決まり、
  ウォーク作用素のせいではありません。

クエリ計算量: 1クエリで何が買えるか
--------------------------------------------------------------------
 target error  Trotter-1 rot.  Trotter-2 rot.  qubitization queries
        1e-03       1.616e+03       7.971e+01             2.827e+03
        1e-06       1.616e+06       2.521e+03             2.827e+06
        1e-09       1.616e+09       7.971e+04             2.827e+09
  Trotterは次数で決まる指数の poly(1/eps) を要します。
  qubitizationは時間発展に O(alpha t + log(1/eps))、固有値そのものに
  O(alpha/eps) を要し、クエリモデルで最適です。
```

**注目すべき点。** 固有位相の表が主張そのものであり、検証済みです。$W$ の16個の固有値のうち8個が4組の $\pm\theta$ の対をなし、その余弦が $H/\alpha$ の4つの固有値を $4 \times 10^{-10}$ で再現します。残差は非エルミート固有値ソルバーのもので、構成のものではありません。残る8個は $\theta = 0$ と $\theta = \pi$ に4個ずつあります。これがジャンク部分空間で、そこでは $W$ は単なる鏡映として働き、$H$ についての情報を運びません。

位相推定のブロックは、制御時間発展を $W$ に置き換えただけの第2章と同じ回路であり、特徴的な2ピーク構造を示します。$\lvert 0^m \rangle \lvert E_k \rangle$ は位相 $\pm\theta_k$ をもつ $W$ の2つの固有ベクトルの等重率重ね合わせなので、読み出しは $k$ と $2^m - k$ にピークをもちます。$\cos$ が偶関数なので両方が同じエネルギーを与え、解消すべき符号の曖昧さはありません。$e^{-iHt}$ に対する教科書的QPEでは位相が巻き付いてしまうのと対照的です。

最後のクエリ計算量の表は、この節が存在する理由そのものである比較です。$\varepsilon = 10^{-9}$ で2次の積公式は $8.0 \times 10^{4}$ 回転、qubitizationは $2.8 \times 10^{9}$ クエリ。この4次元のおもちゃではqubitizationが大差で*負けます*。これはどちらの手法の誤りでもありません。$O(\alpha/\varepsilon)$ の数値は状態準備の近道なしに固有値を固定精度で得るコストであり、Trotterの数値は発展時間1単位分のものです。加えてここでは $\alpha = 1.8$ ですが、実際のハミルトニアンでは $\alpha$ は数百です。qubitizationの漸近的な優位は*発展*における $t$ と $\log(1/\varepsilon)$、そしてウォークコストの $L$ 非依存性にあります。4次元には漸近すべきものが何もないのです。

### 量子信号処理を1段落で

以上のすべては一般化されます。$H/\alpha$ のブロック符号化が与えられたとき、$U_A$ ともう1つの補助量子ビット上の単一量子ビット回転を交互に入れると、$[-1,1]$ 上で1に抑えられた本質的に任意の $d$ 次多項式 $p$ について $p(H/\alpha)$ のブロック符号化が $d$ 回のクエリで実装できます。$p$ を $\cos(\alpha t x)$ と $\sin(\alpha t x)$ の近似に選べばハミルトニアンシミュレーション、スペクトルギャップ上で $p \approx 1/x$ に選べば行列反転、階段関数に選べば基底状態への射影になります。qubitizationは多項式がChebyshev多項式である特別な場合です。$\cos(d \arccos x) = T_d(x)$ なので、ウォークの固有位相が $\arccos$ で出てきたのはまさにそのためです。この統一は細部が本章の範囲を超えていても知っておく価値があります。「規格化 $\alpha$ のブロック符号化がある」ということが、FTQCアルゴリズム設計者に必要な唯一のインターフェースだという意味だからです。

| | Trotter / Suzuki | qDRIFT | qubitization / QSP |
| --- | --- | --- | --- |
| 作用させるもの | $e^{-iH_j \delta}$、決定的に | $e^{-i\lambda\tau P_l}$、ランダム抽出で | $U_A$、$H/\alpha$ のブロック符号化 |
| 補助量子ビット | 不要 | 不要 | $\lceil \log_2 L \rceil$ ＋配線 |
| $\varepsilon$ 依存 | $\varepsilon^{-1/2k}$ | $\varepsilon^{-1}$ | $\log(1/\varepsilon)$ |
| $L$ 依存 | ステップあたり線形 | なし | 1項あたりToffoli 1個、1度だけ |
| 現れるノルム | 交換子の和 | $\lambda = \sum \lvert c_l \rvert$ | $\alpha = \sum \lvert c_l \rvert$ |
| 出力 | ユニタリ | チャネル（混合状態） | ユニタリ |
| 実務上の適所 | ダイナミクス、項数が少ない場合 | 非常に多数の小さな項 | 固有値、FTQC見積り |

* * *

## 4.3 qDRIFT: ランダム化コンパイル

### チャネル

qDRIFTは、回路が $e^{-iHt}$ をユニタリとして近似すべきだという考えを放棄します。代わりに、*平均*が正しい発展に近くなるランダム回路を定義します。項の添字 $l$ を確率 $p_l = \lvert c_l \rvert / \lambda$（ただし $\lambda = \sum_l \lvert c_l \rvert$）で抽出し、単一の回転

$$ V_l = \exp\left(-i\,\mathrm{sgn}(c_l)\,\frac{\lambda t}{N}\,P_l\right) $$

を作用させ、これを $N$ 回繰り返します。すべての回転は $c_l$ の大小に関わらず*同じ*角度 $\lambda t / N$ をもち、係数は項が選ばれる頻度としてのみ入ります。ランダム性について平均を取ると量子チャネルになり、その1ステップは

$$ \mathcal{E}_1(\rho) = \sum_l \frac{\lvert c_l \rvert}{\lambda}\, V_l \rho V_l^{\dagger}, \qquad \mathcal{E} = \mathcal{E}_1^{N} $$

です。誤差限界は

$$ \left\lVert \mathcal{E} - e^{-iHt}(\cdot)e^{iHt} \right\rVert_{\diamond} \le O\left(\frac{\lambda^{2} t^{2}}{N}\right) $$

であり、際立った特徴は*欠けているもの*です。$L$ がなく、交換子もありません。与えられた精度に必要なゲート数は、ハミルトニアンが何項もつかに依存しないのです。1ノルムが同じなら、100万個の小さな項をもつハミルトニアンと4項のハミルトニアンでqDRIFTのコストは同じになります。

### どこで役立ち、どこで役立たないか

取引の内容は、Trotterの $1/r^{2}$ 以上に対する $1/N$ 誤差を、ステップあたりの $L$ の因子を落とすことで買うというものです。$r$ ステップの2次Trotterは $2Lr$ 回転を要して誤差は $\sim C_2/r^2$、$N$ 回転のqDRIFTは誤差 $\sim 2\lambda^2 t^2/N$ です。ゲート数を等しく置くと、qDRIFTが勝つのは

$$ \frac{2\lambda^{2}t^{2}}{G} < \frac{4L^{2}C_2'\,t^{3}}{G^{2}} \quad \Longleftrightarrow \quad G < \frac{2L^{2}C_2' t}{\lambda^{2}} $$

のときです。つまりqDRIFTが良い選択なのは*粗い*精度と*大きな*項数のときで、予算が十分大きくなれば2次Trotterが追い越します。どちらの言明も検証可能で、しかもランダム化アルゴリズムとしては珍しく、サンプリング誤差なしに検証できます。qDRIFTチャネルはスーパー作用素であり、1ステップは $d^2 \times d^2$ 行列、$N$ ステップはその $N$ 乗です。以下のすべては厳密です。

用いる距離は $\tfrac{1}{2}\lVert J(\mathcal{E}_1) - J(\mathcal{E}_2) \rVert_{1}$、すなわち規格化したChoi行列間のトレース距離です。ダイヤモンドノルムではありませんが、基底に依らず、厳密に計算でき、そしてユニタリなTrotter回路と混合状態のqDRIFTチャネルを同じ土俵で扱えます。それが要点です。ユニタリの誤差とチャネルの誤差を他のどんな経路で比べても、りんごとみかんの間違いを招くからです。

### Code Example 6: 同一ゲート数でのTrotterとqDRIFT

```python
"""第4章 Code Example 6: 同一ゲート数でのTrotterとqDRIFTの比較。
Code Example 5 の続き（同一セッション）。"""
def superop(kraus):
    """列積みベクトル化におけるスーパー作用素 sum_k conj(K) (x) K"""
    return sum(np.kron(K.conj(), K) for K in kraus)


def choi(S, d):
    """d x d 行列に作用するスーパー作用素の、規格化されたChoi行列"""
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d), dtype=complex)
            E[i, j] = 1.0
            out = (S @ E.reshape(-1, order='F')).reshape(d, d, order='F')
            J[i*d:(i+1)*d, j*d:(j+1)*d] = out
    return J / d


def choi_distance(S1, S2, d):
    """(1/2)||J1 - J2||_1。厳密かつ基底に依らないチャネル間距離です"""
    D = choi(S1, d) - choi(S2, d)
    return 0.5 * float(np.abs(np.linalg.eigvalsh((D + D.conj().T) / 2)).sum())


def trotter_unitary(terms, t, r, order=1):
    step = np.eye(2 ** len(terms[0][0]), dtype=complex)
    dt = t / r
    if order == 1:
        for s, c in terms:
            step = expm_hermitian(to_matrix({s: c}), -1j * dt) @ step
    else:
        for s, c in terms:
            step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
        for s, c in reversed(terms):
            step = expm_hermitian(to_matrix({s: c}), -1j * dt / 2) @ step
    return np.linalg.matrix_power(step, r)


def qdrift_superop(terms, t, N):
    """厳密なqDRIFTチャネル: N回のi.i.d.抽出を、スーパー作用素として評価します"""
    lam = sum(abs(c) for _, c in terms)
    d = 2 ** len(terms[0][0])
    kraus = []
    for s, c in terms:
        p = abs(c) / lam
        V = expm_hermitian(to_matrix({s: np.sign(c)}), -1j * lam * t / N)
        kraus.append(np.sqrt(p) * V)
    S1 = superop(kraus)
    return np.linalg.matrix_power(S1, N)


print("TrotterとqDRIFTをチャネルとして比較します（厳密、サンプリングなし）")
print("=" * 68)
print("qDRIFTチャネルは E_1^N で、E_1(rho) = sum_l (|c_l|/lambda)"
      " V_l rho V_l^dag、")
print("V_l = exp(-i sign(c_l) lambda t P_l / N) です。スーパー作用素の合成で")
print("厳密に評価できるので、以下の数値にMonte-Carlo誤差は一切入りません。")

lam4 = sum(abs(c) for _, c in terms)
S_exact4 = superop([U_exact])
print(f"\nハミルトニアンA: L = {len(terms)} 項, lambda = {lam4:.3f},"
      f" t = {tau}")
print(f"{'N (samples)':>12} {'channel error':>15} {'error x N':>11}")
for N in (4, 8, 16, 32, 64, 128, 256, 512):
    e = choi_distance(qdrift_superop(terms, tau, N), S_exact4, 4)
    print(f"{N:12d} {e:15.6e} {e*N:11.5f}")
print("  誤差 x N は収束します。qDRIFTは O(lambda^2 t^2 / N) で、L に依りません。")

print("\n同一ゲート数での比較、ハミルトニアンA (L = 4)")
print(f"{'rotations G':>12} {'Trotter-1 (r=G/L)':>19} {'Trotter-2':>13}"
      f" {'qDRIFT (N=G)':>14} {'winner':>12}")
for G in (16, 32, 64, 128, 256, 512):
    r1 = G // len(terms)
    e1 = choi_distance(superop([trotter_unitary(terms, tau, r1, 1)]),
                       S_exact4, 4)
    r2 = max(1, G // (2 * len(terms)))
    e2 = choi_distance(superop([trotter_unitary(terms, tau, r2, 2)]),
                       S_exact4, 4)
    eq = choi_distance(qdrift_superop(terms, tau, G), S_exact4, 4)
    best = min((e1, 'Trotter-1'), (e2, 'Trotter-2'), (eq, 'qDRIFT'))[1]
    print(f"{G:12d} {e1:19.6e} {e2:13.6e} {eq:14.6e} {best:>12}")

# パートC: lambda と予算を固定し、項数だけを増やします
import itertools

rngB = np.random.default_rng(20260813)
ALL_P4 = [''.join(p) for p in itertools.product('IXYZ', repeat=4)][1:]
PERM = rngB.permutation(len(ALL_P4))
RAW = 1e-4 * (0.05 / 1e-4) ** rngB.random(len(ALL_P4))   # 対数一様。分子積分の
LAM0, G0 = 2.0, 1024                                     # 分布に倣います


def make_H4(L):
    """1ノルムを固定した、L項のPauli文字列からなる4量子ビットハミルトニアン"""
    idx = PERM[:L]
    c = RAW[idx]
    c = c / c.sum() * LAM0
    return {ALL_P4[i]: float(x) for i, x in zip(idx, c)}


print(f"\nパートC: 1ノルム lambda = {LAM0} と予算"
      f" G = {G0} 回転を固定、t = {tau}")
print("  変えるのは項数 L だけです。係数は対数一様分布 — 分子積分が実際に")
print("  もつ裾の重い分布です。")
print(f"{'L':>6} {'||H||':>9} {'Trotter-1':>13} {'Trotter-2':>13}"
      f" {'qDRIFT':>13} {'T2/qDRIFT':>11}")
ratios = []
for L in (4, 15, 63, 255):
    HB = make_H4(L)
    tB = sorted(HB.items())
    HmB = to_matrix(HB)
    SeB = superop([expm_hermitian(HmB, -1j * tau)])
    r1 = max(1, G0 // L)
    r2 = max(1, G0 // (2 * L))
    e1 = choi_distance(superop([trotter_unitary(tB, tau, r1, 1)]), SeB, 16)
    e2 = choi_distance(superop([trotter_unitary(tB, tau, r2, 2)]), SeB, 16)
    eq = choi_distance(qdrift_superop(tB, tau, G0), SeB, 16)
    ratios.append((L, e2 / eq))
    print(f"{L:6d} {np.linalg.norm(HmB, ord=2):9.4f} {e1:13.4e} {e2:13.4e}"
          f" {eq:13.4e} {e2/eq:11.4f}")

Ls = np.array([r[0] for r in ratios], dtype=float)
Rs = np.array([r[1] for r in ratios])
slope, icpt = np.polyfit(np.log(Ls), np.log(Rs), 1)
L_cross = np.exp(-icpt / slope)
print(f"\n  qDRIFTの誤差は L に対して平坦です（lambda だけに依ります）。"
      "Trotterの誤差は増えます。")
print(f"  フィット  error(Trotter-2)/error(qDRIFT) ~ L^{slope:.3f}"
      f"  ->  交差点は L ~ {L_cross:.3g}")
print("  上の表では2次Trotterが全行で勝っており、ハミルトニアンの項数が")
print(f"  {L_cross:.0e} 程度になるまで勝ち続けます。数百スピン軌道の分子・材料系")
print("  ハミルトニアンは 1e5 から 1e8 項をもち、それがランダム化コンパイルが")
print("  提案された領域です。そしてそこは同時に、1/eps を log(1/eps) に変える")
print("  qubitizationが両者を置き換える領域でもあります。")
```

```text
TrotterとqDRIFTをチャネルとして比較します（厳密、サンプリングなし）
====================================================================
qDRIFTチャネルは E_1^N で、E_1(rho) = sum_l (|c_l|/lambda) V_l rho V_l^dag、
V_l = exp(-i sign(c_l) lambda t P_l / N) です。スーパー作用素の合成で
厳密に評価できるので、以下の数値にMonte-Carlo誤差は一切入りません。

ハミルトニアンA: L = 4 項, lambda = 1.800, t = 1.0
 N (samples)   channel error   error x N
           4    4.103292e-01     1.64132
           8    2.359050e-01     1.88724
          16    1.275293e-01     2.04047
          32    6.645621e-02     2.12660
          64    3.394306e-02     2.17236
         128    1.715587e-02     2.19595
         256    8.624748e-03     2.20794
         512    4.324169e-03     2.21397
  誤差 x N は収束します。qDRIFTは O(lambda^2 t^2 / N) で、L に依りません。

同一ゲート数での比較、ハミルトニアンA (L = 4)
 rotations G   Trotter-1 (r=G/L)     Trotter-2   qDRIFT (N=G)       winner
          16        7.751628e-02  2.053134e-02   1.275293e-01    Trotter-2
          32        3.866575e-02  5.062782e-03   6.645621e-02    Trotter-2
          64        1.932131e-02  1.261309e-03   3.394306e-02    Trotter-2
         128        9.659206e-03  3.150531e-04   1.715587e-02    Trotter-2
         256        4.829422e-03  7.874614e-05   8.624748e-03    Trotter-2
         512        2.414688e-03  1.968546e-05   4.324169e-03    Trotter-2

パートC: 1ノルム lambda = 2.0 と予算 G = 1024 回転を固定、t = 1.0
  変えるのは項数 L だけです。係数は対数一様分布 — 分子積分が実際に
  もつ裾の重い分布です。
     L     ||H||     Trotter-1     Trotter-2        qDRIFT   T2/qDRIFT
     4    1.9341    1.7688e-04    1.7414e-06    4.2964e-04      0.0041
    15    1.1935    4.7436e-03    6.8973e-05    3.1512e-03      0.0219
    63    0.7943    7.7544e-03    2.6280e-04    3.6527e-03      0.0719
   255    0.5280    8.1246e-03    6.1149e-04    3.8532e-03      0.1587

  qDRIFTの誤差は L に対して平坦です（lambda だけに依ります）。Trotterの誤差は増えます。
  フィット  error(Trotter-2)/error(qDRIFT) ~ L^0.875  ->  交差点は L ~ 1.62e+03
  上の表では2次Trotterが全行で勝っており、ハミルトニアンの項数が
  2e+03 程度になるまで勝ち続けます。数百スピン軌道の分子・材料系
  ハミルトニアンは 1e5 から 1e8 項をもち、それがランダム化コンパイルが
  提案された領域です。そしてそこは同時に、1/eps を log(1/eps) に変える
  qubitizationが両者を置き換える領域でもあります。
```

**注目すべき点。** パートAはスケーリングを確認します。4項ハミルトニアンで誤差 $\times N$ は約2.21に収束するので、qDRIFTの誤差は確かに $O(1/N)$ で、定数は $\lambda^2 t^2 = 3.24$ の程度です。4.3節で引用した限界は $2\lambda^2t^2/N = 6.48/N$ ですから、このハミルトニアンでは限界が約3倍緩いことになります。ダイヤモンドノルムの限界をChoi行列のトレース距離による測定と比べれば、そうなるのが当然です。

パートBは小さなハミルトニアンでの同予算の判定で、結果は明快です。2次Trotterがすべてのゲート数で勝ち、$G = 512$ では2桁の差をつけます。$L = 4$ ではqDRIFTの $L$ 非依存性が救うものが何もありません。

パートCが興味深い部分で、機構を切り出すために設計しました。1ノルムを $\lambda = 2.0$、ゲート予算を1024回転に固定し、変えるのは項数だけです。係数を対数一様に取ったのは、実際の分子積分がもつ裾の重い分布がそれだからです。qDRIFTの誤差は平坦で、$L = 4$ で $4.3 \times 10^{-4}$、$L = 255$ で $3.9 \times 10^{-4}$。一方2次Trotterの誤差は350倍に増えます。固定予算で買えるステップ数がどんどん減るからです。比をフィットすると $\propto L^{0.875}$ で、交差点は $L \approx 1.6 \times 10^{3}$ 付近です。このフィット指数は、上の同予算の代数計算が予測する $L^{2}$ ではありません。理由は、その代数計算が $C_2'$ を固定しているのに対し、ここのハミルトニアン列はそうではないことです。1ノルムを固定したまま対数一様な項を足していくと、交換子和は項数よりずっとゆっくり増えます。したがってこの交差点は $4 \le L \le 255$ のフィットの外挿であって測定された点ではなく、その不確かさは 0.875 と 2 の間の指数のどこにあるかという形で現れます。

この数値がこの節の正直な結論です。**シミュレートできる範囲のすべてで2次Trotterが勝ち、ハミルトニアンが数千項になるまで勝ち続けます。** 数百スピン軌道の分子・材料系ハミルトニアンは $10^5$ から $10^8$ 項をもち、それがまさにランダム化コンパイルが提案された領域です。そしてそこは同時に、$1/\varepsilon$ を $\log(1/\varepsilon)$ に変えるのがqubitizationだけであるために、qubitizationが両者を押しのける領域でもあります。qDRIFTの道具箱における本当の居場所は、巨大なハミルトニアンに対して*中程度*の精度を最も安く得る手段であり、そしてランダム化がより良いスケーリングを買う異例に明快な実例です。

* * *

## 4.4 リソース見積りの作法

### 通貨は非Cliffordゲート

誤り耐性量子コンピュータは、ゲートに一律の料金を課しません。Cliffordゲート — $H$、$S$、CNOT — は表面符号でトランスバーサルかそれに近く、Gottesman-Knillの定理によりそれだけで作られた回路は古典的にシミュレート可能なので、興味深い仕事をしているはずがありません。興味深い仕事をするのは非Cliffordゲートであり、表面符号でそれを供給するのは**マジックステート蒸留**です。工場が多数のノイジーな物理状態を消費して高忠実度の論理 $T$ またはToffoli状態を1つ産出し、その面積と時間のコストが計算全体を支配します。

だからこそ誤り耐性アルゴリズムの論文が報告する数値は2つです。

  * **Toffoli数**（あるいは等価にT数。標準的な構成でToffoli 1個 $\approx$ T 4個）。これが実行時間です。
  * **論理量子ビット数**。これが幅であり、系の量子ビット、算術とQROMのための補助量子ビット、そして配線用の空間を含みます。通常は系レジスタの数倍になります。

物理量子ビット数と実時間は、符号距離とサイクル時間を選ぶことでこの2つから導かれます。だから誰かが工場のレイアウトを再最適化するたびに動くのです。Toffoli数が、安定したアルゴリズムレベルの言明です。

### qubitized位相推定の式

4.2節のすべてが1行に凝縮されます。ウォーク作用素上の位相推定でエネルギー分解能 $\varepsilon$ を得るには

$$ \text{ウォークステップ数} \approx \frac{\pi}{2}\,\frac{\alpha}{\varepsilon}, \qquad \text{Toffoli数} \approx \frac{\pi}{2}\,\frac{\alpha}{\varepsilon}\, \times C_{\text{walk}} $$

が必要です。$C_{\text{walk}}$ は PREPARE-SELECT-PREPARE$^\dagger$ 1組と鏡映のToffoliコストです。入力は3つ、それだけです。Hartree単位の1ノルム $\alpha$、目標分解能 $\varepsilon$、そして1ステップあたりのコストです。化学精度は慣例的に $\varepsilon = 1.6 \times 10^{-3}$ Hartree（1 kcal/mol）を意味し、この慣例はかなり大きな働きをしています。$\varepsilon$ の10倍は実行時間の10倍です。

### Code Example 7: Toffoli数と論理量子ビット数によるリソース見積り

```python
"""第4章 Code Example 7: Toffoli数と論理量子ビット数によるリソース見積り。
Code Example 6 の続き（同一セッション）。"""
print("この分野が実際に使う単位でのリソース見積り")
print("=" * 68)
CHEM_ACC = 1.6e-3        # Hartree。慣例的な化学精度の目標値です


def qubitized_qpe_toffolis(lam, eps, toffolis_per_walk):
    """qubitized QPEのToffoli数: (pi/2)(lam/eps) ウォークステップ"""
    steps = np.pi * lam / (2 * eps)
    return steps, steps * toffolis_per_walk


print("  Toffoli数 = (pi/2) (lambda/eps) x (1ウォークステップあたりToffoli数)")
print(f"  以下すべて eps = {CHEM_ACC:.1e} Hartree（化学精度）です")
print("  lambda は選んだ因子分解におけるハミルトニアンの1ノルムで、"
      "単位はHartreeです\n")
print(f"{'system':>34} {'orbitals':>9} {'qubits':>7} {'lambda(Ha)':>11}"
      f" {'walk steps':>12}")
rows = [("H2, minimal basis", 2, 0.7),
        ("N2, moderate active space", 20, 40.0),
        ("FeMoco active space, low estimate", 76, 300.0),
        ("FeMoco active space, high estimate", 76, 4000.0)]
for name, norb, lam in rows:
    steps, _ = qubitized_qpe_toffolis(lam, CHEM_ACC, 1)
    print(f"{name:>34} {norb:9d} {2*norb:7d} {lam:11.1f} {steps:12.3e}")

print("\nFeMocoのToffoli数が、2つの不確かな入力にどれだけ敏感か")
print("-" * 68)
print(f"{'lambda (Ha)':>12}", end='')
for cw in (1e4, 3e4, 1e5):
    print(f" {'C_walk=' + f'{cw:.0e}':>16}", end='')
print()
for lam in (300.0, 1000.0, 4000.0):
    line = f"{lam:12.0f}"
    for cw in (1e4, 3e4, 1e5):
        steps, tof = qubitized_qpe_toffolis(lam, CHEM_ACC, cw)
        line += f" {tof:14.2e}" + (" *" if 1e10 <= tof <= 1e11 else "  ")
    print(line.rstrip())
print("  * は、この系について公表されている誤り耐性見積りが占める")
print("  1e10 から 1e11 Toffoliの帯に入るセルです。どちらの入力も数倍の精度でしか")
print("  分かっていないので、成果物は指数であって仮数ではありません。そしてその")
print("  指数は、因子分解とマジックステート蒸留の10年の仕事で数単位下がりました。")

print("\nToffoli数から実時間へ")
print("-" * 68)
T_TOF = 1e-5             # 論理Toffoli 1個あたりの秒数。標準的な仮の値です
print(f"{'Toffolis':>10} {'seconds':>12} {'days':>9}"
      f"   論理Toffoli 1個 {T_TOF*1e6:.0f} us のとき")
for tof in (1e9, 1e10, 1e11):
    print(f"{tof:10.0e} {tof*T_TOF:12.3e} {tof*T_TOF/86400:9.2f}")

print("\nToffoli 1個あたりに必要な論理誤り率と、表面符号の距離")
print("-" * 68)
print(f"{'Toffolis':>10} {'p_L needed':>12} {'d (p=1e-3)':>11}"
      f" {'phys/logical':>13} {'phys. qubits':>13}")
for tof in (1e9, 1e10, 1e11):
    pL = 0.1 / tof                     # 全体の失敗確率を0.1とします
    d = 1
    # p_L は比を大きなべきに上げた量なので、比較には相対許容誤差を持たせます。
    # 目標をちょうど満たす距離は2進浮動小数点では目標の数ulp上に着地し
    # （0.1 * 0.1**9 は 1.0000000000000006e-10 になります）、素朴な > では
    # その距離が棄却されて1つ上の距離が返ってしまいます。
    while 0.1 * (1e-3 / 1e-2) ** ((d + 1) / 2) > pL * (1 + 1e-9):
        d += 2
    per = 2 * d * d
    n_log = 2 * 76 + 1000              # 系 + 配線と補助量子ビット
    print(f"{tof:10.0e} {pL:12.1e} {d:11d} {per:13d} {per*n_log:13.2e}")
print("  これに加えてマジックステート工場が必要で、公表されたレイアウトでは")
print("  同等かそれ以上の面積を占めます。数百万の物理量子ビットと数日の実行時間、")
print("  というのが現時点の結論であり、これまでのアルゴリズムの進歩は"
      "それを取り除いていません。")

print("\n同じ見積りをTrotterで行うとどうなるか")
print("-" * 68)
print(f"{'method':>26} {'scaling in t and eps':>34} {'exponent of 1/eps':>18}")
for name, sc, ex in [("first-order Trotter", "O(L (alpha t)^2 / eps)", "1"),
                     ("2k-th order Trotter",
                      "O(L (alpha t)^{1+1/2k} / eps^{1/2k})", "1/2k"),
                     ("qDRIFT", "O(lambda^2 t^2 / eps)", "1"),
                     ("qubitization / QSP",
                      "O(alpha t + log(1/eps))", "log")]:
    print(f"{name:>26} {sc:>34} {ex:>18}")
print("  最後の行の log(1/eps) こそ、FTQC量子化学の見積りがTrotterステップ数")
print("  ではなくqubitizationの言葉で語られる理由のすべてです。")
```

```text
この分野が実際に使う単位でのリソース見積り
====================================================================
  Toffoli数 = (pi/2) (lambda/eps) x (1ウォークステップあたりToffoli数)
  以下すべて eps = 1.6e-03 Hartree（化学精度）です
  lambda は選んだ因子分解におけるハミルトニアンの1ノルムで、単位はHartreeです

                            system  orbitals  qubits  lambda(Ha)   walk steps
                 H2, minimal basis         2       4         0.7    6.872e+02
         N2, moderate active space        20      40        40.0    3.927e+04
 FeMoco active space, low estimate        76     152       300.0    2.945e+05
FeMoco active space, high estimate        76     152      4000.0    3.927e+06

FeMocoのToffoli数が、2つの不確かな入力にどれだけ敏感か
--------------------------------------------------------------------
 lambda (Ha)     C_walk=1e+04     C_walk=3e+04     C_walk=1e+05
         300       2.95e+09         8.84e+09         2.95e+10 *
        1000       9.82e+09         2.95e+10 *       9.82e+10 *
        4000       3.93e+10 *       1.18e+11         3.93e+11
  * は、この系について公表されている誤り耐性見積りが占める
  1e10 から 1e11 Toffoliの帯に入るセルです。どちらの入力も数倍の精度でしか
  分かっていないので、成果物は指数であって仮数ではありません。そしてその
  指数は、因子分解とマジックステート蒸留の10年の仕事で数単位下がりました。

Toffoli数から実時間へ
--------------------------------------------------------------------
  Toffolis      seconds      days   論理Toffoli 1個 10 us のとき
     1e+09    1.000e+04      0.12
     1e+10    1.000e+05      1.16
     1e+11    1.000e+06     11.57

Toffoli 1個あたりに必要な論理誤り率と、表面符号の距離
--------------------------------------------------------------------
  Toffolis   p_L needed  d (p=1e-3)  phys/logical  phys. qubits
     1e+09      1.0e-10          17           578      6.66e+05
     1e+10      1.0e-11          19           722      8.32e+05
     1e+11      1.0e-12          21           882      1.02e+06
  これに加えてマジックステート工場が必要で、公表されたレイアウトでは
  同等かそれ以上の面積を占めます。数百万の物理量子ビットと数日の実行時間、
  というのが現時点の結論であり、これまでのアルゴリズムの進歩はそれを取り除いていません。

同じ見積りをTrotterで行うとどうなるか
--------------------------------------------------------------------
                    method               scaling in t and eps  exponent of 1/eps
       first-order Trotter             O(L (alpha t)^2 / eps)                  1
       2k-th order Trotter O(L (alpha t)^{1+1/2k} / eps^{1/2k})               1/2k
                    qDRIFT              O(lambda^2 t^2 / eps)                  1
        qubitization / QSP            O(alpha t + log(1/eps))                log
  最後の行の log(1/eps) こそ、FTQC量子化学の見積りがTrotterステップ数
  ではなくqubitizationの言葉で語られる理由のすべてです。
```

**注目すべき点。** この例の実質は感度の格子表です。FeMoco規模の計算のToffoli数は、どちらも数倍の精度でしか分かっていない2つの数の積であり、その積はもっともらしい入力の範囲で $3 \times 10^{9}$ から $4 \times 10^{11}$ まで振れます。$10^{10}$ から $10^{11}$ の帯 — 公表されているこの系の誤り耐性見積りが位置し、姉妹コース第4章もそこに置いている帯 — に入るセルは、表の真ん中を斜めに走る帯です。**リソース見積りの成果物は指数です。** $\alpha$、$\varepsilon$、因子分解を併記せずにリソース見積りを有効数字2桁で引用する人は、自分で擁護できない数値を引用しています。

実時間と表面符号のブロックは、指数をハードウェアに翻訳します。論理Toffoli 1個あたり10マイクロ秒 — 符号サイクル時間 $\times$ 距離 $\times$ 工場の遅延で決まる標準的な仮の値 — とすると、$10^{10}$ Toffoliは1.2日、$10^{11}$ は11.6日です。$10^{11}$ Toffoliにわたって全体の失敗確率を0.1に抑えるには1ゲートあたり $10^{-12}$ の論理誤り率が必要で、物理誤り率 $10^{-3}$ では表面符号距離21、したがって論理量子ビット1個あたり約880物理量子ビット、したがって約1150論理量子ビットのレジスタに対して $10^{6}$ の程度の物理量子ビットが必要です。しかもこれはマジックステート工場の*前*の話で、公表されたレイアウトでは工場が同等かそれ以上の面積を占めます。数百万の物理量子ビットと数日の実行時間 — 姉妹コースが到達したのと同じ結論であり、これは緩められていません。

### なぜ電子構造が本命で、それは何を意味するか

この応用がFTQC文献を支配している理由は明示的に述べる価値があります。論拠が「化学は重要だ」ではなく、もっと鋭いものだからです。

  * **問題が固有値問題であること。** 位相推定は、オラクル仮定やデータ読み込み仮定に依存しない指数的分離をもつ唯一の基本要素です。基底状態エネルギーはまさにそれが計算するものです。
  * **古典側の競合が組合せ的で、厳密手法についてはそれが証明されていること。** $M$ 軌道の完全配置間相互作用は $\binom{M}{N_\alpha}\binom{M}{N_\beta}$ 個の行列式を要します。近似手法 — DMRG、量子モンテカルロ、結合クラスター — は強力ですが、それぞれ既知の破綻モードをもち、強い静的相関はその大半を同時に引き起こします。
  * **答えが小さいこと。** 量子コンピュータが1回の計算で出すのは波動関数ではなく1つの数値です。提案される他の多くの量子応用を沈める出力帯域の問題が、ここでは障害になりません。
  * **標的が材料の標的であること。** 遷移金属酸化物、多核触媒クラスター、Mott絶縁体、そしてそれ以外はDFTで扱える固体に埋め込まれた強相関活性空間。これは姉妹コースが特定したのと同じリストであり、材料研究のリストです。

そしてこれが*立証していない*ことも、同じくらい明示的に述べる価値があります。第一に、これらの基底状態問題が古典的に難しいという証明はありません。根拠は既知の古典手法の破綻であり、計算複雑性の分離よりずっと弱い言明です。第二に、より具体的に、**位相推定は目標の固有ベクトルと無視できない重なりをもつ初期状態を必要とします。** 成功確率は $\lvert \langle \phi_0 \lvert \psi_{\text{init}} \rangle \rvert^{2}$ であり、強相関系では容易に準備できる参照状態 — Hartree-Fock行列式、小さなCI展開 — の重なりが系のサイズに対して指数的に減衰しえます。アルゴリズムの中に状態を用意してくれる部分はなく、しかも古典手法を打ち負かすのと同じ静的相関が重なりを縮めるのです。これはFTQC量子化学の計画全体で最も重みを担う前提であり、5.5節で高速化の地図を描くときに戻ってきます。

### リソース見積りを読むためのチェックリスト

  1. **$\alpha$ は何で、単位は何か。** 1ノルムを報告していない論文は、コストを報告していません。
  2. **2電子テンソルのどの因子分解か。** スパース、単一因子分解、二重因子分解、テンソル超縮約で $\alpha$ も $C_{\text{walk}}$ も変わり、異なる選択をした2本の論文を比べることは無意味です。
  3. **$\varepsilon$ は何で、どの量に対してか。** 全エネルギーを化学精度で得るのは、誤差が部分的に打ち消し合う反応座標に沿った*エネルギー差*よりはるかに難しい目標です。
  4. **初期状態の重なりは議論されているか、仮定されているか。** 重なりを黙って $O(1)$ としているなら、その見積りはどれだけ緩いか分からない下限です。
  5. **ToffoliかTゲートか。** 4倍の違いがしばしば暗黙にされます。
  6. **論理量子ビットか物理量子ビットか、符号距離と物理誤り率はいくつか。** 両方なしに物理量子ビット数は無意味です。
  7. **マジックステート工場は量子ビット予算に含まれているか。** 面積には含まれ、報告は別、ということがしばしばあります。

* * *

## 演習

#### 演習1: ブロック符号化の算術

3量子ビットのハミルトニアンが $H = 1.2\,ZZI - 0.8\,IXX + 0.5\,XIY + 0.3\,YYZ - 0.2\,IIZ$ で与えられています。

  1. LCUブロック符号化の $\alpha$ はいくらで、PREPAREに必要な補助量子ビットは何個ですか。
  2. PREPAREレジスタは項数より多くの基底状態をもちます。余った振幅はどうなっていなければならず、そうでない場合に何が起きますか。
  3. $\lVert H \lvert \psi \rangle \rVert = 1.5$ の状態について、LCU 1ラウンドの成功確率を求めてください。
  4. その成功をほぼ確実にする振幅増幅のラウンド数はいくつですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\alpha = 1.2 + 0.8 + 0.5 + 0.3 + 0.2 = 3.0\) です（符号はユニタリ側に吸収されるので、1ノルムは絶対値を使います）。\(L = 5\) 項なので、PREPAREには \(\lceil \log_2 5 \rceil = 3\) 個の補助量子ビットが必要で、5項に対して8個の基底状態が得られます。</p>

<p><strong>2.</strong> 使わない3つの振幅はちょうど0でなければなりません。そうでない場合、ブロック符号化は \(\sum_l c_l U_l / \alpha\) を実装しますが、余分な \(l\) が未定義の制御値に対してSELECTが行うこと — 実際上は恒等演算 — を寄与し、符号化された演算子に余計な項が静かに加わります。SELECTを未使用添字で恒等演算にパディングすることと、振幅を強制的に0にすることは同じ要求の2つの言い方であり、数値で確認すべきは後者です。</p>

<p><strong>3.</strong> \(P = \lVert H \lvert \psi \rangle \rVert^2 / \alpha^2 = 1.5^2/3.0^2 = 0.25\) です。</p>

<p><strong>4.</strong> \(\theta = \arcsin\sqrt{0.25} = \arcsin(0.5) = \pi/6\) なので最適ラウンド数は \(k = \lfloor (\pi/2 - \theta)/(2\theta) \rfloor = \lfloor (\pi/3)/(\pi/3) \rfloor = 1\) で、1ラウンドでちょうど \(\sin^2(3\theta) = \sin^2(\pi/2) = 1\) になります。成功確率がちょうど1/4という幸運な場合には、1ラウンドが完璧です。</p>

```python
import numpy as np
c = np.array([1.2, 0.8, 0.5, 0.3, 0.2])
print(c.sum(), int(np.ceil(np.log2(len(c)))))        # 3.0 3
p = 1.5**2 / 3.0**2
th = np.arcsin(np.sqrt(p))
k = int((np.pi/2 - th) / (2*th))
print(round(p, 4), k, round(np.sin((2*k+1)*th)**2, 6))   # 0.25 1 1.0
```

</details>

#### 演習2: ウォークのスペクトルを逆向きに読む

$\alpha = 12.0$ Hartree のブロック符号化を渡され、そのウォーク作用素上で $m = 12$ 補助量子ビットの位相推定を行うと、読み出し $k = 891$ にピークが出ました。

  1. これはどのエネルギーに対応しますか。
  2. スペクトルのこの点で $m = 12$ が意味するエネルギー分解能はいくらで、なぜスペクトル上のどこにいるかに依存するのですか。
  3. 固定した $m$ に対して、スペクトルのどの部分が最もよく分解され、どの部分が最も悪いですか。
  4. 同じ点で化学精度 $\varepsilon = 1.6 \times 10^{-3}$ Hartree を得るには、補助量子ビットは何個必要ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\theta = 2\pi k/2^{m} = 2\pi \times 891/4096 = 1.36678\) rad なので、\(E = \alpha\cos\theta = 12.0 \times 0.20261 = 2.4313\) Hartree です。</p>

<p><strong>2.</strong> 最下位1ビットは \(\delta\theta = 2\pi/4096 = 1.534\times10^{-3}\) rad で、\(\lvert dE/d\theta \rvert = \alpha \lvert \sin\theta \rvert = 12.0 \times 0.97926 = 11.75\) なので \(\delta E = 1.8026\times10^{-2}\) Hartree です。分解能が位置に依存するのはヤコビアン \(\alpha \sin\theta\) のせいで、写像 \(E = \alpha\cos\theta\) が一様でないからです。</p>

<p><strong>3.</strong> \(\delta E = \alpha\lvert\sin\theta\rvert\,\delta\theta\) なので、分解能が最も良い（\(\delta E\) が最小の）のは \(\lvert\sin\theta\rvert\) が最小のところ、つまりバンド端 \(E \to \pm\alpha\) であり、最も悪いのは \(\lvert\sin\theta\rvert = 1\) となるバンド中央 \(E \approx 0\) です。基底状態は端の近くにあるので、これは珍しい幸運です。qubitized QPEが最も鋭く分解するのはまさに極端な固有値であり、誰も欲しがらない密なスペクトル中央部が最も悪く分解される部分なのです。</p>

<p><strong>4.</strong> \(\alpha \lvert\sin\theta\rvert \, 2\pi/2^{m} \le 1.6\times10^{-3}\) が必要なので \(2^{m} \ge 2\pi \times 11.75/1.6\times10^{-3} = 4.615\times10^{4}\)、すなわち \(m = 16\)、\(2^{m} = 65\,536\) です。これは読み出しレジスタの幅であり、最も深い制御べきは \(W^{32768}\)、ウォークステップ数の合計は \(2^{m} - 1 \approx 6.6\times10^{4}\) で、目安の \(\pi\alpha/2\varepsilon = 1.18\times10^{4}\) と同じ桁です。</p>

```python
import numpy as np
alpha, m, k = 12.0, 12, 891
th = 2*np.pi*k/2**m
print(round(th, 5), round(alpha*np.cos(th), 4))            # 1.36678 2.4313
dE = alpha*abs(np.sin(th))*2*np.pi/2**m
print(f"{dE:.4e}")                                          # 1.8026e-02
need = 2*np.pi*alpha*abs(np.sin(th))/1.6e-3
print(int(np.ceil(np.log2(need))))                          # 16
```

</details>

#### 演習3: qDRIFTとTrotterのどちらを選ぶか

あるハミルトニアンは $L = 40\,000$ 個のPauli項をもち、1ノルムは $\lambda = 400$ Hartree です。$t = 1$ Hartree$^{-1}$ の $e^{-iHt}$ をチャネル誤差 $10^{-3}$ で得たいとします。

  1. $\text{誤差} \approx 2\lambda^2 t^2/N$ からqDRIFTの回転数を見積もってください。
  2. 2次Trotterは $2Lr$ 回転で $r$ ステップを要し、誤差は $\approx C_2/r^2$ です。交換子の和の粗い代用として $C_2 \approx L^{1.5}$ を取り、回転数を見積もってください。
  3. どちらが勝ち、その差はどれだけですか。
  4. 答えを変えるものは何ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(N = 2\lambda^2 t^2/\varepsilon = 2 \times 1.6\times10^{5}/10^{-3} = 3.2\times10^{8}\) 回転です。</p>

<p><strong>2.</strong> \(C_2 \approx 40000^{1.5} = 8.0\times10^{6}\) なので \(r = \sqrt{C_2/\varepsilon} = \sqrt{8.0\times10^{9}} = 8.944\times10^{4}\) ステップ、回転数は \(2Lr = 2 \times 4\times10^{4} \times 8.944\times10^{4} = 7.16\times10^{9}\) です。</p>

<p><strong>3.</strong> qDRIFTが約22倍で勝ちます。機構はすべてのTrotterステップに掛かる \(2L\) です。40 000項では2次Trotterの1ステップだけで80 000回転かかり、これはqDRIFTの全予算の4分の1に当たります。</p>

<p><strong>4.</strong> 3つあります。より厳しい精度目標: qDRIFTは \(1/\varepsilon\)、2次Trotterは \(1/\sqrt{\varepsilon}\) でスケールするので、\(\varepsilon = 10^{-6}\) では回転数が \(3.2\times10^{11}\) と \(2.3\times10^{11}\) になりTrotterが先に立ちます。\(L\) を固定した \(\lambda\) の縮小 — より良い因子分解が達成するもの — はqDRIFTを二次的に助けます。そして項のグループ化: 可換な項は同時に指数化できるので、\(\lambda\) に触れずにTrotter側の実効 \(L\) を減らせます。いずれも、qubitizationが \(1/\varepsilon\) を \(\log(1/\varepsilon)\) に変えるので本格的な精度ではqubitizationが勝つ、という事実を変えません。</p>

```python
import numpy as np
L, lam, t = 40000, 400.0, 1.0
for eps in (1e-3, 1e-6):
    N = 2*lam**2*t**2/eps
    r = np.sqrt(L**1.5/eps)
    print(f"eps={eps:.0e}  qDRIFT {N:.2e}   Trotter-2 {2*L*r:.2e}")
# eps=1e-03  qDRIFT 3.20e+08   Trotter-2 7.16e+09
# eps=1e-06  qDRIFT 3.20e+11   Trotter-2 2.26e+11
```

</details>

#### 演習4: 擁護しなければならないリソース見積り

同僚が遷移金属酸化物クラスターの誤り耐性計算を提案しています。空間軌道60個、使う予定の因子分解で $\alpha = 800$ Hartree、$C_{\text{walk}} = 2 \times 10^{4}$ Toffoli、目標は化学精度です。

  1. Toffoli数と、論理Toffoli 1個あたり10 $\mu$s での実時間を求めてください。
  2. 続けて同僚は、因子分解を切り替えれば $\alpha$ を半分にできるが $C_{\text{walk}}$ が3倍になると言います。切り替えるべきですか。
  3. 同僚は全エネルギーではなく2つのスピン状態間のエネルギー*差*が欲しいので $\varepsilon$ を $5 \times 10^{-3}$ Hartree に緩められると主張します。それで何が得られますか。
  4. この見積りの中で、他のすべてを合わせたよりも大きく答えを狂わせうる入力を1つ挙げてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> ウォークステップ数 \(= \pi\alpha/2\varepsilon = 1.571 \times 800/1.6\times10^{-3} = 7.85\times10^{5}\)、Toffoli数 \(= 7.85\times10^{5} \times 2\times10^{4} = 1.57\times10^{10}\)、時間 \(= 1.57\times10^{5}\) 秒 \(= 1.82\) 日。公表されている帯の中にある、擁護可能な数値です。</p>

<p><strong>2.</strong> いいえ。積は \(0.5 \times 3 = 1.5\) 倍になるので、コストは50%増えます。経験則として \(\alpha\) と \(C_{\text{walk}}\) は積として対称に入るので、あらゆる取引は積だけで判断しなければなりません。そして \(\alpha\) を下げる因子分解はたいてい1ステップあたりのコストが上がるので、小さいノルムのほうが有利だと仮定せずに明示的に比較する必要があるのです。</p>

<p><strong>3.</strong> \(5\times10^{-3}/1.6\times10^{-3} = 3.125\) 倍で、見積りは \(5.0\times10^{9}\) Toffoli、0.58日になります。この業界で得られる最も安い節約であり、関心のある量が差であって2つの計算が系統誤差を共有する場合には物理的に正当です。だからこそリソース見積りは、どの量を標的にしているかを常に述べるべきなのです。</p>

<p><strong>4.</strong> 初期状態の重なりです。上のすべての数値は位相推定が成功すること、すなわち準備可能な参照状態が目標の固有ベクトルと \(O(1)\) の重なりをもつことを仮定しています。重なりが \(10^{-2}\) なら計算全体を \(\sim 10^{4}\) 回繰り返すか振幅増幅のラッパーを付ける必要があり、見積りは4桁動きます。これは \(\alpha\)、\(\varepsilon\)、\(C_{\text{walk}}\) を合わせても動かせない幅です。しかも最も定量化されないことが多い入力です。</p>

```python
import numpy as np
def toffolis(alpha, eps, cw): return np.pi*alpha/(2*eps)*cw
base = toffolis(800, 1.6e-3, 2e4)
print(f"{base:.3e}  {base*1e-5/86400:.2f} 日")       # 1.571e+10  1.82 日
print(f"{toffolis(400, 1.6e-3, 6e4)/base:.2f}x")     # 1.50x
print(f"{toffolis(800, 5e-3, 2e4):.3e}")             # 5.027e+09
```

</details>

#### 演習5: 他人のブロック符号化を検証する

$2^{m+n} \times 2^{m+n}$ 行列 `UA` とスカラー `alpha` を返し、あるハミルトニアンをブロック符号化していると主張するコードを渡されました。

  1. 主張を立証する3つの検査を、実行する順に書いてください。
  2. ある検査は通り、別の検査がPREPAREとSELECTで項の順序が食い違っていることを示す形で落ちました。どの検査がそれを捕まえ、落ち方はどう見えますか。
  3. `alpha` が1ノルムではなくスペクトルノルムだったら何が見えますか。
  4. 左上ブロックを検査するだけでは*不十分*なのはなぜですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> (i) ユニタリ性: \(\max \lvert U_A^\dagger U_A - I \rvert\) が機械精度であること。これが落ちれば他のすべてが無意味だからです。(ii) 定義的性質: 独立に構成した \(H\) に対して \(\max \lvert U_A[:2^n,:2^n] - H/\alpha \rvert\)。(iii) シミュレータ上の操作的検査: ランダムな \(\lvert \psi \rangle\) に対して \(U_A\) を \(\lvert 0^m \rangle \lvert \psi \rangle\) に作用させ、出力の先頭ブロックが \(H\lvert\psi\rangle/\alpha\) に等しく、その二乗ノルムが \(\lVert H \lvert\psi\rangle\rVert^2/\alpha^2\) に等しいことを確認します。(iii) は、ハミルトニアンが偶然対称な場合に (ii) が見逃す添字とエンディアンの誤りを捕まえます。</p>

<p><strong>2.</strong> 検査 (ii) が捕まえます。どのユニタリをどの振幅に割り当てるかを入れ替えても \(U_A\) はユニタリのままなのでユニタリ性は通りますが、左上ブロックは食い違った置換 \(\sigma\) に対する \(\sum_l \sqrt{c_l c_{\sigma(l)}}\,U_{\sigma(l)}/\alpha\) になります。つまりノルム構造は正しく成分が誤った*別の*エルミート行列です。特徴は、機械精度ではなく係数のばらつき程度の食い違いが出ることで、しかもブロックはたいていエルミートのままです。だから「ハミルトニアンらしく見える」ことは証拠になりません。</p>

<p><strong>3.</strong> 左上ブロックのスペクトルノルムが \(\lVert H \rVert/\alpha < 1\) ではなく1になり、ユニタリ性が落ちます。ユニタリがノルム1のブロックをもてるのは、その行と列の残りが消えるときだけであり、一般の \(H\) ではそうなりません。したがって検査 (i) が即座に捕まえます。同じことですが、LCU構成は \(\alpha < \sum_l \lvert c_l \rvert\) を作れません。</p>

<p><strong>4.</strong> *ユニタリでない*行列の左上ブロックは何でもありうるからです。検査 (i) なしでは「左上ブロックが \(H/\alpha\) である」は、どの量子回路も実装しない行列についての言明です。主張は2つの検査の対であり、どちらか一方だけでは主張になりません。</p>

</details>

* * *

## まとめ

### 要点

**1\. 積公式は $1/\varepsilon$ への多項式依存をもち、どんな次数でも消えない**

  * 本章の4項ハミルトニアンで実測した定数: 1次 $C_1 = 0.4039$、2次 $C_2 = 0.0993$。誤差 $\times r$ と誤差 $\times r^2$ の収束で両方を確認しました。
  * $10^{-9}$ に到達するには1次で $1.6\times10^{9}$ 回転、2次で $8.0\times10^{4}$ 回転。2万倍の差ですが、それでも多項式です。
  * コストは項数 $L$ に線形で、電子構造では $L = O(M^4)$ です。

**2\. ブロック符号化はユニタリの左上隅に $H$ を厳密に置く**

  * $(\langle 0^m \rvert \otimes I) U_A (\lvert 0^m \rangle \otimes I) = H/\alpha$ を、明示的な $16 \times 16$ 行列で $8 \times 10^{-17}$ まで検証しました。
  * LCU構成はPREPARE、SELECT、PREPARE$^\dagger$。実振幅4個のPREPAREは $R_y$ 3個とCNOT 2個で、SELECTはユナリ反復で $L-1$ Toffoliです。
  * 非対角ブロックは小さく*ありません* — ユニタリ性の残りの予算を担っています — そしてこの手法の要点は、悪いブロックを縮めることではなく良いブロックを扱えるようにすることです。

**3\. $\alpha$ は1ノルムであり、あらゆるコスト式に入るのはそれ**

  * 1ラウンドの成功確率は $\lVert H\lvert\psi\rangle\rVert^2/\alpha^2$。シミュレータに対して8桁で一致し、Haar平均は $\mathrm{Tr}(H^2)/(d\alpha^2)$ です。
  * 振幅増幅は $O(\alpha/\lVert H\lvert\psi\rangle\rVert)$ ラウンドを要します。
  * Heisenberg鎖では $\alpha$ は項数にちょうど比例して増え、$\lVert H \rVert$ はそれより遅く増えます。比は2サイトから10サイトで1.0から1.6へ流れ、分子ではもっと悪くなります。$\alpha$ の削減がFTQC量子化学のコストを動かす主要なてこです。

**4\. qubitizationはウォークの固有位相をエネルギーの $\arccos$ にする**

  * $W = (2\Pi - I)U_A$ の固有値は $e^{\pm i\theta_k}$ で $\cos\theta_k = E_k/\alpha$。$4\times10^{-10}$ で検証し、ジャンク部分空間は $\theta = 0, \pi$ にありました。
  * $W$ 上の位相推定は基底状態エネルギーを復元し、$\pm\theta_k$ の特徴的な2ピーク構造が出ます。$\cos$ が偶関数なのでこれは無害です。
  * 固有値は $O(\alpha/\varepsilon)$ クエリ、発展は $O(\alpha t + \log(1/\varepsilon))$ クエリで、どちらもクエリモデルで最適です。そしてこの対数がTrotterとの質的な断絶です。

**5\. qDRIFTは $1/N$ 誤差と引き換えに項数への完全な非依存を得る**

  * スーパー作用素として厳密に評価: 4項ハミルトニアンで誤差 $\times N \to 2.21$、すなわち $O(\lambda^2t^2/N)$ です。
  * 同一ゲート数では、シミュレートできるすべての比較で2次Trotterが勝ち、$L = 4$ では2桁の差です。
  * $\lambda$ と予算を固定して $L$ だけを変えると、qDRIFTの誤差は平坦でTrotterの誤差は $L^{0.875}$ で増えます。フィットした交差点は $L \approx 1.6\times10^{3}$ で、これがランダム化コンパイルが $10^5$ 項以上のハミルトニアンに対して提案された理由です。

**6\. リソース見積りとは、Toffoli数と前提条件の一覧である**

  * qubitized QPEは $\approx (\pi/2)(\alpha/\varepsilon)$ ウォークステップを要し、各ステップが $C_{\text{walk}}$ Toffoli。入力は3つで、答えはその積です。
  * FeMoco規模の活性空間についてもっともらしい $(\alpha, C_{\text{walk}})$ の範囲で、Toffoli数は $3\times10^{9}$ から $4\times10^{11}$ まで振れ、公表された $10^{10}$–$10^{11}$ の帯は表の真ん中を斜めに走ります。
  * $10^{11}$ Toffoliは10 $\mu$s 換算で11.6日、論理誤り率 $10^{-12}$、表面符号距離21、マジックステート工場の前で $10^{6}$ の程度の物理量子ビットが必要です。数百万の物理量子ビットと数日の実行時間です。

**実務上の含意**

  * ブロック符号化のコストを報告するときは必ず $\alpha$ を報告すること。1ノルムのないコストはコストではありません。
  * ブロック符号化は独立な2つの検査 — ユニタリ性と定義的ブロック — と、シミュレータ上の操作的検査1つで検証すること。どれか1つだけでは誤った構成でも通ります。
  * シミュレーション手法は流行ではなく領域で選ぶこと。項数が少なく精度が厳しいならTrotter、非常に多数の小さな項で中程度の精度ならqDRIFT、標的が固有値なら常にqubitizationです。
  * 「量子コンピュータならFeMocoが計算できる」を読んだら、$\alpha$、$\varepsilon$、因子分解、そして初期状態の重なりを問うこと。前の3つが指数を決め、4つ目はそれを無効にしえます。

### 次章へ

これでFTQC側の道具は揃いました。厳密なブロック符号化、スペクトルがハミルトニアンのものであるウォーク、そしてそれらのコストを述べる語彙です。そのすべては誤り耐性機械を前提にしています。第5章は反対側 — 変分的で近未来の側 — に向かい、量子近似最適化アルゴリズムを扱い、同じ規律をそれに適用します。MaxCutをIsing問題として定式化し、同じシミュレータ上にQAOAを組み立て、近似比が深さとともに、また固定した断熱スケジュールとともに登っていくのを見て、それから貪欲法・局所探索・焼きなまし法・Goemans-Williamson丸めと同予算で対決させ、すべての差に対応のある区間を付けます。この章はシリーズを、証明可能な高速化が実際にどこに住んでいて、それぞれが何を前提にしているかの地図で締めくくります。いま終えた2つの章も含めてです。

[← 第3章: Shorのアルゴリズム](<chapter-3.html>) [第5章: QAOAと最適化 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の1ノルム、1ステップあたりToffoliコスト、符号距離、実時間の数値は、リソース見積りの算術を示すために選んだ桁の目安であり、公表された見積りの代わりにはなりません。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
