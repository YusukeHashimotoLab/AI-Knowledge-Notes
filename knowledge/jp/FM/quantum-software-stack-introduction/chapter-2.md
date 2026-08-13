---
title: "第2章: 回路最適化とゲート合成"
chapter_title: "第2章: 回路最適化とゲート合成"
subtitle: 意味を保つ書き換え規則、EulerとKAK分解、そして1つのゲートだけが他のすべてより高価である理由
reading_time: 45-50分
difficulty: 上級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-software-stack-introduction/chapter-2.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子ソフトウェアスタック入門](<index.html>) > 第2章

第1章は本コース全体がパスをかける対象の層を作り、結果ではなく計測で終わりました。Code Example 5 のピープホール規則はルーティング後の回路から何も除去できず、その理由は、相殺すべきゲートが*量子ビットの上では*隣接していても*リストの上では*隣接していなかったことでした。本章はそれを直し、さらに先へ進みます。扱うのは、回路を手にしたコンパイラが行う2つのこと — 同じ意味のより短い回路へ**書き換える**ことと、任意のユニタリを機械が実際に持つゲートへ**合成する**ことです。

この2つには別種の数学が必要です。書き換えは組み合わせ的です。局所的な恒等式の集合、2つのゲートを交換してよいかの判定、そして不動点までのループからなり、その正しさは議論するのではなく検査すべき性質です。だからこそ第1章のユニタリ等価チェッカが以下のすべての例に登場します。合成のほうは線形代数です。1量子ビットにはEuler分解、2量子ビットにはKAK分解、そして誤り耐性のゲート集合には、任意回転1個のコストに固い下限を与える数え上げの議論があります。この最後のものが本章で最も重大な数値であり、しかも引用ではなく計測で得ます。任意の1量子ビット回転を小数6桁の精度で実現するコストは、Toffoli約10個分です。

## 学習目標

本章を修了すると、以下のことができるようになります。

  * 局所的な書き換え規則の3系統 — 融合・相殺・交換 — を回路断片の間の恒等式として述べ、それぞれを信用するのではなく位相を除いた行列比較で検証できる
  * *完全*ではなく*健全*な交換判定関数を書き、書き換えパスに必要なのが健全性であることを説明し、構文的な規則集合が完全性をどれだけ諦めているかを計測できる
  * 不動点に到達するピープホール最適化器を実装し、成功する規則が必ず回路を短くすることから停止性を示し、その収量を種を固定したランダム回路とコンパイラ出力の回路で計測できる
  * 任意の $U \in U(2)$ に対するZYZ分解 $U = e^{i\delta} R_z(a) R_y(b) R_z(c)$ を導き、2つの退化した分岐を扱い、合成ルーチンが3つの角に加えて位相 $\delta$ も返さなければならない理由を説明できる
  * 制御$U$ をCXゲート2個で、SWAPを3個で構成し、KAK分解を述べ、2量子ビットゲートの局所不変量からそれが必要とする最小CX数を読み取れる
  * 誤り耐性のゲート集合が第1章のものではなくClifford$+T$ である理由と、そこでは総ゲート数ではなく $T$ 数が問題になる理由を説明できる
  * $\varepsilon$ 精度の回転の $T$ 数に対する下限 $t \gtrsim 3\log_2(1/\varepsilon)$ を数え上げから導き、全数探索で確認できる

* * *

## 2.1 書き換え規則

### 規則とは何か

**書き換え規則**とは、同じユニタリをもつ回路断片の対です。定義はこれだけで、唯一の微妙な点は「同じ」という語です。本コースではこれは*グローバル位相を除いて等しい*ことを意味し、第1章が `phase_free_error` を作って検査できるようにした関係そのものです。規則の適用とは、回路の中に左辺を見つけて右辺に置き換えることであり、その置換が正当であるのは両辺がこの検査を通るとき、ちょうどそのときです。

局所的な規則が回路を短くしうる方法はちょうど3通りしかなく、これまで書かれたあらゆるピープホール最適化器はその何らかの組み合わせです。

| 系統 | 規則の形 | 例 | 必要な知識 |
| --- | --- | --- | --- |
| **融合** | 同じ量子ビット上の2ゲートが1つになる | $R_z(a)\,R_z(b) = R_z(a+b)$、$T\,T = S$ | 2つのゲートが同じ1パラメータ部分群に属すること |
| **相殺** | 2ゲートが0個になる | $H\,H = I$、$\mathrm{CX}\,\mathrm{CX} = I$ | そのゲートが対合であること |
| **交換** | 2ゲートを入れ替え、第3の規則を発動させる | $Z_0\,\mathrm{CX}_{0,1} = \mathrm{CX}_{0,1}\,Z_0$ | どの対のゲートが交換するか |

実質的な仕事をするのは交換であり、また3系統のうち交換だけは単独では何も短くしません。その役割は他の2つを適用可能にすることです。制御側の回転で隔てられた2つのCXゲートは、リストの隣接項を見る規則では相殺できませんが、その回転をどちらかのCXの向こう側へ滑らせれば隣接し、そこで消滅します。この1つの仕組みが、ルーティング後に何も見つけられなかった第1章の最適化器と、Code Example 3 の実演回路から2量子ビットゲートをすべて除去する本章の最適化器との差です。

規則を適用する前にもう1つ述べておくことがあります。本コースの回路はPythonのリストであり、リストでは添字 $i$ の次のゲートは添字 $i+1$ のゲートです。一方、回路の上でゲート $g$ の「次」のゲートとは、*$g$ の量子ビットのいずれかに触れる*次のゲートです。その間にあるものは別の量子ビットに作用しており、$g$ と自明に交換します。第1章の `next_touching` はまさにそれを実装しており、以下のすべての規則はこれを用いて述べられます。実運用のコンパイラは、回路をリストではなく有向非巡回グラフとして保持することで同じ点を構造的に処理します。

### 健全性と完全性

交換判定関数は2つの方向に誤りえますが、両者は対称ではありません。

  * 交換しない2ゲートを交換すると答えると、その上に作られたすべてのパスが黙って回路の意味を変えます。これが**不健全性**であり、致命的です。
  * 交換する2ゲートを交換しないと答えると、最適化の機会を失うだけです。これが**不完全性**であり、単に逃した最適化にすぎません。

そこで判定関数は保守的に書き、その不完全性は修正すべきバグではなく計測すべき数値になります。

### Code Example 1: 契約の再掲

本章のすべては第1章の3つのモジュール上で動きます。本章が自己完結するようにここに再掲します。1つ目は[量子コンピューティング入門](<../quantum-computing-introduction/chapter-2.html>)の状態ベクトルシミュレータで、本章が使う関数のみ逐語で載せます。同ファイルの `probs` と `sample` は本章では不要なので省いています。

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
```

2つ目は回路IRで、第1章 Code Example 2 から逐語です。ゲートタプルの形式、ビッグエンディアンの量子ビット順序、そして `run_circuit`、`circuit_depth`、`gate_counts` の3つのシグネチャが、本コースの全章で共有する契約です。

```python
"""Chapter 1, Example 2: 本コースの回路IRです。

回路はゲートタプルのリストです。ゲート名は文字列、量子ビットは整数
（ビッグエンディアン、量子ビット0が最左）です。このファイルを qir.py として
保存してください。以下のすべてのコード例は `from qir import *` から始まり、
以降のすべての章がこれを再掲します。

    ("h", q)   ("x", q)   ("z", q)   ("s", q)   ("t", q)
    ("rx", theta, q)      ("ry", theta, q)      ("rz", theta, q)
    ("cx", control, target)                     ("cz", q1, q2)
"""
import numpy as np
from qcsim import *

CZ4 = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

FIXED_1Q = {"h": H, "x": X, "z": Z, "s": S, "t": T}
ROT_1Q = {"rx": rx, "ry": ry, "rz": rz}
TWO_Q = ("cx", "cz")


def gate_qubits(g):
    """ゲートタプル1個が触れる量子ビットを、書かれた順に返します"""
    if g[0] in ROT_1Q:
        return (g[2],)
    if g[0] in TWO_Q:
        return (g[1], g[2])
    if g[0] in FIXED_1Q:
        return (g[1],)
    raise ValueError(f"unknown gate name {g[0]!r}")


def apply_ir_gate(state, g, n):
    """ゲートタプル1個を n量子ビット状態ベクトルに作用させます"""
    name = g[0]
    if name in FIXED_1Q:
        return apply_gate(state, FIXED_1Q[name], [g[1]], n)
    if name in ROT_1Q:
        return apply_gate(state, ROT_1Q[name](g[1]), [g[2]], n)
    if name == "cx":
        return cnot(state, g[1], g[2], n)
    if name == "cz":
        return apply_gate(state, CZ4, [g[1], g[2]], n)
    raise ValueError(f"unknown gate name {name!r}")


def run_circuit(circ, n, psi0=None):
    """ゲートタプルのリストを状態ベクトルシミュレータで実行し、最終状態を返します。

    psi0 の既定値は |00...0> です。ゲートは左から右へ作用させるので、回路の
    行列はゲート行列の逆順の積になります。
    """
    state = ket("0" * n) if psi0 is None else np.asarray(psi0, dtype=complex)
    for g in circ:
        state = apply_ir_gate(state, g, n)
    return state


def circuit_depth(circ, n):
    """量子ビットの排他性による貪欲な層分け: 回路が必要とする層数です。

    すべてのゲートが1単位時間を要すると仮定しています。これはハードウェア上では
    誤りであり、第4章で修正します。
    """
    ready = [0] * n              # 各量子ビットが最初に空く層
    for g in circ:
        qs = gate_qubits(g)
        layer = max(ready[q] for q in qs)
        for q in qs:
            ready[q] = layer + 1
    return max(ready) if n else 0


def gate_counts(circ):
    """ゲート名 -> 個数。加えてキー "2q" に2量子ビットゲートの総数を入れます"""
    counts = {}
    for g in circ:
        counts[g[0]] = counts.get(g[0], 0) + 1
    counts["2q"] = sum(counts.get(name, 0) for name in TWO_Q)
    return counts
```

3つ目は検査器で、第1章 Code Example 4 から逐語です。`qcheck.py` として保存してください。

```python
"""The unitary-equivalence checker of Chapter 1, re-listed.

このファイルを qcheck.py として保存してください。以降の各例は
`from qcheck import *` で始まります。
"""
import numpy as np
from qir import *


def unitary_of(circ, n):
    """回路の 2^n x 2^n 行列: 各基底状態に対して1回ずつ実行します"""
    dim = 2 ** n
    U = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        U[:, j] = run_circuit(circ, n, psi0=e)
    return U


def best_global_phase(U, V):
    """e^{i phi} V を U に位相だけで最も近づける位相を返します。

    これはHilbert-Schmidt重なり tr(V^dagger U) から得られます。Cauchy-Schwarzに
    より、その絶対値が 2^n に達するのは U = e^{i phi} V のときに限るので、
    重なりがほぼゼロであること自体が両者が非等価であることの証明になります。
    """
    tr = np.trace(V.conj().T @ U)
    return 1.0 + 0.0j if abs(tr) < 1e-12 else tr / abs(tr)


def phase_free_error(U, V):
    """最良の大域位相を除去したあとの max |U - e^{i phi} V|"""
    return float(np.max(np.abs(U - best_global_phase(U, V) * V)))


def assert_equivalent(a, b, n, label="", atol=1e-10):
    """本コースのすべての書き換えパスを守るテストです"""
    err = phase_free_error(unitary_of(a, n), unitary_of(b, n))
    if err > atol:
        raise AssertionError(f"{label}: circuits differ, max error {err:.3e}")
    return err
```

3つを軽く動かし、本章のすべてのパスを守る accept/REJECT の判定を示します。

```python
"""第2章 Example 1: 書き換えを始める前に、契約を動かして確かめます。"""
import numpy as np
from qcheck import *

bell = [("h", 0), ("cx", 0, 1)]
print(f"第1章の回路IR: {bell}、深さ {circuit_depth(bell, 2)}、"
      f"ゲート数 {gate_counts(bell)}")

# 書き換えの候補4つ。3つは2.1節の規則で、1つは誤りです。
candidates = [
    ("H H -> (nothing)", 1, [("h", 0), ("h", 0)], []),
    ("Rz(0.4) Rz(0.9) -> Rz(1.3)", 1,
     [("rz", 0.4, 0), ("rz", 0.9, 0)], [("rz", 1.3, 0)]),
    ("CX CX -> (nothing)", 2, [("cx", 0, 1), ("cx", 0, 1)], []),
    ("H H -> X", 1, [("h", 0), ("h", 0)], [("x", 0)]),
]
header = f"{'candidate rewrite':<30}{'n':>3}{'phase-free error':>18}  verdict"
print(f"\n{header}")
print("-" * len(header))
for label, n, before, after in candidates:
    err = phase_free_error(unitary_of(before, n), unitary_of(after, n))
    print(f"{label:<30}{n:>3}{err:>18.2e}  "
          f"{'accept' if err < 1e-10 else 'REJECT'}")
```

```text
第1章の回路IR: [('h', 0), ('cx', 0, 1)]、深さ 2、ゲート数 {'h': 1, 'cx': 1, '2q': 1}

candidate rewrite               n  phase-free error  verdict
------------------------------------------------------------
H H -> (nothing)                1          2.22e-16  accept
Rz(0.4) Rz(0.9) -> Rz(1.3)      1          1.11e-16  accept
CX CX -> (nothing)              2          0.00e+00  accept
H H -> X                        1          1.00e+00  REJECT
```

**着目点。** 最後の行が検査器の存在理由です。4つの候補のうち3つは上の表の規則で、4つ目はもっともらしく見えるだけの偽の主張です。位相を除いた誤差は両者を16桁の差で分離します。本章のすべてのパスはこの検査に包まれており、この検査を通れないパスは最適化ではなくバグです。

### Code Example 2: 規則と、交換可能性の判定関数

まず恒等式、次に判定関数、最後に判定関数と行列の交換関係との全数照合です。

```python
"""第2章 Example 2: 書き換え規則と、交換可能性の判定関数。
Code Example 1 の続き（同一セッション）。"""
import itertools
import numpy as np
from qcheck import *

PI = np.pi

# ---- Example 3 の最適化器を組み立てる元になる恒等式 ----------------------
IDENTITIES = [
    ("H H = I", 1, [("h", 0), ("h", 0)], []),
    ("S S = Z", 1, [("s", 0), ("s", 0)], [("z", 0)]),
    ("T T = S", 1, [("t", 0), ("t", 0)], [("s", 0)]),
    ("Rz(a) Rz(b) = Rz(a+b)", 1,
     [("rz", 0.4, 0), ("rz", -1.1, 0)], [("rz", -0.7, 0)]),
    ("H X H = Z", 1, [("h", 0), ("x", 0), ("h", 0)], [("z", 0)]),
    ("CX CX = I", 2, [("cx", 0, 1), ("cx", 0, 1)], []),
    ("CZ = H(1) CX H(1)", 2,
     [("cz", 0, 1)], [("h", 1), ("cx", 0, 1), ("h", 1)]),
    ("Z(0) CX = CX Z(0)", 2,
     [("z", 0), ("cx", 0, 1)], [("cx", 0, 1), ("z", 0)]),
    ("X(1) CX = CX X(1)", 2,
     [("x", 1), ("cx", 0, 1)], [("cx", 0, 1), ("x", 1)]),
    ("CX Z(1) CX = Z(0) Z(1)", 2,
     [("cx", 0, 1), ("z", 1), ("cx", 0, 1)], [("z", 0), ("z", 1)]),
    ("SWAP = 3 CX", 2, [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)],
     [("cx", 1, 0), ("cx", 0, 1), ("cx", 1, 0)]),
    ("CX(0,1) CX(0,2) commute", 3,
     [("cx", 0, 1), ("cx", 0, 2)], [("cx", 0, 2), ("cx", 0, 1)]),
    ("CX(0,1) CX(1,2) do NOT", 3,
     [("cx", 0, 1), ("cx", 1, 2)], [("cx", 1, 2), ("cx", 0, 1)]),
]

header = f"{'identity':<28}{'n':>3}{'phase-free error':>18}  holds"
print(header)
print("-" * len(header))
for label, n, left, right in IDENTITIES:
    err = phase_free_error(unitary_of(left, n), unitary_of(right, n))
    print(f"{label:<28}{n:>3}{err:>18.2e}  {'yes' if err < 1e-10 else 'NO'}")

# ---- 最適化器が参照する交換可能性の判定関数 ------------------------------
DIAGONAL = ("z", "s", "t", "rz", "cz")     # 計算基底で対角
X_LIKE = ("x", "rx")                       # X基底で対角


def family(g):
    """ゲートの交換族を返します。'diag'、'x'、'y'、または None です。"""
    if g[0] in DIAGONAL:
        return "diag"
    if g[0] in X_LIKE:
        return "x"
    if g[0] == "ry":
        return "y"
    return None


def commutes(g, h):
    """「g h = h g」に対する、健全だが意図的に不完全な判定です。

    健全とは、交換しない対に対して True を返すことが決してない、という意味です。
    したがってこれに基づくパスが回路の意味を変えることはありません。不完全とは、
    上の規則しか知らないので偶然の交換を見落とす、という意味です。
    """
    qg, qh = set(gate_qubits(g)), set(gate_qubits(h))
    if not (qg & qh):
        return True                            # 台が交わらない
    if g == h:
        return True                            # ゲートは自分自身と交換する
    fg, fh = family(g), family(h)
    if fg == "diag" and fh == "diag":
        return True                            # 対角ゲートは互いに交換する
    if len(qg) == 1 and len(qh) == 1 and fg is not None and fg == fh:
        return True                            # 同じ回転軸、同じ量子ビット
    if g[0] == "cx" and h[0] == "cx":
        return g[1] == h[1] or g[2] == h[2]    # 制御を共有、または標的を共有
    for a, b in ((g, h), (h, g)):
        if a[0] == "cx" and len(gate_qubits(b)) == 1:
            q = gate_qubits(b)[0]
            return family(b) == "diag" if q == a[1] else family(b) == "x"
        if a[0] == "cx" and b[0] == "cz":
            return a[2] not in gate_qubits(b)
        if a[0] == "cz" and len(gate_qubits(b)) == 1:
            return family(b) == "diag"
    return False


# ---- 判定関数を、行列の総当たり比較で検証する ----------------------------
LIBRARY = ([(name, q) for name in ("h", "x", "z", "s", "t") for q in range(3)]
           + [(name, 0.7, q) for name in ("rx", "ry", "rz") for q in range(3)]
           + [("cx", a, b) for a, b in itertools.permutations(range(3), 2)]
           + [("cz", a, b) for a, b in itertools.combinations(range(3), 2)]
           + [("ry", 0.0, 0), ("rz", 2 * PI, 1)])   # 変装した恒等ゲート2つ

mats = {g: unitary_of([g], 3) for g in LIBRARY}
truth_yes = predicate_yes = unsound = missed = 0
examples = []
for g, h in itertools.product(LIBRARY, repeat=2):
    A, B = mats[g], mats[h]
    truth = np.max(np.abs(A @ B - B @ A)) < 1e-12
    said = commutes(g, h)
    truth_yes += truth
    predicate_yes += said
    if said and not truth:
        unsound += 1
    if truth and not said:
        missed += 1
        if len(examples) < 3 and g[0] != h[0]:
            examples.append((g, h))

print(f"\n3量子ビット上の {len(LIBRARY)} 個のゲートに対する判定関数の全数検査")
print(f"  検査した順序対の数              : {len(LIBRARY) ** 2}")
print(f"  実際に交換する対の数            : {truth_yes}")
print(f"  判定関数が交換と認めた対の数    : {predicate_yes}")
print(f"  不健全な答え（0であるべき）     : {unsound}")
print(f"  見落とした真の交換の数          : {missed}")
print(f"  見落としの例                    : {examples[0]} | {examples[1]}")
```

```text
identity                      n  phase-free error  holds
--------------------------------------------------------
H H = I                       1          2.22e-16  yes
S S = Z                       1          0.00e+00  yes
T T = S                       1          1.11e-16  yes
Rz(a) Rz(b) = Rz(a+b)         1          5.55e-17  yes
H X H = Z                     1          2.22e-16  yes
CX CX = I                     2          0.00e+00  yes
CZ = H(1) CX H(1)             2          2.22e-16  yes
Z(0) CX = CX Z(0)             2          0.00e+00  yes
X(1) CX = CX X(1)             2          0.00e+00  yes
CX Z(1) CX = Z(0) Z(1)        2          0.00e+00  yes
SWAP = 3 CX                   2          0.00e+00  yes
CX(0,1) CX(0,2) commute       3          0.00e+00  yes
CX(0,1) CX(1,2) do NOT        3          1.00e+00  NO

3量子ビット上の 35 個のゲートに対する判定関数の全数検査
  検査した順序対の数              : 1225
  実際に交換する対の数            : 889
  判定関数が交換と認めた対の数    : 851
  不健全な答え（0であるべき）     : 0
  見落とした真の交換の数          : 38
  見落としの例                    : (('h', 0), ('ry', 0.0, 0)) | (('h', 1), ('rz', 6.283185307179586, 1))
```

**着目点。** 恒等式の表のうち1行は恒等式ではありません。`CX(0,1) CX(1,2) do NOT` は交換せず、誤差 $1.00$ がそう述べています。これを表に含めたのは、規則集合が何を含むかと同じくらい何を除外するかで定義されるからです。そして `CZ = H(1) CX H(1)` が厳密に成り立ちます。これはあらゆる超伝導向けコンパイラが行う基底変換であり、1行で書けます。

判定関数の報告が正直な部分です。1225個の順序対に対して不健全な答えが0であることが重要な性質です。この判定関数に基づくパスが回路の意味を変えることはありえません。38件の見落としはすべて、ライブラリに意図的に混ぜた2つのゲート — $R_y(0)$ と $R_z(2\pi)$、すなわち構文的な規則には認識できない形で書かれた恒等ゲート — に関わるものです。対策はより良い判定関数ではなく、角度を $(-\pi, \pi]$ に折り込み、0回転を削除する**正規化**パスをどの規則より先に置くことであり、Code Example 3 の最適化器はそれを備えています。

### Code Example 3: ピープホール最適化器

最適化器は2つのパスを不動点まで回すものです。`local_pass` は1ゲート規則と隣接対規則を適用し、`commute_pass` は1つのゲートを交換可能なゲートを越えて前へ滑らせ、その移動が融合を生む場合にだけ発動します。停止性は直ちに従います。成功する規則は必ず回路を厳密に短くするので、ループは `len(circ)` 回より多くは回れません。

```python
"""第2章 Example 3: これらの規則から組み立てたピープホール最適化器。
Code Example 2 の続き（同一セッション）。"""
import numpy as np
from qcheck import *

TWO_PI = 2.0 * np.pi
DIAG_ANGLE = {"z": np.pi, "s": np.pi / 2, "t": np.pi / 4}
NAMED = (("diag", np.pi, "z"), ("diag", np.pi / 2, "s"),
         ("diag", np.pi / 4, "t"), ("x", np.pi, "x"))
AXIS_GATE = {"diag": "rz", "x": "rx", "y": "ry"}


def as_angle(g):
    """g が x, y, z いずれかの軸の回転なら (軸, 角度, 量子ビット)、他は None です。"""
    if g[0] in DIAG_ANGLE:
        return "diag", DIAG_ANGLE[g[0]], g[1]
    if g[0] == "x":
        return "x", np.pi, g[1]
    if g[0] in ("rx", "ry", "rz"):
        return {"rz": "diag", "rx": "x", "ry": "y"}[g[0]], g[1], g[2]
    return None


def wrap(theta):
    """角度を (-pi, pi] に折り込みます。差はグローバル位相だけです。"""
    t = (theta + np.pi) % TWO_PI - np.pi
    return np.pi if abs(t + np.pi) < 1e-12 else t


def canonical(axis, theta, q):
    """0個、名前付きゲート1個、回転1個のうち最も短いものを返します。"""
    theta = wrap(theta)
    if abs(theta) < 1e-12:
        return []
    for a, ref, name in NAMED:
        if a == axis and abs(theta - ref) < 1e-12:
            return [(name, q)]
    return [(AXIS_GATE[axis], theta, q)]


def simplify_one(g):
    """1ゲート規則。角度を折り込み、恒等ゲートを削り、名前が付くものに名前を付けます。"""
    a = as_angle(g)
    if a is None:
        return None
    out = canonical(*a)
    if len(out) == 1 and out[0][0] == g[0]:
        folded = g[0] not in ("rx", "ry", "rz") or abs(wrap(g[1]) - g[1]) < 1e-9
        return None if folded else out
    return out


def merge_two(g, h):
    """同じ量子ビット上で隣接する対の置換結果、または None を返します。"""
    ag, ah = as_angle(g), as_angle(h)
    if ag is not None and ah is not None and ag[0] == ah[0] and ag[2] == ah[2]:
        return canonical(ag[0], ag[1] + ah[1], ag[2])      # 同じ軸なら角度を足す
    if g == h and g[0] in ("h", "cx"):
        return []                                          # 自己逆元
    if g[0] == "cz" and h[0] == "cz":
        return []                                          # CZ は対称
    return None


def next_touching(circ, i, qs):
    """i より後で qs と量子ビットを共有する最初のゲートの添字、なければ len(circ) です。"""
    j = i + 1
    while j < len(circ) and not (qs & set(gate_qubits(circ[j]))):
        j += 1
    return j


def local_pass(circ):
    """1ゲート規則と隣接対規則を1回だけ走査します。"""
    out, i, fired = list(circ), 0, 0
    while i < len(out):
        rule = simplify_one(out[i])
        if rule is not None:
            out[i:i + 1] = rule
            fired += 1
            i = max(i - 1, 0)
            continue
        qs = set(gate_qubits(out[i]))
        j = next_touching(out, i, qs)
        if j < len(out) and set(gate_qubits(out[j])) == qs:
            merged = merge_two(out[i], out[j])
            if merged is not None:
                del out[j]
                out[i:i + 1] = merged
                fired += 1
                i = max(i - 1, 0)
                continue
        i += 1
    return out, fired


def commute_pass(circ):
    """交換可能なゲートを越えて1つのゲートを前へ滑らせ、融合できるときだけ実行します。"""
    for i, g in enumerate(circ):
        qs = set(gate_qubits(g))
        for j in range(i + 1, len(circ)):
            h = circ[j]
            if not (set(gate_qubits(h)) & qs):
                continue                        # 台が交わらない。滑って通過
            merged = merge_two(g, h) if set(gate_qubits(h)) == qs else None
            if merged is not None:
                if j > i + 1:                   # j == i + 1 は局所パスの仕事
                    return circ[:i] + circ[i + 1:j] + merged + circ[j + 1:], 1
                break
            if not commutes(g, h):
                break                           # 遮られた。この i では何もしない
    return list(circ), 0


def peephole(circ, trace=None):
    """両方のパスを不動点まで回します。成功するたび回路は必ず短くなります。"""
    cur = list(circ)
    while True:
        cur, a = local_pass(cur)
        cur, b = commute_pass(cur)
        if trace is not None:
            trace.append((len(cur), a, b))
        if a == 0 and b == 0:
            return cur


# ---- すべての規則を発動させるために手で組んだ回路 ------------------------
demo = [("h", 0), ("h", 0),                            # H H = I
        ("t", 1), ("t", 1),                            # T T = S
        ("rz", 0.3, 2), ("rz", -0.3, 2),               # 互いに逆の回転
        ("cx", 0, 1), ("rz", 0.5, 0), ("cx", 0, 1),    # Rz が制御側を滑り抜ける
        ("h", 2), ("x", 2), ("h", 2),                  # H X H = Z。規則にはない
        ("ry", 0.0, 1),                                # 変装した恒等ゲート
        ("s", 1), ("s", 1),                            # S S = Z
        ("cz", 0, 2), ("t", 0), ("cz", 2, 0)]          # T が CZ を滑り抜ける

trace = []
opt = peephole(demo, trace)
print("2.1節のすべての規則を発動させるために組んだ回路")
print(f"  前: {len(demo)} ゲート、深さ {circuit_depth(demo, 3)}、"
      f"{gate_counts(demo)['2q']} 個の2量子ビットゲート")
print(f"  後: {len(opt)} ゲート、深さ {circuit_depth(opt, 3)}、"
      f"{gate_counts(opt)['2q']} 個の2量子ビットゲート")


def show(circ):
    """回路を読める文字列にします。角度は丸めます。"""
    return " ".join(
        g[0] + "(" + ",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                              for v in g[1:]) + ")" for g in circ) or "(empty)"


print(f"  結果: {show(opt)}")
print(f"  等価性の誤差: {assert_equivalent(demo, opt, 3, 'peephole'):.2e}")

print(f"\n{'round':>6}{'gates left':>12}{'local rules':>13}{'commutations':>14}")
print("-" * 45)
for k, (size, a, b) in enumerate(trace, start=1):
    print(f"{k:>6}{size:>12}{a:>13}{b:>14}")
```

```text
2.1節のすべての規則を発動させるために組んだ回路
  前: 18 ゲート、深さ 8、4 個の2量子ビットゲート
  後: 5 ゲート、深さ 3、0 個の2量子ビットゲート
  結果: rz(-1.5708,1) h(2) x(2) h(2) rz(1.2854,0)
  等価性の誤差: 4.58e-16

 round  gates left  local rules  commutations
---------------------------------------------
     1           9            5             1
     2           7            1             1
     3           5            0             1
     4           5            0             0
```

**着目点。** 18ゲートが5ゲートになり、4つの2量子ビットゲートがすべて消えます。どちらのパス単独でも到達できません。2つのCXの間の $R_z(0.5)$ と2つのCZの間の $T$ が相殺を遮っており、その遮りを外すにはそれぞれ交換が1回必要です。ラウンドの表がその交錯を示しています。第1ラウンドで局所規則が5回、その後は遮りが外れるたびに次が現れるので3ラウンドにわたって交換が1回ずつです。

残ったものも示唆的です。量子ビット1上の $R_z(-\pi/2)$ は4つの対角ゲート $T\,T\,S\,S$ が1つに畳まれたもの、量子ビット0上の $R_z(0.5 + \pi/4)$ は2つのCXの間にあった回転が2つのCZの間にあった $T$ と融合したものです。そして量子ビット2上の $H\,X\,H$ は手つかずです。Code Example 2 のどの規則も $H$ による共役を書き換えないからです。恒等式 $H X H = Z$ は表にはありますが、*最適化器*にはその形の規則がありません。2.2節はこれを別の道筋で除去します。共役の規則を1つずつ足していくのではなく、再合成するという道筋であり、こちらが正しい向きです。

### Code Example 4: 最適化器が稼ぐもの

収量は入力の上で計測しなければならず、中立な入力というものは存在しません。ここでの生成器は第1章の `random_circuit` を逐語で再掲したもので、これによりあの章の弱いパスが出した数値と比較できます。最初のブロックは答えが分かっているテストです。回路の後に自身の逆回路を置いたものは、空回路に最適化されなければなりません。

```python
"""第2章 Example 4: 最適化器がどれだけ稼ぐかを計測します。
Code Example 3 の続き（同一セッション）。"""
import numpy as np
from qcheck import *

NAMES = ["h", "x", "z", "s", "t", "rx", "ry", "rz", "cx", "cz"]
ANGLES = [k * np.pi / 4 for k in (-3, -2, -1, 1, 2, 3, 4)]


def random_circuit(n, length, rng):
    """IRのゲート集合上のランダム回路。角度は pi/4 の整数倍です。

    第1章 Example 5 から逐語で再掲しています。あの章の弱いパスが出した数値と
    以下の数値が比較できるようにするためです。
    """
    circ = []
    for _ in range(length):
        name = NAMES[int(rng.integers(len(NAMES)))]
        if name in TWO_Q:
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            circ.append((name, a, b))
        elif name in ROT_1Q:
            circ.append((name, ANGLES[int(rng.integers(len(ANGLES)))],
                         int(rng.integers(n))))
        else:
            circ.append((name, int(rng.integers(n))))
    return circ


def inverse(circ):
    """逆回路。順序を反転し、各ゲートを逆元に置き換えます。"""
    inv = []
    for g in reversed(circ):
        if g[0] in ("h", "x", "z", "cx", "cz"):
            inv.append(g)                                  # 自己逆元
        elif g[0] == "s":
            inv.append(("rz", -np.pi / 2, g[1]))
        elif g[0] == "t":
            inv.append(("rz", -np.pi / 4, g[1]))
        else:
            inv.append((g[0], -g[1], g[2]))
    return inv


# ---- 答えが分かっているテスト。U の後に U の逆を置けば消えるはず ---------
print("健全性テスト: 回路の後に自身の逆回路を置くと、最適化で何も残らないはずです")
print(f"{'seed':>5}{'n':>4}{'gates in':>10}{'gates out':>11}{'error':>11}")
print("-" * 41)
for seed in range(3):
    rng = np.random.default_rng(seed)
    c = random_circuit(4, 30, rng)
    pair = c + inverse(c)
    out = peephole(pair)
    print(f"{seed:>5}{4:>4}{len(pair):>10}{len(out):>11}"
          f"{assert_equivalent(pair, out, 4, 'inverse pair'):>11.1e}")

# ---- 乱数種を固定したランダム回路 ---------------------------------------
print("\n種を固定したランダム回路。5量子ビット、60ゲート、1本ずつ検証します")
head = (f"{'seed':>5}{'gates':>13}{'depth':>11}{'two-qubit':>12}"
        f"{'phase-free err':>16}")
print(head)
print("-" * len(head))
tot = {"g0": 0, "g1": 0, "d0": 0, "d1": 0, "q0": 0, "q1": 0}
worst = 0.0
n, m, trials = 5, 60, 200
for seed in range(trials):
    rng = np.random.default_rng(1000 + seed)
    c = random_circuit(n, m, rng)
    o = peephole(c)
    err = assert_equivalent(c, o, n, f"seed {seed}")
    worst = max(worst, err)
    g0, g1 = len(c), len(o)
    d0, d1 = circuit_depth(c, n), circuit_depth(o, n)
    q0, q1 = gate_counts(c)["2q"], gate_counts(o)["2q"]
    for k, v in zip(("g0", "g1", "d0", "d1", "q0", "q1"), (g0, g1, d0, d1, q0, q1)):
        tot[k] += v
    if seed < 4:
        print(f"{seed:>5}{f'{g0} -> {g1}':>13}{f'{d0} -> {d1}':>11}"
              f"{f'{q0} -> {q1}':>12}{err:>16.1e}")
print(f"{'...':>5}")
print(f"\n{trials} 本の回路の平均（n = {n}、各 {m} ゲート）:")
print(f"  gates      {tot['g0']/trials:6.2f} -> {tot['g1']/trials:6.2f}"
      f"   ({100*(1-tot['g1']/tot['g0']):5.1f}% removed)")
print(f"  depth      {tot['d0']/trials:6.2f} -> {tot['d1']/trials:6.2f}"
      f"   ({100*(1-tot['d1']/tot['d0']):5.1f}% removed)")
print(f"  two-qubit  {tot['q0']/trials:6.2f} -> {tot['q1']/trials:6.2f}"
      f"   ({100*(1-tot['q1']/tot['q0']):5.1f}% removed)")
print(f"  {trials} 本すべてを通じた等価性誤差の最大値: {worst:.2e}")
```

```text
健全性テスト: 回路の後に自身の逆回路を置くと、最適化で何も残らないはずです
 seed   n  gates in  gates out      error
-----------------------------------------
    0   4        60          0    1.6e-15
    1   4        60          0    7.9e-16
    2   4        60          0    1.1e-15

種を固定したランダム回路。5量子ビット、60ゲート、1本ずつ検証します
 seed        gates      depth   two-qubit  phase-free err
---------------------------------------------------------
    0     60 -> 40   23 -> 18    13 -> 11         3.2e-16
    1     60 -> 44   26 -> 22    15 -> 15         2.7e-16
    2     60 -> 47   32 -> 29    15 -> 15         3.3e-16
    3     60 -> 48   22 -> 19    14 -> 14         2.8e-16
  ...

200 本の回路の平均（n = 5、各 60 ゲート）:
  gates       60.00 ->  44.20   ( 26.3% removed)
  depth       22.65 ->  18.39   ( 18.8% removed)
  two-qubit   12.13 ->  11.28   (  7.1% removed)
  200 本すべてを通じた等価性誤差の最大値: 6.78e-16
```

**着目点。** 逆回路のテストが本章で最も強い正しさの証拠です。60ゲートが入って0ゲートが出て、それが3回続き、等価性検査は空回路が本当に入力の計算内容であることを確認します。相殺規則に微妙な誤りがあれば残骸が残るか検査に落ちるかのどちらかであり、このテストは両方を捕まえます。一様ランダム回路での収量はゲートの4分の1、深さの5分の1、そして — 重要な数値ですが — 2量子ビットゲートのわずか7パーセントです。この最後の数値が正直な見出しです。ピープホール最適化は局所的な手法であり、一様ランダム回路は構成上局所的な冗長性をほとんど持たず、その2量子ビットゲートは大半が別の対に作用して出会うことがありません。コンパイラが吐いた回路は逆の場合です。各段が自分の基底変換を加え、その各々が次の段の基底変換の逆と出会うからです。第1章 Example 5 で各CXを独立に変換したCX鎖が、この最適化器が食べる形そのものです。

* * *

## 2.2 1量子ビットの合成

### 3つの角と1つの位相

$U(2)$ のすべての元は

$$ U = e^{i\delta}\, R_z(a)\, R_y(b)\, R_z(c), \qquad R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \cr 0 & e^{i\theta/2} \end{pmatrix}, \quad R_y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \cr \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix} $$

と書けます。数は合っています。$U(2)$ は実4パラメータであり、この形も4つです。構成は初等的です。$U$ を $\sqrt{\det U}$ で割って $SU(2)$ の元 $V$ を得ます。これで $\delta = \frac{1}{2}\arg\det U$ が決まります。すると

$$ V = \begin{pmatrix} \cos\frac{b}{2}\, e^{-i(a+c)/2} & -\sin\frac{b}{2}\, e^{-i(a-c)/2} \cr \sin\frac{b}{2}\, e^{i(a-c)/2} & \cos\frac{b}{2}\, e^{i(a+c)/2} \end{pmatrix} $$

なので、第1列の絶対値から $b = 2\arctan\big(|V_{10}|/|V_{00}|\big)$ が得られ、$V_{11}$ と $V_{10}$ の位相からそれぞれ $(a+c)/2$ と $(a-c)/2$ が得られます。したがって

$$ a = \arg V_{11} + \arg V_{10}, \qquad c = \arg V_{11} - \arg V_{10} $$

です。退化した分岐が2つあり、これを無視するルーチンは `nan` を返します。$V_{00} = 0$ なら $b = \pi$ で $a - c$ だけが決まり、$V_{10} = 0$ なら $b = 0$ で $a + c$ だけが決まります。どちらの場合も2つの $R_z$ 角の一方を0に取れます。だからこそ以下で $Z$、$S$、$T$ の合成が3個ではなく1個のゲートになります。

### なぜ位相を返さなければならないのか

$\delta$ はグローバル位相なので観測できず、それでも返さなければなりません。理由は第1章の演習3が述べたとおりです。**グローバル位相は、その断片が制御の下に置かれた瞬間にグローバルでなくなります**。$\delta$ を捨てるルーチンは正しい1量子ビット回路と誤った制御$U$ を作ります。位相が制御ブロックの片方の枝にだけ乗るからです。2.3節はまさにそれを必要とし、だから `zyz_angles` は4つの数を返します。

### 最適化パスとしての再合成

1量子ビットの合成ができれば、それを翻訳器ではなく最適化器として使えます。1つの量子ビット上で連続する1量子ビットゲートの極大な連なりを取り、行列を掛け合わせて再合成すれば、連なりが何であったかに関わらず結果は高々3個の回転です。これは有限個の書き換え規則の並びには持ちえない性質 — 1量子ビットの連なりに対する*完全性* — を持ちます。$H X H$ は3ゲートの連なりで、その積は $Z$ ですから、誰も恒等式 $H X H = Z$ を書き下さなくても再合成は1ゲートを返します。

このパスは代償を伴います。入力が名前付きゲートだった箇所に `rz` と `ry` の回転を吐き出すので、ネイティブ集合が異なる機械では後退になりえます。だから実運用のコンパイラはこれを遅い段で、ハードウェアに合わせた基底で走らせます。

### Code Example 5: ZYZ分解と再合成パス

```python
"""第2章 Example 5: ZYZ合成と、それが回路に対して何をするか。
Code Example 4 の続き（同一セッション）。"""
import numpy as np
from qcheck import *


def zyz_angles(U):
    """任意の U in U(2) に対し U = exp(i delta) Rz(a) Ry(b) Rz(c) となる (delta, a, b, c)。"""
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    delta = 0.5 * np.angle(det)                  # U / exp(i delta) は SU(2) の元
    V = U * np.exp(-1j * delta)
    b = 2.0 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    if abs(V[0, 0]) < 1e-12:                     # b = pi。a - c だけが決まる
        a, c = 2.0 * np.angle(V[1, 0]), 0.0
    elif abs(V[1, 0]) < 1e-12:                   # b = 0。a + c だけが決まる
        a, c = 2.0 * np.angle(V[1, 1]), 0.0
    else:
        p, m = np.angle(V[1, 1]), np.angle(V[1, 0])
        a, c = p + m, p - m
    return delta, a, b, c


def zyz_circuit(U, q):
    """1量子ビットゲートを、量子ビット q 上の高々3個の回転として回路順に返します。"""
    _, a, b, c = zyz_angles(U)
    out = []
    for axis, theta in (("diag", c), ("y", b), ("diag", a)):
        out += canonical(axis, theta, q)
    return out


def haar_1q(rng):
    """複素ガウス行列のQR分解による、U(2) のHaarランダムな元です。"""
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / abs(np.diag(R)))


rng = np.random.default_rng(11)
worst_exact = worst_free = 0.0
lengths = {0: 0, 1: 0, 2: 0, 3: 0}
for _ in range(2000):
    U = haar_1q(rng)
    delta, a, b, c = zyz_angles(U)
    rebuilt = np.exp(1j * delta) * (rz(a) @ ry(b) @ rz(c))
    worst_exact = max(worst_exact, np.max(np.abs(U - rebuilt)))
    circ = zyz_circuit(U, 0)
    worst_free = max(worst_free, phase_free_error(U, unitary_of(circ, 1)))
    lengths[len(circ)] += 1

print("U(2) のHaarランダムな元2000個に対するZYZ分解")
print(f"  誤差の最大値（位相を含む）  : {worst_exact:.2e}")
print(f"  誤差の最大値（位相を除く）  : {worst_free:.2e}")
print(f"  合成結果のゲート数の分布    : {lengths}")

print("\n2つのRz角のいずれかが決まらない特別な場合")
for label, U in [("identity", np.eye(2, dtype=complex)), ("H", H),
                 ("Z", Z), ("T", T), ("Ry(0.7)", ry(0.7))]:
    d, a, b, c = zyz_angles(U)
    circ = zyz_circuit(U, 0)
    print(f"  {label:<10} delta/pi = {d/np.pi:+.3f}  "
          f"(a,b,c)/pi = ({a/np.pi:+.3f},{b/np.pi:+.3f},{c/np.pi:+.3f})  "
          f"-> {len(circ)} gate(s)")


# ---- パス本体。1量子ビットゲートの連なりをすべて畳み込む -----------------
def matrix_1q(g):
    """1量子ビットゲートタプルの 2x2 行列です。"""
    if g[0] in FIXED_1Q:
        return FIXED_1Q[g[0]]
    if g[0] in ROT_1Q:
        return ROT_1Q[g[0]](g[1])
    return None


def resynthesize(circ, n):
    """1つの量子ビット上の1量子ビットゲートの極大連鎖を、そのZYZ形に置き換えます。

    量子ビットごとに独立に吐き出すので、台が交わらないゲートの順序が入れ替わり
    ます。これは常に許される操作であり、検査器がそれを確認します。
    """
    pending = {q: [] for q in range(n)}
    out = []

    def flush(q):
        run, pending[q] = pending[q], []
        if len(run) == 1:
            out.append(run[0][0])              # 1個だけなら改善の余地はない
        elif run:
            U = np.eye(2, dtype=complex)
            for _, M in run:
                U = M @ U
            out.extend(zyz_circuit(U, q))

    for g in circ:
        qs = gate_qubits(g)
        M = matrix_1q(g) if len(qs) == 1 else None
        if M is not None:
            pending[qs[0]].append((g, M))
        else:
            for q in qs:
                flush(q)
            out.append(g)
    for q in range(n):
        flush(q)
    return out


print("\nExample 3 のピープホール規則が手を出せなかった H X H:")
block = [("h", 0), ("x", 0), ("h", 0)]
print(f"  ピープホール : {show(peephole(block))}")
print(f"  再合成       : {show(resynthesize(block, 1))}")
print(f"  誤差         : {assert_equivalent(block, resynthesize(block, 1), 1):.2e}")

print("\n2つのパスの組み合わせ。Example 4 の種固定ランダム回路上で")
head = (f"{'pipeline':<34}{'gates':>8}{'1q':>6}{'2q':>6}{'depth':>7}"
        f"{'worst err':>12}")
print(head)
print("-" * len(head))
n, m, trials = 5, 60, 200
stages = [("as written", lambda c: c),
          ("peephole only", lambda c: peephole(c)),
          ("ZYZ resynthesis only", lambda c: resynthesize(c, n)),
          ("peephole, resynthesis, peephole",
           lambda c: peephole(resynthesize(peephole(c), n)))]
for label, pipeline in stages:
    g = q = d = 0
    worst = 0.0
    for seed in range(trials):
        c = random_circuit(n, m, np.random.default_rng(1000 + seed))
        o = pipeline(c)
        worst = max(worst, assert_equivalent(c, o, n, label))
        g += len(o)
        q += gate_counts(o)["2q"]
        d += circuit_depth(o, n)
    print(f"{label:<34}{g/trials:>8.2f}{(g-q)/trials:>6.2f}{q/trials:>6.2f}"
          f"{d/trials:>7.2f}{worst:>12.1e}")
```

```text
U(2) のHaarランダムな元2000個に対するZYZ分解
  誤差の最大値（位相を含む）  : 8.47e-16
  誤差の最大値（位相を除く）  : 9.49e-16
  合成結果のゲート数の分布    : {0: 0, 1: 0, 2: 0, 3: 2000}

2つのRz角のいずれかが決まらない特別な場合
  identity   delta/pi = +0.000  (a,b,c)/pi = (+0.000,+0.000,+0.000)  -> 0 gate(s)
  H          delta/pi = +0.500  (a,b,c)/pi = (+0.000,+0.500,+1.000)  -> 2 gate(s)
  Z          delta/pi = +0.500  (a,b,c)/pi = (+1.000,+0.000,+0.000)  -> 1 gate(s)
  T          delta/pi = +0.125  (a,b,c)/pi = (+0.250,+0.000,+0.000)  -> 1 gate(s)
  Ry(0.7)    delta/pi = +0.000  (a,b,c)/pi = (+0.000,+0.223,+0.000)  -> 1 gate(s)

Example 3 のピープホール規則が手を出せなかった H X H:
  ピープホール : h(0) x(0) h(0)
  再合成       : z(0)
  誤差         : 2.22e-16

2つのパスの組み合わせ。Example 4 の種固定ランダム回路上で
pipeline                             gates    1q    2q  depth   worst err
-------------------------------------------------------------------------
as written                           60.00 47.87 12.13  22.65     0.0e+00
peephole only                        44.20 32.93 11.28  18.39     6.8e-16
ZYZ resynthesis only                 43.25 31.11 12.13  18.93     1.1e-15
peephole, resynthesis, peephole      36.40 25.14 11.27  16.54     1.0e-15
```

**着目点。** 分解は2000個のHaarランダム入力に対し、位相を含めても除いても $10^{-15}$ の精度で厳密です。そして一般の入力では常に3ゲートを生みます。一般の $U(2)$ の元は自明でない角を3つ持つので、そうならなければなりません。特別な場合の表は退化した分岐が働いていることを示します。$Z$、$T$、$R_y(0.7)$ はそれぞれ1ゲート、恒等元は0ゲート、$H$ は2ゲートです。

パイプラインの表が正直な比較であり、その800行はすべて検査済みです。3通りの最適化パイプラインにわたる書き換え600件の検証に加え、`as written` パイプライン200回 — これは恒等写像であり、したがって無操作の対照 — の検証です。単独では2つのパスは近く、失敗する箇所が異なります。再合成は43.25ゲートでピープホール規則の44.20より良いのですが、その2量子ビット数は入力値そのままの12.13です。2量子ビットゲートを除去できるのはピープホール規則だけだからです。1量子ビットゲートの連なりを、その中身に関わらず3回転で抑えられるのは再合成だけであり、だから $H X H$ が消えます。両方合わせると36.40ゲート・深さ16.54に達し、書いたままの60.00・22.65と比べられます。最後のピープホールパスは飾りではありません。再合成が吐いた回転は交換の新しい候補であり、パスはピープホール規則単独より2量子ビットゲートを2個多く見つけます。200本の回路全体で2255個に対して2253個であり、これが平均値11.28と11.27の差のすべてです。

* * *

## 2.3 2量子ビットの合成

### 2量子ビットゲートの構造

1量子ビットの話は一般化され、その一般化が2量子ビットのコンパイルの中心定理です。任意の $U \in SU(4)$ は

$$ U = (A_1 \otimes A_2)\, \exp\big[ i\big( t_x\, X{\otimes}X + t_y\, Y{\otimes}Y + t_z\, Z{\otimes}Z \big) \big]\, (B_1 \otimes B_2) $$

と $A_i, B_i \in SU(2)$ を用いて分解できます。これが**KAK分解**、すなわち部分群 $SU(2)\otimes SU(2)$ に関する $SU(4)$ のCartan分解です。中央の因子が**正準ゲート**で、3つの数 $(t_x, t_y, t_z)$ — 自明な対称性で割れば四面体に収まります — が $U$ の非局所的な内容のすべてです。残りは局所的であり、局所ゲートは重要な意味で無料です。エンタングルメントを消費せず、ハードウェア上でも2量子ビットゲートの100分の1程度の誤差しか持ちません。

ここから2つの帰結が従います。正準座標が一致する2つのゲートは必要なCX数も同じなので、「このゲートはどれだけ高価か」は $4\times4$ 行列についての問いではなく四面体上の1点についての問いになります。そして有名な目印はその角です。CXとCZは $(\pi/4, 0, 0)$、iSWAPは $(\pi/4, \pi/4, 0)$、SWAPは $(\pi/4, \pi/4, \pi/4)$、恒等元は原点です。

### 明示的な構成

4つの構成でコンパイラが吐くもののほとんどが覆え、いずれも厳密です。

| 対象 | CX数 | 構成 |
| --- | --- | --- |
| CZ | 1 | CXの標的側を $H$ で共役する |
| $\exp(-i\frac{\theta}{2} Z{\otimes}Z)$ | 2 | $\mathrm{CX}$、標的側の $R_z(\theta)$、$\mathrm{CX}$。$H\otimes H$ で共役すれば $XX$ 版 |
| 一般の $U$ に対する制御$U$ | 2 | ZYZ角から作る $ABC = I$ の $A X B X C$ |
| SWAP | 3 | $\mathrm{CX}_{0,1}\,\mathrm{CX}_{1,0}\,\mathrm{CX}_{0,1}$ |

書き出す価値があるのは制御$U$ の構成です。2.2節が位相 $\delta$ を返すことにこだわった理由がここで報われます。$U = e^{i\delta}R_z(a)R_y(b)R_z(c)$ に対して

$$ A = R_z(a)R_y(b/2), \qquad B = R_y(-b/2)R_z\left(-\tfrac{a+c}{2}\right), \qquad C = R_z\left(\tfrac{c-a}{2}\right) $$

と置きます。すると $ABC = R_z(a)R_z(-a) = I$ なので、制御が $|0\rangle$ の枝では何も起きません。また $X R_z(\theta) X = R_z(-\theta)$、$X R_y(\theta) X = R_y(-\theta)$ より

$$ A\,X B X\,C = R_z(a)\,R_y(b/2)\cdot R_y(b/2) R_z\left(\tfrac{a+c}{2}\right)\cdot R_z\left(\tfrac{c-a}{2}\right) = R_z(a)R_y(b)R_z(c) $$

となるので、$|1\rangle$ の枝は因子 $e^{i\delta}$ を除いて $U$ を作用させ、その因子は*制御*線上の $R_z(\delta)$ が戻します。2つの $X$ が2つのCXゲートです。位相を気にしない実装で何が壊れるかに注意してください。角度が許すところで放出される回転を $S$ や $T$ に改名すると — 制御されていないゲートに対して最適化器の `canonical` が行うのは正しくそれです — $ABC$ が $I$ ではなく $e^{i\varphi}I$ になり、2つの枝が別々の位相を拾い、結果は制御$U$ ではなくなります。Code Example 6 がこのために別の放出関数 `rot` を持つのはこの理由です。

### CX数と、その読み取り方

与えられた2量子ビットゲートに必要なCX数は何個でしょうか。単純なパラメータ数え上げでは答えられません。両側と中間に局所ブロックを置いた2個のCXの回路は、すでに $3 \times 2 \times 3 = 18$ 個の自由パラメータを持ち、$PU(4)$ の15を上回ります。したがって障害は次元的ではなく構造的です。これを解くのは、両側の局所ゲートで不変な2つの量です。

**マジック基底**

$$ M = \frac{1}{\sqrt2}\begin{pmatrix} 1 & 0 & 0 & i \cr 0 & i & 1 & 0 \cr 0 & i & -1 & 0 \cr 1 & 0 & 0 & -i \end{pmatrix} $$

を用い、$U$ を $SU(4)$ に規格化して $\tilde U = M^\dagger U M$ と置き、

$$ m(U) = \tilde U^{\mathsf T}\, \tilde U $$

とします。$m$ のスペクトルは $U \mapsto (A_1\otimes A_2) U (B_1 \otimes B_2)$ で変わりません。マジック基底では局所ゲートが*実*直交行列になり、$m$ は実直交な因子が打ち消えるように作られているからです。したがって $\operatorname{tr} m$ と $\operatorname{tr} m^2$ は $(t_x,t_y,t_z)$ だけの関数であり、分類は次のようになります。

| $m(U)$ に対する条件 | 最小CX数 | 正準座標 |
| --- | --- | --- |
| $\lvert\operatorname{tr} m\rvert = 4$ かつ $\operatorname{tr} m^2 = 4$ | 0 | $(0,0,0)$ |
| $\operatorname{tr} m = 0$ かつ $\operatorname{tr} m^2 = -4$ | 1 | $(\pi/4, 0, 0)$ |
| $\operatorname{tr} m$ が実 | 2 | $t_z = 0$ |
| それ以外 | 3 | 一般の場合 |

$(\det U)^{-1/4}$ による規格化には $m$ の符号を反転させる4通りの任意性がありますが、表のすべての項目はその符号に鈍感なので、分枝を選ばずに使えます。そして3行目が罠です。SWAPは $\operatorname{tr} m = -4i$ と純虚なので、$(\operatorname{tr} m)^2$ は実の負数になります。$\operatorname{tr} m$ ではなく $(\operatorname{tr} m)^2$ で述べた判定条件はしたがってSWAPを2CXのゲートと分類してしまいますが、それは誤りです。

3個のCXが常に*十分*であることはKAK定理の構成的な半分であり、ここでは再導出しません。最も難しい類であるSWAPに対する明示的な回路は上の表にあります。

### Code Example 6: 2量子ビット合成とCX数

```python
"""第2章 Example 6: 2量子ビット合成とCX数。
Code Example 5 の続き（同一セッション）。"""
import numpy as np
from qcheck import *


def controlled(U):
    """量子ビット0が |1> のとき量子ビット1に U を作用させる2量子ビット行列（ビッグエンディアン）。"""
    C = np.eye(4, dtype=complex)
    C[2:, 2:] = U
    return C


def rot(axis, theta, q):
    """厳密な回転ゲート1個、角度が0なら何も返しません。

    ここで canonical() を使ってはいけません。Rz(pi/2) を S に改名するとゲートは
    位相分だけ変わり、制御ブロックの片方の枝に付く位相はグローバルではありません。
    """
    if abs(theta) < 1e-12:
        return []
    return [({"diag": "rz", "x": "rx", "y": "ry"}[axis], theta, q)]


def controlled_circuit(U, control, target):
    """U のZYZ角から作る、CXゲート2個の制御U。

    U = exp(i d) Rz(a) Ry(b) Rz(c) に対し A = Rz(a) Ry(b/2)、
    B = Ry(-b/2) Rz(-(a+c)/2)、C = Rz((c-a)/2) と置くと A B C = I であり、
    A X B X C は exp(i d) を除いて U です。この因子は制御線上の Rz(d) が戻します。
    """
    d, a, b, c = zyz_angles(U)
    out = rot("diag", (c - a) / 2, target)                  # C
    out.append(("cx", control, target))
    out += rot("diag", -(a + c) / 2, target)                # B。右端の因子を先に
    out += rot("y", -b / 2, target)
    out.append(("cx", control, target))
    out += rot("y", b / 2, target)                          # A。右端の因子を先に
    out += rot("diag", a, target)
    out += rot("diag", d, control)                          # 残った位相
    return out


rng = np.random.default_rng(23)
worst, counts = 0.0, {}
for _ in range(400):
    U = haar_1q(rng)
    circ = controlled_circuit(U, 0, 1)
    worst = max(worst, phase_free_error(controlled(U), unitary_of(circ, 2)))
    k = gate_counts(circ)["2q"]
    counts[k] = counts.get(k, 0) + 1
print("CXゲート2個の制御U。U(2) のHaarランダムな U を400個")
print(f"  位相を除いた誤差の最大値: {worst:.2e}")
print(f"  使われたCX数            : {counts}")


# ---- 標準的な明示的構成。すべて検証済み ---------------------------------
def rzz_circuit(theta, a, b):
    """CXゲート2個による exp(-i theta/2 Z Z) です。"""
    return [("cx", a, b), ("rz", theta, b), ("cx", a, b)]


def exp_pauli(theta, P):
    """P P = I を満たすPauli語 P に対する exp(-i theta/2 P) です。"""
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * P


ZZ = np.kron(Z, Z)
SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 dtype=complex)
ISWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                  dtype=complex)

CONSTRUCTIONS = [
    ("CZ", CZ4, [("h", 1), ("cx", 0, 1), ("h", 1)]),
    ("controlled-T", controlled(T), controlled_circuit(T, 0, 1)),
    ("exp(-i 0.3 Z Z)", exp_pauli(0.6, ZZ), rzz_circuit(0.6, 0, 1)),
    ("SWAP", SWAP4, [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)]),
]
head = f"{'target':<22}{'CX':>4}{'gates':>7}{'phase-free error':>19}"
print(f"\nCX基底での明示的構成")
print(head)
print("-" * len(head))
for label, target, circ in CONSTRUCTIONS:
    err = phase_free_error(target, unitary_of(circ, 2))
    assert err < 1e-10, label
    print(f"{label:<22}{gate_counts(circ)['2q']:>4}{len(circ):>7}{err:>19.2e}")


# ---- 局所不変量と、それが含意するCX数 -----------------------------------
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
                 dtype=complex) / np.sqrt(2)


def magic_m(U):
    """U を SU(4) に規格化したうえでの m = (M^dagger U M)^T (M^dagger U M) です。

    m のスペクトルは両側の1量子ビットゲートで不変なので、U の局所同値類の
    ラベルになります。
    """
    U4 = U / np.linalg.det(U) ** 0.25
    Ut = MAGIC.conj().T @ U4 @ MAGIC
    return Ut.T @ Ut


def cx_count(U, tol=1e-9):
    """m の不変量から求めた、U に必要なCXゲートの最小数です。"""
    m = magic_m(U)
    tr, tr2 = np.trace(m), np.trace(m @ m)
    if abs(abs(tr) - 4) < tol and abs(tr2 - 4) < tol:
        return 0                                   # U は局所ゲートの積
    if abs(tr) < tol and abs(tr2 + 4) < tol:
        return 1                                   # CXの類
    return 2 if abs(tr.imag) < tol else 3          # tr m が実なら2個で足りる


print(f"\nマジック基底のユニタリ性検査: "
      f"{np.max(np.abs(MAGIC.conj().T @ MAGIC - np.eye(4))):.1e}")
sqrt_swap = np.array([[1, 0, 0, 0], [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                      [0, (1 - 1j) / 2, (1 + 1j) / 2, 0], [0, 0, 0, 1]],
                     dtype=complex)
GATES = [("identity", np.eye(4, dtype=complex)), ("CX", CNOT4),
         ("controlled-H", controlled(H)), ("controlled-T", controlled(T)),
         ("iSWAP", ISWAP4), ("exp(-i 0.3 Z Z)", exp_pauli(0.6, ZZ)),
         ("SWAP", SWAP4), ("sqrt(SWAP)", sqrt_swap)]
head = f"{'gate':<22}{'Re tr m':>10}{'Im tr m':>10}{'tr m^2':>18}{'CX':>5}"
print(f"\n標準的な2量子ビットゲートの局所不変量")
print(head)
print("-" * len(head))
for label, U in GATES:
    m = magic_m(U)
    tr, tr2 = np.trace(m), np.trace(m @ m)
    print(f"{label:<22}{tr.real:>10.4f}{tr.imag:>10.4f}"
          f"{f'{tr2.real:+.4f}{tr2.imag:+.4f}j':>18}{cx_count(U):>5}")

# ---- 同じ主張を実験として ------------------------------------------------
def random_2q_circuit(k, rng):
    """CXゲート k 個の間にHaarランダムな1量子ビットゲートを挟んだ回路です。"""
    circ = []
    for layer in range(k + 1):
        for q in (0, 1):
            circ += zyz_circuit(haar_1q(rng), q)
        if layer < k:
            circ.append(("cx", 0, 1) if layer % 2 == 0 else ("cx", 1, 0))
    return circ


print("\n各CX数につきランダム回路2000本を、不変量で分類した結果")
head = (f"{'CX in the circuit':>18}{'max |Im tr m|':>16}{'median':>12}"
        f"{'counts returned':>20}")
print(head)
print("-" * len(head))
for k in range(4):
    rng = np.random.default_rng(500 + k)
    ims, verdicts = [], {}
    for _ in range(2000):
        U = unitary_of(random_2q_circuit(k, rng), 2)
        ims.append(abs(np.trace(magic_m(U)).imag))
        v = cx_count(U)
        verdicts[v] = verdicts.get(v, 0) + 1
    print(f"{k:>18}{max(ims):>16.2e}{np.median(ims):>12.2e}"
          f"{str(verdicts):>20}")
```

```text
CXゲート2個の制御U。U(2) のHaarランダムな U を400個
  位相を除いた誤差の最大値: 7.11e-16
  使われたCX数            : {2: 400}

CX基底での明示的構成
target                  CX  gates   phase-free error
----------------------------------------------------
CZ                       1      3           2.22e-16
controlled-T             2      6           2.28e-16
exp(-i 0.3 Z Z)          2      3           0.00e+00
SWAP                     3      3           0.00e+00

マジック基底のユニタリ性検査: 2.2e-16

標準的な2量子ビットゲートの局所不変量
gate                     Re tr m   Im tr m            tr m^2   CX
-----------------------------------------------------------------
identity                  4.0000    0.0000   +4.0000+0.0000j    0
CX                        0.0000    0.0000   -4.0000-0.0000j    1
controlled-H              0.0000    0.0000   -4.0000-0.0000j    1
controlled-T              3.6955    0.0000   +2.8284+0.0000j    2
iSWAP                     0.0000    0.0000   +4.0000+0.0000j    2
exp(-i 0.3 Z Z)           3.3013    0.0000   +1.4494+0.0000j    2
SWAP                      0.0000   -4.0000   -4.0000-0.0000j    3
sqrt(SWAP)                1.4142   -1.4142   -0.0000-4.0000j    3

各CX数につきランダム回路2000本を、不変量で分類した結果
 CX in the circuit   max |Im tr m|      median     counts returned
------------------------------------------------------------------
                 0        6.50e-16    6.52e-17           {0: 2000}
                 1        1.50e-15    2.22e-16           {1: 2000}
                 2        1.55e-15    2.22e-16           {2: 2000}
                 3        3.76e+00    4.15e-01           {3: 2000}
```

**着目点。** 制御$U$ の構成は400個のHaarランダムな対象に対し $7\times10^{-16}$ の精度で厳密であり、毎回CXゲートを2個使います。続く構成の表は階層全体を1か所にまとめており、各行は対象の密行列と照合済みです。

不変量の表には名前を挙げるべき驚きが3つあります。制御$H$ は2個ではなく**1個**のCXで足ります。$H^2 = I$ なので標的側の基底変換1つでCXがそれに変わり、不変量もCXの類に置きます。制御$T$ は制御ゲートの中では一般の場合であり、2個必要です。そしてiSWAPはCXとまったく同じく $\operatorname{tr} m = 0$ ですが、$\operatorname{tr} m^2$ が $-4$ ではなく $+4$ であり、これが2CXの類と1CXの類を分けます。

最後の表は下限を実験にしたものです。各CX数につきランダム回路2000本、局所ゲートはすべてHaarランダムで、分類は各CX数について2000回中2000回正しくなっています。CX 2個では局所ゲートが何であれ $\lvert\operatorname{Im}\operatorname{tr} m\rvert$ は $1.6\times10^{-15}$ を超えず、3個では中央値 $0.42$ です。したがって2CXの回路がSWAPになることはありえず、SWAPの3CX構成は都合がよいだけでなく最適です。

* * *

## 2.4 Clifford$+T$ と $T$ 数

### 誤り耐性の境界でゲート集合が変わる

ここまでは第1章のゲート集合を所与として扱っており、近未来のハードウェアではそれが正しいです。誤り訂正符号の内側では事情が反転します。論理ゲートは誤りを広げずに符号化された量子ビット上で実装できなければならず、標準的な符号でそれが可能なのは離散集合だけ — $H$、$S$、CX が生成する**Clifford群**です。Cliffordゲートが安価なのは、横断的な操作や格子手術で実行でき、場合によっては符号の枠を付け替えるだけで無料に済むからです。

Clifford群は普遍的でもなく、その失敗ぶりは見事です。Gottesman-Knillの定理により、計算基底の入力と測定をもつ $n$ 量子ビットのCliffordゲート回路は $n$ の多項式時間で古典シミュレートできます。Clifford回路が何をしようと、ノートPCが同じことをできます。普遍性にはもう1つゲートが必要で、標準的な選択は

$$ T = \begin{pmatrix} 1 & 0 \cr 0 & e^{i\pi/4} \end{pmatrix} $$

です。これはCliffordを安価にする符号では横断的に実行できません。代わりに**マジック状態蒸留**で作ります。特定の状態のノイジーな複製を多数用意し、より少なくより清浄な複製を出力するClifford回路で消費し、これを繰り返します。出力は蒸留した状態1個あたり $T$ ゲート1個であり、蒸留の時空コストはCliffordゲートのそれを何桁も上回ります。第5章がその算術を明示します。ここでは帰結だけで十分です。

> 誤り耐性計算の内側では、回路のコストはその $T$ 数です。Cliffordゲートはほぼ無料であり、総ゲート数はほぼ無関係です。

### 1つの回転が高価な理由

ゲート集合 $\lbrace H, S, T\rbrace$ は離散で、$U(2)$ は離散ではありません。したがって任意の回転は表現すること自体ができず、近似することしかできません。そしてその近似のコストが回転のコストです。**Solovay-Kitaevの定理**は、[量子コンピューティング入門](<../quantum-computing-introduction/index.html>)で扱ったとおり、任意の稠密な生成集合から $\mathrm{polylog}(1/\varepsilon)$ 個のゲートで近似が常に可能であることを述べ、構成的なアルゴリズムを与えます。ここでは再実装しません。

ここで導けるのは下限であり、必要なのは数え上げだけです。$T$ 数が高々 $t$ の1量子ビットClifford$+T$ 演算子の個数を $N(t)$ とします。3次元多様体 $U(2)/\text{位相}$ を精度 $\varepsilon$ で覆うには $\varepsilon^{-3}$ 程度の演算子が必要なので、どんなゲート集合でも $T$ 数 $t$ で精度 $\varepsilon$ に届くのは

$$ N(t) \gtrsim \varepsilon^{-3} $$

のときに限られます。Code Example 7 は $N(t)$ を計測して $N(t+1)/N(t) \to 1.98$、すなわち $N(t) \sim 2^t$ を得ます。代入すると

$$ 2^{t} \gtrsim \varepsilon^{-3} \qquad \Longrightarrow \qquad t \gtrsim 3\log_2(1/\varepsilon) $$

精度1ビットあたり $T$ ゲート3個であり、どんな合成アルゴリズムもこれより良くはできません。短い語に対する全数探索の実測コストは1ビットあたり $3.30$ なので、$t \le 12$ の時点でこの下限はほぼ飽和しています。

### Code Example 7: Clifford$+T$ と、任意回転1個の値段

まずToffoliです。これが計算の標準単位だからです。3量子ビット、CXゲート6個、そしてT型回転7個であり、7はToffoliに対する既知の最適 $T$ 数です。IRに $T^\dagger$ はありません。$R_z(-\pi/4)$ は $T^\dagger$ の $e^{i\pi/8}$ 倍であり、これらは制御されていない1量子ビットゲートなので残るスカラーはグローバル位相で、検査器はそれを無視します。

```python
"""第2章 Example 7: Clifford+Tと、任意回転1個の値段。
Code Example 6 の続き（同一セッション）。"""
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from qcheck import *


def tdg(q):
    """Tダガー。IRにこのゲートはありません。Rz(-pi/4) は Tダガー の exp(i pi/8) 倍で、
    制御されていないスカラー倍はグローバル位相なので検査器は無視します。"""
    return ("rz", -np.pi / 4, q)


def toffoli(a, b, c):
    """制御 a, b と標的 c のCCX。CXゲート6個とT型回転7個です。"""
    return [("h", c),
            ("cx", b, c), tdg(c), ("cx", a, c), ("t", c),
            ("cx", b, c), tdg(c), ("cx", a, c), ("t", b), ("t", c),
            ("h", c),
            ("cx", a, b), tdg(b), ("cx", a, b), ("t", a)]


def t_count(circ):
    """マジック状態を消費するゲート。T と、pi/4 の奇数倍の Rz です。"""
    return sum(1 for g in circ if g[0] == "t"
               or (g[0] == "rz" and abs(abs(wrap(g[1])) - np.pi / 4) < 1e-12))


def permutation_matrix(n, action):
    """ビット列上の古典可逆写像の 2^n x 2^n 行列です。"""
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        bits = [(j >> (n - 1 - q)) & 1 for q in range(n)]
        out = action(list(bits))
        U[sum(b << (n - 1 - q) for q, b in enumerate(out)), j] = 1.0
    return U


ccx = toffoli(0, 1, 2)
ccz = ccx[1:10] + ccx[11:]                    # 量子ビット2上のHゲートを両方落とす
fredkin = [("cx", 2, 1)] + ccx + [("cx", 2, 1)]
CCX8 = permutation_matrix(3, lambda b: [b[0], b[1], b[2] ^ (b[0] & b[1])])
CSWAP8 = permutation_matrix(
    3, lambda b: [b[0], b[2], b[1]] if b[0] else b)
CCZ8 = np.diag([1.0] * 7 + [-1.0]).astype(complex)

print("Clifford+Tによる3量子ビットゲート3種。いずれも行列と照合済み")
head = (f"{'gate':<10}{'gates':>7}{'CX':>5}{'T':>4}{'depth':>7}"
        f"{'phase-free error':>19}")
print(head)
print("-" * len(head))
for label, circ, target in [("Toffoli", ccx, CCX8), ("CCZ", ccz, CCZ8),
                            ("Fredkin", fredkin, CSWAP8)]:
    err = phase_free_error(target, unitary_of(circ, 3))
    assert err < 1e-10, label
    print(f"{label:<10}{len(circ):>7}{gate_counts(circ)['2q']:>5}"
          f"{t_count(circ):>4}{circuit_depth(circ, 3):>7}{err:>19.2e}")

opt = peephole(ccx)
print("\nExample 3 のピープホール最適化器をToffoliに適用:")
print(f"  前: {len(ccx)} ゲート、T {t_count(ccx)} 個、"
      f"CX {gate_counts(ccx)['2q']} 個")
print(f"  後: {len(opt)} ゲート、T {t_count(opt)} 個、"
      f"CX {gate_counts(opt)['2q']} 個、誤差 "
      f"{assert_equivalent(ccx, opt, 3, 'toffoli'):.1e}")
# ---- 演算子は何個あり、どこまで精度が出るのか ----------------------------
def phase_key(U, digits=6):
    """グローバル位相に依らない、2x2 ユニタリのハッシュ可能なラベルです。"""
    flat = U.ravel()
    i = int(np.argmax(np.abs(flat)))
    return tuple(np.round(flat * np.conj(flat[i]) / abs(flat[i]), digits))


def clifford_t_ball(t_max):
    """T数が t_max 以下のすべての1量子ビットClifford+T演算子を、H と S（無料）と
    T（コスト1）に対する0-1幅優先探索で列挙します。"""
    start = np.eye(2, dtype=complex)
    cost, reps = {phase_key(start): 0}, {phase_key(start): start}
    queue = deque([(0, start)])
    while queue:
        t, U = queue.popleft()
        for G, dt in ((H, 0), (S, 0), (T, 1)):
            if t + dt > t_max:
                continue
            V, k = G @ U, phase_key(G @ U)
            if k not in cost or t + dt < cost[k]:
                cost[k], reps[k] = t + dt, V
                (queue.appendleft if dt == 0 else queue.append)((t + dt, V))
    return cost, reps


T_MAX = 12
cost, reps = clifford_t_ball(T_MAX)
keys = list(reps.keys())
ops = np.array([reps[k] for k in keys])
tcost = np.array([cost[k] for k in keys])
sizes = [int(np.sum(tcost <= t)) for t in range(T_MAX + 1)]

rng = np.random.default_rng(3)
targets = [haar_1q(rng) for _ in range(200)]
best = np.zeros((len(targets), T_MAX + 1))
for i, U in enumerate(targets):
    tr = np.einsum("nij,ij->n", ops.conj(), U)      # すべての V に対する tr(V^dagger U)
    ph = np.where(np.abs(tr) < 1e-12, 1.0, tr / np.abs(tr))
    err = np.abs(U[None] - ph[:, None, None] * ops).reshape(len(ops), -1).max(1)
    for t in range(T_MAX + 1):
        best[i, t] = err[tcost <= t].min()
median = np.median(best, axis=0)

print(f"\n恒等元まわりのClifford+T球と、それが U(2) をどれだけ覆うか")
head = (f"{'T count':>8}{'operators':>11}{'growth':>8}"
        f"{'median error, 200 targets':>27}{'log2(1/err)':>13}")
print(head)
print("-" * len(head))
for t in range(T_MAX + 1):
    ratio = f"{sizes[t] / sizes[t - 1]:.3f}" if t else "-"
    print(f"{t:>8}{sizes[t]:>11}{ratio:>8}{median[t]:>27.4f}"
          f"{np.log2(1 / median[t]):>13.2f}")

bits = np.log2(1.0 / median)
slope = np.polyfit(bits[6:], np.arange(T_MAX + 1)[6:], 1)[0]
print(f"\n精度1ビットあたりのTゲート数（t >= 6 でフィット）: {slope:.2f}")
print(f"Tゲート1個あたりの演算子数（実測）             : "
      f"{sizes[T_MAX] / sizes[T_MAX - 1]:.2f}")
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.semilogy(range(T_MAX + 1), median, "o-", color="tab:blue",
            label="exhaustive search, median of 200 targets")
ax.semilogy(range(T_MAX + 1), median[0] * 2.0 ** (-np.arange(T_MAX + 1) / 3.0),
            "--", color="k", lw=1, label=r"$\varepsilon \propto 2^{-t/3}$")
ax.set_xlabel("T count"); ax.set_ylabel("best achievable error")
ax.set_title("The price of one arbitrary single-qubit rotation")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
```

```text
Clifford+Tによる3量子ビットゲート3種。いずれも行列と照合済み
gate        gates   CX   T  depth   phase-free error
----------------------------------------------------
Toffoli        15    6   7     12           2.48e-16
CCZ            13    6   7     11           2.43e-16
Fredkin        17    8   7     13           2.48e-16

Example 3 のピープホール最適化器をToffoliに適用:
  前: 15 ゲート、T 7 個、CX 6 個
  後: 15 ゲート、T 7 個、CX 6 個、誤差 0.0e+00

恒等元まわりのClifford+T球と、それが U(2) をどれだけ覆うか
 T count  operators  growth  median error, 200 targets  log2(1/err)
-------------------------------------------------------------------
       0         34       -                     0.3188         1.65
       1        154   4.529                     0.2013         2.31
       2        420   2.727                     0.1474         2.76
       3        977   2.326                     0.1096         3.19
       4       2056   2.104                     0.0874         3.52
       5       4081   1.985                     0.0733         3.77
       6       7987   1.957                     0.0563         4.15
       7      15632   1.957                     0.0441         4.50
       8      30382   1.944                     0.0380         4.72
       9      59476   1.958                     0.0322         4.96
      10     116887   1.965                     0.0262         5.25
      11     231311   1.979                     0.0207         5.59
      12     458360   1.982                     0.0152         6.04

精度1ビットあたりのTゲート数（t >= 6 でフィット）: 3.30
Tゲート1個あたりの演算子数（実測）             : 1.98
```

**着目点。** 3量子ビットゲート3種はいずれも厳密な行列と照合が取れ、いずれも $T$ 数が7です。CCZはToffoliから外側のHadamard 2個を削ったもの、FredkinはToffoliをCXで共役したものなので、どちらもマジック状態を1個も余分に使いません。Code Example 3 のピープホール最適化器はToffoliの中に何も見つけません。15ゲートが入って15ゲートが出ます。これは正しい答えであり、率直に述べる価値があります。$T$ 数を減らすことはゲート数を減らすこととは別の問題であり、局所的な規則では解けず、いまも研究の対象です。

主要な表は数え上げの議論を計測したものです。$T$ 数が高々 $t$ の演算子の球は $T$ ゲート1個あたり $1.98$ 倍に増え、200個のHaarランダムな対象に対する最良近似誤差の中央値は $T$ ゲート1個あたり $2^{1/3.30}$ 倍で減ります。この2つは上の被覆下限で結ばれた同じ主張です。フィットした $3.30$ を外挿すると、任意の回転1個のコストは $10^{-3}$ で $T$ ゲート約33個、$10^{-6}$ で66個、$10^{-10}$ で110個 — 6桁でToffoli約10個分、10桁で約15個分です。ここから3つの帰結が従い、いずれも第5章に再登場します。誤り耐性のコンパイラは可能な限り任意回転を吐かないようにし、CliffordとToffoliを好みます。姉妹コースの近似QFTが最小の制御回転を捨てるのは最適化ではなく必然です。それらの回転こそ高価なものであり、しかもその大半はアルゴリズムが必要とする精度より下にあります。そして「ゲート数」で述べた資源見積りは解釈できません。意味のある通貨は $T$ 数だけです。

* * *

## 演習

#### 演習1: ときどきしか規則でない規則

最適化器の `canonical` は $R_z(\pi)$ を $Z$ に書き換え、Code Example 1 は両者が位相を除いて等しいことを確認しました。

  1. $R_z(\pi)$ と $Z$ を明示的に計算し、両者を結ぶ位相を与えてください。
  2. $R_z(\pi)$ を $Z$ に置き換えると位相だけでなく物理的な結果が変わる回路を挙げてください。
  3. 位相を落としてよいかを決める規則を述べてください。
  4. Code Example 6 の `controlled_circuit` は `canonical` ではなく `rot` を通して回転を放出します。`canonical` を使ったとき、放出される3つのブロックのうちどれが最初に壊れますか。またその理由は何ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(R_z(\pi) = \mathrm{diag}(e^{-i\pi/2}, e^{i\pi/2}) = \mathrm{diag}(-i, i) = -i\,\mathrm{diag}(1,-1) = -i\,Z\) です。位相は \(e^{-i\pi/2}\) です。</p>

<p><strong>2.</strong> その断片が制御の下に置かれるあらゆる回路です。制御\(Z\) は \(\mathrm{diag}(1,1,1,-1)\)、制御\(R_z(\pi)\) は \(\mathrm{diag}(1,1,-i,i)\) です。これらは \(|1\rangle\) の枝で因子 \(-i\) だけ異なり \(|0\rangle\) の枝では一致しないので、単一の位相を除いて等しくはなりません。素朴な誤差は \(\sqrt2\)、位相を除いた誤差は0ではなく 0.765 です。第1章 Code Example 4 がまさにこの比較を出力しています。</p>

<p><strong>3.</strong> グローバル位相を捨ててよいのは、その断片が他の量子ビットで条件づけられることが決してなく、自身の制御版を作るために使われることも決してない場合に限ります。言い換えれば、位相は回路の最上位では落とせますが、後続のパスが制御をかけるブロックの内側では落とせません。</p>

<p><strong>4.</strong> 単一の \(R_z((c-a)/2)\) である \(C\) ブロック、そして \(B\) ブロックです。この構成は \(ABC = I\) が<em>厳密に</em>成り立つことに依存しており、回転を \(Z\)、\(S\)、\(T\) に改名するたびにそのブロックが位相分だけ変わります。すると \(ABC = e^{i\varphi}I\) となり、\(|0\rangle\) の枝が \(e^{i\varphi}\) を得て \(|1\rangle\) の枝は得ないので、結果は制御\(U\) ではありません。Code Example 6 を <code>rot</code> の代わりに <code>canonical</code> で走らせると、誤差の最大値は \(7\times10^{-16}\) ではなく \(7\times10^{-1}\) 程度になります。</p>

</details>

#### 演習2: 2つのCXが交換するとき

Code Example 2 の判定関数は、$\mathrm{CX}_{a,b}$ と $\mathrm{CX}_{c,d}$ が交換するのは $a = c$ または $b = d$ のとき、かつそのときに限ると述べます。

  1. $\mathrm{CX}_{a,b} = |0\rangle\langle 0|_a \otimes I + |1\rangle\langle 1|_a \otimes X_b$ を用いて、制御を共有する場合の「ならば」の向きを証明してください。
  2. 標的を共有する場合を証明してください。
  3. $\mathrm{CX}_{0,1}$ と $\mathrm{CX}_{1,2}$ が交換しないことを、$|110\rangle$ に両方の順序で作用させて示してください。
  4. Code Example 2 のライブラリには、3量子ビット上で台が重なり制御と標的が異なるCXの順序対が含まれます。そのうち何個が交換し、判定関数はそのすべてを捕まえていますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 制御 \(a\) を共有すると、両方のゲートは \(a\) の基底でブロック対角です。\(|0\rangle_a\) ブロックはどちらも恒等、\(|1\rangle_a\) ブロックは \(b \ne d\) の \(X_b\) と \(X_d\) で、別の量子ビットに作用するので交換します。したがって積はブロックごとに一致します。</p>

<p><strong>2.</strong> 標的 \(b\) を共有すると、2つのゲートは \(X_b^{n_a}\) と \(X_b^{n_c}\) です（\(n_a, n_c\) は2つの制御の数演算子）。それらの制御は互いに異なり \(b\) とも異なるので、\(n_a\) と \(n_c\) は周囲のすべてと交換し、2つのゲートはともに同じ \(X_b\) の冪です。</p>

<p><strong>3.</strong> \(\mathrm{CX}_{1,2}\mathrm{CX}_{0,1}|110\rangle\) では、まず制御0が立っているので量子ビット1が反転して \(|100\rangle\)、次は制御1が下りているので何もせず \(|100\rangle\) です。逆順では、\(\mathrm{CX}_{1,2}\) が制御1が立っているので量子ビット2を反転して \(|111\rangle\)、次に \(\mathrm{CX}_{0,1}\) が量子ビット1を反転して \(|101\rangle\) です。両者は異なります。</p>

<p><strong>4.</strong> 6個です。制御を共有する対 \((0{,}1)(0{,}2)\)、\((1{,}0)(1{,}2)\)、\((2{,}0)(2{,}1)\) とその逆順で、標的を共有するものも同数の6個です。いずれも判定関数の1行 <code>g[1] == h[1] or g[2] == h[2]</code> が捕まえます。全数検査はCXの対の見落としを報告しておらず、38件の見落としはすべて変装した恒等ゲート2つに関わるものです。</p>

</details>

#### 演習3: 手で行うZYZ

  1. 2.2節の式を $H$ に適用し、Code Example 5 の出力行と照合してください。
  2. 同じ式を $T$ に適用し、どちらの退化分岐が取られるかを説明してください。
  3. ある人の実装は3つの角を返し位相を返さず、それでも10000個のランダム入力で正しく検証できたと報告します。彼らのテストは何で、何を見落としていますか。
  4. 一般の $U$ の合成が、ときどき2ゲートになるのではなく常に3ゲートになるのはなぜですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\det H = -1\) なので \(\delta = \frac{1}{2}\arg(-1) = \pi/2\) であり \(V = e^{-i\pi/2}H = -iH\) です。このとき \(|V_{00}| = |V_{10}| = 1/\sqrt2\) なので \(b = 2\arctan 1 = \pi/2\)。\(V_{11} = i/\sqrt2\) の位相は \(\pi/2\)、\(V_{10} = -i/\sqrt2\) の位相は \(-\pi/2\) なので \(a = 0\)、\(c = \pi\) です。これが出力行 \(\delta/\pi = +0.5\)、\((a,b,c)/\pi = (0, 0.5, 1)\) であり、放出される回路は \(Z\) のあとに \(R_y(\pi/2)\) の2ゲートです。\(a = 0\) が何も寄与しないからです。</p>

<p><strong>2.</strong> \(\det T = e^{i\pi/4}\) なので \(\delta = \pi/8\)、\(V = \mathrm{diag}(e^{-i\pi/8}, e^{i\pi/8}) = R_z(\pi/4)\) です。ここで \(V_{10} = 0\) なので \(b = 0\) の分岐であり、\(a + c\) だけが決まります。コードは \(c = 0\) と置き、\(a = 2\arg V_{11} = \pi/4\) です。1ゲートです。</p>

<p><strong>3.</strong> 彼らのテストは位相を除いた比較であり、それはまさに欠けた数を見ることができないテストです。あらゆる入力で通ってしまいます。見落としているのは、そのルーチンが制御構成の内側で使われるすべての場合 — 2.3節の制御\(U\)、第3章のルーティング後の制御ゲート、そして第5章の誤り緩和のようにコヒーレントに2つの回路を比較する場合です。バグは出力に制御がかかった最初のときに現れ、それ以前には現れません。</p>

<p><strong>4.</strong> 一般の \(SU(2)\) の元は自明でないEuler角を3つ持ち、放出される各ゲートが1つのパラメータを担います。2ゲートでは2パラメータの部分集合しか覆えず、それは測度0です。2000個のHaar標本に対する出力の分布が \(\lbrace 3: 2000\rbrace\) なのはこのためであり、短い出力は測度0の退化集合上でだけ起こります。</p>

</details>

#### 演習4: 不変量からCX数を読む

制御$S$ ゲート $\mathrm{diag}(1,1,1,i)$ と $\sqrt{\mathrm{CZ}} = \mathrm{diag}(1,1,1,i)$ — 同じ行列の2つの名前 — を考えます。

  1. Code Example 6 の `magic_m` と `cx_count` を使うと、どのCX数が返りますか。実行前に予測してください。
  2. 制御位相 $\mathrm{diag}(1,1,1,e^{i\varphi})$ の正準座標は $(\varphi/4, 0, 0)$ です。どの $\varphi$ で2個ではなく1個のCXで足りますか。
  3. iSWAPと $\mathrm{CX}$ はどちらも $\operatorname{tr} m = 0$ です。両者を分ける不変量は何で、その2つの値は何ですか。
  4. ある人が $G_1 = (\operatorname{tr} m)^2/(16\det U)$ として「$\operatorname{Im} G_1 = 0$ ならCXは高々2個」という判定条件を提案します。SWAPで検査し、失敗の理由を説明してください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 2個です。これは \(\varphi = \pi/2\) の制御位相なので正準座標は \((\pi/8, 0, 0)\) であり、原点でもCXの角でもありません。\(\operatorname{tr} m\) が実なので分類は2を返します。数値では \(\operatorname{tr} m = 3.4142\)、\(\operatorname{tr} m^2 = 2.0000\) です。</p>

<p><strong>2.</strong> \(\varphi = \pi\) だけです。これが座標を \((\pi/4,0,0)\)、すなわちCZの角に置きます。他のどんな制御位相も、\(\varphi\) がどれほど小さくてもCXゲート2個を要します。だからこそ小さな制御回転を捨てるコンパイラは1個の端数ではなくCXを2個節約するのであり、これは2.4節の近似QFTの議論のルーティング層での対応物です。</p>

<p><strong>3.</strong> \(\operatorname{tr} m^2\) です。CXでは \(-4\)、iSWAPでは \(+4\) です。正準座標では両者は \((\pi/4,0,0)\) と \((\pi/4,\pi/4,0)\) で、どちらも \(t_z = 0\) なのでともに2個のCXで到達できますが、1個で到達できるのは前者だけです。</p>

<p><strong>4.</strong> SWAPは \(\operatorname{tr} m = -4i\) なので \((\operatorname{tr} m)^2 = -16\) は実であり \(\operatorname{Im} G_1 = 0\) となって、判定条件は誤って「CX 2個で足りる」と報告します。トレースを2乗すると「実」と「純虚」の区別が消えますが、2CXの条件はまさにその区別に立っています。判定条件は \((\operatorname{tr} m)^2\) ではなく \(\operatorname{tr} m\) で述べなければなりません。</p>

</details>

#### 演習5: $T$ 数の予算

あるアルゴリズムがToffoliゲート $10^4$ 個と、精度 $10^{-8}$ の任意1量子ビット回転 $3\times10^3$ 個を必要とします。

  1. フィットした精度1ビットあたり $3.30$ を用いて、回転の $T$ 数、Toffoliの $T$ 数、そして合計を見積もってください。
  2. どちらが支配的で、その比はどれだけですか。
  3. コンパイラが改良され、回転の半分を厳密なCliffordゲートで置き換えられるようになりました。新しい合計はいくらで、それは何割の節約ですか。
  4. 第2の提案は、回転の個数を2倍にする代償でToffoliの個数を半分にします。これは改善ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\log_2(10^{8}) = 26.6\) ビットなので回転1個あたり \(3.30 \times 26.6 \approx 88\) 個の \(T\) ゲートであり、\(3\times10^3 \times 88 \approx 2.6\times10^{5}\) です。Toffoliは \(7 \times 10^4 = 7\times10^{4}\) です。合計は \(\approx 3.3\times10^{5}\) です。</p>

<p><strong>2.</strong> 回転が支配的で、約3.8倍です。個数が3分の1しかないのにそうなります。この精度では任意回転1個がToffoli約12個分の価値です。</p>

<p><strong>3.</strong> \(1.5\times10^3\) 個の回転に88を掛けて \(1.3\times10^{5}\)、これに \(7\times10^{4}\) を加えて \(2.0\times10^{5}\) です。片方のゲート種の半分を除くだけで総 \(T\) 数の \(40\%\) を節約したことになります。この非対称性こそ、誤り耐性コンパイルが主として「回転をどう避けるか」の研究である理由です。</p>

<p><strong>4.</strong> いいえ。ToffoliでTゲート \(3.5\times10^{4}\) 個を節約し、回転で \(2.6\times10^{5}\) 個を追加するので、合計は約1.7倍に増えます。ゲート数という通貨で「ゲートが半分」に見える取引が、唯一意味のある通貨では大きな損失です。誤り耐性機械向けの2つの回路を比較する前に、必ず \(T\) 数に換算してください。</p>

</details>

* * *

## まとめ

### 要点

**1\. 書き換え規則は同じユニタリをもつ断片の対であり、「同じ」は検査しなければならない**

  * 3つの系統がすべての仕事をします。融合、相殺、交換 — うち3番目だけは単独では何も短くせず、それが他の2つを適用可能にします。
  * 回路上の隣接とは、リストの次の項ではなく同じ量子ビットに触れる次のゲートです。この区別こそ第1章の最適化器がルーティング後に何も見つけられなかった理由であり、本章のあらゆる書き換えは位相を除いた行列比較で守られ、誤差の最大値は $10^{-15}$ です。

**2\. 交換判定関数は完全ではなく健全であるべきである**

  * 不健全性は黙って回路の意味を変え、不完全性は最適化を1つ失うだけです。だから判定関数は保守的に書き、その不完全性を計測します。
  * 35ゲートのライブラリから作った1225個の順序対に対し、不健全な答えは0、見落とした交換は38件で、すべて $R_y(0)$ か $R_z(2\pi)$ — 構文的な規則には見えない形の恒等ゲート — に関わります。対策は規則を増やすことではなく正規化です。

**3\. ピープホール最適化の収量は入力の性質である**

  * 200本の一様ランダム回路ではゲートの $26.3\%$、深さの $18.8\%$、そして2量子ビットゲートはわずか $7.1\%$ です。回路の後に自身の逆回路を置いた場合は毎回すべてが消えます。
  * 停止性は無料 — 成功する規則は必ず回路を短くします — ですが正しさは無料ではなく、だから検査が先に来ます。

**4\. 規則が完全でない場所で合成は完全である**

  * ZYZ: 任意の $U \in U(2)$ は $e^{i\delta}R_z(a)R_y(b)R_z(c)$ であり、2000個のHaar標本に対して $10^{-15}$ の精度、明示的に扱うべき退化分岐が2つ、そして位相 $\delta$ は断片に制御がかかった瞬間にグローバルでなくなるので返します。
  * 1量子ビットゲートの連なりを再合成すると、本章の規則一覧のどれも触れられなかった $H X H$ が消えます。ピープホールパスと合わせると、書いたままの60.00ゲートに対して36.40ゲートに達します。3通りの最適化パイプラインにわたる検証済みの書き換え600件と、無操作の対照パイプライン200回での結果です。

**5\. 2量子ビットは3つの数と、固いCX数**

  * KAK: $U = (A_1\otimes A_2)\exp\big[ i(t_xXX + t_yYY + t_zZZ) \big]\,(B_1\otimes B_2)$ なので、ゲートのコストは四面体上の1点です。CZはCX 1個、$\exp(-i\frac{\theta}{2}ZZ)$ と一般の制御$U$ は2個、SWAPは3個です。
  * その個数は推測ではなくマジック基底での $\operatorname{tr} m$ と $\operatorname{tr} m^2$ から読み取り、実験としても検証しました。各CX数につきランダム回路2000本を2000回正しく分類し、CX 2個では $\lvert\operatorname{Im}\operatorname{tr} m\rvert < 1.6\times10^{-15}$、3個では中央値 $0.42$ です。

**6\. 誤り耐性計算では通貨は $T$ 数だけである**

  * Cliffordゲートは安価で普遍的ではありません。Gottesman-Knillはノートパソコンでシミュレートできると述べます。$T$ は普遍的で、マジック状態蒸留から来ます。
  * $T$ 数 $\le t$ のClifford$+T$ 演算子の個数は $T$ ゲート1個あたり $1.98$ 倍に増え、$U(2)/$位相 を $\varepsilon$ で覆うには $\varepsilon^{-3}$ 個が必要なので $t \gtrsim 3\log_2(1/\varepsilon)$ です。全数探索での実測は1ビットあたり $3.30$ なので、任意回転1個は $10^{-6}$ で $T$ ゲート66個、Toffoli 1個まるごとが7個です。

**実務上の含意**

  * パスを書く前に等価性検査を書き、答えが分かっているテストを含めてください。回路の後に自身の逆回路を置いたものは、何も残らずに最適化されなければなりません。
  * 最適化器の収量は総ゲート数・深さ・2量子ビットゲート数を別々に報告し、入力を何が生成したかを述べてください。3つの数は別々に動き、誤差を予言するのは最後のものだけです。
  * 後続のパスが制御をかけうる断片の内側でグローバル位相を落とさないでください。そして誤り耐性回路のゲート数を引用しないでください。まず $T$ 数に換算してください。

### この先へ

本章のすべての回路は、どの量子ビットも他のどれとでも相互作用できるものとしてコンパイルしましたが、それが真であるハードウェアはちょうど1系統だけです。第3章はその仮定を外します。第1章の層の地図と[量子ハードウェア入門](<../quantum-hardware-introduction/index.html>)が動機づける3つのトポロジー — 全結合、正方格子、heavy-hex — の結合グラフを作り、接続性が生む2つの問題に向き合います。どの物理量子ビットがどの論理量子ビットを保持するかを選ぶ問題と、その選択が誤りだった場合にSWAPネットワークを挿入する問題です。そこでの計測値は本章のものを圧倒します。本章の最適化器は回路のゲートの4分の1を除去しましたが、同じ回路を疎なグラフへルーティングすると2倍から5倍に増え、さらに等価性検査も作り直さなければなりません。ルーティング後の回路は、書き下したユニタリと意図的に*異なる*ものだからです。

[← 第1章: アルゴリズムからパルスまでのスタック](<chapter-1.html>) [第3章: トランスパイル — 接続性への写像 →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章のゲート数・$T$ 数・削減率は、コード中に明示した特定の回路と乱数種に対する測定値であり、いかなるコンパイラやハードウェアのベンチマークでもありません。2.4節の外挿による回転コストは、明記したフィットから導いた桁レベルの教育用概算です。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
