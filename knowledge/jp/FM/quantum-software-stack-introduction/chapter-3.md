---
title: "第3章: トランスパイル — 接続性への写像"
chapter_title: "第3章: トランスパイル — 接続性への写像"
subtitle: 物理の帰結としての結合グラフ、配置問題、SWAP挿入、そして正しいルータともっともらしいルータを見分ける方法
reading_time: 45-50分
difficulty: 上級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-software-stack-introduction/chapter-3.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子ソフトウェアスタック入門](<index.html>) > 第3章

第2章は、どの量子ビットも他のどれとでも相互作用できるものとして回路をコンパイルしました。そのように振る舞うハードウェアはちょうど1系統だけで、理由は物理的です。イオン鎖は共有された運動モードを介してすべての対を結合するので、結合グラフは完全グラフになります。超伝導チップは平面上で量子ビットを容量的に近傍と結合し、さらに隣接する遷移周波数が衝突してはならないという制約が加わります。その結果できるグラフは次数3か4で、直径はチップとともに伸びます。[量子ハードウェア入門](<../quantum-hardware-introduction/index.html>)はそれらのグラフをゲート機構から導きます。本章はそれを所与とし、コンパイラがそれに対して何をしなければならないかを問います。

やるべきことは2つあり、述べやすいのはそのうち1つだけです。**配置**は、最初にどの物理量子ビットがどの論理量子ビットを保持するかの選択であり、どのコンパイラも厳密には解かない難しい組み合わせ問題です。**ルーティング**は、回路が要求するゲートに対して配置が不適切だと判明したときにSWAPゲートを挿入することであり、ネイティブSWAPを持たないハードウェアではSWAP 1個がCXゲート3個分です。以下の計測値は第2章のものを圧倒します。あちらの最適化器は回路のゲートの4分の1を除去しましたが、12量子ビットのQFTをheavy-hexグラフへルーティングするとCX数は4.6倍になります。接続性は二次的な考慮事項ではありません。

さらに新しい正しさの問題があり、これが本章の方法論的な中心です。ルーティング後の回路は、書き下したユニタリと意図的に*異なります* — SWAPがどの物理線がどの論理量子ビットを保持するかを置換したからです。したがって第1章の等価性検査はそのままでは使えず、置換の報告を粗く扱うルータは、置換を無視するあらゆるテストを通過してハードウェア上で失敗します。3.3節は検査を作り直し、Code Example 4 は素朴な版が何を見落とすかを示します。SWAPが挿入された回路でちょうど1のオーダーの誤差です。

## 学習目標

本章を修了すると、以下のことができるようになります。

  * 結合グラフをデータ — 隣接関係と全点対最短経路表 — として表現し、全結合・1次元鎖・正方格子・heavy-hexのトポロジーを構成できる
  * 結合グラフの平均距離から回路のルーティングオーバーヘッドを予測し、その見積りが予測ではなく上界である理由を説明できる
  * 配置問題を述べ、それがNP困難である理由と、それでも厳密解の価値がきわめて小さい理由を説明できる
  * 自身の置換を追跡する最近傍SWAPルータを実装し、SABRE型のルータが行っていて自分が行っていないことを正確に述べられる
  * 置換を込めた等価性検査 $U_{\mathrm{phys}}P(\ell_0) = P(\ell_f)(U_{\mathrm{log}}\otimes I)$ を作り、小さな装置では厳密に、大きな装置では標本で検証し、置換を無視した版が失敗することを実証できる
  * GHZとQFT回路について3つのトポロジーでSWAP数を計測し、グラフの寄与と回路の相互作用構造の寄与を分離できる
  * ヒューリスティックなルータを、両方が動く小さな事例で厳密なルータと比較し、超過分をゲート順序・先読み・配置・目的関数に帰属させられる

* * *

## 3.1 接続性グラフ

### グラフはどこから来るのか

結合グラフはゲート機構についての主張であり、本章で使う4つのトポロジーは4つの異なる機構です。

| トポロジー | 物理的な機構 | 帰結 |
| --- | --- | --- |
| **全結合** | すべての量子ビットが1つの共有ボゾンモード — イオン結晶の運動モード、あるいはバス共振器 — に結合する | ルーティングは不要。ゲート速度は全対で共有され、モードが混み合うと劣化する |
| **1次元鎖** | 一列に並んだ量子ビットの最近傍交換相互作用。ゲート定義スピンなど | 平均距離が $n/3$ で伸びる。ルーティングにとって最悪の場合 |
| **正方格子** | 平面上の容量結合、次数4 | 平均距離が $\sqrt{n}$ で伸びる |
| **heavy-hex** | 六角格子の各結合上に量子ビットを1個追加し、次数が3を超えないようにしたもの | 周波数衝突とクロストークが扱える範囲に留まる。グラフは非常に疎 |

行われている取引が見えるのはheavy-hexの行です。その目的は接続性ではなく — 同じ量子ビット数の格子より接続性は*低い* — 次数の上限です。近傍が4つある周波数固定の超伝導量子ビットは周波数衝突の機会が4つ、クロストーク経路が4つあります。次数を3で抑え、しかも各結合上に量子ビットを1個挿入して計算用の量子ビットが直接隣り合わないようにすることで、平均距離が伸びる代償として歩留まりと較正の安定性を買います。これは材料と制御の判断がグラフとして現れたものであり、そのつけをコンパイラが払います。

### 平均距離がコストの予測子である

結合グラフ上で距離 $d$ にある量子ビット間の2量子ビットゲートは、両者を近づけるのに $d - 1$ 個のSWAPを要し、SWAP 1個はCXゲート3個分です。相互作用する対が装置上に一様に散らばった回路では、2量子ビットゲートあたりの余分なCXゲート数の期待値は

$$ \text{オーバーヘッド} \;\approx\; 3\left( \bar{d} - 1 \right), \qquad \bar{d} = \frac{2}{n(n-1)}\sum_{i<j} \mathrm{dist}(i,j) $$

です。これはルータが手元にないときに使う見積りで、Code Example 1 が各トポロジーについて計算します。実際には*上界*であり、その理由はいま理解しておく価値があります。この見積りは各ゲートが毎回まっさらなランダム配置から始まると仮定していますが、ルータは自分のSWAPを元に戻さないので、近くの量子ビット上で連続するゲートは独立なゲートより安く済みます。Code Example 5 がその差 — 格子で約2倍 — を計測します。

### Code Example 1: データとしての接続性グラフ

本章のすべては第1章の3つのモジュール上で動きます。本章が自己完結するようにここに再掲します。1つ目は[量子コンピューティング入門](<../quantum-computing-introduction/chapter-2.html>)の状態ベクトルシミュレータで、本章が使う関数のみ逐語です。同ファイルの `probs` と `sample` は本章では不要です。

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

2つ目は回路IRで、第1章 Code Example 2 から逐語です。

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

3つ目は検査器で、第1章 Code Example 4 から逐語です。3.3節はこれを置き換えるのではなく拡張します。

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

さてグラフです。装置は辞書で表します。隣接集合、幅優先探索で計算した全点対最短経路表、そしてソート済みの辺リストです。

```python
"""第3章 Example 1: データとしての接続性グラフ。"""
from itertools import combinations
import numpy as np
from qcheck import *


def device(name, n, edges):
    """結合グラフ。隣接集合と全点対最短経路表です。"""
    adj = {q: set() for q in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):                              # s からの幅優先探索
        dist[s][s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for p in frontier:
                for r in sorted(adj[p]):
                    if dist[s][r] < 0:
                        dist[s][r] = dist[s][p] + 1
                        nxt.append(r)
            frontier = nxt
    return {"name": name, "n": n, "adj": adj, "dist": dist,
            "edges": sorted(tuple(sorted(e)) for e in edges)}


def all_to_all(n):
    """すべての対が結合。イオン鎖など、バスを介する方式です。"""
    return device(f"all-to-all {n}", n, list(combinations(range(n), 2)))


def line(n):
    """1次元鎖。一列に並べたゲート定義スピン量子ビットです。"""
    return device(f"line {n}", n, [(q, q + 1) for q in range(n - 1)])


def grid(rows, cols):
    """正方格子。平面超伝導チップの自然な配置です。"""
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return device(f"grid {rows}x{cols}", rows * cols, edges)


def heavy_hex_7():
    """最小のheavy-hex断片。次数3の量子ビットを2個もつH字形です。"""
    return device("heavy-hex 7", 7,
                  [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)])


def heavy_hex_16():
    """heavy hexagon 1つ、すなわち12周期の閉路に、フラグ量子ビット4個が付いた形です。

    heavy-hex格子は、六角格子の各結合上に量子ビットを1個追加したもので、どの
    量子ビットも隣接数が3を超えません。この上限こそが要点です。周波数固定の超伝導
    チップで周波数衝突とクロストークを扱える範囲に保つためのものであり、その代償が
    非常に疎なグラフです。
    """
    return device("heavy-hex 16", 16,
                  [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8),
                   (6, 7), (7, 10), (8, 9), (8, 11), (10, 12), (11, 14),
                   (12, 13), (12, 15), (13, 14)])


DEVICES = [all_to_all(16), grid(4, 4), heavy_hex_16(), line(16),
           all_to_all(7), grid(2, 3), heavy_hex_7()]

head = (f"{'device':<16}{'qubits':>7}{'edges':>7}{'deg (mean)':>12}"
        f"{'deg (max)':>11}{'diameter':>10}{'mean dist':>11}")
print(head)
print("-" * len(head))
for d in DEVICES:
    n = d["n"]
    degs = [len(d["adj"][q]) for q in range(n)]
    pairs = [d["dist"][a][b] for a, b in combinations(range(n), 2)]
    print(f"{d['name']:<16}{n:>7}{len(d['edges']):>7}{np.mean(degs):>12.2f}"
          f"{max(degs):>11}{max(pairs):>10}{np.mean(pairs):>11.2f}")

print(f"{'device':<16}{'mean dist':>11}{'extra CX per 2q gate':>23}")
print("-" * 50)
for d in DEVICES[:4]:
    md = np.mean([d["dist"][a][b] for a, b in combinations(range(d["n"]), 2)])
    print(f"{d['name']:<16}{md:>11.2f}{3 * (md - 1):>23.2f}")
```

```text
device           qubits  edges  deg (mean)  deg (max)  diameter  mean dist
--------------------------------------------------------------------------
all-to-all 16        16    120       15.00         15         1       1.00
grid 4x4             16     24        3.00          4         6       2.67
heavy-hex 16         16     16        2.00          3         8       3.68
line 16              16     15        1.88          2        15       5.67
all-to-all 7          7     21        6.00          6         1       1.00
grid 2x3              6      7        2.33          3         3       1.67
heavy-hex 7           7      6        1.71          3         4       2.29
device            mean dist   extra CX per 2q gate
--------------------------------------------------
all-to-all 16          1.00                   0.00
grid 4x4               2.67                   5.00
heavy-hex 16           3.68                   8.05
line 16                5.67                  14.00
```

**着目点。** 16量子ビット、4つのトポロジー、辺の数は120から15へ落ち、平均距離は1.00から5.67へ上がります。heavy-hex断片は格子と同じ量子ビット数で辺は3分の2、平均距離は38%大きい — これが次数3の上限の値段を定量化したものです。2つ目の表は平均距離を上の見積りに換算します。全結合は何も払わず、$4\times4$ 格子は2量子ビットゲートあたりCX 5個、heavy-hexは8個、16量子ビットの1次元鎖は14個です。

回路のゲートの4分の1を除去するために骨を折った第2章と比べてください。接続性はゲートを5倍にしえます。2つの効果は同じ大きさではなく、配置を無視してピープホール規則に労力を注ぐコンパイラは配分を誤っています。

### 需要側 — 回路が要求するもの

結合グラフは問題の半分にすぎません。もう半分は回路の**相互作用グラフ**、すなわち回路がどの論理量子ビットの対を、何回結合する必要があるかです。2つの回路がその幅を挟みます。

**GHZ鎖** — Hadamard 1個と一列の $n-1$ 個のCXゲート — は相互作用グラフがパスです。パスは連結なあらゆる装置の内側に存在するので、GHZ鎖を完全にネイティブにする配置が*何かしら*あり、それを見つけることが配置問題の最も易しい形です。

**QFT** はすべての対をちょうど1回必要とするので、相互作用グラフは完全グラフです。どんな配置も助けになりません。全結合以外のハードウェアでは、辺でないすべての対をルーティングしなければなりません。これが標準的な負荷試験になる理由であり、しかも実在の回路です。[量子アルゴリズム中級](<../quantum-algorithms-intermediate/index.html>)の第2章がこれを作り、その上に位相推定を構築します。

### Code Example 2: ルーティングの対象となる回路

どちらの回路もIRで書き、どちらも使用前に検証します。制御位相ゲートは厳密な行列と、QFT回路はビット反転置換を手で適用した上で密なDFT行列と照合します。

```python
"""第3章 Example 2: ルーティング対象の2つの回路と、その相互作用グラフ。
Code Example 1 の続き（同一セッション）。"""
from itertools import combinations
import numpy as np
from qcheck import *


def ghz_chain(n):
    """CNOT鎖によるGHZ。相互作用グラフはパスです。"""
    return [("h", 0)] + [("cx", q, q + 1) for q in range(n - 1)]


def cphase(theta, a, b):
    """IRでの制御位相ゲート。CXゲート2個とRz回転3個です。"""
    return [("rz", theta / 2, a), ("rz", theta / 2, b),
            ("cx", a, b), ("rz", -theta / 2, b), ("cx", a, b)]


def qft(n):
    """最後の反転を省いた教科書的QFT。すべての対が1回ずつ相互作用します。"""
    circ = []
    for q in range(n):
        circ.append(("h", q))
        for r in range(q + 1, n):
            circ += cphase(np.pi / 2 ** (r - q), q, r)
    return circ


def interaction_graph(circ, n):
    """回路が結合を必要とする量子ビット対の集合と、その回数です。"""
    weight = {}
    for g in circ:
        qs = gate_qubits(g)
        if len(qs) == 2:
            key = tuple(sorted(qs))
            weight[key] = weight.get(key, 0) + 1
    return weight


CP = np.diag([1.0, 1.0, 1.0, np.exp(1j * 0.7)])
print("制御位相ゲートを行列と照合:")
print(f"  theta = 0.7 での位相を除いた誤差: "
      f"{phase_free_error(CP, unitary_of(cphase(0.7, 0, 1), 2)):.2e}")


def dft_matrix(n):
    """QFT回路の検査用、2^n 振幅上の密なDFT行列です。"""
    dim = 2 ** n
    j = np.arange(dim)
    return np.exp(2j * np.pi * np.outer(j, j) / dim) / np.sqrt(dim)


def reversal(n):
    """教科書的QFTが呼び出し側に残すビット反転置換です。"""
    dim = 2 ** n
    P = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        bits = format(x, f"0{n}b")
        P[int(bits[::-1], 2), x] = 1.0
    return P


print("\nQFT回路と密なDFT行列の照合（反転は手で適用）:")
for n in (2, 3, 4, 5):
    err = phase_free_error(dft_matrix(n), reversal(n) @ unitary_of(qft(n), n))
    print(f"  n = {n}: 位相を除いた誤差 {err:.2e}")

print("\nコンパイル問題としての2つの回路")
head = (f"{'circuit':<14}{'qubits':>7}{'gates':>7}{'CX':>5}{'depth':>7}"
        f"{'pairs used':>12}{'pairs possible':>16}")
print(head)
print("-" * len(head))
for label, circ, n in [("GHZ chain", ghz_chain(6), 6), ("QFT", qft(6), 6),
                       ("GHZ chain", ghz_chain(10), 10), ("QFT", qft(10), 10)]:
    w = interaction_graph(circ, n)
    print(f"{label:<14}{n:>7}{len(circ):>7}{gate_counts(circ)['2q']:>5}"
          f"{circuit_depth(circ, n):>7}{len(w):>12}{n * (n - 1) // 2:>16}")

print("\n回路にルーティングが必要かどうかは、2つのグラフの関係の問題です")
head = f"{'circuit':<20}" + "".join(f"{d['name']:>16}" for d in
                                    (all_to_all(6), grid(2, 3), heavy_hex_7()))
print(head)
print("-" * len(head))
for label, circ, n in [("GHZ chain, n = 6", ghz_chain(6), 6),
                       ("QFT, n = 6", qft(6), 6),
                       ("GHZ chain, n = 5", ghz_chain(5), 5),
                       ("QFT, n = 5", qft(5), 5)]:
    pairs = interaction_graph(circ, n)
    row = ""
    for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
        if d["n"] < n:
            row += f"{'too small':>16}"
            continue
        native = sum(1 for (a, b) in pairs if b in d["adj"][a])
        row += f"{f'{native}/{len(pairs)} native':>16}"
    print(f"{label:<20}{row}")
```

```text
制御位相ゲートを行列と照合:
  theta = 0.7 での位相を除いた誤差: 1.12e-16

QFT回路と密なDFT行列の照合（反転は手で適用）:
  n = 2: 位相を除いた誤差 1.92e-16
  n = 3: 位相を除いた誤差 1.31e-15
  n = 4: 位相を除いた誤差 2.09e-15
  n = 5: 位相を除いた誤差 3.19e-15

コンパイル問題としての2つの回路
circuit        qubits  gates   CX  depth  pairs used  pairs possible
--------------------------------------------------------------------
GHZ chain           6      6    5      6           5              15
QFT                 6     81   30     38          15              15
GHZ chain          10     10    9     10           9              45
QFT                10    235   90     70          45              45

回路にルーティングが必要かどうかは、2つのグラフの関係の問題です
circuit                 all-to-all 6        grid 2x3     heavy-hex 7
--------------------------------------------------------------------
GHZ chain, n = 6          5/5 native      4/5 native      3/5 native
QFT, n = 6              15/15 native     7/15 native     5/15 native
GHZ chain, n = 5          4/4 native      3/4 native      2/4 native
QFT, n = 5              10/10 native     5/10 native     3/10 native
```

**着目点。** QFT回路は $n = 5$ で密なDFTを $3\times10^{-15}$ の精度で再現します。これがベンチマークとして使う許可です。続くコンパイル問題の表が2つの極端を示します。$n = 10$ でGHZ鎖は使える45対のうち9対、QFTは45対すべてを使い、QFTの90個のCXゲートは45個の制御位相の各2個から来ます。

最後の表が次節への導入です。*自明な*配置 — 論理 $q$ を物理 $q$ に — で読むと、GHZ鎖は $2\times3$ 格子上で5対のうち4対しかネイティブではありません。物理量子ビット2と3が結合していないからです。別の配置なら5対中5対になります。QFTは15対中7対で、どんな配置も改善しません。完全グラフ以外のどの6量子ビット装置も辺を15本持たないからです。

* * *

## 3.2 配置問題

### 定式化

**配置**とは、回路の論理量子ビットから装置の物理量子ビットへの単射です。結合グラフ $G$ と回路が与えられたとき、配置問題はその後のルーティングコストを最小化する単射を選ぶことです。

これが難しい理由は2つの特別な場合が示します。ある配置の下で回路の相互作用グラフが $G$ の部分グラフになるなら、その配置のコストは0であり、それを見つけることはまさに**部分グラフ同型**、すなわちNP完全問題です。そのような配置が存在しない場合 — 通常はこちらです — 目的関数は後段のルータが生むSWAP数になるので、問題の難しさはグラフだけでなくルータにも依存します。いずれにせよ期待できる厳密アルゴリズムはなく、探索空間は $n$ 論理量子ビットを $N$ 物理量子ビットへ写す
$$ \frac{N!}{(N-n)!} $$
通りの単射 — 127量子ビット装置に16論理量子ビットなら $2\times10^{33}$ 通りです。

### 厳密解の価値が小さい理由

ここでのNP困難性に対する誠実な応答は諦めではなく計測であり、それを行うのが Code Example 6 です。6量子ビットや7量子ビットの装置なら全数探索が可能なので最適値が計算でき、安価なヒューリスティクスをそれと比較できます。計測が示すのは、最適値から1〜2ゲート内に収まるヒューリスティクスは神託と同じだけの価値がある、ということです。残る差が、まともな回路の間のばらつきより小さいからです。

知っておく価値のあるヒューリスティクスはSABREが導入したもので、6行です。自明な配置から回路を順方向にルーティングし、ルータが終了時に持っていた写像を保持します。次にそこから*逆順*のゲート列をルーティングし、その写像を保持し、これを繰り返します。各パスは開始写像を、回路が実際に含むゲートに合ったものへ近づけ、配置に対する探索は一切行いません。直感はこうです。ルータが最後に持つ写像は構成上、回路の終わりに適応しています。だからそこから回路を逆向きに走らせれば、回路の始まりに適応した写像が得られます。

配置の質を測るにはルータが必要なので、本節のコードはルータが存在してからの Code Example 6 です。

* * *

## 3.3 ルーティングと、その検査法

### SWAP挿入

配置が与えられると、ルータは回路をたどり、2量子ビットゲートが結合グラフの辺でない対に作用するたびに、辺になるまで一方または両方の量子ビットを動かします。この移動がSWAPであり、ネイティブSWAPを持たないハードウェアではSWAPはCXゲート3個です。

$$ \mathrm{SWAP}_{a,b} = \mathrm{CX}_{a,b}\,\mathrm{CX}_{b,a}\,\mathrm{CX}_{a,b} $$

第2章はこれを検証し、しかも最適であることを示しました。SWAPの正準座標は $(\pi/4,\pi/4,\pi/4)$ であり、2CXの回路はそこに到達できません。距離 $d$ のゲートを満たす最も安い方法は最短経路に沿った $d-1$ 個のSWAPで、その個数は片方の端点が全部歩いても両方が近づいても同じです。違うのは深さだけです。

Code Example 3 のルータは、動く中で最も単純なことを行い、その名も正直に付けています。次の2量子ビットゲートについて、その一方の量子ビットを最短経路に沿ってもう一方へ歩かせ、1ホップにつきSWAP 1個、ゲートは書かれた順、先読みなしです。この選択の2つの性質が重要です。

**自分の置換を保持します。** SWAPの後、ある論理量子ビットを保持する物理線は変わっており、ルータはSWAPを元に戻すのではなく新しい写像を記録します。元に戻せば写像は自明なままで等価性検査は易しくなります — 第1章 Code Example 7 がまさにそれを行い、そう明言しています — が、代償はSWAP数の倍増です。置換を保持するのが本物のトランスパイラの振る舞いであり、だからこそ検査を作り直す必要があります。

**これはベースラインであり、ルータではありません。** 3.4節が何を諦めているかを計測します。

### SABREは代わりに何をするか

実運用ツールのルータはSABREの子孫であり、その2つの着想は原理のレベルで述べる価値があります。どちらも本章では実装しません。

**ゲート順序ではなくフロント層で作業する。** 回路は列ではなく半順序です。台が交わらないゲートはどちらの順でも実行できます。SABREは先行ゲートがすべて済んだゲートの集合 — **フロント層** — を保持し、その中の実行可能な任意の要素を実行できます。書かれた順序では先にあるゲートがすでに辺の上にあることは多く、それを先に実行すればSWAPの必要が丸ごと消えます。

**先読みコストで候補SWAPを採点する。** 目の前のゲートのための最短経路に決め打ちするのではなく、SABREはフロント層に隣接する各辺上のSWAPをすべて考え、フロント層を残す距離の総和*に加えて*、後続ゲートの拡張集合からの減衰した寄与でそれを採点します。最近使った量子ビットに減衰係数をかけることで、同じ量子ビットを繰り返し動かすことを抑制します。結果は「現在のゲートの距離」よりずっと形の良いコスト関数上の貪欲探索です。

どちらの着想も実装の複雑さを要し、どちらも漸近的な振る舞いを変えません。それらが買うものを Code Example 7 が計測します。

### 等価性検査を作り直す

$\ell$ を仮想線から物理線への写像とし、$\ell[v]$ が仮想量子ビット $v$ を保持する物理線であるとします。$\ell_0$ をその初期値、$\ell_f$ をルーティング後の値とし、$P(\ell)$ を $\ell$ に従って線を付け替えるユニタリとします。すると「ルーティングは回路の意味を保つ」という主張はちょうど

$$ U_{\mathrm{phys}}\, P(\ell_0) \;=\; P(\ell_f)\, \left( U_{\mathrm{log}} \otimes I \right) $$

がグローバル位相を除いて成り立つことです。ここで $U_{\mathrm{log}}$ は $n$ 個の論理量子ビットに、$I$ は $N - n$ 個の遊休量子ビットに作用します。左から右へ読んでください。論理量子ビットを初期の物理線に置き、物理回路を走らせると、論理回路を走らせて結果を*最終の*物理線に置いたものと同じ状態になります。ルータが正直に報告すべきものはすべて $\ell_f$ に入っています。

このテストには2つの実装が必要です。$2^N \times 2^N$ の行列を作れるほど小さな装置 — ここでは $N \le 7$ — では恒等式を厳密に検査できます。16量子ビットの装置では行列は複素数 $4\times10^9$ 個、状態ベクトルは65536個なので、代わりにランダムな入力状態でテストを走らせます。これは証明ではなく確率的な検査であり、何が買えるかは明確にしておく価値があります。置換の取り違えは直ちに捕まります。置換の誤りはほとんどすべての入力に対して $O(1)$ の誤差だからです。

### Code Example 3: 最近傍ルータ

```python
"""第3章 Example 3: 最近傍ルータ。
Code Example 2 の続き（同一セッション）。"""
import numpy as np
from qcheck import *


def remap(g, loc):
    """ゲートタプルを仮想量子ビットから物理線へ書き換えます。"""
    if g[0] in ROT_1Q:
        return (g[0], g[1], loc[g[2]])
    if g[0] in TWO_Q:
        return (g[0], loc[g[1]], loc[g[2]])
    return (g[0], loc[g[1]])


def swap_gates(p, q):
    """結合した2本の物理線上のSWAP。ネイティブSWAPはないのでCXゲート3個です。"""
    return [("cx", p, q), ("cx", q, p), ("cx", p, q)]


def next_hop(dev, p, target):
    """target に1歩近い p の隣接点。添字が最小のものを選びます。"""
    for r in sorted(dev["adj"][p]):
        if dev["dist"][r][target] == dev["dist"][p][target] - 1:
            return r
    raise ValueError("disconnected coupling graph")


def route(circ, dev, layout=None):
    """すべての2量子ビットゲートが結合した対に作用するまでSWAPを挿入します。

    方針は動く中で最も単純なもので、名前も正直に付けています。次の2量子ビット
    ゲートについて、その2つの量子ビットの一方を最短経路に沿ってもう一方へ歩かせ、
    1ホップにつきSWAP 1個を挿入します。ゲートは書かれた順に保ち、先読みは一切
    しません。SABREが代わりに何をするかは3.3節で述べ、その差はExample 7で計測
    します。

    loc[v] は仮想量子ビット v をいま保持している物理線です。物理回路、初期と最終
    の写像、そしてSWAP数を返します。
    """
    loc = list(range(dev["n"])) if layout is None else list(layout)
    loc0, out, swaps = list(loc), [], 0
    for g in circ:
        qs = gate_qubits(g)
        if len(qs) == 1:
            out.append(remap(g, loc))
            continue
        a, b = qs
        while dev["dist"][loc[a]][loc[b]] > 1:
            p, r = loc[a], next_hop(dev, loc[a], loc[b])
            out += swap_gates(p, r)
            swaps += 1
            u, v = loc.index(p), loc.index(r)        # 2つの仮想量子ビットが動く
            loc[u], loc[v] = r, p
        out.append(remap(g, loc))
    return out, loc0, loc, swaps


dev = grid(2, 3)
circ = ghz_chain(5)
phys, loc0, locf, swaps = route(circ, dev)
print("5量子ビットのGHZ鎖を、自明なレイアウトで 2x3 格子上にルーティング")
print(f"  結合辺              : {dev['edges']}")
print(f"  論理回路            : {circ}")
print(f"  物理回路            : {phys}")
print(f"  挿入したSWAP数      : {swaps}   (CXゲート "
      f"{gate_counts(circ)['2q']} -> {gate_counts(phys)['2q']})")
print(f"  初期写像 v -> p     : {loc0}")
print(f"  最終写像 v -> p     : {locf}")
print(f"  深さ                : {circuit_depth(circ, 5)} -> "
      f"{circuit_depth(phys, dev['n'])}")

print("\nルーティング後の回路のすべての2量子ビットゲートが結合対の上にあること:")
bad = [g for g in phys if len(gate_qubits(g)) == 2
       and gate_qubits(g)[1] not in dev["adj"][gate_qubits(g)[0]]]
print(f"  結合していない対の上のゲート数: {len(bad)}")

print("\n同じ回路を3つのトポロジーで")
head = (f"{'device':<16}{'SWAPs':>7}{'CX in':>7}{'CX out':>8}{'blow-up':>9}"
        f"{'depth in':>10}{'depth out':>11}")
print(head)
print("-" * len(head))
for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
    p, l0, lf, s = route(circ, d)
    cin, cout = gate_counts(circ)["2q"], gate_counts(p)["2q"]
    print(f"{d['name']:<16}{s:>7}{cin:>7}{cout:>8}{cout / cin:>9.2f}"
          f"{circuit_depth(circ, 5):>10}{circuit_depth(p, d['n']):>11}")

print("\nそしてすべての対を必要とするQFT")
print(head)
print("-" * len(head))
for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
    q = qft(5)
    p, l0, lf, s = route(q, d)
    cin, cout = gate_counts(q)["2q"], gate_counts(p)["2q"]
    print(f"{d['name']:<16}{s:>7}{cin:>7}{cout:>8}{cout / cin:>9.2f}"
          f"{circuit_depth(q, 5):>10}{circuit_depth(p, d['n']):>11}")
```

```text
5量子ビットのGHZ鎖を、自明なレイアウトで 2x3 格子上にルーティング
  結合辺              : [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
  論理回路            : [('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 3), ('cx', 3, 4)]
  物理回路            : [('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('cx', 2, 1), ('cx', 1, 2), ('cx', 2, 1), ('cx', 1, 0), ('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 3), ('cx', 3, 4)]
  挿入したSWAP数      : 2   (CXゲート 4 -> 10)
  初期写像 v -> p     : [0, 1, 2, 3, 4, 5]
  最終写像 v -> p     : [1, 2, 0, 3, 4, 5]
  深さ                : 5 -> 11

ルーティング後の回路のすべての2量子ビットゲートが結合対の上にあること:
  結合していない対の上のゲート数: 0

同じ回路を3つのトポロジーで
device            SWAPs  CX in  CX out  blow-up  depth in  depth out
--------------------------------------------------------------------
all-to-all 6          0      4       4     1.00         5          5
grid 2x3              2      4      10     2.50         5         11
heavy-hex 7           2      4      10     2.50         5         11

そしてすべての対を必要とするQFT
device            SWAPs  CX in  CX out  blow-up  depth in  depth out
--------------------------------------------------------------------
all-to-all 6          0     20      20     1.00        30         30
grid 2x3              9     20      47     2.35        30         67
heavy-hex 7          12     20      56     2.80        30         57
```

**着目点。** 5量子ビットのGHZ鎖は $2\times3$ 格子で2個のSWAPを要し、CX数は4から10 — 相互作用グラフがパスの回路に対して2.5倍です。原因はすべて、自明な配置が論理2と論理3を結合していない物理線に置いたことにあります。最終写像 `[1, 2, 0, 3, 4, 5]` は論理量子ビットの行き先を記録し、結合していない対の検査は出力のすべての2量子ビットゲートが装置上で合法であることを確認します。

覚えておくべきはQFTの行です。$2\times3$ 格子では20個のCXが47個になり深さは30から67へ、7量子ビットのheavy-hex断片では20個が56個になります。heavy-hexの装置は量子ビットが1個多いのにルーティングコストは*悪い* — これも次数の上限が現れたものです。

### Code Example 4: 追跡した置換を込めた検査

```python
"""第3章 Example 4: 追跡した置換を込めた等価性検査。
Code Example 3 の続き（同一セッション）。"""
import numpy as np
from qcheck import *


def place(psi, loc, n):
    """仮想線 v 上の量子ビットを物理線 loc[v] へ移します。"""
    return np.transpose(psi.reshape([2] * n), np.argsort(loc)).reshape(-1)


def permutation_unitary(loc, n):
    """その付け替えの 2^n x 2^n 行列です。"""
    dim = 2 ** n
    P = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        P[:, j] = place(e, loc, n)
    return P


def routed_error(circ, n_log, dev, phys, loc0, locf):
    """U_phys P(loc0) と P(locf) (U_log tensor I) の、位相を除いた誤差の最大値です。

    「ルーティングは回路の意味を保つ」という主張の中身はこれで尽きています。
    ルーティング後の回路は、書き下した回路とは別のユニタリです。保たれるのは、
    そのユニタリにSWAPが行った付け替えを合成したものであり、ルータが正しいのは
    その付け替えを正直に報告している場合に限ります。
    """
    n = dev["n"]
    U_log = unitary_of(circ, n_log)
    U_emb = np.kron(U_log, np.eye(2 ** (n - n_log))) if n > n_log else U_log
    left = unitary_of(phys, n) @ permutation_unitary(loc0, n)
    return phase_free_error(left, permutation_unitary(locf, n) @ U_emb)


def routed_error_sampled(circ, n_log, dev, phys, loc0, locf, trials=8, seed=0):
    """同じ検査をランダムな入力状態で行います。行列が大きすぎる装置向けです。"""
    n, rng, worst = dev["n"], np.random.default_rng(seed), 0.0
    idle = ket("0" * (n - n_log)) if n > n_log else None
    for _ in range(trials):
        v = rng.normal(size=2 ** n_log) + 1j * rng.normal(size=2 ** n_log)
        psi_l = v / np.linalg.norm(v)
        out_l = run_circuit(circ, n_log, psi0=psi_l)
        psi_v = np.kron(psi_l, idle) if idle is not None else psi_l
        out_v = np.kron(out_l, idle) if idle is not None else out_l
        got = run_circuit(phys, n, psi0=place(psi_v, loc0, n))
        want = place(out_v, locf, n)
        ph = np.vdot(want, got)
        ph = ph / abs(ph) if abs(ph) > 1e-12 else 1.0
        worst = max(worst, float(np.max(np.abs(got - ph * want))))
    return worst


print("まず付け替え自身の検査。P([1, 0]) はSWAP行列でなければなりません。")
print(f"  誤差: "
      f"{np.max(np.abs(permutation_unitary([1, 0], 2) - unitary_of([('cx', 0, 1), ('cx', 1, 0), ('cx', 0, 1)], 2))):.2e}")

print("\nルーティング後の回路を、厳密検査と標本検査の両方で確認")
head = (f"{'circuit':<12}{'device':<16}{'SWAPs':>7}{'matrix check':>14}"
        f"{'sampled check':>15}{'ignoring the perm':>19}")
print(head)
print("-" * len(head))
CASES = [("GHZ n=5", ghz_chain(5), 5), ("QFT n=4", qft(4), 4),
         ("QFT n=5", qft(5), 5)]
for label, circ, n_log in CASES:
    for d in (all_to_all(6), grid(2, 3), heavy_hex_7()):
        phys, loc0, locf, swaps = route(circ, d)
        exact = routed_error(circ, n_log, d, phys, loc0, locf)
        sampled = routed_error_sampled(circ, n_log, d, phys, loc0, locf, seed=1)
        naive = routed_error(circ, n_log, d, phys, loc0, loc0)
        assert exact < 1e-10, (label, d["name"])
        print(f"{label:<12}{d['name']:<16}{swaps:>7}{exact:>14.1e}"
              f"{sampled:>15.1e}{naive:>19.1e}")

print("\n自明でない初期レイアウトも同じ恒等式で扱えます")
head = f"{'layout v -> p':<22}{'SWAPs':>6}  {'final map':<22}{'matrix check':>14}"
print(head)
print("-" * len(head))
for layout in ([0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2], [1, 0, 3, 2, 5, 4],
               [0, 1, 2, 5, 4, 3], [3, 0, 1, 2, 5, 4]):
    phys, loc0, locf, swaps = route(ghz_chain(5), grid(2, 3), layout)
    err = routed_error(ghz_chain(5), 5, grid(2, 3), phys, loc0, locf)
    print(f"{str(layout):<22}{swaps:>6}  {str(locf):<22}{err:>14.1e}")
```

```text
まず付け替え自身の検査。P([1, 0]) はSWAP行列でなければなりません。
  誤差: 0.00e+00

ルーティング後の回路を、厳密検査と標本検査の両方で確認
circuit     device            SWAPs  matrix check  sampled check  ignoring the perm
-----------------------------------------------------------------------------------
GHZ n=5     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
GHZ n=5     grid 2x3              2       0.0e+00        0.0e+00            7.1e-01
GHZ n=5     heavy-hex 7           2       0.0e+00        0.0e+00            7.1e-01
QFT n=4     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
QFT n=4     grid 2x3              5       0.0e+00        0.0e+00            5.0e-01
QFT n=4     heavy-hex 7           3       0.0e+00        0.0e+00            5.0e-01
QFT n=5     all-to-all 6          0       0.0e+00        0.0e+00            0.0e+00
QFT n=5     grid 2x3              9       0.0e+00        0.0e+00            3.5e-01
QFT n=5     heavy-hex 7          12       0.0e+00        0.0e+00            3.5e-01

自明でない初期レイアウトも同じ恒等式で扱えます
layout v -> p          SWAPs  final map               matrix check
------------------------------------------------------------------
[0, 1, 2, 3, 4, 5]         2  [1, 2, 0, 3, 4, 5]           0.0e+00
[3, 4, 5, 0, 1, 2]         3  [3, 4, 0, 1, 2, 5]           0.0e+00
[1, 0, 3, 2, 5, 4]         2  [0, 3, 1, 2, 5, 4]           0.0e+00
[0, 1, 2, 5, 4, 3]         0  [0, 1, 2, 5, 4, 3]           0.0e+00
[3, 0, 1, 2, 5, 4]         0  [3, 0, 1, 2, 5, 4]           0.0e+00
```

**着目点。** まず付け替えのユニタリを、第2章のSWAP回路と照合して検証します。$P([1,0])$ がSWAP行列でないなら、その後の一切に意味がありません。次にルーティング後のすべての回路が厳密検査と標本検査の両方を通り、誤差は $10^{-16}$ ではなくちょうど0です。これは偶然ではありません。ルーティングは振幅テンソルのどの軸に各因子が掛かるかを変えるだけで、追加されるCXゲートは振幅の厳密な置換なので、同じ演算が同じ順序で行われ丸め誤差の差が生じないのです。

最後の列がこの例の要点です。ルーティング後の回路を元の回路と比べて置換を*忘れる*と、SWAPが挿入されたときは $0.35$ から $0.71$ の誤差になり、挿入されなかったときはちょうど0になります。これが、壊れたルータを易しい事例で正しく見せる失敗の様式です。全結合の行は通り、疎な行は落ちるので、前者しか含まないテスト群は成功を報告します。

配置の表が3.2節との輪を閉じます。4つの配置、4つのSWAP数、うち2つは0です。論理鎖 $0{-}1{-}2{-}3{-}4$ を物理パス $0{-}1{-}2{-}5{-}4$ と $3{-}0{-}1{-}2{-}5$ に写す配置で、どちらも格子が備えています。ルータがそれらの配置を見つけたのではなく、与えられました。見つけるのが Code Example 6 です。

* * *

## 3.4 コストを計測する

### この数値が何に依存するか

回路のルーティングオーバーヘッドは3つのものの積であり、そのうち1つだけを報告しても情報になりません。

  * **結合グラフ**。平均距離を通して効きます。Code Example 1 が計算した係数です。
  * **回路の相互作用構造**。パスはどの装置でもほとんど無料、完全グラフは最大です。現実のものはすべてその間にあり、ゲート数よりその位置が重要です。
  * **配置とルータ**。両者合わせて2倍程度の価値があり、Code Example 6 と 7 が計測します。

だからこそ「トポロジーXのオーバーヘッド」に単一の数値を引用するベンチマークは意味を持たず、また合成ベンチマークがあのように構成される理由でもあります。量子ボリュームのようなベンチマークは回路の*構造*を固定します — ランダムな対に作用するランダムな2量子ビットゲートの正方形回路 — ので、得られる数値は特定のアルゴリズムの性質ではなく機械とコンパイラを合わせた性質になります。本コースはベンダー数値を一切引用せず、以下の表は上で定義したグラフに対する上のコードの測定値であり、それ以上のものではありません。

### Code Example 5: 3つのトポロジーでのSWAP数

```python
"""第3章 Example 5: 接続性のコストを計測します。
Code Example 4 の続き（同一セッション）。"""
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from qcheck import *

BIG = [all_to_all(16), grid(4, 4), heavy_hex_16()]

print("GHZ鎖。挿入したSWAP数と、CX数の前 -> 後")
head = f"{'n':>3}" + "".join(f"{d['name']:>22}" for d in BIG)
print(head)
print("-" * len(head))
for n in range(4, 13):
    row = ""
    for d in BIG:
        phys, l0, lf, s = route(ghz_chain(n), d)
        cin, cout = gate_counts(ghz_chain(n))["2q"], gate_counts(phys)["2q"]
        row += f"{f'{s} swap, {cin}->{cout}':>22}"
    print(f"{n:>3}{row}")

print("\nQFT。すべての対を必要とする回路について同じ表")
print(head)
print("-" * len(head))
qft_swaps = {d["name"]: [] for d in BIG}
ns = list(range(4, 13))
for n in ns:
    row = ""
    for d in BIG:
        phys, l0, lf, s = route(qft(n), d)
        cin, cout = gate_counts(qft(n))["2q"], gate_counts(phys)["2q"]
        qft_swaps[d["name"]].append(s)
        row += f"{f'{s} swap, {cin}->{cout}':>22}"
    print(f"{n:>3}{row}")

print("\n膨張率と、平均距離による見積りの予測値")
head = (f"{'device':<16}{'QFT n=12 CX in':>16}{'CX out':>9}{'measured':>10}"
        f"{'predicted':>11}")
print(head)
print("-" * len(head))
for d in BIG:
    circ = qft(12)
    phys, l0, lf, s = route(circ, d)
    cin, cout = gate_counts(circ)["2q"], gate_counts(phys)["2q"]
    md = np.mean([d["dist"][a][b] for a, b in combinations(range(d["n"]), 2)])
    print(f"{d['name']:<16}{cin:>16}{cout:>9}{cout / cin:>10.2f}"
          f"{1 + 3 * (md - 1):>11.2f}")
print("\n同じ行を、16物理量子ビット上での標本検査で検証")
head = f"{'circuit':<12}{'device':<16}{'SWAPs':>7}{'gates out':>11}{'sampled check':>15}"
print(head)
print("-" * len(head))
for label, circ, n_log in [("GHZ n=6", ghz_chain(6), 6),
                           ("GHZ n=12", ghz_chain(12), 12),
                           ("QFT n=6", qft(6), 6), ("QFT n=10", qft(10), 10)]:
    for d in BIG[1:]:
        phys, loc0, locf, s = route(circ, d)
        err = routed_error_sampled(circ, n_log, d, phys, loc0, locf,
                                   trials=3, seed=5)
        assert err < 1e-9, (label, d["name"])
        print(f"{label:<12}{d['name']:<16}{s:>7}{len(phys):>11}{err:>15.1e}")
fig, ax = plt.subplots(figsize=(6.2, 4))
for d, style in zip(BIG, ("o-", "s-", "^-")):
    ax.plot(ns, qft_swaps[d["name"]], style, label=d["name"])
ax.set_xlabel("logical qubits in the QFT")
ax.set_ylabel("SWAPs inserted")
ax.set_title("Routing cost of the QFT on 16-qubit devices")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
```

```text
GHZ鎖。挿入したSWAP数と、CX数の前 -> 後
  n         all-to-all 16              grid 4x4          heavy-hex 16
---------------------------------------------------------------------
  4          0 swap, 3->3          0 swap, 3->3          0 swap, 3->3
  5          0 swap, 4->4         3 swap, 4->13         2 swap, 4->10
  6          0 swap, 5->5         3 swap, 5->14         5 swap, 5->20
  7          0 swap, 6->6         3 swap, 6->15        10 swap, 6->36
  8          0 swap, 7->7         3 swap, 7->16        11 swap, 7->40
  9          0 swap, 8->8         6 swap, 8->26        15 swap, 8->53
 10          0 swap, 9->9         6 swap, 9->27        15 swap, 9->54
 11        0 swap, 10->10        6 swap, 10->28       20 swap, 10->70
 12        0 swap, 11->11        6 swap, 11->29       24 swap, 11->83

QFT。すべての対を必要とする回路について同じ表
  n         all-to-all 16              grid 4x4          heavy-hex 16
---------------------------------------------------------------------
  4        0 swap, 12->12        6 swap, 12->30        6 swap, 12->30
  5        0 swap, 20->20       12 swap, 20->56        9 swap, 20->47
  6        0 swap, 30->30       17 swap, 30->81       17 swap, 30->81
  7        0 swap, 42->42      25 swap, 42->117      45 swap, 42->177
  8        0 swap, 56->56      24 swap, 56->128      63 swap, 56->245
  9        0 swap, 72->72      50 swap, 72->222      68 swap, 72->276
 10        0 swap, 90->90      59 swap, 90->267      83 swap, 90->339
 11      0 swap, 110->110     73 swap, 110->329    124 swap, 110->482
 12      0 swap, 132->132     74 swap, 132->354    158 swap, 132->606

膨張率と、平均距離による見積りの予測値
device            QFT n=12 CX in   CX out  measured  predicted
--------------------------------------------------------------
all-to-all 16                132      132      1.00       1.00
grid 4x4                     132      354      2.68       6.00
heavy-hex 16                 132      606      4.59       9.05

同じ行を、16物理量子ビット上での標本検査で検証
circuit     device            SWAPs  gates out  sampled check
-------------------------------------------------------------
GHZ n=6     grid 4x4              3         15        0.0e+00
GHZ n=6     heavy-hex 16          5         21        0.0e+00
GHZ n=12    grid 4x4              6         30        0.0e+00
GHZ n=12    heavy-hex 16         24         84        0.0e+00
QFT n=6     grid 4x4             17        132        0.0e+00
QFT n=6     heavy-hex 16         17        132        0.0e+00
QFT n=10    grid 4x4             59        412        0.0e+00
QFT n=10    heavy-hex 16         83        484        0.0e+00
```

**着目点。** GHZの表は易しい場合ですが、それでも無料ではありません。$4\times4$ 格子では鎖に $n = 5$ で3個、$n = 12$ で6個のSWAPがかかります。すべて自明な配置が論理量子ビットを行の境界をまたいで歩かせるためです。heavy-hexでは $n = 12$ で24個 — 相互作用グラフがパスの回路に対して、長いパスを含む連結な装置上で、です。これはルーティングの失敗ではなく配置の失敗であり、Code Example 6 がそのすべてを回復します。

QFTの表が本章の正直な見出しです。$n = 12$ でCX数は格子で132から354へ、heavy-hexで606へ — 2.7倍と4.6倍です。平均距離による見積りは6.0と9.05を予測したので、約2倍の過大評価です。理由は3.1節で述べたとおりです。QFTは近くの量子ビットを繰り返し相互作用させ、ルータは動かした場所に量子ビットを置いたままにするので、連続するゲートは独立なゲートより安く済みます。見積りは桁としては正しく、ルータが手元にないときに使うべきものです。ルータを走らせることの代用ではありません。

標本のブロックが検証です。4つの回路、2つの16量子ビット装置、ルーティング出力は最大484ゲートで、置換を込めた確率的検査は全行を通ります。$4^{16}$ の行列は複素数 $4\times10^9$ 個ですが、ランダムな $2^{16}$ 状態ベクトル3本は無料に近く、しかも重要な誤りを捕まえます。

### Code Example 6: 配置の価値

```python
"""第3章 Example 6: レイアウト問題と、その安価なヒューリスティクス。
Code Example 5 の続き（同一セッション）。"""
from itertools import permutations
from math import factorial
import numpy as np
from qcheck import *


def layout_search(circ, dev):
    """すべてのレイアウトを、その後ルータが必要とするSWAP数で採点します。小さな装置専用です。"""
    scores = [route(circ, dev, p)[3] for p in permutations(range(dev["n"]))]
    return min(scores), max(scores), float(np.mean(scores)), scores


def reverse_traversal(circ, dev, rounds=2):
    """SABREの初期レイアウトの工夫を6行で。

    自明なレイアウトから回路を順方向にルーティングし、終了時の写像を保持します。
    そこから逆順のゲート列をルーティングしてその写像を保持し、これを繰り返します。
    各パスは開始写像を、回路が実際に含むゲートに合った写像へ近づけます。レイアウト
    に対する探索は一切行いません。
    """
    loc = list(range(dev["n"]))
    for r in range(2 * rounds):
        gates = circ if r % 2 == 0 else list(reversed(circ))
        loc = route(gates, dev, loc)[2]
    return loc


CASES = [("GHZ n=5", ghz_chain(5), 5), ("QFT n=4", qft(4), 4),
         ("QFT n=5", qft(5), 5)]
print("6量子ビットと7量子ビットの装置2種でのレイアウト全数探索")
head = (f"{'circuit':<10}{'device':<14}{'layouts':>9}{'best':>6}{'worst':>7}"
        f"{'mean':>7}{'trivial':>9}{'reverse trav.':>15}")
print(head)
print("-" * len(head))
for label, circ, n_log in CASES:
    for d in (grid(2, 3), heavy_hex_7()):
        best, worst, mean, _ = layout_search(circ, d)
        triv = route(circ, d)[3]
        phys, loc0, locf, rev = route(circ, d, reverse_traversal(circ, d))
        err = routed_error(circ, n_log, d, phys, loc0, locf)
        assert err < 1e-10, label
        print(f"{label:<10}{d['name']:<14}{factorial(d['n']):>9}{best:>6}"
              f"{worst:>7}{mean:>7.2f}{triv:>9}{rev:>15}")

print("\n探索が続けられない理由。N 物理量子ビット上に n 論理量子ビットを置くレイアウト数")
head = f"{'N':>4}" + "".join(f"{f'n = {n}':>16}" for n in (4, 8, 12, 16))
print(head)
print("-" * len(head))
for N in (7, 16, 27, 65, 127):
    row = ""
    for n in (4, 8, 12, 16):
        row += (f"{factorial(N) // factorial(N - n):>16.3g}" if n <= N
                else f"{'-':>16}")
    print(f"{N:>4}{row}")
print("\n全数探索の答えが存在しない16量子ビット装置でのヒューリスティクス")
head = (f"{'circuit':<10}{'device':<14}{'trivial':>9}{'reverse trav.':>15}"
        f"{'best of 200 random':>20}{'check':>9}")
print(head)
print("-" * len(head))
for label, circ, n_log in [("QFT n=6", qft(6), 6), ("QFT n=8", qft(8), 8),
                           ("QFT n=10", qft(10), 10),
                           ("GHZ n=12", ghz_chain(12), 12)]:
    for d in (grid(4, 4), heavy_hex_16()):
        triv = route(circ, d)[3]
        lay = reverse_traversal(circ, d)
        rev = route(circ, d, lay)[3]
        rng = np.random.default_rng(17)
        rand = min(route(circ, d, list(rng.permutation(d["n"])))[3]
                   for _ in range(200))
        phys, loc0, locf, _ = route(circ, d, lay)
        err = routed_error_sampled(circ, n_log, d, phys, loc0, locf,
                                   trials=2, seed=9)
        assert err < 1e-9, label
        print(f"{label:<10}{d['name']:<14}{triv:>9}{rev:>15}{rand:>20}"
              f"{err:>9.0e}")
```

```text
6量子ビットと7量子ビットの装置2種でのレイアウト全数探索
circuit   device          layouts  best  worst   mean  trivial  reverse trav.
-----------------------------------------------------------------------------
GHZ n=5   grid 2x3            720     0      7   3.04        2              0
GHZ n=5   heavy-hex 7        5040     0     12   6.03        2              0
QFT n=4   grid 2x3            720     2      9   4.97        5              4
QFT n=4   heavy-hex 7        5040     2     15   8.34        3              2
QFT n=5   grid 2x3            720     4     15   8.41        9              8
QFT n=5   heavy-hex 7        5040     5     24  13.98       12             10

探索が続けられない理由。N 物理量子ビット上に n 論理量子ビットを置くレイアウト数
   N           n = 4           n = 8          n = 12          n = 16
--------------------------------------------------------------------
   7             840               -               -               -
  16        4.37e+04        5.19e+08        8.72e+11        2.09e+13
  27        4.21e+05        8.95e+10        8.33e+15        2.73e+20
  65        1.62e+07        2.04e+14        1.93e+21        1.36e+28
 127        2.48e+08         5.4e+16        1.03e+25        1.71e+33

全数探索の答えが存在しない16量子ビット装置でのヒューリスティクス
circuit   device          trivial  reverse trav.  best of 200 random    check
-----------------------------------------------------------------------------
QFT n=6   grid 4x4             17             18                  11    0e+00
QFT n=6   heavy-hex 16         17             15                  20    0e+00
QFT n=8   grid 4x4             24             32                  27    0e+00
QFT n=8   heavy-hex 16         63             55                  35    0e+00
QFT n=10  grid 4x4             59             61                  41    0e+00
QFT n=10  heavy-hex 16         83             99                  77    0e+00
GHZ n=12  grid 4x4              6              0                   9    0e+00
GHZ n=12  heavy-hex 16         24             18                  17    0e+00
```

**着目点。** 小さな装置では配置の地形が全部見えます。GHZの行では最良の配置がSWAPをすべて除去し、最悪の配置は7個または12個を挿入します。全配置の平均は3.04と6.03なので、自明な配置の2は平均より良く、それでも良くはありません。$n = 5$ のQFTをheavy-hexに置く場合、幅は5から24です。逆走ヒューリスティクスは6行すべてで自明な配置に勝ち、うち3行で真の最適に到達します。コストはルーティング4パスで、探索はなしです。

階乗の表が探索がそこで止まる理由です。127物理量子ビットに16論理量子ビットなら配置は $2\times10^{33}$ 通りあり、しかもNP困難なので厳密なものは来ません。

最後の表に誠実さがあります。16量子ビット装置ではヒューリスティクスが自明な配置より*悪い*ことがあります — 格子上の $n = 8$ のQFTで24に対して32 — 本章の版がSABREの逆走の工夫だけを残して先読みコスト関数を落としているので、長い回路では誤ったものを最適化するのです。そして本物の探索である200個のランダム配置は、ルーティング4パスの代わりに200パスというコストで、ほとんどの行でヒューリスティクスに勝ちます。この取引 — 探索時間対SWAP数 — がトランスパイラの設計空間のすべてであり、実運用ツールが単一のアルゴリズムではなく最適化レベルを露出している理由です。

### Code Example 7: 単純なルータが失うもの

ヒューリスティクスのコストを知る唯一の方法は、最適値が計算できるところで最適値を計算することです。（装置写像、すでに実行した2量子ビットゲート数）に対する幅優先探索がまさにそれを行います。SWAPのコストはすべて1なので素の幅優先探索が最小値を返し、開始写像として可能なすべてを種にすれば同時に配置問題も厳密に解けます。物理量子ビット数に対して階乗なので、代替手段ではなく測定器です。

```python
"""第3章 Example 7: 単純なルータが失うものを、最適解と比べて計測します。
Code Example 6 の続き（同一セッション）。"""
from collections import deque
from itertools import permutations
import numpy as np
from qcheck import *


def optimal_swaps(circ, dev, layout=None):
    """装置写像に対する幅優先探索で求めたSWAP数の最小値です。

    状態は（写像、すでに実行した2量子ビットゲート数）です。1量子ビットゲートは
    ルーティングを要さないので2量子ビットゲートだけが入り、書かれた順序を保ちます。
    これは route() が用いる模型とまったく同じなので、比較は公平です。SWAPのコストは
    すべて1なので、素の幅優先探索が最適解を返します。layout=None とすると開始写像が
    自由になり、レイアウト問題も厳密に解くことになります。
    """
    pairs = [gate_qubits(g) for g in circ if len(gate_qubits(g)) == 2]
    n = dev["n"]

    def advance(loc, k):
        while k < len(pairs):
            a, b = pairs[k]
            if loc[b] in dev["adj"][loc[a]]:
                k += 1
            else:
                return k
        return k

    starts = ([tuple(layout)] if layout is not None
              else [p for p in permutations(range(n))])
    queue, seen = deque(), set()
    for st in starts:
        s = (st, advance(st, 0))
        if s not in seen:
            seen.add(s)
            queue.append((s, 0))
    while queue:
        (loc, k), cost = queue.popleft()
        if k == len(pairs):
            return cost, loc
        for p, q in dev["edges"]:
            nl = list(loc)
            u, v = nl.index(p), nl.index(q)
            nl[u], nl[v] = q, p
            nxt = (tuple(nl), advance(tuple(nl), k))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, cost + 1))
    raise RuntimeError("unreachable on a connected device")


print("単純なルータと厳密なルータの比較。両方が動く小さな事例で")
head = (f"{'circuit':<10}{'device':<14}{'ours, trivial':>14}"
        f"{'optimal, trivial':>18}{'ours, rev trav.':>17}"
        f"{'optimal, free layout':>22}")
print(head)
print("-" * len(head))
rows = []
for label, circ, n_log in [("GHZ n=4", ghz_chain(4), 4),
                           ("GHZ n=5", ghz_chain(5), 5),
                           ("QFT n=3", qft(3), 3),
                           ("QFT n=4", qft(4), 4),
                           ("QFT n=5", qft(5), 5)]:
    for d in (grid(2, 3), heavy_hex_7()):
        ours = route(circ, d)[3]
        opt_fixed = optimal_swaps(circ, d, list(range(d["n"])))[0]
        rev = route(circ, d, reverse_traversal(circ, d))[3]
        opt_free, best_loc = optimal_swaps(circ, d)
        phys, loc0, locf, _ = route(circ, d, best_loc)
        assert routed_error(circ, n_log, d, phys, loc0, locf) < 1e-10
        rows.append((ours, opt_fixed, rev, opt_free))
        print(f"{label:<10}{d['name']:<14}{ours:>14}{opt_fixed:>18}"
              f"{rev:>17}{opt_free:>22}")

ours = np.array([r[0] for r in rows], float)
opt_fixed = np.array([r[1] for r in rows], float)
rev = np.array([r[2] for r in rows], float)
opt_free = np.array([r[3] for r in rows], float)
print(f"\n10行の合計: 本章のルータ {ours.sum():.0f}、"
      f"同じレイアウトでの最適 {opt_fixed.sum():.0f}、"
      f"逆走ヒューリスティクス付き {rev.sum():.0f}、"
      f"レイアウトも最適化した最適 {opt_free.sum():.0f}")
print(f"  最適解に対する超過（同一レイアウト）: "
      f"{100 * (ours.sum() / opt_fixed.sum() - 1):.0f}%")
print(f"  最適解に対する超過（レイアウト込み）: "
      f"{100 * (ours.sum() / opt_free.sum() - 1):.0f}%")
print(f"  レイアウトのヒューリスティクス併用  : "
      f"{100 * (rev.sum() / opt_free.sum() - 1):.0f}%")

print("\n厳密なルータは代替手段ではなく、測定器です:")
from math import factorial
head = f"{'device':<14}{'maps':>10}{'states for QFT n=5':>20}"
print(head)
print("-" * len(head))
pairs = sum(1 for g in qft(5) if len(gate_qubits(g)) == 2)
for N, name in ((6, "grid 2x3"), (7, "heavy-hex 7"), (16, "grid 4x4"),
                (27, "heavy-hex 27")):
    print(f"{name:<14}{factorial(N):>10.3g}{factorial(N) * (pairs + 1):>20.3g}")
```

```text
単純なルータと厳密なルータの比較。両方が動く小さな事例で
circuit   device         ours, trivial  optimal, trivial  ours, rev trav.  optimal, free layout
-----------------------------------------------------------------------------------------------
GHZ n=4   grid 2x3                   2                 2                0                     0
GHZ n=4   heavy-hex 7                1                 1                2                     0
GHZ n=5   grid 2x3                   2                 2                0                     0
GHZ n=5   heavy-hex 7                2                 2                0                     0
QFT n=3   grid 2x3                   2                 1                1                     1
QFT n=3   heavy-hex 7                2                 1                1                     1
QFT n=4   grid 2x3                   5                 3                4                     2
QFT n=4   heavy-hex 7                3                 3                2                     2
QFT n=5   grid 2x3                   9                 5                8                     4
QFT n=5   heavy-hex 7               12                 7               10                     4

10行の合計: 本章のルータ 40、同じレイアウトでの最適 27、逆走ヒューリスティクス付き 28、レイアウトも最適化した最適 14
  最適解に対する超過（同一レイアウト）: 48%
  最適解に対する超過（レイアウト込み）: 186%
  レイアウトのヒューリスティクス併用  : 100%

厳密なルータは代替手段ではなく、測定器です:
device              maps  states for QFT n=5
--------------------------------------------
grid 2x3             720            1.51e+04
heavy-hex 7     5.04e+03            1.06e+05
grid 4x4        2.09e+13            4.39e+14
heavy-hex 27    1.09e+28            2.29e+29
```

**着目点。** 10行の合計で、本章のルータはSWAPを40個挿入し、同じ配置での厳密なルータは27個、配置も選ぶ厳密なルータは14個です。配置を固定すると48%、配置を含めると186%の超過です。前段に逆走ヒューリスティクスを置くと超過は100%に下がります。これらが本章のルータのコストであり、率直に述べたものです。

超過はどこから来るのでしょうか。GHZの行からはどこからも来ません。相互作用グラフがパスなので1つの量子ビットをそれに沿って歩かせるのがすでに最適で、本章のルータは2、1、2、2で厳密なルータと一致します。超過はすべてQFTの行にあり、4つの原因に分離できます。

  * **ゲート順序。** 書かれた順に実行します。本物のルータは依存グラフ上で作業し、フロント層の任意のゲートを実行できるので、しばしばSWAPを丸ごと消せます。4つの中で最大の要因です。
  * **先読み。** 目の前のゲートのために量子ビットを動かし、次のゲートが何を欲しいか一度も問いません。
  * **両端点。** 片方の量子ビットに全部歩かせます。両方を歩かせれば深さは半分になりますが、SWAP数は変わりません。
  * **配置。** 指定がなければ自明な配置であり、表はそれだけで2倍程度の価値があることを示しています。

5番目があり、これは質の差ではなく目的関数の差です。本章はSWAP数を最小化します。ハードウェア上で重要なのは総誤差であり、それはゲート数だけでなく深さにも、そしてどの物理量子ビットを使ったかにも依存します。SWAPを2個余分に払って悪い量子ビットを避けるルータは正しいことも十分にありえます。もう1つ注記すべきは、厳密なルータも本章と同じく書かれたゲート順序の制約を共有しているので、その27は*本章の模型における*最適値であり真の最適値ではないことです。フロント層を使うルータはそれに勝てます。

最後に状態数の表です。探索は（写像、進捗）の対ごとに1状態を訪れるので、$N!$ に2量子ビットゲート数を掛けたものになります。$2\times3$ 格子で1万5千、$4\times4$ 格子で $4\times10^{14}$、27量子ビットで $2\times10^{29}$ です。おもちゃの装置を超えればすべてヒューリスティックであり、ルータを報告する唯一誠実な方法はこの表が行っていることです。最適値が計算できるところでは最適値と比べ、できないところでは別のヒューリスティクスと比べる、ということです。

* * *

## 演習

#### 演習1: 手で求める平均距離

$0{-}1{-}2{-}3{-}4{-}5{-}0$ と結合した6量子ビットの環を考えます。

  1. 全点対の距離表を書き、平均距離 $\bar{d}$ を計算してください。
  2. $3(\bar{d}-1)$ から2量子ビットゲートあたりの余分なCX数を見積もり、Code Example 1 の $2\times3$ 格子と比較してください。
  3. 環の辺は6本、格子は7本です。平均距離が小さいのはどちらで、辺の数だけでそれを予測できますか。
  4. 環の辺を1本切って6量子ビットの1次元鎖にします。$\bar{d}$ は何倍になりますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 6環では距離は隔たり \(k = |i-j| \bmod 6\) だけで決まります。\(k \in \lbrace 1, 5\rbrace\) で \(d = 1\)、\(k \in \lbrace 2,4 \rbrace\) で \(d = 2\)、\(k = 3\) で \(d = 3\) です。15個の無順序対のうち距離1が6個、距離2が6個、距離3が3個なので \(\bar{d} = (6 + 12 + 9)/15 = 1.80\) です。</p>

<p><strong>2.</strong> \(3(1.80 - 1) = 2.40\) 個です。\(2\times3\) 格子は \(\bar{d} = 1.67\) なので \(2.00\) 個です。格子のほうが安いです。</p>

<p><strong>3.</strong> 格子で、\(1.67\) 対 \(1.80\) です。格子は辺も1本多いので、ここでは辺の数が順序を予測しています。ただし一般には予測しません。16量子ビットのheavy-hex断片は辺16本で \(\bar{d} = 3.68\)、16量子ビットの1次元鎖は辺15本で \(\bar{d} = 5.67\) であり、辺1本では説明できない大きな差です。重要なのは辺の配置であり、単一の要約としては直径のほうが良いことが多いです。</p>

<p><strong>4.</strong> 6量子ビットの1次元鎖では15対の距離が \(1\) が5対、\(2\) が4対、\(3\) が3対、\(4\) が2対、\(5\) が1対なので \(\bar{d} = (5 + 8 + 9 + 8 + 5)/15 = 2.33\) です。環から辺を1本外すと \(\bar{d}\) は \(2.33/1.80 = 1.30\) 倍になり、見積りオーバーヘッドは2.40から4.00へ — 結合器を1個削るだけでルーティングコストが67%増えます。接続性は非線形に劣化するので、実機の死んだ結合器は無視されるのではなく報告され、迂回されます。</p>

</details>

#### 演習2: SWAPが0の配置

Code Example 5 は、12量子ビットのGHZ鎖が $4\times4$ 格子で自明な配置なら6個のSWAP、逆走配置なら0個であることを見つけました。

  1. 自明な配置が失敗する理由を1文で説明してください。
  2. 12論理量子ビットを $4\times4$ 格子に置き、GHZ鎖を完全にネイティブにする配置を書き下し、連続する各対が辺であることを手で確認してください。
  3. 同じ格子上で16論理量子ビットのGHZ鎖に対してそのような配置は存在しますか。16量子ビットのheavy-hex断片ではどうですか。
  4. 一般化してください。$n$ 量子ビットのGHZ鎖をSWAP 0で配置できることを保証する結合グラフの性質は何ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 自明な配置は格子を行ごとに番号付けするので、論理量子ビット3と4が物理3と4に落ち、これは隣接する行の両端で結合していません。同じことがすべての行境界で起こります。</p>

<p><strong>2.</strong> 蛇行（ブストロフェドン）経路です。論理 \(0\) から \(11\) に対して物理 \(0{,}1{,}2{,}3{,}7{,}6{,}5{,}4{,}8{,}9{,}10{,}11\) とします。連続する対は第1行に沿った \((0,1),(1,2),(2,3)\)、列を下る \((3,7)\)、第2行を戻る \((7,6),(6,5),(5,4)\)、列を下る \((4,8)\)、そして \((8,9),(9,10),(10,11)\) で、いずれも格子の辺です。</p>

<p><strong>3.</strong> 格子では存在します。蛇行を第4行まで延ばして \(\ldots,11,15,14,13,12\) とします。定義したheavy-hex断片では存在しません。16ノード16辺で長さ12の閉路1つと次数1のペンダント量子ビット4個からなり、Hamilton経路は各ペンダントに入って出なければなりませんが次数1がそれを禁じます。そこで可能な最長経路は13ノード（12閉路にペンダント1個）なので、3個の論理量子ビットはルーティングが必要になります。</p>

<p><strong>4.</strong> グラフが \(n\) 頂点上のパスを含むこと、\(n = N\) のときはHamilton経路を含むことです。これも一般にはNP完全な問いであり、最も易しい回路に対してさえ配置問題の難しさが最も純粋な形で現れています。</p>

</details>

#### 演習3: 置換の帳簿

あるルータが4物理量子ビット上で自明な配置から始め、順に $\mathrm{SWAP}_{0,1}$、$\mathrm{SWAP}_{1,2}$、$\mathrm{SWAP}_{0,1}$ を挿入します。

  1. 各SWAPの後の $\ell$ を追跡してください。$\ell[v]$ は仮想量子ビット $v$ の物理線です。
  2. $\ell_f$ は何で、それは*出力*線のどの置換に対応しますか。
  3. ある人のルータは、常に $\ell_f = \ell_0$ となるように各SWAPの逆を回路の末尾に追加します。それは何を要し、何を買いますか。
  4. その人が正しいのはどういう状況ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 初期は \(\ell = [0,1,2,3]\) です。\(\mathrm{SWAP}_{0,1}\) は物理線0と1にある仮想量子ビットを交換し \(\ell = [1,0,2,3]\) になります。\(\mathrm{SWAP}_{1,2}\) は線1と2にあるものを交換します。仮想0が線1、仮想2が線2にあるので \(\ell = [2,0,1,3]\) です。\(\mathrm{SWAP}_{0,1}\) は線0と1、すなわち仮想1と仮想0を交換し \(\ell = [0,2,1,3]\) になります。</p>

<p><strong>2.</strong> \(\ell_f = [0,2,1,3]\) です。論理0が物理0、論理1が物理2、論理2が物理1、論理3が物理3です。3個のSWAPが1個の互換しか生んでおらず、先読みするルータならもっと短い列を見つけたはずだという徴候です。論理量子ビット1を読み出すには物理線2を測ります。置換は古典的な測定記録の付け替えであり、実行時のコストは0です。</p>

<p><strong>3.</strong> SWAP数が最大で倍になることを要します。SWAP 1個は最もノイジーなゲート種のCX 3個なので、利便性を買う最も高価な方法です。買えるのは第1章の等価性検査がそのまま使えることであり、まさにそれが第1章 Code Example 7 がそうした理由であり、そう明言した理由です。</p>

<p><strong>4.</strong> 置換を古典的な後処理に吸収できない場合です。たとえば回路が、付け替えを知らされない呼び出し側によって出力線を固定されたサブルーチンである場合、回路中測定が特定の物理線へフィードバックされる場合、あるいは同じ物理量子ビットを後続ブロックで既知の割り当てのまま再利用しなければならない場合です。そうした場合には置換を物理的に実現しなければなりません。それ以外の場所では、追跡は無料で、元に戻すのは無駄です。</p>

</details>

#### 演習4: フロント層の自由度

1次元鎖 $0{-}1{-}2{-}3$ 上、自明な配置で3ゲートの回路 $\mathrm{CX}_{0,2}$、$\mathrm{CX}_{1,3}$、$\mathrm{CX}_{0,1}$ を考えます。

  1. Code Example 3 の方針 — 書かれた順 — でルーティングし、SWAP数を数えてください。
  2. 依存関係だけを守ってゲートを並べ替えることを許します。合法な順序は何通りで、もう一方の順序でルータは何個のSWAPを出しますか。
  3. この配置でのSWAP数の最小値と、配置も自由にした場合の最小値を求めてください。
  4. これは Code Example 7 で計測した48%の超過について何を語りますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 4個です。\(\mathrm{CX}_{0,2}\) は距離2なので線0と1のSWAP 1個で \(\ell = [1,0,2,3]\) となりゲートが実行できます。次に \(\mathrm{CX}_{1,3}\) は線0にいる論理1を線3の論理3に近づける必要があり、SWAPがさらに2個で \(\ell = [0,2,1,3]\) に終わります。\(\mathrm{CX}_{0,1}\) はそこで距離2なのでもう1個必要です。<code>route</code> を走らせると4が返ります。</p>

<p><strong>2.</strong> 3番目のゲートは1番目と量子ビット0、2番目と量子ビット1を共有するので最後でなければならず、最初の2つは独立です。合法な順序は \((0{,}2)(1{,}3)(0{,}1)\) と \((1{,}3)(0{,}2)(0{,}1)\) の2通りで、ルータは前者で4個、後者で<strong>2個</strong>を出します。並べ替えだけでコストが半分になり、ルーティングのアルゴリズムは一切変えていません。</p>

<p><strong>3.</strong> 自明な配置では2個で、<code>optimal_swaps</code> がそれを確認します。まず線1と2をSWAPすると \(\ell = [0,2,1,3]\) となり、論理0が論理2の隣に<em>かつ</em>論理1が論理3の隣に同時に来るので、両方のゲートが実行できます。あと1個のSWAPで論理0と1が並びます。SWAP 1個が2ゲートに奉仕することこそ先読みコスト関数が見つけるように設計されているものであり、ゲートごとの最短経路には見えないものです。配置を自由にすると答えは<strong>0</strong>です。相互作用グラフはパス \(2{-}0{-}1{-}3\)、すなわち4頂点上のパスなので1次元鎖にちょうど収まり、配置は \(\ell_0 = [1,2,0,3]\) です。</p>

<p><strong>4.</strong> 超過は調整の問題ではなく構造的であり、しかも Code Example 7 が見つけたのと同じ分かれ方をする、ということです。ここではゲート順序だけで4が2になり、配置だけで2が0になりました。本章のルータは目の前のゲートのための最短経路に決め打ちするので、2ゲートに同時に奉仕するSWAPが見えません。並べ替えと先読みはその盲点を2方向から攻めるものであり、だからSABREは両方を持ち、だから計測された超過がQFTの行に集中したのです。QFTはまさに、1個のSWAPが複数のゲートに奉仕できる対において密です。</p>

</details>

#### 演習5: トランスパイラの報告書を読む

あるトランスパイラが、2量子ビットゲート100個をもつ同一の20量子ビット回路について次を報告しました。

| 対象 | 後のCX | 後の深さ | SWAP |
| --- | --- | --- | --- |
| 全結合 | 100 | 41 | 0 |
| 格子 $5\times4$ | 361 | 158 | 87 |
| heavy-hex、20量子ビット | 634 | 265 | 178 |

  1. SWAP数をCX数と照合してください。整合していますか。
  2. $3(\bar{d}-1)$ を用いて各グラフの平均距離をオーバーヘッドから逆算し、それが妥当かを述べてください。
  3. heavy-hexの行は格子の行の1.8倍のCX数です。そのうちどれだけがグラフで、どれだけが配置でありえますか。
  4. 次に1つだけ数値を求めるとしたら何にしますか。理由も述べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 整合しています。SWAP 1個はCX 3個なので \(100 + 3\times87 = 361\)、\(100 + 3\times178 = 634\) です。もし算術が閉じていなければ、そのトランスパイラはネイティブSWAPを出しているか、ルーティング後にゲートを融合しているかであり、どちらも数値を読む前に知っておく必要があります。</p>

<p><strong>2.</strong> 格子はゲートあたり余分なCXが \(261/100 = 2.61\) 個なので \(\bar{d} \approx 1 + 2.61/3 = 1.87\)。heavy-hexは \(534/100 = 5.34\) なので \(\bar{d} \approx 2.78\) です。どちらも真の平均距離より<em>小さい</em>です。\(5\times4\) 格子は \(\bar{d} = 3.00\)、20量子ビットのheavy-hexはさらに疎で、Code Example 1 の16量子ビット断片ですでに \(3.68\) です。小さいのは期待される向きです。\(3(\bar{d}-1)\) は上界であり、ルータは動かした場所に量子ビットを置いたままにするからで、Code Example 5 が同じ2倍の差を計測しました。逆算した \(\bar{d}\) が真の値より<em>大きい</em>なら、ルータが何か誤っている徴候です。</p>

<p><strong>3.</strong> ほとんどがグラフです。このサイズではheavy-hexの平均距離が格子の約1.4倍であり、難しいグラフはルータに選択肢も与えないのでルーティングコストは平均距離よりやや速く増えます。残りの、おそらく1.2から1.3倍が配置でありえます。Code Example 6 はヒューリスティクスが格子よりheavy-hexで悪く働く行をいくつか見つけました。両者を分離する方法は、複数の配置で再実行して幅を報告することです。</p>

<p><strong>4.</strong> 配置に対する幅、あるいは同じことですが \(k\) 個のランダム配置の最良でのSWAP数です。1回のルーティングパスは、Code Example 6 が2倍以上の幅を示した分布の1点しか報告しません。幅がなければ、良いグラフと運の良い配置を区別する方法がありません。次に近いのは辺ごとの2量子ビットゲート誤り率です。悪い結合器を通る経路は、SWAPを3個余分に払うより高くつくことがあります。</p>

</details>

* * *

## まとめ

### 要点

**1\. 結合グラフはデータとして書かれたゲート機構である**

  * 全結合は共有ボゾンモードから、格子は平面上の容量結合から、heavy-hexは周波数衝突とクロストークを抑えるために次数を3で抑えたことから来ます。代償はより疎なグラフです。
  * 16量子ビットでは辺の数が120から15、平均距離が1.00から5.67まで動きます。heavy-hexは格子の3分の2の辺と38%大きい平均距離を持ちます。

**2\. 平均距離はオーバーヘッドを、上界として予測する**

  * 2量子ビットゲートあたり余分なCXは $3(\bar{d}-1)$ 個。全結合で0、$4\times4$ 格子で5.00、heavy-hexで8.05、16量子ビットの1次元鎖で14.00です。
  * 実測のオーバーヘッドはその約半分です。ルータは動かした場所に量子ビットを置いたままにするので、連続するゲートは独立なゲートより安いからです。ルータがないときに使い、ルータの代用にはしないでください。

**3\. 需要側は回路の相互作用グラフである**

  * GHZ鎖はパスであり、連結なあらゆる装置で何らかの配置によりネイティブになります。QFTは完全グラフで、どんな配置も助けになりません。
  * 12量子ビットのQFTをルーティングするとCX数は格子で2.7倍、heavy-hexで4.6倍になります。第2章の最適化器は回路のゲートの26%を除去しましたが、接続性は最大5倍に増やします。

**4\. 配置はNP困難で、良いヒューリスティクスは神託とほぼ同じ価値がある**

  * コスト0の厳密な場合は部分グラフ同型であり、配置は $N!/(N-n)!$ 通り — 127物理量子ビットに16論理量子ビットなら $2\times10^{33}$ 通りです。
  * 全数探索が可能な装置では、最良と最悪の配置の幅は2倍から4倍です。SABREの逆走の工夫は試した6行すべてで自明な配置に勝ち、3行で最適に到達しました。ルーティング4パス、探索なしです。
  * 本章の簡略版は16量子ビット装置では自明な配置より悪いこともあります。逆走を残して先読みコストを落としているためで、アルゴリズムを半分だけ持つことの価値を正直に計測したものです。

**5\. ルーティング後の回路は書いた回路ではなく、検査はそう述べなければならない**

  * 正しい主張は位相を除いた $U_{\mathrm{phys}}P(\ell_0) = P(\ell_f)(U_{\mathrm{log}}\otimes I)$ であり、試したすべてのルーティング後回路で誤差ちょうど0で通りました。
  * 置換を忘れると、SWAPが挿入されたときは0.35から0.71の誤差、されなかったときはちょうど0になります。つまり置換を無視するテスト群は全結合で通り、実機で黙って失敗します。
  * 16量子ビット装置では行列は $4\times10^9$ 個の数、状態ベクトルは65536個なので検査はランダム入力で走らせます。これは証明ではありませんが、置換の取り違えはほとんどすべての入力で $O(1)$ の誤差です。

**6\. ヒューリスティックなルータは、最適値が存在するところで最適値と比べて報告すべきである**

  * 10個の小さな事例の合計で、本章のルータは40個、同じ配置での厳密なルータは27個、配置も選ぶ厳密なルータは14個 — 48%と186%の超過であり、配置ヒューリスティクスで100%に下がります。
  * 超過はすべてQFTの行にあり、GHZの行では本章のルータはすでに最適です。原因はゲート順序、先読み、片方の端点だけを歩かせること、配置で、おおよそこの重要度順です。
  * 厳密なルーティングは $N!$ にゲート数を掛けた数の状態を訪れます。6量子ビットで $1.5\times10^4$、27量子ビットで $2\times10^{29}$ です。代替手段ではなく測定器です。

**実務上の含意**

  * グラフ、回路の相互作用構造、配置手法を名指しせずにルーティングオーバーヘッドを引用しないでください。3つは乗算的であり、3つ目だけで2倍の価値があります。
  * ルータより先に置換を込めた等価性検査を作り、テスト群に疎な装置を必ず含めてください。置換のバグは全結合では見えません。
  * トランスパイルした回路が予想外に大きいときは、$\mathrm{CX}_{\text{後}} = \mathrm{CX}_{\text{前}} + 3\times\mathrm{SWAP}$ が閉じるかを確認してください。閉じるなら増加はルーティングで、直し方は配置です。閉じないなら増加は合成で、直し方は第2章にあります。

### この先へ

第2章と第3章で、回路は特定の機械が実行できるゲート列まで落ち、その各ゲートは1単位時間かかる抽象的なユニタリとして扱われました。第4章はその抽象化を開きます。回転ゲートは共鳴駆動であり、その長さと形が最下層でのコンパイラの出力であり、そもそもパルスを整形しなければならない理由は、超伝導量子ビットが2準位系ではないことです。速すぎるパルスは第3準位を占有し、リーケージは量子ビットの誤りとして記述できません。あの章は3準位のパルスシミュレータを作り、リーケージを計測し、DRAGで抑制し、それから制御スタックが自分の機械に対して走らせる較正ループを実装します。Rabi振幅、Ramsey周波数、DRAG係数の3つで、いずれも意図的にずらしたパラメータを回復します。ソフトウェアがソフトウェアであることをやめ、実験になる層です。

[← 第2章: 回路最適化とゲート合成](<chapter-2.html>) [第4章: パルスと較正 →](<chapter-4.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の結合グラフ・SWAP数・オーバーヘッド係数は、コード中に定義した特定のグラフ・回路・配置に対する測定値であり、いかなる装置やトランスパイラのベンチマークでもありません。演習5のトランスパイラ報告書は実機のデータではなく教育用に構成した例です。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
