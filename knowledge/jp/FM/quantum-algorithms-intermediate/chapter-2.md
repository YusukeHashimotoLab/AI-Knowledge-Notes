---
title: "第2章: 量子フーリエ変換と位相推定"
chapter_title: "第2章: 量子フーリエ変換と位相推定"
subtitle: 出力を誰も読み出せない O(n²) 回路と、そこから組み立てられる固有値アルゴリズム
reading_time: 45-50分
difficulty: 中級
code_examples: 7
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-algorithms-intermediate/chapter-2.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子アルゴリズム（中級）](<index.html>) > 第2章

第1章は振幅の話でした。Groverのアルゴリズムは確率をマークされた状態へ寄せ、その高速化は二次であり、その語に付くべき留保もすべて付きました。本章は位相の話であり、留保の種類が違います。**量子フーリエ変換**は $O(n^2)$ ゲートで済み、同じ個数の振幅に対する古典高速フーリエ変換は $O(n 2^n)$ 演算を要します。これは指数的高速化のように聞こえますが、そうではありません。そこから作られる**位相推定**アルゴリズムは、ユニタリの固有値を $t$ ビット精度で取り出すのに、そのユニタリの制御適用を $2^t - 1$ 回使います。これは控えめに聞こえますが、誤り耐性量子計算における最重要の基本操作です。

この2つの言明を両方とも正しく述べることが本章の仕事です。QFTが「速いFFT」でないのは、入力が既に量子状態でなければならず、出力振幅を読み出せないからです。QFTができるのは、状態の構造に隠れた*周期*を*測定可能な*ピークに変えることであり、これははるかに狭く、はるかに有用な能力です。位相推定はそれを使い切る機械です。2.4節では材料研究者にとって重要な帰結をまとめます。電子構造とはまさに固有値問題であり、位相推定は入門コースの変分アルゴリズムの誤り耐性時代における後継 — VQEがその近未来的な代用品であるところの本命 — です。

本章のすべては[量子コンピューティング入門 第2章](<../quantum-computing-introduction/chapter-2.html>)で構築したNumPy製の状態ベクトルシミュレータ上で動きます。本章が自己完結するよう下に再掲します。第3章では、ここで構成したQFTと位相推定回路をそのまま使って整数を因数分解しますから、以下の実装は例示ではなく荷重を受ける部材です。

## 学習目標

本章を終えると、次のことができるようになります。

  * 量子フーリエ変換をHadamardと制御位相回転の積として書き、そのゲート数を $n(n+1)/2 + \lfloor n/2 \rfloor$ と数え、回路を密なDFT行列と数値的に照合できる
  * QFTが指数的に速いFFTではない理由 — 入力の書き込み問題、出力振幅の読み出し不能性、1個の振幅の推定にすら要する $1/\delta^2$ のコスト — を正確に述べられる
  * QFTが周期を測定可能なピークに変える一方でオフセットを完全に位相に残すことを示し、周期が $2^n$ を割らないときに何が起きるかを説明できる
  * 位相推定回路を $U$ の制御べき乗と逆QFTから導き、標準的な限界 $t = n + \lceil \log_2(2 + 1/(2\varepsilon)) \rceil$ を用いて、目標精度 $2^{-n}$ のためのレジスタ長を選べる
  * 2、3、4量子ビットのユニタリの固有位相を任意精度で回復し、計数量子ビットを1本増やすごとに位相誤差が半減し深さが倍になることを示せる
  * 補助量子ビット1個と古典フィードバックによる反復的位相推定を実装し、教科書的回路との交換条件を述べられる
  * 位相推定が電子構造への誤り耐性ルートである理由と、試行状態と目的固有ベクトルの重なりがそもそも成功するかを決める量である理由を説明できる

* * *

## 2.1 量子フーリエ変換

### 変換そのもの

$2^n$ 個の複素数からなるベクトルの離散フーリエ変換は、次のユニタリ写像です。

$$ \tilde{x}_k = \frac{1}{\sqrt{2^n}} \sum_{j=0}^{2^n-1} e^{2\pi i jk/2^n}\, x_j $$

量子フーリエ変換はこの同じ行列を、$n$ 量子ビットレジスタの振幅に作用させたものです。基底状態に対しては

$$ \mathrm{QFT}\, \lvert j \rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}\, \lvert k \rangle $$

であり、一般の状態には線形性で作用します。定義に量子的なところは何もありません。ふつうのDFTです。符号の約束 — 順方向で $e^{+2\pi i jk/2^n}$ — は本コース全体で一貫して用いるもので、これにより*逆*QFTは位相を複素共役にした変換になります。

### 回路が短い理由

$k$ を2進で $k = \sum_{l=1}^{n} k_l 2^{n-l}$ と書くと $k/2^n = \sum_l k_l 2^{-l} = 0.k_1k_2\ldots k_n$ です。すると指数関数は $k$ のビットについて因数分解します。

$$ \frac{1}{\sqrt{2^n}}\sum_k e^{2\pi i jk/2^n}\lvert k \rangle = \bigotimes_{l=1}^{n} \frac{\lvert 0 \rangle + e^{2\pi i j/2^{l}}\lvert 1 \rangle}{\sqrt{2}} $$

これがこのアルゴリズムの内容の全部です。基底状態のフーリエ変換は**積状態**であってエンタングルメントが一切なく、各因子は $j$ のビットに依存する位相を1個だけ必要とします。$j = 0.j_1 j_2 \ldots j_n$ も同じように読むと、$l$ 番目の出力因子が担う位相は $2\pi \times 0.j_{n-l+1}\ldots j_n$ であり、これは量子ビット $n-l+1$ へのHadamardと、より下位の各量子ビットからの制御位相回転です。

$$ R_m = \begin{pmatrix} 1 & 0 \cr 0 & e^{2\pi i/2^m} \end{pmatrix} $$

と書けば、回路は次のようになります。量子ビット1にHadamard、量子ビット2から制御-$R_2$、量子ビット3から制御-$R_3$、と続け、次に量子ビット2にHadamardとその制御回転、そして最後に量子ビットの順序を反転します。この構成では出力ビットが逆順に出るからです。ゲート数は

$$ \underbrace{n}_{\text{Hadamard}} + \underbrace{\frac{n(n-1)}{2}}_{\text{制御位相}} + \underbrace{\left\lfloor n/2 \right\rfloor}_{\text{SWAP}} = \frac{n(n+1)}{2} + \left\lfloor \frac{n}{2} \right\rfloor $$

で、互いに素な量子ビット対の回転を並列化すれば深さは $O(n)$ まで下がります。実務的な注意が2つあります。位相 $2\pi/2^m$ は $m$ が大きいと分解不能なほど小さくなり、$m > O(\log n)$ の回転をすべて落としても結果への影響は証明可能に無害です。この**近似QFT**は $O(n\log n)$ ゲートで、実装は例外なくこれを使います。そして最後のSWAPは、出力を消費する側で量子ビットの名前を付け替えれば通常まるごと削除できます。

### QFTではないもの

過大な結論を誘う比較の相手は古典FFTです。$2^n$ 個の数に対して $\Theta(n 2^n)$ の算術演算を要し、QFTの $\Theta(n^2)$ ゲートと比べられます。3つの独立した事実がその結論を阻みます。

**入力が既にそこにある必要があります。** FFTはメモリ上の $2^n$ 個の数を受け取ります。QFTは、その数を振幅として*持っている*量子状態を受け取ります。古典的なリストから任意の $2^n$ 振幅状態を用意するには一般に $\Theta(2^n)$ ゲートが必要です。これが状態準備問題であり、第1章で論じたQRAM問題が細部ではない理由です。Shorのアルゴリズムのように状態が先行する量子計算から出てくるなら、このコストは生じません。データが古典的であれば生じ、それだけで優位が消えます。

**出力は読み出せません。** QFTの後、変換された係数は振幅です。測定は添字 $k$ を1つ、確率 $\lvert \tilde{x}_k \rvert^2$ で返すのであって、数 $\tilde{x}_k$ を返しません。1個の確率を加法的精度 $\delta$ で推定するには標本抽出で $O(1/\delta^2)$ 回の繰り返しが必要で（これは下限ではありません。振幅推定はコヒーレントな深さを払って $O(1/\delta)$ に達します）、$2^n$ 個すべてを回復するには少なくとも $2^n$ 個の数を出力しなければならないので、指数的な節約は残りえません。*位相*の回復にはさらに干渉実験が必要です。

**有用なことを何も頼んでいません。** 入力と出力が無料だとしても、古典FFTが誰かのボトルネックであることはまずありません。小さな定数を伴う $n2^n$ は科学計算で最速のアルゴリズムの一つです。QFTが役目を得るのは、より速い変換だからではなく、変換後の状態に*ただ1つ*の問い — 「ピークはどこか」 — を投げる回路の内側に座るからです。それはちょうど測定1回ぶんの情報量です。

| | 古典FFT | 量子FFT（QFT） |
| --- | --- | --- |
| 入力 | メモリ上の $2^n$ 個の数 | $n$ 量子ビット状態の振幅 |
| コスト | $\Theta(n 2^n)$ 算術演算 | $\Theta(n^2)$ ゲート、近似版なら $\Theta(n \log n)$ |
| 出力 | 変換後の $2^n$ 個の数すべて | 1回の実行につき添字1個 |
| 係数1個の読み出し | 無料 | 標本抽出で $O(1/\delta^2)$ 回の繰り返し、しかも大きさのみ |
| 有用な用途 | フィルタ、畳み込み、スペクトル | 周期や位相の抽出、そして1回の測定 |

誠実な一行要約はこうです。QFTはデータを変換するために使う変換ではありません。隠れた周期性が測定1回の答えになる、そういう基底変換です。

### Code Example 1: シミュレータの再掲

本コース全体は入門シリーズの状態ベクトルシミュレータ上で動きます。ここに逐語で — 99行すべて、無変更で — 再掲しますので、本章のどこも探しに行かねばならないファイルに依存しません。`qcsim.py` として保存してください。以降の各例は `from qcsim import *` で始まります。

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

このファイルの約束が2つ、以下で常時使われるので再掲しておきます。量子ビットの順序は**ビッグエンディアン**で、qubit 0 がケットの最左であり振幅添字の最上位ビットです。したがって $t$ 量子ビットの計数レジスタで測定される整数は $k = \sum_j q_j 2^{t-1-j}$ です。そして `apply_gate` は $2^k \times 2^k$ 行列を任意の $k$ 個の指定量子ビットに作用させます。これが以下で制御多量子ビット演算を、添字算術の演習ではなく1行で書ける理由です。

### Code Example 2: QFT回路とDFT行列の照合

回路に関する主張は行列で検査すべきです。以下は上で導いたとおりHadamardと制御位相からQFTを組み、それを全基底状態に作用させて実装している行列を取り出し、密なDFT行列と列ごとに比較します。

```python
import numpy as np
from qcsim import *

SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def cphase(theta):
    """量子ビット対に作用する制御位相ゲート diag(1, 1, 1, exp(i theta)) です。"""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


def qft(state, qubits, n):
    """指定した量子ビット列へのQFTです。qubits[0] が最上位ビットです。

    各量子ビットにHadamardを掛け、続いてより下位の各量子ビットから制御位相
    を掛け、最後に量子ビットの順序を反転します。ゲート数は回転が m(m+1)/2
    個、SWAPが floor(m/2) 個です。
    """
    m = len(qubits)
    for j in range(m):
        state = apply_gate(state, H, [qubits[j]], n)
        for k in range(j + 1, m):
            state = apply_gate(state, cphase(np.pi / 2 ** (k - j)),
                               [qubits[k], qubits[j]], n)
    for j in range(m // 2):
        state = apply_gate(state, SWAP4, [qubits[j], qubits[m - 1 - j]], n)
    return state


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


def qft_matrix_from_circuit(m):
    """第 j 列は基底状態 |j> に対する回路の出力です。"""
    cols = []
    for j in range(2 ** m):
        psi = np.zeros(2 ** m, dtype=complex)
        psi[j] = 1.0
        cols.append(qft(psi, list(range(m)), m))
    return np.column_stack(cols)


def dft_matrix(m):
    """ユニタリなDFT行列 F[k, j] = exp(2 pi i j k / 2^m) / sqrt(2^m) です。"""
    d = 2 ** m
    j, k = np.meshgrid(np.arange(d), np.arange(d))
    return np.exp(2j * np.pi * j * k / d) / np.sqrt(d)


print("QFT circuit against the DFT matrix")
print("-" * 68)
print(f"  {'m':>3}{'dim':>7}{'H+phase gates':>14}{'swaps':>7}"
      f"{'max |U_circ - F|':>20}")
for m in range(1, 8):
    U = qft_matrix_from_circuit(m)
    F = dft_matrix(m)
    n_hp = m * (m + 1) // 2          # Hadamard m 個 + 位相ゲート m(m-1)/2 個
    print(f"  {m:>3}{2**m:>7}{n_hp:>14}{m//2:>7}"
          f"{np.max(np.abs(U - F)):>20.2e}")

print("\nUnitarity and inversion, m = 5")
print("-" * 68)
m = 5
U = qft_matrix_from_circuit(m)
print(f"  max |U^dag U - I|            = "
      f"{np.max(np.abs(U.conj().T @ U - np.eye(2**m))):.2e}")
rng = np.random.default_rng(7)
psi = rng.normal(size=2 ** m) + 1j * rng.normal(size=2 ** m)
psi /= np.linalg.norm(psi)
back = iqft(qft(psi, list(range(m)), m), list(range(m)), m)
print(f"  max |iqft(qft(psi)) - psi|   = {np.max(np.abs(back - psi)):.2e}")
fwd = qft(psi, list(range(m)), m)
print(f"  max |qft(psi) - F psi|       = "
      f"{np.max(np.abs(fwd - dft_matrix(m) @ psi)):.2e}")

print("\nQFT of a uniform state and of a single basis state, m = 3")
print("-" * 68)
m = 3
unif = np.ones(2 ** m, dtype=complex) / np.sqrt(2 ** m)
out = qft(unif, list(range(m)), m)
print("  QFT|+++>  amplitudes:",
      "  ".join(f"{v if abs(v) > 5e-4 else 0.0:+.3f}" for v in out.real))
out = qft(ket('001'), list(range(m)), m)
print("  QFT|001>  |amp|     :",
      "  ".join(f"{abs(v):.3f}" for v in out))
print("  QFT|001>  phase/2pi :",
      "  ".join(f"{np.angle(v)/(2*np.pi) % 1.0:.3f}" for v in out))

print("\nGate count: QFT versus a classical FFT on 2^m numbers")
print("-" * 68)
print(f"  {'m':>4}{'2^m':>22}{'QFT gates':>12}{'FFT ops ~ m 2^m':>22}")
for m in [3, 10, 20, 30, 50]:
    print(f"  {m:>4}{2**m:>22d}{m*(m+1)//2 + m//2:>12d}{m*2**m:>22d}")
```

```text
QFT circuit against the DFT matrix
--------------------------------------------------------------------
    m    dim H+phase gates  swaps    max |U_circ - F|
    1      2             1      0            8.66e-17
    2      4             3      1            2.69e-16
    3      8             6      1            1.42e-15
    4     16            10      2            3.78e-15
    5     32            15      2            5.62e-15
    6     64            21      3            7.98e-15
    7    128            28      3            1.21e-14

Unitarity and inversion, m = 5
--------------------------------------------------------------------
  max |U^dag U - I|            = 9.99e-16
  max |iqft(qft(psi)) - psi|   = 3.86e-16
  max |qft(psi) - F psi|       = 2.73e-15

QFT of a uniform state and of a single basis state, m = 3
--------------------------------------------------------------------
  QFT|+++>  amplitudes: +1.000  +0.000  +0.000  +0.000  +0.000  +0.000  +0.000  +0.000
  QFT|001>  |amp|     : 0.354  0.354  0.354  0.354  0.354  0.354  0.354  0.354
  QFT|001>  phase/2pi : 0.000  0.125  0.250  0.375  0.500  0.625  0.750  0.875

Gate count: QFT versus a classical FFT on 2^m numbers
--------------------------------------------------------------------
     m                   2^m   QFT gates       FFT ops ~ m 2^m
     3                     8           7                    24
    10                  1024          60                 10240
    20               1048576         220              20971520
    30            1073741824         480           32212254720
    50      1125899906842624        1300     56294995342131200
```

**注目すべき点。** 回路は $m = 7$ でDFT行列を $10^{-14}$ まで再現し、ずれは累積丸めぶんだけ増えます。これは近似ではなく検証です。ゲート数の列は上で予測した $m(m+1)/2$ そのもので、`iqft(qft(psi))` は入力を $4 \times 10^{-16}$ で戻します。逆順・共役位相の構成が本当に逆変換であることの確認です。

最後の2ブロックが覚えるべきところです。$\mathrm{QFT}\lvert +++ \rangle = \lvert 000 \rangle$ で、一様重ね合わせはゼロ周波数状態です。古典変換とまったく同じです。$\mathrm{QFT}\lvert 001 \rangle$ は8通りすべてで*大きさが一様*で、情報はすべて位相にあり、添字1つ進むごとに $1/8$ 回転します。基底状態とその変換は局在の両極端であり、後者は計算基底では測定不能です。上の節の内容が3行の出力に現れています。最後に $m = 50$ ではQFTが1300ゲート、FFTが $5.6 \times 10^{16}$ 演算です。この比較は本物の算術ですが、それでも有用な高速化にはなりません。理由は既に述べた3つです。

### Code Example 3: 読める周期と読めない位相

QFTが実際に提供する能力は周期検出です。この例では公差 $r$ の等差数列に台を持つ状態を用意し、変換して、測定が何を返すかを見ます。まず $r$ がレジスタ長を割る場合、次に割らない場合 — 第3章が付き合わなければならない状況です。

```python
"""第2章 Code Example 3: QFTが与えてくれるものと、与えてくれないものです。
Code Example 2 の続き（同一セッション）です。"""

import numpy as np
import matplotlib.pyplot as plt


def periodic_state(m, r, offset):
    """2^m 次元レジスタ上の {j : j = offset (mod r)} の一様重ね合わせです。"""
    psi = np.zeros(2 ** m, dtype=complex)
    psi[np.arange(offset, 2 ** m, r)] = 1.0
    return psi / np.linalg.norm(psi)


m = 6
print(f"Case 1: period r = 8 divides 2^m = {2**m}")
print("-" * 70)
p0 = probs(qft(periodic_state(m, 8, 0), list(range(m)), m))
print("  nonzero probabilities of the QFT output (offset s = 0):")
print("   " + "   ".join(f"k={k}: {p0[k]:.4f}"
                         for k in np.flatnonzero(p0 > 1e-9)))
print("\n  the same distribution for every offset s:")
for s in range(1, 8):
    ps = probs(qft(periodic_state(m, 8, s), list(range(m)), m))
    print(f"    s = {s}: max |p_s - p_0| = {np.max(np.abs(ps - p0)):.2e}")

print("\n  the offset lives in the phases, which no measurement returns:")
a0 = qft(periodic_state(m, 8, 0), list(range(m)), m)
a3 = qft(periodic_state(m, 8, 3), list(range(m)), m)
print(f"  {'k':>4}{'|amp| s=0':>12}{'|amp| s=3':>12}"
      f"{'phase diff/2pi':>17}{'k s / 2^m mod 1':>18}")
for k in np.flatnonzero(p0 > 1e-9):
    d = (np.angle(a3[k]) - np.angle(a0[k])) / (2 * np.pi) % 1.0
    print(f"  {k:>4}{abs(a0[k]):>12.4f}{abs(a3[k]):>12.4f}{d:>17.4f}"
          f"{(k*3/2**m) % 1.0:>18.4f}")

print(f"\nCase 2: period r = 5 does not divide 2^m = {2**m}")
print("-" * 70)
out5 = qft(periodic_state(m, 5, 0), list(range(m)), m)
p5 = probs(out5)
top = np.sort(np.argsort(p5)[::-1][:5])
print(f"  2^m / r = {2**m/5:.2f}; the five largest peaks:")
print("   " + "   ".join(f"k={k}: {p5[k]:.4f}" for k in top))
print(f"  total probability in those five bins: {p5[top].sum():.4f}")
print("  peaks near, not at, multiples of 12.80 -- the leakage that "
      "Chapter 3 has to postprocess")

print("\nCost of reading the output amplitudes")
print("-" * 70)
rng = np.random.default_rng(11)
psi = rng.normal(size=2 ** m) + 1j * rng.normal(size=2 ** m)
psi /= np.linalg.norm(psi)
p_exact = probs(qft(psi, list(range(m)), m))
print(f"  {'shots':>10}{'max |p_hat - p|':>18}{'1/sqrt(shots)':>16}")
for shots in [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]:
    counts = rng.multinomial(shots, p_exact) / shots
    print(f"  {shots:>10d}{np.max(np.abs(counts - p_exact)):>18.5f}"
          f"{1/np.sqrt(shots):>16.5f}")
print("\n  shots to pin all 2^m probabilities to 1 per cent:"
      f" ~ m ln2/delta^2 = {m * np.log(2) * 10**4:.0e}")
print("  (logarithmic in the number of bins, not linear: the table above"
      " crosses")
print("  1 per cent nearer 3e+03 shots)")
print("  and the phases would need separate interference experiments.")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for s, style in [(0, "-o"), (3, "--s")]:
    ax[0].plot(probs(qft(periodic_state(m, 8, s), list(range(m)), m)),
               style, ms=4, lw=1, label=f"offset s = {s}")
ax[0].set_title("r = 8 divides 64: exact peaks")
ax[0].legend(fontsize=8)
ax[1].plot(p5, "-o", ms=4, lw=1, color="tab:red")
for j in range(1, 5):
    ax[1].axvline(j * 2 ** m / 5, color="k", ls=":", lw=0.8)
ax[1].set_title("r = 5 does not: peaks leak")
ax[2].bar(np.arange(2 ** m), p_exact, color="tab:purple", width=0.8)
ax[2].set_title("random state: nothing to read")
for a in ax:
    a.set_xlabel("measured index k"); a.set_ylabel("probability")
plt.tight_layout()
plt.show()
```

```text
Case 1: period r = 8 divides 2^m = 64
----------------------------------------------------------------------
  nonzero probabilities of the QFT output (offset s = 0):
   k=0: 0.1250   k=8: 0.1250   k=16: 0.1250   k=24: 0.1250   k=32: 0.1250   k=40: 0.1250   k=48: 0.1250   k=56: 0.1250

  the same distribution for every offset s:
    s = 1: max |p_s - p_0| = 1.60e-50
    s = 2: max |p_s - p_0| = 4.28e-50
    s = 3: max |p_s - p_0| = 2.14e-50
    s = 4: max |p_s - p_0| = 2.14e-50
    s = 5: max |p_s - p_0| = 2.14e-50
    s = 6: max |p_s - p_0| = 2.14e-50
    s = 7: max |p_s - p_0| = 4.81e-50

  the offset lives in the phases, which no measurement returns:
     k   |amp| s=0   |amp| s=3   phase diff/2pi   k s / 2^m mod 1
     0      0.3536      0.3536           0.0000            0.0000
     8      0.3536      0.3536           0.3750            0.3750
    16      0.3536      0.3536           0.7500            0.7500
    24      0.3536      0.3536           0.1250            0.1250
    32      0.3536      0.3536           0.5000            0.5000
    40      0.3536      0.3536           0.8750            0.8750
    48      0.3536      0.3536           0.2500            0.2500
    56      0.3536      0.3536           0.6250            0.6250

Case 2: period r = 5 does not divide 2^m = 64
----------------------------------------------------------------------
  2^m / r = 12.80; the five largest peaks:
   k=0: 0.2031   k=13: 0.1771   k=26: 0.1146   k=38: 0.1146   k=51: 0.1771
  total probability in those five bins: 0.7865
  peaks near, not at, multiples of 12.80 -- the leakage that Chapter 3 has to postprocess

Cost of reading the output amplitudes
----------------------------------------------------------------------
       shots   max |p_hat - p|   1/sqrt(shots)
         100           0.04393         0.10000
        1000           0.01644         0.03162
       10000           0.00339         0.01000
      100000           0.00210         0.00316
     1000000           0.00047         0.00100

  shots to pin all 2^m probabilities to 1 per cent: ~ m ln2/delta^2 = 4e+04
  (logarithmic in the number of bins, not linear: the table above crosses
  1 per cent nearer 3e+03 shots)
  and the phases would need separate interference experiments.
```

**注目すべき点。** 64次元レジスタで $r = 8$ のとき、出力はちょうど8個の添字、すなわち $2^m/r = 8$ の倍数に台を持ち、それぞれ確率 $1/8$ です。オフセット $s$ を変えても分布は $10^{-50}$ しか変わりません。つまりまったく変わりません。オフセットは失われたのではなく位相に格納されており、表は $s = 0$ と $s = 3$ の位相差がどのピークでも厳密に $ks/2^m$ であることを確認しています。**QFTはオフセットを周期と引き換えにする**のであり、測定は測定可能な側の半分を回収します。

$r = 5$ のブロックが誠実な版です。5は64を割らず、ピークは $12.8$ の倍数の近くに来て上には来ません。最良の5つのビンには確率の $79\%$ しか座らず、残りは他の59ビンに散ります。何も壊れていません — ピークは周期が言う場所に1ビン以内で来ています — が、測定された添字を周期に変える古典後処理が必要になり、その段が第3章の連分数です。

最後のブロックは代案の値札ですが、その値札は問題の見かけから予想されるものとは違います。変換後の確率を標本から推定すると $1/\sqrt{\text{ショット数}}$ で収束し、$2^m$ 個のビンのうち*最悪*のものを絶対精度 $\delta$ に抑えるのに必要なショット数は、ビン数に対して対数的にしか増えません。$O(m/\delta^2)$ であり、$m = 6$、$\delta = 0.01$ なら $4 \times 10^4$ 程度、実測の表が1%を切るのは $3 \times 10^3$ 付近です。フーリエ変換エンジンという読み方を絶望的にしているのはこの数ではなく、絶対精度 $\delta$ が何を買うかです。確率そのものが $2^{-m}$ の程度なので、フーリエ変換に期待される*相対*精度を要求すると $\delta \sim \varepsilon 2^{-m}$ が必要になり、したがって $O(2^m m/\varepsilon^2)$ ショットになります。しかもその $2^m$ 個の数は、測定された添字1個ずつしか出てきません。

* * *

## 2.2 位相推定

### 問題設定

$U$ をユニタリ、$\lvert u \rangle$ をその固有ベクトルとします。

$$ U \lvert u \rangle = e^{2\pi i \varphi} \lvert u \rangle, \qquad \varphi \in [0, 1) $$

別の量子ビットで制御して $U$ を作用させる能力と、$\lvert u \rangle$ のコピーが与えられたとき、$\varphi$ を推定してください。仕様はこれだけで、その射程はどれだけ多くの問題がこれに当てはまるかから来ます。ハミルトニアン $H$ に対して $U = e^{-iH\tau}$ とすれば、固有位相はスケールされたエネルギーです。$U$ を $N$ を法とする $a$ 倍写像とすれば、固有位相は $s/r$ で $r$ は乗法的位数、これが第3章です。$U$ をGroverのアルゴリズムの鏡映とすれば、その位相の推定は解の個数を数えることになり、これが振幅推定です。

### 干渉が起きる場所

計数量子ビット $t$ 個を一様重ね合わせに、系を $\lvert u \rangle$ に用意します。計数量子ビット $j$ で制御して $U^{2^{t-1-j}}$ を作用させます。$\lvert u \rangle$ は固有ベクトルなので、各制御演算は量子ビット $j$ が $\lvert 1 \rangle$ である枝に数 $e^{2\pi i \varphi 2^{t-1-j}}$ を掛け、系レジスタには触りません。因子を集めると

$$ \frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi i \varphi k}\, \lvert k \rangle \otimes \lvert u \rangle $$

で、$k = \sum_j q_j 2^{t-1-j}$ は通例どおりです。これを基底状態のQFTと比べてください。$2^t\varphi$ が整数であるときこれはちょうど $\mathrm{QFT}\lvert 2^t \varphi \rangle$ です。したがって計数レジスタに**逆**QFTを掛けると $\lvert 2^t \varphi \rangle$ が現れ、これを測定すると $\varphi$ の2進最初の $t$ 桁が確実に返ります。位相推定はQFTを逆に走らせたものであり、制御べき乗はそもそも位相をレジスタに書き込む部分です。

ここから2つの帰結が直ちに従います。$U$ の作用回数の合計は $1 + 2 + \cdots + 2^{t-1} = 2^t - 1$ なので、精度 $\varepsilon \sim 2^{-t}$ は $\Theta(1/\varepsilon)$ 回の作用を要します。深さは精度に反比例します。そして固有ベクトルは一度も乱されていません。つまり入力が重ね合わせ $\sum_l c_l \lvert u_l \rangle$ なら*推定値の*重ね合わせが生じ、計数レジスタの測定は固有位相 $\varphi_l$ を確率 $\lvert c_l \rvert^2$ で返します。これが厳密に成り立つのはどの $\varphi_l$ も $t$ ビットで厳密に表せるときで、そうでなければ各 $\lvert c_l \rvert^2$ は、下で述べる $P_{\text{best}}$ の形で $\varphi_l$ の周りのビンに広がります。位相推定は分光器であり、入力状態は何を見るかの選択です。

### ビット数、精度、そして標準的な限界

$2^t \varphi$ が整数でないとき、ピークはもはやデルタ関数ではありません。$\varphi$ から最近接の $2^{-t}$ の倍数までの距離を $\delta$ と書くと、その最近接添字を測る確率は

$$ P_{\text{best}} = \frac{1}{2^{2t}} \frac{\sin^2\left(2^t \pi \delta\right)}{\sin^2\left(\pi \delta\right)} \; \ge \; \frac{4}{\pi^2} \approx 0.405 $$

で、最悪はちょうどビンの中間 $\delta = 2^{-t-1}$ です。ここから従う標準的な言明は、実務でレジスタ長を決めるやり方そのものなので覚える価値があります。$\lvert \tilde{\varphi} - \varphi \rvert < 2^{-n}$ を満たす推定値 $\tilde{\varphi}$ を確率 $1-\varepsilon$ 以上で得るには

$$ t = n + \left\lceil \log_2\left(2 + \frac{1}{2\varepsilon}\right) \right\rceil $$

個の計数量子ビットを使います。追加ビットは分解能ではなく保険です。正しいビンに落ちる確率を買うのであり、代価は回路深さの $2^{\text{追加}}$ 倍です。保証されるのは*数* $\tilde{\varphi}$ についてであって、先頭のビット列についてではありません。この違いは実在します。$\varphi = 1/2 - 2^{-12}$ では最近接の4ビット格子点は $0.1000_2$ で、そのビット列は $\varphi$ 自身の先頭ビットの補数ですが、$2^{-4}$ 以内にあり正解です。だから Code Example 4 の3番目のブロックが測るのは $\Pr[\lvert \tilde{\varphi} - \varphi \rvert < 2^{-n}]$ であり、これが定理の限界づける量です。この定理が*言っていない*ことにも注意してください。与えた固有ベクトルが望んだものかどうかについては何も言いません。それが2.4節の問題です。

### Code Example 4: 既知の位相に対する位相推定

考えられる最も清潔な検査対象は、1量子ビットユニタリ $\mathrm{diag}(1, e^{2\pi i \varphi})$ です。固有ベクトル $\lvert 1 \rangle$ は基底状態で、固有位相は自分で選んだものです。

```python
"""第2章 Code Example 4: 既知の位相に対する位相推定です。
Code Example 3 の続き（同一セッション）です。"""

import numpy as np


def controlled(U):
    """U のブロック対角な制御版です。制御量子ビットは先頭の1個です。"""
    d = U.shape[0]
    C = np.eye(2 * d, dtype=complex)
    C[d:, d:] = U
    return C


def qpe_state(U, sys_state, t):
    """教科書どおりのQPEを実行します。計数量子ビット t 個（qubit 0 が最上位）
    の後ろに系のレジスタが並びます。

    逆QFTを掛けた後の全状態を返します。制御べき乗の総コストは U の作用
    2^t - 1 回です。
    """
    n_sys = int(np.log2(sys_state.size))
    n = t + n_sys
    state = np.kron(ket('0' * t), sys_state)
    for j in range(t):
        state = apply_gate(state, H, [j], n)
    for j in range(t):
        Up = np.linalg.matrix_power(U, 2 ** (t - 1 - j))
        state = apply_gate(state, controlled(Up), [j] + list(range(t, n)), n)
    return iqft(state, list(range(t)), n)


def counting_probs(state, t):
    """計数量子ビット t 個の周辺分布です。"""
    return probs(state).reshape(2 ** t, -1).sum(axis=1)


def phase_gate(phi):
    """1量子ビットの U = diag(1, exp(2 pi i phi)) です。|1> が固有ベクトルです。"""
    return np.array([[1.0, 0.0], [0.0, np.exp(2j * np.pi * phi)]],
                    dtype=complex)


print("A phase that is exactly representable: phi = 0.375 = 0.011 (binary)")
print("-" * 74)
phi = 0.375
print(f"  {'t':>3}{'best k':>8}{'k/2^t':>12}{'|error|':>12}"
      f"{'P(best)':>10}{'ctrl-U calls':>14}")
for t in [3, 4, 6]:
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    k = int(np.argmax(p))
    print(f"  {t:>3}{k:>8}{k/2**t:>12.6f}{abs(k/2**t - phi):>12.2e}"
          f"{p[k]:>10.6f}{2**t - 1:>14d}")

print("\nA phase that is not: phi = 1/3 = 0.0101010101... (binary)")
print("-" * 74)
phi = 1.0 / 3.0
print(f"  {'t':>3}{'best k':>8}{'k/2^t':>12}{'|error|':>12}{'2^-t':>10}"
      f"{'P(best)':>10}{'P(best 2)':>11}")
for t in range(3, 13):
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    order = np.argsort(p)[::-1]
    k = int(order[0])
    print(f"  {t:>3}{k:>8}{k/2**t:>12.6f}{abs(k/2**t - phi):>12.2e}"
          f"{2.0**-t:>10.2e}{p[k]:>10.6f}{p[order[0]]+p[order[1]]:>11.6f}")

print("\nThe standard guarantee: t = n + ceil(log2(2 + 1/(2 eps)))")
print("-" * 74)
print(f"  {'n bits':>7}{'eps':>9}{'extra':>7}{'t':>5}"
      f"{'measured P(|err| < 2^-n)':>27}")
for n_bits, eps in [(2, 0.25), (2, 0.05), (4, 0.25), (4, 0.05), (6, 0.05)]:
    extra = int(np.ceil(np.log2(2.0 + 1.0 / (2 * eps))))
    t = n_bits + extra
    p = counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t)
    ks = np.arange(2 ** t)
    err = np.minimum(np.abs(ks / 2 ** t - phi), 1.0 - np.abs(ks / 2 ** t - phi))
    good = p[err < 2.0 ** -n_bits].sum()
    print(f"  {n_bits:>7}{eps:>9.2f}{extra:>7}{t:>5}{good:>27.6f}")

print("\nSuperposition input: the register measures which eigenvalue it found")
print("-" * 74)
U = np.diag([np.exp(2j * np.pi * 0.25), np.exp(2j * np.pi * 0.75)])
mix = np.array([np.sqrt(0.8), np.sqrt(0.2)], dtype=complex)
t = 4
p = counting_probs(qpe_state(U, mix, t), t)
for k in np.flatnonzero(p > 1e-9):
    print(f"  k = {k:>2}  ->  phi = {k/2**t:.4f}   probability {p[k]:.4f}")
print("  the two eigenphases are returned with the input's own weights, "
      "0.8 and 0.2")
```

```text
A phase that is exactly representable: phi = 0.375 = 0.011 (binary)
--------------------------------------------------------------------------
    t  best k       k/2^t     |error|   P(best)  ctrl-U calls
    3       3    0.375000    0.00e+00  1.000000             7
    4       6    0.375000    0.00e+00  1.000000            15
    6      24    0.375000    0.00e+00  1.000000            63

A phase that is not: phi = 1/3 = 0.0101010101... (binary)
--------------------------------------------------------------------------
    t  best k       k/2^t     |error|      2^-t   P(best)  P(best 2)
    3       3    0.375000    4.17e-02  1.25e-01  0.687838   0.862778
    4       5    0.312500    2.08e-02  6.25e-02  0.684895   0.856855
    5      11    0.343750    1.04e-02  3.12e-02  0.684162   0.855386
    6      21    0.328125    5.21e-03  1.56e-02  0.683979   0.855020
    7      43    0.335938    2.60e-03  7.81e-03  0.683933   0.854928
    8      85    0.332031    1.30e-03  3.91e-03  0.683922   0.854905
    9     171    0.333984    6.51e-04  1.95e-03  0.683919   0.854899
   10     341    0.333008    3.26e-04  9.77e-04  0.683918   0.854898
   11     683    0.333496    1.63e-04  4.88e-04  0.683918   0.854898
   12    1365    0.333252    8.14e-05  2.44e-04  0.683918   0.854898

The standard guarantee: t = n + ceil(log2(2 + 1/(2 eps)))
--------------------------------------------------------------------------
   n bits      eps  extra    t   measured P(|err| < 2^-n)
        2     0.25      2    4                   0.970284
        2     0.05      4    6                   0.992542
        4     0.25      2    6                   0.962624
        4     0.05      4    8                   0.990626
        6     0.05      4   10                   0.990511

Superposition input: the register measures which eigenvalue it found
--------------------------------------------------------------------------
  k =  4  ->  phi = 0.2500   probability 0.8000
  k = 12  ->  phi = 0.7500   probability 0.2000
  the two eigenphases are returned with the input's own weights, 0.8 and 0.2
```

**注目すべき点。** $\varphi = 0.375 = 0.011_2$ ではアルゴリズムは厳密です。計数量子ビット3個が $k = 3$ を確率1で返し、量子ビットを増やしても深さ以外は何も変わりません。これが決定論的な場合で、第3章の $N = 15$ がまさにこの場合になります。

$\varphi = 1/3$ では絵が一般的になり、この表が身体に入れるべきものです。誤差は $2^{-t}$ を追って $t = 12$ で $8 \times 10^{-5}$ に達し、約束どおり量子ビット1本ごとに半減します。一方 $P_{\text{best}}$ は $0.6839$ に落ち着き、最良の2ビンで $0.8549$ です。どちらも $4/\pi^2$ の床より十分上にあり、どちらも*$t$ に依存しません*。これが重要な構造的事実です。追加の計数量子ビットが買うのは分解能で、確信度ではありません。確信度は $\lceil \log_2(2 + 1/(2\varepsilon))\rceil$ の詰めビットから来ます。3番目のブロックがその限界を経験的に確認しています。$\varphi$ から $2^{-n}$ 以内に落ちる測定確率 — これが定理の約束する量です — はどの行でも $1 - \varepsilon$ を超え、しかも余裕をもって超えます。限界が保守的だからです。

最後のブロックは分光器としての位相推定です。重み $0.8$ と $0.2$ で2つの固有ベクトルを重ね合わせた入力は、2つの固有位相を確率 $0.8$ と $0.2$ で返します。平均は取られません。レジスタはこの実行がどの固有値を見つけたかを報告します。

### Code Example 5: 多量子ビットユニタリの固有位相

さて本題です。2、3、4量子ビット上のハミルトニアンに対する $U = e^{-iH\tau}$ です。ここで実務的な点が2つ初めて現れます。$H$ の定数シフトは無料であり、スペクトル全体を位相の窓 $[0,1)$ に収めるために必要です。そして $\tau$ はその窓が許す限り大きく取るべきです。1ビットあたりの分解能を決めるからです。

```python
"""第2章 Code Example 5: 多量子ビットユニタリの固有位相と、計数量子ビット数
に対する誤差の減り方です。
Code Example 4 の続き（同一セッション）です。"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def kron_all(mats):
    return reduce(np.kron, mats)


def tfim_matrix(n, h):
    """横磁場Ising鎖 H = -sum Z_i Z_{i+1} - h sum X_i です。"""
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        M -= kron_all([Z if q in (i, i + 1) else I2 for q in range(n)])
    for i in range(n):
        M -= h * kron_all([X if q == i else I2 for q in range(n)])
    return M


def unitary_from_h(M, tau):
    """固有値分解による exp(-i M tau) です。M はエルミートである必要があります。"""
    w, v = np.linalg.eigh(M)
    return (v * np.exp(-1j * w * tau)) @ v.conj().T


def phase_window(w, frac=0.87, pad=0.05):
    """全固有位相を (0, 1) に収める定数シフト c と発展時間 tau を返します。

    基底状態を phi = frac に、スペクトルの上端を phi = frac*pad/(1+pad) 付近
    に置きます。ハミルトニアンの定数シフトは無料です。固有ベクトルもエネル
    ギー差も動かしません。スペクトル幅が許す限り tau を大きく取ることが、
    この窓を有効に使う条件です。
    """
    span = w[-1] - w[0]
    c = w[-1] + pad * span
    tau = 2 * np.pi * frac / (c - w[0])
    return c, tau


h_field = 0.5
print(f"Transverse-field Ising chain, h = {h_field}: eigenphases from QPE")
print("-" * 76)
for n_sys, t in [(2, 8), (3, 8), (4, 8)]:
    M = tfim_matrix(n_sys, h_field)
    w, v = np.linalg.eigh(M)
    c, tau = phase_window(w)
    U = unitary_from_h(M - c * np.eye(2 ** n_sys), tau)
    print(f"\n  n = {n_sys} system qubits, t = {t} counting qubits, "
          f"tau = {tau:.6f}, shift c = {c:.6f}")
    print(f"  {'level':>6}{'E exact':>12}{'phi exact':>12}{'best k':>8}"
          f"{'phi QPE':>11}{'E from QPE':>13}{'|dE|':>10}{'P(best)':>10}")
    for k_lev in range(min(4, 2 ** n_sys)):
        phi_ex = (-(w[k_lev] - c) * tau / (2 * np.pi)) % 1.0
        p = counting_probs(qpe_state(U, v[:, k_lev].astype(complex), t), t)
        k = int(np.argmax(p))
        E_qpe = c - (k / 2 ** t) * 2 * np.pi / tau
        print(f"  {k_lev:>6}{w[k_lev].real:>12.6f}{phi_ex:>12.6f}{k:>8}"
              f"{k/2**t:>11.6f}{E_qpe:>13.6f}"
              f"{abs(E_qpe - w[k_lev].real):>10.2e}{p[k]:>10.4f}")

print("\nPrecision scaling: ground state of the n = 3 chain")
print("-" * 76)
n_sys = 3
M = tfim_matrix(n_sys, h_field)
w, v = np.linalg.eigh(M)
c, tau = phase_window(w)
U = unitary_from_h(M - c * np.eye(2 ** n_sys), tau)
psi0 = v[:, 0].astype(complex)
print(f"  exact E_0 = {w[0].real:.9f},  exact phi_0 = "
      f"{(-(w[0]-c)*tau/(2*np.pi)) % 1.0:.9f}")
print(f"  {'t':>3}{'ctrl-U calls':>14}{'phi QPE':>12}{'|d phi|':>11}"
      f"{'2^-(t+1)':>11}{'|dE| (energy)':>15}{'P(best)':>10}")
errs = []
ts = list(range(4, 15))
for t in ts:
    p = counting_probs(qpe_state(U, psi0, t), t)
    k = int(np.argmax(p))
    dphi = abs(k / 2 ** t - (-(w[0] - c) * tau / (2 * np.pi)) % 1.0)
    dE = abs((c - (k / 2 ** t) * 2 * np.pi / tau) - w[0].real)
    errs.append(dE)
    print(f"  {t:>3}{2**t - 1:>14d}{k/2**t:>12.6f}{dphi:>11.2e}"
          f"{2.0**-(t+1):>11.2e}{dE:>15.2e}{p[k]:>10.4f}")

sl = np.polyfit(np.array(ts, float), np.log2(np.array(errs)), 1)[0]
print(f"\n  log2(energy error) versus t: slope = {sl:.3f} "
      f"(exact halving would be -1)")
print("  Cost is 2^t - 1 controlled applications of U, so precision eps "
      "costs O(1/eps) depth.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t, style in [(5, "-o"), (8, "--s")]:
    p = counting_probs(qpe_state(U, psi0, t), t)
    ax[0].plot(np.arange(2 ** t) / 2 ** t, p, style, ms=3, lw=1,
               label=f"t = {t}")
ax[0].axvline((-(w[0] - c) * tau / (2 * np.pi)) % 1.0, color="k", ls=":",
              lw=1, label="exact $\\phi_0$")
ax[0].set_xlabel("$k/2^t$"); ax[0].set_ylabel("probability")
ax[0].set_title("QPE output, ground state of the 3-site chain")
ax[0].legend(fontsize=8)

ax[1].semilogy(ts, errs, "o-", color="tab:red", label="measured")
ax[1].semilogy(ts, [2.0 ** -(t + 1) * 2 * np.pi / tau for t in ts], "k--",
               label="$2^{-(t+1)} \\cdot 2\\pi/\\tau$")
ax[1].set_xlabel("counting qubits $t$"); ax[1].set_ylabel("energy error")
ax[1].set_title("One extra qubit, one more bit")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
Transverse-field Ising chain, h = 0.5: eigenphases from QPE
----------------------------------------------------------------------------

  n = 2 system qubits, t = 8 counting qubits, tau = 1.840623, shift c = 1.555635
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -1.414214    0.870000     223   0.871094    -1.417947  3.73e-03    0.7673
       1   -1.000000    0.748659     192   0.750000    -1.004579  4.58e-03    0.6675
       2    1.000000    0.162770      42   0.164062     0.995588  4.41e-03    0.6879
       3    1.414214    0.041429      11   0.042969     1.408956  5.26e-03    0.5825

  n = 3 system qubits, t = 8 counting qubits, tau = 1.083148, shift c = 2.643533
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -2.403212    0.870000     223   0.871094    -2.409557  6.34e-03    0.7673
       1   -2.209275    0.836568     214   0.835938    -2.205620  3.66e-03    0.9173
       2   -0.500000    0.541908     139   0.542969    -0.506151  6.15e-03    0.7799
       3   -0.306063    0.508476     130   0.507812    -0.302214  3.85e-03    0.9086

  n = 4 system qubits, t = 8 counting qubits, tau = 0.759559, shift c = 3.769737
   level     E exact   phi exact  best k    phi QPE   E from QPE      |dE|   P(best)
       0   -3.427034    0.870000     223   0.871094    -3.436082  9.05e-03    0.7673
       1   -3.332247    0.858541     220   0.859375    -3.339142  6.90e-03    0.8589
       2   -1.826838    0.676556     173   0.675781    -1.820427  6.41e-03    0.8770
       3   -1.732051    0.665098     170   0.664062    -1.723488  8.56e-03    0.7893

Precision scaling: ground state of the n = 3 chain
----------------------------------------------------------------------------
  exact E_0 = -2.403211926,  exact phi_0 = 0.870000000
    t  ctrl-U calls     phi QPE    |d phi|   2^-(t+1)  |dE| (energy)   P(best)
    4            15    0.875000   5.00e-03   3.12e-02       2.90e-02    0.9792
    5            31    0.875000   5.00e-03   1.56e-02       2.90e-02    0.9186
    6            63    0.875000   5.00e-03   7.81e-03       2.90e-02    0.7054
    7           127    0.867188   2.81e-03   3.91e-03       1.63e-02    0.6401
    8           255    0.871094   1.09e-03   1.95e-03       6.34e-03    0.7673
    9           511    0.869141   8.59e-04   9.77e-04       4.99e-03    0.5050
   10          1023    0.870117   1.17e-04   4.88e-04       6.80e-04    0.9535
   11          2047    0.870117   1.17e-04   2.44e-04       6.80e-04    0.8243
   12          4095    0.870117   1.17e-04   1.22e-04       6.80e-04    0.4380
   13          8191    0.869995   4.88e-06   6.10e-05       2.83e-05    0.9947
   14         16383    0.869995   4.88e-06   3.05e-05       2.83e-05    0.9791

  log2(energy error) versus t: slope = -1.079 (exact halving would be -1)
  Cost is 2^t - 1 controlled applications of U, so precision eps costs O(1/eps) depth.
```

**注目すべき点。** どの鎖のどの固有値も $t = 8$ でエネルギー誤差 $10^{-3}$ 台まで回復されており、これはスペクトル幅の $2^{-9}$ — レジスタ長が許す分解能そのもので、それ以上ではありません。回復されたエネルギーはフィットも平均もしていません。測定された整数1個から算術1段です。

精度の表は本章の中心となるスケーリング結果です。位相誤差はどの $t$ でも $2^{-(t+1)}$ を下回り、エネルギー誤差は計数量子ビット1本ごとに半分になり、$\log_2(\text{誤差})$ の $t$ に対する傾きは $-1.079$ です。$-1$ からの超過はビンの離散性であって系統効果ではありません。第2列の数値がその代価です。$2.8 \times 10^{-5}$ のエネルギー誤差に達するには $2^{14}-1 = 16383$ 回の $U$ の制御適用が必要です（$t = 13$ 行の $6.10 \times 10^{-5}$ はその行の $2^{-(t+1)}$ という限界値であって、達成された誤差ではありません）。精度は回路深さに比例して買われ、これが第4章で具体化しなければならない交換です。実際の問題では $U$ の1回の適用そのものが $e^{-iH\tau}$ の高価なシミュレーションだからです。

成功確率の列は傾向なく $0.44$ から $0.99$ をさまよいます。これは例4と同じ教訓を別の角度から見たものです。ピークの高さは $\varphi$ がビン境界に対してどこに来るかに依り、ビンが何個あるかには決して依りません。

* * *

## 2.3 反復的位相推定

### $t$ 個の代わりに補助1個

教科書的回路は、$2^t - 1$ 回の $U$ の作用が走る間、計数量子ビット $t$ 個をコヒーレントに保持する必要があります。ハードウェアでは量子ビットのほうが希少な資源であり、回路中測定が使えるので、両者の間に交換レートが立ちます。計数レジスタは最後に測定されてそれ以降使われないこと、そして逆QFTの非Clifford成分は*他の*測定結果で決まる角度の制御位相回転だけであること、この2つに注意してください。どちらも同じ方向を指しています。量子的な制御を古典的な制御に置き換えるのです。

**反復的位相推定**がこれを行います。$\varphi = 0.b_1b_2\ldots b_t$ と2進で書き、最下位ビットから上へ抽出します。ラウンド $k$ では $b_{k+1},\ldots,b_t$ が既知として、

  1. 補助にHadamardを掛けます。
  2. それで制御して $U^{2^{k-1}}$ を作用させます。これで $\lvert 1 \rangle$ の枝に位相 $2\pi \left(2^{k-1}\varphi\right) = 2\pi\left(\text{整数} + 0.b_k b_{k+1}\ldots b_t\right)$ が書き込まれます。
  3. 補助を $-2\pi \times 0.0b_{k+1}\ldots b_t$ だけ回転させ、既知のビットをすべて打ち消します。
  4. Hadamardを掛けて測定します。残る相対位相は $\pi b_k$ なので、結果は確実に $b_k$ です。

段3のフィードバックは、逆QFTの制御回転を古典的に実行したものです。$\varphi$ がちょうど $t$ ビットで終わるならどのラウンドも決定論的です。そうでなければ段3が知りえない裾 $0.b_{k+1}\ldots$ が各ラウンドを偏らせ、1ビットの誤りはそれより上のビットすべてを汚します。対処は各ラウンドを繰り返して多数決を取ることで、$U$ の作用回数は増えますが量子ビットは増えません。

交換レートを率直に述べます。反復的QPEは $t + n_{\text{sys}}$ 個ではなく $1 + n_{\text{sys}}$ 個の量子ビットを使い、最良の場合は同じ $2^t - 1$ 回の $U$ の作用、実務ではその小さな倍数を要し、コヒーレンス時間の内側で古典フィードバック付きの回路中測定を $t$ ラウンド必要とします。また系レジスタが $t$ ラウンドすべてを生き延びる必要があります。厳密な固有ベクトルなら $U$ がそれを動かさないので自動的に満たされ、これがこの変種が固有値問題で実務的である理由です。

### Code Example 6: 反復的位相推定

```python
"""第2章 Code Example 6: 補助量子ビット1個による反復的位相推定です。
Code Example 5 の続き（同一セッション）です。"""

import numpy as np


def iqpe_round(U, sys_state, power, feedback):
    """1ラウンド分です。補助＋系、制御 U^power、Rzによるフィードバック、読み出し。

    P(補助 = 1) を返します。補助はqubit 0で、残りが系です。sys_state は U の
    固有ベクトルなので変化せずに戻り、だからこそ1個の補助量子ビットで各
    ラウンドを次々に実行できます。
    """
    n_sys = int(np.log2(sys_state.size))
    n = 1 + n_sys
    state = np.kron(ket('0'), sys_state)
    state = apply_gate(state, H, [0], n)
    Up = np.linalg.matrix_power(U, power)
    state = apply_gate(state, controlled(Up), list(range(n)), n)
    state = apply_gate(state, rz(-2 * np.pi * feedback), [0], n)
    state = apply_gate(state, H, [0], n)
    p = probs(state)
    return float(p[p.size // 2:].sum())


def iqpe(U, sys_state, t, rng, reps=1):
    """反復的QPEです。最下位ビットから上へ t ラウンド進みます。

    bits[j] は phi の (j+1) 桁目なので phi = sum bits[j] 2^-(j+1) です。
    ラウンド k では既知の桁を差し引いた上で bits[k-1] を測定します。
    各ラウンドは `reps` 回繰り返し、多数決で決めます。
    """
    bits = [0] * t
    for k in range(t, 0, -1):
        feedback = sum(bits[j] / 2.0 ** (j - k + 2) for j in range(k, t))
        p1 = iqpe_round(U, sys_state, 2 ** (k - 1), feedback)
        votes = sum(1 for _ in range(reps) if rng.random() < p1)
        bits[k - 1] = 1 if 2 * votes > reps else 0
    return sum(b / 2.0 ** (j + 1) for j, b in enumerate(bits)), bits


print("A dyadic phase is returned bit by bit, deterministically")
print("-" * 74)
rng = np.random.default_rng(2024)
for phi in [0.375, 0.8125, 0.65625]:
    t = 5
    est, bits = iqpe(phase_gate(phi), ket('1'), t, rng)
    print(f"  phi = {phi:.5f} = 0.{''.join(str(int(b)) for b in bits)} "
          f"(binary)   estimate {est:.5f}   error {abs(est-phi):.1e}")
print("  every round had P(1) equal to 0 or 1, so no sampling was involved:")
for phi in [0.375, 0.8125]:
    exact = [int(phi * 2 ** (j + 1)) % 2 for j in range(5)]
    ps = []
    for k in range(5, 0, -1):
        fb = sum(exact[j] / 2.0 ** (j - k + 2) for j in range(k, 5))
        ps.append(iqpe_round(phase_gate(phi), ket('1'), 2 ** (k - 1), fb))
    print(f"    phi = {phi:.5f}: P(1) per round = "
          + "  ".join(f"{p:.3f}" for p in ps))

print("\nA non-dyadic phase needs repetition: phi = 1/3, t = 6")
print("-" * 74)
phi = 1.0 / 3.0
t = 6
n_trials = 2000
print(f"  {'reps/round':>11}{'U calls/trial':>15}{'mean |error|':>14}"
      f"{'P(|error| < 2^-t)':>20}")
for reps in [1, 3, 5, 9, 25]:
    rng = np.random.default_rng(7)
    errs = np.array([abs(iqpe(phase_gate(phi), ket('1'), t, rng, reps)[0] - phi)
                     for _ in range(n_trials)])
    errs = np.minimum(errs, 1.0 - errs)
    calls = reps * (2 ** t - 1)
    print(f"  {reps:>11}{calls:>15}{errs.mean():>14.5f}"
          f"{float((errs < 2.0**-t).mean()):>20.4f}")
print(f"  textbook QPE with t = {t}: {2**t - 1} calls, "
      f"{t} counting qubits, P(best bin) = "
      f"{counting_probs(qpe_state(phase_gate(phi), ket('1'), t), t).max():.4f}")

print("\nSame comparison on the 3-site Ising ground state, t = 8")
print("-" * 74)
M = tfim_matrix(3, h_field)
w, v = np.linalg.eigh(M)
c, tau = phase_window(w)
U = unitary_from_h(M - c * np.eye(8), tau)
psi0 = v[:, 0].astype(complex)
phi_ex = (-(w[0] - c) * tau / (2 * np.pi)) % 1.0
t = 8
print(f"  exact phi_0 = {phi_ex:.6f},  exact E_0 = {w[0].real:.6f}")
print(f"  {'method':<28}{'qubits':>8}{'U calls':>10}{'phi':>12}"
      f"{'E':>13}{'|dE|':>11}")
p = counting_probs(qpe_state(U, psi0, t), t)
k = int(np.argmax(p))
E = c - (k / 2 ** t) * 2 * np.pi / tau
print(f"  {'textbook QPE':<28}{t+3:>8}{2**t-1:>10}{k/2**t:>12.6f}"
      f"{E:>13.6f}{abs(E-w[0].real):>11.2e}")
for reps in [1, 9]:
    rng = np.random.default_rng(31)
    est, _ = iqpe(U, psi0, t, rng, reps)
    E = c - est * 2 * np.pi / tau
    print(f"  {f'iterative QPE, reps = {reps}':<28}{1+3:>8}"
          f"{reps*(2**t-1):>10}{est:>12.6f}{E:>13.6f}"
          f"{abs(E-w[0].real):>11.2e}")
print("  The counting register is gone: t qubits become one, at the price of "
      "t rounds\n  of measurement and classical feedback.")
```

```text
A dyadic phase is returned bit by bit, deterministically
--------------------------------------------------------------------------
  phi = 0.37500 = 0.01100 (binary)   estimate 0.37500   error 0.0e+00
  phi = 0.81250 = 0.11010 (binary)   estimate 0.81250   error 0.0e+00
  phi = 0.65625 = 0.10101 (binary)   estimate 0.65625   error 0.0e+00
  every round had P(1) equal to 0 or 1, so no sampling was involved:
    phi = 0.37500: P(1) per round = 0.000  0.000  1.000  1.000  0.000
    phi = 0.81250: P(1) per round = 0.000  1.000  0.000  1.000  1.000

A non-dyadic phase needs repetition: phi = 1/3, t = 6
--------------------------------------------------------------------------
   reps/round  U calls/trial  mean |error|   P(|error| < 2^-t)
            1             63       0.01367              0.8510
            3            189       0.00689              0.9665
            5            315       0.00593              0.9900
            9            567       0.00551              0.9970
           25           1575       0.00522              1.0000
  textbook QPE with t = 6: 63 calls, 6 counting qubits, P(best bin) = 0.6840

Same comparison on the 3-site Ising ground state, t = 8
--------------------------------------------------------------------------
  exact phi_0 = 0.870000,  exact E_0 = -2.403212
  method                        qubits   U calls         phi            E       |dE|
  textbook QPE                      11       255    0.871094    -2.409557   6.34e-03
  iterative QPE, reps = 1            4       255    0.867188    -2.386897   1.63e-02
  iterative QPE, reps = 9            4      2295    0.871094    -2.409557   6.34e-03
  The counting register is gone: t qubits become one, at the price of t rounds
  of measurement and classical feedback.
```

**注目すべき点。** 2進有限桁の位相ではラウンドごとの確率が厳密に0か1です。アルゴリズムは2進展開を読み上げるだけで標本抽出は関与せず、推定は厳密です。この構成が設計された場合そのものです。読み方の注意を1つ。ラウンドごとの一覧はラウンドの実行順、すなわち最下位ビットから印字されます。すぐ上の行の2進展開とは逆順で、$0.01100$ は $0, 0, 1, 1, 0$ と現れます。

$\varphi = 1/3$ では中央の表が交換条件を明示します。ラウンドあたり1回の繰り返しで正しい $t$ ビット答が $85\%$、9回で $99.7\%$、その代価は $U$ の作用回数9倍です。25回では2000試行で確実になり、平均誤差は $0.00522$ に収束します。これは打ち切り誤差 $\lvert 1/3 - 21/64 \rvert$ であってアルゴリズムの失敗ではありません。同じ $t$ の教科書的QPEは作用63回で計数量子ビット6本を要しますから、比較はこうです。深さは同じ、量子ビットは6本節約、同等の信頼性のための総作用回数は $9$ 倍。

最後のブロックは両方の変種を実際の固有ベクトルに走らせたもので、算術がその要点です。11量子ビットが4量子ビットになります。単一ショットの反復実行は教科書的答から1ビンずれ、ラウンドあたり9回繰り返すと厳密に同じビン、つまり厳密に同じエネルギーに落ちます。この交換に価値があるかはハードウェアの問題 — 回路中測定のコストと量子ビットのコストの比 — であり、どちらの答も実際に使われています。

* * *

## 2.4 位相推定は何のためにあるか

### 固有値こそが目的

材料研究者が量子コンピュータに求めるものは、ほとんどすべて固有値です。基底状態エネルギーは電子ハミルトニアンの最小固有値、バンド構造は固有値の族、反応障壁はその2つの差、振動スペクトル、磁気交換定数、光学ギャップも同様です。[量子コンピューティング入門 第4章](<../quantum-computing-introduction/chapter-4.html>)がこれを詳しく設定し、第二量子化とJordan-Wigner変換で電子ハミルトニアンを量子ビットへ写し、固有値を取り出す2つの方法を比較しました。その比較こそ本章が存在する理由であり、いま完成させられます。

**変分量子固有値ソルバー**はパラメータ付き試行状態を用意し、Pauli項の標本抽出で $\langle H \rangle$ を測り、古典最適化器に最小化させます。回路は浅く、だからノイズのある実機で走ります。代価は2箇所です。答はansatzが厳密でない限り誤差が未知の変分上限であり、精度 $\varepsilon$ に達する測定コストは $1/\varepsilon^2$ でスケールします。平均によって期待値を推定しているからです。

**位相推定**は種類の違うことをします。何も平均しません。固有値を桁ごとにレジスタに書き込んで読みます。精度コストは $1/\varepsilon^2$ ではなく $1/\varepsilon$ — Heisenbergスケーリング — であり、答は与えたハミルトニアンの固有値そのもので変分ギャップはありません。代価は深さです。$e^{-iH\tau}$ のコヒーレントな作用が $\Theta(1/\varepsilon)$ 回必要で、これは誤り訂正なしの装置の遥か彼方であり、誤り耐性機を作る最強の論拠です。

| | VQE | 位相推定 |
| --- | --- | --- |
| 返すもの | 変分上限 $\ge E_0$ | $H$ の固有値 |
| ansatzが不適切なときの誤差 | 未知、片側 | なし — ただし違う固有値かもしれない |
| 精度のコスト | $O(1/\varepsilon^2)$ 回の測定 | $O(1/\varepsilon)$ 回の $U$ の作用 |
| 回路深さ | 浅く固定 | $\Theta(1/\varepsilon)$、コヒーレント |
| 誤り訂正が必要か | 不要 | 必要 |
| 良い試行状態が必要か | 精度のために必要 | そもそも成功するために必要 |
| 励起状態 | 困難。拘束や折り畳みが必要 | 無料 — 他のピークがそれ |

最後の2行はふつう飛ばされます。位相推定は励起状態を無料で得ます。同じ分布の他のピークがそれだからで、これは任意の変分法に対する真の構造的優位です。そして両方のアルゴリズムが良い試行状態を必要としますが、理由が違います。VQEはそれが*正確*である必要があり、位相推定は*重なっている*必要があります。

### 重なりの問題を率直に

位相推定に試行状態 $\lvert \psi \rangle = \sum_l c_l \lvert u_l \rangle$ を与えると、例4が既に示したとおり $E_l$ を確率 $\lvert c_l \rvert^2$ で返します。したがって基底状態エネルギーを知るには $p_0 = \lvert \langle \psi_0 \vert \psi \rangle \rvert^2$ が小さすぎてはならず、必要な繰り返し回数は $1/p_0$ でスケールします。小さな分子ならHartree-Fock行列式で $p_0$ は1に近く、これは問題になりません。この事業全体を動機づける強相関系ではまさにそこが難所です。基底状態は指数的に多い行列式の重ね合わせであり、$p_0$ は系のサイズとともに指数的に落ちえます。一般的な解決は知られていません。断熱的状態準備、量子選択配置間相互作用、収束したVQE状態を位相推定の入力に使うこと、が標準的な対応であり、そのいずれもサブルーチンではなく研究課題です。

これがこのアルゴリズムの誠実な境界であり、その位置を正確に述べる価値があります。位相推定は電子構造を「解く」のではありません。固有値を*計算する*問題を、まともな重なりを持つ状態を*準備する*問題に変換します。2番目の問題は少なくとも時には易しく1番目は決して易しくないので、これは本物で相当な還元ですが、解決と同じものではありません。

### Code Example 7: 電子構造計算の手法としての位相推定

具体版として、入門コース第3章が変分的に解いた2量子ビットのH$_2$ハミルトニアンを使います。同じ行列、同じ参照エネルギー、まったく違うアルゴリズムです。

```python
"""第2章 Code Example 7: 電子構造計算の手法としての位相推定です。
Code Example 6 の続き（同一セッション）です。"""

import numpy as np
import matplotlib.pyplot as plt

# 入門コース第3章と同じ2量子ビットH2ハミルトニアンです。
# STO-3G、R = 0.735 A、frozen coreによる2量子ビット還元です。
H2_TERMS = {'II': 0.252992, 'ZI': 0.344368, 'IZ': -0.451507,
            'ZZ': 0.574116, 'YY': 0.090466, 'XX': 0.090466}


def pauli_matrix(pauli):
    M = np.array([[1.0 + 0j]])
    for ch in pauli:
        M = np.kron(M, PAULI[ch])
    return M


Hm = sum(c * pauli_matrix(p) for p, c in H2_TERMS.items())
w, v = np.linalg.eigh(Hm)
c_shift, tau = phase_window(w)
U = unitary_from_h(Hm - c_shift * np.eye(4), tau)
phi_exact = [(-(e - c_shift) * tau / (2 * np.pi)) % 1.0 for e in w.real]

print("H2 at R = 0.735 A, STO-3G, two-qubit reduction")
print("-" * 72)
print(f"  exact spectrum (Ha): " + "  ".join(f"{e:+.6f}" for e in w.real))
print(f"  tau = {tau:.6f}, constant shift c = {c_shift:.6f} Ha")
print(f"  eigenphases:         " + "  ".join(f"{p:.6f}" for p in phi_exact))
print(f"  ground state = " + "  ".join(
    f"{v[i,0].real:+.4f}|{b}>" for i, b in enumerate(['00', '01', '10', '11'])
    if abs(v[i, 0]) > 1e-8))

hf = ket('10')          # Hartree-Fock: sigma_g 軌道が二重占有された状態です
p0 = abs(np.vdot(v[:, 0], hf)) ** 2
print(f"  Hartree-Fock overlap |<HF|psi_0>|^2 = {p0:.6f}")

print("\nQPE from the Hartree-Fock state: energy versus counting qubits")
print("-" * 72)
print(f"  {'t':>3}{'ctrl-U calls':>14}{'best k':>8}{'E (Ha)':>13}"
      f"{'error (Ha)':>13}{'P(peak)':>10}{'chem. acc.':>12}")
for t in range(4, 13):
    p = counting_probs(qpe_state(U, hf, t), t)
    k = int(np.argmax(p))
    E = c_shift - (k / 2 ** t) * 2 * np.pi / tau
    err = E - w[0].real
    print(f"  {t:>3}{2**t-1:>14d}{k:>8}{E:>13.6f}{err:>13.2e}{p[k]:>10.4f}"
          f"{'yes' if abs(err) < 1.6e-3 else 'no':>12}")
print("  chemical accuracy is 1 kcal/mol = 1.6 mHa")

print("\nThe overlap is the whole story: four trial states, t = 9")
print("-" * 72)
t = 9
trials = [("Hartree-Fock |10>", ket('10')),
          ("doubly excited |01>", ket('01')),
          ("equal mixture", (ket('10') + ket('01')) / np.sqrt(2)),
          ("exact first excited", v[:, 1].astype(complex))]
k_ground = int(round(phi_exact[0] * 2 ** t))
for label, psi in trials:
    p = counting_probs(qpe_state(U, psi, t), t)
    overlaps = [abs(np.vdot(v[:, j], psi)) ** 2 for j in range(4)]
    near = p[max(0, k_ground - 2):k_ground + 3].sum()
    k = int(np.argmax(p))
    E = c_shift - (k / 2 ** t) * 2 * np.pi / tau
    print(f"  {label:<22} |<psi_0|trial>|^2 = {overlaps[0]:.4f}"
          f"   peak E = {E:+.6f} Ha   P(within 2 bins of E_0) = {near:.4f}")
print("  A trial state with no ground-state amplitude returns a different "
      "eigenvalue,\n  correctly and confidently. QPE does not know which "
      "eigenvalue you wanted.")

print("\nWhat the depth costs, extrapolated")
print("-" * 72)
print(f"  {'target error (Ha)':>19}{'t needed':>10}{'ctrl-U calls':>15}"
      f"{'x100 Trotter steps':>21}")
for eps in [1e-2, 1.6e-3, 1e-4, 1e-6]:
    t_need = int(np.ceil(np.log2(2 * np.pi / tau / eps)))
    calls = 2 ** t_need - 1
    print(f"  {eps:>19.1e}{t_need:>10}{calls:>15d}"
          f"{calls * 100:>21d}")
print("  The last column assumes a merely nominal 100 Trotter steps per "
      "controlled U;\n  Chapter 4 replaces that guess with a real "
      "block-encoding cost.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t_, style in [(6, "-o"), (9, "-")]:
    p = counting_probs(qpe_state(U, hf, t_), t_)
    E_axis = c_shift - (np.arange(2 ** t_) / 2 ** t_) * 2 * np.pi / tau
    ax[0].plot(E_axis, p, style, ms=3, lw=1, label=f"t = {t_}")
for e in w.real:
    ax[0].axvline(e, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("energy (Ha)"); ax[0].set_ylabel("probability")
ax[0].set_title("QPE spectrum of H$_2$ from the HF state")
ax[0].legend(fontsize=8)

t_ = 9
for label, psi in [("HF |10>", ket('10')), ("|01>", ket('01'))]:
    p = counting_probs(qpe_state(U, psi, t_), t_)
    E_axis = c_shift - (np.arange(2 ** t_) / 2 ** t_) * 2 * np.pi / tau
    ax[1].plot(E_axis, p, lw=1, label=label)
for e in w.real:
    ax[1].axvline(e, color="k", ls=":", lw=0.8)
ax[1].set_xlabel("energy (Ha)"); ax[1].set_ylabel("probability")
ax[1].set_title("The trial state selects the eigenvalue")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
H2 at R = 0.735 A, STO-3G, two-qubit reduction
------------------------------------------------------------------------
  exact spectrum (Ha): -1.137306  +0.495058  +0.719969  +0.934247
  tau = 2.513123, constant shift c = 1.037825 Ha
  eigenphases:         0.870000  0.217094  0.127135  0.041429
  ground state = -0.1115|01>  +0.9938|10>
  Hartree-Fock overlap |<HF|psi_0>|^2 = 0.987560

QPE from the Hartree-Fock state: energy versus counting qubits
------------------------------------------------------------------------
    t  ctrl-U calls  best k       E (Ha)   error (Ha)   P(peak)  chem. acc.
    4            15      14    -1.149807    -1.25e-02    0.9671          no
    5            31      28    -1.149807    -1.25e-02    0.9072          no
    6            63      56    -1.149807    -1.25e-02    0.6967          no
    7           127     111    -1.130275     7.03e-03    0.6321          no
    8           255     223    -1.140041    -2.73e-03    0.7577          no
    9           511     445    -1.135158     2.15e-03    0.4987          no
   10          1023     891    -1.137599    -2.93e-04    0.9417         yes
   11          2047    1782    -1.137599    -2.93e-04    0.8140         yes
   12          4095    3564    -1.137599    -2.93e-04    0.4326         yes
  chemical accuracy is 1 kcal/mol = 1.6 mHa

The overlap is the whole story: four trial states, t = 9
------------------------------------------------------------------------
  Hartree-Fock |10>      |<psi_0|trial>|^2 = 0.9876   peak E = -1.135158 Ha   P(within 2 bins of E_0) = 0.9090
  doubly excited |01>    |<psi_0|trial>|^2 = 0.0124   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.0115
  equal mixture          |<psi_0|trial>|^2 = 0.3892   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.3582
  exact first excited    |<psi_0|trial>|^2 = 0.0000   peak E = +0.495800 Ha   P(within 2 bins of E_0) = 0.0000
  A trial state with no ground-state amplitude returns a different eigenvalue,
  correctly and confidently. QPE does not know which eigenvalue you wanted.

What the depth costs, extrapolated
------------------------------------------------------------------------
    target error (Ha)  t needed   ctrl-U calls   x100 Trotter steps
              1.0e-02         8            255                25500
              1.6e-03        11           2047               204700
              1.0e-04        15          32767              3276700
              1.0e-06        22        4194303            419430300
  The last column assumes a merely nominal 100 Trotter steps per controlled U;
  Chapter 4 replaces that guess with a real block-encoding cost.
```

**注目すべき点。** このハミルトニアンの厳密な基底状態エネルギーは $-1.137306$ Ha で、位相推定は $t = 10$ で化学精度 — 計算された反応エネルギーが化学的に意味を持つ閾値である $1.6$ mHa — に達し、$e^{-iH\tau}$ の制御適用1023回を使います。何かを最適化してではありません。答は測定された整数 $k$ 1個から $c - (k/2^t)\cdot 2\pi/\tau$ です。同じハミルトニアンに対する入門コースのVQEと比べてください。あちらは $10^{-15}$ Ha に達しましたが、それはansatzがたまたま厳密でシミュレーションが無ノイズだったからで、実機で $1.6$ mHa なら1回のエネルギー評価に $10^6$ 程度のショットを要したはずです。

試行状態の表は重なりの問題を4行で示します。Hartree-Fock行列式は基底状態と $98.8\%$ の重なりを持ち、QPEの結果の $90.9\%$ が $E_0$ から2ビン以内に落ちます。二重励起行列式は重なり $1.2\%$ で、ピークは*第一励起*エネルギー $+0.4958$ Ha に移ります。自信をもって、そして正しく報告されています。この入力が大部分含んでいる固有値は本当にそれだからです。等重率の混合は基底状態の重み $38.9\%$ でも励起状態にピークが立ちます。$61\%$ が $39\%$ に勝つからです。位相推定は聞かれた問いに答え、その問いは入力状態が決めます。

最後のブロックは深さを外挿します。このおもちゃで化学精度は $U$ の制御適用 $2 \times 10^3$ 回、$10^{-6}$ Ha は $4 \times 10^6$ 回です。$e^{-iH\tau}$ 1回の実装コストを掛けてください — 最後の列の名目上の係数100は仮置きで、第4章が本物のブロック符号化コストに置き換えます — 位相推定が近未来の提案ではなく誤り耐性の論拠である理由は、意見ではなく算術です。[入門コース第5章](<../quantum-computing-introduction/chapter-5.html>)は2サイトHubbardモデルに対する対応する誤り率要求を $p \lesssim 10^{-9}$ とし、これは誤り訂正なしのハードウェアより6桁下です。

* * *

## 演習

#### 演習1: 積形式を手で

$n = 3$、$j = 5 = 101_2$ とします。

  1. $\mathrm{QFT}\lvert 101 \rangle$ を8個の基底状態の和として書き、各振幅を $\frac{1}{\sqrt{8}}e^{i\theta_k}$ の形で $\theta_k$ を $2\pi$ 単位で与えてください。
  2. 同じ状態を積形式 $\bigotimes_l (\lvert 0 \rangle + e^{2\pi i j/2^l}\lvert 1 \rangle)/\sqrt{2}$ で書き、展開して1と一致することを確認してください。
  3. 積形式のどの1量子ビット位相が $-1$ になり、それは*回路*出力のどの量子ビットに対応しますか。最後のSWAPの前と後で答えてください。
  4. 8個の振幅のうち大きさが異なるものはいくつありますか。それは計算基底での出力測定について何を意味しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\theta_k/2\pi = 5k/8 \bmod 1\)、すなわち \(k = 0,\ldots,7\) に対して \(0, \tfrac{5}{8}, \tfrac{2}{8}, \tfrac{7}{8}, \tfrac{4}{8}, \tfrac{1}{8}, \tfrac{6}{8}, \tfrac{3}{8}\) です。8個の大きさはすべて \(1/\sqrt{8}\) です。</p>

<p><strong>2.</strong> 3つの因子が担う位相は \(e^{2\pi i \cdot 5/2} = e^{i\pi} = -1\)、\(e^{2\pi i \cdot 5/4} = e^{i\pi/2} = i\)、\(e^{2\pi i \cdot 5/8}\) です。積を展開すると \(2^{-3/2}\sum_{k_1k_2k_3} (-1)^{k_1} i^{k_2} e^{2\pi i \cdot 5k_3/8}\lvert k_1k_2k_3\rangle\) となり、\(k = 4k_1 + 2k_2 + k_3\) とおけば指数は法1で \(2\pi i \cdot 5k/8\) となって1と一致します。</p>

<p><strong>3.</strong> \(l = 1\)、位相 \(-1\) の因子です。積形式では<em>最初の</em>テンソル因子ですが、導出が示すようにそれは \(j_n\) 相当の位相、つまり<em>最下位</em>ビットぶんの位相を担います。回路はそれを最後の量子ビットに作り、最後のSWAPが先頭へ移します。SWAPを削って出力を名前替えするのは厳密に等価であり、多くの実装がそうしています。</p>

<p><strong>4.</strong> 1種類です。8個の大きさはすべて等しい。したがって \(\mathrm{QFT}\lvert j \rangle\) の計算基底測定は一様乱数であり、\(j\) についての情報を一切運びません。<em>単一の</em>基底状態の変換は最大限に無情報であり、構造をもつ状態 — 例3のような周期 — だけがピークのある出力を生みます。</p>

</details>

#### 演習2: レジスタ長を選ぶ

あるユニタリの固有位相を、確率 $0.99$ 以上で2進6桁正しく必要とします。

  1. $t = n + \lceil \log_2(2 + 1/(2\varepsilon)) \rceil$ から $t$ を求め、$U$ の作用回数を述べてください。
  2. 要求信頼度が $0.99$ から $0.999$ になると作用回数はどう変わりますか。要求精度が6ビットから7ビットになる場合はどうですか。
  3. 同僚が、1の $t$ で3回走らせて多数決を取れば $0.999$ に届くと提案しました。$U$ の総作用回数を見積り、2と比べてください。
  4. コヒーレンス時間が固定のハードウェアでは、2つの戦略のどちらが先に破綻しますか。理由も述べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\varepsilon = 0.01\) なので \(1/(2\varepsilon) = 50\)、\(\lceil \log_2 52 \rceil = 6\)、よって \(t = 12\)、\(U\) の作用は \(2^{12}-1 = 4095\) 回です。</p>

<p><strong>2.</strong> \(\varepsilon = 0.001\) では \(\lceil \log_2 502 \rceil = 9\) なので \(t = 15\)、32767 回 — 信頼度1桁ぶんで8倍です。\(\varepsilon = 0.01\) で7ビットなら \(t = 13\)、8191 回 — 精度1ビットで2倍です。得られるもの単位あたりで、信頼度は高価、精度は安価です。</p>

<p><strong>3.</strong> \(t = 12\) で3回は \(3 \times 4095 = 12285\) 回、深い1回の 32767 回に対して総仕事量で2.7倍安いです。最良ビンの確率が \(\ge 0.99\) の分布からの独立標本3個の多数決が失敗する確率は \(\approx 3 \times 10^{-4}\) なので、目標には届きます。</p>

<p><strong>4.</strong> 深い1回のほうが先に破綻します。最大コヒーレント深さが8倍大きく、デコヒーレンス時間が制限する資源はコヒーレント深さです。繰り返す浅い実行は1の深さしか要らず、時間的にいくらでも離せます。これは「深さを繰り返しと交換する」あらゆる方式の背後にある一般原理であり、\(\alpha\)-QPE系列も同様です。そしてその限界が最大深さ \(D\) 固定での限界 \(\varepsilon \gtrsim 1/(D\sqrt{N})\) であり、完全なHeisenbergスケーリングが本当に深さ \(\propto 1/\varepsilon\) を要する理由です。</p>

</details>

#### 演習3: ピーク高さの公式

$P_{\text{best}} \ge 4/\pi^2$ の主張とその文脈を検証してください。

  1. QFT前の状態 $2^{-t/2}\sum_k e^{2\pi i \varphi k}\lvert k \rangle$ から出発して、逆QFT後に結果 $m$ の振幅が $2^{-t}\sum_k e^{2\pi i (\varphi - m/2^t) k}$ であることを示し、等比級数を和してください。
  2. $\delta = \varphi - m/2^t$ と書いて $P(m) = \sin^2(2^t\pi\delta) / \left(2^{2t}\sin^2(\pi\delta)\right)$ を示してください。
  3. 最悪の $\delta = 2^{-t-1}$ で評価し、$t \to \infty$ を取ってください。
  4. $\varphi = 1/3$、$t = 8$、$m = 85$ で評価し、Code Example 4 が印字した $0.683922$ と比べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 逆QFTは \(\lvert k \rangle \mapsto 2^{-t/2}\sum_m e^{-2\pi i km/2^t}\lvert m \rangle\) なので \(\lvert m \rangle\) の振幅は \(2^{-t}\sum_k e^{2\pi i(\varphi - m/2^t)k}\) です。比 \(z = e^{2\pi i \delta}\) の等比和は \((z^{2^t}-1)/(z-1)\) です。</p>

<p><strong>2.</strong> \(\lvert (z^{2^t}-1)/(z-1) \rvert = \lvert \sin(2^t \pi \delta)/\sin(\pi\delta)\rvert\) で、\(2^{-t}\) の前因子をつけて2乗すれば所要の \(P(m)\) になります。</p>

<p><strong>3.</strong> \(\delta = 2^{-t-1}\) では \(\sin^2(2^t \pi \delta) = \sin^2(\pi/2) = 1\)、\(2^{2t}\sin^2(\pi 2^{-t-1}) \to 2^{2t}(\pi 2^{-t-1})^2 = \pi^2/4\) なので \(P \to 4/\pi^2 = 0.4053\) です。</p>

<p><strong>4.</strong> \(\delta = 1/3 - 85/256 = 1/768\) です。\(\sin^2(256\pi/768) = \sin^2(\pi/3) = 3/4\)、\(2^{16}\sin^2(\pi/768) = 65536 \times 1.673304\times10^{-5} = 1.096617\) なので \(P = 0.75/1.096617 = 0.683922\)。式は厳密で何も近似していないので、印字値と完全に一致します。（小角極限 \(2^{16}(\pi/768)^2 = 1.096623\) でも同じ5桁になるため、この違いは見落としやすいのです。）</p>

</details>

#### 演習4: 反復的QPEは優雅に失敗する

$\varphi$ に対して $t$ ラウンド、ラウンドあたり1回の反復的QPEを考えます。

  1. $\varphi$ が厳密に $t$ ビット展開を持つなら、各ラウンドが確率1でそのビットを測ることを論じてください。厳密性は議論のどこで使われますか。
  2. ビット $t$ より先の裾が $0.0\ldots0b_{t+1}b_{t+2}\ldots$ だとします。ラウンド $k$ が正しい $b_k$ を測る確率が、ある残差 $\eta_k$ に対して $\cos^2(\pi \eta_k)$ であることを示し、$\eta_k$ を同定してください。
  3. *下位*ビットの誤りがそれより上のビットを汚すのはなぜですか。教科書的QPEでは誤測定がその結果1つにしか影響しないのはなぜですか。
  4. Code Example 6 は $t = 6$、$\varphi = 1/3$ で、ラウンドあたり1回のとき平均誤差 $0.01367$、25回のとき $0.00522$ を報告しました。両方を説明してください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\varphi = 0.b_1\ldots b_t\) なら \(2^{k-1}\varphi = \text{整数} + 0.b_kb_{k+1}\ldots b_t\) であり、フィードバックは \(0.0b_{k+1}\ldots b_t\) を厳密に差し引くので相対位相 \(\pi b_k\) が残ります。Hadamardは \((\lvert 0\rangle + e^{i\pi b_k}\lvert 1 \rangle)/\sqrt{2}\) を確実に \(\lvert b_k \rangle\) に写します。厳密性は2度使われます。裾がビット \(t\) で終わること、そして既測定のビットが正しいことです。</p>

<p><strong>2.</strong> 残る相対位相は \(2\pi(0.b_k + \text{裾})\) から \(\pi b_k\) を引いたもの、すなわち \(2\pi \eta_k\) で、\(\eta_k = 2^{k-1}\varphi - (\text{既知ビット}) - b_k/2 \bmod 1\)、打ち消せなかった裾 \(0.00b_{t+1}b_{t+2}\ldots\) を \(2^{k-1}\) 倍したものです。Hadamard後の測定は \(b_k\) を確率 \(\cos^2(\pi\eta_k)\) で返します。</p>

<p><strong>3.</strong> ラウンド \(k\) のフィードバック角がラウンド \(k+1,\ldots,t\) で測ったビットから計算されるからです。そこが誤っていると打ち消しが誤った量だけずれ、以降のどのラウンドでも残留位相が変位して誤りが積み上がります。教科書的QPEにはフィードバックがなく、各ショットは固定分布の独立標本です。</p>

<p><strong>4.</strong> 25回繰り返せばラウンドごとの投票はほぼ常に正しいので、推定値は最良の6ビット打ち切り \(21/64 = 0.328125\) となり、\(\lvert 1/3 - 21/64\rvert = 0.005208\) です。印字された \(0.00522\) はその床に残留失敗率が乗ったものです。1回のときの追加の \(0.0085\) はビットが反転した実行の寄与です。下位ビットの誤りは上へ伝播するので、そうした実行は最下位1ビットどころではなく大きく外れます。だから平均誤差が床のわずかに上ではなく2.6倍になります。</p>

</details>

#### 演習5: 資源に関する主張を読む

セミナーの講演者がこう述べました。「本手法は100軌道の活性空間の基底状態エネルギーを、200論理量子ビットの位相推定で化学精度で計算します。」

  1. この文が省いている量を3つ挙げてください。いずれもコストを桁で変えます。
  2. 100軌道ハミルトニアンのスペクトル幅を100 Ha程度として、含意される $e^{-iH\tau}$ の作用回数を見積ってください。
  3. 講演者の試行状態は単一のHartree-Fock行列式で、系は強相関です。これは繰り返し回数に何をしますか。またそれをどう礼儀正しく尋ねますか。
  4. この主張をもっともらしいだけでなく検査可能にするには何が必要ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> (i) 回路深さ、同等に非Cliffordゲート総数。入門コース第4章が強調したとおり、200論理量子ビットは3つの資源のうち<em>最も安い</em>ものです。(ii) \(e^{-iH\tau}\) 1回の実装コスト。これはシミュレーション手法（Trotter次数、ブロック符号化、qubitization）とハミルトニアンの構造に完全に依存します。(iii) 試行状態と目的固有ベクトルの重なり \(p_0\)。実行時間全体に \(1/p_0\) が掛かります。</p>

<p><strong>2.</strong> 化学精度は \(1.6\times10^{-3}\) Ha です。スペクトル幅 \(\Lambda \sim 100\) Ha なら必要な位相分解能は \(\varepsilon_\varphi \sim 1.6\times10^{-3}/100 = 1.6\times10^{-5}\) なので \(t \approx \lceil \log_2(1/1.6\times10^{-5})\rceil = 16\) に詰めビットを足し、\(2^t - 1 \sim 10^5\) 回の \(e^{-iH\tau}\) です。その1回1回が手法によって \(10^3\)–\(10^6\) ゲートなので合計 \(10^8\)–\(10^{11}\)。量子化学の誤り耐性見積りが実際に占める範囲です。</p>

<p><strong>3.</strong> 強相関の基底状態のHartree-Fock重なりは小さくなりえ、一般に系サイズとともに縮みます。期待される繰り返し回数は \(1/p_0\) で、大きくなりえます。礼儀正しい質問は単に「あなたの試行状態の \(p_0\) はいくらで、活性空間の大きさに対してどうスケールしますか」です。完全な解析なら既に計算済みの、明確な数値の答がある質問です。</p>

<p><strong>4.</strong> 全回路のT数またはToffoli数、物理量子ビット数の背後にある仮定した物理誤り率と符号距離、\(p_0\) の値、そして同じパイプラインのより小さな実例を厳密対角化と照合したもの。最後のものが本コースが要求するものです。本章のすべての主張は同じハミルトニアンに対する <code>numpy.linalg.eigh</code> と照合されています。</p>

</details>

* * *

## まとめ

### 要点

**1\. QFTは見慣れた行列に対する短い回路である**

  * $\mathrm{QFT}\lvert j \rangle$ は量子ビット1本あたり位相1個の積状態に因数分解し、だから回路はHadamard $n$ 個、制御位相回転 $n(n-1)/2$ 個、SWAP $\lfloor n/2 \rfloor$ 個です。
  * $n = 7$ で密なDFT行列と $10^{-14}$ まで照合済み。最小の回転を落とせば実務で使われる $O(n\log n)$ の近似QFTになります。
  * 古典FFTとの比較は指数的高速化ではありません。入力が既に量子状態でなければならず、1回の実行は添字1個を返し、出力確率1個の推定に標本抽出では $O(1/\delta^2)$ ショットかかります。

**2\. QFTが実際に届けるのは周期である**

  * 公差 $r$ の等差数列に台を持つ状態は $2^n/r$ の倍数のピークに変換され、オフセットは完全に位相に格納されます。オフセットが違う分布は $10^{-50}$ まで一致します。
  * $r$ が $2^n$ を割るときピークは厳密で確率の $100\%$ がその上に乗り、割らないときは最良ビンに $79\%$ で古典後処理が必要になります。
  * その後処理が連分数であり、第3章の主題です。

**3\. 位相推定はQFTを逆に走らせたものである**

  * 制御べき乗 $U^{2^j}$ が $2^{-t/2}\sum_k e^{2\pi i \varphi k}\lvert k \rangle$ を計数レジスタに書き込み、逆QFTがそれを $\lvert 2^t\varphi \rangle$ として読みます。
  * コストは $U$ の作用 $2^t - 1$ 回なので精度 $\varepsilon$ は深さ $\Theta(1/\varepsilon)$。試した全 $t$ で位相誤差は $2^{-(t+1)}$ 未満、エネルギー誤差は量子ビット1本ごとに半減し、フィット傾き $-1.079$ でした。
  * 追加量子ビットが買うのは分解能で確信度ではありません。$\varphi = 1/3$ の $P_{\text{best}}$ は $t$ に依らず $0.6839$、床は $4/\pi^2$ です。確信度は $\lceil\log_2(2+1/(2\varepsilon))\rceil$ の詰めビットから来て、$\Pr[\lvert\tilde{\varphi}-\varphi\rvert < 2^{-n}] \ge 1-\varepsilon$ を買います。これは精度の保証であって、先頭 $n$ ビットについての約束ではありません。

**4\. 反復的QPEは量子ビットをラウンドと交換する**

  * 補助1個と古典フィードバック付き測定 $t$ ラウンドが、計数量子ビット $t$ 個と逆QFTを置き換えます。
  * 2進有限桁の位相では厳密。一般の位相では下位ビットの誤りが上へ伝播するのでラウンドを繰り返して多数決します — 1回で $85\%$、9回で $99.7\%$、代価は $U$ の作用9倍です。
  * 3サイト鎖では11量子ビットが4量子ビットになり、同一のエネルギーを返しました。

**5\. 応用は固有値、障害は重なり**

  * 電子構造は固有値問題なので、位相推定はVQEの誤り耐性時代の後継です。変分限界ではなく固有値、$1/\varepsilon^2$ ではなく $1/\varepsilon$、励起状態は無料、代価はコヒーレント深さ $\Theta(1/\varepsilon)$。
  * 2量子ビットH$_2$ハミルトニアンでは $t = 10$、$e^{-iH\tau}$ の制御適用1023回で化学精度に達しました。最適化器も平均もなしです。
  * 基底状態との重なりが $1.2\%$ の状態を与えると、自信をもって第一励起エネルギーを返します。繰り返し回数は $1/p_0$ でスケールし、強相関系では $p_0$ が未解決問題です。

**実務上の含意**

  * $t$ は要求精度*と*要求信頼度の両方から選んでください。値段の違う別々の買い物です。
  * 位相推定のコストを述べるときは必ず試行状態の重なりも報告してください。$p_0$ なしの資源見積りは不完全です。
  * 変分法と比べるときは $1/\varepsilon$ 対 $1/\varepsilon^2$、コヒーレント深さ対ショット数で比べてください。両者が違うのはその軸です。

### 次章へ

第3章はここで作った2つの回路 — 例2のQFTと例4・5の位相推定機構 — を、固有位相が整数 $N$ を法とする乗法的位数 $r$ に対する $s/r$ となるユニタリに向けます。結果がShorのアルゴリズムで、本コースで高速化が超多項式的かつ議論の余地がない唯一の場合です。第3章はシミュレータ上で15と21をエンドツーエンドで因数分解し、例3が不可避と示した連分数後処理も含めて実行し、そのうえで同じ回路が暗号規模で何を要するか、そしてそのコストに対する標準的な対応が既に何であるかを率直に述べます。

[← 第1章: 振幅増幅とGroverのアルゴリズム](<chapter-1.html>) [第3章: Shorのアルゴリズム →](<chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章の資源見積り — ゲート数、Trotterステップ数の仮置き、誤り率要求 — は明記した仮定から導いた桁レベルの教育用概算であり、測定値でも予測でもありません。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
