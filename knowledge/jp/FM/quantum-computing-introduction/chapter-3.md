---
title: "第3章: 変分量子固有値ソルバー（VQE）"
chapter_title: "第3章: 変分量子固有値ソルバー（VQE）"
subtitle: ⚛️ ansatz、Pauli測定、古典最適化、そしてH₂の基底状態
reading_time: 40-45分
difficulty: 上級
code_examples: 9
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-computing-introduction/chapter-3.html>) | Last sync: 2026-08-12

[基礎数理道場](<../index.html>) > [量子コンピューティング入門](<index.html>) > 第3章

本章は、これまでの道具立てが化学の計算を始める章です。最小基底の水素分子を取り上げ、その電子ハミルトニアンを2量子ビット上の6個のPauli文字列の和として書き、試行波動関数を用意する1パラメータ回路を構成し、その回路上でエネルギーを測定して、得られた数値を古典最適化器に渡します。これが**変分量子固有値ソルバー（VQE）**であり、2014年にPeruzzoらが提案し、いまなおノイズのある実機での量子化学を支配しているアルゴリズムです。

そして検証します。この計算は同じハミルトニアンを厳密に対角化できるほど小さいので、あらゆるVQEエネルギーに照合すべき参照値が存在します。しかも平衡点だけでなく解離曲線全体で照合します。3.5節での一致は \\(10^{-15}\\) ハートリー水準ですが、これは化学についての主張ではなくアルゴリズムについての主張です。厳密なansatz、ノイズのないシミュレーション、収束した最適化器がそろえば、VQEは与えられたハミルトニアンの厳密な基底状態を返します。実機で目にするずれのすべては、この3条件のいずれかを緩めたことから生じ、3.6節でそれぞれを検討します。

この計算が何を示し何を示さないかについて一言。2量子ビットは腕時計でもシミュレートできます。ここには古典計算の手に届かないものは何もありませんし、私たちが再現するSTO-3Gエネルギーは真の非相対論的なH₂基底状態から40ミリハートリー離れています。それは*基底関数系*が粗いからで、アルゴリズムのせいではありません。この演習が確立するのは、パイプライン全体 — 軌道積分から量子ビットハミルトニアン、回路、最適化器、エネルギーまで — が端から端まで正しいという点です。各段が独立な参照値と照合されているからです。最終的な標的が一切照合できない手法に向き合うには、これが唯一の誠実な方法です。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * 変分原理を述べ、それがノイズのある量子コンピュータを基底状態エネルギー計算に有用にする理由を説明できる
  * 化学に基づくansatzとハードウェア効率型ansatzを区別し、各々が探索する状態空間の領域の違いを定量化できる
  * Pauli指数関数 \\(\exp(-i\theta X_0 Y_1)\\) をCNOTと1個の \\(R_z\\) にコンパイルし、得られた回路が粒子数とスピンを保存することを検証できる
  * 分子の電子ハミルトニアンをPauli文字列の重み付き和として表し、`expval` でその期待値を評価できる
  * 可換なPauli項を測定設定にまとめ、目標精度に到達するためのショット数を見積れる
  * 勾配フリー法と勾配法でVQE最適化を実行し、2回の回路評価で厳密な解析的勾配を与えるparameter-shift則を導出できる
  * H₂の解離曲線を再現し、厳密対角化および公表されたSTO-3G参照値と照合できる
  * VQEのスケールアップを縛る制約が量子ビット数ではなくbarren plateauと測定コストであることを、数値を伴って説明できる

* * *

## 3.1 なぜ変分法なのか

### すべてを規定する制約

エネルギーを求める教科書的な量子アルゴリズムは存在しますが、それはVQEではありません。**量子位相推定（QPE、第4章）**は、制御時間発展を \\(O(2^m)\\) 回適用する回路によって \\(H\\) の固有値を \\(m\\) ビット精度で取り出します。計算量理論の意味では効率的であり、そして今日のところ完全に手が届きません。化学的に興味ある分子では数百万ゲートをコヒーレントに実行する必要があり、現在の装置は数百ゲートでコヒーレンスを失います。

VQEはその制約への応答です。1本の深い回路の代わりに**多数の浅い回路**を使い、難しい部分 — 探索 — を古典計算機に移します：

  1. 浅いパラメータ化回路で \\(\lvert \psi(\boldsymbol{\theta}) \rangle\\) を用意する。
  2. \\(E(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) \rvert H \lvert \psi(\boldsymbol{\theta}) \rangle\\) を測定する。
  3. 古典最適化器がより良い \\(\boldsymbol{\theta}\\) を提案する。
  4. 繰り返す。

回路の深さは要求精度ではなくansatzが決めます。だからVQEは現存するハードウェアで走ります。代償は回路の繰り返し回数で支払われ、3.6節が示すようにこの代償は厳しいものです。

### 変分原理

数学的な保証は、[量子力学入門](<../quantum-mechanics/index.html>)コースで既に出会った事実から来ます。基底状態エネルギー \\(E_0\\) をもつ任意のハミルトニアン \\(H\\) と任意の規格化状態 \\(\lvert \psi \rangle\\) について

\\[ \langle \psi \rvert H \lvert \psi \rangle \geq E_0 \\]

等号成立は \\(\lvert \psi \rangle\\) が基底状態のとき、かつそのときのみです。証明は1行です。\\(H\\) の固有基底で \\(\lvert \psi \rangle = \sum_k c_k \lvert \phi_k \rangle\\) と展開すると

\\[ \langle \psi \rvert H \lvert \psi \rangle = \sum_k \lvert c_k \rvert^2 E_k \geq E_0 \sum_k \lvert c_k \rvert^2 = E_0 \\]

2つの帰結が、これをノイズのある装置にとって正しい土台にしています。

**あらゆる答えが上界である。** 試行エネルギーは誤り得ますが、*低すぎる*ことはあり得ません。2つのansatzが \\(-1.1361\\) と \\(-1.1373\\) ハートリーを与えたなら、真の値を知らずとも後者が真値に近いと断言できます。誤差の符号が不明な手法では改善を検証できないことと比べてみてください。

**誤差は状態の誤差の2次である。** \\(\lvert \psi \rangle\\) が真の基底状態から励起状態方向に振幅 \\(\varepsilon\\) だけずれているとき、エネルギー誤差は \\(O(\varepsilon^2)\\) です。重なりが99%の試行状態はエネルギー誤差1%程度であって10%ではありません。この2次的な抑制こそ、変分法が1世紀にわたって電子構造理論を支配してきた理由であり、VQEが不完全な状態生成に寛容である理由です。

**しばしば見過ごされる注意点。** 上界 \\(E(\boldsymbol{\theta}) \geq E_0\\) が成り立つのは*純粋状態*の*厳密な*期待値についてです。実機ではデコヒーレンスによって状態が混合状態になり、読み出し誤差が推定量に偏りを与えるため、測定されたVQEエネルギーは真の基底状態を*下回る*ことが日常的に起こります。厳密値を下回るVQEエネルギーの報告を見たとき、それは成果ではなく未補正の偏りの診断です。

### VQEが機能するための条件

要件 | 意味 | 満たされないと何が起きるか
---|---|---
表現力のあるansatz | 回路が基底状態に近い状態に到達できる | 最適化器では取り除けない系統誤差
浅いansatz | コヒーレンス時間内の深さ | ノイズが信号を覆う
効率的な測定 | Pauli群が少数で分散も許容範囲 | ショット数が爆発する
訓練可能な地形 | 勾配が検出できる大きさ | barren plateau：最適化が停滞する
良い初期推定 | 解の近くから始める | 局所解、収束の遅さ

最初の2行の緊張関係がこの分野の中心的な設計問題です。3.2節の内容はすべて、深さではなく物理を使って表現力を安く買おうとする試みです。

* * *

## 3.2 パラメータ化回路とansatz

### 2つの流儀

**ansatz**とはパラメータから状態への写像 \\(\boldsymbol{\theta} \mapsto \lvert \psi(\boldsymbol{\theta}) \rangle\\) を回路として実現したものです。選び方には2つの流儀があります。

**ハードウェア効率型ansatz**は装置が得意なゲートをそのまま使います。1量子ビット回転の層と固有のエンタングルゲートを交互に \\(L\\) 回繰り返す形です。浅くハードウェアに優しい一方、問題については何も知りません。そのパラメータはヒルベルト空間全体を探索し、その圧倒的大部分は電子数もスピンも間違った領域です。

**化学に基づくansatz**は物理から構成します。代表例が**ユニタリ結合クラスター（UCC）**で、Hartree-Fock参照状態に指数化した励起演算子を作用させます：

\\[ \lvert \psi(\boldsymbol{\theta}) \rangle = e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})} \lvert \Phi_{\mathrm{HF}} \rangle, \qquad T = \sum_{ia} \theta_{ia} a_a^\dagger a_i + \sum_{ijab} \theta_{ijab} a_a^\dagger a_b^\dagger a_i a_j + \cdots \\]

二重励起で打ち切ったUCCSDがこの分野の主力です。第4章のJordan-Wigner変換を経れば各励起項はPauli文字列になり、各Pauli文字列は第2章のコンパイル恒等式で回路になります。パラメータ数は占有軌道 \\(N\\) 個・非占有軌道 \\(M\\) 個に対して \\(O(N^2 M^2)\\) — 多項式ですが係数が大きく、真の障害は回路の深さです。

ansatzの系統 | パラメータ数 | 深さ | 対称性の保存 | 弱点
---|---|---|---|---
ハードウェア効率型 | \\(O(nL)\\) | \\(O(L)\\) | しない | barren plateau、非物理的状態
UCCSD | \\(O(N^2M^2)\\) | 大きい | する | 現行ハードウェアには深すぎる
k-UpCCGSD | \\(O(kn^2)\\) | 中程度 | する | 精度が \\(k\\) に依存
対称性保存型 | 問題依存 | 中程度 | する | 系ごとに設計が必要
ADAPT-VQE | 適応的に成長 | 目標精度に対して最小 | する | 追加の測定が多い

### H₂のansatz：なぜ1パラメータで足りるのか

2量子ビットのH₂ハミルトニアンでは、UCCSDの階層全体がただ1つの二重励起に縮退します。すなわち2個の電子を結合性軌道 \\(\sigma_g\\) から反結合性軌道 \\(\sigma_u\\) へ昇位させる励起です。フェルミオン-量子ビット写像を経た生成子はPauli文字列 \\(X_0 Y_1\\) となり、ansatzは

\\[ \lvert \psi(\theta) \rangle = \exp\left(-i\theta X_0 Y_1\right) \lvert \Phi_{\mathrm{HF}} \rangle, \qquad \lvert \Phi_{\mathrm{HF}} \rangle = \lvert 10 \rangle \\]

本シリーズのビッグエンディアン規約では、量子ビット0が \\(\sigma_g\\) の二重占有を、量子ビット1が \\(\sigma_u\\) の二重占有を記録するので、Hartree-Fock状態は \\(\lvert 10 \rangle\\) — 状態ベクトルの添字2 — です。（この状態を \\(\lvert 01 \rangle\\) と書く論文はビット順が逆なだけで、物理は同一です。）

\\(\lvert 10 \rangle\\) に作用させると、指数関数は2項の重ね合わせを生みます。\\(X_0 Y_1 \lvert 10 \rangle = i \lvert 01 \rangle\\) かつ \\(\left(X_0Y_1\right)^2 = I\\) なので

\\[ \lvert \psi(\theta) \rangle = \cos\theta \, \lvert 10 \rangle + \sin\theta \, \lvert 01 \rangle \\]

すなわちHartree-Fock配置と二重励起配置が張る2次元空間内の回転です。真の基底状態がまさにその空間に住んでいることには、述べておく価値のある理由があります。ハミルトニアンは粒子数と全スピンと可換なので、\\(\lvert 10 \rangle\\) を \\(\lvert 00 \rangle\\)（電子ゼロ個）や \\(\lvert 11 \rangle\\)（4個）の領域に結びつけられないのです。**このansatzが厳密なのは、基底状態が属する対称性セクター全体を張っているから**です。ただしここではそのセクターがわずか2次元であり、「小さいこと」も確かに効いています。より大きな分子へ移植できるのは対称性の側だけです。対称性を回路に埋め込めば最適化器がそれを発見する必要はない、という原理は一般に成り立ちます。移植できないのは厳密性です。本物の活性空間では粒子数・スピンのセクターも依然として指数的に大きく、1パラメータの回路が張るのはその無視できるほど小さな一部にすぎません。

### ansatzをゲートにコンパイルする

回路は第2章で検証した処方に従います。\\(X\\) と \\(Y\\) を \\(Z\\) に変える基底変換で共役をとり、\\(ZZ\\) 指数関数を1個の \\(R_z\\) を挟むCNOTのはしごとして実装し、基底変換をほどきます：

\\[ \exp(-i\theta X_0 Y_1) = W^\dagger \left[\mathrm{CNOT}\_{0\to1} \left(I \otimes R_z(2\theta)\right) \mathrm{CNOT}\_{0\to1}\right] W, \qquad W = H \otimes (H S^\dagger) \\]

総コストは状態生成の \\(X\\) 1個、基底変換の1量子ビットゲート4個、CNOT 2個、パラメータ化された \\(R_z\\) 1個。8ゲート、1パラメータ、深さ6です。これまでに作られたどの装置でも走る回路です。

Code Example 1: ミニシミュレータ（第2章と同一）

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

Code Example 2: ansatz回路と対称性が買うもの

```python
import numpy as np
from qcsim import *

Sdg = S.conj().T

def hf_state():
    """Hartree-Fock参照状態 |10>: 結合性軌道 sigma_g が二重占有"""
    return apply_gate(ket('00'), X, [0], 2)

def ansatz(theta):
    """exp(-i theta X0 Y1)|HF> を H, S, CNOT, Rz にコンパイルしたもの"""
    psi = hf_state()
    psi = apply_gate(psi, H, [0], 2)        # 量子ビット0で X -> Z の基底変換
    psi = apply_gate(psi, H @ Sdg, [1], 2)  # 量子ビット1で Y -> Z の基底変換
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)    # 基底変換をほどく
    psi = apply_gate(psi, H, [0], 2)
    return psi

def hardware_efficient(params):
    """汎用の2量子ビットansatz: Ry層、CNOT、Ry層（4パラメータ）"""
    psi = ket('00')
    psi = apply_gate(psi, ry(params[0]), [0], 2)
    psi = apply_gate(psi, ry(params[1]), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, ry(params[2]), [0], 2)
    psi = apply_gate(psi, ry(params[3]), [1], 2)
    return psi

print("The Hartree-Fock reference state")
print("-" * 70)
print(f"  |HF> amplitudes = {np.round(hf_state().real, 4)}   (index 2 = |10>)")
print(f"  <ZI> = {expval(hf_state(), 'ZI'):+.1f} (qubit 0 occupied), "
      f"<IZ> = {expval(hf_state(), 'IZ'):+.1f} (qubit 1 empty)")

print("\nThe compiled circuit reproduces cos(theta)|10> + sin(theta)|01>")
print("-" * 70)
print(f"  {'theta':>8} {'amp|00>':>9} {'amp|01>':>9} {'amp|10>':>9} {'amp|11>':>9} "
      f"{'sin(th)':>9} {'cos(th)':>9}")
for th in [0.0, 0.2, -0.35, 1.0, np.pi / 2]:
    psi = ansatz(th)
    print(f"  {th:8.4f} {psi[0].real:9.6f} {psi[1].real:9.6f} {psi[2].real:9.6f} "
          f"{psi[3].real:9.6f} {np.sin(th):9.6f} {np.cos(th):9.6f}")
    assert abs(psi[0]) < 1e-12 and abs(psi[3]) < 1e-12
print("\n  |00> and |11> stay exactly empty: the ansatz never leaves the")
print("  two-electron, spin-singlet sector. Symmetry is built into the circuit,")
print("  not imposed on the optimizer.")

print("\nGate count of the chemistry-inspired ansatz")
print("-" * 70)
print("  1 X (state preparation) + 4 single-qubit basis changes + 2 CNOT + 1 Rz")
print("  parameters: 1")

print("\nA generic hardware-efficient ansatz for comparison")
print("-" * 70)
rng = np.random.default_rng(0)
print(f"  {'trial':>5} {'amp|00>':>9} {'amp|01>':>9} {'amp|10>':>9} {'amp|11>':>9} "
      f"{'in singlet sector?':>19}")
for trial in range(4):
    p = rng.uniform(0, 2 * np.pi, 4)
    psi = hardware_efficient(p)
    leak = abs(psi[0]) ** 2 + abs(psi[3]) ** 2
    print(f"  {trial:5d} {psi[0].real:9.6f} {psi[1].real:9.6f} {psi[2].real:9.6f} "
          f"{psi[3].real:9.6f} {'no, leak %.3f' % leak:>19}")
print("  parameters: 4, and generic values put weight on |00> and |11>, which")
print("  correspond to the wrong number of electrons.")

print("\nHow much of the two-qubit state space does each ansatz reach?")
print("-" * 70)
def reachable_rank(sampler, npar, samples=4000, seed=1):
    rng = np.random.default_rng(seed)
    states = np.array([sampler(rng.uniform(-np.pi, np.pi, npar)) for _ in range(samples)])
    sv = np.linalg.svd(states, compute_uv=False)
    return sv / sv[0]

sv_chem = reachable_rank(lambda p: ansatz(p[0]), 1)
sv_hw = reachable_rank(hardware_efficient, 4)
print(f"  chemistry ansatz, normalised singular values : "
      f"{np.round(sv_chem, 6).tolist()}")
print(f"  hardware-efficient ansatz                    : "
      f"{np.round(sv_hw, 6).tolist()}")
print("  the chemistry ansatz spans a 2-dimensional subspace (|10>, |01>) and")
print("  nothing else; the generic ansatz spans all four dimensions and therefore")
print("  wastes its parameters exploring unphysical states.")
```

```text
The Hartree-Fock reference state
----------------------------------------------------------------------
  |HF> amplitudes = [0. 0. 1. 0.]   (index 2 = |10>)
  <ZI> = -1.0 (qubit 0 occupied), <IZ> = +1.0 (qubit 1 empty)

The compiled circuit reproduces cos(theta)|10> + sin(theta)|01>
----------------------------------------------------------------------
     theta   amp|00>   amp|01>   amp|10>   amp|11>   sin(th)   cos(th)
    0.0000  0.000000  0.000000  1.000000  0.000000  0.000000  1.000000
    0.2000 -0.000000  0.198669  0.980067 -0.000000  0.198669  0.980067
   -0.3500  0.000000 -0.342898  0.939373  0.000000 -0.342898  0.939373
    1.0000  0.000000  0.841471  0.540302 -0.000000  0.841471  0.540302
    1.5708 -0.000000  1.000000  0.000000  0.000000  1.000000  0.000000

  |00> and |11> stay exactly empty: the ansatz never leaves the
  two-electron, spin-singlet sector. Symmetry is built into the circuit,
  not imposed on the optimizer.

Gate count of the chemistry-inspired ansatz
----------------------------------------------------------------------
  1 X (state preparation) + 4 single-qubit basis changes + 2 CNOT + 1 Rz
  parameters: 1

A generic hardware-efficient ansatz for comparison
----------------------------------------------------------------------
  trial   amp|00>   amp|01>   amp|10>   amp|11>  in singlet sector?
      0 -0.340646 -0.405554  0.610523  0.588852      no, leak 0.463
      1 -0.166294 -0.685427 -0.438994  0.556615      no, leak 0.337
      2 -0.226494  0.551667 -0.101712  0.796253      no, leak 0.685
      3  0.611648  0.070399 -0.412223 -0.671567      no, leak 0.825
  parameters: 4, and generic values put weight on |00> and |11>, which
  correspond to the wrong number of electrons.

How much of the two-qubit state space does each ansatz reach?
----------------------------------------------------------------------
  chemistry ansatz, normalised singular values : [1.0, 0.991608, 0.0, 0.0]
  hardware-efficient ansatz                    : [1.0, 0.998371, 0.995819, 0.969004]
  the chemistry ansatz spans a 2-dimensional subspace (|10>, |01>) and
  nothing else; the generic ansatz spans all four dimensions and therefore
  wastes its parameters exploring unphysical states.
```

**着目点。** コンパイルされた8ゲート回路は \\(\cos\theta \lvert 10 \rangle + \sin\theta \lvert 01 \rangle\\) を6桁の精度で再現し、残り2つの振幅は \\(10^{-17}\\) です。対称性の保存は近似ではなく厳密です。調整の結果ではなく生成子の構造から従うからです。

最後のブロックは2つの流儀の違いを定量化します。4000組のランダムなパラメータを取り、得られた状態行列の特異値を見ると到達可能集合の次元が分かります。化学ansatzは**2**（非ゼロの特異値2個、厳密にゼロが2個）、汎用ansatzは**4**です。ハードウェア効率型回路はパラメータが4倍で、次元は2倍を探索します。しかもその半分はこの問題では物理的に無意味です。2量子ビットではこの浪費は許容できます。\\(n\\) 量子ビットでは物理的な領域が全空間の指数関数的に小さい割合となり、浪費は3.6節のbarren plateauに化けます。

* * *

## 3.3 Pauli文字列としてのハミルトニアンと、その測り方

### 2量子ビットの水素分子

第二量子化した分子の電子ハミルトニアンは、スピン軌道にわたる1電子項と2電子項の和です。第4章でその道具立て — 第二量子化とJordan-Wigner変換 — を導出します。ここでは、最小基底STO-3GのH₂について、粒子数とスピンの対称性を用いて2量子ビットに縮約した結果を、O'Malleyらが用いた標準形（*Phys. Rev. X* **6**, 031007, 2016）で受け取ります：

\\[ H = g_0\, II + g_1\, Z_0 + g_2\, Z_1 + g_3\, Z_0 Z_1 + g_4\, Y_0 Y_1 + g_5\, X_0 X_1 \\]

対称性により \\(g_4 = g_5\\) であり、6個の係数すべてが核間距離 \\(R\\) に依存します。基底 \\(\lbrace \lvert 00 \rangle, \lvert 01 \rangle, \lvert 10 \rangle, \lvert 11 \rangle \rbrace\\) で行列として書くとブロック対角です。\\(\lvert 00 \rangle\\)（電子ゼロ個）と \\(\lvert 11 \rangle\\)（4個）は孤立し、\\(\lvert 01 \rangle\\) と \\(\lvert 10 \rangle\\) が \\(2 \times 2\\) ブロックを作り、その下側の固有値が分子の基底状態エネルギーになります。核間反発は既に含まれているので、固有値はハートリー単位の全エネルギーです。

### 係数表

以下の係数はSTO-3G積分から標準的な縮約手続きで計算したものです（計算全体をCode Example 8として再現するので、ここに魔法の数値はありません）。公表されているSTO-3G参照値を再現します。平衡点は \\(R = 0.735\\) Åで \\(E = -1.137306\\) ハートリー、その配置でのHartree-Fockエネルギーは \\(-1.116999\\) ハートリー、解離極限は \\(2 \times E(\mathrm{H}) = -0.933164\\) ハートリーに近づきます。

\\(R\\) (Å) | \\(g_0\\) | \\(g_1\\) | \\(g_2\\) | \\(g_3\\) | \\(g_4 = g_5\\) | \\(E_{\mathrm{HF}}\\) | \\(E_{\mathrm{exact}}\\)
---|---|---|---|---|---|---|---
0.300 | 1.684963 | 0.517383 | -1.099915 | 0.661493 | 0.080409 | -0.593828 | -0.601804
0.400 | 1.116353 | 0.470577 | -0.907062 | 0.643076 | 0.082258 | -0.904361 | -0.914150
0.500 | 0.745968 | 0.427871 | -0.738289 | 0.622805 | 0.084435 | -1.042996 | -1.055160
0.600 | 0.488827 | 0.389617 | -0.598410 | 0.601928 | 0.086865 | -1.101128 | -1.116286
0.650 | 0.389395 | 0.372033 | -0.538834 | 0.591525 | 0.088159 | -1.112997 | -1.129905
0.700 | 0.304795 | 0.355426 | -0.485486 | 0.581232 | 0.089500 | -1.117349 | -1.136189
**0.735** | **0.252992** | **0.344368** | **-0.451507** | **0.574116** | **0.090466** | **-1.116999** | **-1.137306**
0.750 | 0.232435 | 0.339769 | -0.437726 | 0.571091 | 0.090886 | -1.116151 | -1.137117
0.800 | 0.170196 | 0.325033 | -0.394886 | 0.561128 | 0.092313 | -1.110850 | -1.134148
0.900 | 0.069455 | 0.298150 | -0.321425 | 0.541795 | 0.095286 | -1.091914 | -1.120560
1.000 | -0.007740 | 0.274331 | -0.260726 | 0.523311 | 0.098395 | -1.066109 | -1.101150
1.100 | -0.068023 | 0.253080 | -0.209712 | 0.505724 | 0.101611 | -1.036539 | -1.079193
1.200 | -0.115657 | 0.233973 | -0.166406 | 0.489070 | 0.104896 | -1.005107 | -1.056741
1.300 | -0.153517 | 0.216707 | -0.129509 | 0.473378 | 0.108209 | -0.973111 | -1.035186
1.500 | -0.207755 | 0.186913 | -0.071290 | 0.444916 | 0.114768 | -0.910874 | -0.998149
1.750 | -0.248929 | 0.157229 | -0.020552 | 0.414639 | 0.122538 | -0.841349 | -0.966335
2.000 | -0.272905 | 0.134559 | 0.013303 | 0.389632 | 0.129569 | -0.783793 | -0.948641
2.500 | -0.296664 | 0.105297 | 0.051028 | 0.352010 | 0.141105 | -0.702944 | -0.936055

この表には有用な検算が隠れています。\\(g_0 + g_1 + g_2 + g_3 = \langle 00 \rvert H \lvert 00 \rangle\\) は電子ゼロ個の状態のエネルギーであり、原子単位で核間反発 \\(1/R\\) にちょうど等しくなければなりません。\\(R = 0.735\\) Å \\(= 1.388946\\) ボーアでは \\(0.719969\\) ハートリーで、4つの係数の和も \\(0.719969\\) です。縮約に誤りがあればこの恒等式が破れます。

Code Example 3: ハミルトニアンの構成と厳密対角化

```python
import numpy as np
from qcsim import *

H2_COEFFS = {
    #  R (A):   (g0,        g1,        g2,        g3,        g4,        g5)
    0.300: (  1.684963,  0.517383,  -1.099915,  0.661493,  0.080409,  0.080409),
    0.400: (  1.116353,  0.470577,  -0.907062,  0.643076,  0.082258,  0.082258),
    0.500: (  0.745968,  0.427871,  -0.738289,  0.622805,  0.084435,  0.084435),
    0.600: (  0.488827,  0.389617,  -0.598410,  0.601928,  0.086865,  0.086865),
    0.650: (  0.389395,  0.372033,  -0.538834,  0.591525,  0.088159,  0.088159),
    0.700: (  0.304795,  0.355426,  -0.485486,  0.581232,  0.089500,  0.089500),
    0.735: (  0.252992,  0.344368,  -0.451507,  0.574116,  0.090466,  0.090466),
    0.750: (  0.232435,  0.339769,  -0.437726,  0.571091,  0.090886,  0.090886),
    0.800: (  0.170196,  0.325033,  -0.394886,  0.561128,  0.092313,  0.092313),
    0.900: (  0.069455,  0.298150,  -0.321425,  0.541795,  0.095286,  0.095286),
    1.000: ( -0.007740,  0.274331,  -0.260726,  0.523311,  0.098395,  0.098395),
    1.100: ( -0.068023,  0.253080,  -0.209712,  0.505724,  0.101611,  0.101611),
    1.200: ( -0.115657,  0.233973,  -0.166406,  0.489070,  0.104896,  0.104896),
    1.300: ( -0.153517,  0.216707,  -0.129509,  0.473378,  0.108209,  0.108209),
    1.500: ( -0.207755,  0.186913,  -0.071290,  0.444916,  0.114768,  0.114768),
    1.750: ( -0.248929,  0.157229,  -0.020552,  0.414639,  0.122538,  0.122538),
    2.000: ( -0.272905,  0.134559,   0.013303,  0.389632,  0.129569,  0.129569),
    2.500: ( -0.296664,  0.105297,   0.051028,  0.352010,  0.141105,  0.141105),
}

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']

def h2_hamiltonian(R):
    """2量子ビットH₂ハミルトニアンのPauli分解 {文字列: 係数}"""
    return dict(zip(TERMS, H2_COEFFS[R]))

def pauli_matrix(pauli):
    M = np.array([[1.0 + 0j]])
    for ch in pauli:
        M = np.kron(M, PAULI[ch])
    return M

def hamiltonian_matrix(terms):
    return sum(c * pauli_matrix(p) for p, c in terms.items())

R = 0.735
terms = h2_hamiltonian(R)
Hm = hamiltonian_matrix(terms)

print(f"H2 at R = {R} A, STO-3G, two-qubit reduction")
print("-" * 66)
print("  H = " + " + ".join(f"({c:+.6f}) {p}" for p, c in terms.items()))
print("\n  matrix (real part; the imaginary part vanishes):")
for row in Hm.real:
    print("   ", "  ".join(f"{v:+9.6f}" for v in row))
print(f"\n  Hermitian: {np.allclose(Hm, Hm.conj().T)}")
print(f"  imaginary part: max |Im| = {np.max(np.abs(Hm.imag)):.1e}")

w, v = np.linalg.eigh(Hm)
print("\nExact diagonalization")
print("-" * 66)
basis = ['00', '01', '10', '11']
for k in range(4):
    comp = "  ".join(f"{v[i, k].real:+.4f}|{basis[i]}>" for i in range(4)
                     if abs(v[i, k]) > 1e-8)
    print(f"  E_{k} = {w[k]:+.6f} Ha   {comp}")

E_hf = Hm[2, 2].real          # |10> = sigma_g が二重占有 = Hartree-Fock
print(f"\n  Hartree-Fock energy  <10|H|10> = {E_hf:+.6f} Ha")
print(f"  exact ground state             = {w[0]:+.6f} Ha")
print(f"  correlation energy             = {w[0] - E_hf:+.6f} Ha "
      f"({(w[0]-E_hf)*627.509:+.2f} kcal/mol)")
print(f"  HF overlap |<10|psi_0>|^2      = {abs(v[2, 0])**2:.6f}")

print("\nDissociation curve by exact diagonalization")
print("-" * 66)
print(f"  {'R (A)':>7} {'E_HF':>11} {'E_exact':>11} {'E_corr':>10} {'|<HF|psi>|^2':>13}")
curve = []
for R in sorted(H2_COEFFS):
    M = hamiltonian_matrix(h2_hamiltonian(R))
    w, v = np.linalg.eigh(M)
    curve.append((R, M[2, 2].real, w[0]))
    print(f"  {R:7.3f} {M[2,2].real:11.6f} {w[0]:11.6f} {w[0]-M[2,2].real:10.6f} "
          f"{abs(v[2,0])**2:13.6f}")

Rs = np.array([c[0] for c in curve])
Es = np.array([c[2] for c in curve])
i = int(np.argmin(Es))
p = np.polyfit(Rs[i-1:i+2], Es[i-1:i+2], 2)
R_eq = -p[1] / (2 * p[0])
E_eq = np.polyval(p, R_eq)
print("\nEquilibrium from a parabola through the three lowest points")
print("-" * 66)
print(f"  R_eq   = {R_eq:.4f} A      (experiment 0.741 A, STO-3G/FCI 0.735 A)")
print(f"  E_min  = {E_eq:.6f} Ha    (STO-3G/FCI reference -1.1373 Ha)")
print(f"  D_e    = {(Es[-1] - E_eq) * 27.2114:.3f} eV  from E(2.5 A) - E(R_eq)")
print(f"  E(2.5 A) = {Es[-1]:.6f} Ha vs 2 x E(H atom, STO-3G) = {2*-0.4665818:.6f} Ha")
```

```text
H2 at R = 0.735 A, STO-3G, two-qubit reduction
------------------------------------------------------------------
  H = (+0.252992) II + (+0.344368) ZI + (-0.451507) IZ + (+0.574116) ZZ + (+0.090466) YY + (+0.090466) XX

  matrix (real part; the imaginary part vanishes):
    +0.719969  +0.000000  +0.000000  +0.000000
    +0.000000  +0.474751  +0.180932  +0.000000
    +0.000000  +0.180932  -1.116999  +0.000000
    +0.000000  +0.000000  +0.000000  +0.934247

  Hermitian: True
  imaginary part: max |Im| = 0.0e+00

Exact diagonalization
------------------------------------------------------------------
  E_0 = -1.137306 Ha   -0.1115|01>  +0.9938|10>
  E_1 = +0.495058 Ha   -0.9938|01>  -0.1115|10>
  E_2 = +0.719969 Ha   +1.0000|00>
  E_3 = +0.934247 Ha   +1.0000|11>

  Hartree-Fock energy  <10|H|10> = -1.116999 Ha
  exact ground state             = -1.137306 Ha
  correlation energy             = -0.020307 Ha (-12.74 kcal/mol)
  HF overlap |<10|psi_0>|^2      = 0.987560

Dissociation curve by exact diagonalization
------------------------------------------------------------------
    R (A)        E_HF     E_exact     E_corr  |<HF|psi>|^2
    0.300   -0.593828   -0.601804  -0.007976      0.997546
    0.400   -0.904362   -0.914150  -0.009788      0.996472
    0.500   -1.042997   -1.055160  -0.012163      0.994839
    0.600   -1.101128   -1.116286  -0.015158      0.992445
    0.650   -1.112997   -1.129905  -0.016908      0.990888
    0.700   -1.117349   -1.136189  -0.018840      0.989043
    0.735   -1.116999   -1.137306  -0.020307      0.987560
    0.750   -1.116151   -1.137117  -0.020966      0.986871
    0.800   -1.110851   -1.134148  -0.023297      0.984327
    0.900   -1.091915   -1.120561  -0.028646      0.977904
    1.000   -1.066108   -1.101149  -0.035041      0.969267
    1.100   -1.036539   -1.079193  -0.042654      0.957806
    1.200   -1.005106   -1.056740  -0.051634      0.942884
    1.300   -0.973111   -1.035187  -0.062076      0.923981
    1.500   -0.910874   -0.998150  -0.087276      0.873689
    1.750   -0.841349   -0.966336  -0.124987      0.793593
    2.000   -0.783793   -0.948641  -0.164848      0.711909
    2.500   -0.702943   -0.936055  -0.233112      0.594420

Equilibrium from a parabola through the three lowest points
------------------------------------------------------------------
  R_eq   = 0.7354 A      (experiment 0.741 A, STO-3G/FCI 0.735 A)
  E_min  = -1.137306 Ha    (STO-3G/FCI reference -1.1373 Ha)
  D_e    = 5.476 eV  from E(2.5 A) - E(R_eq)
  E(2.5 A) = -0.936055 Ha vs 2 x E(H atom, STO-3G) = -0.933164 Ha
```

**着目点。** 行列がブロック構造を可視化します。孤立した対角要素2つと、\\(\lvert 01 \rangle\\) と \\(\lvert 10 \rangle\\) を非対角要素 \\(2g_4 = 0.180932\\) で結ぶ \\(2 \times 2\\) ブロック1つ。この非対角要素は結合性軌道と反結合性軌道の間の交換積分 \\(K_{gu}\\) です。基底状態は \\(0.9938\lvert 10 \rangle - 0.1115\lvert 01 \rangle\\)、すなわちほぼHartree-Fockで、二重励起配置に11%の振幅が乗っています。

Hartree-Fockが取りこぼす部分である相関エネルギーは平衡点で \\(-20.3\\) ミリハートリー、12.7 kcal/molです。「化学的精度」の閾値1.6ミリハートリーの12.7倍であり、だから相関を無視できません。同時に全エネルギーに比べれば*小さく*、だから計算が難しいのです。

最も教訓的なのは最後の列です。Hartree-Fockとの重なりは短距離の0.998から2.5 Åの0.594まで落ち、相関エネルギーは \\(-8\\) ミリハートリーから \\(-233\\) ミリハートリーへ増大します。これが**静的相関**です。結合が伸びるにつれて2つの配置が縮退に近づき、単一の行列式では状態を記述できなくなります。制限Hartree-Fockはここで精度を失うのではなく、定性的に破綻し、2.5 Åでエネルギーを0.23ハートリー高く予測します。強相関物質 — Mott絶縁体、遷移金属酸化物、フラストレート磁性体 — はこの破綻が*平衡構造で*起きる系です。誰かが化学のために量子コンピュータを作る理由がこれであり、第4章で正面から取り上げます。

数値について1点留保します。この2点から得た \\(D_e = 5.48\\) eVは実験値4.75 eVではありませんが、その原因は結合長の有限性ではなく基底関数系です。不完全な解離は*逆向き*に効きます。\\(E(2.5\\,\text{Å}) = -0.936055\\) ハートリーの代わりに真のSTO-3G解離極限 \\(2E(\mathrm{H}) = -0.933164\\) ハートリーを使うと \\(D_e = 5.56\\) eV、すなわち*さらに大きく*なります。つまり2.5 Åに残った結合は誤差の原因ではなく、誤差の一部を隠しているのです。残る約0.8 eVが最小基底によるもので、最小基底はH₂を大きく過剰結合させます。アルゴリズムは厳密で、モデルが粗いのです。

### エネルギーの測定：まとめ方とショットのコスト

量子コンピュータは \\(\langle \psi \rvert H \lvert \psi \rangle\\) に直接アクセスできません。測れるのは量子ビットです。橋渡しは線形性です。\\(H = \sum_j c_j P_j\\) とPauli文字列の和で書けば

\\[ E = \sum_j c_j \langle \psi \rvert P_j \lvert \psi \rangle \\]

各 \\(\langle P_j \rangle\\) は回路を繰り返し、固有値 \\(\pm 1\\) をショット平均して推定します。装置が直接測れるPauliは \\(Z\\) だけなので、各項にはまず基底変換が必要です。\\(X\\) 因子には \\(H\\)、\\(Y\\) 因子には \\(HS^\dagger\\) — 第2章の演習1とまったく同じです。

これを実行可能に、あるいは不可能にする事実が2つあります。

**可換な項は回路を共有する。** \\(I\\) と \\(Z\\) から作られる項はすべて対角なので、同じ測定記録から推定できます。1本の回路があらゆるビット列を与え、そこから \\(\langle Z_0 \rangle\\)、\\(\langle Z_1 \rangle\\)、\\(\langle Z_0 Z_1 \rangle\\) がすべて従います。6項のハミルトニアンに必要な測定設定は**3**つだけです。\\(O(N^4)\\) 個のPauli項をもつ現実の分子では、可換な族への高度なグループ化が実行可能と絶望を分けます。

**分散は \\(1/\sqrt{N}\\) でしか減らない。** 各 \\(\langle P_j \rangle\\) は \\(\pm 1\\) 変数の標本平均なので、標準誤差は \\(\sqrt{(1-\langle P_j\rangle^2)/N}\\) です。エネルギーで目標精度 \\(\epsilon\\) に達するには合計

\\[ N \sim \left(\frac{\sum_j \lvert c_j \rvert}{\epsilon}\right)^2 \\]

ショットを要します。2乗を回避する方法はありません。量子測定は標本抽出であり、標本抽出にはコストがかかります。

Code Example 4: 項別の測定と精度の代価

```python
import numpy as np
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
terms = dict(zip(TERMS, COEFFS_735))
Sdg = S.conj().T
R = 0.735

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

theta = -0.111769
psi = ansatz(theta)

print(f"Term-by-term energy at R = {R} A, theta = {theta}")
print("-" * 62)
print(f"  {'Pauli':>6} {'coefficient':>13} {'<P>':>11} {'contribution':>14}")
total = 0.0
for p, c in terms.items():
    e = expval(psi, p)
    total += c * e
    print(f"  {p:>6} {c:+13.6f} {e:+11.6f} {c*e:+14.6f}")
print(f"  {'':6} {'':13} {'sum':>11} {total:+14.6f}")

M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
print(f"\n  one-liner with coeff_map: {sum(expval(psi, p, terms) for p in terms):+.9f}")
print(f"  <psi|H|psi> from the matrix: {np.vdot(psi, M @ psi).real:+.9f}")

print("\nMeasurement settings: commuting terms share one circuit")
print("-" * 62)
groups = {'Z basis  (II, ZI, IZ, ZZ)': ['II', 'ZI', 'IZ', 'ZZ'],
          'X basis  (XX)': ['XX'],
          'Y basis  (YY)': ['YY']}
for name, ps in groups.items():
    print(f"  {name:26s} -> {len(ps)} term(s), 1 circuit")
print(f"  total: {len(terms)} Pauli terms but only {len(groups)} distinct circuits")

def measure_in_basis(psi, basis, shots, rng):
    """測定基底を Z に回してからビット列を標本抽出する"""
    phi = psi.copy()
    for q in range(2):
        if basis[q] == 'X':
            phi = apply_gate(phi, H, [q], 2)
        elif basis[q] == 'Y':
            phi = apply_gate(phi, H @ Sdg, [q], 2)
    p = probs(phi)
    return rng.choice(4, size=shots, p=p / p.sum())

def estimate_energy(psi, shots_per_setting, rng):
    """3つの測定設定を用いたショットベースのエネルギー推定"""
    E = terms['II']
    z = measure_in_basis(psi, 'ZZ', shots_per_setting, rng)
    s0 = 1 - 2 * ((z >> 1) & 1)         # 量子ビット0の Z の固有値
    s1 = 1 - 2 * (z & 1)                # 量子ビット1の Z の固有値
    E += terms['ZI'] * s0.mean() + terms['IZ'] * s1.mean() + terms['ZZ'] * (s0 * s1).mean()
    for basis, label in [('XX', 'XX'), ('YY', 'YY')]:
        m = measure_in_basis(psi, basis, shots_per_setting, rng)
        t0 = 1 - 2 * ((m >> 1) & 1)
        t1 = 1 - 2 * (m & 1)
        E += terms[label] * (t0 * t1).mean()
    return E

exact = np.linalg.eigvalsh(M)[0]
print("\nFinite sampling: the price of every expectation value")
print("-" * 62)
print(f"  {'shots/setting':>14} {'mean E':>12} {'std':>10} {'|bias|':>10} {'std*sqrt(N)':>13}")
rng = np.random.default_rng(0)
for shots in [100, 1000, 10000, 100000]:
    runs = np.array([estimate_energy(psi, shots, rng) for _ in range(200)])
    print(f"  {shots:14d} {runs.mean():12.6f} {runs.std():10.6f} "
          f"{abs(runs.mean() - exact):10.6f} {runs.std()*np.sqrt(shots):13.4f}")
print("\n  the statistical error falls only as 1/sqrt(N): reaching 1 mHa ('chemical")
print("  accuracy' is 1.6 mHa) already needs of order 10^5-10^6 shots per setting,")
print("  and this is for the smallest molecule there is.")
```

```text
Term-by-term energy at R = 0.735 A, theta = -0.111769
--------------------------------------------------------------
   Pauli   coefficient         <P>   contribution
      II     +0.252992   +1.000000      +0.252992
      ZI     +0.344368   -0.975119      -0.335800
      IZ     -0.451507   +0.975119      -0.440273
      ZZ     +0.574116   -1.000000      -0.574116
      YY     +0.090466   -0.221681      -0.020055
      XX     +0.090466   -0.221681      -0.020055
                               sum      -1.137306

  one-liner with coeff_map: -1.137306213
  <psi|H|psi> from the matrix: -1.137306213

Measurement settings: commuting terms share one circuit
--------------------------------------------------------------
  Z basis  (II, ZI, IZ, ZZ)  -> 4 term(s), 1 circuit
  X basis  (XX)              -> 1 term(s), 1 circuit
  Y basis  (YY)              -> 1 term(s), 1 circuit
  total: 6 Pauli terms but only 3 distinct circuits

Finite sampling: the price of every expectation value
--------------------------------------------------------------
   shots/setting       mean E        std     |bias|   std*sqrt(N)
             100    -1.137723   0.022003   0.000417        0.2200
            1000    -1.137365   0.006386   0.000059        0.2019
           10000    -1.137390   0.002192   0.000084        0.2192
          100000    -1.137297   0.000660   0.000009        0.2088

  the statistical error falls only as 1/sqrt(N): reaching 1 mHa ('chemical
  accuracy' is 1.6 mHa) already needs of order 10^5-10^6 shots per setting,
  and this is for the smallest molecule there is.
```

**着目点。** 項別の表はエネルギーの出どころを示します。\\(ZZ\\) 項が \\(-0.574\\) を寄与し、2つの単一 \\(Z\\) 項はほぼ打ち消し合い、非対角の2項 \\(XX\\)・\\(YY\\) は合わせて \\(-0.040\\) しか寄与しません。しかしその小さな項こそが相関エネルギーの全体です。取り除けばHartree-Fockに戻ります。精度が最も要求されるのが最小の寄与項である、という標本推定量にとって居心地の悪い事実です。

標本抽出の表は \\(1/\sqrt{N}\\) 則を正確に確認します。積 \\(\sigma\sqrt{N}\\) はショット数3桁にわたって \\(0.21 \pm 0.01\\) ハートリーで一定です。この定数がVQEのコストを支配します。\\(\sigma = 1\\) ミリハートリーに達するには設定あたり \\(N \approx (0.21/0.001)^2 \approx 4 \times 10^4\\) ショットが必要で、しかもこれは化学で最小の分子について、1つの配置での1回のエネルギー評価にすぎません。収束した最適化にはそれが数十回必要です。3.6節で外挿します。

* * *

## 3.4 古典最適化ループ

### 勾配フリー法

最適化器から見えるのはブラックボックスです。\\(\boldsymbol{\theta}\\) を提案し、ノイズを含む \\(E(\boldsymbol{\theta})\\) を受け取ります。実機では関数値そのものが統計ノイズを帯び、有限差分の勾配はそれを増幅するので、勾配フリー法が自然な第一選択になります。

  * **COBYLA** は点の単体から線形モデルを作ります。VQE文献での事実上の標準です。評価回数が少なく、勾配不要で、中程度のノイズに耐えます。
  * **Nelder-Mead** は頑健だが遅く、ノイズのある地形では単体が潰れることがあります。
  * **Powell** は共役方向の逐次直線探索です。正確ですが評価回数を食います。
  * **SPSA** は次元によらず2回の評価で確率的勾配を推定し、パラメータの多い本当にノイジーな実機で選ばれる手法です。

### parameter-shift則

量子回路の構造は有限差分よりも良いものを与えます。シフトしたパラメータ値での2回の回路評価から得られる**厳密な**微分です。パラメータが \\(\exp(-i\theta P)\\)（\\(P^2 = I\\)）を通じて入るとしましょう。すると \\(\lvert \psi(\theta) \rangle = (\cos\theta - i \sin\theta P)\lvert \psi_0 \rangle\\) であり、エネルギーは単一のFourierモードになります：

\\[ E(\theta) = A\cos(2\theta) + B\sin(2\theta) + C \\]

定数はハミルトニアンと参照状態で決まります。微分して三角関数の加法定理を使うと、*大きな*シフトでの厳密な有限差分が得られます：

\\[ \frac{dE}{d\theta} = E\left(\theta + \frac{\pi}{4}\right) - E\left(\theta - \frac{\pi}{4}\right) \\]

慣習的な半角形式 \\(\exp(-i\theta P/2)\\) — \\(R_x\\), \\(R_y\\), \\(R_z\\) が使う形 — では同じ導出からより馴染みのある形が出ます：

\\[ \frac{dE}{d\theta} = \frac{1}{2}\left[E\left(\theta + \frac{\pi}{2}\right) - E\left(\theta - \frac{\pi}{2}\right)\right] \\]

これは近似ではありません。調整すべき刻み幅も打ち切り誤差もなく、そして実機では決定的なことに、2つの評価点が大きく離れているのでショットノイズが小さな数での除算によって増幅されません。コストはパラメータあたり回路評価2回であり、\\(p\\) パラメータのansatzの完全な勾配は \\(2p\\) 回の評価です。

同じ3つの係数 \\(A\\), \\(B\\), \\(C\\) は3回のエネルギー評価から再構成でき、その後は最小値が閉じた形 \\(\theta^\* = \tfrac{1}{2}\mathrm{atan2}(-B, -A)\\) で得られます。1パラメータのansatzではこれが最適化器を不要にします。**rotosolve**として知られる事実であり、変分アルゴリズムの古典側が常に難所とは限らないことの証しです。

Code Example 5: 同じ地形に対する4つの最適化器

```python
import numpy as np
from scipy.optimize import minimize
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
Sdg = S.conj().T

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def make_energy(terms):
    def energy(params):
        psi = ansatz(float(params[0]))
        return sum(expval(psi, p, terms) for p in terms)
    return energy

R = 0.735
terms = dict(zip(TERMS, COEFFS_735))
energy = make_energy(terms)
M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
exact = float(np.linalg.eigvalsh(M)[0])

print(f"VQE for H2 at R = {R} A  (exact ground state {exact:.9f} Ha)")
print("-" * 76)
print(f"  {'optimizer':>13} {'theta*':>11} {'E_VQE (Ha)':>14} "
      f"{'E_VQE - E_exact':>17} {'evals':>7}")
history = {}
for method, opts in [('COBYLA', {'tol': 1e-12, 'maxiter': 2000}),
                     ('Nelder-Mead', {'xatol': 1e-12, 'fatol': 1e-14}),
                     ('Powell', {'xtol': 1e-12, 'ftol': 1e-14}),
                     ('BFGS', {'gtol': 1e-10})]:
    trace = []
    def wrapped(x):
        e = energy(x)
        trace.append(e)
        return e
    res = minimize(wrapped, x0=[0.0], method=method, options=opts)
    history[method] = trace
    print(f"  {method:>13} {float(res.x[0]):11.6f} {res.fun:14.9f} "
          f"{res.fun - exact:17.2e} {len(trace):7d}")

print("\nConvergence of the COBYLA run (energy after each evaluation)")
print("-" * 76)
tr = history['COBYLA']
for i in [0, 1, 2, 3, 4, 5, 8, 12, len(tr) - 1]:
    if i < len(tr):
        print(f"  evaluation {i:3d}: E = {tr[i]:12.9f} Ha, "
              f"error = {tr[i] - exact:+.2e} Ha")

print("\nStarting point matters less than you might fear (COBYLA)")
print("-" * 76)
print(f"  {'theta_0':>9} {'theta*':>11} {'E_VQE':>14} {'error':>12}")
for x0 in [-1.5, -0.5, 0.0, 0.5, 1.5, 3.0]:
    res = minimize(energy, x0=[x0], method='COBYLA', options={'tol': 1e-12, 'maxiter': 2000})
    print(f"  {x0:9.2f} {float(res.x[0]):11.6f} {res.fun:14.9f} {res.fun - exact:12.2e}")
print("  (theta and theta + pi give the same energy: the ansatz is pi-periodic in E)")

print("\nThe variational principle is a strict bound, never an accident")
print("-" * 76)
for th in [-0.111769, -0.05, 0.0, 0.3]:
    E = energy([th])
    print(f"  theta = {th:+.6f}: E = {E:+.9f} Ha, E - E_exact = {E - exact:+.9e} "
          f"{'<-- optimum' if E - exact < 1e-9 else ''}")
print("  every trial energy lies above the exact ground state; the optimizer can")
print("  only push it down towards the true value, never below it.")
```

```text
VQE for H2 at R = 0.735 A  (exact ground state -1.137306213 Ha)
----------------------------------------------------------------------------
      optimizer      theta*     E_VQE (Ha)   E_VQE - E_exact   evals
         COBYLA   -0.111769   -1.137306213          6.66e-16      51
    Nelder-Mead   -0.111769   -1.137306213          6.66e-16      98
         Powell   -0.111769   -1.137306213          4.44e-16      88
           BFGS   -0.111769   -1.137306213          1.11e-15      28

Convergence of the COBYLA run (energy after each evaluation)
----------------------------------------------------------------------------
  evaluation   0: E = -1.116999000 Ha, error = +2.03e-02 Ha
  evaluation   1: E =  0.174597866 Ha, error = +1.31e+00 Ha
  evaluation   2: E = -0.154444138 Ha, error = +9.83e-01 Ha
  evaluation   3: E = -0.598888069 Ha, error = +5.38e-01 Ha
  evaluation   4: E = -1.106313443 Ha, error = +3.10e-02 Ha
  evaluation   5: E = -1.065188848 Ha, error = +7.21e-02 Ha
  evaluation   8: E = -1.131086000 Ha, error = +6.22e-03 Ha
  evaluation  12: E = -1.137118244 Ha, error = +1.88e-04 Ha
  evaluation  50: E = -1.137306213 Ha, error = +6.66e-16 Ha

Starting point matters less than you might fear (COBYLA)
----------------------------------------------------------------------------
    theta_0      theta*          E_VQE        error
      -1.50   -0.111769   -1.137306213     6.66e-16
      -0.50   -0.111769   -1.137306213     6.66e-16
       0.00   -0.111769   -1.137306213     6.66e-16
       0.50   -0.111769   -1.137306213     4.44e-16
       1.50    3.029824   -1.137306213     6.66e-16
       3.00    3.029824   -1.137306213     6.66e-16
  (theta and theta + pi give the same energy: the ansatz is pi-periodic in E)

The variational principle is a strict bound, never an accident
----------------------------------------------------------------------------
  theta = -0.111769: E = -1.137306213 Ha, E - E_exact = +3.774758284e-15 <-- optimum
  theta = -0.050000: E = -1.131086000 Ha, E - E_exact = +6.220212870e-03 
  theta = +0.000000: E = -1.116999000 Ha, E - E_exact = +2.030721265e-02 
  theta = +0.300000: E = -0.875826091 Ha, E - E_exact = +2.614801221e-01 
  every trial energy lies above the exact ground state; the optimizer can
  only push it down towards the true value, never below it.
```

**着目点。** 4つの最適化器はすべて \\(\theta^\* = -0.111769\\) に到達し、厳密対角化と \\(10^{-15}\\) ハートリー — 浮動小数点のノイズ — で一致するエネルギーを返します。評価回数は3倍の幅があり、実機ではそれが実時間と総ショット数の3倍の差になります。答えが同じでも最適化器の選択が実質的な設計判断である理由です。

COBYLAの最初の評価はHartree-Fockエネルギーです。\\(\theta_0 = 0\\) が参照状態をそのまま残すからです。これは標準的な初期化であり、良い初期化です。最良の単一行列式状態から探索を始め、しかも勾配がゼロでない点から始まります。誤差はその後12回の評価で \\(2 \times 10^{-2}\\) から \\(2 \times 10^{-4}\\) へ落ちます。1桁あたり約4回、これが期待すべき実務的なスケーリングです。

最後のブロックは変分原理そのものです。最適でないどの \\(\theta\\) でもエネルギーは厳密値の上に厳密に位置し、最小値近傍ではずれが2次で増えます（\\(\theta\\) が0.06ずれると6ミリハートリー、0.11ずれると20ミリハートリー）。この2次的な平坦さがVQEを頑健にし、同時に終盤の収束を遅くします。

Code Example 6: parameter-shift勾配

```python
import numpy as np
from qcsim import *

TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
COEFFS_735 = (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466)
terms = dict(zip(TERMS, COEFFS_735))
Sdg = S.conj().T
R = 0.735

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def energy(theta):
    psi = ansatz(theta)
    return sum(expval(psi, p, terms) for p in terms)

def grad_parameter_shift(theta):
    """2回のエネルギー評価から得る厳密な微分（シフトは pi/4）"""
    return energy(theta + np.pi / 4) - energy(theta - np.pi / 4)

M = sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())
A = (M[2, 2].real - M[1, 1].real) / 2
B = M[1, 2].real
def grad_analytic(theta):
    return -2 * A * np.sin(2 * theta) + 2 * B * np.cos(2 * theta)

print("Parameter-shift rule against finite differences")
print("-" * 78)
print(f"  {'theta':>8} {'analytic':>12} {'param-shift':>13} {'finite h=1e-2':>15} "
      f"{'finite h=1e-6':>15}")
for th in [-0.5, -0.111769, 0.0, 0.4, 1.2]:
    fd2 = (energy(th + 1e-2) - energy(th - 1e-2)) / 2e-2
    fd6 = (energy(th + 1e-6) - energy(th - 1e-6)) / 2e-6
    print(f"  {th:8.4f} {grad_analytic(th):12.8f} {grad_parameter_shift(th):13.8f} "
          f"{fd2:15.8f} {fd6:15.8f}")
print("\n  the parameter-shift value is exact at every theta, while the finite")
print("  difference carries a truncation error at large h and, on hardware, an")
print("  amplified shot-noise error at small h.")

err_ps = max(abs(grad_parameter_shift(t) - grad_analytic(t))
             for t in np.linspace(-2, 2, 41))
print(f"\n  max |param-shift - analytic| over theta in [-2, 2] : {err_ps:.2e}")

print("\nGradient descent driven by parameter-shift gradients")
print("-" * 78)
exact = float(np.linalg.eigvalsh(M)[0])
theta, lr = 0.6, 0.25
print(f"  {'step':>5} {'theta':>11} {'E (Ha)':>14} {'dE/dtheta':>12} {'error':>11}")
for step in range(21):
    g = grad_parameter_shift(theta)
    E = energy(theta)
    if step % 2 == 0 or step == 20:
        print(f"  {step:5d} {theta:11.6f} {E:14.9f} {g:12.6f} {E - exact:11.2e}")
    theta -= lr * g
print(f"\n  converged theta = {theta:.9f}, exact stationary point "
      f"{0.5*np.arctan2(-B, -A):.9f}")
print(f"  final energy = {energy(theta):.9f} Ha, exact = {exact:.9f} Ha, "
      f"difference = {energy(theta) - exact:.2e} Ha")

print("\nWhy shift = pi/4: the landscape is a single Fourier mode")
print("-" * 78)
print(f"  E(theta) = A cos(2 theta) + B sin(2 theta) + C with")
print(f"  A = {A:+.6f}, B = {B:+.6f}, C = {(M[2,2].real + M[1,1].real)/2:+.6f}")
print("  For a generator P with P^2 = I the exact rule is")
print("  dE/dtheta = E(theta + pi/4) - E(theta - pi/4).")
print("  Three energy evaluations therefore determine the whole one-parameter")
print("  landscape - and its exact minimum - without any optimizer at all:")
E0, Ep, Em = energy(0.0), energy(np.pi / 4), energy(-np.pi / 4)
A_r = E0 - (Ep + Em) / 2
B_r = (Ep - Em) / 2
C_r = (Ep + Em) / 2
th_star = 0.5 * np.arctan2(-B_r, -A_r)
print(f"    reconstructed A = {A_r:+.6f}, B = {B_r:+.6f}, C = {C_r:+.6f}")
print(f"    theta* = {th_star:.9f}, E = {energy(th_star):.9f} Ha, "
      f"error = {energy(th_star) - exact:.2e} Ha")
```

```text
Parameter-shift rule against finite differences
------------------------------------------------------------------------------
     theta     analytic   param-shift   finite h=1e-2   finite h=1e-6
   -0.5000  -1.14389549   -1.14389549     -1.14381923     -1.14389549
   -0.1118  -0.00000014   -0.00000014     -0.00000014     -0.00000014
    0.0000   0.36186400    0.36186400      0.36183988      0.36186400
    0.4000   1.39396463    1.39396463      1.39387171      1.39396463
    1.2000   0.80833228    0.80833228      0.80827839      0.80833228

  the parameter-shift value is exact at every theta, while the finite
  difference carries a truncation error at large h and, on hardware, an
  amplified shot-noise error at small h.

  max |param-shift - analytic| over theta in [-2, 2] : 1.55e-15

Gradient descent driven by parameter-shift gradients
------------------------------------------------------------------------------
   step       theta         E (Ha)    dE/dtheta       error
      0    0.600000   -0.440879782     1.614697    6.96e-01
      2   -0.039522   -1.128800751     0.235046    8.51e-03
      4   -0.109289   -1.137296172     0.008097    1.00e-05
      6   -0.111685   -1.137306201     0.000274    1.15e-08
      8   -0.111766   -1.137306213     0.000009    1.31e-11
     10   -0.111769   -1.137306213     0.000000    1.53e-14
     12   -0.111769   -1.137306213     0.000000    8.88e-16
     14   -0.111769   -1.137306213     0.000000    6.66e-16
     16   -0.111769   -1.137306213     0.000000    1.11e-15
     18   -0.111769   -1.137306213     0.000000    1.11e-15
     20   -0.111769   -1.137306213     0.000000    6.66e-16

  converged theta = -0.111768957, exact stationary point -0.111768957
  final energy = -1.137306213 Ha, exact = -1.137306213 Ha, difference = 4.44e-16 Ha

Why shift = pi/4: the landscape is a single Fourier mode
------------------------------------------------------------------------------
  E(theta) = A cos(2 theta) + B sin(2 theta) + C with
  A = -0.795875, B = +0.180932, C = -0.321124
  For a generator P with P^2 = I the exact rule is
  dE/dtheta = E(theta + pi/4) - E(theta - pi/4).
  Three energy evaluations therefore determine the whole one-parameter
  landscape - and its exact minimum - without any optimizer at all:
    reconstructed A = -0.795875, B = +0.180932, C = -0.321124
    theta* = -0.111768957, E = -1.137306213 Ha, error = 8.88e-16 Ha
```

**着目点。** parameter-shiftによる微分はパラメータ全域で解析値と \\(1.6 \times 10^{-15}\\) で一致する一方、\\(h = 10^{-2}\\) の有限差分はすでに小数第4位で誤っています。ノイズのないシミュレータなら \\(h = 10^{-6}\\) を使えば済みますが、各エネルギーが \\(10^{-3}\\) 程度の誤差を帯びる実機では、\\(2h = 2\times10^{-6}\\) で割ることがその誤差を \\(5 \times 10^5\\) 倍します。parameter-shift則は何でも割りません。

勾配降下は幾何級数的に収束し、2ステップごとにエネルギー誤差が1桁下がります。そして最後の再構成が要点です。3回のエネルギー評価が \\(A\\), \\(B\\), \\(C\\) を6桁で復元し、厳密な最小点を返します。1パラメータのansatzでは、VQEに最適化器はまったく必要ないのです。

* * *

## 3.5 解離曲線を検証する

いよいよ全体の計算です。各結合長で独立にVQEを走らせ、各点を同一ハミルトニアンの厳密対角化と照合します。これが本章全体が目指してきた検証です。

Code Example 7: H₂解離曲線、VQE対厳密対角化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qcsim import *

H2_COEFFS = {
    0.300: (  1.684963,  0.517383,  -1.099915,  0.661493,  0.080409,  0.080409),
    0.400: (  1.116353,  0.470577,  -0.907062,  0.643076,  0.082258,  0.082258),
    0.500: (  0.745968,  0.427871,  -0.738289,  0.622805,  0.084435,  0.084435),
    0.600: (  0.488827,  0.389617,  -0.598410,  0.601928,  0.086865,  0.086865),
    0.650: (  0.389395,  0.372033,  -0.538834,  0.591525,  0.088159,  0.088159),
    0.700: (  0.304795,  0.355426,  -0.485486,  0.581232,  0.089500,  0.089500),
    0.735: (  0.252992,  0.344368,  -0.451507,  0.574116,  0.090466,  0.090466),
    0.750: (  0.232435,  0.339769,  -0.437726,  0.571091,  0.090886,  0.090886),
    0.800: (  0.170196,  0.325033,  -0.394886,  0.561128,  0.092313,  0.092313),
    0.900: (  0.069455,  0.298150,  -0.321425,  0.541795,  0.095286,  0.095286),
    1.000: ( -0.007740,  0.274331,  -0.260726,  0.523311,  0.098395,  0.098395),
    1.100: ( -0.068023,  0.253080,  -0.209712,  0.505724,  0.101611,  0.101611),
    1.200: ( -0.115657,  0.233973,  -0.166406,  0.489070,  0.104896,  0.104896),
    1.300: ( -0.153517,  0.216707,  -0.129509,  0.473378,  0.108209,  0.108209),
    1.500: ( -0.207755,  0.186913,  -0.071290,  0.444916,  0.114768,  0.114768),
    1.750: ( -0.248929,  0.157229,  -0.020552,  0.414639,  0.122538,  0.122538),
    2.000: ( -0.272905,  0.134559,   0.013303,  0.389632,  0.129569,  0.129569),
    2.500: ( -0.296664,  0.105297,   0.051028,  0.352010,  0.141105,  0.141105),
}
TERMS = ['II', 'ZI', 'IZ', 'ZZ', 'YY', 'XX']
Sdg = S.conj().T

def ansatz(theta):
    psi = apply_gate(ket('00'), X, [0], 2)
    psi = apply_gate(psi, H, [0], 2)
    psi = apply_gate(psi, H @ Sdg, [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, rz(2 * theta), [1], 2)
    psi = cnot(psi, 0, 1, 2)
    psi = apply_gate(psi, S @ H, [1], 2)
    psi = apply_gate(psi, H, [0], 2)
    return psi

def hamiltonian_matrix(terms):
    return sum(c * np.kron(PAULI[p[0]], PAULI[p[1]]) for p, c in terms.items())

def run_vqe(terms, theta0=0.0):
    def energy(x):
        psi = ansatz(float(x[0]))
        return sum(expval(psi, p, terms) for p in terms)
    res = minimize(energy, x0=[theta0], method='COBYLA',
                   options={'tol': 1e-12, 'maxiter': 2000})
    return float(res.x[0]), float(res.fun), res.nfev

print("H2 dissociation curve: VQE against exact diagonalization")
print("-" * 88)
print(f"  {'R (A)':>6} {'theta*':>10} {'E_VQE':>13} {'E_exact':>13} "
      f"{'E_VQE-E_exact':>15} {'E_HF':>11} {'calls':>6}")
rows = []
for R in sorted(H2_COEFFS):
    terms = dict(zip(TERMS, H2_COEFFS[R]))
    M = hamiltonian_matrix(terms)
    E_exact = float(np.linalg.eigvalsh(M)[0])
    E_hf = float(M[2, 2].real)
    th, E_vqe, nfev = run_vqe(terms)
    rows.append((R, th, E_vqe, E_exact, E_hf))
    print(f"  {R:6.3f} {th:10.6f} {E_vqe:13.9f} {E_exact:13.9f} "
          f"{E_vqe - E_exact:15.2e} {E_hf:11.6f} {nfev:6d}")

rows = np.array(rows)
dev = np.abs(rows[:, 2] - rows[:, 3])
print("-" * 88)
print(f"  maximum |E_VQE - E_exact| over the whole curve : {dev.max():.3e} Ha")
print(f"  mean    |E_VQE - E_exact|                      : {dev.mean():.3e} Ha")
print(f"  chemical accuracy (1 kcal/mol)                 : {1.6e-3:.3e} Ha")
print(f"  the agreement is {1.6e-3/max(dev.max(), 1e-18):.1e} times tighter than "
      "chemical accuracy")

i = int(np.argmin(rows[:, 3]))
p = np.polyfit(rows[i-1:i+2, 0], rows[i-1:i+2, 2], 2)
R_eq = -p[1] / (2 * p[0])
print(f"\n  VQE equilibrium bond length : {R_eq:.4f} A  "
      f"(STO-3G/FCI 0.735 A, experiment 0.741 A)")
print(f"  VQE minimum energy          : {np.polyval(p, R_eq):.6f} Ha  "
      f"(STO-3G/FCI -1.137306 Ha)")
print(f"  binding energy from E(2.5 A) : "
      f"{(rows[-1, 2] - np.polyval(p, R_eq)) * 27.2114:.3f} eV  "
      "(STO-3G limit 5.55 eV, experiment 4.75 eV)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(rows[:, 0], rows[:, 4], 's--', label='Hartree-Fock', color='tab:orange')
ax1.plot(rows[:, 0], rows[:, 3], '-', label='exact diagonalization',
         color='black', linewidth=2)
ax1.plot(rows[:, 0], rows[:, 2], 'o', label='VQE', color='tab:blue',
         markerfacecolor='none', markersize=9)
ax1.axhline(2 * -0.4665818, color='grey', linestyle=':',
            label='2 x E(H), STO-3G')
ax1.set_xlabel('bond length R (Å)', fontsize=12)
ax1.set_ylabel('energy (Hartree)', fontsize=12)
ax1.set_title('H₂ dissociation curve, STO-3G / 2 qubits', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2.semilogy(rows[:, 0], np.maximum(dev, 1e-18), 'o-', color='tab:blue',
             label='|E_VQE - E_exact|')
ax2.axhline(1.6e-3, color='tab:red', linestyle='--', label='chemical accuracy')
ax2.set_xlabel('bond length R (Å)', fontsize=12)
ax2.set_ylabel('absolute error (Hartree)', fontsize=12)
ax2.set_title('VQE error against exact diagonalization', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()
```

```text
H2 dissociation curve: VQE against exact diagonalization
----------------------------------------------------------------------------------------
   R (A)     theta*         E_VQE       E_exact   E_VQE-E_exact        E_HF  calls
   0.300  -0.049555  -0.601803900  -0.601803900        1.11e-16   -0.593828     60
   0.400  -0.059428  -0.914150378  -0.914150378        5.55e-16   -0.904362     55
   0.500  -0.071904  -1.055160480  -1.055160480        8.88e-16   -1.042997     54
   0.600  -0.087028  -1.116285662  -1.116285662        8.88e-16   -1.101128     47
   0.650  -0.095603  -1.129905150  -1.129905150        1.11e-15   -1.112997     47
   0.700  -0.104867  -1.136189285  -1.136189285        6.66e-16   -1.117349     53
   0.735  -0.111769  -1.137306213  -1.137306213        6.66e-16   -1.116999     51
   0.750  -0.114833  -1.137116729  -1.137116729        4.44e-16   -1.116151     50
   0.800  -0.125522  -1.134148070  -1.134148070        2.22e-16   -1.110851     53
   0.900  -0.149200  -1.120561311  -1.120561311        8.88e-16   -1.091915     50
   1.000  -0.176218  -1.101149498  -1.101149498        4.44e-16   -1.066108     57
   1.100  -0.206885  -1.079192958  -1.079192958        6.66e-16   -1.036539     48
   1.200  -0.241325  -1.056740304  -1.056740304        6.66e-16   -1.005106     50
   1.300  -0.279334  -1.035186892  -1.035186892        6.66e-16   -0.973111     51
   1.500  -0.363345  -0.998149747  -0.998149747        6.66e-16   -0.910874     49
   1.750  -0.471609  -0.966335782  -0.966335782        8.88e-16   -0.841349     43
   2.000  -0.566570  -0.948641038  -0.948641038        6.66e-16   -0.783793     51
   2.500  -0.690408  -0.936054599  -0.936054599        6.66e-16   -0.702943     56
----------------------------------------------------------------------------------------
  maximum |E_VQE - E_exact| over the whole curve : 1.110e-15 Ha
  mean    |E_VQE - E_exact|                      : 6.538e-16 Ha
  chemical accuracy (1 kcal/mol)                 : 1.600e-03 Ha
  the agreement is 1.4e+12 times tighter than chemical accuracy

  VQE equilibrium bond length : 0.7354 A  (STO-3G/FCI 0.735 A, experiment 0.741 A)
  VQE minimum energy          : -1.137306 Ha  (STO-3G/FCI -1.137306 Ha)
  binding energy from E(2.5 A) : 5.476 eV  (STO-3G limit 5.55 eV, experiment 4.75 eV)
```

**検証結果の読み方。** 18個の配置にわたるVQEと厳密対角化の最大の差は \\(1.1 \times 10^{-15}\\) ハートリー、すなわち1のオーダーの数に対する倍精度演算の分解能です。VQEは基底状態を近似したのではなく、見つけたのです。ansatzが厳密解を含み、シミュレーションにノイズがなく、最適化器が収束していれば、これが期待される結果であり、これ以外の結果が出ればバグを疑うべきです。

自分自身ではなく文献と照合できる量が3つあり、いずれも一致します。

量 | 本計算 | 参照値（STO-3G） | 実験
---|---|---|---
平衡核間距離 | 0.7354 Å | 0.735 Å | 0.741 Å
最小エネルギー | -1.137306 Ha | -1.1373 Ha | —
0.735 ÅでのHartree-Fockエネルギー | -1.116999 Ha | -1.1170 Ha | —
解離極限 | 2.5 Åで -0.9361 Ha | -0.9332 Ha（\\(2\times\\)H） | —
結合エネルギー | 5.48 eV | 5.55 eV（最小基底の極限） | 4.75 eV

左列がアルゴリズムの計算値、中列がモデルの予測、右列が自然です。中列と右列の差は最小基底に由来し — STO-3GはH₂を約0.8 eV過剰に結合させます — どんな量子アルゴリズムもこの差を埋められません。**基底関数系を選ぶことが化学であり、その中で解くことが量子コンピュータの仕事です。** この2つを混同することが、量子化学のベンチマークを読む際の最も多い誤りです。

\\(\theta^\*\\) の列にも注目してください。短い結合長では最適パラメータは小さく（\\(-0.05\\)）、Hartree-Fockがほぼ正しいことを意味します。2.5 Åでは \\(-0.69\\) まで成長し、2つの配置が等しく寄与する値 \\(-\pi/4\\) に近づきます。VQEのただ1つのパラメータが弱相関から強相関への移行を追跡しており、しかも滑らかに追跡しています。良いansatzのパラメータがまさに振る舞うべき姿です。

### 係数はどこから来るのか

3.3節の表は論文からの引用ではなく計算結果であり、その計算がこれです。水素のSTO-3G基底は3つのGaussianの固定された縮約なので、すべての積分は閉じた形をもち、2つの分子軌道は対称性から決まり、ハミルトニアンの6係数は4つの積分から従います。この縮約が*なぜ*この形になるのかは第4章で説明します。この例が確立するのは、数値が正しいという*事実*です。

Code Example 8: STO-3G積分からの係数

```python
import numpy as np
from scipy.special import erf

BOHR = 0.52917721092                       # 1ボーアあたりのÅ
ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])   # 水素のSTO-3G
COEF = np.array([0.15432897, 0.53532814, 0.44463454])
D = COEF * (2 * ALPHA / np.pi) ** 0.75     # 規格化を含む縮約係数

def boys0(t):
    """F_0(t) = int_0^1 exp(-t u^2) du"""
    return 1.0 if t < 1e-12 else 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))

def overlap(a, b, RAB2):
    return (np.pi / (a + b)) ** 1.5 * np.exp(-a * b / (a + b) * RAB2)

def kinetic(a, b, RAB2):
    p = a * b / (a + b)
    return p * (3 - 2 * p * RAB2) * (np.pi / (a + b)) ** 1.5 * np.exp(-p * RAB2)

def nuclear(a, b, RAB2, RPC2):
    return -2 * np.pi / (a + b) * np.exp(-a * b / (a + b) * RAB2) * boys0((a + b) * RPC2)

def repulsion(a, b, c, d, RAB2, RCD2, RPQ2):
    return (2 * np.pi ** 2.5 / ((a + b) * (c + d) * np.sqrt(a + b + c + d))
            * np.exp(-a * b / (a + b) * RAB2 - c * d / (c + d) * RCD2)
            * boys0((a + b) * (c + d) / (a + b + c + d) * RPQ2))

def ao_integrals(R):
    """H2の重なり積分、コアハミルトニアン、2電子積分（Rはボーア単位）"""
    C = np.array([0.0, R])
    S = np.zeros((2, 2)); T = np.zeros((2, 2)); V = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            RAB2 = (C[A] - C[B]) ** 2
            for a, da in zip(ALPHA, D):
                for b, db in zip(ALPHA, D):
                    w = da * db
                    S[A, B] += w * overlap(a, b, RAB2)
                    T[A, B] += w * kinetic(a, b, RAB2)
                    P = (a * C[A] + b * C[B]) / (a + b)
                    for nuc in range(2):
                        V[A, B] += w * nuclear(a, b, RAB2, (P - C[nuc]) ** 2)
    ERI = np.zeros((2, 2, 2, 2))
    for A in range(2):
        for B in range(2):
            for Cc in range(2):
                for Dd in range(2):
                    RAB2 = (C[A] - C[B]) ** 2
                    RCD2 = (C[Cc] - C[Dd]) ** 2
                    s = 0.0
                    for a, da in zip(ALPHA, D):
                        for b, db in zip(ALPHA, D):
                            P = (a * C[A] + b * C[B]) / (a + b)
                            for c, dc in zip(ALPHA, D):
                                for d, dd in zip(ALPHA, D):
                                    Q = (c * C[Cc] + d * C[Dd]) / (c + d)
                                    s += da * db * dc * dd * repulsion(
                                        a, b, c, d, RAB2, RCD2, (P - Q) ** 2)
                    ERI[A, B, Cc, Dd] = s
    return S, T + V, ERI

def two_qubit_coefficients(R_angstrom):
    """STO-3G積分から得る2量子ビットH2ハミルトニアンの g0...g5"""
    R = R_angstrom / BOHR
    S, Hcore, ERI = ao_integrals(R)
    s = S[0, 1]
    # 対称性に適合した分子軌道: 結合性(g)と反結合性(u)
    Cmo = np.column_stack([np.array([1, 1]) / np.sqrt(2 + 2 * s),
                           np.array([1, -1]) / np.sqrt(2 - 2 * s)])
    h = Cmo.T @ Hcore @ Cmo
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', Cmo, Cmo, Cmo, Cmo, ERI)
    h1, h2 = h[0, 0], h[1, 1]
    J11, J22, J12, K12 = g[0, 0, 0, 0], g[1, 1, 1, 1], g[0, 0, 1, 1], g[0, 1, 0, 1]
    Enuc = 1.0 / R
    # 4つの占有パターン |q0 q1> のエネルギー
    E00 = Enuc                                                   # 電子なし
    E10 = Enuc + 2 * h1 + J11                                    # sigma_g^2  (HF)
    E01 = Enuc + 2 * h2 + J22                                    # sigma_u^2
    E11 = Enuc + 2 * h1 + 2 * h2 + J11 + J22 + 4 * J12 - 2 * K12  # 電子4個
    return ((E00 + E01 + E10 + E11) / 4,      # g0  (恒等)
            (E00 + E01 - E10 - E11) / 4,      # g1  (Z0)
            (E00 - E01 + E10 - E11) / 4,      # g2  (Z1)
            (E00 - E01 - E10 + E11) / 4,      # g3  (Z0 Z1)
            K12 / 2,                          # g4  (Y0 Y1)
            K12 / 2)                          # g5  (X0 X1)

R = 0.735
S, Hcore, ERI = ao_integrals(R / BOHR)
print(f"STO-3G integrals for H2 at R = {R} A ({R/BOHR:.5f} Bohr)")
print("-" * 70)
print(f"  overlap      S_AB  = {S[0,1]:+.6f}")
print(f"  core         H_AA  = {Hcore[0,0]:+.6f},  H_AB = {Hcore[0,1]:+.6f}")
print(f"  two-electron (AA|AA) = {ERI[0,0,0,0]:+.6f},  (AA|BB) = {ERI[0,0,1,1]:+.6f}")
print(f"               (AB|AB) = {ERI[0,1,0,1]:+.6f},  (AA|AB) = {ERI[0,0,0,1]:+.6f}")

print("\nCoefficients recomputed from the integrals vs the tabulated values")
print("-" * 70)
TABLE = {0.500: (0.745968, 0.427871, -0.738289, 0.622805, 0.084435, 0.084435),
         0.735: (0.252992, 0.344368, -0.451507, 0.574116, 0.090466, 0.090466),
         1.000: (-0.007740, 0.274331, -0.260726, 0.523311, 0.098395, 0.098395),
         2.000: (-0.272905, 0.134559, 0.013303, 0.389632, 0.129569, 0.129569)}
worst = 0.0
for R, tab in TABLE.items():
    calc = two_qubit_coefficients(R)
    diff = max(abs(a - b) for a, b in zip(calc, tab))
    worst = max(worst, diff)
    print(f"  R = {R:5.3f} A: " + " ".join(f"{v:+9.6f}" for v in calc)
          + f"   max deviation {diff:.1e}")
print(f"\n  largest deviation over all listed bond lengths: {worst:.1e}")
print("  (the table in this chapter is exactly this calculation, rounded to 6 digits)")

print("\nReference values reproduced by the same integrals")
print("-" * 70)
Rs = np.linspace(0.60, 0.90, 61)
Es = []
for R in Rs:
    g0, g1, g2, g3, g4, g5 = two_qubit_coefficients(R)
    E01 = g0 + g1 - g2 - g3
    E10 = g0 - g1 + g2 - g3
    Es.append(0.5 * (E01 + E10) - np.sqrt(0.25 * (E01 - E10) ** 2 + (2 * g4) ** 2))
Es = np.array(Es)
k = int(np.argmin(Es))
print(f"  equilibrium bond length : {Rs[k]:.3f} A   (literature STO-3G/FCI 0.735 A)")
print(f"  minimum energy          : {Es[k]:.6f} Ha  (literature -1.1373 Ha)")
g = two_qubit_coefficients(0.735)
print(f"  Hartree-Fock at 0.735 A : {g[0]-g[1]+g[2]-g[3]:.6f} Ha  "
      "(literature -1.1170 Ha)")
gd = two_qubit_coefficients(3.0)
E01d, E10d = gd[0] + gd[1] - gd[2] - gd[3], gd[0] - gd[1] + gd[2] - gd[3]
Ed = 0.5 * (E01d + E10d) - np.sqrt(0.25 * (E01d - E10d) ** 2 + (2 * gd[4]) ** 2)
print(f"  E(3.0 A)                : {Ed:.6f} Ha  (2 x E(H) = {2*-0.4665818:.6f} Ha)")
```

```text
STO-3G integrals for H2 at R = 0.735 A (1.38895 Bohr)
----------------------------------------------------------------------
  overlap      S_AB  = +0.663146
  core         H_AA  = -1.124218,  H_AB = -0.965257
  two-electron (AA|AA) = +0.774606,  (AA|BB) = +0.571877
               (AB|AB) = +0.300918,  (AA|AB) = +0.447446

Coefficients recomputed from the integrals vs the tabulated values
----------------------------------------------------------------------
  R = 0.500 A: +0.745968 +0.427871 -0.738289 +0.622805 +0.084435 +0.084435   max deviation 4.3e-07
  R = 0.735 A: +0.252992 +0.344368 -0.451507 +0.574116 +0.090466 +0.090466   max deviation 4.0e-07
  R = 1.000 A: -0.007740 +0.274331 -0.260726 +0.523311 +0.098395 +0.098395   max deviation 4.7e-07
  R = 2.000 A: -0.272905 +0.134559 +0.013303 +0.389632 +0.129569 +0.129569   max deviation 4.0e-07

  largest deviation over all listed bond lengths: 4.7e-07
  (the table in this chapter is exactly this calculation, rounded to 6 digits)

Reference values reproduced by the same integrals
----------------------------------------------------------------------
  equilibrium bond length : 0.735 A   (literature STO-3G/FCI 0.735 A)
  minimum energy          : -1.137306 Ha  (literature -1.1373 Ha)
  Hartree-Fock at 0.735 A : -1.116999 Ha  (literature -1.1170 Ha)
  E(3.0 A)                : -0.933632 Ha  (2 x E(H) = -0.933164 Ha)
```

**着目点。** 再計算した係数は表の値と \\(5 \times 10^{-7}\\) で一致し、これはまさに表を6桁に丸めた誤差です。3.3節のすべての数値が、3つのGaussian指数と3つの縮約係数から再現可能であることになります。外部の量子化学パッケージも、説明のつかない定数もありません。

最後のブロックが文献との輪を閉じます。平衡核間距離は0.735 Å、最小値は \\(-1.137306\\) ハートリー、Hartree-Fockエネルギーは \\(-1.116999\\) ハートリー、そして3.0 Åではエネルギーが \\(-0.933632\\) ハートリーまで下がり、厳密な解離極限 \\(2 \times (-0.4665818) = -0.933164\\) ハートリーに対して残留する見かけの結合は0.5ミリハートリーです。これらはH₂のSTO-3G標準値であり、私たちがVQEに渡したハミルトニアンは正しいものでした。

* * *

## 3.6 VQEの限界

3.5節の結果は厳密であり、それだけを取り出せば誤解を招きます。4つの条件が私たちに有利に働いており、そのどれもスケールアップに耐えません。

### Barren plateau（不毛な台地）

\\(n\\) 量子ビット上の深く構造をもたないansatzをランダムなパラメータで走らせると、典型的なコスト関数の勾配は平均ゼロで、分散が**\\(n\\) に対して指数関数的に減衰**します。十分に深いランダム回路が生む状態はHaarランダム状態に近づき、そうした状態上で平均した局所観測量の微分はゼロ近傍に集中するのです。大きさ \\(g\\) の勾配を分解するのに必要なショット数は \\(O(1/g^2)\\) なので、指数的に小さな勾配は指数的に大きな測定コストを意味します。最適化は派手に失敗するのではなく、ただ前進をやめます。

Code Example 9: Barren plateauを測る

```python
import numpy as np
from qcsim import *

def hardware_efficient(params, n, layers):
    """Rz-Ry回転の層とCNOTのリングを繰り返す"""
    psi = ket('0' * n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            psi = apply_gate(psi, rz(params[k]), [q], n); k += 1
            psi = apply_gate(psi, ry(params[k]), [q], n); k += 1
        for q in range(n):
            psi = cnot(psi, q, (q + 1) % n, n)
    return psi

def cost(params, n, layers):
    """単一の局所観測量: <Z0 Z1>"""
    psi = hardware_efficient(params, n, layers)
    return expval(psi, 'ZZ' + 'I' * (n - 2))

def grad_component(params, j, n, layers):
    """パラメータ j に関するparameter-shift微分（Rz/Ryなのでシフトは pi/2）"""
    p = params.copy()
    p[j] += np.pi / 2
    plus = cost(p, n, layers)
    p[j] -= np.pi
    minus = cost(p, n, layers)
    return (plus - minus) / 2

print("Barren plateaus: variance of one gradient component over random parameters")
print("-" * 84)
print(f"  {'qubits':>7} {'layers':>7} {'params':>7} {'mean dE/dtheta':>16} "
      f"{'variance':>12} {'std':>10}")
rng = np.random.default_rng(0)
samples = 120
results = []
for n in range(2, 10):
    layers = 3 * n
    npar = 2 * n * layers
    j = npar // 2
    gs = []
    for _ in range(samples):
        params = rng.uniform(0, 2 * np.pi, npar)
        gs.append(grad_component(params, j, n, layers))
    gs = np.array(gs)
    results.append((n, gs.var()))
    print(f"  {n:7d} {layers:7d} {npar:7d} {gs.mean():16.6f} "
          f"{gs.var():12.3e} {gs.std():10.5f}")

print("\nScaling of the variance")
print("-" * 84)
ns = np.array([r[0] for r in results], dtype=float)
vs = np.array([r[1] for r in results])
slope, intercept = np.polyfit(ns, np.log(vs), 1)
print(f"  fit  log(Var) = {slope:.4f} * n + {intercept:.4f}")
print(f"  i.e. Var ~ {np.exp(slope):.3f}^n : the variance decays by a factor "
      f"{1/np.exp(slope):.2f} per added qubit")
for k in range(len(ns) - 1):
    print(f"    n = {int(ns[k])} -> {int(ns[k+1])}: ratio "
          f"{vs[k+1]/vs[k]:.3f}")
print("\n  With a gradient of typical size sqrt(Var), the number of shots needed to")
print("  resolve its sign grows as 1/Var - exponentially in the number of qubits.")
print(f"  {'qubits':>7} {'typical |grad|':>15} {'shots to resolve':>18}")
for n, v in results:
    print(f"  {n:7d} {np.sqrt(v):15.5f} {1/v:18.0f}")
print("\n  This is why deep, unstructured, 'hardware-efficient' ansaetze do not")
print("  scale, and why chemistry-inspired ansaetze with a good starting point")
print("  (the Hartree-Fock state) are the only route that currently works.")
```

```text
Barren plateaus: variance of one gradient component over random parameters
------------------------------------------------------------------------------------
   qubits  layers  params   mean dE/dtheta     variance        std
        2       6      24         0.011346    1.422e-01    0.37712
        3       9      54         0.023363    4.746e-02    0.21785
        4      12      96         0.009125    3.292e-02    0.18144
        5      15     150         0.006421    1.557e-02    0.12477
        6      18     216         0.002469    7.933e-03    0.08907
        7      21     294        -0.000764    3.124e-03    0.05589
        8      24     384         0.000784    1.600e-03    0.04000
        9      27     486        -0.002680    8.513e-04    0.02918

Scaling of the variance
------------------------------------------------------------------------------------
  fit  log(Var) = -0.7205 * n + -0.6233
  i.e. Var ~ 0.487^n : the variance decays by a factor 2.06 per added qubit
    n = 2 -> 3: ratio 0.334
    n = 3 -> 4: ratio 0.694
    n = 4 -> 5: ratio 0.473
    n = 5 -> 6: ratio 0.510
    n = 6 -> 7: ratio 0.394
    n = 7 -> 8: ratio 0.512
    n = 8 -> 9: ratio 0.532

  With a gradient of typical size sqrt(Var), the number of shots needed to
  resolve its sign grows as 1/Var - exponentially in the number of qubits.
   qubits  typical |grad|   shots to resolve
        2         0.37712                  7
        3         0.21785                 21
        4         0.18144                 30
        5         0.12477                 64
        6         0.08907                126
        7         0.05589                320
        8         0.04000                625
        9         0.02918               1175

  This is why deep, unstructured, 'hardware-efficient' ansaetze do not
  scale, and why chemistry-inspired ansaetze with a good starting point
  (the Hartree-Fock state) are the only route that currently works.
```

**着目点。** 勾配の平均はどのサイズでも標本誤差の範囲でゼロです。追うべき系統的な傾斜が地形に存在しないのです。分散は量子ビットを1個加えるごとに約2分の1になり、測定した8つのサイズにわたってきれいな指数減衰を示します。したがって1つの勾配成分の*符号*を決めるのに必要なショット数は、2量子ビットの7から9量子ビットの1175へ増えます。フィットした減衰率で外挿すると、50量子ビットでは1つのパラメータの1つの勾配成分におよそ \\(10^{15}\\) ショットが必要です。現実的な繰り返し速度では研究者の一生より長い時間です。

緩和策は3つが活発に研究されており、それぞれが何を買うのかを正確に述べる価値があります。**構造をもつansatz**（UCC、対称性保存型）は回路を物理的に意味のある部分空間に制限するので、集中化の議論が適用されません。**良い初期化**（Hartree-Fockから始める、ADAPT-VQEのように層ごとに成長させる）は最適化器を勾配の大きい領域に留めます。**局所コスト関数**と浅い回路の組み合わせは、減衰が緩やかであることが証明されています。いずれも一般的な解決策ではなく、最悪の場合の指数を取り除く手法は現在知られていません。

### 測定の壁

3.4節でショット数の法則の定数を測りました。6項のPauliをもつH₂で \\(\sigma\sqrt{N} \approx 0.21\\) ハートリーです。分子ハミルトニアンのPauli項数はスピン軌道数に対して \\(O(N^4)\\) で増え、\\(\sum_j \lvert c_j \rvert\\) もそれに応じて増えるため、固定精度に必要な総ショット数はグループ化前で概ね \\(N^4\\) 以上で増大します。産業的に意義のある分子についての公表された見積りは、*1回のエネルギー評価あたり* \\(10^9\\) から \\(10^{13}\\) 回の測定で、最適化1回にはそうした評価が数百回必要です。改良されたグループ化、classical shadows、低ランク分解によってこれらの数値は大きく下がりましたが、依然として支配的なコストです。

### ノイズ

以上はすべてノイズのない回路を仮定していました。実機では各ゲートが誤差を寄与し、状態は混合状態になり、変分の上界はもはや上界ではありません。測定されたエネルギーは、ショットを増やしても縮まらない偏りを帯びます。縮めるにはより良いゲートか誤り緩和が必要です。第5章では明示的なノイズモデルでこれを定量化し、回路深さに対する忠実度の減衰を示します。

### 意味のある精度

化学的精度 — 1 kcal/mol、すなわち1.6ミリハートリー — は、計算した反応速度が予測力をもつ閾値です。速度は障壁に指数関数的に依存するからです。私たちのVQEは自分のハミルトニアンに対して \\(10^{-15}\\) ハートリーに到達しましたが、そのハミルトニアン自体は基底関数系のためにH₂の真の基底状態から40ミリハートリー離れています。現実の予測には、大きな基底（はるかに多くの量子ビット）、収束したansatz（より深い回路）、そして今日よりはるかに低い誤り率が必要です。この3つを同時に、というのが要件であり、化学における量子優位性の信頼できるロードマップがNISQ装置ではなく誤り耐性ハードウェアを指している理由です。

**VQEが本当に確立したこと。** 動く、ということです。アルゴリズムは正しく、積分からエネルギーまでのパイプラインは端から端まで検証可能で、誤り緩和を伴う小分子については実機でも実証されています。原理の証明であって、まだ道具ではありません。そして上記4つの限界のうちどれが自分の問題で最も強く効くかを正確に知っていることが、有益な研究計画とプレスリリースを分ける差です。

* * *

## 演習

#### 演習1: 変分の上界を解析的に

\\(A = (E_{10} - E_{01})/2\\)、\\(B = \langle 01 \rvert H \lvert 10 \rangle\\)、\\(C = (E_{10}+E_{01})/2\\) として \\(E(\theta) = A\cos 2\theta + B\sin 2\theta + C\\) を用いて、(a) 停留点を求めてください。(b) 最小値が \\(2 \times 2\\) ブロックの下側固有値に等しいことを示してください。(c) \\(R = 0.735\\) Åで評価し表と比較してください。

<details><summary>解答</summary>
<p>(a) \(dE/d\theta = -2A\sin 2\theta + 2B\cos 2\theta = 0\) より \(\tan 2\theta = B/A\)。最小値を選ぶのは \(\theta^* = \tfrac{1}{2}\mathrm{atan2}(-B, -A)\) で、もう一方の根 \(\theta^* + \pi/2\) は最大値です。</p>
<p>(b) 最小値では \(E = C - \sqrt{A^2 + B^2}\) です。ブロック \(\begin{pmatrix} E_{01} & B \\ B & E_{10}\end{pmatrix}\) の固有値は \(\tfrac{1}{2}(E_{01}+E_{10}) \pm \sqrt{\tfrac{1}{4}(E_{01}-E_{10})^2 + B^2}\) で、これはちょうど \(C \pm \sqrt{A^2+B^2}\) です。1パラメータansatzが厳密であるのは偶然ではなく構成上の帰結です。</p>
<p>(c) \(A = -0.795875\)、\(B = +0.180932\)、\(C = -0.321124\) として \(\theta^* = -0.111769\)、\(E = -0.321124 - \sqrt{0.633417 + 0.032736} = -1.137306\) ハートリー。表の値ともVQEの結果とも表示桁すべてで一致します。</p>
</details>

#### 演習2: 解離でHartree-Fockが破綻する理由

3.3節の表から、0.735 Åと2.5 Åでの相関エネルギーをkcal/molで、またHartree-Fockとの重なりを両方で求めてください。2つの配置の言葉で制限Hartree-Fockが伸びた結合を記述できない理由を説明し、材料科学における対応物を挙げてください。

<details><summary>解答</summary>
<p>0.735 Å: \(E_{\mathrm{corr}} = -1.137306 - (-1.116999) = -0.020307\) Ha \(= -12.74\) kcal/mol、重なりは \(\lvert\langle \mathrm{HF}\rvert\psi_0\rangle\rvert^2 = 0.9876\)。2.5 Å: \(E_{\mathrm{corr}} = -0.936055 - (-0.702944) = -0.233112\) Ha \(= -146.28\) kcal/mol、重なりは 0.5944。</p>
<p>\(R\) が大きくなると結合性軌道と反結合性軌道が縮退に近づくので \(E_{01} \to E_{10}\) となり、2つの配置 \(\sigma_g^2\) と \(\sigma_u^2\) がほぼ等しく寄与します。厳密な状態は \((\lvert 10 \rangle - \lvert 01 \rangle)/\sqrt{2}\) に近づきます。単一の行列式では等重み2配置の状態を表現できないので、制限Hartree-Fockは精度が悪いのではなく定性的に誤ります。これが<strong>静的</strong>（強）相関で、短距離の電子回避に由来する動的相関とは別物です。</p>
<p>材料科学の対応物はMott絶縁体です。格子間隔が大きい（オンサイト反発 \(U\) に対してホッピング \(t\) が小さい）とき、Hubbard模型は結合ごとにまさにこの縮退した2配置構造をもち、平均場理論 — 一般的な汎関数を用いた密度汎関数法を含む — は実験が絶縁体を見出すところで金属を予測します。第4章でその模型を構成し解きます。</p>
</details>

#### 演習3: 回転ゲートのparameter-shift則

半角形式 \\(R_y(\theta) = \exp(-i\theta Y/2)\\) に対するparameter-shift則を導出し、1量子ビットのコスト関数 \\(E(\theta) = \langle 0 \rvert R_y^\dagger(\theta) Z R_y(\theta) \lvert 0 \rangle\\) で数値的に検証してください。

<details><summary>解答</summary>
<p>半角の生成子ではエネルギーが \(E(\theta) = A\cos\theta + B\sin\theta + C\) — 周波数2ではなく1の単一Fourierモード — になるので \(dE/d\theta = \tfrac{1}{2}[E(\theta+\pi/2) - E(\theta-\pi/2)]\) です。</p>
<p>このコスト関数では \(E(\theta) = \cos\theta\) が厳密に成り立つので解析的な微分は \(-\sin\theta\) です。\(\theta = 0.3\) での数値: シフト則が \(-0.29552021\)、\(h = 10^{-6}\) の有限差分が \(-0.29552021\)、\(-\sin(0.3) = -0.29552021\)。\(\theta = 1.1\) では3つすべてが \(-0.89120736\) です。</p>
<p>一般則: \(P^2 = I\) をもつゲート \(\exp(-i\theta P/2)\) は固有値差が \(\pm 1\) の2種類なのでシフト \(\pi/2\) と係数 \(1/2\) を伴い、規約 \(\exp(-i\theta P)\) は周波数が倍になるのでシフト \(\pi/4\) で係数なしになります。2つの規約を混同することは、勾配がちょうど2倍ずれる典型的な原因です。</p>
</details>

#### 演習4: ショットの予算を見積る

測定された定数 \\(\sigma\sqrt{N} \approx 0.21\\) ハートリーを用いて、(a) \\(\sigma = 1\\) ミリハートリーに必要な設定あたりショット数を求めてください。(b) 収束した最適化が3設定×60回のエネルギー評価を要するとき、総ショット数はいくらですか。(c) 毎秒5000ショットで、1配置と18点の解離曲線にどれだけ時間がかかりますか。

<details><summary>解答</summary>
<p>(a) \(N = (0.21/0.001)^2 \approx 4.4 \times 10^4\) ショット／設定。</p>
<p>(b) \(60 \times 3 \times 4.4\times10^4 \approx 7.9 \times 10^6\) ショット。</p>
<p>(c) 毎秒5000ショットなら \(1.6 \times 10^3\) 秒 ≈ 26分／配置、18点の曲線で約8時間です。しかもこれは化学で最小の分子について、ノイズのないシミュレータが無償で達成した \(10^{-15}\) Ha よりまだ25桁近く粗い精度での話です。</p>
<p>スケールさせてみましょう。スピン軌道50個の分子ならPauli項は \(50^4 \approx 6\times10^6\) のオーダーで、\(\sum_j \lvert c_j \rvert\) もほぼその数に比例して増えるため、可換グループへのまとめを行ってもショット予算は何桁も上がります。「量子ビットは何個必要か」が最初に問うべき問いではない理由がこの計算です。</p>
</details>

#### 演習5: 冗長なパラメータ

量子ビット1に \\(R_z(2\alpha)\\) の後に \\(R_z(2\beta)\\) を置くようansatzに2つ目の \\(R_z\\) を挿入してください。エネルギーが \\(\alpha + \beta\\) にのみ依存することを数値的に示し、それが最適化の地形について、またパラメータ数の数え方一般について何を意味するか説明してください。

<details><summary>解答</summary>
<p>\(R_z(a)R_z(b) = R_z(a+b)\) なので、2つのゲートはパラメータ \(\alpha+\beta\) をもつ1つのゲートです。数値的に \(E(-0.05, -0.061769) = E(-0.3, 0.188231) = E(-0.111769, 0) = -1.137306213\) ハートリー。和が \(\theta^*\) になる任意の組が同一のエネルギーを与えます。</p>
<p>したがって地形には<strong>平坦な方向</strong>が生じます。\(\alpha + \beta = \theta^*\) に沿って厳密に縮退した最小値の連続した谷です。帰結として、勾配法はHessianが特異になり谷に沿った収束が遅くなります。Fisher情報行列は階数落ちし、「パラメータ数」はansatzの表現力を過大評価します。</p>
<p>これは人工的な例ではありません。現実のハードウェア効率型ansatzはこうした冗長性を多数含み、だからこそ<em>実効的な</em>次元 — 例えば量子Fisher情報の階数や、Code Example 2 で計算した特異値 — がパラメータ数よりも性能の良い予測指標になります。</p>
</details>

* * *

## まとめ

### 要点

**1. VQEは回路深さを繰り返し回数と交換する**

  * 位相推定は \\(O(2^m)\\) のコヒーレントな深さを要する。VQEは多数の浅い回路と古典最適化器で済む。
  * 変分原理は純粋状態の厳密な期待値について \\(E(\boldsymbol{\theta}) \geq E_0\\) を保証するので、あらゆる答えが上界であり改善が検証可能である。
  * エネルギー誤差は状態誤差の2次。重なり99%でエネルギー誤差は約1%。
  * 実機ではデコヒーレンスと読み出し偏りが上界を破る。厳密値を*下回る*測定エネルギーは成果ではなく診断である。

**2. ansatzの設計こそ物理が入る場所**

  * ハードウェア効率型回路は浅いが問題を知らない。化学に基づく回路（UCCとその親族）は粒子数とスピンを尊重する。
  * 2量子ビットH₂では単一の二重励起 \\(\exp(-i\theta X_0 Y_1)\\) を \\(\lvert 10 \rangle\\) に作用させると物理的な2次元領域をちょうど張るので厳密になる。
  * 測定した到達次元は、1パラメータの化学ansatzで2、4パラメータの汎用ansatzで4。
  * コンパイル結果は \\(X\\) 1個、基底変換4個、CNOT 2個、\\(R_z\\) 1個の計8ゲート。

**3. 測定は線形で、そして高価**

  * \\(E = \sum_j c_j \langle P_j \rangle\\)。各Pauli文字列には基底変換と \\(Z\\) 測定が必要。
  * 可換な項は回路を共有する。H₂では6項が3設定になる。
  * ここでの統計誤差は \\(\sigma\sqrt{N} \approx 0.21\\) ハートリーに従うので、1ミリハートリーには設定あたり約 \\(4 \times 10^4\\) ショットを要する。

**4. 勾配は厳密に得られる**

  * \\(P^2 = I\\) の \\(\exp(-i\theta P)\\) では \\(dE/d\theta = E(\theta+\pi/4) - E(\theta-\pi/4)\\) が厳密。\\(1.6\times10^{-15}\\) で検証済み。
  * 半角規約 \\(\exp(-i\theta P /2)\\) ではシフトが \\(\pi/2\\)、係数 \\(1/2\\)。
  * 3回の評価で1パラメータ地形の全体と厳密な最小値が再構成できる（rotosolve）。

**5. H₂での検証**

  * VQEは18の結合長にわたり同一ハミルトニアンの厳密対角化と最大 \\(1.1\times10^{-15}\\) ハートリーで一致した。
  * 平衡点は 0.7354 Å、\\(-1.137306\\) ハートリー。STO-3G参照値の 0.735 Å、\\(-1.1373\\) ハートリーと一致。
  * 係数は3つのGaussian指数から再現可能であり、再計算は \\(5\times10^{-7}\\) で一致した。
  * 相関エネルギーは0.3 Åの \\(-8\\) ミリハートリーから2.5 Åの \\(-233\\) ミリハートリーへ増大する。静的相関が単一行列式法の破綻モードであり、量子アルゴリズムを気にかける理由である。

**6. 限界は量子ビット数ではなく測定と訓練可能性**

  * Barren plateau: 深い構造なしansatzで、勾配分散が量子ビット1個あたり2.06分の1に減衰することを測定した。
  * 産業的に意義のある分子のショット予算は1回のエネルギー評価あたり \\(10^9\\)〜\\(10^{13}\\) と見積られている。
  * 基底関数系の誤差（STO-3GのH₂で40ミリハートリー）は化学でありアルゴリズムではない。量子コンピュータは与えられたモデルを解くだけである。

**実務上の含意**

  * 常にHartree-Fockから初期化し、常にVQEの数値の隣に厳密値または最良の既知参照値を併記すること。
  * 対称性保存型ansatzを優先し、パラメータを数えるのではなく実効次元を測ること。
  * 実機では有限差分ではなくparameter-shift勾配を使うこと。
  * 量子ビットより先にショットの予算を立てること。多くの場合、測定回数が律速である。

### 次章へ

私たちは最小基底の分子1つを解き、答えは厳密でした。ansatzが偶然にも厳密解を張っていたからです。第4章はその幸運を取り払います。一般的な道具立て — 第二量子化とフェルミオン演算子、それらをPauli文字列に変えるJordan-Wigner変換、そしてその結果として得られる、材料科学で重要な模型（横磁場Ising鎖とHubbard模型）の量子ビットハミルトニアン — を構築します。それらのハミルトニアンをコードで構成し、厳密対角化し、VQEを走らせ、打ち切られたansatzがどこで力不足になるかを見ます。そこはまた、VQEと量子位相推定の対比、NISQと誤り耐性計算の対比が具体的になる場所でもあります。

[← 第2章: 量子ゲートと量子回路](<chapter-2.html>) [第4章: 量子化学・材料計算への応用 →](<chapter-4.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
