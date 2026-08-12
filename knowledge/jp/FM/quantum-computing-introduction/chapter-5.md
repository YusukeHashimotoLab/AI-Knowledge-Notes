---
title: "第5章: NISQの現実と展望"
chapter_title: "第5章: NISQの現実と展望"
subtitle: ⚛️ シミュレートできるノイズ、測定できる誤り緩和、そして擁護できる評価
reading_time: 40-45分
difficulty: 上級
code_examples: 5
exercises: 6
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-computing-introduction/chapter-5.html>) | Last sync: 2026-08-12

[基礎数理道場](<../index.html>) > [量子コンピューティング入門](<index.html>) > 第5章

第1章から第4章までのすべての数値は、完璧な量子コンピュータから得られたものでした。状態ベクトルは規格化を保ち、ゲートは厳密で、期待値は計算機精度で計算されました。実機はそのいずれでもなく、両者の隔たりは後で埋め合わせればよい細部ではありません。それは、量子コンピューティングが材料研究に対して今日、来年、そしてその後の10年で何をなしうるかを決定する、まさにその事実です。

本章はこの隔たりを3段階で埋めます。まず、これまで使ってきた同じ状態ベクトルシミュレータに軌跡法でノイズを組み込み、厳密な密度行列の発展と照合して検証します。次に、現実的な誤り率で回路の忠実度が深さとともにどれだけ速く減衰するかを測定し、第4章で計算したまさにそのVQE状態にゼロノイズ外挿を適用して、ノイズ由来のバイアスがどれだけ回収可能でそのコストがいくらかを見ます。最後に、そうして得られた予算 — 深さ、幅、測定 — を量子誤り訂正が要求するものの隣に置き、近未来のハードウェアが何を届けられて何を届けられないかを、できるかぎり率直に述べます。

最後の節が本章の要点です。それは意図的に地味です。ここを読んだ研究者は、量子コンピューティングの主張、ベンダーのロードマップ、あるいはプレスリリースを見て、それが自分の研究に関係するかどうかを数分で判断できるようになるはずです。その技能は、本シリーズのどのアルゴリズムよりも有用です。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * デコヒーレンスを $T_1$、$T_2$、ゲート誤差によって定量的に記述し、関係式 $1/T_2 = 1/(2T_1) + 1/T_\phi$ を述べられる
  * 量子チャネルをKraus形式で書き、同じチャネルを純粋状態シミュレータ上で軌跡法により実装できる
  * 軌跡法のノイズモデルを厳密な密度行列の発展と照合して検証し、Monte Carlo平均の $1/\sqrt{N}$ 収束を予測できる
  * ノイズのある回路の忠実度対深さの減衰を測定し、減衰率を抽出して1層あたりのノイズ発生箇所の数と関係づけられる
  * ノイズのある期待値にゼロノイズ外挿を適用し、達成されたバイアス低減を定量化し、外挿係数の分散コストを説明できる
  * 誤り訂正の閾値を説明し、目標論理誤り率に必要な表面符号の距離を計算し、物理量子ビットと論理量子ビットの比を述べられる
  * 量子計算の3つの独立な予算 — 回路の深さ、量子ビット数、測定ショット数 — を見積もり、どれが拘束条件かを同定できる
  * 材料科学に関する量子コンピューティングの主張を、装置固有の発表に頼らず原理に基づいて評価できる

* * *

## 5.1 ノイズの物理

### 実際に何が起きるのか

量子ビットは、超伝導回路、イオントラップ、シリコン中のスピンなど、はるかに大きな物理系の2準位部分空間であり、環境はその抽象化を尊重しません。ほとんどすべては4つの破綻様式で説明されます。

機構 | 物理的起源 | 時間スケールの記号 | 状態への効果
---|---|---|---
エネルギー緩和 | 環境への自発放出 | $T_1$ | $\lvert 1 \rangle \to \lvert 0 \rangle$、占有が減衰
位相緩和 | エネルギー分裂のゆらぎ（磁束、電荷、磁場ノイズ） | $T_\phi$ | 相対位相がランダム化、コヒーレンスが減衰
ゲート誤差 | 校正不良、パルス歪み、クロストーク | ゲートあたり $p$ | わずかに誤ったユニタリが作用する
読み出し誤差 | 有限の測定忠実度、識別の重なり | ショットあたり | 測定されたビットが真の値と異なる

2つのコヒーレンス時間は

$$ \frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi} $$

として結合するので、常に $T_2 \le 2T_1$ です。緩和は副作用として位相を壊し、純位相緩和は占有を動かさずに位相を壊します。$T_1$ と $T_2$ を記載した装置のデータシートは、$T_\phi$ も述べているのです。

アルゴリズムにとって重要なのは時間そのものではなく、ゲート時間に対する比です。コヒーレンス時間 $T$ の装置で所要時間 $\tau_g$ の2量子ビットゲートは $\tau_g / T$ のオーダーの誤差下限をもち、$N_g$ 個の逐次ゲートからなる回路はおよそ $N_g \tau_g / T$ 相当の誤差を蓄積します。だからこそ「コヒーレンス時間」と「ゲート忠実度」は1つの数の2つの見方なのです。

### 密度行列を最小限に

純粋状態 $\lvert \psi \rangle$ は、環境と相関してしまった系を表現できません。最小限の拡張が**密度行列**

$$ \rho = \sum_k p_k \lvert \psi_k \rangle \langle \psi_k \rvert $$

であり、$\mathrm{Tr}\,\rho = 1$、$\rho = \rho^\dagger$、$\rho \succeq 0$ を満たします。期待値は $\langle A \rangle = \mathrm{Tr}(\rho A)$ になり、**純度** $\mathrm{Tr}(\rho^2)$ は純粋状態で1、最大混合状態で $1/2^n$ です。

ノイズ過程は**量子チャネル**であり、Kraus形式で書かれます：

$$ \rho \mapsto \mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger, \qquad \sum_k K_k^\dagger K_k = I $$

必要なものの大半は3つのチャネルでカバーされます。

**脱分極チャネル** — 標準的な主力モデルで、確率 $p$ で不特定の誤りが起こり、それが $X$、$Y$、$Z$ である確率が等しいもの：

$$ \mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right) $$

**振幅減衰** — $T_1$ 緩和、$\gamma = 1 - e^{-t/T_1}$：

$$ K_0 = \begin{pmatrix} 1 & 0 \\\\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \qquad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\\\ 0 & 0 \end{pmatrix} $$

**位相減衰** — 純位相緩和、$\lambda$ は $T_\phi$ で決まる：

$$ K_0 = \begin{pmatrix} 1 & 0 \\\\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \qquad K_1 = \begin{pmatrix} 0 & 0 \\\\ 0 & \sqrt{\lambda} \end{pmatrix} $$

### 軌跡法

密度行列のシミュレーションは $2^n$ 個ではなく $4^n$ 個の数を要し、到達可能な量子ビット数を半分にします。**軌跡法**（量子ジャンプ法、Monte Carlo波動関数法）はそれを避けます。**純粋**状態を保持し、ノイズの各箇所でランダムに選ばれた誤りを作用させ、観測量を多数の独立なランで平均するのです。軌跡数が多い極限で、この平均は密度行列を厳密に再現します。なぜなら

$$ \rho = \mathbb{E}\left[\lvert \psi_{\text{traj}} \rangle \langle \psi_{\text{traj}} \rvert\right] $$

は、チャネルが純粋状態写像の確率的混合であるという言明そのものだからです。脱分極チャネルではレシピは即座に得られます。確率 $p$ で一様ランダムなPauliを作用させるだけです。コストは統計誤差が $1/\sqrt{N_{\text{traj}}}$ で減ることであり、しかも都合よく実機の振る舞いを写しています。実機も1度に1サンプルしか与えないからです。

微妙な点は、すべてのチャネルの軌跡形が自明ではないことです。振幅減衰は状態依存のジャンプ確率と、ジャンプしない分岐の再規格化を要します。位相減衰はキックの確率に注意が必要です。確率 $q$ でランダムに $Z$ を作用させると非対角要素は $(1-2q)$ 倍になりますが、Krausチャネルでは $\sqrt{1-\lambda}$ 倍になるので、$q = (1 - \sqrt{1-\lambda})/2$ であって $\lambda/2$ ではありません。これを間違えると、もっともらしい減衰が誤った率で現れます。だからノイズモデルに対して最初にすべきことは、厳密なチャネルとの照合です。

Code Example 1: ノイズチャネル、軌跡法と厳密な密度行列の比較

```python
"""Chapter 5, Example 1: noise channels by the trajectory method, checked
against the exact density-matrix evolution."""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def bloch(rho):
    """Bloch vector (x, y, z) of a single-qubit density matrix."""
    return np.array([np.real(np.trace(rho @ P)) for P in (X, Y, Z)])


# ---------------------------------------------------------------------
# Exact channels, written with Kraus operators
# ---------------------------------------------------------------------

def kraus_apply(rho, kraus):
    return sum(K @ rho @ K.conj().T for K in kraus)


def depolarizing_kraus(p):
    """rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)."""
    return [np.sqrt(1 - p) * I2,
            np.sqrt(p / 3) * X, np.sqrt(p / 3) * Y, np.sqrt(p / 3) * Z]


def amplitude_damping_kraus(gamma):
    """Energy relaxation (T1): |1> decays to |0> with probability gamma."""
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping_kraus(lam):
    """Pure dephasing: destroys coherence without moving population."""
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]


# ---------------------------------------------------------------------
# Trajectory ("quantum jump") realisation: one pure state per shot
# ---------------------------------------------------------------------

def depolarizing_trajectory(psi, p, rng):
    """With probability p, apply one uniformly chosen Pauli error."""
    if rng.random() < p:
        return (X, Y, Z)[rng.integers(3)] @ psi
    return psi


def amplitude_damping_trajectory(psi, gamma, rng):
    """Jump |1> -> |0> with probability gamma |<1|psi>|^2; otherwise
    apply the no-jump operator and renormalize."""
    if rng.random() < gamma * abs(psi[1]) ** 2:
        return np.array([1.0 + 0j, 0.0 + 0j])
    out = np.array([psi[0], np.sqrt(1 - gamma) * psi[1]])
    return out / np.linalg.norm(out)


def phase_damping_trajectory(psi, lam, rng):
    """A random Z kick reproduces pure dephasing. A kick with probability q
    multiplies the off-diagonal element by (1 - 2q), while the Kraus channel
    multiplies it by sqrt(1 - lam), so q = (1 - sqrt(1 - lam)) / 2."""
    q = (1 - np.sqrt(1 - lam)) / 2
    if rng.random() < q:
        return Z @ psi
    return psi


def trajectory_average(psi0, step, arg, trials, seed):
    """Monte Carlo average of |psi><psi| over independent trajectories."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((2, 2), dtype=complex)
    for _ in range(trials):
        psi = step(psi0.copy(), arg, rng)
        acc += np.outer(psi, psi.conj())
    return acc / trials


# =====================================================================
np.set_printoptions(precision=5, suppress=True)
plus = H @ np.array([1.0 + 0j, 0.0 + 0j])          # |+>
trials = 200_000

print("Trajectory method vs exact density matrix, initial state |+>")
print("=" * 74)

for name, exact_kraus, traj_step, arg in (
        ("depolarizing, p = 0.15", depolarizing_kraus(0.15),
         depolarizing_trajectory, 0.15),
        ("amplitude damping, gamma = 0.30", amplitude_damping_kraus(0.30),
         amplitude_damping_trajectory, 0.30),
        ("phase damping, lambda = 0.40", phase_damping_kraus(0.40),
         phase_damping_trajectory, 0.40)):
    rho_exact = kraus_apply(np.outer(plus, plus.conj()), exact_kraus)
    rho_traj = trajectory_average(plus, traj_step, arg, trials, seed=7)
    print(f"\n  {name}")
    print(f"    exact Bloch vector      = {bloch(rho_exact)}")
    print(f"    trajectory Bloch vector = {bloch(rho_traj)}")
    print(f"    max |rho_exact - rho_traj| = "
          f"{np.abs(rho_exact - rho_traj).max():.5f}")
    print(f"    purity Tr(rho^2): exact "
          f"{np.real(np.trace(rho_exact @ rho_exact)):.5f}"
          f"   trajectory {np.real(np.trace(rho_traj @ rho_traj)):.5f}")

print()
print("Convergence of the trajectory average (depolarizing, p = 0.15)")
print("-" * 74)
print("  (mean over 8 independent runs; Monte Carlo error falls as 1/sqrt(N))")
rho_exact = kraus_apply(np.outer(plus, plus.conj()), depolarizing_kraus(0.15))
print(f"  {'trials':>10} {'mean max error':>16} {'1/sqrt(N)':>12}")
for n_tr in (100, 1_000, 10_000, 100_000):
    errs = [np.abs(trajectory_average(plus, depolarizing_trajectory,
                                      0.15, n_tr, seed=s) - rho_exact).max()
            for s in range(8)]
    print(f"  {n_tr:10,d} {np.mean(errs):16.6f} {1/np.sqrt(n_tr):12.6f}")

print()
print("Free decay: T1 and T2 as repeated weak channels")
print("-" * 74)
T1, T2 = 100.0, 60.0                      # microseconds, illustrative values
dt = 1.0
gamma = 1 - np.exp(-dt / T1)              # per-step relaxation probability
rate_phi = 1 / T2 - 1 / (2 * T1)          # 1/T2 = 1/(2 T1) + 1/T_phi
lam = 1 - np.exp(-2 * dt * rate_phi)
print(f"  T1 = {T1} us, T2 = {T2} us -> per-step gamma = {gamma:.5f},"
      f" lambda = {lam:.5f}")
print(f"  pure-dephasing time T_phi = {1/rate_phi:.2f} us")
print(f"  {'t (us)':>8} {'population':>13} {'exp(-t/T1)':>12} "
      f"{'coherence':>11} {'exp(-t/T2)':>12}")
rho_e = np.array([[0, 0], [0, 1]], dtype=complex)      # excited state |1>
rho_p = np.outer(plus, plus.conj())                    # superposition |+>
for step in range(0, 201):
    if step % 40 == 0:
        t = step * dt
        print(f"  {t:8.0f} {np.real(rho_e[1, 1]):13.6f} {np.exp(-t/T1):12.6f} "
              f"{2*abs(rho_p[0, 1]):11.6f} {np.exp(-t/T2):12.6f}")
    rho_e = kraus_apply(kraus_apply(rho_e, amplitude_damping_kraus(gamma)),
                        phase_damping_kraus(lam))
    rho_p = kraus_apply(kraus_apply(rho_p, amplitude_damping_kraus(gamma)),
                        phase_damping_kraus(lam))
```

```text
Trajectory method vs exact density matrix, initial state |+>
==========================================================================

  depolarizing, p = 0.15
    exact Bloch vector      = [0.8 0.  0. ]
    trajectory Bloch vector = [0.7999 0.     0.    ]
    max |rho_exact - rho_traj| = 0.00005
    purity Tr(rho^2): exact 0.82000   trajectory 0.81992

  amplitude damping, gamma = 0.30
    exact Bloch vector      = [0.83666 0.      0.3    ]
    trajectory Bloch vector = [0.83758 0.      0.29923]
    max |rho_exact - rho_traj| = 0.00046
    purity Tr(rho^2): exact 0.89500   trajectory 0.89554

  phase damping, lambda = 0.40
    exact Bloch vector      = [0.7746 0.     0.    ]
    trajectory Bloch vector = [0.77739 0.      0.     ]
    max |rho_exact - rho_traj| = 0.00140
    purity Tr(rho^2): exact 0.80000   trajectory 0.80217

Convergence of the trajectory average (depolarizing, p = 0.15)
--------------------------------------------------------------------------
  (mean over 8 independent runs; Monte Carlo error falls as 1/sqrt(N))
      trials   mean max error    1/sqrt(N)
         100         0.016250     0.100000
       1,000         0.006500     0.031623
      10,000         0.001962     0.010000
     100,000         0.000623     0.003162

Free decay: T1 and T2 as repeated weak channels
--------------------------------------------------------------------------
  T1 = 100.0 us, T2 = 60.0 us -> per-step gamma = 0.00995, lambda = 0.02306
  pure-dephasing time T_phi = 85.71 us
    t (us)    population   exp(-t/T1)   coherence   exp(-t/T2)
         0      1.000000     1.000000    1.000000     1.000000
        40      0.670320     0.670320    0.513417     0.513417
        80      0.449329     0.449329    0.263597     0.263597
       120      0.301194     0.301194    0.135335     0.135335
       160      0.201897     0.201897    0.069483     0.069483
       200      0.135335     0.135335    0.035674     0.035674
```

**着目点。** 最初のブロックが、これ以降のすべてを正当化する検証です。3つのチャネルすべてで軌跡平均がMonte Carlo誤差の範囲で厳密な密度行列を再現し、しかも重要なことに**純度**も再現しています。これが非自明な検査です。誤った軌跡則でもトレース1の行列は得られますが、純度は本当にどれだけ混合が起きたかに敏感です。$\lvert + \rangle$ に対する脱分極チャネルはBlochベクトルを1から $1 - 4p/3 = 0.8$ に縮め、厳密列と軌跡列の双方がまさにその値を示しています。

収束の表が代価を示します。誤差は100軌跡の0.016から100,000軌跡の0.0006へ、1000倍の労力に対して26倍の改善、すなわち期待どおりの $1/\sqrt{N}$ です。これを回避する方法はありません。統計的サンプリングは実機がやっていることでもあり、5.4節では量子ビット数ではなくショット予算が提案された計算をたいてい殺すことを示します。

最後のブロックは、繰り返される弱いチャネルから組み立てた $T_1$/$T_2$ の描像です。1ステップあたりのパラメータを $\gamma = 1 - e^{-\Delta t/T_1}$、$\lambda = 1 - e^{-2\Delta t/T_\phi}$ と選んだので、離散的な発展は印字されたすべての時刻で $e^{-t/T_1}$ と $e^{-t/T_2}$ を厳密に再現します。$T_1 = 100\ \mu\text{s}$、$T_2 = 60\ \mu\text{s}$ のとき純位相緩和時間は $T_\phi = 85.7\ \mu\text{s}$ で、200 $\mu\text{s}$ 後にコヒーレンスは3.6%まで失われます。これを回路の所要時間と比べてください。2量子ビットゲート1個が数百ナノ秒なら、200 $\mu\text{s}$ で買えるのは数百個の逐次ゲートです。この1つの比較がNISQの制約のすべてです。

* * *

## 5.2 ノイズのある回路をシミュレートする

### ノイズをどこに置くか

回路をゲートの層としてモデル化し、各ゲートの後、そのゲートが触れたすべての量子ビットに脱分極キックを1つ置きます。第3-4章のhardware-efficient ansatzの1層は $n$ 個の1量子ビット回転と $n-1$ 個のCNOTをもつので、1層あたりの**ノイズ発生箇所**の数は

$$ N_{\text{sites}} = n + 2(n-1) $$

です。$n = 4$ なら10です。CNOTにかかる係数2に注意してください。このモデルは2量子ビットゲートが触れる**それぞれの**量子ビットに独立なキックを置くので、モデル内での実効的な2量子ビットゲート誤差は $2p$ であり、公称の2量子ビットゲート誤差が $p_{2Q}$ の装置はここでは $p = p_{2Q}/2$ に対応します。本章を通して $p$ は「量子ビットあたり・ゲートあたり」の率であり、それが買う「ゲート予算」はゲート数ではなくノイズ発生**箇所**の数で数えられています。これは意図的に単純なモデルであり — 実機は1量子ビットと2量子ビットで誤り率が異なり、相関誤差、クロストーク、リーケージもあります — しかし重要な1つの特徴を捉えています。誤りはゲート作用の回数とともに蓄積し、回路の使える深さはその蓄積で決まる、という特徴です。

自然な性能指標は理想状態とノイズ状態の間の**状態忠実度**

$$ F(d) = \mathbb{E}_{\text{traj}}\left[\left\lvert \langle \psi_{\text{ideal}}(d) \mid \psi_{\text{noisy}}(d) \rangle \right\rvert^2\right] $$

で、1から出発して完全脱分極値 $1/2^n$ に向かって減衰します。

Code Example 2: 忠実度対回路深さ

```python
"""Chapter 5, Example 2: fidelity vs circuit depth on a noisy state-vector simulator.

This block is the toolbox for the rest of the chapter: run it first, then
Examples 3 and 4 in the same session (or paste everything into one file).
"""
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
# Trajectory noise: one random Pauli kick per noisy gate location
# =====================================================================

def depol_kick(state, q, n, p, rng):
    """Depolarizing channel on qubit q, trajectory realisation."""
    if p and rng.random() < p:
        return apply_gate(state, (X, Y, Z)[rng.integers(3)], [q], n)
    return state


def noisy_layer(state, n, thetas, p, rng):
    """One hardware-efficient layer: Ry on every qubit, then a CNOT ladder.
    Each gate is followed by a depolarizing kick on every qubit it touched."""
    for q in range(n):
        state = apply_gate(state, ry(thetas[q]), [q], n)
        state = depol_kick(state, q, n, p, rng)
    for q in range(n - 1):
        state = cnot(state, q, q + 1, n)
        state = depol_kick(state, q, n, p, rng)
        state = depol_kick(state, q + 1, n, p, rng)
    return state


def noise_sites_per_layer(n):
    """n single-qubit gates + (n-1) two-qubit gates, each two-qubit gate
    contributing a kick on both of its qubits."""
    return n + 2 * (n - 1)


def fidelity_curve(n, max_depth, angles, p, trajectories, seed):
    """F(d) = E_traj |<psi_ideal(d) | psi_noisy(d)>|^2 for every depth d.

    One pass per trajectory records all depths, so the whole curve costs
    about as much as a single run of the deepest circuit.
    """
    ideal, st = [], ket('0' * n)
    for d in range(max_depth):
        st = noisy_layer(st, n, angles[d], 0.0, None)
        ideal.append(st.copy())

    rng = np.random.default_rng(seed)
    acc = np.zeros(max_depth)
    for _ in range(trajectories):
        st = ket('0' * n)
        for d in range(max_depth):
            st = noisy_layer(st, n, angles[d], p, rng)
            acc[d] += abs(np.vdot(ideal[d], st)) ** 2
    return acc / trajectories


# =====================================================================
n, max_depth, trajectories = 4, 24, 2000
angles = np.random.default_rng(2).uniform(0, 2 * np.pi, size=(max_depth, n))
sites = noise_sites_per_layer(n)
depths = np.arange(1, max_depth + 1)

print(f"n = {n} qubits, {sites} noise sites per layer, "
      f"{trajectories} trajectories per point")
print(f"fully depolarized floor 1/2^n = {1/2**n:.4f}")

for p in (0.001, 0.005, 0.01):
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    gamma = -np.polyfit(depths, np.log(F), 1)[0]
    print(f"\nper-gate depolarizing probability p = {p}")
    print(f"  {'depth':>6} {'gates':>6} {'F (measured)':>13} "
          f"{'(1-p)^gates':>13} {'ratio':>7}")
    for i in range(0, max_depth, 2):
        n_sites = sites * depths[i]
        survive = (1 - p) ** n_sites
        print(f"  {depths[i]:6d} {n_sites:6d} {F[i]:13.4f} {survive:13.4f} "
              f"{F[i]/survive:7.3f}")
    print(f"  exponential fit F ~ exp(-gamma d): gamma = {gamma:.5f}")
    print(f"  per-layer survival exp(-gamma) = {np.exp(-gamma):.5f}")
    print(f"  depth where F = 0.5: {np.log(2)/gamma:.1f} layers"
          f"  ({np.log(2)/gamma*sites:.0f} noisy gates)")

print("\nHow deep can we go before the state is meaningless?")
print("-" * 70)
print(f"  {'p':>8} {'F=0.9 depth':>12} {'F=0.5 depth':>12} {'gate budget':>12}")
for p in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    gamma = -np.polyfit(depths, np.log(F), 1)[0]
    d90, d50 = np.log(1 / 0.9) / gamma, np.log(2) / gamma
    print(f"  {p:8.4f} {d90:12.1f} {d50:12.1f} {d50*sites:12.0f}")
```

```text
n = 4 qubits, 10 noise sites per layer, 2000 trajectories per point
fully depolarized floor 1/2^n = 0.0625

per-gate depolarizing probability p = 0.001
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9927        0.9900   1.003
       3     30        0.9698        0.9704   0.999
       5     50        0.9513        0.9512   1.000
       7     70        0.9360        0.9324   1.004
       9     90        0.9219        0.9139   1.009
      11    110        0.9064        0.8958   1.012
      13    130        0.8856        0.8780   1.009
      15    150        0.8660        0.8606   1.006
      17    170        0.8495        0.8436   1.007
      19    190        0.8308        0.8269   1.005
      21    210        0.8115        0.8105   1.001
      23    230        0.7985        0.7944   1.005
  exponential fit F ~ exp(-gamma d): gamma = 0.00983
  per-layer survival exp(-gamma) = 0.99022
  depth where F = 0.5: 70.5 layers  (705 noisy gates)

per-gate depolarizing probability p = 0.005
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9581        0.9511   1.007
       3     30        0.8718        0.8604   1.013
       5     50        0.7897        0.7783   1.015
       7     70        0.7169        0.7041   1.018
       9     90        0.6598        0.6369   1.036
      11    110        0.5961        0.5762   1.035
      13    130        0.5461        0.5212   1.048
      15    150        0.4918        0.4715   1.043
      17    170        0.4460        0.4265   1.046
      19    190        0.4028        0.3858   1.044
      21    210        0.3720        0.3490   1.066
      23    230        0.3431        0.3157   1.087
  exponential fit F ~ exp(-gamma d): gamma = 0.04698
  per-layer survival exp(-gamma) = 0.95410
  depth where F = 0.5: 14.8 layers  (148 noisy gates)

per-gate depolarizing probability p = 0.01
   depth  gates  F (measured)   (1-p)^gates   ratio
       1     10        0.9253        0.9044   1.023
       3     30        0.7842        0.7397   1.060
       5     50        0.6433        0.6050   1.063
       7     70        0.5235        0.4948   1.058
       9     90        0.4401        0.4047   1.087
      11    110        0.3714        0.3310   1.122
      13    130        0.3124        0.2708   1.154
      15    150        0.2649        0.2215   1.196
      17    170        0.2305        0.1811   1.273
      19    190        0.2018        0.1481   1.362
      21    210        0.1738        0.1212   1.434
      23    230        0.1520        0.0991   1.534
  exponential fit F ~ exp(-gamma d): gamma = 0.08229
  per-layer survival exp(-gamma) = 0.92100
  depth where F = 0.5: 8.4 layers  (84 noisy gates)

How deep can we go before the state is meaningless?
----------------------------------------------------------------------
         p  F=0.9 depth  F=0.5 depth  gate budget
    0.0200          0.9          6.1           61
    0.0100          1.3          8.4           84
    0.0050          2.2         14.8          148
    0.0020          5.4         35.6          356
    0.0010         10.7         70.5          705
    0.0005         20.3        133.3         1333
```

**着目点。** 減衰は深さについて指数関数的で、その率は封筒の裏の計算が予言するとおりです。素朴なモデル — 「どこにも誤りが起きなければ回路は動く」 — は生存確率 $(1-p)^{N_{\text{gates}}}$ を与え、測定された忠実度は $p = 0.001$ と $p = 0.005$ において比1.00〜1.09でそれを追跡します。比が1を超えるのは、たまたまその場にある状態に対して無害なPauli誤りがあるためで、真の忠実度は「まったく誤りなし」よりわずかに良いのです。$p = 0.01$ では深さ23で比が1.53まで上がります。忠実度が $1/2^n = 0.0625$ の床に近づいているのに、素朴なモデルはその下へ落ち続けるからです。

最後の表が記憶すべきものです。ゲートあたりの誤り率を**ゲート予算** — 状態が半分誤りになる前に使えるノイズありゲート作用の回数 — に換算します。

ゲートあたり誤り | $F = 0.5$ での使用可能ゲート数 | $F = 0.9$ での使用可能ゲート数
---|---|---
$2 \times 10^{-2}$ | 61 | 9
$1 \times 10^{-2}$ | 84 | 13
$5 \times 10^{-3}$ | 148 | 22
$1 \times 10^{-3}$ | 705 | 107
$5 \times 10^{-4}$ | 1333 | 203

予算は当然 $1/p$ でスケールします。第2列に注目してください。**有用な**計算には50%ではなく高い忠実度が必要で、$F = 0.9$ の予算はおよそ7分の1です。このモデルでの $p$ の意味を思い出してください。$p$ は「量子ビットあたり・ゲートあたり」の率なので、$p = 10^{-3}$ は**2量子ビット**ゲート誤差が $2\times10^{-3}$ の装置を表し、そのような装置は90%忠実度でノイズ発生箇所を百個程度支えます。第4章のTrotter解析は、**4量子ビットのおもちゃ模型**で精度 $10^{-3}$ を得るのに $3.4 \times 10^4$ 回のPauli回転を要しました。予算107に対して約300倍、すなわち2.5桁です。ソフトウェアの巧妙さで2.5桁を埋めることはできません。

### 減衰曲線を描く

同じデータを線形と対数の両方で描くと、2つの領域が同時に見えます。範囲の大部分では純粋な指数関数、そして $1/2^n$ での飽和です。

Code Example 3: 忠実度減衰曲線

```python
"""Chapter 5, Example 3: the fidelity-decay curve, plotted.
Continues from Example 2 (same session)."""
import matplotlib.pyplot as plt

n, max_depth, trajectories = 4, 30, 1500
angles = np.random.default_rng(2).uniform(0, 2 * np.pi, size=(max_depth, n))
depths = np.arange(1, max_depth + 1)
sites = noise_sites_per_layer(n)
rates = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
summary = []
for p in rates:
    F = fidelity_curve(n, max_depth, angles, p, trajectories, seed=17)
    mask = F > 0.3            # fit only where the decay is still exponential
    gamma = -np.polyfit(depths[mask], np.log(F[mask]), 1)[0]
    summary.append((p, gamma, np.log(2) / gamma))
    ax1.plot(depths, F, 'o-', ms=3.5, lw=1.4, label=f'p = {p}')
    ax2.semilogy(depths, F, 'o-', ms=3.5, lw=1.4, label=f'p = {p}')

for ax in (ax1, ax2):
    ax.axhline(1 / 2 ** n, color='k', ls=':', lw=1.2)
    ax.axhline(0.5, color='gray', ls='--', lw=1.0)
    ax.set_xlabel('circuit depth (layers)')
    ax.set_ylabel('state fidelity $F(d)$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
ax1.set_title(f'Fidelity vs depth, {n} qubits, {sites} noise sites per layer')
ax2.set_title('Same data, log scale: the decay is a pure exponential')
ax2.text(0.5, 1 / 2 ** n * 1.15, '$1/2^n$ floor', fontsize=8)
plt.tight_layout()
plt.show()

print(f"{'p':>8} {'gamma (per layer)':>19} {'F=0.5 depth':>13} "
      f"{'gamma/(sites*p)':>17}")
for p, gamma, d50 in summary:
    print(f"{p:8.4f} {gamma:19.5f} {d50:13.1f} {gamma/(sites*p):17.4f}")
print("\nThe last column is close to 1: the decay rate per layer is")
print("(noise sites per layer) x (error probability), with a prefactor")
print("slightly below 1 because some Pauli errors leave the state unchanged.")
```

```text
       p   gamma (per layer)   F=0.5 depth   gamma/(sites*p)
  0.0005             0.00524         132.4            1.0471
  0.0010             0.00990          70.0            0.9901
  0.0020             0.02000          34.7            1.0000
  0.0050             0.04599          15.1            0.9198
  0.0100             0.09297           7.5            0.9297
  0.0200             0.17561           3.9            0.8781

The last column is close to 1: the decay rate per layer is
(noise sites per layer) x (error probability), with a prefactor
slightly below 1 because some Pauli errors leave the state unchanged.
```

**着目点。** 最後の列が6本の曲線を1つの数に潰します。誤り率の40倍の範囲にわたって $\gamma / (N_{\text{sites}} p)$ は0.88から1.05に収まり、1層あたりの減衰率が単に

$$ \gamma \approx N_{\text{sites}}\, p \qquad \Longrightarrow \qquad F(d) \approx e^{-N_{\text{sites}} p \, d} $$

であることを述べています。これは記憶しておく価値があります。シミュレーションを一切せずに回路の忠実度を見積もれるからです。ゲートを数え、誤り率を掛け、指数をとる。40量子ビットで深さ100の回路は1層あたりおよそ $40 + 2\times39 = 118$ 箇所、つまり $1.2 \times 10^4$ 回のゲート作用をもつので、$p = 10^{-3}$ なら忠実度は $e^{-12} \approx 6 \times 10^{-6}$ です。この回路が生み出すのはノイズです。

対数スケールのパネルは、この単純な規則が効く理由と破れる場所を示します。減衰は直線 — 真の指数関数 — であり、忠実度が $1/2^n$ に近づいた時点で、状態は本質的に最大混合状態でありそれ以上悪くなれないのです。

* * *

## 5.3 誤り緩和と誤り訂正

ノイズへの対応はまったく異なる2つが存在し、両者を混同することがよくある混乱の原因です。

**誤り緩和**はノイズを受け入れて**統計**を補正します。追加の量子ビットを必要とせず、今日動作し、期待値のバイアスを減らします。しかし量子状態を回復するわけではなく、そのサンプリングコストは回路の大きさとともに急速に増大します。NISQ時代の技術です。

**誤り訂正**は1個の論理量子ビットを多数の物理量子ビットに符号化してシンドロームを継続的に測定し、**計算**からノイズを除去します。任意の深さの計算を回復しますが、閾値を下回る物理誤り率と、論理量子ビット1個あたり数百から数千の物理量子ビットというオーバーヘッドを要求します。誤り耐性時代のものです。

### ゼロノイズ外挿

最も広く使われる緩和技術が**ゼロノイズ外挿**（ZNE）です。既知の係数 $\lambda$ で意図的にノイズを増やし、複数の $\lambda$ で観測量を測定し、曲線を当てはめて $\lambda = 0$ へ外挿します：

$$ \langle A \rangle_{\lambda} \approx \langle A \rangle_0 + c_1 \lambda + c_2\lambda^2 + \cdots \quad\Longrightarrow\quad \langle A \rangle_0 \approx \sum_i w_i \langle A \rangle_{\lambda_i} $$

実務では $\lambda$ はゲートパルスを伸ばすか、打ち消し合うゲート対を挿入すること（unitary folding、$U \to U U^\dagger U$）でスケールします。ここでは誤り確率を直接スケールしますが、これは同じ発想の理想化版です。

他の一般的な技術を1行ずつ挙げます。

技術 | 発想 | 追加量子ビット | サンプリングのオーバーヘッド | 直すもの
---|---|---|---|---
ゼロノイズ外挿 | 増幅したノイズで測定しゼロへ外挿 | なし | $\sim 10$ | 期待値のバイアス
確率的誤りキャンセル | ノイズの擬確率的逆写像からサンプリング | なし | $\sim 10^2\text{-}10^4$ | バイアス、より厳密に
読み出し誤差補正 | 測定された混同行列を反転 | なし | $\sim 1$ | 測定誤差のみ
対称性検証 | 粒子数やスピンを破るショットを捨てる | なし | $\sim 1\text{-}10$ | 対称性を破る誤り
動的デカップリング | 待機中の位相緩和を再収束させるパルス列 | なし | $\sim 1$ | 待機時間の位相緩和
純化・仮想蒸留 | $M$ 個のコピーで非コヒーレント誤りを抑制 | $\times M$ | $\sim 10^2$ | 非コヒーレント誤り（コヒーレント誤りは不可）

「追加量子ビット」の列はすべて「なし」か小さな倍数であり、サンプリングの列はすべて、すでに大きなショット予算への掛け算です。それが本質的な取引です。緩和は精度をサンプル数で買うのです。

Code Example 4: ノイズのあるVQEエネルギーへのゼロノイズ外挿

```python
"""Chapter 5, Example 4: zero-noise extrapolation of a noisy VQE energy.
Continues from Example 2 (same session)."""


def noisy_ansatz(theta, n, layers, p=0.0, rng=None):
    """The Chapter 3-4 hardware-efficient ansatz with depolarizing kicks
    after every gate. p = 0 reproduces the noiseless circuit exactly."""
    psi, k = ket('0' * n), 0
    for q in range(n):
        psi = apply_gate(psi, ry(theta[k]), [q], n)
        k += 1
        psi = depol_kick(psi, q, n, p, rng)
    for _ in range(layers):
        for q in range(n - 1):
            psi = cnot(psi, q, q + 1, n)
            psi = depol_kick(psi, q, n, p, rng)
            psi = depol_kick(psi, q + 1, n, p, rng)
        for q in range(n):
            psi = apply_gate(psi, ry(theta[k]), [q], n)
            k += 1
            psi = depol_kick(psi, q, n, p, rng)
    return psi


def tfim_hamiltonian(N, J, h):
    terms = {}
    for i in range(N - 1):
        s = 'I' * i + 'ZZ' + 'I' * (N - i - 2)
        terms[s] = terms.get(s, 0.0) - J
    for i in range(N):
        s = 'I' * i + 'X' + 'I' * (N - i - 1)
        terms[s] = terms.get(s, 0.0) - h
    return terms


def exact_ground_energy(terms):
    n = len(next(iter(terms)))
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in terms.items():
        A = np.array([[1.0 + 0j]])
        for ch in s:
            A = np.kron(A, PAULI[ch])
        M += c * A
    return float(np.linalg.eigvalsh(M)[0])


def energy(theta, terms, n, layers):
    psi = noisy_ansatz(theta, n, layers)
    return sum(expval(psi, s, terms) for s in terms)


def gradient(theta, terms, n, layers):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy(tp, terms, n, layers)
                      - energy(tm, terms, n, layers))
    return g


def noisy_energy(theta, terms, n, layers, p, trajectories, rng):
    """Trajectory-averaged <H> as a noisy device would report it."""
    tot = 0.0
    for _ in range(trajectories):
        psi = noisy_ansatz(theta, n, layers, p=p, rng=rng)
        tot += sum(expval(psi, s, terms) for s in terms)
    return tot / trajectories


n, layers = 4, 3
terms = tfim_hamiltonian(n, 1.0, 1.0)
E_exact = exact_ground_energy(terms)

# noiseless VQE first, so we know the target the noisy device should reproduce
theta = np.random.default_rng(1).normal(0.0, 0.3, size=n * (layers + 1))
for _ in range(800):
    theta -= 0.3 * gradient(theta, terms, n, layers)
E_clean = energy(theta, terms, n, layers)

print("Zero-noise extrapolation, 4-qubit transverse-field Ising chain")
print("=" * 74)
print(f"  exact ground state       E0 = {E_exact:+.6f}")
print(f"  noiseless VQE (3 layers) E  = {E_clean:+.6f}"
      f"   (ansatz error {E_clean - E_exact:+.2e})")
print(f"  circuit: {n*(layers+1)} Ry gates, {layers*(n-1)} CNOTs,"
      f" {n*(layers+1) + 2*layers*(n-1)} noise sites")

trajectories = 6000
for p0 in (0.002, 0.005):
    print(f"\n  base error rate p0 = {p0}"
          f"   ({trajectories} trajectories per noise scale)")
    lams = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(101)
    Es = []
    for lam in lams:
        E = noisy_energy(theta, terms, n, layers, p0 * lam, trajectories, rng)
        Es.append(E)
        print(f"    lambda = {lam:.0f}  (p = {p0*lam:.3f}):"
              f"  <H> = {E:+.6f}   bias = {E - E_clean:+.6f}")
    Es = np.array(Es)
    lin = np.polyval(np.polyfit(lams, Es, 1), 0.0)
    quad = np.polyval(np.polyfit(lams, Es, 2), 0.0)
    print(f"    unmitigated (lambda = 1) : {Es[0]:+.6f}"
          f"   residual bias {Es[0]-E_clean:+.6f}")
    print(f"    linear extrapolation     : {lin:+.6f}"
          f"   residual bias {lin-E_clean:+.6f}")
    print(f"    quadratic extrapolation  : {quad:+.6f}"
          f"   residual bias {quad-E_clean:+.6f}")
    print(f"    bias reduction (linear)  : "
          f"{abs(Es[0]-E_clean)/abs(lin-E_clean):.1f}x")
    # One run is one seed.  Replicate to separate genuine bias from
    # sampling fluctuation: the two fits differ in variance, not in bias.
    lin_b, quad_b = [], []
    for seed in range(201, 207):
        rng_s = np.random.default_rng(seed)
        Es_s = np.array([noisy_energy(theta, terms, n, layers, p0 * lam,
                                      trajectories, rng_s) for lam in lams])
        lin_b.append(np.polyval(np.polyfit(lams, Es_s, 1), 0.0) - E_clean)
        quad_b.append(np.polyval(np.polyfit(lams, Es_s, 2), 0.0) - E_clean)
    lin_b, quad_b = np.array(lin_b), np.array(quad_b)
    print("    the printed run is ONE seed; over 6 independent seeds:")
    print(f"      linear    residual bias = {lin_b.mean():+.4f}"
          f"  +/- {lin_b.std(ddof=1):.4f}")
    print(f"      quadratic residual bias = {quad_b.mean():+.4f}"
          f"  +/- {quad_b.std(ddof=1):.4f}")

print("\n  Why the bias has this sign: depolarizing noise pulls the state")
print("  towards the maximally mixed state, whose energy is Tr(H)/2^n = 0,")
print("  so a negative ground-state energy is systematically raised.")

print("\n  The cost of the mitigation")
print("  " + "-" * 66)
print("  Richardson extrapolation is E(0) = sum_i w_i E(lambda_i), and the")
print("  weights alternate in sign and grow with the number of noise scales:")
print(f"    {'noise scales':<24} {'weights w_i':<24} {'||w||_2':>8} {'sum|w_i|':>9}")
for lams_try in ([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]):
    V = np.vander(np.array(lams_try), len(lams_try), increasing=True)
    w = np.linalg.inv(V)[0]
    print(f"    {str(lams_try):<24} {str(np.round(w, 2)):<24} "
          f"{np.linalg.norm(w):8.2f} {np.abs(w).sum():9.2f}")
print("  Those are the weights of an EXACT interpolation through m points, which\n"
      "  is what the quadratic fit above is for m = 3.  The linear fit above is a\n"
      "  least-squares line through the same three points, and its weights are\n"
      "  much gentler:")
w_lin = np.linalg.pinv(np.vander(lams, 2, increasing=True))[0]
print(f"    {str([float(x) for x in lams]):<24} {str(np.round(w_lin, 3)):<24} "
      f"{np.linalg.norm(w_lin):8.2f} {np.abs(w_lin).sum():9.2f}")
print("  With m noise scales the shot budget must grow by roughly ||w||^2 to")
print("  hold the statistical error fixed: mitigation buys bias with variance.")
```

```text
Zero-noise extrapolation, 4-qubit transverse-field Ising chain
==========================================================================
  exact ground state       E0 = -4.758770
  noiseless VQE (3 layers) E  = -4.749403   (ansatz error +9.37e-03)
  circuit: 16 Ry gates, 9 CNOTs, 34 noise sites

  base error rate p0 = 0.002   (6000 trajectories per noise scale)
    lambda = 1  (p = 0.002):  <H> = -4.569866   bias = +0.179537
    lambda = 2  (p = 0.004):  <H> = -4.406701   bias = +0.342703
    lambda = 3  (p = 0.006):  <H> = -4.237564   bias = +0.511840
    unmitigated (lambda = 1) : -4.569866   residual bias +0.179537
    linear extrapolation     : -4.737013   residual bias +0.012390
    quadratic extrapolation  : -4.727061   residual bias +0.022343
    bias reduction (linear)  : 14.5x
    the printed run is ONE seed; over 6 independent seeds:
      linear    residual bias = +0.0127  +/- 0.0098
      quadratic residual bias = +0.0124  +/- 0.0341

  base error rate p0 = 0.005   (6000 trajectories per noise scale)
    lambda = 1  (p = 0.005):  <H> = -4.301924   bias = +0.447479
    lambda = 2  (p = 0.010):  <H> = -3.950008   bias = +0.799395
    lambda = 3  (p = 0.015):  <H> = -3.569222   bias = +1.180181
    unmitigated (lambda = 1) : -4.301924   residual bias +0.447479
    linear extrapolation     : -4.673087   residual bias +0.076316
    quadratic extrapolation  : -4.624970   residual bias +0.124434
    bias reduction (linear)  : 5.9x
    the printed run is ONE seed; over 6 independent seeds:
      linear    residual bias = +0.0585  +/- 0.0124
      quadratic residual bias = +0.0282  +/- 0.0770

  Why the bias has this sign: depolarizing noise pulls the state
  towards the maximally mixed state, whose energy is Tr(H)/2^n = 0,
  so a negative ground-state energy is systematically raised.

  The cost of the mitigation
  ------------------------------------------------------------------
  Richardson extrapolation is E(0) = sum_i w_i E(lambda_i), and the
  weights alternate in sign and grow with the number of noise scales:
    noise scales             weights w_i               ||w||_2  sum|w_i|
    [1.0, 2.0]               [ 2. -1.]                    2.24      3.00
    [1.0, 2.0, 3.0]          [ 3. -3.  1.]                4.36      7.00
    [1.0, 2.0, 3.0, 4.0]     [ 4. -6.  4. -1.]            8.31     15.00
  Those are the weights of an EXACT interpolation through m points, which
  is what the quadratic fit above is for m = 3.  The linear fit above is a
  least-squares line through the same three points, and its weights are
  much gentler:
    [1.0, 2.0, 3.0]          [ 1.333  0.333 -0.667]       1.53      2.33
  With m noise scales the shot budget must grow by roughly ||w||^2 to
  hold the statistical error fixed: mitigation buys bias with variance.
```

**着目点。** 重要度の低い順に4点。

**ZNEは機能し、有用な倍率で機能します。** $p = 0.002$ では生のノイズありエネルギーのバイアスが $+0.180$ で、線形外挿がそれを $+0.012$ にします。14.5倍の低減です。$p = 0.005$ では5.9倍です。この技術は化粧ではなく実質です。

**バイアスはansatz誤差よりはるかに大きく、意味づけが逆です。** 3層のansatzは厳密基底状態を $+9.4 \times 10^{-3}$ 外します。$p = 0.002$ のノイズは $+0.18$ のバイアスを加え、19倍大きいのです。ノイズのバイアスをansatz誤差より下げるまで、ansatzの設計に費やす労力はすべて無駄です。この順序 — ノイズが先、アルゴリズムが後 — が近未来の研究の正しい優先順位であり、文献ではしばしば逆になっています。

**高次の外挿はバイアスの点で劣るのではなく、ノイズに弱いのです。そして1回の実行ではどちらか判断できません。** 印字された実行では2次の当てはめが両方の場合で線形より悪く（$+0.022$ 対 $+0.012$、$+0.124$ 対 $+0.076$）、ここから誤った結論を引き出すのは容易です。誤った結論とは「2次は自由度が1つ余り、それをノイズの当てはめに費やす」というものです。**3**点を通る2次曲線には余分な自由度はなく、3点をちょうど内挿するので、過剰適合する対象がありません。実際に起きているのはノイズの**増幅**です。その下に印字された6シードの反復が本当の振る舞いを見せます。$p_0 = 0.002$ では平均バイアスが区別できず（線形 $+0.013$、2次 $+0.012$）、$p_0 = 0.005$ では2次のほうが**バイアスが小さい**（$+0.059$ 対 $+0.028$）。展開の $\lambda$ 項だけでなく $\lambda^2$ 項まで打ち消すので、そうなるべきなのです。しかしそのばらつきは3〜6倍大きく（$\pm 0.010$・$\pm 0.012$ に対して $\pm 0.034$・$\pm 0.077$）、これはブロック末尾に印字された重みの分散ペナルティそのものです。3点の厳密内挿は $\lVert w \rVert^2 = 19$、最小二乗直線は $2.33$ で、分散で8倍、標準偏差で $2.9$ 倍です。印字された1回の実行は、2次のばらつきが悪い方向へ振れたシードだったということです。教訓は自由度ではなく分散についてであり、そして1回のモンテカルロ実行から方法論的な結論を出してはいけないということです。

**分散のコストは明示的です。** Richardsonの重みは2点・3点・4点のノイズスケールに対して $(2, -1)$、$(3, -3, 1)$、$(4, -6, 4, -1)$ で、$\lVert w \rVert_2 = 2.24, 4.36, 8.31$ です。分散 $\sigma^2$ の独立な推定値は分散 $\sigma^2 \lVert w \rVert^2$ に結合するので、2点から4点へ移ると必要ショット数は $(8.31/2.24)^2 \approx 14$ 倍になります。しかもこれは、ノイズの多い回路では信号そのものが小さくなり各 $\langle A \rangle_{\lambda_i}$ の推定自体が難しくなるという事実の上に乗ります。緩和はバイアスの問題をサンプリングの問題に変換し、5.4節が示すようにサンプリングの問題はすでに拘束条件だったのです。

### 誤り訂正と閾値

誤り訂正は論理量子ビットを冗長に符号化し、**シンドローム** — 符号化された状態を明かさずに誤りが起きたかを明かす観測量 — を測定します。表面符号は超伝導ハードウェアの最有力候補です。2次元格子上に物理量子ビットを並べ最近接のパリティ検査を行い、パッチを大きくすることで符号距離 $d$ を増やせ、復号器がシンドロームの履歴から最も確からしい誤りを推定します。

中心的な事実は**閾値定理**です。臨界的な物理誤り率 $p_{\text{th}}$ を下回れば、論理誤り率は符号距離について指数関数的に減少します：

$$ p_L \approx A\left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2} $$

閾値を上回ると、量子ビットを増やすことは事態を**悪化**させます。追加された各量子ビットが、符号が訂正できる以上の誤りを持ち込むからです。標準的な回路レベルノイズモデルでの表面符号の閾値は $10^{-2}$ のオーダーであり、係数 $A$ と正確な指数は符号・復号器・ノイズモデルに依存します。したがって以下はすべて桁のオーダーの言明であり、そのように扱うべきものです。

回転表面符号は論理量子ビット1個あたり $2d^2 - 1$ 個の物理量子ビットを使います。

Code Example 5: 訂正・深さ・測定の予算

```python
"""Chapter 5, Example 5: order-of-magnitude budgets for correction, depth
and measurement. Self-contained: only arithmetic, no simulator needed."""
import numpy as np

P_THRESHOLD = 1e-2      # representative surface-code threshold, order of magnitude
A_PREFACTOR = 0.1       # dimensionless prefactor, order of magnitude


def logical_error(p_phys, d, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Surface-code scaling p_L ~ A (p/p_th)^((d+1)/2).
    Order of magnitude only: the prefactor and the threshold are
    code-, decoder- and noise-model dependent."""
    return A * (p_phys / p_th) ** ((d + 1) / 2)


def required_distance(p_phys, target, p_th=P_THRESHOLD, A=A_PREFACTOR):
    """Smallest odd code distance reaching a target logical error rate.

    The comparison carries a relative tolerance because p_L is a ratio raised
    to a large power: a distance that meets the target *exactly* lands a few
    ulps above it in binary floating point (0.1 * 0.1**5 evaluates to
    1.0000000000000004e-06), and a bare `<= target` would reject it and return
    the next distance up."""
    if p_phys >= p_th:
        return None                # at or above threshold, more qubits do not help
    for d in range(3, 201, 2):
        if logical_error(p_phys, d, p_th, A) <= target * (1 + 1e-9):
            return d
    return None


def physical_per_logical(d):
    """Rotated surface code: 2 d^2 - 1 physical qubits per logical qubit."""
    return 2 * d * d - 1


print("A. Where the error-correction threshold bites")
print("=" * 74)
print(f"  assumed threshold p_th = {P_THRESHOLD:.0e}, prefactor A = {A_PREFACTOR}")
print(f"\n  {'p_phys':>9} {'d = 3':>10} {'d = 7':>10} {'d = 11':>10} "
      f"{'d = 21':>10} {'d = 31':>10}")
for p_phys in (2e-2, 1e-2, 5e-3, 1e-3, 3e-4, 1e-4):
    row = "  ".join(f"{logical_error(p_phys, d):10.2e}" for d in (3, 7, 11, 21, 31))
    print(f"  {p_phys:9.0e} {row}")
print("\n  Above threshold, increasing d makes the logical error WORSE.")
print("  Below threshold it falls exponentially in d. That is the whole game.")

print("\nB. Qubit overhead for a target logical error rate")
print("=" * 74)
print(f"  {'p_phys':>9} {'target p_L':>12} {'distance d':>11} "
      f"{'physical/logical':>17} {'100 logical qubits':>19}")
for p_phys in (1e-3, 3e-4, 1e-4):
    for target in (1e-6, 1e-10, 1e-15):
        d = required_distance(p_phys, target)
        if d is None:
            print(f"  {p_phys:9.0e} {target:12.0e} {'unreachable':>11}")
            continue
        per = physical_per_logical(d)
        print(f"  {p_phys:9.0e} {target:12.0e} {d:11d} {per:17,d} {100*per:19,d}")

print("\nC. Gate budget without error correction")
print("=" * 74)
print("  A circuit carries information only while (gates) x (error rate) << 1.")
print(f"  {'per-gate error':>15} {'gates at error 1':>18} "
      f"{'gates at error 0.1':>20}")
for p in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-10, 1e-12):
    print(f"  {p:15.0e} {1/p:18,.0f} {0.1/p:20,.0f}")

print("\n  Circuit sizes that materials problems actually ask for:")
for label, gates in (("2-site Hubbard, one Trotter step", 1e1),
                     ("2-site Hubbard, phase estimation to 1e-3", 3.4e7),
                     ("20-orbital active space, VQE ansatz", 1e4),
                     ("50-orbital active space, phase estimation", 1e11),
                     ("FeMoco-scale phase estimation, order of mag", 1e11)):
    print(f"    {label:45s}: ~{gates:8.0e} gates  -> needs p < {0.1/gates:.0e}")

print("\nD. Measurement cost of chemical accuracy")
print("=" * 74)
target = 1.6e-3          # Hartree; 1 kcal/mol, the usual 'chemical accuracy'
print(f"  target precision = {target:.1e} Ha (1 kcal/mol)")
print("  shots ~ (sum of term variances) / epsilon^2, variance ~ 1 per term.")
print("  This is the BEST case: it assumes the terms are perfectly grouped into")
print("  one commuting family.  Measuring each Pauli term in its own circuit")
print("  costs (sum_j |c_j| sigma_j)^2 / epsilon^2 instead, which is larger.")
print(f"\n  {'orbitals':>9} {'Pauli terms ~n^4':>17} {'shots (best case)':>18} "
      f"{'time at 1e4/s':>16}")
for n_orb in (4, 10, 20, 50, 100):
    n_terms = n_orb ** 4
    shots = n_terms / target ** 2
    seconds = shots / 1e4
    years = seconds / 3.156e7
    t = f"{years:.2e} yr" if years > 1 else f"{seconds:.2e} s"
    print(f"  {n_orb:9d} {n_terms:17,d} {shots:18.3e} {t:>16}")

print("\n  Precision is quadratically expensive:")
for eps in (1e-1, 1e-2, 1.6e-3, 1e-4):
    print(f"    epsilon = {eps:8.1e} Ha  ->  shots x {(1/eps)**2:12.3e} per term")

print("\nE. The three budgets side by side")
print("=" * 74)
print("  A NISQ calculation must satisfy all three at once:")
print("    (1) depth    : gates x error rate << 1")
print("    (2) width    : qubits <= device size, with no correction overhead")
print("    (3) sampling : shots x circuit time <= available wall-clock time")
print("\n  Worked case: 20-orbital active space (40 qubits), VQE")
n_orb, gates = 20, 1e4
n_terms = n_orb ** 4
shots = n_terms / (1.6e-3) ** 2
print(f"    qubits               : {2*n_orb}")
print(f"    Pauli terms          : {n_terms:,}")
print(f"    circuit gates        : {gates:.0e}  -> needs p < {0.1/gates:.0e}")
print(f"    shots for 1 kcal/mol : {shots:.2e}  (best case, perfect grouping)")
print(f"    at 1e4 circuits/s    : {shots/1e4/3.156e7:.2e} years"
      f" for ONE energy evaluation")
print(f"    a geometry optimization needs ~1e2 evaluations:"
      f" {1e2*shots/1e4/3.156e7:.2e} years")
```

```text
A. Where the error-correction threshold bites
==========================================================================
  assumed threshold p_th = 1e-02, prefactor A = 0.1

     p_phys      d = 3      d = 7     d = 11     d = 21     d = 31
      2e-02   4.00e-01    1.60e+00    6.40e+00    2.05e+02    6.55e+03
      1e-02   1.00e-01    1.00e-01    1.00e-01    1.00e-01    1.00e-01
      5e-03   2.50e-02    6.25e-03    1.56e-03    4.88e-05    1.53e-06
      1e-03   1.00e-03    1.00e-05    1.00e-07    1.00e-12    1.00e-17
      3e-04   9.00e-05    8.10e-08    7.29e-11    1.77e-18    4.30e-26
      1e-04   1.00e-05    1.00e-09    1.00e-13    1.00e-23    1.00e-33

  Above threshold, increasing d makes the logical error WORSE.
  Below threshold it falls exponentially in d. That is the whole game.

B. Qubit overhead for a target logical error rate
==========================================================================
     p_phys   target p_L  distance d  physical/logical  100 logical qubits
      1e-03        1e-06           9               161              16,100
      1e-03        1e-10          17               577              57,700
      1e-03        1e-15          27             1,457             145,700
      3e-04        1e-06           7                97               9,700
      3e-04        1e-10          11               241              24,100
      3e-04        1e-15          19               721              72,100
      1e-04        1e-06           5                49               4,900
      1e-04        1e-10           9               161              16,100
      1e-04        1e-15          13               337              33,700

C. Gate budget without error correction
==========================================================================
  A circuit carries information only while (gates) x (error rate) << 1.
   per-gate error   gates at error 1   gates at error 0.1
            1e-02                100                   10
            1e-03              1,000                  100
            1e-04             10,000                1,000
            1e-05            100,000               10,000
            1e-06          1,000,000              100,000
            1e-10     10,000,000,000        1,000,000,000
            1e-12  1,000,000,000,000      100,000,000,000

  Circuit sizes that materials problems actually ask for:
    2-site Hubbard, one Trotter step             : ~   1e+01 gates  -> needs p < 1e-02
    2-site Hubbard, phase estimation to 1e-3     : ~   3e+07 gates  -> needs p < 3e-09
    20-orbital active space, VQE ansatz          : ~   1e+04 gates  -> needs p < 1e-05
    50-orbital active space, phase estimation    : ~   1e+11 gates  -> needs p < 1e-12
    FeMoco-scale phase estimation, order of mag  : ~   1e+11 gates  -> needs p < 1e-12

D. Measurement cost of chemical accuracy
==========================================================================
  target precision = 1.6e-03 Ha (1 kcal/mol)
  shots ~ (sum of term variances) / epsilon^2, variance ~ 1 per term.
  This is the BEST case: it assumes the terms are perfectly grouped into
  one commuting family.  Measuring each Pauli term in its own circuit
  costs (sum_j |c_j| sigma_j)^2 / epsilon^2 instead, which is larger.

   orbitals  Pauli terms ~n^4  shots (best case)    time at 1e4/s
          4               256          1.000e+08       1.00e+04 s
         10            10,000          3.906e+09       3.91e+05 s
         20           160,000          6.250e+10       6.25e+06 s
         50         6,250,000          2.441e+12      7.74e+00 yr
        100       100,000,000          3.906e+13      1.24e+02 yr

  Precision is quadratically expensive:
    epsilon =  1.0e-01 Ha  ->  shots x    1.000e+02 per term
    epsilon =  1.0e-02 Ha  ->  shots x    1.000e+04 per term
    epsilon =  1.6e-03 Ha  ->  shots x    3.906e+05 per term
    epsilon =  1.0e-04 Ha  ->  shots x    1.000e+08 per term

E. The three budgets side by side
==========================================================================
  A NISQ calculation must satisfy all three at once:
    (1) depth    : gates x error rate << 1
    (2) width    : qubits <= device size, with no correction overhead
    (3) sampling : shots x circuit time <= available wall-clock time

  Worked case: 20-orbital active space (40 qubits), VQE
    qubits               : 40
    Pauli terms          : 160,000
    circuit gates        : 1e+04  -> needs p < 1e-05
    shots for 1 kcal/mol : 6.25e+10  (best case, perfect grouping)
    at 1e4 circuits/s    : 1.98e-01 years for ONE energy evaluation
    a geometry optimization needs ~1e2 evaluations: 1.98e+01 years
```

**着目点。** パートAには誤り耐性の論理のすべてが1つの表に入っています。$p = 10^{-2}$ の行は $0.1$ で平坦です。ちょうど閾値では符号距離が何もしません。$p = 2\times10^{-2}$ の行は $d$ とともに**上昇**し、$d = 31$ で6550に達します。閾値を上回れば、大きな符号は悪い符号なのです。$p = 10^{-4}$ の行は $d = 31$ で $10^{-33}$ まで落ちます。閾値を下回ることは量的な改善ではなく、領域の質的な変化です。

パートBがそれに値札を付けます。$p = 10^{-3}$ では論理誤り率 $10^{-10}$ に距離17が必要で、これは論理量子ビット1個あたり577個の物理量子ビット、つまり控えめな100論理量子ビットの機械には57,700個の物理量子ビットです。物理誤り率を $3\times10^{-4}$ に改善すればこれは24,100個に減ります。ハードウェアのグループがゲート忠実度をこれほど必死に追う理由がこれです。物理誤り率が3分の1になるたびに量子ビット数がおよそ3分の1になり、それが複利で効きます。

パートCは、誰かが近未来の応用を主張したときに引用すべき箇所です。回路は（ゲート数）×（誤り率）が1より十分小さいあいだだけ意味を持ちます。**2サイト**Hubbard模型に対する位相推定は $3.4 \times 10^7$ ゲートを要し（第4章Code Example 3、同章の楽観的な数え方による）、したがって $p < 3\times10^{-9}$ です。訂正なしのハードウェアより6桁先です。50軌道の活性空間は $p < 10^{-12}$ を要し、FeMoco規模の見積りも同じ場所に着きます。ゲート数が $10^{10}$ から $10^{11}$ のオーダーなので、やはり $p \lesssim 10^{-12}$ を要します。これらの数は誤り訂正なしには到達できず、まさにそれがFeMocoが誤り耐性の論拠である理由です。なおこのリストの指数はすべて桁として受け取ってください。公表されたFeMocoの見積りはすでに数桁動いています。

パートDは人々が忘れる制約です。**完璧で**ノイズのない量子コンピュータをもってしても、しかもPauli項が1つの可換族に完全にグループ化できると仮定してさえ（印字された表が前提にしている最良の場合です）、20軌道の活性空間のVQEが化学精度に達するには $6\times10^{10}$ 回の回路実行が必要で、毎秒 $10^4$ 回なら1つのエネルギーに2.4か月の連続運転です。100個のエネルギーを要する構造最適化なら20年です。測定コストは**アルゴリズム**の性質であってハードウェアの性質ではありません。$\varepsilon \propto 1/\sqrt{N}$ と $O(M^4)$ の項数から従うものです。より良い測定戦略（可換な項のグループ化、classical shadows、低ランク因子分解）は係数を大幅に減らしますが、$1/\varepsilon^2$ のスケーリングは法則です。

パートEが3つを合わせ、結論は居心地の悪いものです。深さを脇に置いてさえ、化学的に興味深い活性空間へのVQEは今日のハードウェアで難しいだけでなく、サンプリングで期待値を推定するいかなるハードウェアでも難しいのです。これは、精度コストが $1/\varepsilon^2$ ではなく $1/\varepsilon$ である（ただし回路の深さも $1/\varepsilon$ で増える）位相推定への最も強い論拠の1つであり、したがってNISQアルゴリズムを最適化するより誤り耐性を追求すべきことの最も強い論拠の1つです。

* * *

## 5.4 冷静な評価

本節が本章の中心です。上のすべては測定でした。ここは判断であり、できるかぎり率直に述べます。

### NISQ装置が今日、材料研究に対してできること

  * **答えが既知の模型で量子アルゴリズムをベンチマークする。** 4量子ビットのIsingやHubbard模型でVQEを走らせて厳密対角化と比べることは真に有用です。ソフトウェアスタックを検証し、ハードウェアを特性づけ、人を訓練します。新しい物理は何も生みませんし、生んだかのように記述してはいけません。
  * **ハードウェアの物理を特性づける。** ランダム化ベンチマーキング、サイクルベンチマーキング、クロスエントロピーベンチマーキング、トモグラフィは実在の量子系の実在の測定であり、誤り率が改善する経路です。装置が実験対象なのです。
  * **誤り緩和を開発し試験する。** ZNEや確率的誤りキャンセルのような技術は、真の答えが独立に既知である場所、すなわち小さな系でしか検証できません。
  * **古典-量子の界面を探る。** 埋め込み手法（DMET、DFT埋め込み）、活性空間の選択、測定削減の戦略、ansatzの設計はすべて古典的な研究課題であり、その答えは後で必要になるもので、今研究できます。
  * **格子模型のアナログシミュレーション。** 古典的到達範囲を超える結果への信頼できる主張をもつ唯一の領域です。冷却原子とイオントラップのプラットフォームは、厳密な古典シミュレーションが不可能な規模で量子スピン模型やHubbard模型のダイナミクスを測定してきました。これはゲート型の量子コンピューティングではなく、結果は化学ではなく物理ですが、実在します。

### NISQ装置ができないこと

  * **実在の材料でDFTに勝つ。** DFTは数百から数千個の原子を有用な精度で扱います。NISQ装置はひと握りの軌道をノイズ制限された精度で扱います。量子装置が勝つ重なりは存在しません。
  * **化学的に興味深い系で化学精度に達する。** 上のパートD。ノイズを考える前に、ショット数だけがそれを禁じます。
  * **何かに対して位相推定を走らせる。** 深さの要求が訂正なしのハードウェアを5〜10桁超えます。
  * **現実的な系のサイズを扱う。** 50軌道の触媒活性空間は100量子ビットと $6\times10^6$ 個のPauli項です。深さもショット予算も古典最適化も手が届きません。
  * **化学・材料で検証された量子優位性を届ける。** 本稿執筆時点で、量子ハードウェア上で実行され、かつ (a) 最良の古典手法より正確で、(b) 独立に検証され、(c) 科学的に有用である、分子や材料の性質の計算は存在しません。それに反する主張は繰り返し、数か月のうちに改良された古典手法に追いつかれるか追い越されてきました。

### 量子優位性の主張の読み方

ここ数年のパターンは一貫しています。量子実験が古典的到達範囲を超える課題を実行したと主張し、数か月のうちに古典アルゴリズム — しばしばサンプルされた回路の特定の構造を利用するテンソルネットワーク法 — が同じ結果を再現するのです。これはスキャンダルではなく、境界が実際にどこにあるかをこの分野が確立する仕方です。しかし主張は注意深く読まなければならない、ということでもあります。

問うべき質問を順に挙げます。

  1. **その課題は有用か、それとも作られたものか。** ランダム回路サンプリングやボソンサンプリングは古典計算機に難しくなるよう設計されており、他の何にも有用ではありません。それを実証することは正当な物理のマイルストーンであり、化学については何も語りません。
  2. **古典的なベースラインは何で、誰が計算したか。** 素朴な古典アルゴリズムとの比較は比較ではありません。既知の最良の古典手法が使われたか、同等の工学的労力が与えられたか、そして古典ベースラインの作者がその枠組みに同意しているかを問いましょう。
  3. **量子の結果は検証されたか。** 答えが古典的に確認できないなら、正しさはどう確立されたのか。より小さな検証可能な事例からの外挿が通常の手法であり、それは仮定であって証明ではありません。
  4. **どれだけの精度が達成されたか。** 0.1 Hartree精度の量子エネルギーは化学の結果ではありません。化学精度は $1.6\times10^{-3}$ です。
  5. **何をどれだけのコストで緩和したか。** 重い後処理は、下敷きの量子状態の忠実度が無視できるのに正解に近い数値を出せます。生の結果とショット数を求めましょう。
  6. **手法はスケールするか。** 多くの実証は、対称性、小さなサイズ、問題固有の技巧に依存しており、それらは大きなスケールでは消えます。2倍のサイズでの資源見積りを問いましょう。

### 発表ではなく原理に基づく判断基準

装置の発表はすぐに古びますし、量子ビット数に紐づいた評価は公表された時点で時代遅れです。より良い方法は、**任意の**提案された量子計算について、いずれも問題の物理から答えられる3つの定量的な問いを立てることです。

問い | 計算すべき量 | もっともらしさの閾値
---|---|---
回路は十分浅いか | （ゲート数）×（ゲートあたり誤り率） | 1より十分小さく、理想的には0.1以下
サンプリングは払えるか | （Pauli項数）/ $\varepsilon^2$ ×（回路時間）、最良の場合 | 利用可能な実時間以下
古典的に難しいか | 最良の古典手法のコストと精度 | 古典手法が単に遅いのではなく破綻すること

提案が3つのいずれかを満たさないなら、プレスリリースで発表される種類のハードウェア改善では救えません。必要なのはアルゴリズムの変更か、時代の変更です。逆に3つすべてを通る提案は、誰が言っているかに関わらず真剣な注意に値します。

### 何が絵を変えるか

進歩がどのようなものかを明確にするため、上の評価を真に変える展開を挙げます。

  * **スケールした状態での物理2量子ビット誤り率 $10^{-4}$ 以下**。これで距離9の表面符号と $10^{-10}$ の論理誤り率が、論理量子ビットあたり161個の物理量子ビットで手に入ります。
  * **構成要素の物理量子ビットより低い誤り率をもつ論理量子ビットの、多ラウンドにわたる持続的な実証** — 「実験としての誤り訂正」から「インフラとしての誤り訂正」への移行です。
  * **$1/\varepsilon^2$ ではなく $1/\varepsilon$ の精度スケーリングをもち、NISQの深さで動くアルゴリズム**。これは測定のボトルネックを取り除きます。この項目は他と性質が違い、単に未解決なのではなく定理によって制約されています。補間的な手法（$\alpha$-QPE、ロバスト振幅推定・最尤振幅推定）は実際に存在し、高速化の一部を買えます。すなわち $\alpha \in [0,1]$ に対して総クエリコスト $\sim \varepsilon^{-(1+\alpha)}$、最大コヒーレント深さ $\sim \varepsilon^{-(1-\alpha)}$ を達成します。しかしこのトレードオフ自体が結果の内容です。最大深さ $D$ を**固定**すると、総ショット数 $N$ から得られる精度は $\varepsilon \gtrsim 1/(D\sqrt{N})$ で抑えられ、完全なHeisenbergスケーリング $1/\varepsilon$ には深さ $\propto 1/\varepsilon$ が必要です。精度はコヒーレンスで買うものであり、コヒーレンスこそNISQに欠けているものです。つまりこれは独立した希望ではなく、誤り訂正が必要だという主張の言い換えなのです。
  * **古典手法が再現できない材料の性質の、検証された量子計算**。しかも量子コンピューティングのコミュニティ外の誰かが気にする問題について。
  * **誤り耐性の化学の資源見積りが、さらに3〜4桁減ること**。この10年の趨勢の継続です。

最初の2つは明確な道筋のある工学的問題です。3つ目は不可能かもしれません。4つ目が実際の目標です。5つ目は理論家が最も貢献できる場所です。

* * *

## 5.5 エコシステム

実運用のために自分のシミュレータを書くことはないでしょう。3つのオープンソースフレームワークが支配的で、両者の違いは能力よりも思想にあります。バージョン間で変わるAPIではなく、位置づけを述べます。

フレームワーク | 出自 | 思想 | 最も得意なところ
---|---|---|---
Qiskit | IBM | 回路中心、ハードウェア志向、大きなエコシステム | IBMハードウェアでの実行、トランスパイル、誤り緩和モジュール
Cirq | Google | ゲートスケジューリングと装置トポロジーの明示的制御 | ハードウェアを意識した回路構成、NISQ実験
PennyLane | Xanadu | 微分可能プログラミング、自動微分との統合 | 変分アルゴリズム、量子機械学習、ハイブリッド勾配

その周囲に、カテゴリで知っておくべき専門ツールがあります。フェルミオンハミルトニアンを生成して量子ビット写像を適用する量子化学インタフェース（第4章Code Example 2が担った役割）、高性能な状態ベクトルおよびテンソルネットワークシミュレータ（ハードウェアが必要だと主張する前に比較すべき対象）、そしてZNEや確率的誤りキャンセルを実装した誤り緩和ライブラリです。

実務的な推奨が2つ。第1に、**3つを浅く学ぶよりも1つをきちんと学ぶこと**。概念は移りますが、APIは移りません。第2に、**必ず古典シミュレータを先に走らせること**。30量子ビットの状態ベクトルシミュレーションで問いに答えられるなら、ハードウェアが加えるのはノイズだけです。

### 参考文献と発展的な読み物

分類 | 入口としての推奨
---|---
教科書 | Nielsen & Chuang, *Quantum Computation and Quantum Information*（標準的な参考書）、Preskillの量子計算講義ノート（無償で入手可能）
NISQの枠組み | Preskill, "Quantum Computing in the NISQ era and beyond" (2018) — この時代を名づけその限界を述べた論文
量子コンピュータ上の量子化学 | Cao et al. の *Chemical Reviews* 総説、McArdle et al. の *Reviews of Modern Physics* 総説
変分アルゴリズム | Cerezo et al. の *Nature Reviews Physics* 総説、McClean et al. (2018) に始まるbarren plateauの文献
誤り訂正 | Fowler et al., "Surface codes: towards practical large-scale quantum computation"、Terhalの量子メモリの誤り訂正の総説
誤り緩和 | Cai et al. の量子誤り緩和の総説、実装についてはMitiqのソフトウェア論文
古典側の競争 | Schollwöck のDMRG総説、量子スプレマシー実験の古典シミュレーションの文献（境界が実際に引かれている場所）
資源見積り | 歴代のFeMoco資源見積り論文を年代順に読むこと — 誤り耐性のコストを何が支配するかについて入手可能な最良の教材

* * *

## 5.6 シリーズ総括と学習ロードマップ

### 本シリーズが扱ったこと

章 | 内容 | できるようになったこと
---|---|---
1 | 量子ビット、重ね合わせ、測定、テンソル積 | 多量子ビット状態を表現しサンプリングする。$2^n$ が資源でも呪いでもある理由を説明する
2 | ゲート、回路、エンタングルメント、普遍性、シミュレータ | 任意のユニタリを任意の量子ビットに作用させる。エンタングルメントを定量化する。Pauli指数関数をコンパイルする
3 | 変分量子固有値ソルバー | ansatzを構成し、Pauli分解された観測量を測定し、parameter-shift勾配で完全なVQEを走らせる
4 | 第二量子化、Jordan-Wigner、モデルハミルトニアン | フェルミオン問題を量子ビットに写して検証する。IsingとHubbard模型を厳密対角化する。VQEを厳密解と比較する
5 | ノイズ、緩和、訂正、評価 | ノイズのある回路をシミュレートする。忠実度の減衰を測る。ZNEを適用する。深さ・幅・ショットを見積もる。主張を評価する

第1-2章で作ったミニシミュレータが、以降のすべての計算を担いました。それを作ることの意味はそこにあります。99行のNumPyがあれば本シリーズのすべての量子アルゴリズムを再現し検証し理解できるのであり、そのやり方で再現できないものは、おそらくまだ理解できていないのです。

### 次に進む場所

やりたいことに応じて3つの道があります。

**この分野を横目で見ておきたい材料研究者なら。** 本質的なところは終わりました。最も価値の高い続きは強相関系の**古典**手法です。量子計算の結果はそれに勝たなければならないからです。DMRGと行列積状態、量子モンテカルロと符号問題、動的平均場理論、埋め込み手法。プレプリントよりも年次レビューを読み、面白そうに見えるものには5.4節の3つの基準を適用してください。

**量子アルゴリズムの研究をしたいなら。** 理論を深めましょう。量子位相推定とその現代的な後継（qubitization、量子信号処理）、Trotterを超えるHamiltonianシミュレーション、barren plateauの文献と訓練可能性についてそれが語ること、classical shadowsを含む測定削減の戦略、そして量子誤り訂正そのもの。本道場の前提コース — [線形代数とテンソル解析](<../linear-algebra-tensor/index.html>)、[量子力学入門](<../quantum-mechanics/index.html>)、[量子場の理論入門](<../quantum-field-theory-introduction/index.html>) — に数学があります。

**作りたいなら。** フレームワークを1つ選び、自分で生成した積分から自分で選んだ分子のVQEを実装し、実機で走らせ、自分の厳密対角化と比べてください。それからZNEを実装してどれだけ助けになるかを測ってください。自分が選んだ問題について自分で測ったシミュレータと実機の差は、どんな総説よりも多くを教えます。

### 推奨する順序

段階 | 焦点 | 概算の労力
---|---|---
1 | 本シリーズのすべてのコード例を、見ずにゼロから再現する | 2〜3週間
2 | シミュレータを拡張する：密度行列、2つ目のノイズモデル、より良い最適化器 | 2〜3週間
3 | 古典の強相関手法を1つ（スピン鎖のDMRGが理想的） | 1〜2か月
4 | フレームワーク1つ、実在の分子1つ、実機での実行1回 | 1〜2か月
5 | 誤り耐性の資源見積りの文献を年代順に読む | 継続

段階3は人が飛ばしがちで、飛ばすべきでないものです。DMRGが1次元問題を事実上厳密に解く理由と、2次元で破綻する理由を理解することは、量子コンピューティングがどこで貢献できるかを判断するための最良の準備です。

### 締めくくりに

材料科学のための量子コンピューティングは今、優れた物理と、実在する工学的進歩と、明確な長期目標と、近未来の応用の不在をあわせもつ分野です。この4つの言明はすべて同時に真であり、それらを同時に保持できることが、この分野のマーケティングにも切り捨てにも寄りかからず理解している者の印です。

有用な姿勢は熱狂でも懐疑でもなく、**リテラシー**です。3つの予算を計算し、古典的なベースラインを同定し、自分自身の結論に到達する能力です。それができるなら — そして本章のあとであなたはできます — この分野の主張を、真であると判明するものも含めて、これからのキャリアを通じて評価できるでしょう。

* * *

## 演習

本章のコードを手元に置いて取り組んでください。各問のあとに解答があります。

#### 演習1: コヒーレンス時間

ある装置が $T_1 = 80\ \mu\text{s}$、$T_2 = 120\ \mu\text{s}$ と報告しました。(a) この報告は内部で整合していますか。適用される上限を述べ、それが誤って記憶されやすい理由も述べてください。(b) $T_1 = 80\ \mu\text{s}$、$T_2 = 40\ \mu\text{s}$ のとき $T_\phi$ を求めてください。(c) 2量子ビットゲートが300 nsかかるとき、$T_2$ の中におおよそ何個の逐次ゲートが収まり、それは5.2節のゲート予算とどう比べられますか。

<details><summary>解答</summary>
<p>(a) 整合しています。そこがこの問いの要点です。上限は \(T_2 \le 2T_1\) であり、\(T_\phi &gt; 0\) を伴う \(1/T_2 = 1/(2T_1) + 1/T_\phi\) から従います。ここでは \(2T_1 = 160\ \mu\mathrm{s}\)、\(T_2 = 120\ \mu\mathrm{s} &lt; 160\ \mu\mathrm{s}\) なので、何も問題ありません。罠は上限を \(T_2 \le T_1\) と誤って記憶することです。\(T_2 &gt; T_1\) はまったく物理的であり、純粋な位相緩和がほとんどなく緩和で制限された装置がそう見えます。ここでは \(1/T_\phi = 1/120 - 1/160\)、すなわち \(T_\phi = 480\ \mu\mathrm{s}\) で \(T_1\) の4倍 — 位相緩和はほぼ無視できます。</p>
<p>(b) \(1/T_\phi = 1/T_2 - 1/(2T_1) = 1/40 - 1/160 = 0.025 - 0.00625 = 0.01875\ \mu\mathrm{s}^{-1}\) なので \(T_\phi = 53.3\ \mu\mathrm{s}\)。位相緩和が支配的です。</p>
<p>(c) \(T_2\) の中には \(40\ \mu\mathrm{s} / 300\ \mathrm{ns} \approx 133\) 個のゲートが収まります。ただし \(T_2\) に収まることと使えることは別であり、比較は5.2節の正しい行に対して行わなければなりません。この装置のコヒーレンス制限ゲート誤差は \(\tau_g/T_2 = 300\ \mathrm{ns}/40\ \mu\mathrm{s} = 7.5\times10^{-3}\) で、\(p = 5\times10^{-3}\) の行（\(F = 0.5\) で148箇所、\(F = 0.9\) で22箇所）と \(p = 10^{-2}\) の行（84と13）の間に入ります。補間すれば \(F = 0.5\) で約92、\(F = 0.9\) で約14です。したがって \(T_2\) からの数と忠実度予算は50%程度の精度で一致し（133対約92）、<em>使える</em>深さはそのどちらよりも10倍小さいということになります。この装置に \(p = 10^{-3}\) の行（705と107）を当てるのは1桁誤りです。あの行はゲート誤差が10倍良い装置を表しています。</p>
</details>

#### 演習2: 誤った軌跡則

Code Example 1 の `phase_damping_trajectory` を、$Z$ を $(1-\sqrt{1-\lambda})/2$ ではなく $\lambda/2$ の確率で作用させる版に置き換えてください。(a) $\lambda = 0.4$ で軌跡のBlochベクトルはどうなりますか。(b) 印字された出力のどの量が誤りを最も明確に暴きますか。(c) このバグがとりわけ危険なのはなぜですか。

<details><summary>解答</summary>
<p>(a) 確率 \(q\) の \(Z\) キックは非対角要素を \((1-2q)\) 倍します。\(q = \lambda/2 = 0.2\) ならコヒーレンスは \(1 - 0.4 = 0.6\) になり、Blochベクトルは正しい \(\sqrt{1-\lambda} = 0.7746\) ではなく \((0.60, 0, 0)\) になります。</p>
<p>(b) 純度です。正しいチャネルは \(\mathrm{Tr}(\rho^2) = 0.800\) を与えますが、誤ったほうは \(0.68\) を与えます。Blochベクトルも違いますが、純度は状態について2次であるため過剰な混合に二重に敏感で、より鋭い診断になります。</p>
<p>(c) 誤ったモデルでも、もっともらしい率で指数関数的減衰が出るからです。定性的な特徴はすべて生き残ります。コヒーレンスは減衰し、占有は動かず、チャネルはトレースを保存します。誤っているのは数値的な率だけです — ただしその誤りは、どの \(\lambda\) でもちょうど2倍です。正しい規則は1回の適用ごとにコヒーレンスを \(\sqrt{1-\lambda}\) 倍し、誤った規則は \((1-\lambda)\) 倍します。そして \(\ln(1-\lambda) = 2\ln\sqrt{1-\lambda}\) が恒等的に成り立つので、抽出される位相緩和率は \(\lambda\) によらず正確に2倍大きくなります。（\(\sim\)23% というのは1回適用後のコヒーレンスの誤差、0.7746に対する0.60のことです。<em>率</em>の誤りは100%です。）2倍の誤りは、減衰曲線から読む \(T_2\) や忠実度の推定にそのまま伝播します。だからこそ Code Example 1 が存在するのです。ノイズモデルは何かに使う前に厳密なチャネルと照合しなければなりません。</p>
</details>

#### 演習3: シミュレーションせずに忠実度を予測する

Code Example 3 の規則 $F \approx \exp(-N_{\text{sites}} p\, d)$ を使って、(a) $p = 2\times10^{-3}$ での10量子ビット・20層のhardware-efficient回路の忠実度を見積もってください。(b) $F = 0.9$ で10量子ビットが支えられる深さはいくらですか。(c) $1/2^n$ の床が見積りに効かなくなるのはどの $n$ からですか。

<details><summary>解答</summary>
<p>(a) \(N_{\mathrm{sites}} = n + 2(n-1) = 10 + 18 = 28\) が1層あたりです。総ノイズ箇所は \(28 \times 20 = 560\)。\(F \approx e^{-560 \times 2\times10^{-3}} = e^{-1.12} = 0.33\)。振幅のおよそ3分の1が生き残ります — すでにぎりぎりです。</p>
<p>(b) \(F = 0.9\) には \(N_{\mathrm{sites}} p\, d = \ln(1/0.9) = 0.105\) が必要なので \(d = 0.105/(28 \times 2\times10^{-3}) = 1.9\) 層、つまり2層です。これが「NISQ」の実務的な意味です。誤り率 \(2\times10^{-3}\) の10量子ビット装置は、有用な忠実度では2層の回路を支えます。</p>
<p>(c) 床が効くのは \(e^{-N_{\mathrm{sites}} p d}\) が \(2^{-n}\) に近づくとき、すなわち \(N_{\mathrm{sites}} p d \gtrsim n \ln 2\) のときです。\(n = 4\) なら \(2.8\) であり、Code Example 2 は \(p = 0.01\) でまさにその付近から比が1から外れることを示しています。より大きな \(n\) では床が指数関数的に低いので、単純な指数則がずっと広い範囲で成り立ちます。床は無関係になり、正直な読み方は「大きなノイズありの回路は床で飽和しているのではなく、単に無用である」となります。</p>
</details>

#### 演習4: 緩和はいつ価値があるか

Code Example 4 より、$p_0 = 0.002$ で線形ZNEは3つのノイズスケールを使ってバイアスを14.5倍減らしました。(a) ショット予算は何倍に増えなければなりませんか。(b) 同じ総ショット数を緩和なしの回路に費やしていたら統計誤差はいくらになり、そちらのほうが得な取引だったでしょうか。(c) ZNEが明らかに価値をもたないのはどんな状況ですか。

<details><summary>解答</summary>
<p>(a) 3つのノイズスケールは3つの別々の期待値推定を意味するので、外挿の重みを考える前でも3倍です。次に、実際に使った推定量の重みを使います。線形外挿は \(\lambda = 1, 2, 3\) を通る<em>最小二乗直線</em>であり、その切片の重みは \((4/3, 1/3, -2/3)\)、\(\lVert w \rVert^2 = 2.33\) です — Code Example 4 の最後の行に印字されている値です。外挿値の統計誤差を一定に保つには1点あたりさらに2.33倍が必要で、合計およそ7倍です。（Richardsonの重み \((3, -3, 1)\)、\(\lVert w \rVert^2 = 19\) が当てはまるのは<em>2次</em>外挿、すなわち3点の厳密内挿のほうで、そちらなら約57倍になります。）</p>
<p>(b) 7倍のショットを緩和なしの回路に費やすと<em>統計</em>誤差は \(\sqrt{7} = 2.6\) 倍改善しますが、\(+0.180\) の<em>バイアス</em>には何もしません。バイアスは平均化で消えません。したがってこの取引はバイアスが統計誤差を上回るときにこそ価値があり、それは多数のショットをかけた浅い回路では通常の状況です。緩和はサンプリングでは対処できない誤差を攻めるのです。</p>
<p>(c) 3つの場合があります。(i) すでに統計誤差がバイアスを支配しているとき — 生の回路にショットを増やすほうが良い。(ii) ノイズが強すぎて外挿が信頼できないとき：\(p_0 = 0.005\) では残留バイアスが \(+0.076\) でansatz誤差の8倍のままであり、\(\lambda = 3\) の点が脱分極の床に近づくにつれ当てはめの質が劣化します。(iii) 観測量のバイアスが \(\lambda\) について滑らかでないとき。これはコヒーレント（脱分極でない）誤りで起こります。ZNEはノイズ強度についての解析的な依存性を仮定しますが、コヒーレント誤りはそれを満たす必要がありません。</p>
</details>

#### 演習5: 誤り訂正の算術

Code Example 5 を使って、(a) $p = 5\times10^{-3}$ で $p_L = 10^{-9}$ に達する符号距離と量子ビットのオーバーヘッドを求めてください。(b) 有用なアルゴリズムが $10^{12}$ 回の論理ゲートを要するとします。必要な論理誤り率はいくらで、それを供給する物理誤り率と距離はいくらですか。(c) 仮定した係数 $A$ が比 $p/p_{\text{th}}$ より重要でないのはなぜですか。

<details><summary>解答</summary>
<p>(a) 出力のパートAより、\(p = 5\times10^{-3}\) では \(d = 31\) で \(p_L = 1.53\times10^{-6}\) です。\(10^{-9}\) に達するには \(p/p_\mathrm{th} = 0.5\) で \((p/p_\mathrm{th})^{(d+1)/2} = 10^{-8}\)、すなわち \((d+1)/2 = 8/\log_{10}2 = 26.6\) なので \(d = 53\)、\(2d^2 - 1 = 5{,}617\) 個の物理量子ビットが論理量子ビット1個あたり必要です。閾値の半分では誤り訂正は技術的には機能しますが破滅的に高価です。閾値の2倍以内で運用するのは成立する工学的な動作点ではありません。</p>
<p>(b) \(10^{12}\) 回の論理ゲートには、計算全体が正しい見込みをもつために \(p_L \lesssim 10^{-13}\)（より保守的には \(10^{-15}\)）が必要です。パートBより \(p_L = 10^{-15}\) には \(p = 10^{-3}\) で \(d = 27\)（論理あたり1,457個）、\(3\times10^{-4}\) で \(d = 19\)（721個）、\(10^{-4}\) で \(d = 13\)（337個）です。</p>
<p>(c) \(A\) は線形に入るのに対し \(p/p_\mathrm{th}\) は \((d+1)/2\) 乗で入るからです。\(A\) を0.1から0.01に変えても必要な距離は約2しか動きませんが、\(p/p_\mathrm{th}\) を0.1から0.03に変えると倍率で動きます。現実的な見積りがすべて閾値に対する物理誤り率を強調し、係数をノイズとして扱う理由であり、Example 5 のすべての結果を桁のオーダーとして提示している理由です。</p>
</details>

#### 演習6: 主張を評価する

あるプレプリントが次のように報告しています。「127量子ビットの超伝導プロセッサとゼロノイズ外挿を用い、20サイトHeisenberg鎖の基底状態エネルギーを厳密値の2%以内で計算した。これは古典計算機には扱えない計算である。」5.4節の6つの質問を適用してください。

<details><summary>解答</summary>
<p><strong>1. 有用か作られたものか。</strong> 20サイトHeisenberg鎖はよく研究された模型で未知の物理はありません。この計算は発見ではなくベンチマークです。それとしては正当です。</p>
<p><strong>2. 古典ベースラインは。</strong> 古典的に扱えないという主張は誤りです。20サイトのスピン1/2鎖は \(2^{20} = 10^6\) 次元 — 厳密対角化はノートPCで数秒、DMRGは数百サイトを計算機精度近くで扱います。「扱えない」という語は<em>回路</em>の総当たり状態シミュレーションを指しているように見えますが、それは別の主張であり、しかも関係のない主張です。</p>
<p><strong>3. 検証されたか。</strong> 暗に、はい。著者は厳密値と比較しており、だからこそ2%と言えます。良い実践であり、同時に扱えないという主張を反証しています。</p>
<p><strong>4. 精度は。</strong> 基底状態エネルギーの2%は化学精度から遠く、DMRGが与えるもの（1次元鎖なら相対 \(10^{-10}\) 以下）から遠いです。物理 — 臨界指数、相関関数 — を取り出すには、エネルギーの2%は使えません。</p>
<p><strong>5. 何を緩和したか。</strong> ZNEなので生の忠実度を求めるべきです。緩和なしの結果が30%外れていて後処理が2%にしたのなら、最終的な数値に量子状態はほとんど関与していません。ショット数と使ったノイズスケールも問いましょう。</p>
<p><strong>6. スケールするか。</strong> Heisenberg鎖はJordan-Wigner文字列なしで量子ビットに写り、\(O(N)\) 個の局所項をもつ — 考えうる最も易しい場合です。\(O(M^4)\) 項と非局所文字列をもつフェルミオンハミルトニアンについては何も語りません。</p>
<p><strong>判定。</strong> 立派なハードウェアのベンチマークに、支持できない枠づけが付いたものです。正しい要約はこうなります。「127量子ビットのプロセッサが誤り緩和とともに、古典的に厳密な結果を2%で再現し、装置性能の向上を実証した。」それは真の進歩であり公表に値します。扱えないという主張は値しません。</p>
</details>

* * *

## まとめ

### 要点

**1. ノイズの原因は少数で、モデルは単純である**

  * $T_1$（緩和）、$T_\phi$（位相緩和）、ゲート誤差、読み出し誤差でほとんどすべてが説明され、$1/T_2 = 1/(2T_1) + 1/T_\phi$ かつ $T_2 \le 2T_1$。
  * チャネルはKraus写像であり、脱分極チャネルが標準的な主力である。
  * 軌跡法はチャネルを純粋状態シミュレータ上で厳密に再現し、統計コストは $1/\sqrt{N}$ — しかも実機が与えるものを写している。
  * ノイズモデルは厳密なチャネルと照合し、Blochベクトルだけでなく純度も検査すること。本章の位相減衰則には $\lambda/2$ ではなく $q = (1-\sqrt{1-\lambda})/2$ が必要だった。

**2. 忠実度は指数関数的に減衰し、その率はシミュレーションなしで予測できる**

  * $F(d) \approx \exp(-N_{\text{sites}}\, p\, d)$、1層あたり $N_{\text{sites}} = n + 2(n-1)$。測定された $\gamma/(N_{\text{sites}}p)$ は $p$ の40倍の範囲で0.88〜1.05。
  * $F = 0.5$ でのゲート予算：$p = 10^{-2}$ でノイズ発生箇所84個、$10^{-3}$ で705、$5\times10^{-4}$ で1333。$F = 0.9$ ではおよそ7分の1。CNOT 1個がこの箇所を2つ消費するので、対応する2量子ビットゲート誤差は $2p$ である。
  * 減衰は $1/2^n$ で飽和するが、大きな $n$ では指数則が無用になるところまで成り立つ。

**3. 緩和はバイアスを減らし、分散で支払う**

  * 線形ZNEは $p_0 = 0.002$ でノイズのバイアスを14.5倍、$p_0 = 0.005$ で5.9倍減らした。
  * ノイズのバイアス（$+0.18$）はansatz誤差（$+0.0094$）の19倍だった。ansatzを磨く前にノイズを直すこと。
  * 2次外挿が線形より悪かったのは*印字された1回の実行に限った話*である。6シードで見れば平均バイアスは同等かむしろ小さく、代わりにばらつきが3〜6倍大きい。3点の厳密内挿は $\lVert w \rVert^2 = 19$、最小二乗直線は $2.33$ だからである。
  * Richardsonの重み $(2,-1)$、$(3,-3,1)$、$(4,-6,4,-1)$ は $\lVert w \rVert = 2.24, 4.36, 8.31$。ショット予算は $\lVert w \rVert^2$ でスケールする。

**4. 誤り訂正は改善ではなく領域の変化である**

  * $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$：閾値を上回れば大きな符号は悪く、下回れば論理誤りは $d$ について指数関数的に減る。
  * $p = 10^{-3}$ で $p_L = 10^{-10}$ に達するには $d = 17$、すなわち論理量子ビット1個あたり577個の物理量子ビット — 100論理量子ビットの機械で57,700個。
  * 閾値の半分での運用は「機能」し、論理あたり5,617個の物理量子ビットを要する。閾値からの余裕がすべてである。

**5. 3つの予算を同時に満たさなければならず、拘束条件は必ずしも深さではない**

  * 深さ：**2サイト**Hubbard模型への位相推定が $p < 3\times10^{-9}$、FeMoco規模の見積り（$10^{10}$〜$10^{11}$ ゲート）が $p \lesssim 10^{-12}$ を要求する。
  * 幅：量子ビット数は最も安い資源であり、誤り訂正はそれを $10^2\text{-}10^3$ 倍する。
  * サンプリング：20軌道の活性空間で化学精度に $6\times10^{10}$ ショット（完全なグループ化を仮定した最良の場合） — 毎秒 $10^4$ 回でエネルギー1つに2.4か月、構造最適化に20年、しかも**完璧なハードウェアで**。
  * $1/\varepsilon^2$ の測定スケーリングは装置ではなくアルゴリズムの性質である。そこから逃れるにはコヒーレントな深さを払う必要があり、このトレードオフは未解決問題ではなく定理である。だからこそ位相推定、したがって誤り耐性への最も強い論拠になる。

**6. 正直な評価**

  * NISQ装置はアルゴリズムのベンチマーク、ハードウェアの特性づけ、緩和の開発、そして格子模型のアナログシミュレーションができる。
  * 実在の材料でDFTに勝つこと、化学的に興味深い系で化学精度に達すること、位相推定を走らせること、化学・材料で検証された優位性を届けることはできない。
  * 主張は6つの質問で読むこと。課題は有用か、古典ベースラインは何か、検証されたか、どれだけの精度か、何をいくらのコストで緩和したか、スケールするか。
  * 原理で判断すること — 3つの予算を計算し古典ベースラインを同定する — であって、数か月で古びる装置の発表で判断しないこと。

**実務上の含意**

  * まず古典でシミュレートすること。30量子ビットの状態ベクトル計算で問いに答えられるなら、ハードウェアが加えるのはノイズだけである。
  * どんな回路を書く前にも3つの予算を見積もること。答えはたいてい「まだ無理」であり、その理由を知ることが価値のある部分である。
  * ハードウェアの結果を報告するときは、生の値と緩和後の値、ショット数、そして真剣な労力で計算した古典ベースラインを与えること。
  * フレームワークを1つきちんと学び、古典の強相関手法を1つ学ぶこと（DMRGが最良の選択）。それが量子計算の結果が勝たなければならない相手である。
  * 2つの考えを同時に保持すること。この分野には優れた物理と明確な長期目標があり、そして材料研究に対する近未来の応用はない。両方が真であり、そう言うことは悲観ではなくリテラシーである。

### 次章へ

シリーズの最後に到達しました。空のファイルから量子シミュレータを作り、それで変分固有値ソルバーを走らせ、フェルミオンハミルトニアンを量子ビットに写して写像を検証し、数値から物理を読み取れるモデルハミルトニアンを対角化し、ノイズを加えて忠実度の減衰を観察し、そのノイズを緩和してコストを測り、実際の計算が必要とする資源に値札を付けました。

残るのはそれを応用することです。次にあなたが出会う量子コンピューティングの主張 — セミナー、提案書、公募、プレスリリースのいずれであれ — は、いまや数分で定量的に評価できるものです。それを実行し、実行し続けてください。そうすれば、何かが真に変わったときに違いがわかる人々の一員になれます。量子計算の結果が勝たなければならない古典手法については[計算統計力学](<../computational-statistical-mechanics/index.html>)と強相関の文献へ、アルゴリズムの下にある理論については[量子場の理論入門](<../quantum-field-theory-introduction/index.html>)へ進んでください。

[← 第4章: 量子化学・材料計算への応用](<chapter-4.html>) [シリーズ目次に戻る →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章に引用した誤り率、閾値、コヒーレンス時間、資源見積りは教育目的で選んだ桁のオーダーの例示値であり、装置の仕様ではありません。評価や提案書に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
