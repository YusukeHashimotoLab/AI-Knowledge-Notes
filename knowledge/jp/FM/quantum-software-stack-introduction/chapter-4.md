---
title: "第4章: パルスと較正"
chapter_title: "第4章: パルスと較正"
subtitle: 制御の言葉で語る共鳴駆動、リークに抗するパルス整形、実験としての較正ループ、そしてランダマイズドベンチマーキングがゲート誤差とSPAMを分離する理由
reading_time: 50-55分
difficulty: 上級
code_examples: 9
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/quantum-software-stack-introduction/chapter-4.html>) | Last sync: 2026-08-13

[基礎数理道場](<../index.html>) > [量子ソフトウェアスタック入門](<index.html>) > 第4章

ここまでの3章は、ゲートを記号として扱ってきました。`("rx", theta, q)` が最適化器に入り、短くなって出てきて、ルータに入り、SWAPに囲まれて出てくる。その過程のどこでも、$R_x$ が物理的に*何であるか*を気にする部品はありませんでした。本章はその最後の箱を開けます。ゲート列の各要素の下にあるのは、数十ナノ秒の整形されたマイクロ波包絡です。そして包絡の下にあるのは3つか4つの数値 — 振幅、周波数、微分の重み — であり、これらはコンパイラには知りえず、データシートにも書けません。装置ごとに違い、火曜と水曜で違うからです。これらは*測定される*のです。ソフトウェアによって、実機上で、継続的に。その測定ループが本章の作る層です。

主張は2つあり、どちらも本質的です。第一に、パルスレベルの制御は特殊な物理ではありません。全体が回転系の3準位ハミルトニアンであり、40行のNumPyで、制御スタックが基礎とするリークの数値、DRAGによる抑制、較正曲線が再現できます。第二に、較正ルーチンはプログラムとして書かれた実験であり — パラメータ付きのシーケンス、フィット、更新則 — その正しさは、本コースの他の部分と同じ方法で検証できます。既知の誤差をシミュレート装置に仕込み、ルーチンからは隠し、ルーチンがそれを見つけることを確認するのです。4.3節のすべてのループにこの扱いを施し、そのうち1つは教訓的な形でこのテストに落ちます。4.4節では、較正ループ単独では答えられない問いを扱います。ゲートの測定はどれも準備と読み出しの誤差で汚染されているのに、ゲートは実際どれだけよく動いているのか、という問いです。

## 学習目標

本章を読み終えると、以下ができるようになります：

  * 駆動された3準位ハミルトニアンを回転系で書き下し、ゲートが較正されたパルス*面積*である理由を説明し、パルスレベルAPIが実際に露出しているハミルトニアンの要素を同定できる
  * リークが非調和性の位置で評価した包絡のFourier係数であることを導き、そこから矩形パルスのリークが $\tau^{-2}$ で落ち、ガウシアンのそれがはるかに速く落ちる理由を予測できる
  * DRAG補正を実装し、それが生むリーク抑制とゲート誤差抑制の両方を実測し、両者の最適重みが異なる理由を説明できる
  * 較正ルーチンをパラメータ付きシーケンス＋フィット＋更新として構成し、誤差増幅を使って有限のショット予算でミリラジアン精度に到達できる
  * Rabi振幅・Ramsey周波数・DRAG重みの較正を実装し、意図的にずらしたパラメータをそれぞれが回復することを検証し、うち1つが系統誤差で停滞する場合を診断できる
  * 残留ゲート誤差を $X$・$Y$・$Z$ に分解し、どの較正ノブがどの成分に対応するか、なぜループを反復しなければならないかを述べられる
  * ランダマイズドベンチマーキングの模型 $F(m) = Ap^m + B$ を述べ、状態準備・測定の誤差が $A$ と $B$ に、ゲート誤差が $p$ に入ることを数値的に示し、RBが*語らない*ことを説明できる

### 引き継ぐもの

4.1節と4.2節の物理は[量子ハードウェア入門 第2章](<../quantum-hardware-introduction/chapter-2.html>)の物理を、制御スタックの言葉で言い直したものです。同章はJosephsonハミルトニアンからトランズモンのスペクトルを導き、$E_J/h = 13.775$ GHz、$E_C/h = 0.250$ GHz、遷移周波数 $f_{01} = 4.9851$ GHz、非調和性 $\alpha/2\pi = -284.87$ MHz に到達します。この4つの数値が本章の入力であり、Code Example 1 が同じ対角化から再現します。本章のすべてのパルスはこのスペクトル上を走ります。本章は準位を5ではなく3で打ち切るので、20 nsガウシアン $\pi$ パルスのリークは $3.117\times10^{-5}$ から $3.201\times10^{-5}$ に動きます。3%の違いであり、これを明記する理由は、リークの数値が保持した準位数と一緒でなければ意味をもたないことです。

単位の規約は姉妹コースのものを例外なく踏襲します。ハミルトニアンは $\hbar = 1$ で書きます。$\Omega$ や $\Delta$ といった記号は*角*周波数であり、引用する数値は*周波数*です。コード中の $2\pi$ はそのためにすべて明示します。時間は一貫してナノ秒、周波数はGHzなので、$\alpha \tau$ は表からそのまま読める無次元量になります。

本章は[第1章](<chapter-1.html>)の回路IRを必要としません。パルスレベルの模型は1量子ビットと1つのユニタリであり、ゲート列は登場しません。[第5章](<chapter-5.html>)でIRに戻ります。

* * *

## 4.1 ゲートの下にあるもの

### ゲートとはパルス面積である

量子ビットのキャパシタにマイクロ波を加えると、ハミルトニアンに次の項が加わります。

$$ H_d(t) = \Omega(t)\cos(\omega_d t + \phi)\,\hat{n} $$

ここで $\Omega(t)$ は制御エレクトロニクスが生成する包絡、$\omega_d$ は搬送周波数、$\phi$ は搬送位相、$\hat{n}$ は電荷演算子です。$\omega_d$ で回転する系に移り、逆回転項を落とすと、正真正銘の2準位系については次が残ります。

$$ H_{\mathrm{rf}} = \frac{\Delta}{2}\sigma_z + \frac{\Omega_x(t)}{2}\sigma_x + \frac{\Omega_y(t)}{2}\sigma_y, \qquad \Delta = \omega_{01} - \omega_d $$

ここで $\Omega_x = \Omega\cos\phi$、$\Omega_y = \Omega\sin\phi$ です。共鳴時（$\Delta = 0$）これは $\phi$ が定める赤道面内の軸まわりの回転であり、回転角は

$$ \theta = \int_0^{\tau}\Omega(t)\,dt $$

です。これが「ゲート」の内容のすべてです。$R_x(\theta)$ とは*積分*が $\theta$ で搬送位相が0の包絡であり、$R_y(\theta)$ は同じ包絡を $\phi = \pi/2$ で使ったものです。包絡の形は自由で、パルス整形という主題はまるごとこの自由度をうまく使い切ることについてです。$R_z(\theta)$ はさらに安上がりです。回転系はソフトウェア上の構成物なので、$Z$ 回転は*以降のすべてのパルスの位相を定義し直す*ことで実装でき、時間も誤差もゼロで済みます。コンパイラが $Z$ 回転を回路の末尾に押し出すという判断 — 第2章のピープホール規則 — が単なる帳簿上の話ではなく物理的な見返りをもつのは、このためです。

### ゲート集合に入っていない準位

トランズモンは2準位系ではなく、その補正は小さくありません。3準位を保持すると、回転系のハミルトニアンは

$$ H_{\mathrm{rf}}(t) = \sum_{j=0}^{2}(E_j - j\omega_d)\lvert j \rangle\langle j \rvert + \frac{\Omega_x(t)}{2}\left(b + b^{\dagger}\right) + \frac{\Omega_y(t)}{2}\,i\left(b^{\dagger} - b\right) $$

となり、はしご演算子 $b = \sum_j r_j \lvert j \rangle\langle j+1 \rvert$ は $r_0 = 1$ となるよう規格化されています。このハミルトニアンについての2つの事実が、以降のすべてを決めます。準位2は回転系で0ではなく離調 $\alpha$ の位置にあるので、駆動は $1\to2$ 遷移に対して $|\alpha| = 285$ MHz 非共鳴です。そして $r_1 = |n_{12}/n_{01}| = 1.3726$ は1より*大きい*ので、駆動が届いてはいけない遷移のほうが行列要素が強いのです。

帰結は速度制限です。長さ $\tau$ のパルスは $1/\tau$ 程度までのスペクトル成分をもちます。それが $|\alpha|$ と重なると、駆動は $1\to2$ を励起し、population が計算部分空間を出ていきます。リークは通常の量子ビット誤差ではありません。どんな2準位誤差模型もこれを記述せず、どんな標準的な誤り訂正符号もこれを訂正しません。だから $|\alpha|\tau$ は、制御技術者が装置について最初に計算する数値なのです。

### Code Example 1: 3準位の制御モデル

```python
"""第4章 Code Example 1: 3準位の制御モデル。
量子ハードウェア入門 第2章のトランズモンスペクトルを、1量子ビットゲートが
到達しうる3準位に切り詰め、パルス層が実際にプログラムする回転系の生成子として
書き下す。周波数はGHz、時間はns、hbar = 1。"""
import numpy as np
from scipy.linalg import expm, logm

TWOPI = 2.0 * np.pi
NLEV = 3


def transmon_eigen(EJ, EC, ncut=60, m=NLEV):
    """トランズモンの固有エネルギー（GHz、基底状態を基準）と電荷演算子を、
    ともにトランズモン固有基底で返します。"""
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4.0 * EC * n ** 2.0)
    off = -0.5 * EJ * np.ones(2 * ncut)
    H += np.diag(off, 1) + np.diag(off, -1)
    E, V = np.linalg.eigh(H)
    nop = V[:, :m].conj().T @ np.diag(n.astype(float)) @ V[:, :m]
    return E[:m] - E[0], nop


EJ, EC = 13.775, 0.250
E, nop = transmon_eigen(EJ, EC)
f01 = E[1]
alpha = (E[2] - E[1]) - E[1]
print("トランズモンスペクトルから作る3準位の制御モデル")
print("=" * 70)
print(f"  EJ/h = {EJ:.3f} GHz, EC/h = {EC:.3f} GHz, EJ/EC = {EJ / EC:.1f}")
print(f"  f01 = {f01:.4f} GHz, alpha/2pi = {alpha * 1e3:.2f} MHz")
print(f"  相対非調和性 |alpha|/f01 = {abs(alpha) / f01 * 100:.2f} %")
print(f"  駆動行列要素の比 |n12/n01| = {abs(nop[1, 2] / nop[0, 1]):.4f}"
      f"   (調和振動子の値 sqrt(2) = {np.sqrt(2):.4f})")

# ---- 回転系の生成子 -----------------------------------------------------
LOWER = np.zeros((NLEV, NLEV), dtype=complex)
for j in range(NLEV - 1):
    LOWER[j, j + 1] = abs(nop[j, j + 1] / nop[0, 1])
AX = TWOPI * 0.5 * (LOWER + LOWER.conj().T)
AY = TWOPI * 0.5 * 1j * (LOWER.conj().T - LOWER)


def frame(f_drive):
    """駆動周波数 f_drive における回転系ハミルトニアンの対角部分です。"""
    return TWOPI * np.diag([E[j] - j * f_drive for j in range(NLEV)])


H0 = frame(f01)
print("\n  f_drive = f01 における回転系での準位離調 (GHz):",
      np.round(np.diag(H0).real / TWOPI, 6))
print(f"  Ax/2pi は0-1要素が0.5、1-2要素が"
      f" {AX[1, 2].real / TWOPI:.4f}")


def propagate(t, ox, oy, H0):
    """H0 + Ox(t) Ax + Oy(t) Ay の時間順序付き発展演算子、中点則です。"""
    U = np.eye(NLEV, dtype=complex)
    dt = t[1] - t[0]
    for k in range(len(t) - 1):
        H = H0 + 0.5 * (ox[k] + ox[k + 1]) * AX + 0.5 * (oy[k] + oy[k + 1]) * AY
        U = expm(-1j * H * dt) @ U
    return U


IDENT = np.eye(NLEV, dtype=complex)
XGATE = np.array([[0, 1], [1, 0]], dtype=complex)


def leakage(U):
    """3つの入力状態にわたる、量子ビット部分空間の外の population の最悪値です。"""
    return max(float(abs(U @ psi)[2] ** 2) for psi in
               [IDENT[0], IDENT[1], (IDENT[0] + IDENT[1]) / np.sqrt(2)])


def gate_error(U, target):
    """量子ビットブロックの target に対する平均ゲート誤差、グローバル位相は除去します。"""
    Uq = U[:2, :2]
    return 1.0 - (abs(np.trace(target.conj().T @ Uq)) ** 2
                  + np.trace(Uq.conj().T @ Uq).real) / 6.0


# ---- 最初のパルス: このモデルはゲートらしく振る舞うか ------------------
tau, nstep = 20.0, 800
t = np.linspace(0.0, tau, nstep + 1)
env = np.exp(-((t - tau / 2) ** 2) / (2 * (tau / 4) ** 2)) - np.exp(-2.0)
ox = 0.5 * env / np.trapezoid(env, t)          # パルス面積 = pi
U = propagate(t, ox, 0.0 * ox, H0)
print(f"\n  {tau:.0f} ns のガウシアン pi パルス、DRAGなし:")
print(f"    peak Omega/2pi   = {ox.max() * 1e3:7.1f} MHz")
print(f"    pulse area / pi  = {2 * np.trapezoid(ox, t):7.4f}")
print(f"    leakage          = {leakage(U):7.3e}")
print(f"    gate error vs X  = {gate_error(U, XGATE):7.3e}")
```

```text
トランズモンスペクトルから作る3準位の制御モデル
======================================================================
  EJ/h = 13.775 GHz, EC/h = 0.250 GHz, EJ/EC = 55.1
  f01 = 4.9851 GHz, alpha/2pi = -284.87 MHz
  相対非調和性 |alpha|/f01 = 5.71 %
  駆動行列要素の比 |n12/n01| = 1.3726   (調和振動子の値 sqrt(2) = 1.4142)

  f_drive = f01 における回転系での準位離調 (GHz): [ 0.        0.       -0.284873]
  Ax/2pi は0-1要素が0.5、1-2要素が 0.6863

  20 ns のガウシアン pi パルス、DRAGなし:
    peak Omega/2pi   =    46.7 MHz
    pulse area / pi  =  1.0000
    leakage          = 3.201e-05
    gate error vs X  = 2.970e-03
```

**注目点。** この模型は姉妹コースのスペクトルを厳密に再現します — $f_{01} = 4.9851$ GHz、$\alpha/2\pi = -284.87$ MHz、$r_1 = 1.3726$（調和振動子の $\sqrt{2} = 1.4142$ に対して） — 同じ対角化だからです。新しいのは最後のブロックです。面積1への規格化 `ox = 0.5 * env / trapezoid(env, t)` はパルス面積をちょうど $\pi$ にしますが、量子ビット部分空間を出た population が $3.2\times10^{-5}$ しかないのに、得られたゲートは $3.0\times10^{-3}$ 間違っています。ゲート誤差はリークの100倍です。この比が本節でもっとも重要な点であり、4.2節がその説明を与えます。損害の大半は*位相*であり、population が $\lvert 2 \rangle$ を訪れて戻ってくる間に量子ビットが獲得したものなのです。

`propagate`・`leakage`・`gate_error` は以降のすべての例が呼ぶ3つの関数です。ゲート誤差の指標は、量子ビットブロックの標的に対する平均ゲート非忠実度をグローバル位相を除いて計算したものです。リークのある発展演算子のブロックはユニタリでないため $\mathrm{tr}(U^{\dagger}U) < 2$ となり、リークは別途表示されるだけでなく誤差にも計上されます。

* * *

## 4.2 パルス整形

### リークはFourier係数である

$1 \to 2$ 結合を、意図した回転の上に乗る摂動として扱います。1次では $\lvert 2 \rangle$ に流れ込む振幅は

$$ a_{1\to2} \simeq -\frac{i r_1}{2}\int_0^{\tau}\Omega_x(t)\,e^{i\alpha t}\,dt = -\frac{i r_1}{2}\,\tilde{\Omega}_x(\alpha) $$

です。リークは、非調和性の位置で評価した包絡のFourier係数の2乗です。この1行がパルス層全体の設計則です。**包絡を、$\alpha$ でのスペクトル重みができるだけ小さくなるように整形せよ。** 面積1の矩形パルスではスペクトルはsincであり、

$$ \left\lvert\frac{\tilde{\Omega}(\alpha)}{\tilde{\Omega}(0)}\right\rvert^{2} = \mathrm{sinc}^{2}(\alpha\tau), \qquad \mathrm{sinc}(x) = \frac{\sin \pi x}{\pi x} $$

その包絡は $1/(\alpha\tau)^2$ でしか落ちません。不連続な包絡は $1/f$ のスペクトル裾をもつからです。ガウシアンはガウシアンの裾をもち、はるかによく振る舞います。整形が買うのは前係数ではなく指数です。

### Code Example 2: 矩形、ガウシアン、そしてリークの出どころ

```python
"""第4章 Code Example 2: 矩形、ガウシアン、そしてリークの出どころ。
Code Example 1 の続き（同一セッション）。"""


def envelope(shape, tau, nstep):
    """[0, tau] 上の面積1の包絡: 'square' または 'gauss'（4シグマで打ち切り）。"""
    t = np.linspace(0.0, tau, nstep + 1)
    if shape == "square":
        env = np.ones_like(t)
    elif shape == "gauss":
        sig = tau / 4.0
        env = np.exp(-((t - tau / 2) ** 2) / (2 * sig ** 2)) - np.exp(-2.0)
    else:
        raise ValueError(shape)
    return t, env / np.trapezoid(env, t)


def pulse(shape, tau, theta=np.pi, beta=0.0, amp=1.0, axis="x", nstep=400):
    """axis 軸まわり theta 回転の制御波形、DRAG重み beta つきです。"""
    t, u = envelope(shape, tau, nstep)
    ox = amp * (theta / TWOPI) * u
    oy = -beta * np.gradient(ox, t) / (TWOPI * 2.0 * alpha)
    return (t, ox, oy) if axis == "x" else (t, -oy, ox)


def run(shape, tau, theta=np.pi, beta=0.0, amp=1.0, axis="x",
        f_drive=None, nstep=400):
    """整形されたパルス1つの発展演算子です。"""
    t, cx, cy = pulse(shape, tau, theta, beta, amp, axis, nstep)
    return propagate(t, cx, cy, frame(f01 if f_drive is None else f_drive))


def spectral_weight(shape, tau, f, nstep=4000):
    """包絡の |Omega(f)|^2 / |Omega(0)|^2、すなわち f におけるFourier重みです。"""
    t, u = envelope(shape, tau, nstep)
    return abs(np.trapezoid(u * np.exp(-2j * np.pi * f * t), t)) ** 2 \
        / abs(np.trapezoid(u, t)) ** 2


print("A. 非調和性の位置における包絡のFourier重み")
print("=" * 74)
print(f"  1-2遷移は駆動から {abs(alpha) * 1e3:.1f} MHz 離れており、そこに重みを持つ")
print("  包絡はこの遷移を駆動します。相対重み |Omega(alpha)|^2 /"
      " |Omega(0)|^2:")
print(f"\n  {'tau (ns)':>9} {'|alpha| tau':>12} {'square':>12}"
      f" {'sinc^2 (exact)':>15} {'gauss':>12}")
for tau in (8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0):
    x = np.pi * alpha * tau
    print(f"  {tau:9.1f} {abs(alpha) * tau:12.2f}"
          f" {spectral_weight('square', tau, alpha):12.3e}"
          f" {(np.sin(x) / x) ** 2:15.3e}"
          f" {spectral_weight('gauss', tau, alpha):12.3e}")

print("\nB. pi パルスのリークとゲート誤差の実測")
print("=" * 74)
print(f"  {'tau (ns)':>9} {'shape':>7} {'peak Om/2pi':>12} {'leak |1>->|2>':>14}"
      f" {'worst leak':>12} {'gate error':>12}")
meas = {"square": [], "gauss": []}
taus = (8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0)
for tau in taus:
    for shape in ("square", "gauss"):
        t, ox, oy = pulse(shape, tau)
        U = run(shape, tau)
        l12 = abs(U[2, 1]) ** 2
        meas[shape].append(l12)
        print(f"  {tau:9.1f} {shape:>7} {ox.max() * 1e3:12.1f} {l12:14.3e}"
              f" {leakage(U):12.3e} {gate_error(U, XGATE):12.3e}")

print("\n  tau = 10 から 40 ns における |1>->|2> リークのべき乗則:")
for shape in ("square", "gauss"):
    y = np.array(meas[shape][1:])
    x = np.array(taus[1:])
    slope = np.polyfit(np.log(x), np.log(y), 1)[0]
    print(f"    {shape:>7}: leakage ~ tau^{slope:+.2f}"
          f"   -> tauを倍にすると {2 ** (-slope):6.1f}倍 改善")
```

```text
A. 非調和性の位置における包絡のFourier重み
==========================================================================
  1-2遷移は駆動から 284.9 MHz 離れており、そこに重みを持つ
  包絡はこの遷移を駆動します。相対重み |Omega(alpha)|^2 / |Omega(0)|^2:

   tau (ns)  |alpha| tau       square  sinc^2 (exact)        gauss
        8.0         2.28    1.152e-02       1.152e-02    3.250e-04
       10.0         2.85    2.614e-03       2.614e-03    9.376e-05
       12.0         3.42    8.114e-03       8.114e-03    2.832e-05
       16.0         4.56    4.717e-03       4.717e-03    4.078e-08
       20.0         5.70    2.067e-03       2.067e-03    2.512e-06
       30.0         8.55    1.358e-03       1.358e-03    2.729e-09
       40.0        11.39    6.983e-04       6.983e-04    1.352e-07

B. pi パルスのリークとゲート誤差の実測
==========================================================================
   tau (ns)   shape  peak Om/2pi  leak |1>->|2>   worst leak   gate error
        8.0  square         62.5      2.138e-02    3.461e-02    2.583e-02
        8.0   gauss        116.8      9.393e-05    1.054e-04    1.815e-02
       10.0  square         50.0      1.382e-02    1.562e-02    1.973e-02
       10.0   gauss         93.4      3.427e-04    5.821e-04    1.198e-02
       12.0  square         41.7      9.853e-03    1.089e-02    1.284e-02
       12.0   gauss         77.8      9.594e-05    1.093e-04    8.270e-03
       16.0  square         31.2      5.614e-03    5.988e-03    7.683e-03
       16.0   gauss         58.4      4.052e-05    5.310e-05    4.648e-03
       20.0  square         25.0      3.607e-03    3.928e-03    5.004e-03
       20.0   gauss         46.7      1.761e-05    3.198e-05    2.970e-03
       30.0  square         16.7      1.609e-03    1.648e-03    2.146e-03
       30.0   gauss         31.1      3.140e-06    3.989e-06    1.318e-03
       40.0  square         12.5      9.039e-04    1.356e-03    1.172e-03
       40.0   gauss         23.4      9.251e-07    9.652e-07    7.407e-04

  tau = 10 から 40 ns における |1>->|2> リークのべき乗則:
     square: leakage ~ tau^-1.97   -> tauを倍にすると    3.9倍 改善
      gauss: leakage ~ tau^-4.09   -> tauを倍にすると   17.1倍 改善
```

**注目点。** パートAはスペクトル像を検証します。矩形包絡のFourier重みを数値積分した値は $\mathrm{sinc}^2(\alpha\tau)$ と表示桁すべてで一致し、ガウシアンの列は3桁から5桁小さいです。ガウシアンの列は桁単位で*振動*もしています — $\tau = 16$ nsで $4\times10^{-8}$、20 nsで $2.5\times10^{-6}$ — が、これは数値誤差ではありません。ここで使う包絡は $\pm 2\sigma$ で*打ち切った*ガウシアンから台座を引いたもので、両端で微分に小さな不連続が残ります。その端点由来のsinc的な裾が、特定の長さでゼロをもつのです。実際の制御スタックがまさにこの理由で端をさらになまらせます。

パートBが実測で、最後の2行が成果物です。$\tau = 10$ から 40 ns でフィットすると、矩形パルスのリークは $\tau^{-1.97}$、ガウシアンのリークは $\tau^{-4.09}$ で落ちます。パルス長を倍にすると、矩形では3.9倍、ガウシアンでは17倍の改善が買えます。$\tau = 20$ nsでは両者のリークは205倍違います。

誠実な留保が2つあります。第一に、$\tau = 8$ nsではガウシアンのほうがゲート誤差が*悪い*のであり、その理由はピーク振幅の列にあります。面積と長さを固定すると、ガウシアンは116.8 MHzに達しなければならず、矩形は62.5 MHzで足ります。ピークRabi周波数が $|\alpha|$ に近づけば摂動的な描像はまるごと崩れます。整形が効くのは、パルスが整形できるだけ長い領域だけです。第二に、どの行でもゲート誤差はリークを上回っており、長いパルスでは2桁上回ります。整形だけでゲートは直りません。

### DRAG

超過分は位相です。パルスの間、population は $\lvert 2 \rangle$ へ*仮想的な*遠回りをして戻ります。この遠回りが駆動中の量子ビットの実効周波数をずらし、蓄積された位相は意図した回転の一部ではありません。標準的な対処がDRAG（Derivative Removal by Adiabatic Gate）であり、直交成分を同相包絡の微分で駆動します。

$$ \Omega_y(t) = -\beta\,\frac{\dot{\Omega}_x(t)}{2\alpha} $$

1次の理論は $\beta = 1$ を与えます。微分項は主要なリーク振幅を打ち消すように選ばれており、同時に主要な位相誤差も打ち消します。実際のスタックでは $\beta$ は定数ではなく較正される数値であり、Code Example 3 がその理由を示します。

### Code Example 3: DRAGと、共有されない2つの最適点

```python
"""第4章 Code Example 3: DRAGと、共有されない2つの最適点。
Code Example 2 の続き（同一セッション）。"""
print("A. tau = 20 ns でDRAG重みを走査する")
print("=" * 74)
TAU = 20.0
PAULI3 = {"X": XGATE, "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def error_axis(U, target):
    """target を打ち消した後に残る量子ビットブロックの回転を、X・Y・Z成分に
    分解します。返り値は (theta n_x, theta n_y, theta n_z) です。"""
    R = U[:2, :2] @ target.conj().T
    R = R / np.sqrt(complex(np.linalg.det(R)))       # SU(2)へ
    if np.trace(R).real < 0.0:
        R = -R                                       # Iに最も近い持ち上げ
    A = logm(R)
    A = A - 0.5 * np.trace(A) * np.eye(2)
    return {k: float((1j * np.trace(P @ A)).real) for k, P in PAULI3.items()}


print(f"  {'beta':>6} {'leak |1>->|2>':>14} {'gate error':>12}"
      f" {'theta_x err':>12} {'theta_y err':>12} {'theta_z err':>12}")
for beta in np.arange(0.0, 2.26, 0.25):
    U = run("gauss", TAU, np.pi, beta)
    e = error_axis(U, XGATE)
    print(f"  {beta:6.2f} {abs(U[2, 1]) ** 2:14.3e} {gate_error(U, XGATE):12.3e}"
          f" {e['X']:12.2e} {e['Y']:12.2e} {e['Z']:12.2e}")


def golden_min(f, lo, hi, tol=1e-6):
    """単峰な f の [lo, hi] 上での黄金分割最小化です。最適化ライブラリを
    読み込むより60行短く、以降の較正ループでも再利用します。"""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


b_err = golden_min(lambda b: gate_error(run("gauss", TAU, np.pi, b), XGATE),
                   0.5, 1.5)
b_leak = golden_min(lambda b: abs(run("gauss", TAU, np.pi, b)[2, 1]) ** 2,
                    1.5, 2.5)
print("\nB. 2つの最適点は異なる — ここが要点である")
print("=" * 74)
for name, b in (("DRAGなし", 0.0), ("ゲート誤差最適", b_err),
                ("リーク最適", b_leak)):
    U = run("gauss", TAU, np.pi, b)
    print(f"  {name:>19}  beta = {b:7.4f}   leakage = {abs(U[2, 1]) ** 2:9.3e}"
          f"   gate error = {gate_error(U, XGATE):9.3e}")
print(f"\n  ゲート誤差最適点での改善:"
      f" {gate_error(run('gauss', TAU, np.pi, 0.0), XGATE) / gate_error(run('gauss', TAU, np.pi, b_err), XGATE):.0f}倍")
print(f"  リーク最適点での悪化      :"
      f" {gate_error(run('gauss', TAU, np.pi, b_leak), XGATE) / gate_error(run('gauss', TAU, np.pi, b_err), XGATE):.0f}倍")

print("\nC. ゲート誤差の最適点はパルス長とともに動く")
print("=" * 74)
print(f"  {'tau (ns)':>9} {'beta*':>7} {'err (beta=0)':>13} {'err (beta*)':>13}"
      f" {'improvement':>12}")
for tau in (8.0, 12.0, 20.0, 30.0, 40.0):
    bs = golden_min(lambda b: gate_error(run("gauss", tau, np.pi, b), XGATE),
                    0.0, 2.0)
    e0 = gate_error(run("gauss", tau, np.pi, 0.0), XGATE)
    e1 = gate_error(run("gauss", tau, np.pi, bs), XGATE)
    print(f"  {tau:9.1f} {bs:7.4f} {e0:13.3e} {e1:13.3e} {e0 / e1:11.0f}x")

```

```text
A. tau = 20 ns でDRAG重みを走査する
==========================================================================
    beta  leak |1>->|2>   gate error  theta_x err  theta_y err  theta_z err
    0.00      1.761e-05    2.970e-03    -7.35e-03     1.33e-01    -8.98e-17
    0.25      1.307e-05    1.604e-03    -6.96e-03     9.75e-02    -3.02e-16
    0.50      9.307e-06    6.569e-04    -6.69e-03     6.20e-02     1.18e-16
    0.75      6.262e-06    1.304e-04    -6.52e-03     2.65e-02    -2.05e-16
    1.00      3.883e-06    2.423e-05    -6.46e-03    -8.96e-03     2.97e-16
    1.25      2.116e-06    3.378e-04    -6.52e-03    -4.44e-02     3.18e-16
    1.50      9.130e-07    1.070e-03    -6.68e-03    -7.98e-02     1.16e-16
    1.75      2.232e-07    2.220e-03    -6.96e-03    -1.15e-01    -1.10e-16
    2.00      1.091e-10    3.784e-03    -7.35e-03    -1.51e-01     9.63e-17
    2.25      1.983e-07    5.762e-03    -7.85e-03    -1.86e-01     5.55e-17

B. 2つの最適点は異なる — ここが要点である
==========================================================================
               DRAGなし  beta =  0.0000   leakage = 1.761e-05   gate error = 2.970e-03
              ゲート誤差最適  beta =  0.9382   leakage = 4.412e-06   gate error = 1.140e-05
                リーク最適  beta =  2.0030   leakage = 7.859e-11   gate error = 3.806e-03

  ゲート誤差最適点での改善: 261倍
  リーク最適点での悪化      : 334倍

C. ゲート誤差の最適点はパルス長とともに動く
==========================================================================
   tau (ns)   beta*  err (beta=0)   err (beta*)  improvement
        8.0  0.9122     1.815e-02     3.297e-04          55x
       12.0  0.9311     8.270e-03     8.380e-05          99x
       20.0  0.9382     2.970e-03     1.140e-05         261x
       30.0  0.9403     1.318e-03     2.241e-06         588x
       40.0  0.9411     7.407e-04     7.061e-07        1049x
```

**注目点。** パートAは $\pi$ パルスの残留誤差を3つのPauli軸に分解します。本章の残りを可能にする診断です。列を縦に読んでください。$Z$ 成分は機械精度でゼロです — $x$ まわりの $\pi$ パルス1回では $Z$ 誤差は蓄積できません。まさにこれが、4.3節で周波数の較正に $\pi$ パルスではなくRamsey実験が必要な理由です。$X$ 成分は $-6.5\times10^{-3}$ rad にあり、$\beta$ とともにほとんど動きません。これは回転角の誤差で、DRAGは手が出せません。$\beta$ が制御するのは $Y$ 成分で、$+0.133$ rad からゼロを通って $-0.186$ rad まで走ります。1つのノブ、1つの誤差成分です。

パートBが要点です。*ゲート誤差*を最小にする $\beta$ は0.9382、*リーク*を最小にする $\beta$ は2.0030です。両者は異なり、リーク最適点に移すとゲートは334倍悪くなります。$\beta = 2$ ではリークが $8\times10^{-11}$ で、位相誤差はまったく打ち消されていないからです。$\lvert 2 \rangle$ の population を最小化してDRAGを調整した人は、間違った量を最適化したことになります。その兆候は、悪いゲートに付随する美しいリークの数値です。パートCが加えるのは、ゲート誤差の最適点は解析的な $\beta = 1$ でもなく、表の範囲で0.912から0.941まで動くという事実です。パルスレベルAPIが $\beta$ をパラメータとして露出しているのは、埋め込むべき値が存在しないからです。

* * *

## 4.3 実験としての較正ループ

### ループの形

あらゆる制御スタックのあらゆる較正ルーチンは、同じ3つの部分をもちます。

  1. **パラメータ付きシーケンス。** 結果が較正したいパラメータに依存し、そして — ここが難所です — 他の何よりも*強く*依存するもの。
  2. **フィット。** ノイズを含む読み出し頻度の集合をパラメータの推定値に写すもの。
  3. **更新。** 推定値を書き戻し、通常は反復するかどうかを判断するもの。

2つの設計原理がほとんどの仕事をします。第一は**誤差増幅**です。小さなパラメータ誤差が繰り返し数 $N$ 倍されて結果に現れるようにシーケンスを組みます。振幅誤差1%の $\pi$ パルス1回は読み出しを1から0.02%しか動かさず、現実的なショット予算では分解できません。81回なら誤差は81%になり、1ショットで分解できます。第二は**構成による相殺**です。観測量を2つのシーケンスの*差*として組み、欲しい誤差は残り、欲しくない誤差は消えるように選びます。読み出し誤差は良い実機でも2〜5%、気にしたいゲート誤差は $10^{-4}$ です。前者を後者から引く方法はショット数では得られないので、シーケンス側でやらせるしかありません。

### ソフトウェアに見せてよいもの

較正ルーチンを検証する誠実な方法は、真値をルーチンから隠すことです。Code Example 4 は、開示されない4つのパラメータ — 増幅ゲイン、真の量子ビット周波数、2つの読み出し誤り率 — に加えて初期化誤差と準静的な周波数ばらつきをもつシミュレート装置を定義します。以降のすべてのルーチンは `device()` を呼び、0から1の1つの数値に二項分布のショットノイズが乗ったものを受け取ります。それ以上は何も受け取りません。どのルーチンも `HIDDEN` を読みません。

### Code Example 4: 較正ソフトウェアから見た装置

```python
"""第4章 Code Example 4: 較正ソフトウェアに見えている装置。
Code Example 3 の続き（同一セッション）。"""
# 以下の数値は、制御ソフトウェアが知らないもの一切を代表します。増幅系のゲイン、
# 真の量子ビット周波数、そして2つの読み出し誤り率です。Code Example 5 から 8 の
# どのルーチンもこの辞書を読みません。呼ぶのは device() だけであり、それがこの
# 演習の要点そのものです。
HIDDEN = {"gain": 1.137, "f_qubit": f01, "eps01": 0.020, "eps10": 0.045,
          "init_err": 0.012, "sigma_f": 2.0e-4}

TAU_G = 20.0                 # パルス層が採用したゲート長
_CACHE = {}


def _prop(theta, axis, amp, beta, f_drive):
    """隠れたゲインを適用した、パルス1つの発展演算子（キャッシュ付き）です。"""
    key = (theta, axis, amp, beta, f_drive)
    if key not in _CACHE:
        _CACHE[key] = run("gauss", TAU_G, theta, beta, amp * HIDDEN["gain"],
                          axis, f_drive=f_drive)
    return _CACHE[key]


def sequence_unitary(seq, f_drive, amp, beta, df=0.0):
    """シーケンスの発展演算子。('p', theta, axis) はパルス、('i', T) は待機です。

    df は量子ビット周波数の静的なずれで、Ramseyフリンジに包絡を与える準静的な
    周波数ノイズを平均するために使います。
    """
    U = np.eye(NLEV, dtype=complex)
    for item in seq:
        if item[0] == "p":
            U = _prop(item[1], item[2], amp, beta, f_drive - df) @ U
        else:
            ph = [TWOPI * ((E[j] + j * df) - j * f_drive) * item[1]
                  for j in range(NLEV)]
            U = np.diag(np.exp(-1j * np.array(ph))) @ U
    return U


def p_read1(seq, f_drive, amp, beta, nquad=9):
    """1と読まれる確率です。初期化誤差と読み出し誤差、および準静的な周波数
    ばらつきの平均を含みます。"""
    hz, hw = np.polynomial.hermite_e.hermegauss(nquad)
    hw = hw / hw.sum()
    p1 = p2 = 0.0
    for z, w in zip(hz, hw):
        U = sequence_unitary(seq, f_drive, amp, beta, df=HIDDEN["sigma_f"] * z)
        a0, a1 = U[:, 0], U[:, 1]
        p1 += w * ((1 - HIDDEN["init_err"]) * abs(a0[1]) ** 2
                   + HIDDEN["init_err"] * abs(a1[1]) ** 2)
        p2 += w * ((1 - HIDDEN["init_err"]) * abs(a0[2]) ** 2
                   + HIDDEN["init_err"] * abs(a1[2]) ** 2)
    p0 = 1.0 - p1 - p2
    return p1 * (1 - HIDDEN["eps10"]) + p0 * HIDDEN["eps01"] + p2


def device(seq, f_drive, amp, beta, shots, rng):
    """1回の実験: 準備し、シーケンスを走らせ、読み出し、頻度を返します。

    shots=None は厳密な確率を返します。無限のショット予算はどの実験も持ちません
    が、較正ループの設計者が残差を統計的か系統的かを見分けるのに使えます。
    """
    p = p_read1(seq, f_drive, amp, beta)
    return p if shots is None else rng.binomial(shots, p) / shots


print("較正ソフトウェアから見た装置")
print("=" * 74)
rng = np.random.default_rng(20260813)
for label, seq in [("何もしない（|0>の読み出し）", []),
                   ("公称 pi パルス1回", [("p", np.pi, "x")]),
                   ("公称 pi パルス2回", [("p", np.pi, "x")] * 2),
                   ("pi/2、500 ns待機、pi/2",
                    [("p", np.pi / 2, "x"), ("i", 500.0),
                     ("p", np.pi / 2, "x")])]:
    v = device(seq, f01, 1.0, 0.0, 8000, rng)
    print(f"  {label:>26}: read-1 fraction = {v:.4f}")
```

```text
較正ソフトウェアから見た装置
==========================================================================
             何もしない（|0>の読み出し）: read-1 fraction = 0.0359
                 公称 pi パルス1回: read-1 fraction = 0.9019
                 公称 pi パルス2回: read-1 fraction = 0.1852
          pi/2、500 ns待機、pi/2: read-1 fraction = 0.8161
```

**注目点。** 初期化直後の量子ビットの読み出しは0ではなく0.036、公称 $\pi$ パルス1回は1ではなく0.90です。どちらもゲートと無関係な誤差に支配されています。初期化は1.2%を $\lvert 1 \rangle$ に残し、読み出しは0の2.0%と1の4.5%を取り違えます。$\pi/2$・待機・$\pi/2$ のシーケンスが0.82を返すのは、$\sigma_f = 200$ kHzの準静的ばらつきの下で500 nsも待つとコントラストが目に見えて失われるからです。厳密な確率は9点のGauss-Hermite求積でそのばらつきについて平均されており、これが Code Example 6 のRamseyフリンジを現実らしく減衰させます。`shots=None` の分岐はルーチン*設計者*のための意図的な便宜であって、ルーチンのためのものではありません。厳密な確率を返すので、ショットノイズあり・なしでループの残差を比べれば、系統誤差と統計誤差を見分けられます。

### Code Example 5: 較正ループ I — Rabi振幅

```python
"""第4章 Code Example 5: 較正ループ I — Rabi振幅。
Code Example 4 の続き（同一セッション）。"""


def fit_cosine(x, y, w_grid):
    """w のグリッド上で y = c0 + c1 cos(w x) を最小二乗フィットします。

    w を固定すると (c0, c1) について線形なので、フィット全体は内側に2x2の解を
    もつ1次元走査になります。最適化ライブラリも初期値も要りません。
    """
    best = (np.inf, None)
    for w in w_grid:
        A = np.column_stack([np.ones_like(x), np.cos(w * x)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ c - y) ** 2))
        if r < best[0]:
            best = (r, w)
    return best[1]


def rabi_calibration(stages, start, shots, rng, f_drive, beta):
    """粗い振幅走査のあと、誤差増幅による精密化を行います。

    pi パルスを n_rep 回繰り返すと、振幅ノブに対するフリンジ周波数が n_rep 倍に
    なるので、同じ走査・同じショット予算で pi 振幅が n_rep 倍鋭く決まります。
    これが誤差増幅であり、有限の予算でミリラジアン精度に届く唯一の理由です。
    """
    est, trace = start, []
    for n_rep in stages:
        half = 0.45 if n_rep == 1 else 0.40 / n_rep
        npts = 41 if n_rep == 1 else 21
        amps = np.linspace(est * (1 - half), est * (1 + half), npts)
        seq = [("p", np.pi, "x")] * n_rep
        y = np.array([device(seq, f_drive, a, beta, shots, rng) for a in amps])
        grid = np.linspace(est * (1 - half), est * (1 + half), 8001)
        est = n_rep * np.pi / fit_cosine(amps, y, n_rep * np.pi / grid)
        trace.append((n_rep, npts, est))
    return est, trace


def true_pi_amplitude(beta, f_drive):
    """オラクル。答えを報告するためだけに使い、較正ループは決して呼びません。"""
    return golden_min(lambda a: -abs(run("gauss", TAU_G, np.pi, beta,
                                        a * HIDDEN["gain"],
                                        f_drive=f_drive)[1, 0]) ** 2, 0.7, 1.1)


print("較正ループ I: Rabi振幅")
print("=" * 74)
amp_true = true_pi_amplitude(0.0, f01)
print(f"  隠れた増幅ゲイン      : {HIDDEN['gain']:.4f} なので、公称 amp = 1 の")
print(f"  パルスは {(HIDDEN['gain'] - 1) * np.pi * 1e3:.0f} mrad 過回転します")
print(f"  真の pi 振幅（オラクル）: {amp_true:.6f}"
      f"   (1/gain = {1 / HIDDEN['gain']:.6f})")

STAGES = (1, 5, 21, 81)
print("\nA. DRAG重みがまだ0のままでのループ")
print("  'exact' は同じループを無限ショット予算で繰り返したものです。どの実験も")
print("  持てませんが、系統誤差と統計誤差を切り分けられます。")
print(f"  {'n_rep':>6} {'points':>7} {'amp_pi (2000 shots)':>20}"
      f" {'over-rot':>10} {'amp_pi (exact)':>15} {'over-rot':>10}")
rng = np.random.default_rng(4242)
amp_true = true_pi_amplitude(0.0, f01)
est, exact = 1.0, 1.0
for n_rep in STAGES:
    est, tr = rabi_calibration((n_rep,), est, 2000, rng, f01, 0.0)
    exact, _ = rabi_calibration((n_rep,), exact, None, None, f01, 0.0)
    print(f"  {n_rep:6d} {tr[0][1]:7d} {est:20.6f}"
          f" {abs(est - amp_true) * np.pi * 1e3:7.2f} mrad {exact:15.6f}"
          f" {abs(exact - amp_true) * np.pi * 1e3:7.2f} mrad")
amp_cal = est

print("\nB. Code Example 7 のDRAG重みを先に入れた場合の同じループ")
b_known = 0.9382
amp_true_b = true_pi_amplitude(b_known, f01)
print(f"  {'n_rep':>6} {'amp_pi found':>13} {'residual':>11} {'over-rotation':>15}")
rng = np.random.default_rng(4242)
est = 1.0
for n_rep in STAGES:
    est, _ = rabi_calibration((n_rep,), est, 2000, rng, f01, b_known)
    print(f"  {n_rep:6d} {est:13.6f} {est - amp_true_b:+11.2e}"
          f" {abs(est - amp_true_b) * np.pi * 1e3:12.2f} mrad")
e0 = gate_error(run("gauss", TAU_G, np.pi, 0.0, HIDDEN["gain"]), XGATE)
e1 = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"]), XGATE)
print(f"\n  公称振幅でのゲート誤差: {e0:.3e}")
print(f"  ループAの後のゲート誤差: {e1:.3e}")
```

```text
較正ループ I: Rabi振幅
==========================================================================
  隠れた増幅ゲイン      : 1.1370 なので、公称 amp = 1 の
  パルスは 430 mrad 過回転します
  真の pi 振幅（オラクル）: 0.879429   (1/gain = 0.879507)

A. DRAG重みがまだ0のままでのループ
  'exact' は同じループを無限ショット予算で繰り返したものです。どの実験も
  持てませんが、系統誤差と統計誤差を切り分けられます。
   n_rep  points  amp_pi (2000 shots)   over-rot  amp_pi (exact)   over-rot
       1      41             0.879062    1.15 mrad        0.879962    1.67 mrad
       5      21             0.881735    7.24 mrad        0.881494    6.49 mrad
      21      21             0.881684    7.08 mrad        0.881586    6.78 mrad
      81      21             0.881706    7.15 mrad        0.881688    7.10 mrad

B. Code Example 7 のDRAG重みを先に入れた場合の同じループ
   n_rep  amp_pi found    residual   over-rotation
       1      0.881763   +4.35e-04         1.37 mrad
       5      0.881198   -1.29e-04         0.41 mrad
      21      0.881076   -2.51e-04         0.79 mrad
      81      0.881292   -3.55e-05         0.11 mrad

  公称振幅でのゲート誤差: 3.298e-02
  ループAの後のゲート誤差: 2.981e-03
```

**注目点。** 増幅系は公称より13.7%大きい振幅を出すので、公称 $\pi$ パルスは430 mrad 過回転します。パートAの粗い走査は $\pi$ 振幅を1.15 mradまで決めますが、そのあと増幅段は事態を*悪化*させ、改善が止まります。$n_{\mathrm{rep}} = 5, 21, 81$ で 7.24、7.08、7.15 mrad です。`exact` の列が診断を確定させます。無限のショット予算でも残差は同じ7 mradで停滞するので、これは系統誤差でありサンプリングの問題ではありません。原因は較正されていないDRAG重みです。パルスを繰り返すと振幅誤差だけでなく Code Example 3 の $Y$ 誤差も蓄積し、振幅ノブについてきれいなコサインを仮定するフィットが、その一部をフィット周波数に吸収してしまうのです。

パートBは、$\beta$ を Code Example 7 が測る値に設定した同じループで、宣伝どおりに振る舞います。1.37、0.41、0.79、0.11 mrad です。13.7%の振幅誤差が0.1 mrad程度まで回復し、ゲート誤差は $3.3\times10^{-2}$ から $3.0\times10^{-3}$ に落ちます。これは*まだ*較正されていないDRAG重みが定める天井です。較正ループは独立ではありません。決まった順序で走らせ、そしてもう一度走らせるのであり、Code Example 8 はその不動点が存在することの実証です。

### 周波数には待機が必要である

Code Example 3 の感度解析は、$\pi$ パルス1回では $Z$ 誤差が生じないと述べました。ここで重要なのはその逆です。*離調*も $\pi$ パルスにはほとんど痕跡を残しません。駆動が離調に比べて強く、誤差をただ回してしまうからです。20 nsパルスに1.5 MHzの離調が乗ってもゲート誤差は $5\times10^{-4}$ で、すでにあるDRAG誤差より小さいのです。離調を見えるようにするには駆動を*切って*位相を蓄積させる必要があり、それがRamseyシーケンスのすることです。$\pi/2$、$T$ 待つ、$\pi/2$。蓄積位相は $2\pi\Delta f\,T$ で、$T$ が $T_2^{\ast}$ に達するまで感度は $T$ に比例して伸びます。

避けられない細部が1つあり、これは後で発見するよりルーチンに組み込むに値します。Ramseyフリンジが測るのは $\lvert \Delta f \rvert$ であって $\Delta f$ ではありません。標準的な解決は、既知の $f_{\mathrm{art}}$ だけ駆動を両方向に意図的にずらし、2つのフリンジ周波数の差を取ることです。これで符号が回復し、おまけにゼロ近傍のフリンジをフィットに解かせることも避けられます。

### Code Example 6: 較正ループ II — 量子ビット周波数

```python
"""第4章 Code Example 6: 較正ループ II — 量子ビット周波数。
Code Example 5 の続き（同一セッション）。"""
T2_GRID = np.linspace(300.0, 2500.0, 12)


def fit_ramsey(delays, y, nu_hi, rounds=2, npts=401):
    """y = c0 + exp(-(T/T2)^2)(c1 cos 2 pi nu T + c2 sin 2 pi nu T) をフィットします。

    (nu, T2) を固定すると (c0, c1, c2) について線形なので、非線形な2パラメータの
    走査の内側で3x3を解く形になり、最良点のまわりで1回精密化します。返り値は
    フリンジ周波数とガウシアン包絡の時定数です。
    """
    lo, hi = 0.0, nu_hi
    nu = t2 = None
    for _ in range(rounds):
        best = (np.inf, None, None)
        for t2c in T2_GRID:
            dec = np.exp(-(delays / t2c) ** 2)
            for nuc in np.linspace(lo, hi, npts):
                A = np.column_stack([np.ones_like(delays),
                                     dec * np.cos(TWOPI * nuc * delays),
                                     dec * np.sin(TWOPI * nuc * delays)])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = float(np.sum((A @ c - y) ** 2))
                if r < best[0]:
                    best = (r, nuc, t2c)
        _, nu, t2 = best
        step = (hi - lo) / (npts - 1)
        lo, hi = nu - 2 * step, nu + 2 * step
    return nu, t2


def ramsey(f_drive, amp, beta, delays, shots, rng):
    """Ramsey走査1回: pi/2、待機、pi/2、読み出しです。"""
    return np.array([device([("p", np.pi / 2, "x"), ("i", T),
                             ("p", np.pi / 2, "x")],
                            f_drive, amp, beta, shots, rng) for T in delays])


F_ART = 2.0e-3          # 意図的な2 MHzのずれ、両側に加える
DELAYS = np.linspace(0.0, 2000.0, 41)

print("較正ループ II: Ramseyによる量子ビット周波数")
print("=" * 74)
print(f"  隠れた量子ビット周波数 : {HIDDEN['f_qubit']:.7f} GHz")
print(f"  準静的ばらつき sigma_f : {HIDDEN['sigma_f'] * 1e6:.0f} kHz ->"
      f" T2* = sqrt(2)/(2 pi sigma_f) ="
      f" {np.sqrt(2) / (TWOPI * HIDDEN['sigma_f']):.0f} ns")
f_est = HIDDEN["f_qubit"] + 1.5e-3          # ソフトウェアは1.5 MHz高い値から出発
print(f"  ソフトウェアの初期推定 : {f_est:.7f} GHz"
      f"  ({(f_est - HIDDEN['f_qubit']) * 1e6:+.0f} kHz のずれ)")
print(f"\n  1反復あたり2回の走査、{F_ART * 1e6:.0f} kHz を両側にずらして行います:")
print("  nu+ = |D - f_art|、nu- = |D + f_art| なので D = (nu- - nu+)/2、そして")
print("  (nu- + nu+)/2 は f_art を返すはずです。フィットの無料検算になります。")

rng = np.random.default_rng(90210)
print(f"\n  {'iter':>5} {'f_est (GHz)':>13} {'nu+ (kHz)':>10} {'nu- (kHz)':>10}"
      f" {'check (kHz)':>12} {'T2* (ns)':>9} {'residual (kHz)':>15}")
for it in range(1, 5):
    yp = ramsey(f_est + F_ART, amp_cal, 0.0, DELAYS, 2000, rng)
    ym = ramsey(f_est - F_ART, amp_cal, 0.0, DELAYS, 2000, rng)
    nup, t2p = fit_ramsey(DELAYS, yp, 5.0e-3)
    num, t2m = fit_ramsey(DELAYS, ym, 5.0e-3)
    delta = 0.5 * (num - nup)
    f_est = f_est + delta
    print(f"  {it:5d} {f_est:13.7f} {nup * 1e6:10.1f} {num * 1e6:10.1f}"
          f" {0.5 * (num + nup) * 1e6:12.1f} {0.5 * (t2p + t2m):9.0f}"
          f" {(f_est - HIDDEN['f_qubit']) * 1e6:+15.2f}")

f_cal = f_est
e_bad = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"],
                       f_drive=HIDDEN["f_qubit"] + 1.5e-3), XGATE)
e_ok = gate_error(run("gauss", TAU_G, np.pi, 0.0, amp_cal * HIDDEN["gain"],
                      f_drive=f_cal), XGATE)
print(f"\n  初期推定での pi パルスのゲート誤差: {e_bad:.3e}")
print(f"  較正後の f での pi パルスのゲート誤差: {e_ok:.3e}")
```

```text
較正ループ II: Ramseyによる量子ビット周波数
==========================================================================
  隠れた量子ビット周波数 : 4.9850876 GHz
  準静的ばらつき sigma_f : 200 kHz -> T2* = sqrt(2)/(2 pi sigma_f) = 1125 ns
  ソフトウェアの初期推定 : 4.9865876 GHz  (+1500 kHz のずれ)

  1反復あたり2回の走査、2000 kHz を両側にずらして行います:
  nu+ = |D - f_art|、nu- = |D + f_art| なので D = (nu- - nu+)/2、そして
  (nu- + nu+)/2 は f_art を返すはずです。フィットの無料検算になります。

   iter   f_est (GHz)  nu+ (kHz)  nu- (kHz)  check (kHz)  T2* (ns)  residual (kHz)
      1     4.9850834     3507.8      499.4       2003.6      1100           -4.19
      2     4.9850847     1997.8     2000.2       1999.0      1100           -2.94
      3     4.9850882     1995.5     2002.6       1999.1      1100           +0.63
      4     4.9850899     2001.0     2004.2       2002.6      1100           +2.25

  初期推定での pi パルスのゲート誤差: 4.962e-04
  較正後の f での pi パルスのゲート誤差: 2.975e-03
```

**注目点。** 1反復で1500 kHzの誤差が4.2 kHzになり、そのあとループはショットノイズと有限の $T_2^{\ast}$ が定める数kHzの水準をさまよいます。さらに反復しても得はなく、反復を続けるルーチンはノイズを追うために実機時間を使うことになります。検算は役目を果たします。$(\nu_- + \nu_+)/2$ は意図的に加えた2000 kHzに対して2004、1999、1999、2003 kHzを返しており、フィットが正しいフリンジ（エイリアスではないもの）に掛かっていることの無料のテストになっています。フィットされた $T_2^{\ast}$ は、$\sigma_f = 200$ kHz が $T_2^{\ast} = \sqrt{2}/(2\pi\sigma_f)$ を通じて含意する1125 nsに対して1100 nsと返ってきます。この関係は[量子ハードウェア入門 第1章](<../quantum-hardware-introduction/chapter-1.html>)が、静的な周波数ばらつきを母材の乱れについての言明として読むのに使うものと同じです。周波数較正ルーチンは、おまけに材料測定でもあるのです。

最後の2行は本章最良の偶然です。ゲート誤差は正しい周波数（$3.0\times10^{-3}$）より間違った周波数（$5.0\times10^{-4}$）のほうが*小さい*のです。1.5 MHzの離調による $Y$ 誤差が、欠けたDRAG重みによる $Y$ 誤差を部分的に打ち消すからです。較正されていない2つのパラメータが共謀して良く見せることは、制御スタックが操作者を騙す標準的な手口であり、パラメータを単一の性能指標ではなく、それを切り分けるシーケンスに対して較正する理由です。

### Code Example 7: 較正ループ III — DRAG重み

```python
"""第4章 Code Example 7: 較正ループ III — DRAG重み。
Code Example 6 の続き（同一セッション）。"""


def pingpong(beta, f_drive, amp, shots, rng):
    """鏡像の2シーケンス。読み出しが交差する点が正しい beta です。

    (X90, Y180) と (Y90, X180) は、残留する面外誤差が入る符号を除けば同じ回転
    です。したがって両者の差はその誤差について奇、それ以外について偶になります。
    とくに読み出し誤差は差から消えます。だから較正の観測量は、どちらかの読み出し
    ではなく差なのです。
    """
    a = [("p", np.pi / 2, "x"), ("p", np.pi, "y")]
    b = [("p", np.pi / 2, "y"), ("p", np.pi, "x")]
    return (device(a, f_drive, amp, beta, shots, rng)
            - device(b, f_drive, amp, beta, shots, rng))


def drag_calibration(betas, f_drive, amp, shots, rng):
    """beta を走査し、直線をフィットし、ゼロ交差点を返します。"""
    d = np.array([pingpong(b, f_drive, amp, shots, rng) for b in betas])
    slope, intercept = np.polyfit(betas, d, 1)
    return -intercept / slope, slope, d


print("較正ループ III: DRAG重み")
print("=" * 74)
b_opt = golden_min(lambda b: gate_error(
    run("gauss", TAU_G, np.pi, b, amp_cal * HIDDEN["gain"], f_drive=f_cal),
    XGATE), 0.0, 2.0)
print(f"  オラクル: ゲート誤差を最小にする beta は {b_opt:.4f} です。ループは")
print("  ゲート誤差を評価せず、ある差の符号が変わる点を探します。")

BETAS = np.linspace(0.4, 1.6, 7)
rng = np.random.default_rng(1357)
print(f"\nA. 観測量、1点あたり {8000} ショット")
print(f"  {'beta':>6} {'(X90,Y180)':>12} {'(Y90,X180)':>12} {'difference':>12}")
for b in BETAS:
    a = device([("p", np.pi / 2, "x"), ("p", np.pi, "y")], f_cal, amp_cal, b,
               8000, rng)
    c = device([("p", np.pi / 2, "y"), ("p", np.pi, "x")], f_cal, amp_cal, b,
               8000, rng)
    print(f"  {b:6.2f} {a:12.4f} {c:12.4f} {a - c:+12.4f}")

print("\nB. ショット予算に対する直線のゼロ交差点")
print("  予算ごとに独立な5シードを用い、残差の系統成分と")
print("  統計成分を切り分けます。")
print(f"  {'shots/pt':>9} {'slope':>8} {'mean beta':>10} {'mean resid':>11}"
      f" {'spread':>9} {'predicted':>10} {'mean gate error':>16}")
for shots in (500, 2000, 8000, 32000):
    bs, es, sl = [], [], []
    for seed in range(5):
        r = np.random.default_rng(1000 + 17 * seed)
        b_hat, slope, _ = drag_calibration(BETAS, f_cal, amp_cal, shots, r)
        bs.append(b_hat)
        sl.append(slope)
        es.append(gate_error(run("gauss", TAU_G, np.pi, b_hat,
                                 amp_cal * HIDDEN["gain"], f_drive=f_cal),
                             XGATE))
    # 各点のショットノイズは sqrt(2 p(1-p)/N) ~ sqrt(0.5/N)。7点の直線フィット
    # はこれを sqrt(7) で割る
    pred = np.sqrt(0.5 / shots) / (np.mean(sl) * np.sqrt(7.0))
    print(f"  {shots:9d} {np.mean(sl):8.4f} {np.mean(bs):10.4f}"
          f" {np.mean(bs) - b_opt:+11.4f} {np.std(bs):9.4f} {pred:10.4f}"
          f" {np.mean(es):16.3e}")
beta_cal = np.mean(bs)
print(f"\n  較正された beta = {beta_cal:.4f}、ゲート誤差"
      f" {gate_error(run('gauss', TAU_G, np.pi, beta_cal, amp_cal * HIDDEN['gain'], f_drive=f_cal), XGATE):.3e}")
```

```text
較正ループ III: DRAG重み
==========================================================================
  オラクル: ゲート誤差を最小にする beta は 0.9373 です。ループは
  ゲート誤差を評価せず、ある差の符号が変わる点を探します。

A. 観測量、1点あたり 8000 ショット
    beta   (X90,Y180)   (Y90,X180)   difference
    0.40       0.4541       0.5194      -0.0653
    0.60       0.4680       0.5121      -0.0441
    0.80       0.4808       0.4965      -0.0157
    1.00       0.4981       0.4836      +0.0145
    1.20       0.4966       0.4680      +0.0286
    1.40       0.5069       0.4589      +0.0480
    1.60       0.5324       0.4445      +0.0879

B. ショット予算に対する直線のゼロ交差点
  予算ごとに独立な5シードを用い、残差の系統成分と
  統計成分を切り分けます。
   shots/pt    slope  mean beta  mean resid    spread  predicted  mean gate error
        500   0.1205     0.9521     +0.0149    0.1451     0.0992        7.663e-05
       2000   0.1200     0.9410     +0.0038    0.0282     0.0498        7.484e-06
       8000   0.1282     0.9333     -0.0039    0.0206     0.0233        6.246e-06
      32000   0.1288     0.9333     -0.0040    0.0108     0.0116        5.198e-06

  較正された beta = 0.9333、ゲート誤差 4.806e-06
```

**注目点。** パートAは、なぜどちらかの読み出しではなく差が観測量なのかを示します。両シーケンスは0.5から数%の範囲の値を返し、単独ではどちらも情報になりません。両者の差は $-0.065$ から $+0.088$ まで走り、$\beta$ について線形です。読み出し誤差 — 2.0%と4.5% — は両シーケンスに共通で、差からは厳密に消えます。これが「構成による相殺」の原理の最も単純な形です。

パートBは予算ごとに5シードを使って統計と系統を切り分けます。ばらつきは $0.145, 0.028, 0.021, 0.011$ と落ち、ショットノイズの予測 $0.099, 0.050, 0.023, 0.012$ に対してスケールも大きさも合っています。平均残差はどの予算でもばらつきの内側にあり、振幅ループとは違ってここには停滞すべき系統誤差がありません。較正された $\beta = 0.9333$ はゲート誤差 $4.8\times10^{-6}$ を与え、オラクルの $\beta = 0.9373$ における $4.6\times10^{-6}$ に対して、$\beta$ の残差は制限要因ではありません。

*うまくいかないこと*も記録に値します。鏡像のパルス対を繰り返してもこの観測量は増幅されません。この対は回転の不動点に状態を返すので、感度は繰り返しの間で蓄積せずに相殺し、素朴な $n_{\mathrm{rep}}$ 走査はゼロに縮む傾きと無意味な交差点を生みます。誤差増幅は特定のシーケンスの性質であって、どれにでも適用できる小技ではありません。

### Code Example 8: 3つのループを通しで

```python
"""第4章 Code Example 8: 3つのループを通しで走らせる。
Code Example 7 の続き（同一セッション）。"""


def calibrate(rounds, shots, rng, f_start, amp_start, beta_start):
    """周波数、次に振幅、次にDRAG重み、これを繰り返します。履歴を返します。"""
    f, a, b = f_start, amp_start, beta_start
    trace = []
    for r in range(1, rounds + 1):
        yp = ramsey(f + F_ART, a, b, DELAYS, shots, rng)
        ym = ramsey(f - F_ART, a, b, DELAYS, shots, rng)
        f = f + 0.5 * (fit_ramsey(DELAYS, ym, 5.0e-3)[0]
                       - fit_ramsey(DELAYS, yp, 5.0e-3)[0])
        a, _ = rabi_calibration(STAGES, a, shots, rng, f, b)
        b, _, _ = drag_calibration(BETAS, f, a, 4 * shots, rng)
        U = run("gauss", TAU_G, np.pi, b, a * HIDDEN["gain"], f_drive=f)
        trace.append((r, f, a, b, U))
    return (f, a, b), trace


def report(label, f, a, b):
    U = run("gauss", TAU_G, np.pi, b, a * HIDDEN["gain"], f_drive=f)
    e = error_axis(U, XGATE)
    print(f"  {label:>9} {(f - HIDDEN['f_qubit']) * 1e6:+10.1f} {a:9.5f}"
          f" {b:7.4f} {e['X']:+9.1e} {e['Y']:+9.1e} {e['Z']:+9.1e}"
          f" {abs(U[2, 1]) ** 2:10.2e} {gate_error(U, XGATE):11.3e}")


print("3つのループを通しで走らせる")
print("=" * 74)
print("  出発点: 周波数が1.5 MHz高い、振幅が13.7%高い、DRAG重みが0。")
print("  順序: 周波数、振幅、DRAG、これを繰り返す。")
print(f"\n  {'round':>9} {'df (kHz)':>10} {'amp':>9} {'beta':>7} {'theta_x':>9}"
      f" {'theta_y':>9} {'theta_z':>9} {'leakage':>10} {'gate error':>11}")
F0, A0, B0 = HIDDEN["f_qubit"] + 1.5e-3, 1.0, 0.0
report("出発点", F0, A0, B0)
rng = np.random.default_rng(31415)
(f_f, a_f, b_f), trace = calibrate(3, 2000, rng, F0, A0, B0)
for r, f, a, b, U in trace:
    report(f"{r}", f, a, b)

b_star = golden_min(lambda bb: gate_error(run("gauss", TAU_G, np.pi, bb,
                                             a_f * HIDDEN["gain"],
                                             f_drive=HIDDEN["f_qubit"]),
                                         XGATE), 0.0, 2.0)
a_star = true_pi_amplitude(b_star, HIDDEN["f_qubit"])
report("オラクル", HIDDEN["f_qubit"], a_star, b_star)

n_shots = 3 * (2 * 41 * 2000 + (41 + 3 * 21) * 2000 + 7 * 2 * 8000)
print(f"\n  3ラウンドで費やした総ショット数: {n_shots:,}")
```

```text
3つのループを通しで走らせる
==========================================================================
  出発点: 周波数が1.5 MHz高い、振幅が13.7%高い、DRAG重みが0。
  順序: 周波数、振幅、DRAG、これを繰り返す。

      round   df (kHz)       amp    beta   theta_x   theta_y   theta_z    leakage  gate error
        出発点    +1500.0   1.00000  0.0000  +4.1e-01  +1.0e-01  +3.4e-16   2.30e-05   2.940e-02
          1       -7.5   0.88164  0.9420  +1.2e-03  -3.4e-04  -2.2e-16   4.42e-06   4.655e-06
          2       -0.2   0.88134  0.9365  +5.5e-05  +5.3e-05  -2.4e-16   4.46e-06   4.457e-06
          3       +0.9   0.88139  0.9464  +2.2e-04  -1.4e-03  -5.1e-18   4.37e-06   4.711e-06
       オラクル       +0.0   0.88133  0.9381  -7.4e-06  -1.8e-04  +2.2e-16   4.44e-06   4.448e-06

  3ラウンドで費やした総ショット数: 1,452,000
```

**注目点。** これが本章の目指したテストです。3つのパラメータを意図的に間違えて設定し — 周波数1.5 MHz高い、振幅13.7%高い、DRAG重みゼロ — 真値を読まない3つのループが3つすべてを回復します。1ラウンド後には周波数は7.5 kHz以内、振幅は $3\times10^{-4}$ 以内、そしてゲート誤差は $2.94\times10^{-2}$ から $4.66\times10^{-6}$ に落ちます。6300倍であり、この20 nsガウシアンが到達しうる $4.45\times10^{-6}$ の5%以内に着地しています。ラウンド2は $4.457\times10^{-6}$ に達し、3桁でオラクルそのものです。ラウンド3はわずかに悪く、これはドリフトではなく $\beta$ のフィットに乗ったショットノイズです。実際の制御スタックは、新しい値が統計的に有意な改善でない限り前の値を保持します。まさにノイズの中を歩き回るのを避けるためです。

3つの誤差列を横に読むと本節全体の構造が現れます。$\theta_x$ は $4.1\times10^{-1}$ から $5.5\times10^{-5}$、$\theta_y$ は $1.0\times10^{-1}$ から $5.3\times10^{-5}$、$\theta_z$ は最初から問題ではありませんでした。そしてリークの列はまったく改善しません — $2.3\times10^{-5}$ から $4.4\times10^{-6}$ で、それも振幅が変わったからにすぎません。**較正はリークを直せません。** リークはパルス長と非調和性が定めるものであり、これらは設計上の選択であって較正パラメータではありません。触れられるノブは、長いパルスか、よりよく設計された回路だけです。

代価ははっきり述べる価値があります。1量子ビットの1量子ビットゲートについて3ラウンドで145万ショット。実機では量子ビットあたり数分であり、装置がドリフトするたびに繰り返さねばなりません。較正は実機のデューティサイクルのかなりの部分を食うスケジュール済みバックグラウンドジョブであり、ベンダーが量子ビットごとではなく「中央値」の誤り率を公表する理由の一部は、量子ビットごとの数値が最後の較正パスの時点の鮮度しかもたないことです。

* * *

## 4.4 ベンチマーキング

### ゲートを測ることの難しさ

較正ループは改善が止まったことは分かりますが、ゲートがどれだけ良いかは分かりません。ゲートの測定はどれも汚染されています。状態は完全には準備されず、読み出しは忠実ではなく、Code Example 4 ではこの2つが合わさって*理想的な* $\pi$ パルスの読み出しを1から0.90に動かしました。$10^{-4}$ のゲート誤差を10%のずれを含む測定から取り出すことは、いくら平均してもできません。ずれはバイアスであり分散ではないからです。

**ランダマイズドベンチマーキング**は、値ではなく*減衰率*を測ることでこれを解決します。ランダムなCliffordを $m$ 個引いて合成し、その積を反転する唯一のCliffordを追加し、初期状態に戻る確率 — *生存確率* — を測ります。多数のランダムシーケンスについて繰り返して平均します。模型は

$$ F(m) = A\,p^{m} + B $$

であり、この構成の要点は、状態準備と測定の誤差が $A$ と $B$ にしか効かず、ゲートあたり誤差が $p$ にしか効かないことです。ゲート誤差は**Cliffordあたり誤差**として読み出されます。

$$ \mathrm{EPC} = \left(1 - \frac{1}{d}\right)(1-p) = \frac{1-p}{2} \quad (d = 2) $$

### なぜ効くのか

Clifford群についての平均は誤差チャネルを**ツワリング**します。任意のチャネル $\mathcal{E}$ について

$$ \bar{\mathcal{E}} = \frac{1}{\lvert \mathbb{C} \rvert}\sum_{C \in \mathbb{C}} \mathcal{C}^{-1} \circ \mathcal{E} \circ \mathcal{C} = \mathcal{D}_{p} $$

は $\mathcal{E}$ と同じ平均忠実度をもつ脱分極チャネルです。ツワリングされたチャネルを $m$ 個並べたものはパラメータ $p^m$ の脱分極チャネルであり、その強さの脱分極チャネルは、準備と読み出しだけで定まる $A$ と $B$ をもつ*厳密に* $A p^m + B$ の生存確率を生みます。SPAMが指数に入らないのは、SPAMが1回だけ起こりゲートが $m$ 回起こるからです。両者は $m$ についてのスケーリングで分離されるのであり、SPAMを較正で消す巧妙さによるのではありません。

### Code Example 9: ランダマイズドベンチマーキングと、SPAMを無視できる理由

```python
"""第4章 Code Example 9: ランダマイズドベンチマーキングと、SPAMを無視できる理由。
Code Example 8 の続き（同一セッション）。"""
SIG = [np.eye(2, dtype=complex), XGATE, PAULI3["Y"], PAULI3["Z"]]


def ptm(U):
    """1量子ビットユニタリのPauli転送行列。実の 4 x 4 行列です。"""
    return np.array([[0.5 * np.trace(SIG[i] @ U @ SIG[j] @ U.conj().T).real
                      for j in range(4)] for i in range(4)])


def canon(U):
    """2 x 2 ユニタリのグローバル位相を除いた指紋で、群の索引に使います。"""
    k = int(np.argmax(np.abs(U).ravel() > 1e-9)) if np.any(np.abs(U) > 1e-9) else 0
    z = U.ravel()[k]
    return tuple(np.round((U * np.conj(z) / abs(z)).ravel(), 6))


def clifford_group():
    """1量子ビットClifford群24元。H と S からグローバル位相を除いて閉じます。"""
    S_ = np.array([[1, 0], [0, 1j]], dtype=complex)
    H_ = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    seen, group = {canon(np.eye(2, dtype=complex)): 0}, [np.eye(2, dtype=complex)]
    frontier = [np.eye(2, dtype=complex)]
    while frontier:
        nxt = []
        for U in frontier:
            for G in (S_, H_):
                V = G @ U
                c = canon(V)
                if c not in seen:
                    seen[c] = len(group)
                    group.append(V)
                    nxt.append(V)
        frontier = nxt
    return group, seen


CLIFF, CLIFF_INDEX = clifford_group()
CLIFF_PTM = [ptm(U) for U in CLIFF]
print("ランダマイズドベンチマーキング")
print("=" * 74)
print(f"  1量子ビットClifford群: {len(CLIFF)} 元")


def depolarizing_ptm(eps):
    """平均ゲート非忠実度がちょうど eps になる脱分極チャネルです。"""
    return np.diag([1.0, 1.0 - 2 * eps, 1.0 - 2 * eps, 1.0 - 2 * eps])


def rb_survival(lengths, noise, seqs, rng, init_err, eps01, eps10):
    """RBシーケンスの平均生存確率です。各シーケンスは厳密に計算します。

    ノイズチャネルは反転用のCliffordを含むすべてのCliffordの後に作用させるので、
    長さ m のシーケンスはノイズを伴うゲートを m + 1 個含みます。
    """
    out = []
    for m in lengths:
        tot = 0.0
        for _ in range(seqs):
            idx = rng.integers(0, len(CLIFF), size=m)
            M = np.eye(4)
            U = np.eye(2, dtype=complex)
            for i in idx:
                M = noise @ CLIFF_PTM[i] @ M
                U = CLIFF[i] @ U
            inv = CLIFF_INDEX[canon(U.conj().T)]
            M = noise @ CLIFF_PTM[inv] @ M
            z = M[3, 3] * (1.0 - 2 * init_err) + M[3, 0]
            p0 = 0.5 * (1.0 + z)
            tot += p0 * (1 - eps01) + (1 - p0) * eps10
        out.append(tot / seqs)
    return np.array(out)


def fit_rb(lengths, y, rounds=3, npts=2001):
    """y = A p^m + B をフィットします。p を固定すると (A, B) について線形なので、
    p を走査して精密化します。

    A または B が (0, 1) の外に出る候補は棄却します。この保護がないと、p が1に
    近づくところでフィットは退化します。A p^m + B が定数に近づき、大きな A と
    B = -A の組がどれも同じようにデータに合ってしまうためです。制約なしの走査は
    平然と A = 2.6e4、B = -2.6e4、Cliffordあたり誤差 1e-10 を返します。どちらの
    係数も確率であり、そう述べることがフィットを適切な問題にします。
    """
    lo, hi = 0.5, 1.0
    out = (None, None, None)
    for _ in range(rounds):
        best = (np.inf, None, None, None)
        for p in np.linspace(lo, hi, npts):
            A = np.column_stack([p ** lengths, np.ones_like(y)])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if r < best[0] and 0.0 < c[0] < 1.0 and 0.0 < c[1] < 1.0:
                best = (r, p, c[0], c[1])
        _, p, a, b = best
        step = (hi - lo) / (npts - 1)
        lo, hi = p - 2 * step, min(p + 2 * step, 1.0)
        out = (p, a, b)
    return out


LENGTHS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
print("\nA. 設定した誤差がそのまま返ってくる")
print(f"  {'configured EPC':>15} {'fitted p':>10} {'EPC = (1-p)/2':>15}"
      f" {'ratio':>8} {'A':>8} {'B':>8}")
for eps in (1e-4, 1e-3, 5e-3, 2e-2):
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, depolarizing_ptm(eps), 60, rng, 0.012, 0.020, 0.045)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {eps:15.1e} {p:10.6f} {(1 - p) / 2:15.3e}"
          f" {(1 - p) / 2 / eps:8.4f} {A:8.4f} {B:8.4f}")

print("\nB. 同じゲート、3通りの読み出し誤差と初期化誤差")
print(f"  {'init err':>9} {'eps01':>7} {'eps10':>7} {'F(m=1)':>9} {'A':>8}"
      f" {'B':>8} {'fitted EPC':>12} {'ratio':>8}")
for ie, e01, e10 in ((0.000, 0.000, 0.000), (0.012, 0.020, 0.045),
                     (0.050, 0.100, 0.150)):
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, depolarizing_ptm(1e-3), 60, rng, ie, e01, e10)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {ie:9.3f} {e01:7.3f} {e10:7.3f} {y[0]:9.4f} {A:8.4f} {B:8.4f}"
          f" {(1 - p) / 2:12.3e} {(1 - p) / 2 / 1e-3:8.4f}")

print("\nC. ツワリングが効く理由と、確率的誤差ではなくコヒーレント誤差の場合")


def clifford_twirl(noise):
    """チャネルをClifford群で平均します: (1/24) sum_C C^-1 N C。

    ユニタリのPTMは直交行列なので、逆行列は転置です。結果は常に脱分極チャネルに
    なり、これがRBの依拠する定理です。誤差が何であれ、シーケンス平均はその平均
    忠実度だけを見ます。
    """
    return sum(M.T @ noise @ M for M in CLIFF_PTM) / len(CLIFF_PTM)


print(f"  {'over-rotation':>14} {'exact infidelity':>17} {'twirled channel':>16}"
      f" {'off-diagonal':>13} {'RB, 200 seqs':>13} {'ratio':>8}")
for phi in (0.03, 0.05, 0.10, 0.20, 0.30):
    V = expm(-0.5j * phi * XGATE)
    r_exact = (2.0 - abs(np.trace(V)) ** 2 / 2.0) / 3.0
    Nbar = clifford_twirl(ptm(V))
    off = float(np.max(np.abs(Nbar - np.diag(np.diag(Nbar)))))
    rng = np.random.default_rng(11)
    y = rb_survival(LENGTHS, ptm(V), 200, rng, 0.012, 0.020, 0.045)
    p, A, B = fit_rb(LENGTHS, y)
    print(f"  {phi:14.3f} {r_exact:17.3e} {(1 - Nbar[1, 1]) / 2:16.3e}"
          f" {off:13.1e} {(1 - p) / 2:13.3e} {(1 - p) / 2 / r_exact:8.4f}")
```

```text
ランダマイズドベンチマーキング
==========================================================================
  1量子ビットClifford群: 24 元

A. 設定した誤差がそのまま返ってくる
   configured EPC   fitted p   EPC = (1-p)/2    ratio        A        B
          1.0e-04   0.999800       1.000e-04   1.0000   0.4562   0.5125
          1.0e-03   0.998000       1.000e-03   1.0000   0.4554   0.5125
          5.0e-03   0.990000       5.000e-03   1.0000   0.4517   0.5125
          2.0e-02   0.960000       2.000e-02   1.0000   0.4380   0.5125

B. 同じゲート、3通りの読み出し誤差と初期化誤差
   init err   eps01   eps10    F(m=1)        A        B   fitted EPC    ratio
      0.000   0.000   0.000    0.9980   0.4990   0.5000    1.000e-03   1.0000
      0.012   0.020   0.045    0.9670   0.4554   0.5125    1.000e-03   1.0000
      0.050   0.100   0.150    0.8612   0.3368   0.5250    1.000e-03   1.0000

C. ツワリングが効く理由と、確率的誤差ではなくコヒーレント誤差の場合
   over-rotation  exact infidelity  twirled channel  off-diagonal  RB, 200 seqs    ratio
           0.030         1.500e-04        1.500e-04       1.1e-17     1.336e-04   0.8907
           0.050         4.166e-04        4.166e-04       1.1e-17     4.442e-04   1.0664
           0.100         1.665e-03        1.665e-03       1.1e-17     1.802e-03   1.0824
           0.200         6.644e-03        6.644e-03       1.2e-17     6.059e-03   0.9119
           0.300         1.489e-02        1.489e-02       1.2e-17     1.451e-02   0.9744
```

**注目点。** この例の性質が1つ、3つのパートすべてを枠づけます。4.3節のどのルーチンとも違い、`rb_survival` は各シーケンスの生存確率を*厳密に*計算します。docstringにそう書いてあるとおりで、以下のどこにもショットノイズはありません。現れるばらつきはランダムなCliffordのシーケンス間変動であり、サンプリング誤差ではありません。実際の実験では両方が乗ります。

パートAは正しさのテストです。Cliffordあたり誤差を $10^{-4}$、$10^{-3}$、$5\times10^{-3}$、$2\times10^{-2}$ と設定すると、フィットされたEPCはどの行でも比1.0000で設定値を返します。4桁にわたってです。

パートBがSPAMについての主張で、1列ではなく3列として読む価値があります。完全な測定から初期化誤差5%・読み出し誤差10%と15%へ移ると、$m = 1$ での生の生存確率は0.9980から0.8612へ、14パーセントポイント近く動きます。測っているゲート誤差の137倍です。同じ3行で $A$ は0.4990から0.3368へ、$B$ は0.5000から0.5250へ動きます。そしてフィットされたCliffordあたり誤差は3行すべてで4桁一致の $1.000\times10^{-3}$ です。SPAMは完全に $A$ と $B$ に住み、ゲート誤差は完全に $p$ に住みます。この分離が、直接の忠実度測定ではなくRBがあらゆるハードウェアグループの報告する数値である理由です。

パートCはツワリングを明示し、それを検証します。3列目は24個すべてのCliffordについて平均した*厳密に*ツワリングされたチャネルの平均非忠実度で、2列目のコヒーレント過回転の厳密な平均非忠実度と表示桁すべてで一致し、ツワリング後のPauli転送行列の非対角成分は $10^{-17}$ まで消えています。これがRBの依拠する定理です。5列目は200個のランダムシーケンスが実際に返す値で、10%程度で一致しています。

### RBが語らないこと

3つの限界があり、それぞれ誰かを誤らせてきました。

  * **群についての平均であって最悪値ではなく、コヒーレント誤差と確率的誤差を区別できません。** パートCの過回転は完全にユニタリですが、RBは同じ平均忠実度の脱分極誤差として報告します。これが問題になるのは、コヒーレント誤差は深い回路で*振幅*として蓄積し、確率的誤差は確率として蓄積するからです。同じEPCの2つのゲートが深さ1000ではまったく違う振る舞いをしえます。パートCのばらつきが同程度の確率的誤差の場合より小さなコヒーレント誤差で大きいのも、シーケンスごとの生存確率がシーケンス間で大きく振れるからです。減衰が指数的なのは平均を取った後だけで、取る前はそうではありません。
  * **リークが見えません。** ツワリングは量子ビット部分空間に作用するClifford群についてのものであり、その部分空間を出た population は模型がまったく記述しません。Code Example 3 の $\beta = 2.0030$ のパルスはリークが $8\times10^{-11}$、ゲート誤差は最適値の334倍悪いのですが、標準的なRBのフィットはこれを、同じ平均忠実度でリークのないパルスと区別できません。リークには専用の実験が必要です。
  * **測っているのは*Clifford*の誤差であって、関心のあるゲートの誤差ではありません。** EPCは24元についての平均であり、各元は1つか2つの物理パルスにコンパイルされます。特定のゲートの誤差を取り出すにはインターリーブRB — すべてのCliffordの間に対象ゲートを挿入して同じ実験を走らせ、2つの減衰定数の比を取る — が必要で、この測定は両方のフィットの不確かさを引き継ぎます。

* * *

## 4.5 ソフトウェアとしての較正層

本章のすべてはプログラムです。そしてインタフェースに名前を付けておく価値があります。どんな名前で呼ばれていようと、あらゆるSDKが露出しているものだからです。

**ゲート層の下にはパルス層があり、それは別種のオブジェクトです。** ゲート列は離散でハードウェア非依存、厳密に合成可能です。パルススケジュールは連続で装置固有、サンプルレート・メモリ深さ・チャネル数に制約されます。両者の境界が*較正テーブル*です。ゲート名と量子ビット番号から、波形とそのパラメータへの写像です。コンパイラはゲート名を出し、制御スタックがそれを引きます。だから同じ回路が較正パスの前後で走り、記号を1つも変えずに違う誤り率を出すのです。

**較正パラメータは状態であり、状態はドリフトします。** 今朝正しかった周波数は午後には数十kHzずれ、振幅はアッテネータの温度に従います。したがってあらゆる実運用スタックは検査の階層を走らせます。高価な再較正が必要かを判断する安価な検査と、100個の量子ビットのうちどれに次の1分を使うかを決めるスケジューラです。この主題の文献は物理よりはるかに*判断方針*についてのものです。

**パルス層は、ユーザがコンパイラにできないことをできる場所です。** パルスレベルAPIが存在する理由は、ベンダーが提供しなかったゲートを実装できるようにするため、ゲート抽象では表現できないシーケンス — 待機中の動的デカップリング、独自の2量子ビットゲート、分光走査 — を走らせるため、あるいはゲート抽象が隠している装置の何かを測るためです。SDKのドキュメントで対応する項目は、ベンダー中立な言い方をすれば*パルススケジュールビルダ*、*バックエンド較正データ*、*チャネルマップ*、そしてRabi・Ramsey・DRAG・RBのルーチンからなる*実験ライブラリ*です。そのどれもが本章がゼロから作ったものであり、それが要点でした。APIとは、ここですでに構成したオブジェクトへの名前の集合なのです。

* * *

## 演習

#### 演習1: 非調和性が速度制限を与える

$E_C/h = 0.180$ GHz、$E_J/h = 19.0$ GHz で設計された別の装置を考えます。$\lvert \alpha \rvert$ が小さくなります。

  1. 新しいトランズモンを対角化し、$f_{01}$、$\alpha/2\pi$、$r_1$ を報告してください。
  2. 両方の装置について $\tau = 20$ ns で $\mathrm{sinc}^{2}(\alpha\tau)$ を評価してください。どちらの装置が矩形パルスでより漏れるか、これで分かりますか。注意して答えてください。
  3. 新しい装置の $\tau = 20$ ns でのガウシアンのリークを実測し、本章の $1.76\times10^{-5}$ と比べてください。
  4. 新しい装置で元のリークを回復する長さは何nsですか。答えを $\lvert \alpha \rvert \tau$ についての言明として解釈してください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(f_{01} = 5.0438\) GHz、\(\alpha/2\pi = -196.64\) MHz、\(r_1 = 1.3861\)。相対非調和性は5.71%から3.90%に落ち、\(r_1\) は \(\sqrt{2}\) に近づきます。はしごが調和的になったということで、これはゲートが難しくなる方向です。</p>

<p><strong>2.</strong> 分かりません。ここが罠です。元の装置は \(\alpha\tau = -5.697\) で \(\mathrm{sinc}^2 = 2.07\times10^{-3}\)、新しい装置は \(\alpha\tau = -3.933\) で \(\mathrm{sinc}^2 = 2.88\times10^{-4}\) であり、非調和性が悪いのに7倍<em>小さい</em>のです。sincは振動しており、\(\alpha\tau = -3.93\) はたまたまそのゼロの近くに座っています。振動する関数を1点で比べることは無意味であり、物理的な言明は包絡のスケーリング \((\alpha\tau)^{-2}\) だけです。2つの装置を1つの長さで比べることはそれを検証する方法ではありません。</p>

<p><strong>3.</strong> \(1.76\times10^{-5}\) に対して \(7.76\times10^{-5}\)、4.4倍悪化します。ガウシアンの裾は \(\alpha\tau\) についてべき乗ではなく指数的なので、矩形パルスの見積りが示唆するよりはるかに非調和性に敏感です。</p>

<p><strong>4.</strong> 29 ns で、そこでの実測リークは \(1.80\times10^{-5}\) — 元の装置の値の2%以内です。元の装置は \(\lvert \alpha \rvert \tau = 5.70\) で、新しい装置は \(0.19664 \times 29 = 5.70\) に達します。無次元の積が性能指標なので、31%小さい非調和性は45%長いゲートを買うことになり、あらゆるデコヒーレンス経路が45%長く作用します。これが超伝導の1量子ビットゲートを数十ナノ秒に固定しているトレードオフです。</p>

```python
import numpy as np
E2, nop2 = transmon_eigen(19.0, 0.180)
a2 = (E2[2] - E2[1]) - E2[1]
print(f"f01 = {E2[1]:.4f} GHz   alpha/2pi = {a2*1e3:.2f} MHz"
      f"   r1 = {abs(nop2[1, 2]/nop2[0, 1]):.4f}")
for name, aa in (("old", alpha), ("new", a2)):
    x = np.pi * aa * 20.0
    print(f"  {name}: alpha*tau = {aa*20:6.3f}   sinc^2 = {(np.sin(x)/x)**2:.3e}")
LOW2 = np.zeros((NLEV, NLEV), dtype=complex)
for j in range(NLEV - 1):
    LOW2[j, j + 1] = abs(nop2[j, j + 1] / nop2[0, 1])
E_s, AX_s, AY_s, al_s = E, AX, AY, alpha
E, alpha = E2, a2
AX = TWOPI * 0.5 * (LOW2 + LOW2.conj().T)
AY = TWOPI * 0.5 * 1j * (LOW2.conj().T - LOW2)
for tau in (20.0, 29.0):
    U = run("gauss", tau, np.pi, 0.0, 1.0, f_drive=E2[1])
    print(f"  新しい装置、{tau:4.1f} ns のガウシアン pi パルス:"
          f" leakage = {abs(U[2, 1])**2:.3e}")
E, AX, AY, alpha = E_s, AX_s, AY_s, al_s
print(f"  |alpha|*tau が等しくなる長さ: {abs(alpha)*20.0/abs(a2):.1f} ns")
# f01 = 5.0438 GHz   alpha/2pi = -196.64 MHz   r1 = 1.3861
#   old: alpha*tau = -5.697   sinc^2 = 2.067e-03
#   new: alpha*tau = -3.933   sinc^2 = 2.883e-04
#   新しい装置、20.0 ns のガウシアン pi パルス: leakage = 7.757e-05
#   新しい装置、29.0 ns のガウシアン pi パルス: leakage = 1.796e-05
#   |alpha|*tau が等しくなる長さ: 29.0 ns
```

</details>

#### 演習2: 振幅ループはなぜ停滞したか

Code Example 5 の振幅ループは $\beta = 0$ で7 mradの系統誤差に停滞し、$\beta = 0.9382$ で収束しました。

  1. $\beta = 0, 0.5, 0.9382, 1.4$ で無限ショット予算のループを走らせ、停滞値を対応するパルスの $\theta_y$ とともに表にしてください。関係式は何ですか。
  2. その関係式から、$\theta_y$ の符号は効きますか。それは機構について何を語りますか。
  3. 同僚が、純粋なコサインの代わりにコサイン＋線形ドリフトでフィットすれば停滞が直ると提案しました。効きますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 停滞値は \(\theta_y = +0.1330, +0.0620, -0.0002, -0.0657\) rad に対して \(+7.10, +1.61, +0.05, +1.68\) mrad です。停滞値を \(\theta_y^{2}\) で割ると、非ゼロの3行について401、420、389 mrad/rad\(^2\) となります。停滞は面外誤差について2次であり、係数は4%の範囲で一定です。</p>

<p><strong>2.</strong> 効きません。停滞は \(\theta_y\) の両符号で正です。符号に依らない2次の汚染は、付加的な回転ではなく<em>軸の傾き</em>の兆候です。回転軸を赤道面から角 \(\epsilon\) 傾けると、測定軸への軌跡の射影が \(\cos\epsilon \approx 1 - \epsilon^2/2\) 倍に縮み、コサインのフィットはこれを、どちら向きに傾いたかに関わらず少し違うフリンジ周波数として報告します。</p>

<p><strong>3.</strong> 効きません。汚染は振幅方向のドリフトではなく、\(\theta_y\) について偶で繰り返し数とともに増大するフリンジ振幅の歪みです。線形項を加えればフィットにノイズを吸収する自由度を1つ与えるだけで、バイアスは残ります。唯一の修正は物理的原因を除くことであり、それが Code Example 8 でループを反復することです。</p>

```python
for b in (0.0, 0.5, 0.9382, 1.4):
    at = true_pi_amplitude(b, f01)
    e = 1.0
    for n in STAGES:
        e, _ = rabi_calibration((n,), e, None, None, f01, b)
    ty = error_axis(run("gauss", TAU_G, np.pi, b), XGATE)["Y"]
    print(f"beta = {b:6.4f}  theta_y = {ty:+8.4f} rad"
          f"  stall = {(e - at)*np.pi*1e3:+6.2f} mrad"
          f"  stall/theta_y^2 = {(e - at)*np.pi*1e3/ty**2:7.1f}")
# beta = 0.0000  theta_y =  +0.1330 rad  stall =  +7.10 mrad  stall/theta_y^2 =   401.1
# beta = 0.5000  theta_y =  +0.0620 rad  stall =  +1.61 mrad  stall/theta_y^2 =   420.0
# beta = 0.9382  theta_y =  -0.0002 rad  stall =  +0.05 mrad  stall/theta_y^2 = 1263089.4
# beta = 1.4000  theta_y =  -0.0657 rad  stall =  +1.68 mrad  stall/theta_y^2 =   388.6
```

</details>

#### 演習3: 間違った包絡でRamseyフリンジをフィットする

Code Example 6 はガウシアン包絡 $\exp[-(T/T_2^{\ast})^2]$ でフィットしており、これは準静的ノイズに対して正しい形です。

  1. 同じ走査を指数包絡 $\exp(-T/T_2^{\ast})$ で再フィットしてください。フリンジ周波数はどれだけ動きますか。時定数はどれだけ動きますか。
  2. なぜ周波数は包絡の模型に対して頑健で、時定数はそうでないのですか。
  3. [量子ハードウェア入門 第1章](<../quantum-hardware-introduction/chapter-1.html>)は $\sigma = \sqrt{2}/T_2^{\ast}$ で $T_2^{\ast}$ を静的な周波数ばらつきに変換します。両方のフィットについてこれを行い、模型の選択が推定される材料物性にどれだけの代価を課すか述べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> フリンジ周波数は3507.8から3503.1 kHzへ — 4.7 kHz、0.13% — 動き、時定数は1100 nsから900 nsへ、素朴な予想とは逆向きに18%動きます。</p>

<p><strong>2.</strong> 周波数は振動の<em>位相</em>に担われており、包絡は実で正の緩やかに変化する乗数なので、1次ではゼロ交差を動かしません。時定数は包絡<em>そのもの</em>なので、関数形を誤ればそのまま乗ります。ガウシアンと指数は \(T = 0\) 近傍の曲率で最も違い、そこはSN比が最良でフィットの重みが最も大きい場所です。</p>

<p><strong>3.</strong> \(\sigma/2\pi = \sqrt{2}/(2\pi T_2^{\ast})\) はガウシアンのフィットから205 kHz（<code>HIDDEN</code> に設定した200 kHzに対して）、指数のフィットから250 kHzを与えます。母材の推定物性に25%の誤りが、フィットルーチンの1行だけによって生じるのです。引用された \(T_2^{\ast}\) がフィット模型を併記しなければ解釈できないと姉妹コースが主張する、具体的な理由がこれです。</p>

```python
rng = np.random.default_rng(90210)
yr = ramsey(HIDDEN["f_qubit"] + 1.5e-3 + F_ART, amp_cal, 0.0, DELAYS, 2000, rng)


def fit_exp(delays, y, nu_hi, rounds=2, npts=401):
    lo, hi, out = 0.0, nu_hi, (None, None)
    for _ in range(rounds):
        best = (np.inf, None, None)
        for t2 in np.linspace(300.0, 3000.0, 28):
            dec = np.exp(-delays / t2)
            for nu in np.linspace(lo, hi, npts):
                A = np.column_stack([np.ones_like(delays),
                                     dec * np.cos(TWOPI * nu * delays),
                                     dec * np.sin(TWOPI * nu * delays)])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = float(np.sum((A @ c - y) ** 2))
                if r < best[0]:
                    best = (r, nu, t2)
        _, nu, t2 = best
        step = (hi - lo) / (npts - 1)
        lo, hi = nu - 2 * step, nu + 2 * step
        out = (nu, t2)
    return out


for name, f in (("gaussian", fit_ramsey), ("exponential", fit_exp)):
    nu, t2 = f(DELAYS, yr, 5.0e-3)
    print(f"{name:>12}: nu = {nu*1e6:7.1f} kHz   T2* = {t2:6.0f} ns"
          f"   sigma/2pi = {np.sqrt(2)/(TWOPI*t2)*1e6:5.0f} kHz")
#     gaussian: nu =  3507.8 kHz   T2* =   1100 ns   sigma/2pi =   205 kHz
#  exponential: nu =  3503.1 kHz   T2* =    900 ns   sigma/2pi =   250 kHz
```

</details>

#### 演習4: ランダマイズドベンチマーキングのフィットを読み違える2通り

  1. 0.01 radのコヒーレント過回転をベンチマークし、$0 < A < 1$、$0 < B < 1$ の保護なし・ありの両方でフィットしてください。厳密な平均非忠実度 $2\sin^{2}(\phi/2)/3$ と比べてください。
  2. あるゲートのCliffordあたり誤差が $10^{-3}$ で、そのうち半分が量子ビット部分空間からのリークだとします。本章のRB模型は何を報告し、何が欠けますか。
  3. インターリーブRB実験が $p_{\mathrm{ref}} = 0.9980$、$p_{\mathrm{int}} = 0.9955$ を与えました。インターリーブされたゲートの誤差はいくらで、その数値の主要な系統誤差は何ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 保護なしの走査は \(A = 2.58\times10^{4}\)、\(B = -2.58\times10^{4}\)、Cliffordあたり誤差 \(2.8\times10^{-10}\) を返します。厳密値 \(1.67\times10^{-5}\) に対して5桁の間違いです。\(p \to 1\) では模型 \(Ap^m + B\) は定数に退化し、大きな \(A\) と \(B \approx -A\) の組はどれもほぼ平らなデータに同じくらい合ってしまいます。保護を入れると \(7.5\times10^{-6}\) を返しますが、これはまだ2.2倍小さく、\(B \to 0\) の境界に座っています。保護は問題を可識別にしますが、200シーケンスでは \(10^{-5}\) のコヒーレント誤差は分解できません。両方の係数が確率であることは物理的な制約であり、それを課すのは数値上の便宜ではありません。</p>

<p><strong>2.</strong> Cliffordあたり誤差 \(10^{-3}\) を報告し、平均非忠実度としてはそれは正しく、そしてリークについては何も語りません。漏れた population がどう入るかは読み出しが \(\lvert 2 \rangle\) をどう分類するかに依ります。\(\lvert 2 \rangle\) が1と読まれるならリークは通常の脱分極のように見えて \(p\) の中に隠れ、0と読まれるなら減衰に第2の指数が加わり、単一指数のフィットはそれを平均してしまいます。どちらにしても運用上決定的な事実 — 誤差が符号空間の外にあり、したがって誤り訂正符号が直せるものの外にあるということ — は見えず、専用のリーク実験が必要です。</p>

<p><strong>3.</strong> \(r = (1 - p_{\mathrm{int}}/p_{\mathrm{ref}})(1 - 1/d) = (1 - 0.9955/0.9980)/2 = 1.25\times10^{-3}\)。主要な系統誤差は、参照シーケンスとインターリーブシーケンスが同じツワリングを標本していないことです。対象ゲートを挿入すると、どのCliffordがどれの隣に来るか、どうコンパイルされるかが変わるので、2つの減衰定数は同じチャネルの測定ではありません。したがって公表されたインターリーブRBの不確かさは、\(p_{\mathrm{int}}\) が \(p_{\mathrm{ref}}\) に近いときは常に測定量と同程度の系統誤差を抱えます。そしてそこが面白い領域なのです。</p>

```python
V = expm(-0.005j * XGATE)
rq = np.random.default_rng(11)
yq = rb_survival(LENGTHS, ptm(V), 200, rq, 0.012, 0.020, 0.045)
lo, hi = 0.5, 1.0
for _ in range(3):
    best = (np.inf, None, None, None)
    for p in np.linspace(lo, hi, 2001):
        A = np.column_stack([p ** LENGTHS, np.ones_like(yq)])
        c, *_ = np.linalg.lstsq(A, yq, rcond=None)
        r = float(np.sum((A @ c - yq) ** 2))
        if r < best[0]:
            best = (r, p, c[0], c[1])
    _, p, a, b = best
    step = (hi - lo) / 2000
    lo, hi = p - 2 * step, min(p + 2 * step, 1.0)
print(f"  保護なし: A = {a:.3e}  B = {b:.3e}  EPC = {(1-p)/2:.3e}")
p2, a2f, b2 = fit_rb(LENGTHS, yq)
print(f"  保護あり: A = {a2f:.4f}  B = {b2:.4f}  EPC = {(1-p2)/2:.3e}")
print(f"  厳密な平均非忠実度 = {(2 - abs(np.trace(V))**2/2)/3:.3e}")
print(f"  インターリーブされたゲートの誤差 = {(1 - 0.9955/0.9980)/2:.3e}")
#   保護なし: A = 2.578e+04  B = -2.578e+04  EPC = 2.813e-10
#   保護あり: A = 0.9687  B = 0.0000  EPC = 7.512e-06
#   厳密な平均非忠実度 = 1.667e-05
#   インターリーブされたゲートの誤差 = 1.253e-03
```

</details>

#### 演習5: 再較正スケジュールを設計する

ある装置がドリフトします。量子ビット周波数は毎時40 kHz、増幅ゲインは毎時0.15%です。較正済みのゲート誤差は $4.5\times10^{-6}$ で、目標は $1\times10^{-4}$ 未満に保つことです。

  1. 本章の感度から、各パラメータを較正しないまま置ける時間はどれだけですか。
  2. 再較正の周期を決めるのはどのループですか。
  3. 3ループ1ラウンドはオーバーヘッド込み毎秒5000ショットで48万ショットかかります。この周期で1量子ビットの1量子ビットゲートを較正するのに装置時間の何%を使いますか。1000量子ビットの機械では何を意味しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> ゲインのドリフト \(g\) は回転角誤差 \(\theta_x \approx \pi g\) を生み、小さなコヒーレント回転誤差の平均ゲート誤差は \(\approx \theta_x^2/6\) です。\(10^{-4}\) に達するのは \(\theta_x = 0.0245\) rad、すなわち \(g = 0.78\%\) で、毎時0.15%なら5.2時間です。周波数については、本章は1.5 MHzの離調でゲート誤差 \(4.96\times10^{-4}\) を実測しました。2次でスケールさせると \(10^{-4}\) は674 kHzで到達し、毎時40 kHzなら17時間です。</p>

<p><strong>2.</strong> 振幅で、3倍の差があります。これは超伝導ハードウェアの一般的な事情でもあります。振幅と位相の較正は周波数の較正よりはるかに頻繁に走らされ、周波数の較正はしばしばスケジュールではなくトリガで走ります。</p>

<p><strong>3.</strong> 48万ショットを毎秒5000で96秒、5.2時間ごとなので量子ビットあたり装置時間の0.51%です。1000量子ビットを1つずつ較正すれば使える時間の5.1倍になります。だから較正は量子ビット間で強く並列化され、完全な再較正が必要かを判断する安価な検査がこれほど重要になり、そして（本章では扱わず、ペアあたりのコストがかなり高い）2量子ビットゲートの較正が真の予算問題になるのです。ここでの算術が、大規模機の誤り率がすべて新鮮に測られたものではない誠実な理由です。</p>

```python
th = np.sqrt(6 * 1e-4)
print(f"  theta_x の許容 {th:.4f} rad -> ゲイン {th/np.pi*100:.2f} %"
      f" -> {th/np.pi*100/0.15:.1f} h")
print(f"  周波数の許容 {1500*np.sqrt(1e-4/4.96e-4):.0f} kHz"
      f" -> {1500*np.sqrt(1e-4/4.96e-4)/40:.0f} h")
duty = 480000 / 5000 / 3600 / (th / np.pi * 100 / 0.15)
print(f"  量子ビットあたり占有率 {duty*100:.2f} %  -> x1000 = {duty*1000:.1f}")
#   theta_x の許容 0.0245 rad -> ゲイン 0.78 % -> 5.2 h
#   周波数の許容 674 kHz -> 17 h
#   量子ビットあたり占有率 0.51 %  -> x1000 = 5.1
```

</details>

* * *

## まとめ

### 要点

**1\. ゲートとは較正されたパルス面積であり、$Z$ は無料である**

  * 回転系では共鳴駆動が量子ビットを搬送位相の定める軸まわりに $\theta = \int \Omega\,dt$ 回転させます。包絡の形は自由で、パルス層の唯一の自由度です。
  * $Z$ 回転は以降のすべてのパルスの位相の定義変更です。時間ゼロ、誤差ゼロ。$Z$ 回転を末尾に押し出すコンパイラの物理的な見返りがこれです。
  * トランズモンの第3準位は $\lvert \alpha \rvert = 285$ MHz 離れたところにあり、行列要素は*より大きい*。調和振動子の $\sqrt{2}$ に対して $r_1 = 1.3726$ です。

**2\. リークはFourier係数なので、整形は指数を変える**

  * $a_{1\to2} \simeq -(i r_1/2)\tilde{\Omega}_x(\alpha)$。矩形パルスのsinc裾は実測でリーク $\propto \tau^{-1.97}$、ガウシアンは $\tau^{-4.09}$ を与えます。
  * $\tau = 20$ nsで両者は205倍違い、$\tau$ を倍にすると矩形では3.9倍、ガウシアンでは17倍の改善です。
  * 整形が効くのはパルスが整形できるだけ長いときだけです。$\tau = 8$ nsではガウシアンのほうが*悪く*、面積固定でピーク振幅が117 MHz、矩形の62 MHzに対して大きいからです。

**3\. DRAGは位相を直し、その2つの最適点は異なる**

  * $\Omega_y = -\beta\dot{\Omega}_x/2\alpha$。$\tau = 20$ nsでゲート誤差の最適点は $\beta = 0.9382$ で、誤差を $2.97\times10^{-3}$ から $1.14\times10^{-5}$ へ261倍改善します。
  * リークの最適点は $\beta = 2.0030$ で、そこではリークが $8\times10^{-11}$、ゲートは**334倍悪い**。リークを最小化してDRAGを調整するのは間違った量の最適化です。
  * $\beta^{\ast}$ は解析的な1ではなく、$\tau = 8$ から40 nsで0.912から0.941まで動きます。だからこれは較正されるパラメータなのです。

**4\. 較正ループはシーケンス・フィット・更新であり、ループは互いに結合している**

  * 誤差増幅: $\pi$ パルス81回の繰り返しは、同じショット予算で1回の場合より振幅推定を約50倍鋭くします。
  * 構成による相殺: DRAGのピンポン観測量は2シーケンスの*差*であり、2.0%と4.5%の読み出し誤差が厳密に消えます。
  * 振幅ループは $\beta$ が間違っていると7 mradの系統誤差に停滞し、その停滞は無限ショット予算でも同じです。これが系統と統計を切り分ける診断です。$\beta$ が正しければ0.11 mradに達します。
  * Ramseyに待機が必要なのは $\pi$ パルスが $Z$ 誤差をまったく生まないからです。意図的にずらした2回の走査で符号が回復し、おまけに $(\nu_-+\nu_+)/2 = f_{\mathrm{art}}$ が無料の検算になります。
  * 通しで: 周波数1.5 MHz高い、振幅13.7%高い、$\beta = 0$ — 1ラウンドでゲート誤差 $4.66\times10^{-6}$（到達可能な $4.45\times10^{-6}$ に対して）まで回復。6300倍の改善で、代価は145万ショットです。

**5\. 較正はリークを直せず、2つの間違いは正しく見えうる**

  * 3ラウンドを通してリークは $4\times10^{-6}$ のままです。リークは $\tau$ と $\alpha$ が定めるもので、これらは設計パラメータであって較正パラメータではありません。
  * ゲート誤差は正しい周波数（$3.0\times10^{-3}$）より間違った周波数（$5.0\times10^{-4}$）で*小さかった*。離調誤差が欠けたDRAGを部分的に打ち消したからです。パラメータを切り分けるシーケンスに対して較正すべきで、単一の性能指標に対してではありません。

**6\. ランダマイズドベンチマーキングは $m$ についてのスケーリングでゲート誤差とSPAMを分離する**

  * $F(m) = Ap^m + B$、$\mathrm{EPC} = (1-p)/2$。設定した $10^{-4}$ から $2\times10^{-2}$ の誤差が比1.0000で返ってきます。
  * SPAMの設定を変えると $m=1$ での生の生存確率は14パーセントポイント動き、$A$ は0.499から0.337へ動きますが、フィットされたEPCはどの場合も $1.000\times10^{-3}$ です。
  * 任意の1量子ビットチャネルのCliffordツワリングは厳密に脱分極的です — Pauli転送行列の非対角成分が $10^{-17}$ まで消えることで検証済み — これが手法の依拠する定理です。
  * RBは誤差のコヒーレンスに盲目で、リークに盲目で、特定のゲートではなくClifford平均を報告します。また $A$ と $B$ を確率に制約しない限り、$p \to 1$ でフィットは退化します。

**実務上の含意**

  * リークの数値を引用する前に $\lvert \alpha \rvert \tau$ を引用し、シミュレーションが保持した準位数を述べること。3準位と5準位で本章の20 nsのリークは3%動きます。
  * DRAG重みをリークに対して調整しないこと。誤差に敏感なシーケンスの差に対して調整し、得られたゲート誤差は別途確認すること。
  * すべての較正ルーチンを、既知の誤差を隠して検証すること。そして1回は無限ショット予算で走らせること。残差が縮まないなら、ショットを増やしても直りません。
  * $T_2^{\ast}$ はフィットに使った包絡模型と併記して報告すること。ガウシアンか指数かの選択が、演習3では推定される静的ばらつきを真値200 kHzに対して205 kHzから250 kHzへ動かしました。
  * ゲートあたり誤差の数値を読むときは、それがClifford平均かインターリーブ測定か、リークが別途測られているか、装置がいつ較正されたかを問うこと。

### 次章へ

パルス層はスタックの底であり、そこへ到達したことで、本コースがアルゴリズムから始めた下降は完結します。残るのは逆方向です。ゲートが本章で測ってきたような誤り率をもつとして、ソフトウェアは何ができるのか。[第5章](<chapter-5.html>)は近未来のハードウェアが実際に使う答え — 読み出し誤差の緩和、ゲート折り返しによるゼロノイズ外挿、確率的誤差キャンセル — を実装し、それぞれが何を買い、サンプル数で何を支払うかを実測し、それから誤り耐性という代替案が何を要求するかを述べるリソース見積りパイプラインを構築します。その比較の両半分に数値が付きます。誤り緩和についての誠実な言明は、それが今日たしかに本質的な役割を果たしており*かつ*指数的に高価だということであり、この文のどちらの半分も落としてはならないからです。

[← 第3章: トランスパイル — 接続性への写像](<chapter-3.html>) [第5章: 誤り緩和の実装層とリソース見積り →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章のトランズモンのパラメータ、パルス長、ドリフト率、読み出し誤り率、ショット予算は、較正とベンチマーキングの算術を追って検算できるように選んだ文献規模の代表値であり、装置の仕様ではありません。提案書や論文に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
