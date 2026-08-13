---
title: "第4章: 原子時計と原子干渉計"
chapter_title: "第4章: 原子時計と原子干渉計"
subtitle: 時間標準としてのRamsey、光パルス干渉計、そして極低温を必要としない磁力計
reading_time: 40-45分
difficulty: 上級
code_examples: 6
exercises: 5
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/quantum-sensing-introduction/chapter-4.html>) | Last sync: 2026-08-13

[材料科学基礎道場](<../index.html>) > [量子センシング入門](<index.html>) > 第4章

第2章と第3章は磁場を固体で測りました。ダイヤモンド中の欠陥と、超伝導体のリングです。本章が測る道具は自由原子であり、そうすることで第1章が開いた議論を閉じます。1.2節のRamsey系列は位相推定器として導入されました。時計とはその推定器をフィードバックループに組み込んだものであり、原子干渉計とは同じ推定器の2つの経路をエネルギーだけでなく空間的にも隔てたものです。新しいものは何も必要ありません。変わるのは、蓄積された位相のうちどの項が信号でどれが系統誤差なのかという割り当てだけであり、その付け替えこそが計量学の内容のすべてです。

原子時計に試料が挿入されることは決してないにもかかわらず、本章が材料研究者の時間に値する理由が3つあります。第1に、時計は誤差バジェットという規律が発明された場所であり、その規律はそのまま移植できます。**安定度** — 不確かさが平均化時間とともにどう下がるか — と**正確度** — 平均化では到達できない床 — の区別は、この主題全体で最も有用な考え方であり、第3章のSQUIDにも第2章のNV中心にもその区別があります。第2に、光パルス原子干渉計は重力計であり勾配計であり、したがって密度プローブです。触れることなく質量分布を測るのです。第3に、蒸気セル磁力計は150 °Cのガラスセルの中で、極低温をまったく使わずにフェムトテスラの感度に達します — そして帯域と物理的な大きさでその代価を払い、そのトレードオフは計算できるほど定量的です。

**単位と規約。** $T_1$、$T_2$、$T_2^\ast$、Ramsey系列とエコー系列、そして単位$/\sqrt{\mathrm{Hz}}$ で表す感度記法 $\eta$ は、[第1章](<chapter-1.html>) 1.3節から1.4節で固定したものと厳密に同じで、それ自身は姉妹コース[量子ハードウェア入門](<../../FM/quantum-hardware-introduction/chapter-1.html>)に従っています。$\omega$、$\delta$、$\Omega$ のような記号は rad/s の*角*周波数です。数値は $\nu$ または $\omega/2\pi$ と書く周波数（Hz）で示します。コード中の $2\pi$ はすべて明示します。分数周波数は $y = (\nu - \nu_0)/\nu_0$ で、平均化時間 $\tau$ における $y$ のAllan偏差を $\sigma_y(\tau)$ と書きます。磁場感度は T$/\sqrt{\mathrm{Hz}}$、加速度は m s$^{-2}/\sqrt{\mathrm{Hz}}$ で示します。

## 学習目標

本章を修了すると、以下のことができるようになります：

  * Ramsey時計の射影ノイズ限界での分数安定度 $\sigma_y(\tau) = 1/(\pi Q \sqrt{N})\sqrt{T_c/\tau}$ を導出し、光遷移が勝つのは原子について何かがあるからではなく $Q$ が5桁大きいからだと説明できる
  * 周波数の記録からAllan偏差を計算し、その傾きから白色・フリッカー・ランダムウォーク周波数ノイズを同定し、それ以上平均すると時計が悪くなる平均化時間を特定できる
  * 安定度と正確度を区別し、黒体放射・Zeeman・重力・光シフトの項をもつ系統シフトのバジェットを組み立て、どの項を最初に攻めるべきか述べられる
  * 光格子時計が魔法波長を必要とし、イオン時計がマイクロモーション制御を必要とする理由を、姉妹ハードウェアコースの双極子トラップとPaulトラップの物理を再利用して説明できる
  * 光パルス原子干渉計のMach-Zehnder位相 $k_\mathrm{eff} a T^2$、そのショットノイズ限界での加速度感度、それが要求する装置高さ、そして生じる縞の不定性を計算できる
  * 光ポンピングされた蒸気セルのスピンダイナミクスを積分し、自由誘導減衰からLarmor周波数と横緩和レートを取り出し、磁場勾配が $T_2^\ast$ の問題であると見抜ける
  * スピン交換緩和のない領域を設計上のトレードオフとして説明し、どれだけの磁場感度のためにどれだけの帯域と空間分解能を手放すのかを定量化できる
  * 蒸気セル、SQUID、NV中心を1枚の感度対寸法の地図に配置し、それらの間の選択を決める物理量を述べられる

* * *

## 4.1 時間標準としてのRamsey

### 位相推定器から周波数基準へ

時計とは発振器に補正を加えたものです。発振器は技術的なもの — 水晶振動子、マイクロ波シンセサイザ、共振器に固定したレーザー — であり、ドリフトします。原子はドリフトしません。原子の役割は、発振器がどれだけずれたかを教えることだけです。発振器を駆動源として1.2節のRamsey系列を走らせ、縞を読み、発振器を原子共鳴へ引き戻します。時計がどれだけ良いかを決めるものはすべて、縞が周波数誤差をどれだけ鋭く報告するかの中にあります。

標準的な $\pi/2$ - 待ち時間 $T$ - $\pi/2$ の系列を、駆動が共鳴から角周波数で $\delta$ だけ離調している状況で考えます。蓄積される位相は $\varphi = \delta T$ で、第1章の規約では励起状態確率は

$$ P(1) = \frac{1}{2}\left(1 - \cos \delta T\right) $$

です。これは*弁別器*です。その微分が周波数誤差を分布数の変化に変換します。微分が最大になるのは $P = 1/2$ となる $\delta T = \pi/2$ のところで、そこでは

$$ \left| \frac{\partial P}{\partial \delta} \right| = \frac{T}{2} $$

となります。したがって自由発展が長いほど弁別器は急峻になり、その関係は厳密に比例です。これが時計が長い $T$ を欲する理由のすべてであり、同時に $T$ が $T_2$ で上から抑えられる理由でもあります。コヒーレンス時間を超えると、縞には勾配をもつだけのコントラストが残っていないのです。

### 安定度の式

ここにノイズを加えます。$N$ 個の原子はそれぞれ独立に射影されるので、縞の半値バイアス点では $P$ の推定値の標準偏差は $1/(2\sqrt{N})$ となり、これを勾配で割ると1ショットの周波数不確かさが得られます。

$$ \sigma_\delta = \frac{1}{T\sqrt{N}} \qquad\Longrightarrow\qquad \sigma_\nu = \frac{1}{2\pi T \sqrt{N}} $$

分数で表し、さらに継続時間 $T_c$ の独立なサイクル $\tau/T_c$ 回にわたって平均すると

$$ \sigma_y(\tau) = \frac{1}{2\pi \nu_0 T \sqrt{N}} \sqrt{\frac{T_c}{\tau}} = \frac{1}{\pi Q \sqrt{N}}\sqrt{\frac{T_c}{\tau}}, \qquad Q \equiv \frac{\nu_0}{\Delta\nu} = 2\nu_0 T $$

となります。ここで $\Delta\nu = 1/2T$ はRamsey線幅です。覚えるべきは2番目の形で、3つのてこがきれいに分離されているからです。**Q値**は遷移周波数を線幅で割ったもの。**原子数 $N$** は平方根の下にしか現れません。**デューティ比 $T_c/T$**、すなわちサイクル間のデッドタイムは純粋な損失です。何が現れていないかにも注意してください。原子の*種類*については、$\nu_0$ と、そのコヒーレンスが許す $T$ を通してしか入ってきません。

これは[第1章](<chapter-1.html>) 1.3節で既に導いた磁場感度の結果の、周波数版の双子です。そこでは減衰包絡線 $\exp[-(t/T_2)^p]$ のもとでの最適観測時間が $\tau_\mathrm{opt} = T_2/(2p)^{1/p}$ であり、指数減衰の場合 $p = 1$ ではこれが $T_2/2$ になって、結果として得られる最良感度が $\eta_\mathrm{min} = \sqrt{2e}/(\gamma\sqrt{N T_2})$ でした。時計を支配するのも同じ構造です。自由発展 $T$ は縞のコントラストを失わずにコヒーレンス時間を超えることはできず、最適値は $T_2$ の一定の割合に位置し、そこに現れる係数 $\sqrt{e}$ は、感度をもつのに十分長く走りながらコントラストが残るくらいには短くとどまるための代価です。第5章5.2節はまさにこの評価指標を使ってエンタングルメントの値段を付けるので、$T_2/2$ と $\sqrt{2e}$ をいま頭に入れておく価値があります。

### Code Example 1: 弁別器と、それが許す安定度

```python
"""第4章 Code Example 1: 周波数弁別器としてのRamsey縞と、射影ノイズが許す分数安定度。"""
import numpy as np

TWO_PI = 2.0 * np.pi


def ramsey_prob(delta, T):
    """pi/2 - T - pi/2 のRamsey系列の後に |1> が得られる確率を返します。

    delta は角周波数での離調 omega - omega_0（rad/s）、T は自由発展時間（s）です。
    規約は第1章で固定したものと同じで、共鳴で P = 0、最急勾配は delta*T = pi/2
    に位置します。
    """
    return 0.5 * (1.0 - np.cos(delta * T))


def discriminator_slope(delta, T):
    """Ramsey縞の dP/d(delta) を、秒/ラジアンの単位で返します。"""
    return 0.5 * T * np.sin(delta * T)


T_free = 0.5                      # 自由発展時間、s
print("弁別器としてのRamsey縞（T = "
      f"{T_free:.1f} s、したがって縞の周期は {1.0 / T_free:.1f} Hz）")
print(f"{'detuning (Hz)':>15}{'delta*T (rad)':>15}{'P(1)':>10}"
      f"{'|dP/dnu| (1/Hz)':>18}")
print("-" * 58)
for nu_det in [0.0, 0.125, 0.25, 0.375, 0.5]:
    delta = TWO_PI * nu_det
    print(f"{nu_det:>15.3f}{delta * T_free:>15.4f}"
          f"{ramsey_prob(delta, T_free):>10.4f}"
          f"{abs(discriminator_slope(delta, T_free)) * TWO_PI:>18.4f}")

# --- 射影ノイズと、1ショットの周波数不確かさ ------------------------------
# 縞の半値バイアス点 P = 1/2 では原子1個あたりの分散が 1/4 なので、N 原子から
# 推定した P の不確かさは 1/(2 sqrt(N)) です。これを勾配 T/2 で割ると
# sigma_delta = 1/(T sqrt(N)) が得られます。
rng = np.random.default_rng(20260813)
N_atoms = 10_000
delta_bias = 0.5 * np.pi / T_free            # 縞の半値バイアス点
trials = 4000
counts = rng.binomial(N_atoms, ramsey_prob(delta_bias, T_free), size=trials)
p_hat = counts / N_atoms
# 縞を局所的に反転: delta_hat = delta_bias + (p_hat - 0.5)/(T/2)
delta_hat = delta_bias + (p_hat - 0.5) / (0.5 * T_free)
print(f"\nモンテカルロ検証（N = {N_atoms} 原子、{trials} 試行）")
print(f"  rms of estimated angular detuning : {delta_hat.std(ddof=1):.6f} rad/s")
print(f"  prediction 1/(T sqrt(N))          : "
      f"{1.0 / (T_free * np.sqrt(N_atoms)):.6f} rad/s")
print(f"  ratio                             : "
      f"{delta_hat.std(ddof=1) * T_free * np.sqrt(N_atoms):.4f}")


def stability(nu0, T, N, T_cycle, tau):
    """Ramsey時計の、射影ノイズ限界でのAllan偏差を返します。

    sigma_y(tau) = 1/(2 pi nu0 T sqrt(N)) * sqrt(T_cycle/tau) であり、これは
    Q = nu0/(1/2T) を使った見慣れた 1/(pi Q sqrt(N)) * sqrt(T_cycle/tau) と
    同じものです。
    """
    per_shot = 1.0 / (TWO_PI * nu0 * T * np.sqrt(N))
    return per_shot * np.sqrt(T_cycle / tau)


# 遷移周波数は原子定数ですが、T、N、T_cycle は説明のために丸めた運転パラメータ
# であって、どの装置の仕様でもありません。
clocks = [
    # ラベル,                     nu0 (Hz),   T (s), N,      T_cycle (s)
    ("Cs fountain, microwave",   9.192631770e9, 0.5, 1.0e6, 1.0),
    ("Rb vapour cell, microwave", 6.834682611e9, 0.005, 1.0e10, 0.01),
    ("Sr lattice, optical",      4.292280042e14, 1.0, 1.0e4, 2.0),
    ("Al+ single ion, optical",  1.121015393e15, 1.0, 1.0, 2.0),
]

print(f"\n{'clock':<26}{'nu0 (Hz)':>12}{'Q = 2 nu0 T':>13}{'N':>10}"
      f"{'sigma_y(1 s)':>14}{'tau to 1e-18':>16}")
print("-" * 91)
for label, nu0, T, N, Tc in clocks:
    Q = 2.0 * nu0 * T
    s1 = stability(nu0, T, N, Tc, 1.0)
    # Q を使った形を、直接の形と照合します
    s1_Q = 1.0 / (np.pi * Q * np.sqrt(N)) * np.sqrt(Tc / 1.0)
    assert abs(s1 / s1_Q - 1.0) < 1e-12
    tau_18 = Tc * (1.0 / (TWO_PI * nu0 * T * np.sqrt(N)) / 1e-18) ** 2
    print(f"{label:<26}{nu0:>12.4e}{Q:>13.3e}{N:>10.0e}{s1:>14.3e}"
          f"{tau_18:>13.3e} s")

print("\nSr光格子の行について、tau^(-1/2) 則:")
nu0, T, N, Tc = clocks[2][1:]
print(f"{'tau (s)':>12}{'sigma_y(tau)':>16}{'slope on log-log':>20}")
print("-" * 48)
taus = [1.0, 10.0, 100.0, 1000.0, 10000.0, 86400.0]
prev = None
for tau in taus:
    s = stability(nu0, T, N, Tc, tau)
    if prev is None:
        slope = "  (reference)"
    else:
        slope = f"{np.log(s / prev[1]) / np.log(tau / prev[0]):>20.4f}"
    print(f"{tau:>12.0f}{s:>16.4e}{slope:>20}")
    prev = (tau, s)
```

```text
弁別器としてのRamsey縞（T = 0.5 s、したがって縞の周期は 2.0 Hz）
  detuning (Hz)  delta*T (rad)      P(1)   |dP/dnu| (1/Hz)
----------------------------------------------------------
          0.000         0.0000    0.0000            0.0000
          0.125         0.3927    0.0381            0.6011
          0.250         0.7854    0.1464            1.1107
          0.375         1.1781    0.3087            1.4512
          0.500         1.5708    0.5000            1.5708

モンテカルロ検証（N = 10000 原子、4000 試行）
  rms of estimated angular detuning : 0.019956 rad/s
  prediction 1/(T sqrt(N))          : 0.020000 rad/s
  ratio                             : 0.9978

clock                         nu0 (Hz)  Q = 2 nu0 T         N  sigma_y(1 s)    tau to 1e-18
-------------------------------------------------------------------------------------------
Cs fountain, microwave      9.1926e+09    9.193e+09     1e+06     3.463e-14    1.199e+09 s
Rb vapour cell, microwave   6.8347e+09    6.835e+07     1e+10     4.657e-15    2.169e+07 s
Sr lattice, optical         4.2923e+14    8.585e+14     1e+04     5.244e-18    2.750e+01 s
Al+ single ion, optical     1.1210e+15    2.242e+15     1e+00     2.008e-16    4.031e+04 s

Sr光格子の行について、tau^(-1/2) 則:
     tau (s)    sigma_y(tau)    slope on log-log
------------------------------------------------
           1      5.2438e-18         (reference)
          10      1.6582e-18             -0.5000
         100      5.2438e-19             -0.5000
        1000      1.6582e-19             -0.5000
       10000      5.2438e-20             -0.5000
       86400      1.7840e-20             -0.5000
```

**着目点。** 弁別器の表は、縞の半分だけ離調した点が動作点であること、そして時計はわざと共鳴*から外して* — $\delta T = \pi/2$ で — 運転されることを述べています。そこが信号が周波数誤差にそもそも応答する場所だからです。共鳴上では微分が消え、原子は何も報告しません。続くモンテカルロのブロックは、$4\times10^7$ 回のシミュレートされた原子測定から射影ノイズの式を千分の2の精度で確認します。これは一度やっておく価値があり、そうすれば $1/(T\sqrt{N})$ は引用された結果であることをやめます。

物理が宿っているのは方式の表です。セシウム噴水とストロンチウム光格子時計は $Q$ で5桁、$N$ では*不利な*方向に2桁違うのに、光格子時計はそれでも $\sigma_y(1\,\mathrm{s})$ で4桁勝ちます。$5.244\times10^{-18}$ 対 $3.463\times10^{-14}$ です。理由はすべて分子の $\nu_0$ にあります。これが光時計を推す議論を一行にまとめたものであり、秒の定義が見直しの対象になっている理由でもあります — セシウムが悪い原子だからではなく、9.19 GHz が小さい数だからです。単一アルミニウムイオンの行は興味深いところで、$N = 1$ によって光格子に対して $\sqrt{N}$ で100倍を捨てておきながら、なお $2.008\times10^{-16}$ に達します。表の中で $Q$ が最大だからです。そして蒸気セルの行は注意書きです。$4.657\times10^{-15}$ は $10^{10}$ 原子に対する*射影ノイズの限界値*であって達成可能な安定度ではありません。この種の時計は射影ノイズを見るはるか前に光シフトと検出ノイズで律速されるからです。1つのノイズ源から計算した限界は予測ではありません。

最後のブロックは $\tau^{-1/2}$ 則を4桁で検証します。これは成り立たなければならないもので、起きているのは独立な推定値の平均化だけなのですから — だからこそ次の Example でそこから*外れる*ことが情報になるのです。

### Allan偏差、再訪

1.4節はAllan分散を周波数記録の2標本分散として導入しました。

$$ \sigma_y^2(\tau) = \frac{1}{2}\left\langle \left( \bar{y}_{k+1} - \bar{y}_k \right)^2 \right\rangle $$

ここで $\bar{y}_k$ は長さ $\tau$ の $k$ 番目の区間で平均した分数周波数です。その美点は、通常の分散が収束しないノイズ過程に対しても収束することで、ここでの用途は診断です。両対数プロット上の傾きがノイズ過程の名前を教えます。

| ノイズ過程 | $y$ のパワースペクトル密度 | Allan傾き | 時計における物理的起源 |
| --- | --- | --- | --- |
| 白色周波数 | $S_y \propto f^0$ | $\tau^{-1/2}$ | 射影ノイズ、検出のショットノイズ |
| フリッカー周波数 | $S_y \propto 1/f$ | $\tau^{0}$ | 1.4節と同じ $1/f$ の欠陥集団、共振器とレーザーのフリッカー |
| ランダムウォーク周波数 | $S_y \propto 1/f^2$ | $\tau^{+1/2}$ | 遅い環境ドリフト、温度 |
| 線形周波数ドリフト | 定常過程ではない | $\tau^{+1}$ | 部品の経年変化 |

フリッカーの行がこのシリーズの残りと繋がるところです。時計の $1/f$ ノイズは、[ハードウェアコース第2章](<../../FM/quantum-hardware-introduction/chapter-2.html>)でトランズモンを律速する $1/f$ ノイズや、本コース[第3章](<chapter-3.html>)でSQUIDを律速する $1/f$ 磁束ノイズと同じ微視的起源 — 対数一様に分布した遷移レートをもつ2準位揺動子の集団 — をもちます。まったく異なる3つの装置に、1つの材料問題です。

**どの推定量か。** 上の定義は2標本の非重複Allan分散であり、第1章の Code Example 4 が計算したのもこれです。下の Code Example 2 では代わりに**重複（overlapping）**推定量を使います。互いに素なブロックではなく、あらゆる開始位相について平均するものです。どちらも同じ $\sigma_y(\tau)$ を推定し期待値では一致しますが、重複版は長い $\tau$ で自由度が多いだけです。非重複の推定は長い $\tau$ で独立な差分がわずかしか残らず、ばらつきが大きくなります。時計の分野が重複版を使う理由がこれであり、2つの章が違う理由もこれです。傾きと領域の境界は変わりません。変わるのは誤差の幅だけです。

### Code Example 2: Allanの傾きと、助けにならなくなる平均化時間

```python
"""第4章 Code Example 2: 合成した時計ノイズのAllan偏差。
Code Example 1 の続き（同一セッション）。"""


def synth_power_law(n, alpha, rng, sigma=1.0):
    """片側PSDが 1/f^alpha に従う分数周波数の系列 y_k を生成します。

    alpha = 0 で白色周波数ノイズ、alpha = 1 でフリッカー周波数ノイズ、
    alpha = 2 でランダムウォーク周波数ノイズになります。呼び出し側の sigma を
    掛ける前に標準偏差1に規格化しているので、3種のノイズが同程度の重みで
    バジェットに入ります。
    """
    m = n // 2 + 1
    f = np.arange(m, dtype=float)
    f[0] = 1.0
    amp = f ** (-0.5 * alpha)
    amp[0] = 0.0                      # DC項は観測できないので落とします
    phase = rng.uniform(0.0, TWO_PI, m)
    spec = amp * np.exp(1j * phase)
    y = np.fft.irfft(spec, n)
    return sigma * y / y.std(ddof=1)


def overlapping_avar(y, tau0, m_list):
    """分数周波数の系列から、重なりありAllan分散を計算します。

    sigma_y^2(m tau0) = sum_j ( sum_{i=j}^{j+m-1} (y_{i+m} - y_i) )^2
                        / (2 m^2 (M - 2m + 1))
    """
    y = np.asarray(y, dtype=float)
    M = len(y)
    taus, avars = [], []
    for m in m_list:
        if M - 2 * m + 1 < 1:
            continue
        d = y[m:] - y[:-m]                       # 長さ M - m
        c = np.concatenate(([0.0], np.cumsum(d)))
        s = c[m:] - c[:-m]                       # m 項の移動和
        s = s[:M - 2 * m + 1]
        avars.append(np.sum(s ** 2) / (2.0 * m ** 2 * len(s)))
        taus.append(m * tau0)
    return np.array(taus), np.array(avars)


def loglog_slope(tau, adev, lo, hi):
    """[lo, hi] の範囲で log(adev) を log(tau) に対して最小二乗回帰した傾き。"""
    k = (tau >= lo) & (tau <= hi)
    return np.polyfit(np.log(tau[k]), np.log(adev[k]), 1)[0]


n_pts, tau0 = 2 ** 18, 1.0
rng2 = np.random.default_rng(4242)
m_list = np.unique(np.round(np.logspace(0, 4.4, 30)).astype(int))

print(f"1秒サンプル {n_pts} 点の、重なりありAllan偏差")
print(f"{'noise type':<28}{'PSD':<12}{'slope fit':>12}{'expected':>11}")
print("-" * 63)
series = {}
for label, alpha, expect in [("white frequency", 0.0, -0.5),
                             ("flicker frequency", 1.0, 0.0),
                             ("random-walk frequency", 2.0, +0.5)]:
    y = synth_power_law(n_pts, alpha, rng2, sigma=1e-13)
    series[label] = y
    tau, av = overlapping_avar(y, tau0, m_list)
    s = loglog_slope(tau, np.sqrt(av), 4.0, 3000.0)
    print(f"{label:<28}{'1/f^' + str(int(alpha)):<12}{s:>12.4f}{expect:>11.2f}")

# 純粋な線形ドリフトはそもそもノイズ過程ではなく、そのAllan傾きは +1 です。
t_axis = np.arange(n_pts) * tau0
y_drift = 2e-19 * t_axis
tau_d, av_d = overlapping_avar(y_drift, tau0, m_list)
print(f"{'linear frequency drift':<28}{'-':<12}"
      f"{loglog_slope(tau_d, np.sqrt(av_d), 4.0, 3000.0):>12.4f}{1.0:>11.2f}")

# --- 白色周波数ノイズの法則を、sqrt(tau0/tau) の予測と照合します ---------
y_w = series["white frequency"]
tau_w, av_w = overlapping_avar(y_w, tau0, m_list)
ad_w = np.sqrt(av_w)
print("\n白色周波数ノイズ: 測定値と sigma_y(1 s) sqrt(1 s / tau) の比較")
print(f"{'tau (s)':>10}{'measured':>14}{'predicted':>14}{'ratio':>9}")
print("-" * 47)
ref = ad_w[0]
for target in [1, 10, 100, 1000, 10000]:
    k = int(np.argmin(np.abs(tau_w - target)))
    pred = ref * np.sqrt(tau_w[0] / tau_w[k])
    print(f"{tau_w[k]:>10.0f}{ad_w[k]:>14.4e}{pred:>14.4e}"
          f"{ad_w[k] / pred:>9.4f}")

# --- 現実的なバジェット: 白色ノイズ、フリッカー床、ドリフト ---------------
parts = {
    "white": series["white frequency"],                     # sigma は 1e-13
    "flicker": 0.08 * series["flicker frequency"],          # sigma は 8e-15
    "drift": y_drift,                                       # 毎秒 2e-19
}
y_tot = sum(parts.values())
tau_t, av_t = overlapping_avar(y_tot, tau0, m_list)
ad_t = np.sqrt(av_t)
ad_parts = {}
for label, y_i in parts.items():
    _, av_i = overlapping_avar(y_i, tau0, m_list)
    ad_parts[label] = np.sqrt(av_i)
k_min = int(np.argmin(ad_t))
print("\n合成バジェット: どの項が支配的かは主張ではなく計算で決めます")
head = (f"{'tau (s)':>9}{'white':>12}{'flicker':>12}{'drift':>12}"
        f"{'quadrature':>13}{'combined':>12}{'dominant':>10}")
print(head)
print("-" * len(head))
for k in range(0, len(tau_t), 3):
    vals = {L: ad_parts[L][k] for L in parts}
    quad = np.sqrt(sum(v ** 2 for v in vals.values()))
    dom = max(vals, key=lambda L: vals[L])
    print(f"{tau_t[k]:>9.0f}{vals['white']:>12.3e}{vals['flicker']:>12.3e}"
          f"{vals['drift']:>12.3e}{quad:>13.3e}{ad_t[k]:>12.3e}{dom:>10}")
print(f"\n  最良安定度 {ad_t[k_min]:.4e}、tau = {tau_t[k_min]:.0f} s のとき。")
print("  これより長く平均すると、この時計は良くなるどころか悪くなります。")
ratio = ad_t / np.sqrt(sum(ad_parts[L] ** 2 for L in parts))
print(f"  合成 / 二乗和平方根、全 tau で: 最小 {ratio.min():.4f}、"
      f"最大 {ratio.max():.4f}")
print("  -- 3つの項は、そうあるべきように二乗和で加わっています。")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for label in series:
    tau_i, av_i = overlapping_avar(series[label], tau0, m_list)
    ax.loglog(tau_i, np.sqrt(av_i), marker="o", ms=3, label=label)
ax.loglog(tau_t, ad_t, "k-", lw=1.6, label="combined budget")
ax.set_xlabel("averaging time tau (s)")
ax.set_ylabel("Allan deviation sigma_y(tau)")
ax.set_title("Three noise types, three slopes")   # 3種のノイズ、3つの傾き
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
```

```text
1秒サンプル 262144 点の、重なりありAllan偏差
noise type                  PSD            slope fit   expected
---------------------------------------------------------------
white frequency             1/f^0            -0.4995      -0.50
flicker frequency           1/f^1            -0.0016       0.00
random-walk frequency       1/f^2             0.4995       0.50
linear frequency drift      -                 1.0000       1.00

白色周波数ノイズ: 測定値と sigma_y(1 s) sqrt(1 s / tau) の比較
   tau (s)      measured     predicted    ratio
-----------------------------------------------
         1    1.0000e-13    1.0000e-13   1.0000
         8    3.5355e-14    3.5355e-14   1.0000
        94    1.0312e-14    1.0314e-14   0.9997
      1083    3.0421e-15    3.0387e-15   1.0011
      8807    1.0472e-15    1.0656e-15   0.9828

合成バジェット: どの項が支配的かは主張ではなく計算で決めます
  tau (s)       white     flicker       drift   quadrature    combined  dominant
--------------------------------------------------------------------------------
        1   1.000e-13   2.921e-15   1.414e-19    1.000e-13   1.000e-13     white
        4   5.000e-14   2.718e-15   5.657e-19    5.007e-14   5.007e-14     white
       12   2.887e-14   2.685e-15   1.697e-18    2.899e-14   2.899e-14     white
       33   1.741e-14   2.680e-15   4.667e-18    1.761e-14   1.760e-14     white
       94   1.031e-14   2.680e-15   1.329e-17    1.065e-14   1.062e-14     white
      268   6.112e-15   2.682e-15   3.790e-17    6.674e-15   6.629e-15     white
      763   3.623e-15   2.667e-15   1.079e-16    4.500e-15   4.549e-15     white
     2177   2.157e-15   2.683e-15   3.079e-16    3.456e-15   3.419e-15   flicker
     6210   1.287e-15   2.730e-15   8.782e-16    3.143e-15   3.253e-15   flicker
    17712   6.969e-16   2.585e-15   2.505e-15    3.666e-15   3.654e-15   flicker

  最良安定度 3.2428e-15、tau = 4379 s のとき。
  これより長く平均すると、この時計は良くなるどころか悪くなります。
  合成 / 二乗和平方根、全 tau で: 最小 0.8711、最大 1.0412
  -- 3つの項は、そうあるべきように二乗和で加わっています。
```

**着目点。** 4つの傾きは、予測される $-1/2$、$0$、$+1/2$、$+1$ に対して $-0.4995$、$-0.0016$、$+0.4995$、$+1.0000$ と出ます。これが診断の実行です。測定されたAllan偏差だけが与えられたとき、傾きが支配的なノイズ過程を同定し、したがってその原因をどこに探すべきかを教えます。$-1/2$ の傾きは、時計が原子の許す限りうまくやっており、原子か時間がもっと必要だと言っています。$0$ の傾きは、測定の時間尺度で何かが揺らいでおり、いくら平均しても助けにならないと言っています。$+1$ の傾きは、部品が老化していると言っています。

白色ノイズの検証は最後の行と一緒に読む価値があります。$\tau = 8807$ s では測定値が $\tau^{-1/2}$ の予測より $1.7\%$ 低く出ますが、これは法則が破れたからではなく、$2^{18}$ 秒の記録にはその $\tau$ で独立な差が30個ほどしか含まれていないからです。どのプロットでもAllan偏差の右端は雑な推定値であり、そこの誤差棒は飾りではありません。

合成バジェットは実在するあらゆる時計がもつ形です。数百秒より下では白色項が支配して平均化が報われ、およそ2000秒から2万秒のあいだはフリッカー床が引き継いで平均化が報われなくなり、それより長くなるとドリフト項が増えて平均化は積極的に害になります。この合成時計の最良安定度は $\tau = 4379$ s での $3.2428\times10^{-15}$ で、1日走らせるのは1時間走らせるより悪いのです。最後の検証 — 3つの成分が全範囲にわたって約13%以内で二乗和として加わること、比は0.8711から1.0412まで — は、そもそもバジェットを独立な項の和として扱うことを許可するものです。13%の不足は独立性の破れではなく、1つの合成実現における有限標本のばらつきであり、2つの項が交差してどちらも支配しない場所で最悪になります。実現のアンサンブルを取れば縮まる量であり、有限長の記録1本から期待される不一致の大きさです。

この Example から意図的に外した物理効果が1つあります。**Dick効果**です。パルス動作の時計はデッドタイムのあいだ目を閉じているため、局所発振器のノイズが測定帯域へ折り返されて入り込み、デューティ比 $50\%$ の時計は原子ではなくレーザーで律速されることがあります。ここでは概念としてのみ登場させます。モデル化にはパルス系列の感度関数、すなわち1.4節のフィルタ関数の仕組みを単一系列ではなく観測サイクルに適用したものが必要になるからです。しかし実際上の帰結は覚えやすいものです。最良の光時計では原子ではなく局所発振器が短期安定度を決めており、その対策はより良い共振器 — すなわち鏡のコーティング損失と熱雑音についての材料問題 — です。

* * *

## 4.2 光格子時計とイオン時計

### 原子のエネルギー準位を動かさずに原子を保持する

光時計は原子を1秒間静止させ、しかも保持しているあいだ摂動しないことを要求します。この2つの要求は衝突し、2種類の光時計はその衝突を別々に解決します。

**光格子時計**は数千個の中性原子を定在波の強度極大に閉じ込めます。物理は[ハードウェアコース第4章](<../../FM/quantum-hardware-introduction/chapter-4.html>) 4.2節の双極子トラップです。大きく離調したレーザーが原子準位を $-\frac{1}{2}\alpha(\omega_L) \langle E^2\rangle$ だけシフトさせ、そのシフトの勾配が保存力になります。困難はすぐに現れます。トラップ光が時計の準位をシフトさせるので、原子を保持しているそのものが測りたい周波数を動かすのです。逃げ道は、2つの時計準位が異なる分極率 $\alpha_g(\omega_L)$ と $\alpha_e(\omega_L)$ をもち、その曲線が交差することです。$\alpha_g = \alpha_e$ となる**魔法波長**ではトラップが両準位を等しくシフトさせ、遷移周波数は1次では影響を受けません。残留シフトはその交点からの離調とトラップ深さに比例するので、格子レーザーの波長安定化が時計の仕様になります。

**イオン時計**は単一の荷電原子を[ハードウェアコース第3章](<../../FM/quantum-hardware-introduction/chapter-3.html>) 3.1節の高周波Paulトラップに保持します。トラップ光がまったくないので光シフトもありません。その代わりイオンは高周波電場の節に座り、その節から押し出す迷走静電場があれば**余剰マイクロモーション**、すなわち高周波駆動の振動が生じます。マイクロモーションは時計に2つのことをします。遷移を変調して線にサイドバンドを付け、そしてイオンに平均二乗速度を与えて2次Doppler（時間の遅れ）シフト $-\langle v^2\rangle/2c^2$ を生みます。これはハードウェアコースの安定図と同じMathieu方程式の物理を、トラップの帰結ではなく相対論的な帰結として読んだものです。

トレードオフは明快です。光格子時計は $N \sim 10^4$ をもち、したがって射影ノイズで100倍を得ており、設計で消さねばならない光シフトを代価として払います。イオン時計は $N = 1$ で安定度に100倍を支払い、はるかに単純な系統バジェットを買います。Example 1 の表には両方が現れており、両方が異なる経路で同程度の正確度に達しています。

### 数え上げなければならないシフト

以下の各項は既知の関数形をもつ周波数シフトであり、バジェットはそれぞれの大きさではなく*不確かさ*を抑えることから成ります。$10^4$ 分の1で分かっている大きなシフトは無害です。半分が分かっていない小さなシフトは無害ではありません。

| シフト | スケーリング | 制御しなければならないもの |
| --- | --- | --- |
| 黒体放射 | $\propto \Delta\alpha\, T^4$ | 原子から見えるすべての面の温度 |
| 2次Zeeman | $\propto B^2$ | バイアス磁場の大きさ、交流成分を含む |
| 重力赤方偏移 | $\propto g h/c^2$ | 基準ジオイドからの原子の高さ |
| 格子光シフト | $\propto (\lambda - \lambda_\mathrm{magic})\times$ 深さ | 格子の波長と強度 |
| 低温衝突 | $\propto$ 原子密度 | 原子が何個で、どう分布しているか |
| 2次Doppler | $\propto -\langle v^2\rangle/2c^2$ | マイクロモーションと残留する熱運動 |

最初の行の $T^4$ は表の中で最も強いてこであり、時計の実験室に極低温の囲いが現れる理由です。温度誤差に対する感度は $T^3$ でスケールするので、周囲を冷やせばシフトに4次で、不確かさに3次で効きます。これは物理の問題に対する材料工学の答えです。

### Code Example 3: 系統バジェットと、それが定める床

```python
"""第4章 Code Example 3: 系統シフトのバジェットと、正確度と安定度が別の数である理由。
Code Example 1-2 の続き（同一セッション）。"""

c_light = 2.99792458e8            # m/s
g_earth = 9.80665                 # m/s^2

# 以下の係数はすべて、スケーリングを見せるために選んだ桁のオーダーの代用値であり、
# 特定の装置について測定された値ではありません。ここで示しているのはバジェットの
# 「構造」であって、その中身ではありません。

nu_optical = 4.292280042e14       # Sr時計遷移、Hz（原子定数）


def bbr_shift(T_kelvin, frac_at_300K=-5.5e-15):
    """黒体放射シフト。示差分極率により T^4 でスケールします。"""
    return frac_at_300K * (T_kelvin / 300.0) ** 4


def bbr_uncertainty(T_kelvin, dT, frac_at_300K=-5.5e-15):
    """囲いの温度不確かさ dT から生じる、黒体放射シフトの不確かさ。"""
    return abs(4.0 * bbr_shift(T_kelvin, frac_at_300K) * dT / T_kelvin)


print("黒体放射: T^4 というてこ")
print(f"{'enclosure T (K)':>17}{'shift':>13}{'unc. for dT = 1 K':>21}")
print("-" * 51)
for T_env in [300.0, 200.0, 100.0, 77.0]:
    print(f"{T_env:>17.0f}{bbr_shift(T_env):>13.2e}"
          f"{bbr_uncertainty(T_env, 1.0):>21.2e}")

beta_zeeman = 2.33e7              # Hz/T^2、2次Zeeman係数
print("\n2次Zeeman効果: B^2 というてこ")
print(f"{'bias field (T)':>16}{'shift (Hz)':>14}{'fractional':>13}"
      f"{'unc. at 1% of B':>18}")
print("-" * 61)
for B in [1e-3, 1e-4, 1e-5]:
    shift = beta_zeeman * B ** 2
    frac = shift / nu_optical
    print(f"{B:>16.0e}{shift:>14.4e}{frac:>13.2e}{2.0 * 0.01 * frac:>18.2e}")

print("\n重力赤方偏移: 高さ1メートルあたり g h / c^2")
print(f"  fractional shift per metre : {g_earth / c_light ** 2:.3e}")
for dh in [1.0, 0.01, 0.001]:
    print(f"  uncertainty for dh = {dh * 1e3:7.1f} mm : "
          f"{g_earth * dh / c_light ** 2:.3e}")

# イオントラップの余剰マイクロモーションによる2次Doppler効果。v_rf の背後にある
# Mathieu方程式の物理は量子ハードウェア入門の第3章で導いたものと同じで、ここで
# 必要なのは相対論的な帰結だけです。
print("\nイオンのマイクロモーションによる2次Doppler効果: -<v^2>/(2 c^2)")
print(f"{'rf micromotion amplitude v (m/s)':>34}{'fractional shift':>19}")
print("-" * 53)
for v_rf in [1.0, 0.3, 0.1, 0.03]:
    print(f"{v_rf:>34.2f}{-0.5 * v_rf ** 2 / (2 * c_light ** 2):>19.2e}")

# --- バジェット本体 --------------------------------------------------------
budget = [
    ("blackbody radiation, 300 K enclosure, dT = 1 K",
     bbr_shift(300.0), bbr_uncertainty(300.0, 1.0)),
    ("second-order Zeeman, B = 0.1 mT known to 1%",
     beta_zeeman * 1e-4 ** 2 / nu_optical,
     2.0 * 0.01 * beta_zeeman * 1e-4 ** 2 / nu_optical),
    ("gravitational redshift, height known to 10 mm",
     0.0, g_earth * 0.010 / c_light ** 2),
    ("lattice light shift, residual after magic-wavelength tuning",
     0.0, 2.0e-17),
    ("cold-collision density shift", -1.0e-17, 5.0e-18),
    ("residual second-order Doppler, v = 0.1 m/s", -5.6e-19, 5.6e-19),
]
print(f"\n{'contribution':<60}{'shift':>12}{'uncertainty':>14}")
print("-" * 86)
tot_sq = 0.0
for name, shift, unc in budget:
    tot_sq += unc ** 2
    print(f"{name:<60}{shift:>12.1e}{unc:>14.1e}")
u_tot = np.sqrt(tot_sq)
print("-" * 86)
print(f"{'total, added in quadrature':<60}{'':>12}{u_tot:>14.1e}")

# Example 1 の統計的不確かさは、いつその床に到達するでしょうか。
print(f"\nExample 1 の射影ノイズが {u_tot:.1e} の床に達するまでの時間:")
print(f"{'clock':<26}{'sigma_y(1 s)':>14}{'tau to reach floor':>21}")
print("-" * 61)
for label, nu0_i, T_i, N_i, Tc_i in clocks:
    per_shot = 1.0 / (TWO_PI * nu0_i * T_i * np.sqrt(N_i))
    tau_floor = Tc_i * (per_shot / u_tot) ** 2
    print(f"{label:<26}{stability(nu0_i, T_i, N_i, Tc_i, 1.0):>14.3e}"
          f"{tau_floor:>18.3e} s")
print("  この時間を超えると、平均化が買うのは精度であって正確度ではありません。")
print("  系統バジェットは tau で平均されて小さくならないのです。Sr の行はこの床に")
print("  ミリ秒で、単一イオンは数秒で到達します。だからこそ光時計の仕事は、まず")
print("  系統誤差低減のプログラムであり、安定度のプログラムは二の次なのです。")
print(f"\n囲いを300 Kではなく100 Kにすると、黒体放射の項は "
      f"{bbr_uncertainty(100.0, 1.0):.1e} まで下がり、")
u_cold = np.sqrt(tot_sq - bbr_uncertainty(300.0, 1.0) ** 2
                 + bbr_uncertainty(100.0, 1.0) ** 2)
print(f"  合計は {u_cold:.1e} になります -- クライオスタット1台で "
      f"{u_tot / u_cold:.1f} 倍です。いまや格子光シフトが")
print("  単独で最大の項なので、次の努力はそこへ向かいます。")
```

```text
黒体放射: T^4 というてこ
  enclosure T (K)        shift    unc. for dT = 1 K
---------------------------------------------------
              300    -5.50e-15             7.33e-17
              200    -1.09e-15             2.17e-17
              100    -6.79e-17             2.72e-18
               77    -2.39e-17             1.24e-18

2次Zeeman効果: B^2 というてこ
  bias field (T)    shift (Hz)   fractional   unc. at 1% of B
-------------------------------------------------------------
           1e-03    2.3300e+01     5.43e-14          1.09e-15
           1e-04    2.3300e-01     5.43e-16          1.09e-17
           1e-05    2.3300e-03     5.43e-18          1.09e-19

重力赤方偏移: 高さ1メートルあたり g h / c^2
  fractional shift per metre : 1.091e-16
  uncertainty for dh =  1000.0 mm : 1.091e-16
  uncertainty for dh =    10.0 mm : 1.091e-18
  uncertainty for dh =     1.0 mm : 1.091e-19

イオンのマイクロモーションによる2次Doppler効果: -<v^2>/(2 c^2)
  rf micromotion amplitude v (m/s)   fractional shift
-----------------------------------------------------
                              1.00          -2.78e-18
                              0.30          -2.50e-19
                              0.10          -2.78e-20
                              0.03          -2.50e-21

contribution                                                       shift   uncertainty
--------------------------------------------------------------------------------------
blackbody radiation, 300 K enclosure, dT = 1 K                  -5.5e-15       7.3e-17
second-order Zeeman, B = 0.1 mT known to 1%                      5.4e-16       1.1e-17
gravitational redshift, height known to 10 mm                    0.0e+00       1.1e-18
lattice light shift, residual after magic-wavelength tuning      0.0e+00       2.0e-17
cold-collision density shift                                    -1.0e-17       5.0e-18
residual second-order Doppler, v = 0.1 m/s                      -5.6e-19       5.6e-19
--------------------------------------------------------------------------------------
total, added in quadrature                                                     7.7e-17

Example 1 の射影ノイズが 7.7e-17 の床に達するまでの時間:
clock                       sigma_y(1 s)   tau to reach floor
-------------------------------------------------------------
Cs fountain, microwave         3.463e-14         2.025e+05 s
Rb vapour cell, microwave      4.657e-15         3.663e+03 s
Sr lattice, optical            5.244e-18         4.643e-03 s
Al+ single ion, optical        2.008e-16         6.807e+00 s
  この時間を超えると、平均化が買うのは精度であって正確度ではありません。
  系統バジェットは tau で平均されて小さくならないのです。Sr の行はこの床に
  ミリ秒で、単一イオンは数秒で到達します。だからこそ光時計の仕事は、まず
  系統誤差低減のプログラムであり、安定度のプログラムは二の次なのです。

囲いを300 Kではなく100 Kにすると、黒体放射の項は 2.7e-18 まで下がり、
  合計は 2.3e-17 になります -- クライオスタット1台で 3.3 倍です。いまや格子光シフトが
  単独で最大の項なので、次の努力はそこへ向かいます。
```

**着目点。** まず4つのてこを読んでください。室温での黒体放射は $-5.5\times10^{-15}$ のシフトで、誰かが気にする水準より4桁上にあり、囲いの温度の1 Kの不確かさがそのうち $7.33\times10^{-17}$ を未知のまま残します。囲いを100 Kに冷やすとシフトは3分の1になるのではなく $3^4 = 81$ 分の1になり、不確かさは27分の1になって $2.72\times10^{-18}$ です。2次Zeeman項は4乗の代わりに2乗で同じ構造を示します。$0.1$ mT ではシフトは $0.233$ Hz で、磁場を $1\%$ で知っていれば $1.09\times10^{-17}$ が残ります。重力赤方偏移は調整できる物理が一切ない項です。高さ1メートルあたり $1.091\times10^{-16}$ なので、時計の周波数が意味をもつには、その*標高*をセンチメートルで測量しておかなければなりません。同じ建物の別の階にある2台の時計は一致せず、一般相対性理論はそれで良いと言っています。

バジェット本体は二乗和で $7.7\times10^{-17}$ となり、その下の表がこの節の要点です。Example 1 のSr光格子の行はその統計的不確かさに $4.6$ ミリ秒で到達します。その4ミリ秒より後はすべて、系統バジェットが引き受けてくれない精度を買っているのです。これが安定度と正確度の区別の運用上の内容であり、光時計の文献が原子についてではなく黒体放射の囲い、格子波長のサーボ、測地測量についての文献である理由です。最後のブロックは設計上の帰結を明示します。囲いを冷やすと合計は $3.3$ 倍改善し、そのあとは格子光シフトが単独で最大の項になるので、次の努力はそこへ向かいます。誤差バジェットとは、大きさで並べ替えたタスクリストなのです。

同じ論法を第3章はSQUIDに、第2章はNV中心に適用しており、それぞれの場合で最上位の項が違うだけです。それが本章で移植できる技能です。

* * *

## 4.3 光パルス原子干渉計

### レーザーパルスがビームスプリッタになる

ここまではRamsey干渉計の2つの経路をエネルギーだけで隔てていました。今度は空間で隔てます。2光子RamanまたはBraggパルスは、内部遷移を駆動するのと同時に運動量 $\hbar k_\mathrm{eff}$ を移行させます。対向するビームでは $k_\mathrm{eff} = k_1 + k_2 \approx 2k$ です。2つの内部状態の重ね合わせにある原子は、そのとき2つの運動量の重ね合わせにもあり、飛行しながら2つの成分は物理的に離れていきます。$\pi/2$ パルスがビームスプリッタ、$\pi$ パルスが鏡であり、系列 $\pi/2$ - $T$ - $\pi$ - $T$ - $\pi/2$ は原子でできた**Mach-Zehnder干渉計**です。

$k_\mathrm{eff}$ 方向の一様加速度 $a$ に対する2経路の位相差は

$$ \Delta\varphi = k_\mathrm{eff}\, a\, T^2 $$

です。この $T^2$ がこの装置が存在する理由です。これは時計の $T$ のようなコヒーレンス時間の因子ではありません。変位が時間の2乗で増えるという古典的な言明であり、自由発展の間隔を2倍にすれば4倍の価値があるという意味です。問題中の他のてこはどれも価値が低くなります。原子数は $\sqrt{N}$ で入り、$k_\mathrm{eff}$ は光子対を追加することでしか倍にできず、しかも忠実度の代価を伴います。

回転も同じ形で入ります。回転系では加速度 $2\boldsymbol{\Omega}\times\mathbf{v}$ が現れるからです。原子が横方向に $v$ で動くMach-Zehnder干渉計は、したがって同じ位相・加速度変換で回転を測り、同じ装置がジャイロスコープになります。鉛直方向に隔てた2台の干渉計は $g$ の*差*を測る勾配計になり、これは他では支配的になるノイズ、すなわち共通モードのプラットフォーム振動を除去します。

### Code Example 4: $T$ に対するMach-Zehnder感度

```python
"""第4章 Code Example 4: 光パルスMach-Zehnder原子干渉計。
Code Example 1-3 の続き（同一セッション）。"""

hbar = 1.054571817e-34
u_mass = 1.66053906660e-27
m_Rb = 86.909180527 * u_mass       # Rb-87 の質量
lam_D2 = 780.241209e-9             # m
k_eff = 2.0 * (TWO_PI / lam_D2)    # 2光子Raman/Bragg遷移の運動量移行


def mz_phase(T, a=g_earth, keff=k_eff):
    """一様加速度 a に対するMach-Zehnder位相の主要項: keff a T^2。"""
    return keff * a * T ** 2


def accel_sensitivity(T, N, keff=k_eff):
    """射影ノイズ限界での、1ショットの加速度不確かさ（m/s^2）。

    縞の半値バイアス点では位相の不確かさは 1/sqrt(N) で、Example 1 のRamseyの
    場合とまったく同じです。位相から加速度への変換係数は keff T^2 です。
    """
    return 1.0 / (np.sqrt(N) * keff * T ** 2)


print(f"Rb-87 の2光子ビームスプリッタ: k_eff = {k_eff:.4e} 1/m、")
print(f"  反跳速度 hbar k_eff / m = "
      f"{hbar * k_eff / m_Rb * 1e3:.3f} mm/s")

N_shot = 1.0e6
print(f"\nMach-Zehnder重力計、1ショットあたり N = {N_shot:.0e} 原子")
head = (f"{'T (ms)':>8}{'phase (rad)':>14}{'fringes':>11}{'apex h (m)':>12}"
        f"{'sep. (um)':>11}{'d_a (m/s^2)':>14}{'d_g/g':>11}")
print(head)
print("-" * len(head))
for T_ms in [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    T = T_ms * 1e-3
    phi = mz_phase(T)
    apex = 0.5 * g_earth * T ** 2                 # 射出点から見た噴水の頂点
    sep = hbar * k_eff / m_Rb * T                 # 頂点での2経路の間隔
    da = accel_sensitivity(T, N_shot)
    print(f"{T_ms:>8.0f}{phi:>14.4e}{phi / TWO_PI:>11.3e}{apex:>12.4f}"
          f"{sep * 1e6:>11.2f}{da:>14.3e}{da / g_earth:>11.2e}")

print("\nT^2 というてこを、傾きとして述べます:")
Ts = np.array([1e-3, 1e-2, 1e-1, 1.0])
das = accel_sensitivity(Ts, N_shot)
slope = np.polyfit(np.log(Ts), np.log(das), 1)[0]
print(f"  d log(d_a) / d log(T) = {slope:.4f}   （厳密に -2）")
print("  T を2倍にすると4倍の価値、N を2倍にしても1.41倍の価値しかありません。")

# --- 同じ装置が g 以外に測れるもの ----------------------------------------
Omega_earth = 7.2921150e-5         # rad/s
T_ref, N_ref = 0.1, N_shot
da_ref = accel_sensitivity(T_ref, N_ref)
print(f"\n1台の装置（T = {T_ref * 1e3:.0f} ms、N = {N_ref:.0e}）、"
      f"1ショットの d_a = {da_ref:.3e} m/s^2:")
print(f"{'quantity':<34}{'signal':>16}{'signal / d_a':>16}")
print("-" * 66)
rows = [
    ("g itself", g_earth),
    ("Coriolis, Omega_Earth x v, v = 1 m/s", 2.0 * Omega_earth * 1.0),
    ("gravity gradient over 1 m, 3e-6 /s^2", 3.0e-6 * 1.0),
    ("tidal variation of g, ~1e-6 m/s^2", 1.0e-6),
    ("1 tonne at 5 m, G M / r^2", 6.674e-11 * 1.0e3 / 25.0),
]
for name, sig in rows:
    print(f"{name:<34}{sig:>16.4e}{sig / da_ref:>16.3e}")

# --- 平均化: 第1章1.3節の eta 規約 ---------------------------------------
print("\n1.3節の eta = (単位)/sqrt(Hz) 規約による感度。")
print("サイクル時間 T_c あたり1ショットとします:")
print(f"{'T (ms)':>8}{'T_c (s)':>10}{'eta_a (m/s^2/rtHz)':>21}"
      f"{'time for 1 nano-g (s)':>23}")
print("-" * 62)
for T_ms in [10.0, 100.0, 300.0]:
    T = T_ms * 1e-3
    Tc = 2.0 * T + 0.5                            # デッドタイムが支配的
    eta_a = accel_sensitivity(T, N_shot) * np.sqrt(Tc)
    target = 1e-9 * g_earth
    print(f"{T_ms:>8.0f}{Tc:>10.2f}{eta_a:>21.3e}{(eta_a / target) ** 2:>23.2f}")

print("\nダイナミックレンジ: 位相は巻き戻しが必要です。")
for T_ms in [10.0, 100.0]:
    T = T_ms * 1e-3
    da_wrap = TWO_PI / (k_eff * T ** 2)
    print(f"  T = {T_ms:5.0f} ms: 縞1本が d_a = {da_wrap:.3e} m/s^2、"
          f"すなわち {da_wrap / g_earth:.2e} g")

fig, ax = plt.subplots(figsize=(6.2, 4.2))
T_grid = np.logspace(-3, 0, 60)
for N_i, style in [(1e4, "--"), (1e6, "-"), (1e8, ":")]:
    ax.loglog(T_grid * 1e3, accel_sensitivity(T_grid, N_i), style,
              label=f"N = {N_i:.0e}")
ax.set_xlabel("free-evolution time T (ms)")
ax.set_ylabel("single-shot acceleration uncertainty (m/s$^2$)")
ax.set_title("Mach-Zehnder sensitivity: $T^{-2}$ and $N^{-1/2}$")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
```

```text
Rb-87 の2光子ビームスプリッタ: k_eff = 1.6106e+07 1/m、
  反跳速度 hbar k_eff / m = 11.769 mm/s

Mach-Zehnder重力計、1ショットあたり N = 1e+06 原子
  T (ms)   phase (rad)    fringes  apex h (m)  sep. (um)   d_a (m/s^2)      d_g/g
---------------------------------------------------------------------------------
       1    1.5794e+02  2.514e+01      0.0000      11.77     6.209e-05   6.33e-06
      10    1.5794e+04  2.514e+03      0.0005     117.69     6.209e-07   6.33e-08
      30    1.4215e+05  2.262e+04      0.0044     353.07     6.899e-08   7.03e-09
     100    1.5794e+06  2.514e+05      0.0490    1176.91     6.209e-09   6.33e-10
     300    1.4215e+07  2.262e+06      0.4413    3530.72     6.899e-10   7.03e-11
    1000    1.5794e+08  2.514e+07      4.9033   11769.08     6.209e-11   6.33e-12

T^2 というてこを、傾きとして述べます:
  d log(d_a) / d log(T) = -2.0000   （厳密に -2）
  T を2倍にすると4倍の価値、N を2倍にしても1.41倍の価値しかありません。

1台の装置（T = 100 ms、N = 1e+06）、1ショットの d_a = 6.209e-09 m/s^2:
quantity                                    signal    signal / d_a
------------------------------------------------------------------
g itself                                9.8066e+00       1.579e+09
Coriolis, Omega_Earth x v, v = 1 m/s      1.4584e-04       2.349e+04
gravity gradient over 1 m, 3e-6 /s^2      3.0000e-06       4.832e+02
tidal variation of g, ~1e-6 m/s^2       1.0000e-06       1.611e+02
1 tonne at 5 m, G M / r^2               2.6696e-09       4.300e-01

1.3節の eta = (単位)/sqrt(Hz) 規約による感度。
サイクル時間 T_c あたり1ショットとします:
  T (ms)   T_c (s)   eta_a (m/s^2/rtHz)  time for 1 nano-g (s)
--------------------------------------------------------------
      10      0.52            4.477e-07                2084.49
     100      0.70            5.195e-09                   0.28
     300      1.10            7.236e-10                   0.01

ダイナミックレンジ: 位相は巻き戻しが必要です。
  T =    10 ms: 縞1本が d_a = 3.901e-03 m/s^2、すなわち 3.98e-04 g
  T =   100 ms: 縞1本が d_a = 3.901e-05 m/s^2、すなわち 3.98e-06 g
```

**着目点。** 最初の表にはこの分野の工学的問題のすべてが入っています。$T = 100$ ms では干渉計は $1.579\times10^{6}$ ラジアンの位相を蓄積し、2つの経路は $1.18$ mm 離れ、$10^{6}$ 原子から1ショットの加速度不確かさは $6.209\times10^{-9}$ m s$^{-2}$、すなわち $g$ の $6.33\times10^{-10}$ です。$T$ を1秒まで押せば感度は100倍良くなります — そして原子は2秒間自由落下していなければならず、これは射出点から $4.90$ m 上の噴水頂点と、塔のような装置を意味します。だから長基線の原子干渉計は使われなくなった立坑や落下塔に建てられるのです。メートル単位の高さは $gT^2/2$ であり、これを回避する方法はありません。測定された傾きは $\partial \log \delta a/\partial \log T = -2.0000$ を厳密に確認します。

信号の表はそのような装置が何を見られるかを述べています。地球の自転は、1 m/s で動く原子に対してCoriolis加速度として入り、1ショットのノイズの $2.3\times10^{4}$ 倍です。だから原子ジャイロスコープが動作し、同時に回転が重力計にとっては*系統誤差*であって鏡を傾けて消さなければならないのです。1メートルにわたる重力勾配はノイズの480倍。5メートル先の1トンの質量は $2.7\times10^{-9}$ m s$^{-2}$ を生み、これは1ショットのノイズの $0.43$ なので1ショットでは見えず、数百ショットでは容易です。これが地下の空洞や密度異常を探すのにこの種の装置を使う際の動作原理です。原子干渉計が材料計測の装置であるというのはこの意味です。接触もせず、組成についての仮定も置かず、質量分布を遠隔で測るのです。

$\eta$ のブロックは1.3節の感度規約に変換し、デューティ比が何を要求するかを示します。$T = 100$ ms ではサイクル時間は干渉計ではなく $0.5$ s の準備時間で支配されるので $\eta_a = 5.195\times10^{-9}$ m s$^{-2}/\sqrt{\mathrm{Hz}}$ となり、1ナノ $g$ は1秒の何分の1かで済みます。デッドタイムは $T$ を長くしたい2番目の理由です。準備時間を薄く引き延ばしてくれるからです。

最後のブロックは誠実な注意書きです。$T = 100$ ms では縞1本が $3.901\times10^{-5}$ m s$^{-2}$ に対応するので、それより粗くしか分かっていない加速度は整数本の縞だけ不定になります。感度とダイナミックレンジは、4.4節の蒸気セル磁力計や3.2節のSQUIDの磁束固定ループとまったく同じようにトレードオフの関係にあり、標準的な解決も3つとも同じです。まず粗くて不定性のない測定を行い、それから干渉測定を行うのです。

* * *

## 4.4 蒸気セル磁力計とSERF領域

### 冷却ではなく光から偏極を得る

SQUIDはフェムトテスラを測り、液体ヘリウムを必要とします。NV中心は室温で動作し、数十立方ナノメートルの体積でマイクロテスラを測ります。蒸気セル磁力計は第3の角を占めます。フェムトテスラの感度、極低温なし、そして立方ミリメートルの検出体積です。

冷凍機を必要としない理由は、その偏極が熱平衡から来ていないことです。数ナノテスラでのZeeman分裂は420 Kの $k_BT$ より12桁小さいので、熱的偏極は実用上まさにゼロです。代わりに**光ポンピング**が偏極を供給します。D線に共鳴した円偏光が原子を基底状態の1つの副準位から追い出し、原子はもう一方に蓄積して、温度に関わらずミリ秒でオーダー1の偏極に達します。これは散逸的な準備機構が状態準備を温度から切り離すというDiVincenzoの初期化条件についての議論と同じもので、ここではセンサ設計の原理として使われています。

そのダイナミクスは1.6節のBloch方程式に2つを加えたものです。スピンをビーム方向へ駆動する光ポンピング項と、固体のバスではなく衝突から来る緩和レートです。

$$ \frac{d\mathbf{S}}{dt} = \gamma\, \mathbf{S}\times\mathbf{B} \;-\; \Gamma_{2}\, \mathbf{S}_\perp \;-\; \Gamma_{1}\left(S_z - S_z^{0}\right)\hat{z} \;+\; R_\mathrm{op}\left(\tfrac{1}{2}\hat{s} - \mathbf{S}\right) $$

磁気回転比は電子のものを、核による減速係数 $q$（オーダー数）で割ったものです。電子スピンが核と結合しているからで、$\gamma_\mathrm{eff} = \gamma_e/q$ となります。それでも核の磁気回転比より4桁大きく、これが希ガスではなくアルカリ蒸気を使う理由です。

### Code Example 5: セル中のスピン歳差

```python
"""第4章 Code Example 5: 蒸気セル中のスピン歳差を数値積分します。
Code Example 1-4 の続き（同一セッション）。"""
from scipy.integrate import solve_ivp

gamma_e = 1.760859630e11           # rad/s/T、自由電子の磁気回転比
q_slow = 6.0                       # 核スピンによる減速係数、オーダー1
gamma_eff = gamma_e / q_slow       # アルカリ原子の実効磁気回転比


def spin_rhs(t, S, B_vec, Gamma1, Gamma2, R_op, s_hat):
    """光ポンピングされたアルカリ蒸気に対する現象論的Bloch方程式。

    S は集団のスピン偏極ベクトル（無次元）です。3つの項は、B のまわりのLarmor
    歳差、ゼロへ向かう異方的緩和、そしてビーム方向 s_hat へ向かうレート R_op の
    光ポンピングです。
    """
    Sx, Sy, Sz = S
    Bx, By, Bz = B_vec
    prec = gamma_eff * np.array([Sy * Bz - Sz * By,
                                 Sz * Bx - Sx * Bz,
                                 Sx * By - Sy * Bx])
    relax = np.array([Gamma2 * Sx, Gamma2 * Sy, Gamma1 * Sz])
    pump = R_op * (0.5 * np.array(s_hat) - S)
    return prec - relax + pump


def free_induction(B0, Gamma2, t_end, n_pts=20001, Gamma1=None, S0=None):
    """磁場 B0 z 中で、横向きスピンの自由歳差を積分します。"""
    if Gamma1 is None:
        Gamma1 = Gamma2
    if S0 is None:
        S0 = [0.5, 0.0, 0.0]
    t_grid = np.linspace(0.0, t_end, n_pts)
    sol = solve_ivp(spin_rhs, (0.0, t_end), S0, t_eval=t_grid,
                    args=((0.0, 0.0, B0), Gamma1, Gamma2, 0.0, (1.0, 0.0, 0.0)),
                    rtol=1e-10, atol=1e-13, method="DOP853")
    return t_grid, sol.y


def fit_fid(t, sx, sy):
    """自由誘導減衰から、Larmor周波数と横緩和レートを取り出します。"""
    env = np.hypot(sx, sy)
    k = env > 1e-6 * env[0]
    Gamma_fit = -np.polyfit(t[k], np.log(env[k]), 1)[0]
    phase = np.unwrap(np.arctan2(sy, sx))
    omega_fit = abs(np.polyfit(t, phase, 1)[0])
    return omega_fit, Gamma_fit


print("アルカリ蒸気の自由誘導減衰 "
      f"（gamma_eff/2pi = {gamma_eff / TWO_PI:.4e} Hz/T）")
head = (f"{'B0 (nT)':>10}{'Gamma2 (1/s)':>14}{'nu_L input (Hz)':>17}"
        f"{'nu_L fit':>12}{'Gamma2 fit':>12}")
print(head)
print("-" * len(head))
for B0_nT, G2 in [(1000.0, 100.0), (100.0, 100.0), (100.0, 10.0),
                  (10.0, 10.0)]:
    B0 = B0_nT * 1e-9
    omega0 = gamma_eff * B0
    t_end = min(8.0 / G2, 400.0 / max(omega0, 1e-9))
    t, S = free_induction(B0, G2, t_end)
    om_fit, G_fit = fit_fid(t, S[0], S[1])
    print(f"{B0_nT:>10.0f}{G2:>14.1f}{omega0 / TWO_PI:>17.4f}"
          f"{om_fit / TWO_PI:>12.4f}{G_fit:>12.4f}")

# --- セル内の磁場勾配: 第1章の意味での T2* -------------------------------
# セル内の位置が違えば見る磁場も違うので、集団平均はわずかに異なるLarmor周波数を
# もつ自由誘導減衰の和になります。これはまさに第1章の不均一デフェージングであり、
# 指数関数ではなく T2* = sqrt(2)/sigma_omega のGauss型包絡線を生みます。
B_mean, dB_spread, Gamma2_hom = 1000e-9, 20e-9, 5.0
n_sub = 41
offsets = np.linspace(-4.0, 4.0, n_sub) * dB_spread
weights = np.exp(-0.5 * (offsets / dB_spread) ** 2)
weights /= weights.sum()
t_end = 0.06
t_grid = np.linspace(0.0, t_end, 12001)
Sx_avg = np.zeros_like(t_grid)
Sy_avg = np.zeros_like(t_grid)
for w, off in zip(weights, offsets):
    _, S_i = free_induction(B_mean + off, Gamma2_hom, t_end,
                            n_pts=len(t_grid))
    Sx_avg += w * S_i[0]
    Sy_avg += w * S_i[1]
# 横成分の大きさは、搬送波を除いたコヒーレンスの包絡線そのものです。
env = np.hypot(Sx_avg, Sy_avg)
sigma_omega = gamma_eff * dB_spread
print(f"\nセル内 {dB_spread * 1e9:.0f} nT のばらつきによる不均一デフェージング")
print(f"  均一な Gamma_2 = {Gamma2_hom:.1f} 1/s、したがって T2 = "
      f"{1.0 / Gamma2_hom * 1e3:.1f} ms")
print(f"  sigma_omega = gamma_eff dB = {sigma_omega:.1f} rad/s")
print(f"  予測される T2* = sqrt(2)/sigma_omega = "
      f"{np.sqrt(2.0) / sigma_omega * 1e3:.4f} ms")
model_g = 0.5 * np.exp(-Gamma2_hom * t_grid
                       - 0.5 * (sigma_omega * t_grid) ** 2)
model_e = 0.5 * np.exp(-Gamma2_hom * t_grid
                       - t_grid / (np.sqrt(2.0) / sigma_omega))
print(f"{'t (ms)':>9}{'envelope':>12}{'Gaussian model':>16}"
      f"{'exponential':>14}")
print("-" * 52)
for t_ms in [0.0, 1.0, 2.0, 3.0, 5.0]:
    k = int(np.argmin(np.abs(t_grid - t_ms * 1e-3)))
    print(f"{t_ms:>9.1f}{env[k]:>12.6f}{model_g[k]:>16.6f}"
          f"{model_e[k]:>14.6f}")
print("  Gauss型の列は測定包絡線を追いますが、指数関数の列は追いません。")
print("  勾配は T2* の問題であり、光強度をいくら上げても直りません。")
print("  直せるのは磁場のシミングだけです。")

# --- 定常状態の磁力計応答: 分散型の線形 -----------------------------------
# 蒸気セル磁力計の実際の配置です。ポンプ光を z 方向に当て、測りたい磁場は x
# 方向にあり、磁場がポンプ方向から回した y 成分のスピンを読み出します。
print("\n連続ポンピング下の定常状態: 分散型の Sy 信号")
R_op, Gamma_rel = 30.0, 60.0
Gamma_tot = Gamma_rel + R_op
print(f"  R_op = {R_op:.0f} 1/s、Gamma_rel = {Gamma_rel:.0f} 1/s、"
      f"合計レート = {Gamma_tot:.0f} 1/s")
print(f"{'Bx (nT)':>10}{'x = w0/Gtot':>14}{'Sy':>12}{'Sy analytic':>14}"
      f"{'Sz':>11}")
print("-" * 61)
S_sat = R_op / (2.0 * Gamma_tot)
for Bx_nT in [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]:
    Bx = Bx_nT * 1e-9
    sol = solve_ivp(spin_rhs, (0.0, 3.0), [0.0, 0.0, 0.0],
                    args=((Bx, 0.0, 0.0), Gamma_rel, Gamma_rel, R_op,
                          (0.0, 0.0, 1.0)),
                    rtol=1e-11, atol=1e-14, method="DOP853")
    Sx, Sy, Sz = sol.y[:, -1]
    x = gamma_eff * Bx / Gamma_tot
    print(f"{Bx_nT:>10.1f}{x:>14.4f}{Sy:>12.6f}"
          f"{S_sat * x / (1.0 + x ** 2):>14.6f}{Sz:>11.6f}")
slope_num = (gamma_eff / Gamma_tot) * S_sat
print(f"  ゼロ磁場での勾配 dSy/dBx = {slope_num * 1e-9:.4e} / nT")
print("  磁場単位での半値幅は Gamma_tot/gamma_eff = "
      f"{Gamma_tot / gamma_eff * 1e9:.3f} nT です。")
print("  緩和が遅いほど伝達関数は急峻になり、同時にダイナミックレンジは")
print("  狭くなります。このトレードオフが Example 6 の主題です。")

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for B0_nT, G2 in [(1000.0, 100.0), (1000.0, 20.0)]:
    t, S = free_induction(B0_nT * 1e-9, G2, 0.25)
    ax.plot(t * 1e3, S[0], lw=0.7,
            label=f"B0 = {B0_nT:.0f} nT, $\\Gamma_2$ = {G2:.0f} s$^{{-1}}$")
ax.set_xlabel("time (ms)")
ax.set_ylabel("$S_x$")
ax.set_title("Vapour-cell free induction decay")   # 蒸気セルの自由誘導減衰
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```text
アルカリ蒸気の自由誘導減衰 （gamma_eff/2pi = 4.6708e+09 Hz/T）
   B0 (nT)  Gamma2 (1/s)  nu_L input (Hz)    nu_L fit  Gamma2 fit
-----------------------------------------------------------------
      1000         100.0        4670.8252   4670.8252    100.0000
       100         100.0         467.0825    467.0825    100.0000
       100          10.0         467.0825    467.0825     10.0000
        10          10.0          46.7083     46.7083     10.0000

セル内 20 nT のばらつきによる不均一デフェージング
  均一な Gamma_2 = 5.0 1/s、したがって T2 = 200.0 ms
  sigma_omega = gamma_eff dB = 587.0 rad/s
  予測される T2* = sqrt(2)/sigma_omega = 2.4094 ms
   t (ms)    envelope  Gaussian model   exponential
----------------------------------------------------
      0.0    0.500000        0.500000      0.500000
      1.0    0.418815        0.418782      0.328511
      2.0    0.248537        0.248534      0.215839
      3.0    0.104511        0.104511      0.141811
      5.0    0.006558        0.006574      0.061216
  Gauss型の列は測定包絡線を追いますが、指数関数の列は追いません。
  勾配は T2* の問題であり、光強度をいくら上げても直りません。
  直せるのは磁場のシミングだけです。

連続ポンピング下の定常状態: 分散型の Sy 信号
  R_op = 30 1/s、Gamma_rel = 60 1/s、合計レート = 90 1/s
   Bx (nT)   x = w0/Gtot          Sy   Sy analytic         Sz
-------------------------------------------------------------
     -20.0       -6.5217   -0.024969     -0.024969   0.003829
      -5.0       -1.6304   -0.074280     -0.074280   0.045559
      -1.0       -0.3261   -0.049124     -0.049124   0.150648
       0.0        0.0000    0.000000      0.000000   0.166667
       1.0        0.3261    0.049124      0.049124   0.150648
       5.0        1.6304    0.074280      0.074280   0.045559
      20.0        6.5217    0.024969      0.024969   0.003829
  ゼロ磁場での勾配 dSy/dBx = 5.4348e-02 / nT
  磁場単位での半値幅は Gamma_tot/gamma_eff = 3.067 nT です。
  緩和が遅いほど伝達関数は急峻になり、同時にダイナミックレンジは
  狭くなります。このトレードオフが Example 6 の主題です。
```

**着目点。** 最初の表は、積分器が含んでいるべき物理に対する校正です。フィットされたLarmor周波数と横緩和レートが入力を小数4桁で返すので、以下のもっと自明でない場合についてもソルバを信頼できます。その中で有用な数値は $\gamma_\mathrm{eff}/2\pi = 4.67$ Hz/nT です。この種の磁力計が働く数十ナノテスラの磁場ではアルカリスピンはキロヘルツで歳差し、地磁気中ではメガヘルツです。

2番目のブロックは第1章の語彙が働くところです。セル内の20 nTの磁場のばらつき — 揺らぎではなく勾配 — は原子ごとに歳差の速さを違わせ、個々の原子は $T_2 = 200$ ms のあいだコヒーレントであり続けるのに、集団平均は $T_2^\ast = \sqrt{2}/\sigma_\omega = 2.409$ ms の**Gauss型**で減衰します。1 msでの測定包絡線 $0.418815$ はGauss型の予測 $0.418782$ と5桁で一致し、指数関数の $0.328511$ には遠く及びません。これは1.4節の診断を逆向きに使ったもので、減衰の*形*が問題が揺らぎなのか静的な不均一なのかを述べており、ここではセルのコーティングを良くするのではなく磁気シールドを良くせよと言っています。この Example では2桁の感度が勾配に失われており、レーザー強度をいくら上げても取り戻せません。

3番目のブロックは実際の伝達関数を計算します。ポンプを $z$ に、測りたい磁場を $x$ にとると、定常状態の $S_y$ は $x = \omega_0/\Gamma_\mathrm{tot}$ とした分散型Lorentz関数 $S_\mathrm{sat}\, x/(1+x^2)$ となり — 閉じた形と6桁で検証されています — そのゼロ磁場での勾配 $5.43\times10^{-2}$ / nT が磁力計が実際に読む量です。半値幅 $\Gamma_\mathrm{tot}/\gamma_\mathrm{eff} = 3.067$ nT が線形範囲です。構造に注意してください。緩和を遅くすると勾配は急峻になり、*同時に*線形範囲は狭くなり、その関係は厳密に比例です。これは4.3節の干渉計の縞と同じ感度対ダイナミックレンジのトレードオフであり、次の Example への導入になります。

### スピン交換と、それが問題でなくなる領域

高密度アルカリ蒸気での支配的な緩和は**スピン交換衝突**です。2つのアルカリ原子が衝突して電子スピンを交換し、全スピンは保存されますが各原子の超微細状態はランダム化されます。Larmor歳差が衝突レート $R_\mathrm{se}$ より速い通常の領域では、これが位相をランダム化して横緩和に $R_\mathrm{se}/2$ 程度を寄与します。$R_\mathrm{se}$ は密度とともに増え、密度こそがセンサに原子を与えるものなので、これは硬い天井のように見えます。

逃げ道は、スピン交換が全スピンを保存することにあります。衝突が歳差よりずっと*速い*なら、相対位相が育つ前に原子は何度もスピンを交換し、集団は単一の集団スピンとして歳差して、スピン交換の寄与は $(\omega_0/R_\mathrm{se})^2$ で抑制されます。これが**スピン交換緩和のない（SERF）**領域であり、そこへ到達するには条件が $\omega_0 \ll R_\mathrm{se}$ なので小さな磁場が必要です。抑制は目を見張るもので、そして即座に代価が付いてきます。ナノテスラの磁場の中に座らねばならない磁力計は磁気シールドを必要とし、自分がシールドしている磁場を測ることはできず、そして — 苦労して下げた緩和レートが帯域も決めるので — 遅いのです。

### Code Example 6: SERFの移り変わりに値段を付ける

```python
"""第4章 Code Example 6: SERFの移り変わりと、それが強いる感度-帯域のトレードオフ。
Code Example 1-5 の続き（同一セッション）。"""

kB = 1.380649e-23


def gamma_spin_exchange(omega0, R_se):
    """スピン交換による横緩和を、移り変わりの領域をまたいで内挿します。

    2つの極限が物理です。omega0 >> R_se ではZeeman準位が分離され、交換衝突の
    たびに位相がランダム化されて R_se/2 を与えます。omega0 << R_se では衝突が
    速すぎてスピンは衝突の合間に位相差を作れず、寄与は (omega0/R_se)^2 で
    抑制されます。以下のLorentz型内挿は両端を正しいべきで再現しますが、
    移り変わりの領域については定性的なものにすぎません。
    """
    x = (omega0 / R_se) ** 2
    return 0.5 * R_se * x / (1.0 + x)


def cell_rates(B0, R_se, Gamma_sd, Gamma_wall, R_op):
    """アルカリ蒸気の全横緩和レート（1/s）。"""
    omega0 = gamma_eff * B0
    return (gamma_spin_exchange(omega0, R_se) + Gamma_sd + Gamma_wall + R_op)


def dB_sensitivity(Gamma2, n_density, volume, gamma=gamma_eff):
    """スピン射影ノイズ限界での磁場感度（T/sqrt(Hz)）。

    delta_B = (1/gamma) sqrt(Gamma2 / (n V)) です。これは、それぞれ 1/Gamma2
    のあいだコヒーレントな無相関スピン n*V 個の集団に対する標準的な結果です。
    """
    return np.sqrt(Gamma2 / (n_density * volume)) / gamma


n_alkali = 1.0e20        # atoms/m^3、150 °C 付近の高温セル
V_cell = (3e-3) ** 3     # 3 mm 立方
R_se = 1.0e5             # 1/s、その密度でのスピン交換レート
Gamma_sd = 30.0          # 1/s、スピン破壊
Gamma_wall = 20.0        # 1/s、壁と拡散による損失
R_op_bg = 30.0           # 1/s、光ポンピング

print("SERFの移り変わりをまたいだスピン交換緩和")
print(f"  R_se = {R_se:.0e} 1/s なので、移り変わりは "
      f"B0 = {R_se / gamma_eff * 1e9:.1f} nT に位置します")
head = (f"{'B0 (nT)':>10}{'nu_L (Hz)':>12}{'G_SE (1/s)':>12}"
        f"{'G_2 (1/s)':>11}{'BW (Hz)':>9}{'eta_B (fT/rtHz)':>17}")
print(head)
print("-" * len(head))
for B0_nT in [10_000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0]:
    B0 = B0_nT * 1e-9
    G_se = gamma_spin_exchange(gamma_eff * B0, R_se)
    G2 = cell_rates(B0, R_se, Gamma_sd, Gamma_wall, R_op_bg)
    eta = dB_sensitivity(G2, n_alkali, V_cell)
    print(f"{B0_nT:>10.0f}{gamma_eff * B0 / TWO_PI:>12.2f}{G_se:>12.2f}"
          f"{G2:>11.2f}{G2 / TWO_PI:>9.2f}{eta * 1e15:>17.3f}")

print("\nトレードオフを率直に述べます:")
B_hi, B_lo = 3000e-9, 3e-9
G_hi = cell_rates(B_hi, R_se, Gamma_sd, Gamma_wall, R_op_bg)
G_lo = cell_rates(B_lo, R_se, Gamma_sd, Gamma_wall, R_op_bg)
print(f"  at B0 = 3000 nT: Gamma_2 = {G_hi:8.1f} 1/s, "
      f"bandwidth {G_hi / TWO_PI:7.1f} Hz, "
      f"eta = {dB_sensitivity(G_hi, n_alkali, V_cell) * 1e15:7.2f} fT/rtHz")
print(f"  at B0 =    3 nT: Gamma_2 = {G_lo:8.1f} 1/s, "
      f"bandwidth {G_lo / TWO_PI:7.1f} Hz, "
      f"eta = {dB_sensitivity(G_lo, n_alkali, V_cell) * 1e15:7.2f} fT/rtHz")
print(f"  感度で {np.sqrt(G_hi / G_lo):.1f} 倍を、帯域の "
      f"{G_hi / G_lo:.1f} 倍で支払っています。")

# --- 密度もまた自由なパラメータではありません -----------------------------
print("\n密度を上げれば衝突レートも上がるので、利得は飽和します:")
print(f"{'n (1/m^3)':>12}{'R_se (1/s)':>12}{'G_SE at 3 nT':>14}"
      f"{'G_2 (1/s)':>11}{'eta (fT/rtHz)':>15}")
print("-" * 64)
for n_i in [1e19, 3e19, 1e20, 3e20, 1e21, 3e21, 1e22]:
    R_se_i = 1.0e5 * (n_i / 1.0e20)
    G_sd_i = 30.0 * (n_i / 1.0e20)          # spin destruction also scales with n
    G2_i = cell_rates(B_lo, R_se_i, G_sd_i, Gamma_wall, R_op_bg)
    eta_i = dB_sensitivity(G2_i, n_i, V_cell)
    print(f"{n_i:>12.0e}{R_se_i:>12.0e}"
          f"{gamma_spin_exchange(gamma_eff * B_lo, R_se_i):>14.2e}"
          f"{G2_i:>11.1f}{eta_i * 1e15:>15.3f}")
# スピン破壊が支配的になると Gamma_2 は n に比例し、eta は改善しなくなります。
# 漸近値を決めるのは原子1個あたりのスピン破壊断面積だけで、セル内の原子数では
# ありません。
eta_inf = np.sqrt(30.0 / 1e20 / V_cell) / gamma_eff
print(f"  スピン破壊による漸近値 sqrt(G_sd/n / V)/gamma = "
      f"{eta_inf * 1e15:.3f} fT/rtHz")
print("  Gamma_2 が n に比例して増えるようになると、原子を増やしても何も得ません。")

# --- 体積のスケーリング: 空間分解能とのトレードオフ -----------------------
print("\n密度と Gamma_2 を固定したときの、センサ寸法に対する感度:")
print(f"{'cell edge a':>13}{'volume (m^3)':>15}{'eta_B (fT/rtHz)':>18}"
      f"{'eta * a^1.5 (a.u.)':>21}")
print("-" * 67)
G2_fixed = cell_rates(B_lo, R_se, Gamma_sd, Gamma_wall, R_op_bg)
for a_mm in [10.0, 3.0, 1.0, 0.3, 0.1]:
    a = a_mm * 1e-3
    eta_a = dB_sensitivity(G2_fixed, n_alkali, a ** 3)
    print(f"{a_mm:>10.1f} mm{a ** 3:>15.2e}{eta_a * 1e15:>18.3f}"
          f"{eta_a * a ** 1.5 * 1e15:>21.4e}")
print("  eta_B は a^(-3/2) でスケールします。空間分解能を1桁上げるたびに")
print("  磁場感度は 31.6 倍悪くなります。第5章がこれに戻ってくるのは、")
print("  局在した信号源では磁場のほうがもっと速く増えるからです。")

print("\nこの Example には極低温がどこにも出てきません:")
h_planck = 6.62607015e-34
eV = 1.602176634e-19
kT = kB * 420.0
hnu = h_planck * gamma_eff * B_lo / TWO_PI
print(f"  セル温度は約420 K、熱エネルギー kT = {kT / eV * 1e3:.2f} meV")
print(f"  3 nT でのZeeman分裂: h nu = {hnu / eV * 1e15:.3f} feV")
print(f"  h nu / kT = {hnu / kT:.3e} なので、熱的偏極は完全に無視できます")
print("  -- そしてそれは問題になりません。偏極は kT ではなく光ポンピングから")
print("  来るからです。動作温度が超伝導ギャップで決まる第3章のSQUIDと")
print("  比べてみてください。")
```

```text
SERFの移り変わりをまたいだスピン交換緩和
  R_se = 1e+05 1/s なので、移り変わりは B0 = 3407.4 nT に位置します
   B0 (nT)   nu_L (Hz)  G_SE (1/s)  G_2 (1/s)  BW (Hz)  eta_B (fT/rtHz)
-----------------------------------------------------------------------
     10000    46708.25    44798.63   44878.63  7142.66            4.393
      3000    14012.48    21833.47   21913.47  3487.64            3.070
      1000     4670.83     3964.93    4044.93   643.77            1.319
       300     1401.25      384.60     464.60    73.94            0.447
       100      467.08       43.03     123.03    19.58            0.230
        30      140.12        3.88      83.88    13.35            0.190
        10       46.71        0.43      80.43    12.80            0.186
         3       14.01        0.04      80.04    12.74            0.186
         1        4.67        0.00      80.00    12.73            0.185

トレードオフを率直に述べます:
  at B0 = 3000 nT: Gamma_2 =  21913.5 1/s, bandwidth  3487.6 Hz, eta =    3.07 fT/rtHz
  at B0 =    3 nT: Gamma_2 =     80.0 1/s, bandwidth    12.7 Hz, eta =    0.19 fT/rtHz
  感度で 16.5 倍を、帯域の 273.8 倍で支払っています。

密度を上げれば衝突レートも上がるので、利得は飽和します:
   n (1/m^3)  R_se (1/s)  G_SE at 3 nT  G_2 (1/s)  eta (fT/rtHz)
----------------------------------------------------------------
       1e+19       1e+04      3.88e-01       53.4          0.479
       3e+19       3e+04      1.29e-01       59.1          0.291
       1e+20       1e+05      3.88e-02       80.0          0.186
       3e+20       3e+05      1.29e-02      140.0          0.142
       1e+21       1e+06      3.88e-03      350.0          0.123
       3e+21       3e+06      1.29e-03      950.0          0.117
       1e+22       1e+07      3.88e-04     3050.0          0.115
  スピン破壊による漸近値 sqrt(G_sd/n / V)/gamma = 0.114 fT/rtHz
  Gamma_2 が n に比例して増えるようになると、原子を増やしても何も得ません。

密度と Gamma_2 を固定したときの、センサ寸法に対する感度:
  cell edge a   volume (m^3)   eta_B (fT/rtHz)   eta * a^1.5 (a.u.)
-------------------------------------------------------------------
      10.0 mm       1.00e-06             0.030           3.0484e-05
       3.0 mm       2.70e-08             0.186           3.0484e-05
       1.0 mm       1.00e-09             0.964           3.0484e-05
       0.3 mm       2.70e-11             5.867           3.0484e-05
       0.1 mm       1.00e-12            30.484           3.0484e-05
  eta_B は a^(-3/2) でスケールします。空間分解能を1桁上げるたびに
  磁場感度は 31.6 倍悪くなります。第5章がこれに戻ってくるのは、
  局在した信号源では磁場のほうがもっと速く増えるからです。

この Example には極低温がどこにも出てきません:
  セル温度は約420 K、熱エネルギー kT = 36.19 meV
  3 nT でのZeeman分裂: h nu = 57.951 feV
  h nu / kT = 1.601e-12 なので、熱的偏極は完全に無視できます
  -- そしてそれは問題になりません。偏極は kT ではなく光ポンピングから
  来るからです。動作温度が超伝導ギャップで決まる第3章のSQUIDと
  比べてみてください。
```

**着目点。** 移り変わりは $\omega_0 = R_\mathrm{se}$ となるところ、このセルでは $3407$ nT に位置し、表はそこを横断します。$B_0 = 3000$ nT ではスピン交換項は $2.18\times10^{4}$ s$^{-1}$ で他のすべてを支配し、$B_0 = 30$ nT では $3.88$ s$^{-1}$ まで落ちて残りの緩和はスピン破壊、壁衝突、ポンプ光になります。磁場感度は $3.07$ から $0.19$ fT$/\sqrt{\mathrm{Hz}}$ へ、$16.5$ 倍改善し、帯域は $3488$ Hz から $12.7$ Hz へ、$274$ 倍落ちます。これがSERFのトレードオフであり、いまや形容詞ではなく数値です。**感度の16倍は帯域の274倍で支払われます**。感度は $\sqrt{\Gamma_2}$ で、帯域は $\Gamma_2$ で動くからです。

密度の走査は、自然だが誤った直観を訂正します。原子が増えれば $1/\sqrt{n}$ で感度が良くなるはずですが、衝突レートも $n$ とともに増えます。SERF領域ではスピン交換は抑制されますが、スピン*破壊*は抑制されず、$\Gamma_2 \propto n$ になると感度 $\sqrt{\Gamma_2/n}$ は改善をやめます。走査は $0.115$ fT$/\sqrt{\mathrm{Hz}}$ で平らになり、計算されたスピン破壊の漸近値 $0.114$ と合います。残るてこは原子1個あたりのスピン破壊断面積で、これは化学 — バッファガス、セルのコーティング、どのアルカリか — であって物理ではありません。

体積の走査は第5章へ持って行くべき数値です。感度はセルの線寸法で $a^{-3/2}$ とスケールするので、空間分解能を1桁上げるたびに磁場感度は $31.6$ 倍悪くなり、積 $\eta_B a^{3/2}$ は列を下って5桁で一定です。$0.186$ fT$/\sqrt{\mathrm{Hz}}$ の3 mmセルは $30.5$ fT$/\sqrt{\mathrm{Hz}}$ の100 µmセルになります。これが悪いトレードオフかどうかは完全に信号源に依存します。5.4節が示すように、局在した信号源では使える磁場が $a^{-3}$ で増えて感度の劣化より速いので小さいプローブが勝ち、一様磁場では勝ちません。

締めのブロックは熱力学的な要点を率直に述べます。3 nTでのZeemanエネルギーはセル内の $k_BT$ の $1.6\times10^{-12}$ なので熱的偏極はまったく無きに等しく、それでもセンサは働きます。光ポンピングは気にしないからです。[第3章](<chapter-3.html>)のSQUIDと比べてください。あちらの動作温度は設計上の選択ではなく、超伝導ギャップを必要とすることの帰結です。2つのフェムトテスラ磁力計、動作温度の理由はまったく別。

### 本コースの3つの磁力計はどこに位置するか

| | 蒸気セル（SERF） | SQUID | NV中心 |
| --- | --- | --- | --- |
| 結合する物理量 | アルカリ原子の電子スピン | ループを貫く磁束 | 格子欠陥の電子スピン |
| 動作温度 | 約420 K、極低温なし | 膜の $T_c$ 以下 | 4 Kから600 K |
| 検出体積 | mm$^3$ | ループ面積、µm$^2$ 以上 | nm$^3$（単一）からmm$^3$（集団） |
| 到達できる距離 | mm | µm | nm |
| 帯域の限界 | 緩和レート、SERFでは数十Hz | 増幅器とループ、MHz | $1/T_2$ か駆動、MHzからGHz |
| 最良時の支配ノイズ | スピン射影、スピン破壊 | 表面スピン由来の $1/f$ 磁束ノイズ | スピン射影、表面スピンバス |
| 材料としての問い | セルのコーティング、バッファガス、アルカリの選択 | 膜表面、接合の酸化膜 | ダイヤモンド表面終端、N から NV への変換率 |
| ベクトルかスカラーか | 配置次第でどちらも | 磁束、したがって1成分 | ベクトル、N-V軸から |

下から2番目の行を読めば、3つとも表面に律速されています。セル壁、超伝導膜の酸化膜、終端されたダイヤモンド表面です。これは量子ビットについてハードウェアコースが到達する結論と同じであり、本コースが材料科学の道場に置かれている理由です。

* * *

## 4.5 本章が加えるもの

移植できる3項目を、簡潔に述べます。

**成果物は誤差バジェットです。** 感度の数値はバジェットの1行です。装置が有用かどうかを決めるのは残りの行であり、それを書き下す規律 — 各項に関数形、見積もられた大きさ、抑えられた不確かさを与えること — が Example 3 が示すものです。走査NV顕微鏡やSQUID磁化率計に適用すれば、同じ表は違う中身と同じ構造をもちます。

**安定度と正確度は別の数です。** 平均化は一方を減らして他方を減らさず、その交差時間は計算できます。系統の床が到来する平均化時間を書かずに感度を報告するのは、結果の半分を報告しているだけです。

**自由原子は材料問題を取り除き、その下にあるものを露わにします。** 蒸気セルの検出体積には格子も界面も欠陥集団もなく、その限界は衝突断面積、壁のコーティング、光ポンピング効率です。これはハードウェアコースが中性原子から引き出すのと同じ教訓です。固体を取り除いても限界は取り除かれず、位置が変わるだけなのです。第5章は、エンタングルメントがそれをもう一度動かせるかを問います。

* * *

## 演習

#### 演習1: 時計の安定度の算術

あるRamsey時計が $\nu_0 = 4.0\times10^{14}$ Hz の遷移で、自由発展時間 $T = 0.4$ s、原子数 $N = 2500$、サイクル時間 $T_c = 1.0$ s で動作しています。

  1. $Q$、Ramsey線幅、$\sigma_y(1\ \mathrm{s})$ を計算してください。
  2. 分数不確かさ $1\times10^{-17}$ に達するには何秒平均する必要がありますか。
  3. $T$ を2倍にするのと $N$ を4倍にするのでは、どちらが価値がありますか。両方の係数を示し、どちらが物理的に容易か述べてください。
  4. $T$ を変えずにサイクル時間を $1.0$ s から $0.5$ s に減らしました。$\sigma_y(\tau)$ は何倍改善しますか。またこれがしばしば最も安価な利得である理由を述べてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(Q = 2\nu_0 T = 2 \times 4.0\times10^{14} \times 0.4 = 3.2\times10^{14}\) です。Ramsey線幅は \(\Delta\nu = 1/2T = 1.25\) Hz。したがって \(\sigma_y(1\ \mathrm{s}) = 1/(\pi Q \sqrt{N}) \times \sqrt{T_c/\tau} = 1/(\pi \times 3.2\times10^{14} \times 50) = 1.989\times10^{-17}\) です。</p>

<p><strong>2.</strong> \(\sigma_y \propto \tau^{-1/2}\) なので \(\tau = (1.989\times10^{-17}/1\times10^{-17})^2 = 3.96\) s です。\(10^{-17}\) に達するには4秒の平均化で済み、これが Example 3 の要点です。この問題の統計的な部分は易しいのです。</p>

<p><strong>3.</strong> \(T\) を2倍にすると \(Q\) が2倍になって2倍の利得、\(N\) を4倍にしても \(\sqrt{4} = 2\) で同じく2倍です。まったく同じ価値です。物理的には、\(T\) を2倍にするのは \(T_2\) と観測レーザーのコヒーレンスで上から抑えられ、1秒を超えるとふつうはレーザーが律速します。\(N\) を4倍にするのは低温衝突による密度シフトで抑えられ、これは安定度ではなく正確度のバジェットに入ります。どちらも無料ではなく、どちらが容易かはどちらのバジェットに余裕があるかで決まります。</p>

<p><strong>4.</strong> \(\sigma_y(\tau) \propto \sqrt{T_c}\) なので、\(T_c\) を半分にすれば安定度は \(\sqrt{2} = 1.41\) 倍改善します。しばしば最も安価なのは、デッドタイムが準備と読み出し、すなわち物理ではなく工学であるからで、さらに局所発振器ノイズのDick効果による折り返しも減って、2つ目の項を同時に小さくするからです。</p>

```python
import numpy as np
nu0, T, N, Tc = 4.0e14, 0.4, 2500.0, 1.0
Q = 2 * nu0 * T
s1 = 1.0 / (np.pi * Q * np.sqrt(N)) * np.sqrt(Tc / 1.0)
print(f"Q = {Q:.3e}, 線幅 = {1/(2*T):.2f} Hz, sigma_y(1 s) = {s1:.4e}")
print(f"1e-17 までの tau = {(s1 / 1e-17)**2:.2f} s")
print(f"Tc を半分にすると {np.sqrt(2):.3f} 倍")
# Q = 3.200e+14, 線幅 = 1.25 Hz, sigma_y(1 s) = 1.9894e-17
# 1e-17 までの tau = 3.96 s
# Tc を半分にすると 1.414 倍
```

</details>

#### 演習2: Allanプロットを読む

ある時計のAllan偏差が $\sigma_y(1\ \mathrm{s}) = 2\times10^{-14}$ と測定され、$\tau = 100$ s まで $\tau^{-1/2}$ で下がり、$\tau = 10^{4}$ s まで $2\times10^{-15}$ で平坦、その後 $\tau^{+1}$ で上昇しました。

  1. 3つの領域それぞれで支配的なノイズ過程を挙げてください。
  2. 達成できる最良の安定度はいくらで、どの平均化時間ですか。
  3. 同僚がこの時計について「$\sigma_y = 4\times10^{-16}$」と報告しました。どのような測定ならそれが成り立ちえますか。また、なぜその主張は不完全ですか。
  4. 平坦な領域の原因が観測レーザーだと判明しました。それは安定度の問題ですか正確度の問題ですか。またその答えは労力をどこに割くべきかについて何を意味しますか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> 白色周波数ノイズ（傾き \(-1/2\)）、フリッカー周波数ノイズ（傾き 0）、線形周波数ドリフト（傾き \(+1\)）です。中間の領域は減衰ではなく床です。</p>

<p><strong>2.</strong> 最良の安定度は床の \(2\times10^{-15}\) で、\(\tau = 100\) s で最初に到達し \(10^{4}\) s まで保たれます。\(10^{4}\) s を超えて平均すると結果は悪くなるので、有用な動作点はその1桁半の平坦部のどこかです。</p>

<p><strong>3.</strong> この時計ではどの \(\tau\) でも成り立ちません。曲線の最小値が \(2\times10^{-15}\) だからです。平均化時間を伴わない安定度の数値は無意味であり、そうした数値が現れる最もありふれた経路は、\(\tau^{-1/2}\) の領域を、それが適用されなくなった先まで外挿することです。ここでは \(\tau = 2500\) s で \(2\times10^{-14}/\sqrt{2500} = 4\times10^{-16}\) となりますが、そこでは実際の曲線はすでに上昇しています。</p>

<p><strong>4.</strong> 安定度の問題です。揺らぐ局所発振器は時計の出力を動かしますが平均値にバイアスを与えないので、正確度のバジェットには入りません。したがって労力は原子ではなく共振器 — 熱雑音限界のコーティング損失、スペーサ材料、温度制御 — へ向かいます。これがレーザーが原子時計を律速するDick効果の経路であり、基準共振器の材料が活発な主題である理由です。</p>

</details>

#### 演習3: バジェットの意思決定

ある光時計の不確かさの寄与が次のとおりです。黒体放射 $8\times10^{-17}$（囲いは300 K、温度は1 Kの精度で既知）、格子光シフト $2\times10^{-17}$、2次Zeeman $1\times10^{-17}$、密度シフト $5\times10^{-18}$、赤方偏移 $1\times10^{-18}$。

  1. 二乗和での合計を計算してください。
  2. 囲いを150 Kに冷やすか、囲いの温度の知識を $0.1$ Kまで改善するかを選べます。それぞれの新しい合計を計算し、選択してください。
  3. 選んだ対策の後、どの項が支配的になりますか。また次に大きい項より小さくなるには何倍改善する必要がありますか。
  4. 査読者が「時計を地下に置けば赤方偏移の項は単純に消せるのではないか」と尋ねました。答えてください。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\sqrt{80^2 + 20^2 + 10^2 + 5^2 + 1^2}\times10^{-18} = \sqrt{6400 + 400 + 100 + 25 + 1}\times10^{-18} = 83.4\times10^{-18} = 8.34\times10^{-17}\) です。</p>

<p><strong>2.</strong> 150 Kに冷やすと黒体放射の不確かさは \((150/300)^3 = 1/8\) 倍になって \(1\times10^{-17}\)、合計は \(\sqrt{10^2+20^2+10^2+5^2+1^2} = 24.6\times10^{-18} = 2.46\times10^{-17}\) です。温度の知識を10倍改善すると黒体放射の不確かさは10分の1の \(8\times10^{-18}\)、合計は \(\sqrt{8^2+20^2+10^2+5^2+1^2} = 23.9\times10^{-18} = 2.39\times10^{-17}\) です。丸めの範囲で両者は等価なので、選択はコストと副作用で決まります。極低温の囲いは迷走する熱光による交流Starkの寄与と背景ガスとの衝突レートも減らしますが、より良い温度計はどちらもしません。算術は引き分けでも、冷却のほうが良い工学の答えです。</p>

<p><strong>3.</strong> \(2\times10^{-17}\) の格子光シフトが、どちらの選択でも支配的になります。次に大きい項（\(1\times10^{-17}\) の2次Zeeman）より下に落ちるには2倍を超える改善が必要で、それは格子波長を安定化するか、より浅いトラップ深さで動作させることを意味します。後者は原子数、したがって安定度を代価にします。バジェットは互いに結合しているのです。</p>

<p><strong>4.</strong> 赤方偏移は取り除くべき厄介事ではなく、実在してよく理解された周波数差だからです。下にある時計は本当に、1メートルあたり \(1.09\times10^{-16}\) だけ遅く進みます。この項の不確かさは時計の<em>基準ジオイドからの高さ</em>の不確かさであり、それは物理ではなく測地の測定です。地下へ移しても減らず、地下を測量すれば減ります。だからこそ時計の比較それ自体が測地の道具として提案されているのです。</p>

</details>

#### 演習4: 原子干渉計の寸法決め

あるMach-Zehnder重力計が $k_\mathrm{eff} = 1.61\times10^{7}$ m$^{-1}$、1ショットあたり $N = 10^{6}$ 原子を使います。

  1. 1ショットの不確かさ $1\times10^{-9}\,g$ を得るには $T$ をいくらにする必要がありますか。
  2. その $T$ が要求する噴水の頂点高さはいくらですか。また反跳速度が $11.8$ mm/s なら頂点での2経路の間隔はいくらですか。
  3. その $T$ で縞1本に対応する加速度はいくらですか。局所の $g$ が事前に $1\times10^{-5}\,g$ の精度で分かっているとして、測定は一意に決まりますか。
  4. 反射鏡の振動が実効値 $10^{-6}$ m s$^{-2}$ で存在します。それは測定に何をしますか。また標準的な対策は何ですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> \(\delta a = 1/(\sqrt{N} k_\mathrm{eff} T^2)\) なので \(T = \left[1/(\sqrt{N} k_\mathrm{eff}\, \delta a)\right]^{1/2}\)、ここで \(\delta a = 10^{-9} \times 9.807 = 9.807\times10^{-9}\) m s\(^{-2}\) です。したがって \(T^2 = 1/(10^3 \times 1.61\times10^7 \times 9.807\times10^{-9}) = 6.33\times10^{-3}\) s\(^2\)、すなわち \(T = 79.6\) ms です。</p>

<p><strong>2.</strong> 頂点は \(h = gT^2/2 = 9.807 \times 6.33\times10^{-3}/2 = 3.1\) cm、射出点より上です。控えめな高さで、だからナノ \(g\) の重力計は机に載ります。頂点での経路間隔は \(v_\mathrm{rec} T = 11.8\ \mathrm{mm/s} \times 0.0796\ \mathrm{s} = 0.94\) mm です。</p>

<p><strong>3.</strong> 縞1本は \(\delta a_\mathrm{fringe} = 2\pi/(k_\mathrm{eff}T^2) = 2\pi/(1.61\times10^7 \times 6.33\times10^{-3}) = 6.16\times10^{-5}\) m s\(^{-2}\)、すなわち \(6.3\times10^{-6}\,g\) です。\(10^{-5}\,g\) の事前知識は縞1本より広い範囲にわたるので、測定は一意に<em>決まりません</em>。事前知識を改善する（機械式重力計、あるいは先に短い \(T\) で走らせる）か、\(T\) を走査して複数の値で整合する位相を見つけるかが必要です。</p>

<p><strong>4.</strong> 鏡の振動は原子の加速度と区別できません。干渉計が測るのは相対加速度だからです。\(10^{-6}\) m s\(^{-2}\) は1ショットのノイズの100倍なので完全に支配し、ショットごとに縞をランダム化します。対策はすべて差動です。鏡に地震計を付けて測定位相から差し引く、受動的に振動を絶縁する、あるいは最も頑健には勾配計 — 同じレーザーと鏡を共有する2台の干渉計で、その<em>差</em>は共通のプラットフォーム運動に鈍感 — にすることです。この最後の選択肢が、より小さい量を測るのに勾配測定が絶対重力測定より易しい理由です。</p>

```python
import numpy as np
keff, N, g = 1.61e7, 1e6, 9.80665
da = 1e-9 * g
T = (1.0 / (np.sqrt(N) * keff * da)) ** 0.5
print(f"T = {T*1e3:.1f} ms、頂点 = {0.5*g*T**2*100:.1f} cm、"
      f"経路間隔 = {11.8e-3*T*1e3:.2f} mm")
print(f"縞1本 = {2*np.pi/(keff*T**2):.3e} m/s^2 = "
      f"{2*np.pi/(keff*T**2)/g:.2e} g")
# T = 79.6 ms、頂点 = 3.1 cm、経路間隔 = 0.94 mm
# 縞1本 = 6.162e-05 m/s^2 = 6.28e-06 g
```

</details>

#### 演習5: SERFセルとSQUIDのどちらを選ぶか

ある計測が、300 Kに保たれた試料の表面から $2$ mm 下にある信号源からの $50$ fT の信号を、$200$ Hz で検出することを要求しています。

  1. Example 6 から、SERF領域の深くで運転したときの蒸気セルの帯域はいくらですか。200 Hz の信号は見えますか。
  2. 帯域を上げる選択肢は何があり、それぞれ何を代価にしますか。
  3. 第3章のSQUIDは帯域が十分にあります。試料が300 Kであることを考えると、代わりに何を代価にしますか。
  4. 信号源が表面から $2$ mm ではなく $2$ µm 下にあり、必要な空間分解能が $1$ µm だとします。本コースの3つのセンサのうちどれが生き残りますか。またその劣った磁場感度が失格の理由にならないのはなぜですか。

<details>
<summary>解答</summary>

<p><strong>1.</strong> SERF領域の深くでは Example 6 の全横緩和レートは約 \(80\ \mathrm{s^{-1}}\)、すなわち帯域 \(12.7\) Hz です。200 Hz の信号はその16倍上にあり、1極の応答ではおよそ \(200/12.7 \approx 16\) 倍減衰します。したがってこの構成では見えません。</p>

<p><strong>2.</strong> 選択肢は3つ、いずれもトレードオフです。(i) ポンプレート \(R_\mathrm{op}\) を上げる。帯域は \(\Gamma_2\) に比例して増え、感度は \(\sqrt{\Gamma_2}\) で劣化するので、200 Hz に達する代価は \(\eta_B\) の \(\sqrt{16} = 4\) 倍 — \(0.19\) から約 \(0.75\) fT\(/\sqrt{\mathrm{Hz}}\) で、なお50 fTには十分です。(ii) SERF領域を離れてより大きな磁場で運転する。応答も広がりますがスピン交換項が加わります。(iii) 閉ループ構成を使い、\(\Gamma_2\) を変えずに<em>使える</em>帯域を伸ばす。代価はループの複雑さです。ここでは (i) で十分で、それが誠実な答えです。感度の余裕は使えるだけ大きいのです。</p>

<p><strong>3.</strong> SQUIDは膜の臨界温度以下でなければならず、試料は300 Kなので、両者は真空の間隙と窓で隔てられます。その距離が代価です。2 mmの信号源の深さに極低温の窓が加わると合計の距離は数ミリメートルに達することがあり、局在した信号源では磁場は \(1/d^3\) で落ちます。5 mmのフェムトテスラセンサは、0.5 mmのピコテスラセンサより悪くなりえます。室温動作は便利さではなく、距離のバジェットなのです。</p>

<p><strong>4.</strong> NV中心だけです。検出体積をナノメートルにでき、試料に接して300 Kで動作するので、1 µm 以下のオーダーの距離が達成できます。磁場感度はどちらの代替案よりも桁で劣り、それが失格の理由にならないのは、局在した信号源からの磁場が \(1/d^3\) で増えるのに対し、小さくなるプローブの感度は \(a^{-3/2}\) でしか劣化しないからです（Example 6）。距離を 5 mm から 1 µm に縮めるのは磁場で \(1.25\times10^{11}\) 倍です。本コースにそれほど大きな感度比はどこにもありません。第5章5.4節がこの比較を定量的に行います。</p>

```python
import numpy as np
G2_serf, eta_serf = 80.0, 0.186e-15
needed_bw = 200.0
G2_needed = 2 * np.pi * needed_bw
print(f"実装時の帯域       {G2_serf/(2*np.pi):.1f} Hz")
print(f"必要な Gamma_2     {G2_needed:.0f} 1/s、倍率 "
      f"{G2_needed/G2_serf:.1f}")
print(f"広げた後の eta     {eta_serf*np.sqrt(G2_needed/G2_serf)*1e15:.2f} "
      f"fT/rtHz（信号 50 fT に対して）")
# 実装時の帯域       12.7 Hz
# 必要な Gamma_2     1257 1/s、倍率 15.7
# 広げた後の eta     0.74 fT/rtHz（信号 50 fT に対して）
```

</details>

* * *

## まとめ

### 要点

**1\. 時計とはフィードバックループに入れたRamsey推定器**

  * 縞は半値バイアス点で勾配 $T/2$ をもつ周波数弁別器なので、時計はわざと共鳴から外して運転されます。
  * 射影ノイズは $\sigma_y(\tau) = 1/(\pi Q\sqrt{N})\sqrt{T_c/\tau}$（$Q = 2\nu_0 T$）を与え、Example 1 のモンテカルロ検証は $1/(T\sqrt{N})$ の項を $0.2\%$ で再現します。
  * 光時計は $Q$ だけで勝ちます。$\sqrt{N}$ で100倍不利でありながら、1秒で $5.244\times10^{-18}$ 対 $3.463\times10^{-14}$ です。

**2\. Allanの傾きがノイズの名前を教える**

  * 白色・フリッカー・ランダムウォーク周波数ノイズとドリフトについて、測定された傾きは $-0.4995$、$-0.0016$、$+0.4995$、$+1.0000$。
  * 3つの項は二乗和で加わり、その結果の浴槽曲線の最小値 — Example 2 では $\tau = 4379$ s での $3.2428\times10^{-15}$ — が助けになる最長の平均化時間です。
  * フリッカー床は、このシリーズの他の場所でトランズモンやSQUIDを律速する $1/f$ ノイズと同じ2準位揺動子の起源をもちます。

**3\. 安定度と正確度は別の量**

  * 系統バジェットは各シフトの大きさではなく*不確かさ*を抑えます。Example 3 の例示バジェットは合計 $7.7\times10^{-17}$ です。
  * ストロンチウムの行は $4.6$ ms の平均化でその床に達するので、光時計の困難は本質的にすべて系統誤差です。
  * 黒体放射は $T^4$ でスケールし不確かさは $T^3$ なので、100 Kの囲いはその項に27倍、合計に3.3倍の価値があります。

**4\. 光パルス干渉計は $T^2$ を感度に変える**

  * $\Delta\varphi = k_\mathrm{eff} a T^2$ なので $\delta a \propto T^{-2}N^{-1/2}$ で、測定された傾きは厳密に $-2.0000$。
  * $T$ の代価は高さです。$T = 1$ s は $4.90$ m の噴水を意味し、だから長基線の装置は塔や立坑になります。
  * 同じ装置がCoriolis項を通して回転を、勾配を通して質量分布を測り、縞の周期性 — $T = 100$ ms で $3.901\times10^{-5}$ m s$^{-2}$ — がダイナミックレンジを抑えます。

**5\. 蒸気セルは室温を買い、帯域と寸法で支払う**

  * 光ポンピングが熱的偏極を置き換え、3 nTではZeemanエネルギーは $k_BT$ の $1.6\times10^{-12}$ — 無関係です。偏極のどれ一つも $k_BT$ から来ていないからです。
  * 磁場勾配はGauss型の $T_2^\ast$ 減衰（$200$ ms の $T_2$ に対して $2.409$ ms）を生み、1.4節とまったく同様に減衰の形から診断できます。
  * SERFはスピン交換を $(\omega_0/R_\mathrm{se})^2$ で抑制し、感度の $16.5$ 倍を帯域の $274$ 倍で買います。密度を上げてもスピン破壊の漸近値 $0.114$ fT$/\sqrt{\mathrm{Hz}}$ で飽和します。
  * 感度はセル寸法で $a^{-3/2}$ とスケールするので、空間分解能1桁は磁場感度 $31.6$ 倍の代価です — その判定は信号源の幾何に依存します。

**実務上の含意**

  * 平均化時間と帯域なしに感度を引用しないこと。3つは1つの言明です。
  * より長い平均化を設計する前に、系統の床が到来する時刻を計算すること。
  * 減衰が指数関数ではなくGauss型なら、不均一を疑い、センサを改善する前に磁場をシミングすること。
  * 磁力計はまず距離と信号源の幾何で、次に $\eta_B$ で選ぶこと。

### 次章へ

本章とその前の3章のあらゆる感度は $1/\sqrt{N}$ の因子を含み、そのどれもがプローブが独立であることを仮定していました。第5章はその仮定を外します。エンタングルメントは原理的には $1/\sqrt{N}$ を $1/N$ に置き換えられ、第5章はそのどれだけがデコヒーレンスとの接触を生き延びるのか、準備に何がかかるのか、実際にどこで既に本質的な役割を担っているのか、そしてセンサの出力を数値ではなく量子データとして扱うとどう見えるのかを、どちらの方向にもおもねらず数値で詰めます。

[← 第3章: SQUID](<chapter-3.html>) [第5章: 量子限界を超える →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 本章に引用した時計遷移周波数は原子定数ですが、観測時間・原子数・サイクル時間・シフト係数・緩和レート・感度は教育目的で選んだ桁のオーダーの例示値であり、装置の仕様ではありません。設計や提案書に用いる前に一次資料で確認してください。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
