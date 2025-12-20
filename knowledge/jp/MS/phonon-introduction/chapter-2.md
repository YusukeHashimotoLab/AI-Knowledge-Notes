---
title: "第2章：フォノン分散関係"
chapter_title: "第2章：フォノン分散関係"
subtitle: "波数と振動数の関係、および音響・光学モード"
reading_time: "約90分"
difficulty: "入門〜中級"
topics: "分散関係・Brillouin帯・音響/光学モード"
---

[🌐 EN](../../../en/MS/phonon-introduction/chapter-2.md) | 🇯🇵 JP | Last sync: 2025-12-20

[材料科学道場](../index.html) > [フォノン入門](index.md) > 第2章

# 第2章：フォノン分散関係

波数と振動数の関係、および音響・光学モード

## 学習目標

- フォノン分散関係の物理的意味を理解する
- 音響モードと光学モードの違いを説明できる
- Brillouin帯と逆格子の概念を把握する
- 二原子基底格子における分散関係を導出できる
- 群速度と位相速度の違いを理解する
- 実際の材料のフォノンスペクトルを解釈できる

## 2.1 はじめに

前章では、一次元単原子格子の分散関係 \\(\omega(k) = 2\sqrt{K/m}|\sin(ka/2)|\\) を導出しました。本章では、この分散関係をより詳しく調べ、実際の三次元結晶における複雑なフォノンスペクトルへと発展させていきます。

分散関係は、フォノンの**波数 \\(k\\)** と**角振動数 \\(\omega\\)** の関係を表すもので、材料の動的性質を決定する最も基本的な情報です。これを理解することで、熱伝導、比熱、中性子散乱実験など、多くの物理現象を定量的に扱えるようになります。

## 2.2 単原子格子の分散関係の詳細

### 2.2.1 第一Brillouin帯

一次元格子の分散関係を再確認しましょう：

\\[
\omega(k) = 2\sqrt{\frac{K}{m}} \left|\sin\left(\frac{ka}{2}\right)\right|
\\]

この式には重要な周期性があります：

\\[
\omega(k + \frac{2\pi n}{a}) = \omega(k), \quad n = 0, \pm 1, \pm 2, \ldots
\\]

つまり、\\(\omega(k)\\) は \\(k\\) の周期関数であり、周期は \\(2\pi/a\\) です。そのため、物理的に独立な情報は、**第一Brillouin帯**（First Brillouin Zone）と呼ばれる範囲：

\\[
-\frac{\pi}{a} \leq k \leq \frac{\pi}{a}
\\]

に含まれています。この範囲外の \\(k\\) は、Brillouin帯内の \\(k\\) に換算できるため、冗長です。

**Brillouin帯の物理的意味**

Brillouin帯の境界 \\(k = \pm\pi/a\\) は、波長 \\(\lambda = 2\pi/k = 2a\\) に対応します。つまり、格子定数の2倍の波長を持つ波が、最も短い波長の意味のある格子振動となります。これより短い波長の波は、離散的な原子配列では物理的に区別できません。

### 2.2.2 長波長極限：音波

\\(k \to 0\\) の長波長極限では、\\(\sin(ka/2) \approx ka/2\\) なので、

\\[
\omega(k) \approx 2\sqrt{\frac{K}{m}} \cdot \frac{ka}{2} = \sqrt{Ka^2/m} \cdot k \equiv v_s k
\\]

ここで、

\\[
v_s = a\sqrt{\frac{K}{m}}
\\]

は**音速**（sound velocity）です。長波長極限では、分散関係が線形になり、格子振動は連続的な弾性波（音波）と同じ振る舞いをします。

**例：アルミニウムの音速**

アルミニウム（Al）の場合、実験的に測定された縦波音速は約6400 m/s、横波音速は約3100 m/s です。これらの値から、バネ定数 \\(K\\) の等価的な値を推定することができます。

### 2.2.3 群速度と位相速度

**位相速度**（phase velocity）\\(v_p\\) は、波の山（位相）が移動する速度です：

\\[
v_p = \frac{\omega}{k}
\\]

一方、**群速度**（group velocity）\\(v_g\\) は、波束（エネルギー・情報）が伝播する速度です：

\\[
v_g = \frac{d\omega}{dk}
\\]

単原子格子の場合、

\\[
v_g = \frac{d\omega}{dk} = a\sqrt{\frac{K}{m}} \cos\left(\frac{ka}{2}\right)
\\]

となります。

**群速度の振る舞い**

- \\(k = 0\\)：\\(v_g = a\sqrt{K/m} = v_s\\)（最大）
- \\(k = \pi/a\\)（Brillouin帯端）：\\(v_g = 0\\)

Brillouin帯端で群速度がゼロになることは、**Bragg反射**と関係しています。波長 \\(\lambda = 2a\\) の波は、格子による散乱が強く、前進波と後進波が重ね合わさって定在波となり、エネルギーが伝播しないのです。

### 2.2.4 状態密度

フォノンの**状態密度**（density of states, DOS）\\(g(\omega)\\) は、単位振動数あたりのフォノンモード数を表します。これは熱的性質の計算に不可欠です。

一次元単原子格子の場合、\\(k\\) と \\(\omega\\) の対応が一対一でないため、\\(g(\omega)\\) は一様ではありません。分散関係から、

\\[
g(\omega) = \frac{2}{\pi a} \frac{1}{|d\omega/dk|}
\\]

となります。\\(d\omega/dk = v_g\\) なので、群速度が小さい（分散関係が平坦な）領域で状態密度が大きくなります。特に、Brillouin帯端（\\(k = \pm\pi/a\\)）では、\\(v_g \to 0\\) なので、\\(g(\omega)\\) は**Van Hove特異点**と呼ばれる発散的なピークを示します。

## 2.3 二原子基底格子：音響モードと光学モード

### 2.3.1 二原子基底格子のモデル

実際の多くの結晶（NaCl, Si, GaAsなど）は、単位格子内に複数の原子を含みます。最も単純な例として、**一次元二原子格子**を考えましょう。

**モデル設定**

- 格子定数：\\(a\\)
- 単位格子内に質量 \\(m_1\\) と \\(m_2\\) の2つの原子
- 原子間のバネ定数：\\(K\\)（簡単のため、すべて同じとする）

\\(n\\) 番目の単位格子における2つの原子の変位を \\(u_n\\)（質量 \\(m_1\\)）と \\(v_n\\)（質量 \\(m_2\\)）とします。運動方程式は：

\\[
m_1 \frac{d^2u_n}{dt^2} = K(v_n + v_{n-1} - 2u_n)
\\]

\\[
m_2 \frac{d^2v_n}{dt^2} = K(u_{n+1} + u_n - 2v_n)
\\]

### 2.3.2 分散関係の導出

波動解を仮定します：

\\[
u_n = Ae^{i(kna - \omega t)}, \quad v_n = Be^{i(kna - \omega t)}
\\]

これを運動方程式に代入して整理すると、\\(A\\) と \\(B\\) に関する連立方程式が得られます：

\\[
\begin{pmatrix}
2K - m_1\omega^2 & -K(1 + e^{-ika}) \\
-K(1 + e^{ika}) & 2K - m_2\omega^2
\end{pmatrix}
\begin{pmatrix}
A \\
B
\end{pmatrix}
=
\begin{pmatrix}
0 \\
0
\end{pmatrix}
\\]

非自明な解が存在する条件（行列式がゼロ）から、分散関係が求まります：

**二原子格子の分散関係**

\\[
\omega_{\pm}^2(k) = K\left(\frac{1}{m_1} + \frac{1}{m_2}\right) \pm K\sqrt{\left(\frac{1}{m_1} + \frac{1}{m_2}\right)^2 - \frac{4\sin^2(ka/2)}{m_1m_2}}
\\]

ここで、\\(\omega_+\\) を**光学モード**（optical mode）、\\(\omega_-\\) を**音響モード**（acoustic mode）と呼びます。

### 2.3.3 音響モードと光学モードの性質

**音響モード（\\(\omega_-\\)）**

- \\(k = 0\\) で \\(\omega_- = 0\\)（ゼロギャップ）
- 長波長極限で \\(\omega_- \approx v_s k\\)（線形分散、音波）
- 振動パターン：単位格子内の2つの原子が**同位相**で振動（重心運動）

**光学モード（\\(\omega_+\\)）**

- \\(k = 0\\) で有限の振動数 \\(\omega_+ = \sqrt{2K(1/m_1 + 1/m_2)}\\)（ギャップあり）
- 振動パターン：単位格子内の2つの原子が**逆位相**で振動（内部振動）
- 名前の由来：イオン結晶では、この振動が電磁波（光）と強く結合し、赤外吸収を起こす

**振幅比**

音響モードと光学モードでは、2つの原子の振幅比 \\(A/B\\) が異なります：

- **音響モード**（\\(k = 0\\)）：\\(A/B \approx 1\\)（同位相、重心運動）
- **光学モード**（\\(k = 0\\)）：\\(A/B \approx -m_2/m_1\\)（逆位相、重心静止）

### 2.3.4 極限的な場合の考察

**\\(m_1 = m_2 = m\\) の場合**

質量が等しい場合、光学モードと音響モードの区別が曖昧になります。実際、この極限では：

- 音響モード：\\(\omega_- = 2\sqrt{K/m}|\sin(ka/4)|\\)
- 光学モード：\\(\omega_+ = 2\sqrt{K/m}|\cos(ka/4)|\\)

となり、これらは単原子格子を2倍の周期（\\(2a\\)）を持つと見たときの2つのブランチに対応します。

**\\(m_1 \ll m_2\\) の場合**

軽い原子と重い原子の組み合わせ（例：HCl分子）では、

- 音響モード：主に重い原子 \\(m_2\\) が振動
- 光学モード：主に軽い原子 \\(m_1\\) が振動（\\(\omega_+ \approx \sqrt{2K/m_1}\\)）

となります。

## 2.4 三次元結晶のフォノン

### 2.4.1 三次元への拡張

実際の結晶は三次元です。一次元の議論を三次元に拡張すると、以下のような特徴が現れます：

- **波数ベクトル**：\\(\mathbf{k} = (k_x, k_y, k_z)\\) は三次元ベクトル
- **第一Brillouin帯**：逆格子空間における原点周りの多面体領域
- **偏極**：各 \\(\mathbf{k}\\) に対して、3つの独立な偏極方向（1つの縦波、2つの横波）

単位格子に \\(r\\) 個の原子がある場合、\\(\mathbf{k}\\) ごとに \\(3r\\) 本のフォノンブランチが存在します：

- **音響ブランチ**：3本（1本の縦波LA、2本の横波TA）
- **光学ブランチ**：\\(3(r-1)\\) 本（\\((r-1)\\) 本の縦波LO、\\(2(r-1)\\) 本の横波TO）

### 2.4.2 対称性と高対称点

三次元結晶のフォノン分散は、結晶の対称性を反映します。Brillouin帯内の特別な点（**高対称点**）では、対称性により縮退や選択則が生じます。

**立方晶系（fcc, bcc, ダイヤモンド構造）の高対称点**

- \\(\Gamma\\) 点：\\(\mathbf{k} = (0, 0, 0)\\)（Brillouin帯中心）
- \\(X\\) 点：\\(\mathbf{k} = (2\pi/a)(1, 0, 0)\\)（Brillouin帯端の一つ）
- \\(L\\) 点：\\(\mathbf{k} = (2\pi/a)(1/2, 1/2, 1/2)\\)（対角方向）
- \\(K\\) 点（または \\(W\\) 点）：その他の高対称点

フォノン分散は通常、これらの高対称点を結ぶ経路に沿ってプロットされます：例えば \\(L \to \Gamma \to X \to W \to K \to \Gamma\\)。

### 2.4.3 典型的な材料のフォノンスペクトル

**シリコン（Si）**

- ダイヤモンド構造（fcc格子 + 2原子基底）
- 6本のブランチ：3本の音響（LA, TA, TA）+ 3本の光学（LO, TO, TO）
- \\(\Gamma\\) 点でのLO/TOモード：約15.5 THz（520 cm⁻¹）
- 特徴：高いDebye温度（645 K）、強い共有結合

**塩化ナトリウム（NaCl）**

- 岩塩構造（fcc格子 + 2原子基底）
- 6本のブランチ
- \\(\Gamma\\) 点でLO-TO分裂が大きい（約100 cm⁻¹）
- 原因：イオン性による長距離クーロン相互作用

**ガリウム砒素（GaAs）**

- 閃亜鉛鉱構造（fcc格子 + 2原子基底）
- \\(\Gamma\\) 点でのLO-TO分裂：約24 cm⁻¹
- 部分的なイオン性により、Siより大きいがNaClより小さい

## 2.5 逆格子とBrillouin帯

### 2.5.1 逆格子の定義

実空間の格子ベクトル \\(\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\\) に対して、**逆格子ベクトル** \\(\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3\\) を次のように定義します：

\\[
\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi \delta_{ij}
\\]

具体的には、

\\[
\mathbf{b}_1 = 2\pi \frac{\mathbf{a}_2 \times \mathbf{a}_3}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}, \quad \text{etc.}
\\]

逆格子ベクトルの整数線形結合で表されるベクトルを**逆格子ベクトル** \\(\mathbf{G}\\) といいます：

\\[
\mathbf{G} = h\mathbf{b}_1 + k\mathbf{b}_2 + l\mathbf{b}_3, \quad h, k, l \in \mathbb{Z}
\\]

### 2.5.2 第一Brillouin帯の定義

**第一Brillouin帯**は、逆格子空間において、原点により近い逆格子点がない領域として定義されます。これは、原点から各逆格子点への中点を通る垂直二等分面で囲まれた多面体領域です（Wigner-Seitz セルに相当）。

**物理的意味**

Brillouin帯は、波数空間における周期性を表します。波数 \\(\mathbf{k}\\) と \\(\mathbf{k} + \mathbf{G}\\)（\\(\mathbf{G}\\) は逆格子ベクトル）は、物理的に同等なフォノンを表します。したがって、すべての物理的に独立な情報は第一Brillouin帯内に含まれます。

### 2.5.3 fcc格子の逆格子とBrillouin帯

**面心立方（fcc）格子**

実空間の格子ベクトル：

\\[
\mathbf{a}_1 = \frac{a}{2}(0, 1, 1), \quad \mathbf{a}_2 = \frac{a}{2}(1, 0, 1), \quad \mathbf{a}_3 = \frac{a}{2}(1, 1, 0)
\\]

この逆格子は**体心立方（bcc）格子**になります：

\\[
\mathbf{b}_1 = \frac{2\pi}{a}(-1, 1, 1), \quad \mathbf{b}_2 = \frac{2\pi}{a}(1, -1, 1), \quad \mathbf{b}_3 = \frac{2\pi}{a}(1, 1, -1)
\\]

第一Brillouin帯は、切頂八面体（truncated octahedron）の形をしています。

## 2.6 Pythonによる分散関係の可視化

**二原子格子の分散関係プロット**

```python
import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
K = 1.0       # バネ定数
m1 = 1.0      # 軽い原子の質量
m2 = 2.0      # 重い原子の質量
a = 1.0       # 格子定数

# 波数範囲（第一Brillouin帯）
k = np.linspace(-np.pi/a, np.pi/a, 500)

# 分散関係の計算
def dispersion_diatomic(k, K, m1, m2):
    """二原子格子の分散関係"""
    term1 = K * (1/m1 + 1/m2)
    term2_sq = (1/m1 + 1/m2)**2 - 4*np.sin(k*a/2)**2 / (m1*m2)
    term2_sq = np.maximum(term2_sq, 0)  # 負の値を避ける

    omega_plus = np.sqrt(term1 + K*np.sqrt(term2_sq))    # 光学モード
    omega_minus = np.sqrt(term1 - K*np.sqrt(term2_sq))   # 音響モード

    return omega_minus, omega_plus

omega_acoustic, omega_optical = dispersion_diatomic(k, K, m1, m2)

# プロット
plt.figure(figsize=(10, 7))

plt.plot(k, omega_acoustic, 'b-', linewidth=2, label='音響モード（Acoustic）')
plt.plot(k, omega_optical, 'r-', linewidth=2, label='光学モード（Optical）')

# バンドギャップ
gap_bottom = omega_acoustic[len(k)//2]
gap_top = omega_optical[len(k)//2]
plt.axhspan(gap_bottom, gap_top, alpha=0.2, color='yellow', label='バンドギャップ')

# Γ点（k=0）での光学モードの振動数
omega_optical_gamma = np.sqrt(2*K*(1/m1 + 1/m2))
plt.axhline(y=omega_optical_gamma, color='red', linestyle='--', alpha=0.5)
plt.text(0.05, omega_optical_gamma+0.05, r'$\omega_+ (k=0)$', fontsize=11)

# 軸設定
plt.xlabel(r'波数 $k$ [1/a]', fontsize=14)
plt.ylabel(r'角振動数 $\omega$ [rad/s]', fontsize=14)
plt.title(f'二原子格子の分散関係（$m_2/m_1 = {m2/m1}$）', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/a, np.pi/a)
plt.ylim(0, max(omega_optical)*1.1)

# 高対称点の表示
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=-np.pi/a, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=np.pi/a, color='gray', linestyle=':', alpha=0.5)
plt.text(0, -0.15, r'$\Gamma$', fontsize=12, ha='center')
plt.text(-np.pi/a, -0.15, r'$-\pi/a$', fontsize=10, ha='center')
plt.text(np.pi/a, -0.15, r'$\pi/a$', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('diatomic_dispersion.png', dpi=150)
plt.show()
```

**群速度の計算**

```python
# 群速度の計算
dk = k[1] - k[0]
vg_acoustic = np.gradient(omega_acoustic, dk)
vg_optical = np.gradient(omega_optical, dk)

plt.figure(figsize=(10, 6))
plt.plot(k, vg_acoustic, 'b-', linewidth=2, label='音響モード')
plt.plot(k, vg_optical, 'r-', linewidth=2, label='光学モード')

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel(r'波数 $k$ [1/a]', fontsize=14)
plt.ylabel(r'群速度 $v_g = d\omega/dk$ [m/s]', fontsize=14)
plt.title('二原子格子の群速度', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diatomic_group_velocity.png', dpi=150)
plt.show()

# k=0での音速を計算
idx_zero = len(k) // 2
v_sound = vg_acoustic[idx_zero]
print(f"音速（k=0での群速度）: {v_sound:.3f} [単位系依存]")
```

**状態密度の計算**

```python
from scipy.interpolate import interp1d

# 振動数範囲を設定
omega_range = np.linspace(0, max(omega_optical), 1000)

# 状態密度を計算（簡略版：ヒストグラム法）
dos_acoustic, bins_a = np.histogram(omega_acoustic, bins=omega_range, density=True)
dos_optical, bins_o = np.histogram(omega_optical, bins=omega_range, density=True)

omega_bins = (omega_range[:-1] + omega_range[1:]) / 2

plt.figure(figsize=(10, 6))
plt.plot(omega_bins, dos_acoustic * len(k) / np.pi, 'b-', linewidth=2, label='音響モード')
plt.plot(omega_bins, dos_optical * len(k) / np.pi, 'r-', linewidth=2, label='光学モード')

plt.xlabel(r'角振動数 $\omega$ [rad/s]', fontsize=14)
plt.ylabel(r'状態密度 $g(\omega)$ [任意単位]', fontsize=14)
plt.title('二原子格子の状態密度', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, max(omega_optical)*1.05)
plt.tight_layout()
plt.savefig('diatomic_dos.png', dpi=150)
plt.show()
```

**コードの解説**

- **分散関係プロット**：音響モード（青）と光学モード（赤）を同時に表示。\\(k = 0\\) でのバンドギャップ（黄色の領域）が明確に見えます。

- **群速度**：音響モードは \\(k = 0\\) で最大、光学モードは全体的に小さい（ほとんどゼロに近い）ことがわかります。

- **状態密度**：Van Hove特異点（バンド端での発散）が見られます。これは熱的性質の計算で重要です。

## 2.7 実験との対応

### 2.7.1 中性子散乱実験

フォノン分散関係は、**非弾性中性子散乱**実験により直接測定できます。中性子は原子核と相互作用し、フォノンを生成・消滅させながら散乱されます。

**エネルギー・運動量保存則**

入射中性子（波数 \\(\mathbf{k}_i\\)、エネルギー \\(E_i\\)）がフォノン（波数 \\(\mathbf{q}\\)、エネルギー \\(\hbar\omega\\)）を生成して散乱中性子（波数 \\(\mathbf{k}_f\\)、エネルギー \\(E_f\\)）になる場合：

- **エネルギー保存**：\\(E_i = E_f + \hbar\omega\\)
- **運動量保存**：\\(\mathbf{k}_i = \mathbf{k}_f + \mathbf{q} + \mathbf{G}\\)（\\(\mathbf{G}\\) は逆格子ベクトル）

これらの式から、散乱角とエネルギー損失を測定することで、\\(\omega(\mathbf{q})\\) を決定できます。

### 2.7.2 ラマン散乱とブリルアン散乱

**ラマン散乱**

可視光レーザーを試料に照射し、散乱光の振動数シフトを測定します。主に \\(\mathbf{q} \approx 0\\)（Brillouin帯中心付近）の光学フォノンを観測します。シリコンのラマンスペクトルでは、520 cm⁻¹ に鋭いピークが現れ、これが \\(\Gamma\\) 点のLO/TOモードに対応します。

**ブリルアン散乱**

ラマン散乱と同様ですが、より小さな振動数シフト（GHz帯）を測定し、音響フォノンを観測します。音速や弾性定数の測定に用いられます。

### 2.7.3 赤外・テラヘルツ分光

赤外・テラヘルツ光を用いて、\\(\mathbf{q} \approx 0\\) の光学フォノンの吸収を測定します。特にイオン結晶では、LO-TO分裂を直接観測できます。

## 2.8 まとめ

**本章の要点**

- フォノン分散関係 \\(\omega(\mathbf{k})\\) は、波数と振動数の関係を表し、材料の動的性質を決定する

- 第一Brillouin帯内の情報で、すべての物理的に独立なフォノンモードが記述される

- 単位格子に \\(r\\) 個の原子がある場合、\\(3r\\) 本のフォノンブランチが存在する

- **音響モード**（3本）：長波長極限で音波（\\(\omega \propto k\\)）、単位格子内の原子が同位相で振動

- **光学モード**（\\(3(r-1)\\) 本）：\\(k = 0\\) で有限の振動数、単位格子内の原子が逆位相で振動

- 群速度 \\(v_g = d\omega/dk\\) はエネルギー伝播速度であり、Brillouin帯端でゼロになる

- 状態密度 \\(g(\omega)\\) はVan Hove特異点を持ち、比熱などの熱的性質の計算に重要

- 実験（中性子散乱、ラマン分光など）により、フォノン分散関係を直接測定できる

## 演習問題

### 問題1：分散関係の理解

一次元二原子格子において、\\(m_1 = m\\)、\\(m_2 = 4m\\)、\\(K = 1\\) N/m、\\(a = 1\\) Åとする。

1. \\(k = 0\\) での光学モードの角振動数 \\(\omega_+(0)\\) を計算しなさい。
2. \\(k = 0\\) での音響モードの群速度（音速）を求めなさい。

### 問題2：群速度とエネルギー輸送

単原子格子の分散関係 \\(\omega(k) = \omega_{\max}|\sin(ka/2)|\\) において、

1. 群速度が最大となる \\(k\\) の値を求めなさい。
2. Brillouin帯端（\\(k = \pi/a\\)）で群速度がゼロになることの物理的意味を説明しなさい。

### 問題3：状態密度（発展）

一次元単原子格子の状態密度 \\(g(\omega)\\) を、\\(\omega(k) = \omega_{\max}|\sin(ka/2)|\\) から導出しなさい。特に、\\(\omega = \omega_{\max}\\) 近傍での \\(g(\omega)\\) の振る舞いを調べなさい。

### 問題4：光学モードと音響モードの比較

以下の表を完成させなさい：

| 性質 | 音響モード | 光学モード |
|------|------------|------------|
| \\(k = 0\\) での \\(\omega\\) | ? | ? |
| 単位格子内の振動パターン | ? | ? |
| 長波長極限での分散関係 | ? | ? |
| 典型的な振動数範囲（Si） | ? | ? |

### 問題5：実験との対応

シリコン（Si）のラマンスペクトルで520 cm⁻¹にピークが観測される。

1. これは \\(\Gamma\\) 点（\\(k = 0\\)）のどのモードに対応するか？
2. この振動数を角振動数 \\(\omega\\)（rad/s）およびエネルギー \\(\hbar\omega\\)（meV）に換算しなさい。
   （\\(1\\) cm⁻¹ \\(= 2\pi c \times 10^2\\) rad/s、\\(c = 3 \times 10^8\\) m/s、\\(\hbar = 6.58 \times 10^{-16}\\) eV·s）

---

[← 前章：フォノンとは何か？](chapter-1.md) | [次章：フォノン状態密度 →](chapter-3.md)

---

## 免責事項

この教育コンテンツは、橋本研究室のナレッジベース用にAIの支援を受けて作成されました。内容の正確性には十分注意を払っていますが、学術的な引用や重要な意思決定には、査読付き論文や標準的な教科書を参照することをお勧めします。
