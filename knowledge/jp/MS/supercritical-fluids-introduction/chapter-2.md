---
title: "第2章: 超臨界流体の熱力学"
subtitle: "状態方程式、臨界現象、および相平衡"
description: "超臨界流体の熱力学的基礎：状態方程式、臨界現象、相平衡を体系的に解説"
keywords: "超臨界流体, 熱力学, 状態方程式, van der Waals方程式, Peng-Robinson方程式, 臨界現象, 相平衡, 材料科学"
author: "AI_Homepage"
date: "2025-12-25"
version: "1.0"
category: "材料科学"
tags: ["超臨界流体", "熱力学", "状態方程式", "相平衡"]
series: "超臨界流体入門"
series_order: 2
prev_article: "./chapter-1.md"
next_article: "./chapter-3.md"
language: "ja"
---

# 第2章: 超臨界流体の熱力学

## 学習目標

本章を読むことで、以下を習得できます：

1. **状態方程式の理解**: 理想気体から実在気体への拡張、van der WaalsおよびPeng-Robinson方程式の物理的意味と応用
2. **臨界現象の把握**: 臨界点近傍での異常な物性変化とその物理的起源
3. **相平衡の計算**: 超臨界流体を含む系の相図と溶解度予測
4. **熱力学計算の実践**: フガシティ、化学ポテンシャル、混合則を用いた実用計算

---

## 2.1 超臨界流体の状態方程式

### 理想気体の法則の限界

理想気体の状態方程式は以下の形で表されます：

$$
PV = nRT
$$

ここで、$P$ は圧力、$V$ は体積、$n$ は物質量、$R$ は気体定数（8.314 J/(mol·K)）、$T$ は絶対温度です。

しかし、理想気体の法則は以下の仮定に基づいています：

- **分子間相互作用がない**（引力も斥力もない）
- **分子自身の体積が無視できる**（点粒子）

実在の気体、特に**高圧・高密度条件**や**臨界点近傍**では、これらの仮定が破綻します：

1. **高圧下**: 分子間距離が小さくなり、分子間引力（van der Waals力）が無視できない
2. **高密度**: 分子自身の体積（排除体積）が全体積に対して無視できない
3. **臨界点近傍**: 気液の区別がなくなり、密度ゆらぎが大きい

### van der Waals方程式

van der Waals（1873年）は、実在気体の振る舞いを記述するため、理想気体の式に2つの補正を加えました：

$$
\left(P + \frac{a}{V_m^2}\right)(V_m - b) = RT
$$

ここで、$V_m$ はモル体積、$a$ と $b$ は物質固有の定数です。

**物理的意味**：

- **$a$ パラメータ**（分子間引力の補正）:
  - 次元: Pa·m⁶/mol²
  - 分子間の引力により、実際の圧力は理想気体より小さくなる
  - $a$ が大きいほど分子間引力が強い（極性分子や大きな分子）

- **$b$ パラメータ**（排除体積の補正）:
  - 次元: m³/mol
  - 分子自身が占める体積により、自由に動ける空間が減少
  - $b$ は分子の大きさに比例（分子の体積の約4倍）

**臨界定数との関係**：

van der Waals方程式の臨界点では、以下の関係が成り立ちます：

$$
\begin{aligned}
a &= \frac{27R^2T_c^2}{64P_c} \\
b &= \frac{RT_c}{8P_c}
\end{aligned}
$$

**van der Waals等温線**：

van der Waals方程式を $P$ について解くと：

$$
P = \frac{RT}{V_m - b} - \frac{a}{V_m^2}
$$

この式は温度によって異なる等温線（$P$-$V$ 曲線）を与えます：

- **$T > T_c$（超臨界領域）**: 単調減少曲線（相転移なし）
- **$T = T_c$（臨界等温線）**: 変曲点を持つ曲線
- **$T < T_c$（二相領域）**: S字型の曲線（Maxwell構成則により水平線で置き換え）

### Peng-Robinson方程式

Peng-Robinson方程式（1976年）は、炭化水素や超臨界流体に対してより高精度な状態方程式です：

$$
P = \frac{RT}{V_m - b} - \frac{a\alpha(T)}{V_m^2 + 2bV_m - b^2}
$$

ここで：

$$
\begin{aligned}
a &= 0.45724\frac{R^2T_c^2}{P_c} \\
b &= 0.07780\frac{RT_c}{P_c} \\
\alpha(T) &= \left[1 + \kappa\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2 \\
\kappa &= 0.37464 + 1.54226\omega - 0.26992\omega^2
\end{aligned}
$$

**偏心因子 $\omega$**：

偏心因子は分子の非球形性を表すパラメータです：

$$
\omega = -\log_{10}\left(\frac{P_r^{\text{sat}}}{P_c}\right)\bigg|_{T_r=0.7} - 1.0
$$

ここで、$P_r^{\text{sat}}$ は換算温度 $T_r = 0.7$ における飽和蒸気圧です。

**物質別の偏心因子**：

| 物質 | $\omega$ | 特徴 |
|------|----------|------|
| Ar, Kr, Xe | 0.00 ~ 0.01 | 球形の希ガス |
| CH₄ | 0.011 | ほぼ球形 |
| CO₂ | 0.228 | 直線分子 |
| H₂O | 0.344 | 極性、水素結合 |
| n-ヘキサン | 0.301 | 鎖状分子 |

### 各流体に対するEOS精度の比較

以下の表は、CO₂を例とした各状態方程式の予測精度を示します：

| 状態方程式 | 密度誤差（%） | 圧縮率誤差（%） | 計算コスト |
|-----------|--------------|----------------|-----------|
| 理想気体 | > 50 | > 100 | 非常に低 |
| van der Waals | 10-20 | 20-30 | 低 |
| Peng-Robinson | 2-5 | 5-10 | 中 |
| SAFT系 | < 1 | < 2 | 高 |

**適用指針**：

- **理想気体**: 低圧気体のみ（$P < 1$ MPa）
- **van der Waals**: 定性的理解、教育目的
- **Peng-Robinson**: 工業プロセス設計、炭化水素系
- **SAFT系**: 高精度が必要な研究、複雑な分子系

---

## 2.2 臨界現象

### 臨界乳光（Critical Opalescence）

臨界点に近づくと、透明だった流体が**乳白色に濁る**現象が観察されます。これは**臨界乳光**と呼ばれます。

**物理的起源**：

- 臨界点では密度ゆらぎの相関長 $\xi$ が発散：$\xi \propto |T - T_c|^{-\nu}$（$\nu \approx 0.63$）
- ゆらぎのスケールが可視光の波長（400-700 nm）と同程度になると、強い光散乱が起こる
- Rayleigh散乱強度 $I \propto \xi^6$

### 圧縮率と熱容量の発散

臨界点近傍では、以下の熱力学量が異常に大きくなります：

**等温圧縮率 $\kappa_T$**：

$$
\kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T \propto |T - T_c|^{-\gamma}
$$

ここで、$\gamma \approx 1.24$ は臨界指数です。

**定圧熱容量 $C_P$**：

$$
C_P = T\left(\frac{\partial S}{\partial T}\right)_P \propto |T - T_c|^{-\alpha}
$$

ここで、$\alpha \approx 0.11$ です。

**物理的意味**：

- $\kappa_T$ の発散: わずかな圧力変化で大きく体積が変化（ピストン効果）
- $C_P$ の発散: わずかな温度変化で大きく熱を吸収（温度制御の困難）

### 普遍性と臨界指数

驚くべきことに、**異なる物質の臨界現象は同じ臨界指数で記述**されます。これを**普遍性**と呼びます。

**主要な臨界指数**：

| 記号 | 定義 | 3次元Ising模型 |
|------|------|---------------|
| $\alpha$ | 熱容量 | 0.110 |
| $\beta$ | 秩序変数（密度差） | 0.326 |
| $\gamma$ | 感受率（圧縮率） | 1.237 |
| $\delta$ | 臨界等温線 | 4.789 |
| $\nu$ | 相関長 | 0.630 |

**秩序変数（密度差）**：

$$
\rho_L - \rho_G \propto |T - T_c|^\beta
$$

ここで、$\rho_L$ は液相密度、$\rho_G$ は気相密度です。

### くりこみ群理論の概説

普遍性の起源は**くりこみ群理論**（Renormalization Group Theory）によって説明されます。

**基本的なアイデア**：

1. 臨界点では、系のスケール変換に対する不変性が現れる
2. 短距離の詳細（分子の種類など）は、長距離の振る舞いに影響しない
3. 空間次元と秩序変数の次元のみが臨界指数を決定

**実用的意味**：

- 1つの物質（例：CO₂）の臨界挙動を調べれば、他の物質（H₂O、エタノールなど）の臨界挙動も予測可能
- 対応状態の原理の理論的基礎

---

## 2.3 臨界点近傍の熱力学的性質

### エンタルピーとエントロピーの挙動

**エンタルピー $H$**：

臨界点では、気化エンタルピー（蒸発潜熱）がゼロに収束します：

$$
\Delta H_{\text{vap}} = H_G - H_L \propto |T - T_c|^\beta \to 0
$$

これは、気液の区別がなくなることを意味します。

**エントロピー $S$**：

同様に、気化エントロピーもゼロに収束します：

$$
\Delta S_{\text{vap}} = S_G - S_L \to 0
$$

Clausius-Clapeyron式との整合性：

$$
\frac{dP}{dT} = \frac{\Delta S_{\text{vap}}}{\Delta V_{\text{vap}}} \to \frac{0}{0} \quad (\text{不定形})
$$

### 熱容量の異常

**定圧熱容量 $C_P$ のピーク**：

臨界点に沿って温度を変化させると、$C_P$ は鋭いピークを示します（発散）。

**定積熱容量 $C_V$**：

$C_V$ も臨界点で異常を示しますが、$C_P$ ほど顕著ではありません：

$$
C_V \propto |T - T_c|^{-\alpha'}, \quad \alpha' \approx 0.11
$$

**実用的影響**：

- 臨界点近傍での加熱・冷却は非常に緩慢（熱容量が大きい）
- 温度制御が困難（オーバーシュートしやすい）

### 臨界点近傍の音速

音速 $c$ は以下の式で与えられます：

$$
c = \sqrt{\frac{1}{\rho \kappa_S}} = \sqrt{\left(\frac{\partial P}{\partial \rho}\right)_S}
$$

ここで、$\kappa_S$ は断熱圧縮率です。

臨界点では $\kappa_T \to \infty$ ですが、$\kappa_S$ は有限のため、**音速は極小値**を取ります。

**CO₂の例**：

- 臨界点: $c \approx 250$ m/s（極小）
- 常圧気体: $c \approx 260$ m/s
- 液体: $c \approx 200-300$ m/s

### プロセス設計への実用的影響

臨界点近傍での操作は以下の課題があります：

1. **圧力制御の困難**：わずかな密度変化で圧力が大きく変動
2. **温度制御の困難**：熱容量が大きく、温度応答が遅い
3. **流量制御の困難**：粘度と密度が急激に変化
4. **相分離の可能性**：わずかな条件変化で二相に分離

**設計指針**：

- 臨界点から適度に離れた条件で操作（$T_r = 1.05-1.2$、$P_r = 1.1-2.0$）
- 十分な安全マージンと制御幅を確保
- 動的シミュレーションによる過渡応答の評価

---

## 2.4 超臨界流体系における相平衡

### 気液平衡（VLE）

純成分の気液平衡は、**Clausius-Clapeyron式**で記述されます：

$$
\frac{dP}{dT} = \frac{\Delta H_{\text{vap}}}{T\Delta V_{\text{vap}}}
$$

理想気体近似と $\Delta H_{\text{vap}}$ が温度に依らないと仮定すると：

$$
\ln P = -\frac{\Delta H_{\text{vap}}}{RT} + C
$$

**Antoine式**（実用的な蒸気圧式）：

$$
\log_{10} P = A - \frac{B}{C + T}
$$

ここで、$A, B, C$ は物質固有の定数です（温度は℃、圧力はmmHg）。

### SCFを含む二成分系相図

超臨界流体と溶質の二成分系では、複雑な相挙動が現れます。

**典型的な相図の特徴**：

1. **臨界曲線（Critical Locus）**：
   - 純成分1の臨界点と純成分2の臨界点を結ぶ曲線
   - 常に単調ではなく、極大や極小を持つことがある

2. **三相平衡線（Three-Phase Line）**：
   - 固体-液体-気体が共存する線
   - 純成分の三重点から延びる

3. **逆行凝縮領域**（後述）

**圧力-組成線図（$P$-$x$ diagram）**：

一定温度での圧力と組成の関係を示します。SCFを含む系では、以下の特徴があります：

- 高圧側で相分離が起こる場合がある（Type I-V分類）
- 臨界点近傍で溶解度が圧力に対して非常に敏感

### 溶解度モデル：Chrastil式

Chrastil（1982年）は、超臨界流体中の固体溶解度を以下の半経験式で表しました：

$$
\ln C = k \ln \rho + \frac{a}{T} + b
$$

ここで：

- $C$: 溶解度（kg溶質/kg SCF）
- $\rho$: SCFの密度（kg/m³）
- $T$: 温度（K）
- $k, a, b$: フィッティングパラメータ

**物理的意味**：

- $k$: 溶質-溶媒間の分子会合数（通常2-8）
- $a$: 溶解エンタルピーに関連（$a \propto -\Delta H_{\text{sol}}/R$）
- $b$: エントロピー項

**応用例**：

CO₂中のカフェイン溶解度は以下のように表されます：

$$
\ln C = 7.5 \ln \rho - \frac{5200}{T} - 15.3
$$

### 逆行凝縮（Retrograde Condensation）

通常、圧力を上げると気体は液化します。しかし、臨界点近傍では**逆の現象**が起こることがあります：

**逆行凝縮**：

- **等温圧縮**により、液体が気化する（通常とは逆）
- **等圧冷却**により、気体が凝縮する（通常と同じ）

**発生条件**：

- 温度が臨界温度より高い（$T > T_c$）
- 圧力が臨界圧力付近（$P \approx P_c$）
- 混合物の臨界曲線が負の勾配を持つ領域

**実用的重要性**：

- 天然ガス処理（コンデンセート回収）
- 超臨界流体抽出プロセスの最適化

---

## 2.5 熱力学計算

### フガシティとフガシティ係数

実在気体の化学ポテンシャルは、**フガシティ** $f$ を用いて表されます：

$$
\mu = \mu^\circ(T) + RT\ln\frac{f}{f^\circ}
$$

理想気体では $f = P$ ですが、実在気体では $f \neq P$ です。

**フガシティ係数 $\phi$**：

$$
\phi = \frac{f}{P}
$$

状態方程式から計算できます：

$$
\ln\phi = \frac{1}{RT}\int_V^\infty\left[P - \frac{RT}{V}\right]dV - \ln Z
$$

ここで、$Z = PV/(RT)$ は圧縮因子です。

**Peng-Robinson方程式での計算**：

$$
\ln\phi = (Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B}\ln\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}
$$

ここで：

$$
A = \frac{a\alpha P}{R^2T^2}, \quad B = \frac{bP}{RT}
$$

### SCF相における化学ポテンシャル

超臨界流体中の溶質 $i$ の化学ポテンシャルは：

$$
\mu_i = \mu_i^{\text{pure}}(T, P) + RT\ln\gamma_i x_i
$$

ここで、$\gamma_i$ は活量係数、$x_i$ はモル分率です。

**相平衡条件**：

2つの相（例：固体と超臨界流体）が平衡にあるとき：

$$
\mu_i^{\text{solid}} = \mu_i^{\text{SCF}}
$$

これより、溶解度 $x_i$ が計算できます：

$$
x_i = \frac{1}{\gamma_i}\exp\left[\frac{\mu_i^{\text{pure}}(T, P) - \mu_i^{\text{solid}}(T, P)}{RT}\right]
$$

### 多成分系の混合則

多成分系の状態方程式パラメータは、純成分パラメータから**混合則**で計算します。

**van der Waals型混合則**：

$$
\begin{aligned}
a_{\text{mix}} &= \sum_i\sum_j x_i x_j a_{ij} \\
b_{\text{mix}} &= \sum_i x_i b_i
\end{aligned}
$$

**クロスパラメータ $a_{ij}$**：

$$
a_{ij} = \sqrt{a_i a_j}(1 - k_{ij})
$$

ここで、$k_{ij}$ は**二成分相互作用パラメータ**（Binary Interaction Parameter, BIP）です。

**$k_{ij}$ の決定**：

- 実験データからフィッティング
- 通常、$k_{ij} = 0.01-0.15$（類似分子では小さい）
- $k_{ii} = 0$（自己相互作用）

**例**：

| 成分ペア | $k_{ij}$ |
|---------|---------|
| CO₂ - エタノール | 0.10 |
| CO₂ - 水 | 0.19 |
| CO₂ - ヘキサン | 0.13 |

---

## 2.6 Pythonコード例

### 例1: van der Waals方程式による等温線の計算

```python
import numpy as np
import matplotlib.pyplot as plt

# CO2のパラメータ
Tc = 304.1  # K
Pc = 7.38e6  # Pa
R = 8.314  # J/(mol·K)

# van der Waalsパラメータ
a = 27 * R**2 * Tc**2 / (64 * Pc)
b = R * Tc / (8 * Pc)

def van_der_waals_pressure(Vm, T):
    """van der Waals方程式から圧力を計算"""
    return R * T / (Vm - b) - a / Vm**2

# 体積範囲
Vm = np.linspace(1.5*b, 20*b, 500)

# 異なる温度での等温線
temperatures = [280, 300, 304.1, 310, 330]  # K
colors = ['blue', 'green', 'red', 'orange', 'purple']

plt.figure(figsize=(10, 6))
for T, color in zip(temperatures, colors):
    P = van_der_waals_pressure(Vm, T)
    # 負圧を除外
    valid = P > 0
    label = f'{T} K'
    if T == Tc:
        label += ' (臨界)'
    plt.plot(Vm[valid]*1e6, P[valid]/1e6, color=color, label=label, linewidth=2)

plt.axhline(Pc/1e6, color='red', linestyle='--', alpha=0.5, label=f'臨界圧力 ({Pc/1e6:.1f} MPa)')
plt.xlabel('モル体積 (cm³/mol)', fontsize=12)
plt.ylabel('圧力 (MPa)', fontsize=12)
plt.title('CO₂のvan der Waals等温線', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 500)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('vdw_isotherms.png', dpi=150)
plt.show()
```

### 例2: Peng-Robinson方程式による圧縮因子の計算

```python
import numpy as np
from scipy.optimize import fsolve

def peng_robinson_Z(T, P, Tc, Pc, omega):
    """Peng-Robinson方程式から圧縮因子Zを計算"""
    R = 8.314  # J/(mol·K)

    # パラメータ計算
    a = 0.45724 * R**2 * Tc**2 / Pc
    b = 0.07780 * R * Tc / Pc
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2

    A = a * alpha * P / (R**2 * T**2)
    B = b * P / (R * T)

    # 圧縮因子の三次方程式
    # Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
    def equation(Z):
        return Z**3 - (1-B)*Z**2 + (A - 3*B**2 - 2*B)*Z - (A*B - B**2 - B**3)

    # 初期推定値（気相と液相）
    Z_gas = fsolve(equation, 1.0)[0]
    Z_liquid = fsolve(equation, 0.1)[0]

    # 物理的に意味のある解を選択（最大と最小）
    roots = np.sort([Z_gas, Z_liquid])
    return roots[-1], roots[0]  # (気相, 液相)

# CO2のパラメータ
Tc = 304.1  # K
Pc = 7.38e6  # Pa
omega = 0.228

# 条件
T = 320  # K (超臨界)
pressures = np.linspace(1e6, 15e6, 100)  # Pa

Z_values = []
for P in pressures:
    Z_gas, Z_liquid = peng_robinson_Z(T, P, Tc, Pc, omega)
    Z_values.append(Z_gas)

plt.figure(figsize=(10, 6))
plt.plot(pressures/1e6, Z_values, 'b-', linewidth=2)
plt.axhline(1.0, color='gray', linestyle='--', label='理想気体 (Z=1)')
plt.axvline(Pc/1e6, color='red', linestyle='--', alpha=0.5, label=f'臨界圧力 ({Pc/1e6:.1f} MPa)')
plt.xlabel('圧力 (MPa)', fontsize=12)
plt.ylabel('圧縮因子 Z', fontsize=12)
plt.title(f'CO₂の圧縮因子 (T = {T} K, Peng-Robinson)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_compressibility.png', dpi=150)
plt.show()

print(f"T = {T} K, P = {Pc/1e6:.1f} MPa (臨界圧力)")
Z_gas, Z_liquid = peng_robinson_Z(T, Pc, Tc, Pc, omega)
print(f"圧縮因子 Z = {Z_gas:.3f}")
```

### 例3: 臨界点近傍の密度ゆらぎ（臨界指数）

```python
import numpy as np
import matplotlib.pyplot as plt

def density_difference(T, Tc, rho_c, beta=0.326):
    """
    臨界点近傍の気液密度差
    ρ_L - ρ_G ∝ |T - Tc|^β
    """
    epsilon = np.abs((T - Tc) / Tc)
    # 比例定数は適当に設定
    B0 = 2.0 * rho_c
    return B0 * epsilon**beta

# パラメータ（CO2）
Tc = 304.1  # K
rho_c = 467.6  # kg/m³
beta = 0.326  # 臨界指数

# 温度範囲（臨界点以下）
T = np.linspace(280, Tc-0.01, 100)

# 密度差の計算
delta_rho = density_difference(T, Tc, rho_c, beta)

# 対数プロット
epsilon = (Tc - T) / Tc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 線形プロット
ax1.plot(T, delta_rho, 'b-', linewidth=2)
ax1.set_xlabel('温度 (K)', fontsize=12)
ax1.set_ylabel('密度差 ρ_L - ρ_G (kg/m³)', fontsize=12)
ax1.set_title('臨界点近傍の気液密度差', fontsize=14)
ax1.axvline(Tc, color='red', linestyle='--', label=f'臨界温度 ({Tc} K)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 対数プロット（べき乗則の検証）
ax2.loglog(epsilon, delta_rho, 'bo-', linewidth=2, label='データ')
# 理論直線（傾き = β）
fit_line = delta_rho[0] * (epsilon / epsilon[0])**beta
ax2.loglog(epsilon, fit_line, 'r--', linewidth=2, label=f'べき乗則 (β={beta})')
ax2.set_xlabel('換算温度差 ε = (Tc - T) / Tc', fontsize=12)
ax2.set_ylabel('密度差 (kg/m³)', fontsize=12)
ax2.set_title('べき乗則の検証（対数プロット）', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('critical_exponent.png', dpi=150)
plt.show()

print(f"臨界指数 β = {beta}")
print(f"ρ_L - ρ_G ∝ |T - Tc|^{beta}")
```

### 例4: Chrastil式による溶解度予測

```python
import numpy as np
import matplotlib.pyplot as plt

def chrastil_solubility(rho, T, k, a, b):
    """
    Chrastil式による溶解度計算
    ln(C) = k*ln(ρ) + a/T + b

    Parameters:
    -----------
    rho : float or array
        SCFの密度 (kg/m³)
    T : float
        温度 (K)
    k, a, b : float
        Chrastilパラメータ

    Returns:
    --------
    C : float or array
        溶解度 (kg溶質/kg SCF)
    """
    ln_C = k * np.log(rho) + a / T + b
    return np.exp(ln_C)

# カフェインの例（CO2中）
# パラメータ（文献値）
k = 7.5
a = -5200  # K
b = -15.3

# 条件
T = 313.15  # K (40°C)
rho = np.linspace(200, 900, 100)  # kg/m³

# 溶解度計算
C = chrastil_solubility(rho, T, k, a, b)

# 異なる温度での比較
temperatures = [313.15, 323.15, 333.15]  # K (40, 50, 60°C)
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))
for T, color in zip(temperatures, colors):
    C = chrastil_solubility(rho, T, k, a, b)
    plt.plot(rho, C*1000, color=color, linewidth=2, label=f'{T-273.15:.0f}°C')

plt.xlabel('CO₂密度 (kg/m³)', fontsize=12)
plt.ylabel('溶解度 (g/kg-CO₂)', fontsize=12)
plt.title('超臨界CO₂中のカフェイン溶解度 (Chrastil式)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(200, 900)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('chrastil_solubility.png', dpi=150)
plt.show()

# 圧力への変換（状態方程式が必要だが簡易的に）
print("\n溶解度の密度依存性:")
print(f"ρ = 400 kg/m³: C = {chrastil_solubility(400, 313.15, k, a, b)*1000:.2f} g/kg")
print(f"ρ = 700 kg/m³: C = {chrastil_solubility(700, 313.15, k, a, b)*1000:.2f} g/kg")
print(f"密度を1.75倍にすると溶解度は約 {(700/400)**k:.1f}倍")
```

### 例5: フガシティ係数の計算（Peng-Robinson）

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def fugacity_coefficient_PR(T, P, Tc, Pc, omega):
    """
    Peng-Robinson方程式からフガシティ係数を計算
    """
    R = 8.314  # J/(mol·K)

    # パラメータ
    a = 0.45724 * R**2 * Tc**2 / Pc
    b = 0.07780 * R * Tc / Pc
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2

    A = a * alpha * P / (R**2 * T**2)
    B = b * P / (R * T)

    # 圧縮因子の計算
    def equation(Z):
        return Z**3 - (1-B)*Z**2 + (A - 3*B**2 - 2*B)*Z - (A*B - B**2 - B**3)

    Z = fsolve(equation, 1.0)[0]

    # フガシティ係数
    sqrt2 = np.sqrt(2)
    ln_phi = (Z - 1) - np.log(Z - B) - \
             A/(2*sqrt2*B) * np.log((Z + (1+sqrt2)*B) / (Z + (1-sqrt2)*B))

    phi = np.exp(ln_phi)
    return phi, Z

# CO2のパラメータ
Tc = 304.1  # K
Pc = 7.38e6  # Pa
omega = 0.228

# 温度と圧力範囲
temperatures = [300, 320, 350]  # K
pressures = np.linspace(0.1e6, 20e6, 100)  # Pa

plt.figure(figsize=(10, 6))
for T in temperatures:
    phi_values = []
    for P in pressures:
        phi, Z = fugacity_coefficient_PR(T, P, Tc, Pc, omega)
        phi_values.append(phi)

    label = f'{T} K'
    if T < Tc:
        label += ' (亜臨界)'
    else:
        label += ' (超臨界)'
    plt.plot(pressures/1e6, phi_values, linewidth=2, label=label)

plt.axhline(1.0, color='gray', linestyle='--', label='理想気体 (φ=1)')
plt.axvline(Pc/1e6, color='red', linestyle='--', alpha=0.5, label=f'臨界圧力')
plt.xlabel('圧力 (MPa)', fontsize=12)
plt.ylabel('フガシティ係数 φ = f/P', fontsize=12)
plt.title('CO₂のフガシティ係数 (Peng-Robinson)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fugacity_coefficient.png', dpi=150)
plt.show()

# 特定条件での値
T_target = 320  # K
P_target = 10e6  # Pa
phi, Z = fugacity_coefficient_PR(T_target, P_target, Tc, Pc, omega)
f = phi * P_target
print(f"\nT = {T_target} K, P = {P_target/1e6} MPa:")
print(f"圧縮因子 Z = {Z:.3f}")
print(f"フガシティ係数 φ = {phi:.3f}")
print(f"フガシティ f = {f/1e6:.2f} MPa")
print(f"実在性の程度: {abs(1-phi)*100:.1f}%")
```

### 例6: 二成分系相平衡（混合則の適用）

```python
import numpy as np
import matplotlib.pyplot as plt

def mixing_rule(x1, a1, a2, b1, b2, k12=0):
    """
    van der Waals型混合則

    Parameters:
    -----------
    x1 : float
        成分1のモル分率
    a1, a2 : float
        各成分のaパラメータ
    b1, b2 : float
        各成分のbパラメータ
    k12 : float
        二成分相互作用パラメータ

    Returns:
    --------
    a_mix, b_mix : float
        混合系のパラメータ
    """
    x2 = 1 - x1

    # クロスパラメータ
    a12 = np.sqrt(a1 * a2) * (1 - k12)

    # 混合則
    a_mix = x1**2 * a1 + 2*x1*x2*a12 + x2**2 * a2
    b_mix = x1 * b1 + x2 * b2

    return a_mix, b_mix

# CO2とエタノールの例
R = 8.314  # J/(mol·K)

# CO2
Tc1, Pc1 = 304.1, 7.38e6  # K, Pa
a1 = 27 * R**2 * Tc1**2 / (64 * Pc1)
b1 = R * Tc1 / (8 * Pc1)

# エタノール
Tc2, Pc2 = 513.9, 6.14e6  # K, Pa
a2 = 27 * R**2 * Tc2**2 / (64 * Pc2)
b2 = R * Tc2 / (8 * Pc2)

# 相互作用パラメータ
k12_values = [0.0, 0.05, 0.10, 0.15]

# モル分率範囲
x_CO2 = np.linspace(0, 1, 100)

# プロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for k12 in k12_values:
    a_mix_values = []
    b_mix_values = []

    for x1 in x_CO2:
        a_mix, b_mix = mixing_rule(x1, a1, a2, b1, b2, k12)
        a_mix_values.append(a_mix)
        b_mix_values.append(b_mix)

    ax1.plot(x_CO2, np.array(a_mix_values)*1e6, linewidth=2, label=f'k₁₂ = {k12}')
    ax2.plot(x_CO2, np.array(b_mix_values)*1e6, linewidth=2, label=f'k₁₂ = {k12}')

ax1.set_xlabel('CO₂モル分率', fontsize=12)
ax1.set_ylabel('a_mix (Pa·m⁶/mol² × 10⁻⁶)', fontsize=12)
ax1.set_title('混合系のaパラメータ', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('CO₂モル分率', fontsize=12)
ax2.set_ylabel('b_mix (m³/mol × 10⁶)', fontsize=12)
ax2.set_title('混合系のbパラメータ', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mixing_rule.png', dpi=150)
plt.show()

# 相互作用パラメータの影響
print("\n相互作用パラメータk₁₂の影響（x_CO₂ = 0.5）:")
for k12 in k12_values:
    a_mix, b_mix = mixing_rule(0.5, a1, a2, b1, b2, k12)
    print(f"k₁₂ = {k12:.2f}: a_mix = {a_mix*1e6:.3f} (×10⁻⁶), b_mix = {b_mix*1e6:.3f} (×10⁶)")
```

---

## まとめ

本章では、超臨界流体の熱力学的基礎を包括的に学びました。

**主要なポイント**：

1. **状態方程式**：
   - 理想気体の法則は高圧・高密度では破綻
   - van der Waals方程式は分子間力と排除体積を補正
   - Peng-Robinson方程式は実用的な高精度EOS

2. **臨界現象**：
   - 臨界点では物性が異常な振る舞いを示す（発散、極値）
   - 普遍性により、異なる物質も同じ臨界指数で記述
   - くりこみ群理論が理論的基礎を提供

3. **相平衡**：
   - 超臨界流体を含む系では複雑な相挙動
   - Chrastil式など半経験式が溶解度予測に有用
   - 逆行凝縮などの特異な現象

4. **熱力学計算**：
   - フガシティにより実在気体の化学ポテンシャルを記述
   - 混合則により多成分系の性質を予測
   - 二成分相互作用パラメータが精度の鍵

**次章への接続**：

第3章では、超臨界流体の**輸送物性**（粘度、拡散係数、熱伝導率）を学び、プロセス設計に必要な動的特性を理解します。

---

## ナビゲーション

- [← 第1章: 超臨界流体とは何か](./chapter-1.md)
- [シリーズ目次に戻る](./index.md)
- [第3章: 超臨界流体の輸送物性 →](./chapter-3.md)

---

**更新履歴**:
- 2025-12-25: 初版公開
