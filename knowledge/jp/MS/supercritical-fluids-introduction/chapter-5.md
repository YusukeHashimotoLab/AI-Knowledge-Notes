---
title: "第5章: Pythonによる実践計算"
subtitle: "物性計算、相図作成、およびプロセスシミュレーション"
author: "AI Materials Informatics Lab"
date: "2025-12-25"
reading_time: "35分"
categories: ["超臨界流体", "Python", "熱力学計算"]
tags: ["CoolProp", "状態方程式", "相図", "プロセスシミュレーション"]
difficulty: "intermediate"
prerequisites: ["第1章-4章の内容", "Python基礎", "NumPy/Matplotlib"]
---

# 第5章: Pythonによる実践計算

## 学習目標

本章を通じて、以下のスキルを習得できます：

- **CoolPropライブラリ**を使用した超臨界流体の物性計算
- **相図の構築**とMatplotlibによる可視化
- **状態方程式の実装**（van der Waals、Peng-Robinson）
- **プロセスシミュレーション**の基礎モデリング
- **実験データのフィッティング**と検証手法

---

## 5.1 超臨界流体計算ライブラリの紹介

### CoolPropの概要

[CoolProp](http://www.coolprop.org/)は、純物質および混合物の熱物性を高精度で計算できるオープンソースライブラリです。

**主な特徴：**
- 122種類以上の純物質データベース
- 状態方程式（Helmholtz自由エネルギー、立方EOS等）
- 高速計算（C++バックエンド）
- Python/MATLAB/Excel等多言語対応

**インストール：**

```bash
pip install CoolProp
```

### thermoライブラリの概要

[thermo](https://github.com/CalebBell/thermo)は、化学工学向けの物性計算ライブラリです。

**主な特徴：**
- 20,000種類以上の化合物データ
- 推算法のライブラリ（Group Contribution法等）
- 混合物の相平衡計算

**インストール：**

```bash
pip install thermo
```

### ライブラリの比較

| 特徴 | CoolProp | thermo |
|------|----------|--------|
| 物質数 | 122+ | 20,000+ |
| 精度 | 非常に高い | 高い（推算法含む） |
| 速度 | 非常に速い | 速い |
| 混合物 | 対応 | 強力な対応 |
| 学習曲線 | 緩やか | やや急 |

**本章の方針：**
- 主にCoolPropを使用（高精度・高速）
- 必要に応じてカスタム実装を追加

---

## 5.2 CoolPropによる物性計算

### 基本的な使用パターン

CoolPropの基本関数は`PropsSI`（SI単位系）です：

```python
from CoolProp.CoolProp import PropsSI

# 構文: PropsSI('出力物性', '入力1名', 値1, '入力2名', 値2, '物質名')
# 例: 20℃、1barのCO₂密度 [kg/m³]
rho = PropsSI('D', 'T', 293.15, 'P', 1e5, 'CO2')
print(f"密度: {rho:.2f} kg/m³")
```

**主要な物性記号：**

| 記号 | 物性 | 単位 |
|------|------|------|
| `D` | 密度 (Density) | kg/m³ |
| `T` | 温度 (Temperature) | K |
| `P` | 圧力 (Pressure) | Pa |
| `H` | エンタルピー (Enthalpy) | J/kg |
| `S` | エントロピー (Entropy) | J/(kg·K) |
| `V` | 粘度 (Viscosity) | Pa·s |
| `L` | 熱伝導率 (Thermal conductivity) | W/(m·K) |
| `C` | 定圧比熱 (Cp) | J/(kg·K) |
| `O` | 定容比熱 (Cv) | J/(kg·K) |
| `A` | 音速 (Speed of sound) | m/s |

### 例1: CO₂物性表の生成

```python
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

def generate_co2_property_table(T_celsius, P_bar):
    """
    CO₂の物性表を生成

    Parameters
    ----------
    T_celsius : float
        温度 [℃]
    P_bar : float
        圧力 [bar]

    Returns
    -------
    dict
        物性辞書
    """
    T = T_celsius + 273.15  # K
    P = P_bar * 1e5  # Pa

    properties = {
        '温度 [℃]': T_celsius,
        '圧力 [bar]': P_bar,
        '密度 [kg/m³]': PropsSI('D', 'T', T, 'P', P, 'CO2'),
        '粘度 [μPa·s]': PropsSI('V', 'T', T, 'P', P, 'CO2') * 1e6,
        '熱伝導率 [mW/(m·K)]': PropsSI('L', 'T', T, 'P', P, 'CO2') * 1e3,
        'Cp [J/(kg·K)]': PropsSI('C', 'T', T, 'P', P, 'CO2'),
        'エンタルピー [kJ/kg]': PropsSI('H', 'T', T, 'P', P, 'CO2') / 1e3,
        '音速 [m/s]': PropsSI('A', 'T', T, 'P', P, 'CO2'),
    }

    return properties

# 超臨界CO₂の条件範囲でテーブル生成
temperatures = [35, 40, 50, 60, 80, 100]  # ℃
pressures = [80, 100, 150, 200, 300]  # bar

data = []
for T in temperatures:
    for P in pressures:
        try:
            props = generate_co2_property_table(T, P)
            data.append(props)
        except Exception as e:
            print(f"エラー (T={T}℃, P={P}bar): {e}")

df = pd.DataFrame(data)
print("\n超臨界CO₂物性表:")
print(df.to_string(index=False))

# CSVエクスポート
df.to_csv('co2_properties.csv', index=False, encoding='utf-8-sig')
```

### 臨界点の取得

```python
# CO₂の臨界点
T_crit = PropsSI('Tcrit', 'CO2')  # K
P_crit = PropsSI('Pcrit', 'CO2')  # Pa
rho_crit = PropsSI('rhocrit', 'CO2')  # kg/m³

print(f"CO₂臨界点:")
print(f"  温度: {T_crit - 273.15:.2f} ℃")
print(f"  圧力: {P_crit / 1e5:.2f} bar")
print(f"  密度: {rho_crit:.2f} kg/m³")
```

---

## 5.3 相図の構築

### 例2: P-T相図（相境界付き）

```python
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

def plot_pt_diagram_co2():
    """CO₂のP-T相図を作成（相境界付き）"""

    # 臨界点
    T_crit = PropsSI('Tcrit', 'CO2')
    P_crit = PropsSI('Pcrit', 'CO2')

    # 三重点
    T_triple = PropsSI('Ttriple', 'CO2')
    P_triple = PropsSI('ptriple', 'CO2')

    # 昇華曲線（固-気）近似
    T_sublimation = np.linspace(T_triple, 216.6, 50)
    P_sublimation = [PropsSI('P', 'T', T, 'Q', 0, 'CO2') for T in T_sublimation]

    # 融解曲線（固-液）近似（CO₂は特殊な勾配）
    T_melting = np.linspace(T_triple, 273.15, 30)
    # 簡易モデル: P = P_triple + a*(T - T_triple)
    a = 1.6e7  # Pa/K (実験値に基づく近似)
    P_melting = P_triple + a * (T_melting - T_triple)

    # 蒸気圧曲線（液-気）
    T_vaporization = np.linspace(T_triple, T_crit, 100)
    P_vaporization = [PropsSI('P', 'T', T, 'Q', 0, 'CO2') for T in T_vaporization]

    # プロット
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot((T_sublimation - 273.15), np.array(P_sublimation) / 1e5,
            'b-', linewidth=2, label='昇華曲線')
    ax.plot((T_melting - 273.15), np.array(P_melting) / 1e5,
            'g-', linewidth=2, label='融解曲線')
    ax.plot((T_vaporization - 273.15), np.array(P_vaporization) / 1e5,
            'r-', linewidth=2, label='蒸気圧曲線')

    # 臨界点
    ax.plot(T_crit - 273.15, P_crit / 1e5, 'ko', markersize=10,
            label=f'臨界点 ({T_crit-273.15:.1f}℃, {P_crit/1e5:.1f} bar)')

    # 三重点
    ax.plot(T_triple - 273.15, P_triple / 1e5, 'ks', markersize=10,
            label=f'三重点 ({T_triple-273.15:.1f}℃, {P_triple/1e5:.2f} bar)')

    # 超臨界領域の塗りつぶし
    ax.fill_between([T_crit - 273.15, 100], [P_crit / 1e5, P_crit / 1e5],
                     [P_crit / 1e5, 500], alpha=0.2, color='orange',
                     label='超臨界領域')

    # 相領域のラベル
    ax.text(-50, 20, '固相', fontsize=14, ha='center')
    ax.text(10, 20, '液相', fontsize=14, ha='center')
    ax.text(-20, 2, '気相', fontsize=14, ha='center')
    ax.text(50, 150, '超臨界流体', fontsize=14, ha='center', color='orange')

    ax.set_xlabel('温度 [℃]', fontsize=12)
    ax.set_ylabel('圧力 [bar]', fontsize=12)
    ax.set_title('CO₂のP-T相図', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-80, 100)
    ax.set_ylim(0.1, 500)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('co2_pt_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_pt_diagram_co2()
```

### 例3: P-ρ図（等温線）

```python
def plot_p_rho_isotherms_co2():
    """CO₂のP-ρ図（等温線付き）"""

    T_crit = PropsSI('Tcrit', 'CO2') - 273.15  # ℃
    P_crit = PropsSI('Pcrit', 'CO2') / 1e5  # bar

    # 等温線の温度リスト
    temperatures = [20, 30, T_crit, 40, 60, 80]  # ℃

    fig, ax = plt.subplots(figsize=(10, 7))

    for T_celsius in temperatures:
        T = T_celsius + 273.15  # K

        # 圧力範囲
        P_range = np.linspace(1, 300, 200)  # bar
        densities = []
        pressures_valid = []

        for P_bar in P_range:
            try:
                P = P_bar * 1e5  # Pa
                rho = PropsSI('D', 'T', T, 'P', P, 'CO2')
                densities.append(rho)
                pressures_valid.append(P_bar)
            except:
                pass

        label = f'{T_celsius:.0f}℃'
        if abs(T_celsius - T_crit) < 0.5:
            label += ' (臨界)'
            ax.plot(densities, pressures_valid, linewidth=3, label=label)
        else:
            ax.plot(densities, pressures_valid, linewidth=2, label=label)

    ax.set_xlabel('密度 [kg/m³]', fontsize=12)
    ax.set_ylabel('圧力 [bar]', fontsize=12)
    ax.set_title('CO₂のP-ρ図（等温線）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 300)

    plt.tight_layout()
    plt.savefig('co2_p_rho_isotherms.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_p_rho_isotherms_co2()
```

---

## 5.4 状態方程式の実装

### van der Waals状態方程式

第2章で紹介したvan der Waals式をPythonで実装します：

$$
\left(P + \frac{a}{V_m^2}\right)(V_m - b) = RT
$$

```python
import numpy as np
from scipy.optimize import fsolve

class VanDerWaalsEOS:
    """van der Waals状態方程式"""

    def __init__(self, Tc, Pc):
        """
        Parameters
        ----------
        Tc : float
            臨界温度 [K]
        Pc : float
            臨界圧力 [Pa]
        """
        self.Tc = Tc
        self.Pc = Pc
        self.R = 8.314  # J/(mol·K)

        # van der Waalsパラメータ
        self.a = 27 * (self.R * Tc)**2 / (64 * Pc)  # Pa·m⁶/mol²
        self.b = self.R * Tc / (8 * Pc)  # m³/mol

    def pressure(self, V, T):
        """
        圧力計算

        Parameters
        ----------
        V : float
            モル体積 [m³/mol]
        T : float
            温度 [K]

        Returns
        -------
        float
            圧力 [Pa]
        """
        return self.R * T / (V - self.b) - self.a / V**2

    def molar_volume(self, P, T):
        """
        モル体積計算（3次方程式の解）

        Parameters
        ----------
        P : float
            圧力 [Pa]
        T : float
            温度 [K]

        Returns
        -------
        array
            モル体積の解 [m³/mol]（実根のみ）
        """
        # van der Waals式を展開した3次方程式の係数
        # V³ - (b + RT/P)V² + (a/P)V - ab/P = 0
        coeffs = [
            1,
            -(self.b + self.R * T / P),
            self.a / P,
            -self.a * self.b / P
        ]

        roots = np.roots(coeffs)
        # 実根かつ正の値のみ
        real_positive_roots = roots[(np.isreal(roots)) & (roots.real > 0)].real

        return real_positive_roots

# CO₂でテスト
Tc_co2 = PropsSI('Tcrit', 'CO2')
Pc_co2 = PropsSI('Pcrit', 'CO2')

vdw = VanDerWaalsEOS(Tc_co2, Pc_co2)

# 超臨界条件での計算
T = 313.15  # 40℃
P = 100e5  # 100 bar

V_roots = vdw.molar_volume(P, T)
print(f"\nvan der Waals式による計算 (T={T-273.15}℃, P={P/1e5} bar):")
print(f"  モル体積の解: {V_roots * 1e6:.2f} cm³/mol")

# CoolPropとの比較
V_coolprop = 1 / PropsSI('D', 'T', T, 'P', P, 'CO2') * 44.01  # g/molをm³/molに変換
print(f"  CoolProp: {V_coolprop * 1e6:.2f} cm³/mol")
print(f"  相対誤差: {abs(V_roots[0] - V_coolprop) / V_coolprop * 100:.1f}%")
```

### Peng-Robinson状態方程式

より高精度なPeng-Robinson式の実装：

$$
P = \frac{RT}{V_m - b} - \frac{a\alpha(T)}{V_m^2 + 2bV_m - b^2}
$$

```python
class PengRobinsonEOS:
    """Peng-Robinson状態方程式"""

    def __init__(self, Tc, Pc, omega):
        """
        Parameters
        ----------
        Tc : float
            臨界温度 [K]
        Pc : float
            臨界圧力 [Pa]
        omega : float
            偏心因子 [-]
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.R = 8.314  # J/(mol·K)

        # PRパラメータ
        self.a = 0.45724 * (self.R * Tc)**2 / Pc
        self.b = 0.07780 * self.R * Tc / Pc

        # κパラメータ
        if omega <= 0.49:
            self.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        else:
            self.kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3

    def alpha(self, T):
        """温度補正項"""
        Tr = T / self.Tc
        return (1 + self.kappa * (1 - np.sqrt(Tr)))**2

    def pressure(self, V, T):
        """圧力計算"""
        alpha_T = self.alpha(T)
        return (self.R * T / (V - self.b) -
                self.a * alpha_T / (V**2 + 2*self.b*V - self.b**2))

    def compressibility_factor(self, P, T):
        """圧縮因子Z（3次方程式の解）"""
        alpha_T = self.alpha(T)
        A = self.a * alpha_T * P / (self.R * T)**2
        B = self.b * P / (self.R * T)

        # Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [
            1,
            -(1 - B),
            A - 3*B**2 - 2*B,
            -(A*B - B**2 - B**3)
        ]

        Z_roots = np.roots(coeffs)
        Z_real = Z_roots[np.isreal(Z_roots)].real

        return Z_real

# CO₂でテスト（偏心因子: 0.224）
pr = PengRobinsonEOS(Tc_co2, Pc_co2, omega=0.224)

T = 313.15  # 40℃
P = 100e5  # 100 bar

Z_pr = pr.compressibility_factor(P, T)
V_pr = Z_pr * 8.314 * T / P

print(f"\nPeng-Robinson式による計算:")
print(f"  圧縮因子Z: {Z_pr[0]:.4f}")
print(f"  モル体積: {V_pr[0] * 1e6:.2f} cm³/mol")
print(f"  CoolPropとの相対誤差: {abs(V_pr[0] - V_coolprop) / V_coolprop * 100:.1f}%")
```

---

## 5.5 プロセスシミュレーション例

### 例4: 超臨界CO₂抽出プロセスモデル

```python
class SCF_ExtractionModel:
    """超臨界流体抽出の簡易モデル"""

    def __init__(self, solute_mw, K_chrastil_params):
        """
        Parameters
        ----------
        solute_mw : float
            溶質の分子量 [g/mol]
        K_chrastil_params : dict
            Chrastil式のパラメータ {'k': ..., 'a': ..., 'b': ...}
        """
        self.solute_mw = solute_mw
        self.k = K_chrastil_params['k']
        self.a = K_chrastil_params['a']
        self.b = K_chrastil_params['b']

    def solubility_chrastil(self, T, rho):
        """
        Chrastil式による溶解度計算

        S = ρ^k * exp(a/T + b)

        Parameters
        ----------
        T : float
            温度 [K]
        rho : float
            CO₂密度 [kg/m³]

        Returns
        -------
        float
            溶解度 [kg溶質/kg CO₂]
        """
        return (rho / 1000)**self.k * np.exp(self.a / T + self.b)

    def extraction_yield(self, T_celsius, P_bar, flow_rate_co2, time_hours,
                        solute_mass_initial):
        """
        抽出収率のシミュレーション（簡易モデル）

        Parameters
        ----------
        T_celsius : float
            温度 [℃]
        P_bar : float
            圧力 [bar]
        flow_rate_co2 : float
            CO₂流量 [kg/h]
        time_hours : float
            抽出時間 [h]
        solute_mass_initial : float
            初期溶質量 [kg]

        Returns
        -------
        dict
            抽出結果
        """
        T = T_celsius + 273.15
        P = P_bar * 1e5

        # CO₂密度
        rho = PropsSI('D', 'T', T, 'P', P, 'CO2')

        # 溶解度
        S = self.solubility_chrastil(T, rho)

        # 総CO₂使用量
        total_co2 = flow_rate_co2 * time_hours  # kg

        # 理論最大抽出量
        max_extracted = S * total_co2  # kg

        # 実際の抽出量（簡易指数減衰モデル）
        # E(t) = E_max * (1 - exp(-kt))
        k_rate = 0.5  # 1/h (抽出速度定数、実験的に決定)
        actual_extracted = min(
            max_extracted * (1 - np.exp(-k_rate * time_hours)),
            solute_mass_initial
        )

        yield_percent = (actual_extracted / solute_mass_initial) * 100

        return {
            'CO₂密度 [kg/m³]': rho,
            '溶解度 [g/kg-CO₂]': S * 1000,
            '総CO₂使用量 [kg]': total_co2,
            '抽出量 [kg]': actual_extracted,
            '抽出収率 [%]': yield_percent,
            'CO₂原単位 [kg-CO₂/kg-product]': total_co2 / actual_extracted if actual_extracted > 0 else np.inf
        }

# カフェイン抽出の例（Chrastilパラメータは文献値）
caffeine_params = {'k': 8.0, 'a': -5000, 'b': 15}
extractor = SCF_ExtractionModel(solute_mw=194.19, K_chrastil_params=caffeine_params)

# 運転条件
conditions = {
    'T_celsius': 50,
    'P_bar': 200,
    'flow_rate_co2': 10,  # kg/h
    'time_hours': 3,
    'solute_mass_initial': 0.5  # kg
}

result = extractor.extraction_yield(**conditions)

print("\n超臨界CO₂抽出シミュレーション結果:")
for key, value in result.items():
    print(f"  {key}: {value:.2f}")
```

### 例5: RESS法の粒子生成モデル

```python
def ress_particle_size_model(T_celsius, P_bar, nozzle_diameter_mm,
                             expansion_ratio):
    """
    RESS (Rapid Expansion of Supercritical Solutions) 法の粒子径予測

    簡易モデル: Weber数による予測

    Parameters
    ----------
    T_celsius : float
        ノズル上流温度 [℃]
    P_bar : float
        ノズル上流圧力 [bar]
    nozzle_diameter_mm : float
        ノズル径 [mm]
    expansion_ratio : float
        膨張比 P_upstream / P_downstream

    Returns
    -------
    dict
        粒子径予測結果
    """
    T = T_celsius + 273.15
    P_upstream = P_bar * 1e5
    P_downstream = P_upstream / expansion_ratio

    # 上流側物性
    rho_upstream = PropsSI('D', 'T', T, 'P', P_upstream, 'CO2')
    mu_upstream = PropsSI('V', 'T', T, 'P', P_upstream, 'CO2')

    # 音速（等エントロピー膨張の理論速度近似に使用）
    a = PropsSI('A', 'T', T, 'P', P_upstream, 'CO2')

    # ノズル出口速度（等エントロピー膨張の簡易計算）
    # v = sqrt(2 * Δh)、Δh ≈ a² * ln(P1/P2) / γ
    gamma = PropsSI('C', 'T', T, 'P', P_upstream, 'CO2') / PropsSI('O', 'T', T, 'P', P_upstream, 'CO2')
    v_exit = np.sqrt(2 * a**2 * np.log(expansion_ratio) / gamma)

    # Weber数 We = ρ * v² * d / σ
    # 表面張力σ（簡易推定、実際は溶質依存）
    sigma = 0.01  # N/m (典型値)
    d_nozzle = nozzle_diameter_mm * 1e-3  # m
    We = rho_upstream * v_exit**2 * d_nozzle / sigma

    # 粒子径予測（経験式）
    # d_particle ≈ d_nozzle / We^0.5
    d_particle = d_nozzle / np.sqrt(We)  # m

    return {
        'ノズル出口速度 [m/s]': v_exit,
        'Weber数 [-]': We,
        '予測粒子径 [μm]': d_particle * 1e6,
        'CO₂密度 [kg/m³]': rho_upstream,
        '膨張比 [-]': expansion_ratio
    }

# シミュレーション実行
ress_result = ress_particle_size_model(
    T_celsius=60,
    P_bar=150,
    nozzle_diameter_mm=0.1,
    expansion_ratio=150
)

print("\nRESS法粒子径予測:")
for key, value in ress_result.items():
    print(f"  {key}: {value:.2f}")
```

---

## 5.6 データ解析とフィッティング

### 例6: 実験溶解度データのフィッティング

```python
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 実験データ（例：CO₂中のナフタレン溶解度）
experimental_data = {
    'T': [308.15, 308.15, 308.15, 318.15, 318.15, 318.15],  # K
    'P': [90e5, 120e5, 150e5, 90e5, 120e5, 150e5],  # Pa
    'S': [2.1e-3, 3.5e-3, 4.8e-3, 1.8e-3, 3.0e-3, 4.2e-3]  # kg/kg
}

def chrastil_model(X, k, a, b):
    """
    Chrastil式 S = ρ^k * exp(a/T + b)

    Parameters
    ----------
    X : tuple
        (T, P) - 温度[K]と圧力[Pa]
    k, a, b : float
        フィッティングパラメータ

    Returns
    -------
    float
        溶解度 [kg/kg]
    """
    T, P = X
    rho = PropsSI('D', 'T', T, 'P', P, 'CO2')
    return (rho / 1000)**k * np.exp(a / T + b)

# データ準備
T_data = np.array(experimental_data['T'])
P_data = np.array(experimental_data['P'])
S_data = np.array(experimental_data['S'])

# フィッティング
X_data = (T_data, P_data)
initial_guess = [8.0, -5000, 15]

params, covariance = curve_fit(
    chrastil_model,
    X_data,
    S_data,
    p0=initial_guess,
    maxfev=10000
)

k_fit, a_fit, b_fit = params
errors = np.sqrt(np.diag(covariance))

print("\nChrastil式フィッティング結果:")
print(f"  k = {k_fit:.3f} ± {errors[0]:.3f}")
print(f"  a = {a_fit:.1f} ± {errors[1]:.1f} K")
print(f"  b = {b_fit:.3f} ± {errors[2]:.3f}")

# モデル検証
S_predicted = chrastil_model(X_data, k_fit, a_fit, b_fit)
rmse = np.sqrt(np.mean((S_data - S_predicted)**2))
r_squared = 1 - np.sum((S_data - S_predicted)**2) / np.sum((S_data - np.mean(S_data))**2)

print(f"\nモデル精度:")
print(f"  RMSE: {rmse*1000:.4f} g/kg")
print(f"  R²: {r_squared:.4f}")

# プロット
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(S_data * 1000, S_predicted * 1000, s=100, alpha=0.7, edgecolors='black')
ax.plot([0, max(S_data)*1000*1.1], [0, max(S_data)*1000*1.1], 'r--', label='理想線')
ax.set_xlabel('実験値 [g/kg-CO₂]', fontsize=12)
ax.set_ylabel('予測値 [g/kg-CO₂]', fontsize=12)
ax.set_title(f'Chrastil式フィッティング (R²={r_squared:.3f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chrastil_fitting.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 5.7 発展的トピック（概説）

### 分子動力学シミュレーション

超臨界流体の分子レベル挙動を理解するためには、分子動力学（MD）シミュレーションが有効です。

**代表的なツール：**
- **LAMMPS**：汎用MD（CO₂ - 溶質系の溶解度予測）
- **GROMACS**：生体分子系（超臨界乾燥プロセス）
- **NAMD**：大規模系

**Pythonインターフェース：**
```bash
pip install lammps  # LAMMPS Python API
pip install MDAnalysis  # 軌跡解析
```

### 機械学習による物性予測

深層学習を使った物性推算：

```python
# 例: scikit-learnによる溶解度予測モデル
from sklearn.ensemble import RandomForestRegressor

# 特徴量: [T, P, 溶質の分子量, logP, 極性表面積...]
# ターゲット: 溶解度

# model = RandomForestRegressor()
# model.fit(X_train, y_train)
```

**有用なライブラリ：**
- `RDKit`：分子記述子計算
- `DeepChem`：化学専用深層学習
- `ChemML`：ケモインフォマティクス

### プロセスシミュレータ連携

産業用プロセスシミュレータとの連携：

**Aspen Plus / HYSYS**
- Pythonから`pywin32`経由でCOM接続
- 超臨界抽出プロセスの最適化

**DWSIM (オープンソース)**
- Python API内蔵
- 超臨界流体プロセスのシミュレーション

---

## まとめ

本章では、Pythonを用いた超臨界流体の実践的計算手法を学びました：

**習得した技術：**
1. **CoolProp**による高精度物性計算
2. **相図の構築**とMatplotlibによる可視化
3. **状態方程式の実装**（van der Waals、Peng-Robinson）
4. **プロセスモデル**の構築（抽出、RESS）
5. **実験データフィッティング**とモデル検証

**次のステップ：**
- 実際の研究データへの適用
- より複雑なプロセスモデルの開発
- 機械学習による物性予測モデルの構築

---

## 演習問題

### 問題1: 物性計算の精度比較

**課題：**
水（H₂O）の超臨界条件（400℃、250 bar）において、以下を計算せよ：
1. CoolPropによる密度、粘度、熱伝導率
2. van der Waals式による密度（臨界点から計算）
3. 両者の相対誤差

**ヒント：**
```python
# 水の臨界点
Tc_water = PropsSI('Tcrit', 'Water')
Pc_water = PropsSI('Pcrit', 'Water')
```

---

### 問題2: 相図のカスタマイズ

**課題：**
エタノール（Ethanol）のP-T相図を作成し、以下を追加せよ：
1. 臨界点と三重点のマーカー
2. 1 atm（1.013 bar）の等圧線
3. 超臨界領域の塗りつぶし

**CoolPropの物質名：**`'Ethanol'`

---

### 問題3: プロセス最適化

**課題：**
超臨界CO₂によるカフェイン抽出プロセスを最適化せよ。

**条件：**
- 初期カフェイン量：1 kg
- 目標抽出収率：≥ 90%
- CO₂流量：5-20 kg/h
- 温度範囲：40-80℃
- 圧力範囲：100-300 bar

**最適化目標：**
CO₂原単位（kg-CO₂/kg-caffeine）を最小化

**ヒント：**
```python
from scipy.optimize import minimize

def objective(x):
    # x = [T, P, flow_rate, time]
    # return: CO₂原単位
    pass
```

---

### 問題4: 実験データの統計解析

**課題：**
以下の実験データに対してChrastil式をフィッティングし、95%信頼区間を計算せよ。

**データ（CO₂中のイブプロフェン溶解度）：**

| T [℃] | P [bar] | S [mg/kg] |
|-------|---------|-----------|
| 40 | 100 | 1.2 |
| 40 | 150 | 2.8 |
| 40 | 200 | 4.5 |
| 60 | 100 | 0.9 |
| 60 | 150 | 2.1 |
| 60 | 200 | 3.8 |

**要求出力：**
- フィッティングパラメータ（k, a, b）と標準誤差
- R²値
- 残差プロット

---

### 問題5: 発展課題 - 混合溶媒系

**課題：**
CO₂-エタノール混合超臨界流体（10 mol%エタノール）の密度を以下の条件で計算せよ：
- 温度：50℃
- 圧力：150 bar

**ヒント：**
CoolPropの混合物計算：
```python
from CoolProp.CoolProp import PropsSI

# 混合物の指定方法
backend = 'HEOS'  # Helmholtz EOS
mixture = 'CO2[0.9]&Ethanol[0.1]'
rho = PropsSI('D', 'T', 323.15, 'P', 150e5, backend + '::' + mixture)
```

---

## ナビゲーション

[← 第4章: 超臨界流体の応用例](chapter-4.md) | [目次](index.md)

---

**次章予告：**
第6章では、超臨界流体技術の最新研究動向と今後の展望について解説します。

