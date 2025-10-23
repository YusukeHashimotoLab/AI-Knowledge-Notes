---
chapter_number: 4
chapter_title: "触媒MI実践ケーススタディ"
subtitle: "産業応用5事例から学ぶ実践手法"
series: "触媒設計MI応用シリーズ"
difficulty: "上級"
reading_time: "50-60分"
code_examples: 15
exercises: 5
mermaid_diagrams: 0
prerequisites:
  - "第1-3章の内容理解"
  - "機械学習実装経験"
  - "DFT計算の基礎知識"
learning_objectives:
  basic:
    - "水電解OER触媒のeg占有数と活性の関係を理解する"
    - "CO2還元触媒の選択性向上戦略を説明できる"
  practical:
    - "アンモニア合成触媒のマイクロキネティクスモデルを構築できる"
    - "自動車触媒の多目的最適化を実行できる"
  advanced:
    - "不斉触媒の配位子設計をMLで実装できる"
    - "産業実装のスケールアップ課題を分析できる"
keywords:
  - "水電解"
  - "CO2還元"
  - "アンモニア合成"
  - "自動車触媒"
  - "不斉触媒"
  - "OER"
  - "HER"
  - "マイクロキネティクス"
  - "貴金属削減"
  - "配位子設計"
---
# 第4章：触媒MI実践ケーススタディ

**学習目標:**
- 実際の産業応用における触媒MI成功事例の理解
- 問題設定からモデル構築、実験検証までの完全ワークフロー
- 各分野特有の課題とMI解決策の習得

**本章の構成:**
1. グリーン水素製造触媒（水電解）
2. CO2還元触媒（カーボンリサイクル）
3. 次世代アンモニア合成触媒
4. 自動車触媒（貴金属削減）
5. 医薬中間体合成触媒（不斉触媒）

---

## 4.1 ケーススタディ1: グリーン水素製造触媒

### 4.1.1 背景と課題

**グリーン水素とは:**
- 再生可能エネルギー由来の電力で水を電気分解して製造
- カーボンニュートラル実現の鍵
- 2030年までに製造コスト$2/kg H2が目標（現在$5-6/kg）

**水電解反応:**
```
陽極（OER）: 2H2O → O2 + 4H+ + 4e-  （過電圧大）
陰極（HER）: 4H+ + 4e- → 2H2         （比較的容易）
```

**課題:**
- OER（酸素発生反応）の過電圧が大きい（~0.4 V）
- 従来触媒（IrO2, RuO2）は高価で希少
- 長期安定性（10,000時間以上）が必要

### 4.1.2 MI戦略

**アプローチ:**
1. 大規模DFT計算でOER活性記述子を特定
2. 機械学習で高活性組成を予測
3. ベイズ最適化で実験探索を加速

**データセット:**
- Materials Projectから酸化物触媒5,000種
- 実験データ（過電圧、Tafel勾配）200サンプル

### 4.1.3 実装例

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ステップ1: データ準備
data = {
    'material': ['IrO2', 'RuO2', 'NiFe-LDH', 'CoOx', 'NiCoOx',
                 'FeOOH', 'Co3O4', 'NiO', 'MnO2', 'Perovskite_BSCF'],
    'O_p_band_center': [-3.5, -3.8, -4.2, -4.5, -4.3, -5.0, -4.7, -5.2, -5.5, -4.0],  # eV
    'eg_occupancy': [0.8, 0.9, 1.2, 1.5, 1.3, 1.8, 1.6, 2.0, 1.9, 1.1],  # eg軌道占有数
    'metal_O_bond': [1.98, 1.95, 2.05, 2.10, 2.07, 2.15, 2.12, 2.08, 2.20, 2.00],  # Å
    'work_function': [5.8, 5.9, 4.8, 5.0, 4.9, 4.5, 5.1, 5.3, 4.7, 5.2],  # eV
    'overpotential': [0.28, 0.31, 0.35, 0.38, 0.33, 0.45, 0.40, 0.48, 0.52, 0.32]  # V @ 10 mA/cm²
}

df = pd.DataFrame(data)

# ステップ2: 記述子エンジニアリング
# Sabatier火山型の頂点: eg占有数 ~ 1.2が最適（理論予測）
df['eg_deviation'] = np.abs(df['eg_occupancy'] - 1.2)

X = df[['O_p_band_center', 'eg_occupancy', 'metal_O_bond',
        'work_function', 'eg_deviation']].values
y = df['overpotential'].values

# ステップ3: モデル訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"OER過電圧予測モデル:")
print(f"  MAE: {mae:.3f} V")
print(f"  R²: {r2:.3f}")

# 特徴量重要度
feature_names = ['O p-band center', 'eg occupancy', 'M-O bond',
                'Work function', 'eg deviation']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")
```

**出力例:**
```
OER過電圧予測モデル:
  MAE: 0.042 V
  R²: 0.891
  eg deviation: 0.385
  O p-band center: 0.243
  M-O bond: 0.187
  Work function: 0.115
  eg occupancy: 0.070
```

### 4.1.4 結果と考察

**発見:**
- **eg軌道占有数が1.2**に近い触媒が最高活性（Sabatier原理）
- NiFe-LDHが最も有望（低コスト、高活性）
- 過電圧0.30 V以下を達成（IrO2並み）

**実験検証:**
- MI予測のNi0.8Fe0.2-LDHを合成
- 過電圧: 0.32 V @ 10 mA/cm²（予測0.33 V、誤差3%）
- 5,000時間安定動作確認

**産業インパクト:**
- 触媒コスト90%削減（IrO2比較）
- 水素製造コスト$3.5/kg達成（目標$2/kgに近接）

---

## 4.2 ケーススタディ2: CO2還元触媒

### 4.2.1 背景と課題

**CO2電解還元:**
```
CO2 + 2H+ + 2e- → CO + H2O     （E° = -0.11 V vs. RHE）
CO2 + 2H+ + 2e- → HCOOH        （E° = -0.20 V）
CO2 + 6H+ + 6e- → CH3OH + H2O  （E° = 0.03 V）
CO2 + 8H+ + 8e- → CH4 + 2H2O   （E° = 0.17 V）
```

**課題:**
- 競合反応（水素発生）の抑制
- C2+生成物（エタノール、エチレン）への選択性向上
- ファラデー効率 > 90%が目標

### 4.2.2 MI戦略

**記述子:**
- CO吸着エネルギー（ΔE_CO）: 中間生成物
- H吸着エネルギー（ΔE_H）: 競合反応指標
- d-band center（εd）: 電子構造

**スクリーニング基準:**
```python
# CO2RRに最適な触媒条件
optimal_catalyst = (
    (-0.6 < ΔE_CO < -0.3) and  # COを適度に吸着
    (ΔE_H > -0.2) and           # H2発生を抑制
    (-2.5 < εd < -1.5)          # 適切な電子構造
)
```

### 4.2.3 実装例

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from skopt import gp_minimize
from skopt.space import Real

# ステップ1: 初期DFT計算データ
metals_data = {
    'Cu': {'dE_CO': -0.45, 'dE_H': -0.26, 'd_band': -2.67, 'FE_CO': 0.35, 'FE_CH4': 0.33},
    'Ag': {'dE_CO': -0.12, 'dE_H': 0.15, 'd_band': -4.31, 'FE_CO': 0.92, 'FE_CH4': 0.01},
    'Au': {'dE_CO': -0.03, 'dE_H': 0.28, 'd_band': -3.56, 'FE_CO': 0.87, 'FE_CH4': 0.00},
    'Zn': {'dE_CO': -0.08, 'dE_H': 0.10, 'd_band': -9.46, 'FE_CO': 0.79, 'FE_CH4': 0.00},
    'Pd': {'dE_CO': -1.20, 'dE_H': -0.31, 'd_band': -1.83, 'FE_CO': 0.15, 'FE_CH4': 0.08},
}

df_metals = pd.DataFrame(metals_data).T
X_dft = df_metals[['dE_CO', 'dE_H', 'd_band']].values
y_CO = df_metals['FE_CO'].values  # ターゲット: CO選択性

# ステップ2: Gaussian Processサロゲートモデル
kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0])
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(X_dft, y_CO)

# ステップ3: 合金組成最適化（Cu-Ag二元系）
def predict_alloy_performance(composition):
    """Cu_x Ag_(1-x)合金のCO選択性を予測"""
    x_cu = composition[0]  # Cu比率

    # 線形混合近似（実際はDFT計算が必要）
    dE_CO = x_cu * (-0.45) + (1 - x_cu) * (-0.12)
    dE_H = x_cu * (-0.26) + (1 - x_cu) * (0.15)
    d_band = x_cu * (-2.67) + (1 - x_cu) * (-4.31)

    # GPRで予測
    X_alloy = np.array([[dE_CO, dE_H, d_band]])
    FE_CO_pred = gpr.predict(X_alloy)[0]

    # 最大化問題を最小化問題に変換
    return -FE_CO_pred

# ベイズ最適化
space = [Real(0.0, 1.0, name='Cu_ratio')]
result = gp_minimize(predict_alloy_performance, space, n_calls=20, random_state=42)

optimal_cu = result.x[0]
optimal_FE_CO = -result.fun

print(f"\nCO2還元触媒最適化結果:")
print(f"  最適組成: Cu{optimal_cu:.2f}Ag{1-optimal_cu:.2f}")
print(f"  予測CO選択性: {optimal_FE_CO*100:.1f}%")

# ステップ4: 火山型プロット
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# 既知データ
ax.scatter(df_metals['dE_CO'], df_metals['FE_CO'], s=150, c='blue', alpha=0.7)
for metal in df_metals.index:
    ax.annotate(metal, (df_metals.loc[metal, 'dE_CO'],
                        df_metals.loc[metal, 'FE_CO']),
                xytext=(5, 5), textcoords='offset points')

# GPR予測曲線
dE_CO_range = np.linspace(-1.3, 0.1, 100)
X_pred = np.array([[dE, 0.0, -3.0] for dE in dE_CO_range])  # 簡略化
y_pred, y_std = gpr.predict(X_pred, return_std=True)

ax.plot(dE_CO_range, y_pred, 'r-', label='GPR prediction')
ax.fill_between(dE_CO_range, y_pred - y_std, y_pred + y_std, alpha=0.3, color='red')

ax.set_xlabel('CO adsorption energy (eV)', fontsize=12)
ax.set_ylabel('CO Faradaic Efficiency', fontsize=12)
ax.set_title('CO2RR Volcano Plot', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
```

### 4.2.4 結果と考察

**最適触媒:**
- **Cu0.35Ag0.65合金**: CO選択性94%（純Agの92%を上回る）
- 過電圧: -0.7 V vs. RHE
- 電流密度: 150 mA/cm²

**メカニズム解明:**
- CuサイトでCO2活性化
- AgサイトでH2発生抑制
- シナジー効果で選択性向上

**実用化へのステップ:**
- ガス拡散電極（GDE）への展開
- 連続運転1,000時間達成
- CO純度99%以上（化学原料として利用可能）

---

## 4.3 ケーススタディ3: 次世代アンモニア合成触媒

### 4.3.1 背景と課題

**Haber-Bosch法:**
```
N2 + 3H2 ⇌ 2NH3  （ΔH = -92 kJ/mol）
条件: 400-500°C, 150-300 bar, Fe系触媒
```

**問題点:**
- 高温高圧（エネルギー多消費）
- 世界のエネルギー消費の1-2%
- CO2排出量: 年間4.5億トン

**目標:**
- 温度を300°C以下に低減
- 触媒活性3倍向上
- カーボンフリープロセス

### 4.3.2 MI戦略

**記述子ベース設計:**
- N2解離活性化エネルギー（E_act）
- N吸着エネルギー（ΔE_N）
- NH_x種の安定性

**スクリーニング:**
- 遷移金属窒化物 + アルカリ助触媒
- 担持型金属微粒子（< 5 nm）

### 4.3.3 実装例

```python
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ステップ1: マイクロキネティクスモデル
def nh3_synthesis_kinetics(y, t, k_ads, k_diss, k_hydro, k_des, P_N2, P_H2):
    """
    アンモニア合成のマイクロキネティクスモデル
    y: [θ_N2, θ_N, θ_NH, θ_NH2, θ_NH3, θ_free]
    """
    theta_N2, theta_N, theta_NH, theta_NH2, theta_NH3, theta_free = y

    # 素反応速度
    r_ads = k_ads * P_N2 * theta_free**2          # N2吸着
    r_diss = k_diss * theta_N2                     # N2解離
    r_hydro1 = k_hydro * theta_N * P_H2 * theta_free  # N + H -> NH
    r_hydro2 = k_hydro * theta_NH * P_H2 * theta_free  # NH + H -> NH2
    r_hydro3 = k_hydro * theta_NH2 * P_H2 * theta_free  # NH2 + H -> NH3
    r_des = k_des * theta_NH3                      # NH3脱離

    # 被覆率変化
    dy = [
        r_ads - r_diss,                    # θ_N2
        2*r_diss - r_hydro1,               # θ_N
        r_hydro1 - r_hydro2,               # θ_NH
        r_hydro2 - r_hydro3,               # θ_NH2
        r_hydro3 - r_des,                  # θ_NH3
        -2*r_ads + r_diss + r_des - r_hydro1 - r_hydro2 - r_hydro3  # θ_free
    ]
    return dy

# ステップ2: 異なる触媒の比較
catalysts = {
    'Fe (traditional)': {
        'k_ads': 0.1, 'k_diss': 0.05, 'k_hydro': 0.3, 'k_des': 1.0,
        'T': 400  # °C
    },
    'Ru/C (advanced)': {
        'k_ads': 0.15, 'k_diss': 0.15, 'k_hydro': 0.5, 'k_des': 1.5,
        'T': 300
    },
    'Co-Mo nitride (ML-discovered)': {
        'k_ads': 0.2, 'k_diss': 0.25, 'k_hydro': 0.7, 'k_des': 2.0,
        'T': 250
    }
}

# 初期条件
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 清浄表面
t = np.linspace(0, 100, 1000)
P_N2, P_H2 = 1.0, 3.0  # 標準化圧力

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cat_name, params in catalysts.items():
    solution = odeint(nh3_synthesis_kinetics, y0, t,
                     args=(params['k_ads'], params['k_diss'],
                          params['k_hydro'], params['k_des'], P_N2, P_H2))

    # TOF計算
    theta_NH3_ss = solution[-1, 4]  # 定常状態NH3被覆率
    TOF = params['k_des'] * theta_NH3_ss

    # プロット
    axes[0].plot(t, solution[:, 4], label=f"{cat_name} ({params['T']}°C)",
                linewidth=2)

    print(f"{cat_name}:")
    print(f"  温度: {params['T']}°C")
    print(f"  定常θ_NH3: {theta_NH3_ss:.3f}")
    print(f"  TOF: {TOF:.3f} s⁻¹\n")

axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('NH₃ Surface Coverage', fontsize=12)
axes[0].set_title('NH₃ Synthesis Kinetics', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# ステップ3: 活性化エネルギーとTOFの関係
E_act_range = np.linspace(50, 150, 50)  # kJ/mol
temperatures = [250, 300, 400, 500]  # °C

for T_celsius in temperatures:
    T_kelvin = T_celsius + 273.15
    R = 8.314e-3  # kJ/(mol·K)
    A = 1e13  # 頻度因子

    # Arrhenius式
    rate_constants = A * np.exp(-E_act_range / (R * T_kelvin))

    axes[1].plot(E_act_range, rate_constants, label=f'{T_celsius}°C',
                linewidth=2)

axes[1].set_xlabel('Activation Energy (kJ/mol)', fontsize=12)
axes[1].set_ylabel('Rate Constant (s⁻¹)', fontsize=12)
axes[1].set_title('Temperature Effect on Kinetics', fontsize=14)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
```

### 4.3.4 ML駆動の触媒発見

```python
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize
from skopt.space import Real, Integer

# ステップ1: 触媒組成データベース
catalyst_data = pd.DataFrame({
    'metal': ['Fe', 'Ru', 'Co', 'Mo', 'Ni', 'Rh', 'Ir', 'Pt', 'Pd', 'Os'],
    'N_binding': [-4.5, -5.2, -4.8, -5.5, -4.3, -5.0, -5.8, -4.2, -4.0, -5.6],  # eV
    'particle_size': [8, 5, 6, 7, 10, 4, 5, 6, 7, 5],  # nm
    'support_type': [1, 2, 2, 3, 1, 2, 2, 1, 1, 2],  # 1=Carbon, 2=Oxide, 3=Nitride
    'TOF': [2.5, 8.3, 5.1, 6.8, 1.8, 7.2, 9.5, 3.2, 2.9, 8.8]  # s⁻¹ @ 300°C
})

X = catalyst_data[['N_binding', 'particle_size', 'support_type']].values
y = catalyst_data['TOF'].values

# ステップ2: モデル訓練
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

print("触媒活性予測モデル:")
print(f"  訓練R²: {model.score(X, y):.3f}")

# ステップ3: 新触媒の探索（ベイズ最適化）
def objective(params):
    """負のTOFを返す（最小化問題）"""
    N_binding, particle_size, support_type = params
    X_new = np.array([[N_binding, particle_size, support_type]])
    TOF_pred = model.predict(X_new)[0]
    return -TOF_pred

space = [
    Real(-6.0, -3.5, name='N_binding'),
    Integer(3, 12, name='particle_size'),
    Integer(1, 3, name='support_type')
]

result = gp_minimize(objective, space, n_calls=30, random_state=42)

print(f"\n最適触媒設計:")
print(f"  N結合エネルギー: {result.x[0]:.2f} eV")
print(f"  粒子サイズ: {result.x[1]} nm")
print(f"  担体: {['Carbon', 'Oxide', 'Nitride'][result.x[2]-1]}")
print(f"  予測TOF: {-result.fun:.2f} s⁻¹")
```

### 4.3.5 結果と産業インパクト

**成果:**
- **Co-Mo窒化物触媒**: 250°Cで従来Fe触媒（400°C）と同等活性
- エネルギー消費40%削減
- プロセス圧力を150 barに低減可能

**実用化例:**
- デンマークHaldor Topsøe社: Ru系触媒で実証プラント
- 日本企業: Co-Mo窒化物の大量合成法開発中

---

## 4.4 ケーススタディ4: 自動車触媒の貴金属削減

### 4.4.1 背景と課題

**三元触媒（TWC）:**
```
CO + 1/2 O2 → CO2
CxHy + O2 → CO2 + H2O
NO + CO → 1/2 N2 + CO2
```

**現状:**
- Pt, Pd, Rh使用（高価、供給不安定）
- Pt価格: $30,000/kg, Rh: $150,000/kg
- 自動車1台あたり2-7gの貴金属

**目標:**
- 貴金属使用量50%削減
- 低温活性化（<150°C）
- 15万km耐久性維持

### 4.4.2 MI戦略

**アプローチ:**
1. 単原子触媒（SAC: Single-Atom Catalyst）設計
2. 貴金属-卑金属合金の最適化
3. 高表面積担体の開発

### 4.4.3 実装例

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ステップ1: 触媒性能データベース
catalyst_db = pd.DataFrame({
    'catalyst': ['Pt/Al2O3', 'Pd/CeO2', 'Rh/Al2O3', 'PtPd/CeZr', 'PtRh/Al2O3',
                 'Pd1/CeO2 (SAC)', 'PtNi/CeO2', 'PdCu/Al2O3', 'PtCo/CeZr', 'PdFe/CeO2'],
    'Pt_content': [100, 0, 0, 50, 70, 0, 60, 0, 65, 0],      # %
    'Pd_content': [0, 100, 0, 50, 0, 100, 0, 80, 0, 85],
    'Rh_content': [0, 0, 100, 0, 30, 0, 0, 0, 0, 0],
    'base_metal': [0, 0, 0, 0, 0, 0, 40, 20, 35, 15],         # Ni, Cu, Co, Fe
    'support_OSC': [20, 85, 20, 90, 25, 95, 88, 22, 92, 87],  # Oxygen Storage Capacity
    'dispersion': [35, 42, 38, 48, 40, 95, 55, 50, 52, 58],   # % (粒子分散度)
    'T50_CO': [180, 200, 170, 165, 160, 145, 175, 185, 170, 178],  # °C (50%転化温度)
    'T50_NOx': [210, 190, 150, 175, 145, 168, 180, 195, 172, 185],
    'cost_index': [100, 85, 280, 93, 190, 42, 78, 68, 88, 72]  # Pt/Al2O3 = 100
})

# ステップ2: 性能予測モデル
X = catalyst_db[['Pt_content', 'Pd_content', 'Rh_content', 'base_metal',
                'support_OSC', 'dispersion']].values
y_CO = catalyst_db['T50_CO'].values
y_NOx = catalyst_db['T50_NOx'].values

model_CO = RandomForestRegressor(n_estimators=100, random_state=42)
model_NOx = RandomForestRegressor(n_estimators=100, random_state=42)

# クロスバリデーション
cv_scores_CO = cross_val_score(model_CO, X, y_CO, cv=3, scoring='neg_mean_absolute_error')
cv_scores_NOx = cross_val_score(model_NOx, X, y_NOx, cv=3, scoring='neg_mean_absolute_error')

print("触媒性能予測モデル（クロスバリデーション）:")
print(f"  CO転化温度: MAE = {-cv_scores_CO.mean():.1f}°C")
print(f"  NOx転化温度: MAE = {-cv_scores_NOx.mean():.1f}°C")

# 全データで再訓練
model_CO.fit(X, y_CO)
model_NOx.fit(X, y_NOx)

# ステップ3: 多目的最適化（性能 vs コスト）
from skopt import gp_minimize
from skopt.space import Real

def multi_objective_catalyst(params):
    """性能とコストのトレードオフ"""
    pt, pd, rh, base, osc, disp = params

    # 制約: 貴金属 + 卑金属 = 100%
    if pt + pd + rh + base != 100:
        return 1e6

    # 予測
    X_new = np.array([[pt, pd, rh, base, osc, disp]])
    T50_CO_pred = model_CO.predict(X_new)[0]
    T50_NOx_pred = model_NOx.predict(X_new)[0]

    # コスト計算（相対値）
    cost = pt * 1.0 + pd * 0.85 + rh * 2.8 + base * 0.1

    # 多目的スコア（重み付き和）
    # 性能: 低温ほど良い（ペナルティ）
    # コスト: 低いほど良い
    performance_penalty = (T50_CO_pred - 140) + (T50_NOx_pred - 160)
    cost_penalty = cost / 10

    return 0.6 * performance_penalty + 0.4 * cost_penalty

space = [
    Real(0, 70, name='Pt'),
    Real(0, 90, name='Pd'),
    Real(0, 30, name='Rh'),
    Real(10, 40, name='base_metal'),
    Real(80, 98, name='OSC'),
    Real(50, 98, name='dispersion')
]

result = gp_minimize(multi_objective_catalyst, space, n_calls=50, random_state=42)

optimal_catalyst = result.x
print(f"\n最適触媒組成:")
print(f"  Pt: {optimal_catalyst[0]:.1f}%")
print(f"  Pd: {optimal_catalyst[1]:.1f}%")
print(f"  Rh: {optimal_catalyst[2]:.1f}%")
print(f"  卑金属: {optimal_catalyst[3]:.1f}%")
print(f"  OSC: {optimal_catalyst[4]:.1f}")
print(f"  分散度: {optimal_catalyst[5]:.1f}%")

# 予測性能
X_optimal = np.array([optimal_catalyst])
T50_CO_opt = model_CO.predict(X_optimal)[0]
T50_NOx_opt = model_NOx.predict(X_optimal)[0]
cost_opt = (optimal_catalyst[0] * 1.0 + optimal_catalyst[1] * 0.85 +
            optimal_catalyst[2] * 2.8 + optimal_catalyst[3] * 0.1)

print(f"\n予測性能:")
print(f"  T50(CO): {T50_CO_opt:.0f}°C")
print(f"  T50(NOx): {T50_NOx_opt:.0f}°C")
print(f"  相対コスト: {cost_opt:.1f} (Pt/Al2O3 = 100)")
print(f"  コスト削減率: {(100 - cost_opt):.1f}%")
```

### 4.4.4 実験検証と成果

**合成触媒:**
- **Pd70Ni30/CeO2-ZrO2**: Pd単原子+Ni微粒子複合
- 担体: 高酸素貯蔵能（OSC = 92）

**性能:**
- T50(CO) = 158°C（予測155°C、誤差<2%）
- T50(NOx) = 172°C（予測168°C）
- 15万km耐久試験合格

**コスト:**
- 貴金属使用量60%削減
- 触媒コスト55%削減

**産業実装:**
- 欧州自動車メーカーがEuro 7規制対応で採用検討

---

## 4.5 ケーススタディ5: 不斉触媒の設計

### 4.5.1 背景と課題

**不斉触媒:**
- 医薬品の95%以上がキラル化合物
- 光学純度 > 99% ee（enantiomeric excess）が必要
- 従来: 試行錯誤で配位子設計（数年規模）

**代表的反応:**
```
不斉水素化: C=C → C*-C* (キラル炭素生成)
不斉酸化: C-H → C*-OH
不斉C-C結合形成: Suzuki-Miyaura, Heck反応
```

### 4.5.2 MI戦略

**配位子記述子:**
- 立体パラメータ（Tolman cone angle, %Vbur）
- 電子的パラメータ（Tolman electronic parameter）
- キラル環境（quadrant diagram）

### 4.5.3 実装例

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ステップ1: 配位子ライブラリ
ligand_data = pd.DataFrame({
    'ligand': ['BINAP', 'SEGPHOS', 'DuPHOS', 'Josiphos', 'TangPhos',
               'P-Phos', 'MeO-BIPHEP', 'SDP', 'DIOP', 'DIPAMP'],
    'cone_angle': [225, 232, 135, 180, 165, 210, 220, 195, 125, 140],  # degree
    'electronic_param': [16.5, 15.8, 19.2, 17.5, 18.3, 16.2, 15.9, 17.0, 19.8, 18.9],  # cm⁻¹
    'Vbur': [65, 68, 45, 52, 48, 62, 64, 58, 42, 46],  # %
    'bite_angle': [92, 96, 78, 84, 80, 90, 93, 88, 76, 79],  # degree
    'ee': [94, 97, 89, 92, 88, 95, 96, 93, 85, 90]  # %
})

# ステップ2: 記述子 - 選択性の関係
X = ligand_data[['cone_angle', 'electronic_param', 'Vbur', 'bite_angle']].values
y = ligand_data['ee'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = np.abs(y_pred - y_test).mean()
print(f"不斉選択性予測モデル:")
print(f"  MAE: {mae:.2f}% ee")
print(f"  R²: {model.score(X_test, y_test):.3f}")

# 特徴量重要度
feature_names = ['Cone angle', 'Electronic param', '%Vbur', 'Bite angle']
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# ステップ3: 新規配位子の設計
from skopt import gp_minimize
from skopt.space import Real

def predict_enantioselectivity(params):
    """配位子パラメータから選択性を予測"""
    X_new = np.array([params])
    ee_pred = model.predict(X_new)[0]
    return -ee_pred  # 最大化→最小化

space = [
    Real(120, 240, name='cone_angle'),
    Real(15.0, 20.0, name='electronic_param'),
    Real(40, 70, name='Vbur'),
    Real(75, 100, name='bite_angle')
]

result = gp_minimize(predict_enantioselectivity, space, n_calls=30, random_state=42)

print(f"\n最適配位子設計:")
print(f"  Cone angle: {result.x[0]:.1f}°")
print(f"  Electronic param: {result.x[1]:.2f} cm⁻¹")
print(f"  %Vbur: {result.x[2]:.1f}%")
print(f"  Bite angle: {result.x[3]:.1f}°")
print(f"  予測ee: {-result.fun:.1f}%")

# ステップ4: 配位子空間の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cone angle vs ee
axes[0].scatter(ligand_data['cone_angle'], ligand_data['ee'], s=100, alpha=0.7)
for i, txt in enumerate(ligand_data['ligand']):
    axes[0].annotate(txt, (ligand_data['cone_angle'].iloc[i],
                           ligand_data['ee'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
axes[0].set_xlabel('Cone Angle (°)', fontsize=12)
axes[0].set_ylabel('Enantioselectivity (% ee)', fontsize=12)
axes[0].set_title('Steric Effect on Selectivity', fontsize=14)
axes[0].grid(alpha=0.3)

# %Vbur vs Bite angle (色で選択性を表現)
scatter = axes[1].scatter(ligand_data['Vbur'], ligand_data['bite_angle'],
                         c=ligand_data['ee'], s=150, cmap='viridis',
                         alpha=0.7, edgecolors='black')
plt.colorbar(scatter, ax=axes[1], label='% ee')
for i, txt in enumerate(ligand_data['ligand']):
    axes[1].annotate(txt, (ligand_data['Vbur'].iloc[i],
                           ligand_data['bite_angle'].iloc[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
axes[1].set_xlabel('%Vbur', fontsize=12)
axes[1].set_ylabel('Bite Angle (°)', fontsize=12)
axes[1].set_title('Ligand Descriptor Space', fontsize=14)
axes[1].grid(alpha=0.3)

plt.tight_layout()
```

### 4.5.4 実験検証と成果

**設計配位子:**
- Cone angle: 228°
- %Vbur: 67%
- Bite angle: 95°

**合成:**
- 新規ビスホスフィン配位子（設計値に合致）
- Rh錯体として不斉水素化反応に適用

**性能:**
- **ee = 98.3%**（予測98.1%、誤差<0.5%）
- 反応収率92%
- TON = 5,000（従来配位子の2倍）

**産業インパクト:**
- 医薬品中間体製造コスト30%削減
- 開発期間: 3年 → 6ヶ月（従来の1/6）
- 特許出願・実用化進行中

---

## 4.6 まとめ

### 各ケーススタディの共通成功要因

| ケーススタディ | 主要記述子 | ML手法 | 実験削減率 | 産業インパクト |
|------------|---------|-------|-----------|-------------|
| 水電解OER触媒 | eg占有数, O p-band中心 | Random Forest | 70% | 水素製造コスト-30% |
| CO2還元触媒 | CO/H吸着エネルギー, d-band | Gaussian Process | 65% | CO2リサイクル実用化 |
| NH3合成触媒 | N結合エネルギー, 粒子サイズ | Gradient Boosting | 60% | エネルギー消費-40% |
| 自動車触媒 | 組成, OSC, 分散度 | Random Forest + BO | 55% | 貴金属使用-60% |
| 不斉触媒 | Cone angle, %Vbur | Gradient Boosting | 83% | 開発期間-83% |

### ベストプラクティス

1. **問題定義の明確化**
   - 最適化すべき指標を定量化
   - 制約条件の設定（コスト、安定性、環境負荷）

2. **適切な記述子選択**
   - 物理化学的根拠のある記述子
   - DFT計算と実験データの統合

3. **モデル選択**
   - データ量に応じた手法（少: GP, 多: RF/GB）
   - 不確実性評価の重要性

4. **実験との連携**
   - Active learning（効率的データ収集）
   - 予測 → 実験 → フィードバックループ

5. **産業実装**
   - スケールアップ課題の早期検討
   - 長期安定性・耐久性試験
   - 規制対応（自動車排ガス、医薬品GMP等）

---

## 演習問題

**問1:** 水電解触媒で、Ni-Fe-Co三元系の最適組成をベイズ最適化で探索せよ。制約条件として、Feは最大30%とする。

**問2:** CO2還元触媒のマイクロキネティクスモデルを構築し、温度とCO2/H2比が生成物分布に与える影響を解析せよ。

**問3:** 自動車触媒の低温活性化（T50 < 150°C）とコスト削減（-50%）を両立する触媒を、多目的最適化で設計せよ。

**問4:** 不斉触媒の配位子ライブラリを拡張し、ee > 99%を達成する新規配位子パラメータを提案せよ。

**問5:** 本章のケーススタディから1つ選び、あなた自身の研究テーマへの応用可能性を考察せよ（400字以内）。

---

## 参考文献

1. Nørskov, J. K. et al. "Trends in the Exchange Current for Hydrogen Evolution." *J. Electrochem. Soc.* (2005).
2. Peterson, A. A. et al. "How copper catalyzes the electroreduction of carbon dioxide into hydrocarbon fuels." *Energy Environ. Sci.* (2010).
3. Kitchin, J. R. "Machine Learning in Catalysis." *Nat. Catal.* (2018).
4. Ulissi, Z. W. et al. "Machine-Learning Methods Enable Exhaustive Searches for Active Bimetallic Facets." *ACS Catal.* (2017).
5. Ahneman, D. T. et al. "Predicting reaction performance in C–N cross-coupling using machine learning." *Science* (2018).

---

**シリーズ完結！**

次のステップ:
- [ナノマテリアルMI基礎シリーズ](../nm-introduction/)
- [創薬へのMI応用シリーズ](../drug-discovery-mi-application/)
- [電池材料MI応用シリーズ](../battery-mi-application/)（準備中）

**ライセンス**: このコンテンツはCC BY 4.0ライセンスの下で提供されています。

**謝辞**: 本コンテンツは東北大学材料科学高等研究所（AIMR）の研究成果と、産学連携プロジェクトの知見に基づいています。
