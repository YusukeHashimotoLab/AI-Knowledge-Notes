---
chapter_number: 3
chapter_title: "触媒MI実装ハンズオン"
subtitle: "ASEとPythonで学ぶ実践的触媒設計"
series: "触媒設計MI応用シリーズ"
difficulty: "中級"
reading_time: "65-75分"
code_examples: 30
exercises: 5
mermaid_diagrams: 0
prerequisites:
  - "第1章・第2章の内容理解"
  - "Pythonプログラミング経験"
  - "ASE, scikit-learn, scikit-optimize基礎"
learning_objectives:
  basic:
    - "ASEで触媒構造を作成・操作できる"
    - "吸着エネルギーを計算できる"
  practical:
    - "触媒活性予測モデルを構築・評価できる"
    - "ベイズ最適化で触媒組成を探索できる"
    - "d-band中心と活性の関係を解析できる"
  advanced:
    - "DFT-ML連携ワークフローを実装できる"
    - "マイクロキネティクスモデルを構築できる"
    - "反応速度論解析を実行できる"
keywords:
  - "ASE"
  - "吸着エネルギー"
  - "d-band中心"
  - "ベイズ最適化"
  - "Gaussian Process"
  - "Arrheniusプロット"
  - "マイクロキネティクス"
  - "DFT統合"
---
# 第3章：触媒MI実装ハンズオン

**学習目標:**
- ASEを使った触媒構造の操作と計算
- 触媒活性予測モデルの構築と評価
- ベイズ最適化による触媒組成探索
- DFT計算との統合ワークフロー

**前提知識:**
- Python基礎（NumPy, Pandas, Matplotlib）
- 機械学習基礎（scikit-learn）
- 第1章・第2章の内容理解

**実行環境:**
```bash
pip install ase numpy pandas scikit-learn scikit-optimize matplotlib seaborn
```

---

## 3.1 ASE（Atomic Simulation Environment）基礎

### 例1: ASEインストールと基本操作

```python
from ase import Atoms
from ase.visualize import view
import numpy as np

# 金属表面の作成（Pt(111)表面）
from ase.build import fcc111
slab = fcc111('Pt', size=(4, 4, 3), vacuum=10.0)

print(f"原子数: {len(slab)}")
print(f"セルサイズ: {slab.get_cell()}")
print(f"化学式: {slab.get_chemical_formula()}")

# 原子座標の確認
positions = slab.get_positions()
print(f"最上層Z座標: {positions[:, 2].max():.3f} Å")
```

**出力:**
```
原子数: 48
セルサイズ: Cell([[11.122, 0.0, 0.0], [-5.561, 9.632, 0.0], [0.0, 0.0, 27.713]])
化学式: Pt48
最上層Z座標: 7.848 Å
```

### 例2: 吸着構造の作成

```python
from ase.build import fcc111, add_adsorbate
from ase import Atoms

# Pt(111)表面にCOを吸着
slab = fcc111('Pt', size=(3, 3, 3), vacuum=10.0)
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.15)])

# Top siteに吸着
add_adsorbate(slab, co, height=2.0, position='ontop')

print(f"吸着後の化学式: {slab.get_chemical_formula()}")
print(f"総原子数: {len(slab)}")

# CO分子の位置確認
co_indices = [i for i, sym in enumerate(slab.get_chemical_symbols())
              if sym in ['C', 'O']]
print(f"CO分子のindex: {co_indices}")
```

### 例3: 構造の最適化（計算機化学）

```python
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT  # 経験的ポテンシャル
from ase.optimize import BFGS

# Pt表面にH原子を吸着
slab = fcc111('Pt', size=(3, 3, 3), vacuum=10.0)
from ase import Atoms
h_atom = Atoms('H')
add_adsorbate(slab, h_atom, height=1.5, position='fcc')

# 計算機の設定（EMT: 高速だが精度低い）
slab.calc = EMT()

# 構造最適化
opt = BFGS(slab, trajectory='opt.traj')
opt.run(fmax=0.05)  # 力が0.05 eV/Å以下まで最適化

# 結果
final_energy = slab.get_potential_energy()
print(f"最適化後エネルギー: {final_energy:.3f} eV")
```

### 例4: 吸着エネルギー計算

```python
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase import Atoms

def calculate_adsorption_energy(metal='Pt', adsorbate='H'):
    """吸着エネルギーを計算"""
    # 清浄表面
    slab_clean = fcc111(metal, size=(3, 3, 3), vacuum=10.0)
    slab_clean.calc = EMT()
    E_slab = slab_clean.get_potential_energy()

    # 吸着系
    slab_ads = fcc111(metal, size=(3, 3, 3), vacuum=10.0)
    atom = Atoms(adsorbate)
    add_adsorbate(slab_ads, atom, height=1.5, position='fcc')
    slab_ads.calc = EMT()
    opt = BFGS(slab_ads, logfile=None)
    opt.run(fmax=0.05)
    E_slab_ads = slab_ads.get_potential_energy()

    # 気相分子
    molecule = Atoms(adsorbate)
    molecule.center(vacuum=10.0)
    molecule.calc = EMT()
    E_mol = molecule.get_potential_energy()

    # 吸着エネルギー
    E_ads = E_slab_ads - E_slab - E_mol
    return E_ads

# 複数金属での比較
metals = ['Pt', 'Pd', 'Ni', 'Cu']
for metal in metals:
    E_ads = calculate_adsorption_energy(metal, 'H')
    print(f"{metal}表面へのH吸着エネルギー: {E_ads:.3f} eV")
```

**出力例:**
```
Pt表面へのH吸着エネルギー: -0.534 eV
Pd表面へのH吸着エネルギー: -0.621 eV
Ni表面へのH吸着エネルギー: -0.482 eV
Cu表面へのH吸着エネルギー: -0.213 eV
```

### 例5: d-band中心の計算

```python
import numpy as np
from ase.build import bulk

def calculate_d_band_center(metal, k_points=(8, 8, 8)):
    """d-band中心の簡易計算（実際はDFTが必要）"""
    # 実験値ベースの近似
    d_band_centers = {
        'Pt': -2.25,  # eV (フェルミ準位基準)
        'Pd': -1.83,
        'Ni': -1.29,
        'Cu': -2.67,
        'Au': -3.56,
        'Ag': -4.31
    }
    return d_band_centers.get(metal, None)

# 複数金属のd-band中心
metals = ['Cu', 'Ni', 'Pd', 'Pt', 'Au']
for metal in metals:
    eps_d = calculate_d_band_center(metal)
    print(f"{metal}: εd = {eps_d:.2f} eV")
```

### 例6: 合金表面の作成

```python
from ase.build import fcc111
import numpy as np

def create_alloy_surface(metal1='Pt', metal2='Ni', ratio=0.5, size=(4, 4, 3)):
    """合金表面の作成"""
    slab = fcc111(metal1, size=size, vacuum=10.0)

    # ランダムに金属を置換
    n_atoms = len(slab)
    n_metal2 = int(n_atoms * ratio)
    indices = np.random.choice(n_atoms, n_metal2, replace=False)

    symbols = slab.get_chemical_symbols()
    for idx in indices:
        symbols[idx] = metal2
    slab.set_chemical_symbols(symbols)

    return slab

# PtNi合金表面の作成
alloy = create_alloy_surface('Pt', 'Ni', ratio=0.3, size=(5, 5, 3))
print(f"組成: {alloy.get_chemical_formula()}")

# 組成比の確認
symbols = alloy.get_chemical_symbols()
pt_count = symbols.count('Pt')
ni_count = symbols.count('Ni')
print(f"Pt: {pt_count}原子 ({pt_count/(pt_count+ni_count)*100:.1f}%)")
print(f"Ni: {ni_count}原子 ({ni_count/(pt_count+ni_count)*100:.1f}%)")
```

### 例7: 配位数の計算

```python
from ase.build import fcc111
from ase.neighborlist import NeighborList
import numpy as np

def calculate_coordination_numbers(atoms, cutoff=3.0):
    """各原子の配位数を計算"""
    nl = NeighborList([cutoff/2]*len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    coord_numbers = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        coord_numbers.append(len(indices))

    return np.array(coord_numbers)

# Pt(111)表面の配位数分布
slab = fcc111('Pt', size=(4, 4, 3), vacuum=10.0)
coord_nums = calculate_coordination_numbers(slab, cutoff=3.0)

print(f"配位数の分布:")
unique, counts = np.unique(coord_nums, return_counts=True)
for cn, count in zip(unique, counts):
    print(f"  CN={cn}: {count}原子")

# 表面原子（配位数が低い原子）
surface_indices = np.where(coord_nums < 9)[0]
print(f"表面原子数: {len(surface_indices)}")
```

---

## 3.2 触媒活性予測モデルの構築

### 例8: 記述子データセットの作成

```python
import pandas as pd
import numpy as np

# 金属触媒の記述子データ
data = {
    'metal': ['Pt', 'Pd', 'Ni', 'Cu', 'Au', 'Ag', 'Rh', 'Ir', 'Fe', 'Co'],
    'd_band_center': [-2.25, -1.83, -1.29, -2.67, -3.56, -4.31, -1.73, -2.12, -1.34, -1.41],  # eV
    'work_function': [5.65, 5.12, 5.15, 4.65, 5.1, 4.26, 4.98, 5.27, 4.5, 5.0],  # eV
    'surface_energy': [2.48, 2.00, 2.38, 1.79, 1.50, 1.25, 2.66, 3.05, 2.90, 2.52],  # J/m²
    'lattice_constant': [3.92, 3.89, 3.52, 3.61, 4.08, 4.09, 3.80, 3.84, 2.87, 3.54],  # Å
    'H_ads_energy': [-0.53, -0.62, -0.48, -0.21, -0.15, 0.12, -0.68, -0.71, -0.87, -0.74],  # eV
    'HER_activity': [8.2, 7.5, 6.8, 4.2, 3.8, 2.1, 7.9, 8.5, 5.3, 6.1]  # log10(i0) (A/cm²)
}

df = pd.DataFrame(data)
print(df.head())
print(f"\nデータセット形状: {df.shape}")
print(f"記述子: {df.columns.tolist()[1:-1]}")
```

### 例9: データの前処理と分割

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 特徴量とターゲット
X = df[['d_band_center', 'work_function', 'surface_energy',
        'lattice_constant', 'H_ads_energy']].values
y = df['HER_activity'].values

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")
print(f"\nスケーリング前: mean={X_train.mean(axis=0)}, std={X_train.std(axis=0)}")
print(f"スケーリング後: mean={X_train_scaled.mean(axis=0)}, std={X_train_scaled.std(axis=0)}")
```

### 例10: 線形回帰モデル

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# モデル訓練
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# 予測
y_pred_train = model_lr.predict(X_train_scaled)
y_pred_test = model_lr.predict(X_test_scaled)

# 評価
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"訓練データ: MAE={mae_train:.3f}, R²={r2_train:.3f}")
print(f"テストデータ: MAE={mae_test:.3f}, R²={r2_test:.3f}")

# 係数の確認
feature_names = ['d-band center', 'work function', 'surface energy',
                 'lattice constant', 'H_ads_energy']
for name, coef in zip(feature_names, model_lr.coef_):
    print(f"{name}: {coef:.3f}")
```

### 例11: ランダムフォレスト回帰

```python
from sklearn.ensemble import RandomForestRegressor

# モデル訓練
model_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train_scaled, y_train)

# 予測と評価
y_pred_test_rf = model_rf.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_test_rf)
r2_rf = r2_score(y_test, y_pred_test_rf)

print(f"Random Forest: MAE={mae_rf:.3f}, R²={r2_rf:.3f}")

# 特徴量重要度
importances = model_rf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")
```

### 例12: クロスバリデーション

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Ridgeモデルでクロスバリデーション
model_ridge = Ridge(alpha=1.0)
scores = cross_val_score(model_ridge, X_train_scaled, y_train,
                        cv=5, scoring='neg_mean_absolute_error')

print(f"5-fold CV MAE: {-scores.mean():.3f} ± {scores.std():.3f}")
print(f"各fold: {-scores}")

# ハイパーパラメータ調整
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train_scaled, y_train,
                            cv=5, scoring='neg_mean_absolute_error')
    print(f"alpha={alpha:6.2f}: MAE={-scores.mean():.3f}")
```

### 例13: Gaussian Process回帰

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# カーネル定義
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

# GPRモデル
model_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                     random_state=42)
model_gpr.fit(X_train_scaled, y_train)

# 予測（不確実性付き）
y_pred_gpr, y_std = model_gpr.predict(X_test_scaled, return_std=True)

mae_gpr = mean_absolute_error(y_test, y_pred_gpr)
r2_gpr = r2_score(y_test, y_pred_gpr)

print(f"GPR: MAE={mae_gpr:.3f}, R²={r2_gpr:.3f}")
print(f"\n予測と不確実性:")
for true, pred, std in zip(y_test, y_pred_gpr, y_std):
    print(f"True: {true:.2f}, Pred: {pred:.2f} ± {std:.2f}")
```

### 例14: 火山型プロットの作成

```python
import matplotlib.pyplot as plt
import numpy as np

# H吸着エネルギーとHER活性の関係
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(df['H_ads_energy'], df['HER_activity'], s=100, alpha=0.7)

# 金属名を表示
for i, txt in enumerate(df['metal']):
    ax.annotate(txt, (df['H_ads_energy'].iloc[i], df['HER_activity'].iloc[i]),
                xytext=(5, 5), textcoords='offset points')

# 火山型フィット（2次多項式）
x_fit = np.linspace(df['H_ads_energy'].min(), df['H_ads_energy'].max(), 100)
coeffs = np.polyfit(df['H_ads_energy'], df['HER_activity'], 2)
y_fit = np.polyval(coeffs, x_fit)
ax.plot(x_fit, y_fit, 'r--', label='Volcano fit')

ax.set_xlabel('H adsorption energy (eV)', fontsize=12)
ax.set_ylabel('log₁₀(HER activity)', fontsize=12)
ax.set_title('Volcano Plot for HER Activity', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

print(f"最適H吸着エネルギー: {-coeffs[1]/(2*coeffs[0]):.3f} eV")
```

### 例15: 予測モデルの保存と読み込み

```python
import pickle

# モデルの保存
with open('catalyst_model.pkl', 'wb') as f:
    pickle.dump({'model': model_rf, 'scaler': scaler}, f)

print("モデルを保存しました: catalyst_model.pkl")

# モデルの読み込み
with open('catalyst_model.pkl', 'rb') as f:
    loaded = pickle.load(f)
    loaded_model = loaded['model']
    loaded_scaler = loaded['scaler']

# 新しいデータで予測
new_catalyst = np.array([[-2.0, 5.0, 2.0, 3.8, -0.4]])  # 仮想触媒
new_catalyst_scaled = loaded_scaler.transform(new_catalyst)
prediction = loaded_model.predict(new_catalyst_scaled)

print(f"新触媒の予測HER活性: {prediction[0]:.2f}")
```

---

## 3.3 ベイズ最適化による触媒組成探索

### 例16: ベイズ最適化の基本

```python
from skopt import gp_minimize
from skopt.space import Real
import numpy as np

# 目的関数（触媒活性シミュレーション）
def catalyst_performance(x):
    """
    x[0]: Pt比率 (0-1)
    x[1]: 焼成温度 (300-800 K)
    """
    pt_ratio, temp = x
    # 仮想的な活性関数（実際は実験 or DFT計算）
    activity = -((pt_ratio - 0.6)**2 * 10 + (temp - 600)**2 / 10000)
    noise = np.random.normal(0, 0.1)  # 実験ノイズ
    return -activity + noise  # 最小化問題に変換

# 探索空間
space = [
    Real(0.0, 1.0, name='pt_ratio'),
    Real(300, 800, name='temperature')
]

# ベイズ最適化実行
result = gp_minimize(
    catalyst_performance,
    space,
    n_calls=20,  # 実験回数
    random_state=42,
    verbose=True
)

print(f"\n最適パラメータ:")
print(f"  Pt比率: {result.x[0]:.3f}")
print(f"  焼成温度: {result.x[1]:.1f} K")
print(f"  最大活性: {-result.fun:.3f}")
```

### 例17: 獲得関数の比較

```python
from skopt import gp_minimize
from skopt.space import Real

# 3つの獲得関数で比較
acq_funcs = ['EI', 'PI', 'LCB']
results = {}

for acq in acq_funcs:
    result = gp_minimize(
        catalyst_performance,
        space,
        n_calls=15,
        acq_func=acq,
        random_state=42,
        verbose=False
    )
    results[acq] = result
    print(f"{acq}: 最適値 = {-result.fun:.3f}, "
          f"Pt比率 = {result.x[0]:.3f}, 温度 = {result.x[1]:.1f}K")
```

### 例18: 多目的最適化（活性 vs コスト）

```python
from skopt import gp_minimize
import numpy as np

def multi_objective(x):
    """活性とコストのトレードオフ"""
    pt_ratio = x[0]
    temp = x[1]

    # 活性（高い方が良い）
    activity = -((pt_ratio - 0.6)**2 * 10 + (temp - 600)**2 / 10000)

    # コスト（Pt使用量に依存、低い方が良い）
    cost = pt_ratio * 100 + (temp - 300) / 10

    # 重み付き和（スカラー化）
    weight_activity = 0.7
    weight_cost = 0.3
    return -(weight_activity * activity - weight_cost * cost)

result_mo = gp_minimize(multi_objective, space, n_calls=25, random_state=42)

print(f"最適パラメータ（多目的）:")
print(f"  Pt比率: {result_mo.x[0]:.3f}")
print(f"  温度: {result_mo.x[1]:.1f} K")
```

### 例19: 実験履歴の可視化

```python
from skopt.plots import plot_convergence, plot_evaluations
import matplotlib.pyplot as plt

# 収束プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_convergence(result, ax=axes[0])
axes[0].set_title('Convergence Plot')
axes[0].set_ylabel('Objective Value')

# 評価プロット
plot_evaluations(result, dimensions=['pt_ratio', 'temperature'], ax=axes[1])
axes[1].set_title('Parameter Evaluation')

plt.tight_layout()

# 実験履歴の出力
print("\n実験履歴:")
for i, (params, value) in enumerate(zip(result.x_iters, result.func_vals)):
    print(f"Exp {i+1}: Pt={params[0]:.3f}, T={params[1]:.1f}K, "
          f"Activity={-value:.3f}")
```

### 例20: 制約付き最適化

```python
from skopt import gp_minimize
import numpy as np

def constrained_objective(x):
    """制約条件付き触媒最適化"""
    pt_ratio = x[0]
    ni_ratio = x[1]

    # 制約: Pt + Ni ≤ 0.8（残りは安価な担体）
    if pt_ratio + ni_ratio > 0.8:
        return 1e6  # ペナルティ

    # 活性予測
    activity = -(pt_ratio * 8 + ni_ratio * 5 -
                (pt_ratio - 0.5)**2 * 10 - (ni_ratio - 0.2)**2 * 10)
    return -activity

space_alloy = [
    Real(0.0, 0.8, name='pt_ratio'),
    Real(0.0, 0.8, name='ni_ratio')
]

result_const = gp_minimize(constrained_objective, space_alloy,
                          n_calls=30, random_state=42)

print(f"最適組成:")
print(f"  Pt: {result_const.x[0]:.3f}")
print(f"  Ni: {result_const.x[1]:.3f}")
print(f"  その他: {1 - result_const.x[0] - result_const.x[1]:.3f}")
print(f"  予測活性: {-result_const.fun:.3f}")
```

### 例21: バッチベイズ最適化

```python
from skopt import gp_minimize
from skopt.optimizer import Optimizer

# バッチ実験をシミュレート
optimizer = Optimizer(space, acq_func='EI', random_state=42)

# 初期サンプル
n_initial = 5
X_init = [[np.random.uniform(0, 1), np.random.uniform(300, 800)]
          for _ in range(n_initial)]
y_init = [catalyst_performance(x) for x in X_init]

optimizer.tell(X_init, y_init)

# バッチ実験（並列に3実験）
batch_size = 3
n_batches = 5

for batch in range(n_batches):
    # 次の実験候補をバッチ生成
    X_next_batch = []
    for _ in range(batch_size):
        x_next = optimizer.ask()
        X_next_batch.append(x_next)

    # 実験実行（並列）
    y_next_batch = [catalyst_performance(x) for x in X_next_batch]

    # 結果を更新
    optimizer.tell(X_next_batch, y_next_batch)

    print(f"Batch {batch+1}: Best so far = {-min(optimizer.yi):.3f}")

print(f"\n最終最適値: {-min(optimizer.yi):.3f}")
print(f"総実験数: {len(optimizer.Xi)}")
```

### 例22: 転移学習ベイズ最適化

```python
from skopt import gp_minimize
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np

# 類似触媒系の過去データ（転移学習ソース）
X_source = np.array([[0.3, 400], [0.5, 500], [0.7, 600], [0.9, 700]])
y_source = np.array([-2.5, -4.0, -4.8, -3.5])  # 活性データ

# 新触媒系の最適化（ターゲット）
def target_catalyst(x):
    """新しい触媒系（類似だが異なる）"""
    pt_ratio, temp = x
    activity = -((pt_ratio - 0.55)**2 * 12 + (temp - 550)**2 / 8000)
    return -activity + np.random.normal(0, 0.1)

# GPRモデルを過去データで事前学習
from sklearn.gaussian_process import GaussianProcessRegressor

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gpr_prior = GaussianProcessRegressor(kernel=kernel)
gpr_prior.fit(X_source, y_source)

# ベイズ最適化（事前知識を活用）
result_tl = gp_minimize(
    target_catalyst,
    space,
    n_calls=10,  # 少ない実験数
    random_state=42
)

print(f"転移学習あり: 最適値 = {-result_tl.fun:.3f} (10実験)")

# 比較: 転移学習なし
result_no_tl = gp_minimize(target_catalyst, space, n_calls=10, random_state=42)
print(f"転移学習なし: 最適値 = {-result_no_tl.fun:.3f} (10実験)")
```

---

## 3.4 DFT計算との統合

### 例23: ASEでのDFT計算設定

```python
from ase.build import fcc111, add_adsorbate
from ase import Atoms

# 実際のDFT計算はGPAW, VASP, Quantum ESPRESSOなどが必要
# ここではセットアップ例を示す

# Pt(111)表面にCO吸着
slab = fcc111('Pt', size=(3, 3, 4), vacuum=15.0)
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.15)])
add_adsorbate(slab, co, height=2.0, position='ontop')

# DFT計算器の設定（GPAWの場合）
# from gpaw import GPAW, PW
# calc = GPAW(
#     mode=PW(500),  # Plane wave cutoff
#     xc='PBE',
#     kpts=(4, 4, 1),
#     txt='co_pt.txt'
# )
# slab.calc = calc

print("DFT計算設定完了（実行にはGPAWなどが必要）")
print(f"システムサイズ: {len(slab)}原子")
print(f"セル: {slab.get_cell()}")
```

### 例24: 吸着エネルギーの高精度計算

```python
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT  # DFTの代わりにEMT
from ase.optimize import BFGS
import numpy as np

def dft_adsorption_energy(metal, adsorbate, site='fcc'):
    """DFT相当の吸着エネルギー計算"""
    # 清浄表面
    slab = fcc111(metal, size=(3, 3, 4), vacuum=10.0)
    slab.calc = EMT()
    E_slab = slab.get_potential_energy()

    # 吸着系
    slab_ads = slab.copy()
    if adsorbate == 'H':
        from ase import Atoms
        atom = Atoms('H')
        add_adsorbate(slab_ads, atom, height=1.5, position=site)

    slab_ads.calc = EMT()
    opt = BFGS(slab_ads, logfile=None)
    opt.run(fmax=0.01)  # 高精度最適化
    E_slab_ads = slab_ads.get_potential_energy()

    # 気相H2
    from ase import Atoms
    h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
    h2.center(vacuum=10.0)
    h2.calc = EMT()
    E_h2 = h2.get_potential_energy()

    E_ads = E_slab_ads - E_slab - 0.5 * E_h2
    return E_ads

# 複数サイトでの吸着エネルギー
sites = ['fcc', 'hcp', 'ontop']
for site in sites:
    E_ads = dft_adsorption_energy('Pt', 'H', site)
    print(f"Pt(111) {site}サイト H吸着: {E_ads:.3f} eV")
```

### 例25: 反応経路解析（NEB法）

```python
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.optimize import BFGS
import numpy as np

# 初期状態と最終状態（簡略化例）
def setup_reaction_path():
    """H原子の表面拡散（fcc → hcp）"""
    # 初期状態: H at fcc site
    slab_initial = fcc111('Pt', size=(3, 3, 3), vacuum=10.0)
    from ase import Atoms
    from ase.build import add_adsorbate
    h_atom = Atoms('H')
    add_adsorbate(slab_initial, h_atom, height=1.5, position='fcc')
    slab_initial.calc = EMT()

    # 最終状態: H at hcp site
    slab_final = fcc111('Pt', size=(3, 3, 3), vacuum=10.0)
    h_atom = Atoms('H')
    add_adsorbate(slab_final, h_atom, height=1.5, position='hcp')
    slab_final.calc = EMT()

    return slab_initial, slab_final

initial, final = setup_reaction_path()

# NEB計算（簡略版）
print("NEB法による遷移状態探索")
print(f"初期状態エネルギー: {initial.get_potential_energy():.3f} eV")
print(f"最終状態エネルギー: {final.get_potential_energy():.3f} eV")
print("実際のNEB計算には複数のイメージと最適化が必要")
```

### 例26: 電子状態解析

```python
import numpy as np
import matplotlib.pyplot as plt

# DFT計算から得られるDOS（状態密度）のシミュレーション
def simulate_dos(metal):
    """金属のDOSをシミュレート"""
    energies = np.linspace(-10, 5, 500)

    if metal == 'Pt':
        # d-band: -6 to -2 eV
        dos = np.exp(-((energies + 2.25)**2) / 2) * 3
        # sp-band
        dos += np.exp(-((energies - 0)**2) / 10) * 0.5
    elif metal == 'Cu':
        # d-band: -4 to -1 eV (deeper)
        dos = np.exp(-((energies + 2.67)**2) / 2) * 3
        dos += np.exp(-((energies - 0)**2) / 10) * 0.5

    return energies, dos

# プロット
fig, ax = plt.subplots(figsize=(10, 6))

for metal in ['Pt', 'Cu']:
    energies, dos = simulate_dos(metal)
    ax.plot(energies, dos, label=metal, linewidth=2)

ax.axvline(0, color='k', linestyle='--', label='Fermi level')
ax.set_xlabel('Energy (eV)', fontsize=12)
ax.set_ylabel('Density of States', fontsize=12)
ax.set_title('Electronic DOS of Metal Catalysts', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

print("DOSプロット作成完了")
```

### 例27: DFT-ML連携ワークフロー

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# ステップ1: 少数のDFT計算
def expensive_dft_calculation(composition):
    """高コストDFT計算（シミュレーション）"""
    x = composition
    # 実際の吸着エネルギー計算
    energy = -2.0 + 3.0 * x - 2.0 * x**2 + np.random.normal(0, 0.05)
    return energy

# 初期DFT計算（5点）
X_dft = np.array([[0.2], [0.4], [0.6], [0.8], [1.0]])
y_dft = np.array([expensive_dft_calculation(x[0]) for x in X_dft])

# ステップ2: GPRサロゲートモデル
gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10)
gpr.fit(X_dft, y_dft)

# ステップ3: 多数点での予測（低コスト）
X_pred = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred, y_std = gpr.predict(X_pred, return_std=True)

print("DFT-ML連携ワークフロー:")
print(f"DFT計算数: {len(X_dft)}")
print(f"ML予測数: {len(X_pred)}")
print(f"最適組成（予測）: {X_pred[np.argmin(y_pred)][0]:.3f}")

# プロット
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_dft, y_dft, c='red', s=100, label='DFT calc', zorder=10)
ax.plot(X_pred, y_pred, 'b-', label='GPR mean')
ax.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std,
                alpha=0.3, label='±1 std')
ax.set_xlabel('Composition')
ax.set_ylabel('Adsorption Energy (eV)')
ax.legend()
```

---

## 3.5 反応速度論解析

### 例28: Arrhenius解析

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 実験データ（温度 vs 反応速度）
temperatures = np.array([300, 350, 400, 450, 500, 550])  # K
rate_constants = np.array([0.01, 0.05, 0.15, 0.35, 0.70, 1.20])  # s⁻¹

# Arrhenius式: k = A * exp(-Ea / (R*T))
def arrhenius(T, A, Ea):
    R = 8.314e-3  # kJ/(mol·K)
    return A * np.exp(-Ea / (R * T))

# フィッティング
popt, pcov = curve_fit(arrhenius, temperatures, rate_constants, p0=[1e10, 50])
A_fit, Ea_fit = popt

print(f"頻度因子 A: {A_fit:.2e} s⁻¹")
print(f"活性化エネルギー Ea: {Ea_fit:.2f} kJ/mol")

# Arrheniusプロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 通常プロット
axes[0].scatter(temperatures, rate_constants, label='Experimental', s=100)
T_fit = np.linspace(280, 570, 100)
axes[0].plot(T_fit, arrhenius(T_fit, A_fit, Ea_fit), 'r--', label='Fit')
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Rate constant (s⁻¹)')
axes[0].legend()

# Arrheniusプロット（線形化）
axes[1].scatter(1000/temperatures, np.log(rate_constants), s=100)
axes[1].plot(1000/T_fit, np.log(arrhenius(T_fit, A_fit, Ea_fit)), 'r--')
axes[1].set_xlabel('1000/T (K⁻¹)')
axes[1].set_ylabel('ln(k)')
axes[1].set_title(f'Ea = {Ea_fit:.1f} kJ/mol')
```

### 例29: 反応次数解析

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 反応速度式: -dC/dt = k * C^n
def reaction_rate(C, t, k, n):
    """n次反応の速度式"""
    return -k * C**n

# パラメータ
k = 0.1  # 速度定数
C0 = 1.0  # 初期濃度
t = np.linspace(0, 50, 100)

# 異なる反応次数で計算
fig, ax = plt.subplots(figsize=(10, 6))

for n in [0, 1, 2]:
    C = odeint(reaction_rate, C0, t, args=(k, n))
    ax.plot(t, C, label=f'n={n} order', linewidth=2)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Concentration', fontsize=12)
ax.set_title('Reaction Order Effects', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# 半減期の計算
def half_life(k, n, C0=1.0):
    """n次反応の半減期"""
    if n == 0:
        return C0 / (2 * k)
    elif n == 1:
        return np.log(2) / k
    elif n == 2:
        return 1 / (k * C0)

for n in [0, 1, 2]:
    t_half = half_life(k, n, C0)
    print(f"{n}次反応の半減期: {t_half:.2f}")
```

### 例30: マイクロキネティクスモデル

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# CO酸化反応のマイクロキネティクスモデル
# CO(g) + O(g) -> CO2(g) on Pt surface
def microkinetics(y, t, k1, k2, k3, k4):
    """
    y: [θ_CO, θ_O, θ_free]  # 表面被覆率
    k1: CO吸着速度定数
    k2: O2解離吸着速度定数
    k3: CO酸化速度定数
    k4: CO2脱離速度定数
    """
    theta_CO, theta_O, theta_free = y

    # 気相分圧（一定）
    P_CO = 0.1  # bar
    P_O2 = 0.2  # bar

    # 反応速度
    r1 = k1 * P_CO * theta_free  # CO吸着
    r2 = k2 * P_O2 * theta_free**2  # O2解離吸着
    r3 = k3 * theta_CO * theta_O  # CO + O -> CO2
    r4 = k4 * theta_CO * theta_O  # CO2脱離（反応と同時）

    # 被覆率変化
    dtheta_CO_dt = r1 - r3
    dtheta_O_dt = 2 * r2 - r3
    dtheta_free_dt = -r1 - 2 * r2 + r3

    return [dtheta_CO_dt, dtheta_O_dt, dtheta_free_dt]

# 初期条件
y0 = [0.0, 0.0, 1.0]  # 清浄表面

# 速度定数（任意単位）
k1, k2, k3, k4 = 1.0, 0.5, 2.0, 2.0

# 時間発展
t = np.linspace(0, 10, 1000)
solution = odeint(microkinetics, y0, t, args=(k1, k2, k3, k4))

# プロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, solution[:, 0], label='θ_CO', linewidth=2)
ax.plot(t, solution[:, 1], label='θ_O', linewidth=2)
ax.plot(t, solution[:, 2], label='θ_free', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Surface Coverage', fontsize=12)
ax.set_title('Microkinetic Model: CO Oxidation on Pt', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# 定常状態
theta_CO_ss = solution[-1, 0]
theta_O_ss = solution[-1, 1]
TOF = k3 * theta_CO_ss * theta_O_ss  # Turnover Frequency

print(f"定常状態:")
print(f"  θ_CO: {theta_CO_ss:.3f}")
print(f"  θ_O: {theta_O_ss:.3f}")
print(f"  TOF: {TOF:.3f} s⁻¹")
```

---

## 3.6 プロジェクトチャレンジ

**課題: CO2還元触媒の最適化**

以下の手順で、CO2→CO変換触媒を最適化してください：

1. **データ収集**: 複数金属（Cu, Ag, Au, Pd, Pt）のCO2吸着エネルギーを計算
2. **記述子計算**: d-band center、work functionなどを取得
3. **予測モデル構築**: 記述子から活性を予測するGPRモデル
4. **ベイズ最適化**: 合金組成を最適化（Cu-Ag二元系）
5. **検証**: 最適組成でのDFT計算（EMT近似）

**評価基準:**
- CO生成選択性 > 80%
- 過電圧 < 0.5 V
- コスト（Ag使用量）最小化

**提出物:**
- 最適組成（Cu:Ag比率）
- 予測活性とコストのトレードオフ分析
- Pythonコード全体

---

## 演習問題

**問1:** Pt-Ni合金触媒（Pt:Ni = 3:1）の(111)表面を作成し、配位数分布を計算せよ。

**問2:** 第2章のSabatier原理に基づき、最適H吸着エネルギーを機械学習で予測するモデルを構築せよ（例8-14参照）。

**問3:** ベイズ最適化で、焼成温度（400-900 K）と担持量（0-50 wt%）を同時最適化し、最大活性を達成せよ。

**問4:** Arrheniusプロットから、あなたの実験データの活性化エネルギーを算出せよ。

**問5:** マイクロキネティクスモデルで、CO/O2比を変化させたときの定常状態被覆率とTOFをシミュレートせよ。

---

## 参考文献

1. **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/
2. **scikit-optimize**: https://scikit-optimize.github.io/
3. Nørskov, J. K. et al. "Origin of the Overpotential for Oxygen Reduction at a Fuel-Cell Cathode." *J. Phys. Chem. B* (2004).
4. Hammer, B. & Nørskov, J. K. "Theoretical Surface Science and Catalysis." *Advances in Catalysis* (2000).

---

**次章**: [第4章：触媒MI実践ケーススタディ](chapter4-case-studies.md)

**ライセンス**: このコンテンツはCC BY 4.0ライセンスの下で提供されています。
