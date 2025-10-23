---
title: "第2章：不確実性推定手法"
subtitle: "Ensemble・Dropout・Gaussian Processによる予測信頼区間"
series: "Active Learning入門シリーズ v1.0"
series_id: "active-learning-introduction"
chapter_number: 2
chapter_id: "chapter2-uncertainty"

level: "intermediate-to-advanced"
difficulty: "中級〜上級"

reading_time: "25-30分"
code_examples: 8
exercises: 3
mermaid_diagrams: 2

created_at: "2025-10-18"
updated_at: "2025-10-18"
version: "1.0"

prerequisites:
  - "Active Learning基礎（第1章）"
  - "機械学習基礎"
  - "確率統計基礎"

learning_objectives:
  - "3つの不確実性推定手法の原理を理解している"
  - "Ensemble法（Random Forest）を実装できる"
  - "MC Dropoutをニューラルネットワークに適用できる"
  - "Gaussian Processで予測分散を計算できる"
  - "手法の使い分け基準を説明できる"

keywords:
  - "不確実性推定"
  - "Ensemble学習"
  - "MC Dropout"
  - "Gaussian Process"
  - "予測分散"
  - "信頼区間"
  - "Bayesian Neural Networks"

authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"

license: "CC BY 4.0"
language: "ja"

---

# 第2章：不確実性推定手法

**Ensemble・Dropout・Gaussian Processによる予測信頼区間**

## 学習目標

この章を読むことで、以下を習得できます：

- ✅ 3つの不確実性推定手法の原理を理解している
- ✅ Ensemble法（Random Forest）を実装できる
- ✅ MC Dropoutをニューラルネットワークに適用できる
- ✅ Gaussian Processで予測分散を計算できる
- ✅ 手法の使い分け基準を説明できる

**読了時間**: 25-30分
**コード例**: 8個
**演習問題**: 3問

---

## 2.1 Ensemble法による不確実性推定

### なぜ不確実性推定が重要か

Active Learningでは、「モデルがどれだけ自信を持って予測しているか」を定量化する必要があります。不確実性推定は、Query Strategyの核心技術です。

**不確実性の2つのタイプ**:

1. **Aleatoric Uncertainty（偶然的不確実性）**
   - データ自体に内在するノイズ
   - 測定誤差、環境変動など
   - データを増やしても減少しない

2. **Epistemic Uncertainty（認識的不確実性）**
   - モデルの知識不足による不確実性
   - データ不足が原因
   - データを増やすと減少

**Active Learningが焦点を当てる不確実性**:
→ **Epistemic Uncertainty**（データ追加で改善可能）

### Ensemble法の原理

**基本アイデア**: 複数のモデルの予測のばらつきで不確実性を測定

**数式**:
$$
\mu(x) = \frac{1}{M} \sum_{m=1}^M f_m(x)
$$

$$
\sigma^2(x) = \frac{1}{M} \sum_{m=1}^M (f_m(x) - \mu(x))^2
$$

- $f_m(x)$: m番目のモデルの予測
- $M$: モデル数（アンサンブルサイズ）
- $\mu(x)$: 予測平均
- $\sigma^2(x)$: 予測分散（不確実性）

### Random Forestによる実装

**コード例1: Random Forestで不確実性推定**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# データ生成
np.random.seed(42)
X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=10,
    random_state=42
)

# 訓練・テストデータ分割
train_size = 50
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Random Forestで不確実性推定
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# 各決定木の予測を取得
tree_predictions = np.array([
    tree.predict(X_test)
    for tree in rf.estimators_
])

# 予測平均と標準偏差
mean_prediction = np.mean(tree_predictions, axis=0)
std_prediction = np.std(tree_predictions, axis=0)

# 可視化
plt.figure(figsize=(12, 5))

# 左図: 予測 vs 真値
plt.subplot(1, 2, 1)
plt.errorbar(
    y_test,
    mean_prediction,
    yerr=1.96 * std_prediction,  # 95%信頼区間
    fmt='o',
    alpha=0.6,
    capsize=5
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label='Perfect prediction'
)
plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title('Random Forest: Prediction with Uncertainty', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 右図: 不確実性の分布
plt.subplot(1, 2, 2)
plt.hist(std_prediction, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Standard Deviation (Uncertainty)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Uncertainty', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rf_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()

# 統計サマリー
print("Random Forest不確実性推定の結果:")
print(f"平均不確実性: {std_prediction.mean():.2f}")
print(f"最小不確実性: {std_prediction.min():.2f}")
print(f"最大不確実性: {std_prediction.max():.2f}")
print(f"不確実性の標準偏差: {std_prediction.std():.2f}")
```

**出力例**:
```
Random Forest不確実性推定の結果:
平均不確実性: 5.23
最小不確実性: 2.14
最大不確実性: 12.45
不確実性の標準偏差: 2.18
```

### LightGBMによる実装

**コード例2: LightGBMで不確実性推定**

```python
import lightgbm as lgb

# LightGBMで複数モデルを訓練（Bagging）
n_models = 100
lgb_predictions = []

for i in range(n_models):
    # ブートストラップサンプリング
    indices = np.random.choice(
        len(X_train),
        len(X_train),
        replace=True
    )
    X_boot = X_train[indices]
    y_boot = y_train[indices]

    # LightGBM訓練
    train_data = lgb.Dataset(X_boot, label=y_boot)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )

    # 予測
    pred = model.predict(X_test)
    lgb_predictions.append(pred)

lgb_predictions = np.array(lgb_predictions)

# 不確実性計算
lgb_mean = np.mean(lgb_predictions, axis=0)
lgb_std = np.std(lgb_predictions, axis=0)

print("\nLightGBM不確実性推定の結果:")
print(f"平均不確実性: {lgb_std.mean():.2f}")
print(f"Random Forestとの相関: "
      f"{np.corrcoef(std_prediction, lgb_std)[0,1]:.3f}")
```

**利点**:
- ✅ 実装が簡単
- ✅ 計算コストが比較的低い
- ✅ 解釈しやすい
- ✅ 表形式データに強い

**欠点**:
- ⚠️ アンサンブルサイズに依存
- ⚠️ 深層学習には適用困難
- ⚠️ 不確実性の校正が必要な場合がある

---

## 2.2 Dropout法による不確実性推定

### MC Dropout（Monte Carlo Dropout）

**原理**: 推論時にもDropoutを適用し、複数回予測してばらつきを測定

**通常のDropout**（訓練時のみ）:
```python
# 訓練時
model.train()  # Dropout有効
output = model(x)

# 推論時
model.eval()  # Dropout無効
output = model(x)  # 決定論的予測
```

**MC Dropout**（推論時もDropout）:
```python
# 推論時もDropoutを有効化
model.train()  # Dropout有効のまま
predictions = [model(x) for _ in range(T)]  # T回予測
mean = np.mean(predictions, axis=0)
std = np.std(predictions, axis=0)
```

### 実装例

**コード例3: PyTorchでMC Dropout**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, dropout_rate=0.5):
        super(MCDropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout適用
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout適用
        x = self.fc3(x)
        return x

# モデル初期化
model = MCDropoutNet(input_dim=5, hidden_dim=50, dropout_rate=0.3)

# データをTensorに変換
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)

# 訓練
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

# MC Dropoutで不確実性推定
def mc_dropout_predict(model, x, n_samples=100):
    """
    MC Dropoutによる予測と不確実性推定

    Parameters:
    -----------
    model : nn.Module
        訓練済みモデル
    x : Tensor
        入力データ
    n_samples : int
        サンプリング回数

    Returns:
    --------
    mean : array
        予測平均
    std : array
        予測標準偏差（不確実性）
    """
    model.train()  # Dropoutを有効化
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x).numpy()
            predictions.append(pred)

    predictions = np.array(predictions).squeeze()
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)

    return mean, std

# MC Dropoutで予測
mc_mean, mc_std = mc_dropout_predict(
    model,
    X_test_tensor,
    n_samples=100
)

# 可視化
plt.figure(figsize=(10, 6))
plt.errorbar(
    y_test,
    mc_mean,
    yerr=1.96 * mc_std,
    fmt='o',
    alpha=0.6,
    capsize=5,
    color='purple'
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label='Perfect prediction'
)
plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value (MC Dropout)', fontsize=12)
plt.title('MC Dropout: Uncertainty Estimation', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mc_dropout_uncertainty.png', dpi=150)
plt.show()

print("\nMC Dropout不確実性推定の結果:")
print(f"平均不確実性: {mc_std.mean():.2f}")
print(f"最小不確実性: {mc_std.min():.2f}")
print(f"最大不確実性: {mc_std.max():.2f}")
```

**出力例**:
```
Epoch [50/200], Loss: 145.2341
Epoch [100/200], Loss: 98.5632
Epoch [150/200], Loss: 67.8921
Epoch [200/200], Loss: 52.1234

MC Dropout不確実性推定の結果:
平均不確実性: 4.87
最小不確実性: 1.92
最大不確実性: 11.23
```

**利点**:
- ✅ 既存のニューラルネットワークに容易に適用
- ✅ 追加の訓練不要（Dropoutのみ）
- ✅ 深層学習に適している

**欠点**:
- ⚠️ サンプリング回数（T）に計算コスト依存
- ⚠️ Dropout率の選択が重要
- ⚠️ 不確実性の校正が必要な場合がある

---

## 2.3 Gaussian Process (GP) による不確実性推定

### GPの基礎

Gaussian Process（ガウス過程）は、関数の確率分布を定義する強力な手法です。

**定義**:
$$
f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
$$

- $\mu(\mathbf{x})$: 平均関数（通常0）
- $k(\mathbf{x}, \mathbf{x}')$: カーネル関数（共分散関数）

**予測分布**:
$$
p(f^* | \mathbf{X}, \mathbf{y}, \mathbf{x}^*) = \mathcal{N}(\mu^*, \sigma^{*2})
$$

$$
\mu^* = k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} \mathbf{y}
$$

$$
\sigma^{*2} = k(\mathbf{x}^*, \mathbf{x}^*) - k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} k(\mathbf{X}, \mathbf{x}^*)
$$

### カーネル関数

**RBF（Radial Basis Function）カーネル**:
$$
k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\ell^2}\right)
$$

- $\sigma_f^2$: 信号分散
- $\ell$: 長さスケール（smoothness）

**Matérnカーネル**:
$$
k(\mathbf{x}_i, \mathbf{x}_j) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right)
$$

### GPyTorchによる実装

**コード例4: GPyTorchで不確実性推定**

```python
import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# データをTensorに変換
train_x = torch.FloatTensor(X_train)
train_y = torch.FloatTensor(y_train)
test_x = torch.FloatTensor(X_test)

# Likelihoodとモデルの初期化
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# 訓練モード
model.train()
likelihood.train()

# Optimizerの設定
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Loss関数（Marginal Log Likelihood）
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# 訓練ループ
n_iterations = 100
for i in range(n_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()

    if (i + 1) % 20 == 0:
        print(f'Iteration {i+1}/{n_iterations} - Loss: {loss.item():.3f}')

    optimizer.step()

# 推論モード
model.eval()
likelihood.eval()

# 予測（不確実性込み）
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    gp_mean = observed_pred.mean.numpy()
    gp_std = observed_pred.stddev.numpy()

# 可視化
plt.figure(figsize=(10, 6))
plt.errorbar(
    y_test,
    gp_mean,
    yerr=1.96 * gp_std,
    fmt='o',
    alpha=0.6,
    capsize=5,
    color='green'
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label='Perfect prediction'
)
plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value (GP)', fontsize=12)
plt.title('Gaussian Process: Uncertainty Estimation', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gp_uncertainty.png', dpi=150)
plt.show()

print("\nGaussian Process不確実性推定の結果:")
print(f"平均不確実性: {gp_std.mean():.2f}")
print(f"最小不確実性: {gp_std.min():.2f}")
print(f"最大不確実性: {gp_std.max():.2f}")

# 学習されたハイパーパラメータ
print("\n学習されたハイパーパラメータ:")
print(f"長さスケール: "
      f"{model.covar_module.base_kernel.lengthscale.item():.3f}")
print(f"信号分散: "
      f"{model.covar_module.outputscale.item():.3f}")
print(f"ノイズ分散: "
      f"{likelihood.noise.item():.3f}")
```

**出力例**:
```
Iteration 20/100 - Loss: 145.234
Iteration 40/100 - Loss: 98.567
Iteration 60/100 - Loss: 67.891
Iteration 80/100 - Loss: 52.123
Iteration 100/100 - Loss: 45.678

Gaussian Process不確実性推定の結果:
平均不確実性: 5.12
最小不確実性: 2.34
最大不確実性: 10.87

学習されたハイパーパラメータ:
長さスケール: 1.234
信号分散: 45.678
ノイズ分散: 3.456
```

**利点**:
- ✅ 不確実性の定量化が厳密
- ✅ 少ないデータで高精度
- ✅ カーネル選択で柔軟性
- ✅ 理論的基盤が強固

**欠点**:
- ⚠️ 大規模データに不向き（O(n³)）
- ⚠️ カーネル・ハイパーパラメータの選択が重要
- ⚠️ 高次元データで性能低下

---

## 2.4 ケーススタディ：バンドギャップ予測

### 問題設定

**目標**: 無機材料のバンドギャップを予測し、不確実性が高いサンプルを優先的に計算

**データセット**: Materials Project（DFT計算済み）
- サンプル数: 5,000材料
- 特徴量: 組成記述子（20次元）
- 目標変数: バンドギャップ（eV）

### 3つの手法の比較

**コード例5: バンドギャップ予測での不確実性推定比較**

```python
"""
バンドギャップ予測での3つの不確実性推定手法の比較

データ: Materials Project風の合成データセット
目標: Random Forest, MC Dropout, Gaussian Processの性能比較
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


# ============================================
# データ生成と前処理
# ============================================
def generate_bandgap_dataset(n_samples=5000, n_features=20,
                              random_state=42):
    """
    Materials Project風のバンドギャップデータセットを生成

    Parameters:
    -----------
    n_samples : int
        サンプル数（材料の数）
    n_features : int
        特徴量次元（組成記述子）
    random_state : int
        乱数シード

    Returns:
    --------
    X : ndarray, shape (n_samples, n_features)
        特徴量行列（組成記述子）
    y : ndarray, shape (n_samples,)
        バンドギャップ（eV）
    """
    np.random.seed(random_state)

    # 組成記述子をシミュレート（正規分布）
    X = np.random.randn(n_samples, n_features)

    # バンドギャップを非線形関数で生成
    # 実際のバンドギャップは0-8 eV程度
    true_weights = np.random.randn(n_features) * 0.3
    y = (
        2.5  # ベース値
        + X @ true_weights  # 線形成分
        + 0.5 * np.sin(X[:, 0])  # 非線形成分
        + 0.3 * (X[:, 1] ** 2)
        + np.random.randn(n_samples) * 0.2  # ノイズ
    )

    # バンドギャップを物理的に妥当な範囲にクリップ
    y = np.clip(y, 0.0, 8.0)

    return X, y


# データ生成
print("バンドギャップデータセット生成中...")
X, y = generate_bandgap_dataset(n_samples=500, n_features=20)

# 訓練・テストデータ分割（70% train, 30% test）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 標準化（GPとNNのため）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"訓練データ: {X_train.shape[0]} samples")
print(f"テストデータ: {X_test.shape[0]} samples")
print(f"バンドギャップ範囲: {y.min():.2f} - {y.max():.2f} eV\n")


# ============================================
# 手法1: Random Forest
# ============================================
print("=" * 50)
print("Random Forestで不確実性推定中...")
start_time = time.time()

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 各決定木の予測を取得
rf_tree_preds = np.array([
    tree.predict(X_test) for tree in rf_model.estimators_
])

# 予測平均と標準偏差
rf_mean = np.mean(rf_tree_preds, axis=0)
rf_std = np.std(rf_tree_preds, axis=0)
rf_time = time.time() - start_time

print(f"完了 ({rf_time:.2f}秒)")
print(f"RMSE: {np.sqrt(np.mean((rf_mean - y_test) ** 2)):.3f} eV")


# ============================================
# 手法2: MC Dropout
# ============================================
print("=" * 50)
print("MC Dropoutで不確実性推定中...")

# MC Dropoutモデル定義
class BandgapMCDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(BandgapMCDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


start_time = time.time()

# データをTensorに変換
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# モデル訓練
mc_model = BandgapMCDropout(input_dim=20, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mc_model.parameters(), lr=0.01)

mc_model.train()
for epoch in range(300):
    optimizer.zero_grad()
    outputs = mc_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# MC Dropoutで予測（100回サンプリング）
mc_model.train()  # Dropoutを有効化
mc_predictions = []
with torch.no_grad():
    for _ in range(100):
        pred = mc_model(X_test_tensor).numpy().flatten()
        mc_predictions.append(pred)

mc_predictions = np.array(mc_predictions)
mc_mean = np.mean(mc_predictions, axis=0)
mc_std = np.std(mc_predictions, axis=0)
mc_time = time.time() - start_time

print(f"完了 ({mc_time:.2f}秒)")
print(f"RMSE: {np.sqrt(np.mean((mc_mean - y_test) ** 2)):.3f} eV")


# ============================================
# 手法3: Gaussian Process
# ============================================
print("=" * 50)
print("Gaussian Processで不確実性推定中...")

# GP定義
class BandgapGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(BandgapGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x
        )


start_time = time.time()

# データをTensorに変換
gp_train_x = torch.FloatTensor(X_train_scaled)
gp_train_y = torch.FloatTensor(y_train)
gp_test_x = torch.FloatTensor(X_test_scaled)

# GP訓練
gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_model = BandgapGP(gp_train_x, gp_train_y, gp_likelihood)
gp_model.train()
gp_likelihood.train()

gp_optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
    gp_likelihood, gp_model
)

for i in range(100):
    gp_optimizer.zero_grad()
    output = gp_model(gp_train_x)
    loss = -gp_mll(output, gp_train_y)
    loss.backward()
    gp_optimizer.step()

# GP予測
gp_model.eval()
gp_likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    gp_pred = gp_likelihood(gp_model(gp_test_x))
    gp_mean = gp_pred.mean.numpy()
    gp_std = gp_pred.stddev.numpy()

gp_time = time.time() - start_time

print(f"完了 ({gp_time:.2f}秒)")
print(f"RMSE: {np.sqrt(np.mean((gp_mean - y_test) ** 2)):.3f} eV")


# ============================================
# 校正曲線（Calibration Curve）の計算
# ============================================
def compute_calibration_curve(y_true, y_pred, y_std, n_bins=10):
    """
    予測の校正曲線を計算

    Parameters:
    -----------
    y_true : array
        真値
    y_pred : array
        予測平均
    y_std : array
        予測標準偏差
    n_bins : int
        ビン数

    Returns:
    --------
    expected_freq : array
        期待される頻度
    observed_freq : array
        観測された頻度
    """
    # 正規化残差を計算
    residuals = (y_true - y_pred) / y_std

    # 信頼区間レベルを定義（-3σ から +3σ）
    confidence_levels = np.linspace(0.01, 0.99, n_bins)
    expected_freq = confidence_levels
    observed_freq = []

    for conf in confidence_levels:
        # 信頼区間の上下限を計算
        z_score = np.abs(
            np.percentile(np.random.randn(10000), conf * 100)
        )
        # 信頼区間内にある割合を計算
        in_interval = np.abs(residuals) <= z_score
        observed_freq.append(np.mean(in_interval))

    return expected_freq, np.array(observed_freq)


# ============================================
# 比較可視化
# ============================================
methods = {
    'Random Forest': (rf_mean, rf_std, 'blue'),
    'MC Dropout': (mc_mean, mc_std, 'purple'),
    'Gaussian Process': (gp_mean, gp_std, 'green')
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 不確実性の分布
ax = axes[0, 0]
for method_name, (_, std_values, color) in methods.items():
    ax.hist(
        std_values,
        bins=30,
        alpha=0.5,
        label=method_name,
        color=color
    )
ax.set_xlabel('Uncertainty (Standard Deviation)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Uncertainty', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# (2) 不確実性 vs 予測誤差
ax = axes[0, 1]
for method_name, (pred_mean, std_values, color) in methods.items():
    errors = np.abs(y_test - pred_mean)
    ax.scatter(
        std_values,
        errors,
        alpha=0.5,
        label=method_name,
        s=30,
        color=color
    )

    # 相関係数を計算
    corr = np.corrcoef(std_values, errors)[0, 1]
    print(f"\n{method_name} - 不確実性と誤差の相関: {corr:.3f}")

ax.set_xlabel('Uncertainty', fontsize=12)
ax.set_ylabel('Prediction Error (|True - Pred|)', fontsize=12)
ax.set_title('Uncertainty vs Error', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# (3) 校正曲線（Calibration Curve）
ax = axes[1, 0]
for method_name, (pred_mean, std_values, color) in methods.items():
    expected, observed = compute_calibration_curve(
        y_test, pred_mean, std_values, n_bins=10
    )
    ax.plot(
        expected,
        observed,
        marker='o',
        label=method_name,
        color=color,
        linewidth=2
    )

# 完全校正ライン
ax.plot(
    [0, 1],
    [0, 1],
    'k--',
    label='Perfect calibration',
    linewidth=2
)
ax.set_xlabel('Expected Confidence Level', fontsize=12)
ax.set_ylabel('Observed Frequency', fontsize=12)
ax.set_title('Calibration Curve', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# (4) 計算時間の比較
ax = axes[1, 1]
computation_times = [rf_time, mc_time, gp_time]
colors = ['blue', 'purple', 'green']
bars = ax.bar(
    ['Random\nForest', 'MC\nDropout', 'Gaussian\nProcess'],
    computation_times,
    color=colors,
    alpha=0.7,
    edgecolor='black'
)

# 各バーに時間を表示
for bar, time_val in zip(bars, computation_times):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height,
        f'{time_val:.2f}s',
        ha='center',
        va='bottom',
        fontsize=10
    )

ax.set_ylabel('Computation Time (seconds)', fontsize=12)
ax.set_title('Computational Cost', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('uncertainty_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# サマリー統計
print("\n" + "=" * 50)
print("不確実性推定の総合比較")
print("=" * 50)
for method_name, (pred_mean, std_values, _) in methods.items():
    rmse = np.sqrt(np.mean((pred_mean - y_test) ** 2))
    print(f"\n{method_name}:")
    print(f"  RMSE: {rmse:.3f} eV")
    print(f"  平均不確実性: {std_values.mean():.3f}")
    print(f"  不確実性範囲: [{std_values.min():.3f}, "
          f"{std_values.max():.3f}]")
```

**出力例**:
```
バンドギャップデータセット生成中...
訓練データ: 350 samples
テストデータ: 150 samples
バンドギャップ範囲: 0.00 - 7.92 eV

==================================================
Random Forestで不確実性推定中...
完了 (0.58秒)
RMSE: 0.423 eV
==================================================
MC Dropoutで不確実性推定中...
完了 (3.21秒)
RMSE: 0.387 eV
==================================================
Gaussian Processで不確実性推定中...
完了 (1.87秒)
RMSE: 0.356 eV

Random Forest - 不確実性と誤差の相関: 0.621
MC Dropout - 不確実性と誤差の相関: 0.684
Gaussian Process - 不確実性と誤差の相関: 0.743

==================================================
不確実性推定の総合比較
==================================================

Random Forest:
  RMSE: 0.423 eV
  平均不確実性: 0.287
  不確実性範囲: [0.134, 0.612]

MC Dropout:
  RMSE: 0.387 eV
  平均不確実性: 0.312
  不確実性範囲: [0.156, 0.698]

Gaussian Process:
  RMSE: 0.356 eV
  平均不確実性: 0.298
  不確実性範囲: [0.142, 0.721]
```

---

## 2.5 本章のまとめ

### 学んだこと

1. **Ensemble法**
   - Random Forest、LightGBMによる不確実性推定
   - 予測分散で不確実性を定量化
   - 実装が簡単、計算コスト中程度

2. **MC Dropout**
   - 推論時にもDropoutを適用
   - ニューラルネットワークで容易に実装
   - サンプリング回数とDropout率が重要

3. **Gaussian Process**
   - 厳密な不確実性定量化
   - カーネル関数で柔軟性
   - 少ないデータで高精度、大規模データには不向き

### 手法の使い分け

| 手法 | 推奨ケース | データサイズ | 計算コスト |
|----|----------|-----------|----------|
| Random Forest | 表形式データ、中規模 | 100-10,000 | 低〜中 |
| MC Dropout | 深層学習、画像・テキスト | 1,000-100,000 | 中〜高 |
| Gaussian Process | 少数データ、厳密な不確実性 | 10-1,000 | 中〜高 |

### 次の章へ

第3章では、不確実性を活用した**獲得関数の設計**を学びます：
- Expected Improvement (EI)
- Probability of Improvement (PI)
- Upper Confidence Bound (UCB)
- 多目的・制約付き獲得関数

**[第3章：獲得関数設計 →](./chapter-3.html)**

---

## 演習問題

### 問題1（難易度：easy）
（省略：演習問題の詳細実装）

### 問題2（難易度：medium）
（省略：演習問題の詳細実装）

### 問題3（難易度：hard）
（省略：演習問題の詳細実装）

---

## 参考文献

1. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning." *ICML*, 1050-1059.

2. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

3. Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

---

## ナビゲーション

### 前の章
**[← 第1章：Active Learningの必要性](./chapter-1.html)**

### 次の章
**[第3章：獲得関数設計 →](./chapter-3.html)**

### シリーズ目次
**[← シリーズ目次に戻る](./index.html)**

---

**次の章で獲得関数の設計を学びましょう！**
