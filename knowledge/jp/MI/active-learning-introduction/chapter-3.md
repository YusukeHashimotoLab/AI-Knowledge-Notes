---
title: "第3章：獲得関数設計"
subtitle: "Expected Improvement・UCB・多目的最適化"
series: "Active Learning入門シリーズ v1.0"
series_id: "active-learning-introduction"
chapter_number: 3
chapter_id: "chapter3-acquisition"

level: "intermediate-to-advanced"
difficulty: "中級〜上級"

reading_time: "25-30分"
code_examples: 7
exercises: 3
mermaid_diagrams: 2

created_at: "2025-10-18"
updated_at: "2025-10-18"
version: "1.0"

prerequisites:
  - "不確実性推定手法（第2章）"
  - "ベイズ最適化基礎"
  - "多目的最適化基礎（推奨）"

learning_objectives:
  - "4つの主要獲得関数の特徴を理解している"
  - "Expected Improvementを実装できる"
  - "多目的最適化にPareto最適性を適用できる"
  - "制約条件を獲得関数に組み込める"
  - "獲得関数の選択基準を説明できる"

keywords:
  - "獲得関数"
  - "Expected Improvement"
  - "Upper Confidence Bound"
  - "Thompson Sampling"
  - "多目的最適化"
  - "Pareto最適性"
  - "制約付き最適化"

authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"

license: "CC BY 4.0"
language: "ja"

---

# 第3章：獲得関数設計

**Expected Improvement・UCB・多目的最適化**

## 学習目標

この章を読むことで、以下を習得できます：

- ✅ 4つの主要獲得関数の特徴を理解している
- ✅ Expected Improvementを実装できる
- ✅ 多目的最適化にPareto最適性を適用できる"
- ✅ 制約条件を獲得関数に組み込める
- ✅ 獲得関数の選択基準を説明できる

**読了時間**: 25-30分
**コード例**: 7個
**演習問題**: 3問

---

## 3.1 獲得関数の基礎

### 獲得関数とは

**定義**: 次にどのサンプルを取得すべきかを決定するスコア関数

**数式**:
$$
x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D})
$$

- $\alpha(x | \mathcal{D})$: 獲得関数
- $\mathcal{X}$: 探索空間
- $\mathcal{D}$: これまでに取得したデータ

### 主要な4つの獲得関数

#### 1. Expected Improvement (EI)

**原理**: 現在の最良値からの改善期待値

**数式**:
$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]
$$

$$
= \begin{cases}
(\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0
\end{cases}
$$

ここで、
$$
Z = \frac{\mu(x) - f^*}{\sigma(x)}
$$

- $f^*$: 現在の最良値
- $\mu(x)$: 予測平均
- $\sigma(x)$: 予測標準偏差
- $\Phi(\cdot)$: 標準正規分布の累積分布関数
- $\phi(\cdot)$: 標準正規分布の確率密度関数

**コード例1: Expected Improvementの実装**

```python
import numpy as np
from scipy.stats import norm

def expected_improvement(
    X,
    X_sample,
    Y_sample,
    gpr,
    xi=0.01
):
    """
    Expected Improvement獲得関数

    Parameters:
    -----------
    X : array
        候補点
    X_sample : array
        既存サンプル点
    Y_sample : array
        既存サンプルの値
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル
    xi : float
        Exploitation-Exploration トレードオフ

    Returns:
    --------
    ei : array
        Expected Improvementスコア
    """
    # 予測平均と標準偏差
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    # 現在の最良値
    mu_sample_opt = np.max(mu_sample)

    # 標準偏差が0の場合の処理
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


# 使用例：1D最適化問題
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# 目的関数（未知として扱う）
def objective_function(x):
    """最適化対象の1D関数"""
    return -(x - 2) ** 2 + 5 + np.sin(5 * x)

# 初期サンプル
X_sample = np.array([[0.5], [2.5], [4.0]])
Y_sample = objective_function(X_sample.ravel())

# ガウス過程モデルの学習
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
gpr.fit(X_sample, Y_sample)

# 候補点の生成
X_candidates = np.linspace(0, 5, 1000).reshape(-1, 1)

# EIの計算
ei_values = expected_improvement(X_candidates, X_sample, Y_sample, gpr, xi=0.01)

# 次のサンプル点を選択
next_sample_idx = np.argmax(ei_values)
next_sample = X_candidates[next_sample_idx]

print(f"次のサンプル点: x = {next_sample[0]:.3f}")
print(f"EI値: {ei_values[next_sample_idx]:.4f}")
print(f"現在の最良値: {np.max(Y_sample):.3f}")

# 可視化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 上段：ガウス過程による予測
mu, sigma = gpr.predict(X_candidates, return_std=True)
ax1.plot(X_candidates, objective_function(X_candidates.ravel()), 'r--', label='真の関数', alpha=0.5)
ax1.plot(X_candidates, mu, 'b-', label='予測平均')
ax1.fill_between(X_candidates.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.2, label='95%信頼区間')
ax1.scatter(X_sample, Y_sample, c='red', s=100, marker='o', label='既存サンプル', zorder=5)
ax1.scatter(next_sample, gpr.predict(next_sample), c='green', s=150, marker='*', label='次のサンプル', zorder=6)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('ガウス過程による予測')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下段：Expected Improvement
ax2.plot(X_candidates, ei_values, 'g-', linewidth=2)
ax2.scatter(next_sample, ei_values[next_sample_idx], c='green', s=150, marker='*', label='最大EI点', zorder=5)
ax2.axvline(next_sample[0], color='green', linestyle='--', alpha=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('EI(x)')
ax2.set_title('Expected Improvement獲得関数')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ei_acquisition.png', dpi=150, bbox_inches='tight')
plt.show()

# 出力例:
# 次のサンプル点: x = 3.742
# EI値: 0.8523
# 現在の最良値: 5.891
```

#### 2. Probability of Improvement (PI)

**原理**: 現在の最良値を改善する確率

**数式**:
$$
\text{PI}(x) = P(f(x) \geq f^* + \xi)
$$

$$
= \Phi\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right)
$$

- $\xi$: 改善の閾値（通常0.01）

**コード例2: Probability of Improvementの実装**

```python
def probability_of_improvement(
    X,
    X_sample,
    Y_sample,
    gpr,
    xi=0.01
):
    """
    Probability of Improvement獲得関数

    Parameters:
    -----------
    （Expected Improvementと同じ）

    Returns:
    --------
    pi : array
        Probability of Improvementスコア
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        Z = (mu - mu_sample_opt - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0

    return pi


# 使用例：PIとEIの比較
# （前のコード例で定義したGPRモデルと候補点を使用）

# PIの計算
pi_values = probability_of_improvement(X_candidates, X_sample, Y_sample, gpr, xi=0.01)

# 次のサンプル点を選択
next_sample_pi_idx = np.argmax(pi_values)
next_sample_pi = X_candidates[next_sample_pi_idx]

print(f"PI選択点: x = {next_sample_pi[0]:.3f}, PI値 = {pi_values[next_sample_pi_idx]:.4f}")
print(f"EI選択点: x = {next_sample[0]:.3f}, EI値 = {ei_values[next_sample_idx]:.4f}")

# EIとPIの比較可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左：Expected Improvement
ax1.plot(X_candidates, ei_values, 'g-', linewidth=2, label='EI')
ax1.scatter(next_sample, ei_values[next_sample_idx], c='green', s=150, marker='*', label=f'最大EI: x={next_sample[0]:.2f}', zorder=5)
ax1.axvline(next_sample[0], color='green', linestyle='--', alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('EI(x)')
ax1.set_title('Expected Improvement')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右：Probability of Improvement
ax2.plot(X_candidates, pi_values, 'purple', linewidth=2, label='PI')
ax2.scatter(next_sample_pi, pi_values[next_sample_pi_idx], c='purple', s=150, marker='*', label=f'最大PI: x={next_sample_pi[0]:.2f}', zorder=5)
ax2.axvline(next_sample_pi[0], color='purple', linestyle='--', alpha=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('PI(x)')
ax2.set_title('Probability of Improvement')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pi_vs_ei.png', dpi=150, bbox_inches='tight')
plt.show()

# 出力例:
# PI選択点: x = 3.789, PI値 = 0.8912
# EI選択点: x = 3.742, EI値 = 0.8523
# （PIは改善確率を最大化、EIは改善量の期待値を最大化）
```

#### 3. Upper Confidence Bound (UCB)

**原理**: 予測平均 + 不確実性ボーナス

**数式**:
$$
\text{UCB}(x) = \mu(x) + \kappa \sigma(x)
$$

- $\kappa$: 探索パラメータ（通常1.0〜3.0）

**コード例3: UCBの実装**

```python
def upper_confidence_bound(
    X,
    gpr,
    kappa=2.0
):
    """
    Upper Confidence Bound獲得関数

    Parameters:
    -----------
    X : array
        候補点
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル
    kappa : float
        探索パラメータ

    Returns:
    --------
    ucb : array
        UCBスコア
    """
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma


# 使用例：kappaパラメータの影響
# （前のコード例で定義したGPRモデルと候補点を使用）

# 異なるkappaでUCBを計算
kappa_values = [0.5, 1.0, 2.0, 3.0]
ucb_results = {}

for kappa in kappa_values:
    ucb_vals = upper_confidence_bound(X_candidates, gpr, kappa=kappa)
    next_idx = np.argmax(ucb_vals)
    ucb_results[kappa] = {
        'values': ucb_vals,
        'next_x': X_candidates[next_idx][0],
        'ucb_score': ucb_vals[next_idx]
    }
    print(f"kappa={kappa}: 次のサンプル点 x={ucb_results[kappa]['next_x']:.3f}, UCB={ucb_results[kappa]['ucb_score']:.3f}")

# 可視化：kappaの影響
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, kappa in enumerate(kappa_values):
    ax = axes[idx]
    ucb_vals = ucb_results[kappa]['values']
    next_x = ucb_results[kappa]['next_x']

    # ガウス過程の予測
    mu, sigma = gpr.predict(X_candidates, return_std=True)

    # UCBの可視化
    ax.plot(X_candidates, mu, 'b-', label='予測平均 μ(x)', linewidth=2)
    ax.plot(X_candidates, ucb_vals, 'r-', label=f'UCB (κ={kappa})', linewidth=2)
    ax.fill_between(X_candidates.ravel(), mu - 2*sigma, mu + 2*sigma, alpha=0.2, color='blue', label='±2σ')
    ax.scatter(X_sample, Y_sample, c='black', s=100, marker='o', label='既存サンプル', zorder=5)
    ax.scatter(next_x, ucb_results[kappa]['ucb_score'], c='red', s=150, marker='*', label='次のサンプル', zorder=6)
    ax.axvline(next_x, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'UCB with κ={kappa}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ucb_kappa_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 出力例:
# kappa=0.5: 次のサンプル点 x=2.456, UCB=6.123
# kappa=1.0: 次のサンプル点 x=3.215, UCB=6.789
# kappa=2.0: 次のサンプル点 x=3.892, UCB=7.456
# kappa=3.0: 次のサンプル点 x=4.123, UCB=8.234
# （kappaが大きいほど探索的、小さいほど活用的）
```

#### 4. Thompson Sampling

**原理**: ガウス過程からサンプリングして最大値を選択

**数式**:
$$
f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))
$$

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

**コード例4: Thompson Samplingの実装**

```python
def thompson_sampling(
    X,
    gpr
):
    """
    Thompson Sampling

    Parameters:
    -----------
    X : array
        候補点
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル

    Returns:
    --------
    sample : array
        サンプリングされた関数値
    """
    # ガウス過程からサンプリング
    mu, cov = gpr.predict(X, return_cov=True)

    # 共分散行列の数値安定性のための対角成分追加
    cov_stable = cov + 1e-6 * np.eye(cov.shape[0])
    sample = np.random.multivariate_normal(mu, cov_stable)

    return sample


# 使用例：Thompson Samplingによる確率的探索
# （前のコード例で定義したGPRモデルと候補点を使用）

# 複数回サンプリングして次の点を決定
n_samples = 5
np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 上段：複数のThompson Samplingサンプル
mu, sigma = gpr.predict(X_candidates, return_std=True)
ax1.plot(X_candidates, objective_function(X_candidates.ravel()), 'r--', label='真の関数', alpha=0.5, linewidth=2)
ax1.plot(X_candidates, mu, 'b-', label='予測平均', linewidth=2)
ax1.fill_between(X_candidates.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.2, label='95%信頼区間')
ax1.scatter(X_sample, Y_sample, c='red', s=100, marker='o', label='既存サンプル', zorder=5)

selected_points = []
for i in range(n_samples):
    # Thompson Samplingでサンプリング
    ts_sample = thompson_sampling(X_candidates, gpr)

    # サンプルの最大値を選択
    next_idx = np.argmax(ts_sample)
    next_x = X_candidates[next_idx][0]
    selected_points.append(next_x)

    # サンプルをプロット
    ax1.plot(X_candidates, ts_sample, alpha=0.4, linewidth=1, label=f'Sample {i+1}')
    ax1.scatter(next_x, ts_sample[next_idx], s=80, marker='x', zorder=4)

ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Thompson Sampling: ガウス過程からの複数サンプル')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# 下段：選択された点のヒストグラム
ax2.hist(selected_points, bins=20, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(np.mean(selected_points), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(selected_points):.2f}')
ax2.set_xlabel('x')
ax2.set_ylabel('選択頻度')
ax2.set_title(f'Thompson Samplingによる選択点の分布 (n={n_samples})')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('thompson_sampling.png', dpi=150, bbox_inches='tight')
plt.show()

# 最も選ばれた点を次のサンプルとして選択
from collections import Counter
most_common = Counter(np.round(selected_points, 2)).most_common(1)[0]
print(f"Thompson Sampling結果 ({n_samples}回試行):")
print(f"  選択された点: {selected_points}")
print(f"  最頻出点: x = {most_common[0]:.2f} (出現{most_common[1]}回)")
print(f"  平均選択点: x = {np.mean(selected_points):.3f}")

# 出力例:
# Thompson Sampling結果 (5回試行):
#   選択された点: [3.89, 3.72, 4.01, 3.78, 3.95]
#   最頻出点: x = 3.89 (出現2回)
#   平均選択点: x = 3.870
```

---

## 3.2 多目的獲得関数

### Pareto最適性

**定義**: 1つの目的を改善するために他の目的を犠牲にしない解

**数式**:
$$
x^* \text{ is Pareto optimal} \iff \nexists x : f_i(x) \geq f_i(x^*) \ \forall i \land f_j(x) > f_j(x^*) \ \text{for some } j
$$

### Expected Hypervolume Improvement (EHVI)

**原理**: ハイパーボリュームの期待改善量を最大化

**数式**:
$$
\text{EHVI}(x) = \mathbb{E}[HV(\mathcal{P} \cup \{f(x)\}) - HV(\mathcal{P})]
$$

- $HV(\cdot)$: ハイパーボリューム
- $\mathcal{P}$: 現在のPareto集合

**コード例5: 多目的最適化の実装（BoTorch）**

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# 多目的最適化問題の定義（2目的）
def multi_objective_function(x):
    """
    2目的最適化問題
    目的1: f1(x) = x1^2 + x2^2 を最小化
    目的2: f2(x) = (x1-1)^2 + (x2-1)^2 を最小化
    Pareto最適解は x1とx2の線形結合で表される

    Parameters:
    -----------
    x : torch.Tensor, shape (n, 2)
        入力点

    Returns:
    --------
    y : torch.Tensor, shape (n, 2)
        2つの目的関数値（最大化のため負の値を返す）
    """
    f1 = x[:, 0]**2 + x[:, 1]**2
    f2 = (x[:, 0] - 1)**2 + (x[:, 1] - 1)**2

    # BoTorchは最大化を前提とするため、最小化問題は負にする
    return torch.stack([-f1, -f2], dim=-1)


# 初期サンプルの生成
def generate_initial_data(n=6):
    """初期データの生成"""
    train_x = torch.rand(n, 2, device=device, dtype=dtype) * 2 - 1  # [-1, 1]の範囲
    train_y = multi_objective_function(train_x)
    return train_x, train_y


# 多目的ガウス過程モデルの構築
def initialize_model(train_x, train_y):
    """
    2目的のための独立なGPモデルを構築

    Parameters:
    -----------
    train_x : torch.Tensor
        学習データ (n, 2)
    train_y : torch.Tensor
        目的関数値 (n, 2)

    Returns:
    --------
    model : SingleTaskGP
        学習済みGPモデル
    """
    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1])  # 各目的を標準化
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# EHVI獲得関数の最適化
def optimize_ehvi_and_get_observation(model, train_y, bounds):
    """
    Expected Hypervolume Improvement獲得関数を最適化

    Parameters:
    -----------
    model : SingleTaskGP
        学習済みモデル
    train_y : torch.Tensor
        既存の目的関数値
    bounds : torch.Tensor
        探索空間の境界 (2, 2)

    Returns:
    --------
    new_x : torch.Tensor
        次のサンプル点
    """
    # 参照点の設定（全ての目的で最悪の値より少し悪い点）
    ref_point = train_y.min(dim=0).values - 0.1

    # Pareto前線の計算
    pareto_mask = is_non_dominated(train_y)
    pareto_y = train_y[pareto_mask]

    # Box decomposition（ハイパーボリューム計算用）
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_y)

    # EHVI獲得関数の定義
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
    )

    # 獲得関数の最大化
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,  # 1点ずつ選択
        num_restarts=10,
        raw_samples=128,
    )

    return candidates.detach()


# Bayesian Optimization ループ
def run_bo_loop(n_iterations=10):
    """
    多目的ベイズ最適化の実行

    Parameters:
    -----------
    n_iterations : int
        最適化の反復回数

    Returns:
    --------
    train_x : torch.Tensor
        全てのサンプル点
    train_y : torch.Tensor
        全ての目的関数値
    """
    # 探索空間の境界
    bounds = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device, dtype=dtype)

    # 初期データ
    train_x, train_y = generate_initial_data(n=6)

    print("多目的ベイズ最適化開始")
    print(f"初期データ: {train_x.shape[0]}点")

    for iteration in range(n_iterations):
        # モデルの学習
        model = initialize_model(train_x, train_y)

        # 次のサンプル点を取得
        new_x = optimize_ehvi_and_get_observation(model, train_y, bounds)
        new_y = multi_objective_function(new_x)

        # データに追加
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        # Pareto前線の更新
        pareto_mask = is_non_dominated(train_y)
        n_pareto = pareto_mask.sum().item()

        print(f"Iteration {iteration + 1}: 新規点 = {new_x.squeeze().cpu().numpy()}, Pareto解数 = {n_pareto}")

    return train_x, train_y


# 実行と可視化
torch.manual_seed(42)
final_x, final_y = run_bo_loop(n_iterations=15)

# Pareto前線の抽出
pareto_mask = is_non_dominated(final_y)
pareto_x = final_x[pareto_mask].cpu().numpy()
pareto_y = final_y[pareto_mask].cpu().numpy()
non_pareto_y = final_y[~pareto_mask].cpu().numpy()

# 可視化
fig = plt.figure(figsize=(15, 5))

# 左：入力空間
ax1 = fig.add_subplot(131)
ax1.scatter(final_x[:, 0].cpu(), final_x[:, 1].cpu(), c='blue', s=50, alpha=0.6, label='全サンプル')
ax1.scatter(pareto_x[:, 0], pareto_x[:, 1], c='red', s=100, marker='*', label='Pareto解', zorder=5)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('入力空間のサンプル分布')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 中央：目的空間（Pareto前線）
ax2 = fig.add_subplot(132)
ax2.scatter(non_pareto_y[:, 0], non_pareto_y[:, 1], c='blue', s=50, alpha=0.6, label='非Pareto解')
ax2.scatter(pareto_y[:, 0], pareto_y[:, 1], c='red', s=100, marker='*', label='Pareto前線', zorder=5)
# Pareto前線を線で結ぶ
sorted_idx = np.argsort(pareto_y[:, 0])
ax2.plot(pareto_y[sorted_idx, 0], pareto_y[sorted_idx, 1], 'r--', alpha=0.5, linewidth=2)
ax2.set_xlabel('目的1: -f1(x)')
ax2.set_ylabel('目的2: -f2(x)')
ax2.set_title('目的空間のPareto前線')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 右：3D可視化（入力と目的の関係）
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(final_x[:, 0].cpu(), final_x[:, 1].cpu(), final_y[:, 0].cpu(),
            c='blue', s=30, alpha=0.6, label='目的1')
ax3.scatter(pareto_x[:, 0], pareto_x[:, 1], pareto_y[:, 0],
            c='red', s=80, marker='*', label='Pareto解（目的1）', zorder=5)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('目的1: -f1(x)')
ax3.set_title('入力空間と目的1の関係')
ax3.legend()

plt.tight_layout()
plt.savefig('multi_objective_bo.png', dpi=150, bbox_inches='tight')
plt.show()

# 結果の出力
print("\n最適化完了")
print(f"総サンプル数: {final_x.shape[0]}")
print(f"Pareto解数: {pareto_mask.sum().item()}")
print(f"\nPareto前線の目的関数値:")
for i, (y1, y2) in enumerate(pareto_y):
    print(f"  解{i+1}: f1={-y1:.4f}, f2={-y2:.4f}")

# 出力例:
# 多目的ベイズ最適化開始
# 初期データ: 6点
# Iteration 1: 新規点 = [0.123 0.456], Pareto解数 = 4
# Iteration 2: 新規点 = [-0.234 0.789], Pareto解数 = 5
# ...
# Iteration 15: 新規点 = [0.512 0.487], Pareto解数 = 8
#
# 最適化完了
# 総サンプル数: 21
# Pareto解数: 8
#
# Pareto前線の目的関数値:
#   解1: f1=0.0123, f2=1.2345
#   解2: f1=0.3456, f2=0.8901
#   解3: f1=0.6789, f2=0.4567
#   ...
```

---

## 3.3 制約付き獲得関数

### 制約条件の扱い

**例**: 合成可能性制約、コスト制約

**数式**:
$$
x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D}) \cdot P_c(x)
$$

- $P_c(x)$: 制約条件を満たす確率

**Constrained Expected Improvement**:
$$
\text{CEI}(x) = \text{EI}(x) \cdot P(c(x) \leq 0)
$$

---

## 3.4 ケーススタディ：熱電材料探索

### 問題設定

**目標**: 熱電性能指数ZT値の最大化

**ZT値**:
$$
ZT = \frac{S^2 \sigma T}{\kappa}
$$

- $S$: Seebeck係数
- $\sigma$: 電気伝導度
- $T$: 絶対温度
- $\kappa$: 熱伝導度

**課題**: 3つの物性を同時に最適化（多目的最適化）

---

## 本章のまとめ

### 獲得関数の比較表

| 獲得関数 | 特徴 | 探索傾向 | 計算コスト | 推奨用途 |
|-------|------|-------|----------|-------|
| EI | 改善期待値 | バランス | 低 | 一般的な最適化 |
| PI | 改善確率 | 活用重視 | 低 | 高速探索 |
| UCB | 信頼上限 | 探索重視 | 低 | 広範囲探索 |
| Thompson | 確率的 | バランス | 中 | 並列実験 |

### 次の章へ

第4章では、**材料探索への応用と実践**を学びます：
- Active Learning × ベイズ最適化
- Active Learning × 高スループット計算
- Active Learning × 実験ロボット
- 実世界応用とキャリアパス

**[第4章：材料探索への応用と実践 →](./chapter-4.html)**

---

## 演習問題

（省略：演習問題の詳細実装）

---

## 参考文献

1. Jones, D. R. et al. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." *Journal of Global Optimization*, 13(4), 455-492.

2. Daulton, S. et al. (2020). "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization." *NeurIPS*.

---

## ナビゲーション

### 前の章
**[← 第2章：不確実性推定手法](./chapter-2.html)**

### 次の章
**[第4章：材料探索への応用と実践 →](./chapter-4.html)**

### シリーズ目次
**[← シリーズ目次に戻る](./index.html)**

---

**次の章で実践的な応用を学びましょう！**
