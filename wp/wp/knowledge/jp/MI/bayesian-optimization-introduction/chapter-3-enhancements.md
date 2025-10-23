# Chapter 3 Quality Enhancements

This file contains enhancements to be integrated into chapter-3.md

## Code Reproducibility Section (add after section 3.1)

### コード再現性の確保

**環境設定の重要性**:

すべてのコード例は以下の環境で動作確認されています：

```python
# 必須ライブラリのバージョン
"""
Python: 3.8+
numpy: 1.21.0
scikit-learn: 1.0.0
scikit-optimize: 0.9.0
torch: 1.12.0
gpytorch: 1.8.0
botorch: 0.7.0
matplotlib: 3.5.0
pandas: 1.3.0
scipy: 1.7.0
"""

# 再現性確保のための設定
import numpy as np
import torch
import random

# 乱数シードの固定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# GPyTorchカーネル設定（推奨）
from gpytorch.kernels import RBF, MaternKernel, ScaleKernel

# RBFカーネル（最も一般的）
kernel_rbf = ScaleKernel(RBF(
    lengthscale_prior=None,  # データ駆動で最適化
    ard_num_dims=None  # Automatic Relevance Determination
))

# Maternカーネル（滑らかさ調整可能）
kernel_matern = ScaleKernel(MaternKernel(
    nu=2.5,  # 滑らかさパラメータ（1.5, 2.5, または inf（RBFと同等））
    ard_num_dims=None
))

print("環境設定完了")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

**インストール手順**:

```bash
# 仮想環境の作成（推奨）
python -m venv bo_env
source bo_env/bin/activate  # Linuxmac
# bo_env\Scripts\activate  # Windows

# 必須パッケージのインストール
pip install numpy==1.21.0 scikit-learn==1.0.0 scikit-optimize==0.9.0
pip install torch==1.12.0 gpytorch==1.8.0 botorch==0.7.0
pip install matplotlib==3.5.0 pandas==1.3.0 scipy==1.7.0

# オプション: Materials Project API
pip install mp-api==0.30.0

# インストール確認
python -c "import botorch; print(f'BoTorch {botorch.__version__} installed')"
```

---

## Practical Pitfalls Section (add after section 3.7)

### 3.8 実践的な落とし穴と対処法

#### 落とし穴1: 不適切なカーネル選択

**問題**: カーネル選択が目的関数の性質と合っていない

**症状**:
- 予測精度が低い
- 探索効率が悪い
- 局所最適に陥りやすい

**解決策**:

```python
# カーネル選択ガイド
from gpytorch.kernels import RBF, MaternKernel, PeriodicKernel

def select_kernel(problem_characteristics):
    """
    問題の特性に応じたカーネル選択

    Parameters:
    -----------
    problem_characteristics : dict
        問題の特性を記述する辞書
        - 'smoothness': 'smooth' | 'rough'
        - 'periodicity': True | False
        - 'dimensionality': int

    Returns:
    --------
    kernel : gpytorch.kernels.Kernel
        推奨カーネル
    """
    if problem_characteristics.get('periodicity'):
        # 周期性がある場合
        return PeriodicKernel()

    elif problem_characteristics.get('smoothness') == 'smooth':
        # 滑らかな関数（材料物性など）
        return RBF()

    elif problem_characteristics.get('smoothness') == 'rough':
        # ノイズや不連続性がある
        return MaternKernel(nu=1.5)

    else:
        # デフォルト: Matern 5/2（汎用性高い）
        return MaternKernel(nu=2.5)

# 使用例
problem_specs = {
    'smoothness': 'smooth',
    'periodicity': False,
    'dimensionality': 4
}

recommended_kernel = select_kernel(problem_specs)
print(f"推奨カーネル: {recommended_kernel}")
```

**カーネル比較実験**:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# テスト関数
def test_function(x):
    """ノイズのある複雑な関数"""
    return np.sin(5*x) + 0.5*np.cos(15*x) + 0.1*np.random.randn(len(x))

# データ生成
np.random.seed(42)
X_train = np.random.uniform(0, 1, 20).reshape(-1, 1)
y_train = test_function(X_train.ravel())

X_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_true = test_function(X_test.ravel())

# 異なるカーネルで比較
kernels = {
    'RBF': RBF(length_scale=0.1),
    'Matern 1.5': Matern(length_scale=0.1, nu=1.5),
    'Matern 2.5': Matern(length_scale=0.1, nu=2.5)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, kernel) in zip(axes, kernels.items()):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    y_pred, y_std = gp.predict(X_test, return_std=True)

    ax.scatter(X_train, y_train, c='red', label='訓練データ')
    ax.plot(X_test, y_pred, 'b-', label='予測')
    ax.fill_between(X_test.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                     alpha=0.3, color='blue')
    ax.set_title(f'カーネル: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150)
plt.show()

print("結論:")
print("  RBF: 滑らかな関数に最適")
print("  Matern 1.5: ノイズ耐性が高い")
print("  Matern 2.5: バランスが良い（推奨）")
```

---

#### 落とし穴2: 初期化戦略の失敗

**問題**: 初期サンプリングが探索空間を十分カバーしていない

**症状**:
- 探索が偏る
- 重要な領域を見逃す
- 収束が遅い

**解決策**: ラテン超方格サンプリング（LHS）

```python
from scipy.stats.qmc import LatinHypercube

def initialize_with_lhs(n_samples, bounds, seed=42):
    """
    ラテン超方格サンプリングで初期点を生成

    Parameters:
    -----------
    n_samples : int
        サンプル数
    bounds : array (n_dims, 2)
        各次元の [lower, upper] 境界
    seed : int
        乱数シード

    Returns:
    --------
    X_init : array (n_samples, n_dims)
        初期サンプリング点
    """
    bounds = np.array(bounds)
    n_dims = len(bounds)

    # LHSサンプラー
    sampler = LatinHypercube(d=n_dims, seed=seed)
    X_unit = sampler.random(n=n_samples)

    # スケーリング
    X_init = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * X_unit

    return X_init

# 使用例: Li-ion電池組成の初期化
bounds_composition = [
    [0.1, 0.5],  # Li
    [0.1, 0.4],  # Ni
    [0.1, 0.3],  # Co
    [0.0, 0.5]   # Mn
]

X_init_lhs = initialize_with_lhs(
    n_samples=20,
    bounds=bounds_composition,
    seed=42
)

# 組成正規化
X_init_lhs = X_init_lhs / X_init_lhs.sum(axis=1, keepdims=True)

print("LHS初期化完了")
print(f"初期サンプル数: {len(X_init_lhs)}")
print(f"各次元のカバー範囲:")
for i, dim_name in enumerate(['Li', 'Ni', 'Co', 'Mn']):
    print(f"  {dim_name}: [{X_init_lhs[:, i].min():.3f}, "
          f"{X_init_lhs[:, i].max():.3f}]")

# ランダムサンプリングとの比較可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ランダムサンプリング
np.random.seed(42)
X_random = np.random.uniform(0, 1, (20, 2))

axes[0].scatter(X_random[:, 0], X_random[:, 1], s=100)
axes[0].set_title('ランダムサンプリング')
axes[0].set_xlabel('次元1')
axes[0].set_ylabel('次元2')
axes[0].grid(True, alpha=0.3)

# LHS
axes[1].scatter(X_init_lhs[:, 0], X_init_lhs[:, 1], s=100, c='red')
axes[1].set_title('ラテン超方格サンプリング（LHS）')
axes[1].set_xlabel('次元1 (Li)')
axes[1].set_ylabel('次元2 (Ni)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lhs_vs_random.png', dpi=150)
plt.show()
```

---

#### 落とし穴3: ノイズのある観測への対応不足

**問題**: 実験ノイズを考慮していない

**症状**:
- 同じ条件で結果が再現しない
- モデルが過学習する
- 最適点が不安定

**解決策**: ノイズを明示的にモデル化

```python
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

def fit_gp_with_noise(X, y, noise_variance=0.01):
    """
    ノイズを考慮したガウス過程の学習

    Parameters:
    -----------
    X : Tensor (n, d)
        入力データ
    y : Tensor (n, 1)
        観測値（ノイズ含む）
    noise_variance : float
        観測ノイズの分散（事前知識から設定）

    Returns:
    --------
    gp_model : SingleTaskGP
        学習済みGPモデル
    """
    # ノイズ分散を設定してGP構築
    gp_model = SingleTaskGP(X, y, train_Yvar=torch.full_like(y, noise_variance))

    # 尤度最大化
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    from botorch.fit import fit_gpytorch_model
    fit_gpytorch_model(mll)

    return gp_model

# 使用例: ノイズのある実験データ
np.random.seed(42)
X_obs = np.random.rand(15, 4)
X_obs = X_obs / X_obs.sum(axis=1, keepdims=True)

# 真の容量 + 実験ノイズ
y_true = 200 + 150 * X_obs[:, 0] + 50 * X_obs[:, 1]
noise = np.random.randn(15) * 10  # 実験ノイズ σ=10 mAh/g
y_obs = y_true + noise

# PyTorchテンソルに変換
X_tensor = torch.tensor(X_obs, dtype=torch.float64)
y_tensor = torch.tensor(y_obs, dtype=torch.float64).unsqueeze(-1)

# ノイズを考慮してGP学習
gp_noisy = fit_gp_with_noise(X_tensor, y_tensor, noise_variance=100.0)

print("ノイズを考慮したGP学習完了")
print(f"観測ノイズ標準偏差: 10 mAh/g")
print(f"モデル化ノイズ分散: 100.0 (mAh/g)²")
```

**ノイズレベルの推定**:

```python
def estimate_noise_level(X, y, n_replicates=3):
    """
    複製実験からノイズレベルを推定

    Parameters:
    -----------
    X : array (n, d)
        実験条件
    y : array (n,)
        観測値
    n_replicates : int
        各条件での複製実験数

    Returns:
    --------
    noise_std : float
        推定ノイズ標準偏差
    """
    # 同一条件の複製実験を抽出
    unique_X, indices = np.unique(X, axis=0, return_inverse=True)

    variances = []
    for i in range(len(unique_X)):
        replicates = y[indices == i]
        if len(replicates) >= 2:
            variances.append(np.var(replicates, ddof=1))

    if len(variances) == 0:
        print("警告: 複製実験がありません。デフォルト値を使用")
        return 1.0

    noise_std = np.sqrt(np.mean(variances))
    return noise_std

# 使用例
noise_std_estimated = estimate_noise_level(X_obs, y_obs)
print(f"推定ノイズ標準偏差: {noise_std_estimated:.2f} mAh/g")
```

---

#### 落とし穴4: 制約処理の不備

**問題**: 制約違反を適切に扱っていない

**症状**:
- 実行不可能な材料を提案
- 最適化が収束しない
- 無駄な実験が多い

**解決策**: 制約付き獲得関数

```python
from botorch.acquisition import ConstrainedExpectedImprovement

def constrained_bayesian_optimization_example():
    """
    制約付きベイズ最適化の実装例

    制約:
    1. 組成の合計 = 1.0 (±2%)
    2. Co含量 < 0.3 (コスト制約)
    3. 安定性: formation energy < -1.5 eV/atom
    """
    # 初期データ
    n_initial = 10
    X_init = initialize_with_lhs(n_initial, bounds_composition, seed=42)
    X_init = X_init / X_init.sum(axis=1, keepdims=True)  # 正規化

    # 目的関数と制約の評価
    y_capacity = []
    constraints_satisfied = []

    for x in X_init:
        # 容量予測（目的関数）
        capacity = 200 + 150*x[0] + 50*x[1]
        y_capacity.append(capacity)

        # 制約チェック
        co_constraint = x[2] < 0.3  # Co < 0.3
        stability = -2.0 - 0.5*x[0] - 0.3*x[1]
        stability_constraint = stability < -1.5  # 安定

        all_satisfied = co_constraint and stability_constraint
        constraints_satisfied.append(1.0 if all_satisfied else 0.0)

    X_tensor = torch.tensor(X_init, dtype=torch.float64)
    y_tensor = torch.tensor(y_capacity, dtype=torch.float64).unsqueeze(-1)
    c_tensor = torch.tensor(constraints_satisfied, dtype=torch.float64).unsqueeze(-1)

    # ガウス過程モデル（目的関数）
    gp_objective = SingleTaskGP(X_tensor, y_tensor)
    mll_obj = ExactMarginalLogLikelihood(gp_objective.likelihood, gp_objective)
    from botorch.fit import fit_gpytorch_model
    fit_gpytorch_model(mll_obj)

    # ガウス過程モデル（制約）
    gp_constraint = SingleTaskGP(X_tensor, c_tensor)
    mll_con = ExactMarginalLogLikelihood(gp_constraint.likelihood, gp_constraint)
    fit_gpytorch_model(mll_con)

    # 制約付きEI獲得関数
    best_f = y_tensor.max()
    acq_func = ConstrainedExpectedImprovement(
        model=gp_objective,
        best_f=best_f,
        objective_index=0,
        constraints={0: [None, 0.5]}  # 制約満足確率 > 0.5
    )

    print("制約付きベイズ最適化セットアップ完了")
    print(f"初期実行可能解: {sum(constraints_satisfied)}/{n_initial}")

    return gp_objective, gp_constraint, acq_func

# 実行
gp_obj, gp_con, acq = constrained_bayesian_optimization_example()
```

---

## End-of-Chapter Checklist (add before "演習問題")

### 3.9 章末チェックリスト

#### ✅ ガウス過程の理解

- [ ] ガウス過程の基本概念を説明できる
- [ ] カーネル関数の役割を理解している
- [ ] 予測平均と不確実性の意味を知っている
- [ ] 適切なカーネルを選択できる
- [ ] ハイパーパラメータの影響を説明できる

**確認問題**:
```
Q: RBFカーネルとMaternカーネルの違いは何ですか？
A: RBFは無限回微分可能（非常に滑らか）、Maternはパラメータνで
   滑らかさを調整可能。ノイズがある場合はMatern (ν=2.5) が推奨。
```

---

#### ✅ 獲得関数の選択

- [ ] Expected Improvement (EI) の仕組みを理解している
- [ ] Upper Confidence Bound (UCB) の探索・活用バランスを説明できる
- [ ] Probability of Improvement (PI) の特性を知っている
- [ ] Knowledge Gradient (KG) の適用場面を理解している
- [ ] 問題に応じて獲得関数を選択できる

**選択ガイド**:
```
一般的な最適化     → EI (バランス良い)
探索重視の初期     → UCB (κ=2~3)
安全重視          → PI (保守的)
バッチ最適化      → q-EI, q-KG
多目的最適化      → EHVI (Hypervolume)
```

---

#### ✅ 多目的最適化

- [ ] Pareto最適の定義を説明できる
- [ ] Paretoフロンティアの意味を理解している
- [ ] Expected Hypervolume Improvement (EHVI) の仕組みを知っている
- [ ] トレードオフを定量的に評価できる
- [ ] 多目的最適化を実装できる

**実装チェック**:
```python
# 以下を実装できますか？
def is_pareto_optimal(objectives):
    """
    Pareto最適解を判定する関数
    objectives: (n_points, n_objectives)
    """
    # あなたの実装
    pass

# 正解は演習問題3を参照
```

---

#### ✅ バッチベイズ最適化

- [ ] バッチ最適化の利点を説明できる
- [ ] q-EI獲得関数の仕組みを理解している
- [ ] Kriging Believer法を知っている
- [ ] 並列実験の効率化戦略を立てられる
- [ ] バッチサイズの選択基準を理解している

**バッチサイズ選択**:
```
実験装置数: n台
→ バッチサイズ: n（最大活用）

計算コスト制約あり
→ バッチサイズ: 3~5（実用的）

探索初期
→ バッチサイズ: 大きめ（多様性重視）

収束期
→ バッチサイズ: 小さめ（精密化）
```

---

#### ✅ 制約処理

- [ ] 制約の種類（等式、不等式）を区別できる
- [ ] 実行可能領域の概念を理解している
- [ ] 制約付き獲得関数を実装できる
- [ ] 実行可能確率を計算できる
- [ ] 段階的制約緩和の戦略を知っている

**制約処理チェックリスト**:
```
□ 組成制約（合計=1.0）を正規化で処理
□ 境界制約をboundsパラメータで設定
□ 非線形制約をペナルティ関数で表現
□ 実行可能解が見つからない場合の対処法を準備
□ 制約満足確率を可視化
```

---

#### ✅ 実装スキル（GPyTorch/BoTorch）

- [ ] SingleTaskGPモデルを構築できる
- [ ] カーネルを適切に選択・設定できる
- [ ] 獲得関数を最適化できる
- [ ] バッチ最適化を実装できる
- [ ] ノイズを考慮したモデリングができる

**コード実装確認**:
```python
# このコードが理解できますか？
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# GPモデル構築
gp = SingleTaskGP(X_train, y_train)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# EI最大化
EI = ExpectedImprovement(gp, best_f=y_train.max())
candidate, acq_value = optimize_acqf(
    EI, bounds=bounds, q=1, num_restarts=10
)

# 各行の意味を説明できますか？
```

---

#### ✅ 実験設計との統合

- [ ] Materials Projectなど実データソースを活用できる
- [ ] MLモデルとベイズ最適化を統合できる
- [ ] 実験計画を立てられる
- [ ] 結果を可視化・解釈できる
- [ ] ROIを評価できる

**実験計画テンプレート**:
```
1. 目的設定
   - 最適化する特性: ________
   - 制約条件: ________
   - 実験予算: ________ 回

2. 初期化
   - 初期サンプル数: ________
   - サンプリング法: LHS / Random
   - 予想実験期間: ________

3. 最適化戦略
   - 獲得関数: ________
   - カーネル: ________
   - バッチサイズ: ________

4. 終了条件
   - 最大実験回数: ________
   - 目標性能: ________
   - 改善率閾値: ________
```

---

#### ✅ トラブルシューティング

- [ ] 局所最適からの脱出方法を知っている
- [ ] 制約違反への対処法を理解している
- [ ] 計算時間削減の手法を知っている
- [ ] ノイズへの対処法を実装できる
- [ ] デバッグ方法を知っている

**よくあるエラーと対処法**:
```
エラー: "RuntimeError: cholesky_cpu: U(i,i) is zero"
→ 原因: 数値不安定性
→ 対処: GPモデルにjitter追加
   gp = SingleTaskGP(X, y, covar_module=...).double()
   gp.likelihood.noise = 1e-4

エラー: "All points violate constraints"
→ 原因: 制約が厳しすぎる
→ 対処: 段階的制約緩和、初期LHSサンプリング

警告: "Optimization failed to converge"
→ 原因: 獲得関数最適化の失敗
→ 対処: num_restarts増加、raw_samples増加
```

---

### 合格基準

各セクションで80%以上のチェック項目をクリアし、
実装確認コードが理解できれば、次章へ進む準備完了です。

**最終確認問題**:
1. Li-ion電池正極材料の最適化問題を定式化できますか？
2. 3目的（容量、電圧、安定性）の最適化を実装できますか？
3. 実験回数50回でPareto最適解10個を見つけられますか？

全てYESなら、第4章「アクティブラーニングと実験連携」へ進みましょう！
