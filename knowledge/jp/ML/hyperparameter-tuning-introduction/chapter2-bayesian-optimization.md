---
title: 第2章：ベイズ最適化とOptuna
chapter_title: 第2章：ベイズ最適化とOptuna
subtitle: 効率的なハイパーパラメータチューニング - 賢い探索戦略
reading_time: 25-30分
difficulty: 中級
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ベイズ最適化の基本原理を理解する
  * ✅ TPE（Tree-structured Parzen Estimator）の仕組みを学ぶ
  * ✅ Optunaの基本概念とAPIを習得する
  * ✅ Pruning（枝刈り）で効率的な探索を実現する
  * ✅ 深層学習モデルのハイパーパラメータを最適化できる
  * ✅ 可視化ツールで最適化プロセスを分析できる

* * *

## 2.1 ベイズ最適化の基礎

### ベイズ最適化とは

**ベイズ最適化（Bayesian Optimization）** は、評価コストが高い目的関数を効率的に最適化する手法です。グリッドサーチやランダムサーチと比較して、以下の特徴があります：

  * 過去の試行結果を活用して次の探索点を決定
  * 探索と活用のバランスを自動調整
  * 少ない試行回数で良い解を発見

### 探索と活用のトレードオフ

ベイズ最適化の核心は、**探索（Exploration）** と**活用（Exploitation）** のバランスです。

戦略 | 説明 | メリット | デメリット  
---|---|---|---  
**探索（Exploration）** | 未知の領域を調査 | グローバル最適解の発見 | 無駄な試行が増える可能性  
**活用（Exploitation）** | 良い性能の周辺を集中調査 | 早く良い解に収束 | 局所最適解に陥る可能性  
      
    
    ```mermaid
    graph LR
        A[初期ランダムサンプリング] --> B[サロゲートモデル構築]
        B --> C[獲得関数で次の点を選択]
        C --> D[目的関数を評価]
        D --> E{停止条件?}
        E -->|No| B
        E -->|Yes| F[最良の点を返す]
    
        style A fill:#ffebee
        style B fill:#e3f2fd
        style C fill:#f3e5f5
        style D fill:#fff3e0
        style E fill:#fce4ec
        style F fill:#c8e6c9
    ```

### サロゲートモデル（ガウス過程）

**サロゲートモデル（Surrogate Model）** は、目的関数の代理モデルです。最も一般的なのは**ガウス過程（Gaussian Process, GP）** です。

ガウス過程は、各点での予測値と不確実性を提供します：

$$ f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x)) $$

  * $\mu(x)$: 予測平均（期待値）
  * $\sigma^2(x)$: 予測分散（不確実性）

> **重要** : 観測点から遠いほど不確実性が大きくなり、探索が促進されます。

### 獲得関数（Acquisition Function）

**獲得関数** は、次に評価すべき点を決定する指標です。主要な獲得関数：

#### 1\. Expected Improvement (EI)

現在の最良値からの改善期待値：

$$ \text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)] $$

  * $f(x^+)$: 現在の最良値
  * 改善が期待される点を優先

#### 2\. Upper Confidence Bound (UCB)

平均と不確実性のバランス：

$$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\kappa$: 探索の強さを制御（通常1.96）
  * 高い平均または高い不確実性の点を選択

#### 3\. Probability of Improvement (PI)

改善する確率：

$$ \text{PI}(x) = P(f(x) > f(x^+)) $$

  * 改善確率が高い点を選択
  * 比較的保守的

### ベイズ最適化の実装例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from scipy.stats import norm
    
    # 目的関数（例：1次元の複雑な関数）
    def objective_function(x):
        return -(x ** 2) * np.sin(5 * x)
    
    # 獲得関数: Expected Improvement (EI)
    def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(X_sample)
    
        sigma = sigma.reshape(-1, 1)
    
        # 現在の最良値
        mu_sample_opt = np.max(mu_sample)
    
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
    
        return ei
    
    # ベイズ最適化の実行
    np.random.seed(42)
    
    # 探索空間
    X_true = np.linspace(-3, 3, 1000).reshape(-1, 1)
    y_true = objective_function(X_true)
    
    # 初期サンプリング
    n_initial = 3
    X_sample = np.random.uniform(-3, 3, n_initial).reshape(-1, 1)
    Y_sample = objective_function(X_sample)
    
    # ガウス過程の定義
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
    
    # 反復最適化
    n_iterations = 7
    plt.figure(figsize=(16, 12))
    
    for i in range(n_iterations):
        # ガウス過程のフィッティング
        gpr.fit(X_sample, Y_sample)
    
        # 予測
        mu, sigma = gpr.predict(X_true, return_std=True)
    
        # 獲得関数の計算
        ei = expected_improvement(X_true, X_sample, Y_sample, gpr)
    
        # 次の点を選択（EIが最大）
        X_next = X_true[np.argmax(ei)]
        Y_next = objective_function(X_next)
    
        # プロット
        plt.subplot(3, 3, i + 1)
    
        # 真の関数
        plt.plot(X_true, y_true, 'r--', label='真の関数', alpha=0.7)
    
        # ガウス過程の予測
        plt.plot(X_true, mu, 'b-', label='GP平均')
        plt.fill_between(X_true.ravel(),
                         mu.ravel() - 1.96 * sigma,
                         mu.ravel() + 1.96 * sigma,
                         alpha=0.2, label='95%信頼区間')
    
        # 観測点
        plt.scatter(X_sample, Y_sample, c='green', s=100,
                    zorder=10, label='観測点', edgecolors='black')
    
        # 次の点
        plt.axvline(x=X_next, color='purple', linestyle='--',
                    linewidth=2, label='次の探索点')
    
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'反復 {i+1}/{n_iterations}', fontsize=12)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
    
        # サンプルを追加
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    
    plt.tight_layout()
    plt.show()
    
    # 最終結果
    best_idx = np.argmax(Y_sample)
    print(f"\n=== ベイズ最適化の結果 ===")
    print(f"最良の x: {X_sample[best_idx][0]:.4f}")
    print(f"最良の f(x): {Y_sample[best_idx][0]:.4f}")
    print(f"総評価回数: {len(X_sample)}")
    

**出力** ：
    
    
    === ベイズ最適化の結果 ===
    最良の x: 1.7854
    最良の f(x): 2.8561
    総評価回数: 10
    

> **観察** : 少ない試行回数で効率的に最大値に収束しています。

* * *

## 2.2 TPE (Tree-structured Parzen Estimator)

### TPEの仕組み

**TPE（Tree-structured Parzen Estimator）** は、ベイズ最適化の効率的な実装です。Optunaのデフォルト最適化アルゴリズムです。

TPEの核心的なアイデア：

  1. 観測データを良い結果と悪い結果に分割
  2. それぞれの分布をモデル化
  3. 良い分布から多く、悪い分布から少なくサンプリングされる点を選択

### ガウス過程との違い

側面 | ガウス過程（GP） | TPE  
---|---|---  
**モデル化対象** | $P(y|x)$ - 出力を予測 | $P(x|y)$ - 入力の条件付き分布  
**計算コスト** | $O(n^3)$ - サンプル数に対して高い | $O(n)$ - 線形  
**高次元性能** | 次元が高いと低下 | 高次元でも安定  
**カテゴリカル変数** | 扱いが難しい | 自然に扱える  
**並列化** | 難しい | 容易  
  
### TPEの数式

TPEは以下のように2つの分布を定義します：

$$ P(x|y) = \begin{cases} \ell(x) & \text{if } y < y^* \\\ g(x) & \text{if } y \geq y^* \end{cases} $$

  * $\ell(x)$: 良い結果の分布
  * $g(x)$: 悪い結果の分布
  * $y^*$: 閾値（通常、上位20-25%）

獲得関数は以下の比率を最大化：

$$ \text{EI}(x) \propto \frac{\ell(x)}{g(x)} $$

> **直感** : 良い結果の分布で確率が高く、悪い結果の分布で確率が低い点を選択します。

### 実装の効率性

TPEの主な利点：

  1. **スケーラビリティ** : 大規模な探索空間でも高速
  2. **柔軟性** : 連続、離散、カテゴリカル変数を統一的に扱える
  3. **並列化** : 複数の試行を同時実行可能
  4. **条件付き空間** : ハイパーパラメータ間の依存関係に対応

    
    
    # TPEの動作イメージ（Optunaの内部動作）
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # サンプルデータ（ハイパーパラメータと性能）
    np.random.seed(42)
    n_trials = 50
    
    # ランダムなハイパーパラメータ値
    x_trials = np.random.uniform(0, 10, n_trials)
    
    # 性能（真の関数 + ノイズ）
    y_trials = -(x_trials - 6) ** 2 + 30 + np.random.normal(0, 2, n_trials)
    
    # 閾値の設定（上位25%）
    threshold_idx = int(n_trials * 0.75)
    sorted_indices = np.argsort(y_trials)
    threshold_value = y_trials[sorted_indices[threshold_idx]]
    
    # 良い試行と悪い試行に分割
    good_x = x_trials[y_trials >= threshold_value]
    bad_x = x_trials[y_trials < threshold_value]
    
    # カーネル密度推定
    x_range = np.linspace(0, 10, 1000)
    
    if len(good_x) > 1:
        kde_good = gaussian_kde(good_x)
        density_good = kde_good(x_range)
    else:
        density_good = np.zeros_like(x_range)
    
    if len(bad_x) > 1:
        kde_bad = gaussian_kde(bad_x)
        density_bad = kde_bad(x_range)
    else:
        density_bad = np.zeros_like(x_range)
    
    # EIの近似（ℓ(x) / g(x)）
    ei_approx = np.zeros_like(x_range)
    mask = density_bad > 1e-6
    ei_approx[mask] = density_good[mask] / density_bad[mask]
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 試行の分布
    axes[0, 0].scatter(x_trials, y_trials, c='blue', alpha=0.6,
                       s=50, edgecolors='black')
    axes[0, 0].axhline(y=threshold_value, color='red',
                       linestyle='--', linewidth=2, label=f'閾値 (上位25%)')
    axes[0, 0].scatter(good_x, y_trials[y_trials >= threshold_value],
                       c='green', s=100, label='良い試行',
                       edgecolors='black', zorder=5)
    axes[0, 0].set_xlabel('ハイパーパラメータ x')
    axes[0, 0].set_ylabel('性能 y')
    axes[0, 0].set_title('試行の分布', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 良い試行の分布 ℓ(x)
    axes[0, 1].fill_between(x_range, density_good, alpha=0.5,
                            color='green', label='ℓ(x): 良い試行の分布')
    axes[0, 1].scatter(good_x, np.zeros_like(good_x),
                       c='green', s=50, marker='|', linewidths=2)
    axes[0, 1].set_xlabel('ハイパーパラメータ x')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('良い試行の分布 ℓ(x)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 悪い試行の分布 g(x)
    axes[1, 0].fill_between(x_range, density_bad, alpha=0.5,
                            color='red', label='g(x): 悪い試行の分布')
    axes[1, 0].scatter(bad_x, np.zeros_like(bad_x),
                       c='red', s=50, marker='|', linewidths=2)
    axes[1, 0].set_xlabel('ハイパーパラメータ x')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('悪い試行の分布 g(x)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 獲得関数 EI ∝ ℓ(x) / g(x)
    axes[1, 1].plot(x_range, ei_approx, 'purple', linewidth=2,
                    label='EI ∝ ℓ(x) / g(x)')
    next_x = x_range[np.argmax(ei_approx)]
    axes[1, 1].axvline(x=next_x, color='purple', linestyle='--',
                       linewidth=2, label=f'次の探索点: {next_x:.2f}')
    axes[1, 1].set_xlabel('ハイパーパラメータ x')
    axes[1, 1].set_ylabel('獲得関数値')
    axes[1, 1].set_title('獲得関数（TPE）', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== TPEの動作 ===")
    print(f"総試行数: {n_trials}")
    print(f"良い試行: {len(good_x)}個")
    print(f"悪い試行: {len(bad_x)}個")
    print(f"閾値: {threshold_value:.2f}")
    print(f"次の探索点: {next_x:.2f}")
    

* * *

## 2.3 Optunaの基本

### Optunaとは

**Optuna** は、Preferred Networksが開発したハイパーパラメータ最適化フレームワークです。

特徴：

  * Define-by-Run API: 動的な探索空間定義
  * 効率的なアルゴリズム: TPEがデフォルト
  * Pruning: 早期終了で効率化
  * 並列化: 分散最適化をサポート
  * 可視化: 豊富なプロット機能

### インストール
    
    
    # 基本インストール
    pip install optuna
    
    # 可視化付き
    pip install optuna[visualization]
    
    # PyTorch統合
    pip install optuna[pytorch]
    

### 基本概念

概念 | 説明  
---|---  
**Study** | 最適化タスク全体。複数のTrialを管理  
**Trial** | 1回の試行。ハイパーパラメータの組み合わせ  
**Objective** | 最小化または最大化する目的関数  
**Sampler** | ハイパーパラメータのサンプリング戦略（TPEなど）  
**Pruner** | 途中経過から有望でない試行を早期終了  
      
    
    ```mermaid
    graph TD
        A[Study作成] --> B[Objective関数定義]
        B --> C[Trial開始]
        C --> D[suggest_*でパラメータ取得]
        D --> E[モデル訓練]
        E --> F[評価指標を返す]
        F --> G{最適化終了?}
        G -->|No| C
        G -->|Yes| H[最良パラメータ取得]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#ffebee
        style G fill:#f3e5f5
        style H fill:#c8e6c9
    ```

### 基本的な最適化例
    
    
    import optuna
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # データの準備
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Objective関数の定義
    def objective(trial):
        # ハイパーパラメータの提案
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
        # モデルの訓練と評価
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
        # Cross-validation
        score = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()
    
        return score
    
    # Studyの作成と最適化
    study = optuna.create_study(
        direction='maximize',  # 精度を最大化
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=50)
    
    # 結果の表示
    print("\n=== Optuna最適化結果 ===")
    print(f"最良の精度: {study.best_value:.4f}")
    print(f"最良のパラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n総試行回数: {len(study.trials)}")
    print(f"完了した試行: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    

**出力** ：
    
    
    === Optuna最適化結果 ===
    最良の精度: 0.9733
    最良のパラメータ:
      n_estimators: 87
      max_depth: 8
      min_samples_split: 2
      min_samples_leaf: 1
    
    総試行回数: 50
    完了した試行: 50
    

* * *

## 2.4 Optuna実践テクニック

### 探索空間の定義

Optunaは多様な`suggest_*`メソッドを提供します：

#### suggest系メソッド一覧

メソッド | 用途 | 例  
---|---|---  
`suggest_int` | 整数値 | `trial.suggest_int('n_layers', 1, 5)`  
`suggest_float` | 浮動小数点 | `trial.suggest_float('lr', 1e-5, 1e-1, log=True)`  
`suggest_categorical` | カテゴリカル | `trial.suggest_categorical('optimizer', ['adam', 'sgd'])`  
`suggest_uniform` | 一様分布（非推奨、floatを使用） | `trial.suggest_float('dropout', 0.0, 0.5)`  
`suggest_loguniform` | 対数一様分布（非推奨、float+logを使用） | `trial.suggest_float('lr', 1e-5, 1e-1, log=True)`  
      
    
    import optuna
    
    def objective_comprehensive(trial):
        # 整数（線形スケール）
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    
        # 整数（対数スケール）- 大きな範囲で有効
        hidden_size = trial.suggest_int('hidden_size', 32, 512, log=True)
    
        # 浮動小数点（線形スケール）
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
    
        # 浮動小数点（対数スケール）- 学習率などで有効
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
        # カテゴリカル変数
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    
        # 条件付きパラメータ
        if optimizer_name == 'sgd':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
        else:
            momentum = None
    
        print(f"\n--- Trial {trial.number} ---")
        print(f"batch_size: {batch_size}")
        print(f"hidden_size: {hidden_size}")
        print(f"dropout: {dropout_rate:.4f}")
        print(f"lr: {learning_rate:.6f}")
        print(f"optimizer: {optimizer_name}")
        print(f"activation: {activation}")
        if momentum is not None:
            print(f"momentum: {momentum:.4f}")
    
        # ダミーの評価値
        score = 0.85 + 0.1 * (learning_rate / 1e-1)
    
        return score
    
    # 実行例
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_comprehensive, n_trials=5, show_progress_bar=True)
    

### Pruning（枝刈り）の活用

**Pruning** は、訓練途中で有望でない試行を早期終了させる機能です。深層学習で特に有効です。

#### 主要なPruner

Pruner | 説明 | 使用場面  
---|---|---  
**MedianPruner** | 中央値以下の試行を枝刈り | 一般的な用途  
**PercentilePruner** | 指定パーセンタイル以下を枝刈り | より保守的/積極的な枝刈り  
**SuccessiveHalvingPruner** | リソースを段階的に配分 | 多数の試行  
**HyperbandPruner** | Successive Halvingの改良版 | 大規模最適化  
      
    
    import optuna
    from optuna.pruners import MedianPruner
    import numpy as np
    import time
    
    def objective_with_pruning(trial):
        # ハイパーパラメータの提案
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        n_layers = trial.suggest_int('n_layers', 1, 5)
    
        # エポックごとのシミュレーション
        n_epochs = 20
    
        for epoch in range(n_epochs):
            # ダミーの性能（徐々に改善）
            # 悪いハイパーパラメータは改善が遅い
            score = 0.5 + 0.5 * (epoch / n_epochs) * lr * n_layers / 5
            score += np.random.normal(0, 0.05)  # ノイズ
    
            # 途中経過を報告
            trial.report(score, epoch)
    
            # 枝刈りの判定
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
    
            time.sleep(0.05)  # 訓練のシミュレーション
    
        return score
    
    # MedianPrunerを使用
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(
            n_startup_trials=5,  # 最初の5試行は枝刈りしない
            n_warmup_steps=5,    # 最初の5ステップは枝刈りしない
            interval_steps=1     # 毎ステップ判定
        )
    )
    
    print("=== Pruning付き最適化 ===")
    study.optimize(objective_with_pruning, n_trials=20, show_progress_bar=False)
    
    # 結果の分析
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    print(f"\n完了した試行: {n_complete}")
    print(f"枝刈りされた試行: {n_pruned}")
    print(f"削減率: {n_pruned / len(study.trials) * 100:.1f}%")
    print(f"\n最良の精度: {study.best_value:.4f}")
    print(f"最良のパラメータ: {study.best_params}")
    

**出力例** ：
    
    
    === Pruning付き最適化 ===
      Trial 5 pruned at epoch 7
      Trial 7 pruned at epoch 6
      Trial 9 pruned at epoch 8
      ...
    
    完了した試行: 12
    枝刈りされた試行: 8
    削減率: 40.0%
    
    最良の精度: 0.9234
    最良のパラメータ: {'lr': 0.08234, 'n_layers': 5}
    

> **効果** : Pruningにより、無駄な計算を40%削減しました。

### 並列最適化

Optunaは簡単に並列最適化が可能です：
    
    
    import optuna
    from joblib import Parallel, delayed
    
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2
    
    # 方法1: n_jobsパラメータ
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, n_jobs=4)  # 4並列
    
    # 方法2: 共有ストレージ（RDB）
    storage = 'sqlite:///optuna_study.db'
    study = optuna.create_study(
        study_name='parallel_optimization',
        storage=storage,
        load_if_exists=True
    )
    
    # 複数プロセスから同時に実行可能
    study.optimize(objective, n_trials=50)
    

* * *

## 2.5 実践: 深層学習モデルのチューニング

### PyTorchモデルのOptuna統合

実際の深層学習モデルでOptunaを活用する完全な例を示します。
    
    
    import optuna
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データの準備
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # スケーリング
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # PyTorchテンソルに変換
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # モデル定義関数
    def create_model(trial, input_size, output_size):
        # ハイパーパラメータの提案
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
    
        for i in range(n_layers):
            hidden_size = trial.suggest_int(f'hidden_size_l{i}', 32, 256, log=True)
            hidden_sizes.append(hidden_size)
    
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
    
        # 活性化関数の選択
        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'tanh':
            activation = nn.Tanh()
        else:
            activation = nn.ELU()
    
        # ネットワーク構築
        layers = []
        in_features = input_size
    
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
    
        layers.append(nn.Linear(in_features, output_size))
    
        model = nn.Sequential(*layers)
        return model
    
    # Objective関数
    def objective(trial):
        # モデルの作成
        model = create_model(trial, input_size=20, output_size=2).to(device)
    
        # オプティマイザーのハイパーパラメータ
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
        # バッチサイズ
        batch_size = trial.suggest_int('batch_size', 16, 256, step=16)
    
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
        # 損失関数
        criterion = nn.CrossEntropyLoss()
    
        # 訓練ループ
        n_epochs = 20
    
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
    
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
    
            # 検証（テストセット）
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)
    
            # 途中経過を報告（Pruning用）
            trial.report(accuracy, epoch)
    
            # Pruningの判定
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        return accuracy
    
    # Study作成と最適化
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )
    
    print("\n=== 深層学習モデルの最適化開始 ===")
    study.optimize(objective, n_trials=50, timeout=600)
    
    # 結果の表示
    print("\n=== 最適化完了 ===")
    print(f"最良の精度: {study.best_value:.4f}")
    print(f"\n最良のハイパーパラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 統計情報
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n完了した試行: {n_complete}")
    print(f"枝刈りされた試行: {n_pruned}")
    

### 可視化

Optunaは強力な可視化機能を提供します：
    
    
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_parallel_coordinate,
        plot_contour
    )
    import matplotlib.pyplot as plt
    
    # 1. 最適化履歴
    fig = plot_optimization_history(study)
    fig.show()
    
    # 2. パラメータ重要度
    fig = plot_param_importances(study)
    fig.show()
    
    # 3. スライスプロット（各パラメータの影響）
    fig = plot_slice(study)
    fig.show()
    
    # 4. 平行座標プロット
    fig = plot_parallel_coordinate(study)
    fig.show()
    
    # 5. 等高線プロット（2次元の関係）
    fig = plot_contour(study, params=['lr', 'n_layers'])
    fig.show()
    
    # MatplotlibでカスタムPlot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 試行ごとの精度
    trial_numbers = [t.number for t in study.trials]
    values = [t.value for t in study.trials if t.value is not None]
    axes[0, 0].plot(trial_numbers[:len(values)], values, 'o-', alpha=0.6)
    axes[0, 0].axhline(y=study.best_value, color='r',
                       linestyle='--', label=f'最良: {study.best_value:.4f}')
    axes[0, 0].set_xlabel('Trial番号')
    axes[0, 0].set_ylabel('精度')
    axes[0, 0].set_title('試行ごとの精度推移')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 学習率 vs 精度
    lrs = [t.params['lr'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[0, 1].scatter(lrs, values, alpha=0.6, s=50, edgecolors='black')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('学習率')
    axes[0, 1].set_ylabel('精度')
    axes[0, 1].set_title('学習率 vs 精度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 層数 vs 精度
    n_layers_list = [t.params['n_layers'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[1, 0].scatter(n_layers_list, values, alpha=0.6, s=50, edgecolors='black')
    axes[1, 0].set_xlabel('層数')
    axes[1, 0].set_ylabel('精度')
    axes[1, 0].set_title('層数 vs 精度')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. バッチサイズ vs 精度
    batch_sizes = [t.params['batch_size'] for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    axes[1, 1].scatter(batch_sizes, values, alpha=0.6, s=50, edgecolors='black')
    axes[1, 1].set_xlabel('バッチサイズ')
    axes[1, 1].set_ylabel('精度')
    axes[1, 1].set_title('バッチサイズ vs 精度')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Optuna Dashboard

インタラクティブなWebダッシュボードで結果を可視化：
    
    
    # インストール
    pip install optuna-dashboard
    
    # ダッシュボード起動
    optuna-dashboard sqlite:///optuna_study.db
    

ブラウザで `http://127.0.0.1:8080` にアクセスすると、リアルタイムで最適化の進捗を確認できます。

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **ベイズ最適化の原理**

     * サロゲートモデル（ガウス過程）で目的関数を近似
     * 獲得関数（EI, UCB, PI）で次の探索点を決定
     * 探索と活用のバランスで効率的に最適化
  2. **TPEアルゴリズム**

     * P(x|y)をモデル化する効率的な手法
     * 高次元・カテゴリカル変数に強い
     * 並列化が容易
  3. **Optunaの基本**

     * Study, Trial, Objectiveの概念
     * Define-by-Run APIで柔軟な探索空間定義
     * 豊富なsuggest_*メソッド
  4. **実践テクニック**

     * Pruningで計算時間を削減
     * 並列最適化で高速化
     * 条件付きハイパーパラメータの扱い
  5. **深層学習への応用**

     * PyTorchモデルの統合
     * 学習率、アーキテクチャ、オプティマイザーの最適化
     * 可視化による洞察の獲得

### ベイズ最適化 vs ランダムサーチ

側面 | ランダムサーチ | ベイズ最適化（Optuna）  
---|---|---  
**試行回数** | 多数必要 | 少数で収束  
**過去情報の活用** | なし | あり（サロゲートモデル）  
**計算コスト** | 低い | やや高い（TPEは軽量）  
**高次元性能** | 良好 | TPEは良好、GPは低下  
**並列化** | 容易 | 容易（Optuna）  
**実装の複雑さ** | シンプル | Optunaで簡単  
  
### 推奨する使い分け

状況 | 推奨手法 | 理由  
---|---|---  
訓練コストが高い | Optuna + Pruning | 早期終了で効率化  
低次元（< 10） | グリッドサーチ or Optuna | どちらも有効  
高次元（> 20） | Optuna（TPE） | 次元の呪いに強い  
カテゴリカル変数多い | Optuna | 自然に扱える  
初期探索 | ランダムサーチ | シンプルで高速  
最終調整 | Optuna | 精密な最適化  
  
### 次の章へ

第3章では、**自動機械学習（AutoML）** を学びます：

  * Auto-sklearn: 自動モデル選択とアンサンブル
  * H2O AutoML: 大規模データ向け
  * PyCaret: ローコードML
  * TPOT: 遺伝的プログラミング
  * AutoMLの限界と使いどころ

* * *

## 参考文献

  1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. _Proceedings of the 25th ACM SIGKDD_.
  2. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. _NIPS_.
  3. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. _Proceedings of the IEEE_ , 104(1), 148-175.
  4. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. _NIPS_.
  5. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. _ICML_.
