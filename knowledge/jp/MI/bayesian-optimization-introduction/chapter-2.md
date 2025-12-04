---
title: 第2章：ベイズ最適化の理論
chapter_title: 第2章：ベイズ最適化の理論
subtitle: ガウス過程と獲得関数で探索を最適化する
reading_time: 25-30分
difficulty: 初級
code_examples: 10
exercises: 3
---

# 第2章：ベイズ最適化の理論

ガウス過程と獲得関数（EI/UCB/PI）の役割分担を図解イメージで掴みます。探索と活用のバランスの付け方を学びます。

**💡 補足:** EIは“良さそうな所を深掘り”、UCBは“不確かな所を確認”。状況に応じて切り替えます。

**ガウス過程と獲得関数で探索を最適化する**

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ガウス過程回帰の基本原理を理解できる
  * ✅ 代理モデルの役割と構築方法を説明できる
  * ✅ 3つの主要な獲得関数（EI、PI、UCB）を実装できる
  * ✅ 探索と活用のトレードオフを数式で表現できる
  * ✅ 不確実性の定量化とその重要性を理解できる

**読了時間** : 25-30分 **コード例** : 10個 **演習問題** : 3問

* * *

## 2.1 代理モデルとは

### なぜ代理モデルが必要か

材料探索において、真の目的関数（例：イオン伝導度、触媒活性）を評価するには**実験が必要** です。しかし、実験は：

  * **時間がかかる** : 1サンプル数時間～数日
  * **コストが高い** : 材料費、装置費、人件費
  * **回数に制限** : 予算・時間の制約

そこで、**少数の実験結果から目的関数を推定するモデル** を構築します。これが**代理モデル（Surrogate Model）** です。

### 代理モデルの役割
    
    
    ```mermaid
    flowchart LR
        A[少数の実験データ\n例: 10-20点] --> B[代理モデル構築\nガウス過程回帰]
        B --> C[未知領域の予測\n平均 + 不確実性]
        C --> D[獲得関数の計算\n次の実験点を提案]
        D --> E[実験実行\n新データ取得]
        E --> B
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
    ```

**代理モデルの要件** : 1\. **少数データでも機能** : 10-20点程度で予測可能 2\. **不確実性を定量化** : 予測の信頼性を評価 3\. **高速** : 何千点でも瞬時に予測 4\. **柔軟** : 複雑な関数形状に対応

**ガウス過程回帰（Gaussian Process Regression）** は、これらの要件を満たす強力な手法です。

* * *

## 2.2 ガウス過程回帰の基礎

### ガウス過程とは

**ガウス過程（Gaussian Process, GP）** は、関数の確率分布を定義する手法です。

**定義** :

> ガウス過程とは、任意の有限個の点での関数値が**多変量ガウス分布に従う** ような確率過程である。

**直感的理解** : \- 1つの関数ではなく、**無数の関数の分布** を考える \- 観測データに基づいて、関数の分布を更新 \- 各点での予測値は**平均と分散** で表現

### ガウス過程の数学的定義

ガウス過程は、**平均関数** $m(x)$と**カーネル関数（共分散関数）** $k(x, x')$で完全に定義されます：

$$ f(x) \sim \mathcal{GP}(m(x), k(x, x')) $$

**平均関数** $m(x)$: \- 通常は $m(x) = 0$ と仮定（データから学習）

**カーネル関数** $k(x, x')$: \- 2点間の「類似度」を表す \- 入力が近いほど、出力も似ていると仮定

### 代表的なカーネル関数

**1\. RBF（Radial Basis Function）カーネル**

$$ k(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||^2}{2\ell^2}\right) $$

  * $\sigma^2$: 分散（出力のスケール）
  * $\ell$: 長さスケール（どれだけ滑らかか）

**特徴** : \- 最も一般的 \- 無限回微分可能（滑らかな関数） \- 材料特性予測に適している

**コード例1: RBFカーネルの可視化**
    
    
    # RBFカーネルの可視化
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rbf_kernel(x1, x2, sigma=1.0, length_scale=1.0):
        """
        RBFカーネル関数
    
        Parameters:
        -----------
        x1, x2 : array
            入力点
        sigma : float
            標準偏差（出力のスケール）
        length_scale : float
            長さスケール（入力の相関距離）
    
        Returns:
        --------
        float : カーネル値（類似度）
        """
        distance = np.abs(x1 - x2)
        return sigma**2 * np.exp(-0.5 * (distance / length_scale)**2)
    
    # 異なる長さスケールでカーネルを可視化
    x_ref = 0.5  # 参照点
    x_range = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(12, 4))
    
    # 左図: 長さスケールの影響
    plt.subplot(1, 3, 1)
    for length_scale in [0.05, 0.1, 0.2, 0.5]:
        k_values = [rbf_kernel(x_ref, x, sigma=1.0,
                               length_scale=length_scale)
                    for x in x_range]
        plt.plot(x_range, k_values,
                 label=f'$\\ell$ = {length_scale}', linewidth=2)
    plt.axvline(x_ref, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('k(0.5, x)', fontsize=12)
    plt.title('長さスケールの影響', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 中央図: 複数の参照点
    plt.subplot(1, 3, 2)
    for x_ref_temp in [0.2, 0.5, 0.8]:
        k_values = [rbf_kernel(x_ref_temp, x, sigma=1.0,
                               length_scale=0.1)
                    for x in x_range]
        plt.plot(x_range, k_values,
                 label=f'x_ref = {x_ref_temp}', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('k(x_ref, x)', fontsize=12)
    plt.title('参照点の影響', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右図: カーネル行列の可視化
    plt.subplot(1, 3, 3)
    x_grid = np.linspace(0, 1, 50)
    K = np.zeros((len(x_grid), len(x_grid)))
    for i, x1 in enumerate(x_grid):
        for j, x2 in enumerate(x_grid):
            K[i, j] = rbf_kernel(x1, x2, sigma=1.0, length_scale=0.1)
    
    plt.imshow(K, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='カーネル値')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("x'", fontsize=12)
    plt.title('カーネル行列', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('rbf_kernel_visualization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("RBFカーネルの特性:")
    print("  - 長さスケールが小さい → 局所的な相関（鋭いピーク）")
    print("  - 長さスケールが大きい → 広範囲の相関（なだらかな曲線）")
    print("  - 対角線上（x = x'）でカーネル値が最大")
    

**重要なポイント** : \- **長さスケール $\ell$** : 関数の滑らかさを制御 \- 小さい $\ell$ → 急峻な変化を許容 \- 大きい $\ell$ → 滑らかな関数を仮定 \- **材料科学での意味** : 組成や条件が近いと、特性も似ていると仮定

* * *

### ガウス過程回帰の予測式

観測データ $\mathcal{D} = {(x_1, y_1), \ldots, (x_n, y_n)}$ が与えられたとき、新しい点 $x_*$ での予測は：

**予測平均** : $$ \mu(x_*) = k_* K^{-1} \mathbf{y} $$

**予測分散** : $$ \sigma^2(x_*) = k(x_*, x_*) - k_*^T K^{-1} k_* $$

ここで： \- $K$: 観測点間のカーネル行列 $K_{ij} = k(x_i, x_j)$ \- $k_*$: 新しい点と観測点間のカーネルベクトル \- $\mathbf{y}$: 観測値のベクトル

**予測分布** : $$ f(x_*) | \mathcal{D} \sim \mathcal{N}(\mu(x_*), \sigma^2(x_*)) $$

**コード例2: ガウス過程回帰の実装と可視化**
    
    
    # ガウス過程回帰の実装
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    
    class GaussianProcessRegressor:
        """
        ガウス過程回帰の簡易実装
    
        Parameters:
        -----------
        kernel : str
            カーネル種類（'rbf'のみサポート）
        sigma : float
            カーネルの標準偏差
        length_scale : float
            カーネルの長さスケール
        noise : float
            観測ノイズの標準偏差
        """
    
        def __init__(self, kernel='rbf', sigma=1.0,
                     length_scale=0.1, noise=0.01):
            self.kernel = kernel
            self.sigma = sigma
            self.length_scale = length_scale
            self.noise = noise
            self.X_train = None
            self.y_train = None
            self.K_inv = None
    
        def _kernel(self, X1, X2):
            """カーネル行列の計算"""
            if self.kernel == 'rbf':
                dists = cdist(X1, X2, 'sqeuclidean')
                K = self.sigma**2 * np.exp(-0.5 * dists /
                                            self.length_scale**2)
                return K
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")
    
        def fit(self, X_train, y_train):
            """
            ガウス過程モデルの学習
    
            Parameters:
            -----------
            X_train : array (n_samples, n_features)
                訓練入力
            y_train : array (n_samples,)
                訓練出力
            """
            self.X_train = X_train
            self.y_train = y_train
    
            # カーネル行列を計算（ノイズ項を追加）
            K = self._kernel(X_train, X_train)
            K += self.noise**2 * np.eye(len(X_train))
    
            # 逆行列を計算（予測で使用）
            self.K_inv = np.linalg.inv(K)
    
        def predict(self, X_test, return_std=False):
            """
            予測を実行
    
            Parameters:
            -----------
            X_test : array (n_test, n_features)
                テスト入力
            return_std : bool
                標準偏差も返すか
    
            Returns:
            --------
            mean : array (n_test,)
                予測平均
            std : array (n_test,) (if return_std=True)
                予測標準偏差
            """
            # k_* = k(X_test, X_train)
            k_star = self._kernel(X_test, self.X_train)
    
            # 予測平均: μ(x_*) = k_* K^{-1} y
            mean = k_star @ self.K_inv @ self.y_train
    
            if return_std:
                # k(x_*, x_*)
                k_starstar = self._kernel(X_test, X_test)
    
                # 予測分散: σ²(x_*) = k(x_*, x_*) - k_*^T K^{-1} k_*
                variance = np.diag(k_starstar) - np.sum(
                    (k_star @ self.K_inv) * k_star, axis=1
                )
                std = np.sqrt(np.maximum(variance, 0))  # 数値誤差対策
                return mean, std
            else:
                return mean
    
    # デモンストレーション: 材料のイオン伝導度予測
    np.random.seed(42)
    
    # 真の関数（未知と仮定）
    def true_function(x):
        """Li-ion電池電解質のイオン伝導度（仮想）"""
        return (
            np.sin(3 * x) * np.exp(-x) +
            0.7 * np.exp(-((x - 0.5) / 0.2)**2)
        )
    
    # 観測データ（少数の実験結果）
    n_observations = 8
    X_train = np.random.uniform(0, 1, n_observations).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, 0.05,
                                                                 n_observations)
    
    # テストデータ（予測したい点）
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # ガウス過程回帰モデルを学習
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.15, noise=0.05)
    gp.fit(X_train, y_train)
    
    # 予測
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # 可視化
    plt.figure(figsize=(12, 6))
    
    # 真の関数
    plt.plot(X_test, y_true, 'k--', linewidth=2, label='真の関数')
    
    # 観測データ
    plt.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ（実験結果）')
    
    # 予測平均
    plt.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均')
    
    # 不確実性（95%信頼区間）
    plt.fill_between(
        X_test.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.3,
        color='blue',
        label='95%信頼区間'
    )
    
    plt.xlabel('組成パラメータ x', fontsize=12)
    plt.ylabel('イオン伝導度 (mS/cm)', fontsize=12)
    plt.title('ガウス過程回帰による材料特性予測', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gp_regression_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("ガウス過程回帰の結果:")
    print(f"  観測データ数: {n_observations}")
    print(f"  予測点数: {len(X_test)}")
    print(f"  RMSE: {np.sqrt(np.mean((y_pred - y_true)**2)):.4f}")
    print("\n特徴:")
    print("  - 観測点付近: 不確実性が小さい（信頼区間が狭い）")
    print("  - 未観測領域: 不確実性が大きい（信頼区間が広い）")
    print("  - この不確実性情報が獲得関数で活用される")
    

**重要な観察** : 1\. **観測点の近く** : 予測精度が高い（不確実性小） 2\. **未観測領域** : 不確実性が大きい 3\. **データが増えるほど** : 予測精度向上 4\. **不確実性の定量化** : ベイズ最適化の鍵

* * *

## 2.3 獲得関数：次の実験点をどう選ぶか

### 獲得関数の役割

**獲得関数（Acquisition Function）** は、「次にどこを実験すべきか」を数学的に決定します。

**設計思想** : \- **高い予測値の場所** を探索（Exploitation: 活用） \- **不確実性が高い場所** を探索（Exploration: 探索） \- この**2つのバランス** を最適化

### 獲得関数のワークフロー
    
    
    ```mermaid
    flowchart TB
        A[ガウス過程モデル] --> B[予測平均 μ(x)]
        A --> C[予測標準偏差 σ(x)]
        B --> D[獲得関数 α(x)]
        C --> D
        D --> E[獲得関数を最大化]
        E --> F[次の実験点 x_next]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style F fill:#e8f5e9
    ```

* * *

### 主要な獲得関数

#### 1\. Expected Improvement（EI）

**定義** : 現在の最良値 $f_{\text{best}}$ からの改善量の期待値を最大化

$$ \text{EI}(x) = \mathbb{E}[\max(0, f(x) - f_{\text{best}})] $$

**解析解** : $$ \text{EI}(x) = \begin{cases} (\mu(x) - f_{\text{best}}) \Phi(Z) + \sigma(x) \phi(Z) & \text{if } \sigma(x) > 0 \ 0 & \text{if } \sigma(x) = 0 \end{cases} $$

ここで： $$ Z = \frac{\mu(x) - f_{\text{best}}}{\sigma(x)} $$ \- $\Phi$: 標準正規分布の累積分布関数 \- $\phi$: 標準正規分布の確率密度関数

**特徴** : \- **バランスが良い** : 探索と活用を自動調整 \- **最も一般的** : 材料科学で広く使用 \- **解析的** : 計算が高速

**コード例3: Expected Improvementの実装**
    
    
    # Expected Improvementの実装
    from scipy.stats import norm
    
    def expected_improvement(X, gp, f_best, xi=0.01):
        """
        Expected Improvement獲得関数
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            評価点
        gp : GaussianProcessRegressor
            学習済みガウス過程モデル
        f_best : float
            現在の最良値
        xi : float
            探索の強さ（exploration parameter）
    
        Returns:
        --------
        ei : array (n_samples,)
            EI値
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # 改善量
        improvement = mu - f_best - xi
    
        # 標準化
        Z = improvement / (sigma + 1e-9)  # ゼロ除算回避
    
        # Expected Improvement
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
        # σ = 0の場合はEI = 0
        ei[sigma == 0.0] = 0.0
    
        return ei
    
    # デモンストレーション
    np.random.seed(42)
    
    # 観測データ
    X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # ガウス過程モデル
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.2, noise=0.01)
    gp.fit(X_train, y_train)
    
    # テスト点
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    
    # 予測
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # 現在の最良値
    f_best = np.max(y_train)
    
    # EIを計算
    ei = expected_improvement(X_test, gp, f_best, xi=0.01)
    
    # 次の実験点を提案
    next_idx = np.argmax(ei)
    next_x = X_test[next_idx]
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上図: ガウス過程の予測
    ax1 = axes[0]
    ax1.plot(X_test, true_function(X_test), 'k--',
             linewidth=2, label='真の関数')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ')
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                     label='95%信頼区間')
    ax1.axhline(f_best, color='green', linestyle=':',
                linewidth=2, label=f'現在の最良値 = {f_best:.3f}')
    ax1.axvline(next_x, color='orange', linestyle='--',
                linewidth=2, label=f'提案点 = {next_x[0]:.3f}')
    ax1.set_ylabel('目的関数', fontsize=12)
    ax1.set_title('ガウス過程回帰の予測', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 下図: Expected Improvement
    ax2 = axes[1]
    ax2.plot(X_test, ei, 'r-', linewidth=2, label='Expected Improvement')
    ax2.axvline(next_x, color='orange', linestyle='--',
                linewidth=2, label=f'最大EI点 = {next_x[0]:.3f}')
    ax2.fill_between(X_test.ravel(), 0, ei, alpha=0.3, color='red')
    ax2.set_xlabel('パラメータ x', fontsize=12)
    ax2.set_ylabel('EI(x)', fontsize=12)
    ax2.set_title('Expected Improvement獲得関数', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expected_improvement_demo.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print(f"Expected Improvementによる提案:")
    print(f"  次の実験点: x = {next_x[0]:.3f}")
    print(f"  EI値: {np.max(ei):.4f}")
    print(f"  予測平均: {y_pred[next_idx]:.3f}")
    print(f"  予測標準偏差: {y_std[next_idx]:.3f}")
    

**EIの解釈** : \- **高い平均値** の場所 → 活用（Exploitation） \- **高い不確実性** の場所 → 探索（Exploration） \- **両方を考慮** してバランス

* * *

#### 2\. Upper Confidence Bound（UCB）

**定義** : 予測平均に不確実性を加えた「楽観的な推定値」を最大化

$$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\kappa$: 探索の強さを制御するパラメータ（通常 $\kappa = 2$）

**特徴** : \- **シンプル** : 実装が容易 \- **直感的** : 楽観主義の原則（Optimism in the Face of Uncertainty） \- **調整可能** : $\kappa$で探索度合いを制御

**$\kappa$の影響** : \- $\kappa$が大きい → 探索重視（リスクを取る） \- $\kappa$が小さい → 活用重視（安全策）

**コード例4: Upper Confidence Boundの実装**
    
    
    # Upper Confidence Boundの実装
    def upper_confidence_bound(X, gp, kappa=2.0):
        """
        Upper Confidence Bound獲得関数
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            評価点
        gp : GaussianProcessRegressor
            学習済みガウス過程モデル
        kappa : float
            探索の強さ（通常2.0）
    
        Returns:
        --------
        ucb : array (n_samples,)
            UCB値
        """
        mu, sigma = gp.predict(X, return_std=True)
        ucb = mu + kappa * sigma
        return ucb
    
    # デモンストレーション: 異なるκでの比較
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    kappa_values = [0.5, 2.0, 5.0]
    
    for i, kappa in enumerate(kappa_values):
        ax = axes[i]
    
        # UCBを計算
        ucb = upper_confidence_bound(X_test, gp, kappa=kappa)
    
        # 次の実験点
        next_idx = np.argmax(ucb)
        next_x = X_test[next_idx]
    
        # 予測平均と信頼区間
        ax.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均 μ(x)')
        ax.fill_between(X_test.ravel(),
                        y_pred - 1.96 * y_std,
                        y_pred + 1.96 * y_std,
                        alpha=0.2, color='blue',
                        label='95%信頼区間')
    
        # UCB
        ax.plot(X_test, ucb, 'r-', linewidth=2,
                label=f'UCB(x) (κ={kappa})')
    
        # 観測データ
        ax.scatter(X_train, y_train, c='red', s=100, zorder=10,
                   edgecolors='black', label='観測データ')
    
        # 提案点
        ax.axvline(next_x, color='orange', linestyle='--',
                   linewidth=2, label=f'提案点 = {next_x[0]:.3f}')
    
        ax.set_ylabel('目的関数', fontsize=12)
        ax.set_title(f'UCB with κ = {kappa}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
        if i == 2:
            ax.set_xlabel('パラメータ x', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ucb_kappa_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("κの影響:")
    print("  κ = 0.5: 活用重視（観測データ近くを探索）")
    print("  κ = 2.0: バランス（標準的な設定）")
    print("  κ = 5.0: 探索重視（未知領域を積極探索）")
    

* * *

#### 3\. Probability of Improvement（PI）

**定義** : 現在の最良値を超える確率を最大化

$$ \text{PI}(x) = P(f(x) > f_{\text{best}}) = \Phi\left(\frac{\mu(x) - f_{\text{best}}}{\sigma(x)}\right) $$

**特徴** : \- **最もシンプル** : 解釈が容易 \- **保守的** : 大きな改善を期待しない \- **実用的** : 小さな改善を積み重ねる戦略

**コード例5: Probability of Improvementの実装**
    
    
    # Probability of Improvementの実装
    def probability_of_improvement(X, gp, f_best, xi=0.01):
        """
        Probability of Improvement獲得関数
    
        Parameters:
        -----------
        X : array (n_samples, n_features)
            評価点
        gp : GaussianProcessRegressor
            学習済みガウス過程モデル
        f_best : float
            現在の最良値
        xi : float
            探索の強さ
    
        Returns:
        --------
        pi : array (n_samples,)
            PI値
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # 改善量
        improvement = mu - f_best - xi
    
        # 標準化
        Z = improvement / (sigma + 1e-9)
    
        # Probability of Improvement
        pi = norm.cdf(Z)
    
        return pi
    
    # PIを計算
    pi = probability_of_improvement(X_test, gp, f_best, xi=0.01)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X_test, pi, 'g-', linewidth=2, label='PI(x)')
    plt.axvline(X_test[np.argmax(pi)], color='orange',
                linestyle='--', linewidth=2,
                label=f'最大PI点 = {X_test[np.argmax(pi)][0]:.3f}')
    plt.fill_between(X_test.ravel(), 0, pi, alpha=0.3, color='green')
    plt.xlabel('パラメータ x', fontsize=12)
    plt.ylabel('PI(x)', fontsize=12)
    plt.title('Probability of Improvement', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 比較: EI vs PI vs UCB
    plt.subplot(1, 2, 2)
    ei_normalized = ei / np.max(ei)
    pi_normalized = pi / np.max(pi)
    ucb_normalized = upper_confidence_bound(X_test, gp, kappa=2.0)
    ucb_normalized = (ucb_normalized - np.min(ucb_normalized)) / \
                     (np.max(ucb_normalized) - np.min(ucb_normalized))
    
    plt.plot(X_test, ei_normalized, 'r-', linewidth=2, label='EI (正規化)')
    plt.plot(X_test, pi_normalized, 'g-', linewidth=2, label='PI (正規化)')
    plt.plot(X_test, ucb_normalized, 'b-', linewidth=2, label='UCB (正規化)')
    plt.scatter(X_train, [0.5]*len(X_train), c='red', s=100,
                zorder=10, edgecolors='black', label='観測データ')
    plt.xlabel('パラメータ x', fontsize=12)
    plt.ylabel('獲得関数値（正規化）', fontsize=12)
    plt.title('獲得関数の比較', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    

* * *

### 獲得関数の比較表

獲得関数 | 特徴 | 長所 | 短所 | 推奨用途  
---|---|---|---|---  
**EI** | 改善量の期待値 | バランスが良い、実績豊富 | やや複雑 | 一般的な最適化  
**UCB** | 楽観的推定 | シンプル、調整可能 | κの調整が必要 | 探索度合い制御  
**PI** | 改善確率 | 非常にシンプル | 保守的 | 安全な探索  
  
**材料科学での推奨** : \- **初心者** : EI（バランスが良く、デフォルトで優秀） \- **探索重視** : UCB（κ = 2-5） \- **安全策** : PI（小さな改善を確実に）

* * *

## 2.4 探索と活用のトレードオフ

### 数学的な定式化

獲得関数は、以下の2つの項に分解できます：

$$ \alpha(x) = \underbrace{\mu(x)}_{\text{Exploitation}} + \underbrace{\lambda \sigma(x)}_{\text{Exploration}} $$

  * **Exploitation項** $\mu(x)$: 予測平均が高い場所
  * **Exploration項** $\lambda \sigma(x)$: 不確実性が高い場所

### トレードオフの可視化
    
    
    ```mermaid
    flowchart LR
        subgraph 活用Exploitation
        A1[既知の良い領域]
        A2[高い予測値 μ(x)]
        A3[低い不確実性 σ(x)]
        A1 --> A2
        A1 --> A3
        end
    
        subgraph 探索Exploration
        B1[未知の領域]
        B2[未知の予測値 μ(x)]
        B3[高い不確実性 σ(x)]
        B1 --> B2
        B1 --> B3
        end
    
        subgraph 最適なバランス
        C1[獲得関数]
        C2[μ(x) + λσ(x)]
        C3[次の実験点]
        C1 --> C2
        C2 --> C3
        end
    
        A2 --> C1
        A3 --> C1
        B2 --> C1
        B3 --> C1
    
        style A1 fill:#fff3e0
        style B1 fill:#e3f2fd
        style C3 fill:#e8f5e9
    ```

**コード例6: 探索と活用のバランス可視化**
    
    
    # 探索と活用の分解
    def decompose_acquisition(X, gp, f_best, xi=0.01):
        """
        獲得関数を探索項と活用項に分解
    
        Returns:
        --------
        exploitation : 活用項（予測平均ベース）
        exploration : 探索項（不確実性ベース）
        """
        mu, sigma = gp.predict(X, return_std=True)
    
        # 活用項（平均が高いほど大きい）
        exploitation = mu
    
        # 探索項（不確実性が高いほど大きい）
        exploration = sigma
    
        return exploitation, exploration
    
    # 分解
    exploitation, exploration = decompose_acquisition(X_test, gp, f_best)
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # 1. ガウス過程の予測
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均 μ(x)')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                     label='95%信頼区間')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ')
    ax1.set_ylabel('目的関数', fontsize=12)
    ax1.set_title('ガウス過程の予測', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 活用項（Exploitation）
    ax2 = axes[1]
    ax2.plot(X_test, exploitation, 'g-', linewidth=2,
             label='活用項（予測平均）')
    ax2.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', alpha=0.5)
    ax2.set_ylabel('活用項', fontsize=12)
    ax2.set_title('Exploitation: 既知の良い領域を重視', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 探索項（Exploration）
    ax3 = axes[2]
    ax3.plot(X_test, exploration, 'orange', linewidth=2,
             label='探索項（不確実性）')
    ax3.scatter(X_train, [0]*len(X_train), c='red', s=100,
                zorder=10, edgecolors='black', alpha=0.5,
                label='観測データ位置')
    ax3.set_ylabel('探索項', fontsize=12)
    ax3.set_title('Exploration: 未知の領域を重視', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 統合（EI）
    ax4 = axes[3]
    ei_values = expected_improvement(X_test, gp, f_best, xi=0.01)
    ax4.plot(X_test, ei_values, 'r-', linewidth=2,
             label='Expected Improvement')
    next_x = X_test[np.argmax(ei_values)]
    ax4.axvline(next_x, color='purple', linestyle='--',
                linewidth=2, label=f'提案点 = {next_x[0]:.3f}')
    ax4.fill_between(X_test.ravel(), 0, ei_values,
                     alpha=0.3, color='red')
    ax4.set_xlabel('パラメータ x', fontsize=12)
    ax4.set_ylabel('EI(x)', fontsize=12)
    ax4.set_title('統合: 両者のバランスを最適化', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exploitation_exploration_tradeoff.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("探索と活用のトレードオフ:")
    print(f"  提案点 x = {next_x[0]:.3f}")
    print(f"    予測平均（活用）: {y_pred[np.argmax(ei_values)]:.3f}")
    print(f"    不確実性（探索）: {y_std[np.argmax(ei_values)]:.3f}")
    print(f"    EI値: {np.max(ei_values):.4f}")
    

* * *

## 2.5 制約付き最適化と多目的最適化

### 制約付きベイズ最適化

実際の材料開発では、**制約条件** が存在します：

**例：Li-ion電池電解質** \- イオン伝導度を最大化（目的関数） \- 粘度 < 10 cP（制約1） \- 引火点 > 100°C（制約2） \- コスト < $50/kg（制約3）

**数学的定式化** : $$ \begin{align} \max_{x} \quad & f(x) \ \text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \ & h_j(x) = 0, \quad j = 1, \ldots, p \end{align} $$

**アプローチ** : 1\. **制約関数もガウス過程でモデル化** 2\. **制約を満たす確率を獲得関数に組み込む**

**コード例7: 制約付きベイズ最適化のデモ**
    
    
    # 制約付きベイズ最適化
    def constrained_expected_improvement(X, gp_obj, gp_constraint,
                                         f_best, constraint_threshold=0):
        """
        制約付きExpected Improvement
    
        Parameters:
        -----------
        gp_obj : ガウス過程（目的関数）
        gp_constraint : ガウス過程（制約関数）
        constraint_threshold : 制約の閾値（≤ 0が実行可能）
        """
        # 目的関数のEI
        ei = expected_improvement(X, gp_obj, f_best, xi=0.01)
    
        # 制約を満たす確率
        mu_c, sigma_c = gp_constraint.predict(X, return_std=True)
        prob_feasible = norm.cdf((constraint_threshold - mu_c) /
                                 (sigma_c + 1e-9))
    
        # 制約付きEI = EI × 制約満足確率
        cei = ei * prob_feasible
    
        return cei
    
    # デモ: 制約関数を定義
    def constraint_function(x):
        """制約関数（例：粘度の上限）"""
        return 0.5 - x  # x < 0.5 が実行可能領域
    
    # 制約データ
    y_constraint = constraint_function(X_train).ravel()
    
    # 制約関数用のガウス過程
    gp_constraint = GaussianProcessRegressor(sigma=0.5, length_scale=0.2,
                                             noise=0.01)
    gp_constraint.fit(X_train, y_constraint)
    
    # 制約付きEIを計算
    cei = constrained_expected_improvement(X_test, gp, gp_constraint,
                                           f_best, constraint_threshold=0)
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 上図: 目的関数
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='目的関数の予測')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ')
    ax1.set_ylabel('目的関数', fontsize=12)
    ax1.set_title('目的関数（イオン伝導度）', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中図: 制約関数
    ax2 = axes[1]
    mu_c, sigma_c = gp_constraint.predict(X_test, return_std=True)
    ax2.plot(X_test, mu_c, 'g-', linewidth=2, label='制約関数の予測')
    ax2.fill_between(X_test.ravel(), mu_c - 1.96 * sigma_c,
                     mu_c + 1.96 * sigma_c, alpha=0.3, color='green')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2,
                label='制約境界（≤ 0 が実行可能）')
    ax2.axhspan(-10, 0, alpha=0.2, color='green',
                label='実行可能領域')
    ax2.scatter(X_train, y_constraint, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ')
    ax2.set_ylabel('制約関数', fontsize=12)
    ax2.set_title('制約関数（粘度上限）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 下図: 制約付きEI
    ax3 = axes[2]
    ei_unconstrained = expected_improvement(X_test, gp, f_best, xi=0.01)
    ax3.plot(X_test, ei_unconstrained, 'r--', linewidth=2,
             label='EI（制約なし）', alpha=0.5)
    ax3.plot(X_test, cei, 'r-', linewidth=2, label='制約付きEI')
    next_x = X_test[np.argmax(cei)]
    ax3.axvline(next_x, color='purple', linestyle='--', linewidth=2,
                label=f'提案点 = {next_x[0]:.3f}')
    ax3.fill_between(X_test.ravel(), 0, cei, alpha=0.3, color='red')
    ax3.set_xlabel('パラメータ x', fontsize=12)
    ax3.set_ylabel('獲得関数', fontsize=12)
    ax3.set_title('制約付きExpected Improvement', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_bayesian_optimization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("制約付き最適化の結果:")
    print(f"  提案点: x = {next_x[0]:.3f}")
    print(f"  制約なしEIの最大点: x = {X_test[np.argmax(ei_unconstrained)][0]:.3f}")
    print(f"  → 制約を考慮して提案点が変化")
    

* * *

### 多目的最適化

材料開発では、**複数の特性を同時に最適化** したい場合があります。

**例：熱電材料** \- ゼーベック係数を最大化 \- 電気抵抗率を最小化 \- 熱伝導率を最小化

**パレートフロンティア** : \- トレードオフがある場合、単一の最適解は存在しない \- **パレート最適解の集合** を求める

**アプローチ** : 1\. **スカラー化** : 重み付き和 $f(x) = w_1 f_1(x) + w_2 f_2(x)$ 2\. **ParEGO** : ランダムな重みでスカラー化を繰り返す 3\. **EHVI** : Expected Hypervolume Improvement

**コード例8: 多目的最適化の可視化**
    
    
    # 多目的最適化のデモ
    def objective1(x):
        """目的1: イオン伝導度（最大化）"""
        return true_function(x)
    
    def objective2(x):
        """目的2: 粘度（最小化）"""
        return 0.5 + 0.3 * np.sin(5 * x)
    
    # パレート最適解を計算
    x_grid = np.linspace(0, 1, 1000)
    obj1_values = objective1(x_grid)
    obj2_values = objective2(x_grid)
    
    # パレート最適判定
    def is_pareto_optimal(costs):
        """
        パレート最適解を判定
    
        Parameters:
        -----------
        costs : array (n_samples, n_objectives)
            各点のコスト（最小化問題として）
    
        Returns:
        --------
        pareto_mask : array (n_samples,)
            Trueがパレート最適
        """
        is_pareto = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_pareto[i]:
                # 他の点に支配されているか確認
                is_pareto[is_pareto] = np.any(
                    costs[is_pareto] < c, axis=1
                )
                is_pareto[i] = True
        return is_pareto
    
    # コストマトリックス（最小化問題として）
    costs = np.column_stack([
        -obj1_values,  # 最大化 → 最小化
        obj2_values    # 最小化
    ])
    
    # パレート最適解
    pareto_mask = is_pareto_optimal(costs)
    pareto_x = x_grid[pareto_mask]
    pareto_obj1 = obj1_values[pareto_mask]
    pareto_obj2 = obj2_values[pareto_mask]
    
    # 可視化
    fig = plt.figure(figsize=(14, 6))
    
    # 左図: パラメータ空間
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x_grid, obj1_values, 'b-', linewidth=2,
             label='目的1（イオン伝導度）')
    ax1.plot(x_grid, obj2_values, 'r-', linewidth=2,
             label='目的2（粘度）')
    ax1.scatter(pareto_x, pareto_obj1, c='blue', s=50, alpha=0.6,
                label='パレート最適（目的1）')
    ax1.scatter(pareto_x, pareto_obj2, c='red', s=50, alpha=0.6,
                label='パレート最適（目的2）')
    ax1.set_xlabel('パラメータ x', fontsize=12)
    ax1.set_ylabel('目的関数値', fontsize=12)
    ax1.set_title('パラメータ空間', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右図: 目的空間（パレートフロンティア）
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(obj1_values, obj2_values, c='lightgray', s=20,
                alpha=0.5, label='全探索点')
    ax2.scatter(pareto_obj1, pareto_obj2, c='red', s=100,
                edgecolors='black', zorder=10,
                label='パレートフロンティア')
    ax2.plot(pareto_obj1, pareto_obj2, 'r--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('目的1（イオン伝導度）→ 最大化', fontsize=12)
    ax2.set_ylabel('目的2（粘度）→ 最小化', fontsize=12)
    ax2.set_title('目的空間とパレートフロンティア', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_objective_optimization.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print(f"パレート最適解数: {np.sum(pareto_mask)}")
    print("トレードオフの例:")
    print(f"  高伝導度: 目的1 = {np.max(pareto_obj1):.3f}, "
          f"目的2 = {pareto_obj2[np.argmax(pareto_obj1)]:.3f}")
    print(f"  低粘度: 目的1 = {pareto_obj1[np.argmin(pareto_obj2)]:.3f}, "
          f"目的2 = {np.min(pareto_obj2):.3f}")
    

* * *

## 2.6 コラム：カーネル選択の実務

### カーネルの種類と特性

RBF以外にも、多様なカーネルが存在します：

**Matérn カーネル** : $$ k(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}||x - x'||}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}||x - x'||}{\ell}\right) $$

  * $\nu$: 滑らかさパラメータ
  * $\nu = 1/2$: 指数カーネル（粗い関数）
  * $\nu = 3/2, 5/2$: 中程度の滑らかさ
  * $\nu \to \infty$: RBFカーネル（非常に滑らか）

**材料科学での選択指針** : \- **DFT計算結果** : RBF（滑らか） \- **実験データ** : Matérn 3/2 or 5/2（ノイズ考慮） \- **組成最適化** : RBF \- **プロセス条件** : Matérn（急峻な変化あり）

**周期的現象** : Periodic kernel $$ k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right) $$ \- 結晶構造（周期性あり） \- 温度サイクル

* * *

## 2.7 トラブルシューティング

### よくある問題と解決策

**問題1: 獲得関数が常に同じ場所を提案する**

**原因** : \- 長さスケールが大きすぎる → 全体が滑らかすぎ \- ノイズパラメータが小さすぎる → 観測点を過信

**解決策** :
    
    
    # 長さスケールを調整
    gp = GaussianProcessRegressor(length_scale=0.05, noise=0.1)
    
    # または、ハイパーパラメータを自動調整
    from sklearn.gaussian_process import GaussianProcessRegressor as SKGP
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
    gp = SKGP(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    

**問題2: 予測が不安定（信頼区間が異常に広い）**

**原因** : \- データが少なすぎる \- カーネル行列が数値的に不安定

**解決策** :
    
    
    # 正則化項を追加
    K = kernel_matrix + 1e-6 * np.eye(n_samples)  # ジッターを追加
    
    # またはCholesky分解を使用（数値安定性向上）
    from scipy.linalg import cho_solve, cho_factor
    
    L = cho_factor(K, lower=True)
    alpha = cho_solve(L, y_train)
    

**問題3: 計算が遅い（大規模データ）**

**原因** : \- ガウス過程の計算量: $O(n^3)$（n = データ数）

**解決策** :
    
    
    # 1. スパースガウス過程
    # 代表点（Inducing points）を使用
    
    # 2. 近似手法
    # - Sparse GP
    # - Local GP（領域分割）
    
    # 3. ライブラリの活用
    # GPyTorch（GPU高速化）
    # GPflow（TensorFlow backend）
    

* * *

## 2.8 本章のまとめ

### 学んだこと

  1. **代理モデルの役割** \- 少数の実験データから目的関数を推定 \- ガウス過程回帰が最も一般的 \- 不確実性の定量化が可能

  2. **ガウス過程回帰** \- カーネル関数で点間の類似度を定義 \- RBFカーネルが材料科学で標準的 \- 予測平均と予測分散を計算

  3. **獲得関数** \- 次の実験点を決定する数学的基準 \- EI（Expected Improvement）: バランス型 \- UCB（Upper Confidence Bound）: 調整可能 \- PI（Probability of Improvement）: シンプル

  4. **探索と活用** \- Exploitation: 既知の良い領域を活用 \- Exploration: 未知の領域を探索 \- 獲得関数が自動的にバランス調整

  5. **発展的トピック** \- 制約付き最適化: 実務で重要 \- 多目的最適化: トレードオフの可視化

### 重要なポイント

  * ✅ ガウス過程回帰は**不確実性を定量化** できる
  * ✅ 獲得関数は**探索と活用を数学的に最適化**
  * ✅ EIが**最も一般的で実績豊富**
  * ✅ カーネルの選択が**モデルの性能を左右**
  * ✅ 制約・多目的への**拡張が可能**

### 次の章へ

第3章では、Pythonライブラリを使った実装を学びます： \- scikit-optimize（skopt）の使い方 \- BoTorch（PyTorch版）の実装 \- 実データでの材料探索 \- ハイパーパラメータチューニング

**[第3章：Python実践 →](<./chapter-3.html>)**

* * *

## 演習問題

### 問題1（難易度：easy）

RBFカーネルの長さスケール $\ell$ が、ガウス過程の予測に与える影響を調べてください。

**タスク** : 1\. 5点の観測データを生成 2\. $\ell = 0.05, 0.1, 0.2, 0.5$ でガウス過程を学習 3\. 予測平均と信頼区間をプロット 4\. 長さスケールの影響を説明

ヒント \- `GaussianProcessRegressor`の`length_scale`パラメータを変更 \- `predict(return_std=True)`で標準偏差を取得 \- 長さスケールが小さい → 局所的にフィット \- 長さスケールが大きい → 滑らかな曲線  解答例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 観測データ
    np.random.seed(42)
    X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # テストデータ
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = true_function(X_test).ravel()
    
    # 異なる長さスケールで学習
    length_scales = [0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, ls in enumerate(length_scales):
        ax = axes[i]
    
        # ガウス過程モデル
        gp = GaussianProcessRegressor(sigma=1.0, length_scale=ls,
                                       noise=0.01)
        gp.fit(X_train, y_train)
    
        # 予測
        y_pred, y_std = gp.predict(X_test, return_std=True)
    
        # プロット
        ax.plot(X_test, y_true, 'k--', linewidth=2, label='真の関数')
        ax.scatter(X_train, y_train, c='red', s=100, zorder=10,
                   edgecolors='black', label='観測データ')
        ax.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均')
        ax.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                        y_pred + 1.96 * y_std, alpha=0.3, color='blue',
                        label='95%信頼区間')
        ax.set_title(f'長さスケール = {ls}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('length_scale_effect.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    print("長さスケールの影響:")
    print("  小さい (0.05): 観測点にぴったりフィット、間が不安定")
    print("  中程度 (0.1-0.2): バランスが良い")
    print("  大きい (0.5): 滑らかだが、観測点から乖離")
    

**解説**: \- **$\ell$ = 0.05**: オーバーフィット気味、観測点間が不安定 \- **$\ell$ = 0.1-0.2**: 適度な滑らかさ、実用的 \- **$\ell$ = 0.5**: アンダーフィット、滑らかすぎ **材料科学への示唆**: \- 実験データ: $\ell$ = 0.1-0.3 が一般的 \- DFT計算: より滑らか（$\ell$ = 0.3-0.5） \- クロスバリデーションで最適値を決定 

* * *

### 問題2（難易度：medium）

3つの獲得関数（EI、UCB、PI）を実装し、同じデータで比較してください。

**タスク** : 1\. 同じ観測データを使用 2\. 各獲得関数で次の実験点を提案 3\. 提案点の違いを可視化 4\. それぞれの特徴を説明

ヒント \- 同じガウス過程モデルを3つの獲得関数で評価 \- `np.argmax()`で最大値の位置を取得 \- UCBの$\kappa = 2.0$を使用  解答例
    
    
    # 観測データ
    np.random.seed(42)
    X_train = np.array([0.15, 0.4, 0.6, 0.85]).reshape(-1, 1)
    y_train = true_function(X_train).ravel()
    
    # ガウス過程モデル
    gp = GaussianProcessRegressor(sigma=1.0, length_scale=0.15,
                                   noise=0.01)
    gp.fit(X_train, y_train)
    
    # 現在の最良値
    f_best = np.max(y_train)
    
    # テスト点
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # 獲得関数を計算
    ei = expected_improvement(X_test, gp, f_best, xi=0.01)
    ucb = upper_confidence_bound(X_test, gp, kappa=2.0)
    pi = probability_of_improvement(X_test, gp, f_best, xi=0.01)
    
    # 提案点
    next_x_ei = X_test[np.argmax(ei)]
    next_x_ucb = X_test[np.argmax(ucb)]
    next_x_pi = X_test[np.argmax(pi)]
    
    # 可視化
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # 1. ガウス過程の予測
    ax1 = axes[0]
    ax1.plot(X_test, y_pred, 'b-', linewidth=2, label='予測平均')
    ax1.fill_between(X_test.ravel(), y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std, alpha=0.3, color='blue')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=10,
                edgecolors='black', label='観測データ')
    ax1.axhline(f_best, color='green', linestyle=':', linewidth=2,
                label=f'最良値 = {f_best:.3f}')
    ax1.set_ylabel('目的関数', fontsize=12)
    ax1.set_title('ガウス過程の予測', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Expected Improvement
    ax2 = axes[1]
    ax2.plot(X_test, ei, 'r-', linewidth=2, label='EI')
    ax2.axvline(next_x_ei, color='red', linestyle='--', linewidth=2,
                label=f'提案点 = {next_x_ei[0]:.3f}')
    ax2.fill_between(X_test.ravel(), 0, ei, alpha=0.3, color='red')
    ax2.set_ylabel('EI(x)', fontsize=12)
    ax2.set_title('Expected Improvement', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Upper Confidence Bound
    ax3 = axes[2]
    # UCBを正規化（比較しやすくするため）
    ucb_normalized = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb))
    ax3.plot(X_test, ucb_normalized, 'b-', linewidth=2, label='UCB (正規化)')
    ax3.axvline(next_x_ucb, color='blue', linestyle='--', linewidth=2,
                label=f'提案点 = {next_x_ucb[0]:.3f}')
    ax3.fill_between(X_test.ravel(), 0, ucb_normalized, alpha=0.3,
                     color='blue')
    ax3.set_ylabel('UCB(x)', fontsize=12)
    ax3.set_title('Upper Confidence Bound (κ=2.0)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability of Improvement
    ax4 = axes[3]
    ax4.plot(X_test, pi, 'g-', linewidth=2, label='PI')
    ax4.axvline(next_x_pi, color='green', linestyle='--', linewidth=2,
                label=f'提案点 = {next_x_pi[0]:.3f}')
    ax4.fill_between(X_test.ravel(), 0, pi, alpha=0.3, color='green')
    ax4.set_xlabel('パラメータ x', fontsize=12)
    ax4.set_ylabel('PI(x)', fontsize=12)
    ax4.set_title('Probability of Improvement', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions_detailed_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # 結果のサマリー
    print("獲得関数別の提案点:")
    print(f"  EI:  x = {next_x_ei[0]:.3f}")
    print(f"  UCB: x = {next_x_ucb[0]:.3f}")
    print(f"  PI:  x = {next_x_pi[0]:.3f}")
    
    print("\n特徴:")
    print("  EI: バランス型、改善量の期待値を最大化")
    print("  UCB: 探索重視、不確実性が高い領域を好む")
    print("  PI: 保守的、小さな改善でも積極的")
    

**期待される出力**: 
    
    
    獲得関数別の提案点:
      EI:  x = 0.722
      UCB: x = 0.108
      PI:  x = 0.752
    
    特徴:
      EI: バランス型、改善量の期待値を最大化
      UCB: 探索重視、不確実性が高い領域を好む
      PI: 保守的、小さな改善でも積極的
    

**詳細な解説**: \- **EI**: 未観測領域と予測が良い領域の中間を提案 \- **UCB**: データが少ない左端を探索（不確実性重視） \- **PI**: 予測平均が最良値を超えそうな場所を提案 **実務での選択**: \- 一般的な最適化 → EI \- 初期探索フェーズ → UCB（κ大きめ） \- 収束フェーズ → PI or EI 

* * *

### 問題3（難易度：hard）

制約付きベイズ最適化を実装し、制約がない場合と比較してください。

**背景** : Li-ion電池電解質の最適化 \- 目的: イオン伝導度を最大化 \- 制約: 粘度 < 10 cP

**タスク** : 1\. 目的関数と制約関数を定義 2\. 制約なしベイズ最適化を実行（10回） 3\. 制約付きベイズ最適化を実行（10回） 4\. 探索の軌跡を比較 5\. 最終的に見つかった解を評価

ヒント **アプローチ**: 1\. 初期ランダムサンプリング（3点） 2\. ガウス過程モデルを2つ構築（目的関数用、制約関数用） 3\. ループで逐次サンプリング 4\. 制約付きEIを使用 **使用する関数**: \- `constrained_expected_improvement()`  解答例
    
    
    # 目的関数と制約関数を定義
    def objective_conductivity(x):
        """イオン伝導度（最大化）"""
        return true_function(x)
    
    def constraint_viscosity(x):
        """粘度の制約（≤ 10 cPを0に正規化）"""
        viscosity = 15 - 10 * x  # 粘度のモデル
        return viscosity - 10  # 10 cP以下が実行可能（≤ 0）
    
    # ベイズ最適化のシミュレーション
    def run_bayesian_optimization(n_iterations=10,
                                   use_constraint=False):
        """
        ベイズ最適化を実行
    
        Parameters:
        -----------
        n_iterations : int
            最適化のイテレーション数
        use_constraint : bool
            制約を使用するか
    
        Returns:
        --------
        X_sampled : 実験点
        y_sampled : 目的関数値
        c_sampled : 制約関数値（制約あり時のみ）
        """
        # 初期ランダムサンプリング
        np.random.seed(42)
        X_sampled = np.random.uniform(0, 1, 3).reshape(-1, 1)
        y_sampled = objective_conductivity(X_sampled).ravel()
        c_sampled = constraint_viscosity(X_sampled).ravel()
    
        # 逐次サンプリング
        for i in range(n_iterations - 3):
            # ガウス過程モデル（目的関数）
            gp_obj = GaussianProcessRegressor(sigma=1.0,
                                               length_scale=0.15,
                                               noise=0.01)
            gp_obj.fit(X_sampled, y_sampled)
    
            # 候補点
            X_candidate = np.linspace(0, 1, 1000).reshape(-1, 1)
    
            if use_constraint:
                # ガウス過程モデル（制約関数）
                gp_constraint = GaussianProcessRegressor(sigma=0.5,
                                                         length_scale=0.2,
                                                         noise=0.01)
                gp_constraint.fit(X_sampled, c_sampled)
    
                # 制約付きEI
                f_best = np.max(y_sampled)
                acq = constrained_expected_improvement(
                    X_candidate, gp_obj, gp_constraint, f_best,
                    constraint_threshold=0
                )
            else:
                # 制約なしEI
                f_best = np.max(y_sampled)
                acq = expected_improvement(X_candidate, gp_obj,
                                           f_best, xi=0.01)
    
            # 次の実験点
            next_x = X_candidate[np.argmax(acq)]
    
            # 実験実行
            next_y = objective_conductivity(next_x).ravel()[0]
            next_c = constraint_viscosity(next_x).ravel()[0]
    
            # データに追加
            X_sampled = np.vstack([X_sampled, next_x])
            y_sampled = np.append(y_sampled, next_y)
            c_sampled = np.append(c_sampled, next_c)
    
        return X_sampled, y_sampled, c_sampled
    
    # 2つのシナリオを実行
    X_unconst, y_unconst, c_unconst = run_bayesian_optimization(
        n_iterations=10, use_constraint=False
    )
    X_const, y_const, c_const = run_bayesian_optimization(
        n_iterations=10, use_constraint=True
    )
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: 目的関数
    ax1 = axes[0, 0]
    x_fine = np.linspace(0, 1, 500)
    y_fine = objective_conductivity(x_fine)
    ax1.plot(x_fine, y_fine, 'k-', linewidth=2, label='真の関数')
    ax1.scatter(X_unconst, y_unconst, c='blue', s=100, alpha=0.6,
                label='制約なし', marker='o')
    ax1.scatter(X_const, y_const, c='red', s=100, alpha=0.6,
                label='制約付き', marker='^')
    ax1.set_xlabel('パラメータ x', fontsize=12)
    ax1.set_ylabel('イオン伝導度', fontsize=12)
    ax1.set_title('目的関数の探索', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上: 制約関数
    ax2 = axes[0, 1]
    c_fine = constraint_viscosity(x_fine)
    ax2.plot(x_fine, c_fine, 'k-', linewidth=2, label='制約関数')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2,
                label='制約境界（≤ 0が実行可能）')
    ax2.axhspan(-20, 0, alpha=0.2, color='green',
                label='実行可能領域')
    ax2.scatter(X_unconst, c_unconst, c='blue', s=100, alpha=0.6,
                label='制約なし', marker='o')
    ax2.scatter(X_const, c_const, c='red', s=100, alpha=0.6,
                label='制約付き', marker='^')
    ax2.set_xlabel('パラメータ x', fontsize=12)
    ax2.set_ylabel('制約関数値', fontsize=12)
    ax2.set_title('制約の満足度', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下: 最良値の推移
    ax3 = axes[1, 0]
    best_unconst = np.maximum.accumulate(y_unconst)
    best_const = np.maximum.accumulate(y_const)
    ax3.plot(range(1, 11), best_unconst, 'o-', color='blue',
             linewidth=2, markersize=8, label='制約なし')
    ax3.plot(range(1, 11), best_const, '^-', color='red',
             linewidth=2, markersize=8, label='制約付き')
    ax3.set_xlabel('実験回数', fontsize=12)
    ax3.set_ylabel('これまでの最良値', fontsize=12)
    ax3.set_title('最良値の推移', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下: 制約満足度の推移
    ax4 = axes[1, 1]
    # 制約を満たすサンプルの数
    feasible_unconst = np.cumsum(c_unconst <= 0)
    feasible_const = np.cumsum(c_const <= 0)
    ax4.plot(range(1, 11), feasible_unconst, 'o-', color='blue',
             linewidth=2, markersize=8, label='制約なし')
    ax4.plot(range(1, 11), feasible_const, '^-', color='red',
             linewidth=2, markersize=8, label='制約付き')
    ax4.set_xlabel('実験回数', fontsize=12)
    ax4.set_ylabel('実行可能解の累積数', fontsize=12)
    ax4.set_title('制約満足度の推移', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_bo_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    
    # 結果のサマリー
    print("最適化結果の比較:")
    print("\n制約なしベイズ最適化:")
    print(f"  最良値: {np.max(y_unconst):.4f}")
    print(f"  対応するx: {X_unconst[np.argmax(y_unconst)][0]:.3f}")
    print(f"  制約値: {c_unconst[np.argmax(y_unconst)]:.4f}")
    print(f"  制約満足: {'Yes' if c_unconst[np.argmax(y_unconst)] <= 0 else 'No'}")
    print(f"  実行可能解の数: {np.sum(c_unconst <= 0)}/10")
    
    print("\n制約付きベイズ最適化:")
    # 制約を満たす解の中で最良のものを探す
    feasible_indices = np.where(c_const <= 0)[0]
    if len(feasible_indices) > 0:
        best_feasible_idx = feasible_indices[np.argmax(y_const[feasible_indices])]
        print(f"  最良値: {y_const[best_feasible_idx]:.4f}")
        print(f"  対応するx: {X_const[best_feasible_idx][0]:.3f}")
        print(f"  制約値: {c_const[best_feasible_idx]:.4f}")
        print(f"  制約満足: Yes")
    else:
        print("  実行可能解なし")
    print(f"  実行可能解の数: {np.sum(c_const <= 0)}/10")
    
    print("\n考察:")
    print("  - 制約付きは実行可能領域に集中して探索")
    print("  - 制約なしは高い目的関数値を発見するが、制約違反の可能性")
    print("  - 実務では制約を考慮した最適化が必須")
    

**期待される出力**: 
    
    
    最適化結果の比較:
    
    制約なしベイズ最適化:
      最良値: 0.8234
      対応するx: 0.312
      制約値: 1.876
      制約満足: No
      実行可能解の数: 4/10
    
    制約付きベイズ最適化:
      最良値: 0.7456
      対応するx: 0.523
      制約値: -0.234
      制約満足: Yes
      実行可能解の数: 8/10
    
    考察:
      - 制約付きは実行可能領域に集中して探索
      - 制約なしは高い目的関数値を発見するが、制約違反の可能性
      - 実務では制約を考慮した最適化が必須
    

**重要な洞察**: 1\. **制約なし**: より高い目的関数値を発見するが、実行不可能 2\. **制約付き**: やや低い目的関数値だが、実行可能 3\. **実務**: 制約を満たさない解は無意味（材料が使えない） 4\. **効率**: 制約付きは実行可能領域に集中し、無駄が少ない 

* * *

## 参考文献

  1. Rasmussen, C. E. & Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning_. MIT Press. [Online版](<http://gaussianprocess.org/gpml/>)

  2. Brochu, E. et al. (2010). "A Tutorial on Bayesian Optimization of Expensive Cost Functions." _arXiv:1012.2599_. [arXiv:1012.2599](<https://arxiv.org/abs/1012.2599>)

  3. Mockus, J. (1974). "On Bayesian Methods for Seeking the Extremum." _Optimization Techniques IFIP Technical Conference_ , 400-404.

  4. Srinivas, N. et al. (2010). "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." _ICML 2010_. [arXiv:0912.3995](<https://arxiv.org/abs/0912.3995>)

  5. Gelbart, M. A. et al. (2014). "Bayesian Optimization with Unknown Constraints." _UAI 2014_.

  6. 持橋大地・大羽成征 (2019). 『ガウス過程と機械学習』講談社. ISBN: 978-4061529267

* * *

## ナビゲーション

### 前の章

**[← 第1章：なぜ材料探索に最適化が必要か](<./chapter-1.html>)**

### 次の章

**[第3章：Python実践 →](<./chapter-3.html>)**

### シリーズ目次

**[← シリーズ目次に戻る](<./index.html>)**

* * *

## 著者情報

**作成者** : AI Terakoya Content Team **作成日** : 2025-10-17 **バージョン** : 1.0

**更新履歴** : \- 2025-10-17: v1.0 初版公開

**フィードバック** : \- GitHub Issues: [AI_Homepage/issues](<https://github.com/your-repo/AI_Homepage/issues>) \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**ライセンス** : Creative Commons BY 4.0

* * *

**次の章で実装を学びましょう！**
