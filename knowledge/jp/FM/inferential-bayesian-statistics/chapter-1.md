---
title: "第1章: 推定理論の基礎"
chapter_title: "第1章: 推定理論の基礎"
subtitle: Point Estimation, Maximum Likelihood, and Estimation Theory
---

[基礎数理道場](<../index.html>) > [推測統計学とベイズ統計](<index.html>) > 第1章 

## 1.1 推定理論の概要

**推定理論（Estimation Theory）** は、標本データから母集団のパラメータを推測する統計学の基礎です。 材料科学では、少数の試験片データから材料全体の特性（平均強度、分散）を推定する場面で必須となります。 

#### 📘 推定理論の基本概念

**母集団（Population）** ：調査対象の全体集合

**標本（Sample）** ：母集団から抽出した一部のデータ

**推定量（Estimator）** ：標本から計算されるパラメータの推定値を与える関数

**推定値（Estimate）** ：推定量に標本データを代入して得られる具体的な数値

例えば、母平均 \\( \mu \\) の推定量として標本平均 \\( \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i \\) を用いるとき、 実際のデータ \\( x_1, \ldots, x_n \\) から計算される \\( \bar{x} \\) が推定値です。 

## 1.2 点推定と推定量の性質

### 1.2.1 不偏性（Unbiasedness）

#### 📘 不偏推定量の定義

推定量 \\( \hat{\theta} \\) が母数 \\( \theta \\) の**不偏推定量** であるとは：

$$ E[\hat{\theta}] = \theta $$ 

すなわち、推定量の期待値が真の母数と一致することを意味します。

#### 💻 コード例1: 標本平均と標本分散の不偏性検証
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 真の母集団パラメータ
    mu_true = 50  # 母平均
    sigma_true = 10  # 母標準偏差
    
    # シミュレーション設定
    n_samples = 30  # 標本サイズ
    n_simulations = 10000  # シミュレーション回数
    
    # 推定量の分布を記録
    sample_means = []
    sample_vars_biased = []  # バイアスあり（n で割る）
    sample_vars_unbiased = []  # 不偏推定量（n-1 で割る）
    
    np.random.seed(42)
    for _ in range(n_simulations):
        # 標本抽出
        sample = np.random.normal(mu_true, sigma_true, n_samples)
    
        # 標本平均
        sample_means.append(np.mean(sample))
    
        # 標本分散（バイアスあり）
        sample_vars_biased.append(np.var(sample, ddof=0))
    
        # 標本分散（不偏推定量）
        sample_vars_unbiased.append(np.var(sample, ddof=1))
    
    # 結果の検証
    print("=== 不偏性の検証 ===")
    print(f"母平均: {mu_true}")
    print(f"標本平均の期待値: {np.mean(sample_means):.4f}")
    print(f"標本平均の標準誤差: {np.std(sample_means):.4f}")
    print()
    print(f"母分散: {sigma_true**2}")
    print(f"バイアスあり標本分散の期待値: {np.mean(sample_vars_biased):.4f}")
    print(f"不偏標本分散の期待値: {np.mean(sample_vars_unbiased):.4f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 標本平均の分布
    axes[0].hist(sample_means, bins=50, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black')
    axes[0].axvline(mu_true, color='red', linestyle='--', linewidth=2,
                    label=f'真の母平均: {mu_true}')
    axes[0].axvline(np.mean(sample_means), color='blue', linestyle='--',
                    linewidth=2, label=f'推定量の期待値: {np.mean(sample_means):.2f}')
    axes[0].set_xlabel('標本平均')
    axes[0].set_ylabel('密度')
    axes[0].set_title('標本平均の分布（不偏性の確認）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 標本分散の分布
    axes[1].hist(sample_vars_biased, bins=50, density=True, alpha=0.5,
                 color='orange', edgecolor='black', label='バイアスあり (ddof=0)')
    axes[1].hist(sample_vars_unbiased, bins=50, density=True, alpha=0.5,
                 color='green', edgecolor='black', label='不偏推定量 (ddof=1)')
    axes[1].axvline(sigma_true**2, color='red', linestyle='--',
                    linewidth=2, label=f'真の母分散: {sigma_true**2}')
    axes[1].set_xlabel('標本分散')
    axes[1].set_ylabel('密度')
    axes[1].set_title('標本分散の分布（不偏性の比較）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 理論的な標準誤差との比較
    theoretical_se = sigma_true / np.sqrt(n_samples)
    print(f"\n理論的標準誤差: {theoretical_se:.4f}")
    print(f"実測標準誤差: {np.std(sample_means):.4f}")

**📌 重要ポイント**  
標本平均は母平均の不偏推定量ですが、標本分散は \\( n-1 \\) で割る必要があります（Besselの補正）。 これは標本分散が過小推定する傾向を補正するためです。 

### 1.2.2 一致性（Consistency）

#### 📘 一致推定量の定義

推定量 \\( \hat{\theta}_n \\) が**一致推定量** であるとは、標本サイズ \\( n \to \infty \\) のとき：

$$ \hat{\theta}_n \xrightarrow{P} \theta $$ 

すなわち、標本サイズが大きくなるにつれて真の母数に確率収束します。

### 1.2.3 有効性（Efficiency）

複数の不偏推定量が存在する場合、**分散が最小** のものが最も有効です。 Cramér-Rao下界は、不偏推定量の分散が達成できる理論的下限を与えます。 

## 1.3 最尤推定法（Maximum Likelihood Estimation）

#### 📘 最尤推定法の原理

観測データ \\( x_1, \ldots, x_n \\) が得られたとき、これらが最も生じやすいパラメータ \\( \theta \\) を推定する方法です。

**尤度関数（Likelihood Function）** ：

$$ L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta) $$ 

**対数尤度関数（Log-Likelihood）** ：

$$ \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta) $$ 

**最尤推定量（MLE）** は \\( \ell(\theta) \\) を最大化する \\( \hat{\theta}_{\text{MLE}} \\) です。

#### 💻 コード例2: 正規分布パラメータの最尤推定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy import stats
    
    # 真のパラメータ
    mu_true = 100
    sigma_true = 15
    
    # データ生成
    np.random.seed(42)
    n = 50
    data = np.random.normal(mu_true, sigma_true, n)
    
    # 対数尤度関数（正規分布）
    def neg_log_likelihood(params, data):
        """負の対数尤度（最小化のため）"""
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        n = len(data)
        return n/2 * np.log(2*np.pi*sigma**2) + np.sum((data - mu)**2) / (2*sigma**2)
    
    # 最尤推定（数値最適化）
    initial_guess = [np.mean(data), np.std(data)]
    result = minimize(neg_log_likelihood, initial_guess, args=(data,),
                      method='Nelder-Mead')
    
    mu_mle, sigma_mle = result.x
    
    print("=== 最尤推定結果 ===")
    print(f"真の母平均: {mu_true}, MLE: {mu_mle:.4f}")
    print(f"真の母標準偏差: {sigma_true}, MLE: {sigma_mle:.4f}")
    print(f"標本平均（解析解）: {np.mean(data):.4f}")
    print(f"標本標準偏差（解析解, ddof=0）: {np.std(data, ddof=0):.4f}")
    
    # 尤度関数の可視化
    mu_grid = np.linspace(90, 110, 100)
    sigma_grid = np.linspace(10, 20, 100)
    MU, SIGMA = np.meshgrid(mu_grid, sigma_grid)
    
    # 対数尤度の計算
    log_likelihood = np.zeros_like(MU)
    for i in range(len(mu_grid)):
        for j in range(len(sigma_grid)):
            log_likelihood[j, i] = -neg_log_likelihood([MU[j, i], SIGMA[j, i]], data)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 対数尤度の等高線図
    contour = axes[0].contourf(MU, SIGMA, log_likelihood, levels=30, cmap='viridis')
    axes[0].plot(mu_true, sigma_true, 'r*', markersize=15, label='真の値')
    axes[0].plot(mu_mle, sigma_mle, 'wo', markersize=10, label='MLE')
    axes[0].set_xlabel('μ (平均)')
    axes[0].set_ylabel('σ (標準偏差)')
    axes[0].set_title('対数尤度関数の等高線図')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(contour, ax=axes[0], label='対数尤度')
    
    # データのヒストグラムと推定された分布
    axes[1].hist(data, bins=15, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black', label='データ')
    x_range = np.linspace(data.min(), data.max(), 200)
    axes[1].plot(x_range, stats.norm.pdf(x_range, mu_true, sigma_true),
                 'r-', linewidth=2, label=f'真の分布 N({mu_true}, {sigma_true}²)')
    axes[1].plot(x_range, stats.norm.pdf(x_range, mu_mle, sigma_mle),
                 'b--', linewidth=2, label=f'MLE分布 N({mu_mle:.1f}, {sigma_mle:.1f}²)')
    axes[1].set_xlabel('値')
    axes[1].set_ylabel('確率密度')
    axes[1].set_title('データと推定された分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

**📌 重要ポイント**  
正規分布の場合、MLEの解析解は標本平均と標本分散（\\( n \\)で割ったもの）に一致します。 一般的な分布では数値最適化が必要になります。 

#### 💻 コード例3: 二項分布・ポアソン分布の最尤推定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # === 二項分布の最尤推定 ===
    print("=== 二項分布の最尤推定 ===")
    # データ生成（n=10の試行を50回実施）
    n_trials = 10
    p_true = 0.3
    np.random.seed(42)
    data_binomial = np.random.binomial(n_trials, p_true, 50)
    
    # MLEの解析解: p_hat = (総成功数) / (総試行数)
    p_mle = np.sum(data_binomial) / (len(data_binomial) * n_trials)
    print(f"真のp: {p_true}, MLE: {p_mle:.4f}")
    
    # === ポアソン分布の最尤推定 ===
    print("\n=== ポアソン分布の最尤推定 ===")
    # データ生成
    lambda_true = 5.0
    data_poisson = np.random.poisson(lambda_true, 100)
    
    # MLEの解析解: lambda_hat = 標本平均
    lambda_mle = np.mean(data_poisson)
    print(f"真のλ: {lambda_true}, MLE: {lambda_mle:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 二項分布
    x_binom = np.arange(0, n_trials+1)
    axes[0].hist(data_binomial, bins=np.arange(-0.5, n_trials+1.5, 1),
                 density=True, alpha=0.7, color='skyblue',
                 edgecolor='black', label='データ')
    axes[0].plot(x_binom, stats.binom.pmf(x_binom, n_trials, p_true),
                 'ro-', markersize=8, linewidth=2, label=f'真の分布 B({n_trials}, {p_true})')
    axes[0].plot(x_binom, stats.binom.pmf(x_binom, n_trials, p_mle),
                 'b^--', markersize=6, linewidth=2, label=f'MLE分布 B({n_trials}, {p_mle:.2f})')
    axes[0].set_xlabel('成功回数')
    axes[0].set_ylabel('確率質量')
    axes[0].set_title('二項分布の最尤推定')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ポアソン分布
    x_poisson = np.arange(0, max(data_poisson)+1)
    axes[1].hist(data_poisson, bins=np.arange(-0.5, max(data_poisson)+1.5, 1),
                 density=True, alpha=0.7, color='lightgreen',
                 edgecolor='black', label='データ')
    axes[1].plot(x_poisson, stats.poisson.pmf(x_poisson, lambda_true),
                 'ro-', markersize=6, linewidth=2, label=f'真の分布 Poisson({lambda_true})')
    axes[1].plot(x_poisson, stats.poisson.pmf(x_poisson, lambda_mle),
                 'b^--', markersize=5, linewidth=2, label=f'MLE分布 Poisson({lambda_mle:.2f})')
    axes[1].set_xlabel('発生回数')
    axes[1].set_ylabel('確率質量')
    axes[1].set_title('ポアソン分布の最尤推定')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 1.4 モーメント法（Method of Moments）

#### 📘 モーメント法の原理

母集団のモーメント（積率）と標本モーメントを等値することでパラメータを推定します。

**k次モーメント** ：

$$ m_k = E[X^k], \quad \hat{m}_k = \frac{1}{n}\sum_{i=1}^n X_i^k $$ 

パラメータ数と同数のモーメント方程式を立てて解きます。

#### 💻 コード例4: モーメント法による推定（ガンマ分布）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import fsolve
    
    # 真のパラメータ（ガンマ分布）
    # ガンマ分布: Gamma(k, θ), E[X] = kθ, Var[X] = kθ²
    k_true = 3.0  # 形状パラメータ
    theta_true = 2.0  # スケールパラメータ
    
    # データ生成
    np.random.seed(42)
    n = 200
    data = np.random.gamma(k_true, theta_true, n)
    
    # モーメント法による推定
    # E[X] = kθ, Var[X] = kθ² より
    # m1 = kθ, m2 - m1² = kθ²
    m1 = np.mean(data)
    m2 = np.mean(data**2)
    var_sample = m2 - m1**2
    
    # 連立方程式: m1 = kθ, var = kθ²
    # var/m1 = θ より θ_hat = var/m1
    # k_hat = m1/θ_hat = m1²/var
    theta_mom = var_sample / m1
    k_mom = m1**2 / var_sample
    
    print("=== モーメント法による推定（ガンマ分布） ===")
    print(f"真の形状パラメータ k: {k_true}")
    print(f"モーメント法推定 k: {k_mom:.4f}")
    print(f"真のスケールパラメータ θ: {theta_true}")
    print(f"モーメント法推定 θ: {theta_mom:.4f}")
    
    # 最尤推定との比較（数値解法が必要）
    def gamma_mle_equations(params, data):
        """ガンマ分布のMLEの方程式"""
        k, theta = params
        n = len(data)
        eq1 = n * np.log(theta) + np.sum(np.log(data)) - n * (np.log(k) + stats.digamma(k))
        eq2 = np.sum(data) - n * k * theta
        return [eq1, eq2]
    
    # 初期値としてモーメント法の結果を使用
    initial_guess = [k_mom, theta_mom]
    k_mle, theta_mle = fsolve(gamma_mle_equations, initial_guess, args=(data,))
    
    print(f"\n最尤推定 k: {k_mle:.4f}")
    print(f"最尤推定 θ: {theta_mle:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # データのヒストグラムと推定分布
    x_range = np.linspace(0, data.max(), 200)
    axes[0].hist(data, bins=30, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black', label='データ')
    axes[0].plot(x_range, stats.gamma.pdf(x_range, k_true, scale=theta_true),
                 'r-', linewidth=2.5, label=f'真の分布 Γ({k_true}, {theta_true})')
    axes[0].plot(x_range, stats.gamma.pdf(x_range, k_mom, scale=theta_mom),
                 'g--', linewidth=2, label=f'モーメント法 Γ({k_mom:.2f}, {theta_mom:.2f})')
    axes[0].plot(x_range, stats.gamma.pdf(x_range, k_mle, scale=theta_mle),
                 'b:', linewidth=2, label=f'MLE Γ({k_mle:.2f}, {theta_mle:.2f})')
    axes[0].set_xlabel('値')
    axes[0].set_ylabel('確率密度')
    axes[0].set_title('ガンマ分布の推定（モーメント法 vs MLE）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Qプロット（推定精度の視覚的評価）
    theoretical_quantiles = np.linspace(0.01, 0.99, 100)
    data_sorted = np.sort(data)
    empirical_quantiles = np.linspace(0, 1, len(data_sorted))
    
    for method, k, theta, color, linestyle in [
        ('真の分布', k_true, theta_true, 'red', '-'),
        ('モーメント法', k_mom, theta_mom, 'green', '--'),
        ('MLE', k_mle, theta_mle, 'blue', ':')
    ]:
        theoretical_values = stats.gamma.ppf(theoretical_quantiles, k, scale=theta)
        axes[1].plot(theoretical_values,
                     np.quantile(data, theoretical_quantiles),
                     color=color, linestyle=linestyle, linewidth=2, label=method)
    
    axes[1].plot([0, data.max()], [0, data.max()], 'k--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('理論分位数')
    axes[1].set_ylabel('標本分位数')
    axes[1].set_title('Q-Qプロット（推定精度の比較）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 1.5 推定量のバイアス-分散トレードオフ

#### 📘 平均二乗誤差（MSE）の分解

推定量 \\( \hat{\theta} \\) の**平均二乗誤差（Mean Squared Error）** は：

$$ \text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta}) $$ 

ここで：

  * **バイアス** : \\( \text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta \\)
  * **分散** : \\( \text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2] \\)

不偏推定量でも分散が大きければMSEが大きくなる可能性があります。

#### 💻 コード例5: バイアス-分散トレードオフの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 真の母平均と母分散
    mu_true = 100
    sigma_true = 20
    
    # 標本サイズ
    n = 20
    
    # シミュレーション
    np.random.seed(42)
    n_simulations = 5000
    
    # 3種類の推定量
    # 1. 標本平均（不偏）
    # 2. 定数推定量（バイアスあり、分散ゼロ）
    # 3. Shrinkage推定量（バイアスあり、分散小）
    
    estimates_unbiased = []
    estimates_constant = []
    estimates_shrinkage = []
    
    constant_value = 95  # 事前知識からの推測値
    shrinkage_factor = 0.7  # 縮小係数
    
    for _ in range(n_simulations):
        sample = np.random.normal(mu_true, sigma_true, n)
    
        # 不偏推定量（標本平均）
        estimates_unbiased.append(np.mean(sample))
    
        # 定数推定量
        estimates_constant.append(constant_value)
    
        # Shrinkage推定量: 標本平均を定数値に縮小
        estimates_shrinkage.append(
            shrinkage_factor * np.mean(sample) + (1 - shrinkage_factor) * constant_value
        )
    
    # MSEの計算
    def compute_mse_components(estimates, true_value):
        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_value
        variance = np.var(estimates)
        mse = np.mean((estimates - true_value)**2)
        return bias, variance, mse
    
    bias_u, var_u, mse_u = compute_mse_components(estimates_unbiased, mu_true)
    bias_c, var_c, mse_c = compute_mse_components(estimates_constant, mu_true)
    bias_s, var_s, mse_s = compute_mse_components(estimates_shrinkage, mu_true)
    
    print("=== バイアス-分散トレードオフ ===")
    print(f"\n不偏推定量（標本平均）:")
    print(f"  バイアス: {bias_u:.4f}, 分散: {var_u:.4f}, MSE: {mse_u:.4f}")
    print(f"\n定数推定量:")
    print(f"  バイアス: {bias_c:.4f}, 分散: {var_c:.4f}, MSE: {mse_c:.4f}")
    print(f"\nShrinkage推定量:")
    print(f"  バイアス: {bias_s:.4f}, 分散: {var_s:.4f}, MSE: {mse_s:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 推定量の分布
    axes[0, 0].hist(estimates_unbiased, bins=50, density=True, alpha=0.6,
                    color='blue', edgecolor='black', label='不偏推定量')
    axes[0, 0].hist(estimates_shrinkage, bins=50, density=True, alpha=0.6,
                    color='green', edgecolor='black', label='Shrinkage推定量')
    axes[0, 0].axvline(mu_true, color='red', linestyle='--', linewidth=2,
                       label=f'真の値: {mu_true}')
    axes[0, 0].axvline(constant_value, color='orange', linestyle='--',
                       linewidth=2, label=f'定数値: {constant_value}')
    axes[0, 0].set_xlabel('推定値')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('推定量の分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE分解の棒グラフ
    methods = ['不偏推定量', '定数推定量', 'Shrinkage推定量']
    biases_sq = [bias_u**2, bias_c**2, bias_s**2]
    variances = [var_u, var_c, var_s]
    x_pos = np.arange(len(methods))
    
    axes[0, 1].bar(x_pos, biases_sq, width=0.35, label='バイアス²',
                   color='orange', alpha=0.8)
    axes[0, 1].bar(x_pos, variances, width=0.35, bottom=biases_sq,
                   label='分散', color='skyblue', alpha=0.8)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 1].set_ylabel('値')
    axes[0, 1].set_title('MSEの分解（バイアス² + 分散）')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 二乗誤差の分布
    sq_errors_u = (np.array(estimates_unbiased) - mu_true)**2
    sq_errors_s = (np.array(estimates_shrinkage) - mu_true)**2
    
    axes[1, 0].hist(sq_errors_u, bins=50, density=True, alpha=0.6,
                    color='blue', edgecolor='black', label='不偏推定量')
    axes[1, 0].hist(sq_errors_s, bins=50, density=True, alpha=0.6,
                    color='green', edgecolor='black', label='Shrinkage推定量')
    axes[1, 0].axvline(mse_u, color='blue', linestyle='--', linewidth=2)
    axes[1, 0].axvline(mse_s, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('二乗誤差')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('二乗誤差の分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Shrinkage係数とMSEの関係
    shrinkage_factors = np.linspace(0, 1, 50)
    mse_by_shrinkage = []
    
    for sf in shrinkage_factors:
        est = [sf * e + (1-sf) * constant_value for e in estimates_unbiased]
        _, _, mse = compute_mse_components(est, mu_true)
        mse_by_shrinkage.append(mse)
    
    axes[1, 1].plot(shrinkage_factors, mse_by_shrinkage, 'b-', linewidth=2)
    axes[1, 1].axvline(shrinkage_factor, color='green', linestyle='--',
                       linewidth=2, label=f'選択した係数: {shrinkage_factor}')
    axes[1, 1].axhline(mse_u, color='blue', linestyle=':', linewidth=2,
                       label=f'不偏推定量のMSE: {mse_u:.2f}')
    optimal_sf = shrinkage_factors[np.argmin(mse_by_shrinkage)]
    axes[1, 1].axvline(optimal_sf, color='red', linestyle='--',
                       linewidth=2, label=f'最適係数: {optimal_sf:.2f}')
    axes[1, 1].set_xlabel('Shrinkage係数')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Shrinkage係数とMSEの関係')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最適Shrinkage係数: {optimal_sf:.4f}")

**📌 重要ポイント**  
不偏推定量が常に最良とは限りません。少しバイアスを許容することで分散を大きく減らせる場合、 MSEは小さくなります（James-Stein推定量など）。 

## 1.6 Fisher情報量とCramér-Rao下界

#### 📘 Fisher情報量

パラメータ \\( \theta \\) の**Fisher情報量** は：

$$ I(\theta) = E\left[\left(\frac{\partial \log f(X;\theta)}{\partial \theta}\right)^2\right] = -E\left[\frac{\partial^2 \log f(X;\theta)}{\partial \theta^2}\right] $$ 

データがパラメータに関して持つ情報量を表します。

#### 📘 Cramér-Rao下界

任意の不偏推定量 \\( \hat{\theta} \\) の分散は：

$$ \text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)} $$ 

この下界を達成する推定量が**有効推定量** です。

#### 💻 コード例6: Fisher情報量とCramér-Rao下界の計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 正規分布 N(μ, σ²) のFisher情報量
    # I(μ) = 1/σ², I(σ²) = 1/(2σ⁴)
    
    def normal_fisher_info_mean(sigma):
        """正規分布の平均に関するFisher情報量"""
        return 1 / sigma**2
    
    def normal_fisher_info_var(sigma):
        """正規分布の分散に関するFisher情報量"""
        return 1 / (2 * sigma**4)
    
    # パラメータ設定
    mu_true = 50
    sigma_true = 10
    sample_sizes = np.array([10, 20, 50, 100, 200, 500])
    
    # シミュレーション
    n_simulations = 10000
    np.random.seed(42)
    
    results = []
    for n in sample_sizes:
        sample_means = []
        sample_vars = []
    
        for _ in range(n_simulations):
            sample = np.random.normal(mu_true, sigma_true, n)
            sample_means.append(np.mean(sample))
            sample_vars.append(np.var(sample, ddof=1))
    
        # 実測分散
        empirical_var_mean = np.var(sample_means)
        empirical_var_var = np.var(sample_vars)
    
        # Cramér-Rao下界
        cr_bound_mean = 1 / (n * normal_fisher_info_mean(sigma_true))
        cr_bound_var = 1 / (n * normal_fisher_info_var(sigma_true))
    
        results.append({
            'n': n,
            'empirical_var_mean': empirical_var_mean,
            'cr_bound_mean': cr_bound_mean,
            'empirical_var_var': empirical_var_var,
            'cr_bound_var': cr_bound_var
        })
    
    # 結果の表示
    print("=== Fisher情報量とCramér-Rao下界 ===")
    print(f"正規分布 N({mu_true}, {sigma_true}²)")
    print(f"\nFisher情報量（1標本あたり）:")
    print(f"  I(μ) = {normal_fisher_info_mean(sigma_true):.6f}")
    print(f"  I(σ²) = {normal_fisher_info_var(sigma_true):.10f}")
    print()
    
    for r in results:
        print(f"n={r['n']}:")
        print(f"  平均の推定量: 実測分散={r['empirical_var_mean']:.4f}, "
              f"CR下界={r['cr_bound_mean']:.4f}, "
              f"効率={(r['cr_bound_mean']/r['empirical_var_mean'])*100:.2f}%")
        print(f"  分散の推定量: 実測分散={r['empirical_var_var']:.4f}, "
              f"CR下界={r['cr_bound_var']:.4f}, "
              f"効率={(r['cr_bound_var']/r['empirical_var_var'])*100:.2f}%")
        print()
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 平均の推定量
    empirical_vars_mean = [r['empirical_var_mean'] for r in results]
    cr_bounds_mean = [r['cr_bound_mean'] for r in results]
    
    axes[0].plot(sample_sizes, empirical_vars_mean, 'bo-',
                 markersize=8, linewidth=2, label='実測分散')
    axes[0].plot(sample_sizes, cr_bounds_mean, 'r^--',
                 markersize=8, linewidth=2, label='Cramér-Rao下界')
    axes[0].set_xlabel('標本サイズ n')
    axes[0].set_ylabel('分散')
    axes[0].set_title('標本平均の分散 vs Cramér-Rao下界')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # 分散の推定量
    empirical_vars_var = [r['empirical_var_var'] for r in results]
    cr_bounds_var = [r['cr_bound_var'] for r in results]
    
    axes[1].plot(sample_sizes, empirical_vars_var, 'bo-',
                 markersize=8, linewidth=2, label='実測分散')
    axes[1].plot(sample_sizes, cr_bounds_var, 'r^--',
                 markersize=8, linewidth=2, label='Cramér-Rao下界')
    axes[1].set_xlabel('標本サイズ n')
    axes[1].set_ylabel('分散')
    axes[1].set_title('標本分散の分散 vs Cramér-Rao下界')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()

**📌 重要ポイント**  
正規分布の場合、標本平均はCramér-Rao下界を達成する（100%効率的）のに対し、 標本分散は漸近的にのみ下界を達成します。 

## 1.7 材料特性データの推定

#### 💻 コード例7: 材料強度データの統計的推定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import minimize
    
    # 実験データの生成（実際には実験から得られる）
    # 材料強度の分布は正規分布またはWeibull分布でモデル化されることが多い
    np.random.seed(42)
    
    # シナリオ1: 正規分布でモデル化
    n_samples = 30
    true_mean_strength = 450  # MPa
    true_std_strength = 30  # MPa
    strength_data_normal = np.random.normal(true_mean_strength, true_std_strength, n_samples)
    
    # シナリオ2: Weibull分布でモデル化（脆性材料に適している）
    # Weibull分布: パラメータ (k: 形状, λ: スケール)
    k_true = 15  # 形状パラメータ（大きいほど分散が小さい）
    lambda_true = 470  # スケールパラメータ
    strength_data_weibull = np.random.weibull(k_true, n_samples) * lambda_true
    
    print("=== 材料強度データの統計的推定 ===")
    print("\n【正規分布モデル】")
    # 正規分布の推定
    mean_est = np.mean(strength_data_normal)
    std_est = np.std(strength_data_normal, ddof=1)
    
    print(f"標本平均（MLE）: {mean_est:.2f} MPa (真値: {true_mean_strength} MPa)")
    print(f"標本標準偏差: {std_est:.2f} MPa (真値: {true_std_strength} MPa)")
    
    # 95%信頼区間（次章で詳しく扱う）
    se = std_est / np.sqrt(n_samples)
    ci_95 = stats.t.interval(0.95, n_samples-1, loc=mean_est, scale=se)
    print(f"平均強度の95%信頼区間: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] MPa")
    
    print("\n【Weibull分布モデル】")
    # Weibull分布のMLE（SciPyの関数を使用）
    # scipy.stats.weibull_min.fit() は (c, loc, scale) を返す
    # c = k (形状), scale = λ (スケール)
    params = stats.weibull_min.fit(strength_data_weibull, floc=0)
    k_est, loc_est, lambda_est = params
    
    print(f"形状パラメータk（MLE）: {k_est:.2f} (真値: {k_true})")
    print(f"スケールパラメータλ（MLE）: {lambda_est:.2f} (真値: {lambda_true})")
    print(f"平均強度（Weibull）: {lambda_est * stats.gamma(1 + 1/k_est):.2f} MPa")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 正規分布モデル
    axes[0, 0].hist(strength_data_normal, bins=12, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black', label='実験データ')
    x_range = np.linspace(strength_data_normal.min(), strength_data_normal.max(), 200)
    axes[0, 0].plot(x_range, stats.norm.pdf(x_range, true_mean_strength, true_std_strength),
                    'r-', linewidth=2.5, label=f'真の分布 N({true_mean_strength}, {true_std_strength}²)')
    axes[0, 0].plot(x_range, stats.norm.pdf(x_range, mean_est, std_est),
                    'b--', linewidth=2, label=f'推定分布 N({mean_est:.1f}, {std_est:.1f}²)')
    axes[0, 0].set_xlabel('強度 [MPa]')
    axes[0, 0].set_ylabel('確率密度')
    axes[0, 0].set_title('正規分布モデル: 材料強度の推定')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 正規分布のQ-Qプロット
    stats.probplot(strength_data_normal, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('正規Q-Qプロット（正規性の確認）')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weibull分布モデル
    axes[1, 0].hist(strength_data_weibull, bins=12, density=True, alpha=0.7,
                    color='lightgreen', edgecolor='black', label='実験データ')
    x_range_w = np.linspace(0, strength_data_weibull.max(), 200)
    axes[1, 0].plot(x_range_w, stats.weibull_min.pdf(x_range_w, k_true, scale=lambda_true),
                    'r-', linewidth=2.5, label=f'真の分布 Weibull({k_true}, {lambda_true})')
    axes[1, 0].plot(x_range_w, stats.weibull_min.pdf(x_range_w, k_est, scale=lambda_est),
                    'b--', linewidth=2, label=f'推定分布 Weibull({k_est:.1f}, {lambda_est:.1f})')
    axes[1, 0].set_xlabel('強度 [MPa]')
    axes[1, 0].set_ylabel('確率密度')
    axes[1, 0].set_title('Weibull分布モデル: 材料強度の推定')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Weibull確率プロット
    sorted_data = np.sort(strength_data_weibull)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n+1) / (n+1)
    weibull_y = np.log(-np.log(1 - empirical_cdf))
    weibull_x = np.log(sorted_data)
    
    axes[1, 1].plot(weibull_x, weibull_y, 'bo', markersize=6, label='実験データ')
    # 推定されたWeibull分布の理論直線
    x_fit = np.array([weibull_x.min(), weibull_x.max()])
    y_fit = k_est * (x_fit - np.log(lambda_est))
    axes[1, 1].plot(x_fit, y_fit, 'r-', linewidth=2, label='推定Weibull直線')
    axes[1, 1].set_xlabel('ln(強度)')
    axes[1, 1].set_ylabel('ln(-ln(1-F))')
    axes[1, 1].set_title('Weibull確率プロット（適合度の確認）')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 破壊確率の推定（工学的に重要）
    print("\n【破壊確率の推定】")
    critical_strength = 400  # MPa
    prob_failure_normal = stats.norm.cdf(critical_strength, mean_est, std_est)
    prob_failure_weibull = stats.weibull_min.cdf(critical_strength, k_est, scale=lambda_est)
    
    print(f"400 MPa以下で破壊する確率:")
    print(f"  正規分布モデル: {prob_failure_normal*100:.2f}%")
    print(f"  Weibullモデル: {prob_failure_weibull*100:.2f}%")

**📌 実践的ポイント**  
材料強度データの解析では： 

  * 延性材料（金属など）→ 正規分布またはログ正規分布
  * 脆性材料（セラミックスなど）→ Weibull分布
  * Q-Qプロットや確率プロットで分布の適合度を確認
  * 破壊確率や信頼性評価に推定結果を活用

#### 📝 練習問題

  1. 標本中央値は母平均の不偏推定量か調べてください（正規分布の場合）。
  2. 指数分布 \\( f(x; \lambda) = \lambda e^{-\lambda x} \\) のパラメータ \\( \lambda \\) の最尤推定量を導出してください。
  3. 一様分布 \\( U(0, \theta) \\) のパラメータ \\( \theta \\) に対して、\\( \max(X_1, \ldots, X_n) \\) が一致推定量であることを示してください。
  4. ベルヌーイ分布のパラメータ \\( p \\) に関するFisher情報量を計算してください。

## まとめ

  * 推定理論は標本から母集団パラメータを推測する統計学の基礎である
  * 不偏性・一致性・有効性は推定量の重要な性質である
  * 最尤推定法は最も一般的かつ強力な推定手法である
  * モーメント法は計算が簡単で初期推定値として有用である
  * バイアス-分散トレードオフは推定量の設計で重要な考慮事項である
  * Fisher情報量とCramér-Rao下界は推定の理論的限界を与える
  * 材料科学では正規分布やWeibull分布を用いた推定が重要である

[← シリーズ目次](<index.html>) [第2章: 区間推定と信頼区間 →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
