---
title: "第4章: ベイズ推論の基礎とMCMC"
chapter_title: "第4章: ベイズ推論の基礎とMCMC"
subtitle: Bayesian Inference and Markov Chain Monte Carlo
---

[基礎数理道場](<../index.html>) > [推測統計学とベイズ統計](<index.html>) > 第4章 

## 4.1 ベイズ推論の基本概念

ベイズ推論は、事前知識とデータを組み合わせて不確実性を定量化する統計的枠組みです。 頻度論的推測統計と異なり、パラメータを確率変数として扱います。 

#### 📘 ベイズの定理

パラメータ \\( \theta \\) とデータ \\( D \\) について：

$$ P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} $$ 

各項の意味：

  * **事後分布 \\( P(\theta | D) \\)** : データ観測後のパラメータの確率分布
  * **尤度 \\( P(D | \theta) \\)** : パラメータ \\( \theta \\) のもとでデータ \\( D \\) が得られる確率
  * **事前分布 \\( P(\theta) \\)** : データ観測前のパラメータに関する信念
  * **周辺尤度 \\( P(D) \\)** : データの確率（正規化定数）

実用的には：

$$ P(\theta | D) \propto P(D | \theta) P(\theta) $$ 

事後分布は尤度と事前分布の積に比例します。

#### 💻 コード例1: ベイズの定理の実装（コイン投げ問題）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # コイン投げ問題: コインの表が出る確率pを推定
    # データ: 10回投げて7回表が出た
    n_trials = 10
    n_success = 7
    
    # 事前分布: Beta(2, 2) (弱い情報のある事前分布)
    alpha_prior = 2
    beta_prior = 2
    
    # 尤度: Binomial(n_success | n_trials, p)
    # 共役事前分布の性質により、事後分布もBeta分布
    # Beta(alpha_prior + n_success, beta_prior + n_trials - n_success)
    alpha_post = alpha_prior + n_success
    beta_post = beta_prior + (n_trials - n_success)
    
    print("=== ベイズ推論: コイン投げ問題 ===")
    print(f"データ: {n_trials}回中{n_success}回表")
    print(f"事前分布: Beta({alpha_prior}, {beta_prior})")
    print(f"事後分布: Beta({alpha_post}, {beta_post})")
    print(f"\n事後平均: {alpha_post/(alpha_post + beta_post):.4f}")
    print(f"事後モード: {(alpha_post-1)/(alpha_post+beta_post-2):.4f}")
    
    # 95%信用区間 (Credible Interval)
    ci_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
    ci_upper = stats.beta.ppf(0.975, alpha_post, beta_post)
    print(f"95%信用区間: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 最尤推定との比較
    mle = n_success / n_trials
    print(f"\n頻度論的MLE: {mle:.4f}")
    
    # 可視化
    p_values = np.linspace(0, 1, 200)
    prior_pdf = stats.beta.pdf(p_values, alpha_prior, beta_prior)
    likelihood = stats.binom.pmf(n_success, n_trials, p_values)
    posterior_pdf = stats.beta.pdf(p_values, alpha_post, beta_post)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 事前分布
    axes[0, 0].plot(p_values, prior_pdf, 'b-', linewidth=2)
    axes[0, 0].fill_between(p_values, 0, prior_pdf, alpha=0.3, color='blue')
    axes[0, 0].set_xlabel('p (表の確率)')
    axes[0, 0].set_ylabel('確率密度')
    axes[0, 0].set_title(f'事前分布 Beta({alpha_prior}, {beta_prior})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 尤度関数
    axes[0, 1].plot(p_values, likelihood, 'g-', linewidth=2)
    axes[0, 1].fill_between(p_values, 0, likelihood, alpha=0.3, color='green')
    axes[0, 1].axvline(mle, color='red', linestyle='--', linewidth=2,
                       label=f'MLE: {mle:.2f}')
    axes[0, 1].set_xlabel('p')
    axes[0, 1].set_ylabel('尤度')
    axes[0, 1].set_title(f'尤度関数 Bin({n_success}|{n_trials}, p)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 事後分布
    axes[1, 0].plot(p_values, posterior_pdf, 'r-', linewidth=2)
    axes[1, 0].fill_between(p_values, 0, posterior_pdf, alpha=0.3, color='red')
    axes[1, 0].axvline(alpha_post/(alpha_post+beta_post), color='blue',
                       linestyle='--', linewidth=2, label='事後平均')
    axes[1, 0].axvspan(ci_lower, ci_upper, alpha=0.2, color='yellow',
                       label='95%信用区間')
    axes[1, 0].set_xlabel('p')
    axes[1, 0].set_ylabel('確率密度')
    axes[1, 0].set_title(f'事後分布 Beta({alpha_post}, {beta_post})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3つの分布を重ねて表示
    axes[1, 1].plot(p_values, prior_pdf / prior_pdf.max(), 'b-',
                    linewidth=2, label='事前分布（正規化）')
    axes[1, 1].plot(p_values, likelihood / likelihood.max(), 'g--',
                    linewidth=2, label='尤度（正規化）')
    axes[1, 1].plot(p_values, posterior_pdf / posterior_pdf.max(), 'r-',
                    linewidth=2, label='事後分布（正規化）')
    axes[1, 1].axvline(mle, color='orange', linestyle=':', linewidth=2,
                       label=f'MLE: {mle:.2f}')
    axes[1, 1].set_xlabel('p')
    axes[1, 1].set_ylabel('正規化された確率密度')
    axes[1, 1].set_title('ベイズ更新の可視化')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 逐次ベイズ更新のデモ
    print("\n=== 逐次ベイズ更新 ===")
    data_sequence = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]  # 1=表, 0=裏
    alpha_seq = alpha_prior
    beta_seq = beta_prior
    
    print(f"初期事前分布: Beta({alpha_seq}, {beta_seq})")
    for i, outcome in enumerate(data_sequence, 1):
        if outcome == 1:
            alpha_seq += 1
        else:
            beta_seq += 1
        mean = alpha_seq / (alpha_seq + beta_seq)
        print(f"  データ{i}点後: Beta({alpha_seq}, {beta_seq}), 平均={mean:.4f}")
    

**📌 ベイズ推論 vs 頻度論的推論**  

  * **ベイズ** : 「pが0.6～0.8の範囲にある確率が95%」と解釈できる
  * **頻度論** : 「このような区間推定を100回行えば、95回は真のpを含む」

ベイズ推論は確率的言明が直感的で、事前知識を形式的に組み込める利点があります。 

## 4.2 共役事前分布

#### 📘 共役事前分布の性質

事前分布と事後分布が同じ分布族に属する場合、その事前分布を**共役事前分布** と呼びます。

**主な共役対：**

  * 二項分布の尤度 + Beta事前分布 → Beta事後分布
  * ポアソン分布の尤度 + Gamma事前分布 → Gamma事後分布
  * 正規分布の尤度（既知分散）+ 正規事前分布 → 正規事後分布
  * 正規分布の尤度（未知分散）+ Normal-Gamma事前分布 → Normal-Gamma事後分布

共役事前分布を使うと解析的に事後分布を計算でき、MCMCが不要になる利点があります。

#### 💻 コード例2: 共役事前分布（Beta-Binomial）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 様々な事前分布でベイズ更新を比較
    n_trials = 20
    n_success = 15
    
    # 3種類の事前分布
    priors = [
        {"name": "無情報 Uniform", "alpha": 1, "beta": 1},
        {"name": "弱情報 Beta(2,2)", "alpha": 2, "beta": 2},
        {"name": "強情報 Beta(8,2)", "alpha": 8, "beta": 2}  # 表が出やすいと信じている
    ]
    
    p_values = np.linspace(0, 1, 200)
    
    fig, axes = plt.subplots(len(priors), 3, figsize=(15, 10))
    
    for i, prior in enumerate(priors):
        alpha_pr = prior["alpha"]
        beta_pr = prior["beta"]
        alpha_po = alpha_pr + n_success
        beta_po = beta_pr + (n_trials - n_success)
        
        # 事前分布
        prior_pdf = stats.beta.pdf(p_values, alpha_pr, beta_pr)
        axes[i, 0].plot(p_values, prior_pdf, 'b-', linewidth=2)
        axes[i, 0].fill_between(p_values, 0, prior_pdf, alpha=0.3, color='blue')
        axes[i, 0].set_title(f'{prior["name"]}\nBeta({alpha_pr}, {beta_pr})')
        axes[i, 0].set_ylabel('確率密度')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 尤度
        likelihood = stats.binom.pmf(n_success, n_trials, p_values)
        axes[i, 1].plot(p_values, likelihood, 'g-', linewidth=2)
        axes[i, 1].fill_between(p_values, 0, likelihood, alpha=0.3, color='green')
        axes[i, 1].set_title(f'尤度\nBin({n_success}|{n_trials}, p)')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 事後分布
        posterior_pdf = stats.beta.pdf(p_values, alpha_po, beta_po)
        axes[i, 2].plot(p_values, posterior_pdf, 'r-', linewidth=2)
        axes[i, 2].fill_between(p_values, 0, posterior_pdf, alpha=0.3, color='red')
        post_mean = alpha_po / (alpha_po + beta_po)
        axes[i, 2].axvline(post_mean, color='blue', linestyle='--',
                           linewidth=2, label=f'平均: {post_mean:.3f}')
        axes[i, 2].set_title(f'事後分布\nBeta({alpha_po}, {beta_po})')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        if i == len(priors) - 1:
            axes[i, 0].set_xlabel('p')
            axes[i, 1].set_xlabel('p')
            axes[i, 2].set_xlabel('p')
    
    plt.suptitle(f'事前分布の違いによる事後分布の変化（データ: {n_trials}回中{n_success}回成功）',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()
    
    # 数値的まとめ
    print("=== 事前分布の影響 ===")
    print(f"データ: {n_trials}回中{n_success}回成功 (MLE={n_success/n_trials:.3f})\n")
    for prior in priors:
        alpha_pr = prior["alpha"]
        beta_pr = prior["beta"]
        alpha_po = alpha_pr + n_success
        beta_po = beta_pr + (n_trials - n_success)
        
        prior_mean = alpha_pr / (alpha_pr + beta_pr)
        post_mean = alpha_po / (alpha_po + beta_po)
        
        print(f"{prior['name']}:")
        print(f"  事前平均: {prior_mean:.4f}")
        print(f"  事後平均: {post_mean:.4f}")
        print(f"  変化: {post_mean - prior_mean:+.4f}\n")
    

#### 💻 コード例3: 正規分布の共役事前分布
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 正規分布データの平均μの推定（分散σ²は既知）
    sigma_known = 10  # 既知の標準偏差
    n = 15
    np.random.seed(42)
    true_mu = 100
    data = np.random.normal(true_mu, sigma_known, n)
    
    # 事前分布: N(μ0, τ0²)
    mu_0 = 95  # 事前平均
    tau_0 = 8  # 事前標準偏差
    
    # データの統計量
    data_mean = np.mean(data)
    data_se = sigma_known / np.sqrt(n)
    
    # 事後分布（正規分布の共役性）
    # 精度（分散の逆数）での計算が便利
    precision_prior = 1 / tau_0**2
    precision_likelihood = n / sigma_known**2
    precision_post = precision_prior + precision_likelihood
    
    mu_post = (precision_prior * mu_0 + precision_likelihood * data_mean) / precision_post
    tau_post = 1 / np.sqrt(precision_post)
    
    print("=== 正規分布の共役ベイズ推論 ===")
    print(f"データ: n={n}, 標本平均={data_mean:.2f}, 既知σ={sigma_known}")
    print(f"\n事前分布: N({mu_0}, {tau_0}²)")
    print(f"事後分布: N({mu_post:.2f}, {tau_post:.2f}²)")
    print(f"\n頻度論的推定:")
    print(f"  標本平均: {data_mean:.2f}")
    print(f"  標準誤差: {data_se:.2f}")
    
    # 可視化
    x = np.linspace(70, 120, 300)
    prior_pdf = stats.norm.pdf(x, mu_0, tau_0)
    likelihood_pdf = stats.norm.pdf(x, data_mean, data_se)
    posterior_pdf = stats.norm.pdf(x, mu_post, tau_post)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 分布の重ね合わせ
    axes[0].plot(x, prior_pdf, 'b-', linewidth=2, label=f'事前分布 N({mu_0}, {tau_0}²)')
    axes[0].plot(x, likelihood_pdf, 'g--', linewidth=2,
                 label=f'尤度 N({data_mean:.1f}, {data_se:.1f}²)')
    axes[0].plot(x, posterior_pdf, 'r-', linewidth=2,
                 label=f'事後分布 N({mu_post:.1f}, {tau_post:.1f}²)')
    axes[0].axvline(true_mu, color='black', linestyle=':', linewidth=2,
                    label=f'真の値: {true_mu}')
    axes[0].set_xlabel('μ')
    axes[0].set_ylabel('確率密度')
    axes[0].set_title('正規分布の共役ベイズ推論')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # データサイズの影響
    sample_sizes = [5, 10, 20, 50, 100]
    post_means = []
    post_stds = []
    
    np.random.seed(42)
    for ns in sample_sizes:
        data_temp = np.random.normal(true_mu, sigma_known, ns)
        dm = np.mean(data_temp)
        prec_lik = ns / sigma_known**2
        prec_po = precision_prior + prec_lik
        mu_po = (precision_prior * mu_0 + prec_lik * dm) / prec_po
        tau_po = 1 / np.sqrt(prec_po)
        post_means.append(mu_po)
        post_stds.append(tau_po)
    
    axes[1].errorbar(sample_sizes, post_means,
                     yerr=[2*s for s in post_stds],
                     fmt='o-', markersize=8, capsize=8, linewidth=2,
                     label='事後平均±2SD')
    axes[1].axhline(mu_0, color='blue', linestyle='--', linewidth=2,
                    label=f'事前平均: {mu_0}')
    axes[1].axhline(true_mu, color='red', linestyle='--', linewidth=2,
                    label=f'真の値: {true_mu}')
    axes[1].set_xlabel('データサイズ n')
    axes[1].set_ylabel('μの推定値')
    axes[1].set_title('データサイズの影響（事前分布から真の値への収束）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    

## 4.3 Markov Chain Monte Carlo (MCMC)

複雑な事後分布は解析的に計算できないことが多く、サンプリング手法が必要です。 MCMCはマルコフ連鎖を用いて事後分布からサンプルを生成する手法です。 

#### 📘 Metropolis-Hastings法

任意の確率分布 \\( p(\theta) \\) からサンプリングする汎用的アルゴリズム：

  1. 現在の状態 \\( \theta^{(t)} \\) から提案分布 \\( q(\theta^* | \theta^{(t)}) \\) で候補 \\( \theta^* \\) を生成
  2. 受理確率を計算： 

$$ \alpha = \min\left(1, \frac{p(\theta^*) q(\theta^{(t)} | \theta^*)}{p(\theta^{(t)}) q(\theta^* | \theta^{(t)})}\right) $$ 

  3. 確率 \\( \alpha \\) で候補を受理（\\( \theta^{(t+1)} = \theta^* \\)）、さもなくば棄却（\\( \theta^{(t+1)} = \theta^{(t)} \\)）

対称提案分布（\\( q(\theta^* | \theta) = q(\theta | \theta^*) \\)）の場合、受理確率は：

$$ \alpha = \min\left(1, \frac{p(\theta^*)}{p(\theta^{(t)})}\right) $$ 

#### 💻 コード例4: Metropolis-Hastings法の実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # ターゲット分布: 2つの正規分布の混合（解析的に扱いにくい例）
    def target_distribution(theta):
        """混合正規分布 0.3*N(0,1) + 0.7*N(5,1.5)"""
        component1 = 0.3 * stats.norm.pdf(theta, 0, 1)
        component2 = 0.7 * stats.norm.pdf(theta, 5, 1.5)
        return component1 + component2
    
    # Metropolis-Hastingsアルゴリズム
    def metropolis_hastings(target_func, n_samples, proposal_std=1.0, initial=0):
        samples = np.zeros(n_samples)
        samples[0] = initial
        n_accept = 0
        
        for t in range(1, n_samples):
            current = samples[t-1]
            
            # 提案（対称正規分布）
            proposal = current + np.random.normal(0, proposal_std)
            
            # 受理確率
            p_current = target_func(current)
            p_proposal = target_func(proposal)
            alpha = min(1, p_proposal / p_current) if p_current > 0 else 1
            
            # 受理・棄却の判定
            if np.random.rand() < alpha:
                samples[t] = proposal
                n_accept += 1
            else:
                samples[t] = current
        
        acceptance_rate = n_accept / (n_samples - 1)
        return samples, acceptance_rate
    
    # MCMCの実行
    np.random.seed(42)
    n_samples = 10000
    samples, acc_rate = metropolis_hastings(target_distribution, n_samples,
                                            proposal_std=2.0, initial=0)
    
    print("=== Metropolis-Hastings法 ===")
    print(f"サンプル数: {n_samples}")
    print(f"受理率: {acc_rate:.3f}")
    print(f"バーンイン後の平均: {np.mean(samples[1000:]):.4f}")
    print(f"バーンイン後の標準偏差: {np.std(samples[1000:]):.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # トレースプロット
    axes[0, 0].plot(samples[:500], 'b-', linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].axhline(5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel('イテレーション')
    axes[0, 0].set_ylabel('θ')
    axes[0, 0].set_title(f'トレースプロット（最初の500サンプル）\n受理率: {acc_rate:.2%}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ヒストグラムと真の分布
    burnin = 1000
    x = np.linspace(-5, 10, 300)
    true_pdf = target_distribution(x)
    
    axes[0, 1].hist(samples[burnin:], bins=60, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black', label='MCMCサンプル')
    axes[0, 1].plot(x, true_pdf, 'r-', linewidth=2, label='真の分布')
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_ylabel('確率密度')
    axes[0, 1].set_title(f'事後分布の推定（バーンイン: {burnin}サンプル）')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 自己相関
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(samples[burnin:], lags=50, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('自己相関関数（収束と独立性の確認）')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 提案分布の標準偏差の影響
    proposal_stds = [0.1, 0.5, 1.0, 2.0, 5.0]
    acceptance_rates = []
    
    for pstd in proposal_stds:
        _, ar = metropolis_hastings(target_distribution, 5000,
                                    proposal_std=pstd, initial=0)
        acceptance_rates.append(ar)
    
    axes[1, 1].plot(proposal_stds, acceptance_rates, 'bo-',
                    markersize=8, linewidth=2)
    axes[1, 1].axhline(0.234, color='red', linestyle='--', linewidth=2,
                       label='理想的受理率 (≈23.4%)')
    axes[1, 1].set_xlabel('提案分布の標準偏差')
    axes[1, 1].set_ylabel('受理率')
    axes[1, 1].set_title('提案分布のチューニング')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    

**📌 MCMCの実践的ポイント**  

  * **バーンイン** : 初期値の影響を除くため、最初の数千サンプルを捨てる
  * **受理率** : 20～40%が目安（低すぎると収束遅い、高すぎると探索不十分）
  * **自己相関** : サンプル間の独立性を確認（高い場合は間引く）
  * **収束診断** : 複数チェーン、Gelman-Rubin統計量、トレースプロット

#### 💻 コード例5: Gibbs Samplingによるベイズ推論
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # 2変数の同時分布からのサンプリング
    # 例: 正規分布の平均μと精度τ（分散の逆数）を同時推定
    
    # データ生成
    np.random.seed(42)
    true_mu = 50
    true_sigma = 10
    n = 30
    data = np.random.normal(true_mu, true_sigma, n)
    
    # 事前分布のパラメータ
    # μ ~ N(μ0, (λ0*τ)^{-1})
    mu_0 = 45
    lambda_0 = 0.1
    
    # τ ~ Gamma(α0, β0)
    alpha_0 = 2
    beta_0 = 20
    
    # Gibbs Sampling
    def gibbs_sampling_normal(data, n_iter=5000, burnin=1000):
        n = len(data)
        data_mean = np.mean(data)
        data_sum_sq = np.sum((data - data_mean)**2)
        
        # 初期値
        mu = data_mean
        tau = 1 / np.var(data)
        
        # サンプル保存
        mu_samples = np.zeros(n_iter)
        tau_samples = np.zeros(n_iter)
        
        for i in range(n_iter):
            # μの条件付き事後分布からサンプリング
            lambda_n = lambda_0 + n * tau
            mu_n = (lambda_0 * mu_0 + n * tau * data_mean) / lambda_n
            mu = np.random.normal(mu_n, 1/np.sqrt(lambda_n))
            
            # τの条件付き事後分布からサンプリング
            alpha_n = alpha_0 + n/2
            beta_n = beta_0 + 0.5 * (np.sum((data - mu)**2) + lambda_0 * (mu - mu_0)**2)
            tau = np.random.gamma(alpha_n, 1/beta_n)
            
            mu_samples[i] = mu
            tau_samples[i] = tau
        
        return mu_samples[burnin:], tau_samples[burnin:]
    
    # Gibbs Samplingの実行
    mu_samples, tau_samples = gibbs_sampling_normal(data, n_iter=10000, burnin=2000)
    sigma_samples = 1 / np.sqrt(tau_samples)
    
    print("=== Gibbs Sampling: 正規分布のパラメータ推定 ===")
    print(f"データ: n={n}, 標本平均={np.mean(data):.2f}, 標本SD={np.std(data):.2f}")
    print(f"真の値: μ={true_mu}, σ={true_sigma}")
    print(f"\n事後推定（平均）:")
    print(f"  μ: {np.mean(mu_samples):.2f} (95%CI: [{np.percentile(mu_samples, 2.5):.2f}, {np.percentile(mu_samples, 97.5):.2f}])")
    print(f"  σ: {np.mean(sigma_samples):.2f} (95%CI: [{np.percentile(sigma_samples, 2.5):.2f}, {np.percentile(sigma_samples, 97.5):.2f}])")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # μのトレースプロット
    axes[0, 0].plot(mu_samples[:1000], 'b-', linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(true_mu, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('イテレーション')
    axes[0, 0].set_ylabel('μ')
    axes[0, 0].set_title('μのトレースプロット')
    axes[0, 0].grid(True, alpha=0.3)
    
    # μの事後分布
    axes[0, 1].hist(mu_samples, bins=50, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black')
    axes[0, 1].axvline(true_mu, color='red', linestyle='--',
                       linewidth=2, label=f'真の値: {true_mu}')
    axes[0, 1].axvline(np.mean(mu_samples), color='blue', linestyle='-',
                       linewidth=2, label=f'事後平均: {np.mean(mu_samples):.1f}')
    axes[0, 1].set_xlabel('μ')
    axes[0, 1].set_ylabel('確率密度')
    axes[0, 1].set_title('μの事後分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # σのトレースプロット
    axes[0, 2].plot(sigma_samples[:1000], 'g-', linewidth=0.5, alpha=0.7)
    axes[0, 2].axhline(true_sigma, color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('イテレーション')
    axes[0, 2].set_ylabel('σ')
    axes[0, 2].set_title('σのトレースプロット')
    axes[0, 2].grid(True, alpha=0.3)
    
    # σの事後分布
    axes[1, 0].hist(sigma_samples, bins=50, density=True, alpha=0.7,
                    color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(true_sigma, color='red', linestyle='--',
                       linewidth=2, label=f'真の値: {true_sigma}')
    axes[1, 0].axvline(np.mean(sigma_samples), color='green', linestyle='-',
                       linewidth=2, label=f'事後平均: {np.mean(sigma_samples):.1f}')
    axes[1, 0].set_xlabel('σ')
    axes[1, 0].set_ylabel('確率密度')
    axes[1, 0].set_title('σの事後分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 同時事後分布
    axes[1, 1].hexbin(mu_samples, sigma_samples, gridsize=50, cmap='Blues')
    axes[1, 1].plot(true_mu, true_sigma, 'r*', markersize=15, label='真の値')
    axes[1, 1].set_xlabel('μ')
    axes[1, 1].set_ylabel('σ')
    axes[1, 1].set_title('μとσの同時事後分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 事後予測分布
    n_pred = 1000
    pred_samples = np.random.normal(mu_samples[:n_pred], sigma_samples[:n_pred])
    
    axes[1, 2].hist(data, bins=15, density=True, alpha=0.5,
                    color='orange', edgecolor='black', label='観測データ')
    axes[1, 2].hist(pred_samples, bins=50, density=True, alpha=0.5,
                    color='skyblue', edgecolor='black', label='事後予測分布')
    axes[1, 2].set_xlabel('値')
    axes[1, 2].set_ylabel('密度')
    axes[1, 2].set_title('事後予測分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 4.4 PyMC3によるベイズ推論

#### 💻 コード例6: PyMC3を用いたベイズ推論
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pymc3 as pm
    import arviz as az
    
    # データ生成
    np.random.seed(42)
    true_alpha = 2.5
    true_beta = 1.5
    n = 50
    x = np.linspace(0, 10, n)
    y_true = true_alpha + true_beta * x
    y = y_true + np.random.normal(0, 2, n)
    
    print("=== PyMC3によるベイズ線形回帰 ===")
    print(f"データ数: {n}")
    print(f"真のパラメータ: α={true_alpha}, β={true_beta}")
    
    # PyMC3モデルの定義
    with pm.Model() as model:
        # 事前分布
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=5)
        
        # 尤度
        mu = alpha + beta * x
        y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
        
        # サンプリング
        trace = pm.sample(2000, tune=1000, return_inferencedata=True,
                          random_seed=42, progressbar=False)
    
    # 結果のサマリー
    print("\n事後分布のサマリー:")
    print(az.summary(trace, var_names=['alpha', 'beta', 'sigma']))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # データと回帰直線
    axes[0, 0].scatter(x, y, alpha=0.5, color='blue', s=50, label='データ')
    axes[0, 0].plot(x, y_true, 'r-', linewidth=2, label=f'真の直線')
    
    # 事後分布からの回帰直線サンプル
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()
    
    for i in np.random.choice(len(alpha_samples), 100):
        y_pred = alpha_samples[i] + beta_samples[i] * x
        axes[0, 0].plot(x, y_pred, 'gray', alpha=0.05)
    
    # 事後平均の回帰直線
    y_mean = np.mean(alpha_samples) + np.mean(beta_samples) * x
    axes[0, 0].plot(x, y_mean, 'g--', linewidth=2,
                    label=f'事後平均直線')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('ベイズ線形回帰')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 事後分布（α, β, σ）
    az.plot_posterior(trace, var_names=['alpha', 'beta', 'sigma'],
                      ref_val=[true_alpha, true_beta, 2], ax=axes[0, 1])
    axes[0, 1].set_title('パラメータの事後分布')
    
    # トレースプロット
    az.plot_trace(trace, var_names=['alpha', 'beta', 'sigma'],
                  compact=False)
    plt.suptitle('トレースプロット（収束診断）', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # 事後予測チェック
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model), ax=ax)
    ax.set_title('事後予測チェック（モデルの適合度確認）')
    plt.tight_layout()
    plt.show()
    
    print("\nGelman-Rubin統計量（収束診断、1.0に近いほど良い）:")
    print(az.rhat(trace))
    

## 4.5 材料特性のベイズ推定

#### 💻 コード例7: 材料強度データのベイズ推定と予測
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pymc3 as pm
    import arviz as az
    from scipy import stats
    
    # 材料強度データ（少数サンプル）
    np.random.seed(42)
    n_samples = 15
    true_mean = 480
    true_std = 25
    strength_data = np.random.normal(true_mean, true_std, n_samples)
    
    print("=== 材料強度のベイズ推定 ===")
    print(f"データ数: {n_samples}（少数サンプル）")
    print(f"標本平均: {np.mean(strength_data):.2f} MPa")
    print(f"標本SD: {np.std(strength_data, ddof=1):.2f} MPa")
    
    # PyMC3モデル
    with pm.Model() as strength_model:
        # 事前分布（過去の経験から）
        mu = pm.Normal('mu', mu=500, sd=50)  # 過去のデータから500±50 MPa程度
        sigma = pm.HalfNormal('sigma', sd=30)  # ばらつきは最大30 MPa程度
        
        # 尤度
        y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=strength_data)
        
        # サンプリング
        trace = pm.sample(3000, tune=1500, return_inferencedata=True,
                          random_seed=42, progressbar=False)
    
    # 結果
    mu_samples = trace.posterior['mu'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()
    
    mu_mean = np.mean(mu_samples)
    mu_ci = np.percentile(mu_samples, [2.5, 97.5])
    sigma_mean = np.mean(sigma_samples)
    sigma_ci = np.percentile(sigma_samples, [2.5, 97.5])
    
    print(f"\nベイズ推定結果:")
    print(f"  平均強度μ: {mu_mean:.2f} MPa (95%CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}])")
    print(f"  標準偏差σ: {sigma_mean:.2f} MPa (95%CI: [{sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}])")
    
    # 頻度論的推定との比較
    freq_mean = np.mean(strength_data)
    freq_std = np.std(strength_data, ddof=1)
    freq_ci = stats.t.interval(0.95, n_samples-1,
                                loc=freq_mean,
                                scale=freq_std/np.sqrt(n_samples))
    
    print(f"\n頻度論的推定（参考）:")
    print(f"  平均強度: {freq_mean:.2f} MPa (95%CI: [{freq_ci[0]:.2f}, {freq_ci[1]:.2f}])")
    print(f"  標準偏差: {freq_std:.2f} MPa")
    
    # 事後予測分布（次のサンプルの強度予測）
    n_pred = 5000
    idx_random = np.random.choice(len(mu_samples), n_pred)
    predicted_strength = np.random.normal(mu_samples[idx_random],
                                          sigma_samples[idx_random])
    
    pred_mean = np.mean(predicted_strength)
    pred_ci = np.percentile(predicted_strength, [2.5, 97.5])
    
    print(f"\n次のサンプルの強度予測（事後予測分布）:")
    print(f"  予測平均: {pred_mean:.2f} MPa")
    print(f"  95%予測区間: [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}] MPa")
    
    # 破壊確率の推定
    design_strength = 450  # 設計基準強度
    prob_failure = np.mean(predicted_strength < design_strength)
    print(f"\n設計基準強度{design_strength} MPa以下の確率: {prob_failure:.4f} ({prob_failure*100:.2f}%)")
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 事後分布（μ, σ）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(mu_samples, bins=50, density=True, alpha=0.7,
             color='skyblue', edgecolor='black')
    ax1.axvline(true_mean, color='red', linestyle='--',
                linewidth=2, label=f'真の値: {true_mean}')
    ax1.axvline(mu_mean, color='blue', linestyle='-',
                linewidth=2, label=f'事後平均: {mu_mean:.1f}')
    ax1.axvspan(mu_ci[0], mu_ci[1], alpha=0.2, color='blue',
                label='95%信用区間')
    ax1.set_xlabel('平均強度μ [MPa]')
    ax1.set_ylabel('確率密度')
    ax1.set_title('平均強度の事後分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(sigma_samples, bins=50, density=True, alpha=0.7,
             color='lightgreen', edgecolor='black')
    ax2.axvline(true_std, color='red', linestyle='--',
                linewidth=2, label=f'真の値: {true_std}')
    ax2.axvline(sigma_mean, color='green', linestyle='-',
                linewidth=2, label=f'事後平均: {sigma_mean:.1f}')
    ax2.set_xlabel('標準偏差σ [MPa]')
    ax2.set_ylabel('確率密度')
    ax2.set_title('標準偏差の事後分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 同時事後分布
    ax3 = fig.add_subplot(gs[1, :])
    ax3.hexbin(mu_samples, sigma_samples, gridsize=50, cmap='Blues')
    ax3.plot(true_mean, true_std, 'r*', markersize=20, label='真の値')
    ax3.set_xlabel('μ [MPa]')
    ax3.set_ylabel('σ [MPa]')
    ax3.set_title('μとσの同時事後分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 事後予測分布と観測データ
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(predicted_strength, bins=60, density=True, alpha=0.6,
             color='lightblue', edgecolor='black', label='事後予測分布')
    ax4.hist(strength_data, bins=10, density=True, alpha=0.6,
             color='orange', edgecolor='black', label='観測データ')
    ax4.axvline(design_strength, color='red', linestyle='--',
                linewidth=2, label=f'設計基準: {design_strength} MPa')
    ax4.axvspan(pred_ci[0], pred_ci[1], alpha=0.2, color='green',
                label='95%予測区間')
    ax4.set_xlabel('強度 [MPa]')
    ax4.set_ylabel('確率密度')
    ax4.set_title('事後予測分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 破壊確率の可視化
    ax5 = fig.add_subplot(gs[2, 1])
    threshold_range = np.linspace(400, 550, 100)
    failure_probs = []
    
    for threshold in threshold_range:
        prob = np.mean(predicted_strength < threshold)
        failure_probs.append(prob)
    
    ax5.plot(threshold_range, failure_probs, 'b-', linewidth=2)
    ax5.axvline(design_strength, color='red', linestyle='--',
                linewidth=2, label=f'設計基準: {design_strength} MPa')
    ax5.axhline(prob_failure, color='orange', linestyle=':',
                linewidth=2, label=f'破壊確率: {prob_failure:.3f}')
    ax5.fill_between(threshold_range, 0, failure_probs,
                      where=(np.array(threshold_range) <= design_strength),
                      alpha=0.3, color='red')
    ax5.set_xlabel('強度閾値 [MPa]')
    ax5.set_ylabel('破壊確率')
    ax5.set_title('強度閾値と破壊確率の関係')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'材料強度のベイズ推定と予測（n={n_samples}）', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 実践的まとめ
    print("\n=== 実践的解釈 ===")
    print(f"✓ 少数サンプル（n={n_samples}）でも事前知識を活用して安定した推定")
    print(f"✓ 平均強度は95%の確率で{mu_ci[0]:.1f}～{mu_ci[1]:.1f} MPaの範囲")
    print(f"✓ 次のサンプルは95%の確率で{pred_ci[0]:.1f}～{pred_ci[1]:.1f} MPa")
    print(f"✓ 設計基準{design_strength} MPa以下で破壊する確率は{prob_failure*100:.2f}%")
    

#### 📝 練習問題

  1. ベイズ推論と頻度論的推論の違いを、確率の解釈の観点から説明してください。
  2. 事前分布が無情報（一様分布）の場合、事後分布はどうなるか議論してください。
  3. Metropolis-Hastings法で、受理率が極端に高い（>90%）場合と低い（<10%）場合の問題点を説明してください。
  4. PyMC3を用いて、ポアソン分布のλパラメータをベイズ推定するコードを書いてください。

## まとめ

  * ベイズ推論は事前知識とデータを統合して不確実性を定量化する
  * 事後分布は尤度と事前分布の積に比例する
  * 共役事前分布を使うと解析的に事後分布を計算できる
  * MCMCは複雑な事後分布からサンプリングする強力な手法である
  * Metropolis-Hastings法とGibbs Samplingが代表的なMCMCアルゴリズム
  * PyMC3を使えば複雑なベイズモデルを簡潔に記述・推論できる
  * 材料科学では少数サンプルでも事前知識を活用して信頼性の高い推定が可能
  * 事後予測分布は新しいデータの予測と不確実性評価に有用

[← 第3章: 仮説検定と検定力分析](<chapter-3.html>) [第5章: 階層ベイズモデルと応用 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
