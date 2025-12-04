---
title: "第1章: 確率変数と確率分布の基礎"
chapter_title: "第1章: 確率変数と確率分布の基礎"
subtitle: Random Variables and Probability Distributions
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/probability-stochastic-processes/chapter-1.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [確率論と確率過程](<index.html>) > 第1章 

## 1.1 離散確率変数と確率質量関数

**📐 定義: 離散確率変数**  
離散確率変数 \\(X\\) は、可算個の値 \\(\\{x_1, x_2, \ldots\\}\\) をとる確率変数です。  
確率質量関数（PMF: Probability Mass Function）は： \\[P(X = x_i) = p_i, \quad \sum_{i} p_i = 1\\] 各値の生起確率を表します。 

### 💻 コード例1: 離散確率分布の可視化

import numpy as np import matplotlib.pyplot as plt from scipy import stats # サイコロの例（一様離散分布） outcomes = np.arange(1, 7) # 1から6 probabilities = np.ones(6) / 6 # 各面の確率は1/6 fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # PMFの棒グラフ axes[0].bar(outcomes, probabilities, color='#667eea', alpha=0.7, edgecolor='black') axes[0].set_xlabel('結果 (x)', fontsize=12) axes[0].set_ylabel('確率 P(X=x)', fontsize=12) axes[0].set_title('サイコロの確率質量関数（PMF）', fontsize=14, fontweight='bold') axes[0].set_xticks(outcomes) axes[0].grid(axis='y', alpha=0.3) # 累積分布関数（CDF） cdf = np.cumsum(probabilities) axes[1].step(outcomes, cdf, where='post', color='#764ba2', linewidth=2) axes[1].scatter(outcomes, cdf, color='#764ba2', s=50, zorder=5) axes[1].set_xlabel('結果 (x)', fontsize=12) axes[1].set_ylabel('累積確率 P(X≤x)', fontsize=12) axes[1].set_title('累積分布関数（CDF）', fontsize=14, fontweight='bold') axes[1].set_xticks(outcomes) axes[1].grid(alpha=0.3) axes[1].set_ylim([0, 1.1]) plt.tight_layout() plt.show() # 期待値と分散の計算 E_X = np.sum(outcomes * probabilities) E_X2 = np.sum(outcomes**2 * probabilities) Var_X = E_X2 - E_X**2 print("サイコロの統計量:") print(f"期待値 E[X] = {E_X:.4f}") print(f"分散 Var(X) = {Var_X:.4f}") print(f"標準偏差 σ = {np.sqrt(Var_X):.4f}")

サイコロの統計量: 期待値 E[X] = 3.5000 分散 Var(X) = 2.9167 標準偏差 σ = 1.7078

## 1.2 連続確率変数と確率密度関数

**📐 定義: 連続確率変数**  
連続確率変数 \\(X\\) は実数値を取り、確率密度関数（PDF: Probability Density Function）\\(f(x)\\) によって特徴付けられます： \\[P(a \leq X \leq b) = \int_a^b f(x) \, dx, \quad \int_{-\infty}^{\infty} f(x) \, dx = 1\\] 累積分布関数（CDF）は： \\[F(x) = P(X \leq x) = \int_{-\infty}^x f(t) \, dt\\] 

### 💻 コード例2: 連続確率分布とPDF/CDF

# 一様分布 U(0, 1)の例 x = np.linspace(-0.5, 1.5, 1000) uniform_dist = stats.uniform(loc=0, scale=1) pdf = uniform_dist.pdf(x) cdf = uniform_dist.cdf(x) fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # PDF axes[0].plot(x, pdf, color='#667eea', linewidth=2.5, label='PDF') axes[0].fill_between(x, 0, pdf, where=(x >= 0.3) & (x <= 0.7), alpha=0.3, color='#764ba2', label='P(0.3 ≤ X ≤ 0.7)') axes[0].set_xlabel('x', fontsize=12) axes[0].set_ylabel('f(x)', fontsize=12) axes[0].set_title('一様分布 U(0,1) の確率密度関数', fontsize=14, fontweight='bold') axes[0].legend(fontsize=10) axes[0].grid(alpha=0.3) axes[0].set_ylim([0, 1.5]) # CDF axes[1].plot(x, cdf, color='#764ba2', linewidth=2.5) axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='P(X≤0.7) = 0.7') axes[1].axvline(x=0.7, color='red', linestyle='--', alpha=0.5) axes[1].set_xlabel('x', fontsize=12) axes[1].set_ylabel('F(x)', fontsize=12) axes[1].set_title('累積分布関数（CDF）', fontsize=14, fontweight='bold') axes[1].legend(fontsize=10) axes[1].grid(alpha=0.3) plt.tight_layout() plt.show() # 確率計算 prob = uniform_dist.cdf(0.7) - uniform_dist.cdf(0.3) print(f"P(0.3 ≤ X ≤ 0.7) = {prob:.4f}")

P(0.3 ≤ X ≤ 0.7) = 0.4000

## 1.3 期待値・分散・モーメント

**📊 定理: 期待値と分散**  
**期待値（平均）:** \\[E[X] = \sum_i x_i p_i \quad (\text{離散}), \quad E[X] = \int_{-\infty}^{\infty} x f(x) \, dx \quad (\text{連続})\\] **分散:** \\[Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2\\] **k次モーメント:** \\[E[X^k] = \sum_i x_i^k p_i \quad (\text{離散}), \quad E[X^k] = \int_{-\infty}^{\infty} x^k f(x) \, dx \quad (\text{連続})\\] 

### 💻 コード例3: 期待値と分散の計算

# 複数の分布の期待値・分散を計算 distributions = { '一様分布 U(0,1)': stats.uniform(0, 1), '正規分布 N(5,2²)': stats.norm(5, 2), '指数分布 Exp(λ=1)': stats.expon(scale=1), 'ベータ分布 Beta(2,5)': stats.beta(2, 5) } print("各分布の期待値と分散:\n" + "="*50) for name, dist in distributions.items(): mean = dist.mean() var = dist.var() std = dist.std() # k次モーメント（k=3,4）の計算 if hasattr(dist, 'moment'): third_moment = dist.moment(3) fourth_moment = dist.moment(4) else: third_moment = np.nan fourth_moment = np.nan print(f"\n{name}:") print(f" 期待値 E[X] = {mean:.4f}") print(f" 分散 Var(X) = {var:.4f}") print(f" 標準偏差 σ = {std:.4f}") # 可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() x_ranges = [ np.linspace(0, 1, 500), np.linspace(-1, 11, 500), np.linspace(0, 6, 500), np.linspace(0, 1, 500) ] for i, ((name, dist), x_range) in enumerate(zip(distributions.items(), x_ranges)): pdf = dist.pdf(x_range) mean = dist.mean() std = dist.std() axes[i].plot(x_range, pdf, color='#667eea', linewidth=2.5, label='PDF') axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'E[X]={mean:.2f}') axes[i].axvline(mean - std, color='orange', linestyle=':', alpha=0.7, label=f'σ={std:.2f}') axes[i].axvline(mean + std, color='orange', linestyle=':', alpha=0.7) axes[i].fill_between(x_range, 0, pdf, alpha=0.2, color='#764ba2') axes[i].set_xlabel('x', fontsize=11) axes[i].set_ylabel('f(x)', fontsize=11) axes[i].set_title(name, fontsize=12, fontweight='bold') axes[i].legend(fontsize=9) axes[i].grid(alpha=0.3) plt.tight_layout() plt.show()

## 1.4 二項分布とポアソン分布

**📐 定義: 二項分布とポアソン分布**  
**二項分布 \\(B(n, p)\\):** n回の独立試行で成功確率pのとき、成功回数Xの分布 \\[P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad E[X] = np, \quad Var(X) = np(1-p)\\] **ポアソン分布 \\(Pois(\lambda)\\):** 単位時間あたりの平均発生回数λのとき \\[P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad E[X] = Var(X) = \lambda\\] ポアソン分布は二項分布の極限（\\(n \to \infty, p \to 0, np = \lambda\\)）として得られます。 

### 💻 コード例4: 二項分布とポアソン分布の比較

# 二項分布とポアソン分布の比較 n = 100 # 試行回数 p = 0.03 # 成功確率 lam = n * p # λ = np = 3 binomial = stats.binom(n, p) poisson = stats.poisson(lam) k = np.arange(0, 15) fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # PMFの比較 axes[0].bar(k - 0.2, binomial.pmf(k), width=0.4, alpha=0.7, color='#667eea', label=f'二項分布 B({n},{p})') axes[0].bar(k + 0.2, poisson.pmf(k), width=0.4, alpha=0.7, color='#764ba2', label=f'ポアソン分布 Pois({lam})') axes[0].set_xlabel('k (成功回数)', fontsize=12) axes[0].set_ylabel('確率 P(X=k)', fontsize=12) axes[0].set_title('二項分布とポアソン分布の比較', fontsize=14, fontweight='bold') axes[0].legend(fontsize=10) axes[0].grid(axis='y', alpha=0.3) # 誤差の可視化 error = np.abs(binomial.pmf(k) - poisson.pmf(k)) axes[1].bar(k, error, color='red', alpha=0.6) axes[1].set_xlabel('k', fontsize=12) axes[1].set_ylabel('|P_binomial - P_poisson|', fontsize=12) axes[1].set_title('近似誤差', fontsize=14, fontweight='bold') axes[1].grid(axis='y', alpha=0.3) plt.tight_layout() plt.show() # 統計量の比較 print("統計量の比較:") print(f"二項分布: E[X]={binomial.mean():.4f}, Var(X)={binomial.var():.4f}") print(f"ポアソン分布: E[X]={poisson.mean():.4f}, Var(X)={poisson.var():.4f}") # 材料科学への応用例：欠陥の発生確率 print("\n【材料科学への応用例】") print("薄膜成膜プロセスで、1cm²あたり平均3個の欠陥が発生するとき：") area = 5 # cm² lam_defects = 3 * area poisson_defects = stats.poisson(lam_defects) print(f"5cm²領域で10個以下の欠陥が発生する確率: {poisson_defects.cdf(10):.4f}") print(f"5cm²領域で15個以上の欠陥が発生する確率: {1 - poisson_defects.cdf(14):.4f}")

統計量の比較: 二項分布: E[X]=3.0000, Var(X)=2.9100 ポアソン分布: E[X]=3.0000, Var(X)=3.0000 【材料科学への応用例】 薄膜成膜プロセスで、1cm²あたり平均3個の欠陥が発生するとき： 5cm²領域で10個以下の欠陥が発生する確率: 0.1185 5cm²領域で15個以上の欠陥が発生する確率: 0.5289

## 1.5 正規分布（ガウス分布）

**📐 定義: 正規分布**  
正規分布 \\(N(\mu, \sigma^2)\\) の確率密度関数： \\[f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\\] 期待値 \\(E[X] = \mu\\), 分散 \\(Var(X) = \sigma^2\\)  
**標準正規分布:** \\(N(0, 1)\\) は \\(\mu=0, \sigma=1\\) の正規分布  
任意の正規分布は標準化変換 \\(Z = \frac{X - \mu}{\sigma}\\) で標準正規分布に変換できます。 

### 💻 コード例5: 正規分布の性質と応用

# 正規分布の可視化 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) 異なるμの効果 x = np.linspace(-10, 10, 1000) mus = [0, 2, -2] sigma = 1 axes[0, 0].set_title('平均μの効果（σ=1固定）', fontsize=12, fontweight='bold') for mu in mus: dist = stats.norm(mu, sigma) axes[0, 0].plot(x, dist.pdf(x), linewidth=2, label=f'μ={mu}') axes[0, 0].set_xlabel('x', fontsize=11) axes[0, 0].set_ylabel('f(x)', fontsize=11) axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) # (2) 異なるσの効果 mu = 0 sigmas = [0.5, 1, 2] axes[0, 1].set_title('標準偏差σの効果（μ=0固定）', fontsize=12, fontweight='bold') for sigma in sigmas: dist = stats.norm(mu, sigma) axes[0, 1].plot(x, dist.pdf(x), linewidth=2, label=f'σ={sigma}') axes[0, 1].set_xlabel('x', fontsize=11) axes[0, 1].set_ylabel('f(x)', fontsize=11) axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 68-95-99.7ルール（3σルール） mu, sigma = 0, 1 dist = stats.norm(mu, sigma) x_range = np.linspace(-4, 4, 1000) axes[1, 0].plot(x_range, dist.pdf(x_range), 'black', linewidth=2) axes[1, 0].fill_between(x_range, 0, dist.pdf(x_range), where=(x_range >= -sigma) & (x_range <= sigma), alpha=0.3, color='green', label='68% (±1σ)') axes[1, 0].fill_between(x_range, 0, dist.pdf(x_range), where=((x_range >= -2*sigma) & (x_range < -sigma)) | ((x_range > sigma) & (x_range <= 2*sigma)), alpha=0.3, color='yellow', label='95% (±2σ)') axes[1, 0].fill_between(x_range, 0, dist.pdf(x_range), where=((x_range >= -3*sigma) & (x_range < -2*sigma)) | ((x_range > 2*sigma) & (x_range <= 3*sigma)), alpha=0.3, color='orange', label='99.7% (±3σ)') axes[1, 0].set_title('68-95-99.7ルール（3σルール）', fontsize=12, fontweight='bold') axes[1, 0].set_xlabel('x', fontsize=11) axes[1, 0].set_ylabel('f(x)', fontsize=11) axes[1, 0].legend(fontsize=9) axes[1, 0].grid(alpha=0.3) # (4) 標準化とパーセンタイル standard_normal = stats.norm(0, 1) percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] z_values = [standard_normal.ppf(p) for p in percentiles] axes[1, 1].plot(x_range, standard_normal.pdf(x_range), 'black', linewidth=2) for p, z in zip(percentiles, z_values): axes[1, 1].axvline(z, color='red', linestyle='--', alpha=0.5) axes[1, 1].text(z, 0.4, f'{int(p*100)}%', fontsize=8, ha='center') axes[1, 1].set_title('標準正規分布のパーセンタイル', fontsize=12, fontweight='bold') axes[1, 1].set_xlabel('z', fontsize=11) axes[1, 1].set_ylabel('φ(z)', fontsize=11) axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show() # 確率計算 print("正規分布 N(100, 15²) における確率計算:") mu, sigma = 100, 15 exam_scores = stats.norm(mu, sigma) print(f"P(X ≥ 120) = {1 - exam_scores.cdf(120):.4f}") print(f"P(85 ≤ X ≤ 115) = {exam_scores.cdf(115) - exam_scores.cdf(85):.4f}") # 標準化 x_value = 120 z_score = (x_value - mu) / sigma print(f"\nX=120のz-score: z = {z_score:.4f}") print(f"P(Z ≥ {z_score:.4f}) = {1 - standard_normal.cdf(z_score):.4f}")

## 1.6 指数分布と待ち時間問題

**📐 定義: 指数分布**  
指数分布 \\(Exp(\lambda)\\) は、ポアソン過程における事象間の待ち時間を表します： \\[f(x) = \lambda e^{-\lambda x} \quad (x \geq 0), \quad F(x) = 1 - e^{-\lambda x}\\] 期待値 \\(E[X] = \frac{1}{\lambda}\\), 分散 \\(Var(X) = \frac{1}{\lambda^2}\\)  
**無記憶性:** \\(P(X > s+t \mid X > s) = P(X > t)\\)（過去の情報が未来に影響しない） 

### 💻 コード例6: 指数分布と待ち時間問題

# 指数分布のシミュレーション lambda_rate = 0.5 # 平均到着率（1時間あたり0.5回 = 2時間に1回） exp_dist = stats.expon(scale=1/lambda_rate) # PDF, CDFの可視化 x = np.linspace(0, 10, 1000) pdf = exp_dist.pdf(x) cdf = exp_dist.cdf(x) fig, axes = plt.subplots(1, 3, figsize=(16, 5)) # (1) PDFとCDF axes[0].plot(x, pdf, color='#667eea', linewidth=2.5, label='PDF') axes[0].fill_between(x, 0, pdf, alpha=0.2, color='#667eea') axes[0].set_xlabel('時間 (時間)', fontsize=11) axes[0].set_ylabel('f(x)', fontsize=11) axes[0].set_title('指数分布の確率密度関数', fontsize=12, fontweight='bold') axes[0].grid(alpha=0.3) axes[0].legend() axes[1].plot(x, cdf, color='#764ba2', linewidth=2.5, label='CDF') axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='中央値') median = exp_dist.median() axes[1].axvline(x=median, color='red', linestyle='--', alpha=0.5) axes[1].set_xlabel('時間 (時間)', fontsize=11) axes[1].set_ylabel('F(x)', fontsize=11) axes[1].set_title('累積分布関数', fontsize=12, fontweight='bold') axes[1].grid(alpha=0.3) axes[1].legend() # (2) 無記憶性の検証 # P(X > 3+2 | X > 3) = P(X > 2)を確認 t = 2 s = 3 prob_uncond = 1 - exp_dist.cdf(t) # P(X > 2) prob_cond = (1 - exp_dist.cdf(s+t)) / (1 - exp_dist.cdf(s)) # P(X > 5 | X > 3) print("無記憶性の検証:") print(f"P(X > {t}) = {prob_uncond:.6f}") print(f"P(X > {s+t} | X > {s}) = {prob_cond:.6f}") print(f"差: {abs(prob_uncond - prob_cond):.10f}") # (3) シミュレーション np.random.seed(42) n_simulations = 10000 samples = exp_dist.rvs(size=n_simulations) axes[2].hist(samples, bins=50, density=True, alpha=0.6, color='#764ba2', edgecolor='black', label='シミュレーション') axes[2].plot(x, pdf, color='#667eea', linewidth=2.5, label='理論PDF') axes[2].set_xlabel('時間 (時間)', fontsize=11) axes[2].set_ylabel('密度', fontsize=11) axes[2].set_title(f'シミュレーション (n={n_simulations})', fontsize=12, fontweight='bold') axes[2].legend() axes[2].grid(alpha=0.3) plt.tight_layout() plt.show() # 統計量 print(f"\n理論値: E[X]={exp_dist.mean():.4f}, Var(X)={exp_dist.var():.4f}") print(f"実測値: E[X]={np.mean(samples):.4f}, Var(X)={np.var(samples):.4f}") # 材料科学への応用：装置故障間隔 print("\n【材料科学への応用例】") print("製造装置の故障間隔が指数分布Exp(λ=0.1 [1/日])に従うとき：") lambda_failure = 0.1 failure_dist = stats.expon(scale=1/lambda_failure) print(f"平均故障間隔（MTBF）: {failure_dist.mean():.2f} 日") print(f"7日以内に故障する確率: {failure_dist.cdf(7):.4f}") print(f"30日以上故障しない確率: {1 - failure_dist.cdf(30):.4f}")

無記憶性の検証: P(X > 2) = 0.367879 P(X > 5 | X > 3) = 0.367879 差: 0.0000000000 理論値: E[X]=2.0000, Var(X)=4.0000 実測値: E[X]=2.0124, Var(X)=4.0346 【材料科学への応用例】 製造装置の故障間隔が指数分布Exp(λ=0.1 [1/日])に従うとき： 平均故障間隔（MTBF）: 10.00 日 7日以内に故障する確率: 0.5034 30日以上故障しない確率: 0.0498

## 1.7 scipy.statsによる統計分布の活用

### 💻 コード例7: scipy.statsによる統計分布の活用

# scipy.statsで利用可能な様々な分布 from scipy.stats import ( bernoulli, binom, poisson, geom, # 離散分布 uniform, norm, expon, gamma, beta, weibull_min # 連続分布 ) # 統一的なインターフェース distributions = { 'ベルヌーイ': bernoulli(p=0.3), '二項': binom(n=10, p=0.3), 'ポアソン': poisson(mu=3), '幾何': geom(p=0.3), '一様': uniform(loc=0, scale=1), '正規': norm(loc=0, scale=1), '指数': expon(scale=2), 'ガンマ': gamma(a=2, scale=2), 'ベータ': beta(a=2, b=5), 'ワイブル': weibull_min(c=1.5, scale=1) } print("scipy.stats による統計分布の操作:\n" + "="*60) # 各分布の主要メソッド example_dist = norm(loc=5, scale=2) print("\n主要メソッド（正規分布 N(5, 2²) の例）:") print(f" .mean() 平均: {example_dist.mean():.4f}") print(f" .var() 分散: {example_dist.var():.4f}") print(f" .std() 標準偏差: {example_dist.std():.4f}") print(f" .pdf(6) 確率密度（x=6）: {example_dist.pdf(6):.6f}") print(f" .cdf(7) 累積確率（x≤7）: {example_dist.cdf(7):.6f}") print(f" .ppf(0.95) 95%点: {example_dist.ppf(0.95):.4f}") print(f" .rvs(5) 乱数生成（5個）: {example_dist.rvs(5)}") # 各分布の特性 print("\n\n各分布の統計量:") print(f"{'分布名':<12} {'平均':>10} {'分散':>10} {'歪度':>10} {'尖度':>10}") print("="*60) for name, dist in distributions.items(): try: mean = dist.mean() var = dist.var() skew = dist.stats(moments='s') kurt = dist.stats(moments='k') print(f"{name:<12} {mean:>10.4f} {var:>10.4f} {skew:>10.4f} {kurt:>10.4f}") except: print(f"{name:<12} (計算不可)") # 分位点の計算（信頼区間など） print("\n\n正規分布 N(0,1) の分位点:") alpha_levels = [0.90, 0.95, 0.99] standard_normal = norm(0, 1) for alpha in alpha_levels: lower = standard_normal.ppf((1-alpha)/2) upper = standard_normal.ppf((1+alpha)/2) print(f"{int(alpha*100)}% 信頼区間: [{lower:.4f}, {upper:.4f}]") # 適合度検定（Kolmogorov-Smirnov検定） from scipy.stats import kstest np.random.seed(42) sample_data = norm(loc=10, scale=2).rvs(1000) # 仮説: データは N(10, 2²) に従う ks_stat, p_value = kstest(sample_data, lambda x: norm(10, 2).cdf(x)) print(f"\n\nKolmogorov-Smirnov 適合度検定:") print(f"検定統計量: {ks_stat:.6f}") print(f"p値: {p_value:.6f}") print(f"結論: {'帰無仮説を棄却できない（分布は適合）' if p_value > 0.05 else '帰無仮説を棄却（分布は不適合）'}") # 材料科学への応用：材料強度のワイブル分布解析 print("\n\n【材料科学への応用例】セラミックスの強度分布（ワイブル分布）:") shape_param = 10 # 形状パラメータ（大きいほど均一） scale_param = 500 # 尺度パラメータ（特性強度） weibull = weibull_min(c=shape_param, scale=scale_param) print(f"ワイブル分布: shape={shape_param}, scale={scale_param} MPa") print(f"平均強度: {weibull.mean():.2f} MPa") print(f"標準偏差: {weibull.std():.2f} MPa") print(f"450 MPa以下で破壊する確率: {weibull.cdf(450):.4f}") print(f"設計強度（99%信頼性）: {weibull.ppf(0.01):.2f} MPa")

**💡 Note:** scipy.statsは70以上の確率分布を提供し、統一的なインターフェース（pdf, cdf, ppf, rvs, mean, var等）で操作できます。材料科学では、ワイブル分布（脆性材料の強度）、対数正規分布（粒径分布）、ガンマ分布（待ち時間）などがよく使われます。 

## 演習問題

**📝 演習1: 品質管理における統計分布**  
ある製品の不良率が2%であるとき、100個の製品ロットから： 

  1. 二項分布とポアソン分布で不良品数の分布をモデル化し、比較せよ
  2. 3個以上の不良品が含まれる確率を計算せよ
  3. 不良品数が期待値±1標準偏差の範囲内に入る確率を求めよ

**📝 演習2: 正規分布による測定誤差解析**  
測定装置の誤差が正規分布 N(0, 0.5²) に従うとき： 

  1. 誤差が±1 mm以内に収まる確率を計算せよ
  2. 95%信頼区間を求めよ
  3. 誤差の絶対値が0.8 mmを超える確率を計算せよ

**📝 演習3: 指数分布による故障解析**  
電子部品の寿命が指数分布 Exp(λ=0.001 [1/時間]) に従うとき： 

  1. 平均寿命（MTBF）を計算せよ
  2. 1000時間以上動作する確率を求めよ
  3. 無記憶性を利用して、既に500時間動作した部品が、さらに1000時間以上動作する確率を計算せよ

[← シリーズトップ](<index.html>) [第2章へ →](<chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
