---
title: "第4章: 確率微分方程式とWiener過程"
chapter_title: "第4章: 確率微分方程式とWiener過程"
subtitle: Stochastic Differential Equations and Wiener Process
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/Qihm4DwRnZ4"
    title="確率論と確率過程 第4章: 確率微分方程式とWiener過程"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

---

🌐 JP | [🇬🇧 EN](<../../../en/FM/probability-stochastic-processes/chapter-4.html>) | Last sync: 2025-11-16

[基礎数理道場](<../index.html>) > [確率論と確率過程](<index.html>) > 第4章 

## 4.1 Wiener過程（ブラウン運動）

**📐 定義: Wiener過程（標準ブラウン運動）**  
確率過程 \\(\\{W(t)\\}_{t \geq 0}\\) が**Wiener過程** であるとは： 

  1. \\(W(0) = 0\\) （ゼロから開始）
  2. 独立増分性: \\(W(t_1), W(t_2) - W(t_1), W(t_3) - W(t_2), \ldots\\) は独立
  3. 定常増分性: \\(W(t+s) - W(s) \sim N(0, t)\\)
  4. 連続な経路: \\(W(t)\\) は \\(t\\) について連続関数

**性質:** \\[E[W(t)] = 0, \quad Var(W(t)) = t, \quad Cov(W(s), W(t)) = \min(s, t)\\] 

### 💻 コード例1: Wiener過程（ブラウン運動）のシミュレーション

import numpy as np import matplotlib.pyplot as plt from scipy import stats # Wiener過程のシミュレーション np.random.seed(42) def simulate_wiener_process(T, dt): """Wiener過程のシミュレーション""" n_steps = int(T / dt) t = np.linspace(0, T, n_steps + 1) # 増分 dW ~ N(0, dt) dW = np.random.normal(0, np.sqrt(dt), n_steps) # 累積和でWiener過程を構築 W = np.concatenate([[0], np.cumsum(dW)]) return t, W # パラメータ T = 10 # 終了時刻 dt = 0.01 # 時間刻み # 複数経路のシミュレーション n_paths = 10 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) サンプル経路 for _ in range(n_paths): t, W = simulate_wiener_process(T, dt) axes[0, 0].plot(t, W, alpha=0.6, linewidth=1.5) axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='E[W(t)] = 0') axes[0, 0].set_xlabel('時間 t', fontsize=11) axes[0, 0].set_ylabel('W(t)', fontsize=11) axes[0, 0].set_title('Wiener過程のサンプル経路', fontsize=12, fontweight='bold') axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) # (2) 分散の時間発展 n_simulations = 1000 t_checkpoints = [1, 2, 5, 10] for t_check in t_checkpoints: W_values = [] for _ in range(n_simulations): t, W = simulate_wiener_process(t_check, dt) W_values.append(W[-1]) W_values = np.array(W_values) # ヒストグラム axes[0, 1].hist(W_values, bins=50, alpha=0.4, density=True, label=f't={t_check}') # 理論分布 N(0, t) x = np.linspace(-6, 6, 1000) for t_check in t_checkpoints: theoretical_pdf = stats.norm.pdf(x, 0, np.sqrt(t_check)) axes[0, 1].plot(x, theoretical_pdf, linewidth=2, linestyle='--') axes[0, 1].set_xlabel('W(t)', fontsize=11) axes[0, 1].set_ylabel('密度', fontsize=11) axes[0, 1].set_title('W(t)の分布（異なる時刻）', fontsize=12, fontweight='bold') axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 平均と分散の時間推移 t, W_sample = simulate_wiener_process(T, dt) n_paths_analysis = 500 all_paths = [] for _ in range(n_paths_analysis): t, W = simulate_wiener_process(T, dt) all_paths.append(W) all_paths = np.array(all_paths) mean_W = all_paths.mean(axis=0) std_W = all_paths.std(axis=0) axes[1, 0].plot(t, mean_W, color='#667eea', linewidth=2.5, label='実測平均') axes[1, 0].fill_between(t, mean_W - std_W, mean_W + std_W, alpha=0.3, color='#667eea', label='実測±1σ') axes[1, 0].plot(t, np.zeros_like(t), 'r--', linewidth=2, label='理論平均=0') axes[1, 0].plot(t, np.sqrt(t), 'g--', linewidth=2, label='理論σ=√t') axes[1, 0].plot(t, -np.sqrt(t), 'g--', linewidth=2) axes[1, 0].set_xlabel('時間 t', fontsize=11) axes[1, 0].set_ylabel('W(t)', fontsize=11) axes[1, 0].set_title('平均と分散の時間推移', fontsize=12, fontweight='bold') axes[1, 0].legend() axes[1, 0].grid(alpha=0.3) # (4) 共分散構造 Cov(W(s), W(t)) = min(s, t) s_values = np.linspace(0, T, 100) t_values = np.linspace(0, T, 100) S, T_grid = np.meshgrid(s_values, t_values) # 理論共分散 Cov_theory = np.minimum(S, T_grid) im = axes[1, 1].imshow(Cov_theory, extent=[0, T, 0, T], origin='lower', cmap='viridis', aspect='auto') axes[1, 1].set_xlabel('時刻 s', fontsize=11) axes[1, 1].set_ylabel('時刻 t', fontsize=11) axes[1, 1].set_title('Cov(W(s), W(t)) = min(s, t)', fontsize=12, fontweight='bold') plt.colorbar(im, ax=axes[1, 1]) plt.tight_layout() plt.show() print("Wiener過程の性質検証:") print("="*60) for t_check in t_checkpoints: W_values = [] for _ in range(n_simulations): t, W = simulate_wiener_process(t_check, dt) W_values.append(W[-1]) W_values = np.array(W_values) print(f"t={t_check}:") print(f" 理論: E[W(t)]={0:.4f}, Var(W(t))={t_check:.4f}") print(f" 実測: E[W(t)]={W_values.mean():.4f}, Var(W(t))={W_values.var():.4f}")

## 4.2 ドリフトとボラティリティ

**📐 定義: ドリフト付きWiener過程**  
ドリフト係数 \\(\mu\\) とボラティリティ係数 \\(\sigma\\) を持つWiener過程： \\[X(t) = X_0 + \mu t + \sigma W(t)\\] **期待値と分散:** \\[E[X(t)] = X_0 + \mu t, \quad Var(X(t)) = \sigma^2 t\\] ドリフト \\(\mu\\) は決定的なトレンド、ボラティリティ \\(\sigma\\) はランダムな変動の大きさを表します。 

### 💻 コード例2: ドリフトとボラティリティの効果

# ドリフト付きWiener過程のシミュレーション np.random.seed(42) def simulate_drift_wiener(X0, mu, sigma, T, dt): """ドリフト付きWiener過程のシミュレーション""" t, W = simulate_wiener_process(T, dt) X = X0 + mu * t + sigma * W return t, X # パラメータ設定 X0 = 100 T = 5 dt = 0.01 # 異なるドリフトとボラティリティの組み合わせ configs = [ {'mu': 0, 'sigma': 1, 'label': 'μ=0, σ=1 (標準Wiener)'}, {'mu': 2, 'sigma': 1, 'label': 'μ=2, σ=1 (正のドリフト)'}, {'mu': -1, 'sigma': 1, 'label': 'μ=-1, σ=1 (負のドリフト)'}, {'mu': 0, 'sigma': 3, 'label': 'μ=0, σ=3 (高ボラティリティ)'}, ] fig, axes = plt.subplots(2, 2, figsize=(14, 10)) axes = axes.flatten() for idx, config in enumerate(configs): mu = config['mu'] sigma = config['sigma'] # 複数経路をシミュレーション for _ in range(20): t, X = simulate_drift_wiener(X0, mu, sigma, T, dt) axes[idx].plot(t, X, alpha=0.4, linewidth=1.5, color='#667eea') # 期待値 t_theory = np.linspace(0, T, 1000) E_X = X0 + mu * t_theory axes[idx].plot(t_theory, E_X, 'r--', linewidth=2.5, label=f'E[X(t)]={X0}+{mu}t') # ±1σ 信頼帯 std_X = sigma * np.sqrt(t_theory) axes[idx].fill_between(t_theory, E_X - std_X, E_X + std_X, alpha=0.2, color='red', label='E[X]±σ') axes[idx].set_xlabel('時間 t', fontsize=11) axes[idx].set_ylabel('X(t)', fontsize=11) axes[idx].set_title(config['label'], fontsize=12, fontweight='bold') axes[idx].legend() axes[idx].grid(alpha=0.3) plt.tight_layout() plt.show() # 統計的検証 print("\nドリフト付きWiener過程の統計的検証:") print("="*60) for config in configs: mu = config['mu'] sigma = config['sigma'] # 複数経路の最終値 n_sims = 1000 final_values = [] for _ in range(n_sims): t, X = simulate_drift_wiener(X0, mu, sigma, T, dt) final_values.append(X[-1]) final_values = np.array(final_values) print(f"\n{config['label']}:") print(f" 理論: E[X({T})]={X0 + mu*T:.2f}, Var(X({T}))={sigma**2*T:.2f}") print(f" 実測: E[X({T})]={final_values.mean():.2f}, Var(X({T}))={final_values.var():.2f}")

## 4.3 確率微分方程式（SDE）とEuler-Maruyama法

**📐 定義: 確率微分方程式（SDE）**  
確率微分方程式の一般形： \\[dX(t) = \mu(X, t) \, dt + \sigma(X, t) \, dW(t)\\] ここで： 

  * \\(\mu(X, t)\\): ドリフト項（決定論的部分）
  * \\(\sigma(X, t)\\): 拡散項（確率論的部分）
  * \\(dW(t)\\): Wiener過程の微小増分

**Euler-Maruyama法（数値解法）:** \\[X_{n+1} = X_n + \mu(X_n, t_n) \Delta t + \sigma(X_n, t_n) \sqrt{\Delta t} \, Z_n\\] ここで \\(Z_n \sim N(0, 1)\\) は独立な標準正規乱数。 

### 💻 コード例3: Euler-Maruyama法によるSDE数値解法

# Euler-Maruyama法の実装 def euler_maruyama(mu_func, sigma_func, X0, T, dt): """ Euler-Maruyama法によるSDEの数値解法 Parameters: \----------- mu_func : function ドリフト項 μ(X, t) sigma_func : function 拡散項 σ(X, t) X0 : float 初期値 T : float 終了時刻 dt : float 時間刻み """ n_steps = int(T / dt) t = np.linspace(0, T, n_steps + 1) X = np.zeros(n_steps + 1) X[0] = X0 for i in range(n_steps): dW = np.random.normal(0, np.sqrt(dt)) X[i+1] = X[i] + mu_func(X[i], t[i]) * dt + sigma_func(X[i], t[i]) * dW return t, X # 例1: 線形SDE dX = -θX dt + σ dW （Ornstein-Uhlenbeck過程の特殊ケース） theta = 0.5 sigma = 1.0 mu_func = lambda X, t: -theta * X sigma_func = lambda X, t: sigma np.random.seed(42) X0 = 5 T = 10 dt = 0.01 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) 複数経路のシミュレーション for _ in range(20): t, X = euler_maruyama(mu_func, sigma_func, X0, T, dt) axes[0, 0].plot(t, X, alpha=0.4, linewidth=1.5, color='#667eea') axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='平衡点') axes[0, 0].set_xlabel('時間 t', fontsize=11) axes[0, 0].set_ylabel('X(t)', fontsize=11) axes[0, 0].set_title('線形SDE: dX = -θX dt + σ dW', fontsize=12, fontweight='bold') axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) # (2) 異なる時間刻みでの比較 dts = [0.1, 0.01, 0.001] np.random.seed(42) for dt_test in dts: t, X = euler_maruyama(mu_func, sigma_func, X0, T, dt_test) axes[0, 1].plot(t, X, linewidth=2, alpha=0.7, label=f'Δt={dt_test}') axes[0, 1].set_xlabel('時間 t', fontsize=11) axes[0, 1].set_ylabel('X(t)', fontsize=11) axes[0, 1].set_title('時間刻みの影響', fontsize=12, fontweight='bold') axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 平均への収束（平均回帰） n_paths = 500 all_paths = [] for _ in range(n_paths): t, X = euler_maruyama(mu_func, sigma_func, X0, T, dt) all_paths.append(X) all_paths = np.array(all_paths) mean_X = all_paths.mean(axis=0) std_X = all_paths.std(axis=0) axes[1, 0].plot(t, mean_X, color='#667eea', linewidth=2.5, label='実測平均') axes[1, 0].fill_between(t, mean_X - std_X, mean_X + std_X, alpha=0.3, color='#667eea', label='±1σ') # 理論的な平均（解析解）: E[X(t)] = X0 * exp(-θt) E_X_theory = X0 * np.exp(-theta * t) axes[1, 0].plot(t, E_X_theory, 'r--', linewidth=2, label='理論平均') axes[1, 0].set_xlabel('時間 t', fontsize=11) axes[1, 0].set_ylabel('E[X(t)]', fontsize=11) axes[1, 0].set_title('平均への収束（平均回帰）', fontsize=12, fontweight='bold') axes[1, 0].legend() axes[1, 0].grid(alpha=0.3) # (4) 定常分布 # 十分時間が経過した後の分布（t→∞で N(0, σ²/(2θ))） final_values = all_paths[:, -1] axes[1, 1].hist(final_values, bins=50, density=True, alpha=0.6, color='#667eea', edgecolor='black', label='実測') # 理論的定常分布 x = np.linspace(final_values.min(), final_values.max(), 1000) stationary_var = sigma**2 / (2 * theta) stationary_pdf = stats.norm.pdf(x, 0, np.sqrt(stationary_var)) axes[1, 1].plot(x, stationary_pdf, 'r-', linewidth=2.5, label=f'N(0, {stationary_var:.2f})') axes[1, 1].set_xlabel('X(∞)', fontsize=11) axes[1, 1].set_ylabel('密度', fontsize=11) axes[1, 1].set_title('定常分布', fontsize=12, fontweight='bold') axes[1, 1].legend() axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show() print("線形SDEの検証:") print(f"理論定常分散: σ²/(2θ) = {stationary_var:.4f}") print(f"実測定常分散: {final_values.var():.4f}")

## 4.4 幾何ブラウン運動（株価モデル）

**📊 定理: 幾何ブラウン運動**  
幾何ブラウン運動は次のSDEで定義されます： \\[dS(t) = \mu S(t) \, dt + \sigma S(t) \, dW(t)\\] **解析解（Itôの公式による）:** \\[S(t) = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t)\right]\\] 期待値: \\(E[S(t)] = S_0 e^{\mu t}\\)  
分散: \\(Var(S(t)) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)\\) **応用:** Black-Scholesモデル（株価、為替レート） 

### 💻 コード例4: 幾何ブラウン運動（株価モデル）

# 幾何ブラウン運動のシミュレーション np.random.seed(42) def geometric_brownian_motion(S0, mu, sigma, T, dt): """幾何ブラウン運動のシミュレーション（解析解を使用）""" n_steps = int(T / dt) t = np.linspace(0, T, n_steps + 1) # Wiener過程を生成 dW = np.random.normal(0, np.sqrt(dt), n_steps) W = np.concatenate([[0], np.cumsum(dW)]) # 解析解 S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W) return t, S # パラメータ（株価モデル） S0 = 100 # 初期株価 mu = 0.1 # ドリフト（年率リターン） sigma = 0.2 # ボラティリティ（年率） T = 1 # 1年間 dt = 1/252 # 1営業日 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) サンプル経路 n_paths = 50 for _ in range(n_paths): t, S = geometric_brownian_motion(S0, mu, sigma, T, dt) axes[0, 0].plot(t, S, alpha=0.4, linewidth=1.5, color='#667eea') # 期待値 t_theory = np.linspace(0, T, 1000) E_S = S0 * np.exp(mu * t_theory) axes[0, 0].plot(t_theory, E_S, 'r--', linewidth=2.5, label=f'E[S(t)]={S0}e^{{μt}}') axes[0, 0].set_xlabel('時間（年）', fontsize=11) axes[0, 0].set_ylabel('株価 S(t)', fontsize=11) axes[0, 0].set_title('幾何ブラウン運動（株価モデル）', fontsize=12, fontweight='bold') axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) axes[0, 0].set_ylim([0, 200]) # (2) 対数収益率の分布 # log(S(t)/S0) ~ N((μ - σ²/2)t, σ²t) n_sims = 5000 final_prices = [] for _ in range(n_sims): t, S = geometric_brownian_motion(S0, mu, sigma, T, dt) final_prices.append(S[-1]) final_prices = np.array(final_prices) log_returns = np.log(final_prices / S0) axes[0, 1].hist(log_returns, bins=50, density=True, alpha=0.6, color='#667eea', edgecolor='black', label='シミュレーション') # 理論分布 x = np.linspace(log_returns.min(), log_returns.max(), 1000) theoretical_mean = (mu - 0.5 * sigma**2) * T theoretical_std = sigma * np.sqrt(T) theoretical_pdf = stats.norm.pdf(x, theoretical_mean, theoretical_std) axes[0, 1].plot(x, theoretical_pdf, 'r-', linewidth=2.5, label='理論分布') axes[0, 1].set_xlabel('対数収益率 log(S(T)/S₀)', fontsize=11) axes[0, 1].set_ylabel('密度', fontsize=11) axes[0, 1].set_title('対数収益率の分布', fontsize=12, fontweight='bold') axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 異なるボラティリティでの比較 sigmas = [0.1, 0.2, 0.3, 0.4] colors = ['blue', 'green', 'orange', 'red'] for sig, color in zip(sigmas, colors): for _ in range(10): t, S = geometric_brownian_motion(S0, mu, sig, T, dt) axes[1, 0].plot(t, S, alpha=0.3, linewidth=1.5, color=color) # 各ボラティリティの期待値 axes[1, 0].plot(t_theory, S0 * np.exp(mu * t_theory), '--', linewidth=2, color=color, label=f'σ={sig}') axes[1, 0].set_xlabel('時間（年）', fontsize=11) axes[1, 0].set_ylabel('株価 S(t)', fontsize=11) axes[1, 0].set_title('異なるボラティリティでの比較', fontsize=12, fontweight='bold') axes[1, 0].legend() axes[1, 0].grid(alpha=0.3) # (4) モンテカルロ価格評価（ヨーロピアンコールオプション） K = 100 # 行使価格 r = 0.05 # リスクフリーレート # オプションのペイオフ max(S(T) - K, 0) payoffs = np.maximum(final_prices - K, 0) option_price_mc = np.exp(-r * T) * payoffs.mean() # Black-Scholes公式（解析解） from scipy.stats import norm as norm_dist d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)) d2 = d1 - sigma*np.sqrt(T) option_price_bs = S0*norm_dist.cdf(d1) - K*np.exp(-r*T)*norm_dist.cdf(d2) axes[1, 1].hist(payoffs, bins=50, alpha=0.6, color='#667eea', edgecolor='black') axes[1, 1].axvline(payoffs.mean(), color='red', linestyle='--', linewidth=2, label=f'平均ペイオフ={payoffs.mean():.2f}') axes[1, 1].set_xlabel('ペイオフ max(S(T)-K, 0)', fontsize=11) axes[1, 1].set_ylabel('頻度', fontsize=11) axes[1, 1].set_title('オプションペイオフの分布', fontsize=12, fontweight='bold') axes[1, 1].legend() axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show() print("幾何ブラウン運動の検証:") print(f"理論: E[S(T)] = {S0 * np.exp(mu * T):.2f}") print(f"実測: E[S(T)] = {final_prices.mean():.2f}") print(f"\nオプション価格:") print(f"モンテカルロ: {option_price_mc:.4f}") print(f"Black-Scholes: {option_price_bs:.4f}")

## 4.5 Ornstein-Uhlenbeck過程（平均回帰）

**📐 定義: Ornstein-Uhlenbeck過程**  
Ornstein-Uhlenbeck過程は、平均回帰性を持つSDEです： \\[dX(t) = \theta(\mu - X(t)) \, dt + \sigma \, dW(t)\\] ここで： 

  * \\(\mu\\): 長期平均（平衡点）
  * \\(\theta > 0\\): 回帰速度
  * \\(\sigma\\): ボラティリティ

**解析解:** \\[X(t) = X_0 e^{-\theta t} + \mu(1 - e^{-\theta t}) + \sigma \int_0^t e^{-\theta(t-s)} dW(s)\\] 定常分布: \\(X(\infty) \sim N\left(\mu, \frac{\sigma^2}{2\theta}\right)\\) 

### 💻 コード例5: Ornstein-Uhlenbeck過程（平均回帰）

# Ornstein-Uhlenbeck過程のシミュレーション np.random.seed(42) def ornstein_uhlenbeck(X0, theta, mu, sigma, T, dt): """Ornstein-Uhlenbeck過程のシミュレーション""" mu_func = lambda X, t: theta * (mu - X) sigma_func = lambda X, t: sigma return euler_maruyama(mu_func, sigma_func, X0, T, dt) # パラメータ X0 = 10 theta = 2.0 # 回帰速度 mu = 5 # 長期平均 sigma = 1.5 # ボラティリティ T = 10 dt = 0.01 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) 異なる初期値からの平均回帰 initial_values = [0, 3, 5, 7, 10] colors = plt.cm.viridis(np.linspace(0, 1, len(initial_values))) for X0_test, color in zip(initial_values, colors): for _ in range(5): t, X = ornstein_uhlenbeck(X0_test, theta, mu, sigma, T, dt) axes[0, 0].plot(t, X, alpha=0.4, linewidth=1.5, color=color) axes[0, 0].axhline(y=mu, color='red', linestyle='--', linewidth=2.5, label=f'長期平均 μ={mu}') axes[0, 0].set_xlabel('時間 t', fontsize=11) axes[0, 0].set_ylabel('X(t)', fontsize=11) axes[0, 0].set_title('平均回帰の可視化', fontsize=12, fontweight='bold') axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) # (2) 異なる回帰速度の比較 thetas = [0.5, 1.0, 2.0, 5.0] for theta_test in thetas: for _ in range(10): t, X = ornstein_uhlenbeck(X0, theta_test, mu, sigma, T, dt) axes[0, 1].plot(t, X, alpha=0.3, linewidth=1.5, label=f'θ={theta_test}' if _ == 0 else None) axes[0, 1].axhline(y=mu, color='red', linestyle='--', linewidth=2) axes[0, 1].set_xlabel('時間 t', fontsize=11) axes[0, 1].set_ylabel('X(t)', fontsize=11) axes[0, 1].set_title('回帰速度θの効果', fontsize=12, fontweight='bold') axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 定常分布 n_paths = 5000 final_values = [] for _ in range(n_paths): t, X = ornstein_uhlenbeck(X0, theta, mu, sigma, T, dt) final_values.append(X[-1]) final_values = np.array(final_values) axes[1, 0].hist(final_values, bins=50, density=True, alpha=0.6, color='#667eea', edgecolor='black', label='シミュレーション') # 理論的定常分布 N(μ, σ²/(2θ)) x = np.linspace(final_values.min(), final_values.max(), 1000) stationary_var = sigma**2 / (2 * theta) stationary_pdf = stats.norm.pdf(x, mu, np.sqrt(stationary_var)) axes[1, 0].plot(x, stationary_pdf, 'r-', linewidth=2.5, label=f'N({mu}, {stationary_var:.2f})') axes[1, 0].set_xlabel('X(∞)', fontsize=11) axes[1, 0].set_ylabel('密度', fontsize=11) axes[1, 0].set_title('定常分布', fontsize=12, fontweight='bold') axes[1, 0].legend() axes[1, 0].grid(alpha=0.3) # (4) 平均と分散の時間発展 all_paths = [] for _ in range(1000): t, X = ornstein_uhlenbeck(X0, theta, mu, sigma, T, dt) all_paths.append(X) all_paths = np.array(all_paths) mean_X = all_paths.mean(axis=0) var_X = all_paths.var(axis=0) # 理論的平均と分散 E_X_theory = X0 * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t)) Var_X_theory = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*t)) axes[1, 1].plot(t, mean_X, color='#667eea', linewidth=2.5, label='実測平均') axes[1, 1].plot(t, E_X_theory, 'r--', linewidth=2, label='理論平均') axes[1, 1].fill_between(t, mean_X - np.sqrt(var_X), mean_X + np.sqrt(var_X), alpha=0.2, color='#667eea', label='実測±1σ') axes[1, 1].fill_between(t, E_X_theory - np.sqrt(Var_X_theory), E_X_theory + np.sqrt(Var_X_theory), alpha=0.2, color='red') axes[1, 1].set_xlabel('時間 t', fontsize=11) axes[1, 1].set_ylabel('E[X(t)]', fontsize=11) axes[1, 1].set_title('平均と分散の時間発展', fontsize=12, fontweight='bold') axes[1, 1].legend() axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show() print("Ornstein-Uhlenbeck過程の検証:") print(f"理論定常分布: N({mu}, {stationary_var:.4f})") print(f"実測定常分布: N({final_values.mean():.4f}, {final_values.var():.4f})")

## 4.6 確率積分のシミュレーション

### 💻 コード例6: 確率積分のシミュレーション

# Itô積分のシミュレーション: I(t) = ∫₀ᵗ f(s) dW(s) np.random.seed(42) def stochastic_integral(f_func, T, dt): """確率積分 ∫₀ᵗ f(s) dW(s) のシミュレーション""" n_steps = int(T / dt) t = np.linspace(0, T, n_steps + 1) # Wiener過程の増分 dW = np.random.normal(0, np.sqrt(dt), n_steps) # 確率積分の計算（Itô積分） I = np.zeros(n_steps + 1) for i in range(n_steps): I[i+1] = I[i] + f_func(t[i]) * dW[i] return t, I T = 5 dt = 0.01 fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) f(t) = 1 の場合: ∫₀ᵗ dW(s) = W(t) f_const = lambda t: 1 n_paths = 20 for _ in range(n_paths): t, I = stochastic_integral(f_const, T, dt) axes[0, 0].plot(t, I, alpha=0.4, linewidth=1.5, color='#667eea') axes[0, 0].set_xlabel('時間 t', fontsize=11) axes[0, 0].set_ylabel('∫₀ᵗ dW(s)', fontsize=11) axes[0, 0].set_title('Itô積分: ∫₀ᵗ dW(s) = W(t)', fontsize=12, fontweight='bold') axes[0, 0].grid(alpha=0.3) # (2) f(t) = t の場合: ∫₀ᵗ s dW(s) f_linear = lambda t: t for _ in range(n_paths): t, I = stochastic_integral(f_linear, T, dt) axes[0, 1].plot(t, I, alpha=0.4, linewidth=1.5, color='#764ba2') axes[0, 1].set_xlabel('時間 t', fontsize=11) axes[0, 1].set_ylabel('∫₀ᵗ s dW(s)', fontsize=11) axes[0, 1].set_title('Itô積分: ∫₀ᵗ s dW(s)', fontsize=12, fontweight='bold') axes[0, 1].grid(alpha=0.3) # (3) Itô積分の性質: E[I(t)] = 0 n_sims = 1000 final_values_const = [] final_values_linear = [] for _ in range(n_sims): t, I_const = stochastic_integral(f_const, T, dt) t, I_linear = stochastic_integral(f_linear, T, dt) final_values_const.append(I_const[-1]) final_values_linear.append(I_linear[-1]) axes[1, 0].hist(final_values_const, bins=50, alpha=0.6, density=True, color='#667eea', edgecolor='black', label='∫₀ᵗ dW(s)') axes[1, 0].hist(final_values_linear, bins=50, alpha=0.6, density=True, color='#764ba2', edgecolor='black', label='∫₀ᵗ s dW(s)') axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='E[I]=0') axes[1, 0].set_xlabel('I(T)', fontsize=11) axes[1, 0].set_ylabel('密度', fontsize=11) axes[1, 0].set_title('Itô積分の分布', fontsize=12, fontweight='bold') axes[1, 0].legend() axes[1, 0].grid(alpha=0.3) # (4) Itô積分の分散: Var(∫₀ᵗ f(s) dW(s)) = ∫₀ᵗ f(s)² ds # f(t) = t の場合: Var = ∫₀ᵗ s² ds = t³/3 theoretical_var = T**3 / 3 print("Itô積分の性質:") print(f"∫₀ᵗ dW(s): E={np.mean(final_values_const):.6f}, Var={np.var(final_values_const):.4f} (理論Var={T:.4f})") print(f"∫₀ᵗ s dW(s): E={np.mean(final_values_linear):.6f}, Var={np.var(final_values_linear):.4f} (理論Var={theoretical_var:.4f})") # Itô積分の二乗可変分 # (∫₀ᵗ f(s) dW(s))² - ∫₀ᵗ f(s)² ds は martingale squared_integrals = [] quadratic_variations = [] for _ in range(n_sims): t, I = stochastic_integral(f_linear, T, dt) squared_integrals.append(I[-1]**2) # ∫₀ᵗ s² ds = t³/3 quadratic_variations.append(I[-1]**2 - T**3/3) axes[1, 1].hist(quadratic_variations, bins=50, alpha=0.6, color='#667eea', edgecolor='black') axes[1, 1].axvline(np.mean(quadratic_variations), color='red', linestyle='--', linewidth=2, label=f'平均={np.mean(quadratic_variations):.4f}') axes[1, 1].set_xlabel('I² - [I,I]', fontsize=11) axes[1, 1].set_ylabel('頻度', fontsize=11) axes[1, 1].set_title('二乗可変分（Martingale性）', fontsize=12, fontweight='bold') axes[1, 1].legend() axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show()

## 4.7 ランジュバン方程式

### 💻 コード例7: ランジュバン方程式

# ランジュバン方程式（過減衰系）: dv = -γv dt + σ dW # 材料科学への応用：ブラウン粒子の運動 np.random.seed(42) # パラメータ gamma = 1.0 # 摩擦係数 sigma = np.sqrt(2 * gamma) # ボラティリティ（揺動散逸定理による） T_sim = 20 dt = 0.01 # 速度のシミュレーション（Ornstein-Uhlenbeck過程） def simulate_velocity(v0, gamma, sigma, T, dt): """ランジュバン方程式による速度のシミュレーション""" mu_func = lambda v, t: -gamma * v sigma_func = lambda v, t: sigma return euler_maruyama(mu_func, sigma_func, v0, T, dt) fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # (1) 速度の時間発展 v0 = 5 n_paths = 50 for _ in range(n_paths): t, v = simulate_velocity(v0, gamma, sigma, T_sim, dt) axes[0, 0].plot(t, v, alpha=0.4, linewidth=1.5, color='#667eea') axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='平衡速度=0') axes[0, 0].set_xlabel('時間', fontsize=11) axes[0, 0].set_ylabel('速度 v(t)', fontsize=11) axes[0, 0].set_title('ランジュバン方程式による速度の時間発展', fontsize=12, fontweight='bold') axes[0, 0].legend() axes[0, 0].grid(alpha=0.3) # (2) 速度の定常分布（Maxwell-Boltzmann分布） n_sims = 5000 final_velocities = [] for _ in range(n_sims): t, v = simulate_velocity(v0, gamma, sigma, T_sim, dt) final_velocities.append(v[-1]) final_velocities = np.array(final_velocities) axes[0, 1].hist(final_velocities, bins=50, density=True, alpha=0.6, color='#667eea', edgecolor='black', label='シミュレーション') # 理論分布（揺動散逸定理により N(0, σ²/(2γ))） x = np.linspace(final_velocities.min(), final_velocities.max(), 1000) stationary_var = sigma**2 / (2 * gamma) stationary_pdf = stats.norm.pdf(x, 0, np.sqrt(stationary_var)) axes[0, 1].plot(x, stationary_pdf, 'r-', linewidth=2.5, label=f'N(0, {stationary_var:.2f})') axes[0, 1].set_xlabel('速度 v', fontsize=11) axes[0, 1].set_ylabel('密度', fontsize=11) axes[0, 1].set_title('速度の定常分布（Maxwell-Boltzmann）', fontsize=12, fontweight='bold') axes[0, 1].legend() axes[0, 1].grid(alpha=0.3) # (3) 位置の時間発展（速度を積分） def simulate_position(v0, gamma, sigma, T, dt): """位置のシミュレーション（速度を時間積分）""" t, v = simulate_velocity(v0, gamma, sigma, T, dt) x = np.cumsum(v) * dt return t, x, v for _ in range(20): t, x, v = simulate_position(v0, gamma, sigma, T_sim, dt) axes[1, 0].plot(t, x, alpha=0.4, linewidth=1.5, color='#764ba2') axes[1, 0].set_xlabel('時間', fontsize=11) axes[1, 0].set_ylabel('位置 x(t)', fontsize=11) axes[1, 0].set_title('ブラウン粒子の位置（拡散）', fontsize=12, fontweight='bold') axes[1, 0].grid(alpha=0.3) # (4) 平均二乗変位（MSD: Mean Square Displacement） # 拡散係数の測定 n_paths_msd = 500 all_positions = [] for _ in range(n_paths_msd): t, x, v = simulate_position(0, gamma, sigma, T_sim, dt) all_positions.append(x) all_positions = np.array(all_positions) msd = np.mean(all_positions**2, axis=0) # 理論的MSD（長時間極限）:  = 2Dt, D = σ²/(2γ²) = 1/γ D_theory = 1 / gamma msd_theory = 2 * D_theory * t axes[1, 1].plot(t, msd, color='#667eea', linewidth=2.5, label='実測MSD') axes[1, 1].plot(t, msd_theory, 'r--', linewidth=2, label=f'理論MSD=2Dt (D={D_theory:.2f})') axes[1, 1].set_xlabel('時間', fontsize=11) axes[1, 1].set_ylabel('', fontsize=11) axes[1, 1].set_title('平均二乗変位（拡散係数の測定）', fontsize=12, fontweight='bold') axes[1, 1].legend() axes[1, 1].grid(alpha=0.3) plt.tight_layout() plt.show() # 拡散係数の測定 # MSD = 2Dt の傾きから拡散係数を推定 from scipy.stats import linregress slope, intercept, r_value, p_value, std_err = linregress(t[100:], msd[100:]) D_measured = slope / 2 print("ランジュバン方程式による拡散:") print(f"理論拡散係数: D = σ²/(2γ²) = {D_theory:.4f}") print(f"測定拡散係数: D = {D_measured:.4f}") print(f"Einstein関係式: D = kT/γ（揺動散逸定理）")

**💡 Note:** ランジュバン方程式は、ブラウン粒子の運動、コロイドの拡散、ポリマーの動力学など、材料科学における様々な現象のモデリングに使われます。揺動散逸定理により、熱平衡状態では摩擦とランダム力の強度が関係づけられます。 

## 演習問題

**📝 演習1: Wiener過程の性質**  
Wiener過程 W(t) について： 

  1. W(2) - W(1) と W(4) - W(3) が独立であることをシミュレーションで確認せよ
  2. Cov(W(2), W(5)) = min(2, 5) = 2 をシミュレーションで検証せよ
  3. W(t) の経路が至る所微分不可能であることを、数値微分で確認せよ

**📝 演習2: 幾何ブラウン運動の応用**  
初期株価 S₀=100, μ=0.08, σ=0.25 の幾何ブラウン運動を考える： 

  1. 1年後の株価分布をシミュレーションし、ヒストグラムで可視化せよ
  2. 行使価格 K=110 のヨーロピアンコールオプションの価格をモンテカルロ法で計算せよ
  3. Black-Scholes公式と比較し、誤差を評価せよ

**📝 演習3: Ornstein-Uhlenbeck過程の解析**  
OU過程 dX = 3(2-X)dt + 1.5 dW について： 

  1. X₀=5 から開始するシミュレーションを実行し、平均への収束を可視化せよ
  2. 定常分布を理論値と比較せよ
  3. 回帰速度 θ を変化させたとき、収束時間がどう変わるかを調べよ

[← 第3章](<chapter-3.html>) [第5章へ →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
