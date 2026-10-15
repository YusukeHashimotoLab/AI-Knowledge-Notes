---
title: 第1章：ベイズ最適化の基礎
chapter_title: 第1章：ベイズ最適化の基礎
subtitle: ブラックボックス最適化の革新的アプローチ
---

## 1.1 ブラックボックス最適化問題

プロセス産業では、入力と出力の関係が複雑で解析的に記述できない「ブラックボックス」な問題が多く存在します。例えば、化学反応器の最適操作条件を探索する場合、温度・圧力・反応時間などのパラメータと収率の関係は、実験やシミュレーションでしか評価できません。

**💡 ブラックボックス最適化の特徴**

  * **目的関数が未知** : f(x) の数式が分からない
  * **評価コストが高い** : 1回の実験に数時間〜数日
  * **勾配情報が利用不可** : ∇f(x) が計算できない
  * **ノイズの存在** : 測定誤差が含まれる

### Example 1: ブラックボックス問題の定式化（化学反応器）

重合反応プロセスを想定し、温度・圧力・触媒濃度の3パラメータを最適化する問題を定義します。以下の2回の評価は初期条件と参照条件の比較であり、参照点は最適点ではありません。触媒の項は濃度とともに増加するため、ノイズのない最適点では触媒濃度が上限の C = 1.0 mol/L になります。観測ノイズの再現性を確保するため、`np.random.seed(42)` を追加しています。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    np.random.seed(42)

    # ===================================
    # Example 1: Black-box optimization problem for chemical reactor
    # ===================================

    class ChemicalReactor:
        """Chemical reactor black-box simulator

        Parameters:
            - Temperature (T): 300-400 K
            - Pressure (P): 1-5 bar
            - Catalyst concentration (C): 0.1-1.0 mol/L

        Objective: Maximize yield
        """

        def __init__(self, noise_level=0.02):
            self.noise_level = noise_level
            self.bounds = np.array([[300, 400], [1, 5], [0.1, 1.0]])
            self.dim_names = ['Temperature (K)', 'Pressure (bar)', 'Catalyst (mol/L)']
            self.reference_x = np.array([350, 3.0, 0.5])  # Reference conditions, not the optimum

        def evaluate(self, x):
            """Calculate yield (equivalent to actual experiment/simulation)

            Args:
                x: [temperature, pressure, catalyst concentration]

            Returns:
                yield: Yield [0-1] + noise
            """
            T, P, C = x

            # Temperature dependency (Arrhenius type)
            T_opt = 350
            temp_factor = np.exp(-((T - T_opt) / 30)**2)

            # Pressure dependency (parabolic type)
            P_opt = 3.0
            pressure_factor = 1 - 0.3 * ((P - P_opt) / 2)**2

            # Catalyst concentration dependency (Langmuir type)
            catalyst_factor = C / (0.2 + C)

            # Interaction term (synergistic effect of temperature and pressure)
            interaction = 0.1 * np.sin((T - 300) / 50 * np.pi) * (P - 1) / 4

            # Yield calculation
            yield_val = 0.85 * temp_factor * pressure_factor * catalyst_factor + interaction

            # Add noise (measurement error)
            noise = np.random.normal(0, self.noise_level)

            return float(np.clip(yield_val + noise, 0, 1))

        def plot_landscape(self, fixed_catalyst=0.5):
            """Visualize objective function (catalyst concentration fixed)"""
            T_range = np.linspace(300, 400, 50)
            P_range = np.linspace(1, 5, 50)
            T_grid, P_grid = np.meshgrid(T_range, P_range)

            Y_grid = np.zeros_like(T_grid)
            for i in range(len(T_range)):
                for j in range(len(P_range)):
                    Y_grid[j, i] = self.evaluate([T_grid[j, i], P_grid[j, i], fixed_catalyst])

            fig = plt.figure(figsize=(10, 4))

            # 3D surface
            ax1 = fig.add_subplot(121, projection='3d')
            surf = ax1.plot_surface(T_grid, P_grid, Y_grid, cmap=cm.viridis, alpha=0.8)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Pressure (bar)')
            ax1.set_zlabel('Yield')
            ax1.set_title('Chemical Reactor Response Surface')

            # Contour
            ax2 = fig.add_subplot(122)
            contour = ax2.contourf(T_grid, P_grid, Y_grid, levels=20, cmap='viridis')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Pressure (bar)')
            ax2.set_title('Yield Contour (Catalyst=0.5 mol/L)')
            plt.colorbar(contour, ax=ax2, label='Yield')

            plt.tight_layout()
            return fig

    # Usage example
    reactor = ChemicalReactor(noise_level=0.02)

    # Evaluate at initial conditions
    x_initial = np.array([320, 2.0, 0.3])
    yield_initial = reactor.evaluate(x_initial)
    print(f"Initial conditions: T={x_initial[0]}K, P={x_initial[1]}bar, C={x_initial[2]}mol/L")
    print(f"Initial yield: {yield_initial:.3f}")

    # Evaluate reference conditions
    x_reference = np.array([350, 3.0, 0.5])
    yield_reference = reactor.evaluate(x_reference)
    print(f"\nReference conditions: T={x_reference[0]}K, P={x_reference[1]}bar, C={x_reference[2]}mol/L")
    print(f"Reference yield: {yield_reference:.3f}")

    # Visualization
    fig = reactor.plot_landscape()
    plt.show()
    

**出力例 / Example output:**  
Initial conditions: T=320.0K, P=2.0bar, C=0.3mol/L  
Initial yield: 0.207  
  
Reference conditions: T=350.0K, P=3.0bar, C=0.5mol/L  
Reference yield: 0.604  

**💡 実務への示唆**

シードを固定した評価では初期条件と参照条件の収率はそれぞれ0.207と0.604ですが、この2点の比較だけでは最適化の効率は示せず、ベイズ最適化は評価コストが高い場合に有用となり得ます。

## 1.2 逐次設計戦略

ベイズ最適化の核心は「逐次設計（Sequential Design）」にあります。前回までの観測結果を活用して、次に評価すべき点を賢く選択することで、効率的な最適化を実現します。

### Example 2: 逐次設計 vs ランダムサンプリング

GP-EIによる逐次設計とランダムサンプリングを、それぞれ正確に13回の目的関数評価で比較します（GP-EIは初期3点と追加10点）。両者とも `default_rng(42)` を使用し、獲得関数の探索では学習済みGPだけを使います。真の関数の曲線は可視化のために別途評価します。
    
    
    import warnings

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


    def make_gp(noise: float = 1e-4, d: int = 1, min_length_scale: float = 1e-2) -> GaussianProcessRegressor:
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * RBF(length_scale=np.ones(d) * 0.2, length_scale_bounds=(min_length_scale, 1e1))
                  + WhiteKernel(noise, (1e-6, 1e-1)))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


    def scale(X, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return (np.asarray(X, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])


    def unscale(U, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return b[:, 0] + np.asarray(U, dtype=float) * (b[:, 1] - b[:, 0])


    def expected_improvement(mu, sigma, best, xi=0.01) -> np.ndarray:
        mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
        imp = mu - best - xi
        out = np.maximum(imp, 0.0)  # sigma == 0: the improvement is certain (or certainly absent)
        pos = sigma > 1e-12
        z = imp[pos] / sigma[pos]
        out[pos] = imp[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
        return out


    def ucb(mu, sigma, kappa) -> np.ndarray:
        return np.asarray(mu, dtype=float) + kappa * np.asarray(sigma, dtype=float)


    def _fit(X, Y, bounds, gp_kwargs: dict | None = None) -> GaussianProcessRegressor:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        gp = make_gp(d=X.shape[1], **(gp_kwargs or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(scale(X, bounds), np.asarray(Y, dtype=float))
        return gp


    def _acq(gp, U, best, acq, kappa):
        mu, sd = gp.predict(np.atleast_2d(U), return_std=True)
        return expected_improvement(mu, sd, best) if acq == "ei" else ucb(mu, sd, kappa)


    def propose(gp, bounds, best, acq: str, rng, kappa=2.0, n_candidates=2000) -> np.ndarray:
        """Maximise the acquisition: n_candidates uniform points in the unit cube, then the best 5
        refined with L-BFGS-B. Returns the proposal in original units."""
        bounds = np.asarray(bounds, dtype=float); d = len(bounds)
        U = rng.random((n_candidates, d))
        a = _acq(gp, U, best, acq, kappa)
        starts = U[np.argsort(-a, kind="stable")[:5]]
        best_u, best_a = starts[0], a[np.argsort(-a, kind="stable")[0]]
        for u0 in starts:
            res = minimize(lambda u: -_acq(gp, u, best, acq, kappa)[0], u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            if -res.fun > best_a:
                best_u, best_a = np.clip(res.x, 0.0, 1.0), -res.fun
        return unscale(best_u, bounds)


    import matplotlib.pyplot as plt

    # ===================================
    # Example 2: Demonstration of sequential design strategy
    # ===================================

    def simple_objective(x):
        """1D test function (analytical solution available)"""
        return -(x - 3)**2 * np.sin(5 * x) + 2

    class SequentialDesigner:
        """Optimization by sequential design"""

        def __init__(self, objective_func, bounds, n_initial=3):
            self.objective = objective_func
            self.bounds = bounds
            self.X_observed = []
            self.Y_observed = []

            # Initial points (uniform random sampling)
            self.rng = np.random.default_rng(42)
            for _ in range(n_initial):
                x = self.rng.uniform(bounds[0], bounds[1])
                y = objective_func(x)
                self.X_observed.append(x)
                self.Y_observed.append(y)

        def select_next_point(self):
            """Maximize GP expected improvement without evaluating the objective."""
            bounds = np.asarray([self.bounds], dtype=float)
            gp = _fit(np.asarray(self.X_observed).reshape(-1, 1), self.Y_observed, bounds)
            return propose(gp, bounds, max(self.Y_observed), "ei", self.rng)[0]

        def optimize(self, n_iterations=10):
            """Execute optimization"""
            for i in range(n_iterations):
                x_next = self.select_next_point()
                y_next = self.objective(x_next)
                self.X_observed.append(x_next)
                self.Y_observed.append(y_next)

                current_best = max(self.Y_observed)
                print(f"Iteration {i+1}: x={x_next:.2f}, y={y_next:.3f}, best={current_best:.3f}")

            return self.X_observed, self.Y_observed

    # Comparison experiment
    bounds = [0, 5]
    n_total = 13  # 3 initial points + 10 additional points

    # 1. Sequential design
    print("=== Sequential Design ===")
    seq_designer = SequentialDesigner(simple_objective, bounds, n_initial=3)
    X_seq, Y_seq = seq_designer.optimize(n_iterations=10)

    # 2. Random sampling
    print("\n=== Random Sampling ===")
    rng = np.random.default_rng(42)
    X_random = rng.uniform(bounds[0], bounds[1], n_total)
    Y_random = [simple_objective(x) for x in X_random]
    for i, (x, y) in enumerate(zip(X_random, Y_random), 1):
        current_best = max(Y_random[:i])
        print(f"Sample {i}: x={x:.2f}, y={y:.3f}, best={current_best:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True function
    x_true = np.linspace(0, 5, 200)
    y_true = simple_objective(x_true)

    # Sequential design
    axes[0].plot(x_true, y_true, 'k-', linewidth=2, label='True function', alpha=0.7)
    axes[0].scatter(X_seq, Y_seq, c=range(len(X_seq)), cmap='viridis',
                    s=100, edgecolor='black', linewidth=1.5, label='Sequential samples', zorder=5)
    axes[0].scatter(X_seq[np.argmax(Y_seq)], max(Y_seq), color='red', s=300,
                    marker='*', edgecolor='black', linewidth=2, label='Best found', zorder=6)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Sequential Design (Best: {:.3f})'.format(max(Y_seq)))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Random sampling
    axes[1].plot(x_true, y_true, 'k-', linewidth=2, label='True function', alpha=0.7)
    axes[1].scatter(X_random, Y_random, c=range(len(X_random)), cmap='plasma',
                    s=100, edgecolor='black', linewidth=1.5, label='Random samples', zorder=5)
    axes[1].scatter(X_random[np.argmax(Y_random)], max(Y_random), color='red', s=300,
                    marker='*', edgecolor='black', linewidth=2, label='Best found', zorder=6)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Random Sampling (Best: {:.3f})'.format(max(Y_random)))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Convergence comparison
    best_seq = [max(Y_seq[:i+1]) for i in range(len(Y_seq))]
    best_random = [max(Y_random[:i+1]) for i in range(len(Y_random))]

    plt.figure(figsize=(8, 5))
    plt.plot(best_seq, 'o-', linewidth=2, markersize=8, label='Sequential Design')
    plt.plot(best_random, 's-', linewidth=2, markersize=8, label='Random Sampling')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Objective Value')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

**出力例 / Example output:**  
=== Sequential Design ===  
Iteration 1: x=1.16, y=3.587, best=3.587  
Iteration 2: x=0.00, y=2.000, best=3.587  
Iteration 3: x=1.43, y=0.122, best=3.587  
Iteration 4: x=1.14, y=3.980, best=3.980  
Iteration 5: x=1.07, y=5.048, best=5.048  
Iteration 6: x=0.86, y=6.202, best=6.202  
Iteration 7: x=0.90, y=6.312, best=6.312  
Iteration 8: x=3.00, y=2.000, best=6.312  
Iteration 9: x=5.00, y=2.529, best=6.312  
Iteration 10: x=3.43, y=2.187, best=6.312  
  
=== Random Sampling ===  
Sample 1: x=3.87, y=1.638, best=1.638  
Sample 2: x=2.19, y=2.649, best=2.649  
Sample 3: x=4.29, y=1.160, best=2.649  
Sample 4: x=3.49, y=2.234, best=2.649  
Sample 5: x=0.47, y=-2.531, best=2.649  
Sample 6: x=4.88, y=4.384, best=4.384  
Sample 7: x=3.81, y=1.884, best=4.384  
Sample 8: x=3.93, y=1.378, best=4.384  
Sample 9: x=0.64, y=2.341, best=4.384  
Sample 10: x=2.25, y=2.540, best=4.384  
Sample 11: x=1.85, y=1.797, best=4.384  
Sample 12: x=4.63, y=4.466, best=4.466  
Sample 13: x=3.22, y=2.018, best=4.466  

**このシードでの実行では、同じ13回の評価予算でGP-EIが6.312、ランダムサンプリングが4.466となり、41.3%向上しました。**

## 1.3 探索と活用のトレードオフ

ベイズ最適化の最も重要な概念が「探索（Exploration）」と「活用（Exploitation）」のバランスです。

  * **活用（Exploitation）** : 現在の最良点付近を集中的に調査し、局所的な改善を追求
  * **探索（Exploration）** : 未知の領域を調査し、より良い大域的最適解を発見

### Example 3: 探索 vs 活用の可視化

入力を [0, 1] にスケーリングしたガウス過程の事後分布と、UCB = 平均 + α × 標準偏差を可視化します。観測ノイズに `np.random.seed(42)` を追加しています。活用は x = 0.90、不確実性の最大化は x = 10.00、UCBは x = 0.30（α = 0.5）または x = 0.00（α = 1.5）を選択します。
    
    
    import warnings

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


    def make_gp(noise: float = 1e-4, d: int = 1, min_length_scale: float = 1e-2) -> GaussianProcessRegressor:
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * RBF(length_scale=np.ones(d) * 0.2, length_scale_bounds=(min_length_scale, 1e1))
                  + WhiteKernel(noise, (1e-6, 1e-1)))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


    def scale(X, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return (np.asarray(X, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])


    def unscale(U, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return b[:, 0] + np.asarray(U, dtype=float) * (b[:, 1] - b[:, 0])


    def _fit(X, Y, bounds, gp_kwargs: dict | None = None) -> GaussianProcessRegressor:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        gp = make_gp(d=X.shape[1], **(gp_kwargs or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(scale(X, bounds), np.asarray(Y, dtype=float))
        return gp


    import matplotlib.pyplot as plt

    np.random.seed(42)

    # ===================================
    # Example 3: Exploration-Exploitation Trade-off
    # ===================================

    class ExplorationExploitationDemo:
        """Visualize exploration-exploitation balance"""

        def __init__(self):
            # Sample function: has multiple local optima
            self.x_range = np.linspace(0, 10, 200)
            self.true_func = lambda x: np.sin(x) + 0.3 * np.sin(3*x) + 0.5 * np.cos(5*x)

            # Observed points (5 points)
            self.X_obs = np.array([1.0, 3.0, 4.5, 7.0, 9.0])
            self.Y_obs = self.true_func(self.X_obs) + np.random.normal(0, 0.1, len(self.X_obs))

        def predict_with_uncertainty(self, x):
            """Gaussian process posterior mean and standard deviation."""
            bounds = np.array([[0.0, 10.0]])
            gp = _fit(self.X_obs.reshape(-1, 1), self.Y_obs, bounds)
            return gp.predict(scale(np.asarray(x).reshape(-1, 1), bounds), return_std=True)

        def exploitation_strategy(self):
            """Exploitation strategy: Select point with maximum predicted mean"""
            mean, _ = self.predict_with_uncertainty(self.x_range)
            return self.x_range[np.argmax(mean)]

        def exploration_strategy(self):
            """Exploration strategy: Select point with maximum uncertainty"""
            _, uncertainty = self.predict_with_uncertainty(self.x_range)
            return self.x_range[np.argmax(uncertainty)]

        def balanced_strategy(self, alpha=0.5):
            """Balanced strategy: UCB (Upper Confidence Bound)"""
            mean, uncertainty = self.predict_with_uncertainty(self.x_range)
            ucb = mean + alpha * uncertainty
            return self.x_range[np.argmax(ucb)]

        def visualize(self):
            """Visualization"""
            mean, uncertainty = self.predict_with_uncertainty(self.x_range)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # (1) True function and prediction
            ax = axes[0, 0]
            ax.plot(self.x_range, self.true_func(self.x_range), 'k--',
                    linewidth=2, label='True function', alpha=0.7)
            ax.plot(self.x_range, mean, 'b-', linewidth=2, label='Predicted mean')
            ax.fill_between(self.x_range, mean - uncertainty, mean + uncertainty,
                            alpha=0.3, label='Uncertainty (±1σ)')
            ax.scatter(self.X_obs, self.Y_obs, c='red', s=100, zorder=5,
                      edgecolor='black', linewidth=1.5, label='Observations')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Model Prediction with Uncertainty')
            ax.legend()
            ax.grid(alpha=0.3)

            # (2) Exploitation strategy
            ax = axes[0, 1]
            x_exploit = self.exploitation_strategy()
            ax.plot(self.x_range, mean, 'b-', linewidth=2, label='Predicted mean')
            ax.scatter(self.X_obs, self.Y_obs, c='gray', s=80, zorder=4, alpha=0.5)
            ax.axvline(x_exploit, color='red', linestyle='--', linewidth=2,
                      label=f'Next point (Exploitation)\nx={x_exploit:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Predicted mean')
            ax.set_title('Exploitation Strategy: Maximize Mean')
            ax.legend()
            ax.grid(alpha=0.3)

            # (3) Exploration strategy
            ax = axes[1, 0]
            x_explore = self.exploration_strategy()
            ax.plot(self.x_range, uncertainty, 'g-', linewidth=2, label='Uncertainty')
            ax.scatter(self.X_obs, np.zeros_like(self.X_obs), c='gray', s=80,
                      zorder=4, alpha=0.5, label='Observations')
            ax.axvline(x_explore, color='blue', linestyle='--', linewidth=2,
                      label=f'Next point (Exploration)\nx={x_explore:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Uncertainty')
            ax.set_title('Exploration Strategy: Maximize Uncertainty')
            ax.legend()
            ax.grid(alpha=0.3)

            # (4) Balanced strategy (UCB)
            ax = axes[1, 1]
            alpha = 1.5
            x_balanced = self.balanced_strategy(alpha=alpha)
            ucb = mean + alpha * uncertainty
            ax.plot(self.x_range, mean, 'b-', linewidth=1.5, label='Mean', alpha=0.7)
            ax.plot(self.x_range, ucb, 'purple', linewidth=2, label=f'UCB (α={alpha})')
            ax.fill_between(self.x_range, mean, ucb, alpha=0.2, color='purple')
            ax.scatter(self.X_obs, self.Y_obs, c='gray', s=80, zorder=4, alpha=0.5)
            ax.axvline(x_balanced, color='purple', linestyle='--', linewidth=2,
                      label=f'Next point (Balanced)\nx={x_balanced:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Value')
            ax.set_title('Balanced Strategy: UCB = Mean + α × Uncertainty')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            return fig

    # Execution
    demo = ExplorationExploitationDemo()

    print("=== Exploration vs Exploitation ===")
    print(f"Exploitation (best predicted point): x = {demo.exploitation_strategy():.2f}")
    print(f"Exploration (maximum uncertainty): x = {demo.exploration_strategy():.2f}")
    print(f"Balanced (UCB, α=0.5): x = {demo.balanced_strategy(alpha=0.5):.2f}")
    print(f"Balanced (UCB, α=1.5): x = {demo.balanced_strategy(alpha=1.5):.2f}")

    fig = demo.visualize()
    plt.show()
    

**出力例 / Example output:**  
=== Exploration vs Exploitation ===  
Exploitation (best predicted point): x = 0.90  
Exploration (maximum uncertainty): x = 10.00  
Balanced (UCB, α=0.5): x = 0.30  
Balanced (UCB, α=1.5): x = 0.00  

**⚠️ 実務での注意点**

活用に偏りすぎると局所最適解に陥り、探索に偏りすぎると収束が遅くなります。ハイパーパラメータ（例: UCBのα）の調整が重要です。一般的な目安は α = 1.0〜2.0 です。

## 1.4 ベイズ最適化の基本ループ

ベイズ最適化のアルゴリズムは、以下の4ステップを反復します：

  1. **サロゲートモデルの構築** : 観測データから目的関数を近似
  2. **獲得関数の計算** : 次に評価すべき点の有望度を定量化
  3. **次点の選択** : 獲得関数を最大化する点を選択
  4. **評価と更新** : 実際に評価し、観測データに追加

### Example 4: シンプルなベイズ最適化の実装

最小限の実装で全体の流れを理解します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.optimize import minimize

    # ===================================
    # Example 4: Simple Bayesian Optimization Loop
    # ===================================

    class SimpleBayesianOptimization:
        """Simple Bayesian optimization (1D)"""

        def __init__(self, objective_func, bounds, noise_std=0.01):
            self.objective = objective_func
            self.bounds = bounds
            self.noise_std = noise_std

            self.X_obs = []
            self.Y_obs = []

            # Initial sampling (3 points)
            np.random.seed(42)
            for _ in range(3):
                x = np.random.uniform(bounds[0], bounds[1])
                y = self.evaluate(x)
                self.X_obs.append(x)
                self.Y_obs.append(y)

        def evaluate(self, x):
            """Evaluate objective function (with noise)"""
            return self.objective(x) + np.random.normal(0, self.noise_std)

        def gaussian_kernel(self, x1, x2, length_scale=0.5):
            """RBF kernel"""
            return np.exp(-0.5 * ((x1 - x2) / length_scale)**2)

        def predict(self, x_test):
            """Prediction by Gaussian process (simplified version)"""
            X_obs_array = np.array(self.X_obs).reshape(-1, 1)
            Y_obs_array = np.array(self.Y_obs).reshape(-1, 1)
            x_test_array = np.array(x_test).reshape(-1, 1)

            # Kernel matrix
            K = np.zeros((len(self.X_obs), len(self.X_obs)))
            for i in range(len(self.X_obs)):
                for j in range(len(self.X_obs)):
                    K[i, j] = self.gaussian_kernel(self.X_obs[i], self.X_obs[j])

            # Add noise term
            K += self.noise_std**2 * np.eye(len(self.X_obs))

            # Kernel with test point
            k_star = np.array([self.gaussian_kernel(self.X_obs[i], x_test)
                              for i in range(len(self.X_obs))])

            # Predicted mean
            K_inv = np.linalg.inv(K)
            mean = k_star.T @ K_inv @ Y_obs_array

            # Predicted variance
            k_star_star = self.gaussian_kernel(x_test, x_test)
            variance = k_star_star - k_star.T @ K_inv @ k_star
            if variance.ndim == 2:
                variance = np.diag(variance)  # Marginal variance for each query point
            std = np.sqrt(np.maximum(variance, 0))

            return mean.flatten(), std.flatten()

        def expected_improvement(self, x):
            """Expected Improvement acquisition function"""
            mean, std = self.predict(x)
            best_y = max(self.Y_obs)

            # EI calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (mean - best_y) / (std + 1e-9)
                ei = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)
                ei[std == 0] = 0.0

            return -ei  # Convert to minimization problem

        def select_next_point(self):
            """Select next evaluation point (maximize EI)"""
            result = minimize(
                lambda x: self.expected_improvement(x),
                x0=np.random.uniform(self.bounds[0], self.bounds[1]),
                bounds=[self.bounds],
                method='L-BFGS-B'
            )
            return result.x[0]

        def optimize(self, n_iterations=10, verbose=True):
            """Execute optimization"""
            for i in range(n_iterations):
                # Select next point
                x_next = self.select_next_point()
                y_next = self.evaluate(x_next)

                # Update data
                self.X_obs.append(x_next)
                self.Y_obs.append(y_next)

                current_best = max(self.Y_obs)
                if verbose:
                    print(f"Iter {i+1}: x={x_next:.3f}, y={y_next:.3f}, best={current_best:.3f}")

            best_idx = np.argmax(self.Y_obs)
            return self.X_obs[best_idx], self.Y_obs[best_idx]

        def plot_progress(self):
            """Visualize optimization progress"""
            x_plot = np.linspace(self.bounds[0], self.bounds[1], 200)
            y_true = [self.objective(x) for x in x_plot]
            mean, std = self.predict(x_plot)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Surrogate model
            ax1.plot(x_plot, y_true, 'k--', linewidth=2, label='True function', alpha=0.7)
            ax1.plot(x_plot, mean, 'b-', linewidth=2, label='GP mean')
            ax1.fill_between(x_plot, mean - 2*std, mean + 2*std, alpha=0.3, label='95% CI')
            ax1.scatter(self.X_obs, self.Y_obs, c='red', s=100, zorder=5,
                       edgecolor='black', linewidth=1.5, label='Observations')
            best_idx = np.argmax(self.Y_obs)
            ax1.scatter(self.X_obs[best_idx], self.Y_obs[best_idx],
                       c='gold', s=300, marker='*', zorder=6,
                       edgecolor='black', linewidth=2, label='Best')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Gaussian Process Surrogate Model')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Acquisition function
            ei_values = -self.expected_improvement(x_plot)
            ax2.plot(x_plot, ei_values, 'g-', linewidth=2, label='Expected Improvement')
            ax2.fill_between(x_plot, 0, ei_values, alpha=0.3, color='green')
            ax2.axvline(self.X_obs[-1], color='red', linestyle='--',
                       linewidth=2, label=f'Last selected: x={self.X_obs[-1]:.3f}')
            ax2.set_xlabel('x')
            ax2.set_ylabel('EI(x)')
            ax2.set_title('Acquisition Function (Expected Improvement)')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            return fig

    # Test function
    def test_function(x):
        return -(x - 2.5)**2 * np.sin(10 * x) + 3

    # Execution
    print("=== Simple Bayesian Optimization ===\n")
    bo = SimpleBayesianOptimization(test_function, bounds=[0, 5], noise_std=0.05)
    x_best, y_best = bo.optimize(n_iterations=12, verbose=True)

    print(f"\n=== Final Result ===")
    print(f"Best x: {x_best:.4f}")
    print(f"Best y: {y_best:.4f}")

    fig = bo.plot_progress()
    plt.show()
    

**出力例 / Example output:**  
=== Simple Bayesian Optimization ===  
  
Iter 1: x=2.058, y=2.858, best=2.993  
Iter 2: x=0.000, y=2.971, best=2.993  
Iter 3: x=0.917, y=2.343, best=2.993  
Iter 4: x=1.331, y=1.941, best=2.993  
Iter 5: x=1.456, y=2.055, best=2.993  
Iter 6: x=4.405, y=2.732, best=2.993  
Iter 7: x=1.966, y=2.756, best=2.993  
Iter 8: x=2.280, y=3.004, best=3.004  
Iter 9: x=2.787, y=2.998, best=3.004  
Iter 10: x=2.663, y=2.980, best=3.004  
Iter 11: x=0.853, y=0.851, best=3.004  
Iter 12: x=0.325, y=3.494, best=3.494  
  
=== Final Result ===  
Best x: 0.3253  
Best y: 3.4938  

## 1.5 比較: BO vs グリッドサーチ vs ランダムサーチ

同じBranin問題と評価予算で、3手法が到達した最良値を比較します。

### Example 5: 3手法の性能比較
    
    
    import warnings

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


    def make_gp(noise: float = 1e-4, d: int = 1, min_length_scale: float = 1e-2) -> GaussianProcessRegressor:
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * RBF(length_scale=np.ones(d) * 0.2, length_scale_bounds=(min_length_scale, 1e1))
                  + WhiteKernel(noise, (1e-6, 1e-1)))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


    def scale(X, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return (np.asarray(X, dtype=float) - b[:, 0]) / (b[:, 1] - b[:, 0])


    def unscale(U, bounds) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        return b[:, 0] + np.asarray(U, dtype=float) * (b[:, 1] - b[:, 0])


    def expected_improvement(mu, sigma, best, xi=0.01) -> np.ndarray:
        mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
        imp = mu - best - xi
        out = np.maximum(imp, 0.0)  # sigma == 0: the improvement is certain (or certainly absent)
        pos = sigma > 1e-12
        z = imp[pos] / sigma[pos]
        out[pos] = imp[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
        return out


    def ucb(mu, sigma, kappa) -> np.ndarray:
        return np.asarray(mu, dtype=float) + kappa * np.asarray(sigma, dtype=float)


    def _fit(X, Y, bounds, gp_kwargs: dict | None = None) -> GaussianProcessRegressor:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        gp = make_gp(d=X.shape[1], **(gp_kwargs or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(scale(X, bounds), np.asarray(Y, dtype=float))
        return gp


    def _acq(gp, U, best, acq, kappa):
        mu, sd = gp.predict(np.atleast_2d(U), return_std=True)
        return expected_improvement(mu, sd, best) if acq == "ei" else ucb(mu, sd, kappa)


    def propose(gp, bounds, best, acq: str, rng, kappa=2.0, n_candidates=2000) -> np.ndarray:
        """Maximise the acquisition: n_candidates uniform points in the unit cube, then the best 5
        refined with L-BFGS-B. Returns the proposal in original units."""
        bounds = np.asarray(bounds, dtype=float); d = len(bounds)
        U = rng.random((n_candidates, d))
        a = _acq(gp, U, best, acq, kappa)
        starts = U[np.argsort(-a, kind="stable")[:5]]
        best_u, best_a = starts[0], a[np.argsort(-a, kind="stable")[0]]
        for u0 in starts:
            res = minimize(lambda u: -_acq(gp, u, best, acq, kappa)[0], u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            if -res.fun > best_a:
                best_u, best_a = np.clip(res.x, 0.0, 1.0), -res.fun
        return unscale(best_u, bounds)


    import matplotlib.pyplot as plt

    # ===================================
    # Example 5: BO vs Grid Search vs Random Search
    # ===================================

    # Test function (2D)
    def branin_function(x):
        """Branin function (global optimization benchmark)"""
        x1, x2 = x[0], x[1]
        a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
        r, s, t = 6, 10, 1/(8*np.pi)

        term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s

        return -(term1 + term2 + term3)  # Convert to maximization problem

    class OptimizationComparison:
        """Comparison experiment of three methods"""

        def __init__(self, objective, bounds, budget=30):
            self.objective = objective
            self.bounds = np.array(bounds)  # [[x1_min, x1_max], [x2_min, x2_max]]
            self.budget = budget
            self.dim = len(bounds)

        def grid_search(self):
            """Grid search"""
            n_per_dim = int(np.ceil(self.budget ** (1/self.dim)))

            grid_1d = [np.linspace(b[0], b[1], n_per_dim) for b in self.bounds]
            grid = np.meshgrid(*grid_1d)

            X_grid = np.column_stack([g.ravel() for g in grid])[:self.budget]
            Y_grid = [self.objective(x) for x in X_grid]

            return X_grid, Y_grid

        def random_search(self, seed=42):
            """Random search"""
            rng = np.random.default_rng(seed)
            X_random = rng.uniform(
                self.bounds[:, 0], self.bounds[:, 1],
                size=(self.budget, self.dim)
            )
            Y_random = [self.objective(x) for x in X_random]

            return X_random, Y_random

        def bayesian_optimization(self, seed=42):
            """Gaussian process optimization with expected improvement."""
            rng = np.random.default_rng(seed)
            n_init = 5
            X = unscale(rng.random((n_init, self.dim)), self.bounds)
            Y = [self.objective(x) for x in X]

            for _ in range(self.budget - n_init):
                gp = _fit(X, Y, self.bounds)
                x_next = propose(gp, self.bounds, max(Y), "ei", rng)
                y_next = self.objective(x_next)
                X = np.vstack([X, x_next])
                Y.append(y_next)

            return X, Y

        def compare(self, n_trials=5):
            """Compare average performance over multiple trials"""
            results = {
                'Grid Search': [],
                'Random Search': [],
                'Bayesian Optimization': []
            }

            for trial in range(n_trials):
                print(f"\n=== Trial {trial + 1}/{n_trials} ===")

                # Grid Search
                X_grid, Y_grid = self.grid_search()
                best_grid = [max(Y_grid[:i+1]) for i in range(len(Y_grid))]
                results['Grid Search'].append(best_grid)
                print(f"Grid Search best: {max(Y_grid):.4f}")

                # Random Search
                X_rand, Y_rand = self.random_search(seed=trial)
                best_rand = [max(Y_rand[:i+1]) for i in range(len(Y_rand))]
                results['Random Search'].append(best_rand)
                print(f"Random Search best: {max(Y_rand):.4f}")

                # Bayesian Optimization
                X_bo, Y_bo = self.bayesian_optimization(seed=trial)
                best_bo = [max(Y_bo[:i+1]) for i in range(len(Y_bo))]
                results['Bayesian Optimization'].append(best_bo)
                print(f"Bayesian Optimization best: {max(Y_bo):.4f}")

            return results

        def plot_comparison(self, results):
            """Visualize comparison results"""
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = {'Grid Search': 'blue', 'Random Search': 'orange',
                     'Bayesian Optimization': 'green'}

            for method, trials in results.items():
                trials_array = np.array(trials)
                mean_curve = trials_array.mean(axis=0)
                std_curve = trials_array.std(axis=0)

                x_axis = np.arange(1, len(mean_curve) + 1)
                ax.plot(x_axis, mean_curve, linewidth=2.5, label=method,
                       color=colors[method], marker='o', markersize=4)
                ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve,
                               alpha=0.2, color=colors[method])

            ax.set_xlabel('Number of Evaluations', fontsize=12)
            ax.set_ylabel('Best Objective Value Found', fontsize=12)
            ax.set_title('Optimization Performance Comparison (Mean ± Std over 5 trials)',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)

            return fig

    # Run experiment
    bounds = [[-5, 10], [0, 15]]  # Domain of Branin function
    comparison = OptimizationComparison(branin_function, bounds, budget=30)

    print("Running optimization comparison...")
    results = comparison.compare(n_trials=5)

    # Final performance summary
    print("\n=== Final Performance Summary ===")
    for method, trials in results.items():
        final_values = [trial[-1] for trial in trials]
        print(f"{method:25s}: {np.mean(final_values):.4f} ± {np.std(final_values):.4f}")

    fig = comparison.plot_comparison(results)
    plt.show()
    

**出力例 / Example output:**  
Running optimization comparison...  
  
=== Trial 1/5 ===  
Grid Search best: -1.9431  
Random Search best: -1.6409  
Bayesian Optimization best: -0.4049  
  
=== Trial 2/5 ===  
Grid Search best: -1.9431  
Random Search best: -1.7697  
Bayesian Optimization best: -0.3980  
  
=== Trial 3/5 ===  
Grid Search best: -1.9431  
Random Search best: -0.8426  
Bayesian Optimization best: -0.3981  
  
=== Trial 4/5 ===  
Grid Search best: -1.9431  
Random Search best: -5.0113  
Bayesian Optimization best: -0.3980  
  
=== Trial 5/5 ===  
Grid Search best: -1.9431  
Random Search best: -2.7390  
Bayesian Optimization best: -0.3984  
  
=== Final Performance Summary ===  
Grid Search              : -1.9431 ± 0.0000  
Random Search            : -2.4007 ± 1.4375  
Bayesian Optimization    : -0.3995 ± 0.0027  

**3つの手法はいずれも同じ 30 回の評価予算** で実行しているため、ここで比較しているのは「速さ」ではなく「同じ予算で得られた最良値の良さ」です。これらの目的関数値は負の値（Branin関数の符号を反転して最大化問題にしている）なので、**比をとっても意味がありません**。代わりに最適値までの**残差**で比較します。この探索範囲での最適値は約 -0.397887 です。表ではこの値を用いて、5回の試行の最良値の平均から残差を計算しています：

手法 | 30回の評価での最良値 | 最適値までの残差  
---|---|---  
グリッドサーチ | -1.9431 | 1.5452  
ランダムサーチ | -2.4007 | 2.0028  
ベイズ最適化 | -0.3995 | 0.0016  

**✅ ベイズ最適化の優位性**

  * **評価効率** : 各30回の評価を行う5つのシードでの試行では、GP-EIの平均結果が最良で、グリッドサーチが残した残差の約99.9%、ランダムサーチが残した残差の約99.9%を埋めています
  * **予算のスケール** : ここでのグリッドサーチは 1 次元あたり `ceil(30**(1/2)) = 6` 点しか取れません（6×6 格子を 30 点に切り詰めています）。2次元で格子幅を半分にするには評価回数が 4 倍必要ですが、GP-EIは学習済みモデルが期待改善量を最大と予測する点に1回ずつ評価を振り向けます
  * **ロバスト性** : GP-EIの最終値の標準偏差はこの5回の試行で0.0027です。この実験での安定性を示しますが、一般的な保証ではありません

## 1.6 収束解析とイテレーション追跡

最適化の進捗を定量的に評価し、収束判定の方法を学びます。

### Example 6: 収束診断ツール

この診断例では、GPの事後分布ではなく、距離に基づく重み付き平均を用いた簡易探索による別の履歴を維持しています。真の最適値は与えません。3.62はこの目的関数の正しい最適値ではなかったため、`true_optimum=None` として最適値までの残差を省きます。この実行では7.006472が得られますが停止条件を満たさず、最大の改善は反復16（Δy = 3.494641）で生じます。
    
    
    import numpy as np
    import matplotlib.pyplot as plt

    # ===================================
    # Example 6: Convergence analysis and iteration tracking
    # ===================================

    class ConvergenceAnalyzer:
        """Analyze convergence of Bayesian optimization"""

        def __init__(self, X_history, Y_history, true_optimum=None):
            self.X_history = np.array(X_history)
            self.Y_history = np.array(Y_history)
            self.true_optimum = true_optimum
            self.n_iter = len(Y_history)

        def compute_metrics(self):
            """Calculate convergence metrics"""
            # Cumulative best value
            cumulative_best = [max(self.Y_history[:i+1]) for i in range(self.n_iter)]

            # Improvement (progress at each iteration)
            improvements = [0]
            for i in range(1, self.n_iter):
                improvements.append(max(0, cumulative_best[i] - cumulative_best[i-1]))

            # Optimality gap (if true optimum is known)
            if self.true_optimum is not None:
                optimality_gap = [self.true_optimum - cb for cb in cumulative_best]
            else:
                optimality_gap = None

            # Convergence rate (standard deviation of recent 5 improvements)
            convergence_rate = []
            window = 5
            for i in range(self.n_iter):
                if i < window:
                    convergence_rate.append(np.nan)
                else:
                    recent_improvements = improvements[i-window+1:i+1]
                    convergence_rate.append(np.std(recent_improvements))

            return {
                'cumulative_best': cumulative_best,
                'improvements': improvements,
                'optimality_gap': optimality_gap,
                'convergence_rate': convergence_rate
            }

        def is_converged(self, tolerance=1e-3, patience=5):
            """Convergence determination"""
            metrics = self.compute_metrics()
            improvements = metrics['improvements']

            # Improvement is less than tolerance for recent patience iterations
            if len(improvements) < patience:
                return False

            recent_improvements = improvements[-patience:]
            return all(imp < tolerance for imp in recent_improvements)

        def plot_diagnostics(self):
            """Diagnostic plots"""
            metrics = self.compute_metrics()

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            iterations = np.arange(1, self.n_iter + 1)

            # (1) Transition of cumulative best value
            ax = axes[0, 0]
            ax.plot(iterations, metrics['cumulative_best'], 'b-', linewidth=2, marker='o')
            if self.true_optimum is not None:
                ax.axhline(self.true_optimum, color='red', linestyle='--',
                          linewidth=2, label=f'True optimum: {self.true_optimum:.3f}')
                ax.legend()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Value Found')
            ax.set_title('Convergence: Cumulative Best')
            ax.grid(alpha=0.3)

            # (2) Improvement at each iteration
            ax = axes[0, 1]
            ax.bar(iterations, metrics['improvements'], color='green', alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Improvement')
            ax.set_title('Improvement per Iteration')
            ax.grid(alpha=0.3, axis='y')

            # (3) Optimality gap (log scale)
            ax = axes[1, 0]
            if metrics['optimality_gap'] is not None:
                gap = np.array(metrics['optimality_gap'])
                gap[gap <= 0] = 1e-10  # Handle negative values
                ax.semilogy(iterations, gap, 'r-', linewidth=2, marker='s')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Optimality Gap (log scale)')
                ax.set_title('Distance to True Optimum')
                ax.grid(alpha=0.3, which='both')
            else:
                ax.text(0.5, 0.5, 'True optimum unknown',
                       ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])

            # (4) Convergence rate (variability of improvements)
            ax = axes[1, 1]
            valid_idx = ~np.isnan(metrics['convergence_rate'])
            ax.plot(iterations[valid_idx], np.array(metrics['convergence_rate'])[valid_idx],
                   'purple', linewidth=2, marker='d')
            ax.axhline(1e-3, color='orange', linestyle='--',
                      linewidth=2, label='Convergence threshold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Convergence Rate (Std of recent improvements)')
            ax.set_title('Convergence Rate Indicator')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            return fig

        def print_summary(self):
            """Summary report"""
            metrics = self.compute_metrics()

            print("=== Convergence Analysis Summary ===")
            print(f"Total iterations: {self.n_iter}")
            print(f"Best value found: {max(self.Y_history):.6f}")
            print(f"Final improvement: {metrics['improvements'][-1]:.6f}")

            if self.true_optimum is not None:
                final_gap = self.true_optimum - max(self.Y_history)
                print(f"True optimum: {self.true_optimum:.6f}")
                print(f"Optimality gap: {final_gap:.6f} ({final_gap/self.true_optimum*100:.2f}%)")

            converged = self.is_converged()
            print(f"Converged: {'Yes' if converged else 'No'}")

            # Iteration with maximum improvement
            max_imp_iter = np.argmax(metrics['improvements'])
            print(f"Largest improvement at iteration: {max_imp_iter + 1} "
                  f"(Δy = {metrics['improvements'][max_imp_iter]:.6f})")

    # Demo execution (same objective as Example 4, separate noisy history)
    np.random.seed(42)

    def test_func(x):
        return -(x - 2.5)**2 * np.sin(10 * x) + 3

    # Run BO and obtain history
    from scipy.stats import norm
    from scipy.optimize import minimize

    X_hist, Y_hist = [], []
    bounds = [0, 5]

    # Initial sampling
    for _ in range(3):
        x = np.random.uniform(bounds[0], bounds[1])
        X_hist.append(x)
        Y_hist.append(test_func(x) + np.random.normal(0, 0.02))

    # Sequential optimization (15 iterations)
    for iteration in range(15):
        # Distance-weighted heuristic prediction (not a GP posterior)
        def gp_predict(x_test):
            dists = np.abs(np.array(X_hist) - x_test)
            weights = np.exp(-dists**2 / 0.5)
            mean = weights @ Y_hist / (weights.sum() + 1e-10)
            std = 0.5 * (1 - np.exp(-np.min(dists) / 1.0))
            return mean, std

        # EI acquisition function
        def neg_ei(x):
            mean, std = gp_predict(x)
            z = (mean - max(Y_hist)) / (std + 1e-9)
            ei = (mean - max(Y_hist)) * norm.cdf(z) + std * norm.pdf(z)
            return -ei

        # Next point selection
        res = minimize(neg_ei, x0=np.random.uniform(bounds[0], bounds[1]),
                      bounds=[bounds], method='L-BFGS-B')

        x_next = res.x[0]
        y_next = test_func(x_next) + np.random.normal(0, 0.02)

        X_hist.append(x_next)
        Y_hist.append(y_next)

    # Convergence analysis
    analyzer = ConvergenceAnalyzer(X_hist, Y_hist, true_optimum=None)
    analyzer.print_summary()
    fig = analyzer.plot_diagnostics()
    plt.show()
    

**出力例 / Example output:**  
=== Convergence Analysis Summary ===  
Total iterations: 18  
Best value found: 7.006472  
Final improvement: 0.000000  
Converged: No  
Largest improvement at iteration: 16 (Δy = 3.494641)  

## 1.7 ハイパーパラメータチューニング実践例

ハイパーパラメータチューニングはベイズ最適化の応用先ですが、この簡潔な例では距離に基づく重み付き平均の簡易サロゲートとEI形式の選択規則を維持しており、Example 2・3・5で使うGPの事後分布は実装していません。木の予測の平均化と探索結果の再現性を確保するため、両方のRandom Forestに `n_jobs=1` を指定します。

### Example 7: Random Forestのハイパーパラメータ最適化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_regression
    from scipy.stats import norm
    from scipy.optimize import minimize

    # ===================================
    # Example 7: Hyperparameter tuning
    # ===================================

    # Generate sample dataset (assuming sensor data from chemical process)
    np.random.seed(42)
    X_data, y_data = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)

    class HyperparameterOptimizer:
        """EI-style search of Random Forest hyperparameters with a heuristic surrogate"""

        def __init__(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train

            # Three hyperparameters to optimize
            self.param_bounds = {
                'n_estimators': [10, 200],      # Number of decision trees
                'max_depth': [3, 20],            # Maximum depth
                'min_samples_split': [2, 20]     # Minimum samples for split
            }

            self.X_obs = []
            self.Y_obs = []

            # Initial random sampling (5 points)
            for _ in range(5):
                params = self._sample_random_params()
                score = self._evaluate(params)
                self.X_obs.append(params)
                self.Y_obs.append(score)

        def _sample_random_params(self):
            """Randomly sample hyperparameters"""
            return [
                np.random.randint(self.param_bounds['n_estimators'][0],
                                self.param_bounds['n_estimators'][1] + 1),
                np.random.randint(self.param_bounds['max_depth'][0],
                                self.param_bounds['max_depth'][1] + 1),
                np.random.randint(self.param_bounds['min_samples_split'][0],
                                self.param_bounds['min_samples_split'][1] + 1)
            ]

        def _evaluate(self, params):
            """Evaluate hyperparameter performance (5-fold CV)"""
            n_est, max_d, min_split = [int(p) for p in params]

            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_split,
                random_state=42,
                n_jobs=1
            )

            # Cross-validation R² score
            scores = cross_val_score(model, self.X_train, self.y_train,
                                    cv=5, scoring='r2')
            return scores.mean()

        def _gp_predict(self, params_test):
            """Distance-weighted heuristic prediction (not a GP posterior)"""
            X_obs_array = np.array(self.X_obs)
            params_array = np.array(params_test).reshape(1, -1)

            # Normalization
            X_obs_norm = (X_obs_array - X_obs_array.mean(axis=0)) / (X_obs_array.std(axis=0) + 1e-10)
            params_norm = (params_array - X_obs_array.mean(axis=0)) / (X_obs_array.std(axis=0) + 1e-10)

            # Distance-based weights
            dists = np.linalg.norm(X_obs_norm - params_norm, axis=1)
            weights = np.exp(-dists**2 / 2.0)

            mean = weights @ self.Y_obs / (weights.sum() + 1e-10)
            std = 0.2 * (1 - np.exp(-np.min(dists) / 1.5))

            return mean, std

        def _expected_improvement(self, params):
            """EI acquisition function"""
            mean, std = self._gp_predict(params)
            best_y = max(self.Y_obs)

            z = (mean - best_y) / (std + 1e-9)
            ei = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)

            return -ei  # Minimization problem

        def optimize(self, n_iterations=15):
            """Execute the heuristic EI-style search"""
            print("=== Hyperparameter Optimization ===\n")

            for i in range(n_iterations):
                # Select next hyperparameters
                bounds_array = [
                    self.param_bounds['n_estimators'],
                    self.param_bounds['max_depth'],
                    self.param_bounds['min_samples_split']
                ]

                result = minimize(
                    self._expected_improvement,
                    x0=self._sample_random_params(),
                    bounds=bounds_array,
                    method='L-BFGS-B'
                )

                params_next = [int(p) for p in result.x]
                score_next = self._evaluate(params_next)

                self.X_obs.append(params_next)
                self.Y_obs.append(score_next)

                current_best = max(self.Y_obs)
                print(f"Iter {i+1}: n_est={params_next[0]}, max_depth={params_next[1]}, "
                      f"min_split={params_next[2]} → R²={score_next:.4f} (best={current_best:.4f})")

            # Best parameters
            best_idx = np.argmax(self.Y_obs)
            best_params = self.X_obs[best_idx]
            best_score = self.Y_obs[best_idx]

            print(f"\n=== Best Hyperparameters ===")
            print(f"n_estimators: {best_params[0]}")
            print(f"max_depth: {best_params[1]}")
            print(f"min_samples_split: {best_params[2]}")
            print(f"Best R² score: {best_score:.4f}")

            return best_params, best_score

        def plot_optimization_history(self):
            """Visualize optimization history"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            iterations = np.arange(1, len(self.Y_obs) + 1)
            cumulative_best = [max(self.Y_obs[:i+1]) for i in range(len(self.Y_obs))]

            # (1) Transition of R² score
            ax = axes[0, 0]
            ax.plot(iterations, self.Y_obs, 'o-', linewidth=2, markersize=8,
                   label='Observed R²', alpha=0.7)
            ax.plot(iterations, cumulative_best, 's-', linewidth=2.5, markersize=8,
                   color='red', label='Best R²')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('R² Score')
            ax.set_title('Optimization Progress')
            ax.legend()
            ax.grid(alpha=0.3)

            # (2) Exploration trajectory of n_estimators
            ax = axes[0, 1]
            n_estimators = [x[0] for x in self.X_obs]
            scatter = ax.scatter(iterations, n_estimators, c=self.Y_obs,
                               cmap='viridis', s=100, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('n_estimators')
            ax.set_title('Parameter Exploration: n_estimators')
            plt.colorbar(scatter, ax=ax, label='R² Score')
            ax.grid(alpha=0.3)

            # (3) Exploration trajectory of max_depth
            ax = axes[1, 0]
            max_depth = [x[1] for x in self.X_obs]
            scatter = ax.scatter(iterations, max_depth, c=self.Y_obs,
                               cmap='viridis', s=100, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('max_depth')
            ax.set_title('Parameter Exploration: max_depth')
            plt.colorbar(scatter, ax=ax, label='R² Score')
            ax.grid(alpha=0.3)

            # (4) Parameter space exploration (2D projection: n_estimators vs max_depth)
            ax = axes[1, 1]
            scatter = ax.scatter(n_estimators, max_depth, c=self.Y_obs,
                               s=150, cmap='viridis', edgecolor='black', linewidth=1.5)
            best_idx = np.argmax(self.Y_obs)
            ax.scatter(n_estimators[best_idx], max_depth[best_idx],
                      s=500, marker='*', color='red', edgecolor='black', linewidth=2,
                      label='Best', zorder=5)
            ax.set_xlabel('n_estimators')
            ax.set_ylabel('max_depth')
            ax.set_title('Parameter Space Exploration')
            plt.colorbar(scatter, ax=ax, label='R² Score')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            return fig

    # Execution
    optimizer = HyperparameterOptimizer(X_data, y_data)
    best_params, best_score = optimizer.optimize(n_iterations=15)

    fig = optimizer.plot_optimization_history()
    plt.show()

    # Comparison with baseline
    print("\n=== Comparison with Default Parameters ===")
    default_model = RandomForestRegressor(random_state=42, n_jobs=1)
    default_score = cross_val_score(default_model, X_data, y_data, cv=5, scoring='r2').mean()
    print(f"Default R² score: {default_score:.4f}")
    print(f"Optimized R² score: {best_score:.4f}")
    print(f"Improvement: {(best_score - default_score) / default_score * 100:.2f}%")
    

**出力例 / Example output:**  
=== Hyperparameter Optimization ===  
  
Iter 1: n_est=185, max_depth=16, min_split=20 → R²=0.6402 (best=0.7324)  
Iter 2: n_est=10, max_depth=20, min_split=2 → R²=0.6670 (best=0.7324)  
Iter 3: n_est=50, max_depth=20, min_split=10 → R²=0.6945 (best=0.7324)  
Iter 4: n_est=10, max_depth=20, min_split=20 → R²=0.5485 (best=0.7324)  
Iter 5: n_est=161, max_depth=9, min_split=12 → R²=0.7046 (best=0.7324)  
Iter 6: n_est=10, max_depth=5, min_split=9 → R²=0.6190 (best=0.7324)  
Iter 7: n_est=200, max_depth=20, min_split=2 → R²=0.7397 (best=0.7397)  
Iter 8: n_est=98, max_depth=15, min_split=19 → R²=0.6435 (best=0.7397)  
Iter 9: n_est=16, max_depth=3, min_split=19 → R²=0.5038 (best=0.7397)  
Iter 10: n_est=200, max_depth=3, min_split=20 → R²=0.5538 (best=0.7397)  
Iter 11: n_est=10, max_depth=12, min_split=2 → R²=0.6664 (best=0.7397)  
Iter 12: n_est=90, max_depth=3, min_split=11 → R²=0.5750 (best=0.7397)  
Iter 13: n_est=118, max_depth=20, min_split=2 → R²=0.7381 (best=0.7397)  
Iter 14: n_est=200, max_depth=10, min_split=2 → R²=0.7368 (best=0.7397)  
Iter 15: n_est=67, max_depth=15, min_split=10 → R²=0.7061 (best=0.7397)  
  
=== Best Hyperparameters ===  
n_estimators: 200  
max_depth: 20  
min_samples_split: 2  
Best R² score: 0.7397  
  
=== Comparison with Default Parameters ===  
Default R² score: 0.7368  
Optimized R² score: 0.7397  
Improvement: 0.40%  

この実行での最良パラメータは `n_estimators=200`、`max_depth=20`、`min_samples_split=2` です。交差検証の平均R²は0.7368から0.7397へと相対的に0.40%だけ向上していますが、同じ分割での小さな改善であり、独立したテスト集合で検証された改善ではありません。

**💡 実務での活用**

このアプローチは以下のようなプロセス産業の問題に直接適用できます：

  * **品質予測モデル** : センサーデータから製品品質を予測
  * **異常検知モデル** : プラント運転データから異常を早期検出
  * **制御パラメータ最適化** : PIDゲインなどの調整

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ ブラックボックス最適化問題の特徴と課題を説明できる
  * ✅ 逐次設計戦略の利点をランダムサーチと比較して述べられる
  * ✅ 探索と活用のトレードオフの概念を理解している

### 実践スキル

  * ✅ 化学プロセスのブラックボックス問題を定式化できる
  * ✅ シンプルなベイズ最適化ループを実装できる
  * ✅ 3手法（BO/Grid/Random）の性能を比較評価できる
  * ✅ 収束診断ツールを使って最適化の進捗を分析できる

### 応用力

  * ✅ 機械学習モデルのハイパーパラメータチューニングに応用できる
  * ✅ 実務問題に対して適切な最適化手法を選択できる
  * ✅ 最適化結果の信頼性を評価できる

## 次のステップ

第1章では、ベイズ最適化の基本概念と実装を学びました。次章では、ベイズ最適化の中核技術である「ガウス過程」について詳しく学びます。

**📚 次章の内容（第2章予告）**

  * ガウス過程の数学的基礎
  * カーネル関数の種類と選択
  * ハイパーパラメータの最尤推定
  * 実データへのフィッティング

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
