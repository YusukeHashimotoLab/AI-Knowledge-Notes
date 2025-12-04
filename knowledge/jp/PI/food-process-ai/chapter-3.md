---
title: 第3章 プロセス最適化
chapter_title: 第3章 プロセス最適化
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/food-process-ai/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Food Process Ai](<../../PI/food-process-ai/index.html>)›Chapter 3

## 3.1 食品プロセス最適化の課題

食品製造プロセスの最適化は、品質向上、コスト削減、エネルギー効率改善の観点から極めて重要です。 しかし、原材料の変動性、複雑な非線形関係、多目的最適化の必要性など、多くの課題があります。 AI技術、特にベイズ最適化や遺伝的アルゴリズムは、これらの課題に対処する強力な手法です。 

### 食品プロセス最適化の主要な課題

  * **多変数・非線形性** : 温度、時間、pH、配合比など多数のパラメータが複雑に相互作用
  * **評価コストの高さ** : 実験評価に時間とコストがかかる（1実験あたり数時間〜数日）
  * **多目的最適化** : 品質、コスト、エネルギー効率を同時に最適化
  * **制約条件** : 食品安全基準（HACCP）、装置能力の制約
  * **バッチ間変動** : 原材料の季節変動、ロット差

### 🎯 最適化手法の選択ガイド

手法 | 適用場面 | 長所 | 短所  
---|---|---|---  
**ベイズ最適化** | 評価コストが高い、少数の実験で最適化 | サンプル効率が高い、不確実性を考慮 | 計算コストやや高い  
**遺伝的アルゴリズム** | 多目的最適化、離散変数を含む | 大域的最適解探索、実装が容易 | 収束に時間がかかる  
**応答曲面法（RSM）** | 実験計画法と組み合わせ | 統計的根拠、可視化が容易 | 非線形性が強いと精度低下  
**粒子群最適化（PSO）** | 連続変数の最適化 | 実装が簡単、パラメータ調整が少ない | 局所解に陥りやすい  
  
📊 コード例1: ベイズ最適化による加熱条件の最適化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from scipy.stats import norm
    import warnings
    warnings.filterwarnings('ignore')
    
    # 食品加熱プロセスの目的関数（実際のプロセスをシミュレート）
    def food_processing_objective(temperature, time):
        """
        目的関数: 品質スコアを最大化（温度と時間の関数）
        最適点付近: temperature=90°C, time=20分
        """
        # 複雑な非線形関数（実際の食品プロセスを模擬）
        quality = (
            100 * np.exp(-0.5 * ((temperature - 90)/10)**2) * 
            np.exp(-0.5 * ((time - 20)/5)**2) +
            10 * np.sin(temperature / 10) * np.cos(time / 5) +
            np.random.normal(0, 2)  # 測定ノイズ
        )
        return quality
    
    # ベイズ最適化の実装
    class BayesianOptimizer:
        def __init__(self, bounds, n_init=5):
            self.bounds = np.array(bounds)
            self.n_init = n_init
            self.X_sample = []
            self.y_sample = []
            
            # ガウス過程回帰モデル
            kernel = C(1.0, (1e-3, 1e3)) * RBF([10, 10], (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                               alpha=1e-6, normalize_y=True)
        
        def acquisition_function(self, X, xi=0.01):
            """
            獲得関数: Expected Improvement (EI)
            """
            mu, sigma = self.gp.predict(X, return_std=True)
            mu_sample_opt = np.max(self.y_sample)
            
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            return ei
        
        def propose_location(self):
            """
            次の評価点を提案（EIを最大化）
            """
            dim = self.bounds.shape[0]
            min_val = float('inf')
            min_x = None
            
            # ランダムサンプリング + 最適化
            n_restarts = 25
            for _ in range(n_restarts):
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=dim)
                res = self._minimize_acquisition(x0)
                if res < min_val:
                    min_val = res
                    min_x = x0
            
            return min_x
        
        def _minimize_acquisition(self, x0):
            """
            獲得関数の最小化（負のEIを最小化）
            """
            from scipy.optimize import minimize
            
            def objective(x):
                return -self.acquisition_function(x.reshape(1, -1))
            
            bounds_list = [(self.bounds[i, 0], self.bounds[i, 1]) 
                           for i in range(self.bounds.shape[0])]
            
            res = minimize(objective, x0, bounds=bounds_list, method='L-BFGS-B')
            return res.fun
        
        def optimize(self, objective_func, n_iter=20):
            """
            ベイズ最適化の実行
            """
            # 初期ランダムサンプリング
            for _ in range(self.n_init):
                x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                      size=self.bounds.shape[0])
                y = objective_func(x[0], x[1])
                self.X_sample.append(x)
                self.y_sample.append(y)
            
            # ベイズ最適化ループ
            for i in range(n_iter):
                # GPモデル更新
                self.gp.fit(np.array(self.X_sample), np.array(self.y_sample))
                
                # 次の評価点を提案
                x_next = self.propose_location()
                y_next = objective_func(x_next[0], x_next[1])
                
                self.X_sample.append(x_next)
                self.y_sample.append(y_next)
                
                if (i + 1) % 5 == 0:
                    print(f"Iteration {i+1}: Best quality = {np.max(self.y_sample):.2f} "
                          f"at T={self.X_sample[np.argmax(self.y_sample)][0]:.1f}°C, "
                          f"t={self.X_sample[np.argmax(self.y_sample)][1]:.1f}min")
            
            return np.array(self.X_sample), np.array(self.y_sample)
    
    # 最適化実行
    bounds = [[75, 105], [10, 30]]  # [温度範囲, 時間範囲]
    optimizer = BayesianOptimizer(bounds, n_init=5)
    X_sample, y_sample = optimizer.optimize(food_processing_objective, n_iter=25)
    
    # 最適解
    best_idx = np.argmax(y_sample)
    best_temp, best_time = X_sample[best_idx]
    best_quality = y_sample[best_idx]
    
    print(f"\n=== 最適化結果 ===")
    print(f"最適温度: {best_temp:.2f}°C")
    print(f"最適時間: {best_time:.2f}分")
    print(f"最大品質スコア: {best_quality:.2f}")
    print(f"総評価回数: {len(X_sample)}")
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 真の目的関数（参考）
    ax1 = fig.add_subplot(gs[0, 0])
    T_grid = np.linspace(75, 105, 100)
    t_grid = np.linspace(10, 30, 100)
    T_mesh, t_mesh = np.meshgrid(T_grid, t_grid)
    quality_grid = np.zeros_like(T_mesh)
    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            quality_grid[i, j] = food_processing_objective(T_mesh[i, j], t_mesh[i, j])
    
    contour = ax1.contourf(T_mesh, t_mesh, quality_grid, levels=20, cmap='viridis', alpha=0.8)
    ax1.scatter(X_sample[:5, 0], X_sample[:5, 1], c='red', s=100, 
                marker='o', edgecolors='white', linewidth=2, label='初期サンプル', zorder=5)
    ax1.scatter(X_sample[5:, 0], X_sample[5:, 1], c='white', s=80, 
                marker='s', edgecolors='black', linewidth=1.5, label='BO提案点', zorder=5)
    ax1.scatter(best_temp, best_time, c='gold', s=300, marker='*', 
                edgecolors='red', linewidth=2, label='最適解', zorder=10)
    ax1.set_xlabel('温度 (°C)', fontsize=12)
    ax1.set_ylabel('時間 (分)', fontsize=12)
    ax1.set_title('ベイズ最適化の探索過程', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    plt.colorbar(contour, ax=ax1, label='品質スコア')
    
    # 2. 品質スコアの推移
    ax2 = fig.add_subplot(gs[0, 1])
    iterations = np.arange(1, len(y_sample) + 1)
    best_so_far = np.maximum.accumulate(y_sample)
    ax2.plot(iterations, y_sample, 'o-', color='#11998e', alpha=0.6, 
             linewidth=2, markersize=6, label='各評価点の品質')
    ax2.plot(iterations, best_so_far, 'r-', linewidth=3, label='最良値の推移')
    ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='初期サンプル終了')
    ax2.set_xlabel('評価回数', fontsize=12)
    ax2.set_ylabel('品質スコア', fontsize=12)
    ax2.set_title('最適化の収束過程', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. ガウス過程による予測（温度方向のスライス、時間=best_time）
    ax3 = fig.add_subplot(gs[1, 0])
    T_test = np.linspace(75, 105, 200).reshape(-1, 1)
    t_test = np.full_like(T_test, best_time)
    X_test = np.hstack([T_test, t_test])
    
    mu, sigma = optimizer.gp.predict(X_test, return_std=True)
    
    ax3.plot(T_test, mu, 'b-', linewidth=2, label='GP予測平均')
    ax3.fill_between(T_test.ravel(), mu - 1.96*sigma, mu + 1.96*sigma,
                     alpha=0.3, color='blue', label='95%信頼区間')
    ax3.scatter(X_sample[:, 0], y_sample, c='red', s=80, marker='o', 
                edgecolors='white', linewidth=1.5, label='観測点', zorder=5)
    ax3.axvline(x=best_temp, color='green', linestyle='--', linewidth=2, label='最適温度')
    ax3.set_xlabel('温度 (°C)', fontsize=12)
    ax3.set_ylabel('品質スコア', fontsize=12)
    ax3.set_title(f'ガウス過程予測（時間={best_time:.1f}分固定）', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 獲得関数（Expected Improvement）
    ax4 = fig.add_subplot(gs[1, 1])
    ei = optimizer.acquisition_function(X_test)
    ax4.plot(T_test, ei, 'g-', linewidth=2)
    ax4.fill_between(T_test.ravel(), 0, ei, alpha=0.3, color='green')
    ax4.set_xlabel('温度 (°C)', fontsize=12)
    ax4.set_ylabel('Expected Improvement', fontsize=12)
    ax4.set_title('獲得関数（次の評価点の選択基準）', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('bayesian_optimization_food.png', dpi=300, bbox_inches='tight')
    plt.show()

## 3.2 遺伝的アルゴリズムによる多目的最適化

食品プロセスでは、品質、コスト、エネルギー効率など複数の目的を同時に最適化する必要があります。 遺伝的アルゴリズム（GA）、特にNSGA-II（Non-dominated Sorting Genetic Algorithm II）は、 パレート最適解を効率的に探索できます。 

📊 コード例2: 多目的最適化（品質 vs コスト）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import differential_evolution
    
    # 多目的最適化問題の定義
    class FoodProcessMultiObjective:
        def __init__(self):
            # パラメータ範囲
            self.bounds = [
                (75, 105),   # 温度 (°C)
                (10, 30),    # 時間 (分)
                (3.0, 5.0),  # pH
                (50, 150),   # 流量 (L/min)
            ]
        
        def quality_objective(self, params):
            """
            目的関数1: 品質スコアを最大化（最小化問題なので負値）
            """
            temp, time, ph, flow = params
            
            # 品質関数（非線形）
            quality = (
                100 * np.exp(-0.5 * ((temp - 90)/10)**2) *
                np.exp(-0.5 * ((time - 20)/5)**2) *
                np.exp(-0.5 * ((ph - 4.0)/0.5)**2) *
                (1 + 0.1 * np.sin(flow / 20))
            )
            
            return -quality  # 最小化問題に変換
        
        def cost_objective(self, params):
            """
            目的関数2: コストを最小化
            """
            temp, time, ph, flow = params
            
            # エネルギーコスト（温度と時間に依存）
            energy_cost = 0.1 * (temp - 70) * time
            
            # pH調整剤コスト
            ph_cost = 5 * abs(ph - 4.0)
            
            # ポンプ運転コスト
            pump_cost = 0.05 * flow * time
            
            total_cost = energy_cost + ph_cost + pump_cost
            
            return total_cost
        
        def constraints_check(self, params):
            """
            制約条件のチェック（HACCP基準など）
            """
            temp, time, ph, flow = params
            
            # F値（殺菌効果）の計算: F0 = time * 10^((temp-121)/10)
            F0 = time * (10 ** ((temp - 121) / 10))
            
            # 最小F値要求（例: F0 >= 3）
            if F0 < 3:
                return False
            
            # pH範囲制約
            if not (3.0 <= ph <= 5.0):
                return False
            
            return True
    
    # 簡易的なパレート最適化（グリッドサーチ + パレート判定）
    problem = FoodProcessMultiObjective()
    
    # ランダムサンプリングでパレートフロント近似
    np.random.seed(42)
    n_samples = 1000
    
    # ランダムサンプル生成
    samples = []
    qualities = []
    costs = []
    
    for _ in range(n_samples):
        params = [
            np.random.uniform(low, high) 
            for low, high in problem.bounds
        ]
        
        if problem.constraints_check(params):
            quality = -problem.quality_objective(params)  # 元に戻す（最大化）
            cost = problem.cost_objective(params)
            
            samples.append(params)
            qualities.append(quality)
            costs.append(cost)
    
    samples = np.array(samples)
    qualities = np.array(qualities)
    costs = np.array(costs)
    
    # パレート最適解の判定
    def is_pareto_efficient(costs_qualities):
        """
        パレート最適解を判定（2目的の場合）
        cost: 最小化, quality: 最大化
        """
        is_efficient = np.ones(costs_qualities.shape[0], dtype=bool)
        for i, c in enumerate(costs_qualities):
            if is_efficient[i]:
                # 他の解がこの解を支配しているかチェック
                # 支配: cost <= c[0] かつ quality >= c[1] でどちらかが真に優れている
                dominated = np.logical_and(
                    costs_qualities[:, 0] <= c[0],  # コストが同じか少ない
                    costs_qualities[:, 1] >= c[1]   # 品質が同じか高い
                )
                # 真に優れている（等しくない）
                strictly_better = np.logical_or(
                    costs_qualities[:, 0] < c[0],
                    costs_qualities[:, 1] > c[1]
                )
                dominated = np.logical_and(dominated, strictly_better)
                
                if np.any(dominated):
                    is_efficient[i] = False
        
        return is_efficient
    
    # パレート最適解の抽出
    costs_qualities = np.column_stack([costs, qualities])
    pareto_mask = is_pareto_efficient(costs_qualities)
    pareto_samples = samples[pareto_mask]
    pareto_costs = costs[pareto_mask]
    pareto_qualities = qualities[pareto_mask]
    
    # パレートフロントをコストでソート
    sorted_indices = np.argsort(pareto_costs)
    pareto_costs_sorted = pareto_costs[sorted_indices]
    pareto_qualities_sorted = pareto_qualities[sorted_indices]
    pareto_samples_sorted = pareto_samples[sorted_indices]
    
    # 代表的な3つの解を選択
    n_pareto = len(pareto_costs_sorted)
    representative_indices = [0, n_pareto // 2, n_pareto - 1]
    representative_solutions = []
    
    for idx in representative_indices:
        sol = {
            'params': pareto_samples_sorted[idx],
            'quality': pareto_qualities_sorted[idx],
            'cost': pareto_costs_sorted[idx],
            'label': ['コスト重視', 'バランス型', '品質重視'][representative_indices.index(idx)]
        }
        representative_solutions.append(sol)
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. パレートフロント
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(costs, qualities, c='lightgray', alpha=0.4, s=30, label='全サンプル')
    ax1.plot(pareto_costs_sorted, pareto_qualities_sorted, 'r-', linewidth=2.5, 
             label='パレートフロント', zorder=10)
    ax1.scatter(pareto_costs_sorted, pareto_qualities_sorted, c='red', s=100, 
                marker='o', edgecolors='white', linewidth=2, label='パレート最適解', zorder=11)
    
    # 代表解をハイライト
    colors_rep = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, sol in enumerate(representative_solutions):
        ax1.scatter(sol['cost'], sol['quality'], c=colors_rep[i], s=300, 
                    marker='*', edgecolors='black', linewidth=2, 
                    label=sol['label'], zorder=12)
        ax1.annotate(sol['label'], 
                    xy=(sol['cost'], sol['quality']),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc=colors_rep[i], alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_xlabel('コスト（円/バッチ）', fontsize=13)
    ax1.set_ylabel('品質スコア', fontsize=13)
    ax1.set_title('多目的最適化：パレートフロント', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2-4. 代表解のパラメータ比較
    param_names = ['温度 (°C)', '時間 (分)', 'pH', '流量 (L/min)']
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    
    # 温度と時間
    ax2 = axes[0]
    x_pos = np.arange(len(representative_solutions))
    width = 0.35
    
    temp_values = [sol['params'][0] for sol in representative_solutions]
    time_values = [sol['params'][1] for sol in representative_solutions]
    
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x_pos - width/2, temp_values, width, label='温度', color='#ff6b6b', alpha=0.8)
    bars2 = ax2_twin.bar(x_pos + width/2, time_values, width, label='時間', color='#4ecdc4', alpha=0.8)
    
    ax2.set_xlabel('解のタイプ', fontsize=12)
    ax2.set_ylabel('温度 (°C)', fontsize=12, color='#ff6b6b')
    ax2_twin.set_ylabel('時間 (分)', fontsize=12, color='#4ecdc4')
    ax2.set_title('代表解のパラメータ比較（温度・時間）', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([sol['label'] for sol in representative_solutions])
    ax2.tick_params(axis='y', labelcolor='#ff6b6b')
    ax2_twin.tick_params(axis='y', labelcolor='#4ecdc4')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # pH と流量
    ax3 = axes[1]
    ph_values = [sol['params'][2] for sol in representative_solutions]
    flow_values = [sol['params'][3] for sol in representative_solutions]
    
    ax3_twin = ax3.twinx()
    bars3 = ax3.bar(x_pos - width/2, ph_values, width, label='pH', color='#95e1d3', alpha=0.8)
    bars4 = ax3_twin.bar(x_pos + width/2, flow_values, width, label='流量', color='#f38181', alpha=0.8)
    
    ax3.set_xlabel('解のタイプ', fontsize=12)
    ax3.set_ylabel('pH', fontsize=12, color='#95e1d3')
    ax3_twin.set_ylabel('流量 (L/min)', fontsize=12, color='#f38181')
    ax3.set_title('代表解のパラメータ比較（pH・流量）', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([sol['label'] for sol in representative_solutions])
    ax3.tick_params(axis='y', labelcolor='#95e1d3')
    ax3_twin.tick_params(axis='y', labelcolor='#f38181')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('multi_objective_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # レポート出力
    print("=== 多目的最適化結果 ===")
    print(f"総サンプル数: {n_samples}")
    print(f"制約を満たすサンプル数: {len(samples)}")
    print(f"パレート最適解数: {len(pareto_samples)}")
    
    print("\n=== 代表的な3つの解 ===")
    for i, sol in enumerate(representative_solutions):
        print(f"\n{i+1}. {sol['label']}")
        print(f"   温度: {sol['params'][0]:.2f}°C")
        print(f"   時間: {sol['params'][1]:.2f}分")
        print(f"   pH: {sol['params'][2]:.2f}")
        print(f"   流量: {sol['params'][3]:.2f} L/min")
        print(f"   品質スコア: {sol['quality']:.2f}")
        print(f"   コスト: {sol['cost']:.2f} 円/バッチ")
        
        # F値計算
        F0 = sol['params'][1] * (10 ** ((sol['params'][0] - 121) / 10))
        print(f"   殺菌効果（F0値）: {F0:.2f}")

## 3.3 応答曲面法（RSM）による最適化

応答曲面法（Response Surface Methodology: RSM）は、実験計画法と統計モデリングを組み合わせた最適化手法です。 比較的少ない実験回数で、プロセスパラメータと応答変数の関係を2次多項式でモデル化し、最適条件を探索します。 

📊 コード例3: 応答曲面法（Box-Behnken計画）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from mpl_toolkits.mplot3d import Axes3D
    
    # Box-Behnken実験計画（3因子）
    def box_behnken_design(factors):
        """
        Box-Behnken実験計画を生成（3因子の場合）
        """
        # 符号化レベル: -1, 0, +1
        design = np.array([
            [-1, -1,  0], [1, -1,  0], [-1,  1,  0], [1,  1,  0],
            [-1,  0, -1], [1,  0, -1], [-1,  0,  1], [1,  0,  1],
            [ 0, -1, -1], [0,  1, -1], [ 0, -1,  1], [0,  1,  1],
            [ 0,  0,  0], [0,  0,  0], [ 0,  0,  0]  # 中心点（3回反復）
        ])
        
        # 実際の値に変換
        actual_design = []
        for row in design:
            actual_row = []
            for i, level in enumerate(row):
                low, center, high = factors[i]
                if level == -1:
                    actual_row.append(low)
                elif level == 0:
                    actual_row.append(center)
                else:  # level == 1
                    actual_row.append(high)
            actual_design.append(actual_row)
        
        return np.array(actual_design)
    
    # 因子レベル設定（低・中・高）
    factors = [
        (80, 90, 100),   # 温度 (°C)
        (15, 20, 25),    # 時間 (分)
        (3.5, 4.0, 4.5), # pH
    ]
    factor_names = ['温度', '時間', 'pH']
    
    # 実験計画生成
    X_design = box_behnken_design(factors)
    
    # 応答変数（品質スコア）のシミュレート
    def response_function(temp, time, ph):
        """
        真の応答関数（2次多項式 + 交互作用 + ノイズ）
        """
        response = (
            50 +
            2 * (temp - 90) + 0.5 * (time - 20) + 10 * (ph - 4.0) +
            -0.05 * (temp - 90)**2 - 0.1 * (time - 20)**2 - 15 * (ph - 4.0)**2 +
            0.02 * (temp - 90) * (time - 20) +
            0.5 * (temp - 90) * (ph - 4.0) +
            np.random.normal(0, 0.5)  # 実験誤差
        )
        return response
    
    y_response = np.array([response_function(x[0], x[1], x[2]) for x in X_design])
    
    # データフレーム作成
    df_experiment = pd.DataFrame(X_design, columns=factor_names)
    df_experiment['品質スコア'] = y_response
    
    print("=== Box-Behnken実験計画 ===")
    print(df_experiment)
    
    # 2次応答曲面モデルの構築
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_design)
    model = LinearRegression()
    model.fit(X_poly, y_response)
    
    # モデル性能
    y_pred = model.predict(X_poly)
    r2_score = model.score(X_poly, y_response)
    print(f"\n=== 応答曲面モデル ===")
    print(f"R² スコア: {r2_score:.4f}")
    
    # 係数の表示
    feature_names = poly.get_feature_names_out(factor_names)
    coefficients = pd.DataFrame({
        '項': feature_names,
        '係数': model.coef_
    })
    print("\n=== モデル係数 ===")
    print(coefficients)
    
    # 最適化：グリッドサーチで最大値を探索
    temp_range = np.linspace(80, 100, 50)
    time_range = np.linspace(15, 25, 50)
    ph_fixed = 4.0  # pHを固定
    
    T_mesh, t_mesh = np.meshgrid(temp_range, time_range)
    response_mesh = np.zeros_like(T_mesh)
    
    for i in range(T_mesh.shape[0]):
        for j in range(T_mesh.shape[1]):
            X_test = np.array([[T_mesh[i, j], t_mesh[i, j], ph_fixed]])
            X_test_poly = poly.transform(X_test)
            response_mesh[i, j] = model.predict(X_test_poly)[0]
    
    # 最適点
    max_idx = np.unravel_index(np.argmax(response_mesh), response_mesh.shape)
    optimal_temp = T_mesh[max_idx]
    optimal_time = t_mesh[max_idx]
    optimal_response = response_mesh[max_idx]
    
    print(f"\n=== 最適条件（pH={ph_fixed}固定） ===")
    print(f"最適温度: {optimal_temp:.2f}°C")
    print(f"最適時間: {optimal_time:.2f}分")
    print(f"予測品質スコア: {optimal_response:.2f}")
    
    # 可視化
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D応答曲面
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(T_mesh, t_mesh, response_mesh, cmap='viridis', alpha=0.8)
    ax1.scatter(X_design[:, 0], X_design[:, 1], y_response, c='red', s=100, 
                marker='o', edgecolors='white', linewidth=2, label='実験点')
    ax1.scatter(optimal_temp, optimal_time, optimal_response, c='gold', s=300, 
                marker='*', edgecolors='red', linewidth=2, label='最適点')
    ax1.set_xlabel('温度 (°C)', fontsize=11)
    ax1.set_ylabel('時間 (分)', fontsize=11)
    ax1.set_zlabel('品質スコア', fontsize=11)
    ax1.set_title(f'応答曲面（pH={ph_fixed}固定）', fontsize=13, fontweight='bold')
    plt.colorbar(surf, ax=ax1, shrink=0.5, label='品質スコア')
    
    # 2. 等高線図
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contourf(T_mesh, t_mesh, response_mesh, levels=15, cmap='viridis', alpha=0.9)
    contour_lines = ax2.contour(T_mesh, t_mesh, response_mesh, levels=10, colors='white', 
                                 linewidths=0.5, alpha=0.7)
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    ax2.scatter(X_design[:, 0], X_design[:, 1], c='red', s=100, 
                marker='o', edgecolors='white', linewidth=2, label='実験点')
    ax2.scatter(optimal_temp, optimal_time, c='gold', s=300, 
                marker='*', edgecolors='red', linewidth=2, label='最適点')
    ax2.set_xlabel('温度 (°C)', fontsize=12)
    ax2.set_ylabel('時間 (分)', fontsize=12)
    ax2.set_title(f'等高線図（pH={ph_fixed}固定）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    plt.colorbar(contour, ax=ax2, label='品質スコア')
    
    # 3. 予測vs実測値
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(y_response, y_pred, c='#11998e', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
    ax3.plot([y_response.min(), y_response.max()], [y_response.min(), y_response.max()],
             'r--', linewidth=2, label='理想ライン')
    ax3.set_xlabel('実測値', fontsize=12)
    ax3.set_ylabel('予測値', fontsize=12)
    ax3.set_title(f'予測精度（R²={r2_score:.4f}）', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 主効果プロット（温度の効果）
    ax4 = fig.add_subplot(2, 2, 4)
    temp_test = np.linspace(80, 100, 100)
    time_fixed = 20
    ph_test_fixed = 4.0
    
    response_temp_effect = []
    for t in temp_test:
        X_test = np.array([[t, time_fixed, ph_test_fixed]])
        X_test_poly = poly.transform(X_test)
        response_temp_effect.append(model.predict(X_test_poly)[0])
    
    ax4.plot(temp_test, response_temp_effect, linewidth=2.5, color='#11998e', label='予測応答')
    ax4.scatter(X_design[:, 0], y_response, c='red', s=80, alpha=0.6, 
                edgecolors='white', linewidth=1.5, label='実験点')
    ax4.axvline(x=optimal_temp, color='green', linestyle='--', linewidth=2, label='最適温度')
    ax4.set_xlabel('温度 (°C)', fontsize=12)
    ax4.set_ylabel('品質スコア', fontsize=12)
    ax4.set_title(f'主効果プロット（時間={time_fixed}分, pH={ph_test_fixed}固定）', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('response_surface_method.png', dpi=300, bbox_inches='tight')
    plt.show()

### ⚠️ 最適化実装時の注意点

  * **局所最適解** : 非線形性が強い場合、局所解に陥る可能性あり。複数の初期値から最適化を実行
  * **制約条件の厳守** : 食品安全基準（HACCP、F値）は必ず満たす
  * **実験誤差の考慮** : 中心点を複数回繰り返して純誤差を推定
  * **外挿の危険性** : 実験範囲外での予測は信頼性が低い
  * **バッチ間変動** : 原材料変動を考慮したロバスト最適化が望ましい
  * **スケールアップ** : ラボスケールの最適条件が工場スケールで再現されない場合がある

## まとめ

本章では、食品プロセスの最適化手法を学びました：

  * ベイズ最適化による効率的な実験計画と最適条件探索
  * 遺伝的アルゴリズムによる多目的最適化（品質・コスト・エネルギー）
  * 応答曲面法（Box-Behnken計画）による統計的最適化
  * 制約条件（HACCP、食品安全基準）を考慮した実用的最適化

次章では、予知保全とトラブルシューティングの実践的手法を学びます。

[← 第2章: プロセス監視と品質管理](<chapter-2.html>) 第4章: 予知保全とトラブルシューティング →（準備中）

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
