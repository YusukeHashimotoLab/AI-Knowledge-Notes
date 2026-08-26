---
title: 第4章：多目的・制約付き最適化
chapter_title: 第4章：多目的・制約付き最適化
subtitle: 複雑な実世界問題への対応
---

## 4.1 実世界の複雑性

実際のプロセス最適化では、単一の目的関数を最大化するだけでは不十分です。 品質と効率のトレードオフ、安全性制約、物理的制約など、複数の要求を同時に満たす必要があります。 

#### 💡 実プロセスの典型的な要求

  * **多目的** : 収率↑ かつ エネルギー↓ かつ コスト↓
  * **制約** : 例として 温度 ≤ 150°C、圧力 ≤ 5 bar、pH 6-8
  * **実現可能性** : 物理的に可能な条件のみ
  * **安全性** : 爆発限界外、毒性物質濃度制限

上の数値はあくまで低圧プロセスの一例です。本章の各実装例はそれぞれ独自の操作範囲を定義しており、節をまたいで数値を引き継いではいません。例えば §4.4 の制約付き最適化の例は別のより高圧の反応器で、温度 < 130°C、圧力 < 60 bar を制約としています。

本章では、これらの複雑な要求に対応するベイズ最適化の拡張手法を学びます。 

## 4.2 パレートフロント探索

多目的最適化では、すべての目的で最良の単一解は存在しません。 代わりに、パレート最適解の集合（パレートフロント）を求めます。 

### Example 1: 多目的ベイズ最適化（収率 vs エネルギー）
    
    
    # 多目的ベイズ最適化: パレートフロント探索
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # 2目的のプロセス関数（収率とエネルギー消費）
    def process_objectives(X):
        """
        Args:
            X: [temperature, pressure] (N, 2)
        Returns:
            yield: 収率 (最大化したい)
            energy: エネルギー消費 (最小化したい)
        """
        temp, pres = X[:, 0], X[:, 1]
    
        # 収率モデル（温度・圧力の関数）
        yield_rate = (
            50 + 20 * np.sin(temp/20) +
            15 * np.cos(pres/10) -
            0.1 * (temp - 100)**2 -
            0.05 * (pres - 50)**2 +
            np.random.normal(0, 1, len(temp))
        )
    
        # エネルギー消費（温度・圧力に比例）
        energy = (
            0.5 * temp + 0.3 * pres +
            0.01 * temp * pres +
            np.random.normal(0, 2, len(temp))
        )
    
        return yield_rate, energy
    
    # パレート支配の判定
    def is_pareto_efficient(costs):
        """パレート最適解の判定（最小化問題に変換）
    
        Args:
            costs: (N, M) 目的関数値（最小化方向）
        Returns:
            is_efficient: (N,) bool配列
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # iより全ての目的で劣る点は非効率
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )
                is_efficient[i] = True
        return is_efficient
    
    # 多目的ベイズ最適化の実行
    np.random.seed(42)
    n_iterations = 30
    
    # 初期サンプリング
    X = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1, Y2 = process_objectives(X)
    
    # 記録用
    all_X = X.copy()
    all_Y1 = Y1.copy()
    all_Y2 = Y2.copy()
    
    for iteration in range(n_iterations):
        # 各目的でGPモデルを構築
        kernel = C(1.0) * RBF([10.0, 10.0])
    
        gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp1.fit(all_X, all_Y1)
    
        gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp2.fit(all_X, all_Y2)
    
        # 候補点の生成
        X_candidates = np.random.uniform([50, 20], [150, 80], (500, 2))
    
        # 各候補での予測
        mu1, sigma1 = gp1.predict(X_candidates, return_std=True)
        mu2, sigma2 = gp2.predict(X_candidates, return_std=True)
    
        # スカラー化: weighted sum with uncertainty
        # 収率最大化 = -収率最小化
        # 重みを反復ごとに変化させて多様なパレート解を探索
        weight = iteration / n_iterations  # 0→1
        scalarized = (
            -(1 - weight) * (mu1 + 2 * sigma1) +  # 収率（最大化）
            weight * (mu2 - 2 * sigma2)            # エネルギー（最小化）
        )
    
        # 次の実験点
        next_idx = np.argmin(scalarized)
        next_x = X_candidates[next_idx]
    
        # 実験実行
        next_y1, next_y2 = process_objectives(next_x.reshape(1, -1))
    
        # データ追加
        all_X = np.vstack([all_X, next_x])
        all_Y1 = np.hstack([all_Y1, next_y1])
        all_Y2 = np.hstack([all_Y2, next_y2])
    
    # パレートフロントの抽出
    costs = np.column_stack([-all_Y1, all_Y2])  # 収率は負に（最小化問題化）
    pareto_mask = is_pareto_efficient(costs)
    pareto_X = all_X[pareto_mask]
    pareto_Y1 = all_Y1[pareto_mask]
    pareto_Y2 = all_Y2[pareto_mask]
    
    # ソート（可視化用）
    sorted_idx = np.argsort(pareto_Y1)
    pareto_Y1_sorted = pareto_Y1[sorted_idx]
    pareto_Y2_sorted = pareto_Y2[sorted_idx]
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 目的関数空間
    ax1.scatter(all_Y1, all_Y2, c='lightgray', alpha=0.5, s=30, label='All evaluations')
    ax1.scatter(pareto_Y1, pareto_Y2, c='red', s=100, marker='*',
                label=f'Pareto front ({len(pareto_Y1)} points)', zorder=10)
    ax1.plot(pareto_Y1_sorted, pareto_Y2_sorted, 'r--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Yield (%)', fontsize=12)
    ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax1.set_title('Pareto Front in Objective Space', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 決定変数空間
    scatter = ax2.scatter(pareto_X[:, 0], pareto_X[:, 1], c=pareto_Y1,
                          s=150, cmap='RdYlGn', marker='*', edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Pressure (bar)', fontsize=12)
    ax2.set_title('Pareto Solutions in Decision Space', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Yield (%)', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=150, bbox_inches='tight')
    
    print(f"発見したパレート最適解: {len(pareto_Y1)}個")
    print(f"\n代表的なトレードオフ解:")
    print(f"  高収率重視: 収率={pareto_Y1.max():.1f}%, エネルギー={pareto_Y2[np.argmax(pareto_Y1)]:.1f}kWh")
    print(f"  低エネルギー重視: 収率={pareto_Y1[np.argmin(pareto_Y2)]:.1f}%, エネルギー={pareto_Y2.min():.1f}kWh")
    

#### ✅ パレート最適化の利点

  * 複数の目的間のトレードオフを可視化
  * 意思決定者が好みに応じて解を選択可能
  * 目的の重要度が事前に不明でも対応可能

## 4.3 Expected Hypervolume Improvement (EHVI)

多目的最適化専用の獲得関数。パレートフロントが占める超体積の改善を最大化します。 

### Example 2: EHVI獲得関数の実装
    
    
    # Expected Hypervolume Improvement (EHVI)
    from scipy.stats import norm
    
    def compute_hypervolume_2d(pareto_front, ref_point):
        """2次元のハイパーボリューム計算（簡易版）
    
        Args:
            pareto_front: (N, 2) パレート最適解（最小化問題）
            ref_point: (2,) 参照点
        """
        # ソート
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
    
        hv = 0.0
        prev_x = ref_point[0]
    
        for point in sorted_front:
            width = prev_x - point[0]
            height = ref_point[1] - point[1]
            if width > 0 and height > 0:
                hv += width * height
            prev_x = point[0]
    
        return hv
    
    def expected_hypervolume_improvement(X, gp1, gp2, pareto_front, ref_point, n_samples=100):
        """EHVI獲得関数（モンテカルロ近似）
    
        Args:
            X: 評価点
            gp1, gp2: 各目的のGPモデル
            pareto_front: 現在のパレートフロント
            ref_point: 参照点
            n_samples: サンプリング数
        """
        mu1, sigma1 = gp1.predict(X, return_std=True)
        mu2, sigma2 = gp2.predict(X, return_std=True)
    
        # 現在のハイパーボリューム
        current_hv = compute_hypervolume_2d(pareto_front, ref_point)
    
        ehvi = np.zeros(len(X))
    
        for i in range(len(X)):
            hv_improvements = []
    
            for _ in range(n_samples):
                # 新点での目的関数値をサンプリング
                y1_new = np.random.normal(mu1[i], sigma1[i])
                y2_new = np.random.normal(mu2[i], sigma2[i])
    
                # 新点を追加したパレートフロント
                new_front_candidates = np.vstack([
                    pareto_front,
                    [-y1_new, y2_new]  # 収率は負に変換
                ])
    
                # 新しいパレートフロント
                new_pareto_mask = is_pareto_efficient(new_front_candidates)
                new_pareto_front = new_front_candidates[new_pareto_mask]
    
                # ハイパーボリューム改善
                new_hv = compute_hypervolume_2d(new_pareto_front, ref_point)
                hv_improvements.append(max(0, new_hv - current_hv))
    
            ehvi[i] = np.mean(hv_improvements)
    
        return ehvi
    
    # EHVIを用いた多目的BO
    print("EHVI最適化実行中...")
    
    # 初期設定
    X_init = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1_init, Y2_init = process_objectives(X_init)
    
    X_all = X_init.copy()
    Y1_all = Y1_init.copy()
    Y2_all = Y2_init.copy()
    
    n_iter = 20
    
    for iteration in range(n_iter):
        # GPモデル構築
        gp1 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp1.fit(X_all, Y1_all)
    
        gp2 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp2.fit(X_all, Y2_all)
    
        # 現在のパレートフロント
        costs = np.column_stack([-Y1_all, Y2_all])
        pareto_mask = is_pareto_efficient(costs)
        current_pareto = costs[pareto_mask]
    
        # 参照点（最悪値より少し悪い点）
        ref_point = np.array([
            costs[:, 0].max() + 10,
            costs[:, 1].max() + 10
        ])
    
        # 候補点
        X_candidates = np.random.uniform([50, 20], [150, 80], (100, 2))
    
        # EHVI計算
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}/{n_iter}")
    
        ehvi = expected_hypervolume_improvement(
            X_candidates, gp1, gp2, current_pareto, ref_point, n_samples=50
        )
    
        # 次実験点
        next_x = X_candidates[np.argmax(ehvi)]
        next_y1, next_y2 = process_objectives(next_x.reshape(1, -1))
    
        X_all = np.vstack([X_all, next_x])
        Y1_all = np.hstack([Y1_all, next_y1])
        Y2_all = np.hstack([Y2_all, next_y2])
    
    # 最終パレートフロント
    costs_final = np.column_stack([-Y1_all, Y2_all])
    pareto_mask_final = is_pareto_efficient(costs_final)
    
    print(f"\nEHVI最適化結果:")
    print(f"  評価点数: {len(X_all)}")
    print(f"  パレート最適解: {pareto_mask_final.sum()}個")
    print(f"  ハイパーボリューム: {compute_hypervolume_2d(costs_final[pareto_mask_final], ref_point):.2f}")
    

#### 💡 EHVIの特徴

  * パレートフロント全体の改善を考慮（単一点ではなく）
  * 3目的以上でも理論的に拡張可能
  * 計算コストは高い（近似手法が推奨）

## 4.4 制約付き最適化

実プロセスでは、物理的・安全性の制約が存在します。 制約違反点での評価を避けながら最適化する手法を学びます。 

### Example 3: ガウス過程による制約ハンドリング
    
    
    # 制約付きベイズ最適化
    def constrained_process(X):
        """制約付きプロセス
    
        目的: 反応速度を最大化
        制約: 温度 < 130°C かつ 圧力 < 60 bar（安全性）
        """
        temp, pres = X[:, 0], X[:, 1]
    
        # 目的関数（反応速度）
        rate = (
            10 + 0.5*temp + 0.3*pres -
            0.002*temp**2 - 0.001*pres**2 +
            0.01*temp*pres +
            np.random.normal(0, 0.5, len(temp))
        )
    
        # 制約関数（負なら制約違反）
        # g1: 130 - temp >= 0
        # g2: 60 - pres >= 0
        constraint1 = 130 - temp
        constraint2 = 60 - pres
    
        return rate, constraint1, constraint2
    
    def probability_of_feasibility(X, gp_constraints):
        """実行可能確率の計算
    
        Args:
            X: 評価点
            gp_constraints: [GP(c1), GP(c2), ...]
        Returns:
            prob: 全制約を満たす確率
        """
        prob = np.ones(len(X))
    
        for gp_c in gp_constraints:
            mu_c, sigma_c = gp_c.predict(X, return_std=True)
            # c >= 0 の確率
            prob *= norm.cdf(mu_c / (sigma_c + 1e-6))
    
        return prob
    
    def constrained_ei(X, gp_obj, gp_constraints, y_best, xi=0.01):
        """制約付きExpected Improvement
    
        Args:
            X: 評価点
            gp_obj: 目的関数のGP
            gp_constraints: 制約関数のGPリスト
            y_best: 実行可能領域での最良値
        """
        # 通常のEI
        mu, sigma = gp_obj.predict(X, return_std=True)
        imp = mu - y_best - xi
        Z = imp / (sigma + 1e-6)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    
        # 実行可能確率で重み付け
        pof = probability_of_feasibility(X, gp_constraints)
    
        return ei * pof
    
    # 制約付きBO実行
    np.random.seed(42)
    
    # 初期サンプリング（制約違反含む）
    X = np.random.uniform([80, 30], [150, 80], (15, 2))
    rate, c1, c2 = constrained_process(X)
    
    X_all = X.copy()
    rate_all = rate.copy()
    c1_all = c1.copy()
    c2_all = c2.copy()
    
    n_iter = 25
    
    for iteration in range(n_iter):
        # GP構築（目的と制約）
        gp_obj = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_obj.fit(X_all, rate_all)
    
        gp_c1 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c1.fit(X_all, c1_all)
    
        gp_c2 = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c2.fit(X_all, c2_all)
    
        # 実行可能領域での最良値
        feasible_mask = (c1_all >= 0) & (c2_all >= 0)
        if feasible_mask.any():
            y_best = rate_all[feasible_mask].max()
        else:
            y_best = rate_all.min()  # まだ実行可能解がない
    
        # 候補点
        X_cand = np.random.uniform([80, 30], [150, 80], (500, 2))
    
        # 制約付きEI
        cei = constrained_ei(X_cand, gp_obj, [gp_c1, gp_c2], y_best)
    
        # 次実験点
        next_x = X_cand[np.argmax(cei)]
        next_rate, next_c1, next_c2 = constrained_process(next_x.reshape(1, -1))
    
        X_all = np.vstack([X_all, next_x])
        rate_all = np.hstack([rate_all, next_rate])
        c1_all = np.hstack([c1_all, next_c1])
        c2_all = np.hstack([c2_all, next_c2])
    
    # 結果分析
    final_feasible = (c1_all >= 0) & (c2_all >= 0)
    feasible_X = X_all[final_feasible]
    feasible_rate = rate_all[final_feasible]
    
    best_idx = np.argmax(feasible_rate)
    best_X = feasible_X[best_idx]
    best_rate = feasible_rate[best_idx]
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 探索履歴
    colors = ['red' if not f else 'green' for f in final_feasible]
    ax1.scatter(X_all[:, 0], X_all[:, 1], c=colors, alpha=0.6, s=50)
    ax1.scatter(best_X[0], best_X[1], c='gold', s=300, marker='*',
                edgecolors='black', linewidth=2, label='Best feasible', zorder=10)
    ax1.axvline(130, color='red', linestyle='--', alpha=0.7, label='Temp constraint')
    ax1.axhline(60, color='blue', linestyle='--', alpha=0.7, label='Pressure constraint')
    ax1.fill_between([80, 130], 30, 60, alpha=0.1, color='green', label='Feasible region')
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Pressure (bar)', fontsize=12)
    ax1.set_title('Constrained Optimization History', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 収束曲線
    feasible_best = []
    for i in range(len(rate_all)):
        mask = final_feasible[:i+1]
        if mask.any():
            feasible_best.append(rate_all[:i+1][mask].max())
        else:
            feasible_best.append(np.nan)
    
    ax2.plot(feasible_best, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best Feasible Reaction Rate', fontsize=12)
    ax2.set_title('Convergence in Feasible Region', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_optimization.png', dpi=150, bbox_inches='tight')
    
    print(f"\n制約付き最適化結果:")
    print(f"  総評価点数: {len(X_all)}")
    print(f"  実行可能点: {final_feasible.sum()} ({final_feasible.sum()/len(X_all)*100:.1f}%)")
    print(f"  最良実行可能解:")
    print(f"    温度 = {best_X[0]:.1f}°C, 圧力 = {best_X[1]:.1f} bar")
    print(f"    反応速度 = {best_rate:.2f}")
    

#### ✅ 制約ハンドリング戦略

  * **ソフト制約** : 実行可能確率で重み付け（上記例）
  * **ハード制約** : 制約違反点を候補から除外
  * **ペナルティ法** : 目的関数にペナルティ項追加

## 4.5 Probability of Feasibility

制約を満たす確率を明示的にモデル化する手法。 

### Example 4: Probability of Feasibilityの実装
    
    
    # Probability of Feasibility (PoF)の詳細実装
    def visualize_feasibility_probability(gp_constraints, X_grid):
        """実行可能確率の可視化"""
        pof = probability_of_feasibility(X_grid, gp_constraints)
        return pof
    
    # グリッド生成
    temp_range = np.linspace(80, 150, 100)
    pres_range = np.linspace(30, 80, 100)
    Temp, Pres = np.meshgrid(temp_range, pres_range)
    X_grid = np.column_stack([Temp.ravel(), Pres.ravel()])
    
    # 最終モデルでのPoF
    gp_c1_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_c1_final.fit(X_all, c1_all)
    
    gp_c2_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_c2_final.fit(X_all, c2_all)
    
    pof_grid = visualize_feasibility_probability([gp_c1_final, gp_c2_final], X_grid)
    PoF = pof_grid.reshape(Temp.shape)
    
    # 目的関数の予測
    gp_obj_final = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]))
    gp_obj_final.fit(X_all, rate_all)
    rate_pred = gp_obj_final.predict(X_grid).reshape(Temp.shape)
    
    # 可視化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # PoF
    im1 = ax1.contourf(Temp, Pres, PoF, levels=20, cmap='RdYlGn')
    ax1.contour(Temp, Pres, PoF, levels=[0.9, 0.95, 0.99], colors='black',
                linewidths=1.5, linestyles='--')
    ax1.scatter(X_all[:, 0], X_all[:, 1], c='blue', s=30, alpha=0.6, edgecolors='white')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Probability of Feasibility')
    plt.colorbar(im1, ax=ax1, label='PoF')
    
    # 予測反応速度
    im2 = ax2.contourf(Temp, Pres, rate_pred, levels=20, cmap='viridis')
    ax2.contour(Temp, Pres, PoF, levels=[0.5], colors='red', linewidths=2, linestyles='-')
    ax2.scatter(best_X[0], best_X[1], c='gold', s=300, marker='*', edgecolors='black', linewidth=2)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Predicted Reaction Rate')
    plt.colorbar(im2, ax=ax2, label='Rate')
    
    # 制約付きEI
    cei_grid = constrained_ei(X_grid, gp_obj_final, [gp_c1_final, gp_c2_final],
                              rate_all[final_feasible].max())
    CEI = cei_grid.reshape(Temp.shape)
    
    im3 = ax3.contourf(Temp, Pres, CEI, levels=20, cmap='plasma')
    ax3.scatter(X_all[:, 0], X_all[:, 1], c='white', s=20, alpha=0.8)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Pressure (bar)')
    ax3.set_title('Constrained EI Acquisition')
    plt.colorbar(im3, ax=ax3, label='CEI')
    
    plt.tight_layout()
    plt.savefig('pof_landscape.png', dpi=150, bbox_inches='tight')
    
    print("\n実行可能確率の統計:")
    print(f"  PoF > 0.5: {(PoF > 0.5).sum() / PoF.size * 100:.1f}%")
    print(f"  PoF > 0.9: {(PoF > 0.9).sum() / PoF.size * 100:.1f}%")
    print(f"  PoF > 0.99: {(PoF > 0.99).sum() / PoF.size * 100:.1f}%")
    

#### ⚠️ PoFの解釈

PoF = 0.9は「90%の確率で制約を満たす」を意味します。 安全クリティカルな応用では、PoF > 0.95または0.99を要求することが推奨されます。 

## 4.6 Safe Bayesian Optimization

未知の制約違反点での評価を避けるSafe BOアルゴリズム。 

### Example 5: Safe Bayesian Optimizationの実装
    
    
    # Safe Bayesian Optimization
    def safe_ucb(X, gp_obj, gp_constraints, beta_obj=2.0, beta_safe=3.0):
        """Safe UCB獲得関数
    
        Args:
            X: 候補点
            gp_obj: 目的関数GP
            gp_constraints: 制約GP
            beta_obj: 目的関数の探索パラメータ
            beta_safe: 安全性の信頼度パラメータ
        """
        # 目的関数のUCB
        mu_obj, sigma_obj = gp_obj.predict(X, return_std=True)
        ucb_obj = mu_obj + beta_obj * sigma_obj
    
        # 安全性の確率（保守的: mu - beta*sigma >= 0）
        safety_prob = np.ones(len(X))
        for gp_c in gp_constraints:
            mu_c, sigma_c = gp_c.predict(X, return_std=True)
            # Lower confidence bound が正なら安全
            lcb_c = mu_c - beta_safe * sigma_c
            safety_prob *= (lcb_c >= 0).astype(float)
    
        # 安全な点のみ選択
        ucb_obj[safety_prob == 0] = -np.inf
    
        return ucb_obj, safety_prob
    
    # Safe BO実行
    print("\nSafe Bayesian Optimization実行中...")
    
    # 初期点（既知の安全領域から開始）
    X_safe = np.array([
        [100, 40],
        [105, 45],
        [110, 50],
        [95, 35],
        [90, 40]
    ])
    rate_safe, c1_safe, c2_safe = constrained_process(X_safe)
    
    X_all_safe = X_safe.copy()
    rate_all_safe = rate_safe.copy()
    c1_all_safe = c1_safe.copy()
    c2_all_safe = c2_safe.copy()
    
    safety_violations = 0
    
    for iteration in range(20):
        # GP構築
        gp_obj_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_obj_s.fit(X_all_safe, rate_all_safe)
    
        gp_c1_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c1_s.fit(X_all_safe, c1_all_safe)
    
        gp_c2_s = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_c2_s.fit(X_all_safe, c2_all_safe)
    
        # 候補点
        X_cand = np.random.uniform([80, 30], [150, 80], (500, 2))
    
        # Safe UCB
        safe_ucb_values, safety = safe_ucb(X_cand, gp_obj_s, [gp_c1_s, gp_c2_s],
                                            beta_obj=2.0, beta_safe=3.0)
    
        if np.all(np.isinf(safe_ucb_values)):
            print(f"  警告: Iteration {iteration} - 安全な候補点なし")
            # 最も安全そうな点を選択（緊急措置）
            pof = probability_of_feasibility(X_cand, [gp_c1_s, gp_c2_s])
            next_x = X_cand[np.argmax(pof)]
        else:
            next_x = X_cand[np.argmax(safe_ucb_values)]
    
        # 実験
        next_rate, next_c1, next_c2 = constrained_process(next_x.reshape(1, -1))
    
        # 安全性チェック
        if next_c1[0] < 0 or next_c2[0] < 0:
            safety_violations += 1
            print(f"  ⚠️ 制約違反発生 (Iteration {iteration})")
    
        X_all_safe = np.vstack([X_all_safe, next_x])
        rate_all_safe = np.hstack([rate_all_safe, next_rate])
        c1_all_safe = np.hstack([c1_all_safe, next_c1])
        c2_all_safe = np.hstack([c2_all_safe, next_c2])
    
    print(f"\nSafe BO結果:")
    print(f"  総評価点数: {len(X_all_safe)}")
    print(f"  制約違反: {safety_violations}回 ({safety_violations/len(X_all_safe)*100:.1f}%)")
    
    # 通常BOと比較
    print(f"\n通常BO（参考）:")
    print(f"  制約違反: {(~final_feasible).sum()}回 ({(~final_feasible).sum()/len(X_all)*100:.1f}%)")
    
    print(f"\nSafe BOによる安全性向上: {((~final_feasible).sum() - safety_violations)}回削減")
    

#### ✅ Safe BOの利点

  * 危険領域での実験を高確率で回避
  * 安全クリティカルなプロセスに適用可能
  * 保守的な探索（β値で調整可能）

## 4.7 バッチベイズ最適化

並列実験装置を活用するため、複数点を同時に提案するバッチ最適化。 

### Example 6: バッチ獲得関数
    
    
    # Batch Bayesian Optimization
    def batch_ucb_selection(X_candidates, gp, batch_size=5, kappa=2.0, diversity_weight=0.1):
        """バッチUCB選択（逐次的貪欲法）
    
        Args:
            X_candidates: 候補点
            gp: GP model
            batch_size: バッチサイズ
            kappa: UCB探索パラメータ
            diversity_weight: 多様性重み
        """
        selected_batch = []
        remaining_candidates = X_candidates.copy()
    
        for i in range(batch_size):
            # UCB計算
            mu, sigma = gp.predict(remaining_candidates, return_std=True)
            ucb = mu + kappa * sigma
    
            # 多様性ペナルティ（既選択点から遠い点を優遇）
            if len(selected_batch) > 0:
                selected_array = np.array(selected_batch)
                for selected_x in selected_array:
                    distances = np.linalg.norm(remaining_candidates - selected_x, axis=1)
                    diversity_bonus = diversity_weight * distances.min() / distances
                    ucb += diversity_bonus
    
            # 最良点を選択
            best_idx = np.argmax(ucb)
            selected_batch.append(remaining_candidates[best_idx])
    
            # 選択済み点を除外
            remaining_candidates = np.delete(remaining_candidates, best_idx, axis=0)
    
        return np.array(selected_batch)
    
    # バッチBOの実行
    print("\nBatch Bayesian Optimization実行中...")
    
    batch_size = 5
    n_batches = 8
    
    # 初期サンプリング
    X_batch = np.random.uniform([50, 20], [150, 80], (10, 2))
    Y1_batch, Y2_batch = process_objectives(X_batch)
    
    for batch_idx in range(n_batches):
        # GP構築（目的1のみで簡略化）
        gp_batch = GaussianProcessRegressor(kernel=C(1.0)*RBF([10, 10]), normalize_y=True)
        gp_batch.fit(X_batch, Y1_batch)
    
        # バッチ選択
        X_cand = np.random.uniform([50, 20], [150, 80], (200, 2))
        next_batch = batch_ucb_selection(X_cand, gp_batch, batch_size=batch_size,
                                         kappa=2.0, diversity_weight=0.5)
    
        # 並列実験（シミュレーション）
        next_y1, next_y2 = process_objectives(next_batch)
    
        # データ追加
        X_batch = np.vstack([X_batch, next_batch])
        Y1_batch = np.hstack([Y1_batch, next_y1])
        Y2_batch = np.hstack([Y2_batch, next_y2])
    
        print(f"  Batch {batch_idx+1}/{n_batches}: Best yield = {Y1_batch.max():.2f}")
    
    # 結果可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # バッチごとの色分け
    colors = np.repeat(np.arange(n_batches + 1), [10] + [batch_size]*n_batches)
    scatter = ax1.scatter(X_batch[:, 0], X_batch[:, 1], c=colors, cmap='tab10',
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('Batch Optimization Trajectory')
    plt.colorbar(scatter, ax=ax1, label='Batch Number', ticks=range(n_batches+1))
    ax1.grid(alpha=0.3)
    
    # バッチ内の多様性（最後のバッチ）
    last_batch = X_batch[-batch_size:]
    distances = []
    for i in range(len(last_batch)):
        for j in range(i+1, len(last_batch)):
            dist = np.linalg.norm(last_batch[i] - last_batch[j])
            distances.append(dist)
    
    ax2.hist(distances, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Diversity in Last Batch ({batch_size} points)')
    ax2.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances):.2f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_optimization.png', dpi=150, bbox_inches='tight')
    
    print(f"\nバッチ最適化結果:")
    print(f"  総評価点数: {len(X_batch)}")
    print(f"  バッチ数: {n_batches}")
    print(f"  並列実験による高速化: {batch_size}倍（理論値）")
    print(f"  最良収率: {Y1_batch.max():.2f}")
    

#### 💡 バッチ選択戦略

  * **逐次貪欲法** : 1点ずつ選択、多様性ペナルティ追加
  * **並列EI** : Kriging Believerアプローチ
  * **Thompson Sampling** : 自然に多様性確保
  * **多様性重視** : クラスタリング併用

## 4.8 高次元最適化

パラメータ数が多い（10次元以上）場合の戦略。 

### Example 7: 高次元問題への対応
    
    
    # 高次元ベイズ最適化（次元削減アプローチ）
    from sklearn.decomposition import PCA
    
    def high_dimensional_process(X):
        """10次元プロセス関数
    
        Args:
            X: (N, 10) 10パラメータ
        Returns:
            y: 目的関数値
        """
        # 実際は低次元部分空間に最適解が存在（よくあるケース）
        # 有効次元: x0, x1, x5のみ
        effective_dims = X[:, [0, 1, 5]]
    
        y = (
            -np.sum((effective_dims - 0.5)**2, axis=1) +
            0.1 * np.sum(X, axis=1) +  # 他次元の小さな影響
            np.random.normal(0, 0.01, len(X))
        )
        return y
    
    # 高次元BOの戦略1: ランダム埋め込み
    print("\n高次元BO実行中...")
    
    n_dim = 10
    n_effective_dim = 3  # 推定有効次元
    
    # 初期サンプリング
    n_init = 20  # 高次元では多めに
    X_high = np.random.uniform(0, 1, (n_init, n_dim))
    y_high = high_dimensional_process(X_high)
    
    X_all_high = X_high.copy()
    y_all_high = y_high.copy()
    
    # PCAで次元削減
    pca = PCA(n_components=n_effective_dim)
    X_reduced = pca.fit_transform(X_all_high)
    
    best_values_high = [y_all_high.max()]
    
    for iteration in range(30):
        # 低次元空間でGP構築
        gp_high = GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0]*n_effective_dim),
            normalize_y=True,
            n_restarts_optimizer=5
        )
        gp_high.fit(X_reduced, y_all_high)
    
        # 低次元空間で候補生成
        X_cand_reduced = np.random.uniform(
            X_reduced.min(axis=0),
            X_reduced.max(axis=0),
            (500, n_effective_dim)
        )
    
        # UCBで選択
        mu, sigma = gp_high.predict(X_cand_reduced, return_std=True)
        ucb = mu + 2.0 * sigma
        next_x_reduced = X_cand_reduced[np.argmax(ucb)]
    
        # 元空間に逆変換
        next_x_high = pca.inverse_transform(next_x_reduced.reshape(1, -1))
    
        # 範囲チェック・クリップ
        next_x_high = np.clip(next_x_high, 0, 1)
    
        # 実験
        next_y_high = high_dimensional_process(next_x_high)
    
        # データ追加
        X_all_high = np.vstack([X_all_high, next_x_high])
        y_all_high = np.hstack([y_all_high, next_y_high])
    
        # PCA更新
        X_reduced = pca.fit_transform(X_all_high)
    
        best_values_high.append(y_all_high.max())
    
    # 比較: ランダムサーチ
    X_random = np.random.uniform(0, 1, (len(y_all_high), n_dim))
    y_random = high_dimensional_process(X_random)
    best_random = [y_random[:i+1].max() for i in range(len(y_random))]
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 収束比較
    ax1.plot(best_values_high, 'g-', linewidth=2, marker='o', markersize=4,
             label='BO with PCA (3D)')
    ax1.plot(best_random, 'gray', linewidth=2, marker='s', markersize=4,
             alpha=0.6, label='Random Search')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Value Found')
    ax1.set_title('High-Dimensional Optimization (10D)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PCA説明分散
    pca_final = PCA(n_components=n_dim)
    pca_final.fit(X_all_high)
    explained_var = pca_final.explained_variance_ratio_
    
    ax2.bar(range(1, n_dim+1), explained_var, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.plot(range(1, n_dim+1), np.cumsum(explained_var), 'ro-', linewidth=2, markersize=6,
             label='Cumulative')
    ax2.axhline(0.95, color='green', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Analysis of Search Space')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('high_dimensional_bo.png', dpi=150, bbox_inches='tight')
    
    print(f"\n高次元BO結果:")
    print(f"  次元数: {n_dim}")
    print(f"  有効次元（推定）: {n_effective_dim}")
    print(f"  最良値（BO+PCA）: {y_all_high.max():.4f}")
    print(f"  最良値（Random）: {y_random.max():.4f}")
    print(f"  改善率: {(y_all_high.max() - y_random.max())/abs(y_random.max())*100:.1f}%")
    print(f"\nPCA上位3成分の説明分散: {explained_var[:3].sum()*100:.1f}%")
    

#### ⚠️ 高次元BOの課題

次元数が増えるとGPの計算コストが急増（O(n³)）、サンプル効率も低下します。 対策: (1) 次元削減（PCA、UMAP）、(2) Random Embedding、(3) Additive GP、(4) スパース近似 

次元数 | 推奨手法 | 初期サンプル数 | 注意点  
---|---|---|---  
1-3D | 標準BO（EI, UCB） | 5-10 | 問題なし  
4-7D | 標準BO + 良いカーネル | 10-20 | 計算コスト増  
8-15D | 次元削減 + BO | 20-50 | 有効次元の推定重要  
16D+ | Random Embedding/TuRBO | 50-100 | スケーラビリティ要  
  
## まとめ

本章では、実世界の複雑な最適化問題に対応する高度な手法を学習しました。

### 重要ポイント

  * **多目的最適化** : パレートフロント探索、EHVIで複数目的を同時達成
  * **制約付き最適化** : PoFとCEIで安全な実験計画
  * **Safe BO** : 危険領域を回避しながら最適化
  * **バッチ最適化** : 並列実験で高速化（多様性確保が鍵）
  * **高次元対応** : 次元削減、Random Embeddingで実用化

### 次章の予告

第5章では、これまで学んだ技術を統合し、**実際の産業プロセス** への応用を詳しく解説します。 反応器最適化、触媒設計、品質管理など、7つの実践的ケーススタディを通じて即戦力を養います。 

[← 第3章: 獲得関数](<chapter-3.html>) [目次に戻る](<index.html>) [第5章: 産業応用 →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
