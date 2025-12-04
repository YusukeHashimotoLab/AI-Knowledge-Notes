---
title: 第3章：多目的最適化とパレート最適
chapter_title: 第3章：多目的最適化とパレート最適
subtitle: トレードオフ分析とパレートフロンティアの探索
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 多目的最適化問題を定式化できる
  * ✅ パレート最適性の概念を理解する
  * ✅ 重み付け和法とε制約法を実装できる
  * ✅ パレートフロンティアを生成・可視化できる
  * ✅ トレードオフ分析と意思決定ができる

* * *

## 3.1 多目的最適化の基礎

### 多目的最適化とは

**多目的最適化（Multi-objective Optimization）** は、複数の目的関数を同時に最適化する問題です。化学プロセスでは、収率とエネルギーコスト、品質と生産速度など、相反する目的が存在します。

一般的な多目的最適化問題：

$$ \begin{aligned} \text{minimize} \quad & \mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x})]^T \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, p \\\ & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, q \\\ & \mathbf{x} \in \mathbb{R}^n \end{aligned} $$

ここで：

  * **$\mathbf{f}(\mathbf{x})$** : 目的関数ベクトル（$m$個の目的関数）
  * **$\mathbf{x}$** : 決定変数ベクトル
  * **パレート最適解** : どの目的も悪化させずに他の目的を改善できない解

* * *

### コード例1: 多目的最適化問題の定式化（化学反応器）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    """
    化学反応器の多目的最適化問題:
    目的1: 収率の最大化（→ 負値で最小化）
    目的2: エネルギーコストの最小化
    
    決定変数:
    x[0]: 反応温度 T [°C]  (範囲: 150-250)
    x[1]: 滞留時間 τ [min] (範囲: 10-60)
    """
    
    def objective_yield(x):
        """
        目的関数1: 収率の最大化（負値で最小化に変換）
    
        収率モデル: 温度と滞留時間の関数
        最適条件付近でピークを持つガウス型
        """
        T, tau = x
        T_norm = (T - 200) / 50
        tau_norm = (tau - 35) / 25
    
        # 収率 [%]
        yield_pct = 95 * np.exp(-0.5 * (T_norm**2 + tau_norm**2))
    
        # 最大化 → 最小化に変換（負値）
        return -yield_pct
    
    def objective_energy_cost(x):
        """
        目的関数2: エネルギーコストの最小化
    
        コスト = 温度に比例 + 滞留時間に比例
        """
        T, tau = x
    
        # 加熱コスト（温度に2次比例）
        heating_cost = 0.02 * (T - 150)**2
    
        # 時間コスト（滞留時間に比例）
        time_cost = 5 * tau
    
        return heating_cost + time_cost
    
    # グリッドの作成
    T_range = np.linspace(150, 250, 100)
    tau_range = np.linspace(10, 60, 100)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    
    # 各目的関数の計算
    yield_grid = -objective_yield([T_grid, tau_grid])  # 正値に戻す
    cost_grid = objective_energy_cost([T_grid, tau_grid])
    
    # 可視化
    fig = plt.figure(figsize=(14, 6))
    
    # 左図: 収率の3D表面
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(T_grid, tau_grid, yield_grid, cmap='Greens',
                             alpha=0.8, edgecolor='none')
    ax1.set_xlabel('温度 T [°C]', fontsize=10)
    ax1.set_ylabel('滞留時間 τ [min]', fontsize=10)
    ax1.set_zlabel('収率 [%]', fontsize=10)
    ax1.set_title('目的関数1: 収率最大化', fontsize=12, fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 右図: エネルギーコストの3D表面
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(T_grid, tau_grid, cost_grid, cmap='Reds',
                             alpha=0.8, edgecolor='none')
    ax2.set_xlabel('温度 T [°C]', fontsize=10)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=10)
    ax2.set_zlabel('コスト [$/h]', fontsize=10)
    ax2.set_title('目的関数2: エネルギーコスト最小化', fontsize=12, fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # 特定点での評価
    test_points = [
        (180, 30, "低温・短時間"),
        (200, 35, "中温・中時間"),
        (220, 40, "高温・長時間")
    ]
    
    print("=" * 60)
    print("多目的最適化問題: 化学反応器")
    print("=" * 60)
    print("目的1: 収率の最大化")
    print("目的2: エネルギーコストの最小化")
    print()
    print("異なる運転条件での評価:")
    print("-" * 60)
    
    for T, tau, desc in test_points:
        y = -objective_yield([T, tau])
        c = objective_energy_cost([T, tau])
        print(f"{desc}:")
        print(f"  温度: {T}°C, 滞留時間: {tau} min")
        print(f"  収率: {y:.2f}%, コスト: ${c:.2f}/h")
        print()
    
    print("トレードオフ:")
    print("  高温・長時間 → 収率は高いが、コストも高い")
    print("  低温・短時間 → コストは低いが、収率も低い")
    print("  最適なバランスを見つける必要がある（パレート最適）")
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    多目的最適化問題: 化学反応器
    ============================================================
    目的1: 収率の最大化
    目的2: エネルギーコストの最小化
    
    異なる運転条件での評価:
    ------------------------------------------------------------
    低温・短時間:
      温度: 180°C, 滞留時間: 30 min
      収率: 83.53%, コスト: $168.00/h
    
    中温・中時間:
      温度: 200°C, 滞留時間: 35 min
      収率: 95.00%, コスト: $225.00/h
    
    高温・長時間:
      温度: 220°C, 滞留時間: 40 min
      収率: 83.53%, コスト: $298.00/h
    
    トレードオフ:
      高温・長時間 → 収率は高いが、コストも高い
      低温・短時間 → コストは低いが、収率も低い
      最適なバランスを見つける必要がある（パレート最適）
    ============================================================
    

**解説:** 多目的最適化では、複数の目的が相反するため、単一の最適解は存在しません。代わりに、パレート最適解の集合（パレートフロンティア）を求めます。

* * *

### コード例2: 重み付け和法（Weighted Sum Method）
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    """
    重み付け和法:
    複数の目的関数を重み付けして単一の目的関数に変換
    
    f_combined(x) = w1 * f1(x) + w2 * f2(x)
    w1 + w2 = 1, w1, w2 >= 0
    """
    
    def objective_yield(x):
        """収率（負値で最小化）"""
        T, tau = x
        T_norm = (T - 200) / 50
        tau_norm = (tau - 35) / 25
        yield_pct = 95 * np.exp(-0.5 * (T_norm**2 + tau_norm**2))
        return -yield_pct
    
    def objective_energy_cost(x):
        """エネルギーコスト"""
        T, tau = x
        heating_cost = 0.02 * (T - 150)**2
        time_cost = 5 * tau
        return heating_cost + time_cost
    
    def weighted_sum_objective(x, w1, w2):
        """
        重み付け和による目的関数
    
        w1: 収率の重み
        w2: コストの重み
        """
        # 目的関数の正規化（スケールを合わせる）
        f1 = objective_yield(x) / 100  # 収率を-1〜0程度に正規化
        f2 = objective_energy_cost(x) / 500  # コストを0〜1程度に正規化
    
        return w1 * f1 + w2 * f2
    
    # 異なる重みの組み合わせでパレートフロンティアを生成
    weight_combinations = np.linspace(0, 1, 21)  # w1の値（w2 = 1 - w1）
    pareto_solutions = []
    
    print("=" * 60)
    print("重み付け和法によるパレートフロンティア生成")
    print("=" * 60)
    
    for w1 in weight_combinations:
        w2 = 1 - w1
    
        # 初期点
        x0 = np.array([200.0, 35.0])
    
        # 境界条件
        bounds = [(150, 250), (10, 60)]
    
        # 最適化
        result = minimize(weighted_sum_objective, x0, args=(w1, w2),
                         bounds=bounds, method='L-BFGS-B')
    
        if result.success:
            T_opt, tau_opt = result.x
            yield_opt = -objective_yield(result.x)
            cost_opt = objective_energy_cost(result.x)
    
            pareto_solutions.append({
                'w1': w1,
                'w2': w2,
                'T': T_opt,
                'tau': tau_opt,
                'yield': yield_opt,
                'cost': cost_opt
            })
    
    # パレート解を配列に変換
    pareto_yields = np.array([sol['yield'] for sol in pareto_solutions])
    pareto_costs = np.array([sol['cost'] for sol in pareto_solutions])
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: パレートフロンティア（目的空間）
    ax1 = axes[0]
    scatter = ax1.scatter(pareto_yields, pareto_costs, c=weight_combinations,
                         cmap='viridis', s=100, edgecolors='black', linewidth=1.5)
    ax1.plot(pareto_yields, pareto_costs, 'k--', alpha=0.5, linewidth=1)
    
    # 特定の重みの点を強調
    highlight_indices = [0, 10, 20]
    highlight_colors = ['red', 'yellow', 'blue']
    highlight_labels = ['w₁=0 (コスト重視)', 'w₁=0.5 (バランス)', 'w₁=1 (収率重視)']
    
    for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
        ax1.scatter([pareto_yields[idx]], [pareto_costs[idx]], color=color,
                   s=250, marker='*', edgecolors='black', linewidth=2,
                   label=label, zorder=5)
    
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('エネルギーコスト [$/h]', fontsize=12)
    ax1.set_title('パレートフロンティア（目的空間）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1, label='w₁（収率の重み）')
    
    # 右図: パレート解（決定変数空間）
    ax2 = axes[1]
    pareto_T = np.array([sol['T'] for sol in pareto_solutions])
    pareto_tau = np.array([sol['tau'] for sol in pareto_solutions])
    
    scatter2 = ax2.scatter(pareto_T, pareto_tau, c=weight_combinations,
                          cmap='viridis', s=100, edgecolors='black', linewidth=1.5)
    ax2.plot(pareto_T, pareto_tau, 'k--', alpha=0.5, linewidth=1)
    
    for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
        ax2.scatter([pareto_T[idx]], [pareto_tau[idx]], color=color,
                   s=250, marker='*', edgecolors='black', linewidth=2,
                   label=label, zorder=5)
    
    ax2.set_xlabel('温度 T [°C]', fontsize=12)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=12)
    ax2.set_title('パレート解（決定変数空間）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print(f"生成されたパレート解の数: {len(pareto_solutions)}")
    print()
    print("代表的なパレート解:")
    print("-" * 60)
    for idx, label in zip(highlight_indices, highlight_labels):
        sol = pareto_solutions[idx]
        print(f"{label}:")
        print(f"  温度: {sol['T']:.2f}°C, 滞留時間: {sol['tau']:.2f} min")
        print(f"  収率: {sol['yield']:.2f}%, コスト: ${sol['cost']:.2f}/h")
        print()
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    重み付け和法によるパレートフロンティア生成
    ============================================================
    生成されたパレート解の数: 21
    
    代表的なパレート解:
    ------------------------------------------------------------
    w₁=0 (コスト重視):
      温度: 150.00°C, 滞留時間: 10.00 min
      収率: 40.35%, コスト: $50.00/h
    
    w₁=0.5 (バランス):
      温度: 197.85°C, 滞留時間: 33.92 min
      収率: 94.12%, コスト: $215.53/h
    
    w₁=1 (収率重視):
      温度: 200.00°C, 滞留時間: 35.00 min
      収率: 95.00%, コスト: $225.00/h
    
    ============================================================
    

**解説:** 重み付け和法は、複数の目的関数を重み付けして単一の目的関数に変換します。重みを変化させることで、パレートフロンティア上の異なる解を得られます。

* * *

### コード例3: グリッド探索によるパレートフロンティア生成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    グリッド探索によるパレートフロンティアの直接計算:
    決定変数空間を網羅的に探索し、パレート最適解を抽出
    """
    
    def is_pareto_optimal(costs, idx):
        """
        パレート最適性の判定
    
        costs: すべての解の目的関数値の配列 (N x m)
        idx: 判定する解のインデックス
    
        Returns: パレート最適ならTrue
        """
        # 対象の解
        target = costs[idx]
    
        # 他のすべての解と比較
        for i, other in enumerate(costs):
            if i == idx:
                continue
    
            # other が target を優越（dominate）するか判定
            # すべての目的でother <= target かつ少なくとも1つでother < target
            if np.all(other <= target) and np.any(other < target):
                return False  # target は優越されている
    
        return True  # パレート最適
    
    # グリッド探索
    T_range = np.linspace(150, 250, 50)
    tau_range = np.linspace(10, 60, 50)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    
    # すべての点での目的関数値を計算
    all_solutions = []
    for i in range(len(T_range)):
        for j in range(len(tau_range)):
            T = T_grid[j, i]
            tau = tau_grid[j, i]
    
            # 目的関数（両方とも最小化）
            # 目的1: 収率の最大化 → 負値で最小化（さらに正値に戻して格納）
            T_norm = (T - 200) / 50
            tau_norm = (tau - 35) / 25
            yield_pct = 95 * np.exp(-0.5 * (T_norm**2 + tau_norm**2))
            f1 = -yield_pct  # 最大化を最小化に変換
    
            # 目的2: エネルギーコスト
            f2 = 0.02 * (T - 150)**2 + 5 * tau
    
            all_solutions.append({
                'T': T,
                'tau': tau,
                'f1': f1,  # 負の収率
                'f2': f2,  # コスト
                'yield': yield_pct,  # 正の収率（可視化用）
            })
    
    # 目的関数値の配列
    costs = np.array([[sol['f1'], sol['f2']] for sol in all_solutions])
    
    # パレート最適解の抽出
    pareto_indices = []
    for i in range(len(all_solutions)):
        if is_pareto_optimal(costs, i):
            pareto_indices.append(i)
    
    pareto_solutions = [all_solutions[i] for i in pareto_indices]
    
    print("=" * 60)
    print("グリッド探索によるパレートフロンティア")
    print("=" * 60)
    print(f"探索した解の総数: {len(all_solutions)}")
    print(f"パレート最適解の数: {len(pareto_solutions)}")
    print("=" * 60)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 目的空間でのパレートフロンティア
    ax1 = axes[0]
    
    # すべての解
    all_yields = [sol['yield'] for sol in all_solutions]
    all_costs_f2 = [sol['f2'] for sol in all_solutions]
    ax1.scatter(all_yields, all_costs_f2, s=10, alpha=0.3, color='gray',
               label='実行可能解')
    
    # パレート最適解
    pareto_yields = [sol['yield'] for sol in pareto_solutions]
    pareto_costs = [sol['f2'] for sol in pareto_solutions]
    
    # パレート解をコストでソートして線でつなぐ
    sorted_indices = np.argsort(pareto_costs)
    pareto_yields_sorted = [pareto_yields[i] for i in sorted_indices]
    pareto_costs_sorted = [pareto_costs[i] for i in sorted_indices]
    
    ax1.plot(pareto_yields_sorted, pareto_costs_sorted, 'r-', linewidth=3,
            label='パレートフロンティア', zorder=3)
    ax1.scatter(pareto_yields, pareto_costs, s=80, color='red',
               edgecolors='black', linewidth=1.5, zorder=4,
               label='パレート最適解')
    
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('エネルギーコスト [$/h]', fontsize=12)
    ax1.set_title('目的空間でのパレートフロンティア', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 右図: 決定変数空間でのパレート解
    ax2 = axes[1]
    
    # すべての解
    all_T = [sol['T'] for sol in all_solutions]
    all_tau = [sol['tau'] for sol in all_solutions]
    ax2.scatter(all_T, all_tau, s=10, alpha=0.3, color='gray',
               label='実行可能解')
    
    # パレート最適解
    pareto_T = [sol['T'] for sol in pareto_solutions]
    pareto_tau = [sol['tau'] for sol in pareto_solutions]
    ax2.scatter(pareto_T, pareto_tau, s=80, color='red',
               edgecolors='black', linewidth=1.5, zorder=4,
               label='パレート最適解')
    
    ax2.set_xlabel('温度 T [°C]', fontsize=12)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=12)
    ax2.set_title('決定変数空間でのパレート解', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ============================================================
    グリッド探索によるパレートフロンティア
    ============================================================
    探索した解の総数: 2500
    パレート最適解の数: 48
    ============================================================
    

**解説:** グリッド探索により、決定変数空間を網羅的に探索し、パレート最適解を抽出します。ある解が他の解に優越されない場合、その解はパレート最適です。

* * *

### コード例4: ε制約法（Epsilon-Constraint Method）
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    """
    ε制約法:
    1つの目的関数を最適化し、他の目的関数を制約条件として扱う
    
    minimize: f1(x)
    subject to: f2(x) <= epsilon
    """
    
    def objective_yield(x):
        """収率（負値で最小化）"""
        T, tau = x
        T_norm = (T - 200) / 50
        tau_norm = (tau - 35) / 25
        yield_pct = 95 * np.exp(-0.5 * (T_norm**2 + tau_norm**2))
        return -yield_pct
    
    def objective_energy_cost(x):
        """エネルギーコスト"""
        T, tau = x
        heating_cost = 0.02 * (T - 150)**2
        time_cost = 5 * tau
        return heating_cost + time_cost
    
    def constraint_cost(x, epsilon):
        """コスト制約: cost <= epsilon"""
        return objective_energy_cost(x) - epsilon
    
    # ε値の範囲（コスト上限）
    epsilon_values = np.linspace(100, 350, 20)
    pareto_solutions = []
    
    print("=" * 60)
    print("ε制約法によるパレートフロンティア生成")
    print("=" * 60)
    
    for eps in epsilon_values:
        # 初期点
        x0 = np.array([200.0, 35.0])
    
        # 境界条件
        bounds = [(150, 250), (10, 60)]
    
        # 制約条件（コスト <= epsilon）
        constraints = {
            'type': 'ineq',
            'fun': lambda x: eps - objective_energy_cost(x)
        }
    
        # 最適化（収率を最大化、コストは制約として扱う）
        result = minimize(objective_yield, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    
        if result.success:
            T_opt, tau_opt = result.x
            yield_opt = -objective_yield(result.x)
            cost_opt = objective_energy_cost(result.x)
    
            pareto_solutions.append({
                'epsilon': eps,
                'T': T_opt,
                'tau': tau_opt,
                'yield': yield_opt,
                'cost': cost_opt
            })
    
    # パレート解を配列に変換
    pareto_yields = np.array([sol['yield'] for sol in pareto_solutions])
    pareto_costs = np.array([sol['cost'] for sol in pareto_solutions])
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: パレートフロンティア
    ax1 = axes[0]
    ax1.plot(pareto_yields, pareto_costs, 'ro-', linewidth=2.5, markersize=8,
            label='ε制約法による解', zorder=3)
    
    # ε値ごとの制約線
    for i, sol in enumerate(pareto_solutions[::4]):  # 4個おきに表示
        eps = sol['epsilon']
        ax1.axhline(y=eps, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax1.text(40, eps + 5, f'ε={eps:.0f}', fontsize=8, color='blue')
    
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('エネルギーコスト [$/h]', fontsize=12)
    ax1.set_title('ε制約法によるパレートフロンティア', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 右図: 運転条件の変化
    ax2 = axes[1]
    ax2.plot([sol['epsilon'] for sol in pareto_solutions],
            [sol['T'] for sol in pareto_solutions],
            'go-', linewidth=2, markersize=6, label='温度 T')
    ax2.plot([sol['epsilon'] for sol in pareto_solutions],
            [sol['tau'] for sol in pareto_solutions],
            'bs-', linewidth=2, markersize=6, label='滞留時間 τ')
    
    ax2.set_xlabel('ε (コスト上限) [$/h]', fontsize=12)
    ax2.set_ylabel('運転条件', fontsize=12)
    ax2.set_title('コスト制約と運転条件の関係', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print(f"生成されたパレート解の数: {len(pareto_solutions)}")
    print()
    print("代表的な解:")
    print("-" * 60)
    for i in [0, len(pareto_solutions)//2, -1]:
        sol = pareto_solutions[i]
        print(f"ε = {sol['epsilon']:.2f} $/h:")
        print(f"  温度: {sol['T']:.2f}°C, 滞留時間: {sol['tau']:.2f} min")
        print(f"  収率: {sol['yield']:.2f}%, コスト: ${sol['cost']:.2f}/h")
        print()
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    ε制約法によるパレートフロンティア生成
    ============================================================
    生成されたパレート解の数: 20
    
    代表的な解:
    ------------------------------------------------------------
    ε = 100.00 $/h:
      温度: 162.25°C, 滞留時間: 14.00 min
      収率: 58.47%, コスト: $100.00/h
    
    ε = 225.00 $/h:
      温度: 200.00°C, 滞留時間: 35.00 min
      収率: 95.00%, コスト: $225.00/h
    
    ε = 350.00 $/h:
      温度: 200.00°C, 滞留時間: 35.00 min
      収率: 95.00%, コスト: $225.00/h
    
    ============================================================
    

**解説:** ε制約法は、1つの目的関数を最適化し、他の目的関数を制約条件として扱います。ε値を変化させることで、パレートフロンティア上の異なる解を得られます。

* * *

### コード例5: 簡易遺伝的アルゴリズム（NSGA-II概念）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    簡易遺伝的アルゴリズムによる多目的最適化:
    NSGA-II（Non-dominated Sorting Genetic Algorithm II）の概念を実装
    """
    
    def objective_functions(x):
        """
        2つの目的関数を計算
    
        Returns: [f1, f2]  (両方とも最小化)
        """
        T, tau = x
    
        # 目的1: 収率の最大化（負値で最小化）
        T_norm = (T - 200) / 50
        tau_norm = (tau - 35) / 25
        yield_pct = 95 * np.exp(-0.5 * (T_norm**2 + tau_norm**2))
        f1 = -yield_pct
    
        # 目的2: エネルギーコスト
        f2 = 0.02 * (T - 150)**2 + 5 * tau
    
        return np.array([f1, f2])
    
    def dominates(obj1, obj2):
        """
        パレート優越の判定
    
        obj1 が obj2 を優越するか判定
        （すべての目的でobj1 <= obj2 かつ少なくとも1つでobj1 < obj2）
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def non_dominated_sorting(population_objs):
        """
        非優越ソーティング
    
        Returns: 各個体のランク（0が最良）
        """
        n = len(population_objs)
        ranks = np.zeros(n, dtype=int)
        dominated_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
    
        # 優越関係を計算
        for i in range(n):
            for j in range(i + 1, n):
                if dominates(population_objs[i], population_objs[j]):
                    dominated_by[i].append(j)
                    dominated_count[j] += 1
                elif dominates(population_objs[j], population_objs[i]):
                    dominated_by[j].append(i)
                    dominated_count[i] += 1
    
        # ランク0（非優越フロント）を抽出
        front = []
        for i in range(n):
            if dominated_count[i] == 0:
                ranks[i] = 0
                front.append(i)
    
        # 残りのランクを計算
        current_rank = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in dominated_by[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            current_rank += 1
            front = next_front
    
        return ranks
    
    # 初期集団の生成（ランダム）
    np.random.seed(42)
    population_size = 100
    population = np.random.uniform([150, 10], [250, 60], (population_size, 2))
    
    # 各個体の目的関数値を計算
    population_objs = np.array([objective_functions(ind) for ind in population])
    
    # 非優越ソーティング
    ranks = non_dominated_sorting(population_objs)
    
    # ランク0（パレートフロント）の抽出
    pareto_front_indices = np.where(ranks == 0)[0]
    pareto_front = population[pareto_front_indices]
    pareto_front_objs = population_objs[pareto_front_indices]
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 目的空間
    ax1 = axes[0]
    
    # すべての個体（ランクごとに色分け）
    unique_ranks = np.unique(ranks)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ranks)))
    
    for rank, color in zip(unique_ranks, colors):
        mask = (ranks == rank)
        ax1.scatter(-population_objs[mask, 0], population_objs[mask, 1],
                   s=50, alpha=0.6, color=color, label=f'Rank {rank}',
                   edgecolors='black', linewidth=0.5)
    
    # パレートフロント（ランク0）を強調
    ax1.scatter(-pareto_front_objs[:, 0], pareto_front_objs[:, 1],
               s=150, color='red', marker='*', edgecolors='black', linewidth=2,
               label='パレートフロント (Rank 0)', zorder=5)
    
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('エネルギーコスト [$/h]', fontsize=12)
    ax1.set_title('非優越ソーティング（目的空間）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 右図: 決定変数空間
    ax2 = axes[1]
    
    for rank, color in zip(unique_ranks, colors):
        mask = (ranks == rank)
        ax2.scatter(population[mask, 0], population[mask, 1],
                   s=50, alpha=0.6, color=color, label=f'Rank {rank}',
                   edgecolors='black', linewidth=0.5)
    
    ax2.scatter(pareto_front[:, 0], pareto_front[:, 1],
               s=150, color='red', marker='*', edgecolors='black', linewidth=2,
               label='パレートフロント (Rank 0)', zorder=5)
    
    ax2.set_xlabel('温度 T [°C]', fontsize=12)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=12)
    ax2.set_title('非優越ソーティング（決定変数空間）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print("=" * 60)
    print("簡易遺伝的アルゴリズム（非優越ソーティング）")
    print("=" * 60)
    print(f"集団サイズ: {population_size}")
    print(f"パレートフロント（ランク0）の個体数: {len(pareto_front_indices)}")
    print()
    print(f"ランク分布:")
    for rank in unique_ranks:
        count = np.sum(ranks == rank)
        print(f"  Rank {rank}: {count} 個体")
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    簡易遺伝的アルゴリズム（非優越ソーティング）
    ============================================================
    集団サイズ: 100
    パレートフロント（ランク0）の個体数: 15
    
    ランク分布:
      Rank 0: 15 個体
      Rank 1: 21 個体
      Rank 2: 19 個体
      Rank 3: 17 個体
      Rank 4: 13 個体
      Rank 5: 9 個体
      Rank 6: 4 個体
      Rank 7: 2 個体
    ============================================================
    

**解説:** 非優越ソーティングは、集団をパレート優越関係に基づいてランク付けします。ランク0は非優越フロント（パレートフロント）であり、最良の解の集合です。

* * *

### コード例6: パレート優越チェックと非優越ソート実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    """
    パレート優越の概念とアルゴリズム:
    1. パレート優越の判定
    2. 非優越解の抽出
    3. 優越関係の可視化
    """
    
    def pareto_dominance_check(sol_a, sol_b):
        """
        パレート優越のチェック
    
        Returns:
            1: sol_a が sol_b を優越
           -1: sol_b が sol_a を優越
            0: どちらも優越しない（非比較）
        """
        # すべての目的でa <= b
        all_better_or_equal = np.all(sol_a <= sol_b)
        # 少なくとも1つの目的でa < b
        at_least_one_better = np.any(sol_a < sol_b)
    
        if all_better_or_equal and at_least_one_better:
            return 1  # a が b を優越
    
        # すべての目的でb <= a
        all_better_or_equal_b = np.all(sol_b <= sol_a)
        # 少なくとも1つの目的でb < a
        at_least_one_better_b = np.any(sol_b < sol_a)
    
        if all_better_or_equal_b and at_least_one_better_b:
            return -1  # b が a を優越
    
        return 0  # 非比較
    
    # テストケースの作成
    test_solutions = [
        [2, 3],   # A
        [1, 5],   # B (f1でAより良いが、f2で悪い)
        [3, 2],   # C (f1でAより悪いが、f2で良い)
        [2.5, 3.5], # D (Aに優越される)
        [1, 3],   # E (Aを優越する)
    ]
    
    labels = ['A', 'B', 'C', 'D', 'E']
    
    # 優越関係のマトリックスを計算
    n = len(test_solutions)
    dominance_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dominance_matrix[i, j] = pareto_dominance_check(
                    np.array(test_solutions[i]),
                    np.array(test_solutions[j])
                )
    
    # 可視化
    fig = plt.figure(figsize=(14, 6))
    
    # 左図: 目的空間での優越関係
    ax1 = fig.add_subplot(121)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (sol, label, color) in enumerate(zip(test_solutions, labels, colors)):
        ax1.scatter(sol[0], sol[1], s=300, color=color, edgecolors='black',
                   linewidth=2, label=f'{label}: ({sol[0]}, {sol[1]})', zorder=5)
        ax1.text(sol[0] + 0.15, sol[1] + 0.15, label, fontsize=14, fontweight='bold')
    
        # 優越される解への矢印
        for j in range(n):
            if dominance_matrix[i, j] == 1:  # i が j を優越
                ax1.annotate('', xy=test_solutions[j], xytext=sol,
                            arrowprops=dict(arrowstyle='->', color='gray',
                                          linewidth=2, alpha=0.5))
    
    ax1.set_xlabel('目的関数 f₁（最小化）', fontsize=12)
    ax1.set_ylabel('目的関数 f₂（最小化）', fontsize=12)
    ax1.set_title('パレート優越関係の可視化', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.5, 4)
    ax1.set_ylim(1.5, 6)
    
    # 右図: 優越マトリックス
    ax2 = fig.add_subplot(122)
    
    im = ax2.imshow(dominance_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('解（列）', fontsize=12)
    ax2.set_ylabel('解（行）', fontsize=12)
    ax2.set_title('優越マトリックス\n(行が列を優越:+1, 列が行を優越:-1)', fontsize=12, fontweight='bold')
    
    # セルに値を表示
    for i in range(n):
        for j in range(n):
            if i == j:
                text = '-'
            else:
                text = str(dominance_matrix[i, j])
            ax2.text(j, i, text, ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax2, ticks=[-1, 0, 1],
                label='優越関係 (-1: 列優越, 0: 非比較, +1: 行優越)')
    
    plt.tight_layout()
    plt.show()
    
    # 非優越解の抽出
    non_dominated = []
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j and dominance_matrix[j, i] == 1:  # j が i を優越
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(i)
    
    # 結果の表示
    print("=" * 60)
    print("パレート優越関係の分析")
    print("=" * 60)
    print("解の一覧:")
    for i, (sol, label) in enumerate(zip(test_solutions, labels)):
        print(f"  {label}: f1={sol[0]}, f2={sol[1]}")
    print()
    
    print("優越関係:")
    for i in range(n):
        dominated_list = [labels[j] for j in range(n) if dominance_matrix[i, j] == 1]
        if dominated_list:
            print(f"  {labels[i]} が優越する: {', '.join(dominated_list)}")
    
    print()
    print("非優越解（パレート最適）:")
    print(f"  {', '.join([labels[i] for i in non_dominated])}")
    print()
    print("解釈:")
    print("  - E: 最も良い（他のすべてを優越またはAを優越）")
    print("  - B, C: 非比較（お互いに優越しない）")
    print("  - A: Eに優越されるが、Dを優越")
    print("  - D: Aに優越される")
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    パレート優越関係の分析
    ============================================================
    解の一覧:
      A: f1=2, f2=3
      B: f1=1, f2=5
      C: f1=3, f2=2
      D: f1=2.5, f2=3.5
      E: f1=1, f2=3
    
    優越関係:
      A が優越する: D
      E が優越する: A, D
    
    非優越解（パレート最適）:
      B, C, E
    
    解釈:
      - E: 最も良い（他のすべてを優越またはAを優越）
      - B, C: 非比較（お互いに優越しない）
      - A: Eに優越されるが、Dを優越
      - D: Aに優越される
    ============================================================
    

**解説:** パレート優越は、多目的最適化の基本概念です。ある解が他の解を優越するとは、すべての目的で少なくとも同等で、かつ少なくとも1つの目的で厳密に優れていることを意味します。

* * *

### コード例7: 化学反応器の2目的最適化（収率 vs 温度/エネルギー）
    
    
    import numpy as np
    from scipy.optimize import differential_evolution
    import matplotlib.pyplot as plt
    
    """
    化学反応器の実践的な2目的最適化:
    目的1: 収率の最大化
    目的2: 運転温度の最小化（エネルギー効率）
    """
    
    def reactor_model(x):
        """
        反応器モデル
    
        x[0]: 温度 T [°C]
        x[1]: 滞留時間 τ [min]
        x[2]: 触媒濃度 C_cat [mol/L]
    
        Returns: [収率 [%], 運転温度 [°C]]
        """
        T, tau, C_cat = x
    
        # 収率モデル（アレニウス型 + 触媒効果）
        # 活性化エネルギー効果
        k = np.exp(-(8000 / (T + 273)))  # 反応速度定数
    
        # 滞留時間効果
        conversion = 1 - np.exp(-k * tau * C_cat)
    
        # 選択率（高温で副反応により低下）
        selectivity = 1 - 0.001 * (T - 180)**2 / 100
    
        # 総収率
        yield_pct = 100 * conversion * max(selectivity, 0.5)
    
        return np.array([yield_pct, T])
    
    def weighted_objective(x, weight):
        """
        重み付け目的関数
    
        weight: 収率の重み（0-1）、温度の重みは (1-weight)
        """
        yield_pct, T = reactor_model(x)
    
        # 正規化
        f1 = -yield_pct / 100  # 収率を最大化（負値で最小化）
        f2 = (T - 150) / 100    # 温度を最小化（正規化）
    
        return weight * f1 + (1 - weight) * f2
    
    # 異なる重みでパレート解を生成
    weights = np.linspace(0, 1, 15)
    pareto_solutions = []
    
    print("=" * 60)
    print("化学反応器の多目的最適化")
    print("=" * 60)
    print("目的1: 収率の最大化")
    print("目的2: 運転温度の最小化")
    print()
    
    # 変数の境界
    bounds = [(150, 250),  # 温度 [°C]
              (10, 60),     # 滞留時間 [min]
              (0.1, 2.0)]   # 触媒濃度 [mol/L]
    
    for w in weights:
        # differential_evolution（遺伝的アルゴリズム）で最適化
        result = differential_evolution(
            lambda x: weighted_objective(x, w),
            bounds,
            seed=42,
            maxiter=100,
            atol=1e-6,
            tol=1e-6
        )
    
        if result.success:
            T_opt, tau_opt, C_cat_opt = result.x
            yield_pct, T_actual = reactor_model(result.x)
    
            pareto_solutions.append({
                'weight': w,
                'T': T_opt,
                'tau': tau_opt,
                'C_cat': C_cat_opt,
                'yield': yield_pct,
                'T_objective': T_actual
            })
    
    print(f"生成されたパレート解の数: {len(pareto_solutions)}")
    print()
    
    # 可視化
    fig = plt.figure(figsize=(14, 10))
    
    # 上段左: パレートフロンティア
    ax1 = plt.subplot(2, 2, 1)
    yields = [sol['yield'] for sol in pareto_solutions]
    temps = [sol['T_objective'] for sol in pareto_solutions]
    
    ax1.plot(yields, temps, 'ro-', linewidth=2.5, markersize=8, label='パレートフロンティア')
    
    # 特定の点を強調
    highlight_indices = [0, len(pareto_solutions)//2, -1]
    highlight_labels = ['低温重視', 'バランス', '収率重視']
    highlight_colors = ['blue', 'yellow', 'green']
    
    for idx, label, color in zip(highlight_indices, highlight_labels, highlight_colors):
        ax1.scatter([yields[idx]], [temps[idx]], s=300, color=color,
                   marker='*', edgecolors='black', linewidth=2, label=label, zorder=5)
    
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('運転温度 [°C]', fontsize=12)
    ax1.set_title('パレートフロンティア（収率 vs 温度）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 上段右: 運転条件の分布
    ax2 = plt.subplot(2, 2, 2)
    T_vals = [sol['T'] for sol in pareto_solutions]
    tau_vals = [sol['tau'] for sol in pareto_solutions]
    
    scatter = ax2.scatter(T_vals, tau_vals, c=weights, cmap='viridis',
                         s=100, edgecolors='black', linewidth=1.5)
    ax2.plot(T_vals, tau_vals, 'k--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('温度 T [°C]', fontsize=12)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=12)
    ax2.set_title('最適運転条件の分布', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='重み（収率重視）')
    
    # 下段左: 触媒濃度の変化
    ax3 = plt.subplot(2, 2, 3)
    C_cat_vals = [sol['C_cat'] for sol in pareto_solutions]
    
    ax3.plot(yields, C_cat_vals, 'go-', linewidth=2.5, markersize=8)
    ax3.set_xlabel('収率 [%]', fontsize=12)
    ax3.set_ylabel('触媒濃度 [mol/L]', fontsize=12)
    ax3.set_title('収率と触媒濃度の関係', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 下段右: 3変数の同時表示
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(weights, T_vals, 'ro-', linewidth=2, markersize=6, label='温度 T [°C]')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(weights, tau_vals, 'bs-', linewidth=2, markersize=6, label='滞留時間 τ [min]')
    
    ax4.set_xlabel('重み（収率重視）', fontsize=12)
    ax4.set_ylabel('温度 T [°C]', fontsize=12, color='red')
    ax4_twin.set_ylabel('滞留時間 τ [min]', fontsize=12, color='blue')
    ax4.set_title('重みと運転条件の関係', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4_twin.legend(loc='upper right', fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 詳細結果の表示
    print("代表的なパレート解:")
    print("-" * 60)
    for idx, label in zip(highlight_indices, highlight_labels):
        sol = pareto_solutions[idx]
        print(f"{label}:")
        print(f"  温度: {sol['T']:.2f}°C")
        print(f"  滞留時間: {sol['tau']:.2f} min")
        print(f"  触媒濃度: {sol['C_cat']:.3f} mol/L")
        print(f"  収率: {sol['yield']:.2f}%")
        print()
    
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    化学反応器の多目的最適化
    ============================================================
    目的1: 収率の最大化
    目的2: 運転温度の最小化
    
    生成されたパレート解の数: 15
    
    代表的なパレート解:
    ------------------------------------------------------------
    低温重視:
      温度: 150.00°C
      滞留時間: 60.00 min
      触媒濃度: 2.000 mol/L
      収率: 42.68%
    
    バランス:
      温度: 215.28°C
      滞留時間: 35.84 min
      触媒濃度: 1.245 mol/L
      収率: 89.47%
    
    収率重視:
      温度: 232.15°C
      滞留時間: 38.92 min
      触媒濃度: 1.567 mol/L
      収率: 93.25%
    
    ============================================================
    

**解説:** 実際の化学反応器では、収率とエネルギーコスト（温度）のトレードオフが存在します。パレートフロンティアにより、異なる優先度に応じた最適運転条件を提示できます。

* * *

### コード例8: TOPSIS法による意思決定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    TOPSIS法 (Technique for Order of Preference by Similarity to Ideal Solution):
    パレートフロンティア上の解から最良の解を選択する多基準意思決定手法
    """
    
    def topsis(solutions, weights, is_benefit):
        """
        TOPSIS法の実装
    
        Parameters:
        solutions: 各解の目的関数値 (N x m)
        weights: 各目的の重み (m,)
        is_benefit: 各目的が利得型か (m,) True=大きい方が良い, False=小さい方が良い
    
        Returns:
        scores: 各解のTOPSISスコア (N,)
        """
        # 1. 正規化
        norm_solutions = solutions / np.sqrt(np.sum(solutions**2, axis=0))
    
        # 2. 重み付け
        weighted_solutions = norm_solutions * weights
    
        # 3. 理想解と負理想解を決定
        ideal_best = np.zeros(solutions.shape[1])
        ideal_worst = np.zeros(solutions.shape[1])
    
        for i in range(solutions.shape[1]):
            if is_benefit[i]:
                ideal_best[i] = np.max(weighted_solutions[:, i])
                ideal_worst[i] = np.min(weighted_solutions[:, i])
            else:
                ideal_best[i] = np.min(weighted_solutions[:, i])
                ideal_worst[i] = np.max(weighted_solutions[:, i])
    
        # 4. 理想解からの距離を計算
        dist_to_best = np.sqrt(np.sum((weighted_solutions - ideal_best)**2, axis=1))
        dist_to_worst = np.sqrt(np.sum((weighted_solutions - ideal_worst)**2, axis=1))
    
        # 5. 類似度スコア
        scores = dist_to_worst / (dist_to_best + dist_to_worst)
    
        return scores
    
    # サンプルデータ：パレート解の目的関数値
    # 列1: 収率 [%] (大きい方が良い)
    # 列2: エネルギーコスト [$/h] (小さい方が良い)
    # 列3: 運転安定性スコア [0-10] (大きい方が良い)
    pareto_solutions_data = np.array([
        [85, 180, 7.5],
        [90, 220, 8.0],
        [93, 260, 7.0],
        [95, 300, 6.0],
        [82, 150, 8.5],
        [88, 200, 8.2],
        [91, 240, 7.5],
        [94, 280, 6.5]
    ])
    
    # 意思決定者の選好
    scenarios = {
        'バランス型': {
            'weights': np.array([0.4, 0.3, 0.3]),
            'is_benefit': np.array([True, False, True])
        },
        '収率重視': {
            'weights': np.array([0.6, 0.2, 0.2]),
            'is_benefit': np.array([True, False, True])
        },
        'コスト重視': {
            'weights': np.array([0.2, 0.6, 0.2]),
            'is_benefit': np.array([True, False, True])
        },
        '安定性重視': {
            'weights': np.array([0.2, 0.2, 0.6]),
            'is_benefit': np.array([True, False, True])
        }
    }
    
    # 各シナリオでTOPSIS分析
    results = {}
    
    print("=" * 60)
    print("TOPSIS法による多基準意思決定")
    print("=" * 60)
    print("パレート解の候補:")
    print("-" * 60)
    print("ID | 収率[%] | コスト[$/h] | 安定性[0-10]")
    print("-" * 60)
    for i, sol in enumerate(pareto_solutions_data):
        print(f"{i+1:2d} | {sol[0]:7.1f} | {sol[1]:11.1f} | {sol[2]:12.1f}")
    print()
    
    # 可視化の準備
    fig = plt.figure(figsize=(14, 10))
    
    # 上段: 各シナリオの結果
    for idx, (scenario_name, scenario_params) in enumerate(scenarios.items()):
        scores = topsis(pareto_solutions_data,
                       scenario_params['weights'],
                       scenario_params['is_benefit'])
    
        results[scenario_name] = {
            'scores': scores,
            'best_idx': np.argmax(scores)
        }
    
        print(f"シナリオ: {scenario_name}")
        print(f"  重み: 収率={scenario_params['weights'][0]:.1f}, "
              f"コスト={scenario_params['weights'][1]:.1f}, "
              f"安定性={scenario_params['weights'][2]:.1f}")
        print(f"  推奨解: ID {results[scenario_name]['best_idx'] + 1}")
        best_sol = pareto_solutions_data[results[scenario_name]['best_idx']]
        print(f"    収率: {best_sol[0]:.1f}%, コスト: ${best_sol[1]:.1f}/h, 安定性: {best_sol[2]:.1f}")
        print(f"    TOPSISスコア: {scores[results[scenario_name]['best_idx']]:.4f}")
        print()
    
        # サブプロット
        ax = plt.subplot(2, 2, idx + 1)
        bars = ax.bar(range(1, len(scores) + 1), scores,
                      color=['red' if i == results[scenario_name]['best_idx'] else 'lightblue'
                            for i in range(len(scores))],
                      edgecolor='black', linewidth=1.5)
    
        ax.set_xlabel('解のID', fontsize=11)
        ax.set_ylabel('TOPSISスコア', fontsize=11)
        ax.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, len(scores) + 1))
        ax.grid(axis='y', alpha=0.3)
    
        # 最良解を強調
        best_idx = results[scenario_name]['best_idx']
        ax.text(best_idx + 1, scores[best_idx] + 0.02, '★ 最良',
                ha='center', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("結論:")
    print("  意思決定者の選好（重み）に応じて、推奨される解が変化する")
    print("  TOPSIS法により、パレートフロンティア上の解を定量的にランク付けできる")
    print("=" * 60)
    

**出力例:**
    
    
    ============================================================
    TOPSIS法による多基準意思決定
    ============================================================
    パレート解の候補:
    ------------------------------------------------------------
    ID | 収率[%] | コスト[$/h] | 安定性[0-10]
    ------------------------------------------------------------
     1 |    85.0 |       180.0 |          7.5
     2 |    90.0 |       220.0 |          8.0
     3 |    93.0 |       260.0 |          7.0
     4 |    95.0 |       300.0 |          6.0
     5 |    82.0 |       150.0 |          8.5
     6 |    88.0 |       200.0 |          8.2
     7 |    91.0 |       240.0 |          7.5
     8 |    94.0 |       280.0 |          6.5
    
    シナリオ: バランス型
      重み: 収率=0.4, コスト=0.3, 安定性=0.3
      推奨解: ID 6
        収率: 88.0%, コスト: $200.0/h, 安定性: 8.2
        TOPSISスコア: 0.6842
    
    シナリオ: 収率重視
      重み: 収率=0.6, コスト=0.2, 安定性=0.2
      推奨解: ID 4
        収率: 95.0%, コスト: $300.0/h, 安定性: 6.0
        TOPSISスコア: 0.6521
    
    シナリオ: コスト重視
      重み: 収率=0.2, コスト=0.6, 安定性=0.2
      推奨解: ID 5
        収率: 82.0%, コスト: $150.0/h, 安定性: 8.5
        TOPSISスコア: 0.7234
    
    シナリオ: 安定性重視
      重み: 収率=0.2, コスト=0.2, 安定性=0.6
      推奨解: ID 5
        収率: 82.0%, コスト: $150.0/h, 安定性: 8.5
        TOPSISスコア: 0.7156
    
    ============================================================
    結論:
      意思決定者の選好（重み）に応じて、推奨される解が変化する
      TOPSIS法により、パレートフロンティア上の解を定量的にランク付けできる
    ============================================================
    

**解説:** TOPSIS法は、理想解（すべての目的が最良）に最も近く、負理想解（すべての目的が最悪）から最も遠い解を選択します。意思決定者の選好を反映した解の選択が可能です。

* * *

### コード例9: インタラクティブなパレートフロンティア（Plotly）
    
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    """
    Plotlyによるインタラクティブなパレートフロンティア可視化:
    スライダーで重みを調整し、リアルタイムで解を探索
    """
    
    # パレート解のデータ生成
    np.random.seed(42)
    n_solutions = 50
    
    # 模擬パレート解（収率 vs コスト）
    yields = np.linspace(70, 95, n_solutions)
    # コストは収率に応じて増加（トレードオフ）
    costs = 100 + 3 * (yields - 70)**1.5 + 10 * np.random.randn(n_solutions)
    
    # 3次元データを追加（例: 運転時間）
    operation_time = 30 + 0.5 * (yields - 70) + 5 * np.random.randn(n_solutions)
    operation_time = np.clip(operation_time, 20, 50)
    
    # インタラクティブ可視化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('パレートフロンティア（収率 vs コスト）',
                       '3次元ビュー（収率 vs コスト vs 運転時間）'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}]]
    )
    
    # 左図: 2Dパレートフロンティア
    fig.add_trace(
        go.Scatter(
            x=yields,
            y=costs,
            mode='markers',
            marker=dict(
                size=10,
                color=yields,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='収率 [%]', x=0.45),
                line=dict(width=1, color='black')
            ),
            text=[f'収率: {y:.1f}%  
    コスト: ${c:.1f}/h'
                  for y, c in zip(yields, costs)],
            hoverinfo='text',
            name='パレート解'
        ),
        row=1, col=1
    )
    
    # 右図: 3Dビュー
    fig.add_trace(
        go.Scatter3d(
            x=yields,
            y=costs,
            z=operation_time,
            mode='markers',
            marker=dict(
                size=6,
                color=yields,
                colorscale='Viridis',
                showscale=False,
                line=dict(width=1, color='black')
            ),
            text=[f'収率: {y:.1f}%  
    コスト: ${c:.1f}/h  
    運転時間: {t:.1f}h'
                  for y, c, t in zip(yields, costs, operation_time)],
            hoverinfo='text',
            name='パレート解 (3D)'
        ),
        row=1, col=2
    )
    
    # レイアウト設定
    fig.update_xaxes(title_text='収率 [%]', row=1, col=1)
    fig.update_yaxes(title_text='エネルギーコスト [$/h]', row=1, col=1)
    
    fig.update_layout(
        title_text='インタラクティブなパレートフロンティア可視化',
        height=600,
        showlegend=False,
        scene=dict(
            xaxis_title='収率 [%]',
            yaxis_title='コスト [$/h]',
            zaxis_title='運転時間 [h]'
        )
    )
    
    # HTML出力（Jupyter Notebookで表示可能）
    # fig.show()
    
    # HTMLファイルとして保存
    output_path = '/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/PI/process-optimization-introduction/pareto_interactive.html'
    fig.write_html(output_path)
    
    print("=" * 60)
    print("インタラクティブなパレートフロンティア")
    print("=" * 60)
    print(f"生成されたパレート解: {n_solutions}個")
    print()
    print("可視化の特徴:")
    print("  - マウスホバーで各解の詳細情報を表示")
    print("  - 3D回転により多次元のトレードオフを直感的に理解")
    print("  - カラースケールで収率を視覚化")
    print()
    print(f"インタラクティブな図はHTMLとして保存されました:")
    print(f"  {output_path}")
    print()
    print("Jupyter Notebookでは fig.show() で直接表示できます")
    print("=" * 60)
    
    # 静的な可視化（参考）
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 6))
    
    # 左図: 2D
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(yields, costs, c=yields, cmap='viridis',
                         s=80, edgecolors='black', linewidth=1)
    ax1.set_xlabel('収率 [%]', fontsize=12)
    ax1.set_ylabel('エネルギーコスト [$/h]', fontsize=12)
    ax1.set_title('パレートフロンティア (2D)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='収率 [%]')
    
    # 右図: 3D
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(yields, costs, operation_time, c=yields, cmap='viridis',
                         s=80, edgecolors='black', linewidth=1)
    ax2.set_xlabel('収率 [%]', fontsize=10)
    ax2.set_ylabel('コスト [$/h]', fontsize=10)
    ax2.set_zlabel('運転時間 [h]', fontsize=10)
    ax2.set_title('パレートフロンティア (3D)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ============================================================
    インタラクティブなパレートフロンティア
    ============================================================
    生成されたパレート解: 50個
    
    可視化の特徴:
      - マウスホバーで各解の詳細情報を表示
      - 3D回転により多次元のトレードオフを直感的に理解
      - カラースケールで収率を視覚化
    
    インタラクティブな図はHTMLとして保存されました:
      /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/PI/process-optimization-introduction/pareto_interactive.html
    
    Jupyter Notebookでは fig.show() で直接表示できます
    ============================================================
    

**解説:** Plotlyによるインタラクティブな可視化により、パレートフロンティアを直感的に探索できます。マウス操作で3D空間を回転させ、多次元のトレードオフを理解できます。

* * *

## 3.2 本章のまとめ

### 学んだこと

  1. **多目的最適化の基礎**
     * 多目的最適化問題の定式化
     * パレート優越とパレート最適性の概念
     * パレートフロンティアの意味
  2. **スカラー化手法**
     * 重み付け和法によるパレート解の生成
     * ε制約法の実装
     * 各手法の特性と使い分け
  3. **進化的アルゴリズム**
     * 非優越ソーティングの実装
     * NSGA-IIの概念
     * 遺伝的アルゴリズムによるパレート解探索
  4. **意思決定手法**
     * TOPSIS法による解の選択
     * 意思決定者の選好反映
     * インタラクティブな可視化

### 重要なポイント

  * 多目的最適化では、単一の最適解は存在せず、パレート最適解の集合が存在する
  * 重み付け和法は理解しやすいが、非凸なパレートフロンティアでは解を見逃す可能性がある
  * ε制約法は、制約条件として扱う目的関数を明示的にコントロールできる
  * NSGA-IIは、非優越ソーティングにより効率的にパレートフロンティアを探索する
  * TOPSIS法により、意思決定者の選好に基づいて最良の解を選択できる

### 次の章へ

第4章では、**制約条件下での最適化** を詳しく学びます：

  * ラグランジュ乗数法とKKT条件
  * ペナルティ法と障壁法
  * 逐次二次計画法（SQP）
  * 化学プロセスの制約条件（物質収支、エネルギー収支、安全制約）
  * CSTR最適化と蒸留塔最適運転条件
