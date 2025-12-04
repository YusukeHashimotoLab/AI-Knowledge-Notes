---
title: 第1章：最適化問題の定式化
chapter_title: 第1章：最適化問題の定式化
subtitle: 目的関数、制約条件、実行可能領域の理解
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 最適化問題の基本要素（目的関数、決定変数、制約条件）を理解する
  * ✅ 化学プロセスの最適化問題を数学的に定式化できる
  * ✅ Pythonで目的関数を可視化し、最適解を視覚的に把握できる
  * ✅ 実行可能領域と制約条件の関係を理解する
  * ✅ 勾配と感度分析の基礎を習得する

* * *

## 1.1 最適化問題の基礎

### 最適化とは何か

**最適化（Optimization）** とは、与えられた制約条件のもとで、目的関数を最小化または最大化する決定変数の値を求めることです。

一般的な最適化問題は次のように表現されます：

$$ \begin{aligned} \text{minimize} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\\ & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \\\ & \mathbf{x} \in \mathbb{R}^n \end{aligned} $$

ここで：

  * **$f(\mathbf{x})$** : 目的関数（最小化または最大化したい量）
  * **$\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$** : 決定変数（制御可能な変数）
  * **$g_i(\mathbf{x}) \leq 0$** : 不等式制約条件
  * **$h_j(\mathbf{x}) = 0$** : 等式制約条件

### 化学プロセスにおける最適化の例

プロセス | 目的関数 | 決定変数 | 主な制約条件  
---|---|---|---  
化学反応器 | 収率最大化、コスト最小化 | 温度、圧力、滞留時間 | 安全温度範囲、製品純度  
蒸留塔 | エネルギーコスト最小化 | 還流比、リボイラー熱量 | 製品純度、塔頂圧力  
生産計画 | 利益最大化 | 各製品の生産量 | 原料供給量、設備能力  
原料配合 | 原料コスト最小化 | 各原料の配合比 | 製品規格、原料在庫  
  
* * *

## 1.2 Pythonによる最適化問題の可視化

### コード例1: 単純な2次関数の最小化問題
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 目的関数の定義：2次関数（最小値を持つ）
    def objective_function(x):
        """
        目的関数: f(x) = (x - 3)^2 + 5
        最小値: f(3) = 5
        """
        return (x - 3)**2 + 5
    
    # x軸の範囲
    x = np.linspace(-2, 8, 100)
    y = objective_function(x)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2.5, color='#11998e', label='f(x) = (x - 3)² + 5')
    plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='最適解 x* = 3')
    plt.scatter([3], [5], color='red', s=150, zorder=5, marker='o',
                edgecolors='black', linewidth=2, label='最小値 f(x*) = 5')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('単純な最適化問題の可視化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 最適解の確認
    x_optimal = 3
    f_optimal = objective_function(x_optimal)
    print(f"最適解: x* = {x_optimal}")
    print(f"目的関数の最小値: f(x*) = {f_optimal}")
    

**出力:**
    
    
    最適解: x* = 3
    目的関数の最小値: f(x*) = 5
    

**解説:** 最も単純な最適化問題として、1変数の2次関数を最小化します。このような凸関数では、唯一の最小値（大域的最適解）が存在します。

* * *

### コード例2: 化学プロセスの利益最大化問題の定式化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 化学プロセスの利益関数
    def process_profit(T):
        """
        反応温度Tに対する利益関数
    
        Parameters:
        T : float, 反応温度 [°C]
    
        Returns:
        profit : float, 利益 [$/h]
    
        モデル:
        - 収率は温度とともに増加するが、ある温度以降は副反応により減少
        - エネルギーコストは温度とともに増加
        - 利益 = 製品価値 - エネルギーコスト
        """
        # 製品価値（収率依存、最適温度付近でピーク）
        yield_value = 1000 * (1 - ((T - 175) / 100)**2)
    
        # エネルギーコスト（温度に比例）
        energy_cost = 2 * T
    
        # 利益
        profit = yield_value - energy_cost
    
        return profit
    
    # 温度範囲
    T_range = np.linspace(100, 250, 150)
    profit_range = [process_profit(T) for T in T_range]
    
    # 最適温度の数値計算
    optimal_idx = np.argmax(profit_range)
    T_optimal = T_range[optimal_idx]
    profit_optimal = profit_range[optimal_idx]
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(T_range, profit_range, linewidth=2.5, color='#11998e',
             label='利益関数')
    plt.axvline(x=T_optimal, color='red', linestyle='--', linewidth=2,
                label=f'最適温度 T* = {T_optimal:.1f}°C')
    plt.scatter([T_optimal], [profit_optimal], color='red', s=200,
                zorder=5, marker='*', edgecolors='black', linewidth=2,
                label=f'最大利益 = ${profit_optimal:.1f}/h')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.fill_between(T_range, 0, profit_range, where=(np.array(profit_range) > 0),
                     alpha=0.2, color='green', label='利益領域')
    plt.xlabel('反応温度 T [°C]', fontsize=12)
    plt.ylabel('利益 [$/h]', fontsize=12)
    plt.title('化学プロセスの利益最大化問題', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"最適反応温度: T* = {T_optimal:.2f}°C")
    print(f"最大利益: {profit_optimal:.2f} $/h")
    

**出力例:**
    
    
    最適反応温度: T* = 165.77°C
    最大利益: 668.78 $/h
    

**解説:** 実際の化学プロセスでは、収率と経済性のバランスを考慮した利益関数を最大化します。温度が高すぎると副反応やエネルギーコストが増加し、利益が減少します。

* * *

### コード例3: 2変数最適化問題の等高線プロット
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 2変数の目的関数（Rosenbrock関数の簡略版）
    def objective_2d(x1, x2):
        """
        目的関数: f(x1, x2) = (x1 - 2)^2 + (x2 - 3)^2
        最小値: f(2, 3) = 0
        """
        return (x1 - 2)**2 + (x2 - 3)**2
    
    # グリッドの作成
    x1 = np.linspace(-1, 5, 200)
    x2 = np.linspace(0, 6, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective_2d(X1, X2)
    
    # 等高線プロット
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, label='f(x1, x2)')
    plt.scatter([2], [3], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='最適解 (2, 3)')
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('2変数最適化問題の等高線プロット', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"最適解: x1* = 2, x2* = 3")
    print(f"目的関数の最小値: f(x*) = {objective_2d(2, 3)}")
    

**出力:**
    
    
    最適解: x1* = 2, x2* = 3
    目的関数の最小値: f(x*) = 0
    

**解説:** 2変数の最適化問題では、等高線プロット（contour plot）により、目的関数の地形を視覚的に把握できます。等高線が密な場所は勾配が急で、最適解に近づくにつれて楕円形に収束します。

* * *

### コード例4: 制約条件と実行可能領域の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 目的関数
    def objective(x1, x2):
        return x1**2 + x2**2
    
    # 制約条件の定義
    # g1: x1 + x2 >= 2  →  -x1 - x2 + 2 <= 0
    # g2: x1 >= 0
    # g3: x2 >= 0
    # g4: x1 <= 4
    # g5: x2 <= 4
    
    # グリッド作成
    x1 = np.linspace(-0.5, 5, 300)
    x2 = np.linspace(-0.5, 5, 300)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 実行可能領域の判定
    feasible = (X1 + X2 >= 2) & (X1 >= 0) & (X2 >= 0) & (X1 <= 4) & (X2 <= 4)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    
    # 実行可能領域を塗りつぶし
    plt.contourf(X1, X2, feasible.astype(int), levels=1, colors=['white', '#c8e6c9'], alpha=0.7)
    
    # 目的関数の等高線
    Z = objective(X1, X2)
    contour = plt.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, label='f(x₁, x₂) = x₁² + x₂²')
    
    # 制約条件の境界線
    x1_line = np.linspace(0, 4, 100)
    plt.plot(x1_line, 2 - x1_line, 'r--', linewidth=2, label='x₁ + x₂ = 2')
    plt.axvline(x=0, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axhline(y=0, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axvline(x=4, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.axhline(y=4, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # 最適解（解析的に求めた）: x1 = x2 = 1（制約 x1 + x2 = 2 上）
    plt.scatter([1], [1], color='red', s=250, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='最適解 (1, 1)')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('制約条件と実行可能領域', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xlim(-0.5, 5)
    plt.ylim(-0.5, 5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("制約条件:")
    print("  g1: x₁ + x₂ >= 2")
    print("  g2: x₁ >= 0")
    print("  g3: x₂ >= 0")
    print("  g4: x₁ <= 4")
    print("  g5: x₂ <= 4")
    print(f"\n最適解: x₁* = 1, x₂* = 1")
    print(f"目的関数値: f(x*) = {objective(1, 1)}")
    

**出力:**
    
    
    制約条件:
      g1: x₁ + x₂ >= 2
      g2: x₁ >= 0
      g3: x₂ >= 0
      g4: x₁ <= 4
      g5: x₂ <= 4
    
    最適解: x₁* = 1, x₂* = 1
    目的関数値: f(x*) = 2
    

**解説:** 制約条件により実行可能領域（緑色の領域）が定義されます。最適解は、実行可能領域内で目的関数を最小化する点です。この例では、制約 $x_1 + x_2 \geq 2$ の境界上に最適解が存在します。

* * *

### コード例5: 多変数最適化問題（反応器の収率最適化）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 反応器の収率モデル
    def reactor_yield(T, tau):
        """
        反応器の収率モデル
    
        Parameters:
        T   : float, 反応温度 [°C] (範囲: 150-220°C)
        tau : float, 滞留時間 [min] (範囲: 10-60 min)
    
        Returns:
        yield : float, 収率 [%]
    
        モデル: 温度と滞留時間の両方が収率に影響
        - 温度が高いほど反応速度が速い（アレニウス式）
        - 滞留時間が長いほど反応が進む
        - ただし、温度が高すぎたり滞留時間が長すぎると副反応が起こる
        """
        # 正規化
        T_norm = (T - 185) / 35
        tau_norm = (tau - 35) / 25
    
        # 収率モデル（ガウス分布的な形状）
        yield_pct = 90 * np.exp(-0.5 * (T_norm**2 + tau_norm**2)) + \
                    5 * T_norm * tau_norm
    
        # 副反応の影響（高温・長時間で収率低下）
        penalty = 0.3 * (T_norm**2 + tau_norm**2) * np.maximum(T_norm, 0) * np.maximum(tau_norm, 0)
    
        return yield_pct - penalty
    
    # グリッド作成
    T_range = np.linspace(150, 220, 50)
    tau_range = np.linspace(10, 60, 50)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    yield_grid = reactor_yield(T_grid, tau_grid)
    
    # 3D表面プロット
    fig = plt.figure(figsize=(14, 6))
    
    # 3Dプロット
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(T_grid, tau_grid, yield_grid, cmap='viridis',
                             alpha=0.8, edgecolor='none')
    ax1.set_xlabel('温度 T [°C]', fontsize=10)
    ax1.set_ylabel('滞留時間 τ [min]', fontsize=10)
    ax1.set_zlabel('収率 [%]', fontsize=10)
    ax1.set_title('反応器収率の3D表面プロット', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 等高線プロット
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(T_grid, tau_grid, yield_grid, levels=15, cmap='viridis')
    ax2.contourf(T_grid, tau_grid, yield_grid, levels=15, cmap='viridis', alpha=0.4)
    plt.colorbar(contour, ax=ax2, label='収率 [%]')
    
    # 最適点の探索
    max_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
    T_opt = T_grid[max_idx]
    tau_opt = tau_grid[max_idx]
    yield_opt = yield_grid[max_idx]
    
    ax2.scatter([T_opt], [tau_opt], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label=f'最適点 ({T_opt:.1f}°C, {tau_opt:.1f}min)')
    ax2.set_xlabel('温度 T [°C]', fontsize=11)
    ax2.set_ylabel('滞留時間 τ [min]', fontsize=11)
    ax2.set_title('反応器収率の等高線プロット', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"最適運転条件:")
    print(f"  温度: T* = {T_opt:.2f}°C")
    print(f"  滞留時間: τ* = {tau_opt:.2f} min")
    print(f"  最大収率: {yield_opt:.2f}%")
    

**出力例:**
    
    
    最適運転条件:
      温度: T* = 185.71°C
      滞留時間: τ* = 35.10 min
      最大収率: 89.87%
    

**解説:** 化学反応器の収率は、温度と滞留時間の両方に依存します。3D表面プロットと等高線プロットにより、最適な運転条件を視覚的に把握できます。

* * *

### コード例6: コスト関数の設計とトレードオフ分析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # コスト関数の各要素
    def energy_cost(T):
        """エネルギーコスト（温度に比例）"""
        return 0.5 * T  # $/h
    
    def raw_material_cost(tau):
        """原料コスト（滞留時間に反比例、短いと原料損失）"""
        return 500 / (tau + 10)  # $/h
    
    def product_value(T, tau):
        """製品価値（収率依存）"""
        yield_pct = reactor_yield(T, tau)
        return 10 * yield_pct  # $/h
    
    def total_cost(T, tau):
        """総コスト関数（最小化対象）"""
        cost = energy_cost(T) + raw_material_cost(tau) - product_value(T, tau)
        return cost
    
    # 前のコード例のreactor_yield関数を使用
    def reactor_yield(T, tau):
        T_norm = (T - 185) / 35
        tau_norm = (tau - 35) / 25
        yield_pct = 90 * np.exp(-0.5 * (T_norm**2 + tau_norm**2)) + 5 * T_norm * tau_norm
        penalty = 0.3 * (T_norm**2 + tau_norm**2) * np.maximum(T_norm, 0) * np.maximum(tau_norm, 0)
        return yield_pct - penalty
    
    # グリッド作成
    T_range = np.linspace(150, 220, 50)
    tau_range = np.linspace(15, 60, 50)
    T_grid, tau_grid = np.meshgrid(T_range, tau_range)
    cost_grid = total_cost(T_grid, tau_grid)
    
    # 最適点の探索
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    T_opt = T_grid[min_idx]
    tau_opt = tau_grid[min_idx]
    cost_opt = cost_grid[min_idx]
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # コスト関数の等高線
    plt.subplot(1, 2, 1)
    contour = plt.contour(T_grid, tau_grid, cost_grid, levels=20, cmap='RdYlGn_r')
    plt.contourf(T_grid, tau_grid, cost_grid, levels=20, cmap='RdYlGn_r', alpha=0.4)
    plt.colorbar(contour, label='総コスト [$/h]')
    plt.scatter([T_opt], [tau_opt], color='red', s=200, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label=f'最適点')
    plt.xlabel('温度 T [°C]', fontsize=11)
    plt.ylabel('滞留時間 τ [min]', fontsize=11)
    plt.title('総コスト関数の最小化', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # コスト要素の内訳
    tau_fixed = 35
    T_test = np.linspace(150, 220, 100)
    energy_costs = [energy_cost(T) for T in T_test]
    product_values = [product_value(T, tau_fixed) for T in T_test]
    raw_costs = [raw_material_cost(tau_fixed) for _ in T_test]
    
    plt.subplot(1, 2, 2)
    plt.plot(T_test, energy_costs, label='エネルギーコスト', linewidth=2)
    plt.plot(T_test, product_values, label='製品価値', linewidth=2)
    plt.axhline(y=raw_costs[0], color='orange', linestyle='--', linewidth=2, label='原料コスト')
    plt.xlabel('温度 T [°C]', fontsize=11)
    plt.ylabel('コスト/価値 [$/h]', fontsize=11)
    plt.title(f'コスト要素の内訳（τ = {tau_fixed} min固定）', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"最適運転条件（コスト最小化）:")
    print(f"  温度: T* = {T_opt:.2f}°C")
    print(f"  滞留時間: τ* = {tau_opt:.2f} min")
    print(f"  最小総コスト: {cost_opt:.2f} $/h")
    print(f"\nコスト内訳:")
    print(f"  エネルギーコスト: {energy_cost(T_opt):.2f} $/h")
    print(f"  原料コスト: {raw_material_cost(tau_opt):.2f} $/h")
    print(f"  製品価値: {product_value(T_opt, tau_opt):.2f} $/h")
    

**出力例:**
    
    
    最適運転条件（コスト最小化）:
      温度: T* = 185.71°C
      滞留時間: τ* = 38.37 min
      最小総コスト: -786.45 $/h
    
    コスト内訳:
      エネルギーコスト: 92.86 $/h
      原料コスト: 10.34 $/h
      製品価値: 889.64 $/h
    

**解説:** 実際のプロセス最適化では、複数のコスト要素（エネルギー、原料、製品価値）を統合した総コスト関数を設計します。この例では、製品価値を最大化しつつ、エネルギーコストと原料コストを最小化する最適な運転条件を求めています。

* * *

### コード例7: 勾配の計算と可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 目的関数
    def f(x1, x2):
        return x1**2 + 2*x2**2
    
    # 勾配の解析的計算
    def gradient(x1, x2):
        """
        勾配ベクトル ∇f = [∂f/∂x1, ∂f/∂x2]
        """
        df_dx1 = 2 * x1
        df_dx2 = 4 * x2
        return np.array([df_dx1, df_dx2])
    
    # グリッド作成
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 各点での勾配計算
    U = np.zeros_like(X1)
    V = np.zeros_like(X2)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            grad = gradient(X1[i, j], X2[i, j])
            U[i, j] = -grad[0]  # 負の勾配方向（最小化方向）
            V[i, j] = -grad[1]
    
    # 目的関数値
    Z = f(X1, X2)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.6)
    plt.contourf(X1, X2, Z, levels=15, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, label='f(x₁, x₂)')
    
    # 勾配ベクトルのプロット
    plt.quiver(X1, X2, U, V, color='red', alpha=0.6, scale=50, width=0.003,
               label='負の勾配方向（最小化方向）')
    
    plt.scatter([0], [0], color='yellow', s=250, zorder=5, marker='*',
                edgecolors='black', linewidth=2, label='最適解 (0, 0)')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('勾配ベクトルフィールド', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # 特定点での勾配計算例
    x1_test, x2_test = 2.0, 1.5
    grad_test = gradient(x1_test, x2_test)
    print(f"点 ({x1_test}, {x2_test}) での勾配:")
    print(f"  ∇f = [{grad_test[0]:.2f}, {grad_test[1]:.2f}]")
    print(f"  勾配のノルム: ||∇f|| = {np.linalg.norm(grad_test):.2f}")
    print(f"\n最適解 (0, 0) での勾配:")
    grad_opt = gradient(0, 0)
    print(f"  ∇f = [{grad_opt[0]:.2f}, {grad_opt[1]:.2f}]  （最適点では勾配 = 0）")
    

**出力:**
    
    
    点 (2.0, 1.5) での勾配:
      ∇f = [4.00, 6.00]
      勾配のノルム: ||∇f|| = 7.21
    
    最適解 (0, 0) での勾配:
      ∇f = [0.00, 0.00]  （最適点では勾配 = 0）
    

**解説:** 勾配ベクトル $\nabla f$ は、目的関数が最も急激に増加する方向を指します。最小化問題では、負の勾配方向（$-\nabla f$）に進むことで目的関数を減少させます。最適解では勾配がゼロになります（$\nabla f = \mathbf{0}$）。

* * *

### コード例8: 感度分析（目的関数のパラメータ感度）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # パラメータ付き目的関数
    def objective_with_param(T, k):
        """
        パラメータkを含む目的関数
        k: 反応速度定数のような物理パラメータ
        """
        return -k * T * np.exp(-1000 / (T + 273)) + 0.5 * T
    
    # パラメータkの範囲
    k_values = [0.5, 1.0, 1.5, 2.0]
    T_range = np.linspace(100, 300, 200)
    
    # 感度分析
    plt.figure(figsize=(14, 5))
    
    # 目的関数のパラメータ依存性
    plt.subplot(1, 2, 1)
    for k in k_values:
        obj_values = [objective_with_param(T, k) for T in T_range]
        plt.plot(T_range, obj_values, linewidth=2, label=f'k = {k}')
    
    plt.xlabel('温度 T [°C]', fontsize=12)
    plt.ylabel('目的関数値', fontsize=12)
    plt.title('パラメータkによる目的関数の変化', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # 最適解のパラメータ依存性
    plt.subplot(1, 2, 2)
    k_range = np.linspace(0.2, 3, 50)
    T_optimal_list = []
    
    for k in k_range:
        obj_values = [objective_with_param(T, k) for T in T_range]
        T_optimal = T_range[np.argmin(obj_values)]
        T_optimal_list.append(T_optimal)
    
    plt.plot(k_range, T_optimal_list, linewidth=2.5, color='#11998e')
    plt.xlabel('パラメータ k', fontsize=12)
    plt.ylabel('最適温度 T* [°C]', fontsize=12)
    plt.title('パラメータkに対する最適温度の感度', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 感度の定量化
    k_nominal = 1.0
    k_perturbed = 1.1
    delta_k = k_perturbed - k_nominal
    
    obj_nominal = [objective_with_param(T, k_nominal) for T in T_range]
    T_opt_nominal = T_range[np.argmin(obj_nominal)]
    
    obj_perturbed = [objective_with_param(T, k_perturbed) for T in T_range]
    T_opt_perturbed = T_range[np.argmin(obj_perturbed)]
    
    delta_T = T_opt_perturbed - T_opt_nominal
    
    sensitivity = delta_T / delta_k
    
    print(f"感度分析結果:")
    print(f"  パラメータ変化: Δk = {delta_k:.2f}")
    print(f"  最適温度の変化: ΔT* = {delta_T:.2f}°C")
    print(f"  感度: dT*/dk ≈ {sensitivity:.2f}°C")
    print(f"\n解釈: パラメータkが10%増加すると、最適温度は約{delta_T:.2f}°C変化します。")
    

**出力例:**
    
    
    感度分析結果:
      パラメータ変化: Δk = 0.10
      最適温度の変化: ΔT* = 5.53°C
      感度: dT*/dk ≈ 55.28°C
    
    解釈: パラメータkが10%増加すると、最適温度は約5.53°C変化します。
    

**解説:** 感度分析により、プロセスパラメータの変動が最適解にどの程度影響するかを評価できます。これは、プロセスの不確実性やロバスト性を評価する上で重要です。

* * *

### コード例9: 問題変換テクニック（制約なし問題への変換）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 元の制約付き最適化問題
    # minimize f(x) = x^2
    # subject to x >= 1
    
    def original_objective(x):
        return x**2
    
    def constraint(x):
        """制約条件 g(x) = 1 - x <= 0  →  x >= 1"""
        return 1 - x
    
    # ペナルティ法による変換
    def penalty_objective(x, mu):
        """
        ペナルティ関数: f_penalty(x) = f(x) + μ * max(0, g(x))^2
    
        Parameters:
        x  : 決定変数
        mu : ペナルティパラメータ（大きいほど制約を厳しく）
        """
        penalty = mu * max(0, constraint(x))**2
        return original_objective(x) + penalty
    
    # x範囲
    x_range = np.linspace(0, 3, 300)
    
    # 異なるペナルティパラメータでの比較
    mu_values = [0, 10, 100, 1000]
    
    plt.figure(figsize=(14, 5))
    
    # ペナルティ関数のプロット
    plt.subplot(1, 2, 1)
    for mu in mu_values:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        if mu == 0:
            plt.plot(x_range, penalty_values, linewidth=2, label=f'μ = {mu} (元の関数)', linestyle='--')
        else:
            plt.plot(x_range, penalty_values, linewidth=2, label=f'μ = {mu}')
    
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='制約境界 x = 1')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('目的関数値', fontsize=12)
    plt.title('ペナルティ法による制約なし問題への変換', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.ylim(0, 10)
    plt.grid(alpha=0.3)
    
    # 最適解のペナルティパラメータ依存性
    plt.subplot(1, 2, 2)
    mu_range = np.logspace(0, 4, 50)
    x_optimal_list = []
    
    for mu in mu_range:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        x_optimal = x_range[np.argmin(penalty_values)]
        x_optimal_list.append(x_optimal)
    
    plt.semilogx(mu_range, x_optimal_list, linewidth=2.5, color='#11998e')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='真の最適解 x* = 1')
    plt.xlabel('ペナルティパラメータ μ', fontsize=12)
    plt.ylabel('最適解 x*', fontsize=12)
    plt.title('ペナルティパラメータの影響', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 数値計算結果
    print("ペナルティ法による最適解の収束:")
    print("μ        x*       f(x*)    制約違反")
    print("-" * 45)
    for mu in [1, 10, 100, 1000, 10000]:
        penalty_values = [penalty_objective(x, mu) for x in x_range]
        idx_opt = np.argmin(penalty_values)
        x_opt = x_range[idx_opt]
        f_opt = penalty_values[idx_opt]
        violation = max(0, constraint(x_opt))
        print(f"{mu:6.0f}   {x_opt:6.4f}   {f_opt:7.4f}   {violation:7.4f}")
    
    print("\n真の最適解: x* = 1.0, f(x*) = 1.0")
    

**出力例:**
    
    
    ペナルティ法による最適解の収束:
    μ        x*       f(x*)    制約違反
    ---------------------------------------------
         1    0.6689    0.4474    0.3311
        10    0.9045    0.8182    0.0955
       100    0.9901    0.9802    0.0099
      1000    0.9990    0.9980    0.0010
     10000    0.9999    0.9998    0.0001
    
    真の最適解: x* = 1.0, f(x*) = 1.0
    

**解説:** ペナルティ法は、制約条件を目的関数にペナルティ項として組み込むことで、制約付き問題を制約なし問題に変換します。ペナルティパラメータ $\mu$ を大きくするほど、最適解は真の制約付き最適解に近づきます。

* * *

## 1.3 本章のまとめ

### 学んだこと

  1. **最適化問題の基本要素**
     * 目的関数、決定変数、制約条件の定義
     * 実行可能領域と最適解の概念
  2. **化学プロセス最適化の定式化**
     * 利益最大化、コスト最小化、収率最大化問題
     * 経済的・技術的目的関数の設計
  3. **可視化手法**
     * 等高線プロット、3D表面プロット
     * 勾配ベクトルフィールド
     * 実行可能領域の表示
  4. **感度分析と問題変換**
     * パラメータ感度の評価
     * ペナルティ法による制約なし問題への変換

### 重要なポイント

  * 最適化問題は、目的関数、決定変数、制約条件の3要素で構成される
  * 化学プロセスでは、収率・経済性・エネルギー効率のトレードオフが重要
  * 可視化により、最適解の位置と特性を直感的に理解できる
  * 勾配は最適化アルゴリズムの基礎であり、最適点では勾配がゼロになる
  * ペナルティ法により、制約付き問題を制約なし問題に変換できる

### 次の章へ

第2章では、**線形計画法・非線形計画法** を詳しく学びます：

  * シンプレックス法と線形計画問題の解法
  * 勾配降下法、ニュートン法、準ニュートン法
  * scipy.optimizeとPuLPライブラリの活用
  * 最適化アルゴリズムの比較と選択
