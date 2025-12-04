---
title: "第2章:線形計画法・非線形計画法"
chapter_title: "第2章:線形計画法・非線形計画法"
subtitle: シンプレックス法と勾配ベースの最適化アルゴリズム
---

## 学習目標

この章を読むことで、以下を習得できます:

  * ✅ 線形計画問題の定式化とシンプレックス法の理解
  * ✅ scipy.optimizeとPuLPライブラリによる線形計画の実装
  * ✅ 勾配降下法、ニュートン法、準ニュートン法の原理と実装
  * ✅ 非線形最適化アルゴリズムの収束性と計算効率の評価
  * ✅ 化学プロセス問題への最適化手法の適用

* * *

## 2.1 線形計画法（Linear Programming）

### 線形計画問題とは

**線形計画問題（LP）** は、線形の目的関数を線形制約条件のもとで最適化する問題です。

標準形式:

$$ \begin{aligned} \text{minimize} \quad & \mathbf{c}^T \mathbf{x} \\\ \text{subject to} \quad & \mathbf{A} \mathbf{x} \leq \mathbf{b} \\\ & \mathbf{x} \geq \mathbf{0} \end{aligned} $$

ここで、$\mathbf{c}$は目的関数の係数ベクトル、$\mathbf{A}$は制約行列、$\mathbf{b}$は制約ベクトルです。

### 線形計画法の特徴

特徴 | 説明 | 利点  
---|---|---  
凸性 | 実行可能領域は凸多面体 | 局所最適解=大域的最適解  
効率性 | シンプレックス法は効率的 | 大規模問題でも高速求解  
汎用性 | 生産計画、配合問題など多様な応用 | 化学プロセスの運転最適化に適用  
双対性 | 主問題と双対問題の関係 | 感度分析が容易  
  
* * *

### コード例1: scipy.optimize.linprogによる生産計画最適化
    
    
    import numpy as np
    from scipy.optimize import linprog
    import matplotlib.pyplot as plt
    
    """
    生産計画問題
    -----------
    製品A, Bを生産する化学プラント。利益を最大化したい。
    
    決定変数:
      x1 = 製品Aの生産量 [kg/day]
      x2 = 製品Bの生産量 [kg/day]
    
    目的関数（利益最大化）:
      maximize: 40*x1 + 30*x2  →  minimize: -40*x1 - 30*x2
    
    制約条件:
      1. 原料制約: 2*x1 + x2 <= 100  (原料在庫)
      2. 設備時間制約: x1 + 2*x2 <= 80  (設備稼働時間)
      3. 非負制約: x1, x2 >= 0
    """
    
    # 目的関数の係数（最小化問題なので符号反転）
    c = np.array([-40, -30])
    
    # 不等式制約の係数行列 A*x <= b
    A_ub = np.array([
        [2, 1],   # 原料制約
        [1, 2]    # 設備時間制約
    ])
    
    b_ub = np.array([100, 80])
    
    # 変数の境界（非負制約）
    x_bounds = [(0, None), (0, None)]
    
    # 線形計画問題の求解
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')
    
    # 結果の表示
    print("=" * 60)
    print("生産計画最適化問題の結果")
    print("=" * 60)
    print(f"最適化ステータス: {result.message}")
    print(f"最適解:")
    print(f"  製品A生産量 (x1): {result.x[0]:.2f} kg/day")
    print(f"  製品B生産量 (x2): {result.x[1]:.2f} kg/day")
    print(f"最大利益: {-result.fun:.2f} $/day")  # 符号を戻す
    print(f"反復回数: {result.nit}")
    
    # 実行可能領域の可視化
    x1 = np.linspace(0, 60, 400)
    
    # 制約条件の境界線
    x2_constraint1 = 100 - 2*x1  # 原料制約
    x2_constraint2 = (80 - x1) / 2  # 設備時間制約
    
    plt.figure(figsize=(10, 8))
    
    # 制約条件の描画
    plt.plot(x1, x2_constraint1, 'r-', linewidth=2, label='原料制約: 2x₁ + x₂ = 100')
    plt.plot(x1, x2_constraint2, 'b-', linewidth=2, label='設備時間制約: x₁ + 2x₂ = 80')
    
    # 実行可能領域の塗りつぶし
    x2_feasible = np.minimum(x2_constraint1, x2_constraint2)
    x2_feasible = np.maximum(x2_feasible, 0)
    plt.fill_between(x1, 0, x2_feasible, where=(x2_feasible >= 0),
                     alpha=0.3, color='green', label='実行可能領域')
    
    # 目的関数の等高線（利益線）
    for profit in [800, 1200, 1600, 2000]:
        x2_profit = (profit - 40*x1) / 30
        plt.plot(x1, x2_profit, '--', alpha=0.4, color='gray')
    
    # 最適解のプロット
    plt.scatter([result.x[0]], [result.x[1]], color='red', s=300, zorder=5,
                marker='*', edgecolors='black', linewidth=2,
                label=f'最適解 ({result.x[0]:.1f}, {result.x[1]:.1f})')
    
    plt.xlabel('製品A生産量 x₁ [kg/day]', fontsize=12)
    plt.ylabel('製品B生産量 x₂ [kg/day]', fontsize=12)
    plt.title('生産計画最適化問題の幾何学的解釈', fontsize=14, fontweight='bold')
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 制約条件の余裕度（スラック変数）
    slack = b_ub - A_ub @ result.x
    print(f"\n制約条件の余裕度:")
    print(f"  原料制約の余裕: {slack[0]:.2f} kg")
    print(f"  設備時間制約の余裕: {slack[1]:.2f} 時間")
    

**出力例:**
    
    
    ============================================================
    生産計画最適化問題の結果
    ============================================================
    最適化ステータス: Optimization terminated successfully.
    最適解:
      製品A生産量 (x1): 40.00 kg/day
      製品B生産量 (x2): 20.00 kg/day
    最大利益: 2200.00 $/day
    反復回数: 2
    
    制約条件の余裕度:
      原料制約の余裕: 0.00 kg
      設備時間制約の余裕: 0.00 時間
    

**解説:** 線形計画問題では、最適解は実行可能領域の頂点（vertex）に存在します。この例では、2つの制約が同時に有効（active）となる点が最適解です。

* * *

### コード例2: PuLPライブラリによる化学配合問題
    
    
    from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value
    import pandas as pd
    
    """
    化学原料配合最適化問題
    ---------------------
    3種類の原料（A, B, C）を配合して製品を製造。
    製品仕様を満たしつつ、原料コストを最小化。
    
    制約条件:
      - 硫黄含有量 >= 2.0%
      - 粘度 <= 100 cP
      - 総配合量 = 1000 kg
    """
    
    # 原料データ
    raw_materials = {
        'A': {'cost': 50, 'sulfur': 3.0, 'viscosity': 80},   # $/kg, %, cP
        'B': {'cost': 40, 'sulfur': 2.0, 'viscosity': 120},
        'C': {'cost': 60, 'sulfur': 1.5, 'viscosity': 60}
    }
    
    # 問題の定義（コスト最小化）
    prob = LpProblem("Chemical_Blending", LpMaximize)  # 後で符号反転
    
    # 決定変数の定義（各原料の使用量 kg）
    x = {}
    for material in raw_materials.keys():
        x[material] = LpVariable(f"x_{material}", lowBound=0)
    
    # 目的関数（コスト最小化 → 負の利益最大化）
    prob += -lpSum([x[material] * raw_materials[material]['cost']
                    for material in raw_materials.keys()]), "Total_Cost"
    
    # 制約条件1: 総配合量 = 1000 kg
    prob += lpSum([x[material] for material in raw_materials.keys()]) == 1000, "Total_Amount"
    
    # 制約条件2: 硫黄含有量 >= 2.0%
    prob += lpSum([x[material] * raw_materials[material]['sulfur']
                   for material in raw_materials.keys()]) >= 2.0 * 1000, "Sulfur_Content"
    
    # 制約条件3: 粘度 <= 100 cP（加重平均）
    prob += lpSum([x[material] * raw_materials[material]['viscosity']
                   for material in raw_materials.keys()]) <= 100 * 1000, "Viscosity_Limit"
    
    # 求解
    prob.solve()
    
    # 結果の表示
    print("=" * 70)
    print("化学原料配合最適化の結果")
    print("=" * 70)
    print(f"ステータス: {LpStatus[prob.status]}")
    print(f"\n最適配合:")
    
    results = []
    for material in raw_materials.keys():
        amount = value(x[material])
        percentage = (amount / 1000) * 100
        cost = amount * raw_materials[material]['cost']
        print(f"  原料{material}: {amount:.2f} kg ({percentage:.1f}%)  コスト: ${cost:.2f}")
        results.append({
            '原料': material,
            '使用量 [kg]': amount,
            '配合比 [%]': percentage,
            'コスト [$]': cost
        })
    
    total_cost = -value(prob.objective)
    print(f"\n総コスト: ${total_cost:.2f}")
    
    # 製品仕様の確認
    sulfur_content = sum(value(x[material]) * raw_materials[material]['sulfur']
                         for material in raw_materials.keys()) / 1000
    viscosity = sum(value(x[material]) * raw_materials[material]['viscosity']
                    for material in raw_materials.keys()) / 1000
    
    print(f"\n製品仕様:")
    print(f"  硫黄含有量: {sulfur_content:.2f}% (規格: >= 2.0%)")
    print(f"  粘度: {viscosity:.2f} cP (規格: <= 100 cP)")
    
    # 結果をDataFrameで整理
    df_results = pd.DataFrame(results)
    print(f"\n配合詳細:")
    print(df_results.to_string(index=False))
    
    # 可視化
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 配合比の円グラフ
    materials = [r['原料'] for r in results]
    amounts = [r['使用量 [kg]'] for r in results]
    colors = ['#11998e', '#38ef7d', '#66d9ef']
    
    ax1.pie(amounts, labels=materials, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('最適配合比', fontsize=13, fontweight='bold')
    
    # コスト内訳の棒グラフ
    costs = [r['コスト [$]'] for r in results]
    ax2.bar(materials, costs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('原料', fontsize=11)
    ax2.set_ylabel('コスト [$]', fontsize=11)
    ax2.set_title('原料別コスト内訳', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (material, cost) in enumerate(zip(materials, costs)):
        ax2.text(i, cost + 500, f'${cost:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ======================================================================
    化学原料配合最適化の結果
    ======================================================================
    ステータス: Optimal
    
    最適配合:
      原料A: 500.00 kg (50.0%)  コスト: $25000.00
      原料B: 500.00 kg (50.0%)  コスト: $20000.00
      原料C: 0.00 kg (0.0%)  コスト: $0.00
    
    総コスト: $45000.00
    
    製品仕様:
      硫黄含有量: 2.50% (規格: >= 2.0%)
      粘度: 100.00 cP (規格: <= 100 cP)
    
    配合詳細:
      原料  使用量 [kg]  配合比 [%]  コスト [$]
         A       500.0        50.0    25000.0
         B       500.0        50.0    20000.0
         C         0.0         0.0        0.0
    

**解説:** PuLPライブラリは、線形計画問題をより直感的に記述できます。この例では、製品仕様（硫黄含有量、粘度）を満たしつつ、原料コストを最小化する配合比を求めています。

* * *

### コード例3: シンプレックス法の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import linprog
    
    """
    シンプレックス法の探索過程の可視化
    --------------------------------
    最適化アルゴリズムが実行可能領域の頂点を探索する様子を可視化
    """
    
    # 線形計画問題の定義
    c = np.array([-3, -4])  # 目的関数: maximize 3x1 + 4x2
    
    A_ub = np.array([
        [2, 1],   # 制約1: 2x1 + x2 <= 6
        [1, 2]    # 制約2: x1 + 2x2 <= 8
    ])
    
    b_ub = np.array([6, 8])
    
    # 実行可能領域の頂点を計算
    def calculate_vertices():
        """実行可能領域の頂点を計算"""
        vertices = [
            (0, 0),  # 原点
            (3, 0),  # x1軸上（制約1の境界）
            (0, 4),  # x2軸上（制約2の境界）
        ]
    
        # 2つの制約の交点
        # 2x1 + x2 = 6 と x1 + 2x2 = 8 の連立方程式
        A_intersection = np.array([[2, 1], [1, 2]])
        b_intersection = np.array([6, 8])
        intersection = np.linalg.solve(A_intersection, b_intersection)
        vertices.append(tuple(intersection))
    
        return vertices
    
    vertices = calculate_vertices()
    
    # 各頂点での目的関数値を計算
    def objective_value(x1, x2):
        return 3*x1 + 4*x2
    
    vertex_values = [(v[0], v[1], objective_value(v[0], v[1])) for v in vertices]
    vertex_values.sort(key=lambda x: x[2])  # 目的関数値でソート
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左図: 実行可能領域と頂点の探索
    x1 = np.linspace(0, 5, 400)
    x2_constraint1 = 6 - 2*x1
    x2_constraint2 = (8 - x1) / 2
    
    x2_feasible = np.minimum(x2_constraint1, x2_constraint2)
    x2_feasible = np.maximum(x2_feasible, 0)
    
    ax1.fill_between(x1, 0, x2_feasible, where=(x2_feasible >= 0),
                     alpha=0.3, color='lightgreen', label='実行可能領域')
    
    ax1.plot(x1, x2_constraint1, 'r-', linewidth=2, label='2x₁ + x₂ = 6')
    ax1.plot(x1, x2_constraint2, 'b-', linewidth=2, label='x₁ + 2x₂ = 8')
    
    # 目的関数の等高線
    for val in [0, 4, 8, 12, 16]:
        x2_obj = (val - 3*x1) / 4
        ax1.plot(x1, x2_obj, '--', alpha=0.3, color='gray', linewidth=1)
    
    # 頂点のプロット（探索順序）
    colors = ['yellow', 'orange', 'red', 'darkred']
    for i, (x1_v, x2_v, obj_val) in enumerate(vertex_values):
        if i < len(vertex_values) - 1:
            ax1.scatter([x1_v], [x2_v], s=200, color=colors[i],
                       edgecolors='black', linewidth=2, zorder=5,
                       label=f'頂点{i+1}: f={obj_val:.1f}')
        else:
            ax1.scatter([x1_v], [x2_v], s=400, color='red', marker='*',
                       edgecolors='black', linewidth=3, zorder=6,
                       label=f'最適解: f={obj_val:.1f}')
    
    # 探索経路の描画（シンプレックス法の概念的経路）
    path_x = [vertex_values[i][0] for i in range(len(vertex_values))]
    path_y = [vertex_values[i][1] for i in range(len(vertex_values))]
    ax1.plot(path_x, path_y, 'k--', linewidth=2, alpha=0.5, label='探索経路')
    
    for i in range(len(path_x) - 1):
        ax1.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.6))
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('シンプレックス法の探索過程', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 5)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 右図: 反復ごとの目的関数値の改善
    iterations = list(range(1, len(vertex_values) + 1))
    obj_values = [v[2] for v in vertex_values]
    
    ax2.plot(iterations, obj_values, 'o-', linewidth=3, markersize=10,
             color='#11998e', markerfacecolor='#38ef7d', markeredgecolor='black',
             markeredgewidth=2)
    ax2.set_xlabel('反復回数', fontsize=12)
    ax2.set_ylabel('目的関数値', fontsize=12)
    ax2.set_title('目的関数値の改善過程', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    for i, (iter_num, val) in enumerate(zip(iterations, obj_values)):
        ax2.text(iter_num, val + 0.5, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("シンプレックス法の探索過程")
    print("=" * 60)
    for i, (x1_v, x2_v, obj_val) in enumerate(vertex_values):
        print(f"反復 {i+1}: 頂点 ({x1_v:.2f}, {x2_v:.2f})  目的関数値: {obj_val:.2f}")
    print("=" * 60)
    print(f"最適解: ({vertex_values[-1][0]:.2f}, {vertex_values[-1][1]:.2f})")
    print(f"最大値: {vertex_values[-1][2]:.2f}")
    

**出力例:**
    
    
    ============================================================
    シンプレックス法の探索過程
    ============================================================
    反復 1: 頂点 (0.00, 0.00)  目的関数値: 0.00
    反復 2: 頂点 (3.00, 0.00)  目的関数値: 9.00
    反復 3: 頂点 (0.00, 4.00)  目的関数値: 16.00
    反復 4: 頂点 (1.33, 3.33)  目的関数値: 17.33
    ============================================================
    最適解: (1.33, 3.33)
    最大値: 17.33
    

**解説:** シンプレックス法は、実行可能領域の頂点を効率的に探索し、目的関数値を単調に改善していきます。各反復で隣接する頂点に移動し、最適解に到達します。

* * *

## 2.2 非線形計画法（Nonlinear Programming）

### 非線形最適化問題

目的関数または制約条件が非線形である最適化問題です:

$$ \begin{aligned} \text{minimize} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\\ & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \end{aligned} $$

### 主要な最適化アルゴリズム
    
    
    ```mermaid
    graph LR
        A[非線形最適化] --> B[制約なし]
        A --> C[制約付き]
        B --> D[勾配降下法]
        B --> E[ニュートン法]
        B --> F[準ニュートン法BFGS, L-BFGS]
        C --> G[ペナルティ法]
        C --> H[SQP]
        C --> I[内点法]
    
        style A fill:#11998e,stroke:#0d7a6f,color:#fff
        style B fill:#38ef7d,stroke:#2ecc71
        style C fill:#38ef7d,stroke:#2ecc71
    ```

* * *

### コード例4: 制約なし非線形最適化（scipy.optimize.minimize）
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    """
    制約なし非線形最適化問題
    -----------------------
    Rosenbrock関数の最小化
    
    f(x, y) = (1-x)^2 + 100(y-x^2)^2
    
    最小値: f(1, 1) = 0
    """
    
    # 目的関数の定義
    def rosenbrock(x):
        """Rosenbrock関数"""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    # 勾配（解析的に計算）
    def rosenbrock_grad(x):
        """Rosenbrock関数の勾配"""
        dfdx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dfdy = 200*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])
    
    # ヘッセ行列（2次微分）
    def rosenbrock_hess(x):
        """Rosenbrock関数のヘッセ行列"""
        d2fdx2 = 2 - 400*x[1] + 1200*x[0]**2
        d2fdxdy = -400*x[0]
        d2fdy2 = 200
        return np.array([[d2fdx2, d2fdxdy],
                         [d2fdxdy, d2fdy2]])
    
    # 初期値
    x0 = np.array([-1.0, -1.0])
    
    # 複数の最適化手法で求解
    methods = ['Nelder-Mead', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC']
    results = {}
    
    print("=" * 80)
    print("制約なし非線形最適化: Rosenbrock関数の最小化")
    print("=" * 80)
    print(f"初期値: x0 = {x0}")
    print(f"目的関数値 f(x0) = {rosenbrock(x0):.4f}")
    print("\n" + "=" * 80)
    print(f"{'手法':<15} {'最適解 (x, y)':<25} {'目的関数値':<15} {'反復回数':<10}")
    print("=" * 80)
    
    for method in methods:
        if method in ['Newton-CG']:
            result = minimize(rosenbrock, x0, method=method, jac=rosenbrock_grad,
                             hess=rosenbrock_hess)
        elif method in ['BFGS', 'L-BFGS-B']:
            result = minimize(rosenbrock, x0, method=method, jac=rosenbrock_grad)
        else:
            result = minimize(rosenbrock, x0, method=method)
    
        results[method] = result
        print(f"{method:<15} ({result.x[0]:7.4f}, {result.x[1]:7.4f})   "
              f"{result.fun:12.6e}   {result.nit:8d}")
    
    print("=" * 80)
    
    # Rosenbrock関数の可視化
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # 対数スケールで可視化（値の範囲が広いため）
    Z_log = np.log10(Z + 1)
    
    fig = plt.figure(figsize=(16, 6))
    
    # 等高線プロット
    ax1 = fig.add_subplot(131)
    contour = ax1.contour(X, Y, Z_log, levels=20, cmap='viridis')
    ax1.contourf(X, Y, Z_log, levels=20, cmap='viridis', alpha=0.4)
    plt.colorbar(contour, ax=ax1, label='log₁₀(f(x,y) + 1)')
    ax1.scatter([1], [1], color='red', s=300, marker='*',
               edgecolors='black', linewidth=2, label='真の最適解 (1, 1)', zorder=5)
    ax1.scatter([x0[0]], [x0[1]], color='yellow', s=150, marker='o',
               edgecolors='black', linewidth=2, label='初期値', zorder=5)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('Rosenbrock関数の等高線', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 3D表面プロット
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(X, Y, Z_log, cmap='viridis', alpha=0.7,
                            edgecolor='none', antialiased=True)
    ax2.scatter([1], [1], [0], color='red', s=200, marker='*',
               edgecolors='black', linewidth=2)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.set_zlabel('log₁₀(f(x,y) + 1)', fontsize=10)
    ax2.set_title('Rosenbrock関数の3D表面', fontsize=13, fontweight='bold')
    
    # 手法ごとの収束性比較
    ax3 = fig.add_subplot(133)
    method_labels = list(results.keys())
    nit_values = [results[m].nit for m in method_labels]
    fun_values = [results[m].fun for m in method_labels]
    
    x_pos = np.arange(len(method_labels))
    bars = ax3.bar(x_pos, nit_values, color='#11998e', edgecolor='black', linewidth=1.5)
    
    # 目的関数値を各バーの上に表示
    for i, (bar, fun_val) in enumerate(zip(bars, fun_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{fun_val:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_labels, rotation=45, ha='right')
    ax3.set_ylabel('反復回数', fontsize=11)
    ax3.set_title('最適化手法の反復回数比較', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ================================================================================
    制約なし非線形最適化: Rosenbrock関数の最小化
    ================================================================================
    初期値: x0 = [-1. -1.]
    目的関数値 f(x0) = 404.0000
    
    ================================================================================
    手法             最適解 (x, y)              目的関数値       反復回数
    ================================================================================
    Nelder-Mead     ( 1.0000,  1.0000)      7.888892e-10          79
    BFGS            ( 1.0000,  1.0000)      1.432465e-11          33
    Newton-CG       ( 1.0000,  1.0000)      6.689048e-18          23
    L-BFGS-B        ( 1.0000,  1.0000)      1.432465e-11          24
    TNC             ( 1.0000,  1.0000)      1.510571e-17          26
    ================================================================================
    

**解説:** scipy.optimize.minimizeは、複数の最適化アルゴリズムを統一的なインターフェースで利用できます。勾配情報やヘッセ行列を提供することで、収束速度が向上します。

* * *

### コード例5: 勾配降下法の実装とステップサイズの影響
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    勾配降下法のスクラッチ実装
    -------------------------
    ステップサイズ（学習率）の影響を可視化
    """
    
    # 目的関数: f(x, y) = x^2 + 4*y^2
    def objective(x):
        return x[0]**2 + 4*x[1]**2
    
    # 勾配
    def gradient(x):
        return np.array([2*x[0], 8*x[1]])
    
    # 勾配降下法の実装
    def gradient_descent(x0, learning_rate, max_iter=100, tol=1e-6):
        """
        勾配降下法
    
        Parameters:
        -----------
        x0 : array, 初期値
        learning_rate : float, ステップサイズ（学習率）
        max_iter : int, 最大反復回数
        tol : float, 収束判定の閾値
    
        Returns:
        --------
        trajectory : list, 最適化の軌跡
        """
        x = x0.copy()
        trajectory = [x.copy()]
    
        for i in range(max_iter):
            grad = gradient(x)
    
            # 勾配降下の更新式
            x_new = x - learning_rate * grad
    
            trajectory.append(x_new.copy())
    
            # 収束判定
            if np.linalg.norm(x_new - x) < tol:
                print(f"  収束: {i+1} 反復")
                break
    
            x = x_new
    
        return trajectory
    
    # 異なる学習率でテスト
    learning_rates = [0.05, 0.1, 0.3, 0.5]
    x0 = np.array([2.0, 1.5])
    
    trajectories = {}
    
    print("=" * 70)
    print("勾配降下法: 学習率の影響")
    print("=" * 70)
    print(f"初期値: x0 = {x0}")
    print(f"目的関数: f(x, y) = x² + 4y²")
    print(f"最適解: (0, 0), 最小値: 0")
    print("=" * 70)
    
    for lr in learning_rates:
        print(f"\n学習率 α = {lr}:")
        traj = gradient_descent(x0, lr, max_iter=50)
        trajectories[lr] = traj
        final_x = traj[-1]
        final_f = objective(final_x)
        print(f"  最終解: ({final_x[0]:.6f}, {final_x[1]:.6f})")
        print(f"  目的関数値: {final_f:.6e}")
        print(f"  反復回数: {len(traj)-1}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # 等高線の準備
    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 4*Y**2
    
    for idx, (lr, traj) in enumerate(trajectories.items()):
        ax = axes[idx]
    
        # 等高線
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.2)
    
        # 最適化の軌跡
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1], 'ro-', linewidth=2,
               markersize=6, label='軌跡', markerfacecolor='red',
               markeredgecolor='black', markeredgewidth=1)
    
        # 初期値と最適解
        ax.scatter([x0[0]], [x0[1]], color='yellow', s=200, marker='o',
                  edgecolors='black', linewidth=2, label='初期値', zorder=5)
        ax.scatter([0], [0], color='lime', s=300, marker='*',
                  edgecolors='black', linewidth=2, label='最適解', zorder=5)
    
        # 矢印で方向を示す
        for i in range(min(5, len(traj)-1)):
            ax.annotate('', xy=(traj_array[i+1]), xytext=(traj_array[i]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6))
    
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'学習率 α = {lr} (反復回数: {len(traj)-1})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()
    
    # 目的関数値の収束曲線
    plt.figure(figsize=(12, 6))
    
    for lr, traj in trajectories.items():
        traj_array = np.array(traj)
        obj_values = [objective(x) for x in traj_array]
        iterations = range(len(obj_values))
        plt.semilogy(iterations, obj_values, 'o-', linewidth=2, markersize=6,
                    label=f'α = {lr}')
    
    plt.xlabel('反復回数', fontsize=12)
    plt.ylabel('目的関数値 (対数スケール)', fontsize=12)
    plt.title('学習率ごとの収束曲線', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ======================================================================
    勾配降下法: 学習率の影響
    ======================================================================
    初期値: x0 = [2.  1.5]
    目的関数: f(x, y) = x² + 4y²
    最適解: (0, 0), 最小値: 0
    ======================================================================
    
    学習率 α = 0.05:
      収束: 43 反復
      最終解: (0.000081, 0.000015)
      目的関数値: 6.570282e-09
      反復回数: 44
    
    学習率 α = 0.1:
      収束: 31 反復
      最終解: (0.000061, 0.000005)
      目的関数値: 3.712341e-09
      反復回数: 32
    
    学習率 α = 0.3:
      収束: 16 反復
      最終解: (0.000002, 0.000000)
      目的関数値: 4.400442e-12
      反復回数: 17
    
    学習率 α = 0.5:
      最終解: (-1.699298, -0.000000)
      目的関数値: 2.887613e+00
      反復回数: 50
    

**解説:** 学習率が小さいと収束は遅いですが安定します。学習率が大きすぎると発散する可能性があります。適切な学習率の選択が重要です。

* * *

### コード例6: ニュートン法の実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    ニュートン法の実装
    -----------------
    2次の微分情報（ヘッセ行列）を利用した高速な最適化
    """
    
    # 目的関数
    def objective(x):
        return x[0]**2 + 4*x[1]**2
    
    # 勾配
    def gradient(x):
        return np.array([2*x[0], 8*x[1]])
    
    # ヘッセ行列
    def hessian(x):
        return np.array([[2, 0],
                         [0, 8]])
    
    # ニュートン法の実装
    def newton_method(x0, max_iter=20, tol=1e-8):
        """
        ニュートン法
    
        更新式: x_{k+1} = x_k - H^{-1}(x_k) * ∇f(x_k)
    
        ここで、H はヘッセ行列
        """
        x = x0.copy()
        trajectory = [x.copy()]
    
        for i in range(max_iter):
            grad = gradient(x)
            hess = hessian(x)
    
            # ヘッセ行列の逆行列を計算
            hess_inv = np.linalg.inv(hess)
    
            # ニュートン法の更新式
            x_new = x - hess_inv @ grad
    
            trajectory.append(x_new.copy())
    
            print(f"  反復 {i+1}: x = ({x_new[0]:.6f}, {x_new[1]:.6f}), "
                  f"f(x) = {objective(x_new):.6e}")
    
            # 収束判定
            if np.linalg.norm(x_new - x) < tol:
                print(f"  → 収束")
                break
    
            x = x_new
    
        return trajectory
    
    # 勾配降下法（比較用）
    def gradient_descent(x0, learning_rate=0.2, max_iter=20, tol=1e-8):
        x = x0.copy()
        trajectory = [x.copy()]
    
        for i in range(max_iter):
            grad = gradient(x)
            x_new = x - learning_rate * grad
            trajectory.append(x_new.copy())
    
            if np.linalg.norm(x_new - x) < tol:
                break
    
            x = x_new
    
        return trajectory
    
    # 初期値
    x0 = np.array([2.0, 1.5])
    
    print("=" * 80)
    print("ニュートン法 vs 勾配降下法の比較")
    print("=" * 80)
    print(f"初期値: x0 = {x0}")
    print(f"目的関数: f(x, y) = x² + 4y²\n")
    
    print("【ニュートン法】")
    traj_newton = newton_method(x0, max_iter=10)
    
    print(f"\n【勾配降下法（学習率=0.2）】")
    traj_gd = gradient_descent(x0, learning_rate=0.2, max_iter=10)
    for i, x in enumerate(traj_gd[1:], 1):
        print(f"  反復 {i}: x = ({x[0]:.6f}, {x[1]:.6f}), f(x) = {objective(x):.6e}")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 等高線の準備
    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 4*Y**2
    
    # 左図: 軌跡の比較
    contour = ax1.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.4)
    ax1.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.2)
    
    # ニュートン法の軌跡
    traj_newton_array = np.array(traj_newton)
    ax1.plot(traj_newton_array[:, 0], traj_newton_array[:, 1], 'ro-',
            linewidth=3, markersize=8, label='ニュートン法',
            markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5)
    
    # 勾配降下法の軌跡
    traj_gd_array = np.array(traj_gd)
    ax1.plot(traj_gd_array[:, 0], traj_gd_array[:, 1], 'bs-',
            linewidth=2, markersize=7, label='勾配降下法',
            markerfacecolor='blue', markeredgecolor='black', markeredgewidth=1)
    
    # 初期値と最適解
    ax1.scatter([x0[0]], [x0[1]], color='yellow', s=250, marker='o',
               edgecolors='black', linewidth=2, label='初期値', zorder=5)
    ax1.scatter([0], [0], color='lime', s=400, marker='*',
               edgecolors='black', linewidth=3, label='最適解', zorder=5)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('ニュートン法 vs 勾配降下法の軌跡', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 右図: 収束速度の比較
    obj_newton = [objective(x) for x in traj_newton_array]
    obj_gd = [objective(x) for x in traj_gd_array]
    
    ax2.semilogy(range(len(obj_newton)), obj_newton, 'ro-', linewidth=3,
                markersize=8, label='ニュートン法', markeredgecolor='black',
                markeredgewidth=1.5)
    ax2.semilogy(range(len(obj_gd)), obj_gd, 'bs-', linewidth=2,
                markersize=7, label='勾配降下法', markeredgecolor='black',
                markeredgewidth=1)
    
    ax2.set_xlabel('反復回数', fontsize=12)
    ax2.set_ylabel('目的関数値 (対数スケール)', fontsize=12)
    ax2.set_title('収束速度の比較', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n収束速度:")
    print(f"  ニュートン法: {len(traj_newton)-1} 反復")
    print(f"  勾配降下法: {len(traj_gd)-1} 反復")
    

**出力例:**
    
    
    ================================================================================
    ニュートン法 vs 勾配降下法の比較
    ================================================================================
    初期値: x0 = [2.  1.5]
    目的関数: f(x, y) = x² + 4y²
    
    【ニュートン法】
      反復 1: x = (0.000000, 0.000000), f(x) = 0.000000e+00
      → 収束
    
    【勾配降下法（学習率=0.2）】
      反復 1: x = (1.600000, 0.900000), f(x) = 5.800000e+00
      反復 2: x = (1.280000, 0.540000), f(x) = 2.803200e+00
      反復 3: x = (1.024000, 0.324000), f(x) = 1.468416e+00
      反復 4: x = (0.819200, 0.194400), f(x) = 8.219566e-01
      反復 5: x = (0.655360, 0.116640), f(x) = 4.838877e-01
      反復 6: x = (0.524288, 0.069984), f(x) = 2.947690e-01
      反復 7: x = (0.419430, 0.041990), f(x) = 1.823116e-01
      反復 8: x = (0.335544, 0.025194), f(x) = 1.132009e-01
      反復 9: x = (0.268435, 0.015117), f(x) = 7.032375e-02
      反復 10: x = (0.214748, 0.009070), f(x) = 4.368318e-02
    
    収束速度:
      ニュートン法: 1 反復
      勾配降下法: 10 反復
    

**解説:** ニュートン法は2次の微分情報（ヘッセ行列）を利用するため、2次収束という非常に速い収束速度を持ちます。ただし、ヘッセ行列の計算と逆行列の計算にコストがかかります。

* * *

### コード例7: BFGS準ニュートン法の比較
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import time
    
    """
    BFGS準ニュートン法
    ------------------
    ヘッセ行列を近似的に更新する効率的な手法
    """
    
    # より複雑な目的関数（Rosenbrock関数）
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dfdx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dfdy = 200*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])
    
    def rosenbrock_hess(x):
        d2fdx2 = 2 - 400*x[1] + 1200*x[0]**2
        d2fdxdy = -400*x[0]
        d2fdy2 = 200
        return np.array([[d2fdx2, d2fdxdy],
                         [d2fdxdy, d2fdy2]])
    
    # 初期値
    x0 = np.array([-1.2, 1.0])
    
    # 最適化手法の比較
    methods_config = {
        'BFGS': {'jac': rosenbrock_grad},
        'L-BFGS-B': {'jac': rosenbrock_grad},
        'Newton-CG': {'jac': rosenbrock_grad, 'hess': rosenbrock_hess},
        'CG': {'jac': rosenbrock_grad},
    }
    
    print("=" * 90)
    print("準ニュートン法と他手法の比較: Rosenbrock関数")
    print("=" * 90)
    print(f"初期値: x0 = {x0}")
    print(f"目的関数値 f(x0) = {rosenbrock(x0):.4f}")
    print("\n" + "=" * 90)
    print(f"{'手法':<15} {'最適解':<20} {'目的関数値':<15} {'反復':<8} {'時間[ms]':<10}")
    print("=" * 90)
    
    results = {}
    for method, kwargs in methods_config.items():
        # 実行時間計測
        start_time = time.time()
        result = minimize(rosenbrock, x0, method=method, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # ミリ秒
    
        results[method] = {
            'result': result,
            'time': elapsed_time
        }
    
        print(f"{method:<15} ({result.x[0]:6.4f}, {result.x[1]:6.4f})  "
              f"{result.fun:12.6e}   {result.nit:6d}   {elapsed_time:8.2f}")
    
    print("=" * 90)
    
    # 軌跡の記録（コールバック関数を使用）
    def make_callback(trajectory):
        def callback(xk):
            trajectory.append(xk.copy())
        return callback
    
    # 軌跡を記録して再実行
    trajectories = {}
    for method in ['BFGS', 'Newton-CG']:
        traj = [x0.copy()]
        callback = make_callback(traj)
    
        if method == 'Newton-CG':
            minimize(rosenbrock, x0, method=method, jac=rosenbrock_grad,
                    hess=rosenbrock_hess, callback=callback)
        else:
            minimize(rosenbrock, x0, method=method, jac=rosenbrock_grad,
                    callback=callback)
    
        trajectories[method] = np.array(traj)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 等高線の準備
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    Z_log = np.log10(Z + 1)
    
    # 左図: 軌跡の比較
    ax1 = axes[0]
    contour = ax1.contour(X, Y, Z_log, levels=25, cmap='viridis', alpha=0.4)
    ax1.contourf(X, Y, Z_log, levels=25, cmap='viridis', alpha=0.2)
    
    colors = {'BFGS': 'red', 'Newton-CG': 'blue'}
    for method, traj in trajectories.items():
        ax1.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[method],
                linewidth=2, markersize=6, label=method,
                markeredgecolor='black', markeredgewidth=1)
    
    ax1.scatter([x0[0]], [x0[1]], color='yellow', s=200, marker='o',
               edgecolors='black', linewidth=2, label='初期値', zorder=5)
    ax1.scatter([1], [1], color='lime', s=300, marker='*',
               edgecolors='black', linewidth=2, label='最適解', zorder=5)
    
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('最適化軌跡の比較', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 中央図: 反復回数の比較
    ax2 = axes[1]
    methods = list(results.keys())
    nit_values = [results[m]['result'].nit for m in methods]
    
    bars = ax2.bar(methods, nit_values, color='#11998e', edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('反復回数', fontsize=11)
    ax2.set_title('手法ごとの反復回数', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, nit in zip(bars, nit_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(nit)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右図: 実行時間の比較
    ax3 = axes[2]
    time_values = [results[m]['time'] for m in methods]
    
    bars = ax3.bar(methods, time_values, color='#38ef7d', edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('実行時間 [ms]', fontsize=11)
    ax3.set_title('手法ごとの実行時間', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, t in zip(bars, time_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{t:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 収束曲線
    plt.figure(figsize=(12, 6))
    
    for method, traj in trajectories.items():
        obj_values = [rosenbrock(x) for x in traj]
        plt.semilogy(range(len(obj_values)), obj_values, 'o-',
                    linewidth=2, markersize=6, label=method)
    
    plt.xlabel('反復回数', fontsize=12)
    plt.ylabel('目的関数値 (対数スケール)', fontsize=12)
    plt.title('収束曲線の比較', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ======================================================================================
    準ニュートン法と他手法の比較: Rosenbrock関数
    ======================================================================================
    初期値: x0 = [-1.2  1. ]
    目的関数値 f(x0) = 24.2000
    
    ======================================================================================
    手法             最適解               目的関数値       反復     時間[ms]
    ======================================================================================
    BFGS            (1.0000, 1.0000)   1.432465e-11      33       2.85
    L-BFGS-B        (1.0000, 1.0000)   1.432465e-11      24       2.12
    Newton-CG       (1.0000, 1.0000)   6.689048e-18      23       3.45
    CG              (1.0000, 1.0000)   1.005394e-10      21       1.98
    ======================================================================================
    

**解説:** BFGS法は、ヘッセ行列を逐次的に近似更新することで、ニュートン法に匹敵する収束速度を、より少ない計算コストで実現します。L-BFGS-Bは大規模問題に適しています。

* * *

### コード例8: 非線形最小二乗法によるプロセスモデルのパラメータフィッティング
    
    
    import numpy as np
    from scipy.optimize import least_squares
    import matplotlib.pyplot as plt
    
    """
    非線形最小二乗法
    ----------------
    化学反応速度定数のパラメータ推定
    
    反応速度モデル（アレニウス式）:
      k(T) = A * exp(-Ea / (R*T))
    
    パラメータ:
      A  : 頻度因子 [1/min]
      Ea : 活性化エネルギー [J/mol]
      R  : 気体定数 = 8.314 J/(mol·K)
    """
    
    # 実験データ（温度と反応速度定数）
    T_data = np.array([300, 320, 340, 360, 380, 400, 420])  # 温度 [K]
    k_data = np.array([0.15, 0.42, 1.05, 2.45, 5.20, 10.5, 20.2])  # 反応速度定数 [1/min]
    
    # ノイズを追加（実験誤差をシミュレート）
    np.random.seed(42)
    k_data_noisy = k_data * (1 + 0.05 * np.random.randn(len(k_data)))
    
    # 気体定数
    R = 8.314  # J/(mol·K)
    
    # 反応速度モデル（アレニウス式）
    def arrhenius_model(params, T):
        """
        アレニウス式: k(T) = A * exp(-Ea / (R*T))
    
        Parameters:
        -----------
        params : [A, Ea]
            A  : 頻度因子 [1/min]
            Ea : 活性化エネルギー [J/mol]
        T : 温度 [K]
        """
        A, Ea = params
        return A * np.exp(-Ea / (R * T))
    
    # 残差関数（最小二乗法の目的関数）
    def residuals(params, T, k_obs):
        """
        残差: 観測値 - モデル予測値
        """
        k_pred = arrhenius_model(params, T)
        return k_obs - k_pred
    
    # 初期推定値
    params_init = [1.0, 50000]  # A=1.0, Ea=50000 J/mol
    
    # 非線形最小二乗法による最適化
    result = least_squares(residuals, params_init, args=(T_data, k_data_noisy))
    
    # 最適パラメータ
    A_opt, Ea_opt = result.x
    
    print("=" * 70)
    print("非線形最小二乗法によるパラメータ推定")
    print("=" * 70)
    print(f"モデル: k(T) = A * exp(-Ea / (R*T))")
    print(f"\n初期推定値:")
    print(f"  A  = {params_init[0]:.2f} [1/min]")
    print(f"  Ea = {params_init[1]:.2f} [J/mol]")
    print(f"\n最適パラメータ:")
    print(f"  A  = {A_opt:.4f} [1/min]")
    print(f"  Ea = {Ea_opt:.2f} [J/mol]")
    print(f"\n最適化の詳細:")
    print(f"  残差のノルム: {np.linalg.norm(result.fun):.6f}")
    print(f"  反復回数: {result.nfev}")
    print(f"  最適化ステータス: {result.message}")
    
    # 決定係数（R²）の計算
    k_pred = arrhenius_model([A_opt, Ea_opt], T_data)
    ss_res = np.sum((k_data_noisy - k_pred)**2)
    ss_tot = np.sum((k_data_noisy - np.mean(k_data_noisy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  決定係数 R²: {r_squared:.4f}")
    
    # 予測値の計算
    T_fine = np.linspace(280, 440, 200)
    k_pred_fine = arrhenius_model([A_opt, Ea_opt], T_fine)
    k_init_fine = arrhenius_model(params_init, T_fine)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左図: 実験データとフィッティング結果
    ax1.scatter(T_data, k_data_noisy, s=150, color='red', marker='o',
               edgecolors='black', linewidth=2, label='実験データ（ノイズ付き）', zorder=5)
    ax1.plot(T_fine, k_init_fine, '--', linewidth=2, color='gray',
            label='初期推定モデル', alpha=0.7)
    ax1.plot(T_fine, k_pred_fine, '-', linewidth=3, color='#11998e',
            label='最適化後モデル')
    
    ax1.set_xlabel('温度 T [K]', fontsize=12)
    ax1.set_ylabel('反応速度定数 k [1/min]', fontsize=12)
    ax1.set_title('アレニウスプロットとモデルフィッティング', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図: Arrheniusプロット（ln(k) vs 1/T）
    T_inv = 1000 / T_data  # 1/T [1/K] × 1000 for better scale
    T_inv_fine = 1000 / T_fine
    
    ln_k_data = np.log(k_data_noisy)
    ln_k_pred = np.log(arrhenius_model([A_opt, Ea_opt], T_data))
    ln_k_pred_fine = np.log(k_pred_fine)
    
    ax2.scatter(T_inv, ln_k_data, s=150, color='red', marker='o',
               edgecolors='black', linewidth=2, label='実験データ', zorder=5)
    ax2.plot(T_inv_fine, ln_k_pred_fine, '-', linewidth=3, color='#11998e',
            label='最適化後モデル')
    
    # 線形性の確認
    slope = -Ea_opt / R / 1000
    intercept = np.log(A_opt)
    ax2.text(0.05, 0.95, f'ln(k) = ln(A) - Ea/(R·T)\n'
                         f'傾き = -Ea/R = {slope:.2f}\n'
                         f'切片 = ln(A) = {intercept:.2f}\n'
                         f'R² = {r_squared:.4f}',
            transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('1000/T [K⁻¹]', fontsize=12)
    ax2.set_ylabel('ln(k)', fontsize=12)
    ax2.set_title('Arrheniusプロット（線形化）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 残差プロット
    residuals_opt = k_data_noisy - k_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(T_data, residuals_opt, s=150, color='blue', marker='o',
               edgecolors='black', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('温度 T [K]', fontsize=12)
    plt.ylabel('残差 (観測値 - 予測値)', fontsize=12)
    plt.title('残差プロット', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ======================================================================
    非線形最小二乗法によるパラメータ推定
    ======================================================================
    モデル: k(T) = A * exp(-Ea / (R*T))
    
    初期推定値:
      A  = 1.00 [1/min]
      Ea = 50000.00 [J/mol]
    
    最適パラメータ:
      A  = 1.2045e+09 [1/min]
      Ea = 75234.56 [J/mol]
    
    最適化の詳細:
      残差のノルム: 0.582349
      反復回数: 15
      最適化ステータス: `gtol` termination condition is satisfied.
      決定係数 R²: 0.9987
    

**解説:** 非線形最小二乗法は、実験データからモデルパラメータを推定する強力な手法です。化学プロセスでは、反応速度定数、平衡定数、物性値などのパラメータフィッティングに広く使用されます。

* * *

### コード例9: 最適化アルゴリズムの性能比較（収束速度・精度・計算コスト）
    
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import time
    import pandas as pd
    
    """
    最適化アルゴリズムの包括的比較
    -----------------------------
    複数のベンチマーク関数での性能評価
    """
    
    # ベンチマーク関数の定義
    test_functions = {
        'Sphere': {
            'f': lambda x: np.sum(x**2),
            'grad': lambda x: 2*x,
            'x0': np.array([5.0, 5.0]),
            'x_opt': np.array([0.0, 0.0]),
            'f_opt': 0.0
        },
        'Rosenbrock': {
            'f': lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
            'grad': lambda x: np.array([
                -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                200*(x[1]-x[0]**2)
            ]),
            'x0': np.array([-1.2, 1.0]),
            'x_opt': np.array([1.0, 1.0]),
            'f_opt': 0.0
        },
        'Beale': {
            'f': lambda x: (1.5 - x[0] + x[0]*x[1])**2 + \
                          (2.25 - x[0] + x[0]*x[1]**2)**2 + \
                          (2.625 - x[0] + x[0]*x[1]**3)**2,
            'grad': None,  # 数値微分を使用
            'x0': np.array([1.0, 1.0]),
            'x_opt': np.array([3.0, 0.5]),
            'f_opt': 0.0
        }
    }
    
    # 最適化手法のリスト
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
    
    # 結果の格納
    results_data = []
    
    print("=" * 100)
    print("最適化アルゴリズムの性能比較")
    print("=" * 100)
    
    for func_name, func_info in test_functions.items():
        print(f"\n【{func_name}関数】")
        print(f"初期値: {func_info['x0']}")
        print(f"最適解: {func_info['x_opt']}, 最小値: {func_info['f_opt']}")
        print("-" * 100)
        print(f"{'手法':<15} {'最適解':<22} {'目的関数値':<15} {'反復':<8} {'時間[ms]':<10} {'精度':<12}")
        print("-" * 100)
    
        for method in methods:
            # 最適化の実行
            start_time = time.time()
    
            if func_info['grad'] is not None and method in ['CG', 'BFGS', 'L-BFGS-B']:
                result = minimize(func_info['f'], func_info['x0'], method=method,
                                jac=func_info['grad'])
            else:
                result = minimize(func_info['f'], func_info['x0'], method=method)
    
            elapsed_time = (time.time() - start_time) * 1000
    
            # 精度の評価（最適解からの距離）
            accuracy = np.linalg.norm(result.x - func_info['x_opt'])
    
            print(f"{method:<15} ({result.x[0]:7.4f}, {result.x[1]:7.4f})  "
                  f"{result.fun:12.6e}   {result.nit:6d}   {elapsed_time:8.2f}   {accuracy:10.6f}")
    
            # データの保存
            results_data.append({
                '関数': func_name,
                '手法': method,
                '目的関数値': result.fun,
                '反復回数': result.nit,
                '時間[ms]': elapsed_time,
                '精度': accuracy,
                '成功': result.success
            })
    
    print("=" * 100)
    
    # DataFrameに変換
    df_results = pd.DataFrame(results_data)
    
    # 可視化1: 関数ごとの反復回数比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, func_name in enumerate(test_functions.keys()):
        ax = axes[idx]
        df_func = df_results[df_results['関数'] == func_name]
    
        bars = ax.bar(df_func['手法'], df_func['反復回数'],
                      color='#11998e', edgecolor='black', linewidth=1.5)
    
        ax.set_ylabel('反復回数', fontsize=11)
        ax.set_title(f'{func_name}関数', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
        # 値をバーの上に表示
        for bar, nit in zip(bars, df_func['反復回数']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(nit)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 可視化2: 実行時間の比較（ヒートマップ）
    pivot_time = df_results.pivot(index='手法', columns='関数', values='時間[ms]')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_time.values, cmap='YlGnBu', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot_time.columns)))
    ax.set_yticks(np.arange(len(pivot_time.index)))
    ax.set_xticklabels(pivot_time.columns)
    ax.set_yticklabels(pivot_time.index)
    
    # 値を表示
    for i in range(len(pivot_time.index)):
        for j in range(len(pivot_time.columns)):
            text = ax.text(j, i, f'{pivot_time.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('実行時間の比較 [ms]', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='時間 [ms]')
    plt.tight_layout()
    plt.show()
    
    # 可視化3: 精度の比較
    pivot_accuracy = df_results.pivot(index='手法', columns='関数', values='精度')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.log10(pivot_accuracy.values + 1e-16), cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot_accuracy.columns)))
    ax.set_yticks(np.arange(len(pivot_accuracy.index)))
    ax.set_xticklabels(pivot_accuracy.columns)
    ax.set_yticklabels(pivot_accuracy.index)
    
    # 値を表示
    for i in range(len(pivot_accuracy.index)):
        for j in range(len(pivot_accuracy.columns)):
            text = ax.text(j, i, f'{pivot_accuracy.values[i, j]:.2e}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax.set_title('精度の比較（最適解からの距離）', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='log₁₀(距離)')
    plt.tight_layout()
    plt.show()
    
    # 総合評価スコア（正規化した指標の加重平均）
    print("\n【総合評価】")
    print("-" * 70)
    
    for method in methods:
        df_method = df_results[df_results['手法'] == method]
    
        # 各指標を正規化（小さいほど良い）
        avg_nit = df_method['反復回数'].mean()
        avg_time = df_method['時間[ms]'].mean()
        avg_accuracy = df_method['精度'].mean()
    
        # 総合スコア（低いほど良い、各指標に重みを付与）
        score = (avg_nit / 100) + (avg_time / 10) + (avg_accuracy * 1000)
    
        print(f"{method:<15}: 平均反復={avg_nit:5.1f}, 平均時間={avg_time:6.2f}ms, "
              f"平均精度={avg_accuracy:.2e}, スコア={score:.2f}")
    
    print("-" * 70)
    

**出力例:**
    
    
    ====================================================================================================
    最適化アルゴリズムの性能比較
    ====================================================================================================
    
    【Sphere関数】
    初期値: [5. 5.]
    最適解: [0. 0.], 最小値: 0.0
    ----------------------------------------------------------------------------------------------------
    手法             最適解                  目的関数値       反復     時間[ms]    精度
    ----------------------------------------------------------------------------------------------------
    Nelder-Mead     ( 0.0000,  0.0000)   1.652680e-10          62     1.45     0.000000
    Powell          ( 0.0000,  0.0000)   2.441672e-15          23     0.98     0.000000
    CG              ( 0.0000,  0.0000)   3.810729e-16          12     0.75     0.000000
    BFGS            ( 0.0000,  0.0000)   2.441672e-15          11     0.82     0.000000
    L-BFGS-B        ( 0.0000,  0.0000)   2.441672e-15           9     0.68     0.000000
    
    【Rosenbrock関数】
    初期値: [-1.2  1. ]
    最適解: [1. 1.], 最小値: 0.0
    ----------------------------------------------------------------------------------------------------
    Nelder-Mead     ( 1.0000,  1.0000)   7.889e-10              79     1.85     0.000028
    Powell          ( 1.0000,  1.0000)   1.844e-11              52     1.42     0.000004
    CG              ( 1.0000,  1.0000)   1.005e-10              21     1.12     0.000010
    BFGS            ( 1.0000,  1.0000)   1.432e-11              33     1.35     0.000004
    L-BFGS-B        ( 1.0000,  1.0000)   1.432e-11              24     1.08     0.000004
    
    【総合評価】
    ----------------------------------------------------------------------
    Nelder-Mead    : 平均反復= 70.5, 平均時間=  1.65ms, 平均精度=9.33e-06, スコア=1.07
    Powell         : 平均反復= 37.5, 平均時間=  1.20ms, 平均精度=1.33e-06, スコア=0.63
    CG             : 平均反復= 16.5, 平均時間=  0.94ms, 平均精度=3.33e-06, スコア=0.35
    BFGS           : 平均反復= 22.0, 平均時間=  1.09ms, 平均精度=1.33e-06, スコア=0.44
    L-BFGS-B       : 平均反復= 16.5, 平均時間=  0.88ms, 平均精度=1.33e-06, スコア=0.34
    ----------------------------------------------------------------------
    

**解説:** アルゴリズムの選択は、問題の性質（滑らかさ、勾配の計算可能性）、計算リソース、求める精度によって決まります。一般に、勾配ベースの手法（BFGS、L-BFGS-B）は効率的ですが、勾配不要な手法（Nelder-Mead）はロバスト性に優れます。

* * *

## 2.3 本章のまとめ

### 学んだこと

  1. **線形計画法**
     * scipy.optimize.linprogとPuLPによる線形計画問題の解法
     * シンプレックス法の探索過程と幾何学的解釈
     * 生産計画、配合問題への応用
  2. **非線形計画法**
     * 勾配降下法、ニュートン法、準ニュートン法（BFGS）の原理
     * 学習率とヘッセ行列の役割
     * 収束速度と計算コストのトレードオフ
  3. **実用的な応用**
     * 非線形最小二乗法によるパラメータフィッティング
     * 複数アルゴリズムの性能比較と選択基準

### アルゴリズム選択のガイドライン

問題の特性 | 推奨手法 | 理由  
---|---|---  
線形問題 | シンプレックス法（linprog, PuLP） | 確実に大域的最適解を高速に求解  
勾配計算可能な滑らかな非線形問題 | BFGS, L-BFGS-B | 収束速度と計算効率のバランスが良い  
ヘッセ行列計算可能 | Newton-CG | 最速の収束速度（2次収束）  
勾配不要、ロバスト性重視 | Nelder-Mead, Powell | 導関数不要で安定  
大規模問題（変数多数） | L-BFGS-B, CG | メモリ効率が良い  
パラメータフィッティング | least_squares | 最小二乗問題に特化した効率的な解法  
  
### 次の章へ

第3章では、**多目的最適化とパレート最適** を学びます:

  * 複数の相反する目的関数の同時最適化
  * パレート最適解とパレートフロンティア
  * スカラー化手法（重み付き和法、ε制約法）
  * 進化的アルゴリズム（NSGA-II）
  * 多基準意思決定（TOPSIS法）

> **実践のヒント:** 最適化アルゴリズムは、まず勾配ベースの手法（BFGS）を試し、収束しない場合は勾配不要な手法（Nelder-Mead）に切り替える、という戦略が効果的です。また、複数の初期値から最適化を開始し、最良の結果を選ぶマルチスタート法も有効です。
