---
title: 第3章：最適化理論
chapter_title: 第3章：最適化理論
subtitle: 機械学習のための数学基礎 - 勾配降下法・制約付き最適化・凸最適化
reading_time: 30-40分
difficulty: 上級
code_examples: 6
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 最適化問題の定式化と凸性の概念を理解する
  * ✅ 勾配降下法の原理と実装方法を習得する
  * ✅ Momentum、Adam等の最適化アルゴリズムを使い分けられる
  * ✅ ラグランジュ乗数法とKKT条件を理解する
  * ✅ 凸最適化の理論と実践を理解する
  * ✅ 機械学習モデルの最適化を実装できる

* * *

## 3.1 最適化の基礎

### 最適化問題の定式化

**最適化問題（Optimization Problem）** は、目的関数を最小化または最大化する問題です。

$$ \begin{aligned} \min_{\mathbf{x}} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\\ & h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \end{aligned} $$

  * $f(\mathbf{x})$: 目的関数（最小化対象）
  * $g_i(\mathbf{x}) \leq 0$: 不等式制約
  * $h_j(\mathbf{x}) = 0$: 等式制約

### 凸関数と凸集合

**凸集合（Convex Set）** ：2点を結ぶ線分が集合内に含まれる

$$ \mathbf{x}, \mathbf{y} \in C, \ \theta \in [0, 1] \Rightarrow \theta \mathbf{x} + (1-\theta) \mathbf{y} \in C $$

**凸関数（Convex Function）** ：2点間の関数値が線分以下

$$ f(\theta \mathbf{x} + (1-\theta) \mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta) f(\mathbf{y}) $$

> **重要性** ：凸最適化問題は局所最適解 = 大域最適解が保証される

### 凸性の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 凸関数の例：二次関数
    def convex_function(x, y):
        return x**2 + y**2
    
    # 非凸関数の例：Himmelblau関数
    def non_convex_function(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    # グリッドの作成
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    Z_convex = convex_function(X, Y)
    Z_non_convex = non_convex_function(X, Y)
    
    # 可視化
    fig = plt.figure(figsize=(15, 6))
    
    # 凸関数
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z_convex, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.set_title('凸関数: $f(x, y) = x^2 + y^2$', fontsize=14)
    
    # 非凸関数
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_non_convex, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x, y)')
    ax2.set_title('非凸関数: Himmelblau関数', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 凸性の確認 ===")
    print("凸関数: 単一の大域最適解（原点）")
    print("非凸関数: 複数の局所最適解が存在")
    

### 勾配とヘシアン

**勾配（Gradient）** ：関数の変化率を表すベクトル

$$ \nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\\ \frac{\partial f}{\partial x_2} \\\ \vdots \\\ \frac{\partial f}{\partial x_n} \end{bmatrix} $$

**ヘシアン行列（Hessian Matrix）** ：2次偏微分の行列

$$ \mathbf{H}(f) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\\ \vdots & \vdots & \ddots \end{bmatrix} $$

### 最適性条件

**1次条件（必要条件）** ：

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

**2次条件（十分条件）** ：

$$\mathbf{H}(f)(\mathbf{x}^*) \succeq 0 \quad \text{(半正定値)}$$

* * *

## 3.2 勾配降下法

### 勾配降下法の原理

**勾配降下法（Gradient Descent）** は、勾配の逆方向に反復的に移動して最適解を探索します。

$$ \mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t) $$

  * $\alpha$: 学習率（ステップサイズ）
  * $\nabla f(\mathbf{x}_t)$: 現在位置での勾配

### 学習率の選択
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 目的関数：f(x) = x^2 + 4x + 4
    def f(x):
        return x**2 + 4*x + 4
    
    # 勾配：f'(x) = 2x + 4
    def grad_f(x):
        return 2*x + 4
    
    # 勾配降下法
    def gradient_descent(x0, lr, n_iterations):
        x = x0
        trajectory = [x]
    
        for _ in range(n_iterations):
            x = x - lr * grad_f(x)
            trajectory.append(x)
    
        return np.array(trajectory)
    
    # 異なる学習率での実験
    learning_rates = [0.1, 0.5, 0.9, 1.1]
    x0 = 5.0
    n_iter = 20
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x_range = np.linspace(-3, 6, 100)
    y_range = f(x_range)
    
    for i, lr in enumerate(learning_rates):
        trajectory = gradient_descent(x0, lr, n_iter)
    
        axes[i].plot(x_range, y_range, 'b-', linewidth=2, label='$f(x) = x^2 + 4x + 4$')
        axes[i].plot(trajectory, f(trajectory), 'ro-', markersize=6,
                     linewidth=1.5, alpha=0.7, label='最適化の軌跡')
        axes[i].plot(-2, 0, 'g*', markersize=20, label='最適解')
    
        axes[i].set_xlabel('x', fontsize=12)
        axes[i].set_ylabel('f(x)', fontsize=12)
        axes[i].set_title(f'学習率 α = {lr}', fontsize=14)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
        # 収束判定
        if abs(trajectory[-1] - (-2)) < 0.01:
            status = "✓ 収束"
        elif lr >= 1.0:
            status = "✗ 発散"
        else:
            status = "△ 収束が遅い"
    
        axes[i].text(0.05, 0.95, status, transform=axes[i].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print("=== 学習率の影響 ===")
    print("α = 0.1: 収束が遅い")
    print("α = 0.5: 適切な収束")
    print("α = 0.9: 高速収束")
    print("α = 1.1: 発散（学習率が大きすぎる）")
    

### 確率的勾配降下法（SGD）

**バッチ勾配降下法** ：全データで勾配計算（遅い）

**確率的勾配降下法（SGD）** ：1サンプルで勾配計算（速い、ノイズあり）

**ミニバッチ勾配降下法** ：小バッチで勾配計算（バランス良い）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ生成（線形回帰）
    np.random.seed(42)
    n_samples = 100
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    
    # パラメータ初期化
    theta_batch = np.random.randn(2, 1)
    theta_sgd = theta_batch.copy()
    theta_minibatch = theta_batch.copy()
    
    # バイアス項追加
    X_b = np.c_[np.ones((n_samples, 1)), X]
    
    # ハイパーパラメータ
    n_epochs = 50
    learning_rate = 0.01
    batch_size = 10
    
    # 損失関数（MSE）
    def compute_loss(X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        return loss
    
    # 勾配計算
    def compute_gradient(X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        return gradient
    
    # 学習履歴
    history_batch = []
    history_sgd = []
    history_minibatch = []
    
    # バッチ勾配降下法
    for epoch in range(n_epochs):
        gradient = compute_gradient(X_b, y, theta_batch)
        theta_batch -= learning_rate * gradient
        history_batch.append(compute_loss(X_b, y, theta_batch))
    
    # 確率的勾配降下法
    for epoch in range(n_epochs):
        for i in range(n_samples):
            random_index = np.random.randint(n_samples)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = compute_gradient(xi, yi, theta_sgd)
            theta_sgd -= learning_rate * gradient
        history_sgd.append(compute_loss(X_b, y, theta_sgd))
    
    # ミニバッチ勾配降下法
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
    
        for i in range(0, n_samples, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradient = compute_gradient(xi, yi, theta_minibatch)
            theta_minibatch -= learning_rate * gradient
        history_minibatch.append(compute_loss(X_b, y, theta_minibatch))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 学習曲線
    axes[0].plot(history_batch, label='バッチGD', linewidth=2)
    axes[0].plot(history_sgd, label='SGD', alpha=0.7, linewidth=2)
    axes[0].plot(history_minibatch, label='ミニバッチGD', linewidth=2)
    axes[0].set_xlabel('エポック', fontsize=12)
    axes[0].set_ylabel('損失（MSE）', fontsize=12)
    axes[0].set_title('学習曲線の比較', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 回帰直線
    axes[1].scatter(X, y, alpha=0.5, label='データ')
    x_plot = np.array([[0], [2]])
    x_plot_b = np.c_[np.ones((2, 1)), x_plot]
    
    axes[1].plot(x_plot, x_plot_b.dot(theta_batch), 'r-',
                 linewidth=2, label=f'バッチGD')
    axes[1].plot(x_plot, x_plot_b.dot(theta_sgd), 'g--',
                 linewidth=2, label=f'SGD')
    axes[1].plot(x_plot, x_plot_b.dot(theta_minibatch), 'b:',
                 linewidth=2, label=f'ミニバッチGD')
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('学習された回帰直線', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 最終パラメータ ===")
    print(f"バッチGD:      θ0={theta_batch[0][0]:.3f}, θ1={theta_batch[1][0]:.3f}")
    print(f"SGD:           θ0={theta_sgd[0][0]:.3f}, θ1={theta_sgd[1][0]:.3f}")
    print(f"ミニバッチGD:  θ0={theta_minibatch[0][0]:.3f}, θ1={theta_minibatch[1][0]:.3f}")
    print(f"\n真の値:        θ0=4.000, θ1=3.000")
    

### 高度な最適化アルゴリズム

#### Momentum

勾配の移動平均を使い、振動を抑制します。

$$ \begin{aligned} \mathbf{v}_{t+1} &= \beta \mathbf{v}_t - \alpha \nabla f(\mathbf{x}_t) \\\ \mathbf{x}_{t+1} &= \mathbf{x}_t + \mathbf{v}_{t+1} \end{aligned} $$

#### Adam（Adaptive Moment Estimation）

勾配の1次と2次モーメントを適応的に調整します。

$$ \begin{aligned} \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla f(\mathbf{x}_t) \\\ \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla f(\mathbf{x}_t))^2 \\\ \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t} \\\ \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1-\beta_2^t} \\\ \mathbf{x}_{t+1} &= \mathbf{x}_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \end{aligned} $$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Rosenbrock関数（最適化の難しい非凸関数）
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # 最適化アルゴリズムの実装
    def sgd(start, grad_func, lr=0.001, n_iterations=1000):
        x = start.copy()
        trajectory = [x.copy()]
    
        for _ in range(n_iterations):
            grad = grad_func(x[0], x[1])
            x -= lr * grad
            trajectory.append(x.copy())
    
        return np.array(trajectory)
    
    def momentum(start, grad_func, lr=0.001, beta=0.9, n_iterations=1000):
        x = start.copy()
        v = np.zeros_like(x)
        trajectory = [x.copy()]
    
        for _ in range(n_iterations):
            grad = grad_func(x[0], x[1])
            v = beta * v - lr * grad
            x += v
            trajectory.append(x.copy())
    
        return np.array(trajectory)
    
    def adam(start, grad_func, lr=0.01, beta1=0.9, beta2=0.999,
             epsilon=1e-8, n_iterations=1000):
        x = start.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        trajectory = [x.copy()]
    
        for t in range(1, n_iterations + 1):
            grad = grad_func(x[0], x[1])
    
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
    
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
    
            x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            trajectory.append(x.copy())
    
        return np.array(trajectory)
    
    # 最適化の実行
    start_point = np.array([-1.0, 1.0])
    n_iter = 500
    
    traj_sgd = sgd(start_point, rosenbrock_grad, lr=0.0005, n_iterations=n_iter)
    traj_momentum = momentum(start_point, rosenbrock_grad, lr=0.0005, n_iterations=n_iter)
    traj_adam = adam(start_point, rosenbrock_grad, lr=0.01, n_iterations=n_iter)
    
    # 等高線プロット
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize=(15, 5))
    
    # SGD
    plt.subplot(131)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], 'r.-', markersize=3,
             linewidth=1, alpha=0.7, label='SGD')
    plt.plot(1, 1, 'g*', markersize=20, label='最適解')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SGD', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Momentum
    plt.subplot(132)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.plot(traj_momentum[:, 0], traj_momentum[:, 1], 'b.-', markersize=3,
             linewidth=1, alpha=0.7, label='Momentum')
    plt.plot(1, 1, 'g*', markersize=20, label='最適解')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Momentum', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adam
    plt.subplot(133)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.plot(traj_adam[:, 0], traj_adam[:, 1], 'm.-', markersize=3,
             linewidth=1, alpha=0.7, label='Adam')
    plt.plot(1, 1, 'g*', markersize=20, label='最適解')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 最適化アルゴリズムの比較 ===")
    print(f"SGD最終位置:      ({traj_sgd[-1][0]:.4f}, {traj_sgd[-1][1]:.4f})")
    print(f"Momentum最終位置: ({traj_momentum[-1][0]:.4f}, {traj_momentum[-1][1]:.4f})")
    print(f"Adam最終位置:     ({traj_adam[-1][0]:.4f}, {traj_adam[-1][1]:.4f})")
    print(f"真の最適解:       (1.0000, 1.0000)")
    

### 最適化アルゴリズムの比較

アルゴリズム | 長所 | 短所 | 推奨用途  
---|---|---|---  
**SGD** | シンプル、メモリ効率 | 収束が遅い、振動 | 大規模データ  
**Momentum** | 振動抑制、高速収束 | 慣性で行き過ぎ | 谷間の関数  
**AdaGrad** | 特徴ごとに学習率調整 | 学習率が急減 | 疎なデータ  
**RMSprop** | 学習率減衰緩和 | パラメータ調整必要 | RNN  
**Adam** | 適応的、高性能 | 過学習リスク | 汎用（デフォルト）  
  
* * *

## 3.3 制約付き最適化

### ラグランジュ乗数法

**等式制約付き最適化** ：

$$ \begin{aligned} \min_{\mathbf{x}} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & h(\mathbf{x}) = 0 \end{aligned} $$

**ラグランジュ関数** ：

$$ \mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x}) $$

**最適性条件** ：

$$ \begin{aligned} \nabla_{\mathbf{x}} \mathcal{L} &= 0 \\\ \nabla_{\lambda} \mathcal{L} &= 0 \end{aligned} $$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    # 目的関数：f(x, y) = (x - 2)^2 + (y - 1)^2
    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    # 等式制約：h(x, y) = x + y - 2 = 0
    def constraint_eq(x):
        return x[0] + x[1] - 2
    
    # 制約を辞書形式で定義
    constraints = {'type': 'eq', 'fun': constraint_eq}
    
    # 初期点
    x0 = np.array([0.0, 0.0])
    
    # 最適化
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    # 可視化
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 2)**2 + (Y - 1)**2
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='$f(x, y)$')
    
    # 制約線
    x_constraint = np.linspace(-1, 4, 100)
    y_constraint = 2 - x_constraint
    plt.plot(x_constraint, y_constraint, 'r-', linewidth=3, label='制約: $x + y = 2$')
    
    # 最適点
    plt.plot(result.x[0], result.x[1], 'r*', markersize=20, label='最適解')
    
    # 制約なし最適点
    plt.plot(2, 1, 'g*', markersize=20, label='制約なし最適解')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('ラグランジュ乗数法による等式制約付き最適化', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print("=== ラグランジュ乗数法の結果 ===")
    print(f"最適解: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"目的関数値: f(x*) = {result.fun:.4f}")
    print(f"制約充足: h(x*) = {constraint_eq(result.x):.6f}")
    print(f"制約なし最適解: x = 2.0, y = 1.0, f = 0.0")
    

### KKT条件

**不等式制約付き最適化** のための必要条件

$$ \begin{aligned} \min_{\mathbf{x}} \quad & f(\mathbf{x}) \\\ \text{subject to} \quad & g_i(\mathbf{x}) \leq 0 \end{aligned} $$

**KKT条件（Karush-Kuhn-Tucker Conditions）** ：

  1. **定常性** : $\nabla f(\mathbf{x}^*) + \sum_i \mu_i \nabla g_i(\mathbf{x}^*) = 0$
  2. **原初実行可能性** : $g_i(\mathbf{x}^*) \leq 0$
  3. **双対実行可能性** : $\mu_i \geq 0$
  4. **相補性** : $\mu_i g_i(\mathbf{x}^*) = 0$

### SVMへの応用

**サポートベクターマシン（SVM）** は制約付き最適化問題として定式化されます。

$$ \begin{aligned} \min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\\ \text{subject to} \quad & y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n \end{aligned} $$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.datasets import make_blobs
    
    # 線形分離可能なデータ生成
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                      cluster_std=1.0, center_box=(-5, 5))
    
    # SVMモデル（線形カーネル）
    svm = SVC(kernel='linear', C=1000)  # 大きなCでハードマージン近似
    svm.fit(X, y)
    
    # 決定境界の可視化
    def plot_svm_decision_boundary(ax, X, y, model):
        # グリッド作成
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
    
        # 決定境界
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        # プロット
        ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                   linestyles=['--', '-', '--'], colors=['r', 'k', 'b'], linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                   s=50, edgecolors='k', alpha=0.7)
    
        # サポートベクター
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=200, facecolors='none', edgecolors='g', linewidths=2,
                   label='サポートベクター')
    
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plot_svm_decision_boundary(ax, X, y, svm)
    plt.title('SVMによる線形分離（KKT条件による最適化）', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("=== SVM最適化の結果 ===")
    print(f"重みベクトル w: {svm.coef_[0]}")
    print(f"バイアス b: {svm.intercept_[0]:.4f}")
    print(f"サポートベクター数: {len(svm.support_vectors_)}")
    print(f"マージン: {2 / np.linalg.norm(svm.coef_):.4f}")
    

* * *

## 3.4 凸最適化

### 凸最適化の性質

> **重要な性質** ：凸最適化問題では、局所最適解 = 大域最適解

これにより、効率的なアルゴリズムで確実に最適解を発見できます。

### 線形計画法（Linear Programming）

$$ \begin{aligned} \min_{\mathbf{x}} \quad & \mathbf{c}^T \mathbf{x} \\\ \text{subject to} \quad & \mathbf{A} \mathbf{x} \leq \mathbf{b} \\\ & \mathbf{x} \geq 0 \end{aligned} $$
    
    
    import numpy as np
    from scipy.optimize import linprog
    import matplotlib.pyplot as plt
    
    # 線形計画問題
    # 目的関数：最小化 -x - 2y （= 最大化 x + 2y）
    c = [-1, -2]
    
    # 不等式制約：A_ub * x <= b_ub
    # 制約1: x + y <= 4
    # 制約2: 2x + y <= 5
    # 制約3: x >= 0, y >= 0（bounds で指定）
    A_ub = np.array([[1, 1],
                     [2, 1]])
    b_ub = np.array([4, 5])
    
    # 変数の範囲
    bounds = [(0, None), (0, None)]
    
    # 最適化
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # 可視化
    x = np.linspace(0, 5, 100)
    
    plt.figure(figsize=(10, 8))
    
    # 制約の可視化
    y1 = 4 - x
    y2 = 5 - 2*x
    
    plt.plot(x, y1, 'r-', linewidth=2, label='$x + y \leq 4$')
    plt.plot(x, y2, 'b-', linewidth=2, label='$2x + y \leq 5$')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    # 実行可能領域
    x_fill = np.linspace(0, 2.5, 100)
    y_upper = np.minimum(4 - x_fill, 5 - 2*x_fill)
    y_upper = np.maximum(y_upper, 0)
    
    plt.fill_between(x_fill, 0, y_upper, alpha=0.3, color='green',
                     label='実行可能領域')
    
    # 目的関数の等高線
    X, Y = np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100))
    Z = X + 2*Y
    plt.contour(X, Y, Z, levels=10, alpha=0.3, cmap='viridis')
    
    # 最適解
    plt.plot(result.x[0], result.x[1], 'r*', markersize=20, label='最適解')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('線形計画法', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()
    
    print("=== 線形計画法の結果 ===")
    print(f"最適解: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"目的関数値（最大化）: {-result.fun:.4f}")
    print(f"最適化成功: {result.success}")
    

### 二次計画法（Quadratic Programming）

$$ \begin{aligned} \min_{\mathbf{x}} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{Q} \mathbf{x} + \mathbf{c}^T \mathbf{x} \\\ \text{subject to} \quad & \mathbf{A} \mathbf{x} \leq \mathbf{b} \end{aligned} $$

### CVXPYによる凸最適化
    
    
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ポートフォリオ最適化問題
    # 目的：リスクを最小化しつつ、期待リターンを目標値以上にする
    
    # 資産の期待リターンと共分散行列（サンプルデータ）
    np.random.seed(42)
    n_assets = 5
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T / 100  # 正定値行列
    
    # 変数：資産配分
    w = cp.Variable(n_assets)
    
    # パラメータ：目標リターン
    target_return = 0.10
    
    # 目的関数：リスク（分散）の最小化
    risk = cp.quad_form(w, cov_matrix)
    
    # 制約
    constraints = [
        cp.sum(w) == 1,           # 資産配分の合計 = 1
        w >= 0,                    # ショート禁止
        expected_returns @ w >= target_return  # 目標リターン達成
    ]
    
    # 問題の定義と解決
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()
    
    # 効率的フロンティアの計算
    target_returns = np.linspace(0.06, 0.14, 20)
    risks = []
    portfolios = []
    
    for target in target_returns:
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            expected_returns @ w >= target
        ]
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
    
        if problem.status == 'optimal':
            risks.append(np.sqrt(problem.value))
            portfolios.append(w.value)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 効率的フロンティア
    axes[0].plot(risks, target_returns, 'b-', linewidth=2, label='効率的フロンティア')
    axes[0].plot(np.sqrt(problem.value), target_return, 'r*',
                 markersize=15, label=f'選択ポートフォリオ (リターン={target_return})')
    axes[0].set_xlabel('リスク（標準偏差）', fontsize=12)
    axes[0].set_ylabel('期待リターン', fontsize=12)
    axes[0].set_title('効率的フロンティア', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 資産配分
    if problem.status == 'optimal':
        axes[1].bar(range(n_assets), w.value, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('資産', fontsize=12)
        axes[1].set_ylabel('配分比率', fontsize=12)
        axes[1].set_title('最適資産配分', fontsize=14)
        axes[1].set_xticks(range(n_assets))
        axes[1].set_xticklabels([f'資産{i+1}' for i in range(n_assets)])
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("=== ポートフォリオ最適化の結果 ===")
    print(f"目標リターン: {target_return:.2%}")
    print(f"達成リターン: {(expected_returns @ w.value):.2%}")
    print(f"リスク（標準偏差）: {np.sqrt(problem.value):.2%}")
    print(f"\n最適資産配分:")
    for i, weight in enumerate(w.value):
        print(f"  資産{i+1}: {weight:.2%}")
    

> **注意** ：CVXPYを使うには、`pip install cvxpy` でインストールが必要です。

* * *

## 3.5 実践：機械学習への応用

### ロジスティック回帰の最適化

ロジスティック回帰は凸最適化問題として解けます。

$$ \min_{\mathbf{w}} \sum_{i=1}^n \log(1 + \exp(-y_i \mathbf{w}^T \mathbf{x}_i)) + \lambda \|\mathbf{w}\|^2 $$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # データ生成
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                              n_redundant=0, n_clusters_per_class=1,
                              random_state=42)
    y = 2 * y - 1  # {0, 1} -> {-1, 1}
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # シグモイド関数
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # 損失関数（交差エントロピー + L2正則化）
    def logistic_loss(w, X, y, lambda_reg=0.01):
        z = X @ w
        loss = np.mean(np.log(1 + np.exp(-y * z)))
        reg = lambda_reg * np.sum(w**2)
        return loss + reg
    
    # 勾配
    def logistic_gradient(w, X, y, lambda_reg=0.01):
        z = X @ w
        grad = -X.T @ (y * sigmoid(-y * z)) / len(y)
        grad += 2 * lambda_reg * w
        return grad
    
    # 勾配降下法での最適化
    def train_logistic_regression(X, y, lr=0.1, n_iterations=1000, lambda_reg=0.01):
        n_features = X.shape[1]
        w = np.zeros(n_features)
    
        losses = []
    
        for i in range(n_iterations):
            grad = logistic_gradient(w, X, y, lambda_reg)
            w -= lr * grad
    
            if i % 10 == 0:
                loss = logistic_loss(w, X, y, lambda_reg)
                losses.append(loss)
    
        return w, losses
    
    # 学習
    w_optimal, losses = train_logistic_regression(
        X_train_scaled, y_train, lr=0.5, n_iterations=1000, lambda_reg=0.01
    )
    
    # 予測
    def predict(X, w):
        z = X @ w
        return np.sign(z)
    
    y_pred_train = predict(X_train_scaled, w_optimal)
    y_pred_test = predict(X_test_scaled, w_optimal)
    
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 学習曲線
    axes[0].plot(losses, linewidth=2)
    axes[0].set_xlabel('イテレーション（×10）', fontsize=12)
    axes[0].set_ylabel('損失', fontsize=12)
    axes[0].set_title('学習曲線', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 決定境界
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = predict(np.c_[xx.ravel(), yy.ravel()], w_optimal)
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm', levels=1)
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                    c=y_train, cmap='coolwarm', edgecolors='k', s=50, alpha=0.7)
    axes[1].set_xlabel('特徴量1', fontsize=12)
    axes[1].set_ylabel('特徴量2', fontsize=12)
    axes[1].set_title(f'決定境界（精度: {train_acc:.2%}）', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== ロジスティック回帰の結果 ===")
    print(f"訓練精度: {train_acc:.2%}")
    print(f"テスト精度: {test_acc:.2%}")
    print(f"最適パラメータ: {w_optimal}")
    

### 正則化項の効果

正則化は過学習を防ぎます。

  * **L1正則化（Lasso）** ：$\lambda \|\mathbf{w}\|_1$ → スパース解
  * **L2正則化（Ridge）** ：$\lambda \|\mathbf{w}\|_2^2$ → 重み減衰

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # データ生成
    X, y = make_regression(n_samples=100, n_features=50, n_informative=10,
                           noise=10, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 異なる正則化パラメータでの学習
    alphas = np.logspace(-3, 3, 50)
    
    ridge_train_scores = []
    ridge_test_scores = []
    lasso_train_scores = []
    lasso_test_scores = []
    
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_train_scores.append(ridge.score(X_train_scaled, y_train))
        ridge_test_scores.append(ridge.score(X_test_scaled, y_test))
    
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        lasso_train_scores.append(lasso.score(X_train_scaled, y_train))
        lasso_test_scores.append(lasso.score(X_test_scaled, y_test))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ridge
    axes[0].semilogx(alphas, ridge_train_scores, 'b-', linewidth=2, label='訓練')
    axes[0].semilogx(alphas, ridge_test_scores, 'r-', linewidth=2, label='テスト')
    axes[0].set_xlabel('正則化パラメータ α', fontsize=12)
    axes[0].set_ylabel('R² スコア', fontsize=12)
    axes[0].set_title('Ridge回帰（L2正則化）', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Lasso
    axes[1].semilogx(alphas, lasso_train_scores, 'b-', linewidth=2, label='訓練')
    axes[1].semilogx(alphas, lasso_test_scores, 'r-', linewidth=2, label='テスト')
    axes[1].set_xlabel('正則化パラメータ α', fontsize=12)
    axes[1].set_ylabel('R² スコア', fontsize=12)
    axes[1].set_title('Lasso回帰（L1正則化）', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最適なαでの係数比較
    ridge_best = Ridge(alpha=1.0)
    ridge_best.fit(X_train_scaled, y_train)
    
    lasso_best = Lasso(alpha=0.1, max_iter=10000)
    lasso_best.fit(X_train_scaled, y_train)
    
    print("=== 正則化の効果 ===")
    print(f"Ridge - 非ゼロ係数: {np.sum(np.abs(ridge_best.coef_) > 0.01)}/{len(ridge_best.coef_)}")
    print(f"Lasso - 非ゼロ係数: {np.sum(np.abs(lasso_best.coef_) > 0.01)}/{len(lasso_best.coef_)}")
    print(f"\nRidge最良スコア: {ridge_best.score(X_test_scaled, y_test):.3f}")
    print(f"Lasso最良スコア: {lasso_best.score(X_test_scaled, y_test):.3f}")
    

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **最適化の基礎**

     * 凸性の重要性：凸最適化は大域最適解が保証
     * 勾配とヘシアンによる最適性の判定
  2. **勾配降下法**

     * 学習率の選択が収束速度と安定性を決定
     * SGD、Momentum、Adamなどの発展手法
     * 実装と収束性の理解
  3. **制約付き最適化**

     * ラグランジュ乗数法による等式制約の扱い
     * KKT条件による不等式制約の理論
     * SVMへの応用
  4. **凸最適化**

     * 線形計画法と二次計画法
     * CVXPYによる実装
     * ポートフォリオ最適化などの応用
  5. **機械学習への応用**

     * ロジスティック回帰の最適化
     * 正則化による過学習防止
     * 実データでの実装

### 最適化アルゴリズムの選択指針

問題の性質 | 推奨手法 | 理由  
---|---|---  
凸関数、小規模 | 勾配降下法 | 確実に最適解発見  
凸関数、大規模 | SGD、Adam | 計算効率  
非凸、深層学習 | Adam、RMSprop | 局所最適解回避  
等式制約あり | ラグランジュ乗数法 | 制約充足保証  
不等式制約あり | KKT条件、SLSQP | 実行可能解  
線形・凸二次 | 専用ソルバー（CVXPY） | 高速・安定  
  
### 次の章へ

第4章では、**確率・統計の基礎** を学びます：

  * 確率分布と期待値
  * 最尤推定とベイズ推定
  * 仮説検定と信頼区間
  * 情報理論の基礎

* * *

## 演習問題

### 問題1（難易度：easy）

関数 $f(x) = x^2 + 4x + 4$ を最小化する $x$ を、勾配を用いて求めてください（解析的に）。

解答例

**解答** ：

勾配を0にする点を求めます。

$$ \nabla f(x) = \frac{df}{dx} = 2x + 4 $$

最適性条件：

$$ 2x + 4 = 0 \Rightarrow x^* = -2 $$

2次条件（十分条件）：

$$ \frac{d^2 f}{dx^2} = 2 > 0 $$

よって、$x^* = -2$ で最小値 $f(x^*) = 0$ を取ります。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 関数と勾配
    def f(x):
        return x**2 + 4*x + 4
    
    def grad_f(x):
        return 2*x + 4
    
    # 可視化
    x = np.linspace(-5, 2, 100)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='$f(x) = x^2 + 4x + 4$')
    plt.plot(-2, 0, 'r*', markersize=20, label='最適解: $x^* = -2$')
    plt.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    plt.axvline(x=-2, color='r', linewidth=0.5, linestyle='--')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('最適化問題の解析解', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"最適解: x* = -2")
    print(f"最小値: f(x*) = {f(-2)}")
    print(f"勾配: f'(x*) = {grad_f(-2)}")
    

### 問題2（難易度：medium）

学習率 $\alpha = 0.1$ で、初期値 $x_0 = 5$ から勾配降下法を用いて $f(x) = x^2$ を最小化してください。5回のイテレーションでの $x$ の値を報告してください。

解答例
    
    
    import numpy as np
    
    # 目的関数と勾配
    def f(x):
        return x**2
    
    def grad_f(x):
        return 2*x
    
    # 勾配降下法
    x = 5.0
    alpha = 0.1
    n_iterations = 5
    
    print("=== 勾配降下法の軌跡 ===")
    print(f"イテレーション 0: x = {x:.6f}, f(x) = {f(x):.6f}")
    
    for i in range(1, n_iterations + 1):
        grad = grad_f(x)
        x = x - alpha * grad
        print(f"イテレーション {i}: x = {x:.6f}, f(x) = {f(x):.6f}, grad = {grad:.6f}")
    
    print(f"\n最終値: x = {x:.6f}")
    print(f"真の最適解: x* = 0")
    

**出力** ：
    
    
    === 勾配降下法の軌跡 ===
    イテレーション 0: x = 5.000000, f(x) = 25.000000
    イテレーション 1: x = 4.000000, f(x) = 16.000000, grad = 10.000000
    イテレーション 2: x = 3.200000, f(x) = 10.240000, grad = 8.000000
    イテレーション 3: x = 2.560000, f(x) = 6.553600, grad = 6.400000
    イテレーション 4: x = 2.048000, f(x) = 4.194304, grad = 5.120000
    イテレーション 5: x = 1.638400, f(x) = 2.684355, grad = 4.096000
    
    最終値: x = 1.638400
    真の最適解: x* = 0
    

5回のイテレーションでは完全には収束していませんが、最適解に近づいています。

### 問題3（難易度：medium）

等式制約 $x + y = 1$ の下で、$f(x, y) = x^2 + y^2$ を最小化する問題を、ラグランジュ乗数法で解いてください。

解答例

**解答** ：

ラグランジュ関数：

$$ \mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 1) $$

最適性条件：

$$ \begin{aligned} \frac{\partial \mathcal{L}}{\partial x} &= 2x + \lambda = 0 \\\ \frac{\partial \mathcal{L}}{\partial y} &= 2y + \lambda = 0 \\\ \frac{\partial \mathcal{L}}{\partial \lambda} &= x + y - 1 = 0 \end{aligned} $$

1式と2式から $x = y$、3式に代入して $2x = 1 \Rightarrow x = y = 0.5$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    # 目的関数
    def objective(vars):
        x, y = vars
        return x**2 + y**2
    
    # 等式制約
    def constraint(vars):
        x, y = vars
        return x + y - 1
    
    # 制約の定義
    constraints = {'type': 'eq', 'fun': constraint}
    
    # 初期点
    x0 = [0.0, 0.0]
    
    # 最適化
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    print("=== ラグランジュ乗数法の解 ===")
    print(f"最適解: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"目的関数値: f(x*, y*) = {result.fun:.4f}")
    print(f"制約充足: x + y = {result.x[0] + result.x[1]:.6f}")
    
    # 可視化
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='$f(x, y)$')
    
    # 制約線
    x_line = np.linspace(-0.5, 1.5, 100)
    y_line = 1 - x_line
    plt.plot(x_line, y_line, 'r-', linewidth=3, label='制約: $x + y = 1$')
    
    # 最適点
    plt.plot(result.x[0], result.x[1], 'r*', markersize=20, label='最適解')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('等式制約付き最適化', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    

### 問題4（難易度：hard）

Momentum法（$\beta = 0.9$）を用いて、Rosenbrock関数 $f(x, y) = (1-x)^2 + 100(y-x^2)^2$ を初期点 $(-1, 1)$ から最適化してください。学習率 $\alpha = 0.001$ で100回イテレーションし、軌跡を可視化してください。

解答例
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Rosenbrock関数
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # Momentum法
    def momentum(start, grad_func, lr=0.001, beta=0.9, n_iterations=100):
        pos = start.copy()
        velocity = np.zeros_like(pos)
        trajectory = [pos.copy()]
    
        for i in range(n_iterations):
            grad = grad_func(pos[0], pos[1])
            velocity = beta * velocity - lr * grad
            pos += velocity
            trajectory.append(pos.copy())
    
        return np.array(trajectory)
    
    # 最適化
    start = np.array([-1.0, 1.0])
    trajectory = momentum(start, rosenbrock_grad, lr=0.001, beta=0.9, n_iterations=100)
    
    # 可視化
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize=(12, 9))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.colorbar(label='$f(x, y)$')
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=4,
             linewidth=1.5, alpha=0.7, label='Momentum軌跡')
    plt.plot(start[0], start[1], 'bo', markersize=10, label='開始点')
    plt.plot(1, 1, 'g*', markersize=20, label='最適解')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=10, label='終了点')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Momentum法によるRosenbrock関数の最適化', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Momentum法の結果 ===")
    print(f"初期点: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"最終点: ({trajectory[-1][0]:.4f}, {trajectory[-1][1]:.4f})")
    print(f"最適解: (1.0000, 1.0000)")
    print(f"最終関数値: {rosenbrock(trajectory[-1][0], trajectory[-1][1]):.6f}")
    

### 問題5（難易度：hard）

L1正則化とL2正則化の違いを説明し、どちらがスパース解（多くの係数が0）を生成しやすいか、理由とともに述べてください。

解答例

**解答** ：

**L2正則化（Ridge）** ：

  * ペナルティ項: $\lambda \sum_i w_i^2$
  * 特徴: 係数を小さくするが、0にはしない
  * 勾配: $\nabla (\lambda w^2) = 2\lambda w$ （連続的に0に近づく）

**L1正則化（Lasso）** ：

  * ペナルティ項: $\lambda \sum_i |w_i|$
  * 特徴: 係数を正確に0にする（スパース解）
  * 勾配: $\nabla (\lambda |w|) = \lambda \cdot \text{sign}(w)$ （0付近で非連続）

**スパース性の理由** ：

  1. **幾何学的解釈** ：L1制約の実行可能領域は角を持つ（例：菱形）。目的関数の等高線が角と交わりやすく、そこでは一部の係数が0になる。
  2. **勾配の性質** ：L1の勾配は係数の大きさに依存せず一定（$\pm \lambda$）のため、小さい係数も大きい係数も同じペナルティを受け、小さい係数が0に押し込まれやすい。

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # L1とL2の制約領域の可視化
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # L1制約（|w1| + |w2| <= 1）
    w1_l1 = np.cos(theta) * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    w2_l1 = np.sin(theta) * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    
    # L2制約（w1^2 + w2^2 <= 1）
    w1_l2 = np.cos(theta)
    w2_l2 = np.sin(theta)
    
    # 目的関数の等高線（仮想的な例）
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = (W1 - 1.5)**2 + (W2 - 1.0)**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # L1正則化
    axes[0].contour(W1, W2, Z, levels=15, alpha=0.6, cmap='viridis')
    axes[0].fill(w1_l1, w2_l1, alpha=0.3, color='red', label='L1制約領域')
    axes[0].plot(0, 0, 'r*', markersize=20, label='スパース解（w1=0）')
    axes[0].set_xlabel('$w_1$', fontsize=12)
    axes[0].set_ylabel('$w_2$', fontsize=12)
    axes[0].set_title('L1正則化（Lasso）', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # L2正則化
    axes[1].contour(W1, W2, Z, levels=15, alpha=0.6, cmap='viridis')
    axes[1].fill(w1_l2, w2_l2, alpha=0.3, color='blue', label='L2制約領域')
    circle_intersect_x = 0.8
    circle_intersect_y = 0.6
    axes[1].plot(circle_intersect_x, circle_intersect_y, 'b*',
                 markersize=20, label='非スパース解')
    axes[1].set_xlabel('$w_1$', fontsize=12)
    axes[1].set_ylabel('$w_2$', fontsize=12)
    axes[1].set_title('L2正則化（Ridge）', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("=== L1 vs L2正則化 ===")
    print("L1（Lasso）: スパース解を生成（多くの係数が0）")
    print("L2（Ridge）: 小さいが非ゼロの係数を生成")
    print("\n用途:")
    print("- L1: 特徴選択が必要な場合")
    print("- L2: すべての特徴を使いつつ過学習を防ぐ場合")
    

* * *

## 参考文献

  1. Boyd, S., & Vandenberghe, L. (2004). _Convex Optimization_. Cambridge University Press.
  2. Nocedal, J., & Wright, S. (2006). _Numerical Optimization_ (2nd ed.). Springer.
  3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
  4. Ruder, S. (2016). "An overview of gradient descent optimization algorithms". arXiv:1609.04747.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
