---
title: 第3章：活性化関数と最適化
chapter_title: 第3章：活性化関数と最適化
subtitle: 深層学習の性能を決定する重要要素
reading_time: 25-30分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 様々な活性化関数（Sigmoid、ReLU、Leaky ReLU、ELU、Swish）の特徴を理解する
  * ✅ 勾配消失問題（Vanishing Gradient Problem）とその対策を説明できる
  * ✅ 高度な最適化アルゴリズム（Momentum、AdaGrad、RMSprop、Adam）を実装できる
  * ✅ 学習率スケジューリングの重要性を理解する
  * ✅ 重みの初期化戦略（Xavier、He初期化）を適用できる

* * *

## 3.1 活性化関数（Activation Functions）

### 活性化関数の役割

**活性化関数** は、ニューラルネットワークに**非線形性** を導入します。活性化関数がなければ、何層重ねても線形変換にしかならず、複雑なパターンを学習できません。

> 「活性化関数は、ニューラルネットワークの"表現力"を決定する重要な要素です。」
    
    
    ```mermaid
    graph LR
        A[線形変換z = Wx + b] --> B[活性化関数a = f(z)]
        B --> C[非線形出力]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

* * *

### 3.1.1 Sigmoid関数

**数式** ：

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

**導関数** ：

$$ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $$
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def sigmoid(x):
        """Sigmoid関数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        """Sigmoidの導関数"""
        s = sigmoid(x)
        return s * (1 - s)
    
    # 可視化
    x = np.linspace(-10, 10, 200)
    y = sigmoid(x)
    dy = sigmoid_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, linewidth=2, label='σ(x)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('σ(x)', fontsize=12)
    plt.title('Sigmoid関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy, linewidth=2, color='red', label="σ'(x)")
    plt.xlabel('x', fontsize=12)
    plt.ylabel("σ'(x)", fontsize=12)
    plt.title('Sigmoidの導関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

**特徴** ：

  * ✅ 出力範囲: (0, 1)
  * ✅ 滑らかで微分可能
  * ❌ **勾配消失問題** : $|x|$が大きいとき、導関数が0に近づく
  * ❌ 出力が0中心でない（学習の収束が遅い）

* * *

### 3.1.2 tanh関数（双曲線正接）

**数式** ：

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

**導関数** ：

$$ \tanh'(x) = 1 - \tanh^2(x) $$
    
    
    def tanh(x):
        """tanh関数"""
        return np.tanh(x)
    
    def tanh_derivative(x):
        """tanhの導関数"""
        return 1 - np.tanh(x) ** 2
    
    # 可視化
    y_tanh = tanh(x)
    dy_tanh = tanh_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_tanh, linewidth=2, color='green')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('tanh(x)', fontsize=12)
    plt.title('tanh関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy_tanh, linewidth=2, color='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("tanh'(x)", fontsize=12)
    plt.title('tanhの導関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**特徴** ：

  * ✅ 出力範囲: (-1, 1)
  * ✅ 0中心の出力（Sigmoidより優れている）
  * ❌ 勾配消失問題は残る

* * *

### 3.1.3 ReLU（Rectified Linear Unit）

**数式** ：

$$ \text{ReLU}(x) = \max(0, x) $$

**導関数** ：

$$ \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\\ 0 & \text{if } x \leq 0 \end{cases} $$
    
    
    def relu(x):
        """ReLU関数"""
        return np.maximum(0, x)
    
    def relu_derivative(x):
        """ReLUの導関数"""
        return (x > 0).astype(float)
    
    # 可視化
    y_relu = relu(x)
    dy_relu = relu_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, linewidth=2, color='purple')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('ReLU(x)', fontsize=12)
    plt.title('ReLU関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy_relu, linewidth=2, color='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("ReLU'(x)", fontsize=12)
    plt.title('ReLUの導関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**特徴** ：

  * ✅ 計算が非常に高速（max演算のみ）
  * ✅ 勾配消失問題が大幅に軽減
  * ✅ **現在最も広く使われる活性化関数**
  * ❌ **Dying ReLU問題** : 負の入力でニューロンが死ぬ（勾配0）

* * *

### 3.1.4 Leaky ReLU

**数式** ：

$$ \text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\\ \alpha x & \text{if } x \leq 0 \end{cases} $$

通常、$\alpha = 0.01$
    
    
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU関数"""
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(x, alpha=0.01):
        """Leaky ReLUの導関数"""
        return np.where(x > 0, 1.0, alpha)
    
    # 可視化
    y_leaky = leaky_relu(x)
    dy_leaky = leaky_relu_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_leaky, linewidth=2, color='orange')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Leaky ReLU(x)', fontsize=12)
    plt.title('Leaky ReLU関数 (α=0.01)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy_leaky, linewidth=2, color='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("Leaky ReLU'(x)", fontsize=12)
    plt.title('Leaky ReLUの導関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**特徴** ：

  * ✅ Dying ReLU問題を解決
  * ✅ 負の入力でも小さな勾配が流れる

* * *

### 3.1.5 ELU（Exponential Linear Unit）

**数式** ：

$$ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases} $$
    
    
    def elu(x, alpha=1.0):
        """ELU関数"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(x, alpha=1.0):
        """ELUの導関数"""
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    # 可視化
    y_elu = elu(x)
    dy_elu = elu_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_elu, linewidth=2, color='brown')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('ELU(x)', fontsize=12)
    plt.title('ELU関数 (α=1.0)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy_elu, linewidth=2, color='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel("ELU'(x)", fontsize=12)
    plt.title('ELUの導関数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**特徴** ：

  * ✅ 滑らかで微分可能
  * ✅ 平均が0に近い出力
  * ❌ 指数関数の計算コストが高い

* * *

### 活性化関数の比較

関数 | 出力範囲 | 勾配消失 | 計算速度 | 推奨用途  
---|---|---|---|---  
**Sigmoid** | (0, 1) | ❌ あり | ⚡ 遅い | 出力層（二値分類）  
**tanh** | (-1, 1) | ❌ あり | ⚡ 遅い | RNN（過去）  
**ReLU** | [0, ∞) | ✅ なし | ⚡⚡⚡ 非常に速い | 隠れ層（デフォルト）  
**Leaky ReLU** | (-∞, ∞) | ✅ なし | ⚡⚡⚡ 非常に速い | ReLUで失敗時  
**ELU** | [-α, ∞) | ✅ なし | ⚡⚡ 中程度 | 高精度が必要な場合  
  
* * *

## 3.2 勾配消失問題（Vanishing Gradient Problem）

### 問題の本質

深いネットワークでは、逆伝播時に勾配が**指数関数的に小さくなる** 現象が発生します。

連鎖律により：

$$ \frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial w^{(10)}} \cdot \frac{\partial w^{(10)}}{\partial w^{(9)}} \cdot \ldots \cdot \frac{\partial w^{(2)}}{\partial w^{(1)}} $$

各層で $|\frac{\partial w^{(l)}}{\partial w^{(l-1)}}| < 1$ の場合、勾配が消失します。
    
    
    def demonstrate_vanishing_gradient():
        """勾配消失問題のデモンストレーション"""
    
        # Sigmoidネットワーク（10層）
        def forward_sigmoid_deep(x, n_layers=10):
            a = x
            activations = [a]
    
            for _ in range(n_layers):
                z = a * 0.5  # 簡略化した重み
                a = sigmoid(z)
                activations.append(a)
    
            return activations
    
        # 勾配の計算
        x = np.array([1.0])
        activations = forward_sigmoid_deep(x, n_layers=10)
    
        # 各層の勾配を計算
        gradients = []
        grad = 1.0
    
        for i in range(len(activations) - 1, 0, -1):
            a = activations[i]
            grad = grad * (a * (1 - a)) * 0.5  # 連鎖律
            gradients.append(grad)
    
        gradients = gradients[::-1]
    
        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(gradients) + 1), gradients, marker='o',
                 linewidth=2, markersize=8, label='勾配の大きさ')
        plt.xlabel('層の深さ', fontsize=12)
        plt.ylabel('勾配', fontsize=12)
        plt.title('勾配消失問題の可視化（Sigmoid、10層）', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
        print("=== 各層の勾配 ===")
        for i, grad in enumerate(gradients, 1):
            print(f"第{i}層: {grad:.10f}")
    
    demonstrate_vanishing_gradient()
    

### 対策

  1. **ReLUを使う** : 勾配が1（$x > 0$の場合）
  2. **Batch Normalization** : 各層の入力を正規化
  3. **Residual Connection** : 勾配をショートカット
  4. **適切な初期化** : Xavier、He初期化

* * *

## 3.3 最適化アルゴリズム（Optimization Algorithms）

### 3.3.1 SGD（Stochastic Gradient Descent）

**更新式** ：

$$ w \leftarrow w - \eta \frac{\partial L}{\partial w} $$
    
    
    class SGD:
        """確率的勾配降下法"""
    
        def __init__(self, learning_rate=0.01):
            self.learning_rate = learning_rate
    
        def update(self, params, grads):
            """
            パラメータの更新
    
            Args:
                params: パラメータの辞書 {'W1': ..., 'b1': ...}
                grads: 勾配の辞書 {'W1': ..., 'b1': ...}
            """
            for key in params.keys():
                params[key] -= self.learning_rate * grads[key]
    

* * *

### 3.3.2 Momentum

**更新式** ：

$$ \begin{align} v &\leftarrow \beta v - \eta \frac{\partial L}{\partial w} \\\ w &\leftarrow w + v \end{align} $$

通常、$\beta = 0.9$
    
    
    class Momentum:
        """Momentum最適化"""
    
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocity = None
    
        def update(self, params, grads):
            if self.velocity is None:
                self.velocity = {}
                for key, val in params.items():
                    self.velocity[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
                params[key] += self.velocity[key]
    

**特徴** ：

  * ✅ 過去の勾配を考慮
  * ✅ 振動を抑制し、収束を加速

* * *

### 3.3.3 AdaGrad（Adaptive Gradient）

**更新式** ：

$$ \begin{align} h &\leftarrow h + \left(\frac{\partial L}{\partial w}\right)^2 \\\ w &\leftarrow w - \frac{\eta}{\sqrt{h} + \epsilon} \frac{\partial L}{\partial w} \end{align} $$
    
    
    class AdaGrad:
        """AdaGrad最適化"""
    
        def __init__(self, learning_rate=0.01):
            self.learning_rate = learning_rate
            self.h = None
            self.epsilon = 1e-8
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.h[key] += grads[key] ** 2
                params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)
    

**特徴** ：

  * ✅ パラメータごとに学習率を調整
  * ❌ 学習率が徐々に小さくなりすぎる

* * *

### 3.3.4 RMSprop

**更新式** ：

$$ \begin{align} h &\leftarrow \beta h + (1 - \beta) \left(\frac{\partial L}{\partial w}\right)^2 \\\ w &\leftarrow w - \frac{\eta}{\sqrt{h} + \epsilon} \frac{\partial L}{\partial w} \end{align} $$
    
    
    class RMSprop:
        """RMSprop最適化"""
    
        def __init__(self, learning_rate=0.01, decay_rate=0.99):
            self.learning_rate = learning_rate
            self.decay_rate = decay_rate
            self.h = None
            self.epsilon = 1e-8
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * grads[key] ** 2
                params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)
    

**特徴** ：

  * ✅ AdaGradの改良版
  * ✅ 指数移動平均で学習率の減衰を緩和

* * *

### 3.3.5 Adam（Adaptive Moment Estimation）

**更新式** ：

$$ \begin{align} m &\leftarrow \beta_1 m + (1 - \beta_1) \frac{\partial L}{\partial w} \\\ v &\leftarrow \beta_2 v + (1 - \beta_2) \left(\frac{\partial L}{\partial w}\right)^2 \\\ \hat{m} &\leftarrow \frac{m}{1 - \beta_1^t} \\\ \hat{v} &\leftarrow \frac{v}{1 - \beta_2^t} \\\ w &\leftarrow w - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m} \end{align} $$
    
    
    class Adam:
        """Adam最適化（最も推奨）"""
    
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.m = None
            self.v = None
            self.t = 0
            self.epsilon = 1e-8
    
        def update(self, params, grads):
            if self.m is None:
                self.m = {}
                self.v = {}
                for key, val in params.items():
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)
    
            self.t += 1
    
            for key in params.keys():
                # 1次モーメント（平均）
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
    
                # 2次モーメント（分散）
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
    
                # バイアス補正
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
    
                # パラメータ更新
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    

**特徴** ：

  * ✅ MomentumとRMSpropの長所を組み合わせ
  * ✅ **現在最も広く使われる最適化アルゴリズム**
  * ✅ ハイパーパラメータのチューニングがほぼ不要

* * *

### 最適化アルゴリズムの比較
    
    
    def compare_optimizers():
        """最適化アルゴリズムの比較"""
    
        # テスト関数: f(x, y) = x^2 + 10*y^2（楕円）
        def f(x, y):
            return x ** 2 + 10 * y ** 2
    
        def grad_f(x, y):
            return np.array([2*x, 20*y])
    
        # 初期値
        init_pos = (-7.0, 2.0)
        learning_rate = 0.1
        iterations = 30
    
        # 各最適化アルゴリズムで最適化
        optimizers = {
            'SGD': SGD(learning_rate=learning_rate),
            'Momentum': Momentum(learning_rate=learning_rate),
            'AdaGrad': AdaGrad(learning_rate=learning_rate),
            'RMSprop': RMSprop(learning_rate=learning_rate),
            'Adam': Adam(learning_rate=learning_rate)
        }
    
        trajectories = {}
    
        for name, optimizer in optimizers.items():
            pos = np.array(init_pos)
            params = {'pos': pos}
            trajectory = [pos.copy()]
    
            for _ in range(iterations):
                grads = {'pos': grad_f(pos[0], pos[1])}
                optimizer.update(params, grads)
                pos = params['pos']
                trajectory.append(pos.copy())
    
            trajectories[name] = np.array(trajectory)
    
        # プロット
        plt.figure(figsize=(12, 10))
    
        # 等高線の描画
        x = np.linspace(-8, 2, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
    
        plt.contour(X, Y, Z, levels=20, alpha=0.3)
    
        # 各最適化アルゴリズムの軌跡
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for (name, trajectory), color in zip(trajectories.items(), colors):
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o',
                     label=name, color=color, linewidth=2, markersize=4)
    
        plt.plot(0, 0, 'r*', markersize=20, label='最適解')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('最適化アルゴリズムの比較', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    compare_optimizers()
    

* * *

## 3.4 重みの初期化

### なぜ初期化が重要か

重みの初期値が不適切だと：

  * ❌ 勾配消失・勾配爆発
  * ❌ 学習が進まない
  * ❌ 局所最適解に陥る

### 3.4.1 Xavier初期化

**数式** （Sigmoid、tanh用）：

$$ W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right) $$
    
    
    def xavier_init(n_in, n_out):
        """Xavier初期化"""
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
    
    # 例
    W = xavier_init(100, 50)
    print(f"Xavier初期化: 平均={W.mean():.4f}, 標準偏差={W.std():.4f}")
    

### 3.4.2 He初期化

**数式** （ReLU用）：

$$ W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right) $$
    
    
    def he_init(n_in, n_out):
        """He初期化（ReLU用）"""
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    
    # 例
    W = he_init(100, 50)
    print(f"He初期化: 平均={W.mean():.4f}, 標準偏差={W.std():.4f}")
    

### 初期化の比較

初期化手法 | 数式 | 推奨活性化関数  
---|---|---  
**ゼロ初期化** | $W = 0$ | ❌ 使用不可  
**ランダム初期化** | $W \sim \mathcal{N}(0, 0.01)$ | 基本的に非推奨  
**Xavier初期化** | $\sqrt{2/(n_{in}+n_{out})}$ | Sigmoid、tanh  
**He初期化** | $\sqrt{2/n_{in}}$ | ReLU、Leaky ReLU  
  
* * *

## 3.5 本章のまとめ

### 学んだこと

  1. **活性化関数**

     * ReLU: 現在のデフォルト
     * Leaky ReLU: Dying ReLU対策
     * Sigmoid/tanh: 勾配消失問題あり
  2. **勾配消失問題**

     * 深いネットワークでの課題
     * ReLU、Batch Norm、適切な初期化で対策
  3. **最適化アルゴリズム**

     * Adam: 最も推奨
     * Momentum: 収束を加速
     * SGD: 基本だが遅い
  4. **重みの初期化**

     * ReLU → He初期化
     * Sigmoid/tanh → Xavier初期化

### 推奨設定

要素 | 推奨  
---|---  
**活性化関数** | ReLU（隠れ層）  
**最適化** | Adam  
**初期化** | He初期化  
**学習率** | 0.001（Adam）
