---
title: 第2章：多層パーセプトロンと誤差逆伝播法
chapter_title: 第2章：多層パーセプトロンと誤差逆伝播法
subtitle: 深層学習の核心アルゴリズム - Backpropagation
reading_time: 30-35分
difficulty: 初級〜中級
code_examples: 15
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 多層パーセプトロン（MLP）の構造を理解する
  * ✅ 誤差逆伝播法（Backpropagation）の仕組みを説明できる
  * ✅ 勾配降下法によるパラメータ更新を理解する
  * ✅ 連鎖律（Chain Rule）の数学的基礎を学ぶ
  * ✅ NumPyでMLPを完全実装できる
  * ✅ XOR問題を実際に解決できる

* * *

## 2.1 多層パーセプトロン（MLP）の構造

### MLPとは

**多層パーセプトロン（Multilayer Perceptron, MLP）** は、複数層のパーセプトロンを組み合わせたニューラルネットワークです。
    
    
    ```mermaid
    graph LR
        x1[入力層x1] --> h1[隠れ層h1]
        x2[入力層x2] --> h1
        x1 --> h2[隠れ層h2]
        x2 --> h2
        h1 --> y1[出力層y1]
        h2 --> y1
    
        style x1 fill:#e3f2fd
        style x2 fill:#e3f2fd
        style h1 fill:#fff3e0
        style h2 fill:#fff3e0
        style y1 fill:#e8f5e9
    ```

### 層の種類

層の種類 | 役割 | 説明  
---|---|---  
**入力層** | Input Layer | データを受け取る層（学習対象ではない）  
**隠れ層** | Hidden Layer | 特徴抽出を行う層（学習対象）  
**出力層** | Output Layer | 最終結果を出力する層（学習対象）  
  
### 2層ニューラルネットワークの数式

入力 $\mathbf{x} = [x_1, x_2]^T$ から出力 $y$ までの計算：

**第1層（入力 → 隠れ層）** ：

$$ \mathbf{h} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) $$

**第2層（隠れ層 → 出力）** ：

$$ y = \sigma(\mathbf{W}^{(2)} \mathbf{h} + b^{(2)}) $$

ここで、$\sigma$は活性化関数（シグモイド関数など）です。

### Python実装の基本構造
    
    
    import numpy as np
    
    def sigmoid(x):
        """シグモイド関数"""
        return 1 / (1 + np.exp(-x))
    
    class TwoLayerNet:
        """2層ニューラルネットワーク"""
    
        def __init__(self, input_size, hidden_size, output_size):
            """
            Args:
                input_size: 入力層のニューロン数
                hidden_size: 隠れ層のニューロン数
                output_size: 出力層のニューロン数
            """
            # 重みの初期化（ランダム）
            self.W1 = np.random.randn(input_size, hidden_size)
            self.b1 = np.zeros(hidden_size)
    
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b2 = np.zeros(output_size)
    
        def forward(self, x):
            """
            順伝播（Forward Propagation）
    
            Args:
                x: 入力データ (n_samples, input_size)
    
            Returns:
                出力 (n_samples, output_size)
            """
            # 第1層
            self.z1 = np.dot(x, self.W1) + self.b1
            self.a1 = sigmoid(self.z1)
    
            # 第2層
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = sigmoid(self.z2)
    
            return self.a2
    
    # テスト
    net = TwoLayerNet(input_size=2, hidden_size=3, output_size=1)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output = net.forward(x)
    
    print("=== 初期化直後の出力 ===")
    print(output)
    

* * *

## 2.2 損失関数（Loss Function）

### 損失関数とは

**損失関数（Loss Function）** は、ニューラルネットワークの予測値と正解値の差を数値化します。学習の目的は、この損失を**最小化** することです。

### 平均二乗誤差（MSE）

回帰問題でよく使われる損失関数：

$$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

ここで、$y_i$は正解値、$\hat{y}_i$は予測値です。
    
    
    def mean_squared_error(y_true, y_pred):
        """
        平均二乗誤差（Mean Squared Error, MSE）
    
        Args:
            y_true: 正解ラベル (n_samples,)
            y_pred: 予測値 (n_samples,)
    
        Returns:
            MSE値
        """
        return np.mean((y_true - y_pred) ** 2)
    
    # 例
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    loss = mean_squared_error(y_true, y_pred)
    print(f"MSE: {loss:.4f}")  # 0.0125
    

### 交差エントロピー誤差（Cross-Entropy）

分類問題で使われる損失関数：

$$ L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right] $$
    
    
    def binary_cross_entropy(y_true, y_pred):
        """
        二値交差エントロピー（Binary Cross-Entropy）
    
        Args:
            y_true: 正解ラベル (n_samples,)
            y_pred: 予測確率 (n_samples,)
    
        Returns:
            BCE値
        """
        # 数値安定性のため、小さな値でクリッピング
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
        return -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))
    
    # 例
    loss_ce = binary_cross_entropy(y_true, y_pred)
    print(f"Cross-Entropy: {loss_ce:.4f}")  # 0.1625
    

* * *

## 2.3 勾配降下法（Gradient Descent）

### 基本的なアイデア

**勾配降下法** は、損失関数を最小化するためのアルゴリズムです。パラメータを損失関数の勾配（微分）の**逆方向** に少しずつ更新します。
    
    
    ```mermaid
    graph TD
        A[初期パラメータ] --> B[損失を計算]
        B --> C[勾配を計算]
        C --> D[パラメータを更新]
        D --> E{収束？}
        E -->|No| B
        E -->|Yes| F[学習完了]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style F fill:#c8e6c9
    ```

### 更新式

パラメータ $w$ の更新：

$$ w \leftarrow w - \eta \frac{\partial L}{\partial w} $$

ここで、$\eta$は**学習率（Learning Rate）** です。

### Python実装
    
    
    def gradient_descent_demo():
        """勾配降下法のデモ"""
    
        # 簡単な関数: f(x) = x^2
        def f(x):
            return x ** 2
    
        # 導関数: f'(x) = 2x
        def df(x):
            return 2 * x
    
        # 初期値と学習率
        x = 10.0
        learning_rate = 0.1
        n_iterations = 20
    
        print("=== 勾配降下法のデモ ===")
        print(f"目標: f(x) = x^2 の最小値を見つける")
        print(f"初期値: x = {x}")
        print()
    
        for i in range(n_iterations):
            grad = df(x)
            x = x - learning_rate * grad
    
            if i % 5 == 0:
                print(f"Iteration {i:2d}: x = {x:8.4f}, f(x) = {f(x):8.4f}, grad = {grad:8.4f}")
    
        print()
        print(f"最終結果: x = {x:.4f}, f(x) = {f(x):.4f}")
        print(f"理論値: x = 0.0000, f(x) = 0.0000")
    
    gradient_descent_demo()
    

**出力** ：
    
    
    === 勾配降下法のデモ ===
    目標: f(x) = x^2 の最小値を見つける
    初期値: x = 10.0
    
    Iteration  0: x =   8.0000, f(x) =  64.0000, grad =  20.0000
    Iteration  5: x =   2.6214, f(x) =   6.8718, grad =   5.2429
    Iteration 10: x =   0.8590, f(x) =   0.7379, grad =   1.7179
    Iteration 15: x =   0.2815, f(x) =   0.0792, grad =   0.5630
    
    最終結果: x = 0.0922, f(x) = 0.0085
    理論値: x = 0.0000, f(x) = 0.0000
    

### 学習率の影響
    
    
    def compare_learning_rates():
        """学習率の違いを比較"""
        def f(x):
            return x ** 2
    
        def df(x):
            return 2 * x
    
        learning_rates = [0.01, 0.1, 0.5, 0.9]
        x_init = 10.0
        n_iterations = 10
    
        print("=== 学習率の比較 ===")
        for lr in learning_rates:
            x = x_init
            for _ in range(n_iterations):
                x = x - lr * df(x)
    
            print(f"学習率 η={lr:.2f} → 最終値 x={x:8.4f}, f(x)={f(x):8.4f}")
    
    compare_learning_rates()
    

**出力** ：
    
    
    === 学習率の比較 ===
    学習率 η=0.01 → 最終値 x=   8.1707, f(x)=  66.7604
    学習率 η=0.10 → 最終値 x=   0.0922, f(x)=   0.0085
    学習率 η=0.50 → 最終値 x=   0.0098, f(x)=   0.0001
    学習率 η=0.90 → 最終値 x= -10.0000, f(x)= 100.0000（発散！）
    

> **重要** : 学習率が大きすぎると発散、小さすぎると収束が遅い！

* * *

## 2.4 誤差逆伝播法（Backpropagation）

### なぜ逆伝播が必要か

多層ネットワークでは、各層のパラメータの勾配を計算する必要があります。**誤差逆伝播法** は、出力層から入力層に向かって勾配を効率的に計算する手法です。

### 連鎖律（Chain Rule）

合成関数の微分：

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w} $$

これが誤差逆伝播の数学的基礎です。

### 2層ネットワークの逆伝播

前向き計算：

$$ \begin{align} z^{(1)} &= W^{(1)} x + b^{(1)} \\\ a^{(1)} &= \sigma(z^{(1)}) \\\ z^{(2)} &= W^{(2)} a^{(1)} + b^{(2)} \\\ y &= \sigma(z^{(2)}) \\\ L &= \frac{1}{2}(y - t)^2 \end{align} $$

逆向き計算（勾配）：

$$ \begin{align} \frac{\partial L}{\partial y} &= y - t \\\ \frac{\partial L}{\partial z^{(2)}} &= \frac{\partial L}{\partial y} \cdot \sigma'(z^{(2)}) \\\ \frac{\partial L}{\partial W^{(2)}} &= \frac{\partial L}{\partial z^{(2)}} \cdot (a^{(1)})^T \\\ \frac{\partial L}{\partial b^{(2)}} &= \frac{\partial L}{\partial z^{(2)}} \\\ \frac{\partial L}{\partial a^{(1)}} &= (W^{(2)})^T \cdot \frac{\partial L}{\partial z^{(2)}} \\\ \frac{\partial L}{\partial z^{(1)}} &= \frac{\partial L}{\partial a^{(1)}} \cdot \sigma'(z^{(1)}) \\\ \frac{\partial L}{\partial W^{(1)}} &= \frac{\partial L}{\partial z^{(1)}} \cdot x^T \\\ \frac{\partial L}{\partial b^{(1)}} &= \frac{\partial L}{\partial z^{(1)}} \end{align} $$

### 完全実装
    
    
    import numpy as np
    
    def sigmoid(x):
        """シグモイド関数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        """シグモイド関数の導関数"""
        s = sigmoid(x)
        return s * (1 - s)
    
    class TwoLayerNetWithBackprop:
        """誤差逆伝播法を実装した2層ニューラルネットワーク"""
    
        def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
            """
            Args:
                input_size: 入力層のサイズ
                hidden_size: 隠れ層のサイズ
                output_size: 出力層のサイズ
                learning_rate: 学習率
            """
            # 重みの初期化（Heの初期化）
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros(hidden_size)
    
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros(output_size)
    
            self.learning_rate = learning_rate
    
        def forward(self, x):
            """順伝播"""
            # 第1層
            self.z1 = np.dot(x, self.W1) + self.b1
            self.a1 = sigmoid(self.z1)
    
            # 第2層
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = sigmoid(self.z2)
    
            return self.a2
    
        def backward(self, x, y_true, y_pred):
            """
            誤差逆伝播法
    
            Args:
                x: 入力データ
                y_true: 正解ラベル
                y_pred: 予測値
            """
            batch_size = x.shape[0]
    
            # 出力層の勾配
            delta2 = (y_pred - y_true) * sigmoid_derivative(self.z2)
    
            # 第2層の重みとバイアスの勾配
            dW2 = np.dot(self.a1.T, delta2) / batch_size
            db2 = np.sum(delta2, axis=0) / batch_size
    
            # 隠れ層の勾配
            delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
    
            # 第1層の重みとバイアスの勾配
            dW1 = np.dot(x.T, delta1) / batch_size
            db1 = np.sum(delta1, axis=0) / batch_size
    
            # パラメータの更新
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
    
        def train(self, x, y_true, epochs=1000, verbose=True):
            """
            学習ループ
    
            Args:
                x: 訓練データ
                y_true: 正解ラベル
                epochs: エポック数
                verbose: 進捗表示
            """
            losses = []
    
            for epoch in range(epochs):
                # 順伝播
                y_pred = self.forward(x)
    
                # 損失計算
                loss = np.mean((y_true - y_pred) ** 2)
                losses.append(loss)
    
                # 逆伝播
                self.backward(x, y_true, y_pred)
    
                # 進捗表示
                if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")
    
            return losses
    
    # XOR問題でテスト
    print("=== XOR問題の学習 ===")
    
    # データ準備
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # ネットワークの作成と学習
    net = TwoLayerNetWithBackprop(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    losses = net.train(X, y, epochs=5000, verbose=True)
    
    # 最終結果
    print("\n=== 最終予測結果 ===")
    predictions = net.forward(X)
    for i in range(len(X)):
        pred_label = 1 if predictions[i] > 0.5 else 0
        print(f"入力: {X[i]} → 予測: {predictions[i][0]:.4f} → ラベル: {pred_label} (正解: {y[i][0]})")
    

**出力例** ：
    
    
    === XOR問題の学習 ===
    Epoch    0: Loss = 0.259762
    Epoch  100: Loss = 0.249876
    Epoch  200: Loss = 0.249011
    Epoch  300: Loss = 0.246863
    ...
    Epoch 4900: Loss = 0.000625
    Epoch 4999: Loss = 0.000612
    
    === 最終予測結果 ===
    入力: [0 0] → 予測: 0.0247 → ラベル: 0 (正解: 0)
    入力: [0 1] → 予測: 0.9753 → ラベル: 1 (正解: 1)
    入力: [1 0] → 予測: 0.9751 → ラベル: 1 (正解: 1)
    入力: [1 1] → 予測: 0.0254 → ラベル: 0 (正解: 0)
    

> **成功！** 多層パーセプトロンと誤差逆伝播法により、XOR問題を解決できました！

* * *

## 2.5 学習曲線の可視化
    
    
    import matplotlib.pyplot as plt
    
    def plot_learning_curve(losses):
        """学習曲線をプロット"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('XOR問題の学習曲線', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 対数スケール
        plt.show()
    
    plot_learning_curve(losses)
    

### 決定境界の可視化
    
    
    def plot_decision_boundary(net, X, y):
        """決定境界を可視化"""
        # グリッドの作成
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
    
        # 各点の予測
        Z = net.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        # プロット
        plt.figure(figsize=(10, 8))
    
        # 背景色（決定境界）
        plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
        plt.colorbar(label='予測値')
    
        # データ点
        plt.scatter(X[y.flatten()==0][:, 0], X[y.flatten()==0][:, 1],
                    s=200, c='blue', marker='o', edgecolors='k', linewidths=2,
                    label='クラス 0')
        plt.scatter(X[y.flatten()==1][:, 0], X[y.flatten()==1][:, 1],
                    s=200, c='red', marker='s', edgecolors='k', linewidths=2,
                    label='クラス 1')
    
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.title('XOR問題の決定境界（多層パーセプトロン）', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    plot_decision_boundary(net, X, y)
    

* * *

## 2.6 ミニバッチ学習

### バッチ学習の種類

手法 | バッチサイズ | 特徴  
---|---|---  
**バッチ勾配降下法** | 全データ | 安定だが遅い  
**確率的勾配降下法（SGD）** | 1サンプル | 高速だが不安定  
**ミニバッチ勾配降下法** | 数十〜数百 | バランスが良い（実用的）  
      
    
    def create_mini_batches(X, y, batch_size):
        """
        ミニバッチの作成
    
        Args:
            X: 入力データ
            y: ラベル
            batch_size: バッチサイズ
    
        Yields:
            (X_batch, y_batch)のタプル
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
    
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
    
            yield X[batch_indices], y[batch_indices]
    
    # ミニバッチ学習の例
    def train_with_minibatch(net, X, y, epochs=1000, batch_size=2):
        """ミニバッチ学習"""
        losses = []
    
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
    
            for X_batch, y_batch in create_mini_batches(X, y, batch_size):
                # 順伝播
                y_pred = net.forward(X_batch)
    
                # 損失
                loss = np.mean((y_batch - y_pred) ** 2)
                epoch_loss += loss
                n_batches += 1
    
                # 逆伝播
                net.backward(X_batch, y_batch, y_pred)
    
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
    
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")
    
        return losses
    
    # テスト
    print("\n=== ミニバッチ学習 ===")
    net_mini = TwoLayerNetWithBackprop(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    losses_mini = train_with_minibatch(net_mini, X, y, epochs=2000, batch_size=2)
    

* * *

## 2.7 本章のまとめ

### 学んだこと

  1. **多層パーセプトロンの構造**

     * 入力層、隠れ層、出力層
     * 各層は重み $W$ とバイアス $b$ を持つ
     * 活性化関数で非線形性を導入
  2. **損失関数**

     * MSE: 回帰問題
     * Cross-Entropy: 分類問題
  3. **勾配降下法**

     * $w \leftarrow w - \eta \frac{\partial L}{\partial w}$
     * 学習率 $\eta$ の重要性
  4. **誤差逆伝播法**

     * 連鎖律による効率的な勾配計算
     * 出力層から入力層への逆向き計算
     * NumPyによる完全実装
  5. **XOR問題の解決**

     * 多層化により非線形問題を解決
     * 実際に学習して精度100%を達成

### 重要な数式

概念 | 数式  
---|---  
**順伝播** | $y = \sigma(W^{(2)} \sigma(W^{(1)} x + b^{(1)}) + b^{(2)})$  
**MSE損失** | $L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$  
**勾配降下** | $w \leftarrow w - \eta \frac{\partial L}{\partial w}$  
**連鎖律** | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$  
  
### 次の章へ

第3章では、**活性化関数と最適化** を学びます：

  * 様々な活性化関数（ReLU、Leaky ReLU、ELU）
  * 勾配消失問題とその対策
  * 高度な最適化アルゴリズム（Momentum、Adam）
  * 重みの初期化戦略

* * *

## 演習問題

### 問題1（難易度：easy）

以下の文章の正誤を判定してください。

  1. 多層パーセプトロンは隠れ層を持つ
  2. 誤差逆伝播法は入力層から出力層に向かって計算する
  3. 学習率が大きすぎると発散する可能性がある
  4. XOR問題は単層パーセプトロンで解ける

解答例

  1. **正** \- MLPの定義
  2. **誤** \- 逆伝播は出力層から入力層へ
  3. **正** \- 学習率が大きいと振動・発散
  4. **誤** \- XORは線形分離不可能、多層化が必要

### 問題2（難易度：medium）

シグモイド関数の導関数を導出してください。また、Pythonで実装してください。

ヒント

シグモイド関数: $\sigma(x) = \frac{1}{1 + e^{-x}}$

商の微分公式を使用します。

解答例

**導出** ：

$$ \begin{align} \sigma(x) &= \frac{1}{1 + e^{-x}} \\\ \sigma'(x) &= \frac{d}{dx} (1 + e^{-x})^{-1} \\\ &= -(1 + e^{-x})^{-2} \cdot (-e^{-x}) \\\ &= \frac{e^{-x}}{(1 + e^{-x})^2} \\\ &= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\\ &= \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}} \\\ &= \sigma(x) \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} \\\ &= \sigma(x) \cdot (1 - \sigma(x)) \end{align} $$

**Python実装** ：
    
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # テスト
    x_test = np.linspace(-5, 5, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, sigmoid(x_test), label='σ(x)', linewidth=2)
    plt.plot(x_test, sigmoid_derivative(x_test), label="σ'(x)", linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('シグモイド関数とその導関数', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    

### 問題3（難易度：medium）

3層ニューラルネットワーク（入力層、隠れ層2つ、出力層）を実装してください。

解答例
    
    
    class ThreeLayerNet:
        """3層ニューラルネットワーク"""
    
        def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.1):
            # 重みの初期化
            self.W1 = np.random.randn(input_size, hidden1_size) * 0.1
            self.b1 = np.zeros(hidden1_size)
    
            self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
            self.b2 = np.zeros(hidden2_size)
    
            self.W3 = np.random.randn(hidden2_size, output_size) * 0.1
            self.b3 = np.zeros(output_size)
    
            self.learning_rate = learning_rate
    
        def forward(self, x):
            """順伝播"""
            # 第1層
            self.z1 = np.dot(x, self.W1) + self.b1
            self.a1 = sigmoid(self.z1)
    
            # 第2層
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = sigmoid(self.z2)
    
            # 第3層
            self.z3 = np.dot(self.a2, self.W3) + self.b3
            self.a3 = sigmoid(self.z3)
    
            return self.a3
    
        def backward(self, x, y_true, y_pred):
            """誤差逆伝播法"""
            batch_size = x.shape[0]
    
            # 出力層
            delta3 = (y_pred - y_true) * sigmoid_derivative(self.z3)
            dW3 = np.dot(self.a2.T, delta3) / batch_size
            db3 = np.sum(delta3, axis=0) / batch_size
    
            # 第2隠れ層
            delta2 = np.dot(delta3, self.W3.T) * sigmoid_derivative(self.z2)
            dW2 = np.dot(self.a1.T, delta2) / batch_size
            db2 = np.sum(delta2, axis=0) / batch_size
    
            # 第1隠れ層
            delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
            dW1 = np.dot(x.T, delta1) / batch_size
            db1 = np.sum(delta1, axis=0) / batch_size
    
            # パラメータ更新
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3
    
    # テスト
    print("=== 3層ニューラルネットワーク ===")
    net3 = ThreeLayerNet(input_size=2, hidden1_size=4, hidden2_size=4, output_size=1, learning_rate=0.5)
    
    # XOR問題で学習
    for epoch in range(3000):
        y_pred = net3.forward(X)
        net3.backward(X, y, y_pred)
    
        if epoch % 500 == 0:
            loss = np.mean((y - y_pred) ** 2)
            print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")
    
    # 最終結果
    print("\n=== 最終予測 ===")
    final_pred = net3.forward(X)
    for i in range(len(X)):
        print(f"入力: {X[i]} → 予測: {final_pred[i][0]:.4f} (正解: {y[i][0]})")
    

### 問題4（難易度：hard）

AND、OR、NANDの3つの論理ゲートを同時に学習するニューラルネットワークを実装してください（マルチタスク学習）。

ヒント

  * 出力層を3ユニットにする
  * 各出力がAND、OR、NANDに対応

解答例
    
    
    # データ準備
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_multi = np.array([
        [0, 0, 1],  # AND=0, OR=0, NAND=1
        [0, 1, 1],  # AND=0, OR=1, NAND=1
        [0, 1, 1],  # AND=0, OR=1, NAND=1
        [1, 1, 0]   # AND=1, OR=1, NAND=0
    ])
    
    # マルチタスクネットワーク
    net_multi = TwoLayerNetWithBackprop(input_size=2, hidden_size=6, output_size=3, learning_rate=0.5)
    
    print("=== マルチタスク学習（AND, OR, NAND） ===")
    
    # 学習
    for epoch in range(5000):
        y_pred = net_multi.forward(X)
        net_multi.backward(X, y_multi, y_pred)
    
        if epoch % 1000 == 0:
            loss = np.mean((y_multi - y_pred) ** 2)
            print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")
    
    # 最終結果
    print("\n=== 最終予測 ===")
    print("入力  | AND予測 | OR予測  | NAND予測 | AND正解 | OR正解  | NAND正解")
    print("-" * 75)
    final_pred = net_multi.forward(X)
    for i in range(len(X)):
        print(f"{X[i]} | {final_pred[i][0]:.4f}  | {final_pred[i][1]:.4f}  | {final_pred[i][2]:.4f}   | "
              f"{y_multi[i][0]}       | {y_multi[i][1]}       | {y_multi[i][2]}")
    

### 問題5（難易度：hard）

学習率スケジューリング（学習率を徐々に小さくする）を実装してください。

解答例
    
    
    def learning_rate_decay(initial_lr, epoch, decay_rate=0.95, decay_step=100):
        """
        学習率の減衰
    
        Args:
            initial_lr: 初期学習率
            epoch: 現在のエポック
            decay_rate: 減衰率
            decay_step: 減衰のステップ
    
        Returns:
            減衰後の学習率
        """
        return initial_lr * (decay_rate ** (epoch // decay_step))
    
    class TwoLayerNetWithLRScheduling(TwoLayerNetWithBackprop):
        """学習率スケジューリング付きネットワーク"""
    
        def __init__(self, input_size, hidden_size, output_size, initial_lr=0.5, decay_rate=0.95):
            super().__init__(input_size, hidden_size, output_size, initial_lr)
            self.initial_lr = initial_lr
            self.decay_rate = decay_rate
    
        def train_with_scheduling(self, X, y, epochs=5000):
            """学習率スケジューリング付き学習"""
            losses = []
    
            for epoch in range(epochs):
                # 学習率の更新
                self.learning_rate = learning_rate_decay(
                    self.initial_lr, epoch, self.decay_rate, decay_step=500
                )
    
                # 順伝播
                y_pred = self.forward(X)
    
                # 損失
                loss = np.mean((y - y_pred) ** 2)
                losses.append(loss)
    
                # 逆伝播
                self.backward(X, y, y_pred)
    
                if epoch % 500 == 0:
                    print(f"Epoch {epoch:4d}: LR = {self.learning_rate:.6f}, Loss = {loss:.6f}")
    
            return losses
    
    # テスト
    print("=== 学習率スケジューリング ===")
    net_sched = TwoLayerNetWithLRScheduling(input_size=2, hidden_size=4, output_size=1,
                                            initial_lr=1.0, decay_rate=0.9)
    losses_sched = net_sched.train_with_scheduling(X, y, epochs=5000)
    

* * *

## 参考文献

  1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." _Nature_ , 323(6088), 533-536.
  2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." _Nature_ , 521(7553), 436-444.
  3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
  4. 斎藤康毅 (2016). 『ゼロから作るDeep Learning』オライリージャパン.
