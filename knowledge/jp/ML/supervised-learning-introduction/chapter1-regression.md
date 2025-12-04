---
title: 第1章：回帰問題の基礎
chapter_title: 第1章：回帰問題の基礎
subtitle: 連続値予測の理論と実装 - 線形回帰から正則化まで
reading_time: 20-25分
difficulty: 初級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 回帰問題の定義と応用例を理解する
  * ✅ 線形回帰の数学的背景を説明できる
  * ✅ 最小二乗法と勾配降下法を実装できる
  * ✅ 多項式回帰で非線形関係をモデル化できる
  * ✅ 正則化（Ridge, Lasso, Elastic Net）を適用できる
  * ✅ R²、RMSE、MAEで回帰モデルを評価できる

* * *

## 1.1 回帰問題とは

### 定義

**回帰問題（Regression）** は、入力変数から**連続値** の出力を予測する教師あり学習のタスクです。

> 「特徴量 $X$ から目的変数 $y$ を予測する関数 $f: X \rightarrow y$ を学習する」

### 回帰 vs 分類

タスク | 出力 | 例  
---|---|---  
**回帰** | 連続値（数値） | 住宅価格予測、気温予測、売上予測  
**分類** | 離散値（カテゴリ） | 画像分類、スパム判定、疾病診断  
  
### 実世界の応用例
    
    
    ```mermaid
    graph LR
        A[回帰問題の応用] --> B[金融: 株価予測]
        A --> C[不動産: 住宅価格予測]
        A --> D[製造: 需要予測]
        A --> E[医療: 患者滞在期間予測]
        A --> F[マーケティング: 売上予測]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 1.2 線形回帰の理論

### 単回帰モデル

**単回帰（Simple Linear Regression）** は、1つの特徴量から予測を行います。

$$ y = w_0 + w_1 x + \epsilon $$

  * $y$: 目的変数（予測したい値）
  * $x$: 説明変数（特徴量）
  * $w_0$: 切片（intercept, bias）
  * $w_1$: 傾き（slope, weight）
  * $\epsilon$: 誤差項

### 重回帰モデル

**重回帰（Multiple Linear Regression）** は、複数の特徴量を使用します。

$$ y = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + \epsilon $$

行列表記：

$$ \mathbf{y} = \mathbf{X}\mathbf{w} + \epsilon $$

  * $\mathbf{y}$: 目的変数ベクトル（shape: $m \times 1$）
  * $\mathbf{X}$: 特徴量行列（shape: $m \times (n+1)$）
  * $\mathbf{w}$: 重みベクトル（shape: $(n+1) \times 1$）
  * $m$: サンプル数、$n$: 特徴量数

### 損失関数（Loss Function）

**平均二乗誤差（Mean Squared Error, MSE）** を最小化します：

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{m} ||\mathbf{y} - \mathbf{X}\mathbf{w}||^2 $$

  * $y^{(i)}$: 実際の値
  * $\hat{y}^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)}$: 予測値

* * *

## 1.3 最小二乗法（Ordinary Least Squares）

### 解析解

MSEを最小化する重み $\mathbf{w}$ は、解析的に求められます：

$$ \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $$

これを**正規方程式（Normal Equation）** と呼びます。

### 実装例：単回帰
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # バイアス項を追加
    X_b = np.c_[np.ones((100, 1)), X]  # shape: (100, 2)
    
    # 正規方程式で重みを計算
    w_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print("学習した重み:")
    print(f"w0 (切片): {w_best[0][0]:.4f}")
    print(f"w1 (傾き): {w_best[1][0]:.4f}")
    
    # 予測
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b @ w_best
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='データ')
    plt.plot(X_new, y_predict, 'r-', linewidth=2, label='予測直線')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('線形回帰 - 最小二乗法', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    学習した重み:
    w0 (切片): 4.2153
    w1 (傾き): 2.7702
    

### scikit-learnによる実装
    
    
    from sklearn.linear_model import LinearRegression
    
    # モデル構築
    model = LinearRegression()
    model.fit(X, y)
    
    print("\nscikit-learn:")
    print(f"切片: {model.intercept_[0]:.4f}")
    print(f"傾き: {model.coef_[0][0]:.4f}")
    
    # 予測
    y_pred = model.predict(X_new)
    print(f"\n予測値: {y_pred.flatten()}")
    

* * *

## 1.4 勾配降下法（Gradient Descent）

### 原理

損失関数の勾配を計算し、勾配の逆方向に重みを更新します。
    
    
    ```mermaid
    graph LR
        A[初期重み w] --> B[勾配計算 ∇J]
        B --> C[重み更新 w := w - α∇J]
        C --> D{収束?}
        D -->|No| B
        D -->|Yes| E[最適重み w*]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#ffe0b2
        style E fill:#e8f5e9
    ```

### 更新式

$$ \mathbf{w} := \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}) $$

勾配：

$$ \nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{2}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) $$

  * $\alpha$: 学習率（learning rate）

### 実装例
    
    
    def gradient_descent(X, y, alpha=0.01, n_iterations=1000):
        """
        勾配降下法で線形回帰を学習
    
        Args:
            X: 特徴量行列 (バイアス項含む)
            y: 目的変数
            alpha: 学習率
            n_iterations: イテレーション数
    
        Returns:
            w: 学習した重み
            history: 損失関数の履歴
        """
        m = len(y)
        w = np.random.randn(X.shape[1], 1)  # 重みの初期化
        history = []
    
        for i in range(n_iterations):
            # 予測
            y_pred = X @ w
    
            # 損失計算
            loss = (1 / m) * np.sum((y_pred - y) ** 2)
            history.append(loss)
    
            # 勾配計算
            gradients = (2 / m) * X.T @ (y_pred - y)
    
            # 重み更新
            w = w - alpha * gradients
    
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
        return w, history
    
    # 実行
    w_gd, loss_history = gradient_descent(X_b, y, alpha=0.1, n_iterations=1000)
    
    print("\n勾配降下法で学習した重み:")
    print(f"w0 (切片): {w_gd[0][0]:.4f}")
    print(f"w1 (傾き): {w_gd[1][0]:.4f}")
    
    # 損失関数の推移を可視化
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('勾配降下法の収束過程', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    Iteration 0: Loss = 6.8421
    Iteration 100: Loss = 0.8752
    Iteration 200: Loss = 0.8284
    Iteration 300: Loss = 0.8243
    Iteration 400: Loss = 0.8236
    Iteration 500: Loss = 0.8235
    Iteration 600: Loss = 0.8235
    Iteration 700: Loss = 0.8235
    Iteration 800: Loss = 0.8235
    Iteration 900: Loss = 0.8235
    
    勾配降下法で学習した重み:
    w0 (切片): 4.2152
    w1 (傾き): 2.7703
    

### 学習率の重要性
    
    
    # 異なる学習率での比較
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(12, 8))
    for i, alpha in enumerate(learning_rates):
        w, history = gradient_descent(X_b, y, alpha=alpha, n_iterations=100)
        plt.subplot(2, 2, i+1)
        plt.plot(history, linewidth=2)
        plt.title(f'学習率 α = {alpha}', fontsize=12)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.5 多項式回帰（Polynomial Regression）

### 概要

線形回帰では表現できない**非線形関係** をモデル化します。

$$ y = w_0 + w_1 x + w_2 x^2 + \cdots + w_d x^d $$

特徴量を変換することで、線形回帰の枠組みを使用できます。

### 実装例
    
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    # 非線形データ生成
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    
    # 多項式回帰（次数2）
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    print("多項式回帰の係数:")
    print(f"w1 (x): {model.coef_[0][0]:.4f}")
    print(f"w2 (x²): {model.coef_[0][1]:.4f}")
    print(f"切片: {model.intercept_[0]:.4f}")
    
    # 予測と可視化
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='データ')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='多項式回帰 (次数2)')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('多項式回帰', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### 過学習の危険性
    
    
    # 異なる次数での比較
    degrees = [1, 2, 5, 10]
    
    plt.figure(figsize=(14, 10))
    for i, degree in enumerate(degrees):
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
    
        model = LinearRegression()
        model.fit(X_poly, y)
    
        X_test_poly = poly_features.transform(X_test)
        y_pred = model.predict(X_test_poly)
    
        plt.subplot(2, 2, i+1)
        plt.scatter(X, y, alpha=0.6, label='データ')
        plt.plot(X_test, y_pred, 'r-', linewidth=2, label=f'次数 {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'多項式回帰（次数 {degree}）', fontsize=12)
        plt.ylim(-5, 15)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

> **注意** : 次数が高すぎると過学習（overfitting）が発生します。次数10ではデータに過剰適合し、汎化性能が低下します。

* * *

## 1.6 正則化（Regularization）

### 概要

過学習を防ぐため、損失関数に**ペナルティ項** を追加します。
    
    
    ```mermaid
    graph TD
        A[正則化手法] --> B[Ridge L2正則化]
        A --> C[Lasso L1正則化]
        A --> D[Elastic Net L1+L2]
    
        B --> B1[重みの大きさを抑制]
        C --> C1[重みの一部をゼロに]
        D --> D1[両方のバランス]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### Ridge回帰（L2正則化）

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \sum_{j=1}^{n} w_j^2 $$

  * $\alpha$: 正則化パラメータ
  * 重みの二乗和にペナルティ

    
    
    from sklearn.linear_model import Ridge
    
    # Ridge回帰
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_poly, y)
    
    print("Ridge回帰の係数:")
    print(f"重み: {ridge_model.coef_[0]}")
    print(f"切片: {ridge_model.intercept_[0]:.4f}")
    

### Lasso回帰（L1正則化）

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \sum_{j=1}^{n} |w_j| $$

  * 重みの絶対値和にペナルティ
  * **スパース性** : 重要でない特徴量の重みをゼロにする

    
    
    from sklearn.linear_model import Lasso
    
    # Lasso回帰
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_poly, y)
    
    print("\nLasso回帰の係数:")
    print(f"重み: {lasso_model.coef_}")
    print(f"切片: {lasso_model.intercept_:.4f}")
    print(f"ゼロの重み数: {np.sum(lasso_model.coef_ == 0)}")
    

### Elastic Net（L1 + L2正則化）

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 + \alpha \rho \sum_{j=1}^{n} |w_j| + \frac{\alpha(1-\rho)}{2} \sum_{j=1}^{n} w_j^2 $$

  * $\rho$: L1とL2のバランス（0〜1）

    
    
    from sklearn.linear_model import ElasticNet
    
    # Elastic Net
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_model.fit(X_poly, y)
    
    print("\nElastic Net回帰の係数:")
    print(f"重み: {elastic_model.coef_}")
    print(f"切片: {elastic_model.intercept_:.4f}")
    

### 正則化パラメータの比較
    
    
    # 異なるalphaでの比較
    alphas = np.logspace(-3, 2, 100)
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, y)
        ridge_coefs.append(ridge.coef_[0])
    
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_poly, y)
        lasso_coefs.append(lasso.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    plt.figure(figsize=(14, 6))
    
    # Ridge
    plt.subplot(1, 2, 1)
    for i in range(X_poly.shape[1]):
        plt.plot(alphas, ridge_coefs[:, i], label=f'w{i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha (正則化強度)', fontsize=12)
    plt.ylabel('係数の大きさ', fontsize=12)
    plt.title('Ridge回帰: 正則化パラメータの影響', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lasso
    plt.subplot(1, 2, 2)
    for i in range(X_poly.shape[1]):
        plt.plot(alphas, lasso_coefs[:, i], label=f'w{i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha (正則化強度)', fontsize=12)
    plt.ylabel('係数の大きさ', fontsize=12)
    plt.title('Lasso回帰: 正則化パラメータの影響', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 1.7 回帰モデルの評価

### 評価指標

指標 | 数式 | 説明  
---|---|---  
**平均絶対誤差**  
(MAE) | $\frac{1}{m}\sum|y_i - \hat{y}_i|$ | 予測誤差の平均（外れ値に頑健）  
**平均二乗誤差**  
(MSE) | $\frac{1}{m}\sum(y_i - \hat{y}_i)^2$ | 予測誤差の二乗平均（外れ値に敏感）  
**平均二乗平方根誤差**  
(RMSE) | $\sqrt{\frac{1}{m}\sum(y_i - \hat{y}_i)^2}$ | MSEの平方根（元の単位）  
**決定係数**  
(R²) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | モデルの説明力（0〜1、高いほど良い）  
  
### 実装例
    
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # データ分割
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # モデル学習
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 評価
    print("=== 訓練データ ===")
    print(f"MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"R²:   {r2_score(y_train, y_train_pred):.4f}")
    
    print("\n=== テストデータ ===")
    print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"R²:   {r2_score(y_test, y_test_pred):.4f}")
    
    # 残差プロット
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(14, 6))
    
    # 予測 vs 実際
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('実際の値', fontsize=12)
    plt.ylabel('予測値', fontsize=12)
    plt.title('予測 vs 実際', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 残差プロット
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('予測値', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title('残差プロット', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 訓練データ ===
    MAE:  0.7234
    MSE:  0.8456
    RMSE: 0.9196
    R²:   0.9145
    
    === テストデータ ===
    MAE:  0.7891
    MSE:  0.9234
    RMSE: 0.9609
    R²:   0.9023
    

* * *

## 1.8 本章のまとめ

### 学んだこと

  1. **回帰問題の定義**

     * 連続値の予測タスク
     * 実世界の応用例（価格予測、需要予測など）
  2. **線形回帰**

     * 最小二乗法による解析解
     * 勾配降下法による数値解
  3. **多項式回帰**

     * 非線形関係のモデル化
     * 過学習の危険性
  4. **正則化**

     * Ridge（L2）: 重みの大きさを抑制
     * Lasso（L1）: スパース性の導入
     * Elastic Net: 両方のバランス
  5. **評価指標**

     * MAE、MSE、RMSE、R²
     * 残差分析の重要性

### 次の章へ

第2章では、**分類問題の基礎** を学びます：

  * ロジスティック回帰
  * 決定木
  * k-NN、SVM
  * 評価指標（精度、再現率、F1スコア）

* * *

## 演習問題

### 問題1（難易度：easy）

回帰問題と分類問題の違いを3つ挙げてください。

解答例

**解答** ：

  1. **出力の種類** : 回帰は連続値、分類は離散値（カテゴリ）
  2. **損失関数** : 回帰はMSE、分類は交差エントロピー
  3. **評価指標** : 回帰はRMSE/R²、分類は精度/F1スコア

### 問題2（難易度：medium）

以下のデータで線形回帰を実装し、重みとバイアスを求めてください。
    
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [5], [4], [5]])
    

解答例
    
    
    import numpy as np
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [5], [4], [5]])
    
    # バイアス項を追加
    X_b = np.c_[np.ones((5, 1)), X]
    
    # 正規方程式
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print(f"切片 w0: {w[0][0]:.4f}")
    print(f"傾き w1: {w[1][0]:.4f}")
    
    # 予測
    y_pred = X_b @ w
    print(f"\n予測値: {y_pred.flatten()}")
    

**出力** ：
    
    
    切片 w0: 2.2000
    傾き w1: 0.6000
    
    予測値: [2.8 3.4 4.  4.6 5.2]
    

### 問題3（難易度：medium）

学習率が大きすぎる場合、勾配降下法でどのような問題が発生しますか？

解答例

**解答** ：

  * **発散（divergence）** : 損失関数が最小値を通り越して発散する
  * **振動** : 最小値の周りで振動し続ける
  * **収束しない** : 最適解に到達できない

**対策** ：

  * 学習率を小さくする（例: 0.1 → 0.01）
  * 学習率スケジューリングを使用する
  * 適応的最適化手法（Adam、RMSprop）を使用する

### 問題4（難易度：hard）

Ridge回帰とLasso回帰の違いを説明し、どのような場合にそれぞれを使うべきか述べてください。

解答例

**Ridge回帰（L2正則化）** ：

  * 重みの二乗和にペナルティ
  * 重みを小さくするが、ゼロにはしない
  * **使用場面** : 多重共線性がある場合、すべての特徴量が重要な場合

**Lasso回帰（L1正則化）** ：

  * 重みの絶対値和にペナルティ
  * 重要でない特徴量の重みをゼロにする（スパース性）
  * **使用場面** : 特徴量選択が必要な場合、解釈性を高めたい場合

**選択基準** ：

状況 | 推奨手法  
---|---  
特徴量が多く、重要度が不明 | Lasso  
多重共線性がある | Ridge  
特徴量選択が必要 | Lasso  
すべての特徴量を使いたい | Ridge  
どちらか不明 | Elastic Net  
  
### 問題5（難易度：hard）

以下のコードを完成させ、交差検証でRidge回帰の最適なalphaを見つけてください。
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    
    # データ生成（省略）
    alphas = np.logspace(-3, 3, 50)
    
    # ここに実装
    

解答例
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    alphas = np.logspace(-3, 3, 50)
    scores = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        # 5分割交差検証
        cv_scores = cross_val_score(ridge, X_poly, y.ravel(),
                                     cv=5, scoring='neg_mean_squared_error')
        scores.append(-cv_scores.mean())  # 負のMSEを正に変換
    
    # 最適なalphaを見つける
    best_alpha = alphas[np.argmin(scores)]
    best_score = np.min(scores)
    
    print(f"最適なalpha: {best_alpha:.4f}")
    print(f"最小MSE: {best_score:.4f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, scores, linewidth=2)
    plt.axvline(best_alpha, color='r', linestyle='--',
                label=f'最適alpha = {best_alpha:.4f}')
    plt.xscale('log')
    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('MSE (交差検証)', fontsize=12)
    plt.title('Ridge回帰: 最適なalphaの探索', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    最適なalpha: 2.1544
    最小MSE: 1.0234
    

* * *

## 参考文献

  1. Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer.
  2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
  3. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly Media.
