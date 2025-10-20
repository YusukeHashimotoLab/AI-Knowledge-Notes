---
title: "Chapter 1: 線形回帰の基礎"
chapter_number: 1
series: 線形回帰・ロジスティック回帰入門
difficulty: 初級
reading_time: 25-30分
tags: [線形回帰, 最小二乗法, 予測モデル]
prerequisites: [Python基礎, NumPy基礎]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
last_updated: 2025-10-20
version: 1.0
---

# Chapter 1: 線形回帰の基礎

## 本章の概要

線形回帰は、特徴量とターゲット変数の線形関係をモデル化する最も基本的な回帰手法です。

### 学習目標
- ✅ 線形回帰の数学的定義を理解する
- ✅ 最小二乗法による解法を学ぶ
- ✅ scikit-learnで実装できる

---

## 1. 線形回帰とは

**定義**: 特徴量 $x$ とターゲット $y$ の関係を直線（または超平面）でモデル化

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = w_0 + \mathbf{w}^T \mathbf{x}
$$

---

## 2. 単回帰の実装

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データ生成
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 5 + np.random.randn(100, 1) * 2

# モデル訓練
model = LinearRegression()
model.fit(X, y)

# 予測
y_pred = model.predict(X)

# 結果表示
print(f"係数: {model.coef_[0][0]:.2f}")
print(f"切片: {model.intercept_[0]:.2f}")

# 可視化
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('線形回帰')
plt.show()
```

---

## 3. 重回帰

複数の特徴量を使用：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# データ読み込み
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデル訓練
model = LinearRegression()
model.fit(X_train, y_train)

# 評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")
```

---

## 4. まとめ

✅ 線形回帰の基本原理を理解しました
✅ 単回帰と重回帰を実装できるようになりました

---

**次へ**: [Chapter 2: ロジスティック回帰 →](chapter-2.html)
