---
title: "Chapter 3: 正則化手法"
chapter_number: 3
series: 線形回帰・ロジスティック回帰入門
difficulty: 初級〜中級
reading_time: 20-25分
tags: [正則化, Ridge, Lasso, ElasticNet]
prerequisites: [Chapter 1-2]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
last_updated: 2025-10-20
version: 1.0
---

# Chapter 3: 正則化手法

## 本章の概要

正則化は過学習を防ぎ、モデルの汎化性能を向上させる重要な技術です。

### 学習目標
- ✅ L1/L2正則化の違いを理解する
- ✅ Ridge/Lasso/ElasticNetを使い分けられる

---

## 1. Ridge回帰（L2正則化）

損失関数に重みの2乗和を追加：

$$
J(w) = MSE + \alpha \sum_{i=1}^{n} w_i^2
$$

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# データ標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge回帰
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

print(f"R²: {ridge.score(X_test_scaled, y_test):.3f}")
```

---

## 2. Lasso回帰（L1正則化）

重みの絶対値和を追加（特徴量選択効果）：

$$
J(w) = MSE + \alpha \sum_{i=1}^{n} |w_i|
$$

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# ゼロでない係数の数
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"選択された特徴量数: {n_nonzero}")
```

---

## 3. ElasticNet

L1とL2の組み合わせ：

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
```

---

## 4. アルファ値の選択

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(Ridge(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print(f"最適なアルファ: {grid.best_params_['alpha']}")
```

---

**次へ**: [Chapter 4: 実践とチューニング →](chapter-4.html)
