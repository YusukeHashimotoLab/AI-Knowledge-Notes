---
title: "Chapter 4: 実践とチューニング"
chapter_number: 4
series: 線形回帰・ロジスティック回帰入門
difficulty: 初級〜中級
reading_time: 20-25分
tags: [特徴量選択, 多項式特徴量, チューニング]
prerequisites: [Chapter 1-3]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
last_updated: 2025-10-20
version: 1.0
---

# Chapter 4: 実践とチューニング

## 本章の概要

線形モデルの性能を最大化する実践的テクニックを学びます。

---

## 1. 多項式特徴量

非線形関係を線形モデルで扱う：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 2次の多項式特徴量
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

poly_pipeline.fit(X_train, y_train)
print(f"R²: {poly_pipeline.score(X_test, y_test):.3f}")
```

---

## 2. 特徴量選択

```python
from sklearn.feature_selection import SelectKBest, f_regression

# 上位k個の特徴量を選択
selector = SelectKBest(f_regression, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

model = LinearRegression()
model.fit(X_train_selected, y_train)
```

---

## 3. 完全なパイプライン

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# エンドツーエンドパイプライン
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=10))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 4. まとめ

✅ 線形回帰・ロジスティック回帰を完全にマスターしました
✅ 実践的なチューニング手法を習得しました

**本シリーズ完了！次は決定木シリーズへ進んでください。**

---

**目次へ**: [↑ シリーズ目次](index.html)
