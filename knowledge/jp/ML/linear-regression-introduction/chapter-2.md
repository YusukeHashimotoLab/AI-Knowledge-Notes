---
title: "Chapter 2: ロジスティック回帰"
chapter_number: 2
series: 線形回帰・ロジスティック回帰入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [ロジスティック回帰, 分類, シグモイド関数]
prerequisites: [Chapter 1]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
last_updated: 2025-10-20
version: 1.0
---

# Chapter 2: ロジスティック回帰

## 本章の概要

ロジスティック回帰は、線形モデルを分類問題に拡張した手法です。

### 学習目標
- ✅ シグモイド関数の役割を理解する
- ✅ 確率的分類を実装できる
- ✅ 多クラス分類に対応できる

---

## 1. シグモイド関数

線形モデルの出力を0〜1の確率に変換：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

## 2. 2値分類の実装

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# データ読み込み
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデル訓練
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 評価
print(f"正解率: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))
```

---

## 3. 多クラス分類

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-vs-Rest 方式
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X_train, y_train)

print(f"正解率: {model.score(X_test, y_test):.2%}")
```

---

## 4. まとめ

✅ ロジスティック回帰による分類を実装しました
✅ 多クラス分類にも対応できるようになりました

---

**次へ**: [Chapter 3: 正則化手法 →](chapter-3.html)
