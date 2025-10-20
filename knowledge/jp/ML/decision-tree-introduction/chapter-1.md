---
title: "Chapter 1: 決定木の基礎"
chapter_number: 1
series: 決定木・ランダムフォレスト入門
difficulty: 初級
reading_time: 25-30分
tags: [決定木, ジニ不純度, 情報利得]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 1: 決定木の基礎

## 1. 決定木とは

if-then-elseルールで構成される木構造のモデル。直感的で解釈しやすい。

## 2. 分岐基準

**ジニ不純度:**
$$
Gini = 1 - \sum_{i=1}^{C} p_i^2
$$

## 3. 実装

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# 木構造の可視化
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=iris.feature_names,
         class_names=iris.target_names, filled=True)
plt.show()

# 特徴量重要度
import pandas as pd
importance = pd.DataFrame({
    '特徴量': iris.feature_names,
    '重要度': model.feature_importances_
}).sort_values('重要度', ascending=False)
print(importance)
```

---

**次へ**: [Chapter 2: 過学習の制御 →](chapter-2.html)
