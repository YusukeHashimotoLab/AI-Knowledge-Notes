---
title: "Chapter 2: 過学習の制御"
chapter_number: 2
series: 決定木・ランダムフォレスト入門
difficulty: 初級〜中級
reading_time: 20-25分
tags: [過学習, 剪定, ハイパーパラメータ]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 2: 過学習の制御

## 1. 過学習の問題

決定木は深くなりすぎると訓練データに過度に適合します。

## 2. 制御パラメータ

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,           # 最大深さ
    min_samples_split=20,  # 分岐に必要な最小サンプル数
    min_samples_leaf=10,   # 葉の最小サンプル数
    max_features='sqrt',   # 使用する特徴量数
    random_state=42
)
```

## 3. パラメータチューニング

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                   param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"最適パラメータ: {grid.best_params_}")
```

---

**次へ**: [Chapter 3: ランダムフォレスト →](chapter-3.html)
