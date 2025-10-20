---
title: "Chapter 4: 実践とXGBoost"
chapter_number: 4
series: 決定木・ランダムフォレスト入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [XGBoost, 勾配ブースティング, 実践]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 4: 実践とXGBoost

## 1. 勾配ブースティング

順次的に弱学習器を追加し、誤差を修正していく手法。

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
print(f"スコア: {gb.score(X_test, y_test):.3f}")
```

## 2. XGBoost

```python
# XGBoost (pip install xgboost)
try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    print(f"XGBoost スコア: {xgb_model.score(X_test, y_test):.3f}")
except ImportError:
    print("XGBoostがインストールされていません")
```

## 3. 比較

```python
from sklearn.model_selection import cross_val_score

models = {
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f}")
```

**本シリーズ完了！**

---

**目次へ**: [↑ シリーズ目次](index.html)
