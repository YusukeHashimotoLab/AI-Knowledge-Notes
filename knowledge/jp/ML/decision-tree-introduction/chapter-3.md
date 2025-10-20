---
title: "Chapter 3: ランダムフォレスト"
chapter_number: 3
series: 決定木・ランダムフォレスト入門
difficulty: 初級〜中級
reading_time: 30-35分
tags: [ランダムフォレスト, バギング, アンサンブル]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 3: ランダムフォレスト

## 1. ランダムフォレストとは

複数の決定木を組み合わせたアンサンブル手法。

## 2. 実装

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(
    n_estimators=100,  # 木の数
    max_depth=10,
    random_state=42
)

# 交差検証
scores = cross_val_score(rf, X, y, cv=5)
print(f"平均スコア: {scores.mean():.3f} (+/- {scores.std():.3f})")

rf.fit(X_train, y_train)
print(f"テストスコア: {rf.score(X_test, y_test):.3f}")
```

## 3. Out-of-Bag評価

```python
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # OOBスコアを計算
    random_state=42
)
rf_oob.fit(X_train, y_train)
print(f"OOBスコア: {rf_oob.oob_score_:.3f}")
```

---

**次へ**: [Chapter 4: 実践とXGBoost →](chapter-4.html)
