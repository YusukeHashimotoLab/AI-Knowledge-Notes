---
title: "Chapter 2: 特徴量選択"
chapter_number: 2
series: 特徴量エンジニアリング入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [特徴量選択, フィルタ法, ラッパー法, 埋め込み法]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 2: 特徴量選択

## 1. フィルタ法（Filter Method）

統計的指標でスコア付けして選択。

### 相関係数による選択
```python
import pandas as pd
import numpy as np

# 目的変数との相関
correlation = df.corr()['target'].abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.3].index.tolist()

# 多重共線性の除去
corr_matrix = df[selected_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
```

### 分散による選択
```python
from sklearn.feature_selection import VarianceThreshold

# 低分散特徴量の削除
selector = VarianceThreshold(threshold=0.01)
X_high_variance = selector.fit_transform(X)
```

### 統計的検定
```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# F値による選択（分類）
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# カイ二乗検定（カテゴリ変数）
selector_chi2 = SelectKBest(score_func=chi2, k=10)
X_selected = selector_chi2.fit_transform(X_positive, y)
```

## 2. ラッパー法（Wrapper Method）

モデルの性能を評価しながら選択。

### 再帰的特徴削除（RFE）
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFEで重要度の低い特徴量を削除
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector.fit(X, y)

selected_features = X.columns[selector.support_]
print(f"選択された特徴量: {selected_features.tolist()}")
```

### 順方向選択・後方削除
```python
from mlxtend.feature_selection import SequentialFeatureSelector

# 順方向選択
sfs = SequentialFeatureSelector(
    estimator,
    k_features=10,
    forward=True,
    scoring='accuracy',
    cv=5
)
sfs.fit(X, y)
selected_features = list(sfs.k_feature_names_)
```

## 3. 埋め込み法（Embedded Method）

モデル学習と同時に選択。

### L1正則化（Lasso）
```python
from sklearn.linear_model import LassoCV

# Lassoで特徴量選択
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

# 係数が0でない特徴量を選択
selected_features = X.columns[lasso.coef_ != 0]
```

### 木ベースモデルの特徴量重要度
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Random Forestで重要度を計算
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 重要度の可視化
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
plt.xlabel('Importance')
plt.title('Feature Importance (Top 15)')
plt.show()

# 閾値で選択
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_selected = selector.transform(X)
```

## 4. 実践的な選択プロセス

```python
# ステップ1: 低分散特徴量を削除
var_selector = VarianceThreshold(threshold=0.01)
X_step1 = var_selector.fit_transform(X)

# ステップ2: 相関の高い特徴量を削除
# (実装は上記参照)

# ステップ3: RFEで最終選択
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
X_final = rfe.fit_transform(X_step2, y)

print(f"最終的な特徴量数: {X_final.shape[1]}")
```

---

**次へ**: [Chapter 3: 特徴量抽出 →](chapter-3.html)
