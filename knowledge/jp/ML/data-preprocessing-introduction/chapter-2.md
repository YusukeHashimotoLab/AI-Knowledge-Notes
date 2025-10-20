---
title: "Chapter 2: 欠損値処理"
chapter_number: 2
series: データ前処理入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [欠損値, 補完, SimpleImputer]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 2: 欠損値処理

## 1. 欠損値の検出

```python
# 欠損値の確認
print(df.isnull().sum())

# 欠損値の割合
missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio)

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.show()
```

## 2. 削除による処理

```python
# 欠損値を含む行を削除
df_dropna = df.dropna()

# 特定列の欠損値のみ削除
df_dropna = df.dropna(subset=['age', 'income'])

# 閾値を設定して削除（50%以上欠損している列を削除）
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)
```

## 3. 補完による処理

```python
from sklearn.impute import SimpleImputer

# 平均値で補完
imputer_mean = SimpleImputer(strategy='mean')
df['age'] = imputer_mean.fit_transform(df[['age']])

# 中央値で補完
imputer_median = SimpleImputer(strategy='median')
df['income'] = imputer_median.fit_transform(df[['income']])

# 最頻値で補完
imputer_mode = SimpleImputer(strategy='most_frequent')
df['category'] = imputer_mode.fit_transform(df[['category']])

# 定数で補完
imputer_const = SimpleImputer(strategy='constant', fill_value='Unknown')
df['status'] = imputer_const.fit_transform(df[['status']])
```

## 4. 高度な補完手法

```python
# K-NN補完
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    knn_imputer.fit_transform(df),
    columns=df.columns
)

# 前方補完（時系列データ）
df['value'] = df['value'].fillna(method='ffill')

# 後方補完
df['value'] = df['value'].fillna(method='bfill')

# 線形補間
df['value'] = df['value'].interpolate(method='linear')
```

---

**次へ**: [Chapter 3: エンコーディングとスケーリング →](chapter-3.html)
