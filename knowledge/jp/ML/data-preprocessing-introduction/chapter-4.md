---
title: "Chapter 4: 外れ値検出と特徴量変換"
chapter_number: 4
series: データ前処理入門
difficulty: 初級〜中級
reading_time: 20-25分
tags: [外れ値, IQR, Isolation Forest, 対数変換]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 4: 外れ値検出と特徴量変換

## 1. 外れ値の検出方法

### IQR法（四分位範囲法）
```python
# IQRによる外れ値検出
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"外れ値数: {len(outliers)}")
```

### Z-score法
```python
from scipy import stats

# Z-scoreによる外れ値検出
z_scores = np.abs(stats.zscore(df['income']))
outliers = df[z_scores > 3]
```

### Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# 機械学習による外れ値検出
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(df[['age', 'income']])

df['outlier'] = outlier_labels
outliers = df[df['outlier'] == -1]
```

## 2. 外れ値の対処法

```python
# 削除
df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# 上限・下限でクリッピング
df['price_clipped'] = df['price'].clip(lower=lower_bound, upper=upper_bound)

# 中央値で置換
median = df['price'].median()
df['price_fixed'] = df['price'].apply(
    lambda x: median if (x < lower_bound or x > upper_bound) else x
)
```

## 3. 特徴量変換

### 対数変換
```python
# 対数変換（正の歪度を持つデータ）
df['income_log'] = np.log1p(df['income'])

# Box-Cox変換
from scipy.stats import boxcox
df['income_boxcox'], _ = boxcox(df['income'] + 1)
```

### ビニング（離散化）
```python
# 等幅ビニング
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['A', 'B', 'C', 'D', 'E'])

# 等頻度ビニング
df['income_bin'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 多項式特徴量
```python
from sklearn.preprocessing import PolynomialFeatures

# 2次多項式特徴量の生成
poly = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly.fit_transform(df[['feature1', 'feature2']])
```

## 4. 完全なパイプライン

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 前処理パイプライン
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_processed = preprocessing_pipeline.fit_transform(X_train)
```

**本シリーズ完了！**

---

**目次へ**: [↑ シリーズ目次](index.html)
