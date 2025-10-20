---
title: "Chapter 1: 特徴量の基礎"
chapter_number: 1
series: 特徴量エンジニアリング入門
difficulty: 初級
reading_time: 25-30分
tags: [特徴量, 交互作用, 集約]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 1: 特徴量の基礎

## 1. 特徴量エンジニアリングとは

機械学習において最も重要なのは「どんなアルゴリズムを使うか」ではなく「どんな特徴量を設計するか」。

## 2. 基本的な特徴量生成

### 数値特徴量の変換
```python
import pandas as pd
import numpy as np

# 対数変換
df['price_log'] = np.log1p(df['price'])

# 平方根変換
df['area_sqrt'] = np.sqrt(df['area'])

# 2乗変換
df['age_squared'] = df['age'] ** 2

# ビニング（離散化）
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['young', 'adult', 'middle', 'senior'])
```

### 交互作用特徴量
```python
# 四則演算
df['price_per_area'] = df['price'] / df['area']
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# 掛け算による交互作用
df['age_income_interaction'] = df['age'] * df['income']

# PolynomialFeaturesで自動生成
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
interactions = poly.fit_transform(df[['feature1', 'feature2', 'feature3']])
```

## 3. カテゴリ特徴量の処理

### 集約特徴量
```python
# グループごとの統計量
user_stats = df.groupby('user_id').agg({
    'purchase_amount': ['mean', 'sum', 'count', 'std'],
    'visit_count': 'sum'
})

# カウントエンコーディング
category_counts = df['category'].value_counts()
df['category_count'] = df['category'].map(category_counts)

# ターゲットエンコーディング
target_mean = df.groupby('category')['target'].mean()
df['category_target_mean'] = df['category'].map(target_mean)
```

## 4. 時間特徴量

```python
# 日時から特徴量を抽出
df['datetime'] = pd.to_datetime(df['datetime'])

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_holiday'] = df['datetime'].dt.date.isin(holidays).astype(int)

# 周期性の表現（円周率を使用）
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

**次へ**: [Chapter 2: 特徴量選択 →](chapter-2.html)
