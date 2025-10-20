---
title: "Chapter 3: エンコーディングとスケーリング"
chapter_number: 3
series: データ前処理入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [エンコーディング, スケーリング, 標準化, 正規化]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 3: エンコーディングとスケーリング

## 1. カテゴリ変数のエンコーディング

### ラベルエンコーディング
```python
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディング
le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])  # S, M, L → 0, 1, 2
```

### One-Hotエンコーディング
```python
# One-Hotエンコーディング
df_encoded = pd.get_dummies(df, columns=['category', 'color'])

# scikit-learnで実装
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')
encoded = ohe.fit_transform(df[['category']])
encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(['category'])
)
```

### 順序エンコーディング
```python
from sklearn.preprocessing import OrdinalEncoder

# 順序がある場合
ordinal_encoder = OrdinalEncoder(
    categories=[['Low', 'Medium', 'High']]
)
df['priority_encoded'] = ordinal_encoder.fit_transform(df[['priority']])
```

## 2. 数値変数のスケーリング

### 標準化（Standardization）
```python
from sklearn.preprocessing import StandardScaler

# 平均0、標準偏差1に変換
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['age', 'income', 'score']])

df_scaled = pd.DataFrame(
    df_scaled,
    columns=['age_scaled', 'income_scaled', 'score_scaled']
)
```

### 正規化（Normalization）
```python
from sklearn.preprocessing import MinMaxScaler

# 0-1の範囲に変換
normalizer = MinMaxScaler()
df_normalized = normalizer.fit_transform(df[['age', 'income']])

# カスタム範囲
normalizer_custom = MinMaxScaler(feature_range=(-1, 1))
df_normalized = normalizer_custom.fit_transform(df[['score']])
```

### ロバストスケーリング
```python
from sklearn.preprocessing import RobustScaler

# 中央値と四分位範囲を使用（外れ値に強い）
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df[['income', 'age']])
```

## 3. 手法の選択基準

```python
# 比較実験
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scalers = {
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler()
}

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(df[['feature']])
    print(f"{name}: min={scaled.min():.3f}, max={scaled.max():.3f}")
```

**選択基準:**
- **StandardScaler**: 正規分布に近いデータ、外れ値が少ない
- **MinMaxScaler**: 範囲が明確なデータ、ニューラルネットワーク
- **RobustScaler**: 外れ値が多いデータ

---

**次へ**: [Chapter 4: 外れ値検出と特徴量変換 →](chapter-4.html)
