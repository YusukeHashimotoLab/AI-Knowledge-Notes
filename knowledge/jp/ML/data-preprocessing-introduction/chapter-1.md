---
title: "Chapter 1: データクリーニングの基礎"
chapter_number: 1
series: データ前処理入門
difficulty: 初級
reading_time: 20-25分
tags: [データクリーニング, 重複処理, データ型]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 1: データクリーニングの基礎

## 1. データ品質の確認

```python
import pandas as pd
import numpy as np

# データの読み込み
df = pd.read_csv('data.csv')

# 基本情報の確認
print(df.info())
print(df.describe())

# データ型の確認
print(df.dtypes)

# ユニーク値の確認
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

## 2. 重複データの処理

```python
# 重複行の確認
print(f"重複行数: {df.duplicated().sum()}")

# 重複行の詳細確認
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

# 重複行の削除
df_clean = df.drop_duplicates()

# 特定列での重複削除
df_clean = df.drop_duplicates(subset=['user_id', 'date'], keep='last')
```

## 3. データ型の統一

```python
# 日付型への変換
df['date'] = pd.to_datetime(df['date'])

# カテゴリ型への変換
df['category'] = df['category'].astype('category')

# 数値型への変換
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 文字列の標準化
df['name'] = df['name'].str.strip().str.lower()
```

---

**次へ**: [Chapter 2: 欠損値処理 →](chapter-2.html)
