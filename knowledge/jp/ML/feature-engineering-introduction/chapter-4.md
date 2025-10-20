---
title: "Chapter 4: ドメイン知識の活用"
chapter_number: 4
series: 特徴量エンジニアリング入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [ドメイン知識, 時系列, テキスト特徴量, 画像特徴量]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 4: ドメイン知識の活用

## 1. 時系列特徴量

### ラグ特徴量
```python
import pandas as pd

# 過去のデータをラグ特徴量として追加
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_7'] = df['sales'].shift(7)
df['sales_lag_30'] = df['sales'].shift(30)

# 移動平均
df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
df['sales_ma_30'] = df['sales'].rolling(window=30).mean()

# 移動標準偏差
df['sales_std_7'] = df['sales'].rolling(window=7).std()
```

### 変化率特徴量
```python
# 前日からの変化
df['sales_diff'] = df['sales'].diff()
df['sales_pct_change'] = df['sales'].pct_change()

# 移動平均との差分
df['sales_diff_from_ma'] = df['sales'] - df['sales_ma_7']

# トレンド特徴量
df['sales_trend'] = df['sales'].rolling(window=7).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0]
)
```

### 季節性特徴量
```python
# 同じ曜日の平均
df['dayofweek_mean'] = df.groupby('dayofweek')['sales'].transform('mean')

# 同じ月の平均
df['month_mean'] = df.groupby('month')['sales'].transform('mean')

# 前年同月比
df['yoy_sales'] = df['sales'] / df['sales'].shift(365) - 1
```

## 2. テキスト特徴量

### 基本的な統計量
```python
# 文字数・単語数
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()])
)

# 特殊文字のカウント
df['exclamation_count'] = df['text'].str.count('!')
df['question_count'] = df['text'].str.count('\?')
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)
```

### TF-IDF特徴量
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF特徴量
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(df['text'])

# DataFrameに変換
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=tfidf.get_feature_names_out()
)
```

### 感情スコア
```python
# 簡易的なポジティブ/ネガティブ判定
positive_words = ['good', 'great', 'excellent', 'love']
negative_words = ['bad', 'terrible', 'hate', 'worst']

df['positive_count'] = df['text'].apply(
    lambda x: sum(word in x.lower() for word in positive_words)
)
df['negative_count'] = df['text'].apply(
    lambda x: sum(word in x.lower() for word in negative_words)
)
df['sentiment_score'] = df['positive_count'] - df['negative_count']
```

## 3. 画像特徴量

### 基本的な統計量
```python
from PIL import Image
import numpy as np

def extract_image_features(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    features = {
        'width': img.width,
        'height': img.height,
        'aspect_ratio': img.width / img.height,
        'mean_r': img_array[:, :, 0].mean(),
        'mean_g': img_array[:, :, 1].mean(),
        'mean_b': img_array[:, :, 2].mean(),
        'std_r': img_array[:, :, 0].std(),
        'brightness': img_array.mean()
    }
    return features

df['image_features'] = df['image_path'].apply(extract_image_features)
```

### 事前学習モデルの特徴量
```python
# 例: ResNetからの特徴量抽出（概念）
# from torchvision.models import resnet50
# model = resnet50(pretrained=True)
# features = model(image_tensor)
```

## 4. 実践的なパイプライン

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# カスタム特徴量変換関数
def create_interaction_features(X):
    X_new = X.copy()
    X_new['interaction'] = X[:, 0] * X[:, 1]
    return X_new

# パイプラインの構築
feature_pipeline = Pipeline([
    ('union', FeatureUnion([
        ('numeric', Pipeline([
            ('scaler', StandardScaler())
        ])),
        ('interactions', FunctionTransformer(create_interaction_features))
    ])),
    ('pca', PCA(n_components=10))
])

X_transformed = feature_pipeline.fit_transform(X)
```

## 5. 特徴量エンジニアリングのベストプラクティス

1. **ドメイン知識を活用**: 業界特有のパターンを理解
2. **シンプルから始める**: 基本的な特徴量から試す
3. **検証**: 交差検証で特徴量の有効性を確認
4. **過学習に注意**: 訓練データのみから作成しない
5. **自動化**: パイプライン化して再現性を確保

```python
# 良い例: 訓練データで学習、テストデータに適用
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # fit_transformではない

# 悪い例: データリークの危険
# scaler.fit(X_all)  # テストデータも含めて学習
```

**本シリーズ完了！**

---

**目次へ**: [↑ シリーズ目次](index.html)
