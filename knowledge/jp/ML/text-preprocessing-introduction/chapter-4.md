---
title: "Chapter 4: ベクトル化"
chapter_number: 4
series: テキスト前処理入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [ベクトル化, Bag of Words, TF-IDF, Word Embeddings]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 4: ベクトル化

## 1. Bag of Words（BoW）

単語の出現回数をカウント。

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]

# CountVectorizerの初期化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# 特徴量名の確認
print(vectorizer.get_feature_names_out())
# ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']

# ベクトル表現
print(X.toarray())
# [[0 1 1 1 0 0 1 0 1]
#  [0 2 0 1 0 1 1 0 1]
#  [1 0 0 1 1 0 1 1 1]]

# DataFrameに変換
import pandas as pd
df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_bow)
```

### パラメータ調整
```python
vectorizer = CountVectorizer(
    max_features=100,        # 上位100単語のみ
    min_df=2,                # 2文書以上に出現
    max_df=0.8,              # 80%以下の文書に出現
    ngram_range=(1, 2),      # unigram + bigram
    stop_words='english'     # ストップワード除去
)
X = vectorizer.fit_transform(corpus)
```

## 2. TF-IDF（Term Frequency-Inverse Document Frequency）

単語の重要度を計算。

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDFベクトル化
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

print(X_tfidf.toarray())

# DataFrameに変換
df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)
print(df_tfidf)
```

### BoW vs TF-IDF比較
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# BoW
axes[0].imshow(X.toarray(), cmap='Blues', aspect='auto')
axes[0].set_title('Bag of Words')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Documents')

# TF-IDF
axes[1].imshow(X_tfidf.toarray(), cmap='Blues', aspect='auto')
axes[1].set_title('TF-IDF')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Documents')

plt.tight_layout()
plt.show()
```

## 3. Word Embeddings基礎

単語を密なベクトルで表現。

### Word2Vec（概念）
```python
# Gensimを使用した例（概念）
# pip install gensim
from gensim.models import Word2Vec

# サンプルデータ
sentences = [
    ['this', 'is', 'first', 'sentence'],
    ['this', 'is', 'second', 'sentence'],
    ['yet', 'another', 'sentence']
]

# Word2Vecモデルの学習
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 単語ベクトルの取得
vector = model.wv['sentence']
print(f"Vector shape: {vector.shape}")

# 類似単語の検索
similar_words = model.wv.most_similar('sentence', topn=3)
print(similar_words)
```

### 事前学習済みEmbeddings
```python
# GloVeの使用例（概念）
# wget http://nlp.stanford.edu/data/glove.6B.zip
import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# embeddings = load_glove_embeddings('glove.6B.100d.txt')
# vector = embeddings.get('word', None)
```

## 4. 実践的な使用例

### テキスト分類パイプライン
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# データ準備
texts = ["positive example", "negative example", ...]
labels = [1, 0, ...]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# パイプライン構築
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', LogisticRegression())
])

# 学習と評価
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# 予測
new_text = ["this is a new example"]
prediction = pipeline.predict(new_text)
```

### 特徴量の重要度分析
```python
# TF-IDFの重要な単語を抽出
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = X_tfidf.toarray()[0]  # 1文書目

# スコアの高い順にソート
top_indices = tfidf_scores.argsort()[-10:][::-1]
top_features = [(feature_names[i], tfidf_scores[i]) for i in top_indices]

print("Top 10 important words:")
for word, score in top_features:
    print(f"{word:15} {score:.4f}")
```

**本シリーズ完了！**

---

**目次へ**: [↑ シリーズ目次](index.html)
