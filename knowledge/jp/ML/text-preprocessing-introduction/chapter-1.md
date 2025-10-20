---
title: "Chapter 1: テキストクリーニング"
chapter_number: 1
series: テキスト前処理入門
difficulty: 初級
reading_time: 20-25分
tags: [テキストクリーニング, 正規表現, ノイズ除去]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 1: テキストクリーニング

## 1. 基本的なクリーニング

```python
import re
import string

text = "  Hello, World!  This is a TEST.  "

# 前後の空白削除
text = text.strip()

# 小文字化
text = text.lower()

# 句読点の削除
text = text.translate(str.maketrans('', '', string.punctuation))

print(text)  # "hello world this is a test"
```

## 2. 正規表現によるクリーニング

```python
# URLの削除
text = re.sub(r'http\S+|www\S+', '', text)

# メールアドレスの削除
text = re.sub(r'\S+@\S+', '', text)

# メンション削除（@username）
text = re.sub(r'@\w+', '', text)

# ハッシュタグ削除
text = re.sub(r'#\w+', '', text)

# 数字の削除
text = re.sub(r'\d+', '', text)

# 複数の空白を1つに
text = re.sub(r'\s+', ' ', text)

# HTMLタグの削除
text = re.sub(r'<.*?>', '', text)
```

## 3. 特殊文字の処理

```python
# 絵文字の削除
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

text = remove_emoji(text)

# 改行・タブの処理
text = text.replace('\n', ' ').replace('\t', ' ')

# Unicode正規化
import unicodedata
text = unicodedata.normalize('NFKC', text)
```

## 4. パイプライン化

```python
import pandas as pd

def clean_text(text):
    """テキストクリーニングのパイプライン"""
    # 小文字化
    text = text.lower()

    # URL削除
    text = re.sub(r'http\S+|www\S+', '', text)

    # メンション・ハッシュタグ削除
    text = re.sub(r'[@#]\w+', '', text)

    # HTMLタグ削除
    text = re.sub(r'<.*?>', '', text)

    # 句読点削除
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 数字削除
    text = re.sub(r'\d+', '', text)

    # 余分な空白削除
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# データフレームに適用
df['cleaned_text'] = df['text'].apply(clean_text)
```

---

**次へ**: [Chapter 2: トークン化と正規化 →](chapter-2.html)
