---
title: "Chapter 2: トークン化と正規化"
chapter_number: 2
series: テキスト前処理入門
difficulty: 初級〜中級
reading_time: 25-30分
tags: [トークン化, ステミング, レンマ化, NLTK]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 2: トークン化と正規化

## 1. トークン化（Tokenization）

### 単語トークン化
```python
# 基本的な分割
text = "This is a sample sentence."
tokens = text.split()
print(tokens)  # ['This', 'is', 'a', 'sample', 'sentence.']

# NLTKによるトークン化
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)  # ['This', 'is', 'a', 'sample', 'sentence', '.']
```

### 文トークン化
```python
from nltk.tokenize import sent_tokenize

text = "This is the first sentence. This is the second one."
sentences = sent_tokenize(text)
print(sentences)
# ['This is the first sentence.', 'This is the second one.']
```

### 正規表現トークナイザー
```python
from nltk.tokenize import RegexpTokenizer

# アルファベットのみ抽出
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
```

## 2. ステミング（Stemming）

語尾を削除して語幹を抽出。

```python
from nltk.stem import PorterStemmer, SnowballStemmer

# Porter Stemmer
porter = PorterStemmer()
words = ["running", "runs", "ran", "runner"]
stemmed = [porter.stem(word) for word in words]
print(stemmed)  # ['run', 'run', 'ran', 'runner']

# Snowball Stemmer（多言語対応）
snowball = SnowballStemmer('english')
stemmed = [snowball.stem(word) for word in words]
```

## 3. レンマ化（Lemmatization）

文法的に正しい基本形に変換。

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "better", "worse"]

# 動詞として処理
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(lemmatized)  # ['run', 'run', 'run', 'better', 'worse']

# 形容詞として処理
adjectives = ["better", "worse"]
lemmatized_adj = [lemmatizer.lemmatize(word, pos='a') for word in adjectives]
print(lemmatized_adj)  # ['good', 'bad']
```

## 4. 品詞タグ付け

```python
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
print(tagged)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ...]

# 品詞に基づいたレンマ化
def get_wordnet_pos(tag):
    """品詞タグをWordNet形式に変換"""
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default: noun

lemmatized = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    for word, tag in tagged
]
```

## 5. spaCyによる高度な処理

```python
# インストール: pip install spacy
# python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The quick brown fox jumps over the lazy dog")

# トークン化、レンマ化、品詞タグ付けを一度に
for token in doc:
    print(f"{token.text:10} {token.lemma_:10} {token.pos_:5}")

# 出力例:
# The        the        DET
# quick      quick      ADJ
# brown      brown      ADJ
# fox        fox        NOUN
# jumps      jump       VERB
```

---

**次へ**: [Chapter 3: ストップワードと特殊処理 →](chapter-3.html)
