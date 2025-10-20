---
title: "Chapter 3: ストップワードと特殊処理"
chapter_number: 3
series: テキスト前処理入門
difficulty: 初級
reading_time: 20-25分
tags: [ストップワード, N-gram, 固有表現抽出]
ai_generated: true
ai_model: Claude 3.5 Sonnet
generation_date: 2025-10-20
version: 1.0
---

# Chapter 3: ストップワードと特殊処理

## 1. ストップワード除去

頻出するが意味のない単語（a, the, is など）を削除。

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

# 英語のストップワード
stop_words = set(stopwords.words('english'))
print(f"ストップワード数: {len(stop_words)}")

text = "This is a sample sentence showing off stop word filtration"
tokens = word_tokenize(text.lower())

# ストップワード除去
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)  # ['sample', 'sentence', 'showing', 'stop', 'word', 'filtration']
```

### カスタムストップワード
```python
# カスタムストップワードの追加
custom_stop_words = stop_words.union({'sample', 'showing'})

filtered_tokens = [word for word in tokens if word not in custom_stop_words]
```

## 2. N-gramの生成

連続するN個の単語の組み合わせ。

```python
from nltk import ngrams

text = "natural language processing is fun"
tokens = word_tokenize(text)

# Bigram（2-gram）
bigrams = list(ngrams(tokens, 2))
print(bigrams)
# [('natural', 'language'), ('language', 'processing'), ('processing', 'is'), ('is', 'fun')]

# Trigram（3-gram）
trigrams = list(ngrams(tokens, 3))
print(trigrams)
# [('natural', 'language', 'processing'), ('language', 'processing', 'is'), ...]

# scikit-learnで実装
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2))  # unigram + bigram
X = vectorizer.fit_transform([text])
print(vectorizer.get_feature_names_out())
```

## 3. 固有表現抽出（NER）

人名、地名、組織名などを抽出。

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
doc = nlp(text)

# 固有表現の抽出
for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_:10} {spacy.explain(ent.label_)}")

# 出力例:
# Apple Inc.           ORG        Companies, agencies, institutions
# Steve Jobs           PERSON     People, including fictional
# Cupertino            GPE        Countries, cities, states
# California           GPE        Countries, cities, states
```

## 4. 辞書ベースの処理

### 短縮形の展開
```python
# 短縮形の辞書
contractions_dict = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "i'm": "i am"
}

def expand_contractions(text, contractions_dict):
    pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                        flags=re.IGNORECASE|re.DOTALL)

    def replace(match):
        return contractions_dict[match.group(0).lower()]

    return pattern.sub(replace, text)

text = "I don't think it's gonna work"
expanded = expand_contractions(text, contractions_dict)
print(expanded)  # "I do not think it is gonna work"
```

### スペルチェック
```python
# pip install textblob
from textblob import TextBlob

text = "I havv a speling mistke"
corrected = TextBlob(text).correct()
print(corrected)  # "I have a spelling mistake"
```

## 5. 完全な前処理パイプライン

```python
def preprocess_text(text, remove_stopwords=True):
    """包括的なテキスト前処理"""
    # 小文字化
    text = text.lower()

    # URL、メンション削除
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)

    # HTMLタグ削除
    text = re.sub(r'<.*?>', '', text)

    # トークン化
    tokens = word_tokenize(text)

    # 句読点除去
    tokens = [word for word in tokens if word.isalnum()]

    # ストップワード除去
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # レンマ化
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# 使用例
text = "The quick brown foxes are jumping over the lazy dogs!"
processed = preprocess_text(text)
print(processed)  # ['quick', 'brown', 'fox', 'jumping', 'lazy', 'dog']
```

---

**次へ**: [Chapter 4: ベクトル化 →](chapter-4.html)
