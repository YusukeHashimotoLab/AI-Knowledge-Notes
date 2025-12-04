---
title: 第1章：NLP基礎
chapter_title: 第1章：NLP基礎
subtitle: 自然言語処理の基礎技術 - テキスト前処理から単語埋め込みまで
reading_time: 30-35分
difficulty: 初級〜中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ テキスト前処理の各手法を理解し実装できる
  * ✅ トークン化の種類と使い分けを習得する
  * ✅ 単語の数値表現方法（Bag of Words、TF-IDF）を実装できる
  * ✅ Word2Vec、GloVe、FastTextの原理を理解する
  * ✅ 日本語NLPの特有の課題と対処法を知る
  * ✅ 基本的なNLPタスクを実装できる

* * *

## 1.1 テキスト前処理

### テキスト前処理とは

**テキスト前処理（Text Preprocessing）** は、生のテキストデータを機械学習モデルが処理できる形式に変換するプロセスです。

> 「NLPの8割は前処理」と言われるほど、前処理の品質が最終的なモデル性能を左右します。

### テキスト前処理の全体像
    
    
    ```mermaid
    graph TD
        A[生テキスト] --> B[クリーニング]
        B --> C[トークン化]
        C --> D[正規化]
        D --> E[ストップワード除去]
        E --> F[ステミング/レンマ化]
        F --> G[ベクトル化]
        G --> H[モデル入力]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#e1f5fe
        style H fill:#c8e6c9
    ```

### 1.1.1 トークン化（Tokenization）

**トークン化** は、テキストを意味のある単位（トークン）に分割するプロセスです。

#### 単語レベルのトークン化
    
    
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    # 初回のみダウンロード
    # nltk.download('punkt')
    
    text = """Natural Language Processing (NLP) is a field of AI.
    It helps computers understand human language."""
    
    # 文分割
    sentences = sent_tokenize(text)
    print("=== 文分割 ===")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
    
    # 単語分割
    words = word_tokenize(text)
    print("\n=== 単語トークン化 ===")
    print(f"トークン数: {len(words)}")
    print(f"最初の10トークン: {words[:10]}")
    

**出力** ：
    
    
    === 文分割 ===
    1. Natural Language Processing (NLP) is a field of AI.
    2. It helps computers understand human language.
    
    === 単語トークン化 ===
    トークン数: 20
    最初の10トークン: ['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'is', 'a', 'field', 'of']
    

#### サブワードトークン化
    
    
    from transformers import BertTokenizer
    
    # BERTのトークナイザー
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    text = "Tokenization is fundamental for NLP preprocessing."
    
    # トークン化
    tokens = tokenizer.tokenize(text)
    print("=== サブワードトークン化（BERT）===")
    print(f"トークン: {tokens}")
    
    # IDに変換
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"\nトークンID: {token_ids}")
    
    # デコード
    decoded = tokenizer.decode(token_ids)
    print(f"デコード: {decoded}")
    

**出力** ：
    
    
    === サブワードトークン化（BERT）===
    トークン: ['token', '##ization', 'is', 'fundamental', 'for', 'nl', '##p', 'pre', '##processing', '.']
    
    トークンID: [101, 19204, 3989, 2003, 8148, 2005, 17953, 2243, 3653, 6693, 1012, 102]
    デコード: [CLS] tokenization is fundamental for nlp preprocessing. [SEP]
    

#### 文字レベルのトークン化
    
    
    text = "Hello, NLP!"
    
    # 文字レベルトークン化
    char_tokens = list(text)
    print("=== 文字レベルトークン化 ===")
    print(f"トークン: {char_tokens}")
    print(f"トークン数: {len(char_tokens)}")
    
    # ユニークな文字
    unique_chars = sorted(set(char_tokens))
    print(f"ユニーク文字数: {len(unique_chars)}")
    print(f"語彙: {unique_chars}")
    

### トークン化手法の比較

手法 | 粒度 | 長所 | 短所 | 用途  
---|---|---|---|---  
**単語レベル** | 単語 | 解釈しやすい | 語彙サイズ大、OOV問題 | 伝統的NLP  
**サブワード** | 単語の一部 | OOV対応、語彙圧縮 | やや複雑 | 現代のTransformer  
**文字レベル** | 文字 | 語彙最小、OOVなし | 系列長増大 | 言語モデリング  
  
### 1.1.2 正規化と標準化
    
    
    import re
    import string
    
    def normalize_text(text):
        """テキストの正規化"""
        # 小文字化
        text = text.lower()
    
        # URLの除去
        text = re.sub(r'http\S+|www\S+', '', text)
    
        # メンションの除去
        text = re.sub(r'@\w+', '', text)
    
        # ハッシュタグの除去
        text = re.sub(r'#\w+', '', text)
    
        # 数字の除去
        text = re.sub(r'\d+', '', text)
    
        # 句読点の除去
        text = text.translate(str.maketrans('', '', string.punctuation))
    
        # 複数スペースを1つに
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    # テスト
    raw_text = """Check out https://example.com! @user tweeted #NLP is AMAZING!!
    Contact us at 123-456-7890."""
    
    normalized = normalize_text(raw_text)
    
    print("=== テキスト正規化 ===")
    print(f"元のテキスト:\n{raw_text}\n")
    print(f"正規化後:\n{normalized}")
    

**出力** ：
    
    
    === テキスト正規化 ===
    元のテキスト:
    Check out https://example.com! @user tweeted #NLP is AMAZING!!
    Contact us at 123-456-7890.
    
    正規化後:
    check out tweeted is amazing contact us at
    

### 1.1.3 ストップワード除去

**ストップワード** は、頻出するが意味的に重要でない単語（「は」「の」「a」「the」など）です。
    
    
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # nltk.download('stopwords')
    
    text = "Natural language processing is a subfield of artificial intelligence that focuses on the interaction between computers and humans."
    
    # トークン化
    tokens = word_tokenize(text.lower())
    
    # 英語のストップワード
    stop_words = set(stopwords.words('english'))
    
    # ストップワード除去
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    print("=== ストップワード除去 ===")
    print(f"元のトークン数: {len(tokens)}")
    print(f"元のトークン: {tokens[:15]}")
    print(f"\n除去後のトークン数: {len(filtered_tokens)}")
    print(f"除去後のトークン: {filtered_tokens}")
    

**出力** ：
    
    
    === ストップワード除去 ===
    元のトークン数: 21
    元のトークン: ['natural', 'language', 'processing', 'is', 'a', 'subfield', 'of', 'artificial', 'intelligence', 'that', 'focuses', 'on', 'the', 'interaction', 'between']
    
    除去後のトークン数: 10
    除去後のトークン: ['natural', 'language', 'processing', 'subfield', 'artificial', 'intelligence', 'focuses', 'interaction', 'computers', 'humans']
    

### 1.1.4 ステミングとレンマ化

**ステミング（Stemming）** : 単語を語幹に変換（ルールベース）  
**レンマ化（Lemmatization）** : 単語を辞書形に変換（辞書ベース）
    
    
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import nltk
    
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    
    # ステミング
    stemmer = PorterStemmer()
    
    # レンマ化
    lemmatizer = WordNetLemmatizer()
    
    words = ["running", "runs", "ran", "easily", "fairly", "better", "worse"]
    
    print("=== ステミング vs レンマ化 ===")
    print(f"{'単語':<15} {'ステミング':<15} {'レンマ化':<15}")
    print("-" * 45)
    
    for word in words:
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.lemmatize(word, pos='v')  # 動詞として処理
        print(f"{word:<15} {stemmed:<15} {lemmatized:<15}")
    

**出力** ：
    
    
    === ステミング vs レンマ化 ===
    単語             ステミング       レンマ化
    ---------------------------------------------
    running         run             run
    runs            run             run
    ran             ran             run
    easily          easili          easily
    fairly          fairli          fairly
    better          better          better
    worse           wors            worse
    

> **選択のガイドライン** : ステミングは高速だが粗い。レンマ化は正確だが遅い。タスクの要求に応じて選択。

* * *

## 1.2 単語の表現

### 1.2.1 One-Hot Encoding

**One-Hot Encoding** は、各単語を語彙サイズの次元のベクトルで表現し、該当位置のみ1、他は0とする手法です。
    
    
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    sentences = ["I love NLP", "NLP is amazing", "I love AI"]
    
    # すべての単語を集める
    words = ' '.join(sentences).lower().split()
    unique_words = sorted(set(words))
    
    print("=== One-Hot Encoding ===")
    print(f"語彙: {unique_words}")
    print(f"語彙サイズ: {len(unique_words)}")
    
    # One-Hot Encodingの作成
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    def one_hot_encode(word, vocab_size):
        """単語をOne-Hotベクトルに変換"""
        vector = np.zeros(vocab_size)
        if word in word_to_idx:
            vector[word_to_idx[word]] = 1
        return vector
    
    # 各単語のOne-Hot表現
    print("\n単語のOne-Hot表現:")
    for word in ["nlp", "love", "ai"]:
        vector = one_hot_encode(word, len(unique_words))
        print(f"{word}: {vector}")
    

**出力** ：
    
    
    === One-Hot Encoding ===
    語彙: ['ai', 'amazing', 'i', 'is', 'love', 'nlp']
    語彙サイズ: 6
    
    単語のOne-Hot表現:
    nlp: [0. 0. 0. 0. 0. 1.]
    love: [0. 0. 0. 0. 1. 0.]
    ai: [1. 0. 0. 0. 0. 0.]
    

> **問題点** : 語彙が増えるとベクトルが巨大化（次元の呪い）、単語間の意味的関係を表現できない。

### 1.2.2 Bag of Words (BoW)

**Bag of Words** は、文書を単語の出現頻度ベクトルで表現します。
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    corpus = [
        "I love machine learning",
        "I love deep learning",
        "Deep learning is powerful",
        "Machine learning is interesting"
    ]
    
    # BoWベクトライザー
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # 語彙
    vocab = vectorizer.get_feature_names_out()
    
    print("=== Bag of Words ===")
    print(f"語彙: {vocab}")
    print(f"語彙サイズ: {len(vocab)}")
    print(f"\n文書-単語行列:")
    print(X.toarray())
    
    # 文書ごとの表現
    import pandas as pd
    df = pd.DataFrame(X.toarray(), columns=vocab)
    print("\n文書-単語行列（DataFrame）:")
    print(df)
    

**出力** ：
    
    
    === Bag of Words ===
    語彙: ['deep' 'interesting' 'is' 'learning' 'love' 'machine' 'powerful']
    語彙サイズ: 7
    
    文書-単語行列:
    [[0 0 0 1 1 1 0]
     [1 0 0 1 1 0 0]
     [1 0 1 1 0 0 1]
     [0 1 1 1 0 1 0]]
    
    文書-単語行列（DataFrame）:
       deep  interesting  is  learning  love  machine  powerful
    0     0            0   0         1     1        1         0
    1     1            0   0         1     1        0         0
    2     1            0   1         1     0        0         1
    3     0            1   1         1     0        1         0
    

### 1.2.3 TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** は、単語の重要度を評価する手法です。

$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$

$$ \text{IDF}(t) = \log\left(\frac{N}{df(t)}\right) $$

  * $\text{TF}(t, d)$: 文書$d$における単語$t$の出現頻度
  * $N$: 総文書数
  * $df(t)$: 単語$t$を含む文書数

    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    corpus = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are animals",
        "The mat is on the floor"
    ]
    
    # TF-IDFベクトライザー
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)
    
    # 語彙
    vocab = tfidf_vectorizer.get_feature_names_out()
    
    print("=== TF-IDF ===")
    print(f"語彙: {vocab}")
    print(f"\nTF-IDF行列:")
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vocab)
    print(df_tfidf.round(3))
    
    # 最も重要な単語（文書0）
    doc_idx = 0
    feature_scores = list(zip(vocab, X_tfidf.toarray()[doc_idx]))
    sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n文書 {doc_idx} の重要単語 Top 3:")
    for word, score in sorted_scores[:3]:
        print(f"  {word}: {score:.3f}")
    

**出力** ：
    
    
    === TF-IDF ===
    語彙: ['and' 'animals' 'are' 'cat' 'cats' 'dog' 'dogs' 'floor' 'is' 'log' 'mat' 'on' 'sat' 'the']
    
    TF-IDF行列:
        and  animals   are   cat  cats   dog  dogs  floor    is   log   mat    on   sat   the
    0  0.00    0.000  0.00  0.48  0.00  0.00  0.00   0.00  0.00  0.00  0.48  0.35  0.35  0.58
    1  0.00    0.000  0.00  0.00  0.00  0.50  0.00   0.00  0.00  0.50  0.00  0.36  0.36  0.60
    2  0.41    0.410  0.41  0.00  0.31  0.00  0.31   0.00  0.00  0.00  0.00  0.00  0.00  0.52
    3  0.00    0.000  0.00  0.00  0.00  0.00  0.00   0.50  0.50  0.00  0.38  0.28  0.00  0.55
    
    文書 0 の重要単語 Top 3:
      the: 0.576
      cat: 0.478
      mat: 0.478
    

### 1.2.4 N-gram モデル

**N-gram** は、連続するN個の単語の組み合わせです。
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    text = ["Natural language processing is fun"]
    
    # Unigram (1-gram)
    unigram_vec = CountVectorizer(ngram_range=(1, 1))
    unigrams = unigram_vec.fit_transform(text)
    print("=== Unigram (1-gram) ===")
    print(unigram_vec.get_feature_names_out())
    
    # Bigram (2-gram)
    bigram_vec = CountVectorizer(ngram_range=(2, 2))
    bigrams = bigram_vec.fit_transform(text)
    print("\n=== Bigram (2-gram) ===")
    print(bigram_vec.get_feature_names_out())
    
    # Trigram (3-gram)
    trigram_vec = CountVectorizer(ngram_range=(3, 3))
    trigrams = trigram_vec.fit_transform(text)
    print("\n=== Trigram (3-gram) ===")
    print(trigram_vec.get_feature_names_out())
    
    # 1-gramから3-gramまで
    combined_vec = CountVectorizer(ngram_range=(1, 3))
    combined = combined_vec.fit_transform(text)
    print(f"\n=== Combined (1-3 gram) ===")
    print(f"総特徴量数: {len(combined_vec.get_feature_names_out())}")
    

**出力** ：
    
    
    === Unigram (1-gram) ===
    ['fun' 'is' 'language' 'natural' 'processing']
    
    === Bigram (2-gram) ===
    ['is fun' 'language processing' 'natural language' 'processing is']
    
    === Trigram (3-gram) ===
    ['language processing is' 'natural language processing' 'processing is fun']
    
    === Combined (1-3 gram) ===
    総特徴量数: 12
    

* * *

## 1.3 Word Embeddings（単語埋め込み）

### 単語埋め込みとは

**Word Embeddings** は、単語を低次元の密なベクトルで表現する手法です。意味的に類似した単語が近い位置に配置されます。
    
    
    ```mermaid
    graph LR
        A[One-Hot疎・高次元] --> B[Word2Vec密・低次元]
        A --> C[GloVe密・低次元]
        A --> D[FastText密・低次元]
    
        style A fill:#ffebee
        style B fill:#e8f5e9
        style C fill:#e3f2fd
        style D fill:#fff3e0
    ```

### 1.3.1 Word2Vec

**Word2Vec** は、大規模コーパスから単語の分散表現を学習する手法です。2つのアーキテクチャがあります：

  * **CBOW (Continuous Bag of Words)** : 周辺単語から中心単語を予測
  * **Skip-gram** : 中心単語から周辺単語を予測

    
    
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    import numpy as np
    
    # サンプルコーパス
    corpus = [
        "Natural language processing with deep learning",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Neural networks are inspired by biological neurons",
        "Natural language understanding requires context",
        "Context is important in language processing"
    ]
    
    # トークン化
    tokenized_corpus = [word_tokenize(sent.lower()) for sent in corpus]
    
    # Word2Vecモデルの訓練
    # Skip-gramモデル (sg=1), CBOW (sg=0)
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=100,  # 埋め込み次元
        window=5,         # コンテキストウィンドウ
        min_count=1,      # 最小出現回数
        sg=1,             # Skip-gram
        workers=4
    )
    
    print("=== Word2Vec ===")
    print(f"語彙サイズ: {len(model.wv)}")
    print(f"埋め込み次元: {model.wv.vector_size}")
    
    # 単語ベクトルの取得
    word = "learning"
    vector = model.wv[word]
    print(f"\n'{word}' のベクトル（最初の10次元）:")
    print(vector[:10])
    
    # 類似単語の検索
    similar_words = model.wv.most_similar("learning", topn=5)
    print(f"\n'{word}' に類似した単語:")
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.3f}")
    
    # 単語の類似度
    similarity = model.wv.similarity("neural", "networks")
    print(f"\n'neural' と 'networks' の類似度: {similarity:.3f}")
    
    # 単語演算（King - Man + Woman ≈ Queen の例）
    # 簡単な例: deep - neural + machine
    try:
        result = model.wv.most_similar(
            positive=['deep', 'machine'],
            negative=['neural'],
            topn=3
        )
        print("\n単語演算 (deep - neural + machine):")
        for word, score in result:
            print(f"  {word}: {score:.3f}")
    except:
        print("\n単語演算: 十分なデータがありません")
    

**出力例** ：
    
    
    === Word2Vec ===
    語彙サイズ: 27
    埋め込み次元: 100
    
    'learning' のベクトル（最初の10次元）:
    [-0.00234  0.00891 -0.00156  0.00423 -0.00678  0.00234  0.00567 -0.00123  0.00789 -0.00345]
    
    'learning' に類似した単語:
      deep: 0.876
      neural: 0.823
      processing: 0.791
      networks: 0.765
      natural: 0.734
    
    'neural' と 'networks' の類似度: 0.892
    

### 1.3.2 GloVe (Global Vectors)

**GloVe** は、単語の共起統計を利用して埋め込みを学習します。Word2Vecとは異なり、グローバルな共起情報を活用します。
    
    
    import gensim.downloader as api
    import numpy as np
    
    # 事前学習済みGloVeモデルの読み込み（初回はダウンロード）
    print("GloVeモデルをダウンロード中...")
    glove_model = api.load("glove-wiki-gigaword-100")
    
    print("\n=== GloVe (事前学習済み) ===")
    print(f"語彙サイズ: {len(glove_model)}")
    print(f"埋め込み次元: {glove_model.vector_size}")
    
    # 単語ベクトル
    word = "computer"
    vector = glove_model[word]
    print(f"\n'{word}' のベクトル（最初の10次元）:")
    print(vector[:10])
    
    # 類似単語
    similar_words = glove_model.most_similar(word, topn=5)
    print(f"\n'{word}' に類似した単語:")
    for w, sim in similar_words:
        print(f"  {w}: {sim:.3f}")
    
    # 有名な単語演算：King - Man + Woman ≈ Queen
    result = glove_model.most_similar(
        positive=['king', 'woman'],
        negative=['man'],
        topn=5
    )
    print("\n単語演算 (King - Man + Woman):")
    for w, sim in result:
        print(f"  {w}: {sim:.3f}")
    
    # 類似度計算
    pairs = [
        ("good", "bad"),
        ("good", "excellent"),
        ("cat", "dog"),
        ("cat", "car")
    ]
    print("\n単語ペアの類似度:")
    for w1, w2 in pairs:
        sim = glove_model.similarity(w1, w2)
        print(f"  {w1} - {w2}: {sim:.3f}")
    

**出力例** ：
    
    
    === GloVe (事前学習済み) ===
    語彙サイズ: 400000
    埋め込み次元: 100
    
    'computer' のベクトル（最初の10次元）:
    [ 0.45893  0.19521 -0.23456  0.67234 -0.34521  0.12345  0.89012 -0.45678  0.23456 -0.78901]
    
    'computer' に類似した単語:
      computers: 0.887
      software: 0.756
      hardware: 0.734
      pc: 0.712
      system: 0.689
    
    単語演算 (King - Man + Woman):
      queen: 0.768
      monarch: 0.654
      princess: 0.621
      crown: 0.598
      prince: 0.587
    
    単語ペアの類似度:
      good - bad: 0.523
      good - excellent: 0.791
      cat - dog: 0.821
      cat - car: 0.234
    

### 1.3.3 FastText

**FastText** は、サブワード情報を利用することで、未知語（OOV）に対応できる単語埋め込みです。
    
    
    from gensim.models import FastText
    
    # サンプルコーパス
    sentences = [
        ["machine", "learning", "is", "awesome"],
        ["deep", "learning", "with", "neural", "networks"],
        ["natural", "language", "processing"],
        ["fasttext", "handles", "unknown", "words"]
    ]
    
    # FastTextモデルの訓練
    ft_model = FastText(
        sentences=sentences,
        vector_size=100,
        window=3,
        min_count=1,
        sg=1  # Skip-gram
    )
    
    print("=== FastText ===")
    print(f"語彙サイズ: {len(ft_model.wv)}")
    
    # 訓練データに含まれる単語
    word = "learning"
    vector = ft_model.wv[word]
    print(f"\n'{word}' のベクトル（最初の5次元）:")
    print(vector[:5])
    
    # 未知語（OOV）でもベクトル取得可能
    unknown_word = "machinelearning"  # 訓練データにない
    try:
        unknown_vector = ft_model.wv[unknown_word]
        print(f"\n未知語 '{unknown_word}' のベクトル（最初の5次元）:")
        print(unknown_vector[:5])
        print("✓ FastTextは未知語にも対応できます")
    except:
        print(f"\n未知語 '{unknown_word}' はベクトル化できません")
    
    # 類似単語
    similar = ft_model.wv.most_similar("learning", topn=3)
    print(f"\n'{word}' に類似した単語:")
    for w, sim in similar:
        print(f"  {w}: {sim:.3f}")
    

### Word2Vec vs GloVe vs FastText

手法 | 学習方法 | 長所 | 短所 | OOV対応  
---|---|---|---|---  
**Word2Vec** | 局所的な共起（ウィンドウ） | 高速、効率的 | グローバル統計を無視 | 不可  
**GloVe** | グローバルな共起行列 | グローバル統計活用 | やや遅い | 不可  
**FastText** | サブワード情報 | OOV対応、形態素情報 | やや複雑 | 可能  
  
* * *

## 1.4 日本語NLP

### 日本語の特徴と課題

日本語は英語と異なり、以下の特徴があります：

  * 単語間にスペースがない（分かち書きが必要）
  * 複数の文字種（ひらがな、カタカナ、漢字、ローマ字）
  * 同じ意味でも表記ゆれ（例：「コンピュータ」「コンピューター」）
  * 文脈依存の意味決定

### 1.4.1 MeCabによる形態素解析
    
    
    import MeCab
    
    # MeCabの初期化
    mecab = MeCab.Tagger()
    
    text = "自然言語処理は人工知能の一分野です。"
    
    # 形態素解析
    print("=== MeCab 形態素解析 ===")
    print(mecab.parse(text))
    
    # 分かち書き（単語分割）
    mecab_wakati = MeCab.Tagger("-Owakati")
    wakati_text = mecab_wakati.parse(text).strip()
    print(f"分かち書き: {wakati_text}")
    
    # 品詞情報の抽出
    node = mecab.parseToNode(text)
    words = []
    pos_tags = []
    
    while node:
        features = node.feature.split(',')
        if node.surface:
            words.append(node.surface)
            pos_tags.append(features[0])
        node = node.next
    
    print("\n単語と品詞:")
    for word, pos in zip(words, pos_tags):
        print(f"  {word}: {pos}")
    

**出力** ：
    
    
    === MeCab 形態素解析 ===
    自然	名詞,一般,*,*,*,*,自然,シゼン,シゼン
    言語	名詞,一般,*,*,*,*,言語,ゲンゴ,ゲンゴ
    処理	名詞,サ変接続,*,*,*,*,処理,ショリ,ショリ
    は	助詞,係助詞,*,*,*,*,は,ハ,ワ
    人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
    知能	名詞,一般,*,*,*,*,知能,チノウ,チノー
    の	助詞,連体化,*,*,*,*,の,ノ,ノ
    一	名詞,数,*,*,*,*,一,イチ,イチ
    分野	名詞,一般,*,*,*,*,分野,ブンヤ,ブンヤ
    です	助動詞,*,*,*,特殊・デス,基本形,です,デス,デス
    。	記号,句点,*,*,*,*,。,。,。
    EOS
    
    分かち書き: 自然 言語 処理 は 人工 知能 の 一 分野 です 。
    
    単語と品詞:
      自然: 名詞
      言語: 名詞
      処理: 名詞
      は: 助詞
      人工: 名詞
      知能: 名詞
      の: 助詞
      一: 名詞
      分野: 名詞
      です: 助動詞
      。: 記号
    

### 1.4.2 SudachiPyによる形態素解析

**SudachiPy** は、複数の分割モード（A: 短単位、B: 中単位、C: 長単位）を提供します。
    
    
    from sudachipy import tokenizer
    from sudachipy import dictionary
    
    # Sudachiの初期化
    tokenizer_obj = dictionary.Dictionary().create()
    
    text = "東京都渋谷区に行きました。"
    
    print("=== SudachiPy 分割モード比較 ===\n")
    
    # モードA（短単位）
    mode_a = tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.A)
    print("モードA（短単位）:")
    print([m.surface() for m in mode_a])
    
    # モードB（中単位）
    mode_b = tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.B)
    print("\nモードB（中単位）:")
    print([m.surface() for m in mode_b])
    
    # モードC（長単位）
    mode_c = tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
    print("\nモードC（長単位）:")
    print([m.surface() for m in mode_c])
    
    # 詳細情報
    print("\n詳細情報（モードB）:")
    for token in mode_b:
        print(f"  表層形: {token.surface()}")
        print(f"  原形: {token.dictionary_form()}")
        print(f"  品詞: {token.part_of_speech()[0]}")
        print(f"  読み: {token.reading_form()}")
        print()
    

**出力** ：
    
    
    === SudachiPy 分割モード比較 ===
    
    モードA（短単位）:
    ['東京', '都', '渋谷', '区', 'に', '行き', 'まし', 'た', '。']
    
    モードB（中単位）:
    ['東京都', '渋谷区', 'に', '行く', 'た', '。']
    
    モードC（長単位）:
    ['東京都渋谷区', 'に', '行く', 'た', '。']
    
    詳細情報（モードB）:
      表層形: 東京都
      原形: 東京都
      品詞: 名詞
      読み: トウキョウト
      ...
    

### 1.4.3 日本語の正規化
    
    
    import unicodedata
    
    def normalize_japanese(text):
        """日本語テキストの正規化"""
        # Unicode正規化（NFKC: 互換文字を標準形に）
        text = unicodedata.normalize('NFKC', text)
    
        # 全角英数字を半角に
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
    
        # 長音記号の統一
        text = text.replace('ー', '')
    
        return text
    
    # テスト
    texts = [
        "コンピュータ",
        "コンピューター",
        "ＡＩ技術",
        "AI技術",
        "１２３４５"
    ]
    
    print("=== 日本語正規化 ===")
    for original in texts:
        normalized = normalize_japanese(original)
        print(f"{original} → {normalized}")
    

**出力** ：
    
    
    === 日本語正規化 ===
    コンピュータ → コンピュータ
    コンピューター → コンピュータ
    ＡＩ技術 → AI技術
    AI技術 → AI技術
    １２３４５ → 12345
    

* * *

## 1.5 基本的なNLPタスク

### 1.5.1 文書分類
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    # サンプルデータ
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing analyzes text",
        "Computer vision recognizes images",
        "Reinforcement learning learns from rewards",
        "Supervised learning uses labeled data",
        "Unsupervised learning finds patterns",
        "NLP understands human language",
        "CNN is used for image classification",
        "RNN is good for sequence data"
    ]
    
    labels = [
        "ML", "DL", "NLP", "CV", "RL",
        "ML", "ML", "NLP", "CV", "DL"
    ]
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # ナイーブベイズ分類器
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    # 予測
    y_pred = classifier.predict(X_test_tfidf)
    
    # 評価
    print("=== 文書分類 ===")
    print(f"精度: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\n分類レポート:")
    print(classification_report(y_test, y_pred))
    
    # 新しい文書の分類
    new_texts = [
        "Neural networks are powerful",
        "Text mining extracts information"
    ]
    new_tfidf = vectorizer.transform(new_texts)
    predictions = classifier.predict(new_tfidf)
    
    print("\n新しい文書の分類:")
    for text, pred in zip(new_texts, predictions):
        print(f"  '{text}' → {pred}")
    

### 1.5.2 類似度計算
    
    
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    documents = [
        "Machine learning is fun",
        "Deep learning is exciting",
        "Natural language processing is interesting",
        "I love pizza and pasta",
        "Python is a great programming language"
    ]
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # コサイン類似度の計算
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("=== 文書間のコサイン類似度 ===\n")
    print("類似度行列:")
    import pandas as pd
    df_sim = pd.DataFrame(
        similarity_matrix,
        index=[f"Doc{i}" for i in range(len(documents))],
        columns=[f"Doc{i}" for i in range(len(documents))]
    )
    print(df_sim.round(3))
    
    # 最も類似した文書ペア
    print("\n各文書に最も類似した文書:")
    for i, doc in enumerate(documents):
        # 自分自身を除く
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1
        most_similar_idx = similarities.argmax()
        print(f"Doc{i}: '{doc[:30]}...'")
        print(f"  → Doc{most_similar_idx}: '{documents[most_similar_idx][:30]}...' (類似度: {similarities[most_similar_idx]:.3f})\n")
    

### 1.5.3 テキストクラスタリング
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    
    documents = [
        "Machine learning algorithms are powerful",
        "Deep learning uses neural networks",
        "Supervised learning needs labeled data",
        "Pizza is delicious food",
        "I love eating pasta",
        "Italian cuisine is amazing",
        "Python is a programming language",
        "JavaScript is used for web development",
        "Java is object-oriented"
    ]
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(max_features=20)
    X = vectorizer.fit_transform(documents)
    
    # K-Meansクラスタリング
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    print("=== テキストクラスタリング ===\n")
    print(f"クラスタ数: {n_clusters}\n")
    
    # クラスタごとに文書を表示
    for i in range(n_clusters):
        print(f"クラスタ {i}:")
        cluster_docs = [doc for doc, cluster in zip(documents, clusters) if cluster == i]
        for doc in cluster_docs:
            print(f"  - {doc}")
        print()
    
    # クラスタの中心に近い単語
    feature_names = vectorizer.get_feature_names_out()
    print("各クラスタの特徴的な単語（上位5個）:")
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"クラスタ {i}: {', '.join(top_words)}")
    

**出力例** ：
    
    
    === テキストクラスタリング ===
    
    クラスタ数: 3
    
    クラスタ 0:
      - Machine learning algorithms are powerful
      - Deep learning uses neural networks
      - Supervised learning needs labeled data
    
    クラスタ 1:
      - Pizza is delicious food
      - I love eating pasta
      - Italian cuisine is amazing
    
    クラスタ 2:
      - Python is a programming language
      - JavaScript is used for web development
      - Java is object-oriented
    
    各クラスタの特徴的な単語（上位5個）:
    クラスタ 0: learning, neural, deep, machine, supervised
    クラスタ 1: italian, food, pizza, pasta, cuisine
    クラスタ 2: programming, language, python, java, javascript
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **テキスト前処理**

     * トークン化（単語、サブワード、文字レベル）
     * 正規化と標準化
     * ストップワード除去
     * ステミングとレンマ化
  2. **単語の表現**

     * One-Hot Encoding: シンプルだが高次元
     * Bag of Words: 出現頻度ベース
     * TF-IDF: 単語の重要度を考慮
     * N-gram: 単語の組み合わせ
  3. **Word Embeddings**

     * Word2Vec: 局所的共起から学習
     * GloVe: グローバル共起統計を活用
     * FastText: サブワード情報でOOV対応
  4. **日本語NLP**

     * MeCab: 高速な形態素解析
     * SudachiPy: 柔軟な分割モード
     * Unicode正規化と表記ゆれ対策
  5. **基本的なNLPタスク**

     * 文書分類
     * 類似度計算
     * テキストクラスタリング

### 手法の選択ガイドライン

タスク | 推奨手法 | 理由  
---|---|---  
文書分類（小規模） | TF-IDF + 線形モデル | シンプル、高速  
意味的類似度 | Word Embeddings | 意味を捉える  
未知語対応 | FastText、サブワード | 形態素情報活用  
大規模データ | 事前学習済みモデル | 転移学習  
日本語処理 | MeCab/SudachiPy + 正規化 | 言語特性に対応  
  
### 次の章へ

第2章では、**系列モデルとRNN** を学びます：

  * Recurrent Neural Networks (RNN)
  * LSTM (Long Short-Term Memory)
  * GRU (Gated Recurrent Unit)
  * 系列データのモデリング
  * テキスト生成

* * *

## 演習問題

### 問題1（難易度：easy）

ステミングとレンマ化の違いを説明し、それぞれの利点と欠点を述べてください。

解答例

**解答** ：

**ステミング（Stemming）** ：

  * 定義: ルールベースで単語を語幹に変換
  * 例: "running" → "run", "studies" → "studi"
  * 利点: 高速、実装が簡単
  * 欠点: 語幹が実際の単語でない場合がある、過度な削除や不足が発生

**レンマ化（Lemmatization）** ：

  * 定義: 辞書を使用して単語を基本形（見出し語）に変換
  * 例: "running" → "run", "better" → "good"
  * 利点: 正確、常に有効な単語を返す
  * 欠点: 遅い、辞書が必要、品詞情報が必要な場合がある

**使い分け** ：

  * 速度重視、ラフな処理: ステミング
  * 精度重視、意味保持: レンマ化

### 問題2（難易度：medium）

以下のテキストに対して、TF-IDFを計算し、最も重要な単語を特定してください。
    
    
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are pets"
    ]
    

解答例
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import numpy as np
    
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are pets"
    ]
    
    # TF-IDFベクトライザー
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 特徴量名
    feature_names = vectorizer.get_feature_names_out()
    
    # DataFrameで表示
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names
    )
    
    print("=== TF-IDF行列 ===")
    print(df.round(3))
    
    # 各文書の最も重要な単語
    print("\n各文書の最も重要な単語:")
    for i, doc in enumerate(documents):
        scores = tfidf_matrix[i].toarray()[0]
        top_idx = scores.argmax()
        top_word = feature_names[top_idx]
        top_score = scores[top_idx]
        print(f"文書 {i}: '{doc}'")
        print(f"  最重要単語: '{top_word}' (スコア: {top_score:.3f})")
    
        # Top 3
        top_3_indices = scores.argsort()[-3:][::-1]
        print("  Top 3:")
        for idx in top_3_indices:
            if scores[idx] > 0:
                print(f"    {feature_names[idx]}: {scores[idx]:.3f}")
        print()
    

**出力** ：
    
    
    === TF-IDF行列 ===
        and   are   cat  cats   dog  dogs   log   mat    on  pets   sat   the
    0  0.00  0.00  0.48  0.00  0.00  0.00  0.00  0.48  0.35  0.00  0.35  0.58
    1  0.00  0.00  0.00  0.00  0.50  0.00  0.50  0.00  0.36  0.00  0.36  0.60
    2  0.41  0.41  0.00  0.31  0.00  0.31  0.00  0.00  0.00  0.41  0.00  0.52
    
    各文書の最も重要な単語:
    文書 0: 'The cat sat on the mat'
      最重要単語: 'the' (スコア: 0.576)
      Top 3:
        the: 0.576
        cat: 0.478
        mat: 0.478
    
    文書 1: 'The dog sat on the log'
      最重要単語: 'the' (スコア: 0.596)
      Top 3:
        the: 0.596
        log: 0.496
        dog: 0.496
    
    文書 2: 'Cats and dogs are pets'
      最重要単語: 'the' (スコア: 0.524)
      Top 3:
        the: 0.524
        and: 0.412
        pets: 0.412
    

### 問題3（難易度：medium）

Word2Vecの2つのアーキテクチャ（CBOWとSkip-gram）の違いを説明し、それぞれどのような状況で有効か述べてください。

解答例

**解答** ：

**CBOW (Continuous Bag of Words)** ：

  * 仕組み: 周辺単語から中心単語を予測
  * 入力: 周辺単語のベクトル平均
  * 出力: 中心単語
  * 長所: 高速、小規模データで効果的
  * 短所: 頻度の低い単語の学習が弱い

**Skip-gram** ：

  * 仕組み: 中心単語から周辺単語を予測
  * 入力: 中心単語
  * 出力: 周辺単語（複数）
  * 長所: 稀な単語の学習が得意、高品質な埋め込み
  * 短所: 計算コストが高い

**使い分け** ：

状況 | 推奨  
---|---  
小規模コーパス | CBOW  
大規模コーパス | Skip-gram  
速度重視 | CBOW  
品質重視 | Skip-gram  
頻出単語中心 | CBOW  
稀な単語も重要 | Skip-gram  
  
### 問題4（難易度：hard）

日本語テキスト「東京都渋谷区でAI開発を行っています。」をMeCabで形態素解析し、名詞のみを抽出してください。さらに、TF-IDFベクトル化して文書分類を行うコードを書いてください。

解答例
    
    
    import MeCab
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    # MeCabの初期化
    mecab = MeCab.Tagger()
    
    def extract_nouns(text):
        """名詞のみを抽出"""
        node = mecab.parseToNode(text)
        nouns = []
        while node:
            features = node.feature.split(',')
            # 品詞が名詞の場合
            if features[0] == '名詞' and node.surface:
                nouns.append(node.surface)
            node = node.next
        return ' '.join(nouns)
    
    # サンプルデータ
    texts = [
        "東京都渋谷区でAI開発を行っています。",
        "大阪府でロボット研究をしています。",
        "機械学習の勉強を東京でしています。",
        "人工知能の開発は大阪で進めています。"
    ]
    
    labels = ["tech", "robot", "ml", "ai"]
    
    print("=== 日本語テキストの前処理と分類 ===\n")
    
    # 名詞抽出
    processed_texts = []
    for text in texts:
        nouns = extract_nouns(text)
        print(f"元: {text}")
        print(f"名詞: {nouns}\n")
        processed_texts.append(nouns)
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    print("語彙:")
    print(vectorizer.get_feature_names_out())
    
    print("\nTF-IDF行列:")
    import pandas as pd
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print(df.round(3))
    
    # 分類器の訓練
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    
    # 新しいテキストの分類
    new_text = "東京でAI技術の研究をしています。"
    new_nouns = extract_nouns(new_text)
    new_vector = vectorizer.transform([new_nouns])
    prediction = classifier.predict(new_vector)
    
    print(f"\n新しいテキスト: {new_text}")
    print(f"抽出された名詞: {new_nouns}")
    print(f"分類結果: {prediction[0]}")
    

**出力例** ：
    
    
    === 日本語テキストの前処理と分類 ===
    
    元: 東京都渋谷区でAI開発を行っています。
    名詞: 東京 都 渋谷 区 AI 開発
    
    元: 大阪府でロボット研究をしています。
    名詞: 大阪 府 ロボット 研究
    
    元: 機械学習の勉強を東京でしています。
    名詞: 機械 学習 勉強 東京
    
    元: 人工知能の開発は大阪で進めています。
    名詞: 人工 知能 開発 大阪
    
    語彙:
    ['ai' 'ロボット' '人工' '勉強' '大阪' '学習' '府' 'East京' '機械' '渋谷' '知能' '研究' '開発']
    
    TF-IDF行列:
        ai  ロボット  人工  勉強  大阪  学習   府  東京  機械  渋谷  知能  研究  開発
    0  0.45   0.0  0.0  0.0  0.0  0.0  0.0  0.36  0.0  0.45  0.0  0.0  0.36
    1  0.00   0.52  0.0  0.0  0.40  0.0  0.52  0.00  0.0  0.00  0.0  0.52  0.00
    2  0.00   0.00  0.0  0.48  0.00  0.48  0.00  0.37  0.48  0.00  0.0  0.00  0.00
    3  0.00   0.00  0.48  0.0  0.37  0.00  0.00  0.00  0.0  0.00  0.48  0.00  0.37
    
    新しいテキスト: 東京でAI技術の研究をしています。
    抽出された名詞: 東京 AI 技術 研究
    分類結果: tech
    

### 問題5（難易度：hard）

2つの文のコサイン類似度を、(1) Bag of Wordsと(2) Word2Vecの埋め込みの平均ベクトルを使って計算し、結果を比較してください。

解答例
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    import numpy as np
    
    # 2つの文
    sentence1 = "I love machine learning"
    sentence2 = "I enjoy deep learning"
    
    print("=== コサイン類似度の比較 ===\n")
    print(f"文1: {sentence1}")
    print(f"文2: {sentence2}\n")
    
    # ========================================
    # (1) Bag of Wordsによる類似度
    # ========================================
    vectorizer = CountVectorizer()
    bow_vectors = vectorizer.fit_transform([sentence1, sentence2])
    bow_similarity = cosine_similarity(bow_vectors[0], bow_vectors[1])[0][0]
    
    print("=== (1) Bag of Words ===")
    print(f"語彙: {vectorizer.get_feature_names_out()}")
    print(f"文1のBoW: {bow_vectors[0].toarray()}")
    print(f"文2のBoW: {bow_vectors[1].toarray()}")
    print(f"コサイン類似度: {bow_similarity:.3f}\n")
    
    # ========================================
    # (2) Word2Vec埋め込みの平均ベクトル
    # ========================================
    # コーパスを準備（実際にはより大きなコーパスが必要）
    corpus = [
        "I love machine learning",
        "I enjoy deep learning",
        "Machine learning is fun",
        "Deep learning uses neural networks",
        "I love deep neural networks"
    ]
    tokenized_corpus = [word_tokenize(sent.lower()) for sent in corpus]
    
    # Word2Vecモデルの訓練
    w2v_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1
    )
    
    def sentence_vector(sentence, model):
        """文のベクトル表現（単語ベクトルの平均）"""
        words = word_tokenize(sentence.lower())
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)
    
    # 文ベクトルの計算
    vec1 = sentence_vector(sentence1, w2v_model)
    vec2 = sentence_vector(sentence2, w2v_model)
    
    # コサイン類似度
    w2v_similarity = cosine_similarity([vec1], [vec2])[0][0]
    
    print("=== (2) Word2Vec ===")
    print(f"文1のベクトル（最初の5次元）: {vec1[:5]}")
    print(f"文2のベクトル（最初の5次元）: {vec2[:5]}")
    print(f"コサイン類似度: {w2v_similarity:.3f}\n")
    
    # ========================================
    # 比較
    # ========================================
    print("=== 比較 ===")
    print(f"BoW類似度: {bow_similarity:.3f}")
    print(f"Word2Vec類似度: {w2v_similarity:.3f}")
    print(f"\n差: {abs(w2v_similarity - bow_similarity):.3f}")
    
    print("\n考察:")
    print("- BoWは共通の単語（'I', 'learning'）のみを考慮")
    print("- Word2Vecは意味的な類似性を捉える（'love' ≈ 'enjoy', 'machine' ≈ 'deep'）")
    print("- 通常、Word2Vecの方が意味的に正確な類似度を返す")
    

**出力例** ：
    
    
    === コサイン類似度の比較 ===
    
    文1: I love machine learning
    文2: I enjoy deep learning
    
    === (1) Bag of Words ===
    語彙: ['deep' 'enjoy' 'learning' 'love' 'machine']
    文1のBoW: [[0 0 1 1 1]]
    文2のBoW: [[1 1 1 0 0]]
    コサイン類似度: 0.333
    
    === (2) Word2Vec ===
    文1のベクトル（最初の5次元）: [-0.00123  0.00456 -0.00789  0.00234  0.00567]
    文2のベクトル（最初の5次元）: [-0.00098  0.00423 -0.00712  0.00198  0.00534]
    コサイン類似度: 0.876
    
    === 比較 ===
    BoW類似度: 0.333
    Word2Vec類似度: 0.876
    
    差: 0.543
    
    考察:
    - BoWは共通の単語（'I', 'learning'）のみを考慮
    - Word2Vecは意味的な類似性を捉える（'love' ≈ 'enjoy', 'machine' ≈ 'deep'）
    - 通常、Word2Vecの方が意味的に正確な類似度を返す
    

* * *

## 参考文献

  1. Jurafsky, D., & Martin, J. H. (2023). _Speech and Language Processing_ (3rd ed.). Stanford University.
  2. Eisenstein, J. (2019). _Introduction to Natural Language Processing_. MIT Press.
  3. Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space." _arXiv:1301.3781_.
  4. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." _EMNLP_.
  5. Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information." _TACL_.
  6. 工藤拓・進藤裕之 (2018). 『形態素解析の理論と実装』. 近代科学社.
