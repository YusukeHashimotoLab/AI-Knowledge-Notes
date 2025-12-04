---
title: 第5章：NLP応用実践
chapter_title: 第5章：NLP応用実践
subtitle: 感情分析から質問応答まで - 実世界のNLPタスク完全実装
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 感情分析（Sentiment Analysis）の実装と評価
  * ✅ 固有表現認識（NER）でエンティティを抽出できる
  * ✅ 質問応答システム（QA）を構築できる
  * ✅ テキスト要約（Summarization）の実装方法を理解する
  * ✅ エンドツーエンドのNLPパイプラインを構築できる
  * ✅ 本番環境へのデプロイとモニタリング手法を習得する

* * *

## 5.1 感情分析（Sentiment Analysis）

### 感情分析とは

**感情分析（Sentiment Analysis）** は、テキストから著者の意見や感情（肯定的・否定的・中立的）を判定するタスクです。

> 応用例：製品レビュー分析、SNS監視、カスタマーサポート、ブランドモニタリング

### 感情分析のタイプ

タイプ | 説明 | 例  
---|---|---  
**Binary Classification** | 肯定/否定の2クラス分類 | レビューが好意的か否定的か  
**Multi-class Classification** | 複数の感情カテゴリ | Very Negative, Negative, Neutral, Positive, Very Positive  
**Aspect-based Sentiment** | 特定の側面に対する感情 | 「料理は美味しいがサービスが悪い」→料理:肯定、サービス:否定  
**Emotion Detection** | 感情の種類を検出 | 喜び、怒り、悲しみ、恐れ、驚き  
  
### Binary感情分析の実装
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # サンプルデータ（映画レビュー）
    reviews = [
        "This movie is absolutely fantastic! I loved every minute.",
        "Terrible film, waste of time and money.",
        "An amazing masterpiece with brilliant acting.",
        "Boring and predictable. Would not recommend.",
        "One of the best movies I've ever seen!",
        "Awful story, poor direction, disappointing overall.",
        "Great cinematography and compelling narrative.",
        "Not worth watching. Very disappointing.",
        "Excellent performances by all actors!",
        "Dull and uninspiring. Fell asleep halfway through.",
        "A true work of art! Highly recommended!",
        "Complete disaster. Avoid at all costs.",
        "Wonderful film with a heartwarming message.",
        "Poorly executed and hard to follow.",
        "Outstanding! A must-see for everyone.",
        "Waste of time. Very poor quality.",
        "Beautiful story and great music.",
        "Terrible acting and weak plot.",
        "Phenomenal! Best movie this year!",
        "Boring and overrated. Not impressed."
    ]
    
    # ラベル（1: Positive, 0: Negative）
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
              1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # データフレーム作成
    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    
    print("=== データセット ===")
    print(df.head(10))
    print(f"\nデータ数: {len(df)}")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'],
        test_size=0.3, random_state=42, stratify=df['sentiment']
    )
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # ロジスティック回帰モデル
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 予測と評価
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=== モデル性能 ===")
    print(f"Accuracy: {accuracy:.3f}")
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive']))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Sentiment Analysis')
    plt.tight_layout()
    plt.show()
    
    # 新しいレビューの予測
    new_reviews = [
        "This is an incredible movie!",
        "What a terrible waste of time.",
        "Pretty good, I enjoyed it."
    ]
    
    new_tfidf = vectorizer.transform(new_reviews)
    predictions = model.predict(new_tfidf)
    probabilities = model.predict_proba(new_tfidf)
    
    print("\n=== 新しいレビューの予測 ===")
    for review, pred, prob in zip(new_reviews, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = prob[pred]
        print(f"Review: {review}")
        print(f"  → {sentiment} (confidence: {confidence:.2%})\n")
    

### BERTによる感情分析
    
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    import torch
    
    # 事前学習済みBERT感情分析モデル
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    # パイプライン作成
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )
    
    # サンプルレビュー（英語と日本語）
    reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Very disappointed with this item.",
        "この商品は素晴らしいです！とても満足しています。",
        "最悪の品質です。がっかりしました。",
        "It's okay. Nothing special but does the job."
    ]
    
    print("=== BERT感情分析 ===\n")
    for review in reviews:
        result = sentiment_pipeline(review)[0]
        stars = int(result['label'].split()[0])
        confidence = result['score']
    
        print(f"Review: {review}")
        print(f"  → Rating: {stars} stars (confidence: {confidence:.2%})")
        print(f"  → Sentiment: {'Positive' if stars >= 4 else 'Negative' if stars <= 2 else 'Neutral'}\n")
    

**出力** ：
    
    
    === BERT感情分析 ===
    
    Review: This product is absolutely amazing! Best purchase ever!
      → Rating: 5 stars (confidence: 87.34%)
      → Sentiment: Positive
    
    Review: Terrible quality. Very disappointed with this item.
      → Rating: 1 stars (confidence: 92.15%)
      → Sentiment: Negative
    
    Review: この商品は素晴らしいです！とても満足しています。
      → Rating: 5 stars (confidence: 78.92%)
      → Sentiment: Positive
    
    Review: 最悪の品質です。がっかりしました。
      → Rating: 1 stars (confidence: 85.67%)
      → Sentiment: Negative
    
    Review: It's okay. Nothing special but does the job.
      → Rating: 3 stars (confidence: 65.43%)
      → Sentiment: Neutral
    

### Aspect-based感情分析
    
    
    import spacy
    from transformers import pipeline
    
    # ABSA（Aspect-Based Sentiment Analysis）の実装
    class AspectBasedSentimentAnalyzer:
        def __init__(self):
            # 感情分析パイプライン
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            # 名詞句抽出用（aspectの候補）
            self.nlp = spacy.load("en_core_web_sm")
    
        def extract_aspects(self, text):
            """テキストからaspect候補を抽出"""
            doc = self.nlp(text)
            aspects = []
    
            # 名詞と形容詞の組み合わせを抽出
            for chunk in doc.noun_chunks:
                aspects.append(chunk.text)
    
            return aspects
    
        def analyze_aspect_sentiment(self, text, aspect):
            """特定aspectに対する感情を分析"""
            # aspectを含む文を抽出
            sentences = text.split('.')
            relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]
    
            if not relevant_sentences:
                return None
    
            # 感情分析
            combined_text = '. '.join(relevant_sentences)
            result = self.sentiment_analyzer(combined_text[:512])[0]  # BERT max length
    
            stars = int(result['label'].split()[0])
            sentiment = 'Positive' if stars >= 4 else 'Negative' if stars <= 2 else 'Neutral'
    
            return {
                'aspect': aspect,
                'sentiment': sentiment,
                'stars': stars,
                'confidence': result['score']
            }
    
        def analyze(self, text):
            """完全なABSA分析"""
            aspects = self.extract_aspects(text)
            results = []
    
            for aspect in aspects:
                result = self.analyze_aspect_sentiment(text, aspect)
                if result:
                    results.append(result)
    
            return results
    
    # 使用例
    analyzer = AspectBasedSentimentAnalyzer()
    
    review = """
    The food at this restaurant was absolutely delicious, especially the pasta.
    However, the service was quite slow and the staff seemed unfriendly.
    The ambiance was nice and cozy. The prices are a bit high but worth it for the quality.
    """
    
    print("=== Aspect-Based Sentiment Analysis ===\n")
    print(f"Review:\n{review}\n")
    
    results = analyzer.analyze(review)
    
    print("Aspect-level Sentiments:")
    for r in results:
        print(f"  {r['aspect']}: {r['sentiment']} ({r['stars']} stars, {r['confidence']:.1%} confidence)")
    
    # 全体的な集計
    positive = sum(1 for r in results if r['sentiment'] == 'Positive')
    negative = sum(1 for r in results if r['sentiment'] == 'Negative')
    neutral = sum(1 for r in results if r['sentiment'] == 'Neutral')
    
    print(f"\nOverall Summary:")
    print(f"  Positive aspects: {positive}")
    print(f"  Negative aspects: {negative}")
    print(f"  Neutral aspects: {neutral}")
    

* * *

## 5.2 固有表現認識（Named Entity Recognition）

### 固有表現認識とは

**固有表現認識（NER: Named Entity Recognition）** は、テキストから人名、組織名、地名、日付などのエンティティを抽出・分類するタスクです。

### 主要なエンティティタイプ

タイプ | 説明 | 例  
---|---|---  
**PERSON** | 人名 | Barack Obama, 山田太郎  
**ORG** | 組織名 | Google, 東京大学  
**GPE** | 地名（国、都市など） | Tokyo, United States  
**DATE** | 日付 | 2025年10月21日, yesterday  
**MONEY** | 金額 | $100, 1万円  
**PRODUCT** | 製品名 | iPhone, Windows  
  
### BIO タギング方式

NERでは**BIO tagging scheme** が一般的です：

  * **B** (Begin): エンティティの開始
  * **I** (Inside): エンティティの内部
  * **O** (Outside): エンティティ外

例：「Barack Obama visited New York」

  * Barack: `B-PERSON`
  * Obama: `I-PERSON`
  * visited: `O`
  * New: `B-GPE`
  * York: `I-GPE`

### spaCyによるNER
    
    
    import spacy
    from spacy import displacy
    import pandas as pd
    
    # 英語モデル読み込み
    nlp = spacy.load("en_core_web_sm")
    
    # サンプルテキスト
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
    in April 1976 in Cupertino, California. The company's first product was
    the Apple I computer. In 2011, Apple became the world's most valuable
    publicly traded company. Tim Cook became CEO in August 2011, succeeding
    Steve Jobs. Today, Apple employs over 150,000 people worldwide and
    generates over $300 billion in annual revenue.
    """
    
    # NER実行
    doc = nlp(text)
    
    print("=== Named Entity Recognition (spaCy) ===\n")
    print(f"Text:\n{text}\n")
    
    # エンティティ抽出
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    # 結果表示
    df_entities = pd.DataFrame(entities)
    print("\nExtracted Entities:")
    print(df_entities.to_string(index=False))
    
    # ラベルごとに集計
    print("\n\nEntity Count by Type:")
    label_counts = df_entities['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # エンティティのハイライト表示（HTMLとして保存可能）
    print("\n\nVisualizing entities...")
    html = displacy.render(doc, style="ent", jupyter=False)
    
    # カスタム色設定で可視化
    colors = {
        "ORG": "#7aecec",
        "PERSON": "#aa9cfc",
        "GPE": "#feca74",
        "DATE": "#ff9561",
        "MONEY": "#9cc9cc"
    }
    options = {"ents": ["ORG", "PERSON", "GPE", "DATE", "MONEY"], "colors": colors}
    displacy.render(doc, style="ent", options=options, jupyter=False)
    

### BERT-based NER（Transformers）
    
    
    from transformers import pipeline
    import pandas as pd
    
    # BERTベースのNERパイプライン
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )
    
    # サンプルテキスト
    text = """
    Elon Musk announced that Tesla will open a new factory in Berlin, Germany.
    The facility is expected to produce 500,000 vehicles per year starting in 2024.
    This follows Tesla's successful Shanghai factory which opened in 2019.
    """
    
    print("=== BERT-based NER ===\n")
    print(f"Text:\n{text}\n")
    
    # NER実行
    entities = ner_pipeline(text)
    
    # 結果表示
    print("\nExtracted Entities:")
    for ent in entities:
        print(f"  {ent['word']:<20} → {ent['entity_group']:<10} (score: {ent['score']:.3f})")
    
    # エンティティをグループ化
    entity_dict = {}
    for ent in entities:
        entity_type = ent['entity_group']
        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        entity_dict[entity_type].append(ent['word'])
    
    print("\n\nGrouped by Entity Type:")
    for entity_type, words in entity_dict.items():
        print(f"  {entity_type}: {', '.join(words)}")
    

**出力** ：
    
    
    === BERT-based NER ===
    
    Text:
    Elon Musk announced that Tesla will open a new factory in Berlin, Germany.
    The facility is expected to produce 500,000 vehicles per year starting in 2024.
    This follows Tesla's successful Shanghai factory which opened in 2019.
    
    Extracted Entities:
      Elon Musk            → PER        (score: 0.999)
      Tesla                → ORG        (score: 0.997)
      Berlin               → LOC        (score: 0.999)
      Germany              → LOC        (score: 0.999)
      Tesla                → ORG        (score: 0.998)
      Shanghai             → LOC        (score: 0.999)
    
    Grouped by Entity Type:
      PER: Elon Musk
      ORG: Tesla, Tesla
      LOC: Berlin, Germany, Shanghai
    

### 日本語NER（GiNZA + BERT）
    
    
    import spacy
    
    # 日本語NER（GiNZAモデル）
    nlp_ja = spacy.load("ja_ginza")
    
    # 日本語サンプルテキスト
    text_ja = """
    2025年10月21日、トヨタ自動車の豊田章男社長が東京で記者会見を開き、
    新型電気自動車の開発計画を発表した。同社は2030年までに100万台の
    生産を目指すとしている。会見には日経新聞やNHKなどのメディアが参加した。
    """
    
    print("=== 日本語 Named Entity Recognition ===\n")
    print(f"テキスト:\n{text_ja}\n")
    
    # NER実行
    doc_ja = nlp_ja(text_ja)
    
    # エンティティ抽出
    print("抽出されたエンティティ:")
    entities_ja = []
    for ent in doc_ja.ents:
        entities_ja.append({
            'テキスト': ent.text,
            'タイプ': ent.label_,
            '詳細': spacy.explain(ent.label_)
        })
        print(f"  {ent.text:<15} → {ent.label_:<10} ({spacy.explain(ent.label_)})")
    
    # DataFrame化
    df_ja = pd.DataFrame(entities_ja)
    print("\n\nエンティティ一覧:")
    print(df_ja.to_string(index=False))
    
    # タイプ別集計
    print("\n\nタイプ別集計:")
    for label, count in df_ja['タイプ'].value_counts().items():
        print(f"  {label}: {count}個")
    

### カスタムNERモデルの訓練
    
    
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification
    )
    from datasets import Dataset
    import numpy as np
    
    # カスタムNERデータセット作成（簡略版）
    train_data = [
        {
            "tokens": ["Apple", "is", "headquartered", "in", "Cupertino"],
            "ner_tags": [3, 0, 0, 0, 5]  # 3: B-ORG, 0: O, 5: B-LOC
        },
        {
            "tokens": ["Steve", "Jobs", "founded", "Apple", "Inc"],
            "ner_tags": [1, 2, 0, 3, 4]  # 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG
        },
        # ... 実際にはもっと多くのデータが必要
    ]
    
    # ラベルマッピング
    label_list = [
        "O",           # 0
        "B-PER",       # 1: Person (Begin)
        "I-PER",       # 2: Person (Inside)
        "B-ORG",       # 3: Organization (Begin)
        "I-ORG",       # 4: Organization (Inside)
        "B-LOC",       # 5: Location (Begin)
        "I-LOC"        # 6: Location (Inside)
    ]
    
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    print("=== カスタムNERモデル訓練 ===\n")
    print(f"ラベル数: {len(label_list)}")
    print(f"ラベル: {label_list}\n")
    
    # トークナイザーとモデル
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    # データセット準備
    def tokenize_and_align_labels(examples):
        """トークン化とラベルの整列"""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True
        )
    
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
    
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # 特殊トークンは無視
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # サブワードは無視
                previous_word_idx = word_idx
    
            labels.append(label_ids)
    
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # データセット変換
    dataset = Dataset.from_list(train_data)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    
    print("訓練データセットの準備完了")
    print(f"サンプル数: {len(tokenized_dataset)}")
    print("\n注意: 実際の訓練には数千〜数万のサンプルが必要です")
    

* * *

## 5.3 質問応答システム（Question Answering）

### 質問応答のタイプ

タイプ | 説明 | 例  
---|---|---  
**Extractive QA** | 文書から答えの箇所を抽出 | SQuAD, NewsQA  
**Abstractive QA** | 文書を理解して新しい文を生成 | 要約型QA  
**Multiple Choice** | 選択肢から正解を選ぶ | RACE, ARC  
**Open-domain QA** | 知識ベース全体から回答 | Google検索的QA  
  
### Extractive QA（BERT）
    
    
    from transformers import pipeline
    
    # BERTベースのQAパイプライン
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/bert-base-cased-squad2"
    )
    
    # コンテキスト（文書）
    context = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical
    rainforest in the Amazon biome that covers most of the Amazon basin of South America.
    This basin encompasses 7 million square kilometers, of which 5.5 million square
    kilometers are covered by the rainforest. The majority of the forest is contained
    within Brazil, with 60% of the rainforest, followed by Peru with 13%, and Colombia
    with 10%. The Amazon represents over half of the planet's remaining rainforests and
    comprises the largest and most biodiverse tract of tropical rainforest in the world,
    with an estimated 390 billion individual trees divided into 16,000 species.
    """
    
    # 質問リスト
    questions = [
        "Where is the Amazon rainforest located?",
        "How many square kilometers does the Amazon basin cover?",
        "What percentage of the Amazon rainforest is in Brazil?",
        "How many tree species are in the Amazon?",
        "Which country has the second largest portion of the Amazon?"
    ]
    
    print("=== Extractive Question Answering ===\n")
    print(f"Context:\n{context}\n")
    print("=" * 70)
    
    for i, question in enumerate(questions, 1):
        result = qa_pipeline(question=question, context=context)
    
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {result['answer']}")
        print(f"   Confidence: {result['score']:.2%}")
        print(f"   Position: characters {result['start']}-{result['end']}")
    

**出力** ：
    
    
    === Extractive Question Answering ===
    
    Context:
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical
    rainforest in the Amazon biome that covers most of the Amazon basin of South America.
    ...
    
    ======================================================================
    
    Q1: Where is the Amazon rainforest located?
    A1: South America
       Confidence: 98.76%
       Position: characters 159-172
    
    Q2: How many square kilometers does the Amazon basin cover?
    A2: 7 million square kilometers
       Confidence: 95.43%
       Position: characters 193-218
    
    Q3: What percentage of the Amazon rainforest is in Brazil?
    A3: 60%
       Confidence: 99.12%
       Position: characters 333-336
    
    Q4: How many tree species are in the Amazon?
    A4: 16,000 species
       Confidence: 97.58%
       Position: characters 602-616
    
    Q5: Which country has the second largest portion of the Amazon?
    A5: Peru
       Confidence: 96.34%
       Position: characters 364-368
    

### 日本語質問応答
    
    
    from transformers import pipeline
    
    # 日本語QAモデル
    qa_pipeline_ja = pipeline(
        "question-answering",
        model="cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    
    # 日本語コンテキスト
    context_ja = """
    富士山は日本の最高峰で、標高3,776メートルの活火山です。
    山梨県と静岡県にまたがり、日本の象徴として国内外に知られています。
    2013年6月にユネスコの世界文化遺産に登録されました。
    富士山は約10万年前から現在の形になり、最後の噴火は1707年の宝永大噴火です。
    毎年7月と8月の登山シーズンには、約30万人の登山者が訪れます。
    """
    
    questions_ja = [
        "富士山の標高は何メートルですか？",
        "富士山が世界遺産に登録されたのはいつですか？",
        "富士山の最後の噴火はいつですか？",
        "登山シーズンに何人くらいが訪れますか？"
    ]
    
    print("=== 日本語質問応答 ===\n")
    print(f"コンテキスト:\n{context_ja}\n")
    print("=" * 70)
    
    for i, question in enumerate(questions_ja, 1):
        result = qa_pipeline_ja(question=question, context=context_ja)
    
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {result['answer']}")
        print(f"   信頼度: {result['score']:.2%}")
    

### Retrieval-based QA（検索拡張）
    
    
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class RetrievalQA:
        """検索ベースの質問応答システム"""
    
        def __init__(self, documents):
            self.documents = documents
    
            # 文書埋め込みモデル
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
            # QAパイプライン
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/bert-base-cased-squad2"
            )
    
            # 文書ベクトル化（事前計算）
            self.doc_embeddings = self._encode_documents()
    
        def _encode_text(self, text):
            """テキストをベクトル化"""
            inputs = self.tokenizer(text, return_tensors='pt',
                                   truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()
    
        def _encode_documents(self):
            """全文書をベクトル化"""
            embeddings = []
            for doc in self.documents:
                emb = self._encode_text(doc)
                embeddings.append(emb)
            return np.vstack(embeddings)
    
        def retrieve_relevant_docs(self, query, top_k=3):
            """質問に関連する文書を検索"""
            query_emb = self._encode_text(query)
            similarities = cosine_similarity(query_emb, self.doc_embeddings)[0]
    
            # Top-k文書のインデックス
            top_indices = np.argsort(similarities)[::-1][:top_k]
    
            relevant_docs = []
            for idx in top_indices:
                relevant_docs.append({
                    'document': self.documents[idx],
                    'similarity': similarities[idx],
                    'index': idx
                })
    
            return relevant_docs
    
        def answer_question(self, question, top_k=3):
            """質問に回答"""
            # 関連文書を検索
            relevant_docs = self.retrieve_relevant_docs(question, top_k=top_k)
    
            # 最も関連性の高い文書で回答
            best_doc = relevant_docs[0]['document']
            result = self.qa_pipeline(question=question, context=best_doc)
    
            return {
                'question': question,
                'answer': result['answer'],
                'confidence': result['score'],
                'source_document': relevant_docs[0]['index'],
                'similarity': relevant_docs[0]['similarity'],
                'all_relevant_docs': relevant_docs
            }
    
    # 文書コレクション
    documents = [
        """Python is a high-level programming language created by Guido van Rossum
        and first released in 1991. It emphasizes code readability and uses
        significant indentation. Python is dynamically typed and garbage-collected.""",
    
        """Machine learning is a branch of artificial intelligence that focuses on
        building systems that learn from data. Common algorithms include decision trees,
        neural networks, and support vector machines.""",
    
        """Deep learning is a subset of machine learning based on artificial neural
        networks with multiple layers. It has achieved remarkable results in computer
        vision, natural language processing, and speech recognition.""",
    
        """Natural language processing (NLP) is a field of AI concerned with the
        interaction between computers and human language. Tasks include sentiment
        analysis, machine translation, and question answering.""",
    
        """The Transformer architecture, introduced in 2017, revolutionized NLP.
        It uses self-attention mechanisms and has led to models like BERT, GPT,
        and T5 that achieve state-of-the-art results."""
    ]
    
    # システム初期化
    print("=== Retrieval-based Question Answering ===\n")
    print("文書をベクトル化中...")
    qa_system = RetrievalQA(documents)
    print(f"完了！ {len(documents)}個の文書を準備しました\n")
    
    # 質問リスト
    questions = [
        "Who created Python?",
        "What is deep learning?",
        "What does NLP stand for?",
        "When was the Transformer architecture introduced?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_system.answer_question(question, top_k=2)
    
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Source: Document #{result['source_document']} (similarity: {result['similarity']:.3f})")
    
        print(f"\nRelevant documents:")
        for i, doc in enumerate(result['all_relevant_docs'], 1):
            print(f"  {i}. Doc #{doc['index']} (similarity: {doc['similarity']:.3f})")
            print(f"     {doc['document'][:100]}...")
    

* * *

## 5.4 テキスト要約（Text Summarization）

### 要約のタイプ

タイプ | 説明 | 手法  
---|---|---  
**Extractive** | 元テキストから重要文を抽出 | TextRank, LexRank  
**Abstractive** | 内容を理解して新しい文を生成 | BART, T5, GPT  
**Single-document** | 1つの文書を要約 | ニュース記事要約  
**Multi-document** | 複数文書を統合要約 | トピック要約  
  
### Extractive Summarization（TextRank）
    
    
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    import nltk
    from nltk.tokenize import sent_tokenize
    
    # NLTK データダウンロード（初回のみ）
    # nltk.download('punkt')
    
    class TextRankSummarizer:
        """TextRankアルゴリズムによる抽出型要約"""
    
        def __init__(self, similarity_threshold=0.1):
            self.similarity_threshold = similarity_threshold
    
        def _build_similarity_matrix(self, sentences):
            """文間の類似度行列を構築"""
            # TF-IDFベクトル化
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
    
            # コサイン類似度計算
            similarity_matrix = cosine_similarity(tfidf_matrix)
    
            # 閾値以下を0に
            similarity_matrix[similarity_matrix < self.similarity_threshold] = 0
    
            return similarity_matrix
    
        def summarize(self, text, num_sentences=3):
            """テキストを要約"""
            # 文分割
            sentences = sent_tokenize(text)
    
            if len(sentences) <= num_sentences:
                return text
    
            # 類似度行列構築
            similarity_matrix = self._build_similarity_matrix(sentences)
    
            # グラフ構築
            graph = nx.from_numpy_array(similarity_matrix)
    
            # PageRank計算
            scores = nx.pagerank(graph)
    
            # スコアでランキング
            ranked_sentences = sorted(
                ((scores[i], s) for i, s in enumerate(sentences)),
                reverse=True
            )
    
            # Top-k文を取得（元の順序を保持）
            top_sentences = sorted(
                ranked_sentences[:num_sentences],
                key=lambda x: sentences.index(x[1])
            )
    
            # 要約生成
            summary = ' '.join([sent for score, sent in top_sentences])
    
            return summary, scores
    
    # サンプルテキスト
    article = """
    Artificial intelligence has made remarkable progress in recent years.
    Deep learning, a subset of machine learning, has been particularly successful.
    Neural networks with many layers can learn complex patterns from data.
    These models have achieved human-level performance on many tasks.
    Computer vision has benefited greatly from deep learning advances.
    Image classification, object detection, and segmentation are now highly accurate.
    Natural language processing has also seen dramatic improvements.
    Machine translation quality has improved significantly with neural approaches.
    Language models can now generate coherent and contextually appropriate text.
    However, challenges remain in areas like reasoning and common sense understanding.
    AI systems still struggle with tasks that humans find easy.
    Researchers are working on more robust and interpretable AI systems.
    The future of AI holds both great promise and important challenges.
    """
    
    print("=== Extractive Summarization (TextRank) ===\n")
    print(f"Original Text ({len(sent_tokenize(article))} sentences):")
    print(article)
    print("\n" + "=" * 70)
    
    summarizer = TextRankSummarizer()
    
    for num_sents in [3, 5]:
        summary, scores = summarizer.summarize(article, num_sentences=num_sents)
        print(f"\n{num_sents}-Sentence Summary:")
        print(summary)
        print(f"\nCompression ratio: {len(summary) / len(article):.1%}")
    

### Abstractive Summarization（BART/T5）
    
    
    from transformers import pipeline
    
    # BARTベースの要約パイプライン
    summarizer_bart = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    
    # T5ベースの要約パイプライン
    summarizer_t5 = pipeline(
        "summarization",
        model="t5-base"
    )
    
    # 長い記事
    long_article = """
    Climate change is one of the most pressing challenges facing humanity today.
    The Earth's average temperature has increased by approximately 1.1 degrees Celsius
    since the pre-industrial era, primarily due to human activities that release
    greenhouse gases into the atmosphere. The burning of fossil fuels for energy,
    deforestation, and industrial processes are the main contributors to this warming trend.
    
    The effects of climate change are already visible worldwide. Extreme weather events,
    such as hurricanes, droughts, and heatwaves, are becoming more frequent and severe.
    Sea levels are rising due to thermal expansion of water and melting ice sheets,
    threatening coastal communities. Ecosystems are being disrupted, with many species
    facing extinction as their habitats change faster than they can adapt.
    
    To address climate change, a global effort is required. The Paris Agreement,
    adopted in 2015, aims to limit global warming to well below 2 degrees Celsius
    above pre-industrial levels. Countries are implementing various strategies,
    including transitioning to renewable energy sources, improving energy efficiency,
    and developing carbon capture technologies. Individual actions, such as reducing
    energy consumption and supporting sustainable practices, also play a crucial role.
    
    Despite progress, significant challenges remain. Many countries still rely heavily
    on fossil fuels, and the transition to clean energy requires substantial investment.
    Political will and international cooperation are essential for achieving climate goals.
    Scientists emphasize that immediate and sustained action is necessary to prevent
    the most catastrophic impacts of climate change and ensure a livable planet for
    future generations.
    """
    
    print("=== Abstractive Summarization ===\n")
    print(f"Original Article ({len(long_article.split())} words):")
    print(long_article)
    print("\n" + "=" * 70)
    
    # BART要約
    print("\n### BART Summary ###")
    bart_summary = summarizer_bart(
        long_article,
        max_length=100,
        min_length=50,
        do_sample=False
    )
    print(bart_summary[0]['summary_text'])
    print(f"Length: {len(bart_summary[0]['summary_text'].split())} words")
    
    # T5要約（異なる長さ）
    print("\n### T5 Summary (Short) ###")
    t5_summary_short = summarizer_t5(
        long_article,
        max_length=60,
        min_length=30
    )
    print(t5_summary_short[0]['summary_text'])
    
    print("\n### T5 Summary (Long) ###")
    t5_summary_long = summarizer_t5(
        long_article,
        max_length=120,
        min_length=60
    )
    print(t5_summary_long[0]['summary_text'])
    

**出力** ：
    
    
    === Abstractive Summarization ===
    
    Original Article (234 words):
    Climate change is one of the most pressing challenges...
    
    ======================================================================
    
    ### BART Summary ###
    Climate change is one of the most pressing challenges facing humanity today.
    The Earth's average temperature has increased by approximately 1.1 degrees Celsius.
    Effects include extreme weather events, rising sea levels, and ecosystem disruption.
    The Paris Agreement aims to limit global warming to below 2 degrees Celsius.
    Length: 51 words
    
    ### T5 Summary (Short) ###
    climate change is caused by human activities that release greenhouse gases.
    extreme weather events are becoming more frequent and severe.
    Length: 19 words
    
    ### T5 Summary (Long) ###
    the earth's average temperature has increased by 1.1 degrees celsius since
    pre-industrial era. burning of fossil fuels, deforestation are main contributors.
    paris agreement aims to limit global warming to below 2 degrees. countries are
    implementing strategies including renewable energy and carbon capture.
    Length: 45 words
    

### 日本語テキスト要約
    
    
    from transformers import pipeline
    
    # 日本語要約モデル
    summarizer_ja = pipeline(
        "summarization",
        model="sonoisa/t5-base-japanese"
    )
    
    # 日本語記事
    article_ja = """
    人工知能（AI）技術は近年急速に発展しており、私たちの生活のあらゆる面に影響を与えています。
    特に、深層学習と呼ばれる技術の進歩により、画像認識や自然言語処理の分野で飛躍的な
    性能向上が実現されました。
    
    現在、AIは医療診断、自動運転、音声アシスタント、レコメンデーションシステムなど、
    多様な分野で活用されています。医療分野では、AIが画像診断で医師を支援し、
    病気の早期発見に貢献しています。自動運転技術は、交通事故の削減と
    移動の効率化を目指して開発が進められています。
    
    しかし、AI技術の発展には課題も存在します。倫理的な問題、プライバシーの保護、
    雇用への影響などが懸念されています。また、AIの判断プロセスが不透明である
    ブラックボックス問題も指摘されています。
    
    今後、AI技術をより良く活用するためには、技術的な進歩だけでなく、
    社会的な議論と適切な規制の整備が必要です。人間とAIが協調する社会の
    実現に向けて、継続的な取り組みが求められています。
    """
    
    print("=== 日本語テキスト要約 ===\n")
    print(f"元記事 ({len(article_ja)}文字):")
    print(article_ja)
    print("\n" + "=" * 70)
    
    # 要約生成
    summary_ja = summarizer_ja(
        article_ja,
        max_length=100,
        min_length=30
    )
    
    print("\n要約:")
    print(summary_ja[0]['summary_text'])
    print(f"\n圧縮率: {len(summary_ja[0]['summary_text']) / len(article_ja):.1%}")
    

* * *

## 5.5 エンドツーエンド実践プロジェクト

### Multi-task NLPパイプライン
    
    
    from transformers import pipeline
    import spacy
    from typing import Dict, List
    import json
    
    class NLPPipeline:
        """包括的なNLPパイプライン"""
    
        def __init__(self):
            print("NLPパイプラインを初期化中...")
    
            # 各タスクのモデル読み込み
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
    
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
    
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/bert-base-cased-squad2"
            )
    
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
    
            # spaCy（トークン化、品詞タグ付け）
            self.nlp = spacy.load("en_core_web_sm")
    
            print("初期化完了！\n")
    
        def analyze_text(self, text: str) -> Dict:
            """テキストの包括的分析"""
            results = {}
    
            # 1. 基本統計
            doc = self.nlp(text)
            results['statistics'] = {
                'num_characters': len(text),
                'num_words': len([token for token in doc if not token.is_punct]),
                'num_sentences': len(list(doc.sents)),
                'num_unique_words': len(set([token.text.lower() for token in doc
                                            if not token.is_punct]))
            }
    
            # 2. 感情分析
            sentiment = self.sentiment_analyzer(text[:512])[0]
            results['sentiment'] = {
                'label': sentiment['label'],
                'score': round(sentiment['score'], 4)
            }
    
            # 3. 固有表現認識
            entities = self.ner_pipeline(text)
            results['entities'] = [
                {
                    'text': ent['word'],
                    'type': ent['entity_group'],
                    'score': round(ent['score'], 4)
                }
                for ent in entities
            ]
    
            # 4. キーワード抽出（名詞句）
            keywords = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # 3語以下
                    keywords.append(chunk.text)
            results['keywords'] = list(set(keywords))[:10]
    
            # 5. 品詞タグ分布
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            results['pos_distribution'] = pos_counts
    
            return results
    
        def process_document(self, text: str,
                            questions: List[str] = None,
                            summarize: bool = True) -> Dict:
            """文書の完全処理"""
            results = {
                'original_text': text,
                'analysis': self.analyze_text(text)
            }
    
            # 要約
            if summarize and len(text.split()) > 50:
                summary = self.summarizer(
                    text,
                    max_length=100,
                    min_length=30,
                    do_sample=False
                )
                results['summary'] = summary[0]['summary_text']
    
            # 質問応答
            if questions:
                results['qa'] = []
                for q in questions:
                    answer = self.qa_pipeline(question=q, context=text)
                    results['qa'].append({
                        'question': q,
                        'answer': answer['answer'],
                        'confidence': round(answer['score'], 4)
                    })
    
            return results
    
    # システム初期化
    pipeline = NLPPipeline()
    
    # サンプル文書
    document = """
    Apple Inc. announced record quarterly earnings on Tuesday, with revenue
    reaching $90 billion. CEO Tim Cook stated that the strong performance was
    driven by robust iPhone sales and growing services revenue. The company's
    stock price jumped 5% following the announcement.
    
    Apple also revealed plans to invest $50 billion in research and development
    over the next five years, focusing on artificial intelligence and augmented
    reality technologies. The investment will create thousands of new jobs in
    the United States and internationally.
    
    However, analysts expressed concerns about potential supply chain disruptions
    and increasing competition in the smartphone market. Despite these challenges,
    Apple remains optimistic about future growth prospects.
    """
    
    # 質問リスト
    questions = [
        "How much revenue did Apple report?",
        "Who is the CEO of Apple?",
        "How much will Apple invest in R&D?",
        "What technologies will Apple focus on?"
    ]
    
    print("=== Multi-task NLP Pipeline ===\n")
    print("文書を処理中...\n")
    
    # 完全処理
    results = pipeline.process_document(
        text=document,
        questions=questions,
        summarize=True
    )
    
    # 結果表示
    print("### 1. 基本統計 ###")
    stats = results['analysis']['statistics']
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n### 2. 感情分析 ###")
    sentiment = results['analysis']['sentiment']
    print(f"  Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.1%})")
    
    print("\n### 3. 固有表現 ###")
    for ent in results['analysis']['entities'][:10]:
        print(f"  {ent['text']:<20} → {ent['type']:<10} ({ent['score']:.1%})")
    
    print("\n### 4. キーワード ###")
    print(f"  {', '.join(results['analysis']['keywords'])}")
    
    print("\n### 5. 要約 ###")
    print(f"  {results['summary']}")
    
    print("\n### 6. 質問応答 ###")
    for qa in results['qa']:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']} (confidence: {qa['confidence']:.1%})\n")
    
    # JSON出力
    print("\n### JSON出力 ###")
    json_output = json.dumps(results, indent=2, ensure_ascii=False)
    print(json_output[:500] + "...")
    

### FastAPIでのAPI開発
    
    
    # ファイル名: nlp_api.py
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from transformers import pipeline
    from typing import List, Optional
    import uvicorn
    
    # FastAPIアプリケーション
    app = FastAPI(
        title="NLP API",
        description="包括的な自然言語処理API",
        version="1.0.0"
    )
    
    # リクエストモデル
    class TextInput(BaseModel):
        text: str
        max_length: Optional[int] = 100
    
    class QAInput(BaseModel):
        question: str
        context: str
    
    class BatchTextInput(BaseModel):
        texts: List[str]
    
    # モデル初期化
    sentiment_analyzer = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_pipeline = pipeline("question-answering")
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    
    # エンドポイント
    
    @app.get("/")
    async def root():
        """APIのルート"""
        return {
            "message": "NLP API へようこそ",
            "endpoints": [
                "/sentiment",
                "/summarize",
                "/qa",
                "/ner",
                "/batch-sentiment"
            ]
        }
    
    @app.post("/sentiment")
    async def analyze_sentiment(input_data: TextInput):
        """感情分析エンドポイント"""
        try:
            result = sentiment_analyzer(input_data.text[:512])[0]
            return {
                "text": input_data.text,
                "sentiment": result['label'],
                "confidence": round(result['score'], 4)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/summarize")
    async def summarize_text(input_data: TextInput):
        """テキスト要約エンドポイント"""
        try:
            summary = summarizer(
                input_data.text,
                max_length=input_data.max_length,
                min_length=30,
                do_sample=False
            )
            return {
                "original_text": input_data.text,
                "summary": summary[0]['summary_text'],
                "compression_ratio": len(summary[0]['summary_text']) / len(input_data.text)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/qa")
    async def answer_question(input_data: QAInput):
        """質問応答エンドポイント"""
        try:
            result = qa_pipeline(
                question=input_data.question,
                context=input_data.context
            )
            return {
                "question": input_data.question,
                "answer": result['answer'],
                "confidence": round(result['score'], 4),
                "start": result['start'],
                "end": result['end']
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ner")
    async def extract_entities(input_data: TextInput):
        """固有表現認識エンドポイント"""
        try:
            entities = ner_pipeline(input_data.text)
            return {
                "text": input_data.text,
                "entities": [
                    {
                        "text": ent['word'],
                        "type": ent['entity_group'],
                        "score": round(ent['score'], 4)
                    }
                    for ent in entities
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch-sentiment")
    async def batch_sentiment_analysis(input_data: BatchTextInput):
        """バッチ感情分析"""
        try:
            results = []
            for text in input_data.texts:
                result = sentiment_analyzer(text[:512])[0]
                results.append({
                    "text": text,
                    "sentiment": result['label'],
                    "confidence": round(result['score'], 4)
                })
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ヘルスチェック
    @app.get("/health")
    async def health_check():
        """APIのヘルスチェック"""
        return {"status": "healthy", "models_loaded": True}
    
    # サーバー起動
    if __name__ == "__main__":
        print("NLP APIを起動中...")
        print("ドキュメント: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    

**使用方法** ：
    
    
    # サーバー起動
    python nlp_api.py
    
    # curlでテスト
    curl -X POST "http://localhost:8000/sentiment" \
      -H "Content-Type: application/json" \
      -d '{"text": "This is an amazing product!"}'
    
    # Pythonクライアント
    import requests
    
    response = requests.post(
        "http://localhost:8000/sentiment",
        json={"text": "I love this API!"}
    )
    print(response.json())
    

* * *

## 5.6 本章のまとめ

### 学んだこと

  1. **感情分析**

     * Binary、Multi-class、Aspect-based分類
     * TF-IDF + ロジスティック回帰
     * BERT事前学習モデルの活用
     * 日本語感情分析
  2. **固有表現認識**

     * BIOタギング方式
     * spaCy、BERT-based NER
     * 日本語NER（GiNZA）
     * カスタムNERモデルの訓練
  3. **質問応答システム**

     * Extractive QA（BERT）
     * Retrieval-based QA
     * 日本語QA
     * 文書検索と回答生成の統合
  4. **テキスト要約**

     * Extractive（TextRank）
     * Abstractive（BART、T5）
     * 日本語要約
     * 要約品質の評価
  5. **エンドツーエンド実装**

     * Multi-task NLPパイプライン
     * FastAPIでのAPI開発
     * 本番環境デプロイ
     * モニタリングと評価

### 実装のベストプラクティス

項目 | 推奨事項  
---|---  
**モデル選択** | タスクに応じて適切なモデルを選択（精度 vs 速度のトレードオフ）  
**前処理** | テキストクリーニング、正規化を統一  
**評価指標** | タスクごとに適切な指標（F1、BLEU、ROUGE等）  
**エラーハンドリング** | 入力長制限、例外処理を実装  
**パフォーマンス** | バッチ処理、モデルキャッシュ、GPU活用  
**モニタリング** | 推論時間、精度、エラー率を記録  
  
### 次のステップ

  * 大規模言語モデル（LLM）の理解と活用
  * プロンプトエンジニアリング
  * RAG（Retrieval-Augmented Generation）
  * Fine-tuningとドメイン適応
  * マルチモーダルNLP（テキスト+画像）

* * *

## 演習問題

### 問題1（難易度：easy）

Extractive要約とAbstractive要約の違いを説明し、それぞれの長所と短所を述べてください。

解答例

**解答** ：

**Extractive要約** ：

  * 定義: 元テキストから重要な文をそのまま抽出
  * 手法: TextRank, LexRank, TF-IDF
  * 長所: 
    * 文法的に正しい（元の文を使用）
    * 計算コストが低い
    * 事実の歪曲が少ない
  * 短所: 
    * 冗長性が残る
    * 文脈に応じた表現変更ができない
    * 要約の流暢性が低い場合がある

**Abstractive要約** ：

  * 定義: 内容を理解して新しい文を生成
  * 手法: BART, T5, GPT
  * 長所: 
    * 簡潔で流暢な要約
    * パラフレーズや言い換えが可能
    * より人間らしい要約
  * 短所: 
    * 事実の誤り（Hallucination）のリスク
    * 計算コストが高い
    * 大量の訓練データが必要

### 問題2（難易度：medium）

以下のコードを完成させて、カスタム感情分析器を実装してください。データセットは映画レビューとし、Positive/Negativeの2クラス分類を行います。
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # データ（完成させてください）
    reviews = [
        # Positive reviews (少なくとも5個)
    
        # Negative reviews (少なくとも5個)
    ]
    labels = []  # 対応するラベル
    
    # モデル実装（完成させてください）
    

解答例
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    # データセット
    reviews = [
        # Positive reviews
        "This movie is absolutely fantastic! Loved it!",
        "Amazing performances and brilliant storyline.",
        "One of the best films I've ever seen.",
        "Highly recommended. A true masterpiece!",
        "Wonderful cinematography and great acting.",
        "Excellent movie with a heartwarming message.",
        "Phenomenal! Must-see for everyone.",
    
        # Negative reviews
        "Terrible film. Complete waste of time.",
        "Boring and poorly executed.",
        "Very disappointing. Would not recommend.",
        "Awful story and weak performances.",
        "Dull and uninspiring throughout.",
        "Poor quality. Not worth watching.",
        "Complete disaster. Avoid at all costs."
    ]
    
    labels = [1, 1, 1, 1, 1, 1, 1,  # Positive
              0, 0, 0, 0, 0, 0, 0]  # Negative
    
    print("=== カスタム感情分析器 ===\n")
    print(f"データ数: {len(reviews)}")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}\n")
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # モデル訓練
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 評価
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=== モデル評価 ===")
    print(f"Accuracy: {accuracy:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive']))
    
    # 新しいレビューで予測
    new_reviews = [
        "Incredible movie! Best I've seen this year!",
        "Absolutely terrible. Don't waste your money."
    ]
    
    new_tfidf = vectorizer.transform(new_reviews)
    predictions = model.predict(new_tfidf)
    probabilities = model.predict_proba(new_tfidf)
    
    print("\n=== 新しいレビューの予測 ===")
    for review, pred, prob in zip(new_reviews, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = prob[pred]
        print(f"Review: {review}")
        print(f"  → {sentiment} (confidence: {confidence:.1%})\n")
    

### 問題3（難易度：medium）

BIO タギング方式を使って、以下の文にNERラベルを付けてください。

文：「Apple Inc. CEO Tim Cook visited Tokyo on October 21, 2025.」

解答例

**解答** ：

Token | BIO Tag | Entity Type  
---|---|---  
Apple | B-ORG | Organization (Begin)  
Inc. | I-ORG | Organization (Inside)  
CEO | O | Outside  
Tim | B-PER | Person (Begin)  
Cook | I-PER | Person (Inside)  
visited | O | Outside  
Tokyo | B-LOC | Location (Begin)  
on | O | Outside  
October | B-DATE | Date (Begin)  
21 | I-DATE | Date (Inside)  
, | I-DATE | Date (Inside)  
2025 | I-DATE | Date (Inside)  
. | O | Outside  
  
**エンティティまとめ** ：

  * ORG: Apple Inc.
  * PER: Tim Cook
  * LOC: Tokyo
  * DATE: October 21, 2025

### 問題4（難易度：hard）

Retrieval-based QAシステムを実装してください。複数の文書から質問に関連する文書を検索し、その文書を使って回答を生成する仕組みを作成してください。

解答例
    
    
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SimpleRetrievalQA:
        def __init__(self, documents):
            self.documents = documents
    
            # 文書埋め込み用モデル
            self.tokenizer = AutoTokenizer.from_pretrained(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            self.encoder = AutoModel.from_pretrained(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
    
            # QAパイプライン
            self.qa = pipeline(
                "question-answering",
                model="deepset/bert-base-cased-squad2"
            )
    
            # 文書ベクトル化
            print("文書をベクトル化中...")
            self.doc_embeddings = self._encode_documents()
            print(f"完了！ {len(documents)}個の文書を準備")
    
        def _encode_text(self, text):
            """テキストをベクトル化"""
            inputs = self.tokenizer(
                text, return_tensors='pt',
                truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            # Mean pooling
            return outputs.last_hidden_state.mean(dim=1).numpy()
    
        def _encode_documents(self):
            """全文書をベクトル化"""
            return np.vstack([self._encode_text(doc) for doc in self.documents])
    
        def retrieve(self, query, top_k=2):
            """関連文書を検索"""
            query_emb = self._encode_text(query)
            similarities = cosine_similarity(query_emb, self.doc_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
    
            return [
                {
                    'doc': self.documents[i],
                    'similarity': similarities[i],
                    'index': i
                }
                for i in top_indices
            ]
    
        def answer(self, question, top_k=2):
            """質問に回答"""
            # 関連文書を検索
            docs = self.retrieve(question, top_k)
    
            # 最も関連性の高い文書で回答
            best_doc = docs[0]['doc']
            result = self.qa(question=question, context=best_doc)
    
            return {
                'question': question,
                'answer': result['answer'],
                'confidence': result['score'],
                'source_doc_index': docs[0]['index'],
                'source_similarity': docs[0]['similarity'],
                'retrieved_docs': docs
            }
    
    # 文書コレクション
    documents = [
        """Python is a high-level programming language created by Guido van Rossum.
        It was first released in 1991 and emphasizes code readability.""",
    
        """Machine learning is a subset of AI that enables systems to learn from data.
        Popular algorithms include decision trees and neural networks.""",
    
        """Deep learning uses neural networks with multiple layers. It excels at
        computer vision, NLP, and speech recognition tasks.""",
    
        """Natural language processing (NLP) deals with human-computer language
        interaction. Tasks include sentiment analysis and machine translation.""",
    
        """The Transformer architecture, introduced in 2017, revolutionized NLP
        with self-attention mechanisms. It led to BERT and GPT models."""
    ]
    
    # システム初期化と使用
    print("=== Retrieval-based QA System ===\n")
    qa_system = SimpleRetrievalQA(documents)
    
    questions = [
        "Who created Python?",
        "What is deep learning good at?",
        "When was the Transformer introduced?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = qa_system.answer(q)
        print(f"A: {result['answer']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Source: Doc #{result['source_doc_index']} "
              f"(similarity: {result['source_similarity']:.3f})")
    

### 問題5（難易度：hard）

FastAPIを使って、感情分析、NER、要約の3つの機能を提供するREST APIを実装してください。エラーハンドリングとレスポンス形式の統一も考慮してください。

解答例
    
    
    # ファイル名: complete_nlp_api.py
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, validator
    from transformers import pipeline
    from typing import Optional, List, Dict, Any
    import uvicorn
    from datetime import datetime
    
    app = FastAPI(
        title="Complete NLP API",
        description="感情分析、NER、要約を提供するAPI",
        version="1.0.0"
    )
    
    # リクエストモデル
    class TextInput(BaseModel):
        text: str
    
        @validator('text')
        def text_not_empty(cls, v):
            if not v or not v.strip():
                raise ValueError('テキストは空にできません')
            return v
    
    class SummarizeInput(TextInput):
        max_length: Optional[int] = 100
        min_length: Optional[int] = 30
    
        @validator('max_length')
        def valid_max_length(cls, v):
            if v < 10 or v > 500:
                raise ValueError('max_lengthは10-500の範囲で指定してください')
            return v
    
    # レスポンスモデル
    class APIResponse(BaseModel):
        success: bool
        timestamp: str
        data: Optional[Dict[Any, Any]] = None
        error: Optional[str] = None
    
    # モデル初期化
    print("モデルを読み込み中...")
    sentiment_analyzer = pipeline("sentiment-analysis")
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("完了！")
    
    # ヘルパー関数
    def create_response(success: bool, data: Dict = None, error: str = None) -> APIResponse:
        """統一レスポンス作成"""
        return APIResponse(
            success=success,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            error=error
        )
    
    # エンドポイント
    @app.get("/")
    async def root():
        return create_response(
            success=True,
            data={
                "message": "Complete NLP API",
                "endpoints": {
                    "sentiment": "/api/sentiment",
                    "ner": "/api/ner",
                    "summarize": "/api/summarize",
                    "health": "/health"
                }
            }
        )
    
    @app.post("/api/sentiment", response_model=APIResponse)
    async def analyze_sentiment(input_data: TextInput):
        """感情分析"""
        try:
            result = sentiment_analyzer(input_data.text[:512])[0]
            return create_response(
                success=True,
                data={
                    "text": input_data.text,
                    "sentiment": result['label'],
                    "confidence": round(result['score'], 4)
                }
            )
        except Exception as e:
            return create_response(
                success=False,
                error=f"感情分析エラー: {str(e)}"
            )
    
    @app.post("/api/ner", response_model=APIResponse)
    async def extract_entities(input_data: TextInput):
        """固有表現認識"""
        try:
            entities = ner_pipeline(input_data.text)
            return create_response(
                success=True,
                data={
                    "text": input_data.text,
                    "entities": [
                        {
                            "text": ent['word'],
                            "type": ent['entity_group'],
                            "confidence": round(ent['score'], 4)
                        }
                        for ent in entities
                    ],
                    "count": len(entities)
                }
            )
        except Exception as e:
            return create_response(
                success=False,
                error=f"NERエラー: {str(e)}"
            )
    
    @app.post("/api/summarize", response_model=APIResponse)
    async def summarize_text(input_data: SummarizeInput):
        """テキスト要約"""
        try:
            if len(input_data.text.split()) < 30:
                return create_response(
                    success=False,
                    error="テキストが短すぎます（最低30語必要）"
                )
    
            summary = summarizer(
                input_data.text,
                max_length=input_data.max_length,
                min_length=input_data.min_length,
                do_sample=False
            )
    
            return create_response(
                success=True,
                data={
                    "original_text": input_data.text,
                    "summary": summary[0]['summary_text'],
                    "original_length": len(input_data.text.split()),
                    "summary_length": len(summary[0]['summary_text'].split()),
                    "compression_ratio": round(
                        len(summary[0]['summary_text']) / len(input_data.text), 3
                    )
                }
            )
        except Exception as e:
            return create_response(
                success=False,
                error=f"要約エラー: {str(e)}"
            )
    
    @app.get("/health")
    async def health_check():
        """ヘルスチェック"""
        return create_response(
            success=True,
            data={
                "status": "healthy",
                "models": {
                    "sentiment": "loaded",
                    "ner": "loaded",
                    "summarizer": "loaded"
                }
            }
        )
    
    if __name__ == "__main__":
        print("\n=== Complete NLP API 起動 ===")
        print("ドキュメント: http://localhost:8000/docs")
        print("API: http://localhost:8000/")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    

**使用例（Pythonクライアント）** ：
    
    
    import requests
    
    API_URL = "http://localhost:8000"
    
    # 感情分析
    response = requests.post(
        f"{API_URL}/api/sentiment",
        json={"text": "This API is amazing!"}
    )
    print("Sentiment:", response.json())
    
    # NER
    response = requests.post(
        f"{API_URL}/api/ner",
        json={"text": "Apple Inc. CEO Tim Cook visited Tokyo."}
    )
    print("\nNER:", response.json())
    
    # 要約
    long_text = """
    Artificial intelligence has made remarkable progress...
    (長いテキスト)
    """
    response = requests.post(
        f"{API_URL}/api/summarize",
        json={"text": long_text, "max_length": 80}
    )
    print("\nSummary:", response.json())
    

* * *

## 参考文献

  1. Devlin, J., et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL.
  2. Lewis, M., et al. (2020). _BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension_. ACL.
  3. Raffel, C., et al. (2020). _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)_. JMLR.
  4. Rajpurkar, P., et al. (2016). _SQuAD: 100,000+ Questions for Machine Comprehension of Text_. EMNLP.
  5. Mihalcea, R., & Tarau, P. (2004). _TextRank: Bringing Order into Text_. EMNLP.
  6. Lample, G., et al. (2016). _Neural Architectures for Named Entity Recognition_. NAACL.
  7. Socher, R., et al. (2013). _Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank_. EMNLP.
  8. Jurafsky, D., & Martin, J. H. (2023). _Speech and Language Processing_ (3rd ed.). Prentice Hall.
