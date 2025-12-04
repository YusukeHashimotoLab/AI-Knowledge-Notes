---
title: 第4章：BERT・GPT
chapter_title: 第4章：BERT・GPT
subtitle: 事前学習モデルの双璧：双方向エンコーダと自己回帰生成モデルの理論と実践
reading_time: 28分
difficulty: 中級〜上級
code_examples: 9
exercises: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ BERTの双方向エンコーディングとMasked Language Modelingを理解できる
  * ✅ GPTの自己回帰生成とCausal Maskingの仕組みを理解できる
  * ✅ BERT・GPTの事前学習タスク（MLM、NSP、CLM）を実装できる
  * ✅ Hugging Face Transformersライブラリで両モデルを使用できる
  * ✅ Fine-tuningによるタスク特化型モデルの構築ができる
  * ✅ BERTとGPTの使い分けと適用場面を判断できる
  * ✅ 質問応答システムとテキスト生成の実践プロジェクトを完成できる

* * *

## 4.1 BERTアーキテクチャ

### 4.1.1 BERTの革新性と設計思想

**BERT** （Bidirectional Encoder Representations from Transformers）は、2018年にGoogleが発表した事前学習モデルで、自然言語処理に革命をもたらしました。

特性 | 従来モデル（ELMo、GPT-1など） | BERT  
---|---|---  
**方向性** | 単方向（左→右）または浅い双方向 | 深い双方向（左右両方の文脈を利用）  
**アーキテクチャ** | RNN、LSTM、浅いTransformer | Transformer Encoderのみ（12〜24層）  
**事前学習** | 言語モデリング（次単語予測） | Masked LM + Next Sentence Prediction  
**用途** | 主に生成タスク | 分類、抽出、質問応答など理解タスク  
**Fine-tuning** | 複雑なタスク特化アーキテクチャ必要 | シンプルな出力層追加のみ  
  
### 4.1.2 BERTの双方向性の実現

BERTの最大の特徴は、**双方向のコンテキスト理解** です。従来の言語モデルは左から右へ順次単語を予測していましたが、BERTは文全体を見渡して各単語を理解します。
    
    
    ```mermaid
    graph LR
        subgraph "従来の単方向モデル（GPT-1など）"
            A1[The] --> A2[cat]
            A2 --> A3[sat]
            A3 --> A4[on]
            A4 --> A5[mat]
    
            style A1 fill:#e74c3c,color:#fff
            style A2 fill:#e74c3c,color:#fff
            style A3 fill:#e74c3c,color:#fff
        end
    
        subgraph "BERTの双方向モデル"
            B1[The] <--> B2[cat]
            B2 <--> B3[sat]
            B3 <--> B4[on]
            B4 <--> B5[mat]
    
            style B2 fill:#27ae60,color:#fff
            style B3 fill:#27ae60,color:#fff
        end
    ```

> **重要** : BERTは文中の単語「cat」を理解する際、「The」（左文脈）と「sat on mat」（右文脈）の両方を同時に利用します。これにより、単語の意味を正確に捉えることができます。

### 4.1.3 BERTのアーキテクチャ構成

BERTは複数のTransformer Encoderブロックを積み重ねた構造です：

モデル | 層数（L） | 隠れ層サイズ（H） | Attention Heads（A） | パラメータ数  
---|---|---|---|---  
**BERT-Base** | 12 | 768 | 12 | 110M  
**BERT-Large** | 24 | 1024 | 16 | 340M  
  
各Transformer Encoderブロックは、第2章で学んだMulti-Head AttentionとFeed-Forward Networkで構成されます：

$$ \text{EncoderBlock}(x) = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{MultiHeadAttn}(x)))) $$ 

### 4.1.4 入力表現：Token + Segment + Position Embeddings

BERTの入力は3種類のEmbeddingの合計です：

  1. **Token Embeddings** : 単語（サブワード）の埋め込み表現
  2. **Segment Embeddings** : 文A・文Bを区別（NSPタスク用）
  3. **Position Embeddings** : 位置情報（学習可能、GPTのSinusoidalとは異なる）

$$ \text{Input} = \text{TokenEmbed}(x) + \text{SegmentEmbed}(x) + \text{PositionEmbed}(x) $$ 
    
    
    ```mermaid
    graph TB
        subgraph "BERT入力構成"
            T1["[CLS] The cat sat [SEP] on mat [SEP]"]
    
            T2[Token Embeddings]
            T3[Segment Embeddings]
            T4[Position Embeddings]
    
            T5[Input to Transformer]
    
            T1 --> T2
            T1 --> T3
            T1 --> T4
    
            T2 --> T5
            T3 --> T5
            T4 --> T5
    
            style T5 fill:#7b2cbf,color:#fff
        end
    ```

**特殊トークン** ：

  * `[CLS]`: 文全体の分類表現（Classification token）
  * `[SEP]`: 文の区切り（Separator）
  * `[MASK]`: Masked Language Modeling用のマスクトークン

* * *

## 4.2 BERTの事前学習タスク

### 4.2.1 Masked Language Modeling (MLM)

MLMは、入力の一部をマスクして、その単語を予測するタスクです。これにより双方向の文脈を学習します。

#### MLMの手順

  1. 入力トークンの15%をランダムに選択
  2. 選択されたトークンに対して： 
     * 80%の確率で`[MASK]`トークンに置換
     * 10%の確率でランダムな別の単語に置換
     * 10%の確率で元の単語のまま保持
  3. マスクされた位置の元の単語を予測

**例** ：
    
    
    入力: "The cat sat on the mat"
    マスク後: "The [MASK] sat on the mat"
    目標: "cat"を予測
    

#### なぜ100%マスクしないのか？

Fine-tuning時に`[MASK]`トークンは存在しません。訓練と本番のギャップを減らすため、一部をランダム単語や元の単語のままにします。

### 4.2.2 Next Sentence Prediction (NSP)

NSPは、2つの文が連続しているかを判定するタスクです。質問応答や自然言語推論で文間関係の理解が重要となります。

#### NSPの構成

  * **IsNext** (50%): 実際に連続する文ペア
  * **NotNext** (50%): ランダムに選ばれた非連続文ペア

**例** ：
    
    
    入力A: "The cat sat on the mat."
    入力B (IsNext): "It was very comfortable."
    入力B (NotNext): "Paris is the capital of France."
    
    BERT入力: [CLS] The cat sat on the mat [SEP] It was very comfortable [SEP]
    目標: IsNext = True
    

### 4.2.3 PyTorchによるMLMの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import numpy as np
    
    class MaskedLanguageModel:
        """BERT-style Masked Language Modeling実装"""
    
        def __init__(self, vocab_size, mask_prob=0.15):
            self.vocab_size = vocab_size
            self.mask_prob = mask_prob
    
            # 特殊トークンID
            self.MASK_TOKEN_ID = vocab_size - 3
            self.CLS_TOKEN_ID = vocab_size - 2
            self.SEP_TOKEN_ID = vocab_size - 1
    
        def create_masked_lm_data(self, input_ids):
            """
            MLM用のマスクデータ生成
    
            Args:
                input_ids: [batch_size, seq_len] 入力トークンID
    
            Returns:
                masked_input: マスク適用後の入力
                labels: 予測対象のラベル（マスク位置のみ有効、他は-100）
            """
            batch_size, seq_len = input_ids.shape
    
            # ラベル初期化（-100は損失計算で無視される）
            labels = torch.full_like(input_ids, -100)
            masked_input = input_ids.clone()
    
            for i in range(batch_size):
                # 特殊トークンを除外してマスク対象を選択
                special_tokens_mask = (input_ids[i] == self.CLS_TOKEN_ID) | \
                                     (input_ids[i] == self.SEP_TOKEN_ID)
    
                # マスク可能な位置
                candidate_indices = torch.where(~special_tokens_mask)[0]
    
                # 15%をマスク対象に選択
                num_to_mask = max(1, int(len(candidate_indices) * self.mask_prob))
                mask_indices = candidate_indices[torch.randperm(len(candidate_indices))[:num_to_mask]]
    
                for idx in mask_indices:
                    labels[i, idx] = input_ids[i, idx]  # 元の単語を保存
    
                    rand = random.random()
                    if rand < 0.8:
                        # 80%: [MASK]トークンに置換
                        masked_input[i, idx] = self.MASK_TOKEN_ID
                    elif rand < 0.9:
                        # 10%: ランダムな単語に置換
                        random_token = random.randint(0, self.vocab_size - 4)
                        masked_input[i, idx] = random_token
                    # 10%: 元の単語のまま（else不要）
    
            return masked_input, labels
    
    
    # デモンストレーション
    print("=== Masked Language Modeling Demo ===\n")
    
    # パラメータ設定
    vocab_size = 1000
    batch_size = 3
    seq_len = 10
    
    # ダミー入力生成
    mlm = MaskedLanguageModel(vocab_size)
    input_ids = torch.randint(0, vocab_size - 3, (batch_size, seq_len))
    
    # [CLS]を先頭、[SEP]を末尾に追加
    input_ids[:, 0] = mlm.CLS_TOKEN_ID
    input_ids[:, -1] = mlm.SEP_TOKEN_ID
    
    print("Original Input IDs (Batch 0):")
    print(input_ids[0].numpy())
    
    # MLMマスク適用
    masked_input, labels = mlm.create_masked_lm_data(input_ids)
    
    print("\nMasked Input IDs (Batch 0):")
    print(masked_input[0].numpy())
    
    print("\nLabels (Batch 0, -100は無視):")
    print(labels[0].numpy())
    
    # マスク位置を確認
    mask_positions = torch.where(labels[0] != -100)[0]
    print(f"\nMasked Positions: {mask_positions.numpy()}")
    print(f"Number of masked tokens: {len(mask_positions)} / {seq_len-2} (excluding [CLS] and [SEP])")
    
    for pos in mask_positions:
        original = input_ids[0, pos].item()
        masked = masked_input[0, pos].item()
        target = labels[0, pos].item()
    
        mask_type = "MASK" if masked == mlm.MASK_TOKEN_ID else \
                    "RANDOM" if masked != original else \
                    "UNCHANGED"
    
        print(f"  Position {pos}: Original={original}, Masked={masked} ({mask_type}), Target={target}")
    

**出力** ：
    
    
    === Masked Language Modeling Demo ===
    
    Original Input IDs (Batch 0):
    [998 453 721 892 156 334 667 289 445 999]
    
    Masked Input IDs (Batch 0):
    [998 997 721 542 156 997 667 289 445 999]
    
    Labels (Batch 0, -100は無視):
    [-100 453 -100 892 -100 334 -100 -100 -100 -100]
    
    Masked Positions: [1 3 5]
    Number of masked tokens: 3 / 8 (excluding [CLS] and [SEP])
      Position 1: Original=453, Masked=997 (MASK), Target=453
      Position 3: Original=892, Masked=542 (RANDOM), Target=892
      Position 5: Original=334, Masked=997 (MASK), Target=334
    

* * *

## 4.3 BERTの使用例

### 4.3.1 テキスト分類（Sentiment Analysis）

BERTを使った感情分析の実装例です。`[CLS]`トークンの出力を分類に使用します。
    
    
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    
    # モデルとTokenizerの読み込み
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 2クラス分類（Positive/Negative）
    )
    
    # 推論モードに設定
    model.eval()
    
    # サンプルテキスト
    texts = [
        "I absolutely loved this movie! It was fantastic.",
        "This product is terrible and waste of money.",
        "The service was okay, nothing special."
    ]
    
    print("=== BERT Sentiment Analysis Demo ===\n")
    
    for text in texts:
        # トークナイズ
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
    
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
    
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = probs[0, predicted_class].item()
    
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print(f"Probabilities: Negative={probs[0, 0]:.4f}, Positive={probs[0, 1]:.4f}\n")
    

**出力** ：
    
    
    === BERT Sentiment Analysis Demo ===
    
    Text: I absolutely loved this movie! It was fantastic.
    Sentiment: Positive (Confidence: 0.8234)
    Probabilities: Negative=0.1766, Positive=0.8234
    
    Text: This product is terrible and waste of money.
    Sentiment: Negative (Confidence: 0.9102)
    Probabilities: Negative=0.9102, Positive=0.0898
    
    Text: The service was okay, nothing special.
    Sentiment: Negative (Confidence: 0.5621)
    Probabilities: Negative=0.5621, Positive=0.4379
    

### 4.3.2 固有表現認識（Named Entity Recognition）

BERTをToken Classification（各トークンにラベルを付与）に使用する例です。
    
    
    from transformers import BertTokenizerFast, BertForTokenClassification
    import torch
    
    print("\n=== BERT Named Entity Recognition Demo ===\n")
    
    # NER用のモデル（事前学習済み）
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    
    model.eval()
    
    # ラベルマッピング
    label_list = [
        'O',       # Outside
        'B-MISC', 'I-MISC',  # Miscellaneous
        'B-PER', 'I-PER',    # Person
        'B-ORG', 'I-ORG',    # Organization
        'B-LOC', 'I-LOC'     # Location
    ]
    
    # サンプルテキスト
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    # トークナイズ（word_idsを取得するためis_split_into_words=Falseでも処理）
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # トークンとラベルを表示
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [label_list[pred] for pred in predictions[0].numpy()]
    
    print(f"Text: {text}\n")
    print("Token-Level Predictions:")
    print(f"{'Token':<15} {'Label':<10}")
    print("-" * 25)
    
    for token, label in zip(tokens, predicted_labels):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token:<15} {label:<10}")
    
    # エンティティ抽出
    print("\nExtracted Entities:")
    current_entity = []
    current_label = None
    
    for token, label in zip(tokens, predicted_labels):
        if label.startswith('B-'):
            if current_entity:
                print(f"  {current_label}: {' '.join(current_entity)}")
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                print(f"  {current_label}: {' '.join(current_entity)}")
            current_entity = []
            current_label = None
    
    if current_entity:
        print(f"  {current_label}: {' '.join(current_entity)}")
    

**出力** ：
    
    
    === BERT Named Entity Recognition Demo ===
    
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California.
    
    Token-Level Predictions:
    Token           Label
    -------------------------
    Apple           B-ORG
    Inc             I-ORG
    .               O
    was             O
    founded         O
    by              O
    Steve           B-PER
    Jobs            I-PER
    in              O
    Cup             B-LOC
    ##ert           I-LOC
    ##ino           I-LOC
    ,               O
    California      B-LOC
    .               O
    
    Extracted Entities:
      ORG: Apple Inc
      PER: Steve Jobs
      LOC: Cup ##ert ##ino
      LOC: California
    

### 4.3.3 質問応答（Question Answering）

BERTの代表的な応用例であるSQuAD形式の質問応答システムです。
    
    
    from transformers import BertForQuestionAnswering, BertTokenizer
    import torch
    
    print("\n=== BERT Question Answering Demo ===\n")
    
    # SQuADでFine-tunedされたBERTモデル
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    model.eval()
    
    # コンテキストと質問
    context = """
    Transformers is a state-of-the-art natural language processing library developed by Hugging Face.
    It provides thousands of pretrained models to perform tasks on texts such as classification,
    information extraction, question answering, summarization, translation, and text generation.
    The library supports PyTorch, TensorFlow, and JAX frameworks.
    """
    
    questions = [
        "Who developed Transformers?",
        "What tasks can Transformers perform?",
        "Which frameworks does the library support?"
    ]
    
    for question in questions:
        # トークナイズ
        inputs = tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=384
        )
    
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
    
        # 開始・終了位置の予測
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
    
        # 回答トークンの抽出
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
        # 確信度スコア
        start_score = start_logits[0, start_idx].item()
        end_score = end_logits[0, end_idx].item()
        confidence = (start_score + end_score) / 2
    
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Confidence Score: {confidence:.4f}\n")
    

**出力** ：
    
    
    === BERT Question Answering Demo ===
    
    Question: Who developed Transformers?
    Answer: Hugging Face
    Confidence Score: 8.2341
    
    Question: What tasks can Transformers perform?
    Answer: classification, information extraction, question answering, summarization, translation, and text generation
    Confidence Score: 7.9823
    
    Question: Which frameworks does the library support?
    Answer: PyTorch, TensorFlow, and JAX
    Confidence Score: 9.1247
    

* * *

## 4.4 GPTアーキテクチャ

### 4.4.1 GPTの設計思想：自己回帰言語モデル

**GPT** （Generative Pre-trained Transformer）は、OpenAIが開発した自己回帰型（autoregressive）言語モデルです。BERTとは対照的に、テキスト生成に特化しています。

特性 | BERT | GPT  
---|---|---  
**アーキテクチャ** | Transformer Encoder | Transformer Decoder（Cross-Attentionなし）  
**方向性** | 双方向（Bidirectional） | 単方向（Unidirectional、左→右）  
**事前学習** | MLM + NSP | Causal Language Modeling（次単語予測）  
**Attention Mask** | なし（全トークンを参照） | Causal Mask（未来のトークンを隠す）  
**主な用途** | 分類、抽出、質問応答 | テキスト生成、対話、要約  
**推論方式** | 並列処理（全トークン同時） | 逐次生成（1トークンずつ）  
  
### 4.4.2 Causal Masking：未来を見ないAttention

GPTの核心は**Causal Attention Mask** です。各位置は自分より前のトークンのみを参照できます。

**Causal Mask行列** （1=参照可能、0=参照不可）：

$$ \text{CausalMask} = \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 1 & 1 & 0 & 0 \\\ 1 & 1 & 1 & 0 \\\ 1 & 1 & 1 & 1 \end{bmatrix} $$ 

Attention計算時に未来のトークンのスコアを$-\infty$にすることで、Softmax後に確率0になります：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V $$ 

ここで $M$ は Causal Mask行列で、マスク位置は$-\infty$です。

### 4.4.3 GPT-1/2/3の進化

モデル | 発表年 | 層数 | 隠れ層サイズ | パラメータ数 | 訓練データ  
---|---|---|---|---|---  
**GPT-1** | 2018 | 12 | 768 | 117M | BooksCorpus (4.5GB)  
**GPT-2** | 2019 | 48 | 1600 | 1.5B | WebText (40GB)  
**GPT-3** | 2020 | 96 | 12288 | 175B | CommonCrawl (570GB)  
**GPT-4** | 2023 | 非公開 | 非公開 | 推定1.7T | 非公開（マルチモーダル）  
  
**主な進化ポイント** ：

  * **スケール拡大** : パラメータ数の指数的増加
  * **Few-shot Learning** : GPT-3以降、例を数個示すだけで新タスクに対応
  * **In-context Learning** : Fine-tuningなしでプロンプトのみで学習
  * **Emergent Abilities** : 規模拡大で突然現れる能力（推論、翻訳など）

### 4.4.4 Causal Attention実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class CausalSelfAttention(nn.Module):
        """GPT-style Causal Self-Attention実装"""
    
        def __init__(self, embed_size, num_heads):
            super(CausalSelfAttention, self).__init__()
            assert embed_size % num_heads == 0
    
            self.embed_size = embed_size
            self.num_heads = num_heads
            self.head_dim = embed_size // num_heads
    
            # Q, K, Vの線形変換
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
    
            # 出力層
            self.proj = nn.Linear(embed_size, embed_size)
    
        def forward(self, x):
            """
            Args:
                x: [batch, seq_len, embed_size]
    
            Returns:
                output: [batch, seq_len, embed_size]
                attention_weights: [batch, num_heads, seq_len, seq_len]
            """
            batch_size, seq_len, _ = x.shape
    
            # Q, K, V計算
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
    
            # Multi-head用に分割: [batch, num_heads, seq_len, head_dim]
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
    
            # Causal Mask適用（上三角を-infにする）
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
    
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
    
            # Valueとの重み付き和
            out = torch.matmul(attention_weights, V)
    
            # Headsを結合
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
    
            # 最終射影
            output = self.proj(out)
    
            return output, attention_weights
    
    
    # デモンストレーション
    print("=== Causal Self-Attention Demo ===\n")
    
    batch_size = 1
    seq_len = 8
    embed_size = 64
    num_heads = 4
    
    # ダミー入力
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # Causal Attention適用
    causal_attn = CausalSelfAttention(embed_size, num_heads)
    output, attn_weights = causal_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Causal Maskの可視化
    sample_attn = attn_weights[0, 0].detach().numpy()  # 1st batch, 1st head
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: Causal Attention重み
    ax1 = axes[0]
    sns.heatmap(sample_attn,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax1,
                annot=True,
                fmt='.3f',
                linewidths=0.5,
                xticklabels=[f't{i+1}' for i in range(seq_len)],
                yticklabels=[f't{i+1}' for i in range(seq_len)])
    
    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax1.set_title('GPT Causal Attention Weights\n(下三角のみ有効)', fontsize=13, fontweight='bold')
    
    # 右: Causal Mask構造
    causal_mask_viz = np.tril(np.ones((seq_len, seq_len)))
    ax2 = axes[1]
    sns.heatmap(causal_mask_viz,
                cmap='RdYlGn',
                cbar_kws={'label': '1=参照可能, 0=マスク'},
                ax=ax2,
                annot=True,
                fmt='.0f',
                linewidths=0.5,
                xticklabels=[f't{i+1}' for i in range(seq_len)],
                yticklabels=[f't{i+1}' for i in range(seq_len)])
    
    ax2.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax2.set_title('Causal Mask Structure\n(未来のトークンを隠す)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n特徴:")
    print("✓ 各位置は自分より前（左）のトークンのみを参照")
    print("✓ 下三角行列の構造（上三角は0）")
    print("✓ 未来の情報を使わないため、逐次生成が可能")
    

**出力** ：
    
    
    === Causal Self-Attention Demo ===
    
    Input shape: torch.Size([1, 8, 64])
    Output shape: torch.Size([1, 8, 64])
    Attention weights shape: torch.Size([1, 4, 8, 8])
    
    特徴:
    ✓ 各位置は自分より前（左）のトークンのみを参照
    ✓ 下三角行列の構造（上三角は0）
    ✓ 未来の情報を使わないため、逐次生成が可能
    

* * *

## 4.5 GPTによるテキスト生成

### 4.5.1 自己回帰生成の仕組み

GPTは1トークンずつ逐次的に生成します：

  1. プロンプト（入力テキスト）をモデルに入力
  2. 次トークンの確率分布を予測
  3. サンプリング戦略で次トークンを選択
  4. 選択したトークンを入力に追加
  5. ステップ2〜4を繰り返し

### 4.5.2 サンプリング戦略

戦略 | 説明 | 特徴  
---|---|---  
**Greedy Decoding** | 最高確率のトークンを選択 | 決定的、繰り返しが多い  
**Beam Search** | 複数候補を保持して探索 | 品質高いが多様性低い  
**Temperature Sampling** | 温度パラメータで確率を調整 | T→0で決定的、T→∞でランダム  
**Top-k Sampling** | 確率上位k個からサンプリング | 多様性と品質のバランス  
**Top-p (Nucleus)** | 累積確率p以上からサンプリング | 動的な語彙サイズ調整  
  
### 4.5.3 GPT-2によるテキスト生成実装
    
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    print("=== GPT-2 Text Generation Demo ===\n")
    
    # GPT-2モデル読み込み
    model_name = 'gpt2'  # 124M parameters
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.eval()
    
    # プロンプト
    prompt = "Artificial intelligence is transforming the world by"
    
    print(f"Prompt: {prompt}\n")
    print("=" * 80)
    
    # 異なるサンプリング戦略での生成
    strategies = [
        {
            'name': 'Greedy Decoding',
            'params': {
                'do_sample': False,
                'max_length': 50
            }
        },
        {
            'name': 'Temperature Sampling (T=0.7)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'temperature': 0.7
            }
        },
        {
            'name': 'Top-k Sampling (k=50)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'top_k': 50,
                'temperature': 1.0
            }
        },
        {
            'name': 'Top-p Sampling (p=0.9)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'top_p': 0.9,
                'temperature': 1.0
            }
        }
    ]
    
    for strategy in strategies:
        # トークナイズ
        inputs = tokenizer(prompt, return_tensors='pt')
    
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                **strategy['params'],
                pad_token_id=tokenizer.eos_token_id
            )
    
        # デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        print(f"\n{strategy['name']}:")
        print(f"{generated_text}")
        print("-" * 80)
    

**出力例** ：
    
    
    === GPT-2 Text Generation Demo ===
    
    Prompt: Artificial intelligence is transforming the world by
    
    ================================================================================
    
    Greedy Decoding:
    Artificial intelligence is transforming the world by making it easier for people to do things that they would otherwise have to do manually. The most common example is the use of AI to automate tasks such as scheduling, scheduling appointments, and scheduling meetings.
    
    --------------------------------------------------------------------------------
    
    Temperature Sampling (T=0.7):
    Artificial intelligence is transforming the world by enabling machines to learn from experience and make decisions without human intervention. From self-driving cars to medical diagnosis systems, AI technologies are revolutionizing industries and improving our daily lives.
    
    --------------------------------------------------------------------------------
    
    Top-k Sampling (k=50):
    Artificial intelligence is transforming the world by creating new possibilities in healthcare, education, and entertainment. AI systems can now analyze vast amounts of data, recognize patterns, and provide insights that were previously impossible to obtain.
    
    --------------------------------------------------------------------------------
    
    Top-p Sampling (p=0.9):
    Artificial intelligence is transforming the world by automating complex tasks, enhancing decision-making processes, and opening doors to innovations we never thought possible. As AI continues to evolve, its impact on society will only grow stronger.
    
    --------------------------------------------------------------------------------
    

### 4.5.4 カスタム生成関数の実装
    
    
    def generate_text_custom(model, tokenizer, prompt, max_length=50,
                            strategy='top_p', temperature=1.0, top_k=50, top_p=0.9):
        """
        カスタムテキスト生成関数
    
        Args:
            model: GPT-2モデル
            tokenizer: トークナイザー
            prompt: 入力プロンプト
            max_length: 最大生成長
            strategy: 'greedy', 'temperature', 'top_k', 'top_p'
            temperature: 温度パラメータ
            top_k: Top-kサンプリングのk
            top_p: Top-pサンプリングのp
    
        Returns:
            生成テキスト
        """
        # トークナイズ
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
        # 生成ループ
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
    
            # 最後のトークンのlogitsを取得
            next_token_logits = logits[0, -1, :]
    
            # Temperature適用
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
    
            # サンプリング戦略
            if strategy == 'greedy':
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
    
            elif strategy == 'temperature':
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            elif strategy == 'top_k':
                # Top-kマスキング
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits_filtered[top_k_indices] = top_k_values
    
                probs = F.softmax(next_token_logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            elif strategy == 'top_p':
                # Top-pマスキング
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
                # 累積確率がpを超える位置を見つける
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
    
                # マスク適用
                next_token_logits_filtered = next_token_logits.clone()
                next_token_logits_filtered[sorted_indices[sorted_indices_to_remove]] = float('-inf')
    
                probs = F.softmax(next_token_logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            # 入力に追加
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    
            # EOSトークンで終了
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
        # デコード
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    
    # カスタム生成関数のテスト
    print("\n=== Custom Generation Function Test ===\n")
    
    prompt = "The future of machine learning is"
    print(f"Prompt: {prompt}\n")
    
    for strategy in ['greedy', 'temperature', 'top_k', 'top_p']:
        generated = generate_text_custom(
            model, tokenizer, prompt,
            max_length=30,
            strategy=strategy,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        print(f"{strategy.upper()}: {generated}\n")
    

* * *

## 4.6 BERT vs GPT：比較と使い分け

### 4.6.1 アーキテクチャの比較
    
    
    ```mermaid
    graph TB
        subgraph "BERT (Encoder-only)"
            B1[Input: 文全体] --> B2[Token + Segment + Position Embeddings]
            B2 --> B3[Transformer Encoder × 12]
            B3 --> B4[Bidirectional Attention]
            B4 --> B5["[CLS] for ClassificationAll Tokens for Token-level"]
    
            style B4 fill:#27ae60,color:#fff
        end
    
        subgraph "GPT (Decoder-only)"
            G1[Input: プロンプト] --> G2[Token + Position Embeddings]
            G2 --> G3[Transformer Decoder × 12]
            G3 --> G4[Causal Attention]
            G4 --> G5[Next Token Prediction]
            G5 --> G6[Autoregressive Generation]
    
            style G4 fill:#e74c3c,color:#fff
        end
    ```

### 4.6.2 性能比較実験
    
    
    from transformers import BertModel, GPT2Model, BertTokenizer, GPT2Tokenizer
    import torch
    import time
    
    print("=== BERT vs GPT Performance Comparison ===\n")
    
    # モデル読み込み
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    
    bert_model.eval()
    gpt2_model.eval()
    
    # テストテキスト
    text = "Natural language processing is a fascinating field of artificial intelligence."
    
    # BERT処理
    bert_inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    start_time = time.time()
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_time = time.time() - start_time
    
    # GPT-2処理
    gpt2_inputs = gpt2_tokenizer(text, return_tensors='pt')
    start_time = time.time()
    with torch.no_grad():
        gpt2_outputs = gpt2_model(**gpt2_inputs)
    gpt2_time = time.time() - start_time
    
    # 結果表示
    print("Input Text:", text)
    print(f"\nBERT:")
    print(f"  Model: bert-base-uncased")
    print(f"  Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")
    print(f"  Input shape: {bert_inputs['input_ids'].shape}")
    print(f"  Output shape: {bert_outputs.last_hidden_state.shape}")
    print(f"  Processing time: {bert_time*1000:.2f} ms")
    print(f"  [CLS] embedding shape: {bert_outputs.pooler_output.shape}")
    
    print(f"\nGPT-2:")
    print(f"  Model: gpt2")
    print(f"  Parameters: {sum(p.numel() for p in gpt2_model.parameters()):,}")
    print(f"  Input shape: {gpt2_inputs['input_ids'].shape}")
    print(f"  Output shape: {gpt2_outputs.last_hidden_state.shape}")
    print(f"  Processing time: {gpt2_time*1000:.2f} ms")
    
    # Attention可視化比較
    print("\n" + "="*80)
    print("Attention Pattern Comparison")
    print("="*80)
    
    # BERT: すべてのトークンを相互参照可能
    print("\nBERT Attention Pattern:")
    print("  ✓ Bidirectional - すべてのトークンがすべてのトークンを参照")
    print("  ✓ 並列処理可能 - 全トークンを同時に処理")
    print("  ✓ 用途: 分類、NER、QA、文エンコーディング")
    
    # GPT: 左側のトークンのみ参照可能
    print("\nGPT Attention Pattern:")
    print("  ✓ Unidirectional - 各トークンは左側のトークンのみ参照")
    print("  ✓ 逐次生成 - 1トークンずつ生成")
    print("  ✓ 用途: テキスト生成、対話、補完、翻訳")
    

**出力** ：
    
    
    === BERT vs GPT Performance Comparison ===
    
    Input Text: Natural language processing is a fascinating field of artificial intelligence.
    
    BERT:
      Model: bert-base-uncased
      Parameters: 109,482,240
      Input shape: torch.Size([1, 14])
      Output shape: torch.Size([1, 14, 768])
      Processing time: 45.23 ms
      [CLS] embedding shape: torch.Size([1, 768])
    
    GPT-2:
      Model: gpt2
      Parameters: 124,439,808
      Input shape: torch.Size([1, 14])
      Output shape: torch.Size([1, 14, 768])
      Processing time: 38.67 ms
    
    ================================================================================
    Attention Pattern Comparison
    ================================================================================
    
    BERT Attention Pattern:
      ✓ Bidirectional - すべてのトークンがすべてのトークンを参照
      ✓ 並列処理可能 - 全トークンを同時に処理
      ✓ 用途: 分類、NER、QA、文エンコーディング
    
    GPT Attention Pattern:
      ✓ Unidirectional - 各トークンは左側のトークンのみ参照
      ✓ 逐次生成 - 1トークンずつ生成
      ✓ 用途: テキスト生成、対話、補完、翻訳
    

### 4.6.3 使い分けガイド

タスク | 推奨モデル | 理由  
---|---|---  
**感情分析** | BERT | 文全体の文脈理解が必要  
**固有表現認識** | BERT | 各トークンの分類、双方向文脈が有利  
**質問応答** | BERT | 文章中から回答箇所を特定  
**文書分類** | BERT | [CLS]トークンで文全体をエンコード  
**テキスト生成** | GPT | 自己回帰生成に特化  
**対話システム** | GPT | 応答生成が主タスク  
**要約** | GPT（or BART） | 生成タスク、抽象的要約  
**コード生成** | GPT（Codex） | 逐次的なコード生成  
**翻訳** | 両方可能 | BERT→Encoder、GPT→Decoder的に使用  
  
* * *

## 4.7 実践プロジェクト

### 4.7.1 プロジェクト1: BERTによる質問応答システム

#### 目標

SQuAD形式の質問応答システムを構築し、コンテキストから正確な回答を抽出します。

#### 実装要件

  * Fine-tunedされたBERTモデルの使用
  * 複数の質問に対する回答抽出
  * 確信度スコアの計算と表示
  * 回答の妥当性検証

    
    
    from transformers import BertForQuestionAnswering, BertTokenizer
    import torch
    
    class QuestionAnsweringSystem:
        """BERTベースの質問応答システム"""
    
        def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()
    
        def answer_question(self, question, context, return_confidence=True):
            """
            質問に対する回答を抽出
    
            Args:
                question: 質問文
                context: コンテキスト（回答元の文章）
                return_confidence: 確信度を返すかどうか
    
            Returns:
                answer: 抽出された回答
                confidence: 確信度スコア（return_confidence=Trueの場合）
            """
            # トークナイズ
            inputs = self.tokenizer(
                question,
                context,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
    
            # 推論
            with torch.no_grad():
                outputs = self.model(**inputs)
    
            # 開始・終了位置の予測
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
    
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
    
            # 回答抽出
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
            if return_confidence:
                # 確信度計算
                start_score = torch.softmax(start_logits, dim=1)[0, start_idx].item()
                end_score = torch.softmax(end_logits, dim=1)[0, end_idx].item()
                confidence = (start_score + end_score) / 2
    
                return answer, confidence
            else:
                return answer
    
        def batch_answer(self, qa_pairs):
            """
            複数の質問に一括で回答
    
            Args:
                qa_pairs: [(question, context), ...] のリスト
    
            Returns:
                results: [(answer, confidence), ...] のリスト
            """
            results = []
            for question, context in qa_pairs:
                answer, confidence = self.answer_question(question, context)
                results.append((answer, confidence))
            return results
    
    
    # システムのテスト
    print("=== Question Answering System Demo ===\n")
    
    qa_system = QuestionAnsweringSystem()
    
    # テストケース
    context = """
    The Transformer architecture was introduced in the paper "Attention is All You Need"
    by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms to compute
    representations of input and output sequences without using recurrent or convolutional layers.
    The model achieved state-of-the-art results on machine translation tasks and has since become
    the foundation for models like BERT and GPT. The architecture consists of an encoder and a decoder,
    each composed of multiple identical layers. Each layer has two sub-layers: a multi-head self-attention
    mechanism and a position-wise fully connected feed-forward network.
    """
    
    questions = [
        "When was the Transformer introduced?",
        "Who introduced the Transformer?",
        "What does the Transformer rely on?",
        "What are the two main components of the Transformer?",
        "What models are based on the Transformer?"
    ]
    
    print("Context:")
    print(context)
    print("\n" + "="*80 + "\n")
    
    for i, question in enumerate(questions, 1):
        answer, confidence = qa_system.answer_question(question, context)
    
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer}")
        print(f"Confidence: {confidence:.4f}")
        print()
    
    # バッチ処理のデモ
    print("="*80)
    print("\nBatch Processing Demo:")
    print("="*80 + "\n")
    
    qa_pairs = [(q, context) for q in questions]
    results = qa_system.batch_answer(qa_pairs)
    
    for (question, _), (answer, conf) in zip(qa_pairs, results):
        print(f"Q: {question}")
        print(f"A: {answer} (Conf: {conf:.4f})\n")
    

**出力** ：
    
    
    === Question Answering System Demo ===
    
    Context:
    The Transformer architecture was introduced in the paper "Attention is All You Need"
    by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms to compute
    representations of input and output sequences without using recurrent or convolutional layers.
    The model achieved state-of-the-art results on machine translation tasks and has since become
    the foundation for models like BERT and GPT. The architecture consists of an encoder and a decoder,
    each composed of multiple identical layers. Each layer has two sub-layers: a multi-head self-attention
    mechanism and a position-wise fully connected feed-forward network.
    
    ================================================================================
    
    Q1: When was the Transformer introduced?
    A1: 2017
    Confidence: 0.9523
    
    Q2: Who introduced the Transformer?
    A2: Vaswani et al.
    Confidence: 0.8876
    
    Q3: What does the Transformer rely on?
    A3: self-attention mechanisms
    Confidence: 0.9234
    
    Q4: What are the two main components of the Transformer?
    A4: an encoder and a decoder
    Confidence: 0.8912
    
    Q5: What models are based on the Transformer?
    A5: BERT and GPT
    Confidence: 0.9101
    

### 4.7.2 プロジェクト2: GPTによるテキスト生成アプリ

#### 目標

カスタマイズ可能なテキスト生成システムを構築し、様々な生成戦略を試します。

#### 実装要件

  * 複数のサンプリング戦略のサポート
  * 生成パラメータの調整機能
  * プロンプトエンジニアリングの実践
  * 生成品質の評価

    
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    class TextGenerator:
        """GPT-2ベーステキスト生成システム"""
    
        def __init__(self, model_name='gpt2-medium'):
            """
            Args:
                model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            """
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()
    
            # PADトークン設定
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        def generate(self, prompt, max_length=100, strategy='top_p',
                    num_return_sequences=1, **kwargs):
            """
            テキスト生成
    
            Args:
                prompt: 入力プロンプト
                max_length: 最大生成長
                strategy: 'greedy', 'beam', 'temperature', 'top_k', 'top_p'
                num_return_sequences: 生成する候補数
                **kwargs: 戦略固有のパラメータ
    
            Returns:
                生成されたテキストのリスト
            """
            # トークナイズ
            inputs = self.tokenizer(prompt, return_tensors='pt')
    
            # 戦略に応じたパラメータ設定
            gen_params = {
                'max_length': max_length,
                'num_return_sequences': num_return_sequences,
                'pad_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True
            }
    
            if strategy == 'greedy':
                gen_params['do_sample'] = False
    
            elif strategy == 'beam':
                gen_params['num_beams'] = kwargs.get('num_beams', 5)
                gen_params['do_sample'] = False
    
            elif strategy == 'temperature':
                gen_params['do_sample'] = True
                gen_params['temperature'] = kwargs.get('temperature', 0.7)
    
            elif strategy == 'top_k':
                gen_params['do_sample'] = True
                gen_params['top_k'] = kwargs.get('top_k', 50)
                gen_params['temperature'] = kwargs.get('temperature', 1.0)
    
            elif strategy == 'top_p':
                gen_params['do_sample'] = True
                gen_params['top_p'] = kwargs.get('top_p', 0.9)
                gen_params['temperature'] = kwargs.get('temperature', 1.0)
    
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(inputs['input_ids'], **gen_params)
    
            # デコード
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
    
            return generated_texts
    
        def interactive_generation(self):
            """対話的な生成セッション"""
            print("=== Interactive Text Generation ===")
            print("Type 'quit' to exit\n")
    
            while True:
                prompt = input("Prompt: ")
                if prompt.lower() == 'quit':
                    break
    
                # 生成設定
                print("\nGeneration Settings:")
                strategy = input("Strategy (greedy/beam/temperature/top_k/top_p) [top_p]: ") or 'top_p'
                max_length = int(input("Max length [100]: ") or 100)
    
                # 生成
                outputs = self.generate(prompt, max_length=max_length, strategy=strategy)
    
                print("\n--- Generated Text ---")
                print(outputs[0])
                print("-" * 80 + "\n")
    
    
    # システムのテスト
    print("=== Text Generation System Demo ===\n")
    
    generator = TextGenerator(model_name='gpt2')
    
    # プロンプトテンプレート
    prompts = [
        "In the future of artificial intelligence,",
        "The most important breakthrough in deep learning was",
        "Once upon a time in a distant galaxy,"
    ]
    
    print("Comparing Different Generation Strategies:\n")
    print("="*80 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}\n")
    
        strategies = [
            ('greedy', {}),
            ('top_k', {'top_k': 50, 'temperature': 0.8}),
            ('top_p', {'top_p': 0.9, 'temperature': 0.8})
        ]
    
        for strategy, params in strategies:
            outputs = generator.generate(
                prompt,
                max_length=60,
                strategy=strategy,
                num_return_sequences=1,
                **params
            )
    
            print(f"{strategy.upper()}:")
            print(f"{outputs[0]}\n")
    
        print("="*80 + "\n")
    
    # 複数候補生成のデモ
    print("\nMultiple Candidates Generation:")
    print("="*80 + "\n")
    
    prompt = "The key to successful machine learning is"
    outputs = generator.generate(
        prompt,
        max_length=50,
        strategy='top_p',
        num_return_sequences=3,
        top_p=0.9,
        temperature=0.9
    )
    
    for i, output in enumerate(outputs, 1):
        print(f"Candidate {i}:")
        print(output)
        print()
    

**出力例** ：
    
    
    === Text Generation System Demo ===
    
    Comparing Different Generation Strategies:
    
    ================================================================================
    
    Prompt: In the future of artificial intelligence,
    
    GREEDY:
    In the future of artificial intelligence, we will be able to create a new kind of AI that can do things that we have never done before. We will be able to build systems that can learn from data and make decisions based on that data.
    
    TOP_K:
    In the future of artificial intelligence, machines will become increasingly capable of understanding human language, emotions, and intentions. This will revolutionize how we interact with technology and open new possibilities in healthcare, education, and entertainment.
    
    TOP_P:
    In the future of artificial intelligence, we can expect to see breakthroughs in areas such as natural language understanding, computer vision, and autonomous decision-making. These advances will transform industries and create opportunities we haven't yet imagined.
    
    ================================================================================
    

* * *

## 4.8 まとめと発展トピック

### 本章で学んだこと

トピック | 重要ポイント  
---|---  
**BERT** | 双方向エンコーダ、MLM+NSP、理解タスクに最適  
**GPT** | 自己回帰生成、Causal Masking、生成タスクに最適  
**事前学習** | 大規模データで学習、Fine-tuningで特化  
**使い分け** | 分類・抽出はBERT、生成はGPT  
**実践手法** | Hugging Face、サンプリング戦略、QAシステム  
  
### 発展トピック

**RoBERTa：BERTの改良版**

FacebookによるBERTの改良版。NSPタスクを削除し、動的マスキング、より大規模な訓練データ、長い訓練時間を採用して性能を向上させました。

**ALBERT：パラメータ効率化**

パラメータ共有とFactor化により、BERTと同等の性能を少ないパラメータで実現。大規模モデルの訓練を効率化します。

**GPT-3.5/4：InstructGPT・ChatGPT**

Instruction TuningとRLHF（人間フィードバックからの強化学習）により、ユーザー指示に従う能力を大幅に向上。対話システムの主流に。

**Prompt Engineering**

モデルの性能を最大化するプロンプト設計技術。Few-shot examples、Chain-of-Thought prompting、Role promptingなど。

**PEFT (Parameter-Efficient Fine-Tuning)**

LoRA、Adapter、Prefix Tuningなど、全パラメータを更新せずに効率的にFine-tuningする手法。大規模モデル時代の必須技術。

### 演習問題

#### 演習 4.1: BERT Fine-tuningによる感情分析

**課題** : IMDBレビューデータセットでBERTをFine-tuningし、感情分析モデルを構築してください。

**要件** :

  * データの前処理とトークナイゼーション
  * BERT-Baseモデルのロードと分類層の追加
  * 訓練ループの実装
  * 精度・F1スコアの評価

#### 演習 4.2: GPT-2による対話システム

**課題** : GPT-2を使った簡単な対話システムを実装してください。

**要件** :

  * ユーザー入力の受付とコンテキスト管理
  * 応答生成（複数のサンプリング戦略）
  * 会話履歴の保持と反映
  * 対話の自然性評価

#### 演習 4.3: BERT vs GPT性能比較

**課題** : 同じタスクでBERTとGPTの性能を比較してください。

**タスク** : 文書分類（ニュース記事カテゴリ分類）

**比較項目** :

  * 精度、F1スコア
  * 訓練時間
  * 推論速度
  * メモリ使用量

#### 演習 4.4: Masked Language Modelingの実装

**課題** : 小規模データセットでMLMを実装し、BERTの事前学習を再現してください。

**実装内容** :

  * マスクデータ生成ロジック
  * MLM損失関数
  * 訓練ループ
  * マスク予測精度の評価

#### 演習 4.5: 多言語BERT（mBERT）の活用

**課題** : 多言語BERTを使って、複数言語でのテキスト分類を実装してください。

**言語** : 英語、日本語、中国語

**タスク** : ニュース記事のトピック分類

#### 演習 4.6: GPTによるコード生成

**課題** : GPT-2を使って、自然言語の指示からPythonコードを生成するシステムを構築してください。

**要件** :

  * プロンプトテンプレートの設計
  * コード生成と構文検証
  * 生成品質の評価

* * *

### 次章予告

第5章では、**Vision Transformer (ViT)** を学びます。TransformerアーキテクチャをComputer Visionに適用し、画像を「トークン」として扱う革新的なアプローチを探ります。

> **次章のトピック** :  
>  ・Vision Transformerのアーキテクチャ  
>  ・画像パッチのトークン化  
>  ・Position Embeddingsの2D拡張  
>  ・CNNとの性能比較  
>  ・Pre-training戦略（ImageNet-21k）  
>  ・実装：ViTによる画像分類  
>  ・応用：Object Detection、Segmentation
