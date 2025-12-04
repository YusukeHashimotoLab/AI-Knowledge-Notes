---
title: 第3章：TransformerとBERT
chapter_title: 第3章：TransformerとBERT
subtitle: 注意機構から事前学習済みモデルまで - 自然言語処理の革命
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Transformerアーキテクチャの仕組みを理解する
  * ✅ Self-AttentionとMulti-Head Attentionを実装できる
  * ✅ Positional Encodingの必要性を説明できる
  * ✅ BERTの事前学習とファインチューニングを実行できる
  * ✅ HuggingFace Transformersライブラリを使いこなせる
  * ✅ 日本語BERTモデルを実務で活用できる

* * *

## 3.1 Transformerアーキテクチャ

### Transformerの誕生

**Transformer** は、2017年にGoogleが発表した「Attention is All You Need」論文で提案されたアーキテクチャです。RNNやLSTMを使わず、**Self-Attention機構** のみで系列処理を実現しました。

> 「RNNの逐次処理を排除し、全トークン間の関係を並列計算する」

### Transformerの利点

項目 | RNN/LSTM | Transformer  
---|---|---  
**並列化** | 逐次処理（遅い） | 完全並列（速い）  
**長距離依存** | 勾配消失で困難 | 直接接続で容易  
**計算複雑度** | O(n) | O(n²)  
**解釈性** | 低い | Attention可視化で高い  
  
### 全体アーキテクチャ
    
    
    ```mermaid
    graph TB
        A[入力文] --> B[Input Embedding]
        B --> C[Positional Encoding]
        C --> D[Encoder Stack]
        D --> E[Decoder Stack]
        E --> F[Linear + Softmax]
        F --> G[出力文]
    
        D --> |Context| E
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#fff9c4
        style G fill:#e0f2f1
    ```

* * *

## 3.2 Self-Attention機構

### Self-Attentionの原理

**Self-Attention（自己注意）** は、入力系列内の各トークンが他のすべてのトークンとの関係を計算する機構です。

3つの重み行列を使用します：

  * $\mathbf{W}_Q$: Query（クエリ）行列
  * $\mathbf{W}_K$: Key（キー）行列
  * $\mathbf{W}_V$: Value（値）行列

### 計算手順

**ステップ1: Query、Key、Valueを計算**

$$ \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V $$

**ステップ2: Attention Scoreを計算**

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} $$

  * $d_k$: Keyの次元数（スケーリング因子）

### 実装例：Scaled Dot-Product Attention
    
    
    import numpy as np
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention
    
        Args:
            Q: Query行列 (batch_size, seq_len, d_k)
            K: Key行列 (batch_size, seq_len, d_k)
            V: Value行列 (batch_size, seq_len, d_v)
            mask: マスク (オプション)
    
        Returns:
            output: Attention適用後の出力
            attention_weights: Attention重み
        """
        d_k = Q.shape[-1]
    
        # Attention Score計算
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
        # マスク適用（オプション）
        if mask is not None:
            scores = scores + (mask * -1e9)
    
        # Softmax
        attention_weights = softmax(scores, axis=-1)
    
        # 重み付き和
        output = np.matmul(attention_weights, V)
    
        return output, attention_weights
    
    def softmax(x, axis=-1):
        """Softmax関数"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # 使用例
    batch_size, seq_len, d_model = 2, 5, 64
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"出力形状: {output.shape}")
    print(f"Attention重み形状: {weights.shape}")
    print(f"\nAttention重み（最初のサンプル）:\n{weights[0]}")
    

**出力** ：
    
    
    出力形状: (2, 5, 64)
    Attention重み形状: (2, 5, 5)
    
    Attention重み（最初のサンプル）:
    [[0.21 0.19 0.20 0.18 0.22]
     [0.20 0.21 0.19 0.20 0.20]
     [0.19 0.20 0.21 0.20 0.20]
     [0.20 0.20 0.19 0.21 0.20]
     [0.22 0.18 0.20 0.19 0.21]]
    

### PyTorchによる実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ScaledDotProductAttention(nn.Module):
        def __init__(self, d_k):
            super().__init__()
            self.d_k = d_k
    
        def forward(self, Q, K, V, mask=None):
            # Attention Score
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    
            # マスク適用
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
    
            # 重み付き和
            output = torch.matmul(attention_weights, V)
    
            return output, attention_weights
    
    # 使用例
    d_model = 64
    attention = ScaledDotProductAttention(d_k=d_model)
    
    Q = torch.randn(2, 5, d_model)
    K = torch.randn(2, 5, d_model)
    V = torch.randn(2, 5, d_model)
    
    output, weights = attention(Q, K, V)
    print(f"出力形状: {output.shape}")
    print(f"Attention重み形状: {weights.shape}")
    

* * *

## 3.3 Multi-Head Attention

### 概要

**Multi-Head Attention** は、複数のAttention headを並列に実行し、異なる表現部分空間から情報を捉えます。
    
    
    ```mermaid
    graph LR
        A[入力 X] --> B1[Head 1]
        A --> B2[Head 2]
        A --> B3[Head 3]
        A --> B4[Head h]
    
        B1 --> C[Concat]
        B2 --> C
        B3 --> C
        B4 --> C
    
        C --> D[Linear]
        D --> E[出力]
    
        style A fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style B3 fill:#fff3e0
        style B4 fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#e0f2f1
    ```

### 数式

$$ \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O $$

各headは：

$$ \text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V) $$

### 実装例
    
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            """
            Multi-Head Attention
    
            Args:
                d_model: モデルの次元数
                num_heads: Attention headの数
            """
            super().__init__()
            assert d_model % num_heads == 0
    
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
    
            # Linear層
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)
            self.W_O = nn.Linear(d_model, d_model)
    
            self.attention = ScaledDotProductAttention(self.d_k)
    
        def split_heads(self, x, batch_size):
            """複数headに分割"""
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)
    
            # Linear変換
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
    
            # 複数headに分割
            Q = self.split_heads(Q, batch_size)
            K = self.split_heads(K, batch_size)
            V = self.split_heads(V, batch_size)
    
            # Attention適用
            output, attention_weights = self.attention(Q, K, V, mask)
    
            # Concat
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, -1, self.d_model)
    
            # 最終Linear層
            output = self.W_O(output)
    
            return output, attention_weights
    
    # 使用例
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = mha(x, x, x)
    print(f"出力形状: {output.shape}")
    print(f"Attention重み形状: {weights.shape}")
    

**出力** ：
    
    
    出力形状: torch.Size([2, 10, 512])
    Attention重み形状: torch.Size([2, 8, 10, 10])
    

* * *

## 3.4 Positional Encoding

### 必要性

Self-Attentionは順序情報を持たないため、**Positional Encoding（位置エンコーディング）** で系列の位置情報を追加します。

### Sinusoidal Positional Encoding

$$ \text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

$$ \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

  * $pos$: トークンの位置
  * $i$: 次元のインデックス

### 実装例
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            """
            Positional Encoding
    
            Args:
                d_model: モデルの次元数
                max_len: 最大系列長
            """
            super().__init__()
    
            # Positional Encodingを事前計算
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            """
            Args:
                x: (batch_size, seq_len, d_model)
            """
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
    
    # 使用例
    d_model = 128
    max_len = 100
    
    pe_layer = PositionalEncoding(d_model, max_len)
    x = torch.randn(2, 50, d_model)
    output = pe_layer(x)
    
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {output.shape}")
    
    # Positional Encodingの可視化
    pe_matrix = pe_layer.pe[0, :max_len, :].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Dimension', fontsize=12)
    plt.title('Positional Encoding (Sinusoidal)', fontsize=14)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Feed-Forward Networks

### Position-wise Feed-Forward Networks

各位置に独立に適用される2層のニューラルネットワークです：

$$ \text{FFN}(x) = \max(0, x\mathbf{W}_1 + b_1)\mathbf{W}_2 + b_2 $$

### 実装例
    
    
    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            """
            Position-wise Feed-Forward Networks
    
            Args:
                d_model: モデルの次元数
                d_ff: 中間層の次元数（通常 4 * d_model）
                dropout: ドロップアウト率
            """
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
    
        def forward(self, x):
            # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
    
            # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
            x = self.linear2(x)
            return x
    
    # 使用例
    d_model = 512
    d_ff = 2048
    
    ffn = PositionwiseFeedForward(d_model, d_ff)
    x = torch.randn(2, 10, d_model)
    
    output = ffn(x)
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {output.shape}")
    

* * *

## 3.6 BERT（Bidirectional Encoder Representations from Transformers）

### BERTの特徴

**BERT** は、2018年にGoogleが発表した双方向の事前学習モデルです。

> 「左から右だけでなく、双方向のコンテキストを同時に学習する」
    
    
    ```mermaid
    graph LR
        A[大規模コーパス] --> B[事前学習]
        B --> C[BERT Base/Large]
        C --> D1[ファインチューニング: 分類]
        C --> D2[ファインチューニング: NER]
        C --> D3[ファインチューニング: QA]
        C --> D4[特徴抽出]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
        style D4 fill:#e8f5e9
    ```

### BERTのモデル構成

モデル | 層数 | Hidden Size | Attention Heads | パラメータ数  
---|---|---|---|---  
**BERT-Base** | 12 | 768 | 12 | 110M  
**BERT-Large** | 24 | 1024 | 16 | 340M  
  
### 事前学習タスク

#### 1\. Masked Language Modeling (MLM)

入力の15%のトークンをマスク（[MASK]）し、予測します。
    
    
    入力:  The [MASK] is beautiful today.
    目標:  The weather is beautiful today.
    

#### 2\. Next Sentence Prediction (NSP)

2つの文が連続しているかを予測します。
    
    
    文A: The cat sat on the mat.
    文B: It was very comfortable.
    ラベル: IsNext (1)
    
    文A: The cat sat on the mat.
    文B: The economy is growing.
    ラベル: NotNext (0)
    

### HuggingFace Transformersで始めるBERT
    
    
    from transformers import BertTokenizer, BertModel
    import torch
    
    # トークナイザとモデルのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # テキストのエンコード
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    print("トークン:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    print("入力ID:", inputs['input_ids'])
    print("Attention Mask:", inputs['attention_mask'])
    
    # モデルの推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 出力
    last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    pooler_output = outputs.pooler_output  # (batch_size, hidden_size) [CLS]トークンの出力
    
    print(f"\nLast Hidden States形状: {last_hidden_states.shape}")
    print(f"Pooler Output形状: {pooler_output.shape}")
    

**出力** ：
    
    
    トークン: ['[CLS]', 'hello', ',', 'how', 'are', 'you', '?', '[SEP]']
    入力ID: tensor([[  101,  7592,  1010,  2129,  2024,  2017,  1029,   102]])
    Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    
    Last Hidden States形状: torch.Size([1, 8, 768])
    Pooler Output形状: torch.Size([1, 768])
    

* * *

## 3.7 BERTのファインチューニング

### テキスト分類タスク
    
    
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    from datasets import load_dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    # データセットのロード（例: IMDb映画レビュー）
    dataset = load_dataset('imdb')
    
    # トークナイザとモデル
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # データの前処理
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 小規模サブセットでテスト
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(200))
    
    # 評価関数
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
    
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
    
        return {'accuracy': acc, 'f1': f1}
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
    )
    
    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # ファインチューニング
    trainer.train()
    
    # 評価
    results = trainer.evaluate()
    print(f"\n評価結果: {results}")
    

### Named Entity Recognition (NER)
    
    
    from transformers import BertForTokenClassification, pipeline
    
    # NER用モデルのロード
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    ner_pipeline = pipeline('ner', model=model_name, tokenizer=model_name)
    
    # テキストの固有表現抽出
    text = "Apple Inc. is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO."
    results = ner_pipeline(text)
    
    print("固有表現抽出結果:")
    for entity in results:
        print(f"{entity['word']}: {entity['entity']} (信頼度: {entity['score']:.4f})")
    

**出力** ：
    
    
    固有表現抽出結果:
    Apple: B-ORG (信頼度: 0.9987)
    Inc: I-ORG (信頼度: 0.9983)
    U: B-LOC (信頼度: 0.9976)
    K: I-LOC (信頼度: 0.9945)
    Tim: B-PER (信頼度: 0.9995)
    Cook: I-PER (信頼度: 0.9993)
    CEO: B-MISC (信頼度: 0.8734)
    

### Question Answering
    
    
    from transformers import pipeline
    
    # QA用モデルのロード
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # コンテキストと質問
    context = """
    The Transformer is a deep learning model introduced in 2017, used primarily in the field of
    natural language processing (NLP). Like recurrent neural networks (RNNs), Transformers are
    designed to handle sequential data, such as natural language, for tasks such as translation
    and text summarization. However, unlike RNNs, Transformers do not require that the sequential
    data be processed in order.
    """
    
    question = "When was the Transformer introduced?"
    
    # 質問応答
    result = qa_pipeline(question=question, context=context)
    
    print(f"質問: {question}")
    print(f"回答: {result['answer']}")
    print(f"信頼度: {result['score']:.4f}")
    print(f"開始位置: {result['start']}, 終了位置: {result['end']}")
    

**出力** ：
    
    
    質問: When was the Transformer introduced?
    回答: 2017
    信頼度: 0.9812
    開始位置: 50, 終了位置: 54
    

* * *

## 3.8 日本語BERTモデル

### 代表的な日本語BERTモデル

モデル | 提供元 | トークナイザ | 特徴  
---|---|---|---  
**東北大BERT** | 東北大学 | MeCab + WordPiece | 日本語Wikipediaで学習  
**京都大BERT** | 京都大学 | Juman++ + WordPiece | 高品質な形態素解析  
**NICT BERT** | NICT | SentencePiece | 大規模コーパス  
**早稲田RoBERTa** | 早稲田大学 | SentencePiece | RoBERTa（NSPなし）  
  
### 東北大BERTの使用例
    
    
    from transformers import BertJapaneseTokenizer, BertModel
    import torch
    
    # 東北大BERTのロード
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 日本語テキスト
    text = "自然言語処理は人工知能の重要な分野です。"
    
    # トークナイズ
    inputs = tokenizer(text, return_tensors='pt')
    print("トークン:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nLast Hidden States形状: {outputs.last_hidden_state.shape}")
    print(f"Pooler Output形状: {outputs.pooler_output.shape}")
    

### 日本語テキスト分類
    
    
    from transformers import BertForSequenceClassification, BertJapaneseTokenizer
    import torch
    import torch.nn.functional as F
    
    # モデルとトークナイザのロード
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    
    # 感情分析用にカスタマイズ（例: ポジティブ/ネガティブ）
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # テキスト例
    texts = [
        "この映画は本当に素晴らしかった！",
        "全く面白くなくて時間の無駄だった。",
        "普通の作品で特に印象に残らなかった。"
    ]
    
    # 推論
    model.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
    
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
        label = "ポジティブ" if predicted_class == 1 else "ネガティブ"
        print(f"\nテキスト: {text}")
        print(f"予測: {label} (信頼度: {confidence:.4f})")
    

* * *

## 3.9 BERTの応用テクニック

### Feature Extraction（特徴抽出）

BERTを固定の特徴抽出器として使用します。
    
    
    from transformers import BertModel, BertTokenizer
    import torch
    import numpy as np
    
    # モデルとトークナイザ
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # 評価モード
    
    def get_sentence_embedding(text, pooling='mean'):
        """
        文の埋め込みベクトルを取得
    
        Args:
            text: 入力テキスト
            pooling: プーリング方法 ('mean', 'max', 'cls')
    
        Returns:
            embedding: 埋め込みベクトル
        """
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Last Hidden States
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    
        if pooling == 'mean':
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        elif pooling == 'max':
            # Max pooling
            embedding = torch.max(last_hidden, 1)[0]
        elif pooling == 'cls':
            # [CLS]トークン
            embedding = outputs.pooler_output
    
        return embedding.squeeze().numpy()
    
    # 使用例
    texts = [
        "Natural language processing is fascinating.",
        "I love machine learning.",
        "The weather is nice today."
    ]
    
    embeddings = [get_sentence_embedding(text, pooling='mean') for text in texts]
    
    # コサイン類似度計算
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings)
    
    print("コサイン類似度行列:")
    print(similarity_matrix)
    print(f"\n文1と文2の類似度: {similarity_matrix[0, 1]:.4f}")
    print(f"文1と文3の類似度: {similarity_matrix[0, 2]:.4f}")
    

### Sentence Embeddings（文埋め込み）

Sentence-BERTを使用した高品質な文埋め込み：
    
    
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sentence-BERTモデルのロード
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # 文のリスト
    sentences = [
        "The cat sits on the mat.",
        "A feline rests on a rug.",
        "The dog plays in the park.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks."
    ]
    
    # 埋め込みベクトルの取得
    embeddings = model.encode(sentences)
    
    print(f"埋め込みベクトルの形状: {embeddings.shape}")
    
    # 類似度計算
    similarity = cosine_similarity(embeddings)
    
    print("\n文の類似度行列:")
    for i, sent in enumerate(sentences):
        print(f"\n{i}: {sent}")
    
    print("\n類似度:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"文{i} と 文{j}: {similarity[i, j]:.4f}")
    

### Domain Adaptation（ドメイン適応）

特定ドメインのデータで追加の事前学習を行います：
    
    
    from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset
    
    # 医療ドメインの例
    domain_texts = [
        "患者の血圧は正常範囲内です。",
        "糖尿病の治療には食事療法が重要です。",
        "この薬剤は副作用のリスクがあります。",
        # ... 大量のドメイン特化テキスト
    ]
    
    # データセット作成
    dataset = Dataset.from_dict({'text': domain_texts})
    
    # トークナイザとモデル
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
    
    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Data Collator（MLM用）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir='./domain_adapted_bert',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 追加の事前学習
    # trainer.train()  # コメント解除して実行
    
    print("ドメイン適応済みモデルの準備完了")
    

* * *

## 3.10 本章のまとめ

### 学んだこと

  1. **Transformerアーキテクチャ**

     * Self-Attention機構の原理と実装
     * Multi-Head Attentionで複数の表現を学習
     * Positional Encodingで位置情報を付与
     * Feed-Forward Networksで非線形変換
  2. **BERTの基礎**

     * 双方向事前学習の重要性
     * MLMとNSPタスク
     * HuggingFace Transformersの使い方
  3. **ファインチューニング**

     * テキスト分類、NER、QAタスク
     * 日本語BERTモデルの活用
  4. **応用テクニック**

     * 特徴抽出と文埋め込み
     * ドメイン適応
     * 実務での活用方法

### 次の章へ

第4章では、**BERTの発展モデル** を学びます：

  * RoBERTa、ALBERT、DistilBERT
  * GPT系列（GPT-2、GPT-3）
  * T5、BART（Seq2Seqモデル）
  * 最新の大規模言語モデル

* * *

## 演習問題

### 問題1（難易度：easy）

Self-AttentionとCross-Attentionの違いを説明してください。

解答例

**解答** ：

  * **Self-Attention** : Query、Key、Valueがすべて同じ入力系列から生成される。入力系列内の各要素間の関係を学習する。
  * **Cross-Attention** : QueryとKey/Valueが異なる系列から生成される（例: DecoderのQueryとEncoderのKey/Value）。異なる系列間の関係を学習する。

**使用場面** ：

  * Self-Attention: BERTのEncoder、文内の単語間依存関係
  * Cross-Attention: 機械翻訳のDecoder、原文と訳文の対応関係

### 問題2（難易度：medium）

Positional Encodingがなぜ必要か、またなぜSinusoidal関数を使用するのか説明してください。

解答例

**必要性** ：

  * Self-Attentionは順序情報を持たない（順列不変性）
  * "cat sat on mat"と"mat on sat cat"が区別できない
  * 位置情報を追加することで、系列の順序を保持

**Sinusoidal関数を使う理由** ：

  1. **固定長系列への対応** : 学習時に見ていない長さの系列にも対応可能
  2. **相対位置の表現** : $\text{PE}_{pos+k}$は$\text{PE}_{pos}$の線形変換で表現可能
  3. **パラメータ不要** : 学習不要で計算コストが低い
  4. **周期性** : 異なる周波数で短期・長期の位置関係を捉える

### 問題3（難易度：medium）

BERTのMLM（Masked Language Modeling）において、なぜ入力の15%のトークンをマスクするのか、その設計理由を説明してください。

解答例

**15%という割合の理由** ：

  * **バランス** : 低すぎると学習が遅い、高すぎるとコンテキスト不足
  * **実験的最適値** : BERTの論文で様々な割合を試した結果

**マスクの内訳** ：

  * 80%: [MASK]トークンに置換
  * 10%: ランダムなトークンに置換
  * 10%: 元のトークンのまま

**この設計の理由** ：

  1. 80%が[MASK]: 主要な学習タスク
  2. 10%がランダム: モデルが[MASK]だけに依存しないようにする
  3. 10%が元のまま: 実際のトークンの表現も学習する

これにより、ファインチューニング時に[MASK]トークンが現れなくても、モデルが適切に動作します。

### 問題4（難易度：hard）

Multi-Head Attentionで複数のheadを使用する利点を、単一のAttentionとの比較で説明してください。また、head数が多すぎる場合の問題点も述べてください。

解答例

**複数headの利点** ：

  1. **異なる表現部分空間**

     * 各headが異なる種類の関係を学習
     * 例: 構文的関係、意味的関係、長距離依存など
  2. **並列計算**

     * 複数headを同時に計算可能
     * GPUでの効率的な処理
  3. **冗長性とロバスト性**

     * 一部のheadが失敗しても他のheadが補完
     * 多様な情報を捕捉
  4. **アンサンブル効果**

     * 複数の視点からの情報を統合
     * より豊かな表現を学習

**head数が多すぎる場合の問題** ：

  1. **計算コスト増加** : メモリと計算時間の増加
  2. **過学習リスク** : パラメータ数増加による過学習
  3. **冗長性** : 似たような役割のheadが増える
  4. **最適化の困難性** : 多数のheadの調整が難しい

**実務での推奨** ：

モデルサイズ | 推奨head数  
---|---  
Small (d=256) | 4-8  
Base (d=512-768) | 8-12  
Large (d=1024) | 12-16  
  
### 問題5（難易度：hard）

以下のコードを完成させ、BERTを使った感情分析モデルを実装してください。データは自分で用意するか、サンプルデータを生成してください。
    
    
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    import torch
    
    # データセットクラスを実装
    class SentimentDataset(Dataset):
        # ここに実装
        pass
    
    # 学習関数を実装
    def train_model(model, train_loader, optimizer, device):
        # ここに実装
        pass
    
    # 評価関数を実装
    def evaluate_model(model, test_loader, device):
        # ここに実装
        pass
    

解答例
    
    
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # サンプルデータ
    train_texts = [
        "この商品は素晴らしい！",
        "最悪の体験でした。",
        "普通です。",
        "とても満足しています。",
        "二度と買いません。",
    ]
    train_labels = [1, 0, 1, 1, 0]  # 1: ポジティブ, 0: ネガティブ
    
    test_texts = [
        "良い商品だと思います。",
        "期待外れでした。"
    ]
    test_labels = [1, 0]
    
    # データセットクラス
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
    
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
    
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # 学習関数
    def train_model(model, train_loader, optimizer, device, epochs=3):
        model.train()
    
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
    
                loss = outputs.loss
                total_loss += loss.item()
    
                loss.backward()
                optimizer.step()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 評価関数
    def evaluate_model(model, test_loader, device):
        model.eval()
        predictions = []
        true_labels = []
    
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
    
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
    
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
    
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\n精度: {accuracy:.4f}")
        print("\n分類レポート:")
        print(classification_report(true_labels, predictions, target_names=['ネガティブ', 'ポジティブ']))
    
        return accuracy
    
    # メイン処理
    def main():
        # トークナイザとモデル
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)
    
        # データセットとデータローダー
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
        # オプティマイザ
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
        # 学習
        print("学習開始...")
        train_model(model, train_loader, optimizer, device, epochs=3)
    
        # 評価
        print("\n評価開始...")
        evaluate_model(model, test_loader, device)
    
    if __name__ == '__main__':
        main()
    

**出力例** ：
    
    
    学習開始...
    Epoch 1/3, Loss: 0.6234
    Epoch 2/3, Loss: 0.4521
    Epoch 3/3, Loss: 0.3012
    
    評価開始...
    精度: 1.0000
    
    分類レポート:
                  precision    recall  f1-score   support
    
      ネガティブ       1.00      1.00      1.00         1
    ポジティブ       1.00      1.00      1.00         1
    
        accuracy                           1.00         2
       macro avg       1.00      1.00      1.00         2
    weighted avg       1.00      1.00      1.00         2
    

* * *

## 参考文献

  1. Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS.
  2. Devlin, J., et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL.
  3. Liu, Y., et al. (2019). _RoBERTa: A Robustly Optimized BERT Pretraining Approach_. arXiv.
  4. Lan, Z., et al. (2020). _ALBERT: A Lite BERT for Self-supervised Learning of Language Representations_. ICLR.
  5. Sanh, V., et al. (2019). _DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter_. NeurIPS Workshop.
  6. HuggingFace Transformers Documentation. <https://huggingface.co/docs/transformers/>
  7. 東北大学BERTモデル. <https://github.com/cl-tohoku/bert-japanese>
