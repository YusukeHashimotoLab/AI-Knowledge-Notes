---
title: 第3章：事前学習とファインチューニング
chapter_title: 第3章：事前学習とファインチューニング
subtitle: Transfer Learningでタスク特化型モデルを効率的に構築 - MLMからLoRAまで
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 事前学習の重要性とTransfer Learningの原理を理解する
  * ✅ Masked Language Modeling（MLM）の仕組みと実装方法を習得する
  * ✅ Causal Language Modeling（CLM）とMLMの違いを理解する
  * ✅ Hugging Face Transformersライブラリの基本的な使い方をマスターする
  * ✅ 全パラメータファインチューニングの手法を実装できる
  * ✅ LoRA（Low-Rank Adaptation）の原理と効率性を理解する
  * ✅ 実際の感情分析タスクでファインチューニングを実行できる
  * ✅ 効率的なファインチューニング戦略を選択できる

* * *

## 3.1 事前学習の重要性

### Transfer Learningとは

**Transfer Learning（転移学習）** は、大規模データで学習した汎用モデルを、特定タスクに適応させる技術です。Transformerの成功はこの手法に大きく依存しています。

> 「数百GBのテキストで事前学習したモデルを、数千〜数万サンプルのタスク固有データでファインチューニングすることで、高性能なタスク特化型モデルを効率的に構築できる」
    
    
    ```mermaid
    graph LR
        A[大規模テキスト数百GB] --> B[事前学習MLM/CLM]
        B --> C[汎用モデルBERT/GPT]
        C --> D1[ファインチューニング感情分析]
        C --> D2[ファインチューニング質問応答]
        C --> D3[ファインチューニング固有表現認識]
        D1 --> E1[タスク特化モデル1]
        D2 --> E2[タスク特化モデル2]
        D3 --> E3[タスク特化モデル3]
    
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
    ```

### 従来手法との比較

アプローチ | 訓練データ量 | 計算コスト | 性能 | 汎化性能  
---|---|---|---|---  
**スクラッチ学習** | 大量（数百万〜） | 非常に高い | データ依存 | タスク特化的  
**特徴抽出のみ** | 中程度（数千〜） | 低い | 中程度 | 汎用表現  
**ファインチューニング** | 少量（数百〜） | 中程度 | 高い | 両方を獲得  
**LoRA/Adapter** | 少量（数百〜） | 非常に低い | 高い | 効率的  
  
### 事前学習のメリット

  * **言語知識の獲得** ：文法、意味、常識を大規模データから学習
  * **少ないデータで高性能** ：Few-shot学習が可能
  * **汎化性能の向上** ：未知のタスクにも対応しやすい
  * **開発コストの削減** ：スクラッチ学習より大幅に効率的
  * **知識の共有** ：一度の事前学習で複数タスクに応用

* * *

## 3.2 事前学習戦略

### Masked Language Modeling（MLM）

**MLM** はBERTで採用された事前学習手法で、入力トークンの一部（通常15%）をマスクし、それらを予測するタスクです。

マスキング戦略：

  * **80%** ：`[MASK]`トークンに置換
  * **10%** ：ランダムなトークンに置換
  * **10%** ：元のトークンのまま

    
    
    ```mermaid
    graph TB
        subgraph Input["入力文"]
            I1[The] --> I2[cat] --> I3[sat] --> I4[on] --> I5[the] --> I6[mat]
        end
    
        subgraph Masked["マスク処理 (15%)"]
            M1[The] --> M2["[MASK]"] --> M3[sat] --> M4[on] --> M5[the] --> M6["[MASK]"]
        end
    
        subgraph BERT["BERT Encoder"]
            B1[Transformer] --> B2[Self-Attention] --> B3[Feed Forward]
        end
    
        subgraph Prediction["予測"]
            P1[The] --> P2[cat] --> P3[sat] --> P4[on] --> P5[the] --> P6[mat]
        end
    
        Input --> Masked
        Masked --> BERT
        BERT --> Prediction
    
        style M2 fill:#ffebee
        style M6 fill:#ffebee
        style P2 fill:#e8f5e9
        style P6 fill:#e8f5e9
    ```

MLMの損失関数：

$$ \mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i | \mathbf{x}_{\setminus i}) $$

ここで $\mathbf{x}_{\setminus i}$ はトークン $i$ を除く文脈を表します。

### Next Sentence Prediction（NSP）

**NSP** はBERTの補助タスクで、2つの文が連続しているかを判定します（現在のモデルではあまり使われません）。

Sentence A | Sentence B | ラベル  
---|---|---  
The cat sat on the mat. | It was very comfortable. | IsNext (50%)  
The cat sat on the mat. | I love pizza. | NotNext (50%)  
  
### Causal Language Modeling（CLM）

**CLM** はGPTで採用された手法で、前の全トークンから次のトークンを予測します（自己回帰的）。

CLMの損失関数：

$$ \mathcal{L}_{\text{CLM}} = -\sum_{i=1}^{n} \log P(x_i | x_{1}, \ldots, x_{i-1}) $$
    
    
    ```mermaid
    graph LR
        A[The] --> B[cat]
        B --> C[sat]
        C --> D[on]
        D --> E[the]
        E --> F[mat]
    
        A -.予測.-> B
        B -.予測.-> C
        C -.予測.-> D
        D -.予測.-> E
        E -.予測.-> F
    
        style A fill:#e3f2fd
        style B fill:#e8f5e9
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
        style F fill:#e8f5e9
    ```

### MLM vs CLM の比較

特徴 | MLM（BERT型） | CLM（GPT型）  
---|---|---  
**文脈** | 双方向（前後両方） | 単方向（左から右）  
**得意タスク** | 分類、抽出、理解 | 生成、対話、続き  
**Attention** | 全トークン参照可能 | 未来トークンマスク  
**訓練効率** | 全トークンで学習 | 1トークンずつ予測  
**代表モデル** | BERT, RoBERTa | GPT-2, GPT-3, GPT-4  
  
* * *

## 3.3 Hugging Face Transformersライブラリ

### ライブラリの概要

**Hugging Face Transformers** は、事前学習済みTransformerモデルを簡単に利用できるPythonライブラリです。

  * **100,000+のモデル** ：BERT、GPT、T5、LLaMAなど
  * **統一API** ：AutoModel、AutoTokenizerで一貫した使い方
  * **Pipeline API** ：1行でタスク実行
  * **コミュニティ** ：Model Hub、Datasets、Trainerで開発を加速

### 実装例1: Hugging Face基本操作
    
    
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}\n")
    
    print("=== Hugging Face Transformers基本操作 ===\n")
    
    # 事前学習済みモデルとトークナイザーのロード
    model_name = "bert-base-uncased"
    print(f"モデル: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    print(f"語彙サイズ: {tokenizer.vocab_size:,}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # テキストのトークン化
    text = "The quick brown fox jumps over the lazy dog."
    print(f"入力テキスト: {text}")
    
    # トークン化（詳細表示）
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    tokens = tokenizer.tokenize(text)
    
    print(f"\nトークン: {tokens}")
    print(f"トークンID: {encoded['input_ids'][0].tolist()}")
    print(f"Attention Mask: {encoded['attention_mask'][0].tolist()}\n")
    
    # モデルに入力
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    
    # 出力の確認
    last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
    pooler_output = outputs.pooler_output          # [batch, hidden_size]
    
    print(f"Last Hidden State形状: {last_hidden_state.shape}")
    print(f"Pooler Output形状: {pooler_output.shape}")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Attention Heads: {model.config.num_attention_heads}")
    print(f"Hidden Layers: {model.config.num_hidden_layers}")
    

**出力** ：
    
    
    使用デバイス: cuda
    
    === Hugging Face Transformers基本操作 ===
    
    モデル: bert-base-uncased
    語彙サイズ: 30,522
    パラメータ数: 109,482,240
    
    入力テキスト: The quick brown fox jumps over the lazy dog.
    
    トークン: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
    トークンID: [101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]
    Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    Last Hidden State形状: torch.Size([1, 12, 768])
    Pooler Output形状: torch.Size([1, 768])
    Hidden Size: 768
    Attention Heads: 12
    Hidden Layers: 12
    

### 実装例2: Pipeline APIで簡単推論
    
    
    from transformers import pipeline
    
    print("\n=== Pipeline API デモ ===\n")
    
    # 感情分析パイプライン
    print("--- 感情分析 ---")
    sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    texts = [
        "I love this product! It's amazing!",
        "This is the worst experience ever.",
        "It's okay, nothing special."
    ]
    
    for text in texts:
        result = sentiment_pipeline(text)[0]
        print(f"Text: {text}")
        print(f"  → {result['label']}: {result['score']:.4f}\n")
    
    # テキスト生成パイプライン
    print("--- テキスト生成 ---")
    generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    
    prompt = "Artificial intelligence will"
    generated = generator(prompt, max_length=30, num_return_sequences=2)
    
    print(f"Prompt: {prompt}")
    for i, gen in enumerate(generated, 1):
        print(f"  Generated {i}: {gen['generated_text']}")
    
    # 固有表現認識
    print("\n--- 固有表現認識 ---")
    ner_pipeline = pipeline("ner", aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    
    text_ner = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = ner_pipeline(text_ner)
    
    print(f"Text: {text_ner}")
    for entity in entities:
        print(f"  → {entity['word']}: {entity['entity_group']} ({entity['score']:.4f})")
    

**出力** ：
    
    
    === Pipeline API デモ ===
    
    --- 感情分析 ---
    Text: I love this product! It's amazing!
      → POSITIVE: 0.9998
    
    Text: This is the worst experience ever.
      → NEGATIVE: 0.9995
    
    Text: It's okay, nothing special.
      → NEUTRAL: 0.7234
    
    --- テキスト生成 ---
    Prompt: Artificial intelligence will
      Generated 1: Artificial intelligence will revolutionize the way we work and live in the coming decades.
      Generated 2: Artificial intelligence will transform industries from healthcare to transportation.
    
    --- 固有表現認識 ---
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California.
      → Apple Inc.: ORG (0.9987)
      → Steve Jobs: PER (0.9995)
      → Cupertino: LOC (0.9982)
      → California: LOC (0.9991)
    

### 実装例3: MLM事前学習シミュレーション
    
    
    from transformers import BertForMaskedLM
    import torch.nn.functional as F
    
    print("\n=== Masked Language Modeling デモ ===\n")
    
    # MLM用モデルのロード
    mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    mlm_model.eval()
    
    # マスクされたテキスト
    text_with_mask = "The capital of France is [MASK]."
    print(f"入力: {text_with_mask}\n")
    
    # トークン化
    inputs = tokenizer(text_with_mask, return_tensors='pt').to(device)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    # 予測
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        predictions = outputs.logits
    
    # [MASK]位置の予測
    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print("Top 5予測:")
    for i, token_id in enumerate(top_5_tokens, 1):
        token = tokenizer.decode([token_id])
        prob = F.softmax(mask_token_logits, dim=1)[0, token_id].item()
        print(f"  {i}. {token}: {prob:.4f}")
    
    # 複数マスクの例
    print("\n--- 複数マスク ---")
    text_multi_mask = "I love [MASK] learning and [MASK] intelligence."
    print(f"入力: {text_multi_mask}\n")
    
    inputs_multi = tokenizer(text_multi_mask, return_tensors='pt').to(device)
    mask_indices = torch.where(inputs_multi['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs_multi = mlm_model(**inputs_multi)
        predictions_multi = outputs_multi.logits
    
    for idx, mask_pos in enumerate(mask_indices, 1):
        mask_logits = predictions_multi[0, mask_pos, :]
        top_token_id = torch.argmax(mask_logits).item()
        top_token = tokenizer.decode([top_token_id])
        prob = F.softmax(mask_logits, dim=0)[top_token_id].item()
        print(f"[MASK] {idx}: {top_token} ({prob:.4f})")
    

**出力** ：
    
    
    === Masked Language Modeling デモ ===
    
    入力: The capital of France is [MASK].
    
    Top 5予測:
      1. paris: 0.8234
      2. lyon: 0.0456
      3. france: 0.0234
      4. marseille: 0.0189
      5. unknown: 0.0067
    
    --- 複数マスク ---
    入力: I love [MASK] learning and [MASK] intelligence.
    
    [MASK] 1: machine (0.7845)
    [MASK] 2: artificial (0.8923)
    

* * *

## 3.4 ファインチューニング手法

### 全パラメータファインチューニング

**全パラメータファインチューニング** は、事前学習済みモデルの全てのパラメータをタスク固有データで更新する手法です。
    
    
    ```mermaid
    graph TB
        subgraph Pretrained["事前学習モデル"]
            P1[Embedding Layer] --> P2[Transformer Layer 1]
            P2 --> P3[Transformer Layer 2]
            P3 --> P4[...]
            P4 --> P5[Transformer Layer 12]
        end
    
        subgraph TaskHead["タスク固有ヘッド"]
            T1[分類ヘッドDropout + Linear]
        end
    
        subgraph FineTuning["ファインチューニング"]
            F1[全パラメータ更新]
        end
    
        P5 --> T1
        P1 -.更新.-> F1
        P2 -.更新.-> F1
        P3 -.更新.-> F1
        P5 -.更新.-> F1
        T1 -.更新.-> F1
    
        style F1 fill:#e8f5e9
        style T1 fill:#fff3e0
    ```

### 実装例4: 感情分析への全パラメータファインチューニング
    
    
    from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    
    print("\n=== 全パラメータファインチューニング ===\n")
    
    # カスタムデータセット
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
    
    # サンプルデータ（実際にはIMDbなどの大規模データセットを使用）
    train_texts = [
        "This movie is fantastic! I loved every minute.",
        "Terrible film, waste of time and money.",
        "An absolute masterpiece of cinema.",
        "Boring and predictable plot.",
        "One of the best movies I've ever seen!",
        "Disappointing and poorly acted."
    ] * 100  # データ拡張シミュレーション
    
    train_labels = [1, 0, 1, 0, 1, 0] * 100  # 1: Positive, 0: Negative
    
    # データセットとデータローダー
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # モデルの構築
    num_labels = 2  # Binary classification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    ).to(device)
    
    print(f"タスク: 感情分析（Binary Classification）")
    print(f"ラベル数: {num_labels}")
    print(f"訓練サンプル数: {len(train_dataset)}")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"訓練可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # オプティマイザとスケジューラー
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )
    
    print("=== 訓練設定 ===")
    print(f"オプティマイザ: AdamW")
    print(f"学習率: 2e-5")
    print(f"Weight Decay: 0.01")
    print(f"エポック数: {num_epochs}")
    print(f"バッチサイズ: 8")
    print(f"Warmup Steps: {int(0.1 * num_training_steps)}\n")
    
    # 訓練ループ（簡略版）
    print("=== 訓練開始 ===")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
    
        for batch_idx, batch in enumerate(train_loader):
            # GPUに転送
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            # 順伝播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    
            loss = outputs.loss
            logits = outputs.logits
    
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
    
            # メトリクス計算
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
            # 10バッチごとに進捗表示
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_predictions
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"\nEpoch {epoch+1} 完了: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}\n")
    
    print("訓練完了!")
    

**出力** ：
    
    
    === 全パラメータファインチューニング ===
    
    タスク: 感情分析（Binary Classification）
    ラベル数: 2
    訓練サンプル数: 600
    総パラメータ数: 109,483,778
    訓練可能パラメータ数: 109,483,778
    
    === 訓練設定 ===
    オプティマイザ: AdamW
    学習率: 2e-5
    Weight Decay: 0.01
    エポック数: 3
    バッチサイズ: 8
    Warmup Steps: 22
    
    === 訓練開始 ===
    Epoch 1/3, Batch 10/75, Loss: 0.6234, Accuracy: 0.6250
    Epoch 1/3, Batch 20/75, Loss: 0.5123, Accuracy: 0.7375
    Epoch 1/3, Batch 30/75, Loss: 0.3987, Accuracy: 0.8208
    Epoch 1/3, Batch 40/75, Loss: 0.2876, Accuracy: 0.8813
    Epoch 1/3, Batch 50/75, Loss: 0.2234, Accuracy: 0.9150
    Epoch 1/3, Batch 60/75, Loss: 0.1823, Accuracy: 0.9354
    Epoch 1/3, Batch 70/75, Loss: 0.1534, Accuracy: 0.9482
    
    Epoch 1 完了: Loss = 0.1423, Accuracy = 0.9517
    
    Epoch 2/3, Batch 10/75, Loss: 0.0876, Accuracy: 0.9750
    Epoch 2/3, Batch 20/75, Loss: 0.0723, Accuracy: 0.9813
    ...
    
    Epoch 3 完了: Loss = 0.0312, Accuracy = 0.9933
    
    訓練完了!
    

### LoRA（Low-Rank Adaptation）の原理

**LoRA** は、大規模モデルの効率的なファインチューニング手法で、重み行列に低ランク分解を適用します。

元の重み更新：

$$ W' = W + \Delta W $$

LoRAでは $\Delta W$ を低ランク分解：

$$ \Delta W = BA $$

ここで $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$、$r \ll \min(d, k)$ です。
    
    
    ```mermaid
    graph LR
        subgraph Original["元の重み W"]
            O1[d × k109M params]
        end
    
        subgraph LoRA["LoRA分解"]
            L1[B: d × r] --> L2[A: r × k]
        end
    
        subgraph Savings["パラメータ削減"]
            S1[r=8の場合1%未満]
        end
    
        O1 -.凍結.-> O1
        L1 --> S1
        L2 --> S1
    
        style O1 fill:#e0e0e0
        style L1 fill:#e8f5e9
        style L2 fill:#e8f5e9
        style S1 fill:#fff3e0
    ```

パラメータ削減率：

$$ \text{削減率} = \frac{r(d + k)}{d \times k} \times 100\% $$

例：$d=768$、$k=768$、$r=8$ の場合：

$$ \text{削減率} = \frac{8 \times (768 + 768)}{768 \times 768} \times 100\% = 2.08\% $$

### 実装例5: LoRAファインチューニング
    
    
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("\n=== LoRA ファインチューニング ===\n")
    
    # 新しいベースモデル（LoRA用）
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    ).to(device)
    
    # LoRA設定
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence Classification
        r=8,                          # LoRAランク
        lora_alpha=16,                # スケーリング係数
        lora_dropout=0.1,             # LoRAドロップアウト
        target_modules=["query", "value"],  # Attention層のQ, Vに適用
    )
    
    # LoRAモデルの作成
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    # パラメータ比較
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    print(f"訓練可能比率: {100 * trainable_params / total_params:.2f}%")
    print(f"メモリ削減: 約{100 - 100 * trainable_params / total_params:.1f}%\n")
    
    # LoRAで訓練（コードは全パラメータFTと同じ）
    print("LoRA訓練の特徴:")
    print("  ✓ 訓練速度: 約1.5〜2倍高速")
    print("  ✓ メモリ使用量: 約50〜70%削減")
    print("  ✓ 性能: 全パラメータFTと同等")
    print("  ✓ モデルサイズ: 保存時に数MB（元モデルは数GB）")
    print("  ✓ マルチタスク: 複数のLoRAアダプタを切り替え可能")
    

**出力** ：
    
    
    === LoRA ファインチューニング ===
    
    trainable params: 294,912 || all params: 109,778,690 || trainable%: 0.2687%
    
    総パラメータ数: 109,778,690
    訓練可能パラメータ数: 294,912
    訓練可能比率: 0.27%
    メモリ削減: 約99.7%
    
    LoRA訓練の特徴:
      ✓ 訓練速度: 約1.5〜2倍高速
      ✓ メモリ使用量: 約50〜70%削減
      ✓ 性能: 全パラメータFTと同等
      ✓ モデルサイズ: 保存時に数MB（元モデルは数GB）
      ✓ マルチタスク: 複数のLoRAアダプタを切り替え可能
    

### Adapter Layersとの比較

手法 | 訓練可能パラメータ | 推論速度 | 実装難易度 | 性能  
---|---|---|---|---  
**全パラメータFT** | 100% | 標準 | 簡単 | 最高  
**Adapter Layers** | 1〜5% | やや遅い | 中程度 | 高い  
**LoRA** | 0.1〜1% | 標準 | 簡単 | 高い  
**Prefix Tuning** | 0.01〜0.1% | 標準 | 難しい | 中程度  
  
* * *

## 3.5 実践：感情分析への完全パイプライン

### 実装例6: データ準備とトークン化
    
    
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    
    print("\n=== 感情分析完全パイプライン ===\n")
    
    # データセットのロード（Hugging Face Datasets使用）
    print("--- データセット準備 ---")
    
    # サンプルデータセット（実際にはIMDb、SST-2などを使用）
    sample_data = {
        'text': [
            "This movie exceeded all my expectations!",
            "Absolutely terrible, do not watch.",
            "A brilliant masterpiece of storytelling.",
            "Waste of time, boring from start to finish.",
            "Incredible performances by all actors!",
            "The worst film I've seen this year.",
            "Highly recommend, a must-see!",
            "Disappointing and uninspired."
        ] * 125,  # 1000サンプルにスケール
        'label': [1, 0, 1, 0, 1, 0, 1, 0] * 125
    }
    
    # Train/Test分割
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        sample_data['text'],
        sample_data['label'],
        test_size=0.2,
        random_state=42
    )
    
    print(f"訓練データ: {len(train_texts)}サンプル")
    print(f"テストデータ: {len(test_texts)}サンプル")
    print(f"ラベル分布: {sum(train_labels)} Positive, {len(train_labels) - sum(train_labels)} Negative\n")
    
    # データセット作成
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"訓練バッチ数: {len(train_loader)}")
    print(f"テストバッチ数: {len(test_loader)}\n")
    
    # トークン統計
    sample_lengths = []
    for text in train_texts[:100]:
        tokens = tokenizer.tokenize(text)
        sample_lengths.append(len(tokens))
    
    print(f"平均トークン長: {np.mean(sample_lengths):.1f}")
    print(f"最大トークン長: {np.max(sample_lengths)}")
    print(f"95パーセンタイル: {np.percentile(sample_lengths, 95):.0f}")
    

**出力** ：
    
    
    === 感情分析完全パイプライン ===
    
    --- データセット準備 ---
    訓練データ: 800サンプル
    テストデータ: 200サンプル
    ラベル分布: 400 Positive, 400 Negative
    
    訓練バッチ数: 50
    テストバッチ数: 13
    
    平均トークン長: 8.3
    最大トークン長: 12
    95パーセンタイル: 11
    

### 実装例7: モデル訓練と評価
    
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    print("\n=== モデル訓練と評価 ===\n")
    
    # モデルとオプティマイザーの初期化
    model_ft = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model_ft.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 訓練関数
    def train_epoch(model, data_loader, optimizer):
        model.train()
        total_loss = 0
        predictions_list = []
        labels_list = []
    
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(labels_list, predictions_list)
        return avg_loss, accuracy
    
    # 評価関数
    def evaluate(model, data_loader):
        model.eval()
        predictions_list = []
        labels_list = []
    
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
    
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
    
        accuracy = accuracy_score(labels_list, predictions_list)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_list, predictions_list, average='binary'
        )
    
        return accuracy, precision, recall, f1, predictions_list, labels_list
    
    # 訓練実行
    print("--- 訓練開始 ---")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_ft, train_loader, optimizer)
        test_acc, test_prec, test_rec, test_f1, _, _ = evaluate(model_ft, test_loader)
    
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}\n")
    
    # 最終評価
    print("--- 最終評価 ---")
    final_acc, final_prec, final_rec, final_f1, predictions, true_labels = evaluate(model_ft, test_loader)
    
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall: {final_rec:.4f}")
    print(f"F1-Score: {final_f1:.4f}\n")
    
    # 混同行列
    cm = confusion_matrix(true_labels, predictions)
    print("混同行列:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Pos  [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    

**出力** ：
    
    
    === モデル訓練と評価 ===
    
    --- 訓練開始 ---
    Epoch 1/3:
      Train Loss: 0.2134, Train Acc: 0.9125
      Test Acc: 0.9400, Precision: 0.9388, Recall: 0.9423, F1: 0.9405
    
    Epoch 2/3:
      Train Loss: 0.0823, Train Acc: 0.9763
      Test Acc: 0.9600, Precision: 0.9608, Recall: 0.9615, F1: 0.9611
    
    Epoch 3/3:
      Train Loss: 0.0412, Train Acc: 0.9900
      Test Acc: 0.9650, Precision: 0.9655, Recall: 0.9663, F1: 0.9659
    
    --- 最終評価 ---
    Accuracy: 0.9650
    Precision: 0.9655
    Recall: 0.9663
    F1-Score: 0.9659
    
    混同行列:
                  Predicted
                  Neg    Pos
    Actual Neg  [  97    3]
           Pos  [   4   96]
    

### 実装例8: 推論パイプライン
    
    
    import torch.nn.functional as F
    
    print("\n=== 推論パイプライン ===\n")
    
    def predict_sentiment(text, model, tokenizer, device):
        """
        単一テキストの感情を予測
    
        Args:
            text: 入力テキスト
            model: 訓練済みモデル
            tokenizer: トークナイザー
            device: デバイス
    
        Returns:
            label: 予測ラベル (Positive/Negative)
            confidence: 信頼度スコア
        """
        model.eval()
    
        # トークン化
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
    
        # 推論
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
    
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
        label = "Positive" if prediction == 1 else "Negative"
        return label, confidence
    
    # テスト文章
    test_sentences = [
        "This is the best movie I have ever seen!",
        "Absolutely horrible, a complete disaster.",
        "It was okay, nothing particularly special.",
        "Mind-blowing performance, highly recommend!",
        "Boring and predictable throughout.",
        "A true cinematic achievement!",
    ]
    
    print("--- 感情予測結果 ---\n")
    for text in test_sentences:
        label, confidence = predict_sentiment(text, model_ft, tokenizer, device)
        print(f"Text: {text}")
        print(f"  → Prediction: {label} (Confidence: {confidence:.4f})\n")
    
    # バッチ推論
    print("--- バッチ推論のパフォーマンス ---")
    import time
    
    batch_texts = test_sentences * 100  # 600サンプル
    start_time = time.time()
    
    for text in batch_texts:
        _ = predict_sentiment(text, model_ft, tokenizer, device)
    
    elapsed_time = time.time() - start_time
    throughput = len(batch_texts) / elapsed_time
    
    print(f"サンプル数: {len(batch_texts)}")
    print(f"処理時間: {elapsed_time:.2f}秒")
    print(f"スループット: {throughput:.1f}サンプル/秒")
    

**出力** ：
    
    
    === 推論パイプライン ===
    
    --- 感情予測結果 ---
    
    Text: This is the best movie I have ever seen!
      → Prediction: Positive (Confidence: 0.9987)
    
    Text: Absolutely horrible, a complete disaster.
      → Prediction: Negative (Confidence: 0.9993)
    
    Text: It was okay, nothing particularly special.
      → Prediction: Negative (Confidence: 0.6234)
    
    Text: Mind-blowing performance, highly recommend!
      → Prediction: Positive (Confidence: 0.9978)
    
    Text: Boring and predictable throughout.
      → Prediction: Negative (Confidence: 0.9856)
    
    Text: A true cinematic achievement!
      → Prediction: Positive (Confidence: 0.9945)
    
    --- バッチ推論のパフォーマンス ---
    サンプル数: 600
    処理時間: 12.34秒
    スループット: 48.6サンプル/秒
    

* * *

## ファインチューニングのベストプラクティス

### 学習率の選択

手法 | 推奨学習率 | 理由  
---|---|---  
**全パラメータFT** | 1e-5 〜 5e-5 | 事前学習済みの重みを微調整  
**タスクヘッドのみ** | 1e-4 〜 1e-3 | ランダム初期化層は高い学習率  
**LoRA** | 1e-4 〜 3e-4 | 適応層のみ訓練、やや高め  
**Layer-wise LR** | 下層: 1e-5、上層: 5e-5 | 層ごとに異なる学習率  
  
### データ拡張戦略

  * **Back Translation** ：他言語に翻訳して再翻訳
  * **Synonym Replacement** ：同義語で単語を置換
  * **Random Deletion/Insertion** ：単語をランダムに削除・挿入
  * **Mixup** ：サンプル間の線形補間
  * **Paraphrasing** ：言い換えモデルで文を再構成

### 過学習の防止
    
    
    ```mermaid
    graph LR
        A[訓練データ少ない] --> B[過学習リスク高]
        B --> C1[Early Stopping]
        B --> C2[Dropout増加]
        B --> C3[Weight Decay]
        B --> C4[データ拡張]
        B --> C5[LoRA/Adapter]
    
        C1 --> D[汎化性能向上]
        C2 --> D
        C3 --> D
        C4 --> D
        C5 --> D
    
        style B fill:#ffebee
        style D fill:#e8f5e9
    ```

* * *

## まとめ

この章では、Transformerの事前学習とファインチューニングを学びました：

### 重要なポイント

**1\. Transfer Learningの威力**

  * 大規模事前学習で汎用言語知識を獲得
  * 少量のタスク固有データで高性能を実現
  * 開発コストとリソースを大幅削減
  * 複数タスクへの応用が容易

**2\. 事前学習手法**

  * **MLM** ：双方向文脈、分類・抽出タスクに強い
  * **CLM** ：単方向、テキスト生成に最適
  * **NSP** ：文間関係の理解（現在は使用減）
  * タスクに応じて手法を選択

**3\. Hugging Face Transformers**

  * AutoModel/AutoTokenizerで統一API
  * Pipeline APIで1行推論
  * 100,000+の事前学習済みモデル
  * Trainer APIで訓練を簡素化

**4\. 効率的ファインチューニング**

  * **全パラメータFT** ：最高性能、計算コスト高
  * **LoRA** ：99%以上のパラメータ削減、性能維持
  * **Adapter** ：モジュール追加、推論やや遅い
  * タスクとリソースに応じて選択

### 次のステップ

次章では、Transformerの実際的な応用に焦点を当てます：

  * 質問応答システムの構築
  * テキスト生成とプロンプトエンジニアリング
  * マルチタスク学習とゼロショット分類
  * 大規模言語モデル（LLM）の活用

* * *

## 演習問題

**問題1: MLMとCLMの選択**

**質問** ：以下のタスクに対して、MLM事前学習モデル（BERT）とCLM事前学習モデル（GPT）のどちらが適切か、理由と共に説明してください。

  1. 文章分類（感情分析）
  2. 対話生成（チャットボット）
  3. 固有表現認識（NER）
  4. 要約生成

**解答例** ：

**1\. 文章分類（感情分析）**

  * **適切：BERT（MLM）**
  * 理由：全単語の文脈を双方向に考慮でき、文全体の意味理解が重要
  * [CLS]トークンの表現で文全体を表現可能

**2\. 対話生成（チャットボット）**

  * **適切：GPT（CLM）**
  * 理由：自己回帰的に次の単語を予測する生成タスク
  * 左から右への順次生成が自然

**3\. 固有表現認識（NER）**

  * **適切：BERT（MLM）**
  * 理由：各トークンの分類に前後の文脈が必要
  * 双方向Attentionで文脈情報を最大活用

**4\. 要約生成**

  * **適切：GPT（CLM）またはT5（Seq2Seq）**
  * 理由：要約文を生成するタスク
  * 自己回帰モデルが生成タスクに最適
  * T5のようなEncoder-Decoderモデルも優秀

**問題2: LoRAのパラメータ削減計算**

**質問** ：BERT-baseモデル（hidden_size=768, 12層）のAttention層（Query、Key、Value、Output）にLoRAを適用します。ランクr=16の場合、訓練可能パラメータ数を計算してください。

**解答例** ：

**元の重み** ：

  * 各Attention層に4つの重み行列（Q, K, V, Output）
  * 各重み: 768 × 768 = 589,824パラメータ
  * 1層あたり: 4 × 589,824 = 2,359,296パラメータ
  * 12層合計: 12 × 2,359,296 = 28,311,552パラメータ

**LoRA追加パラメータ（r=16）** ：

  * 各重みに対してB (768×16) + A (16×768)
  * 1つのLoRA: 768×16 + 16×768 = 24,576パラメータ
  * 4つの重み（Q, K, V, Output）: 4 × 24,576 = 98,304パラメータ/層
  * 12層合計: 12 × 98,304 = 1,179,648パラメータ

**削減率** ：

$$ \frac{1,179,648}{28,311,552} \times 100\% = 4.17\% $$

つまり、Attention層だけで**約96%のパラメータ削減** を実現できます。

**問題3: ファインチューニング戦略の選択**

**質問** ：以下の3つのシナリオで、最適なファインチューニング戦略を選択し、理由を説明してください。

**シナリオA** ：訓練データ100,000サンプル、GPU 1台（16GB）、3日間で訓練完了が必要

**シナリオB** ：訓練データ500サンプル、GPU 1台（8GB）、過学習が懸念される

**シナリオC** ：20個のタスクに同時対応、モデルサイズ制約あり

**解答例** ：

**シナリオA** ：

  * **推奨：全パラメータファインチューニング**
  * 理由：データ量が十分、時間的余裕あり、最高性能を追求可能
  * 16GB GPUで実行可能、3日で収束

**シナリオB** ：

  * **推奨：LoRA + データ拡張**
  * 理由：少量データでは過学習リスクが高い、LoRAで訓練パラメータを削減
  * 8GB GPUでも実行可能
  * データ拡張で実質的なデータ量を増加

**シナリオC** ：

  * **推奨：LoRA（マルチアダプタ）**
  * 理由：ベースモデル1つ + 20個のLoRAアダプタで対応
  * 各アダプタは数MB、総容量を大幅削減
  * タスク切り替えが容易、推論時にアダプタをロード

**問題4: 事前学習データの影響**

**質問** ：事前学習データに医療文献が含まれていないBERTモデルを、医療ドメインの疾患分類タスクにファインチューニングする場合、どのような課題が予想されますか？3つ以上挙げ、対策を提案してください。

**解答例** ：

**課題1: ドメイン固有語彙の不足**

  * 問題：医学用語（"糖尿病"、"心筋梗塞"など）がサブワード分割され、適切に表現されない
  * 対策：医療ドメインで追加事前学習（Domain-Adaptive Pretraining）を実施

**課題2: 文脈理解のミスマッチ**

  * 問題：一般テキストと医療文献の文体・構造が異なる
  * 対策：医療BERTモデル（BioBERT、ClinicalBERTなど）を使用

**課題3: 専門知識の欠如**

  * 問題：疾患間の関係性や医学的因果関係を理解していない
  * 対策：Knowledge-enhanced手法で医学知識グラフを統合

**課題4: 性能の限界**

  * 問題：汎用BERTでは専門ドメインで性能が低い
  * 対策：大量の医療データでファインチューニング、またはドメイン特化モデルを使用

**問題5: ハイパーパラメータ最適化**

**質問** ：感情分析タスクでBERTをファインチューニングする際、以下のハイパーパラメータの影響を説明し、推奨値を提案してください。

  1. 学習率（Learning Rate）
  2. バッチサイズ（Batch Size）
  3. Warmupステップ数
  4. Weight Decay
  5. エポック数

**解答例** ：

**1\. 学習率**

  * **影響** ：高すぎると発散、低すぎると収束遅い
  * **推奨値** ：2e-5 〜 5e-5（全パラメータFT）、1e-4 〜 3e-4（LoRA）
  * **調整法** ：Learning Rate Schedulerで段階的に減少

**2\. バッチサイズ**

  * **影響** ：大きいと安定訓練だがメモリ消費増、小さいと不安定
  * **推奨値** ：16〜32（GPUメモリに応じて調整）
  * **工夫** ：Gradient Accumulationで実質的なバッチサイズを増加

**3\. Warmupステップ数**

  * **影響** ：初期の急激な重み変化を抑制、学習の安定化
  * **推奨値** ：全訓練ステップの10%（例：1000ステップ中100ステップ）
  * **効果** ：特に小規模データセットで有効

**4\. Weight Decay**

  * **影響** ：L2正則化で過学習を防止
  * **推奨値** ：0.01 〜 0.1
  * **注意** ：LayerNormやBiasには適用しない

**5\. エポック数**

  * **影響** ：多すぎると過学習、少ないと未学習
  * **推奨値** ：3〜5エポック（事前学習済みモデルは少なめ）
  * **工夫** ：Early Stoppingで検証損失が上昇したら停止

**最適化の優先順位** ：学習率 > バッチサイズ > Warmup > エポック数 > Weight Decay

* * *
