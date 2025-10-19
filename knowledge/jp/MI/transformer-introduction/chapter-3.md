# 第3章: 事前学習モデルと転移学習

**学習時間**: 25-30分 | **難易度**: 中級〜上級

## 📋 この章で学ぶこと

- 事前学習（Pre-training）の重要性と原理
- MatBERT、MolBERTなど材料科学向け事前学習モデル
- ファインチューニング（Fine-tuning）の戦略
- Few-shot学習とプロンプトエンジニアリング
- ドメイン適応（Domain Adaptation）

---

## 3.1 事前学習の重要性

### なぜ事前学習が必要か

**材料科学の課題**:
- ❌ ラベル付きデータが少ない（実験データは高コスト）
- ❌ ドメイン固有の知識が必要
- ❌ ゼロから学習すると時間とコストがかかる

**事前学習の利点**:
- ✅ 大規模な**ラベルなしデータ**で一般的な知識を獲得
- ✅ 少量のラベル付きデータで**高精度**を実現
- ✅ 開発期間の**大幅短縮**（数週間→数時間）

<div class="mermaid">
graph LR
    A[大規模ラベルなしデータ] --> B[事前学習]
    B --> C[汎用表現モデル]
    C --> D[ファインチューニング]
    E[少量ラベル付きデータ] --> D
    D --> F[タスク特化モデル]

    style B fill:#e1f5ff
    style D fill:#ffe1e1
</div>

### 事前学習のタスク

**自然言語処理での例**:
- **Masked Language Model (MLM)**: 一部の単語をマスクして予測
- **Next Sentence Prediction (NSP)**: 2文の連続性を予測

**材料科学での応用**:
- **Masked Atom Prediction**: 一部の原子をマスクして予測
- **Property Prediction**: 複数の材料特性を同時予測
- **Contrastive Learning**: 類似材料を近く、異なる材料を遠くに配置

---

## 3.2 MatBERT: Materials BERT

### 概要

**MatBERT**は、材料の組成式をBERTで学習したモデルです。

**特徴**:
- **500k材料**の組成式で事前学習
- **マスク原子予測**タスク
- 転移学習で様々な特性予測に適用可能

### 組成式のトークン化

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class CompositionTokenizer:
    def __init__(self):
        # カスタム語彙（周期表の元素）
        self.vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] + [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            # ... 全元素
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def tokenize(self, composition):
        """
        組成式をトークン化

        Args:
            composition: 'Fe2O3' のような組成式
        Returns:
            tokens: トークンのリスト
        """
        import re
        # 元素と数字を分割
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, composition)

        tokens = ['[CLS]']
        for element, count in matches:
            if element in self.vocab:
                # 元素を追加
                tokens.append(element)
                # 数が1より大きい場合、その回数だけ繰り返す（簡略化）
                if count and float(count) > 1:
                    for _ in range(int(float(count)) - 1):
                        tokens.append(element)
        tokens.append('[SEP]')

        return tokens

    def encode(self, compositions, max_length=32):
        """
        組成式をIDに変換

        Args:
            compositions: 組成式のリスト
            max_length: 最大長
        Returns:
            input_ids: (batch_size, max_length)
            attention_mask: (batch_size, max_length)
        """
        batch_input_ids = []
        batch_attention_mask = []

        for comp in compositions:
            tokens = self.tokenize(comp)
            ids = [self.token_to_id.get(token, 0) for token in tokens]

            # パディング
            attention_mask = [1] * len(ids)
            while len(ids) < max_length:
                ids.append(0)  # [PAD]
                attention_mask.append(0)

            # トランケーション
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]

            batch_input_ids.append(ids)
            batch_attention_mask.append(attention_mask)

        return torch.tensor(batch_input_ids), torch.tensor(batch_attention_mask)

# 使用例
tokenizer = CompositionTokenizer()

compositions = [
    'Fe2O3',     # 酸化鉄
    'LiCoO2',    # リチウムコバルト酸化物（電池材料）
    'BaTiO3'     # チタン酸バリウム（誘電体）
]

input_ids, attention_mask = tokenizer.encode(compositions)
print(f"Input IDs shape: {input_ids.shape}")
print(f"First composition tokens: {input_ids[0][:10]}")
```

### MatBERTモデル

```python
class MatBERT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8):
        super(MatBERT, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.d_model = d_model

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        token_embeddings = self.embedding(input_ids)

        # Positional embedding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)

        # 合計
        embeddings = token_embeddings + position_embeddings

        # Transformer
        # attention_maskをTransformer用に変換（0→-inf, 1→0）
        transformer_mask = (1 - attention_mask).bool()
        output = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_mask)

        return output

# 使用例
vocab_size = len(tokenizer.vocab)
model = MatBERT(vocab_size, d_model=512, num_layers=6, num_heads=8)

embeddings = model(input_ids, attention_mask)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 32, 512)
```

### 事前学習: Masked Atom Prediction

```python
def masked_atom_prediction_loss(model, input_ids, attention_mask, mask_prob=0.15):
    """
    マスク原子予測による事前学習

    Args:
        model: MatBERTモデル
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        mask_prob: マスクする確率
    Returns:
        loss: 損失
    """
    batch_size, seq_len = input_ids.shape

    # ランダムにマスク
    mask_token_id = tokenizer.token_to_id['[MASK]']
    mask = torch.rand(batch_size, seq_len) < mask_prob
    mask = mask & (attention_mask == 1)  # パディング部分は除外

    # 元のトークンを保存
    original_input_ids = input_ids.clone()

    # マスクを適用
    input_ids[mask] = mask_token_id

    # Forward
    embeddings = model(input_ids, attention_mask)

    # 予測ヘッド
    prediction_head = nn.Linear(model.d_model, vocab_size)
    logits = prediction_head(embeddings)

    # 損失計算（マスクされた位置のみ）
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    labels = original_input_ids.clone()
    labels[~mask] = -100  # マスクされていない部分は無視

    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

    return loss

# 事前学習ループ（簡略版）
def pretrain_matbert(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in dataloader:
            loss = masked_atom_prediction_loss(model, input_ids, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Pretraining Loss: {avg_loss:.4f}")

    return model
```

---

## 3.3 ファインチューニング戦略

### ファインチューニングとは

**定義**: 事前学習モデルを特定タスクに適応させる追加学習

**戦略**:
1. **Full Fine-tuning**: すべてのパラメータを更新
2. **Feature Extraction**: 埋め込み層のみ使用、予測ヘッドのみ学習
3. **Partial Fine-tuning**: 一部の層のみ更新

<div class="mermaid">
graph TD
    A[事前学習済みMatBERT] --> B{ファインチューニング戦略}
    B --> C[Full Fine-tuning]
    B --> D[Feature Extraction]
    B --> E[Partial Fine-tuning]

    C --> F[全層を更新]
    D --> G[埋め込み固定、予測ヘッドのみ学習]
    E --> H[上位層のみ更新]

    style C fill:#ffe1e1
    style D fill:#e1f5ff
    style E fill:#f5ffe1
</div>

### 実装: バンドギャップ予測

```python
class MatBERTForBandgap(nn.Module):
    def __init__(self, matbert_model, d_model=512):
        super(MatBERTForBandgap, self).__init__()
        self.matbert = matbert_model

        # 予測ヘッド
        self.bandgap_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        # MatBERT埋め込み
        embeddings = self.matbert(input_ids, attention_mask)

        # [CLS]トークンの埋め込みを使用
        cls_embedding = embeddings[:, 0, :]

        # バンドギャップ予測
        bandgap = self.bandgap_predictor(cls_embedding)
        return bandgap

# ファインチューニング
def finetune_for_bandgap(pretrained_model, train_loader, val_loader, strategy='full'):
    """
    バンドギャップ予測へのファインチューニング

    Args:
        pretrained_model: 事前学習済みMatBERT
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        strategy: 'full', 'feature', 'partial'
    """
    model = MatBERTForBandgap(pretrained_model)

    # 戦略に応じてパラメータの固定
    if strategy == 'feature':
        # MatBERTを固定
        for param in model.matbert.parameters():
            param.requires_grad = False
    elif strategy == 'partial':
        # 下位層を固定、上位層のみ更新
        for i, layer in enumerate(model.matbert.transformer_encoder.layers):
            if i < 3:  # 下位3層を固定
                for param in layer.parameters():
                    param.requires_grad = False

    # 最適化
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion = nn.MSELoss()

    # 訓練ループ
    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = 0
        for input_ids, attention_mask, bandgaps in train_loader:
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, bandgaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, bandgaps in val_loader:
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, bandgaps)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_matbert_bandgap.pt')

    return model
```

---

## 3.4 Few-shot学習

### 概要

**Few-shot学習**: 少量のサンプル（数個〜数十個）で新しいタスクを学習

**材料科学での重要性**:
- 新規材料のデータは非常に少ない
- 実験データの取得は高コスト
- 迅速なプロトタイピングが必要

### Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, matbert_model, d_model=512):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = matbert_model

    def forward(self, support_ids, support_mask, query_ids, query_mask, support_labels):
        """
        Prototypical Networksによる分類

        Args:
            support_ids: サポートセット入力 (n_support, seq_len)
            support_mask: サポートセットマスク
            query_ids: クエリ入力 (n_query, seq_len)
            query_mask: クエリマスク
            support_labels: サポートセットラベル (n_support,)
        Returns:
            predictions: クエリの予測ラベル
        """
        # サポートセットとクエリの埋め込み
        support_embeddings = self.encoder(support_ids, support_mask)[:, 0, :]  # [CLS]
        query_embeddings = self.encoder(query_ids, query_mask)[:, 0, :]

        # 各クラスのプロトタイプ（平均埋め込み）を計算
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = (support_labels == label)
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (num_classes, d_model)

        # クエリとプロトタイプ間の距離
        distances = torch.cdist(query_embeddings, prototypes)  # (n_query, num_classes)

        # 最も近いプロトタイプのクラスを予測
        predictions = torch.argmin(distances, dim=1)

        return predictions

# 使用例: 3-way 5-shot分類
# 3クラス、各クラス5サンプル
n_classes = 3
n_support_per_class = 5
n_query = 10

support_ids = torch.randint(0, vocab_size, (n_classes * n_support_per_class, 32))
support_mask = torch.ones_like(support_ids)
support_labels = torch.arange(n_classes).repeat_interleave(n_support_per_class)

query_ids = torch.randint(0, vocab_size, (n_query, 32))
query_mask = torch.ones_like(query_ids)

proto_net = PrototypicalNetwork(model)
predictions = proto_net(support_ids, support_mask, query_ids, query_mask, support_labels)
print(f"Predictions: {predictions}")
```

---

## 3.5 プロンプトエンジニアリング

### 材料科学でのプロンプト

**プロンプト**: モデルに追加情報を与えて性能を向上

**例**:
```python
# 通常: 'Fe2O3'
# プロンプト付き: '[OXIDE] Fe2O3 [BANDGAP]'
```

### 実装

```python
class PromptedMatBERT(nn.Module):
    def __init__(self, matbert_model, d_model=512):
        super(PromptedMatBERT, self).__init__()
        self.matbert = matbert_model

        # タスク別プロンプト埋め込み（学習可能）
        self.task_prompts = nn.Parameter(torch.randn(10, d_model))  # 10種類のタスク

    def forward(self, input_ids, attention_mask, task_id=0):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            task_id: タスクID (0-9)
        """
        batch_size = input_ids.size(0)

        # 通常の埋め込み
        embeddings = self.matbert(input_ids, attention_mask)

        # タスクプロンプトを先頭に追加
        task_prompt = self.task_prompts[task_id].unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat([task_prompt, embeddings], dim=1)

        return embeddings

# 使用例
prompted_model = PromptedMatBERT(model)

# タスク0: バンドギャップ予測
embeddings_task0 = prompted_model(input_ids, attention_mask, task_id=0)

# タスク1: 形成エネルギー予測
embeddings_task1 = prompted_model(input_ids, attention_mask, task_id=1)

print(f"Embeddings with prompt shape: {embeddings_task0.shape}")
```

---

## 3.6 ドメイン適応

### 概要

**ドメイン適応**: ソースドメインで訓練したモデルをターゲットドメインに適応

**例**:
- ソース: 無機材料データ
- ターゲット: 有機分子データ

### Adversarial Domain Adaptation

```python
class DomainClassifier(nn.Module):
    def __init__(self, d_model=512):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # ソース or ターゲット
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)

class DomainAdaptiveMatBERT(nn.Module):
    def __init__(self, matbert_model):
        super(DomainAdaptiveMatBERT, self).__init__()
        self.matbert = matbert_model
        self.domain_classifier = DomainClassifier()
        self.task_predictor = nn.Linear(512, 1)  # 例: バンドギャップ予測

    def forward(self, input_ids, attention_mask, alpha=1.0):
        """
        Args:
            alpha: ドメイン適応の強さ
        """
        embeddings = self.matbert(input_ids, attention_mask)[:, 0, :]

        # タスク予測
        task_output = self.task_predictor(embeddings)

        # ドメイン予測（勾配反転層を使用）
        # ここでは簡略化のため省略
        domain_output = self.domain_classifier(embeddings)

        return task_output, domain_output

# 訓練ループ（簡略版）
def train_domain_adaptive(model, source_loader, target_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    task_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (source_ids, source_mask, source_labels), (target_ids, target_mask, _) in zip(source_loader, target_loader):
            # ソースドメイン
            source_task, source_domain = model(source_ids, source_mask)
            source_domain_labels = torch.zeros(source_ids.size(0), dtype=torch.long)  # ソース = 0

            # ターゲットドメイン
            target_task, target_domain = model(target_ids, target_mask)
            target_domain_labels = torch.ones(target_ids.size(0), dtype=torch.long)  # ターゲット = 1

            # 損失
            task_loss = task_criterion(source_task, source_labels)
            domain_loss = domain_criterion(source_domain, source_domain_labels) + \
                          domain_criterion(target_domain, target_domain_labels)

            total_loss = task_loss + 0.1 * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Task Loss: {task_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}")
```

---

## 3.7 まとめ

### 重要ポイント

1. **事前学習**: 大規模ラベルなしデータで一般的知識を獲得
2. **ファインチューニング**: 少量データでタスク特化
3. **Few-shot学習**: 数個のサンプルで新タスク学習
4. **プロンプトエンジニアリング**: タスク情報を埋め込みで表現
5. **ドメイン適応**: 異なるドメイン間で知識転移

### 次章への準備

第4章では、拡散モデルによる分子生成と材料逆設計を学びます。

---

## 📝 演習問題

### 問題1: 概念理解
ファインチューニングの3つの戦略（Full、Feature Extraction、Partial）について、それぞれどのような場合に適しているか説明してください。

<details>
<summary>解答例</summary>

1. **Full Fine-tuning**:
   - **適用場面**: ターゲットドメインのデータが比較的多い（数千サンプル以上）
   - **利点**: 最高精度を達成可能
   - **欠点**: 過学習リスク、計算コスト大

2. **Feature Extraction**:
   - **適用場面**: データが非常に少ない（数十〜数百サンプル）
   - **利点**: 過学習を防ぎやすい、高速
   - **欠点**: ドメインが大きく異なる場合は精度低下

3. **Partial Fine-tuning**:
   - **適用場面**: 中程度のデータ量、ドメインが類似
   - **利点**: バランスの取れた性能とコスト
   - **欠点**: どの層を更新するか選択が難しい
</details>

### 問題2: 実装
以下のコードの空欄を埋めて、事前学習済みモデルをロードしてファインチューニングする関数を完成させてください。

```python
def load_and_finetune(pretrained_path, train_loader, val_loader):
    # 事前学習済みモデルをロード
    matbert = MatBERT(vocab_size=______, d_model=512)
    matbert.load_state_dict(torch.load(______))

    # ファインチューニング用モデルを構築
    model = MatBERTForBandgap(______)

    # 最適化
    optimizer = torch.optim.Adam(______.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # 訓練ループ
    for epoch in range(10):
        model.train()
        for input_ids, attention_mask, targets in train_loader:
            predictions = model(______, ______)
            loss = ______(predictions, targets)

            optimizer.zero_grad()
            ______.backward()
            optimizer.step()

    return model
```

<details>
<summary>解答例</summary>

```python
def load_and_finetune(pretrained_path, train_loader, val_loader):
    # 事前学習済みモデルをロード
    matbert = MatBERT(vocab_size=len(tokenizer.vocab), d_model=512)
    matbert.load_state_dict(torch.load(pretrained_path))

    # ファインチューニング用モデルを構築
    model = MatBERTForBandgap(matbert)

    # 最適化
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # 訓練ループ
    for epoch in range(10):
        model.train()
        for input_ids, attention_mask, targets in train_loader:
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```
</details>

### 問題3: 応用
材料科学で Few-shot学習が特に有用な3つのシナリオを挙げ、それぞれの理由を説明してください。

<details>
<summary>解答例</summary>

1. **新規材料の迅速評価**:
   - **シナリオ**: 新しいクラスの材料（例: 新型ペロブスカイト）
   - **理由**: 実験データがまだ少なく、数サンプルで特性予測が必要

2. **実験計画の効率化**:
   - **シナリオ**: 高コストな実験（単結晶成長、高圧合成）
   - **理由**: 少数の実験結果から次の実験条件を提案

3. **企業の独自材料開発**:
   - **シナリオ**: 競合に公開できない独自材料
   - **理由**: 社内データのみで学習、外部データは使えない
</details>

---

## 🚀 実装演習: Transformer for Materials

### 演習1: MatBERT実装（BERT for Materials）

```python
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class MaterialsBERT(nn.Module):
    def __init__(self, vocab_size=120, d_model=768, num_layers=12, num_heads=12):
        """
        Materials BERT implementation

        Args:
            vocab_size: 原子種数 + 特殊トークン
            d_model: 隠れ層次元
            num_layers: Transformerレイヤー数
            num_heads: Attentionヘッド数
        """
        super().__init__()

        # BERT configuration
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512
        )

        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) 原子番号シーケンス
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
        Returns:
            outputs: BERT outputs with pooler_output
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        return outputs

# 使用例
mat_bert = MaterialsBERT(vocab_size=120, d_model=768)

# ダミーデータ: Fe2O3 (酸化鉄)
# [CLS] Fe Fe O O O [SEP]
input_ids = torch.tensor([[101, 26, 26, 8, 8, 8, 102]])  # 101=[CLS], 102=[SEP]
attention_mask = torch.ones_like(input_ids)

outputs = mat_bert(input_ids, attention_mask)
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")  # (1, 7, 768)
print(f"Pooler output shape: {outputs.pooler_output.shape}")  # (1, 768)
```

### 演習2: MatGPT実装（GPT for Materials Generation）

```python
from transformers import GPT2Config, GPT2LMHeadModel

class MaterialsGPT(nn.Module):
    def __init__(self, vocab_size=120, d_model=768, num_layers=12, num_heads=12):
        """
        Materials GPT for generative tasks

        Args:
            vocab_size: 原子種数 + 特殊トークン
            d_model: 隠れ層次元
            num_layers: Transformerレイヤー数
            num_heads: Attentionヘッド数
        """
        super().__init__()

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=d_model,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )

        self.gpt = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) for training
        """
        outputs = self.gpt(input_ids=input_ids, labels=labels)
        return outputs

    def generate_composition(self, start_tokens, max_length=50, temperature=1.0):
        """
        組成式生成

        Args:
            start_tokens: (1, start_len) 開始トークン
            max_length: 最大生成長
            temperature: サンプリング温度
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - start_tokens.size(1)):
                outputs = self.gpt(start_tokens)
                logits = outputs.logits[:, -1, :] / temperature

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                start_tokens = torch.cat([start_tokens, next_token], dim=1)

                # [SEP]トークンで停止
                if next_token.item() == 102:
                    break

        return start_tokens

# 使用例
mat_gpt = MaterialsGPT(vocab_size=120, d_model=768)

# 生成: [CLS] Fe ... (酸化物を生成)
start = torch.tensor([[101, 26]])  # [CLS] Fe
generated = mat_gpt.generate_composition(start, max_length=20)
print(f"Generated sequence: {generated}")
```

### 演習3: MatT5実装（T5 for Materials Seq2Seq）

```python
from transformers import T5Config, T5ForConditionalGeneration

class MaterialsT5(nn.Module):
    def __init__(self, vocab_size=120, d_model=512, num_layers=6):
        """
        Materials T5 for sequence-to-sequence tasks
        (e.g., composition → properties description)

        Args:
            vocab_size: 語彙サイズ
            d_model: モデル次元
            num_layers: エンコーダ・デコーダ層数
        """
        super().__init__()

        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=64,
            d_ff=d_model * 4,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=8,
            dropout_rate=0.1
        )

        self.t5 = T5ForConditionalGeneration(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids: (batch_size, src_len) 入力系列
            labels: (batch_size, tgt_len) ターゲット系列
        """
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def predict_properties(self, composition_ids, max_length=50):
        """
        組成式から特性記述を生成

        Args:
            composition_ids: (batch_size, seq_len) 組成式
            max_length: 最大生成長
        """
        self.eval()
        with torch.no_grad():
            outputs = self.t5.generate(
                composition_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        return outputs

# 使用例
mat_t5 = MaterialsT5(vocab_size=120, d_model=512)

# 入力: Fe2O3 → 出力: "semiconductor bandgap 2.0 eV"
input_ids = torch.tensor([[26, 26, 8, 8, 8]])  # Fe Fe O O O
outputs = mat_t5.predict_properties(input_ids, max_length=20)
print(f"Predicted properties: {outputs}")
```

---

## 🧪 SMILES/SELFIES トークン化の実装

### SMILES Tokenizer

```python
import re
from typing import List, Dict

class SMILESTokenizer:
    """
    SMILES文字列の完全トークン化

    対応:
    - 芳香族性 (c, n, o, s)
    - 立体化学 (@, @@, /, \\)
    - 分岐 ((, ))
    - 結合 (-, =, #, :)
    - 環 (数字)
    """

    def __init__(self):
        # 正規表現パターン（優先順位順）
        self.pattern = r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'

        # 特殊トークン
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[MASK]': 3,
            '[UNK]': 4
        }

        # 語彙の構築
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def _build_vocab(self) -> List[str]:
        """語彙を構築"""
        vocab = list(self.special_tokens.keys())

        # 元素記号
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                   'c', 'n', 'o', 's', 'p']  # 芳香族

        # 記号
        symbols = ['(', ')', '[', ']', '=', '#', '-', '+', '\\', '/',
                  ':', '.', '@', '@@']

        # 数字
        numbers = [str(i) for i in range(10)]

        vocab.extend(elements + symbols + numbers)

        return vocab

    def tokenize(self, smiles: str) -> List[str]:
        """
        SMILES文字列をトークン化

        Args:
            smiles: SMILES文字列

        Returns:
            tokens: トークンのリスト

        Examples:
            >>> tokenizer = SMILESTokenizer()
            >>> tokenizer.tokenize("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
            ['C', 'C', '(', 'C', ')', 'C', 'c', '1', 'c', 'c', 'c', '(', ...]
        """
        tokens = re.findall(self.pattern, smiles)
        return ['[CLS]'] + tokens + ['[SEP]']

    def encode(self, smiles: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        SMILES文字列をIDに変換

        Args:
            smiles: SMILES文字列
            max_length: 最大長

        Returns:
            encoding: input_ids, attention_mask
        """
        tokens = self.tokenize(smiles)

        # トークンをIDに変換
        ids = [self.token_to_id.get(token, self.token_to_id['[UNK]'])
               for token in tokens]

        # パディング
        attention_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.token_to_id['[PAD]'])
            attention_mask.append(0)

        # トランケーション
        ids = ids[:max_length]
        attention_mask = attention_mask[:max_length]

        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

    def decode(self, ids: List[int]) -> str:
        """IDからSMILES文字列に復元"""
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        # 特殊トークンを除去
        tokens = [t for t in tokens if t not in self.special_tokens]
        return ''.join(tokens)

# 使用例
tokenizer = SMILESTokenizer()

# イブプロフェン
smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
tokens = tokenizer.tokenize(smiles)
print(f"Tokens: {tokens[:10]}...")

encoding = tokenizer.encode(smiles)
print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"First 10 IDs: {encoding['input_ids'][0][:10]}")

# デコード
decoded = tokenizer.decode(encoding['input_ids'][0].tolist())
print(f"Decoded: {decoded}")
```

### SELFIES Tokenizer

```python
try:
    import selfies as sf
except ImportError:
    print("Install selfies: pip install selfies")

class SELFIESTokenizer:
    """
    SELFIES (SELF-referencIng Embedded Strings) Tokenizer

    利点:
    - 100%有効な分子を生成
    - 文法的に正しい
    - SMILESより頑健
    """

    def __init__(self):
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[MASK]': 3
        }

        # 一般的なSELFIESトークン
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def _build_vocab(self) -> List[str]:
        """
        SELFIES語彙を構築

        一般的なトークン:
        [C], [N], [O], [=C], [=N], [Ring1], [Branch1], etc.
        """
        vocab = list(self.special_tokens.keys())

        # 基本トークン
        common_tokens = [
            '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
            '[=C]', '[=N]', '[=O]', '[#C]', '[#N]',
            '[Ring1]', '[Ring2]', '[Branch1]', '[Branch2]',
            '[O-1]', '[N+1]', '[nop]'
        ]

        vocab.extend(common_tokens)
        return vocab

    def smiles_to_selfies(self, smiles: str) -> str:
        """SMILESをSELFIESに変換"""
        try:
            selfies = sf.encoder(smiles)
            return selfies
        except Exception as e:
            print(f"Encoding error: {e}")
            return ""

    def selfies_to_smiles(self, selfies: str) -> str:
        """SELFIESをSMILESに変換"""
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception as e:
            print(f"Decoding error: {e}")
            return ""

    def tokenize(self, selfies: str) -> List[str]:
        """
        SELFIES文字列をトークン化

        Args:
            selfies: SELFIES文字列

        Returns:
            tokens: トークンのリスト

        Examples:
            >>> tokenizer = SELFIESTokenizer()
            >>> tokenizer.tokenize("[C][C][Branch1][C][C][C]")
            ['[CLS]', '[C]', '[C]', '[Branch1]', '[C]', '[C]', '[C]', '[SEP]']
        """
        tokens = list(sf.split_selfies(selfies))
        return ['[CLS]'] + tokens + ['[SEP]']

    def encode(self, selfies: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """SELFIES文字列をIDに変換"""
        tokens = self.tokenize(selfies)

        # トークンをIDに変換（未知トークンは動的に追加）
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab.append(token)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token
            ids.append(self.token_to_id[token])

        # パディング
        attention_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.token_to_id['[PAD]'])
            attention_mask.append(0)

        # トランケーション
        ids = ids[:max_length]
        attention_mask = attention_mask[:max_length]

        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

# 使用例
if 'sf' in dir():
    tokenizer_selfies = SELFIESTokenizer()

    # SMILESからSELFIESに変換
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    selfies = tokenizer_selfies.smiles_to_selfies(smiles)
    print(f"SELFIES: {selfies}")

    # トークン化
    tokens = tokenizer_selfies.tokenize(selfies)
    print(f"Tokens: {tokens[:10]}...")

    # エンコード
    encoding = tokenizer_selfies.encode(selfies)
    print(f"Encoded shape: {encoding['input_ids'].shape}")
```

---

## ⚠️ 実践的な落とし穴と対処法

### 1. ファインチューニングの過学習
**問題**: 少量データでの訓練で検証損失が発散

```python
# ❌ 問題: 全パラメータを大きな学習率で更新
def wrong_finetuning():
    model = MatBERTForBandgap(pretrained_matbert)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 大きすぎ！

    for epoch in range(100):  # エポック数も多すぎ
        for batch in train_loader:
            loss = compute_loss(batch)
            loss.backward()
            optimizer.step()

# ✅ 解決策: Layer-wise learning rate decay + Early stopping
def correct_finetuning():
    model = MatBERTForBandgap(pretrained_matbert)

    # Layer-wise learning rate
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.matbert.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': 2e-5  # 事前学習部分は小さく
        },
        {
            'params': [p for n, p in model.matbert.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': 2e-5
        },
        {
            'params': model.bandgap_predictor.parameters(),
            'lr': 1e-4  # 予測ヘッドは大きく
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ベストモデルを復元
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

### 2. ドメインシフトの問題
**問題**: 無機材料で事前学習したモデルを有機分子に適用

```python
# ❌ 問題: ドメインが異なるのに直接適用
def wrong_domain_adaptation():
    # 無機材料で事前学習
    matbert = pretrained_on_inorganic_materials()

    # 有機分子データで直接ファインチューニング
    # → 性能が低い！
    finetune_on_organic_molecules(matbert)

# ✅ 解決策: Intermediate task transfer
def correct_domain_adaptation():
    # Step 1: 無機材料で事前学習
    matbert = pretrained_on_inorganic_materials()

    # Step 2: 中間タスク（無機と有機の中間）で継続学習
    # 例: 金属有機骨格 (MOF) データ
    matbert = continual_pretrain_on_mof(matbert)

    # Step 3: 有機分子データでファインチューニング
    model = finetune_on_organic_molecules(matbert)

    return model

# または: Domain-adversarial training
class DomainAdversarialTraining:
    def train(self, source_data, target_data):
        for source_batch, target_batch in zip(source_data, target_data):
            # Source domain: タスク損失
            source_output = model(source_batch)
            task_loss = compute_task_loss(source_output, source_batch.labels)

            # Both domains: ドメイン分類損失（逆転勾配）
            source_domain_pred = domain_classifier(source_output, reverse_gradient=True)
            target_domain_pred = domain_classifier(target_output, reverse_gradient=True)

            domain_loss = compute_domain_loss(source_domain_pred, target_domain_pred)

            total_loss = task_loss + 0.1 * domain_loss
            total_loss.backward()
            optimizer.step()
```

### 3. Masked Language Modelingのマスク戦略ミス
**問題**: マスクパターンが偏っている

```python
# ❌ 問題: ランダムにマスク（化学的に無意味）
def wrong_masking(composition_ids):
    mask_prob = 0.15
    mask = torch.rand(composition_ids.shape) < mask_prob
    composition_ids[mask] = MASK_TOKEN_ID
    return composition_ids

# ✅ 解決策: 化学的に意味のあるマスク
def chemically_aware_masking(composition_ids, element_groups):
    """
    元素グループを考慮したマスク

    Args:
        composition_ids: (batch, seq_len)
        element_groups: {group_id: [element_ids]}
            例: {0: [26, 27, 28], 1: [8, 16]}  # 遷移金属、カルコゲン
    """
    mask_prob = 0.15
    masked_ids = composition_ids.clone()

    for i in range(composition_ids.size(0)):
        # 化学的グループ単位でマスク
        for group_id, element_ids in element_groups.items():
            group_positions = torch.isin(composition_ids[i], torch.tensor(element_ids))
            if group_positions.sum() > 0:
                # グループ内の一部をマスク
                mask_within_group = torch.rand(group_positions.sum()) < mask_prob
                group_indices = torch.where(group_positions)[0]
                masked_positions = group_indices[mask_within_group]
                masked_ids[i, masked_positions] = MASK_TOKEN_ID

    return masked_ids

# 使用例
element_groups = {
    0: [26, 27, 28, 29],  # Fe, Co, Ni, Cu（遷移金属）
    1: [8, 16, 34],       # O, S, Se（カルコゲン）
    2: [3, 11, 19]        # Li, Na, K（アルカリ金属）
}

masked_composition = chemically_aware_masking(composition_ids, element_groups)
```

### 4. Few-shot学習のサポートセット選択ミス
**問題**: サポートセットが偏っている

```python
# ❌ 問題: ランダムにサポートセットを選択
def wrong_support_selection(dataset, k=5):
    indices = torch.randperm(len(dataset))[:k]
    return dataset[indices]

# ✅ 解決策: 多様性を考慮したサポートセット選択
def diverse_support_selection(dataset, embeddings, k=5):
    """
    K-meansで多様なサンプルを選択

    Args:
        dataset: データセット
        embeddings: (N, d) サンプルの埋め込み
        k: サポートセット size
    """
    from sklearn.cluster import KMeans

    # K-meansでクラスタリング
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings.numpy())

    # 各クラスタの中心に最も近いサンプルを選択
    support_indices = []
    for i in range(k):
        cluster_indices = torch.where(torch.tensor(labels) == i)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[i]

        # 中心に最も近いサンプル
        distances = torch.norm(cluster_embeddings - torch.tensor(cluster_center), dim=1)
        closest_idx = cluster_indices[torch.argmin(distances)]
        support_indices.append(closest_idx.item())

    return dataset[support_indices]

# 使用例
# データセット埋め込みを事前計算
embeddings = compute_embeddings(dataset, matbert)
support_set = diverse_support_selection(dataset, embeddings, k=10)
```

### 5. プロンプトエンジニアリングの最適化不足
**問題**: 固定プロンプトで性能が低い

```python
# ❌ 問題: 手動で設計した固定プロンプト
class FixedPromptModel(nn.Module):
    def __init__(self, matbert):
        super().__init__()
        self.matbert = matbert
        # 固定プロンプト
        self.prompt = nn.Parameter(torch.randn(1, 10, 768), requires_grad=False)

# ✅ 解決策: 学習可能なプロンプト（Prefix-Tuning）
class LearnablePromptModel(nn.Module):
    def __init__(self, matbert, prompt_length=10, num_tasks=5):
        super().__init__()
        self.matbert = matbert
        self.prompt_length = prompt_length

        # タスク別の学習可能なプロンプト
        self.task_prompts = nn.Parameter(torch.randn(num_tasks, prompt_length, 768))

        # MatBERTのパラメータは固定
        for param in self.matbert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, task_id=0):
        batch_size = input_ids.size(0)

        # 入力埋め込み
        input_embeddings = self.matbert.embeddings(input_ids)

        # タスク固有プロンプトを追加
        prompt = self.task_prompts[task_id].unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat([prompt, input_embeddings], dim=1)

        # Transformerに通す
        outputs = self.matbert.encoder(embeddings)

        return outputs

# 訓練
model = LearnablePromptModel(pretrained_matbert, prompt_length=10, num_tasks=5)

# プロンプトのみ最適化（パラメータ数を大幅削減）
optimizer = torch.optim.Adam([model.task_prompts], lr=1e-3)
```

---

## ✅ 第3章完了チェックリスト

### 概念理解（10項目）
- [ ] 事前学習の重要性と利点を説明できる
- [ ] Masked Language Modelingの原理を理解している
- [ ] Full/Feature Extraction/Partial Fine-tuningの違いを説明できる
- [ ] Few-shot学習の原理（Prototypical Networks）を理解している
- [ ] プロンプトエンジニアリングの概念を理解している
- [ ] ドメイン適応の必要性を説明できる
- [ ] 事前学習タスクと下流タスクの関係を理解している
- [ ] Transfer Learningの効果を定量的に評価できる
- [ ] MatBERT、MolBERTなど材料特化モデルの特徴を理解している
- [ ] BERT/GPT/T5の違いと適用場面を説明できる

### 実装スキル（15項目）
- [ ] `MatBERT`を実装できる
- [ ] `MatGPT`を実装できる
- [ ] `MatT5`を実装できる
- [ ] SMILESトークナイザーを実装できる
- [ ] SELFIESトークナイザーを実装できる
- [ ] Masked Atom Predictionを実装できる
- [ ] Fine-tuning戦略（Full/Feature/Partial）を実装できる
- [ ] Prototypical Networksを実装できる
- [ ] 学習可能なプロンプトを実装できる
- [ ] Domain-adversarial trainingを実装できる
- [ ] Early stoppingを実装できる
- [ ] Layer-wise learning rateを設定できる
- [ ] 事前学習済みモデルを保存・読み込みできる
- [ ] Hugging Face Transformersライブラリを活用できる
- [ ] カスタムトークナイザーをTransformersに統合できる

### デバッグスキル（5項目）
- [ ] 過学習を検出し、正則化で対処できる
- [ ] ドメインシフトを検出し、適応手法を適用できる
- [ ] マスク戦略の妥当性を評価できる
- [ ] Few-shotのサポートセット品質を評価できる
- [ ] プロンプトの効果を可視化・分析できる

### 応用力（5項目）
- [ ] 新しい材料特性予測タスクに事前学習モデルを適用できる
- [ ] 複数の事前学習タスクを組み合わせて性能向上できる
- [ ] ドメイン適応戦略を設計できる
- [ ] Few-shot学習をデータ拡張と組み合わせられる
- [ ] プロンプトエンジニアリングで性能を最適化できる

### データ処理（5項目）
- [ ] SMILESデータを前処理できる
- [ ] SELFIESに変換できる
- [ ] データ拡張（SMILES enumeration）を実装できる
- [ ] ドメイン別にデータを分割できる
- [ ] Few-shot用のエピソードを生成できる

### 評価スキル（5項目）
- [ ] 事前学習の効果を定量評価できる（vs from scratch）
- [ ] Fine-tuning戦略を比較評価できる
- [ ] Few-shot性能を適切に評価できる（N-way K-shot）
- [ ] ドメイン適応の効果を測定できる
- [ ] プロンプトの影響を分析できる

### 理論的背景（5項目）
- [ ] MatBERT/MolBERT論文を読んだ
- [ ] BERT論文（Devlin et al., 2019）を読んだ
- [ ] GPT論文を読んだ
- [ ] Few-shot学習の論文を1本以上読んだ
- [ ] Transfer Learning理論を理解している

### 完了基準
- **最低基準**: 40項目以上達成（80%）
- **推奨基準**: 45項目以上達成（90%）
- **優秀基準**: 50項目全て達成（100%）

---

**次章**: **[第4章: 生成モデルと逆設計](chapter-4.html)**

---

**作成者**: 橋本佑介（東北大学）
**最終更新**: 2025年10月19日
