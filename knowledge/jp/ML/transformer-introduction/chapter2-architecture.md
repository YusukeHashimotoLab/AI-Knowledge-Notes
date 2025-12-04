---
title: 第2章：Transformerアーキテクチャ（Transformer Architecture）
chapter_title: 第2章：Transformerアーキテクチャ（Transformer Architecture）
subtitle: Encoder-Decoderの完全理解とPyTorchによる実装
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Transformerの全体アーキテクチャ（Encoder-Decoder構造）を理解する
  * ✅ Encoderの構成要素（Multi-Head Attention、FFN、Layer Norm、Residual）を説明できる
  * ✅ Decoderの特徴（Masked Self-Attention、Cross-Attention）を理解する
  * ✅ PyTorchでTransformerを完全実装できる
  * ✅ 自己回帰生成のメカニズムを理解する
  * ✅ 実践的な機械翻訳システムを構築できる

* * *

## 2.1 Transformer全体像

### アーキテクチャの概要

**Transformer** は、"Attention is All You Need" (Vaswani et al., 2017)で提案された革新的なアーキテクチャで、RNN・CNNを使わず、**Attention機構のみ** で系列変換を実現します。
    
    
    ```mermaid
    graph TB
        Input["入力系列(Source)"] --> Encoder["Encoder(N層スタック)"]
        Encoder --> Memory["エンコード済み表現(Memory)"]
        Memory --> Decoder["Decoder(N層スタック)"]
        Target["目標系列(Target)"] --> Decoder
        Decoder --> Output["出力系列(Prediction)"]
    
        style Encoder fill:#b3e5fc
        style Decoder fill:#ffab91
        style Memory fill:#fff9c4
    ```

### 主要コンポーネント

コンポーネント | 役割 | 特徴  
---|---|---  
**Encoder** | 入力系列を文脈表現に変換 | 6層スタック、並列処理可能  
**Decoder** | エンコード表現から出力系列を生成 | 6層スタック、自己回帰生成  
**Multi-Head Attention** | 複数の観点から依存関係を捉える | 8ヘッド並列実行  
**Feed-Forward Network** | 各位置を独立に変換 | 2層MLP（ReLU活性化）  
**Positional Encoding** | 位置情報を注入 | Sin/Cos関数ベース  
**Layer Normalization** | 学習を安定化 | 各サブレイヤー後に適用  
**Residual Connection** | 勾配流を改善 | Skip connection  
  
### RNNとの違い

項目 | RNN/LSTM | Transformer  
---|---|---  
**処理方式** | 逐次的（sequential） | 並列的（parallel）  
**長期依存** | 距離が離れると弱まる | 距離に関係なく直接接続  
**計算複雑度** | $O(n)$時間 | $O(1)$時間（並列化可能）  
**メモリ** | 隠れ状態で圧縮 | 全位置の情報を保持  
**訓練速度** | 遅い | 高速（GPU活用）  
  
> 「Transformerは並列処理により、RNNより10倍以上高速に訓練できます！」

### 基本構造の可視化
    
    
    import torch
    import torch.nn as nn
    import math
    
    # Transformerの基本パラメータ
    print("=== Transformerの基本設定 ===")
    d_model = 512        # モデルの次元数
    nhead = 8            # Attentionヘッド数
    num_layers = 6       # Encoder/Decoderの層数
    d_ff = 2048          # Feed-Forwardの隠れ層サイズ
    dropout = 0.1        # Dropout率
    max_len = 5000       # 最大系列長
    
    print(f"モデル次元: d_model = {d_model}")
    print(f"Attentionヘッド数: nhead = {nhead}")
    print(f"各ヘッドの次元: d_k = d_v = {d_model // nhead}")
    print(f"Encoder/Decoder層数: {num_layers}")
    print(f"FFN隠れ層サイズ: {d_ff}")
    print(f"総パラメータ数（概算）: {(num_layers * 2) * (4 * d_model**2 + 2 * d_model * d_ff):,}")
    
    # 入出力サイズの例
    batch_size = 32
    src_len = 20  # ソース系列長
    tgt_len = 15  # ターゲット系列長
    
    print(f"\n=== 入出力例 ===")
    print(f"入力（ソース）: ({batch_size}, {src_len}, {d_model})")
    print(f"入力（ターゲット）: ({batch_size}, {tgt_len}, {d_model})")
    print(f"出力: ({batch_size}, {tgt_len}, {d_model})")
    

* * *

## 2.2 Encoder構造

### Encoderの役割

Encoderは入力系列を、各位置の文脈を考慮した高次元表現に変換します。N層（通常6層）のEncoderLayerをスタックします。
    
    
    ```mermaid
    graph TB
        Input["入力埋め込み + 位置エンコーディング"] --> E1["Encoder Layer 1"]
        E1 --> E2["Encoder Layer 2"]
        E2 --> E3["..."]
        E3 --> EN["Encoder Layer N"]
        EN --> Output["エンコード済み表現"]
    
        style Input fill:#e1f5ff
        style Output fill:#b3e5fc
    ```

### EncoderLayerの構造

各EncoderLayerは、以下の2つのサブレイヤーで構成されます：

  1. **Multi-Head Self-Attention** ：入力系列内の依存関係を捉える
  2. **Position-wise Feed-Forward Network** ：各位置を独立に変換

各サブレイヤーには、**Residual Connection** と**Layer Normalization** が適用されます。
    
    
    ```mermaid
    graph TB
        X["入力 x"] --> MHA["Multi-HeadSelf-Attention"]
        MHA --> Add1["Add & Norm"]
        X --> Add1
        Add1 --> FFN["Feed-ForwardNetwork"]
        Add1 --> Add2["Add & Norm"]
        FFN --> Add2
        Add2 --> Y["出力 y"]
    
        style MHA fill:#b3e5fc
        style FFN fill:#ffccbc
        style Add1 fill:#c5e1a5
        style Add2 fill:#c5e1a5
    ```

### Multi-Head Attentionの数式

Multi-Head Attentionは、複数の異なる表現部分空間で並列にAttentionを計算します：

$$ \begin{align} \text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\\ \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\\ \text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \end{align} $$ 

ここで：

  * $h$：ヘッド数（通常8）
  * $d_k = d_v = d_{\text{model}} / h$：各ヘッドの次元
  * $W_i^Q, W_i^K, W_i^V, W^O$：学習可能な射影行列

### EncoderLayerの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MultiHeadAttention(nn.Module):
        """Multi-Head Attention機構"""
        def __init__(self, d_model, nhead, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
    
            self.d_model = d_model
            self.nhead = nhead
            self.d_k = d_model // nhead
    
            # Q, K, V の線形変換
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
    
            # 出力の線形変換
            self.W_o = nn.Linear(d_model, d_model)
    
            self.dropout = nn.Dropout(dropout)
    
        def split_heads(self, x):
            """(batch, seq_len, d_model) -> (batch, nhead, seq_len, d_k)"""
            batch_size, seq_len, d_model = x.size()
            return x.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
    
        def combine_heads(self, x):
            """(batch, nhead, seq_len, d_k) -> (batch, seq_len, d_model)"""
            batch_size, nhead, seq_len, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
        def forward(self, query, key, value, mask=None):
            """
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len) or (batch, seq_len, seq_len)
            """
            # 線形変換
            Q = self.W_q(query)  # (batch, seq_len, d_model)
            K = self.W_k(key)
            V = self.W_v(value)
    
            # ヘッドに分割
            Q = self.split_heads(Q)  # (batch, nhead, seq_len, d_k)
            K = self.split_heads(K)
            V = self.split_heads(V)
    
            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            # scores: (batch, nhead, seq_len, seq_len)
    
            # マスク適用
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
    
            # Valueとの積
            attn_output = torch.matmul(attn_weights, V)
            # attn_output: (batch, nhead, seq_len, d_k)
    
            # ヘッドを結合
            attn_output = self.combine_heads(attn_output)
            # attn_output: (batch, seq_len, d_model)
    
            # 出力線形変換
            output = self.W_o(attn_output)
    
            return output, attn_weights
    
    
    class PositionwiseFeedForward(nn.Module):
        """Position-wise Feed-Forward Network"""
        def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            # x: (batch, seq_len, d_model)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    
    class EncoderLayer(nn.Module):
        """Transformer Encoder Layer"""
        def __init__(self, d_model, nhead, d_ff, dropout=0.1):
            super(EncoderLayer, self).__init__()
    
            # Multi-Head Self-Attention
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
    
            # Feed-Forward Network
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    
            # Layer Normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
    
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
    
        def forward(self, x, mask=None):
            """
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len) - パディングマスク
            """
            # Self-Attention + Residual + Norm
            attn_output, attn_weights = self.self_attn(x, x, x, mask)
            x = x + self.dropout1(attn_output)  # Residual connection
            x = self.norm1(x)  # Layer normalization
    
            # Feed-Forward + Residual + Norm
            ffn_output = self.ffn(x)
            x = x + self.dropout2(ffn_output)  # Residual connection
            x = self.norm2(x)  # Layer normalization
    
            return x, attn_weights
    
    
    # 動作確認
    print("=== EncoderLayerの動作確認 ===")
    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 32
    seq_len = 20
    
    encoder_layer = EncoderLayer(d_model, nhead, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = encoder_layer(x)
    
    print(f"入力: {x.shape}")
    print(f"出力: {output.shape}")
    print(f"Attention重み: {attn_weights.shape}")
    print("→ 入力と出力のサイズが同じ（残差接続のため）")
    
    # パラメータ数
    total_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"\nEncoderLayer パラメータ数: {total_params:,}")
    

### 完全なEncoderの実装
    
    
    import torch
    import torch.nn as nn
    import math
    
    class PositionalEncoding(nn.Module):
        """位置エンコーディング（Sin/Cos）"""
        def __init__(self, d_model, max_len=5000, dropout=0.1):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(dropout)
    
            # 位置エンコーディング行列を作成
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                               -(math.log(10000.0) / d_model))
    
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            """
            x: (batch, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class TransformerEncoder(nn.Module):
        """完全なTransformer Encoder"""
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                     d_ff=2048, dropout=0.1, max_len=5000):
            super(TransformerEncoder, self).__init__()
    
            self.d_model = d_model
    
            # 単語埋め込み
            self.embedding = nn.Embedding(vocab_size, d_model)
    
            # 位置エンコーディング
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
    
            # EncoderLayerをスタック
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ])
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, src, src_mask=None):
            """
            src: (batch, src_len) - トークンID
            src_mask: (batch, 1, src_len) - パディングマスク
            """
            # 埋め込み + スケーリング
            x = self.embedding(src) * math.sqrt(self.d_model)
    
            # 位置エンコーディング追加
            x = self.pos_encoding(x)
    
            # 各EncoderLayerを通過
            attn_weights_list = []
            for layer in self.layers:
                x, attn_weights = layer(x, src_mask)
                attn_weights_list.append(attn_weights)
    
            return x, attn_weights_list
    
    
    # 動作確認
    print("\n=== 完全なEncoderの動作確認 ===")
    vocab_size = 10000
    encoder = TransformerEncoder(vocab_size, d_model=512, nhead=8, num_layers=6)
    
    # ダミーデータ
    batch_size = 16
    src_len = 25
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    
    # パディングマスク作成（例：最後の5トークンがパディング）
    src_mask = torch.ones(batch_size, 1, src_len)
    src_mask[:, :, -5:] = 0
    
    # Encoder実行
    encoder_output, attn_weights_list = encoder(src, src_mask)
    
    print(f"入力トークン: {src.shape}")
    print(f"Encoder出力: {encoder_output.shape}")
    print(f"Attention重みの数: {len(attn_weights_list)} (層ごと)")
    print(f"各Attention重み: {attn_weights_list[0].shape}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

### Layer NormalizationとResidual Connectionの重要性
    
    
    import torch
    import torch.nn as nn
    
    # Layer Normalizationの効果
    print("=== Layer Normalizationの効果 ===")
    
    x = torch.randn(32, 20, 512)  # (batch, seq_len, d_model)
    
    # Layer Normalization前
    print(f"正規化前 - 平均: {x.mean():.4f}, 標準偏差: {x.std():.4f}")
    
    layer_norm = nn.LayerNorm(512)
    x_normalized = layer_norm(x)
    
    # Layer Normalization後
    print(f"正規化後 - 平均: {x_normalized.mean():.4f}, 標準偏差: {x_normalized.std():.4f}")
    print("→ 各サンプル・位置で平均0、標準偏差1に正規化")
    
    # Residual Connectionの効果
    print("\n=== Residual Connectionの効果 ===")
    
    class WithoutResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)  # Residual接続なし
            return x
    
    class WithResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)  # Residual接続あり
            return x
    
    # 勾配流を比較
    model_without = WithoutResidual(d_model=512, num_layers=10)
    model_with = WithResidual(d_model=512, num_layers=10)
    
    x = torch.randn(1, 512, requires_grad=True)
    
    # Forward + Backward
    out_without = model_without(x)
    out_without.sum().backward()
    grad_without = x.grad.norm().item()
    
    x.grad = None
    out_with = model_with(x)
    out_with.sum().backward()
    grad_with = x.grad.norm().item()
    
    print(f"Residual接続なし - 勾配ノルム: {grad_without:.6f}")
    print(f"Residual接続あり - 勾配ノルム: {grad_with:.6f}")
    print("→ Residual接続により勾配が消失せず、深い層でも学習可能")
    

* * *

## 2.3 Decoder構造

### Decoderの役割

Decoderは、Encoderの出力（メモリ）と既に生成したトークンから、次のトークンを**自己回帰的** に生成します。
    
    
    ```mermaid
    graph TB
        Target["ターゲット系列(シフト済み)"] --> D1["Decoder Layer 1"]
        Memory["Encoder出力(Memory)"] --> D1
        D1 --> D2["Decoder Layer 2"]
        Memory --> D2
        D2 --> D3["..."]
        Memory --> D3
        D3 --> DN["Decoder Layer N"]
        Memory --> DN
        DN --> Output["出力(次トークン予測)"]
    
        style Target fill:#e1f5ff
        style Memory fill:#fff9c4
        style Output fill:#ffab91
    ```

### DecoderLayerの構造

各DecoderLayerは、**3つのサブレイヤー** で構成されます：

  1. **Masked Multi-Head Self-Attention** ：未来のトークンを見ないようマスク
  2. **Cross-Attention** ：Encoderの出力（メモリ）を参照
  3. **Position-wise Feed-Forward Network** ：各位置を独立に変換

    
    
    ```mermaid
    graph TB
        X["入力 x"] --> MMHA["Masked Multi-HeadSelf-Attention"]
        MMHA --> Add1["Add & Norm"]
        X --> Add1
    
        Add1 --> CA["Cross-Attention(Encoder出力参照)"]
        Memory["Encoder Memory"] --> CA
        CA --> Add2["Add & Norm"]
        Add1 --> Add2
    
        Add2 --> FFN["Feed-ForwardNetwork"]
        FFN --> Add3["Add & Norm"]
        Add2 --> Add3
        Add3 --> Y["出力 y"]
    
        style MMHA fill:#ffab91
        style CA fill:#ce93d8
        style FFN fill:#ffccbc
        style Memory fill:#fff9c4
    ```

### Masked Self-Attentionの重要性

**Causal Masking（因果マスク）** により、位置$i$は位置$i$以前のトークンのみを参照できます。これにより、訓練時でも推論時と同じ自己回帰的な条件を保ちます。

$$ \text{Mask}_{ij} = \begin{cases} 0 & \text{if } i < j \text{ (未来のトークン)} \\\ 1 & \text{if } i \geq j \text{ (過去のトークン)} \end{cases} $$ 

### DecoderLayerの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class DecoderLayer(nn.Module):
        """Transformer Decoder Layer"""
        def __init__(self, d_model, nhead, d_ff, dropout=0.1):
            super(DecoderLayer, self).__init__()
    
            # 1. Masked Self-Attention
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
    
            # 2. Cross-Attention（Encoderの出力を参照）
            self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
    
            # 3. Feed-Forward Network
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    
            # Layer Normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
    
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
    
        def forward(self, x, memory, tgt_mask=None, memory_mask=None):
            """
            x: (batch, tgt_len, d_model) - ターゲット系列
            memory: (batch, src_len, d_model) - Encoderの出力
            tgt_mask: (batch, tgt_len, tgt_len) - 因果マスク
            memory_mask: (batch, 1, src_len) - パディングマスク
            """
            # 1. Masked Self-Attention + Residual + Norm
            self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            x = x + self.dropout1(self_attn_output)
            x = self.norm1(x)
    
            # 2. Cross-Attention + Residual + Norm
            # Query: Decoderの出力, Key/Value: Encoderの出力
            cross_attn_output, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
            x = x + self.dropout2(cross_attn_output)
            x = self.norm2(x)
    
            # 3. Feed-Forward + Residual + Norm
            ffn_output = self.ffn(x)
            x = x + self.dropout3(ffn_output)
            x = self.norm3(x)
    
            return x, self_attn_weights, cross_attn_weights
    
    
    # 動作確認
    print("=== DecoderLayerの動作確認 ===")
    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 32
    tgt_len = 15
    src_len = 20
    
    decoder_layer = DecoderLayer(d_model, nhead, d_ff)
    
    # ダミーデータ
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    
    # 因果マスク作成
    def create_causal_mask(seq_len):
        """下三角行列（未来をマスク）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    tgt_mask = create_causal_mask(tgt_len)
    
    # Decoder実行
    output, self_attn_weights, cross_attn_weights = decoder_layer(tgt, memory, tgt_mask)
    
    print(f"ターゲット入力: {tgt.shape}")
    print(f"Encoderメモリ: {memory.shape}")
    print(f"Decoder出力: {output.shape}")
    print(f"Self-Attention重み: {self_attn_weights.shape}")
    print(f"Cross-Attention重み: {cross_attn_weights.shape}")
    print("→ Cross-AttentionでEncoderの情報を参照")
    

### 因果マスクの可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    
    # 因果マスクの作成と可視化
    def create_and_visualize_causal_mask(seq_len=10):
        """因果マスクを作成して可視化"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
    
        print(f"=== 因果マスク（系列長={seq_len}） ===")
        print(mask.numpy())
        print("\n1 = 参照可能（過去・現在）")
        print("0 = 参照不可（未来）")
        print("\n例: 位置3は位置0,1,2,3のみ参照可能（位置4以降は見えない）")
    
        return mask
    
    # マスク作成
    causal_mask = create_and_visualize_causal_mask(seq_len=8)
    
    # Attentionスコアへの適用例
    print("\n=== マスク適用の効果 ===")
    scores = torch.randn(8, 8)  # ランダムなAttentionスコア
    
    print("マスク前のスコア（一部）:")
    print(scores[:4, :4].numpy())
    
    # マスク適用（未来を-infに）
    masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    
    print("\nマスク後のスコア（一部）:")
    print(masked_scores[:4, :4].numpy())
    
    # Softmax適用
    attn_weights = F.softmax(masked_scores, dim=-1)
    
    print("\nSoftmax後の重み（一部）:")
    print(attn_weights[:4, :4].numpy())
    print("→ 未来の位置（-inf）の重みは0になる")
    

### 完全なDecoderの実装
    
    
    import torch
    import torch.nn as nn
    import math
    
    class TransformerDecoder(nn.Module):
        """完全なTransformer Decoder"""
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                     d_ff=2048, dropout=0.1, max_len=5000):
            super(TransformerDecoder, self).__init__()
    
            self.d_model = d_model
    
            # 単語埋め込み
            self.embedding = nn.Embedding(vocab_size, d_model)
    
            # 位置エンコーディング
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
    
            # DecoderLayerをスタック
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ])
    
            # 出力層
            self.fc_out = nn.Linear(d_model, vocab_size)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            """
            tgt: (batch, tgt_len) - ターゲットトークンID
            memory: (batch, src_len, d_model) - Encoderの出力
            tgt_mask: (batch, tgt_len, tgt_len) - 因果マスク
            memory_mask: (batch, 1, src_len) - パディングマスク
            """
            # 埋め込み + スケーリング
            x = self.embedding(tgt) * math.sqrt(self.d_model)
    
            # 位置エンコーディング追加
            x = self.pos_encoding(x)
    
            # 各DecoderLayerを通過
            self_attn_weights_list = []
            cross_attn_weights_list = []
    
            for layer in self.layers:
                x, self_attn_weights, cross_attn_weights = layer(x, memory, tgt_mask, memory_mask)
                self_attn_weights_list.append(self_attn_weights)
                cross_attn_weights_list.append(cross_attn_weights)
    
            # 語彙への射影
            logits = self.fc_out(x)  # (batch, tgt_len, vocab_size)
    
            return logits, self_attn_weights_list, cross_attn_weights_list
    
    
    # 動作確認
    print("\n=== 完全なDecoderの動作確認 ===")
    vocab_size = 10000
    decoder = TransformerDecoder(vocab_size, d_model=512, nhead=8, num_layers=6)
    
    # ダミーデータ
    batch_size = 16
    tgt_len = 20
    src_len = 25
    
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    memory = torch.randn(batch_size, src_len, 512)
    
    # 因果マスク
    tgt_mask = create_causal_mask(tgt_len)
    
    # Decoder実行
    logits, self_attn_weights, cross_attn_weights = decoder(tgt, memory, tgt_mask)
    
    print(f"ターゲット入力: {tgt.shape}")
    print(f"Encoderメモリ: {memory.shape}")
    print(f"Decoder出力（ロジット）: {logits.shape}")
    print(f"→ 各位置で語彙全体に対する確率分布を出力")
    
    # パラメータ数
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

* * *

## 2.4 完全なTransformerモデル

### EncoderとDecoderの統合
    
    
    import torch
    import torch.nn as nn
    
    class Transformer(nn.Module):
        """完全なTransformerモデル（Encoder-Decoder）"""
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                     num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                     dropout=0.1, max_len=5000):
            super(Transformer, self).__init__()
    
            # Encoder
            self.encoder = TransformerEncoder(
                src_vocab_size, d_model, nhead, num_encoder_layers,
                d_ff, dropout, max_len
            )
    
            # Decoder
            self.decoder = TransformerDecoder(
                tgt_vocab_size, d_model, nhead, num_decoder_layers,
                d_ff, dropout, max_len
            )
    
            self.d_model = d_model
    
            # パラメータ初期化
            self._reset_parameters()
    
        def _reset_parameters(self):
            """Xavier初期化"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """
            src: (batch, src_len) - ソーストークン
            tgt: (batch, tgt_len) - ターゲットトークン
            src_mask: (batch, 1, src_len) - ソースのパディングマスク
            tgt_mask: (batch, tgt_len, tgt_len) - ターゲットの因果マスク
            """
            # Encoderでソースを処理
            memory, _ = self.encoder(src, src_mask)
    
            # Decoderでターゲットを生成
            output, _, _ = self.decoder(tgt, memory, tgt_mask, src_mask)
    
            return output
    
        def encode(self, src, src_mask=None):
            """Encoderのみ実行（推論時に使用）"""
            memory, _ = self.encoder(src, src_mask)
            return memory
    
        def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
            """Decoderのみ実行（推論時に使用）"""
            output, _, _ = self.decoder(tgt, memory, tgt_mask, memory_mask)
            return output
    
    
    # モデル作成
    print("=== 完全なTransformerモデル ===")
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    # 動作確認
    batch_size = 16
    src_len = 25
    tgt_len = 20
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # マスク作成
    src_mask = torch.ones(batch_size, 1, src_len)
    tgt_mask = create_causal_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"ソース入力: {src.shape}")
    print(f"ターゲット入力: {tgt.shape}")
    print(f"モデル出力: {output.shape}")
    print(f"→ 出力は (batch, tgt_len, tgt_vocab_size) の形状")
    
    # 総パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    

### 自己回帰生成の実装
    
    
    import torch
    import torch.nn.functional as F
    
    def generate_greedy(model, src, src_mask, max_len, start_token, end_token):
        """
        Greedy Decodingによる系列生成
    
        Args:
            model: Transformerモデル
            src: (batch, src_len) - ソース系列
            src_mask: (batch, 1, src_len) - ソースマスク
            max_len: 最大生成長
            start_token: 開始トークンID
            end_token: 終了トークンID
    
        Returns:
            generated: (batch, gen_len) - 生成系列
        """
        model.eval()
        batch_size = src.size(0)
        device = src.device
    
        # Encoderで一度だけ処理
        memory = model.encode(src, src_mask)
    
        # 生成系列の初期化（開始トークン）
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
        # 自己回帰的に生成
        for _ in range(max_len - 1):
            # 因果マスク作成
            tgt_len = generated.size(1)
            tgt_mask = create_causal_mask(tgt_len).to(device)
    
            # Decoder実行
            output = model.decode(generated, memory, tgt_mask, src_mask)
    
            # 最後の位置の予測を取得
            next_token_logits = output[:, -1, :]  # (batch, vocab_size)
    
            # Greedy選択（最も確率が高いトークン）
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
    
            # 生成系列に追加
            generated = torch.cat([generated, next_token], dim=1)
    
            # 全サンプルが終了トークンに到達したら終了
            if (next_token == end_token).all():
                break
    
        return generated
    
    
    def generate_beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5):
        """
        Beam Searchによる系列生成
    
        Args:
            model: Transformerモデル
            src: (1, src_len) - ソース系列（バッチサイズ1）
            src_mask: (1, 1, src_len)
            max_len: 最大生成長
            start_token: 開始トークンID
            end_token: 終了トークンID
            beam_size: ビームサイズ
    
        Returns:
            best_sequence: (1, gen_len) - 最良の生成系列
        """
        model.eval()
        device = src.device
    
        # Encoder
        memory = model.encode(src, src_mask)  # (1, src_len, d_model)
        memory = memory.repeat(beam_size, 1, 1)  # (beam_size, src_len, d_model)
    
        # ビーム初期化
        beams = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # 最初は1つのビームのみ有効
    
        finished_beams = []
    
        for step in range(max_len - 1):
            tgt_len = beams.size(1)
            tgt_mask = create_causal_mask(tgt_len).to(device)
    
            # Decoder
            output = model.decode(beams, memory, tgt_mask, src_mask.repeat(beam_size, 1, 1))
            next_token_logits = output[:, -1, :]  # (beam_size, vocab_size)
    
            # Log確率
            log_probs = F.log_softmax(next_token_logits, dim=-1)
    
            # ビームスコア更新
            vocab_size = log_probs.size(-1)
            scores = beam_scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)
            scores = scores.view(-1)  # (beam_size * vocab_size)
    
            # Top-k選択
            top_scores, top_indices = scores.topk(beam_size, largest=True)
    
            # 新しいビーム
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
    
            new_beams = []
            new_scores = []
    
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_scores)):
                # ビームを拡張
                new_beam = torch.cat([beams[beam_idx], token_idx.unsqueeze(0)])
    
                # 終了トークンに到達したら完成ビームに追加
                if token_idx == end_token:
                    finished_beams.append((new_beam, score.item()))
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
    
            # 完成ビームが十分あれば終了
            if len(finished_beams) >= beam_size:
                break
    
            # ビームが残っていない場合も終了
            if len(new_beams) == 0:
                break
    
            # ビームを更新
            beams = torch.stack(new_beams)
            beam_scores = torch.tensor(new_scores, device=device)
    
        # 最良のビームを選択
        if finished_beams:
            best_beam, best_score = max(finished_beams, key=lambda x: x[1])
        else:
            best_beam = beams[0]
    
        return best_beam.unsqueeze(0)
    
    
    # 動作確認
    print("\n=== 自己回帰生成のテスト ===")
    
    # ダミーモデルとデータ
    src_vocab_size = 100
    tgt_vocab_size = 100
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                       num_encoder_layers=2, num_decoder_layers=2)
    
    src = torch.randint(1, src_vocab_size, (1, 10))
    src_mask = torch.ones(1, 1, 10)
    
    start_token = 1
    end_token = 2
    max_len = 20
    
    # Greedy Decoding
    with torch.no_grad():
        generated_greedy = generate_greedy(model, src, src_mask, max_len, start_token, end_token)
    
    print(f"ソース系列: {src.shape}")
    print(f"Greedy生成: {generated_greedy.shape}")
    print(f"生成系列: {generated_greedy[0].tolist()}")
    
    # Beam Search
    with torch.no_grad():
        generated_beam = generate_beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5)
    
    print(f"\nBeam Search生成: {generated_beam.shape}")
    print(f"生成系列: {generated_beam[0].tolist()}")
    

* * *

## 2.5 実践：機械翻訳システム

### データセットの準備
    
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from collections import Counter
    import re
    
    class TranslationDataset(Dataset):
        """簡易的な翻訳データセット"""
        def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
            self.src_sentences = src_sentences
            self.tgt_sentences = tgt_sentences
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
    
        def __len__(self):
            return len(self.src_sentences)
    
        def __getitem__(self, idx):
            src = self.src_sentences[idx]
            tgt = self.tgt_sentences[idx]
    
            # トークンIDに変換
            src_ids = [self.src_vocab.get(w, self.src_vocab['']) for w in src.split()]
            tgt_ids = [self.tgt_vocab.get(w, self.tgt_vocab['']) for w in tgt.split()]
    
            return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
    
    def build_vocab(sentences, max_vocab_size=10000):
        """語彙を構築"""
        words = []
        for sent in sentences:
            words.extend(sent.split())
    
        # 頻度カウント
        word_counts = Counter(words)
        most_common = word_counts.most_common(max_vocab_size - 4)  # 特殊トークン分を除く
    
        # 語彙辞書作成
        vocab = {'': 0, '': 1, '': 2, '': 3}
        for word, _ in most_common:
            vocab[word] = len(vocab)
    
        return vocab
    
    
    # ダミーデータ（実際はMulti30kやWMTなどを使用）
    src_sentences = [
        "i love machine learning",
        "transformers are powerful",
        "attention is all you need",
        "deep learning is amazing",
        "natural language processing"
    ]
    
    tgt_sentences = [
        "私 は 機械 学習 が 好き です",
        "トランスフォーマー は 強力 です",
        "アテンション が 全て です",
        "深層 学習 は 素晴らしい です",
        "自然 言語 処理"
    ]
    
    # 語彙構築
    src_vocab = build_vocab(src_sentences)
    tgt_vocab = build_vocab(tgt_sentences)
    
    print("=== 翻訳データセットの準備 ===")
    print(f"ソース語彙サイズ: {len(src_vocab)}")
    print(f"ターゲット語彙サイズ: {len(tgt_vocab)}")
    print(f"\nソース語彙（一部）: {list(src_vocab.items())[:10]}")
    print(f"ターゲット語彙（一部）: {list(tgt_vocab.items())[:10]}")
    
    # データセット作成
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    
    # サンプル確認
    src_sample, tgt_sample = dataset[0]
    print(f"\nサンプル 0:")
    print(f"ソース: {src_sentences[0]}")
    print(f"ソースID: {src_sample.tolist()}")
    print(f"ターゲット: {tgt_sentences[0]}")
    print(f"ターゲットID: {tgt_sample.tolist()}")
    

### 訓練ループの実装
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    
    def collate_fn(batch, src_vocab, tgt_vocab):
        """バッチのコレート関数"""
        src_batch, tgt_batch = zip(*batch)
    
        # パディング
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab[''])
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab[''])
    
        return src_padded, tgt_padded
    
    
    def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
        """マスクを作成"""
        # ソースのパディングマスク
        src_mask = (src != src_pad_idx).unsqueeze(1)  # (batch, 1, src_len)
    
        # ターゲットの因果マスク + パディングマスク
        tgt_len = tgt.size(1)
        tgt_mask = create_causal_mask(tgt_len).to(tgt.device)  # (1, tgt_len, tgt_len)
        tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        tgt_mask = tgt_mask & tgt_pad_mask
    
        return src_mask, tgt_mask
    
    
    def train_epoch(model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, device):
        """1エポックの訓練"""
        model.train()
        total_loss = 0
    
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
    
            # ターゲットを入力と教師データに分割
            tgt_input = tgt[:, :-1]  # を除く
            tgt_output = tgt[:, 1:]  # を除く
    
            # マスク作成
            src_mask, tgt_mask = create_masks(src, tgt_input,
                                             src_vocab[''], tgt_vocab[''])
    
            # Forward
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
    
            # Loss計算（パディングを無視）
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
    
            loss = criterion(output, tgt_output)
    
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            total_loss += loss.item()
    
        return total_loss / len(dataloader)
    
    
    # 訓練設定
    print("\n=== 翻訳モデルの訓練 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")
    
    # モデル作成
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        dropout=0.1
    ).to(device)
    
    # DataLoader
    from functools import partial
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(collate_fn, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    )
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[''])
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 訓練
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, src_vocab, tgt_vocab, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
    
    print("\n訓練完了！")
    

### 翻訳推論
    
    
    import torch
    
    def translate(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
        """文を翻訳"""
        model.eval()
    
        # ソース文をトークンIDに変換
        src_tokens = src_sentence.split()
        src_ids = [src_vocab.get(w, src_vocab['']) for w in src_tokens]
        src = torch.tensor(src_ids).unsqueeze(0).to(device)  # (1, src_len)
    
        # ソースマスク
        src_mask = torch.ones(1, 1, src.size(1)).to(device)
    
        # Greedy Decodingで生成
        with torch.no_grad():
            generated = generate_greedy(
                model, src, src_mask, max_len,
                start_token=tgt_vocab[''],
                end_token=tgt_vocab['']
            )
    
        # トークンIDを単語に変換
        idx_to_word = {v: k for k, v in tgt_vocab.items()}
        translated = [idx_to_word.get(idx.item(), '') for idx in generated[0]]
    
        # とを除去
        if translated[0] == '':
            translated = translated[1:]
        if '' in translated:
            eos_idx = translated.index('')
            translated = translated[:eos_idx]
    
        return ' '.join(translated)
    
    
    # 翻訳テスト
    print("\n=== 翻訳テスト ===")
    
    test_sentences = [
        "i love machine learning",
        "transformers are powerful",
        "attention is all you need"
    ]
    
    for src_sent in test_sentences:
        translated = translate(model, src_sent, src_vocab, tgt_vocab, device)
        print(f"ソース: {src_sent}")
        print(f"翻訳: {translated}")
        print()
    
    print("→ 小規模データのため完璧ではないが、基本的な翻訳機能を実装")
    

* * *

## 2.6 Transformerの学習テクニック

### 学習率のウォームアップ

Transformerの訓練では、**ウォームアップスケジューラ** が重要です。初期は学習率を小さく保ち、徐々に増やしてから減衰させます。

$$ \text{lr}(step) = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot \text{warmup_steps}^{-1.5}) $$ 
    
    
    import torch.optim as optim
    
    class NoamOpt:
        """Noam学習率スケジューラ（論文実装）"""
        def __init__(self, d_model, warmup_steps, optimizer):
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            self.optimizer = optimizer
            self._step = 0
            self._rate = 0
    
        def step(self):
            """1ステップ更新"""
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()
    
        def rate(self, step=None):
            """現在の学習率を計算"""
            if step is None:
                step = self._step
            return (self.d_model ** (-0.5)) * min(step ** (-0.5),
                                                   step * self.warmup_steps ** (-1.5))
    
    # 使用例
    print("=== Noam学習率スケジューラ ===")
    d_model = 512
    warmup_steps = 4000
    
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(d_model, warmup_steps, optimizer)
    
    # 学習率の推移を可視化
    steps = list(range(1, 20000))
    lrs = [scheduler.rate(step) for step in steps]
    
    print(f"初期学習率（step=1）: {lrs[0]:.6f}")
    print(f"ピーク学習率（step={warmup_steps}）: {lrs[warmup_steps-1]:.6f}")
    print(f"後期学習率（step=20000）: {lrs[-1]:.6f}")
    print("→ ウォームアップで徐々に増加、その後減衰")
    

### Label Smoothing

**Label Smoothing** は、正解ラベルの確率を1ではなく0.9程度にし、他のクラスに少し確率を分散させる正則化手法です。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LabelSmoothingLoss(nn.Module):
        """Label Smoothing付きCross Entropy Loss"""
        def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
            super(LabelSmoothingLoss, self).__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.ignore_index = ignore_index
            self.confidence = 1.0 - smoothing
    
        def forward(self, pred, target):
            """
            pred: (batch * seq_len, num_classes) - ロジット
            target: (batch * seq_len) - 正解ラベル
            """
            # Log-softmax
            log_probs = F.log_softmax(pred, dim=-1)
    
            # 正解位置の確率を取得
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    
            # 全クラスの平均log確率
            smooth_loss = -log_probs.mean(dim=-1)
    
            # 組み合わせ
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            # ignore_indexをマスク
            if self.ignore_index >= 0:
                mask = (target != self.ignore_index).float()
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
    
            return loss
    
    
    # 比較
    print("\n=== Label Smoothingの効果 ===")
    
    num_classes = 10
    criterion_normal = nn.CrossEntropyLoss()
    criterion_smooth = LabelSmoothingLoss(num_classes, smoothing=0.1)
    
    # ダミーデータ
    pred = torch.randn(32, num_classes)
    target = torch.randint(0, num_classes, (32,))
    
    loss_normal = criterion_normal(pred, target)
    loss_smooth = criterion_smooth(pred, target)
    
    print(f"通常のCross Entropy Loss: {loss_normal.item():.4f}")
    print(f"Label Smoothing Loss: {loss_smooth.item():.4f}")
    print("→ Label Smoothingは過信を防ぎ、汎化性能を向上")
    

### Mixed Precision Training
    
    
    import torch
    from torch.cuda.amp import autocast, GradScaler
    
    # Mixed Precision Training（GPU利用時）
    if torch.cuda.is_available():
        print("\n=== Mixed Precision Trainingの例 ===")
    
        device = torch.device('cuda')
        model = model.to(device)
    
        scaler = GradScaler()
    
        # 訓練ループの一部
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
    
            optimizer.zero_grad()
    
            # Mixed Precisionで計算
            with autocast():
                output = model(src, tgt_input)
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                loss = criterion(output, tgt_output)
    
            # スケールされた勾配で逆伝播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        print("→ FP16計算で高速化＆メモリ削減（最大2倍高速）")
    else:
        print("\n=== Mixed Precision Training ===")
        print("GPUが利用できないため、スキップします")
    

* * *

## 演習問題

**演習1：Multi-Head Attentionのヘッド数の影響**

異なるヘッド数（1, 2, 4, 8, 16）でモデルを訓練し、性能と計算コストを比較してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: ヘッド数を変えて複数のモデルを作成
    # TODO: 同じデータで訓練し、性能・訓練時間・パラメータ数を比較
    # TODO: Attention可視化でヘッドごとの役割分担を分析
    # ヒント: ヘッド数が多いほど性能向上するが、計算コストも増加
    

**演習2：Positional Encodingの実験**

Sin/Cos位置エンコーディングと学習可能な位置埋め込みを比較してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 2種類の位置エンコーディングを実装
    # 1. Sin/Cos（固定）
    # 2. nn.Embedding（学習可能）
    
    # TODO: 同じタスクで性能比較
    # TODO: 系列長の汎化性能を評価（訓練より長い系列でテスト）
    # 期待: Sin/Cosは任意長に汎化可能
    

**演習3：因果マスクの可視化**

Decoderの因果マスクがAttention重みにどう影響するか可視化してください。
    
    
    import torch
    import matplotlib.pyplot as plt
    
    # TODO: DecoderLayerのSelf-Attention重みを取得
    # TODO: マスク有り・無しでAttention重みを可視化
    # TODO: ヒートマップで未来のトークンが見えないことを確認
    

**演習4：Beam Searchのビームサイズ最適化**

異なるビームサイズ（1, 3, 5, 10, 20）で翻訳品質と速度を比較してください。
    
    
    import torch
    import time
    
    # TODO: ビームサイズを変えて翻訳を実行
    # TODO: BLEU スコア、生成時間を計測
    # TODO: ビームサイズ vs 品質・速度のグラフを作成
    # 期待: ビームサイズ5-10で品質と速度のバランスが良い
    

**演習5：Layer数の影響を調査**

Encoder/Decoderの層数（1, 2, 4, 6, 12）を変えて性能を比較してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 異なる層数でモデルを作成
    # TODO: 訓練Loss、検証Loss、パラメータ数、訓練時間を記録
    # TODO: 層数 vs 性能のグラフを作成
    # 分析: 深すぎると過学習・訓練時間増、浅すぎると表現力不足
    

* * *

## まとめ

この章では、Transformerの完全なアーキテクチャを学びました。

### 重要ポイント

  * **Transformerの構造** ：Encoder-Decoderアーキテクチャ、6層スタック
  * **Encoder** ：Multi-Head Self-Attention + FFN、並列処理可能
  * **Decoder** ：Masked Self-Attention + Cross-Attention + FFN、自己回帰生成
  * **Multi-Head Attention** ：複数の観点から依存関係を捉える
  * **Positional Encoding** ：Sin/Cos関数で位置情報を注入
  * **Residual + Layer Norm** ：深い層でも学習を安定化
  * **因果マスク** ：未来のトークンを見ないよう制御
  * **自己回帰生成** ：Greedy Decoding、Beam Search
  * **学習テクニック** ：ウォームアップ、Label Smoothing、Mixed Precision
  * **実践** ：機械翻訳システムの完全実装

### 次のステップ

次章では、**Transformerの学習と最適化** について学びます。効率的な訓練手法、データ拡張、評価指標、ハイパーパラメータチューニングなど、実用的なテクニックを習得します。
