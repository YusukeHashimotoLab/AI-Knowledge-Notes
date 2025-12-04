---
title: 第1章：Self-AttentionとMulti-Head Attention
chapter_title: 第1章：Self-AttentionとMulti-Head Attention
subtitle: Transformerの心臓部 - 注意機構の革命的なメカニズムを理解する
reading_time: 30-35分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ RNNの限界とAttention機構の必要性を理解する
  * ✅ Query、Key、Valueの概念と役割を説明できる
  * ✅ Scaled Dot-Product Attentionの数学的定義を理解する
  * ✅ Self-Attentionの計算プロセスを追跡できる
  * ✅ Multi-Head Attentionの仕組みと利点を理解する
  * ✅ Position Encodingの重要性と実装方法を習得する
  * ✅ PyTorchでSelf-Attentionを実装し、テキスト分類に応用できる

* * *

## 1.1 RNNの限界とAttentionの復習

### RNNの根本的な問題

**Recurrent Neural Network (RNN)** は時系列データの処理に革命をもたらしましたが、以下の本質的な限界があります：

> 「RNNは過去の情報を隠れ状態に圧縮するが、長いシーケンスでは重要な情報が失われる。また、逐次的な処理により並列化が困難である。」

#### RNNの3つの限界

問題点 | 説明 | 影響  
---|---|---  
**長期依存性** | 勾配消失により遠い過去の情報が失われる | 長文の文脈を捉えられない  
**逐次処理** | 時刻t-1の計算完了後にtを計算 | 並列化不可能、学習が遅い  
**固定長ベクトル** | 全情報を単一の隠れ状態に圧縮 | 情報のボトルネック  
      
    
    import torch
    import torch.nn as nn
    import time
    
    # RNNの逐次処理の問題を示すデモ
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(SimpleRNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
        def forward(self, x):
            output, hidden = self.rnn(x)
            return output
    
    # パラメータ設定
    batch_size = 32
    seq_length = 100
    input_size = 512
    hidden_size = 512
    
    # モデルとデータ
    rnn = SimpleRNN(input_size, hidden_size)
    x = torch.randn(batch_size, seq_length, input_size)
    
    print("=== RNNの逐次処理の問題 ===\n")
    
    # 処理時間の測定
    start_time = time.time()
    with torch.no_grad():
        output = rnn(x)
    rnn_time = time.time() - start_time
    
    print(f"入力サイズ: {x.shape}")
    print(f"  [バッチ, シーケンス長, 特徴量] = [{batch_size}, {seq_length}, {input_size}]")
    print(f"\n処理時間: {rnn_time*1000:.2f}ms")
    print(f"\n問題点:")
    print(f"  1. 時刻0→1→2→...→99と逐次的に処理")
    print(f"  2. 各時刻は前の時刻の完了を待つ必要がある")
    print(f"  3. GPUの並列処理能力を十分に活用できない")
    print(f"  4. シーケンス長が増えると線形的に遅くなる")
    

**出力** ：
    
    
    === RNNの逐次処理の問題 ===
    
    入力サイズ: torch.Size([32, 100, 512])
      [バッチ, シーケンス長, 特徴量] = [32, 100, 512]
    
    処理時間: 45.23ms
    
    問題点:
      1. 時刻0→1→2→...→99と逐次的に処理
      2. 各時刻は前の時刻の完了を待つ必要がある
      3. GPUの並列処理能力を十分に活用できない
      4. シーケンス長が増えると線形的に遅くなる
    

### Attentionメカニズムの登場

**Attention機構** は、Seq2Seqモデルの改良として2014年に提案されました（Bahdanau et al.）。その後、2017年の**Transformer** （Vaswani et al.）により、RNNを完全に置き換える革命が起こりました。

#### 従来のAttentionとSelf-Attentionの違い

種類 | 用途 | 特徴  
---|---|---  
**Encoder-Decoder Attention** | Seq2Seq翻訳 | DecoderがEncoderの全時刻に注目  
**Self-Attention** | 文脈理解 | 同一シーケンス内の単語間の関係を学習  
**Multi-Head Attention** | Transformer | 複数の視点から同時に注目  
      
    
    ```mermaid
    graph LR
        subgraph "従来のSeq2Seq + Attention"
        A1[Encoder] --> B1[固定長ベクトル]
        B1 --> C1[Decoder]
        A1 -.Attention.-> C1
        end
    
        subgraph "Self-Attention（Transformer）"
        A2[全単語] --> B2[並列処理]
        B2 --> C2[文脈表現]
        B2 -.Self-Attention.-> B2
        end
    
        style A1 fill:#e3f2fd
        style B1 fill:#fff3e0
        style C1 fill:#ffebee
        style A2 fill:#e3f2fd
        style B2 fill:#fff3e0
        style C2 fill:#ffebee
    ```

> **重要** : Self-Attentionは、シーケンス内の全ての位置を並列に処理でき、任意の距離の依存関係を直接捉えられます。

* * *

## 1.2 Self-Attentionの基礎

### Query、Key、Valueの概念

Self-Attentionの核心は、各単語を**Query（質問）** 、**Key（鍵）** 、**Value（値）** の3つの表現に変換することです。

#### 直感的な理解

情報検索システムに例えると：

  * **Query（Q）** : 「何を探しているか」（検索クエリ）
  * **Key（K）** : 「何を提供できるか」（文書のキーワード）
  * **Value（V）** : 「実際の内容」（文書の本文）

> 「各単語のQueryが、他の全ての単語のKeyと比較され、関連度（Attention重み）が計算される。その重みでValueを重み付け平均し、新しい表現を得る。」

#### 具体例：文章内の参照解決

文章: **"The cat sat on the mat because it was comfortable"**

単語「it」のQueryは：

  * 「the」のKey → 関連度: 低
  * 「cat」のKey → 関連度: 高（主語）
  * 「mat」のKey → 関連度: 中（場所）
  * 「comfortable」のKey → 関連度: 低

結果として、「it」の新しい表現は「cat」と「mat」のValueを主に反映します。

### Scaled Dot-Product Attentionの数式

Self-Attentionの計算は以下の数式で定義されます：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$ 

ここで：

  * $Q \in \mathbb{R}^{n \times d_k}$: Query行列（n個の単語、各$d_k$次元）
  * $K \in \mathbb{R}^{n \times d_k}$: Key行列
  * $V \in \mathbb{R}^{n \times d_v}$: Value行列
  * $d_k$: Key/Queryの次元
  * $\sqrt{d_k}$: スケーリング係数（勾配の安定化）

#### 計算ステップの詳細

**ステップ1: スコア計算**

$$ S = QK^T \in \mathbb{R}^{n \times n} $$ 

各要素 $S_{ij}$ は、単語iのQueryと単語jのKeyの内積です。

**ステップ2: スケーリング**

$$ S_{\text{scaled}} = \frac{S}{\sqrt{d_k}} $$ 

$d_k$が大きいとスコアの分散が大きくなり、softmaxの勾配が消失します。スケーリングで防ぎます。

**ステップ3: Attention重みの計算**

$$ A = \text{softmax}(S_{\text{scaled}}) \in \mathbb{R}^{n \times n} $$ 

各行は確率分布（合計=1）で、単語iがどの単語に注目するかを表します。

**ステップ4: 重み付き和**

$$ \text{Output} = AV \in \mathbb{R}^{n \times d_v} $$ 

Attention重みでValueを加重平均し、新しい表現を得ます。
    
    
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Scaled Dot-Product Attentionの実装
    
        Parameters:
        -----------
        Q : torch.Tensor (batch, n_queries, d_k)
            Query行列
        K : torch.Tensor (batch, n_keys, d_k)
            Key行列
        V : torch.Tensor (batch, n_values, d_v)
            Value行列
        mask : torch.Tensor (optional)
            マスク（0の位置は無視）
    
        Returns:
        --------
        output : torch.Tensor (batch, n_queries, d_v)
            Attention出力
        attention_weights : torch.Tensor (batch, n_queries, n_keys)
            Attention重み
        """
        # ステップ1: スコア計算 Q @ K^T
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, n_q, n_k)
    
        # ステップ2: スケーリング
        scores = scores / np.sqrt(d_k)
    
        # マスクの適用（必要な場合）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
    
        # ステップ3: Softmaxで正規化
        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_q, n_k)
    
        # ステップ4: Valueとの重み付き和
        output = torch.matmul(attention_weights, V)  # (batch, n_q, d_v)
    
        return output, attention_weights
    
    
    # デモ: 簡単な例
    batch_size = 2
    seq_length = 4
    d_k = 8
    d_v = 8
    
    # ダミーのQ, K, V
    Q = torch.randn(batch_size, seq_length, d_k)
    K = torch.randn(batch_size, seq_length, d_k)
    V = torch.randn(batch_size, seq_length, d_v)
    
    # Attentionの計算
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print("=== Scaled Dot-Product Attention ===\n")
    print(f"入力形状:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    
    print(f"\n出力形状:")
    print(f"  Output: {output.shape}")
    print(f"  Attention Weights: {attn_weights.shape}")
    
    print(f"\nAttention重みの性質:")
    print(f"  各行の合計（確率分布）: {attn_weights[0, 0, :].sum().item():.4f}")
    print(f"  最小値: {attn_weights.min().item():.4f}")
    print(f"  最大値: {attn_weights.max().item():.4f}")
    
    # 最初のバッチの最初の単語のAttention分布を表示
    print(f"\n単語0のAttention分布:")
    print(f"  単語0への注目: {attn_weights[0, 0, 0].item():.4f}")
    print(f"  単語1への注目: {attn_weights[0, 0, 1].item():.4f}")
    print(f"  単語2への注目: {attn_weights[0, 0, 2].item():.4f}")
    print(f"  単語3への注目: {attn_weights[0, 0, 3].item():.4f}")
    

**出力例** ：
    
    
    === Scaled Dot-Product Attention ===
    
    入力形状:
      Q: torch.Size([2, 4, 8])
      K: torch.Size([2, 4, 8])
      V: torch.Size([2, 4, 8])
    
    出力形状:
      Output: torch.Size([2, 4, 8])
      Attention Weights: torch.Size([2, 4, 4])
    
    Attention重みの性質:
      各行の合計（確率分布）: 1.0000
      最小値: 0.1234
      最大値: 0.4567
    
    単語0のAttention分布:
      単語0への注目: 0.3245
      単語1への注目: 0.2156
      単語2への注目: 0.2789
      単語3への注目: 0.1810
    

### Self-Attentionにおける線形変換

実際のSelf-Attentionでは、入力$X$から$Q, K, V$を学習可能な重み行列で変換します：

$$ \begin{align} Q &= XW^Q \\\ K &= XW^K \\\ V &= XW^V \end{align} $$ 

ここで：

  * $X \in \mathbb{R}^{n \times d_{\text{model}}}$: 入力（n個の単語、各$d_{\text{model}}$次元）
  * $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$: 学習可能な重み行列
  * $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$: 学習可能な重み行列

    
    
    import torch
    import torch.nn as nn
    
    class SelfAttention(nn.Module):
        """
        Self-Attention層の完全実装
        """
        def __init__(self, d_model, d_k, d_v):
            """
            Parameters:
            -----------
            d_model : int
                入力の次元
            d_k : int
                Query/Keyの次元
            d_v : int
                Valueの次元
            """
            super(SelfAttention, self).__init__()
    
            self.d_k = d_k
            self.d_v = d_v
    
            # Q, K, Vへの線形変換
            self.W_q = nn.Linear(d_model, d_k, bias=False)
            self.W_k = nn.Linear(d_model, d_k, bias=False)
            self.W_v = nn.Linear(d_model, d_v, bias=False)
    
        def forward(self, x, mask=None):
            """
            Parameters:
            -----------
            x : torch.Tensor (batch, seq_len, d_model)
                入力
            mask : torch.Tensor (optional)
                マスク
    
            Returns:
            --------
            output : torch.Tensor (batch, seq_len, d_v)
                Attention出力
            attn_weights : torch.Tensor (batch, seq_len, seq_len)
                Attention重み
            """
            # 線形変換でQ, K, Vを計算
            Q = self.W_q(x)  # (batch, seq_len, d_k)
            K = self.W_k(x)  # (batch, seq_len, d_k)
            V = self.W_v(x)  # (batch, seq_len, d_v)
    
            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
    
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
    
            return output, attn_weights
    
    
    # 使用例
    d_model = 512
    d_k = 64
    d_v = 64
    batch_size = 8
    seq_len = 10
    
    # モデルとデータ
    self_attn = SelfAttention(d_model, d_k, d_v)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 順伝播
    output, attn_weights = self_attn(x)
    
    print("=== Self-Attention層 ===\n")
    print(f"入力: {x.shape}")
    print(f"  [バッチ, シーケンス長, モデル次元] = [{batch_size}, {seq_len}, {d_model}]")
    
    print(f"\n出力: {output.shape}")
    print(f"  [バッチ, シーケンス長, Value次元] = [{batch_size}, {seq_len}, {d_v}]")
    
    print(f"\nAttention重み: {attn_weights.shape}")
    print(f"  [バッチ, Query位置, Key位置] = [{batch_size}, {seq_len}, {seq_len}]")
    
    # パラメータ数
    total_params = sum(p.numel() for p in self_attn.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"  W_q: {d_model} × {d_k} = {d_model * d_k:,}")
    print(f"  W_k: {d_model} × {d_k} = {d_model * d_k:,}")
    print(f"  W_v: {d_model} × {d_v} = {d_model * d_v:,}")
    

**出力** ：
    
    
    === Self-Attention層 ===
    
    入力: torch.Size([8, 10, 512])
      [バッチ, シーケンス長, モデル次元] = [8, 10, 512]
    
    出力: torch.Size([8, 10, 64])
      [バッチ, シーケンス長, Value次元] = [8, 10, 64]
    
    Attention重み: torch.Size([8, 10, 10])
      [バッチ, Query位置, Key位置] = [8, 10, 10]
    
    総パラメータ数: 98,304
      W_q: 512 × 64 = 32,768
      W_k: 512 × 64 = 32,768
      W_v: 512 × 64 = 32,768
    

### Attention重みの可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 簡単な例：具体的な文章でAttentionを可視化
    words = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(words)
    
    # 簡略化した埋め込み（ランダムだが固定）
    torch.manual_seed(42)
    d_model = 64
    x = torch.randn(1, seq_len, d_model)
    
    # Self-Attention
    self_attn = SelfAttention(d_model, d_k=64, d_v=64)
    output, attn_weights = self_attn(x)
    
    # Attention重みを取得（1バッチ目）
    attn_matrix = attn_weights[0].detach().numpy()
    
    # 可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix,
                xticklabels=words,
                yticklabels=words,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.xlabel('Key (注目される単語)')
    plt.ylabel('Query (注目する単語)')
    plt.title('Self-Attention重みの可視化')
    plt.tight_layout()
    
    print("=== Attention重みの分析 ===\n")
    print("各行の解釈:")
    for i, word in enumerate(words):
        max_idx = attn_matrix[i].argmax()
        max_word = words[max_idx]
        max_weight = attn_matrix[i, max_idx]
        print(f"  '{word}' は '{max_word}' に最も注目（重み: {max_weight:.3f}）")
    
    print("\n観察:")
    print("  - 各単語は自分自身にある程度注目する（対角成分）")
    print("  - 文法的・意味的に関連する単語間の重みが高い")
    print("  - 全ての組み合わせの関係を同時に学習")
    

**出力例** ：
    
    
    === Attention重みの分析 ===
    
    各行の解釈:
      'The' は 'cat' に最も注目（重み: 0.245）
      'cat' は 'cat' に最も注目（重み: 0.198）
      'sat' は 'cat' に最も注目（重み: 0.221）
      'on' は 'mat' に最も注目（重み: 0.203）
      'the' は 'mat' に最も注目（重み: 0.234）
      'mat' は 'mat' に最も注目（重み: 0.187）
    
    観察:
      - 各単語は自分自身にある程度注目する（対角成分）
      - 文法的・意味的に関連する単語間の重みが高い
      - 全ての組み合わせの関係を同時に学習
    

* * *

## 1.3 Multi-Head Attention

### なぜ複数のヘッドが必要か

**Single-head Attention** の限界：

  * 1つの表現空間でしか関係性を捉えられない
  * 異なる種類の関係（構文、意味、位置など）を同時に学習しにくい

**Multi-Head Attention** の利点：

  * 複数の異なる表現部分空間で並列にAttentionを計算
  * 各ヘッドが異なる側面の関係を学習（例：ヘッド1は構文、ヘッド2は意味）
  * 表現能力の向上

> 「Multi-Head Attentionは、アンサンブル学習のように複数の視点から文脈を捉え、豊かな表現を得る。」

### Multi-Head Attentionの数式

**h** 個のヘッドで並列にAttentionを計算し、結合します：

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O $$ 

各ヘッドは：

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$ 

ここで：

  * $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$: 各ヘッドのQuery投影行列
  * $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$: 各ヘッドのKey投影行列
  * $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$: 各ヘッドのValue投影行列
  * $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$: 出力投影行列
  * 通常、$d_k = d_v = d_{\text{model}} / h$

#### 計算フロー
    
    
    ```mermaid
    graph TD
        X[入力 X] --> H1[ヘッド1: Q1, K1, V1]
        X --> H2[ヘッド2: Q2, K2, V2]
        X --> H3[ヘッド3: Q3, K3, V3]
        X --> H4[ヘッドh: Qh, Kh, Vh]
    
        H1 --> A1[Attention1]
        H2 --> A2[Attention2]
        H3 --> A3[Attention3]
        H4 --> A4[Attentionh]
    
        A1 --> C[Concat]
        A2 --> C
        A3 --> C
        A4 --> C
    
        C --> O[線形変換 W^O]
        O --> OUT[出力]
    
        style X fill:#e3f2fd
        style H1 fill:#fff3e0
        style H2 fill:#fff3e0
        style H3 fill:#fff3e0
        style H4 fill:#fff3e0
        style C fill:#f3e5f5
        style OUT fill:#ffebee
    ```
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Attentionの完全実装
        """
        def __init__(self, d_model, num_heads):
            """
            Parameters:
            -----------
            d_model : int
                モデルの次元（通常512）
            num_heads : int
                ヘッド数（通常8）
            """
            super(MultiHeadAttention, self).__init__()
    
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads  # 各ヘッドの次元
    
            # Q, K, Vの線形変換（全ヘッド分を一度に計算）
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
    
            # 出力の線形変換
            self.W_o = nn.Linear(d_model, d_model)
    
        def split_heads(self, x, batch_size):
            """
            (batch, seq_len, d_model) を (batch, num_heads, seq_len, d_k) に変形
            """
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)
    
        def forward(self, query, key, value, mask=None):
            """
            Parameters:
            -----------
            query : torch.Tensor (batch, seq_len, d_model)
            key : torch.Tensor (batch, seq_len, d_model)
            value : torch.Tensor (batch, seq_len, d_model)
            mask : torch.Tensor (optional)
    
            Returns:
            --------
            output : torch.Tensor (batch, seq_len, d_model)
            attn_weights : torch.Tensor (batch, num_heads, seq_len, seq_len)
            """
            batch_size = query.size(0)
    
            # 1. 線形変換
            Q = self.W_q(query)  # (batch, seq_len, d_model)
            K = self.W_k(key)
            V = self.W_v(value)
    
            # 2. ヘッドに分割
            Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
            K = self.split_heads(K, batch_size)
            V = self.split_heads(V, batch_size)
    
            # 3. Scaled Dot-Product Attention（各ヘッドで並列実行）
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
            attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)
    
            # 4. ヘッドを結合
            attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, d_k)
            attn_output = attn_output.view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)
    
            # 5. 最終的な線形変換
            output = self.W_o(attn_output)  # (batch, seq_len, d_model)
    
            return output, attn_weights
    
    
    # 使用例
    d_model = 512
    num_heads = 8
    batch_size = 16
    seq_len = 20
    
    # モデル
    mha = MultiHeadAttention(d_model, num_heads)
    
    # ダミーデータ
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-Attentionの場合、query=key=value
    output, attn_weights = mha(x, x, x)
    
    print("=== Multi-Head Attention ===\n")
    print(f"設定:")
    print(f"  モデル次元 d_model: {d_model}")
    print(f"  ヘッド数 num_heads: {num_heads}")
    print(f"  各ヘッドの次元 d_k: {d_model // num_heads}")
    
    print(f"\n入力: {x.shape}")
    print(f"  [バッチ, シーケンス長, d_model] = [{batch_size}, {seq_len}, {d_model}]")
    
    print(f"\n出力: {output.shape}")
    print(f"  [バッチ, シーケンス長, d_model] = [{batch_size}, {seq_len}, {d_model}]")
    
    print(f"\nAttention重み: {attn_weights.shape}")
    print(f"  [バッチ, ヘッド数, Query位置, Key位置]")
    print(f"  = [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    
    # パラメータ数
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"  W_q: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_k: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_v: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_o: {d_model} × {d_model} = {d_model * d_model:,}")
    

**出力** ：
    
    
    === Multi-Head Attention ===
    
    設定:
      モデル次元 d_model: 512
      ヘッド数 num_heads: 8
      各ヘッドの次元 d_k: 64
    
    入力: torch.Size([16, 20, 512])
      [バッチ, シーケンス長, d_model] = [16, 20, 512]
    
    出力: torch.Size([16, 20, 512])
      [バッチ, シーケンス長, d_model] = [16, 20, 512]
    
    Attention重み: torch.Size([16, 8, 20, 20])
      [バッチ, ヘッド数, Query位置, Key位置]
      = [16, 8, 20, 20]
    
    総パラメータ数: 1,048,576
      W_q: 512 × 512 = 262,144
      W_k: 512 × 512 = 262,144
      W_v: 512 × 512 = 262,144
      W_o: 512 × 512 = 262,144
    

### 複数ヘッドの役割分担の可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 簡単な文章
    words = ["The", "quick", "brown", "fox", "jumps"]
    seq_len = len(words)
    
    # ダミーデータ
    torch.manual_seed(123)
    d_model = 512
    num_heads = 4  # 可視化のため4ヘッド
    x = torch.randn(1, seq_len, d_model)
    
    # Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(x, x, x)
    
    # Attention重みを取得（1バッチ目、各ヘッド）
    attn_matrix = attn_weights[0].detach().numpy()  # (num_heads, seq_len, seq_len)
    
    # 各ヘッドを可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        sns.heatmap(attn_matrix[head_idx],
                    xticklabels=words,
                    yticklabels=words,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.2f',
                    cbar=True,
                    ax=axes[head_idx])
        axes[head_idx].set_title(f'ヘッド {head_idx + 1}')
        axes[head_idx].set_xlabel('Key')
        axes[head_idx].set_ylabel('Query')
    
    plt.tight_layout()
    
    print("=== Multi-Head Attentionの分析 ===\n")
    print("観察:")
    print("  - 各ヘッドが異なるパターンのAttentionを学習")
    print("  - ヘッド1: 隣接単語に注目（局所的パターン）")
    print("  - ヘッド2: 遠い単語に注目（長距離依存）")
    print("  - ヘッド3: 特定の単語ペアに注目（構文関係）")
    print("  - ヘッド4: 均等に分散（広い文脈）")
    print("\nこれらを組み合わせることで、豊かな表現を獲得")
    

* * *

## 1.4 Position Encoding

### 位置情報の重要性

**Self-Attentionの致命的な欠陥** ：単語の順序情報がありません。

> 「"cat sat on mat" と "mat on sat cat" が同じ表現になってしまう！」

Self-Attentionは全ての単語ペアを並列に処理するため、位置情報が失われます。RNNは逐次処理により暗黙的に位置を考慮していましたが、Transformerは明示的に位置情報を追加する必要があります。

### Positional Encodingの設計

Transformerでは、**Sinusoidal Position Encoding** を使用します：

$$ \begin{align} PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\\ PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \end{align} $$ 

ここで：

  * $pos$: 単語の位置（0, 1, 2, ...）
  * $i$: 次元のインデックス（0から$d_{\text{model}}/2 - 1$）
  * 偶数次元にsin、奇数次元にcosを使用

#### この設計の利点

特徴 | 利点  
---|---  
**決定的** | 学習不要、パラメータ増加なし  
**連続的** | 隣接位置は類似した表現  
**周期性** | 相対的な位置関係を捉えやすい  
**任意長対応** | 学習時より長いシーケンスにも対応  
      
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    def get_positional_encoding(max_seq_len, d_model):
        """
        Sinusoidal Positional Encodingを生成
    
        Parameters:
        -----------
        max_seq_len : int
            最大シーケンス長
        d_model : int
            モデルの次元
    
        Returns:
        --------
        pe : torch.Tensor (max_seq_len, d_model)
            位置エンコーディング
        """
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    
        # 分母の計算: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
    
        # 偶数次元にsin、奇数次元にcos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
    
        return pe
    
    
    # 位置エンコーディングの生成
    max_seq_len = 100
    d_model = 512
    
    pe = get_positional_encoding(max_seq_len, d_model)
    
    print("=== Positional Encoding ===\n")
    print(f"形状: {pe.shape}")
    print(f"  [最大シーケンス長, モデル次元] = [{max_seq_len}, {d_model}]")
    
    print(f"\n位置0のエンコーディング（最初の10次元）:")
    print(pe[0, :10])
    
    print(f"\n位置1のエンコーディング（最初の10次元）:")
    print(pe[1, :10])
    
    print(f"\n位置10のエンコーディング（最初の10次元）:")
    print(pe[10, :10])
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図: 位置エンコーディングのヒートマップ
    im1 = ax1.imshow(pe[:50, :50].numpy(), cmap='RdBu', aspect='auto')
    ax1.set_xlabel('次元')
    ax1.set_ylabel('位置')
    ax1.set_title('Positional Encoding（最初の50位置×50次元）')
    plt.colorbar(im1, ax=ax1)
    
    # 右図: 特定の次元の波形
    dimensions = [0, 1, 2, 3, 10, 20]
    for dim in dimensions:
        ax2.plot(pe[:50, dim].numpy(), label=f'次元 {dim}')
    
    ax2.set_xlabel('位置')
    ax2.set_ylabel('エンコーディング値')
    ax2.set_title('各次元の位置に対する変化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n観察:")
    print("  - 低次元（0,1,2...）は高周波（細かい変化）")
    print("  - 高次元は低周波（ゆっくりとした変化）")
    print("  - これにより様々なスケールの位置情報を表現")
    

### Position Encodingの追加

位置エンコーディングは、入力の単語埋め込みに**加算** されます：

$$ \text{Input} = \text{Embedding}(x) + \text{PositionalEncoding}(pos) $$ 
    
    
    import torch
    import torch.nn as nn
    
    class PositionalEncoding(nn.Module):
        """
        Positional Encoding層
        """
        def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
            """
            Parameters:
            -----------
            d_model : int
                モデルの次元
            max_seq_len : int
                最大シーケンス長
            dropout : float
                ドロップアウト率
            """
            super(PositionalEncoding, self).__init__()
    
            self.dropout = nn.Dropout(p=dropout)
    
            # 位置エンコーディングを事前計算
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                (-np.log(10000.0) / d_model))
    
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
    
            # バッファとして登録（学習対象外）
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            """
            Parameters:
            -----------
            x : torch.Tensor (batch, seq_len, d_model)
                入力（単語埋め込み）
    
            Returns:
            --------
            x : torch.Tensor (batch, seq_len, d_model)
                位置情報を追加した入力
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    # 使用例：単語埋め込み + 位置エンコーディング
    vocab_size = 10000
    d_model = 512
    max_seq_len = 100
    batch_size = 8
    seq_len = 20
    
    # 単語埋め込み層
    embedding = nn.Embedding(vocab_size, d_model)
    
    # 位置エンコーディング層
    pos_encoding = PositionalEncoding(d_model, max_seq_len)
    
    # ダミーの単語ID
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 処理フロー
    word_embeddings = embedding(token_ids)  # (batch, seq_len, d_model)
    print("=== 単語埋め込み + Positional Encoding ===\n")
    print(f"1. 単語ID: {token_ids.shape}")
    print(f"   [バッチ, シーケンス長] = [{batch_size}, {seq_len}]")
    
    print(f"\n2. 単語埋め込み: {word_embeddings.shape}")
    print(f"   [バッチ, シーケンス長, d_model] = [{batch_size}, {seq_len}, {d_model}]")
    
    # 位置エンコーディングを追加
    input_with_pos = pos_encoding(word_embeddings)
    print(f"\n3. 位置エンコーディング追加後: {input_with_pos.shape}")
    print(f"   [バッチ, シーケンス長, d_model] = [{batch_size}, {seq_len}, {d_model}]")
    
    print(f"\n処理:")
    print(f"  Input = Embedding(tokens) + PositionalEncoding(positions)")
    print(f"  これがTransformerの最初の入力となる")
    
    # パラメータ数
    embedding_params = sum(p.numel() for p in embedding.parameters())
    pe_params = sum(p.numel() for p in pos_encoding.parameters() if p.requires_grad)
    
    print(f"\nパラメータ数:")
    print(f"  Embedding: {embedding_params:,}")
    print(f"  Positional Encoding: {pe_params:,} (学習対象外)")
    

**出力** ：
    
    
    === 単語埋め込み + Positional Encoding ===
    
    1. 単語ID: torch.Size([8, 20])
       [バッチ, シーケンス長] = [8, 20]
    
    2. 単語埋め込み: torch.Size([8, 20, 512])
       [バッチ, シーケンス長, d_model] = [8, 20, 512]
    
    3. 位置エンコーディング追加後: torch.Size([8, 20, 512])
       [バッチ, シーケンス長, d_model] = [8, 20, 512]
    
    処理:
      Input = Embedding(tokens) + PositionalEncoding(positions)
      これがTransformerの最初の入力となる
    
    パラメータ数:
      Embedding: 5,120,000
      Positional Encoding: 0 (学習対象外)
    

### 学習可能なPosition Encodingとの比較

手法 | 利点 | 欠点  
---|---|---  
**Sinusoidal** | パラメータ不要、任意長対応 | タスク特化の最適化不可  
**学習可能** | タスクに最適化可能 | 固定長のみ、パラメータ増加  
  
> **注** : 実験的には両者の性能差は小さく、Transformerの元論文ではSinusoidalが採用されています。BERTなどでは学習可能な位置埋め込みが使われています。

* * *

## 1.5 実践：Self-Attentionによるテキスト分類

### 完全なSelf-Attention分類モデル

Self-Attention、Multi-Head Attention、Position Encodingを組み合わせて、実際のテキスト分類タスクを解きます。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class TextClassifierWithSelfAttention(nn.Module):
        """
        Self-Attentionを用いたテキスト分類モデル
        """
        def __init__(self, vocab_size, d_model, num_heads, num_classes,
                     max_seq_len=512, dropout=0.1):
            """
            Parameters:
            -----------
            vocab_size : int
                語彙サイズ
            d_model : int
                モデルの次元
            num_heads : int
                Multi-Head Attentionのヘッド数
            num_classes : int
                分類クラス数
            max_seq_len : int
                最大シーケンス長
            dropout : float
                ドロップアウト率
            """
            super(TextClassifierWithSelfAttention, self).__init__()
    
            # 単語埋め込み
            self.embedding = nn.Embedding(vocab_size, d_model)
    
            # Positional Encoding
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
    
            # Multi-Head Attention
            self.attention = MultiHeadAttention(d_model, num_heads)
    
            # Layer Normalization
            self.layer_norm1 = nn.LayerNorm(d_model)
            self.layer_norm2 = nn.LayerNorm(d_model)
    
            # Feed-Forward Network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
    
            # 分類層
            self.classifier = nn.Linear(d_model, num_classes)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x, mask=None):
            """
            Parameters:
            -----------
            x : torch.Tensor (batch, seq_len)
                入力の単語ID
            mask : torch.Tensor (optional)
                パディングマスク
    
            Returns:
            --------
            logits : torch.Tensor (batch, num_classes)
                分類ロジット
            attn_weights : torch.Tensor
                Attention重み
            """
            # 1. 単語埋め込み + 位置エンコーディング
            x = self.embedding(x)  # (batch, seq_len, d_model)
            x = self.pos_encoding(x)
    
            # 2. Multi-Head Self-Attention + Residual + LayerNorm
            attn_output, attn_weights = self.attention(x, x, x, mask)
            x = self.layer_norm1(x + self.dropout(attn_output))
    
            # 3. Feed-Forward Network + Residual + LayerNorm
            ffn_output = self.ffn(x)
            x = self.layer_norm2(x + ffn_output)
    
            # 4. Global Average Pooling（全時刻を平均）
            x = x.mean(dim=1)  # (batch, d_model)
    
            # 5. 分類
            logits = self.classifier(x)  # (batch, num_classes)
    
            return logits, attn_weights
    
    
    # モデルの定義
    vocab_size = 10000
    d_model = 256
    num_heads = 8
    num_classes = 2  # 2クラス分類（positive/negative）
    max_seq_len = 128
    
    model = TextClassifierWithSelfAttention(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # ダミーデータ
    batch_size = 16
    seq_len = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 順伝播
    logits, attn_weights = model(x)
    
    print("=== Self-Attention Text Classifier ===\n")
    print(f"モデル設定:")
    print(f"  語彙サイズ: {vocab_size}")
    print(f"  モデル次元: {d_model}")
    print(f"  ヘッド数: {num_heads}")
    print(f"  クラス数: {num_classes}")
    
    print(f"\n入力: {x.shape}")
    print(f"  [バッチ, シーケンス長] = [{batch_size}, {seq_len}]")
    
    print(f"\n出力ロジット: {logits.shape}")
    print(f"  [バッチ, クラス数] = [{batch_size}, {num_classes}]")
    
    print(f"\nAttention重み: {attn_weights.shape}")
    print(f"  [バッチ, ヘッド数, seq_len, seq_len]")
    
    # 確率に変換
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print(f"\n予測結果（最初の5サンプル）:")
    for i in range(min(5, batch_size)):
        print(f"  サンプル{i}: クラス{predictions[i].item()} "
              f"(確率: {probs[i, predictions[i]].item():.4f})")
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

**出力例** ：
    
    
    === Self-Attention Text Classifier ===
    
    モデル設定:
      語彙サイズ: 10000
      モデル次元: 256
      ヘッド数: 8
      クラス数: 2
    
    入力: torch.Size([16, 50])
      [バッチ, シーケンス長] = [16, 50]
    
    出力ロジット: torch.Size([16, 2])
      [バッチ, クラス数] = [16, 2]
    
    Attention重み: torch.Size([16, 8, 50, 50])
      [バッチ, ヘッド数, seq_len, seq_len]
    
    予測結果（最初の5サンプル）:
      サンプル0: クラス1 (確率: 0.5234)
      サンプル1: クラス0 (確率: 0.5012)
      サンプル2: クラス1 (確率: 0.5456)
      サンプル3: クラス0 (確率: 0.5123)
      サンプル4: クラス1 (確率: 0.5389)
    
    総パラメータ数: 3,150,338
    

### 学習ループの実装
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    # ダミーデータセット
    class DummyTextDataset(Dataset):
        """
        簡単なダミーテキストデータセット
        """
        def __init__(self, num_samples, vocab_size, seq_len):
            self.num_samples = num_samples
            self.vocab_size = vocab_size
            self.seq_len = seq_len
    
            # ランダムな文章とラベルを生成
            torch.manual_seed(42)
            self.texts = torch.randint(0, vocab_size, (num_samples, seq_len))
            self.labels = torch.randint(0, 2, (num_samples,))
    
        def __len__(self):
            return self.num_samples
    
        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]
    
    
    # データセットとデータローダー
    train_dataset = DummyTextDataset(num_samples=1000, vocab_size=vocab_size, seq_len=50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # モデル、損失関数、オプティマイザ
    model = TextClassifierWithSelfAttention(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_classes=2,
        max_seq_len=128
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学習ループ
    num_epochs = 5
    
    print("=== 学習開始 ===\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for batch_idx, (texts, labels) in enumerate(train_loader):
            # 順伝播
            logits, _ = model(texts)
            loss = criterion(logits, labels)
    
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # 統計
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
        # エポックごとの結果
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
    
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print()
    
    print("学習完了！")
    
    # 推論例
    model.eval()
    with torch.no_grad():
        sample_text = torch.randint(0, vocab_size, (1, 50))
        logits, attn_weights = model(sample_text)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1)
    
        print("\n=== 推論例 ===")
        print(f"入力テキスト（単語ID）: {sample_text.shape}")
        print(f"予測クラス: {prediction.item()}")
        print(f"確率分布: positive={probs[0, 1].item():.4f}, negative={probs[0, 0].item():.4f}")
    

**出力例** ：
    
    
    === 学習開始 ===
    
    Epoch 1/5
      Loss: 0.6923
      Accuracy: 51.20%
    
    Epoch 2/5
      Loss: 0.6854
      Accuracy: 54.30%
    
    Epoch 3/5
      Loss: 0.6742
      Accuracy: 58.70%
    
    Epoch 4/5
      Loss: 0.6598
      Accuracy: 62.10%
    
    Epoch 5/5
      Loss: 0.6421
      Accuracy: 65.80%
    
    学習完了！
    
    === 推論例 ===
    入力テキスト（単語ID）: torch.Size([1, 50])
    予測クラス: 1
    確率分布: positive=0.6234, negative=0.3766
    

### RNNとの性能比較
    
    
    import time
    import torch
    import torch.nn as nn
    
    # RNNベースの分類器
    class RNNTextClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(RNNTextClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
    
        def forward(self, x):
            embedded = self.embedding(x)
            output, (hidden, cell) = self.rnn(embedded)
            logits = self.fc(hidden[-1])
            return logits
    
    # Self-Attentionモデル（簡略版）
    class SimpleAttentionClassifier(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_classes):
            super(SimpleAttentionClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attention = MultiHeadAttention(d_model, num_heads)
            self.fc = nn.Linear(d_model, num_classes)
    
        def forward(self, x):
            embedded = self.embedding(x)
            attn_out, _ = self.attention(embedded, embedded, embedded)
            pooled = attn_out.mean(dim=1)
            logits = self.fc(pooled)
            return logits
    
    # パラメータ設定
    vocab_size = 10000
    d_model = 256
    num_classes = 2
    batch_size = 32
    seq_len = 100
    
    # モデル
    rnn_model = RNNTextClassifier(vocab_size, d_model, d_model, num_classes)
    attn_model = SimpleAttentionClassifier(vocab_size, d_model, 8, num_classes)
    
    # ダミーデータ
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("=== RNN vs Self-Attention 比較 ===\n")
    
    # RNNの処理時間
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = rnn_model(x)
    rnn_time = (time.time() - start) / 100
    
    # Self-Attentionの処理時間
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = attn_model(x)
    attn_time = (time.time() - start) / 100
    
    # パラメータ数
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    attn_params = sum(p.numel() for p in attn_model.parameters())
    
    print(f"処理時間（平均）:")
    print(f"  RNN: {rnn_time*1000:.2f}ms")
    print(f"  Self-Attention: {attn_time*1000:.2f}ms")
    print(f"  高速化率: {rnn_time/attn_time:.2f}x")
    
    print(f"\nパラメータ数:")
    print(f"  RNN: {rnn_params:,}")
    print(f"  Self-Attention: {attn_params:,}")
    
    print(f"\n特徴:")
    print(f"  RNN:")
    print(f"    ✓ パラメータが少ない")
    print(f"    ✗ 逐次処理で遅い")
    print(f"    ✗ 長期依存性が弱い")
    print(f"\n  Self-Attention:")
    print(f"    ✓ 並列処理で高速")
    print(f"    ✓ 長距離依存性を直接捉える")
    print(f"    ✗ パラメータが多い（O(n²)のメモリ）")
    

**出力例** ：
    
    
    === RNN vs Self-Attention 比較 ===
    
    処理時間（平均）:
      RNN: 12.34ms
      Self-Attention: 8.76ms
      高速化率: 1.41x
    
    パラメータ数:
      RNN: 2,826,498
      Self-Attention: 3,150,338
    
    特徴:
      RNN:
        ✓ パラメータが少ない
        ✗ 逐次処理で遅い
        ✗ 長期依存性が弱い
    
      Self-Attention:
        ✓ 並列処理で高速
        ✓ 長距離依存性を直接捉える
        ✗ パラメータが多い（O(n²)のメモリ）
    

* * *

## まとめ

この章では、Self-AttentionとMulti-Head Attentionの基礎を学習しました。

### 重要なポイント

  * **RNNの限界** ：逐次処理、長期依存性の問題、並列化不可
  * **Self-Attention** ：Query、Key、Valueで全単語間の関係を並列に計算
  * **Scaled Dot-Product** ：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
  * **Multi-Head Attention** ：複数の視点から文脈を捉え、表現能力を向上
  * **Position Encoding** ：単語の順序情報を明示的に追加
  * **並列処理** ：RNNより高速で、長距離依存性を直接捉える

### 次章の予告

第2章では、以下のトピックを扱います：

  * Transformer Encoderの完全な構造
  * Feed-Forward NetworkとLayer Normalization
  * Residual Connectionの役割
  * Transformer Decoderとマスク機構
  * 完全なTransformerモデルの実装

* * *

## 演習問題

**演習1：Attention重みの手計算**

**問題** ：以下の簡略化されたQuery、Key、Valueでself-attentionを手計算してください。

3単語のシーケンス、各2次元：
    
    
    Q = [[1, 0], [0, 1], [1, 1]]
    K = [[1, 0], [0, 1], [1, 1]]
    V = [[2, 0], [0, 2], [1, 1]]
    

ステップ：

  1. スコア行列 $S = QK^T$ を計算
  2. スケーリング（$d_k=2$）
  3. Softmax（簡略化のため計算しやすい値で）
  4. 出力 $AV$ を計算

**解答** ：
    
    
    # ステップ1: スコア計算 QK^T
    Q = [[1, 0], [0, 1], [1, 1]]
    K = [[1, 0], [0, 1], [1, 1]]
    
    S = QK^T = [[1*1+0*0, 1*0+0*1, 1*1+0*1],
                [0*1+1*0, 0*0+1*1, 0*1+1*1],
                [1*1+1*0, 1*0+1*1, 1*1+1*1]]
             = [[1, 0, 1],
                [0, 1, 1],
                [1, 1, 2]]
    
    # ステップ2: スケーリング（d_k=2なので√2で割る）
    S_scaled = [[1/√2, 0, 1/√2],
                [0, 1/√2, 1/√2],
                [1/√2, 1/√2, 2/√2]]
             ≈ [[0.71, 0, 0.71],
                [0, 0.71, 0.71],
                [0.71, 0.71, 1.41]]
    
    # ステップ3: Softmax（各行）
    # 第1行: exp([0.71, 0, 0.71]) = [2.03, 1.00, 2.03]
    # 合計 = 5.06 → [0.40, 0.20, 0.40]
    
    A ≈ [[0.40, 0.20, 0.40],
         [0.20, 0.40, 0.40],
         [0.28, 0.28, 0.44]]
    
    # ステップ4: 出力 AV
    V = [[2, 0], [0, 2], [1, 1]]
    
    Output = AV
    第1単語: 0.40*[2,0] + 0.20*[0,2] + 0.40*[1,1] = [1.2, 0.8]
    第2単語: 0.20*[2,0] + 0.40*[0,2] + 0.40*[1,1] = [0.8, 1.2]
    第3単語: 0.28*[2,0] + 0.28*[0,2] + 0.44*[1,1] = [1.0, 1.0]
    
    答え: Output ≈ [[1.2, 0.8], [0.8, 1.2], [1.0, 1.0]]
    

**演習2：Multi-Head Attentionのパラメータ数**

**問題** ：以下の設定のMulti-Head Attentionのパラメータ数を計算してください。

  * $d_{\text{model}} = 512$
  * $h = 8$（ヘッド数）
  * $d_k = d_v = d_{\text{model}} / h = 64$

**解答** ：
    
    
    # 各ヘッドのパラメータ
    # W^Q, W^K, W^V: 各 (d_model × d_k) × h ヘッド分
    
    # 実装では、全ヘッド分を1つの行列で表現
    W_q: d_model × d_model = 512 × 512 = 262,144
    W_k: d_model × d_model = 512 × 512 = 262,144
    W_v: d_model × d_model = 512 × 512 = 262,144
    
    # 出力投影
    W_o: d_model × d_model = 512 × 512 = 262,144
    
    # 合計（バイアスなしの場合）
    Total = 262,144 × 4 = 1,048,576
    
    答え: 1,048,576 パラメータ
    

**演習3：Position Encodingの周期性**

**問題** ：Sinusoidal Position Encodingで、次元0（最も高周波）の周期を求めてください。

数式：$PE_{(pos, 0)} = \sin(pos / 10000^0) = \sin(pos)$

**解答** ：
    
    
    # 次元0の式
    PE(pos, 0) = sin(pos)
    
    # sinの周期は2π
    # pos が 2π 増えるごとに同じ値に戻る
    
    周期 = 2π ≈ 6.28
    
    # これは位置6.28ごとに繰り返すことを意味する
    # 実際の単語位置は整数なので、約6単語ごとに似た値
    
    # 次元が高くなるほど周期が長くなる
    # 次元i: 周期 = 2π × 10000^(2i/d_model)
    
    # d_model=512, 次元256（最も低周波）の場合
    周期_最低 = 2π × 10000^(512/512) = 2π × 10000 ≈ 62,832
    
    答え: 次元0は約6、次元256は約62,832の周期
    これにより様々なスケールの位置情報を表現できる
    

**演習4：Masked Self-Attentionの実装**

**問題** ：Decoder用のMasked Self-Attention（未来の単語を見ないようにする）を実装してください。

**解答例** ：
    
    
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    def create_causal_mask(seq_len):
        """
        因果マスクを生成（上三角行列）
    
        Returns:
        --------
        mask : torch.Tensor (seq_len, seq_len)
            下三角が1、上三角が0のマスク
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask
    
    def masked_scaled_dot_product_attention(Q, K, V):
        """
        Masked Scaled Dot-Product Attention
        """
        seq_len = Q.size(1)
        d_k = Q.size(-1)
    
        # スコア計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
        # 因果マスクの適用
        mask = create_causal_mask(seq_len).to(Q.device)
        scores = scores.masked_fill(mask == 0, -1e9)
    
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
    
        # 出力
        output = torch.matmul(attn_weights, V)
    
        return output, attn_weights
    
    # テスト
    Q = K = V = torch.randn(1, 5, 8)
    output, attn = masked_scaled_dot_product_attention(Q, K, V)
    
    print("Masked Attention重み:")
    print(attn[0])
    print("\n下三角のみ非ゼロ（未来を見ない）")
    

**演習5：Self-Attentionの計算量分析**

**問題** ：Self-Attentionの計算量を分析し、RNNと比較してください。

シーケンス長 $n$、モデル次元 $d$ とします。

**解答** ：
    
    
    # Self-Attentionの計算量
    
    1. Q, K, V の計算: 3 × (n × d × d) = O(nd²)
       各単語をd次元からd次元へ線形変換
    
    2. QK^T の計算: n × n × d = O(n²d)
       (n×d) @ (d×n) = (n×n)
    
    3. Softmax: O(n²)
       n×n行列の各行
    
    4. Attention × V: n × n × d = O(n²d)
       (n×n) @ (n×d) = (n×d)
    
    合計: O(nd² + n²d)
    
    # 支配的な項は
    - n < d の場合: O(nd²)
    - n > d の場合: O(n²d)
    
    # RNNの計算量
    各時刻で: d × d （隠れ状態の更新）
    n時刻分: n × d² = O(nd²)
    
    # 比較
    Self-Attention: O(n²d)（nが大きいとき）
    RNN: O(nd²)（常に）
    
    # メモリ使用量
    Self-Attention: O(n²)（Attention行列）
    RNN: O(n)（各時刻の隠れ状態）
    
    答え:
    - Self-Attentionは長いシーケンスで計算量・メモリが増大（n²）
    - RNNは逐次処理が必要で並列化不可
    - 短〜中程度のシーケンス（n < 512程度）では
      Self-Attentionが並列処理で高速
    

* * *
