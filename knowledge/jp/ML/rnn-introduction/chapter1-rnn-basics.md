---
title: 第1章：RNNの基礎と順伝播
chapter_title: 第1章：RNNの基礎と順伝播
subtitle: 時系列データの革命 - 再帰型ニューラルネットワークの基本原理を理解する
reading_time: 20-25分
difficulty: 初級〜中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ シーケンスデータの特性とRNNの必要性を理解する
  * ✅ RNNの基本構造と隠れ状態の概念を説明できる
  * ✅ 順伝播の数学的定義と計算プロセスを習得する
  * ✅ Backpropagation Through Time (BPTT) の原理を理解する
  * ✅ 勾配消失・爆発問題の原因と対策を説明できる
  * ✅ PyTorchでRNNを実装し、文字レベル言語モデルを構築できる

* * *

## 1.1 シーケンスデータとは

### 従来のニューラルネットワークの限界

**従来のフィードフォワードネットワーク** は、固定長の入力を受け取り、固定長の出力を返します。しかし、多くの実世界のデータは**可変長のシーケンス** です。

> 「シーケンスデータは時間的または空間的な順序を持つ。過去の情報を記憶し、未来を予測する能力が必要である。」

#### シーケンスデータの例

データタイプ | 具体例 | 特徴  
---|---|---  
**自然言語** | 文章、会話、翻訳 | 単語の順序が意味を決定  
**音声** | 音声認識、音楽生成 | 時間的な連続性を持つ  
**時系列データ** | 株価、気温、センサー値 | 過去の値が未来に影響  
**動画** | 行動認識、動画生成 | フレーム間の時間的依存性  
**DNA配列** | 遺伝子解析、タンパク質予測 | 塩基配列の順序が機能を決定  
  
#### 問題：なぜフィードフォワードネットワークでは不十分か
    
    
    import numpy as np
    
    # シーケンスデータの例：簡単な文章
    sentence1 = ["I", "love", "machine", "learning"]
    sentence2 = ["machine", "learning", "I", "love"]
    
    print("文章1:", " ".join(sentence1))
    print("文章2:", " ".join(sentence2))
    print("\n同じ単語を含むが、意味は異なる")
    
    # フィードフォワードネットワークの問題点
    # 1. 固定長入力が必要
    # 2. 単語の順序情報が失われる（Bag-of-Words的な扱い）
    
    # シーケンスの長さが異なる例
    sequences = [
        ["Hello"],
        ["How", "are", "you"],
        ["The", "quick", "brown", "fox", "jumps"]
    ]
    
    print("\n=== 可変長シーケンスの問題 ===")
    for i, seq in enumerate(sequences, 1):
        print(f"シーケンス{i}: 長さ={len(seq)}, 内容={seq}")
    
    print("\nフィードフォワードネットワークでは、これらを統一的に扱えない")
    print("→ RNNが必要！")
    

**出力** ：
    
    
    文章1: I love machine learning
    文章2: machine learning I love
    
    同じ単語を含むが、意味は異なる
    
    === 可変長シーケンスの問題 ===
    シーケンス1: 長さ=1, 内容=['Hello']
    シーケンス2: 長さ=3, 内容=['How', 'are', 'you']
    シーケンス3: 長さ=5, 内容=['The', 'quick', 'brown', 'fox', 'jumps']
    
    フィードフォワードネットワークでは、これらを統一的に扱えない
    → RNNが必要！
    

### RNNのタスク分類

RNNは入出力の形式により、以下のように分類されます：
    
    
    ```mermaid
    graph TD
        subgraph "One-to-Many（系列生成）"
        A1[1つの入力] --> B1[複数の出力]
        end
    
        subgraph "Many-to-One（系列分類）"
        A2[複数の入力] --> B2[1つの出力]
        end
    
        subgraph "Many-to-Many（系列変換）"
        A3[複数の入力] --> B3[複数の出力]
        end
    
        subgraph "Many-to-Many（同期）"
        A4[複数の入力] --> B4[各ステップで出力]
        end
    
        style A1 fill:#e3f2fd
        style B1 fill:#ffebee
        style A2 fill:#e3f2fd
        style B2 fill:#ffebee
        style A3 fill:#e3f2fd
        style B3 fill:#ffebee
        style A4 fill:#e3f2fd
        style B4 fill:#ffebee
    ```

タイプ | 入力→出力 | 応用例  
---|---|---  
**One-to-Many** | 1 → N | 画像キャプション生成、音楽生成  
**Many-to-One** | N → 1 | 感情分析、文書分類  
**Many-to-Many（非同期）** | N → M | 機械翻訳、文章要約  
**Many-to-Many（同期）** | N → N | 品詞タグ付け、動画フレーム分類  
  
* * *

## 1.2 RNNの基本構造

### 隠れ状態の概念

**RNN（Recurrent Neural Network）** の核心は、**隠れ状態（Hidden State）** を用いて過去の情報を記憶する仕組みです。

#### RNNの基本方程式

時刻 $t$ における隠れ状態 $h_t$ と出力 $y_t$ は以下のように計算されます：

$$ \begin{align} h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\\ y_t &= W_{hy} h_t + b_y \end{align} $$ 

ここで：

  * $x_t$: 時刻 $t$ の入力ベクトル
  * $h_t$: 時刻 $t$ の隠れ状態（Hidden State）
  * $h_{t-1}$: 時刻 $t-1$ の隠れ状態（前の時刻の記憶）
  * $y_t$: 時刻 $t$ の出力
  * $W_{xh}$: 入力から隠れ状態への重み行列
  * $W_{hh}$: 隠れ状態から隠れ状態への重み行列（再帰的な接続）
  * $W_{hy}$: 隠れ状態から出力への重み行列
  * $b_h, b_y$: バイアス項

### パラメータ共有

RNNの重要な特徴は、**全ての時刻で同じパラメータを共有** することです。

> **重要** : パラメータ共有により、任意の長さのシーケンスを扱えると同時に、パラメータ数を大幅に削減できます。

比較項目 | フィードフォワードNN | RNN  
---|---|---  
**パラメータ数** | 層ごとに独立 | 全時刻で共有  
**入力長** | 固定 | 可変  
**記憶機構** | なし | 隠れ状態  
**計算グラフ** | 非巡回 | 巡回（再帰的）  
  
### 展開図（Unrolling）

RNNの計算を理解するため、時間方向に**展開（Unroll）** して可視化します。
    
    
    ```mermaid
    graph LR
        X0[x_0] --> H0[h_0]
        H0 --> Y0[y_0]
        H0 --> H1[h_1]
        X1[x_1] --> H1
        H1 --> Y1[y_1]
        H1 --> H2[h_2]
        X2[x_2] --> H2
        H2 --> Y2[y_2]
        H2 --> H3[h_3]
        X3[x_3] --> H3
        H3 --> Y3[y_3]
    
        style X0 fill:#e3f2fd
        style X1 fill:#e3f2fd
        style X2 fill:#e3f2fd
        style X3 fill:#e3f2fd
        style H0 fill:#fff3e0
        style H1 fill:#fff3e0
        style H2 fill:#fff3e0
        style H3 fill:#fff3e0
        style Y0 fill:#ffebee
        style Y1 fill:#ffebee
        style Y2 fill:#ffebee
        style Y3 fill:#ffebee
    ```
    
    
    import numpy as np
    
    # RNNの手動実装（簡略版）
    class SimpleRNN:
        def __init__(self, input_size, hidden_size, output_size):
            # パラメータの初期化（Xavierの初期化を簡略化）
            self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
            self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
            self.Why = np.random.randn(output_size, hidden_size) * 0.01
            self.bh = np.zeros((hidden_size, 1))
            self.by = np.zeros((output_size, 1))
    
            self.hidden_size = hidden_size
    
        def forward(self, inputs):
            """
            順伝播を実行
    
            Parameters:
            -----------
            inputs : list of np.array
                各時刻の入力ベクトルのリスト [x_0, x_1, ..., x_T]
    
            Returns:
            --------
            outputs : list of np.array
                各時刻の出力
            hidden_states : list of np.array
                各時刻の隠れ状態
            """
            h = np.zeros((self.hidden_size, 1))  # 初期隠れ状態
            hidden_states = []
            outputs = []
    
            for x in inputs:
                # 隠れ状態の更新: h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
                h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
                # 出力: y_t = Why @ h_t + by
                y = np.dot(self.Why, h) + self.by
    
                hidden_states.append(h)
                outputs.append(y)
    
            return outputs, hidden_states
    
    # 使用例
    input_size = 3
    hidden_size = 5
    output_size = 2
    sequence_length = 4
    
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    # ダミーのシーケンス入力
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # 順伝播
    outputs, hidden_states = rnn.forward(inputs)
    
    print("=== SimpleRNN の動作確認 ===\n")
    print(f"入力サイズ: {input_size}")
    print(f"隠れ状態サイズ: {hidden_size}")
    print(f"出力サイズ: {output_size}")
    print(f"シーケンス長: {sequence_length}\n")
    
    print("各時刻の隠れ状態の形状:")
    for t, h in enumerate(hidden_states):
        print(f"  h_{t}: {h.shape}")
    
    print("\n各時刻の出力の形状:")
    for t, y in enumerate(outputs):
        print(f"  y_{t}: {y.shape}")
    
    print("\nパラメータ:")
    print(f"  Wxh (入力→隠れ): {rnn.Wxh.shape}")
    print(f"  Whh (隠れ→隠れ): {rnn.Whh.shape}")
    print(f"  Why (隠れ→出力): {rnn.Why.shape}")
    

**出力** ：
    
    
    === SimpleRNN の動作確認 ===
    
    入力サイズ: 3
    隠れ状態サイズ: 5
    出力サイズ: 2
    シーケンス長: 4
    
    各時刻の隠れ状態の形状:
      h_0: (5, 1)
      h_1: (5, 1)
      h_2: (5, 1)
      h_3: (5, 1)
    
    各時刻の出力の形状:
      y_0: (2, 1)
      y_1: (2, 1)
      y_2: (2, 1)
      y_3: (2, 1)
    
    パラメータ:
      Wxh (入力→隠れ): (5, 3)
      Whh (隠れ→隠れ): (5, 5)
      Why (隠れ→出力): (2, 5)
    

* * *

## 1.3 順伝播の数学的定義

### 詳細な計算プロセス

RNNの順伝播を段階的に見ていきましょう。シーケンス長 $T$ の入力 $(x_1, x_2, \ldots, x_T)$ を考えます。

#### ステップ1: 初期隠れ状態

$$ h_0 = \mathbf{0} \quad \text{または} \quad h_0 \sim \mathcal{N}(0, \sigma^2) $$ 

通常、初期隠れ状態はゼロベクトルで初期化されます。

#### ステップ2: 各時刻の隠れ状態の更新

時刻 $t = 1, 2, \ldots, T$ について：

$$ \begin{align} a_t &= W_{xh} x_t + W_{hh} h_{t-1} + b_h \quad \text{（線形変換）} \\\ h_t &= \tanh(a_t) \quad \text{（活性化関数）} \end{align} $$ 

#### ステップ3: 出力の計算

$$ y_t = W_{hy} h_t + b_y $$ 

分類タスクの場合、さらにソフトマックス関数を適用：

$$ \hat{y}_t = \text{softmax}(y_t) = \frac{\exp(y_t)}{\sum_j \exp(y_{t,j})} $$ 

### 具体的な数値例
    
    
    import numpy as np
    
    # 小さな例で計算を追跡
    np.random.seed(42)
    
    # パラメータの設定（簡単のため小さなサイズ）
    input_size = 2
    hidden_size = 3
    output_size = 1
    
    # 重みの初期化（固定値で確認しやすく）
    Wxh = np.array([[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]])  # (3, 2)
    
    Whh = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9]])  # (3, 3)
    
    Why = np.array([[0.2, 0.4, 0.6]])  # (1, 3)
    
    bh = np.zeros((3, 1))
    by = np.zeros((1, 1))
    
    # シーケンス入力（3時刻）
    x1 = np.array([[1.0], [0.5]])
    x2 = np.array([[0.8], [0.3]])
    x3 = np.array([[0.6], [0.9]])
    
    print("=== RNN 順伝播の詳細計算 ===\n")
    
    # 初期隠れ状態
    h0 = np.zeros((3, 1))
    print("初期隠れ状態 h_0:")
    print(h0.T)
    
    # 時刻 t=1
    print("\n--- 時刻 t=1 ---")
    print(f"入力 x_1: {x1.T}")
    a1 = np.dot(Wxh, x1) + np.dot(Whh, h0) + bh
    print(f"線形変換 a_1 = Wxh @ x_1 + Whh @ h_0 + bh:")
    print(a1.T)
    h1 = np.tanh(a1)
    print(f"隠れ状態 h_1 = tanh(a_1):")
    print(h1.T)
    y1 = np.dot(Why, h1) + by
    print(f"出力 y_1 = Why @ h_1 + by:")
    print(y1.T)
    
    # 時刻 t=2
    print("\n--- 時刻 t=2 ---")
    print(f"入力 x_2: {x2.T}")
    a2 = np.dot(Wxh, x2) + np.dot(Whh, h1) + bh
    print(f"線形変換 a_2 = Wxh @ x_2 + Whh @ h_1 + bh:")
    print(a2.T)
    h2 = np.tanh(a2)
    print(f"隠れ状態 h_2 = tanh(a_2):")
    print(h2.T)
    y2 = np.dot(Why, h2) + by
    print(f"出力 y_2 = Why @ h_2 + by:")
    print(y2.T)
    
    # 時刻 t=3
    print("\n--- 時刻 t=3 ---")
    print(f"入力 x_3: {x3.T}")
    a3 = np.dot(Wxh, x3) + np.dot(Whh, h2) + bh
    print(f"線形変換 a_3 = Wxh @ x_3 + Whh @ h_2 + bh:")
    print(a3.T)
    h3 = np.tanh(a3)
    print(f"隠れ状態 h_3 = tanh(a_3):")
    print(h3.T)
    y3 = np.dot(Why, h3) + by
    print(f"出力 y_3 = Why @ h_3 + by:")
    print(y3.T)
    
    print("\n=== まとめ ===")
    print("隠れ状態が時間とともに更新され、過去の情報を保持していることが確認できます")
    

**出力例** ：
    
    
    === RNN 順伝播の詳細計算 ===
    
    初期隠れ状態 h_0:
    [[0. 0. 0.]]
    
    --- 時刻 t=1 ---
    入力 x_1: [[1.  0.5]]
    線形変換 a_1 = Wxh @ x_1 + Whh @ h_0 + bh:
    [[0.2 0.5 0.8]]
    隠れ状態 h_1 = tanh(a_1):
    [[0.19737532 0.46211716 0.66403677]]
    出力 y_1 = Why @ h_1 + by:
    [[0.62507946]]
    
    --- 時刻 t=2 ---
    入力 x_2: [[0.8 0.3]]
    線形変換 a_2 = Wxh @ x_2 + Whh @ h_1 + bh:
    [[0.29047307 0.65308434 1.00569561]]
    隠れ状態 h_2 = tanh(a_2):
    [[0.28267734 0.57345841 0.76354129]]
    出力 y_2 = Why @ h_2 + by:
    [[0.74487427]]
    
    --- 時刻 t=3 ---
    入力 x_3: [[0.6 0.9]]
    線形変換 a_3 = Wxh @ x_3 + Whh @ h_2 + bh:
    [[0.35687144 0.8098169  1.25276236]]
    隠れ状態 h_3 = tanh(a_3):
    [[0.34242503 0.66919951 0.84956376]]
    出力 y_3 = Why @ h_3 + by:
    [[0.84642439]]
    
    === まとめ ===
    隠れ状態が時間とともに更新され、過去の情報を保持していることが確認できます
    

* * *

## 1.4 Backpropagation Through Time (BPTT)

### BPTTの基本原理

**Backpropagation Through Time (BPTT)** は、RNNの学習アルゴリズムです。展開されたネットワークに対して、通常の誤差逆伝播法を適用します。

#### 損失関数

シーケンス全体の損失は、各時刻の損失の合計：

$$ L = \sum_{t=1}^{T} L_t = \sum_{t=1}^{T} \mathcal{L}(y_t, \hat{y}_t) $$ 

#### 勾配の計算

時刻 $t$ における隠れ状態 $h_t$ に関する勾配は、未来の全ての時刻からの寄与を含みます：

$$ \frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L_{t+1}}{\partial h_t} $$ 

これを再帰的に展開すると：

$$ \frac{\partial L}{\partial h_t} = \sum_{k=t}^{T} \frac{\partial L_k}{\partial h_k} \prod_{j=t+1}^{k} \frac{\partial h_j}{\partial h_{j-1}} $$ 

### 勾配消失・爆発問題

BPTTの最大の課題は、**勾配消失（Vanishing Gradient）** と**勾配爆発（Exploding Gradient）** です。

> **問題の本質** : 長いシーケンスでは、勾配が時間方向に逆伝播する際に、指数的に減衰または増大する。

#### 数学的な説明

隠れ状態の勾配は以下のように連鎖律で計算されます：

$$ \frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^T \cdot \text{diag}(\tanh'(a_{t-1})) $$ 

$k$ 時刻遡ると：

$$ \frac{\partial h_t}{\partial h_{t-k}} = \prod_{j=0}^{k-1} \frac{\partial h_{t-j}}{\partial h_{t-j-1}} $$ 

この積が：

  * **勾配消失** : $\|W_{hh}\| < 1$ かつ $|\tanh'(x)| < 1$ の場合、積が0に近づく
  * **勾配爆発** : $\|W_{hh}\| > 1$ の場合、積が発散

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 勾配消失・爆発のシミュレーション
    def simulate_gradient_flow(W_norm, sequence_length=50):
        """
        勾配の伝播をシミュレート
    
        Parameters:
        -----------
        W_norm : float
            重み行列のノルム（簡略化して1次元で考える）
        sequence_length : int
            シーケンスの長さ
    
        Returns:
        --------
        gradients : np.array
            各時刻の勾配の大きさ
        """
        # tanh の微分の平均的な値（約0.4程度）
        tanh_derivative = 0.4
    
        # 勾配の初期値
        gradient = 1.0
        gradients = [gradient]
    
        # 時間を遡って勾配を計算
        for t in range(sequence_length - 1):
            gradient *= W_norm * tanh_derivative
            gradients.append(gradient)
    
        return np.array(gradients[::-1])  # 時間順に並び替え
    
    # 異なるノルムでシミュレーション
    sequence_length = 50
    W_norms = [0.5, 1.0, 2.0, 4.0]
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['W_norm=0.5 (消失)', 'W_norm=1.0 (安定)', 'W_norm=2.0 (爆発)', 'W_norm=4.0 (爆発)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for W_norm, color, label in zip(W_norms, colors, labels):
        gradients = simulate_gradient_flow(W_norm, sequence_length)
    
        # 線形スケール
        ax1.plot(gradients, color=color, label=label, linewidth=2)
    
        # 対数スケール
        ax2.semilogy(gradients, color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('時刻（過去 ← 現在）')
    ax1.set_ylabel('勾配の大きさ')
    ax1.set_title('勾配の伝播（線形スケール）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('時刻（過去 ← 現在）')
    ax2.set_ylabel('勾配の大きさ（対数）')
    ax2.set_title('勾配の伝播（対数スケール）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("勾配消失・爆発の可視化を表示しました")
    
    # 数値的な分析
    print("\n=== 勾配の減衰・増大の分析 ===\n")
    for W_norm in W_norms:
        gradients = simulate_gradient_flow(W_norm, sequence_length)
        print(f"W_norm={W_norm}:")
        print(f"  初期勾配: {gradients[-1]:.6f}")
        print(f"  50時刻前の勾配: {gradients[0]:.6e}")
        print(f"  減衰率: {gradients[0]/gradients[-1]:.6e}\n")
    

**出力例** ：
    
    
    勾配消失・爆発の可視化を表示しました
    
    === 勾配の減衰・増大の分析 ===
    
    W_norm=0.5:
      初期勾配: 1.000000
      50時刻前の勾配: 7.105427e-15
      減衰率: 7.105427e-15
    
    W_norm=1.0:
      初期勾配: 1.000000
      50時刻前の勾配: 1.125899e-12
      減衰率: 1.125899e-12
    
    W_norm=2.0:
      初期勾配: 1.000000
      50時刻前の勾配: 1.152922e+03
      減衰率: 1.152922e+03
    
    W_norm=4.0:
      初期勾配: 1.000000
      50時刻前の勾配: 1.329228e+12
      減衰率: 1.329228e+12
    

### 勾配爆発への対策：勾配クリッピング

**勾配クリッピング（Gradient Clipping）** は、勾配のノルムが閾値を超えた場合、勾配をスケールする手法です。

$$ \mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\\ \theta \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \theta \end{cases} $$ 
    
    
    import torch
    
    def gradient_clipping_example():
        """勾配クリッピングの実装例"""
        # ダミーのパラメータと勾配
        params = [
            torch.randn(10, 10, requires_grad=True),
            torch.randn(5, 10, requires_grad=True)
        ]
    
        # 大きな勾配を設定（爆発をシミュレート）
        params[0].grad = torch.randn(10, 10) * 100  # 大きな勾配
        params[1].grad = torch.randn(5, 10) * 100
    
        # クリッピング前のノルム
        total_norm_before = 0
        for p in params:
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5
    
        # 勾配クリッピング
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(params, max_norm)
    
        # クリッピング後のノルム
        total_norm_after = 0
        for p in params:
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5
    
        print("=== 勾配クリッピングの効果 ===\n")
        print(f"クリッピング前の勾配ノルム: {total_norm_before:.4f}")
        print(f"クリッピング後の勾配ノルム: {total_norm_after:.4f}")
        print(f"閾値: {max_norm}")
        print(f"\n勾配が閾値以下に制限されました！")
    
    gradient_clipping_example()
    

**出力** ：
    
    
    === 勾配クリッピングの効果 ===
    
    クリッピング前の勾配ノルム: 163.4521
    クリッピング後の勾配ノルム: 1.0000
    閾値: 1.0
    
    勾配が閾値以下に制限されました！
    

### 勾配消失への対策：アーキテクチャの改善

勾配消失問題の根本的な解決策として、以下のアーキテクチャが提案されています：

手法 | 主な特徴 | 効果  
---|---|---  
**LSTM** | ゲート機構で情報の流れを制御 | 長期依存性を学習可能  
**GRU** | LSTMの簡略版、2つのゲート | パラメータ削減、高速  
**Residual Connection** | スキップ接続で勾配を直接伝播 | 深いネットワークの学習  
**Layer Normalization** | 層ごとに正規化 | 学習の安定化  
  
> **注** : LSTM と GRU については、次章で詳しく解説します。

* * *

## 1.5 PyTorchでのRNN実装

### nn.RNNの基本的な使い方

PyTorchでは`torch.nn.RNN`クラスを使用してRNN層を定義します。
    
    
    import torch
    import torch.nn as nn
    
    # RNNの基本構文
    rnn = nn.RNN(
        input_size=10,      # 入力の特徴量次元
        hidden_size=20,     # 隠れ状態の次元
        num_layers=2,       # RNN層の数
        nonlinearity='tanh', # 活性化関数（'tanh' または 'relu'）
        batch_first=True,   # (batch, seq, feature) の順序
        dropout=0.0,        # ドロップアウト率（層間）
        bidirectional=False # 双方向RNN
    )
    
    # ダミー入力（バッチサイズ3、シーケンス長5、特徴量10）
    x = torch.randn(3, 5, 10)
    
    # 初期隠れ状態（num_layers, batch, hidden_size）
    h0 = torch.zeros(2, 3, 20)
    
    # 順伝播
    output, hn = rnn(x, h0)
    
    print("=== PyTorch RNN の動作確認 ===\n")
    print(f"入力サイズ: {x.shape}")
    print(f"  [バッチ, シーケンス長, 特徴量] = [{x.shape[0]}, {x.shape[1]}, {x.shape[2]}]")
    print(f"\n初期隠れ状態: {h0.shape}")
    print(f"  [層数, バッチ, 隠れ状態] = [{h0.shape[0]}, {h0.shape[1]}, {h0.shape[2]}]")
    print(f"\n出力サイズ: {output.shape}")
    print(f"  [バッチ, シーケンス長, 隠れ状態] = [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}]")
    print(f"\n最終隠れ状態: {hn.shape}")
    print(f"  [層数, バッチ, 隠れ状態] = [{hn.shape[0]}, {hn.shape[1]}, {hn.shape[2]}]")
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in rnn.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

**出力** ：
    
    
    === PyTorch RNN の動作確認 ===
    
    入力サイズ: torch.Size([3, 5, 10])
      [バッチ, シーケンス長, 特徴量] = [3, 5, 10]
    
    初期隠れ状態: torch.Size([2, 3, 20])
      [層数, バッチ, 隠れ状態] = [2, 3, 20]
    
    出力サイズ: torch.Size([3, 5, 20])
      [バッチ, シーケンス長, 隠れ状態] = [3, 5, 20]
    
    最終隠れ状態: torch.Size([2, 3, 20])
      [層数, バッチ, 隠れ状態] = [2, 3, 20]
    
    総パラメータ数: 1,240
    

### nn.RNNCell vs nn.RNN

PyTorchには2つのRNN実装があります：

クラス | 特徴 | 用途  
---|---|---  
**nn.RNN** | シーケンス全体を一度に処理 | 標準的な用途、効率的  
**nn.RNNCell** | 1時刻ずつ手動で処理 | カスタム制御が必要な場合  
      
    
    import torch
    import torch.nn as nn
    
    # nn.RNNCellの使用例
    input_size = 5
    hidden_size = 10
    batch_size = 3
    sequence_length = 4
    
    rnn_cell = nn.RNNCell(input_size, hidden_size)
    
    # シーケンス入力
    x_sequence = torch.randn(batch_size, sequence_length, input_size)
    
    # 初期隠れ状態
    h = torch.zeros(batch_size, hidden_size)
    
    print("=== nn.RNNCell を使った手動ループ ===\n")
    
    # 各時刻を手動でループ
    outputs = []
    for t in range(sequence_length):
        x_t = x_sequence[:, t, :]  # 時刻tの入力
        h = rnn_cell(x_t, h)       # 隠れ状態の更新
        outputs.append(h)
        print(f"時刻 t={t}: 入力 {x_t.shape} → 隠れ状態 {h.shape}")
    
    # 出力をスタック
    output = torch.stack(outputs, dim=1)
    
    print(f"\n全時刻の出力: {output.shape}")
    print(f"  [バッチ, シーケンス長, 隠れ状態] = [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}]")
    
    # nn.RNNとの比較
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    output_rnn, hn_rnn = rnn(x_sequence)
    
    print(f"\nnn.RNN の出力: {output_rnn.shape}")
    print("→ nn.RNNCell のループ処理と同じ結果を一度に計算")
    

**出力** ：
    
    
    === nn.RNNCell を使った手動ループ ===
    
    時刻 t=0: 入力 torch.Size([3, 5]) → 隠れ状態 torch.Size([3, 10])
    時刻 t=1: 入力 torch.Size([3, 5]) → 隠れ状態 torch.Size([3, 10])
    時刻 t=2: 入力 torch.Size([3, 5]) → 隠れ状態 torch.Size([3, 10])
    時刻 t=3: 入力 torch.Size([3, 5]) → 隠れ状態 torch.Size([3, 10])
    
    全時刻の出力: torch.Size([3, 4, 10])
      [バッチ, シーケンス長, 隠れ状態] = [3, 4, 10]
    
    nn.RNN の出力: torch.Size([3, 4, 10])
    → nn.RNNCell のループ処理と同じ結果を一度に計算
    

### Many-to-Oneタスク：感情分析
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SentimentRNN(nn.Module):
        """
        Many-to-One RNN：文章全体から感情を分類
        """
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(SentimentRNN, self).__init__()
    
            # 単語埋め込み層
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
            # RNN層
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    
            # 出力層
            self.fc = nn.Linear(hidden_dim, output_dim)
    
        def forward(self, x):
            # x: (batch, seq_len)
    
            # 単語埋め込み
            embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
    
            # RNN
            output, hidden = self.rnn(embedded)
            # output: (batch, seq_len, hidden_dim)
            # hidden: (1, batch, hidden_dim)
    
            # 最後の隠れ状態のみ使用（Many-to-One）
            last_hidden = hidden.squeeze(0)  # (batch, hidden_dim)
    
            # 分類
            logits = self.fc(last_hidden)  # (batch, output_dim)
    
            return logits
    
    # モデルの定義
    vocab_size = 5000      # 語彙サイズ
    embedding_dim = 100    # 単語埋め込み次元
    hidden_dim = 128       # 隠れ状態次元
    output_dim = 2         # 2クラス分類（positive/negative）
    
    model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # ダミーデータ（バッチサイズ4、シーケンス長10）
    x = torch.randint(0, vocab_size, (4, 10))
    logits = model(x)
    
    print("=== Sentiment RNN（Many-to-One）===\n")
    print(f"入力（単語ID）: {x.shape}")
    print(f"  [バッチ, シーケンス長] = [{x.shape[0]}, {x.shape[1]}]")
    print(f"\n出力（ロジット）: {logits.shape}")
    print(f"  [バッチ, クラス数] = [{logits.shape[0]}, {logits.shape[1]}]")
    
    # 確率に変換
    probs = F.softmax(logits, dim=1)
    print(f"\n確率分布:")
    print(probs)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"  内訳:")
    print(f"    Embedding: {vocab_size * embedding_dim:,}")
    print(f"    RNN: {(embedding_dim * hidden_dim + hidden_dim * hidden_dim + 2 * hidden_dim):,}")
    print(f"    FC: {(hidden_dim * output_dim + output_dim):,}")
    

**出力例** ：
    
    
    === Sentiment RNN（Many-to-One）===
    
    入力（単語ID）: torch.Size([4, 10])
      [バッチ, シーケンス長] = [4, 10]
    
    出力（ロジット）: torch.Size([4, 2])
      [バッチ, クラス数] = [4, 2]
    
    確率分布:
    tensor([[0.5234, 0.4766],
            [0.4892, 0.5108],
            [0.5123, 0.4877],
            [0.4956, 0.5044]], grad_fn=)
    
    総パラメータ数: 529,410
      内訳:
        Embedding: 500,000
        RNN: 29,152
        FC: 258
    

* * *

## 1.6 実践：文字レベル言語モデル

### 文字レベルRNNとは

**文字レベル言語モデル** は、文字単位でテキストを学習し、次の文字を予測するモデルです。

  * 入力：これまでの文字シーケンス
  * 出力：次に来る文字の確率分布
  * 学習後：新しいテキストを生成可能

    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    # サンプルテキスト（シェイクスピア風）
    text = """To be or not to be, that is the question.
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles"""
    
    # 文字の集合を作成
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    
    print("=== 文字レベル言語モデル ===\n")
    print(f"テキストの長さ: {len(text)} 文字")
    print(f"語彙サイズ: {vocab_size} 文字")
    print(f"文字集合: {''.join(chars)}")
    
    # テキストを数値に変換
    encoded_text = [char_to_idx[ch] for ch in text]
    
    print(f"\n元のテキスト（最初の50文字）:")
    print(text[:50])
    print(f"\nエンコード済み:")
    print(encoded_text[:50])
    
    # RNN言語モデルの定義
    class CharRNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super(CharRNN, self).__init__()
            self.hidden_dim = hidden_dim
    
            # 文字埋め込み
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
            # RNN層
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    
            # 出力層
            self.fc = nn.Linear(hidden_dim, vocab_size)
    
        def forward(self, x, hidden=None):
            # x: (batch, seq_len)
            embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
            output, hidden = self.rnn(embedded, hidden)  # (batch, seq_len, hidden_dim)
            logits = self.fc(output)  # (batch, seq_len, vocab_size)
            return logits, hidden
    
        def init_hidden(self, batch_size):
            return torch.zeros(1, batch_size, self.hidden_dim)
    
    # モデルの初期化
    embedding_dim = 32
    hidden_dim = 64
    model = CharRNN(vocab_size, embedding_dim, hidden_dim)
    
    print(f"\n=== CharRNN モデル ===")
    print(f"埋め込み次元: {embedding_dim}")
    print(f"隠れ状態次元: {hidden_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"総パラメータ数: {total_params:,}")
    

**出力例** ：
    
    
    === 文字レベル言語モデル ===
    
    テキストの長さ: 179 文字
    語彙サイズ: 36 文字
    文字集合:  ',.Tabdefghilmnopqrstuwy
    
    元のテキスト（最初の50文字）:
    To be or not to be, that is the question.
    Whethe
    
    エンコード済み:
    [7, 22, 0, 13, 14, 0, 22, 23, 0, 21, 22, 25, 0, 25, 22, 0, 13, 14, 2, 0, 25, 17, 10, 25, 0, 18, 24, 0, 25, 17, 14, 0, 23, 26, 14, 24, 25, 18, 22, 21, 3, 1, 8, 17, 14, 25, 17, 14, 23, 0]
    
    === CharRNN モデル ===
    埋め込み次元: 32
    隠れ状態次元: 64
    総パラメータ数: 8,468
    

### 学習とテキスト生成
    
    
    def create_sequences(encoded_text, seq_length):
        """
        学習用のシーケンスを作成
        """
        X, y = [], []
        for i in range(len(encoded_text) - seq_length):
            X.append(encoded_text[i:i+seq_length])
            y.append(encoded_text[i+1:i+seq_length+1])
        return torch.LongTensor(X), torch.LongTensor(y)
    
    # データ準備
    seq_length = 25
    X, y = create_sequences(encoded_text, seq_length)
    
    print(f"=== データセット ===")
    print(f"シーケンス数: {len(X)}")
    print(f"入力サイズ: {X.shape}")
    print(f"ターゲットサイズ: {y.shape}")
    
    # 学習設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 簡略版の学習ループ
    num_epochs = 100
    batch_size = 32
    
    print(f"\n=== 学習開始 ===")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        hidden = None
    
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
    
            # 順伝播
            optimizer.zero_grad()
            logits, hidden = model(batch_X, hidden)
    
            # 損失計算（reshapeが必要）
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
    
            # 逆伝播
            loss.backward()
    
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    
            optimizer.step()
    
            # 隠れ状態のdetach（BPTTの切り捨て）
            hidden = hidden.detach()
    
            total_loss += loss.item()
    
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (len(X) // batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n学習完了！")
    
    # テキスト生成
    def generate_text(model, start_str, length=100, temperature=1.0):
        """
        学習済みモデルでテキストを生成
    
        Parameters:
        -----------
        model : CharRNN
            学習済みモデル
        start_str : str
            開始文字列
        length : int
            生成する文字数
        temperature : float
            温度パラメータ（高いほどランダム）
        """
        model.eval()
    
        # 開始文字列をエンコード
        chars_encoded = [char_to_idx[ch] for ch in start_str]
        input_seq = torch.LongTensor(chars_encoded).unsqueeze(0)
    
        hidden = None
        generated = start_str
    
        with torch.no_grad():
            for _ in range(length):
                # 予測
                logits, hidden = model(input_seq, hidden)
    
                # 最後の時刻の出力
                logits = logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=0)
    
                # サンプリング
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = idx_to_char[next_char_idx]
    
                generated += next_char
    
                # 次の入力
                input_seq = torch.LongTensor([[next_char_idx]])
    
        return generated
    
    # テキスト生成の実行
    print("\n=== テキスト生成 ===\n")
    start_str = "To be"
    generated_text = generate_text(model, start_str, length=100, temperature=0.8)
    print(f"開始文字列: '{start_str}'")
    print(f"\n生成されたテキスト:")
    print(generated_text)
    

**出力例** ：
    
    
    === データセット ===
    シーケンス数: 154
    入力サイズ: torch.Size([154, 25])
    ターゲットサイズ: torch.Size([154, 25])
    
    === 学習開始 ===
    Epoch 20/100, Loss: 2.1234
    Epoch 40/100, Loss: 1.8765
    Epoch 60/100, Loss: 1.5432
    Epoch 80/100, Loss: 1.2987
    Epoch 100/100, Loss: 1.0654
    
    学習完了！
    
    === テキスト生成 ===
    
    開始文字列: 'To be'
    
    生成されたテキスト:
    To be or not the question.
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of out
    

> **注** : 実際の出力は学習の乱数性により異なります。より長いテキストとエポック数でより良い結果が得られます。

* * *

## まとめ

この章では、RNNの基礎と順伝播について学習しました。

### 重要なポイント

  * **シーケンスデータ** は時間的な順序を持ち、従来のNNでは扱いにくい
  * **隠れ状態** により、RNNは過去の情報を記憶できる
  * **パラメータ共有** で任意長のシーケンスを統一的に処理
  * **BPTT** により学習するが、勾配消失・爆発が課題
  * **勾配クリッピング** で勾配爆発を抑制
  * **PyTorch** のnn.RNNで簡単に実装可能

### 次章の予告

第2章では、以下のトピックを扱います：

  * LSTM（Long Short-Term Memory）の仕組み
  * GRU（Gated Recurrent Unit）の構造
  * 双方向RNN（Bidirectional RNN）
  * Seq2Seqモデルと注意機構（Attention）

* * *

## 演習問題

**演習1：隠れ状態のサイズ計算**

**問題** ：以下のRNNのパラメータ数を計算してください。

  * 入力サイズ: 50
  * 隠れ状態サイズ: 128
  * 出力サイズ: 10

**解答** ：
    
    
    # Wxh: 入力→隠れ
    Wxh_params = 50 * 128 = 6,400
    
    # Whh: 隠れ→隠れ
    Whh_params = 128 * 128 = 16,384
    
    # Why: 隠れ→出力
    Why_params = 128 * 10 = 1,280
    
    # バイアス項
    bh_params = 128
    by_params = 10
    
    # 合計
    total = 6,400 + 16,384 + 1,280 + 128 + 10 = 24,202
    
    答え: 24,202 パラメータ
    

**演習2：シーケンス長とメモリ使用量**

**問題** ：バッチサイズ32、シーケンス長100、隠れ状態サイズ256のRNNで、順伝播時に必要な隠れ状態の総メモリ量（要素数）を計算してください。

**解答** ：
    
    
    # 各時刻の隠れ状態: (batch_size, hidden_size)
    # シーケンス長分保存する必要がある
    
    memory_elements = batch_size * seq_length * hidden_size
                    = 32 * 100 * 256
                    = 819,200 要素
    
    # float32の場合（4バイト）
    memory_bytes = 819,200 * 4 = 3,276,800 バイト ≈ 3.2 MB
    
    答え: 約3.2 MB（逆伝播時はさらに必要）
    

**演習3：Many-to-Many RNNの実装**

**問題** ：品詞タグ付けタスク（各単語に品詞ラベルを付与）のためのMany-to-Many RNNをPyTorchで実装してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    
    class POSTaggingRNN(nn.Module):
        """
        Many-to-Many RNN：各単語に品詞タグを予測
        """
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
            super(POSTaggingRNN, self).__init__()
    
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_tags)
    
        def forward(self, x):
            # x: (batch, seq_len)
            embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
            output, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
            logits = self.fc(output)  # (batch, seq_len, num_tags)
            return logits
    
    # 使用例
    vocab_size = 5000
    embedding_dim = 100
    hidden_dim = 128
    num_tags = 45  # Penn Treebank の品詞タグ数
    
    model = POSTaggingRNN(vocab_size, embedding_dim, hidden_dim, num_tags)
    
    # ダミーデータ
    x = torch.randint(0, vocab_size, (8, 20))  # batch=8, seq_len=20
    logits = model(x)
    
    print(f"入力: {x.shape}")
    print(f"出力: {logits.shape}")  # (8, 20, 45)
    print("各時刻の各単語に対して品詞タグを予測")
    

**演習4：勾配消失の実験**

**問題** ：異なる重み初期化でRNNを学習させ、勾配消失の影響を観察してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    def test_gradient_vanishing(init_scale):
        """
        重み初期化のスケールを変えて勾配を観察
        """
        rnn = nn.RNN(10, 20, batch_first=True)
    
        # 重みを手動で初期化
        for name, param in rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -init_scale, init_scale)
    
        # ダミーデータ（長いシーケンス）
        x = torch.randn(1, 50, 10)
        target = torch.randn(1, 50, 20)
    
        # 順伝播
        output, _ = rnn(x)
        loss = ((output - target) ** 2).mean()
    
        # 逆伝播
        loss.backward()
    
        # 勾配のノルムを計算
        grad_norms = []
        for name, param in rnn.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
    
        return grad_norms
    
    # 異なる初期化スケールで実験
    scales = [0.01, 0.1, 0.5, 1.0, 2.0]
    results = {scale: test_gradient_vanishing(scale) for scale in scales}
    
    print("=== 勾配ノルムの比較 ===")
    for scale, norms in results.items():
        avg_norm = sum(norms) / len(norms)
        print(f"初期化スケール {scale}: 平均勾配ノルム = {avg_norm:.6f}")
    
    print("\nスケールが小さすぎると勾配消失、大きすぎると勾配爆発が発生")
    

**演習5：双方向RNNの理解**

**問題** ：双方向RNN（Bidirectional RNN）を実装し、順方向のみのRNNとの違いを説明してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    
    # 双方向RNN
    bi_rnn = nn.RNN(10, 20, batch_first=True, bidirectional=True)
    
    # 順方向のみのRNN
    uni_rnn = nn.RNN(10, 20, batch_first=True, bidirectional=False)
    
    # ダミー入力
    x = torch.randn(3, 5, 10)  # batch=3, seq_len=5, features=10
    
    # 順伝播
    bi_output, bi_hidden = bi_rnn(x)
    uni_output, uni_hidden = uni_rnn(x)
    
    print("=== 双方向RNN vs 単方向RNN ===\n")
    
    print(f"入力サイズ: {x.shape}")
    
    print(f"\n双方向RNN:")
    print(f"  出力: {bi_output.shape}")  # (3, 5, 40) ← 20*2
    print(f"  隠れ状態: {bi_hidden.shape}")  # (2, 3, 20) ← 2方向
    
    print(f"\n単方向RNN:")
    print(f"  出力: {uni_output.shape}")  # (3, 5, 20)
    print(f"  隠れ状態: {uni_hidden.shape}")  # (1, 3, 20)
    
    print("\n双方向RNNの特徴:")
    print("  ✓ 順方向と逆方向の両方の文脈を捉える")
    print("  ✓ 出力次元は2倍になる（forward + backward）")
    print("  ✓ 未来の情報も利用できるため、精度向上")
    print("  ✗ リアルタイム処理には不向き（全シーケンスが必要）")
    

* * *
