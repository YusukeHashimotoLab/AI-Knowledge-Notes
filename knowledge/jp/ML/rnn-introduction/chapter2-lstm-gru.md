---
title: 第2章：LSTM・GRU（Long Short-Term Memory and Gated Recurrent Unit）
chapter_title: 第2章：LSTM・GRU（Long Short-Term Memory and Gated Recurrent Unit）
subtitle: 長期依存関係を扱うゲート付きRNNアーキテクチャの理論と実装
reading_time: 25-30分
difficulty: 中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Vanilla RNNの限界（勾配消失・爆発問題）を理解する
  * ✅ LSTMのセルステートとゲート機構（忘却・入力・出力ゲート）を説明できる
  * ✅ GRUのアーキテクチャとLSTMとの違いを理解する
  * ✅ PyTorchでLSTM・GRUを実装し、実際の問題に適用できる
  * ✅ 双方向RNN（Bidirectional RNN）の仕組みと利点を理解する
  * ✅ IMDb感情分析タスクで実践的なLSTMモデルを構築できる

* * *

## 2.1 Vanilla RNNの限界

### 勾配消失・爆発問題

第1章で学んだ標準的なRNN（Vanilla RNN）は、理論上は任意の長さの系列を扱えますが、実際には**長期依存関係** の学習が困難です。

RNNのBPTT（Backpropagation Through Time）では、時刻$t$での勾配は以下のように時間をさかのぼって伝播します：

$$ \frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} $$ 

この積の項が問題です：

  * **勾配消失** ：$\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| < 1$ の場合、時間ステップが増えると勾配が指数的に減少
  * **勾配爆発** ：$\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| > 1$ の場合、勾配が指数的に増大

> 「勾配消失により、Vanilla RNNは10ステップ以上の長期依存関係をほとんど学習できません」

### 問題の可視化
    
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Vanilla RNNで勾配の大きさを観察
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
        def forward(self, x):
            output, hidden = self.rnn(x)
            return output, hidden
    
    # モデル作成
    input_size = 10
    hidden_size = 20
    sequence_length = 50
    
    model = SimpleRNN(input_size, hidden_size)
    
    # ランダムな入力
    x = torch.randn(1, sequence_length, input_size, requires_grad=True)
    
    # Forward pass
    output, hidden = model(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # 各時刻の勾配ノルムを計算
    gradients = []
    for t in range(sequence_length):
        grad = x.grad[0, t, :].norm().item()
        gradients.append(grad)
    
    print("=== Vanilla RNNの勾配伝播 ===")
    print(f"初期時刻の勾配ノルム: {gradients[0]:.6f}")
    print(f"中間時刻の勾配ノルム: {gradients[25]:.6f}")
    print(f"最終時刻の勾配ノルム: {gradients[49]:.6f}")
    print(f"\n勾配の減衰率: {gradients[0] / gradients[49]:.2f}倍")
    print("→ 過去にさかのぼるほど勾配が小さくなる（勾配消失）")
    

### 具体例：長期依存タスク
    
    
    import torch
    import torch.nn as nn
    
    # タスク: 系列の最初の要素を最後に予測する
    def create_long_dependency_task(batch_size=32, seq_length=50):
        """
        最初の時刻に重要な情報があり、最後でそれを使う必要があるタスク
        例: [5, 0, 0, ..., 0] → 最後に5を予測
        """
        x = torch.zeros(batch_size, seq_length, 10)
        targets = torch.randint(0, 10, (batch_size,))
    
        # 最初の時刻に正解ラベルをエンコード
        for i in range(batch_size):
            x[i, 0, targets[i]] = 1.0
    
        return x, targets
    
    # Vanilla RNNで学習
    class VanillaRNNClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(VanillaRNNClassifier, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            output, hidden = self.rnn(x)
            # 最後の時刻の出力を使用
            logits = self.fc(output[:, -1, :])
            return logits
    
    # 実験
    model = VanillaRNNClassifier(input_size=10, hidden_size=32, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練
    num_epochs = 100
    for epoch in range(num_epochs):
        x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
                print(f"Epoch {epoch+1}: 精度 = {accuracy*100:.2f}%")
    
    print("\n→ Vanilla RNNは長期依存関係を学習できず、精度が低い（ランダム予測と同程度）")
    

### 解決策：ゲート機構

この問題を解決するために、**LSTM（Long Short-Term Memory）** と**GRU（Gated Recurrent Unit）** が提案されました。これらは**ゲート機構** により、情報の流れを制御し、長期依存関係を効果的に学習できます。

* * *

## 2.2 LSTM (Long Short-Term Memory)

### LSTMの概要

**LSTM** は、1997年にHochreiterとSchmidhuberによって提案された、ゲート付きRNNアーキテクチャです。Vanilla RNNの隠れ状態に加えて、**セルステート（Cell State）** という長期記憶を持つことが特徴です。
    
    
    ```mermaid
    graph LR
        A["入力 x_t"] --> B["LSTM Cell"]
        C["前の隠れ状態 h_{t-1}"] --> B
        D["前のセルステート c_{t-1}"] --> B
    
        B --> E["出力 h_t"]
        B --> F["新しいセルステート c_t"]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style D fill:#fff9c4
        style E fill:#4fc3f7
        style F fill:#ffeb3b
    ```

### LSTMの4つのコンポーネント

LSTMセルは、以下の4つの要素で構成されます：

ゲート | 役割 | 数式  
---|---|---  
**忘却ゲート**  
(Forget Gate) | 過去の記憶をどれだけ保持するか | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$  
**入力ゲート**  
(Input Gate) | 新しい情報をどれだけ追加するか | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$  
**候補値**  
(Candidate Cell) | 追加する新しい情報の内容 | $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$  
**出力ゲート**  
(Output Gate) | セルステートからどれだけ出力するか | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$  
  
### LSTMの数学的定義

LSTMの完全な更新式は以下の通りです：

$$ \begin{align} f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad &\text{(忘却ゲート)} \\\ i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad &\text{(入力ゲート)} \\\ \tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \quad &\text{(候補値)} \\\ C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &\text{(セルステート更新)} \\\ o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad &\text{(出力ゲート)} \\\ h_t &= o_t \odot \tanh(C_t) \quad &\text{(隠れ状態更新)} \end{align} $$ 

ここで：

  * $\sigma$：シグモイド関数（0〜1の値を出力、ゲートの制御に使用）
  * $\odot$：要素ごとの積（Hadamard積）
  * $[h_{t-1}, x_t]$：ベクトルの連結
  * $W_*, b_*$：学習可能なパラメータ

### セルステートの役割

セルステート$C_t$は、情報のハイウェイとして機能します：

$$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$ 

  * $f_t \odot C_{t-1}$：過去の記憶を選択的に保持（忘却ゲートで制御）
  * $i_t \odot \tilde{C}_t$：新しい情報を選択的に追加（入力ゲートで制御）

> 「勾配はセルステートを通じて直接流れるため、勾配消失が起こりにくい！」

### LSTMの手動実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LSTMCell(nn.Module):
        """LSTMセルの手動実装（教育目的）"""
        def __init__(self, input_size, hidden_size):
            super(LSTMCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
    
            # 忘却ゲート
            self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
            # 入力ゲート
            self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
            # 候補値
            self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
            # 出力ゲート
            self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
    
        def forward(self, x_t, h_prev, C_prev):
            """
            x_t: (batch_size, input_size) - 現在の入力
            h_prev: (batch_size, hidden_size) - 前の隠れ状態
            C_prev: (batch_size, hidden_size) - 前のセルステート
            """
            # 入力と隠れ状態を連結
            combined = torch.cat([h_prev, x_t], dim=1)
    
            # 忘却ゲート: どの情報を忘れるか
            f_t = torch.sigmoid(self.W_f(combined))
    
            # 入力ゲート: どの情報を追加するか
            i_t = torch.sigmoid(self.W_i(combined))
    
            # 候補値: 追加する情報の内容
            C_tilde = torch.tanh(self.W_C(combined))
    
            # セルステート更新
            C_t = f_t * C_prev + i_t * C_tilde
    
            # 出力ゲート: どの情報を出力するか
            o_t = torch.sigmoid(self.W_o(combined))
    
            # 隠れ状態更新
            h_t = o_t * torch.tanh(C_t)
    
            return h_t, C_t
    
    
    class ManualLSTM(nn.Module):
        """複数時刻のLSTM処理"""
        def __init__(self, input_size, hidden_size):
            super(ManualLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.cell = LSTMCell(input_size, hidden_size)
    
        def forward(self, x, init_states=None):
            """
            x: (batch_size, seq_length, input_size)
            """
            batch_size, seq_length, _ = x.size()
    
            # 初期状態
            if init_states is None:
                h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
                C_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            else:
                h_t, C_t = init_states
    
            # 各時刻の出力を保存
            outputs = []
    
            # 系列を時刻ごとに処理
            for t in range(seq_length):
                h_t, C_t = self.cell(x[:, t, :], h_t, C_t)
                outputs.append(h_t.unsqueeze(1))
    
            # 出力を連結
            outputs = torch.cat(outputs, dim=1)
    
            return outputs, (h_t, C_t)
    
    
    # 動作確認
    batch_size = 4
    seq_length = 10
    input_size = 8
    hidden_size = 16
    
    model = ManualLSTM(input_size, hidden_size)
    x = torch.randn(batch_size, seq_length, input_size)
    
    outputs, (h_final, C_final) = model(x)
    
    print("=== 手動実装LSTMの動作確認 ===")
    print(f"入力サイズ: {x.shape}")
    print(f"出力サイズ: {outputs.shape}")
    print(f"最終隠れ状態: {h_final.shape}")
    print(f"最終セルステート: {C_final.shape}")
    

### PyTorchのnn.LSTMを使う

実際の開発では、PyTorchの最適化された`nn.LSTM`を使用します：
    
    
    import torch
    import torch.nn as nn
    
    # PyTorchのLSTM
    lstm = nn.LSTM(
        input_size=10,      # 入力の次元数
        hidden_size=20,     # 隠れ状態の次元数
        num_layers=2,       # LSTMの層数
        batch_first=True,   # (batch, seq, feature)の順序
        dropout=0.2,        # 層間のDropout
        bidirectional=False # 双方向かどうか
    )
    
    # ダミーデータ
    batch_size = 32
    seq_length = 15
    input_size = 10
    
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    print("=== PyTorch nn.LSTMの使用 ===")
    print(f"入力: {x.shape}")
    print(f"出力: {output.shape}  # (batch, seq, hidden_size)")
    print(f"最終隠れ状態: {h_n.shape}  # (num_layers, batch, hidden_size)")
    print(f"最終セルステート: {c_n.shape}  # (num_layers, batch, hidden_size)")
    
    # パラメータ数の確認
    total_params = sum(p.numel() for p in lstm.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    print("→ 各層に4つのゲート（f, i, C, o）があるため、パラメータが多い")
    

### LSTMと長期依存関係
    
    
    import torch
    import torch.nn as nn
    
    # 先ほどの長期依存タスクをLSTMで解く
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            # 最後の時刻の出力を使用
            logits = self.fc(output[:, -1, :])
            return logits
    
    # タスク作成関数（前と同じ）
    def create_long_dependency_task(batch_size=32, seq_length=50):
        x = torch.zeros(batch_size, seq_length, 10)
        targets = torch.randint(0, 10, (batch_size,))
        for i in range(batch_size):
            x[i, 0, targets[i]] = 1.0
        return x, targets
    
    # LSTMで訓練
    model = LSTMClassifier(input_size=10, hidden_size=32, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=== LSTMで長期依存関係を学習 ===")
    num_epochs = 100
    for epoch in range(num_epochs):
        x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
                print(f"Epoch {epoch+1}: 精度 = {accuracy*100:.2f}%")
    
    print("\n→ LSTMは長期依存関係を効果的に学習でき、高精度を達成！")
    

* * *

## 2.3 GRU (Gated Recurrent Unit)

### GRUの概要

**GRU（Gated Recurrent Unit）** は、2014年にChoらによって提案された、LSTMを簡略化したアーキテクチャです。LSTMより少ないパラメータで、同等以上の性能を発揮することが多いです。

### LSTMとGRUの違い

項目 | LSTM | GRU  
---|---|---  
**ゲート数** | 3つ（忘却、入力、出力） | 2つ（リセット、更新）  
**状態** | 隠れ状態$h_t$とセルステート$C_t$ | 隠れ状態$h_t$のみ  
**パラメータ数** | 多い | 少ない（LSTMの約75%）  
**計算速度** | やや遅い | やや速い  
**性能** | タスクによる | タスクによる（短い系列では有利）  
  
### GRUの数学的定義

GRUの更新式は以下の通りです：

$$ \begin{align} r_t &= \sigma(W_r [h_{t-1}, x_t] + b_r) \quad &\text{(リセットゲート)} \\\ z_t &= \sigma(W_z [h_{t-1}, x_t] + b_z) \quad &\text{(更新ゲート)} \\\ \tilde{h}_t &= \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \quad &\text{(候補隠れ状態)} \\\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{(隠れ状態更新)} \end{align} $$ 

各ゲートの役割：

  * **リセットゲート$r_t$** ：過去の情報をどれだけ無視するか（0に近いと過去を無視）
  * **更新ゲート$z_t$** ：過去と現在の情報をどの程度混ぜるか（0に近いと過去を保持、1に近いと新情報を採用）

> 「GRUは更新ゲート$z_t$で、LSTMの忘却ゲートと入力ゲートを統合しています」

### GRUの手動実装
    
    
    import torch
    import torch.nn as nn
    
    class GRUCell(nn.Module):
        """GRUセルの手動実装"""
        def __init__(self, input_size, hidden_size):
            super(GRUCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
    
            # リセットゲート
            self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
            # 更新ゲート
            self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
            # 候補隠れ状態
            self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
        def forward(self, x_t, h_prev):
            """
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            """
            # 入力と隠れ状態を連結
            combined = torch.cat([h_prev, x_t], dim=1)
    
            # リセットゲート
            r_t = torch.sigmoid(self.W_r(combined))
    
            # 更新ゲート
            z_t = torch.sigmoid(self.W_z(combined))
    
            # 候補隠れ状態（リセットゲートで過去をフィルタリング）
            combined_reset = torch.cat([r_t * h_prev, x_t], dim=1)
            h_tilde = torch.tanh(self.W_h(combined_reset))
    
            # 隠れ状態更新（更新ゲートで過去と現在を混合）
            h_t = (1 - z_t) * h_prev + z_t * h_tilde
    
            return h_t
    
    
    class ManualGRU(nn.Module):
        """複数時刻のGRU処理"""
        def __init__(self, input_size, hidden_size):
            super(ManualGRU, self).__init__()
            self.hidden_size = hidden_size
            self.cell = GRUCell(input_size, hidden_size)
    
        def forward(self, x, init_state=None):
            batch_size, seq_length, _ = x.size()
    
            # 初期状態
            if init_state is None:
                h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            else:
                h_t = init_state
    
            outputs = []
    
            for t in range(seq_length):
                h_t = self.cell(x[:, t, :], h_t)
                outputs.append(h_t.unsqueeze(1))
    
            outputs = torch.cat(outputs, dim=1)
            return outputs, h_t
    
    
    # 動作確認
    model = ManualGRU(input_size=8, hidden_size=16)
    x = torch.randn(4, 10, 8)
    
    outputs, h_final = model(x)
    
    print("=== 手動実装GRUの動作確認 ===")
    print(f"入力: {x.shape}")
    print(f"出力: {outputs.shape}")
    print(f"最終隠れ状態: {h_final.shape}")
    print("→ GRUはセルステートがなく、隠れ状態のみ")
    

### PyTorchのnn.GRUを使う
    
    
    import torch
    import torch.nn as nn
    
    # PyTorchのGRU
    gru = nn.GRU(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
        bidirectional=False
    )
    
    x = torch.randn(32, 15, 10)
    output, h_n = gru(x)
    
    print("=== PyTorch nn.GRUの使用 ===")
    print(f"入力: {x.shape}")
    print(f"出力: {output.shape}")
    print(f"最終隠れ状態: {h_n.shape}")
    
    # LSTMとのパラメータ数比較
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    gru_params = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    
    print(f"\nGRU パラメータ数: {gru_params:,}")
    print(f"LSTM パラメータ数: {lstm_params:,}")
    print(f"差: {lstm_params - gru_params:,} (GRUの方が {(lstm_params/gru_params - 1)*100:.1f}% 少ない)")
    

### LSTMとGRUの性能比較
    
    
    import torch
    import torch.nn as nn
    import time
    
    class SequenceClassifier(nn.Module):
        """汎用系列分類器"""
        def __init__(self, input_size, hidden_size, num_classes, rnn_type='lstm'):
            super(SequenceClassifier, self).__init__()
    
            if rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            elif rnn_type == 'gru':
                self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            else:
                raise ValueError("rnn_type must be 'lstm' or 'gru'")
    
            self.fc = nn.Linear(hidden_size, num_classes)
            self.rnn_type = rnn_type
    
        def forward(self, x):
            if self.rnn_type == 'lstm':
                output, (h_n, c_n) = self.rnn(x)
            else:
                output, h_n = self.rnn(x)
    
            logits = self.fc(output[:, -1, :])
            return logits
    
    # 比較実験
    def compare_models(seq_length=50):
        print(f"\n=== 系列長={seq_length}での比較 ===")
    
        # モデル作成
        lstm_model = SequenceClassifier(10, 32, 10, rnn_type='lstm')
        gru_model = SequenceClassifier(10, 32, 10, rnn_type='gru')
    
        # データ生成
        x, targets = create_long_dependency_task(batch_size=32, seq_length=seq_length)
    
        criterion = nn.CrossEntropyLoss()
    
        # LSTM訓練
        optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        start = time.time()
        for _ in range(50):
            optimizer_lstm.zero_grad()
            logits = lstm_model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_lstm.step()
        lstm_time = time.time() - start
    
        # GRU訓練
        optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001)
        start = time.time()
        for _ in range(50):
            optimizer_gru.zero_grad()
            logits = gru_model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_gru.step()
        gru_time = time.time() - start
    
        # 精度評価
        x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=seq_length)
    
        with torch.no_grad():
            logits_lstm = lstm_model(x_test)
            logits_gru = gru_model(x_test)
    
            _, pred_lstm = logits_lstm.max(1)
            _, pred_gru = logits_gru.max(1)
    
            acc_lstm = (pred_lstm == targets_test).float().mean().item()
            acc_gru = (pred_gru == targets_test).float().mean().item()
    
        print(f"LSTM - 精度: {acc_lstm*100:.2f}%, 訓練時間: {lstm_time:.2f}秒")
        print(f"GRU  - 精度: {acc_gru*100:.2f}%, 訓練時間: {gru_time:.2f}秒")
    
    # 異なる系列長で比較
    compare_models(seq_length=20)
    compare_models(seq_length=50)
    compare_models(seq_length=100)
    
    print("\n→ 短い系列ではGRUが効率的、長い系列ではLSTMが有利な傾向")
    

* * *

## 2.4 双方向RNN (Bidirectional RNN)

### 双方向RNNとは

**双方向RNN（Bidirectional RNN）** は、系列を前から後ろ（順方向）と後ろから前（逆方向）の両方向から処理し、両方の情報を統合します。
    
    
    ```mermaid
    graph LR
        A["x_1"] --> B["順方向→"]
        B --> C["x_2"]
        C --> D["順方向→"]
        D --> E["x_3"]
    
        E --> F["逆方向←"]
        F --> C
        C --> G["逆方向←"]
        G --> A
    
        B --> H["h_1"]
        D --> I["h_2"]
        F --> J["h_3 (逆)"]
        G --> K["h_2 (逆)"]
    
        style B fill:#b3e5fc
        style D fill:#b3e5fc
        style F fill:#ffab91
        style G fill:#ffab91
    ```

### 双方向RNNの利点

  * **文脈の完全な把握** ：各位置で、前後両方の文脈を考慮できる
  * **品詞タグ付け** ：単語の前後を見て品詞を決定
  * **感情分析** ：文全体を見て感情を判断
  * **機械翻訳** ：エンコーダとして使用

> 「双方向RNNは、時刻$t$での出力が未来の情報にも依存するため、リアルタイム処理には使えません。オフライン処理（全系列が利用可能）に適しています。」

### 双方向LSTMの実装
    
    
    import torch
    import torch.nn as nn
    
    # PyTorchでは bidirectional=True を指定するだけ
    class BidirectionalLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BidirectionalLSTM, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                bidirectional=True  # 双方向を有効化
            )
    
            # 双方向なので hidden_size * 2
            self.fc = nn.Linear(hidden_size * 2, num_classes)
    
        def forward(self, x):
            # output: (batch, seq, hidden_size * 2)
            output, (h_n, c_n) = self.lstm(x)
    
            # 最後の時刻の出力を使用
            logits = self.fc(output[:, -1, :])
            return logits
    
    # 動作確認
    model = BidirectionalLSTM(input_size=10, hidden_size=32, num_classes=10)
    x = torch.randn(4, 15, 10)
    
    logits = model(x)
    
    print("=== 双方向LSTMの動作確認 ===")
    print(f"入力: {x.shape}")
    print(f"出力: {logits.shape}")
    
    # パラメータ数の比較
    uni_lstm = nn.LSTM(10, 32, batch_first=True, bidirectional=False)
    bi_lstm = nn.LSTM(10, 32, batch_first=True, bidirectional=True)
    
    uni_params = sum(p.numel() for p in uni_lstm.parameters())
    bi_params = sum(p.numel() for p in bi_lstm.parameters())
    
    print(f"\n単方向LSTM: {uni_params:,} パラメータ")
    print(f"双方向LSTM: {bi_params:,} パラメータ")
    print(f"→ 双方向は約2倍のパラメータ数")
    

### 双方向vs単方向の性能比較
    
    
    import torch
    import torch.nn as nn
    
    class DirectionalClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, bidirectional=False):
            super(DirectionalClassifier, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            )
    
            fc_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Linear(fc_input_size, num_classes)
    
        def forward(self, x):
            output, _ = self.lstm(x)
            logits = self.fc(output[:, -1, :])
            return logits
    
    # 比較実験
    def compare_directionality():
        print("\n=== 単方向 vs 双方向の比較 ===")
    
        # モデル作成
        uni_model = DirectionalClassifier(10, 32, 10, bidirectional=False)
        bi_model = DirectionalClassifier(10, 32, 10, bidirectional=True)
    
        criterion = nn.CrossEntropyLoss()
    
        # 訓練
        for model, name in [(uni_model, "単方向"), (bi_model, "双方向")]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
            for epoch in range(100):
                x, targets = create_long_dependency_task(batch_size=32, seq_length=50)
    
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
    
            # 評価
            x_test, targets_test = create_long_dependency_task(batch_size=100, seq_length=50)
            with torch.no_grad():
                logits_test = model(x_test)
                _, predicted = logits_test.max(1)
                accuracy = (predicted == targets_test).float().mean().item()
    
            print(f"{name}LSTM - 精度: {accuracy*100:.2f}%")
    
    compare_directionality()
    print("\n→ このタスクでは情報が最初にあるため、双方向の優位性は小さい")
    print("  品詞タグ付けなど、前後文脈が重要なタスクでは双方向が有利")
    

* * *

## 2.5 実践：IMDb感情分析

### IMDbデータセット

**IMDb（Internet Movie Database）** は、映画レビューの感情分析データセットです：

  * 50,000件の映画レビュー（訓練25,000件、テスト25,000件）
  * 2クラス分類：肯定的（Positive）、否定的（Negative）
  * 各レビューは英語のテキスト

### データ準備
    
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from collections import Counter
    
    # トークナイザー
    tokenizer = get_tokenizer('basic_english')
    
    # データセット読み込み
    train_iter = IMDB(split='train')
    
    # 語彙構築
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    # 語彙を構築（頻度上位10,000語）
    vocab = build_vocab_from_iterator(
        yield_tokens(IMDB(split='train')),
        specials=['', ''],
        max_tokens=10000
    )
    vocab.set_default_index(vocab[''])
    
    print("=== IMDbデータセットの準備 ===")
    print(f"語彙サイズ: {len(vocab)}")
    print(f"トークンのインデックス: {vocab['']}")
    print(f"トークンのインデックス: {vocab['']}")
    
    # サンプルのトークン化
    sample_text = "This movie is great!"
    tokens = tokenizer(sample_text)
    indices = [vocab[token] for token in tokens]
    print(f"\nサンプル: '{sample_text}'")
    print(f"トークン: {tokens}")
    print(f"インデックス: {indices}")
    

### データセットクラス
    
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    
    class IMDbDataset(Dataset):
        def __init__(self, split='train'):
            self.data = list(IMDB(split=split))
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            label, text = self.data[idx]
    
            # ラベルを数値に変換（neg=0, pos=1）
            label = 1 if label == 'pos' else 0
    
            # テキストをトークン化してインデックスに変換
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
    
            return torch.tensor(indices), torch.tensor(label)
    
    def collate_batch(batch):
        """
        バッチ内の系列を同じ長さにパディング
        """
        texts, labels = zip(*batch)
    
        # パディング
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab[''])
        labels = torch.stack(labels)
    
        return texts_padded, labels
    
    # データローダー作成
    train_dataset = IMDbDataset(split='train')
    test_dataset = IMDbDataset(split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    print("\n=== データローダーの確認 ===")
    texts, labels = next(iter(train_loader))
    print(f"バッチサイズ: {texts.shape[0]}")
    print(f"系列長（最長）: {texts.shape[1]}")
    print(f"ラベル: {labels[:5]}")
    

### LSTM感情分析モデル
    
    
    import torch
    import torch.nn as nn
    
    class LSTMSentimentClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
            super(LSTMSentimentClassifier, self).__init__()
    
            # 埋め込み層
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab[''])
    
            # LSTM層
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
    
            # 分類層
            self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双方向なので *2
    
            # Dropout
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, text):
            # text: (batch, seq_len)
    
            # 埋め込み: (batch, seq_len, embedding_dim)
            embedded = self.dropout(self.embedding(text))
    
            # LSTM: output (batch, seq_len, hidden_dim * 2)
            output, (hidden, cell) = self.lstm(embedded)
    
            # 最後の時刻の出力を使用
            # または、順方向と逆方向の最終隠れ状態を連結
            # hidden: (num_layers * 2, batch, hidden_dim)
    
            # 最終層の順方向と逆方向を連結
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
    
            # Dropout + 分類
            hidden_concat = self.dropout(hidden_concat)
            logits = self.fc(hidden_concat)
    
            return logits
    
    # モデル作成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用デバイス: {device}")
    
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

### 訓練ループ
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
    
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
    
            loss.backward()
            # 勾配クリッピング（勾配爆発を防ぐ）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def test_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for texts, labels in loader:
                texts, labels = texts.to(device), labels.to(device)
    
                logits = model(texts)
                loss = criterion(logits, labels)
    
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    # 訓練実行
    num_epochs = 5
    best_acc = 0
    
    print("\n=== 訓練開始 ===")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_imdb_lstm.pth')
    
    print(f"\n訓練完了！ベスト精度: {best_acc:.2f}%")
    

### 推論と解釈
    
    
    import torch
    
    def predict_sentiment(model, text, vocab, tokenizer, device):
        """単一テキストの感情を予測"""
        model.eval()
    
        # トークン化
        tokens = tokenizer(text)
        indices = [vocab[token] for token in tokens]
    
        # テンソルに変換
        text_tensor = torch.tensor(indices).unsqueeze(0).to(device)  # (1, seq_len)
    
        # 予測
        with torch.no_grad():
            logits = model(text_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
    
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = probs[0, pred].item()
    
        return sentiment, confidence
    
    # テスト
    test_reviews = [
        "This movie is absolutely amazing! I loved every moment.",
        "Terrible film. Waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen!",
        "Boring and predictable. Would not recommend."
    ]
    
    print("\n=== 感情分析の予測結果 ===")
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(model, review, vocab, tokenizer, device)
        print(f"\nレビュー: {review}")
        print(f"予測: {sentiment} (信頼度: {confidence*100:.2f}%)")
    

* * *

## 2.6 LSTMとGRUの使い分けガイドライン

### 選択基準

状況 | 推奨 | 理由  
---|---|---  
長い系列（>100） | LSTM | セルステートで長期記憶を保持  
短い系列（<50） | GRU | パラメータが少なく効率的  
計算リソース制約 | GRU | パラメータ数が約25%少ない  
高精度が必須 | LSTM | 表現力が高い  
リアルタイム処理 | GRU | 計算が高速  
前後文脈が必要 | 双方向LSTM/GRU | 両方向の情報を活用  
不明な場合 | 両方試す | タスク依存性が高い  
  
### ハイパーパラメータの選び方

  * **隠れ層のサイズ** ：64〜512（タスクの複雑さに応じて）
  * **層数** ：1〜3層（深すぎると過学習のリスク）
  * **Dropout** ：0.2〜0.5（過学習を防ぐ）
  * **埋め込み次元** ：50〜300（語彙サイズに応じて）
  * **学習率** ：0.0001〜0.001（Adamが推奨）
  * **バッチサイズ** ：32〜128（メモリに応じて）

### 実装のベストプラクティス
    
    
    import torch
    import torch.nn as nn
    
    class BestPracticeLSTM(nn.Module):
        """ベストプラクティスを組み込んだLSTMモデル"""
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super(BestPracticeLSTM, self).__init__()
    
            # 1. Embedding層にpadding_idxを指定
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
            # 2. 双方向LSTM
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3,  # 層間Dropout
                bidirectional=True
            )
    
            # 3. Batch Normalization（オプション）
            self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
    
            # 4. Dropout
            self.dropout = nn.Dropout(0.5)
    
            # 5. 分類層
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            output, (hidden, cell) = self.lstm(embedded)
    
            # 順方向と逆方向の最終隠れ状態を連結
            hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
    
            # Batch Norm（オプション）
            hidden_concat = self.batch_norm(hidden_concat)
    
            # Dropout + 分類
            hidden_concat = self.dropout(hidden_concat)
            logits = self.fc(hidden_concat)
    
            return logits
    
    # 訓練時の注意点
    def train_with_best_practices(model, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        # 学習率スケジューラ
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    
        for epoch in range(num_epochs):
            model.train()
            for texts, labels in train_loader:
                optimizer.zero_grad()
                logits = model(texts)
                loss = criterion(logits, labels)
                loss.backward()
    
                # 勾配クリッピング（必須）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
                optimizer.step()
    
            # 検証ロスで学習率を調整
            val_loss = evaluate(model, val_loader)
            scheduler.step(val_loss)
    
    print("=== ベストプラクティス ===")
    print("1. padding_idxを指定してを学習対象外に")
    print("2. 双方向LSTMで文脈を完全把握")
    print("3. Dropoutで過学習を防止")
    print("4. 勾配クリッピングで勾配爆発を防止")
    print("5. 学習率スケジューラで最適化を改善")
    

* * *

## 演習問題

**演習1：LSTMのゲート動作を観察**

LSTMの各ゲート（忘却、入力、出力）の値を可視化し、どのように情報を制御しているか確認してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: LSTMCellを修正して、各ゲートの値を返すようにする
    # TODO: 簡単な系列データで各ゲートの値を時系列にプロット
    # ヒント: f_t, i_t, o_t の値を記録し、matplotlib でグラフ化
    

**演習2：GRUとLSTMの収束速度比較**

同じタスクでGRUとLSTMを訓練し、訓練曲線（損失と精度）を比較してください。
    
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    # TODO: GRUとLSTMモデルを作成
    # TODO: 同じデータで訓練し、各エポックの損失と精度を記録
    # TODO: 訓練曲線をプロット
    # 評価指標: 収束速度、最終精度、訓練時間
    

**演習3：双方向RNNの効果を検証**

品詞タグ付けタスクで、単方向と双方向RNNの性能を比較してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 単方向と双方向のLSTMモデルを実装
    # TODO: 品詞タグ付けタスク（各単語の品詞を予測）で性能比較
    # ヒント: torchtext.datasets のUD_English などを使用
    # 前後の文脈が重要なタスクで双方向の優位性を確認
    

**演習4：系列長と性能の関係**

異なる系列長（10, 50, 100, 200）でLSTMとGRUの性能を比較し、どちらが長期依存に強いか確認してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 系列長を変えて長期依存タスクを生成
    # TODO: LSTMとGRUで精度を比較
    # TODO: 系列長 vs 精度のグラフを作成
    # どの系列長から性能差が顕著になるか分析
    

**演習5：IMDb感情分析の改善**

基本のLSTMモデルを改善し、テスト精度を向上させてください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 以下の手法を試してモデルを改善
    # 1. 事前学習済み埋め込み（GloVe, Word2Vec）を使用
    # 2. Attention機構を追加
    # 3. 層数やhidden_sizeを調整
    # 4. Data Augmentation（バックトランスレーション等）
    # 5. アンサンブル学習
    
    # 目標: ベースラインから+2%以上の精度向上
    

* * *

## まとめ

この章では、LSTM・GRUとその応用について学びました。

### 重要ポイント

  * **Vanilla RNNの限界** ：勾配消失・爆発により長期依存関係の学習が困難
  * **LSTM** ：セルステートとゲート機構（忘却・入力・出力）で長期記憶を実現
  * **GRU** ：LSTMを簡略化、2つのゲート（リセット・更新）で効率的に動作
  * **LSTMとGRUの違い** ：パラメータ数、計算速度、性能のトレードオフ
  * **双方向RNN** ：前後両方向から処理し、文脈を完全に把握
  * **実践** ：IMDb感情分析で実際のNLPタスクに適用
  * **ベストプラクティス** ：勾配クリッピング、Dropout、学習率スケジューリング

### 次のステップ

次章では、**Sequence-to-Sequence（Seq2Seq）** と**Attention機構** について学びます。機械翻訳や要約などの系列変換タスクに必須の技術を習得します。
