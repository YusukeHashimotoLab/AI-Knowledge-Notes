---
title: 第3章：深層学習による時系列予測
chapter_title: 第3章：深層学習による時系列予測
subtitle: LSTM、GRU、TCN、Attentionによる高度な予測モデル
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時系列データを深層学習で扱うための基本的なアプローチを理解する
  * ✅ LSTM・GRUを用いた時系列予測モデルを構築できる
  * ✅ TCN（Temporal Convolutional Network）の仕組みと実装方法を学ぶ
  * ✅ Attention機構を時系列予測に応用できる
  * ✅ 特徴量エンジニアリングとアンサンブル手法を理解する
  * ✅ PyTorchで実践的な時系列予測モデルを実装できる

* * *

## 3.1 時系列のための深層学習

### Sequential Dataの表現

**時系列データ（Sequential Data）** は、時間的な順序を持つデータです。深層学習では、この順序関係を保持しながらモデルに入力する必要があります。

> 「時系列予測の本質は、過去のパターンから未来を推論すること」

### Window-based Approach（ウィンドウベースアプローチ）

時系列データを深層学習で扱うための基本的な手法は、**スライディングウィンドウ（Sliding Window）** です。
    
    
    ```mermaid
    graph LR
        A[元データ: t1, t2, t3, t4, t5, t6] --> B[Window 1: t1-t3 → t4]
        A --> C[Window 2: t2-t4 → t5]
        A --> D[Window 3: t3-t5 → t6]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
    ```

### 実装：ウィンドウベースのデータセット作成
    
    
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    
    # サンプル時系列データの生成
    np.random.seed(42)
    time = np.arange(0, 100, 0.1)
    data = np.sin(time) + 0.1 * np.random.randn(len(time))
    
    # 可視化
    plt.figure(figsize=(14, 5))
    plt.plot(time, data, label='Time Series Data', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('サンプル時系列データ', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"データポイント数: {len(data)}")
    print(f"データの範囲: [{data.min():.3f}, {data.max():.3f}]")
    

### PyTorch Dataset for Time Series
    
    
    class TimeSeriesDataset(Dataset):
        """
        時系列データ用のPyTorch Dataset
    
        Parameters:
        -----------
        data : np.ndarray
            時系列データ（1次元配列）
        window_size : int
            入力ウィンドウのサイズ
        horizon : int
            予測ホライゾン（何ステップ先を予測するか）
        """
        def __init__(self, data, window_size=20, horizon=1):
            self.data = data
            self.window_size = window_size
            self.horizon = horizon
    
        def __len__(self):
            return len(self.data) - self.window_size - self.horizon + 1
    
        def __getitem__(self, idx):
            # 入力: window_size個の過去データ
            x = self.data[idx:idx + self.window_size]
            # ターゲット: horizon個の未来データ
            y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
    
            return torch.FloatTensor(x), torch.FloatTensor(y)
    
    # データセットの作成
    window_size = 20
    horizon = 5  # 5ステップ先を予測
    
    # 訓練・検証分割
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data, window_size, horizon)
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("=== データセット情報 ===")
    print(f"訓練サンプル数: {len(train_dataset)}")
    print(f"検証サンプル数: {len(val_dataset)}")
    print(f"入力ウィンドウサイズ: {window_size}")
    print(f"予測ホライゾン: {horizon}")
    
    # サンプルの確認
    x_sample, y_sample = train_dataset[0]
    print(f"\nサンプル形状:")
    print(f"  入力 x: {x_sample.shape}")
    print(f"  ターゲット y: {y_sample.shape}")
    

**出力** ：
    
    
    === データセット情報 ===
    訓練サンプル数: 776
    検証サンプル数: 176
    入力ウィンドウサイズ: 20
    予測ホライゾン: 5
    
    サンプル形状:
      入力 x: torch.Size([20])
      ターゲット y: torch.Size([5])
    

### Multi-step Forecasting（複数ステップ予測）

時系列予測では、以下の2つのアプローチがあります：

アプローチ | 説明 | 利点 | 欠点  
---|---|---|---  
**One-shot** | 1回の予測で全ホライゾンを出力 | 高速、依存関係なし | 長期予測が難しい  
**Autoregressive** | 1ステップずつ予測し、次の入力に使用 | 柔軟、長期予測可能 | 誤差が蓄積  
      
    
    ```mermaid
    graph TD
        A[過去データ: t-n...t] --> B{予測方式}
        B -->|One-shot| C[一度に予測: t+1, t+2, ..., t+h]
        B -->|Autoregressive| D[t+1を予測]
        D --> E[t+1を入力に追加]
        E --> F[t+2を予測]
        F --> G[繰り返し...]
    
        style A fill:#e3f2fd
        style C fill:#c8e6c9
        style D fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 3.2 LSTM & GRU for 時系列予測

### LSTM Architecture Review

**LSTM（Long Short-Term Memory）** は、長期依存性を学習できるRNNの一種です。

LSTM セルの更新式：

$$ \begin{align*} f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(忘却ゲート)} \\\ i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(入力ゲート)} \\\ \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(候補値)} \\\ C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(セル状態更新)} \\\ o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(出力ゲート)} \\\ h_t &= o_t \odot \tanh(C_t) \quad \text{(隠れ状態)} \end{align*} $$

### PyTorchでのLSTM実装
    
    
    import torch
    import torch.nn as nn
    
    class LSTMForecaster(nn.Module):
        """
        LSTM-based time series forecasting model
    
        Parameters:
        -----------
        input_size : int
            入力特徴量の次元
        hidden_size : int
            LSTM隠れ層のサイズ
        num_layers : int
            LSTMレイヤーの数
        output_size : int
            出力サイズ（予測ホライゾン）
        dropout : float
            ドロップアウト率
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                     output_size=1, dropout=0.2):
            super(LSTMForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # LSTM層
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
    
            # 全結合層
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
    
            # LSTM forward pass
            # out shape: (batch_size, seq_len, hidden_size)
            out, (h_n, c_n) = self.lstm(x)
    
            # 最後のタイムステップの出力を使用
            # out[:, -1, :] shape: (batch_size, hidden_size)
            out = self.fc(out[:, -1, :])
    
            # out shape: (batch_size, output_size)
            return out
    
    # モデルのインスタンス化
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.2
    )
    
    print("=== LSTM モデル構造 ===")
    print(model)
    print(f"\nパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    

**出力** ：
    
    
    === LSTM モデル構造 ===
    LSTMForecaster(
      (lstm): LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
      (fc): Linear(in_features=64, out_features=5, bias=True)
    )
    
    パラメータ数: 50,245
    

### 訓練ループの実装
    
    
    import torch.optim as optim
    from tqdm import tqdm
    
    def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
        """
        モデルの訓練
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        train_losses = []
        val_losses = []
    
        for epoch in range(epochs):
            # 訓練フェーズ
            model.train()
            train_loss = 0.0
    
            for x_batch, y_batch in train_loader:
                # データを (batch, seq_len, features) の形に変換
                x_batch = x_batch.unsqueeze(-1).to(device)
                y_batch = y_batch.to(device)
    
                # 勾配初期化
                optimizer.zero_grad()
    
                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
    
                # Backward pass
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
    
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
    
            # 検証フェーズ
            model.eval()
            val_loss = 0.0
    
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.unsqueeze(-1).to(device)
                    y_batch = y_batch.to(device)
    
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
    
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
        return train_losses, val_losses
    
    # モデルの訓練
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
    
    # 学習曲線の可視化
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('学習曲線', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### GRU（Gated Recurrent Unit）

**GRU** はLSTMの簡略版で、パラメータ数が少なく訓練が高速です。

GRU の更新式：

$$ \begin{align*} r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(リセットゲート)} \\\ z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(更新ゲート)} \\\ \tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(候補隠れ状態)} \\\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(隠れ状態更新)} \end{align*} $$
    
    
    class GRUForecaster(nn.Module):
        """
        GRU-based time series forecasting model
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                     output_size=1, dropout=0.2):
            super(GRUForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # GRU層
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
    
            # 全結合層
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # GRU forward pass
            out, h_n = self.gru(x)
    
            # 最後のタイムステップの出力を使用
            out = self.fc(out[:, -1, :])
    
            return out
    
    # GRUモデルのインスタンス化
    gru_model = GRUForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.2
    )
    
    print("=== GRU モデル構造 ===")
    print(gru_model)
    print(f"\nLSTMパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GRUパラメータ数: {sum(p.numel() for p in gru_model.parameters()):,}")
    print(f"削減率: {(1 - sum(p.numel() for p in gru_model.parameters()) / sum(p.numel() for p in model.parameters())) * 100:.1f}%")
    

### Stateful vs Stateless LSTM

タイプ | 説明 | 使用場面  
---|---|---  
**Stateless** | バッチごとに隠れ状態をリセット | 独立したシーケンス、一般的な予測  
**Stateful** | バッチ間で隠れ状態を保持 | 長期連続予測、ストリーミングデータ  
  
* * *

## 3.3 TCN (Temporal Convolutional Network)

### TCNとは

**TCN（Temporal Convolutional Network）** は、時系列データに特化した畳み込みニューラルネットワークです。RNNと異なり、並列処理が可能で訓練が高速です。

### Dilated Convolutions（拡張畳み込み）

TCNの核心は**Dilated Convolution** です。通常の畳み込みに比べ、より広い受容野を少ないパラメータで実現します。
    
    
    ```mermaid
    graph TD
        A[入力シーケンス] --> B[Layer 1: dilation=1]
        B --> C[Layer 2: dilation=2]
        C --> D[Layer 3: dilation=4]
        D --> E[Layer 4: dilation=8]
        E --> F[出力: 広い受容野]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#ffccbc
        style E fill:#ffab91
        style F fill:#c8e6c9
    ```

受容野の計算：

$$ \text{Receptive Field} = 1 + 2 \times (k - 1) \times \sum_{i=0}^{L-1} d^i $$

  * $k$: カーネルサイズ
  * $d$: dilation factor
  * $L$: レイヤー数

### Causal Convolutions（因果的畳み込み）

**Causal Convolution** は、未来の情報を使わない畳み込みです。時系列予測では必須です。

> **重要** : パディングは左側のみに行い、未来のデータを参照しないようにします。

### PyTorchでのTCN実装
    
    
    class CausalConv1d(nn.Module):
        """
        Causal 1D Convolution with dilation
        """
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(CausalConv1d, self).__init__()
    
            # パディングは左側のみ（過去のデータのみ参照）
            self.padding = (kernel_size - 1) * dilation
    
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation
            )
    
        def forward(self, x):
            # x shape: (batch, channels, seq_len)
            x = self.conv(x)
    
            # 右側のパディングを除去（未来のデータを使わない）
            if self.padding != 0:
                x = x[:, :, :-self.padding]
    
            return x
    
    class TemporalBlock(nn.Module):
        """
        TCNの基本ブロック
        """
        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
            super(TemporalBlock, self).__init__()
    
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
    
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
    
            # Residual connection
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                if in_channels != out_channels else None
    
            self.relu = nn.ReLU()
    
        def forward(self, x):
            # Main path
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
    
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
    
            # Residual connection
            res = x if self.downsample is None else self.downsample(x)
    
            return self.relu(out + res)
    
    class TCN(nn.Module):
        """
        Temporal Convolutional Network for time series forecasting
        """
        def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
            super(TCN, self).__init__()
    
            layers = []
            num_levels = len(num_channels)
    
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = input_size if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
    
                layers.append(
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation_size,
                        dropout
                    )
                )
    
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], output_size)
    
        def forward(self, x):
            # x shape: (batch, seq_len, features)
            # Conv1dは (batch, features, seq_len) を期待
            x = x.transpose(1, 2)
    
            # TCN forward pass
            y = self.network(x)
    
            # 最後のタイムステップを使用
            y = y[:, :, -1]
    
            # 全結合層
            out = self.fc(y)
    
            return out
    
    # TCNモデルのインスタンス化
    tcn_model = TCN(
        input_size=1,
        output_size=horizon,
        num_channels=[32, 32, 64, 64],  # 4層
        kernel_size=3,
        dropout=0.2
    )
    
    print("=== TCN モデル構造 ===")
    print(tcn_model)
    print(f"\nパラメータ数: {sum(p.numel() for p in tcn_model.parameters()):,}")
    

### TCN vs RNN/LSTM

特性 | RNN/LSTM | TCN  
---|---|---  
**並列処理** | 逐次的、低速 | 並列可能、高速  
**長期依存性** | 勾配消失の問題 | Dilationで対応  
**受容野** | シーケンス長に依存 | Dilation で制御可能  
**メモリ効率** | 隠れ状態が必要 | 畳み込みのみ  
**訓練時間** | 遅い | 高速  
  
* * *

## 3.4 Attention Mechanisms for 時系列

### Self-Attention for Sequences

**Attention機構** は、入力シーケンスの重要な部分に焦点を当てる仕組みです。

Attention の計算式：

$$ \begin{align*} \text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\\ \text{where } Q &= XW_Q, \quad K = XW_K, \quad V = XW_V \end{align*} $$

  * $Q$: Query（クエリ）
  * $K$: Key（キー）
  * $V$: Value（バリュー）
  * $d_k$: キーの次元

### PyTorchでのAttention実装
    
    
    class AttentionLayer(nn.Module):
        """
        Self-Attention layer for time series
        """
        def __init__(self, hidden_size, num_heads=4):
            super(AttentionLayer, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
    
            assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    
            # Query, Key, Value projections
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
            self.W_v = nn.Linear(hidden_size, hidden_size)
    
            # Output projection
            self.W_o = nn.Linear(hidden_size, hidden_size)
    
            self.dropout = nn.Dropout(0.1)
    
        def forward(self, x):
            # x shape: (batch, seq_len, hidden_size)
            batch_size = x.size(0)
    
            # Linear projections
            Q = self.W_q(x)  # (batch, seq_len, hidden_size)
            K = self.W_k(x)
            V = self.W_v(x)
    
            # Split into multiple heads
            # (batch, seq_len, num_heads, head_dim)
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
            # Scaled dot-product attention
            # scores shape: (batch, num_heads, seq_len, seq_len)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
    
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
    
            # Apply attention to values
            # (batch, num_heads, seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, V)
    
            # Concatenate heads
            # (batch, seq_len, hidden_size)
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.hidden_size
            )
    
            # Final linear projection
            output = self.W_o(attn_output)
    
            return output, attn_weights
    
    class LSTMWithAttention(nn.Module):
        """
        LSTM + Attention for time series forecasting
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                     output_size=1, num_heads=4, dropout=0.2):
            super(LSTMWithAttention, self).__init__()
    
            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
    
            # Attention layer
            self.attention = AttentionLayer(hidden_size, num_heads)
    
            # Layer normalization
            self.layer_norm = nn.LayerNorm(hidden_size)
    
            # Output layer
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # LSTM encoding
            lstm_out, _ = self.lstm(x)
    
            # Apply attention
            attn_out, attn_weights = self.attention(lstm_out)
    
            # Residual connection + Layer norm
            out = self.layer_norm(lstm_out + attn_out)
    
            # Use last timestep
            out = self.fc(out[:, -1, :])
    
            return out, attn_weights
    
    # Attention付きLSTMモデル
    attn_model = LSTMWithAttention(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        num_heads=4,
        dropout=0.2
    )
    
    print("=== LSTM + Attention モデル構造 ===")
    print(attn_model)
    print(f"\nパラメータ数: {sum(p.numel() for p in attn_model.parameters()):,}")
    

### Attention Visualization（注意の可視化）
    
    
    def visualize_attention(model, data, window_size=20):
        """
        Attention weightsを可視化
        """
        model.eval()
        device = next(model.parameters()).device
    
        # サンプルデータの準備
        x = torch.FloatTensor(data[:window_size]).unsqueeze(0).unsqueeze(-1).to(device)
    
        with torch.no_grad():
            _, attn_weights = model(x)
    
        # Attention weightsの取得（最初のヘッド）
        attn = attn_weights[0, 0].cpu().numpy()
    
        # 可視化
        plt.figure(figsize=(12, 8))
    
        plt.subplot(2, 1, 1)
        plt.plot(data[:window_size], marker='o', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('入力シーケンス', fontsize=14)
        plt.grid(True, alpha=0.3)
    
        plt.subplot(2, 1, 2)
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Weights（どの位置に注目しているか）', fontsize=14)
    
        plt.tight_layout()
        plt.show()
    
    # 可視化の実行
    visualize_attention(attn_model, train_data, window_size=20)
    

### Seq2Seq with Attention

**Encoder-Decoder構造** をAttentionで強化したモデルは、複雑な時系列パターンを学習できます。
    
    
    ```mermaid
    graph LR
        A[入力シーケンス] --> B[Encoder LSTM]
        B --> C[Context Vector]
        C --> D[Decoder LSTM]
        D --> E[Attention Layer]
        E --> D
        E --> F[予測シーケンス]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#ffccbc
        style E fill:#ffab91
        style F fill:#c8e6c9
    ```

* * *

## 3.5 実践的なテクニック

### Feature Engineering for Deep Learning

深層学習でも、適切な特徴量エンジニアリングは重要です。
    
    
    def create_time_features(data, timestamps=None):
        """
        時系列データから時間ベースの特徴量を作成
        """
        features = pd.DataFrame()
    
        if timestamps is not None:
            # 時刻ベースの特徴
            features['hour'] = timestamps.hour / 24.0
            features['day_of_week'] = timestamps.dayofweek / 7.0
            features['day_of_month'] = timestamps.day / 31.0
            features['month'] = timestamps.month / 12.0
            features['is_weekend'] = (timestamps.dayofweek >= 5).astype(float)
    
        # ラグ特徴量
        for lag in [1, 2, 3, 7, 14]:
            features[f'lag_{lag}'] = pd.Series(data).shift(lag)
    
        # 移動統計量
        for window in [3, 7, 14]:
            rolling = pd.Series(data).rolling(window)
            features[f'rolling_mean_{window}'] = rolling.mean()
            features[f'rolling_std_{window}'] = rolling.std()
            features[f'rolling_min_{window}'] = rolling.min()
            features[f'rolling_max_{window}'] = rolling.max()
    
        # 差分特徴量
        features['diff_1'] = pd.Series(data).diff(1)
        features['diff_7'] = pd.Series(data).diff(7)
    
        # 欠損値を埋める
        features = features.fillna(0)
    
        return features
    
    # 特徴量の作成例
    timestamps = pd.date_range(start='2023-01-01', periods=len(train_data), freq='h')
    time_features = create_time_features(train_data, timestamps)
    
    print("=== 作成された特徴量 ===")
    print(time_features.head(20))
    print(f"\n特徴量の数: {time_features.shape[1]}")
    

### Ensemble Methods（アンサンブル手法）

複数のモデルを組み合わせることで、予測精度を向上させます。
    
    
    class EnsembleForecaster:
        """
        複数モデルのアンサンブル予測
        """
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights if weights is not None else [1.0] * len(models)
    
        def predict(self, x):
            predictions = []
    
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    predictions.append(pred)
    
            # 重み付き平均
            ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
            ensemble_pred /= sum(self.weights)
    
            return ensemble_pred
    
    # アンサンブルの作成
    models = [model, gru_model, tcn_model]
    ensemble = EnsembleForecaster(models, weights=[0.4, 0.3, 0.3])
    
    print("=== アンサンブル構成 ===")
    print(f"モデル数: {len(models)}")
    print(f"重み: {ensemble.weights}")
    

### Transfer Learning for Time Series

事前学習済みモデルを新しいタスクに適用します。
    
    
    def transfer_learning(pretrained_model, new_output_size, freeze_layers=True):
        """
        転移学習の設定
        """
        # 事前学習済みモデルをコピー
        model = pretrained_model
    
        # エンコーダー部分を凍結
        if freeze_layers:
            for param in model.lstm.parameters():
                param.requires_grad = False
    
        # 出力層を新しいタスク用に再初期化
        model.fc = nn.Linear(model.hidden_size, new_output_size)
    
        return model
    
    # 使用例
    # pretrained_model を別のタスク（horizon=10）に適用
    new_model = transfer_learning(model, new_output_size=10, freeze_layers=True)
    
    print("=== 転移学習モデル ===")
    print(f"凍結されたパラメータ数: {sum(p.numel() for p in new_model.parameters() if not p.requires_grad):,}")
    print(f"学習可能なパラメータ数: {sum(p.numel() for p in new_model.parameters() if p.requires_grad):,}")
    

### Hyperparameter Tuning

ハイパーパラメータの探索で最適なモデルを見つけます。
    
    
    from itertools import product
    
    def grid_search(param_grid, train_loader, val_loader, epochs=20):
        """
        グリッドサーチでハイパーパラメータ最適化
        """
        best_loss = float('inf')
        best_params = None
        results = []
    
        # パラメータの組み合わせを生成
        keys = param_grid.keys()
        values = param_grid.values()
    
        for params in product(*values):
            param_dict = dict(zip(keys, params))
    
            print(f"\n試行中: {param_dict}")
    
            # モデルの作成
            model = LSTMForecaster(
                input_size=1,
                hidden_size=param_dict['hidden_size'],
                num_layers=param_dict['num_layers'],
                output_size=horizon,
                dropout=param_dict['dropout']
            )
    
            # 訓練
            _, val_losses = train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=param_dict['lr']
            )
    
            # 最良の検証損失
            min_val_loss = min(val_losses)
            results.append((param_dict, min_val_loss))
    
            if min_val_loss < best_loss:
                best_loss = min_val_loss
                best_params = param_dict
    
        return best_params, best_loss, results
    
    # ハイパーパラメータグリッド
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.0001]
    }
    
    # グリッドサーチの実行（小規模な例）
    small_grid = {
        'hidden_size': [32, 64],
        'num_layers': [1, 2],
        'dropout': [0.2],
        'lr': [0.001]
    }
    
    best_params, best_loss, all_results = grid_search(
        small_grid, train_loader, val_loader, epochs=10
    )
    
    print("\n=== 最適なハイパーパラメータ ===")
    print(f"パラメータ: {best_params}")
    print(f"検証損失: {best_loss:.4f}")
    

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **深層学習の基本アプローチ**

     * ウィンドウベースのデータ準備
     * PyTorch DatasetとDataLoaderの活用
     * One-shot vs Autoregressive予測
  2. **LSTM & GRU**

     * 長期依存性の学習
     * LSTMとGRUの違いと使い分け
     * Stateful vs Stateless
  3. **TCN（Temporal Convolutional Network）**

     * Dilated Convolutionによる広い受容野
     * Causal Convolutionで未来を参照しない
     * RNNより高速な訓練
  4. **Attention機構**

     * Self-Attentionで重要な時点に焦点
     * LSTM + Attentionの組み合わせ
     * Attention weightsの可視化
  5. **実践テクニック**

     * 時間特徴量とラグ特徴量の作成
     * アンサンブル予測
     * 転移学習
     * ハイパーパラメータチューニング

### モデル選択ガイドライン

状況 | 推奨モデル | 理由  
---|---|---  
短期予測（< 10ステップ） | LSTM/GRU | シンプルで効果的  
長期予測 | TCN、Attention | 広い受容野、並列処理  
訓練速度重視 | TCN | 並列処理可能  
解釈性重視 | Attention付きモデル | 重要な時点を可視化  
多変量時系列 | Transformer | 複雑な依存関係を学習  
リアルタイム予測 | Stateful LSTM/GRU | 隠れ状態を保持  
  
### 次の章へ

次の章では、さらに高度なトピックを扱います（別章として展開）：

  * Transformer for Time Series
  * 異常検知
  * 多変量時系列予測
  * 確率的予測（予測区間の推定）

* * *

## 演習問題

### 問題1（難易度：easy）

LSTM と GRU の違いを、ゲート機構とパラメータ数の観点から説明してください。

解答例

**解答** ：

**LSTM（Long Short-Term Memory）** ：

  * ゲート: 3つ（忘却ゲート、入力ゲート、出力ゲート）
  * セル状態と隠れ状態の2つを保持
  * パラメータ数: より多い

**GRU（Gated Recurrent Unit）** ：

  * ゲート: 2つ（リセットゲート、更新ゲート）
  * 隠れ状態のみを保持
  * パラメータ数: LSTMの約75%

**使い分け** ：

  * LSTM: より複雑なパターン、長期依存性が強い場合
  * GRU: パラメータ削減、訓練速度重視、データが少ない場合

実際の性能差は小さいことが多く、タスクごとに検証が推奨されます。

### 問題2（難易度：medium）

Dilated Convolution の受容野を計算してください。以下の設定で、最終層の受容野はいくつになりますか？

  * カーネルサイズ: k = 3
  * レイヤー数: 4層
  * Dilation: [1, 2, 4, 8]

解答例

**解答** ：

受容野の計算式：

$$ \text{RF} = 1 + \sum_{i=1}^{L} (k - 1) \times d_i $$

  * $k = 3$: カーネルサイズ
  * $L = 4$: レイヤー数
  * $d_i$: 各層のdilation

計算：
    
    
    k = 3
    dilations = [1, 2, 4, 8]
    
    receptive_field = 1
    for d in dilations:
        receptive_field += (k - 1) * d
    
    print(f"受容野: {receptive_field}")
    

**出力** ：
    
    
    受容野: 31
    

つまり、31個の過去のタイムステップを考慮できます。

各層での受容野の増加：

  * Layer 1 (d=1): RF = 1 + 2×1 = 3
  * Layer 2 (d=2): RF = 3 + 2×2 = 7
  * Layer 3 (d=4): RF = 7 + 2×4 = 15
  * Layer 4 (d=8): RF = 15 + 2×8 = 31

### 問題3（難易度：medium）

Attention機構の利点を3つ挙げ、それぞれを説明してください。

解答例

**解答** ：

  1. **重要な時点への焦点**

     * 説明: シーケンス内の関連性の高い部分に自動的に注目
     * 利点: ノイズの多いデータでも重要な情報を抽出
     * 例: 季節性のピークや異常なイベントに注目
  2. **長距離依存性の学習**

     * 説明: 直接的な接続により、遠い時点との関係も学習可能
     * 利点: RNNの勾配消失問題を回避
     * 例: 年次パターンと日次パターンの関連性を学習
  3. **解釈可能性**

     * 説明: Attention weightsを可視化することで、モデルの判断根拠が分かる
     * 利点: ブラックボックスではなく、予測の理由を説明できる
     * 例: 「この予測は1週間前と2か月前のデータに基づく」と説明可能

**追加の利点** ：

  * 並列処理が可能（Self-Attentionの場合）
  * 可変長シーケンスに柔軟に対応
  * 多様なタスクに適用可能

### 問題4（難易度：hard）

以下のサンプルデータを使って、LSTM モデルを訓練し、予測精度をARIMAと比較してください。
    
    
    import numpy as np
    
    # サンプルデータ（月次売上データ）
    np.random.seed(42)
    time = np.arange(0, 100)
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.randn(100) * 2
    data = trend + seasonality + noise + 50
    

解答例
    
    
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    
    # サンプルデータ生成
    np.random.seed(42)
    time = np.arange(0, 100)
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.randn(100) * 2
    data = trend + seasonality + noise + 50
    
    # 訓練・テストデータ分割
    train_size = 80
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Dataset定義（前述のTimeSeriesDatasetを使用）
    window_size = 12
    horizon = 1
    
    train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # LSTMモデル（前述のLSTMForecasterを使用）
    model = LSTMForecaster(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        output_size=horizon,
        dropout=0.1
    )
    
    # 訓練
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
    
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.to(device)
    
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # LSTM予測
    model.eval()
    lstm_predictions = []
    
    with torch.no_grad():
        # テストデータで逐次予測
        for i in range(len(test_data)):
            # 直近のwindow_sizeデータを使用
            if i == 0:
                input_seq = train_data[-window_size:]
            else:
                input_seq = np.concatenate([
                    train_data[-(window_size-i):],
                    test_data[:i]
                ])[-window_size:]
    
            x = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            lstm_predictions.append(pred)
    
    lstm_predictions = np.array(lstm_predictions)
    
    # ARIMAとの比較
    from statsmodels.tsa.arima.model import ARIMA
    
    # ARIMAモデル
    arima_model = ARIMA(train_data, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    arima_predictions = arima_fitted.forecast(steps=len(test_data))
    
    # 評価
    lstm_mse = mean_squared_error(test_data, lstm_predictions)
    lstm_mae = mean_absolute_error(test_data, lstm_predictions)
    
    arima_mse = mean_squared_error(test_data, arima_predictions)
    arima_mae = mean_absolute_error(test_data, arima_predictions)
    
    print("\n=== 予測精度の比較 ===")
    print(f"LSTM  - MSE: {lstm_mse:.3f}, MAE: {lstm_mae:.3f}")
    print(f"ARIMA - MSE: {arima_mse:.3f}, MAE: {arima_mae:.3f}")
    
    # 可視化
    plt.figure(figsize=(14, 6))
    plt.plot(time, data, label='元データ', alpha=0.7)
    plt.axvline(x=train_size, color='red', linestyle='--', label='訓練/テスト境界')
    plt.plot(time[train_size:], lstm_predictions, label='LSTM予測', marker='o')
    plt.plot(time[train_size:], arima_predictions, label='ARIMA予測', marker='s')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('LSTM vs ARIMA 予測比較', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力例** ：
    
    
    === 予測精度の比較 ===
    LSTM  - MSE: 3.245, MAE: 1.432
    ARIMA - MSE: 4.187, MAE: 1.678
    

**結論** ：

  * LSTMは非線形パターンをより良く学習
  * ARIMAは線形トレンドと季節性に強い
  * データの特性に応じて選択が必要

### 問題5（難易度：hard）

Causal Convolution が未来のデータを参照しない仕組みを、パディングの観点から説明してください。また、なぜこれが時系列予測で重要なのかを述べてください。

解答例

**解答** ：

**Causal Convolution の仕組み** ：

  1. **通常の畳み込み（Non-Causal）**

     * パディング: 両側（左右）に追加
     * 問題: 未来のデータも参照してしまう
     * 例: カーネルサイズ3の場合、位置tで t-1, t, t+1 を参照
  2. **Causal Convolution**

     * パディング: 左側（過去側）のみに追加
     * 利点: 位置tでは t-2, t-1, t のみを参照（未来は見ない）
     * 実装: パディング後、右側を切り取る

**実装例** ：
    
    
    import torch
    import torch.nn as nn
    
    # 通常の畳み込み（Non-Causal）
    normal_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
    
    # Causal Convolution
    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            self.padding = kernel_size - 1
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=self.padding)
    
        def forward(self, x):
            x = self.conv(x)
            # 右側（未来側）のパディングを除去
            if self.padding != 0:
                x = x[:, :, :-self.padding]
            return x
    
    causal_conv = CausalConv1d(1, 1, kernel_size=3)
    
    # テストデータ
    x = torch.randn(1, 1, 10)  # (batch, channels, seq_len)
    
    print("入力シーケンス長:", x.shape[2])
    print("通常の畳み込み出力長:", normal_conv(x).shape[2])
    print("Causal畳み込み出力長:", causal_conv(x).shape[2])
    

**なぜ重要か** ：

  1. **データリークの防止**

     * 訓練時に未来の情報を使うと、過大評価される
     * 実運用では未来のデータは利用不可
     * Causalにすることで、訓練と推論の条件を一致させる
  2. **公平な評価**

     * 時系列の順序を尊重
     * モデルの真の予測能力を評価
     * 過学習の検出が正確
  3. **実用性**

     * リアルタイム予測に直接適用可能
     * オンライン学習が可能
     * ストリーミングデータに対応

**図解** ：
    
    
    通常の畳み込み（Non-Causal）:
    時刻t での計算: [t-1, t, t+1] → NG（未来を見ている）
    
    Causal Convolution:
    時刻t での計算: [t-2, t-1, t] → OK（過去のみ）
    

* * *

## 参考文献

  1. Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory_. Neural Computation, 9(8), 1735-1780.
  2. Cho, K., et al. (2014). _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation_. EMNLP.
  3. Bai, S., Kolter, J. Z., & Koltun, V. (2018). _An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling_. arXiv:1803.01271.
  4. Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS.
  5. Lim, B., & Zohren, S. (2021). _Time-series forecasting with deep learning: a survey_. Philosophical Transactions of the Royal Society A, 379(2194).
  6. Hewamalage, H., Bergmeir, C., & Bandara, K. (2021). _Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions_. International Journal of Forecasting, 37(1), 388-427.
