---
title: "Chapter 3: Time Series Forecasting with Deep Learning"
chapter_title: "Chapter 3: Time Series Forecasting with Deep Learning"
subtitle: Advanced Forecasting Models with LSTM, GRU, TCN, and Attention
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Time Series Forecasting with Deep Learning. You will learn Build time series forecasting models using LSTM, attention mechanisms to time series forecasting, and feature engineering.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the fundamental approaches for handling time series data with deep learning
  * ✅ Build time series forecasting models using LSTM and GRU
  * ✅ Learn the mechanisms and implementation of TCN (Temporal Convolutional Networks)
  * ✅ Apply attention mechanisms to time series forecasting
  * ✅ Understand feature engineering and ensemble methods
  * ✅ Implement practical time series forecasting models in PyTorch

* * *

## 3.1 Deep Learning for Time Series

### Sequential Data Representation

**Time series data (Sequential Data)** is data with a temporal ordering. In deep learning, we need to feed this data into models while preserving the temporal relationships.

> "The essence of time series forecasting is to infer the future from past patterns"

### Window-based Approach

The fundamental method for handling time series data in deep learning is the **Sliding Window** approach.
    
    
    ```mermaid
    graph LR
        A[Original Data: t1, t2, t3, t4, t5, t6] --> B[Window 1: t1-t3 → t4]
        A --> C[Window 2: t2-t4 → t5]
        A --> D[Window 3: t3-t5 → t6]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
    ```

### Implementation: Creating a Window-based Dataset
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation: Creating a Window-based Dataset
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    
    # Generate sample time series data
    np.random.seed(42)
    time = np.arange(0, 100, 0.1)
    data = np.sin(time) + 0.1 * np.random.randn(len(time))
    
    # Visualization
    plt.figure(figsize=(14, 5))
    plt.plot(time, data, label='Time Series Data', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Sample Time Series Data', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Number of data points: {len(data)}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    

### PyTorch Dataset for Time Series
    
    
    class TimeSeriesDataset(Dataset):
        """
        PyTorch Dataset for time series data
    
        Parameters:
        -----------
        data : np.ndarray
            Time series data (1D array)
        window_size : int
            Size of the input window
        horizon : int
            Prediction horizon (how many steps ahead to predict)
        """
        def __init__(self, data, window_size=20, horizon=1):
            self.data = data
            self.window_size = window_size
            self.horizon = horizon
    
        def __len__(self):
            return len(self.data) - self.window_size - self.horizon + 1
    
        def __getitem__(self, idx):
            # Input: window_size past data points
            x = self.data[idx:idx + self.window_size]
            # Target: horizon future data points
            y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
    
            return torch.FloatTensor(x), torch.FloatTensor(y)
    
    # Create datasets
    window_size = 20
    horizon = 5  # Predict 5 steps ahead
    
    # Train-validation split
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data, window_size, horizon)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("=== Dataset Information ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Input window size: {window_size}")
    print(f"Prediction horizon: {horizon}")
    
    # Check a sample
    x_sample, y_sample = train_dataset[0]
    print(f"\nSample shapes:")
    print(f"  Input x: {x_sample.shape}")
    print(f"  Target y: {y_sample.shape}")
    

**Output** :
    
    
    === Dataset Information ===
    Training samples: 776
    Validation samples: 176
    Input window size: 20
    Prediction horizon: 5
    
    Sample shapes:
      Input x: torch.Size([20])
      Target y: torch.Size([5])
    

### Multi-step Forecasting

In time series forecasting, there are two approaches:

Approach | Description | Advantages | Disadvantages  
---|---|---|---  
**One-shot** | Output entire horizon in one prediction | Fast, no dependencies | Difficult for long-term forecasting  
**Autoregressive** | Predict one step at a time, use as next input | Flexible, capable of long-term forecasting | Error accumulation  
      
    
    ```mermaid
    graph TD
        A[Past Data: t-n...t] --> B{Prediction Method}
        B -->|One-shot| C[Predict at once: t+1, t+2, ..., t+h]
        B -->|Autoregressive| D[Predict t+1]
        D --> E[Add t+1 to input]
        E --> F[Predict t+2]
        F --> G[Repeat...]
    
        style A fill:#e3f2fd
        style C fill:#c8e6c9
        style D fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 3.2 LSTM & GRU for Time Series Forecasting

### LSTM Architecture Review

**LSTM (Long Short-Term Memory)** is a type of RNN capable of learning long-term dependencies.

LSTM cell update equations:

$$ \begin{align*} f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)} \\\ i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)} \\\ \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate value)} \\\ C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell state update)} \\\ o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)} \\\ h_t &= o_t \odot \tanh(C_t) \quad \text{(Hidden state)} \end{align*} $$

### LSTM Implementation in PyTorch
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class LSTMForecaster(nn.Module):
        """
        LSTM-based time series forecasting model
    
        Parameters:
        -----------
        input_size : int
            Input feature dimension
        hidden_size : int
            LSTM hidden layer size
        num_layers : int
            Number of LSTM layers
        output_size : int
            Output size (prediction horizon)
        dropout : float
            Dropout rate
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                     output_size=1, dropout=0.2):
            super(LSTMForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
    
            # Fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
    
            # LSTM forward pass
            # out shape: (batch_size, seq_len, hidden_size)
            out, (h_n, c_n) = self.lstm(x)
    
            # Use the output of the last timestep
            # out[:, -1, :] shape: (batch_size, hidden_size)
            out = self.fc(out[:, -1, :])
    
            # out shape: (batch_size, output_size)
            return out
    
    # Model instantiation
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.2
    )
    
    print("=== LSTM Model Structure ===")
    print(model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):,}")
    

**Output** :
    
    
    === LSTM Model Structure ===
    LSTMForecaster(
      (lstm): LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
      (fc): Linear(in_features=64, out_features=5, bias=True)
    )
    
    Number of parameters: 50,245
    

### Training Loop Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - tqdm>=4.65.0
    
    import torch.optim as optim
    from tqdm import tqdm
    
    def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
        """
        Model training
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        train_losses = []
        val_losses = []
    
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
    
            for x_batch, y_batch in train_loader:
                # Convert data to (batch, seq_len, features) shape
                x_batch = x_batch.unsqueeze(-1).to(device)
                y_batch = y_batch.to(device)
    
                # Initialize gradients
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
    
            # Validation phase
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
    
    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
    
    # Visualize learning curves
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### GRU (Gated Recurrent Unit)

**GRU** is a simplified version of LSTM with fewer parameters and faster training.

GRU update equations:

$$ \begin{align*} r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(Reset gate)} \\\ z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(Update gate)} \\\ \tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(Candidate hidden state)} \\\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Hidden state update)} \end{align*} $$
    
    
    class GRUForecaster(nn.Module):
        """
        GRU-based time series forecasting model
        """
        def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                     output_size=1, dropout=0.2):
            super(GRUForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # GRU layer
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
    
            # Fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # GRU forward pass
            out, h_n = self.gru(x)
    
            # Use the output of the last timestep
            out = self.fc(out[:, -1, :])
    
            return out
    
    # GRU model instantiation
    gru_model = GRUForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.2
    )
    
    print("=== GRU Model Structure ===")
    print(gru_model)
    print(f"\nLSTM parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GRU parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
    print(f"Reduction rate: {(1 - sum(p.numel() for p in gru_model.parameters()) / sum(p.numel() for p in model.parameters())) * 100:.1f}%")
    

### Stateful vs Stateless LSTM

Type | Description | Use Case  
---|---|---  
**Stateless** | Reset hidden state for each batch | Independent sequences, general forecasting  
**Stateful** | Preserve hidden state between batches | Long continuous forecasting, streaming data  
  
* * *

## 3.3 TCN (Temporal Convolutional Network)

### What is TCN

**TCN (Temporal Convolutional Network)** is a convolutional neural network specialized for time series data. Unlike RNNs, it enables parallel processing and faster training.

### Dilated Convolutions

The core of TCN is **Dilated Convolution**. Compared to standard convolution, it achieves a wider receptive field with fewer parameters.
    
    
    ```mermaid
    graph TD
        A[Input Sequence] --> B[Layer 1: dilation=1]
        B --> C[Layer 2: dilation=2]
        C --> D[Layer 3: dilation=4]
        D --> E[Layer 4: dilation=8]
        E --> F[Output: Wide receptive field]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#ffccbc
        style E fill:#ffab91
        style F fill:#c8e6c9
    ```

Receptive field calculation:

$$ \text{Receptive Field} = 1 + 2 \times (k - 1) \times \sum_{i=0}^{L-1} d^i $$

  * $k$: Kernel size
  * $d$: Dilation factor
  * $L$: Number of layers

### Causal Convolutions

**Causal Convolution** is a convolution that doesn't use future information. This is essential for time series forecasting.

> **Important** : Padding is applied only on the left side to avoid referencing future data.

### TCN Implementation in PyTorch
    
    
    class CausalConv1d(nn.Module):
        """
        Causal 1D Convolution with dilation
        """
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(CausalConv1d, self).__init__()
    
            # Padding only on the left side (only reference past data)
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
    
            # Remove right-side padding (don't use future data)
            if self.padding != 0:
                x = x[:, :, :-self.padding]
    
            return x
    
    class TemporalBlock(nn.Module):
        """
        Basic TCN block
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
            # Conv1d expects (batch, features, seq_len)
            x = x.transpose(1, 2)
    
            # TCN forward pass
            y = self.network(x)
    
            # Use the last timestep
            y = y[:, :, -1]
    
            # Fully connected layer
            out = self.fc(y)
    
            return out
    
    # TCN model instantiation
    tcn_model = TCN(
        input_size=1,
        output_size=horizon,
        num_channels=[32, 32, 64, 64],  # 4 layers
        kernel_size=3,
        dropout=0.2
    )
    
    print("=== TCN Model Structure ===")
    print(tcn_model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in tcn_model.parameters()):,}")
    

### TCN vs RNN/LSTM

Feature | RNN/LSTM | TCN  
---|---|---  
**Parallel Processing** | Sequential, slow | Parallelizable, fast  
**Long-term Dependencies** | Vanishing gradient problem | Addressed with dilation  
**Receptive Field** | Depends on sequence length | Controlled by dilation  
**Memory Efficiency** | Requires hidden state | Convolution only  
**Training Time** | Slow | Fast  
  
* * *

## 3.4 Attention Mechanisms for Time Series

### Self-Attention for Sequences

**Attention mechanisms** enable the model to focus on important parts of the input sequence.

Attention calculation formula:

$$ \begin{align*} \text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\\ \text{where } Q &= XW_Q, \quad K = XW_K, \quad V = XW_V \end{align*} $$

  * $Q$: Query
  * $K$: Key
  * $V$: Value
  * $d_k$: Dimension of keys

### Attention Implementation in PyTorch
    
    
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
    
    # LSTM with Attention model
    attn_model = LSTMWithAttention(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        num_heads=4,
        dropout=0.2
    )
    
    print("=== LSTM + Attention Model Structure ===")
    print(attn_model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in attn_model.parameters()):,}")
    

### Attention Visualization
    
    
    def visualize_attention(model, data, window_size=20):
        """
        Visualize attention weights
        """
        model.eval()
        device = next(model.parameters()).device
    
        # Prepare sample data
        x = torch.FloatTensor(data[:window_size]).unsqueeze(0).unsqueeze(-1).to(device)
    
        with torch.no_grad():
            _, attn_weights = model(x)
    
        # Get attention weights (first head)
        attn = attn_weights[0, 0].cpu().numpy()
    
        # Visualization
        plt.figure(figsize=(12, 8))
    
        plt.subplot(2, 1, 1)
        plt.plot(data[:window_size], marker='o', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Input Sequence', fontsize=14)
        plt.grid(True, alpha=0.3)
    
        plt.subplot(2, 1, 2)
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Weights (What positions are being focused on)', fontsize=14)
    
        plt.tight_layout()
        plt.show()
    
    # Execute visualization
    visualize_attention(attn_model, train_data, window_size=20)
    

### Seq2Seq with Attention

An **Encoder-Decoder architecture** enhanced with attention can learn complex time series patterns.
    
    
    ```mermaid
    graph LR
        A[Input Sequence] --> B[Encoder LSTM]
        B --> C[Context Vector]
        C --> D[Decoder LSTM]
        D --> E[Attention Layer]
        E --> D
        E --> F[Prediction Sequence]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#ffccbc
        style E fill:#ffab91
        style F fill:#c8e6c9
    ```

* * *

## 3.5 Practical Techniques

### Feature Engineering for Deep Learning

Even with deep learning, proper feature engineering is important.
    
    
    def create_time_features(data, timestamps=None):
        """
        Create time-based features from time series data
        """
        features = pd.DataFrame()
    
        if timestamps is not None:
            # Time-based features
            features['hour'] = timestamps.hour / 24.0
            features['day_of_week'] = timestamps.dayofweek / 7.0
            features['day_of_month'] = timestamps.day / 31.0
            features['month'] = timestamps.month / 12.0
            features['is_weekend'] = (timestamps.dayofweek >= 5).astype(float)
    
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            features[f'lag_{lag}'] = pd.Series(data).shift(lag)
    
        # Rolling statistics
        for window in [3, 7, 14]:
            rolling = pd.Series(data).rolling(window)
            features[f'rolling_mean_{window}'] = rolling.mean()
            features[f'rolling_std_{window}'] = rolling.std()
            features[f'rolling_min_{window}'] = rolling.min()
            features[f'rolling_max_{window}'] = rolling.max()
    
        # Difference features
        features['diff_1'] = pd.Series(data).diff(1)
        features['diff_7'] = pd.Series(data).diff(7)
    
        # Fill missing values
        features = features.fillna(0)
    
        return features
    
    # Example of feature creation
    timestamps = pd.date_range(start='2023-01-01', periods=len(train_data), freq='h')
    time_features = create_time_features(train_data, timestamps)
    
    print("=== Created Features ===")
    print(time_features.head(20))
    print(f"\nNumber of features: {time_features.shape[1]}")
    

### Ensemble Methods

Combining multiple models can improve forecasting accuracy.
    
    
    class EnsembleForecaster:
        """
        Ensemble forecasting with multiple models
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
    
            # Weighted average
            ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
            ensemble_pred /= sum(self.weights)
    
            return ensemble_pred
    
    # Create ensemble
    models = [model, gru_model, tcn_model]
    ensemble = EnsembleForecaster(models, weights=[0.4, 0.3, 0.3])
    
    print("=== Ensemble Configuration ===")
    print(f"Number of models: {len(models)}")
    print(f"Weights: {ensemble.weights}")
    

### Transfer Learning for Time Series

Apply pre-trained models to new tasks.
    
    
    def transfer_learning(pretrained_model, new_output_size, freeze_layers=True):
        """
        Transfer learning setup
        """
        # Copy pre-trained model
        model = pretrained_model
    
        # Freeze encoder part
        if freeze_layers:
            for param in model.lstm.parameters():
                param.requires_grad = False
    
        # Re-initialize output layer for new task
        model.fc = nn.Linear(model.hidden_size, new_output_size)
    
        return model
    
    # Example usage
    # Apply pretrained_model to another task (horizon=10)
    new_model = transfer_learning(model, new_output_size=10, freeze_layers=True)
    
    print("=== Transfer Learning Model ===")
    print(f"Frozen parameters: {sum(p.numel() for p in new_model.parameters() if not p.requires_grad):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in new_model.parameters() if p.requires_grad):,}")
    

### Hyperparameter Tuning

Find the optimal model through hyperparameter search.
    
    
    from itertools import product
    
    def grid_search(param_grid, train_loader, val_loader, epochs=20):
        """
        Hyperparameter optimization with grid search
        """
        best_loss = float('inf')
        best_params = None
        results = []
    
        # Generate parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
    
        for params in product(*values):
            param_dict = dict(zip(keys, params))
    
            print(f"\nTrying: {param_dict}")
    
            # Create model
            model = LSTMForecaster(
                input_size=1,
                hidden_size=param_dict['hidden_size'],
                num_layers=param_dict['num_layers'],
                output_size=horizon,
                dropout=param_dict['dropout']
            )
    
            # Train
            _, val_losses = train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=param_dict['lr']
            )
    
            # Best validation loss
            min_val_loss = min(val_losses)
            results.append((param_dict, min_val_loss))
    
            if min_val_loss < best_loss:
                best_loss = min_val_loss
                best_params = param_dict
    
        return best_params, best_loss, results
    
    # Hyperparameter grid
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.0001]
    }
    
    # Execute grid search (small-scale example)
    small_grid = {
        'hidden_size': [32, 64],
        'num_layers': [1, 2],
        'dropout': [0.2],
        'lr': [0.001]
    }
    
    best_params, best_loss, all_results = grid_search(
        small_grid, train_loader, val_loader, epochs=10
    )
    
    print("\n=== Optimal Hyperparameters ===")
    print(f"Parameters: {best_params}")
    print(f"Validation loss: {best_loss:.4f}")
    

* * *

## 3.6 Chapter Summary

### What We Learned

  1. **Basic Deep Learning Approaches**

     * Window-based data preparation
     * Using PyTorch Dataset and DataLoader
     * One-shot vs Autoregressive forecasting
  2. **LSTM & GRU**

     * Learning long-term dependencies
     * Differences and use cases for LSTM and GRU
     * Stateful vs Stateless
  3. **TCN (Temporal Convolutional Network)**

     * Wide receptive field with dilated convolution
     * Causal convolution prevents future reference
     * Faster training than RNN
  4. **Attention Mechanisms**

     * Self-attention focuses on important timesteps
     * Combining LSTM + Attention
     * Visualizing attention weights
  5. **Practical Techniques**

     * Creating time features and lag features
     * Ensemble forecasting
     * Transfer learning
     * Hyperparameter tuning

### Model Selection Guidelines

Situation | Recommended Model | Reason  
---|---|---  
Short-term forecasting (< 10 steps) | LSTM/GRU | Simple and effective  
Long-term forecasting | TCN, Attention | Wide receptive field, parallel processing  
Training speed priority | TCN | Parallel processing  
Interpretability priority | Models with Attention | Visualize important timesteps  
Multivariate time series | Transformer | Learn complex dependencies  
Real-time forecasting | Stateful LSTM/GRU | Preserve hidden state  
  
### Next Chapter

The next chapter will cover more advanced topics including Transformers for Time Series analysis, anomaly detection techniques, multivariate time series forecasting methods, and probabilistic forecasting with prediction intervals. These advanced approaches build on the deep learning foundations covered in this chapter.

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Explain the differences between LSTM and GRU from the perspectives of gating mechanisms and number of parameters.

Sample Answer

**Answer** :

**LSTM (Long Short-Term Memory)** :

  * Gates: 3 gates (forget gate, input gate, output gate)
  * Maintains both cell state and hidden state
  * Number of parameters: More

**GRU (Gated Recurrent Unit)** :

  * Gates: 2 gates (reset gate, update gate)
  * Maintains only hidden state
  * Number of parameters: About 75% of LSTM

**Usage** :

  * LSTM: More complex patterns, strong long-term dependencies
  * GRU: Parameter reduction, training speed priority, limited data

The actual performance difference is often small, and validation for each task is recommended.

### Exercise 2 (Difficulty: Medium)

Calculate the receptive field of dilated convolution. With the following settings, what is the receptive field of the final layer?

  * Kernel size: k = 3
  * Number of layers: 4 layers
  * Dilation: [1, 2, 4, 8]

Sample Answer

**Answer** :

Receptive field calculation formula:

$$ \text{RF} = 1 + \sum_{i=1}^{L} (k - 1) \times d_i $$

  * $k = 3$: Kernel size
  * $L = 4$: Number of layers
  * $d_i$: Dilation for each layer

Calculation:
    
    
    k = 3
    dilations = [1, 2, 4, 8]
    
    receptive_field = 1
    for d in dilations:
        receptive_field += (k - 1) * d
    
    print(f"Receptive field: {receptive_field}")
    

**Output** :
    
    
    Receptive field: 31
    

This means it can consider 31 past timesteps.

Receptive field increase at each layer:

  * Layer 1 (d=1): RF = 1 + 2×1 = 3
  * Layer 2 (d=2): RF = 3 + 2×2 = 7
  * Layer 3 (d=4): RF = 7 + 2×4 = 15
  * Layer 4 (d=8): RF = 15 + 2×8 = 31

### Exercise 3 (Difficulty: Medium)

List three advantages of attention mechanisms and explain each.

Sample Answer

**Answer** :

  1. **Focus on Important Timesteps**

     * Explanation: Automatically focuses on highly relevant parts in the sequence
     * Advantage: Extracts important information even from noisy data
     * Example: Focuses on seasonal peaks or anomalous events
  2. **Learning Long-range Dependencies**

     * Explanation: Direct connections enable learning relationships with distant timesteps
     * Advantage: Avoids RNN's vanishing gradient problem
     * Example: Learning relationships between annual and daily patterns
  3. **Interpretability**

     * Explanation: Visualizing attention weights reveals the model's decision basis
     * Advantage: Not a black box; can explain prediction reasoning
     * Example: Can explain "this prediction is based on data from 1 week ago and 2 months ago"

**Additional Advantages** :

  * Parallel processing is possible (for self-attention)
  * Flexibly handles variable-length sequences
  * Applicable to diverse tasks

### Exercise 4 (Difficulty: Hard)

Using the following sample data, train an LSTM model and compare its forecasting accuracy with ARIMA.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Using the following sample data, train an LSTM model and com
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Sample data (monthly sales data)
    np.random.seed(42)
    time = np.arange(0, 100)
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.randn(100) * 2
    data = trend + seasonality + noise + 50
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Using the following sample data, train an LSTM model and com
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    time = np.arange(0, 100)
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)
    noise = np.random.randn(100) * 2
    data = trend + seasonality + noise + 50
    
    # Train-test split
    train_size = 80
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Dataset definition (using TimeSeriesDataset from earlier)
    window_size = 12
    horizon = 1
    
    train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # LSTM model (using LSTMForecaster from earlier)
    model = LSTMForecaster(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        output_size=horizon,
        dropout=0.1
    )
    
    # Training
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
    
    # LSTM predictions
    model.eval()
    lstm_predictions = []
    
    with torch.no_grad():
        # Sequential prediction on test data
        for i in range(len(test_data)):
            # Use recent window_size data
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
    
    # Comparison with ARIMA
    from statsmodels.tsa.arima.model import ARIMA
    
    # ARIMA model
    arima_model = ARIMA(train_data, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    arima_predictions = arima_fitted.forecast(steps=len(test_data))
    
    # Evaluation
    lstm_mse = mean_squared_error(test_data, lstm_predictions)
    lstm_mae = mean_absolute_error(test_data, lstm_predictions)
    
    arima_mse = mean_squared_error(test_data, arima_predictions)
    arima_mae = mean_absolute_error(test_data, arima_predictions)
    
    print("\n=== Forecasting Accuracy Comparison ===")
    print(f"LSTM  - MSE: {lstm_mse:.3f}, MAE: {lstm_mae:.3f}")
    print(f"ARIMA - MSE: {arima_mse:.3f}, MAE: {arima_mae:.3f}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(time, data, label='Original Data', alpha=0.7)
    plt.axvline(x=train_size, color='red', linestyle='--', label='Train/Test Boundary')
    plt.plot(time[train_size:], lstm_predictions, label='LSTM Predictions', marker='o')
    plt.plot(time[train_size:], arima_predictions, label='ARIMA Predictions', marker='s')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('LSTM vs ARIMA Prediction Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Example Output** :
    
    
    === Forecasting Accuracy Comparison ===
    LSTM  - MSE: 3.245, MAE: 1.432
    ARIMA - MSE: 4.187, MAE: 1.678
    

**Conclusion** :

  * LSTM learns nonlinear patterns better
  * ARIMA is strong with linear trends and seasonality
  * Selection depends on data characteristics

### Exercise 5 (Difficulty: Hard)

Explain how causal convolution avoids referencing future data from a padding perspective. Also, explain why this is important for time series forecasting.

Sample Answer

**Answer** :

**Causal Convolution Mechanism** :

  1. **Standard Convolution (Non-Causal)**

     * Padding: Added on both sides (left and right)
     * Problem: References future data
     * Example: With kernel size 3, position t references t-1, t, t+1
  2. **Causal Convolution**

     * Padding: Added only on the left side (past side)
     * Advantage: Position t only references t-2, t-1, t (no future)
     * Implementation: After padding, trim the right side

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Standard convolution (Non-Causal)
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
            # Remove right side (future side) padding
            if self.padding != 0:
                x = x[:, :, :-self.padding]
            return x
    
    causal_conv = CausalConv1d(1, 1, kernel_size=3)
    
    # Test data
    x = torch.randn(1, 1, 10)  # (batch, channels, seq_len)
    
    print("Input sequence length:", x.shape[2])
    print("Standard convolution output length:", normal_conv(x).shape[2])
    print("Causal convolution output length:", causal_conv(x).shape[2])
    

**Why It's Important** :

  1. **Preventing Data Leakage**

     * Using future information during training leads to overestimation
     * Future data is unavailable in real operation
     * Causal approach ensures training and inference conditions match
  2. **Fair Evaluation**

     * Respects temporal ordering
     * Evaluates true predictive ability of the model
     * Accurate detection of overfitting
  3. **Practicality**

     * Directly applicable to real-time forecasting
     * Enables online learning
     * Handles streaming data

**Illustration** :
    
    
    Standard Convolution (Non-Causal):
    Calculation at time t: [t-1, t, t+1] → NG (looking at future)
    
    Causal Convolution:
    Calculation at time t: [t-2, t-1, t] → OK (past only)
    

* * *

## References

  1. Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory_. Neural Computation, 9(8), 1735-1780.
  2. Cho, K., et al. (2014). _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation_. EMNLP.
  3. Bai, S., Kolter, J. Z., & Koltun, V. (2018). _An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling_. arXiv:1803.01271.
  4. Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS.
  5. Lim, B., & Zohren, S. (2021). _Time-series forecasting with deep learning: a survey_. Philosophical Transactions of the Royal Society A, 379(2194).
  6. Hewamalage, H., Bergmeir, C., & Bandara, K. (2021). _Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions_. International Journal of Forecasting, 37(1), 388-427.
