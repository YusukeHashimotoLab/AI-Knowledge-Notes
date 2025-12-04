---
title: 第4章：Transformerによる時系列予測
chapter_title: 第4章：Transformerによる時系列予測
subtitle: Temporal Fusion Transformer、Informer、長期予測への最先端アプローチ
reading_time: 30-35分
difficulty: 上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時系列データに対するTransformerのPositional Encodingを理解する
  * ✅ Temporal Attentionメカニズムの仕組みと実装方法を習得する
  * ✅ Multi-horizon予測の概念と実装を理解する
  * ✅ Temporal Fusion Transformer（TFT）の完全なアーキテクチャを理解する
  * ✅ Variable Selection NetworkとInterpretable Attentionを実装できる
  * ✅ InformerのProbSparse Attentionと長期予測手法を習得する
  * ✅ Autoformer、FEDformer、Patch TSTなどの最新モデルを理解する
  * ✅ pytorch-forecastingを用いた本格的な予測パイプラインを構築できる

* * *

## 4.1 Transformer for 時系列

### 時系列におけるTransformerの適用

**Transformer** は、元々自然言語処理のために設計されましたが、時系列予測においても強力なツールとなっています。Attention機構により、長期依存関係を効率的に捉えることができます。

> 「TransformerのSelf-Attentionは、時系列における任意の時点間の依存関係を直接的にモデル化できる。これはRNN/LSTMの逐次処理の制約を超える」

### Positional Encoding for Time

時系列データでは、時間的な順序情報が極めて重要です。Transformerには再帰構造がないため、**Positional Encoding** で時間情報を明示的に注入する必要があります。

#### 標準的なSinusoidal Encoding

$$ \begin{align} PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\\ PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) \end{align} $$

ここで：

  * $pos$：時間ステップの位置
  * $i$：次元のインデックス
  * $d_{model}$：モデルの次元数

#### 時系列特有のTemporal Encoding

時系列では、さらに以下の情報をエンコードすることが有効です：

  * **絶対時刻** ：時刻、日付、曜日、月など
  * **相対位置** ：現在からの相対的な時間距離
  * **周期性** ：日次、週次、年次などの周期パターン

### Temporal Attention

時系列用のAttention機構では、通常のSelf-Attentionに加えて、時間的な構造を考慮した工夫が施されます。

#### Masked Temporal Attention

予測タスクでは、未来の情報を見てはいけないため、**Causal Masking** を適用します：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

ここで、マスク行列$M$は：

$$ M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\\ -\infty & \text{if } i < j \end{cases} $$

### Multi-horizon Forecasting

**Multi-horizon forecasting** は、複数の将来時点を同時に予測するタスクです。Transformerのdecoderを使い、autoregressive方式または直接予測方式で実装します。
    
    
    ```mermaid
    graph LR
        A[Past Contextt-n...t] --> B[EncoderSelf-Attention]
        B --> C[DecoderMasked Attention]
        C --> D[Multi-step Outputt+1...t+h]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
    ```

### Vanilla Transformer Example
    
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PositionalEncoding(nn.Module):
        """時系列用のPositional Encoding"""
    
        def __init__(self, d_model, max_len=5000):
            super().__init__()
    
            # Positional encodingの計算
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                (-np.log(10000.0) / d_model))
    
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
    
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            """
            Args:
                x: Tensor of shape (seq_len, batch_size, d_model)
            """
            x = x + self.pe[:x.size(0)]
            return x
    
    
    class TimeSeriesTransformer(nn.Module):
        """時系列予測用のTransformer"""
    
        def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                     num_decoder_layers, dim_feedforward, output_len, dropout=0.1):
            super().__init__()
    
            self.d_model = d_model
            self.output_len = output_len
    
            # Input embedding
            self.encoder_input_layer = nn.Linear(input_dim, d_model)
            self.decoder_input_layer = nn.Linear(input_dim, d_model)
    
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model)
    
            # Transformer
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False
            )
    
            # Output layer
            self.output_layer = nn.Linear(d_model, input_dim)
    
        def generate_square_subsequent_mask(self, sz):
            """Causal maskの生成"""
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
            return mask
    
        def forward(self, src, tgt):
            """
            Args:
                src: (seq_len, batch_size, input_dim) - 過去データ
                tgt: (output_len, batch_size, input_dim) - デコーダ入力
            """
            # Embedding
            src = self.encoder_input_layer(src) * np.sqrt(self.d_model)
            tgt = self.decoder_input_layer(tgt) * np.sqrt(self.d_model)
    
            # Positional encoding
            src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
    
            # Causal mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
    
            # Transformer
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
    
            # Output projection
            output = self.output_layer(output)
    
            return output
    
    
    # モデルの使用例
    def train_transformer_example():
        """Transformerの訓練例"""
    
        # 合成データ生成（サイン波 + ノイズ）
        def generate_data(n_samples=1000, seq_len=50, output_len=10):
            X, y = [], []
            t = np.linspace(0, 100, n_samples + seq_len + output_len)
            data = np.sin(t * 0.1) + np.random.normal(0, 0.1, len(t))
    
            for i in range(n_samples):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len:i+seq_len+output_len])
    
            return np.array(X), np.array(y)
    
        # データ準備
        X_train, y_train = generate_data(n_samples=800)
        X_test, y_test = generate_data(n_samples=200)
    
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (800, 50, 1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)  # (800, 10, 1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
        # モデル構築
        model = TimeSeriesTransformer(
            input_dim=1,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            output_len=10,
            dropout=0.1
        )
    
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        # 訓練
        n_epochs = 50
        batch_size = 32
    
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
    
            # ミニバッチ訓練
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].transpose(0, 1)  # (seq_len, batch, 1)
                batch_y = y_train[i:i+batch_size].transpose(0, 1)  # (output_len, batch, 1)
    
                # デコーダ入力（teacher forcing用）
                # 最初のタイムステップはエンコーダの最後の値を使用
                decoder_input = torch.cat([
                    batch_X[-1:],  # 最後の値
                    batch_y[:-1]   # ターゲットの最初からn-1個
                ], dim=0)
    
                optimizer.zero_grad()
                output = model(batch_X, decoder_input)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(X_train):.6f}')
    
        # 評価
        model.eval()
        with torch.no_grad():
            # テストデータで予測（autoregressiveモード）
            test_X = X_test[0:1].transpose(0, 1)  # (50, 1, 1)
    
            # 初期デコーダ入力
            decoder_input = test_X[-1:]  # (1, 1, 1)
            predictions = []
    
            for _ in range(10):
                output = model(test_X, decoder_input)
                next_pred = output[-1:]  # 最後の予測
                predictions.append(next_pred.squeeze().item())
                decoder_input = torch.cat([decoder_input, next_pred], dim=0)
    
            # 可視化
            plt.figure(figsize=(12, 5))
            plt.plot(range(50), X_test[0].numpy(), label='Input', marker='o')
            plt.plot(range(50, 60), y_test[0].numpy(), label='True Future', marker='s')
            plt.plot(range(50, 60), predictions, label='Predicted', marker='^')
            plt.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Transformer Time Series Forecasting')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('transformer_forecast.png', dpi=150, bbox_inches='tight')
            plt.close()
    
            print(f"予測結果を保存しました: transformer_forecast.png")
            print(f"MSE: {np.mean((np.array(predictions) - y_test[0].numpy().flatten())**2):.6f}")
    
    if __name__ == "__main__":
        train_transformer_example()
    

* * *

## 4.2 Temporal Fusion Transformer (TFT)

### TFTの概要

**Temporal Fusion Transformer (TFT)** は、Google Researchが2021年に発表した時系列予測に特化したTransformerアーキテクチャです。解釈可能性と高精度を両立した設計が特徴です。

> 「TFTは、Variable Selection Network、LSTM-based Encoder-Decoder、Interpretable Multi-head Attentionを組み合わせることで、予測精度と解釈可能性を同時に実現する」

### TFT Architecture

TFTは以下の主要コンポーネントで構成されます：
    
    
    ```mermaid
    graph TB
        A[Input Variables] --> B[Variable Selection Network]
        B --> C[Static Covariate Encoder]
        B --> D[Temporal Processing]
    
        D --> E[LSTM EncoderPast]
        D --> F[LSTM DecoderFuture]
    
        C --> G[Context Vector]
        E --> G
        F --> G
    
        G --> H[Gated Residual Network]
        H --> I[Multi-head Attention]
        I --> J[Feed-Forward]
        J --> K[Quantile Output]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style I fill:#e8f5e9
        style K fill:#f3e5f5
    ```

### Variable Selection Network

**Variable Selection Network (VSN)** は、入力変数の重要度を学習し、自動的に特徴選択を行います。

各変数$v_i$に対する重要度重み$w_i$を計算：

$$ \mathbf{w} = \text{Softmax}(\text{GRN}(\mathbf{v}_1, \ldots, \mathbf{v}_n)) $$

選択された変数：

$$ \mathbf{\xi} = \sum_{i=1}^{n} w_i \cdot \text{GRN}(\mathbf{v}_i) $$

ここで、**GRN (Gated Residual Network)** は、以下の構造を持つブロックです：

$$ \text{GRN}(\mathbf{a}, \mathbf{c}) = \text{LayerNorm}(\mathbf{a} + \text{GLU}(\eta_1)) \\\ \eta_1 = \mathbf{W}_1\eta_2 + \mathbf{b}_1 \\\ \eta_2 = \text{ELU}(\mathbf{W}_2\mathbf{a} + \mathbf{W}_3\mathbf{c} + \mathbf{b}_2) $$

### Interpretable Multi-head Attention

TFTのAttention機構は、各時点の重要度を可視化できるように設計されています：

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Attention weightsを平均化することで、各時点の重要度を解釈：

$$ \alpha_t = \frac{1}{H}\sum_{h=1}^{H} \text{Softmax}\left(\frac{Q_hK_h^T}{\sqrt{d_k}}\right)_{t,:} $$

### pytorch-forecasting Library

`pytorch-forecasting`は、TFTを含む時系列予測モデルを簡単に使えるライブラリです。
    
    
    import pandas as pd
    import numpy as np
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning import Trainer
    import matplotlib.pyplot as plt
    
    def create_tft_example():
        """TFTを使った予測例"""
    
        # 合成データ生成
        np.random.seed(42)
        n_samples = 1000
    
        data = []
        for store_id in range(5):
            for day in range(n_samples):
                # トレンド + 季節性 + ノイズ
                trend = day * 0.1
                seasonality = 10 * np.sin(2 * np.pi * day / 30)  # 月次周期
                weekly = 5 * np.sin(2 * np.pi * day / 7)  # 週次周期
                noise = np.random.normal(0, 2)
                store_effect = store_id * 5
    
                value = 50 + trend + seasonality + weekly + noise + store_effect
    
                data.append({
                    'time_idx': day,
                    'store_id': str(store_id),
                    'value': max(0, value),
                    'day_of_week': day % 7,
                    'day_of_month': (day % 30) + 1,
                    'month': ((day // 30) % 12) + 1
                })
    
        df = pd.DataFrame(data)
    
        # TimeSeriesDataSetの作成
        max_encoder_length = 60  # 過去60日を使用
        max_prediction_length = 20  # 20日先を予測
        training_cutoff = df["time_idx"].max() - max_prediction_length
    
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="value",
            group_ids=["store_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["store_id"],
            time_varying_known_reals=["time_idx", "day_of_week", "day_of_month", "month"],
            time_varying_unknown_reals=["value"],
            target_normalizer=GroupNormalizer(
                groups=["store_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    
        # Validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True
        )
    
        # DataLoaders
        batch_size = 64
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
    
        # TFTモデルの構築
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # quantile outputの数
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    
        print(f"TFTモデルのパラメータ数: {tft.size()/1e3:.1f}k")
    
        # 訓練
        trainer = Trainer(
            max_epochs=30,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=30,
            enable_checkpointing=True,
        )
    
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    
        # 予測
        best_model = tft.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
        # 最初のバッチで予測
        predictions = best_model.predict(val_dataloader, return_x=True)
    
        # 可視化
        for idx in range(min(3, len(predictions.output))):
            best_model.plot_prediction(
                predictions.x, predictions.output, idx=idx, add_loss_to_title=True
            )
            plt.savefig(f'tft_prediction_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
        print(f"予測結果を保存しました: tft_prediction_*.png")
    
        # Variable importanceの可視化
        interpretation = best_model.interpret_output(predictions.output, reduction="sum")
    
        # Attention weights
        fig, ax = plt.subplots(figsize=(10, 5))
        attention = interpretation["attention"].mean(0).cpu().numpy()
    
        im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Encoder Time Steps')
        ax.set_ylabel('Decoder Time Steps')
        ax.set_title('TFT Attention Weights (Interpretability)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('tft_attention.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"Attention weightsを保存しました: tft_attention.png")
    
        # Variable importance
        importance = best_model.interpret_output(
            predictions.output, reduction="sum"
        )
    
        return best_model, predictions, importance
    
    if __name__ == "__main__":
        model, preds, importance = create_tft_example()
    

* * *

## 4.3 Informer

### Informerの動機

**Informer** は、長期時系列予測（LSTF: Long Sequence Time-series Forecasting）のために設計されたTransformerです。標準的なTransformerの計算量とメモリ使用量の問題を解決します。

> 「標準的なTransformerのAttentionは$O(L^2)$の計算量を持つため、長い系列（例：1000ステップ以上）では実用的でない。InformerはProbSparse Attentionにより$O(L\log L)$に削減する」

### ProbSparse Attention

**ProbSparse Self-Attention** は、重要なクエリのみを選択的に計算する効率的なAttention機構です。

#### Query Sparsity Measurement

各クエリ$q_i$の「重要度」をスパース性で測定：

$$ M(q_i, K) = \ln \sum_{j=1}^{L_k} e^{\frac{q_i k_j^T}{\sqrt{d}}} - \frac{1}{L_k}\sum_{j=1}^{L_k}\frac{q_i k_j^T}{\sqrt{d}} $$

この値が大きいほど、クエリは特定のキーに集中しており（スパース）、重要です。

#### Top-u Selection

上位$u$個のクエリのみでAttentionを計算：

$$ \bar{Q} = \text{Top-}u(M(q_i, K)) $$

$$ \text{ProbSparseAttention}(\bar{Q}, K, V) = \text{Softmax}\left(\frac{\bar{Q}K^T}{\sqrt{d}}\right)V $$

他のクエリは平均値で埋める：

$$ \text{Attention}(Q, K, V) = [\text{ProbSparseAttention}(\bar{Q}, K, V); \bar{V}] $$

### Self-Attention Distilling

Informerは、エンコーダ層ごとに系列長を半分にする**Distilling操作** を適用します：

  1. Self-Attention層を通過
  2. 1D Convolution + Max Poolingで系列長を半分に
  3. 次の層へ

これにより、$L \to L/2 \to L/4 \to \ldots$と系列長が減少し、メモリ効率が向上します。
    
    
    ```mermaid
    graph LR
        A[Input L] --> B[ProbSparse Attn]
        B --> C[DistillingL/2]
        C --> D[ProbSparse Attn]
        D --> E[DistillingL/4]
        E --> F[...]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#e8f5e9
    ```

### Informer Implementation
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ProbAttention(nn.Module):
        """ProbSparse Self-Attention"""
    
        def __init__(self, d_model, n_heads, factor=5):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.factor = factor
    
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
    
        def forward(self, queries, keys, values, attn_mask=None):
            B, L_q, _ = queries.shape
            _, L_k, _ = keys.shape
    
            # Linear projection
            Q = self.W_q(queries).view(B, L_q, self.n_heads, self.d_k)
            K = self.W_k(keys).view(B, L_k, self.n_heads, self.d_k)
            V = self.W_v(values).view(B, L_k, self.n_heads, self.d_k)
    
            # Transpose for multi-head attention
            Q = Q.transpose(1, 2)  # (B, n_heads, L_q, d_k)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
    
            # ProbSparse Attention
            # サンプリング数
            u = self.factor * int(np.ceil(np.log(L_q)))
            u = min(u, L_q)
    
            # ランダムサンプリング（簡易版、本来はスパース性で選択）
            Q_sample = Q[:, :, :u, :]
    
            # Attention scores
            scores = torch.matmul(Q_sample, K.transpose(-2, -1)) / np.sqrt(self.d_k)
    
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask[:, :, :u, :] == 0, -1e9)
    
            attn = F.softmax(scores, dim=-1)
    
            # Apply attention to values
            out_sample = torch.matmul(attn, V)  # (B, n_heads, u, d_k)
    
            # 残りは平均値で埋める
            V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L_q - u, -1)
            out = torch.cat([out_sample, V_mean], dim=2)
    
            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
            out = self.W_o(out)
    
            return out, attn
    
    
    class Distilling(nn.Module):
        """Distilling操作（系列長を半分に）"""
    
        def __init__(self, d_model):
            super().__init__()
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(d_model)
            self.activation = nn.ELU()
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
        def forward(self, x):
            # x: (B, L, d_model)
            x = x.transpose(1, 2)  # (B, d_model, L)
            x = self.conv(x)
            x = self.activation(x)
            x = self.maxpool(x)
            x = x.transpose(1, 2)  # (B, L/2, d_model)
            x = self.norm(x)
            return x
    
    
    class InformerEncoder(nn.Module):
        """Informerのエンコーダ"""
    
        def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
            super().__init__()
    
            self.attn_layers = nn.ModuleList([
                ProbAttention(d_model, n_heads) for _ in range(n_layers)
            ])
    
            self.distilling_layers = nn.ModuleList([
                Distilling(d_model) for _ in range(n_layers - 1)
            ])
    
            self.ffn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ) for _ in range(n_layers)
            ])
    
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(2 * n_layers)
            ])
    
        def forward(self, x):
            attns = []
    
            for i, (attn, ffn) in enumerate(zip(self.attn_layers, self.ffn_layers)):
                # Self-attention
                new_x, attn_weights = attn(x, x, x)
                x = self.norm_layers[2*i](x + new_x)
                attns.append(attn_weights)
    
                # FFN
                new_x = ffn(x)
                x = self.norm_layers[2*i+1](x + new_x)
    
                # Distilling（最後の層以外）
                if i < len(self.distilling_layers):
                    x = self.distilling_layers[i](x)
    
            return x, attns
    
    
    def test_informer():
        """Informerのテスト"""
    
        # パラメータ
        batch_size = 4
        seq_len = 96  # 長い系列
        d_model = 64
        n_heads = 4
        d_ff = 256
        n_layers = 3
    
        # エンコーダ
        encoder = InformerEncoder(d_model, n_heads, d_ff, n_layers)
    
        # ダミー入力
        x = torch.randn(batch_size, seq_len, d_model)
    
        # Forward
        output, attns = encoder(x)
    
        print(f"入力サイズ: {x.shape}")
        print(f"出力サイズ: {output.shape}")
        print(f"Attention weights数: {len(attns)}")
    
        # 系列長の減少を確認
        print("\n層ごとの系列長:")
        test_x = x
        for i, distill in enumerate(encoder.distilling_layers):
            test_x = distill(test_x)
            print(f"  Layer {i+1}: {test_x.shape[1]}")
    
        # Attention weightsの可視化
        fig, axes = plt.subplots(1, len(attns), figsize=(15, 3))
        for i, attn in enumerate(attns):
            # 最初のバッチ、最初のヘッドのattentionを表示
            attn_map = attn[0, 0].detach().numpy()
            axes[i].imshow(attn_map, cmap='viridis', aspect='auto')
            axes[i].set_title(f'Layer {i+1}')
            axes[i].set_xlabel('Key')
            axes[i].set_ylabel('Query (sampled)')
    
        plt.tight_layout()
        plt.savefig('informer_attention.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"\nAttention weightsを保存しました: informer_attention.png")
    
    if __name__ == "__main__":
        test_informer()
    

* * *

## 4.4 その他のTransformerモデル

### Autoformer

**Autoformer** （2021）は、時系列の分解とAuto-Correlationメカニズムを導入したモデルです。

#### 主な特徴

  * **Series Decomposition Block** ：トレンドと季節成分を分離
  * **Auto-Correlation Mechanism** ：時系列の周期性を直接捉える
  * **Progressive Decomposition** ：各層で分解を繰り返す

#### Auto-Correlation

従来のAttentionの代わりに、時系列の自己相関を利用：

$$ \text{AutoCorr}(Q, K, V) = \text{Softmax}\left(\frac{\mathcal{R}_{Q,K}}{\tau}\right)V $$

ここで、$\mathcal{R}_{Q,K}$は自己相関関数です。

### FEDformer

**FEDformer** （Frequency Enhanced Decomposed Transformer, 2022）は、周波数領域での処理を導入したモデルです。

#### 主な特徴

  * **Frequency Enhanced Block (FEB)** ：FFTを用いた周波数領域での処理
  * **Seasonal-Trend Decomposition** ：周波数領域での分解
  * **Fourier Enhanced Attention** ：周波数成分に基づくAttention

#### 周波数領域処理

$$ \hat{X} = \text{FFT}(X) \\\ \hat{X}' = \text{FrequencyAttention}(\hat{X}) \\\ X' = \text{IFFT}(\hat{X}') $$

### Patch TST (PatchTST)

**PatchTST** （2023）は、時系列をパッチに分割してTransformerに入力する新しいアプローチです。

#### 主な特徴

  * **Patching** ：連続する時間ステップをパッチとして扱う
  * **Channel Independence** ：各変数を独立に処理
  * **Efficient Architecture** ：パラメータ数と計算量を削減

#### Patching操作

長さ$L$の系列を、サイズ$P$のパッチ$N = L/P$個に分割：

$$ X = [x_1, x_2, \ldots, x_L] \to [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N] $$

各パッチ$\mathbf{p}_i \in \mathbb{R}^P$をTransformerのトークンとして扱います。

### Model Comparison

モデル | 主な特徴 | 計算量 | 長期予測 | 解釈性  
---|---|---|---|---  
**Vanilla Transformer** | 標準的なAttention | $O(L^2)$ | △ | △  
**TFT** | Variable Selection、解釈可能 | $O(L^2)$ | 〇 | ◎  
**Informer** | ProbSparse Attention | $O(L\log L)$ | ◎ | △  
**Autoformer** | Auto-Correlation、分解 | $O(L\log L)$ | ◎ | 〇  
**FEDformer** | 周波数領域処理 | $O(L)$ | ◎ | 〇  
**PatchTST** | Patching、効率的 | $O((L/P)^2)$ | ◎ | △  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    def visualize_model_comparison():
        """Transformerモデルの比較可視化"""
    
        models = ['Vanilla\nTransformer', 'TFT', 'Informer',
                  'Autoformer', 'FEDformer', 'PatchTST']
    
        # 性能指標（仮想的なスコア 0-10）
        accuracy = [7, 8.5, 8, 8.5, 9, 8.5]
        efficiency = [4, 5, 8, 8, 9, 9]
        interpretability = [5, 9, 5, 7, 7, 4]
        long_term = [5, 7, 9, 9, 9.5, 9]
    
        x = np.arange(len(models))
        width = 0.2
    
        fig, ax = plt.subplots(figsize=(14, 6))
    
        ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#7b2cbf')
        ax.bar(x - 0.5*width, efficiency, width, label='Efficiency', color='#9d4edd')
        ax.bar(x + 0.5*width, interpretability, width, label='Interpretability', color='#c77dff')
        ax.bar(x + 1.5*width, long_term, width, label='Long-term Forecasting', color='#e0aaff')
    
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Score (0-10)', fontweight='bold')
        ax.set_title('Transformer Models Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 10)
    
        plt.tight_layout()
        plt.savefig('transformer_models_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print("モデル比較グラフを保存しました: transformer_models_comparison.png")
    
        # 計算量の比較
        fig, ax = plt.subplots(figsize=(10, 6))
    
        seq_lengths = np.arange(100, 2001, 100)
    
        # 計算量（正規化）
        vanilla = (seq_lengths ** 2) / 1000
        informer = (seq_lengths * np.log(seq_lengths)) / 100
        fedformer = seq_lengths / 10
        patch_size = 16
        patchtst = ((seq_lengths / patch_size) ** 2) / 1000
    
        ax.plot(seq_lengths, vanilla, label='Vanilla ($O(L^2)$)',
                linewidth=2, marker='o', markersize=3, color='#e63946')
        ax.plot(seq_lengths, informer, label='Informer ($O(L\\log L)$)',
                linewidth=2, marker='s', markersize=3, color='#f77f00')
        ax.plot(seq_lengths, fedformer, label='FEDformer ($O(L)$)',
                linewidth=2, marker='^', markersize=3, color='#06a77d')
        ax.plot(seq_lengths, patchtst, label='PatchTST ($O((L/P)^2)$, P=16)',
                linewidth=2, marker='d', markersize=3, color='#7b2cbf')
    
        ax.set_xlabel('Sequence Length', fontweight='bold')
        ax.set_ylabel('Computational Cost (normalized)', fontweight='bold')
        ax.set_title('Computational Complexity Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(100, 2000)
    
        plt.tight_layout()
        plt.savefig('transformer_complexity.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print("計算量比較グラフを保存しました: transformer_complexity.png")
    
    if __name__ == "__main__":
        visualize_model_comparison()
    

* * *

## 4.5 実践プロジェクト

### Multi-variate Forecasting with Exogenous Variables

実際のビジネス問題では、複数の時系列変数と外生変数を同時に扱う必要があります。ここでは、TFTを使った完全な予測パイプラインを構築します。

### プロジェクト設定

**タスク** ：小売店の売上予測

**データ** ：

  * **Target** ：日次売上
  * **Time-varying known** ：価格、プロモーション、曜日、祝日
  * **Time-varying unknown** ：競合の活動、天候（予測困難）
  * **Static** ：店舗カテゴリ、地域

### Complete TFT Pipeline
    
    
    import pandas as pd
    import numpy as np
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, SMAPE
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    
    def generate_retail_data():
        """小売売上データの生成"""
    
        np.random.seed(42)
    
        # 店舗情報
        stores = [
            {'store_id': 'A', 'category': 'urban', 'region': 'north'},
            {'store_id': 'B', 'category': 'urban', 'region': 'south'},
            {'store_id': 'C', 'category': 'suburban', 'region': 'north'},
            {'store_id': 'D', 'category': 'suburban', 'region': 'south'},
            {'store_id': 'E', 'category': 'rural', 'region': 'west'},
        ]
    
        data = []
        n_days = 730  # 2年分
    
        for store in stores:
            store_id = store['store_id']
            base_sales = {'urban': 1000, 'suburban': 600, 'rural': 300}[store['category']]
    
            for day in range(n_days):
                # 日付特徴
                date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=day)
                day_of_week = date.dayofweek
                month = date.month
                is_weekend = int(day_of_week >= 5)
                is_holiday = int(month == 12 and date.day >= 20)  # 年末
    
                # トレンド
                trend = day * 0.5
    
                # 季節性
                yearly_season = 200 * np.sin(2 * np.pi * day / 365)
                weekly_season = 150 * (1 if day_of_week in [5, 6] else 0)
    
                # 外生変数
                price = 100 + np.random.normal(0, 5)
                promotion = int(np.random.random() < 0.15)  # 15%の確率
                competitor_activity = np.random.normal(0.5, 0.2)
    
                # 売上計算
                sales = base_sales + trend + yearly_season + weekly_season
                sales *= (1 + 0.3 * promotion)  # プロモーション効果
                sales *= (1 - 0.2 * competitor_activity)  # 競合の影響
                sales *= (0.9 if day_of_week == 0 else 1.0)  # 月曜は低い
                sales += np.random.normal(0, 50)
                sales = max(0, sales)
    
                data.append({
                    'date': date,
                    'time_idx': day,
                    'store_id': store_id,
                    'category': store['category'],
                    'region': store['region'],
                    'sales': sales,
                    'price': price,
                    'promotion': promotion,
                    'day_of_week': day_of_week,
                    'month': month,
                    'is_weekend': is_weekend,
                    'is_holiday': is_holiday,
                    'competitor_activity': competitor_activity,
                })
    
        return pd.DataFrame(data)
    
    
    def build_tft_forecaster():
        """TFT予測器の構築と訓練"""
    
        # データ生成
        print("データ生成中...")
        df = generate_retail_data()
    
        print(f"データサイズ: {len(df)} rows")
        print(f"店舗数: {df['store_id'].nunique()}")
        print(f"期間: {df['date'].min()} to {df['date'].max()}")
    
        # データセット作成
        max_encoder_length = 60
        max_prediction_length = 30
        training_cutoff = df["time_idx"].max() - max_prediction_length
    
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="sales",
            group_ids=["store_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
    
            # Static features
            static_categoricals=["store_id", "category", "region"],
    
            # Time-varying known (future values are known)
            time_varying_known_categoricals=["day_of_week", "month", "is_weekend", "is_holiday"],
            time_varying_known_reals=["time_idx", "price", "promotion"],
    
            # Time-varying unknown (future values are not known)
            time_varying_unknown_reals=["sales", "competitor_activity"],
    
            # Normalization
            target_normalizer=GroupNormalizer(
                groups=["store_id"], transformation="softplus"
            ),
    
            # Additional features
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    
        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True
        )
    
        # DataLoaders
        batch_size = 32
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0, shuffle=True
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
    
        print(f"\n訓練バッチ数: {len(train_dataloader)}")
        print(f"検証バッチ数: {len(val_dataloader)}")
    
        # TFTモデル
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    
        print(f"\nTFTモデル構築完了")
        print(f"パラメータ数: {tft.size()/1e3:.1f}k")
    
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=False,
            mode="min"
        )
    
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=False
        )
    
        # Trainer
        trainer = Trainer(
            max_epochs=100,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            limit_train_batches=50,
            limit_val_batches=10,
            enable_checkpointing=True,
        )
    
        # 訓練
        print("\n訓練開始...")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    
        # ベストモデルのロード
        best_model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
    
        # 評価
        print("\n評価中...")
        predictions = best_model.predict(val_dataloader, return_x=True, return_y=True)
    
        # メトリクス計算
        actuals = predictions.y[0].cpu().numpy()
        preds = predictions.output['prediction'].cpu().numpy()
    
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    
        print(f"\n評価結果:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
        # 可視化
        visualize_predictions(best_model, predictions, df)
        visualize_interpretation(best_model, predictions)
    
        return best_model, predictions, df
    
    
    def visualize_predictions(model, predictions, df):
        """予測結果の可視化"""
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    
        for idx in range(min(4, len(predictions.output))):
            ax = axes[idx]
    
            # データ取得
            x = predictions.x
            y_true = predictions.y[0][idx].cpu().numpy()
            y_pred = predictions.output['prediction'][idx].cpu().numpy()
    
            # 予測区間（quantiles）
            quantiles = predictions.output['quantiles'][idx].cpu().numpy()
    
            time_steps = np.arange(len(y_true))
    
            # プロット
            ax.plot(time_steps, y_true, 'o-', label='Actual', color='#2d3748', linewidth=2)
            ax.plot(time_steps, y_pred, 's-', label='Predicted', color='#7b2cbf', linewidth=2)
    
            # 予測区間（10%-90%）
            ax.fill_between(
                time_steps,
                quantiles[:, 0],  # 10% quantile
                quantiles[:, -1],  # 90% quantile
                alpha=0.2,
                color='#9d4edd',
                label='10%-90% Prediction Interval'
            )
    
            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Sales', fontweight='bold')
            ax.set_title(f'Store {idx+1}: Sales Forecast', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('tft_sales_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"予測結果を保存しました: tft_sales_predictions.png")
    
    
    def visualize_interpretation(model, predictions):
        """Variable ImportanceとAttentionの可視化"""
    
        interpretation = model.interpret_output(
            predictions.output, reduction="sum"
        )
    
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
        # Variable importance
        ax = axes[0]
    
        # Encoder variable importance
        encoder_importance = interpretation["encoder_variables"].cpu().numpy()
        encoder_vars = list(interpretation["encoder_variables_names"])
    
        y_pos = np.arange(len(encoder_vars))
        ax.barh(y_pos, encoder_importance, color='#7b2cbf')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(encoder_vars)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title('Encoder Variable Importance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
        # Attention weights
        ax = axes[1]
        attention = interpretation["attention"].mean(0).cpu().numpy()
    
        im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Encoder Time Steps', fontweight='bold')
        ax.set_ylabel('Decoder Time Steps', fontweight='bold')
        ax.set_title('Average Attention Weights', fontweight='bold')
        plt.colorbar(im, ax=ax)
    
        plt.tight_layout()
        plt.savefig('tft_interpretation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"解釈可能性の可視化を保存しました: tft_interpretation.png")
    
    
    if __name__ == "__main__":
        model, predictions, df = build_tft_forecaster()
    
        print("\n" + "="*60)
        print("TFT予測パイプライン完了！")
        print("="*60)
    

### Production Deployment Considerations

本番環境へのデプロイ時には、以下の点に注意が必要です：

#### 1\. モデルの保存とロード
    
    
    # モデルの保存
    model.save("tft_model.pt")
    
    # モデルのロード
    from pytorch_forecasting import TemporalFusionTransformer
    loaded_model = TemporalFusionTransformer.load_from_checkpoint("tft_model.pt")
    

#### 2\. バッチ予測の最適化
    
    
    # 大規模データでの予測
    predictions = model.predict(
        dataloader,
        mode="raw",  # 生の出力を取得
        return_index=True,  # インデックスも返す
        trainer_kwargs={"accelerator": "gpu"}  # GPUを使用
    )
    

#### 3\. リアルタイム予測API
    
    
    from fastapi import FastAPI
    import torch
    
    app = FastAPI()
    
    # モデルをグローバルにロード
    model = TemporalFusionTransformer.load_from_checkpoint("tft_model.pt")
    model.eval()
    
    @app.post("/predict")
    async def predict(input_data: dict):
        # データの前処理
        dataset = prepare_dataset(input_data)
        dataloader = dataset.to_dataloader(train=False, batch_size=1)
    
        # 予測
        with torch.no_grad():
            predictions = model.predict(dataloader)
    
        return {"predictions": predictions.tolist()}
    

#### 4\. モニタリングとドリフト検出

  * **予測精度のモニタリング** ：定期的にMAE/RMSEを計算
  * **データドリフト検出** ：入力分布の変化を監視
  * **再訓練トリガー** ：精度低下時に自動再訓練

* * *

## 練習問題

**問題1：Positional Encoding** \- 時系列用のPositional Encodingを実装し、その役割を説明してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    class TemporalPositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
    
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                (-np.log(10000.0) / d_model))
    
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            self.register_buffer('pe', pe.unsqueeze(0))
    
        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]
    
    # 可視化
    pe = TemporalPositionalEncoding(d_model=128, max_len=100)
    encoding = pe.pe[0].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150)
    plt.close()
    
    print("Positional Encodingは、系列の位置情報を注入する役割を持つ")
    print("異なる周波数のsin/cos波により、相対位置も捉えられる")
    

**役割** ：

  * Transformerには再帰構造がないため、位置情報を明示的に追加
  * 異なる周波数により、近い位置と遠い位置の両方を区別
  * 時系列では、絶対時刻や周期性も追加エンコード可能

**問題2：TFTのVariable Selection** \- Variable Selection Networkの実装と、その利点を説明してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    
    class GatedResidualNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.gate = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)
    
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = None
    
        def forward(self, x):
            # GRN計算
            eta2 = torch.relu(self.fc1(x))
            eta1 = self.fc2(eta2)
            gate = torch.sigmoid(self.gate(eta2))
    
            # Gated output
            output = gate * eta1
            output = self.dropout(output)
    
            # Skip connection
            if self.skip is not None:
                x = self.skip(x)
    
            return self.layer_norm(x + output)
    
    
    class VariableSelectionNetwork(nn.Module):
        def __init__(self, input_dims, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.input_dims = input_dims
    
            # 各変数用のGRN
            self.variable_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, output_dim, dropout)
                for _ in range(len(input_dims))
            ])
    
            # 重み計算用のGRN
            self.weight_grn = GatedResidualNetwork(
                sum(input_dims), hidden_dim, len(input_dims), dropout
            )
    
        def forward(self, variables):
            # variables: list of tensors
            # 各変数を変換
            transformed = [grn(v.unsqueeze(-1)) for grn, v in
                          zip(self.variable_grns, variables)]
    
            # 重み計算
            concat_vars = torch.cat(variables, dim=-1)
            weights = torch.softmax(self.weight_grn(concat_vars), dim=-1)
    
            # 加重和
            output = sum(w.unsqueeze(-1) * t for w, t in
                        zip(weights.split(1, dim=-1), transformed))
    
            return output, weights
    
    # テスト
    n_vars = 5
    batch_size = 32
    seq_len = 50
    
    vsn = VariableSelectionNetwork(
        input_dims=[1]*n_vars, hidden_dim=64, output_dim=32
    )
    
    variables = [torch.randn(batch_size, seq_len) for _ in range(n_vars)]
    output, weights = vsn(variables)
    
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Variable importance: {weights[0, 0]}")
    

**利点** ：

  * 自動的に重要な変数を選択し、ノイズを削減
  * 解釈可能性：どの変数が重要かを可視化
  * 過学習の防止：不要な変数の影響を抑制

**問題3：ProbSparse Attention** \- InformerのProbSparse Attentionの効率性を、標準的なAttentionと比較してください。

**解答例** ：
    
    
    import torch
    import time
    import matplotlib.pyplot as plt
    
    def standard_attention(Q, K, V):
        """標準的なAttention O(L^2)"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
    
    def probsparse_attention(Q, K, V, factor=5):
        """ProbSparse Attention O(L log L)"""
        L_q = Q.size(1)
        L_k = K.size(1)
        d_k = Q.size(-1)
    
        # Top-u selection
        u = factor * int(np.ceil(np.log(L_q)))
        u = min(u, L_q)
    
        # サンプリング（簡易版）
        Q_sample = Q[:, :u, :]
    
        scores = torch.matmul(Q_sample, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
    
        out_sample = torch.matmul(attn, V)
    
        # 残りは平均
        V_mean = V.mean(dim=1, keepdim=True).expand(-1, L_q - u, -1)
        output = torch.cat([out_sample, V_mean], dim=1)
    
        return output
    
    # ベンチマーク
    seq_lengths = [100, 200, 500, 1000, 1500, 2000]
    standard_times = []
    probsparse_times = []
    
    batch_size = 8
    d_model = 64
    
    for seq_len in seq_lengths:
        Q = K = V = torch.randn(batch_size, seq_len, d_model)
    
        # Standard Attention
        start = time.time()
        _ = standard_attention(Q, K, V)
        standard_times.append(time.time() - start)
    
        # ProbSparse Attention
        start = time.time()
        _ = probsparse_attention(Q, K, V)
        probsparse_times.append(time.time() - start)
    
        print(f"Seq={seq_len}: Standard={standard_times[-1]:.4f}s, "
              f"ProbSparse={probsparse_times[-1]:.4f}s")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, standard_times, 'o-', label='Standard Attention', linewidth=2)
    plt.plot(seq_lengths, probsparse_times, 's-', label='ProbSparse Attention', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Mechanism Efficiency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('attention_efficiency.png', dpi=150)
    plt.close()
    
    speedup = [s/p for s, p in zip(standard_times, probsparse_times)]
    print(f"\nAverage speedup: {np.mean(speedup):.2f}x")
    

**比較結果** ：

  * 系列長が長いほど、ProbSparseの優位性が顕著
  * $L=2000$では約5-10倍の高速化
  * メモリ使用量も大幅に削減

**問題4：Multi-horizon Prediction** \- Autoregressiveモードと Direct モードの違いを実装し、比較してください。

**解答例** ：
    
    
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    
    class AutoregressivePredictor(nn.Module):
        """Autoregressive multi-step prediction"""
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim)
    
        def forward(self, x, n_steps):
            # x: (batch, seq_len, input_dim)
            predictions = []
    
            for _ in range(n_steps):
                out, _ = self.rnn(x)
                pred = self.fc(out[:, -1:, :])  # 最後のステップを予測
                predictions.append(pred)
                x = torch.cat([x, pred], dim=1)  # 予測を入力に追加
    
            return torch.cat(predictions, dim=1)
    
    
    class DirectPredictor(nn.Module):
        """Direct multi-step prediction"""
        def __init__(self, input_dim, hidden_dim, n_steps):
            super().__init__()
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim * n_steps)
            self.n_steps = n_steps
            self.input_dim = input_dim
    
        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            out, _ = self.rnn(x)
            pred = self.fc(out[:, -1, :])  # 全ステップを一度に予測
            return pred.view(-1, self.n_steps, self.input_dim)
    
    # テストデータ
    def generate_test_data(n_samples=100):
        t = np.linspace(0, 10, n_samples)
        data = np.sin(t) + 0.1 * np.random.randn(n_samples)
        return torch.FloatTensor(data).unsqueeze(-1)
    
    # 訓練
    seq_len = 20
    n_steps = 10
    data = generate_test_data(200)
    
    X_train = torch.stack([data[i:i+seq_len] for i in range(150)])
    y_train = torch.stack([data[i+seq_len:i+seq_len+n_steps] for i in range(150)])
    
    # モデル
    auto_model = AutoregressivePredictor(1, 32)
    direct_model = DirectPredictor(1, 32, n_steps)
    
    criterion = nn.MSELoss()
    
    # Autoregressive訓練
    optimizer = torch.optim.Adam(auto_model.parameters())
    for epoch in range(100):
        pred = auto_model(X_train, n_steps)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Direct訓練
    optimizer = torch.optim.Adam(direct_model.parameters())
    for epoch in range(100):
        pred = direct_model(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 評価
    X_test = X_train[0:1]
    y_test = y_train[0:1]
    
    auto_pred = auto_model(X_test, n_steps).detach().numpy()
    direct_pred = direct_model(X_test).detach().numpy()
    
    # 可視化
    plt.figure(figsize=(12, 5))
    plt.plot(range(seq_len), X_test[0].numpy(), 'o-', label='Input')
    plt.plot(range(seq_len, seq_len+n_steps), y_test[0].numpy(), 's-', label='True')
    plt.plot(range(seq_len, seq_len+n_steps), auto_pred[0], '^-', label='Autoregressive')
    plt.plot(range(seq_len, seq_len+n_steps), direct_pred[0], 'd-', label='Direct')
    plt.axvline(x=seq_len, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('multihorizon_comparison.png', dpi=150)
    plt.close()
    
    print("Autoregressive: 逐次予測、誤差が蓄積")
    print("Direct: 一度に予測、並列計算可能")
    

**問題5：Production Deployment** \- TFTモデルを本番環境にデプロイするための完全なパイプラインを設計してください。

**解答例** ：
    
    
    """
    本番環境デプロイパイプライン
    
    1. モデルの保存とバージョン管理
    2. 推論APIの構築
    3. モニタリングとロギング
    4. 自動再訓練パイプライン
    """
    
    # 1. モデルの保存
    class ModelManager:
        def __init__(self, model_dir="models"):
            self.model_dir = model_dir
    
        def save_model(self, model, version):
            import joblib
            path = f"{self.model_dir}/tft_v{version}.pkl"
            joblib.dump(model, path)
            print(f"Model saved: {path}")
    
        def load_latest_model(self):
            import glob, joblib
            models = glob.glob(f"{self.model_dir}/tft_v*.pkl")
            latest = max(models, key=lambda x: int(x.split('v')[-1].split('.')[0]))
            return joblib.load(latest)
    
    # 2. FastAPI推論サーバー
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import pandas as pd
    
    app = FastAPI()
    
    class PredictionRequest(BaseModel):
        store_id: str
        historical_data: list
        future_covariates: dict
    
    class PredictionResponse(BaseModel):
        predictions: list
        confidence_intervals: dict
        variable_importance: dict
    
    @app.on_event("startup")
    async def load_model():
        global model
        manager = ModelManager()
        model = manager.load_latest_model()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        try:
            # データ準備
            df = pd.DataFrame(request.historical_data)
            dataset = prepare_dataset(df, request.future_covariates)
    
            # 予測
            predictions = model.predict(dataset)
            interpretation = model.interpret_output(predictions)
    
            return PredictionResponse(
                predictions=predictions.tolist(),
                confidence_intervals={
                    "lower": predictions.quantile(0.1).tolist(),
                    "upper": predictions.quantile(0.9).tolist()
                },
                variable_importance=interpretation["encoder_variables"].tolist()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # 3. モニタリング
    import logging
    from prometheus_client import Counter, Histogram
    
    prediction_counter = Counter('predictions_total', 'Total predictions')
    prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
    
    @app.middleware("http")
    async def monitor_requests(request, call_next):
        with prediction_latency.time():
            response = await call_next(request)
        prediction_counter.inc()
        return response
    
    # 4. 自動再訓練
    class AutoRetrainer:
        def __init__(self, threshold_mae=50.0):
            self.threshold_mae = threshold_mae
    
        def check_performance(self, predictions, actuals):
            mae = np.mean(np.abs(predictions - actuals))
    
            if mae > self.threshold_mae:
                print(f"Performance degraded: MAE={mae:.2f}")
                self.trigger_retraining()
    
        def trigger_retraining(self):
            # 再訓練ジョブをスケジュール
            import subprocess
            subprocess.run(["python", "train_tft.py"])
    
    # 5. デプロイスクリプト
    def deploy_pipeline():
        # Docker化
        dockerfile = """
        FROM python:3.9
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
        """
    
        # Kubernetes manifest
        k8s_deployment = """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: tft-predictor
        spec:
          replicas: 3
          template:
            spec:
              containers:
              - name: tft
                image: tft-predictor:latest
                resources:
                  limits:
                    memory: "2Gi"
                    cpu: "1000m"
        """
    
        print("Deployment configuration generated")
    
    if __name__ == "__main__":
        deploy_pipeline()
    

**デプロイ要素** ：

  * モデルバージョン管理とロールバック機能
  * 高速な推論API（FastAPI）
  * メトリクスモニタリング（Prometheus）
  * 自動再訓練パイプライン
  * コンテナ化とオーケストレーション（Docker/K8s）

* * *

## 参考文献

### 論文

  * Vaswani et al. (2017). _Attention Is All You Need_. NeurIPS.
  * Lim et al. (2021). _Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting_. International Journal of Forecasting.
  * Zhou et al. (2021). _Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting_. AAAI.
  * Wu et al. (2021). _Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting_. NeurIPS.
  * Zhou et al. (2022). _FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting_. ICML.
  * Nie et al. (2023). _A Time Series is Worth 64 Words: Long-term Forecasting with Transformers_. ICLR (PatchTST).

### 書籍

  * Hyndman & Athanasopoulos. _Forecasting: Principles and Practice_ (3rd edition).
  * Nielsen, A. _Practical Time Series Analysis_. O'Reilly Media.

### ライブラリとツール

  * [PyTorch Forecasting](<https://pytorch-forecasting.readthedocs.io/>) \- TFT、N-BEATS等の実装
  * [Autoformer GitHub](<https://github.com/thuml/Autoformer>) \- 公式実装
  * [FEDformer GitHub](<https://github.com/MAZiqing/FEDformer>) \- 公式実装
  * [PatchTST GitHub](<https://github.com/yuqinie98/PatchTST>) \- 公式実装

### オンラインリソース

  * [HuggingFace Time Series Guide](<https://huggingface.co/blog/time-series-transformers>)
  * [Temporal Fusion Transformer Tutorial](<https://towardsdatascience.com/temporal-fusion-transformer-for-time-series-forecasting-9e9b8c47f0a5>)
  * [Time Series Transformers Survey](<https://arxiv.org/abs/2310.06625>) \- 最新サーベイ論文

* * *
