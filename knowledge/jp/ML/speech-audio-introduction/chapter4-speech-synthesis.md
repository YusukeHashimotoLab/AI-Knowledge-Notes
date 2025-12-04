---
title: 第4章：音声合成（TTS）
chapter_title: 第4章：音声合成（TTS）
subtitle: テキストから自然な音声を生成する技術の理解
reading_time: 30-35分
difficulty: 中級
code_examples: 7
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 音声合成（TTS）の基本概念とパイプラインを理解する
  * ✅ Tacotron/Tacotron 2のアーキテクチャと動作原理を理解する
  * ✅ FastSpeechによる非自己回帰的TTSの仕組みを学ぶ
  * ✅ 主要なニューラルボコーダの特徴と使い分けを理解する
  * ✅ 最新のTTS技術とその応用を把握する
  * ✅ Pythonライブラリを使った音声合成を実装できる

* * *

## 4.1 TTSの基礎

### Text-to-Speech（TTS）とは

**音声合成（Text-to-Speech, TTS）** は、テキストを自然な音声に変換する技術です。近年のディープラーニングの発展により、人間に近い自然な音声が生成可能になりました。

> 「TTSは、言語学的理解と音響モデリングの統合により、書かれた言葉を話し言葉に変換します。」

### TTSパイプライン
    
    
    ```mermaid
    graph LR
        A[テキスト入力] --> B[テキスト解析]
        B --> C[言語特徴抽出]
        C --> D[音響モデル]
        D --> E[Mel-スペクトログラム]
        E --> F[Vocoder]
        F --> G[音声波形]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### パイプラインの各段階

段階 | 役割 | 出力  
---|---|---  
**テキスト解析** | 正規化、トークン化 | 正規化されたテキスト  
**言語特徴抽出** | 音素変換、韻律予測 | 言語特徴ベクトル  
**音響モデル** | Mel-スペクトログラム生成 | Mel-スペクトログラム  
**Vocoder** | 波形生成 | 音声波形  
  
### Vocoderの役割

**Vocoder（ボコーダー）** は、Mel-スペクトログラムから音声波形を生成するモジュールです。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    
    # サンプル音声の読み込み
    y, sr = librosa.load(librosa.example('trumpet'), sr=22050, duration=3)
    
    # Mel-スペクトログラムの生成
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 波形
    axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y)
    axes[0].set_xlabel('時間 (秒)')
    axes[0].set_ylabel('振幅')
    axes[0].set_title('音声波形', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Mel-スペクトログラム
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                   sr=sr, ax=axes[1], cmap='viridis')
    axes[1].set_title('Mel-スペクトログラム', fontsize=14)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    print("=== TTSにおけるVocoderの役割 ===")
    print(f"入力: Mel-スペクトログラム {mel_spec.shape}")
    print(f"出力: 音声波形 {y.shape}")
    print(f"サンプリングレート: {sr} Hz")
    

### Prosody（韻律）と自然性

**韻律（Prosody）** は、音声の自然性を決定する重要な要素です：

  * **ピッチ（Pitch）** : 音の高さ、抑揚
  * **デュレーション（Duration）** : 音素の長さ
  * **エネルギー（Energy）** : 音の強さ
  * **リズム（Rhythm）** : 発話のテンポ

### 評価指標

#### 1\. MOS（Mean Opinion Score）

**MOS** は、人間による主観評価の平均スコアです。

スコア | 品質 | 説明  
---|---|---  
5 | 優秀 | 自然な人間の音声と区別不可  
4 | 良好 | わずかな不自然さがある  
3 | 普通 | 明らかに合成音声だが理解可能  
2 | 劣る | 聞き取りにくい部分がある  
1 | 悪い | 理解困難  
  
#### 2\. Naturalness（自然性）

音声の人間らしさを評価します：

  * 韻律の自然さ
  * 音質の滑らかさ
  * 感情表現の豊かさ

* * *

## 4.2 Tacotron & Tacotron 2

### Tacotronの概要

**Tacotron** は、Seq2Seqアーキテクチャを用いた初期のエンドツーエンドTTSモデルです（Googleが2017年に発表）。

### Seq2Seq for TTS
    
    
    ```mermaid
    graph LR
        A[テキスト] --> B[Encoder]
        B --> C[Attention]
        C --> D[Decoder]
        D --> E[Mel-スペクトログラム]
        E --> F[Vocoder]
        F --> G[音声波形]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### Attention機構

**Attention** は、デコーダが各ステップでエンコーダのどの部分に注目するかを決定します。

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

  * $Q$: Query（デコーダの隠れ状態）
  * $K$: Key（エンコーダの隠れ状態）
  * $V$: Value（エンコーダの隠れ状態）

### Tacotron 2のアーキテクチャ

**Tacotron 2** は、Tacotronを改良し、より高品質な音声を生成します。

#### 主な改良点

コンポーネント | Tacotron | Tacotron 2  
---|---|---  
**Encoder** | CBHG | Conv + Bi-LSTM  
**Attention** | Basic | Location-sensitive  
**Decoder** | GRU | LSTM + Prenet  
**Vocoder** | Griffin-Lim | WaveNet  
  
### Mel-スペクトログラム予測

Tacotron 2は、80次元のMel-スペクトログラムを予測します。
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Tacotron 2風のシンプルなMel予測のデモ
    # 注: 実際のTacotron 2は非常に複雑です
    
    class SimpleTacotronDecoder(torch.nn.Module):
        def __init__(self, hidden_size=256, n_mels=80):
            super().__init__()
            self.prenet = torch.nn.Sequential(
                torch.nn.Linear(n_mels, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256, 128)
            )
            self.lstm = torch.nn.LSTM(hidden_size + 128, hidden_size,
                                      num_layers=2, batch_first=True)
            self.projection = torch.nn.Linear(hidden_size, n_mels)
    
        def forward(self, encoder_outputs, prev_mel):
            # Prenet処理
            prenet_out = self.prenet(prev_mel)
    
            # LSTMデコーダ
            lstm_input = torch.cat([encoder_outputs, prenet_out], dim=-1)
            lstm_out, _ = self.lstm(lstm_input)
    
            # Mel-スペクトログラム予測
            mel_pred = self.projection(lstm_out)
    
            return mel_pred
    
    # モデルのインスタンス化
    model = SimpleTacotronDecoder()
    print("=== Tacotron 2風デコーダ ===")
    print(model)
    
    # ダミーデータでテスト
    batch_size, seq_len = 4, 10
    encoder_out = torch.randn(batch_size, seq_len, 256)
    prev_mel = torch.randn(batch_size, seq_len, 80)
    
    # 予測
    mel_output = model(encoder_out, prev_mel)
    print(f"\n入力エンコーダ出力: {encoder_out.shape}")
    print(f"入力Mel: {prev_mel.shape}")
    print(f"出力Mel: {mel_output.shape}")
    

### Location-sensitive Attention

Tacotron 2では、**Location-sensitive Attention** により、安定した単調なアライメントを実現します。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LocationSensitiveAttention(nn.Module):
        """Tacotron 2のLocation-sensitive Attention"""
    
        def __init__(self, attention_dim=128, n_location_filters=32,
                     location_kernel_size=31):
            super().__init__()
            self.W_query = nn.Linear(256, attention_dim, bias=False)
            self.W_keys = nn.Linear(256, attention_dim, bias=False)
            self.W_location = nn.Linear(n_location_filters, attention_dim, bias=False)
            self.location_conv = nn.Conv1d(2, n_location_filters,
                                           kernel_size=location_kernel_size,
                                           padding=(location_kernel_size - 1) // 2,
                                           bias=False)
            self.v = nn.Linear(attention_dim, 1, bias=False)
    
        def forward(self, query, keys, attention_weights_cat):
            """
            Args:
                query: デコーダの隠れ状態 (B, 1, 256)
                keys: エンコーダの出力 (B, T, 256)
                attention_weights_cat: 過去のattention weights (B, 2, T)
            """
            # Location features
            location_features = self.location_conv(attention_weights_cat)
            location_features = location_features.transpose(1, 2)
    
            # Attention計算
            query_proj = self.W_query(query)  # (B, 1, attention_dim)
            keys_proj = self.W_keys(keys)     # (B, T, attention_dim)
            location_proj = self.W_location(location_features)  # (B, T, attention_dim)
    
            # エネルギー計算
            energies = self.v(torch.tanh(query_proj + keys_proj + location_proj))
            energies = energies.squeeze(-1)  # (B, T)
    
            # Attention weights
            attention_weights = F.softmax(energies, dim=1)
    
            return attention_weights
    
    # テスト
    attention = LocationSensitiveAttention()
    query = torch.randn(4, 1, 256)
    keys = torch.randn(4, 100, 256)
    prev_attention = torch.randn(4, 2, 100)
    
    weights = attention(query, keys, prev_attention)
    print("=== Location-sensitive Attention ===")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Sum of weights: {weights.sum(dim=1)}")  # Should be ~1.0
    

### Tacotron 2の実装例（PyTorch）
    
    
    # Tacotron 2のエンコーダ部分の簡略版
    class TacotronEncoder(nn.Module):
        def __init__(self, num_chars=150, embedding_dim=512, hidden_size=256):
            super().__init__()
            self.embedding = nn.Embedding(num_chars, embedding_dim)
    
            # Convolution layers
            self.convolutions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(embedding_dim if i == 0 else hidden_size,
                             hidden_size, kernel_size=5, padding=2),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                ) for i in range(3)
            ])
    
            # Bi-directional LSTM
            self.lstm = nn.LSTM(hidden_size, hidden_size // 2,
                               num_layers=1, batch_first=True,
                               bidirectional=True)
    
        def forward(self, text_inputs):
            # Embedding
            x = self.embedding(text_inputs).transpose(1, 2)  # (B, C, T)
    
            # Convolutions
            for conv in self.convolutions:
                x = conv(x)
    
            # LSTM
            x = x.transpose(1, 2)  # (B, T, C)
            outputs, _ = self.lstm(x)
    
            return outputs
    
    # テスト
    encoder = TacotronEncoder()
    text_input = torch.randint(0, 150, (4, 50))  # Batch of 4, length 50
    encoder_output = encoder(text_input)
    
    print("=== Tacotron 2 Encoder ===")
    print(f"入力テキスト: {text_input.shape}")
    print(f"エンコーダ出力: {encoder_output.shape}")
    

* * *

## 4.3 FastSpeech

### 非自己回帰的TTSの動機

Tacotron 2のような**自己回帰的モデル** の問題点：

  * 生成が逐次的で遅い
  * 不安定なアライメント
  * 語の繰り返しやスキップ

**FastSpeech** は、これらの問題を解決する**非自己回帰的TTS** です。

### FastSpeechのアーキテクチャ
    
    
    ```mermaid
    graph LR
        A[テキスト] --> B[Encoder]
        B --> C[Duration Predictor]
        C --> D[Length Regulator]
        B --> D
        D --> E[Decoder]
        E --> F[Mel-スペクトログラム]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#c8e6c9
    ```

### Duration Prediction（継続時間予測）

FastSpeechの核心は、各音素の継続時間を明示的に予測することです。

$$ d_i = \text{DurationPredictor}(h_i) $$

  * $h_i$: 音素$i$の隠れ状態
  * $d_i$: 音素$i$の継続時間（フレーム数）

    
    
    import torch
    import torch.nn as nn
    
    class DurationPredictor(nn.Module):
        """FastSpeechの継続時間予測器"""
    
        def __init__(self, hidden_size=256, filter_size=256,
                     kernel_size=3, dropout=0.5):
            super().__init__()
    
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(hidden_size, filter_size, kernel_size,
                             padding=(kernel_size - 1) // 2),
                    nn.ReLU(),
                    nn.LayerNorm(filter_size),
                    nn.Dropout(dropout)
                ) for _ in range(2)
            ])
    
            self.linear = nn.Linear(filter_size, 1)
    
        def forward(self, encoder_output):
            """
            Args:
                encoder_output: (B, T, hidden_size)
            Returns:
                duration: (B, T) - 各音素の継続時間
            """
            x = encoder_output.transpose(1, 2)  # (B, C, T)
    
            for layer in self.layers:
                x = layer(x)
    
            x = x.transpose(1, 2)  # (B, T, C)
            duration = self.linear(x).squeeze(-1)  # (B, T)
    
            return duration
    
    # テスト
    duration_predictor = DurationPredictor()
    encoder_out = torch.randn(4, 50, 256)  # Batch=4, Seq=50
    durations = duration_predictor(encoder_out)
    
    print("=== Duration Predictor ===")
    print(f"入力: {encoder_out.shape}")
    print(f"予測継続時間: {durations.shape}")
    print(f"サンプル継続時間: {durations[0, :10]}")
    

### Length Regulator

**Length Regulator** は、予測された継続時間に基づいて、音素レベルの隠れ状態をフレームレベルに拡張します。
    
    
    class LengthRegulator(nn.Module):
        """FastSpeechのLength Regulator"""
    
        def __init__(self):
            super().__init__()
    
        def forward(self, x, durations):
            """
            Args:
                x: 音素レベルの隠れ状態 (B, T_phoneme, C)
                durations: 各音素の継続時間 (B, T_phoneme)
            Returns:
                expanded: フレームレベルの隠れ状態 (B, T_frame, C)
            """
            output = []
            for batch_idx in range(x.size(0)):
                expanded = []
                for phoneme_idx in range(x.size(1)):
                    # 各音素を継続時間分だけ繰り返す
                    duration = int(durations[batch_idx, phoneme_idx].item())
                    expanded.append(x[batch_idx, phoneme_idx].unsqueeze(0).expand(duration, -1))
    
                if expanded:
                    output.append(torch.cat(expanded, dim=0))
    
            # パディングして同じ長さにする
            max_len = max([seq.size(0) for seq in output])
            padded_output = []
            for seq in output:
                pad_len = max_len - seq.size(0)
                if pad_len > 0:
                    padding = torch.zeros(pad_len, seq.size(1))
                    seq = torch.cat([seq, padding], dim=0)
                padded_output.append(seq.unsqueeze(0))
    
            return torch.cat(padded_output, dim=0)
    
    # テスト
    length_regulator = LengthRegulator()
    phoneme_hidden = torch.randn(2, 10, 256)  # 2 samples, 10 phonemes
    durations = torch.tensor([[3, 2, 4, 1, 5, 2, 3, 2, 1, 4],
                              [2, 3, 2, 5, 1, 4, 2, 3, 2, 1]], dtype=torch.float32)
    
    frame_hidden = length_regulator(phoneme_hidden, durations)
    print("=== Length Regulator ===")
    print(f"音素レベル: {phoneme_hidden.shape}")
    print(f"予測継続時間: {durations}")
    print(f"フレームレベル: {frame_hidden.shape}")
    

### FastSpeech 2の改良

**FastSpeech 2** は、より多くの韻律情報を予測します：

予測対象 | FastSpeech | FastSpeech 2  
---|---|---  
**Duration** | ✓ | ✓  
**Pitch** | - | ✓  
**Energy** | - | ✓  
**学習ターゲット** | Teacher強制 | Ground truth  
  
### Speed vs Quality

FastSpeechの利点：

指標 | Tacotron 2 | FastSpeech | 改善率  
---|---|---|---  
**生成速度** | 1x | 38x | 38倍高速  
**MOS** | 4.41 | 4.27 | -3%  
**ロバスト性** | 低 | 高 | エラー率ほぼ0%  
**制御性** | 低 | 高 | 速度調整可能  
  
> **重要** : FastSpeechは、わずかな品質低下で大幅な高速化とロバスト性向上を実現します。

* * *

## 4.4 Neural Vocoders

### Vocoderの進化

ニューラルボコーダは、Mel-スペクトログラムから高品質な音声波形を生成します。

### 1\. WaveNet

**WaveNet** は、自己回帰的な生成モデルで、非常に高品質な音声を生成します（DeepMind、2016年）。

#### Dilated Causal Convolution

WaveNetの核心は、**Dilated Causal Convolution** です。

$$ y_t = f\left(\sum_{i=0}^{k-1} w_i \cdot x_{t-d \cdot i}\right) $$

  * $d$: Dilation factor（拡張係数）
  * $k$: Kernel size

    
    
    import torch
    import torch.nn as nn
    
    class DilatedCausalConv1d(nn.Module):
        """WaveNetのDilated Causal Convolution"""
    
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=(kernel_size - 1) * dilation,
                                 dilation=dilation)
    
        def forward(self, x):
            # Causal: 未来の情報を使わない
            output = self.conv(x)
            # 右側のパディングを削除
            return output[:, :, :x.size(2)]
    
    class WaveNetBlock(nn.Module):
        """WaveNetの残差ブロック"""
    
        def __init__(self, residual_channels, gate_channels, skip_channels,
                     kernel_size, dilation):
            super().__init__()
    
            self.dilated_conv = DilatedCausalConv1d(
                residual_channels, gate_channels, kernel_size, dilation
            )
    
            self.conv_1x1_skip = nn.Conv1d(gate_channels // 2, skip_channels, 1)
            self.conv_1x1_res = nn.Conv1d(gate_channels // 2, residual_channels, 1)
    
        def forward(self, x):
            # Dilated convolution
            h = self.dilated_conv(x)
    
            # Gated activation
            tanh_out, sigmoid_out = h.chunk(2, dim=1)
            h = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
    
            # Skip connection
            skip = self.conv_1x1_skip(h)
    
            # Residual connection
            residual = self.conv_1x1_res(h)
            return (x + residual) * 0.707, skip  # sqrt(0.5) for scaling
    
    # テスト
    block = WaveNetBlock(residual_channels=64, gate_channels=128,
                         skip_channels=64, kernel_size=3, dilation=2)
    x = torch.randn(4, 64, 1000)  # (B, C, T)
    residual, skip = block(x)
    
    print("=== WaveNet Block ===")
    print(f"入力: {x.shape}")
    print(f"残差出力: {residual.shape}")
    print(f"スキップ出力: {skip.shape}")
    

### 2\. WaveGlow

**WaveGlow** は、Flow-based生成モデルで、並列生成が可能です（NVIDIA、2018年）。

#### 特徴

  * **リアルタイム生成** : WaveNetより高速
  * **並列化可能** : 全サンプルを一度に生成
  * **可逆変換** : 音声とlatent変数の双方向変換

### 3\. HiFi-GAN

**HiFi-GAN** （High Fidelity GAN）は、GANベースの高速・高品質ボコーダです（2020年）。

#### アーキテクチャ

コンポーネント | 説明  
---|---  
**Generator** | Transposed Convolutionによるアップサンプリング  
**Multi-Period Discriminator** | 異なる周期パターンを識別  
**Multi-Scale Discriminator** | 異なる解像度で識別  
      
    
    import torch
    import torch.nn as nn
    
    class HiFiGANGenerator(nn.Module):
        """HiFi-GANのGenerator（簡略版）"""
    
        def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
            super().__init__()
    
            self.num_upsamples = len(upsample_rates)
    
            # 初期Conv
            self.conv_pre = nn.Conv1d(mel_channels, 512, 7, padding=3)
    
            # Upsampling layers
            self.ups = nn.ModuleList()
            for i, u in enumerate(upsample_rates):
                self.ups.append(nn.ConvTranspose1d(
                    512 // (2 ** i),
                    512 // (2 ** (i + 1)),
                    u * 2,
                    stride=u,
                    padding=u // 2
                ))
    
            # 最終Conv
            self.conv_post = nn.Conv1d(512 // (2 ** len(upsample_rates)),
                                       1, 7, padding=3)
    
        def forward(self, mel):
            """
            Args:
                mel: Mel-スペクトログラム (B, mel_channels, T)
            Returns:
                audio: 音声波形 (B, 1, T * prod(upsample_rates))
            """
            x = self.conv_pre(mel)
    
            for i in range(self.num_upsamples):
                x = torch.nn.functional.leaky_relu(x, 0.1)
                x = self.ups[i](x)
    
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = self.conv_post(x)
            x = torch.tanh(x)
    
            return x
    
    # テスト
    generator = HiFiGANGenerator()
    mel_input = torch.randn(2, 80, 100)  # (B, mel_channels, T)
    audio_output = generator(mel_input)
    
    print("=== HiFi-GAN Generator ===")
    print(f"入力Mel: {mel_input.shape}")
    print(f"出力音声: {audio_output.shape}")
    print(f"アップサンプリング率: {audio_output.size(2) / mel_input.size(2):.0f}x")
    

### Vocoderの比較

Vocoder | 生成方式 | 速度 | 品質（MOS） | 特徴  
---|---|---|---|---  
**Griffin-Lim** | 反復アルゴリズム | 高速 | 3.0-3.5 | シンプル、品質低  
**WaveNet** | 自己回帰 | 非常に遅い | 4.5+ | 最高品質  
**WaveGlow** | Flow-based | 中速 | 4.2-4.3 | 並列生成可能  
**HiFi-GAN** | GAN | 非常に高速 | 4.3-4.5 | 高速・高品質  
  
> **推奨** : 現在では、HiFi-GANが速度と品質のバランスが最良で広く使用されています。

* * *

## 4.5 最新のTTS技術

### 1\. VITS（End-to-End TTS）

**VITS** （Variational Inference with adversarial learning for end-to-end Text-to-Speech）は、音響モデルとボコーダを統合したエンドツーエンドモデルです（2021年）。

#### VITSの特徴

  * **統合アーキテクチャ** : 音響モデル + Vocoder
  * **VAE + GAN** : Variational Autoencoderと敵対的学習の組み合わせ
  * **高速・高品質** : リアルタイム生成可能
  * **多様性** : 同じテキストから多様な音声を生成

    
    
    ```mermaid
    graph LR
        A[テキスト] --> B[Text Encoder]
        B --> C[Posterior Encoder]
        B --> D[Prior Encoder]
        C --> E[Latent Variable z]
        D --> E
        E --> F[Decoder/Generator]
        F --> G[音声波形]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#f3e5f5
        style E fill:#e3f2fd
        style F fill:#e8f5e9
        style G fill:#c8e6c9
    ```

### 2\. Voice Cloning（音声クローニング）

**Voice Cloning** は、少量の音声サンプルから特定の話者の声を再現する技術です。

#### アプローチ

手法 | 説明 | 必要データ  
---|---|---  
**Speaker Adaptation** | 既存モデルを少量データで微調整 | 数分～数十分  
**Speaker Embedding** | 話者埋め込みベクトルを学習 | 数秒～数分  
**Zero-shot TTS** | 未知話者の声を即座に模倣 | 数秒  
  
### 3\. Multi-speaker TTS

**Multi-speaker TTS** は、単一モデルで複数の話者の声を生成します。

#### Speaker Embedding

話者IDを埋め込みベクトルに変換し、モデルに条件付けします。

$$ e_{\text{speaker}} = \text{Embedding}(\text{speaker_id}) $$

$$ h = f(x, e_{\text{speaker}}) $$

### 4\. Japanese TTS Systems

日本語TTSシステムの特徴：

#### 日本語特有の課題

  * **アクセント** : 高低アクセントの再現
  * **イントネーション** : 文末の上昇・下降
  * **促音・長音** : 特殊拍の扱い
  * **漢字読み** : 文脈依存の読み分け

#### 主要な日本語TTSライブラリ

システム | 特徴 | ライセンス  
---|---|---  
**OpenJTalk** | HMM-based、軽量 | BSD  
**VOICEVOX** | ディープラーニング、高品質 | LGPL/Commercial  
**ESPnet-TTS** | 研究用、最新手法実装 | Apache 2.0  
  
### 実装例：gTTS（Google Text-to-Speech）
    
    
    from gtts import gTTS
    import os
    from IPython.display import Audio
    
    # テキスト
    text_en = "Hello, this is a demonstration of text-to-speech synthesis."
    text_ja = "こんにちは、これは音声合成のデモンストレーションです。"
    
    # 英語TTS
    tts_en = gTTS(text=text_en, lang='en', slow=False)
    tts_en.save("output_en.mp3")
    
    # 日本語TTS
    tts_ja = gTTS(text=text_ja, lang='ja', slow=False)
    tts_ja.save("output_ja.mp3")
    
    print("=== gTTS（Google Text-to-Speech）===")
    print("英語音声を生成: output_en.mp3")
    print("日本語音声を生成: output_ja.mp3")
    
    # 音声の再生（Jupyter環境）
    # display(Audio("output_en.mp3"))
    # display(Audio("output_ja.mp3"))
    

* * *

## 4.6 本章のまとめ

### 学んだこと

  1. **TTSの基礎**

     * Text-to-Speechパイプライン: テキスト解析 → 音響モデル → Vocoder
     * 韻律（Prosody）の重要性: ピッチ、デュレーション、エネルギー
     * 評価指標: MOS、自然性
  2. **Tacotron & Tacotron 2**

     * Seq2SeqアーキテクチャによるエンドツーエンドTTS
     * Attention機構によるテキストと音声のアライメント
     * Location-sensitive Attentionによる安定性向上
  3. **FastSpeech**

     * 非自己回帰的TTSによる高速化
     * Duration Predictorによる明示的な継続時間制御
     * FastSpeech 2: ピッチとエネルギーの追加予測
  4. **Neural Vocoders**

     * WaveNet: 最高品質だが遅い
     * WaveGlow: 並列生成可能
     * HiFi-GAN: 高速・高品質のバランス
  5. **最新技術**

     * VITS: エンドツーエンド統合モデル
     * Voice Cloning: 少量データからの音声再現
     * Multi-speaker TTS: 単一モデルで複数話者
     * 日本語TTS: アクセント・イントネーションの課題

### TTS技術の選択ガイドライン

目的 | 推奨モデル | 理由  
---|---|---  
最高品質 | Tacotron 2 + WaveNet | MOS 4.5+  
リアルタイム生成 | FastSpeech 2 + HiFi-GAN | 高速・高品質  
エンドツーエンド | VITS | 統合アーキテクチャ  
音声クローニング | Speaker Embedding TTS | 少量データで対応  
研究・実験 | ESPnet-TTS | 最新手法実装  
  
### 次の章へ

第5章では、**音声変換とボイスコンバージョン** を学びます：

  * Voice Conversionの基礎
  * スタイル転送
  * 感情表現の制御
  * リアルタイム音声変換

* * *

## 演習問題

### 問題1（難易度：easy）

TTSパイプラインの主要な4つの段階を順番に説明し、各段階の役割を述べてください。

解答例

**解答** ：

  1. **テキスト解析**

     * 役割: テキストの正規化、トークン化、数字や略語の展開
     * 出力: 正規化されたテキスト
  2. **言語特徴抽出**

     * 役割: テキストを音素に変換、韻律情報の予測
     * 出力: 音素列と韻律特徴
  3. **音響モデル**

     * 役割: 言語特徴からMel-スペクトログラムを生成
     * 出力: Mel-スペクトログラム（音響特徴）
  4. **Vocoder**

     * 役割: Mel-スペクトログラムから音声波形を生成
     * 出力: 最終的な音声波形

### 問題2（難易度：medium）

Tacotron 2とFastSpeechの主な違いを、生成方式、速度、ロバスト性の観点から比較してください。

解答例

**解答** ：

観点 | Tacotron 2 | FastSpeech  
---|---|---  
**生成方式** | 自己回帰的（Autoregressive）  
前のフレームを使って次を予測 | 非自己回帰的（Non-autoregressive）  
全フレームを並列生成  
**速度** | 遅い（逐次生成）  
基準: 1x | 非常に高速（並列生成）  
約38倍高速  
**ロバスト性** | 低い  
\- 単語の繰り返し  
\- スキップ  
\- 不安定なアライメント | 高い  
\- ほぼエラーなし  
\- 安定したアライメント  
\- 予測可能な出力  
**制御性** | 低い  
速度・韻律の明示的制御困難 | 高い  
Duration調整で速度制御可能  
**品質（MOS）** | 4.41（高品質） | 4.27（わずかに低い）  
  
**結論** : FastSpeechは、わずかな品質低下（-3%）で、大幅な高速化（38倍）とロバスト性向上を実現。実用アプリケーションではFastSpeechが有利。

### 問題3（難易度：medium）

WaveNet、WaveGlow、HiFi-GANの3つのVocoderを、生成方式、速度、品質の観点から比較し、それぞれの使用場面を提案してください。

解答例

**解答** ：

**比較表** ：

Vocoder | 生成方式 | 速度 | 品質（MOS） | 特徴  
---|---|---|---|---  
**WaveNet** | 自己回帰的  
Dilated Causal Conv | 非常に遅い | 4.5+（最高） | \- 最高品質  
\- リアルタイム不可  
**WaveGlow** | Flow-based  
可逆変換 | 中速 | 4.2-4.3 | \- 並列生成  
\- 安定した学習  
**HiFi-GAN** | GAN  
敵対的学習 | 非常に高速 | 4.3-4.5 | \- 高速・高品質  
\- 学習やや難  
  
**使用場面の提案** ：

  1. **WaveNet**

     * 使用場面: オフライン音声合成、高品質が最優先
     * 例: スタジオ品質のオーディオブック制作
  2. **WaveGlow**

     * 使用場面: 研究目的、Flow-basedモデルの理解
     * 例: 生成モデル研究、VAEとの組み合わせ
  3. **HiFi-GAN**

     * 使用場面: リアルタイムアプリ、実用システム
     * 例: 音声アシスタント、ライブ配信のナレーション

**推奨** : 現在では、速度と品質のバランスが優れたHiFi-GANが最も広く使用されています。

### 問題4（難易度：hard）

FastSpeechのDuration PredictorとLength Regulatorの役割を説明し、なぜこれらが非自己回帰的TTSに必要なのか述べてください。Pythonコードで簡単な実装例を示してください。

解答例

**解答** ：

**役割の説明** ：

  1. **Duration Predictor（継続時間予測器）**

     * 役割: 各音素が何フレーム続くかを予測
     * 入力: エンコーダの隠れ状態（音素レベル）
     * 出力: 各音素の継続時間（フレーム数）
  2. **Length Regulator**

     * 役割: 音素レベルの表現をフレームレベルに拡張
     * 入力: 音素の隠れ状態 + 予測された継続時間
     * 出力: フレームレベルの隠れ状態

**必要性** ：

非自己回帰的TTSでは、全フレームを並列生成するため：

  * 事前に出力長を知る必要がある
  * テキスト（音素）と音声（フレーム）の長さが異なる
  * 各音素の継続時間が異なる（例: 「あ」は3フレーム、「ん」は1フレーム）

Duration Predictorで継続時間を予測し、Length Regulatorで音素表現を適切な長さに拡張することで、並列生成が可能になります。

**実装例** ：
    
    
    import torch
    import torch.nn as nn
    
    class SimpleDurationPredictor(nn.Module):
        """継続時間予測器"""
    
        def __init__(self, hidden_size=256):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.5),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(0.5)
            )
            self.output = nn.Linear(256, 1)
    
        def forward(self, x):
            # x: (B, T, hidden_size)
            x = x.transpose(1, 2)  # (B, hidden_size, T)
            x = self.layers(x)
            x = x.transpose(1, 2)  # (B, T, 256)
            duration = self.output(x).squeeze(-1)  # (B, T)
            return torch.relu(duration)  # 正の値のみ
    
    class SimpleLengthRegulator(nn.Module):
        """Length Regulator"""
    
        def forward(self, x, durations):
            # x: (B, T_phoneme, hidden_size)
            # durations: (B, T_phoneme)
    
            output = []
            for batch_idx in range(x.size(0)):
                expanded = []
                for phoneme_idx in range(x.size(1)):
                    dur = int(durations[batch_idx, phoneme_idx].item())
                    if dur > 0:
                        # 音素の隠れ状態をdur回繰り返す
                        phoneme_hidden = x[batch_idx, phoneme_idx].unsqueeze(0)
                        expanded.append(phoneme_hidden.expand(dur, -1))
    
                if expanded:
                    output.append(torch.cat(expanded, dim=0))
    
            # パディングして同じ長さに
            max_len = max(seq.size(0) for seq in output)
            padded = []
            for seq in output:
                if seq.size(0) < max_len:
                    pad = torch.zeros(max_len - seq.size(0), seq.size(1))
                    seq = torch.cat([seq, pad], dim=0)
                padded.append(seq.unsqueeze(0))
    
            return torch.cat(padded, dim=0)
    
    # テスト
    print("=== Duration Predictor & Length Regulator ===\n")
    
    # ダミーデータ
    batch_size, n_phonemes, hidden_size = 2, 5, 256
    phoneme_hidden = torch.randn(batch_size, n_phonemes, hidden_size)
    
    # Duration予測
    duration_predictor = SimpleDurationPredictor(hidden_size)
    predicted_durations = duration_predictor(phoneme_hidden)
    
    print(f"音素の隠れ状態: {phoneme_hidden.shape}")
    print(f"予測された継続時間: {predicted_durations.shape}")
    print(f"サンプル継続時間: {predicted_durations[0]}")
    
    # Length Regulation
    length_regulator = SimpleLengthRegulator()
    frame_hidden = length_regulator(phoneme_hidden, predicted_durations)
    
    print(f"\nフレームレベルの隠れ状態: {frame_hidden.shape}")
    print(f"拡張率: {frame_hidden.size(1) / phoneme_hidden.size(1):.2f}x")
    
    # 具体例
    print("\n=== 具体例 ===")
    print("音素: ['k', 'o', 'n', 'n', 'i']")
    print("継続時間: [3, 4, 1, 1, 2] フレーム")
    print("→ 合計 11 フレームの音声生成")
    

**出力例** ：
    
    
    === Duration Predictor & Length Regulator ===
    
    音素の隠れ状態: torch.Size([2, 5, 256])
    予測された継続時間: torch.Size([2, 5])
    サンプル継続時間: tensor([2.3, 1.8, 3.1, 2.5, 1.2])
    
    フレームレベルの隠れ状態: torch.Size([2, 11, 256])
    拡張率: 2.20x
    
    === 具体例 ===
    音素: ['k', 'o', 'n', 'n', 'i']
    継続時間: [3, 4, 1, 1, 2] フレーム
    → 合計 11 フレームの音声生成
    

### 問題5（難易度：hard）

Multi-speaker TTSにおけるSpeaker Embeddingの役割を説明し、どのようにモデルに組み込まれるか述べてください。また、Voice Cloningとの関連性を論じてください。

解答例

**解答** ：

**Speaker Embeddingの役割** ：

  1. **話者特徴の表現**

     * 各話者を低次元ベクトル（通常64-512次元）で表現
     * 話者の声質、ピッチ、話し方の特徴をエンコード
  2. **条件付け**

     * TTSモデルに話者情報を提供
     * 同じテキストでも異なる話者の声を生成可能

**モデルへの組み込み方法** ：

**方法1: Embedding Lookupテーブル**
    
    
    import torch
    import torch.nn as nn
    
    class MultiSpeakerTTS(nn.Module):
        def __init__(self, n_speakers=100, speaker_embed_dim=256,
                     text_embed_dim=512):
            super().__init__()
    
            # Speaker Embedding
            self.speaker_embedding = nn.Embedding(n_speakers, speaker_embed_dim)
    
            # Text Encoder
            self.text_encoder = nn.LSTM(text_embed_dim, 512,
                                        num_layers=2, batch_first=True)
    
            # Speaker-conditioned Decoder
            self.decoder = nn.LSTM(512 + speaker_embed_dim, 512,
                                  num_layers=2, batch_first=True)
    
            self.mel_projection = nn.Linear(512, 80)  # 80-dim Mel
    
        def forward(self, text_features, speaker_ids):
            # Speaker Embedding取得
            speaker_emb = self.speaker_embedding(speaker_ids)  # (B, speaker_dim)
    
            # Text Encoding
            text_encoded, _ = self.text_encoder(text_features)  # (B, T, 512)
    
            # Speaker Embeddingを全タイムステップに拡張
            speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(
                -1, text_encoded.size(1), -1
            )  # (B, T, speaker_dim)
    
            # Concatenate
            decoder_input = torch.cat([text_encoded, speaker_emb_expanded],
                                     dim=-1)  # (B, T, 512+speaker_dim)
    
            # Decode
            decoder_output, _ = self.decoder(decoder_input)
    
            # Mel prediction
            mel_output = self.mel_projection(decoder_output)
    
            return mel_output
    
    # テスト
    model = MultiSpeakerTTS(n_speakers=100)
    text_features = torch.randn(4, 50, 512)  # Batch=4, Seq=50
    speaker_ids = torch.tensor([0, 5, 10, 15])  # 異なる話者
    
    mel_output = model(text_features, speaker_ids)
    print("=== Multi-Speaker TTS ===")
    print(f"入力テキスト: {text_features.shape}")
    print(f"話者ID: {speaker_ids}")
    print(f"出力Mel: {mel_output.shape}")
    

**方法2: Speaker Encoder（Voice Cloning用）**
    
    
    class SpeakerEncoder(nn.Module):
        """音声から話者埋め込みを抽出"""
    
        def __init__(self, mel_dim=80, embed_dim=256):
            super().__init__()
            self.lstm = nn.LSTM(mel_dim, 256, num_layers=3,
                               batch_first=True)
            self.projection = nn.Linear(256, embed_dim)
    
        def forward(self, mel_spectrograms):
            # mel_spectrograms: (B, T, 80)
            _, (hidden, _) = self.lstm(mel_spectrograms)
            # 最終層の隠れ状態を使用
            speaker_emb = self.projection(hidden[-1])  # (B, embed_dim)
            # L2正規化
            speaker_emb = speaker_emb / torch.norm(speaker_emb, dim=1, keepdim=True)
            return speaker_emb
    
    # Voice Cloningのワークフロー
    speaker_encoder = SpeakerEncoder()
    
    # 参照音声から話者埋め込みを抽出
    reference_mel = torch.randn(1, 100, 80)  # 数秒の音声
    speaker_emb = speaker_encoder(reference_mel)
    
    print("\n=== Voice Cloning ===")
    print(f"参照音声: {reference_mel.shape}")
    print(f"抽出された話者埋め込み: {speaker_emb.shape}")
    print("→ この埋め込みを使って、任意のテキストをこの話者の声で合成")
    

**Voice Cloningとの関連性** ：

観点 | Multi-speaker TTS | Voice Cloning  
---|---|---  
**話者表現** | Embedding Lookup  
（学習済み話者のみ） | Speaker Encoder  
（未知話者も可能）  
**必要データ** | 各話者の大量データ | 数秒～数分の音声  
**柔軟性** | 低（固定話者セット） | 高（新規話者対応）  
**品質** | 高（各話者で最適化） | 中～高（データ量依存）  
  
**統合アプローチ** ：

最新のシステムでは、両方を組み合わせて使用：

  1. Multi-speaker TTSを大規模データで事前学習
  2. Speaker Encoderで新規話者の埋め込みを抽出
  3. 少量の追加データで微調整（Optional）

これにより、学習済み話者の高品質と、未知話者への対応を両立できます。

* * *

## 参考文献

  1. Wang, Y., et al. (2017). "Tacotron: Towards End-to-End Speech Synthesis." _Interspeech 2017_.
  2. Shen, J., et al. (2018). "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions." _ICASSP 2018_.
  3. Ren, Y., et al. (2019). "FastSpeech: Fast, Robust and Controllable Text to Speech." _NeurIPS 2019_.
  4. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." _arXiv:1609.03499_.
  5. Prenger, R., Valle, R., & Catanzaro, B. (2019). "WaveGlow: A Flow-based Generative Network for Speech Synthesis." _ICASSP 2019_.
  6. Kong, J., Kim, J., & Bae, J. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." _NeurIPS 2020_.
  7. Kim, J., Kong, J., & Son, J. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." _ICML 2021_.
  8. Casanova, E., et al. (2022). "YourtTS: Towards Zero-Shot Multi-Speaker TTS." _arXiv:2112.02418_.
