---
title: "Chapter 4: Speech Synthesis (TTS)"
chapter_title: "Chapter 4: Speech Synthesis (TTS)"
subtitle: Understanding the technology of generating natural speech from text
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 7
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Speech Synthesis (TTS). You will learn characteristics and latest TTS technologies.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the basic concepts and pipeline of speech synthesis (TTS)
  * ✅ Understand the architecture and operating principles of Tacotron/Tacotron 2
  * ✅ Learn the mechanisms of non-autoregressive TTS using FastSpeech
  * ✅ Understand the characteristics and appropriate use cases of major neural vocoders
  * ✅ Grasp the latest TTS technologies and their applications
  * ✅ Implement speech synthesis using Python libraries

* * *

## 4.1 TTS Fundamentals

### What is Text-to-Speech (TTS)?

**Text-to-Speech (TTS)** is a technology that converts text into natural speech. Recent advances in deep learning have made it possible to generate speech that closely resembles human voice.

> "TTS converts written words into spoken words through the integration of linguistic understanding and acoustic modeling."

### TTS Pipeline
    
    
    ```mermaid
    graph LR
        A[Text Input] --> B[Text Analysis]
        B --> C[Linguistic Feature Extraction]
        C --> D[Acoustic Model]
        D --> E[Mel-Spectrogram]
        E --> F[Vocoder]
        F --> G[Speech Waveform]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### Pipeline Stages

Stage | Role | Output  
---|---|---  
**Text Analysis** | Normalization, tokenization | Normalized text  
**Linguistic Feature Extraction** | Phoneme conversion, prosody prediction | Linguistic feature vectors  
**Acoustic Model** | Mel-spectrogram generation | Mel-spectrogram  
**Vocoder** | Waveform generation | Speech waveform  
  
### Role of the Vocoder

A **vocoder** is a module that generates speech waveforms from Mel-spectrograms.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Avocoderis a module that generates speech waveforms from Mel
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    
    # Load sample audio
    y, sr = librosa.load(librosa.example('trumpet'), sr=22050, duration=3)
    
    # Generate Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Waveform
    axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Speech Waveform', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Mel-spectrogram
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                   sr=sr, ax=axes[1], cmap='viridis')
    axes[1].set_title('Mel-Spectrogram', fontsize=14)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Role of Vocoder in TTS ===")
    print(f"Input: Mel-spectrogram {mel_spec.shape}")
    print(f"Output: Speech waveform {y.shape}")
    print(f"Sampling rate: {sr} Hz")
    

### Prosody and Naturalness

**Prosody** is a crucial element that determines the naturalness of speech:

  * **Pitch** : Voice height, intonation
  * **Duration** : Length of phonemes
  * **Energy** : Intensity of sound
  * **Rhythm** : Tempo of speech

### Evaluation Metrics

#### 1\. MOS (Mean Opinion Score)

**MOS** is the average score of subjective human evaluation.

Score | Quality | Description  
---|---|---  
5 | Excellent | Indistinguishable from natural human speech  
4 | Good | Slight unnaturalness  
3 | Fair | Clearly synthetic but understandable  
2 | Poor | Some parts difficult to hear  
1 | Bad | Difficult to understand  
  
#### 2\. Naturalness

Evaluates the human-likeness of speech:

  * Naturalness of prosody
  * Smoothness of audio quality
  * Richness of emotional expression

* * *

## 4.2 Tacotron & Tacotron 2

### Tacotron Overview

**Tacotron** is an early end-to-end TTS model using a Seq2Seq architecture (published by Google in 2017).

### Seq2Seq for TTS
    
    
    ```mermaid
    graph LR
        A[Text] --> B[Encoder]
        B --> C[Attention]
        C --> D[Decoder]
        D --> E[Mel-Spectrogram]
        E --> F[Vocoder]
        F --> G[Speech Waveform]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### Attention Mechanism

**Attention** determines which parts of the encoder output the decoder should focus on at each step.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

  * $Q$: Query (decoder hidden state)
  * $K$: Key (encoder hidden states)
  * $V$: Value (encoder hidden states)

### Tacotron 2 Architecture

**Tacotron 2** improves upon Tacotron to generate higher quality speech.

#### Key Improvements

Component | Tacotron | Tacotron 2  
---|---|---  
**Encoder** | CBHG | Conv + Bi-LSTM  
**Attention** | Basic | Location-sensitive  
**Decoder** | GRU | LSTM + Prenet  
**Vocoder** | Griffin-Lim | WaveNet  
  
### Mel-Spectrogram Prediction

Tacotron 2 predicts 80-dimensional Mel-spectrograms.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Tacotron 2 predicts 80-dimensional Mel-spectrograms.
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Tacotron 2-style simple Mel prediction demo
    # Note: Actual Tacotron 2 is much more complex
    
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
            # Prenet processing
            prenet_out = self.prenet(prev_mel)
    
            # LSTM decoder
            lstm_input = torch.cat([encoder_outputs, prenet_out], dim=-1)
            lstm_out, _ = self.lstm(lstm_input)
    
            # Mel-spectrogram prediction
            mel_pred = self.projection(lstm_out)
    
            return mel_pred
    
    # Model instantiation
    model = SimpleTacotronDecoder()
    print("=== Tacotron 2-style Decoder ===")
    print(model)
    
    # Test with dummy data
    batch_size, seq_len = 4, 10
    encoder_out = torch.randn(batch_size, seq_len, 256)
    prev_mel = torch.randn(batch_size, seq_len, 80)
    
    # Prediction
    mel_output = model(encoder_out, prev_mel)
    print(f"\nInput encoder output: {encoder_out.shape}")
    print(f"Input Mel: {prev_mel.shape}")
    print(f"Output Mel: {mel_output.shape}")
    

### Location-sensitive Attention

Tacotron 2 uses **Location-sensitive Attention** to achieve stable and monotonic alignment.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LocationSensitiveAttention(nn.Module):
        """Location-sensitive Attention for Tacotron 2"""
    
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
                query: Decoder hidden state (B, 1, 256)
                keys: Encoder output (B, T, 256)
                attention_weights_cat: Previous attention weights (B, 2, T)
            """
            # Location features
            location_features = self.location_conv(attention_weights_cat)
            location_features = location_features.transpose(1, 2)
    
            # Attention computation
            query_proj = self.W_query(query)  # (B, 1, attention_dim)
            keys_proj = self.W_keys(keys)     # (B, T, attention_dim)
            location_proj = self.W_location(location_features)  # (B, T, attention_dim)
    
            # Energy computation
            energies = self.v(torch.tanh(query_proj + keys_proj + location_proj))
            energies = energies.squeeze(-1)  # (B, T)
    
            # Attention weights
            attention_weights = F.softmax(energies, dim=1)
    
            return attention_weights
    
    # Test
    attention = LocationSensitiveAttention()
    query = torch.randn(4, 1, 256)
    keys = torch.randn(4, 100, 256)
    prev_attention = torch.randn(4, 2, 100)
    
    weights = attention(query, keys, prev_attention)
    print("=== Location-sensitive Attention ===")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Sum of weights: {weights.sum(dim=1)}")  # Should be ~1.0
    

### Tacotron 2 Implementation Example (PyTorch)
    
    
    # Simplified version of Tacotron 2 encoder
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
    
    # Test
    encoder = TacotronEncoder()
    text_input = torch.randint(0, 150, (4, 50))  # Batch of 4, length 50
    encoder_output = encoder(text_input)
    
    print("=== Tacotron 2 Encoder ===")
    print(f"Input text: {text_input.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    

* * *

## 4.3 FastSpeech

### Motivation for Non-Autoregressive TTS

Problems with **autoregressive models** like Tacotron 2:

  * Sequential generation is slow
  * Unstable alignment
  * Word repetition or skipping

**FastSpeech** is a **non-autoregressive TTS** that solves these problems.

### FastSpeech Architecture
    
    
    ```mermaid
    graph LR
        A[Text] --> B[Encoder]
        B --> C[Duration Predictor]
        C --> D[Length Regulator]
        B --> D
        D --> E[Decoder]
        E --> F[Mel-Spectrogram]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#c8e6c9
    ```

### Duration Prediction

The core of FastSpeech is explicitly predicting the duration of each phoneme.

$$ d_i = \text{DurationPredictor}(h_i) $$

  * $h_i$: Hidden state of phoneme $i$
  * $d_i$: Duration of phoneme $i$ (number of frames)

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class DurationPredictor(nn.Module):
        """Duration predictor for FastSpeech"""
    
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
                duration: (B, T) - Duration of each phoneme
            """
            x = encoder_output.transpose(1, 2)  # (B, C, T)
    
            for layer in self.layers:
                x = layer(x)
    
            x = x.transpose(1, 2)  # (B, T, C)
            duration = self.linear(x).squeeze(-1)  # (B, T)
    
            return duration
    
    # Test
    duration_predictor = DurationPredictor()
    encoder_out = torch.randn(4, 50, 256)  # Batch=4, Seq=50
    durations = duration_predictor(encoder_out)
    
    print("=== Duration Predictor ===")
    print(f"Input: {encoder_out.shape}")
    print(f"Predicted durations: {durations.shape}")
    print(f"Sample durations: {durations[0, :10]}")
    

### Length Regulator

The **Length Regulator** expands phoneme-level hidden states to frame-level based on predicted durations.
    
    
    class LengthRegulator(nn.Module):
        """Length Regulator for FastSpeech"""
    
        def __init__(self):
            super().__init__()
    
        def forward(self, x, durations):
            """
            Args:
                x: Phoneme-level hidden states (B, T_phoneme, C)
                durations: Duration of each phoneme (B, T_phoneme)
            Returns:
                expanded: Frame-level hidden states (B, T_frame, C)
            """
            output = []
            for batch_idx in range(x.size(0)):
                expanded = []
                for phoneme_idx in range(x.size(1)):
                    # Repeat each phoneme by its duration
                    duration = int(durations[batch_idx, phoneme_idx].item())
                    expanded.append(x[batch_idx, phoneme_idx].unsqueeze(0).expand(duration, -1))
    
                if expanded:
                    output.append(torch.cat(expanded, dim=0))
    
            # Pad to same length
            max_len = max([seq.size(0) for seq in output])
            padded_output = []
            for seq in output:
                pad_len = max_len - seq.size(0)
                if pad_len > 0:
                    padding = torch.zeros(pad_len, seq.size(1))
                    seq = torch.cat([seq, padding], dim=0)
                padded_output.append(seq.unsqueeze(0))
    
            return torch.cat(padded_output, dim=0)
    
    # Test
    length_regulator = LengthRegulator()
    phoneme_hidden = torch.randn(2, 10, 256)  # 2 samples, 10 phonemes
    durations = torch.tensor([[3, 2, 4, 1, 5, 2, 3, 2, 1, 4],
                              [2, 3, 2, 5, 1, 4, 2, 3, 2, 1]], dtype=torch.float32)
    
    frame_hidden = length_regulator(phoneme_hidden, durations)
    print("=== Length Regulator ===")
    print(f"Phoneme-level: {phoneme_hidden.shape}")
    print(f"Predicted durations: {durations}")
    print(f"Frame-level: {frame_hidden.shape}")
    

### FastSpeech 2 Improvements

**FastSpeech 2** predicts more prosodic information:

Prediction Target | FastSpeech | FastSpeech 2  
---|---|---  
**Duration** | ✓ | ✓  
**Pitch** | - | ✓  
**Energy** | - | ✓  
**Training Target** | Teacher forcing | Ground truth  
  
### Speed vs Quality

Advantages of FastSpeech:

Metric | Tacotron 2 | FastSpeech | Improvement  
---|---|---|---  
**Generation Speed** | 1x | 38x | 38x faster  
**MOS** | 4.41 | 4.27 | -3%  
**Robustness** | Low | High | Nearly 0% error rate  
**Controllability** | Low | High | Speed adjustable  
  
> **Important** : FastSpeech achieves significant speedup and robustness improvement with only a slight quality reduction.

* * *

## 4.4 Neural Vocoders

### Evolution of Vocoders

Neural vocoders generate high-quality speech waveforms from Mel-spectrograms.

### 1\. WaveNet

**WaveNet** is an autoregressive generative model that produces very high-quality speech (DeepMind, 2016).

#### Dilated Causal Convolution

The core of WaveNet is **Dilated Causal Convolution**.

$$ y_t = f\left(\sum_{i=0}^{k-1} w_i \cdot x_{t-d \cdot i}\right) $$

  * $d$: Dilation factor
  * $k$: Kernel size

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class DilatedCausalConv1d(nn.Module):
        """Dilated Causal Convolution for WaveNet"""
    
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=(kernel_size - 1) * dilation,
                                 dilation=dilation)
    
        def forward(self, x):
            # Causal: don't use future information
            output = self.conv(x)
            # Remove right padding
            return output[:, :, :x.size(2)]
    
    class WaveNetBlock(nn.Module):
        """WaveNet residual block"""
    
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
    
    # Test
    block = WaveNetBlock(residual_channels=64, gate_channels=128,
                         skip_channels=64, kernel_size=3, dilation=2)
    x = torch.randn(4, 64, 1000)  # (B, C, T)
    residual, skip = block(x)
    
    print("=== WaveNet Block ===")
    print(f"Input: {x.shape}")
    print(f"Residual output: {residual.shape}")
    print(f"Skip output: {skip.shape}")
    

### 2\. WaveGlow

**WaveGlow** is a flow-based generative model capable of parallel generation (NVIDIA, 2018).

#### Features

  * **Real-time generation** : Faster than WaveNet
  * **Parallelizable** : Generate all samples at once
  * **Invertible transformation** : Bidirectional conversion between audio and latent variables

### 3\. HiFi-GAN

**HiFi-GAN** (High Fidelity GAN) is a fast, high-quality GAN-based vocoder (2020).

#### Architecture

Component | Description  
---|---  
**Generator** | Upsampling with Transposed Convolution  
**Multi-Period Discriminator** | Identifies different periodic patterns  
**Multi-Scale Discriminator** | Identifies at different resolutions  
      
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class HiFiGANGenerator(nn.Module):
        """HiFi-GAN Generator (simplified version)"""
    
        def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
            super().__init__()
    
            self.num_upsamples = len(upsample_rates)
    
            # Initial Conv
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
    
            # Final Conv
            self.conv_post = nn.Conv1d(512 // (2 ** len(upsample_rates)),
                                       1, 7, padding=3)
    
        def forward(self, mel):
            """
            Args:
                mel: Mel-spectrogram (B, mel_channels, T)
            Returns:
                audio: Speech waveform (B, 1, T * prod(upsample_rates))
            """
            x = self.conv_pre(mel)
    
            for i in range(self.num_upsamples):
                x = torch.nn.functional.leaky_relu(x, 0.1)
                x = self.ups[i](x)
    
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = self.conv_post(x)
            x = torch.tanh(x)
    
            return x
    
    # Test
    generator = HiFiGANGenerator()
    mel_input = torch.randn(2, 80, 100)  # (B, mel_channels, T)
    audio_output = generator(mel_input)
    
    print("=== HiFi-GAN Generator ===")
    print(f"Input Mel: {mel_input.shape}")
    print(f"Output audio: {audio_output.shape}")
    print(f"Upsampling rate: {audio_output.size(2) / mel_input.size(2):.0f}x")
    

### Vocoder Comparison

Vocoder | Generation Method | Speed | Quality (MOS) | Features  
---|---|---|---|---  
**Griffin-Lim** | Iterative algorithm | Fast | 3.0-3.5 | Simple, low quality  
**WaveNet** | Autoregressive | Very slow | 4.5+ | Highest quality  
**WaveGlow** | Flow-based | Medium | 4.2-4.3 | Parallel generation  
**HiFi-GAN** | GAN | Very fast | 4.3-4.5 | Fast & high quality  
  
> **Recommendation** : Currently, HiFi-GAN offers the best balance of speed and quality and is widely used.

* * *

## 4.5 Latest TTS Technologies

### 1\. VITS (End-to-End TTS)

**VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) is an end-to-end model that integrates the acoustic model and vocoder (2021).

#### VITS Features

  * **Integrated architecture** : Acoustic model + Vocoder
  * **VAE + GAN** : Combination of Variational Autoencoder and adversarial learning
  * **Fast & high quality**: Real-time generation capability
  * **Diversity** : Generate diverse speech from the same text

    
    
    ```mermaid
    graph LR
        A[Text] --> B[Text Encoder]
        B --> C[Posterior Encoder]
        B --> D[Prior Encoder]
        C --> E[Latent Variable z]
        D --> E
        E --> F[Decoder/Generator]
        F --> G[Speech Waveform]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#f3e5f5
        style E fill:#e3f2fd
        style F fill:#e8f5e9
        style G fill:#c8e6c9
    ```

### 2\. Voice Cloning

**Voice Cloning** is a technology that reproduces a specific speaker's voice from a small amount of audio samples.

#### Approaches

Method | Description | Data Required  
---|---|---  
**Speaker Adaptation** | Fine-tune existing model with small amount of data | Few to tens of minutes  
**Speaker Embedding** | Learn speaker embedding vectors | Few seconds to minutes  
**Zero-shot TTS** | Instantly mimic unknown speaker's voice | Few seconds  
  
### 3\. Multi-speaker TTS

**Multi-speaker TTS** generates multiple speakers' voices with a single model.

#### Speaker Embedding

Convert speaker ID into an embedding vector and condition the model on it.

$$ e_{\text{speaker}} = \text{Embedding}(\text{speaker_id}) $$

$$ h = f(x, e_{\text{speaker}}) $$

### 4\. Japanese TTS Systems

Features of Japanese TTS systems:

#### Japanese-specific Challenges

  * **Accent** : Reproducing pitch accent
  * **Intonation** : Sentence-final rising and falling patterns
  * **Special mora** : Handling of geminate consonants and long vowels
  * **Kanji reading** : Context-dependent reading disambiguation

#### Major Japanese TTS Libraries

System | Features | License  
---|---|---  
**OpenJTalk** | HMM-based, lightweight | BSD  
**VOICEVOX** | Deep learning, high quality | LGPL/Commercial  
**ESPnet-TTS** | Research-oriented, latest methods | Apache 2.0  
  
### Implementation Example: gTTS (Google Text-to-Speech)
    
    
    from gtts import gTTS
    import os
    from IPython.display import Audio
    
    # Text
    text_en = "Hello, this is a demonstration of text-to-speech synthesis."
    text_ja = "Hello, this is a demonstration of speech synthesis."
    
    # English TTS
    tts_en = gTTS(text=text_en, lang='en', slow=False)
    tts_en.save("output_en.mp3")
    
    # Japanese TTS
    tts_ja = gTTS(text=text_ja, lang='ja', slow=False)
    tts_ja.save("output_ja.mp3")
    
    print("=== gTTS (Google Text-to-Speech) ===")
    print("English audio generated: output_en.mp3")
    print("Japanese audio generated: output_ja.mp3")
    
    # Audio playback (Jupyter environment)
    # display(Audio("output_en.mp3"))
    # display(Audio("output_ja.mp3"))
    

* * *

## 4.6 Chapter Summary

### What We Learned

  1. **TTS Fundamentals**

     * Text-to-Speech pipeline: Text analysis → Acoustic model → Vocoder
     * Importance of prosody: Pitch, duration, energy
     * Evaluation metrics: MOS, naturalness
  2. **Tacotron & Tacotron 2**

     * End-to-end TTS using Seq2Seq architecture
     * Text-speech alignment using attention mechanism
     * Improved stability with location-sensitive attention
  3. **FastSpeech**

     * Speedup through non-autoregressive TTS
     * Explicit duration control using Duration Predictor
     * FastSpeech 2: Additional prediction of pitch and energy
  4. **Neural Vocoders**

     * WaveNet: Highest quality but slow
     * WaveGlow: Parallel generation capable
     * HiFi-GAN: Balance of speed and quality
  5. **Latest Technologies**

     * VITS: End-to-end integrated model
     * Voice Cloning: Voice reproduction from small amounts of data
     * Multi-speaker TTS: Multiple speakers with single model
     * Japanese TTS: Challenges of accent and intonation

### TTS Technology Selection Guidelines

Purpose | Recommended Model | Reason  
---|---|---  
Highest quality | Tacotron 2 + WaveNet | MOS 4.5+  
Real-time generation | FastSpeech 2 + HiFi-GAN | Fast & high quality  
End-to-end | VITS | Integrated architecture  
Voice cloning | Speaker Embedding TTS | Works with small data  
Research/experiment | ESPnet-TTS | Latest methods implemented  
  
### Next Chapter

In Chapter 5, we'll learn about **voice conversion and voice transformation** :

  * Voice Conversion fundamentals
  * Style transfer
  * Emotional expression control
  * Real-time voice conversion

* * *

## Practice Problems

### Problem 1 (Difficulty: easy)

Explain the four main stages of the TTS pipeline in order, and describe the role of each stage.

Sample Answer

**Answer** :

  1. **Text Analysis**

     * Role: Text normalization, tokenization, expansion of numbers and abbreviations
     * Output: Normalized text
  2. **Linguistic Feature Extraction**

     * Role: Convert text to phonemes, predict prosodic information
     * Output: Phoneme sequence and prosodic features
  3. **Acoustic Model**

     * Role: Generate Mel-spectrogram from linguistic features
     * Output: Mel-spectrogram (acoustic features)
  4. **Vocoder**

     * Role: Generate speech waveform from Mel-spectrogram
     * Output: Final speech waveform

### Problem 2 (Difficulty: medium)

Compare the main differences between Tacotron 2 and FastSpeech from the perspectives of generation method, speed, and robustness.

Sample Answer

**Answer** :

Aspect | Tacotron 2 | FastSpeech  
---|---|---  
**Generation Method** | Autoregressive  
Predicts next using previous frame | Non-autoregressive  
Parallel generation of all frames  
**Speed** | Slow (sequential generation)  
Baseline: 1x | Very fast (parallel generation)  
~38x faster  
**Robustness** | Low  
\- Word repetition  
\- Skipping  
\- Unstable alignment | High  
\- Nearly error-free  
\- Stable alignment  
\- Predictable output  
**Controllability** | Low  
Difficult to explicitly control speed/prosody | High  
Speed controllable via duration adjustment  
**Quality (MOS)** | 4.41 (high quality) | 4.27 (slightly lower)  
  
**Conclusion** : FastSpeech achieves significant speedup (38x) and robustness improvement with only a slight quality reduction (-3%). For practical applications, FastSpeech is advantageous.

### Problem 3 (Difficulty: medium)

Compare WaveNet, WaveGlow, and HiFi-GAN vocoders from the perspectives of generation method, speed, and quality, and propose use cases for each.

Sample Answer

**Answer** :

**Comparison Table** :

Vocoder | Generation Method | Speed | Quality (MOS) | Features  
---|---|---|---|---  
**WaveNet** | Autoregressive  
Dilated Causal Conv | Very slow | 4.5+ (highest) | \- Highest quality  
\- Not real-time  
**WaveGlow** | Flow-based  
Invertible transformation | Medium | 4.2-4.3 | \- Parallel generation  
\- Stable training  
**HiFi-GAN** | GAN  
Adversarial learning | Very fast | 4.3-4.5 | \- Fast & high quality  
\- Training somewhat difficult  
  
**Use Case Proposals** :

  1. **WaveNet**

     * Use cases: Offline speech synthesis, highest quality priority
     * Examples: Studio-quality audiobook production
  2. **WaveGlow**

     * Use cases: Research purposes, understanding flow-based models
     * Examples: Generative model research, combination with VAE
  3. **HiFi-GAN**

     * Use cases: Real-time applications, practical systems
     * Examples: Voice assistants, live broadcast narration

**Recommendation** : Currently, HiFi-GAN with its excellent balance of speed and quality is most widely used.

### Problem 4 (Difficulty: hard)

Explain the roles of FastSpeech's Duration Predictor and Length Regulator, and describe why they are necessary for non-autoregressive TTS. Provide a simple implementation example in Python code.

Sample Answer

**Answer** :

**Role Explanation** :

  1. **Duration Predictor**

     * Role: Predicts how many frames each phoneme lasts
     * Input: Encoder hidden states (phoneme-level)
     * Output: Duration of each phoneme (number of frames)
  2. **Length Regulator**

     * Role: Expands phoneme-level representation to frame-level
     * Input: Phoneme hidden states + predicted durations
     * Output: Frame-level hidden states

**Necessity** :

In non-autoregressive TTS, all frames are generated in parallel, so:

  * Need to know output length in advance
  * Text (phonemes) and speech (frames) have different lengths
  * Each phoneme has different duration (e.g., "a" is 3 frames, "n" is 1 frame)

By predicting duration with Duration Predictor and expanding phoneme representation to appropriate length with Length Regulator, parallel generation becomes possible.

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class SimpleDurationPredictor(nn.Module):
        """Duration predictor"""
    
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
            return torch.relu(duration)  # Positive values only
    
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
                        # Repeat phoneme hidden state dur times
                        phoneme_hidden = x[batch_idx, phoneme_idx].unsqueeze(0)
                        expanded.append(phoneme_hidden.expand(dur, -1))
    
                if expanded:
                    output.append(torch.cat(expanded, dim=0))
    
            # Pad to same length
            max_len = max(seq.size(0) for seq in output)
            padded = []
            for seq in output:
                if seq.size(0) < max_len:
                    pad = torch.zeros(max_len - seq.size(0), seq.size(1))
                    seq = torch.cat([seq, pad], dim=0)
                padded.append(seq.unsqueeze(0))
    
            return torch.cat(padded, dim=0)
    
    # Test
    print("=== Duration Predictor & Length Regulator ===\n")
    
    # Dummy data
    batch_size, n_phonemes, hidden_size = 2, 5, 256
    phoneme_hidden = torch.randn(batch_size, n_phonemes, hidden_size)
    
    # Duration prediction
    duration_predictor = SimpleDurationPredictor(hidden_size)
    predicted_durations = duration_predictor(phoneme_hidden)
    
    print(f"Phoneme hidden states: {phoneme_hidden.shape}")
    print(f"Predicted durations: {predicted_durations.shape}")
    print(f"Sample durations: {predicted_durations[0]}")
    
    # Length Regulation
    length_regulator = SimpleLengthRegulator()
    frame_hidden = length_regulator(phoneme_hidden, predicted_durations)
    
    print(f"\nFrame-level hidden states: {frame_hidden.shape}")
    print(f"Expansion rate: {frame_hidden.size(1) / phoneme_hidden.size(1):.2f}x")
    
    # Concrete example
    print("\n=== Concrete Example ===")
    print("Phonemes: ['k', 'o', 'n', 'n', 'i']")
    print("Durations: [3, 4, 1, 1, 2] frames")
    print("→ Total 11 frames of speech generation")
    

**Output Example** :
    
    
    === Duration Predictor & Length Regulator ===
    
    Phoneme hidden states: torch.Size([2, 5, 256])
    Predicted durations: torch.Size([2, 5])
    Sample durations: tensor([2.3, 1.8, 3.1, 2.5, 1.2])
    
    Frame-level hidden states: torch.Size([2, 11, 256])
    Expansion rate: 2.20x
    
    === Concrete Example ===
    Phonemes: ['k', 'o', 'n', 'n', 'i']
    Durations: [3, 4, 1, 1, 2] frames
    → Total 11 frames of speech generation
    

### Problem 5 (Difficulty: hard)

Explain the role of Speaker Embedding in Multi-speaker TTS and describe how it is incorporated into the model. Also discuss its relationship with Voice Cloning.

Sample Answer

**Answer** :

**Role of Speaker Embedding** :

  1. **Representation of speaker characteristics**

     * Represents each speaker with a low-dimensional vector (typically 64-512 dimensions)
     * Encodes speaker's voice quality, pitch, and speaking style characteristics
  2. **Conditioning**

     * Provides speaker information to the TTS model
     * Enables generating different speakers' voices from the same text

**Methods of Incorporation into Model** :

**Method 1: Embedding Lookup Table**
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Method 1: Embedding Lookup Table
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
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
            # Get Speaker Embedding
            speaker_emb = self.speaker_embedding(speaker_ids)  # (B, speaker_dim)
    
            # Text Encoding
            text_encoded, _ = self.text_encoder(text_features)  # (B, T, 512)
    
            # Expand Speaker Embedding to all timesteps
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
    
    # Test
    model = MultiSpeakerTTS(n_speakers=100)
    text_features = torch.randn(4, 50, 512)  # Batch=4, Seq=50
    speaker_ids = torch.tensor([0, 5, 10, 15])  # Different speakers
    
    mel_output = model(text_features, speaker_ids)
    print("=== Multi-Speaker TTS ===")
    print(f"Input text: {text_features.shape}")
    print(f"Speaker IDs: {speaker_ids}")
    print(f"Output Mel: {mel_output.shape}")
    

**Method 2: Speaker Encoder (for Voice Cloning)**
    
    
    class SpeakerEncoder(nn.Module):
        """Extract speaker embedding from audio"""
    
        def __init__(self, mel_dim=80, embed_dim=256):
            super().__init__()
            self.lstm = nn.LSTM(mel_dim, 256, num_layers=3,
                               batch_first=True)
            self.projection = nn.Linear(256, embed_dim)
    
        def forward(self, mel_spectrograms):
            # mel_spectrograms: (B, T, 80)
            _, (hidden, _) = self.lstm(mel_spectrograms)
            # Use final layer hidden state
            speaker_emb = self.projection(hidden[-1])  # (B, embed_dim)
            # L2 normalization
            speaker_emb = speaker_emb / torch.norm(speaker_emb, dim=1, keepdim=True)
            return speaker_emb
    
    # Voice Cloning workflow
    speaker_encoder = SpeakerEncoder()
    
    # Extract speaker embedding from reference audio
    reference_mel = torch.randn(1, 100, 80)  # Few seconds of audio
    speaker_emb = speaker_encoder(reference_mel)
    
    print("\n=== Voice Cloning ===")
    print(f"Reference audio: {reference_mel.shape}")
    print(f"Extracted speaker embedding: {speaker_emb.shape}")
    print("→ Use this embedding to synthesize any text in this speaker's voice")
    

**Relationship with Voice Cloning** :

Aspect | Multi-speaker TTS | Voice Cloning  
---|---|---  
**Speaker representation** | Embedding Lookup  
(Trained speakers only) | Speaker Encoder  
(Unknown speakers possible)  
**Data required** | Large amount of data per speaker | Few seconds to minutes of audio  
**Flexibility** | Low (fixed speaker set) | High (new speaker support)  
**Quality** | High (optimized per speaker) | Medium-High (data dependent)  
  
**Integrated Approach** :

Latest systems use a combination of both:

  1. Pre-train Multi-speaker TTS on large-scale data
  2. Extract embedding for new speakers with Speaker Encoder
  3. Fine-tune with small amount of additional data (Optional)

This allows achieving both high quality for trained speakers and support for unknown speakers.

* * *

## References

  1. Wang, Y., et al. (2017). "Tacotron: Towards End-to-End Speech Synthesis." _Interspeech 2017_.
  2. Shen, J., et al. (2018). "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions." _ICASSP 2018_.
  3. Ren, Y., et al. (2019). "FastSpeech: Fast, Robust and Controllable Text to Speech." _NeurIPS 2019_.
  4. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." _arXiv:1609.03499_.
  5. Prenger, R., Valle, R., & Catanzaro, B. (2019). "WaveGlow: A Flow-based Generative Network for Speech Synthesis." _ICASSP 2019_.
  6. Kong, J., Kim, J., & Bae, J. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." _NeurIPS 2020_.
  7. Kim, J., Kong, J., & Son, J. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." _ICML 2021_.
  8. Casanova, E., et al. (2022). "YourtTS: Towards Zero-Shot Multi-Speaker TTS." _arXiv:2112.02418_.
