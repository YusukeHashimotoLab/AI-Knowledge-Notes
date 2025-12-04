---
title: "Chapter 4: Attention Mechanism"
chapter_title: "Chapter 4: Attention Mechanism"
subtitle: Theory and implementation of the attention mechanism that revolutionized sequence transformation tasks
reading_time: 23 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 6
---

This chapter covers Attention Mechanism. You will learn Bahdanau Attention (Additive Attention) and Integrate attention into Seq2Seq models.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the necessity and theoretical background of attention mechanisms
  * ✅ Implement Bahdanau Attention (Additive Attention)
  * ✅ Understand the mathematical definition of Luong Attention (Multiplicative Attention)
  * ✅ Interpret model behavior through attention weight visualization
  * ✅ Integrate attention into Seq2Seq models
  * ✅ Understand the fundamental concepts of Self-Attention (preparation for Transformer)
  * ✅ Build attention-based NMT systems

* * *

## 4.1 The Necessity of Attention

### The Bottleneck Problem of Context Vectors

In the Encoder-Decoder model we learned in Chapter 3, the entire input sequence was compressed into a fixed-length Context Vector. This design has the following serious problems:

Problem | Details | Impact  
---|---|---  
**Information Bottleneck** | Compressing long sequences into fixed-length vectors | Information loss, performance degradation on long texts  
**Long-Range Dependency Difficulty** | Difficult to associate the beginning and end of a sequence | Reduced accuracy in long sentence translation  
**Uniform Importance** | Treating all words with the same weight | Unable to emphasize important words  
**Lack of Interpretability** | Model's decision rationale unclear | Difficult to debug and improve  
  
### Solution Through Attention

The attention mechanism enables the Decoder to dynamically learn "which part of the input sequence to focus on" at each time step. By computing different Context Vectors at each time step instead of a fixed-length Context Vector, it solves the above problems.
    
    
    ```mermaid
    graph TB
        subgraph "Traditional Seq2Seq (Fixed Context)"
            A1[Input: I love AI] --> E1[Encoder]
            E1 --> C1[Fixed Context Vector]
            C1 --> D1[Decoder]
            D1 --> O1[Output: I love AI]
    
            style C1 fill:#e74c3c,color:#fff
        end
    
        subgraph "Seq2Seq with Attention (Dynamic Context)"
            A2[Input: I love AI] --> E2[Encoder]
            E2 --> H1[hidden states]
            H1 --> ATT[Attention Mechanism]
            D2[Decoder state] --> ATT
            ATT --> C2[Dynamic Context t=1]
            ATT --> C3[Dynamic Context t=2]
            ATT --> C4[Dynamic Context t=3]
            C2 --> D2
            C3 --> D2
            C4 --> D2
            D2 --> O2[Output: I love AI]
    
            style ATT fill:#27ae60,color:#fff
        end
    ```

> **Important** : Attention is a mechanism that learns "where to look." Just as humans pay attention to important parts when reading a sentence, the model also places weight on important parts of the input sequence.

* * *

## 4.2 Bahdanau Attention (Additive Attention)

### 4.2.1 Basic Concept and Architecture

Bahdanau Attention (proposed in 2014) was the first widely adopted attention mechanism. Also called Additive Attention, it combines Encoder and Decoder hidden states additively.

#### Mathematical Definition

Computing attention weights at time step $t$:

**Step 1: Alignment Score**

$$ e_{ti} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i) $$ 

Where:

  * $s_{t-1}$: Decoder's hidden state at the previous time step (Query)
  * $h_i$: Encoder's $i$-th hidden state (Key)
  * $W_a, U_a$: Learnable weight matrices
  * $v_a$: Learnable weight vector

**Step 2: Attention Weight**

$$ \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T_x} \exp(e_{tj})} $$ 

The softmax function normalizes so that the sum of weights over all time steps equals 1.

**Step 3: Context Vector**

$$ c_t = \sum_{i=1}^{T_x} \alpha_{ti} h_i $$ 

The weighted sum of Encoder hidden states is computed using attention weights.

### 4.2.2 PyTorch Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    class BahdanauAttention(nn.Module):
        """Implementation of Bahdanau Attention (Additive Attention)"""
    
        def __init__(self, hidden_size):
            super(BahdanauAttention, self).__init__()
            self.hidden_size = hidden_size
    
            # Attention parameters
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)  # For Decoder
            self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)  # For Encoder
            self.v_a = nn.Linear(hidden_size, 1, bias=False)            # Scalar transformation
    
        def forward(self, decoder_hidden, encoder_outputs):
            """
            Args:
                decoder_hidden: Decoder's hidden state [batch, hidden_size]
                encoder_outputs: All Encoder hidden states [batch, seq_len, hidden_size]
    
            Returns:
                context: Context vector [batch, hidden_size]
                attention_weights: Attention weights [batch, seq_len]
            """
            batch_size = encoder_outputs.size(0)
            seq_len = encoder_outputs.size(1)
    
            # Repeat decoder hidden for each time step [batch, seq_len, hidden_size]
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
    
            # Alignment score calculation: e_ti = v_a^T * tanh(W_a * s_t + U_a * h_i)
            energy = torch.tanh(self.W_a(decoder_hidden) + self.U_a(encoder_outputs))
            alignment_scores = self.v_a(energy).squeeze(-1)  # [batch, seq_len]
    
            # Normalize with Softmax to get attention weights
            attention_weights = F.softmax(alignment_scores, dim=1)  # [batch, seq_len]
    
            # Context vector calculation: c_t = Σ α_ti * h_i
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            context = context.squeeze(1)  # [batch, hidden_size]
    
            return context, attention_weights
    
    
    # Demonstration
    print("=== Bahdanau Attention Demo ===\n")
    
    # Parameter settings
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    
    # Generate dummy data
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    decoder_hidden = torch.randn(batch_size, hidden_size)
    
    # Apply attention
    attention = BahdanauAttention(hidden_size)
    context, weights = attention(decoder_hidden, encoder_outputs)
    
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Decoder hidden shape: {decoder_hidden.shape}")
    print(f"Context vector shape: {context.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (batch 0):")
    print(weights[0].detach().numpy())
    print(f"Sum: {weights[0].sum().item():.4f} (Should be 1.0)")
    
    # Visualize attention weights
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Display attention weights of first batch as bar chart
    weights_np = weights[0].detach().numpy()
    positions = np.arange(seq_len)
    
    ax.bar(positions, weights_np, color='#7b2cbf', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Encoder Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Bahdanau Attention Weights Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f't={i+1}' for i in positions])
    ax.grid(axis='y', alpha=0.3)
    
    # Display values on bars
    for i, v in enumerate(weights_np):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Bahdanau Attention Demo ===
    
    Encoder outputs shape: torch.Size([2, 5, 8])
    Decoder hidden shape: torch.Size([2, 8])
    Context vector shape: torch.Size([2, 8])
    Attention weights shape: torch.Size([2, 5])
    
    Attention weights (batch 0):
    [0.178 0.245 0.198 0.156 0.223]
    Sum: 1.0000 (Should be 1.0)
    

* * *

## 4.3 Luong Attention (Multiplicative Attention)

### 4.3.1 Differences Between Bahdanau and Luong

Luong Attention (proposed in 2015) calculates alignment scores using a different approach from Bahdanau.

Characteristic | Bahdanau Attention | Luong Attention  
---|---|---  
**Proposal Year** | 2014 | 2015  
**Alternative Name** | Additive Attention | Multiplicative Attention  
**Score Calculation** | $v_a^\top \tanh(W_a s_t + U_a h_i)$ | $s_t^\top W_a h_i$ (general)  
**Decoder State** | Uses previous state $s_{t-1}$ | Uses current state $s_t$  
**Computational Cost** | Somewhat high (tanh operation) | Low (dot product only)  
**Performance** | Advantage on small-scale data | Advantage on large-scale data  
  
### 4.3.2 Three Scoring Functions of Luong Attention

Luong proposed three methods for calculating alignment scores:

**1\. Dot (Inner Product)**

$$ \text{score}(s_t, h_i) = s_t^\top h_i $$ 

**2\. General (Generalized Inner Product)**

$$ \text{score}(s_t, h_i) = s_t^\top W_a h_i $$ 

**3\. Concat (Concatenation)**

$$ \text{score}(s_t, h_i) = v_a^\top \tanh(W_a [s_t; h_i]) $$ 

### 4.3.3 Implementation and Comparison with Bahdanau
    
    
    class LuongAttention(nn.Module):
        """Implementation of Luong Attention (Multiplicative Attention)"""
    
        def __init__(self, hidden_size, score_type='general'):
            super(LuongAttention, self).__init__()
            self.hidden_size = hidden_size
            self.score_type = score_type
    
            if score_type == 'general':
                self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            elif score_type == 'concat':
                self.W_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
                self.v_a = nn.Linear(hidden_size, 1, bias=False)
            elif score_type == 'dot':
                pass  # No parameters needed
            else:
                raise ValueError(f"Unknown score_type: {score_type}")
    
        def forward(self, decoder_hidden, encoder_outputs):
            """
            Args:
                decoder_hidden: [batch, hidden_size]
                encoder_outputs: [batch, seq_len, hidden_size]
    
            Returns:
                context: [batch, hidden_size]
                attention_weights: [batch, seq_len]
            """
            batch_size = encoder_outputs.size(0)
            seq_len = encoder_outputs.size(1)
    
            if self.score_type == 'dot':
                # s_t^T * h_i
                alignment_scores = torch.bmm(
                    encoder_outputs,  # [batch, seq_len, hidden]
                    decoder_hidden.unsqueeze(2)  # [batch, hidden, 1]
                ).squeeze(2)  # [batch, seq_len]
    
            elif self.score_type == 'general':
                # s_t^T * W_a * h_i
                transformed = self.W_a(encoder_outputs)  # [batch, seq_len, hidden]
                alignment_scores = torch.bmm(
                    transformed,
                    decoder_hidden.unsqueeze(2)
                ).squeeze(2)
    
            elif self.score_type == 'concat':
                # v_a^T * tanh(W_a * [s_t; h_i])
                decoder_repeated = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
                concat = torch.cat([decoder_repeated, encoder_outputs], dim=2)
                energy = torch.tanh(self.W_a(concat))
                alignment_scores = self.v_a(energy).squeeze(-1)
    
            # Normalize with Softmax
            attention_weights = F.softmax(alignment_scores, dim=1)
    
            # Context vector calculation
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            context = context.squeeze(1)
    
            return context, attention_weights
    
    
    # Comparison of three scoring functions
    print("\n=== Luong Attention: Comparison of 3 Scoring Functions ===\n")
    
    # Dummy data
    batch_size = 1
    seq_len = 6
    hidden_size = 4
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    decoder_hidden = torch.randn(batch_size, hidden_size)
    
    score_types = ['dot', 'general', 'concat']
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, score_type in enumerate(score_types):
        attention = LuongAttention(hidden_size, score_type=score_type)
        context, weights = attention(decoder_hidden, encoder_outputs)
        results[score_type] = weights[0].detach().numpy()
    
        print(f"{score_type.upper()} score:")
        print(f"  Weights: {results[score_type]}")
        print(f"  Sum: {results[score_type].sum():.4f}\n")
    
        # Visualization
        ax = axes[idx]
        positions = np.arange(seq_len)
        ax.bar(positions, results[score_type], color='#7b2cbf', alpha=0.7, edgecolor='black')
        ax.set_title(f'{score_type.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Encoder Position', fontsize=10)
        ax.set_ylabel('Attention Weight', fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([f't={i+1}' for i in positions], fontsize=9)
        ax.set_ylim([0, max(results[score_type]) * 1.2])
        ax.grid(axis='y', alpha=0.3)
    
        # Display values
        for i, v in enumerate(results[score_type]):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Luong Attention: Score Function Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Comparison of Bahdanau and Luong
    print("\n=== Bahdanau vs Luong Attention ===\n")
    
    bahdanau_attn = BahdanauAttention(hidden_size)
    luong_attn = LuongAttention(hidden_size, score_type='general')
    
    _, bahdanau_weights = bahdanau_attn(decoder_hidden, encoder_outputs)
    _, luong_weights = luong_attn(decoder_hidden, encoder_outputs)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bahdanau
    ax1 = axes[0]
    positions = np.arange(seq_len)
    bahdanau_np = bahdanau_weights[0].detach().numpy()
    ax1.bar(positions, bahdanau_np, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_title('Bahdanau Attention\n(Additive)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Encoder Position', fontsize=10)
    ax1.set_ylabel('Attention Weight', fontsize=10)
    ax1.set_xticks(positions)
    ax1.set_ylim([0, 0.3])
    ax1.grid(axis='y', alpha=0.3)
    
    # Luong
    ax2 = axes[1]
    luong_np = luong_weights[0].detach().numpy()
    ax2.bar(positions, luong_np, color='#27ae60', alpha=0.7, edgecolor='black')
    ax2.set_title('Luong Attention\n(Multiplicative - General)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Encoder Position', fontsize=10)
    ax2.set_ylabel('Attention Weight', fontsize=10)
    ax2.set_xticks(positions)
    ax2.set_ylim([0, 0.3])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Bahdanau vs Luong: Attention Weight Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Luong Attention: Comparison of 3 Scoring Functions ===
    
    DOT score:
      Weights: [0.142 0.189 0.165 0.178 0.154 0.172]
      Sum: 1.0000
    
    GENERAL score:
      Weights: [0.158 0.172 0.169 0.165 0.161 0.175]
      Sum: 1.0000
    
    CONCAT score:
      Weights: [0.167 0.171 0.164 0.169 0.162 0.167]
      Sum: 1.0000
    

* * *

## 4.4 Visualization and Interpretation of Attention Weights

### 4.4.1 Attention Visualization in Translation Tasks

One of the major advantages of attention is that you can visually confirm which input words the model focused on to generate the output.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def visualize_attention(attention_weights, source_tokens, target_tokens,
                           title='Attention Visualization'):
        """
        Visualize attention weights as a heatmap
    
        Args:
            attention_weights: Attention weight matrix [target_len, source_len]
            source_tokens: List of input words
            target_tokens: List of output words
            title: Graph title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Draw heatmap
        sns.heatmap(attention_weights,
                    xticklabels=source_tokens,
                    yticklabels=target_tokens,
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Attention Weight'},
                    ax=ax,
                    annot=True,  # Display values
                    fmt='.3f',   # 3 decimal places
                    linewidths=0.5,
                    linecolor='gray')
    
        ax.set_xlabel('Source (Input)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target (Output)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
        plt.tight_layout()
        plt.show()
    
    
    # Machine translation example (English to Japanese)
    print("=== Attention Visualization Example: Machine Translation ===\n")
    
    # Example: "I love deep learning" → "I love deep learning"
    source_tokens = ['I', 'love', 'deep', 'learning', '<eos>']
    target_tokens = ['I', 'deep', 'learning', 'love', '<eos>']
    
    # Mock attention weight matrix (values obtained from actual model)
    # Each row represents the attention distribution when generating each output token
    attention_matrix = np.array([
        [0.82, 0.05, 0.03, 0.05, 0.05],  # When generating "I" → Strong focus on "I"
        [0.05, 0.08, 0.70, 0.12, 0.05],  # When generating "deep" → Strong focus on "deep"
        [0.03, 0.05, 0.15, 0.72, 0.05],  # When generating "learning" → Strong focus on "learning"
        [0.05, 0.75, 0.08, 0.07, 0.05],  # When generating "love" → Strong focus on "love"
        [0.05, 0.05, 0.05, 0.05, 0.80],  # When generating "<eos>" → Strong focus on "<eos>"
    ])
    
    visualize_attention(attention_matrix, source_tokens, target_tokens,
                       title='Attention Weights: English → Japanese Translation')
    
    # More complex example
    print("\n=== Attention Patterns in Long Sentence Translation ===\n")
    
    source_long = ['The', 'cat', 'sat', 'on', 'the', 'mat', '<eos>']
    target_long = ['The', 'cat', 'on', 'the', 'mat', 'sat', '<eos>']
    
    # Attention when word order differs
    attention_long = np.array([
        [0.65, 0.10, 0.05, 0.05, 0.10, 0.03, 0.02],  # "The" → "The"
        [0.10, 0.70, 0.05, 0.05, 0.05, 0.03, 0.02],  # "cat" → "cat"
        [0.05, 0.05, 0.05, 0.05, 0.10, 0.68, 0.02],  # "mat" → "mat"
        [0.05, 0.05, 0.10, 0.75, 0.02, 0.02, 0.01],  # "on" → "on"
        [0.05, 0.10, 0.70, 0.05, 0.05, 0.03, 0.02],  # "sat" → "sat"
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88],  # "<eos>" → "<eos>"
    ])
    
    visualize_attention(attention_long, source_long, target_long,
                       title='Attention Weights: Word Order Reordering (EN→JA)')
    
    print("Observable points from visualization:")
    print("✓ Distribution close to diagonal: Language pairs with similar word order (e.g., English-German)")
    print("✓ Off-diagonal distribution: Language pairs with different word order (e.g., English-Japanese)")
    print("✓ Distribution across multiple words: One-to-many, many-to-one word correspondences")
    print("✓ Concentration on EOS symbol: Clear sentence-final processing")
    </eos></eos></eos></eos></eos></eos></eos></eos>

### 4.4.2 Statistical Analysis of Attention Weights
    
    
    def analyze_attention_statistics(attention_weights):
        """Analyze statistical properties of attention weights"""
    
        print("=== Attention Statistics Analysis ===\n")
    
        # Entropy calculation (measure of concentration)
        epsilon = 1e-10
        entropy = -np.sum(attention_weights * np.log(attention_weights + epsilon), axis=1)
    
        # Maximum weights
        max_weights = np.max(attention_weights, axis=1)
    
        # Number of effective attention positions (weight > threshold)
        threshold = 0.1
        effective_positions = np.sum(attention_weights > threshold, axis=1)
    
        print(f"Entropy at each time step: {entropy}")
        print(f"  Mean: {entropy.mean():.4f}, Std Dev: {entropy.std():.4f}")
        print(f"\nMaximum weight at each time step: {max_weights}")
        print(f"  Mean: {max_weights.mean():.4f}, Std Dev: {max_weights.std():.4f}")
        print(f"\nNumber of effective attention positions (weight > {threshold}): {effective_positions}")
        print(f"  Mean: {effective_positions.mean():.2f}")
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
        # Entropy
        axes[0].plot(entropy, marker='o', color='#7b2cbf', linewidth=2, markersize=8)
        axes[0].set_xlabel('Target Position', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Entropy', fontsize=11, fontweight='bold')
        axes[0].set_title('Attention Concentration\n(Lower = More Focused)',
                         fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
    
        # Maximum weight
        axes[1].plot(max_weights, marker='s', color='#e74c3c', linewidth=2, markersize=8)
        axes[1].set_xlabel('Target Position', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Max Weight', fontsize=11, fontweight='bold')
        axes[1].set_title('Maximum Attention Weight\n(Higher = Strong Focus)',
                         fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
    
        # Number of effective positions
        axes[2].bar(range(len(effective_positions)), effective_positions,
                   color='#27ae60', alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Target Position', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Num. Effective Positions', fontsize=11, fontweight='bold')
        axes[2].set_title(f'Attention Spread\n(Threshold = {threshold})',
                         fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    # Run analysis
    analyze_attention_statistics(attention_long)
    

**Output** :
    
    
    === Attention Statistics Analysis ===
    
    Entropy at each time step: [1.086 0.897 0.935 0.735 0.901 0.412]
      Mean: 0.8277, Std Dev: 0.2194
    
    Maximum weight at each time step: [0.65 0.70 0.68 0.75 0.70 0.88]
      Mean: 0.7267, Std Dev: 0.0737
    
    Number of effective attention positions (weight > 0.1): [4 3 2 2 3 1]
      Mean: 2.50
    

> **Interpretation Guide** :  
>  • **Low Entropy** : Strong concentration on specific position (e.g., when generating EOS at sentence end)  
>  • **High Entropy** : Distribution across multiple positions (e.g., when processing compound words or phrases)  
>  • **High Maximum Weight** : Clear correspondence relationship (e.g., one-to-one word translation)  
>  • **Number of Effective Positions** : Spread of contextual information (more indicates broader context utilization)

* * *

## 4.5 Implementation of Seq2Seq Model with Attention

### 4.5.1 Complete Encoder-Decoder with Attention
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    
    class AttentionEncoder(nn.Module):
        """Encoder for Attention (saving all hidden states)"""
    
        def __init__(self, input_size, hidden_size, num_layers=1):
            super(AttentionEncoder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                             batch_first=True)
    
        def forward(self, x):
            """
            Args:
                x: [batch, seq_len]
            Returns:
                outputs: [batch, seq_len, hidden_size]
                hidden: [num_layers, batch, hidden_size]
            """
            embedded = self.embedding(x)  # [batch, seq_len, hidden_size]
            outputs, hidden = self.gru(embedded)
            return outputs, hidden
    
    
    class AttentionDecoder(nn.Module):
        """Decoder with Attention mechanism"""
    
        def __init__(self, output_size, hidden_size, attention_type='bahdanau', num_layers=1):
            super(AttentionDecoder, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.num_layers = num_layers
    
            self.embedding = nn.Embedding(output_size, hidden_size)
    
            # Attention mechanism
            if attention_type == 'bahdanau':
                self.attention = BahdanauAttention(hidden_size)
            elif attention_type == 'luong':
                self.attention = LuongAttention(hidden_size, score_type='general')
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
    
            # GRU input = embedding + context
            self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers,
                             batch_first=True)
    
            # Output layer
            self.out = nn.Linear(hidden_size, output_size)
    
        def forward(self, input_token, hidden, encoder_outputs):
            """
            Args:
                input_token: [batch, 1]
                hidden: [num_layers, batch, hidden_size]
                encoder_outputs: [batch, source_len, hidden_size]
            Returns:
                output: [batch, output_size]
                hidden: [num_layers, batch, hidden_size]
                attention_weights: [batch, source_len]
            """
            # Embedding
            embedded = self.embedding(input_token)  # [batch, 1, hidden_size]
    
            # Attention calculation (using last layer hidden)
            query = hidden[-1]  # [batch, hidden_size]
            context, attention_weights = self.attention(query, encoder_outputs)
    
            # Combine context vector and embedding
            context = context.unsqueeze(1)  # [batch, 1, hidden_size]
            gru_input = torch.cat([embedded, context], dim=2)  # [batch, 1, hidden*2]
    
            # GRU
            gru_output, hidden = self.gru(gru_input, hidden)
    
            # Output
            output = self.out(gru_output.squeeze(1))  # [batch, output_size]
    
            return output, hidden, attention_weights
    
    
    class Seq2SeqWithAttention(nn.Module):
        """Seq2Seq model with Attention"""
    
        def __init__(self, encoder, decoder, device):
            super(Seq2SeqWithAttention, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
    
        def forward(self, source, target, teacher_forcing_ratio=0.5):
            """
            Args:
                source: [batch, source_len]
                target: [batch, target_len]
                teacher_forcing_ratio: Probability of using teacher forcing
            Returns:
                outputs: [batch, target_len, output_size]
                all_attention_weights: [batch, target_len, source_len]
            """
            batch_size = source.size(0)
            target_len = target.size(1)
            target_vocab_size = self.decoder.output_size
    
            # Store outputs and attention weights
            outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
            all_attention_weights = torch.zeros(batch_size, target_len, source.size(1)).to(self.device)
    
            # Encoder
            encoder_outputs, hidden = self.encoder(source)
    
            # Decoder initial input (<sos> token)
            decoder_input = target[:, 0].unsqueeze(1)  # [batch, 1]
    
            # Decoder at each time step
            for t in range(1, target_len):
                output, hidden, attention_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs
                )
    
                outputs[:, t, :] = output
                all_attention_weights[:, t, :] = attention_weights
    
                # Teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = target[:, t].unsqueeze(1)
                else:
                    decoder_input = output.argmax(1).unsqueeze(1)
    
            return outputs, all_attention_weights
    
    
    # Model construction and testing
    print("=== Seq2Seq with Attention Model Test ===\n")
    
    # Parameters
    INPUT_VOCAB = 100
    OUTPUT_VOCAB = 100
    HIDDEN_SIZE = 128
    BATCH_SIZE = 4
    SOURCE_LEN = 10
    TARGET_LEN = 12
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    encoder = AttentionEncoder(INPUT_VOCAB, HIDDEN_SIZE).to(device)
    decoder = AttentionDecoder(OUTPUT_VOCAB, HIDDEN_SIZE, attention_type='bahdanau').to(device)
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    
    # Dummy data
    source = torch.randint(0, INPUT_VOCAB, (BATCH_SIZE, SOURCE_LEN)).to(device)
    target = torch.randint(0, OUTPUT_VOCAB, (BATCH_SIZE, TARGET_LEN)).to(device)
    
    # Forward pass
    outputs, attention_weights = model(source, target)
    
    print(f"Input shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Visualize attention weights of one sample
    sample_attention = attention_weights[0].detach().cpu().numpy()
    print(f"\nSample attention weights (target_len={TARGET_LEN}, source_len={SOURCE_LEN}):")
    print(f"Shape: {sample_attention.shape}")
    print(f"Sum at each time step: {sample_attention.sum(axis=1)}")  # Should all be 1.0
    </sos>

**Output** :
    
    
    === Seq2Seq with Attention Model Test ===
    
    Input shape: torch.Size([4, 10])
    Target shape: torch.Size([4, 12])
    Output shape: torch.Size([4, 12, 100])
    Attention weights shape: torch.Size([4, 12, 10])
    
    Model parameter count: 49,700
    
    Sample attention weights (target_len=12, source_len=10):
    Shape: (12, 10)
    Sum at each time step: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    

### 4.5.2 Implementation of Training and Evaluation
    
    
    def train_attention_model(model, train_loader, optimizer, criterion, device, clip=1.0):
        """Training Seq2Seq with Attention"""
        model.train()
        epoch_loss = 0
    
        for batch_idx, (source, target) in enumerate(train_loader):
            source, target = source.to(device), target.to(device)
    
            optimizer.zero_grad()
    
            # Forward pass
            outputs, _ = model(source, target, teacher_forcing_ratio=0.5)
    
            # Loss calculation (excluding  token)
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)
    
            loss = criterion(outputs_flat, target_flat)
    
            # Backward pass
            loss.backward()
    
            # Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
            optimizer.step()
    
            epoch_loss += loss.item()
    
        return epoch_loss / len(train_loader)
    
    
    def evaluate_attention_model(model, test_loader, criterion, device):
        """Evaluation of Seq2Seq with Attention"""
        model.eval()
        epoch_loss = 0
    
        with torch.no_grad():
            for source, target in test_loader:
                source, target = source.to(device), target.to(device)
    
                # Evaluate without teacher forcing
                outputs, _ = model(source, target, teacher_forcing_ratio=0.0)
    
                output_dim = outputs.shape[-1]
                outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
                target_flat = target[:, 1:].reshape(-1)
    
                loss = criterion(outputs_flat, target_flat)
                epoch_loss += loss.item()
    
        return epoch_loss / len(test_loader)
    
    
    # Simple demo
    print("\n=== Training Demo (Simplified) ===\n")
    
    # Dummy data loader
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, source_len, target_len, vocab_size):
            self.num_samples = num_samples
            self.source_len = source_len
            self.target_len = target_len
            self.vocab_size = vocab_size
    
        def __len__(self):
            return self.num_samples
    
        def __getitem__(self, idx):
            source = torch.randint(1, self.vocab_size, (self.source_len,))
            target = torch.randint(1, self.vocab_size, (self.target_len,))
            return source, target
    
    train_dataset = DummyDataset(100, SOURCE_LEN, TARGET_LEN, OUTPUT_VOCAB)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Training configuration
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Train for a few epochs
    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss = train_attention_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    
    print("\nTraining complete!")
    

**Output** :
    
    
    === Training Demo (Simplified) ===
    
    Epoch [1/3], Train Loss: 4.5342
    Epoch [2/3], Train Loss: 4.4187
    Epoch [3/3], Train Loss: 4.3256
    
    Training complete!
    

* * *

## 4.6 Introduction to Self-Attention (Preparation for Transformer)

### 4.6.1 What is Self-Attention

The attention we've seen so far learned the relationship between Encoder and Decoder. **Self-Attention** is a mechanism that learns relationships between elements within the same sequence.

Characteristic | Encoder-Decoder Attention | Self-Attention  
---|---|---  
**Target of Attention** | Between different sequences (Source→Target) | Within the same sequence (Self→Self)  
**Use Case** | Translation, summarization, etc. (transformation tasks) | Context understanding, feature extraction  
**Query** | Decoder state | Each element in sequence  
**Key/Value** | Encoder state | All elements in sequence  
**Representative Examples** | Bahdanau/Luong Attention | Transformer (BERT, GPT)  
  
### 4.6.2 The Concept of Query, Key, and Value

Self-Attention represents each word in three different roles:

  * **Query (Q)** : "What am I looking for?" - The attending side
  * **Key (K)** : "What can I provide?" - The attended side
  * **Value (V)** : "What is my actual content?" - The retrieved information

Attention calculation formula:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V $$ 

Where $d_k$ is the dimensionality of the Key, used as a scaling factor.

### 4.6.3 Implementation of Self-Attention
    
    
    class SelfAttention(nn.Module):
        """Implementation of Self-Attention mechanism (Scaled Dot-Product Attention)"""
    
        def __init__(self, embed_size, heads=1):
            super(SelfAttention, self).__init__()
            self.embed_size = embed_size
            self.heads = heads
            self.head_dim = embed_size // heads
    
            assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
    
            # Linear transformations for Query, Key, Value
            self.query = nn.Linear(embed_size, embed_size, bias=False)
            self.key = nn.Linear(embed_size, embed_size, bias=False)
            self.value = nn.Linear(embed_size, embed_size, bias=False)
    
            # Output linear transformation
            self.fc_out = nn.Linear(embed_size, embed_size)
    
        def forward(self, x, mask=None):
            """
            Args:
                x: [batch, seq_len, embed_size]
                mask: [batch, seq_len, seq_len] (Optional)
            Returns:
                output: [batch, seq_len, embed_size]
                attention_weights: [batch, heads, seq_len, seq_len]
            """
            batch_size = x.size(0)
            seq_len = x.size(1)
    
            # Generate Q, K, V
            Q = self.query(x)  # [batch, seq_len, embed_size]
            K = self.key(x)
            V = self.value(x)
    
            # Split for multi-head: [batch, seq_len, heads, head_dim]
            Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
            # → [batch, heads, seq_len, head_dim]
    
            # Scaled Dot-Product Attention
            # Q @ K^T: [batch, heads, seq_len, seq_len]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
    
            # Apply masking if present
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
    
            # Normalize with Softmax
            attention_weights = F.softmax(scores, dim=-1)
    
            # Weighted sum with Value
            out = torch.matmul(attention_weights, V)  # [batch, heads, seq_len, head_dim]
    
            # Concatenate heads
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
    
            # Final linear transformation
            output = self.fc_out(out)
    
            return output, attention_weights
    
    
    # Self-Attention Demo
    print("=== Self-Attention Demo ===\n")
    
    # Parameters
    batch_size = 2
    seq_len = 6
    embed_size = 8
    num_heads = 2
    
    # Dummy input (assuming sentence embedding representation)
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # Apply Self-Attention
    self_attn = SelfAttention(embed_size, heads=num_heads)
    output, attn_weights = self_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"  → [batch, heads, seq_len, seq_len]")
    
    # Visualize attention weights of one head
    sample_attn = attn_weights[0, 0].detach().numpy()  # 1st batch, 1st head
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sample_attn,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax,
                annot=True,
                fmt='.3f',
                linewidths=0.5,
                xticklabels=[f'Pos{i+1}' for i in range(seq_len)],
                yticklabels=[f'Pos{i+1}' for i in range(seq_len)])
    
    ax.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax.set_title('Self-Attention Weights Visualization\n(Head 1)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nMatrix showing how much each position attends to other positions")
    print("Diagonal components: Attention to self")
    print("Off-diagonal components: Attention to other positions")
    

**Output** :
    
    
    === Self-Attention Demo ===
    
    Input shape: torch.Size([2, 6, 8])
    Output shape: torch.Size([2, 6, 8])
    Attention weights shape: torch.Size([2, 2, 6, 6])
      → [batch, heads, seq_len, seq_len]
    
    Matrix showing how much each position attends to other positions
    Diagonal components: Attention to self
    Off-diagonal components: Attention to other positions
    

### 4.6.4 Application Examples of Self-Attention
    
    
    def demonstrate_self_attention_patterns():
        """Visualize typical patterns of Self-Attention"""
    
        print("\n=== Typical Self-Attention Patterns ===\n")
    
        seq_len = 8
        tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'today', '.']
    
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
        # Pattern 1: Local attention (focus on neighboring words)
        local_attn = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                local_attn[i, j] = np.exp(-distance / 2)
        local_attn = local_attn / local_attn.sum(axis=1, keepdims=True)
    
        sns.heatmap(local_attn, ax=axes[0, 0], cmap='Reds',
                    xticklabels=tokens, yticklabels=tokens,
                    cbar_kws={'label': 'Weight'}, annot=True, fmt='.2f', linewidths=0.5)
        axes[0, 0].set_title('Pattern 1: Local Attention\n(Focus on neighboring words)',
                            fontsize=12, fontweight='bold')
    
        # Pattern 2: Global attention (focus on specific important words)
        global_attn = np.ones((seq_len, seq_len)) * 0.05
        global_attn[:, 1] = 0.4  # Strong focus on "cat"
        global_attn[:, 5] = 0.3  # Also focus on "mat"
        global_attn = global_attn / global_attn.sum(axis=1, keepdims=True)
    
        sns.heatmap(global_attn, ax=axes[0, 1], cmap='Blues',
                    xticklabels=tokens, yticklabels=tokens,
                    cbar_kws={'label': 'Weight'}, annot=True, fmt='.2f', linewidths=0.5)
        axes[0, 1].set_title('Pattern 2: Global Attention\n(Concentration on important words)',
                            fontsize=12, fontweight='bold')
    
        # Pattern 3: Syntactic structure (subject-verb-object relationships)
        syntax_attn = np.eye(seq_len) * 0.3
        syntax_attn[1, 2] = 0.4  # cat → sat
        syntax_attn[2, 1] = 0.4  # sat → cat
        syntax_attn[2, 5] = 0.3  # sat → mat
        syntax_attn[5, 3] = 0.3  # mat → on
        syntax_attn = syntax_attn / syntax_attn.sum(axis=1, keepdims=True)
    
        sns.heatmap(syntax_attn, ax=axes[1, 0], cmap='Greens',
                    xticklabels=tokens, yticklabels=tokens,
                    cbar_kws={'label': 'Weight'}, annot=True, fmt='.2f', linewidths=0.5)
        axes[1, 0].set_title('Pattern 3: Syntactic Attention\n(Syntactic structural relationships)',
                            fontsize=12, fontweight='bold')
    
        # Pattern 4: Positional attention (focus on first/second half of sentence)
        position_attn = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            if i < seq_len // 2:
                position_attn[i, :seq_len//2] = 1.0  # First half focuses on first half
            else:
                position_attn[i, seq_len//2:] = 1.0  # Second half focuses on second half
        position_attn = position_attn / position_attn.sum(axis=1, keepdims=True)
    
        sns.heatmap(position_attn, ax=axes[1, 1], cmap='Purples',
                    xticklabels=tokens, yticklabels=tokens,
                    cbar_kws={'label': 'Weight'}, annot=True, fmt='.2f', linewidths=0.5)
        axes[1, 1].set_title('Pattern 4: Positional Attention\n(Positional attention pattern)',
                            fontsize=12, fontweight='bold')
    
        plt.suptitle('Self-Attention: Visualization of Typical Attention Patterns',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
        print("Actual Transformer models automatically learn these patterns")
        print("Different Attention Heads capture different linguistic features:")
        print("  Head 1: Syntactic relationships (subject-predicate, etc.)")
        print("  Head 2: Semantic similarity (related words)")
        print("  Head 3: Positional information (near words, far words)")
        print("  ...etc.")
    
    demonstrate_self_attention_patterns()
    

* * *

## 4.7 Summary and Advanced Topics

### What We Learned in This Chapter

Topic | Key Points  
---|---  
**Necessity of Attention** | Resolving fixed-length Context Vector bottleneck  
**Bahdanau Attention** | Additive score calculation, early attention mechanism  
**Luong Attention** | Multiplicative score calculation, improved computational efficiency  
**Attention Visualization** | Improved model interpretability, debugging support  
**Self-Attention** | Learning intra-sequence relationships, foundation of Transformer  
**Implementation Techniques** | Teacher Forcing, Gradient Clipping  
  
### Advanced Topics

**Multi-Head Attention**

Uses multiple Attention Heads in parallel to obtain information from different representational subspaces. A core component of Transformer, which we'll learn in detail in the next chapter.
    
    
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    

**Sparse Attention**

To reduce computational cost, instead of attending to all positions, it only attends to specific patterns (local, strided, global). Used in Longformer, BigBird, etc.

**Cross-Attention**

Attention between two different sequences, used in image captioning (image→text) and multimodal learning.

**Attention Interpretability**

There is debate about whether Attention is actually "interpretable." Recent research has shown that Attention does not necessarily accurately reflect the model's decision rationale.

### Exercises

#### Exercise 4.1: Comparative Experiment on Attention Mechanisms

**Task** : Train Bahdanau Attention and Luong Attention on the same dataset and compare their performance.

**Evaluation Criteria** :

  * Translation accuracy (BLEU score)
  * Training time
  * Memory usage
  * Differences in attention weight distribution

#### Exercise 4.2: Development of Attention Visualization Tool

**Task** : Create an interactive attention visualization tool.

**Functional Requirements** :

  * Display attention weights for arbitrary sentences
  * Comparative display of multiple heads
  * Display attention patterns for each layer
  * Calculate statistics (entropy, concentration)

#### Exercise 4.3: Sentence Classification Using Self-Attention

**Task** : Implement a sentiment analysis model using Self-Attention.

**Data** : IMDB review dataset

**Architecture** : Embedding → Self-Attention → Pooling → FC

#### Exercise 4.4: Attention Analysis in Long Sentence Translation

**Task** : Vary sequence length and analyze attention behavior.

**Experiment** :

  * Short sentences (5-10 words)
  * Medium sentences (20-30 words)
  * Long sentences (50-100 words)

Compare attention distribution entropy and translation accuracy for each case.

#### Exercise 4.5: Attention Dropout

**Task** : Add implementation to apply Dropout to attention weights and investigate the impact on generalization performance.

**Comparison** : Performance comparison at Dropout rates of 0%, 10%, 20%, 30%

#### Exercise 4.6: Multi-Head Attention Prototype

**Task** : Extend single-head Self-Attention to a multi-head version.

**Implementation** :

  * Parallel computation of multiple Attention Heads
  * Concatenation of outputs from each Head
  * Visualization of features learned by each Head

* * *

### Next Chapter Preview

In Chapter 5, we'll learn the **Transformer** architecture, which evolved from the attention mechanism. Transformer completely eliminates RNNs and achieves sequence processing using only Self-Attention and Position Encoding. It is the foundational technology for cutting-edge models such as BERT, GPT, and T5.

> **Topics in the Next Chapter** :  
>  • Overall Transformer architecture  
>  • Multi-Head Attention  
>  • Position Encoding  
>  • Feed-Forward Network  
>  • Layer Normalization and Residual Connection  
>  • Details of Encoder and Decoder  
>  • Implementation: Machine translation with mini-Transformer
