---
title: "Chapter 2: Transformer Architecture"
chapter_title: "Chapter 2: Transformer Architecture"
subtitle: Complete Understanding of Encoder-Decoder and PyTorch Implementation
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 12
exercises: 5
---

This chapter covers Transformer Architecture. You will learn Fully implementing Transformer in PyTorch.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understanding the overall Transformer architecture (Encoder-Decoder structure)
  * ✅ Explaining the components of the Encoder (Multi-Head Attention, FFN, Layer Norm, Residual)
  * ✅ Understanding the features of the Decoder (Masked Self-Attention, Cross-Attention)
  * ✅ Fully implementing Transformer in PyTorch
  * ✅ Understanding the mechanism of autoregressive generation
  * ✅ Building a practical machine translation system

* * *

## 2.1 Transformer Overview

### Architecture Overview

The **Transformer** is a revolutionary architecture proposed in "Attention is All You Need" (Vaswani et al., 2017), which achieves sequence-to-sequence transformation using **only Attention mechanisms** without RNNs or CNNs.
    
    
    ```mermaid
    graph TB
        Input["Input Sequence(Source)"] --> Encoder["Encoder(N-layer Stack)"]
        Encoder --> Memory["Encoded Representation(Memory)"]
        Memory --> Decoder["Decoder(N-layer Stack)"]
        Target["Target Sequence(Target)"] --> Decoder
        Decoder --> Output["Output Sequence(Prediction)"]
    
        style Encoder fill:#b3e5fc
        style Decoder fill:#ffab91
        style Memory fill:#fff9c4
    ```

### Main Components

Component | Role | Features  
---|---|---  
**Encoder** | Transforms input sequence into contextual representation | 6-layer stack, parallelizable  
**Decoder** | Generates output sequence from encoded representation | 6-layer stack, autoregressive generation  
**Multi-Head Attention** | Captures dependencies from multiple perspectives | 8 heads in parallel  
**Feed-Forward Network** | Transforms each position independently | 2-layer MLP (ReLU activation)  
**Positional Encoding** | Injects positional information | Sin/Cos function-based  
**Layer Normalization** | Stabilizes training | Applied after each sublayer  
**Residual Connection** | Improves gradient flow | Skip connection  
  
### Differences from RNN

Aspect | RNN/LSTM | Transformer  
---|---|---  
**Processing Method** | Sequential | Parallel  
**Long-term Dependencies** | Weakens with distance | Direct connection regardless of distance  
**Computational Complexity** | $O(n)$ time | $O(1)$ time (parallelizable)  
**Memory** | Compressed in hidden state | Retains information from all positions  
**Training Speed** | Slow | Fast (GPU utilization)  
  
> "Transformers can train more than 10 times faster than RNNs through parallel processing!"

### Basic Structure Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Basic Structure Visualization
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import math
    
    # Basic Transformer parameters
    print("=== Basic Transformer Configuration ===")
    d_model = 512        # Model dimension
    nhead = 8            # Number of Attention heads
    num_layers = 6       # Number of Encoder/Decoder layers
    d_ff = 2048          # Feed-Forward hidden layer size
    dropout = 0.1        # Dropout rate
    max_len = 5000       # Maximum sequence length
    
    print(f"Model dimension: d_model = {d_model}")
    print(f"Number of Attention heads: nhead = {nhead}")
    print(f"Dimension per head: d_k = d_v = {d_model // nhead}")
    print(f"Encoder/Decoder layers: {num_layers}")
    print(f"FFN hidden layer size: {d_ff}")
    print(f"Total parameters (estimate): {(num_layers * 2) * (4 * d_model**2 + 2 * d_model * d_ff):,}")
    
    # Input/output size example
    batch_size = 32
    src_len = 20  # Source sequence length
    tgt_len = 15  # Target sequence length
    
    print(f"\n=== Input/Output Example ===")
    print(f"Input (source): ({batch_size}, {src_len}, {d_model})")
    print(f"Input (target): ({batch_size}, {tgt_len}, {d_model})")
    print(f"Output: ({batch_size}, {tgt_len}, {d_model})")
    

* * *

## 2.2 Encoder Structure

### Role of the Encoder

The Encoder transforms the input sequence into high-dimensional representations that consider the context of each position. It stacks N layers (typically 6) of EncoderLayer.
    
    
    ```mermaid
    graph TB
        Input["Input Embedding + Positional Encoding"] --> E1["Encoder Layer 1"]
        E1 --> E2["Encoder Layer 2"]
        E2 --> E3["..."]
        E3 --> EN["Encoder Layer N"]
        EN --> Output["Encoded Representation"]
    
        style Input fill:#e1f5ff
        style Output fill:#b3e5fc
    ```

### EncoderLayer Structure

Each EncoderLayer consists of two sublayers:

  1. **Multi-Head Self-Attention** : Captures dependencies within the input sequence
  2. **Position-wise Feed-Forward Network** : Transforms each position independently

**Residual Connection** and **Layer Normalization** are applied to each sublayer.
    
    
    ```mermaid
    graph TB
        X["Input x"] --> MHA["Multi-HeadSelf-Attention"]
        MHA --> Add1["Add & Norm"]
        X --> Add1
        Add1 --> FFN["Feed-ForwardNetwork"]
        Add1 --> Add2["Add & Norm"]
        FFN --> Add2
        Add2 --> Y["Output y"]
    
        style MHA fill:#b3e5fc
        style FFN fill:#ffccbc
        style Add1 fill:#c5e1a5
        style Add2 fill:#c5e1a5
    ```

### Multi-Head Attention Formula

Multi-Head Attention computes attention in parallel across multiple different representational subspaces:

$$ \begin{align} \text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\\ \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\\ \text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \end{align} $$ 

Where:

  * $h$: Number of heads (typically 8)
  * $d_k = d_v = d_{\text{model}} / h$: Dimension per head
  * $W_i^Q, W_i^K, W_i^V, W^O$: Learnable projection matrices

### EncoderLayer Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MultiHeadAttention(nn.Module):
        """Multi-Head Attention mechanism"""
        def __init__(self, d_model, nhead, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
    
            self.d_model = d_model
            self.nhead = nhead
            self.d_k = d_model // nhead
    
            # Linear transformations for Q, K, V
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
    
            # Output linear transformation
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
            # Linear transformation
            Q = self.W_q(query)  # (batch, seq_len, d_model)
            K = self.W_k(key)
            V = self.W_v(value)
    
            # Split into heads
            Q = self.split_heads(Q)  # (batch, nhead, seq_len, d_k)
            K = self.split_heads(K)
            V = self.split_heads(V)
    
            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            # scores: (batch, nhead, seq_len, seq_len)
    
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
    
            # Multiply with Value
            attn_output = torch.matmul(attn_weights, V)
            # attn_output: (batch, nhead, seq_len, d_k)
    
            # Combine heads
            attn_output = self.combine_heads(attn_output)
            # attn_output: (batch, seq_len, d_model)
    
            # Output linear transformation
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
            mask: (batch, 1, seq_len) - padding mask
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
    
    
    # Verification
    print("=== EncoderLayer Verification ===")
    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 32
    seq_len = 20
    
    encoder_layer = EncoderLayer(d_model, nhead, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = encoder_layer(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Attention weights: {attn_weights.shape}")
    print("→ Input and output sizes are the same (due to residual connections)")
    
    # Parameter count
    total_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"\nEncoderLayer parameter count: {total_params:,}")
    

### Complete Encoder Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import math
    
    class PositionalEncoding(nn.Module):
        """Positional Encoding (Sin/Cos)"""
        def __init__(self, d_model, max_len=5000, dropout=0.1):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(dropout)
    
            # Create positional encoding matrix
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
        """Complete Transformer Encoder"""
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                     d_ff=2048, dropout=0.1, max_len=5000):
            super(TransformerEncoder, self).__init__()
    
            self.d_model = d_model
    
            # Word embedding
            self.embedding = nn.Embedding(vocab_size, d_model)
    
            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
    
            # Stack EncoderLayers
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ])
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, src, src_mask=None):
            """
            src: (batch, src_len) - token IDs
            src_mask: (batch, 1, src_len) - padding mask
            """
            # Embedding + scaling
            x = self.embedding(src) * math.sqrt(self.d_model)
    
            # Add positional encoding
            x = self.pos_encoding(x)
    
            # Pass through each EncoderLayer
            attn_weights_list = []
            for layer in self.layers:
                x, attn_weights = layer(x, src_mask)
                attn_weights_list.append(attn_weights)
    
            return x, attn_weights_list
    
    
    # Verification
    print("\n=== Complete Encoder Verification ===")
    vocab_size = 10000
    encoder = TransformerEncoder(vocab_size, d_model=512, nhead=8, num_layers=6)
    
    # Dummy data
    batch_size = 16
    src_len = 25
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    
    # Create padding mask (example: last 5 tokens are padding)
    src_mask = torch.ones(batch_size, 1, src_len)
    src_mask[:, :, -5:] = 0
    
    # Execute Encoder
    encoder_output, attn_weights_list = encoder(src, src_mask)
    
    print(f"Input tokens: {src.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    print(f"Number of attention weights: {len(attn_weights_list)} (per layer)")
    print(f"Each attention weight: {attn_weights_list[0].shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    

### Importance of Layer Normalization and Residual Connection
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Importance of Layer Normalization and Residual Connection
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Effect of Layer Normalization
    print("=== Effect of Layer Normalization ===")
    
    x = torch.randn(32, 20, 512)  # (batch, seq_len, d_model)
    
    # Before Layer Normalization
    print(f"Before normalization - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    
    layer_norm = nn.LayerNorm(512)
    x_normalized = layer_norm(x)
    
    # After Layer Normalization
    print(f"After normalization - Mean: {x_normalized.mean():.4f}, Std: {x_normalized.std():.4f}")
    print("→ Normalized to mean 0, std 1 for each sample and position")
    
    # Effect of Residual Connection
    print("\n=== Effect of Residual Connection ===")
    
    class WithoutResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)  # No residual connection
            return x
    
    class WithResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
    
        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)  # With residual connection
            return x
    
    # Compare gradient flow
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
    
    print(f"Without residual connection - gradient norm: {grad_without:.6f}")
    print(f"With residual connection - gradient norm: {grad_with:.6f}")
    print("→ Residual connections prevent gradient vanishing, enabling training of deep layers")
    

* * *

## 2.3 Decoder Structure

### Role of the Decoder

The Decoder **autoregressively** generates the next token from the Encoder output (memory) and the tokens already generated.
    
    
    ```mermaid
    graph TB
        Target["Target Sequence(Shifted)"] --> D1["Decoder Layer 1"]
        Memory["Encoder Output(Memory)"] --> D1
        D1 --> D2["Decoder Layer 2"]
        Memory --> D2
        D2 --> D3["..."]
        Memory --> D3
        D3 --> DN["Decoder Layer N"]
        Memory --> DN
        DN --> Output["Output(Next Token Prediction)"]
    
        style Target fill:#e1f5ff
        style Memory fill:#fff9c4
        style Output fill:#ffab91
    ```

### DecoderLayer Structure

Each DecoderLayer consists of **three sublayers** :

  1. **Masked Multi-Head Self-Attention** : Masks future tokens to prevent looking ahead
  2. **Cross-Attention** : References the Encoder output (memory)
  3. **Position-wise Feed-Forward Network** : Transforms each position independently

    
    
    ```mermaid
    graph TB
        X["Input x"] --> MMHA["Masked Multi-HeadSelf-Attention"]
        MMHA --> Add1["Add & Norm"]
        X --> Add1
    
        Add1 --> CA["Cross-Attention(Reference Encoder Output)"]
        Memory["Encoder Memory"] --> CA
        CA --> Add2["Add & Norm"]
        Add1 --> Add2
    
        Add2 --> FFN["Feed-ForwardNetwork"]
        FFN --> Add3["Add & Norm"]
        Add2 --> Add3
        Add3 --> Y["Output y"]
    
        style MMHA fill:#ffab91
        style CA fill:#ce93d8
        style FFN fill:#ffccbc
        style Memory fill:#fff9c4
    ```

### Importance of Masked Self-Attention

**Causal Masking** ensures that position $i$ can only reference tokens at positions up to and including $i$. This maintains the same autoregressive conditions during training as during inference.

$$ \text{Mask}_{ij} = \begin{cases} 0 & \text{if } i < j \text{ (future tokens)} \\\ 1 & \text{if } i \geq j \text{ (past tokens)} \end{cases} $$ 

### DecoderLayer Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class DecoderLayer(nn.Module):
        """Transformer Decoder Layer"""
        def __init__(self, d_model, nhead, d_ff, dropout=0.1):
            super(DecoderLayer, self).__init__()
    
            # 1. Masked Self-Attention
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
    
            # 2. Cross-Attention (reference Encoder output)
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
            x: (batch, tgt_len, d_model) - target sequence
            memory: (batch, src_len, d_model) - Encoder output
            tgt_mask: (batch, tgt_len, tgt_len) - causal mask
            memory_mask: (batch, 1, src_len) - padding mask
            """
            # 1. Masked Self-Attention + Residual + Norm
            self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            x = x + self.dropout1(self_attn_output)
            x = self.norm1(x)
    
            # 2. Cross-Attention + Residual + Norm
            # Query: Decoder output, Key/Value: Encoder output
            cross_attn_output, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
            x = x + self.dropout2(cross_attn_output)
            x = self.norm2(x)
    
            # 3. Feed-Forward + Residual + Norm
            ffn_output = self.ffn(x)
            x = x + self.dropout3(ffn_output)
            x = self.norm3(x)
    
            return x, self_attn_weights, cross_attn_weights
    
    
    # Verification
    print("=== DecoderLayer Verification ===")
    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 32
    tgt_len = 15
    src_len = 20
    
    decoder_layer = DecoderLayer(d_model, nhead, d_ff)
    
    # Dummy data
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    
    # Create causal mask
    def create_causal_mask(seq_len):
        """Lower triangular matrix (mask future)"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    tgt_mask = create_causal_mask(tgt_len)
    
    # Execute Decoder
    output, self_attn_weights, cross_attn_weights = decoder_layer(tgt, memory, tgt_mask)
    
    print(f"Target input: {tgt.shape}")
    print(f"Encoder memory: {memory.shape}")
    print(f"Decoder output: {output.shape}")
    print(f"Self-Attention weights: {self_attn_weights.shape}")
    print(f"Cross-Attention weights: {cross_attn_weights.shape}")
    print("→ Cross-Attention references Encoder information")
    

### Causal Mask Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    
    # Create and visualize causal mask
    def create_and_visualize_causal_mask(seq_len=10):
        """Create and visualize causal mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
    
        print(f"=== Causal Mask (seq_len={seq_len}) ===")
        print(mask.numpy())
        print("\n1 = Accessible (past/present)")
        print("0 = Inaccessible (future)")
        print("\nExample: Position 3 can only reference positions 0,1,2,3 (cannot see 4 onwards)")
    
        return mask
    
    # Create mask
    causal_mask = create_and_visualize_causal_mask(seq_len=8)
    
    # Example of applying to Attention scores
    print("\n=== Effect of Mask Application ===")
    scores = torch.randn(8, 8)  # Random Attention scores
    
    print("Scores before masking (partial):")
    print(scores[:4, :4].numpy())
    
    # Apply mask (future to -inf)
    masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    
    print("\nScores after masking (partial):")
    print(masked_scores[:4, :4].numpy())
    
    # Apply Softmax
    attn_weights = F.softmax(masked_scores, dim=-1)
    
    print("\nWeights after Softmax (partial):")
    print(attn_weights[:4, :4].numpy())
    print("→ Weights for future positions (-inf) become 0")
    

### Complete Decoder Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import math
    
    class TransformerDecoder(nn.Module):
        """Complete Transformer Decoder"""
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                     d_ff=2048, dropout=0.1, max_len=5000):
            super(TransformerDecoder, self).__init__()
    
            self.d_model = d_model
    
            # Word embedding
            self.embedding = nn.Embedding(vocab_size, d_model)
    
            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
    
            # Stack DecoderLayers
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ])
    
            # Output layer
            self.fc_out = nn.Linear(d_model, vocab_size)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
            """
            tgt: (batch, tgt_len) - target token IDs
            memory: (batch, src_len, d_model) - Encoder output
            tgt_mask: (batch, tgt_len, tgt_len) - causal mask
            memory_mask: (batch, 1, src_len) - padding mask
            """
            # Embedding + scaling
            x = self.embedding(tgt) * math.sqrt(self.d_model)
    
            # Add positional encoding
            x = self.pos_encoding(x)
    
            # Pass through each DecoderLayer
            self_attn_weights_list = []
            cross_attn_weights_list = []
    
            for layer in self.layers:
                x, self_attn_weights, cross_attn_weights = layer(x, memory, tgt_mask, memory_mask)
                self_attn_weights_list.append(self_attn_weights)
                cross_attn_weights_list.append(cross_attn_weights)
    
            # Project to vocabulary
            logits = self.fc_out(x)  # (batch, tgt_len, vocab_size)
    
            return logits, self_attn_weights_list, cross_attn_weights_list
    
    
    # Verification
    print("\n=== Complete Decoder Verification ===")
    vocab_size = 10000
    decoder = TransformerDecoder(vocab_size, d_model=512, nhead=8, num_layers=6)
    
    # Dummy data
    batch_size = 16
    tgt_len = 20
    src_len = 25
    
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    memory = torch.randn(batch_size, src_len, 512)
    
    # Causal mask
    tgt_mask = create_causal_mask(tgt_len)
    
    # Execute Decoder
    logits, self_attn_weights, cross_attn_weights = decoder(tgt, memory, tgt_mask)
    
    print(f"Target input: {tgt.shape}")
    print(f"Encoder memory: {memory.shape}")
    print(f"Decoder output (logits): {logits.shape}")
    print(f"→ Outputs probability distribution over entire vocabulary at each position")
    
    # Parameter count
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    

* * *

## 2.4 Complete Transformer Model

### Integration of Encoder and Decoder
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class Transformer(nn.Module):
        """Complete Transformer Model (Encoder-Decoder)"""
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
    
            # Parameter initialization
            self._reset_parameters()
    
        def _reset_parameters(self):
            """Xavier initialization"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """
            src: (batch, src_len) - source tokens
            tgt: (batch, tgt_len) - target tokens
            src_mask: (batch, 1, src_len) - source padding mask
            tgt_mask: (batch, tgt_len, tgt_len) - target causal mask
            """
            # Process source with Encoder
            memory, _ = self.encoder(src, src_mask)
    
            # Generate target with Decoder
            output, _, _ = self.decoder(tgt, memory, tgt_mask, src_mask)
    
            return output
    
        def encode(self, src, src_mask=None):
            """Execute Encoder only (used during inference)"""
            memory, _ = self.encoder(src, src_mask)
            return memory
    
        def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
            """Execute Decoder only (used during inference)"""
            output, _, _ = self.decoder(tgt, memory, tgt_mask, memory_mask)
            return output
    
    
    # Create model
    print("=== Complete Transformer Model ===")
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
    
    # Verification
    batch_size = 16
    src_len = 25
    tgt_len = 20
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # Create masks
    src_mask = torch.ones(batch_size, 1, src_len)
    tgt_mask = create_causal_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"Source input: {src.shape}")
    print(f"Target input: {tgt.shape}")
    print(f"Model output: {output.shape}")
    print(f"→ Output shape is (batch, tgt_len, tgt_vocab_size)")
    
    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    

### Autoregressive Generation Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    
    def generate_greedy(model, src, src_mask, max_len, start_token, end_token):
        """
        Greedy Decoding for sequence generation
    
        Args:
            model: Transformer model
            src: (batch, src_len) - source sequence
            src_mask: (batch, 1, src_len) - source mask
            max_len: maximum generation length
            start_token: start token ID
            end_token: end token ID
    
        Returns:
            generated: (batch, gen_len) - generated sequence
        """
        model.eval()
        batch_size = src.size(0)
        device = src.device
    
        # Process with Encoder once
        memory = model.encode(src, src_mask)
    
        # Initialize generated sequence (start token)
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
        # Generate autoregressively
        for _ in range(max_len - 1):
            # Create causal mask
            tgt_len = generated.size(1)
            tgt_mask = create_causal_mask(tgt_len).to(device)
    
            # Execute Decoder
            output = model.decode(generated, memory, tgt_mask, src_mask)
    
            # Get prediction at last position
            next_token_logits = output[:, -1, :]  # (batch, vocab_size)
    
            # Greedy selection (highest probability token)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
    
            # Add to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
    
            # Stop if all samples reach end token
            if (next_token == end_token).all():
                break
    
        return generated
    
    
    def generate_beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5):
        """
        Beam Search for sequence generation
    
        Args:
            model: Transformer model
            src: (1, src_len) - source sequence (batch size 1)
            src_mask: (1, 1, src_len)
            max_len: maximum generation length
            start_token: start token ID
            end_token: end token ID
            beam_size: beam size
    
        Returns:
            best_sequence: (1, gen_len) - best generated sequence
        """
        model.eval()
        device = src.device
    
        # Encoder
        memory = model.encode(src, src_mask)  # (1, src_len, d_model)
        memory = memory.repeat(beam_size, 1, 1)  # (beam_size, src_len, d_model)
    
        # Initialize beams
        beams = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # Only first beam is active initially
    
        finished_beams = []
    
        for step in range(max_len - 1):
            tgt_len = beams.size(1)
            tgt_mask = create_causal_mask(tgt_len).to(device)
    
            # Decoder
            output = model.decode(beams, memory, tgt_mask, src_mask.repeat(beam_size, 1, 1))
            next_token_logits = output[:, -1, :]  # (beam_size, vocab_size)
    
            # Log probabilities
            log_probs = F.log_softmax(next_token_logits, dim=-1)
    
            # Update beam scores
            vocab_size = log_probs.size(-1)
            scores = beam_scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)
            scores = scores.view(-1)  # (beam_size * vocab_size)
    
            # Top-k selection
            top_scores, top_indices = scores.topk(beam_size, largest=True)
    
            # New beams
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
    
            new_beams = []
            new_scores = []
    
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_scores)):
                # Extend beam
                new_beam = torch.cat([beams[beam_idx], token_idx.unsqueeze(0)])
    
                # Add to finished beams if end token is reached
                if token_idx == end_token:
                    finished_beams.append((new_beam, score.item()))
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
    
            # Stop if enough finished beams
            if len(finished_beams) >= beam_size:
                break
    
            # Stop if no beams remaining
            if len(new_beams) == 0:
                break
    
            # Update beams
            beams = torch.stack(new_beams)
            beam_scores = torch.tensor(new_scores, device=device)
    
        # Select best beam
        if finished_beams:
            best_beam, best_score = max(finished_beams, key=lambda x: x[1])
        else:
            best_beam = beams[0]
    
        return best_beam.unsqueeze(0)
    
    
    # Verification
    print("\n=== Autoregressive Generation Test ===")
    
    # Dummy model and data
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
    
    print(f"Source sequence: {src.shape}")
    print(f"Greedy generation: {generated_greedy.shape}")
    print(f"Generated sequence: {generated_greedy[0].tolist()}")
    
    # Beam Search
    with torch.no_grad():
        generated_beam = generate_beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5)
    
    print(f"\nBeam Search generation: {generated_beam.shape}")
    print(f"Generated sequence: {generated_beam[0].tolist()}")
    

* * *

## 2.5 Practice: Machine Translation System

### Dataset Preparation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from collections import Counter
    import re
    
    class TranslationDataset(Dataset):
        """Simple translation dataset"""
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
    
            # Convert to token IDs
            src_ids = [self.src_vocab.get(w, self.src_vocab['<unk>']) for w in src.split()]
            tgt_ids = [self.tgt_vocab.get(w, self.tgt_vocab['<unk>']) for w in tgt.split()]
    
            return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
    
    def build_vocab(sentences, max_vocab_size=10000):
        """Build vocabulary"""
        words = []
        for sent in sentences:
            words.extend(sent.split())
    
        # Count frequencies
        word_counts = Counter(words)
        most_common = word_counts.most_common(max_vocab_size - 4)  # Exclude special tokens
    
        # Create vocabulary dictionary
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for word, _ in most_common:
            vocab[word] = len(vocab)
    
        return vocab
    
    
    # Dummy data (in practice, use Multi30k, WMT, etc.)
    src_sentences = [
        "i love machine learning",
        "transformers are powerful",
        "attention is all you need",
        "deep learning is amazing",
        "natural language processing"
    ]
    
    tgt_sentences = [
        "i love machine learning",
        "transformers are powerful",
        "attention is all you need",
        "deep learning is amazing",
        "natural language processing"
    ]
    
    # Build vocabulary
    src_vocab = build_vocab(src_sentences)
    tgt_vocab = build_vocab(tgt_sentences)
    
    print("=== Translation Dataset Preparation ===")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    print(f"\nSource vocabulary (partial): {list(src_vocab.items())[:10]}")
    print(f"Target vocabulary (partial): {list(tgt_vocab.items())[:10]}")
    
    # Create dataset
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    
    # Check sample
    src_sample, tgt_sample = dataset[0]
    print(f"\nSample 0:")
    print(f"Source: {src_sentences[0]}")
    print(f"Source ID: {src_sample.tolist()}")
    print(f"Target: {tgt_sentences[0]}")
    print(f"Target ID: {tgt_sample.tolist()}")
    </unk></eos></sos></pad></unk></unk>

### Training Loop Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    
    def collate_fn(batch, src_vocab, tgt_vocab):
        """Batch collate function"""
        src_batch, tgt_batch = zip(*batch)
    
        # Padding
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<pad>'])
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<pad>'])
    
        return src_padded, tgt_padded
    
    
    def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
        """Create masks"""
        # Source padding mask
        src_mask = (src != src_pad_idx).unsqueeze(1)  # (batch, 1, src_len)
    
        # Target causal mask + padding mask
        tgt_len = tgt.size(1)
        tgt_mask = create_causal_mask(tgt_len).to(tgt.device)  # (1, tgt_len, tgt_len)
        tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        tgt_mask = tgt_mask & tgt_pad_mask
    
        return src_mask, tgt_mask
    
    
    def train_epoch(model, dataloader, optimizer, criterion, src_vocab, tgt_vocab, device):
        """Train one epoch"""
        model.train()
        total_loss = 0
    
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
    
            # Split target into input and teacher data
            tgt_input = tgt[:, :-1]  # Exclude <eos>
            tgt_output = tgt[:, 1:]  # Exclude <sos>
    
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input,
                                             src_vocab['<pad>'], tgt_vocab['<pad>'])
    
            # Forward
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
    
            # Calculate loss (ignore padding)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
    
            loss = criterion(output, tgt_output)
    
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            total_loss += loss.item()
    
        return total_loss / len(dataloader)
    
    
    # Training configuration
    print("\n=== Translation Model Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
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
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, src_vocab, tgt_vocab, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
    
    print("\nTraining complete!")
    </pad></pad></pad></sos></eos></pad></pad>

### Translation Inference
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    
    def translate(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
        """Translate sentence"""
        model.eval()
    
        # Convert source sentence to token IDs
        src_tokens = src_sentence.split()
        src_ids = [src_vocab.get(w, src_vocab['<unk>']) for w in src_tokens]
        src = torch.tensor(src_ids).unsqueeze(0).to(device)  # (1, src_len)
    
        # Source mask
        src_mask = torch.ones(1, 1, src.size(1)).to(device)
    
        # Generate with Greedy Decoding
        with torch.no_grad():
            generated = generate_greedy(
                model, src, src_mask, max_len,
                start_token=tgt_vocab['<sos>'],
                end_token=tgt_vocab['<eos>']
            )
    
        # Convert token IDs to words
        idx_to_word = {v: k for k, v in tgt_vocab.items()}
        translated = [idx_to_word.get(idx.item(), '<unk>') for idx in generated[0]]
    
        # Remove <sos> and <eos>
        if translated[0] == '<sos>':
            translated = translated[1:]
        if '<eos>' in translated:
            eos_idx = translated.index('<eos>')
            translated = translated[:eos_idx]
    
        return ' '.join(translated)
    
    
    # Translation test
    print("\n=== Translation Test ===")
    
    test_sentences = [
        "i love machine learning",
        "transformers are powerful",
        "attention is all you need"
    ]
    
    for src_sent in test_sentences:
        translated = translate(model, src_sent, src_vocab, tgt_vocab, device)
        print(f"Source: {src_sent}")
        print(f"Translation: {translated}")
        print()
    
    print("→ Not perfect due to small dataset, but basic translation functionality implemented")
    </eos></eos></sos></eos></sos></unk></eos></sos></unk>

* * *

## 2.6 Transformer Training Techniques

### Learning Rate Warmup

For Transformer training, a **warmup scheduler** is important. It keeps the learning rate small initially, gradually increases it, and then decays.

$$ \text{lr}(step) = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot \text{warmup_steps}^{-1.5}) $$ 
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch.optim as optim
    
    class NoamOpt:
        """Noam learning rate scheduler (paper implementation)"""
        def __init__(self, d_model, warmup_steps, optimizer):
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            self.optimizer = optimizer
            self._step = 0
            self._rate = 0
    
        def step(self):
            """Update one step"""
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()
    
        def rate(self, step=None):
            """Calculate current learning rate"""
            if step is None:
                step = self._step
            return (self.d_model ** (-0.5)) * min(step ** (-0.5),
                                                   step * self.warmup_steps ** (-1.5))
    
    # Usage example
    print("=== Noam Learning Rate Scheduler ===")
    d_model = 512
    warmup_steps = 4000
    
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(d_model, warmup_steps, optimizer)
    
    # Visualize learning rate progression
    steps = list(range(1, 20000))
    lrs = [scheduler.rate(step) for step in steps]
    
    print(f"Initial learning rate (step=1): {lrs[0]:.6f}")
    print(f"Peak learning rate (step={warmup_steps}): {lrs[warmup_steps-1]:.6f}")
    print(f"Late learning rate (step=20000): {lrs[-1]:.6f}")
    print("→ Gradually increases during warmup, then decays")
    

### Label Smoothing

**Label Smoothing** is a regularization technique that sets the probability of the correct label to around 0.9 instead of 1, distributing some probability to other classes.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LabelSmoothingLoss(nn.Module):
        """Cross Entropy Loss with Label Smoothing"""
        def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
            super(LabelSmoothingLoss, self).__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.ignore_index = ignore_index
            self.confidence = 1.0 - smoothing
    
        def forward(self, pred, target):
            """
            pred: (batch * seq_len, num_classes) - logits
            target: (batch * seq_len) - correct labels
            """
            # Log-softmax
            log_probs = F.log_softmax(pred, dim=-1)
    
            # Get probability at correct position
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    
            # Average log probability over all classes
            smooth_loss = -log_probs.mean(dim=-1)
    
            # Combine
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            # Mask ignore_index
            if self.ignore_index >= 0:
                mask = (target != self.ignore_index).float()
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
    
            return loss
    
    
    # Comparison
    print("\n=== Effect of Label Smoothing ===")
    
    num_classes = 10
    criterion_normal = nn.CrossEntropyLoss()
    criterion_smooth = LabelSmoothingLoss(num_classes, smoothing=0.1)
    
    # Dummy data
    pred = torch.randn(32, num_classes)
    target = torch.randint(0, num_classes, (32,))
    
    loss_normal = criterion_normal(pred, target)
    loss_smooth = criterion_smooth(pred, target)
    
    print(f"Normal Cross Entropy Loss: {loss_normal.item():.4f}")
    print(f"Label Smoothing Loss: {loss_smooth.item():.4f}")
    print("→ Label Smoothing prevents overconfidence and improves generalization")
    

### Mixed Precision Training
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Mixed Precision Training
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from torch.cuda.amp import autocast, GradScaler
    
    # Mixed Precision Training (when GPU is available)
    if torch.cuda.is_available():
        print("\n=== Mixed Precision Training Example ===")
    
        device = torch.device('cuda')
        model = model.to(device)
    
        scaler = GradScaler()
    
        # Part of training loop
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
    
            optimizer.zero_grad()
    
            # Compute with Mixed Precision
            with autocast():
                output = model(src, tgt_input)
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                loss = criterion(output, tgt_output)
    
            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        print("→ FP16 computation accelerates and reduces memory (up to 2x faster)")
    else:
        print("\n=== Mixed Precision Training ===")
        print("GPU not available, skipping")
    

* * *

## Exercises

**Exercise 1: Effect of Multi-Head Attention Head Count**

Train models with different head counts (1, 2, 4, 8, 16) and compare performance and computational cost.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Train models with different head counts (1, 2, 4, 8, 16) and
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Create multiple models with different head counts
    # Exercise: Train on same data and compare performance, training time, parameter count
    # Exercise: Analyze role distribution across heads using Attention visualization
    # Hint: More heads improve performance but increase computational cost
    

**Exercise 2: Positional Encoding Experiments**

Compare Sin/Cos positional encoding with learnable positional embeddings.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare Sin/Cos positional encoding with learnable positiona
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Implement two types of positional encoding
    # 1. Sin/Cos (fixed)
    # 2. nn.Embedding (learnable)
    
    # Exercise: Compare performance on same task
    # Exercise: Evaluate generalization to longer sequences (test on sequences longer than training)
    # Expected: Sin/Cos can generalize to arbitrary lengths
    

**Exercise 3: Causal Mask Visualization**

Visualize how the Decoder's causal mask affects Attention weights.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Visualize how the Decoder's causal mask affects Attention we
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import matplotlib.pyplot as plt
    
    # Exercise: Visualize Attention weights with and without mask
    

**Exercise 4: Beam Search Beam Size Optimization**

Compare translation quality and speed with different beam sizes (1, 3, 5, 10, 20).
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare translation quality and speed with different beam si
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import time
    
    # Exercise: Create graph of beam size vs quality and speed
    # Expected: Beam size 5-10 provides good balance of quality and speed
    

**Exercise 5: Investigate Effect of Layer Count**

Compare performance with different Encoder/Decoder layer counts (1, 2, 4, 6, 12).
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare performance with different Encoder/Decoder layer cou
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Create models with different layer counts
    # Exercise: Record training loss, validation loss, parameter count, training time
    # Exercise: Create graph of layer count vs performance
    # Analysis: Too deep causes overfitting and longer training, too shallow lacks expressiveness
    

* * *

## Summary

In this chapter, we learned the complete architecture of the Transformer.

### Key Points

**1\. Transformer Structure** — Encoder-Decoder architecture with 6-layer stacks.

**2\. Encoder** — Multi-Head Self-Attention + FFN, fully parallelizable.

**3\. Decoder** — Masked Self-Attention + Cross-Attention + FFN for autoregressive generation.

**4\. Multi-Head Attention** — Captures dependencies from multiple perspectives simultaneously.

**5\. Positional Encoding** — Injects positional information using Sin/Cos functions.

**6\. Residual + Layer Norm** — Stabilizes training even in deep layers.

**7\. Causal Mask** — Controls attention to prevent seeing future tokens.

**8\. Autoregressive Generation** — Greedy Decoding and Beam Search strategies.

**9\. Training Techniques** — Warmup scheduling, Label Smoothing, and Mixed Precision training.

**10\. Practice** — Complete implementation of a machine translation system.

### Next Steps

In the next chapter, we will learn about **Training and Optimization of Transformers**. You will master practical techniques including efficient training methods, data augmentation, evaluation metrics, and hyperparameter tuning.
