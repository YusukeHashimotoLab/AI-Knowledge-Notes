---
title: "Chapter 3: Transformer and BERT"
chapter_title: "Chapter 3: Transformer and BERT"
subtitle: From Attention Mechanism to Pre-trained Models - The Revolution in Natural Language Processing
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
version: 1.0
created_at: "by:"
---

This chapter covers Transformer and BERT. You will learn mechanisms of the Transformer architecture, necessity of Positional Encoding, and Execute pre-training.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the mechanisms of the Transformer architecture
  * ✅ Implement Self-Attention and Multi-Head Attention
  * ✅ Explain the necessity of Positional Encoding
  * ✅ Execute pre-training and fine-tuning of BERT
  * ✅ Master the HuggingFace Transformers library
  * ✅ Apply Japanese BERT models in practical tasks

* * *

## 3.1 Transformer Architecture

### Birth of the Transformer

The **Transformer** is an architecture proposed in the "Attention is All You Need" paper published by Google in 2017. It achieved sequence processing using only **Self-Attention mechanisms** , without using RNNs or LSTMs.

> "Eliminate sequential processing of RNNs and compute relationships between all tokens in parallel"

### Advantages of Transformer

Aspect | RNN/LSTM | Transformer  
---|---|---  
**Parallelization** | Sequential processing (slow) | Fully parallel (fast)  
**Long-range Dependencies** | Difficult due to gradient vanishing | Easy with direct connections  
**Computational Complexity** | O(n) | O(n²)  
**Interpretability** | Low | High with Attention visualization  
  
### Overall Architecture
    
    
    ```mermaid
    graph TB
        A[Input Sentence] --> B[Input Embedding]
        B --> C[Positional Encoding]
        C --> D[Encoder Stack]
        D --> E[Decoder Stack]
        E --> F[Linear + Softmax]
        F --> G[Output Sentence]
    
        D --> |Context| E
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#fff9c4
        style G fill:#e0f2f1
    ```

* * *

## 3.2 Self-Attention Mechanism

### Principles of Self-Attention

**Self-Attention** is a mechanism where each token in the input sequence computes its relationship with all other tokens.

It uses three weight matrices:

  * $\mathbf{W}_Q$: Query matrix
  * $\mathbf{W}_K$: Key matrix
  * $\mathbf{W}_V$: Value matrix

### Calculation Steps

**Step 1: Compute Query, Key, Value**

$$ \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V $$

**Step 2: Compute Attention Score**

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} $$

  * $d_k$: Dimensionality of Key (scaling factor)

### Implementation Example: Scaled Dot-Product Attention
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention
    
        Args:
            Q: Query matrix (batch_size, seq_len, d_k)
            K: Key matrix (batch_size, seq_len, d_k)
            V: Value matrix (batch_size, seq_len, d_v)
            mask: Mask (optional)
    
        Returns:
            output: Output after applying Attention
            attention_weights: Attention weights
        """
        d_k = Q.shape[-1]
    
        # Compute Attention Score
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
        # Apply mask (optional)
        if mask is not None:
            scores = scores + (mask * -1e9)
    
        # Softmax
        attention_weights = softmax(scores, axis=-1)
    
        # Weighted sum
        output = np.matmul(attention_weights, V)
    
        return output, attention_weights
    
    def softmax(x, axis=-1):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # Usage example
    batch_size, seq_len, d_model = 2, 5, 64
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (first sample):\n{weights[0]}")
    

**Output** :
    
    
    Output shape: (2, 5, 64)
    Attention weights shape: (2, 5, 5)
    
    Attention weights (first sample):
    [[0.21 0.19 0.20 0.18 0.22]
     [0.20 0.21 0.19 0.20 0.20]
     [0.19 0.20 0.21 0.20 0.20]
     [0.20 0.20 0.19 0.21 0.20]
     [0.22 0.18 0.20 0.19 0.21]]
    

### PyTorch Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: PyTorch Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ScaledDotProductAttention(nn.Module):
        def __init__(self, d_k):
            super().__init__()
            self.d_k = d_k
    
        def forward(self, Q, K, V, mask=None):
            # Attention Score
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
    
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
    
            # Weighted sum
            output = torch.matmul(attention_weights, V)
    
            return output, attention_weights
    
    # Usage example
    d_model = 64
    attention = ScaledDotProductAttention(d_k=d_model)
    
    Q = torch.randn(2, 5, d_model)
    K = torch.randn(2, 5, d_model)
    V = torch.randn(2, 5, d_model)
    
    output, weights = attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    

* * *

## 3.3 Multi-Head Attention

### Overview

**Multi-Head Attention** executes multiple attention heads in parallel to capture information from different representation subspaces.
    
    
    ```mermaid
    graph LR
        A[Input X] --> B1[Head 1]
        A --> B2[Head 2]
        A --> B3[Head 3]
        A --> B4[Head h]
    
        B1 --> C[Concat]
        B2 --> C
        B3 --> C
        B4 --> C
    
        C --> D[Linear]
        D --> E[Output]
    
        style A fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style B3 fill:#fff3e0
        style B4 fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#e0f2f1
    ```

### Formulation

$$ \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O $$

Each head is:

$$ \text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V) $$

### Implementation Example
    
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            """
            Multi-Head Attention
    
            Args:
                d_model: Model dimensionality
                num_heads: Number of attention heads
            """
            super().__init__()
            assert d_model % num_heads == 0
    
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
    
            # Linear layers
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)
            self.W_O = nn.Linear(d_model, d_model)
    
            self.attention = ScaledDotProductAttention(self.d_k)
    
        def split_heads(self, x, batch_size):
            """Split into multiple heads"""
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)
    
            # Linear transformation
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
    
            # Split into multiple heads
            Q = self.split_heads(Q, batch_size)
            K = self.split_heads(K, batch_size)
            V = self.split_heads(V, batch_size)
    
            # Apply attention
            output, attention_weights = self.attention(Q, K, V, mask)
    
            # Concat
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, -1, self.d_model)
    
            # Final linear layer
            output = self.W_O(output)
    
            return output, attention_weights
    
    # Usage example
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    

**Output** :
    
    
    Output shape: torch.Size([2, 10, 512])
    Attention weights shape: torch.Size([2, 8, 10, 10])
    

* * *

## 3.4 Positional Encoding

### Necessity

Since Self-Attention has no order information, **Positional Encoding** adds positional information to the sequence.

### Sinusoidal Positional Encoding

$$ \text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

$$ \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

  * $pos$: Position of the token
  * $i$: Dimension index

### Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            """
            Positional Encoding
    
            Args:
                d_model: Model dimensionality
                max_len: Maximum sequence length
            """
            super().__init__()
    
            # Pre-compute Positional Encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
    
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            """
            Args:
                x: (batch_size, seq_len, d_model)
            """
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
    
    # Usage example
    d_model = 128
    max_len = 100
    
    pe_layer = PositionalEncoding(d_model, max_len)
    x = torch.randn(2, 50, d_model)
    output = pe_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Visualize Positional Encoding
    pe_matrix = pe_layer.pe[0, :max_len, :].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Dimension', fontsize=12)
    plt.title('Positional Encoding (Sinusoidal)', fontsize=14)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Feed-Forward Networks

### Position-wise Feed-Forward Networks

A two-layer neural network applied independently to each position:

$$ \text{FFN}(x) = \max(0, x\mathbf{W}_1 + b_1)\mathbf{W}_2 + b_2 $$

### Implementation Example
    
    
    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            """
            Position-wise Feed-Forward Networks
    
            Args:
                d_model: Model dimensionality
                d_ff: Intermediate layer dimensionality (typically 4 * d_model)
                dropout: Dropout rate
            """
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
    
        def forward(self, x):
            # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
    
            # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
            x = self.linear2(x)
            return x
    
    # Usage example
    d_model = 512
    d_ff = 2048
    
    ffn = PositionwiseFeedForward(d_model, d_ff)
    x = torch.randn(2, 10, d_model)
    
    output = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    

* * *

## 3.6 BERT (Bidirectional Encoder Representations from Transformers)

### Features of BERT

**BERT** is a bidirectional pre-training model announced by Google in 2018.

> "Learn bidirectional context simultaneously, not just left-to-right"
    
    
    ```mermaid
    graph LR
        A[Large Corpus] --> B[Pre-training]
        B --> C[BERT Base/Large]
        C --> D1[Fine-tuning: Classification]
        C --> D2[Fine-tuning: NER]
        C --> D3[Fine-tuning: QA]
        C --> D4[Feature Extraction]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
        style D4 fill:#e8f5e9
    ```

### BERT Model Configuration

Model | Layers | Hidden Size | Attention Heads | Parameters  
---|---|---|---|---  
**BERT-Base** | 12 | 768 | 12 | 110M  
**BERT-Large** | 24 | 1024 | 16 | 340M  
  
### Pre-training Tasks

#### 1\. Masked Language Modeling (MLM)

Mask 15% of input tokens with [MASK] and predict them.
    
    
    Input:  The [MASK] is beautiful today.
    Target: The weather is beautiful today.
    

#### 2\. Next Sentence Prediction (NSP)

Predict whether two sentences are consecutive.
    
    
    Sentence A: The cat sat on the mat.
    Sentence B: It was very comfortable.
    Label: IsNext (1)
    
    Sentence A: The cat sat on the mat.
    Sentence B: The economy is growing.
    Label: NotNext (0)
    

### Getting Started with BERT using HuggingFace Transformers
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Getting Started with BERT using HuggingFace Transformers
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizer, BertModel
    import torch
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Encode text
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    print("Input IDs:", inputs['input_ids'])
    print("Attention Mask:", inputs['attention_mask'])
    
    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Outputs
    last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    pooler_output = outputs.pooler_output  # (batch_size, hidden_size) [CLS] token output
    
    print(f"\nLast Hidden States shape: {last_hidden_states.shape}")
    print(f"Pooler Output shape: {pooler_output.shape}")
    

**Output** :
    
    
    Tokens: ['[CLS]', 'hello', ',', 'how', 'are', 'you', '?', '[SEP]']
    Input IDs: tensor([[  101,  7592,  1010,  2129,  2024,  2017,  1029,   102]])
    Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    
    Last Hidden States shape: torch.Size([1, 8, 768])
    Pooler Output shape: torch.Size([1, 768])
    

* * *

## 3.7 Fine-tuning BERT

### Text Classification Task
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - transformers>=4.30.0
    
    """
    Example: Text Classification Task
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    from datasets import load_dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    # Load dataset (e.g., IMDb movie reviews)
    dataset = load_dataset('imdb')
    
    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Data preprocessing
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Test with small subset
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(200))
    
    # Evaluation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
    
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
    
        return {'accuracy': acc, 'f1': f1}
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Fine-tuning
    trainer.train()
    
    # Evaluation
    results = trainer.evaluate()
    print(f"\nEvaluation results: {results}")
    

### Named Entity Recognition (NER)
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: Named Entity Recognition (NER)
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from transformers import BertForTokenClassification, pipeline
    
    # Load NER model
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    ner_pipeline = pipeline('ner', model=model_name, tokenizer=model_name)
    
    # Extract named entities from text
    text = "Apple Inc. is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO."
    results = ner_pipeline(text)
    
    print("Named Entity Recognition Results:")
    for entity in results:
        print(f"{entity['word']}: {entity['entity']} (confidence: {entity['score']:.4f})")
    

**Output** :
    
    
    Named Entity Recognition Results:
    Apple: B-ORG (confidence: 0.9987)
    Inc: I-ORG (confidence: 0.9983)
    U: B-LOC (confidence: 0.9976)
    K: I-LOC (confidence: 0.9945)
    Tim: B-PER (confidence: 0.9995)
    Cook: I-PER (confidence: 0.9993)
    CEO: B-MISC (confidence: 0.8734)
    

### Question Answering
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    from transformers import pipeline
    
    # Load QA model
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # Context and question
    context = """
    The Transformer is a deep learning model introduced in 2017, used primarily in the field of
    natural language processing (NLP). Like recurrent neural networks (RNNs), Transformers are
    designed to handle sequential data, such as natural language, for tasks such as translation
    and text summarization. However, unlike RNNs, Transformers do not require that the sequential
    data be processed in order.
    """
    
    question = "When was the Transformer introduced?"
    
    # Question answering
    result = qa_pipeline(question=question, context=context)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")
    print(f"Start position: {result['start']}, End position: {result['end']}")
    

**Output** :
    
    
    Question: When was the Transformer introduced?
    Answer: 2017
    Confidence: 0.9812
    Start position: 50, End position: 54
    

* * *

## 3.8 Japanese BERT Models

### Representative Japanese BERT Models

Model | Provider | Tokenizer | Features  
---|---|---|---  
**Tohoku BERT** | Tohoku University | MeCab + WordPiece | Trained on Japanese Wikipedia  
**Kyoto BERT** | Kyoto University | Juman++ + WordPiece | High-quality morphological analysis  
**NICT BERT** | NICT | SentencePiece | Large corpus  
**Waseda RoBERTa** | Waseda University | SentencePiece | RoBERTa (without NSP)  
  
### Usage Example of Tohoku BERT
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Usage Example of Tohoku BERT
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertJapaneseTokenizer, BertModel
    import torch
    
    # Load Tohoku BERT
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Japanese text
    text = "Natural language processing is an important field of artificial intelligence."
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nLast Hidden States shape: {outputs.last_hidden_state.shape}")
    print(f"Pooler Output shape: {outputs.pooler_output.shape}")
    

### Japanese Text Classification
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Japanese Text Classification
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForSequenceClassification, BertJapaneseTokenizer
    import torch
    import torch.nn.functional as F
    
    # Load model and tokenizer
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    
    # Customize for sentiment analysis (e.g., positive/negative)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Example texts
    texts = [
        "This movie was truly wonderful!",
        "It was the worst experience and a waste of time.",
        "It was average and not particularly memorable."
    ]
    
    # Inference
    model.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
    
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
        label = "Positive" if predicted_class == 1 else "Negative"
        print(f"\nText: {text}")
        print(f"Prediction: {label} (confidence: {confidence:.4f})")
    

* * *

## 3.9 BERT Application Techniques

### Feature Extraction

Use BERT as a fixed feature extractor.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import BertModel, BertTokenizer
    import torch
    import numpy as np
    
    # Model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Evaluation mode
    
    def get_sentence_embedding(text, pooling='mean'):
        """
        Get sentence embedding vector
    
        Args:
            text: Input text
            pooling: Pooling method ('mean', 'max', 'cls')
    
        Returns:
            embedding: Embedding vector
        """
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Last Hidden States
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    
        if pooling == 'mean':
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        elif pooling == 'max':
            # Max pooling
            embedding = torch.max(last_hidden, 1)[0]
        elif pooling == 'cls':
            # [CLS] token
            embedding = outputs.pooler_output
    
        return embedding.squeeze().numpy()
    
    # Usage example
    texts = [
        "Natural language processing is fascinating.",
        "I love machine learning.",
        "The weather is nice today."
    ]
    
    embeddings = [get_sentence_embedding(text, pooling='mean') for text in texts]
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings)
    
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)
    print(f"\nSimilarity between sentence 1 and 2: {similarity_matrix[0, 1]:.4f}")
    print(f"Similarity between sentence 1 and 3: {similarity_matrix[0, 2]:.4f}")
    

### Sentence Embeddings

High-quality sentence embeddings using Sentence-BERT:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: High-quality sentence embeddings using Sentence-BERT:
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # List of sentences
    sentences = [
        "The cat sits on the mat.",
        "A feline rests on a rug.",
        "The dog plays in the park.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks."
    ]
    
    # Get embedding vectors
    embeddings = model.encode(sentences)
    
    print(f"Embedding vector shape: {embeddings.shape}")
    
    # Calculate similarity
    similarity = cosine_similarity(embeddings)
    
    print("\nSentence similarity matrix:")
    for i, sent in enumerate(sentences):
        print(f"\n{i}: {sent}")
    
    print("\nSimilarity:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"Sentence {i} and Sentence {j}: {similarity[i, j]:.4f}")
    

### Domain Adaptation

Perform additional pre-training with domain-specific data:
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: Perform additional pre-training with domain-specific data:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset
    
    # Medical domain example
    domain_texts = [
        "The patient's blood pressure is within normal range.",
        "Dietary therapy is important for diabetes treatment.",
        "This medication has risk of side effects.",
        # ... large amount of domain-specific text
    ]
    
    # Create dataset
    dataset = Dataset.from_dict({'text': domain_texts})
    
    # Tokenizer and model
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Data Collator (for MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir='./domain_adapted_bert',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Additional pre-training
    # trainer.train()  # Uncomment to execute
    
    print("Domain-adapted model ready")
    

* * *

## 3.10 Chapter Summary

### What We Learned

  1. **Transformer Architecture**

     * Principles and implementation of Self-Attention mechanism
     * Learning multiple representations with Multi-Head Attention
     * Adding positional information with Positional Encoding
     * Non-linear transformation with Feed-Forward Networks
  2. **BERT Fundamentals**

     * Importance of bidirectional pre-training
     * MLM and NSP tasks
     * How to use HuggingFace Transformers
  3. **Fine-tuning**

     * Text classification, NER, and QA tasks
     * Utilizing Japanese BERT models
  4. **Application Techniques**

     * Feature extraction and sentence embeddings
     * Domain adaptation
     * Practical applications

### To the Next Chapter

In Chapter 4, we will learn about **advanced BERT models** :

  * RoBERTa, ALBERT, DistilBERT
  * GPT series (GPT-2, GPT-3)
  * T5, BART (Seq2Seq models)
  * Latest large language models

* * *

## Practice Problems

### Problem 1 (Difficulty: easy)

Explain the difference between Self-Attention and Cross-Attention.

Answer Example

**Answer** :

  * **Self-Attention** : Query, Key, and Value are all generated from the same input sequence. Learns relationships between elements within the input sequence.
  * **Cross-Attention** : Query and Key/Value are generated from different sequences (e.g., Decoder's Query and Encoder's Key/Value). Learns relationships between different sequences.

**Use Cases** :

  * Self-Attention: BERT's Encoder, word dependencies within a sentence
  * Cross-Attention: Machine translation's Decoder, correspondence between source and target text

### Problem 2 (Difficulty: medium)

Explain why Positional Encoding is necessary and why Sinusoidal functions are used.

Answer Example

**Necessity** :

  * Self-Attention has no order information (permutation invariance)
  * Cannot distinguish "cat sat on mat" from "mat on sat cat"
  * Adding positional information preserves sequence order

**Reasons for Using Sinusoidal Functions** :

  1. **Handling Variable Length Sequences** : Can handle sequences longer than those seen during training
  2. **Representing Relative Positions** : $\text{PE}_{pos+k}$ can be expressed as a linear transformation of $\text{PE}_{pos}$
  3. **No Parameters Required** : No need for learning, low computational cost
  4. **Periodicity** : Captures short-term and long-term positional relationships with different frequencies

### Problem 3 (Difficulty: medium)

In BERT's MLM (Masked Language Modeling), explain the design rationale for masking 15% of input tokens.

Answer Example

**Rationale for 15%** :

  * **Balance** : Too low leads to slow learning, too high leads to insufficient context
  * **Experimentally Optimal Value** : Result of testing various percentages in the BERT paper

**Mask Breakdown** :

  * 80%: Replace with [MASK] token
  * 10%: Replace with random token
  * 10%: Keep original token

**Design Rationale** :

  1. 80% as [MASK]: Primary learning task
  2. 10% as random: Prevents model from relying only on [MASK]
  3. 10% as original: Learns representations of actual tokens

This ensures the model works appropriately during fine-tuning when [MASK] tokens don't appear.

### Problem 4 (Difficulty: hard)

Explain the advantages of using multiple heads in Multi-Head Attention compared to single Attention. Also discuss potential problems when using too many heads.

Answer Example

**Advantages of Multiple Heads** :

  1. **Different Representation Subspaces**

     * Each head learns different types of relationships
     * Examples: syntactic relationships, semantic relationships, long-range dependencies
  2. **Parallel Computation**

     * Multiple heads can be computed simultaneously
     * Efficient GPU processing
  3. **Redundancy and Robustness**

     * Other heads compensate if some heads fail
     * Captures diverse information
  4. **Ensemble Effect**

     * Integrates information from multiple perspectives
     * Learns richer representations

**Problems with Too Many Heads** :

  1. **Increased Computational Cost** : Increased memory and computation time
  2. **Overfitting Risk** : Overfitting due to increased parameters
  3. **Redundancy** : More heads with similar roles
  4. **Optimization Difficulty** : Difficult to coordinate many heads

**Recommendations in Practice** :

Model Size | Recommended Heads  
---|---  
Small (d=256) | 4-8  
Base (d=512-768) | 8-12  
Large (d=1024) | 12-16  
  
### Problem 5 (Difficulty: hard)

Complete the following code to implement a sentiment analysis model using BERT. Prepare your own data or generate sample data.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Complete the following code to implement a sentiment analysi
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    import torch
    
    # Implement dataset class
    class SentimentDataset(Dataset):
        # Implement here
        pass
    
    # Implement training function
    def train_model(model, train_loader, optimizer, device):
        # Implement here
        pass
    
    # Implement evaluation function
    def evaluate_model(model, test_loader, device):
        # Implement here
        pass
    

Answer Example
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Complete the following code to implement a sentiment analysi
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample data
    train_texts = [
        "This product is wonderful!",
        "It was the worst experience.",
        "It's average.",
        "I'm very satisfied.",
        "I'll never buy this again.",
    ]
    train_labels = [1, 0, 1, 1, 0]  # 1: Positive, 0: Negative
    
    test_texts = [
        "I think it's a good product.",
        "It was disappointing."
    ]
    test_labels = [1, 0]
    
    # Dataset class
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
    
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
    
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # Training function
    def train_model(model, train_loader, optimizer, device, epochs=3):
        model.train()
    
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
    
                loss = outputs.loss
                total_loss += loss.item()
    
                loss.backward()
                optimizer.step()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation function
    def evaluate_model(model, test_loader, device):
        model.eval()
        predictions = []
        true_labels = []
    
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
    
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
    
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
    
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))
    
        return accuracy
    
    # Main processing
    def main():
        # Tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)
    
        # Dataset and dataloader
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
        # Training
        print("Starting training...")
        train_model(model, train_loader, optimizer, device, epochs=3)
    
        # Evaluation
        print("\nStarting evaluation...")
        evaluate_model(model, test_loader, device)
    
    if __name__ == '__main__':
        main()
    

**Example Output** :
    
    
    Starting training...
    Epoch 1/3, Loss: 0.6234
    Epoch 2/3, Loss: 0.4521
    Epoch 3/3, Loss: 0.3012
    
    Starting evaluation...
    Accuracy: 1.0000
    
    Classification Report:
                  precision    recall  f1-score   support
    
        Negative       1.00      1.00      1.00         1
        Positive       1.00      1.00      1.00         1
    
        accuracy                           1.00         2
       macro avg       1.00      1.00      1.00         2
    weighted avg       1.00      1.00      1.00         2
    

* * *

## References

  1. Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS.
  2. Devlin, J., et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL.
  3. Liu, Y., et al. (2019). _RoBERTa: A Robustly Optimized BERT Pretraining Approach_. arXiv.
  4. Lan, Z., et al. (2020). _ALBERT: A Lite BERT for Self-supervised Learning of Language Representations_. ICLR.
  5. Sanh, V., et al. (2019). _DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter_. NeurIPS Workshop.
  6. HuggingFace Transformers Documentation. <https://huggingface.co/docs/transformers/>
  7. Tohoku University BERT Model. <https://github.com/cl-tohoku/bert-japanese>
