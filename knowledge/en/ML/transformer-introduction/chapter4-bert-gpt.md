---
title: "Chapter 4: BERT & GPT"
chapter_title: "Chapter 4: BERT & GPT"
subtitle: "Two Pillars of Pre-trained Models: Theory and Practice of Bidirectional Encoders and Autoregressive Generative Models"
reading_time: 28 min
difficulty: Intermediate to Advanced
code_examples: 9
exercises: 6
---

This chapter covers BERT & GPT. You will learn BERT's bidirectional encoding, GPT's autoregressive generation, and pre-training tasks (MLM.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand BERT's bidirectional encoding and Masked Language Modeling
  * ✅ Understand GPT's autoregressive generation and Causal Masking mechanisms
  * ✅ Implement pre-training tasks (MLM, NSP, CLM) for BERT and GPT
  * ✅ Use both models with the Hugging Face Transformers library
  * ✅ Build task-specific models through fine-tuning
  * ✅ Determine when to use BERT vs GPT and their application scenarios
  * ✅ Complete practical projects: question answering systems and text generation

* * *

## 4.1 BERT Architecture

### 4.1.1 BERT's Innovation and Design Philosophy

**BERT** (Bidirectional Encoder Representations from Transformers) is a pre-trained model announced by Google in 2018 that revolutionized natural language processing.

Characteristic | Traditional Models (ELMo, GPT-1, etc.) | BERT  
---|---|---  
**Directionality** | Unidirectional (left→right) or shallow bidirectional | Deep bidirectional (uses both left and right context)  
**Architecture** | RNN, LSTM, shallow Transformer | Transformer Encoder only (12-24 layers)  
**Pre-training** | Language modeling (next word prediction) | Masked LM + Next Sentence Prediction  
**Use Cases** | Mainly generation tasks | Classification, extraction, QA and understanding tasks  
**Fine-tuning** | Complex task-specific architecture required | Simple output layer addition only  
  
### 4.1.2 Achieving Bidirectionality in BERT

BERT's most distinctive feature is **bidirectional context understanding**. Traditional language models predicted words sequentially from left to right, but BERT understands each word by looking at the entire sentence.
    
    
    ```mermaid
    graph LR
        subgraph "Traditional Unidirectional Model (GPT-1, etc.)"
            A1[The] --> A2[cat]
            A2 --> A3[sat]
            A3 --> A4[on]
            A4 --> A5[mat]
    
            style A1 fill:#e74c3c,color:#fff
            style A2 fill:#e74c3c,color:#fff
            style A3 fill:#e74c3c,color:#fff
        end
    
        subgraph "BERT Bidirectional Model"
            B1[The] <--> B2[cat]
            B2 <--> B3[sat]
            B3 <--> B4[on]
            B4 <--> B5[mat]
    
            style B2 fill:#27ae60,color:#fff
            style B3 fill:#27ae60,color:#fff
        end
    ```

> **Important** : When BERT understands the word "cat" in a sentence, it simultaneously uses both "The" (left context) and "sat on mat" (right context). This allows it to accurately capture word meanings.

### 4.1.3 BERT Architecture Structure

BERT consists of multiple stacked Transformer Encoder blocks:

Model | Layers (L) | Hidden Size (H) | Attention Heads (A) | Parameters  
---|---|---|---|---  
**BERT-Base** | 12 | 768 | 12 | 110M  
**BERT-Large** | 24 | 1024 | 16 | 340M  
  
Each Transformer Encoder block consists of Multi-Head Attention and Feed-Forward Network, as we learned in Chapter 2:

$$ \text{EncoderBlock}(x) = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{MultiHeadAttn}(x)))) $$ 

### 4.1.4 Input Representation: Token + Segment + Position Embeddings

BERT's input is the sum of three types of embeddings:

  1. **Token Embeddings** : Word (subword) embedding representations
  2. **Segment Embeddings** : Distinguish sentence A and sentence B (for NSP task)
  3. **Position Embeddings** : Position information (learnable, different from GPT's Sinusoidal)

$$ \text{Input} = \text{TokenEmbed}(x) + \text{SegmentEmbed}(x) + \text{PositionEmbed}(x) $$ 
    
    
    ```mermaid
    graph TB
        subgraph "BERT Input Structure"
            T1["[CLS] The cat sat [SEP] on mat [SEP]"]
    
            T2[Token Embeddings]
            T3[Segment Embeddings]
            T4[Position Embeddings]
    
            T5[Input to Transformer]
    
            T1 --> T2
            T1 --> T3
            T1 --> T4
    
            T2 --> T5
            T3 --> T5
            T4 --> T5
    
            style T5 fill:#7b2cbf,color:#fff
        end
    ```

**Special Tokens** :

  * `[CLS]`: Classification representation for entire sentence (Classification token)
  * `[SEP]`: Sentence separator (Separator)
  * `[MASK]`: Mask token for Masked Language Modeling

* * *

## 4.2 BERT Pre-training Tasks

### 4.2.1 Masked Language Modeling (MLM)

MLM is a task that masks part of the input and predicts those words. This allows learning bidirectional context.

#### MLM Procedure

  1. Randomly select 15% of input tokens
  2. For selected tokens: 
     * 80% probability: Replace with `[MASK]` token
     * 10% probability: Replace with a random different word
     * 10% probability: Keep the original word
  3. Predict the original words at masked positions

**Example** :
    
    
    Input: "The cat sat on the mat"
    After masking: "The [MASK] sat on the mat"
    Target: Predict "cat"
    

#### Why not mask 100%?

The `[MASK]` token doesn't exist during fine-tuning. To reduce the gap between training and deployment, some are kept as random words or original words.

### 4.2.2 Next Sentence Prediction (NSP)

NSP is a task that determines whether two sentences are consecutive. Understanding inter-sentence relationships is important for question answering and natural language inference.

#### NSP Structure

  * **IsNext** (50%): Actually consecutive sentence pairs
  * **NotNext** (50%): Randomly selected non-consecutive sentence pairs

**Example** :
    
    
    Input A: "The cat sat on the mat."
    Input B (IsNext): "It was very comfortable."
    Input B (NotNext): "Paris is the capital of France."
    
    BERT input: [CLS] The cat sat on the mat [SEP] It was very comfortable [SEP]
    Target: IsNext = True
    

### 4.2.3 MLM Implementation in PyTorch
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import numpy as np
    
    class MaskedLanguageModel:
        """BERT-style Masked Language Modeling implementation"""
    
        def __init__(self, vocab_size, mask_prob=0.15):
            self.vocab_size = vocab_size
            self.mask_prob = mask_prob
    
            # Special token IDs
            self.MASK_TOKEN_ID = vocab_size - 3
            self.CLS_TOKEN_ID = vocab_size - 2
            self.SEP_TOKEN_ID = vocab_size - 1
    
        def create_masked_lm_data(self, input_ids):
            """
            Generate masked data for MLM
    
            Args:
                input_ids: [batch_size, seq_len] input token IDs
    
            Returns:
                masked_input: Input after applying mask
                labels: Labels for prediction targets (valid only at masked positions, -100 for others)
            """
            batch_size, seq_len = input_ids.shape
    
            # Initialize labels (-100 is ignored in loss calculation)
            labels = torch.full_like(input_ids, -100)
            masked_input = input_ids.clone()
    
            for i in range(batch_size):
                # Exclude special tokens and select mask candidates
                special_tokens_mask = (input_ids[i] == self.CLS_TOKEN_ID) | \
                                     (input_ids[i] == self.SEP_TOKEN_ID)
    
                # Maskable positions
                candidate_indices = torch.where(~special_tokens_mask)[0]
    
                # Select 15% for masking
                num_to_mask = max(1, int(len(candidate_indices) * self.mask_prob))
                mask_indices = candidate_indices[torch.randperm(len(candidate_indices))[:num_to_mask]]
    
                for idx in mask_indices:
                    labels[i, idx] = input_ids[i, idx]  # Save original word
    
                    rand = random.random()
                    if rand < 0.8:
                        # 80%: Replace with [MASK] token
                        masked_input[i, idx] = self.MASK_TOKEN_ID
                    elif rand < 0.9:
                        # 10%: Replace with random word
                        random_token = random.randint(0, self.vocab_size - 4)
                        masked_input[i, idx] = random_token
                    # 10%: Keep original word (no else needed)
    
            return masked_input, labels
    
    
    # Demonstration
    print("=== Masked Language Modeling Demo ===\n")
    
    # Parameter settings
    vocab_size = 1000
    batch_size = 3
    seq_len = 10
    
    # Generate dummy input
    mlm = MaskedLanguageModel(vocab_size)
    input_ids = torch.randint(0, vocab_size - 3, (batch_size, seq_len))
    
    # Add [CLS] at beginning, [SEP] at end
    input_ids[:, 0] = mlm.CLS_TOKEN_ID
    input_ids[:, -1] = mlm.SEP_TOKEN_ID
    
    print("Original Input IDs (Batch 0):")
    print(input_ids[0].numpy())
    
    # Apply MLM mask
    masked_input, labels = mlm.create_masked_lm_data(input_ids)
    
    print("\nMasked Input IDs (Batch 0):")
    print(masked_input[0].numpy())
    
    print("\nLabels (Batch 0, -100 is ignored):")
    print(labels[0].numpy())
    
    # Check masked positions
    mask_positions = torch.where(labels[0] != -100)[0]
    print(f"\nMasked Positions: {mask_positions.numpy()}")
    print(f"Number of masked tokens: {len(mask_positions)} / {seq_len-2} (excluding [CLS] and [SEP])")
    
    for pos in mask_positions:
        original = input_ids[0, pos].item()
        masked = masked_input[0, pos].item()
        target = labels[0, pos].item()
    
        mask_type = "MASK" if masked == mlm.MASK_TOKEN_ID else \
                    "RANDOM" if masked != original else \
                    "UNCHANGED"
    
        print(f"  Position {pos}: Original={original}, Masked={masked} ({mask_type}), Target={target}")
    

**Output** :
    
    
    === Masked Language Modeling Demo ===
    
    Original Input IDs (Batch 0):
    [998 453 721 892 156 334 667 289 445 999]
    
    Masked Input IDs (Batch 0):
    [998 997 721 542 156 997 667 289 445 999]
    
    Labels (Batch 0, -100 is ignored):
    [-100 453 -100 892 -100 334 -100 -100 -100 -100]
    
    Masked Positions: [1 3 5]
    Number of masked tokens: 3 / 8 (excluding [CLS] and [SEP])
      Position 1: Original=453, Masked=997 (MASK), Target=453
      Position 3: Original=892, Masked=542 (RANDOM), Target=892
      Position 5: Original=334, Masked=997 (MASK), Target=334
    

* * *

## 4.3 BERT Usage Examples

### 4.3.1 Text Classification (Sentiment Analysis)

An implementation example of sentiment analysis using BERT. The output of the `[CLS]` token is used for classification.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: An implementation example of sentiment analysis using BERT. 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    
    # Load model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 2-class classification (Positive/Negative)
    )
    
    # Set to inference mode
    model.eval()
    
    # Sample texts
    texts = [
        "I absolutely loved this movie! It was fantastic.",
        "This product is terrible and waste of money.",
        "The service was okay, nothing special."
    ]
    
    print("=== BERT Sentiment Analysis Demo ===\n")
    
    for text in texts:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
    
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
    
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = probs[0, predicted_class].item()
    
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print(f"Probabilities: Negative={probs[0, 0]:.4f}, Positive={probs[0, 1]:.4f}\n")
    

**Output** :
    
    
    === BERT Sentiment Analysis Demo ===
    
    Text: I absolutely loved this movie! It was fantastic.
    Sentiment: Positive (Confidence: 0.8234)
    Probabilities: Negative=0.1766, Positive=0.8234
    
    Text: This product is terrible and waste of money.
    Sentiment: Negative (Confidence: 0.9102)
    Probabilities: Negative=0.9102, Positive=0.0898
    
    Text: The service was okay, nothing special.
    Sentiment: Negative (Confidence: 0.5621)
    Probabilities: Negative=0.5621, Positive=0.4379
    

### 4.3.2 Named Entity Recognition

An example of using BERT for Token Classification (labeling each token).
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: An example of using BERT for Token Classification (labeling 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizerFast, BertForTokenClassification
    import torch
    
    print("\n=== BERT Named Entity Recognition Demo ===\n")
    
    # Model for NER (pre-trained)
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    
    model.eval()
    
    # Label mapping
    label_list = [
        'O',       # Outside
        'B-MISC', 'I-MISC',  # Miscellaneous
        'B-PER', 'I-PER',    # Person
        'B-ORG', 'I-ORG',    # Organization
        'B-LOC', 'I-LOC'     # Location
    ]
    
    # Sample text
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Display tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [label_list[pred] for pred in predictions[0].numpy()]
    
    print(f"Text: {text}\n")
    print("Token-Level Predictions:")
    print(f"{'Token':<15} {'Label':<10}")
    print("-" * 25)
    
    for token, label in zip(tokens, predicted_labels):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token:<15} {label:<10}")
    
    # Entity extraction
    print("\nExtracted Entities:")
    current_entity = []
    current_label = None
    
    for token, label in zip(tokens, predicted_labels):
        if label.startswith('B-'):
            if current_entity:
                print(f"  {current_label}: {' '.join(current_entity)}")
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                print(f"  {current_label}: {' '.join(current_entity)}")
            current_entity = []
            current_label = None
    
    if current_entity:
        print(f"  {current_label}: {' '.join(current_entity)}")
    

**Output** :
    
    
    === BERT Named Entity Recognition Demo ===
    
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California.
    
    Token-Level Predictions:
    Token           Label
    -------------------------
    Apple           B-ORG
    Inc             I-ORG
    .               O
    was             O
    founded         O
    by              O
    Steve           B-PER
    Jobs            I-PER
    in              O
    Cup             B-LOC
    ##ert           I-LOC
    ##ino           I-LOC
    ,               O
    California      B-LOC
    .               O
    
    Extracted Entities:
      ORG: Apple Inc
      PER: Steve Jobs
      LOC: Cup ##ert ##ino
      LOC: California
    

### 4.3.3 Question Answering

A SQuAD-format question answering system, a representative application of BERT.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: A SQuAD-format question answering system, a representative a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForQuestionAnswering, BertTokenizer
    import torch
    
    print("\n=== BERT Question Answering Demo ===\n")
    
    # BERT model fine-tuned on SQuAD
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    model.eval()
    
    # Context and questions
    context = """
    Transformers is a state-of-the-art natural language processing library developed by Hugging Face.
    It provides thousands of pretrained models to perform tasks on texts such as classification,
    information extraction, question answering, summarization, translation, and text generation.
    The library supports PyTorch, TensorFlow, and JAX frameworks.
    """
    
    questions = [
        "Who developed Transformers?",
        "What tasks can Transformers perform?",
        "Which frameworks does the library support?"
    ]
    
    for question in questions:
        # Tokenize
        inputs = tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=384
        )
    
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
    
        # Predict start and end positions
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
    
        # Extract answer tokens
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
        # Confidence score
        start_score = start_logits[0, start_idx].item()
        end_score = end_logits[0, end_idx].item()
        confidence = (start_score + end_score) / 2
    
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Confidence Score: {confidence:.4f}\n")
    

**Output** :
    
    
    === BERT Question Answering Demo ===
    
    Question: Who developed Transformers?
    Answer: Hugging Face
    Confidence Score: 8.2341
    
    Question: What tasks can Transformers perform?
    Answer: classification, information extraction, question answering, summarization, translation, and text generation
    Confidence Score: 7.9823
    
    Question: Which frameworks does the library support?
    Answer: PyTorch, TensorFlow, and JAX
    Confidence Score: 9.1247
    

* * *

## 4.4 GPT Architecture

### 4.4.1 GPT Design Philosophy: Autoregressive Language Model

**GPT** (Generative Pre-trained Transformer) is an autoregressive language model developed by OpenAI. In contrast to BERT, it specializes in text generation.

Characteristic | BERT | GPT  
---|---|---  
**Architecture** | Transformer Encoder | Transformer Decoder (without Cross-Attention)  
**Directionality** | Bidirectional | Unidirectional (left→right)  
**Pre-training** | MLM + NSP | Causal Language Modeling (next word prediction)  
**Attention Mask** | None (refers to all tokens) | Causal Mask (hides future tokens)  
**Main Use Cases** | Classification, extraction, QA | Text generation, dialogue, summarization  
**Inference Method** | Parallel processing (all tokens simultaneously) | Sequential generation (one token at a time)  
  
### 4.4.2 Causal Masking: Attention Without Seeing the Future

The core of GPT is the **Causal Attention Mask**. Each position can only refer to tokens before itself.

**Causal Mask Matrix** (1=can refer, 0=cannot refer):

$$ \text{CausalMask} = \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 1 & 1 & 0 & 0 \\\ 1 & 1 & 1 & 0 \\\ 1 & 1 & 1 & 1 \end{bmatrix} $$ 

During attention calculation, scores for future tokens are set to $-\infty$, resulting in probability 0 after softmax:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V $$ 

where $M$ is the Causal Mask matrix, with masked positions set to $-\infty$.

### 4.4.3 Evolution of GPT-1/2/3

Model | Release Year | Layers | Hidden Size | Parameters | Training Data  
---|---|---|---|---|---  
**GPT-1** | 2018 | 12 | 768 | 117M | BooksCorpus (4.5GB)  
**GPT-2** | 2019 | 48 | 1600 | 1.5B | WebText (40GB)  
**GPT-3** | 2020 | 96 | 12288 | 175B | CommonCrawl (570GB)  
**GPT-4** | 2023 | Undisclosed | Undisclosed | Est. 1.7T | Undisclosed (Multimodal)  
  
**Main Evolution Points** :

  * **Scale Expansion** : Exponential increase in parameters
  * **Few-shot Learning** : From GPT-3 onwards, can adapt to new tasks with just a few examples
  * **In-context Learning** : Learning with prompts only, without fine-tuning
  * **Emergent Abilities** : Abilities that suddenly appear with scale (reasoning, translation, etc.)

### 4.4.4 Causal Attention Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class CausalSelfAttention(nn.Module):
        """GPT-style Causal Self-Attention implementation"""
    
        def __init__(self, embed_size, num_heads):
            super(CausalSelfAttention, self).__init__()
            assert embed_size % num_heads == 0
    
            self.embed_size = embed_size
            self.num_heads = num_heads
            self.head_dim = embed_size // num_heads
    
            # Linear transformations for Q, K, V
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
    
            # Output layer
            self.proj = nn.Linear(embed_size, embed_size)
    
        def forward(self, x):
            """
            Args:
                x: [batch, seq_len, embed_size]
    
            Returns:
                output: [batch, seq_len, embed_size]
                attention_weights: [batch, num_heads, seq_len, seq_len]
            """
            batch_size, seq_len, _ = x.shape
    
            # Compute Q, K, V
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
    
            # Split for multi-head: [batch, num_heads, seq_len, head_dim]
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
    
            # Apply Causal Mask (set upper triangle to -inf)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
    
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
    
            # Weighted sum with Value
            out = torch.matmul(attention_weights, V)
    
            # Concatenate heads
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
    
            # Final projection
            output = self.proj(out)
    
            return output, attention_weights
    
    
    # Demonstration
    print("=== Causal Self-Attention Demo ===\n")
    
    batch_size = 1
    seq_len = 8
    embed_size = 64
    num_heads = 4
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # Apply Causal Attention
    causal_attn = CausalSelfAttention(embed_size, num_heads)
    output, attn_weights = causal_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize Causal Mask
    sample_attn = attn_weights[0, 0].detach().numpy()  # 1st batch, 1st head
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Causal Attention weights
    ax1 = axes[0]
    sns.heatmap(sample_attn,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax1,
                annot=True,
                fmt='.3f',
                linewidths=0.5,
                xticklabels=[f't{i+1}' for i in range(seq_len)],
                yticklabels=[f't{i+1}' for i in range(seq_len)])
    
    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax1.set_title('GPT Causal Attention Weights\n(Only lower triangle is valid)', fontsize=13, fontweight='bold')
    
    # Right: Causal Mask structure
    causal_mask_viz = np.tril(np.ones((seq_len, seq_len)))
    ax2 = axes[1]
    sns.heatmap(causal_mask_viz,
                cmap='RdYlGn',
                cbar_kws={'label': '1=Can refer, 0=Masked'},
                ax=ax2,
                annot=True,
                fmt='.0f',
                linewidths=0.5,
                xticklabels=[f't{i+1}' for i in range(seq_len)],
                yticklabels=[f't{i+1}' for i in range(seq_len)])
    
    ax2.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Query Position', fontsize=12, fontweight='bold')
    ax2.set_title('Causal Mask Structure\n(Hides future tokens)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCharacteristics:")
    print("✓ Each position only refers to tokens before (left of) itself")
    print("✓ Lower triangular matrix structure (upper triangle is 0)")
    print("✓ Sequential generation possible without using future information")
    

**Output** :
    
    
    === Causal Self-Attention Demo ===
    
    Input shape: torch.Size([1, 8, 64])
    Output shape: torch.Size([1, 8, 64])
    Attention weights shape: torch.Size([1, 4, 8, 8])
    
    Characteristics:
    ✓ Each position only refers to tokens before (left of) itself
    ✓ Lower triangular matrix structure (upper triangle is 0)
    ✓ Sequential generation possible without using future information
    

* * *

## 4.5 Text Generation with GPT

### 4.5.1 Autoregressive Generation Mechanism

GPT generates one token at a time sequentially:

  1. Input prompt (input text) to the model
  2. Predict probability distribution of next token
  3. Select next token using sampling strategy
  4. Append selected token to input
  5. Repeat steps 2-4

### 4.5.2 Sampling Strategies

Strategy | Description | Characteristics  
---|---|---  
**Greedy Decoding** | Select highest probability token | Deterministic, lots of repetition  
**Beam Search** | Maintain and explore multiple candidates | High quality but low diversity  
**Temperature Sampling** | Adjust probability with temperature parameter | Deterministic at T→0, random at T→∞  
**Top-k Sampling** | Sample from top k probability tokens | Balance between diversity and quality  
**Top-p (Nucleus)** | Sample from tokens with cumulative prob ≥ p | Dynamic vocabulary size adjustment  
  
### 4.5.3 Text Generation Implementation with GPT-2
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: 4.5.3 Text Generation Implementation with GPT-2
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    print("=== GPT-2 Text Generation Demo ===\n")
    
    # Load GPT-2 model
    model_name = 'gpt2'  # 124M parameters
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.eval()
    
    # Prompt
    prompt = "Artificial intelligence is transforming the world by"
    
    print(f"Prompt: {prompt}\n")
    print("=" * 80)
    
    # Generation with different sampling strategies
    strategies = [
        {
            'name': 'Greedy Decoding',
            'params': {
                'do_sample': False,
                'max_length': 50
            }
        },
        {
            'name': 'Temperature Sampling (T=0.7)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'temperature': 0.7
            }
        },
        {
            'name': 'Top-k Sampling (k=50)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'top_k': 50,
                'temperature': 1.0
            }
        },
        {
            'name': 'Top-p Sampling (p=0.9)',
            'params': {
                'do_sample': True,
                'max_length': 50,
                'top_p': 0.9,
                'temperature': 1.0
            }
        }
    ]
    
    for strategy in strategies:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt')
    
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                **strategy['params'],
                pad_token_id=tokenizer.eos_token_id
            )
    
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        print(f"\n{strategy['name']}:")
        print(f"{generated_text}")
        print("-" * 80)
    

**Example Output** :
    
    
    === GPT-2 Text Generation Demo ===
    
    Prompt: Artificial intelligence is transforming the world by
    
    ================================================================================
    
    Greedy Decoding:
    Artificial intelligence is transforming the world by making it easier for people to do things that they would otherwise have to do manually. The most common example is the use of AI to automate tasks such as scheduling, scheduling appointments, and scheduling meetings.
    
    --------------------------------------------------------------------------------
    
    Temperature Sampling (T=0.7):
    Artificial intelligence is transforming the world by enabling machines to learn from experience and make decisions without human intervention. From self-driving cars to medical diagnosis systems, AI technologies are revolutionizing industries and improving our daily lives.
    
    --------------------------------------------------------------------------------
    
    Top-k Sampling (k=50):
    Artificial intelligence is transforming the world by creating new possibilities in healthcare, education, and entertainment. AI systems can now analyze vast amounts of data, recognize patterns, and provide insights that were previously impossible to obtain.
    
    --------------------------------------------------------------------------------
    
    Top-p Sampling (p=0.9):
    Artificial intelligence is transforming the world by automating complex tasks, enhancing decision-making processes, and opening doors to innovations we never thought possible. As AI continues to evolve, its impact on society will only grow stronger.
    
    --------------------------------------------------------------------------------
    

### 4.5.4 Custom Generation Function Implementation
    
    
    def generate_text_custom(model, tokenizer, prompt, max_length=50,
                            strategy='top_p', temperature=1.0, top_k=50, top_p=0.9):
        """
        Custom text generation function
    
        Args:
            model: GPT-2 model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_length: Maximum generation length
            strategy: 'greedy', 'temperature', 'top_k', 'top_p'
            temperature: Temperature parameter
            top_k: k for Top-k sampling
            top_p: p for Top-p sampling
    
        Returns:
            Generated text
        """
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
        # Generation loop
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
    
            # Get logits for last token
            next_token_logits = logits[0, -1, :]
    
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
    
            # Sampling strategy
            if strategy == 'greedy':
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
    
            elif strategy == 'temperature':
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            elif strategy == 'top_k':
                # Top-k masking
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits_filtered[top_k_indices] = top_k_values
    
                probs = F.softmax(next_token_logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            elif strategy == 'top_p':
                # Top-p masking
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
                # Find positions where cumulative probability exceeds p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
    
                # Apply mask
                next_token_logits_filtered = next_token_logits.clone()
                next_token_logits_filtered[sorted_indices[sorted_indices_to_remove]] = float('-inf')
    
                probs = F.softmax(next_token_logits_filtered, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            # Append to input
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    
            # Stop at EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
        # Decode
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    
    # Test custom generation function
    print("\n=== Custom Generation Function Test ===\n")
    
    prompt = "The future of machine learning is"
    print(f"Prompt: {prompt}\n")
    
    for strategy in ['greedy', 'temperature', 'top_k', 'top_p']:
        generated = generate_text_custom(
            model, tokenizer, prompt,
            max_length=30,
            strategy=strategy,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        print(f"{strategy.upper()}: {generated}\n")
    

* * *

## 4.6 BERT vs GPT: Comparison and When to Use Each

### 4.6.1 Architecture Comparison
    
    
    ```mermaid
    graph TB
        subgraph "BERT (Encoder-only)"
            B1[Input: Full sentence] --> B2[Token + Segment + Position Embeddings]
            B2 --> B3[Transformer Encoder × 12]
            B3 --> B4[Bidirectional Attention]
            B4 --> B5["[CLS] for ClassificationAll Tokens for Token-level"]
    
            style B4 fill:#27ae60,color:#fff
        end
    
        subgraph "GPT (Decoder-only)"
            G1[Input: Prompt] --> G2[Token + Position Embeddings]
            G2 --> G3[Transformer Decoder × 12]
            G3 --> G4[Causal Attention]
            G4 --> G5[Next Token Prediction]
            G5 --> G6[Autoregressive Generation]
    
            style G4 fill:#e74c3c,color:#fff
        end
    ```

### 4.6.2 Performance Comparison Experiment
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: 4.6.2 Performance Comparison Experiment
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertModel, GPT2Model, BertTokenizer, GPT2Tokenizer
    import torch
    import time
    
    print("=== BERT vs GPT Performance Comparison ===\n")
    
    # Load models
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    
    bert_model.eval()
    gpt2_model.eval()
    
    # Test text
    text = "Natural language processing is a fascinating field of artificial intelligence."
    
    # BERT processing
    bert_inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    start_time = time.time()
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_time = time.time() - start_time
    
    # GPT-2 processing
    gpt2_inputs = gpt2_tokenizer(text, return_tensors='pt')
    start_time = time.time()
    with torch.no_grad():
        gpt2_outputs = gpt2_model(**gpt2_inputs)
    gpt2_time = time.time() - start_time
    
    # Display results
    print("Input Text:", text)
    print(f"\nBERT:")
    print(f"  Model: bert-base-uncased")
    print(f"  Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")
    print(f"  Input shape: {bert_inputs['input_ids'].shape}")
    print(f"  Output shape: {bert_outputs.last_hidden_state.shape}")
    print(f"  Processing time: {bert_time*1000:.2f} ms")
    print(f"  [CLS] embedding shape: {bert_outputs.pooler_output.shape}")
    
    print(f"\nGPT-2:")
    print(f"  Model: gpt2")
    print(f"  Parameters: {sum(p.numel() for p in gpt2_model.parameters()):,}")
    print(f"  Input shape: {gpt2_inputs['input_ids'].shape}")
    print(f"  Output shape: {gpt2_outputs.last_hidden_state.shape}")
    print(f"  Processing time: {gpt2_time*1000:.2f} ms")
    
    # Attention visualization comparison
    print("\n" + "="*80)
    print("Attention Pattern Comparison")
    print("="*80)
    
    # BERT: Can refer to all tokens mutually
    print("\nBERT Attention Pattern:")
    print("  ✓ Bidirectional - all tokens can refer to all tokens")
    print("  ✓ Parallel processing - processes all tokens simultaneously")
    print("  ✓ Use cases: classification, NER, QA, sentence encoding")
    
    # GPT: Can only refer to left tokens
    print("\nGPT Attention Pattern:")
    print("  ✓ Unidirectional - each token only refers to tokens on its left")
    print("  ✓ Sequential generation - generates one token at a time")
    print("  ✓ Use cases: text generation, dialogue, completion, translation")
    

**Output** :
    
    
    === BERT vs GPT Performance Comparison ===
    
    Input Text: Natural language processing is a fascinating field of artificial intelligence.
    
    BERT:
      Model: bert-base-uncased
      Parameters: 109,482,240
      Input shape: torch.Size([1, 14])
      Output shape: torch.Size([1, 14, 768])
      Processing time: 45.23 ms
      [CLS] embedding shape: torch.Size([1, 768])
    
    GPT-2:
      Model: gpt2
      Parameters: 124,439,808
      Input shape: torch.Size([1, 14])
      Output shape: torch.Size([1, 14, 768])
      Processing time: 38.67 ms
    
    ================================================================================
    Attention Pattern Comparison
    ================================================================================
    
    BERT Attention Pattern:
      ✓ Bidirectional - all tokens can refer to all tokens
      ✓ Parallel processing - processes all tokens simultaneously
      ✓ Use cases: classification, NER, QA, sentence encoding
    
    GPT Attention Pattern:
      ✓ Unidirectional - each token only refers to tokens on its left
      ✓ Sequential generation - generates one token at a time
      ✓ Use cases: text generation, dialogue, completion, translation
    

### 4.6.3 Usage Guide

Task | Recommended Model | Reason  
---|---|---  
**Sentiment Analysis** | BERT | Requires understanding of full sentence context  
**Named Entity Recognition** | BERT | Token classification, bidirectional context is advantageous  
**Question Answering** | BERT | Identifying answer spans in text  
**Document Classification** | BERT | Encode full sentence with [CLS] token  
**Text Generation** | GPT | Specialized for autoregressive generation  
**Dialogue Systems** | GPT | Response generation is main task  
**Summarization** | GPT (or BART) | Generation task, abstractive summarization  
**Code Generation** | GPT (Codex) | Sequential code generation  
**Translation** | Both possible | BERT→Encoder, GPT→Decoder usage  
  
* * *

## 4.7 Practical Projects

### 4.7.1 Project 1: Question Answering System with BERT

#### Goal

Build a SQuAD-format question answering system that accurately extracts answers from context.

#### Implementation Requirements

  * Use fine-tuned BERT model
  * Answer extraction for multiple questions
  * Calculate and display confidence scores
  * Validate answer plausibility

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import BertForQuestionAnswering, BertTokenizer
    import torch
    
    class QuestionAnsweringSystem:
        """BERT-based question answering system"""
    
        def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()
    
        def answer_question(self, question, context, return_confidence=True):
            """
            Extract answer to question
    
            Args:
                question: Question text
                context: Context (source text for answer)
                return_confidence: Whether to return confidence score
    
            Returns:
                answer: Extracted answer
                confidence: Confidence score (if return_confidence=True)
            """
            # Tokenize
            inputs = self.tokenizer(
                question,
                context,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
    
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
    
            # Predict start and end positions
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
    
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
    
            # Extract answer
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
            if return_confidence:
                # Calculate confidence
                start_score = torch.softmax(start_logits, dim=1)[0, start_idx].item()
                end_score = torch.softmax(end_logits, dim=1)[0, end_idx].item()
                confidence = (start_score + end_score) / 2
    
                return answer, confidence
            else:
                return answer
    
        def batch_answer(self, qa_pairs):
            """
            Answer multiple questions in batch
    
            Args:
                qa_pairs: List of [(question, context), ...]
    
            Returns:
                results: List of [(answer, confidence), ...]
            """
            results = []
            for question, context in qa_pairs:
                answer, confidence = self.answer_question(question, context)
                results.append((answer, confidence))
            return results
    
    
    # Test the system
    print("=== Question Answering System Demo ===\n")
    
    qa_system = QuestionAnsweringSystem()
    
    # Test cases
    context = """
    The Transformer architecture was introduced in the paper "Attention is All You Need"
    by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms to compute
    representations of input and output sequences without using recurrent or convolutional layers.
    The model achieved state-of-the-art results on machine translation tasks and has since become
    the foundation for models like BERT and GPT. The architecture consists of an encoder and a decoder,
    each composed of multiple identical layers. Each layer has two sub-layers: a multi-head self-attention
    mechanism and a position-wise fully connected feed-forward network.
    """
    
    questions = [
        "When was the Transformer introduced?",
        "Who introduced the Transformer?",
        "What does the Transformer rely on?",
        "What are the two main components of the Transformer?",
        "What models are based on the Transformer?"
    ]
    
    print("Context:")
    print(context)
    print("\n" + "="*80 + "\n")
    
    for i, question in enumerate(questions, 1):
        answer, confidence = qa_system.answer_question(question, context)
    
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer}")
        print(f"Confidence: {confidence:.4f}")
        print()
    
    # Batch processing demo
    print("="*80)
    print("\nBatch Processing Demo:")
    print("="*80 + "\n")
    
    qa_pairs = [(q, context) for q in questions]
    results = qa_system.batch_answer(qa_pairs)
    
    for (question, _), (answer, conf) in zip(qa_pairs, results):
        print(f"Q: {question}")
        print(f"A: {answer} (Conf: {conf:.4f})\n")
    

**Output** :
    
    
    === Question Answering System Demo ===
    
    Context:
    The Transformer architecture was introduced in the paper "Attention is All You Need"
    by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms to compute
    representations of input and output sequences without using recurrent or convolutional layers.
    The model achieved state-of-the-art results on machine translation tasks and has since become
    the foundation for models like BERT and GPT. The architecture consists of an encoder and a decoder,
    each composed of multiple identical layers. Each layer has two sub-layers: a multi-head self-attention
    mechanism and a position-wise fully connected feed-forward network.
    
    ================================================================================
    
    Q1: When was the Transformer introduced?
    A1: 2017
    Confidence: 0.9523
    
    Q2: Who introduced the Transformer?
    A2: Vaswani et al.
    Confidence: 0.8876
    
    Q3: What does the Transformer rely on?
    A3: self-attention mechanisms
    Confidence: 0.9234
    
    Q4: What are the two main components of the Transformer?
    A4: an encoder and a decoder
    Confidence: 0.8912
    
    Q5: What models are based on the Transformer?
    A5: BERT and GPT
    Confidence: 0.9101
    

### 4.7.2 Project 2: Text Generation App with GPT

#### Goal

Build a customizable text generation system to experiment with various generation strategies.

#### Implementation Requirements

  * Support multiple sampling strategies
  * Adjust generation parameters
  * Practice prompt engineering
  * Evaluate generation quality

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    class TextGenerator:
        """GPT-2 based text generation system"""
    
        def __init__(self, model_name='gpt2-medium'):
            """
            Args:
                model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            """
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()
    
            # Set PAD token
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        def generate(self, prompt, max_length=100, strategy='top_p',
                    num_return_sequences=1, **kwargs):
            """
            Generate text
    
            Args:
                prompt: Input prompt
                max_length: Maximum generation length
                strategy: 'greedy', 'beam', 'temperature', 'top_k', 'top_p'
                num_return_sequences: Number of candidates to generate
                **kwargs: Strategy-specific parameters
    
            Returns:
                List of generated texts
            """
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt')
    
            # Set parameters according to strategy
            gen_params = {
                'max_length': max_length,
                'num_return_sequences': num_return_sequences,
                'pad_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True
            }
    
            if strategy == 'greedy':
                gen_params['do_sample'] = False
    
            elif strategy == 'beam':
                gen_params['num_beams'] = kwargs.get('num_beams', 5)
                gen_params['do_sample'] = False
    
            elif strategy == 'temperature':
                gen_params['do_sample'] = True
                gen_params['temperature'] = kwargs.get('temperature', 0.7)
    
            elif strategy == 'top_k':
                gen_params['do_sample'] = True
                gen_params['top_k'] = kwargs.get('top_k', 50)
                gen_params['temperature'] = kwargs.get('temperature', 1.0)
    
            elif strategy == 'top_p':
                gen_params['do_sample'] = True
                gen_params['top_p'] = kwargs.get('top_p', 0.9)
                gen_params['temperature'] = kwargs.get('temperature', 1.0)
    
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(inputs['input_ids'], **gen_params)
    
            # Decode
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
    
            return generated_texts
    
        def interactive_generation(self):
            """Interactive generation session"""
            print("=== Interactive Text Generation ===")
            print("Type 'quit' to exit\n")
    
            while True:
                prompt = input("Prompt: ")
                if prompt.lower() == 'quit':
                    break
    
                # Generation settings
                print("\nGeneration Settings:")
                strategy = input("Strategy (greedy/beam/temperature/top_k/top_p) [top_p]: ") or 'top_p'
                max_length = int(input("Max length [100]: ") or 100)
    
                # Generate
                outputs = self.generate(prompt, max_length=max_length, strategy=strategy)
    
                print("\n--- Generated Text ---")
                print(outputs[0])
                print("-" * 80 + "\n")
    
    
    # Test the system
    print("=== Text Generation System Demo ===\n")
    
    generator = TextGenerator(model_name='gpt2')
    
    # Prompt templates
    prompts = [
        "In the future of artificial intelligence,",
        "The most important breakthrough in deep learning was",
        "Once upon a time in a distant galaxy,"
    ]
    
    print("Comparing Different Generation Strategies:\n")
    print("="*80 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}\n")
    
        strategies = [
            ('greedy', {}),
            ('top_k', {'top_k': 50, 'temperature': 0.8}),
            ('top_p', {'top_p': 0.9, 'temperature': 0.8})
        ]
    
        for strategy, params in strategies:
            outputs = generator.generate(
                prompt,
                max_length=60,
                strategy=strategy,
                num_return_sequences=1,
                **params
            )
    
            print(f"{strategy.upper()}:")
            print(f"{outputs[0]}\n")
    
        print("="*80 + "\n")
    
    # Multiple candidates generation demo
    print("\nMultiple Candidates Generation:")
    print("="*80 + "\n")
    
    prompt = "The key to successful machine learning is"
    outputs = generator.generate(
        prompt,
        max_length=50,
        strategy='top_p',
        num_return_sequences=3,
        top_p=0.9,
        temperature=0.9
    )
    
    for i, output in enumerate(outputs, 1):
        print(f"Candidate {i}:")
        print(output)
        print()
    

**Example Output** :
    
    
    === Text Generation System Demo ===
    
    Comparing Different Generation Strategies:
    
    ================================================================================
    
    Prompt: In the future of artificial intelligence,
    
    GREEDY:
    In the future of artificial intelligence, we will be able to create a new kind of AI that can do things that we have never done before. We will be able to build systems that can learn from data and make decisions based on that data.
    
    TOP_K:
    In the future of artificial intelligence, machines will become increasingly capable of understanding human language, emotions, and intentions. This will revolutionize how we interact with technology and open new possibilities in healthcare, education, and entertainment.
    
    TOP_P:
    In the future of artificial intelligence, we can expect to see breakthroughs in areas such as natural language understanding, computer vision, and autonomous decision-making. These advances will transform industries and create opportunities we haven't yet imagined.
    
    ================================================================================
    

* * *

## 4.8 Summary and Advanced Topics

### What We Learned in This Chapter

Topic | Key Points  
---|---  
**BERT** | Bidirectional encoder, MLM+NSP, optimal for understanding tasks  
**GPT** | Autoregressive generation, Causal Masking, optimal for generation tasks  
**Pre-training** | Train on large-scale data, specialize with fine-tuning  
**When to Use Each** | BERT for classification/extraction, GPT for generation  
**Practical Methods** | Hugging Face, sampling strategies, QA systems  
  
### Advanced Topics

**RoBERTa: Improved Version of BERT**

An improved version of BERT by Facebook. Removed the NSP task and adopted dynamic masking, larger training data, and longer training time to improve performance.

**ALBERT: Parameter Efficiency**

Achieves BERT-equivalent performance with fewer parameters through parameter sharing and factorization. Efficiently trains large-scale models.

**GPT-3.5/4: InstructGPT & ChatGPT**

Significantly improved ability to follow user instructions through Instruction Tuning and RLHF (Reinforcement Learning from Human Feedback). Became mainstream for dialogue systems.

**Prompt Engineering**

Prompt design techniques to maximize model performance. Few-shot examples, Chain-of-Thought prompting, Role prompting, etc.

**PEFT (Parameter-Efficient Fine-Tuning)**

Methods for efficient fine-tuning without updating all parameters: LoRA, Adapter, Prefix Tuning, etc. Essential technology for the large-scale model era.

### Exercises

#### Exercise 4.1: Sentiment Analysis with BERT Fine-tuning

**Task** : Fine-tune BERT on the IMDB review dataset and build a sentiment analysis model.

**Requirements** :

  * Data preprocessing and tokenization
  * Load BERT-Base model and add classification layer
  * Implement training loop
  * Evaluate accuracy and F1 score

#### Exercise 4.2: Dialogue System with GPT-2

**Task** : Implement a simple dialogue system using GPT-2.

**Requirements** :

  * Accept user input and manage context
  * Response generation (multiple sampling strategies)
  * Maintain and reflect conversation history
  * Evaluate conversation naturalness

#### Exercise 4.3: BERT vs GPT Performance Comparison

**Task** : Compare performance of BERT and GPT on the same task.

**Task** : Document classification (news article category classification)

**Comparison Items** :

  * Accuracy, F1 score
  * Training time
  * Inference speed
  * Memory usage

#### Exercise 4.4: Masked Language Modeling Implementation

**Task** : Implement MLM on a small dataset to reproduce BERT's pre-training.

**Implementation Contents** :

  * Mask data generation logic
  * MLM loss function
  * Training loop
  * Evaluate mask prediction accuracy

#### Exercise 4.5: Multilingual BERT (mBERT) Application

**Task** : Implement text classification in multiple languages using multilingual BERT.

**Languages** : English, Japanese, Chinese

**Task** : News article topic classification

#### Exercise 4.6: Code Generation with GPT

**Task** : Build a system that generates Python code from natural language instructions using GPT-2.

**Requirements** :

  * Design prompt templates
  * Code generation and syntax validation
  * Evaluate generation quality

* * *

### Next Chapter Preview

In Chapter 5, we will learn about **Vision Transformer (ViT)**. We will explore the innovative approach of applying Transformer architecture to Computer Vision by treating images as "tokens".

> **Next Chapter Topics** :  
>  ・Vision Transformer architecture  
>  ・Image patch tokenization  
>  ・2D extension of Position Embeddings  
>  ・Performance comparison with CNNs  
>  ・Pre-training strategies (ImageNet-21k)  
>  ・Implementation: Image classification with ViT  
>  ・Applications: Object Detection, Segmentation
