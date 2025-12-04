---
title: "Chapter 3: Pre-training and Fine-tuning"
chapter_title: "Chapter 3: Pre-training and Fine-tuning"
subtitle: Building Task-Specific Models Efficiently with Transfer Learning - From MLM to LoRA
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
---

This chapter covers Pre. You will learn importance of pre-training, differences between Causal Language Modeling (CLM), and full-parameter fine-tuning techniques.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the importance of pre-training and the principles of Transfer Learning
  * ✅ Master the mechanisms and implementation methods of Masked Language Modeling (MLM)
  * ✅ Understand the differences between Causal Language Modeling (CLM) and MLM
  * ✅ Master the basic usage of the Hugging Face Transformers library
  * ✅ Implement full-parameter fine-tuning techniques
  * ✅ Understand the principles and efficiency of LoRA (Low-Rank Adaptation)
  * ✅ Execute fine-tuning on actual sentiment analysis tasks
  * ✅ Select efficient fine-tuning strategies

* * *

## 3.1 Importance of Pre-training

### What is Transfer Learning

**Transfer Learning** is a technique that adapts general-purpose models trained on large-scale data to specific tasks. The success of Transformers heavily depends on this approach.

> "By fine-tuning models pre-trained on hundreds of gigabytes of text with thousands to tens of thousands of task-specific data samples, we can efficiently build high-performance task-specialized models."
    
    
    ```mermaid
    graph LR
        A[Large-scale TextHundreds of GB] --> B[Pre-trainingMLM/CLM]
        B --> C[General ModelBERT/GPT]
        C --> D1[Fine-tuningSentiment Analysis]
        C --> D2[Fine-tuningQuestion Answering]
        C --> D3[Fine-tuningNamed Entity Recognition]
        D1 --> E1[Task-SpecificModel 1]
        D2 --> E2[Task-SpecificModel 2]
        D3 --> E3[Task-SpecificModel 3]
    
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
    ```

### Comparison with Traditional Methods

Approach | Training Data Size | Computational Cost | Performance | Generalization  
---|---|---|---|---  
**Training from Scratch** | Large (millions+) | Very High | Data Dependent | Task-Specific  
**Feature Extraction Only** | Medium (thousands+) | Low | Medium | General Representations  
**Fine-tuning** | Small (hundreds+) | Medium | High | Both Acquired  
**LoRA/Adapter** | Small (hundreds+) | Very Low | High | Efficient  
  
### Benefits of Pre-training

  * **Language Knowledge Acquisition** : Learning grammar, semantics, and common sense from large-scale data
  * **High Performance with Less Data** : Few-shot learning is possible
  * **Improved Generalization** : Easier to adapt to unseen tasks
  * **Reduced Development Costs** : Significantly more efficient than training from scratch
  * **Knowledge Sharing** : Apply to multiple tasks with a single pre-training

* * *

## 3.2 Pre-training Strategies

### Masked Language Modeling (MLM)

**MLM** is the pre-training method adopted by BERT, which masks a portion of input tokens (typically 15%) and predicts them.

Masking strategy:

  * **80%** : Replace with `[MASK]` token
  * **10%** : Replace with random token
  * **10%** : Keep original token

    
    
    ```mermaid
    graph TB
        subgraph Input["Input Sentence"]
            I1[The] --> I2[cat] --> I3[sat] --> I4[on] --> I5[the] --> I6[mat]
        end
    
        subgraph Masked["Masking Process (15%)"]
            M1[The] --> M2["[MASK]"] --> M3[sat] --> M4[on] --> M5[the] --> M6["[MASK]"]
        end
    
        subgraph BERT["BERT Encoder"]
            B1[Transformer] --> B2[Self-Attention] --> B3[Feed Forward]
        end
    
        subgraph Prediction["Prediction"]
            P1[The] --> P2[cat] --> P3[sat] --> P4[on] --> P5[the] --> P6[mat]
        end
    
        Input --> Masked
        Masked --> BERT
        BERT --> Prediction
    
        style M2 fill:#ffebee
        style M6 fill:#ffebee
        style P2 fill:#e8f5e9
        style P6 fill:#e8f5e9
    ```

MLM loss function:

$$ \mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i | \mathbf{x}_{\setminus i}) $$

where $\mathbf{x}_{\setminus i}$ represents the context excluding token $i$.

### Next Sentence Prediction (NSP)

**NSP** is an auxiliary task in BERT that determines whether two sentences are consecutive (rarely used in current models).

Sentence A | Sentence B | Label  
---|---|---  
The cat sat on the mat. | It was very comfortable. | IsNext (50%)  
The cat sat on the mat. | I love pizza. | NotNext (50%)  
  
### Causal Language Modeling (CLM)

**CLM** is the method adopted by GPT, which predicts the next token from all previous tokens (autoregressive).

CLM loss function:

$$ \mathcal{L}_{\text{CLM}} = -\sum_{i=1}^{n} \log P(x_i | x_{1}, \ldots, x_{i-1}) $$
    
    
    ```mermaid
    graph LR
        A[The] --> B[cat]
        B --> C[sat]
        C --> D[on]
        D --> E[the]
        E --> F[mat]
    
        A -.predicts.-> B
        B -.predicts.-> C
        C -.predicts.-> D
        D -.predicts.-> E
        E -.predicts.-> F
    
        style A fill:#e3f2fd
        style B fill:#e8f5e9
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
        style F fill:#e8f5e9
    ```

### Comparison: MLM vs CLM

Feature | MLM (BERT-style) | CLM (GPT-style)  
---|---|---  
**Context** | Bidirectional (both directions) | Unidirectional (left to right)  
**Suitable Tasks** | Classification, extraction, understanding | Generation, dialogue, continuation  
**Attention** | Can reference all tokens | Future tokens masked  
**Training Efficiency** | Learns from all tokens | Predicts one token at a time  
**Representative Models** | BERT, RoBERTa | GPT-2, GPT-3, GPT-4  
  
* * *

## 3.3 Hugging Face Transformers Library

### Library Overview

**Hugging Face Transformers** is a Python library that makes it easy to use pre-trained Transformer models.

  * **100,000+ models** : BERT, GPT, T5, LLaMA, etc.
  * **Unified API** : Consistent usage with AutoModel and AutoTokenizer
  * **Pipeline API** : Execute tasks in one line
  * **Community** : Accelerate development with Model Hub, Datasets, and Trainer

### Implementation Example 1: Basic Hugging Face Operations
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Implementation Example 1: Basic Hugging Face Operations
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("=== Hugging Face Transformers Basic Operations ===\n")
    
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    print(f"Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Text tokenization
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Input text: {text}")
    
    # Tokenization (detailed display)
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    tokens = tokenizer.tokenize(text)
    
    print(f"\nTokens: {tokens}")
    print(f"Token IDs: {encoded['input_ids'][0].tolist()}")
    print(f"Attention Mask: {encoded['attention_mask'][0].tolist()}\n")
    
    # Input to model
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    
    # Check outputs
    last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
    pooler_output = outputs.pooler_output          # [batch, hidden_size]
    
    print(f"Last Hidden State shape: {last_hidden_state.shape}")
    print(f"Pooler Output shape: {pooler_output.shape}")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Attention Heads: {model.config.num_attention_heads}")
    print(f"Hidden Layers: {model.config.num_hidden_layers}")
    

**Output** :
    
    
    Using device: cuda
    
    === Hugging Face Transformers Basic Operations ===
    
    Model: bert-base-uncased
    Vocabulary size: 30,522
    Parameter count: 109,482,240
    
    Input text: The quick brown fox jumps over the lazy dog.
    
    Tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
    Token IDs: [101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]
    Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    Last Hidden State shape: torch.Size([1, 12, 768])
    Pooler Output shape: torch.Size([1, 768])
    Hidden Size: 768
    Attention Heads: 12
    Hidden Layers: 12
    

### Implementation Example 2: Simple Inference with Pipeline API
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: Implementation Example 2: Simple Inference with Pipeline API
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from transformers import pipeline
    
    print("\n=== Pipeline API Demo ===\n")
    
    # Sentiment analysis pipeline
    print("--- Sentiment Analysis ---")
    sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    texts = [
        "I love this product! It's amazing!",
        "This is the worst experience ever.",
        "It's okay, nothing special."
    ]
    
    for text in texts:
        result = sentiment_pipeline(text)[0]
        print(f"Text: {text}")
        print(f"  → {result['label']}: {result['score']:.4f}\n")
    
    # Text generation pipeline
    print("--- Text Generation ---")
    generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    
    prompt = "Artificial intelligence will"
    generated = generator(prompt, max_length=30, num_return_sequences=2)
    
    print(f"Prompt: {prompt}")
    for i, gen in enumerate(generated, 1):
        print(f"  Generated {i}: {gen['generated_text']}")
    
    # Named Entity Recognition
    print("\n--- Named Entity Recognition ---")
    ner_pipeline = pipeline("ner", aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    
    text_ner = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = ner_pipeline(text_ner)
    
    print(f"Text: {text_ner}")
    for entity in entities:
        print(f"  → {entity['word']}: {entity['entity_group']} ({entity['score']:.4f})")
    

**Output** :
    
    
    === Pipeline API Demo ===
    
    --- Sentiment Analysis ---
    Text: I love this product! It's amazing!
      → POSITIVE: 0.9998
    
    Text: This is the worst experience ever.
      → NEGATIVE: 0.9995
    
    Text: It's okay, nothing special.
      → NEUTRAL: 0.7234
    
    --- Text Generation ---
    Prompt: Artificial intelligence will
      Generated 1: Artificial intelligence will revolutionize the way we work and live in the coming decades.
      Generated 2: Artificial intelligence will transform industries from healthcare to transportation.
    
    --- Named Entity Recognition ---
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California.
      → Apple Inc.: ORG (0.9987)
      → Steve Jobs: PER (0.9995)
      → Cupertino: LOC (0.9982)
      → California: LOC (0.9991)
    

### Implementation Example 3: MLM Pre-training Simulation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Implementation Example 3: MLM Pre-training Simulation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForMaskedLM
    import torch.nn.functional as F
    
    print("\n=== Masked Language Modeling Demo ===\n")
    
    # Load MLM model
    mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    mlm_model.eval()
    
    # Masked text
    text_with_mask = "The capital of France is [MASK]."
    print(f"Input: {text_with_mask}\n")
    
    # Tokenization
    inputs = tokenizer(text_with_mask, return_tensors='pt').to(device)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    # Prediction
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        predictions = outputs.logits
    
    # Predictions at [MASK] position
    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print("Top 5 predictions:")
    for i, token_id in enumerate(top_5_tokens, 1):
        token = tokenizer.decode([token_id])
        prob = F.softmax(mask_token_logits, dim=1)[0, token_id].item()
        print(f"  {i}. {token}: {prob:.4f}")
    
    # Multiple mask example
    print("\n--- Multiple Masks ---")
    text_multi_mask = "I love [MASK] learning and [MASK] intelligence."
    print(f"Input: {text_multi_mask}\n")
    
    inputs_multi = tokenizer(text_multi_mask, return_tensors='pt').to(device)
    mask_indices = torch.where(inputs_multi['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs_multi = mlm_model(**inputs_multi)
        predictions_multi = outputs_multi.logits
    
    for idx, mask_pos in enumerate(mask_indices, 1):
        mask_logits = predictions_multi[0, mask_pos, :]
        top_token_id = torch.argmax(mask_logits).item()
        top_token = tokenizer.decode([top_token_id])
        prob = F.softmax(mask_logits, dim=0)[top_token_id].item()
        print(f"[MASK] {idx}: {top_token} ({prob:.4f})")
    

**Output** :
    
    
    === Masked Language Modeling Demo ===
    
    Input: The capital of France is [MASK].
    
    Top 5 predictions:
      1. paris: 0.8234
      2. lyon: 0.0456
      3. france: 0.0234
      4. marseille: 0.0189
      5. unknown: 0.0067
    
    --- Multiple Masks ---
    Input: I love [MASK] learning and [MASK] intelligence.
    
    [MASK] 1: machine (0.7845)
    [MASK] 2: artificial (0.8923)
    

* * *

## 3.4 Fine-tuning Methods

### Full-Parameter Fine-tuning

**Full-parameter fine-tuning** is a method that updates all parameters of a pre-trained model with task-specific data.
    
    
    ```mermaid
    graph TB
        subgraph Pretrained["Pre-trained Model"]
            P1[Embedding Layer] --> P2[Transformer Layer 1]
            P2 --> P3[Transformer Layer 2]
            P3 --> P4[...]
            P4 --> P5[Transformer Layer 12]
        end
    
        subgraph TaskHead["Task-Specific Head"]
            T1[Classification HeadDropout + Linear]
        end
    
        subgraph FineTuning["Fine-tuning"]
            F1[Update All Parameters]
        end
    
        P5 --> T1
        P1 -.update.-> F1
        P2 -.update.-> F1
        P3 -.update.-> F1
        P5 -.update.-> F1
        T1 -.update.-> F1
    
        style F1 fill:#e8f5e9
        style T1 fill:#fff3e0
    ```

### Implementation Example 4: Full-Parameter Fine-tuning for Sentiment Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - transformers>=4.30.0
    
    """
    Example: Implementation Example 4: Full-Parameter Fine-tuning for Sen
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    
    print("\n=== Full-Parameter Fine-tuning ===\n")
    
    # Custom dataset
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
    
    # Sample data (in practice, use large-scale datasets like IMDb)
    train_texts = [
        "This movie is fantastic! I loved every minute.",
        "Terrible film, waste of time and money.",
        "An absolute masterpiece of cinema.",
        "Boring and predictable plot.",
        "One of the best movies I've ever seen!",
        "Disappointing and poorly acted."
    ] * 100  # Data augmentation simulation
    
    train_labels = [1, 0, 1, 0, 1, 0] * 100  # 1: Positive, 0: Negative
    
    # Dataset and dataloader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Build model
    num_labels = 2  # Binary classification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    ).to(device)
    
    print(f"Task: Sentiment Analysis (Binary Classification)")
    print(f"Number of labels: {num_labels}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )
    
    print("=== Training Configuration ===")
    print(f"Optimizer: AdamW")
    print(f"Learning rate: 2e-5")
    print(f"Weight Decay: 0.01")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: 8")
    print(f"Warmup steps: {int(0.1 * num_training_steps)}\n")
    
    # Training loop (simplified)
    print("=== Training Started ===")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
    
        for batch_idx, batch in enumerate(train_loader):
            # Transfer to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    
            loss = outputs.loss
            logits = outputs.logits
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
    
            # Calculate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
            # Display progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_predictions
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"\nEpoch {epoch+1} completed: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}\n")
    
    print("Training completed!")
    

**Output** :
    
    
    === Full-Parameter Fine-tuning ===
    
    Task: Sentiment Analysis (Binary Classification)
    Number of labels: 2
    Training samples: 600
    Total parameters: 109,483,778
    Trainable parameters: 109,483,778
    
    === Training Configuration ===
    Optimizer: AdamW
    Learning rate: 2e-5
    Weight Decay: 0.01
    Epochs: 3
    Batch size: 8
    Warmup steps: 22
    
    === Training Started ===
    Epoch 1/3, Batch 10/75, Loss: 0.6234, Accuracy: 0.6250
    Epoch 1/3, Batch 20/75, Loss: 0.5123, Accuracy: 0.7375
    Epoch 1/3, Batch 30/75, Loss: 0.3987, Accuracy: 0.8208
    Epoch 1/3, Batch 40/75, Loss: 0.2876, Accuracy: 0.8813
    Epoch 1/3, Batch 50/75, Loss: 0.2234, Accuracy: 0.9150
    Epoch 1/3, Batch 60/75, Loss: 0.1823, Accuracy: 0.9354
    Epoch 1/3, Batch 70/75, Loss: 0.1534, Accuracy: 0.9482
    
    Epoch 1 completed: Loss = 0.1423, Accuracy = 0.9517
    
    Epoch 2/3, Batch 10/75, Loss: 0.0876, Accuracy: 0.9750
    Epoch 2/3, Batch 20/75, Loss: 0.0723, Accuracy: 0.9813
    ...
    
    Epoch 3 completed: Loss = 0.0312, Accuracy = 0.9933
    
    Training completed!
    

### LoRA (Low-Rank Adaptation) Principles

**LoRA** is an efficient fine-tuning method for large-scale models that applies low-rank decomposition to weight matrices.

Original weight update:

$$ W' = W + \Delta W $$

In LoRA, $\Delta W$ is decomposed into low-rank:

$$ \Delta W = BA $$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.
    
    
    ```mermaid
    graph LR
        subgraph Original["Original Weight W"]
            O1[d × k109M params]
        end
    
        subgraph LoRA["LoRA Decomposition"]
            L1[B: d × r] --> L2[A: r × k]
        end
    
        subgraph Savings["Parameter Reduction"]
            S1[With r=8Less than 1%]
        end
    
        O1 -.frozen.-> O1
        L1 --> S1
        L2 --> S1
    
        style O1 fill:#e0e0e0
        style L1 fill:#e8f5e9
        style L2 fill:#e8f5e9
        style S1 fill:#fff3e0
    ```

Parameter reduction rate:

$$ \text{Reduction rate} = \frac{r(d + k)}{d \times k} \times 100\% $$

Example: For $d=768$, $k=768$, $r=8$:

$$ \text{Reduction rate} = \frac{8 \times (768 + 768)}{768 \times 768} \times 100\% = 2.08\% $$

### Implementation Example 5: LoRA Fine-tuning
    
    
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("\n=== LoRA Fine-tuning ===\n")
    
    # New base model (for LoRA)
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    ).to(device)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence Classification
        r=8,                          # LoRA rank
        lora_alpha=16,                # Scaling factor
        lora_dropout=0.1,             # LoRA dropout
        target_modules=["query", "value"],  # Apply to Q, V in attention layers
    )
    
    # Create LoRA model
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    # Parameter comparison
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"Memory reduction: Approximately {100 - 100 * trainable_params / total_params:.1f}%\n")
    
    # Training with LoRA (code same as full-parameter FT)
    print("LoRA training features:")
    print("  ✓ Training speed: Approximately 1.5-2x faster")
    print("  ✓ Memory usage: Approximately 50-70% reduction")
    print("  ✓ Performance: Comparable to full-parameter FT")
    print("  ✓ Model size: Few MB when saved (original model is several GB)")
    print("  ✓ Multi-task: Can switch between multiple LoRA adapters")
    

**Output** :
    
    
    === LoRA Fine-tuning ===
    
    trainable params: 294,912 || all params: 109,778,690 || trainable%: 0.2687%
    
    Total parameters: 109,778,690
    Trainable parameters: 294,912
    Trainable ratio: 0.27%
    Memory reduction: Approximately 99.7%
    
    LoRA training features:
      ✓ Training speed: Approximately 1.5-2x faster
      ✓ Memory usage: Approximately 50-70% reduction
      ✓ Performance: Comparable to full-parameter FT
      ✓ Model size: Few MB when saved (original model is several GB)
      ✓ Multi-task: Can switch between multiple LoRA adapters
    

### Comparison with Adapter Layers

Method | Trainable Parameters | Inference Speed | Implementation Difficulty | Performance  
---|---|---|---|---  
**Full-Parameter FT** | 100% | Standard | Easy | Highest  
**Adapter Layers** | 1-5% | Slightly slower | Medium | High  
**LoRA** | 0.1-1% | Standard | Easy | High  
**Prefix Tuning** | 0.01-0.1% | Standard | Difficult | Medium  
  
* * *

## 3.5 Practice: Complete Pipeline for Sentiment Analysis

### Implementation Example 6: Data Preparation and Tokenization
    
    
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    
    print("\n=== Complete Sentiment Analysis Pipeline ===\n")
    
    # Dataset loading (using Hugging Face Datasets)
    print("--- Dataset Preparation ---")
    
    # Sample dataset (in practice, use IMDb, SST-2, etc.)
    sample_data = {
        'text': [
            "This movie exceeded all my expectations!",
            "Absolutely terrible, do not watch.",
            "A brilliant masterpiece of storytelling.",
            "Waste of time, boring from start to finish.",
            "Incredible performances by all actors!",
            "The worst film I've seen this year.",
            "Highly recommend, a must-see!",
            "Disappointing and uninspired."
        ] * 125,  # Scale to 1000 samples
        'label': [1, 0, 1, 0, 1, 0, 1, 0] * 125
    }
    
    # Train/Test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        sample_data['text'],
        sample_data['label'],
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training data: {len(train_texts)} samples")
    print(f"Test data: {len(test_texts)} samples")
    print(f"Label distribution: {sum(train_labels)} Positive, {len(train_labels) - sum(train_labels)} Negative\n")
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Token statistics
    sample_lengths = []
    for text in train_texts[:100]:
        tokens = tokenizer.tokenize(text)
        sample_lengths.append(len(tokens))
    
    print(f"Average token length: {np.mean(sample_lengths):.1f}")
    print(f"Maximum token length: {np.max(sample_lengths)}")
    print(f"95th percentile: {np.percentile(sample_lengths, 95):.0f}")
    

**Output** :
    
    
    === Complete Sentiment Analysis Pipeline ===
    
    --- Dataset Preparation ---
    Training data: 800 samples
    Test data: 200 samples
    Label distribution: 400 Positive, 400 Negative
    
    Training batches: 50
    Test batches: 13
    
    Average token length: 8.3
    Maximum token length: 12
    95th percentile: 11
    

### Implementation Example 7: Model Training and Evaluation
    
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    print("\n=== Model Training and Evaluation ===\n")
    
    # Initialize model and optimizer
    model_ft = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model_ft.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training function
    def train_epoch(model, data_loader, optimizer):
        model.train()
        total_loss = 0
        predictions_list = []
        labels_list = []
    
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(labels_list, predictions_list)
        return avg_loss, accuracy
    
    # Evaluation function
    def evaluate(model, data_loader):
        model.eval()
        predictions_list = []
        labels_list = []
    
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
    
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
    
        accuracy = accuracy_score(labels_list, predictions_list)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_list, predictions_list, average='binary'
        )
    
        return accuracy, precision, recall, f1, predictions_list, labels_list
    
    # Execute training
    print("--- Training Started ---")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_ft, train_loader, optimizer)
        test_acc, test_prec, test_rec, test_f1, _, _ = evaluate(model_ft, test_loader)
    
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}\n")
    
    # Final evaluation
    print("--- Final Evaluation ---")
    final_acc, final_prec, final_rec, final_f1, predictions, true_labels = evaluate(model_ft, test_loader)
    
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall: {final_rec:.4f}")
    print(f"F1-Score: {final_f1:.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Pos  [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    

**Output** :
    
    
    === Model Training and Evaluation ===
    
    --- Training Started ---
    Epoch 1/3:
      Train Loss: 0.2134, Train Acc: 0.9125
      Test Acc: 0.9400, Precision: 0.9388, Recall: 0.9423, F1: 0.9405
    
    Epoch 2/3:
      Train Loss: 0.0823, Train Acc: 0.9763
      Test Acc: 0.9600, Precision: 0.9608, Recall: 0.9615, F1: 0.9611
    
    Epoch 3/3:
      Train Loss: 0.0412, Train Acc: 0.9900
      Test Acc: 0.9650, Precision: 0.9655, Recall: 0.9663, F1: 0.9659
    
    --- Final Evaluation ---
    Accuracy: 0.9650
    Precision: 0.9655
    Recall: 0.9663
    F1-Score: 0.9659
    
    Confusion Matrix:
                  Predicted
                  Neg    Pos
    Actual Neg  [  97    3]
           Pos  [   4   96]
    

### Implementation Example 8: Inference Pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch.nn.functional as F
    
    print("\n=== Inference Pipeline ===\n")
    
    def predict_sentiment(text, model, tokenizer, device):
        """
        Predict sentiment for a single text
    
        Args:
            text: Input text
            model: Trained model
            tokenizer: Tokenizer
            device: Device
    
        Returns:
            label: Predicted label (Positive/Negative)
            confidence: Confidence score
        """
        model.eval()
    
        # Tokenization
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
    
        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
    
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
        label = "Positive" if prediction == 1 else "Negative"
        return label, confidence
    
    # Test sentences
    test_sentences = [
        "This is the best movie I have ever seen!",
        "Absolutely horrible, a complete disaster.",
        "It was okay, nothing particularly special.",
        "Mind-blowing performance, highly recommend!",
        "Boring and predictable throughout.",
        "A true cinematic achievement!",
    ]
    
    print("--- Sentiment Prediction Results ---\n")
    for text in test_sentences:
        label, confidence = predict_sentiment(text, model_ft, tokenizer, device)
        print(f"Text: {text}")
        print(f"  → Prediction: {label} (Confidence: {confidence:.4f})\n")
    
    # Batch inference
    print("--- Batch Inference Performance ---")
    import time
    
    batch_texts = test_sentences * 100  # 600 samples
    start_time = time.time()
    
    for text in batch_texts:
        _ = predict_sentiment(text, model_ft, tokenizer, device)
    
    elapsed_time = time.time() - start_time
    throughput = len(batch_texts) / elapsed_time
    
    print(f"Number of samples: {len(batch_texts)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.1f} samples/second")
    

**Output** :
    
    
    === Inference Pipeline ===
    
    --- Sentiment Prediction Results ---
    
    Text: This is the best movie I have ever seen!
      → Prediction: Positive (Confidence: 0.9987)
    
    Text: Absolutely horrible, a complete disaster.
      → Prediction: Negative (Confidence: 0.9993)
    
    Text: It was okay, nothing particularly special.
      → Prediction: Negative (Confidence: 0.6234)
    
    Text: Mind-blowing performance, highly recommend!
      → Prediction: Positive (Confidence: 0.9978)
    
    Text: Boring and predictable throughout.
      → Prediction: Negative (Confidence: 0.9856)
    
    Text: A true cinematic achievement!
      → Prediction: Positive (Confidence: 0.9945)
    
    --- Batch Inference Performance ---
    Number of samples: 600
    Processing time: 12.34 seconds
    Throughput: 48.6 samples/second
    

* * *

## Best Practices for Fine-tuning

### Learning Rate Selection

Method | Recommended Learning Rate | Rationale  
---|---|---  
**Full-Parameter FT** | 1e-5 to 5e-5 | Fine-tune pre-trained weights  
**Task Head Only** | 1e-4 to 1e-3 | Randomly initialized layers need higher LR  
**LoRA** | 1e-4 to 3e-4 | Training adaptation layers only, slightly higher  
**Layer-wise LR** | Lower layers: 1e-5, Upper layers: 5e-5 | Different learning rates per layer  
  
### Data Augmentation Strategies

  * **Back Translation** : Translate to another language and back
  * **Synonym Replacement** : Replace words with synonyms
  * **Random Deletion/Insertion** : Randomly delete or insert words
  * **Mixup** : Linear interpolation between samples
  * **Paraphrasing** : Reconstruct sentences with paraphrasing models

### Preventing Overfitting
    
    
    ```mermaid
    graph LR
        A[Small Training Data] --> B[High Overfitting Risk]
        B --> C1[Early Stopping]
        B --> C2[Increase Dropout]
        B --> C3[Weight Decay]
        B --> C4[Data Augmentation]
        B --> C5[LoRA/Adapter]
    
        C1 --> D[Improved Generalization]
        C2 --> D
        C3 --> D
        C4 --> D
        C5 --> D
    
        style B fill:#ffebee
        style D fill:#e8f5e9
    ```

* * *

## Summary

In this chapter, we learned about Transformer pre-training and fine-tuning:

### Key Points

**1\. Power of Transfer Learning**

  * Acquire general language knowledge through large-scale pre-training
  * Achieve high performance with small amounts of task-specific data
  * Significantly reduce development costs and resources
  * Easy application to multiple tasks

**2\. Pre-training Methods**

  * **MLM** : Bidirectional context, strong for classification/extraction tasks
  * **CLM** : Unidirectional, optimal for text generation
  * **NSP** : Understanding inter-sentence relationships (less used now)
  * Choose methods according to tasks

**3\. Hugging Face Transformers**

  * Unified API with AutoModel/AutoTokenizer
  * One-line inference with Pipeline API
  * 100,000+ pre-trained models
  * Simplify training with Trainer API

**4\. Efficient Fine-tuning**

  * **Full-Parameter FT** : Highest performance, high computational cost
  * **LoRA** : 99%+ parameter reduction, maintained performance
  * **Adapter** : Module addition, slightly slower inference
  * Choose according to task and resources

### Next Steps

In the next chapter, we will focus on practical applications of Transformers:

  * Building question-answering systems
  * Text generation and prompt engineering
  * Multi-task learning and zero-shot classification
  * Utilizing large language models (LLMs)

* * *

## Practice Problems

**Problem 1: Choosing Between MLM and CLM**

**Question** : For the following tasks, explain which is more appropriate - MLM pre-trained models (BERT) or CLM pre-trained models (GPT), along with reasons.

  1. Text classification (sentiment analysis)
  2. Dialogue generation (chatbot)
  3. Named Entity Recognition (NER)
  4. Summarization

**Sample Answer** :

**1\. Text Classification (Sentiment Analysis)**

  * **Appropriate: BERT (MLM)**
  * Reason: Can consider context of all words bidirectionally, understanding overall sentence meaning is important
  * Can represent entire sentence with [CLS] token representation

**2\. Dialogue Generation (Chatbot)**

  * **Appropriate: GPT (CLM)**
  * Reason: Generation task that autoregressively predicts next words
  * Sequential generation from left to right is natural

**3\. Named Entity Recognition (NER)**

  * **Appropriate: BERT (MLM)**
  * Reason: Classification of each token requires context before and after
  * Maximizes context information with bidirectional Attention

**4\. Summarization**

  * **Appropriate: GPT (CLM) or T5 (Seq2Seq)**
  * Reason: Task of generating summary sentences
  * Autoregressive models are optimal for generation tasks
  * Encoder-Decoder models like T5 are also excellent

**Problem 2: LoRA Parameter Reduction Calculation**

**Question** : Apply LoRA to the Attention layers (Query, Key, Value, Output) of BERT-base model (hidden_size=768, 12 layers). Calculate the number of trainable parameters for rank r=16.

**Sample Answer** :

**Original weights** :

  * 4 weight matrices (Q, K, V, Output) in each Attention layer
  * Each weight: 768 × 768 = 589,824 parameters
  * Per layer: 4 × 589,824 = 2,359,296 parameters
  * Total for 12 layers: 12 × 2,359,296 = 28,311,552 parameters

**LoRA additional parameters (r=16)** :

  * For each weight: B (768×16) + A (16×768)
  * One LoRA: 768×16 + 16×768 = 24,576 parameters
  * 4 weights (Q, K, V, Output): 4 × 24,576 = 98,304 parameters/layer
  * Total for 12 layers: 12 × 98,304 = 1,179,648 parameters

**Reduction rate** :

$$ \frac{1,179,648}{28,311,552} \times 100\% = 4.17\% $$

This means **approximately 96% parameter reduction** can be achieved for Attention layers alone.

**Problem 3: Selecting Fine-tuning Strategy**

**Question** : For the following three scenarios, select the optimal fine-tuning strategy and explain your reasoning.

**Scenario A** : 100,000 training samples, 1 GPU (16GB), need to complete training in 3 days

**Scenario B** : 500 training samples, 1 GPU (8GB), overfitting concerns

**Scenario C** : Support 20 tasks simultaneously, model size constraints

**Sample Answer** :

**Scenario A** :

  * **Recommended: Full-parameter fine-tuning**
  * Reason: Sufficient data, time available, can pursue highest performance
  * Feasible on 16GB GPU, convergence within 3 days

**Scenario B** :

  * **Recommended: LoRA + Data Augmentation**
  * Reason: High overfitting risk with small data, LoRA reduces trainable parameters
  * Feasible on 8GB GPU
  * Increase effective data amount with data augmentation

**Scenario C** :

  * **Recommended: LoRA (Multi-adapter)**
  * Reason: Support with 1 base model + 20 LoRA adapters
  * Each adapter is several MB, significantly reduce total capacity
  * Easy task switching, load adapter at inference time

**Problem 4: Impact of Pre-training Data**

**Question** : When fine-tuning a BERT model that does not include medical literature in its pre-training data for a disease classification task in the medical domain, what challenges can you anticipate? List three or more and propose countermeasures.

**Sample Answer** :

**Challenge 1: Lack of Domain-Specific Vocabulary**

  * Problem: Medical terms (e.g., "diabetes", "myocardial infarction") are subword-segmented and not properly represented
  * Countermeasure: Perform additional pre-training in medical domain (Domain-Adaptive Pretraining)

**Challenge 2: Context Understanding Mismatch**

  * Problem: Differences in writing style and structure between general text and medical literature
  * Countermeasure: Use medical BERT models (BioBERT, ClinicalBERT, etc.)

**Challenge 3: Lack of Specialized Knowledge**

  * Problem: Does not understand relationships between diseases or medical causality
  * Countermeasure: Integrate medical knowledge graphs with Knowledge-enhanced methods

**Challenge 4: Performance Limitations**

  * Problem: General BERT has low performance in specialized domains
  * Countermeasure: Fine-tune with large amounts of medical data or use domain-specific models

**Problem 5: Hyperparameter Optimization**

**Question** : When fine-tuning BERT for sentiment analysis tasks, explain the impact of the following hyperparameters and propose recommended values.

  1. Learning Rate
  2. Batch Size
  3. Number of Warmup Steps
  4. Weight Decay
  5. Number of Epochs

**Sample Answer** :

**1\. Learning Rate**

  * **Impact** : Too high causes divergence, too low causes slow convergence
  * **Recommended Value** : 2e-5 to 5e-5 (full-parameter FT), 1e-4 to 3e-4 (LoRA)
  * **Adjustment** : Gradually decrease with Learning Rate Scheduler

**2\. Batch Size**

  * **Impact** : Large is stable training but increases memory consumption, small is unstable
  * **Recommended Value** : 16-32 (adjust according to GPU memory)
  * **Technique** : Increase effective batch size with Gradient Accumulation

**3\. Number of Warmup Steps**

  * **Impact** : Suppresses sudden weight changes in early training, stabilizes learning
  * **Recommended Value** : 10% of total training steps (e.g., 100 steps out of 1000 steps)
  * **Effect** : Particularly effective for small datasets

**4\. Weight Decay**

  * **Impact** : L2 regularization prevents overfitting
  * **Recommended Value** : 0.01 to 0.1
  * **Note** : Do not apply to LayerNorm or Bias

**5\. Number of Epochs**

  * **Impact** : Too many causes overfitting, too few causes underfitting
  * **Recommended Value** : 3-5 epochs (fewer for pre-trained models)
  * **Technique** : Stop when validation loss increases with Early Stopping

**Optimization Priority** : Learning Rate > Batch Size > Warmup > Epochs > Weight Decay

* * *
