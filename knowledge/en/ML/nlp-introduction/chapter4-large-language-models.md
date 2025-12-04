---
title: "Chapter 4: Large Language Models (LLMs)"
chapter_title: "Chapter 4: Large Language Models (LLMs)"
subtitle: GPT, LLaMA, Prompt Engineering - The Frontiers of Language Understanding
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Large Language Models (LLMs). You will learn GPT family architecture, LLM training methods (pre-training, and Practice Prompt Engineering techniques.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand GPT family architecture and autoregressive generation
  * ✅ Understand LLM training methods (pre-training, Instruction Tuning, RLHF)
  * ✅ Practice Prompt Engineering techniques
  * ✅ Understand characteristics of open-source LLMs and quantization techniques
  * ✅ Implement practical applications such as RAG and Function Calling
  * ✅ Build a complete chatbot system

* * *

## 4.1 The GPT Family

### Overview of GPT Architecture

**GPT (Generative Pre-trained Transformer)** is an autoregressive language model that adopts a Decoder-only Transformer architecture.

> **Decoder-only** : Unlike BERT which has both Encoder and Decoder, GPT consists only of Decoder and specializes in next token prediction.

### Features of GPT Architecture

Feature | Description | Advantage  
---|---|---  
**Decoder-only** | Uses only self-attention mechanism | Simple and scalable  
**Causal Masking** | Hides future tokens | Enables autoregressive generation  
**Autoregressive** | Generates sequentially from left to right | Natural text generation  
**Pre-training** | Trained on large-scale text | General language understanding  
  
### Evolution of GPT
    
    
    ```mermaid
    graph LR
        A[GPT-1117M params2018] --> B[GPT-21.5B params2019]
        B --> C[GPT-3175B params2020]
        C --> D[GPT-3.5ChatGPT2022]
        D --> E[GPT-4Multimodal2023]
    
        style A fill:#e3f2fd
        style B fill:#bbdefb
        style C fill:#90caf9
        style D fill:#64b5f6
        style E fill:#42a5f5
    ```

### Comparison of GPT-2/GPT-3/GPT-4

Model | Parameters | Context Length | Key Features  
---|---|---|---  
**GPT-2** | 117M - 1.5B | 1,024 | High-quality text generation  
**GPT-3** | 175B | 2,048 | Few-shot learning, In-context Learning  
**GPT-3.5** | ~175B | 4,096 | Instruction Tuning, improved dialogue  
**GPT-4** | Undisclosed | 8,192 - 32,768 | Multimodal, advanced reasoning  
  
### Autoregressive Generation

GPT generates text **autoregressively** by predicting the next token from previous tokens.

$$ P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1}) $$

The probability of each token is conditioned on all preceding tokens.

### Example: Text Generation with GPT-2
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Example: Text Generation with GPT-2
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Text generation
    prompt = "Artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generation parameters
    generation_config = {
        'max_length': 100,
        'num_return_sequences': 3,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'do_sample': True,
        'no_repeat_ngram_size': 2
    }
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            **generation_config
        )
    
    print("=== Text Generation with GPT-2 ===")
    print(f"Prompt: '{prompt}'\n")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generation {i+1}:")
        print(f"{text}\n")
    

**Sample Output** :
    
    
    === Text Generation with GPT-2 ===
    Prompt: 'Artificial intelligence is'
    
    Generation 1:
    Artificial intelligence is becoming more and more important in our daily lives. From smartphones to self-driving cars, AI systems are transforming the way we work and live. The technology has advanced rapidly...
    
    Generation 2:
    Artificial intelligence is a field of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence, such as visual perception...
    
    Generation 3:
    Artificial intelligence is revolutionizing industries across the globe. Companies are investing billions in AI research to develop systems that can learn from data and make decisions autonomously...
    

### Controlling Generation Parameters
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Controlling Generation Parameters
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generation with different temperatures
    prompt = "The future of AI is"
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    print("=== Impact of Temperature ===\n")
    
    for temp in temperatures:
        inputs = tokenizer(prompt, return_tensors="pt")
    
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                temperature=temp,
                do_sample=True,
                top_k=50
            )
    
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Temperature = {temp}:")
        print(f"{text}\n")
    
    # Visualize temperature and probability distribution
    def softmax_with_temperature(logits, temperature):
        """Apply softmax with temperature parameter"""
        return torch.softmax(logits / temperature, dim=-1)
    
    # Sample logits
    logits = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1])
    temps = [0.5, 1.0, 2.0]
    
    plt.figure(figsize=(12, 4))
    for i, temp in enumerate(temps):
        probs = softmax_with_temperature(logits, temp)
    
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(probs)), probs.numpy())
        plt.title(f'Temperature = {temp}')
        plt.xlabel('Token ID')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Low temperature → Deterministic (select high-probability tokens)")
    print("📊 High temperature → Diverse (also select low-probability tokens)")
    

### Beam Search and Sampling
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: Beam Search and Sampling
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from transformers import GenerationConfig
    
    prompt = "Machine learning can be used for"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("=== Comparison of Generation Strategies ===\n")
    
    # 1. Greedy Decoding
    print("1. Greedy Decoding:")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            do_sample=False
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()
    
    # 2. Beam Search
    print("2. Beam Search (num_beams=5):")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            num_beams=5,
            do_sample=False
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()
    
    # 3. Top-k Sampling
    print("3. Top-k Sampling (k=50):")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            do_sample=True,
            top_k=50
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()
    
    # 4. Top-p (Nucleus) Sampling
    print("4. Top-p Sampling (p=0.95):")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            do_sample=True,
            top_p=0.95,
            top_k=0
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

> **Important** : Greedy/Beam Search are deterministic and consistent, while Sampling provides diversity but lower reproducibility.

* * *

## 4.2 LLM Training Methods

### Overall Picture of LLM Training
    
    
    ```mermaid
    graph TD
        A[Large-scale Text Corpus] --> B[Pre-trainingPre-training]
        B --> C[Base ModelBase LLM]
        C --> D[Instruction TuningInstruction Tuning]
        D --> E[Instruction-following ModelInstruction-following LLM]
        E --> F[RLHFReinforcement Learning from Human Feedback]
        F --> G[Aligned ModelAligned LLM]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### 1\. Pre-training

**Pre-training** involves learning next token prediction tasks on large-scale text corpora.

Element | Content  
---|---  
**Data** | Hundreds of billions to trillions of tokens from web, books, papers, etc.  
**Objective Function** | Next token prediction (Language Modeling)  
**Optimization** | AdamW, learning rate scheduling  
**Compute Resources** | Thousands to tens of thousands of GPUs, weeks to months  
  
**Loss Function** :

$$ \mathcal{L}_{\text{LM}} = -\sum_{i=1}^{n} \log P(x_i | x_{

### 2\. Instruction Tuning

**Instruction Tuning** fine-tunes the model to follow user instructions.

#### Instruction Tuning Data Format
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Instruction Tuning Data Format
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    instruction_data = [
        {
            "instruction": "Summarize the following text.",
            "input": "Artificial Intelligence (AI) is technology that enables computers to think, learn, and solve problems like humans. It includes fields such as machine learning, deep learning, and natural language processing.",
            "output": "AI is technology that gives computers human-like intelligence, including machine learning and natural language processing."
        },
        {
            "instruction": "Answer the following question.",
            "input": "How far is the distance from Earth to the Moon?",
            "output": "The average distance from Earth to the Moon is approximately 384,400 kilometers (238,855 miles)."
        },
        {
            "instruction": "Fix the bug in the following code.",
            "input": "def add(a, b):\n    return a - b",
            "output": "def add(a, b):\n    return a + b"
        }
    ]
    
    # Visualize data format
    import pandas as pd
    
    df = pd.DataFrame(instruction_data)
    print("=== Instruction Tuning Data Format ===")
    print(df.to_string(index=False))
    print(f"\nNumber of data points: {len(instruction_data)} items")
    

#### Instruction Tuning Implementation Example (Conceptual)
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset
    
    # Prepare data
    def format_instruction(example):
        """Convert instruction data to prompt format"""
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example['input']:
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}
    
    # Create dataset
    dataset = Dataset.from_list(instruction_data)
    dataset = dataset.map(format_instruction)
    
    print("=== Formatted Prompt Example ===")
    print(dataset[0]['text'])
    print("\n" + "="*50 + "\n")
    
    # Prepare model and tokenizer (in practice, use large-scale models)
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training configuration (for demo)
    training_args = TrainingArguments(
        output_dir="./instruction-tuned-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no"
    )
    
    print("=== Instruction Tuning Configuration ===")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print("\n✓ Instruction Tuning makes the model follow instructions")
    

### 3\. RLHF (Reinforcement Learning from Human Feedback)

**RLHF** is a technique that uses human feedback to align the model with human values.

#### Three Steps of RLHF
    
    
    ```mermaid
    graph LR
        A[Step 1:Supervised Fine-tuningSFT] --> B[Step 2:Reward Model TrainingReward Model]
        B --> C[Step 3:PPO Reinforcement LearningPolicy Optimization]
    
        style A fill:#e3f2fd
        style B fill:#f3e5f5
        style C fill:#e8f5e9
    ```

#### Step 1: Supervised Fine-Tuning (SFT)

Fine-tune with high-quality human-generated dialogue data.

#### Step 2: Reward Model Training

Train a reward model from human evaluations (rankings).
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Train a reward model from human evaluations (rankings).
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Reward model data format (conceptual)
    reward_data = [
        {
            "prompt": "What is artificial intelligence?",
            "response_1": "Artificial Intelligence (AI) is technology that enables computers to think like humans.",
            "response_2": "I don't know.",
            "preference": 1  # response_1 is better
        },
        {
            "prompt": "How to reverse a list in Python?",
            "response_1": "Use the list.reverse() method or list[::-1] slicing.",
            "response_2": "Cannot do it.",
            "preference": 1
        }
    ]
    
    import pandas as pd
    df_reward = pd.DataFrame(reward_data)
    print("=== Reward Model Training Data ===")
    print(df_reward.to_string(index=False))
    print("\n✓ Learn reward function from human preferences")
    

#### Step 3: PPO (Proximal Policy Optimization)

Optimize the model using reinforcement learning with the reward model.

**Objective Function** :

$$ \mathcal{L}^{\text{PPO}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ r(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}}) \right] $$

  * $r(x, y)$: Reward model score
  * $\beta$: KL regularization coefficient
  * $\pi_{\text{ref}}$: Original model (to prevent large deviation)

> **Effect of RLHF** : ChatGPT's human-like dialogue capabilities are realized through RLHF.

### 4\. Parameter-Efficient Fine-Tuning (PEFT)

Since fine-tuning entire large-scale models is computationally expensive, **parameter-efficient fine-tuning** is important.

#### LoRA (Low-Rank Adaptation)

LoRA applies low-rank decomposition to model weight matrices.

$$ W' = W + \Delta W = W + BA $$

  * $W$: Original weight matrix (frozen)
  * $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Trainable low-rank matrices
  * $r \ll \min(d, k)$: Rank (e.g., 8, 16)

    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: $$
    W' = W + \Delta W = W + BA
    $$
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM
    
    # Load base model
    model_name = "gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=["c_attn"]  # Modules to apply
    )
    
    # Create LoRA model
    lora_model = get_peft_model(base_model, lora_config)
    
    # Compare parameter counts
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print("=== LoRA Parameter Efficiency ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"\n✓ Fine-tune with only {100 * trainable_params / total_params:.2f}% of parameters")
    print(lora_model.print_trainable_parameters())
    

**Sample Output** :
    
    
    === LoRA Parameter Efficiency ===
    Total parameters: 124,439,808
    Trainable parameters: 294,912
    Trainable ratio: 0.24%
    
    ✓ Fine-tune with only 0.24% of parameters
    trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.24
    

#### Adapter Layers

Add small bottleneck networks to each layer of the Transformer.

Method | Trainable Parameters | Memory | Speed  
---|---|---|---  
**Full Fine-tuning** | 100% | High | Slow  
**LoRA** | 0.1% - 1% | Low | Fast  
**Adapter** | 2% - 5% | Medium | Medium  
**Prompt Tuning** | < 0.1% | Very low | Very fast  
  
* * *

## 4.3 Prompt Engineering

### What is Prompt Engineering

**Prompt Engineering** is the technique of designing inputs (prompts) to elicit desired outputs from LLMs.

> "A good prompt can increase model performance tenfold" - OpenAI

### 1\. Zero-shot Learning

Directly ask questions to a pre-trained model without providing any task examples.
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    from transformers import pipeline
    
    # GPT-2 pipeline
    generator = pipeline('text-generation', model='gpt2')
    
    # Zero-shot prompt
    zero_shot_prompt = """
    Question: What is the capital of France?
    Answer:
    """
    
    result = generator(zero_shot_prompt, max_length=50, num_return_sequences=1)
    print("=== Zero-shot Learning ===")
    print(result[0]['generated_text'])
    

### 2\. Few-shot Learning

Provide a few task examples before giving a new input.
    
    
    # Few-shot prompt (In-Context Learning)
    few_shot_prompt = """
    Translate English to French:
    
    English: Hello
    French: Bonjour
    
    English: Thank you
    French: Merci
    
    English: Good morning
    French: Bon matin
    
    English: How are you?
    French:
    """
    
    result = generator(few_shot_prompt, max_length=100, num_return_sequences=1)
    print("\n=== Few-shot Learning ===")
    print(result[0]['generated_text'])
    print("\n✓ By showing examples, the model learns patterns and performs translation")
    

### 3\. Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** is a technique that prompts the model to generate intermediate reasoning steps.
    
    
    # Standard prompt (direct answer)
    standard_prompt = """
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
    Each can has 3 tennis balls. How many tennis balls does he have now?
    Answer:
    """
    
    # Chain-of-Thought prompt
    cot_prompt = """
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
    Each can has 3 tennis balls. How many tennis balls does he have now?
    
    Let's think step by step:
    1. Roger starts with 5 tennis balls.
    2. He buys 2 cans, each containing 3 tennis balls.
    3. So he gets 2 × 3 = 6 new tennis balls.
    4. Total tennis balls = 5 + 6 = 11.
    
    Answer: 11 tennis balls.
    
    Question: A restaurant had 23 customers. Then 11 more customers arrived.
    Each customer ordered 2 drinks. How many drinks were ordered in total?
    
    Let's think step by step:
    """
    
    result = generator(cot_prompt, max_length=200, num_return_sequences=1)
    print("\n=== Chain-of-Thought Prompting ===")
    print(result[0]['generated_text'])
    print("\n✓ Step-by-step reasoning solves complex problems")
    

#### Effectiveness of CoT (Research Results)

Task | Standard Prompt | CoT Prompt | Improvement  
---|---|---|---  
Math problems | 34% | 78% | +44%  
Common sense reasoning | 61% | 89% | +28%  
Logic puzzles | 42% | 81% | +39%  
  
### 4\. Best Practices for Prompt Design
    
    
    # ❌ Bad prompt
    bad_prompt = "Summarize this."
    
    # ✅ Good prompt
    good_prompt = """
    Task: Summarize the following article in 3 bullet points.
    Focus on key findings and implications.
    
    Article: [Long article text...]
    
    Summary:
    -
    """
    
    # Principles of prompt design
    prompt_principles = {
        "Clear instructions": "Describe the task specifically",
        "Format specification": "Show the desired output format",
        "Provide context": "Include necessary background information",
        "State constraints": "Specify character limits, style, etc.",
        "Show examples": "Demonstrate expected output with few-shot",
        "Break down steps": "Divide complex tasks into stages"
    }
    
    print("=== Principles of Prompt Design ===")
    for principle, description in prompt_principles.items():
        print(f"✓ {principle}: {description}")
    
    # Practical example: Structured prompt
    structured_prompt = """
    Role: You are an expert Python programmer.
    
    Task: Review the following code and provide feedback.
    
    Code:
    ```python
    def calculate_average(numbers):
        return sum(numbers) / len(numbers)
    ```
    
    Output Format:
    1. Code Quality (1-10):
    2. Issues Found:
    3. Suggestions:
    4. Improved Code:
    
    Analysis:
    """
    
    print("\n=== Structured Prompt Example ===")
    print(structured_prompt)
    

### 5\. Prompt Management with LangChain
    
    
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms import HuggingFacePipeline
    
    # Wrap HuggingFace pipeline with LangChain
    llm = HuggingFacePipeline(pipeline=generator)
    
    # Create prompt template
    template = """
    Question: {question}
    
    Context: {context}
    
    Please provide a detailed answer based on the context above.
    
    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Execute
    question = "What is machine learning?"
    context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    
    result = chain.run(question=question, context=context)
    
    print("=== LangChain Prompt Template ===")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"\nGenerated answer:")
    print(result)
    
    print("\n✓ Manage reusable prompt templates with LangChain")
    

* * *

## 4.4 Open-Source LLMs

### Major Open-Source LLMs

Model | Developer | Parameters | Features  
---|---|---|---  
**LLaMA** | Meta AI | 7B - 65B | High performance, research use  
**LLaMA-2** | Meta AI | 7B - 70B | Commercial use, Chat version available  
**Falcon** | TII | 7B - 180B | High-quality dataset  
**ELYZA** | ELYZA Inc. | 7B - 13B | Japanese-specialized  
**rinna** | rinna | 3.6B - 36B | Japanese, commercial use allowed  
  
### 1\. LLaMA / LLaMA-2

**LLaMA (Large Language Model Meta AI)** is an open-source LLM developed by Meta AI.

#### Features of LLaMA

  * **Efficient architecture** : Equivalent performance to GPT-3 with fewer parameters
  * **Pre-Normalization** : Uses RMSNorm
  * **SwiGLU activation function** : Improved performance
  * **Rotary Positional Embeddings (RoPE)** : Position encoding

    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Features of LLaMA
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # Load LLaMA-2 model (7B model)
    model_name = "meta-llama/Llama-2-7b-hf"  # From Hugging Face Hub
    
    # Note: Access request required on Hugging Face to use LLaMA-2
    # Using GPT-2 here for demo purposes
    
    model_name = "gpt2"  # Replace with LLaMA-2 in actual environment
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("=== LLaMA Family Information ===")
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: Decoder-only Transformer")
    print(f"Context length: 2048 (LLaMA) / 4096 (LLaMA-2)")
    
    # LLaMA-2 performance (benchmark results)
    benchmarks = {
        "MMLU": {"LLaMA-7B": 35.1, "LLaMA-2-7B": 45.3, "GPT-3.5": 70.0},
        "HellaSwag": {"LLaMA-7B": 76.1, "LLaMA-2-7B": 77.2, "GPT-3.5": 85.5},
        "HumanEval": {"LLaMA-7B": 10.5, "LLaMA-2-7B": 12.8, "GPT-3.5": 48.1}
    }
    
    import pandas as pd
    df_bench = pd.DataFrame(benchmarks)
    print("\n=== Benchmark Performance Comparison ===")
    print(df_bench.to_string())
    

### 2\. Japanese LLMs (ELYZA, rinna)

#### ELYZA-japanese-Llama-2

A model that continues pre-training LLaMA-2 with Japanese text.
    
    
    # ELYZA-japanese-Llama-2 usage example (conceptual)
    # model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    
    japanese_prompt = """
    Please answer the following question in Japanese.
    
    Question: What is the difference between machine learning and deep learning?
    
    Answer:
    """
    
    print("=== Japanese LLM (ELYZA) ===")
    print("✓ LLaMA-2 base + Japanese additional training")
    print("✓ Natural dialogue in Japanese possible")
    print("✓ Commercial use allowed (LLaMA-2 license)")
    print(f"\nPrompt example:\n{japanese_prompt}")
    
    # rinna GPT-NeoX
    print("\n=== Japanese LLM (rinna) ===")
    print("✓ GPT-NeoX architecture")
    print("✓ Trained on Japanese Wikipedia, etc.")
    print("✓ 3.6B, 36B models available")
    print("✓ Commercial use allowed (MIT License)")
    

### 3\. Model Quantization

**Quantization** is a technique to reduce memory and computation by lowering model precision.

#### Types of Quantization

Precision | Memory Reduction | Performance Loss | Use Case  
---|---|---|---  
**FP32 (Original)** | - | - | Training  
**FP16** | 50% | Almost none | Inference  
**8-bit** | 75% | Small | Inference, fine-tuning  
**4-bit** | 87.5% | Medium | Memory-constrained environments  
  
#### 8-bit Quantization Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: 8-bit Quantization Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    # 8-bit quantization configuration
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    # 4-bit quantization configuration (QLoRA)
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_use_double_quant=True
    )
    
    # Load model (8-bit)
    # model_8bit = AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf",
    #     quantization_config=quantization_config_8bit,
    #     device_map="auto"
    # )
    
    print("=== Model Quantization ===")
    print("\n8-bit quantization:")
    print("  ✓ Memory usage: ~7GB (7B model, 75% reduction from FP32)")
    print("  ✓ Performance loss: Minimal (1-2%)")
    print("  ✓ Inference speed: Almost the same as FP32")
    
    print("\n4-bit quantization (QLoRA):")
    print("  ✓ Memory usage: ~3.5GB (7B model, 87.5% reduction from FP32)")
    print("  ✓ Performance loss: Small (5-10%)")
    print("  ✓ Fine-tuning possible")
    
    # Calculate memory usage
    def calculate_model_memory(num_params, precision):
        """Calculate model memory usage"""
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            '8bit': 1,
            '4bit': 0.5
        }
        memory_gb = num_params * bytes_per_param[precision] / (1024**3)
        return memory_gb
    
    params_7b = 7_000_000_000
    
    print("\n=== Memory Usage for 7B Model ===")
    for precision in ['fp32', 'fp16', '8bit', '4bit']:
        memory = calculate_model_memory(params_7b, precision)
        print(f"{precision.upper():6s}: {memory:.2f} GB")
    

**Output** :
    
    
    === Memory Usage for 7B Model ===
    FP32  : 26.08 GB
    FP16  : 13.04 GB
    8BIT  : 6.52 GB
    4BIT  : 3.26 GB
    

> **QLoRA (Quantized LoRA)** : Combines 4-bit quantization with LoRA, enabling fine-tuning of large models on consumer GPUs.

* * *

## 4.5 Practical Applications of LLMs

### 1\. Retrieval-Augmented Generation (RAG)

**RAG** is a technique that retrieves relevant information from an external knowledge base and generates answers based on it.
    
    
    ```mermaid
    graph LR
        A[User Query] --> B[RetrievalRetriever]
        B --> C[Knowledge BaseVector DB]
        C --> D[Relevant Documents]
        D --> E[LLMGenerator]
        A --> E
        E --> F[Answer Generation]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#c8e6c9
    ```

#### RAG Implementation Example
    
    
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    
    # Knowledge base text
    documents = [
        "Artificial Intelligence (AI) is technology that enables computers to think and learn like humans.",
        "Machine learning is a branch of AI that learns patterns from data.",
        "Deep learning is a machine learning method using multi-layer neural networks.",
        "Natural Language Processing (NLP) is technology that enables computers to understand human language.",
        "Transformer is a revolutionary neural network architecture introduced in 2017."
    ]
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    texts = text_splitter.create_documents(documents)
    
    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    print("=== RAG System Construction ===")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of chunks: {len(texts)}")
    print(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    
    # Search test
    query = "What is Transformer?"
    relevant_docs = vectorstore.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print("\nRelevant documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    # Integrate with LLM (conceptual)
    # llm = HuggingFacePipeline(...)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     return_source_documents=True
    # )
    # result = qa_chain({"query": query})
    
    print("\n✓ RAG enables LLM to answer with latest/domain-specific knowledge")
    

### 2\. Function Calling

**Function Calling** is a technique that enables LLMs to call external tools and APIs.
    
    
    import json
    
    # Define available functions
    available_functions = {
        "get_weather": {
            "description": "Get current weather for specified city",
            "parameters": {
                "city": {"type": "string", "description": "City name"}
            }
        },
        "calculate": {
            "description": "Perform mathematical calculation",
            "parameters": {
                "expression": {"type": "string", "description": "Mathematical expression"}
            }
        },
        "search_web": {
            "description": "Search the web",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        }
    }
    
    # Function implementations (dummy)
    def get_weather(city):
        """Get weather information (dummy)"""
        return f"The weather in {city} is sunny, temperature is 25 degrees."
    
    def calculate(expression):
        """Execute calculation"""
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "Calculation error"
    
    def search_web(query):
        """Web search (dummy)"""
        return f"Search results for '{query}': [Related information...]"
    
    # Function Calling prompt
    def create_function_calling_prompt(user_query, functions):
        """Generate prompt for Function Calling"""
        functions_desc = json.dumps(functions, ensure_ascii=False, indent=2)
    
        prompt = f"""
    You are a helpful assistant with access to the following functions:
    
    {functions_desc}
    
    User query: {user_query}
    
    Based on the query, determine which function to call and with what parameters.
    Respond in JSON format:
    {{
        "function": "function_name",
        "parameters": {{...}}
    }}
    
    Response:
    """
        return prompt
    
    # Test
    user_query = "What's the weather in Tokyo?"
    prompt = create_function_calling_prompt(user_query, available_functions)
    
    print("=== Function Calling ===")
    print(f"User query: {user_query}")
    print(f"\nPrompt:\n{prompt}")
    
    # Expected response (actually generated by LLM)
    function_call = {
        "function": "get_weather",
        "parameters": {"city": "Tokyo"}
    }
    
    print(f"\nLLM's function selection:")
    print(json.dumps(function_call, ensure_ascii=False, indent=2))
    
    # Execute function
    if function_call["function"] == "get_weather":
        result = get_weather(**function_call["parameters"])
        print(f"\nExecution result: {result}")
    
    print("\n✓ LLM can use tools to retrieve and process information")
    

### 3\. Multi-turn Conversation

Maintain conversation history to achieve context-aware dialogue.
    
    
    from collections import deque
    
    class ConversationManager:
        """Class to manage conversation history"""
    
        def __init__(self, max_history=10):
            self.history = deque(maxlen=max_history)
    
        def add_message(self, role, content):
            """Add message"""
            self.history.append({"role": role, "content": content})
    
        def get_prompt(self, system_message=""):
            """Generate prompt from conversation history"""
            prompt_parts = []
    
            if system_message:
                prompt_parts.append(f"System: {system_message}\n")
    
            for msg in self.history:
                prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    
            prompt_parts.append("Assistant:")
            return "\n".join(prompt_parts)
    
        def clear(self):
            """Clear history"""
            self.history.clear()
    
    # Usage example
    conv_manager = ConversationManager(max_history=6)
    
    system_msg = "You are a helpful AI assistant."
    
    # Conversation simulation
    conversation = [
        ("User", "Hello!"),
        ("Assistant", "Hello! How can I help you?"),
        ("User", "Please teach me about Python learning methods."),
        ("Assistant", "For Python learning, I recommend first learning basic syntax, then working on actual projects."),
        ("User", "What projects are good for beginners?"),
    ]
    
    print("=== Multi-turn Conversation ===\n")
    
    for i, (role, content) in enumerate(conversation):
        conv_manager.add_message(role, content)
    
        if role == "User":
            print(f"{role}: {content}")
            prompt = conv_manager.get_prompt(system_msg)
            print(f"\n[Generated Prompt]")
            print(prompt)
            print("\n" + "="*50 + "\n")
    
    print("✓ Conversation history enables context-aware dialogue")
    print(f"✓ History length: {len(conv_manager.history)} messages")
    

### 4\. Complete Chatbot Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    class SimpleChatbot:
        """Simple chatbot"""
    
        def __init__(self, model_name="gpt2", max_history=5):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.conversation = ConversationManager(max_history=max_history)
            self.system_message = "You are a helpful AI assistant."
    
        def generate_response(self, user_input, max_length=100):
            """Generate response to user input"""
            # Add to conversation history
            self.conversation.add_message("User", user_input)
    
            # Generate prompt
            prompt = self.conversation.get_prompt(self.system_message)
    
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
    
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=len(inputs['input_ids'][0]) + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
    
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
            # Remove prompt part
            response = response[len(prompt):].strip()
    
            # Add to conversation history
            self.conversation.add_message("Assistant", response)
    
            return response
    
        def chat(self):
            """Dialogue loop"""
            print("=== Chatbot Started ===")
            print("Type 'quit' to exit\n")
    
            while True:
                user_input = input("You: ")
    
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Assistant: Goodbye!")
                    break
    
                response = self.generate_response(user_input)
                print(f"Assistant: {response}\n")
    
    # Instantiate chatbot
    chatbot = SimpleChatbot(model_name="gpt2", max_history=5)
    
    # Demo dialogue (in practice use chatbot.chat() for dialogue loop)
    demo_inputs = [
        "Hello!",
        "What is AI?",
        "Can you explain more?"
    ]
    
    print("=== Chatbot Demo ===\n")
    for user_input in demo_inputs:
        print(f"You: {user_input}")
        response = chatbot.generate_response(user_input, max_length=50)
        print(f"Assistant: {response}\n")
    
    print("✓ Complete chatbot maintaining conversation history")
    print("✓ Context-aware response generation")
    print("✓ Extensible design (can add RAG, Function Calling, etc.)")
    

* * *

## 4.6 Chapter Summary

### What We Learned

  1. **GPT Family**

     * Decoder-only architecture
     * Autoregressive text generation
     * Evolution from GPT-2 to GPT-4
     * Control of generation parameters (temperature, top-k, top-p)
  2. **LLM Training Methods**

     * Pre-training: Next token prediction on large corpus
     * Instruction Tuning: Towards instruction-following models
     * RLHF: Alignment through human feedback
     * PEFT: Efficient fine-tuning with LoRA, Adapter
  3. **Prompt Engineering**

     * Zero-shot / Few-shot Learning
     * Chain-of-Thought Prompting
     * Best practices for prompt design
     * Prompt management with LangChain
  4. **Open-Source LLMs**

     * LLaMA, Falcon, Japanese LLMs
     * Model quantization (8-bit, 4-bit)
     * Efficient fine-tuning with QLoRA
  5. **Practical Applications**

     * RAG: Generation leveraging external knowledge
     * Function Calling: Tool/API integration
     * Multi-turn Conversation: Context-aware dialogue
     * Complete chatbot implementation

### Best Practices for LLM Utilization

Aspect | Recommendation  
---|---  
**Model Selection** | Balance size and performance according to task  
**Prompt Design** | Clear instructions, examples, structure  
**Memory Efficiency** | Utilize quantization, PEFT  
**Knowledge Update** | Integrate latest information with RAG  
**Safety** | Output validation, harmful content filtering  
  
### Next Steps

To deepen your understanding of large language models:

  * Try larger models (LLaMA-2 13B/70B)
  * Fine-tune with custom data (LoRA)
  * Deploy RAG systems to production
  * Explore multimodal LLMs (GPT-4V, etc.)
  * Learn LLMOps (operations and monitoring)

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Explain the difference between GPT's Decoder-only architecture and BERT's Encoder-only architecture. What tasks is each suited for?

Sample Answer

**Answer** :

**GPT (Decoder-only)** :

  * Structure: Self-attention mechanism + Causal Masking (doesn't see future)
  * Training: Next token prediction (autoregressive)
  * Strengths: Text generation, dialogue, creative writing
  * Use cases: ChatGPT, code generation, text composition

**BERT (Encoder-only)** :

  * Structure: Bidirectional self-attention mechanism (references all tokens)
  * Training: Masked Language Modeling (fill-in-the-blank)
  * Strengths: Text understanding, classification, extraction
  * Use cases: Sentiment analysis, named entity recognition, question answering

**Usage Guidelines** :

Task | Recommended Model  
---|---  
Text generation | GPT  
Document classification | BERT  
Question answering (extractive) | BERT  
Question answering (generative) | GPT  
Summarization | GPT (or Encoder-Decoder)  
  
### Exercise 2 (Difficulty: medium)

Explain why Chain-of-Thought (CoT) Prompting is effective for complex reasoning tasks. Provide concrete examples.

Sample Answer

**Answer** :

**Why CoT is Effective** :

  1. **Explicit intermediate reasoning** : Verbalizing steps helps LLM organize logical thinking
  2. **Early error detection** : Easier to notice mistakes at each step
  3. **Complexity decomposition** : Divide difficult problems into smaller sub-problems
  4. **Enhanced In-context Learning** : Learn reasoning patterns

**Concrete Example: Math Problem**
    
    
    # ❌ Standard prompt
    prompt_standard = """
    Question: A store had 25 apples. They sold 8 apples in the morning
    and 12 apples in the afternoon. How many apples are left?
    Answer:
    """
    
    # ✅ CoT prompt
    prompt_cot = """
    Question: A store had 25 apples. They sold 8 apples in the morning
    and 12 apples in the afternoon. How many apples are left?
    
    Let's solve this step by step:
    1. The store started with 25 apples.
    2. They sold 8 apples in the morning: 25 - 8 = 17 apples remaining.
    3. They sold 12 apples in the afternoon: 17 - 12 = 5 apples remaining.
    
    Answer: 5 apples are left.
    """
    

**Demonstrated Effectiveness** (from research):

Task | Standard | CoT | Improvement  
---|---|---|---  
GSM8K (Math) | 17.9% | 58.1% | +40.2%  
SVAMP (Math) | 69.4% | 78.7% | +9.3%  
  
### Exercise 3 (Difficulty: medium)

Explain why LoRA (Low-Rank Adaptation) is parameter-efficient using mathematical formulas.

Sample Answer

**Answer** :

**Principle of LoRA** :

Instead of updating the original weight matrix $W \in \mathbb{R}^{d \times k}$, use low-rank decomposition:

$$ W' = W + \Delta W = W + BA $$

  * $W$: Original weights (frozen, not trained)
  * $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Trainable matrices
  * $r \ll \min(d, k)$: Rank (e.g., $r=8$)

**Parameter Reduction Calculation** :

Example: $d = 4096$, $k = 4096$, $r = 8$:

  * Original weights: $4096 \times 4096 = 16,777,216$ parameters
  * LoRA: $4096 \times 8 + 8 \times 4096 = 65,536$ parameters
  * Reduction rate: $\frac{65,536}{16,777,216} \approx 0.39\%$ (99.6% reduction)

**Implementation Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation Example:
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    class LoRALayer(nn.Module):
        def __init__(self, in_features, out_features, rank=8):
            super().__init__()
            # Original weights (frozen)
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.weight.requires_grad = False
    
            # LoRA matrices (trainable)
            self.lora_A = nn.Parameter(torch.randn(rank, in_features))
            self.lora_B = nn.Parameter(torch.randn(out_features, rank))
    
            self.rank = rank
    
        def forward(self, x):
            # W*x + B*A*x
            return x @ self.weight.T + x @ self.lora_A.T @ self.lora_B.T
    
    # Example
    layer = LoRALayer(4096, 4096, rank=8)
    total = sum(p.numel() for p in layer.parameters())
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    

**Output** :
    
    
    Total parameters: 16,842,752
    Trainable: 65,536 (0.39%)
    

### Exercise 4 (Difficulty: hard)

Implement a RAG (Retrieval-Augmented Generation) system. Meet the following requirements:

  * Retrieval from custom knowledge base
  * Scoring of relevant documents
  * Answer generation using retrieval results

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SimpleRAG:
        """Simple RAG system"""
    
        def __init__(self, knowledge_base, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
            self.knowledge_base = knowledge_base
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model)
    
            # Pre-compute embeddings for knowledge base
            self.kb_embeddings = self._embed_documents(knowledge_base)
    
        def _mean_pooling(self, model_output, attention_mask):
            """Mean pooling"""
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
        def _embed_documents(self, documents):
            """Convert documents to embedding vectors"""
            encoded = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
    
            with torch.no_grad():
                model_output = self.model(**encoded)
    
            embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            return embeddings.numpy()
    
        def retrieve(self, query, top_k=3):
            """Retrieve documents relevant to query"""
            # Query embedding
            query_embedding = self._embed_documents([query])
    
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
    
            # Get top-k documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
    
            results = []
            for idx in top_indices:
                results.append({
                    'document': self.knowledge_base[idx],
                    'score': similarities[idx]
                })
    
            return results
    
        def generate_answer(self, query, retrieved_docs):
            """Generate answer using retrieval results (create prompt)"""
            context = "\n".join([f"- {doc['document']}" for doc in retrieved_docs])
    
            prompt = f"""
    Based on the following context, answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
            return prompt
    
    # Knowledge base
    knowledge_base = [
        "Machine learning is a field of AI where computers learn from data.",
        "Deep learning is a machine learning method using multi-layer neural networks.",
        "Transformer is a revolutionary architecture introduced in 2017.",
        "BERT is a text understanding model using bidirectional Transformers.",
        "GPT is a Decoder-only Transformer excellent at text generation.",
        "Fine-tuning is a method to adapt pre-trained models to specific tasks.",
        "LoRA is a parameter-efficient fine-tuning method.",
        "RAG is a technique combining retrieval and generation."
    ]
    
    # Create RAG system
    rag = SimpleRAG(knowledge_base)
    
    # Test
    query = "Tell me about Transformer"
    
    print("=== RAG System ===")
    print(f"Knowledge base: {len(knowledge_base)} items\n")
    print(f"Query: {query}\n")
    
    # Retrieval
    retrieved = rag.retrieve(query, top_k=3)
    
    print("Retrieval results:")
    for i, doc in enumerate(retrieved, 1):
        print(f"{i}. [Score: {doc['score']:.3f}] {doc['document']}")
    
    # Prompt for answer generation
    prompt = rag.generate_answer(query, retrieved)
    print(f"\nGenerated prompt:\n{prompt}")
    
    print("\n✓ RAG system retrieves relevant knowledge and uses it for answer generation")
    

**Sample Output** :
    
    
    === RAG System ===
    Knowledge base: 8 items
    
    Query: Tell me about Transformer
    
    Retrieval results:
    1. [Score: 0.712] Transformer is a revolutionary architecture introduced in 2017.
    2. [Score: 0.623] BERT is a text understanding model using bidirectional Transformers.
    3. [Score: 0.589] GPT is a Decoder-only Transformer excellent at text generation.
    
    Generated prompt:
    
    Based on the following context, answer the question.
    
    Context:
    - Transformer is a revolutionary architecture introduced in 2017.
    - BERT is a text understanding model using bidirectional Transformers.
    - GPT is a Decoder-only Transformer excellent at text generation.
    
    Question: Tell me about Transformer
    
    Answer:
    

### Exercise 5 (Difficulty: hard)

Explain the three steps of RLHF (SFT, Reward Model Training, PPO), their roles, and why they need to be performed in this order.

Sample Answer

**Answer** :

**Step 1: Supervised Fine-Tuning (SFT)**

  * **Role** : Fine-tune model with high-quality human-generated dialogue data
  * **Purpose** : Adapt base model to dialogue tasks
  * **Data** : Pairs of (prompt, ideal response)
  * **Necessity** : Pre-trained models are not optimized for dialogue

**Step 2: Reward Model Training**

  * **Role** : Train a reward model that learns human preferences
  * **Purpose** : Enable automatic evaluation of "good response" vs "bad response"
  * **Data** : Rankings of multiple responses to the same prompt
  * **Necessity** : Provide reward signal needed for reinforcement learning

**Loss Function** :

$$ \mathcal{L}_{\text{reward}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right] $$

  * $y_w$: Preferred response
  * $y_l$: Non-preferred response
  * $r(x, y)$: Reward score

**Step 3: PPO (Proximal Policy Optimization)**

  * **Role** : Optimize model using reward model
  * **Purpose** : Learn to generate responses aligned with human preferences
  * **Method** : Reinforcement learning (PPO algorithm)
  * **Constraint** : KL regularization to prevent large deviation from original model

**Objective Function** :

$$ \mathcal{L}^{\text{PPO}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ r(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{SFT}}) \right] $$

**Why This Order is Necessary** :

  1. **SFT First** : 
     * Base model is unfamiliar with dialogue
     * Serves as initial policy for reinforcement learning
     * Also used for generating data for reward model training
  2. **Reward Model Second** : 
     * Provides reward signal needed for PPO
     * Learns from human ranking data
     * Evaluates responses generated by SFT model
  3. **PPO Last** : 
     * Cannot optimize without reward model
     * Uses SFT model as initial policy
     * KL regularization prevents large deviation from SFT

**Overall Flow** :
    
    
    ```mermaid
    graph TD
        A[Pre-trained Model] --> B[SFT]
        B --> C[SFT Model]
        C --> D[Response Generation]
        D --> E[Human Ranking]
        E --> F[Reward Model Training]
        F --> G[Reward Model]
        C --> H[PPO]
        G --> H
        H --> I[Aligned ModelChatGPT]
    ```

**Effects** :

Stage | Performance Metric  
---|---  
Base Model | Dialogue quality: Low  
SFT | Dialogue quality: Medium (follows instructions)  
RLHF (PPO) | Dialogue quality: High (aligned with human preferences)  
  
* * *

## References

  1. Vaswani, A., et al. (2017). _Attention is All You Need_. NeurIPS.
  2. Radford, A., et al. (2019). _Language Models are Unsupervised Multitask Learners_ (GPT-2). OpenAI.
  3. Brown, T., et al. (2020). _Language Models are Few-Shot Learners_ (GPT-3). NeurIPS.
  4. Ouyang, L., et al. (2022). _Training language models to follow instructions with human feedback_. NeurIPS.
  5. Hu, E. J., et al. (2021). _LoRA: Low-Rank Adaptation of Large Language Models_. ICLR.
  6. Wei, J., et al. (2022). _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_. NeurIPS.
  7. Touvron, H., et al. (2023). _LLaMA: Open and Efficient Foundation Language Models_. arXiv.
  8. Touvron, H., et al. (2023). _Llama 2: Open Foundation and Fine-Tuned Chat Models_. arXiv.
  9. Lewis, P., et al. (2020). _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_. NeurIPS.
  10. Dettmers, T., et al. (2023). _QLoRA: Efficient Finetuning of Quantized LLMs_. arXiv.
