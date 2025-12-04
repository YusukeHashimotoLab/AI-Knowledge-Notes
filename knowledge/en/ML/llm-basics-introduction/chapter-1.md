---
title: "🤖 Chapter 1: What is LLM"
chapter_title: "🤖 Chapter 1: What is LLM"
subtitle: Definition, History, and Future of Large Language Models
---

## Introduction

Since 2023, with the advent of **ChatGPT** , AI technology has rapidly permeated into general society. The technology behind ChatGPT is the **Large Language Model (LLM)**.

In this chapter, we will learn what LLMs are, the history that led to their current form, and what representative models exist.

## 1.1 Definition of LLM

### What is a Large Language Model (LLM)

A **Large Language Model (LLM)** is a deep learning model trained on massive amounts of text data that performs natural language understanding and generation.

#### 📌 Key Characteristics of LLMs

  * **Large-scale** : Contains billions to trillions of parameters
  * **Pre-training** : Trained on large amounts of internet text
  * **Versatility** : Handles various tasks (summarization, translation, question answering, etc.)
  * **Few-Shot Learning** : Can learn from a small number of examples
  * **Context Understanding** : Responses that consider long context

### Basic Structure of LLMs

Most modern LLMs are based on the **Transformer** architecture. Transformer is a revolutionary neural network structure announced by Google in 2017.
    
    
    ```mermaid
    graph TD
        A[Input Text] --> B[Tokenization]
        B --> C[Embedding Layer]
        C --> D[Transformer Layer x N]
        D --> E[Output Layer]
        E --> F[Predicted Text]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#e3f2fd
    ```

## 1.2 History of LLMs

### Evolution of Language Models

Language models have a long history, but they have developed rapidly since 2018.
    
    
    ```mermaid
    timeline
        title Evolution of LLMs
        2017 : Transformer emerges (Vaswani et al.)
        2018 : BERT (Google), GPT-1 (OpenAI)
        2019 : GPT-2 (OpenAI), T5 (Google)
        2020 : GPT-3 (175 billion parameters)
        2021 : Codex (GitHub Copilot)
        2022 : ChatGPT released (GPT-3.5 based)
        2023 : GPT-4, Claude, LLaMA, Gemini
        2024 : GPT-4 Turbo, Claude 3, LLaMA 3
    ```

### Major Milestones

#### 2017: Transformer

The **Transformer** proposed in Google's paper "Attention is All You Need" became the foundational architecture for LLMs.

  * **Innovation** : Self-Attention mechanism computes relationships between all words in a sentence in parallel
  * **Advantages** : Learning long-range dependencies, acceleration through parallel processing

#### 2018: BERT (Bidirectional Encoder Representations from Transformers)

A **bidirectional** language model announced by Google. Its ability to consider context from both directions was groundbreaking.

  * **Feature** : Masked Language Modeling (masking words and predicting them)
  * **Use Cases** : Text classification, named entity recognition, question answering, etc.

#### 2018: GPT-1 (Generative Pre-trained Transformer)

A **generative** language model announced by OpenAI. It established the pre-training + fine-tuning approach.

  * **Parameters** : 117 million
  * **Feature** : Autoregressive generation predicting the next word

#### 2020: GPT-3

The third generation of the GPT series. A dramatic increase in parameters enabled Few-Shot Learning.

  * **Parameters** : 175 billion (approximately 1500 times GPT-1)
  * **Innovation** : Can execute new tasks with just a few examples

#### 2022: ChatGPT

A chatbot based on GPT-3.5 and tuned with human feedback. It became the catalyst for the democratization of AI technology.

  * **Feature** : Tuned with RLHF (Reinforcement Learning from Human Feedback)
  * **Impact** : Reached 100 million users within 2 months of release

#### 2023: GPT-4

OpenAI's latest model (at the time of writing). It supports multimodal (text + image) capabilities.

  * **Improvements** : More accurate reasoning, long-text understanding, enhanced creativity
  * **Safety** : More robust safety features and ethical considerations

## 1.3 Representative LLM Models

### Comparison of Major LLMs

Model | Developer | Parameters | Features | Availability  
---|---|---|---|---  
**GPT-4** | OpenAI | Undisclosed (estimated 1T+) | Multimodal, high accuracy | Via API  
**Claude 3** | Anthropic | Undisclosed | Long-text understanding, safety-focused | Via API  
**Gemini** | Google | Undisclosed | Multimodal, integrated | Via API  
**LLaMA 3** | Meta | 8B, 70B, 405B | Open source, high efficiency | Fully open  
**Mistral** | Mistral AI | 7B, 8x7B | Small high-performance, MoE | Open source  
  
#### 💡 Parameter Notation

  * **B** : Billion - Example: 7B = 7 billion parameters
  * **M** : Million - Example: 340M = 340 million parameters
  * More parameters generally mean higher performance, but also increase computational costs

### Details of Each Model

#### GPT-4 (OpenAI)

  * **Release** : March 2023
  * **Strengths** : Complex reasoning, creative tasks, multimodal support
  * **Weaknesses** : High cost, API-only, knowledge cutoff exists
  * **Use Cases** : Code generation, document creation, complex problem solving

#### Claude 3 (Anthropic)

  * **Release** : March 2024
  * **Strengths** : Long-text understanding (200k+ tokens), safety, accuracy
  * **Model Variants** : Opus (highest performance), Sonnet (balanced), Haiku (fast)
  * **Use Cases** : Long-text analysis, safety-critical applications

#### Gemini (Google)

  * **Release** : December 2023
  * **Strengths** : Google services integration, multimodal, fast
  * **Model Variants** : Ultra, Pro, Nano
  * **Use Cases** : Google Workspace integration, search integration

#### LLaMA 3 (Meta)

  * **Release** : April 2024
  * **Strengths** : Open source, commercially usable, high efficiency
  * **Sizes** : 8B (small), 70B (medium), 405B (large)
  * **Use Cases** : Self-hosted deployment, customization, research

## 1.4 Tokenization Mechanism

### What are Tokens

LLMs do not process strings directly but split them into units called **tokens**. Tokens can be parts of words, entire words, or punctuation marks.

#### 🔍 Tokenization Example

**Input Text** : "ChatGPT is an amazing AI"

**Token Split** : ["Chat", "G", "PT", " is", " an", " amazing", " AI"]

→ 7 tokens

### Main Tokenization Methods

#### 1\. BPE (Byte Pair Encoding)

  * Used in GPT series
  * Repeatedly merges frequently occurring character pairs
  * Strong against unknown words (subword splitting)

#### 2\. WordPiece

  * Used in BERT
  * Improved version of BPE
  * Selects optimal splits based on likelihood

#### 3\. SentencePiece

  * Multilingual support
  * Language-independent tokenization
  * Used in LLaMA, T5, etc.

### Python Code Example for Tokenization
    
    
    # Tokenization using Hugging Face transformers
    from transformers import AutoTokenizer
    
    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Tokenize text
    text = "ChatGPT is an amazing AI"
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)
    # Example output: ['Chat', 'G', 'PT', ' is', ' an', ' amazing', ' AI']
    
    # Convert to token IDs
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    
    # Check number of tokens
    print(f"Number of tokens: {len(token_ids)}")
    

#### ⚠️ Importance of Token Count

Many LLM APIs charge based on **token count**. Additionally, models have maximum token limits (context length).

  * **GPT-3.5** : 4,096 tokens (approximately 3,000 words)
  * **GPT-4** : 8,192 tokens, or 32,768 tokens
  * **Claude 3** : 200,000 tokens (approximately 150,000 words)

## 1.5 Fundamentals of Transformer Architecture

### Basic Structure of Transformer

Transformer consists of Encoder and Decoder, but most LLMs adopt a **Decoder-Only** architecture.
    
    
    ```mermaid
    graph TD
        A[Input Tokens] --> B[Embedding + Position Encoding]
        B --> C[Multi-Head Self-Attention]
        C --> D[Add & Norm]
        D --> E[Feed-Forward Network]
        E --> F[Add & Norm]
        F --> G[Next Layer or Output]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#f3e5f5
        style G fill:#e8f5e9
    ```

### Main Components

#### 1\. Self-Attention

A mechanism where each word in a sentence learns its relationship with all other words.

  * **Query** : The word to focus on
  * **Key** : The comparison target word
  * **Value** : The information to retrieve

#### 🔍 Self-Attention Example

**Sentence** : "The cat ate the fish"

**Focusing on "ate"** :

  * "cat" → high attention (subject)
  * "fish" → high attention (object)
  * "The" → moderate attention
  * "the" → moderate attention

→ The model automatically learns grammatical structure

#### 2\. Multi-Head Attention

Computes attention from multiple different perspectives (heads) and processes them in parallel.

  * **Advantage** : Learns different types of relationships simultaneously
  * **Typical number of heads** : 8-16

#### 3\. Position Encoding

Transformer requires explicit word order information due to parallel processing.

  * **Absolute Position Encoding** : Unique vector for each position
  * **Relative Position Encoding** : Considers relative distance between words

#### 4\. Feed-Forward Network

A fully connected layer that independently transforms the representation of each token.

### Decoder-Only vs Encoder-Decoder

Architecture | Representative Models | Features | Main Use Cases  
---|---|---|---  
**Decoder-Only** | GPT-3, GPT-4, LLaMA | Autoregressive generation | Text generation, chat  
**Encoder-Only** | BERT | Bidirectional understanding | Text classification, NER  
**Encoder-Decoder** | T5, BART | Input→Output transformation | Translation, summarization  
  
## 1.6 LLM Use Cases

### Main Application Areas

#### 1\. Content Generation

  * Article and blog post creation
  * Marketing copy
  * Email draft responses
  * Creative writing (novels, poetry, scripts)

#### 2\. Code Generation and Assistance

  * GitHub Copilot (Codex-based)
  * Bug fix suggestions
  * Code review
  * Documentation generation

#### 3\. Question Answering and Customer Support

  * FAQ bots
  * Technical support
  * Internal knowledge base search

#### 4\. Translation and Summarization

  * Multilingual translation
  * Document summarization
  * Automatic meeting minutes generation

#### 5\. Educational Support

  * Learning tutors
  * Problem generation
  * Grading and feedback

### Try Using LLM: Simple Code Example
    
    
    # Text generation with GPT-2 using Hugging Face transformers
    from transformers import pipeline
    
    # Create text generation pipeline
    generator = pipeline('text-generation', model='gpt2')
    
    # Generate text with a prompt
    prompt = "When thinking about the future of artificial intelligence"
    result = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    
    print(result[0]['generated_text'])
    

#### 💡 Parameter Explanation

  * **max_length** : Maximum number of tokens to generate
  * **num_return_sequences** : Number of candidates to generate
  * **temperature** : Randomness (0=deterministic, 1=creative)

## 1.7 Limitations and Challenges of LLMs

### Main Challenges

#### 1\. Hallucination

LLMs can generate non-existent information in a plausible manner.

#### ⚠️ Example of Hallucination

**Question** : "Who won the 2024 Nobel Prize in Physics?"

**Example of Incorrect Answer** : "Dr. Taro Yamada won for his research in quantum computing"

→ Models cannot say "I don't know" and may generate plausible lies

#### 2\. Bias and Fairness

  * Learns social biases contained in training data
  * Prejudices related to gender, race, age, etc.
  * Need for ethical considerations

#### 3\. Knowledge Cutoff

  * Does not know information after the training data cutoff date
  * Example: GPT-4 (2023 version) does not know events after April 2023

#### 4\. Computational Cost and Energy

  * Training costs millions to tens of millions of dollars
  * Inference also requires high computational resources
  * Environmental impact

#### 5\. Privacy and Security

  * Risk of information leakage from training data
  * Potential for misuse (phishing, disinformation)
  * Copyright issues

### Countermeasures and Mitigation Strategies

  * **RLHF (Reinforcement Learning from Human Feedback)** : Adopted in ChatGPT, etc.
  * **RAG (Retrieval-Augmented Generation)** : Integration with external knowledge bases
  * **Fact-checking Mechanisms** : Verification of generated content
  * **Transparency and Documentation** : Explicitly stating model limitations

## 1.8 Future of LLMs

### Future Development Directions

#### 1\. Multimodal AI

Models that integrate understanding and generation of not only text but also images, audio, and video.

  * GPT-4V (Vision): Image understanding
  * Gemini: Natively multimodal

#### 2\. More Efficient Models

Development of smaller yet high-performance models.

  * Mistral 7B: High performance with 7 billion parameters
  * Quantization, pruning, distillation

#### 3\. Agent-type AI

AI that can use tools, make plans, and take actions.

  * AutoGPT, BabyAGI
  * Function Calling

#### 4\. Personalization

AI assistants optimized for individuals.

  * Learning user preferences
  * Custom GPTs

#### 5\. Open Source Movement

Trend toward more models being released as open source.

  * LLaMA, Mistral, Falcon, etc.
  * Democratization of research and development

## Summary

In this chapter, we learned the fundamentals of Large Language Models (LLMs).

#### 📌 Key Points

  * LLMs are large-scale neural networks based on the **Transformer** architecture
  * Rapid development from Transformer's emergence in 2017 to ChatGPT's popularization in 2023
  * Various models exist including **GPT-4, Claude, Gemini, LLaMA**
  * **Tokenization** converts strings to numbers for processing
  * **Self-Attention** enables context understanding
  * Various use cases exist but challenges like **hallucination** also persist
  * Future development toward multimodal, efficiency, and agent-type AI

## Exercises

#### 📝 Exercise 1: Basic Knowledge Check

**Question** : Answer the following questions.

  1. What does "large-scale" in LLM refer to?
  2. List two main advantages of the Transformer architecture.
  3. Explain the difference between Decoder-Only and Encoder-Only models.

#### 📝 Exercise 2: Tokenization Practice

**Task** : Run the following code and compare token counts for different texts.
    
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    texts = [
        "Hello",
        "Bonjour",
        "Artificial Intelligence is amazing",
        "人工知能は素晴らしい"
    ]
    
    for text in texts:
        tokens = tokenizer.encode(text)
        print(f"'{text}' → {len(tokens)} tokens")
    

**Analysis** : Are there differences in token counts between different languages? Consider the reasons.

#### 📝 Exercise 3: Model Comparison

**Task** : Choose one from GPT-4, Claude, or LLaMA and research the following:

  * Developer and development background
  * Main features and strengths
  * Usage methods (API, open source, etc.)
  * Representative use cases

**Advanced** : Read the official documentation of the chosen model and summarize technical details.

## Next Chapter

In the next chapter, we will study the **Transformer architecture** , the core technology of LLMs, in detail. You will understand the mechanisms of Self-Attention, Multi-Head Attention, position encoding, etc., and experience them with working code.

[← Series Overview](<./index.html>) [Chapter 2: Transformer Architecture (Coming Soon)](<./index.html>)
