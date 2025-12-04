---
title: "Chapter 5: Large Language Models (LLMs)"
chapter_title: "Chapter 5: Large Language Models (LLMs)"
subtitle: From LLM Scaling Laws to Practical Prompt Engineering
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
---

This chapter covers Large Language Models (LLMs). You will learn Compare major LLM architectures including GPT and prompt engineering techniques such as Zero-shot.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand scaling laws and performance characteristics of Large Language Models (LLMs)
  * ✅ Compare major LLM architectures including GPT, LLaMA, Claude, and Gemini
  * ✅ Implement prompt engineering techniques such as Zero-shot, Few-shot, and Chain-of-Thought
  * ✅ Understand the mechanisms and effective utilization of In-Context Learning
  * ✅ Understand model improvement methods using human feedback through RLHF
  * ✅ Build practical LLM applications and integrate APIs

* * *

## 5.1 LLM Scaling Laws

### What are Large Language Models

**Large Language Models (LLMs)** are massive Transformer-based models pre-trained on enormous text datasets. With tens to hundreds of billions of parameters, they can execute various natural language processing tasks with high accuracy.
    
    
    ```mermaid
    graph TB
        A[Evolution of Language Models] --> B[Small Models~100M params2018-2019]
        A --> C[Medium Models1B-10B params2019-2020]
        A --> D[Large Models100B+ params2020-present]
    
        B --> B1[BERT Base110M]
        B --> B2[GPT-2117M-1.5B]
    
        C --> C1[GPT-3175B]
        C --> C2[T511B]
    
        D --> D1[GPT-4~1.7T estimated]
        D --> D2[Claude 3undisclosed]
        D --> D3[Gemini Ultraundisclosed]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff9c4
        style D fill:#c8e6c9
    ```

### Scaling Laws

The **scaling laws** published by OpenAI in 2020 quantify how model performance scales with respect to the number of parameters, data amount, and compute.

#### Basic Scaling Laws

The model loss $L$ is determined by three factors:

$$ L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C} $$

Where:

  * $N$: Number of model parameters (non-embedding)
  * $D$: Number of training data tokens
  * $C$: Compute (FLOPs)
  * $N_c, D_c, C_c$: Scaling constants
  * $\alpha_N \approx 0.076, \alpha_D \approx 0.095, \alpha_C \approx 0.050$

Scaling Factor | Impact | Practical Meaning  
---|---|---  
**Model Size (N)** | 10× reduces loss to ~0.95× | Larger models perform better  
**Data Amount (D)** | 10× reduces loss to ~0.93× | Data importance is highest  
**Compute (C)** | 10× reduces loss to ~0.97× | Efficient computation is key  
  
> **Chinchilla Paper Discovery (2022)** : DeepMind's research revealed that many LLMs are "over-parameterized" and that training smaller models on more data with the same compute budget is more efficient. The optimal ratio is **data tokens ≈ 20 × parameters**.

### Emergent Abilities

Once LLMs exceed a certain scale, capabilities that were not explicitly trained suddenly **emerge**. These are called **emergent abilities**.
    
    
    ```mermaid
    graph LR
        A[Model Size Increase] --> B[~1B parameters]
        B --> C[Basic Text Generation]
    
        A --> D[~10B parameters]
        D --> E[Few-shot LearningSimple Reasoning]
    
        A --> F[~100B parameters]
        F --> G[Chain-of-ThoughtComplex ReasoningInstruction Following]
    
        style A fill:#e3f2fd
        style G fill:#c8e6c9
    ```

Emergent Ability | Emergence Scale | Description  
---|---|---  
**In-Context Learning** | ~10B+ | Learning tasks from examples  
**Chain-of-Thought** | ~100B+ | Step-by-step reasoning capability  
**Instruction Following** | ~10B+ (after RLHF) | Understanding natural language instructions  
**Multilingual Capability** | ~10B+ | Transfer to unlearned languages  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def scaling_law_loss(N, D, C, N_c=8.8e13, D_c=5.4e13, C_c=1.3e13,
                         alpha_N=0.076, alpha_D=0.095, alpha_C=0.050):
        """
        Calculate loss based on scaling laws
    
        Args:
            N: Number of parameters
            D: Number of data tokens
            C: Compute (FLOPs)
            Others: Scaling constants
    
        Returns:
            Predicted loss value
        """
        loss_N = (N_c / N) ** alpha_N
        loss_D = (D_c / D) ** alpha_D
        loss_C = (C_c / C) ** alpha_C
        return loss_N + loss_D + loss_C
    
    # Visualize the impact of model size
    param_counts = np.logspace(6, 12, 50)  # 1M to 1T parameters
    data_tokens = 1e12  # 1T tokens fixed
    compute = 1e21  # fixed
    
    losses = [scaling_law_loss(N, data_tokens, compute) for N in param_counts]
    
    plt.figure(figsize=(12, 5))
    
    # Parameters vs Loss
    plt.subplot(1, 2, 1)
    plt.loglog(param_counts, losses, linewidth=2, color='#7b2cbf')
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Scaling Law: Model Size vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot major models
    models = {
        'GPT-2': (1.5e9, scaling_law_loss(1.5e9, data_tokens, compute)),
        'GPT-3': (175e9, scaling_law_loss(175e9, data_tokens, compute)),
        'GPT-4 (estimated)': (1.7e12, scaling_law_loss(1.7e12, data_tokens, compute)),
    }
    
    for name, (params, loss) in models.items():
        plt.scatter(params, loss, s=100, zorder=5)
        plt.annotate(name, (params, loss), xytext=(10, 10),
                    textcoords='offset points', fontsize=9)
    
    # Impact of data amount
    plt.subplot(1, 2, 2)
    data_amounts = np.logspace(9, 13, 50)
    model_size = 100e9  # 100B parameters fixed
    
    losses_data = [scaling_law_loss(model_size, D, compute) for D in data_amounts]
    plt.loglog(data_amounts, losses_data, linewidth=2, color='#3182ce')
    plt.xlabel('Training Data Tokens', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Scaling Law: Data Amount vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark Chinchilla optimal point
    optimal_data = 20 * model_size  # 20x rule
    optimal_loss = scaling_law_loss(model_size, optimal_data, compute)
    plt.scatter(optimal_data, optimal_loss, s=150, c='red', marker='*',
               zorder=5, label='Chinchilla Optimal Point')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Practical example: Estimating required computational resources
    def estimate_training_cost(params, tokens, efficiency=0.5):
        """
        Estimate training cost (FLOPs)
    
        Args:
            params: Number of parameters
            tokens: Number of training tokens
            efficiency: Hardware efficiency (0-1)
    
        Returns:
            Required FLOPs, GPU time estimate
        """
        # Approximately 6 × params FLOPs per token
        total_flops = 6 * params * tokens
    
        # A100 GPU: ~312 TFLOPS (FP16)
        a100_flops = 312e12 * efficiency
        gpu_hours = total_flops / a100_flops / 3600
    
        return total_flops, gpu_hours
    
    # Cost estimation for GPT-3 class model
    params_gpt3 = 175e9
    tokens_gpt3 = 300e9
    
    total_flops, gpu_hours = estimate_training_cost(params_gpt3, tokens_gpt3)
    
    print(f"\n=== GPT-3 Scale Model Training Cost Estimate ===")
    print(f"Parameters: {params_gpt3/1e9:.1f}B")
    print(f"Training Tokens: {tokens_gpt3/1e9:.1f}B")
    print(f"Total Compute: {total_flops:.2e} FLOPs")
    print(f"A100 GPU Hours: {gpu_hours:,.0f} hours ({gpu_hours/24:,.0f} days)")
    print(f"GPUs needed (30-day completion): {int(np.ceil(gpu_hours / (24 * 30)))} units")
    

* * *

## 5.2 Representative LLMs

### GPT Series (OpenAI)

#### GPT-3 (2020)

**GPT-3** (Generative Pre-trained Transformer 3) is an autoregressive language model with 175B parameters that demonstrated the effectiveness of Few-shot Learning.

Feature | Details  
---|---  
**Parameters** | 175B (largest version)  
**Architecture** | Decoder-only Transformer, 96 layers, 12,288 dimensions  
**Training Data** | Common Crawl, WebText, Books, Wikipedia, etc. ~300B tokens  
**Context Length** | 2,048 tokens  
**Innovation** | Demonstrated Few-shot Learning, prompt-based versatility  
  
#### GPT-4 (2023)

**GPT-4** is a state-of-the-art multimodal (text + image) model. Details are not disclosed, but it's estimated to have 1.7 trillion parameters.

Capability | GPT-3.5 | GPT-4  
---|---|---  
**US Bar Exam** | Bottom 10% | Top 10%  
**Math Olympiad** | Failed | Top 500 equivalent  
**Coding Ability** | Basic implementation | Complex algorithm design  
**Multimodal** | Text only | Text + image understanding  
  
### LLaMA Series (Meta)

**LLaMA** (Large Language Model Meta AI) is an efficient LLM family released as open source by Meta.

#### LLaMA 2 Features

  * **Model Sizes** : 7B, 13B, 70B in 3 variations
  * **Training Data** : 2 trillion tokens (public data only)
  * **Context Length** : 4,096 tokens
  * **License** : Commercial use allowed (with conditions)
  * **Optimization** : Efficient design based on Chinchilla scaling laws

    
    
    ```mermaid
    graph TB
        A[LLaMA 2 Architecture] --> B[Pre-normalizationRMSNorm]
        A --> C[SwiGLU activationAdopted from PaLM]
        A --> D[Rotary PositionalEmbedding RoPE]
        A --> E[Grouped QueryAttention GQA]
    
        B --> F[Improved training stability]
        C --> G[Better performance]
        D --> H[Long context support]
        E --> I[Faster inference]
    
        style A fill:#e3f2fd
        style F fill:#c8e6c9
        style G fill:#c8e6c9
        style H fill:#c8e6c9
        style I fill:#c8e6c9
    ```

### Claude (Anthropic)

**Claude** is a safety-focused LLM developed by Anthropic using the "Constitutional AI" approach.

Model | Features | Context Length  
---|---|---  
**Claude 3 Opus** | Highest performance, complex reasoning | 200K tokens  
**Claude 3 Sonnet** | Balanced, fast response | 200K tokens  
**Claude 3 Haiku** | Lightweight, fast, cost-efficient | 200K tokens  
  
> **Constitutional AI** : In addition to human feedback (RLHF), this training method uses a "constitution" where AI performs self-criticism and self-improvement. It reduces harmful outputs and generates safer, more helpful responses.

### Gemini (Google)

**Gemini** is a multimodal-native LLM developed by Google that processes text, images, audio, and video in an integrated manner.

  * **Gemini Ultra** : Highest performance, complex task support
  * **Gemini Pro** : Optimized for general purposes
  * **Gemini Nano** : Lightweight version that runs on-device

    
    
    # Major LLM comparison implementation example (via API)
    import os
    from typing import List, Dict
    import time
    
    class LLMComparison:
        """
        Compare multiple LLM APIs using a unified interface
        """
    
        def __init__(self):
            """Get each API key from environment variables"""
            self.openai_key = os.getenv('OPENAI_API_KEY')
            self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            self.google_key = os.getenv('GOOGLE_API_KEY')
    
        def query_gpt4(self, prompt: str, max_tokens: int = 500) -> Dict:
            """Send query to GPT-4"""
            try:
                import openai
                openai.api_key = self.openai_key
    
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                latency = time.time() - start_time
    
                return {
                    'model': 'GPT-4',
                    'response': response.choices[0].message.content,
                    'tokens': response.usage.total_tokens,
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'GPT-4', 'error': str(e)}
    
        def query_claude(self, prompt: str, max_tokens: int = 500) -> Dict:
            """Send query to Claude 3"""
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
    
                start_time = time.time()
                message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                latency = time.time() - start_time
    
                return {
                    'model': 'Claude 3 Opus',
                    'response': message.content[0].text,
                    'tokens': message.usage.input_tokens + message.usage.output_tokens,
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'Claude 3 Opus', 'error': str(e)}
    
        def query_gemini(self, prompt: str, max_tokens: int = 500) -> Dict:
            """Send query to Gemini Pro"""
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.google_key)
                model = genai.GenerativeModel('gemini-pro')
    
                start_time = time.time()
                response = model.generate_content(prompt)
                latency = time.time() - start_time
    
                return {
                    'model': 'Gemini Pro',
                    'response': response.text,
                    'tokens': 'N/A',  # Gemini API doesn't return detailed token count
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'Gemini Pro', 'error': str(e)}
    
        def compare_all(self, prompt: str, max_tokens: int = 500) -> List[Dict]:
            """
            Send the same prompt to all models for comparison
    
            Args:
                prompt: Input prompt
                max_tokens: Maximum generation tokens
    
            Returns:
                List of results from each model
            """
            results = []
    
            print(f"Prompt: {prompt}\n")
            print("=" * 80)
    
            # GPT-4
            print("Querying GPT-4...")
            gpt4_result = self.query_gpt4(prompt, max_tokens)
            results.append(gpt4_result)
            self._print_result(gpt4_result)
    
            # Claude 3
            print("\nQuerying Claude 3...")
            claude_result = self.query_claude(prompt, max_tokens)
            results.append(claude_result)
            self._print_result(claude_result)
    
            # Gemini
            print("\nQuerying Gemini Pro...")
            gemini_result = self.query_gemini(prompt, max_tokens)
            results.append(gemini_result)
            self._print_result(gemini_result)
    
            return results
    
        def _print_result(self, result: Dict):
            """Format and print result"""
            if 'error' in result:
                print(f"❌ {result['model']}: Error - {result['error']}")
            else:
                print(f"✅ {result['model']}:")
                print(f"   Response: {result['response'][:200]}...")
                print(f"   Tokens: {result['tokens']}")
                print(f"   Latency: {result['latency']:.2f}s")
    
    # Usage example
    if __name__ == "__main__":
        # Note: Requires each API key for execution
        comparator = LLMComparison()
    
        # Simple comparison test
        test_prompt = """
        Solve the following problem step by step:
    
        Problem: A company's revenue increases by 20% each year.
        If the current revenue is 100 million yen, what will the revenue be after 5 years?
        Show the calculation process.
        """
    
        results = comparator.compare_all(test_prompt, max_tokens=300)
    
        # Performance comparison
        print("\n" + "=" * 80)
        print("Performance Comparison:")
        for result in results:
            if 'error' not in result:
                print(f"{result['model']:20} - Latency: {result['latency']:.2f}s")
    

* * *

## 5.3 Prompt Engineering

### What is Prompt Engineering

**Prompt engineering** is the technique of designing inputs to elicit desired outputs from LLMs. With appropriate prompts, model performance can be dramatically improved without retraining.
    
    
    ```mermaid
    graph LR
        A[Prompt Techniques] --> B[Zero-shot]
        A --> C[Few-shot]
        A --> D[Chain-of-Thought]
        A --> E[Self-Consistency]
    
        B --> B1[Execute with instructions only]
        C --> C1[Learn from examples]
        D --> D1[Step-by-step reasoning]
        E --> E1[Integrate multiple paths]
    
        style A fill:#e3f2fd
        style D fill:#c8e6c9
    ```

### Zero-shot Learning

**Zero-shot Learning** is a method of executing tasks with instructions only, without examples. It became possible through emergent abilities of large-scale models.
    
    
    class ZeroShotPromptEngine:
        """
        Design and execute Zero-shot prompts
        """
    
        @staticmethod
        def sentiment_analysis(text: str) -> str:
            """Zero-shot prompt for sentiment analysis"""
            prompt = f"""
    Classify the sentiment of the following text as "Positive", "Negative", or "Neutral".
    Return only the classification result.
    
    Text: {text}
    
    Classification:"""
            return prompt
    
        @staticmethod
        def text_summarization(text: str, max_words: int = 50) -> str:
            """Zero-shot prompt for summarization"""
            prompt = f"""
    Summarize the following text in {max_words} words or less.
    Concisely capture the key points.
    
    Text:
    {text}
    
    Summary:"""
            return prompt
    
        @staticmethod
        def question_answering(context: str, question: str) -> str:
            """Zero-shot prompt for question answering"""
            prompt = f"""
    Answer the question based on the following context.
    If the information is not in the context, answer "Insufficient information".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
            return prompt
    
        @staticmethod
        def language_translation(text: str, target_lang: str = "English") -> str:
            """Zero-shot prompt for translation"""
            prompt = f"""
    Translate the following text into {target_lang}.
    Aim for natural and accurate translation.
    
    Text: {text}
    
    Translation:"""
            return prompt
    
    # Usage example
    engine = ZeroShotPromptEngine()
    
    # Sentiment analysis
    text1 = "This product exceeded my expectations. I'm really glad I purchased it."
    prompt1 = engine.sentiment_analysis(text1)
    print("=== Zero-shot Sentiment Analysis ===")
    print(prompt1)
    print()
    
    # Summarization
    text2 = """
    The development of artificial intelligence (AI) is bringing innovation to various fields.
    Particularly in natural language processing, the emergence of large language models (LLMs)
    has achieved near-human performance in tasks such as machine translation, text generation, and question answering.
    In the future, AI will be utilized in even more areas including healthcare, education, and business.
    """
    prompt2 = engine.text_summarization(text2, max_words=30)
    print("=== Zero-shot Summarization ===")
    print(prompt2)
    

### Few-shot Learning

**Few-shot Learning** is a method where the model learns task patterns by presenting a small number of examples (typically 1-10).
    
    
    class FewShotPromptEngine:
        """
        Design and execute Few-shot prompts
        """
    
        @staticmethod
        def sentiment_analysis(text: str, num_examples: int = 3) -> str:
            """Few-shot prompt for sentiment analysis"""
            # Example data
            examples = [
                ("This movie was wonderful! I was moved.", "Positive"),
                ("The food was cold and the service was bad.", "Negative"),
                ("It's an ordinary product. Nothing particularly good or bad.", "Neutral"),
            ]
    
            # Build prompt
            prompt = "Classify the sentiment of the text based on the following examples.\n\n"
    
            for i, (example_text, label) in enumerate(examples[:num_examples], 1):
                prompt += f"Example {i}:\nText: {example_text}\nClassification: {label}\n\n"
    
            prompt += f"Text: {text}\nClassification:"
            return prompt
    
        @staticmethod
        def entity_extraction(text: str) -> str:
            """Few-shot prompt for named entity extraction"""
            prompt = """
    Extract person names, organization names, and locations from the text based on the following examples.
    
    Example 1:
    Text: Taro Tanaka is researching machine learning at the University of Tokyo.
    Extraction: Person=Taro Tanaka, Organization=University of Tokyo, Location=None
    
    Example 2:
    Text: Apple CEO Tim Cook gave a speech in Silicon Valley.
    Extraction: Person=Tim Cook, Organization=Apple, Location=Silicon Valley
    
    Example 3:
    Text: Google and Microsoft announced new AI technologies.
    Extraction: Person=None, Organization=Google, Microsoft, Location=None
    
    Text: {text}
    Extraction:"""
            return prompt.format(text=text)
    
        @staticmethod
        def code_generation(task_description: str) -> str:
            """Few-shot prompt for code generation"""
            prompt = """
    Generate a Python function to perform the task based on the following examples.
    
    Example 1:
    Task: Double list elements
    Code:
    def double_elements(lst):
        return [x * 2 for x in lst]
    
    Example 2:
    Task: Reverse a string
    Code:
    def reverse_string(s):
        return s[::-1]
    
    Example 3:
    Task: Calculate average of a list
    Code:
    def calculate_average(lst):
        return sum(lst) / len(lst) if lst else 0
    
    Task: {task}
    Code:"""
            return prompt.format(task=task_description)
    
        @staticmethod
        def analogical_reasoning(question: str) -> str:
            """Few-shot prompt for analogical reasoning"""
            prompt = """
    Understand the following patterns and answer the question.
    
    Example 1: Tokyo:Japan = Paris:?
    Answer: France
    Reason: Just as Tokyo is the capital of Japan, Paris is the capital of France.
    
    Example 2: Doctor:Hospital = Teacher:?
    Answer: School
    Reason: Just as doctors work at hospitals, teachers work at schools.
    
    Example 3: Dog:Mammal = Hawk:?
    Answer: Bird
    Reason: Just as dogs belong to mammals, hawks belong to birds.
    
    Question: {question}
    Answer:"""
            return prompt.format(question=question)
    
    # Usage example
    few_shot_engine = FewShotPromptEngine()
    
    # Few-shot sentiment analysis
    test_text = "It wasn't as good as I expected, but it's okay."
    prompt = few_shot_engine.sentiment_analysis(test_text)
    print("=== Few-shot Sentiment Analysis ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Few-shot named entity extraction
    test_entity = "An NHK reporter interviewed Hanako Yamada in New York."
    prompt = few_shot_engine.entity_extraction(test_entity)
    print("=== Few-shot Named Entity Extraction ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Few-shot analogical reasoning
    test_analogy = "Book:Author = Movie:?"
    prompt = few_shot_engine.analogical_reasoning(test_analogy)
    print("=== Few-shot Analogical Reasoning ===")
    print(prompt)
    

### Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** is a method that improves accuracy on complex problems by having the model generate step-by-step reasoning processes.

$$ \text{Accuracy}_{\text{CoT}} \approx \text{Accuracy}_{\text{standard}} + \Delta_{\text{reasoning}} $$

Where $\Delta_{\text{reasoning}}$ is the accuracy improvement from reasoning, which becomes larger for more complex problems.
    
    
    class ChainOfThoughtEngine:
        """
        Chain-of-Thought (CoT) prompt engineering
        """
    
        @staticmethod
        def math_problem_basic(problem: str) -> str:
            """Basic CoT for math problems"""
            prompt = f"""
    Solve the following problem step by step.
    Explain what you are calculating at each step.
    
    Problem: {problem}
    
    Solution:
    Step 1:"""
            return prompt
    
        @staticmethod
        def math_problem_with_examples(problem: str) -> str:
            """Few-shot CoT example"""
            prompt = """
    Solve the problem step by step based on the following examples.
    
    Example 1:
    Problem: There are 15 apples and 23 oranges. How many fruits are there in total?
    Solution:
    Step 1: Check number of apples → 15
    Step 2: Check number of oranges → 23
    Step 3: Calculate total → 15 + 23 = 38
    Answer: 38 fruits
    
    Example 2:
    Problem: I bought 3 books at 500 yen each and 5 pens at 120 yen each. What is the total amount?
    Solution:
    Step 1: Calculate total for books → 500 yen × 3 books = 1,500 yen
    Step 2: Calculate total for pens → 120 yen × 5 pens = 600 yen
    Step 3: Calculate grand total → 1,500 yen + 600 yen = 2,100 yen
    Answer: 2,100 yen
    
    Problem: {problem}
    Solution:
    Step 1:"""
            return prompt.format(problem=problem)
    
        @staticmethod
        def logical_reasoning(scenario: str, question: str) -> str:
            """CoT for logical reasoning"""
            prompt = f"""
    Analyze the following situation step by step and reach a logical conclusion.
    
    Situation: {scenario}
    
    Question: {question}
    
    Analysis:
    Observation 1:"""
            return prompt
    
        @staticmethod
        def self_consistency_cot(problem: str, num_paths: int = 3) -> str:
            """
            Self-Consistency CoT: Generate multiple reasoning paths
            and select the most consistent answer
            """
            prompt = f"""
    Solve the following problem using {num_paths} different approaches.
    Reason step by step for each approach, then choose the most certain answer.
    
    Problem: {problem}
    
    Approach 1:
    """
            return prompt
    
    # Practical example: Complex math problem
    cot_engine = ChainOfThoughtEngine()
    
    problem1 = """
    A store's revenue was 1 million yen in year 1.
    Year 2 was 20% increase from previous year, year 3 was 15% decrease from previous year, year 4 was 25% increase from previous year.
    What is the revenue in year 4 in ten thousand yen?
    """
    
    print("=== Chain-of-Thought: Math Problem ===")
    prompt1 = cot_engine.math_problem_with_examples(problem1)
    print(prompt1)
    print("\n" + "="*80 + "\n")
    
    # Logical reasoning example
    scenario = """
    There are conference rooms A, B, and C.
    - Mr. Tanaka is not in conference room A
    - Mr. Sato is in conference room B
    - No one is in conference room C
    - Besides Mr. Tanaka and Mr. Sato, there is Mr. Yamada
    """
    
    question = "Which conference room is Mr. Yamada in?"
    
    print("=== Chain-of-Thought: Logical Reasoning ===")
    prompt2 = cot_engine.logical_reasoning(scenario, question)
    print(prompt2)
    print("\n" + "="*80 + "\n")
    
    # Self-Consistency CoT
    problem2 = """
    There are 5 red balls and 3 blue balls in a bag.
    When drawing 2 balls simultaneously, what is the probability that both are red?
    """
    
    print("=== Self-Consistency CoT ===")
    prompt3 = cot_engine.self_consistency_cot(problem2, num_paths=3)
    print(prompt3)
    

> **Effect of CoT** : In experiments with Google's PaLM model, the accuracy rate for arithmetic problems improved from 34% with standard prompts to 79% with CoT prompts. Particularly large improvements were seen in problems requiring multi-step reasoning.

### Prompt Template Design
    
    
    from typing import Dict, List, Optional
    from dataclasses import dataclass
    
    @dataclass
    class PromptTemplate:
        """Structured management of prompt templates"""
        name: str
        instruction: str
        examples: Optional[List[Dict[str, str]]] = None
        output_format: Optional[str] = None
    
        def render(self, **kwargs) -> str:
            """Convert template to actual prompt"""
            prompt = f"{self.instruction}\n\n"
    
            # Add Few-shot examples
            if self.examples:
                prompt += "Examples:\n"
                for i, example in enumerate(self.examples, 1):
                    prompt += f"\nExample {i}:\n"
                    for key, value in example.items():
                        prompt += f"{key}: {value}\n"
    
            # Add output format
            if self.output_format:
                prompt += f"\nOutput format:\n{self.output_format}\n"
    
            # Insert variables
            prompt += "\nInput:\n"
            for key, value in kwargs.items():
                prompt += f"{key}: {value}\n"
    
            prompt += "\nOutput:"
            return prompt
    
    class PromptLibrary:
        """Library of reusable prompt templates"""
    
        @staticmethod
        def get_classification_template() -> PromptTemplate:
            """Template for classification tasks"""
            return PromptTemplate(
                name="classification",
                instruction="Classify the following text into the specified category.",
                examples=[
                    {
                        "Text": "A new smartphone has been released.",
                        "Category": "Technology"
                    },
                    {
                        "Text": "Stock prices are soaring.",
                        "Category": "Business"
                    }
                ],
                output_format="Return only the category name."
            )
    
        @staticmethod
        def get_extraction_template() -> PromptTemplate:
            """Template for information extraction"""
            return PromptTemplate(
                name="extraction",
                instruction="Extract the specified information from the text.",
                output_format="Return in JSON format: {\"item1\": \"value1\", \"item2\": \"value2\"}"
            )
    
        @staticmethod
        def get_generation_template() -> PromptTemplate:
            """Template for generation tasks"""
            return PromptTemplate(
                name="generation",
                instruction="Generate creative content based on the following conditions.",
                output_format="Output in natural, readable text."
            )
    
        @staticmethod
        def get_reasoning_template() -> PromptTemplate:
            """Template for reasoning tasks"""
            return PromptTemplate(
                name="reasoning",
                instruction="""
    Analyze the following problem step by step and solve it logically.
    Clearly show the reasoning process at each step.
                """,
                output_format="""
    Step 1: [Initial analysis]
    Step 2: [Next reasoning]
    ...
    Conclusion: [Final answer]
                """
            )
    
    # Usage example
    library = PromptLibrary()
    
    # Classification task
    classification_template = library.get_classification_template()
    prompt = classification_template.render(
        Text="AI research is accelerating.",
        Category="Technology, Business, Politics, Sports, Entertainment"
    )
    print("=== Classification Prompt ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Information extraction task
    extraction_template = library.get_extraction_template()
    prompt = extraction_template.render(
        Text="Taro Tanaka (35 years old) is the CTO of ABC Corporation and lives in Tokyo.",
        Extraction_items="Name, Age, Company, Position, Location"
    )
    print("=== Extraction Prompt ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Reasoning task
    reasoning_template = library.get_reasoning_template()
    prompt = reasoning_template.render(
        Problem="There are three boxes A, B, C, and only one contains treasure. Box A says 'The treasure is here', Box B says 'The treasure is not in A', Box C says 'The treasure is in B'. If only one is telling the truth, which box has the treasure?"
    )
    print("=== Reasoning Prompt ===")
    print(prompt)
    

* * *

## 5.4 In-Context Learning

### Mechanism of In-Context Learning

**In-Context Learning (ICL)** is the ability of LLMs to learn directly from examples in the prompt and execute new tasks without parameter updates.
    
    
    ```mermaid
    graph TB
        A[In-Context Learning] --> B[Input Prompt]
        B --> C[Task Instruction]
        B --> D[Few-shot Examples]
        B --> E[Query]
    
        C --> F[Transformer'sSelf-Attention]
        D --> F
        E --> F
    
        F --> G[Pattern Learningwithin Context]
        G --> H[Output Generation]
    
        style A fill:#e3f2fd
        style F fill:#fff3e0
        style G fill:#c8e6c9
    ```

#### Why ICL Works

Recent research (2023) revealed that ICL operates through the following mechanisms:

  1. **Activation of latent concepts** : Knowledge acquired during pre-training is activated by examples
  2. **Formation of task vectors** : Patterns extracted from examples are retained as internal representations
  3. **Analogy-based reasoning** : New inputs are processed by analogy with examples

$$ P(y|x, \text{examples}) \approx \sum_{i=1}^{k} \alpha_i \cdot P(y|x, \text{example}_i) $$

Where $\alpha_i$ is the relevance weight of each example.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from typing import List, Tuple, Dict
    
    class InContextLearningSimulator:
        """
        Simulate the operation of In-Context Learning
        (Simplified mechanism reproduction)
        """
    
        def __init__(self, embedding_dim: int = 128):
            """
            Args:
                embedding_dim: Dimension of embedding vectors
            """
            self.embedding_dim = embedding_dim
            self.task_vector = None
    
        def create_example_embedding(self, input_text: str, output_text: str) -> np.ndarray:
            """
            Create example embedding from input-output pair
            (Actually processed by Transformer, but simplified here)
            """
            # Simplified: Convert string to embedding via hash
            input_hash = hash(input_text) % 10000
            output_hash = hash(output_text) % 10000
    
            np.random.seed(input_hash)
            input_emb = np.random.randn(self.embedding_dim)
    
            np.random.seed(output_hash)
            output_emb = np.random.randn(self.embedding_dim)
    
            # Task vector: difference between output and input
            task_emb = output_emb - input_emb
            return task_emb / (np.linalg.norm(task_emb) + 1e-8)
    
        def learn_from_examples(self, examples: List[Tuple[str, str]]) -> None:
            """
            Learn task vector from Few-shot examples
    
            Args:
                examples: [(input1, output1), (input2, output2), ...]
            """
            task_vectors = []
    
            for input_text, output_text in examples:
                task_vec = self.create_example_embedding(input_text, output_text)
                task_vectors.append(task_vec)
    
            # Acquire task representation as average of multiple examples
            self.task_vector = np.mean(task_vectors, axis=0)
            self.task_vector /= (np.linalg.norm(self.task_vector) + 1e-8)
    
        def predict(self, query: str, candidates: List[str]) -> Dict[str, float]:
            """
            Predict based on learned task vector
    
            Args:
                query: Input query
                candidates: List of candidate outputs
    
            Returns:
                Score dictionary for each candidate
            """
            if self.task_vector is None:
                raise ValueError("Call learn_from_examples() first")
    
            # Query embedding
            query_hash = hash(query) % 10000
            np.random.seed(query_hash)
            query_emb = np.random.randn(self.embedding_dim)
            query_emb /= (np.linalg.norm(query_emb) + 1e-8)
    
            # Calculate score for each candidate
            scores = {}
            for candidate in candidates:
                candidate_hash = hash(candidate) % 10000
                np.random.seed(candidate_hash)
                candidate_emb = np.random.randn(self.embedding_dim)
                candidate_emb /= (np.linalg.norm(candidate_emb) + 1e-8)
    
                # Consistency with task vector
                predicted_output = query_emb + self.task_vector
                similarity = np.dot(predicted_output, candidate_emb)
                scores[candidate] = float(similarity)
    
            # Normalize scores
            total = sum(np.exp(s) for s in scores.values())
            scores = {k: np.exp(v)/total for k, v in scores.items()}
    
            return scores
    
    # Usage example: Sentiment analysis task
    print("=== In-Context Learning Simulation ===\n")
    
    simulator = InContextLearningSimulator(embedding_dim=128)
    
    # Few-shot examples
    examples = [
        ("This movie was wonderful!", "Positive"),
        ("Worst experience ever. Never going back.", "Negative"),
        ("It's ordinary. Nothing particularly memorable.", "Neutral"),
        ("Quality exceeded expectations, very satisfied!", "Positive"),
        ("Disappointed with terrible service.", "Negative"),
    ]
    
    # Learn task vector
    simulator.learn_from_examples(examples)
    print("✅ Learning from Few-shot examples complete\n")
    
    # Predict on new inputs
    test_queries = [
        "Excellent product. Highly recommend.",
        "It was disappointing.",
        "Mediocre performance."
    ]
    
    candidates = ["Positive", "Negative", "Neutral"]
    
    for query in test_queries:
        print(f"Query: {query}")
        scores = simulator.predict(query, candidates)
    
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
        for label, score in sorted_scores:
            print(f"  {label}: {score:.4f} {'★' * int(score * 10)}")
        print(f"  → Prediction: {sorted_scores[0][0]}\n")
    

### Effective Utilization of ICL

#### Example Selection Strategies

Strategy | Method | Advantage  
---|---|---  
**Random Selection** | Randomly select from training data | Simple, less bias  
**Similarity-based** | Select examples similar to query | Higher task performance  
**Diversity-focused** | Include diverse examples | Improved generalization  
**Difficulty Adjustment** | Order from easy to hard | Better learning efficiency  
  
* * *

## 5.5 RLHF (Reinforcement Learning from Human Feedback)

### What is RLHF

**RLHF** is a method to improve LLMs by leveraging human feedback. It is adopted by almost all commercial LLMs including ChatGPT, Claude, and Gemini.
    
    
    ```mermaid
    graph TB
        A[RLHF 3-Stage Process] --> B[Step 1Pre-trained LLM]
    
        B --> C[Step 2Reward Model Training]
        C --> C1[Human evaluationof responses]
        C --> C2[Create preferred/non-preferred pairs]
        C --> C3[Learn Reward Model]
    
        C3 --> D[Step 3Reinforcement Learning via PPO]
        D --> D1[LLM generates responses]
        D --> D2[Reward Modelscores them]
        D --> D3[Reinforce high-scoring responses]
    
        D3 --> E[Optimized LLM]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#fff9c4
        style E fill:#c8e6c9
    ```

### 3 Steps of RLHF

#### Step 1: Pre-training

Train a Transformer on a large corpus to learn basic language patterns.

$$ \mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[\log P_{\theta}(x)\right] $$

#### Step 2: Reward Model Training

Human evaluators compare multiple responses and rank preferences. A **Reward Model** is trained from this data.

$$ \mathcal{L}_{\text{reward}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))\right] $$

Where:

  * $r_{\phi}$: Reward Model (parameters $\phi$)
  * $y_w$: Preferred response (winner)
  * $y_l$: Non-preferred response (loser)
  * $\sigma$: Sigmoid function

#### Step 3: Reinforcement Learning via PPO

Optimize the LLM using the **PPO (Proximal Policy Optimization)** algorithm. KL divergence constraint prevents large deviations from the original pre-trained model.

$$ \mathcal{L}_{\text{RLHF}} = \mathbb{E}_{x, y} \left[r_{\phi}(x, y) - \beta \cdot D_{KL}(\pi_{\theta} || \pi_{\text{ref}})\right] $$

Where:

  * $\pi_{\theta}$: Policy being optimized (LLM)
  * $\pi_{\text{ref}}$: Reference policy (original model)
  * $\beta$: Strength of KL constraint

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    
    class RewardModel(nn.Module):
        """
        Reward Model for RLHF
        """
    
        def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
            """
            Args:
                input_dim: Input embedding dimension (e.g., BERT embedding)
                hidden_dim: Hidden layer dimension
            """
            super().__init__()
    
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)  # Output scalar reward
            )
    
        def forward(self, embeddings):
            """
            Args:
                embeddings: [batch_size, seq_len, input_dim]
    
            Returns:
                rewards: [batch_size] scalar rewards
            """
            # Average pooling
            pooled = embeddings.mean(dim=1)  # [batch_size, input_dim]
            rewards = self.network(pooled).squeeze(-1)  # [batch_size]
            return rewards
    
    class RLHFTrainer:
        """
        RLHF training process simulation
        """
    
        def __init__(self, reward_model: RewardModel, beta: float = 0.1):
            """
            Args:
                reward_model: Trained Reward Model
                beta: Strength of KL divergence constraint
            """
            self.reward_model = reward_model
            self.beta = beta
    
        def compute_reward(self, response_embeddings: torch.Tensor) -> torch.Tensor:
            """
            Calculate reward for responses
    
            Args:
                response_embeddings: [batch_size, seq_len, embed_dim]
    
            Returns:
                rewards: [batch_size]
            """
            with torch.no_grad():
                rewards = self.reward_model(response_embeddings)
            return rewards
    
        def compute_kl_penalty(self,
                              current_logprobs: torch.Tensor,
                              reference_logprobs: torch.Tensor) -> torch.Tensor:
            """
            Calculate KL divergence penalty
    
            Args:
                current_logprobs: Log probabilities of current model
                reference_logprobs: Log probabilities of reference model
    
            Returns:
                kl_penalty: KL divergence
            """
            kl = current_logprobs - reference_logprobs
            return kl.mean()
    
        def ppo_loss(self,
                     old_logprobs: torch.Tensor,
                     new_logprobs: torch.Tensor,
                     advantages: torch.Tensor,
                     epsilon: float = 0.2) -> torch.Tensor:
            """
            Calculate PPO (Proximal Policy Optimization) loss
    
            Args:
                old_logprobs: Log probabilities of old policy
                new_logprobs: Log probabilities of new policy
                advantages: Advantages (rewards - baseline)
                epsilon: Clipping range
    
            Returns:
                ppo_loss: PPO loss
            """
            # Probability ratio
            ratio = torch.exp(new_logprobs - old_logprobs)
    
            # Clipped objective function
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
            # PPO loss (to minimize)
            loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
    
            return loss
    
    # Usage example
    print("=== RLHF Reward Model Demo ===\n")
    
    # Initialize Reward Model
    reward_model = RewardModel(input_dim=768, hidden_dim=256)
    
    # Dummy data: embeddings of 2 responses
    torch.manual_seed(42)
    response1_emb = torch.randn(1, 20, 768)  # Preferred response
    response2_emb = torch.randn(1, 20, 768)  # Non-preferred response
    
    # Calculate rewards
    reward1 = reward_model(response1_emb)
    reward2 = reward_model(response2_emb)
    
    print(f"Reward for response 1: {reward1.item():.4f}")
    print(f"Reward for response 2: {reward2.item():.4f}")
    
    # Calculate pairwise loss (during training)
    pairwise_loss = -F.logsigmoid(reward1 - reward2).mean()
    print(f"\nPairwise ranking loss: {pairwise_loss.item():.4f}")
    
    # Initialize RLHF trainer
    trainer = RLHFTrainer(reward_model, beta=0.1)
    
    # PPO loss calculation example
    old_logprobs = torch.randn(16)  # [batch_size]
    new_logprobs = old_logprobs + torch.randn(16) * 0.1
    advantages = torch.randn(16)
    
    ppo_loss = trainer.ppo_loss(old_logprobs, new_logprobs, advantages)
    print(f"\nPPO loss: {ppo_loss.item():.4f}")
    
    # Calculate KL penalty
    kl_penalty = trainer.compute_kl_penalty(new_logprobs, old_logprobs)
    print(f"KL penalty: {kl_penalty.item():.4f}")
    
    # Total objective function
    total_loss = ppo_loss + trainer.beta * kl_penalty
    print(f"\nTotal loss (PPO + KL constraint): {total_loss.item():.4f}")
    

### Challenges and Improvements in RLHF

Challenge | Description | Solution  
---|---|---  
**Reward Hacking** | Model finds unintended ways to maximize reward | Diverse evaluators, adjust KL constraints  
**Evaluator Bias** | Inconsistency in human evaluations | Multi-evaluator consensus, guideline development  
**Computational Cost** | PPO training is computationally expensive | Alternative methods like DPO (Direct Preference Optimization)  
**Excessive Safety** | Responses become overly cautious | Fine-tuning reward model  
  
* * *

## 5.6 Practical Projects

### Project 1: Few-shot Text Classification System
    
    
    import openai
    import os
    from typing import List, Dict, Tuple
    from collections import Counter
    
    class FewShotClassifier:
        """
        General-purpose text classifier using Few-shot Learning
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
            """
            Args:
                api_key: OpenAI API key (if None, get from environment variable)
                model: Model name to use
            """
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            openai.api_key = self.api_key
    
        def create_few_shot_prompt(self,
                                   examples: List[Tuple[str, str]],
                                   query: str,
                                   labels: List[str]) -> str:
            """
            Build Few-shot prompt
    
            Args:
                examples: [(text1, label1), (text2, label2), ...]
                query: Text to classify
                labels: List of possible labels
    
            Returns:
                Constructed prompt
            """
            prompt = "This is a task to classify sentences into categories.\n\n"
            prompt += f"Available categories: {', '.join(labels)}\n\n"
    
            # Add Few-shot examples
            for i, (text, label) in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Sentence: {text}\n"
                prompt += f"Category: {label}\n\n"
    
            # Add query
            prompt += f"Classify the following sentence:\n"
            prompt += f"Sentence: {query}\n"
            prompt += f"Category:"
    
            return prompt
    
        def classify(self,
                    query: str,
                    examples: List[Tuple[str, str]],
                    labels: List[str],
                    temperature: float = 0.3) -> Dict[str, any]:
            """
            Classify text
    
            Args:
                query: Text to classify
                examples: Few-shot examples
                labels: Possible labels
                temperature: Generation diversity (0-1)
    
            Returns:
                Classification result dictionary
            """
            prompt = self.create_few_shot_prompt(examples, query, labels)
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=50
                )
    
                predicted_label = response.choices[0].message.content.strip()
    
                return {
                    'query': query,
                    'predicted_label': predicted_label,
                    'prompt': prompt,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'query': query,
                    'error': str(e),
                    'success': False
                }
    
        def batch_classify(self,
                          queries: List[str],
                          examples: List[Tuple[str, str]],
                          labels: List[str]) -> List[Dict]:
            """
            Batch classify multiple texts
    
            Args:
                queries: List of texts to classify
                examples: Few-shot examples
                labels: Possible labels
    
            Returns:
                List of classification results
            """
            results = []
            for query in queries:
                result = self.classify(query, examples, labels)
                results.append(result)
    
            return results
    
    # Usage example: News article classification
    print("=== Few-shot Text Classification Demo ===\n")
    
    # Note: Requires OpenAI API key for execution
    # classifier = FewShotClassifier()
    
    # Few-shot training examples
    examples = [
        ("New smartphone released, reservations flooding in.", "Technology"),
        ("Stock market surges, hitting record highs.", "Business"),
        ("Japan's national soccer team wins at World Cup.", "Sports"),
        ("New movie released, breaking box office records.", "Entertainment"),
        ("Government announces new economic policy.", "Politics"),
    ]
    
    # Labels
    labels = ["Technology", "Business", "Sports", "Entertainment", "Politics"]
    
    # Test queries
    test_queries = [
        "Massive investment being made in AI research and development.",
        "Professional baseball championship team has been decided.",
        "New game becomes a worldwide hit.",
    ]
    
    # Execute classification (demo pseudocode)
    print("Few-shot examples:")
    for text, label in examples:
        print(f"  [{label}] {text}")
    
    print("\nTexts to classify:")
    for query in test_queries:
        print(f"  - {query}")
    
    # Actual API call is commented out
    # results = classifier.batch_classify(test_queries, examples, labels)
    #
    # print("\nResults:")
    # for result in results:
    #     if result['success']:
    #         print(f"✅ [{result['predicted_label']}] {result['query']}")
    #     else:
    #         print(f"❌ Error: {result['error']}")
    

### Project 2: Chain-of-Thought Reasoning Engine
    
    
    class ChainOfThoughtReasoner:
        """
        Problem-solving engine implementing Chain-of-Thought reasoning
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            """
            Args:
                api_key: OpenAI API key
                model: Model to use (GPT-4 recommended for CoT)
            """
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            openai.api_key = self.api_key
    
        def solve_math_problem(self, problem: str) -> Dict[str, any]:
            """
            Solve math problem with CoT
    
            Args:
                problem: Problem statement
    
            Returns:
                Solution and reasoning process
            """
            prompt = f"""
    Solve the following math problem step by step.
    Clearly explain what you are calculating at each step.
    
    Problem: {problem}
    
    Solution steps:
    Step 1:"""
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
    
                reasoning = response.choices[0].message.content
    
                # Extract final answer (simplified version)
                lines = reasoning.split('\n')
                answer_line = [l for l in lines if 'Answer' in l or 'answer' in l]
                final_answer = answer_line[-1] if answer_line else "Extraction failed"
    
                return {
                    'problem': problem,
                    'reasoning': reasoning,
                    'final_answer': final_answer,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'problem': problem,
                    'error': str(e),
                    'success': False
                }
    
        def solve_logic_puzzle(self, puzzle: str, question: str) -> Dict[str, any]:
            """
            Solve logic puzzle with CoT
    
            Args:
                puzzle: Puzzle situation description
                question: Question to solve
    
            Returns:
                Solution and reasoning process
            """
            prompt = f"""
    Analyze and solve the following logic puzzle step by step.
    
    Situation:
    {puzzle}
    
    Question: {question}
    
    Analysis steps:
    Observation 1:"""
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600
                )
    
                reasoning = response.choices[0].message.content
    
                return {
                    'puzzle': puzzle,
                    'question': question,
                    'reasoning': reasoning,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'puzzle': puzzle,
                    'error': str(e),
                    'success': False
                }
    
    # Usage example
    print("=== Chain-of-Thought Reasoning Engine Demo ===\n")
    
    # reasoner = ChainOfThoughtReasoner()
    
    # Math problem
    math_problem = """
    An item's regular price is 10,000 yen.
    It was discounted 20% off in a sale, then an additional 500 yen off with a coupon.
    Calculate the final payment amount.
    """
    
    print("Math Problem:")
    print(math_problem)
    print("\nExpected reasoning:")
    print("Step 1: Calculate 20% off amount → 10,000 × 0.2 = 2,000 yen")
    print("Step 2: Price after sale → 10,000 - 2,000 = 8,000 yen")
    print("Step 3: Apply coupon → 8,000 - 500 = 7,500 yen")
    print("Answer: 7,500 yen")
    
    # result = reasoner.solve_math_problem(math_problem)
    # if result['success']:
    #     print("\nCoT Reasoning Result:")
    #     print(result['reasoning'])
    

### Project 3: Integrated LLM Chatbot System
    
    
    from datetime import datetime
    from typing import Optional
    
    class LLMChatbot:
        """
        Chatbot integrating multiple prompt techniques
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            self.conversation_history = []
            openai.api_key = self.api_key
    
        def set_system_prompt(self, persona: str, capabilities: List[str]):
            """
            Set system prompt (bot persona)
    
            Args:
                persona: Bot's personality and role
                capabilities: List of bot's capabilities
            """
            system_prompt = f"""
    You are {persona}.
    
    Your capabilities:
    {chr(10).join('- ' + cap for cap in capabilities)}
    
    In conversations with users, provide kind and accurate information.
    For uncertain information, clearly state that.
    """
            self.conversation_history = [
                {"role": "system", "content": system_prompt}
            ]
    
        def chat(self,
                user_message: str,
                use_cot: bool = False,
                temperature: float = 0.7) -> Dict[str, any]:
            """
            Respond to user message
    
            Args:
                user_message: User's input
                use_cot: Whether to use Chain-of-Thought
                temperature: Response diversity
    
            Returns:
                Response result
            """
            # Add CoT prompt
            if use_cot:
                user_message = f"""
    {user_message}
    
    When answering the above question, please think step by step and explain.
    """
    
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=temperature,
                    max_tokens=500
                )
    
                assistant_message = response.choices[0].message.content
    
                # Add assistant's response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
    
                return {
                    'user': user_message,
                    'assistant': assistant_message,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
    
            except Exception as e:
                return {
                    'user': user_message,
                    'error': str(e),
                    'success': False
                }
    
        def get_conversation_summary(self) -> str:
            """Get summary of conversation history"""
            if len(self.conversation_history) <= 1:
                return "Conversation has not started yet."
    
            summary = "Conversation History:\n"
            for i, msg in enumerate(self.conversation_history[1:], 1):  # Skip system prompt
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                summary += f"{i}. [{role}] {content}\n"
    
            return summary
    
        def clear_history(self, keep_system: bool = True):
            """
            Clear conversation history
    
            Args:
                keep_system: Whether to keep system prompt
            """
            if keep_system and self.conversation_history:
                self.conversation_history = [self.conversation_history[0]]
            else:
                self.conversation_history = []
    
    # Usage example
    print("=== Integrated LLM Chatbot Demo ===\n")
    
    # Initialize chatbot
    # bot = LLMChatbot(model="gpt-3.5-turbo")
    
    # Persona setup
    persona = "a kind and knowledgeable AI assistant"
    capabilities = [
        "Answering general questions",
        "Programming support",
        "Solving math and logic problems",
        "Text summarization and translation",
    ]
    
    # bot.set_system_prompt(persona, capabilities)
    
    print(f"Bot Persona: {persona}\n")
    print("Conversation Simulation:\n")
    
    # Conversation examples (demo)
    demo_conversations = [
        ("Hello! Please tell me how to reverse a list in Python.", False),
        ("If the list has 1 million elements, what would be the most efficient method?", True),  # Use CoT
    ]
    
    for user_msg, use_cot in demo_conversations:
        print(f"👤 User: {user_msg}")
    
        if use_cot:
            print("   (Using Chain-of-Thought reasoning)")
    
        # Actual API call is commented out
        # result = bot.chat(user_msg, use_cot=use_cot)
        # if result['success']:
        #     print(f"🤖 Assistant: {result['assistant']}")
        # else:
        #     print(f"❌ Error: {result['error']}")
    
        print()
    
    # print(bot.get_conversation_summary())
    

* * *

## Exercises

**Exercise 5.1: Understanding Scaling Laws**

**Problem** : When training two models under the following conditions, which would be expected to have higher performance? Explain based on the Chinchilla scaling laws.

  * Model A: 200B parameters, trained on 1 trillion tokens
  * Model B: 70B parameters, trained on 4 trillion tokens

**Hint** : Chinchilla optimal ratio is "data tokens ≈ 20 × parameters".

Sample Answer

**Analysis** :

  * Model A: Optimal data = 200B × 20 = 4 trillion tokens → Actual: 1 trillion tokens (insufficient)
  * Model B: Optimal data = 70B × 20 = 1.4 trillion tokens → Actual: 4 trillion tokens (excessive but acceptable)

**Conclusion** : Model B is more likely to have higher performance. Model A is "over-parameterized" and its performance plateaus due to data shortage. As the Chinchilla paper shows, with the same compute budget, it's more efficient to train a smaller model on more data.

**Exercise 5.2: Few-shot Prompt Design**

**Problem** : Design an effective Few-shot prompt for the following task.

**Task** : Extract "rating score (1-5)" and "main reason" from product reviews

**Requirements** :

  * Include 3 examples
  * Clearly specify output format
  * Consider edge cases (ambiguous ratings)

Sample Answer
    
    
    Extract "rating score (1-5)" and "main reason" from the following product reviews.
    
    Example 1:
    Review: This vacuum has strong suction, is lightweight and easy to use. The price is reasonable and I'm very satisfied.
    Output: {"score": 5, "reason": "Suction power, lightweight, cost performance"}
    
    Example 2:
    Review: The design is good, but battery life is poor and needs frequent charging.
    Output: {"score": 2, "reason": "Short battery duration"}
    
    Example 3:
    Review: It's an ordinary product. Nothing particularly good or bad.
    Output: {"score": 3, "reason": "No notable features"}
    
    Review: {input_review}
    Output:
    

**Design Points** :

  * Include balanced examples: positive (Example 1), negative (Example 2), neutral (Example 3)
  * Specify JSON format for structured output that's easy to parse
  * "Reason" should be a concise summary, guiding against copying the entire review

**Exercise 5.3: Implementing Chain-of-Thought Reasoning**

**Problem** : Create a CoT prompt to solve the following logic puzzle.

**Puzzle** :

> There are 3 suspects A, B, C. 
> 
>   * A says "B is the culprit"
>   * B says "I am innocent"
>   * C says "A is the culprit"
> 
There is one culprit, and only that person is lying. Who is the culprit? 

Sample Answer
    
    
    Analyze and solve the following logic puzzle step by step.
    
    Puzzle:
    There are 3 suspects A, B, C.
    - A says "B is the culprit"
    - B says "I am innocent"
    - C says "A is the culprit"
    There is one culprit, and only that person is lying. Who is the culprit?
    
    Step-by-step analysis:
    
    Hypothesis 1: If A is the culprit
     - A's statement "B is the culprit" is a lie → ✓ Culprit lies
     - B's statement "I am innocent" is true → ✓ B and C tell truth
     - C's statement "A is the culprit" is true → ✓ No contradiction
     Conclusion: A could be the culprit
    
    Hypothesis 2: If B is the culprit
     - A's statement "B is the culprit" is true → ✗ Contradiction (non-culprit also lies?)
     - B's statement "I am innocent" is a lie → ✓ Culprit lies
     - C's statement "A is the culprit" is a lie → ✗ Contradiction (2 people lie?)
     Conclusion: Contradicts conditions
    
    Hypothesis 3: If C is the culprit
     - A's statement "B is the culprit" is a lie → ✗ Contradiction (non-culprit also lies?)
     - B's statement "I am innocent" is true → ✓
     - C's statement "A is the culprit" is a lie → ✓ Culprit lies
     Conclusion: Contradicts conditions
    
    Final Conclusion: A is the culprit.
    Reason: Only Hypothesis 1 satisfies all conditions.
    

**CoT Design Points** :

  * Systematically verify all possibilities
  * Clearly check for contradictions under each hypothesis
  * Use symbols (✓, ✗) for visual clarity

**Exercise 5.4: Understanding RLHF**

**Problem** : Explain the role of the KL divergence constraint $\beta \cdot D_{KL}(\pi_{\theta} || \pi_{\text{ref}})$ used in RLHF. Also describe the problems when $\beta$ is too large or too small.

Sample Answer

**Role of KL Divergence Constraint** :

  1. **Prevent mode collapse** : Constrains the optimizing model $\pi_{\theta}$ from deviating too much from the reference model $\pi_{\text{ref}}$ (pre-trained).
  2. **Preserve language ability** : Prevents loss of basic language abilities like grammar and coherence during reward maximization.
  3. **Avoid reward hacking** : Prevents the model from learning extreme strategies that exploit vulnerabilities in the reward model.

**When $\beta$ is too large** :

  * Problem: Model stays too close to reference model, reducing RLHF effectiveness
  * Result: Human feedback is barely reflected, little improvement seen

**When $\beta$ is too small** :

  * Problem: Model deviates significantly from reference model, generating unnatural outputs
  * Result: Grammar collapse, nonsensical responses, reward hacking

**Practical $\beta$ selection** :

  * Common range: 0.01~0.1
  * Tuning method: Search for optimal value based on human evaluation on validation set

**Exercise 5.5: LLM Application Design**

**Problem** : Design an LLM chatbot for customer support. Propose a system architecture and prompt strategy that meets the following requirements.

**Requirements** :

  * Immediately answer frequently asked questions (FAQ)
  * Handle complex issues step by step
  * Escalate to human operators when uncertain
  * Context understanding considering conversation history

Sample Answer

**System Architecture** :
    
    
    class CustomerSupportChatbot:
        """
        LLM chatbot for customer support
        """
    
        def __init__(self):
            self.faq_database = self.load_faq()
            self.conversation_history = []
            self.escalation_threshold = 0.3  # Confidence threshold
    
        def load_faq(self):
            """Load FAQ database"""
            return {
                "Shipping days": "Typically delivered in 3-5 business days.",
                "Return policy": "Returns accepted within 30 days of purchase.",
                "Payment methods": "We accept credit cards, bank transfers, and cash on delivery.",
            }
    
        def check_faq(self, query: str) -> Optional[str]:
            """Check for matching FAQ questions"""
            # Simplified: Would use embedding-based similarity search in practice
            for question, answer in self.faq_database.items():
                if question in query:
                    return answer
            return None
    
        def classify_complexity(self, query: str) -> str:
            """Classify inquiry complexity"""
            complexity_prompt = f"""
    Classify the following inquiry as "Simple", "Medium", or "Complex".
    
    Inquiry: {query}
    
    Classification:"""
            # LLM call (pseudocode)
            # complexity = call_llm(complexity_prompt)
            return "Medium"  # For demo
    
        def handle_query(self, query: str) -> Dict:
            """Handle inquiry"""
            # Step 1: FAQ check
            faq_answer = self.check_faq(query)
            if faq_answer:
                return {
                    'type': 'faq',
                    'answer': faq_answer,
                    'confidence': 1.0
                }
    
            # Step 2: Complexity assessment
            complexity = self.classify_complexity(query)
    
            # Step 3: Process based on complexity
            if complexity == "Simple":
                return self.simple_response(query)
            elif complexity == "Medium":
                return self.cot_response(query)
            else:
                return self.escalate_to_human(query)
    
        def simple_response(self, query: str):
            """Simple Zero-shot response"""
            prompt = f"""
    You are a customer support assistant.
    Answer the following question concisely.
    
    Question: {query}
    
    Answer:"""
            # response = call_llm(prompt)
            return {'type': 'simple', 'answer': "Response content"}
    
        def cot_response(self, query: str):
            """Handle step by step with Chain-of-Thought"""
            prompt = f"""
    You are a customer support assistant.
    Analyze the following problem step by step and propose a solution.
    
    Problem: {query}
    
    Analysis:
    Step 1:"""
            # response = call_llm(prompt)
            return {'type': 'cot', 'answer': "Step-by-step response"}
    
        def escalate_to_human(self, query: str):
            """Escalate to human operator"""
            return {
                'type': 'escalation',
                'message': "This issue is complex. I'll connect you to a specialized operator.",
                'query': query
            }
    

**Prompt Strategy** :

  1. **System Prompt** : Clearly define bot's role, tone, and constraints
  2. **Few-shot FAQ** : Present similar question examples to improve accuracy
  3. **CoT for Complex Issues** : Analyze complex problems step by step
  4. **Confidence Scoring** : Evaluate response confidence, escalate if low

**Evaluation Metrics** :

  * FAQ match rate: 70% or higher
  * Escalation rate: 15% or lower
  * Customer satisfaction: 4.0/5.0 or higher

* * *

## Summary

In this chapter, we learned the essence and practical utilization of Large Language Models (LLMs):

  * ✅ **Scaling Laws** : Understood the relationship between model size, data amount, and compute, and grasped the Chinchilla optimal ratio
  * ✅ **Major LLMs** : Compared architectures and differentiators of GPT, LLaMA, Claude, Gemini, etc.
  * ✅ **Prompt Engineering** : Implemented techniques such as Zero-shot, Few-shot, and Chain-of-Thought
  * ✅ **In-Context Learning** : Understood the mechanism for learning new tasks without parameter updates
  * ✅ **RLHF** : Learned the model improvement process using human feedback
  * ✅ **Practical Projects** : Built Few-shot classification, CoT reasoning, and integrated chatbots

> **Next Steps** : Once you understand LLM fundamentals, move on to more advanced topics such as domain-specific fine-tuning, RAG (Retrieval-Augmented Generation), and multimodal LLMs. Ethical considerations and bias mitigation techniques for responsible AI development are also important.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
