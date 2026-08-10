---
title: 第4章：大規模言語モデル（LLMs）
chapter_title: 第4章：大規模言語モデル（LLMs）
subtitle: GPT、LLaMA、Prompt Engineering - 言語理解の最前線
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ GPTファミリーのアーキテクチャと自己回帰生成を理解する
  * ✅ LLMsの学習手法（事前学習、Instruction Tuning、RLHF）を理解する
  * ✅ Prompt Engineeringの技法を実践できる
  * ✅ オープンソースLLMsの特性と量子化技術を理解する
  * ✅ RAGやFunction Callingなど実践的応用を実装できる
  * ✅ 完全なチャットボットシステムを構築できる

* * *

## 4.1 GPTファミリー

### GPTアーキテクチャの概要

**GPT（Generative Pre-trained Transformer）** は、Decoder-onlyのTransformerアーキテクチャを採用した自己回帰型言語モデルです。

> **Decoder-only** : EncoderとDecoderを持つBERT等と異なり、GPTはDecoderのみで構成され、次トークン予測に特化しています。

### GPTアーキテクチャの特徴

特徴 | 説明 | 利点  
---|---|---  
**Decoder-only** | 自己注意機構のみ使用 | シンプルで拡張しやすい  
**Causal Masking** | 未来のトークンを隠す | 自己回帰生成を実現  
**Autoregressive** | 左から右へ順次生成 | 自然な文生成  
**事前学習** | 大規模テキストで学習 | 汎用的な言語理解  
  
### GPTの進化
    
    
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

### GPT-2/GPT-3/GPT-4の比較

モデル | パラメータ数 | コンテキスト長 | 主な特徴  
---|---|---|---  
**GPT-2** | 117M - 1.5B | 1,024 | 高品質テキスト生成  
**GPT-3** | 175B | 2,048 | Few-shot学習、In-context Learning  
**GPT-3.5** | ~175B | 4,096 | Instruction Tuning、対話性能向上  
**GPT-4** | 非公開 | 8,192 - 32,768 | マルチモーダル、高度な推論  
  
### 自己回帰生成（Autoregressive Generation）

GPTは、前のトークンから次のトークンを予測する**自己回帰的** にテキストを生成します。

$$ P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1}) $$

各トークンの確率は、それまでのすべてのトークンに条件付けられます。

### 実例：GPT-2でのテキスト生成
    
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    # GPT-2モデルとトークナイザーの読み込み
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # テキスト生成
    prompt = "Artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成パラメータ
    generation_config = {
        'max_length': 100,
        'num_return_sequences': 3,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'do_sample': True,
        'no_repeat_ngram_size': 2
    }
    
    # テキスト生成
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            **generation_config
        )
    
    print("=== GPT-2によるテキスト生成 ===")
    print(f"プロンプト: '{prompt}'\n")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"生成 {i+1}:")
        print(f"{text}\n")
    

**出力例** ：
    
    
    === GPT-2によるテキスト生成 ===
    プロンプト: 'Artificial intelligence is'
    
    生成 1:
    Artificial intelligence is becoming more and more important in our daily lives. From smartphones to self-driving cars, AI systems are transforming the way we work and live. The technology has advanced rapidly...
    
    生成 2:
    Artificial intelligence is a field of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence, such as visual perception...
    
    生成 3:
    Artificial intelligence is revolutionizing industries across the globe. Companies are investing billions in AI research to develop systems that can learn from data and make decisions autonomously...
    

### 生成パラメータの制御
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 異なるtemperatureでの生成
    prompt = "The future of AI is"
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    print("=== Temperature の影響 ===\n")
    
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
    
    # temperatureと確率分布の可視化
    def softmax_with_temperature(logits, temperature):
        """温度パラメータを適用したsoftmax"""
        return torch.softmax(logits / temperature, dim=-1)
    
    # サンプルlogits
    logits = torch.tensor([2.0, 1.0, 0.5, 0.2, 0.1])
    temps = [0.5, 1.0, 2.0]
    
    plt.figure(figsize=(12, 4))
    for i, temp in enumerate(temps):
        probs = softmax_with_temperature(logits, temp)
    
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(probs)), probs.numpy())
        plt.title(f'Temperature = {temp}')
        plt.xlabel('Token ID')
        plt.ylabel('確率')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 低いtemperature → 確定的（高確率トークンを選択）")
    print("📊 高いtemperature → 多様性（低確率トークンも選択）")
    

### Beam SearchとSampling
    
    
    from transformers import GenerationConfig
    
    prompt = "Machine learning can be used for"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("=== 生成戦略の比較 ===\n")
    
    # 1. Greedy Decoding（貪欲法）
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
    

> **重要** : Greedy/Beam Searchは決定論的で一貫性があり、Samplingは多様性がありますが、再現性が低くなります。

* * *

## 4.2 LLMsの学習手法

### LLM学習の全体像
    
    
    ```mermaid
    graph TD
        A[大規模テキストコーパス] --> B[事前学習Pre-training]
        B --> C[ベースモデルBase LLM]
        C --> D[Instruction Tuning指示チューニング]
        D --> E[指示に従うモデルInstruction-following LLM]
        E --> F[RLHF人間フィードバックによる強化学習]
        F --> G[アライメント済みモデルAligned LLM]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#c8e6c9
    ```

### 1\. 事前学習（Pre-training）

**事前学習** は、大規模なテキストコーパスで次トークン予測タスクを学習します。

要素 | 内容  
---|---  
**データ** | Web、書籍、論文など数千億〜数兆トークン  
**目的関数** | 次トークン予測（Language Modeling）  
**最適化** | AdamW、学習率スケジューリング  
**計算資源** | 数千〜数万GPU、数週間〜数ヶ月  
  
**損失関数** ：

$$ \mathcal{L}_{\text{LM}} = -\sum_{i=1}^{n} \log P(x_i | x_{<i}; \theta) $$

### 2\. Instruction Tuning（指示チューニング）

**Instruction Tuning** は、ユーザーの指示に従うようにモデルを微調整します。

#### Instruction Tuningのデータ形式
    
    
    instruction_data = [
        {
            "instruction": "以下の文章を要約してください。",
            "input": "人工知能（AI）は、コンピュータが人間のように考え、学習し、問題を解決する技術です。機械学習、深層学習、自然言語処理などの分野を含みます。",
            "output": "AIは、コンピュータに人間のような知能を持たせる技術で、機械学習や自然言語処理などを含みます。"
        },
        {
            "instruction": "次の質問に答えてください。",
            "input": "地球から月までの距離はどのくらいですか？",
            "output": "地球から月までの平均距離は約384,400キロメートル（238,855マイル）です。"
        },
        {
            "instruction": "以下のコードのバグを修正してください。",
            "input": "def add(a, b):\n    return a - b",
            "output": "def add(a, b):\n    return a + b"
        }
    ]
    
    # データ形式の可視化
    import pandas as pd
    
    df = pd.DataFrame(instruction_data)
    print("=== Instruction Tuning データ形式 ===")
    print(df.to_string(index=False))
    print(f"\nデータ数: {len(instruction_data)}件")
    

#### Instruction Tuningの実装例（概念）
    
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset
    
    # データの準備
    def format_instruction(example):
        """指示データをプロンプト形式に変換"""
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example['input']:
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}
    
    # データセット作成
    dataset = Dataset.from_list(instruction_data)
    dataset = dataset.map(format_instruction)
    
    print("=== フォーマット済みプロンプト例 ===")
    print(dataset[0]['text'])
    print("\n" + "="*50 + "\n")
    
    # モデルとトークナイザーの準備（実際には大規模モデルを使用）
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # トークナイズ関数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    # データセットのトークナイズ
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 訓練設定（デモ用）
    training_args = TrainingArguments(
        output_dir="./instruction-tuned-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no"
    )
    
    print("=== Instruction Tuning 設定 ===")
    print(f"エポック数: {training_args.num_train_epochs}")
    print(f"バッチサイズ: {training_args.per_device_train_batch_size}")
    print(f"学習率: {training_args.learning_rate}")
    print("\n✓ Instruction Tuningにより、モデルは指示に従うようになります")
    

### 3\. RLHF（Reinforcement Learning from Human Feedback）

**RLHF** は、人間のフィードバックを使ってモデルを人間の価値観に合わせる技術です。

#### RLHFの3ステップ
    
    
    ```mermaid
    graph LR
        A[Step 1:教師ありファインチューニングSFT] --> B[Step 2:報酬モデル訓練Reward Model]
        B --> C[Step 3:PPO強化学習Policy Optimization]
    
        style A fill:#e3f2fd
        style B fill:#f3e5f5
        style C fill:#e8f5e9
    ```

#### Step 1: Supervised Fine-Tuning (SFT)

高品質な人間作成の対話データでファインチューニングします。

#### Step 2: Reward Model Training

人間の評価（ランキング）から報酬モデルを学習します。
    
    
    # 報酬モデルのデータ形式（概念）
    reward_data = [
        {
            "prompt": "人工知能とは何ですか？",
            "response_1": "人工知能（AI）は、コンピュータが人間のように考える技術です。",
            "response_2": "わかりません。",
            "preference": 1  # response_1の方が良い
        },
        {
            "prompt": "Pythonでリストを逆順にする方法は？",
            "response_1": "list.reverse()メソッドまたはlist[::-1]スライシングを使います。",
            "response_2": "できません。",
            "preference": 1
        }
    ]
    
    import pandas as pd
    df_reward = pd.DataFrame(reward_data)
    print("=== 報酬モデル訓練データ ===")
    print(df_reward.to_string(index=False))
    print("\n✓ 人間の選好から報酬関数を学習")
    

#### Step 3: PPO (Proximal Policy Optimization)

報酬モデルを使って、強化学習でモデルを最適化します。

**目的関数** ：

$$ \mathcal{L}^{\text{PPO}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ r(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}}) \right] $$

  * $r(x, y)$: 報酬モデルのスコア
  * $\beta$: KL正則化係数
  * $\pi_{\text{ref}}$: 元のモデル（大きく逸脱しないため）

> **RLHF の効果** : ChatGPTの人間らしい対話能力は、RLHFによって実現されています。

### 4\. Parameter-Efficient Fine-Tuning（PEFT）

大規模モデル全体を微調整するのは計算コストが高いため、**パラメータ効率的な微調整** が重要です。

#### LoRA（Low-Rank Adaptation）

LoRAは、モデルの重み行列に低ランク分解を適用します。

$$ W' = W + \Delta W = W + BA $$

  * $W$: 元の重み行列（固定）
  * $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: 学習可能な低ランク行列
  * $r \ll \min(d, k)$: ランク（例: 8, 16）

    
    
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM
    
    # ベースモデルの読み込み
    model_name = "gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # LoRA設定
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # ランク
        lora_alpha=32,  # スケーリング係数
        lora_dropout=0.1,
        target_modules=["c_attn"]  # 適用するモジュール
    )
    
    # LoRAモデルの作成
    lora_model = get_peft_model(base_model, lora_config)
    
    # パラメータ数の比較
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print("=== LoRA パラメータ効率 ===")
    print(f"総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    print(f"訓練可能割合: {100 * trainable_params / total_params:.2f}%")
    print(f"\n✓ わずか{100 * trainable_params / total_params:.2f}%のパラメータで微調整可能")
    print(lora_model.print_trainable_parameters())
    

**出力例** ：
    
    
    === LoRA パラメータ効率 ===
    総パラメータ数: 124,439,808
    訓練可能パラメータ数: 294,912
    訓練可能割合: 0.24%
    
    ✓ わずか0.24%のパラメータで微調整可能
    trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.24
    

#### Adapter Layers

Transformerの各層に小さなボトルネックネットワークを追加します。

手法 | 訓練可能パラメータ | メモリ | 速度  
---|---|---|---  
**Full Fine-tuning** | 100% | 高 | 遅い  
**LoRA** | 0.1% - 1% | 低 | 速い  
**Adapter** | 2% - 5% | 中 | 中  
**Prompt Tuning** | < 0.1% | 非常に低 | 非常に速い  
  
* * *

## 4.3 Prompt Engineering

### プロンプトエンジニアリングとは

**Prompt Engineering** は、LLMから望ましい出力を引き出すための入力（プロンプト）設計技術です。

> 「適切なプロンプトは、モデルの性能を10倍にすることができる」 - OpenAI

### 1\. Zero-shot Learning

事前学習済みモデルに、タスクの例を一切与えずに直接質問します。
    
    
    from transformers import pipeline
    
    # GPT-2パイプライン
    generator = pipeline('text-generation', model='gpt2')
    
    # Zero-shot プロンプト
    zero_shot_prompt = """
    Question: What is the capital of France?
    Answer:
    """
    
    result = generator(zero_shot_prompt, max_length=50, num_return_sequences=1)
    print("=== Zero-shot Learning ===")
    print(result[0]['generated_text'])
    

### 2\. Few-shot Learning

タスクの例を数個提示してから、新しい入力を与えます。
    
    
    # Few-shot プロンプト（In-Context Learning）
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
    print("\n✓ 例を示すことで、モデルはパターンを学習し翻訳を実行")
    

### 3\. Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** は、モデルに中間的な推論ステップを生成させる技法です。
    
    
    # 通常のプロンプト（直接回答）
    standard_prompt = """
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
    Each can has 3 tennis balls. How many tennis balls does he have now?
    Answer:
    """
    
    # Chain-of-Thought プロンプト
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
    print("\n✓ ステップバイステップの推論により、複雑な問題を解決")
    

#### CoTの効果（研究結果）

タスク | 標準プロンプト | CoTプロンプト | 改善率  
---|---|---|---  
算数問題 | 34% | 78% | +44%  
常識推論 | 61% | 89% | +28%  
論理パズル | 42% | 81% | +39%  
  
### 4\. プロンプト設計のベストプラクティス
    
    
    # ❌ 悪いプロンプト
    bad_prompt = "Summarize this."
    
    # ✅ 良いプロンプト
    good_prompt = """
    Task: Summarize the following article in 3 bullet points.
    Focus on key findings and implications.
    
    Article: [長い記事テキスト...]
    
    Summary:
    -
    """
    
    # プロンプト設計の原則
    prompt_principles = {
        "明確な指示": "タスクを具体的に記述する",
        "フォーマット指定": "望ましい出力形式を示す",
        "コンテキスト提供": "必要な背景情報を含める",
        "制約の明示": "文字数制限、スタイルなどを指定",
        "例の提示": "Few-shotで期待される出力を示す",
        "ステップ分解": "複雑なタスクは段階的に"
    }
    
    print("=== プロンプト設計の原則 ===")
    for principle, description in prompt_principles.items():
        print(f"✓ {principle}: {description}")
    
    # 実践例：構造化プロンプト
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
    
    print("\n=== 構造化プロンプト例 ===")
    print(structured_prompt)
    

### 5\. LangChainによるプロンプト管理
    
    
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms import HuggingFacePipeline
    
    # HuggingFace パイプラインをLangChainでラップ
    llm = HuggingFacePipeline(pipeline=generator)
    
    # プロンプトテンプレートの作成
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
    
    # チェーンの作成
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 実行
    question = "What is machine learning?"
    context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    
    result = chain.run(question=question, context=context)
    
    print("=== LangChain プロンプトテンプレート ===")
    print(f"質問: {question}")
    print(f"コンテキスト: {context}")
    print(f"\n生成された回答:")
    print(result)
    
    print("\n✓ LangChainで再利用可能なプロンプトテンプレートを管理")
    

* * *

## 4.4 オープンソースLLMs

### 主要なオープンソースLLM

モデル | 開発元 | パラメータ | 特徴  
---|---|---|---  
**LLaMA** | Meta AI | 7B - 65B | 高性能、研究用  
**LLaMA-2** | Meta AI | 7B - 70B | 商用利用可、Chat版あり  
**Falcon** | TII | 7B - 180B | 高品質データセット  
**ELYZA** | ELYZA Inc. | 7B - 13B | 日本語特化  
**rinna** | rinna | 3.6B - 36B | 日本語、商用利用可  
  
### 1\. LLaMA / LLaMA-2

**LLaMA（Large Language Model Meta AI）** は、Meta AIが開発したオープンソースLLMです。

#### LLaMAの特徴

  * **効率的アーキテクチャ** : GPT-3より少ないパラメータで同等性能
  * **Pre-Normalization** : RMSNorm使用
  * **SwiGLU活性化関数** : 性能向上
  * **Rotary Positional Embeddings (RoPE)** : 位置エンコーディング

    
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # LLaMA-2モデルの読み込み（7Bモデル）
    model_name = "meta-llama/Llama-2-7b-hf"  # Hugging Face Hubから
    
    # 注意: LLaMA-2を使用するにはHugging Faceでアクセス申請が必要
    # ここではデモ用にGPT-2を使用
    
    model_name = "gpt2"  # 実際の環境ではLLaMA-2に置き換え
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("=== LLaMAファミリーの情報 ===")
    print(f"モデル: {model_name}")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"アーキテクチャ: Decoder-only Transformer")
    print(f"コンテキスト長: 2048 (LLaMA) / 4096 (LLaMA-2)")
    
    # LLaMA-2の性能（ベンチマーク結果）
    benchmarks = {
        "MMLU": {"LLaMA-7B": 35.1, "LLaMA-2-7B": 45.3, "GPT-3.5": 70.0},
        "HellaSwag": {"LLaMA-7B": 76.1, "LLaMA-2-7B": 77.2, "GPT-3.5": 85.5},
        "HumanEval": {"LLaMA-7B": 10.5, "LLaMA-2-7B": 12.8, "GPT-3.5": 48.1}
    }
    
    import pandas as pd
    df_bench = pd.DataFrame(benchmarks)
    print("\n=== ベンチマーク性能比較 ===")
    print(df_bench.to_string())
    

### 2\. 日本語LLMs（ELYZA、rinna）

#### ELYZA-japanese-Llama-2

LLaMA-2を日本語で追加事前学習したモデルです。
    
    
    # ELYZA-japanese-Llama-2の使用例（概念）
    # model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    
    japanese_prompt = """
    以下の質問に日本語で答えてください。
    
    質問: 機械学習と深層学習の違いは何ですか？
    
    回答:
    """
    
    print("=== 日本語LLM（ELYZA）===")
    print("✓ LLaMA-2ベース + 日本語追加学習")
    print("✓ 日本語での自然な対話が可能")
    print("✓ 商用利用可能（LLaMA-2ライセンス）")
    print(f"\nプロンプト例:\n{japanese_prompt}")
    
    # rinna GPT-NeoX
    print("\n=== 日本語LLM（rinna）===")
    print("✓ GPT-NeoXアーキテクチャ")
    print("✓ 日本語Wikipediaなどで学習")
    print("✓ 3.6B、36Bモデルあり")
    print("✓ 商用利用可能（MIT License）")
    

### 3\. モデル量子化（Quantization）

**量子化** は、モデルの精度を下げてメモリと計算量を削減する技術です。

#### 量子化の種類

精度 | メモリ削減 | 性能低下 | 用途  
---|---|---|---  
**FP32（元）** | - | - | 学習  
**FP16** | 50% | ほぼなし | 推論  
**8-bit** | 75% | 小 | 推論、ファインチューニング  
**4-bit** | 87.5% | 中 | メモリ制約環境  
  
#### 8-bit量子化の実装
    
    
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    # 8-bit量子化設定
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    # 4-bit量子化設定（QLoRA）
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_use_double_quant=True
    )
    
    # モデル読み込み（8-bit）
    # model_8bit = AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf",
    #     quantization_config=quantization_config_8bit,
    #     device_map="auto"
    # )
    
    print("=== モデル量子化 ===")
    print("\n8-bit量子化:")
    print("  ✓ メモリ使用量: 約7GB（7Bモデル、FP32比75%削減）")
    print("  ✓ 性能低下: 最小限（1-2%程度）")
    print("  ✓ 推論速度: FP32とほぼ同等")
    
    print("\n4-bit量子化 (QLoRA):")
    print("  ✓ メモリ使用量: 約3.5GB（7Bモデル、FP32比87.5%削減）")
    print("  ✓ 性能低下: 小（5-10%程度）")
    print("  ✓ ファインチューニング可能")
    
    # メモリ使用量の計算
    def calculate_model_memory(num_params, precision):
        """モデルのメモリ使用量を計算"""
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            '8bit': 1,
            '4bit': 0.5
        }
        memory_gb = num_params * bytes_per_param[precision] / (1024**3)
        return memory_gb
    
    params_7b = 7_000_000_000
    
    print("\n=== 7Bモデルのメモリ使用量 ===")
    for precision in ['fp32', 'fp16', '8bit', '4bit']:
        memory = calculate_model_memory(params_7b, precision)
        print(f"{precision.upper():6s}: {memory:.2f} GB")
    

**出力** ：
    
    
    === 7Bモデルのメモリ使用量 ===
    FP32  : 26.08 GB
    FP16  : 13.04 GB
    8BIT  : 6.52 GB
    4BIT  : 3.26 GB
    

> **QLoRA（Quantized LoRA）** : 4-bit量子化とLoRAを組み合わせ、消費者向けGPUで大規模モデルをファインチューニング可能にします。

* * *

## 4.5 LLMsの実践的応用

### 1\. Retrieval-Augmented Generation (RAG)

**RAG** は、外部知識ベースから関連情報を検索し、それを基に回答を生成する技術です。
    
    
    ```mermaid
    graph LR
        A[ユーザー質問] --> B[検索Retriever]
        B --> C[知識ベースVector DB]
        C --> D[関連文書]
        D --> E[LLMGenerator]
        A --> E
        E --> F[回答生成]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#c8e6c9
    ```

#### RAGの実装例
    
    
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    
    # 知識ベースのテキスト
    documents = [
        "人工知能（AI）は、コンピュータが人間のように考え、学習する技術です。",
        "機械学習は、データからパターンを学習するAIの一分野です。",
        "深層学習は、多層ニューラルネットワークを使う機械学習の手法です。",
        "自然言語処理（NLP）は、コンピュータが人間の言語を理解する技術です。",
        "Transformerは、2017年に登場した革新的なニューラルネットワークアーキテクチャです。"
    ]
    
    # テキスト分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    texts = text_splitter.create_documents(documents)
    
    # 埋め込みモデル
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # ベクトルストア作成
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    print("=== RAGシステム構築 ===")
    print(f"ドキュメント数: {len(documents)}")
    print(f"チャンク数: {len(texts)}")
    print(f"埋め込みモデル: sentence-transformers/all-MiniLM-L6-v2")
    
    # 検索テスト
    query = "Transformerとは何ですか？"
    relevant_docs = vectorstore.similarity_search(query, k=2)
    
    print(f"\n質問: {query}")
    print("\n関連ドキュメント:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    # LLMと統合（概念）
    # llm = HuggingFacePipeline(...)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     return_source_documents=True
    # )
    # result = qa_chain({"query": query})
    
    print("\n✓ RAGにより、LLMは最新・特定分野の知識で回答可能")
    

### 2\. Function Calling

**Function Calling** は、LLMが外部ツールやAPIを呼び出せるようにする技術です。
    
    
    import json
    
    # 利用可能な関数の定義
    available_functions = {
        "get_weather": {
            "description": "指定された都市の現在の天気を取得",
            "parameters": {
                "city": {"type": "string", "description": "都市名"}
            }
        },
        "calculate": {
            "description": "数学的計算を実行",
            "parameters": {
                "expression": {"type": "string", "description": "計算式"}
            }
        },
        "search_web": {
            "description": "ウェブを検索",
            "parameters": {
                "query": {"type": "string", "description": "検索クエリ"}
            }
        }
    }
    
    # 関数実装（ダミー）
    def get_weather(city):
        """天気情報を取得（ダミー）"""
        return f"{city}の天気は晴れ、気温は25度です。"
    
    def calculate(expression):
        """計算を実行"""
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "計算エラー"
    
    def search_web(query):
        """ウェブ検索（ダミー）"""
        return f"'{query}'の検索結果: [関連情報...]"
    
    # Function Calling プロンプト
    def create_function_calling_prompt(user_query, functions):
        """Function Calling用のプロンプトを生成"""
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
    
    # テスト
    user_query = "東京の天気はどうですか？"
    prompt = create_function_calling_prompt(user_query, available_functions)
    
    print("=== Function Calling ===")
    print(f"ユーザークエリ: {user_query}")
    print(f"\nプロンプト:\n{prompt}")
    
    # 想定される応答（実際にはLLMが生成）
    function_call = {
        "function": "get_weather",
        "parameters": {"city": "東京"}
    }
    
    print(f"\nLLMの関数選択:")
    print(json.dumps(function_call, ensure_ascii=False, indent=2))
    
    # 関数実行
    if function_call["function"] == "get_weather":
        result = get_weather(**function_call["parameters"])
        print(f"\n実行結果: {result}")
    
    print("\n✓ LLMがツールを使って情報を取得・処理可能")
    

### 3\. Multi-turn Conversation（マルチターン対話）

会話履歴を維持して、文脈を理解した対話を実現します。
    
    
    from collections import deque
    
    class ConversationManager:
        """会話履歴を管理するクラス"""
    
        def __init__(self, max_history=10):
            self.history = deque(maxlen=max_history)
    
        def add_message(self, role, content):
            """メッセージを追加"""
            self.history.append({"role": role, "content": content})
    
        def get_prompt(self, system_message=""):
            """会話履歴からプロンプトを生成"""
            prompt_parts = []
    
            if system_message:
                prompt_parts.append(f"System: {system_message}\n")
    
            for msg in self.history:
                prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    
            prompt_parts.append("Assistant:")
            return "\n".join(prompt_parts)
    
        def clear(self):
            """履歴をクリア"""
            self.history.clear()
    
    # 使用例
    conv_manager = ConversationManager(max_history=6)
    
    system_msg = "あなたは親切なAIアシスタントです。"
    
    # 会話シミュレーション
    conversation = [
        ("User", "こんにちは！"),
        ("Assistant", "こんにちは！何かお手伝いできることはありますか？"),
        ("User", "Pythonの学習方法について教えてください。"),
        ("Assistant", "Python学習には、まず基礎文法を学び、その後実際のプロジェクトに取り組むのがおすすめです。"),
        ("User", "初心者向けのプロジェクトは何がいいですか？"),
    ]
    
    print("=== Multi-turn Conversation ===\n")
    
    for i, (role, content) in enumerate(conversation):
        conv_manager.add_message(role, content)
    
        if role == "User":
            print(f"{role}: {content}")
            prompt = conv_manager.get_prompt(system_msg)
            print(f"\n[生成されるプロンプト]")
            print(prompt)
            print("\n" + "="*50 + "\n")
    
    print("✓ 会話履歴により、文脈を保持した対話が可能")
    print(f"✓ 履歴の長さ: {len(conv_manager.history)}メッセージ")
    

### 4\. 完全なチャットボット実装
    
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    class SimpleChatbot:
        """シンプルなチャットボット"""
    
        def __init__(self, model_name="gpt2", max_history=5):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.conversation = ConversationManager(max_history=max_history)
            self.system_message = "You are a helpful AI assistant."
    
        def generate_response(self, user_input, max_length=100):
            """ユーザー入力に対する応答を生成"""
            # 会話履歴に追加
            self.conversation.add_message("User", user_input)
    
            # プロンプト生成
            prompt = self.conversation.get_prompt(self.system_message)
    
            # トークン化
            inputs = self.tokenizer(prompt, return_tensors="pt")
    
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=len(inputs['input_ids'][0]) + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
    
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
            # プロンプト部分を除去
            response = response[len(prompt):].strip()
    
            # 会話履歴に追加
            self.conversation.add_message("Assistant", response)
    
            return response
    
        def chat(self):
            """対話ループ"""
            print("=== チャットボット起動 ===")
            print("終了するには 'quit' と入力してください\n")
    
            while True:
                user_input = input("You: ")
    
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Assistant: さようなら！")
                    break
    
                response = self.generate_response(user_input)
                print(f"Assistant: {response}\n")
    
    # チャットボットのインスタンス化
    chatbot = SimpleChatbot(model_name="gpt2", max_history=5)
    
    # デモ用の対話（実際はchatbot.chat()で対話ループ）
    demo_inputs = [
        "Hello!",
        "What is AI?",
        "Can you explain more?"
    ]
    
    print("=== チャットボット デモ ===\n")
    for user_input in demo_inputs:
        print(f"You: {user_input}")
        response = chatbot.generate_response(user_input, max_length=50)
        print(f"Assistant: {response}\n")
    
    print("✓ 会話履歴を保持した完全なチャットボット")
    print("✓ 文脈を理解した応答生成")
    print("✓ 拡張可能な設計（RAG、Function Calling等を追加可能）")
    

* * *

## 4.6 本章のまとめ

### 学んだこと

  1. **GPTファミリー**

     * Decoder-onlyアーキテクチャ
     * 自己回帰的テキスト生成
     * GPT-2からGPT-4への進化
     * 生成パラメータの制御（temperature、top-k、top-p）
  2. **LLMsの学習手法**

     * 事前学習: 大規模コーパスで次トークン予測
     * Instruction Tuning: 指示に従うモデルへ
     * RLHF: 人間フィードバックによるアライメント
     * PEFT: LoRA、Adapterによる効率的微調整
  3. **Prompt Engineering**

     * Zero-shot / Few-shot Learning
     * Chain-of-Thought Prompting
     * プロンプト設計のベストプラクティス
     * LangChainによるプロンプト管理
  4. **オープンソースLLMs**

     * LLaMA、Falcon、日本語LLMs
     * モデル量子化（8-bit、4-bit）
     * QLoRAによる効率的ファインチューニング
  5. **実践的応用**

     * RAG: 外部知識を活用した生成
     * Function Calling: ツール・API統合
     * Multi-turn Conversation: 文脈保持対話
     * 完全なチャットボット実装

### LLMs活用のベストプラクティス

観点 | 推奨事項  
---|---  
**モデル選択** | タスクに応じたサイズ・性能のバランス  
**プロンプト設計** | 明確な指示、例の提示、構造化  
**メモリ効率** | 量子化、PEFT活用  
**知識更新** | RAGで最新情報を統合  
**安全性** | 出力検証、有害コンテンツフィルタ  
  
### 次のステップ

大規模言語モデルの理解を深めるために：

  * より大規模なモデル（LLaMA-2 13B/70B）を試す
  * 独自データでファインチューニング（LoRA）
  * RAGシステムを本番環境に導入
  * マルチモーダルLLM（GPT-4V等）を探求
  * LLMOps（運用・監視）を学ぶ

* * *

## 演習問題

### 問題1（難易度：easy）

GPTのDecoder-onlyアーキテクチャと、BERTのEncoder-onlyアーキテクチャの違いを説明してください。それぞれどのようなタスクに適していますか？

解答例

**解答** ：

**GPT（Decoder-only）** ：

  * 構造: 自己注意機構 + Causal Masking（未来を見ない）
  * 訓練: 次トークン予測（自己回帰）
  * 強み: テキスト生成、対話、創作
  * 用途: ChatGPT、コード生成、文章作成

**BERT（Encoder-only）** ：

  * 構造: 双方向自己注意機構（全トークンを参照）
  * 訓練: Masked Language Modeling（穴埋め）
  * 強み: テキスト理解、分類、抽出
  * 用途: 感情分析、固有表現認識、質問応答

**使い分け** ：

タスク | 推奨モデル  
---|---  
テキスト生成 | GPT  
文書分類 | BERT  
質問応答（抽出型） | BERT  
質問応答（生成型） | GPT  
要約 | GPT（またはEncoder-Decoder）  
  
### 問題2（難易度：medium）

Chain-of-Thought (CoT) Promptingが、なぜ複雑な推論タスクで効果的なのか説明してください。具体例を挙げてください。

解答例

**解答** ：

**CoTが効果的な理由** ：

  1. **中間推論の明示化** : ステップを言語化することで、LLMが論理的思考を整理
  2. **エラーの早期検出** : 各ステップで誤りに気づきやすい
  3. **複雑性の分解** : 難しい問題を小さな部分問題に分割
  4. **In-context Learning強化** : 推論パターンを学習

**具体例：算数問題**
    
    
    # ❌ 標準プロンプト
    prompt_standard = """
    Question: A store had 25 apples. They sold 8 apples in the morning
    and 12 apples in the afternoon. How many apples are left?
    Answer:
    """
    
    # ✅ CoTプロンプト
    prompt_cot = """
    Question: A store had 25 apples. They sold 8 apples in the morning
    and 12 apples in the afternoon. How many apples are left?
    
    Let's solve this step by step:
    1. The store started with 25 apples.
    2. They sold 8 apples in the morning: 25 - 8 = 17 apples remaining.
    3. They sold 12 apples in the afternoon: 17 - 12 = 5 apples remaining.
    
    Answer: 5 apples are left.
    """
    

**効果の実証** （研究結果より）：

タスク | 標準 | CoT | 改善  
---|---|---|---  
GSM8K（算数） | 17.9% | 58.1% | +40.2%  
SVAMP（算数） | 69.4% | 78.7% | +9.3%  
  
### 問題3（難易度：medium）

LoRA（Low-Rank Adaptation）がなぜパラメータ効率的なのか、数式を用いて説明してください。

解答例

**解答** ：

**LoRAの原理** ：

元の重み行列 $W \in \mathbb{R}^{d \times k}$ を更新する代わりに、低ランク分解を使用：

$$ W' = W + \Delta W = W + BA $$

  * $W$: 元の重み（固定、学習しない）
  * $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: 学習可能な行列
  * $r \ll \min(d, k)$: ランク（例: $r=8$）

**パラメータ削減の計算** ：

例: $d = 4096$, $k = 4096$, $r = 8$ の場合：

  * 元の重み: $4096 \times 4096 = 16,777,216$ パラメータ
  * LoRA: $4096 \times 8 + 8 \times 4096 = 65,536$ パラメータ
  * 削減率: $\frac{65,536}{16,777,216} \approx 0.39\%$（99.6%削減）

**実装例** ：
    
    
    import torch
    import torch.nn as nn
    
    class LoRALayer(nn.Module):
        def __init__(self, in_features, out_features, rank=8):
            super().__init__()
            # 元の重み（固定）
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.weight.requires_grad = False
    
            # LoRA行列（学習可能）
            self.lora_A = nn.Parameter(torch.randn(rank, in_features))
            self.lora_B = nn.Parameter(torch.randn(out_features, rank))
    
            self.rank = rank
    
        def forward(self, x):
            # W*x + B*A*x
            return x @ self.weight.T + x @ self.lora_A.T @ self.lora_B.T
    
    # 例
    layer = LoRALayer(4096, 4096, rank=8)
    total = sum(p.numel() for p in layer.parameters())
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    print(f"総パラメータ: {total:,}")
    print(f"訓練可能: {trainable:,} ({100*trainable/total:.2f}%)")
    

**出力** ：
    
    
    総パラメータ: 16,842,752
    訓練可能: 65,536 (0.39%)
    

### 問題4（難易度：hard）

RAG（Retrieval-Augmented Generation）システムを実装してください。以下の要件を満たすこと：

  * カスタム知識ベースからの検索
  * 関連文書のスコアリング
  * 検索結果を用いた回答生成

解答例
    
    
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SimpleRAG:
        """シンプルなRAGシステム"""
    
        def __init__(self, knowledge_base, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
            self.knowledge_base = knowledge_base
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model)
    
            # 知識ベースの埋め込みを事前計算
            self.kb_embeddings = self._embed_documents(knowledge_base)
    
        def _mean_pooling(self, model_output, attention_mask):
            """平均プーリング"""
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
        def _embed_documents(self, documents):
            """ドキュメントを埋め込みベクトルに変換"""
            encoded = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
    
            with torch.no_grad():
                model_output = self.model(**encoded)
    
            embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            return embeddings.numpy()
    
        def retrieve(self, query, top_k=3):
            """クエリに関連する文書を検索"""
            # クエリの埋め込み
            query_embedding = self._embed_documents([query])
    
            # コサイン類似度計算
            similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
    
            # Top-k文書を取得
            top_indices = np.argsort(similarities)[::-1][:top_k]
    
            results = []
            for idx in top_indices:
                results.append({
                    'document': self.knowledge_base[idx],
                    'score': similarities[idx]
                })
    
            return results
    
        def generate_answer(self, query, retrieved_docs):
            """検索結果を用いて回答を生成（プロンプト作成）"""
            context = "\n".join([f"- {doc['document']}" for doc in retrieved_docs])
    
            prompt = f"""
    Based on the following context, answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
            return prompt
    
    # 知識ベース
    knowledge_base = [
        "機械学習は、コンピュータがデータから学習するAIの分野です。",
        "深層学習は、多層ニューラルネットワークを使用する機械学習の手法です。",
        "Transformerは、2017年に登場した革新的なアーキテクチャです。",
        "BERTは、双方向Transformerを使ったテキスト理解モデルです。",
        "GPTは、Decoder-onlyのTransformerでテキスト生成に優れています。",
        "ファインチューニングは、事前学習モデルを特定タスクに適応させる手法です。",
        "LoRAは、パラメータ効率的なファインチューニング手法です。",
        "RAGは、検索と生成を組み合わせた技術です。"
    ]
    
    # RAGシステムの作成
    rag = SimpleRAG(knowledge_base)
    
    # テスト
    query = "Transformerについて教えてください"
    
    print("=== RAGシステム ===")
    print(f"知識ベース: {len(knowledge_base)}件\n")
    print(f"クエリ: {query}\n")
    
    # 検索
    retrieved = rag.retrieve(query, top_k=3)
    
    print("検索結果:")
    for i, doc in enumerate(retrieved, 1):
        print(f"{i}. [スコア: {doc['score']:.3f}] {doc['document']}")
    
    # 回答生成用プロンプト
    prompt = rag.generate_answer(query, retrieved)
    print(f"\n生成されたプロンプト:\n{prompt}")
    
    print("\n✓ RAGシステムにより、関連知識を検索し回答生成に活用")
    

**出力例** ：
    
    
    === RAGシステム ===
    知識ベース: 8件
    
    クエリ: Transformerについて教えてください
    
    検索結果:
    1. [スコア: 0.712] Transformerは、2017年に登場した革新的なアーキテクチャです。
    2. [スコア: 0.623] BERTは、双方向Transformerを使ったテキスト理解モデルです。
    3. [スコア: 0.589] GPTは、Decoder-onlyのTransformerでテキスト生成に優れています。
    
    生成されたプロンプト:
    
    Based on the following context, answer the question.
    
    Context:
    - Transformerは、2017年に登場した革新的なアーキテクチャです。
    - BERTは、双方向Transformerを使ったテキスト理解モデルです。
    - GPTは、Decoder-onlyのTransformerでテキスト生成に優れています。
    
    Question: Transformerについて教えてください
    
    Answer:
    

### 問題5（難易度：hard）

RLHFの3つのステップ（SFT、Reward Model Training、PPO）について、それぞれの役割と、なぜこの順序で行う必要があるのか説明してください。

解答例

**解答** ：

**Step 1: Supervised Fine-Tuning (SFT)**

  * **役割** : 人間が作成した高品質な対話データでモデルをファインチューニング
  * **目的** : ベースモデルを対話タスクに適応させる
  * **データ** : (プロンプト, 理想的な応答) のペア
  * **必要性** : 事前学習モデルは対話に最適化されていないため

**Step 2: Reward Model Training**

  * **役割** : 人間の選好を学習する報酬モデルを訓練
  * **目的** : 「良い応答」vs「悪い応答」を自動評価できるようにする
  * **データ** : 同じプロンプトに対する複数応答のランキング
  * **必要性** : 強化学習に必要な報酬信号を提供

**損失関数** ：

$$ \mathcal{L}_{\text{reward}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right] $$

  * $y_w$: 好まれる応答
  * $y_l$: 好まれない応答
  * $r(x, y)$: 報酬スコア

**Step 3: PPO (Proximal Policy Optimization)**

  * **役割** : 報酬モデルを使ってモデルを最適化
  * **目的** : 人間の選好に沿った応答を生成するよう学習
  * **手法** : 強化学習（PPOアルゴリズム）
  * **制約** : 元のモデルから大きく逸脱しないようKL正則化

**目的関数** ：

$$ \mathcal{L}^{\text{PPO}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ r(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{SFT}}) \right] $$

**なぜこの順序が必要か** ：

  1. **SFTが最初** : 
     * ベースモデルは対話に不慣れ
     * 強化学習の初期方策として機能
     * 報酬モデル訓練のデータ生成にも使用
  2. **Reward Modelが2番目** : 
     * PPOに必要な報酬信号を提供
     * 人間のランキングデータから学習
     * SFTモデルで生成した応答を評価
  3. **PPOが最後** : 
     * 報酬モデルがないと最適化できない
     * SFTモデルを初期方策として使用
     * KL正則化でSFTから大きく逸脱しない

**全体のフロー** ：
    
    
    ```mermaid
    graph TD
        A[事前学習モデル] --> B[SFT]
        B --> C[SFTモデル]
        C --> D[応答生成]
        D --> E[人間によるランキング]
        E --> F[報酬モデル訓練]
        F --> G[報酬モデル]
        C --> H[PPO]
        G --> H
        H --> I[アライメント済みモデルChatGPT]
    ```

**効果** ：

段階 | 性能指標  
---|---  
ベースモデル | 対話品質: 低  
SFT | 対話品質: 中（指示に従う）  
RLHF (PPO) | 対話品質: 高（人間の選好に沿う）  
  
* * *

## 参考文献

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
