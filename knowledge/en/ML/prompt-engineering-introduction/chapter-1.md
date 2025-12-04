---
title: "Chapter 1: Fundamentals of Prompt Engineering"
chapter_title: "Chapter 1: Fundamentals of Prompt Engineering"
subtitle: Principles and Practical Techniques for Effective Prompt Design
reading_time: 20-25 minutes
difficulty: Beginner
exercises: 5
---

## What is Prompt Engineering

**Prompt Engineering** is the technique of designing effective inputs (prompts) to elicit desired outputs from Large Language Models (LLMs).

While **LLMs** like ChatGPT, Claude, and Gemini are extremely powerful, the appropriate "way of asking questions" and "giving instructions" is crucial to maximizing their capabilities. Even for the same question, the quality of the response can vary greatly depending on how the prompt is written.

#### 🎯 Why Prompt Engineering Matters

  * **Quality Improvement** : Obtain more accurate and detailed responses
  * **Efficiency** : Reduce trial-and-error time
  * **Reproducibility** : Consistently obtain outputs of the same quality
  * **Cost Reduction** : Optimize API token usage
  * **Expanded Applications** : Handle more complex tasks

## Six Characteristics of Good Prompts

Effective prompts have the following six characteristics:

### 1\. Clarity

Eliminate ambiguity and clearly indicate what is being requested.

#### ❌ Bad Example
    
    
    Tell me about AI

**Problem** : Scope too broad, unclear what information is desired

#### ✅ Good Example
    
    
    Explain the differences between supervised learning and unsupervised learning
    in machine learning, including concrete examples, in three key points.

**Improvement** : Topic, format, and level of detail are clear

### 2\. Specificity

Specifically specify the expected output format, length, and style.

#### ❌ Bad Example
    
    
    Write an email

#### ✅ Good Example
    
    
    Create an email to a client regarding sending a proposal.
    
    【Requirements】
    - Recipient: Mr. Tanaka (Department Manager, ABC Corporation)
    - Purpose: Send proposal for new system implementation
    - Tone: Formal and polite
    - Length: 200-300 words
    - Include: Greeting, key points of proposal, next steps

### 3\. Context

Provide background information and prerequisites for the task.

#### 💡 Prompt Example: Including Context
    
    
    You are an experienced Python programmer.
    You are creating programming materials for beginners.
    
    Explain Python list comprehensions in the following format:
    1. Basic concept (within 100 words)
    2. Simple code example
    3. Common use cases (3 examples)
    4. Common mistakes for beginners

### 4\. Constraints

Explicitly state limitations and requirements for the output.

#### 💡 Prompt Example: With Constraints
    
    
    Create a product description with the following constraints:
    
    【Constraints】
    - Length: Within 150 words
    - Tone: Casual and friendly
    - Required keywords: "eco-friendly", "durability", "value"
    - Avoid: Exaggerated claims, technical jargon
    - Target audience: Working professionals in their 20s-30s
    
    【Product】Stainless steel tumbler

### 5\. Examples

Clarify format and quality standards by showing examples of expected output.

#### 💡 Prompt Example: Few-shot Learning
    
    
    Perform sentiment analysis following the examples below.
    
    Example 1:
    Input: Today was the best day ever!
    Output: Positive (joy, satisfaction)
    
    Example 2:
    Input: Plans got cancelled due to rain, disappointing
    Output: Negative (disappointment, sadness)
    
    Example 3:
    Input: I wonder if it will be sunny tomorrow
    Output: Neutral (anticipation, uncertainty)
    
    Now analyze the following sentence:
    Input: A new project is starting but I'm anxious

### 6\. Step-by-step Instructions

Break down complex tasks into step-by-step instructions.

#### 💡 Prompt Example: Step-by-step Instructions
    
    
    Create a data analysis report following these steps:
    
    【Step 1】Review data overview
    - Dataset size (rows, columns)
    - Data types of each column
    - Presence of missing values
    
    【Step 2】Calculate basic statistics
    - Mean, median, standard deviation for numeric columns
    - Frequency of categorical columns
    
    【Step 3】Detect anomalies
    - Identify outliers
    - Infer their causes
    
    【Step 4】Conclusions and recommendations
    - Key findings (3 points)
    - Suggested next actions

## Zero-shot and Few-shot Learning

### Zero-shot Learning

**Zero-shot** is a method of directly instructing tasks without examples. It's suitable for simple tasks or general questions.

#### 💡 Zero-shot Prompt Example
    
    
    Translate the following sentence into English:
    
    "Machine learning is the ability of computers to learn from data and perform tasks."

### Few-shot Learning

**Few-shot** is a method of performing tasks after showing several examples. It's effective when formats are specialized or when you want to clarify quality standards.

#### 💡 Few-shot Prompt Example (2-shot)
    
    
    Convert product names to shortened forms following the examples:
    
    Example 1:
    Input: Super Premium High-Function Multi-Purpose Hybrid Vacuum Cleaner
    Output: Premium Vacuum
    
    Example 2:
    Input: Ultra Slim Design Wireless Charging Compatible Smartphone
    Output: Slim Smartphone
    
    Now convert the following:
    Input: Extra Long Battery Portable Bluetooth Speaker

### One-shot vs Few-shot vs Many-shot

Method | Number of Examples | Advantages | Disadvantages | Best Suited For  
---|---|---|---|---  
**Zero-shot** | 0 | Simple, token-saving | May have lower accuracy | General tasks  
**One-shot** | 1 | Can show format | Insufficient for pattern learning | Simple format specification  
**Few-shot** | 2-5 | Pattern learning, high accuracy | Increased tokens | Complex tasks  
**Many-shot** | 6+ | Highest accuracy | Heavy token consumption | When highest accuracy is essential  
  
## Chain of Thought (CoT) Prompting

**Chain of Thought** is a technique that encourages step-by-step reasoning in LLMs. By adding the instruction "think step by step," you can obtain more logical and accurate responses.

### Basic CoT Prompt

#### ❌ Without CoT
    
    
    Q: There are 5 apples and 3 oranges.
    We used 2 apples and 1 orange.
    How many fruits are left in total?

**Problem** : Calculation process unclear, prone to errors

#### ✅ With CoT
    
    
    Q: There are 5 apples and 3 oranges.
    We used 2 apples and 1 orange.
    How many fruits are left in total?
    
    Think step by step.

**Expected Output** :
    
    
    Step 1: Check initial fruit count
    - Apples: 5
    - Oranges: 3
    - Total: 5 + 3 = 8
    
    Step 2: Subtract used fruits
    - Apples: 5 - 2 = 3
    - Oranges: 3 - 1 = 2
    
    Step 3: Calculate remaining total
    - Remaining: 3 + 2 = 5
    
    Answer: 5 fruits

### Few-shot CoT Prompt

#### 💡 Few-shot CoT Prompt Example
    
    
    Solve the problem following the example below:
    
    Example:
    Q: At a cafe, I ordered 2 coffees (450 yen each) and 1 cake (600 yen).
    I'm paying with two 1000 yen bills. How much change will I get?
    
    A: Let's think step by step.
    1. Cost of 2 coffees: 450 yen × 2 = 900 yen
    2. Total amount: 900 yen + 600 yen = 1500 yen
    3. Payment amount: 1000 yen × 2 = 2000 yen
    4. Change: 2000 yen - 1500 yen = 500 yen
    Answer: 500 yen
    
    Now solve this problem:
    Q: At a bookstore, I bought 3 books (1200 yen each) and 2 magazines (800 yen each).
    I'm paying with a 5000 yen bill. How much change will I get?

### When CoT is Particularly Effective

  * **Mathematical Reasoning** : Calculation problems, logic puzzles
  * **Complex Decision Making** : Judgments considering multiple factors
  * **Causal Analysis** : Explaining cause-and-effect relationships
  * **Multi-step Problem Solving** : Tasks requiring multiple steps
  * **Debugging** : Identifying error causes in code or processes

## Practical Prompt Templates

Here are ready-to-use practical prompt templates.

### Template 1: Task Execution Type
    
    
    【Role】You are an expert in [field of expertise].
    
    【Task】[Specific task description]
    
    【Input】
    [Data or information to process]
    
    【Output Format】
    - [Format 1]
    - [Format 2]
    - [Format 3]
    
    【Constraints】
    - [Constraint 1]
    - [Constraint 2]
    
    【Example】(Optional)
    [Example of expected output]

### Template 2: Analysis Type
    
    
    Analyze the following data:
    
    【Data】
    [Data to analyze]
    
    【Analysis Perspectives】
    1. [Perspective 1]
    2. [Perspective 2]
    3. [Perspective 3]
    
    【Required Output】
    1. Key findings (3 points)
    2. Observable trends from data
    3. Recommended next actions
    
    Analyze step by step.

### Template 3: Creative Type
    
    
    Create content with the following conditions:
    
    【Type】[Blog post / Email / Presentation, etc.]
    
    【Theme】[Main topic]
    
    【Target Audience】
    - Age range: [Age range]
    - Knowledge level: [Beginner / Intermediate / Expert]
    - Interests: [Interests or challenges]
    
    【Tone】[Formal / Casual / Professional, etc.]
    
    【Structure】
    1. [Section 1]
    2. [Section 2]
    3. [Section 3]
    
    【Length】[Word count or character count]
    
    【Required Elements】
    - [Element to include 1]
    - [Element to include 2]

### Template 4: Code Generation Type
    
    
    Create Python code with the following specifications:
    
    【Function】[Description of function to implement]
    
    【Input】[Function arguments or input data]
    
    【Output】[Expected output]
    
    【Requirements】
    - Programming language: Python 3.8+
    - Libraries to use: [Library name]
    - Error handling: Required
    - Comments: Japanese docstrings for each function
    
    【Example】
    Input example: [Concrete example]
    Expected output: [Result example]
    
    Provide code and usage example.

## Common Failure Patterns and Improvements

### Failure Pattern 1: Vague Instructions

#### ❌ Before Improvement
    
    
    Write a business email

#### ✅ After Improvement
    
    
    Create an email introducing a new product to a business partner.
    
    【Requirements】
    - Recipient: Sales Manager
    - Purpose: Introduce new product "XYZ" and obtain business meeting appointment
    - Tone: Formal yet friendly
    - Length: About 300 words
    - Include: Greeting, product features (3 points), specific date proposal

### Failure Pattern 2: Multiple Tasks at Once

#### ❌ Before Improvement
    
    
    Review this code, fix bugs,
    improve performance, and write documentation too

#### ✅ After Improvement
    
    
    【Task 1】First, identify bugs in this code:
    [Code]
    
    Once bugs are found, we'll proceed to the next task.

Then request tasks separately in sequence

### Failure Pattern 3: Lack of Context

#### ❌ Before Improvement
    
    
    Improve this text:
    "Developed a product"

#### ✅ After Improvement
    
    
    【Background】
    Creating a press release for a startup.
    Target audience is investors and media representatives.
    
    【Text to Improve】
    "Developed a product"
    
    【Improvement Direction】
    - More specific and impressive expression
    - Emphasize product innovation
    - Capture reader's interest
    - Professional tone

### Failure Pattern 4: Unspecified Output Format

#### ❌ Before Improvement
    
    
    Explain Python's main features

#### ✅ After Improvement
    
    
    Explain Python's main features in the following format:
    
    【Output Format】
    For each feature:
    1. Feature name
    2. Brief description (within 50 words)
    3. Code example (within 5 lines)
    4. Usage scenarios
    
    【Target Features】
    - List comprehension
    - Decorators
    - Generators
    
    Explain in a beginner-friendly manner.

## Best Practices for Prompt Design

#### 💡 Practical Tips

  * **Iterative Improvement** : Don't expect perfection on first attempt, improve while reviewing outputs
  * **Templatize** : Create templates for frequently used prompts and reuse
  * **Version Control** : Record and manage effective prompts
  * **A/B Testing** : Try multiple prompts and select the best one
  * **Leverage Feedback** : Learn from LLM outputs and improve prompts

### Prompt Design Checklist

  * ☐ Is the task objective clear?
  * ☐ Have you provided necessary context information?
  * ☐ Have you specifically specified output format?
  * ☐ Have you stated constraints?
  * ☐ Have you broken down complex tasks into steps?
  * ☐ Have you included examples when necessary?
  * ☐ Have you specified tone and style?
  * ☐ Have you clarified target audience and usage scenario?

## Chapter Summary

### 🎯 Key Points

  * **Clarity is paramount** : Eliminate ambiguity and give specific instructions
  * **Six characteristics** : Clarity, specificity, context, constraints, examples, step-by-step instructions
  * **Zero-shot vs Few-shot** : Choose based on task complexity
  * **Chain of Thought** : Add "think step by step" for complex reasoning
  * **Leverage Templates** : Reuse effective patterns for efficiency
  * **Iterative Improvement** : Don't aim for perfection initially, improve while reviewing outputs

## Exercises

Practice the knowledge you've learned with the following exercises.

### Exercise 1: Improve Prompts (Difficulty: ★☆☆)

**Problem** : Improve the following vague prompt based on the six characteristics.
    
    
    Tell me a recipe

**Hint** : Clarify dish name, number of servings, cooking time, difficulty, and output format.

### Exercise 2: Create Few-shot Prompt (Difficulty: ★★☆)

**Problem** : Create a Few-shot prompt for sentiment analysis of product reviews (Positive/Negative/Neutral). Include at least 3 examples.

### Exercise 3: Chain of Thought Prompt (Difficulty: ★★☆)

**Problem** : Create a CoT prompt to solve the following problem.
    
    
    Problem: A company has 120 employees.
    Of these, 40% work in sales, 30% in development, and the rest in administration.
    If the development department increases by 10 people, what percentage will the development department represent?

### Exercise 4: Task Execution Prompt (Difficulty: ★★★)

**Problem** : Design a comprehensive prompt template for "replying to customer complaint emails." Include role, task, constraints, and output format.

### Exercise 5: Practical Application (Difficulty: ★★★)

**Problem** : Create one prompt that can be used in your actual work or studies, test it with an LLM, evaluate the results, and list three areas for improvement.

## Next Steps

In Chapter 1, you learned the fundamentals of prompt engineering. In the next chapters, you'll learn more advanced techniques:

  * **Chapter 2** (Coming Soon): Basic techniques like Role Prompting, Context Setting, Output Format
  * **Chapter 3** (Coming Soon): Applied techniques like Tree of Thought, Self-Consistency
  * **Chapter 4** (Coming Soon): Task-specific optimization methods (summarization, translation, code generation, etc.)
  * **Chapter 5** (Coming Soon): Practical projects and prompt library development

[← Series Top](<./index.html>) [Chapter 2 (Coming Soon) →](<#>)

* * *

## References

  * OpenAI. (2023). _GPT Best Practices_
  * Wei, J., et al. (2022). _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_
  * Brown, T., et al. (2020). _Language Models are Few-Shot Learners_
  * Anthropic. (2024). _Prompt Engineering Guide_

* * *

**Update History**

  * **2025-12-01** : v1.0 Initial release
