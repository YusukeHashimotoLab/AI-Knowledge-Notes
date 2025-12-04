---
title: Active Learning Introductory Series v1.0
chapter_title: Active Learning Introductory Series v1.0
subtitle: Strategic exploration to find optimal solutions with fewer experiments
reading_time: 100-120 minutes
difficulty: Intermediate to Advanced
code_examples: 28
exercises: 12
version: 1.0
created_at: 2025-10-18
---

## Series Overview

This series is educational content with a 4-chapter structure that allows learners to progress from beginners just starting to learn Active Learning to those who want to develop practical Materials Exploration skills.

Active Learning is a Machine Learning technique that actively selects data with the highest information value through a limited number of experiments. In Materials Exploration, by intelligently deciding which samples to measure next, you can achieve target performance with one-tenth or fewer experiments compared to Random Sampling. Toyota's Catalyst development achieved an 80% reduction in experiments, while MIT's battery Materials Exploration increased development speed 10-fold.

### Why This Series is Necessary

**Background and Challenges** : The greatest challenge in Materials Science is the vastness of the search space and the high cost of experiments. For example, Catalyst screening involves tens of thousands of candidate materials, and evaluating a single sample can take days to weeks. Measuring all samples is physically and economically impossible. Traditional Random Sampling wastes valuable experimental resources on low-information-value samples.

**What You Will Learn in This Series** : This series systematically teaches Active Learning from theory to practice through executable Code Examples and Materials Science case studies. You will acquire practical skills from day one, including Query Strategies (data selection strategy), Uncertainty Estimation techniques, Acquisition Function design, and automatic integration with experimental equipment.

**Features:**

  * ✅ **Practice-Focused** : 28 executable Code Examples and 5 detailed case studies
  * ✅ **Progressive Structure** : 4 chapters comprehensively covering from fundamentals to applications
  * ✅ **Materials Science Specialization** : Focus on application to Materials Exploration rather than generic ML theory
  * ✅ **Latest Tools** : Covers industry-standard tools like modAL, GPyTorch, and BoTorch
  * ✅ **Theory and Implementation** : Combines both formula-based formulation and Python implementation
  * ✅ **Robotics Integration** : Explains integration methods with automated experimental equipment

**Target Audience** :

  * Graduate students and researchers (those wanting to learn efficient Materials Exploration)
  * Corporate R&D engineers (those wanting to reduce experiment count and costs)
  * Data scientists (those wanting to learn both theory and practice of Active Learning)
  * Bayesian Optimization experienced (those wanting to acquire more advanced exploration strategies)

## How to Learn

### Recommended Learning Order
    
    
    ```mermaid
    flowchart TD
        A["Chapter 1: Why Active Learning is Needed"] --> B["Chapter 2: Uncertainty Estimation Techniques"]
        B --> C["Chapter 3: Acquisition Function Design"]
        C --> D["Chapter 4: Application to Materials Exploration"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

**For Beginners (No prior Active Learning knowledge)** :

  * Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 (all chapters recommended)
  * Time Required: 100-120 minutes
  * Prerequisites: ML fundamentals, Bayesian Optimization introduction

**For Intermediate Learners (Bayesian Optimization experience)** :

  * Chapter 2 → Chapter 3 → Chapter 4
  * Time Required: 70-90 minutes
  * Chapter 1 can be skipped

**For Practical Skill Enhancement (Implementation-focused over theory)** :

  * Chapter 3 (intensive learning) → Chapter 4
  * Time Required: 50-70 minutes
  * Reference Chapter 2 as needed for theory

### Learning Flowchart
    
    
    ```mermaid
    flowchart TD
        Start["Start Learning"] --> Q1{"Bayesian OptimizationExperience?"}
        Q1 -->|First time| PreBO["Prerequisite: BOIntroductory Series"]
        Q1 -->|Experienced| Q2{"Active LearningExperience?"}
    
        PreBO --> Ch1
        Q2 -->|First time| Ch1["Start from Chapter 1"]
        Q2 -->|Basic knowledge| Ch2["Start from Chapter 2"]
        Q2 -->|Implementationexperience| Ch3["Start from Chapter 3"]
    
        Ch1 --> Ch2["Go to Chapter 2"]
        Ch2 --> Ch3["Go to Chapter 3"]
        Ch3 --> Ch4["Go to Chapter 4"]
        Ch4 --> Complete["Series Complete"]
    
        Complete --> Next["Next Steps"]
        Next --> Project["Personal Project"]
        Next --> Robotic["Robotics ExperimentAutomation"]
        Next --> Community["ResearchCommunity"]
    
        style Start fill:#4CAF50,color:#fff
        style Complete fill:#2196F3,color:#fff
        style Next fill:#FF9800,color:#fff
    ```

## Chapter Details

### Chapter 1: The Need for Active Learning

📖 Reading Time: 20-25 minutes 📊 Difficulty: Intermediate 💻 Code Examples: 6-8

#### Learning Content

  * **What is Active Learning** : Definition, Passive Learning vs Active Learning, application areas
  * **Fundamentals of Query Strategies** : Uncertainty Sampling, Diversity Sampling, Expected Model Change, Query-by-Committee
  * **Exploration vs Exploitation** : Trade-offs, epsilon-greedy approach, UCB
  * **Case Study: Catalyst Activity Prediction** : Random Sampling vs Active Learning

#### Learning Objectives

  * ✅ Explain the definition and advantages of Active Learning
  * ✅ Understand the 4 main Query Strategies techniques
  * ✅ Explain the trade-off between Exploration and Exploitation
  * ✅ Name 3 or more successful examples in Materials Science
  * ✅ Perform quantitative comparison with Random Sampling

**[Read Chapter 1 →](<chapter-1.html>)**

### Chapter 2: Uncertainty Estimation Techniques

📖 Reading Time: 25-30 minutes 📊 Difficulty: Intermediate to Advanced 💻 Code Examples: 7-9

#### Learning Content

  * **Uncertainty Estimation via Ensemble Methods** : Bagging/Boosting, prediction variance calculation, implementation with Random Forest/LightGBM
  * **Uncertainty Estimation via Dropout** : MC Dropout, uncertainty in Neural Networks, Bayesian Neural Networks
  * **Uncertainty via Gaussian Process (GP)** : GP fundamentals, kernel functions, prediction mean and variance, GPyTorch implementation
  * **Case Study: Band Gap Prediction** : Comparison of 3 techniques, verification of experiment reduction effects

#### Learning Objectives

  * ✅ Understand the principles of 3 Uncertainty Estimation techniques
  * ✅ Implement Ensemble methods (Random Forest)
  * ✅ Apply MC Dropout to Neural Networks
  * ✅ Calculate prediction variance with Gaussian Process
  * ✅ Explain criteria for selecting appropriate techniques

#### Uncertainty Estimation Flow
    
    
    ```mermaid
    flowchart TD
        A["Training Data"] --> B{"ModelSelection"}
        B -->|Ensemble| C["Random Forest/LightGBM"]
        B -->|Deep Learning| D["MC Dropout"]
        B -->|GP| E["Gaussian Process"]
    
        C --> F["CalculatePrediction Variance"]
        D --> F
        E --> F
    
        F --> G["Select Samples withHigh Uncertainty"]
        G --> H["Experiment Execution"]
        H --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style G fill:#e8f5e9
    ```

**[Read Chapter 2 →](<chapter-2.html>)**

### Chapter 3: Acquisition Function Design

📖 Reading Time: 25-30 minutes 📊 Difficulty: Intermediate to Advanced 💻 Code Examples: 6-8

#### Learning Content

  * **Fundamentals of Acquisition Functions** : Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), Thompson Sampling
  * **Multi-objective Acquisition Functions** : Pareto optimality, Expected Hypervolume Improvement, simultaneous optimization of multiple properties
  * **Constrained Acquisition Functions** : Synthesizability constraint, cost constraint, Constrained Expected Improvement
  * **Case Study: Thermoelectric Materials Exploration** : ZT value maximization, multi-objective optimization, exploration considering synthesizability

#### Learning Objectives

  * ✅ Understand characteristics of 4 main Acquisition Functions
  * ✅ Implement Expected Improvement
  * ✅ Apply Pareto optimality to multi-objective optimization
  * ✅ Incorporate constraints into Acquisition Functions
  * ✅ Explain criteria for selecting Acquisition Functions

#### Acquisition Function Comparison

Acquisition Function | Characteristics | Exploration Tendency | Computation Cost | Recommended Use  
---|---|---|---|---  
EI | Expected Improvement | Balanced | Medium | General Optimization  
PI | Probability of Improvement | Exploitation-focused | Low | Fast Exploration  
UCB | Upper Confidence Bound | Exploration-focused | Low | Wide-range Search  
Thompson | Probabilistic | Balanced | Medium | Parallel Experiments  
  
**[Read Chapter 3 →](<chapter-3.html>)**

### Chapter 4: Applications and Practice in Materials Exploration

📖 Reading Time: 25-30 minutes 📊 Difficulty: Advanced 💻 Code Examples: 6-8

#### Learning Content

  * **Active Learning × Bayesian Optimization** : Integration with Bayesian Optimization, BoTorch implementation, continuous vs discrete space
  * **Active Learning × High-Throughput Computing** : DFT calculation efficiency, prioritization considering computational cost, Batch Active Learning
  * **Active Learning × Experimental Robotics** : Closed-loop optimization, autonomous experimental systems, feedback loop design
  * **Real-World Applications and Career Paths** : Examples from Toyota, MIT, Citrine Informatics, career paths

#### Learning Objectives

  * ✅ Understand integration methods of Active Learning and Bayesian Optimization
  * ✅ Apply optimization to high-throughput computing
  * ✅ Design closed-loop systems
  * ✅ Gain practical knowledge from 5 industrial application examples
  * ✅ Develop concrete career path plans

#### Closed-Loop Optimization
    
    
    ```mermaid
    flowchart LR
        A["Candidate ProposalActive Learning"] --> B["Experiment ExecutionRobotics"]
        B --> C["Measurement &EvaluationSensors"]
        C --> D["Data AccumulationDatabase"]
        D --> E["Model UpdateMachine Learning"]
        E --> F["Acquisition FunctionEvaluation &Next Candidate"]
        F --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#fce4ec
    ```

**[Read Chapter 4 →](<chapter-4.html>)**

## Overall Learning Outcomes

Upon completing this series, you will acquire the following skills and knowledge:

### Knowledge Level (Understanding)

  * ✅ Explain the definition and theoretical foundations of Active Learning
  * ✅ Understand characteristics and appropriate use of 4 Query Strategy types
  * ✅ Compare 3 Uncertainty Estimation techniques (Ensemble, Dropout, GP)
  * ✅ Understand principles of Acquisition Function design
  * ✅ Detail 5 or more real-world success examples

### Practical Skills (Doing)

  * ✅ Implement basic Active Learning with modAL
  * ✅ Implement 3 types of Uncertainty Estimation techniques
  * ✅ Customize Acquisition Functions
  * ✅ Apply optimization to real materials data and evaluate results
  * ✅ Build closed-loop systems

### Application Ability (Applying)

  * ✅ Select appropriate strategies for new Materials Exploration problems
  * ✅ Design automatic integration with experimental equipment
  * ✅ Evaluate industrial implementation examples and apply to your research
  * ✅ Handle multi-objective and constrained optimization

## FAQ (Frequently Asked Questions)

### Q1: What is the difference between Active Learning and Bayesian Optimization?

**A** : Active Learning and Bayesian Optimization are closely related but have different focuses:

  * **Active Learning** : Goal is efficient learning of Machine Learning models, focus is which data to acquire next (Query Strategy)
  * **Bayesian Optimization** : Goal is maximizing/minimizing objective functions, focus is exploring for optimal solutions (Exploration-Exploitation)

**Commonality** : Both perform "intelligent sampling leveraging uncertainty". Bayesian Optimization can be viewed as a special case of Active Learning.

### Q2: Can I understand this with limited Machine Learning experience?

**A** : Yes, if you have basic Machine Learning knowledge (linear regression, decision trees, cross-validation, etc.). However, we recommend the following prerequisites:

  * **Required** : Fundamentals of supervised learning, Python basics (NumPy, pandas), basic statistics
  * **Recommended** : Bayesian Optimization introductory series, experience with scikit-learn

### Q3: Which Uncertainty Estimation technique should I choose?

**A** : Choose based on problem characteristics and available resources:

  * **Ensemble Methods (Random Forest)** : Simple implementation, moderate computational cost, strong with tabular data. Not suitable for high dimensions.
  * **MC Dropout** : Applicable to Deep Learning models, easy integration with existing neural networks. Relatively higher computational cost.
  * **Gaussian Process** : Rigorous uncertainty quantification, high accuracy with small data. Not suitable for large-scale data.

**Recommendation** : Start with Ensemble methods, then transition to GP or Dropout as needed.

### Q4: Can I learn without experimental equipment?

**A** : **Yes, you can**. This series teaches fundamentals with simulation data, provides practice with open datasets (Materials Project, etc.), and teaches closed-loop concepts and code examples. You will acquire knowledge that can be immediately applied when you use experimental equipment in the future.

### Q5: Are there any industrial applications with proven results?

**A** : Many successful examples exist:

  * **Toyota** : Catalyst reaction condition optimization, 80% reduction in experiments (1,000 → 200)
  * **MIT** : Li-ion battery electrolyte exploration, 10-fold increase in development speed
  * **BASF** : Process condition optimization, 30 million euros annual cost savings
  * **Citrine Informatics** : Active Learning specialist startup, 50+ customers

## Prerequisites and Related Series

### Prerequisites

**Required** :

  * Python fundamentals: variables, functions, classes, NumPy, pandas
  * Machine Learning fundamentals: supervised learning, cross-validation, overfitting
  * Basic statistics: normal distribution, mean, variance, standard deviation

**Strongly Recommended** :

  * Bayesian Optimization introduction: Gaussian Process, Acquisition Function, Exploration-Exploitation

### Complete Learning Path
    
    
    ```mermaid
    flowchart TD
        Pre1["Prerequisite:Python Basics"] --> Pre2["Prerequisite:Materials InformaticsIntroduction"]
        Pre2 --> Pre3["Prerequisite:Bayesian OptimizationIntroduction"]
        Pre3 --> Current["Active LearningIntroduction"]
    
        Current --> Next1["Next: RoboticsExperiment Automation"]
        Current --> Next2["Next: ReinforcementLearning Introduction"]
        Current --> Next3["Application: RealMaterials ExplorationProject"]
    
        Next1 --> Advanced["Advanced: AutonomousExperimental Systems"]
        Next2 --> Advanced
        Next3 --> Advanced
    
        style Pre1 fill:#e3f2fd
        style Pre2 fill:#e3f2fd
        style Pre3 fill:#fff3e0
        style Current fill:#4CAF50,color:#fff
        style Next1 fill:#f3e5f5
        style Next2 fill:#f3e5f5
        style Next3 fill:#f3e5f5
        style Advanced fill:#ffebee
    ```

## Key Tools

Tool Name | Purpose | License | Installation  
---|---|---|---  
modAL | Active Learning specialized library | MIT | `pip install modAL-python`  
scikit-learn | Machine Learning foundation | BSD-3 | `pip install scikit-learn`  
GPyTorch | Gaussian Process (GPU-compatible) | MIT | `pip install gpytorch`  
BoTorch | Bayesian Optimization (PyTorch) | MIT | `pip install botorch`  
pandas | Data management | BSD-3 | `pip install pandas`  
matplotlib | Visualization | PSF | `pip install matplotlib`  
numpy | Numerical computation | BSD-3 | `pip install numpy`  
  
## Next Steps

### Recommended Actions After Series Completion

**Immediate (Within 1-2 weeks):**

  * ✅ Create a portfolio on GitHub
  * ✅ Implement a Catalyst exploration project using modAL
  * ✅ Add "Active Learning" skill to LinkedIn profile
  * ✅ Write learning articles on Qiita/Zenn

**Short-term (1-3 months):**

  * ✅ Advance to the Robotics Experiment Automation introductory series
  * ✅ Execute your own Materials Exploration project
  * ✅ Participate in Materials Science study groups/conferences
  * ✅ Participate in Kaggle competitions (Materials Science)
  * ✅ Build a closed-loop system

## Let's Get Started!

Are you ready? Start from Chapter 1 and begin your journey to revolutionize Materials Exploration with Active Learning!

[Chapter 1: Why Active Learning is Needed →](<chapter-1.html>)
