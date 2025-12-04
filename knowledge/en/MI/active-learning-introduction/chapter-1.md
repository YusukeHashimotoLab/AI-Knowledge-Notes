---
title: "Chapter 1: The Need for Active Learning"
chapter_title: "Chapter 1: The Need for Active Learning"
subtitle: Dramatically Reduce Experiment Count Through Active Data Selection
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 7
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 1: The Need for Active Learning

This chapter covers The Need for Active Learning. You will learn four main query strategy techniques and exploration-exploitation tradeoff.

**Dramatically Reduce Experiment Count Through Active Data Selection**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Explain the definition and advantages of Active Learning
  * ✅ Understand the four main query strategy techniques
  * ✅ Explain the exploration-exploitation tradeoff
  * ✅ Provide three or more successful examples in materials science
  * ✅ Perform quantitative comparisons with random sampling

**Reading Time** : 20-25 minutes **Code Examples** : 7 **Exercises** : 3

* * *

## 1.1 What is Active Learning?

### Definition: Efficient Learning Through Active Data Selection

**Active Learning** is a method where machine learning models actively select "which data to acquire next," enabling the construction of high-accuracy models with minimal training data.

**Differences from Passive Learning** :

Aspect | Passive Learning | Active Learning  
---|---|---  
Data Selection | Random or existing datasets | Actively selected by model  
Learning Efficiency | Low (requires large data) | High (high accuracy with small data)  
Data Acquisition Cost | Not considered | Considered  
Application Scenarios | Data is inexpensive | Data is expensive  
  
**Importance in Materials Science** : \- Single experiments take days to weeks \- High experimental costs (catalyst synthesis, DFT calculations, etc.) \- Vast search spaces (10^6 to 10^60 candidates)

### Basic Active Learning Cycle
    
    
    ```mermaid
    flowchart LR
        A["Initial DataFew samples"] --> B["Model TrainingBuild prediction model"]
        B --> C["Candidate EvaluationQuery Strategy"]
        C --> D["Select mostinformative sample"]
        D --> E["Experiment ExecutionData Acquisition"]
        E --> F{"Stopping criteria?Goal achieved orbudget limit"}
        F -->|No| B
        F -->|Yes| G["Final Model"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style G fill:#4CAF50,color:#fff
    ```

**Key Points** : 1\. **Start with small initial data** (typically 10-20 samples) 2\. **Intelligently select next sample** using query strategy 3\. **Execute experiments** adding data one at a time 4\. **Repeat model updates** 5\. **Continue until goal achieved**

* * *

## 1.2 Query Strategy Fundamentals

### 1.2.1 Uncertainty Sampling

**Principle** : Select samples where the model's prediction is most uncertain

**Formula** : $$ x^* = \arg\max_{x \in \mathcal{U}} \text{Uncertainty}(x) $$

where $\mathcal{U}$ is the set of unlabeled samples

**Uncertainty Measurement Methods** :

**Regression Problems** : $$ \text{Uncertainty}(x) = \sigma(x) $$ (standard deviation of prediction)

**Classification Problems (2-class)** : $$ \text{Uncertainty}(x) = 1 - |P(y=1|x) - P(y=0|x)| $$ (inverse of absolute probability difference; closer to 0.5 means more uncertain)

**Code Example 1: Uncertainty Sampling Implementation**
    
    
    __PROTECTED_CODE_0__

**Output** :
    
    
    __PROTECTED_CODE_1__

**Advantages** : \- ✅ Simple and intuitive \- ✅ Low computational cost \- ✅ Effective for many problems

**Disadvantages** : \- ⚠️ Does not consider diversity of search space \- ⚠️ May be biased toward local regions

* * *

### 1.2.2 Diversity Sampling

**Principle** : Select samples that are different (diverse) from existing data

**Formula** : $$ x^* = \arg\max_{x \in \mathcal{U}} \min_{x_i \in \mathcal{L}} d(x, x_i) $$

where $\mathcal{L}$ is the set of labeled samples, and $d(\cdot, \cdot)$ is a distance function

**Distance Measurement Methods** : \- Euclidean distance: $d(x_i, x_j) = |x_i - x_j|_2$ \- Mahalanobis distance: $d(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)}$ \- Cosine distance: $d(x_i, x_j) = 1 - \frac{x_i \cdot x_j}{|x_i| |x_j|}$

**Code Example 2: Diversity Sampling Implementation**
    
    
    __PROTECTED_CODE_2__

**Output** :
    
    
    __PROTECTED_CODE_3__

**Advantages** : \- ✅ Covers wide range of search space \- ✅ Prevents bias toward local optima \- ✅ Works well with clustering

**Disadvantages** : \- ⚠️ Does not consider model uncertainty \- ⚠️ Slightly higher computational cost

* * *

### 1.2.3 Query-by-Committee

**Principle** : Select samples where multiple models (committee) disagree the most

**Formula** : $$ x^* = \arg\max_{x \in \mathcal{U}} \text{Disagreement}(C, x) $$

where $C = {M_1, M_2, ..., M_K}$ is a set of models (committee)

**Disagreement Measurement** :

**Regression Problems (Variance)** : $$ \text{Disagreement}(C, x) = \frac{1}{K} \sum_{k=1}^K (M_k(x) - \bar{M}(x))^2 $$

**Classification Problems (Kullback-Leibler Divergence)** : $$ \text{Disagreement}(C, x) = \frac{1}{K} \sum_{k=1}^K KL(P_k(\cdot|x) | P_C(\cdot|x)) $$

**Code Example 3: Query-by-Committee Implementation**
    
    
    __PROTECTED_CODE_4__

**Output** :
    
    
    __PROTECTED_CODE_5__

**Advantages** : \- ✅ Leverages knowledge from diverse models \- ✅ Reduces model bias \- ✅ Robust uncertainty estimation

**Disadvantages** : \- ⚠️ High computational cost (training multiple models) \- ⚠️ Depends on model selection

* * *

### 1.2.4 Expected Model Change

**Principle** : Select samples that cause the largest change in model parameters

**Formula** (gradient-based): $$ x^* = \arg\max_{x \in \mathcal{U}} |\nabla_\theta \mathcal{L}(\theta; x, \hat{y})| $$

where $\theta$ is model parameters, $\mathcal{L}$ is loss function, $\hat{y}$ is predicted value

**Advantages** : \- ✅ Directly evaluates impact on model improvement \- ✅ Enables efficient learning

**Disadvantages** : \- ⚠️ High computational cost \- ⚠️ Limited to models with computable gradients

* * *

## 1.3 Exploration vs Exploitation

### The Tradeoff Concept

One of the most important concepts in active learning is the **exploration-exploitation tradeoff**.

**Exploration** : \- Explore unknown regions \- Collect diverse samples \- Acquire new information \- Take risks

**Exploitation** : \- Intensively investigate known good regions \- Prioritize high uncertainty regions \- Maximize use of existing knowledge \- Improve safely

### Visualizing the Tradeoff
    
    
    ```mermaid
    flowchart TB
        subgraph Exploration_Focused [Exploration-focused]
        A["Aggressively sample
    unknown regions"]
        A --> B["High discovery potential"]
        A --> C["Slow learning"]
        end
    
        subgraph Exploitation_Focused [Exploitation-focused]
        D["Intensively sample
    high uncertainty regions"]
        D --> E["Fast convergence"]
        D --> F["Risk of
    local optima"]
        end
    
        subgraph Balance
        G["Balanced exploration
    and exploitation"]
        G --> H["Efficient learning"]
        G --> I["Wide and
    deep understanding"]
        end
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style G fill:#e8f5e9
        style I fill:#4CAF50,color:#fff
    ```

### ε-greedy Approach

**Principle** : Explore with probability $\epsilon$, exploit with probability $1-\epsilon$

**Algorithm** :
    
    
    __PROTECTED_CODE_6__

**Code Example 4: ε-greedy Active Learning**
    
    
    __PROTECTED_CODE_7__

**Output** :
    
    
    __PROTECTED_CODE_8__

**Choosing ε** : \- $\epsilon = 0$: Full exploitation (risk of local optima) \- $\epsilon = 1$: Full exploration (random sampling) \- $\epsilon = 0.1 \sim 0.2$: Well-balanced (recommended)

* * *

### Upper Confidence Bound (UCB)

**Principle** : Prediction mean + uncertainty bonus

**Formula** : $$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\mu(x)$: Prediction mean
  * $\sigma(x)$: Prediction standard deviation
  * $\kappa$: Exploration parameter (typically 1.0-3.0)

**Code Example 5: Sample Selection Using UCB**
    
    
    __PROTECTED_CODE_9__

**Output** :
    
    
    __PROTECTED_CODE_10__

**Impact of κ** : \- Large $\kappa$ → Exploration-focused \- Small $\kappa$ → Exploitation-focused \- Recommended: $\kappa = 2.0 \sim 2.5$

* * *

## 1.4 Case Study: Catalyst Activity Prediction

### Problem Setup

**Objective** : Predict catalyst reaction activity and discover the most active catalyst in 10 experiments

**Dataset** : \- Candidate catalysts: 500 types \- Features: Metal composition (3 elements), loading, calcination temperature \- Target variable: Reaction rate constant (k)

**Constraints** : \- Single experiment takes 3 days \- Budget limited to maximum 10 experiments

### Random Sampling vs Active Learning

**Code Example 6: Comparative Experiment for Catalyst Activity Prediction**
    
    
    __PROTECTED_CODE_11__

**Expected Output** :
    
    
    __PROTECTED_CODE_12__

**Important Observations** : \- ✅ Active Learning reaches 97.5% of true optimal value in 10 experiments \- ✅ Random Sampling only reaches 79.3% \- ✅ **23% performance improvement** \- ✅ R² score steadily improves (0.512 → 0.843)

* * *

## 1.5 Chapter Summary

### What We Learned

  1. **Active Learning Definition** \- Efficient learning through active data selection \- Differences from passive learning \- Importance in materials science (experimental cost reduction)

  2. **Query Strategies** \- **Uncertainty Sampling** : Select samples with uncertain predictions \- **Diversity Sampling** : Select diverse samples \- **Query-by-Committee** : Leverage disagreement between models \- **Expected Model Change** : Select by impact on model updates

  3. **Exploration-Exploitation** \- ε-greedy: Probabilistically switch between exploration and exploitation \- UCB: Prediction mean + uncertainty bonus \- Importance of balance

  4. **Practical Examples** \- 23% performance improvement in catalyst activity prediction \- 97.5% achievement rate in 10 experiments \- 1.3× efficiency over random sampling

### Key Takeaways

  * ✅ Active learning excels in **problems with high data acquisition costs**
  * ✅ Query strategy selection **greatly affects exploration efficiency**
  * ✅ **Balance in exploration-exploitation is crucial**
  * ✅ Can **reduce experiments by 50-90%** in materials science
  * ✅ **Significant improvements achievable in 10-20 experiments**

### Next Chapter

In Chapter 2, we will learn the core **uncertainty estimation techniques** for active learning: \- Ensemble methods (Random Forest, LightGBM) \- Dropout methods (Bayesian Neural Networks) \- Gaussian Processes (rigorous uncertainty quantification)

**[Chapter 2: Uncertainty Estimation Techniques →](<chapter-2.html>)**

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

For the following situations, determine which query strategy is most appropriate and explain your reasoning.

**Situation A** : Predicting tensile strength of alloys. 10,000 candidate materials, 50 initial data samples, budget allows 20 additional experiments. Search space is vast, but strength varies relatively smoothly with composition.

**Situation B** : Discovery of novel organic semiconductor materials. 100,000 candidate molecules, 10 initial data samples, budget allows 10 additional experiments. Properties vary very complexly with molecular structure.

Hint \- Situation A: Vast search space → ? \- Situation B: Little data, complex function → ? \- Review characteristics of query strategies  Example Solution **Situation A: Diversity Sampling is optimal** **Reasoning**: 1\. Search space is vast (10,000 types), difficult to cover entirely with 20 experiments 2\. 50 initial samples available, sufficient for reasonable model construction 3\. Strength varies smoothly, so covering wide range enables grasping overall picture 4\. Diversity sampling provides even coverage of search space **Alternative**: UCB sampling (with large exploration parameter κ) **Situation B: Uncertainty Sampling (or Query-by-Committee) is optimal** **Reasoning**: 1\. Very few initial data samples (10 samples) 2\. Properties vary complexly, so should prioritize high uncertainty regions 3\. Budget is limited (10 experiments), requiring efficient learning 4\. Uncertainty sampling selects most informative samples **Alternative**: Query-by-Committee (handles complex functions through model diversity) 

* * *

### Problem 2 (Difficulty: Medium)

Implement ε-greedy Active Learning and compare exploration efficiency for different ε values (0.0, 0.1, 0.2, 0.5).

**Tasks** : 1\. Generate synthetic material properties dataset (500 samples) 2\. Execute ε-greedy AL with 10 initial samples and 15 additional experiments 3\. Plot best value discovered for each ε 4\. Select optimal ε and explain reasoning

Hint \- Reference Code Example 4 to implement ε-greedy \- Run 5 trials for each ε and average results \- Plot: x-axis = experiment count, y-axis = best value discovered  Example Solution
    
    
    __PROTECTED_CODE_13__

**Expected Output**: 
    
    
    __PROTECTED_CODE_14__

**Conclusion**: \- **ε = 0.2 is optimal** (99.0% achievement rate) \- ε = 0.0 prone to local optima (89.2%) \- ε = 0.5 over-explores inefficiently (93.4%) \- **Moderate exploration (ε=0.1-0.2) provides good balance** 

* * *

### Problem 3 (Difficulty: Hard)

Compare three query strategies (Uncertainty, Diversity, Query-by-Committee) on the same dataset and select the most efficient method.

**Requirements** : 1\. Generate synthetic multi-objective material data (1,000 samples, 10 dimensions) 2\. Execute each method with 20 initial samples and 30 additional experiments 3\. Evaluate using these metrics: \- Best value discovered \- R² score (prediction accuracy on all data) \- Computation time 4\. Select most efficient method overall

Hint \- Implement each method independently \- Run 5 trials and average results \- Measure computation time with `time.time()` \- Consider tradeoffs (accuracy vs computation time)  Example Solution
    
    
    __PROTECTED_CODE_15__

**Expected Output**: 
    
    
    __PROTECTED_CODE_16__

**Conclusion**: 1\. **Query-by-Committee (QBC)** achieves highest performance (97.4% achievement, R²=0.856) 2\. However, computation time is over 3× longer (38.12s vs 12.34s) 3\. **Uncertainty Sampling provides best overall balance** \- 96.2% achievement (only 1.2% difference from QBC) \- R²=0.834 (only 0.022 difference from QBC) \- Computation time is 1/3 **Recommendations**: \- **No time constraints**: QBC \- **Balance priority**: Uncertainty Sampling \- **Computation cost priority**: Diversity Sampling 

* * *

## Data Licenses and Citations

### Benchmark Datasets

Benchmark datasets for active learning that can be used in this chapter's code examples:

#### 1\. UCI Machine Learning Repository

  * **License** : CC BY 4.0
  * **Citation** : Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences.
  * **Recommended Datasets** :
  * `make_regression()` (built-in scikit-learn)
  * Wine Quality Dataset
  * Boston Housing Dataset

#### 2\. Materials Project API

  * **License** : CC BY 4.0
  * **Citation** : Jain, A. et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." _APL Materials_ , 1(1), 011002.
  * **Usage** : Active learning experiments in materials science (band gap, formation energy)
  * **API Access** : https://materialsproject.org/api

#### 3\. Matbench Datasets

  * **License** : MIT License
  * **Citation** : Dunn, A. et al. (2020). "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." _npj Computational Materials_ , 6(1), 138.
  * **Usage** : Active learning for material property prediction

### Library Licenses

Licenses of major libraries used in this chapter:

Library | Version | License | Purpose  
---|---|---|---  
modAL | 0.4.1 | MIT | Active Learning Framework  
scikit-learn | 1.3.0 | BSD-3-Clause | Machine Learning & Preprocessing  
numpy | 1.24.3 | BSD-3-Clause | Numerical Computing  
matplotlib | 3.7.1 | PSF (BSD-like) | Visualization  
  
**License Compliance** : \- All are available for commercial use \- Maintain original license notices when redistributing \- Cite appropriately in academic publications

* * *

## Ensuring Reproducibility

### Random Seed Configuration

To make active learning experiments reproducible, set the following random seeds in all code:
    
    
    __PROTECTED_CODE_18__

**Important Points** : \- Data splitting: `train_test_split(..., random_state=SEED)` \- Model initialization: `RandomForestRegressor(..., random_state=SEED)` \- Initial sample selection: Set seed before `np.random.choice(..., replace=False)`

### Library Version Management

To fully reproduce experimental environment, create `requirements.txt`:
    
    
    __PROTECTED_CODE_23__

### Recording Experimental Logs

Record all active learning iterations:
    
    
    __PROTECTED_CODE_24__

* * *

## Common Pitfalls and Solutions

### 1\. Cold Start Problem (Insufficient Initial Data)

**Problem** : Too few initial labeled data leads to unstable uncertainty estimation

**Symptoms** :
    
    
    __PROTECTED_CODE_25__

**Solution** :
    
    
    __PROTECTED_CODE_26__

**Recommended Rules** : \- Minimum 10 samples \- Ideally 3-5× number of features \- More complex models (NN, GP) require more initial data

* * *

### 2\. Query Selection Bias

**Problem** : Using only uncertainty sampling leads to selecting same regions repeatedly

**Symptoms** :
    
    
    __PROTECTED_CODE_27__

**Solution 1: ε-greedy** :
    
    
    __PROTECTED_CODE_28__

**Solution 2: Batch Diversity** :
    
    
    __PROTECTED_CODE_29__

* * *

### 3\. Stopping Criteria Errors

**Problem** : Unclear when to stop active learning

**Bad Example** : Fixed iteration count only
    
    
    __PROTECTED_CODE_30__

**Solution: Multiple stopping criteria** :
    
    
    __PROTECTED_CODE_31__

* * *

### 4\. Distribution Shift

**Problem** : Distribution differs between labeled and unlabeled pools

**Symptoms** :
    
    
    __PROTECTED_CODE_32__

**Solution** :
    
    
    __PROTECTED_CODE_33__

* * *

### 5\. Label Noise Handling

**Problem** : When experimental measurements contain noise, learning incorrect samples

**Solution 1: Uncertainty threshold** :
    
    
    __PROTECTED_CODE_34__

**Solution 2: Ensemble Robustness** :
    
    
    __PROTECTED_CODE_35__

* * *

### 6\. Computational Cost of Uncertainty Estimation

**Problem** : Uncertainty estimation takes too long (e.g., GP's N^3 computational complexity)

**Solution: Pre-filter candidate pool** :
    
    
    __PROTECTED_CODE_36__

* * *

## Quality Checklist

### Experimental Design Checklist

#### Initialization Phase

  * [ ] Random seed configured (`np.random.seed(SEED)`)
  * [ ] Appropriate initial sample count (minimum 10, ideally features × 3-5)
  * [ ] Data split uses stratified sampling (avoid distribution shift)
  * [ ] Library versions recorded in `requirements.txt`

#### Query Strategy Selection

  * [ ] Select appropriate method for task
  * Wide exploration → Diversity Sampling
  * Efficient convergence → Uncertainty Sampling
  * Model robustness → Query-by-Committee
  * [ ] Set exploration-exploitation balance (ε-greedy, UCB)
  * [ ] Consider diversity when batch selecting

#### Stopping Criteria Design

  * [ ] Set maximum iteration count
  * [ ] Define target performance metrics (R², RMSE, etc.)
  * [ ] Set early stopping conditions (patience=5-10)
  * [ ] Clarify budget limits (experimental cost, time)

#### Model Selection

  * [ ] Select models capable of uncertainty estimation
  * Ensemble methods (RF, LightGBM)
  * MC Dropout (NN)
  * Gaussian Process
  * [ ] Select model based on data size
  * Small (<1000) → GP
  * Medium (1000-10000) → RF, LightGBM
  * Large (>10000) → MC Dropout

### Implementation Quality Checklist

#### Data Preprocessing

  * [ ] Missing values handled (deletion or imputation)
  * [ ] Outliers detected and addressed (IQR method, etc.)
  * [ ] Feature scaling applied (standardization or normalization)
  * [ ] No data leakage (test data separated)

#### Code Quality

  * [ ] Type hints added to functions (`def func(x: np.ndarray) -> float:`)
  * [ ] Docstrings written (arguments, return values, purpose)
  * [ ] Error handling implemented (try-except)
  * [ ] Logging output implemented (experiment tracking)

#### Evaluation and Validation

  * [ ] Multiple evaluation metrics calculated (R², RMSE, MAE)
  * [ ] Learning curves plotted (iteration count vs performance)
  * [ ] Compared with random sampling
  * [ ] Statistical significance verified (mean ± std of multiple trials)

### Materials Science-Specific Checklist

#### Physical Constraints

  * [ ] Check physical validity of search space
  * Temperature range: 0-1500°C
  * Composition ratio: Total 100%
  * pH range: 0-14
  * [ ] Verify unit consistency (nm, eV, GPa, etc.)
  * [ ] Synthesizability constraints (experimental feasibility)

#### Domain Knowledge Integration

  * [ ] Leverage physical prior knowledge
  * Kernel selection (periodicity, smoothness)
  * Feature engineering (descriptors)
  * [ ] Verify consistency with known physical laws
  * Band Gap > 0
  * Density > 0

#### Experimental Integration

  * [ ] Account for measurement errors (noise terms)
  * [ ] Define experimental cost function
  * [ ] Design batch experiments (parallelization potential)

* * *

## Additional Practice Exercise Guide

### Complete Solution Example for Exercise 1 (CNT Electrical Conductivity Prediction)

Click to show complete code
    
    
    __PROTECTED_CODE_40__

* * *

## References

  1. Settles, B. (2009). "Active Learning Literature Survey." _Computer Sciences Technical Report 1648_ , University of Wisconsin-Madison.

  2. Lookman, T. et al. (2019). "Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design." _npj Computational Materials_ , 5(1), 1-17. DOI: [10.1038/s41524-019-0153-8](<https://doi.org/10.1038/s41524-019-0153-8>)

  3. Raccuglia, P. et al. (2016). "Machine-learning-assisted materials discovery using failed experiments." _Nature_ , 533(7601), 73-76. DOI: [10.1038/nature17439](<https://doi.org/10.1038/nature17439>)

  4. Ren, F. et al. (2018). "Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments." _Science Advances_ , 4(4), eaaq1566. DOI: [10.1126/sciadv.aaq1566](<https://doi.org/10.1126/sciadv.aaq1566>)

  5. Kusne, A. G. et al. (2020). "On-the-fly closed-loop materials discovery via Bayesian active learning." _Nature Communications_ , 11(1), 5966. DOI: [10.1038/s41467-020-19597-w](<https://doi.org/10.1038/s41467-020-19597-w>)

* * *

## Navigation

### Next Chapter

**[Chapter 2: Uncertainty Estimation Techniques →](<chapter-2.html>)**

### Series Index

**[← Return to Series Index](<./index.html>)**

* * *

## Author Information

**Creator** : AI Terakoya Content Team **Created** : 2025-10-18 **Version** : 1.0

**Update History** : \- 2025-10-18: v1.0 Initial release

**Feedback** : \- GitHub Issues: [AI_Homepage/issues](<https://github.com/your-repo/AI_Homepage/issues>) \- Email: yusuke.hashimoto.b8@tohoku.ac.jp

**License** : Creative Commons BY 4.0

* * *

**Let's learn the details of Uncertainty Estimation in the next chapter!**
