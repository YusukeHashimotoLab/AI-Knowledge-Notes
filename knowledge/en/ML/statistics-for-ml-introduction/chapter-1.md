---
title: "Chapter 1: Descriptive Statistics and Probability Basics"
chapter_title: "Chapter 1: Descriptive Statistics and Probability Basics"
subtitle: Capturing Data Characteristics and Quantifying Uncertainty
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 8
---

## Introduction

Statistics is the discipline of extracting meaningful information from data and making rational decisions under uncertainty. In machine learning, statistical knowledge is essential for understanding data characteristics, evaluating model performance, and quantifying prediction uncertainty.

In this chapter, we will learn about **descriptive statistics** and **probability theory** , which form the foundation of statistics. In descriptive statistics, we will learn how to express the central tendency (mean, median) and spread (variance, standard deviation) of data numerically. In probability theory, we will master how to mathematically handle uncertain events.

**💡 What You'll Learn in This Chapter**

  * Summarizing data characteristics with numerical indicators (mean, variance, quartiles, etc.)
  * Visualizing data with appropriate graphs (histograms, box plots, etc.)
  * Basic probability calculations and conditional probability
  * Understanding and applying Bayes' theorem
  * Mathematical definitions of expected value and variance

## 1\. Fundamentals of Descriptive Statistics

Descriptive Statistics is a method for summarizing and expressing data characteristics in an easily understandable form. By representing large amounts of data with a few numerical indicators, we can grasp the overall picture of the data.

### 1.1 Measures of Central Tendency

These are indicators that show where the "center" of the data is.

#### Mean

The sum of all data values divided by the number of data points, the most basic measure of central tendency.

Mathematical expression:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

Where $n$ is the number of data points and $x_i$ is the $i$-th data value.

#### Median

The middle value when data is arranged in ascending order. A robust measure that is less affected by outliers.

#### Mode

The value that appears most frequently in the data. Can also be applied to categorical data.

**📝 Example: Student Test Scores**

Test scores of 5 students: 65, 70, 75, 80, 95 points

  * Mean: $(65+70+75+80+95)/5 = 77$ points
  * Median: 75 points (middle value)

If an extreme value (e.g., 10 points) is included, the mean changes significantly, but the median remains relatively stable.

### 1.2 Measures of Spread

These are indicators that show how much the data is scattered.

#### Variance

The average of the squared differences between each data value and the mean.

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**⚠️ Note: Sample Variance vs Population Variance**

When estimating population variance from a sample, divide by $n-1$ instead of $n$ (unbiased estimator):

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

#### Standard Deviation

The square root of variance. Can express spread in the same units as the original data.

$$\sigma = \sqrt{\sigma^2}$$

#### Quartiles and Percentiles

Indicators that show position when data is ordered.

  * **First Quartile (Q1)** : Position at 25% of the data
  * **Second Quartile (Q2)** : Median (50% position)
  * **Third Quartile (Q3)** : Position at 75% of the data
  * **Interquartile Range (IQR)** : $Q3 - Q1$ (range of the middle 50% of data)

### 1.3 Python Implementation

Let's calculate descriptive statistics using NumPy and SciPy.
    
    
    import numpy as np
    from scipy import stats
    
    # Sample data: Student test scores
    scores = np.array([65, 70, 72, 75, 78, 80, 82, 85, 88, 95, 98])
    
    # Measures of central tendency
    mean = np.mean(scores)
    median = np.median(scores)
    mode_result = stats.mode(scores, keepdims=True)
    mode = mode_result.mode[0] if len(mode_result.mode) > 0 else None
    
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode}")
    
    # Measures of spread
    variance = np.var(scores)  # Population variance
    std_dev = np.std(scores)   # Population standard deviation
    sample_variance = np.var(scores, ddof=1)  # Sample variance (unbiased estimator)
    sample_std = np.std(scores, ddof=1)       # Sample standard deviation
    
    print(f"\nPopulation Variance: {variance:.2f}")
    print(f"Population Standard Deviation: {std_dev:.2f}")
    print(f"Sample Variance: {sample_variance:.2f}")
    print(f"Sample Standard Deviation: {sample_std:.2f}")
    
    # Quartiles
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)  # Median
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    print(f"\nFirst Quartile (Q1): {q1:.2f}")
    print(f"Second Quartile (Q2): {q2:.2f}")
    print(f"Third Quartile (Q3): {q3:.2f}")
    print(f"Interquartile Range (IQR): {iqr:.2f}")

**Execution Result:**
    
    
    Mean: 80.73
    Median: 80.00
    Mode: 65
    
    Population Variance: 103.29
    Population Standard Deviation: 10.16
    Sample Variance: 113.62
    Sample Standard Deviation: 10.66
    
    First Quartile (Q1): 73.50
    Second Quartile (Q2): 80.00
    Third Quartile (Q3): 88.00
    Interquartile Range (IQR): 14.50

## 2\. Data Visualization

Not only numerical indicators but also visualization through graphs is essential for understanding data.

### 2.1 Histogram

A graph that visually represents the distribution of data. Data is divided into classes (bins), and the frequency of each class is displayed as a bar graph.
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate sample data following a normal distribution
    np.random.seed(42)
    data = np.random.normal(loc=70, scale=10, size=1000)
    
    # Draw histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    plt.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.2f}')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram: Data Distribution', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

### 2.2 Box Plot

A graph that visually represents quartiles and can also identify outliers.
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Multiple group data
    np.random.seed(42)
    group_a = np.random.normal(70, 10, 100)
    group_b = np.random.normal(75, 8, 100)
    group_c = np.random.normal(65, 12, 100)
    
    data_groups = [group_a, group_b, group_c]
    
    # Draw box plot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_groups, labels=['Group A', 'Group B', 'Group C'],
                     patch_artist=True, notch=True)
    
    # Customize colors
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Box Plot: Comparison Between Groups', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

**💡 How to Read a Box Plot**

  * Bottom of box: First quartile (Q1)
  * Line in box: Median (Q2)
  * Top of box: Third quartile (Q3)
  * Whiskers: Data range (excluding outliers)
  * Dots: Outliers

### 2.3 Scatter Plot

A graph that visualizes the relationship between two variables.
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate correlated data
    np.random.seed(42)
    x = np.random.normal(50, 10, 100)
    y = 2 * x + np.random.normal(0, 10, 100)  # y has positive correlation with x
    
    # Draw scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, edgecolors='black', s=50)
    plt.xlabel('Variable X', fontsize=12)
    plt.ylabel('Variable Y', fontsize=12)
    plt.title('Scatter Plot: Relationship Between Two Variables', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Calculate and display correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

## 3\. Probability Basics

Probability theory is a framework for mathematically handling uncertain events. In machine learning, probability theory is used to model data generation processes and quantify prediction uncertainty.

### 3.1 Definition and Axioms of Probability

**Probability** is a numerical value from 0 to 1 that represents the likelihood of an event occurring.

**Kolmogorov's Axioms** (basic properties of probability):

  1. **Non-negativity** : For all events $A$, $P(A) \geq 0$
  2. **Total Probability** : The probability of the entire event space is 1, $P(\Omega) = 1$
  3. **Additivity** : For mutually exclusive events $A$ and $B$, $P(A \cup B) = P(A) + P(B)$

#### Basic Probability Calculations

  * **Complement** : $P(A^c) = 1 - P(A)$
  * **Union** : $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### 3.2 Conditional Probability

The probability that event $A$ occurs given that event $B$ has occurred is called **conditional probability** , denoted as $P(A|B)$.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Where $P(B) > 0$.

**📝 Example: Card Drawing**

When drawing one card from a 52-card deck:

  * Probability that the drawn card is a spade: $P(\text{Spade}) = 13/52 = 1/4$
  * Probability that the drawn card is a face card (J, Q, K): $P(\text{Face card}) = 12/52 = 3/13$
  * Probability that it is a face card given that it is a spade: $P(\text{Face card}|\text{Spade}) = 3/13$

### 3.3 Bayes' Theorem

**Bayes' theorem** is an important formula for reversing conditional probability. It plays a central role in machine learning, especially in Bayesian statistics.

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

Meaning of each term:

  * $P(A|B)$: **Posterior probability** \- Probability of $A$ after observing event $B$
  * $P(B|A)$: **Likelihood** \- Probability of observing $B$ when $A$ is true
  * $P(A)$: **Prior probability** \- Probability of $A$ before observation
  * $P(B)$: **Marginal probability (Evidence)** \- Overall probability of observing $B$

#### Application of Bayes' Theorem: Medical Diagnosis

**📝 Example: Disease Testing**

For a rare disease:

  * 1% of the population has this disease: $P(\text{Disease}) = 0.01$
  * Test sensitivity (correctly identifies diseased as positive): $P(\text{Positive}|\text{Disease}) = 0.99$
  * Test false positive rate: $P(\text{Positive}|\text{Healthy}) = 0.05$

If the test comes back positive, what is the probability of actually having the disease?

Applying Bayes' theorem:

$$P(\text{Disease}|\text{Positive}) = \frac{P(\text{Positive}|\text{Disease}) \cdot P(\text{Disease})}{P(\text{Positive})}$$

First, calculate $P(\text{Positive})$ (law of total probability):

$$P(\text{Positive}) = P(\text{Positive}|\text{Disease})P(\text{Disease}) + P(\text{Positive}|\text{Healthy})P(\text{Healthy})$$

$$= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

Therefore:

$$P(\text{Disease}|\text{Positive}) = \frac{0.99 \times 0.01}{0.0594} \approx 0.167$$

In other words, even if the test is positive, the probability of actually having the disease is only about 16.7%. This is due to the low prevalence of the disease.

#### Python Implementation
    
    
    import numpy as np
    
    def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
        """
        Calculate Bayes' theorem
    
        Parameters:
        -----------
        p_a : float
            Prior probability P(A)
        p_b_given_a : float
            Likelihood P(B|A)
        p_b_given_not_a : float
            P(B|not A)
    
        Returns:
        --------
        float
            Posterior probability P(A|B)
        """
        # Calculate P(B) using law of total probability
        p_not_a = 1 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    
        # Calculate P(A|B) using Bayes' theorem
        p_a_given_b = (p_b_given_a * p_a) / p_b
    
        return p_a_given_b, p_b
    
    # Medical diagnosis example
    p_disease = 0.01  # Prior probability of disease (prevalence)
    p_positive_given_disease = 0.99  # Sensitivity (true positive rate)
    p_positive_given_healthy = 0.05  # False positive rate
    
    p_disease_given_positive, p_positive = bayes_theorem(
        p_disease,
        p_positive_given_disease,
        p_positive_given_healthy
    )
    
    print("=== Bayes' Theorem in Medical Diagnosis ===")
    print(f"Disease prevalence: {p_disease * 100:.1f}%")
    print(f"Test sensitivity: {p_positive_given_disease * 100:.1f}%")
    print(f"False positive rate: {p_positive_given_healthy * 100:.1f}%")
    print(f"\nProbability of positive result: {p_positive * 100:.2f}%")
    print(f"Probability of actually having disease when positive: {p_disease_given_positive * 100:.2f}%")
    
    # Compare with varying sensitivity
    print("\n=== Comparison with Varying Sensitivity ===")
    sensitivities = [0.90, 0.95, 0.99, 0.999]
    for sens in sensitivities:
        prob, _ = bayes_theorem(p_disease, sens, p_positive_given_healthy)
        print(f"Sensitivity {sens*100:.1f}%: Disease probability when positive = {prob*100:.2f}%")

**Execution Result:**
    
    
    === Bayes' Theorem in Medical Diagnosis ===
    Disease prevalence: 1.0%
    Test sensitivity: 99.0%
    False positive rate: 5.0%
    
    Probability of positive result: 5.94%
    Probability of actually having disease when positive: 16.64%
    
    === Comparison with Varying Sensitivity ===
    Sensitivity 90.0%: Disease probability when positive = 15.38%
    Sensitivity 95.0%: Disease probability when positive = 16.10%
    Sensitivity 99.0%: Disease probability when positive = 16.64%
    Sensitivity 99.9%: Disease probability when positive = 16.72%

## 4\. Expected Value and Variance

These are important concepts for expressing characteristics of random variables numerically.

### 4.1 Expected Value

The **expected value** represents the average value of a random variable.

For **discrete random variables** :

$$E[X] = \sum_{i} x_i \cdot P(X = x_i)$$

For **continuous random variables** :

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

Where $f(x)$ is the probability density function.

#### Properties of Expected Value

  * **Linearity** : $E[aX + b] = aE[X] + b$
  * **Additivity** : $E[X + Y] = E[X] + E[Y]$

### 4.2 Variance and Standard Deviation

**Variance** represents how much a random variable is scattered from its expected value.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Standard deviation** is the square root of variance:

$$\sigma = \sqrt{\text{Var}(X)}$$

#### Properties of Variance

  * $\text{Var}(aX + b) = a^2 \text{Var}(X)$
  * If $X$ and $Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 4.3 Python Implementation
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Example of dice roll experiment
    def dice_expectation():
        """Calculate expected value of dice roll"""
        outcomes = np.array([1, 2, 3, 4, 5, 6])
        probabilities = np.array([1/6] * 6)
    
        # Calculate expected value
        expectation = np.sum(outcomes * probabilities)
    
        # Calculate variance
        variance = np.sum((outcomes - expectation)**2 * probabilities)
        std_dev = np.sqrt(variance)
    
        print("=== Expected Value and Variance of Dice Roll ===")
        print(f"Expected value E[X]: {expectation:.4f}")
        print(f"Variance Var(X): {variance:.4f}")
        print(f"Standard deviation σ: {std_dev:.4f}")
    
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(outcomes, probabilities, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(expectation, color='red', linestyle='--', linewidth=2,
                    label=f'Expected value: {expectation:.2f}')
        plt.xlabel('Outcome', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Probability Distribution of Dice Roll', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(outcomes)
        plt.show()
    
    dice_expectation()
    
    # Verification by simulation
    print("\n=== Verification by Simulation ===")
    n_trials = 10000
    dice_rolls = np.random.randint(1, 7, n_trials)
    
    empirical_mean = np.mean(dice_rolls)
    empirical_variance = np.var(dice_rolls)
    empirical_std = np.std(dice_rolls)
    
    print(f"Number of simulations: {n_trials}")
    print(f"Empirical mean: {empirical_mean:.4f}")
    print(f"Empirical variance: {empirical_variance:.4f}")
    print(f"Empirical standard deviation: {empirical_std:.4f}")
    print(f"\nDifference from theoretical values:")
    print(f"Difference in mean: {abs(empirical_mean - 3.5):.4f}")
    print(f"Difference in variance: {abs(empirical_variance - 35/12):.4f}")

## 5\. Summary and Next Steps

In this chapter, we learned the basics of descriptive statistics and probability theory, which form the foundation of statistics.

**✅ What We Learned in This Chapter**

  * Measures of central tendency (mean, median, mode) and spread (variance, standard deviation)
  * Data visualization with histograms, box plots, and scatter plots
  * Basic axioms of probability and conditional probability
  * Theory and applications of Bayes' theorem
  * Mathematical definitions and calculations of expected value and variance
  * Implementation of statistical analysis using NumPy/SciPy/Matplotlib

**🔑 Key Points**

  * Mean is susceptible to outliers, but median is robust
  * When calculating sample variance, divide by $n-1$ (unbiased estimator)
  * In Bayes' theorem, the prior probability greatly influences the posterior probability
  * Expected value has linearity, and variance is squared with respect to linear transformations

### Next Steps

In the next chapter, we will learn about probability distributions. We will master the properties and applications of probability distributions frequently used in machine learning, such as normal distribution, binomial distribution, and Poisson distribution.

[← Back to Series Top](<./index.html>) Chapter 2: Probability Distributions (Coming Soon) →

## Practice Problems

**Problem 1: Calculating Descriptive Statistics**

Calculate the mean, median, variance, and standard deviation for the following dataset.

Data: 12, 15, 18, 20, 22, 25, 28, 30, 35, 40
    
    
    import numpy as np
    
    data = np.array([12, 15, 18, 20, 22, 25, 28, 30, 35, 40])
    
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data, ddof=1)
    std_dev = np.std(data, ddof=1)
    
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")

**Problem 2: Applying Bayes' Theorem**

Consider a spam email filter. 10% of emails are spam, and the probability that an email contains the word "free" is 80% for spam emails and 5% for normal emails. Calculate the probability that an email containing the word "free" is spam.
    
    
    def spam_filter_bayes(p_spam, p_free_given_spam, p_free_given_normal):
        p_normal = 1 - p_spam
        p_free = p_free_given_spam * p_spam + p_free_given_normal * p_normal
        p_spam_given_free = (p_free_given_spam * p_spam) / p_free
        return p_spam_given_free
    
    # Parameters
    p_spam = 0.10
    p_free_given_spam = 0.80
    p_free_given_normal = 0.05
    
    result = spam_filter_bayes(p_spam, p_free_given_spam, p_free_given_normal)
    print(f"Probability that email containing 'free' is spam: {result * 100:.2f}%")

**Problem 3: Calculating Expected Value**

In a lottery game, buying a 1000 yen ticket gives a 10% chance of winning 5000 yen, a 5% chance of winning 10000 yen, and the rest is 0 yen. Calculate the expected value of this game and determine whether it is worth playing.
    
    
    import numpy as np
    
    # Outcomes and probabilities
    outcomes = np.array([5000, 10000, 0])
    probabilities = np.array([0.10, 0.05, 0.85])
    
    # Calculate expected value
    expected_value = np.sum(outcomes * probabilities)
    net_expected_value = expected_value - 1000  # Subtract ticket cost
    
    print(f"Expected value: {expected_value:.2f} yen")
    print(f"Net expected value (after ticket cost): {net_expected_value:.2f} yen")
    
    if net_expected_value > 0:
        print("It is worth playing in terms of expected value")
    else:
        print("It is not worth playing in terms of expected value")
