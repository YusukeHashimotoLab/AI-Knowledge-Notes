---
title: "Chapter 3: Acquisition Function Design"
chapter_title: "Chapter 3: Acquisition Function Design"
subtitle: Expected Improvement・UCB・Multi-Objective Optimization
reading_time: 25-30 min
difficulty: Intermediate-Advanced
code_examples: 7
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 3: Acquisition Function Design

This chapter covers Acquisition Function Design. You will learn Expected Improvement, Pareto optimality to multi-objective optimization, and Incorporate constraints into Acquisition Functions.

**Expected Improvement・UCB・Multi-Objective Optimization**

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the characteristics of four major Acquisition Functions
  * ✅ Implement Expected Improvement
  * ✅ Apply Pareto optimality to multi-objective optimization
  * ✅ Incorporate constraints into Acquisition Functions
  * ✅ Explain selection criteria for Acquisition Functions

**Reading Time** : 25-30 min **Code Examples** : 7 **Exercises** : 3

* * *

## 3.1 Fundamentals of Acquisition Functions

### What is an Acquisition Function?

**Definition** : A scoring function that determines which sample should be acquired next

**Formula** : $$ x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D}) $$

  * $\alpha(x | \mathcal{D})$: Acquisition Function
  * $\mathcal{X}$: Search space
  * $\mathcal{D}$: Data acquired so far

### Four Major Acquisition Functions

#### 1\. Expected Improvement (EI)

**Principle** : Expected value of improvement from the current best value

**Formula** : $$ \text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)] $$

$$ = \begin{cases} (\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z) & \text{if } \sigma(x) > 0 \ 0 & \text{if } \sigma(x) = 0 \end{cases} $$

where, $$ Z = \frac{\mu(x) - f^*}{\sigma(x)} $$

  * $f^*$: Current best value
  * $\mu(x)$: Predicted mean
  * $\sigma(x)$: Predicted standard deviation
  * $\Phi(\cdot)$: Cumulative distribution function of standard normal distribution
  * $\phi(\cdot)$: Probability density function of standard normal distribution

**Code Example 1: Implementation of Expected Improvement**
    
    
    __PROTECTED_CODE_0__

#### 2\. Probability of Improvement (PI)

**Principle** : Probability of improving the current best value

**Formula** : $$ \text{PI}(x) = P(f(x) \geq f^* + \xi) $$

$$ = \Phi\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right) $$

  * $\xi$: Improvement threshold (typically 0.01)

**Code Example 2: Implementation of Probability of Improvement**
    
    
    __PROTECTED_CODE_1__

#### 3\. Upper Confidence Bound (UCB)

**Principle** : Predicted mean + uncertainty bonus

**Formula** : $$ \text{UCB}(x) = \mu(x) + \kappa \sigma(x) $$

  * $\kappa$: Exploration parameter (typically 1.0-3.0)

**Code Example 3: Implementation of UCB**
    
    
    __PROTECTED_CODE_2__

#### 4\. Thompson Sampling

**Principle** : Sample from Gaussian Process and select the maximum value

**Formula** : $$ f(x) \sim \mathcal{GP}(\mu(x), k(x, x')) $$

$$ x^* = \arg\max_{x \in \mathcal{X}} f(x) $$

**Code Example 4: Implementation of Thompson Sampling**
    
    
    __PROTECTED_CODE_3__

* * *

## 3.2 Multi-Objective Acquisition Functions

### Pareto Optimality

**Definition** : A solution that does not sacrifice other objectives to improve one objective

**Formula** : $$ x^* \text{ is Pareto optimal} \iff \nexists x : f_i(x) \geq f_i(x^*) \ \forall i \land f_j(x) > f_j(x^*) \ \text{for some } j $$

### Expected Hypervolume Improvement (EHVI)

**Principle** : Maximize the expected improvement in hypervolume

**Formula** : $$ \text{EHVI}(x) = \mathbb{E}[HV(\mathcal{P} \cup {f(x)}) - HV(\mathcal{P})] $$

  * $HV(\cdot)$: Hypervolume
  * $\mathcal{P}$: Current Pareto set

**Code Example 5: Implementation of Multi-Objective Optimization (BoTorch)**
    
    
    __PROTECTED_CODE_4__

* * *

## 3.3 Constrained Acquisition Functions

### Handling Constraints

**Example** : Synthesizability constraints, cost constraints

**Formula** : $$ x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D}) \cdot P_c(x) $$

  * $P_c(x)$: Probability of satisfying constraints

**Constrained Expected Improvement** : $$ \text{CEI}(x) = \text{EI}(x) \cdot P(c(x) \leq 0) $$

* * *

## 3.4 Case Study: Thermoelectric Materials Exploration

### Problem Setting

**Objective** : Maximize the thermoelectric figure of merit ZT value

**ZT value** : $$ ZT = \frac{S^2 \sigma T}{\kappa} $$

  * $S$: Seebeck coefficient
  * $\sigma$: Electrical conductivity
  * $T$: Absolute temperature
  * $\kappa$: Thermal conductivity

**Challenge** : Simultaneous optimization of three properties (multi-objective optimization)

* * *

## Chapter Summary

### Comparison Table of Acquisition Functions

Acquisition Function | Characteristics | Exploration Tendency | Computational Cost | Recommended Use  
---|---|---|---|---  
EI | Expected improvement | Balanced | Low | General optimization  
PI | Improvement probability | Exploitation-focused | Low | Fast exploration  
UCB | Upper confidence bound | Exploration-focused | Low | Wide-range exploration  
Thompson | Stochastic | Balanced | Medium | Parallel experiments  
  
### Next Chapter

In Chapter 4, you will learn about **Applications and Practice in Materials Exploration** : \- Active Learning × Bayesian Optimization \- Active Learning × High-Throughput Computation \- Active Learning × Experimental Robots \- Real-World Applications and Career Paths

**[Chapter 4: Applications and Practice in Materials Exploration →](<chapter-4.html>)**

* * *

## Exercises

(Omitted: Detailed Implementation of Exercises)

* * *

## References

  1. Jones, D. R. et al. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." _Journal of Global Optimization_ , 13(4), 455-492.

  2. Daulton, S. et al. (2020). "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization." _NeurIPS_.

* * *

## Navigation

### Previous Chapter

**[← Chapter 2: Uncertainty Estimation Techniques](<chapter-2.html>)**

### Next Chapter

**[Chapter 4: Applications and Practice in Materials Exploration →](<chapter-4.html>)**

### Series Table of Contents

**[← Back to Series Table of Contents](<./index.html>)**

* * *

**Learn practical applications in the next chapter!**
