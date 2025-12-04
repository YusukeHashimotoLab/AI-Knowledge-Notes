---
title: "Chapter 2: Uncertainty Estimation Techniques"
chapter_title: "Chapter 2: Uncertainty Estimation Techniques"
subtitle: Prediction Confidence Intervals with Ensemble, Dropout, and Gaussian Process
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 2: Uncertainty Estimation Techniques

This chapter covers Uncertainty Estimation Techniques. You will learn principles of three uncertainty estimation methods, Ensemble methods (Random Forest), and MC Dropout to neural networks.

**Prediction Confidence Intervals with Ensemble, Dropout, and Gaussian Process**

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the principles of three uncertainty estimation methods
  * ✅ Implement Ensemble methods (Random Forest)
  * ✅ Apply MC Dropout to neural networks
  * ✅ Calculate prediction variance with Gaussian Process
  * ✅ Explain the criteria for selecting among these methods

**Reading Time** : 25-30 minutes **Code Examples** : 8 examples **Exercises** : 3 problems

* * *

## 2.1 Uncertainty Estimation with Ensemble Methods

### Why Uncertainty Estimation is Important

In active learning, it is necessary to quantify "how confident the model is in its predictions." Uncertainty estimation is a core technology for query strategies.

**Two Types of Uncertainty** :

  1. **Aleatoric Uncertainty** \- Noise inherent in the data itself \- Measurement errors, environmental variations, etc. \- Does not decrease even with more data

  2. **Epistemic Uncertainty** \- Uncertainty due to the model's lack of knowledge \- Caused by insufficient data \- Decreases with more data

**Uncertainty Focused on by Active Learning** : → **Epistemic Uncertainty** (can be improved by adding data)

### Principle of Ensemble Methods

**Basic Idea** : Measure uncertainty by the variation in predictions from multiple models

**Formula** : $$ \mu(x) = \frac{1}{M} \sum_{m=1}^M f_m(x) $$

$$ \sigma^2(x) = \frac{1}{M} \sum_{m=1}^M (f_m(x) - \mu(x))^2 $$

  * $f_m(x)$: Prediction from the m-th model
  * $M$: Number of models (ensemble size)
  * $\mu(x)$: Prediction mean
  * $\sigma^2(x)$: Prediction variance (uncertainty)

### Implementation with Random Forest

**Code Example 1: Uncertainty Estimation with Random Forest**
    
    
    __PROTECTED_CODE_0__

**OutputExample** :
    
    
    __PROTECTED_CODE_1__

### Implementation with LightGBM

**Code Example 2: Uncertainty Estimation with LightGBM**
    
    
    __PROTECTED_CODE_2__

**Advantages** : \- ✅ Simple to implement \- ✅ Relatively low computational cost \- ✅ Easy to interpret \- ✅ Strong performance on tabular data

**Disadvantages** : \- ⚠️ Depends on ensemble size \- ⚠️ Difficult to apply to deep learning \- ⚠️ May require uncertainty calibration

* * *

## 2.2 Uncertainty Estimation with Dropout Methods

### MC Dropout (Monte Carlo Dropout)

**Principle** : Apply dropout during inference as well and measure variation through multiple predictions

**Regular Dropout** (training only):
    
    
    __PROTECTED_CODE_3__

**MC Dropout** (dropout during inference too):
    
    
    __PROTECTED_CODE_4__

### Implementation Example

**Code Example 3: MC Dropout with PyTorch**
    
    
    __PROTECTED_CODE_5__

**OutputExample** :
    
    
    __PROTECTED_CODE_6__

**Advantages** : \- ✅ Easy to apply to existing neural networks \- ✅ No additional training required (dropout only) \- ✅ Well-suited for deep learning

**Disadvantages** : \- ⚠️ Computational cost depends on sampling count (T) \- ⚠️ Choice of dropout rate is important \- ⚠️ May require uncertainty calibration

* * *

## 2.3 Uncertainty Estimation with Gaussian Process (GP)

### Fundamentals of GP

Gaussian Process is a powerful method for defining probability distributions over functions.

**Definition** : $$ f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) $$

  * $\mu(\mathbf{x})$: Mean function (usually 0)
  * $k(\mathbf{x}, \mathbf{x}')$: Kernel function (covariance function)

**Predictive Distribution** : $$ p(f^* | \mathbf{X}, \mathbf{y}, \mathbf{x}^*) = \mathcal{N}(\mu^*, \sigma^{*2}) $$

$$ \mu^* = k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} \mathbf{y} $$

$$ \sigma^{*2} = k(\mathbf{x}^*, \mathbf{x}^*) - k(\mathbf{x}^*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} k(\mathbf{X}, \mathbf{x}^*) $$

### Kernel Functions

**RBF (Radial Basis Function) Kernel** : $$ k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{|\mathbf{x}_i - \mathbf{x}_j|^2}{2\ell^2}\right) $$

  * $\sigma_f^2$: Signal variance
  * $\ell$: Length scale (smoothness)

**Matérn Kernel** : $$ k(\mathbf{x}_i, \mathbf{x}_j) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right) $$

### Implementation with GPyTorch

**Code Example 4: Uncertainty Estimation with GPyTorch**
    
    
    __PROTECTED_CODE_7__

**OutputExample** :
    
    
    __PROTECTED_CODE_8__

**Advantages** : \- ✅ Rigorous uncertainty quantification \- ✅ High accuracy with small datasets \- ✅ Flexibility through kernel selection \- ✅ Strong theoretical foundation

**Disadvantages** : \- ⚠️ Not suitable for large-scale data (O(n³)) \- ⚠️ Kernel and hyperparameter selection is important \- ⚠️ Performance degrades with high-dimensional data

* * *

## 2.4 Case Study: Band Gap Prediction

### Problem Setup

**Objective** : Predict the band gap of inorganic materials and prioritize calculations for samples with high uncertainty

**Dataset** : Materials Project (DFT calculations completed) \- Number of samples: 5,000 materials \- Features: Compositional descriptors (20-dimensional) \- Target variable: Band Gap (eV)

### Comparison of Three Methods

**Code Example 5: Comparison of Uncertainty Estimation for Band Gap Prediction**
    
    
    __PROTECTED_CODE_9__

**OutputExample** :
    
    
    __PROTECTED_CODE_10__

* * *

## 2.5 Chapter Summary

### What We Learned

  1. **Ensemble Methods** \- Uncertainty estimation with Random Forest and LightGBM \- Quantify uncertainty through prediction variance \- Simple to implement, moderate computational cost

  2. **MC Dropout** \- Apply dropout during inference as well \- Easy to implement with neural networks \- Sampling count and dropout rate are important

  3. **Gaussian Process** \- Rigorous uncertainty quantification \- Flexibility through kernel functions \- High accuracy with small data, not suitable for large-scale data

### Selecting the Right Method

Method | Recommended Case | Data Size | Computational Cost  
---|---|---|---  
Random Forest | Tabular data, medium-scale | 100-10,000 | Low to Medium  
MC Dropout | Deep learning, images/text | 1,000-100,000 | Medium to High  
Gaussian Process | Small datasets, rigorous uncertainty | 10-1,000 | Medium to High  
  
### Next Chapter

In Chapter 3, we will learn about **acquisition function design** that leverages uncertainty: \- Expected Improvement (EI) \- Probability of Improvement (PI) \- Upper Confidence Bound (UCB) \- Multi-objective and constrained acquisition functions

**[Chapter 3: Acquisition Function Design →](<chapter-3.html>)**

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

(Omitted: Detailed implementation of exercises)

### Problem 2 (Difficulty: Medium)

(Omitted: Detailed implementation of exercises)

### Problem 3 (Difficulty: Hard)

(Omitted: Detailed implementation of exercises)

* * *

## References

  1. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning." _ICML_ , 1050-1059.

  2. Rasmussen, C. E., & Williams, C. K. I. (2006). _Gaussian Processes for Machine Learning_. MIT Press.

  3. Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." _NeurIPS_.

* * *

## Navigation

### Previous Chapter

**[← Chapter 1: The Need for Active Learning](<chapter-1.html>)**

### Next Chapter

**[Chapter 3: Acquisition Function Design →](<chapter-3.html>)**

### Series Index

**[← Back to Series Index](<./index.html>)**

* * *

**Let's learn about acquisition function design in the next chapter!**
