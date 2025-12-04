---
title: Introduction to Composition-Based Features Series v1.0
chapter_title: Introduction to Composition-Based Features Series v1.0
subtitle: Accelerate Materials Discovery with Magpie and Machine Learning
---

#### 🎯 What You Will Learn in This Series

**Composition-based features** are classical yet powerful methods for predicting material properties from chemical composition (types and ratios of elements). Centered on Magpie descriptors, this series systematically covers everything from utilizing elemental property databases to Python implementation with matminer.

## Series Overview

In materials discovery, **chemical composition is the most fundamental and important information**. However, a composition formula like "Fe2O3" alone cannot be input into machine learning models. This is where **composition-based features** play a crucial role by combining periodic table information (ionization energy, electronegativity, atomic radius, etc.) to convert composition into numerical vectors.

This series comprehensively covers the following, centered on the widely-used **Magpie descriptors** :

  * ✅ **Theoretical Foundation** : Mathematical definitions and materials science significance of composition-based features
  * ✅ **Practical Skills** : Feature generation workflows using the matminer library
  * ✅ **Comparative Analysis** : When to use composition-based vs structure-based features (CGCNN/MPNN and other GNNs)
  * ✅ **Latest Trends** : Limitations of Magpie and evolution to GNN methods

## Why Composition-Based Features Are Important

#### 💡 Composition-Based vs Structure-Based

Material features have two main approaches:

  * **Composition-Based** (this series): Generate features from chemical composition only (no structure information required)
  * **Structure-Based** ([GNN Introduction Series](<index.html>)): Learn from 3D structures including atomic coordinates and bonding information

**Strengths of Composition-Based** : Effective for exploring new materials with unknown structures, high-speed screening, and cases with limited data

### Typical Applications of Composition-Based Features

  1. **High-Speed Materials Screening** : Formation energy prediction for 1 million compounds (10-100× faster than GNNs)
  2. **Experimental Data-Driven Exploration** : Property prediction from limited experimental data (combined with transfer learning)
  3. **Hybrid Models** : Improved accuracy by combining composition features + GNN features

## How to Study

### Recommended Learning Flow
    
    
    ```mermaid
    timeline
        title Introduction to Composition-Based Features Learning Flow
        section Chapter 1 : Fundamentals
            What are Composition-Based Features : Definitions and history
            Limitations of Conventional Descriptors : Density and symmetry are insufficient
            Background of Magpie : Utilizing elemental properties
        section Chapter 2 : Magpie Details
            Types of Statistical Descriptors : Mean, variance, max, min
            145 Elemental Properties : Periodic table database
            Mathematical Implementation : Weighted statistics
        section Chapter 3 : Databases
            Elemental Property Databases : Magpie/Deml/Jarvis
            Choosing Featurizers : matminer API
            Custom Featurizer Creation : Adding original descriptors
        section Chapter 4 : Machine Learning Integration
            Model Selection : Random Forest/XGBoost/NN
            Hyperparameter Optimization : Optuna/GridSearch
            Feature Importance Analysis : SHAP/LIME
        section Chapter 5 : Python Practice
            matminer Workflow : Data prep → Feature generation → Model training
            Materials Project Data : Property prediction with real data
            Performance Evaluation and Benchmarking : Comparison with GNNs
    ```

**For Beginners (Learning composition features for the first time):**  
\- Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5 (all chapters recommended)  
\- Time required: 150-180 minutes

**For Intermediate Learners (Machine learning experience, want to use matminer):**  
\- Chapter 2 → Chapter 3 → Chapter 5  
\- Time required: 90-120 minutes

**For GNN Learners (Want to compare composition vs structure):**  
\- Chapter 1 → Chapter 2 → Chapter 5 →   
\- Time required: 120-150 minutes

## Chapter Details

### [Chapter 1: Fundamentals of Composition-Based Features](<chapter-1.html>)

**Difficulty** : Introductory  
**Reading Time** : 25-30 minutes  
**Code Examples** : 5

#### Learning Content

  1. **What Are Composition-Based Features** \- Converting chemical composition to numerical vectors
  2. **Historical Background** \- Before and after Ward (2016) Magpie paper
  3. **Limitations of Conventional Descriptors** \- Density, symmetry, and lattice parameters are insufficient
  4. **Utilizing Elemental Properties** \- The power of periodic table databases
  5. **Success Stories** \- Applications in OQMD and Materials Project

#### Learning Objectives

  * ✅ Explain the definition and role of composition-based features
  * ✅ Demonstrate differences from conventional material descriptors with examples
  * ✅ Understand why Magpie is widely used

**[Read Chapter 1 →](<chapter-1.html>)**

* * *

### [Chapter 2: Magpie and Statistical Descriptors](<chapter-2.html>)

**Difficulty** : Beginner to Intermediate  
**Reading Time** : 30-35 minutes  
**Code Examples** : 8

#### Learning Content

  1. **Mathematical Definition of Magpie Descriptors** \- 145-dimensional statistics
  2. **Types of Statistical Descriptors** \- Mean, variance, maximum, minimum, range, mode
  3. **Weighted vs Unweighted** \- Effect of composition ratio weighting
  4. **22 Types of Elemental Properties** \- Ionization energy, electronegativity, atomic radius, etc.
  5. **Implementation Example** \- Manual calculation using NumPy

#### Learning Objectives

  * ✅ Understand Magpie descriptor calculation methods with formulas
  * ✅ List the 22 types of elemental properties
  * ✅ Explain the significance of weighted statistics

**[Read Chapter 2 →](<chapter-2.html>)**

* * *

### [Chapter 3: Elemental Property Databases and Featurizers](<chapter-3.html>)

**Difficulty** : Intermediate  
**Reading Time** : 30-35 minutes  
**Code Examples** : 10

#### Learning Content

  1. **Types of Elemental Property Databases** \- Magpie, Deml, Jarvis, Matscholar
  2. **matminer Featurizer API** \- ElementProperty, Stoichiometry, OxidationStates
  3. **Choosing Featurizers** \- Selection criteria based on application
  4. **Custom Featurizer Creation** \- How to add original elemental properties
  5. **Feature Preprocessing** \- Standardization, missing value handling

#### Learning Objectives

  * ✅ Choose among 3+ elemental property databases appropriately
  * ✅ Select matminer Featurizers based on application
  * ✅ Implement custom Featurizers

**[Read Chapter 3 →](<chapter-3.html>)**

* * *

### [Chapter 4: Integration with Machine Learning Models](<chapter-4.html>)

**Difficulty** : Intermediate to Advanced  
**Reading Time** : 30-35 minutes  
**Code Examples** : 12

#### Learning Content

  1. **Model Selection Criteria** \- Random Forest, XGBoost, LightGBM, Neural Networks
  2. **Hyperparameter Optimization** \- Optuna, GridSearchCV, BayesSearchCV
  3. **Feature Importance Analysis** \- SHAP, LIME, Permutation Importance
  4. **Ensemble Methods** \- Bagging, Boosting, Stacking
  5. **Performance Evaluation Metrics** \- MAE, RMSE, R², Cross-validation

#### Learning Objectives

  * ✅ Select appropriate machine learning models based on tasks
  * ✅ Execute hyperparameter optimization with Optuna
  * ✅ Interpret feature importance using SHAP values

**[Read Chapter 4 →](<chapter-4.html>)**

* * *

### [Chapter 5: Python Practice - matminer Workflow](<chapter-5.html>)

**Difficulty** : Intermediate to Advanced  
**Reading Time** : 35-45 minutes  
**Code Examples** : 15 (all executable)

#### Learning Content

  1. **Environment Setup** \- Anaconda, pip, Google Colab
  2. **Data Preparation** \- Materials Project API, OQMD dataset
  3. **Feature Generation Pipeline** \- Composition formula → Magpie descriptors → Standardization
  4. **Model Training and Evaluation** \- Formation energy prediction, band gap prediction
  5. **Performance Comparison with GNNs** \- Accuracy, speed, interpretability
  6. **Hybrid Approach** \- Composition features + Structure features

#### Learning Objectives

  * ✅ Build end-to-end prediction workflows with matminer
  * ✅ Perform property prediction with Materials Project data (R² > 0.85)
  * ✅ Quantitatively compare composition-based and GNN performance
  * ✅ Achieve improved accuracy with hybrid models

**[Read Chapter 5 →](<chapter-5.html>)**

* * *

## Overall Learning Outcomes

Upon completing this series, you will acquire the following skills and knowledge:

### Knowledge Level (Understanding)

  * ✅ Explain the theoretical foundation and history of composition-based features
  * ✅ Understand the mathematical definition of Magpie descriptors
  * ✅ Know the types and characteristics of elemental property databases
  * ✅ Understand criteria for choosing between composition-based and GNN approaches

### Practical Skills (Doing)

  * ✅ Generate feature vectors from composition using matminer
  * ✅ Extend features by combining multiple Featurizers
  * ✅ Perform property prediction with machine learning models (RF/XGBoost/NN)
  * ✅ Interpret prediction rationale using SHAP values
  * ✅ Quantitatively compare performance with GNNs

### Application Ability (Applying)

  * ✅ Design appropriate features for new materials discovery tasks
  * ✅ Build hybrid models (composition + structure)
  * ✅ Design prediction workflows combining experimental data
  * ✅ Apply feature engineering to industrial applications (battery materials, catalysts)

* * *

## Frequently Asked Questions (FAQ)

Q1: Should I use composition-based features or GNNs (structure-based)?

**A:** It depends on the task and data characteristics:

  * **Composition-based** advantages: (1) Materials discovery with unknown structures, (2) High-speed screening (1 million compound scale), (3) Limited data cases (<1000 samples)
  * **GNN** advantages: (1) Properties with strong structure dependence (elastic modulus, thermal conductivity), (2) Accuracy priority, (3) Sufficient data available (>10000 samples)
  * **Hybrid** is strongest: Using both composition and GNN features improves accuracy (implemented in Chapter 5)

Q2: Aren't 145 dimensions of Magpie descriptors too many? Concerns about overfitting?

**A:** This is rarely a problem in practice:

  * 145 dimensions is considered low in modern machine learning (GNNs learn thousands of dimensions in embeddings)
  * Elemental properties have physical meaning, unlike random high dimensions
  * Dimensionality reduction possible with regularization (L1/L2) or feature selection
  * Good performance reported experimentally even with 100-1000 samples

Q3: Are there libraries other than matminer?

**A:** Yes, these are available:

  * **DScribe** : SOAP, MBTR, ACSF descriptors (supports molecules and crystals)
  * **CFID** : Composition-based Feature Identifier
  * **Pymatgen** : matminer's foundation library (low-level API)
  * **XenonPy** : Integrates deep learning and feature generation

This series focuses on the most widely-used matminer.

Q4: Can Magpie descriptors be calculated automatically from chemical formulas (e.g., Fe2O3)?

**A:** Yes, it's easy with matminer:
    
    
    from matminer.featurizers.composition import ElementProperty
    
    featurizer = ElementProperty.from_preset("magpie")
    features = featurizer.featurize_dataframe(df, col_id="composition")
    # Automatically generates 145-dimensional vectors from df["composition"] column (Fe2O3, etc.)

Chapter 5 provides detailed code examples.

Q5: What is the prediction accuracy difference between GNNs (CGCNN, MPNN) and composition-based features?

**A:** Depends on the dataset and task, but representative benchmark results:

Task | Magpie + RF | CGCNN | Hybrid  
---|---|---|---  
Formation Energy (OQMD) | MAE 0.12 eV | MAE 0.039 eV | MAE 0.035 eV  
Band Gap (Materials Project) | MAE 0.45 eV | MAE 0.39 eV | MAE 0.36 eV  
Inference Speed (1M compounds) | 10 min | 100 min | 110 min  
  
Detailed comparison experiments are conducted in Chapter 5.

Q6: What are the differences between elemental property databases (Magpie/Deml/Jarvis)?

**A:** Characteristics of each database:

  * **Magpie** (Ward+ 2016): 22 elemental properties, most widely used, proven track record in Materials Project
  * **Deml** (Deml+ 2016): Considers oxidation states, particularly strong for oxides
  * **Jarvis** (Choudhary+ 2020): Includes DFT calculation values, latest elemental properties
  * **Matscholar** (Tshitoyan+ 2019): Element embeddings extracted from 2 million papers using NLP

Chapter 3 implements usage of each database.

Q7: Can composition-based features be used for transfer learning?

**A:** Yes, the following approaches are effective:

  * **Pre-training** : Train on Materials Project (60k compounds) → Fine-tune on experimental data (100 samples)
  * **Domain Adaptation** : Train on inorganic materials → Apply to organic-inorganic hybrids
  * **Meta-learning** : Learn common feature representations across multiple tasks (formation energy, band gap, elastic modulus)

Chapter 4 implements transfer learning using XGBoost and neural networks.

Q8: Can custom elemental properties (e.g., rarity, cost) be added?

**A:** Yes, you can create custom Featurizers by inheriting matminer's BaseFeaturizer:
    
    
    from matminer.featurizers.base import BaseFeaturizer
    
    class CustomElementProperty(BaseFeaturizer):
        def featurize(self, comp):
            # Reference custom elemental property database
            rarity = get_element_rarity(comp)
            cost = get_element_cost(comp)
            return [rarity, cost]

Chapter 3 provides detailed custom Featurizer implementation examples.

Q9: How is interpretability (Explainability) with composition-based features?

**A:** There are aspects that are easier to interpret than GNNs:

  * **SHAP Values** : Clear which elemental properties (e.g., average electronegativity) contributed to predictions
  * **Physical Meaning** : Chemical interpretations possible, such as "higher ionization energy leads to lower formation energy"
  * **Integration with Domain Knowledge** : Direct utilization of periodic table knowledge

Chapter 4 implements interpretability analysis using SHAP/LIME.

Q10: After completing this series, what learning resources should I pursue next?

**A:** The following learning paths are recommended:

  * **Comparison with GNNs** : → Quantitatively compare both methods
  * **Extension to Deep Learning** : [GNN Introduction Series](<index.html>) → Learn CGCNN, MPNN
  * **Practical Application** : → Apply to actual projects
  * **Paper Implementation** : Reproduce Ward+ (2016) "A general-purpose machine learning framework for predicting properties of inorganic materials"

* * *

## Prerequisites

To effectively learn this series, the following prerequisites are recommended:

### Required (Must Have)

  * ✅ **Python Basics** : Basic operations with NumPy, Pandas, Matplotlib
  * ✅ **Materials Science Basics** : Concepts of chemical composition, periodic table, elemental properties
  * ✅ **Machine Learning Basics** : Concepts of supervised learning, regression/classification, cross-validation

### Recommended (Nice to Have)

  * 📚 **scikit-learn** : Experience with Random Forest, model evaluation
  * 📚 **Statistics** : Understanding of mean, variance, standard deviation
  * 📚 **Crystallography** : Basics of lattices, symmetry (reviewed in Chapter 1)

### Not Required

  * ❌ **Deep Learning** : GNN knowledge not required (composition-based focuses on classical machine learning)
  * ❌ **Quantum Chemistry** : DFT calculation experience not required

* * *

## Related Series

#### 🔗 Integrated Learning with GNN Series

By learning this series together with the [GNN Introduction Series](<index.html>), you can grasp the complete picture of material features:

  * **Introduction to Composition-Based Features** (this series): Property prediction from chemical composition
  * **GNN Introduction Series** : Property prediction from 3D structures (CGCNN, MPNN, SchNet)
  * **Composition vs GNN Comparison Series** (coming soon): Quantitative benchmarking of both methods

### Recommended Learning Order

  1. **Introduction to Composition-Based Features** (this series) → Build fundamentals
  2. **[GNN Introduction Series](<index.html>)** → Learn structure-based methods
  3. **Composition vs GNN Comparison Series** (coming soon) → Master when to use which
  4. **Materials Screening Workflow** (coming soon) → Practical application

* * *

## Let's Get Started!

Are you ready? Start with Chapter 1 and begin your journey into the world of composition-based features!

**[Chapter 1: Fundamentals of Composition-Based Features →](<chapter-1.html>)**

* * *

**Update History**

  * **2025-11-02** : v1.0 Initial release

* * *

**Your materials discovery journey starts here!**
