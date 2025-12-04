---
title: Drug Discovery-Specific MI Methods
chapter_title: Drug Discovery-Specific MI Methods
subtitle: From Molecular Descriptors to Generative Models
---

# Chapter 2: Drug Discovery-Specific MI Methods

This chapter covers Drug Discovery. You will learn 4 types of molecular representations (SMILES, QSAR principles, and ADMET 5 items specifically.

**From Molecular Representation to Generative Models - Technical Foundation of AI-Driven Drug Discovery**

## 2.1 Molecular Representation and Descriptors

The first step in drug discovery MI is converting chemical structures into formats that computers can understand. This choice of "molecular representation" significantly affects model performance.

### 2.1.1 SMILES Representation

**SMILES (Simplified Molecular Input Line Entry System)** is the most widely used format for representing molecular structures as strings.

**Basic Rules:**
    
    
    Atoms: C (carbon), N (nitrogen), O (oxygen), S (sulfur), etc.
    Bonds: Single bond (omitted), Double bond (=), Triple bond (#)
    Ring structures: Marked with numbers (e.g., C1CCCCC1 = cyclohexane)
    Branches: Expressed with parentheses (e.g., CC(C)C = isobutane)
    Aromatic: Lowercase letters (e.g., c1ccccc1 = benzene)
    

**Examples:**

Compound | SMILES | Structural Features  
---|---|---  
Ethanol | `CCO` | 2 carbons + hydroxyl group  
Aspirin | `CC(=O)OC1=CC=CC=C1C(=O)O` | Ester + carboxylic acid  
Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | Purine scaffold + methyl groups  
Ibuprofen | `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` | Aromatic ring + chiral center  
Penicillin G | `CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O` | β-lactam ring + thiazolidine ring  
  
**Advantages:** \- ✅ Compact (complex molecules in dozens of characters) \- ✅ Human readable and writable \- ✅ Suitable for database searches \- ✅ Uniqueness (Canonical SMILES)

**Disadvantages:** \- ❌ No 3D structural information (conformations lost) \- ❌ Difficulty distinguishing tautomers \- ❌ Existence of invalid SMILES (syntax errors)

### 2.1.2 Molecular Fingerprints

Molecular fingerprints represent molecules as bit vectors (sequences of 0/1). Widely used for similarity searches and QSAR.

#### ECFP (Extended Connectivity Fingerprints)

**Principle:** 1\. Start from each atom 2\. Hash neighborhood structure of radius R (typically 2-4) 3\. Convert to bit vector (length 1024-4096)

**Example: ECFP4 (radius 2, 4-step neighborhood)**
    
    
    Molecule: CCO (ethanol)
    
    Atom 1 (C):
      - Radius 0: C
      - Radius 1: C-C
      - Radius 2: C-C-O
    
    Atom 2 (C):
      - Radius 0: C
      - Radius 1: C-C, C-O
      - Radius 2: C-C-O, C-O
    
    Atom 3 (O):
      - Radius 0: O
      - Radius 1: O-C
      - Radius 2: O-C-C
    
    → Hash these to determine bit positions
    → Set corresponding bits to 1
    

**Types:**

Fingerprint | Size | Features | Applications  
---|---|---|---  
ECFP4 | 1024-4096 bit | Radius 2 environment | Similarity search, QSAR  
ECFP6 | 1024-4096 bit | Radius 3 environment | More precise structure recognition  
MACCS keys | 166 bit | Fixed substructures | Fast search, diversity analysis  
RDKit Fingerprint | 2048 bit | Path-based | General-purpose QSAR  
Morgan Fingerprint | Variable | ECFP implementation | RDKit standard  
  
**Tanimoto Coefficient for Similarity:**
    
    
    Tanimoto Coefficient = (A ∩ B) / (A ∪ B)
    
    A, B: Fingerprint bit vectors of two molecules
    ∩: Bit AND (number of bits that are 1 in both)
    ∪: Bit OR (number of bits that are 1 in at least one)
    
    Range: 0 (completely different) ~ 1 (perfect match)
    

**Practical Thresholds:** \- Tanimoto > 0.85: Very similar (same compound class) \- 0.70-0.85: Similar (possible similar pharmacological activity) \- 0.50-0.70: Somewhat similar \- < 0.50: Different

#### MACCS keys

**Features:** \- Represents presence/absence of 166 fixed substructures \- Examples: benzene ring, carboxylic acid, amino group, halogens, etc.

**Advantages:** \- Interpretable (know which substructures are present) \- Fast computation \- Chemically meaningful similarity

**Disadvantages:** \- Low information content (only 166 bits) \- Difficult to handle novel scaffolds

### 2.1.3 3D Descriptors

3D descriptors quantify the three-dimensional structure of molecules.

**Major 3D Descriptors:**

  1. **Molecular Surface Area** \- TPSA (Topological Polar Surface Area): Polar surface area \- Prediction: Oral absorption (favorable when TPSA < 140 Å²)

  2. **Volume and Shape** \- Molecular Volume \- Sphericity \- Aspect Ratio

  3. **Charge Distribution** \- Partial Charges: Gasteiger, MMFF \- Dipole Moment \- Quadrupole Moment

  4. **Pharmacophore** \- Hydrogen bond donor/acceptor positions \- Hydrophobic regions \- Positive/negative charge centers \- Aromatic ring orientations

**Example: Descriptors Used in Lipinski's Rule of Five**
    
    
    Molecular Weight (MW): < 500 Da
    LogP (lipophilicity): < 5
    Hydrogen Bond Donors (HBD): < 5
    Hydrogen Bond Acceptors (HBA): < 10
    
    Compounds satisfying these tend to have high oral absorption
    

### 2.1.4 Graph Representation (For Graph Neural Networks)

Molecules are represented as mathematical graphs.

**Definition:**
    
    
    G = (V, E)
    
    V: Vertices = Atoms
    E: Edges = Bonds
    
    Each vertex v has feature vector h_v
    Each edge e has feature vector h_e
    

**Atom (Vertex) Features:** \- Atomic number (C=6, N=7, O=8, etc.) \- Atom type (C, N, O, S, F, Cl, Br, I) \- Hybridization (sp, sp2, sp3) \- Formal Charge \- Aromaticity (Aromatic or not) \- Hydrogen Count \- Number of lone pairs \- Chirality (R/S)

**Bond (Edge) Features:** \- Bond order (Single=1, Double=2, Triple=3, Aromatic=1.5) \- Bond type (Covalent, Ionic, etc.) \- Part of ring or not \- Stereochemistry (E/Z)

**Graph Advantages:** \- Direct representation of molecular structure \- Rotation and translation invariance \- Learnable with Graph Neural Networks

* * *

## 2.2 QSAR (Quantitative Structure-Activity Relationship)

### 2.2.1 Basic Principles of QSAR

**QSAR (Quantitative Structure-Activity Relationship)** expresses the quantitative relationship between molecular structure and biological activity using mathematical equations.

**Basic Hypothesis:**

> Similar molecular structures show similar biological activity (Similar Property Principle)

**General QSAR Equation:**
    
    
    Activity = f(Descriptors)
    
    Activity: Biological activity (IC50, EC50, Ki, etc.)
    Descriptors: Molecular descriptors (MW, LogP, TPSA, etc.)
    f: Mathematical function (linear regression, Random Forest, NN, etc.)
    

**Historical QSAR Equation (Hansch-Fujita Equation, 1962):**
    
    
    log(1/C) = a * logP + b * σ + c * Es + d
    
    C: Biological activity concentration (lower = higher activity)
    logP: Partition coefficient (lipophilicity)
    σ: Hammett constant (electronic effect)
    Es: Steric parameter
    a, b, c, d: Regression coefficients
    

### 2.2.2 QSAR Workflow
    
    
    ```mermaid
    flowchart TD
        A["Compound LibrarySMILES + Activity Data"] --> B["Calculate Molecular DescriptorsMW, LogP, ECFP, etc."]
        B --> C["Data SplitTrain 80% / Test 20%"]
        C --> D["Model TrainingRF, SVM, NN"]
        D --> E["Performance EvaluationR^2, MAE, ROC-AUC"]
        E --> F{Performance OK?}
        F -->|No| G["HyperparameterTuning"]
        G --> D
        F -->|Yes| H["Predict New CompoundsVirtual Screening"]
        H --> I["Experimental ValidationTop Candidates Only"]
    
        style A fill:#e3f2fd
        style E fill:#fff3e0
        style H fill:#e8f5e9
        style I fill:#ffebee
    ```

### 2.2.3 Types of QSAR Models

#### Classification Models (Predicting Active/Inactive)

**Purpose:** Predict whether a compound is active or inactive

**Evaluation Metrics:**
    
    
    ROC-AUC: 0.5 (random) ~ 1.0 (perfect)
    Target: > 0.80
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    TP: True Positive (correctly predict active)
    FP: False Positive (incorrectly predict inactive as active)
    FN: False Negative (incorrectly predict active as inactive)
    

**Typical Problem Setting:** \- IC50 < 1 μM → Active (1) \- IC50 ≥ 1 μM → Inactive (0)

#### Regression Models (Predicting Activity Values)

**Purpose:** Predict continuous values like IC50, Ki, EC50

**Evaluation Metrics:**
    
    
    R² (Coefficient of determination): 0 (no correlation) ~ 1 (perfect)
    Target: > 0.70
    
    MAE (Mean Absolute Error) = Σ|y_pred - y_true| / n
    RMSE (Root Mean Square Error) = √(Σ(y_pred - y_true)² / n)
    
    y_pred: Predicted value
    y_true: True value
    n: Number of samples
    

**Log Transformation:** In most cases, activity values (IC50, etc.) are log-transformed before regression
    
    
    pIC50 = -log10(IC50[M])
    
    Example:
    IC50 = 1 nM = 10^-9 M → pIC50 = 9.0
    IC50 = 100 nM = 10^-7 M → pIC50 = 7.0
    
    Range: typically 4-10 (10 μM ~ 0.1 nM)
    

### 2.2.4 Comparison of Machine Learning Methods

Method | Accuracy | Speed | Interpretability | Data Size | Recommended Cases  
---|---|---|---|---|---  
Linear Regression | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100+ | Baseline, linear relationships  
Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 1K-100K | Medium data, non-linear  
SVM (RBF kernel) | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 1K-10K | Small-medium, high-dimensional  
Neural Network | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | 10K+ | Large data, complex relationships  
LightGBM/XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 1K-1M | Kaggle winners, fast  
Graph Neural Network | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 10K+ | Molecular graphs, SOTA  
  
**Real-World Performance (ChEMBL Data):**
    
    
    Task: Kinase inhibition activity prediction (classification)
    
    Linear Regression: ROC-AUC = 0.72
    Random Forest:      ROC-AUC = 0.87
    SVM (RBF):          ROC-AUC = 0.85
    Neural Network:     ROC-AUC = 0.89
    LightGBM:           ROC-AUC = 0.90
    GNN (MPNN):         ROC-AUC = 0.92
    
    Training data: 10,000 compounds
    Test data: 2,000 compounds
    

### 2.2.5 QSAR Applicability Domain

**Problem:** Predictions for compounds very different from training data are unreliable

**Applicability Domain Definition:**
    
    
    AD = {x | included in chemical space of training data}
    
    Determination methods:
    1. Tanimoto distance: Similarity to nearest neighbor > 0.3
    2. Leverage: h_i < 3p/n (h: hat matrix, p: number of features, n: sample size)
    3. Standardized distance: d_i < 3σ (σ: standard deviation)
    

**Compounds Outside AD:** \- Low prediction reliability \- Experimental validation required \- Consider model retraining

**Example:**
    
    
    Training data: Kinase inhibitors (mainly ATP-competitive)
    Novel compound: Allosteric inhibitor (different binding site)
    
    → Likely outside AD
    → Prediction accuracy not guaranteed
    

* * *

## 2.3 ADMET Prediction

### 2.3.1 Importance of ADMET

**30% of clinical trial failures are due to ADMET issues**

Phase | Main Failure Causes | ADMET-Related  
---|---|---  
Phase I | Toxicity (hepatotoxicity, cardiotoxicity) | 30%  
Phase II | Lack of efficacy, poor PK | 20%  
Phase III | Long-term toxicity, side effects | 15%  
  
**Failure Reduction Through ADMET Prediction:** \- Early ADMET evaluation (during lead discovery stage) \- Eliminate problematic compounds \- Reduce development costs by $100M-500M

### 2.3.2 Absorption

#### Caco-2 Permeability

**Caco-2 Cells:** Colon cancer cell line, model of intestinal epithelial cells

**Measurement:** Rate of permeation through cell layer (Papp: apparent permeability coefficient)
    
    
    Papp [cm/s]
    
    Papp > 10^-6 cm/s: High permeability (good absorption)
    10^-7 < Papp < 10^-6: Moderate
    Papp < 10^-7 cm/s: Low permeability (poor absorption)
    

**Prediction Model:**
    
    
    # Simple prediction equation (R² = 0.75)
    log Papp = 0.5 * logP - 0.01 * TPSA + 0.3 * HBD - 2.5
    
    logP: Partition coefficient (lipophilicity)
    TPSA: Topological polar surface area
    HBD: Number of hydrogen bond donors
    

**Machine Learning Prediction Accuracy:** \- Random Forest: R² = 0.80-0.85 \- Neural Network: R² = 0.82-0.88 \- Graph NN: R² = 0.85-0.90

#### Oral Bioavailability (F%)

**Definition:** Fraction of administered dose that reaches systemic circulation
    
    
    F% = (AUC_oral / AUC_iv) * (Dose_iv / Dose_oral) * 100
    
    AUC: Area under the plasma concentration-time curve
    

**Target Values:** \- F% > 30%: Acceptable range \- F% > 50%: Good \- F% > 70%: Excellent

**Predictive Factors:** 1\. Solubility 2\. Permeability 3\. First-pass metabolism (hepatic first-pass metabolism) 4\. P-glycoprotein efflux

**BCS Classification (Biopharmaceutics Classification System):**

Class | Solubility | Permeability | Example | F%  
---|---|---|---|---  
I | High | High | Metoprolol | > 80%  
II | Low | High | Ibuprofen | 50-80%  
III | High | Low | Atenolol | 30-50%  
IV | Low | Low | Tacrolimus | < 30%  
  
### 2.3.3 Distribution

#### Plasma Protein Binding

**Definition:** Percentage of drug bound to plasma proteins (albumin, α1-acid glycoprotein)
    
    
    Binding rate = (Bound / Total) * 100%
    
    Bound: Bound drug concentration
    Total: Total drug concentration
    

**Clinical Significance:** \- High binding (> 90%): Low free drug (active form) \- Risk of drug-drug interactions (binding site competition) \- Affects volume of distribution

**Prediction Models:** \- Random Forest: R² = 0.65-0.75 \- Deep Learning: R² = 0.70-0.80

#### Blood-Brain Barrier (BBB) Permeability

**LogBB (Brain/Blood Ratio):**
    
    
    LogBB = log10(C_brain / C_blood)
    
    LogBB > 0: Concentrates in brain (favorable for CNS drugs)
    LogBB < -1: Does not penetrate brain (avoid CNS side effects)
    

**Predictive Factors:** \- Molecular weight (< 400 Da favorable) \- TPSA (< 60 Å² favorable) \- LogP (2-5 optimal) \- Number of hydrogen bonds (fewer is better)

**Machine Learning Prediction:** \- Random Forest: R² = 0.70-0.80 \- Neural Network: R² = 0.75-0.85

### 2.3.4 Metabolism

#### CYP450 Inhibition

**CYP450 (Cytochrome P450):** Major metabolic enzyme in liver

**Major Isoforms:**

CYP | Percentage of Substrate Drugs | Examples  
---|---|---  
CYP3A4 | 50% | Statins, immunosuppressants  
CYP2D6 | 25% | β-blockers, antidepressants  
CYP2C9 | 15% | Warfarin, NSAIDs  
CYP2C19 | 10% | Proton pump inhibitors  
  
**Problem with Inhibition:** \- Drug-drug interactions (DDI) \- Increased blood concentration of co-administered drugs → toxicity risk

**Prediction (Classification Problem):**
    
    
    Inhibitor: IC50 < 10 μM
    Non-inhibitor: IC50 ≥ 10 μM
    
    Prediction accuracy:
    Random Forest: ROC-AUC = 0.85-0.90
    Neural Network: ROC-AUC = 0.87-0.92
    

#### Metabolic Stability

**Measurement:** Percentage remaining after incubation with liver microsomes
    
    
    t1/2 (half-life) [min]
    
    t1/2 > 60 min: Stable (slow metabolism)
    30 < t1/2 < 60: Moderate
    t1/2 < 30 min: Unstable (fast metabolism)
    

**Prediction:** \- Random Forest: R² = 0.55-0.65 (somewhat difficult) \- Graph NN: R² = 0.60-0.70

### 2.3.5 Excretion

#### Renal Clearance

**Definition:** Rate of drug removal by kidneys
    
    
    CL_renal [mL/min/kg]
    
    CL_renal > 10: High clearance (rapidly excreted)
    1 < CL_renal < 10: Moderate
    CL_renal < 1: Low clearance
    

**Influencing Factors:** \- Molecular weight (smaller = easier excretion) \- Polarity (higher = easier excretion) \- Renal transporter substrate

#### Half-Life (t1/2)

**Definition:** Time for plasma concentration to decrease by half
    
    
    t1/2 = 0.693 / (CL / Vd)
    
    CL: Clearance (systemic)
    Vd: Volume of distribution
    

**Clinical Significance:** \- t1/2 < 2 h: Frequent dosing required (3-4 times/day) \- 2 < t1/2 < 8 h: Twice daily dosing \- t1/2 > 8 h: Once daily dosing possible

**Prediction Accuracy:** \- Random Forest: R² = 0.55-0.65 (difficult) \- Complex combination of pharmacokinetic parameters

### 2.3.6 Toxicity

#### hERG Inhibition (Cardiotoxicity)

**hERG (human Ether-à-go-go-Related Gene):** Cardiac potassium channel

**Result of Inhibition:** QT prolongation → fatal arrhythmia (Torsades de pointes)

**Risk Assessment:**
    
    
    IC50 < 1 μM: High risk (development termination)
    1 < IC50 < 10 μM: Medium risk (careful evaluation needed)
    IC50 > 10 μM: Low risk (safe)
    

**Prediction Accuracy (Highest Accuracy in ADMET Predictions):** \- Random Forest: ROC-AUC = 0.85-0.90 \- Deep Learning: ROC-AUC = 0.90-0.95 \- Graph NN: ROC-AUC = 0.92-0.97

**Structural Alerts (Structures Prone to hERG Inhibition):** \- Basic nitrogen atom (pKa > 7) \- Hydrophobic aromatic rings \- Flexible linker

#### Hepatotoxicity

**DILI (Drug-Induced Liver Injury):** Drug-induced liver damage

**Mechanisms:** 1\. Formation of reactive metabolites 2\. Mitochondrial toxicity 3\. Bile acid transport inhibition

**Prediction (Classification Problem):**
    
    
    Hepatotoxic drug classification:
    
    Random Forest: ROC-AUC = 0.75-0.85
    Neural Network: ROC-AUC = 0.78-0.88
    
    Challenge: Data shortage (many cases discovered post-approval)
    

#### Mutagenicity (Ames Test)

**Ames Test:** Mutagenicity test using bacteria

**Prediction:**
    
    
    Positive: Mutagenic (carcinogenicity risk)
    Negative: Non-mutagenic
    
    Random Forest: ROC-AUC = 0.80-0.90
    Deep Learning: ROC-AUC = 0.85-0.93
    

**Structural Alerts:** \- Nitro group (-NO2) \- Azo group (-N=N-) \- Epoxide \- Alkylating agents

* * *

## 2.4 Molecular Generative Models

### 2.4.1 Need for Generative Models

**Traditional Virtual Screening:** \- Select from existing compound libraries (10^6-10^9 compounds) \- Limitation: Cannot discover compounds not in library

**Generative Model Approach:** \- Generate novel molecules directly \- Freely explore chemical space (10^60 molecules) \- Design molecules with desired properties

**Types of Generative Models:**
    
    
    ```mermaid
    flowchart TD
        A[Molecular Generative Models] --> B["VAEVariational Autoencoder"]
        A --> C["GANGenerative Adversarial Network"]
        A --> D["TransformerGPT-like"]
        A --> E["Reinforcement LearningRL"]
        A --> F["Graph GenerationGNN-based"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fff9c4
        style F fill:#fce4ec
    ```

### 2.4.2 VAE (Variational Autoencoder)

**Architecture:**
    
    
    SMILES → Encoder → Latent space (z) → Decoder → SMILES'
    
    Encoder: Compresses SMILES string to low-dimensional vector (z)
    Latent space: Continuous chemical space (typically 50-500 dimensions)
    Decoder: Reconstructs SMILES string from vector (z)
    

**Training Objective:**
    
    
    Loss = Reconstruction Loss + KL Divergence
    
    Reconstruction Loss: Error in reconstructing input SMILES
    KL Divergence: Regularization of latent space (approach normal distribution)
    

**Molecule Generation Workflow:** 1\. Train VAE on known molecules (training data) 2\. Sample in latent space (z ~ N(0, I)) 3\. Generate novel SMILES with Decoder 4\. Validity check (parseable by RDKit?) 5\. Property prediction (QSAR models, ADMET models) 6\. Select favorable molecules

**Advantages:** \- ✅ Optimization possible in continuous latent space \- ✅ Explore around known molecules (generate similar molecules) \- ✅ Generate intermediate molecules via interpolation

**Disadvantages:** \- ❌ High rate of invalid SMILES generation (30-50%) \- ❌ Difficult to generate novel scaffolds (depends on training data)

**Example: ChemVAE (Gómez-Bombarelli et al., 2018)**
    
    
    Training data: ZINC 250K compounds
    Latent space: 196 dimensions
    Valid SMILES generation rate: 68%
    Application: Property optimization using penalty method (LogP, QED)
    

### 2.4.3 GAN (Generative Adversarial Network)

**Two Networks:**
    
    
    Generator: Noise → Molecular SMILES
    Discriminator: SMILES → Real or Fake?
    
    Adversarial learning:
    - Generator: Generate molecules to fool Discriminator
    - Discriminator: Distinguish Real (training data) from Fake (generated)
    

**Training Process:**
    
    
    1. Generator: Random noise → SMILES generation
    2. Discriminator: Real/Fake classification
    3. Generator: Update with Discriminator's gradient (make Fake "Real-like")
    4. Discriminator: Improve classification accuracy
    5. Repeat → Generator produces molecules similar to training data
    

**Advantages:** \- ✅ Diverse molecule generation (with mode collapse countermeasures) \- ✅ High-quality molecules (filtering effect by Discriminator)

**Disadvantages:** \- ❌ Training instability (mode collapse, gradient vanishing) \- ❌ Difficult to evaluate (quantifying generation quality)

**Example: ORGANIC (Guimaraes et al., 2017)**
    
    
    Generator: LSTM (sequential SMILES generation)
    Discriminator: CNN (SMILES string classification)
    Application: Combined with reinforcement learning for property optimization
    

**Insilico Medicine's Chemistry42:** \- GAN-based \- Discovered IPF treatment candidate in 18 months \- Technology: WGAN-GP (Wasserstein GAN with Gradient Penalty)

### 2.4.4 Transformer (GPT-like Models)

**Principle:** \- Treat SMILES like natural language \- Capture context with Attention mechanism \- Large-scale pre-training (10^6-10^7 molecules)

**Architecture:**
    
    
    SMILES: C C ( = O ) O [EOS]
    ↓
    Tokenization: [C] [C] [(] [=] [O] [)] [O] [EOS]
    ↓
    Transformer Encoder/Decoder
    ↓
    Next token prediction: P(token_i+1 | token_1, ..., token_i)
    

**Generation Method:** 1\. Start from start token ([SOS]) 2\. Probabilistically sample next token 3\. Repeat until [EOS] 4\. Validate as SMILES

**Advantages:** \- ✅ High valid SMILES generation rate (> 90%) \- ✅ Benefits of large-scale pre-training (transfer learning) \- ✅ Controllable generation (conditional generation)

**Disadvantages:** \- ❌ Computational cost (large models) \- ❌ Difficult to control novelty

**Examples:** 1\. **ChemBERTa (HuggingFace, 2020)** \- Pre-training: 10 million SMILES (PubChem) \- Fine-tuning: High-accuracy QSAR with 100 samples

  2. **MolGPT (Bagal et al., 2021)** \- GPT-2 architecture \- Conditional generation (specify property values)

  3. **SMILES-BERT (Wang et al., 2019)** \- Masked Language Model (MLM task) \- High accuracy with small data via transfer learning

### 2.4.5 Reinforcement Learning (RL)

**Setup:**
    
    
    Agent: Molecular generative model
    Environment: Chemical space
    Action: Select next token (SMILES generation)
    State: Current SMILES prefix
    Reward: Properties of generated molecule (QED, LogP, ADMET, etc.)
    

**RL + Generative Model Flow:**
    
    
    ```mermaid
    flowchart LR
        A["AgentGenerative Model"] -->|Action| B[SMILES Generation]
        B -->|State| C["Property PredictionQSAR, ADMET"]
        C -->|Reward| D[Reward Calculation]
        D -->|Update| A
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#e8f5e9
    ```

**Reward Function Design:**
    
    
    Reward = w1 * QED + w2 * LogP_score + w3 * SA_score + w4 * ADMET_score
    
    QED: Quantitative Estimate of Druglikeness
    LogP_score: Lipophilicity score (2-5 optimal)
    SA_score: Synthetic Accessibility
    ADMET_score: ADMET model prediction
    
    w1, w2, w3, w4: Weights (multi-objective optimization)
    

**Learning Algorithms:** 1\. **Policy Gradient (REINFORCE)** ``` ∇J(θ) = E[∇log π(a|s) * R]

π: Policy (generation probability) R: Cumulative reward ```

  2. **Proximal Policy Optimization (PPO)** \- More stable learning \- Clip gradients to prevent large updates

**Advantages:** \- ✅ Explicit optimization (control via reward function) \- ✅ Multi-objective optimization possible \- ✅ Balance exploration and exploitation

**Disadvantages:** \- ❌ Difficult to design reward function \- ❌ Time-consuming training

**Example: ReLeaSE (Popova et al., 2018)**
    
    
    Technology: RL + LSTM
    Objective: Optimize LogP, QED
    Result: Generated molecules with properties superior to known drugs
    

### 2.4.6 Graph Generative Models

**Principle:** Directly generate molecular graphs

**Generation Process:**
    
    
    1. Initial state: Empty graph
    2. Add atom: Which atom type to add (C, N, O, etc.)
    3. Add bond: Between which atoms to add bonds
    4. Repeat: Until desired size is reached
    

**Major Methods:**

  1. **GraphRNN (You et al., 2018)** \- Sequential graph generation with RNN

  2. **Junction Tree VAE (Jin et al., 2018)** \- Generate molecular scaffold (Junction Tree) \- 100% valid SMILES generation rate (theoretically)

  3. **MoFlow (Zang & Wang, 2020)** \- Based on Normalizing Flows \- Exact probability density through invertible transformations

**Advantages:** \- ✅ Generate only valid molecules (incorporate chemical constraints of graphs) \- ✅ Can consider 3D structure

**Disadvantages:** \- ❌ High computational cost \- ❌ Complex implementation

* * *

## 2.5 Major Databases and Tools

### 2.5.1 ChEMBL

**Overview:** \- Bioactivity database (European Bioinformatics Institute, EBI) \- 2+ million compounds \- 15+ million bioactivity data points \- 14,000+ targets

**Data Structure:**
    
    
    Compound: ChEMBL ID, SMILES, InChI, molecular weight, etc.
    Assay: Assay type, target, cell line, etc.
    Activity: IC50, Ki, EC50, Kd, etc.
    Target: Protein, gene, cell, etc.
    

**API Access:**
    
    
    from chembl_webresource_client.new_client import new_client
    
    # Target search
    target = new_client.target
    kinases = target.filter(target_type='PROTEIN KINASE')
    
    # Get compound activity data
    activity = new_client.activity
    egfr_data = activity.filter(target_chembl_id='CHEMBL203', pchembl_value__gte=6)
    # pchembl_value ≥ 6 → IC50 ≤ 1 μM
    

**Use Cases:** \- **ChEMBL** : Bioactivity data, QSAR training \- **PubChem** : Chemical structures, literature search \- **DrugBank** : Approved drugs, clinical information \- **BindingDB** : Protein-ligand binding affinity

### 2.5.2 PubChem

**Overview:** \- Chemical information database (NIH, USA) \- 100+ million compounds \- Bioactivity assay data (PubChem BioAssay)

**Features:** \- Free access \- REST API, FTP provided \- 2D/3D structure data \- Literature links (PubMed)

**Applications:** \- Building compound libraries \- Structure search (similarity, substructure) \- Property data collection

### 2.5.3 DrugBank

**Overview:** \- Approved and clinical trial drug database (Canada) \- 14,000+ drugs \- Detailed pharmacokinetic, pharmacodynamic data

**Information:** \- ADMET properties \- Drug-drug interactions \- Target information \- Clinical trial status

**Applications:** \- Drug repurposing (finding new indications for approved drugs) \- ADMET training data \- Benchmarking

### 2.5.4 BindingDB

**Overview:** \- Protein-ligand binding affinity database \- 2.5+ million binding data points \- 9,000+ proteins

**Data Types:** \- Ki (inhibition constant) \- Kd (dissociation constant) \- IC50 (50% inhibitory concentration) \- EC50 (50% effective concentration)

**Applications:** \- Docking validation \- Binding affinity prediction model training

### 2.5.5 RDKit

**Overview:** \- Open-source cheminformatics library \- Python, C++, Java support \- Standard tool for drug discovery MI

**Main Functions:** 1\. **Molecular I/O** : Read/write SMILES, MOL, SDF 2\. **Descriptor calculation** : 200+ physicochemical properties 3\. **Molecular fingerprints** : ECFP, MACCS, RDKit FP 4\. **Substructure search** : SMARTS, MCS (maximum common substructure) 5\. **2D drawing** : Molecular structure visualization 6\. **3D structure generation** : ETKDG (distance geometry method)

**Usage Example:**
    
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    
    # Read SMILES
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # Aspirin
    
    # Calculate descriptors
    mw = Descriptors.MolWt(mol)  # 180.16
    logp = Descriptors.MolLogP(mol)  # 1.19
    
    # Molecular fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    # Similarity calculation
    mol2 = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')  # Ibuprofen
    similarity = DataStructs.TanimotoSimilarity(fp, fp2)  # 0.31
    

* * *

## 2.6 Overall Drug Discovery MI Workflow
    
    
    ```mermaid
    flowchart TB
        A["Target IdentificationDisease-related Protein"] --> B["Compound Library BuildingChEMBL, PubChem, In-house"]
        B --> C["Virtual ScreeningQSAR, Docking"]
        C --> D["Hit CompoundsTop 0.1-1%"]
        D --> E["ADMET Predictionin silico Evaluation"]
        E --> F{ADMET OK?}
        F -->|No| G["Structure OptimizationScaffold Hopping"]
        G --> E
        F -->|Yes| H["in vitro ValidationExperimental Confirmation"]
        H --> I{Activity OK?}
        I -->|No| J["Active LearningModel Update"]
        J --> C
        I -->|Yes| K[Lead Compound]
        K --> L["Lead OptimizationMulti-objective Optimization"]
        L --> M[Preclinical Studies]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style K fill:#e8f5e9
        style M fill:#fce4ec
    ```

**Time and Cost for Each Stage:**

Stage | Traditional | AI-Enabled | Reduction Rate  
---|---|---|---  
Target Identification | 1-2 years | 0.5-1 year | 50%  
Lead Discovery | 2-3 years | 0.5-1 year | 75%  
Lead Optimization | 2-3 years | 1-1.5 years | 50%  
Preclinical | 1-2 years | 0.5-1 year | 50%  
**Total** | **6-10 years** | **2.5-4.5 years** | **60-70%**  
  
**Cost Reduction:** \- Virtual Screening: $500M → $50M (90% reduction) \- ADMET failure reduction: $200M → $50M (75% reduction) \- Experimental number reduction: 1,000 experiments → 100 experiments (90% reduction)

* * *

## Learning Objectives Check

Upon completing this chapter, you will be able to explain:

### Basic Understanding (Remember & Understand)

  * ✅ Explain 4 types of molecular representations (SMILES, molecular fingerprints, 3D descriptors, Graph)
  * ✅ Understand QSAR principles and 5-step workflow
  * ✅ Explain ADMET 5 items specifically
  * ✅ Understand differences among 4 types of molecular generative models (VAE, GAN, Transformer, RL)
  * ✅ Grasp features of major databases (ChEMBL, PubChem, DrugBank, BindingDB)

### Practical Skills (Apply & Analyze)

  * ✅ Calculate Lipinski's Rule of Five and evaluate druglikeness
  * ✅ Calculate molecular similarity using Tanimoto coefficient
  * ✅ Interpret QSAR model performance metrics (R², ROC-AUC)
  * ✅ Judge development feasibility from ADMET prediction results
  * ✅ Appropriately select machine learning methods (RF, SVM, NN, GNN)

### Application Ability (Evaluate & Create)

  * ✅ Design workflow suitable for new drug discovery projects
  * ✅ Evaluate QSAR model Applicability Domain
  * ✅ Judge application scenarios for molecular generative models
  * ✅ Build datasets using ChEMBL API
  * ✅ Optimize overall drug discovery MI flow

* * *

## Exercises

### Easy (Basic Confirmation)

**Q1** : What compound does the following SMILES string represent?
    
    
    CCO
    

a) Methanol b) Ethanol c) Propanol d) Butanol

View Answer **Correct Answer**: b) Ethanol **Explanation**: \- `C`: Carbon (methyl group) \- `C`: Carbon (methylene group) \- `O`: Oxygen (hydroxyl group) Structure: CH3-CH2-OH = Ethanol **Other Options:** \- a) Methanol: `CO` \- c) Propanol: `CCCO` \- d) Butanol: `CCCCO` 

**Q2** : In Lipinski's Rule of Five, what is the upper limit for molecular weight?

View Answer **Correct Answer**: 500 Da **Lipinski's Rule of Five (Recap):** 1\. Molecular Weight (MW): **< 500 Da** 2\. LogP (lipophilicity): < 5 3\. Hydrogen Bond Donors (HBD): < 5 4\. Hydrogen Bond Acceptors (HBA): < 10 Compounds satisfying these tend to have high oral absorption. 

**Q3** : A compound has an hERG inhibition IC50 of 0.5 μM. How is this classified in risk assessment? a) Low risk b) Medium risk c) High risk

View Answer **Correct Answer**: c) High risk **hERG Risk Assessment Criteria:** \- IC50 < 1 μM: **High risk** (consider development termination) \- 1 < IC50 < 10 μM: Medium risk (careful evaluation needed) \- IC50 > 10 μM: Low risk (safe) IC50 = 0.5 μM < 1 μM → High risk hERG inhibition can cause fatal arrhythmia (Torsades de pointes), making it one of the most important toxicity evaluation items in drug discovery. 

### Medium (Application)

**Q4** : When comparing ECFP4 fingerprints (2048 bits) of two molecules, the following results were obtained. Calculate the Tanimoto coefficient.
    
    
    Molecule A: Number of positions with bit 1 = 250
    Molecule B: Number of positions with bit 1 = 280
    Number of positions with bit 1 in both = 120
    

View Answer **Correct Answer**: Tanimoto = 0.293 **Calculation:** 
    
    
    Tanimoto = (A ∩ B) / (A ∪ B)
    
    A ∩ B = 120 (number of bits that are 1 in both)
    A ∪ B = 250 + 280 - 120 = 410
    
    Tanimoto = 120 / 410 = 0.293
    

**Interpretation:** Tanimoto = 0.293 < 0.50 → Structurally different molecules Similarity guideline: \- > 0.85: Very similar \- 0.70-0.85: Similar \- 0.50-0.70: Somewhat similar \- **< 0.50: Different** (current case) 

**Q5** : A QSAR model was built and the following performance was obtained. Is this model practical?
    
    
    Training data: R² = 0.92, MAE = 0.3
    Test data: R² = 0.58, MAE = 1.2
    

View Answer **Correct Answer**: Not practical (overfitting) **Analysis:** \- Training data: R² = 0.92 (excellent) \- Test data: R² = 0.58 (insufficient) \- **Difference**: 0.92 - 0.58 = 0.34 (too large) **Problem: Overfitting** \- Model over-fitted to training data \- Low generalization performance on unknown data \- Test R² < 0.70 is not practical **Countermeasures:** 1\. Regularization (L1/L2, Dropout) 2\. Add training data 3\. Feature reduction (use only important descriptors) 4\. Simpler model (limit depth of Random Forest trees, etc.) 5\. Hyperparameter tuning with cross-validation 

**Q6** : A compound's Caco-2 permeability is Papp = 5 × 10^-7 cm/s. Evaluate this compound's oral absorption.

View Answer **Correct Answer**: Low to moderate permeability (somewhat poor absorption) **Caco-2 Permeability Criteria:** \- Papp > 10^-6 cm/s: **High permeability** (good absorption) \- 10^-7 < Papp < 10^-6 cm/s: **Moderate** (absorption possible but not optimal) \- Papp < 10^-7 cm/s: Low permeability (poor absorption) **Current Case:** Papp = 5 × 10^-7 cm/s 10^-7 < 5 × 10^-7 < 10^-6 → Moderate **Improvement Strategies:** 1\. Adjust lipophilicity (optimize LogP) 2\. Reduce TPSA (make polar surface area smaller) 3\. Reduce number of hydrogen bonds 4\. Formulation technology (nanoparticles, liposomes) However, oral drugs can still be developed with this value (e.g., atenolol). 

### Hard (Advanced)

**Q7** : When generating 10,000 novel molecules with ChemVAE (latent space 196 dimensions), 6,800 were valid SMILES. Of the generated molecules, 40% had a Tanimoto coefficient > 0.95 with the most similar molecule in the training data (ZINC 250K). How do you evaluate this result? What improvement strategies are available?

View Answer **Evaluation:** **Positive Aspects:** \- Valid SMILES generation rate = 6,800 / 10,000 = 68% → Typical ChemVAE performance (matches paper value of 68%) → Technically successful **Negative Aspects:** \- Tanimoto > 0.95 = 40% = 4,000 molecules → Very similar to training data (low novelty) → Too many "nearly copied" molecules **Conclusion:** Insufficient novelty. Close to regenerating known compounds, limited drug discovery value. **Improvement Strategies:** 1\. **Change Latent Space Sampling Strategy** ```python # Current: Sample from standard normal distribution z = np.random.randn(196) # Improvement: Sample from regions far from training data z = np.random.randn(196) * 2.0 # Increase variance ``` 2\. **Add Penalty Term** ``` Loss = Reconstruction + KL + λ * Novelty Penalty Novelty Penalty = -log(1 - max_tanimoto) (Higher penalty for higher maximum similarity to training data) ``` 3\. **Use Conditional VAE** \- Provide desired properties (LogP, MW, etc.) as conditions \- Explore regions far from training data in property space 4\. **Combine with Reinforcement Learning** \- Optimize VAE-generated molecules with RL \- Add novelty term to reward function ``` Reward = Activity + λ1 * Novelty + λ2 * Druglikeness ``` 5\. **Consider Junction Tree VAE** \- Superior at generating novel scaffolds \- 100% valid SMILES generation rate **Practical Example:** Insilico Medicine discovered IPF treatment candidates by combining GAN and RL, generating molecules with novel scaffolds different from training data. 

**Q8** : For the following drug discovery project, which machine learning method should be selected? Explain with reasoning.
    
    
    Project: Development of EGFR (epidermal growth factor receptor) kinase inhibitor
    Data: EGFR activity data from ChEMBL, 15,000 compounds
    Task: IC50 prediction (regression)
    Goal: R² > 0.80, prediction time < 1 sec/compound
    Additional requirement: Want model interpretability (which structures contribute to activity)
    

View Answer **Recommended Method: Random Forest** **Reasoning:** 1\. **Fit with Data Size** \- 15,000 compounds = medium-scale data \- Random Forest optimal performance at 1K-100K \- Neural Network excels at 10K+, but 15K is borderline \- Random Forest more stable (less prone to overfitting) 2\. **Achieving Performance Goal** \- ChEMBL EGFR activity prediction benchmark: \- Random Forest: R² = 0.82-0.88 (meets goal R² > 0.80) \- SVM: R² = 0.78-0.85 (somewhat unstable) \- Neural Network: R² = 0.85-0.90 (high but overkill) \- LightGBM: R² = 0.85-0.92 (best performance but lower interpretability) 3\. **Prediction Speed** \- Random Forest: < 0.1 sec/compound (10x faster than goal) \- Neural Network: 0.5-2 sec/compound (without GPU) \- Practical even for 1 million compound Virtual Screening 4\. **Interpretability (Most Important Requirement)** \- **Feature Importance**: ```python importances = model.feature_importances_ # Rank which descriptors are important for prediction # Example: LogP, TPSA, number of aromatic rings, etc. ``` \- **SHAP (SHapley Additive exPlanations)**: ```python import shap explainer = shap.TreeExplainer(model) shap_values = explainer.shap_values(X_test) # Visualize contribution of each feature for each compound ``` \- This provides insights like "hydrophobic sites fitting ATP binding pocket are important" **Implementation Example:** 
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: Q8: For the following drug discovery project, which machine 
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import shap
    
    # Model training
    rf = RandomForestRegressor(
        n_estimators=500,  # Number of trees (more = more stable)
        max_depth=20,      # Depth limit (prevent overfitting)
        min_samples_leaf=5,
        n_jobs=-1          # Parallelization
    )
    
    rf.fit(X_train, y_train)
    
    # Performance evaluation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 important features:")
    print(importances.head(10))
    
    # SHAP interpretation
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test[:100])  # Sample 100 compounds
    
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    # → Visualize which features contribute to high/low activity
    

**Alternative Options (Depending on Situation):** \- **If data increases to 100K+**: LightGBM \- Higher accuracy (R² > 0.90 possible) \- Fast (2-5x faster than Random Forest) \- Feature Importance also available \- **If interpretability is top priority**: Linear Regression with feature selection \- Coefficients directly interpretable \- However, performance around R² = 0.70 (doesn't meet goal) \- **If highest accuracy needed**: Graph Neural Network (MPNN) \- R² = 0.90-0.95 \- However, low interpretability, long training time (hours-days) \- Partial interpretation possible with Attention weights **Conclusion:** Random Forest optimal for balance of performance, speed, and interpretability. 

* * *

## Next Steps

In Chapter 2, you understood drug discovery-specific MI methods. In Chapter 3, you'll implement these methods in Python and actually run them. Through 30 code examples using RDKit and ChEMBL, you'll acquire practical skills.

**[Chapter 3: Implementing Drug Discovery MI in Python - RDKit& ChEMBL Practice →](<./chapter3-hands-on.html>)**

* * *

## References

  1. Gómez-Bombarelli, R., et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." _ACS Central Science_ , 4(2), 268-276.

  2. Guimaraes, G. L., et al. (2017). "Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models." _arXiv:1705.10843_.

  3. Jin, W., Barzilay, R., & Jaakkola, T. (2018). "Junction tree variational autoencoder for molecular graph generation." _ICML 2018_.

  4. Lipinski, C. A., et al. (2001). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." _Advanced Drug Delivery Reviews_ , 46(1-3), 3-26.

  5. Popova, M., et al. (2018). "Deep reinforcement learning for de novo drug design." _Science Advances_ , 4(7), eaap7885.

  6. Rogers, D., & Hahn, M. (2010). "Extended-connectivity fingerprints." _Journal of Chemical Information and Modeling_ , 50(5), 742-754.

  7. Wang, S., et al. (2019). "SMILES-BERT: Large scale unsupervised pre-training for molecular property prediction." _BCB 2019_.

  8. Zhavoronkov, A., et al. (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors." _Nature Biotechnology_ , 37(9), 1038-1040.

  9. Gaulton, A., et al. (2017). "The ChEMBL database in 2017." _Nucleic Acids Research_ , 45(D1), D945-D954.

  10. Landrum, G. (2023). RDKit: Open-source cheminformatics. https://www.rdkit.org

* * *

[Return to Series Table of Contents](<./index.html>) | [Return to Chapter 1](<./chapter1-background.html>) | [Proceed to Chapter 3 →](<./chapter3-hands-on.html>)
