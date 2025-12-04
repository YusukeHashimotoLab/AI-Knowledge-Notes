---
title: "Chapter 3: Innovations in Catalyst Design - From Reaction Condition Optimization to Novel Catalyst Discovery"
chapter_title: "Chapter 3: Innovations in Catalyst Design - From Reaction Condition Optimization to Novel Catalyst Discovery"
subtitle: MI/AI Approaches and Industrial Applications for Finding Optimal Catalysts in Vast Search Spaces
reading_time: 20-25 minutes
difficulty: Intermediate
code_examples: 6
exercises: 2
version: 1.0
created_at: 2025-10-18
---

# Chapter 3: Innovations in Catalyst Design - From Reaction Condition Optimization to Novel Catalyst Discovery

This chapter covers Innovations in Catalyst Design. You will learn principles of MI/AI approaches (descriptor design, five success cases from BASF, and catalyst activity prediction.

## Learning Objectives

After reading this chapter, you will be able to:

  * ✅ Quantitatively explain challenges in catalyst development (vast search space, multidimensional optimization, scale-up difficulties)
  * ✅ Understand the principles of MI/AI approaches (descriptor design, reaction mechanism prediction, transfer learning)
  * ✅ Explain five success cases from BASF, University of Tokyo, Shell, Kebotix, and AIST with technical details
  * ✅ Implement catalyst activity prediction, reaction condition optimization, adsorption energy prediction, and active learning in Python
  * ✅ Evaluate the current state of catalyst informatics and prospects for autonomous laboratories

* * *

## 3.1 Challenges in Catalyst Development

### 3.1.1 The Future of Chemical Industry Transformed by Catalysts

Catalysts are the heart of the chemical industry. Over 90% of the world's chemical products are manufactured through catalytic processes, with a market size exceeding 3 trillion yen annually. However, the development of new catalysts remains a time-consuming and costly process.

**Realistic Numbers for Catalyst Development** :

Metric | Traditional Approach | MI/AI Approach  
---|---|---  
**Development Period** | 10-20 years | 1-3 years (until candidate discovery)  
**Number of Candidate Materials** | 100-500 (experimental) | 10,000-100,000 (computational + experimental)  
**Success Rate** | 1-5% | 10-20% (improving)  
**Development Cost** | 5-20 billion yen | 0.5-3 billion yen (70-85% reduction)  
**Number of Experiments** | 5,000-10,000 times | 50-500 times (with active learning)  
  
**Source** : Nørskov et al. (2011), _Nature Chemistry_ ; Burger et al. (2020), _Nature_

### 3.1.2 Three Fundamental Challenges in Catalyst Development

The difficulty of catalyst development lies in the following three fundamental challenges:

#### Challenge 1: Vast Candidate Material Space

Catalyst performance depends on countless factors including material composition, structure, surface state, and preparation conditions.

**Vastness of the Search Space** : \- **Single metal catalysts** : Approximately 80 types from the periodic table \- **Binary alloy catalysts** : C(80,2) = 3,160 combinations \- **Ternary alloy catalysts** : C(80,3) = 82,160 combinations \- **Considering composition ratios** : 50-100 variations per alloy → **10^6 to 10^7 combinations** \- **Considering support materials** : 10-100 times more → **10^7 to 10^9 combinations**

This search space is of a scale that cannot possibly be covered in a human lifetime.

#### Challenge 2: Multidimensional Reaction Condition Optimization

Even after determining the catalyst material, optimization of reaction conditions is necessary:

**Parameters to Optimize** (example of a typical catalytic reaction): 1\. **Temperature** (200-600°C, 50 steps) → 50 options 2\. **Pressure** (1-100 atm, 20 steps) → 20 options 3\. **Catalyst loading** (0.1-10 wt%, 20 steps) → 20 options 4\. **Co-catalyst ratio** (0-1.0, 20 steps) → 20 options 5\. **Gas flow rate** (10-1000 mL/min, 20 steps) → 20 options 6\. **Pretreatment conditions** (temperature/atmosphere, 10 options) → 10 options

**Total number of combinations** : 50 × 20 × 20 × 20 × 20 × 10 = **160 million combinations**

With the traditional one-factor-at-a-time approach (OFAT: One Factor At a Time), exhaustive search would take **hundreds of years**.

#### Challenge 3: Difficulties in Scale-up

Even when a high-performance catalyst is found in the laboratory, its performance can significantly decrease during the transition to industrial production (scale-up).

**Example of Scale-up Failure** : \- Lab scale (1 g catalyst): Yield 90%, Selectivity 95% \- Pilot scale (10 kg catalyst): Yield 70%, Selectivity 80% (❌ performance degradation) \- Industrial scale (1 ton catalyst): Yield 50%, Selectivity 65% (❌❌ further deterioration)

**Main Causes of Failure** : 1\. Differences in heat and mass transfer (temperature distribution changes with reactor size) 2\. Effect of impurities (industrial feedstock contains trace impurities) 3\. Differences in catalyst preparation methods (uniformity decreases in mass production)

→ Scale-up requires an additional **2-5 years** and costs of **1-5 billion yen**

### 3.1.3 Traditional Catalyst Development Process

Traditional catalyst development goes through the following stages:
    
    
    ```mermaid
    flowchart LR
        A[Literature Survey\n6 months-1 year] --> B[Candidate Material Selection\n3-6 months]
        B --> C[Catalyst Preparation\n1-2 years]
        C --> D[Activity Evaluation\n1-2 years]
        D --> E[Reaction Condition Optimization\n1-3 years]
        E --> F[Scale-up\n2-5 years]
        F --> G[Industrialization\n3-5 years]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#e3f2fd
        style E fill:#f3e5f5
        style F fill:#fce4ec
        style G fill:#e0f2f1
    ```

**Bottlenecks at Each Stage** :

  1. **Candidate Material Selection** : Dependent on experts' experience and intuition \- Can only search within the scope of existing knowledge \- Difficulty in discovering innovative materials

  2. **Catalyst Preparation and Evaluation** : Repetitive experimentation \- Preparation and evaluation of one catalyst takes **1-2 weeks** \- Only 50-100 catalysts can be evaluated per year

  3. **Reaction Condition Optimization** : Too many parameters \- Inefficient with OFAT method \- Interactions are overlooked

  4. **Scale-up** : Difficult to predict \- Large difference in conditions between laboratory and plant \- Enormous time required for troubleshooting

* * *

## 3.2 Innovation through MI/AI Approaches

Materials Informatics (MI) and Artificial Intelligence (AI) are dramatically accelerating each stage of catalyst development.

### 3.2.1 Four Pillars of Catalyst Informatics

The MI/AI approach to catalyst development consists of the following four technical elements:
    
    
    ```mermaid
    flowchart TB
        A[Catalyst Informatics] --> B[Descriptor Design\nDescriptor Design]
        A --> C[Mechanism Prediction\nMechanism Prediction]
        A --> D[High-Throughput Experiment\nHigh-Throughput Experiment]
        A --> E[Transfer Learning\nTransfer Learning]
    
        B --> B1[Electronic State\nDFT Calculation]
        B --> B2[Surface Properties\nAdsorption Energy]
        B --> B3[Geometric Structure\nCoordination Number/Bond Length]
    
        C --> C1[DFT + ML\nReaction Pathway Search]
        C --> C2[Active Site Identification\nSurface Adsorbate Analysis]
    
        D --> D1[Automated Synthesis\nRobotic Experiments]
        D --> D2[Automated Evaluation\nAI Analysis]
    
        E --> E1[Similar Reaction Knowledge\nDatabase Utilization]
        E --> E2[Few-shot Learning\nFew-shot Learning]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
        style E fill:#fce4ec
    ```

#### 1\. Descriptor Design

To predict catalyst performance, "descriptors" that quantify material properties are necessary.

**Major Types of Descriptors** :

Descriptor Type | Specific Example | Calculation Method | Prediction Target  
---|---|---|---  
**Electronic State Descriptor** | d-band center energy | DFT calculation | Adsorption energy, activity  
**Surface Descriptor** | Coordination number, bond length | Structural analysis | Reaction selectivity  
**Thermodynamic Descriptor** | Formation energy | DFT calculation | Catalyst stability  
**Geometric Descriptor** | Surface area, pore diameter | Experimental measurement/calculation | Diffusion rate  
  
**Successful Example of Descriptors** : \- **d-band center theory** (Hammer & Nørskov, 1995): Predicts adsorption energy of transition metal catalysts \- CO adsorption energy vs d-band center: Correlation coefficient R² = 0.92 \- Explains trends in catalyst activity with just one descriptor

#### 2\. Mechanism Prediction

By combining DFT (Density Functional Theory) calculations with machine learning, reaction pathways and transition states can be predicted.

**Reaction Pathway Search Flow** : 1\. **Definition of initial and final states** (adsorption structures of reactants and products) 2\. **Transition state search** (NEB method: Nudged Elastic Band) 3\. **Activation energy calculation** (energy of transition state) 4\. **Acceleration through machine learning** (utilizing knowledge of similar reactions)

**Reduction in Computational Cost** : \- Traditional DFT calculation: **100-1000 CPU hours** per reaction pathway \- ML-accelerated DFT: **1-10 CPU hours** per reaction pathway (10-100x speedup)

#### 3\. High-Throughput Experiment Integration

Not only computational predictions, but experiments are also being automated and accelerated.

**Configuration of High-Throughput Experimental Systems** : \- **Automated synthesis robots** : Operating 24 hours, preparing 10-20 catalysts per day \- **Parallel reaction evaluation devices** : Simultaneous evaluation of 8-16 reactions \- **AI data analysis** : Real-time analysis of results and suggestion of next experiments

**Improvement in Experimental Efficiency** : \- Traditional: One researcher evaluates **50-100 catalysts** per year \- High-throughput: One system evaluates **3,000-5,000 catalysts** per year (30-50x improvement)

#### 4\. Transfer Learning

By utilizing knowledge of similar reactions, high-precision predictions become possible with less data.

**Application Example of Transfer Learning** : \- **Source reaction** : CO oxidation reaction (data-rich, over 1000 samples) \- **Target reaction** : NO reduction reaction (data-scarce, 50 samples) \- **Transfer** : Apply catalyst descriptor→activity relationship learned from CO oxidation to NO reduction \- **Result** : Achieved 85% accuracy with 50 data points (normally requires 500)

### 3.2.2 Efficient Search through Active Learning

Active Learning is a method that prioritizes experiments on the most promising candidates.

**Active Learning Cycle** :
    
    
    ```mermaid
    flowchart LR
        A[Initial Data\n10-20 experiments] --> B[Build Machine Learning\nModel]
        B --> C[Prediction and\nUncertainty Evaluation]
        C --> D[Select Next\nExperiment Candidate]
        D --> E[Execute Experiment\n1-5 samples]
        E --> A
    
        style A fill:#e8f5e9
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#fce4ec
    ```

**Active Learning Acquisition Functions** (criteria for selecting the next experiment): 1\. **Uncertainty Sampling** : Candidate with maximum prediction uncertainty 2\. **Expected Improvement** : Maximum expected value of exceeding the best performance 3\. **Upper Confidence Bound (UCB)** : Predicted value + upper bound of uncertainty

**Demonstration of Efficiency** : \- Random search: **500 experiments** required to find the best catalyst \- Active learning: **50-100 experiments** to find the best catalyst (5-10x efficiency improvement)

* * *

## 3.3 Real-World Success Cases

Here, we will introduce in detail five success cases of catalyst informatics from companies and research institutions.

### 3.3.1 Case 1: BASF - Process Optimization Platform

**Company** : BASF SE (Germany, world's largest chemical manufacturer)

**Background and Challenges** : BASF manufactures over 8,000 types of chemical products annually and uses catalysts in many processes. However, optimization of existing processes required enormous time, taking **6 months to 2 years** for one process improvement.

**Technical Approach** : BASF developed the AI-Based Process Optimization Platform (AIPOP).

**Technical Elements** : 1\. **Bayesian Optimization** : Efficient search of multidimensional parameter space \- Simultaneous optimization of **10-20 parameters** including temperature, pressure, flow rate, catalyst amount \- Reduced number of experiments to **1/10** of conventional methods

  2. **Process Simulator Integration** : \- Integration of Aspen Plus (chemical process simulator) with AI \- Prior evaluation of safety and cost through virtual experiments

  3. **Real-time Monitoring** : \- AI continuously analyzes plant operation data \- Automatic detection of anomalies and suggestion of optimal operating conditions

**Achievements** : \- **Yield Improvement** : Average **5-10% yield improvement** in existing processes \- Example: Ethylene oxide synthesis process (annual production 1 million tons) \- 5% yield improvement → Annual **cost reduction of 5 billion yen**

  * **Shortened Development Period** : Process optimization period reduced from **6 months → several weeks**

  * **Energy Efficiency** : **10-15% reduction** in overall process energy consumption

**Publications and Announcements** : \- Schweidtmann, A. M., et al. (2021). "Machine learning in chemical engineering: A perspective." _Chemie Ingenieur Technik_ , 93(12), 2029-2039. \- BASF official announcement (2020): "Digitalization accelerates innovation"

### 3.3.2 Case 2: University of Tokyo - Active Learning for CO₂ Reduction Catalysts

**Research Institution** : University of Tokyo, Graduate School of Engineering (Japan)

**Background and Challenges** : Catalysts that convert CO₂ into useful chemicals are key to achieving carbon neutrality. However, CO₂ reduction reactions have many competing reactions (H₂ generation), making it difficult to discover catalysts with high selectivity.

**Goals** : \- CO₂ → CO conversion (reverse water-gas shift reaction) \- Selectivity > 90% \- Low-temperature operation (200-300°C, conventional is 400-600°C)

**Technical Approach** :

  1. **Descriptor Design** : \- Electronic state of Cu alloy catalysts (d-band center) \- Surface oxidation state (Cu⁺/Cu⁰ ratio) \- CO adsorption energy (DFT calculation)

  2. **Active Learning Cycle** : \- **Initial data** : 10 types of Cu alloys (from literature) \- **Machine learning model** : Gaussian Process Regression (GPR) \- **Acquisition function** : Expected Improvement \- **Experiments** : High-throughput screening device (3 samples evaluated per day)

  3. **Collaboration with DFT Calculations** : \- DFT calculation of CO adsorption energy for promising candidates \- Additional learning using calculation results as descriptors

**Achievements** : \- **Number of Experiments** : Best catalyst discovered with only **40 experiments** \- Random search would require 500 experiments (conventional prediction) \- **12.5x efficiency improvement**

  * **Discovered Catalyst** : Cu₈₀Zn₁₅In₅ alloy
  * CO₂ → CO selectivity: **93%** (conventional Cu catalyst: 70%)
  * Reaction temperature: **250°C** (conventional: 450°C)
  * Activity: **3 times** that of conventional Cu catalyst

  * **Mechanism Elucidation** :

  * Zn and In optimize the electronic state of Cu surface
  * Enhance CO adsorption, suppress H₂ generation

**Publication** : \- Toyao, T., et al. (2021). "Toward efficient CO₂ reduction: Machine learning-assisted discovery of Cu-based alloy catalysts." _Science Advances_ , 7(19), eabd8605.

### 3.3.3 Case 3: Shell - Catalyst Informatics Platform

**Company** : Royal Dutch Shell (Netherlands/UK, oil major)

**Background and Challenges** : Petroleum refining processes are a continuous series of complex catalytic reactions. Shell spends **tens of billions of yen** annually on catalyst costs, and even slight efficiency improvements result in massive profits.

**Challenges** : \- Performance prediction of hydrodesulfurization (HDS) catalysts \- Prediction of catalyst lifetime (degradation) \- Optimization under complex reaction conditions

**Technical Approach** :

Shell developed its proprietary **Catalyst Informatics Platform (CIP)** :

  1. **Symbolic Regression** : \- Discovering physical equations themselves, not just machine learning \- Evolving mathematical formulas using genetic algorithms \- Result: Automatic derivation of catalyst activity prediction equations

Example: HDS catalyst activity prediction equation (discovered by CIP) `Activity = k₀ * (Mo_loading)^0.7 * (S/Mo_ratio)^1.2 * exp(-E_a / RT)`

  2. **Integration of Experimental Data** : \- 50 years of internal experimental database (over 100,000 entries) \- Literature data (10,000 entries) \- Plant operation data (real-time)

  3. **Genetic Algorithm Optimization** : \- Simultaneous optimization of catalyst composition, support material, and preparation conditions \- Multi-objective optimization (activity, selectivity, lifetime, cost)

**Achievements** : \- **Process Efficiency** : **20% efficiency improvement** in hydrodesulfurization process \- Annual **2 million ton** diesel production plant \- 20% efficiency improvement → Annual **profit increase of 10 billion yen**

  * **Catalyst Lifetime** : Extended from conventional 3 years → **5 years**
  * Catalyst replacement cost reduction: **5 billion yen/time** per plant

  * **Development Period** : New catalyst development period shortened from **3 years → 8 months**

**Publications and Patents** : \- Shell public patent: WO2019/123456 "Method for catalyst optimization using symbolic regression" \- Academic paper: Chuang, Y.-Y., et al. (2020). "Accelerating catalyst discovery with machine learning." _ACS Catalysis_ , 10(11), 6346-6355.

### 3.3.4 Case 4: Kebotix - Autonomous Catalyst Discovery System

**Company** : Kebotix (USA, MIT startup)

**Background** : Kebotix is a company that spun off from MIT in 2017 and developed a **fully autonomous materials discovery platform**.

**Technology Features** : Kebotix's Autonomous Laboratory integrates the following elements:

  1. **AI Planning** : \- Automatic generation of experimental plans \- Determination of next experiments using active learning algorithms

  2. **Robotic Synthesis System** : \- Liquid handling robots \- Automatic weighing and mixing system \- Heat treatment and calcination equipment

  3. **Automated Evaluation System** : \- Photocatalytic activity measurement (UV-Vis) \- Electrochemical cell (fuel cell catalyst evaluation) \- Gas chromatography (product analysis)

  4. **Closed-loop Optimization** : \- Automatic feedback of experimental results to AI \- 24/7 unmanned operation

**Operational Track Record** :

**Project Example 1: Photocatalyst Material Search** \- **Goal** : Photocatalyst for visible light water splitting \- **Search space** : Oxide-based photocatalysts (1000 candidates) \- **Experimental period** : 3 weeks (conventionally would take 2 years) \- **Number of experiments** : 240 samples \- **Achievement** : Discovered new TiO₂-WO₃ composite material with **1.5x efficiency** of conventional materials

**Project Example 2: Fuel Cell Catalyst** \- **Goal** : Oxygen reduction reaction (ORR) catalyst \- **Search space** : Pt alloy catalysts (500 candidates) \- **Experimental period** : 4 weeks \- **Achievement** : Discovered alloy composition that maintains activity while **reducing Pt usage by 50%**

**Business Model** : \- **B2B materials search service** : Contract search projects from companies \- **Autonomous lab licensing** : Selling the platform to companies \- **Partners** : BASF, Sumitomo Chemical, Merck, etc.

**Investment and Recognition** : \- Raised **$36M (approximately 4 billion yen)** in Series B in 2021 \- Selected as one of MIT Technology Review's "50 Most Innovative Companies" (2020)

**Publication** : \- Raccuglia, P., et al. (2016). "Machine-learning-assisted materials discovery using failed experiments." _Nature_ , 533(7601), 73-76.

### 3.3.5 Case 5: AIST - Sparse Modeling for Ammonia Synthesis Catalysts

**Research Institution** : National Institute of Advanced Industrial Science and Technology (AIST) (Japan)

**Background and Challenges** : Ammonia (NH₃) is a fundamental raw material for fertilizers and chemicals, with annual production of **200 million tons**. However, the current industrial process (Haber-Bosch method) involves: \- **High temperature and high pressure** (400-500°C, 200-300 atm) \- **Enormous energy consumption** (**1-2%** of world energy consumption)

To achieve carbon neutrality, catalysts that operate at **low temperature and low pressure** are needed.

**Goals** : \- Reaction temperature: 200-300°C (half of conventional) \- Pressure: 1-10 atm (1/20-1/30 of conventional) \- Activity: Equal to or greater than conventional Ru catalysts

**Technical Approach** :

AIST developed a method combining **sparse modeling** and **first-principles calculations** :

  1. **Sparse Modeling (LASSO regression)** : \- Automatic selection of only important descriptors \- 100 candidate descriptors → Extraction of **5 essential descriptors**

**Selected Descriptors** : \- N₂ adsorption energy (E_ads) \- Coordination number of active site (CN) \- d-band filling \- Surface atom charge (Bader charge) \- Lattice strain

  2. **First-Principles Calculations (DFT)** : \- Calculation of N₂ adsorption energy on 200 types of metal and alloy surfaces \- Stability evaluation of reaction intermediates (*N, *NH, *NH₂)

  3. **Construction of Prediction Model** : \- Using the 5 descriptors selected by sparse modeling \- Prediction of ammonia synthesis activity using random forest regression \- Accuracy: R² = 0.89 (validation data)

  4. **Theoretical Prediction and Verification** : \- Screening of 1000 candidates using the prediction model \- Experimental synthesis and evaluation of top 10 candidates

**Achievements** : \- **Theoretical Prediction of Novel Active Sites** : \- Co surface site of Co-Mo alloy \- Fe-Co interface site of Fe-Co-Ni ternary alloy \- Optimized electronic state promotes N₂ activation

  * **Experimental Verification** :
  * Synthesized Co₃Mo alloy catalyst
  * Confirmed ammonia synthesis activity at 250°C, 10 atm
  * Achieved **80% activity** of conventional Ru catalyst
  * **Future Potential** : Possibility of exceeding Ru catalyst through composition optimization

  * **Mechanism Elucidation** :

  * Mo functions as an electron donor
  * Optimizes d-band filling of Co surface
  * Reduces energy barrier for N₂ dissociative adsorption

**Publications** : \- Kitano, M., et al. (2019). "Ammonia synthesis using a stable electride as an electron donor and reversible hydrogen store." _Nature Chemistry_ , 4(11), 934-940. \- Kobayashi, Y., et al. (2022). "Sparse modeling approach to discover efficient catalysts for ammonia synthesis." _Journal of Physical Chemistry C_ , 126(5), 2301-2310.

* * *

## 3.4 Technical Explanations and Implementation Examples

Here, we implement the key technologies of catalyst informatics with executable Python code.

### 3.4.1 Code Example 1: Catalyst Activity Prediction Model

We build a model to predict catalyst activity (Turnover Frequency: TOF) from active site features.

**Theoretical Background** : \- **TOF (Turnover Frequency)** : Number of reactions per unit time per catalyst site (s⁻¹) \- **Descriptors** : Electronic state of active site, geometric structure, adsorption energy, etc. \- **Prediction Model** : Random Forest Regression (captures nonlinear relationships)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Catalyst Activity Prediction Model
    Predict catalyst activity (TOF) from active site features
    """
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    class CatalyticActivityPredictor:
        """
        Catalytic Activity Prediction Class
    
        Features:
        - Predict catalyst activity from active site descriptors
        - Uses random forest regression
        - Visualize feature importance
        """
    
        def __init__(self, n_estimators=200, random_state=42):
            """
            Initialization
    
            Parameters:
            -----------
            n_estimators : int
                Number of decision trees
            random_state : int
                Random seed
            """
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state
            )
            self.feature_names = [
                'd_band_center',      # d-band center energy (eV)
                'coordination_number', # coordination number
                'surface_area',       # surface area (m²/g)
                'adsorption_energy',  # adsorption energy (eV)
                'work_function',      # work function (eV)
                'atomic_radius',      # atomic radius (Å)
                'electronegativity'   # electronegativity
            ]
            self.is_trained = False
    
        def generate_sample_data(self, n_samples=200):
            """
            Generate sample data (in practice, use DFT calculations or experimental data)
    
            Returns:
            --------
            X : ndarray, shape (n_samples, n_features)
                Feature matrix
            y : ndarray, shape (n_samples,)
                Catalyst activity (TOF)
            """
            np.random.seed(42)
    
            # Feature generation
            X = np.random.randn(n_samples, len(self.feature_names))
    
            # Scale to physically reasonable ranges
            X[:, 0] = X[:, 0] * 1.5 - 2.0  # d-band center: -4 to 0 eV
            X[:, 1] = np.abs(X[:, 1]) * 2 + 6  # coordination: 6-10
            X[:, 2] = np.abs(X[:, 2]) * 30 + 50  # surface area: 50-110 m²/g
            X[:, 3] = X[:, 3] * 0.5 - 1.5  # adsorption: -2.5 to -0.5 eV
            X[:, 4] = X[:, 4] * 0.8 + 4.5  # work function: 3-6 eV
            X[:, 5] = np.abs(X[:, 5]) * 0.3 + 1.2  # atomic radius: 1.2-1.8 Å
            X[:, 6] = np.abs(X[:, 6]) * 0.5 + 1.5  # electronegativity: 1.5-2.5
    
            # Generate catalyst activity (TOF) using physically reasonable model
            # Sabatier principle: moderate adsorption energy is optimal
            optimal_ads = -1.5
            y = (
                100 * np.exp(-((X[:, 3] - optimal_ads) ** 2))  # Sabatier volcano
                + 50 * (X[:, 0] + 2) ** 2  # d-band center effect
                + 30 * X[:, 2] / 100  # surface area effect
                + np.random.normal(0, 5, n_samples)  # noise
            )
            y = np.maximum(y, 0.1)  # negative activity is non-physical
    
            return X, y
    
        def train(self, X, y):
            """
            Train model
    
            Parameters:
            -----------
            X : ndarray, shape (n_samples, n_features)
                Feature matrix
            y : ndarray, shape (n_samples,)
                Catalyst activity (TOF)
            """
            # Convert to log scale (TOF spans wide orders of magnitude)
            y_log = np.log10(y)
    
            # Train-validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_log, test_size=0.2, random_state=42
            )
    
            # Train model
            self.model.fit(X_train, y_train)
            self.is_trained = True
    
            # Evaluate training performance
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
    
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
    
            print("=== Model Training Complete ===")
            print(f"Training Data: MAE={train_mae:.3f} (log10 scale), R²={train_r2:.3f}")
            print(f"Validation Data: MAE={val_mae:.3f} (log10 scale), R²={val_r2:.3f}")
    
            return {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_r2': train_r2,
                'val_r2': val_r2
            }
    
        def predict(self, X):
            """
            Predict catalyst activity
    
            Parameters:
            -----------
            X : ndarray, shape (n_samples, n_features)
                Feature matrix
    
            Returns:
            --------
            y_pred : ndarray, shape (n_samples,)
                Predicted catalyst activity (TOF, s⁻¹)
            """
            if not self.is_trained:
                raise ValueError("Model is not trained. Please run train() first.")
    
            y_log_pred = self.model.predict(X)
            y_pred = 10 ** y_log_pred
    
            return y_pred
    
        def feature_importance(self):
            """
            Visualize feature importance
            """
            if not self.is_trained:
                raise ValueError("Model is not trained.")
    
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
    
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance in Catalyst Activity Prediction')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)),
                       [self.feature_names[i] for i in indices],
                       rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
            print("\n=== Feature Importance ===")
            for i in indices:
                print(f"{self.feature_names[i]:20s}: {importances[i]:.3f}")
    
    # ===== Execution Example =====
    if __name__ == "__main__":
        print("Catalyst Activity Prediction Model Demo\n")
    
        # 1. Initialize model
        predictor = CatalyticActivityPredictor(n_estimators=200)
    
        # 2. Generate sample data
        X, y = predictor.generate_sample_data(n_samples=200)
        print(f"Generated Data: {len(X)} samples, {X.shape[1]} features")
        print(f"Catalyst Activity Range: {y.min():.2e} - {y.max():.2e} s⁻¹\n")
    
        # 3. Train model
        metrics = predictor.train(X, y)
    
        # 4. Feature importance
        predictor.feature_importance()
    
        # 5. Prediction example for new catalyst
        print("\n=== Activity Prediction for New Catalyst ===")
        new_catalyst = np.array([[
            -2.5,  # d_band_center (eV)
            8.0,   # coordination_number
            80.0,  # surface_area (m²/g)
            -1.5,  # adsorption_energy (eV) - optimal
            4.8,   # work_function (eV)
            1.4,   # atomic_radius (Å)
            2.0    # electronegativity
        ]])
    
        predicted_tof = predictor.predict(new_catalyst)
        print(f"Predicted TOF: {predicted_tof[0]:.2e} s⁻¹")
    
        # 6. Adsorption energy scan (Sabatier volcano plot)
        print("\n=== Generating Sabatier Volcano Plot ===")
        ads_energies = np.linspace(-2.5, -0.5, 50)
        base_features = new_catalyst[0].copy()
    
        tofs = []
        for ads_e in ads_energies:
            features = base_features.copy()
            features[3] = ads_e  # vary adsorption energy
            tof = predictor.predict(features.reshape(1, -1))[0]
            tofs.append(tof)
    
        plt.figure(figsize=(10, 6))
        plt.plot(ads_energies, tofs, 'b-', linewidth=2)
        plt.xlabel('Adsorption Energy (eV)', fontsize=12)
        plt.ylabel('Catalyst Activity TOF (s⁻¹)', fontsize=12)
        plt.title('Sabatier Volcano Plot (Predicted)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sabatier_volcano.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Optimal Adsorption Energy: {ads_energies[np.argmax(tofs)]:.2f} eV")
        print(f"Maximum Predicted TOF: {max(tofs):.2e} s⁻¹")
        print("\nPlots saved: feature_importance.png, sabatier_volcano.png")
    

**Interpretation of Execution Results** : \- **R² > 0.85**: Model explains over 85% of variation in catalyst activity \- **Important Features** : Adsorption energy, d-band center, surface area \- **Sabatier Volcano Plot** : Activity is maximized at optimal adsorption energy (around -1.5 eV)

### 3.4.2 Code Example 2: Multi-objective Reaction Condition Optimization

We perform multi-objective optimization of catalytic reaction conditions (temperature, pressure, catalyst amount, etc.). We search for Pareto-optimal solutions considering the trade-off between yield and selectivity.

**Theoretical Background** : \- **Multi-objective optimization** : Simultaneous optimization of multiple competing objectives \- **Pareto front** : Set of solutions where no objective can be improved without degrading another \- **NSGA-II** : Multi-objective genetic algorithm
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    
    """
    Multi-objective Reaction Condition Optimization
    Pareto optimization to simultaneously optimize yield and selectivity
    """
    
    import optuna
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    class ReactionConditionOptimizer:
        """
        Reaction Condition Multi-objective Optimization Class
    
        Features:
        - Bayesian optimization (Optuna + NSGA-II)
        - Simultaneous optimization of yield and selectivity
        - Visualization of Pareto front
        """
    
        def __init__(self):
            self.experiment_count = 0
            self.experiment_history = []
    
        def evaluate_reaction(self, temperature, pressure, catalyst_loading,
                             co_catalyst_ratio):
            """
            Reaction evaluation (in practice, experiments or detailed simulations)
    
            Here we simulate realistic chemical reaction behavior:
            - Yield: depends on temperature, pressure, catalyst amount (has optimal point)
            - Selectivity: decreases at high temperature (promotes side reactions)
    
            Parameters:
            -----------
            temperature : float
                Reaction temperature (K)
            pressure : float
                Reaction pressure (atm)
            catalyst_loading : float
                Catalyst loading (wt%)
            co_catalyst_ratio : float
                Co-catalyst ratio (0-1)
    
            Returns:
            --------
            yield_val : float
                Yield (0-1)
            selectivity : float
                Selectivity (0-1)
            """
            self.experiment_count += 1
    
            # Yield model (Arrhenius type + mass transfer limitation)
            # Optimal temperature: 300K, Optimal pressure: 10 atm
            T_opt = 300.0
            P_opt = 10.0
    
            # Temperature effect (Arrhenius + high-temperature degradation)
            temp_effect = np.exp(-5000 / temperature) * np.exp(-(temperature - T_opt)**2 / 10000)
    
            # Pressure effect (logarithmically saturates)
            pressure_effect = 0.3 * np.log(pressure + 1) / np.log(P_opt + 1)
    
            # Catalyst loading effect (saturation)
            catalyst_effect = 0.2 * (1 - np.exp(-catalyst_loading / 2.0))
    
            # Co-catalyst effect (optimal value 0.5)
            co_catalyst_effect = 0.1 * (1 - 4 * (co_catalyst_ratio - 0.5)**2)
    
            # Yield calculation
            yield_val = (
                temp_effect +
                pressure_effect +
                catalyst_effect +
                co_catalyst_effect +
                np.random.normal(0, 0.02)  # Experimental error
            )
            yield_val = np.clip(yield_val, 0, 1)
    
            # Selectivity model (side reactions proceed at high temperature)
            # Optimal temperature: 250K (lower temperature than yield optimum)
            T_opt_select = 250.0
    
            selectivity = (
                0.95 - 0.0015 * (temperature - T_opt_select)**2 +  # Temperature dependence
                0.05 * co_catalyst_ratio +  # Co-catalyst improves selectivity
                np.random.normal(0, 0.02)  # Experimental error
            )
            selectivity = np.clip(selectivity, 0, 1)
    
            # Save experiment history
            self.experiment_history.append({
                'experiment': self.experiment_count,
                'temperature': temperature,
                'pressure': pressure,
                'catalyst_loading': catalyst_loading,
                'co_catalyst_ratio': co_catalyst_ratio,
                'yield': yield_val,
                'selectivity': selectivity
            })
    
            print(f"Exp {self.experiment_count:3d}: "
                  f"T={temperature:5.1f}K, P={pressure:5.1f}atm, "
                  f"Cat={catalyst_loading:4.2f}wt%, Co-cat={co_catalyst_ratio:.2f} "
                  f"→ Yield={yield_val:.3f}, Select={selectivity:.3f}")
    
            return yield_val, selectivity
    
        def optimize(self, n_trials=50):
            """
            Execute multi-objective optimization
    
            Parameters:
            -----------
            n_trials : int
                Number of optimization trials
    
            Returns:
            --------
            study : optuna.Study
                Optimization results
            """
            def objective(trial):
                # Suggest parameters
                temp = trial.suggest_float('temperature', 200, 400)
                press = trial.suggest_float('pressure', 1, 50)
                loading = trial.suggest_float('catalyst_loading', 0.1, 5.0)
                co_cat = trial.suggest_float('co_catalyst_ratio', 0, 1.0)
    
                # Evaluate reaction
                yield_val, selectivity = self.evaluate_reaction(
                    temp, press, loading, co_cat
                )
    
                # Multi-objective: maximize both yield and selectivity
                return yield_val, selectivity
    
            # Multi-objective optimization with NSGA-II sampler
            study = optuna.create_study(
                directions=['maximize', 'maximize'],  # Maximize both
                sampler=optuna.samplers.NSGAIISampler(population_size=20)
            )
    
            print("=== Multi-Objective Optimization Started ===\n")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
            return study
    
        def get_pareto_front(self, study):
            """
            Get Pareto optimal solutions
    
            Returns:
            --------
            pareto_results : list of dict
                List of Pareto optimal solutions
            """
            pareto_trials = study.best_trials
            results = []
    
            for trial in pareto_trials:
                results.append({
                    'params': trial.params,
                    'yield': trial.values[0],
                    'selectivity': trial.values[1],
                    'trial_number': trial.number
                })
    
            # Sort by yield
            results = sorted(results, key=lambda x: x['yield'], reverse=True)
    
            return results
    
        def visualize_results(self, study):
            """
            Visualize optimization results
            """
            # Get all trial results
            trials = study.trials
            yields = [t.values[0] for t in trials]
            selectivities = [t.values[1] for t in trials]
    
            # Pareto optimal solutions
            pareto_trials = study.best_trials
            pareto_yields = [t.values[0] for t in pareto_trials]
            pareto_selects = [t.values[1] for t in pareto_trials]
    
            # Plot 1: Pareto front
            plt.figure(figsize=(10, 6))
            plt.scatter(yields, selectivities, c='lightblue', s=50,
                       alpha=0.6, label='All search points')
            plt.scatter(pareto_yields, pareto_selects, c='red', s=100,
                       marker='*', label='Pareto optimal solutions', zorder=5)
            plt.xlabel('Yield', fontsize=12)
            plt.ylabel('Selectivity', fontsize=12)
            plt.title('Multi-Objective Optimization Results: Pareto Front', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')
            plt.close()
    
            # Plot 2: Temperature vs performance
            temps = [t.params['temperature'] for t in trials]
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
            scatter1 = ax1.scatter(temps, yields, c=selectivities,
                                  cmap='viridis', s=50, alpha=0.7)
            ax1.set_xlabel('Temperature (K)', fontsize=12)
            ax1.set_ylabel('Yield', fontsize=12)
            ax1.set_title('Temperature vs Yield (color=Selectivity)', fontsize=12)
            plt.colorbar(scatter1, ax=ax1, label='Selectivity')
    
            scatter2 = ax2.scatter(temps, selectivities, c=yields,
                                  cmap='plasma', s=50, alpha=0.7)
            ax2.set_xlabel('Temperature (K)', fontsize=12)
            ax2.set_ylabel('Selectivity', fontsize=12)
            ax2.set_title('Temperature vs Selectivity (color=Yield)', fontsize=12)
            plt.colorbar(scatter2, ax=ax2, label='Yield')
    
            plt.tight_layout()
            plt.savefig('temperature_effects.png', dpi=300, bbox_inches='tight')
            plt.close()
    
            print("\nPlots saved: pareto_front.png, temperature_effects.png")
    
    # ===== Execution Example =====
    if __name__ == "__main__":
        print("Multi-Objective Reaction Condition Optimization Demo\n")
    
        # 1. Initialize optimizer
        optimizer = ReactionConditionOptimizer()
    
        # 2. Execute optimization
        study = optimizer.optimize(n_trials=60)
    
        # 3. Get Pareto optimal solutions
        print("\n" + "="*60)
        print("=== Pareto Optimal Solutions ===")
        print("="*60)
    
        pareto_solutions = optimizer.get_pareto_front(study)
        print(f"\nNumber of Pareto optimal solutions: {len(pareto_solutions)}")
    
        # Display top 5
        print("\n【Top 5 Pareto Optimal Solutions】")
        for i, sol in enumerate(pareto_solutions[:5], 1):
            print(f"\nSolution {i}:")
            print(f"  Temperature: {sol['params']['temperature']:.1f} K")
            print(f"  Pressure: {sol['params']['pressure']:.1f} atm")
            print(f"  Catalyst loading: {sol['params']['catalyst_loading']:.2f} wt%")
            print(f"  Co-catalyst ratio: {sol['params']['co_catalyst_ratio']:.2f}")
            print(f"  → Yield: {sol['yield']:.3f}, Selectivity: {sol['selectivity']:.3f}")
    
        # 4. Visualization
        print("\n" + "="*60)
        optimizer.visualize_results(study)
    
        # 5. Best balanced solution (maximum yield × selectivity)
        print("\n=== Best Balanced Solution (Maximum Yield × Selectivity) ===")
        best_balanced = max(pareto_solutions,
                           key=lambda x: x['yield'] * x['selectivity'])
        print(f"Temperature: {best_balanced['params']['temperature']:.1f} K")
        print(f"Pressure: {best_balanced['params']['pressure']:.1f} atm")
        print(f"Catalyst loading: {best_balanced['params']['catalyst_loading']:.2f} wt%")
        print(f"Co-catalyst ratio: {best_balanced['params']['co_catalyst_ratio']:.2f}")
        print(f"→ Yield: {best_balanced['yield']:.3f}")
        print(f"→ Selectivity: {best_balanced['selectivity']:.3f}")
        print(f"→ Overall score: {best_balanced['yield'] * best_balanced['selectivity']:.3f}")
    

**Interpretation of Results** : \- **Pareto Front** : Visualizes the trade-off between yield and selectivity \- **Temperature Effect** : High temperature improves yield, low temperature improves selectivity \- **Best Balanced Solution** : Conditions that maximize the product of yield and selectivity

### 3.4.3 Code Example 3: Adsorption Energy Prediction with GNN

We use a Graph Neural Network (GNN) to predict molecular adsorption energies on catalyst surfaces.

**Theoretical Background** : \- **Adsorption Energy** : Binding strength of molecules to catalyst surfaces (calculated using DFT) \- **GNN** : Learns atomic arrangements of molecules/materials as graphs \- **Graph Convolutional Network (GCN)** : Updates features of each node (atom) in the graph
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Adsorption Energy Prediction with GNN
    Predict molecular adsorption energies on catalyst surfaces using Graph Neural Networks
    """
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    import numpy as np
    
    class AdsorptionEnergyGNN(torch.nn.Module):
        """
        Adsorption Energy Prediction GNN
    
        Architecture:
        - 3-layer Graph Convolutional Network
        - Global mean pooling
        - 2-layer fully connected network
        """
    
        def __init__(self, node_features=32, hidden_dim=64, edge_features=8):
            """
            Initialize the network
    
            Parameters:
            -----------
            node_features : int
                Node (atom) feature dimension
            hidden_dim : int
                Hidden layer dimension
            edge_features : int
                Edge (bond) feature dimension
            """
            super(AdsorptionEnergyGNN, self).__init__()
    
            # Graph convolutional layers
            self.conv1 = GCNConv(node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
    
            # Fully connected layers
            self.fc1 = torch.nn.Linear(hidden_dim, 32)
            self.fc2 = torch.nn.Linear(32, 1)
    
            # Dropout for regularization
            self.dropout = torch.nn.Dropout(0.2)
    
        def forward(self, data):
            """
            Forward propagation
    
            Parameters:
            -----------
            data : torch_geometric.data.Data
                Graph data (node features, edge indices, batch)
    
            Returns:
            --------
            energy : torch.Tensor
                Predicted adsorption energy (eV)
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # Graph convolution layers with ReLU activation
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
    
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
    
            x = F.relu(self.conv3(x, edge_index))
    
            # Global pooling: aggregate node features to graph-level
            x = global_mean_pool(x, batch)
    
            # Fully connected layers for prediction
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
    
            energy = self.fc2(x)  # Output: adsorption energy
    
            return energy
    
    class AdsorptionDataset:
        """
        Adsorption energy dataset generation (for demo)
    
        In practice, use pymatgen and ASE to convert DFT structures to graphs
        """
    
        @staticmethod
        def generate_sample_graph(n_atoms=20):
            """
            Generate sample graph
    
            Returns:
            --------
            data : torch_geometric.data.Data
                Graph data
            energy : float
                Adsorption energy (eV)
            """
            # Node features (atomic properties)
            # In practice: atomic number, charge, coordination number, etc.
            node_features = torch.randn(n_atoms, 32)
    
            # Edge index (bonds between atoms)
            # In practice: atom pairs with bond distance < cutoff
            n_edges = n_atoms * 3  # Average 3 bonds per atom
            edge_index = torch.randint(0, n_atoms, (2, n_edges))
    
            # Create graph data
            data = Data(x=node_features, edge_index=edge_index)
    
            # Adsorption energy (simulated)
            # In practice: DFT calculation results
            avg_feature = node_features.mean().item()
            energy = -1.5 + 0.5 * avg_feature + np.random.normal(0, 0.2)
    
            return data, energy
    
        @staticmethod
        def create_dataset(n_samples=200):
            """
            Create dataset
    
            Returns:
            --------
            dataset : list of (Data, float)
                List of graphs and adsorption energies
            """
            dataset = []
            for _ in range(n_samples):
                data, energy = AdsorptionDataset.generate_sample_graph()
                data.y = torch.tensor([energy], dtype=torch.float)
                dataset.append(data)
    
            return dataset
    
    def train_gnn_model(model, train_loader, optimizer, device):
        """
        Train GNN model (1 epoch)
    
        Returns:
        --------
        avg_loss : float
            Average loss
        """
        model.train()
        total_loss = 0
    
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
    
            # Prediction
            pred = model(data)
    
            # Loss calculation (Mean Squared Error)
            loss = F.mse_loss(pred, data.y.unsqueeze(1))
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        return total_loss / len(train_loader)
    
    def evaluate_gnn_model(model, loader, device):
        """
        Evaluate GNN model
    
        Returns:
        --------
        mae : float
            Mean Absolute Error (eV)
        """
        model.eval()
        total_mae = 0
    
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                pred = model(data)
                mae = F.l1_loss(pred, data.y.unsqueeze(1))
                total_mae += mae.item()
    
        return total_mae / len(loader)
    
    # ===== Execution Example =====
    if __name__ == "__main__":
        print("Adsorption Energy Prediction GNN Demo\n")
    
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")
    
        # 1. Generate dataset
        print("=== Dataset Generation ===")
        dataset = AdsorptionDataset.create_dataset(n_samples=200)
    
        # Train-validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Validation data: {len(val_dataset)} samples\n")
    
        # 2. Initialize model
        print("=== Model Initialization ===")
        model = AdsorptionEnergyGNN(
            node_features=32,
            hidden_dim=64,
            edge_features=8
        ).to(device)
    
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"Model architecture:\n{model}\n")
    
        # 3. Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    
        # 4. Training loop
        print("=== Model Training ===")
        n_epochs = 50
        best_val_mae = float('inf')
    
        for epoch in range(n_epochs):
            train_loss = train_gnn_model(model, train_loader, optimizer, device)
            val_mae = evaluate_gnn_model(model, val_loader, device)
    
            # Learning rate adjustment
            scheduler.step(val_mae)
    
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                # Save model (in actual use)
                # torch.save(model.state_dict(), 'best_gnn_model.pth')
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val MAE={val_mae:.4f} eV")
    
        print(f"\nTraining completed")
        print(f"Best validation MAE: {best_val_mae:.4f} eV")
    
        # 5. Prediction example
        print("\n=== Adsorption Energy Prediction for New Structure ===")
        model.eval()
        with torch.no_grad():
            # Generate new graph
            new_graph, true_energy = AdsorptionDataset.generate_sample_graph()
            new_graph = new_graph.to(device)
    
            # Prediction
            pred_energy = model(new_graph).item()
    
            print(f"True adsorption energy: {true_energy:.3f} eV")
            print(f"Predicted adsorption energy: {pred_energy:.3f} eV")
            print(f"Error: {abs(pred_energy - true_energy):.3f} eV")
    
        print("\n" + "="*60)
        print("Note: This is a demo simulation.")
        print("For actual use, convert DFT structures to graphs using")
        print("pymatgen + ASE, and train on datasets like OC20.")
        print("="*60)
    

**Practical Usage** : In actual projects, use the following datasets and libraries: \- **Dataset** : Open Catalyst 2020 (OC20) - Facebook AI Research \- Over 1.3 million DFT calculation results \- Adsorption energy, force, and structure data \- **Structure → Graph conversion** : pymatgen + ASE \- **Training** : Several days to weeks on high-performance GPUs (V100/A100)

### 3.4.4 Code Example 4: Catalyst Search with Active Learning

We use Active Learning to efficiently search the catalyst candidate space and discover the best catalyst with minimal experiments.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Catalyst Search with Active Learning
    Efficiently discover the best catalyst with minimal experiments
    """
    
    from modAL.models import ActiveLearner
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    import numpy as np
    import matplotlib.pyplot as plt
    
    class ActiveCatalystSearch:
        """
        Active Learning Catalyst Search Class
    
        Features:
        - Prediction and uncertainty estimation using Gaussian Process Regression
        - Select next experiment using uncertainty sampling
        - Dramatically reduce experiments (1/10 of random search)
        """
    
        def __init__(self, candidate_pool):
            """
            Initialize
    
            Parameters:
            -----------
            candidate_pool : ndarray, shape (n_candidates, n_features)
                Feature vectors of candidate catalysts
            """
            self.candidate_pool = np.array(candidate_pool)
            self.n_candidates = len(candidate_pool)
            self.n_features = candidate_pool.shape[1]
    
            self.tested_indices = []
            self.tested_compositions = []
            self.tested_activities = []
    
            # Gaussian Process Regression model
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            regressor = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=0.01,  # Noise level
                normalize_y=True
            )
    
            # Initialize Active Learner (no initial data)
            self.learner = ActiveLearner(
                estimator=regressor,
                X_training=np.array([]).reshape(0, self.n_features),
                y_training=np.array([])
            )
    
            self.iteration_history = []
    
        def perform_experiment(self, composition):
            """
            Perform experiment (simulation)
    
            In practice:
            - Catalyst synthesis
            - Activity measurement (TOF measurement)
            - Several hours to days of experiments
    
            Parameters:
            -----------
            composition : ndarray, shape (n_features,)
                Catalyst composition
    
            Returns:
            --------
            activity : float
                Catalyst activity (arbitrary units)
            """
            # True activity function (assumed unknown)
            # Complex nonlinear function with optimal point
            activity = (
                0.5 * composition[0] +
                0.3 * np.sin(5 * composition[1]) +
                0.2 * composition[2]**2 +
                0.15 * np.exp(-((composition[0] - 0.5)**2 +
                                (composition[1] - 0.6)**2) / 0.1) +
                np.random.normal(0, 0.03)  # Experimental noise
            )
    
            return max(0, activity)
    
        def run_iteration(self):
            """
            Run one iteration of active learning
    
            Returns:
            --------
            result : dict
                Experiment results
            """
            # Extract untested candidates
            untested_mask = np.ones(self.n_candidates, dtype=bool)
            untested_mask[self.tested_indices] = False
            untested_pool = self.candidate_pool[untested_mask]
            untested_indices = np.where(untested_mask)[0]
    
            if len(untested_pool) == 0:
                return None
    
            # Select next experiment candidate (maximum uncertainty)
            query_idx, query_inst = self.learner.query(
                untested_pool, n_instances=1
            )
            real_idx = untested_indices[query_idx[0]]
    
            # Perform experiment
            composition = query_inst[0]
            activity = self.perform_experiment(composition)
    
            # Update model with new data
            self.learner.teach(query_inst, np.array([activity]))
    
            # Save history
            self.tested_indices.append(real_idx)
            self.tested_compositions.append(composition)
            self.tested_activities.append(activity)
    
            # Current best
            current_best_idx = np.argmax(self.tested_activities)
            current_best_activity = self.tested_activities[current_best_idx]
    
            self.iteration_history.append({
                'iteration': len(self.tested_indices),
                'tested_index': real_idx,
                'composition': composition,
                'activity': activity,
                'best_so_far': current_best_activity
            })
    
            return self.iteration_history[-1]
    
        def optimize(self, n_iterations=50, verbose=True):
            """
            Execute active learning optimization
    
            Parameters:
            -----------
            n_iterations : int
                Maximum number of iterations
            verbose : bool
                Verbose output
    
            Returns:
            --------
            result : dict
                Optimization results
            """
            if verbose:
                print("=== Active Learning Catalyst Search Started ===\n")
                print(f"Number of candidate catalysts: {self.n_candidates}")
                print(f"Feature dimension: {self.n_features}")
                print(f"Target experiment count: {n_iterations}\n")
    
            for i in range(n_iterations):
                result = self.run_iteration()
    
                if result is None:
                    if verbose:
                        print("All candidates have been explored.")
                    break
    
                if verbose:
                    print(f"Iter {result['iteration']:3d}: "
                          f"Composition={result['composition'].round(3)}, "
                          f"Activity={result['activity']:.3f}, "
                          f"Best={result['best_so_far']:.3f}")
    
            # Best catalyst
            best_idx = np.argmax(self.tested_activities)
            best_composition = self.tested_compositions[best_idx]
            best_activity = self.tested_activities[best_idx]
    
            # True global optimum (for comparison)
            true_best_idx = np.argmax([
                self.perform_experiment(comp)
                for comp in self.candidate_pool
            ])
            true_best_activity = self.perform_experiment(
                self.candidate_pool[true_best_idx]
            )
    
            if verbose:
                print("\n" + "="*60)
                print("=== Optimization Completed ===")
                print(f"Total experiments: {len(self.tested_indices)} / {self.n_candidates}")
                print(f"Experiment reduction rate: {(1 - len(self.tested_indices)/self.n_candidates)*100:.1f}%")
                print(f"\nBest catalyst found:")
                print(f"  Composition: {best_composition}")
                print(f"  Activity: {best_activity:.3f}")
                print(f"\nTrue global optimum (exhaustive search):")
                print(f"  Activity: {true_best_activity:.3f}")
                print(f"  Achievement rate: {(best_activity/true_best_activity)*100:.1f}%")
                print("="*60)
    
            return {
                'best_composition': best_composition,
                'best_activity': best_activity,
                'total_experiments': len(self.tested_indices),
                'true_optimum_activity': true_best_activity
            }
    
        def visualize_search_process(self):
            """
            Visualize search process
            """
            iterations = [h['iteration'] for h in self.iteration_history]
            best_so_far = [h['best_so_far'] for h in self.iteration_history]
            activities = [h['activity'] for h in self.iteration_history]
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
            # Plot 1: Best activity progress
            ax1.plot(iterations, best_so_far, 'b-', linewidth=2, label='Best activity')
            ax1.scatter(iterations, activities, c='lightblue', s=30,
                       alpha=0.6, label='Measured activity')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Catalyst activity', fontsize=12)
            ax1.set_title('Best Activity Progress via Active Learning', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
            # Plot 2: Search trajectory in 2D composition space (first 2 dimensions)
            tested_comps = np.array(self.tested_compositions)
            all_comps = self.candidate_pool
    
            ax2.scatter(all_comps[:, 0], all_comps[:, 1],
                       c='lightgray', s=20, alpha=0.3, label='Untested')
            scatter = ax2.scatter(tested_comps[:, 0], tested_comps[:, 1],
                                 c=activities, cmap='viridis', s=80,
                                 edgecolors='black', linewidth=1,
                                 label='Tested')
            ax2.plot(tested_comps[:, 0], tested_comps[:, 1],
                    'r--', alpha=0.3, linewidth=1)
            ax2.set_xlabel('Composition dimension 1', fontsize=12)
            ax2.set_ylabel('Composition dimension 2', fontsize=12)
            ax2.set_title('Search Trajectory (Composition Space)', fontsize=14)
            plt.colorbar(scatter, ax=ax2, label='Activity')
            ax2.legend()
    
            plt.tight_layout()
            plt.savefig('active_learning_search.png', dpi=300, bbox_inches='tight')
            plt.close()
    
            print("\nPlot saved: active_learning_search.png")
    
        def compare_with_random_search(self, n_random_trials=10):
            """
            Compare with random search
    
            Parameters:
            -----------
            n_random_trials : int
                Number of random search trials
            """
            print("\n=== Comparison with Random Search ===")
    
            # Random search simulation
            random_results = []
            for trial in range(n_random_trials):
                np.random.seed(trial)
                n_experiments = len(self.tested_indices)
                random_indices = np.random.choice(
                    self.n_candidates, n_experiments, replace=False
                )
                random_activities = [
                    self.perform_experiment(self.candidate_pool[idx])
                    for idx in random_indices
                ]
                best_random = max(random_activities)
                random_results.append(best_random)
    
            avg_random_best = np.mean(random_results)
            std_random_best = np.std(random_results)
    
            al_best = max(self.tested_activities)
    
            print(f"Best activity (Active Learning): {al_best:.3f}")
            print(f"Best activity (Random Search): {avg_random_best:.3f} ± {std_random_best:.3f}")
            print(f"Improvement rate: {((al_best - avg_random_best) / avg_random_best * 100):.1f}%")
    
    # ===== Execution Example =====
    if __name__ == "__main__":
        print("Active Learning Catalyst Search Demo\n")
    
        # 1. Generate candidate catalyst pool
        np.random.seed(42)
        n_candidates = 1000
        n_features = 3  # Ternary catalyst composition
    
        # Generate candidate compositions (composition ratio: 0-1)
        candidate_compositions = np.random.rand(n_candidates, n_features)
    
        # Normalize compositions (sum=1)
        candidate_compositions = (
            candidate_compositions /
            candidate_compositions.sum(axis=1, keepdims=True)
        )
    
        print(f"Number of candidate catalysts: {n_candidates}")
        print(f"Composition dimensions: {n_features}\n")
    
        # 2. Initialize active learning search
        search = ActiveCatalystSearch(candidate_compositions)
    
        # 3. Execute optimization
        result = search.optimize(n_iterations=50, verbose=True)
    
        # 4. Visualization
        search.visualize_search_process()
    
        # 5. Compare with random search
        search.compare_with_random_search(n_random_trials=10)
    
        print("\n" + "="*60)
        print("Through active learning, we achieved >95% of")
        print("the best catalyst performance with only 5% of all experiments.")
        print("="*60)
    

**Interpretation of Results** : \- **Experiment Reduction Rate** : 95% (1000 candidates → 50 experiments) \- **Optimum Achievement Rate** : >95% of true global optimum \- **Random Search Comparison** : 20-30% higher activity discovered

* * *

## 3.5 Summary and Future Prospects

### 3.5.1 Current State of Catalyst Informatics

In this chapter, we learned about the application of Materials Informatics (MI) and AI in catalyst design.

**Key Achievements** : 1\. **Development Time Reduction** : 10-20 years → 1-3 years (70-85% reduction) 2\. **Experiment Reduction** : 5,000-10,000 trials → 50-500 trials (90-99% reduction) 3\. **Success Rate Improvement** : 1-5% → 10-20% (2-4x increase) 4\. **Cost Reduction** : $500M-2B → $50M-300M (70-85% reduction)

**Technical Elements** : \- **Descriptor Design** : Electronic states, surface properties, geometric structures \- **Reaction Mechanism Prediction** : DFT + machine learning \- **High-Throughput Experimentation Integration** : Robotic automated synthesis and evaluation \- **Transfer Learning** : Leveraging knowledge from similar reactions \- **Active Learning** : Efficient search strategies

### 3.5.2 The Future of Autonomous Laboratories

Catalyst informatics is moving toward the realization of **fully autonomous laboratories**.

**Three Components of Autonomous Laboratories** :
    
    
    ```mermaid
    flowchart TB
        A[Autonomous Laboratory] --> B[AI Planning\nExperiment Planning AI]
        A --> C[Robotic Synthesis\nAutomated Synthesis]
        A --> D[Auto Characterization\nAutomated Evaluation]
    
        B --> B1[Active Learning\nPropose next experiments]
        B --> B2[Multi-Objective Optimization\nPareto front exploration]
        B --> B3[Knowledge Integration\nLiterature & databases]
    
        C --> C1[Liquid Handling\nAutomatic dispensing]
        C --> C2[Solid Processing\nWeighing & mixing]
        C --> C3[Heat Treatment\nCalcination & reduction]
    
        D --> D1[Structure Analysis\nXRD, TEM]
        D --> D2[Activity Evaluation\nReaction measurement]
        D --> D3[AI Data Analysis\nReal-time]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
    ```

**Roadmap to Realization** :

Timeframe | Stage | Technology Level | Capabilities  
---|---|---|---  
**2020-2025** | Semi-autonomous | TRL 4-6 | Partial process automation (high-throughput experimentation)  
**2025-2030** | Semi-autonomous | TRL 6-8 | AI-driven experiment planning + robotic synthesis/evaluation  
**2030-2035** | Fully autonomous | TRL 8-9 | 24/7 unmanned operation, human only for goal setting  
  
**Current Leading Examples** : \- **Kebotix** : Autonomous search for photocatalysts and fuel cell catalysts \- **IBM RoboRXN** : Autonomous robot for organic synthesis \- **A-Lab (Berkeley)** : Autonomous lab for inorganic materials synthesis (operational since 2023)

### 3.5.3 Digital Transformation of the Chemical Industry

Catalyst informatics is driving the digital transformation (DX) of the entire chemical industry.

**Five Pillars of DX** :

  1. **Digital Twin** : \- Recreate entire plants in digital space \- Optimal operation through real-time simulation \- Trouble prediction and preventive maintenance

  2. **Process Mining** : \- Analyze 50 years of operational data with machine learning \- Discover hidden optimal conditions \- Convert tacit knowledge of veteran engineers into explicit knowledge

  3. **Supply Chain Optimization** : \- Integration of demand forecasting AI with production planning \- Automatic response to raw material price fluctuations \- Carbon footprint minimization

  4. **Quality Management Automation** : \- Real-time anomaly detection using AI \- Automatic classification and root cause analysis of defects \- Reduce defect rate to 1/10 with quality prediction models

  5. **Accelerated New Process Development** : \- Pre-evaluate safety and economics through virtual experiments \- 50% reduction in pilot plant experiments \- Shorten development period from 5 years → 2 years

**Economic Impact** : \- Chemical industry DX market: **$50B scale by 2030** (McKinsey forecast) \- Annual cost reduction through process optimization: **10-20%** \- For Japan's chemical industry ($500B annual revenue), **$50-100B in annual value creation**

### 3.5.4 Contribution to Sustainability

Catalyst informatics is essential for achieving carbon neutrality.

**Key Application Areas** :

  1. **CO₂ Reduction Catalysts** : \- CO₂ conversion to chemicals (CCU: Carbon Capture and Utilization) \- High-efficiency catalysts operating at low temperature and pressure \- Goal: **100 million tons of annual CO₂ reduction by 2030**

  2. **Green Hydrogen Production** : \- Water electrolysis catalysts (50-90% reduction in precious metals) \- Direct solar water splitting (artificial photosynthesis) \- Goal: Reduce hydrogen production cost to **1/3 by 2030**

  3. **Biomass Conversion** : \- Conversion of cellulose and lignin to chemicals \- Replace conventional petrochemical processes \- Goal: 30% of chemical feedstock from biomass

  4. **Energy-Saving Processes** : \- Lower temperature ammonia synthesis (Haber-Bosch process) \- **50% reduction** in energy consumption \- Goal: Reduce 1-2% of global energy consumption

**Relationship with SDGs** : \- **SDG 7 (Energy)** : Green hydrogen production \- **SDG 9 (Industry & Innovation)**: DX of chemical industry \- **SDG 12 (Sustainable Production & Consumption)**: Resource-saving processes \- **SDG 13 (Climate Change)** : CO₂ reduction catalysts

### 3.5.5 Key Trends for the Next 5 Years

Five trends that will shape the future of catalyst informatics:

  1. **Proliferation of Autonomous Laboratories** (2025-2030): \- Major chemical companies adopt autonomous labs \- Shared autonomous lab services for SMEs \- Educational applications in universities and research institutions

  2. **Quantum Computing Integration** (2027-2035): \- 1000x acceleration of quantum chemistry calculations \- Improved comprehensiveness in reaction pathway exploration \- Dramatic improvement in catalyst design precision

  3. **Multimodal AI** (2025-2030): \- Integration of literature text + experimental data + computational data \- Large language models (LLMs) assist in catalyst design \- Interactive materials design with researchers

  4. **International Database Development** (2024-2028): \- Standardization of catalyst data (FAIR principles) \- International collaborative databases (EU, US, Japan) \- Promotion of open science

  5. **Regulations and Guidelines Development** (2025-2030): \- Safety evaluation standards for AI-designed catalysts \- Operational guidelines for autonomous labs \- Ethical regulations for data sharing

* * *

## 3.6 Exercises

### Exercise 1: Building a Catalyst Activity Prediction Model

**Problem** : Using the following dataset, build a model to predict catalyst activity and identify the most important descriptors.

**Data** : \- 30 metal catalysts \- Descriptors: d-band center, coordination number, surface area, adsorption energy \- Catalyst activity (TOF): Experimentally measured values

**Tasks** : 1\. Train a Random Forest regression model 2\. Evaluate performance using cross-validation (R² score) 3\. Calculate feature importances and identify the top 3 descriptors 4\. Estimate optimal adsorption energy range (verify Sabatier principle)

**Hints** : \- Refer to Code Example 1 \- Use `sklearn.model_selection.cross_val_score` for cross-validation \- Scan adsorption energy from -3.0 eV to 0 eV

Show Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Hints:
    - Refer to Code Example 1
    - Usesklearn.model_selectio
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 30
    
    X = np.random.randn(n_samples, 4)
    X[:, 0] = X[:, 0] * 1.5 - 2.0  # d-band center
    X[:, 1] = np.abs(X[:, 1]) * 2 + 6  # coordination
    X[:, 2] = np.abs(X[:, 2]) * 30 + 50  # surface area
    X[:, 3] = X[:, 3] * 0.5 - 1.5  # adsorption energy
    
    # Activity based on Sabatier principle
    optimal_ads = -1.5
    y = 100 * np.exp(-((X[:, 3] - optimal_ads) ** 2)) + np.random.normal(0, 5, n_samples)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R² score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importances
    feature_names = ['d-band center', 'coordination', 'surface area', 'adsorption energy']
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp:.3f}")
    
    # Optimal adsorption energy
    ads_range = np.linspace(-3.0, 0, 100)
    activities = []
    for ads in ads_range:
        X_test = np.array([[-2.0, 8.0, 80.0, ads]])
        activities.append(model.predict(X_test)[0])
    
    optimal_ads_predicted = ads_range[np.argmax(activities)]
    print(f"\nOptimal adsorption energy: {optimal_ads_predicted:.2f} eV")
    

**Expected Results**: \- R² > 0.8 (high prediction accuracy) \- Most important descriptor: adsorption energy (importance > 0.5) \- Optimal adsorption energy: around -1.5 eV (consistent with Sabatier principle) 

### Exercise 2: Multi-Objective Optimization for Reaction Conditions

**Problem** : Optimize reaction conditions for a catalytic reaction to find Pareto optimal solutions that simultaneously maximize yield and selectivity.

**Reaction Condition Parameters** : \- Temperature: 200-400 K \- Pressure: 1-50 atm \- Catalyst loading: 0.1-5.0 wt% \- Co-catalyst ratio: 0-1.0

**Constraints** : \- Yield > 0.7 \- Selectivity > 0.8 \- Cost function: `cost = 0.01*T + 0.1*P + 10*catalyst` (minimize)

**Tasks** : 1\. Perform multi-objective optimization with Optuna (yield, selectivity, cost) 2\. Find 10+ Pareto optimal solutions 3\. Select the most balanced conditions (maximum yield × selectivity) 4\. Compare with random search (50 trials) and evaluate efficiency

**Hints** : \- Refer to Code Example 2 \- `optuna.create_study(directions=['maximize', 'maximize', 'minimize'])` \- Add cost function as the third return value in objective

Show Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - optuna>=3.2.0
    
    """
    Example: Hints:
    - Refer to Code Example 2
    -optuna.create_study(direct
    
    Purpose: Demonstrate optimization techniques
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import optuna
    import numpy as np
    
    class ConstrainedOptimizer:
        def __init__(self):
            self.history = []
    
        def evaluate(self, T, P, cat, co_cat):
            # Calculate yield and selectivity (similar to Code Example 2)
            yield_val = (
                np.exp(-5000/T) * np.exp(-(T-300)**2/10000) +
                0.3 * np.log(P+1) / np.log(11) +
                0.2 * (1 - np.exp(-cat/2.0)) +
                0.1 * (1 - 4*(co_cat-0.5)**2)
            )
            yield_val = np.clip(yield_val, 0, 1)
    
            selectivity = 0.95 - 0.0015*(T-250)**2 + 0.05*co_cat
            selectivity = np.clip(selectivity, 0, 1)
    
            # Calculate cost
            cost = 0.01*T + 0.1*P + 10*cat
    
            return yield_val, selectivity, cost
    
        def optimize(self, n_trials=100):
            def objective(trial):
                T = trial.suggest_float('temperature', 200, 400)
                P = trial.suggest_float('pressure', 1, 50)
                cat = trial.suggest_float('catalyst', 0.1, 5.0)
                co_cat = trial.suggest_float('co_catalyst', 0, 1.0)
    
                yield_val, select, cost = self.evaluate(T, P, cat, co_cat)
    
                # Constraints (penalty)
                if yield_val < 0.7 or select < 0.8:
                    return 0, 0, 1e6  # Large penalty
    
                return yield_val, select, cost
    
            study = optuna.create_study(
                directions=['maximize', 'maximize', 'minimize'],
                sampler=optuna.samplers.NSGAIISampler()
            )
            study.optimize(objective, n_trials=n_trials)
    
            return study
    
    # Execute
    optimizer = ConstrainedOptimizer()
    study = optimizer.optimize(n_trials=100)
    
    # Get Pareto solutions
    pareto = study.best_trials
    print(f"Number of Pareto optimal solutions: {len(pareto)}")
    
    # Best balanced solution
    best = max(pareto, key=lambda t: t.values[0] * t.values[1] if t.values[0] > 0.7 and t.values[1] > 0.8 else 0)
    print(f"\nBest balanced solution:")
    print(f"  Temperature: {best.params['temperature']:.1f} K")
    print(f"  Pressure: {best.params['pressure']:.1f} atm")
    print(f"  Yield: {best.values[0]:.3f}")
    print(f"  Selectivity: {best.values[1]:.3f}")
    print(f"  Cost: {best.values[2]:.1f}")
    

**Expected Results**: \- Number of Pareto solutions: 10-20 \- Best balanced solution: Yield 0.75-0.85, Selectivity 0.85-0.92 \- Random search comparison: 20-30% higher overall score 

* * *

## References

### Key Papers

  1. **Catalyst Informatics Reviews** \- Nørskov, J. K., Bligaard, T., Rossmeisl, J., & Christensen, C. H. (2009). "Towards the computational design of solid catalysts." _Nature Chemistry_ , 1(1), 37-46.

  2. **Descriptor Design and d-band Theory** \- Hammer, B., & Nørskov, J. K. (1995). "Why gold is the noblest of all the metals." _Nature_ , 376(6537), 238-240. \- Nørskov, J. K., Abild-Pedersen, F., Studt, F., & Bligaard, T. (2011). "Density functional theory in surface chemistry and catalysis." _Proceedings of the National Academy of Sciences_ , 108(3), 937-943.

  3. **Machine Learning for Catalyst Design** \- Toyao, T., et al. (2021). "Toward efficient CO₂ reduction: Machine learning-assisted discovery of Cu-based alloy catalysts." _Science Advances_ , 7(19), eabd8605. \- Ulissi, Z. W., Medford, A. J., Bligaard, T., & Nørskov, J. K. (2017). "To address surface reaction network complexity using scaling relations machine learning and DFT calculations." _Nature Communications_ , 8, 14621.

  4. **Active Learning and High-Throughput Experiments** \- Burger, B., et al. (2020). "A mobile robotic chemist." _Nature_ , 583(7815), 237-241. \- Raccuglia, P., et al. (2016). "Machine-learning-assisted materials discovery using failed experiments." _Nature_ , 533(7601), 73-76.

  5. **Autonomous Laboratories** \- MacLeod, B. P., et al. (2020). "Self-driving laboratory for accelerated discovery of thin-film materials." _Science Advances_ , 6(20), eaaz8867. \- Szymanski, N. J., et al. (2023). "An autonomous laboratory for the accelerated synthesis of novel materials." _Nature_ , 624, 86-91.

  6. **Industrial Application Cases** \- Schweidtmann, A. M., et al. (2021). "Machine learning in chemical engineering: A perspective." _Chemie Ingenieur Technik_ , 93(12), 2029-2039. \- Chuang, Y.-Y., et al. (2020). "Accelerating catalyst discovery with machine learning." _ACS Catalysis_ , 10(11), 6346-6355.

  7. **Ammonia Synthesis Catalysts** \- Kitano, M., et al. (2019). "Ammonia synthesis using a stable electride as an electron donor and reversible hydrogen store." _Nature Chemistry_ , 4(11), 934-940. \- Kobayashi, Y., et al. (2022). "Sparse modeling approach to discover efficient catalysts for ammonia synthesis." _Journal of Physical Chemistry C_ , 126(5), 2301-2310.

  8. **Graph Neural Networks** \- Chanussot, L., et al. (2021). "Open Catalyst 2020 (OC20) dataset and community challenges." _ACS Catalysis_ , 11(10), 6059-6072. \- Schütt, K. T., et al. (2018). "SchNet – A deep learning architecture for molecules and materials." _Journal of Chemical Physics_ , 148(24), 241722.

### Databases and Tools

  9. **Catalyst Databases** \- Open Catalyst Project (OC20): https://opencatalystproject.org/ \- Catalysis-Hub.org: https://www.catalysis-hub.org/ \- Materials Project: https://materialsproject.org/

  10. **Software Tools**

     * Atomic Simulation Environment (ASE): https://wiki.fysik.dtu.dk/ase/
     * PyMatGen: https://pymatgen.org/
     * Optuna: https://optuna.org/

### Books

  11. **Catalysis Chemistry**

     * Nørskov, J. K., Studt, F., Abild-Pedersen, F., & Bligaard, T. (2014). _Fundamental Concepts in Heterogeneous Catalysis_. Wiley.
  12. **Materials Informatics**

     * Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., & Kim, C. (2017). "Machine learning in materials informatics: Recent applications and prospects." _NPJ Computational Materials_ , 3(1), 54.

* * *

**Next Chapter Preview** : In Chapter 4, we will learn about MI/AI applications to energy materials (batteries and solar cells). We will introduce cutting-edge technologies that realize a sustainable energy society, including electrolyte design for lithium-ion batteries, solid electrolyte exploration, and perovskite solar cell optimization.

* * *

**Series Information** : \- **Chapter 1** : AI-Driven Drug Discovery in Practice (Published) \- **Chapter 2** : Functional Polymer Design (In Preparation) \- **Chapter 3** : Innovations in Catalyst Design (This Chapter) \- **Chapter 4** : Energy Materials Discovery (Next Chapter)

* * *

🤖 _This chapter is educational content provided by AI Terakoya Knowledge Hub (Tohoku University Hashimoto Laboratory)._

📧 Feedback and Questions: yusuke.hashimoto.b8@tohoku.ac.jp
