---
title: "Chapter 4: Real-World Applications and Career Paths"
subtitle: "Case Studies and Career Pathways"
chapter: 4
reading_time: "20-25 minutes"
level: "Intermediate"
keywords: ["Case Studies", "CNT", "Quantum Dots", "Gold Nanoparticles", "Graphene", "Drug Delivery", "Career"]
prev_chapter: "chapter3-hands-on.html"
next_chapter: "index.html"
last_updated: "2025-10-16"
---

# Chapter 4: Real-World Applications and Career Paths

**Case Studies and Career Pathways**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. **Understand Practical Applications**: Explain the process from research to commercialization through five success stories: CNT, quantum dots, gold nanoparticles, graphene, and nanomedicine
2. **Role of Machine Learning**: Understand how machine learning contributed to reducing development time and costs in each case study
3. **Challenges and Solutions**: Explain common challenges in nanomaterial commercialization (scale-up, cost, safety) and their solution approaches
4. **Career Paths**: Compare the differences, advantages, and disadvantages of careers in academia, industry, and startups
5. **Required Skills**: Identify technical and business skills needed for success in the nanomaterials field
6. **Future Outlook**: Understand major trends such as AI-driven materials design, sustainable nanomaterials, and nano-bio fusion

---

## 4.1 Case Study 1: Mechanical Property Optimization of Carbon Nanotube Composites

### Background and Challenges

In the aerospace industry, weight reduction is the top priority for improving fuel efficiency. However, achieving both lightweight and maintaining strength/stiffness are conflicting requirements, and conventional aluminum alloys and carbon fiber reinforced plastics (CFRP) alone have reached their limits.

**Potential of Carbon Nanotubes (CNTs)**:
- Theoretical tensile strength: 100 GPa (100 times that of steel)
- Young's modulus: 1 TPa (5 times that of steel)
- Density: 1.3-1.4 g/cm³ (half that of aluminum)

However, composite materials mixing CNTs into matrices such as epoxy resin could only achieve a few percent of the theoretical values. The main challenges were:

1. **CNT Agglomeration**: Difficult to achieve uniform dispersion due to van der Waals forces causing bundling
2. **Poor Interfacial Adhesion**: Weak chemical bonding between CNT surfaces and matrix, resulting in low load transfer efficiency
3. **Unclear Optimal Formulation**: Optimal values for CNT content, length, diameter, and dispersion conditions are multidimensional and complex

### Project Overview

**Goals**:
- 50% improvement in tensile strength (70 MPa → 105 MPa or higher)
- 20% weight reduction
- Keep cost increase within 30%

**Duration**: 2 years (2021-2023)

**Team Composition**:
- University laboratory (CNT synthesis/surface modification): 5 members
- Aircraft manufacturer research institute (composite material evaluation): 8 members
- Data scientists (machine learning model development): 2 members

### Applied Nanomaterial Technologies

#### 1. CNT Surface Modification

Carboxyl groups (-COOH) were introduced to the CNT surface to enhance chemical bonding with epoxy resin.

**Process**:
```
CNT + concentrated nitric/sulfuric acid mixture (3:1) → 80°C, 4 hours
→ Ultrasonic cleaning (pure water) → Vacuum drying (60°C, 12 hours)
```

**Evaluation**:
- X-ray Photoelectron Spectroscopy (XPS): Surface oxygen concentration 5% → 18%
- Fourier Transform Infrared Spectroscopy (FTIR): C=O stretching vibration confirmed at 1730 cm⁻¹
- Raman Spectroscopy: D/G ratio 0.15 → 0.22 (slight structural defect increase)

#### 2. Optimization of Ultrasonic Dispersion Process

**Condition Investigation**:
- Ultrasonic power: 100-500 W
- Processing time: 10-60 minutes
- Solvent: Acetone, ethanol, N-methylpyrrolidone (NMP)
- Dispersant: Sodium dodecylbenzenesulfonate (SDBS)

**Optimal Conditions**:
- Ultrasonic power: 300 W
- Processing time: 30 minutes
- Solvent: NMP
- Dispersant concentration: 0.5 wt%

**Dispersion Evaluation**:
- Transmission Electron Microscopy (TEM): Confirmed individually dispersed CNTs (bundle size 5-10 tubes)
- Dynamic Light Scattering (DLS): Average particle size 150 nm (untreated: 2,500 nm)

#### 3. Length Separation by Centrifugation

Mixing CNTs of different lengths increases stress concentration points and reduces strength. Length uniformity was achieved through centrifugation.

**Conditions**:
- Rotation speed: 10,000 rpm, 30 minutes
- Separate supernatant (short CNTs) and precipitate (long CNTs)
- Optimal length range: 1-3 μm (TEM measurement)

### Machine Learning Methods Used

#### Data Collection

**Input Variables (8 dimensions)**:
1. CNT content: 0.5-5.0 wt%
2. CNT average length: 0.5-5.0 μm
3. CNT average diameter: 5-20 nm
4. Surface oxygen concentration: 5-20%
5. Ultrasonic power: 100-500 W
6. Ultrasonic time: 10-60 minutes
7. Curing temperature: 100-150°C
8. Curing time: 2-8 hours

**Output Variables**:
- Tensile strength (MPa)
- Young's modulus (GPa)
- Elongation at break (%)

**Experimental Data**: 300 samples (3 measurements per condition)

#### Model Selection: Random Forest

**Reasons**:
- Can capture nonlinear relationships
- Can quantify feature importance
- Robust against overfitting (ensemble learning)
- Effective even with small data (300 samples)

**Hyperparameter Tuning**:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Optimal parameters
# n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1
```

**Model Performance**:
- Training data R²: 0.94
- Test data R²: 0.88
- RMSE (tensile strength): 3.2 MPa

#### Bayesian Optimization for Optimal Formulation Search

Using the random forest model, Bayesian optimization was used to search for conditions that maximize tensile strength.

**Algorithm**: Gaussian Process (GP) based
**Acquisition Function**: Expected Improvement (EI)

**Optimization Results**:
| Parameter | Optimal Value |
|-----------|---------------|
| CNT content | 2.3 wt% |
| CNT average length | 2.1 μm |
| CNT average diameter | 12 nm |
| Surface oxygen concentration | 16% |
| Ultrasonic power | 320 W |
| Ultrasonic time | 28 minutes |
| Curing temperature | 130°C |
| Curing time | 4.5 hours |

**Predicted Tensile Strength**: 107 MPa

#### Validation Experiments

Five samples fabricated and measured under optimal conditions:

| Sample | Tensile Strength (MPa) | Young's Modulus (GPa) | Elongation at Break (%) |
|---------|------------------------|----------------------|------------------------|
| 1 | 105.2 | 4.8 | 3.1 |
| 2 | 106.8 | 4.9 | 3.3 |
| 3 | 104.5 | 4.7 | 3.0 |
| 4 | 107.1 | 5.0 | 3.2 |
| 5 | 105.9 | 4.8 | 3.1 |
| **Average** | **105.9 ± 1.0** | **4.84 ± 0.11** | **3.14 ± 0.11** |

The error between predicted value (107 MPa) and measured value (105.9 MPa) was 1%, achieving high-precision prediction.

### Results and Impact

#### Improvement in Mechanical Properties

**Comparison with Conventional Material (Pure Epoxy)**:
| Property | Pure Epoxy | CNT Composite | Improvement |
|---------|-----------|--------------|-------------|
| Tensile Strength | 70 MPa | 106 MPa | **+51%** |
| Young's Modulus | 3.2 GPa | 4.8 GPa | **+50%** |
| Elongation at Break | 4.5% | 3.1% | -31% (Trade-off) |
| Density | 1.20 g/cm³ | 1.23 g/cm³ | +2.5% |

**Effective Weight Reduction**:
Comparing the mass of components with the same strength, CNT composite material achieved **22% weight reduction** compared to conventional materials.

#### Cost and Environmental Impact

**Material Cost (per kg)**:
- Pure epoxy resin: $15
- CNT (surface modified): $250/kg × 2.3% = $5.75
- **CNT composite: $20.75** (conventional +38%)

Although it didn't meet the target increase of within 30%, it was estimated to be recoverable in 5 years through fuel efficiency improvement due to weight reduction (annual savings $50,000/aircraft).

**CO₂ Reduction Effect**:
- 1 kg reduction in aircraft weight → approximately 3,000 L lifetime fuel consumption reduction
- CO₂ emission reduction: approximately 7.5 t-CO₂/aircraft/lifetime

#### Path to Commercialization

**2023**: Consideration began for adoption in Boeing 787 tail spar (main structural member).

**Remaining Challenges**:
1. **Scale-up**: Transition from laboratory scale (100 g) to mass production scale (100 kg) manufacturing process
2. **Quality Control**: Establishment of non-destructive inspection methods for CNT dispersion state and length distribution
3. **Long-term Durability**: Evaluation of fatigue characteristics and environmental degradation for over 20 years of use

### Lessons Learned

1. **Importance of Surface Modification**: Surface chemical modification of CNTs improved interfacial adhesion, enabling performance close to theoretical values
2. **Power of Machine Learning**: Conventional trial and error would have required over 1,000 experiments, but machine learning completed optimization with 300 experiments (1/3 experiments, 1/2 time)
3. **Need for Multi-objective Optimization**: Not only strength, but also cost, processability, and environmental impact must be considered simultaneously
4. **Scale-up Barrier**: Fluid dynamics of dispersion processes differ between laboratory and factory, requiring re-optimization

---

## 4.2 Case Study 2: Emission Wavelength Control of Quantum Dots

### Background and Challenges

Quantum Dots (QDs) are semiconductor nanoparticles whose emission wavelength changes continuously with size. This property is expected to be applied to next-generation displays (QLED).

**Challenges of Conventional Displays**:
- Liquid Crystal Displays (LCD): Narrow color gamut (sRGB coverage 70-80%)
- Organic EL (OLED): Short lifespan of blue elements (less than 10,000 hours)

**Advantages of Quantum Dots**:
- Narrow emission spectrum (half-width 25-35 nm) → High color purity
- Can realize arbitrary wavelengths through size control
- High luminous efficiency (quantum yield 80-95%)
- Long lifespan (over 50,000 hours)

**Technical Challenges**:
1. **Size Uniformity**: Precision of ±5% or less is required (±10% causes color unevenness)
2. **Luminous Efficiency**: Blue QDs have low efficiency (60-75%)
3. **Stability**: Emission degradation due to oxidation and agglomeration

### Project Overview

**Goals**:
- Establish manufacturing process for RGB 3-color quantum dots
- Size uniformity: ±5% or less
- Luminous efficiency: 80% or higher
- Color gamut: DCI-P3 coverage 100% or higher

**Duration**: 18 months (April 2022 - September 2023)

**Team Composition**:
- Display manufacturer research institute: 6 members
- University chemical engineering department (QD synthesis): 4 members
- University information science department (machine learning): 2 members

### Applied Nanomaterial Technologies

#### 1. Synthesis by Hot Injection Method

This is the standard synthesis method for CdSe quantum dots. By rapidly injecting precursors into high-temperature solvent, nucleation and growth are separated to narrow size distribution.

**Reaction Scheme**:
```
Cd(CH₃)₂ + Se powder → [In trioctylphosphine (TOP), room temperature]
→ TOPSe (selenium precursor)

Cd(OAc)₂ + oleic acid → [In octadecene, 280°C]
→ Cd-oleic acid complex

Rapid injection of TOPSe → CdSe nucleation (<1 second)
→ Lower temperature to 220-260°C for growth (5-30 minutes)
```

**Synthesis Condition Control**:
| Target Wavelength | Target Size | Reaction Temperature | Reaction Time | Precursor Ratio (Cd:Se) |
|-------------------|-------------|---------------------|---------------|------------------------|
| Red (650 nm) | 6.2 nm | 260°C | 15 minutes | 1:0.8 |
| Green (550 nm) | 4.1 nm | 240°C | 8 minutes | 1:1.0 |
| Blue (450 nm) | 2.8 nm | 220°C | 5 minutes | 1:1.2 |

#### 2. Size-Selective Precipitation

QDs after synthesis still have size distribution (±10-15%). Distribution is narrowed through size-selective precipitation.

**Process**:
1. Add ethanol gradually to QD toluene dispersion
2. Larger QDs selectively precipitate
3. Centrifugation (5,000 rpm, 10 minutes)
4. Separate supernatant (smaller QDs) and precipitate (larger QDs)
5. Repeat 3-5 times, collect only QDs of target size range

**Effect**:
- Initial size distribution: 4.1 ± 0.6 nm (±14.6%)
- After precipitation: 4.1 ± 0.15 nm (±3.7%)

#### 3. ZnS Shell Coating

Forming ZnS shell on CdSe core surface improves luminous efficiency and prevents oxidation.

**Core/Shell Structure**:
```
[CdSe core (diameter d)] + [ZnS shell (thickness t)]
→ Total particle size = d + 2t
```

**Shell Growth Conditions**:
- Zn(OAc)₂ + sulfur powder (in TOP) → 220°C, slow addition (0.5 mL/h)
- Shell thickness: 0.8-1.2 nm (2-3 monolayers)

**Effects**:
| Property | CdSe Core Only | CdSe/ZnS Core/Shell |
|---------|----------------|---------------------|
| Luminous Efficiency | 45-60% | 75-95% |
| Photostability | 50% decrease after 100 hours continuous irradiation | 10% decrease after 1,000 hours continuous irradiation |
| Chemical Stability | Oxidizes in air within days | Stable in air for months |

### Machine Learning Methods Used

#### Data Collection

**Input Variables (10 dimensions)**:
1. Cd precursor concentration: 0.01-0.1 M
2. Se precursor concentration: 0.008-0.12 M
3. Cd:Se ratio: 0.8-1.5
4. Reaction temperature: 200-280°C
5. Reaction time: 1-30 minutes
6. Oleic acid concentration: 0.5-2.0 M
7. Injection rate: 0.5-5.0 mL/s
8. Shell thickness: 0-1.5 nm
9. Shell growth temperature: 200-240°C
10. Shell growth time: 10-120 minutes

**Output Variables**:
- Average particle size (nm, TEM measurement)
- Size distribution standard deviation (nm)
- Emission wavelength (nm, PL spectroscopy measurement)
- Luminous efficiency (%, integrating sphere measurement)

**Experimental Data**: 450 samples (2 measurements per condition)

#### Model Selection: LightGBM (Gradient Boosting)

**Reasons**:
- Strong with high-dimensional data
- High-precision prediction through gradient boosting
- Faster learning speed than random forest
- Easy feature importance analysis

**Hyperparameter Tuning**:
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    early_stopping_rounds=50
)
```

**Model Performance**:
| Output Variable | Training R² | Test R² | RMSE |
|-----------------|-------------|---------|------|
| Average Particle Size | 0.96 | 0.92 | 0.18 nm |
| Emission Wavelength | 0.94 | 0.89 | 8.5 nm |
| Luminous Efficiency | 0.88 | 0.82 | 4.2% |

#### Combination with Brus Equation

The relationship between quantum dot size and emission wavelength can be approximated by the Brus equation:

$$
E_g(d) = E_{g,bulk} + \frac{\hbar^2 \pi^2}{2d^2} \left( \frac{1}{m_e^*} + \frac{1}{m_h^*} \right) - \frac{1.8e^2}{4\pi \epsilon \epsilon_0 d}
$$

Where:
- $E_g(d)$: Band gap of quantum dot with size $d$ (eV)
- $E_{g,bulk}$: Band gap of bulk CdSe (1.74 eV)
- $m_e^*, m_h^*$: Effective mass of electron and hole
- $\epsilon$: Relative permittivity of CdSe (10.6)

**Hybrid Approach of Physical Model + Machine Learning**:
1. Calculate initial predicted value of wavelength from Brus equation
2. Correct error with machine learning model (effects of shell, surface states)

This hybrid method improved prediction accuracy (RMSE 8.5 nm → 4.2 nm).

#### Optimization Algorithm

**Multi-objective Optimization Problem**:
- Objective 1: Minimize error from target wavelength (450/550/650 nm)
- Objective 2: Maximize luminous efficiency
- Objective 3: Maximize size uniformity (minimize standard deviation)

**Method**: NSGA-II (Non-dominated Sorting Genetic Algorithm)

### Results and Impact

#### Performance of Each Color Quantum Dot

**Red QD (650 nm)**:
| Property | Achieved Value |
|---------|----------------|
| Average Size | 6.2 ± 0.2 nm |
| Size Uniformity | ±3.2% |
| Emission Wavelength | 652 nm |
| Spectral Half-Width | 28 nm |
| Luminous Efficiency | 85% |
| CIE Color Coordinates | (0.68, 0.32) |

**Green QD (550 nm)**:
| Property | Achieved Value |
|---------|----------------|
| Average Size | 4.1 ± 0.15 nm |
| Size Uniformity | ±3.7% |
| Emission Wavelength | 548 nm |
| Spectral Half-Width | 30 nm |
| Luminous Efficiency | 90% |
| CIE Color Coordinates | (0.21, 0.71) |

**Blue QD (450 nm)**:
| Property | Achieved Value |
|---------|----------------|
| Average Size | 2.8 ± 0.1 nm |
| Size Uniformity | ±3.6% |
| Emission Wavelength | 452 nm |
| Spectral Half-Width | 32 nm |
| Luminous Efficiency | 75% |
| CIE Color Coordinates | (0.14, 0.06) |

#### Display Performance

**Color Gamut**:
- DCI-P3 coverage: **110%** (exceeded 100% target)
- Rec.2020 coverage: 85% (future 8K broadcasting standard)

**Comparison with Conventional Technologies**:
| Display Technology | DCI-P3 Coverage | Peak Brightness | Lifespan (Half-life) |
|--------------------|-----------------|-----------------|---------------------|
| Typical LCD | 72% | 300 nits | >50,000 hours |
| High-end LCD (wide color gamut backlight) | 95% | 500 nits | >50,000 hours |
| OLED | 105% | 800 nits | 10,000 hours (blue) |
| **QLED (This Research)** | **110%** | **1,000 nits** | **>50,000 hours** |

#### Commercialization

**2024**: Samsung launched 55-inch QLED TV (Model: QN55S95C)

**Market Response**:
- First month sales: 12,000 units
- Display industry magazine review: Highest rating for color reproduction
- Price: $2,499 (+15% compared to same-size OLED)

**Future Developments**:
- 2025: Launch of 75-inch model, 8K resolution model planned
- 2026: Application to notebook PC displays (15.6 inches)

### Lessons Learned

1. **Size Uniformity is Everything**: Even ±5% size distribution is visually recognized as color unevenness, making size-selective precipitation essential
2. **Importance of Shell Coating**: Core/shell structure doubled luminous efficiency and increased photostability 10-fold
3. **Challenge of Blue QDs**: Blue has small particle size and large surface area/volume ratio, making it more susceptible to surface defects, resulting in relatively low luminous efficiency
4. **Physical Model + Machine Learning**: Combining physical constraints from Brus equation with machine learning flexibility achieved high-precision prediction with limited data
5. **Successful Scale-up**: Scale-up from laboratory synthesis (10 mL) to mass production (10 L) was relatively smooth (temperature and time control only)

---

## 4.3 Case Study 3: Activity Prediction of Gold Nanoparticle Catalysts

### Background and Challenges

Fuel cells are clean energy devices that directly extract electricity from hydrogen and oxygen. However, current fuel cells depend on platinum (Pt) catalysts, and high cost is the biggest barrier to widespread adoption.

**Platinum Challenges**:
- Price: approximately 4,000 USD/oz (gold: approximately 2,000 USD/oz)
- Rarity: World production approximately 200 t/year (1/15 of gold)
- Platinum usage per fuel cell: approximately 30 g → cost approximately $4,000

**Potential of Gold Nanoparticle Catalysts**:
The "CO oxidation activity of gold nanoparticles" discovered by Haruta et al. in 1997 was a turning point in nanomaterial science. Bulk gold is chemically inert, but gold nanoparticles of 2-5 nm oxidize CO even at room temperature.

**CO Oxidation Reaction**:
$$
2\text{CO} + \text{O}_2 \rightarrow 2\text{CO}_2
$$

This reaction is applied to automobile exhaust purification, indoor air purification, and CO removal in fuel cells.

**Technical Challenges**:
1. **Origin of Activity**: Why does it become active at nano-size?
2. **Size Dependence**: What is the optimal size?
3. **Support Effect**: Which support (TiO₂, Al₂O₃, CeO₂, etc.) is optimal?
4. **Durability**: Won't it agglomerate with long-term use?

### Project Overview

**Goals**:
- CO oxidation activity equivalent to platinum catalyst (CO conversion 80% or higher at 100°C)
- Cost: 1/10 or less of platinum catalyst
- Long-term stability: Maintain 80% or more activity after 1,000 hours

**Duration**: 3 years (2020-2023)

**Team Composition**:
- National research institute (catalyst synthesis/evaluation): 8 members
- Automobile manufacturer research institute (practical application study): 5 members
- University computational science department (first-principles calculation): 3 members
- Data scientists (machine learning): 2 members

### Applied Nanomaterial Technologies

#### 1. Size-Controlled Synthesis of Gold Nanoparticles

**Improved Turkevich Method**:
The classic Turkevich method (citrate reduction method) was improved to synthesize gold nanoparticles of size 2-5 nm.

**Synthesis Process**:
```
Heat HAuCl₄ aqueous solution (0.01%) to 100°C
↓
Add trisodium citrate aqueous solution (1%)
↓
Stir for 15 minutes → Color change (pale yellow → wine red)
↓
Cool → Gold nanoparticle colloid
```

**Size Control**:
- Change HAuCl₄/citrate ratio to adjust size
- Ratio 1:1 → 15 nm (Turkevich method standard)
- Ratio 1:5 → 5 nm
- Ratio 1:10 → 2.5 nm

**Problem**: Particles below 2 nm are unstable and easily agglomerate with citrate method

**Improved Method**: Add polyvinylpyrrolidone (PVP) as protective agent
- Addition of PVP stabilizes ultra-fine particles of 1.5-2.0 nm

#### 2. Loading on TiO₂ Support

**Support Selection**:
| Support | Specific Surface Area (m²/g) | CO Oxidation Activity (Relative) | Cost ($/kg) |
|---------|------------------------------|----------------------------------|-------------|
| TiO₂ (Anatase) | 50 | 1.00 (Standard) | $15 |
| TiO₂ (Rutile) | 10 | 0.35 | $12 |
| Al₂O₃ | 200 | 0.52 | $8 |
| CeO₂ | 80 | 0.88 | $45 |
| SiO₂ | 300 | 0.15 | $5 |

TiO₂ (anatase type) was determined to be optimal in balance of activity, cost, and availability.

**Loading Process**:
1. Add TiO₂ powder to gold nanoparticle colloid
2. pH adjustment (pH 7-8, gold particles electrostatically adsorb to TiO₂ surface)
3. Centrifugation/washing
4. Vacuum drying (60°C, 12 hours)

**Gold Loading Amount**: 0.5-3.0 wt% (weight percent)

#### 3. High-Temperature Annealing Treatment

Annealing (calcination) of the catalyst after drying in air strengthens the gold particle-TiO₂ interface and improves activity.

**Annealing Condition Investigation**:
- Temperature: 200-500°C
- Time: 1-6 hours
- Atmosphere: Air, oxygen, nitrogen

**Optimal Conditions**: 350°C, 2 hours, in air

**Effect Mechanism (First-Principles Calculation)**:
- Annealing forms oxygen defects on TiO₂ surface around gold particles
- These oxygen defects function as CO adsorption sites
- Electron transfer is promoted at gold-TiO₂ interface, accelerating CO oxidation

### Machine Learning Methods Used

#### Data Collection

**Input Variables (12 dimensions)**:
1. Gold particle average size: 1.5-10.0 nm (TEM measurement)
2. Gold particle size distribution standard deviation: 0.1-2.0 nm
3. Gold loading amount: 0.5-3.0 wt%
4. Support type: TiO₂ (anatase/rutile), Al₂O₃, CeO₂, SiO₂ (one-hot encoding)
5. Support specific surface area: 10-300 m²/g
6. Annealing temperature: 200-500°C
7. Annealing time: 1-6 hours
8. Annealing atmosphere: Air, oxygen, nitrogen (one-hot encoding)
9. Reaction temperature: 50-200°C
10. CO concentration: 1-5 vol%
11. O₂ concentration: 10-21 vol%
12. Space velocity (GHSV): 10,000-50,000 h⁻¹

**Output Variables**:
- CO conversion (%, per reaction temperature)
- Activation energy (kJ/mol, Arrhenius equation fitting)
- Long-term stability (activity decrease rate after 1,000 hours)

**Experimental Data**: 680 samples (3 measurements per condition)

#### Model Selection: Neural Network

**Reasons**:
- Can learn complex nonlinear relationships
- Can capture interactions between input variables
- Effective for prediction of physical quantities such as activation energy

**Network Structure**:
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # CO conversion (%)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
)
```

**Model Performance**:
- Training data MAE: 2.8% (absolute error of CO conversion)
- Test data MAE: 3.5%
- R²: 0.91

**Feature Importance (SHAP Value Analysis)**:
1. Gold particle size (relative importance: 0.32)
2. Annealing temperature (0.21)
3. Reaction temperature (0.18)
4. Gold loading amount (0.12)
5. Support type (0.09)
6. Others (0.08)

#### Optimization and Mechanism Elucidation

**Bayesian Optimization for Condition Search**:
- Objective: Maximize CO conversion at 100°C
- Search space: 12 dimensions
- Number of evaluations: 50 (initial data 680 + additional experiments 50)

**Optimal Conditions**:
| Parameter | Optimal Value |
|-----------|---------------|
| Gold Particle Size | 3.2 nm |
| Gold Loading Amount | 1.5 wt% |
| Support | TiO₂ (Anatase) |
| Annealing Temperature | 350°C |
| Annealing Time | 2 hours |
| Reaction Conditions | 1% CO, 20% O₂, GHSV=30,000 h⁻¹ |

**Predicted CO Conversion (100°C)**: 87%

**Validation Experiment**: 85.2 ± 1.8% (5 sample average)

**Size Dependence of Activity**:
- 1.5 nm: Conversion 42% (easily agglomerates)
- **3.2 nm: Conversion 85% (optimal)**
- 5.0 nm: Conversion 58% (surface area decrease)
- 10.0 nm: Conversion 28% (bulk-like properties)

**Interpretation by First-Principles Calculation**:
- At 3.2 nm size, the proportion of low-coordination sites (corners, edges) in gold particles is maximized
- These sites are optimal for CO adsorption and activation

### Results and Impact

#### Catalyst Performance

**Optimal Catalyst (Au/TiO₂)**:
| Reaction Temperature | CO Conversion | Pt Catalyst (Comparison) |
|---------------------|---------------|-------------------------|
| 50°C | 28% | 15% |
| 75°C | 62% | 55% |
| 100°C | 85% | 90% |
| 150°C | 99% | 100% |

- Achieved performance almost equivalent to platinum catalyst
- Gold catalyst is rather superior in low-temperature range (50-75°C)

**Activation Energy**:
- Au/TiO₂: 42 kJ/mol
- Pt/Al₂O₃: 38 kJ/mol

**Long-term Stability**:
- After 1,000 hours continuous operation: CO conversion 85% → 77% (90% maintenance)
- TEM observation: Gold particle size 3.2 nm → 4.1 nm (slight growth)

#### Cost Comparison

**Cost per kg of Catalyst**:
| Component | Au/TiO₂ Catalyst | Pt/Al₂O₃ Catalyst |
|-----------|------------------|-------------------|
| Precious Metal (15 g) | $900 (gold) | $1,800 (platinum) |
| Support | $15 (TiO₂) | $8 (Al₂O₃) |
| Manufacturing Process | $85 | $120 |
| **Total** | **$1,000** | **$1,928** |

**Cost Reduction**: Approximately **1/2** of platinum catalyst (target 1/10 not achieved)

Further reduction of gold loading (1.5 wt% → 1.0 wt%) is a future challenge.

#### Application Development

**2023**: Implementation in indoor air purifier (Panasonic "Ziaino")

**Product Specifications**:
- Coverage area: 40 m²
- CO removal performance: Initial concentration 10 ppm → <1 ppm after 30 minutes
- Filter lifespan: 10 years (conventional activated carbon filter: 2 years)
- Price: $1,200 (+20% compared to conventional product)

**Market Response**:
- 10,000 units sold in 3 months after launch
- Well received for smoking rooms, restaurants, medical facilities

**Future Applications**:
- Automobile exhaust catalyst (Pt replacement, commercialization target 2025)
- Fuel cell CO removal filter

### Lessons Learned

1. **Size Determines Everything**: Highest activity around 3 nm. Narrow optimal range where agglomeration occurs at 2 nm and activity decreases at 5 nm
2. **Interaction with Support**: Gold particles alone are inactive. Activity appears when electron transfer occurs at interface with TiO₂
3. **Effect of Machine Learning**: Reduced number of experiments to 1/3 of conventional (2,000 times → 680+50 times). Development period shortened from 3 years → 1.5 years
4. **Collaboration with First-Principles Calculation**: Machine learning predicts "which conditions are good," first-principles calculation elucidates "why they are good"
5. **Scale-up Challenge**: From laboratory level (1 g) to mass production level (100 kg), gold particle agglomeration is significant. Optimization of dispersant necessary

---

## 4.4 Case Study 4: Electrical Property Control of Graphene

### Background and Challenges

Graphene is a single-layer sheet with carbon atoms arranged in a hexagonal honeycomb lattice. In 2004, Andre Geim and Konstantin Novoselov succeeded in isolation using the "mechanical exfoliation method" with Scotch tape and won the Nobel Prize in Physics in 2010.

**Amazing Properties of Graphene**:
- **Electron Mobility**: 200,000 cm²/V·s (room temperature, over 100 times that of silicon)
- **Thermal Conductivity**: 5,000 W/m·K (over 10 times that of copper)
- **Mechanical Strength**: Tensile strength 130 GPa (100 times that of steel)
- **Light Transmittance**: 97.7% (visible light)

**Expectations for Semiconductor Device Applications**:
- High-speed transistors (GHz-THz operation)
- Flexible electronics
- Transparent conductive films (touch panels)

**Fatal Problem: Zero Band Gap**:
Graphene is a zero-gap semiconductor (band contact at Dirac point), so transistor ON/OFF switching cannot be performed.

**ON/OFF Ratio Problem**:
- Silicon transistor: ON/OFF ratio 10⁶-10⁸
- Graphene transistor: ON/OFF ratio 10-100 (not practical)

**Solution**: Band gap opening by graphene nanoribbons (GNR)

### Project Overview

**Goals**:
- Achieve ON/OFF ratio 10⁴ through band gap opening
- Electron mobility >1,000 cm²/V·s (exceeding silicon)
- Operating speed: 5 times that of silicon transistor

**Duration**: 4 years (2019-2023)

**Team Composition**:
- Semiconductor manufacturer research institute: 10 members
- University physics department (GNR synthesis/evaluation): 6 members
- University information science department (computational science, machine learning): 4 members

### Applied Nanomaterial Technologies

#### 1. What is Graphene Nanoribbon (GNR)

When graphene is processed into narrow ribbons less than 10 nm wide, a band gap opens due to quantum confinement effects.

**Width Dependence of Band Gap (Theoretical Formula)**:
$$
E_g \approx \frac{0.7 \text{ eV·nm}}{W}
$$

Where $W$ is the width of GNR (nm).

| GNR Width | Band Gap | Application |
|-----------|----------|-------------|
| 1 nm | 0.7 eV | Short wavelength photodetector |
| 5 nm | 0.14 eV | High-speed transistor |
| 10 nm | 0.07 eV | THz detector |
| 20 nm | 0.035 eV | Close to zero gap |

**Importance of Edge Structure**:
GNR edge structures come in two types with significantly different electrical properties:

1. **Armchair**: Changes between semiconductor/metal depending on width
2. **Zigzag**: Always metallic (shows magnetism)

Armchair type is essential for transistor applications.

#### 2. Bottom-Up Synthesis Method

**Problems with Conventional Method (Top-Down)**:
- Mechanical exfoliation: Width uncontrollable, edge structure random
- Lithography + etching: Rough edges, many defects

**Bottom-Up Synthesis**: Chemically construct GNR from molecular precursors

**Process**:
1. Molecular precursor design (e.g., 10,10'-dibromo-9,9'-bianthryl)
2. Deposit on gold (Au) substrate (ultra-high vacuum, 10⁻⁹ Torr)
3. Heat substrate (200°C) → Debromination, radical generation
4. Further heating (400°C) → Polymerization, cyclization reaction
5. GNR formation (width: determined by precursor molecule width)

**Advantages**:
- Control width and edge structure with atomic-level precision
- Edges are perfectly aligned as armchair type
- Extremely few defects

**Challenges**:
- Can only be synthesized on gold substrate (transfer to device substrate necessary)
- Slow synthesis speed (several hours for substrate area 1 cm²)

#### 3. Edge Structure Control

**Molecular Precursor Design Example**:

**Width 5 nm Armchair-type GNR**:
```
Precursor molecule:
       Br            Br
        |            |
  [Anthracene]--[Anthracene]
        |            |
       Br            Br

After polymerization:
  [Anthracene]--[Anthracene]--[Anthracene]--...
  (Width 5 nm, armchair-type edge)
```

**Synthesis Condition Optimization**:
- Precursor deposition amount: 1-10 ML (monolayer)
- Polymerization temperature: 180-220°C
- Cyclization temperature: 350-450°C
- Reaction time: 30-120 minutes

**Edge Structure Confirmation by Scanning Tunneling Microscopy (STM)**:
- Direct observation of GNR edge structure with atomic resolution
- Confirmed to be armchair type

### Machine Learning Methods Used

#### Data Collection

**Input Variables (8 dimensions)**:
1. GNR width: 1-15 nm (STM measurement)
2. Edge structure: Armchair/Zigzag (categorical variable)
3. Defect density: 0-5% (STM image analysis)
4. Length: 10-500 nm
5. Substrate type: Au, SiO₂, h-BN (hexagonal boron nitride)
6. Doping: n-type/p-type/none
7. Doping concentration: 0-10¹³ cm⁻²
8. Measurement temperature: 4-300 K

**Output Variables**:
- Band gap (eV, optical absorption spectroscopy measurement)
- Electron mobility (cm²/V·s, field-effect transistor measurement)
- ON/OFF ratio (transistor characteristic measurement)

**Experimental Data**: 420 samples + DFT calculation data 2,000 samples

#### Utilization of DFT Calculation Data

Since experimental data alone was insufficient, data was supplemented with first-principles calculation (Density Functional Theory, DFT).

**Calculation Conditions**:
- Software: VASP (Vienna Ab initio Simulation Package)
- Exchange-correlation functional: PBE (Perdew-Burke-Ernzerhof)
- Cutoff energy: 500 eV
- k-point mesh: 1 × 1 × 21 (high density in ribbon longitudinal direction)

**Calculation Target**:
- Armchair-type GNR with width 1-15 nm
- Zigzag-type GNR with width 1-15 nm
- GNR with defects (vacancies, edge defects)

**Validation of Calculation Results**:
- Confirmed agreement between experimental data (420 samples) and DFT calculations
- Band gap RMSE: 0.05 eV (good agreement)

#### Model Selection: Support Vector Regression (SVR)

**Reasons**:
- Strong with high-dimensional data
- Effective even with small number of experimental data
- Physically reasonable interpolation (less prone to overfitting)

**Kernel Function**: RBF (radial basis function) kernel

**Hyperparameter Tuning**:
```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.05, 0.1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Optimal parameters: C=10, gamma=0.01, epsilon=0.05
```

**Model Performance**:
| Output Variable | Training R² | Test R² | RMSE |
|-----------------|-------------|---------|------|
| Band Gap | 0.96 | 0.92 | 0.04 eV |
| Electron Mobility | 0.89 | 0.84 | 180 cm²/V·s |
| ON/OFF Ratio | 0.85 | 0.79 | 0.35 (log₁₀ scale) |

#### Optimization

**Objective**: Maximize ON/OFF ratio, maximize electron mobility (multi-objective optimization)

**Method**: Pareto optimal solution search (NSGA-II)

**Optimal Conditions**:
| Parameter | Optimal Value |
|-----------|---------------|
| GNR Width | 5.2 nm |
| Edge Structure | Armchair type |
| Defect Density | <0.5% |
| Substrate | h-BN |
| Doping | None |

**Predicted Performance**:
- Band gap: 0.38 eV
- ON/OFF ratio: 1.5 × 10⁴
- Electron mobility: 3,200 cm²/V·s

### Results and Impact

#### Performance of Optimal GNR Transistor

**Measured Values (Width 5.2 nm, Armchair-type GNR)**:
| Property | Measured Value | Silicon (Comparison) |
|---------|----------------|---------------------|
| Band Gap | 0.40 eV | 1.12 eV |
| ON/OFF Ratio | 1.2 × 10⁴ | 10⁶-10⁸ |
| Electron Mobility | 3,000 cm²/V·s | 1,400 cm²/V·s |
| Operating Frequency | 100 GHz | 20 GHz |
| Power Consumption | 0.4 mW | 1.0 mW |

**Achievements**:
- ON/OFF ratio: Achieved target 10⁴
- Electron mobility: Over twice that of silicon
- Operating speed: 5 times that of silicon

**Challenges**:
- ON/OFF ratio is 2 orders of magnitude lower than silicon (insufficient for digital circuits)
- Application: Limited to high-frequency analog circuits (RF communication, THz detectors)

#### Cost and Mass Production

**Manufacturing Cost (per wafer)**:
- Molecular precursor synthesis: $500
- Ultra-high vacuum deposition equipment: $2M (equipment depreciation)
- Substrate (Au/h-BN): $200
- Process time: 8 hours/wafer

**Mass Production Challenges**:
- Slow synthesis speed (1/100 of silicon)
- Complex transfer process from gold substrate to h-BN substrate
- Yield: 60% (silicon: 95%)

#### Application Development

**2024**: Samsung announced prototype RF transistor for 5G communication

**Performance**:
- Operating frequency: 100 GHz
- Noise figure: 2.8 dB (silicon: 4.5 dB)
- Power consumption: 40% reduction

**Future Applications**:
- 6G communication (THz band)
- THz imaging (security scan, medical imaging)
- High-speed ADC (analog-to-digital converter)

### Lessons Learned

1. **Edge Structure is Everything**: Armchair and zigzag types have completely different properties. Atomic-level precision control essential
2. **Optimal Width is 5-10 nm**: Achieved practical band gap (0.4 eV) and ON/OFF ratio (10⁴) at width 5 nm
3. **Substrate Effect**: Mobility decreases to 1/10 on SiO₂ substrate due to surface charge traps. h-BN substrate essential
4. **Machine Learning + DFT**: When experimental data is limited, supplementing with DFT calculation data improves prediction accuracy
5. **Mass Production Process is the Biggest Challenge**: Improvement of synthesis speed, transfer process, and yield is key to practical application

---

## 4.5 Case Study 5: Design of Nanomedicine (Drug Delivery)

### Background and Challenges

Anticancer drugs are powerful medications that kill cancer cells but also show toxicity to normal cells, causing serious side effects (hair loss, nausea, immunosuppression).

**Problems with Conventional Anticancer Drugs**:
- Distributed uniformly throughout body → accumulation rate in cancer tissue <5%
- Large damage to normal cells
- High doses cannot be administered → limited therapeutic effect

**Concept of Nanomedicine**:
Encapsulating anticancer drugs in nanoparticles and selectively delivering them to cancer tissue improves therapeutic effect and reduces side effects.

**EPR Effect (Enhanced Permeability and Retention effect)**:
Blood vessels in cancer tissue have higher permeability than normal tissue (gaps in vessel walls), allowing nanoparticles of 100-200 nm to leak out and accumulate.

**Technical Challenges**:
1. **Size Control**: Around 100 nm is optimal (too small causes renal excretion, too large causes hepatic capture)
2. **Blood Circulation Time**: Must circulate for long time to reach cancer tissue
3. **Drug Release**: Mechanism needed to release drug after reaching cancer tissue
4. **Targeting**: EPR effect alone is insufficient. Target receptors on cancer cell surface

### Project Overview

**Goals**:
- Cancer tissue accumulation rate 20% or higher (5 times conventional)
- 30% reduction in side effects (Grade 3 or higher)
- Tumor shrinkage rate 60% or higher (conventional: 40%)

**Duration**: 5 years (2018-2023, including Phase I/II clinical trials)

**Team Composition**:
- Pharmaceutical company research institute: 15 members
- University medical school (clinical trials): 8 members
- University school of pharmacy (DDS design): 6 members
- Data scientists (machine learning): 2 members

### Applied Nanomaterial Technologies

#### 1. PEG-Modified Liposomes

Liposomes are spherical nanoparticles formed by phospholipid bilayers. Can encapsulate water-soluble drugs inside and lipid-soluble drugs in membrane.

**Liposome Structure**:
```
[Outside]
PEG chain (5-10 kDa)
  |
Phospholipid bilayer (DSPC, cholesterol)
  |
[Inside]
Aqueous phase (anticancer drug doxorubicin encapsulated)
```

**PEG Modification Effect**:
- Blood circulation time: 6 hours → 48 hours
- Mechanism: PEG chains exhibit "stealth effect" and avoid capture by immune system (macrophages)

**Optimal PEG Chain Length**:
| PEG Molecular Weight | Blood Half-life | Cancer Tissue Accumulation |
|---------------------|-----------------|----------------------------|
| 2 kDa | 8 hours | 8% |
| 5 kDa | 24 hours | 18% |
| 10 kDa | 48 hours | 22% |
| 20 kDa | 60 hours | 19% (reverse effect due to decreased renal excretion) |

**Optimal Value**: PEG 10 kDa

#### 2. pH-Responsive Drug Release

Since cancer tissue has lower pH than normal tissue (pH 6.5-6.8 vs 7.4), drugs are bound with pH-responsive linkers.

**pH-Responsive Linker: Hydrazone Bond**

```
Drug-NH-NH-CO-Phospholipid

pH 7.4 (blood): Stable (hydrolysis rate <5%/day)
pH 6.5 (cancer tissue): Hydrolysis (half-life 2 hours)
```

**Drug Release Profile (in vitro)**:
| Time | Release Rate at pH 7.4 | Release Rate at pH 6.5 |
|------|------------------------|------------------------|
| 1 hour | 2% | 15% |
| 6 hours | 8% | 58% |
| 24 hours | 15% | 92% |

#### 3. Folate Receptor Targeting

In addition to EPR effect, target folate receptor (FR) highly expressed on cancer cell surface.

**Folate-Modified Liposomes**:
- Folate bound to PEG chain tip
- Selectively binds to cancer cells expressing FR
- Taken into cells by endocytosis

**Targeting Effect (in vitro)**:
| Liposome Type | Cancer Cell Uptake | Normal Cell Uptake |
|---------------|-------------------|-------------------|
| PEG modification only | 1.0 (standard) | 1.0 (standard) |
| Folate modification | 4.2 | 1.1 |

Folate modification improved uptake by cancer cells 4-fold while minimizing impact on normal cells.

#### 4. Optimal Formulation Design

**Formulation Parameters**:
1. Liposome size: 80-150 nm
2. PEG chain length: 2-20 kDa
3. Lipid composition: DSPC/cholesterol ratio
4. Folate modification rate: 0-10 mol%
5. Drug encapsulation amount: 5-20 wt%

### Machine Learning Methods Used

#### Data Collection

**Input Variables (12 dimensions)**:
1. Liposome size: 80-150 nm (dynamic light scattering measurement)
2. Size distribution (PDI, Polydispersity Index): 0.05-0.30
3. PEG chain length: 2-20 kDa
4. PEG density: 3-15 mol%
5. Lipid composition (DSPC/cholesterol ratio): 2:1-10:1
6. Folate modification rate: 0-10 mol%
7. Doxorubicin encapsulation amount: 5-20 wt%
8. pH: 6.0-7.4
9. Temperature: 25-40°C
10. Buffer ionic strength: 0.01-0.3 M
11. Serum protein concentration: 0-100% (stability in presence of serum)
12. Time: 0-72 hours

**Output Variables**:
- Drug release rate (%/h)
- Blood circulation time (half-life, h)
- Cancer tissue accumulation rate (%ID/g, % injected dose per gram tissue)
- Cytotoxicity (IC₅₀, μM)

**Experimental Data**:
- In vitro experiments: 200 samples
- Animal experiments (mice): 80 samples

#### Model Selection: Random Forest

**Reasons**:
- Can capture nonlinear relationships
- Can evaluate feature importance
- Less prone to overfitting even with small data

**Hyperparameter Tuning**:
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf.fit(X_train, y_train)
```

**Model Performance**:
| Output Variable | Training R² | Test R² | RMSE |
|-----------------|-------------|---------|------|
| Drug Release Rate (pH 6.5) | 0.91 | 0.85 | 2.1 %/h |
| Blood Half-life | 0.87 | 0.80 | 5.2 h |
| Cancer Tissue Accumulation | 0.84 | 0.76 | 2.8 %ID/g |
| Cytotoxicity (IC₅₀) | 0.89 | 0.82 | 0.15 μM (log₁₀ scale) |

**Feature Importance (SHAP Values)**:
1. Liposome size (0.28)
2. PEG chain length (0.22)
3. Folate modification rate (0.18)
4. pH (0.15, for drug release rate)
5. Others (0.17)

#### Optimization

**Multi-objective Optimization Problem**:
- Objective 1: Maximize cancer tissue accumulation
- Objective 2: Minimize normal tissue accumulation
- Objective 3: Optimize drug release rate (fast at pH 6.5, slow at pH 7.4)

**Method**: Bayesian optimization

**Optimal Formulation**:
| Parameter | Optimal Value |
|-----------|---------------|
| Liposome Size | 105 nm |
| PEG Chain Length | 10 kDa |
| PEG Density | 8 mol% |
| DSPC/Cholesterol Ratio | 5:1 |
| Folate Modification Rate | 5 mol% |
| Doxorubicin Encapsulation | 12 wt% |

**Predicted Performance (Mouse Model)**:
- Blood half-life: 44 hours
- Cancer tissue accumulation: 23 %ID/g
- Normal tissue accumulation: 3 %ID/g (liver), 2 %ID/g (kidney)

### Results and Impact

#### Preclinical Study (Mouse Model)

**Experimental Conditions**:
- Mice: Nude mice (immunodeficient)
- Cancer model: HeLa cells (cervical cancer) subcutaneous transplantation
- Dose: Doxorubicin equivalent 5 mg/kg
- Control group: Free doxorubicin

**Results**:
| Indicator | Free Doxorubicin | Folate-Modified Liposome |
|-----------|------------------|-------------------------|
| Tumor Shrinkage (21 days) | 38% | 72% |
| Survival Rate (60 days) | 20% | 80% |
| Weight Loss (side effect indicator) | 15% | 5% |
| Cancer Tissue Accumulation | 4.2 %ID/g | 21.5 %ID/g |

#### Phase I Clinical Trial (Safety Evaluation)

**Subjects**: 18 patients with solid tumors (breast cancer, lung cancer, colon cancer)

**Dosage**: Doxorubicin equivalent 20-70 mg/m² (per body surface area)

**Results**:
| Dosage (mg/m²) | Grade 3+ Side Effects | Objective Response Rate |
|----------------|-----------------------|------------------------|
| 20 | 0/3 patients | 0% |
| 35 | 0/3 patients | 33% (1/3 patients) |
| 50 | 1/6 patients (neutropenia) | 50% (3/6 patients) |
| 70 | 3/6 patients (neutropenia, cardiotoxicity) | 50% (3/6 patients) |

**Maximum Tolerated Dose (MTD)**: 50 mg/m² (conventional doxorubicin: 60-75 mg/m²)

**Pharmacokinetics (PK)**:
- Blood half-life: 48 hours (free: 6 hours)
- Cancer tissue accumulation: Estimated 18-22 %ID/g (PET image analysis)

#### Phase II Clinical Trial (Efficacy Evaluation)

**Subjects**: 120 patients with HER2-negative breast cancer (metastatic or postoperative recurrence)

**Trial Design**:
- Group A (60 patients): Folate-modified liposome (50 mg/m², every 3 weeks)
- Group B (60 patients): Conventional chemotherapy (paclitaxel + carboplatin)

**Primary Endpoint**: Objective Response Rate (ORR)

**Results (12 months follow-up)**:
| Indicator | Group A (Liposome) | Group B (Conventional) | p-value |
|-----------|-------------------|----------------------|---------|
| Objective Response Rate (ORR) | 65% | 42% | 0.008 |
| Complete Response Rate (CR) | 18% | 8% | 0.042 |
| Progression-Free Survival (PFS) | 9.2 months | 6.5 months | 0.012 |
| Grade 3+ Side Effects | 25% | 40% | 0.031 |

**Side Effect Breakdown**:
| Side Effect | Group A | Group B |
|------------|---------|---------|
| Neutropenia | 15% | 30% |
| Hair Loss | 5% | 35% (significant difference) |
| Cardiotoxicity | 3% | 8% |
| Peripheral Neuropathy | 2% | 12% (from paclitaxel) |

**Phase II Trial Conclusion**:
Folate-modified liposomes showed higher efficacy and fewer side effects compared to conventional treatment. Progression to Phase III trial was approved.

#### Cost and Drug Price

**Manufacturing Cost (per dose)**:
- Doxorubicin active ingredient: $50
- Liposome materials (lipid, PEG, folate): $120
- Manufacturing process (sterilization, quality control): $80
- **Total: $250** (conventional doxorubicin: $60)

**Expected Drug Price**: $3,000/dose (patient copay after insurance: $300)

**Cost-Effectiveness**:
- Additional cost for PFS extension (2.7 months): $10,000/QALY (quality-adjusted life year)
- Below threshold $50,000/QALY → Cost-effective

### Lessons Learned

1. **Individual Variability in EPR Effect**: Phase II showed large variation in accumulation rate among patients (10-30 %ID/g). Depends on cancer type, degree of angiogenesis, inflammatory state
2. **Importance of pH Responsiveness**: pH-responsive linkers significantly reduced side effects. Suppresses premature release in blood
3. **Synergistic Effect of Targeting**: EPR effect + folate targeting doubled accumulation rate
4. **Development Acceleration by Machine Learning**: Formulation optimization typically takes 3-5 years, completed in 18 months with machine learning
5. **Balance of Safety and Efficacy**: Dosage 50 mg/m² is optimal. At 70 mg/m², side effect increase outweighs efficacy improvement
6. **Long-term Toxicity Concerns**: PEG accumulation in body (accelerated blood clearance, ABC phenomenon) occurs with repeated administration. Caution needed after 4-6 cycles

---

## 4.6 Future Outlook of Nanomaterial Research

### Major Trends

#### 1. Acceleration of AI-Driven Materials Design

**Full-Scale Materials Informatics (MI)**:
- Experimental data + computational data + machine learning + automated experiment robots
- Material development period: 10-20 years → 2-5 years reduction

**Example**: IBM "RoboRXN"
- AI predicts and optimizes organic synthesis reactions
- Robots automatically execute synthesis
- Discovers synthesis route for target compound in hours

**Application to Nanomaterials**:
- Specify emission wavelength of quantum dots, automatically proposes synthesis conditions
- Input target strength of CNT composite, searches for optimal formulation

**Technical Elements**:
- **Bayesian Optimization**: Search for optimal solution with few experiments
- **Transfer Learning**: Utilize data from similar materials
- **Active Learning**: AI proposes experimental conditions with most information

#### 2. Sustainable Nanomaterials

**Green Synthesis Methods**:
Conventional nanomaterial synthesis required organic solvents, high temperatures, and long reaction times. New methods to reduce environmental impact:

**Plant-Derived Reducing Agents**:
- Gold nanoparticles: Reduced with green tea extract (catechin)
- Silver nanoparticles: Reduced with citric acid, ascorbic acid
- Advantages: Non-toxic, low cost, room temperature reaction

**Microbial Utilization**:
- Bacteria (Bacillus) reduce gold ions → Form gold nanoparticles inside cells
- Algae synthesize iron oxide nanoparticles
- Advantages: Complete bioprocess, zero CO₂ emissions

**Biodegradable Nanomaterials**:
- Medical nanoparticles: Polylactic acid (PLA), chitosan-based
- Naturally degrade in body → No long-term toxicity concerns

**Recyclable Design**:
- Recovery and reuse processes for precious metal nanoparticles
- Chemical recycling of CNT (carbonization → regrowth)

#### 3. Nano-Bio Fusion

**DNA Origami Nanostructures**:
- Design DNA base sequences to create arbitrary 3D shapes through self-assembly
- Applications: Drug delivery, molecular sensors, nanorobots

**Example**: Church Lab (Harvard University)
- Created "nano box" with DNA origami
- Mechanism to open/close lid (responds to pH, temperature, specific molecules)
- Encapsulate drug and open in front of cancer cells

**Protein-Nanoparticle Hybrid**:
- Immobilize enzyme on gold nanoparticles → Biocatalyst
- Bind antibody to quantum dots → Cancer cell imaging

**Nanorobots**:
- Molecular machines constructed from DNA or proteins
- Move in body and release drugs at lesion sites
- Current status: Mouse experiment level
- 2030s: Possibility of clinical trial initiation

#### 4. Quantum Nanomaterials

**Quantum Computing Materials**:
- Superconducting qubits: Josephson junction (nanoscale Al/AlOx/Al)
- Topological qubits: Majorana fermions (nanowire + superconductor)
- Challenge: Quantum decoherence (caused by defects in nanomaterials)

**Topological Insulators**:
- Insulator inside, metal on surface
- Applied to spintronics (low power consumption devices using electron spin)
- Materials: Bi₂Se₃, Bi₂Te₃ (nano thin films)

**2D Material Heterostructures**:
- Stack graphene / h-BN / MoS₂ etc.
- New electronic properties emerge from layer combinations
- Applications: Flexible electronics, quantum optics

**Example**: "Magic Angle" Graphene (MIT, 2018)
- Stack two graphene sheets rotated 1.1°
- Superconductivity, Mott insulator properties emerge
- Selected as "2018 Breakthrough" by Nature magazine

---

## 4.7 Career Paths: Working in the Nanomaterials Field

### Academia (Universities/Research Institutes)

#### Job Types

**Postdoctoral Researcher**:
- 2-5 year fixed-term research position after PhD
- Paper writing, research proposal preparation, student guidance assistance

**Assistant Professor**:
- Fixed-term (5-10 years) or tenure track
- Have independent laboratory (or belong to associate professor's laboratory)
- Teaching, student guidance, research funding acquisition

**Associate Professor**:
- Lead independent laboratory
- Teaching, student guidance, research funding acquisition, academic society management

**Professor**:
- Laboratory leadership, department management
- National project leadership, policy recommendations

#### Salary (Japan)

| Position | Salary Range | Average Salary |
|----------|-------------|----------------|
| Postdoc | ¥3.5-5M | ¥4.2M |
| Assistant Professor | ¥5-7M | ¥6M |
| Associate Professor | ¥7-10M | ¥8.5M |
| Professor | ¥10-15M | ¥12M |

**US Salaries (Reference)**:
- Postdoc: $50,000-70,000
- Assistant Professor: $70,000-90,000
- Associate Professor: $90,000-120,000
- Professor: $120,000-200,000+

#### Advantages

1. **Research Freedom**: Can research based on own interests
2. **Rewarding Student Guidance**: Train next generation of researchers
3. **International Conferences**: Interact with researchers worldwide, obtain latest information
4. **Social Status**: Respected as expert
5. **Work-Life Balance**: Large discretion over time (but long hours also common)

#### Disadvantages

1. **Competitive Funding Pressure**: Apply annually to KAKENHI, JST, NEDO, etc.
2. **Many Fixed-Term Positions**: About 70% of assistant professors are fixed-term (Japan)
3. **Salary Level**: Lower than industry (¥6-8M 10 years after PhD)
4. **Publication Pressure**: "Publish or Perish" (cannot survive without papers)
5. **Teaching/Administrative Work**: Much work besides research (lecture preparation, committees, administrative procedures)

#### Career Path Example

**Typical Promotion Path**:
```
PhD completion (27-30 years old)
  ↓
Postdoc (2-5 years, 29-35 years old)
  ↓
Assistant Professor (5-10 years, 35-45 years old)
  ↓
Associate Professor (10-15 years, 45-60 years old)
  ↓
Professor (60-65 retirement)
```

**Keys to Success**:
- Publish papers in high IF (Impact Factor) journals during PhD/postdoc (Nature, Science, JACS, Nano Letters, etc.)
- Build international collaborative research network
- Establish original research theme
- Research funding acquisition ability

---

### Industry (Corporations)

#### Job Types

**R&D Engineer**:
- New material exploration/development
- Application research for products
- Patent filing, paper publication (policy varies by company)

**Product Development Engineer**:
- Commercialization of existing technology
- Prototype fabrication, performance evaluation
- Mass production process design

**Process Engineer**:
- Manufacturing process optimization
- Quality control, cost reduction
- Cooperation with factories

**Data Scientist (Materials Informatics)**:
- Experimental data analysis
- Machine learning model development
- AI application to material development

#### Salary (Japan)

| Years of Experience | Salary Range | Average Salary |
|--------------------|-------------|----------------|
| New Graduate (Master's) | ¥4-5.5M | ¥4.8M |
| 3-5 years | ¥5-7M | ¥6M |
| 5-10 years | ¥6.5-9M | ¥7.5M |
| 10-15 years (Management) | ¥10-15M | ¥12M |
| 15+ years (Specialist) | ¥12-18M | ¥14M |

**Salary by Industry (Average)**:
- Chemical: ¥7M
- Electronics: ¥7.5M
- Pharmaceutical: ¥8.5M
- Foreign companies: Japanese companies +20-30%

#### Major Companies

**Chemical**:
- Toray (carbon fiber, CNT composites)
- Asahi Kasei (lithium-ion battery separators)
- Mitsubishi Chemical (functional materials)
- Sumitomo Chemical (organic EL materials)
- JSR (semiconductor materials, nanoparticles)

**Electronics**:
- Sony (quantum dot displays, image sensors)
- Panasonic (battery materials, catalysts)
- Hitachi (nano processing technology)
- Toshiba (nanomaterial analysis)

**Materials**:
- Nippon Carbon (carbon nanotubes)
- JFE Steel (nano steel materials)
- AGC (glass, nano coatings)

**Pharmaceutical**:
- Takeda Pharmaceutical (nanomedicine, DDS)
- Daiichi Sankyo (antibody drugs)
- Chugai Pharmaceutical (biopharmaceuticals)

#### Advantages

1. **Salary Level**: Higher than academia (¥6-7M at age 30 with master's)
2. **Commercialization Impact**: Products you develop are used in society
3. **Team Development**: Cooperate with diverse experts
4. **Facilities/Funding**: Can use expensive equipment
5. **Stability**: Lifetime employment (at large companies)

#### Disadvantages

1. **Research Freedom**: Follow company business strategy
2. **Short-Term Results**: Results required in 2-3 years
3. **Relocation/Reassignment**: Transfers from research to sales/management also occur
4. **Paper Publication**: Often cannot publish due to company secrets
5. **Long Hours**: Overtime common before project deadlines

#### Career Path Example

**Management Track**:
```
New graduate hire (Master's, 24 years old)
  ↓
Researcher (5-8 years, 29-32 years old)
  ↓
Senior Researcher (3-5 years, 32-37 years old)
  ↓
Section Manager (Group Leader) (5-8 years, 37-45 years old)
  ↓
Department Manager (Research Director) (10-15 years, 45-60 years old)
```

**Specialist Track**:
```
New graduate hire (Master's or PhD, 24-27 years old)
  ↓
Researcher (5-8 years)
  ↓
Senior Researcher (5-10 years)
  ↓
Senior Researcher (Fellow) (lifetime)
```

**Keys to Success**:
- Patent filing (2-5 per year)
- Product commercialization track record
- Internal and external network
- Project management ability

---

### Startups

#### Job Types

**Co-founder (CTO, Chief Technology Officer)**:
- Technology strategy planning
- R&D leadership
- Presentations to investors

**R&D Leader**:
- Core technology development
- Team building
- Patent strategy

**Product Manager**:
- Product specification formulation
- Customer needs survey
- Go-to-market strategy

#### Salary and Equity

**Salary**:
| Stage | Salary Range | Notes |
|-------|-------------|-------|
| Seed (Just founded) | ¥3-5M | Before funding |
| Series A (Initial funding) | ¥5-8M | After raising hundreds of millions |
| Series B+ | ¥7-12M | After raising 1+ billion |

**Stock Options (Equity Compensation)**:
- Founding members: 5-20%
- Early employees: 0.5-2.0%
- Value at IPO: Company market cap × shareholding ratio
  - Example: Market cap ¥10B, 1% holding → ¥100M (before tax)

#### Advantages

1. **Social Implementation of Innovative Technology**: Create products impossible at large companies
2. **Discretion and Responsibility**: Can make own decisions
3. **Large Returns from IPO**: Possibility of tens to hundreds of millions
4. **Skill Growth**: Learn not only technology but also management, fundraising, marketing
5. **Challenging Environment**: Easy for young generation to thrive

#### Disadvantages

1. **Income Instability**: Risk of unpaid salary if fundraising fails
2. **Long Hours**: 60-80 hours/week is normal
3. **Failure Risk**: 90% of startups fail
4. **Benefits**: Not as comprehensive as large companies
5. **Mental Pressure**: Cash flow, competition, customer acquisition

#### Success Stories

**Japanese Nanomaterial Startups**:

1. **Zeon Nano Technology** (CNT mass production technology)
   - Founded: 2017
   - Funding raised: Cumulative ¥5B
   - Technology: Large-scale CNT production by super-growth method
   - Applications: EV batteries, composite materials

2. **Quantum Solutions** (Quantum dot displays)
   - Founded: 2019
   - Funding raised: Cumulative ¥3B
   - Technology: Cadmium-free quantum dots
   - Customers: Major display manufacturers

3. **NanoCare Systems** (Nano DDS)
   - Founded: 2020
   - Funding raised: Cumulative ¥2B
   - Technology: mRNA lipid nanoparticles
   - Pipeline: Cancer vaccine (Phase I clinical trial)

#### Career Path Example

**Success Pattern**:
```
PhD + Postdoc (Hone technology)
  ↓
3-5 years at large company (Industry understanding, network building)
  ↓
Startup founding (30-35 years old)
  ↓
Seed funding (¥10-50M)
  ↓
Series A (¥300M-1B)
  ↓
Product commercialization/revenue expansion
  ↓
IPO or M&A (5-10 years after founding)
```

**Failure Pattern**:
```
Founding
  ↓
Fundraising failure (Technology excellent but business model unclear)
  ↓
Continue development with own funds (3-5 years)
  ↓
Funding depletion, closure
  ↓
Re-employment at large company
```

---

### Required Skill Set

#### Technical Skills

1. **Nanomaterial Synthesis/Characterization**:
   - Chemical synthesis technology (solution method, vapor phase method, solid phase method)
   - Surface/interface control
   - Analytical techniques (TEM, XRD, XPS, Raman, AFM, etc.)

2. **Data Analysis**:
   - Python (NumPy, Pandas, Matplotlib, Scikit-learn)
   - R (statistical analysis)
   - MATLAB (engineering calculations)
   - Origin (graph creation)

3. **Machine Learning/Materials Informatics**:
   - Supervised learning (regression, classification)
   - Bayesian optimization
   - Neural networks
   - Feature engineering

4. **First-Principles Calculation**:
   - VASP (density functional theory)
   - Gaussian (quantum chemistry calculation)
   - LAMMPS (molecular dynamics)

5. **Programming**:
   - Python (essential)
   - C/C++ (high-speed calculation)
   - Julia (scientific computing)
   - Git (version control)

#### Business Skills

1. **Project Management**:
   - Schedule management (Gantt charts)
   - Risk management
   - Team communication

2. **Intellectual Property**:
   - Patent search (PatentScope, J-PlatPat)
   - How to read/write patent specifications
   - Prior art search

3. **Presentation**:
   - Conference presentations (PowerPoint, Keynote)
   - Storytelling
   - Figure/table design

4. **English**:
   - Paper writing (TOEIC 800+ recommended)
   - Conference presentations (oral/poster)
   - Discussion ability

#### Recommended Certifications

**Degree is Most Important**:
- Master's: Minimum requirement for corporate research positions
- PhD: Essential in academia, startups

**Optional Certifications**:
- G Test (JDLA Deep Learning for General): AI fundamental knowledge
- E Test (JDLA Deep Learning for Engineer): AI implementation ability
- Statistics Certification Level 2+: Data analysis fundamentals
- Hazardous Materials Handler (Class A): Handling of chemicals

---

## Summary

In this chapter, we learned about actual success stories of nanomaterial research and careers in this field.

### Key Points

#### 1. Keys to Commercialization

**Power of Machine Learning**:
- CNT composite: 1/3 experiments, 1/2 development time
- Quantum dots: Achieved ±3% size uniformity, conventional ±15%
- Gold catalyst: 70% reduction in experiments
- Graphene: Improved prediction accuracy through combination with DFT calculation
- Nanomedicine: Formulation optimization period 3-5 years → 18 months

**Scale-up Challenges**:
- Transition from laboratory level (g) to mass production level (kg) is the biggest barrier
- Fluid dynamics, heat transport, mass diffusion change
- Re-optimization essential

**Balance of Cost, Performance, and Safety**:
- CNT: Performance +51%, cost +38%
- Gold catalyst: 1/2 cost of platinum, 90% activity
- Nanomedicine: 1.5× efficacy, 15% reduction in side effects

#### 2. Common Points in Success Stories

1. **Clear Goal Setting**: Numerical targets (strength +50%, ON/OFF ratio 10⁴, etc.)
2. **Interdisciplinary Team**: Materials science + machine learning + application field experts
3. **Sufficient Development Period**: 2-5 years (difficult in short period)
4. **Systematic Data Accumulation**: 200-700 samples (necessary for machine learning)
5. **Fusion with Physical Models**: Brus equation, DFT calculation + machine learning

#### 3. Career Options

| Item | Academia | Industry | Startup |
|------|----------|----------|---------|
| **Salary (30 years old)** | ¥5-6M | ¥6-7M | ¥4-6M + stock options |
| **Research Freedom** | High | Moderate | Very high (but funding constraints) |
| **Commercialization** | Indirect | Direct | Direct |
| **Stability** | Low (fixed-term) | High | Very low |
| **Returns** | Limited | Stable | High risk, high return |

#### 4. Required Skills

**Technical (Essential)**:
- Nanomaterial synthesis/evaluation experimental skills
- Python (data analysis, machine learning)
- Paper writing/English presentation

**Technical (Recommended)**:
- First-principles calculation (VASP, Gaussian)
- Statistical/machine learning theoretical understanding
- C/C++ (high-speed calculation)

**Business**:
- Project management
- Patent search/specification writing
- Presentation

### Next Steps

Through this series "Nanomaterials Introduction: From Size Effects to AI Design," we learned:

**Chapter 1**: Nanomaterial fundamentals, size effects, property changes
**Chapter 2**: Machine learning methods, experimental data utilization, feature design
**Chapter 3**: Practical exercises with Python, data analysis, optimization
**Chapter 4**: Real-world applications, success stories, career paths

#### To Deepen Learning Further

**1. Laboratory Practice**:
- University internships (summer intensive courses)
- Corporate research facility tours
- Open lab programs

**2. Online Courses**:
- Coursera: "Nanotechnology: The Basics" (Rice University)
- edX: "Applications of Nanotechnology" (MIT)
- Udemy: "Materials Informatics with Python"

**3. Conference Participation**:
- Chemical Society of Japan (Spring/Autumn meetings)
- Japan Society of Applied Physics
- Materials Research Society (MRS)
- American Chemical Society (ACS) Nano Division

**4. Paper Reading**:
- Nature Nanotechnology (IF: 40+)
- ACS Nano (IF: 18+)
- Nano Letters (IF: 12+)
- Advanced Materials (IF: 30+)
- Journal of Physical Chemistry C (IF: 4+)

**5. Books**:
- "Introduction to Nanotechnology" (Charles P. Poole Jr.)
- "Materials Informatics" (edited by Krishnan Rajan)
- "Handbook of Nanomaterials" (Springer)

#### Message to You

Nanomaterials are an attractive field directly connected to solving social issues such as energy, environment, medicine, and information communication. With the fusion of AI and machine learning, materials development is entering a new era.

**Three Pieces of Advice for Success in This Field**:

1. **Maintain Curiosity**: Always ask "why?" and strive to understand the essence
2. **Learn Interdisciplinarily**: Cross boundaries of chemistry, physics, information science, and biology
3. **Get Hands-On**: Hone skills in both experimentation and programming

We look forward to your success. Good luck!

---

## References

1. **De Volder, M. F. et al.** (2013). Carbon nanotubes: present and future commercial applications. *Science*, 339(6119), 535-539. [DOI: 10.1126/science.1222453](https://doi.org/10.1126/science.1222453)

2. **Carey, G. H. et al.** (2015). Colloidal quantum dot solar cells. *Chemical Reviews*, 115(23), 12732-12763. [DOI: 10.1021/acs.chemrev.5b00063](https://doi.org/10.1021/acs.chemrev.5b00063)

3. **Haruta, M.** (2004). Gold as a novel catalyst in the 21st century: Preparation, working mechanism and applications. *Gold Bulletin*, 37(1-2), 27-36. [DOI: 10.1007/BF03215514](https://doi.org/10.1007/BF03215514)

4. **Shi, Z. et al.** (2016). Vapor-liquid-solid growth of large-area multilayer hexagonal boron nitride on dielectric substrates. *Nature Communications*, 7, 10426. [DOI: 10.1038/ncomms10426](https://doi.org/10.1038/ncomms10426)

5. **Peer, D. et al.** (2007). Nanocarriers as an emerging platform for cancer therapy. *Nature Nanotechnology*, 2(12), 751-760. [DOI: 10.1038/nnano.2007.387](https://doi.org/10.1038/nnano.2007.387)

6. **Butler, K. T. et al.** (2018). Machine learning for molecular and materials science. *Nature*, 559(7715), 547-555. [DOI: 10.1038/s41586-018-0337-2](https://doi.org/10.1038/s41586-018-0337-2)

7. **Sanchez-Lengeling, B. & Aspuru-Guzik, A.** (2018). Inverse molecular design using machine learning: Generative models for matter engineering. *Science*, 361(6400), 360-365. [DOI: 10.1126/science.aat2663](https://doi.org/10.1126/science.aat2663)

8. **Rothemund, P. W.** (2006). Folding DNA to create nanoscale shapes and patterns. *Nature*, 440(7082), 297-302. [DOI: 10.1038/nature04586](https://doi.org/10.1038/nature04586)

9. **Cao, Y. et al.** (2018). Unconventional superconductivity in magic-angle graphene superlattices. *Nature*, 556(7699), 43-50. [DOI: 10.1038/nature26160](https://doi.org/10.1038/nature26160)

10. **Novoselov, K. S. et al.** (2004). Electric field effect in atomically thin carbon films. *Science*, 306(5696), 666-669. [DOI: 10.1126/science.1102896](https://doi.org/10.1126/science.1102896)

---

[← Previous Chapter: Python Practical Tutorial](chapter3-hands-on.html) | [Return to Series Index →](index.html)
