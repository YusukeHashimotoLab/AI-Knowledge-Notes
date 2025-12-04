---
title: Introduction to QSAR/QSPR - Fundamentals of Property Prediction
chapter_title: Introduction to QSAR/QSPR - Fundamentals of Property Prediction
subtitle: 
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 12
exercises: 4
version: 1.0
created_at: 2025-10-18
---

# Chapter 2: Introduction to QSAR/QSPR - Fundamentals of Property Prediction

This chapter covers the fundamentals of Introduction to QSAR/QSPR, which fundamentals of property prediction. You will learn essential concepts and techniques.

## What You Will Learn in This Chapter

In this chapter, you will learn the basics of molecular descriptor calculation and QSAR/QSPR modeling. The technique of predicting properties from molecular structures is essential for efficient drug discovery and materials development.

### Learning Objectives

  * ✅ Understand the types and usage of 1D/2D/3D molecular descriptors
  * ✅ Calculate comprehensive descriptors using mordred
  * ✅ Build and evaluate QSAR/QSPR models
  * ✅ Understand structure-property relationships through feature selection and interpretation
  * ✅ Apply machine learning to real data such as solubility prediction

* * *

## 2.1 Fundamentals of Molecular Descriptors

**Molecular descriptors** are numerical representations of molecular structures used as input for machine learning models.
    
    
    ```mermaid
    flowchart TD
        A[Molecular Structure] --> B[Descriptor Calculation]
        B --> C[1D Descriptors]
        B --> D[2D Descriptors]
        B --> E[3D Descriptors]
    
        C --> F[Molecular Weight, logP, TPSA]
        D --> G[Fingerprints, Graph Descriptors]
        E --> H[Conformation-Dependent Descriptors]
    
        F --> I[Machine Learning Model]
        G --> I
        H --> I
        I --> J[Property Prediction]
    
        style A fill:#e3f2fd
        style I fill:#4CAF50,color:#fff
        style J fill:#FF9800,color:#fff
    ```

### 2.1.1 1D Descriptors: Molecular-Level Properties

1D descriptors represent basic properties independent of molecular structure.

**Main 1D Descriptors** :

Descriptor | Description | Application  
---|---|---  
**Molecular Weight (MW)** | Mass of the molecule | Membrane permeability prediction  
**logP** | Lipophilicity (water-octanol partition coefficient) | Absorption prediction  
**TPSA** | Topological Polar Surface Area | Blood-brain barrier permeability  
**HBA/HBD** | Number of hydrogen bond acceptors/donors | Lipinski's Rule  
**Rotatable Bonds** | Molecular flexibility | Binding affinity  
  
#### Code Example 1: Calculating Basic 1D Descriptors
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 1: Calculating Basic 1D Descriptors
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    
    # Sample drugs
    drugs = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    
    # Descriptor calculation
    data = []
    for name, smiles in drugs.items():
        mol = Chem.MolFromSmiles(smiles)
        data.append({
            'Name': name,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    

**Sample Output:**
    
    
            Name      MW  LogP  TPSA  HBA  HBD  RotBonds
         Aspirin  180.16  1.19 63.60    4    1         3
       Ibuprofen  206.28  3.50 37.30    2    1         4
     Paracetamol  151.16  0.46 49.33    2    2         1
        Caffeine  194.19 -0.07 58.44    6    0         0
    

### 2.1.2 2D Descriptors: Molecular Fingerprints and Graph Descriptors

2D descriptors reflect the topology (connectivity) of molecules.

#### Morgan Fingerprint (ECFP)

Morgan fingerprints are bit vectors created by hashing the environment around each atom.
    
    
    ```mermaid
    flowchart LR
        A[Center Atom] --> B[Radius 1 Environment]
        B --> C[Radius 2 Environment]
        C --> D[Hashing]
        D --> E[Bit Vector]
    
        style A fill:#e3f2fd
        style E fill:#4CAF50,color:#fff
    ```

#### Code Example 2: Calculating Morgan Fingerprints
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 2: Calculating Morgan Fingerprints
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    
    # Prepare molecules
    smiles_list = [
        "CCO",  # Ethanol
        "CCCO",  # Propanol (similar)
        "c1ccccc1"  # Benzene (different)
    ]
    
    # Calculate Morgan fingerprints (radius 2, 2048 bits)
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )
        fps.append(fp)
    
    # Calculate Tanimoto similarity
    from rdkit import DataStructs
    
    print("Tanimoto Similarity Matrix:")
    for i, fp1 in enumerate(fps):
        similarities = []
        for j, fp2 in enumerate(fps):
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(f"{sim:.3f}")
        print(f"{smiles_list[i]:15s} {' '.join(similarities)}")
    

**Sample Output:**
    
    
    Tanimoto Similarity Matrix:
    CCO             1.000 0.571 0.111
    CCCO            0.571 1.000 0.103
    c1ccccc1        0.111 0.103 1.000
    

**Interpretation** : Ethanol and propanol show high similarity (0.571), while benzene shows low similarity to both.

#### Code Example 3: MACCS Keys (Structural Features)
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 3: MACCS Keys (Structural Features)
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    import numpy as np
    
    # MACCS keys are 166-bit structural features
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    
    # Calculate MACCS keys
    maccs = MACCSkeys.GenMACCSKeys(mol)
    
    # Display features with bits set to 1
    on_bits = [i for i in range(len(maccs)) if maccs[i]]
    print(f"Structural features of Aspirin (ON bits): {len(on_bits)} / 166")
    print(f"Feature indices: {on_bits[:20]}...")  # First 20
    

**Sample Output:**
    
    
    Structural features of Aspirin (ON bits): 38 / 166
    Feature indices: [1, 7, 10, 21, 32, 35, 47, 48, 56, 60, ...]
    

### 2.1.3 3D Descriptors: Conformation-Dependent Descriptors

3D descriptors consider the three-dimensional structure of molecules.

#### Code Example 4: Calculating 3D Descriptors
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: Calculating 3D Descriptors
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D
    import pandas as pd
    
    # Prepare molecule
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Calculate 3D descriptors
    descriptors_3d = {
        'PMI1': Descriptors3D.PMI1(mol),  # Principal Moment of Inertia 1
        'PMI2': Descriptors3D.PMI2(mol),  # Principal Moment of Inertia 2
        'PMI3': Descriptors3D.PMI3(mol),  # Principal Moment of Inertia 3
        'NPR1': Descriptors3D.NPR1(mol),  # Normalized Principal Ratio 1
        'NPR2': Descriptors3D.NPR2(mol),  # Normalized Principal Ratio 2
        'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
        'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
        'Asphericity': Descriptors3D.Asphericity(mol),
        'Eccentricity': Descriptors3D.Eccentricity(mol)
    }
    
    df = pd.DataFrame([descriptors_3d])
    print(df.T)
    

**Sample Output:**
    
    
                                  0
    PMI1                    197.45
    PMI2                    598.32
    PMI3                    712.18
    NPR1                      0.28
    NPR2                      0.84
    RadiusOfGyration          3.42
    InertialShapeFactor       0.18
    Asphericity               0.23
    Eccentricity              0.89
    

### 2.1.4 Comprehensive Descriptor Calculation with mordred

mordred is a library that can calculate over 1,800 types of descriptors at once.

#### Code Example 5: Calculating All Descriptors with mordred
    
    
    # Install mordred
    # pip install mordred
    
    from mordred import Calculator, descriptors
    from rdkit import Chem
    import pandas as pd
    
    # Initialize Calculator (all descriptors)
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Molecule list
    smiles_list = [
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ]
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    
    # Descriptor calculation (may take time)
    df = calc.pandas(mols)
    
    print(f"Number of calculated descriptors: {len(df.columns)}")
    print(f"Number of molecules: {len(df)}")
    print("\nFirst 10 descriptors:")
    print(df.iloc[:, :10])
    
    # Remove NaN and infinity
    df_clean = df.select_dtypes(include=[np.number])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(axis=1, how='any')
    
    print(f"\nNumber of descriptors after cleaning: {len(df_clean.columns)}")
    

**Sample Output:**
    
    
    Number of calculated descriptors: 1826
    Number of molecules: 3
    
    First 10 descriptors:
       ABC    ABCGG  nAcid  nBase  SpAbs_A  SpMax_A  SpDiam_A  ...
    0  3.46    3.82      0      0     2.57     1.29      2.31  ...
    1  16.52  17.88      1      0    13.45     2.34      5.67  ...
    2  15.78  16.45      0      6    12.87     2.12      5.34  ...
    
    Number of descriptors after cleaning: 1654
    

* * *

## 2.2 QSAR/QSPR Modeling

### Definitions

  * **QSAR (Quantitative Structure-Activity Relationship)** : Structure-activity relationship
  * Prediction of biological activity (IC50, EC50, etc.)
  * Selection of candidate compounds in drug discovery

  * **QSPR (Quantitative Structure-Property Relationship)** : Structure-property relationship

  * Prediction of physicochemical properties (solubility, melting point, etc.)
  * Property optimization in materials design

    
    
    ```mermaid
    flowchart TD
        A[Molecular Structure\nSMILES] --> B[Descriptor Calculation]
        B --> C[Feature Matrix\nX]
    
        D[Experimental Values\nActivity/Property] --> E[Target Variable\ny]
    
        C --> F[Machine Learning Model]
        E --> F
    
        F --> G[Training]
        G --> H[Evaluation]
        H --> I{Performance OK?}
        I -->|No| J[Feature Selection/\nModel Change]
        J --> F
        I -->|Yes| K[Prediction for New Molecules]
    
        style A fill:#e3f2fd
        style K fill:#4CAF50,color:#fff
    ```

### 2.2.1 Linear Models

#### Code Example 6: Property Prediction with Ridge Model
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 6: Property Prediction with Ridge Model
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Sample data (in practice, calculated with mordred, etc.)
    # X: descriptor matrix, y: solubility
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1.5) + np.random.randn(n_samples) * 0.5
    
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    # Prediction
    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test = ridge.predict(X_test_scaled)
    
    # Evaluation
    print("Ridge Regression Performance:")
    print(f"Training R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"Test R²: {r2_score(y_test, y_pred_test):.3f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
    
    # Lasso model (introducing sparsity)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    
    # Number of non-zero coefficients (selected features)
    non_zero = np.sum(lasso.coef_ != 0)
    print(f"\nFeatures selected by Lasso: {non_zero} / {n_features}")
    

**Sample Output:**
    
    
    Ridge Regression Performance:
    Training R²: 0.923
    Test R²: 0.891
    Test RMSE: 0.542
    
    Features selected by Lasso: 18 / 50
    

### 2.2.2 Nonlinear Models

#### Code Example 7: Prediction with Random Forest
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 7: Prediction with Random Forest
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    
    # Random Forest model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_val_score(
        rf, X_train_scaled, y_train,
        cv=5,
        scoring='r2'
    )
    
    print(f"Random Forest Cross-Validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Training
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    print(f"Test R²: {r2_score(y_test, y_pred_rf):.3f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
    
    # Predicted vs Actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Random Forest: Predicted vs Actual', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_prediction.png', dpi=300)
    plt.close()
    

#### Code Example 8: Fast Prediction with LightGBM
    
    
    # Install LightGBM
    # pip install lightgbm
    
    import lightgbm as lgb
    
    # LightGBM dataset
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Training
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    # Prediction
    y_pred_lgb = gbm.predict(X_test_scaled, num_iteration=gbm.best_iteration)
    
    print(f"LightGBM Test R²: {r2_score(y_test, y_pred_lgb):.3f}")
    print(f"LightGBM Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lgb)):.3f}")
    

**Sample Output:**
    
    
    Random Forest Cross-Validation R²: 0.912 ± 0.034
    Test R²: 0.924
    Test RMSE: 0.451
    
    LightGBM Test R²: 0.931
    LightGBM Test RMSE: 0.429
    

* * *

## 2.3 Feature Selection and Interpretation

### 2.3.1 Redundancy Removal by Correlation Analysis

#### Code Example 9: Correlation Matrix and Redundant Feature Removal
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 9: Correlation Matrix and Redundant Feature Rem
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Calculate correlation matrix
    corr_matrix = pd.DataFrame(X_train_scaled).corr()
    
    # Detect high correlation pairs (threshold 0.95)
    threshold = 0.95
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
    
    print(f"High correlation pairs (|r| > {threshold}): {len(high_corr_pairs)} pairs")
    
    # Remove redundant features
    columns_to_drop = set()
    for i, j, corr in high_corr_pairs:
        columns_to_drop.add(j)  # Drop j-th (arbitrary choice)
    
    X_train_reduced = np.delete(X_train_scaled, list(columns_to_drop), axis=1)
    X_test_reduced = np.delete(X_test_scaled, list(columns_to_drop), axis=1)
    
    print(f"Number of features before reduction: {X_train_scaled.shape[1]}")
    print(f"Number of features after reduction: {X_train_reduced.shape[1]}")
    
    # Draw heatmap (first 20 features only)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix.iloc[:20, :20],
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('Correlation Matrix Heatmap (First 20 Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()
    

### 2.3.2 Feature Importance Analysis

#### Code Example 10: Feature Interpretation with SHAP
    
    
    # Install SHAP
    # pip install shap
    
    import shap
    
    # Retrain with Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=[f'F{i}' for i in range(X_test_scaled.shape[1])],
        show=False,
        max_display=20
    )
    plt.title('SHAP Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    plt.close()
    
    # Individual sample explanation
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test_scaled[0],
            feature_names=[f'F{i}' for i in range(X_test_scaled.shape[1])]
        ),
        max_display=15,
        show=False
    )
    plt.title('Prediction Explanation for Sample 1', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=300)
    plt.close()
    
    print("SHAP interpretation visualizations saved")
    

* * *

## 2.4 Case Study: Solubility Prediction

### Dataset: ESOL (Estimated Solubility)

The ESOL dataset contains water solubility data for 1,128 compounds.

#### Code Example 11: ESOL Dataset Acquisition and Preprocessing
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 11: ESOL Dataset Acquisition and Preprocessing
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 1-3 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    import numpy as np
    
    # Load ESOL dataset
    # Data from https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    df_esol = pd.read_csv(url)
    
    print(f"Number of data points: {len(df_esol)}")
    print(f"\nColumns: {df_esol.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_esol.head())
    
    # Create molecule objects from SMILES
    df_esol['mol'] = df_esol['smiles'].apply(Chem.MolFromSmiles)
    
    # Remove invalid SMILES
    df_esol = df_esol[df_esol['mol'].notna()]
    print(f"\nNumber of valid molecules: {len(df_esol)}")
    
    # Calculate RDKit descriptors
    descriptors_list = [
        'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors',
        'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumHeteroatoms', 'RingCount', 'FractionCsp3'
    ]
    
    for desc_name in descriptors_list:
        desc_func = getattr(Descriptors, desc_name)
        df_esol[desc_name] = df_esol['mol'].apply(desc_func)
    
    # Add Morgan fingerprints
    from rdkit.Chem import AllChem
    
    def get_morgan_fp(mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return np.array(fp)
    
    fp_array = np.array([get_morgan_fp(mol) for mol in df_esol['mol']])
    
    # Create feature matrix
    X_descriptors = df_esol[descriptors_list].values
    X_fingerprints = fp_array
    X_combined = np.hstack([X_descriptors, X_fingerprints])
    
    # Target variable (solubility logS)
    y = df_esol['measured log solubility in mols per litre'].values
    
    print(f"\nFeature matrix shape:")
    print(f"Descriptors only: {X_descriptors.shape}")
    print(f"Fingerprints only: {X_fingerprints.shape}")
    print(f"Combined: {X_combined.shape}")
    

**Sample Output:**
    
    
    Number of data points: 1128
    
    Columns: ['Compound ID', 'smiles', 'measured log solubility in mols per litre', ...]
    
    Number of valid molecules: 1128
    
    Feature matrix shape:
    Descriptors only: (1128, 10)
    Fingerprints only: (1128, 1024)
    Combined: (1128, 1034)
    

#### Code Example 12: Comparing and Optimizing Multiple Models
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 12: Comparing and Optimizing Multiple Models
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    # Model 2: Random Forest (hyperparameter search)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    rf_grid.fit(X_train_scaled, y_train)
    y_pred_rf = rf_grid.predict(X_test_scaled)
    
    print(f"Random Forest Best Parameters: {rf_grid.best_params_}")
    
    # Model 3: LightGBM
    import lightgbm as lgb
    
    lgb_train = lgb.Dataset(X_train_scaled, y_train)
    lgb_test = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)
    
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    gbm = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    y_pred_lgb = gbm.predict(X_test_scaled, num_iteration=gbm.best_iteration)
    
    # Performance comparison
    models = {
        'Ridge': y_pred_ridge,
        'Random Forest': y_pred_rf,
        'LightGBM': y_pred_lgb
    }
    
    print("\n=== Model Performance Comparison ===")
    for name, y_pred in models.items():
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\n{name}:")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
    
    # Predicted vs Actual plot (3 model comparison)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, y_pred) in zip(axes, models.items()):
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2)
        ax.set_xlabel('Actual log(S)', fontsize=12)
        ax.set_ylabel('Predicted log(S)', fontsize=12)
        ax.set_title(f'{name} (R² = {r2_score(y_test, y_pred):.3f})',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('esol_model_comparison.png', dpi=300)
    plt.close()
    
    print("\nModel comparison plot saved")
    

**Sample Output:**
    
    
    Random Forest Best Parameters: {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}
    
    === Model Performance Comparison ===
    
    Ridge:
      R²: 0.789
      RMSE: 0.712
      MAE: 0.543
    
    Random Forest:
      R²: 0.891
      RMSE: 0.511
      MAE: 0.382
    
    LightGBM:
      R²: 0.912
      RMSE: 0.459
      MAE: 0.341
    
    Model comparison plot saved
    

**Interpretation** : \- **LightGBM** achieves the best performance (R² = 0.912) \- **Random Forest** also performs excellently (R² = 0.891) \- **Ridge** shows slightly lower performance as a linear model

* * *

## Exercises

### Exercise 1: Understanding Molecular Descriptors

For each of the following descriptors, explain their meaning and importance in drug discovery.

  1. **logP (lipophilicity)** : What properties does a high value indicate?
  2. **TPSA (topological polar surface area)** : What is the relationship with blood-brain barrier permeability?
  3. **Molecular weight** : What is the threshold in Lipinski's Rule of Five?

Sample Answer 1\. **logP (lipophilicity)** \- Logarithm of the water-octanol partition coefficient \- High value (e.g., logP > 5): High lipophilicity, good membrane permeability but low water solubility \- Low value (e.g., logP < 0): High water solubility but low membrane permeability \- **Importance in drug discovery**: Affects oral absorption and blood-brain barrier permeability 2\. **TPSA (topological polar surface area)** \- Sum of surface areas of polar atoms (N, O, etc.) in the molecule \- TPSA < 140 Ų: Good oral absorption \- TPSA < 60 Ų: Easily passes through blood-brain barrier \- **Importance in drug discovery**: Essential for CNS drug design 3\. **Molecular weight** \- Lipinski's Rule of Five: MW < 500 Da \- High molecular weight (> 500 Da): Decreased membrane permeability \- **Importance in drug discovery**: Prediction of oral absorption 

* * *

### Exercise 2: Implementing Feature Selection

Perform the following tasks:

  1. Calculate descriptors for the ESOL dataset
  2. Detect feature pairs with correlation coefficient > 0.9
  3. Remove redundant features and retrain the model
  4. Evaluate performance changes

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Perform the following tasks:
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd
    
    # Calculate correlation matrix
    X_df = pd.DataFrame(X_train_scaled)
    corr_matrix = X_df.corr()
    
    # Detect high correlation pairs
    threshold = 0.9
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
    
    print(f"Number of high correlation pairs: {len(high_corr_pairs)}")
    
    # Remove redundant features
    columns_to_drop = set([j for i, j, _ in high_corr_pairs])
    X_train_reduced = np.delete(X_train_scaled, list(columns_to_drop), axis=1)
    X_test_reduced = np.delete(X_test_scaled, list(columns_to_drop), axis=1)
    
    print(f"Before reduction: {X_train_scaled.shape[1]} features")
    print(f"After reduction: {X_train_reduced.shape[1]} features")
    
    # Train model (before reduction)
    rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_original.fit(X_train_scaled, y_train)
    y_pred_orig = rf_original.predict(X_test_scaled)
    r2_orig = r2_score(y_test, y_pred_orig)
    
    # Train model (after reduction)
    rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reduced.fit(X_train_reduced, y_train)
    y_pred_red = rf_reduced.predict(X_test_reduced)
    r2_red = r2_score(y_test, y_pred_red)
    
    print(f"\nR² (before reduction): {r2_orig:.3f}")
    print(f"R² (after reduction): {r2_red:.3f}")
    print(f"Performance change: {r2_red - r2_orig:+.3f}")
    

**Expected output:** 
    
    
    Number of high correlation pairs: 145
    Before reduction: 1034 features
    After reduction: 889 features
    
    R² (before reduction): 0.891
    R² (after reduction): 0.887
    Performance change: -0.004
    

**Interpretation**: With a slight performance decrease (0.004), we removed 145 redundant features, making the model more concise. 

* * *

### Exercise 3: Hyperparameter Tuning

Optimize the hyperparameters of LightGBM to achieve an R² of 0.92 or higher on the test set.

**Hint** : \- `num_leaves` (number of leaves) \- `learning_rate` (learning rate) \- `feature_fraction` (feature sampling ratio)

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Hint:
    -num_leaves(number of leaves)
    -learning_rate(learning 
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import lightgbm as lgb
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    
    # Define parameter space
    param_dist = {
        'num_leaves': [15, 31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'feature_fraction': [0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 5, 10],
        'min_child_samples': [10, 20, 30]
    }
    
    # LightGBM model
    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    # Random search
    random_search = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV R²: {random_search.best_score_:.3f}")
    
    # Evaluation on test set
    y_pred_optimized = random_search.predict(X_test_scaled)
    r2_optimized = r2_score(y_test, y_pred_optimized)
    rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
    
    print(f"\nTest R²: {r2_optimized:.3f}")
    print(f"Test RMSE: {rmse_optimized:.3f}")
    
    if r2_optimized >= 0.92:
        print("✅ Goal achieved! R² ≥ 0.92")
    else:
        print(f"❌ Goal not achieved. Need {0.92 - r2_optimized:.3f} more")
    

**Expected output:** 
    
    
    Best parameters: {'num_leaves': 63, 'n_estimators': 300, 'min_child_samples': 10,
                      'learning_rate': 0.05, 'feature_fraction': 0.9,
                      'bagging_freq': 5, 'bagging_fraction': 0.9}
    Best CV R²: 0.908
    
    Test R²: 0.923
    Test RMSE: 0.429
    
    ✅ Goal achieved! R² ≥ 0.92
    

* * *

### Exercise 4: Building an Ensemble Model

Average (ensemble) the predictions of three models: Ridge, Random Forest, and LightGBM to improve performance.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: Average (ensemble) the predictions of three models: Ridge, R
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Train individual models
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    # LightGBM
    lgb_train = lgb.Dataset(X_train_scaled, y_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'verbose': -1
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=200)
    y_pred_lgb = gbm.predict(X_test_scaled)
    
    # Ensemble (simple average)
    y_pred_ensemble = (y_pred_ridge + y_pred_rf + y_pred_lgb) / 3
    
    # Performance comparison
    print("=== Individual Models and Ensemble Performance ===\n")
    models = {
        'Ridge': y_pred_ridge,
        'Random Forest': y_pred_rf,
        'LightGBM': y_pred_lgb,
        'Ensemble (Average)': y_pred_ensemble
    }
    
    for name, y_pred in models.items():
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name:20s} R²: {r2:.3f}  RMSE: {rmse:.3f}")
    
    # Weighted ensemble (weights based on performance)
    weights = [0.1, 0.4, 0.5]  # Ridge, RF, LightGBM
    y_pred_weighted = (
        weights[0] * y_pred_ridge +
        weights[1] * y_pred_rf +
        weights[2] * y_pred_lgb
    )
    
    r2_weighted = r2_score(y_test, y_pred_weighted)
    rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
    
    print(f"\nWeighted Ensemble  R²: {r2_weighted:.3f}  RMSE: {rmse_weighted:.3f}")
    

**Expected output:** 
    
    
    === Individual Models and Ensemble Performance ===
    
    Ridge                R²: 0.789  RMSE: 0.712
    Random Forest        R²: 0.891  RMSE: 0.511
    LightGBM             R²: 0.912  RMSE: 0.459
    Ensemble (Average)   R²: 0.918  RMSE: 0.443
    
    Weighted Ensemble  R²: 0.920  RMSE: 0.437
    

**Interpretation**: The ensemble achieves even better performance than the best single model (LightGBM). 

* * *

## Summary

In this chapter, we learned the following:

### Topics Covered

  1. **Molecular Descriptors** \- 1D descriptors: Molecular weight, logP, TPSA \- 2D descriptors: Morgan fingerprints, MACCS keys \- 3D descriptors: Conformation-dependent descriptors \- Comprehensive calculation with mordred (over 1,800 types)

  2. **QSAR/QSPR Modeling** \- Linear models (Ridge, Lasso) \- Nonlinear models (Random Forest, LightGBM) \- Model evaluation (R², RMSE, MAE)

  3. **Feature Selection and Interpretation** \- Redundancy removal by correlation analysis \- Feature importance with SHAP/LIME \- Understanding which substructures contribute to properties

  4. **Practical Application: ESOL Solubility Prediction** \- Data acquisition and preprocessing \- Comparison of multiple models \- Hyperparameter optimization \- Ensemble learning

### Next Steps

In Chapter 3, we will learn about chemical space exploration and similarity searching.

**[Chapter 3: Chemical Space Exploration and Similarity Searching →](<chapter-3.html>)**

* * *

## References

  1. Todeschini, R., & Consonni, V. (2009). _Molecular Descriptors for Chemoinformatics_. Wiley-VCH. ISBN: 978-3527318520
  2. Delaney, J. S. (2004). "ESOL: Estimating aqueous solubility directly from molecular structure." _Journal of Chemical Information and Computer Sciences_ , 44(3), 1000-1005. DOI: 10.1021/ci034243x
  3. Moriwaki, H. et al. (2018). "Mordred: a molecular descriptor calculator." _Journal of Cheminformatics_ , 10, 4. DOI: 10.1186/s13321-018-0258-y
  4. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." _Advances in Neural Information Processing Systems_ , 30.

* * *

**[← Chapter 1](<chapter-1.html>)** | **[Back to Series Index](<./index.html>)** | **[Chapter 3 →](<chapter-3.html>)**
