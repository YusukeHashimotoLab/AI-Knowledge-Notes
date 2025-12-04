---
title: "Chapter 5: Python Practice: matminer Workflow"
chapter_title: "Chapter 5: Python Practice: matminer Workflow"
subtitle: Building End-to-End Material Discovery Pipelines
---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/composition-features-introduction/chapter-5.html>) | Last sync: 2025-11-16

## Learning Objectives

In this final chapter, we integrate all knowledge from Chapters 1-4 to build a complete workflow that can be used in actual material discovery projects.

### Learning Goals

  * **Fundamental Understanding** : Materials Project API, AutoFeaturizer, complete ML pipeline configuration
  * **Practical Skills** : Implementation of data acquisition → feature extraction → model training → prediction → visualization, joblib save/load
  * **Application Ability** : New material prediction, error analysis and model improvement, batch prediction system construction

## 5.1 Materials Project API Data Acquisition

Materials Project is one of the world's largest open material databases, providing over 150,000 material data points based on DFT calculations.

### 5.1.1 API Key Acquisition and Authentication

To use the Materials Project API, you need a free API key:

  1. Visit [Materials Project](<https://materialsproject.org/>)
  2. Create an account via "Sign Up" in the upper right
  3. After logging in, obtain your API key from "Dashboard" → "API"

    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: To use the Materials Project API, you need a free API key:
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
                <a class="colab-badge" href="https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example1.ipynb" target="_blank">
                    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align: middle;"/>
                </a>
                <h4>Example 1: Materials Project API Data Acquisition (10,000 compounds)</h4>
                <pre><code class="language-python"># ===================================
    # Example 1: Materials Project API Data Acquisition
    # ===================================
    
    # Import necessary libraries
    from mp_api.client import MPRester
    from pymatgen.core import Composition
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    
    # API key configuration (replace with your own key)
    API_KEY = "your_api_key_here"
    
    def fetch_materials_data(api_key, max_compounds=10000):
        """Fetch material data from Materials Project
    
        Args:
            api_key (str): Materials Project API key
            max_compounds (int): Maximum number of compounds to retrieve
    
        Returns:
            pd.DataFrame: Material data (chemical formula, formation energy, band gap, etc.)
        """
        with MPRester(api_key) as mpr:
            # Retrieve formation energy and band gap data
            # Stability criterion: Energy on or near convex hull (e_above_hull < 0.1 eV/atom)
            docs = mpr.materials.summary.search(
                energy_above_hull=(0, 0.1),  # Include metastable materials
                fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                       "band_gap", "elements", "nelements"],
                num_chunks=10,
                chunk_size=1000
            )
    
            # Convert to DataFrame
            data = []
            for doc in docs[:max_compounds]:
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'formation_energy': doc.formation_energy_per_atom,
                    'band_gap': doc.band_gap,
                    'elements': ' '.join(doc.elements),
                    'n_elements': doc.nelements
                })
    
            df = pd.DataFrame(data)
            return df
    
    # Execute data acquisition
    df = fetch_materials_data(API_KEY, max_compounds=10000)
    
    print(f"Number of data points retrieved: {len(df)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Statistical information
    print(f"\nFormation energy range: {df['formation_energy'].min():.3f} ~ {df['formation_energy'].max():.3f} eV/atom")
    print(f"Band gap range: {df['band_gap'].min():.3f} ~ {df['band_gap'].max():.3f} eV")
    print(f"Element count distribution:\n{df['n_elements'].value_counts().sort_index()}")
    
    # Expected output:
    # Number of data points retrieved: 10000
    # First 5 rows:
    #   material_id formula  formation_energy  band_gap  ...
    # 0  mp-1234    Fe2O3    -2.543           2.18       ...
    # 1  mp-5678    TiO2     -4.889           3.25       ...
    # ...
    #
    # Formation energy range: -5.234 ~ 0.099 eV/atom
    # Band gap range: 0.000 ~ 9.876 eV
    # Element count distribution:
    # 2    3456
    # 3    4123
    # 4    1892
    # 5    529
    

## 5.2 Automated Feature Generation with AutoFeaturizer

matminer's `AutoFeaturizer` automatically detects chemical composition or crystal structure and generates appropriate features.

### 5.2.1 How AutoFeaturizer Works

  * **preset selection** : 
    * `express`: Fast (22 features, 10 sec/1000 compounds)
    * `fast`: Medium speed (50 features, 30 sec/1000 compounds)
    * `all`: Complete (145 features, 120 sec/1000 compounds)
  * **Missing value handling** : Automatic processing with DataCleaner
  * **Feature selection** : Integration with VarianceThreshold, FeatureAgglomeration possible

    
    
                [
                    ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
                ](<https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example2.ipynb>)
                
    
    #### Example 2: AutoFeaturizer Application (preset='express')
    
    
                
    
    
    # ===================================
    # Example 2: AutoFeaturizer Application
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    import time
    
    # Convert chemical formula strings to Composition objects
    df = StrToComposition().featurize_dataframe(df, 'formula')
    
    # Instead of AutoFeaturizer, manually build express preset equivalent
    # (Actual AutoFeaturizer automatically selects optimal Featurizer based on preset)
    featurizer = ElementProperty.from_preset("magpie")
    
    # Feature generation
    start_time = time.time()
    df = featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
    elapsed = time.time() - start_time
    
    print(f"Feature generation completed: {elapsed:.2f} seconds")
    print(f"Number of features generated: {len(featurizer.feature_labels())}")
    print(f"Feature names (first 10):\n{featurizer.feature_labels()[:10]}")
    
    # Missing value handling with DataCleaner
    from matminer.utils.data import MixingInfoError
    # Remove rows with missing values (consider imputation in production)
    df_clean = df.dropna()
    print(f"\nAfter missing value processing: {len(df_clean)} rows (original: {len(df)} rows)")
    
    # Expected output:
    # Feature generation completed: 8.54 seconds
    # Number of features generated: 132
    # Feature names (first 10):
    # ['MagpieData minimum Number', 'MagpieData maximum Number', ...]
    #
    # After missing value processing: 9876 rows (original: 10000 rows)
    

`

## 5.3 Building Complete ML Pipeline

Utilizing scikit-learn's Pipeline, we create a consistent workflow from data acquisition to prediction.
    
    
    ```mermaid
    graph LR
        A[Data AcquisitionMP API] --> B[Feature Extractionmatminer]
        B --> C[PreprocessingStandardScaler]
        C --> D[Model TrainingRandomForest]
        D --> E[EvaluationR², MAE]
        E --> F{Performance OK?}
        F -->|Yes| G[Model Savejoblib]
        F -->|No| H[HyperparameterOptimization]
        H --> D
    
        style A fill:#e3f2fd
        style G fill:#e8f5e9
        style F fill:#fff3e0
    ```
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Utilizing scikit-learn's Pipeline, we create a consistent wo
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
                <a class="colab-badge" href="https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example3.ipynb" target="_blank">
                    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
                </a>
                <h4>Example 3: Complete ML Pipeline (Data → Model → Prediction)</h4>
                <pre><code class="language-python"># ===================================
    # Example 3: Complete ML Pipeline
    # ===================================
    
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    
    # Separate features and target
    feature_cols = [col for col in df_clean.columns
                    if col.startswith('MagpieData')]
    X = df_clean[feature_cols].values
    y = df_clean['formation_energy'].values
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target: {y.shape}")
    
    # Train/test data split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Pipeline construction
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Model training
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Prediction and evaluation
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== Performance Evaluation ===")
    print(f"MAE: {mae:.4f} eV/atom")
    print(f"R²:  {r2:.4f}")
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    print(f"\nCV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f} eV/atom")
    
    # Expected output:
    # Feature matrix: (9876, 132)
    # Target: (9876,)
    #
    # Training model...
    #
    # === Performance Evaluation ===
    # MAE: 0.1234 eV/atom
    # R²:  0.8976
    #
    # CV MAE: 0.1298 ± 0.0087 eV/atom
    

## 5.4 Model Save and Load
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    """
    Example: 5.4 Model Save and Load
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
                <a class="colab-badge" href="https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example4.ipynb" target="_blank">
                    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
                </a>
                <h4>Example 4: Model Save and Load (joblib)</h4>
                <pre><code class="language-python"># ===================================
    # Example 4: Model Save and Load
    # ===================================
    
    import joblib
    from pathlib import Path
    
    # Save model
    model_path = Path('composition_formation_energy_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved: {model_path}")
    print(f"File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load model
    loaded_pipeline = joblib.load(model_path)
    print("\nModel loaded successfully")
    
    # Prediction with loaded model (validation)
    y_pred_loaded = loaded_pipeline.predict(X_test[:5])
    y_pred_original = pipeline.predict(X_test[:5])
    
    print("\nPrediction comparison (first 5 samples):")
    print("Original model:    ", y_pred_original)
    print("Loaded model:      ", y_pred_loaded)
    print("Match:", np.allclose(y_pred_original, y_pred_loaded))
    
    # Expected output:
    # Model saved: composition_formation_energy_model.pkl
    # File size: 24.56 MB
    #
    # Model loaded successfully
    #
    # Prediction comparison (first 5 samples):
    # Original model:     [-2.543 -4.889 -1.234 -3.456 -0.987]
    # Loaded model:       [-2.543 -4.889 -1.234 -3.456 -0.987]
    # Match: True
    

## 5.5 New Material Prediction and Visualization

Using the trained model, we predict the properties of unknown materials. For Random Forest, uncertainty can also be estimated from the prediction distribution of all decision trees.

## Learning Objectives Verification

Upon completing this chapter, you will be able to:

### Fundamental Understanding

  * ✅ Understand how to use the Materials Project API
  * ✅ Explain the AutoFeaturizer mechanism and preset selection
  * ✅ List the components of a complete ML pipeline

### Practical Skills

  * ✅ Retrieve 10,000 compound data from MP API
  * ✅ Automatically generate features with matminer
  * ✅ Execute training → evaluation → save with scikit-learn Pipeline
  * ✅ Save and load models with joblib
  * ✅ Execute predictions on new materials

### Application Ability

  * ✅ Design actual material discovery projects
  * ✅ Propose model improvement strategies from error analysis
  * ✅ Build batch prediction systems

## Exercises

### Easy (Basic Verification)

**Q1** : How to retrieve only oxides (O-containing) from Materials Project API?

**Answer** :
    
    
    docs = mpr.materials.summary.search(
        elements=["O"],  # O-containing
        energy_above_hull=(0, 0.1),
        fields=["material_id", "formula_pretty", ...]
    )

**Explanation** : The `elements` parameter filters materials containing specific elements.

## References

  1. Ward, L. et al. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69.
  2. Dunn, A. et al. (2020). "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." _npj Computational Materials_ , 6, 138, pp. 5-8.
  3. Ong, S.P. et al. (2015). "The Materials Application Programming Interface (API)." _Computational Materials Science_ , 97, 209-215.
  4. Materials Project API Documentation. https://docs.materialsproject.org/
  5. matminer Examples Gallery. https://hackingmaterials.lbl.gov/matminer/examples/
  6. pandas Documentation: Data manipulation. https://pandas.pydata.org/docs/
  7. matplotlib/seaborn Documentation. https://matplotlib.org/

## Next Steps

🎉 **Congratulations!** You have completed the Composition-Based Features Introduction Series.

Next learning resources:

  * **gnn-features-comparison** : Detailed comparison of composition-based vs GNN structure-based features
  * **Advanced MI Topics** : Transfer learning, Active Learning, Bayesian optimization
  * **Practical Projects** : Kaggle Materials Science competitions

[← Chapter 4: Integration with Machine Learning Models](<chapter-4.html>) [Return to Series Index](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
