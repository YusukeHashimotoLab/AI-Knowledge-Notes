---
title: "Chapter 2: Sequence Analysis and Machine Learning"
chapter_title: "Chapter 2: Sequence Analysis and Machine Learning"
subtitle: Practical Protein Function Prediction
reading_time: 25-30 min
difficulty: Beginner to Intermediate
code_examples: 9
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 2: Sequence Analysis and Machine Learning

This chapter covers Sequence Analysis and Machine Learning. You will learn Predict protein localization and Predict enzyme activity using Random Forest.

**Practical Protein Function Prediction**

## Learning Objectives

  * ✅ Execute BLAST searches to discover homologous proteins
  * ✅ Extract physicochemical features from amino acid sequences
  * ✅ Predict protein localization and function using machine learning models
  * ✅ Predict enzyme activity using Random Forest and LightGBM
  * ✅ Evaluate and improve model performance

**Reading time** : 25-30 min | **Code examples** : 9 | **Exercises** : 3

* * *

## 2.1 Sequence Alignment

### What is Alignment

**Sequence alignment** is a technique for comparing two or more sequences to evaluate their similarity.
    
    
    ```mermaid
    flowchart LR
        A[Query Sequence] --> B[Alignment\nAlgorithm]
        C[Database] --> B
        B --> D[Homologous Sequences]
        D --> E[Function Inference]
        D --> F[Evolutionary Analysis]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#fff9c4
    ```

**Applications** : \- Function prediction of unknown proteins \- Analysis of evolutionary relationships \- Identification of important residues

* * *

### BLAST Search

**BLAST (Basic Local Alignment Search Tool)** is the most widely used sequence similarity search tool.

**Example 1: BLAST Search using Biopython**
    
    
    from Bio.Blast import NCBIWWW, NCBIXML
    from Bio import SeqIO
    
    # Query sequence (example: insulin)
    query_sequence = """
    MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
    """
    
    print("Running BLAST search...")
    
    # Search on NCBI BLAST server
    result_handle = NCBIWWW.qblast(
        "blastp",  # Protein BLAST
        "nr",  # non-redundant database
        query_sequence,
        hitlist_size=10  # Top 10 hits
    )
    
    # Save results to file
    with open("blast_results.xml", "w") as out_handle:
        out_handle.write(result_handle.read())
    
    result_handle.close()
    
    print("Search complete. Analyzing results...")
    
    # Parse results
    with open("blast_results.xml") as result_handle:
        blast_records = NCBIXML.parse(result_handle)
    
        for blast_record in blast_records:
            print(f"\n=== BLAST Search Results ===")
            print(f"Query length: {blast_record.query_length} aa")
            print(f"Number of hits: {len(blast_record.alignments)}")
    
            # Display top 5 hits
            for alignment in blast_record.alignments[:5]:
                for hsp in alignment.hsps:
                    print(f"\n--- Hit ---")
                    print(f"Sequence: {alignment.title[:60]}...")
                    print(f"E-value: {hsp.expect:.2e}")
                    print(f"Score: {hsp.score}")
                    print(f"Identity: {hsp.identities}/{hsp.align_length} "
                          f"({100*hsp.identities/hsp.align_length:.1f}%)")
                    print(f"Alignment:")
                    print(f"Query: {hsp.query[:60]}")
                    print(f"       {hsp.match[:60]}")
                    print(f"Sbjct: {hsp.sbjct[:60]}")
    

**Sample output** :
    
    
    === BLAST Search Results ===
    Query length: 110 aa
    Number of hits: 50
    
    --- Hit ---
    Sequence: insulin [Homo sapiens]
    E-value: 2.5e-75
    Score: 229
    Identity: 110/110 (100.0%)
    Alignment:
    Query: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED
           ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    Sbjct: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED
    

* * *

### Local vs Global Alignment

**Global Alignment (Needleman-Wunsch)** : \- Compares entire sequences \- Applied to sequences of similar length

**Local Alignment (Smith-Waterman)** : \- Detects most similar subregions \- Applied to sequences of different lengths (used by BLAST)

**Example 2: Alignment with Biopython**
    
    
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    
    # Two sequences
    seq1 = "ACDEFGHIKLMNPQRSTVWY"
    seq2 = "ACDEFGHIKLQNPQRSTVWY"  # 1 character different
    
    # Global alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)
    
    print("=== Global Alignment ===")
    print(format_alignment(*alignments[0]))
    
    # Local alignment
    local_alignments = pairwise2.align.localxx(seq1, seq2)
    
    print("\n=== Local Alignment ===")
    print(format_alignment(*local_alignments[0]))
    
    # Scoring: match +2, mismatch -1, gap -0.5
    gap_penalty = -0.5
    alignments_scored = pairwise2.align.globalms(
        seq1, seq2,
        match=2,
        mismatch=-1,
        open=-0.5,
        extend=-0.1
    )
    
    print("\n=== Alignment with Scoring ===")
    print(format_alignment(*alignments_scored[0]))
    print(f"Score: {alignments_scored[0][2]:.1f}")
    

* * *

## 2.2 Feature Extraction from Sequences

### Physicochemical Properties of Amino Acids

**20 amino acids** have different physicochemical properties:

Property | Amino Acids  
---|---  
Hydrophobic | A, V, I, L, M, F, W, P  
Hydrophilic (Polar) | S, T, N, Q, Y, C  
Basic (Positively charged) | K, R, H  
Acidic (Negatively charged) | D, E  
Aromatic | F, Y, W  
  
**Example 3: Calculating Amino Acid Composition**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from collections import Counter
    import numpy as np
    
    def calculate_aa_composition(sequence):
        """
        Calculate amino acid composition
    
        Returns:
        --------
        dict: Frequency (%) of each amino acid
        """
        # Convert to uppercase
        sequence = sequence.upper()
    
        # 20 standard amino acids
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    
        # Count
        aa_count = Counter(sequence)
    
        # Convert to frequency (%)
        total = len(sequence)
        composition = {aa: 100 * aa_count.get(aa, 0) / total
                       for aa in standard_aa}
    
        return composition
    
    # Test sequence
    sequence = """
    MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY
    TPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSIC
    SLYQLENYCN
    """
    sequence = sequence.replace("\n", "").replace(" ", "")
    
    comp = calculate_aa_composition(sequence)
    
    print("=== Amino Acid Composition ===")
    for aa in sorted(comp.keys(), key=lambda x: comp[x], reverse=True):
        if comp[aa] > 0:
            print(f"{aa}: {comp[aa]:.1f}%")
    
    # Aggregation of physicochemical properties
    hydrophobic = ["A", "V", "I", "L", "M", "F", "W", "P"]
    charged = ["K", "R", "H", "D", "E"]
    polar = ["S", "T", "N", "Q", "Y", "C"]
    
    hydrophobic_pct = sum(comp[aa] for aa in hydrophobic)
    charged_pct = sum(comp[aa] for aa in charged)
    polar_pct = sum(comp[aa] for aa in polar)
    
    print(f"\nHydrophobic amino acids: {hydrophobic_pct:.1f}%")
    print(f"Charged amino acids: {charged_pct:.1f}%")
    print(f"Polar amino acids: {polar_pct:.1f}%")
    

* * *

### k-mer Representation

**k-mer** is a contiguous subsequence of length k (equivalent to n-grams in natural language processing).

**Example 4: k-mer Feature Extraction**
    
    
    from collections import Counter
    
    def extract_kmers(sequence, k=3):
        """
        Extract k-mer features
    
        Parameters:
        -----------
        sequence : str
            Amino acid sequence
        k : int
            Length of k-mer
    
        Returns:
        --------
        dict: Frequency of k-mers
        """
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
    
        # Calculate frequency
        kmer_counts = Counter(kmers)
        total = len(kmers)
    
        # Convert to frequency (%)
        kmer_freq = {kmer: 100 * count / total
                     for kmer, count in kmer_counts.items()}
    
        return kmer_freq
    
    # Test
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
    
    # 3-mer
    kmers_3 = extract_kmers(sequence, k=3)
    
    print("=== Top 10 3-mers ===")
    for kmer, freq in sorted(kmers_3.items(),
                             key=lambda x: x[1],
                             reverse=True)[:10]:
        print(f"{kmer}: {freq:.2f}%")
    
    # Create feature vector
    def create_kmer_vector(sequence, k=3):
        """
        Create k-mer feature vector (for machine learning)
        """
        # Generate all possible k-mers (size: 20^k)
        # In practice, use only frequent k-mers
        kmers = extract_kmers(sequence, k)
    
        # Vectorization (frequency-based)
        vector = list(kmers.values())
        return vector
    
    # Feature vector for machine learning
    feature_vector = create_kmer_vector(sequence, k=2)
    print(f"\nFeature vector dimension: {len(feature_vector)}")
    

* * *

### Physicochemical Descriptors

**Example 5: Calculating Hydrophobicity Profile**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Kyte-Doolittle hydrophobicity scale
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
        'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
        'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
        'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    def calculate_hydrophobicity_profile(sequence, window=9):
        """
        Calculate hydrophobicity profile
    
        Parameters:
        -----------
        sequence : str
            Amino acid sequence
        window : int
            Window size for moving average
    
        Returns:
        --------
        list: Hydrophobicity score at each position
        """
        profile = []
    
        for i in range(len(sequence) - window + 1):
            segment = sequence[i:i+window]
            # Average hydrophobicity within window
            hydrophobicity = np.mean([
                hydrophobicity_scale.get(aa, 0)
                for aa in segment
            ])
            profile.append(hydrophobicity)
    
        return profile
    
    # Test sequence
    sequence = """
    MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY
    """
    sequence = sequence.replace("\n", "").replace(" ", "")
    
    # Hydrophobicity profile
    profile = calculate_hydrophobicity_profile(sequence, window=9)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(profile)), profile, linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Hydrophobicity Score', fontsize=12)
    plt.title('Kyte-Doolittle Hydrophobicity Profile', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('hydrophobicity_profile.png', dpi=300)
    plt.show()
    
    # Predict transmembrane regions (high hydrophobicity regions)
    threshold = 1.6  # Threshold for transmembrane regions
    tm_regions = []
    
    for i, score in enumerate(profile):
        if score > threshold:
            tm_regions.append(i)
    
    if tm_regions:
        print(f"Transmembrane region candidate: {min(tm_regions)}-{max(tm_regions)}")
    else:
        print("No transmembrane regions detected")
    

* * *

## 2.3 Function Prediction with Machine Learning

### Protein Localization Prediction

**Example 6: Subcellular Localization Prediction**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Generate sample data (in practice, retrieve from databases)
    def generate_sample_data(n_samples=500):
        """
        Generate sample data (for demo)
        """
        np.random.seed(42)
    
        data = []
    
        # Nuclear: High in basic amino acids
        for _ in range(n_samples // 5):
            features = {
                'hydrophobic_pct': np.random.normal(30, 5),
                'charged_pct': np.random.normal(35, 5),  # High
                'polar_pct': np.random.normal(25, 5),
                'aromatic_pct': np.random.normal(10, 3),
                'length': np.random.normal(300, 50),
                'localization': 'Nuclear'
            }
            data.append(features)
    
        # Cytoplasmic
        for _ in range(n_samples // 5):
            features = {
                'hydrophobic_pct': np.random.normal(35, 5),
                'charged_pct': np.random.normal(25, 5),
                'polar_pct': np.random.normal(30, 5),
                'aromatic_pct': np.random.normal(10, 3),
                'length': np.random.normal(350, 60),
                'localization': 'Cytoplasmic'
            }
            data.append(features)
    
        # Membrane: High in hydrophobic amino acids
        for _ in range(n_samples // 5):
            features = {
                'hydrophobic_pct': np.random.normal(50, 5),  # High
                'charged_pct': np.random.normal(15, 5),
                'polar_pct': np.random.normal(25, 5),
                'aromatic_pct': np.random.normal(10, 3),
                'length': np.random.normal(280, 40),
                'localization': 'Membrane'
            }
            data.append(features)
    
        # Mitochondrial
        for _ in range(n_samples // 5):
            features = {
                'hydrophobic_pct': np.random.normal(38, 5),
                'charged_pct': np.random.normal(28, 5),
                'polar_pct': np.random.normal(24, 5),
                'aromatic_pct': np.random.normal(10, 3),
                'length': np.random.normal(320, 55),
                'localization': 'Mitochondrial'
            }
            data.append(features)
    
        # Secreted
        for _ in range(n_samples // 5):
            features = {
                'hydrophobic_pct': np.random.normal(32, 5),
                'charged_pct': np.random.normal(22, 5),
                'polar_pct': np.random.normal(36, 5),  # High
                'aromatic_pct': np.random.normal(10, 3),
                'length': np.random.normal(250, 45),
                'localization': 'Secreted'
            }
            data.append(features)
    
        return pd.DataFrame(data)
    
    # Generate data
    df = generate_sample_data(n_samples=500)
    
    print("=== Dataset Overview ===")
    print(df.head())
    print(f"\nLocalization distribution:")
    print(df['localization'].value_counts())
    
    # Separate features and labels
    X = df.drop('localization', axis=1)
    y = df['localization']
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Evaluation
    print("\n=== Evaluation Results ===")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Predict new protein
    new_protein = {
        'hydrophobic_pct': 48.0,
        'charged_pct': 18.0,
        'polar_pct': 24.0,
        'aromatic_pct': 10.0,
        'length': 290.0
    }
    
    prediction = model.predict([list(new_protein.values())])
    print(f"\nPredicted localization for new protein: {prediction[0]}")
    

* * *

## 2.4 Case Study: Enzyme Activity Prediction

### Data Collection and Preprocessing

**Example 7: Preparing Enzyme Dataset**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    def extract_features_from_sequence(sequence):
        """
        Extract features from sequence
        """
        # Amino acid composition
        aa_count = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        aa_comp = {f"aa_{aa}": count / len(sequence)
                   for aa, count in aa_count.items()}
    
        # Physicochemical properties
        hydrophobic = sum(sequence.count(aa) for aa in "AVILMFWP")
        charged = sum(sequence.count(aa) for aa in "KRHDE")
        polar = sum(sequence.count(aa) for aa in "STNQYC")
    
        features = {
            **aa_comp,
            'length': len(sequence),
            'hydrophobic_pct': 100 * hydrophobic / len(sequence),
            'charged_pct': 100 * charged / len(sequence),
            'polar_pct': 100 * polar / len(sequence),
            'molecular_weight': sum(
                aa_count.get(aa, 0) * mw
                for aa, mw in {
                    'A': 89, 'C': 121, 'D': 133, 'E': 147,
                    'F': 165, 'G': 75, 'H': 155, 'I': 131,
                    'K': 146, 'L': 131, 'M': 149, 'N': 132,
                    'P': 115, 'Q': 146, 'R': 174, 'S': 105,
                    'T': 119, 'V': 117, 'W': 204, 'Y': 181
                }.items()
            )
        }
    
        return features
    
    # Sample data (in practice, retrieve from UniProt etc.)
    enzyme_data = [
        {
            'sequence': "ACDEFGHIKLMNPQRSTVWY" * 10,
            'activity': 8.5  # log(kcat/Km)
        },
        # In practice, hundreds to thousands of data points
    ]
    
    print("Feature extraction demo:")
    features = extract_features_from_sequence(enzyme_data[0]['sequence'])
    print(f"Number of features: {len(features)}")
    print(f"Sample features:")
    for key, value in list(features.items())[:5]:
        print(f"  {key}: {value:.3f}")
    

* * *

### Model Training and Evaluation

**Example 8: Enzyme Activity Prediction with LightGBM**
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Example 8: Enzyme Activity Prediction with LightGBM
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    # Generate sample data (in practice, from databases)
    np.random.seed(42)
    
    n_samples = 200
    X = np.random.randn(n_samples, 25)  # 25 features
    # Activity depends on specific features
    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] +
         np.random.randn(n_samples) * 0.5)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"=== Model Performance ===")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)
    plt.xlabel('Observed (log(kcat/Km))', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title(f'Enzyme Activity Prediction (R²={r2:.3f})', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('enzyme_activity_prediction.png', dpi=300)
    plt.show()
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='r2'
    )
    
    print(f"\nCV R²: {cv_scores.mean():.3f} ± "
          f"{cv_scores.std():.3f}")
    

* * *

### Hyperparameter Tuning

**Example 9: Optimization with Optuna**
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - optuna>=3.2.0
    
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        """
        Optuna objective function
        """
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.3
            ),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int(
                'min_child_samples', 5, 50
            ),
            'random_state': 42
        }
    
        # Model
        model = lgb.LGBMRegressor(**params)
    
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring='r2'
        )
    
        return scores.mean()
    
    # Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print(f"\n=== Optimization Results ===")
    print(f"Best R²: {study.best_value:.3f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Retrain with best model
    best_model = lgb.LGBMRegressor(**study.best_params)
    best_model.fit(X_train, y_train)
    
    y_pred_best = best_model.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    
    print(f"\nTest R² (after optimization): {r2_best:.3f}")
    

* * *

## 2.5 Chapter Summary

### What We Learned

  1. **Sequence Alignment** \- Homologous sequence search using BLAST \- Local vs Global Alignment

  2. **Feature Extraction** \- Amino acid composition \- k-mer representation \- Physicochemical descriptors (hydrophobicity, etc.)

  3. **Machine Learning** \- Localization prediction with Random Forest \- Enzyme activity prediction with LightGBM \- Hyperparameter tuning

### Next Chapter

In Chapter 3, we will learn about **Molecular Docking and Interaction Analysis**.

**[Chapter 3: Molecular Docking and Interaction Analysis →](<chapter-3.html>)**

* * *

## Data Licenses and Citations

### Sequence Databases

#### 1\. NCBI (National Center for Biotechnology Information)

  * **License** : Public Domain
  * **Citation** : NCBI Resource Coordinators. (2018). "Database resources of the National Center for Biotechnology Information." _Nucleic Acids Research_ , 46(D1), D8-D13.
  * **BLAST** : https://blast.ncbi.nlm.nih.gov/
  * **Usage** : Protein sequence search, homology analysis

#### 2\. UniProt

  * **License** : CC BY 4.0
  * **Citation** : The UniProt Consortium. (2023). "UniProt: the Universal Protein Knowledgebase in 2023." _Nucleic Acids Research_ , 51(D1), D523-D531.
  * **Access** : https://www.uniprot.org/
  * **Usage** : Annotated sequence data, enzyme activity information

#### 3\. Pfam (Protein families database)

  * **License** : CC0 1.0
  * **Citation** : Mistry, J. et al. (2021). "Pfam: The protein families database in 2021." _Nucleic Acids Research_ , 49(D1), D412-D419.
  * **Access** : https://pfam.xfam.org/
  * **Usage** : Protein domains, functional classification

### Library Licenses

Library | Version | License | Usage  
---|---|---|---  
Biopython | 1.81+ | BSD-3-Clause | BLAST, sequence analysis  
scikit-learn | 1.3+ | BSD-3-Clause | Machine learning  
LightGBM | 4.0+ | MIT | Gradient boosting  
Optuna | 3.3+ | MIT | Hyperparameter optimization  
  
* * *

## Code Reproducibility

### Random Seed Setting
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Random Seed Setting
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # Fix all random numbers
    import numpy as np
    import random
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    
    # scikit-learn
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=SEED)
    
    # LightGBM
    import lightgbm as lgb
    model = lgb.LGBMRegressor(random_state=SEED)
    

### Ensuring BLAST Search Reproducibility
    
    
    from Bio.Blast import NCBIWWW
    from datetime import datetime
    
    # Record timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Execute BLAST
    result_handle = NCBIWWW.qblast(
        "blastp",
        "nr",
        query_sequence,
        hitlist_size=10
    )
    
    # Save results (reproducible)
    filename = f"blast_results_{timestamp.replace(' ', '_')}.xml"
    with open(filename, "w") as f:
        f.write(result_handle.read())
    
    print(f"BLAST results saved: {filename}")
    

* * *

## Common Pitfalls and Solutions

### 1\. Misinterpretation of BLAST E-value

**Problem** : Misunderstanding that all low E-values are significant

**NG** :
    
    
    if hsp.expect < 0.05:  # Confused with statistical significance
        accept_hit()
    

**OK** :
    
    
    # E-value threshold varies by application
    if application == 'homology_search':
        threshold = 1e-10  # Strict threshold
    elif application == 'remote_homology':
        threshold = 1e-3   # Loose threshold
    
    if hsp.expect < threshold and hsp.identities/hsp.align_length > 0.3:
        # Also consider sequence identity
        accept_hit()
    

### 2\. Gap Penalty Setting Error in Sequence Alignment

**Problem** : Using default values as-is

**NG** :
    
    
    # Default values are not always optimal
    alignments = pairwise2.align.globalxx(seq1, seq2)
    

**OK** :
    
    
    # Adjust according to protein type
    if protein_type == 'short_peptide':
        gap_open = -10
        gap_extend = -0.5
    elif protein_type == 'structured':
        gap_open = -5
        gap_extend = -2
    
    alignments = pairwise2.align.globalms(
        seq1, seq2,
        match=2,
        mismatch=-1,
        open=gap_open,
        extend=gap_extend
    )
    

### 3\. Dimensionality Explosion in k-mer Representation

**Problem** : k value too large causing memory issues

**NG** :
    
    
    k = 5  # 20^5 = 3,200,000 dimensions!
    kmers = extract_kmers(sequence, k=k)
    

**OK** :
    
    
    # k=2 or 3 is practical
    k = 3  # 20^3 = 8,000 dimensions
    kmers = extract_kmers(sequence, k=k)
    
    # Or use only frequent k-mers
    from collections import Counter
    kmer_counts = Counter(kmers.keys())
    top_kmers = [k for k, _ in kmer_counts.most_common(1000)]
    

### 4\. Overfitting in Machine Learning Models

**Problem** : Overfitting to training data

**NG** :
    
    
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None  # Unlimited depth
    )
    

**OK** :
    
    
    # Set regularization parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Validate with cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    

### 5\. Ignoring Sequence Length Differences

**Problem** : Directly comparing sequences of different lengths

**NG** :
    
    
    # Comparing absolute values (different lengths)
    if feature1 > feature2:
        select_protein1()
    

**OK** :
    
    
    # Normalize before comparison
    feature1_norm = feature1 / len(sequence1)
    feature2_norm = feature2 / len(sequence2)
    
    if feature1_norm > feature2_norm:
        select_protein1()
    

### 6\. Imbalanced Dataset

**Problem** : Large difference in sample size between classes

**NG** :
    
    
    # Training with imbalanced data
    model.fit(X_train, y_train)
    

**OK** :
    
    
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    
    # Weighted training
    model = RandomForestClassifier(
        class_weight=dict(zip(classes, class_weights)),
        random_state=42
    )
    model.fit(X_train, y_train)
    

* * *

## Quality Checklist

### Data Acquisition Phase

  * [ ] Sequences are correctly saved in FASTA format
  * [ ] Decided on handling policy for non-standard amino acids (X, B, Z, etc.)
  * [ ] Checked sequence length distribution (remove extremely short/long sequences)
  * [ ] Removed duplicate sequences

### BLAST Search Phase

  * [ ] E-value threshold is appropriate for application (1e-3 to 1e-10)
  * [ ] Query sequence length is sufficient (minimum 20 residues)
  * [ ] Database selection is appropriate (nr, swissprot, etc.)
  * [ ] Sufficient number of hits (minimum 3-5)

### Feature Extraction Phase

  * [ ] Amino acid composition sum equals 100%
  * [ ] k-mer k value is appropriate (2-3)
  * [ ] Hydrophobicity scale is standard (Kyte-Doolittle, etc.)
  * [ ] No missing values in features

### Machine Learning Phase

  * [ ] Train/test split uses stratified sampling
  * [ ] Cross-validation performed (k=5 or 10)
  * [ ] Feature scaling implemented (StandardScaler, etc.)
  * [ ] Multiple model performance metrics (Accuracy, F1, AUC, etc.)

### Enzyme Activity Prediction Specific

  * [ ] Activity value units are clear (kcat/Km, IC50, etc.)
  * [ ] Measurement conditions (pH, temperature) are unified
  * [ ] Outlier detection and removal
  * [ ] Consideration of experimental error

* * *

## References

  1. Altschul, S. F. et al. (1990). "Basic local alignment search tool." _Journal of Molecular Biology_ , 215(3), 403-410.

  2. Kyte, J. & Doolittle, R. F. (1982). "A simple method for displaying the hydropathic character of a protein." _Journal of Molecular Biology_ , 157(1), 105-132.

* * *

## Navigation

**[← Chapter 1](<chapter-1.html>)** | **[Chapter 3 →](<chapter-3.html>)** | **[Table of Contents](<./index.html>)**
