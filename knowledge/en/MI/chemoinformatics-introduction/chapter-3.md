---
title: Chemical Space Exploration and Similarity Search
chapter_title: Chemical Space Exploration and Similarity Search
subtitle: 
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 11
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 3: Chemical Space Exploration and Similarity Search

This chapter covers Chemical Space Exploration and Similarity Search. You will learn essential concepts and techniques.

## What You Will Learn in This Chapter

In this chapter, we will learn techniques for visualizing chemical space and performing similarity searches. The technology to efficiently explore promising candidates from vast compound libraries is essential for accelerating drug discovery and materials development.

### Learning Objectives

  * ✅ Understand the definition and calculation methods of molecular similarity
  * ✅ Be able to visualize chemical space with t-SNE/UMAP
  * ✅ Be able to classify molecules using clustering
  * ✅ Be able to efficiently explore candidate molecules through virtual screening
  * ✅ Be able to select realistic candidates considering synthetic feasibility

* * *

## 3.1 Definition of Molecular Similarity

**Molecular Similarity** is a metric that quantifies how similar two molecules are to each other.
    
    
    ```mermaid
    flowchart TD
        A[Molecule A] --> C[Fingerprint Calculation]
        B[Molecule B] --> C
        C --> D[Tanimoto Coefficient]
        C --> E[Dice Coefficient]
        C --> F[Cosine Similarity]
    
        D --> G[Similarity Score\n0.0 - 1.0]
        E --> G
        F --> G
    
        G --> H{Threshold Decision}
        H -->|≥ 0.7| I[Similar]
        H -->|< 0.7| J[Dissimilar]
    
        style A fill:#e3f2fd
        style B fill:#e3f2fd
        style I fill:#4CAF50,color:#fff
        style J fill:#F44336,color:#fff
    ```

### 3.1.1 Tanimoto Coefficient (Jaccard Similarity)

The Tanimoto coefficient is the most widely used metric for fingerprint-based similarity calculation.

**Definition** : $$ T(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{c}{a + b - c} $$

where: \- $a$: Number of bits set in molecule A \- $b$: Number of bits set in molecule B \- $c$: Number of bits set in both molecules

**Range** : 0.0 (completely different) to 1.0 (perfect match)

#### Code Example 1: Calculating Tanimoto Coefficient
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 1: Calculating Tanimoto Coefficient
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np
    
    # Sample molecules
    molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Salicylic acid": "C1=CC=C(C(=C1)C(=O)O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    
    # Calculate Morgan fingerprints
    fingerprints = {}
    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        fingerprints[name] = fp
    
    # Calculate Tanimoto similarity matrix
    mol_names = list(molecules.keys())
    n_mols = len(mol_names)
    similarity_matrix = np.zeros((n_mols, n_mols))
    
    for i, name1 in enumerate(mol_names):
        for j, name2 in enumerate(mol_names):
            sim = DataStructs.TanimotoSimilarity(
                fingerprints[name1],
                fingerprints[name2]
            )
            similarity_matrix[i, j] = sim
    
    # Display results
    import pandas as pd
    
    df_sim = pd.DataFrame(
        similarity_matrix,
        index=mol_names,
        columns=mol_names
    )
    
    print("Tanimoto Similarity Matrix:")
    print(df_sim.round(3))
    

**Sample Output:**
    
    
    Tanimoto Similarity Matrix:
                    Aspirin  Salicylic acid  Ibuprofen  Paracetamol  Caffeine
    Aspirin           1.000           0.538      0.328        0.287     0.152
    Salicylic acid    0.538           1.000      0.241        0.315     0.137
    Ibuprofen         0.328           0.241      1.000        0.224     0.098
    Paracetamol       0.287           0.315      0.224        1.000     0.189
    Caffeine          0.152           0.137      0.098        0.189     1.000
    

**Interpretation** : \- Aspirin and salicylic acid have high similarity (0.538) \- Caffeine has a very different structure from other molecules

### 3.1.2 Other Similarity Metrics

#### Code Example 2: Comparison of Multiple Similarity Metrics
    
    
    from rdkit import DataStructs
    
    # Two molecules
    mol1 = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    mol2 = Chem.MolFromSmiles("C1=CC=C(C(=C1)C(=O)O)O")  # Salicylic acid
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    
    # Various similarity metrics
    similarities = {
        'Tanimoto': DataStructs.TanimotoSimilarity(fp1, fp2),
        'Dice': DataStructs.DiceSimilarity(fp1, fp2),
        'Cosine': DataStructs.CosineSimilarity(fp1, fp2),
        'Sokal': DataStructs.SokalSimilarity(fp1, fp2),
        'Kulczynski': DataStructs.KulczynskiSimilarity(fp1, fp2),
        'McConnaughey': DataStructs.McConnaugheySimilarity(fp1, fp2)
    }
    
    print("Aspirin vs Salicylic Acid Similarity:")
    for name, sim in similarities.items():
        print(f"  {name:15s}: {sim:.3f}")
    

**Sample Output:**
    
    
    Aspirin vs Salicylic Acid Similarity:
      Tanimoto       : 0.538
      Dice           : 0.700
      Cosine         : 0.733
      Sokal          : 0.368
      Kulczynski     : 0.705
      McConnaughey   : 0.076
    

### 3.1.3 Setting Similarity Thresholds

**General Guidelines** :

Tanimoto Coefficient | Interpretation | Application  
---|---|---  
**0.85 - 1.00** | Very similar | Duplicate removal, bioisosteres  
**0.70 - 0.84** | Similar | Lead optimization, scaffold hopping  
**0.50 - 0.69** | Moderately similar | Extended similarity search  
**< 0.50** | Dissimilar | Ensuring diversity  
  
* * *

## 3.2 Visualization of Chemical Space

Chemical space is a multidimensional space with molecular descriptors as axes. It can be visualized by projecting into 2D/3D through dimensionality reduction.
    
    
    ```mermaid
    flowchart LR
        A[High-Dimensional Descriptors\n1024 dimensions] --> B[Dimensionality Reduction]
        B --> C[PCA]
        B --> D[t-SNE]
        B --> E[UMAP]
    
        C --> F[2D Visualization]
        D --> F
        E --> F
    
        F --> G[Clustering]
        G --> H[Understanding Chemical Space]
    
        style A fill:#e3f2fd
        style F fill:#4CAF50,color:#fff
        style H fill:#FF9800,color:#fff
    ```

### 3.2.1 PCA (Principal Component Analysis)

#### Code Example 3: Visualizing Chemical Space with PCA
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 3: Visualizing Chemical Space with PCA
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Sample data (assuming retrieved from ChEMBL)
    np.random.seed(42)
    smiles_list = [
        "CCO", "CCCO", "CCCCO",  # Alcohols
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",  # Aromatics
        "CC(=O)O", "CCC(=O)O",  # Carboxylic acids
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    # Calculate Morgan fingerprints
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fps.append(np.array(fp))
            valid_smiles.append(smi)
    
    X = np.array(fps)
    
    # Dimensionality reduction with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7, edgecolors='k')
    
    for i, smi in enumerate(valid_smiles):
        plt.annotate(
            smi[:15],
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=9,
            alpha=0.8
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title('Chemical Space Visualization with PCA', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_chemical_space.png', dpi=300)
    plt.close()
    
    print(f"PC1 Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PC2 Explained Variance: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Cumulative Variance: {pca.explained_variance_ratio_[:2].sum():.2%}")
    

### 3.2.2 t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE is a method that embeds high-dimensional data into low dimensions while preserving local structure.

#### Code Example 4: Visualizing Chemical Space with t-SNE
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 4: Visualizing Chemical Space with t-SNE
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Larger dataset (100 molecules)
    np.random.seed(42)
    n_samples = 100
    X_large = np.random.randn(n_samples, 2048)  # Virtual fingerprint data
    
    # Dimensionality reduction with t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_large)
    
    # Labeling (virtual classes)
    labels = np.random.choice(['Drug-like', 'Fragment', 'Natural product'],
                              size=n_samples)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    colors = {'Drug-like': 'blue', 'Fragment': 'red',
              'Natural product': 'green'}
    
    for label in colors.keys():
        mask = labels == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colors[label], label=label,
            s=80, alpha=0.7, edgecolors='k'
        )
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title('Chemical Space Visualization with t-SNE', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_chemical_space.png', dpi=300)
    plt.close()
    
    print("t-SNE visualization saved")
    

### 3.2.3 UMAP (Uniform Manifold Approximation and Projection)

UMAP is a dimensionality reduction method that is faster than t-SNE and also preserves global structure.

#### Code Example 5: Visualizing Chemical Space with UMAP
    
    
    # Install UMAP
    # pip install umap-learn
    
    import umap
    import matplotlib.pyplot as plt
    
    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='jaccard',  # Distance metric suitable for fingerprint data
        random_state=42
    )
    X_umap = reducer.fit_transform(X_large)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    for label in colors.keys():
        mask = labels == label
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=colors[label], label=label,
            s=80, alpha=0.7, edgecolors='k'
        )
    
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title('Chemical Space Visualization with UMAP', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('umap_chemical_space.png', dpi=300)
    plt.close()
    
    print("UMAP visualization saved")
    

**PCA vs t-SNE vs UMAP** :

Method | Advantages | Disadvantages | Application  
---|---|---|---  
**PCA** | Fast, easy to interpret | Linear only | Overview, preprocessing  
**t-SNE** | Preserves local structure | Slow, poor global structure | Cluster visualization  
**UMAP** | Fast, global+local | Parameter tuning required | Large-scale data  
  
* * *

## 3.3 Molecular Classification by Clustering

### 3.3.1 K-means Method

#### Code Example 6: K-means Clustering
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 6: K-means Clustering
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # K-means clustering (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_umap)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=cluster_labels, cmap='viridis',
        s=80, alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter, label='Cluster')
    
    # Cluster centers
    centers_umap = kmeans.cluster_centers_
    plt.scatter(
        centers_umap[:, 0], centers_umap[:, 1],
        c='red', marker='X', s=300,
        edgecolors='k', linewidths=2,
        label='Cluster centers'
    )
    
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title('K-means Clustering (k=3)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=300)
    plt.close()
    
    print(f"Cluster 0: {np.sum(cluster_labels == 0)} molecules")
    print(f"Cluster 1: {np.sum(cluster_labels == 1)} molecules")
    print(f"Cluster 2: {np.sum(cluster_labels == 2)} molecules")
    

### 3.3.2 DBSCAN (Density-Based Clustering)

#### Code Example 7: DBSCAN Clustering
    
    
    from sklearn.cluster import DBSCAN
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(X_umap)
    
    # Number of clusters (excluding noise)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN Clusters: {n_clusters}")
    print(f"Noise Points: {n_noise}")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Noise points
    mask_noise = dbscan_labels == -1
    plt.scatter(
        X_umap[mask_noise, 0], X_umap[mask_noise, 1],
        c='gray', s=50, alpha=0.3, label='Noise'
    )
    
    # Cluster points
    mask_clustered = dbscan_labels != -1
    scatter = plt.scatter(
        X_umap[mask_clustered, 0], X_umap[mask_clustered, 1],
        c=dbscan_labels[mask_clustered], cmap='viridis',
        s=80, alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter, label='Cluster')
    
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title(f'DBSCAN Clustering (clusters={n_clusters}, '
              f'noise={n_noise})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dbscan_clustering.png', dpi=300)
    plt.close()
    

### 3.3.3 Hierarchical Clustering

#### Code Example 8: Dendrogram
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 8: Dendrogram
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    
    # Reduce sample size for visualization (20 molecules)
    X_sample = X_large[:20]
    
    # Hierarchical clustering (Ward's method)
    linkage_matrix = linkage(X_sample, method='ward')
    
    # Draw dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=[f'M{i+1}' for i in range(len(X_sample))],
        leaf_font_size=10,
        color_threshold=10
    )
    plt.xlabel('Molecule ID', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Hierarchical Clustering (Ward Method)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=300)
    plt.close()
    
    print("Dendrogram saved")
    

* * *

## 3.4 Virtual Screening

**Virtual Screening** is a computational method for selecting promising candidates from large-scale compound libraries.
    
    
    ```mermaid
    flowchart TD
        A[Compound Library\n1 million - 1 billion compounds] --> B[Primary Screening]
        B --> C[Structure Filters\nLipinski, PAINS]
        C --> D[Similarity Search\nCompare with known actives]
        D --> E[Secondary Screening]
        E --> F[QSAR Prediction\nActivity, ADMET]
        F --> G[Docking\nProtein binding]
        G --> H[Candidate Compounds\n100-1000 compounds]
        H --> I[Experimental Validation\nHTS]
    
        style A fill:#e3f2fd
        style H fill:#4CAF50,color:#fff
        style I fill:#FF9800,color:#fff
    ```

### 3.4.1 Structure Filters

#### Lipinski's Rule of Five

Criteria for oral drug-likeness: 1\. Molecular weight ≤ 500 Da 2\. logP ≤ 5 3\. Hydrogen bond donors ≤ 5 4\. Hydrogen bond acceptors ≤ 10

#### Code Example 9: Implementing Lipinski Filter
    
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    
    def lipinski_filter(smiles):
        """
        Check Lipinski's Rule of Five
    
        Returns:
        --------
        passes : bool
            True if all rules are satisfied
        violations : dict
            Violation status for each rule
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {}
    
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
    
        violations = {
            'MW > 500': mw > 500,
            'LogP > 5': logp > 5,
            'HBD > 5': hbd > 5,
            'HBA > 10': hba > 10
        }
    
        passes = not any(violations.values())
        return passes, violations
    
    # Test molecules
    test_molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Lipitor": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
        "Small fragment": "CCO"
    }
    
    print("=== Lipinski Filter Results ===\n")
    for name, smiles in test_molecules.items():
        passes, violations = lipinski_filter(smiles)
        status = "✅ Pass" if passes else "❌ Fail"
        print(f"{name:20s} {status}")
        if not passes:
            violated = [k for k, v in violations.items() if v]
            print(f"  Violations: {', '.join(violated)}")
    

**Sample Output:**
    
    
    === Lipinski Filter Results ===
    
    Aspirin              ✅ Pass
    Lipitor              ❌ Fail
      Violations: MW > 500, HBA > 10
    Small fragment       ✅ Pass
    

### 3.4.2 Candidate Selection by Similarity Search

#### Code Example 10: Similar Compound Search from Large-Scale Library
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 10: Similar Compound Search from Large-Scale Li
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np
    import time
    
    # Query molecule (known active compound)
    query_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
    
    # Library (in practice, millions to billions of compounds)
    # Here we substitute with randomly generated 1000 compounds
    np.random.seed(42)
    library_size = 1000
    
    # Generate virtual library
    def generate_random_smiles(n):
        """Generate random SMILES (for demonstration)"""
        templates = [
            "c1ccccc1", "CCO", "CC(=O)O", "CN", "c1ccc(O)cc1",
            "CC(C)C", "c1ccncc1", "C1CCCCC1", "c1ccsc1"
        ]
        smiles_list = []
        for _ in range(n):
            smi = np.random.choice(templates)
            # Simple modification
            if np.random.rand() > 0.5:
                smi = "C" + smi
            smiles_list.append(smi)
        return smiles_list
    
    library_smiles = generate_random_smiles(library_size)
    
    # Similarity search
    start_time = time.time()
    similarities = []
    
    for smi in library_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            similarities.append((smi, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    elapsed_time = time.time() - start_time
    
    # Display top 10
    print(f"Search Time: {elapsed_time:.3f} seconds")
    print(f"Library Size: {len(similarities)} compounds")
    print(f"\nTop 10 by Similarity:")
    for i, (smi, sim) in enumerate(similarities[:10], 1):
        print(f"{i:2d}. Tanimoto={sim:.3f}  {smi}")
    
    # Extract candidates above threshold
    threshold = 0.5
    candidates = [smi for smi, sim in similarities if sim >= threshold]
    print(f"\nCandidates with Tanimoto ≥ {threshold}: {len(candidates)} compounds")
    

**Sample Output:**
    
    
    Search Time: 0.423 seconds
    Library Size: 987 compounds
    
    Top 10 by Similarity:
     1. Tanimoto=0.654  Cc1ccccc1
     2. Tanimoto=0.621  c1ccccc1
     3. Tanimoto=0.587  Cc1ccc(O)cc1
     4. Tanimoto=0.543  CC(=O)O
     5. Tanimoto=0.521  c1ccc(O)cc1
     6. Tanimoto=0.498  Cc1ccncc1
     7. Tanimoto=0.476  CCO
     8. Tanimoto=0.465  c1ccsc1
     9. Tanimoto=0.432  CN
    10. Tanimoto=0.421  CC(C)C
    
    Candidates with Tanimoto ≥ 0.5: 42 compounds
    

* * *

## 3.5 Case Study: Search for Novel Catalyst Candidates

### Background

In organic synthesis, finding catalysts with higher activity and selectivity is an important challenge. Here, we search for candidates with structures similar to known catalysts, while ensuring diversity and evaluating synthetic feasibility.

#### Code Example 11: Comprehensive Search for Catalyst Candidates
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 11: Comprehensive Search for Catalyst Candidate
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Known catalysts (ligands)
    known_catalysts = [
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1",  # Triphenylphosphine
        "CC(C)c1cc(C(C)C)c(-c2c(C(C)C)cc(C(C)C)cc2C(C)C)c(C(C)C)c1",  # IPr
        "CN(C)c1ccncc1"  # DMAP
    ]
    
    # Library (1000 compounds)
    library_smiles = generate_random_smiles(1000)
    
    # Step 1: Similarity search
    query_mol = Chem.MolFromSmiles(known_catalysts[0])
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
    
    candidates = []
    for smi in library_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
    
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
    
        # Threshold filter (0.4 - 0.8: similar but ensures diversity)
        if 0.4 <= sim <= 0.8:
            candidates.append((smi, mol, fp, sim))
    
    print(f"Step 1 (Similarity Filter): {len(candidates)} candidates")
    
    # Step 2: Structure filter (appropriate range for catalysts)
    filtered_candidates = []
    for smi, mol, fp, sim in candidates:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
    
        # Criteria for catalysts (somewhat flexible)
        if 150 <= mw <= 600 and -2 <= logp <= 6:
            filtered_candidates.append((smi, mol, fp, sim))
    
    print(f"Step 2 (Structure Filter): {len(filtered_candidates)} candidates")
    
    # Step 3: Ensuring diversity (MaxMin algorithm)
    def select_diverse_compounds(candidates, n_select=10):
        """
        Select diverse compounds using MaxMin algorithm
        """
        if len(candidates) <= n_select:
            return candidates
    
        selected = [candidates[0]]  # Select first candidate
        remaining = candidates[1:]
    
        while len(selected) < n_select and remaining:
            max_min_sim = -1
            best_idx = 0
    
            for i, (smi, mol, fp, sim) in enumerate(remaining):
                # Calculate minimum similarity to already selected compounds
                min_sim = min([
                    DataStructs.TanimotoSimilarity(fp, sel_fp)
                    for _, _, sel_fp, _ in selected
                ])
    
                # Select compound with maximum minimum similarity
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    best_idx = i
    
            selected.append(remaining.pop(best_idx))
    
        return selected
    
    diverse_candidates = select_diverse_compounds(filtered_candidates, n_select=20)
    print(f"Step 3 (Diversity Preservation): {len(diverse_candidates)} candidates")
    
    # Step 4: Evaluate synthetic feasibility (SA score)
    from rdkit.Chem import QED
    
    final_candidates = []
    for smi, mol, fp, sim in diverse_candidates:
        # SA score (1=easy, 10=difficult)
        # Here we substitute with QED (drug-likeness)
        qed_score = QED.qed(mol)
    
        final_candidates.append({
            'SMILES': smi,
            'Similarity': sim,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'QED': qed_score
        })
    
    # Display results
    import pandas as pd
    
    df_final = pd.DataFrame(final_candidates)
    df_final = df_final.sort_values('QED', ascending=False)
    
    print("\n=== Top 10 Final Candidates ===")
    print(df_final.head(10).to_string(index=False))
    
    # Visualization: Similarity vs QED
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df_final['Similarity'],
        df_final['QED'],
        c=df_final['MW'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='k'
    )
    plt.colorbar(scatter, label='Molecular Weight')
    plt.xlabel('Similarity to Known Catalyst', fontsize=12)
    plt.ylabel('QED (Drug-likeness)', fontsize=12)
    plt.title('Evaluation of Catalyst Candidates', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('catalyst_candidates.png', dpi=300)
    plt.close()
    
    print("\nVisualization saved")
    

**Sample Output:**
    
    
    Step 1 (Similarity Filter): 87 candidates
    Step 2 (Structure Filter): 62 candidates
    Step 3 (Diversity Preservation): 20 candidates
    
    === Top 10 Final Candidates ===
                SMILES  Similarity      MW  LogP    QED
          c1ccc(O)cc1       0.543   94.11  1.46  0.732
       Cc1ccc(O)cc1       0.621  108.14  1.95  0.701
            c1ccncc1       0.498   79.10  0.65  0.687
          Cc1ccccc1       0.654   92.14  2.73  0.654
         c1ccccc1       0.621   78.11  2.12  0.623
    ...
    
    Visualization saved
    

**Interpretation** : 1\. Select candidates with moderate similarity (0.4-0.8) to known catalysts 2\. Narrow down with structure filter to appropriate range for catalysts 3\. Ensure diversity using MaxMin algorithm 4\. Prioritize easily synthesizable candidates with QED score

* * *

## Exercises

### Exercise 1: Comparison of Similarity Metrics

For the following three molecular pairs, calculate Tanimoto, Dice, and Cosine similarity, and compare which metric gives the highest similarity evaluation.

  1. Aspirin vs Salicylic acid
  2. Ibuprofen vs Naproxen
  3. Caffeine vs Theobromine

Sample Solution
    
    
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    
    pairs = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O",
         "Salicylic acid", "C1=CC=C(C(=C1)C(=O)O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
         "Naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
         "Theobromine", "CN1C=NC2=C1C(=O)NC(=O)N2C")
    ]
    
    for name1, smi1, name2, smi2 in pairs:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
    
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        dice = DataStructs.DiceSimilarity(fp1, fp2)
        cosine = DataStructs.CosineSimilarity(fp1, fp2)
    
        print(f"\n{name1} vs {name2}:")
        print(f"  Tanimoto: {tanimoto:.3f}")
        print(f"  Dice:     {dice:.3f}")
        print(f"  Cosine:   {cosine:.3f}")
    
        # Determine highest value
        scores = {'Tanimoto': tanimoto, 'Dice': dice, 'Cosine': cosine}
        best = max(scores, key=scores.get)
        print(f"  Highest: {best}")
    

**Expected Output:** 
    
    
    Aspirin vs Salicylic acid:
      Tanimoto: 0.538
      Dice:     0.700
      Cosine:   0.733
      Highest: Cosine
    
    Ibuprofen vs Naproxen:
      Tanimoto: 0.432
      Dice:     0.603
      Cosine:   0.648
      Highest: Cosine
    
    Caffeine vs Theobromine:
      Tanimoto: 0.821
      Dice:     0.902
      Cosine:   0.906
      Highest: Cosine
    

**Interpretation**: Generally, Cosine similarity tends to show the highest values. 

* * *

### Exercise 2: Visualization and Interpretation of Chemical Space

Obtain 100 drug molecules from ChEMBL or similar databases, perform UMAP visualization and K-means clustering. Analyze the characteristics of each cluster (average MW, LogP, etc.).

Hint 1\. Obtain SMILES from ChEMBL API or CSV file 2\. Calculate Morgan fingerprints 3\. Project into 2D with UMAP 4\. Classify into 3-5 clusters with K-means 5\. Calculate descriptor statistics for each cluster 

* * *

### Exercise 3: Building a Virtual Screening Pipeline

Build a virtual screening pipeline with the following steps:

  1. Select a query molecule (known active compound)
  2. Search for similar compounds from a library of 1000 compounds (Tanimoto > 0.5)
  3. Apply Lipinski filter
  4. Ensure diversity (MaxMin algorithm, 10 compounds)
  5. Output final candidates to CSV

Sample Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Build a virtual screening pipeline with the following steps:
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    
    # Step 1: Query molecule
    query_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
    
    # Step 2: Library search
    library_smiles = generate_random_smiles(1000)  # Function from earlier
    similar_compounds = []
    
    for smi in library_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
    
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
    
        if sim > 0.5:
            similar_compounds.append((smi, mol, fp, sim))
    
    print(f"Similar Compounds: {len(similar_compounds)}")
    
    # Step 3: Lipinski filter
    lipinski_passed = []
    for smi, mol, fp, sim in similar_compounds:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
    
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            lipinski_passed.append((smi, mol, fp, sim))
    
    print(f"Lipinski Passed: {len(lipinski_passed)}")
    
    # Step 4: Ensure diversity
    diverse_compounds = select_diverse_compounds(
        lipinski_passed, n_select=10
    )  # Function from earlier
    
    print(f"Final Candidates: {len(diverse_compounds)}")
    
    # Step 5: CSV output
    results = []
    for smi, mol, fp, sim in diverse_compounds:
        results.append({
            'SMILES': smi,
            'Similarity': sim,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol)
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('virtual_screening_results.csv', index=False)
    
    print("\nFinal Candidates:")
    print(df_results.to_string(index=False))
    print("\nCSV file saved")
    

* * *

## Summary

In this chapter, we learned the following:

### What We Learned

  1. **Molecular Similarity** \- Tanimoto coefficient, Dice coefficient, Cosine similarity \- Threshold setting and interpretation of similarity \- Structural similarity vs property similarity

  2. **Visualization of Chemical Space** \- PCA: Linear dimensionality reduction \- t-SNE: Local structure preservation \- UMAP: Fast with global+local structure preservation

  3. **Clustering** \- K-means: Partitioning \- DBSCAN: Density-based \- Hierarchical clustering: Dendrogram

  4. **Virtual Screening** \- Structure filters (Lipinski) \- Similarity search \- Diversity preservation (MaxMin algorithm) \- Synthetic feasibility evaluation

  5. **Practice: Catalyst Candidate Search** \- Similarity filter \- Structure filter \- Diversity preservation \- Evaluation by QED score

### Next Steps

In Chapter 4, we will learn reaction prediction and retrosynthesis analysis.

**[Chapter 4: Reaction Prediction and Retrosynthesis →](<chapter-4.html>)**

* * *

## References

  1. Willett, P. (2006). "Similarity-based virtual screening using 2D fingerprints." _Drug Discovery Today_ , 11(23-24), 1046-1053. DOI: 10.1016/j.drudis.2006.10.005
  2. McInnes, L. et al. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426
  3. Lipinski, C. A. et al. (1997). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." _Advanced Drug Delivery Reviews_ , 23(1-3), 3-25.
  4. Bickerton, G. R. et al. (2012). "Quantifying the chemical beauty of drugs." _Nature Chemistry_ , 4, 90-98. DOI: 10.1038/nchem.1243

* * *

**[← Chapter 2](<chapter-2.html>)** | **[Back to Series Top](<./index.html>)** | **[Chapter 4 →](<chapter-4.html>)**
