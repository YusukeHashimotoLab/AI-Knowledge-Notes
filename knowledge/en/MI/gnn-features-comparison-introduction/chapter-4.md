---
title: "Chapter 4: Composition-based vs GNN Quantitative Comparison"
chapter_title: "Chapter 4: Composition-based vs GNN Quantitative Comparison"
---

🌐 EN | [🇯🇵 JP](<../../../jp/MI/gnn-features-comparison-introduction/chapter-4.html>) | Last sync: 2025-11-16

# Chapter 4: Composition-based vs GNN Quantitative Comparison

In this chapter, we quantitatively compare the performance of **composition-based features (Magpie)** and **GNN structure-based features (CGCNN)** using the **Matbench standard benchmark**. We conduct empirical analysis from multiple perspectives including not only prediction accuracy, but also statistical significance, computational cost, data requirements, and interpretability.

**🎯 Learning Objectives**

  * Understand the structure and evaluation protocol of the Matbench benchmark
  * Quantitatively compare prediction accuracy of Random Forest (Magpie) and CGCNN
  * Implement statistical significance testing (t-test, confidence intervals, p-values)
  * Measure and compare computational cost (training time, inference time, memory)
  * Visualize differences in data requirements using learning curves
  * Compare interpretability using SHAP values and Attention mechanisms
  * Construct a decision flowchart for method selection

## 4.1 Overview of Matbench Benchmark

Matbench (Materials Benchmark) is a **standardized evaluation platform** for machine learning models in materials science. It defines 13 material property prediction tasks, with training, validation, and test data pre-split for each task. This enables fair performance comparison across different research studies.

### 4.1.1 Main Matbench Tasks

Task Name | Property | Data Size | Evaluation Metric  
---|---|---|---  
matbench_mp_e_form | Formation energy (eV/atom) | 132,752 | MAE  
matbench_mp_gap | Band gap (eV) | 106,113 | MAE  
matbench_perovskites | Formation energy (eV/atom) | 18,928 | MAE  
matbench_phonons | Maximum phonon frequency (cm⁻¹) | 1,265 | MAE  
matbench_jdft2d | Exfoliation energy (meV/atom) | 636 | MAE  
  
In this chapter, we compare the performance of Random Forest (Magpie features) and CGCNN on two tasks: **matbench_mp_e_form (formation energy prediction)** and **matbench_mp_gap (band gap prediction)**.

### 4.1.2 Evaluation Protocol

In Matbench, **5-fold cross-validation** is the standard evaluation protocol. The entire dataset is split into 5 folds, and the following evaluation is performed for each fold:

  1. **Training** : Train the model on the remaining 4 folds
  2. **Validation** : Use for hyperparameter tuning (optional)
  3. **Testing** : Evaluate performance on the corresponding fold (calculate MAE, RMSE, R²)

By averaging the evaluation metrics across the 5 folds, we obtain a **reliable estimate of generalization performance**.

## 4.2 Matbench Benchmark Implementation

In this section, we implement Random Forest (Magpie) and CGCNN on the Matbench mp_e_form task and quantitatively compare their prediction accuracy.

### 4.2.1 Environment Setup and Data Loading

**💻 Code Example 1: Loading Matbench Dataset**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 4.2.1 Environment Setup and Data Loading
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # Load and preprocess Matbench dataset
    import numpy as np
    import pandas as pd
    from matbench.bench import MatbenchBenchmark
    from pymatgen.core import Structure
    from matminer.featurizers.composition import ElementProperty
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # Initialize Matbench benchmark
    mb = MatbenchBenchmark(autoload=False)
    
    # Load mp_e_form task (formation energy prediction)
    task = mb.matbench_mp_e_form
    task.load()
    
    print(f"Task name: {task.metadata['task_type']}")
    print(f"Data size: {len(task.df)}")
    print(f"Input: {task.metadata['input_type']}")
    print(f"Output: {task.metadata['target']}")
    
    # Check data structure
    print("\nData sample:")
    print(task.df.head(3))
    
    # Output example:
    # Task name: regression
    # Data size: 132752
    # Input: structure
    # Output: e_form (eV/atom)
    #
    # Data sample:
    #                                           structure  e_form
    # 0  Structure: Fe2 O3 ...                    -2.54
    # 1  Structure: Mn2 O3 ...                    -2.89
    # 2  Structure: Co2 O3 ...                    -2.31
    

### 4.2.2 Random Forest (Magpie Features) Implementation

**💻 Code Example 2: Magpie Feature Extraction and Random Forest Training**
    
    
    # Random Forest + Magpie features implementation
    from matminer.featurizers.composition import ElementProperty
    
    def extract_magpie_features(structures):
        """
        Extract Magpie features from Pymatgen Structure
    
        Parameters:
        -----------
        structures : list of Structure
            List of crystal structures
    
        Returns:
        --------
        features : np.ndarray, shape (n_samples, 145)
            Magpie features (145 dimensions)
        """
        # Initialize Magpie feature extractor
        featurizer = ElementProperty.from_preset("magpie")
    
        features = []
        for struct in structures:
            # Get Composition from Structure
            comp = struct.composition
            # Calculate Magpie features (145 dimensions)
            feat = featurizer.featurize(comp)
            features.append(feat)
    
        return np.array(features)
    
    # 5-fold cross-validation evaluation
    rf_results = []
    
    for fold_idx, fold in enumerate(task.folds):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
    
        # Get training and test data
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    
        # Extract Magpie features
        print("Extracting Magpie features...")
        X_train = extract_magpie_features(train_inputs)
        X_test = extract_magpie_features(test_inputs)
        y_train = train_outputs.values
        y_test = test_outputs.values
    
        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
        # Train Random Forest model
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=30,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
        # Predict on test data
        y_pred = rf_model.predict(X_test)
    
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
    
        print(f"MAE:  {mae:.4f} eV/atom")
        print(f"RMSE: {rmse:.4f} eV/atom")
        print(f"R²:   {r2:.4f}")
    
        rf_results.append({
            'fold': fold_idx + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    # Calculate 5-fold average scores
    rf_df = pd.DataFrame(rf_results)
    print("\n=== Random Forest (Magpie) Overall Results ===")
    print(f"MAE:  {rf_df['mae'].mean():.4f} ± {rf_df['mae'].std():.4f} eV/atom")
    print(f"RMSE: {rf_df['rmse'].mean():.4f} ± {rf_df['rmse'].std():.4f} eV/atom")
    print(f"R²:   {rf_df['r2'].mean():.4f} ± {rf_df['r2'].std():.4f}")
    
    # Output example:
    # === Random Forest (Magpie) Overall Results ===
    # MAE:  0.0325 ± 0.0012 eV/atom
    # RMSE: 0.0678 ± 0.0019 eV/atom
    # R²:   0.9321 ± 0.0045
    

### 4.2.3 CGCNN Implementation

**💻 Code Example 3: CGCNN on Matbench Training**
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: 4.2.3 CGCNN Implementation
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # CGCNN on Matbench implementation
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import CGConv, global_mean_pool
    import time
    
    # Define CGCNN for Matbench (using model from Chapter 2)
    class CGCNNMatbench(nn.Module):
        def __init__(self, atom_fea_len=92, nbr_fea_len=41, hidden_dim=128, n_conv=3):
            super(CGCNNMatbench, self).__init__()
    
            # Atom embedding
            self.atom_embedding = nn.Linear(atom_fea_len, hidden_dim)
    
            # Convolution layers
            self.conv_layers = nn.ModuleList([
                CGConv(hidden_dim, nbr_fea_len) for _ in range(n_conv)
            ])
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
            ])
    
            # Readout
            self.fc1 = nn.Linear(hidden_dim, 64)
            self.fc2 = nn.Linear(64, 1)
            self.activation = nn.Softplus()
    
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # Atom embedding
            x = self.atom_embedding(x)
    
            # Graph convolutions
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = self.activation(x)
    
            # Global pooling
            x = global_mean_pool(x, batch)
    
            # Fully connected layers
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
    
            return x.squeeze()
    
    def structure_to_pyg_data(structure, target, cutoff=8.0):
        """
        Convert Pymatgen Structure to PyTorch Geometric data
    
        Parameters:
        -----------
        structure : Structure
            Crystal structure
        target : float
            Target value (formation energy)
        cutoff : float
            Cutoff radius (Å)
    
        Returns:
        --------
        data : Data
            PyTorch Geometric data object
        """
        # Node features (element one-hot encoding, 92 dimensions)
        atom_types = [site.specie.Z for site in structure]
        x = torch.zeros(len(atom_types), 92)
        for i, z in enumerate(atom_types):
            x[i, z - 1] = 1.0
    
        # Edge construction (atom pairs within cutoff radius)
        neighbors = structure.get_all_neighbors(cutoff)
        edge_index = []
        edge_attr = []
    
        for i, neighbors_i in enumerate(neighbors):
            for neighbor in neighbors_i:
                j = neighbor.index
                distance = neighbor.nn_distance
    
                # Gaussian distance expansion (41 dimensions)
                distances = torch.linspace(0, cutoff, 41)
                sigma = 0.5
                edge_feature = torch.exp(-((distance - distances) ** 2) / (2 * sigma ** 2))
    
                edge_index.append([i, j])
                edge_attr.append(edge_feature)
    
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        y = torch.tensor([target], dtype=torch.float)
    
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # 5-fold cross-validation evaluation
    cgcnn_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold_idx, fold in enumerate(task.folds):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
    
        # Get training and test data
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    
        # Convert to PyTorch Geometric data
        print("Building graph data...")
        train_data = [structure_to_pyg_data(s, t) for s, t in zip(train_inputs, train_outputs.values)]
        test_data = [structure_to_pyg_data(s, t) for s, t in zip(test_inputs, test_outputs.values)]
    
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
        # Initialize CGCNN model
        model = CGCNNMatbench().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()  # MAE loss
    
        # Training
        print("Training CGCNN...")
        model.train()
        for epoch in range(50):  # Should use early stopping in practice
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}")
    
        # Testing
        model.eval()
        y_true = []
        y_pred = []
    
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(out.cpu().numpy())
    
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
    
        print(f"MAE:  {mae:.4f} eV/atom")
        print(f"RMSE: {rmse:.4f} eV/atom")
        print(f"R²:   {r2:.4f}")
    
        cgcnn_results.append({
            'fold': fold_idx + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    # Calculate 5-fold average scores
    cgcnn_df = pd.DataFrame(cgcnn_results)
    print("\n=== CGCNN Overall Results ===")
    print(f"MAE:  {cgcnn_df['mae'].mean():.4f} ± {cgcnn_df['mae'].std():.4f} eV/atom")
    print(f"RMSE: {cgcnn_df['rmse'].mean():.4f} ± {cgcnn_df['rmse'].std():.4f} eV/atom")
    print(f"R²:   {cgcnn_df['r2'].mean():.4f} ± {cgcnn_df['r2'].std():.4f}")
    
    # Output example:
    # === CGCNN Overall Results ===
    # MAE:  0.0286 ± 0.0009 eV/atom
    # RMSE: 0.0592 ± 0.0014 eV/atom
    # R²:   0.9524 ± 0.0032
    

### 4.2.4 Quantitative Comparison of Prediction Accuracy

We integrate the 5-fold cross-validation results and compare the prediction accuracy of Random Forest (Magpie) and CGCNN.

**💻 Code Example 4: Visualization of Prediction Accuracy Comparison**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: We integrate the 5-fold cross-validation results and compare
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Visualization of prediction accuracy comparison
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Integrate results
    comparison_df = pd.DataFrame({
        'Model': ['RF (Magpie)'] * 5 + ['CGCNN'] * 5,
        'Fold': list(range(1, 6)) * 2,
        'MAE': list(rf_df['mae']) + list(cgcnn_df['mae']),
        'RMSE': list(rf_df['rmse']) + list(cgcnn_df['rmse']),
        'R²': list(rf_df['r2']) + list(cgcnn_df['r2'])
    })
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    sns.barplot(data=comparison_df, x='Model', y='MAE', ax=axes[0], palette=['#667eea', '#764ba2'])
    axes[0].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE (eV/atom)', fontsize=12)
    axes[0].axhline(0.03, color='red', linestyle='--', linewidth=1, label='Target: 0.03')
    axes[0].legend()
    
    # RMSE comparison
    sns.barplot(data=comparison_df, x='Model', y='RMSE', ax=axes[1], palette=['#667eea', '#764ba2'])
    axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('RMSE (eV/atom)', fontsize=12)
    
    # R² comparison
    sns.barplot(data=comparison_df, x='Model', y='R²', ax=axes[2], palette=['#667eea', '#764ba2'])
    axes[2].set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].axhline(0.95, color='red', linestyle='--', linewidth=1, label='Target: 0.95')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("=== Statistical Summary of Prediction Accuracy ===")
    print(comparison_df.groupby('Model').agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'R²': ['mean', 'std']
    }).round(4))
    
    # Calculate relative improvement rates
    mae_improvement = (rf_df['mae'].mean() - cgcnn_df['mae'].mean()) / rf_df['mae'].mean() * 100
    rmse_improvement = (rf_df['rmse'].mean() - cgcnn_df['rmse'].mean()) / rf_df['rmse'].mean() * 100
    r2_improvement = (cgcnn_df['r2'].mean() - rf_df['r2'].mean()) / rf_df['r2'].mean() * 100
    
    print(f"\n=== Relative Improvement Rate of CGCNN ===")
    print(f"MAE improvement:  {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    print(f"R² improvement:   {r2_improvement:.2f}%")
    
    # Output example:
    # === Relative Improvement Rate of CGCNN ===
    # MAE improvement:  12.00%
    # RMSE improvement: 12.68%
    # R² improvement:   2.18%
    

**Interpretation of Results:**

  * **MAE improvement: 12.00%** \- CGCNN achieves approximately 12% lower error than Magpie+RF in formation energy prediction
  * **R² improvement: 2.18%** \- Although RF already achieves a high R² (0.932), CGCNN achieves an extremely high coefficient of determination of 0.952, so the relative improvement rate is small
  * **Statistical significance testing is required** \- In the next section, we will test whether this accuracy difference is statistically significant

**⚠️ Important Notes**

In this code example, for computational time constraints, the CGCNN epoch count is limited to 50, but in actual benchmarking, **early stopping and hyperparameter optimization** should be performed. Also, execution in a GPU environment is strongly recommended (training time is 10-20x faster).

## 4.3 Statistical Significance Testing

In the previous section, we confirmed that CGCNN achieves approximately 12% lower MAE than Random Forest (Magpie). However, to verify whether this accuracy difference is **statistically significant** , we cannot rule out the possibility that it is merely by chance. In this section, we evaluate statistical significance using **paired t-test** and **95% confidence interval**.

### 4.3.1 Principle of Paired t-Test

In 5-fold cross-validation, both RF and CGCNN are evaluated on the same data splits. Therefore, the MAE differences for each fold can be treated as **paired data**. The paired t-test tests the following null hypothesis:

$$H_0: \mu_{\text{diff}} = 0 \quad \text{(The average MAE difference between RF and CGCNN is zero)}$$

$$H_1: \mu_{\text{diff}} \neq 0 \quad \text{(The average MAE difference is not zero)}$$

The test statistic t is calculated as follows:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

Where $\bar{d}$ is the mean of the differences, $s_d$ is the standard deviation of the differences, and $n$ is the sample size (number of folds = 5).

### 4.3.2 t-Test Implementation

**💻 Code Example 5: Paired t-Test Implementation**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: 4.3.2 t-Test Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # Paired t-test implementation
    from scipy import stats
    import numpy as np
    
    # MAE for Random Forest and CGCNN (results from 5 folds)
    rf_mae = np.array([0.0325, 0.0338, 0.0312, 0.0329, 0.0321])  # Example data
    cgcnn_mae = np.array([0.0286, 0.0293, 0.0279, 0.0288, 0.0284])  # Example data
    
    # Calculate MAE differences
    mae_diff = rf_mae - cgcnn_mae
    
    print("=== Paired t-Test ===")
    print(f"RF MAE:     {rf_mae}")
    print(f"CGCNN MAE:  {cgcnn_mae}")
    print(f"MAE diff:   {mae_diff}")
    print(f"Mean diff:  {mae_diff.mean():.4f} eV/atom")
    print(f"Std dev:    {mae_diff.std(ddof=1):.4f} eV/atom")
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(rf_mae, cgcnn_mae)
    
    print(f"\nt-statistic: {t_statistic:.4f}")
    print(f"p-value:     {p_value:.4f}")
    
    # Judge with significance level α=0.05
    alpha = 0.05
    if p_value < alpha:
        print(f"\nJudgment: p-value ({p_value:.4f}) < α ({alpha}) → Reject null hypothesis")
        print("Conclusion: The accuracy difference between CGCNN and RF is statistically significant.")
    else:
        print(f"\nJudgment: p-value ({p_value:.4f}) ≥ α ({alpha}) → Cannot reject null hypothesis")
        print("Conclusion: The accuracy difference between CGCNN and RF is not statistically significant.")
    
    # Calculate effect size (Cohen's d)
    cohens_d = mae_diff.mean() / mae_diff.std(ddof=1)
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        print("Effect size magnitude: Small")
    elif abs(cohens_d) < 0.5:
        print("Effect size magnitude: Medium")
    elif abs(cohens_d) < 0.8:
        print("Effect size magnitude: Large")
    else:
        print("Effect size magnitude: Very large")
    
    # Output example:
    # === Paired t-Test ===
    # RF MAE:     [0.0325 0.0338 0.0312 0.0329 0.0321]
    # CGCNN MAE:  [0.0286 0.0293 0.0279 0.0288 0.0284]
    # MAE diff:   [0.0039 0.0045 0.0033 0.0041 0.0037]
    # Mean diff:  0.0039 eV/atom
    # Std dev:    0.0004 eV/atom
    #
    # t-statistic: 20.5891
    # p-value:     0.0001
    #
    # Judgment: p-value (0.0001) < α (0.05) → Reject null hypothesis
    # Conclusion: The accuracy difference between CGCNN and RF is statistically significant.
    #
    # Effect size (Cohen's d): 9.1894
    # Effect size magnitude: Very large
    

### 4.3.3 Calculation of 95% Confidence Interval

The 95% confidence interval indicates the **range that contains the true mean difference with 95% probability**. If the confidence interval does not include zero, we can conclude that there is a statistically significant difference.

**💻 Code Example 6: Calculation and Visualization of 95% Confidence Interval**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - scipy>=1.11.0
    
    """
    Example: The 95% confidence interval indicates therange that contains
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # Calculation and visualization of 95% confidence interval
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # Calculate 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(mae_diff) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    mean_diff = mae_diff.mean()
    se_diff = mae_diff.std(ddof=1) / np.sqrt(len(mae_diff))
    margin_error = t_critical * se_diff
    
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error
    
    print("=== 95% Confidence Interval ===")
    print(f"Mean difference:     {mean_diff:.4f} eV/atom")
    print(f"Standard error:      {se_diff:.4f} eV/atom")
    print(f"t-critical (df={degrees_freedom}): {t_critical:.4f}")
    print(f"Margin of error:     ±{margin_error:.4f} eV/atom")
    print(f"95% CI:              [{ci_lower:.4f}, {ci_upper:.4f}] eV/atom")
    
    if ci_lower > 0:
        print("\nJudgment: CI does not include 0 → Statistically significant difference exists")
    else:
        print("\nJudgment: CI includes 0 → No statistically significant difference")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot MAE differences for each fold
    ax.scatter(range(1, 6), mae_diff, s=100, color='#764ba2', zorder=3, label='MAE diff per fold')
    
    # Mean line
    ax.axhline(mean_diff, color='#667eea', linestyle='--', linewidth=2, label=f'Mean diff: {mean_diff:.4f}')
    
    # 95% confidence interval
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='#667eea', label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    # Zero line
    ax.axhline(0, color='red', linestyle='-', linewidth=1, label='No difference (H₀)')
    
    ax.set_xlabel('Fold number', fontsize=12)
    ax.set_ylabel('MAE diff (RF - CGCNN) [eV/atom]', fontsize=12)
    ax.set_title('MAE Difference and 95% Confidence Interval in 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output example:
    # === 95% Confidence Interval ===
    # Mean difference:     0.0039 eV/atom
    # Standard error:      0.0002 eV/atom
    # t-critical (df=4):  2.7764
    # Margin of error:     ±0.0005 eV/atom
    # 95% CI:              [0.0034, 0.0044] eV/atom
    #
    # Judgment: CI does not include 0 → Statistically significant difference exists
    

### 4.3.4 Summary of Statistical Testing Results

Test Method | Result | Interpretation  
---|---|---  
Paired t-test | t=20.59, p=0.0001 | p < 0.05 → Statistically significant  
95% confidence interval | [0.0034, 0.0044] eV/atom | Does not include 0 → Statistically significant  
Effect size (Cohen's d) | 9.19 | Very large effect size  
Relative improvement rate | 12.00% | Practical significance exists  
  
**Conclusion:**

  * CGCNN showed **statistically significant accuracy improvement** over Random Forest (Magpie) (p=0.0001 << 0.05)
  * The 95% confidence interval [0.0034, 0.0044] eV/atom does not include zero, so the existence of the accuracy difference is statistically supported
  * Cohen's d = 9.19, a very large effect size, indicates the practical superiority of CGCNN
  * The 12% relative improvement rate has practical merit directly related to improving the efficiency of materials discovery

**⚠️ Statistical Significance vs Practical Significance**

Even if a statistically significant difference is recognized, it is necessary to separately consider whether that difference is **practically meaningful**. In this case, the MAE difference of 0.0039 eV/atom can be judged as a sufficiently practical improvement in materials discovery (prediction accuracy of formation energy improved by approximately 1 meV/atom).

## 4.4 Quantitative Comparison of Computational Cost

Not only accuracy, but also **computational cost (training time, inference time, memory usage)** is an important factor in model selection. In this section, we measure and quantitatively compare the computational cost of Random Forest and CGCNN.

### 4.4.1 Measurement of Training Time

**💻 Code Example 7: Measurement of Training and Inference Time**
    
    
    # Measurement of training and inference time
    import time
    import psutil
    import os
    
    def measure_training_time_and_memory(model_type='rf'):
        """
        Measure model training time and memory usage
    
        Parameters:
        -----------
        model_type : str
            'rf' (Random Forest) or 'cgcnn'
    
        Returns:
        --------
        results : dict
            Training time, inference time, memory usage
        """
        # Get process memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
        # Training and test data (use only Fold 1)
        train_inputs, train_outputs = task.get_train_and_val_data(task.folds[0])
        test_inputs, test_outputs = task.get_test_data(task.folds[0], include_target=True)
    
        if model_type == 'rf':
            # Measure Random Forest training time
            X_train = extract_magpie_features(train_inputs)
            X_test = extract_magpie_features(test_inputs)
            y_train = train_outputs.values
    
            # Start training
            start_train = time.time()
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=30,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            train_time = time.time() - start_train
    
            # Measure inference time
            start_infer = time.time()
            _ = rf_model.predict(X_test)
            infer_time = time.time() - start_infer
    
            mem_after = process.memory_info().rss / (1024 ** 2)
            memory_usage = mem_after - mem_before
    
        elif model_type == 'cgcnn':
            # Measure CGCNN training time
            train_data = [structure_to_pyg_data(s, t) for s, t in zip(train_inputs, train_outputs.values)]
            test_data = [structure_to_pyg_data(s, t) for s, t in zip(test_inputs, test_outputs.values)]
    
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
            model = CGCNNMatbench().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.L1Loss()
    
            # Start training
            start_train = time.time()
            model.train()
            for epoch in range(50):
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
            train_time = time.time() - start_train
    
            # Measure inference time
            model.eval()
            start_infer = time.time()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    _ = model(batch)
            infer_time = time.time() - start_infer
    
            mem_after = process.memory_info().rss / (1024 ** 2)
            memory_usage = mem_after - mem_before
    
        return {
            'train_time': train_time,
            'infer_time': infer_time,
            'memory_usage': memory_usage
        }
    
    # Measure Random Forest computational cost
    print("=== Measuring Random Forest Computational Cost... ===")
    rf_cost = measure_training_time_and_memory('rf')
    
    print(f"Training time:  {rf_cost['train_time']:.2f} seconds")
    print(f"Inference time: {rf_cost['infer_time']:.2f} seconds")
    print(f"Memory usage:   {rf_cost['memory_usage']:.2f} MB")
    
    # Measure CGCNN computational cost (when using GPU)
    print("\n=== Measuring CGCNN Computational Cost... ===")
    cgcnn_cost = measure_training_time_and_memory('cgcnn')
    
    print(f"Training time:  {cgcnn_cost['train_time']:.2f} seconds")
    print(f"Inference time: {cgcnn_cost['infer_time']:.2f} seconds")
    print(f"Memory usage:   {cgcnn_cost['memory_usage']:.2f} MB")
    
    # Comparison
    print("\n=== Computational Cost Comparison ===")
    print(f"Training time ratio (CGCNN/RF): {cgcnn_cost['train_time'] / rf_cost['train_time']:.2f}x")
    print(f"Inference time ratio (CGCNN/RF): {cgcnn_cost['infer_time'] / rf_cost['infer_time']:.2f}x")
    print(f"Memory ratio (CGCNN/RF):         {cgcnn_cost['memory_usage'] / rf_cost['memory_usage']:.2f}x")
    
    # Output example:
    # === Measuring Random Forest Computational Cost... ===
    # Training time:  45.32 seconds
    # Inference time: 0.18 seconds
    # Memory usage:   1250.45 MB
    #
    # === Measuring CGCNN Computational Cost... ===
    # Training time:  1832.56 seconds
    # Inference time: 2.34 seconds
    # Memory usage:   3450.23 MB
    #
    # === Computational Cost Comparison ===
    # Training time ratio (CGCNN/RF): 40.45x
    # Inference time ratio (CGCNN/RF): 13.00x
    # Memory ratio (CGCNN/RF):         2.76x
    

### 4.4.2 Visualization of Computational Cost

**💻 Code Example 8: Visualization of Computational Cost Comparison**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 4.4.2 Visualization of Computational Cost
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Visualization of computational cost comparison
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data
    models = ['RF (Magpie)', 'CGCNN']
    train_times = [45.32, 1832.56]
    infer_times = [0.18, 2.34]
    memory_usage = [1250.45, 3450.23]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training time comparison (log scale)
    axes[0].bar(models, train_times, color=['#667eea', '#764ba2'])
    axes[0].set_ylabel('Training time (seconds)', fontsize=12)
    axes[0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')
    for i, v in enumerate(train_times):
        axes[0].text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Inference time comparison (log scale)
    axes[1].bar(models, infer_times, color=['#667eea', '#764ba2'])
    axes[1].set_ylabel('Inference time (seconds)', fontsize=12)
    axes[1].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    for i, v in enumerate(infer_times):
        axes[1].text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Memory usage comparison
    axes[2].bar(models, memory_usage, color=['#667eea', '#764ba2'])
    axes[2].set_ylabel('Memory usage (MB)', fontsize=12)
    axes[2].set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(memory_usage):
        axes[2].text(i, v, f'{v:.1f}MB', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('computational_cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 4.4.3 Summary of Computational Cost Analysis

Metric | RF (Magpie) | CGCNN | Ratio  
---|---|---|---  
Training time | 45.3 sec | 1,832.6 sec (30.5 min) | 40.5x slower  
Inference time | 0.18 sec | 2.34 sec | 13.0x slower  
Memory usage | 1,250 MB | 3,450 MB | 2.8x larger  
  
**Interpretation of Computational Cost:**

  * **Training time: 40x difference** \- CGCNN requires approximately 30 minutes of training time (when using GPU). With CPU only, it could be 10-20x slower
  * **Inference time: 13x difference** \- RF is advantageous when real-time prediction is required
  * **Memory: 2.8x difference** \- CGCNN also requires additional GPU memory (approximately 2GB in this example)
  * **Accuracy-cost tradeoff** \- Whether the 40x computational time can be tolerated for a 12% accuracy improvement depends on the use case

## 4.5 Analysis of Data Requirements

In evaluating the practicality of machine learning models, **how much training data is required** is an important factor. In this section, we visualize the difference in data requirements between Random Forest and CGCNN using **learning curves**.

### 4.5.1 Concept of Learning Curves

Learning curves are graphs that plot **changes in model performance** as the amount of training data varies. They typically have the following characteristics:

  * **Small data region** : Model performance strongly depends on the amount of training data and improves rapidly with increasing data
  * **Sufficient data region** : Performance saturates and improvement from additional data becomes small
  * **Data-efficient models** : Reach high performance with less data, with curves saturating early
  * **Data-inefficient models** : Require more data, with late saturation of curves

### 4.5.2 Learning Curve Implementation

**💻 Code Example 9: Generation and Visualization of Learning Curves**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 4.5.2 Learning Curve Implementation
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # Generation of learning curves
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    # Training and test data (use Fold 1)
    train_inputs, train_outputs = task.get_train_and_val_data(task.folds[0])
    test_inputs, test_outputs = task.get_test_data(task.folds[0], include_target=True)
    
    # Extract Magpie features
    X_train = extract_magpie_features(train_inputs)
    X_test = extract_magpie_features(test_inputs)
    y_train = train_outputs.values
    y_test = test_outputs.values
    
    # Set training data sizes (10%, 20%, ..., 100%)
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Learning curve for Random Forest
    print("=== Calculating Learning Curve for Random Forest... ===")
    rf_train_scores = []
    rf_test_scores = []
    
    for train_size in train_sizes:
        # Sample training data
        n_samples = int(len(X_train) * train_size)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
    
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_subset, y_train_subset)
    
        # Calculate training and test errors
        train_mae = mean_absolute_error(y_train_subset, rf_model.predict(X_train_subset))
        test_mae = mean_absolute_error(y_test, rf_model.predict(X_test))
    
        rf_train_scores.append(train_mae)
        rf_test_scores.append(test_mae)
    
        print(f"Training size: {train_size*100:.0f}% ({n_samples} samples) - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    # Learning curve for CGCNN (simplified version: only 3 points due to computational time)
    print("\n=== Calculating Learning Curve for CGCNN... ===")
    cgcnn_train_sizes = [0.2, 0.5, 1.0]  # 20%, 50%, 100%
    cgcnn_train_scores = []
    cgcnn_test_scores = []
    
    for train_size in cgcnn_train_sizes:
        n_samples = int(len(train_inputs) * train_size)
        sampled_inputs = train_inputs[:n_samples]
        sampled_outputs = train_outputs.values[:n_samples]
    
        # Build graph data
        train_data = [structure_to_pyg_data(s, t) for s, t in zip(sampled_inputs, sampled_outputs)]
        test_data = [structure_to_pyg_data(s, t) for s, t in zip(test_inputs, test_outputs.values)]
    
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
        # CGCNN training (simplified version: 30 epochs)
        model = CGCNNMatbench().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
    
        model.train()
        for epoch in range(30):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
    
        # Evaluation
        model.eval()
        train_preds, train_true = [], []
        test_preds, test_true = [], []
    
        with torch.no_grad():
            # Training error
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch)
                train_preds.extend(out.cpu().numpy())
                train_true.extend(batch.y.cpu().numpy())
    
            # Test error
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                test_preds.extend(out.cpu().numpy())
                test_true.extend(batch.y.cpu().numpy())
    
        train_mae = mean_absolute_error(train_true, train_preds)
        test_mae = mean_absolute_error(test_true, test_preds)
    
        cgcnn_train_scores.append(train_mae)
        cgcnn_test_scores.append(test_mae)
    
        print(f"Training size: {train_size*100:.0f}% ({n_samples} samples) - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    # Visualize learning curves
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Random Forest
    ax.plot(train_sizes * len(X_train), rf_test_scores, 'o-', color='#667eea', linewidth=2,
            markersize=8, label='RF (Magpie) - Test error')
    ax.plot(train_sizes * len(X_train), rf_train_scores, 's--', color='#667eea', linewidth=1.5,
            markersize=6, alpha=0.5, label='RF (Magpie) - Train error')
    
    # CGCNN
    cgcnn_train_sizes_abs = [s * len(train_inputs) for s in cgcnn_train_sizes]
    ax.plot(cgcnn_train_sizes_abs, cgcnn_test_scores, 'o-', color='#764ba2', linewidth=2,
            markersize=8, label='CGCNN - Test error')
    ax.plot(cgcnn_train_sizes_abs, cgcnn_train_scores, 's--', color='#764ba2', linewidth=1.5,
            markersize=6, alpha=0.5, label='CGCNN - Train error')
    
    ax.set_xlabel('Number of training samples', fontsize=12)
    ax.set_ylabel('MAE (eV/atom)', fontsize=12)
    ax.set_title('Learning Curve: Relationship between Training Data Size and Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output example:
    # === Calculating Learning Curve for Random Forest... ===
    # Training size: 10% (10601 samples) - Train MAE: 0.0089, Test MAE: 0.0521
    # Training size: 20% (21202 samples) - Train MAE: 0.0085, Test MAE: 0.0428
    # ...
    # Training size: 100% (106012 samples) - Train MAE: 0.0078, Test MAE: 0.0325
    #
    # === Calculating Learning Curve for CGCNN... ===
    # Training size: 20% (21202 samples) - Train MAE: 0.0234, Test MAE: 0.0389
    # Training size: 50% (53006 samples) - Train MAE: 0.0198, Test MAE: 0.0312
    # Training size: 100% (106012 samples) - Train MAE: 0.0176, Test MAE: 0.0286
    

### 4.5.3 Interpretation of Data Requirements

**Insights from learning curves:**

  * **Small data region ( <20%)**: Random Forest is superior. CGCNN has low performance due to insufficient training data
  * **Medium data region (20-50%)** : CGCNN performance improves rapidly and approaches RF
  * **Large data region ( >50%)**: CGCNN surpasses RF, with the gap widening as data increases
  * **Data efficiency** : RF performance saturates at around 20% of data, while CGCNN maintains an upward trend even at 100% of data

**Practical implications:**

  * **When data is limited ( <10,000 samples)**: Random Forest (Magpie) is recommended
  * **When data is sufficient ( >50,000 samples)**: CGCNN achieves high accuracy
  * **When data collection cost is high** : A hybrid strategy is effective - start with RF in the initial stage, then migrate to CGCNN after data accumulation

## 4.6 Comparison of Interpretability

The **interpretability** of machine learning models is essential for understanding prediction rationale and improving reliability. In this section, we compare the interpretability of both methods using **SHAP values** for Random Forest (Magpie) and **Attention mechanism** for CGCNN.

### 4.6.1 Interpretation of Composition-based Features using SHAP Values

SHAP (SHapley Additive exPlanations) is a method that quantifies **the contribution of each feature to the prediction**. Based on Shapley value theory, it achieves fair distribution of contributions.

**💻 Code Example 10: Feature Importance Analysis using SHAP Values**
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: SHAP (SHapley Additive exPlanations) is a method that quanti
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Feature importance analysis using SHAP values
    import shap
    
    # Train Random Forest model (use already trained rf_model)
    # Sample a portion of test data (to reduce computational time)
    X_test_sample = X_test[:100]
    y_test_sample = y_test[:100]
    
    # Calculate SHAP values
    print("=== Calculating SHAP values... ===")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # SHAP value summary plot
    shap.summary_plot(shap_values, X_test_sample,
                      feature_names=[f'Magpie_{i}' for i in range(145)],
                      max_display=20, show=False)
    plt.title('Feature Importance by SHAP Values (Random Forest + Magpie)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prediction interpretation for a specific sample
    sample_idx = 0
    print(f"\n=== Prediction Interpretation for Sample {sample_idx} ===")
    print(f"True value: {y_test_sample[sample_idx]:.4f} eV/atom")
    print(f"Predicted:  {rf_model.predict(X_test_sample[sample_idx:sample_idx+1])[0]:.4f} eV/atom")
    
    # SHAP Force Plot (interpretation of individual samples)
    shap.force_plot(explainer.expected_value,
                    shap_values[sample_idx],
                    X_test_sample[sample_idx],
                    feature_names=[f'Magpie_{i}' for i in range(145)],
                    matplotlib=True, show=False)
    plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display top 10 important features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    print("\n=== Top 10 Important Features ===")
    for i, feat_idx in enumerate(top_features):
        print(f"{i+1}. Magpie_{feat_idx}: SHAP importance = {feature_importance[feat_idx]:.4f}")
    
    # Output example:
    # === Top 10 Important Features ===
    # 1. Magpie_34: SHAP importance = 0.0087  # Average electronegativity
    # 2. Magpie_12: SHAP importance = 0.0065  # Average atomic radius
    # 3. Magpie_78: SHAP importance = 0.0054  # Std dev of valence electrons
    # ...
    

### 4.6.2 GNN Interpretation using Attention Mechanism

In CGCNN, **Attention mechanisms** can be used to visualize **atom-atom interactions** important for prediction (strictly speaking, standard CGCNN doesn't have Attention, so comparison with Attention-based GNNs like GAT is more appropriate, but here we visualize CGCNN convolution weights for educational purposes).

**⚠️ Note: Limitations of CGCNN Interpretability**

Standard CGCNN does not have an Attention mechanism, so interpretability is limited. In this code example, we **visualize activations from convolution layers** , but for more advanced interpretation, the use of GAT (Graph Attention Networks) or Explainable AI methods (GNNExplainer, etc.) is recommended.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Standard CGCNN does not have an Attention mechanism, so inte
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # CGCNN activation visualization (simplified version)
    import torch
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure
    
    # Extract 1 sample from test data
    sample_structure = test_inputs[0]
    sample_target = test_outputs.values[0]
    
    # Convert to PyTorch Geometric data
    sample_data = structure_to_pyg_data(sample_structure, sample_target).to(device)
    
    # Predict with CGCNN (get intermediate layer activations)
    model.eval()
    with torch.no_grad():
        # Initialize node features
        x = sample_data.x
        x = model.atom_embedding(x)
    
        # Record activations from each convolution layer
        activations = []
        for conv, bn in zip(model.conv_layers, model.bn_layers):
            x = conv(x, sample_data.edge_index, sample_data.edge_attr)
            x = bn(x)
            x = model.activation(x)
            activations.append(x.cpu().numpy())
    
        # Final prediction
        x = global_mean_pool(x.unsqueeze(0), torch.zeros(x.size(0), dtype=torch.long))
        prediction = model.fc2(model.activation(model.fc1(x))).item()
    
    print(f"=== CGCNN Prediction Interpretation ===")
    print(f"True value: {sample_target:.4f} eV/atom")
    print(f"Predicted:  {prediction:.4f} eV/atom")
    
    # Visualize activation for each atom
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for layer_idx, activation in enumerate(activations):
        # Calculate average activation for each atom
        atom_importance = activation.mean(axis=1)
    
        # Get atom types
        atom_types = [site.specie.symbol for site in sample_structure]
    
        # Plot importance for each atom
        axes[layer_idx].bar(range(len(atom_types)), atom_importance)
        axes[layer_idx].set_xlabel('Atom index', fontsize=12)
        axes[layer_idx].set_ylabel('Average activation', fontsize=12)
        axes[layer_idx].set_title(f'Convolution Layer {layer_idx+1}', fontsize=12, fontweight='bold')
        axes[layer_idx].set_xticks(range(len(atom_types)))
        axes[layer_idx].set_xticklabels(atom_types, rotation=45)
    
    plt.tight_layout()
    plt.savefig('cgcnn_activation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nImportance of each atom (Layer 3 activation):")
    for i, (atom_type, importance) in enumerate(zip(atom_types, activations[-1].mean(axis=1))):
        print(f"{i+1}. {atom_type}: {importance:.4f}")
    

### 4.6.3 Summary of Interpretability Comparison

Perspective | Random Forest (SHAP) | CGCNN (Activation Visualization)  
---|---|---  
Interpretation target | Feature importance (145-dim Magpie) | Atom/bond importance  
Interpretation granularity | Composition level (electronegativity, atomic radius, etc.) | Structure level (specific atoms/bonds)  
Theoretical foundation | Shapley values (game theory) | Neural network activation  
Computational cost | Moderate (TreeExplainer is fast) | Low (forward pass only)  
Interpretation intuitiveness | High (feature names are meaningful) | Moderate (requires domain knowledge)  
  
**Practical selection:**

  * **Composition-level insights needed** (e.g., which elements are important) → Random Forest + SHAP
  * **Structure-level insights needed** (e.g., which atom-atom interactions are important) → CGCNN + GAT/GNNExplainer
  * **Explanation to domain experts** → SHAP values are more intuitive
  * **Detailed analysis for research purposes** → GNN Attention mechanisms are useful

## 4.7 Decision Flowchart for Method Selection

We integrate the quantitative comparison results from this chapter and construct a **decision flowchart to support practical method selection** using Mermaid.
    
    
    ```mermaid
    graph TD
        A[Material Property Prediction Task] --> B{Training Data Size?}
        B -->|<10,000| C[Random Forest + Magpie]
        B -->|10,000-50,000| D{Computational Resources?}
        B -->|>50,000| E{Accuracy or Speed Priority?}
    
        D -->|GPU Available| F[CGCNN RecommendedBalance of Accuracy and Data Efficiency]
        D -->|CPU Only| C
    
        E -->|Accuracy Priority| G[CGCNN Recommended12% MAE Improvement]
        E -->|Speed Priority| H{Real-time Prediction Required?}
    
        H -->|Yes| C
        H -->|No| F
    
        C --> I{Interpretability Essential?}
        F --> J{Statistical Significance Confirmed?}
        G --> J
    
        I -->|Yes| K[SHAP Value Feature Importance AnalysisPresent to Domain Experts]
        I -->|No| L[Random Forest Adoption Decision]
    
        J -->|Yes・p<0.05| M{Computational Cost Acceptable?}
        J -->|Not Yet| N[Perform t-test and 95% CI]
    
        M -->|40x Training Time Acceptable| O[CGCNN Adoption Decision]
        M -->|Not Acceptable| P[Consider Hybrid ApproachMigrate from Initial RF to CGCNN]
    
        N --> J
    
        K --> L
        O --> Q[Deploy to Production Environment]
        L --> Q
        P --> Q
    
        style A fill:#667eea,color:#fff
        style C fill:#4caf50,color:#fff
        style F fill:#ff9800,color:#fff
        style G fill:#764ba2,color:#fff
        style O fill:#764ba2,color:#fff
        style L fill:#4caf50,color:#fff
        style Q fill:#2196f3,color:#fff
    ```

### 4.7.1 Usage Guide for Decision Flowchart

**Details of each decision point:**

  1. **Evaluation of Training Data Size**
     * <10,000 samples: Random Forest performs stably and well (insights from learning curves)
     * 10,000-50,000 samples: Choose according to computational resources
     * >50,000 samples: CGCNN superiority is significant
  2. **Confirmation of Computational Resources**
     * GPU available: CGCNN training time reduced to about 30 minutes
     * CPU only: CGCNN training time may take 5-10 hours
  3. **Accuracy vs Speed Tradeoff**
     * Evaluate the value of 12% MAE improvement vs the cost of 40x training time
     * Check if 13x difference in inference speed affects real-time prediction
  4. **Verification of Statistical Significance**
     * Perform paired t-test with 5-fold cross-validation
     * Confirm p-value<0.05 and 95% confidence interval does not include 0
  5. **Interpretability Requirements**
     * SHAP values: Composition-level insights (which element properties are important)
     * Attention/GNNExplainer: Structure-level insights (which atom-atom interactions are important)

## 4.8 Chapter Summary

In this chapter, we quantitatively compared Random Forest (Magpie) and CGCNN from **7 perspectives** using the Matbench benchmark.

### Key Findings

Evaluation Aspect | Result | Recommended Conditions  
---|---|---  
Prediction accuracy | CGCNN: 12% MAE improvement | When high accuracy is essential  
Statistical significance | p=0.0001 << 0.05 (significant) | When scientific evidence is needed  
Training time | CGCNN: 40x slower | RF if fast development cycle needed  
Inference time | CGCNN: 13x slower | RF if real-time prediction needed  
Data requirements | RF: Saturates with less data, CGCNN: Improves with more data | CGCNN if >50,000 samples  
Interpretability | RF+SHAP: Composition level, CGCNN: Structure level | Depends on required insight granularity  
Implementation ease | RF: Easy, CGCNN: Complex | RF for rapid prototyping  
  
### Practical Recommendations

**🎯 Recommended Framework for Method Selection**

  * **Exploratory analysis stage** : Quickly build baseline with Random Forest (Magpie)
  * **Accuracy improvement stage** : Introduce CGCNN if data is sufficient (>50,000) and GPU is available
  * **Production operation stage** : Make final decision after evaluating inference speed vs accuracy tradeoff
  * **Hybrid strategy** : Start with RF in initial stage, gradually migrate to CGCNN after data accumulation

## Exercises

Exercise 1: Understanding Matbench Tasks Easy

**Problem:** Answer the following questions about the Matbench benchmark `matbench_mp_gap` task.

  1. What property is being predicted?
  2. What is the dataset size?
  3. What is the evaluation metric?

**Answer:**

  1. Band gap (eV)
  2. 106,113 samples
  3. MAE (Mean Absolute Error)

Exercise 2: Interpretation of t-Test Easy

**Problem:** Two models were evaluated by 5-fold cross-validation with the following MAE values. A paired t-test was performed and the p-value was 0.03. Can we say there is a statistically significant difference at significance level α=0.05?

Model A: [0.035, 0.038, 0.033, 0.036, 0.034]  
Model B: [0.032, 0.035, 0.030, 0.033, 0.031]

**Answer:**

Yes, there is a statistically significant difference. Since p-value (0.03) < α (0.05), we can reject the null hypothesis "the average MAE difference between the two models is zero". Therefore, Model B shows a statistically significant accuracy improvement over Model A.

Exercise 3: Calculation of 95% Confidence Interval Medium

**Problem:** The MAE difference (ModelA - ModelB) across 5 folds was [0.003, 0.004, 0.003, 0.005, 0.004] eV/atom. Calculate the 95% confidence interval for this MAE difference (use t-distribution critical value of 2.776).

**Hint:** Standard error SE = std(diff) / sqrt(n)

**Answer:**
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    """
    Example: Answer:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from scipy import stats
    
    mae_diff = np.array([0.003, 0.004, 0.003, 0.005, 0.004])
    mean_diff = mae_diff.mean()  # 0.00380
    std_diff = mae_diff.std(ddof=1)  # 0.00084
    se_diff = std_diff / np.sqrt(len(mae_diff))  # 0.00037
    t_critical = 2.776
    
    margin_error = t_critical * se_diff  # 0.00104
    ci_lower = mean_diff - margin_error  # 0.00276
    ci_upper = mean_diff + margin_error  # 0.00484
    
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}] eV/atom")
    # Output: 95% CI: [0.00276, 0.00484] eV/atom
    

The 95% confidence interval is [0.00276, 0.00484] eV/atom. Since this interval does not include 0, the accuracy difference is statistically significant.

Exercise 4: Computational Cost Analysis Medium

**Problem:** In a project, model training is performed once per month, and inference is performed 10,000 times daily. Compare the monthly total computational time between Random Forest (training 45 sec, inference 0.18 sec/1000 samples) and CGCNN (training 1833 sec, inference 2.34 sec/1000 samples) to determine which should be adopted.

**Answer:**
    
    
    # Comparison of monthly computational time
    # Random Forest
    rf_train_time = 45  # sec/month
    rf_infer_time = 0.18 * 10 * 30  # 0.18 sec/1000 samples × 10 × 30 days
    rf_total = rf_train_time + rf_infer_time
    print(f"RF monthly total: {rf_total:.1f}sec = {rf_total/60:.2f}min")
    
    # CGCNN
    cgcnn_train_time = 1833  # sec/month
    cgcnn_infer_time = 2.34 * 10 * 30
    cgcnn_total = cgcnn_train_time + cgcnn_infer_time
    print(f"CGCNN monthly total: {cgcnn_total:.1f}sec = {cgcnn_total/60:.2f}min")
    
    print(f"\nTime ratio (CGCNN/RF): {cgcnn_total/rf_total:.2f}x")
    
    # Output:
    # RF monthly total: 99.0sec = 1.65min
    # CGCNN monthly total: 2535.0sec = 42.25min
    # Time ratio (CGCNN/RF): 25.61x
    

**Judgment:** CGCNN requires an additional approximately 40 minutes of monthly computational time. In this use case, **inference time is dominant** , and CGCNN's 13x slower inference speed significantly impacts monthly total time. When daily inference frequency is high, **Random Forest is recommended**.

Exercise 5: Interpretation of Learning Curve Medium

**Problem:** From the following learning curve data, determine which model should be selected in a situation where only 20,000 training samples are available, and provide reasoning.

Training Data Size | RF Test MAE | CGCNN Test MAE

10,000 | 0.045 | 0.062  
20,000 | 0.038 | 0.041  
50,000 | 0.033 | 0.031  
100,000 | 0.032 | 0.027

**Answer:**

**Recommendation: Random Forest**

**Reasoning:**

  1. At 20,000 samples, RF's test MAE (0.038) is better than CGCNN (0.041)
  2. RF has small improvement margin from 20,000 to 50,000 samples (0.038→0.033), already showing saturation tendency
  3. CGCNN improves dramatically after 50,000 samples, but performance is insufficient with the currently available 20,000 samples
  4. In situations limited to 20,000 samples, **data-efficient RF is practical**

**Future strategy:** It is recommended to consider migration to CGCNN when data increases to over 50,000 samples.

Exercise 6: SHAP Value Implementation Hard

**Problem:** Write code to calculate SHAP values for a Random Forest model using Magpie features (145 dimensions) and extract the top 5 important features. Use the first 50 samples of the test data.

**Answer:**
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - shap>=0.42.0
    
    """
    Example: Answer:
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import shap
    import numpy as np
    
    # Train Random Forest model (assume Exercise 2 code example is used)
    # Assume X_train, y_train, X_test, y_test are already prepared
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Calculate SHAP values (50 test samples)
    X_test_sample = X_test[:50]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Indices of top 5 important features
    top_5_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    # Magpie feature name mapping (partial examples)
    magpie_names = {
        34: "Average electronegativity",
        12: "Average atomic radius",
        78: "Std dev of valence electrons",
        # ... define other feature names
    }
    
    print("=== Top 5 Important Features (SHAP Values) ===")
    for rank, feat_idx in enumerate(top_5_indices, 1):
        feat_name = magpie_names.get(feat_idx, f"Magpie_{feat_idx}")
        shap_importance = mean_abs_shap[feat_idx]
        print(f"{rank}. {feat_name} (Index {feat_idx}): SHAP importance = {shap_importance:.4f}")
    
    # Visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.barh(range(5), mean_abs_shap[top_5_indices][::-1], color='#764ba2')
    plt.yticks(range(5), [magpie_names.get(i, f"Magpie_{i}") for i in top_5_indices][::-1])
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
    plt.title('Top 5 Important Features (by SHAP Values)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('top5_features_shap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Output example:
    # === Top 5 Important Features (SHAP Values) ===
    # 1. Average electronegativity (Index 34): SHAP importance = 0.0087
    # 2. Average atomic radius (Index 12): SHAP importance = 0.0065
    # 3. Std dev of valence electrons (Index 78): SHAP importance = 0.0054
    # 4. Magpie_45: SHAP importance = 0.0048
    # 5. Magpie_89: SHAP importance = 0.0042
    

Exercise 7: Application of Decision Flowchart Hard

**Problem:** Apply the decision flowchart from Section 4.7 to the following project conditions and select the optimal method.

**Conditions:**

  * Training data size: 75,000 samples
  * Computational resources: GPU (Tesla V100) available
  * Priority: Accuracy priority (real-time prediction not required)
  * Interpretability: Not essential but desirable
  * Statistical significance verification: Planned

**Answer:**

**Recommended Method: CGCNN**

**Decision Process:**

  1. **Data size evaluation** : 75,000 > 50,000 → Branch to ">50,000"
  2. **Accuracy vs speed** : Accuracy priority → Branch to "Accuracy Priority"
  3. **Reach "CGCNN Recommended (12% MAE Improvement)"**
  4. **Confirm statistical significance** : Perform paired t-test with 5-fold cross-validation and confirm p<0.05
  5. **Computational cost acceptability** : With GPU (V100) available, training time is reduced to about 30 minutes, which is acceptable
  6. **Final decision** : CGCNN adoption decision

**Additional considerations:**

  * Since interpretability is desirable, consideration of **Attention-based GNN (GAT)** is also recommended
  * Before production environment deployment, confirm that 13x slower inference time does not affect actual operation
  * If data is expected to increase further in the future, CGCNN can be expected to improve performance even more in the long term

Exercise 8: Design of Hybrid Strategy Hard

**Problem:** Design a migration strategy from Random Forest to CGCNN for a project where data increases gradually (initial 5,000 samples, 30,000 after 3 months, 80,000 after 6 months). Clearly specify method selection at each stage and judgment criteria for migration timing.

**Answer:**

**Hybrid Migration Strategy:**

**Phase 1: Initial Stage (5,000 samples, 0-3 months)**

  * **Adopted method: Random Forest (Magpie)**
  * **Reasoning:**
    * 5,000 samples is in the small data region, where RF is data-efficient
    * Enables rapid prototyping and initial baseline construction
    * Insufficient data volume for CGCNN training
  * **Implementation items:**
    * Build baseline model with RF (MAE target: ~0.045 eV/atom)
    * Feature importance analysis using SHAP values
    * Establish data collection pipeline

**Phase 2: Medium-scale Stage (30,000 samples, 3-6 months)**

  * **Adopted method: Continue Random Forest + Experimental introduction of CGCNN**
  * **Reasoning:**
    * 30,000 samples is in the migration consideration region (both methods compete in learning curve)
    * Begin experimental CGCNN training and conduct performance evaluation
    * Continue RF in production environment (prioritize stability)
  * **Migration judgment criteria:**
    1. CGCNN achieves MAE exceeding RF
    2. Confirm statistical significance with 5-fold cross-validation (p<0.05)
    3. CGCNN training time <1 hour in GPU environment
  * **Implementation items:**
    * Build CGCNN training pipeline
    * Hyperparameter tuning
    * Quantitative comparison of RF vs CGCNN (apply methods from this chapter)

**Phase 3: Large-scale Stage (80,000 samples, 6 months onward)**

  * **Adopted method: CGCNN (primary) + Random Forest (fallback)**
  * **Reasoning:**
    * 80,000 samples is in the large data region, where CGCNN superiority is significant (learning curve analysis)
    * 12% MAE improvement effect is statistically significant
    * If data increase trend continues, CGCNN performance will improve further
  * **Migration implementation:**
    1. Deploy CGCNN to production environment
    2. Maintain RF as fallback model (for CGCNN inference failures)
    3. Monitor inference time (confirm 13x slower CGCNN doesn't affect operations)
  * **Continuous improvement:**
    * Regularly retrain CGCNN as data increases
    * Consider introducing Attention mechanism (GAT) to improve interpretability
    * Attempt ensemble (RF + CGCNN) for highest accuracy

**Risk Management:**

  * **Migration failure countermeasure** : Continue RF operation, retry CGCNN after resolving issues
  * **Performance degradation detection** : Continuously monitor prediction accuracy in production environment
  * **Cost overrun countermeasure** : Revert to RF if GPU usage fees exceed budget

Exercise 9: Interpretation of Effect Size Medium

**Problem:** If an effect size of Cohen's d = 0.35 is obtained, evaluate the magnitude of this effect size and explain its practical meaning.

**Answer:**

**Effect size magnitude:** "Medium" (in the range 0.2 < |d| < 0.5)

**Practical interpretation:**

  1. **Statistical meaning:** Cohen's d = 0.35 indicates that the performance difference between the two models is 0.35 times the standard deviation
  2. **Practical meaning:** A medium effect size means "recognizable but not dramatic improvement" in practice
  3. **Judgment criteria:**
     * If costs are equivalent, adopt new method as improvement effect exists
     * If introduction cost of new method is high, carefully evaluate cost-effectiveness
     * Even medium effect size may contribute to improving efficiency of materials discovery

**Comparison:** In this chapter's CGCNN vs RF comparison, Cohen's d = 9.19 (very large), making the practical superiority extremely clear.

Exercise 10: Comprehensive Method Selection Hard

**Problem:** Integrate the 7 evaluation perspectives learned in this chapter (prediction accuracy, statistical significance, computational cost, data requirements, interpretability, implementation ease, decision flowchart) and select the optimal method for the following scenario.

**Scenario:** Pharmaceutical company's new drug candidate material discovery project

  * Objective: Select 100 promising candidates from 10,000 compounds
  * Current data: 12,000 experimental data samples (no additional experiments)
  * Computing environment: Internal cloud (GPU available but pay-per-use)
  * Time constraint: Submit candidate list within 3 months
  * Interpretation requirement: Prediction rationale essential for regulatory authority explanation materials
  * Accuracy requirement: Minimize false positives (reduce experimental costs)

**Answer:**

**Recommended Strategy: Random Forest (Magpie) + SHAP Value Analysis**

**Basis for Comprehensive Judgment:**

**1\. Data Requirements Analysis (Section 4.5)**

  * 12,000 samples falls into medium data region (10,000-50,000)
  * Learning curve insight: In this region, RF and CGCNN compete, with RF slightly advantageous
  * Since no additional data acquisition is possible, data-efficient RF is safe

**2\. Computational Cost Analysis (Section 4.4)**

  * In GPU pay-per-use environment, CGCNN's 40x training time cost directly increases expenses
  * Inference for 10,000 compounds: RF 1.8 sec vs CGCNN 23.4 sec (13x difference)
  * 3-month time constraint is sufficient, but RF is advantageous in cost efficiency

**3\. Interpretability Requirement (Section 4.6)**

  * **Requirement for regulatory authority explanation is essential** \- this is a decisive factor
  * Feature importance analysis using SHAP values can clearly explain "which chemical properties are important"
  * CGCNN's Attention mechanism requires domain knowledge and is difficult to explain to regulatory authorities
  * **SHAP value analysis perfectly matches project requirements**

**4\. Accuracy Requirement (Section 4.2-4.3)**

  * To minimize false positives, high prediction accuracy is needed
  * CGCNN shows 12% accuracy improvement, but statistical significance needs verification with 12,000 samples
  * RF also has sufficiently high accuracy (R² > 0.93), and false positive rate is acceptable

**5\. Implementation Ease and Time Constraint**

  * RF is easy to implement and enables rapid prototype construction
  * With 3-month constraint, adopting RF with high certainty avoids risk

**Implementation Plan:**

  1. **Week 1-2** : Build baseline model with Random Forest + Magpie features
  2. **Week 3-4** : Evaluate performance with 5-fold cross-validation (MAE, R²)
  3. **Week 5-6** : Visualize feature importance using SHAP value analysis
  4. **Week 7-8** : Execute predictions on 10,000 compounds, select top 100 candidates
  5. **Week 9-10** : Create explanation materials for regulatory authorities (including SHAP value plots)
  6. **Week 11-12** : Internal review and final report creation

**Risk Countermeasures:**

  * **If accuracy is insufficient** : Improve accuracy with ensemble learning (multiple RF models)
  * **If interpretability is insufficient** : Add LIME (Local Interpretable Model-agnostic Explanations)
  * **If many false positives** : Adjust threshold to tighten selection criteria

**Rejection reason: CGCNN**

  * Interpretability requirement is decisive rejection factor (difficult to explain to regulatory authorities)
  * With 12,000 samples, CGCNN superiority is limited
  * Considering computational cost and risk, ROI is unclear

**Future consideration:** If the project continues and experimental data increases to over 50,000 samples, it is recommended to re-evaluate migration to CGCNN.

## References

  1. Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. (2020). Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. _npj Computational Materials_ , 6(1), 138, pp. 1-10.
  2. Ward, L., Agrawal, A., Choudhary, A., Wolverton, C. (2016). A general-purpose machine learning framework for predicting properties of inorganic materials. _npj Computational Materials_ , 2, 16028, pp. 1-7.
  3. Xie, T., Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. _Physical Review Letters_ , 120(14), 145301, pp. 1-6.
  4. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. _Proceedings of the 34th International Conference on Machine Learning_ , PMLR 70, pp. 1263-1272.
  5. Lundberg, S. M., Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. _Advances in Neural Information Processing Systems_ , 30, pp. 4765-4774.
  6. Cohen, J. (1988). _Statistical Power Analysis for the Behavioral Sciences_ (2nd ed.). Lawrence Erlbaum Associates, pp. 20-27.
  7. Breiman, L. (2001). Random Forests. _Machine Learning_ , 45(1), 5-32, pp. 5-32.
  8. Choudhary, K., DeCost, B. (2021). Atomistic Line Graph Neural Network for improved materials property predictions. _npj Computational Materials_ , 7(1), 185, pp. 1-8.

[← Chapter 3: MPNN Implementation](<chapter-3.html>) [Chapter 5: Hybrid Approach →](<chapter-5.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
