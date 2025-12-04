---
title: "Chapter 3: Hands-On Python Tutorial"
chapter_title: "Chapter 3: Hands-On Python Tutorial"
subtitle: Nanomaterial Data Analysis and Machine Learning
reading_time: 30-40 min
difficulty: Beginner
code_examples: 0
exercises: 0
---

# Chapter 3: Hands-On Python Tutorial

Build skills for efficiently exploring conditions using regression models effective even with small datasets and Bayesian optimization. Covers essential visualization of MD data and interpretation with SHAP in one go.

**💡 Supplement:** fewfortrialsrowswithgoodconditionsfindofGoal。Bayesian optimizationis“metal detector”guides you to hits。

Nanomaterial Data Analysis and Machine Learning

* * *

## Learning Objectives

By studying this chapter, you will acquire the following skills:

  * ✅ Hands-on nanoparticle data generation, visualization, and preprocessing
  * ✅ Prediction of nanomaterial properties using 5 types of regression models
  * ✅ Optimal design of nanomaterials through Bayesian optimization
  * ✅ Interpretation of machine learning models using SHAP analysis
  * ✅ Trade-off analysis through multi-objective optimization
  * ✅ TEM image analysis and size distribution fitting
  * ✅ Application to quality control through anomaly detection

* * *

## 3.1 Environment Setup

### Required Libraries

Main Python libraries used in this tutorial:
    
    
    # Data processing and visualization
    pandas, numpy, matplotlib, seaborn, scipy
    
    # Machine learning
    scikit-learn, lightgbm
    
    # Optimization
    scikit-optimize
    
    # Model interpretation
    shap
    
    # Multi-objective optimization (optional)
    pymoo
    

### Installation Methods

#### Option 1: Anaconda Environment
    
    
    # Create new Anaconda environment
    conda create -n nanomaterials python=3.10 -y
    conda activate nanomaterials
    
    # Install required libraries
    conda install pandas numpy matplotlib seaborn scipy scikit-learn -y
    conda install -c conda-forge lightgbm scikit-optimize shap -y
    
    # For multi-objective optimization (optional)
    pip install pymoo
    

#### Option 2: venv + pip Environment
    
    
    # Create virtual environment
    python -m venv nanomaterials_env
    
    # Activate virtual environment
    # macOS/Linux:
    source nanomaterials_env/bin/activate
    # Windows:
    nanomaterials_env\Scripts\activate
    
    # Install required libraries
    pip install pandas numpy matplotlib seaborn scipy
    pip install scikit-learn lightgbm scikit-optimize shap pymoo
    

#### Option 3: Google Colab

When using Google Colab, execute the following code in a cell:
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: When using Google Colab, execute the following code in a cel
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    # Install additional packages
    !pip install lightgbm scikit-optimize shap pymoo
    
    # Verify imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("Environment setup complete!")
    

* * *

## 3.2 Nanoparticle Data Preparation and Visualization

### [Example 1] Synthetic Data Generation: Size and Optical Properties of Gold Nanoparticles

The localized surface plasmon resonance (LSPR) wavelength of gold nanoparticles depends on particle size. This relationship is represented with simulated data.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: The localized surface plasmon resonance (LSPR) wavelength of
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Font settings (adjust as needed)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set random seed (for reproducibility)
    np.random.seed(42)
    
    # Number of samples
    n_samples = 200
    
    # Gold nanoparticle size (nm): mean 15 nm, std 5 nm
    size = np.random.normal(15, 5, n_samples)
    size = np.clip(size, 5, 50)  # 5-50 nmclipped to range
    
    # LSPR wavelength (nm): simplified Mie theory approximation
    # Base wavelength 520 nm + size-dependent term + noise
    lspr = 520 + 0.8 * (size - 15) + np.random.normal(0, 5, n_samples)
    
    # Synthesis conditions
    temperature = np.random.uniform(20, 80, n_samples)  # Temperature (°C)
    pH = np.random.uniform(4, 10, n_samples)  # pH
    
    # Create DataFrame
    data = pd.DataFrame({
        'size_nm': size,
        'lspr_nm': lspr,
        'temperature_C': temperature,
        'pH': pH
    })
    
    print("=" * 60)
    print("Gold nanoparticle data generation complete")
    print("=" * 60)
    print(data.head(10))
    print("\nBasic statistics:")
    print(data.describe())
    

### [Example 2] Size Distribution Histogram
    
    
    # Size distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram and KDE (kernel density estimation)
    ax.hist(data['size_nm'], bins=30, alpha=0.6, color='skyblue',
            edgecolor='black', density=True, label='Histogram')
    
    # KDE plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data['size_nm'])
    x_range = np.linspace(data['size_nm'].min(), data['size_nm'].max(), 100)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.set_xlabel('Particle Size (nm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Gold Nanoparticle Size Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average size: {data['size_nm'].mean():.2f} nm")
    print(f"Standard deviation: {data['size_nm'].std():.2f} nm")
    print(f"Median: {data['size_nm'].median():.2f} nm")
    

### [Example 3] Scatter Plot Matrix
    
    
    # Pairplot (scatter plot matrix)
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6},
                 height=2.5, corner=False)
    plt.suptitle('Pairplot of Gold Nanoparticle Data', y=1.01, fontsize=14, fontweight='bold')
    plt.show()
    
    print("Visualized relationships between variables")
    

### [Example 4] Correlation Matrix Heatmap
    
    
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Correlation coefficients:")
    print(correlation_matrix)
    print(f"\nCorrelation between LSPR wavelength and size: {correlation_matrix.loc['lspr_nm', 'size_nm']:.3f}")
    

### [Example 5] 3D Plot: Size vs Temperature vs LSPR
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colormap
    scatter = ax.scatter(data['size_nm'], data['temperature_C'], data['lspr_nm'],
                         c=data['pH'], cmap='viridis', s=50, alpha=0.6, edgecolors='k')
    
    ax.set_xlabel('Size (nm)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_zlabel('LSPR Wavelength (nm)', fontsize=11)
    ax.set_title('3D Scatter: Size vs Temperature vs LSPR (colored by pH)',
                 fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('pH', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualized multidimensional relationships with 3D plot")
    

* * *

## 3.3 Preprocessing and Data Splitting

### [Example 6] Missing Value Handling
    
    
    # Introduce missing values artificially (for practice)
    data_with_missing = data.copy()
    np.random.seed(123)
    
    # Introduce 5% missing values randomly
    missing_indices = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
    data_with_missing.loc[missing_indices, 'temperature_C'] = np.nan
    
    print("=" * 60)
    print("Check missing values")
    print("=" * 60)
    print(f"Number of missing values:\n{data_with_missing.isnull().sum()}")
    
    # Missing value handling method 1: fill with mean
    data_filled_mean = data_with_missing.fillna(data_with_missing.mean())
    
    # Missing value handling method 2: fill with median
    data_filled_median = data_with_missing.fillna(data_with_missing.median())
    
    # Missing value handling method 3: drop
    data_dropped = data_with_missing.dropna()
    
    print(f"\nOriginal data: {len(data_with_missing)}rows")
    print(f"After dropping missing values: {len(data_dropped)}rows")
    print(f"After mean imputation: {len(data_filled_mean)}rows(no missing values)")
    
    # Use original data (no missing values) for subsequent analysis
    data_clean = data.copy()
    print("\n→ Using data without missing values henceforth")
    

### [Example 7] Outlier Detection (IQR Method)
    
    
    # Outlier detection using IQR (interquartile range) method
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    # Detect outliers in size
    outliers, lower, upper = detect_outliers_iqr(data_clean['size_nm'])
    
    print("=" * 60)
    print("Outlier Detection (IQR Method)")
    print("=" * 60)
    print(f"Number of detected outliers: {outliers.sum()}")
    print(f"Lower bound: {lower:.2f} nm, Upper bound: {upper:.2f} nm")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([data_clean['size_nm']], labels=['Size (nm)'], vert=False)
    ax.scatter(data_clean.loc[outliers, 'size_nm'],
               [1] * outliers.sum(), color='red', s=100,
               label=f'Outliers (n={outliers.sum()})', zorder=3)
    ax.set_xlabel('Size (nm)', fontsize=12)
    ax.set_title('Boxplot with Outliers', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("→ Using all data without removing outliers")
    

### [Example 8] Feature Scaling (StandardScaler)
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Separate features and target
    X = data_clean[['size_nm', 'temperature_C', 'pH']]
    y = data_clean['lspr_nm']
    
    # StandardScaler (normalize to mean 0, std 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compare before and after scaling
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("=" * 60)
    print("Statistics before scaling")
    print("=" * 60)
    print(X.describe())
    
    print("\n" + "=" * 60)
    print("Statistics after scaling (mean≈0, std≈1)")
    print("=" * 60)
    print(X_scaled_df.describe())
    
    print("\n→ Scaling unified the scale of each feature")
    

### [Example 9] Train-Test Data Splitting
    
    
    from sklearn.model_selection import train_test_split
    
    # Split into training and test data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("=" * 60)
    print("Data Split")
    print("=" * 60)
    print(f"Total data: {len(X)}samples")
    print(f"Training data: {len(X_train)}samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test data: {len(X_test)}samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\nTraining data statistics:")
    print(pd.DataFrame(X_train, columns=X.columns).describe())
    

* * *

## 3.4 Predicting Nanoparticle Properties with Regression Models

Goal: Predict LSPR wavelength from size, temperature, and pH

### [Example 10] Linear Regression
    
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Build linear regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Prediction
    y_train_pred_lr = model_lr.predict(X_train)
    y_test_pred_lr = model_lr.predict(X_test)
    
    # Evaluation metrics
    r2_train_lr = r2_score(y_train, y_train_pred_lr)
    r2_test_lr = r2_score(y_test, y_test_pred_lr)
    rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
    mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
    
    print("=" * 60)
    print("Linear Regression")
    print("=" * 60)
    print(f"Training R²: {r2_train_lr:.4f}")
    print(f"Test R²: {r2_test_lr:.4f}")
    print(f"Test RMSE: {rmse_test_lr:.4f} nm")
    print(f"Test MAE: {mae_test_lr:.4f} nm")
    
    # Regression coefficients
    print("\ntimesregressioncoefficient:")
    for name, coef in zip(X.columns, model_lr.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {model_lr.intercept_:.4f}")
    
    # Residual plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs actual values
    axes[0].scatter(y_test, y_test_pred_lr, alpha=0.6, edgecolors='k')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual LSPR (nm)', fontsize=11)
    axes[0].set_ylabel('Predicted LSPR (nm)', fontsize=11)
    axes[0].set_title(f'Linear Regression (R² = {r2_test_lr:.3f})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_test_pred_lr
    axes[1].scatter(y_test_pred_lr, residuals, alpha=0.6, edgecolors='k')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted LSPR (nm)', fontsize=11)
    axes[1].set_ylabel('Residuals (nm)', fontsize=11)
    axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### [Example 11] Random Forest Regression
    
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Random forest regression model
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    
    # Prediction
    y_train_pred_rf = model_rf.predict(X_train)
    y_test_pred_rf = model_rf.predict(X_test)
    
    # Evaluation
    r2_train_rf = r2_score(y_train, y_train_pred_rf)
    r2_test_rf = r2_score(y_test, y_test_pred_rf)
    rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
    
    print("=" * 60)
    print("Random Forest Regression")
    print("=" * 60)
    print(f"Training R²: {r2_train_rf:.4f}")
    print(f"Test R²: {r2_test_rf:.4f}")
    print(f"Test RMSE: {rmse_test_rf:.4f} nm")
    print(f"Test MAE: {mae_test_rf:.4f} nm")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nfeature importance:")
    print(feature_importance)
    
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'],
            color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    

### [Example 12] Gradient Boosting (LightGBM)
    
    
    # Requirements:
    # - Python 3.9+
    # - lightgbm>=4.0.0
    
    """
    Example: [Example 12] Gradient Boosting (LightGBM)
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import lightgbm as lgb
    
    # Build LightGBM model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'random_state': 42,
        'verbose': -1
    }
    
    model_lgb = lgb.LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    
    # Prediction
    y_train_pred_lgb = model_lgb.predict(X_train)
    y_test_pred_lgb = model_lgb.predict(X_test)
    
    # Evaluation
    r2_train_lgb = r2_score(y_train, y_train_pred_lgb)
    r2_test_lgb = r2_score(y_test, y_test_pred_lgb)
    rmse_test_lgb = np.sqrt(mean_squared_error(y_test, y_test_pred_lgb))
    mae_test_lgb = mean_absolute_error(y_test, y_test_pred_lgb)
    
    print("=" * 60)
    print("Gradient Boosting (LightGBM)")
    print("=" * 60)
    print(f"Training R²: {r2_train_lgb:.4f}")
    print(f"Test R²: {r2_test_lgb:.4f}")
    print(f"Test RMSE: {rmse_test_lgb:.4f} nm")
    print(f"Test MAE: {mae_test_lgb:.4f} nm")
    
    # Predicted vs actual values plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test_pred_lgb, alpha=0.6, edgecolors='k', s=60)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual LSPR (nm)', fontsize=12)
    ax.set_ylabel('Predicted LSPR (nm)', fontsize=12)
    ax.set_title(f'LightGBM Prediction (R² = {r2_test_lgb:.3f})',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### [Example 13] Support Vector Regression (SVR)
    
    
    from sklearn.svm import SVR
    
    # SVR model (RBF kernel)
    model_svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    model_svr.fit(X_train, y_train)
    
    # Prediction
    y_train_pred_svr = model_svr.predict(X_train)
    y_test_pred_svr = model_svr.predict(X_test)
    
    # Evaluation
    r2_train_svr = r2_score(y_train, y_train_pred_svr)
    r2_test_svr = r2_score(y_test, y_test_pred_svr)
    rmse_test_svr = np.sqrt(mean_squared_error(y_test, y_test_pred_svr))
    mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
    
    print("=" * 60)
    print("Support Vector Regression (SVR)")
    print("=" * 60)
    print(f"Training R²: {r2_train_svr:.4f}")
    print(f"Test R²: {r2_test_svr:.4f}")
    print(f"Test RMSE: {rmse_test_svr:.4f} nm")
    print(f"Test MAE: {mae_test_svr:.4f} nm")
    print(f"Number of support vectors: {len(model_svr.support_)}")
    

### [Example 14] Neural Network (MLP Regressor)
    
    
    from sklearn.neural_network import MLPRegressor
    
    # MLP model
    model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50),
                             activation='relu',
                             solver='adam',
                             alpha=0.001,
                             max_iter=500,
                             random_state=42,
                             early_stopping=True,
                             validation_fraction=0.1,
                             verbose=False)
    
    model_mlp.fit(X_train, y_train)
    
    # Prediction
    y_train_pred_mlp = model_mlp.predict(X_train)
    y_test_pred_mlp = model_mlp.predict(X_test)
    
    # Evaluation
    r2_train_mlp = r2_score(y_train, y_train_pred_mlp)
    r2_test_mlp = r2_score(y_test, y_test_pred_mlp)
    rmse_test_mlp = np.sqrt(mean_squared_error(y_test, y_test_pred_mlp))
    mae_test_mlp = mean_absolute_error(y_test, y_test_pred_mlp)
    
    print("=" * 60)
    print("Neural Network (MLP Regressor)")
    print("=" * 60)
    print(f"Training R²: {r2_train_mlp:.4f}")
    print(f"Test R²: {r2_test_mlp:.4f}")
    print(f"Test RMSE: {rmse_test_mlp:.4f} nm")
    print(f"Test MAE: {mae_test_mlp:.4f} nm")
    print(f"Number of iterations: {model_mlp.n_iter_}")
    print(f"Hidden layer structure: {model_mlp.hidden_layer_sizes}")
    

### [Example 15] Model Performance Comparison
    
    
    # Summarize all model performances
    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'LightGBM', 'SVR', 'MLP'],
        'R² (Train)': [r2_train_lr, r2_train_rf, r2_train_lgb, r2_train_svr, r2_train_mlp],
        'R² (Test)': [r2_test_lr, r2_test_rf, r2_test_lgb, r2_test_svr, r2_test_mlp],
        'RMSE (Test)': [rmse_test_lr, rmse_test_rf, rmse_test_lgb, rmse_test_svr, rmse_test_mlp],
        'MAE (Test)': [mae_test_lr, mae_test_rf, mae_test_lgb, mae_test_svr, mae_test_mlp]
    })
    
    results['Overfit'] = results['R² (Train)'] - results['R² (Test)']
    
    print("=" * 80)
    print("Performance Comparison of All Models")
    print("=" * 80)
    print(results.to_string(index=False))
    
    # Identify best model
    best_model_idx = results['R² (Test)'].idxmax()
    best_model_name = results.loc[best_model_idx, 'Model']
    best_r2 = results.loc[best_model_idx, 'R² (Test)']
    
    print(f"\nBest model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² score comparison
    x_pos = np.arange(len(results))
    axes[0].bar(x_pos, results['R² (Test)'], alpha=0.7, color='steelblue',
                edgecolor='black', label='Test R²')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(results['Model'], rotation=15, ha='right')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Comparison: R² Score', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    # RMSEcomparison
    axes[1].bar(x_pos, results['RMSE (Test)'], alpha=0.7, color='coral',
                edgecolor='black', label='Test RMSE')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(results['Model'], rotation=15, ha='right')
    axes[1].set_ylabel('RMSE (nm)', fontsize=12)
    axes[1].set_title('Model Comparison: RMSE', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.5 Quantum Dot Emission Wavelength Prediction

### [Example 16] Data Generation: CdSe Quantum Dots

CdSequantum dotsofemissionwavelengthis、Brusbased on equationsizedepends on。
    
    
    # CdSequantum dotsdataofgeneration
    np.random.seed(100)
    
    n_qd_samples = 150
    
    # quantum dotsofsize（2-10 nm）
    size_qd = np.random.uniform(2, 10, n_qd_samples)
    
    # Brusequationofsimpleapproximation: emission = 520 + 130/(size^0.8) + noise
    emission = 520 + 130 / (size_qd ** 0.8) + np.random.normal(0, 10, n_qd_samples)
    
    # Synthesis conditions
    synthesis_time = np.random.uniform(10, 120, n_qd_samples)  # min
    precursor_ratio = np.random.uniform(0.5, 2.0, n_qd_samples)  # molar ratio
    
    # create DataFrame
    data_qd = pd.DataFrame({
        'size_nm': size_qd,
        'emission_nm': emission,
        'synthesis_time_min': synthesis_time,
        'precursor_ratio': precursor_ratio
    })
    
    print("=" * 60)
    print("CdSequantum dotsdataofgenerationcomplete")
    print("=" * 60)
    print(data_qd.head(10))
    print("\nBasic statistics:")
    print(data_qd.describe())
    
    # sizeand emissionwavelengthofrelationshipplot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data_qd['size_nm'], data_qd['emission_nm'],
                         c=data_qd['synthesis_time_min'], cmap='plasma',
                         s=80, alpha=0.7, edgecolors='k')
    ax.set_xlabel('Quantum Dot Size (nm)', fontsize=12)
    ax.set_ylabel('Emission Wavelength (nm)', fontsize=12)
    ax.set_title('CdSe Quantum Dot: Size vs Emission Wavelength',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Synthesis Time (min)', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### [Example 17] Quantum Dot Model (LightGBM)
    
    
    # Separate features and target
    X_qd = data_qd[['size_nm', 'synthesis_time_min', 'precursor_ratio']]
    y_qd = data_qd['emission_nm']
    
    # scaling
    scaler_qd = StandardScaler()
    X_qd_scaled = scaler_qd.fit_transform(X_qd)
    
    # training/testminsplit
    X_qd_train, X_qd_test, y_qd_train, y_qd_test = train_test_split(
        X_qd_scaled, y_qd, test_size=0.2, random_state=42
    )
    
    # LightGBMmodel
    model_qd = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    
    model_qd.fit(X_qd_train, y_qd_train)
    
    # Prediction
    y_qd_train_pred = model_qd.predict(X_qd_train)
    y_qd_test_pred = model_qd.predict(X_qd_test)
    
    # Evaluation
    r2_qd_train = r2_score(y_qd_train, y_qd_train_pred)
    r2_qd_test = r2_score(y_qd_test, y_qd_test_pred)
    rmse_qd = np.sqrt(mean_squared_error(y_qd_test, y_qd_test_pred))
    mae_qd = mean_absolute_error(y_qd_test, y_qd_test_pred)
    
    print("=" * 60)
    print("quantum dotsemissionwavelengthpredictionmodel（LightGBM）")
    print("=" * 60)
    print(f"Training R²: {r2_qd_train:.4f}")
    print(f"Test R²: {r2_qd_test:.4f}")
    print(f"Test RMSE: {rmse_qd:.4f} nm")
    print(f"Test MAE: {mae_qd:.4f} nm")
    

### [Example 18] Prediction Result Visualization
    
    
    # Predicted vs actual values plot（with confidence interval）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test dataofplot
    axes[0].scatter(y_qd_test, y_qd_test_pred, alpha=0.6, s=80,
                    edgecolors='k', label='Test Data')
    axes[0].plot([y_qd_test.min(), y_qd_test.max()],
                 [y_qd_test.min(), y_qd_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    
    # ±10 nm ofrangedisplay
    axes[0].fill_between([y_qd_test.min(), y_qd_test.max()],
                         [y_qd_test.min()-10, y_qd_test.max()-10],
                         [y_qd_test.min()+10, y_qd_test.max()+10],
                         alpha=0.2, color='gray', label='±10 nm')
    
    axes[0].set_xlabel('Actual Emission (nm)', fontsize=12)
    axes[0].set_ylabel('Predicted Emission (nm)', fontsize=12)
    axes[0].set_title(f'QD Emission Prediction (R² = {r2_qd_test:.3f})',
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # sizeclassificationofpredictionaccuracy
    size_bins = [2, 4, 6, 8, 10]
    size_labels = ['2-4 nm', '4-6 nm', '6-8 nm', '8-10 nm']
    data_qd_test = pd.DataFrame({
        'size': X_qd.iloc[y_qd_test.index]['size_nm'].values,
        'actual': y_qd_test.values,
        'predicted': y_qd_test_pred
    })
    data_qd_test['size_bin'] = pd.cut(data_qd_test['size'], bins=size_bins, labels=size_labels)
    data_qd_test['error'] = np.abs(data_qd_test['actual'] - data_qd_test['predicted'])
    
    # sizeper binandofaverageerror
    error_by_size = data_qd_test.groupby('size_bin')['error'].mean()
    
    axes[1].bar(range(len(error_by_size)), error_by_size.values,
                color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(len(error_by_size)))
    axes[1].set_xticklabels(error_by_size.index)
    axes[1].set_ylabel('Mean Absolute Error (nm)', fontsize=12)
    axes[1].set_xlabel('QD Size Range', fontsize=12)
    axes[1].set_title('Prediction Error by QD Size', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\noverallofaverageabsolute error: {mae_qd:.2f} nm")
    print("sizeclassificationofaverageabsolute error:")
    print(error_by_size)
    

* * *

## 3.6 Feature Importance Analysis

### [Example 19] Feature Importance (LightGBM)
    
    
    # LightGBMmodeloffeature importance（gain-based）
    importance_gain = model_lgb.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_gain
    }).sort_values('Importance', ascending=False)
    
    print("=" * 60)
    print("feature importance（LightGBM）")
    print("=" * 60)
    print(importance_df)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue', 'coral', 'lightgreen']
    ax.barh(importance_df['Feature'], importance_df['Importance'],
            color=colors, edgecolor='black')
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Feature Importance: LSPR Prediction',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print(f"\nmost important feature: {importance_df.iloc[0]['Feature']}")
    

### [Example 20] SHAP Analysis: Prediction Interpretation
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: [Example 20] SHAP Analysis: Prediction Interpretation
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import shap
    
    # SHAP Explainerofcreate
    explainer = shap.Explainer(model_lgb, X_train)
    shap_values = explainer(X_test)
    
    print("=" * 60)
    print("SHAPminanalysis")
    print("=" * 60)
    print("SHAPvaluecalculationcomplete")
    print(f"SHAPvalueofshape: {shap_values.values.shape}")
    
    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title('SHAP Summary Plot: Feature Impact on LSPR Prediction',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # SHAP Dependence Plot（most important feature）
    top_feature_idx = importance_df.index[0]
    top_feature_name = X.columns[top_feature_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values.values, X_test,
                         feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature_name}',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nSHAPminanalysistofrom、{top_feature_name}LSPRwavelengthpredictiontomost influentialandconfirmed")
    

* * *

## 3.7 Nanomaterial Design with Bayesian Optimization

Goal：GoalLSPRwavelength（550 nm）implementachieveoptimalfor synthesis condition search

### [Example 21] Search Space Definition
    
    
    from skopt.space import Real
    
    # search space definition
    # size: 10-40 nm、temperature: 20-80°C、pH: 4-10
    search_space = [
        Real(10, 40, name='size_nm'),
        Real(20, 80, name='temperature_C'),
        Real(4, 10, name='pH')
    ]
    
    print("=" * 60)
    print("Bayesian optimization：search space definition")
    print("=" * 60)
    for dim in search_space:
        print(f"  {dim.name}: [{dim.bounds[0]}, {dim.bounds[1]}]")
    
    print("\nGoal: LSPRwavelength = 550 nm implementachieve condition search")
    

### [Example 22] Objective Function Setup
    
    
    # objective function：predictionLSPRwavelengthandGoalwavelength（550 nm）ofdifferenceofabsolutevalueminimization
    target_lspr = 550.0
    
    def objective_function(params):
        """
        Bayesian optimizationofobjective function
    
        Parameters:
        -----------
        params : list
            [size_nm, temperature_C, pH]
    
        Returns:
        --------
        float
            Goalwavelengthandoferror（minimizationperformvalue）
        """
        # parametersacquisition
        size, temp, ph = params
    
        # featuresconstruction（scalingapplication）
        features = np.array([[size, temp, ph]])
        features_scaled = scaler.transform(features)
    
        # LSPRwavelengthprediction
        predicted_lspr = model_lgb.predict(features_scaled)[0]
    
        # Goalwavelengthandoferror（absolutevalue）
        error = abs(predicted_lspr - target_lspr)
    
        return error
    
    # test implementationrows
    test_params = [20.0, 50.0, 7.0]
    test_error = objective_function(test_params)
    print(f"\ntest implementationrows:")
    print(f"  parameters: size={test_params[0]} nm, temp={test_params[1]}°C, pH={test_params[2]}")
    print(f"  objective functionvalue（error）: {test_error:.4f} nm")
    

### [Example 23] Running Bayesian Optimization (scikit-optimize)
    
    
    from skopt import gp_minimize
    from skopt.plots import plot_convergence, plot_objective
    
    # Bayesian optimizationofimplementationrows
    print("\n" + "=" * 60)
    print("Bayesian optimizationofimplementationrowsmiddle...")
    print("=" * 60)
    
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=50,  # Evaluationnumber of times
        n_initial_points=10,  # random sampling count
        random_state=42,
        verbose=False
    )
    
    print("optimizationcomplete！")
    print("\n" + "=" * 60)
    print("optimizationresults")
    print("=" * 60)
    print(f"minimumobjective functionvalue（error）: {result.fun:.4f} nm")
    print(f"\noptimalparameters:")
    print(f"  size: {result.x[0]:.2f} nm")
    print(f"  temperature: {result.x[1]:.2f} °C")
    print(f"  pH: {result.x[2]:.2f}")
    
    # under optimal conditionsofpredictionLSPRwavelengthcalculation
    optimal_features = np.array([result.x])
    optimal_features_scaled = scaler.transform(optimal_features)
    predicted_optimal_lspr = model_lgb.predict(optimal_features_scaled)[0]
    
    print(f"\npredictionis performedLSPRwavelength: {predicted_optimal_lspr:.2f} nm")
    print(f"GoalLSPRwavelength: {target_lspr} nm")
    print(f"Achievedaccuracy: {abs(predicted_optimal_lspr - target_lspr):.2f} nm")
    

### [Example 24] Optimization Result Visualization
    
    
    # Optimizationprocessofvisualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # convergence plot
    plot_convergence(result, ax=axes[0])
    axes[0].set_title('Convergence Plot', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Objective Value (Error, nm)', fontsize=11)
    axes[0].set_xlabel('Number of Evaluations', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Evaluationhistoryofplot
    iterations = range(1, len(result.func_vals) + 1)
    axes[1].plot(iterations, result.func_vals, 'o-', alpha=0.6, label='Evaluation')
    axes[1].plot(iterations, np.minimum.accumulate(result.func_vals),
                 'r-', linewidth=2, label='Best So Far')
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Objective Value (Error, nm)', fontsize=11)
    axes[1].set_title('Optimization Progress', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### [Example 25] Convergence Plot
    
    
    # detailedconvergence plot（bestvalueofprogression）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cumulative_min = np.minimum.accumulate(result.func_vals)
    iterations = np.arange(1, len(cumulative_min) + 1)
    
    ax.plot(iterations, cumulative_min, 'b-', linewidth=2, marker='o',
            markersize=4, label='Best Error')
    ax.axhline(y=result.fun, color='r', linestyle='--', linewidth=2,
               label=f'Final Best: {result.fun:.2f} nm')
    ax.fill_between(iterations, 0, cumulative_min, alpha=0.2, color='blue')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Minimum Error (nm)', fontsize=12)
    ax.set_title('Bayesian Optimization: Convergence to Optimal Solution',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{len(result.func_vals)}timesofevaluationwithoptimalsolutiontoconverge")
    print(f"initialevaluationwithofbesterror: {result.func_vals[0]:.2f} nm")
    print(f"final besterror: {result.fun:.2f} nm")
    print(f"improvement rate: {(1 - result.fun/result.func_vals[0])*100:.1f}%")
    

* * *

## 3.8 Multi-Objective Optimization: Size and Emission Efficiency Trade-offs

### [Example 26] Pareto Optimization (NSGA-II)

multi-objective optimizationin、multipleofsimultaneous objectivestooptimization|。herein、quantum dotsofsizeminimization、emissionefficiency（virtual metric）maximization|。
    
    
    # pymoouse|multi-objective optimization
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
    
        # multi-objective optimizationproblemofdefinition
        class QuantumDotOptimization(Problem):
            def __init__(self):
                super().__init__(
                    n_var=3,  # number of variables（size, synthesis_time, precursor_ratio）
                    n_obj=2,  # objective functionnumber（sizeminimization、emissionefficiency maximization）
                    n_constr=0,  # no constraints
                    xl=np.array([2.0, 10.0, 0.5]),  # Lower bound
                    xu=np.array([10.0, 120.0, 2.0])  # Upper bound
                )
    
            def _evaluate(self, X, out, *args, **kwargs):
                # objective function1: sizeofminimization
                obj1 = X[:, 0]  # size
    
                # objective function2: emissionefficiencyofmaximization（negativeofvaluewithminimizationproblemtotransformation）
                # efficiencyisvirtualto、emission wavelength550 nmtohigher when closerandassumption
                features = X  # [size, synthesis_time, precursor_ratio]
                features_scaled = scaler_qd.transform(features)
                predicted_emission = model_qd.predict(features_scaled)
    
                # efficiency：550 nmfromofhigher when deviation is smaller（negativeofvaluewithmaximization→minimization）
                efficiency = -np.abs(predicted_emission - 550)
                obj2 = -efficiency  # maximizationminimizationproblemtotransformation
    
                out["F"] = np.column_stack([obj1, obj2])
    
        # problemofinstantiation
        problem = QuantumDotOptimization()
    
        # NSGA-IIalgorithm
        algorithm = NSGA2(
            pop_size=40,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
    
        # Optimizationimplementationrows
        print("=" * 60)
        print("multi-objective optimization（NSGA-II）implementationrowsmiddle...")
        print("=" * 60)
    
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', 50),  # number of generations
            seed=42,
            verbose=False
        )
    
        print("multi-objective optimizationcomplete！")
        print(f"\nParetooptimalsolutionofnumber: {len(res.F)}")
    
        # Paretooptimalsolutionofdisplay（top5）
        print("\nrepresentative Paretooptimalsolution（top5）:")
        pareto_solutions = pd.DataFrame({
            'Size (nm)': res.X[:, 0],
            'Synthesis Time (min)': res.X[:, 1],
            'Precursor Ratio': res.X[:, 2],
            'Obj1: Size': res.F[:, 0],
            'Obj2: -Efficiency': res.F[:, 1]
        }).head(5)
        print(pareto_solutions.to_string(index=False))
    
        PYMOO_AVAILABLE = True
    
    except ImportError:
        print("=" * 60)
        print("pymoonot installed")
        print("=" * 60)
        print("multi-objective optimizationtoispymoorequiredwith|:")
        print("  pip install pymoo")
        print("\ninsteadto、simplifiedmulti-objective optimizationofExampledisplay|")
    
        # simplifiedgrid searchbymulti-objective optimizationofsimulated
        sizes = np.linspace(2, 10, 20)
        times = np.linspace(10, 120, 20)
        ratios = np.linspace(0.5, 2.0, 20)
    
        # grid search（sampling）
        sample_X = []
        sample_F = []
    
        for size in sizes[::4]:
            for time in times[::4]:
                for ratio in ratios[::4]:
                    features = np.array([[size, time, ratio]])
                    features_scaled = scaler_qd.transform(features)
                    emission = model_qd.predict(features_scaled)[0]
    
                    obj1 = size
                    obj2 = abs(emission - 550)
    
                    sample_X.append([size, time, ratio])
                    sample_F.append([obj1, obj2])
    
        sample_X = np.array(sample_X)
        sample_F = np.array(sample_F)
    
        print("\ngrid searchbysolutionofsearchcomplete")
        print(f"searched solutionsofnumber: {len(sample_F)}")
    
        res = type('Result', (), {
            'X': sample_X,
            'F': sample_F
        })()
    
        PYMOO_AVAILABLE = False
    

### [Example 27] Pareto Front Visualization
    
    
    # Paretofrontofvisualization
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if PYMOO_AVAILABLE:
        # NSGA-IIofresultsplot
        ax.scatter(res.F[:, 0], -res.F[:, 1], c='blue', s=80, alpha=0.6,
                   edgecolors='black', label='Pareto Optimal Solutions')
    
        title_suffix = "(NSGA-II)"
    else:
        # grid searchofresultsplot
        ax.scatter(res.F[:, 0], res.F[:, 1], c='blue', s=60, alpha=0.5,
                   edgecolors='black', label='Sampled Solutions')
    
        title_suffix = "(Grid Search)"
    
    ax.set_xlabel('Objective 1: Size (nm) [Minimize]', fontsize=12)
    
    if PYMOO_AVAILABLE:
        ax.set_ylabel('Objective 2: Efficiency [Maximize]', fontsize=12)
    else:
        ax.set_ylabel('Objective 2: Deviation from 550nm [Minimize]', fontsize=12)
    
    ax.set_title(f'Pareto Front: Size vs Emission Efficiency {title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nPareto front:")
    print("  sizereduceandefficiency decreases、increase efficiencyandsizebecomes larger")
    print("  → trade-offrelationshipclearto")
    

* * *

## 3.9 TEM Image Analysis and Size Distribution

### [Example 28] Simulated TEM Data Generation

TEM（transmission electron microscope）withmeasured nanoparticlessizeis、approximatelylog-normalmindistributiontofollows。
    
    
    from scipy.stats import lognorm
    
    # log-normalminfollows distributionTEMsizedataofgeneration
    np.random.seed(200)
    
    # parameters
    mean_size = 20  # Average size（nm）
    cv = 0.3  # coefficient of variation（Standard deviation/average）
    
    # log-normalmindistributionofparameterscalculation
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_size) - 0.5 * sigma**2
    
    # samplesgeneration（500particles）
    tem_sizes = lognorm.rvs(s=sigma, scale=np.exp(mu), size=500)
    
    print("=" * 60)
    print("TEMmeasurement dataofgeneration（log-normalmindistribution）")
    print("=" * 60)
    print(f"number of samples: {len(tem_sizes)}particles")
    print(f"Average size: {tem_sizes.mean():.2f} nm")
    print(f"Standard deviation: {tem_sizes.std():.2f} nm")
    print(f"Median: {np.median(tem_sizes):.2f} nm")
    print(f"minimumvalue: {tem_sizes.min():.2f} nm")
    print(f"maximumvalue: {tem_sizes.max():.2f} nm")
    
    # histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tem_sizes, bins=40, alpha=0.7, color='lightblue',
            edgecolor='black', density=True, label='TEM Data')
    ax.set_xlabel('Particle Size (nm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('TEM Size Distribution (Lognormal)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    

### [Example 29] Log-Normal Distribution Fitting
    
    
    # log-normalmindistributionoffitting
    shape_fit, loc_fit, scale_fit = lognorm.fit(tem_sizes, floc=0)
    
    # fittedmindistributionofparameters
    fitted_mean = np.exp(np.log(scale_fit) + 0.5 * shape_fit**2)
    fitted_std = fitted_mean * np.sqrt(np.exp(shape_fit**2) - 1)
    
    print("=" * 60)
    print("log-normalmindistribution fittingresults")
    print("=" * 60)
    print(f"shapeparameters (sigma): {shape_fit:.4f}")
    print(f"scaleparameters: {scale_fit:.4f}")
    print(f"fittedAverage size: {fitted_mean:.2f} nm")
    print(f"fittedStandard deviation: {fitted_std:.2f} nm")
    
    # measuredvalueandofcomparison
    print(f"\nmeasuredvalueandofcomparison:")
    print(f"  Average size - measured: {tem_sizes.mean():.2f} nm, fit: {fitted_mean:.2f} nm")
    print(f"  Standard deviation - measured: {tem_sizes.std():.2f} nm, fit: {fitted_std:.2f} nm")
    

### [Example 30] Fitting Result Visualization
    
    
    # fittingresultsofdetailsvisualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # histogramandfitting curve
    axes[0].hist(tem_sizes, bins=40, alpha=0.6, color='lightblue',
                 edgecolor='black', density=True, label='TEM Data')
    
    # fittedlog-normalmindistribution
    x_range = np.linspace(0, tem_sizes.max(), 200)
    fitted_pdf = lognorm.pdf(x_range, shape_fit, loc=loc_fit, scale=scale_fit)
    axes[0].plot(x_range, fitted_pdf, 'r-', linewidth=2,
                 label=f'Lognormal Fit (μ={fitted_mean:.1f}, σ={fitted_std:.1f})')
    
    axes[0].set_xlabel('Particle Size (nm)', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('TEM Size Distribution with Lognormal Fit',
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Q-Qplot（minquantile pointsplot）
    from scipy.stats import probplot
    
    probplot(tem_sizes, dist=lognorm, sparams=(shape_fit, loc_fit, scale_fit),
             plot=axes[1])
    axes[1].set_title('Q-Q Plot: Lognormal Distribution',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nQ-Qplot: data on straight linetoif、log-normalmindistributiontofollows well")
    

* * *

## 3.10 Molecular Dynamics (MD) Data Analysis

### [Example 31] Loading MD Simulation Data

minmolecular dynamicssimulationin、nanoparticlesofatomic configurationoftrack time evolution。
    
    
    # MDsimulation dataofsimulated generation
    # actualofMDdataisLAMMPS, GROMACSobtained from
    
    np.random.seed(300)
    
    n_atoms = 100  # number of atoms
    n_steps = 1000  # number of timesteps
    dt = 0.001  # timestep（ps）
    
    # initial position（nm）
    positions_initial = np.random.uniform(-1, 1, (n_atoms, 3))
    
    # time evolutionofsimulated（random walk）
    positions = np.zeros((n_steps, n_atoms, 3))
    positions[0] = positions_initial
    
    for t in range(1, n_steps):
        # random displacement
        displacement = np.random.normal(0, 0.01, (n_atoms, 3))
        positions[t] = positions[t-1] + displacement
    
    print("=" * 60)
    print("MDsimulation dataofgeneration")
    print("=" * 60)
    print(f"number of atoms: {n_atoms}")
    print(f"number of timesteps: {n_steps}")
    print(f"simulation time: {n_steps * dt:.2f} ps")
    print(f"data shape: {positions.shape} (time, atoms, xyz)")
    
    # middlecentral atom（atom0）oftrajectoryplot
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2],
             'b-', alpha=0.5, linewidth=0.5)
    ax1.scatter(positions[0, 0, 0], positions[0, 0, 1], positions[0, 0, 2],
                c='green', s=100, label='Start', edgecolors='k')
    ax1.scatter(positions[-1, 0, 0], positions[-1, 0, 1], positions[-1, 0, 2],
                c='red', s=100, label='End', edgecolors='k')
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_title('Atom Trajectory (Atom 0)', fontweight='bold')
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 0], label='X')
    ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 1], label='Y')
    ax2.plot(np.arange(n_steps) * dt, positions[:, 0, 2], label='Z')
    ax2.set_xlabel('Time (ps)', fontsize=11)
    ax2.set_ylabel('Position (nm)', fontsize=11)
    ax2.set_title('Position vs Time (Atom 0)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### [Example 32] Radial Distribution Function (RDF) Calculation

radial distribution function（Radial Distribution Function, RDF）is、interatomic distanceofmindistributionrepresents。
    
    
    # radial distribution function（RDF）calculation
    def calculate_rdf(positions, r_max=2.0, n_bins=100):
        """
        radial distribution functioncalculation
    
        Parameters:
        -----------
        positions : ndarray
            atom positions (n_atoms, 3)
        r_max : float
            maximum distance（nm）
        n_bins : int
            number of bins
    
        Returns:
        --------
        r_bins : ndarray
            distance bins
        rdf : ndarray
            radial distribution function
        """
        n_atoms = positions.shape[0]
    
        # all atom pairsofdistance calculation
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < r_max:
                    distances.append(dist)
    
        distances = np.array(distances)
    
        # histogram
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
        r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # normalization（ideal gasandofratio）
        dr = r_max / n_bins
        volume_shell = 4 * np.pi * r_bins**2 * dr
        n_ideal = volume_shell * (n_atoms / (4/3 * np.pi * r_max**3))
    
        rdf = hist / n_ideal / (n_atoms / 2)
    
        return r_bins, rdf
    
    # final framewithRDFcalculation
    final_positions = positions[-1]
    r_bins, rdf = calculate_rdf(final_positions, r_max=1.5, n_bins=150)
    
    print("=" * 60)
    print("radial distribution function（RDF）")
    print("=" * 60)
    print(f"calculationcomplete: {len(r_bins)}unitsofbin")
    
    # RDFofplot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_bins, rdf, 'b-', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=1, label='Ideal Gas (g(r)=1)')
    ax.set_xlabel('Distance r (nm)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('Radial Distribution Function (RDF)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(rdf) * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # peak positionofdetection
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(rdf, height=1.2, distance=10)
    print(f"\nRDFofpeak position（characteristic interatomic distance）:")
    for i, peak_idx in enumerate(peaks[:3], 1):
        print(f"  peak{i}: r = {r_bins[peak_idx]:.3f} nm, g(r) = {rdf[peak_idx]:.2f}")
    

### [Example 33] Diffusion Coefficient Calculation (Mean Squared Displacement)
    
    
    # mean squared displacement（MSD）calculation
    def calculate_msd(positions):
        """
        mean squared displacementcalculation
    
        Parameters:
        -----------
        positions : ndarray
            atom positions (n_steps, n_atoms, 3)
    
        Returns:
        --------
        msd : ndarray
            mean squared displacement (n_steps,)
        """
        n_steps, n_atoms, _ = positions.shape
        msd = np.zeros(n_steps)
    
        # each timestepwithofMSD
        for t in range(n_steps):
            displacement = positions[t] - positions[0]
            squared_displacement = np.sum(displacement**2, axis=1)
            msd[t] = np.mean(squared_displacement)
    
        return msd
    
    # MSDcalculation
    msd = calculate_msd(positions)
    time = np.arange(n_steps) * dt
    
    print("=" * 60)
    print("mean squared displacement（MSD）anddiffusion coefficient")
    print("=" * 60)
    
    # diffusion coefficientcalculation（Einsteinrelationship equation: MSD = 6*D*t）
    # linearfit（latter half50%ofdatause）
    start_idx = n_steps // 2
    fit_coeffs = np.polyfit(time[start_idx:], msd[start_idx:], 1)
    slope = fit_coeffs[0]
    diffusion_coefficient = slope / 6
    
    print(f"diffusion coefficient D = {diffusion_coefficient:.6f} nm²/ps")
    print(f"            = {diffusion_coefficient * 1e3:.6f} × 10⁻⁶ cm²/s")
    
    # MSDplot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, msd, 'b-', linewidth=2, label='MSD')
    ax.plot(time[start_idx:], fit_coeffs[0] * time[start_idx:] + fit_coeffs[1],
            'r--', linewidth=2, label=f'Linear Fit (D={diffusion_coefficient:.4f} nm²/ps)')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('MSD (nm²)', fontsize=12)
    ax.set_title('Mean Squared Displacement (MSD)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\ndiffusion coefficientis、nanoparticlesofquantitatively mobilitytoevaluationimportant indicatorwith|")
    

* * *

## 3.11 Anomaly Detection: Quality Control Applications

### [Example 34] Anomalous Nanoparticle Detection with Isolation Forest

manufacturingprocesswithgenerated nanoparticlesofquality controlto、machine learningbyanomaly detectionapplication|。
    
    
    from sklearn.ensemble import IsolationForest
    
    # normaldataandanomalymix data
    np.random.seed(400)
    
    # normalgold nanoparticlesdata（180samples）
    normal_size = np.random.normal(15, 3, 180)
    normal_lspr = 520 + 0.8 * (normal_size - 15) + np.random.normal(0, 3, 180)
    
    # anomalynanoparticlesdata（20samples）：sizeanomalytolargeorsmall
    anomaly_size = np.concatenate([
        np.random.uniform(5, 8, 10),  # anomalytosmall
        np.random.uniform(35, 50, 10)  # anomalytolarge
    ])
    anomaly_lspr = 520 + 0.8 * (anomaly_size - 15) + np.random.normal(0, 8, 20)
    
    # combine all data
    all_size = np.concatenate([normal_size, anomaly_size])
    all_lspr = np.concatenate([normal_lspr, anomaly_lspr])
    all_data = np.column_stack([all_size, all_lspr])
    
    # label（normal=0、anomaly=1）
    true_labels = np.concatenate([np.zeros(180), np.ones(20)])
    
    print("=" * 60)
    print("anomaly detection（Isolation Forest）")
    print("=" * 60)
    print(f"Total data: {len(all_data)}")
    print(f"normaldata: {int((true_labels == 0).sum())}samples")
    print(f"anomalydata: {int((true_labels == 1).sum())}samples")
    
    # Isolation Forestmodel
    iso_forest = IsolationForest(
        contamination=0.1,  # anomalydataofsplittotal（10%andassumption）
        random_state=42,
        n_estimators=100
    )
    
    # anomaly detection
    predictions = iso_forest.fit_predict(all_data)
    anomaly_scores = iso_forest.score_samples(all_data)
    
    # Predictionresults（1: normal、-1: anomaly）
    predicted_anomalies = (predictions == -1)
    true_anomalies = (true_labels == 1)
    
    # Evaluation metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Prediction0/1totransformation
    predicted_labels = (predictions == -1).astype(int)
    
    print("\nconfusionrowscolumn:")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    
    print("\nminclassification report:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=['Normal', 'Anomaly']))
    
    # detection rate
    detected_anomalies = np.sum(predicted_anomalies & true_anomalies)
    total_anomalies = np.sum(true_anomalies)
    detection_rate = detected_anomalies / total_anomalies * 100
    
    print(f"\nanomalydetection rate: {detection_rate:.1f}% ({detected_anomalies}/{total_anomalies})")
    

### [Example 35] Anomaly Sample Visualization
    
    
    # anomaly detectionresultsofvisualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # scatter plot（trueoflabel）
    axes[0].scatter(all_size[true_labels == 0], all_lspr[true_labels == 0],
                    c='blue', s=60, alpha=0.6, label='Normal', edgecolors='k')
    axes[0].scatter(all_size[true_labels == 1], all_lspr[true_labels == 1],
                    c='red', s=100, alpha=0.8, marker='^', label='True Anomaly',
                    edgecolors='k', linewidths=2)
    axes[0].set_xlabel('Size (nm)', fontsize=12)
    axes[0].set_ylabel('LSPR Wavelength (nm)', fontsize=12)
    axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # scatter plot（predictionresults）
    normal_mask = ~predicted_anomalies
    anomaly_mask = predicted_anomalies
    
    axes[1].scatter(all_size[normal_mask], all_lspr[normal_mask],
                    c='blue', s=60, alpha=0.6, label='Predicted Normal', edgecolors='k')
    axes[1].scatter(all_size[anomaly_mask], all_lspr[anomaly_mask],
                    c='orange', s=100, alpha=0.8, marker='X', label='Predicted Anomaly',
                    edgecolors='k', linewidths=2)
    
    # correctly detectedanomalyemphasis
    correctly_detected = predicted_anomalies & true_anomalies
    axes[1].scatter(all_size[correctly_detected], all_lspr[correctly_detected],
                    c='red', s=150, marker='*', label='Correctly Detected',
                    edgecolors='black', linewidths=1.5, zorder=5)
    
    axes[1].set_xlabel('Size (nm)', fontsize=12)
    axes[1].set_ylabel('LSPR Wavelength (nm)', fontsize=12)
    axes[1].set_title('Isolation Forest Predictions', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # anomaly scoreofmindistribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(anomaly_scores[true_labels == 0], bins=30, alpha=0.6,
            color='blue', label='Normal', edgecolor='black')
    ax.hist(anomaly_scores[true_labels == 1], bins=30, alpha=0.6,
            color='red', label='Anomaly', edgecolor='black')
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\nanomaly scorelow（negativeofvaluelarge）the more、anomalywithhigh possibility")
    

* * *

## Summary

this chapterin、PythonusedNanomaterial Data Analysis and Machine Learningofpractical approachmethod35unitsofcodeExamplewithlearned。

### Key Skills Acquired

  1. **data generationandvisualization** （Example1-5） \- gold nanoparticles、quantum dotsofsynthesisdata generation \- histogram、scatter plot、3Dplot、correlationminanalysis

  2. **data preprocessing** （Example6-9） \- missing value handling、outlier detection、scaling、Data Split

  3. **timesregressionmodelbyphysical propertiesprediction** （Example10-15） \- linear regression、random forest、LightGBM、SVR、MLP \- modelperformance comparison（R²、RMSE、MAE）

  4. **quantum dotsemissionprediction** （Example16-18） \- Brusequationtobased ondata generation \- LightGBMbypredictionmodelconstruction

  5. **feature importanceandmodel interpretation** （Example19-20） \- LightGBMfeature importance \- SHAPminby analysispredictionofinterpretation

  6. **Bayesian optimization** （Example21-25） \- GoalLSPRwavelengthimplementachieveoptimalsynthesis conditionsofsearch \- convergence plot、optimizationprocessofvisualization

  7. **multi-objective optimization** （Example26-27） \- NSGA-IIbyParetooptimization \- sizeand emissionefficiencyoftrade-offminanalysis

  8. **TEMimage analysis** （Example28-30） \- log-normalmindistributionbysizemindistribution fitting \- Q-Qplotbymindistributionofvalidation

  9. **minmolecular dynamicsdata analysis** （Example31-33） \- atomic trajectoryofvisualization \- radial distribution function（RDF）calculation \- diffusion coefficientofcalculation（MSDmethod）

  10. **anomaly detection** （Example34-35）

     * Isolation Forestbyquality control
     * anomalynanoparticlesofselfmotion detection

### Practical Applications

theseoftechniquesis、or lessoflike actualofnanomaterials researchtodirect applicationwithcan：

  * **materials design** : machine learningbyphysical propertiespredictionandoptimizationbyhigh-efficiency materials search
  * **processoptimization** : Bayesian optimizationbyexperimentnumber of timesreductionandoptimalsynthesis condition discovery
  * **quality control** : anomaly detectionbydefective productsofearly detectionandyield improvement
  * **data analysis** : TEMdata、MDsimulation dataofquantitative solutionanalysis
  * **model interpretation** : SHAPminby analysispredictionbasisofvisualizationandreliability improvement

### Preview of Next Chapter

Chapter 4in、theseoftechniquesimplementwhenofnanomaterials research projecttoapplication|5oflearn detailed case studies。carbon nanotube composite materials、quantum dots、gold nanoparticlescatalysts、graphene、nanomedicineofpractical use casesExamplethrough、problem solvingofunderstand overall picture。

* * *

## Exerciseproblem

### Exercise1: carbon nanotubesofelectrical conductivityprediction

carbon nanotubes（CNT）ofelectrical conductivityis、diameter、chirality、lengthdepends on。or lessofdatageneration、LightGBMmodelwithpredictionplease。

**data specifications** ： \- number of samples：150 \- features：diameter（1-3 nm）、length（100-1000 nm）、chirality index（0-1ofcontinuousvalue） \- target：electrical conductivity（10³-10⁷ S/m、log-normalmindistribution）

**task** ： 1\. data generation 2\. training/Test dataminsplit 3\. LightGBMmodelofconstructionandevaluation 4\. feature importanceofvisualization

Sample Solution
    
    
    # data generation
    np.random.seed(500)
    n_samples = 150
    
    diameter = np.random.uniform(1, 3, n_samples)
    length = np.random.uniform(100, 1000, n_samples)
    chirality = np.random.uniform(0, 1, n_samples)
    
    # electrical conductivity（simplemodel: diameterandchiralitytostrongly dependent）
    log_conductivity = 3 + 2*diameter + 3*chirality + 0.001*length + np.random.normal(0, 0.5, n_samples)
    conductivity = 10 ** log_conductivity  # S/m
    
    data_cnt = pd.DataFrame({
        'diameter_nm': diameter,
        'length_nm': length,
        'chirality': chirality,
        'conductivity_Sm': conductivity
    })
    
    # featuresandtarget
    X_cnt = data_cnt[['diameter_nm', 'length_nm', 'chirality']]
    y_cnt = np.log10(data_cnt['conductivity_Sm'])  # logarithmic transformation
    
    # scaling
    scaler_cnt = StandardScaler()
    X_cnt_scaled = scaler_cnt.fit_transform(X_cnt)
    
    # training/testminsplit
    X_cnt_train, X_cnt_test, y_cnt_train, y_cnt_test = train_test_split(
        X_cnt_scaled, y_cnt, test_size=0.2, random_state=42
    )
    
    # LightGBMmodel
    model_cnt = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42, verbose=-1)
    model_cnt.fit(X_cnt_train, y_cnt_train)
    
    # Predictionandevaluation
    y_cnt_pred = model_cnt.predict(X_cnt_test)
    r2_cnt = r2_score(y_cnt_test, y_cnt_pred)
    rmse_cnt = np.sqrt(mean_squared_error(y_cnt_test, y_cnt_pred))
    
    print(f"R²: {r2_cnt:.4f}")
    print(f"RMSE: {rmse_cnt:.4f}")
    
    # Feature importance
    importance_cnt = pd.DataFrame({
        'Feature': X_cnt.columns,
        'Importance': model_cnt.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nfeature importance:")
    print(importance_cnt)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_cnt['Feature'], importance_cnt['Importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance: CNT Conductivity Prediction')
    plt.tight_layout()
    plt.show()
    

### Exercise2: silver nanoparticlesofoptimalsynthesis conditionssearch

silver nanoparticlesofantibacterial activityis、sizesmaller is higher。Bayesian optimizationusing、Target size（10 nm）implementachieveoptimalfor synthesistemperatureandpHplease search。

**conditions** ： \- temperaturerange：20-80°C \- pHrange：6-11 \- Target size：10 nm

Sample Solution
    
    
    # silver nanoparticlesdataofgeneration
    np.random.seed(600)
    n_ag = 100
    
    temp_ag = np.random.uniform(20, 80, n_ag)
    pH_ag = np.random.uniform(6, 11, n_ag)
    
    # sizemodel（temperaturehigh、pHsmaller with lowerandassumption）
    size_ag = 15 - 0.1*temp_ag - 0.8*pH_ag + np.random.normal(0, 1, n_ag)
    size_ag = np.clip(size_ag, 5, 30)
    
    data_ag = pd.DataFrame({
        'temperature': temp_ag,
        'pH': pH_ag,
        'size': size_ag
    })
    
    # modelconstruction（LightGBM）
    X_ag = data_ag[['temperature', 'pH']]
    y_ag = data_ag['size']
    
    scaler_ag = StandardScaler()
    X_ag_scaled = scaler_ag.fit_transform(X_ag)
    
    model_ag = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1)
    model_ag.fit(X_ag_scaled, y_ag)
    
    # Bayesian optimization
    from skopt import gp_minimize
    from skopt.space import Real
    
    space_ag = [
        Real(20, 80, name='temperature'),
        Real(6, 11, name='pH')
    ]
    
    target_size = 10.0
    
    def objective_ag(params):
        temp, ph = params
        features = scaler_ag.transform([[temp, ph]])
        predicted_size = model_ag.predict(features)[0]
        return abs(predicted_size - target_size)
    
    result_ag = gp_minimize(objective_ag, space_ag, n_calls=40, random_state=42, verbose=False)
    
    print("=" * 60)
    print("silver nanoparticlesofoptimalsynthesis conditions")
    print("=" * 60)
    print(f"minimumerror: {result_ag.fun:.2f} nm")
    print(f"optimaltemperature: {result_ag.x[0]:.1f} °C")
    print(f"optimalpH: {result_ag.x[1]:.2f}")
    
    # under optimal conditionsofpredictionsize
    optimal_features = scaler_ag.transform([result_ag.x])
    predicted_size = model_ag.predict(optimal_features)[0]
    print(f"predictionsize: {predicted_size:.2f} nm")
    

### Exercise3: quantum dotsofmulti-color emission design

red（650 nm）、green（550 nm）、blue（450 nm）of3colorofemissionimplementachieveCdSequantum dotsofsize、Bayesian optimizationwithplease design。

**hint** ： \- for each colorandtooptimizationimplementrows \- emissionwavelengthandsizeofrelationshipuse

Sample Solution
    
    
    # quantum dotsdata（Example16ofdata_qduse）
    # model_qd and scaler_qd constructioncompletedandassumption
    
    # 3colorofGoalwavelength
    target_colors = {
        'Red': 650,
        'Green': 550,
        'Blue': 450
    }
    
    results_colors = {}
    
    for color_name, target_emission in target_colors.items():
        # search space
        space_qd = [
            Real(2, 10, name='size_nm'),
            Real(10, 120, name='synthesis_time_min'),
            Real(0.5, 2.0, name='precursor_ratio')
        ]
    
        # objective function
        def objective_qd(params):
            features = scaler_qd.transform([params])
            predicted_emission = model_qd.predict(features)[0]
            return abs(predicted_emission - target_emission)
    
        # Optimization
        result_qd_color = gp_minimize(objective_qd, space_qd, n_calls=30, random_state=42, verbose=False)
    
        # resultssave
        optimal_features = scaler_qd.transform([result_qd_color.x])
        predicted_emission = model_qd.predict(optimal_features)[0]
    
        results_colors[color_name] = {
            'target': target_emission,
            'size': result_qd_color.x[0],
            'time': result_qd_color.x[1],
            'ratio': result_qd_color.x[2],
            'predicted': predicted_emission,
            'error': result_qd_color.fun
        }
    
    # resultsdisplay
    print("=" * 80)
    print("quantum dotsmulti-color emission design")
    print("=" * 80)
    
    results_df = pd.DataFrame(results_colors).T
    print(results_df.to_string())
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_rgb = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
    
    for color_name, result in results_colors.items():
        ax.scatter(result['size'], result['predicted'],
                   s=200, color=colors_rgb[color_name],
                   edgecolors='black', linewidths=2, label=color_name)
    
    ax.set_xlabel('Quantum Dot Size (nm)', fontsize=12)
    ax.set_ylabel('Emission Wavelength (nm)', fontsize=12)
    ax.set_title('Multi-Color Quantum Dot Design', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 3.12 End-of-Chapter Checklist: Quality Assurance of Nanomaterial Data Analysis Skills

this chapterwithlearnedPythonbyNanomaterial Data Analysis and Machine Learningofimplementationskillssystematictocheck。

### 3.12.1 Environment Setup Skills（Environment Setup）

#### Foundation Level

  * [ ] Python 3.9or moreinstalled
  * [ ] 3ofenvironmentconstructionoptional（Anaconda/venv/Colab）ofexplain differencescan
  * [ ] selfminofsituationtooptimalselect environmentcan
  * [ ] virtual environmentcreateactivate/deactivatecan
  * [ ] pip/condawithlibraryinstallcan（pandas、numpy、matplotlib、scikit-learn、lightgbm）
  * [ ] environment verification codeimplementrows、confirm error-free operationcan

#### Applied Level

  * [ ] requirements.txtcreateusecan
  * [ ] Google ColabwithGoogle Drivemount and load data
  * [ ] multipleofvirtual environment usageclassificationtousemincan be done
  * [ ] installation errorselfcapabilitywithtroubleshootingcan
  * [ ] optionallibrary（pymoo、SHAP）requiredtoinstall according tocan

* * *

### 3.12.2 Data Processing & Visualization Skills（Data Processing & Visualization）

#### Foundation Level

  * [ ] NumPywithsynthetic data generationcan（normalmindistribution、uniformmindistribution）
  * [ ] PandaswithDataFramecreateoperationscan
  * [ ] basic statistics（mean、std、median）can calculate
  * [ ] histogramcan create
  * [ ] scatter plotcan create
  * [ ] missingvaluedetectioncan（`isnull().sum()`）
  * [ ] missingvaluedeleteisimputationcan（`dropna()` or `fillna()`）

#### Applied Level

  * [ ] correlationrowscolumn calculationvisualizationcan（`corr()`、seaborn.heatmap）
  * [ ] pairplot（scatter plot matrix）can create（seaborn.pairplot）
  * [ ] 3Dscatter plotcan create（mpl_toolkits.mplot3d）
  * [ ] KDE（kernel density estimation）usecan
  * [ ] outliervalueIQRmethodwithdetectioncan
  * [ ] StandardScalerwithdata standardizationcan
  * [ ] train_test_splitwithdataminsplitcan（80% vs 20%）
  * [ ] `random_state=42`withensure reproducibility

#### Advanced Level

  * [ ] log-normalmindistribution fittingcan be done（scipy.stats.lognorm）
  * [ ] Q-Qplotwithmindistributionofgoodness of fit verificationcan
  * [ ] effective colormaptousecan（viridis、plasma、coolwarm）
  * [ ] multipleofsubplotsusing advancedvisualizationcan be done

* * *

### 3.12.3 Machine Learning Model Implementation Skills（ML Model Implementation）

#### Foundation Level（5ofmodelimplementation）

  * [ ] linear regressionimplement and、coefficientofmeaning explanationcan
  * [ ] random forestimplement and、`n_estimators`ofrolesplitexplanationcan
  * [ ] LightGBMinstallimplementationcan
  * [ ] SVRwithstandardization（StandardScaler）ofunderstand necessity
  * [ ] MLPRegressor（neural network）can implement

#### Applied Level（modelselectionandevaluation）

  * [ ] MAE、R²、RMSEcalculation and interpretationcan
  * [ ] Training dataandTest dataofperformance differenceevaluationcan
  * [ ] can detect overfitting（trainingR² ≫ testR²）
  * [ ] 5ofmodelofperformance comparison tablewithorganizationcan
  * [ ] predicted value vs measuredvalueofscatter plotcan create
  * [ ] residualplotcreate、modelofbias detectioncan

#### Advanced Level

  * [ ] data characteristicstoaccording tooptimalformodelselectioncan
  * strong linearity → linear regression
  * strong nonlinearity → random forest、LightGBM
  * few data points → SVR
  * [ ] eachmodelofhyperparametersofrolesplitunderstand
  * random forest：n_estimators、max_depth
  * LightGBM：learning_rate、num_leaves
  * SVR：C、gamma、epsilon
  * MLP：hidden_layer_sizes、alpha、early_stopping

* * *

### 3.12.4 Feature Importance & Model Interpretation Skills（Feature Importance & Interpretability）

#### Foundation Level

  * [ ] random forestoffeature importanceacquire andvisualizationcan（`feature_importances_`）
  * [ ] LightGBMoffeature importanceacquire andvisualizationcan
  * [ ] feature importanceofresultsinterpretationcan（most influential featureiswhat）

#### Applied Level

  * [ ] SHAPlibraryinstallusecan
  * [ ] SHAP Explainercan create（`shap.Explainer`）
  * [ ] SHAP Summary Plotcan create and interpret
  * [ ] SHAP Dependence Plotcan create and interpret
  * [ ] SHAPvalueofpositive/negativepredictiontoexplain influencecan

#### Advanced Level

  * [ ] multipleofinterpretation methodmethodusemincan be done
  * feature importance：overall importance
  * SHAP：unitsclassificationsamplesofpredictionreason
  * [ ] modelofpredictionbasis for stakeholderstoexplanationcan

* * *

### 3.12.5 Bayesian Optimization Skills（Bayesian Optimization）

#### Foundation Level

  * [ ] scikit-optimizeinstallcan
  * [ ] search space definitioncan（`Real(min, max, name)`）
  * [ ] objective functiondefinitioncan（parametersreceive、errorreturn）
  * [ ] `gp_minimize`implementrowscan
  * [ ] optimizationresults（result.x、result.fun）acquisitioncan

#### Applied Level

  * [ ] n_callsandn_initial_pointsofrolesplitunderstand
  * n_calls：evaluationnumber of times
  * n_initial_points：random sampling count
  * [ ] convergence plotcan create（`plot_convergence`）
  * [ ] optimizationprocessofvisualizationcan be done（evaluationhistory、bestvalueofprogression）
  * [ ] GoalvaluetoofAchievedaccuracyevaluationcan

#### Advanced Level

  * [ ] multipleofGoal（red, green, blueofquantum dots）toforoptimizationimplementrowscan
  * [ ] optimizationresultsexperimental validation plantoutilizationcan
  * [ ] acquisition function（Acquisition Function）ofunderstand the concept

* * *

### 3.12.6 multi-objective optimizationskills（Multi-Objective Optimization）

#### Foundation Level

  * [ ] pymoolibraryinstallcan
  * [ ] multi-objective optimizationproblemofunderstand the concept（sizeminimization vs efficiency maximization）
  * [ ] Pareto frontofconcept explanationcan

#### Applied Level

  * [ ] pymoo.core.probleminheritProblemclass definitioncan
  * [ ] NSGA-IIalgorithmcan implement
  * [ ] Pareto frontvisualizationcan
  * [ ] trade-offrelationshipinterpretationcan

#### Advanced Level

  * [ ] multipleoffrom solution to applicationtoaccording tooptimalsolution selectioncan
  * performance-oriented
  * environment-oriented
  * balanced type
  * [ ] grid searchbyalternative implementationcan be done（pymoowhen unavailable）

* * *

### 3.12.7 nanomaterial-specificofsolutionanalysisskills（Nanomaterial-Specific Analysis）

#### TEMimage analysis

  * [ ] log-normalminfollows distributionsizedatagenerationcan
  * [ ] log-normalmindistributionofparameters（sigma、mu）can calculate
  * [ ] `lognorm.fit`withfittingcan
  * [ ] fittingresultsvisualizationcan（histogram + PDFcurve）
  * [ ] Q-Qplotwithmindistributionofgoodness of fitevaluationcan

#### minmolecular dynamics（MD）data analysis

  * [ ] atomic trajectory dataofunderstand structure（n_steps × n_atoms × 3）
  * [ ] 3Dtrajectoryplotcan create
  * [ ] radial distribution function（RDF）can calculate
  * [ ] RDFofextract characteristic interatomic distances from peak positionscan
  * [ ] mean squared displacement（MSD）can calculate
  * [ ] MSDfromdiffusion coefficientcalculationcan（Einsteinrelationship equation）

#### anomaly detection

  * [ ] Isolation Forestcan implement
  * [ ] contamination（anomaly data ratio）settingscan
  * [ ] anomaly scorecan calculate（`score_samples`）
  * [ ] confusionrowscolumnwithanomalydetectionaccuracyevaluationcan
  * [ ] normaldataandanomalydataofmindistributionvisualizationcan

* * *

### 3.12.8 code qualityskills（Code Quality）

#### Foundation Level

  * [ ] allofcodetorandom seed（`random_state=42`）set
  * [ ] data validation（shape、dtype、missingvalue、range）implementimplemented
  * [ ] variable namesmineasy to understand（`X_train`、`y_test`、`model_lgb`）
  * [ ] commentswithprocessingofexplain purpose
  * [ ] graphtotitle、axis labels、legendExampleadded

#### Applied Level

  * [ ] functionalized for code reuseto| `python def calculate_rdf(positions, r_max, n_bins): ...`
  * [ ] documentation string（Docstring）described
  * [ ] graphofaesthetics arranged（fontsize、grid、alpha）
  * [ ] try-exceptwitherror handlingimplement and|（pymooofImportErrorsupport）

* * *

### 3.12.9 troubleshootingskills（Troubleshooting）

#### Foundation Level（error handling）

  * [ ] `ModuleNotFoundError`can solve（`pip install`）
  * [ ] `ValueError: Input contains NaN`can solve（missing value handling）
  * [ ] `ConvergenceWarning`（MLPofconvergence error）can solve
  * `max_iter`increase
  * standardize data
  * Early Stoppingenable
  * [ ] read error message、search for solutioncan be done

#### Applied Level（performance improvement）

  * [ ] R² < 0.7ofcase、3or moreofimprovement measuresimplementrowscan
  * feature engineering
  * modelchange（linear→nonlinear）
  * hyperparametersadjustment
  * [ ] can detect overfitting（trainingR² ≫ testR²）
  * [ ] detect underfittingcan（trainingR²testR²also low）

* * *

### 3.12.10 overallevaluation：proficiencyleveljudgment

or lessofleveljudgmentwith、selfminofplease check achievement level。

#### level1：beginner（Beginner）

  * Environment Setup Skills：Foundation Level 100%Achieved
  * Data Processing & Visualization Skills：Foundation Level 80%or more achieved
  * Machine Learning Model Implementation Skills：Foundation Level 5middle3or moreimplementation
  * troubleshooting：Foundation Leveloferrorselfcapabilitywithsolution

**Learning Goal:** nanoparticlesdatagenerationvisualization、linear regressionandrandom forestwithLSPRwavelengthpredictioncan implement

* * *

#### level2：middleintermediate（Intermediate）

  * Environment Setup Skills：Applied Level 80%or more achieved
  * Data Processing & Visualization Skills：Foundation Level 100%Achieved + Applied Level 70%or more
  * Machine Learning Model Implementation Skills：Foundation Level 100%Achieved + Applied Level 70%or more
  * Feature Importance & Model Interpretation Skills：Foundation Level 100%Achieved + Applied Level 50%or more
  * Bayesian Optimization Skills：Foundation Level 100%Achieved + Applied Level 50%or more

**Learning Goal:** 5oftimesregressionmodelcomparison、Bayesian optimizationwithGoalLSPRwavelength（550 nm）Achievedfind synthesis conditionscan

* * *

#### level3：advanced（Advanced）

  * all categories：Applied Level 100%Achieved
  * Feature Importance & Model Interpretation Skills：Advanced Level 80%or more
  * Bayesian Optimization Skills：Advanced Level 80%or more
  * multi-objective optimizationskills：Applied Level 100%Achieved
  * nanomaterial-specificofsolutionanalysisskills：TEM、MD、anomaly detectionimplement all

**Learning Goal:** SHAPminanalysiswithmodel interpretation、multi-objective optimizationwithsizeandefficiencyoftrade-offvisualizationcan

* * *

#### level4：expert（Expert）

  * all categories：Advanced Level 80%or more achieved
  * code quality：Applied Level 100%Achieved
  * originalselfofnanomaterial data（experimental dataisliterature data）toapplicationcan
  * custom machine learning pipelineconstructioncan
  * present research at conferences and publish paperscan

**Learning Goal:** \- actual nanomaterial data（TEM、UV-Vis、XRD）integrated solutionanalysis \- machine learningtonovel nanoparticlesofphysical properties90%or moreofaccuracywithprediction \- Bayesian optimizationwithexperimentnumber of timesconventionalof1/5toreduction

* * *

### 3.12.11 practical project check：Exerciseproblemofcompletion

#### Exercise1Completion check（CNTelectrical conductivityprediction）

  * [ ] data generation（150samples、3features）implement
  * [ ] LightGBMmodelwithpredictionimplement
  * [ ] R² > 0.8、RMSE < 0.5Achieved
  * [ ] feature importancevisualization
  * [ ] resultsinterpretation（etcofwhich features are most influential）

#### Exercise2Completion check（optimal silver nanoparticle synthesis conditions）

  * [ ] silver nanoparticlesdata（100samples）generation
  * [ ] LightGBMmodelconstruction
  * [ ] Bayesian optimizationimplementrows（40timesevaluation）
  * [ ] Target size（10 nm）andoferror < 1 nmAchieved
  * [ ] optimaltemperaturepHidentification

#### Exercise3Completion check（quantum dotsmulti-color emission design）

  * [ ] red, green, blueof3colortooptimizationimplementrows
  * [ ] each colorofoptimalsizeidentify synthesis conditions
  * [ ] predictionwavelengthGoalwavelength±10 nmwithintois within
  * [ ] resultsvisualization（size vs wavelengthofplot）

* * *

### 3.12.12 nextofto stepofreadiness check

#### real-world application（Chapter 4）toofpreparation

  * [ ] machine learningofbasic workflow（data preparation→modeltraining→evaluation→optimization）understand
  * [ ] nanomaterial-specificofdata（sizemindistribution、optical properties、electrical properties）can handle
  * [ ] optimizationmethodmethod（Bayesian optimization、multi-objective optimization）can implement
  * [ ] model interpretation（SHAP）ofunderstand importance

#### deep learning and graphsneural networktoofpreparation

  * [ ] neural network（MLP）implement and、activation functionoptimizationalgorithmunderstand
  * [ ] learning curvevisualization、can detect overfitting
  * [ ] Early Stoppingofunderstand the concept

#### to practical researchofpreparation

  * [ ] Jupyter Notebook|isPythonscriptwithcode managementcan
  * [ ] requirements.txtwithreproducible environmentto|
  * [ ] predictionresultsgraphing、reporttoSummarycan be
  * [ ] codetodocumentation described

* * *

**use checklistofhint:** 1\. **regulartoreview** : after learning、1weeks later、1months latertorecheck 2\. **not yetAchievedprioritize items** : checkwithcollect incomplete itemsmiddlelearning 3\. **levelrecord judgment** : growthvisualizationmaintain motivation 4\. **actual projectwithofutilization** : before research/development project starttorequiredskillsconfirmation

* * *

## References

  1. **Pedregosa, F. et al.** (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_ , 12, 2825-2830.

  2. **Ke, G. et al.** (2017). LightGBM: A highly efficient gradient boosting decision tree. _Advances in Neural Information Processing Systems_ , 30, 3146-3154.

  3. **Lundberg, S. M. & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems_ , 30, 4765-4774.

  4. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). Practical Bayesian optimization of machine learning algorithms. _Advances in Neural Information Processing Systems_ , 25, 2951-2959.

  5. **Deb, K. et al.** (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. _IEEE Transactions on Evolutionary Computation_ , 6(2), 182-197. [DOI: 10.1109/4235.996017](<https://doi.org/10.1109/4235.996017>)

  6. **Frenkel, D. & Smit, B.** (2001). _Understanding Molecular Simulation: From Algorithms to Applications_ (2nd ed.). Academic Press.

* * *

[← previous chapter：nanomaterialsoffundamental principles](<chapter2-fundamentals.html>) | [next chapter：real worldofapplicationandcareer →](<index.html>)
