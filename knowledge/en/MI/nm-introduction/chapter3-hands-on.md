---
title: "Chapter 3: Python Hands-On Tutorial"
chapter_title: "Chapter 3: Python Hands-On Tutorial"
subtitle: Nanomaterials Data Analysis and Machine Learning
reading_time: 30-40 min
difficulty: Beginner
code_examples: 0
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Python Hands-On Tutorial

Build the muscle to explore synthesis conditions efficiently, using regression models and Bayesian optimization that work even with small datasets. We go all the way from concise visualization of MD data to model interpretation with SHAP.

**💡 Note:** The goal is to find good conditions with few trials. Bayesian optimization guides you to promising spots, much like a metal detector.

Nanomaterials Data Analysis and Machine Learning

* * *

## Learning Objectives of This Chapter

By working through this chapter, you will acquire the following skills:

✅ Hands-on generation, visualization, and preprocessing of nanoparticle data ✅ Predicting nanomaterial properties with five types of regression models ✅ Optimal design of nanomaterials using Bayesian optimization ✅ Interpreting machine learning models with SHAP analysis ✅ Trade-off analysis with multi-objective optimization ✅ TEM image analysis and size distribution fitting ✅ Applying anomaly detection to quality control

* * *

## 3.1 Environment Setup

### Required Libraries

The main Python libraries used in this tutorial:
    
    
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
    

### Installation

#### Option 1: Anaconda Environment
    
    
    # Create a new environment with Anaconda
    conda create -n nanomaterials python=3.10 -y
    conda activate nanomaterials
    
    # Install the required libraries
    conda install pandas numpy matplotlib seaborn scipy scikit-learn -y
    conda install -c conda-forge lightgbm scikit-optimize shap -y
    
    # For multi-objective optimization (optional)
    pip install pymoo
    

#### Option 2: venv + pip Environment
    
    
    # Create a virtual environment
    python -m venv nanomaterials_env
    
    # Activate the virtual environment
    # macOS/Linux:
    source nanomaterials_env/bin/activate
    # Windows:
    nanomaterials_env\Scripts\activate
    
    # Install the required libraries
    pip install pandas numpy matplotlib seaborn scipy
    pip install scikit-learn lightgbm scikit-optimize shap pymoo
    

#### Option 3: Google Colab

If you use Google Colab, run the following code in a cell:
    
    
    # Install additional packages
    !pip install lightgbm scikit-optimize shap pymoo
    
    # Check the imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("Environment setup complete!")
    

* * *

## 3.2 Preparing and Visualizing Nanoparticle Data

### [Example 1] Synthetic Data Generation: Gold Nanoparticle Size and Optical Properties

The localized surface plasmon resonance (LSPR) wavelength of gold nanoparticles depends on the particle size. We represent this relationship with simulated data.
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Font settings (adjust as needed)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set the random seed (for reproducibility)
    np.random.seed(42)
    
    # Number of samples
    n_samples = 200
    
    # Gold nanoparticle size (nm): mean 15 nm, standard deviation 5 nm
    size = np.random.normal(15, 5, n_samples)
    size = np.clip(size, 5, 50)  # Restrict to the 5-50 nm range
    
    # LSPR wavelength (nm): simple approximation of Mie theory
    # Base wavelength 520 nm + size-dependent term + noise
    lspr = 520 + 0.8 * (size - 15) + np.random.normal(0, 5, n_samples)
    
    # Synthesis conditions
    temperature = np.random.uniform(20, 80, n_samples)  # Temperature (°C)
    pH = np.random.uniform(4, 10, n_samples)  # pH
    
    # Create the DataFrame
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
    

### [Example 2] Histogram of the Size Distribution
    
    
    # Histogram of the size distribution
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
    
    print(f"Mean size: {data['size_nm'].mean():.2f} nm")
    print(f"Standard deviation: {data['size_nm'].std():.2f} nm")
    print(f"Median: {data['size_nm'].median():.2f} nm")
    

### [Example 3] Scatter Plot Matrix
    
    
    # Pairplot (scatter plot matrix)
    sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6},
                 height=2.5, corner=False)
    plt.suptitle('Pairplot of Gold Nanoparticle Data', y=1.01, fontsize=14, fontweight='bold')
    plt.show()
    
    print("Visualized the relationships among all variables")
    

### [Example 4] Correlation Matrix Heatmap
    
    
    # Compute the correlation matrix
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
    
    print("Visualized the multidimensional relationships with a 3D plot")
    

* * *

## 3.3 Preprocessing and Data Splitting

### [Example 6] Handling Missing Values
    
    
    # Artificially introduce missing values (for practice)
    data_with_missing = data.copy()
    np.random.seed(123)
    
    # Randomly introduce 5% missing values
    missing_indices = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
    data_with_missing.loc[missing_indices, 'temperature_C'] = np.nan
    
    print("=" * 60)
    print("Checking for missing values")
    print("=" * 60)
    print(f"Number of missing values:\n{data_with_missing.isnull().sum()}")
    
    # Missing value handling method 1: impute with the mean
    data_filled_mean = data_with_missing.fillna(data_with_missing.mean())
    
    # Missing value handling method 2: impute with the median
    data_filled_median = data_with_missing.fillna(data_with_missing.median())
    
    # Missing value handling method 3: drop
    data_dropped = data_with_missing.dropna()
    
    print(f"\nOriginal data: {len(data_with_missing)} rows")
    print(f"After dropping missing values: {len(data_dropped)} rows")
    print(f"After mean imputation: {len(data_filled_mean)} rows (no missing values)")
    
    # Use the original data (no missing values) for the rest of the analysis
    data_clean = data.copy()
    print("\n→ From here on we use the data without missing values")
    

### [Example 7] Outlier Detection (IQR Method)
    
    
    # Outlier detection using the IQR (interquartile range) method
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    # Detect outliers in the size
    outliers, lower, upper = detect_outliers_iqr(data_clean['size_nm'])
    
    print("=" * 60)
    print("Outlier detection (IQR method)")
    print("=" * 60)
    print(f"Number of detected outliers: {outliers.sum()}")
    print(f"Lower bound: {lower:.2f} nm, upper bound: {upper:.2f} nm")
    
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
    
    print("→ We keep the outliers and use all of the data")
    

### [Example 8] Feature Scaling (StandardScaler)
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Separate features and target
    X = data_clean[['size_nm', 'temperature_C', 'pH']]
    y = data_clean['lspr_nm']
    
    # StandardScaler (standardize to mean 0, standard deviation 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compare before and after scaling
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("=" * 60)
    print("Statistics before scaling")
    print("=" * 60)
    print(X.describe())
    
    print("\n" + "=" * 60)
    print("Statistics after scaling (mean ≈ 0, standard deviation ≈ 1)")
    print("=" * 60)
    print(X_scaled_df.describe())
    
    print("\n→ Scaling has unified the scale of every feature")
    

### [Example 9] Splitting into Training and Test Data
    
    
    from sklearn.model_selection import train_test_split
    
    # Split into training and test data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("=" * 60)
    print("Data split")
    print("=" * 60)
    print(f"Total samples: {len(X)}")
    print(f"Training data: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test data: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\nTraining data statistics:")
    print(pd.DataFrame(X_train, columns=X.columns).describe())
    

* * *

## 3.4 Predicting Nanoparticle Properties with Regression Models

Goal: predict the LSPR wavelength from size, temperature, and pH

### [Example 10] Linear Regression
    
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Build the linear regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Predict
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
    print("\nRegression coefficients:")
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
    
    # Predict
    y_train_pred_rf = model_rf.predict(X_train)
    y_test_pred_rf = model_rf.predict(X_test)
    
    # Evaluate
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
    
    print("\nFeature importance:")
    print(feature_importance)
    
    # Visualize the feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'],
            color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    

### [Example 12] Gradient Boosting (LightGBM)
    
    
    import lightgbm as lgb
    
    # Build the LightGBM model
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
    
    # Predict
    y_train_pred_lgb = model_lgb.predict(X_train)
    y_test_pred_lgb = model_lgb.predict(X_test)
    
    # Evaluate
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
    
    # Predicted vs actual plot
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
    
    # Predict
    y_train_pred_svr = model_svr.predict(X_train)
    y_test_pred_svr = model_svr.predict(X_test)
    
    # Evaluate
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
    
    # Predict
    y_train_pred_mlp = model_mlp.predict(X_train)
    y_test_pred_mlp = model_mlp.predict(X_test)
    
    # Evaluate
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
    
    
    # Summarize the performance of all models
    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'LightGBM', 'SVR', 'MLP'],
        'R² (Train)': [r2_train_lr, r2_train_rf, r2_train_lgb, r2_train_svr, r2_train_mlp],
        'R² (Test)': [r2_test_lr, r2_test_rf, r2_test_lgb, r2_test_svr, r2_test_mlp],
        'RMSE (Test)': [rmse_test_lr, rmse_test_rf, rmse_test_lgb, rmse_test_svr, rmse_test_mlp],
        'MAE (Test)': [mae_test_lr, mae_test_rf, mae_test_lgb, mae_test_svr, mae_test_mlp]
    })
    
    results['Overfit'] = results['R² (Train)'] - results['R² (Test)']
    
    print("=" * 80)
    print("Performance comparison of all models")
    print("=" * 80)
    print(results.to_string(index=False))
    
    # Identify the best model
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
    
    # RMSE comparison
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

## 3.5 Predicting Quantum Dot Emission Wavelength

### [Example 16] Data Generation: CdSe Quantum Dots

The emission wavelength of CdSe quantum dots depends on their size, following the Brus equation.
    
    
    # Generate CdSe quantum dot data
    np.random.seed(100)
    
    n_qd_samples = 150
    
    # Quantum dot size (2-10 nm)
    size_qd = np.random.uniform(2, 10, n_qd_samples)
    
    # Simple approximation of the Brus equation: emission = 520 + 130/(size^0.8) + noise
    emission = 520 + 130 / (size_qd ** 0.8) + np.random.normal(0, 10, n_qd_samples)
    
    # Synthesis conditions
    synthesis_time = np.random.uniform(10, 120, n_qd_samples)  # minutes
    precursor_ratio = np.random.uniform(0.5, 2.0, n_qd_samples)  # molar ratio
    
    # Create the DataFrame
    data_qd = pd.DataFrame({
        'size_nm': size_qd,
        'emission_nm': emission,
        'synthesis_time_min': synthesis_time,
        'precursor_ratio': precursor_ratio
    })
    
    print("=" * 60)
    print("CdSe quantum dot data generation complete")
    print("=" * 60)
    print(data_qd.head(10))
    print("\nBasic statistics:")
    print(data_qd.describe())
    
    # Plot the relationship between size and emission wavelength
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
    
    # Scaling
    scaler_qd = StandardScaler()
    X_qd_scaled = scaler_qd.fit_transform(X_qd)
    
    # Train/test split
    X_qd_train, X_qd_test, y_qd_train, y_qd_test = train_test_split(
        X_qd_scaled, y_qd, test_size=0.2, random_state=42
    )
    
    # LightGBM model
    model_qd = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    
    model_qd.fit(X_qd_train, y_qd_train)
    
    # Predict
    y_qd_train_pred = model_qd.predict(X_qd_train)
    y_qd_test_pred = model_qd.predict(X_qd_test)
    
    # Evaluate
    r2_qd_train = r2_score(y_qd_train, y_qd_train_pred)
    r2_qd_test = r2_score(y_qd_test, y_qd_test_pred)
    rmse_qd = np.sqrt(mean_squared_error(y_qd_test, y_qd_test_pred))
    mae_qd = mean_absolute_error(y_qd_test, y_qd_test_pred)
    
    print("=" * 60)
    print("Quantum dot emission wavelength prediction model (LightGBM)")
    print("=" * 60)
    print(f"Training R²: {r2_qd_train:.4f}")
    print(f"Test R²: {r2_qd_test:.4f}")
    print(f"Test RMSE: {rmse_qd:.4f} nm")
    print(f"Test MAE: {mae_qd:.4f} nm")
    

### [Example 18] Visualizing the Prediction Results
    
    
    # Predicted vs actual plot (with confidence band)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot the test data
    axes[0].scatter(y_qd_test, y_qd_test_pred, alpha=0.6, s=80,
                    edgecolors='k', label='Test Data')
    axes[0].plot([y_qd_test.min(), y_qd_test.max()],
                 [y_qd_test.min(), y_qd_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    
    # Show the ±10 nm range
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
    
    # Prediction accuracy by size
    size_bins = [2, 4, 6, 8, 10]
    size_labels = ['2-4 nm', '4-6 nm', '6-8 nm', '8-10 nm']
    data_qd_test = pd.DataFrame({
        'size': X_qd.iloc[y_qd_test.index]['size_nm'].values,
        'actual': y_qd_test.values,
        'predicted': y_qd_test_pred
    })
    data_qd_test['size_bin'] = pd.cut(data_qd_test['size'], bins=size_bins, labels=size_labels)
    data_qd_test['error'] = np.abs(data_qd_test['actual'] - data_qd_test['predicted'])
    
    # Mean error per size bin
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
    
    print(f"\nOverall mean absolute error: {mae_qd:.2f} nm")
    print("Mean absolute error by size:")
    print(error_by_size)
    

* * *

## 3.6 Feature Importance Analysis

### [Example 19] Feature Importance (LightGBM)
    
    
    # Feature importance of the LightGBM model (gain-based)
    importance_gain = model_lgb.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_gain
    }).sort_values('Importance', ascending=False)
    
    print("=" * 60)
    print("Feature importance (LightGBM)")
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
    
    print(f"\nMost important feature: {importance_df.iloc[0]['Feature']}")
    

### [Example 20] SHAP Analysis: Interpreting Predictions
    
    
    import shap
    
    # Create the SHAP explainer
    explainer = shap.Explainer(model_lgb, X_train)
    shap_values = explainer(X_test)
    
    print("=" * 60)
    print("SHAP analysis")
    print("=" * 60)
    print("SHAP value computation complete")
    print(f"Shape of the SHAP values: {shap_values.values.shape}")
    
    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title('SHAP Summary Plot: Feature Impact on LSPR Prediction',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # SHAP Dependence Plot (most important feature)
    top_feature_idx = importance_df.index[0]
    top_feature_name = X.columns[top_feature_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values.values, X_test,
                         feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature_name}',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nSHAP analysis confirmed that {top_feature_name} has the largest influence on the LSPR wavelength prediction")
    

* * *

## 3.7 Nanomaterial Design with Bayesian Optimization

Goal: find the optimal synthesis conditions that achieve a target LSPR wavelength (550 nm)

### [Example 21] Defining the Search Space
    
    
    from skopt.space import Real
    
    # Define the search space
    # Size: 10-40 nm, temperature: 20-80°C, pH: 4-10
    search_space = [
        Real(10, 40, name='size_nm'),
        Real(20, 80, name='temperature_C'),
        Real(4, 10, name='pH')
    ]
    
    print("=" * 60)
    print("Bayesian optimization: defining the search space")
    print("=" * 60)
    for dim in search_space:
        print(f"  {dim.name}: [{dim.bounds[0]}, {dim.bounds[1]}]")
    
    print("\nGoal: find conditions that achieve an LSPR wavelength of 550 nm")
    

### [Example 22] Setting Up the Objective Function
    
    
    # Objective function: minimize the absolute difference between the predicted LSPR wavelength and the target (550 nm)
    target_lspr = 550.0
    
    def objective_function(params):
        """
        Objective function for Bayesian optimization
    
        Parameters:
        -----------
        params : list
            [size_nm, temperature_C, pH]
    
        Returns:
        --------
        float
            Error relative to the target wavelength (value to minimize)
        """
        # Unpack the parameters
        size, temp, ph = params
    
        # Build the features (apply scaling)
        features = np.array([[size, temp, ph]])
        features_scaled = scaler.transform(features)
    
        # Predict the LSPR wavelength
        predicted_lspr = model_lgb.predict(features_scaled)[0]
    
        # Error relative to the target wavelength (absolute value)
        error = abs(predicted_lspr - target_lspr)
    
        return error
    
    # Test run
    test_params = [20.0, 50.0, 7.0]
    test_error = objective_function(test_params)
    print(f"\nTest run:")
    print(f"  Parameters: size={test_params[0]} nm, temp={test_params[1]}°C, pH={test_params[2]}")
    print(f"  Objective value (error): {test_error:.4f} nm")
    

### [Example 23] Running Bayesian Optimization (scikit-optimize)
    
    
    from skopt import gp_minimize
    from skopt.plots import plot_convergence, plot_objective
    
    # Run the Bayesian optimization
    print("\n" + "=" * 60)
    print("Running Bayesian optimization...")
    print("=" * 60)
    
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=50,  # Number of evaluations
        n_initial_points=10,  # Number of random samples
        random_state=42,
        verbose=False
    )
    
    print("Optimization complete!")
    print("\n" + "=" * 60)
    print("Optimization results")
    print("=" * 60)
    print(f"Minimum objective value (error): {result.fun:.4f} nm")
    print(f"\nOptimal parameters:")
    print(f"  Size: {result.x[0]:.2f} nm")
    print(f"  Temperature: {result.x[1]:.2f} °C")
    print(f"  pH: {result.x[2]:.2f}")
    
    # Compute the predicted LSPR wavelength at the optimal conditions
    optimal_features = np.array([result.x])
    optimal_features_scaled = scaler.transform(optimal_features)
    predicted_optimal_lspr = model_lgb.predict(optimal_features_scaled)[0]
    
    print(f"\nPredicted LSPR wavelength: {predicted_optimal_lspr:.2f} nm")
    print(f"Target LSPR wavelength: {target_lspr} nm")
    print(f"Achieved accuracy: {abs(predicted_optimal_lspr - target_lspr):.2f} nm")
    

### [Example 24] Visualizing the Optimization Results
    
    
    # Visualize the optimization process
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    plot_convergence(result, ax=axes[0])
    axes[0].set_title('Convergence Plot', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Objective Value (Error, nm)', fontsize=11)
    axes[0].set_xlabel('Number of Evaluations', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot the evaluation history
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
    
    
    # Detailed convergence plot (evolution of the best value)
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
    
    print(f"\nConverged to the optimal solution in {len(result.func_vals)} evaluations")
    print(f"Best error at the initial evaluation: {result.func_vals[0]:.2f} nm")
    print(f"Final best error: {result.fun:.2f} nm")
    print(f"Improvement rate: {(1 - result.fun/result.func_vals[0])*100:.1f}%")
    

* * *

## 3.8 Multi-Objective Optimization: Trade-off Between Size and Emission Efficiency

### [Example 26] Pareto Optimization (NSGA-II)

Multi-objective optimization optimizes several objectives simultaneously. Here we minimize quantum dot size while maximizing emission efficiency (a hypothetical metric).
    
    
    # Multi-objective optimization with pymoo
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
    
        # Define the multi-objective optimization problem
        class QuantumDotOptimization(Problem):
            def __init__(self):
                super().__init__(
                    n_var=3,  # Number of variables (size, synthesis_time, precursor_ratio)
                    n_obj=2,  # Number of objectives (minimize size, maximize emission efficiency)
                    n_constr=0,  # No constraints
                    xl=np.array([2.0, 10.0, 0.5]),  # Lower bounds
                    xu=np.array([10.0, 120.0, 2.0])  # Upper bounds
                )
    
            def _evaluate(self, X, out, *args, **kwargs):
                # Objective 1: minimize size
                obj1 = X[:, 0]  # size
    
                # Objective 2: maximize emission efficiency (converted to minimization via negation)
                # We assume efficiency is higher when the emission wavelength is closer to 550 nm
                features = X  # [size, synthesis_time, precursor_ratio]
                features_scaled = scaler_qd.transform(features)
                predicted_emission = model_qd.predict(features_scaled)
    
                # Efficiency: higher when the deviation from 550 nm is smaller (negated for maximization → minimization)
                efficiency = -np.abs(predicted_emission - 550)
                obj2 = -efficiency  # Convert maximization into a minimization problem
    
                out["F"] = np.column_stack([obj1, obj2])
    
        # Instantiate the problem
        problem = QuantumDotOptimization()
    
        # NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=40,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
    
        # Run the optimization
        print("=" * 60)
        print("Running multi-objective optimization (NSGA-II)...")
        print("=" * 60)
    
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', 50),  # Number of generations
            seed=42,
            verbose=False
        )
    
        print("Multi-objective optimization complete!")
        print(f"\nNumber of Pareto-optimal solutions: {len(res.F)}")
    
        # Show representative Pareto-optimal solutions (top 5)
        print("\nRepresentative Pareto-optimal solutions (top 5):")
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
        print("pymoo is not installed")
        print("=" * 60)
        print("Multi-objective optimization requires pymoo:")
        print("  pip install pymoo")
        print("\nShowing a simplified multi-objective optimization example instead")
    
        # Simulate multi-objective optimization with a simple grid search
        sizes = np.linspace(2, 10, 20)
        times = np.linspace(10, 120, 20)
        ratios = np.linspace(0.5, 2.0, 20)
    
        # Grid search (sampling)
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
    
        print("\nGrid search over candidate solutions complete")
        print(f"Number of explored solutions: {len(sample_F)}")
    
        res = type('Result', (), {
            'X': sample_X,
            'F': sample_F
        })()
    
        PYMOO_AVAILABLE = False
    

### [Example 27] Visualizing the Pareto Front
    
    
    # Visualize the Pareto front
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if PYMOO_AVAILABLE:
        # Plot the NSGA-II results
        ax.scatter(res.F[:, 0], -res.F[:, 1], c='blue', s=80, alpha=0.6,
                   edgecolors='black', label='Pareto Optimal Solutions')
    
        title_suffix = "(NSGA-II)"
    else:
        # Plot the grid search results
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
    print("  Shrinking the size lowers the efficiency, and raising the efficiency enlarges the size")
    print("  → The trade-off relationship is now clear")
    

* * *

## 3.9 TEM Image Analysis and Size Distribution

### [Example 28] Generating Simulated TEM Data

Nanoparticle sizes measured by TEM (transmission electron microscopy) often follow a lognormal distribution.
    
    
    from scipy.stats import lognorm
    
    # Generate TEM size data following a lognormal distribution
    np.random.seed(200)
    
    # Parameters
    mean_size = 20  # Mean size (nm)
    cv = 0.3  # Coefficient of variation (std/mean)
    
    # Compute the lognormal distribution parameters
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_size) - 0.5 * sigma**2
    
    # Generate samples (500 particles)
    tem_sizes = lognorm.rvs(s=sigma, scale=np.exp(mu), size=500)
    
    print("=" * 60)
    print("TEM measurement data generation (lognormal distribution)")
    print("=" * 60)
    print(f"Number of samples: {len(tem_sizes)} particles")
    print(f"Mean size: {tem_sizes.mean():.2f} nm")
    print(f"Standard deviation: {tem_sizes.std():.2f} nm")
    print(f"Median: {np.median(tem_sizes):.2f} nm")
    print(f"Minimum: {tem_sizes.min():.2f} nm")
    print(f"Maximum: {tem_sizes.max():.2f} nm")
    
    # Histogram
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
    

### [Example 29] Lognormal Distribution Fitting
    
    
    # Fit the lognormal distribution
    shape_fit, loc_fit, scale_fit = lognorm.fit(tem_sizes, floc=0)
    
    # Parameters of the fitted distribution
    fitted_mean = np.exp(np.log(scale_fit) + 0.5 * shape_fit**2)
    fitted_std = fitted_mean * np.sqrt(np.exp(shape_fit**2) - 1)
    
    print("=" * 60)
    print("Lognormal distribution fitting results")
    print("=" * 60)
    print(f"Shape parameter (sigma): {shape_fit:.4f}")
    print(f"Scale parameter: {scale_fit:.4f}")
    print(f"Fitted mean size: {fitted_mean:.2f} nm")
    print(f"Fitted standard deviation: {fitted_std:.2f} nm")
    
    # Comparison with the measured values
    print(f"\nComparison with the measured values:")
    print(f"  Mean size - measured: {tem_sizes.mean():.2f} nm, fitted: {fitted_mean:.2f} nm")
    print(f"  Standard deviation - measured: {tem_sizes.std():.2f} nm, fitted: {fitted_std:.2f} nm")
    

### [Example 30] Visualizing the Fitting Results
    
    
    # Detailed visualization of the fitting results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram with the fitted curve
    axes[0].hist(tem_sizes, bins=40, alpha=0.6, color='lightblue',
                 edgecolor='black', density=True, label='TEM Data')
    
    # Fitted lognormal distribution
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
    
    # Q-Q plot (quantile plot)
    from scipy.stats import probplot
    
    probplot(tem_sizes, dist=lognorm, sparams=(shape_fit, loc_fit, scale_fit),
             plot=axes[1])
    axes[1].set_title('Q-Q Plot: Lognormal Distribution',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nQ-Q plot: if the data lie on the straight line, they follow the lognormal distribution well")
    

* * *

## 3.10 Molecular Dynamics (MD) Data Analysis

### [Example 31] Loading MD Simulation Data

Molecular dynamics simulations track the time evolution of the atomic configuration of a nanoparticle.
    
    
    # Simulated generation of MD simulation data
    # Real MD data would come from LAMMPS, GROMACS, etc.
    
    np.random.seed(300)
    
    n_atoms = 100  # Number of atoms
    n_steps = 1000  # Number of time steps
    dt = 0.001  # Time step (ps)
    
    # Initial positions (nm)
    positions_initial = np.random.uniform(-1, 1, (n_atoms, 3))
    
    # Simulated time evolution (random walk)
    positions = np.zeros((n_steps, n_atoms, 3))
    positions[0] = positions_initial
    
    for t in range(1, n_steps):
        # Random displacement
        displacement = np.random.normal(0, 0.01, (n_atoms, 3))
        positions[t] = positions[t-1] + displacement
    
    print("=" * 60)
    print("MD simulation data generation")
    print("=" * 60)
    print(f"Number of atoms: {n_atoms}")
    print(f"Number of time steps: {n_steps}")
    print(f"Simulation time: {n_steps * dt:.2f} ps")
    print(f"Data shape: {positions.shape} (time, atoms, xyz)")
    
    # Plot the trajectory of the central atom (atom 0)
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
    

### [Example 32] Computing the Radial Distribution Function (RDF)

The radial distribution function (RDF) describes the distribution of interatomic distances.
    
    
    # Compute the radial distribution function (RDF)
    def calculate_rdf(positions, r_max=2.0, n_bins=100):
        """
        Compute the radial distribution function
    
        Parameters:
        -----------
        positions : ndarray
            Atom positions (n_atoms, 3)
        r_max : float
            Maximum distance (nm)
        n_bins : int
            Number of bins
    
        Returns:
        --------
        r_bins : ndarray
            Distance bins
        rdf : ndarray
            Radial distribution function
        """
        n_atoms = positions.shape[0]
    
        # Compute distances between all atom pairs
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < r_max:
                    distances.append(dist)
    
        distances = np.array(distances)
    
        # Histogram
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
        r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Normalization (ratio to an ideal gas)
        dr = r_max / n_bins
        volume_shell = 4 * np.pi * r_bins**2 * dr
        n_ideal = volume_shell * (n_atoms / (4/3 * np.pi * r_max**3))
    
        rdf = hist / n_ideal / (n_atoms / 2)
    
        return r_bins, rdf
    
    # Compute the RDF for the final frame
    final_positions = positions[-1]
    r_bins, rdf = calculate_rdf(final_positions, r_max=1.5, n_bins=150)
    
    print("=" * 60)
    print("Radial distribution function (RDF)")
    print("=" * 60)
    print(f"Computation complete: {len(r_bins)} bins")
    
    # RDF plot
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
    
    # Detect the peak positions
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(rdf, height=1.2, distance=10)
    print(f"\nRDF peak positions (characteristic interatomic distances):")
    for i, peak_idx in enumerate(peaks[:3], 1):
        print(f"  Peak {i}: r = {r_bins[peak_idx]:.3f} nm, g(r) = {rdf[peak_idx]:.2f}")
    

### [Example 33] Computing the Diffusion Coefficient (Mean Squared Displacement)
    
    
    # Compute the mean squared displacement (MSD)
    def calculate_msd(positions):
        """
        Compute the mean squared displacement
    
        Parameters:
        -----------
        positions : ndarray
            Atom positions (n_steps, n_atoms, 3)
    
        Returns:
        --------
        msd : ndarray
            Mean squared displacement (n_steps,)
        """
        n_steps, n_atoms, _ = positions.shape
        msd = np.zeros(n_steps)
    
        # MSD at each time step
        for t in range(n_steps):
            displacement = positions[t] - positions[0]
            squared_displacement = np.sum(displacement**2, axis=1)
            msd[t] = np.mean(squared_displacement)
    
        return msd
    
    # Compute the MSD
    msd = calculate_msd(positions)
    time = np.arange(n_steps) * dt
    
    print("=" * 60)
    print("Mean squared displacement (MSD) and diffusion coefficient")
    print("=" * 60)
    
    # Compute the diffusion coefficient (Einstein relation: MSD = 6*D*t)
    # Linear fit (using the latter 50% of the data)
    start_idx = n_steps // 2
    fit_coeffs = np.polyfit(time[start_idx:], msd[start_idx:], 1)
    slope = fit_coeffs[0]
    diffusion_coefficient = slope / 6
    
    print(f"Diffusion coefficient D = {diffusion_coefficient:.6f} nm²/ps")
    print(f"            = {diffusion_coefficient * 1e3:.6f} × 10⁻⁶ cm²/s")
    
    # MSD plot
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
    
    print("\nThe diffusion coefficient is a key quantitative measure of nanoparticle mobility")
    

* * *

## 3.11 Anomaly Detection: Application to Quality Control

### [Example 34] Detecting Anomalous Nanoparticles with Isolation Forest

We apply machine learning based anomaly detection to the quality control of nanoparticles produced in a manufacturing process.
    
    
    from sklearn.ensemble import IsolationForest
    
    # Mix normal and anomalous data
    np.random.seed(400)
    
    # Normal gold nanoparticle data (180 samples)
    normal_size = np.random.normal(15, 3, 180)
    normal_lspr = 520 + 0.8 * (normal_size - 15) + np.random.normal(0, 3, 180)
    
    # Anomalous nanoparticle data (20 samples): abnormally large or small sizes
    anomaly_size = np.concatenate([
        np.random.uniform(5, 8, 10),  # Abnormally small
        np.random.uniform(35, 50, 10)  # Abnormally large
    ])
    anomaly_lspr = 520 + 0.8 * (anomaly_size - 15) + np.random.normal(0, 8, 20)
    
    # Combine all data
    all_size = np.concatenate([normal_size, anomaly_size])
    all_lspr = np.concatenate([normal_lspr, anomaly_lspr])
    all_data = np.column_stack([all_size, all_lspr])
    
    # Labels (normal = 0, anomaly = 1)
    true_labels = np.concatenate([np.zeros(180), np.ones(20)])
    
    print("=" * 60)
    print("Anomaly detection (Isolation Forest)")
    print("=" * 60)
    print(f"Total samples: {len(all_data)}")
    print(f"Normal data: {int((true_labels == 0).sum())} samples")
    print(f"Anomalous data: {int((true_labels == 1).sum())} samples")
    
    # Isolation Forest model
    iso_forest = IsolationForest(
        contamination=0.1,  # Fraction of anomalous data (assumed 10%)
        random_state=42,
        n_estimators=100
    )
    
    # Anomaly detection
    predictions = iso_forest.fit_predict(all_data)
    anomaly_scores = iso_forest.score_samples(all_data)
    
    # Prediction results (1: normal, -1: anomaly)
    predicted_anomalies = (predictions == -1)
    true_anomalies = (true_labels == 1)
    
    # Evaluation metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Convert the predictions to 0/1
    predicted_labels = (predictions == -1).astype(int)
    
    print("\nConfusion matrix:")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    
    print("\nClassification report:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=['Normal', 'Anomaly']))
    
    # Detection rate
    detected_anomalies = np.sum(predicted_anomalies & true_anomalies)
    total_anomalies = np.sum(true_anomalies)
    detection_rate = detected_anomalies / total_anomalies * 100
    
    print(f"\nAnomaly detection rate: {detection_rate:.1f}% ({detected_anomalies}/{total_anomalies})")
    

### [Example 35] Visualizing the Anomalous Samples
    
    
    # Visualize the anomaly detection results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot (true labels)
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
    
    # Scatter plot (predictions)
    normal_mask = ~predicted_anomalies
    anomaly_mask = predicted_anomalies
    
    axes[1].scatter(all_size[normal_mask], all_lspr[normal_mask],
                    c='blue', s=60, alpha=0.6, label='Predicted Normal', edgecolors='k')
    axes[1].scatter(all_size[anomaly_mask], all_lspr[anomaly_mask],
                    c='orange', s=100, alpha=0.8, marker='X', label='Predicted Anomaly',
                    edgecolors='k', linewidths=2)
    
    # Highlight correctly detected anomalies
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
    
    # Distribution of the anomaly scores
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
    
    print("\nThe lower the anomaly score (the more negative the value), the more likely the sample is anomalous")
    

* * *

## Summary

In this chapter we learned practical techniques for nanomaterials data analysis and machine learning with Python through 35 code examples.

### Key Techniques Acquired

  1. **Data generation and visualization** (Examples 1-5) \- Synthetic data generation for gold nanoparticles and quantum dots \- Histograms, scatter plots, 3D plots, correlation analysis

  2. **Data preprocessing** (Examples 6-9) \- Missing value handling, outlier detection, scaling, data splitting

  3. **Property prediction with regression models** (Examples 10-15) \- Linear regression, random forest, LightGBM, SVR, MLP \- Model performance comparison (R², RMSE, MAE)

  4. **Quantum dot emission prediction** (Examples 16-18) \- Data generation based on the Brus equation \- Building a prediction model with LightGBM

  5. **Feature importance and model interpretation** (Examples 19-20) \- LightGBM feature importance \- Interpreting predictions with SHAP analysis

  6. **Bayesian optimization** (Examples 21-25) \- Searching for optimal synthesis conditions that achieve a target LSPR wavelength \- Convergence plots and visualization of the optimization process

  7. **Multi-objective optimization** (Examples 26-27) \- Pareto optimization with NSGA-II \- Trade-off analysis between size and emission efficiency

  8. **TEM image analysis** (Examples 28-30) \- Size distribution fitting with a lognormal distribution \- Verifying the distribution with Q-Q plots

  9. **Molecular dynamics data analysis** (Examples 31-33) \- Visualizing atomic trajectories \- Computing the radial distribution function (RDF) \- Deriving the diffusion coefficient (MSD method)

  10. **Anomaly detection** (Examples 34-35)

     * Quality control with Isolation Forest
     * Automatic detection of anomalous nanoparticles

### Practical Applications

These techniques can be applied directly to real nanomaterials research, for example:

  * **Materials design** : Highly efficient materials exploration via machine learning property prediction and optimization
  * **Process optimization** : Reducing the number of experiments and discovering optimal synthesis conditions with Bayesian optimization
  * **Quality control** : Early detection of defective products and improved yield through anomaly detection
  * **Data analysis** : Quantitative analysis of TEM data and MD simulation data
  * **Model interpretation** : Visualizing the basis of predictions and improving reliability with SHAP analysis

### Preview of the Next Chapter

In Chapter 4, we will study five detailed case studies applying these techniques to real nanomaterials research projects. Through practical applications of carbon nanotube composites, quantum dots, gold nanoparticle catalysts, graphene, and nanomedicine, you will gain a complete picture of the problem-solving process.

* * *

## Exercises

### Exercise 1: Predicting Carbon Nanotube Electrical Conductivity

The electrical conductivity of carbon nanotubes (CNTs) depends on their diameter, chirality, and length. Generate the following data and predict it with a LightGBM model.

**Data specification** : \- Number of samples: 150 \- Features: diameter (1-3 nm), length (100-1000 nm), chirality index (continuous value from 0 to 1) \- Target: electrical conductivity (10³-10⁷ S/m, lognormal distribution)

**Tasks** : 1\. Generate the data 2\. Split into training/test data 3\. Build and evaluate the LightGBM model 4\. Visualize the feature importance

Sample Solution
    
    
    # Generate the data
    np.random.seed(500)
    n_samples = 150
    
    diameter = np.random.uniform(1, 3, n_samples)
    length = np.random.uniform(100, 1000, n_samples)
    chirality = np.random.uniform(0, 1, n_samples)
    
    # Electrical conductivity (simple model: strongly dependent on diameter and chirality)
    log_conductivity = 3 + 2*diameter + 3*chirality + 0.001*length + np.random.normal(0, 0.5, n_samples)
    conductivity = 10 ** log_conductivity  # S/m
    
    data_cnt = pd.DataFrame({
        'diameter_nm': diameter,
        'length_nm': length,
        'chirality': chirality,
        'conductivity_Sm': conductivity
    })
    
    # Features and target
    X_cnt = data_cnt[['diameter_nm', 'length_nm', 'chirality']]
    y_cnt = np.log10(data_cnt['conductivity_Sm'])  # Log transform
    
    # Scaling
    scaler_cnt = StandardScaler()
    X_cnt_scaled = scaler_cnt.fit_transform(X_cnt)
    
    # Train/test split
    X_cnt_train, X_cnt_test, y_cnt_train, y_cnt_test = train_test_split(
        X_cnt_scaled, y_cnt, test_size=0.2, random_state=42
    )
    
    # LightGBM model
    model_cnt = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42, verbose=-1)
    model_cnt.fit(X_cnt_train, y_cnt_train)
    
    # Predict and evaluate
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
    
    print("\nFeature importance:")
    print(importance_cnt)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_cnt['Feature'], importance_cnt['Importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance: CNT Conductivity Prediction')
    plt.tight_layout()
    plt.show()
    

### Exercise 2: Finding Optimal Synthesis Conditions for Silver Nanoparticles

The antibacterial activity of silver nanoparticles increases as the size decreases. Using Bayesian optimization, find the optimal synthesis temperature and pH that achieve a target size of 10 nm.

**Conditions** : \- Temperature range: 20-80°C \- pH range: 6-11 \- Target size: 10 nm

Sample Solution
    
    
    # Generate the silver nanoparticle data
    np.random.seed(600)
    n_ag = 100
    
    temp_ag = np.random.uniform(20, 80, n_ag)
    pH_ag = np.random.uniform(6, 11, n_ag)
    
    # Size model (assume the size decreases with higher temperature and lower pH)
    size_ag = 15 - 0.1*temp_ag - 0.8*pH_ag + np.random.normal(0, 1, n_ag)
    size_ag = np.clip(size_ag, 5, 30)
    
    data_ag = pd.DataFrame({
        'temperature': temp_ag,
        'pH': pH_ag,
        'size': size_ag
    })
    
    # Build the model (LightGBM)
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
    print("Optimal synthesis conditions for silver nanoparticles")
    print("=" * 60)
    print(f"Minimum error: {result_ag.fun:.2f} nm")
    print(f"Optimal temperature: {result_ag.x[0]:.1f} °C")
    print(f"Optimal pH: {result_ag.x[1]:.2f}")
    
    # Predicted size at the optimal conditions
    optimal_features = scaler_ag.transform([result_ag.x])
    predicted_size = model_ag.predict(optimal_features)[0]
    print(f"Predicted size: {predicted_size:.2f} nm")
    

### Exercise 3: Multi-Color Emission Design of Quantum Dots

Design the sizes of CdSe quantum dots that achieve three emission colors — red (650 nm), green (550 nm), and blue (450 nm) — using Bayesian optimization.

**Hints** : \- Run the optimization for each color \- Use the relationship between emission wavelength and size

Sample Solution
    
    
    # Quantum dot data (use data_qd from Example 16)
    # Assume model_qd and scaler_qd have already been built
    
    # Three target wavelengths
    target_colors = {
        'Red': 650,
        'Green': 550,
        'Blue': 450
    }
    
    results_colors = {}
    
    for color_name, target_emission in target_colors.items():
        # Search space
        space_qd = [
            Real(2, 10, name='size_nm'),
            Real(10, 120, name='synthesis_time_min'),
            Real(0.5, 2.0, name='precursor_ratio')
        ]
    
        # Objective function
        def objective_qd(params):
            features = scaler_qd.transform([params])
            predicted_emission = model_qd.predict(features)[0]
            return abs(predicted_emission - target_emission)
    
        # Optimization
        result_qd_color = gp_minimize(objective_qd, space_qd, n_calls=30, random_state=42, verbose=False)
    
        # Store the results
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
    
    # Show the results
    print("=" * 80)
    print("Multi-color quantum dot emission design")
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

## 3.12 End-of-Chapter Checklist: Quality Assurance for Nanomaterials Data Analysis Skills

Systematically check the Python nanomaterials data analysis and machine learning implementation skills learned in this chapter.

### 3.12.1 Environment Setup Skills

#### Basic Level

  * [ ] Python 3.9 or later is installed
  * [ ] Can explain the differences between the three environment setup options (Anaconda/venv/Colab)
  * [ ] Can choose the environment best suited to your situation
  * [ ] Can create, activate, and deactivate a virtual environment
  * [ ] Can install libraries with pip/conda (pandas, numpy, matplotlib, scikit-learn, lightgbm)
  * [ ] Can run the environment verification code and confirm it works without errors

#### Applied Level

  * [ ] Can create and use requirements.txt
  * [ ] Can mount Google Drive in Google Colab and load data
  * [ ] Can maintain multiple virtual environments for different purposes
  * [ ] Can troubleshoot installation errors on your own
  * [ ] Can install optional libraries (pymoo, SHAP) as needed

* * *

### 3.12.2 Data Processing & Visualization Skills

#### Basic Level

  * [ ] Can generate synthetic data with NumPy (normal distribution, uniform distribution)
  * [ ] Can create and manipulate DataFrames with Pandas
  * [ ] Can compute basic statistics (mean, std, median)
  * [ ] Can create histograms
  * [ ] Can create scatter plots
  * [ ] Can detect missing values (`isnull().sum()`)
  * [ ] Can drop or impute missing values (`dropna()` or `fillna()`)

#### Applied Level

  * [ ] Can compute and visualize a correlation matrix (`corr()`, seaborn.heatmap)
  * [ ] Can create pairplots (scatter plot matrices) (seaborn.pairplot)
  * [ ] Can create 3D scatter plots (mpl_toolkits.mplot3d)
  * [ ] Can use KDE (kernel density estimation)
  * [ ] Can detect outliers with the IQR method
  * [ ] Can standardize data with StandardScaler
  * [ ] Can split data with train_test_split (80% vs 20%)
  * [ ] Ensure reproducibility with `random_state=42`

#### Advanced Level

  * [ ] Can fit a lognormal distribution (scipy.stats.lognorm)
  * [ ] Can verify the goodness of fit with Q-Q plots
  * [ ] Can use colormaps effectively (viridis, plasma, coolwarm)
  * [ ] Can build advanced visualizations with multiple subplots

* * *

### 3.12.3 ML Model Implementation Skills

#### Basic Level (Implementing the 5 Models)

  * [ ] Can implement linear regression and explain the meaning of the coefficients
  * [ ] Can implement random forest and explain the role of `n_estimators`
  * [ ] Can install and implement LightGBM
  * [ ] Understand why standardization (StandardScaler) is needed for SVR
  * [ ] Can implement MLPRegressor (neural network)

#### Applied Level (Model Selection and Evaluation)

  * [ ] Can compute and interpret MAE, R², and RMSE
  * [ ] Can evaluate the performance gap between training and test data
  * [ ] Can detect overfitting (training R² ≫ test R²)
  * [ ] Can organize the performance of the 5 models into a comparison table
  * [ ] Can create predicted vs actual scatter plots
  * [ ] Can create residual plots and detect model bias

#### Advanced Level

  * [ ] Can select the optimal model based on data characteristics
  * Strongly linear → linear regression
  * Strongly nonlinear → random forest, LightGBM
  * Small dataset → SVR
  * [ ] Understand the role of the hyperparameters of each model
  * Random forest: n_estimators, max_depth
  * LightGBM: learning_rate, num_leaves
  * SVR: C, gamma, epsilon
  * MLP: hidden_layer_sizes, alpha, early_stopping

* * *

### 3.12.4 Feature Importance & Interpretability Skills

#### Basic Level

  * [ ] Can obtain and visualize random forest feature importance (`feature_importances_`)
  * [ ] Can obtain and visualize LightGBM feature importance
  * [ ] Can interpret feature importance results (which feature has the largest influence)

#### Applied Level

  * [ ] Can install and use the SHAP library
  * [ ] Can create a SHAP explainer (`shap.Explainer`)
  * [ ] Can create and interpret SHAP summary plots
  * [ ] Can create and interpret SHAP dependence plots
  * [ ] Can explain how the sign of SHAP values affects the prediction

#### Advanced Level

  * [ ] Can choose between multiple interpretation methods
  * Feature importance: overall importance
  * SHAP: the reason behind an individual sample's prediction
  * [ ] Can explain the basis of model predictions to stakeholders

* * *

### 3.12.5 Bayesian Optimization Skills

#### Basic Level

  * [ ] Can install scikit-optimize
  * [ ] Can define a search space (`Real(min, max, name)`)
  * [ ] Can define an objective function (takes parameters, returns an error)
  * [ ] Can run `gp_minimize`
  * [ ] Can retrieve the optimization results (result.x, result.fun)

#### Applied Level

  * [ ] Understand the roles of n_calls and n_initial_points
  * n_calls: number of evaluations
  * n_initial_points: number of random samples
  * [ ] Can create convergence plots (`plot_convergence`)
  * [ ] Can visualize the optimization process (evaluation history, evolution of the best value)
  * [ ] Can evaluate the accuracy of reaching the target value

#### Advanced Level

  * [ ] Can run optimizations for multiple targets (red, green, and blue quantum dots)
  * [ ] Can use optimization results in experimental validation plans
  * [ ] Understand the concept of acquisition functions

* * *

### 3.12.6 Multi-Objective Optimization Skills

#### Basic Level

  * [ ] Can install the pymoo library
  * [ ] Understand the concept of multi-objective optimization problems (minimize size vs maximize efficiency)
  * [ ] Can explain the concept of a Pareto front

#### Applied Level

  * [ ] Can define a Problem class by inheriting pymoo.core.problem
  * [ ] Can implement the NSGA-II algorithm
  * [ ] Can visualize the Pareto front
  * [ ] Can interpret trade-off relationships

#### Advanced Level

  * [ ] Can select the best solution for a given use case from multiple solutions
  * Performance-oriented
  * Environment-oriented
  * Balanced
  * [ ] Can implement a grid search alternative (when pymoo is unavailable)

* * *

### 3.12.7 Nanomaterial-Specific Analysis Skills

#### TEM Image Analysis

  * [ ] Can generate size data following a lognormal distribution
  * [ ] Can compute the lognormal distribution parameters (sigma, mu)
  * [ ] Can fit with `lognorm.fit`
  * [ ] Can visualize the fitting results (histogram + PDF curve)
  * [ ] Can evaluate the goodness of fit with Q-Q plots

#### Molecular Dynamics (MD) Data Analysis

  * [ ] Understand the structure of atomic trajectory data (n_steps × n_atoms × 3)
  * [ ] Can create 3D trajectory plots
  * [ ] Can compute the radial distribution function (RDF)
  * [ ] Can extract characteristic interatomic distances from RDF peak positions
  * [ ] Can compute the mean squared displacement (MSD)
  * [ ] Can derive the diffusion coefficient from the MSD (Einstein relation)

#### Anomaly Detection

  * [ ] Can implement Isolation Forest
  * [ ] Can set contamination (fraction of anomalous data)
  * [ ] Can compute anomaly scores (`score_samples`)
  * [ ] Can evaluate anomaly detection accuracy with a confusion matrix
  * [ ] Can visualize the distributions of normal and anomalous data

* * *

### 3.12.8 Code Quality Skills

#### Basic Level

  * [ ] Set a random seed (`random_state=42`) in all code
  * [ ] Perform data validation (shape, dtype, missing values, ranges)
  * [ ] Use clear variable names (`X_train`, `y_test`, `model_lgb`)
  * [ ] Explain the purpose of each step with comments
  * [ ] Add titles, axis labels, and legends to plots

#### Applied Level

  * [ ] Turn code into reusable functions `python def calculate_rdf(positions, r_max, n_bins): ...`
  * [ ] Write documentation strings (docstrings)
  * [ ] Polish the appearance of plots (fontsize, grid, alpha)
  * [ ] Implement error handling with try-except (handling the pymoo ImportError)

* * *

### 3.12.9 Troubleshooting Skills

#### Basic Level (Handling Errors)

  * [ ] Can resolve `ModuleNotFoundError` (`pip install`)
  * [ ] Can resolve `ValueError: Input contains NaN` (missing value handling)
  * [ ] Can resolve `ConvergenceWarning` (MLP convergence errors)
  * Increase `max_iter`
  * Standardize the data
  * Enable early stopping
  * [ ] Can read error messages and search for solutions

#### Applied Level (Performance Improvement)

  * [ ] When R² < 0.7, can apply three or more improvement strategies
  * Feature engineering
  * Model change (linear → nonlinear)
  * Hyperparameter tuning
  * [ ] Can detect overfitting (training R² ≫ test R²)
  * [ ] Can detect underfitting (both training R² and test R² are low)

* * *

### 3.12.10 Overall Assessment: Proficiency Level Check

Use the level assessment below to check your progress.

#### Level 1: Beginner

  * Environment setup skills: 100% of the basic level achieved
  * Data processing and visualization skills: 80% or more of the basic level achieved
  * ML model implementation skills: implemented at least 3 of the 5 basic-level models
  * Troubleshooting: resolved basic-level errors on your own

**Target:** Generate and visualize nanoparticle data, and implement LSPR wavelength prediction with linear regression and random forest

* * *

#### Level 2: Intermediate

  * Environment setup skills: 80% or more of the applied level achieved
  * Data processing and visualization skills: 100% of the basic level + 70% or more of the applied level
  * ML model implementation skills: 100% of the basic level + 70% or more of the applied level
  * Feature importance and interpretability skills: 100% of the basic level + 50% or more of the applied level
  * Bayesian optimization skills: 100% of the basic level + 50% or more of the applied level

**Target:** Compare the five regression models and discover synthesis conditions that achieve the target LSPR wavelength (550 nm) with Bayesian optimization

* * *

#### Level 3: Advanced

  * All categories: 100% of the applied level achieved
  * Feature importance and interpretability skills: 80% or more of the advanced level
  * Bayesian optimization skills: 80% or more of the advanced level
  * Multi-objective optimization skills: 100% of the applied level achieved
  * Nanomaterial-specific analysis skills: TEM, MD, and anomaly detection all implemented

**Target:** Interpret models with SHAP analysis and visualize the size-versus-efficiency trade-off with multi-objective optimization

* * *

#### Level 4: Expert

  * All categories: 80% or more of the advanced level achieved
  * Code quality: 100% of the applied level achieved
  * Can apply the techniques to your own nanomaterials data (experimental or literature data)
  * Can build custom machine learning pipelines
  * Can present research results at conferences and submit papers

**Targets:** \- Integrate and analyze real nanomaterials data (TEM, UV-Vis, XRD) \- Predict the properties of new nanoparticles with over 90% accuracy using machine learning \- Reduce the number of experiments to 1/5 of the conventional count with Bayesian optimization

* * *

### 3.12.11 Practical Project Check: Completing the Exercises

#### Exercise 1 Completion Check (CNT Electrical Conductivity Prediction)

  * [ ] Implemented data generation (150 samples, 3 features)
  * [ ] Implemented prediction with a LightGBM model
  * [ ] Achieved R² > 0.8 and RMSE < 0.5
  * [ ] Visualized the feature importance
  * [ ] Interpreted the results (which feature has the largest influence)

#### Exercise 2 Completion Check (Optimal Silver Nanoparticle Synthesis Conditions)

  * [ ] Generated the silver nanoparticle data (100 samples)
  * [ ] Built a LightGBM model
  * [ ] Ran Bayesian optimization (40 evaluations)
  * [ ] Achieved an error < 1 nm relative to the target size (10 nm)
  * [ ] Identified the optimal temperature and pH

#### Exercise 3 Completion Check (Multi-Color Quantum Dot Emission Design)

  * [ ] Ran the optimization for the three colors: red, green, and blue
  * [ ] Identified the optimal size and synthesis conditions for each color
  * [ ] Predicted wavelengths fall within ±10 nm of the target wavelengths
  * [ ] Visualized the results (size vs wavelength plot)

* * *

### 3.12.12 Readiness Check for the Next Steps

#### Preparation for Real-World Applications (Chapter 4)

  * [ ] Understand the basic machine learning workflow (data preparation → model training → evaluation → optimization)
  * [ ] Can handle nanomaterial-specific data (size distributions, optical properties, electrical properties)
  * [ ] Can implement optimization methods (Bayesian optimization, multi-objective optimization)
  * [ ] Understand the importance of model interpretation (SHAP)

#### Preparation for Deep Learning and Graph Neural Networks

  * [ ] Have implemented a neural network (MLP) and understand activation functions and optimization algorithms
  * [ ] Can visualize learning curves and detect overfitting
  * [ ] Understand the concept of early stopping

#### Preparation for Practical Research

  * [ ] Can manage code with Jupyter Notebooks or Python scripts
  * [ ] Make environments reproducible with requirements.txt
  * [ ] Can plot prediction results and compile them into reports
  * [ ] Write documentation in your code

* * *

**Tips for Using the Checklist:** 1\. **Review regularly** : Re-check after studying, then one week later, then one month later 2\. **Prioritize unmet items** : Focus your study on items you cannot yet check off 3\. **Record your level assessment** : Visualize your growth to stay motivated 4\. **Use it in real projects** : Verify the required skills before starting a research or development project

* * *

## References

  1. **Pedregosa, F. et al.** (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_ , 12, 2825-2830.

  2. **Ke, G. et al.** (2017). LightGBM: A highly efficient gradient boosting decision tree. _Advances in Neural Information Processing Systems_ , 30, 3146-3154.

  3. **Lundberg, S. M. & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems_ , 30, 4765-4774.

  4. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). Practical Bayesian optimization of machine learning algorithms. _Advances in Neural Information Processing Systems_ , 25, 2951-2959.

  5. **Deb, K. et al.** (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. _IEEE Transactions on Evolutionary Computation_ , 6(2), 182-197. [DOI: 10.1109/4235.996017](<https://doi.org/10.1109/4235.996017>)

  6. **Frenkel, D. & Smit, B.** (2001). _Understanding Molecular Simulation: From Algorithms to Applications_ (2nd ed.). Academic Press.

* * *

[← Previous Chapter: Fundamentals of Nanomaterials](<chapter2-fundamentals.html>) | [Next Chapter: Real-World Applications and Careers →](<chapter4-real-world.html>)
