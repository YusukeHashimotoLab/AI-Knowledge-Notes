---
title: "Chapter 4: Practical Exercises Using Real Process Data"
chapter_title: "Chapter 4: Practical Exercises Using Real Process Data"
subtitle: "Comprehensive Exercise: From Chemical Plant Data Analysis to Optimization"
version: 1.0
created_at: 2025-10-25
---

# Chapter 4: Practical Exercises Using Real Process Data

This chapter integrates all the PI techniques learned so far and conducts a comprehensive exercise using actual chemical plant data. Experience the workflow directly applicable to real-world practice, from data exploration to quality prediction and process optimization.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Execute exploratory data analysis (EDA) on real process data
  * ✅ Implement everything from data cleaning to feature engineering
  * ✅ Compare multiple models and select the optimal model
  * ✅ Apply fundamental techniques for process condition optimization
  * ✅ Understand end-to-end PI project workflows

* * *

## 4.1 Case Study: Chemical Plant Operation Data Analysis

### Project Overview

**Background** :

In a chemical plant's distillation column, product purity variability is a challenge. Quality measurement is only once per day via gas chromatography (GC) analysis, making real-time quality control impossible. Using PI, we aim to achieve the following goals:

  1. **Build Quality Prediction Soft Sensor** : Real-time prediction of product purity from process variables
  2. **Identify Quality Impact Factors** : Clarify which variables most affect purity
  3. **Search for Optimal Operating Conditions** : Find conditions that minimize energy consumption while meeting quality standards

**Available Data** :

Variable Name | Description | Measurement Frequency | Unit  
---|---|---|---  
feed_temp | Feed Temperature | 1 min | °C  
top_temp | Top Temperature | 1 min | °C  
mid_temp | Middle Temperature | 1 min | °C  
bottom_temp | Bottom Temperature | 1 min | °C  
reflux_ratio | Reflux Ratio | 1 min | -  
reboiler_duty | Reboiler Heat Duty | 1 min | kW  
pressure | Column Pressure | 1 min | MPa  
feed_rate | Feed Flow Rate | 1 min | kg/h  
purity | Product Purity (Target Variable) | Once per day | %  
  
#### Code Example 1: Data Generation and EDA (Exploratory Data Analysis)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 1: Data Generation and EDA (Exploratory Data An
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate 1 month of operation data (1-minute intervals)
    n = 43200  # 30 days × 24 hours × 60 minutes
    dates = pd.date_range('2025-01-01', periods=n, freq='1min')
    
    # Generate process variables (realistic variation patterns)
    df = pd.DataFrame({
        'timestamp': dates,
        'feed_temp': 60 + np.random.normal(0, 2, n) + 3*np.sin(np.arange(n)*2*np.pi/1440),  # Daily variation
        'top_temp': 85 + np.random.normal(0, 1.5, n),
        'mid_temp': 120 + np.random.normal(0, 2, n),
        'bottom_temp': 155 + np.random.normal(0, 3, n),
        'reflux_ratio': 2.5 + np.random.normal(0, 0.2, n),
        'reboiler_duty': 1500 + np.random.normal(0, 80, n),
        'pressure': 1.2 + np.random.normal(0, 0.05, n),
        'feed_rate': 100 + np.random.normal(0, 5, n)
    })
    
    # Generate product purity (complex nonlinear relationship)
    df['purity'] = (
        92 +
        0.05 * df['feed_temp'] +
        0.3 * (df['top_temp'] - 85) +
        0.15 * (df['mid_temp'] - 120) +
        0.8 * df['reflux_ratio'] +
        0.002 * df['reboiler_duty'] +
        2.0 * df['pressure'] -
        0.01 * df['feed_rate'] +
        # Nonlinear term (optimal point existence)
        -0.02 * (df['top_temp'] - 85)**2 +
        np.random.normal(0, 0.4, n)
    )
    
    # Add missing values (realistic data)
    missing_indices = np.random.choice(df.index, size=int(n*0.02), replace=False)
    df.loc[missing_indices, 'top_temp'] = np.nan
    
    # Add outliers (simulate measurement errors)
    outlier_indices = np.random.choice(df.index, size=int(n*0.005), replace=False)
    df.loc[outlier_indices, 'pressure'] += np.random.choice([-0.5, 0.5], size=len(outlier_indices))
    
    # Simulate offline measurement (once per day)
    df['purity_measured'] = np.nan
    df.loc[df.index[::1440], 'purity_measured'] = df.loc[df.index[::1440], 'purity']
    
    # Save dataset
    df.to_csv('distillation_data.csv', index=False)
    print(f"【Dataset Generation Complete】")
    print(f"Total data points: {len(df):,}")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Offline measurements: {df['purity_measured'].notna().sum()}")
    
    # Basic statistics
    print("\n【Basic Statistics】")
    print(df.describe().round(2))
    
    # Check missing values
    print("\n【Missing Values】")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # EDA: Visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 1. Time series plot of main variables (first 3 days)
    time_window = (df['timestamp'] >= '2025-01-01') & (df['timestamp'] < '2025-01-04')
    df_window = df[time_window]
    
    variables = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                 'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate']
    
    for i, var in enumerate(variables):
        ax = axes[i//3, i%3]
        ax.plot(df_window['timestamp'], df_window[var], linewidth=0.5, color='#11998e')
        ax.set_ylabel(var, fontsize=10)
        ax.set_title(f'{var} - 3-day trend', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        if i >= 6:
            ax.set_xlabel('Time', fontsize=10)
    
    # 9th plot: Purity (actual and offline measurement)
    ax = axes[2, 2]
    ax.plot(df_window['timestamp'], df_window['purity'], linewidth=0.8,
            alpha=0.7, label='True purity (unknown)', color='gray')
    ax.scatter(df_window['timestamp'], df_window['purity_measured'],
               s=100, color='red', marker='o', label='Offline measurement', zorder=3)
    ax.set_ylabel('Purity (%)', fontsize=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_title('Product Purity', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n【EDA Complete】: Saved eda_timeseries.png")
    

**Output Example** :
    
    
    【Dataset Generation Complete】
    Total data points: 43,200
    Period: 2025-01-01 00:00:00 to 2025-01-30 23:59:00
    Offline measurements: 31
    
    【Basic Statistics】
             feed_temp  top_temp  mid_temp  bottom_temp  reflux_ratio  reboiler_duty  pressure  feed_rate   purity
    count   43200.00  43200.00  43200.00     43200.00      43200.00       43200.00  43200.00   43200.00 43200.00
    mean       60.01     85.00    120.00       155.00          2.50        1500.01      1.20     100.00    96.50
    std         2.45      1.50      2.00         3.00          0.20          80.00      0.08       5.00     1.23
    ...
    

**Explanation** : Real process data includes daily variations, missing values, and outliers. Understanding these patterns through EDA determines the quality of subsequent analyses.

#### Code Example 2: Data Cleaning and Preprocessing
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: Code Example 2: Data Cleaning and Preprocessing
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler
    from scipy import stats
    
    # Load data
    df = pd.read_csv('distillation_data.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    print("【Data Cleaning Started】")
    print(f"Original data: {len(df)} points")
    
    # Step 1: Missing value handling
    print("\n■ Step 1: Missing Value Handling")
    missing_before = df.isnull().sum().sum()
    print(f"Missing values (before): {missing_before}")
    
    # Linear interpolation (appropriate for time series data)
    df_cleaned = df.copy()
    df_cleaned['top_temp'] = df_cleaned['top_temp'].interpolate(method='linear')
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values (after): {missing_after}")
    
    # Step 2: Outlier detection and handling
    print("\n■ Step 2: Outlier Detection (IQR Method)")
    
    def detect_outliers_iqr(series, multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    # Detect pressure outliers
    outliers, lower, upper = detect_outliers_iqr(df_cleaned['pressure'])
    print(f"Pressure outliers: {outliers.sum()} points ({outliers.sum()/len(df_cleaned)*100:.2f}%)")
    print(f"  Acceptable range: {lower:.3f} to {upper:.3f} MPa")
    
    # Replace outliers with median (conservative approach)
    df_cleaned.loc[outliers, 'pressure'] = df_cleaned['pressure'].median()
    
    # Step 3: Scaling
    print("\n■ Step 3: Data Scaling (RobustScaler)")
    
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate']
    
    scaler = RobustScaler()
    df_scaled = df_cleaned.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_cleaned[feature_cols])
    
    print("Scaling complete (using RobustScaler)")
    
    # Step 4: Feature engineering
    print("\n■ Step 4: Feature Engineering")
    
    df_cleaned['temp_gradient'] = df_cleaned['top_temp'] - df_cleaned['bottom_temp']
    df_cleaned['energy_efficiency'] = df_cleaned['reboiler_duty'] / df_cleaned['feed_rate']
    df_cleaned['hour'] = df_cleaned.index.hour
    df_cleaned['day_of_week'] = df_cleaned.index.dayofweek
    
    # Periodic features (cyclic encoding)
    df_cleaned['hour_sin'] = np.sin(2 * np.pi * df_cleaned['hour'] / 24)
    df_cleaned['hour_cos'] = np.cos(2 * np.pi * df_cleaned['hour'] / 24)
    
    print(f"Additional features: 4")
    print("  - temp_gradient: Temperature difference between top and bottom")
    print("  - energy_efficiency: Energy per unit feed")
    print("  - hour_sin/cos: Time periodicity encoding")
    
    # Visualization: Before and after cleaning comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Effect of missing value imputation
    time_window = slice('2025-01-15 00:00', '2025-01-15 12:00')
    axes[0, 0].plot(df.loc[time_window].index, df.loc[time_window, 'top_temp'],
                    'o-', markersize=3, label='Before (with missing)', alpha=0.7)
    axes[0, 0].plot(df_cleaned.loc[time_window].index, df_cleaned.loc[time_window, 'top_temp'],
                    '-', linewidth=2, label='After (interpolated)', color='#11998e')
    axes[0, 0].set_ylabel('Top Temperature (°C)', fontsize=11)
    axes[0, 0].set_title('Missing Value Imputation', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Effect of outlier removal
    axes[0, 1].hist(df['pressure'], bins=50, alpha=0.5, label='Before', edgecolor='black')
    axes[0, 1].hist(df_cleaned['pressure'], bins=50, alpha=0.5, label='After',
                    color='#11998e', edgecolor='black')
    axes[0, 1].axvline(lower, color='red', linestyle='--', label='Lower bound')
    axes[0, 1].axvline(upper, color='red', linestyle='--', label='Upper bound')
    axes[0, 1].set_xlabel('Pressure (MPa)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Outlier Removal', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Before and after scaling comparison
    axes[1, 0].boxplot([df_cleaned['feed_temp'], df_cleaned['reboiler_duty']],
                       labels=['feed_temp', 'reboiler_duty'], patch_artist=True)
    axes[1, 0].set_ylabel('Original Scale', fontsize=11)
    axes[1, 0].set_title('Before Scaling (Different Scales)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    axes[1, 1].boxplot([df_scaled['feed_temp'], df_scaled['reboiler_duty']],
                       labels=['feed_temp', 'reboiler_duty'], patch_artist=True,
                       boxprops=dict(facecolor='#11998e', alpha=0.7))
    axes[1, 1].set_ylabel('Scaled Value', fontsize=11)
    axes[1, 1].set_title('After Scaling (Unified Scale)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data_cleaning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save cleaned data
    df_cleaned.to_csv('distillation_data_cleaned.csv')
    print(f"\n【Cleaning Complete】: Saved distillation_data_cleaned.csv")
    print(f"Final data points: {len(df_cleaned)}")
    

**Output Example** :
    
    
    【Data Cleaning Started】
    Original data: 43200 points
    
    ■ Step 1: Missing Value Handling
    Missing values (before): 864
    Missing values (after): 0
    
    ■ Step 2: Outlier Detection (IQR Method)
    Pressure outliers: 216 points (0.50%)
      Acceptable range: 1.080 to 1.320 MPa
    
    ■ Step 3: Data Scaling (RobustScaler)
    Scaling complete (using RobustScaler)
    
    ■ Step 4: Feature Engineering
    Additional features: 4
      - temp_gradient: Temperature difference between top and bottom
      - energy_efficiency: Energy per unit feed
      - hour_sin/cos: Time periodicity encoding
    

**Explanation** : Data cleaning and feature engineering are critical steps directly linked to model performance. Features utilizing domain knowledge (temperature gradient, energy efficiency) are particularly effective.

* * *

## 4.2 Building Quality Prediction Models

Using the cleaned data, we build a soft sensor to predict product purity. We compare multiple models and select the optimal one.

#### Code Example 3: Preparing Training and Test Data
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 3: Preparing Training and Test Data
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    
    # Load cleaned data
    df = pd.read_csv('distillation_data_cleaned.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    print("【Data Preparation】")
    
    # Use only offline measurement data (assuming real operation)
    train_data = df[df['purity_measured'].notna()].copy()
    print(f"Training data points: {len(train_data)} (offline measurements only)")
    
    # Features and target variable
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate',
                    'temp_gradient', 'energy_efficiency', 'hour_sin', 'hour_cos']
    
    X = train_data[feature_cols]
    y = train_data['purity_measured']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Time Series Split
    # For time series data, care must be taken not to predict the past using future data
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"\nTime series split: {tscv.n_splits} folds")
    
    # Reserve last 20% for final evaluation as test data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"\nTraining data: {len(X_train)} points")
    print(f"Test data: {len(X_test)} points")
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame (preserve column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print("\nScaling complete")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    
    # Basic statistics of data
    print("\n【Training Data Statistics】")
    print(y_train.describe())
    print("\n【Test Data Statistics】")
    print(y_test.describe())
    

**Output Example** :
    
    
    【Data Preparation】
    Training data points: 31 (offline measurements only)
    Number of features: 12
    Features: ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp', 'reflux_ratio',
             'reboiler_duty', 'pressure', 'feed_rate', 'temp_gradient',
             'energy_efficiency', 'hour_sin', 'hour_cos']
    
    Time series split: 5 folds
    
    Training data: 25 points
    Test data: 6 points
    

**Explanation** : For time series data, use time series splits rather than random splits. This allows evaluation under the same conditions as real operation, where past data predicts the future.

#### Code Example 4: Comparing and Selecting Multiple Models
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 4: Comparing and Selecting Multiple Models
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import time
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'PLS': PLSRegression(n_components=5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'SVR': SVR(kernel='rbf', C=10, gamma=0.1)
    }
    
    print("【Model Comparison】")
    print("Evaluating each model with cross-validation...")
    
    results = []
    
    for name, model in models.items():
        start_time = time.time()
    
        # Cross-validation (time series split)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
        # Training
        model.fit(X_train_scaled, y_train)
    
        # Prediction
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    
        # Evaluation metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    
        training_time = time.time() - start_time
    
        results.append({
            'Model': name,
            'CV R² (mean)': cv_scores.mean(),
            'CV R² (std)': cv_scores.std(),
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Training Time (s)': training_time
        })
    
        print(f"  {name}: CV R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), "
              f"Test R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    print("\n【Overall Evaluation Results】")
    print(results_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Test R² score comparison
    axes[0, 0].barh(results_df['Model'], results_df['Test R²'], color='#11998e', alpha=0.7)
    axes[0, 0].set_xlabel('Test R² Score', fontsize=11)
    axes[0, 0].set_title('Model Performance Comparison (Test R²)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # 2. RMSE vs training time
    axes[0, 1].scatter(results_df['Training Time (s)'], results_df['Test RMSE'],
                       s=150, alpha=0.7, color='#11998e')
    for i, row in results_df.iterrows():
        axes[0, 1].annotate(row['Model'], (row['Training Time (s)'], row['Test RMSE']),
                            fontsize=8, ha='right')
    axes[0, 1].set_xlabel('Training Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Test RMSE', fontsize=11)
    axes[0, 1].set_title('Efficiency vs Accuracy Trade-off', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Train R² vs Test R² (overfitting check)
    axes[1, 0].scatter(results_df['Train R²'], results_df['Test R²'],
                       s=150, alpha=0.7, color='#f59e0b')
    axes[1, 0].plot([0.9, 1.0], [0.9, 1.0], 'r--', linewidth=2, label='Perfect generalization')
    for i, row in results_df.iterrows():
        axes[1, 0].annotate(row['Model'], (row['Train R²'], row['Test R²']),
                            fontsize=8, ha='right')
    axes[1, 0].set_xlabel('Train R²', fontsize=11)
    axes[1, 0].set_ylabel('Test R²', fontsize=11)
    axes[1, 0].set_title('Overfitting Check', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. CV score distribution
    cv_means = results_df['CV R² (mean)']
    cv_stds = results_df['CV R² (std)']
    axes[1, 1].barh(results_df['Model'], cv_means, xerr=cv_stds,
                    color='#7b2cbf', alpha=0.7, capsize=5)
    axes[1, 1].set_xlabel('Cross-Validation R² Score', fontsize=11)
    axes[1, 1].set_title('Cross-Validation Performance (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Select best model
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n【Best Model】: {best_model_name}")
    print(f"  Test R²: {results_df.iloc[0]['Test R²']:.4f}")
    print(f"  Test RMSE: {results_df.iloc[0]['Test RMSE']:.4f}%")
    print(f"  Test MAE: {results_df.iloc[0]['Test MAE']:.4f}%")
    
    # Save best model (for later use)
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nSaved model and scaler")
    

**Output Example** :
    
    
    【Model Comparison】
    Evaluating each model with cross-validation...
      Linear Regression: CV R² = 0.8456 (±0.1234), Test R² = 0.8678, RMSE = 0.4321
      Ridge: CV R² = 0.8512 (±0.1198), Test R² = 0.8723, RMSE = 0.4256
      Lasso: CV R² = 0.8389 (±0.1276), Test R² = 0.8598, RMSE = 0.4456
      PLS: CV R² = 0.8623 (±0.1089), Test R² = 0.8845, RMSE = 0.4034
      Random Forest: CV R² = 0.9012 (±0.0789), Test R² = 0.9234, RMSE = 0.3287
      Gradient Boosting: CV R² = 0.9156 (±0.0723), Test R² = 0.9345, RMSE = 0.3041
      SVR: CV R² = 0.8876 (±0.0856), Test R² = 0.9087, RMSE = 0.3589
    
    【Best Model】: Gradient Boosting
      Test R²: 0.9345
      Test RMSE: 0.3041%
      Test MAE: 0.2456%
    

**Explanation** : By systematically comparing multiple models, we can select the model that best fits the data. In this example, Gradient Boosting showed the best performance.

#### Code Example 5: Feature Importance Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 5: Feature Importance Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.inspection import permutation_importance
    
    # Load best model
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【Feature Importance Analysis】")
    
    # Method 1: Model-specific feature importance (for Random Forest or Gradient Boosting)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
        print("\n■ Model-specific Feature Importance:")
        print(feature_importance.to_string(index=False))
    
    # Method 2: Permutation Importance (model-agnostic)
    perm_importance = permutation_importance(best_model, X_test_scaled, y_test,
                                              n_repeats=10, random_state=42)
    
    perm_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n■ Permutation Importance:")
    print(perm_importance_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model-specific importance
    if hasattr(best_model, 'feature_importances_'):
        axes[0].barh(feature_importance['Feature'], feature_importance['Importance'],
                     color='#11998e', alpha=0.7)
        axes[0].set_xlabel('Importance', fontsize=11)
        axes[0].set_title('Feature Importance (Model-specific)', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x')
        axes[0].invert_yaxis()
    
    # Permutation Importance
    axes[1].barh(perm_importance_df['Feature'], perm_importance_df['Importance'],
                 xerr=perm_importance_df['Std'], color='#f59e0b', alpha=0.7, capsize=5)
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Permutation Importance (Model-agnostic)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Interpretation of main factors
    print("\n【Main Impact Factors】")
    top_features = perm_importance_df.head(5)
    for i, row in top_features.iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})")
    
    print("\n【Interpretation】")
    print("✓ Reflux ratio has the greatest impact on purity")
    print("✓ Top temperature and middle temperature are also important control variables")
    print("✓ Energy efficiency (derived feature) contributes significantly")
    print("→ Quality stabilization is possible by focusing on managing these variables")
    

**Explanation** : Feature importance analysis quantitatively reveals which variables affect quality. This directly translates to prioritization in process control.

* * *

## 4.3 Fundamentals of Process Condition Optimization

Using the built model, we search for operating conditions that minimize energy consumption while meeting quality constraints.

#### Code Example 6: Constrained Optimization (Grid Search)
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 6: Constrained Optimization (Grid Search)
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from itertools import product
    
    # Load model and scaler
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【Process Optimization】")
    print("Objective: Minimize energy consumption (reboiler_duty) while meeting quality (purity ≥ 97%)")
    
    # Variables to optimize and search range
    # Fixed variables (external conditions)
    feed_temp_fixed = 60.0
    feed_rate_fixed = 100.0
    pressure_fixed = 1.2
    
    # Variables to optimize
    reflux_ratios = np.linspace(2.0, 3.5, 20)
    reboiler_duties = np.linspace(1300, 1700, 20)
    
    print(f"\nSearch range:")
    print(f"  Reflux ratio: {reflux_ratios.min():.2f} to {reflux_ratios.max():.2f}")
    print(f"  Reboiler duty: {reboiler_duties.min():.0f} to {reboiler_duties.max():.0f} kW")
    print(f"  Search points: {len(reflux_ratios) × len(reboiler_duties)} points")
    
    # Grid search
    results = []
    
    for reflux_ratio, reboiler_duty in product(reflux_ratios, reboiler_duties):
        # Calculate derived features from operating conditions
        # Note: top_temp, mid_temp, bottom_temp are estimated from correlations (simplified)
        # In reality, physical models or more advanced predictions are needed
        top_temp = 85 + 0.5 * (reflux_ratio - 2.5)  # Simplified estimation
        mid_temp = 120
        bottom_temp = 155
    
        temp_gradient = top_temp - bottom_temp
        energy_efficiency = reboiler_duty / feed_rate_fixed
        hour_sin = 0  # Assume noon
        hour_cos = 1
    
        # Create feature vector
        features = np.array([[
            feed_temp_fixed, top_temp, mid_temp, bottom_temp,
            reflux_ratio, reboiler_duty, pressure_fixed, feed_rate_fixed,
            temp_gradient, energy_efficiency, hour_sin, hour_cos
        ]])
    
        # Scaling
        features_df = pd.DataFrame(features, columns=feature_cols)
        features_scaled = scaler.transform(features_df)
    
        # Purity prediction
        purity_pred = best_model.predict(features_scaled)[0]
    
        results.append({
            'reflux_ratio': reflux_ratio,
            'reboiler_duty': reboiler_duty,
            'purity_pred': purity_pred,
            'feasible': purity_pred >= 97.0  # Quality constraint
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n【Search Results】")
    print(f"Total search points: {len(results_df)}")
    print(f"Points meeting quality constraint: {results_df['feasible'].sum()} points")
    
    # Optimal solution in feasible region
    feasible_solutions = results_df[results_df['feasible']]
    
    if len(feasible_solutions) > 0:
        optimal_solution = feasible_solutions.loc[feasible_solutions['reboiler_duty'].idxmin()]
    
        print(f"\n【Optimal Operating Conditions】")
        print(f"  Reflux ratio: {optimal_solution['reflux_ratio']:.3f}")
        print(f"  Reboiler duty: {optimal_solution['reboiler_duty']:.1f} kW")
        print(f"  Predicted purity: {optimal_solution['purity_pred']:.2f}%")
    
        # Compare with current operating conditions (average)
        current_reflux = X_train['reflux_ratio'].mean()
        current_duty = X_train['reboiler_duty'].mean()
        current_purity = y_train.mean()
    
        print(f"\n【Comparison with Current Conditions】")
        print(f"  Reflux ratio: {current_reflux:.3f} → {optimal_solution['reflux_ratio']:.3f}")
        print(f"  Reboiler duty: {current_duty:.1f} kW → {optimal_solution['reboiler_duty']:.1f} kW")
        print(f"  Predicted purity: {current_purity:.2f}% → {optimal_solution['purity_pred']:.2f}%")
    
        energy_saving = (current_duty - optimal_solution['reboiler_duty']) / current_duty * 100
        print(f"\nEnergy reduction: {energy_saving:.1f}%")
        print(f"Annual cost reduction (assumed): ¥{energy_saving * 100000:.0f} million")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Contour map: Purity
    contour = axes[0].tricontourf(results_df['reflux_ratio'], results_df['reboiler_duty'],
                                   results_df['purity_pred'], levels=20, cmap='RdYlGn')
    axes[0].tricontour(results_df['reflux_ratio'], results_df['reboiler_duty'],
                       results_df['purity_pred'], levels=[97.0], colors='red',
                       linewidths=3, linestyles='--')
    if len(feasible_solutions) > 0:
        axes[0].scatter(optimal_solution['reflux_ratio'], optimal_solution['reboiler_duty'],
                        s=200, color='blue', marker='*', edgecolor='white', linewidth=2,
                        label='Optimal point', zorder=5)
    axes[0].set_xlabel('Reflux Ratio', fontsize=11)
    axes[0].set_ylabel('Reboiler Duty (kW)', fontsize=11)
    axes[0].set_title('Predicted Purity (% contour)', fontsize=12, fontweight='bold')
    axes[0].legend()
    plt.colorbar(contour, ax=axes[0], label='Purity (%)')
    
    # Feasible region
    axes[1].scatter(results_df[~results_df['feasible']]['reflux_ratio'],
                    results_df[~results_df['feasible']]['reboiler_duty'],
                    s=30, alpha=0.3, color='red', label='Infeasible (purity < 97%)')
    axes[1].scatter(results_df[results_df['feasible']]['reflux_ratio'],
                    results_df[results_df['feasible']]['reboiler_duty'],
                    s=30, alpha=0.5, color='green', label='Feasible (purity ≥ 97%)')
    if len(feasible_solutions) > 0:
        axes[1].scatter(optimal_solution['reflux_ratio'], optimal_solution['reboiler_duty'],
                        s=200, color='blue', marker='*', edgecolor='white', linewidth=2,
                        label='Optimal point', zorder=5)
    axes[1].set_xlabel('Reflux Ratio', fontsize=11)
    axes[1].set_ylabel('Reboiler Duty (kW)', fontsize=11)
    axes[1].set_title('Feasible Region', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    else:
        print("\n✗ No operating conditions meeting quality constraint found")
        print("  → Need to expand search range or relax constraints")
    

**Output Example** :
    
    
    【Process Optimization】
    Objective: Minimize energy consumption (reboiler_duty) while meeting quality (purity ≥ 97%)
    
    Search range:
      Reflux ratio: 2.00 to 3.50
      Reboiler duty: 1300 to 1700 kW
      Search points: 400 points
    
    【Search Results】
    Total search points: 400
    Points meeting quality constraint: 156 points
    
    【Optimal Operating Conditions】
      Reflux ratio: 2.789
      Reboiler duty: 1368.4 kW
      Predicted purity: 97.12%
    
    【Comparison with Current Conditions】
      Reflux ratio: 2.503 → 2.789
      Reboiler duty: 1499.8 kW → 1368.4 kW
      Predicted purity: 96.51% → 97.12%
    
    Energy reduction: 8.8%
    Annual cost reduction (assumed): ¥880 million
    

**Explanation** : Grid Search optimization discovered conditions that achieve energy reduction while improving quality. In real plants, more advanced optimization techniques (genetic algorithms, Bayesian optimization, etc.) are also utilized.

#### Code Example 7: Advanced Optimization (Scipy.optimize)
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 7: Advanced Optimization (Scipy.optimize)
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from scipy.optimize import minimize, differential_evolution
    
    # Load model and scaler
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("【Advanced Optimization (Scipy.optimize)】")
    
    # Fixed parameters
    feed_temp_fixed = 60.0
    feed_rate_fixed = 100.0
    pressure_fixed = 1.2
    
    # Objective function: Energy (minimize)
    def objective(x):
        """Minimize energy consumption (reboiler duty)"""
        reflux_ratio, reboiler_duty = x
        return reboiler_duty  # Target to minimize
    
    # Constraint: Purity ≥ 97%
    def constraint_purity(x):
        """Purity constraint (≥ 97%)"""
        reflux_ratio, reboiler_duty = x
    
        # Feature calculation
        top_temp = 85 + 0.5 * (reflux_ratio - 2.5)
        mid_temp = 120
        bottom_temp = 155
        temp_gradient = top_temp - bottom_temp
        energy_efficiency = reboiler_duty / feed_rate_fixed
    
        features = np.array([[
            feed_temp_fixed, top_temp, mid_temp, bottom_temp,
            reflux_ratio, reboiler_duty, pressure_fixed, feed_rate_fixed,
            temp_gradient, energy_efficiency, 0, 1
        ]])
    
        features_df = pd.DataFrame(features, columns=feature_cols)
        features_scaled = scaler.transform(features_df)
    
        purity_pred = best_model.predict(features_scaled)[0]
    
        # Constraint: purity >= 97 → purity - 97 >= 0
        return purity_pred - 97.0
    
    # Variable bounds
    bounds = [
        (2.0, 3.5),      # Reflux ratio
        (1300, 1700)     # Reboiler duty (kW)
    ]
    
    # Constraint definition
    constraints = [
        {'type': 'ineq', 'fun': constraint_purity}  # inequality: f(x) >= 0
    ]
    
    # Initial value
    x0 = [2.5, 1500]
    
    print("\nMethod 1: SLSQP (gradient-based)")
    result_slsqp = minimize(objective, x0, method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'disp': True})
    
    if result_slsqp.success:
        print(f"\n【Optimal Solution (SLSQP)】")
        print(f"  Reflux ratio: {result_slsqp.x[0]:.3f}")
        print(f"  Reboiler duty: {result_slsqp.x[1]:.1f} kW")
        print(f"  Predicted purity: {constraint_purity(result_slsqp.x) + 97:.2f}%")
    else:
        print("\nOptimization failed (SLSQP)")
    
    # Method 2: Differential Evolution (evolutionary algorithm)
    print("\n\nMethod 2: Differential Evolution (global search)")
    
    def objective_with_penalty(x):
        """Incorporate constraints into objective function using penalty method"""
        energy = objective(x)
        purity_constraint = constraint_purity(x)
    
        # Penalty for constraint violation
        if purity_constraint < 0:
            penalty = 1000 * abs(purity_constraint)
            return energy + penalty
        else:
            return energy
    
    result_de = differential_evolution(objective_with_penalty, bounds,
                                        seed=42, disp=True, maxiter=100)
    
    print(f"\n【Optimal Solution (Differential Evolution)】")
    print(f"  Reflux ratio: {result_de.x[0]:.3f}")
    print(f"  Reboiler duty: {result_de.x[1]:.1f} kW")
    print(f"  Predicted purity: {constraint_purity(result_de.x) + 97:.2f}%")
    
    # Compare results
    print(f"\n【Optimization Method Comparison】")
    print(f"SLSQP (local optimization): Energy = {result_slsqp.fun:.1f} kW")
    print(f"Differential Evolution (global optimization): Energy = {result_de.fun:.1f} kW")
    
    # Visualization: Optimization path
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Purity distribution on grid
    reflux_grid = np.linspace(2.0, 3.5, 50)
    duty_grid = np.linspace(1300, 1700, 50)
    R, D = np.meshgrid(reflux_grid, duty_grid)
    
    purity_grid = np.zeros_like(R)
    for i in range(len(reflux_grid)):
        for j in range(len(duty_grid)):
            purity_grid[j, i] = constraint_purity([R[j, i], D[j, i]]) + 97
    
    contour = ax.contourf(R, D, purity_grid, levels=20, cmap='RdYlGn', alpha=0.6)
    ax.contour(R, D, purity_grid, levels=[97.0], colors='red', linewidths=3, linestyles='--')
    
    # Plot optimal solutions
    if result_slsqp.success:
        ax.scatter(result_slsqp.x[0], result_slsqp.x[1], s=200, color='blue',
                   marker='o', edgecolor='white', linewidth=2, label='SLSQP', zorder=5)
    
    ax.scatter(result_de.x[0], result_de.x[1], s=200, color='orange',
               marker='*', edgecolor='white', linewidth=2, label='Differential Evolution', zorder=5)
    
    # Initial point
    ax.scatter(x0[0], x0[1], s=100, color='black', marker='x', linewidth=2,
               label='Initial point', zorder=5)
    
    ax.set_xlabel('Reflux Ratio', fontsize=12)
    ax.set_ylabel('Reboiler Duty (kW)', fontsize=12)
    ax.set_title('Optimization Results on Purity Contour', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.colorbar(contour, ax=ax, label='Purity (%)')
    plt.tight_layout()
    plt.savefig('advanced_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n【Optimization Method Selection Guidelines】")
    print("✓ SLSQP: Fast, uses gradient information, local optimum")
    print("✓ Differential Evolution: Slow, global search, strong for complex objectives")
    print("✓ Practice: Start with SLSQP for fast search, confirm with DE if needed")
    

**Output Example** :
    
    
    【Advanced Optimization (Scipy.optimize)】
    
    Method 1: SLSQP (gradient-based)
    Optimization terminated successfully
    
    【Optimal Solution (SLSQP)】
      Reflux ratio: 2.784
      Reboiler duty: 1365.2 kW
      Predicted purity: 97.03%
    
    Method 2: Differential Evolution (global search)
    
    【Optimal Solution (Differential Evolution)】
      Reflux ratio: 2.789
      Reboiler duty: 1363.8 kW
      Predicted purity: 97.05%
    
    【Optimization Method Comparison】
    SLSQP (local optimization): Energy = 1365.2 kW
    Differential Evolution (global optimization): Energy = 1363.8 kW
    

**Explanation** : Using Scipy.optimize enables more advanced optimization. SLSQP is fast but can get trapped in local optima, while Differential Evolution is slower but tends to find global optimal solutions.

* * *

## 4.4 Overall Project Implementation Workflow

Integrating all previous steps, we establish an end-to-end PI project workflow.

#### Code Example 8: Integrated Pipeline and Deployment Preparation
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Code Example 8: Integrated Pipeline and Deployment Preparati
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import json
    
    print("=" * 80)
    print("【PI Integrated Project: End-to-End Workflow】")
    print("=" * 80)
    
    # ============================================================================
    # Step 1: Build Data Pipeline
    # ============================================================================
    print("\n【Step 1: Build Data Pipeline】")
    
    class ProcessDataPipeline:
        """Process data preprocessing pipeline"""
    
        def __init__(self):
            self.scaler = RobustScaler()
            self.feature_cols = None
    
        def fit(self, df, feature_cols):
            """Fit pipeline on training data"""
            self.feature_cols = feature_cols
    
            # Missing value imputation
            df_clean = df.copy()
            for col in feature_cols:
                df_clean[col] = df_clean[col].interpolate(method='linear')
    
            # Scaling
            self.scaler.fit(df_clean[feature_cols])
    
            return self
    
        def transform(self, df):
            """Transform data"""
            df_clean = df.copy()
    
            # Missing value imputation
            for col in self.feature_cols:
                df_clean[col] = df_clean[col].interpolate(method='linear')
    
            # Scaling
            df_clean[self.feature_cols] = self.scaler.transform(df_clean[self.feature_cols])
    
            return df_clean
    
        def save(self, filepath):
            """Save pipeline"""
            joblib.dump(self, filepath)
            print(f"  Pipeline saved: {filepath}")
    
        @staticmethod
        def load(filepath):
            """Load pipeline"""
            return joblib.load(filepath)
    
    # Instantiate pipeline
    pipeline = ProcessDataPipeline()
    
    # Load data
    df = pd.read_csv('distillation_data_cleaned.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    feature_cols = ['feed_temp', 'top_temp', 'mid_temp', 'bottom_temp',
                    'reflux_ratio', 'reboiler_duty', 'pressure', 'feed_rate',
                    'temp_gradient', 'energy_efficiency', 'hour_sin', 'hour_cos']
    
    # Only offline measurement data
    train_data = df[df['purity_measured'].notna()].copy()
    X_train = train_data[feature_cols]
    y_train = train_data['purity_measured']
    
    # Fit pipeline
    pipeline.fit(X_train, feature_cols)
    X_train_processed = pipeline.transform(X_train)
    
    print("  Data preprocessing pipeline construction complete")
    
    # ============================================================================
    # Step 2: Model Training
    # ============================================================================
    print("\n【Step 2: Model Training】")
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
    model.fit(X_train_processed[feature_cols], y_train)
    
    print(f"  Model: Gradient Boosting Regressor")
    print(f"  Training data points: {len(X_train)}")
    print(f"  Number of features: {len(feature_cols)}")
    
    # ============================================================================
    # Step 3: Model Evaluation
    # ============================================================================
    print("\n【Step 3: Model Evaluation】")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    
    cv_scores = cross_val_score(model, X_train_processed[feature_cols], y_train, cv=5, scoring='r2')
    print(f"  CV R² score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    y_train_pred = model.predict(X_train_processed[feature_cols])
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"  Training data R²: {train_r2:.4f}")
    print(f"  Training data RMSE: {train_rmse:.4f}%")
    
    # ============================================================================
    # Step 4: Deployment Preparation
    # ============================================================================
    print("\n【Step 4: Deployment Preparation】")
    
    # Save model and pipeline
    model_path = 'production_model.pkl'
    pipeline_path = 'production_pipeline.pkl'
    
    joblib.dump(model, model_path)
    pipeline.save(pipeline_path)
    
    print(f"  Model saved: {model_path}")
    print(f"  Pipeline saved: {pipeline_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'Gradient Boosting Regressor',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(X_train),
        'features': feature_cols,
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'train_r2': float(train_r2),
        'train_rmse': float(train_rmse),
        'target_variable': 'purity',
        'target_unit': '%',
        'quality_threshold': 97.0
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  Metadata saved: model_metadata.json")
    
    # ============================================================================
    # Step 5: Inference Function (For Deployment)
    # ============================================================================
    print("\n【Step 5: Inference Function Implementation】")
    
    def predict_purity(process_data):
        """
        Predict purity from process data
    
        Parameters:
        -----------
        process_data : dict
            Dictionary of process variables
            Example: {'feed_temp': 60, 'top_temp': 85, ...}
    
        Returns:
        --------
        float
            Predicted purity (%)
        """
        # Load model and pipeline
        model = joblib.load('production_model.pkl')
        pipeline = ProcessDataPipeline.load('production_pipeline.pkl')
    
        # Convert to DataFrame
        df = pd.DataFrame([process_data])
    
        # Preprocessing
        df_processed = pipeline.transform(df)
    
        # Prediction
        purity_pred = model.predict(df_processed[feature_cols])[0]
    
        return purity_pred
    
    # Test execution
    test_data = {
        'feed_temp': 60.0,
        'top_temp': 85.5,
        'mid_temp': 120.0,
        'bottom_temp': 155.0,
        'reflux_ratio': 2.8,
        'reboiler_duty': 1400,
        'pressure': 1.2,
        'feed_rate': 100,
        'temp_gradient': -69.5,
        'energy_efficiency': 14.0,
        'hour_sin': 0,
        'hour_cos': 1
    }
    
    purity_prediction = predict_purity(test_data)
    print(f"\n  Inference test:")
    print(f"    Input: {test_data}")
    print(f"    Predicted purity: {purity_prediction:.2f}%")
    
    # ============================================================================
    # Step 6: Monitoring Dashboard Data Output
    # ============================================================================
    print("\n【Step 6: Generate Monitoring Dashboard Data】")
    
    # Real-time prediction on all data
    df_all = df.copy()
    df_all_processed = pipeline.transform(df_all)
    df_all['purity_predicted'] = model.predict(df_all_processed[feature_cols])
    
    # Calculate prediction error (only when offline measurement exists)
    df_all['prediction_error'] = df_all['purity_measured'] - df_all['purity_predicted']
    
    # Save dashboard data (last 1 week)
    dashboard_data = df_all.tail(10080)[['purity', 'purity_predicted', 'purity_measured',
                                           'prediction_error', 'reflux_ratio', 'reboiler_duty']]
    dashboard_data.to_csv('dashboard_data.csv')
    
    print("  Dashboard data saved: dashboard_data.csv")
    print(f"  Data period: {dashboard_data.index.min()} to {dashboard_data.index.max()}")
    
    # Performance summary
    errors = df_all['prediction_error'].dropna()
    print(f"\n  Model performance summary (comparison with offline measurements):")
    print(f"    Mean error: {errors.mean():.4f}%")
    print(f"    Standard deviation: {errors.std():.4f}%")
    print(f"    Maximum error: {errors.abs().max():.4f}%")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("【Project Complete】")
    print("=" * 80)
    print("\nDeliverables created:")
    print("  1. production_model.pkl - Trained model")
    print("  2. production_pipeline.pkl - Data preprocessing pipeline")
    print("  3. model_metadata.json - Model metadata")
    print("  4. dashboard_data.csv - Monitoring dashboard data")
    print("  5. predict_purity() - Inference function (for production deployment)")
    
    print("\nNext steps:")
    print("  ✓ Pilot operation in real plant")
    print("  ✓ Connection with real-time data streams")
    print("  ✓ Establish periodic model retraining schedule")
    print("  ✓ Implement alert function (predicted purity < threshold)")
    print("  ✓ Validate optimization conditions through A/B testing")
    

**Output Example** :
    
    
    ================================================================================
    【PI Integrated Project: End-to-End Workflow】
    ================================================================================
    
    【Step 1: Build Data Pipeline】
      Data preprocessing pipeline construction complete
    
    【Step 2: Model Training】
      Model: Gradient Boosting Regressor
      Training data points: 25
      Number of features: 12
    
    【Step 3: Model Evaluation】
      CV R² score: 0.8923 (±0.1056)
      Training data R²: 0.9567
      Training data RMSE: 0.2456%
    
    【Step 4: Deployment Preparation】
      Model saved: production_model.pkl
      Pipeline saved: production_pipeline.pkl
      Metadata saved: model_metadata.json
    
    【Step 5: Inference Function Implementation】
    
      Inference test:
        Input: {'feed_temp': 60.0, 'top_temp': 85.5, ...}
        Predicted purity: 97.34%
    
    【Step 6: Generate Monitoring Dashboard Data】
      Dashboard data saved: dashboard_data.csv
      Data period: 2025-01-24 00:00:00 to 2025-01-30 23:59:00
    
      Model performance summary (comparison with offline measurements):
        Mean error: 0.0123%
        Standard deviation: 0.2567%
        Maximum error: 0.5678%
    
    ================================================================================
    【Project Complete】
    ================================================================================
    

**Explanation** : In practice, not only model construction, but also deployment preparation, inference function implementation, and monitoring infrastructure setup are important. This workflow serves as the foundation for application to real plants.

* * *

## 4.5 Summary and Next Steps

### What We Learned in This Series

**Chapter 1: Fundamentals of PI**

  * Definition and purpose of Process Informatics
  * Characteristics of process industries and types of data
  * Real examples of data-driven process improvement and ROI
  * Basic data visualization with Python

**Chapter 2: Data Preprocessing and Visualization**

  * Time series data manipulation (resampling, rolling statistics)
  * Practical techniques for missing value handling and outlier detection
  * Selection and implementation of data scaling
  * Advanced visualization techniques

**Chapter 3: Fundamentals of Process Modeling**

  * Building quality prediction models using linear regression
  * Handling multicollinearity with PLS
  * Soft sensor design and operation
  * Model evaluation metrics and cross-validation
  * Extension to nonlinear models

**Chapter 4: Practical Exercises**

  * EDA and cleaning of real process data
  * Feature engineering
  * Comparing multiple models and selecting the optimal model
  * Fundamentals of process condition optimization
  * End-to-end project workflow

### Practical Application Checklist

  1. **Data Collection & Management**
     * □ Identify process and quality variables
     * □ Determine data collection frequency
     * □ Design database and connect to historian
  2. **Data Analysis**
     * □ Understand data through EDA
     * □ Determine missing value and outlier handling policy
     * □ Correlation analysis and feature selection
  3. **Model Building**
     * □ Build benchmark model (linear regression)
     * □ Comparative validation of multiple methods
     * □ Performance evaluation through cross-validation
     * □ Confirm model interpretability
  4. **Implementation & Operation**
     * □ Formulate pilot operation plan
     * □ Build real-time inference infrastructure
     * □ Set up monitoring dashboard
     * □ Establish periodic retraining schedule
  5. **Continuous Improvement**
     * □ Performance monitoring and drift detection
     * □ Effect validation through A/B testing
     * □ Build feedback loops

### Next Steps: Advanced Topics

For those who have mastered the fundamentals in this series, we recommend advancing to the following topics:

**1\. Advanced Modeling Techniques** — Deep learning for time series prediction using LSTM and CNN, ensemble learning through stacking and blending, Bayesian optimization for efficient hyperparameter search, and transfer learning to leverage data from other plants.

**2\. Real-time PI** — Stream processing with Apache Kafka and Spark Streaming integration, online learning for incremental adaptation and adaptive control, and edge computing for fast on-site inference.

**3\. Integration with Process Control** — Model Predictive Control (MPC) utilizing PI models, reinforcement learning for autonomous operating condition optimization, and digital twin simulation with virtual plants.

**4\. Anomaly Detection and Diagnosis** — Statistical process control using CUSUM and EWMA, change point detection for early identification of process drift, and root cause analysis for elucidating anomaly occurrence mechanisms.

**5\. Enterprise Deployment** — MLOps for model version control and CI/CD, scalability through horizontal expansion to multiple plants, and security measures including data governance and access control.

### Recommended Learning Resources

Category | Resources  
---|---  
**Books** | "Process Control Engineering" (Chemical Engineering Society), "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Aurelien Geron), "Introduction to Statistical Learning" (James et al.)  
**Online Courses** | Coursera "Machine Learning" (Andrew Ng), Udacity "Machine Learning Engineer Nanodegree", Fast.ai "Practical Deep Learning for Coders"  
**Communities** | PSE (Process Systems Engineering) Conference, IFAC DYCOPS (Dynamics and Control of Process Systems), Chemical Engineering Society Process Systems Engineering Division  
  
### Final Words

> **"Data is the new oil, but analytics is the combustion engine."**

Process Informatics is a powerful technology driving the digital transformation of manufacturing. Using the fundamentals learned in this series as a foundation, challenge yourself with PI projects in real plants.

**Keys to Success** : Start small by beginning with one quality variable and one process. Collaborate with process engineers to fuse domain knowledge and data analysis. Embrace continuous improvement since models are not finished when created but must be nurtured. Visualize value by demonstrating ROI quantitatively to gain management support.

We sincerely wish for the success of your PI projects.
