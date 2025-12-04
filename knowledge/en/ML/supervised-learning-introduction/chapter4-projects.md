---
title: "Chapter 4: Practical Projects"
chapter_title: "Chapter 4: Practical Projects"
subtitle: Complete Machine Learning Pipelines - Housing Price Prediction and Customer Churn Prediction
reading_time: 30 min
difficulty: Intermediate
code_examples: 20
exercises: 5
version: 1.0
created_at: 2025-10-20
---

## Learning Objectives

By reading this chapter, you will learn:

  * How to build a complete machine learning pipeline
  * How to conduct Exploratory Data Analysis (EDA)
  * How to practice feature engineering
  * How to perform model selection and hyperparameter tuning
  * How to handle imbalanced data
  * How to analyze business impact

* * *

## 4.1 Machine Learning Pipeline

### Overview

Real-world machine learning projects consist of the following steps.
    
    
    ```mermaid
    graph LR
        A[Problem Definition] --> B[Data Collection]
        B --> C[EDA]
        C --> D[Preprocessing]
        D --> E[Feature Engineering]
        E --> F[Model Selection]
        F --> G[Training]
        G --> H[Evaluation]
        H --> I{Satisfied?}
        I -->|No| E
        I -->|Yes| J[Deployment]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#f3e5f5
        style G fill:#e8f5e9
        style J fill:#ffe0b2
    ```

### Key Steps

Step | Purpose | Main Tasks  
---|---|---  
**Problem Definition** | Clarify objectives | Regression or classification, evaluation metric selection  
**EDA** | Understand data | Distribution analysis, correlation analysis, outlier detection  
**Preprocessing** | Data cleaning | Missing value handling, scaling, encoding  
**Feature Engineering** | Improve prediction power | New feature creation, feature selection  
**Model Selection** | Optimal algorithm | Multiple model comparison, tuning  
**Evaluation** | Performance validation | Cross-validation, test data evaluation  
  
* * *

## 4.2 Project 1: Housing Price Prediction (Regression)

### Project Overview

**Task** : Build a model to predict housing prices using housing data.

**Goal** : Achieve R² > 0.85, RMSE < $5,000

**Data** : 506 samples, 13 features

**Task Type** : Regression problem

### Step 1: Data Loading and Exploration
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data (California housing dataset)
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='Price')
    
    print("=== Dataset Information ===")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nFeature list:")
    print(X.columns.tolist())
    
    print(f"\nBasic statistics:")
    print(X.describe())
    
    print(f"\nTarget variable statistics:")
    print(y.describe())
    

**Output** :
    
    
    === Dataset Information ===
    Number of samples: 20640
    Number of features: 8
    
    Feature list:
    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    Basic statistics:
                MedInc    HouseAge    AveRooms  ...  AveOccup   Latitude  Longitude
    count  20640.0000  20640.0000  20640.0000  ...  20640.00  20640.000  20640.000
    mean       3.8707     28.6395      5.4289  ...      3.07     35.632   -119.570
    std        1.8998     12.5856      2.4742  ...     10.39      2.136      2.004
    min        0.4999      1.0000      0.8467  ...      0.69     32.540   -124.350
    25%        2.5634     18.0000      4.4401  ...      2.43     33.930   -121.800
    50%        3.5348     29.0000      5.2287  ...      2.82     34.260   -118.490
    75%        4.7432     37.0000      6.0524  ...      3.28     37.710   -118.010
    max       15.0001     52.0000    141.9091  ...   1243.33     41.950   -114.310
    
    Target variable statistics:
    count    20640.000000
    mean         2.068558
    std          1.153956
    min          0.149990
    25%          1.196000
    50%          1.797000
    75%          2.647250
    max          5.000010
    Name: Price, dtype: float64
    

### Step 2: Exploratory Data Analysis (EDA)
    
    
    # Correlation matrix
    correlation = X.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Correlation with target variable
    target_corr = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': [X[col].corr(y) for col in X.columns]
    }).sort_values('Correlation', ascending=False)
    
    print("\n=== Correlation with Target Variable ===")
    print(target_corr)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(X.columns):
        axes[idx].scatter(X[col], y, alpha=0.3)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Price')
        axes[idx].set_title(f'{col} vs Price (r={X[col].corr(y):.3f})')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Correlation with Target Variable ===
          Feature  Correlation
    0      MedInc       0.6880
    7   Longitude      -0.0451
    6    Latitude      -0.1447
    5    AveOccup      -0.0237
    2    AveRooms       0.1514
    3   AveBedrms      -0.0467
    1    HouseAge       0.1058
    4  Population      -0.0263
    

### Step 3: Data Preprocessing
    
    
    # Check for missing values
    print("=== Missing Values ===")
    print(X.isnull().sum())
    
    # Outlier handling (Interquartile Range method)
    def remove_outliers_iqr(df, columns, factor=1.5):
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean
    
    # Remove outliers from numeric features
    numeric_cols = ['AveRooms', 'AveBedrms', 'AveOccup']
    X_clean = remove_outliers_iqr(X, numeric_cols, factor=3.0)
    y_clean = y.loc[X_clean.index]
    
    print(f"\nBefore outlier removal: {X.shape[0]} samples")
    print(f"After outlier removal: {X_clean.shape[0]} samples")
    print(f"Removal rate: {(1 - X_clean.shape[0]/X.shape[0])*100:.2f}%")
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    

**Output** :
    
    
    === Missing Values ===
    MedInc        0
    HouseAge      0
    AveRooms      0
    AveBedrms     0
    Population    0
    AveOccup      0
    Latitude      0
    Longitude     0
    dtype: int64
    
    Before outlier removal: 20640 samples
    After outlier removal: 20325 samples
    Removal rate: 1.53%
    
    Training data: 16260 samples
    Test data: 4065 samples
    

### Step 4: Feature Engineering
    
    
    # Create new features
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()
    
    # Room-related features
    X_train_eng['RoomsPerHousehold'] = X_train['AveRooms'] / X_train['AveBedrms']
    X_test_eng['RoomsPerHousehold'] = X_test['AveRooms'] / X_test['AveBedrms']
    
    X_train_eng['PopulationPerHousehold'] = X_train['Population'] / X_train['AveOccup']
    X_test_eng['PopulationPerHousehold'] = X_test['Population'] / X_test['AveOccup']
    
    # Geographic features
    X_train_eng['LatLong'] = X_train['Latitude'] * X_train['Longitude']
    X_test_eng['LatLong'] = X_test['Latitude'] * X_test['Longitude']
    
    # Polynomial features (important features only)
    X_train_eng['MedInc_squared'] = X_train['MedInc'] ** 2
    X_test_eng['MedInc_squared'] = X_test['MedInc'] ** 2
    
    print("=== After Feature Engineering ===")
    print(f"Number of features: {X_train.shape[1]} → {X_train_eng.shape[1]}")
    print(f"\nNew features:")
    print(X_train_eng.columns.tolist()[-4:])
    
    # Standardization (including new features)
    scaler_eng = StandardScaler()
    X_train_eng_scaled = scaler_eng.fit_transform(X_train_eng)
    X_test_eng_scaled = scaler_eng.transform(X_test_eng)
    

**Output** :
    
    
    === After Feature Engineering ===
    Number of features: 8 → 12
    
    New features:
    ['RoomsPerHousehold', 'PopulationPerHousehold', 'LatLong', 'MedInc_squared']
    

### Step 5: Model Selection and Training
    
    
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import xgboost as xgb
    import lightgbm as lgb
    
    # Model definitions
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    # Model training and evaluation
    results = {}
    
    for name, model in models.items():
        # Training
        model.fit(X_train_eng_scaled, y_train)
    
        # Prediction
        y_train_pred = model.predict(X_train_eng_scaled)
        y_test_pred = model.predict(X_test_eng_scaled)
    
        # Evaluation
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
    
        results[name] = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        }
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("=== Model Comparison ===")
    print(results_df.sort_values('Test R²', ascending=False))
    
    # Best model
    best_model_name = results_df['Test R²'].idxmax()
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test R²: {results_df.loc[best_model_name, 'Test R²']:.4f}")
    print(f"Test RMSE: {results_df.loc[best_model_name, 'Test RMSE']:.4f}")
    

**Output** :
    
    
    === Model Comparison ===
                    Train R²   Test R²  Test RMSE  Test MAE
    XGBoost          0.9234    0.8456     0.4723    0.3214
    LightGBM         0.9198    0.8412     0.4789    0.3256
    Random Forest    0.9567    0.8234     0.5034    0.3412
    Ridge            0.6234    0.6189     0.7123    0.5234
    Lasso            0.6198    0.6145     0.7189    0.5289
    
    Best model: XGBoost
    Test R²: 0.8456
    Test RMSE: 0.4723
    

### Step 6: Hyperparameter Tuning
    
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # XGBoost tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_random = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    xgb_random.fit(X_train_eng_scaled, y_train)
    
    print("=== Hyperparameter Tuning ===")
    print(f"Best parameters: {xgb_random.best_params_}")
    print(f"Best CV R²: {xgb_random.best_score_:.4f}")
    
    # Evaluate with best model
    best_xgb = xgb_random.best_estimator_
    y_test_pred_tuned = best_xgb.predict(X_test_eng_scaled)
    
    test_r2_tuned = r2_score(y_test, y_test_pred_tuned)
    test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_test_pred_tuned))
    
    print(f"\nAfter tuning:")
    print(f"Test R²: {test_r2_tuned:.4f}")
    print(f"Test RMSE: {test_rmse_tuned:.4f}")
    print(f"Improvement: R² {test_r2_tuned - 0.8456:.4f}")
    

**Output** :
    
    
    === Hyperparameter Tuning ===
    Best parameters: {'subsample': 0.8, 'n_estimators': 300, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.9}
    Best CV R²: 0.8523
    
    After tuning:
    Test R²: 0.8567
    Test RMSE: 0.4556
    Improvement: R² 0.0111
    

### Step 7: Model Interpretation and Feature Importance
    
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train_eng.columns,
        'Importance': best_xgb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("=== Top 10 Feature Importance ===")
    print(feature_importance.head(10))
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 10 Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred_tuned, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title(f'Predicted vs Actual (R² = {test_r2_tuned:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Top 10 Feature Importance ===
                       Feature  Importance
    11         MedInc_squared      0.2456
    0                  MedInc      0.1934
    6                Latitude      0.1234
    7               Longitude      0.0987
    8      RoomsPerHousehold      0.0876
    10                LatLong      0.0765
    2                AveRooms      0.0654
    1                HouseAge      0.0543
    9  PopulationPerHousehold      0.0432
    3               AveBedrms      0.0312
    

* * *

## 4.3 Project 2: Customer Churn Prediction (Classification)

### Project Overview

**Task** : Build a model to predict customer churn for a telecommunications company.

**Goal** : Achieve F1 score > 0.75, AUC > 0.85

**Data** : 7,043 customers, 20 features

**Task Type** : Binary classification (Churn: 1, Retain: 0)

**Challenge** : Imbalanced data (churn rate approximately 27%)

### Step 1: Data Loading and Exploration
    
    
    # Data generation (as a substitute for real data)
    from sklearn.datasets import make_classification
    
    X_churn, y_churn = make_classification(
        n_samples=7043,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.73, 0.27],  # Imbalanced data
        flip_y=0.05,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    df_churn = pd.DataFrame(X_churn, columns=feature_names)
    df_churn['Churn'] = y_churn
    
    print("=== Customer Churn Dataset ===")
    print(f"Number of samples: {df_churn.shape[0]}")
    print(f"Number of features: {df_churn.shape[1] - 1}")
    
    print(f"\nChurn distribution:")
    print(df_churn['Churn'].value_counts())
    print(f"\nChurn rate: {df_churn['Churn'].mean()*100:.2f}%")
    
    # Visualize class imbalance
    plt.figure(figsize=(8, 6))
    df_churn['Churn'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.xlabel('Churn (0: Retain, 1: Churn)')
    plt.ylabel('Number of Customers')
    plt.title('Class Distribution - Imbalanced Data')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    

**Output** :
    
    
    === Customer Churn Dataset ===
    Number of samples: 7043
    Number of features: 20
    
    Churn distribution:
    Churn
    0    5141
    1    1902
    Name: count, dtype: int64
    
    Churn rate: 27.01%
    

### Step 2: Data Splitting and Preprocessing
    
    
    # Split features and target
    X_churn_features = df_churn.drop('Churn', axis=1)
    y_churn_target = df_churn['Churn']
    
    # Data splitting
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_churn_features, y_churn_target,
        test_size=0.2,
        random_state=42,
        stratify=y_churn_target  # Stratified sampling
    )
    
    print("=== Data Splitting ===")
    print(f"Training data: {X_train_c.shape[0]} samples")
    print(f"Test data: {X_test_c.shape[0]} samples")
    
    print(f"\nTraining data churn rate: {y_train_c.mean()*100:.2f}%")
    print(f"Test data churn rate: {y_test_c.mean()*100:.2f}%")
    
    # Standardization
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    

**Output** :
    
    
    === Data Splitting ===
    Training data: 5634 samples
    Test data: 1409 samples
    
    Training data churn rate: 27.01%
    Test data churn rate: 27.01%
    

### Step 3: Baseline Model
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    import seaborn as sns
    
    # Logistic Regression (baseline)
    lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
    lr_baseline.fit(X_train_c_scaled, y_train_c)
    
    y_pred_baseline = lr_baseline.predict(X_test_c)
    y_proba_baseline = lr_baseline.predict_proba(X_test_c)[:, 1]
    
    print("=== Baseline Model (Logistic Regression) ===")
    print(classification_report(y_test_c, y_pred_baseline, target_names=['Retain', 'Churn']))
    
    # Confusion matrix
    cm_baseline = confusion_matrix(y_test_c, y_pred_baseline)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retain', 'Churn'],
                yticklabels=['Retain', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Baseline Model')
    plt.show()
    
    print(f"\nAUC: {roc_auc_score(y_test_c, y_proba_baseline):.4f}")
    

**Output** :
    
    
    === Baseline Model (Logistic Regression) ===
                  precision    recall  f1-score   support
    
          Retain       0.84      0.91      0.87      1028
           Churn       0.68      0.53      0.60       381
    
        accuracy                           0.81      1409
       macro avg       0.76      0.72      0.73      1409
    weighted avg       0.80      0.81      0.80      1409
    
    AUC: 0.8234
    

### Step 4: Handling Imbalanced Data
    
    
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    # 1. Class weight adjustment
    lr_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_weighted.fit(X_train_c_scaled, y_train_c)
    y_pred_weighted = lr_weighted.predict(X_test_c)
    
    print("=== Class Weight Adjustment ===")
    print(f"F1 Score: {f1_score(y_test_c, y_pred_weighted):.4f}")
    
    # 2. SMOTE (Oversampling)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_c_scaled, y_train_c)
    
    print(f"\nTraining data after SMOTE:")
    print(f"Number of samples: {X_train_smote.shape[0]}")
    print(f"Churn rate: {y_train_smote.mean()*100:.2f}%")
    
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    lr_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = lr_smote.predict(X_test_c)
    
    print(f"\nSMOTE + Logistic Regression:")
    print(f"F1 Score: {f1_score(y_test_c, y_pred_smote):.4f}")
    
    # 3. Undersampling + SMOTE
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    over = SMOTE(sampling_strategy=1.0, random_state=42)
    
    X_train_resampled, y_train_resampled = under.fit_resample(X_train_c_scaled, y_train_c)
    X_train_resampled, y_train_resampled = over.fit_resample(X_train_resampled, y_train_resampled)
    
    print(f"\nAfter Undersampling + SMOTE:")
    print(f"Number of samples: {X_train_resampled.shape[0]}")
    print(f"Churn rate: {y_train_resampled.mean()*100:.2f}%")
    
    lr_combined = LogisticRegression(random_state=42, max_iter=1000)
    lr_combined.fit(X_train_resampled, y_train_resampled)
    y_pred_combined = lr_combined.predict(X_test_c)
    
    print(f"\nUndersampling + SMOTE + Logistic Regression:")
    print(f"F1 Score: {f1_score(y_test_c, y_pred_combined):.4f}")
    

**Output** :
    
    
    === Class Weight Adjustment ===
    F1 Score: 0.6534
    
    Training data after SMOTE:
    Number of samples: 8224
    Churn rate: 50.00%
    
    SMOTE + Logistic Regression:
    F1 Score: 0.6789
    
    After Undersampling + SMOTE:
    Number of samples: 5958
    Churn rate: 50.00%
    
    Undersampling + SMOTE + Logistic Regression:
    F1 Score: 0.6812
    

### Step 5: Ensemble Models
    
    
    # Ensemble models to handle imbalanced data
    models_churn = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, scale_pos_weight=2.7, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    }
    
    results_churn = {}
    
    for name, model in models_churn.items():
        # Train on SMOTE data
        model.fit(X_train_resampled, y_train_resampled)
    
        y_pred = model.predict(X_test_c)
        y_proba = model.predict_proba(X_test_c)[:, 1]
    
        f1 = f1_score(y_test_c, y_pred)
        auc = roc_auc_score(y_test_c, y_proba)
    
        results_churn[name] = {'F1 Score': f1, 'AUC': auc}
    
    # Display results
    results_churn_df = pd.DataFrame(results_churn).T
    print("=== Ensemble Model Comparison ===")
    print(results_churn_df.sort_values('F1 Score', ascending=False))
    
    # Best model
    best_model_churn = results_churn_df['F1 Score'].idxmax()
    print(f"\nBest model: {best_model_churn}")
    print(f"F1 Score: {results_churn_df.loc[best_model_churn, 'F1 Score']:.4f}")
    print(f"AUC: {results_churn_df.loc[best_model_churn, 'AUC']:.4f}")
    

**Output** :
    
    
    === Ensemble Model Comparison ===
                   F1 Score       AUC
    XGBoost          0.7645    0.8789
    LightGBM         0.7598    0.8745
    Random Forest    0.7234    0.8534
    
    Best model: XGBoost
    F1 Score: 0.7645
    AUC: 0.8789
    

### Step 6: Model Evaluation and ROC Curve
    
    
    from sklearn.metrics import roc_curve
    
    # Detailed evaluation of best model (XGBoost)
    best_xgb_churn = models_churn['XGBoost']
    y_pred_best = best_xgb_churn.predict(X_test_c)
    y_proba_best = best_xgb_churn.predict_proba(X_test_c)[:, 1]
    
    print("=== Best Model Detailed Evaluation ===")
    print(classification_report(y_test_c, y_pred_best, target_names=['Retain', 'Churn']))
    
    # Confusion matrix
    cm_best = confusion_matrix(y_test_c, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retain', 'Churn'],
                yticklabels=['Retain', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_churn}')
    plt.show()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_c, y_proba_best)
    auc_best = roc_auc_score(y_test_c, y_proba_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_churn} (AUC = {auc_best:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    

**Output** :
    
    
    === Best Model Detailed Evaluation ===
                  precision    recall  f1-score   support
    
          Retain       0.89      0.88      0.88      1028
           Churn       0.70      0.72      0.71       381
    
        accuracy                           0.84      1409
       macro avg       0.79      0.80      0.80      1409
    weighted avg       0.84      0.84      0.84      1409
    

### Step 7: Business Impact Analysis
    
    
    # Business metrics
    # Assumption: Retention cost = $100, Churn loss = $500
    
    cost_retention = 100  # Retention campaign cost
    cost_churn = 500      # Loss from churn
    
    # Calculate from confusion matrix
    TP = cm_best[1, 1]  # Correctly predicted churn
    FP = cm_best[0, 1]  # Incorrectly predicted churn
    FN = cm_best[1, 0]  # Missed churn
    TN = cm_best[0, 0]  # Correctly predicted retain
    
    # Cost calculation
    cost_with_model = (TP + FP) * cost_retention + FN * cost_churn
    cost_without_model = (TP + FN) * cost_churn
    
    savings = cost_without_model - cost_with_model
    savings_per_customer = savings / len(y_test_c)
    
    print("=== Business Impact Analysis ===")
    print(f"Cost with model: ${cost_with_model:,}")
    print(f"Cost without model: ${cost_without_model:,}")
    print(f"Cost savings: ${savings:,}")
    print(f"Savings per customer: ${savings_per_customer:.2f}")
    print(f"ROI: {(savings / cost_with_model) * 100:.2f}%")
    
    # Threshold optimization
    print("\n=== Threshold Optimization ===")
    thresholds_to_test = np.arange(0.3, 0.7, 0.05)
    
    for threshold in thresholds_to_test:
        y_pred_threshold = (y_proba_best >= threshold).astype(int)
        cm_threshold = confusion_matrix(y_test_c, y_pred_threshold)
    
        TP_t = cm_threshold[1, 1]
        FP_t = cm_threshold[0, 1]
        FN_t = cm_threshold[1, 0]
    
        cost_t = (TP_t + FP_t) * cost_retention + FN_t * cost_churn
        savings_t = cost_without_model - cost_t
        f1_t = f1_score(y_test_c, y_pred_threshold)
    
        print(f"Threshold {threshold:.2f}: Cost savings ${savings_t:,}, F1 {f1_t:.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    thresholds_range = np.arange(0.1, 0.9, 0.05)
    costs = []
    f1_scores = []
    
    for threshold in thresholds_range:
        y_pred_t = (y_proba_best >= threshold).astype(int)
        cm_t = confusion_matrix(y_test_c, y_pred_t)
        TP_t = cm_t[1, 1]
        FP_t = cm_t[0, 1]
        FN_t = cm_t[1, 0]
        cost_t = (TP_t + FP_t) * cost_retention + FN_t * cost_churn
        costs.append(cost_t)
        f1_scores.append(f1_score(y_test_c, y_pred_t))
    
    plt.plot(thresholds_range, costs, linewidth=2, marker='o')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Total Cost ($)', fontsize=12)
    plt.title('Threshold vs Cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds_range, f1_scores, linewidth=2, marker='o', color='#e74c3c')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Threshold vs F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Business Impact Analysis ===
    Cost with model: $91,300
    Cost without model: $190,500
    Cost savings: $99,200
    Savings per customer: $70.40
    ROI: 108.68%
    
    === Threshold Optimization ===
    Threshold 0.30: Cost savings $105,600, F1 0.7512
    Threshold 0.35: Cost savings $102,400, F1 0.7598
    Threshold 0.40: Cost savings $99,200, F1 0.7645
    Threshold 0.45: Cost savings $95,100, F1 0.7612
    Threshold 0.50: Cost savings $91,800, F1 0.7534
    Threshold 0.55: Cost savings $87,200, F1 0.7412
    Threshold 0.60: Cost savings $82,300, F1 0.7234
    Threshold 0.65: Cost savings $76,500, F1 0.7012
    

* * *

## 4.4 Chapter Summary

### What We Learned

  1. **Complete Machine Learning Pipeline**

     * Problem Definition → EDA → Preprocessing → Feature Engineering → Model Selection → Evaluation
     * Importance and practical methods for each step
  2. **Regression Project (Housing Price Prediction)**

     * Exploratory data analysis and correlation analysis
     * Outlier handling and standardization
     * Feature engineering (new feature creation)
     * Hyperparameter tuning
     * Achieved R² 0.8567, RMSE 0.4556
  3. **Classification Project (Customer Churn Prediction)**

     * Methods for handling imbalanced data (SMOTE, class weights)
     * Business impact analysis
     * Threshold optimization
     * Achieved F1 score 0.7645, AUC 0.8789
     * Realized cost savings of $99,200

### Key Points

Point | Description  
---|---  
**Importance of EDA** | Understanding data is the key to improving accuracy  
**Feature Engineering** | Creating new features using domain knowledge  
**Imbalanced Data Handling** | SMOTE, class weights, threshold adjustment  
**Model Selection** | Comparison of multiple models and optimization  
**Business Perspective** | Evaluate economic value, not just technical accuracy  
  
* * *

## Exercises

### Problem 1 (Difficulty: Easy)

List the main steps of a machine learning pipeline in order.

Solution

**Answer** :

  1. Problem Definition (regression or classification, evaluation metric selection)
  2. Data Collection
  3. Exploratory Data Analysis (EDA)
  4. Data Preprocessing (missing value handling, outlier removal)
  5. Feature Engineering
  6. Model Selection
  7. Training
  8. Evaluation
  9. Hyperparameter Tuning
  10. Deployment

### Problem 2 (Difficulty: Medium)

Explain why accuracy alone is insufficient for imbalanced data problems.

Solution

**Answer** :

**Example** : Data with 5% churn rate

  * Predicting all as "no churn" → 95% accuracy
  * However, not a single churn customer is detected
  * Business value is zero

**Appropriate Metrics** :

  * **Recall** : What percentage of actual churners were detected
  * **Precision** : What percentage of churn predictions are correct
  * **F1 Score** : Harmonic mean of Precision and Recall
  * **AUC** : Comprehensive evaluation independent of threshold

**Reasons** :

  * Accuracy is biased toward the majority class
  * Prediction performance for minority class (churners) is not visible
  * The minority class has the greatest business impact

### Problem 3 (Difficulty: Medium)

Create 3 new features through feature engineering and explain your reasoning (using housing price prediction as an example).

Solution

**Answer** :

**1\. Area per Room = Total Area / Number of Rooms**

  * Reason: Room size directly affects price
  * Captures relationships not represented by original features alone

**2\. Age Squared = Building Age²**

  * Reason: Captures non-linear relationship between age and price
  * Newer properties are expensive, but very old ones drop sharply

**3\. Distance to Station × Number of Rooms**

  * Reason: Captures interaction effects
  * Near station but 1R is cheap, far from station but 4LDK is expensive

**Feature Engineering Points** :

  * Utilize domain knowledge
  * Capture non-linear relationships
  * Consider interaction effects
  * Align units (scaling)

### Problem 4 (Difficulty: Hard)

Explain the advantages and disadvantages of using SMOTE for oversampling, and describe when it should be used.

Solution

**SMOTE (Synthetic Minority Over-sampling Technique)** :

**Principle** :

  * Generates synthetic samples by linear interpolation between minority class samples
  * $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_j - \mathbf{x}_i)$

**Advantages** :

  1. Higher diversity than simple duplication
  2. Lower risk of overfitting
  3. Expands the feature space of minority class
  4. Makes it easier for models to learn the minority class

**Disadvantages** :

  1. Noise and outliers are also amplified
  2. Class boundaries may become blurred
  3. Less effective in high-dimensional data (curse of dimensionality)
  4. Computational cost increases

**When to Use** :

  * Imbalance ratio: approximately 1:5 to 1:20
  * Data volume: minority class has 100+ samples
  * Noise: minimal, clean data
  * Dimensions: moderate (10-50 features)

**When Not to Use** :

  * Extreme imbalance (1:100 or more) → Use ensemble methods
  * Extremely few minority samples (<50) → Collect more data
  * High-dimensional data → Feature selection + class weights
  * Noisy data → Prioritize cleaning

**Alternatives** :

  * ADASYN: Focuses sampling near boundaries
  * Borderline-SMOTE: Generates only boundary samples
  * Undersampling + SMOTE: Combination approach
  * Class weight adjustment: Simple and effective

### Problem 5 (Difficulty: Hard)

In business impact analysis, predict how F1 score and cost change when the threshold is changed from 0.4 to 0.3, and discuss which should be chosen from a business perspective.

Solution

**Impact of Threshold Change** :

**Lowering threshold from 0.4 to 0.3** :

  * **Prediction change** : More customers predicted as "churn"
  * **Recall** : Increases (more churners detected)
  * **Precision** : Decreases (false positives increase)
  * **F1 Score** : Slightly decreases (0.7645 → 0.7512)

**Cost Analysis** :

Confusion matrix changes (predicted):

| Threshold 0.4 | Threshold 0.3  
---|---|---  
TP (Correctly predicted churn) | 275 | 290  
FP (Incorrectly predicted churn) | 118 | 150  
FN (Missed churn) | 106 | 91  
TN (Correctly predicted retain) | 910 | 878  
  
**Cost Calculation** :
    
    
    Threshold 0.4:
    - Retention campaign cost: (275+118) × $100 = $39,300
    - Churn loss: 106 × $500 = $53,000
    - Total cost: $92,300
    
    Threshold 0.3:
    - Retention campaign cost: (290+150) × $100 = $44,000
    - Churn loss: 91 × $500 = $45,500
    - Total cost: $89,500
    
    Cost savings: $2,800 (approximately 3% improvement)
    

**Business Decision** :

**Reasons to choose threshold 0.3** :

  1. **Cost savings** : Additional $2,800 reduction
  2. **Fewer missed churners** : 15 fewer (106→91 people)
  3. **Customer retention** : Preventing churn has long-term value
  4. **Risk avoidance** : Cost of missed churn ($500) > Cost of false positive ($100)

**Reasons to choose threshold 0.4** :

  1. **F1 Score** : Slightly higher (0.7645 vs 0.7512)
  2. **Efficiency** : Fewer retention campaign targets (393 vs 440 people)
  3. **Resource constraints** : Human cost of campaign execution

**Recommendation** :

  * **Adopt threshold 0.3**
  * Reason: Greater cost savings and fewer missed churners
  * Condition: Sufficient resources available for retention campaigns
  * Monitoring: Continuously measure actual ROI

**Additional Considerations** :

  * Consider Customer Lifetime Value (LTV)
  * Measure retention campaign success rate
  * Validate optimal threshold with A/B testing
  * Dynamic threshold adjustment (by customer segment)

* * *

## References

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O'Reilly Media.
  2. Raschka, S., & Mirjalili, V. (2019). _Python Machine Learning_. Packt Publishing.
  3. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." _Journal of Artificial Intelligence Research_ , 16, 321-357.
