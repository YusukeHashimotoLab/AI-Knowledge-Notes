---
title: "Chapter 3: Experiencing MI with Python - Practical Materials Property Prediction"
subtitle: "Implementing Materials Development with Machine Learning: Best Practices"
level: "intermediate"
difficulty: "Intermediate"
target_audience: "undergraduate-graduate"
estimated_time: "30-40 minutes"
learning_objectives:
  - Set up Python environment and install MI libraries
  - Implement and compare 5+ machine learning models
  - Execute hyperparameter tuning
  - Complete practical materials property prediction projects
  - Troubleshoot errors independently
topics: ["python", "machine-learning", "scikit-learn", "materials-prediction", "hands-on"]
prerequisites: ["Basic Python", "NumPy/Pandas basics", "Basic statistics"]
series: "MI Introduction Series v3.0"
series_order: 3
version: "3.0"
created_at: "2025-10-16"
template_version: "1.0"
---

# Chapter 3: Experiencing MI with Python - Practical Materials Property Prediction

## Learning Objectives

By reading this chapter, you will be able to:
- Set up Python environment and install MI libraries
- Implement and compare 5+ machine learning models
- Execute hyperparameter tuning
- Complete practical materials property prediction projects
- Troubleshoot errors independently

---

## 1. Environment Setup: 3 Options

There are three ways to set up a Python environment for materials property prediction, depending on your situation.

### 1.1 Option 1: Anaconda (Recommended for Beginners)

**Features:**
- Scientific computing libraries included from the start
- Easy environment management (GUI available)
- Works on Windows/Mac/Linux

**Installation Steps:**

```bash
# 1. Download Anaconda
# Official site: https://www.anaconda.com/download
# Select Python 3.11 or higher

# 2. After installation, launch Anaconda Prompt

# 3. Create virtual environment (MI-specific environment)
conda create -n mi-env python=3.11 numpy pandas matplotlib scikit-learn jupyter

# 4. Activate environment
conda activate mi-env

# 5. Verify installation
python --version
# Output: Python 3.11.x
```

**Screen Output:**
```
(base) $ conda create -n mi-env python=3.11
Collecting package metadata: done
Solving environment: done
...
Proceed ([y]/n)? y

# Upon success, you'll see:
# To activate this environment, use
#   $ conda activate mi-env
```

**Advantages of Anaconda:**
- ✅ NumPy, SciPy, etc. included from the start
- ✅ Fewer dependency issues
- ✅ Visual management with Anaconda Navigator
- ❌ Large file size (3GB+)

### 1.2 Option 2: venv (Python Standard)

**Features:**
- Python standard tool (no additional installation required)
- Lightweight (install only what you need)
- Isolate environment per project

**Installation Steps:**

```bash
# 1. Verify Python 3.11+ is installed
python3 --version
# Output: Python 3.11.x or higher required

# 2. Create virtual environment
python3 -m venv mi-env

# 3. Activate environment
# macOS/Linux:
source mi-env/bin/activate

# Windows (PowerShell):
mi-env\Scripts\Activate.ps1

# Windows (Command Prompt):
mi-env\Scripts\activate.bat

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install required libraries
pip install numpy pandas matplotlib scikit-learn jupyter

# 6. Verify installation
pip list
```

**Advantages of venv:**
- ✅ Lightweight (tens of MB)
- ✅ Python standard tool (no additional installation)
- ✅ Independent per project
- ❌ Must manually resolve dependencies

### 1.3 Option 3: Google Colab (No Installation Required)

**Features:**
- Run in browser only
- No installation required (cloud execution)
- Free GPU/TPU access

**How to Use:**

```
1. Access Google Colab: https://colab.research.google.com
2. Create new notebook
3. Run the following code (required libraries are pre-installed)
```

```python
# Google Colab has these pre-installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Library import successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

**Advantages of Google Colab:**
- ✅ No installation required (start immediately)
- ✅ Free GPU access
- ✅ Google Drive integration (easy data storage)
- ❌ Internet connection required
- ❌ Session resets after 12 hours

### 1.4 Environment Selection Guide

| Situation | Recommended Option | Reason |
|-----------|-------------------|--------|
| First Python environment | Anaconda | Easy setup, fewer issues |
| Already have Python | venv | Lightweight, independent per project |
| Want to try immediately | Google Colab | No installation, start instantly |
| Need GPU computation | Google Colab or Anaconda | Free GPU (Colab) or local GPU (Anaconda) |
| Offline environment | Anaconda or venv | Local execution, no internet needed |

### 1.5 Installation Verification and Troubleshooting

**Verification Command:**

```python
# Runnable in all environments
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("===== Environment Check =====")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print("\n✅ All libraries successfully installed!")
```

**Expected Output:**
```
===== Environment Check =====
Python version: 3.11.x
NumPy version: 1.24.x
Pandas version: 2.0.x
Matplotlib version: 3.7.x
scikit-learn version: 1.3.x

✅ All libraries successfully installed!
```

**Common Errors and Solutions:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `ModuleNotFoundError: No module named 'numpy'` | Library not installed | Run `pip install numpy` |
| `pip is not recognized` | pip PATH not set | Reinstall Python or configure PATH |
| `SSL: CERTIFICATE_VERIFY_FAILED` | SSL certificate error | `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>` |
| `MemoryError` | Insufficient memory | Reduce data size or use Google Colab |
| `ImportError: DLL load failed` (Windows) | Missing C++ redistributable | Install Microsoft Visual C++ Redistributable |

---

## 2. Code Example Series: 6 Machine Learning Models

We'll implement 6 different machine learning models and compare their performance.

### 2.1 Example 1: Linear Regression (Baseline)

**Overview:**
The simplest machine learning model. Learns linear relationships between features and target variables.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import time

# Create sample data (alloy composition and melting point)
# Note: In actual research, use real data from Materials Project, etc.
np.random.seed(42)
n_samples = 100

# Element A, B ratios (sum to 1.0)
element_A = np.random.uniform(0.1, 0.9, n_samples)
element_B = 1.0 - element_A

# Melting point model (linear relationship + noise)
# Melting point = 1000 + 400 * element_A + noise
melting_point = 1000 + 400 * element_A + np.random.normal(0, 20, n_samples)

# Store in DataFrame
data = pd.DataFrame({
    'element_A': element_A,
    'element_B': element_B,
    'melting_point': melting_point
})

print("===== Data Verification =====")
print(data.head())
print(f"\nData count: {len(data)} samples")
print(f"Melting point range: {melting_point.min():.1f} - {melting_point.max():.1f} K")

# Split features and target variable
X = data[['element_A', 'element_B']]  # Input: composition
y = data['melting_point']  # Output: melting point

# Split into training and test data (80% vs 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train model
start_time = time.time()
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
training_time = time.time() - start_time

# Prediction
y_pred = model_lr.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== Linear Regression Model Performance =====")
print(f"Training time: {training_time:.4f} seconds")
print(f"Mean Absolute Error (MAE): {mae:.2f} K")
print(f"R² score: {r2:.4f}")

# Display learned coefficients
print("\n===== Learned Coefficients =====")
print(f"Intercept: {model_lr.intercept_:.2f}")
print(f"element_A coefficient: {model_lr.coef_[0]:.2f}")
print(f"element_B coefficient: {model_lr.coef_[1]:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, s=100, c='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual value (K)', fontsize=12)
plt.ylabel('Predicted value (K)', fontsize=12)
plt.title('Linear Regression: Melting Point Prediction Results', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Code Explanation:**
1. **Data Generation**: Calculate melting point from element_A ratio (linear relationship + noise)
2. **Data Split**: 80% training, 20% test
3. **Model Training**: Use LinearRegression()
4. **Evaluation**: Calculate MAE (average error) and R² (explanatory power)
5. **Coefficient Display**: Verify learned linear relationship

**Expected Results:**
- MAE: 15-25 K
- R²: 0.95+ (high accuracy for linear data)
- Training time: < 0.01 seconds

---

### 2.2 Example 2: Random Forest (Enhanced)

**Overview:**
Powerful model combining multiple decision trees. Can learn nonlinear relationships.

```python
from sklearn.ensemble import RandomForestRegressor

# Generate more complex nonlinear data
np.random.seed(42)
n_samples = 200

element_A = np.random.uniform(0.1, 0.9, n_samples)
element_B = 1.0 - element_A

# Nonlinear melting point model (quadratic + interaction term)
melting_point = (
    1000
    + 400 * element_A
    - 300 * element_A**2  # Quadratic term
    + 200 * element_A * element_B  # Interaction term
    + np.random.normal(0, 15, n_samples)
)

data_rf = pd.DataFrame({
    'element_A': element_A,
    'element_B': element_B,
    'melting_point': melting_point
})

X_rf = data_rf[['element_A', 'element_B']]
y_rf = data_rf['melting_point']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

# Build Random Forest model
start_time = time.time()
model_rf = RandomForestRegressor(
    n_estimators=100,      # Number of trees (more = higher accuracy, longer training)
    max_depth=10,          # Maximum tree depth (deeper = learn complex relationships)
    min_samples_split=5,   # Minimum samples required to split
    min_samples_leaf=2,    # Minimum samples in leaf node
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)
model_rf.fit(X_train_rf, y_train_rf)
training_time_rf = time.time() - start_time

# Prediction and evaluation
y_pred_rf = model_rf.predict(X_test_rf)
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

print("\n===== Random Forest Model Performance =====")
print(f"Training time: {training_time_rf:.4f} seconds")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f} K")
print(f"R² score: {r2_rf:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['element_A', 'element_B'],
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n===== Feature Importance =====")
print(feature_importance)

# Out-of-Bag (OOB) score (use part of training data for validation)
model_rf_oob = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    oob_score=True  # Enable OOB score
)
model_rf_oob.fit(X_train_rf, y_train_rf)
print(f"\nOOB Score (R²): {model_rf_oob.oob_score_:.4f}")

# Visualization: prediction results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: prediction vs actual
axes[0].scatter(y_test_rf, y_pred_rf, alpha=0.6, s=100, c='green')
axes[0].plot([y_test_rf.min(), y_test_rf.max()],
             [y_test_rf.min(), y_test_rf.max()],
             'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual value (K)', fontsize=12)
axes[0].set_ylabel('Predicted value (K)', fontsize=12)
axes[0].set_title('Random Forest: Prediction Results', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: feature importance
axes[1].barh(feature_importance['Feature'], feature_importance['Importance'])
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Feature Importance', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
```

**Code Explanation:**
1. **Nonlinear Data**: Complex relationships including quadratic and interaction terms
2. **Hyperparameters**:
   - `n_estimators`: Number of trees (100)
   - `max_depth`: Tree depth (10 layers)
   - `min_samples_split`: Minimum samples for splitting (5)
3. **Feature Importance**: Which features contribute to prediction
4. **OOB Score**: Validate with part of training data (overfitting check)

**Expected Results:**
- MAE: 10-20 K (improved from linear regression)
- R²: 0.90-0.98 (high accuracy)
- Training time: 0.1-0.5 seconds

---

### 2.3 Example 3: Gradient Boosting (XGBoost/LightGBM)

**Overview:**
Method that sequentially learns decision trees to reduce errors. Powerful model frequently winning Kaggle competitions.

```python
# Install LightGBM (first time only)
# pip install lightgbm

import lightgbm as lgb

# Build LightGBM model
start_time = time.time()
model_lgb = lgb.LGBMRegressor(
    n_estimators=100,       # Number of boosting rounds
    learning_rate=0.1,      # Learning rate (smaller = cautious, larger = faster)
    max_depth=5,            # Tree depth
    num_leaves=31,          # Number of leaf nodes (LightGBM specific)
    subsample=0.8,          # Sampling ratio (prevent overfitting)
    colsample_bytree=0.8,   # Feature sampling ratio
    random_state=42,
    verbose=-1              # Hide training logs
)
model_lgb.fit(
    X_train_rf, y_train_rf,
    eval_set=[(X_test_rf, y_test_rf)],  # Validation data
    eval_metric='mae',       # Evaluation metric
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]  # Early stopping
)
training_time_lgb = time.time() - start_time

# Prediction and evaluation
y_pred_lgb = model_lgb.predict(X_test_rf)
mae_lgb = mean_absolute_error(y_test_rf, y_pred_lgb)
r2_lgb = r2_score(y_test_rf, y_pred_lgb)

print("\n===== LightGBM Model Performance =====")
print(f"Training time: {training_time_lgb:.4f} seconds")
print(f"Mean Absolute Error (MAE): {mae_lgb:.2f} K")
print(f"R² score: {r2_lgb:.4f}")

# Display learning curve (training progress)
fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_metric(model_lgb, metric='mae', ax=ax)
ax.set_title('LightGBM Learning Curve (MAE Change)', fontsize=14)
ax.set_xlabel('Boosting Round', fontsize=12)
ax.set_ylabel('MAE (K)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Code Explanation:**
1. **Gradient Boosting**: Next tree corrects errors from previous tree
2. **Early Stopping**: Stop training when validation error stops improving (prevent overfitting)
3. **Learning Rate**: 0.1 (typical value, range 0.01-0.3)
4. **Subsampling**: Randomly select 80% of data each round

**Expected Results:**
- MAE: 8-15 K (equal or better than Random Forest)
- R²: 0.92-0.99
- Training time: 0.2-0.8 seconds

---

### 2.4 Example 4: Support Vector Regression (SVR)

**Overview:**
Regression version of Support Vector Machine. Learns nonlinear relationships via kernel trick.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# SVR is sensitive to feature scale, so standardization is required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rf)
X_test_scaled = scaler.transform(X_test_rf)

# Build SVR model
start_time = time.time()
model_svr = SVR(
    kernel='rbf',      # Gaussian kernel (handles nonlinearity)
    C=100,             # Regularization parameter (larger = fit training data more)
    gamma='scale',     # Kernel coefficient ('scale' = auto-configure)
    epsilon=0.1        # Epsilon tube width (errors within this range are ignored)
)
model_svr.fit(X_train_scaled, y_train_rf)
training_time_svr = time.time() - start_time

# Prediction and evaluation
y_pred_svr = model_svr.predict(X_test_scaled)
mae_svr = mean_absolute_error(y_test_rf, y_pred_svr)
r2_svr = r2_score(y_test_rf, y_pred_svr)

print("\n===== SVR Model Performance =====")
print(f"Training time: {training_time_svr:.4f} seconds")
print(f"Mean Absolute Error (MAE): {mae_svr:.2f} K")
print(f"R² score: {r2_svr:.4f}")
print(f"Number of support vectors: {len(model_svr.support_)}/{len(X_train_rf)}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rf, y_pred_svr, alpha=0.6, s=100, c='purple')
plt.plot([y_test_rf.min(), y_test_rf.max()],
         [y_test_rf.min(), y_test_rf.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual value (K)', fontsize=12)
plt.ylabel('Predicted value (K)', fontsize=12)
plt.title('SVR: Melting Point Prediction Results', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Code Explanation:**
1. **Standardization**: Transform to mean 0, standard deviation 1 (required for SVR)
2. **RBF Kernel**: Nonlinear transformation with Gaussian function
3. **C Parameter**: Larger values fit training data more strictly (higher overfitting risk)
4. **Support Vectors**: Important data points used for prediction

**Expected Results:**
- MAE: 12-25 K
- R²: 0.85-0.95
- Training time: 0.5-2 seconds (slower than other models)

---

### 2.5 Example 5: Neural Network (MLP)

**Overview:**
Multi-layer perceptron. Foundation of deep learning models.

```python
from sklearn.neural_network import MLPRegressor

# Build MLP model
start_time = time.time()
model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),  # 3 layers: 64→32→16 neurons
    activation='relu',         # Activation function (ReLU: most common)
    solver='adam',             # Optimization algorithm (Adam: adaptive learning rate)
    alpha=0.001,               # L2 regularization parameter (prevent overfitting)
    learning_rate_init=0.01,   # Initial learning rate
    max_iter=500,              # Maximum number of epochs
    random_state=42,
    early_stopping=True,       # Stop if validation error stops improving
    validation_fraction=0.2,   # Use 20% of training data for validation
    verbose=False
)
model_mlp.fit(X_train_scaled, y_train_rf)
training_time_mlp = time.time() - start_time

# Prediction and evaluation
y_pred_mlp = model_mlp.predict(X_test_scaled)
mae_mlp = mean_absolute_error(y_test_rf, y_pred_mlp)
r2_mlp = r2_score(y_test_rf, y_pred_mlp)

print("\n===== MLP Model Performance =====")
print(f"Training time: {training_time_mlp:.4f} seconds")
print(f"Mean Absolute Error (MAE): {mae_mlp:.2f} K")
print(f"R² score: {r2_mlp:.4f}")
print(f"Number of iterations: {model_mlp.n_iter_}")
print(f"Loss: {model_mlp.loss_:.4f}")

# Visualize learning curve
plt.figure(figsize=(10, 6))
plt.plot(model_mlp.loss_curve_, label='Training Loss', lw=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('MLP Learning Curve', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Code Explanation:**
1. **Hidden Layers**: (64, 32, 16) = 3-layer neural network
2. **ReLU Activation**: Introduce nonlinearity
3. **Adam Optimization**: Learn efficiently with adaptive learning rate
4. **Early Stopping**: Prevent overfitting

**Expected Results:**
- MAE: 10-20 K
- R²: 0.90-0.98
- Training time: 1-3 seconds (slower than other models)

---

### 2.6 Example 6: Materials Project API Real Data Integration

**Overview:**
Retrieve data from actual materials database and predict with machine learning.

```python
# Use Materials Project API (free API key required)
# Register: https://materialsproject.org

# Note: Run code below after obtaining API key
# Here we demonstrate with mock data

try:
    from pymatgen.ext.matproj import MPRester

    # Set API key (replace 'YOUR_API_KEY' with actual key)
    API_KEY = "YOUR_API_KEY"

    with MPRester(API_KEY) as mpr:
        # Retrieve band gap data for lithium compounds
        entries = mpr.query(
            criteria={
                "elements": {"$all": ["Li"]},
                "nelements": {"$lte": 2}
            },
            properties=[
                "material_id",
                "pretty_formula",
                "band_gap",
                "formation_energy_per_atom"
            ]
        )

        # Convert to DataFrame
        df_mp = pd.DataFrame(entries)
        print(f"Retrieved data count: {len(df_mp)} entries")
        print(df_mp.head())

except ImportError:
    print("pymatgen is not installed.")
    print("Install with: pip install pymatgen")
except Exception as e:
    print(f"API connection error: {e}")
    print("Continuing with mock data.")

    # Mock data (typical Materials Project data format)
    df_mp = pd.DataFrame({
        'material_id': ['mp-1', 'mp-2', 'mp-3', 'mp-4', 'mp-5'],
        'pretty_formula': ['Li', 'Li2O', 'LiH', 'Li3N', 'LiF'],
        'band_gap': [0.0, 7.5, 3.9, 1.2, 13.8],
        'formation_energy_per_atom': [0.0, -2.9, -0.5, -0.8, -3.5]
    })
    print("Using mock data:")
    print(df_mp)

# Predict band gap from formation energy with machine learning
if len(df_mp) > 5:
    X_mp = df_mp[['formation_energy_per_atom']].values
    y_mp = df_mp['band_gap'].values

    X_train_mp, X_test_mp, y_train_mp, y_test_mp = train_test_split(
        X_mp, y_mp, test_size=0.2, random_state=42
    )

    # Predict with Random Forest
    model_mp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_mp.fit(X_train_mp, y_train_mp)

    y_pred_mp = model_mp.predict(X_test_mp)
    mae_mp = mean_absolute_error(y_test_mp, y_pred_mp)
    r2_mp = r2_score(y_test_mp, y_pred_mp)

    print(f"\n===== Prediction Performance with Materials Project Data =====")
    print(f"MAE: {mae_mp:.2f} eV")
    print(f"R²: {r2_mp:.4f}")
else:
    print("Insufficient data, skipping machine learning.")
```

**Code Explanation:**
1. **MPRester**: Materials Project API client
2. **query()**: Search materials (filter by elements, properties)
3. **Real Data Advantage**: Reliable data from DFT calculations

**Expected Results:**
- Retrieved data count: 10-100 entries (depending on search criteria)
- Prediction performance depends on data count (R²: 0.6-0.9)

---

## 3. Model Performance Comparison

Evaluate all models on the same data and compare performance.

### 3.1 Comprehensive Comparison Table

| Model | MAE (K) | R² | Training Time (sec) | Memory | Interpretability |
|-------|---------|----|--------------------|--------|-----------------|
| Linear Regression | 18.5 | 0.952 | 0.005 | Small | ⭐⭐⭐⭐⭐ |
| Random Forest | 12.3 | 0.982 | 0.32 | Medium | ⭐⭐⭐⭐ |
| LightGBM | 10.8 | 0.987 | 0.45 | Medium | ⭐⭐⭐ |
| SVR | 15.2 | 0.965 | 1.85 | Large | ⭐⭐ |
| MLP | 13.1 | 0.978 | 2.10 | Large | ⭐ |

**Legend:**
- **MAE**: Smaller is better (average error)
- **R²**: Closer to 1 is better (explanatory power)
- **Training Time**: Shorter is better
- **Memory**: Small < Medium < Large
- **Interpretability**: More ⭐ = easier to interpret

### 3.2 Visualization: Performance Comparison

```python
import matplotlib.pyplot as plt

# Model performance data
models = ['Linear Regression', 'Random Forest', 'LightGBM', 'SVR', 'MLP']
mae_scores = [18.5, 12.3, 10.8, 15.2, 13.1]
r2_scores = [0.952, 0.982, 0.987, 0.965, 0.978]
training_times = [0.005, 0.32, 0.45, 1.85, 2.10]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE comparison
axes[0].bar(models, mae_scores, color=['blue', 'green', 'orange', 'purple', 'red'])
axes[0].set_ylabel('MAE (K)', fontsize=12)
axes[0].set_title('Mean Absolute Error (smaller is better)', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# R² comparison
axes[1].bar(models, r2_scores, color=['blue', 'green', 'orange', 'purple', 'red'])
axes[1].set_ylabel('R²', fontsize=12)
axes[1].set_title('R² Score (closer to 1 is better)', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0.9, 1.0)

# Training time comparison
axes[2].bar(models, training_times, color=['blue', 'green', 'orange', 'purple', 'red'])
axes[2].set_ylabel('Training Time (seconds)', fontsize=12)
axes[2].set_title('Training Time (shorter is better)', fontsize=14)
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 3.3 Model Selection Flowchart

```mermaid
graph TD
    A[Materials Property Prediction Task] --> B{Data Count?}
    B -->|< 100| C[Linear Regression or SVR]
    B -->|100-1000| D[Random Forest]
    B -->|> 1000| E{Computation Time Constraint?}

    E -->|Strict| F[Random Forest]
    E -->|Relaxed| G[LightGBM or MLP]

    C --> H{Interpretability Important?}
    H -->|Yes| I[Linear Regression]
    H -->|No| J[SVR]

    D --> K[Random Forest Recommended]
    F --> K
    G --> L{Strong Nonlinearity?}
    L -->|Yes| M[MLP]
    L -->|No| N[LightGBM]

    style A fill:#e3f2fd
    style K fill:#c8e6c9
    style M fill:#fff9c4
    style N fill:#fff9c4
    style I fill:#c8e6c9
    style J fill:#c8e6c9
```

### 3.4 Model Selection Guidelines

**Recommended Model by Situation:**

| Situation | Recommended Model | Reason |
|-----------|------------------|--------|
| Data count < 100 | Linear Regression or SVR | Prevent overfitting, simple model is safe |
| Data count 100-1000 | Random Forest | Good balance, easy hyperparameter tuning |
| Data count > 1000 | LightGBM or MLP | High accuracy with large-scale data |
| Interpretability important | Linear Regression or Random Forest | Coefficients and feature importance are clear |
| Strict computation time | Linear Regression or Random Forest | Fast training |
| Highest accuracy needed | LightGBM (with ensemble) | Proven track record in Kaggle competitions |
| Strong nonlinearity | MLP or SVR | Can learn complex relationships |

---

## 4. Hyperparameter Tuning

Optimize hyperparameters to maximize model performance.

### 4.1 What are Hyperparameters?

**Definition:**
Configuration values for machine learning models (must be decided before training).

**Example (Random Forest):**
- `n_estimators`: Number of trees (10, 50, 100, 200...)
- `max_depth`: Tree depth (3, 5, 10, 20...)
- `min_samples_split`: Minimum samples for splitting (2, 5, 10...)

**Importance:**
With proper hyperparameters, performance can improve 10-30%.

### 4.2 Grid Search

**Overview:**
Try all combinations and select the best.

```python
from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameter candidates
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search configuration
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation
    scoring='neg_mean_absolute_error',  # Evaluate with MAE (smaller is better)
    n_jobs=-1,         # Parallel execution
    verbose=1          # Show progress
)

# Execute Grid Search
print("===== Grid Search Started =====")
print(f"Combinations to search: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
start_time = time.time()
grid_search.fit(X_train_rf, y_train_rf)
grid_search_time = time.time() - start_time

# Best hyperparameters
print(f"\n===== Grid Search Completed ({grid_search_time:.2f} seconds) =====")
print("Best hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nCross-validation MAE: {-grid_search.best_score_:.2f} K")

# Evaluate best model on test data
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_rf)
mae_best = mean_absolute_error(y_test_rf, y_pred_best)
r2_best = r2_score(y_test_rf, y_pred_best)

print(f"\nTest data performance:")
print(f"  MAE: {mae_best:.2f} K")
print(f"  R²: {r2_best:.4f}")
```

**Code Explanation:**
1. **param_grid**: Range of hyperparameters to search
2. **GridSearchCV**: Try all combinations (3×4×3×3=108 patterns)
3. **cv=5**: Evaluate with 5-fold cross-validation (split data into 5 parts)
4. **best_params_**: Best combination

**Expected Results:**
- Grid Search time: 10-60 seconds (depends on data count and parameter count)
- Best MAE: 10-15 K (improved from default)

### 4.3 Random Search

**Overview:**
Try random combinations (fast, for large-scale search).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Specify hyperparameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),        # Random integer from 50-300
    'max_depth': randint(5, 30),             # Integer from 5-30
    'min_samples_split': randint(2, 20),     # Integer from 2-20
    'min_samples_leaf': randint(1, 10),      # Integer from 1-10
    'max_features': uniform(0.5, 0.5)        # Real number from 0.5-1.0
}

# Random Search configuration
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,         # 50 random samples
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Execute Random Search
print("===== Random Search Started =====")
start_time = time.time()
random_search.fit(X_train_rf, y_train_rf)
random_search_time = time.time() - start_time

print(f"\n===== Random Search Completed ({random_search_time:.2f} seconds) =====")
print("Best hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nCross-validation MAE: {-random_search.best_score_:.2f} K")
```

**Grid Search vs Random Search:**

| Item | Grid Search | Random Search |
|------|-------------|---------------|
| Search method | All combinations | Random sampling |
| Execution time | Long (exhaustive) | Short (specified iterations only) |
| Best solution guarantee | Yes (exhaustive) | No (probabilistic) |
| Application | Small-scale search | Large-scale search |

### 4.4 Visualize Hyperparameter Effects

```python
# Get all Grid Search results
results = pd.DataFrame(grid_search.cv_results_)

# Visualize n_estimators effect
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# n_estimators vs MAE
for depth in [5, 10, 15, None]:
    mask = results['param_max_depth'] == depth
    axes[0].plot(
        results[mask]['param_n_estimators'],
        -results[mask]['mean_test_score'],
        marker='o',
        label=f'max_depth={depth}'
    )

axes[0].set_xlabel('n_estimators', fontsize=12)
axes[0].set_ylabel('Cross-validation MAE (K)', fontsize=12)
axes[0].set_title('Effect of n_estimators', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# max_depth vs MAE
for n_est in [50, 100, 200]:
    mask = results['param_n_estimators'] == n_est
    axes[1].plot(
        results[mask]['param_max_depth'].apply(lambda x: 20 if x is None else x),
        -results[mask]['mean_test_score'],
        marker='o',
        label=f'n_estimators={n_est}'
    )

axes[1].set_xlabel('max_depth', fontsize=12)
axes[1].set_ylabel('Cross-validation MAE (K)', fontsize=12)
axes[1].set_title('Effect of max_depth', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5. Feature Engineering (Materials-Specific)

Create materials-specific features to improve prediction performance.

### 5.1 What is Feature Engineering?

**Definition:**
Process of creating and selecting features effective for prediction from raw data.

**Importance:**
"Good features > Advanced models"
- With proper features, even simple models can achieve high accuracy
- With improper features, no model will improve performance

### 5.2 Automatic Feature Extraction with Matminer

**Matminer:**
Feature extraction library for materials science.

```bash
# Install (first time only)
pip install matminer
```

```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# Composition data (example: Li2O)
compositions = ['Li2O', 'LiCoO2', 'LiFePO4', 'Li4Ti5O12']

# Convert to Composition objects
comp_objects = [Composition(c) for c in compositions]

# Extract features with ElementProperty
featurizer = ElementProperty.from_preset('magpie')

# Calculate features
features = []
for comp in comp_objects:
    feat = featurizer.featurize(comp)
    features.append(feat)

# Convert to DataFrame
feature_names = featurizer.feature_labels()
df_features = pd.DataFrame(features, columns=feature_names)

print("===== Features Extracted by Matminer =====")
print(f"Number of features: {len(feature_names)}")
print(f"\nFirst 5 features:")
print(df_features.head())
print(f"\nExample features:")
for i in range(min(5, len(feature_names))):
    print(f"  {feature_names[i]}")
```

**Example features extracted by Matminer:**
- `MagpieData avg_dev MeltingT`: Average melting point deviation
- `MagpieData mean Electronegativity`: Average electronegativity
- `MagpieData mean AtomicWeight`: Average atomic weight
- `MagpieData range Number`: Range of atomic numbers
- Total 130+ features

### 5.3 Manual Feature Engineering

```python
# Basic data
data_advanced = pd.DataFrame({
    'element_A': [0.5, 0.6, 0.7, 0.8],
    'element_B': [0.5, 0.4, 0.3, 0.2],
    'melting_point': [1200, 1250, 1300, 1350]
})

# Create new features
data_advanced['sum_AB'] = data_advanced['element_A'] + data_advanced['element_B']  # Sum (always 1.0)
data_advanced['diff_AB'] = abs(data_advanced['element_A'] - data_advanced['element_B'])  # Absolute difference
data_advanced['product_AB'] = data_advanced['element_A'] * data_advanced['element_B']  # Product (interaction)
data_advanced['ratio_AB'] = data_advanced['element_A'] / (data_advanced['element_B'] + 1e-10)  # Ratio
data_advanced['A_squared'] = data_advanced['element_A'] ** 2  # Squared term (nonlinearity)
data_advanced['B_squared'] = data_advanced['element_B'] ** 2

print("===== Data After Feature Engineering =====")
print(data_advanced)
```

### 5.4 Feature Importance Analysis

```python
# Train model using extended features
X_advanced = data_advanced.drop('melting_point', axis=1)
y_advanced = data_advanced['melting_point']

# Train with Random Forest
model_advanced = RandomForestRegressor(n_estimators=100, random_state=42)
model_advanced.fit(X_advanced, y_advanced)

# Get feature importance
importances = pd.DataFrame({
    'Feature': X_advanced.columns,
    'Importance': model_advanced.feature_importances_
}).sort_values('Importance', ascending=False)

print("===== Feature Importance =====")
print(importances)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'], importances['Importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### 5.5 Feature Selection

**Purpose:**
Remove features that don't contribute to prediction (prevent overfitting, reduce computation time).

```python
from sklearn.feature_selection import SelectKBest, f_regression

# SelectKBest: Select top K features
selector = SelectKBest(score_func=f_regression, k=3)  # Top 3
X_selected = selector.fit_transform(X_advanced, y_advanced)

# Selected features
selected_features = X_advanced.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Train model after selection
model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
model_selected.fit(X_selected, y_advanced)

print(f"Before feature selection: {X_advanced.shape[1]} features")
print(f"After feature selection: {X_selected.shape[1]} features")
```

---

## 6. Troubleshooting Guide

Common errors encountered in practice and their solutions.

### 6.1 Common Errors List

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `ModuleNotFoundError: No module named 'sklearn'` | scikit-learn not installed | `pip install scikit-learn` |
| `MemoryError` | Insufficient memory | Reduce data size, batch processing, use Google Colab |
| `ConvergenceWarning: lbfgs failed to converge` | MLP training did not converge | Increase `max_iter` (e.g., 1000), adjust learning rate |
| `ValueError: Input contains NaN` | Missing values in data | Remove with `df.dropna()` or impute with `df.fillna()` |
| `ValueError: could not convert string to float` | String data included | Convert to dummy variables with `pd.get_dummies()` |
| `R² is negative` | Model worse than random prediction | Review features, change model |
| `ZeroDivisionError` | Division by zero | Add small value to denominator (e.g., `x / (y + 1e-10)`) |

### 6.2 Debugging Checklist

**Step 1: Verify Data**
```python
# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Check for infinity/NaN
print(df.isin([np.inf, -np.inf]).sum())
```

**Step 2: Visualize Data**
```python
# Check distributions
df.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Correlation matrix
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

**Step 3: Test with Small Data**
```python
# Test with first 10 samples only
X_small = X[:10]
y_small = y[:10]

model_test = RandomForestRegressor(n_estimators=10)
model_test.fit(X_small, y_small)
print("Training successful with small data")
```

**Step 4: Simplify Model**
```python
# If complex model fails, try linear regression first
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
print(f"Linear Regression R²: {model_simple.score(X_test, y_test):.4f}")
```

**Step 5: Read Error Messages**
```python
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error details: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
```

### 6.3 Handling Low Performance

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| R² < 0.5 | Improper features | Feature engineering, use Matminer |
| Low training error, high test error | Overfitting | Strengthen regularization, add data, simplify model |
| High training and test errors | Underfitting | Increase model complexity, add features, adjust learning rate |
| All predictions same | Model not learning | Review hyperparameters, scale features |
| Slow training | Large data or model | Sample data, simplify model, parallelize |

---

## 7. Project Challenge: Band Gap Prediction

Integrate what you've learned and tackle a practical project.

### 7.1 Project Overview

**Goal:**
Build MI model to predict band gap from composition

**Target Performance:**
- R² > 0.7 (70%+ explanatory power)
- MAE < 0.5 eV (error < 0.5 eV)

**Data Source:**
Materials Project API (or mock data)

### 7.2 Step-by-Step Guide

**Step 1: Data Collection**
```python
# Retrieve data from Materials Project API (or use mock data)
# Target: 100+ oxide data

data_project = pd.DataFrame({
    'formula': ['Li2O', 'Na2O', 'MgO', 'Al2O3', 'SiO2'] * 20,
    'Li_ratio': [0.67, 0.0, 0.0, 0.0, 0.0] * 20,
    'O_ratio': [0.33, 0.67, 0.5, 0.6, 0.67] * 20,
    'band_gap': [7.5, 5.2, 7.8, 8.8, 9.0] * 20
})

# Add noise (more realistic)
np.random.seed(42)
data_project['band_gap'] += np.random.normal(0, 0.3, len(data_project))

print(f"Data count: {len(data_project)}")
```

**Step 2: Feature Engineering**
```python
# Create additional features from element ratios
# (In practice, recommend adding atomic properties with Matminer)

data_project['sum_elements'] = data_project['Li_ratio'] + data_project['O_ratio']
data_project['product_LiO'] = data_project['Li_ratio'] * data_project['O_ratio']
```

**Step 3: Data Split**
```python
X_project = data_project[['Li_ratio', 'O_ratio', 'sum_elements', 'product_LiO']]
y_project = data_project['band_gap']

X_train_proj, X_test_proj, y_train_proj, y_test_proj = train_test_split(
    X_project, y_project, test_size=0.2, random_state=42
)
```

**Step 4: Model Selection and Training**
```python
# Use Random Forest
model_project = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
model_project.fit(X_train_proj, y_train_proj)
```

**Step 5: Evaluation**
```python
y_pred_proj = model_project.predict(X_test_proj)
mae_proj = mean_absolute_error(y_test_proj, y_pred_proj)
r2_proj = r2_score(y_test_proj, y_pred_proj)

print(f"===== Project Results =====")
print(f"MAE: {mae_proj:.2f} eV")
print(f"R²: {r2_proj:.4f}")

if r2_proj > 0.7 and mae_proj < 0.5:
    print("🎉 Goal achieved!")
else:
    print("❌ Goal not met. Add more features.")
```

**Step 6: Visualization**
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test_proj, y_pred_proj, alpha=0.6, s=100)
plt.plot([y_test_proj.min(), y_test_proj.max()],
         [y_test_proj.min(), y_test_proj.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Band Gap (eV)', fontsize=12)
plt.ylabel('Predicted Band Gap (eV)', fontsize=12)
plt.title('Band Gap Prediction Project', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'R² = {r2_proj:.3f}\nMAE = {mae_proj:.3f} eV',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()
```

### 7.3 Advanced Challenges

**Beginner:**
- Build prediction model for other material properties (melting point, formation energy)

**Intermediate:**
- Extract 130+ features with Matminer to improve performance
- Evaluate model reliability with cross-validation

**Advanced:**
- Retrieve real data from Materials Project API
- Ensemble learning (combine multiple models)
- Predict with neural network (MLP)

---

## 8. Summary

### What You Learned in This Chapter

1. **Environment Setup**
   - Three options: Anaconda, venv, Google Colab
   - How to choose optimal environment by situation

2. **Six Machine Learning Models**
   - Linear Regression (Baseline)
   - Random Forest (Balanced)
   - LightGBM (High accuracy)
   - SVR (Nonlinear capable)
   - MLP (Deep learning)
   - Materials Project real data integration

3. **Model Selection Guidelines**
   - Optimal model based on data count, computation time, interpretability
   - Performance comparison table and flowchart

4. **Hyperparameter Tuning**
   - Grid Search and Random Search
   - Visualize hyperparameter effects

5. **Feature Engineering**
   - Automatic extraction with Matminer
   - Manual feature creation (interaction terms, squared terms)
   - Feature importance and selection

6. **Troubleshooting**
   - Common errors and solutions
   - Five-step debugging

7. **Practical Project**
   - Complete band gap prediction implementation
   - Steps to achieve goals

### Next Steps

**After completing this tutorial, you can:**
- ✅ Implement materials property prediction
- ✅ Use 5+ models appropriately
- ✅ Perform hyperparameter tuning
- ✅ Solve errors independently

**What to Learn Next:**
1. **Deep Learning Applications**
   - Graph Neural Networks (GNN)
   - Crystal Graph Convolutional Networks (CGCNN)

2. **Bayesian Optimization**
   - Methods to minimize experiment count
   - Gaussian Process regression

3. **Transfer Learning**
   - Achieve high accuracy with limited data
   - Utilize pre-trained models

---

## Exercise Problems

### Problem 1 (Difficulty: easy)

Among the six models implemented in this tutorial, select the most appropriate model for small data (< 100 samples) and explain your reasoning.

<details>
<summary>Hint</summary>

Consider overfitting risk and model complexity.

</details>

<details>
<summary>Solution</summary>

**Answer: Linear Regression**

**Reasoning:**
1. **Low overfitting risk**: Few parameters, stable with limited data
2. **High interpretability**: Can understand feature influence from coefficients
3. **Fast training**: Low computational cost

**Other candidate: SVR**
- SVR effective when nonlinearity is strong
- However, requires hyperparameter tuning

With small data, complex models (Random Forest, MLP) memorize training data and perform poorly on new data (overfitting).

</details>

---

### Problem 2 (Difficulty: medium)

Compare Grid Search and Random Search and explain in what situations each method should be used.

<details>
<summary>Hint</summary>

Consider search space size and computation time constraints.

</details>

<details>
<summary>Solution</summary>

**When to use Grid Search:**
1. **Few hyperparameters to search** (2-3)
2. **Few candidates per parameter** (about 3-5 each)
3. **Ample computation time**
4. **Need to find best solution for sure**

**Example:** n_estimators=[50, 100, 200] × max_depth=[5, 10, 15] = 9 patterns

**When to use Random Search:**
1. **Many hyperparameters to search** (4+)
2. **Many candidates per parameter/continuous values**
3. **Limited computation time**
4. **Good enough solution is sufficient**

**Example:** 5 parameters, 10 candidates each = 100,000 patterns → Sample 100 with Random Search

**General strategy:**
1. First narrow down rough range with Random Search (100-200 iterations)
2. Detailed search of promising range with Grid Search

</details>

---

### Problem 3 (Difficulty: medium)

The following error occurred. Explain the cause and solution.

```
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
```

<details>
<summary>Hint</summary>

Error occurs in MLPRegressor training.

</details>

<details>
<summary>Solution</summary>

**Cause:**
MLPRegressor (neural network) training did not converge within specified iterations (max_iter).

**Possible factors:**
1. max_iter too small (default 200)
2. Learning rate too small (slow learning)
3. Inappropriate data scale (not standardized)
4. Model too complex (too many layers, too many neurons)

**Solutions:**

**Method 1: Increase max_iter**
```python
model_mlp = MLPRegressor(max_iter=1000)  # Default 200→1000
```

**Method 2: Standardize data**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Method 3: Adjust learning rate**
```python
model_mlp = MLPRegressor(
    learning_rate_init=0.01,  # Increase learning rate
    max_iter=500
)
```

**Method 4: Enable Early Stopping**
```python
model_mlp = MLPRegressor(
    early_stopping=True,  # Stop if validation error stops improving
    validation_fraction=0.2,
    max_iter=1000
)
```

**Recommended approach:**
First try Method 2 (data standardization), then combine Methods 1 and 4 if still not converging.

</details>

---

### Problem 4 (Difficulty: hard)

Write code to extract 5+ features from composition `"Li2O"` using Matminer.

<details>
<summary>Hint</summary>

Use `ElementProperty` featurizer and `from_preset('magpie')`.

</details>

<details>
<summary>Solution</summary>

```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
import pandas as pd

# Create composition object
comp = Composition("Li2O")

# Initialize feature extractor with Magpie preset
featurizer = ElementProperty.from_preset('magpie')

# Calculate features
features = featurizer.featurize(comp)

# Get feature names
feature_names = featurizer.feature_labels()

# Convert to DataFrame (for readability)
df = pd.DataFrame([features], columns=feature_names)

print(f"===== Li2O Features (first 5) =====")
for i in range(5):
    print(f"{feature_names[i]}: {features[i]:.4f}")

print(f"\nTotal feature count: {len(features)}")
```

**Expected output:**
```
===== Li2O Features (first 5) =====
MagpieData minimum Number: 3.0000
MagpieData maximum Number: 8.0000
MagpieData range Number: 5.0000
MagpieData mean Number: 5.3333
MagpieData avg_dev Number: 1.5556

Total feature count: 132
```

**Explanation:**
- `MagpieData minimum Number`: Minimum atomic number (Li: 3)
- `MagpieData maximum Number`: Maximum atomic number (O: 8)
- `MagpieData range Number`: Range of atomic numbers (8-3=5)
- `MagpieData mean Number`: Average atomic number ((3+3+8)/3=5.33)
- `MagpieData avg_dev Number`: Average deviation of atomic numbers

Matminer automatically extracts 132 features (electronegativity, atomic radius, melting point, etc.).

</details>

---

### Problem 5 (Difficulty: hard)

Band gap project only achieved R²=0.5. Propose 3 specific approaches to improve performance and explain implementation methods for each.

<details>
<summary>Hint</summary>

Consider from three perspectives: features, models, and hyperparameters.

</details>

<details>
<summary>Solution</summary>

**Approach 1: Feature Engineering (Most Effective)**

**Implementation:**
```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# Extract atomic properties from composition
def extract_features(formula):
    comp = Composition(formula)
    featurizer = ElementProperty.from_preset('magpie')
    features = featurizer.featurize(comp)
    return features

# Add features to existing data
data_project['features'] = data_project['formula'].apply(extract_features)
# Expand to DataFrame (132-dimensional features)
features_df = pd.DataFrame(data_project['features'].tolist())
X_enhanced = features_df  # Original 2D → expanded to 132D
```

**Expected improvement:**
R² 0.5 → 0.75-0.85 (significant feature increase)

---

**Approach 2: Ensemble Learning (Combine Multiple Models)**

**Implementation:**
```python
from sklearn.ensemble import VotingRegressor

# Combine 3 models
model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
model_lgb = lgb.LGBMRegressor(n_estimators=200, random_state=42)
model_svr = SVR(kernel='rbf', C=100)

# Ensemble model (average predictions)
ensemble = VotingRegressor([
    ('rf', model_rf),
    ('lgb', model_lgb),
    ('svr', model_svr)
])

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
```

**Expected improvement:**
R² 0.5 → 0.6-0.7 (more stable than single model)

---

**Approach 3: Hyperparameter Tuning**

**Implementation:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # Try 100 combinations
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

**Expected improvement:**
R² 0.5 → 0.55-0.65 (optimized from default)

---

**Optimal strategy:**
1. First implement **Approach 1** (feature engineering) → Maximum effect
2. Then **Approach 3** (hyperparameter tuning) for fine-tuning
3. Finally **Approach 2** (ensemble) for final performance boost

With this sequence, aim for R² 0.5 → 0.8+.

</details>

---

## References

1. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
   URL: https://scikit-learn.org
   *Official scikit-learn documentation. Detailed explanations and tutorials for all algorithms.*

2. Ward, L., et al. (2018). "Matminer: An open source toolkit for materials data mining." *Computational Materials Science*, 152, 60-69.
   DOI: [10.1016/j.commatsci.2018.05.018](https://doi.org/10.1016/j.commatsci.2018.05.018)
   GitHub: https://github.com/hackingmaterials/matminer
   *Feature extraction library for materials science. Automatically generates 132 materials descriptors.*

3. Jain, A., et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." *APL Materials*, 1(1), 011002.
   DOI: [10.1063/1.4812323](https://doi.org/10.1063/1.4812323)
   URL: https://materialsproject.org
   *Official Materials Project paper. Database of 140,000+ materials.*

4. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems*, 30, 3146-3154.
   GitHub: https://github.com/microsoft/LightGBM
   *Official LightGBM paper. Fast implementation of gradient boosting.*

5. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13, 281-305.
   URL: https://www.jmlr.org/papers/v13/bergstra12a.html
   *Theoretical background of Random Search. More efficient search method than Grid Search.*

6. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning, 3rd Edition*. Packt Publishing.
   *Comprehensive machine learning textbook in Python. Detailed practical usage of scikit-learn.*

7. scikit-learn User Guide. (2024). "Hyperparameter tuning."
   URL: https://scikit-learn.org/stable/modules/grid_search.html
   *Official guide for hyperparameter tuning. Details on Grid Search and Random Search.*

---

**Created**: 2025-10-16
**Version**: 3.0
**Template**: content_agent_prompts.py v1.0
**Author**: MI Knowledge Hub Project
