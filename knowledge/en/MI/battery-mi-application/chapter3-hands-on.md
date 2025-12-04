---
title: Battery MI Implementation Hands-On with Python
chapter_title: Battery MI Implementation Hands-On with Python
subtitle: Learning Battery Materials Design Through Implementation Code
reading_time: 60-70 minutes
difficulty: Intermediate
code_examples: 30
exercises: 5
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Battery MI Implementation Hands-On with Python

This chapter covers Battery MI Implementation Hands. You will learn essential concepts and techniques.

**Learning Objectives:** \- Build and simulate battery models with PyBaMM \- Implement and evaluate capacity/voltage prediction models \- Predict cycle degradation with LSTM \- Explore optimal materials with Bayesian optimization

**Prerequisites:** \- Python fundamentals (NumPy, Pandas, Matplotlib) \- Machine learning basics (scikit-learn) \- Understanding of Chapters 1 and 2 content

**Execution Environment:**
    
    
    pip install pybamm numpy pandas scikit-learn tensorflow scikit-optimize matplotlib seaborn
    

* * *

## 3.1 Battery Data Acquisition and Preprocessing

### Example 1: Acquiring Cathode Material Data from Materials Project
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Example 1: Acquiring Cathode Material Data from Materials Pr
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from pymatgen.ext.matproj import MPRester
    import pandas as pd
    
    # Materials Project API
    API_KEY = "YOUR_API_KEY"  # Obtain from https://materialsproject.org/open
    
    with MPRester(API_KEY) as mpr:
        # Search for Li-containing oxides
        data = mpr.query(
            criteria={
                "elements": {"$all": ["Li"], "$in": ["Co", "Ni", "Mn"]},
                "nelements": {"$lte": 4}
            },
            properties=["material_id", "pretty_formula", "energy_per_atom",
                       "band_gap", "formation_energy_per_atom"]
        )
    
    df = pd.DataFrame(data)
    print(f"Number of materials retrieved: {len(df)}")
    print(df.head())
    

### Example 2: Loading and Visualizing Charge-Discharge Curves
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Sample charge-discharge data
    def load_charge_discharge_data():
        """Generate charge-discharge curve data (actual: file reading)"""
        capacity = np.linspace(0, 200, 100)  # mAh/g
        voltage_charge = 3.0 + 0.7 * (capacity / 200) + 0.3 * np.sin(capacity / 20)
        voltage_discharge = 3.0 + 0.6 * (capacity / 200) + 0.2 * np.sin(capacity / 20)
        return capacity, voltage_charge, voltage_discharge
    
    cap, V_ch, V_dch = load_charge_discharge_data()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cap, V_ch, 'r-', label='Charge', linewidth=2)
    ax.plot(cap, V_dch, 'b-', label='Discharge', linewidth=2)
    ax.set_xlabel('Capacity (mAh/g)', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('Charge-Discharge Curve', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    print(f"Average charge voltage: {V_ch.mean():.2f} V")
    print(f"Average discharge voltage: {V_dch.mean():.2f} V")
    

### Example 3: Potential Profile Calculation
    
    
    from scipy.integrate import cumtrapz
    
    def calculate_average_voltage(capacity, voltage):
        """Calculate average voltage"""
        energy = cumtrapz(voltage, capacity, initial=0)
        avg_voltage = energy[-1] / capacity[-1] if capacity[-1] > 0 else 0
        return avg_voltage
    
    # Average voltage for charge/discharge
    V_avg_ch = calculate_average_voltage(cap, V_ch)
    V_avg_dch = calculate_average_voltage(cap, V_dch)
    
    # Energy density
    capacity_max = cap[-1]  # mAh/g
    energy_density = capacity_max * V_avg_dch * 0.001  # Wh/g
    
    print(f"Average charge voltage: {V_avg_ch:.3f} V")
    print(f"Average discharge voltage: {V_avg_dch:.3f} V")
    print(f"Energy density: {energy_density:.1f} Wh/g")
    

### Example 4: Capacity Calculation and Coulombic Efficiency
    
    
    def calculate_coulombic_efficiency(Q_charge, Q_discharge):
        """Calculate Coulombic efficiency"""
        CE = (Q_discharge / Q_charge) * 100
        return CE
    
    # Sample data
    Q_charge = 195.0  # mAh/g
    Q_discharge = 190.0  # mAh/g
    
    CE = calculate_coulombic_efficiency(Q_charge, Q_discharge)
    print(f"Charge capacity: {Q_charge:.1f} mAh/g")
    print(f"Discharge capacity: {Q_discharge:.1f} mAh/g")
    print(f"Coulombic efficiency: {CE:.2f}%")
    
    if CE < 98:
        print("⚠️ Warning: Low Coulombic efficiency (possible side reactions)")
    elif CE > 99.5:
        print("✅ Excellent: High Coulombic efficiency")
    

### Example 5: Automatic Descriptor Calculation (matminer)
    
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    
    # Cathode material compositions
    compositions = ["LiCoO2", "LiNi0.8Co0.15Al0.05O2", "LiFePO4"]
    
    # Descriptor calculation
    ep_feat = ElementProperty.from_preset("magpie")
    descriptors = []
    
    for comp_str in compositions:
        comp = Composition(comp_str)
        desc = ep_feat.featurize(comp)
        descriptors.append(desc)
    
    # Convert to DataFrame
    feature_labels = ep_feat.feature_labels()
    df_desc = pd.DataFrame(descriptors, columns=feature_labels, index=compositions)
    
    print("Descriptor examples (first 5 columns):")
    print(df_desc.iloc[:, :5])
    print(f"\nTotal number of descriptors: {len(feature_labels)}")
    

### Example 6: Data Cleaning and Outlier Removal
    
    
    # Requirements:
    # - Python 3.9+
    # - scipy>=1.11.0
    
    """
    Example: Example 6: Data Cleaning and Outlier Removal
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    
    # Sample data
    np.random.seed(42)
    capacity_data = np.concatenate([
        np.random.normal(180, 10, 95),  # Normal data
        np.array([250, 280, 300, 310, 50])  # Outliers
    ])
    
    # Outlier detection using Z-score
    z_scores = np.abs(stats.zscore(capacity_data))
    threshold = 3
    outliers = z_scores > threshold
    
    print(f"Number of data points: {len(capacity_data)}")
    print(f"Number of outliers: {outliers.sum()} ({outliers.sum()/len(capacity_data)*100:.1f}%)")
    print(f"Outliers: {capacity_data[outliers]}")
    
    # After cleaning
    capacity_clean = capacity_data[~outliers]
    print(f"Data points after cleaning: {len(capacity_clean)}")
    print(f"Average capacity: {capacity_clean.mean():.1f} ± {capacity_clean.std():.1f} mAh/g")
    

### Example 7: Train/Test Data Splitting
    
    
    from sklearn.model_selection import train_test_split
    
    # Sample dataset
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)  # 10 descriptors
    y = 150 + 30 * X[:, 0] - 20 * X[:, 1] + np.random.randn(n_samples) * 5  # Capacity
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Capacity range: {y.min():.1f} - {y.max():.1f} mAh/g")
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nStandardization complete")
    

* * *

## 3.2 Capacity and Voltage Prediction Models

### Example 8: Random Forest Regression (Capacity Prediction)
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Model training
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    
    # Prediction
    y_pred_train = model_rf.predict(X_train_scaled)
    y_pred_test = model_rf.predict(X_test_scaled)
    
    # Evaluation
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Random Forest Capacity Prediction:")
    print(f"  Training: MAE={mae_train:.2f} mAh/g, R²={r2_train:.3f}")
    print(f"  Test: MAE={mae_test:.2f} mAh/g, R²={r2_test:.3f}")
    

### Example 9: XGBoost (Voltage Prediction)
    
    
    # Requirements:
    # - Python 3.9+
    # - xgboost>=2.0.0
    
    """
    Example: Example 9: XGBoost (Voltage Prediction)
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Beginner
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from xgboost import XGBRegressor
    
    # Voltage data (sample)
    y_voltage = 3.7 + 0.3 * X[:, 0] - 0.2 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    y_v_train, y_v_test = y_voltage[:len(X_train)], y_voltage[len(X_train):]
    
    # XGBoost model
    model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model_xgb.fit(X_train_scaled, y_v_train)
    
    # Prediction and evaluation
    y_v_pred = model_xgb.predict(X_test_scaled)
    mae_voltage = mean_absolute_error(y_v_test, y_v_pred)
    r2_voltage = r2_score(y_v_test, y_v_pred)
    
    print(f"XGBoost Voltage Prediction:")
    print(f"  MAE: {mae_voltage:.3f} V")
    print(f"  R²: {r2_voltage:.3f}")
    

### Example 10: Neural Network (Keras)
    
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Model construction
    model_nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Training
    history = model_nn.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # Evaluation
    y_nn_pred = model_nn.predict(X_test_scaled).flatten()
    mae_nn = mean_absolute_error(y_test, y_nn_pred)
    r2_nn = r2_score(y_test, y_nn_pred)
    
    print(f"Neural Network Capacity Prediction:")
    print(f"  MAE: {mae_nn:.2f} mAh/g")
    print(f"  R²: {r2_nn:.3f}")
    

### Example 11: Graph Neural Network (Conceptual Implementation)
    
    
    # Using PyTorch Geometric (implementation overview)
    """
    from torch_geometric.nn import CGConv, global_mean_pool
    
    class CrystalGNN(torch.nn.Module):
        def __init__(self, node_features, edge_features, hidden_dim):
            super().__init__()
            self.conv1 = CGConv(node_features, edge_features, hidden_dim)
            self.conv2 = CGConv(hidden_dim, edge_features, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim, 1)
    
        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = global_mean_pool(x, data.batch)
            return self.fc(x)
    
    # Training/Prediction (details in Chapter 4)
    """
    
    print("Graph Neural Network Concept:")
    print("  Input: Crystal structure (atomic coordinates, bonding information)")
    print("  Processing: Graph Convolution Layers")
    print("  Output: Capacity, voltage prediction")
    print("  Advantages: No descriptor design needed, high accuracy")
    

### Example 12: Transfer Learning
    
    
    from tensorflow.keras.models import load_model
    
    # Pre-trained model (virtual)
    def create_pretrained_model():
        """Model trained on LIB cathode materials"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        return model
    
    pretrained = create_pretrained_model()
    pretrained.compile(optimizer=Adam(lr=0.001), loss='mse')
    # Virtual training
    pretrained.fit(X_train_scaled[:100], y_train[:100], epochs=50, verbose=0)
    
    # Fine-tuning (all-solid-state battery data)
    X_target = X_train_scaled[100:120]
    y_target = y_train[100:120]
    
    # Re-train final layer
    for layer in pretrained.layers[:-1]:
        layer.trainable = False
    
    pretrained.compile(optimizer=Adam(lr=1e-4), loss='mse')
    pretrained.fit(X_target, y_target, epochs=30, verbose=0)
    
    # Evaluation
    y_tl_pred = pretrained.predict(X_test_scaled).flatten()
    mae_tl = mean_absolute_error(y_test, y_tl_pred)
    
    print(f"Transfer Learning:")
    print(f"  Source: LIB cathode (100 samples)")
    print(f"  Target: All-solid-state battery (20 samples)")
    print(f"  MAE: {mae_tl:.2f} mAh/g")
    

### Example 13: Feature Importance Analysis (SHAP)
    
    
    # Requirements:
    # - Python 3.9+
    # - shap>=0.42.0
    
    """
    Example: Example 13: Feature Importance Analysis (SHAP)
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import shap
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(X_test_scaled[:100])
    
    # Feature importance
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names, show=False)
    
    print("SHAP Analysis:")
    print("  Quantify contribution of each feature to capacity")
    print("  Positive contribution: Capacity increase")
    print("  Negative contribution: Capacity decrease")
    

### Example 14: Cross-Validation
    
    
    from sklearn.model_selection import cross_val_score
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(
        model_rf, X_train_scaled, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    
    print(f"5-fold Cross-Validation:")
    print(f"  MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} mAh/g")
    print(f"  Each fold: {-cv_scores}")
    

### Example 15: Parity Plot
    
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
           'r--', linewidth=2, label='Ideal')
    
    ax.set_xlabel('Actual Capacity (mAh/g)', fontsize=12)
    ax.set_ylabel('Predicted Capacity (mAh/g)', fontsize=12)
    ax.set_title(f'Parity Plot (MAE={mae_test:.2f} mAh/g)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    print("Parity Plot: Comparison of predicted vs. actual values")
    

* * *

## 3.3 Cycle Degradation Prediction

### Example 16: Time-Series Data Preparation for Charge-Discharge Curves
    
    
    def generate_cycle_data(n_cycles=500):
        """Generate cycle data"""
        cycles = np.arange(1, n_cycles + 1)
    
        # Capacity fade (exponential)
        Q_initial = 200  # mAh/g
        decay_rate = 0.0005
        capacity = Q_initial * np.exp(-decay_rate * cycles) + np.random.randn(n_cycles) * 2
    
        # SOH (State of Health)
        SOH = (capacity / Q_initial) * 100
    
        return cycles, capacity, SOH
    
    cycles, capacity, SOH = generate_cycle_data()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cycles, capacity)
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (mAh/g)')
    plt.title('Capacity Fade')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(cycles, SOH)
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH (%)')
    plt.axhline(80, color='r', linestyle='--', label='80% threshold')
    plt.title('State of Health')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    print(f"Initial capacity: {capacity[0]:.1f} mAh/g")
    print(f"Final capacity: {capacity[-1]:.1f} mAh/g")
    print(f"Capacity retention: {SOH[-1]:.1f}%")
    

### Example 17: LSTM (Long Short-Term Memory) Model
    
    
    from tensorflow.keras.layers import LSTM
    
    # Data preparation (time-series window)
    def create_sequences(data, seq_length=50):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 50
    X_lstm, y_lstm = create_sequences(capacity, seq_length)
    X_lstm = X_lstm.reshape(-1, seq_length, 1)
    
    # Train/Test split
    split = int(0.8 * len(X_lstm))
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
    
    # LSTM model
    model_lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_lstm.compile(optimizer=Adam(lr=0.001), loss='mse')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    # Prediction
    y_lstm_pred = model_lstm.predict(X_test_lstm).flatten()
    mae_lstm = mean_absolute_error(y_test_lstm, y_lstm_pred)
    
    print(f"LSTM Degradation Prediction:")
    print(f"  MAE: {mae_lstm:.2f} mAh/g")
    print(f"  Sequence length: {seq_length} cycles")
    

### Example 18: GRU (Gated Recurrent Unit) Model
    
    
    from tensorflow.keras.layers import GRU
    
    # GRU model (lighter than LSTM)
    model_gru = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
        GRU(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_gru.compile(optimizer=Adam(lr=0.001), loss='mse')
    model_gru.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    # Prediction
    y_gru_pred = model_gru.predict(X_test_lstm).flatten()
    mae_gru = mean_absolute_error(y_test_lstm, y_gru_pred)
    
    print(f"GRU Degradation Prediction:")
    print(f"  MAE: {mae_gru:.2f} mAh/g")
    print(f"  LSTM comparison: Parameter count {model_gru.count_params()} vs {model_lstm.count_params()}")
    

### Example 19: Lifetime Prediction (RUL: Remaining Useful Life)
    
    
    def predict_RUL(capacity_history, threshold=160):
        """Predict cycles until 80% capacity (160 mAh/g) is reached"""
        # Predict from initial 100 cycles
        early_cycles = capacity_history[:100]
    
        # Linear fit
        x = np.arange(len(early_cycles))
        coeffs = np.polyfit(x, early_cycles, 1)
        decay_rate = -coeffs[0]
    
        # RUL calculation
        current_capacity = early_cycles[-1]
        remaining = current_capacity - threshold
        RUL = int(remaining / decay_rate) if decay_rate > 0 else np.inf
    
        return RUL, decay_rate
    
    RUL, decay = predict_RUL(capacity)
    actual_life = np.where(capacity < 160)[0][0] if np.any(capacity < 160) else len(capacity)
    
    print(f"Lifetime prediction (from initial 100 cycles):")
    print(f"  Predicted RUL: {RUL} cycles")
    print(f"  Actual lifetime: {actual_life} cycles")
    print(f"  Prediction error: {abs(RUL - actual_life)} cycles ({abs(RUL - actual_life)/actual_life*100:.1f}%)")
    print(f"  Degradation rate: {decay:.3f} mAh/g/cycle")
    

### Example 20: Degradation Rate Prediction
    
    
    def analyze_degradation_rate(capacity, window=50):
        """Degradation rate analysis using moving window"""
        rates = []
        cycles = []
    
        for i in range(window, len(capacity)):
            window_data = capacity[i-window:i]
            x = np.arange(window)
            rate = -np.polyfit(x, window_data, 1)[0]
            rates.append(rate)
            cycles.append(i)
    
        return np.array(cycles), np.array(rates)
    
    cycles_rate, degradation_rates = analyze_degradation_rate(capacity)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cycles_rate, degradation_rates * 1000, linewidth=2)
    plt.xlabel('Cycle Number')
    plt.ylabel('Degradation Rate (mAh/g per 1000 cycles)')
    plt.title('Degradation Rate Evolution')
    plt.grid(alpha=0.3)
    
    print(f"Average degradation rate: {degradation_rates.mean():.4f} mAh/g/cycle")
    print(f"Maximum degradation rate: {degradation_rates.max():.4f} mAh/g/cycle (cycle {cycles_rate[degradation_rates.argmax()]})")
    

### Example 21: Anomaly Detection (Isolation Forest)
    
    
    from sklearn.ensemble import IsolationForest
    
    # Features: capacity, degradation rate
    features = np.column_stack([capacity[50:], degradation_rates])
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    anomalies = clf.fit_predict(features)
    
    n_anomalies = (anomalies == -1).sum()
    print(f"Anomaly Detection:")
    print(f"  Number of anomalous cycles: {n_anomalies}")
    print(f"  Anomaly rate: {n_anomalies/len(anomalies)*100:.1f}%")
    print(f"  Anomalous cycles: {cycles_rate[anomalies == -1]}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(cycles_rate[anomalies == 1], capacity[50:][anomalies == 1],
               c='blue', label='Normal', alpha=0.6)
    plt.scatter(cycles_rate[anomalies == -1], capacity[50:][anomalies == -1],
               c='red', label='Anomaly', s=100, marker='x')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (mAh/g)')
    plt.title('Anomaly Detection in Cycle Data')
    plt.legend()
    plt.grid(alpha=0.3)
    

### Example 22: SOH (State of Health) Estimation
    
    
    def estimate_SOH(current_capacity, initial_capacity=200):
        """SOH estimation"""
        SOH = (current_capacity / initial_capacity) * 100
    
        if SOH > 95:
            status = "Excellent"
        elif SOH > 80:
            status = "Good"
        elif SOH > 70:
            status = "Degradation in progress"
        else:
            status = "Replacement needed"
    
        return SOH, status
    
    # SOH estimation at each cycle
    for cycle in [100, 200, 300, 400, 500]:
        if cycle <= len(capacity):
            soh, status = estimate_SOH(capacity[cycle-1])
            print(f"Cycle {cycle:3d}: SOH={soh:5.1f}%, Capacity={capacity[cycle-1]:5.1f} mAh/g, Status={status}")
    

* * *

## 3.4 Material Exploration via Bayesian Optimization

### Example 23: Gaussian Process Regression
    
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    
    # Sample data (Ni ratio vs capacity)
    X_gp = np.array([[0.3], [0.5], [0.6], [0.7], [0.9]])
    y_gp = np.array([160, 180, 195, 190, 170])
    
    # GPR model
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X_gp, y_gp)
    
    # Prediction
    X_pred = np.linspace(0.2, 1.0, 100).reshape(-1, 1)
    y_pred, y_std = gpr.predict(X_pred, return_std=True)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(X_pred, y_pred, 'b-', label='Mean prediction')
    plt.fill_between(X_pred.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                    alpha=0.3, label='±2σ')
    plt.scatter(X_gp, y_gp, c='red', s=100, label='Observations', zorder=10)
    plt.xlabel('Ni Ratio')
    plt.ylabel('Capacity (mAh/g)')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.grid(alpha=0.3)
    
    print(f"Optimal Ni ratio (predicted): {X_pred[np.argmax(y_pred)][0]:.2f}")
    print(f"Maximum predicted capacity: {y_pred.max():.1f} mAh/g")
    

### Example 24: Bayesian Optimization Loop
    
    
    from skopt import gp_minimize
    from skopt.space import Real
    
    # Objective function (capacity simulation)
    def battery_capacity(x):
        """Predict capacity from Ni ratio (actual: experiment or DFT calculation)"""
        ni_ratio = x[0]
        # Virtual capacity function
        capacity = 200 * ni_ratio - 150 * (ni_ratio - 0.65)**2 + np.random.randn() * 3
        return -capacity  # Convert to minimization problem
    
    # Search space
    space = [Real(0.3, 1.0, name='Ni_ratio')]
    
    # Bayesian optimization
    result = gp_minimize(
        battery_capacity,
        space,
        n_calls=20,
        random_state=42,
        verbose=False
    )
    
    print(f"Bayesian Optimization Results:")
    print(f"  Optimal Ni ratio: {result.x[0]:.3f}")
    print(f"  Maximum capacity: {-result.fun:.1f} mAh/g")
    print(f"  Number of experiments: {len(result.x_iters)}")
    

### Example 25: Multi-Objective Optimization (Capacity & Cycle Life)
    
    
    def multi_objective(x):
        """Trade-off between capacity and cycle life"""
        ni_ratio = x[0]
    
        # Capacity (increases with high Ni ratio)
        capacity = 200 * ni_ratio - 100 * (ni_ratio - 0.7)**2
    
        # Cycle life (improves with low Ni ratio)
        cycle_life = 2000 - 1000 * ni_ratio + 500 * (ni_ratio - 0.5)**2
    
        # Weighted sum (scalarization)
        weight_cap = 0.6
        weight_life = 0.4
    
        score = weight_cap * capacity + weight_life * (cycle_life / 10)
        return -score
    
    result_mo = gp_minimize(multi_objective, space, n_calls=25, random_state=42)
    
    print(f"Multi-Objective Optimization Results:")
    print(f"  Optimal Ni ratio: {result_mo.x[0]:.3f}")
    print(f"  Predicted capacity: {(200 * result_mo.x[0] - 100 * (result_mo.x[0] - 0.7)**2):.1f} mAh/g")
    print(f"  Predicted lifetime: {(2000 - 1000 * result_mo.x[0] + 500 * (result_mo.x[0] - 0.5)**2):.0f} cycles")
    

### Example 26: Constrained Optimization
    
    
    def constrained_optimization(x):
        """Capacity optimization with safety constraints"""
        ni_ratio = x[0]
    
        # Constraint: Ni ratio < 0.85 (safety consideration)
        if ni_ratio > 0.85:
            return 1e6  # Penalty
    
        # Capacity prediction
        capacity = 200 * ni_ratio - 120 * (ni_ratio - 0.7)**2
        return -capacity
    
    result_const = gp_minimize(constrained_optimization, space, n_calls=20, random_state=42)
    
    print(f"Constrained Optimization Results:")
    print(f"  Optimal Ni ratio: {result_const.x[0]:.3f} (< 0.85)")
    print(f"  Maximum capacity: {-result_const.fun:.1f} mAh/g")
    

### Example 27: Pareto Front Visualization
    
    
    # Multi-objective optimization results (capacity vs cycle life)
    ni_ratios = np.linspace(0.3, 1.0, 50)
    capacities = [200 * ni - 100 * (ni - 0.7)**2 for ni in ni_ratios]
    cycle_lives = [2000 - 1000 * ni + 500 * (ni - 0.5)**2 for ni in ni_ratios]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(capacities, cycle_lives, c=ni_ratios, cmap='viridis', s=50)
    plt.colorbar(label='Ni Ratio')
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Cycle Life')
    plt.title('Pareto Front: Capacity vs Cycle Life')
    plt.grid(alpha=0.3)
    
    # Pareto optimal point detection
    pareto_indices = []
    for i in range(len(capacities)):
        dominated = False
        for j in range(len(capacities)):
            if capacities[j] > capacities[i] and cycle_lives[j] > cycle_lives[i]:
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    
    plt.scatter([capacities[i] for i in pareto_indices],
               [cycle_lives[i] for i in pareto_indices],
               c='red', s=100, marker='*', label='Pareto Optimal', zorder=10)
    plt.legend()
    
    print(f"Number of Pareto optimal solutions: {len(pareto_indices)}")
    

* * *

## 3.5 Battery Simulation with PyBaMM

### Example 28: DFN Model (Doyle-Fuller-Newman)
    
    
    import pybamm
    
    # Build DFN model
    model = pybamm.lithium_ion.DFN()
    
    # Parameter settings (Graphite || LCO)
    parameter_values = pybamm.ParameterValues("Chen2020")
    
    # Simulation setup
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    
    # 1C discharge
    sim.solve([0, 3600])  # 0-3600 seconds (1 hour)
    
    # Get results
    time = sim.solution["Time [h]"].entries
    voltage = sim.solution["Terminal voltage [V]"].entries
    current = sim.solution["Current [A]"].entries
    
    print("DFN Model Simulation:")
    print(f"  Initial voltage: {voltage[0]:.3f} V")
    print(f"  Final voltage: {voltage[-1]:.3f} V")
    print(f"  Discharge time: {time[-1]:.2f} h")
    

### Example 29: Charge-Discharge Curve Simulation
    
    
    # Simulation at multiple C-rates
    c_rates = [0.5, 1, 2, 5]
    experiments = []
    
    for c_rate in c_rates:
        experiment = pybamm.Experiment([
            f"Discharge at {c_rate}C until 2.5 V",
            "Rest for 10 minutes",
            "Charge at 1C until 4.2 V",
            "Hold at 4.2 V until C/50"
        ])
        experiments.append(experiment)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, c_rate in enumerate(c_rates):
        sim = pybamm.Simulation(model, parameter_values=parameter_values,
                                experiment=experiments[i])
        sim.solve()
    
        time = sim.solution["Time [h]"].entries
        voltage = sim.solution["Terminal voltage [V]"].entries
    
        ax.plot(time, voltage, label=f'{c_rate}C', linewidth=2)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Discharge Curves at Different C-rates')
    ax.legend()
    ax.grid(alpha=0.3)
    
    print("Discharge simulation at multiple C-rates completed")
    

### Example 30: Parameter Optimization and Fitting
    
    
    # Experimental data (sample)
    experimental_voltage = voltage + np.random.randn(len(voltage)) * 0.05
    
    # Parameter optimization (simplified version)
    def fit_resistance(R_value):
        """Internal resistance fitting"""
        params = parameter_values.copy()
        params["Electrolyte conductivity [S.m-1]"] = R_value
    
        sim_fit = pybamm.Simulation(model, parameter_values=params)
        sim_fit.solve([0, 3600])
    
        sim_voltage = sim_fit.solution["Terminal voltage [V]"].entries
    
        # Error calculation
        mse = np.mean((sim_voltage - experimental_voltage)**2)
        return mse
    
    # Optimization
    from scipy.optimize import minimize_scalar
    
    result_fit = minimize_scalar(fit_resistance, bounds=(0.5, 2.0), method='bounded')
    
    print(f"Parameter Fitting:")
    print(f"  Optimal electrolyte conductivity: {result_fit.x:.3f} S/m")
    print(f"  MSE: {result_fit.fun:.6f}")
    
    # Simulation with optimal parameters
    params_opt = parameter_values.copy()
    params_opt["Electrolyte conductivity [S.m-1]"] = result_fit.x
    sim_opt = pybamm.Simulation(model, parameter_values=params_opt)
    sim_opt.solve([0, 3600])
    
    print("Optimization complete: Fit to experimental data")
    

* * *

## 3.6 Project Challenge

**Challenge: Discovering High-Capacity, Long-Cycle-Life Cathode Materials**

Design optimal NCM cathode materials following these steps:

  1. **Data Collection** : Obtain Ni-Co-Mn oxide data from Materials Project
  2. **Descriptor Calculation** : Composition ratios, lattice constants, band gaps, etc.
  3. **Build Prediction Model** : XGBoost for capacity prediction (target > 200 mAh/g)
  4. **Bayesian Optimization** : Optimize Ni:Co:Mn ratios (constraint: safety)
  5. **Cycle Performance Evaluation** : Cycle life simulation with PyBaMM (target > 2,000 cycles)

**Evaluation Criteria:** \- Capacity > 200 mAh/g \- Cycle life > 2,000 cycles (80% capacity retention) \- Safety: Ni ratio < 0.85 \- Cost: Minimize Co usage

**Deliverables:** \- Optimal composition (Ni:Co:Mn ratios) \- Predicted performance (capacity, lifetime) \- Complete Python code

* * *

## Exercises

**Q1:** Calculate the theoretical capacity of LiNi₀.₈Co₀.₁Mn₀.₁O₂ and determine the Coulombic efficiency when the measured capacity is 180 mAh/g.

**Q2:** Explain why LSTM RNN structures are superior to Feed-forward Neural Networks for cycle degradation prediction.

**Q3:** List two advantages of using EI (Expected Improvement) as the acquisition function in Bayesian optimization.

**Q4:** Predict the impact of changing the anode material from graphite to silicon in PyBaMM's DFN model.

**Q5:** Discuss the challenges of applying Transfer Learning to transfer knowledge from LIB cathode materials to Na-ion battery cathode materials (within 400 characters).

* * *

## References

  1. Sulzer, V. et al. "Python Battery Mathematical Modelling (PyBaMM)." _JOSS_ (2021).
  2. Severson, K. A. et al. "Data-driven prediction of battery cycle life." _Nat. Energy_ (2019).
  3. Chen, C. et al. "A Critical Review of Machine Learning of Energy Materials." _Adv. Energy Mater._ (2020).
  4. Attia, P. M. et al. "Closed-loop optimization of fast-charging protocols." _Nature_ (2020).

* * *

**Next Chapter** : [Chapter 4: Latest Case Studies and Industrial Applications in Battery Development](<chapter4-case-studies.html>)

**License** : This content is provided under the CC BY 4.0 license.
