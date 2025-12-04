---
title: "Chapter 5: Practical Applications of Time Series Analysis"
chapter_title: "Chapter 5: Practical Applications of Time Series Analysis"
subtitle: Anomaly Detection, Multivariate Forecasting, Causal Inference, and End-to-End Systems
reading_time: 30-35 min
difficulty: Advanced
code_examples: 9
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter focuses on practical applications of Practical Applications of Time Series Analysis. You will learn diverse methods for time series anomaly detection, Perform multivariate time series forecasting, and Conduct causal inference.

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Implement diverse methods for time series anomaly detection
  * ✅ Perform multivariate time series forecasting and Granger causality analysis
  * ✅ Conduct causal inference and intervention analysis
  * ✅ Perform advanced forecasting using Facebook Prophet
  * ✅ Build end-to-end forecasting systems

* * *

## 5.1 Time Series Anomaly Detection

### Overview of Anomaly Detection

**Time Series Anomaly Detection** is a technique for identifying data points that deviate from normal patterns.

> Anomaly detection has many practical applications, including early detection of system failures, fraud detection, and quality control.

### 1\. Statistical Methods (Z-score, IQR)
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Example: 1. Statistical Methods (Z-score, IQR)
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Generate sample data (normal pattern + anomalies)
    np.random.seed(42)
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Normal trend + seasonality
    trend = np.linspace(100, 150, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 5, n)
    normal_data = trend + seasonal + noise
    
    # Add anomalies
    data = normal_data.copy()
    anomaly_indices = [50, 120, 200, 280]
    data[anomaly_indices] = data[anomaly_indices] + np.array([40, -35, 50, -40])
    
    df = pd.DataFrame({'date': dates, 'value': data})
    df.set_index('date', inplace=True)
    
    # Anomaly detection using Z-score method
    z_scores = np.abs(stats.zscore(df['value']))
    threshold_z = 3
    anomalies_z = z_scores > threshold_z
    
    # Anomaly detection using IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies_iqr = (df['value'] < lower_bound) | (df['value'] > upper_bound)
    
    print("=== Statistical Anomaly Detection ===")
    print(f"Anomalies detected by Z-score method: {anomalies_z.sum()}")
    print(f"Anomalies detected by IQR method: {anomalies_iqr.sum()}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Z-score method
    axes[0].plot(df.index, df['value'], label='Data', alpha=0.7)
    axes[0].scatter(df.index[anomalies_z], df['value'][anomalies_z],
                    color='red', s=100, label='Anomalies', zorder=5)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Anomaly Detection with Z-score Method (threshold=3)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IQR method
    axes[1].plot(df.index, df['value'], label='Data', alpha=0.7)
    axes[1].scatter(df.index[anomalies_iqr], df['value'][anomalies_iqr],
                    color='red', s=100, label='Anomalies', zorder=5)
    axes[1].axhline(y=lower_bound, color='r', linestyle='--',
                    label=f'Lower bound: {lower_bound:.1f}')
    axes[1].axhline(y=upper_bound, color='r', linestyle='--',
                    label=f'Upper bound: {upper_bound:.1f}')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Anomaly Detection with IQR Method', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 2\. Anomaly Detection with Isolation Forest
    
    
    from sklearn.ensemble import IsolationForest
    
    # Feature engineering
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    df['rolling_std'] = df['value'].rolling(window=7).std()
    df['diff'] = df['value'].diff()
    
    # Remove missing values
    df_features = df[['value', 'rolling_mean', 'rolling_std', 'diff']].dropna()
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.05,  # Assume 5% anomaly rate
        random_state=42,
        n_estimators=100
    )
    
    # Calculate anomaly scores
    anomaly_labels = iso_forest.fit_predict(df_features)
    anomaly_scores = iso_forest.score_samples(df_features)
    
    # -1: anomaly, 1: normal
    anomalies_iso = anomaly_labels == -1
    
    print("\n=== Anomaly Detection with Isolation Forest ===")
    print(f"Anomalies detected: {anomalies_iso.sum()}")
    print(f"Anomaly rate: {anomalies_iso.sum() / len(df_features) * 100:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series and anomalies
    axes[0].plot(df_features.index, df_features['value'], alpha=0.7, label='Data')
    axes[0].scatter(df_features.index[anomalies_iso],
                    df_features['value'][anomalies_iso],
                    color='red', s=100, label='Anomalies', zorder=5)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Anomaly Detection with Isolation Forest', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Anomaly scores
    axes[1].plot(df_features.index, anomaly_scores, alpha=0.7)
    axes[1].scatter(df_features.index[anomalies_iso],
                    anomaly_scores[anomalies_iso],
                    color='red', s=50, label='Anomalies')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Scores (lower values indicate anomalies)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 3\. Anomaly Detection with LSTM Autoencoder
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - tensorflow>=2.13.0, <2.16.0
    
    """
    Example: 3. Anomaly Detection with LSTM Autoencoder
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    
    # Data preparation
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['value']].values)
    
    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    seq_length = 30
    sequences = create_sequences(data_scaled, seq_length)
    
    # Split into training and test data
    train_size = int(0.8 * len(sequences))
    train_seq = sequences[:train_size]
    test_seq = sequences[train_size:]
    
    # LSTM Autoencoder model
    input_dim = 1
    latent_dim = 16
    
    # Encoder
    encoder_inputs = keras.Input(shape=(seq_length, input_dim))
    x = layers.LSTM(32, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(latent_dim)(x)
    encoder = keras.Model(encoder_inputs, x, name='encoder')
    
    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.RepeatVector(seq_length)(decoder_inputs)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
    
    # Autoencoder
    autoencoder_inputs = keras.Input(shape=(seq_length, input_dim))
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_inputs, decoded, name='autoencoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Training
    print("\n=== LSTM Autoencoder Training ===")
    history = autoencoder.fit(
        train_seq, train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # Calculate reconstruction error
    train_pred = autoencoder.predict(train_seq, verbose=0)
    train_mse = np.mean(np.square(train_seq - train_pred), axis=(1, 2))
    
    test_pred = autoencoder.predict(test_seq, verbose=0)
    test_mse = np.mean(np.square(test_seq - test_pred), axis=(1, 2))
    
    # Set threshold for anomaly detection (99th percentile of training data)
    threshold = np.percentile(train_mse, 99)
    anomalies_ae = test_mse > threshold
    
    print(f"Training MSE range: [{train_mse.min():.4f}, {train_mse.max():.4f}]")
    print(f"Test MSE range: [{test_mse.min():.4f}, {test_mse.max():.4f}]")
    print(f"Anomaly detection threshold: {threshold:.4f}")
    print(f"Anomalies detected: {anomalies_ae.sum()} ({anomalies_ae.sum()/len(test_mse)*100:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Training curves
    axes[0].plot(history.history['loss'], label='Training loss')
    axes[0].plot(history.history['val_loss'], label='Validation loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('LSTM Autoencoder Training Curves', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction error
    axes[1].plot(test_mse, alpha=0.7, label='Reconstruction error')
    axes[1].scatter(np.where(anomalies_ae)[0], test_mse[anomalies_ae],
                    color='red', s=50, label='Anomalies', zorder=5)
    axes[1].axhline(y=threshold, color='r', linestyle='--',
                    label=f'Threshold: {threshold:.4f}')
    axes[1].set_xlabel('Test sample')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Reconstruction Error and Anomaly Detection', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 4\. Anomaly Detection with Prophet
    
    
    # Requirements:
    # - Python 3.9+
    # - prophet>=1.1.0
    
    """
    Example: 4. Anomaly Detection with Prophet
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from prophet import Prophet
    
    # Prepare dataframe for Prophet
    df_prophet = df[['value']].reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Train model
    model = Prophet(
        interval_width=0.99,  # 99% confidence interval
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df_prophet)
    
    # Forecast and confidence intervals
    forecast = model.predict(df_prophet)
    
    # Anomaly detection: actual values outside confidence interval
    anomalies_prophet = (
        (df_prophet['y'] < forecast['yhat_lower']) |
        (df_prophet['y'] > forecast['yhat_upper'])
    )
    
    print("\n=== Anomaly Detection with Prophet ===")
    print(f"Anomalies detected: {anomalies_prophet.sum()}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', alpha=0.5, label='Actual values')
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Predicted values')
    ax.fill_between(forecast['ds'],
                     forecast['yhat_lower'],
                     forecast['yhat_upper'],
                     alpha=0.3, label='99% confidence interval')
    ax.scatter(df_prophet['ds'][anomalies_prophet],
               df_prophet['y'][anomalies_prophet],
               color='red', s=100, label='Anomalies', zorder=5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Anomaly Detection with Prophet', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 5.2 Multivariate Time Series Forecasting

### What is Multivariate Time Series?

**Multivariate Time Series** deals with multiple interdependent time series data simultaneously.

### 1\. VAR (Vector AutoRegression) Model

The VAR model captures the interrelationships between multiple time series:

$$ \mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t $$
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: $$
    \mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1}
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    
    # Generate sample data (3-variate time series)
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Three interdependent time series
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = 0.8 * y1 + np.cumsum(np.random.normal(0, 0.5, n))
    y3 = 0.5 * y1 - 0.3 * y2 + np.cumsum(np.random.normal(0, 0.3, n))
    
    df_multi = pd.DataFrame({
        'y1': y1,
        'y2': y2,
        'y3': y3
    }, index=dates)
    
    print("=== Multivariate Time Series Data ===")
    print(df_multi.head())
    print(f"\nShape: {df_multi.shape}")
    
    # Check stationarity
    print("\n=== Stationarity Test (ADF test) ===")
    for col in df_multi.columns:
        result = adfuller(df_multi[col])
        print(f"{col}: p-value={result[1]:.4f} {'(stationary)' if result[1] < 0.05 else '(non-stationary)'}")
    
    # Make stationary by differencing
    df_diff = df_multi.diff().dropna()
    
    print("\n=== Stationarity Test After Differencing ===")
    for col in df_diff.columns:
        result = adfuller(df_diff[col])
        print(f"{col}: p-value={result[1]:.4f} {'(stationary)' if result[1] < 0.05 else '(non-stationary)'}")
    
    # VAR model order selection
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=10)
    print("\n=== VAR Model Order Selection ===")
    print(lag_order.summary())
    
    # Fit VAR model with optimal order
    optimal_lag = lag_order.aic
    var_model = model.fit(optimal_lag)
    print(f"\n=== VAR Model (order={optimal_lag}) ===")
    print(var_model.summary())
    
    # Forecast
    forecast_steps = 30
    forecast = var_model.forecast(df_diff.values[-optimal_lag:], steps=forecast_steps)
    forecast_dates = pd.date_range(df_multi.index[-1] + pd.Timedelta(days=1),
                                    periods=forecast_steps, freq='D')
    df_forecast = pd.DataFrame(forecast, index=forecast_dates, columns=df_multi.columns)
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, col in enumerate(df_multi.columns):
        # Original data (differenced)
        axes[i].plot(df_diff.index, df_diff[col], label='Actual values (differenced)', alpha=0.7)
        # Forecast
        axes[i].plot(df_forecast.index, df_forecast[col],
                     color='red', label='Forecast', linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].set_title(f'{col} Forecast', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()
    

### 2\. Granger Causality Test

**Granger Causality** tests whether one time series is useful for predicting another time series.
    
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    print("\n=== Granger Causality Test ===")
    
    # Test Granger causality for each pair
    max_lag = 5
    variables = df_diff.columns.tolist()
    
    for i, var1 in enumerate(variables):
        for var2 in variables:
            if var1 != var2:
                print(f"\n{var1} → {var2} causality:")
                test_data = df_diff[[var2, var1]]
                try:
                    result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
    
                    # Extract p-values
                    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                    min_p = min(p_values)
    
                    if min_p < 0.05:
                        print(f"  ✓ Causality exists (p-value={min_p:.4f})")
                    else:
                        print(f"  ✗ No causality (p-value={min_p:.4f})")
                except:
                    print("  Test failed")
    

### 3\. Multi-output Model
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # Create lagged features
    def create_lagged_features(df, lags):
        df_lagged = df.copy()
        for col in df.columns:
            for lag in range(1, lags + 1):
                df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df_lagged.dropna()
    
    # Create lagged features
    lags = 5
    df_lagged = create_lagged_features(df_multi, lags)
    
    # Separate features and targets
    target_cols = ['y1', 'y2', 'y3']
    feature_cols = [col for col in df_lagged.columns if col not in target_cols]
    
    X = df_lagged[feature_cols]
    y = df_lagged[target_cols]
    
    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Multi-output Random Forest
    multi_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    
    print("\n=== Multi-output Random Forest Training ===")
    multi_rf.fit(X_train, y_train)
    
    # Prediction
    y_pred = multi_rf.predict(X_test)
    
    # Evaluation
    print("\n=== Prediction Performance ===")
    for i, col in enumerate(target_cols):
        mse = mean_squared_error(y_test[col], y_pred[:, i])
        rmse = np.sqrt(mse)
        print(f"{col}: RMSE={rmse:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, col in enumerate(target_cols):
        axes[i].plot(y_test.index, y_test[col], label='Actual values', alpha=0.7)
        axes[i].plot(y_test.index, y_pred[:, i],
                     label='Predictions', alpha=0.7, linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].set_title(f'{col} Prediction (Multi-output RF)', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()
    

* * *

## 5.3 Causal Inference

### Fundamentals of Causal Inference

**Causal Inference** is a methodology for measuring the effects of interventions and policies.

### 1\. Intervention Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: 1. Intervention Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    
    # Generate sample data (with intervention)
    np.random.seed(42)
    n_pre = 200
    n_post = 100
    n_total = n_pre + n_post
    
    dates = pd.date_range('2020-01-01', periods=n_total, freq='D')
    
    # Pre-intervention data (baseline)
    baseline = 100 + np.cumsum(np.random.normal(0.1, 1, n_total))
    
    # Intervention effect (effect of +20 from day 200)
    intervention_effect = np.concatenate([
        np.zeros(n_pre),
        np.linspace(0, 20, 50),  # Effect gradually appears
        20 * np.ones(n_post - 50)
    ])
    
    # Observed values
    observed = baseline + intervention_effect + np.random.normal(0, 2, n_total)
    
    df_intervention = pd.DataFrame({
        'date': dates,
        'value': observed,
        'intervention': np.concatenate([np.zeros(n_pre), np.ones(n_post)])
    }, index=dates)
    
    # Train ARIMA model on pre-intervention data
    pre_intervention = df_intervention[:n_pre]['value']
    model = ARIMA(pre_intervention, order=(2, 1, 2))
    fitted_model = model.fit()
    
    # Forecast post-intervention (counterfactual: what would have happened without intervention)
    forecast = fitted_model.forecast(steps=n_post)
    forecast_index = df_intervention.index[n_pre:]
    
    # Estimate causal effect
    actual_post = df_intervention[n_pre:]['value']
    causal_effect = actual_post.values - forecast.values
    
    print("=== Intervention Analysis ===")
    print(f"Pre-intervention mean: {pre_intervention.mean():.2f}")
    print(f"Post-intervention mean (actual): {actual_post.mean():.2f}")
    print(f"Post-intervention mean (forecast): {forecast.mean():.2f}")
    print(f"Average causal effect: {causal_effect.mean():.2f}")
    print(f"Cumulative causal effect: {causal_effect.sum():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series and forecast
    axes[0].plot(df_intervention.index[:n_pre],
                 df_intervention['value'][:n_pre],
                 label='Pre-intervention (actual)', color='blue', alpha=0.7)
    axes[0].plot(df_intervention.index[n_pre:],
                 df_intervention['value'][n_pre:],
                 label='Post-intervention (actual)', color='green', alpha=0.7)
    axes[0].plot(forecast_index, forecast,
                 label='Counterfactual (no intervention forecast)', color='red',
                 linestyle='--', linewidth=2)
    axes[0].axvline(x=dates[n_pre], color='black', linestyle=':',
                    label='Intervention point', linewidth=2)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Intervention Analysis: Actual vs Counterfactual', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Causal effect
    axes[1].plot(forecast_index, causal_effect, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].fill_between(forecast_index, 0, causal_effect,
                          alpha=0.3, color='purple')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Causal effect')
    axes[1].set_title(f'Estimated Causal Effect (mean={causal_effect.mean():.2f})', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 2\. CausalImpact Analysis (pycausalimpact)
    
    
    from causalimpact import CausalImpact
    
    # Data preparation (including control variables)
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Control variables (related variables not affected by intervention)
    control1 = np.cumsum(np.random.normal(0, 1, n))
    control2 = np.cumsum(np.random.normal(0, 0.8, n))
    
    # Target variable (correlated with controls, effect after intervention)
    intervention_point = 200
    baseline_correlation = 0.7 * control1 + 0.5 * control2
    intervention_effect = np.concatenate([
        np.zeros(intervention_point),
        15 * np.ones(n - intervention_point)
    ])
    target = baseline_correlation + intervention_effect + np.random.normal(0, 3, n)
    
    df_causal = pd.DataFrame({
        'target': target,
        'control1': control1,
        'control2': control2
    }, index=dates)
    
    # CausalImpact analysis
    pre_period = [0, intervention_point - 1]
    post_period = [intervention_point, n - 1]
    
    ci = CausalImpact(df_causal, pre_period, post_period)
    
    print("\n=== CausalImpact Analysis ===")
    print(ci.summary())
    print("\n=== Detailed Report ===")
    print(ci.summary(output='report'))
    
    # Visualization
    ci.plot()
    plt.tight_layout()
    plt.show()
    

### 3\. Difference-in-Differences (DiD)
    
    
    # Difference-in-Differences (DID) example
    np.random.seed(42)
    n_time = 100
    intervention_time = 50
    
    # Treatment group
    treatment_pre = 50 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    treatment_post = 50 + np.cumsum(np.random.normal(0.1, 1, intervention_time)) + 20
    
    # Control group: no intervention effect
    control_pre = 45 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    control_post = 45 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    
    # DID estimation
    treatment_diff = treatment_post.mean() - treatment_pre.mean()
    control_diff = control_post.mean() - control_pre.mean()
    did_estimate = treatment_diff - control_diff
    
    print("\n=== Difference-in-Differences Analysis ===")
    print(f"Treatment group change: {treatment_diff:.2f}")
    print(f"Control group change: {control_diff:.2f}")
    print(f"DID estimate (intervention effect): {did_estimate:.2f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_points = np.arange(n_time)
    treatment_values = np.concatenate([treatment_pre, treatment_post])
    control_values = np.concatenate([control_pre, control_post])
    
    ax.plot(time_points[:intervention_time], treatment_pre,
            'b-', label='Treatment group (pre-intervention)', linewidth=2)
    ax.plot(time_points[intervention_time:], treatment_post,
            'b--', label='Treatment group (post-intervention)', linewidth=2)
    ax.plot(time_points[:intervention_time], control_pre,
            'r-', label='Control group (pre-intervention)', linewidth=2)
    ax.plot(time_points[intervention_time:], control_post,
            'r--', label='Control group (post-intervention)', linewidth=2)
    ax.axvline(x=intervention_time, color='black', linestyle=':',
               label='Intervention point', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Difference-in-Differences (DID estimate={did_estimate:.2f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 5.4 Prophet: Facebook Time Series Forecasting

### Features of Prophet

**Prophet** is a time series forecasting library developed by Facebook with the following features:

  * Automatically models trend, seasonality, and holiday effects
  * Robust to missing values and trend changes
  * Intuitive parameter tuning

### Prophet's Additive Model

$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$

  * $g(t)$: Trend (growth function)
  * $s(t)$: Seasonality (periodic variation)
  * $h(t)$: Holiday effect
  * $\epsilon_t$: Error term

### 1\. Basic Prophet Forecasting
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - prophet>=1.1.0
    
    """
    Example: 1. Basic Prophet Forecasting
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from prophet import Prophet
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    n = 730  # 2 years
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    
    # Trend + yearly seasonality + weekly seasonality
    trend = np.linspace(100, 200, n)
    yearly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 5, n)
    
    y = trend + yearly_seasonality + weekly_seasonality + noise
    
    df_prophet = pd.DataFrame({'ds': dates, 'y': y})
    
    # Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05  # Flexibility of trend changes
    )
    
    print("=== Prophet Model Training ===")
    model.fit(df_prophet)
    
    # Future forecast (90 days ahead)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    print("\n=== Forecast Results (last 5 rows) ===")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Visualization
    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast Results', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Component visualization
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.show()
    

### 2\. Adding Holiday Effects
    
    
    # Create holidays dataframe
    holidays = pd.DataFrame({
        'holiday': 'special_sale',
        'ds': pd.to_datetime(['2021-11-26', '2021-12-24', '2022-11-25', '2022-12-24']),
        'lower_window': -1,  # From day before holiday
        'upper_window': 1,   # To day after holiday
    })
    
    # Model with holiday effects
    model_holidays = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    
    # Add holiday effect to sample data
    y_with_holidays = y.copy()
    holiday_dates = holidays['ds'].values
    for date in holiday_dates:
        idx = np.where(dates == date)[0]
        if len(idx) > 0:
            y_with_holidays[idx[0]] += 50  # +50 effect on holidays
    
    df_prophet_holidays = pd.DataFrame({'ds': dates, 'y': y_with_holidays})
    
    print("\n=== Prophet Model with Holiday Effects ===")
    model_holidays.fit(df_prophet_holidays)
    
    # Forecast
    future_holidays = model_holidays.make_future_dataframe(periods=90)
    forecast_holidays = model_holidays.predict(future_holidays)
    
    # Visualization
    fig = model_holidays.plot(forecast_holidays)
    plt.title('Prophet Forecast with Holiday Effects', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Components (including holiday effect)
    fig_comp = model_holidays.plot_components(forecast_holidays)
    plt.tight_layout()
    plt.show()
    

### 3\. Changepoint Detection
    
    
    # Generate data with trend changes
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Trend change (slope changes at day 250)
    trend1 = np.linspace(100, 150, 250)
    trend2 = np.linspace(150, 120, 250)
    trend = np.concatenate([trend1, trend2])
    
    y = trend + 20 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.normal(0, 5, n)
    
    df_changepoint = pd.DataFrame({'ds': dates, 'y': y})
    
    # Model with changepoint detection enabled
    model_cp = Prophet(
        changepoint_prior_scale=0.5,  # Larger values allow more flexible changepoint detection
        n_changepoints=25  # Number of candidate changepoints
    )
    
    model_cp.fit(df_changepoint)
    
    # Get changepoints
    changepoints = model_cp.changepoints
    changepoint_dates = pd.to_datetime(changepoints)
    
    print("\n=== Detected Changepoints ===")
    print(f"Number of changepoints: {len(changepoint_dates)}")
    print(f"Major changepoints:")
    # Sort by magnitude of change
    deltas = model_cp.params['delta'].mean(axis=0)
    sorted_indices = np.argsort(np.abs(deltas))[-5:]  # Top 5
    for idx in sorted_indices:
        if idx < len(changepoint_dates):
            print(f"  {changepoint_dates[idx].date()}: Change magnitude={deltas[idx]:.3f}")
    
    # Forecast
    future = model_cp.make_future_dataframe(periods=60)
    forecast = model_cp.predict(future)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(15, 6))
    model_cp.plot(forecast, ax=ax)
    
    # Mark changepoints
    for cp in changepoint_dates:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.3)
    
    ax.set_title('Prophet Forecast with Changepoint Detection', fontsize=14)
    plt.tight_layout()
    plt.show()
    

* * *

## 5.5 End-to-End Forecasting System

### Overall Architecture of Forecasting System
    
    
    ```mermaid
    graph LR
        A[Data Acquisition] --> B[Preprocessing]
        B --> C[Feature Engineering]
        C --> D[Model Selection]
        D --> E[Training & Evaluation]
        E --> F[Prediction]
        F --> G[Monitoring]
        G --> H{Retraining needed?}
        H -->|Yes| B
        H -->|No| F
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#c8e6c9
        style G fill:#ffe0b2
        style H fill:#ffccbc
    ```

### Complete Forecasting Pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - prophet>=1.1.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from prophet import Prophet
    import warnings
    warnings.filterwarnings('ignore')
    
    class TimeSeriesPipeline:
        """End-to-end time series forecasting pipeline"""
    
        def __init__(self):
            self.models = {}
            self.best_model = None
            self.best_model_name = None
            self.best_score = float('inf')
            self.scaler = None
    
        def load_data(self, filepath=None, df=None):
            """Load data"""
            if df is not None:
                self.data = df
            else:
                self.data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            print(f"Data loaded: {self.data.shape}")
            return self
    
        def preprocess(self, target_col='value'):
            """Preprocessing"""
            self.target_col = target_col
    
            # Handle missing values
            self.data = self.data.interpolate(method='linear')
    
            # Handle outliers (IQR method)
            Q1 = self.data[target_col].quantile(0.25)
            Q3 = self.data[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            self.data[target_col] = self.data[target_col].clip(lower, upper)
    
            print("Preprocessing completed")
            return self
    
        def create_features(self, lags=[1, 2, 3, 7, 14, 30]):
            """Feature engineering"""
            df = self.data.copy()
    
            # Lag features
            for lag in lags:
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
    
            # Moving averages
            for window in [7, 14, 30]:
                df[f'ma_{window}'] = df[self.target_col].rolling(window=window).mean()
                df[f'std_{window}'] = df[self.target_col].rolling(window=window).std()
    
            # Time features
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
    
            # Differences
            df['diff_1'] = df[self.target_col].diff(1)
            df['diff_7'] = df[self.target_col].diff(7)
    
            # Remove missing values
            df = df.dropna()
    
            self.feature_data = df
            print(f"Feature engineering completed: {df.shape[1]} features")
            return self
    
        def prepare_train_test(self, test_size=0.2):
            """Prepare train-test split"""
            split_idx = int(len(self.feature_data) * (1 - test_size))
    
            self.train = self.feature_data[:split_idx]
            self.test = self.feature_data[split_idx:]
    
            # Features and targets
            feature_cols = [col for col in self.feature_data.columns
                           if col != self.target_col]
    
            self.X_train = self.train[feature_cols]
            self.y_train = self.train[self.target_col]
            self.X_test = self.test[feature_cols]
            self.y_test = self.test[self.target_col]
    
            print(f"Training data: {len(self.train)}, Test data: {len(self.test)}")
            return self
    
        def add_model(self, name, model):
            """Add model"""
            self.models[name] = model
            return self
    
        def train_and_evaluate(self):
            """Train and evaluate all models"""
            results = {}
    
            for name, model in self.models.items():
                print(f"\n=== Training {name} ===")
    
                # Training
                model.fit(self.X_train, self.y_train)
    
                # Prediction
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
    
                # Evaluation
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
    
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'predictions': y_pred_test
                }
    
                print(f"Training RMSE: {train_rmse:.4f}")
                print(f"Test RMSE: {test_rmse:.4f}")
                print(f"Test MAE: {test_mae:.4f}")
    
                # Update best model
                if test_rmse < self.best_score:
                    self.best_score = test_rmse
                    self.best_model = model
                    self.best_model_name = name
    
            self.results = results
            print(f"\nBest model: {self.best_model_name} (RMSE={self.best_score:.4f})")
            return self
    
        def cross_validate(self, n_splits=5):
            """Time series cross-validation"""
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_results = {}
    
            for name, model in self.models.items():
                scores = []
    
                for train_idx, val_idx in tscv.split(self.X_train):
                    X_cv_train = self.X_train.iloc[train_idx]
                    y_cv_train = self.y_train.iloc[train_idx]
                    X_cv_val = self.X_train.iloc[val_idx]
                    y_cv_val = self.y_train.iloc[val_idx]
    
                    model.fit(X_cv_train, y_cv_train)
                    y_pred = model.predict(X_cv_val)
                    rmse = np.sqrt(mean_squared_error(y_cv_val, y_pred))
                    scores.append(rmse)
    
                cv_results[name] = {
                    'mean_rmse': np.mean(scores),
                    'std_rmse': np.std(scores)
                }
    
            print("\n=== Cross-Validation Results ===")
            for name, result in cv_results.items():
                print(f"{name}: RMSE={result['mean_rmse']:.4f} (±{result['std_rmse']:.4f})")
    
            return cv_results
    
        def plot_results(self):
            """Visualize results"""
            n_models = len(self.results)
            fig, axes = plt.subplots(n_models + 1, 1, figsize=(15, 4 * (n_models + 1)))
    
            # Individual model predictions
            for i, (name, result) in enumerate(self.results.items()):
                ax = axes[i]
                ax.plot(self.test.index, self.y_test, label='Actual values', alpha=0.7)
                ax.plot(self.test.index, result['predictions'],
                       label=f'Predictions ({name})', alpha=0.7)
                ax.set_ylabel('Value')
                ax.set_title(f'{name}: RMSE={result["test_rmse"]:.4f}', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
    
            # All models comparison
            ax = axes[-1]
            ax.plot(self.test.index, self.y_test, label='Actual values',
                   linewidth=2, alpha=0.8, color='black')
            for name, result in self.results.items():
                ax.plot(self.test.index, result['predictions'],
                       label=name, alpha=0.6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('All Models Comparison', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def predict_future(self, steps=30):
            """Future forecasting"""
            # Use last features (simplified)
            last_features = self.X_test.iloc[-1:].copy()
    
            predictions = []
            for _ in range(steps):
                pred = self.best_model.predict(last_features)[0]
                predictions.append(pred)
    
                # Update features (simplified: only update lag features)
                # In practice, more sophisticated updates are needed
    
            future_dates = pd.date_range(
                self.test.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
    
            return pd.DataFrame({'date': future_dates, 'prediction': predictions})
    
    # Pipeline execution example
    print("=== End-to-End Forecasting Pipeline ===")
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    trend = np.linspace(100, 200, n)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    value = trend + seasonal + noise
    
    df_sample = pd.DataFrame({'value': value}, index=dates)
    
    # Execute pipeline
    pipeline = TimeSeriesPipeline()
    pipeline.load_data(df=df_sample)
    pipeline.preprocess(target_col='value')
    pipeline.create_features(lags=[1, 2, 3, 7, 14])
    pipeline.prepare_train_test(test_size=0.2)
    
    # Add models
    pipeline.add_model('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
    pipeline.add_model('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    pipeline.add_model('Ridge', Ridge(alpha=1.0))
    
    # Train and evaluate
    pipeline.train_and_evaluate()
    
    # Cross-validation
    pipeline.cross_validate(n_splits=5)
    
    # Visualize results
    pipeline.plot_results()
    
    # Future forecast
    future_forecast = pipeline.predict_future(steps=30)
    print("\n=== Future Forecast (30 days ahead) ===")
    print(future_forecast.head(10))
    

* * *

## 5.6 Chapter Summary

### What We Learned

  1. **Time Series Anomaly Detection**

     * Statistical methods (Z-score, IQR)
     * Machine learning methods (Isolation Forest)
     * Deep learning (LSTM Autoencoder)
     * Confidence interval-based detection with Prophet
  2. **Multivariate Time Series Forecasting**

     * Modeling interdependencies with VAR models
     * Granger causality testing
     * Multi-output machine learning models
  3. **Causal Inference**

     * Intervention analysis and counterfactual prediction
     * Effect measurement with CausalImpact
     * Difference-in-Differences method
  4. **Prophet**

     * Automatic modeling of trend, seasonality, and holiday effects
     * Changepoint detection
     * Intuitive parameter tuning
  5. **End-to-End Systems**

     * Complete forecasting pipeline construction
     * Automated model selection
     * Cross-validation and performance evaluation
     * Design for production deployment

### Practical Applications

Method | Application Examples | Key Points  
---|---|---  
**Anomaly Detection** | System monitoring, fraud detection | Combination of multiple methods  
**Multivariate Forecasting** | Demand forecasting, inventory management | Understanding causal relationships between variables  
**Causal Inference** | A/B testing, policy evaluation | Proper counterfactual setup  
**Prophet** | General business forecasting | Leveraging domain knowledge  
**E2E Systems** | Production forecasting systems | Monitoring and retraining  
  
* * *

## Exercises

### Exercise 1 (Difficulty: medium)

Explain the differences between Z-score and IQR methods for anomaly detection, and describe which cases each is suitable for.

Sample Answer

**Answer** :

**Z-score Method** :

  * Uses mean and standard deviation: $z = \frac{x - \mu}{\sigma}$
  * Typically considers $|z| > 3$ as anomalous
  * Assumes normal distribution

**IQR Method** :

  * Uses interquartile range: $IQR = Q3 - Q1$
  * Considers $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$ as anomalous
  * No distributional assumptions

**When to Use Each** :

Situation | Recommended Method | Reason  
---|---|---  
Near-normal distribution data | Z-score method | Statistically interpretable  
Skewed distribution | IQR method | Robust to outliers  
Small datasets | IQR method | Mean is unstable  
Extreme outliers present | IQR method | Median-based, robust  
  
### Exercise 2 (Difficulty: medium)

Can Granger causality testing prove true causal relationships? Answer with reasons.

Sample Answer

**Answer** :

**No, Granger causality cannot prove true causal relationships.**

**Reasons** :

  1. **Predictive Causality** :

     * Granger causality tests whether "past values of X are useful for predicting Y"
     * Predictability is different from causality
  2. **Third Variable Problem** :

     * A third variable Z that influences both X and Y may exist
     * Spurious correlation (confounding)
  3. **Possibility of Reverse Causation** :

     * Even if X→Y causality is detected, Y→X may also hold simultaneously
     * Cannot completely rule out bidirectional relationships
  4. **Assumption of Time Lag** :

     * Depends on selecting appropriate lag order
     * Cannot capture instantaneous causal relationships

**Correct Interpretation** :

Granger causality provides evidence that "X is useful for predicting Y," but proving true causal mechanisms requires experimental intervention or domain knowledge.

### Exercise 3 (Difficulty: hard)

For the following scenario, select an appropriate causal inference method and explain your reasoning:

"We want to implement a new advertising campaign in specific regions and measure its impact on sales. There are similar regions where the campaign was not implemented."

Sample Answer

**Answer** :

**Recommended Method** : Difference-in-Differences (DID) or CausalImpact

**Reasoning** :

  1. **Why DID is Suitable** :

     * Treatment group (campaign regions) and control group (non-campaign regions) exist
     * Pre- and post-intervention data available
     * Can remove temporal trend effects
     * Relatively simple assumption (parallel trends assumption)
  2. **Why CausalImpact is Suitable** :

     * When multiple control groups exist, can use them as control variables
     * Statistically estimates counterfactual (what would have happened without campaign)
     * Uncertainty evaluation through confidence intervals
     * Robust estimation with Bayesian structural time series models
  3. **Implementation Example (DID)** :

    
    
    # Treatment group: Sales in campaign regions
    # Control group: Sales in non-campaign regions
    # Intervention point: Campaign start date
    
    treatment_before = Pre-campaign treatment group average
    treatment_after = Post-campaign treatment group average
    control_before = Pre-campaign control group average
    control_after = Post-campaign control group average
    
    # DID estimator
    did_estimate = (treatment_after - treatment_before) - (control_after - control_before)
    
    # did_estimate is the causal effect of the campaign
    

**Important Considerations** :

  * Parallel trends assumption: Assume both groups would have same trend without intervention
  * Regional similarity: Ensure control group is as similar as possible to treatment group
  * External factors: Consider effects of other concurrent events

### Exercise 4 (Difficulty: hard)

In Prophet models, what problems occur when there are too many or too few changepoints? Also explain appropriate parameter tuning methods.

Sample Answer

**Answer** :

**When There Are Too Many Changepoints (Overfitting)** :

  * Problem: Misidentifies noise as trend changes
  * Result: Overfits to training data, poor generalization
  * Forecast: Future forecasts unstable and unreliable

**When There Are Too Few Changepoints (Underfitting)** :

  * Problem: Cannot capture true trend changes
  * Result: Model too simple, misses important patterns
  * Forecast: Systematic errors, reduced forecast accuracy

**Appropriate Parameter Tuning** :

  1. **changepoint_prior_scale** (changepoint flexibility):

    
    
    # Default: 0.05
    # Small values (0.001-0.01): Suppress changes, smoother
    # Large values (0.1-0.5): Allow changes, more flexible
    
    # Tuning example
    model_smooth = Prophet(changepoint_prior_scale=0.01)  # Conservative
    model_flexible = Prophet(changepoint_prior_scale=0.5)  # Flexible
    

  2. **n_changepoints** (number of candidate changepoints):

    
    
    # Default: 25
    # Adjust according to data length
    
    model = Prophet(n_changepoints=50)  # For long time series
    

  3. **Optimization via Cross-Validation** :

    
    
    from prophet.diagnostics import cross_validation, performance_metrics
    
    # Parameter candidates
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    }
    
    best_params = None
    best_rmse = float('inf')
    
    for scale in param_grid['changepoint_prior_scale']:
        model = Prophet(changepoint_prior_scale=scale)
        model.fit(df)
    
        # Cross-validation
        df_cv = cross_validation(model, initial='730 days',
                                 period='180 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
    
        rmse = df_p['rmse'].mean()
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'changepoint_prior_scale': scale}
    
    print(f"Optimal parameters: {best_params}")
    print(f"RMSE: {best_rmse}")
    

**Selection Guidelines** :

Data Characteristics | changepoint_prior_scale  
---|---  
Stable trend | 0.001 - 0.01  
Normal business data | 0.05 (default)  
Frequent trend changes | 0.1 - 0.5  
Uncertain cases | Determine via cross-validation  
  
### Exercise 5 (Difficulty: hard)

Design a mechanism to detect model drift (performance degradation) and trigger automatic retraining in an end-to-end forecasting system.

Sample Answer

**Answer** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    
    class ModelMonitoring:
        """Model drift detection and automatic retraining system"""
    
        def __init__(self, model, threshold_rmse_increase=0.2,
                     window_size=30, retrain_frequency=90):
            """
            Parameters:
            -----------
            model: Prediction model
            threshold_rmse_increase: RMSE increase threshold (retrain at 20% increase)
            window_size: Monitoring window (last 30 days)
            retrain_frequency: Minimum retraining interval (90 days)
            """
            self.model = model
            self.threshold = threshold_rmse_increase
            self.window_size = window_size
            self.retrain_frequency = retrain_frequency
    
            # Monitoring metrics
            self.baseline_rmse = None
            self.current_errors = []
            self.last_retrain_date = None
            self.retrain_history = []
    
        def set_baseline(self, X_val, y_val):
            """Set baseline performance"""
            y_pred = self.model.predict(X_val)
            self.baseline_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            print(f"Baseline RMSE: {self.baseline_rmse:.4f}")
            return self
    
        def monitor_prediction(self, X_new, y_true, date):
            """Monitor new predictions"""
            # Prediction
            y_pred = self.model.predict(X_new)
    
            # Record error
            error = np.abs(y_true - y_pred)
            self.current_errors.append({
                'date': date,
                'error': error,
                'y_true': y_true,
                'y_pred': y_pred
            })
    
            # Remove old data beyond window size
            if len(self.current_errors) > self.window_size:
                self.current_errors.pop(0)
    
            # Drift detection
            if len(self.current_errors) >= self.window_size:
                drift_detected = self._detect_drift()
    
                if drift_detected:
                    print(f"\n⚠️ Model drift detected (date: {date})")
    
                    # Check if retraining needed
                    if self._should_retrain(date):
                        print("🔄 Starting automatic retraining")
                        return True  # Retraining needed
                    else:
                        print(f"Waiting: less than {self.retrain_frequency} days since last retraining")
    
            return False  # No retraining needed
    
        def _detect_drift(self):
            """Detect drift"""
            # Recent window RMSE
            recent_errors = [e['error'] for e in self.current_errors]
            current_rmse = np.sqrt(np.mean(np.square(recent_errors)))
    
            # Compare with baseline
            rmse_increase = (current_rmse - self.baseline_rmse) / self.baseline_rmse
    
            print(f"Current RMSE: {current_rmse:.4f} (increase rate: {rmse_increase*100:.1f}%)")
    
            # Drift detected if threshold exceeded
            return rmse_increase > self.threshold
    
        def _should_retrain(self, current_date):
            """Determine if retraining should occur"""
            if self.last_retrain_date is None:
                return True
    
            days_since_retrain = (current_date - self.last_retrain_date).days
            return days_since_retrain >= self.retrain_frequency
    
        def retrain(self, X_train, y_train, X_val, y_val, date):
            """Retrain model"""
            # Execute retraining
            self.model.fit(X_train, y_train)
    
            # Set new baseline
            self.set_baseline(X_val, y_val)
    
            # Record retraining history
            self.last_retrain_date = date
            self.retrain_history.append({
                'date': date,
                'new_baseline_rmse': self.baseline_rmse
            })
    
            # Clear error history
            self.current_errors = []
    
            print(f"✅ Retraining completed (new baseline RMSE: {self.baseline_rmse:.4f})")
    
        def get_monitoring_report(self):
            """Monitoring report"""
            report = {
                'baseline_rmse': self.baseline_rmse,
                'num_retrains': len(self.retrain_history),
                'last_retrain': self.last_retrain_date,
                'retrain_history': self.retrain_history
            }
            return report
    
    
    # Usage example
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data (with drift)
    np.random.seed(42)
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Initially stable
    data1 = 100 + np.cumsum(np.random.normal(0, 1, 200))
    # Drift from day 200 (trend change)
    data2 = data1[-1] + np.cumsum(np.random.normal(0.5, 2, n - 200))
    data = np.concatenate([data1, data2])
    
    # Features and target (simplified)
    X = np.arange(n).reshape(-1, 1)
    y = data
    
    # Initial training
    train_size = 150
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:200], y[train_size:200]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Monitoring system
    monitor = ModelMonitoring(
        model=model,
        threshold_rmse_increase=0.2,
        window_size=30,
        retrain_frequency=90
    )
    monitor.set_baseline(X_val, y_val)
    
    # Online prediction and monitoring
    print("\n=== Online Monitoring Started ===")
    for i in range(200, n):
        X_new = X[i:i+1]
        y_true = y[i]
        date = dates[i]
    
        # Monitoring
        needs_retrain = monitor.monitor_prediction(X_new, y_true, date)
    
        # If retraining needed
        if needs_retrain:
            # Retraining data (use recent data)
            retrain_start = max(0, i - 150)
            X_retrain = X[retrain_start:i]
            y_retrain = y[retrain_start:i]
            X_val_new = X[i-50:i]
            y_val_new = y[i-50:i]
    
            monitor.retrain(X_retrain, y_retrain, X_val_new, y_val_new, date)
    
    # Monitoring report
    print("\n=== Monitoring Report ===")
    report = monitor.get_monitoring_report()
    print(f"Baseline RMSE: {report['baseline_rmse']:.4f}")
    print(f"Number of retraining: {report['num_retrains']}")
    print(f"Last retraining date: {report['last_retrain']}")
    print("\nRetraining history:")
    for h in report['retrain_history']:
        print(f"  {h['date'].date()}: RMSE={h['new_baseline_rmse']:.4f}")
    

**Design Key Points** :

  1. **Drift Detection Metrics** : 
     * RMSE increase rate (performance degradation)
     * Prediction error distribution changes (KS test, etc.)
     * Feature distribution changes
  2. **Retraining Strategy** : 
     * Periodic retraining (time-based)
     * Performance-based retraining (when drift detected)
     * Hybrid (both conditions)
  3. **Implementation Considerations** : 
     * A/B testing: Run old and new models in parallel
     * Rollback functionality: Handle performance degradation after retraining
     * Alert functionality: For cases requiring human intervention

* * *

## References

  1. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. _The American Statistician_ , 72(1), 37-45.
  2. Brodersen, K. H., et al. (2015). Inferring causal impact using Bayesian structural time-series models. _Annals of Applied Statistics_ , 9(1), 247-274.
  3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. _ACM Computing Surveys_ , 41(3), 1-58.
  4. Lütkepohl, H. (2005). _New Introduction to Multiple Time Series Analysis_. Springer.
  5. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts.
  6. Pearl, J., & Mackenzie, D. (2018). _The Book of Why: The New Science of Cause and Effect_. Basic Books.
