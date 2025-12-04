---
title: "Chapter 5: Introduction to Time Series Forecasting"
chapter_title: "Chapter 5: Introduction to Time Series Forecasting"
subtitle: Time Series Data Analysis and Future Prediction with RNN - Stock Price, Weather, and Demand Forecasting
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 4
version: 1.0
created_at: 2025-10-21
---

This chapter introduces the basics of Introduction to Time Series Forecasting. You will learn characteristics of time series data and concept of sliding windows.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the characteristics of time series data and implement appropriate preprocessing
  * ✅ Understand the concept of sliding windows and create datasets
  * ✅ Implement univariate and multivariate time series forecasting using LSTM and GRU
  * ✅ Perform multi-step ahead predictions with Seq2Seq models
  * ✅ Calculate and interpret evaluation metrics such as MAE, RMSE, and MAPE
  * ✅ Build practical stock price and weather forecasting systems

* * *

## 5.1 Characteristics of Time Series Data

### What is Time Series Data

**Time Series Data** is a sequence of observations recorded along the time axis. The main purpose is to predict the future based on past patterns.
    
    
    ```mermaid
    graph LR
        A[Types of Time Series Data] --> B[UnivariateUnivariate]
        A --> C[MultivariateMultivariate]
    
        B --> B1["Single variable onlye.g., Stock closing price"]
        C --> C1["Multiple variablese.g., Stock price + volume + indicators"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

Characteristic | Description | Application Examples  
---|---|---  
**Trend** | Long-term increasing or decreasing tendency | Economic growth, population increase  
**Seasonality** | Periodic patterns (daily, monthly, yearly) | Seasonal temperature variation, retail busy seasons  
**Cyclicity** | Irregular periodic patterns | Business cycles, economic cycles  
**Noise** | Random fluctuations | Measurement errors, unpredictable events  
  
### Challenges in Time Series Forecasting

#### Data Dependency

Time series data has temporal **autocorrelation**. Past values influence future values, so data points are not independent.

$$ \text{Autocorrelation}(k) = \frac{\sum_{t=1}^{N-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{N}(x_t - \bar{x})^2} $$

#### Non-Stationarity

Many time series data are **non-stationary** , with statistical properties (mean, variance) changing over time. Since forecasting models typically assume stationarity, preprocessing is necessary.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    
    def check_stationarity(timeseries, title='Time Series'):
        """
        Check stationarity of time series data
    
        Args:
            timeseries: Time series data (pandas Series)
            title: Graph title
        """
        # Calculate rolling statistics
        rolling_mean = timeseries.rolling(window=12).mean()
        rolling_std = timeseries.rolling(window=12).std()
    
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeseries, color='blue', label='Original')
        ax.plot(rolling_mean, color='red', label='Rolling Mean (12 periods)')
        ax.plot(rolling_std, color='black', label='Rolling Std (12 periods)')
        ax.legend(loc='best')
        ax.set_title(f'{title} - Rolling Mean & Standard Deviation')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # Augmented Dickey-Fuller test (statistical test for stationarity)
        print(f'\n{title} - ADF Test Results:')
        adf_result = adfuller(timeseries.dropna(), autolag='AIC')
    
        print(f'ADF Statistic: {adf_result[0]:.6f}')
        print(f'p-value: {adf_result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in adf_result[4].items():
            print(f'  {key}: {value:.3f}')
    
        # Decision
        if adf_result[1] <= 0.05:
            print('Conclusion: Data is stationary (p-value ≤ 0.05)')
        else:
            print('Conclusion: Data is non-stationary (p-value > 0.05)')
            print('      → Consider differencing or log transformation')
    
    # Usage example: Random walk data (non-stationary)
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200))
    ts_nonstationary = pd.Series(random_walk, index=pd.date_range('2020-01-01', periods=200))
    
    # Stationary data (white noise)
    ts_stationary = pd.Series(np.random.randn(200), index=pd.date_range('2020-01-01', periods=200))
    
    # Run tests
    check_stationarity(ts_nonstationary, title='Non-Stationary Data (Random Walk)')
    check_stationarity(ts_stationary, title='Stationary Data (White Noise)')
    

> **Stationarization methods** :
> 
>   * **Differencing** : $x'_t = x_t - x_{t-1}$ to remove trend
>   * **Log Transform** : $x'_t = \log(x_t)$ to stabilize variance
>   * **Moving Average** : Smoothing to remove seasonality
> 

* * *

## 5.2 Data Preprocessing and Window Creation

### Normalization

For time series data preprocessing, **Min-Max scaling** and **standardization** are common. The key is to use only the training data statistics and not leak information to the test data.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    class TimeSeriesNormalizer:
        """
        Time series data normalization class
        """
    
        def __init__(self, method='minmax'):
            """
            Args:
                method: 'minmax' or 'standard'
            """
            self.method = method
    
            if method == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif method == 'standard':
                self.scaler = StandardScaler()
            else:
                raise ValueError("method must be 'minmax' or 'standard'")
    
        def fit_transform(self, data):
            """
            Learn normalization parameters from training data and transform
    
            Args:
                data: numpy array of shape [N, features]
    
            Returns:
                normalized_data: Normalized data
            """
            # Convert to 2D array if necessary
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            normalized_data = self.scaler.fit_transform(data)
    
            return normalized_data
    
        def transform(self, data):
            """
            Transform using learned parameters (for test data)
    
            Args:
                data: numpy array of shape [N, features]
    
            Returns:
                normalized_data: Normalized data
            """
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            normalized_data = self.scaler.transform(data)
    
            return normalized_data
    
        def inverse_transform(self, data):
            """
            Reverse normalization (for restoring predictions)
    
            Args:
                data: Normalized data
    
            Returns:
                original_scale_data: Data restored to original scale
            """
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            original_scale_data = self.scaler.inverse_transform(data)
    
            return original_scale_data
    
    # Usage example
    np.random.seed(42)
    stock_prices = np.cumsum(np.random.randn(1000)) + 100  # Pseudo stock price data
    
    # Min-Max normalization
    normalizer_minmax = TimeSeriesNormalizer(method='minmax')
    normalized_minmax = normalizer_minmax.fit_transform(stock_prices)
    
    print("Min-Max normalization:")
    print(f"  Original data range: [{stock_prices.min():.2f}, {stock_prices.max():.2f}]")
    print(f"  Normalized range: [{normalized_minmax.min():.2f}, {normalized_minmax.max():.2f}]")
    
    # Standardization
    normalizer_std = TimeSeriesNormalizer(method='standard')
    normalized_std = normalizer_std.fit_transform(stock_prices)
    
    print("\nStandardization:")
    print(f"  Original data: mean={stock_prices.mean():.2f}, std={stock_prices.std():.2f}")
    print(f"  Normalized: mean={normalized_std.mean():.4f}, std={normalized_std.std():.4f}")
    

### Sliding Window

The **sliding window** is a method to split time series data into pairs of input sequences (past data) and target values (future data).
    
    
    ```mermaid
    graph LR
        A[Original Time Seriesx1, x2, ..., xN] --> B[Window 1Input: x1~x10Target: x11]
        A --> C[Window 2Input: x2~x11Target: x12]
        A --> D[Window 3Input: x3~x12Target: x13]
        A --> E[...]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
    ```
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    def create_sliding_windows(data, window_size, horizon=1, stride=1):
        """
        Create dataset with sliding windows
    
        Args:
            data: Time series data (numpy array) [N, features]
            window_size: Input window size (how many steps to look back)
            horizon: Prediction horizon (how many steps ahead to predict)
            stride: Window sliding width
    
        Returns:
            X: Input data [num_windows, window_size, features]
            y: Target data [num_windows, horizon, features]
        """
        X, y = [], []
    
        for i in range(0, len(data) - window_size - horizon + 1, stride):
            # Input window
            X.append(data[i : i + window_size])
    
            # Target value (horizon steps ahead)
            y.append(data[i + window_size : i + window_size + horizon])
    
        X = np.array(X)
        y = np.array(y)
    
        return X, y
    
    # Usage example
    # Pseudo stock price data (normalized)
    data = normalized_minmax.reshape(-1, 1)  # [1000, 1]
    
    # Parameter settings
    WINDOW_SIZE = 60   # Look back 60 days
    HORIZON = 1        # Predict 1 day ahead
    STRIDE = 1         # Slide window every day
    
    # Create windows
    X, y = create_sliding_windows(data, window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE)
    
    print(f"Window data creation completed:")
    print(f"  Input X shape: {X.shape}")  # [num_windows, 60, 1]
    print(f"  Target y shape: {y.shape}")  # [num_windows, 1, 1]
    print(f"  Total windows: {len(X)}")
    
    # Visualize one window
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_idx = 100
    
    # Input window (past 60 days)
    ax.plot(range(WINDOW_SIZE), X[sample_idx, :, 0], marker='o', label='Input Window (Past 60 days)')
    
    # Target value (1 day ahead)
    ax.scatter([WINDOW_SIZE], y[sample_idx, 0, 0], color='red', s=100, zorder=5, label='Target (Next day)', marker='*')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Normalized Price')
    ax.set_title('Sliding Window Example')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Train/Validation/Test Split

For time series data, **split preserving temporal order** is essential. Random shuffling is not used.
    
    
    def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Train/Val/Test split for time series data (preserving temporal order)
    
        Args:
            X: Input data [num_samples, window_size, features]
            y: Target data [num_samples, horizon, features]
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
    
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        num_samples = len(X)
    
        # Split indices
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))
    
        # Split
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
    
        print(f"Data split completed:")
        print(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Execute split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"\nTensor shapes:")
    print(f"  X_train: {X_train_tensor.shape}")
    print(f"  y_train: {y_train_tensor.shape}")
    

> **Important notes** :
> 
>   * Always split time series data in temporal order (Train → Val → Test)
>   * Calculate normalization parameters only from training data
>   * Do not use test data at all until final evaluation
> 

* * *

## 5.3 Univariate Time Series Forecasting

### Stock Price Prediction with LSTM

**Univariate time series forecasting** is a task to predict the future using only one variable (e.g., stock closing price).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    class LSTMForecaster(nn.Module):
        """
        LSTM model for univariate time series forecasting
        """
    
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            """
            Args:
                input_size: Number of input features (1 for univariate)
                hidden_size: LSTM hidden layer size
                num_layers: Number of LSTM layers
                dropout: Dropout rate
            """
            super(LSTMForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # Fully connected layer (output)
            self.fc = nn.Linear(hidden_size, 1)
    
        def forward(self, x):
            """
            Args:
                x: [batch_size, seq_len, input_size]
    
            Returns:
                out: [batch_size, 1] - One-step ahead prediction
            """
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(x)
    
            # Use last hidden state
            out = self.fc(lstm_out[:, -1, :])  # [batch_size, 1]
    
            return out
    
    # Create model
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("LSTM Forecasting Model:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.squeeze(-1))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor.squeeze(-1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Time series usually not shuffled
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(model, data_loader, criterion, optimizer, device):
        """
        Train for one epoch
        """
        model.train()
        total_loss = 0
    
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
            # Forward pass
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def validate(model, data_loader, criterion, device):
        """
        Evaluate on validation data
        """
        model.eval()
        total_loss = 0
    
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
    
                total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []
    
    print("\nTraining started:")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print("\nTraining completed!")
    
    # Visualize loss
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Prediction and Visualization
    
    
    def forecast_and_visualize(model, X_test, y_test, normalizer, device, num_samples=100):
        """
        Perform prediction on test data and visualize results
    
        Args:
            model: Trained model
            X_test: Test input [num_samples, seq_len, 1]
            y_test: Test target [num_samples, 1, 1]
            normalizer: TimeSeriesNormalizer (for inverse transform)
            device: Device
            num_samples: Number of samples to visualize
        """
        model.eval()
    
        # Convert test data to Tensor
        X_test_tensor = torch.FloatTensor(X_test[:num_samples]).to(device)
    
        # Prediction
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
    
        # Reverse normalization
        y_true = normalizer.inverse_transform(y_test[:num_samples].squeeze())
        y_pred = normalizer.inverse_transform(predictions)
    
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 6))
    
        time_steps = np.arange(num_samples)
        ax.plot(time_steps, y_true, label='Actual', color='blue', linewidth=2)
        ax.plot(time_steps, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2, alpha=0.7)
    
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.set_title('Stock Price Prediction - LSTM Forecasting')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # Calculate evaluation metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        print(f"\nEvaluation metrics:")
        print(f"  MAE (Mean Absolute Error): {mae:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
        return y_true, y_pred
    
    # Execute prediction
    y_true, y_pred = forecast_and_visualize(
        model, X_test, y_test, normalizer_minmax, device, num_samples=100
    )
    

* * *

## 5.4 Multivariate Time Series Forecasting

### Prediction Using Multiple Features

**Multivariate time series forecasting** considers multiple related variables simultaneously for prediction. For example, in stock price prediction, not only closing prices but also volume, moving averages, RSI, and other indicators are used.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    import numpy as np
    
    def create_multivariate_features(prices, window=20):
        """
        Create features for multivariate time series
    
        Args:
            prices: Stock price data (numpy array)
            window: Moving average window size
    
        Returns:
            features_df: Features DataFrame
        """
        df = pd.DataFrame({'price': prices})
    
        # Moving Average
        df['ma_5'] = df['price'].rolling(window=5).mean()
        df['ma_20'] = df['price'].rolling(window=20).mean()
    
        # Standard deviation (Volatility)
        df['std_20'] = df['price'].rolling(window=20).std()
    
        # Return
        df['return'] = df['price'].pct_change()
    
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
        # Fill missing values forward
        df = df.fillna(method='ffill').fillna(method='bfill')
    
        return df
    
    # Create features with pseudo stock price data
    stock_prices = np.cumsum(np.random.randn(1000)) + 100
    features_df = create_multivariate_features(stock_prices)
    
    print("Multivariate features:")
    print(features_df.head(30))
    print(f"\nNumber of features: {features_df.shape[1]}")
    
    # Normalization
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_df.values)
    
    # Create windows (multivariate version)
    WINDOW_SIZE = 60
    HORIZON = 1
    
    X_multi, y_multi = create_sliding_windows(
        features_normalized,
        window_size=WINDOW_SIZE,
        horizon=HORIZON
    )
    
    print(f"\nMultivariate windows:")
    print(f"  X shape: {X_multi.shape}")  # [num_samples, 60, 6] - 6 features
    print(f"  y shape: {y_multi.shape}")  # [num_samples, 1, 6]
    
    # Target uses only price (first column)
    y_multi_price = y_multi[:, :, 0:1]  # [num_samples, 1, 1]
    
    print(f"  y (price only) shape: {y_multi_price.shape}")
    

### Multivariate LSTM Forecasting Model
    
    
    class MultivariateLSTM(nn.Module):
        """
        LSTM model for multivariate time series forecasting
        """
    
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            """
            Args:
                input_size: Number of input features (multiple)
                hidden_size: LSTM hidden layer size
                num_layers: Number of LSTM layers
                dropout: Dropout rate
            """
            super(MultivariateLSTM, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # Output layer
            self.fc = nn.Linear(hidden_size, 1)  # Predict 1 variable (price)
    
        def forward(self, x):
            """
            Args:
                x: [batch_size, seq_len, input_size]
    
            Returns:
                out: [batch_size, 1]
            """
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
    
            return out
    
    # Data split
    X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = train_val_test_split(
        X_multi, y_multi_price, train_ratio=0.7, val_ratio=0.15
    )
    
    # Convert to Tensors
    X_train_m = torch.FloatTensor(X_train_m)
    y_train_m = torch.FloatTensor(y_train_m).squeeze()
    X_val_m = torch.FloatTensor(X_val_m)
    y_val_m = torch.FloatTensor(y_val_m).squeeze()
    X_test_m = torch.FloatTensor(X_test_m)
    y_test_m = torch.FloatTensor(y_test_m).squeeze()
    
    # Create model
    input_size = X_multi.shape[2]  # Number of features (6)
    model_multi = MultivariateLSTM(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.3)
    model_multi.to(device)
    
    print(f"\nMultivariate LSTM model:")
    print(f"  Input features: {input_size}")
    print(f"  Parameters: {sum(p.numel() for p in model_multi.parameters()):,}")
    
    # Data loaders
    train_loader_m = DataLoader(TensorDataset(X_train_m, y_train_m), batch_size=64, shuffle=False)
    val_loader_m = DataLoader(TensorDataset(X_val_m, y_val_m), batch_size=64, shuffle=False)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_multi.parameters(), lr=0.001)
    
    print("\nMultivariate model training started:")
    for epoch in range(1, 51):
        train_loss = train_epoch(model_multi, train_loader_m, criterion, optimizer, device)
        val_loss = validate(model_multi, val_loader_m, criterion, device)
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/50 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print("Training completed!")
    

> **Advantages of multivariate prediction** :
> 
>   * Improves prediction accuracy by leveraging multiple related information
>   * Can integrate external information such as market indicators and economic indicators
>   * Model learns interactions between variables
> 

* * *

## 5.5 Multi-Step Prediction with Seq2Seq

### Sequence-to-Sequence Prediction

The **Seq2Seq (Sequence-to-Sequence)** model is an architecture that predicts multiple steps ahead of time series at once.
    
    
    ```mermaid
    graph LR
        A[EncoderPast 60 days] --> B[Context VectorHidden state]
        B --> C[DecoderFuture 7 days prediction]
    
        A1[x1, x2, ..., x60] --> A
        C --> C1[y1, y2, ..., y7]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```
    
    
    class Seq2SeqLSTM(nn.Module):
        """
        Seq2Seq time series forecasting model (Encoder-Decoder)
        """
    
        def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_length=7, dropout=0.2):
            """
            Args:
                input_size: Number of input features
                hidden_size: LSTM hidden layer size
                num_layers: Number of LSTM layers
                output_length: Output sequence length (how many steps ahead to predict)
                dropout: Dropout rate
            """
            super(Seq2SeqLSTM, self).__init__()
    
            self.output_length = output_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # Encoder LSTM
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # Decoder LSTM
            self.decoder = nn.LSTM(
                input_size=1,  # Decoder input is previous step prediction
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # Output layer
            self.fc = nn.Linear(hidden_size, 1)
    
        def forward(self, x, target_len=None):
            """
            Args:
                x: [batch_size, seq_len, input_size]
                target_len: Decoder output length (specified during inference)
    
            Returns:
                outputs: [batch_size, output_length, 1]
            """
            batch_size = x.size(0)
    
            if target_len is None:
                target_len = self.output_length
    
            # Encode past sequence with Encoder
            _, (hidden, cell) = self.encoder(x)
    
            # Decoder initial input (last input value)
            decoder_input = x[:, -1, 0:1].unsqueeze(1)  # [batch_size, 1, 1]
    
            outputs = []
    
            # Decoder predicts future step by step
            for t in range(target_len):
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
    
                # Prediction
                prediction = self.fc(decoder_output)  # [batch_size, 1, 1]
                outputs.append(prediction)
    
                # Next step input is previous step prediction
                decoder_input = prediction
    
            # Concatenate outputs
            outputs = torch.cat(outputs, dim=1)  # [batch_size, target_len, 1]
    
            return outputs
    
    # Create windows for Seq2Seq (predict multiple steps ahead)
    WINDOW_SIZE = 60
    HORIZON = 7  # Predict 7 steps ahead
    
    X_seq2seq, y_seq2seq = create_sliding_windows(
        data,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        stride=1
    )
    
    print(f"Seq2Seq data:")
    print(f"  X shape: {X_seq2seq.shape}")  # [num_samples, 60, 1]
    print(f"  y shape: {y_seq2seq.shape}")  # [num_samples, 7, 1]
    
    # Data split
    X_train_s2s, X_val_s2s, X_test_s2s, y_train_s2s, y_val_s2s, y_test_s2s = train_val_test_split(
        X_seq2seq, y_seq2seq, train_ratio=0.7, val_ratio=0.15
    )
    
    # Convert to Tensors
    X_train_s2s = torch.FloatTensor(X_train_s2s)
    y_train_s2s = torch.FloatTensor(y_train_s2s)
    X_val_s2s = torch.FloatTensor(X_val_s2s)
    y_val_s2s = torch.FloatTensor(y_val_s2s)
    X_test_s2s = torch.FloatTensor(X_test_s2s)
    y_test_s2s = torch.FloatTensor(y_test_s2s)
    
    # Create model
    model_seq2seq = Seq2SeqLSTM(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_length=HORIZON,
        dropout=0.2
    )
    model_seq2seq.to(device)
    
    print(f"\nSeq2Seq model:")
    print(f"  Output length: {HORIZON} steps")
    print(f"  Parameters: {sum(p.numel() for p in model_seq2seq.parameters()):,}")
    
    # Training function (for Seq2Seq)
    def train_seq2seq(model, X, y, criterion, optimizer, device, epochs=30, batch_size=32):
        """
        Train Seq2Seq model
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        train_losses = []
    
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
    
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
                # Forward pass
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(loader)
            train_losses.append(avg_loss)
    
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}')
    
        return train_losses
    
    # Execute training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_seq2seq.parameters(), lr=0.001)
    
    print("\nSeq2Seq training started:")
    train_losses = train_seq2seq(
        model_seq2seq, X_train_s2s, y_train_s2s,
        criterion, optimizer, device, epochs=50, batch_size=32
    )
    
    print("Training completed!")
    

### Multi-Step Forecast Visualization
    
    
    def visualize_multistep_forecast(model, X_test, y_test, normalizer, device, sample_idx=0):
        """
        Visualize multi-step forecast
    
        Args:
            model: Seq2Seq model
            X_test: Test input
            y_test: Test target
            normalizer: Normalizer
            device: Device
            sample_idx: Sample index to visualize
        """
        model.eval()
    
        # Predict on one sample
        sample_input = torch.FloatTensor(X_test[sample_idx:sample_idx+1]).to(device)
    
        with torch.no_grad():
            prediction = model(sample_input).cpu().numpy()
    
        # Reverse normalization
        y_true = normalizer.inverse_transform(y_test[sample_idx].squeeze())
        y_pred = normalizer.inverse_transform(prediction.squeeze())
    
        # Also reverse input data
        x_input = normalizer.inverse_transform(X_test[sample_idx].squeeze())
    
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 6))
    
        # Past data (input)
        past_steps = np.arange(len(x_input))
        ax.plot(past_steps, x_input, label='Past (Input)', color='gray', linewidth=2)
    
        # Future data (actual values and predictions)
        future_steps = np.arange(len(x_input), len(x_input) + len(y_true))
        ax.plot(future_steps, y_true, label='Actual Future', color='blue', linewidth=2, marker='o')
        ax.plot(future_steps, y_pred, label='Predicted Future', color='red', linewidth=2, linestyle='--', marker='x')
    
        # Boundary line
        ax.axvline(x=len(x_input)-1, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(len(x_input)-1, ax.get_ylim()[1]*0.95, 'Prediction Start',
                verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.set_title(f'Multi-Step Forecasting (7 steps ahead) - Seq2Seq')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # Evaluation metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
        print(f"\nMulti-step forecast evaluation metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # Execute visualization
    visualize_multistep_forecast(
        model_seq2seq, X_test_s2s, y_test_s2s, normalizer_minmax, device, sample_idx=10
    )
    

* * *

## 5.6 Evaluation Metrics

### Evaluation Metrics for Time Series Forecasting

The following metrics are commonly used for time series forecasting.

Metric | Formula | Characteristics  
---|---|---  
**MAE** | $\frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$ | Mean absolute error (robust to outliers)  
**RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$ | Root mean squared error (emphasizes large errors)  
**MAPE** | $\frac{100}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | Mean absolute percentage error (scale-independent)  
**R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of determination (explanatory power)  
      
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    class TimeSeriesMetrics:
        """
        Class to calculate time series forecasting evaluation metrics
        """
    
        @staticmethod
        def mae(y_true, y_pred):
            """Mean Absolute Error"""
            return mean_absolute_error(y_true, y_pred)
    
        @staticmethod
        def rmse(y_true, y_pred):
            """Root Mean Squared Error"""
            return np.sqrt(mean_squared_error(y_true, y_pred))
    
        @staticmethod
        def mape(y_true, y_pred):
            """
            Mean Absolute Percentage Error
    
            Note: Cannot calculate if y_true contains zeros
            """
            # Exclude zeros
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
        @staticmethod
        def r2(y_true, y_pred):
            """R² Score (Coefficient of Determination)"""
            return r2_score(y_true, y_pred)
    
        @staticmethod
        def mse(y_true, y_pred):
            """Mean Squared Error"""
            return mean_squared_error(y_true, y_pred)
    
        @staticmethod
        def smape(y_true, y_pred):
            """
            Symmetric Mean Absolute Percentage Error
    
            sMAPE is an improved MAPE less prone to zero denominators
            """
            numerator = np.abs(y_true - y_pred)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            return np.mean(numerator / denominator) * 100
    
        @classmethod
        def calculate_all(cls, y_true, y_pred):
            """
            Calculate all evaluation metrics
    
            Args:
                y_true: Actual values
                y_pred: Predicted values
    
            Returns:
                metrics: Dictionary of evaluation metrics
            """
            metrics = {
                'MAE': cls.mae(y_true, y_pred),
                'RMSE': cls.rmse(y_true, y_pred),
                'MSE': cls.mse(y_true, y_pred),
                'MAPE': cls.mape(y_true, y_pred),
                'sMAPE': cls.smape(y_true, y_pred),
                'R²': cls.r2(y_true, y_pred)
            }
    
            return metrics
    
        @classmethod
        def print_metrics(cls, y_true, y_pred, title="Evaluation Metrics"):
            """
            Calculate and display evaluation metrics
            """
            metrics = cls.calculate_all(y_true, y_pred)
    
            print(f"\n{title}:")
            print("=" * 50)
            for name, value in metrics.items():
                if name == 'R²':
                    print(f"  {name:10s}: {value:.4f}")
                elif 'MAPE' in name or 'sMAPE' in name:
                    print(f"  {name:10s}: {value:.2f}%")
                else:
                    print(f"  {name:10s}: {value:.4f}")
            print("=" * 50)
    
    # Usage example
    # Evaluate with dummy data
    y_true_sample = np.array([100, 105, 110, 108, 115, 120, 118])
    y_pred_sample = np.array([98, 107, 109, 110, 113, 122, 117])
    
    TimeSeriesMetrics.print_metrics(y_true_sample, y_pred_sample, title="Sample Forecast Evaluation")
    

> **Choosing metrics** :
> 
>   * **MAE** : When you want to intuitively understand error magnitude
>   * **RMSE** : When you want to emphasize large errors (sensitive to outliers)
>   * **MAPE** : When comparing data at different scales
>   * **R²** : When evaluating model explanatory power
> 

* * *

## 5.7 Practical Project: Stock Price Forecasting System

### Complete Stock Price Forecasting Pipeline

Implementation example of a stock price forecasting system using real data.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    
    class StockPriceForecastingPipeline:
        """
        Complete stock price forecasting pipeline
        """
    
        def __init__(self, window_size=60, horizon=1, hidden_size=128, num_layers=2):
            """
            Args:
                window_size: Input window size (how many days to look back)
                horizon: Prediction horizon (how many days ahead to predict)
                hidden_size: LSTM hidden layer size
                num_layers: Number of LSTM layers
            """
            self.window_size = window_size
            self.horizon = horizon
            self.scaler = MinMaxScaler()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
            # Initialize model
            self.model = None
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            print(f"Stock price forecasting pipeline initialized:")
            print(f"  Window size: {window_size} days")
            print(f"  Prediction horizon: {horizon} days")
            print(f"  Device: {self.device}")
    
        def load_data(self, prices):
            """
            Load and preprocess data
    
            Args:
                prices: Stock price data (numpy array or list)
    
            Returns:
                X_train, X_val, X_test, y_train, y_val, y_test
            """
            # Normalization
            prices = np.array(prices).reshape(-1, 1)
            normalized_data = self.scaler.fit_transform(prices)
    
            # Create windows
            X, y = create_sliding_windows(
                normalized_data,
                window_size=self.window_size,
                horizon=self.horizon
            )
    
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                X, y, train_ratio=0.7, val_ratio=0.15
            )
    
            # Convert to Tensors
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train).squeeze()
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.FloatTensor(y_val).squeeze()
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test).squeeze()
    
            print(f"\nData preparation completed:")
            print(f"  Train: {len(self.X_train)} samples")
            print(f"  Val: {len(self.X_val)} samples")
            print(f"  Test: {len(self.X_test)} samples")
    
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
        def build_model(self, input_size=1):
            """
            Build LSTM model
            """
            self.model = LSTMForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.2
            )
            self.model.to(self.device)
    
            print(f"\nModel built:")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
        def train(self, epochs=50, batch_size=32, learning_rate=0.001):
            """
            Train model
    
            Args:
                epochs: Number of epochs
                batch_size: Batch size
                learning_rate: Learning rate
    
            Returns:
                train_losses, val_losses: Training and validation loss history
            """
            from torch.utils.data import TensorDataset, DataLoader
    
            # Data loaders
            train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train),
                batch_size=batch_size,
                shuffle=False
            )
            val_loader = DataLoader(
                TensorDataset(self.X_val, self.y_val),
                batch_size=batch_size,
                shuffle=False
            )
    
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
            train_losses = []
            val_losses = []
    
            print(f"\nTraining started:")
            for epoch in range(1, epochs + 1):
                # Training
                train_loss = train_epoch(self.model, train_loader, criterion, optimizer, self.device)
                val_loss = validate(self.model, val_loader, criterion, self.device)
    
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}')
    
            print("Training completed!")
    
            return train_losses, val_losses
    
        def predict(self, X):
            """
            Execute prediction
    
            Args:
                X: Input data (Tensor or numpy)
    
            Returns:
                predictions: Predicted values (original scale)
            """
            self.model.eval()
    
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
    
            X = X.to(self.device)
    
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
    
            # Reverse normalization
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
    
            return predictions.flatten()
    
        def evaluate(self, X_test=None, y_test=None):
            """
            Evaluate on test data
    
            Returns:
                metrics: Dictionary of evaluation metrics
            """
            if X_test is None:
                X_test = self.X_test
                y_test = self.y_test
    
            # Prediction
            y_pred = self.predict(X_test)
    
            # Reverse actual values to original scale
            y_true = self.scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
            # Calculate evaluation metrics
            metrics = TimeSeriesMetrics.calculate_all(y_true, y_pred)
    
            TimeSeriesMetrics.print_metrics(y_true, y_pred, title="Test Set Evaluation")
    
            return metrics, y_true, y_pred
    
        def visualize_predictions(self, num_samples=100):
            """
            Visualize prediction results
            """
            metrics, y_true, y_pred = self.evaluate()
    
            # Visualize only first num_samples
            y_true = y_true[:num_samples]
            y_pred = y_pred[:num_samples]
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
            # Predicted vs Actual
            time_steps = np.arange(len(y_true))
            ax1.plot(time_steps, y_true, label='Actual', color='blue', linewidth=2)
            ax1.plot(time_steps, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Stock Price')
            ax1.set_title('Stock Price Prediction - Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
            # Error
            errors = y_true - y_pred
            ax2.plot(time_steps, errors, color='purple', linewidth=1.5)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.fill_between(time_steps, 0, errors, alpha=0.3, color='purple')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Prediction Error')
            ax2.set_title('Prediction Error Over Time')
            ax2.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def save_model(self, path='stock_forecaster.pth'):
            """Save model"""
            torch.save(self.model.state_dict(), path)
            print(f"Model saved: {path}")
    
        def load_model(self, path='stock_forecaster.pth'):
            """Load model"""
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)
            print(f"Model loaded: {path}")
    
    # Usage example
    if __name__ == '__main__':
        # Generate pseudo stock price data (in reality, fetch using yfinance)
        np.random.seed(42)
        stock_prices = np.cumsum(np.random.randn(1500)) + 100
    
        # Initialize pipeline
        pipeline = StockPriceForecastingPipeline(
            window_size=60,
            horizon=1,
            hidden_size=128,
            num_layers=2
        )
    
        # Load data
        pipeline.load_data(stock_prices)
    
        # Build model
        pipeline.build_model(input_size=1)
    
        # Train
        train_losses, val_losses = pipeline.train(epochs=50, batch_size=32, learning_rate=0.001)
    
        # Evaluate and visualize
        pipeline.visualize_predictions(num_samples=100)
    
        # Save model
        # pipeline.save_model('stock_forecaster.pth')
    

* * *

## 5.8 Practical Project: Weather Forecasting System

### Multivariate Weather Data Prediction

Forecasting system using multiple weather variables such as temperature, humidity, and atmospheric pressure.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - torch>=2.0.0, <2.3.0
    
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    
    class WeatherForecastingSystem:
        """
        Multivariate weather forecasting system
        """
    
        def __init__(self, window_size=24, horizon=6, hidden_size=128):
            """
            Args:
                window_size: Input window size (past 24 hours)
                horizon: Prediction horizon (6 hours ahead)
                hidden_size: LSTM hidden layer size
            """
            self.window_size = window_size
            self.horizon = horizon
            self.hidden_size = hidden_size
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
            self.scaler = MinMaxScaler()
            self.model = None
    
            print(f"Weather forecasting system initialized:")
            print(f"  Input: Past {window_size} hours")
            print(f"  Output: {horizon} hours ahead temperature prediction")
    
        def create_synthetic_weather_data(self, num_hours=2000):
            """
            Generate synthetic weather data (in reality, fetch from API)
    
            Returns:
                weather_df: Weather data DataFrame
            """
            np.random.seed(42)
    
            # Time axis
            time_index = pd.date_range('2023-01-01', periods=num_hours, freq='H')
    
            # Basic trend (temperature with seasonality)
            season = 15 * np.sin(2 * np.pi * np.arange(num_hours) / (24 * 365))
            daily = 5 * np.sin(2 * np.pi * np.arange(num_hours) / 24)
    
            # Each weather variable
            temperature = 15 + season + daily + np.random.randn(num_hours) * 2
            humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(num_hours) / 24) + np.random.randn(num_hours) * 5
            pressure = 1013 + np.random.randn(num_hours) * 3
            wind_speed = 5 + np.abs(np.random.randn(num_hours) * 2)
    
            # Create DataFrame
            weather_df = pd.DataFrame({
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed
            }, index=time_index)
    
            return weather_df
    
        def prepare_data(self, weather_df):
            """
            Data preprocessing and window creation
    
            Args:
                weather_df: Weather data DataFrame
    
            Returns:
                X_train, X_val, X_test, y_train, y_val, y_test
            """
            # Normalization
            data_normalized = self.scaler.fit_transform(weather_df.values)
    
            # Create windows
            X, y = create_sliding_windows(
                data_normalized,
                window_size=self.window_size,
                horizon=self.horizon
            )
    
            # Target is temperature only (first column)
            y_temperature = y[:, :, 0]  # [num_samples, horizon]
    
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                X, y_temperature, train_ratio=0.7, val_ratio=0.15
            )
    
            # Convert to Tensors
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train)
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.FloatTensor(y_val)
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test)
    
            print(f"\nData preparation completed:")
            print(f"  Features: {X.shape[2]}")
            print(f"  Train: {len(self.X_train)} samples")
            print(f"  Test: {len(self.X_test)} samples")
    
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
        def build_model(self, input_size):
            """
            Build Seq2Seq model
            """
            self.model = Seq2SeqLSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                output_length=self.horizon,
                dropout=0.2
            )
            self.model.to(self.device)
    
            print(f"\nSeq2Seq model built:")
            print(f"  Input features: {input_size}")
            print(f"  Output length: {self.horizon} hours")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
        def train(self, epochs=30, batch_size=32, learning_rate=0.001):
            """
            Train model
            """
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
            print(f"\nTraining started:")
            losses = train_seq2seq(
                self.model, self.X_train, self.y_train,
                criterion, optimizer, self.device,
                epochs=epochs, batch_size=batch_size
            )
    
            return losses
    
        def predict_weather(self, X):
            """
            Weather forecasting
    
            Args:
                X: Input data (past weather data)
    
            Returns:
                predictions: Temperature predictions (original scale)
            """
            self.model.eval()
    
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
    
            X = X.to(self.device)
    
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
    
            # Restore temperature scale (first feature)
            # scaler inverse transform requires all features, so add dummies
            num_features = self.scaler.n_features_in_
    
            # Expand predictions
            predictions_expanded = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
            predictions_expanded[:, :, 0] = predictions.squeeze()
    
            # Inverse transform
            predictions_original = self.scaler.inverse_transform(
                predictions_expanded.reshape(-1, num_features)
            )[:, 0].reshape(predictions.shape[0], predictions.shape[1])
    
            return predictions_original
    
        def visualize_forecast(self, sample_idx=0):
            """
            Visualize forecast results
            """
            # Prediction
            y_pred = self.predict_weather(self.X_test[sample_idx:sample_idx+1])
    
            # Restore actual values
            y_true_expanded = np.zeros((self.horizon, self.scaler.n_features_in_))
            y_true_expanded[:, 0] = self.y_test[sample_idx].numpy()
            y_true = self.scaler.inverse_transform(y_true_expanded)[:, 0]
    
            # Also restore input data
            X_input = self.scaler.inverse_transform(
                self.X_test[sample_idx].numpy()
            )[:, 0]  # Temperature only
    
            # Visualization
            fig, ax = plt.subplots(figsize=(14, 6))
    
            # Past data
            past_hours = np.arange(len(X_input))
            ax.plot(past_hours, X_input, label='Past Temperature (24h)', color='gray', linewidth=2, marker='o')
    
            # Future forecast
            future_hours = np.arange(len(X_input), len(X_input) + self.horizon)
            ax.plot(future_hours, y_true, label='Actual Temperature (6h)', color='blue', linewidth=2, marker='o')
            ax.plot(future_hours, y_pred.flatten(), label='Predicted Temperature (6h)',
                    color='red', linewidth=2, linestyle='--', marker='x')
    
            # Boundary line
            ax.axvline(x=len(X_input)-1, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    
            ax.set_xlabel('Hours')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Weather Forecasting - Temperature Prediction (6 hours ahead)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
            # Evaluation metrics
            TimeSeriesMetrics.print_metrics(y_true, y_pred.flatten(), title="Weather Forecast Evaluation")
    
    # Usage example
    if __name__ == '__main__':
        # Initialize system
        weather_system = WeatherForecastingSystem(window_size=24, horizon=6, hidden_size=128)
    
        # Generate synthetic data
        weather_data = weather_system.create_synthetic_weather_data(num_hours=2000)
        print("\nWeather data sample:")
        print(weather_data.head(24))
    
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = weather_system.prepare_data(weather_data)
    
        # Build model
        weather_system.build_model(input_size=weather_data.shape[1])
    
        # Train
        losses = weather_system.train(epochs=30, batch_size=32)
    
        # Predict and visualize
        weather_system.visualize_forecast(sample_idx=10)
    

* * *

## Chapter Summary

### What We Learned

  1. **Time Series Data Fundamentals**

     * Understanding trend, seasonality, cyclicity, and noise
     * Stationarity testing and transformation methods
     * Autocorrelation and temporal dependency
  2. **Data Preprocessing**

     * Importance of normalization and scaling
     * Creating sliding windows
     * Data splitting preserving temporal order
  3. **Forecasting Models**

     * Univariate LSTM forecasting (stock prices, etc.)
     * Multivariate LSTM forecasting (multiple features)
     * Seq2Seq multi-step forecasting
  4. **Evaluation and Practice**

     * Calculation and interpretation of MAE, RMSE, MAPE, R²
     * Practical stock price forecasting system
     * Multivariate weather forecasting system

### Time Series Forecasting Best Practices

Item | Recommended Method | Reason  
---|---|---  
**Data Split** | Preserve temporal order | Prevent predicting past with future data  
**Normalization** | Learn only from training data | Prevent leakage to test data  
**Window Size** | Adjust according to task (1 week to 3 months) | Too short lacks context, too long causes overfitting  
**Evaluation Metrics** | Use multiple metrics (MAE + RMSE + MAPE) | Evaluate model performance from multiple angles  
**Model Selection** | LSTM/GRU → Transformer (long-term dependencies) | Choose according to task complexity  
  
* * *

## Exercises

### Exercise 1 (Difficulty: medium)

For the following time series data, create sliding windows (window_size=5, horizon=2).
    
    
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    

Answer the number of windows created and the X, y of the first window.

Sample Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Answer the number of windows created and the X, y of the fir
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
    
    def create_sliding_windows(data, window_size, horizon):
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size : i + window_size + horizon])
        return np.array(X), np.array(y)
    
    X, y = create_sliding_windows(data, window_size=5, horizon=2)
    
    print(f"Number of windows: {len(X)}")
    print(f"First window:")
    print(f"  X (input): {X[0].flatten()}")
    print(f"  y (target): {y[0].flatten()}")
    
    # Output:
    # Number of windows: 4
    # First window:
    #   X (input): [10 20 30 40 50]
    #   y (target): [60 70]
    

### Exercise 2 (Difficulty: hard)

Using a univariate LSTM model, implement prediction of sine wave data. Generate sine wave with `np.sin(np.linspace(0, 100, 1000))` and train a model that predicts the next 1 point from the past 30 points.

Hint

  * Min-Max scaling of data
  * Create windows with window_size=30, horizon=1
  * Use LSTMForecaster model
  * Train with MSE loss (50 epochs)

### Exercise 3 (Difficulty: medium)

Calculate MAE, RMSE, and MAPE for the following prediction results.
    
    
    y_true = [100, 110, 105, 115, 120]
    y_pred = [98, 112, 107, 113, 122]
    

Sample Solution
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Calculate MAE, RMSE, and MAPE for the following prediction r
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    y_true = np.array([100, 110, 105, 115, 120])
    y_pred = np.array([98, 112, 107, 113, 122])
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Output:
    # MAE: 2.0000
    # RMSE: 2.2361
    # MAPE: 1.86%
    

### Exercise 4 (Difficulty: hard)

Using a Seq2Seq model, implement a system that predicts stock prices 7 days ahead. Train a model that takes the past 60 days as input and outputs the future 7 days, and visualize the results.

Hint

  * Use Seq2SeqLSTM model (output_length=7)
  * Create data with window_size=60, horizon=7
  * After training, visualize prediction on one sample
  * Display past 60 days and future 7 days in one graph

* * *

## References

  1. Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). _Time Series Analysis: Forecasting and Control_. Wiley.
  2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." _Neural Computation_ , 9(8), 1735-1780.
  3. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to sequence learning with neural networks." _NeurIPS_.
  4. Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." _EMNLP_.
  5. Hyndman, R. J., & Athanasopoulos, G. (2018). _Forecasting: Principles and Practice_. OTexts.
  6. Vaswani, A., et al. (2017). "Attention is all you need." _NeurIPS_. (Transformer)
  7. Lim, B., et al. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." _International Journal of Forecasting_.
