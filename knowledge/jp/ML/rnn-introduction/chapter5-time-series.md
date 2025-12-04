---
title: 第5章：時系列予測入門
chapter_title: 第5章：時系列予測入門
subtitle: RNNによる時系列データの解析と未来予測 - 株価・気象・需要予測
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 8
exercises: 4
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時系列データの特性を理解し、適切な前処理を実装できる
  * ✅ Window（スライディングウィンドウ）の概念を理解し、データセットを作成できる
  * ✅ LSTMとGRUを用いた単変量・多変量時系列予測を実装できる
  * ✅ Seq2Seqモデルで複数ステップ先の予測を行える
  * ✅ MAE、RMSE、MAPEなどの評価指標を計算し解釈できる
  * ✅ 実践的な株価予測・気象予測システムを構築できる

* * *

## 5.1 時系列データの特徴

### 時系列データとは

**時系列データ（Time Series Data）** は、時間軸に沿って記録された一連の観測値です。過去のパターンから未来を予測することが主な目的となります。
    
    
    ```mermaid
    graph LR
        A[時系列データの種類] --> B[単変量Univariate]
        A --> C[多変量Multivariate]
    
        B --> B1["1つの変数のみ例: 株価の終値"]
        C --> C1["複数の変数例: 株価+出来高+指標"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

特徴 | 説明 | 応用例  
---|---|---  
**トレンド（Trend）** | 長期的な増加・減少傾向 | 経済成長、人口増加  
**季節性（Seasonality）** | 周期的なパターン（日次、月次、年次） | 気温の季節変動、小売の繁忙期  
**周期性（Cyclicity）** | 不規則な周期のパターン | 景気循環、ビジネスサイクル  
**ノイズ（Noise）** | ランダムな変動 | 測定誤差、予測不可能な事象  
  
### 時系列予測の課題

#### データの依存性

時系列データは時間的な**自己相関（Autocorrelation）** を持ちます。過去の値が未来の値に影響を与えるため、データポイントは独立ではありません。

$$ \text{Autocorrelation}(k) = \frac{\sum_{t=1}^{N-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{N}(x_t - \bar{x})^2} $$

#### 非定常性（Non-Stationarity）

多くの時系列データは統計的性質（平均、分散）が時間と共に変化する**非定常** データです。予測モデルは通常、定常性を仮定するため、前処理が必要です。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    
    def check_stationarity(timeseries, title='Time Series'):
        """
        時系列データの定常性をチェック
    
        Args:
            timeseries: 時系列データ（pandas Series）
            title: グラフのタイトル
        """
        # ローリング統計量の計算
        rolling_mean = timeseries.rolling(window=12).mean()
        rolling_std = timeseries.rolling(window=12).std()
    
        # 可視化
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeseries, color='blue', label='Original')
        ax.plot(rolling_mean, color='red', label='Rolling Mean (12 periods)')
        ax.plot(rolling_std, color='black', label='Rolling Std (12 periods)')
        ax.legend(loc='best')
        ax.set_title(f'{title} - Rolling Mean & Standard Deviation')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # Augmented Dickey-Fuller検定（定常性の統計的検定）
        print(f'\n{title} - ADF Test Results:')
        adf_result = adfuller(timeseries.dropna(), autolag='AIC')
    
        print(f'ADF Statistic: {adf_result[0]:.6f}')
        print(f'p-value: {adf_result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in adf_result[4].items():
            print(f'  {key}: {value:.3f}')
    
        # 判定
        if adf_result[1] <= 0.05:
            print('結論: データは定常（stationary）です（p-value ≤ 0.05）')
        else:
            print('結論: データは非定常（non-stationary）です（p-value > 0.05）')
            print('      → 差分変換や対数変換を検討してください')
    
    # 使用例: ランダムウォークデータ（非定常）
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200))
    ts_nonstationary = pd.Series(random_walk, index=pd.date_range('2020-01-01', periods=200))
    
    # 定常データ（ホワイトノイズ）
    ts_stationary = pd.Series(np.random.randn(200), index=pd.date_range('2020-01-01', periods=200))
    
    # 検定実行
    check_stationarity(ts_nonstationary, title='Non-Stationary Data (Random Walk)')
    check_stationarity(ts_stationary, title='Stationary Data (White Noise)')
    

> **定常化の手法** :
> 
>   * **差分変換（Differencing）** : $x'_t = x_t - x_{t-1}$ でトレンドを除去
>   * **対数変換（Log Transform）** : $x'_t = \log(x_t)$ で分散を安定化
>   * **移動平均（Moving Average）** : 平滑化により季節性を除去
> 

* * *

## 5.2 データ前処理とWindow作成

### 正規化（Normalization）

時系列データの前処理では、**Min-Maxスケーリング** や**標準化** が一般的です。重要なのは、訓練データの統計量のみを使用し、テストデータにリークさせないことです。
    
    
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    class TimeSeriesNormalizer:
        """
        時系列データの正規化クラス
        """
    
        def __init__(self, method='minmax'):
            """
            Args:
                method: 'minmax' または 'standard'
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
            訓練データで正規化パラメータを学習し、変換
    
            Args:
                data: numpy array of shape [N, features]
    
            Returns:
                normalized_data: 正規化されたデータ
            """
            # 2次元配列に変換（必要な場合）
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            normalized_data = self.scaler.fit_transform(data)
    
            return normalized_data
    
        def transform(self, data):
            """
            学習済みパラメータで変換（テストデータ用）
    
            Args:
                data: numpy array of shape [N, features]
    
            Returns:
                normalized_data: 正規化されたデータ
            """
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            normalized_data = self.scaler.transform(data)
    
            return normalized_data
    
        def inverse_transform(self, data):
            """
            正規化を元に戻す（予測値の復元用）
    
            Args:
                data: 正規化されたデータ
    
            Returns:
                original_scale_data: 元のスケールに戻したデータ
            """
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
    
            original_scale_data = self.scaler.inverse_transform(data)
    
            return original_scale_data
    
    # 使用例
    np.random.seed(42)
    stock_prices = np.cumsum(np.random.randn(1000)) + 100  # 疑似株価データ
    
    # Min-Max正規化
    normalizer_minmax = TimeSeriesNormalizer(method='minmax')
    normalized_minmax = normalizer_minmax.fit_transform(stock_prices)
    
    print("Min-Max正規化:")
    print(f"  元のデータ範囲: [{stock_prices.min():.2f}, {stock_prices.max():.2f}]")
    print(f"  正規化後の範囲: [{normalized_minmax.min():.2f}, {normalized_minmax.max():.2f}]")
    
    # 標準化
    normalizer_std = TimeSeriesNormalizer(method='standard')
    normalized_std = normalizer_std.fit_transform(stock_prices)
    
    print("\n標準化:")
    print(f"  元のデータ: 平均={stock_prices.mean():.2f}, 標準偏差={stock_prices.std():.2f}")
    print(f"  正規化後: 平均={normalized_std.mean():.4f}, 標準偏差={normalized_std.std():.4f}")
    

### スライディングウィンドウ（Sliding Window）

**スライディングウィンドウ** は、時系列データを入力シーケンス（過去のデータ）と目標値（未来のデータ）のペアに分割する手法です。
    
    
    ```mermaid
    graph LR
        A[元の時系列x1, x2, ..., xN] --> B[Window 1入力: x1~x10目標: x11]
        A --> C[Window 2入力: x2~x11目標: x12]
        A --> D[Window 3入力: x3~x12目標: x13]
        A --> E[...]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
    ```
    
    
    def create_sliding_windows(data, window_size, horizon=1, stride=1):
        """
        スライディングウィンドウでデータセットを作成
    
        Args:
            data: 時系列データ（numpy array）[N, features]
            window_size: 入力ウィンドウサイズ（過去何ステップを見るか）
            horizon: 予測ホライズン（何ステップ先を予測するか）
            stride: ウィンドウのスライド幅
    
        Returns:
            X: 入力データ [num_windows, window_size, features]
            y: 目標データ [num_windows, horizon, features]
        """
        X, y = [], []
    
        for i in range(0, len(data) - window_size - horizon + 1, stride):
            # 入力ウィンドウ
            X.append(data[i : i + window_size])
    
            # 目標値（horizon ステップ先）
            y.append(data[i + window_size : i + window_size + horizon])
    
        X = np.array(X)
        y = np.array(y)
    
        return X, y
    
    # 使用例
    # 疑似株価データ（正規化済み）
    data = normalized_minmax.reshape(-1, 1)  # [1000, 1]
    
    # パラメータ設定
    WINDOW_SIZE = 60   # 過去60日分を見る
    HORIZON = 1        # 1日先を予測
    STRIDE = 1         # 1日ごとにウィンドウをスライド
    
    # ウィンドウ作成
    X, y = create_sliding_windows(data, window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE)
    
    print(f"ウィンドウデータ作成完了:")
    print(f"  入力 X の形状: {X.shape}")  # [num_windows, 60, 1]
    print(f"  目標 y の形状: {y.shape}")  # [num_windows, 1, 1]
    print(f"  総ウィンドウ数: {len(X)}")
    
    # 1つのウィンドウを可視化
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_idx = 100
    
    # 入力ウィンドウ（過去60日）
    ax.plot(range(WINDOW_SIZE), X[sample_idx, :, 0], marker='o', label='Input Window (Past 60 days)')
    
    # 目標値（1日先）
    ax.scatter([WINDOW_SIZE], y[sample_idx, 0, 0], color='red', s=100, zorder=5, label='Target (Next day)', marker='*')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Normalized Price')
    ax.set_title('Sliding Window Example')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Train/Validation/Test分割

時系列データでは、**時間順序を保持した分割** が必須です。ランダムシャッフルは使用しません。
    
    
    def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
        """
        時系列データのTrain/Val/Test分割（時間順序を保持）
    
        Args:
            X: 入力データ [num_samples, window_size, features]
            y: 目標データ [num_samples, horizon, features]
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
    
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        num_samples = len(X)
    
        # 分割インデックス
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))
    
        # 分割
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
    
        print(f"データ分割完了:")
        print(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # 分割実行
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15)
    
    # PyTorch Tensorに変換
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"\nTensor形状:")
    print(f"  X_train: {X_train_tensor.shape}")
    print(f"  y_train: {y_train_tensor.shape}")
    

> **重要な注意点** :
> 
>   * 時系列データは必ず時間順に分割（Train → Val → Test の順）
>   * 正規化パラメータは訓練データのみから計算
>   * テストデータは最後の評価まで一切使用しない
> 

* * *

## 5.3 単変量時系列予測

### LSTMによる株価予測

**単変量時系列予測** は、1つの変数（例: 株価の終値）のみを使用して未来を予測するタスクです。
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    class LSTMForecaster(nn.Module):
        """
        単変量時系列予測用LSTMモデル
        """
    
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            """
            Args:
                input_size: 入力特徴数（単変量なら1）
                hidden_size: LSTM隠れ層のサイズ
                num_layers: LSTMレイヤー数
                dropout: ドロップアウト率
            """
            super(LSTMForecaster, self).__init__()
    
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            # LSTM層
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # 全結合層（出力）
            self.fc = nn.Linear(hidden_size, 1)
    
        def forward(self, x):
            """
            Args:
                x: [batch_size, seq_len, input_size]
    
            Returns:
                out: [batch_size, 1] - 1ステップ先の予測
            """
            # LSTM順伝播
            lstm_out, (h_n, c_n) = self.lstm(x)
    
            # 最後の隠れ状態を使用
            out = self.fc(lstm_out[:, -1, :])  # [batch_size, 1]
    
            return out
    
    # モデルの作成
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("LSTM予測モデル:")
    print(model)
    print(f"\n総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # データローダーの作成
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.squeeze(-1))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor.squeeze(-1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 時系列は通常シャッフルしない
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 損失関数とオプティマイザー
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(model, data_loader, criterion, optimizer, device):
        """
        1エポックの訓練
        """
        model.train()
        total_loss = 0
    
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
            # 順伝播
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
    
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def validate(model, data_loader, criterion, device):
        """
        検証データでの評価
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
    
    # 訓練ループ
    num_epochs = 50
    train_losses = []
    val_losses = []
    
    print("\n訓練開始:")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print("\n訓練完了!")
    
    # 損失の可視化
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
    

### 予測と可視化
    
    
    def forecast_and_visualize(model, X_test, y_test, normalizer, device, num_samples=100):
        """
        テストデータで予測を実行し、結果を可視化
    
        Args:
            model: 訓練済みモデル
            X_test: テスト入力 [num_samples, seq_len, 1]
            y_test: テスト目標 [num_samples, 1, 1]
            normalizer: TimeSeriesNormalizer（逆変換用）
            device: デバイス
            num_samples: 可視化するサンプル数
        """
        model.eval()
    
        # テストデータをTensorに変換
        X_test_tensor = torch.FloatTensor(X_test[:num_samples]).to(device)
    
        # 予測
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
    
        # 正規化を元に戻す
        y_true = normalizer.inverse_transform(y_test[:num_samples].squeeze())
        y_pred = normalizer.inverse_transform(predictions)
    
        # 可視化
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
    
        # 評価指標の計算
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        print(f"\n評価指標:")
        print(f"  MAE (Mean Absolute Error): {mae:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
        return y_true, y_pred
    
    # 予測実行
    y_true, y_pred = forecast_and_visualize(
        model, X_test, y_test, normalizer_minmax, device, num_samples=100
    )
    

* * *

## 5.4 多変量時系列予測

### 複数特徴量を用いた予測

**多変量時系列予測** は、複数の関連する変数を同時に考慮して予測を行います。例えば、株価予測では終値だけでなく、出来高、移動平均、RSIなどの指標も使用します。
    
    
    import pandas as pd
    import numpy as np
    
    def create_multivariate_features(prices, window=20):
        """
        多変量時系列の特徴量を作成
    
        Args:
            prices: 株価データ（numpy array）
            window: 移動平均のウィンドウサイズ
    
        Returns:
            features_df: 特徴量DataFrame
        """
        df = pd.DataFrame({'price': prices})
    
        # 移動平均（Moving Average）
        df['ma_5'] = df['price'].rolling(window=5).mean()
        df['ma_20'] = df['price'].rolling(window=20).mean()
    
        # 標準偏差（Volatility）
        df['std_20'] = df['price'].rolling(window=20).std()
    
        # 変化率（Return）
        df['return'] = df['price'].pct_change()
    
        # RSI（Relative Strength Index）
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
        # 欠損値を前方埋め
        df = df.fillna(method='ffill').fillna(method='bfill')
    
        return df
    
    # 疑似株価データで特徴量作成
    stock_prices = np.cumsum(np.random.randn(1000)) + 100
    features_df = create_multivariate_features(stock_prices)
    
    print("多変量特徴量:")
    print(features_df.head(30))
    print(f"\n特徴量数: {features_df.shape[1]}")
    
    # 正規化
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features_df.values)
    
    # ウィンドウ作成（多変量版）
    WINDOW_SIZE = 60
    HORIZON = 1
    
    X_multi, y_multi = create_sliding_windows(
        features_normalized,
        window_size=WINDOW_SIZE,
        horizon=HORIZON
    )
    
    print(f"\n多変量ウィンドウ:")
    print(f"  X形状: {X_multi.shape}")  # [num_samples, 60, 6] - 6特徴量
    print(f"  y形状: {y_multi.shape}")  # [num_samples, 1, 6]
    
    # 目標は価格（最初の列）のみを使用
    y_multi_price = y_multi[:, :, 0:1]  # [num_samples, 1, 1]
    
    print(f"  y（価格のみ）形状: {y_multi_price.shape}")
    

### 多変量LSTM予測モデル
    
    
    class MultivariateLSTM(nn.Module):
        """
        多変量時系列予測用LSTMモデル
        """
    
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            """
            Args:
                input_size: 入力特徴数（複数）
                hidden_size: LSTM隠れ層のサイズ
                num_layers: LSTMレイヤー数
                dropout: ドロップアウト率
            """
            super(MultivariateLSTM, self).__init__()
    
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # 出力層
            self.fc = nn.Linear(hidden_size, 1)  # 1変数（価格）を予測
    
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
    
    # データ分割
    X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = train_val_test_split(
        X_multi, y_multi_price, train_ratio=0.7, val_ratio=0.15
    )
    
    # Tensorに変換
    X_train_m = torch.FloatTensor(X_train_m)
    y_train_m = torch.FloatTensor(y_train_m).squeeze()
    X_val_m = torch.FloatTensor(X_val_m)
    y_val_m = torch.FloatTensor(y_val_m).squeeze()
    X_test_m = torch.FloatTensor(X_test_m)
    y_test_m = torch.FloatTensor(y_test_m).squeeze()
    
    # モデル作成
    input_size = X_multi.shape[2]  # 特徴量数（6）
    model_multi = MultivariateLSTM(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.3)
    model_multi.to(device)
    
    print(f"\n多変量LSTMモデル:")
    print(f"  入力特徴数: {input_size}")
    print(f"  パラメータ数: {sum(p.numel() for p in model_multi.parameters()):,}")
    
    # データローダー
    train_loader_m = DataLoader(TensorDataset(X_train_m, y_train_m), batch_size=64, shuffle=False)
    val_loader_m = DataLoader(TensorDataset(X_val_m, y_val_m), batch_size=64, shuffle=False)
    
    # 訓練
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_multi.parameters(), lr=0.001)
    
    print("\n多変量モデル訓練開始:")
    for epoch in range(1, 51):
        train_loss = train_epoch(model_multi, train_loader_m, criterion, optimizer, device)
        val_loss = validate(model_multi, val_loader_m, criterion, device)
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/50 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print("訓練完了!")
    

> **多変量予測の利点** :
> 
>   * 複数の関連情報を活用し、予測精度を向上
>   * 市場指標、経済指標など外部情報を統合可能
>   * モデルが変数間の相互作用を学習
> 

* * *

## 5.5 Seq2Seqによる多ステップ予測

### Sequence-to-Sequence予測

**Seq2Seq（Sequence-to-Sequence）** モデルは、複数ステップ先の時系列を一度に予測するアーキテクチャです。
    
    
    ```mermaid
    graph LR
        A[Encoder過去60日] --> B[Context Vector隠れ状態]
        B --> C[Decoder未来7日予測]
    
        A1[x1, x2, ..., x60] --> A
        C --> C1[y1, y2, ..., y7]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```
    
    
    class Seq2SeqLSTM(nn.Module):
        """
        Seq2Seq時系列予測モデル（Encoder-Decoder）
        """
    
        def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_length=7, dropout=0.2):
            """
            Args:
                input_size: 入力特徴数
                hidden_size: LSTM隠れ層サイズ
                num_layers: LSTMレイヤー数
                output_length: 出力シーケンス長（何ステップ先まで予測するか）
                dropout: ドロップアウト率
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
                input_size=1,  # Decoderの入力は前のステップの予測値
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
    
            # 出力層
            self.fc = nn.Linear(hidden_size, 1)
    
        def forward(self, x, target_len=None):
            """
            Args:
                x: [batch_size, seq_len, input_size]
                target_len: デコーダーの出力長（推論時に指定）
    
            Returns:
                outputs: [batch_size, output_length, 1]
            """
            batch_size = x.size(0)
    
            if target_len is None:
                target_len = self.output_length
    
            # Encoderで過去のシーケンスをエンコード
            _, (hidden, cell) = self.encoder(x)
    
            # Decoderの初期入力（最後の入力値）
            decoder_input = x[:, -1, 0:1].unsqueeze(1)  # [batch_size, 1, 1]
    
            outputs = []
    
            # Decoderで未来をステップごとに予測
            for t in range(target_len):
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
    
                # 予測値
                prediction = self.fc(decoder_output)  # [batch_size, 1, 1]
                outputs.append(prediction)
    
                # 次のステップの入力は前のステップの予測値
                decoder_input = prediction
    
            # 出力を結合
            outputs = torch.cat(outputs, dim=1)  # [batch_size, target_len, 1]
    
            return outputs
    
    # Seq2Seq用のウィンドウ作成（複数ステップ先を予測）
    WINDOW_SIZE = 60
    HORIZON = 7  # 7ステップ先まで予測
    
    X_seq2seq, y_seq2seq = create_sliding_windows(
        data,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        stride=1
    )
    
    print(f"Seq2Seqデータ:")
    print(f"  X形状: {X_seq2seq.shape}")  # [num_samples, 60, 1]
    print(f"  y形状: {y_seq2seq.shape}")  # [num_samples, 7, 1]
    
    # データ分割
    X_train_s2s, X_val_s2s, X_test_s2s, y_train_s2s, y_val_s2s, y_test_s2s = train_val_test_split(
        X_seq2seq, y_seq2seq, train_ratio=0.7, val_ratio=0.15
    )
    
    # Tensorに変換
    X_train_s2s = torch.FloatTensor(X_train_s2s)
    y_train_s2s = torch.FloatTensor(y_train_s2s)
    X_val_s2s = torch.FloatTensor(X_val_s2s)
    y_val_s2s = torch.FloatTensor(y_val_s2s)
    X_test_s2s = torch.FloatTensor(X_test_s2s)
    y_test_s2s = torch.FloatTensor(y_test_s2s)
    
    # モデル作成
    model_seq2seq = Seq2SeqLSTM(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        output_length=HORIZON,
        dropout=0.2
    )
    model_seq2seq.to(device)
    
    print(f"\nSeq2Seqモデル:")
    print(f"  出力長: {HORIZON} ステップ")
    print(f"  パラメータ数: {sum(p.numel() for p in model_seq2seq.parameters()):,}")
    
    # 訓練関数（Seq2Seq用）
    def train_seq2seq(model, X, y, criterion, optimizer, device, epochs=30, batch_size=32):
        """
        Seq2Seqモデルの訓練
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        train_losses = []
    
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
    
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
                # 順伝播
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
    
                # 逆伝播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(loader)
            train_losses.append(avg_loss)
    
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}')
    
        return train_losses
    
    # 訓練実行
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_seq2seq.parameters(), lr=0.001)
    
    print("\nSeq2Seq訓練開始:")
    train_losses = train_seq2seq(
        model_seq2seq, X_train_s2s, y_train_s2s,
        criterion, optimizer, device, epochs=50, batch_size=32
    )
    
    print("訓練完了!")
    

### 多ステップ予測の可視化
    
    
    def visualize_multistep_forecast(model, X_test, y_test, normalizer, device, sample_idx=0):
        """
        多ステップ予測の可視化
    
        Args:
            model: Seq2Seqモデル
            X_test: テスト入力
            y_test: テスト目標
            normalizer: 正規化器
            device: デバイス
            sample_idx: 可視化するサンプルのインデックス
        """
        model.eval()
    
        # 1つのサンプルで予測
        sample_input = torch.FloatTensor(X_test[sample_idx:sample_idx+1]).to(device)
    
        with torch.no_grad():
            prediction = model(sample_input).cpu().numpy()
    
        # 正規化を元に戻す
        y_true = normalizer.inverse_transform(y_test[sample_idx].squeeze())
        y_pred = normalizer.inverse_transform(prediction.squeeze())
    
        # 入力データも元に戻す
        x_input = normalizer.inverse_transform(X_test[sample_idx].squeeze())
    
        # 可視化
        fig, ax = plt.subplots(figsize=(14, 6))
    
        # 過去データ（入力）
        past_steps = np.arange(len(x_input))
        ax.plot(past_steps, x_input, label='Past (Input)', color='gray', linewidth=2)
    
        # 未来データ（実際の値と予測値）
        future_steps = np.arange(len(x_input), len(x_input) + len(y_true))
        ax.plot(future_steps, y_true, label='Actual Future', color='blue', linewidth=2, marker='o')
        ax.plot(future_steps, y_pred, label='Predicted Future', color='red', linewidth=2, linestyle='--', marker='x')
    
        # 境界線
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
    
        # 評価指標
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
        print(f"\n多ステップ予測 評価指標:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # 可視化実行
    visualize_multistep_forecast(
        model_seq2seq, X_test_s2s, y_test_s2s, normalizer_minmax, device, sample_idx=10
    )
    

* * *

## 5.6 評価指標

### 時系列予測の評価指標

時系列予測では、以下の指標が一般的に使用されます。

指標 | 式 | 特徴  
---|---|---  
**MAE** | $\frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$ | 平均絶対誤差（外れ値に頑健）  
**RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$ | 二乗平均平方根誤差（大きな誤差を重視）  
**MAPE** | $\frac{100}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | 平均絶対パーセント誤差（スケール非依存）  
**R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 決定係数（説明力の指標）  
      
    
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    class TimeSeriesMetrics:
        """
        時系列予測の評価指標を計算するクラス
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
    
            Note: y_trueに0が含まれる場合は計算できない
            """
            # ゼロを含む場合は除外
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
    
            sMAPEは分母がゼロになりにくい改良版MAPE
            """
            numerator = np.abs(y_true - y_pred)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            return np.mean(numerator / denominator) * 100
    
        @classmethod
        def calculate_all(cls, y_true, y_pred):
            """
            全ての評価指標を計算
    
            Args:
                y_true: 実際の値
                y_pred: 予測値
    
            Returns:
                metrics: 評価指標の辞書
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
            評価指標を計算して表示
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
    
    # 使用例
    # ダミーデータで評価
    y_true_sample = np.array([100, 105, 110, 108, 115, 120, 118])
    y_pred_sample = np.array([98, 107, 109, 110, 113, 122, 117])
    
    TimeSeriesMetrics.print_metrics(y_true_sample, y_pred_sample, title="Sample Forecast Evaluation")
    

> **指標の選び方** :
> 
>   * **MAE** : 誤差の大きさを直感的に理解したい場合
>   * **RMSE** : 大きな誤差を重視したい場合（外れ値に敏感）
>   * **MAPE** : 異なるスケールのデータを比較したい場合
>   * **R²** : モデルの説明力を評価したい場合
> 

* * *

## 5.7 実践プロジェクト：株価予測システム

### 完全な株価予測パイプライン

実際のデータを使った株価予測システムの実装例です。
    
    
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    
    class StockPriceForecastingPipeline:
        """
        株価予測の完全なパイプライン
        """
    
        def __init__(self, window_size=60, horizon=1, hidden_size=128, num_layers=2):
            """
            Args:
                window_size: 入力ウィンドウサイズ（過去何日見るか）
                horizon: 予測ホライズン（何日先を予測するか）
                hidden_size: LSTM隠れ層サイズ
                num_layers: LSTMレイヤー数
            """
            self.window_size = window_size
            self.horizon = horizon
            self.scaler = MinMaxScaler()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
            # モデル初期化
            self.model = None
            self.hidden_size = hidden_size
            self.num_layers = num_layers
    
            print(f"株価予測パイプライン初期化:")
            print(f"  ウィンドウサイズ: {window_size}日")
            print(f"  予測ホライズン: {horizon}日")
            print(f"  デバイス: {self.device}")
    
        def load_data(self, prices):
            """
            データの読み込みと前処理
    
            Args:
                prices: 株価データ（numpy array or list）
    
            Returns:
                X_train, X_val, X_test, y_train, y_val, y_test
            """
            # 正規化
            prices = np.array(prices).reshape(-1, 1)
            normalized_data = self.scaler.fit_transform(prices)
    
            # ウィンドウ作成
            X, y = create_sliding_windows(
                normalized_data,
                window_size=self.window_size,
                horizon=self.horizon
            )
    
            # データ分割
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                X, y, train_ratio=0.7, val_ratio=0.15
            )
    
            # Tensorに変換
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train).squeeze()
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.FloatTensor(y_val).squeeze()
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test).squeeze()
    
            print(f"\nデータ準備完了:")
            print(f"  訓練: {len(self.X_train)} samples")
            print(f"  検証: {len(self.X_val)} samples")
            print(f"  テスト: {len(self.X_test)} samples")
    
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
        def build_model(self, input_size=1):
            """
            LSTMモデルの構築
            """
            self.model = LSTMForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.2
            )
            self.model.to(self.device)
    
            print(f"\nモデル構築完了:")
            print(f"  パラメータ数: {sum(p.numel() for p in self.model.parameters()):,}")
    
        def train(self, epochs=50, batch_size=32, learning_rate=0.001):
            """
            モデルの訓練
    
            Args:
                epochs: エポック数
                batch_size: バッチサイズ
                learning_rate: 学習率
    
            Returns:
                train_losses, val_losses: 訓練・検証損失の履歴
            """
            from torch.utils.data import TensorDataset, DataLoader
    
            # データローダー
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
    
            # 損失関数とオプティマイザー
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
            train_losses = []
            val_losses = []
    
            print(f"\n訓練開始:")
            for epoch in range(1, epochs + 1):
                # 訓練
                train_loss = train_epoch(self.model, train_loader, criterion, optimizer, self.device)
                val_loss = validate(self.model, val_loader, criterion, self.device)
    
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}')
    
            print("訓練完了!")
    
            return train_losses, val_losses
    
        def predict(self, X):
            """
            予測を実行
    
            Args:
                X: 入力データ（Tensor or numpy）
    
            Returns:
                predictions: 予測値（元のスケール）
            """
            self.model.eval()
    
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
    
            X = X.to(self.device)
    
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
    
            # 正規化を元に戻す
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
    
            return predictions.flatten()
    
        def evaluate(self, X_test=None, y_test=None):
            """
            テストデータで評価
    
            Returns:
                metrics: 評価指標の辞書
            """
            if X_test is None:
                X_test = self.X_test
                y_test = self.y_test
    
            # 予測
            y_pred = self.predict(X_test)
    
            # 実際の値を元のスケールに戻す
            y_true = self.scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
            # 評価指標計算
            metrics = TimeSeriesMetrics.calculate_all(y_true, y_pred)
    
            TimeSeriesMetrics.print_metrics(y_true, y_pred, title="Test Set Evaluation")
    
            return metrics, y_true, y_pred
    
        def visualize_predictions(self, num_samples=100):
            """
            予測結果の可視化
            """
            metrics, y_true, y_pred = self.evaluate()
    
            # 最初のnum_samplesのみ可視化
            y_true = y_true[:num_samples]
            y_pred = y_pred[:num_samples]
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
            # 予測 vs 実際
            time_steps = np.arange(len(y_true))
            ax1.plot(time_steps, y_true, label='Actual', color='blue', linewidth=2)
            ax1.plot(time_steps, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Stock Price')
            ax1.set_title('Stock Price Prediction - Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
            # 誤差
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
            """モデルの保存"""
            torch.save(self.model.state_dict(), path)
            print(f"モデルを保存: {path}")
    
        def load_model(self, path='stock_forecaster.pth'):
            """モデルの読み込み"""
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)
            print(f"モデルを読み込み: {path}")
    
    # 使用例
    if __name__ == '__main__':
        # 疑似株価データ生成（実際にはyfinanceなどで取得）
        np.random.seed(42)
        stock_prices = np.cumsum(np.random.randn(1500)) + 100
    
        # パイプラインの初期化
        pipeline = StockPriceForecastingPipeline(
            window_size=60,
            horizon=1,
            hidden_size=128,
            num_layers=2
        )
    
        # データ読み込み
        pipeline.load_data(stock_prices)
    
        # モデル構築
        pipeline.build_model(input_size=1)
    
        # 訓練
        train_losses, val_losses = pipeline.train(epochs=50, batch_size=32, learning_rate=0.001)
    
        # 評価と可視化
        pipeline.visualize_predictions(num_samples=100)
    
        # モデル保存
        # pipeline.save_model('stock_forecaster.pth')
    

* * *

## 5.8 実践プロジェクト：気象予測システム

### 多変量気象データの予測

気温、湿度、気圧などの複数の気象変数を使った予測システムです。
    
    
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    
    class WeatherForecastingSystem:
        """
        多変量気象予測システム
        """
    
        def __init__(self, window_size=24, horizon=6, hidden_size=128):
            """
            Args:
                window_size: 入力ウィンドウサイズ（過去24時間）
                horizon: 予測ホライズン（6時間先）
                hidden_size: LSTM隠れ層サイズ
            """
            self.window_size = window_size
            self.horizon = horizon
            self.hidden_size = hidden_size
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
            self.scaler = MinMaxScaler()
            self.model = None
    
            print(f"気象予測システム初期化:")
            print(f"  入力: 過去{window_size}時間")
            print(f"  出力: {horizon}時間先の気温予測")
    
        def create_synthetic_weather_data(self, num_hours=2000):
            """
            疑似気象データの生成（実際にはAPIから取得）
    
            Returns:
                weather_df: 気象データのDataFrame
            """
            np.random.seed(42)
    
            # 時間軸
            time_index = pd.date_range('2023-01-01', periods=num_hours, freq='H')
    
            # 基本トレンド（季節性を持つ気温）
            season = 15 * np.sin(2 * np.pi * np.arange(num_hours) / (24 * 365))
            daily = 5 * np.sin(2 * np.pi * np.arange(num_hours) / 24)
    
            # 各気象変数
            temperature = 15 + season + daily + np.random.randn(num_hours) * 2
            humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(num_hours) / 24) + np.random.randn(num_hours) * 5
            pressure = 1013 + np.random.randn(num_hours) * 3
            wind_speed = 5 + np.abs(np.random.randn(num_hours) * 2)
    
            # DataFrame作成
            weather_df = pd.DataFrame({
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed
            }, index=time_index)
    
            return weather_df
    
        def prepare_data(self, weather_df):
            """
            データの前処理とウィンドウ作成
    
            Args:
                weather_df: 気象データDataFrame
    
            Returns:
                X_train, X_val, X_test, y_train, y_val, y_test
            """
            # 正規化
            data_normalized = self.scaler.fit_transform(weather_df.values)
    
            # ウィンドウ作成
            X, y = create_sliding_windows(
                data_normalized,
                window_size=self.window_size,
                horizon=self.horizon
            )
    
            # 目標は気温（最初の列）のみ
            y_temperature = y[:, :, 0]  # [num_samples, horizon]
    
            # データ分割
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                X, y_temperature, train_ratio=0.7, val_ratio=0.15
            )
    
            # Tensorに変換
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train)
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.FloatTensor(y_val)
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test)
    
            print(f"\nデータ準備完了:")
            print(f"  特徴量数: {X.shape[2]}")
            print(f"  訓練: {len(self.X_train)} samples")
            print(f"  テスト: {len(self.X_test)} samples")
    
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
        def build_model(self, input_size):
            """
            Seq2Seqモデルの構築
            """
            self.model = Seq2SeqLSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                output_length=self.horizon,
                dropout=0.2
            )
            self.model.to(self.device)
    
            print(f"\nSeq2Seqモデル構築完了:")
            print(f"  入力特徴数: {input_size}")
            print(f"  出力長: {self.horizon}時間")
            print(f"  パラメータ数: {sum(p.numel() for p in self.model.parameters()):,}")
    
        def train(self, epochs=30, batch_size=32, learning_rate=0.001):
            """
            モデルの訓練
            """
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
            print(f"\n訓練開始:")
            losses = train_seq2seq(
                self.model, self.X_train, self.y_train,
                criterion, optimizer, self.device,
                epochs=epochs, batch_size=batch_size
            )
    
            return losses
    
        def predict_weather(self, X):
            """
            気象予測
    
            Args:
                X: 入力データ（過去の気象データ）
    
            Returns:
                predictions: 気温の予測値（元のスケール）
            """
            self.model.eval()
    
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
    
            X = X.to(self.device)
    
            with torch.no_grad():
                predictions = self.model(X).cpu().numpy()
    
            # 気温のスケールを元に戻す（最初の特徴量）
            # scalerの逆変換は全特徴量必要なので、ダミーを追加
            num_features = self.scaler.n_features_in_
    
            # 予測値を拡張
            predictions_expanded = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
            predictions_expanded[:, :, 0] = predictions.squeeze()
    
            # 逆変換
            predictions_original = self.scaler.inverse_transform(
                predictions_expanded.reshape(-1, num_features)
            )[:, 0].reshape(predictions.shape[0], predictions.shape[1])
    
            return predictions_original
    
        def visualize_forecast(self, sample_idx=0):
            """
            予測結果の可視化
            """
            # 予測
            y_pred = self.predict_weather(self.X_test[sample_idx:sample_idx+1])
    
            # 実際の値を元に戻す
            y_true_expanded = np.zeros((self.horizon, self.scaler.n_features_in_))
            y_true_expanded[:, 0] = self.y_test[sample_idx].numpy()
            y_true = self.scaler.inverse_transform(y_true_expanded)[:, 0]
    
            # 入力データも元に戻す
            X_input = self.scaler.inverse_transform(
                self.X_test[sample_idx].numpy()
            )[:, 0]  # 気温のみ
    
            # 可視化
            fig, ax = plt.subplots(figsize=(14, 6))
    
            # 過去データ
            past_hours = np.arange(len(X_input))
            ax.plot(past_hours, X_input, label='Past Temperature (24h)', color='gray', linewidth=2, marker='o')
    
            # 未来予測
            future_hours = np.arange(len(X_input), len(X_input) + self.horizon)
            ax.plot(future_hours, y_true, label='Actual Temperature (6h)', color='blue', linewidth=2, marker='o')
            ax.plot(future_hours, y_pred.flatten(), label='Predicted Temperature (6h)',
                    color='red', linewidth=2, linestyle='--', marker='x')
    
            # 境界線
            ax.axvline(x=len(X_input)-1, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    
            ax.set_xlabel('Hours')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Weather Forecasting - Temperature Prediction (6 hours ahead)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
            # 評価指標
            TimeSeriesMetrics.print_metrics(y_true, y_pred.flatten(), title="Weather Forecast Evaluation")
    
    # 使用例
    if __name__ == '__main__':
        # システム初期化
        weather_system = WeatherForecastingSystem(window_size=24, horizon=6, hidden_size=128)
    
        # 疑似データ生成
        weather_data = weather_system.create_synthetic_weather_data(num_hours=2000)
        print("\n気象データサンプル:")
        print(weather_data.head(24))
    
        # データ準備
        X_train, X_val, X_test, y_train, y_val, y_test = weather_system.prepare_data(weather_data)
    
        # モデル構築
        weather_system.build_model(input_size=weather_data.shape[1])
    
        # 訓練
        losses = weather_system.train(epochs=30, batch_size=32)
    
        # 予測と可視化
        weather_system.visualize_forecast(sample_idx=10)
    

* * *

## 本章のまとめ

### 学んだこと

  1. **時系列データの基礎**

     * トレンド、季節性、周期性、ノイズの理解
     * 定常性の検定と変換手法
     * 自己相関と時間依存性
  2. **データ前処理**

     * 正規化とスケーリングの重要性
     * スライディングウィンドウの作成
     * 時間順序を保持したデータ分割
  3. **予測モデル**

     * 単変量LSTM予測（株価など）
     * 多変量LSTM予測（複数特徴量）
     * Seq2Seq多ステップ予測
  4. **評価と実践**

     * MAE、RMSE、MAPE、R²の計算と解釈
     * 実践的な株価予測システム
     * 多変量気象予測システム

### 時系列予測のベストプラクティス

項目 | 推奨手法 | 理由  
---|---|---  
**データ分割** | 時間順序を保持 | 未来のデータで過去を予測する事態を防ぐ  
**正規化** | 訓練データのみで学習 | テストデータへのリークを防止  
**ウィンドウサイズ** | タスクに応じて調整（1週間〜3ヶ月） | 短すぎると文脈不足、長すぎると過学習  
**評価指標** | 複数の指標を併用（MAE + RMSE + MAPE） | 多角的にモデルの性能を評価  
**モデル選択** | LSTM/GRU → Transformer（長期依存） | タスクの複雑さに応じて選択  
  
* * *

## 演習問題

### 問題1（難易度：medium）

以下の時系列データに対して、スライディングウィンドウ（window_size=5, horizon=2）を作成してください。
    
    
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    

作成されるウィンドウの数と、最初のウィンドウのX, yを答えてください。

解答例
    
    
    import numpy as np
    
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
    
    def create_sliding_windows(data, window_size, horizon):
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size : i + window_size + horizon])
        return np.array(X), np.array(y)
    
    X, y = create_sliding_windows(data, window_size=5, horizon=2)
    
    print(f"ウィンドウ数: {len(X)}")
    print(f"最初のウィンドウ:")
    print(f"  X (入力): {X[0].flatten()}")
    print(f"  y (目標): {y[0].flatten()}")
    
    # 出力:
    # ウィンドウ数: 4
    # 最初のウィンドウ:
    #   X (入力): [10 20 30 40 50]
    #   y (目標): [60 70]
    

### 問題2（難易度：hard）

単変量LSTMモデルを使って、sin波データの予測を実装してください。sin波は `np.sin(np.linspace(0, 100, 1000))` で生成し、過去30ポイントから次の1ポイントを予測するモデルを訓練してください。

ヒント

  * データをMin-Maxスケーリング
  * window_size=30, horizon=1でウィンドウ作成
  * LSTMForecasterモデルを使用
  * MSE損失で訓練（50エポック）

### 問題3（難易度：medium）

以下の予測結果に対して、MAE、RMSE、MAPEを計算してください。
    
    
    y_true = [100, 110, 105, 115, 120]
    y_pred = [98, 112, 107, 113, 122]
    

解答例
    
    
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
    
    # 出力:
    # MAE: 2.0000
    # RMSE: 2.2361
    # MAPE: 1.86%
    

### 問題4（難易度：hard）

Seq2Seqモデルを使って、7日先までの株価を予測するシステムを実装してください。過去60日のデータを入力とし、未来7日を出力するモデルを訓練し、結果を可視化してください。

ヒント

  * Seq2SeqLSTMモデルを使用（output_length=7）
  * window_size=60, horizon=7でデータ作成
  * 訓練後、1つのサンプルで予測を可視化
  * 過去60日と未来7日を1つのグラフに表示

* * *

## 参考文献

  1. Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). _Time Series Analysis: Forecasting and Control_. Wiley.
  2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." _Neural Computation_ , 9(8), 1735-1780.
  3. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to sequence learning with neural networks." _NeurIPS_.
  4. Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." _EMNLP_.
  5. Hyndman, R. J., & Athanasopoulos, G. (2018). _Forecasting: Principles and Practice_. OTexts.
  6. Vaswani, A., et al. (2017). "Attention is all you need." _NeurIPS_. (Transformer)
  7. Lim, B., et al. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." _International Journal of Forecasting_.
