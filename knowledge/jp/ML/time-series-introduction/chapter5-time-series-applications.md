---
title: 第5章：時系列分析の実践応用
chapter_title: 第5章：時系列分析の実践応用
subtitle: 異常検知・多変量予測・因果推論・エンドツーエンドシステム
reading_time: 30-35分
difficulty: 上級
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 時系列異常検知の多様な手法を実装できる
  * ✅ 多変量時系列予測とGranger因果性分析を行える
  * ✅ 因果推論と介入分析を実施できる
  * ✅ Facebook Prophetを用いた高度な予測が可能
  * ✅ エンドツーエンドの予測システムを構築できる

* * *

## 5.1 時系列異常検知

### 異常検知の概要

**時系列異常検知（Time Series Anomaly Detection）** は、通常のパターンから逸脱したデータポイントを識別する技術です。

> 異常検知は、システム障害の早期発見、不正検知、品質管理など、多くの実用的な応用があります。

### 1\. 統計的手法（Z-score、IQR）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # サンプルデータ生成（正常パターン + 異常値）
    np.random.seed(42)
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # 正常なトレンド + 季節性
    trend = np.linspace(100, 150, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 5, n)
    normal_data = trend + seasonal + noise
    
    # 異常値を追加
    data = normal_data.copy()
    anomaly_indices = [50, 120, 200, 280]
    data[anomaly_indices] = data[anomaly_indices] + np.array([40, -35, 50, -40])
    
    df = pd.DataFrame({'date': dates, 'value': data})
    df.set_index('date', inplace=True)
    
    # Z-score法による異常検知
    z_scores = np.abs(stats.zscore(df['value']))
    threshold_z = 3
    anomalies_z = z_scores > threshold_z
    
    # IQR法による異常検知
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies_iqr = (df['value'] < lower_bound) | (df['value'] > upper_bound)
    
    print("=== 統計的異常検知 ===")
    print(f"Z-score法で検出された異常: {anomalies_z.sum()}個")
    print(f"IQR法で検出された異常: {anomalies_iqr.sum()}個")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Z-score法
    axes[0].plot(df.index, df['value'], label='データ', alpha=0.7)
    axes[0].scatter(df.index[anomalies_z], df['value'][anomalies_z],
                    color='red', s=100, label='異常値', zorder=5)
    axes[0].set_ylabel('値')
    axes[0].set_title('Z-score法による異常検知（閾値=3）', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IQR法
    axes[1].plot(df.index, df['value'], label='データ', alpha=0.7)
    axes[1].scatter(df.index[anomalies_iqr], df['value'][anomalies_iqr],
                    color='red', s=100, label='異常値', zorder=5)
    axes[1].axhline(y=lower_bound, color='r', linestyle='--',
                    label=f'下限: {lower_bound:.1f}')
    axes[1].axhline(y=upper_bound, color='r', linestyle='--',
                    label=f'上限: {upper_bound:.1f}')
    axes[1].set_xlabel('日付')
    axes[1].set_ylabel('値')
    axes[1].set_title('IQR法による異常検知', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 2\. Isolation Forestによる異常検知
    
    
    from sklearn.ensemble import IsolationForest
    
    # 特徴量エンジニアリング
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    df['rolling_std'] = df['value'].rolling(window=7).std()
    df['diff'] = df['value'].diff()
    
    # 欠損値を削除
    df_features = df[['value', 'rolling_mean', 'rolling_std', 'diff']].dropna()
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.05,  # 異常値の割合を5%と仮定
        random_state=42,
        n_estimators=100
    )
    
    # 異常スコアの計算
    anomaly_labels = iso_forest.fit_predict(df_features)
    anomaly_scores = iso_forest.score_samples(df_features)
    
    # -1: 異常, 1: 正常
    anomalies_iso = anomaly_labels == -1
    
    print("\n=== Isolation Forestによる異常検知 ===")
    print(f"検出された異常: {anomalies_iso.sum()}個")
    print(f"異常率: {anomalies_iso.sum() / len(df_features) * 100:.2f}%")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 時系列と異常値
    axes[0].plot(df_features.index, df_features['value'], alpha=0.7, label='データ')
    axes[0].scatter(df_features.index[anomalies_iso],
                    df_features['value'][anomalies_iso],
                    color='red', s=100, label='異常値', zorder=5)
    axes[0].set_ylabel('値')
    axes[0].set_title('Isolation Forestによる異常検知', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 異常スコア
    axes[1].plot(df_features.index, anomaly_scores, alpha=0.7)
    axes[1].scatter(df_features.index[anomalies_iso],
                    anomaly_scores[anomalies_iso],
                    color='red', s=50, label='異常値')
    axes[1].set_xlabel('日付')
    axes[1].set_ylabel('異常スコア')
    axes[1].set_title('異常スコア（低いほど異常）', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 3\. LSTM Autoencoderによる異常検知
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    
    # データ準備
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['value']].values)
    
    # シーケンス作成
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    seq_length = 30
    sequences = create_sequences(data_scaled, seq_length)
    
    # 訓練データとテストデータに分割
    train_size = int(0.8 * len(sequences))
    train_seq = sequences[:train_size]
    test_seq = sequences[train_size:]
    
    # LSTM Autoencoderモデル
    input_dim = 1
    latent_dim = 16
    
    # エンコーダ
    encoder_inputs = keras.Input(shape=(seq_length, input_dim))
    x = layers.LSTM(32, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(latent_dim)(x)
    encoder = keras.Model(encoder_inputs, x, name='encoder')
    
    # デコーダ
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
    
    # 学習
    print("\n=== LSTM Autoencoderの学習 ===")
    history = autoencoder.fit(
        train_seq, train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # 再構成誤差の計算
    train_pred = autoencoder.predict(train_seq, verbose=0)
    train_mse = np.mean(np.square(train_seq - train_pred), axis=(1, 2))
    
    test_pred = autoencoder.predict(test_seq, verbose=0)
    test_mse = np.mean(np.square(test_seq - test_pred), axis=(1, 2))
    
    # 異常検知の閾値設定（訓練データの99パーセンタイル）
    threshold = np.percentile(train_mse, 99)
    anomalies_ae = test_mse > threshold
    
    print(f"訓練データのMSE範囲: [{train_mse.min():.4f}, {train_mse.max():.4f}]")
    print(f"テストデータのMSE範囲: [{test_mse.min():.4f}, {test_mse.max():.4f}]")
    print(f"異常検知閾値: {threshold:.4f}")
    print(f"検出された異常: {anomalies_ae.sum()}個 ({anomalies_ae.sum()/len(test_mse)*100:.1f}%)")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 学習曲線
    axes[0].plot(history.history['loss'], label='訓練損失')
    axes[0].plot(history.history['val_loss'], label='検証損失')
    axes[0].set_xlabel('エポック')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('LSTM Autoencoderの学習曲線', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 再構成誤差
    axes[1].plot(test_mse, alpha=0.7, label='再構成誤差')
    axes[1].scatter(np.where(anomalies_ae)[0], test_mse[anomalies_ae],
                    color='red', s=50, label='異常値', zorder=5)
    axes[1].axhline(y=threshold, color='r', linestyle='--',
                    label=f'閾値: {threshold:.4f}')
    axes[1].set_xlabel('テストサンプル')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('再構成誤差と異常検知', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 4\. Prophetによる異常検知
    
    
    from prophet import Prophet
    
    # Prophet用のデータフレーム準備
    df_prophet = df[['value']].reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # モデル学習
    model = Prophet(
        interval_width=0.99,  # 99%信頼区間
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df_prophet)
    
    # 予測と信頼区間
    forecast = model.predict(df_prophet)
    
    # 異常検知：実測値が信頼区間外
    anomalies_prophet = (
        (df_prophet['y'] < forecast['yhat_lower']) |
        (df_prophet['y'] > forecast['yhat_upper'])
    )
    
    print("\n=== Prophetによる異常検知 ===")
    print(f"検出された異常: {anomalies_prophet.sum()}個")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', alpha=0.5, label='実測値')
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='予測値')
    ax.fill_between(forecast['ds'],
                     forecast['yhat_lower'],
                     forecast['yhat_upper'],
                     alpha=0.3, label='99%信頼区間')
    ax.scatter(df_prophet['ds'][anomalies_prophet],
               df_prophet['y'][anomalies_prophet],
               color='red', s=100, label='異常値', zorder=5)
    ax.set_xlabel('日付')
    ax.set_ylabel('値')
    ax.set_title('Prophetによる異常検知', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 5.2 多変量時系列予測

### 多変量時系列とは

**多変量時系列（Multivariate Time Series）** は、複数の相互依存する時系列データを同時に扱います。

### 1\. VAR（Vector AutoRegression）モデル

VARモデルは、複数の時系列の相互関係をモデル化します：

$$ \mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t $$
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    
    # サンプルデータ生成（3変量時系列）
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # 相互依存する3つの時系列
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = 0.8 * y1 + np.cumsum(np.random.normal(0, 0.5, n))
    y3 = 0.5 * y1 - 0.3 * y2 + np.cumsum(np.random.normal(0, 0.3, n))
    
    df_multi = pd.DataFrame({
        'y1': y1,
        'y2': y2,
        'y3': y3
    }, index=dates)
    
    print("=== 多変量時系列データ ===")
    print(df_multi.head())
    print(f"\n形状: {df_multi.shape}")
    
    # 定常性の確認
    print("\n=== 定常性テスト（ADF検定）===")
    for col in df_multi.columns:
        result = adfuller(df_multi[col])
        print(f"{col}: p値={result[1]:.4f} {'（定常）' if result[1] < 0.05 else '（非定常）'}")
    
    # 差分を取って定常化
    df_diff = df_multi.diff().dropna()
    
    print("\n=== 差分後の定常性テスト ===")
    for col in df_diff.columns:
        result = adfuller(df_diff[col])
        print(f"{col}: p値={result[1]:.4f} {'（定常）' if result[1] < 0.05 else '（非定常）'}")
    
    # VARモデルの次数選択
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=10)
    print("\n=== VARモデルの次数選択 ===")
    print(lag_order.summary())
    
    # 最適な次数でVARモデルを学習
    optimal_lag = lag_order.aic
    var_model = model.fit(optimal_lag)
    print(f"\n=== VARモデル（次数={optimal_lag}）===")
    print(var_model.summary())
    
    # 予測
    forecast_steps = 30
    forecast = var_model.forecast(df_diff.values[-optimal_lag:], steps=forecast_steps)
    forecast_dates = pd.date_range(df_multi.index[-1] + pd.Timedelta(days=1),
                                    periods=forecast_steps, freq='D')
    df_forecast = pd.DataFrame(forecast, index=forecast_dates, columns=df_multi.columns)
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, col in enumerate(df_multi.columns):
        # 元データ（差分）
        axes[i].plot(df_diff.index, df_diff[col], label='実測値（差分）', alpha=0.7)
        # 予測値
        axes[i].plot(df_forecast.index, df_forecast[col],
                     color='red', label='予測値', linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].set_title(f'{col}の予測', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('日付')
    plt.tight_layout()
    plt.show()
    

### 2\. Granger因果性テスト

**Granger因果性（Granger Causality）** は、ある時系列が別の時系列の予測に有用かを検定します。
    
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    print("\n=== Granger因果性テスト ===")
    
    # 各ペアでGranger因果性をテスト
    max_lag = 5
    variables = df_diff.columns.tolist()
    
    for i, var1 in enumerate(variables):
        for var2 in variables:
            if var1 != var2:
                print(f"\n{var1} → {var2} の因果性:")
                test_data = df_diff[[var2, var1]]
                try:
                    result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
    
                    # p値を抽出
                    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                    min_p = min(p_values)
    
                    if min_p < 0.05:
                        print(f"  ✓ 因果性あり（p値={min_p:.4f}）")
                    else:
                        print(f"  ✗ 因果性なし（p値={min_p:.4f}）")
                except:
                    print("  テスト失敗")
    

### 3\. 多出力モデル（Multi-output）
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # 特徴量作成（ラグ特徴量）
    def create_lagged_features(df, lags):
        df_lagged = df.copy()
        for col in df.columns:
            for lag in range(1, lags + 1):
                df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df_lagged.dropna()
    
    # ラグ特徴量の作成
    lags = 5
    df_lagged = create_lagged_features(df_multi, lags)
    
    # 特徴量とターゲットの分離
    target_cols = ['y1', 'y2', 'y3']
    feature_cols = [col for col in df_lagged.columns if col not in target_cols]
    
    X = df_lagged[feature_cols]
    y = df_lagged[target_cols]
    
    # 訓練・テストデータ分割
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Multi-output Random Forest
    multi_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    
    print("\n=== Multi-output Random Forestの学習 ===")
    multi_rf.fit(X_train, y_train)
    
    # 予測
    y_pred = multi_rf.predict(X_test)
    
    # 評価
    print("\n=== 予測性能 ===")
    for i, col in enumerate(target_cols):
        mse = mean_squared_error(y_test[col], y_pred[:, i])
        rmse = np.sqrt(mse)
        print(f"{col}: RMSE={rmse:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, col in enumerate(target_cols):
        axes[i].plot(y_test.index, y_test[col], label='実測値', alpha=0.7)
        axes[i].plot(y_test.index, y_pred[:, i],
                     label='予測値', alpha=0.7, linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].set_title(f'{col}の予測（Multi-output RF）', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('日付')
    plt.tight_layout()
    plt.show()
    

* * *

## 5.3 因果推論

### 因果推論の基礎

**因果推論（Causal Inference）** は、介入や施策の効果を測定する手法です。

### 1\. 介入分析（Intervention Analysis）
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    
    # サンプルデータ生成（介入あり）
    np.random.seed(42)
    n_pre = 200
    n_post = 100
    n_total = n_pre + n_post
    
    dates = pd.date_range('2020-01-01', periods=n_total, freq='D')
    
    # 介入前のデータ（ベースライン）
    baseline = 100 + np.cumsum(np.random.normal(0.1, 1, n_total))
    
    # 介入効果（200日目から+20の効果）
    intervention_effect = np.concatenate([
        np.zeros(n_pre),
        np.linspace(0, 20, 50),  # 徐々に効果が現れる
        20 * np.ones(n_post - 50)
    ])
    
    # 観測値
    observed = baseline + intervention_effect + np.random.normal(0, 2, n_total)
    
    df_intervention = pd.DataFrame({
        'date': dates,
        'value': observed,
        'intervention': np.concatenate([np.zeros(n_pre), np.ones(n_post)])
    }, index=dates)
    
    # 介入前データでARIMAモデルを学習
    pre_intervention = df_intervention[:n_pre]['value']
    model = ARIMA(pre_intervention, order=(2, 1, 2))
    fitted_model = model.fit()
    
    # 介入後の予測（反実仮想：介入がなかった場合の予測）
    forecast = fitted_model.forecast(steps=n_post)
    forecast_index = df_intervention.index[n_pre:]
    
    # 因果効果の推定
    actual_post = df_intervention[n_pre:]['value']
    causal_effect = actual_post.values - forecast.values
    
    print("=== 介入分析 ===")
    print(f"介入前平均: {pre_intervention.mean():.2f}")
    print(f"介入後平均（実測）: {actual_post.mean():.2f}")
    print(f"介入後平均（予測）: {forecast.mean():.2f}")
    print(f"平均因果効果: {causal_effect.mean():.2f}")
    print(f"累積因果効果: {causal_effect.sum():.2f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 時系列と予測
    axes[0].plot(df_intervention.index[:n_pre],
                 df_intervention['value'][:n_pre],
                 label='介入前（実測）', color='blue', alpha=0.7)
    axes[0].plot(df_intervention.index[n_pre:],
                 df_intervention['value'][n_pre:],
                 label='介入後（実測）', color='green', alpha=0.7)
    axes[0].plot(forecast_index, forecast,
                 label='反実仮想（介入なし予測）', color='red',
                 linestyle='--', linewidth=2)
    axes[0].axvline(x=dates[n_pre], color='black', linestyle=':',
                    label='介入時点', linewidth=2)
    axes[0].set_ylabel('値')
    axes[0].set_title('介入分析：実測値 vs 反実仮想', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 因果効果
    axes[1].plot(forecast_index, causal_effect, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].fill_between(forecast_index, 0, causal_effect,
                          alpha=0.3, color='purple')
    axes[1].set_xlabel('日付')
    axes[1].set_ylabel('因果効果')
    axes[1].set_title(f'推定因果効果（平均={causal_effect.mean():.2f}）', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 2\. CausalImpact分析（pycausalimpact）
    
    
    from causalimpact import CausalImpact
    
    # データ準備（制御変数を含む）
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # 制御変数（介入の影響を受けない関連変数）
    control1 = np.cumsum(np.random.normal(0, 1, n))
    control2 = np.cumsum(np.random.normal(0, 0.8, n))
    
    # 目標変数（制御変数と相関、介入後に効果）
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
    
    # CausalImpact分析
    pre_period = [0, intervention_point - 1]
    post_period = [intervention_point, n - 1]
    
    ci = CausalImpact(df_causal, pre_period, post_period)
    
    print("\n=== CausalImpact分析 ===")
    print(ci.summary())
    print("\n=== 詳細レポート ===")
    print(ci.summary(output='report'))
    
    # 可視化
    ci.plot()
    plt.tight_layout()
    plt.show()
    

### 3\. Difference-in-Differences（差分の差分法）
    
    
    # Difference-in-Differences（DID）サンプル
    np.random.seed(42)
    n_time = 100
    intervention_time = 50
    
    # 処置群（treatment group）
    treatment_pre = 50 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    treatment_post = 50 + np.cumsum(np.random.normal(0.1, 1, intervention_time)) + 20
    
    # 対照群（control group）: 介入効果なし
    control_pre = 45 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    control_post = 45 + np.cumsum(np.random.normal(0.1, 1, intervention_time))
    
    # DID推定
    treatment_diff = treatment_post.mean() - treatment_pre.mean()
    control_diff = control_post.mean() - control_pre.mean()
    did_estimate = treatment_diff - control_diff
    
    print("\n=== Difference-in-Differences分析 ===")
    print(f"処置群の変化: {treatment_diff:.2f}")
    print(f"対照群の変化: {control_diff:.2f}")
    print(f"DID推定値（介入効果）: {did_estimate:.2f}")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_points = np.arange(n_time)
    treatment_values = np.concatenate([treatment_pre, treatment_post])
    control_values = np.concatenate([control_pre, control_post])
    
    ax.plot(time_points[:intervention_time], treatment_pre,
            'b-', label='処置群（介入前）', linewidth=2)
    ax.plot(time_points[intervention_time:], treatment_post,
            'b--', label='処置群（介入後）', linewidth=2)
    ax.plot(time_points[:intervention_time], control_pre,
            'r-', label='対照群（介入前）', linewidth=2)
    ax.plot(time_points[intervention_time:], control_post,
            'r--', label='対照群（介入後）', linewidth=2)
    ax.axvline(x=intervention_time, color='black', linestyle=':',
               label='介入時点', linewidth=2)
    ax.set_xlabel('時間')
    ax.set_ylabel('値')
    ax.set_title(f'Difference-in-Differences（DID推定値={did_estimate:.2f}）', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 5.4 Prophet: Facebook時系列予測

### Prophetの特徴

**Prophet** は、Facebookが開発した時系列予測ライブラリで、以下の特徴があります：

  * トレンド、季節性、休日効果を自動的にモデル化
  * 欠損値やトレンドの変化に頑健
  * 直感的なパラメータ調整

### Prophetの加法モデル

$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$

  * $g(t)$: トレンド（成長関数）
  * $s(t)$: 季節性（周期的変動）
  * $h(t)$: 休日効果
  * $\epsilon_t$: 誤差項

### 1\. 基本的なProphet予測
    
    
    from prophet import Prophet
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 730  # 2年分
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    
    # トレンド + 年次季節性 + 週次季節性
    trend = np.linspace(100, 200, n)
    yearly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 5, n)
    
    y = trend + yearly_seasonality + weekly_seasonality + noise
    
    df_prophet = pd.DataFrame({'ds': dates, 'y': y})
    
    # Prophet モデル
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05  # トレンド変化の柔軟性
    )
    
    print("=== Prophetモデルの学習 ===")
    model.fit(df_prophet)
    
    # 将来予測（90日先まで）
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    print("\n=== 予測結果（最後の5行）===")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # 可視化
    fig1 = model.plot(forecast)
    plt.title('Prophet予測結果', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # コンポーネントの可視化
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.show()
    

### 2\. 休日効果の追加
    
    
    # 休日データフレームの作成
    holidays = pd.DataFrame({
        'holiday': 'special_sale',
        'ds': pd.to_datetime(['2021-11-26', '2021-12-24', '2022-11-25', '2022-12-24']),
        'lower_window': -1,  # 休日の前日から
        'upper_window': 1,   # 休日の翌日まで
    })
    
    # 休日効果を含むモデル
    model_holidays = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    
    # サンプルデータに休日効果を追加
    y_with_holidays = y.copy()
    holiday_dates = holidays['ds'].values
    for date in holiday_dates:
        idx = np.where(dates == date)[0]
        if len(idx) > 0:
            y_with_holidays[idx[0]] += 50  # 休日に+50の効果
    
    df_prophet_holidays = pd.DataFrame({'ds': dates, 'y': y_with_holidays})
    
    print("\n=== 休日効果を含むProphetモデル ===")
    model_holidays.fit(df_prophet_holidays)
    
    # 予測
    future_holidays = model_holidays.make_future_dataframe(periods=90)
    forecast_holidays = model_holidays.predict(future_holidays)
    
    # 可視化
    fig = model_holidays.plot(forecast_holidays)
    plt.title('休日効果を含むProphet予測', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # コンポーネント（休日効果も表示）
    fig_comp = model_holidays.plot_components(forecast_holidays)
    plt.tight_layout()
    plt.show()
    

### 3\. 変化点検出（Changepoint Detection）
    
    
    # トレンド変化を含むデータ生成
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # トレンド変化（250日目で傾きが変わる）
    trend1 = np.linspace(100, 150, 250)
    trend2 = np.linspace(150, 120, 250)
    trend = np.concatenate([trend1, trend2])
    
    y = trend + 20 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.normal(0, 5, n)
    
    df_changepoint = pd.DataFrame({'ds': dates, 'y': y})
    
    # 変化点検出を有効にしたモデル
    model_cp = Prophet(
        changepoint_prior_scale=0.5,  # 大きいほど柔軟に変化点を検出
        n_changepoints=25  # 候補変化点の数
    )
    
    model_cp.fit(df_changepoint)
    
    # 変化点の取得
    changepoints = model_cp.changepoints
    changepoint_dates = pd.to_datetime(changepoints)
    
    print("\n=== 検出された変化点 ===")
    print(f"変化点の数: {len(changepoint_dates)}")
    print(f"主要な変化点:")
    # 変化の大きさでソート
    deltas = model_cp.params['delta'].mean(axis=0)
    sorted_indices = np.argsort(np.abs(deltas))[-5:]  # 上位5つ
    for idx in sorted_indices:
        if idx < len(changepoint_dates):
            print(f"  {changepoint_dates[idx].date()}: 変化量={deltas[idx]:.3f}")
    
    # 予測
    future = model_cp.make_future_dataframe(periods=60)
    forecast = model_cp.predict(future)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(15, 6))
    model_cp.plot(forecast, ax=ax)
    
    # 変化点をマーク
    for cp in changepoint_dates:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.3)
    
    ax.set_title('変化点検出を含むProphet予測', fontsize=14)
    plt.tight_layout()
    plt.show()
    

* * *

## 5.5 エンドツーエンド予測システム

### 予測システムの全体像
    
    
    ```mermaid
    graph LR
        A[データ取得] --> B[前処理]
        B --> C[特徴量エンジニアリング]
        C --> D[モデル選択]
        D --> E[学習・評価]
        E --> F[予測]
        F --> G[モニタリング]
        G --> H{再学習が必要?}
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

### 完全な予測パイプライン
    
    
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
        """エンドツーエンド時系列予測パイプライン"""
    
        def __init__(self):
            self.models = {}
            self.best_model = None
            self.best_model_name = None
            self.best_score = float('inf')
            self.scaler = None
    
        def load_data(self, filepath=None, df=None):
            """データ読み込み"""
            if df is not None:
                self.data = df
            else:
                self.data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            print(f"データ読み込み完了: {self.data.shape}")
            return self
    
        def preprocess(self, target_col='value'):
            """前処理"""
            self.target_col = target_col
    
            # 欠損値処理
            self.data = self.data.interpolate(method='linear')
    
            # 外れ値処理（IQR法）
            Q1 = self.data[target_col].quantile(0.25)
            Q3 = self.data[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            self.data[target_col] = self.data[target_col].clip(lower, upper)
    
            print("前処理完了")
            return self
    
        def create_features(self, lags=[1, 2, 3, 7, 14, 30]):
            """特徴量エンジニアリング"""
            df = self.data.copy()
    
            # ラグ特徴量
            for lag in lags:
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
    
            # 移動平均
            for window in [7, 14, 30]:
                df[f'ma_{window}'] = df[self.target_col].rolling(window=window).mean()
                df[f'std_{window}'] = df[self.target_col].rolling(window=window).std()
    
            # 時間特徴量
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
    
            # 差分
            df['diff_1'] = df[self.target_col].diff(1)
            df['diff_7'] = df[self.target_col].diff(7)
    
            # 欠損値を削除
            df = df.dropna()
    
            self.feature_data = df
            print(f"特徴量エンジニアリング完了: {df.shape[1]}個の特徴量")
            return self
    
        def prepare_train_test(self, test_size=0.2):
            """訓練・テストデータ分割"""
            split_idx = int(len(self.feature_data) * (1 - test_size))
    
            self.train = self.feature_data[:split_idx]
            self.test = self.feature_data[split_idx:]
    
            # 特徴量とターゲット
            feature_cols = [col for col in self.feature_data.columns
                           if col != self.target_col]
    
            self.X_train = self.train[feature_cols]
            self.y_train = self.train[self.target_col]
            self.X_test = self.test[feature_cols]
            self.y_test = self.test[self.target_col]
    
            print(f"訓練データ: {len(self.train)}, テストデータ: {len(self.test)}")
            return self
    
        def add_model(self, name, model):
            """モデル追加"""
            self.models[name] = model
            return self
    
        def train_and_evaluate(self):
            """全モデルの学習と評価"""
            results = {}
    
            for name, model in self.models.items():
                print(f"\n=== {name}の学習 ===")
    
                # 学習
                model.fit(self.X_train, self.y_train)
    
                # 予測
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
    
                # 評価
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
    
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'predictions': y_pred_test
                }
    
                print(f"訓練RMSE: {train_rmse:.4f}")
                print(f"テストRMSE: {test_rmse:.4f}")
                print(f"テストMAE: {test_mae:.4f}")
    
                # 最良モデルの更新
                if test_rmse < self.best_score:
                    self.best_score = test_rmse
                    self.best_model = model
                    self.best_model_name = name
    
            self.results = results
            print(f"\n最良モデル: {self.best_model_name} (RMSE={self.best_score:.4f})")
            return self
    
        def cross_validate(self, n_splits=5):
            """時系列交差検証"""
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
    
            print("\n=== 交差検証結果 ===")
            for name, result in cv_results.items():
                print(f"{name}: RMSE={result['mean_rmse']:.4f} (±{result['std_rmse']:.4f})")
    
            return cv_results
    
        def plot_results(self):
            """結果の可視化"""
            n_models = len(self.results)
            fig, axes = plt.subplots(n_models + 1, 1, figsize=(15, 4 * (n_models + 1)))
    
            # 個別モデルの予測
            for i, (name, result) in enumerate(self.results.items()):
                ax = axes[i]
                ax.plot(self.test.index, self.y_test, label='実測値', alpha=0.7)
                ax.plot(self.test.index, result['predictions'],
                       label=f'予測値（{name}）', alpha=0.7)
                ax.set_ylabel('値')
                ax.set_title(f'{name}: RMSE={result["test_rmse"]:.4f}', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
    
            # 全モデル比較
            ax = axes[-1]
            ax.plot(self.test.index, self.y_test, label='実測値',
                   linewidth=2, alpha=0.8, color='black')
            for name, result in self.results.items():
                ax.plot(self.test.index, result['predictions'],
                       label=name, alpha=0.6)
            ax.set_xlabel('日付')
            ax.set_ylabel('値')
            ax.set_title('全モデル比較', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def predict_future(self, steps=30):
            """将来予測"""
            # 最後の特徴量を使用（簡略化）
            last_features = self.X_test.iloc[-1:].copy()
    
            predictions = []
            for _ in range(steps):
                pred = self.best_model.predict(last_features)[0]
                predictions.append(pred)
    
                # 特徴量の更新（簡略化: ラグ特徴量のみ更新）
                # 実際にはより洗練された更新が必要
    
            future_dates = pd.date_range(
                self.test.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
    
            return pd.DataFrame({'date': future_dates, 'prediction': predictions})
    
    # パイプラインの実行例
    print("=== エンドツーエンド予測パイプライン ===")
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    trend = np.linspace(100, 200, n)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    value = trend + seasonal + noise
    
    df_sample = pd.DataFrame({'value': value}, index=dates)
    
    # パイプライン実行
    pipeline = TimeSeriesPipeline()
    pipeline.load_data(df=df_sample)
    pipeline.preprocess(target_col='value')
    pipeline.create_features(lags=[1, 2, 3, 7, 14])
    pipeline.prepare_train_test(test_size=0.2)
    
    # モデル追加
    pipeline.add_model('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
    pipeline.add_model('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    pipeline.add_model('Ridge', Ridge(alpha=1.0))
    
    # 学習と評価
    pipeline.train_and_evaluate()
    
    # 交差検証
    pipeline.cross_validate(n_splits=5)
    
    # 結果可視化
    pipeline.plot_results()
    
    # 将来予測
    future_forecast = pipeline.predict_future(steps=30)
    print("\n=== 将来予測（30日先）===")
    print(future_forecast.head(10))
    

* * *

## 5.6 本章のまとめ

### 学んだこと

  1. **時系列異常検知**

     * 統計的手法（Z-score、IQR）
     * 機械学習手法（Isolation Forest）
     * 深層学習（LSTM Autoencoder）
     * Prophet による信頼区間ベース検知
  2. **多変量時系列予測**

     * VAR モデルによる相互依存のモデル化
     * Granger 因果性テスト
     * Multi-output 機械学習モデル
  3. **因果推論**

     * 介入分析と反実仮想予測
     * CausalImpact による効果測定
     * Difference-in-Differences 法
  4. **Prophet**

     * トレンド・季節性・休日効果の自動モデル化
     * 変化点検出
     * 直感的なパラメータ調整
  5. **エンドツーエンドシステム**

     * 完全な予測パイプライン構築
     * モデル選択の自動化
     * 交差検証と性能評価
     * 本番運用を見据えた設計

### 実務への応用

手法 | 適用例 | 重要ポイント  
---|---|---  
**異常検知** | システム監視、不正検知 | 複数手法の組み合わせ  
**多変量予測** | 需要予測、在庫管理 | 変数間の因果関係の理解  
**因果推論** | A/Bテスト、施策評価 | 反実仮想の適切な設定  
**Prophet** | ビジネス予測全般 | ドメイン知識の活用  
**E2Eシステム** | 本番予測システム | モニタリングと再学習  
  
* * *

## 演習問題

### 問題1（難易度：medium）

Z-score法とIQR法による異常検知の違いを説明し、それぞれどのような場合に適しているか述べてください。

解答例

**解答** ：

**Z-score法** ：

  * 平均と標準偏差を使用: $z = \frac{x - \mu}{\sigma}$
  * 通常、$|z| > 3$ を異常とする
  * 正規分布を仮定

**IQR法** ：

  * 四分位範囲を使用: $IQR = Q3 - Q1$
  * $x < Q1 - 1.5 \times IQR$ または $x > Q3 + 1.5 \times IQR$ を異常とする
  * 分布の仮定なし

**使い分け** ：

状況 | 推奨手法 | 理由  
---|---|---  
正規分布に近いデータ | Z-score法 | 統計的に解釈しやすい  
歪んだ分布 | IQR法 | 外れ値に頑健  
小さなデータセット | IQR法 | 平均が不安定  
極端な外れ値あり | IQR法 | 中央値ベースで頑健  
  
### 問題2（難易度：medium）

Granger因果性テストは、真の因果関係を証明できますか？理由とともに答えてください。

解答例

**解答** ：

**いいえ、Granger因果性は真の因果関係を証明できません。**

**理由** ：

  1. **予測的因果性** :

     * Granger因果性は「Xの過去の値がYの予測に有用か」を検定
     * 予測可能性と因果関係は異なる
  2. **第三の変数の問題** :

     * XとYの両方に影響を与える第三の変数Zが存在する可能性
     * 見かけ上の因果関係（疑似相関）
  3. **逆向き因果の可能性** :

     * X→Yの因果性が検出されても、Y→Xも同時に成立する可能性
     * 双方向の関係を完全には排除できない
  4. **時間遅れの仮定** :

     * 適切なラグ次数の選択に依存
     * 瞬時的な因果関係は捉えられない

**正しい解釈** ：

Granger因果性は「XがYの予測に有用である」という証拠を提供するが、真の因果メカニズムの証明には、実験的介入やドメイン知識が必要。

### 問題3（難易度：hard）

以下のシナリオで、適切な因果推論手法を選択し、理由を説明してください：

「新しい広告キャンペーンを特定の地域で実施し、売上への影響を測定したい。広告を実施しなかった類似地域もある。」

解答例

**解答** ：

**推奨手法** : Difference-in-Differences（DID）法またはCausalImpact

**理由** ：

  1. **DID法が適している点** :

     * 処置群（広告実施地域）と対照群（未実施地域）が存在
     * 介入前後のデータが利用可能
     * 時間的トレンドの影響を除去できる
     * 比較的単純な仮定（平行トレンド仮定）
  2. **CausalImpactが適している点** :

     * 対照群が複数ある場合、制御変数として利用可能
     * 反実仮想（広告なしの場合）を統計的に推定
     * 信頼区間による効果の不確実性評価
     * ベイズ構造時系列モデルで頑健な推定
  3. **実装例（DID）** :

    
    
    # 処置群: 広告実施地域の売上
    # 対照群: 広告未実施地域の売上
    # 介入時点: 広告開始日
    
    treatment_before = 広告前の処置群平均
    treatment_after = 広告後の処置群平均
    control_before = 広告前の対照群平均
    control_after = 広告後の対照群平均
    
    # DID推定量
    did_estimate = (treatment_after - treatment_before) - (control_after - control_before)
    
    # did_estimate が広告の因果効果
    

**注意点** ：

  * 平行トレンド仮定: 介入がなければ両群のトレンドは同じと仮定
  * 地域の類似性: 対照群が処置群と可能な限り類似していることを確認
  * 外部要因: 同時期の他のイベントの影響を考慮

### 問題4（難易度：hard）

Prophetモデルで、トレンドの変化点（changepoint）が多すぎる場合と少なすぎる場合、それぞれどのような問題が発生しますか？適切なパラメータ調整方法も説明してください。

解答例

**解答** ：

**変化点が多すぎる場合（過学習）** ：

  * 問題: ノイズをトレンド変化と誤認識
  * 結果: 訓練データに過剰適合、汎化性能の低下
  * 予測: 将来予測が不安定で信頼性が低い

**変化点が少なすぎる場合（過小適合）** ：

  * 問題: 真のトレンド変化を捉えられない
  * 結果: モデルが単純すぎ、重要なパターンを見逃す
  * 予測: 系統的な誤差、予測精度の低下

**適切なパラメータ調整** ：

  1. **changepoint_prior_scale** （変化点の柔軟性）:

    
    
    # デフォルト: 0.05
    # 小さい値（0.001-0.01）: 変化を抑制、滑らか
    # 大きい値（0.1-0.5）: 変化を許容、柔軟
    
    # 調整例
    model_smooth = Prophet(changepoint_prior_scale=0.01)  # 保守的
    model_flexible = Prophet(changepoint_prior_scale=0.5)  # 柔軟
    

  2. **n_changepoints** （候補変化点の数）:

    
    
    # デフォルト: 25
    # データ長に応じて調整
    
    model = Prophet(n_changepoints=50)  # 長い時系列の場合
    

  3. **交差検証による最適化** :

    
    
    from prophet.diagnostics import cross_validation, performance_metrics
    
    # パラメータ候補
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    }
    
    best_params = None
    best_rmse = float('inf')
    
    for scale in param_grid['changepoint_prior_scale']:
        model = Prophet(changepoint_prior_scale=scale)
        model.fit(df)
    
        # 交差検証
        df_cv = cross_validation(model, initial='730 days',
                                 period='180 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
    
        rmse = df_p['rmse'].mean()
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'changepoint_prior_scale': scale}
    
    print(f"最適パラメータ: {best_params}")
    print(f"RMSE: {best_rmse}")
    

**選択ガイドライン** ：

データ特性 | changepoint_prior_scale  
---|---  
安定したトレンド | 0.001 - 0.01  
通常のビジネスデータ | 0.05（デフォルト）  
頻繁なトレンド変化 | 0.1 - 0.5  
不確実な場合 | 交差検証で決定  
  
### 問題5（難易度：hard）

エンドツーエンド予測システムにおいて、モデルのドリフト（性能劣化）を検出し、自動再学習をトリガーする仕組みを設計してください。

解答例

**解答** ：
    
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    
    class ModelMonitoring:
        """モデルドリフト検出と自動再学習システム"""
    
        def __init__(self, model, threshold_rmse_increase=0.2,
                     window_size=30, retrain_frequency=90):
            """
            Parameters:
            -----------
            model: 予測モデル
            threshold_rmse_increase: RMSE増加の閾値（20%増加で再学習）
            window_size: モニタリングウィンドウ（直近30日）
            retrain_frequency: 最小再学習間隔（90日）
            """
            self.model = model
            self.threshold = threshold_rmse_increase
            self.window_size = window_size
            self.retrain_frequency = retrain_frequency
    
            # モニタリング指標
            self.baseline_rmse = None
            self.current_errors = []
            self.last_retrain_date = None
            self.retrain_history = []
    
        def set_baseline(self, X_val, y_val):
            """ベースライン性能を設定"""
            y_pred = self.model.predict(X_val)
            self.baseline_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            print(f"ベースラインRMSE: {self.baseline_rmse:.4f}")
            return self
    
        def monitor_prediction(self, X_new, y_true, date):
            """新しい予測をモニタリング"""
            # 予測
            y_pred = self.model.predict(X_new)
    
            # 誤差を記録
            error = np.abs(y_true - y_pred)
            self.current_errors.append({
                'date': date,
                'error': error,
                'y_true': y_true,
                'y_pred': y_pred
            })
    
            # ウィンドウサイズを超えたら古いデータを削除
            if len(self.current_errors) > self.window_size:
                self.current_errors.pop(0)
    
            # ドリフト検出
            if len(self.current_errors) >= self.window_size:
                drift_detected = self._detect_drift()
    
                if drift_detected:
                    print(f"\n⚠️ モデルドリフト検出（日付: {date}）")
    
                    # 再学習の必要性チェック
                    if self._should_retrain(date):
                        print("🔄 自動再学習を開始")
                        return True  # 再学習が必要
                    else:
                        print(f"最終再学習から{self.retrain_frequency}日未満のため待機")
    
            return False  # 再学習不要
    
        def _detect_drift(self):
            """ドリフトを検出"""
            # 直近ウィンドウのRMSE
            recent_errors = [e['error'] for e in self.current_errors]
            current_rmse = np.sqrt(np.mean(np.square(recent_errors)))
    
            # ベースラインとの比較
            rmse_increase = (current_rmse - self.baseline_rmse) / self.baseline_rmse
    
            print(f"現在のRMSE: {current_rmse:.4f} (増加率: {rmse_increase*100:.1f}%)")
    
            # 閾値を超えたらドリフトと判定
            return rmse_increase > self.threshold
    
        def _should_retrain(self, current_date):
            """再学習すべきか判定"""
            if self.last_retrain_date is None:
                return True
    
            days_since_retrain = (current_date - self.last_retrain_date).days
            return days_since_retrain >= self.retrain_frequency
    
        def retrain(self, X_train, y_train, X_val, y_val, date):
            """モデルを再学習"""
            # 再学習実行
            self.model.fit(X_train, y_train)
    
            # 新しいベースライン設定
            self.set_baseline(X_val, y_val)
    
            # 再学習履歴を記録
            self.last_retrain_date = date
            self.retrain_history.append({
                'date': date,
                'new_baseline_rmse': self.baseline_rmse
            })
    
            # エラー履歴をクリア
            self.current_errors = []
    
            print(f"✅ 再学習完了（新ベースラインRMSE: {self.baseline_rmse:.4f}）")
    
        def get_monitoring_report(self):
            """モニタリングレポート"""
            report = {
                'baseline_rmse': self.baseline_rmse,
                'num_retrains': len(self.retrain_history),
                'last_retrain': self.last_retrain_date,
                'retrain_history': self.retrain_history
            }
            return report
    
    
    # 使用例
    from sklearn.ensemble import RandomForestRegressor
    
    # サンプルデータ生成（ドリフトを含む）
    np.random.seed(42)
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # 最初は安定
    data1 = 100 + np.cumsum(np.random.normal(0, 1, 200))
    # 200日目からドリフト（トレンド変化）
    data2 = data1[-1] + np.cumsum(np.random.normal(0.5, 2, n - 200))
    data = np.concatenate([data1, data2])
    
    # 特徴量とターゲット（簡略化）
    X = np.arange(n).reshape(-1, 1)
    y = data
    
    # 初期学習
    train_size = 150
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:200], y[train_size:200]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # モニタリングシステム
    monitor = ModelMonitoring(
        model=model,
        threshold_rmse_increase=0.2,
        window_size=30,
        retrain_frequency=90
    )
    monitor.set_baseline(X_val, y_val)
    
    # オンライン予測とモニタリング
    print("\n=== オンラインモニタリング開始 ===")
    for i in range(200, n):
        X_new = X[i:i+1]
        y_true = y[i]
        date = dates[i]
    
        # モニタリング
        needs_retrain = monitor.monitor_prediction(X_new, y_true, date)
    
        # 再学習が必要な場合
        if needs_retrain:
            # 再学習用データ（直近のデータを使用）
            retrain_start = max(0, i - 150)
            X_retrain = X[retrain_start:i]
            y_retrain = y[retrain_start:i]
            X_val_new = X[i-50:i]
            y_val_new = y[i-50:i]
    
            monitor.retrain(X_retrain, y_retrain, X_val_new, y_val_new, date)
    
    # モニタリングレポート
    print("\n=== モニタリングレポート ===")
    report = monitor.get_monitoring_report()
    print(f"ベースラインRMSE: {report['baseline_rmse']:.4f}")
    print(f"再学習回数: {report['num_retrains']}")
    print(f"最終再学習日: {report['last_retrain']}")
    print("\n再学習履歴:")
    for h in report['retrain_history']:
        print(f"  {h['date'].date()}: RMSE={h['new_baseline_rmse']:.4f}")
    

**設計のポイント** ：

  1. **ドリフト検出指標** : 
     * RMSE増加率（性能劣化）
     * 予測誤差の分布変化（KSテストなど）
     * 特徴量分布の変化
  2. **再学習戦略** : 
     * 定期的再学習（時間ベース）
     * 性能ベース再学習（ドリフト検出時）
     * ハイブリッド（両方の条件）
  3. **実装上の考慮点** : 
     * A/Bテスト: 新旧モデルを並行稼働
     * ロールバック機能: 再学習後の性能悪化に対応
     * アラート機能: 人間の介入が必要な場合

* * *

## 参考文献

  1. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. _The American Statistician_ , 72(1), 37-45.
  2. Brodersen, K. H., et al. (2015). Inferring causal impact using Bayesian structural time-series models. _Annals of Applied Statistics_ , 9(1), 247-274.
  3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. _ACM Computing Surveys_ , 41(3), 1-58.
  4. Lütkepohl, H. (2005). _New Introduction to Multiple Time Series Analysis_. Springer.
  5. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts.
  6. Pearl, J., & Mackenzie, D. (2018). _The Book of Why: The New Science of Cause and Effect_. Basic Books.
