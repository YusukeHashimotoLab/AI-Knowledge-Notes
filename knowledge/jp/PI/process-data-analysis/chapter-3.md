---
title: 第3章：異常検知と故障診断
chapter_title: 第3章：異常検知と故障診断
subtitle: Anomaly Detection and Fault Diagnosis for Process Systems
---

🌐 JP | [🇬🇧 EN](<../../../en/PI/process-data-analysis/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[プロセス・インフォマティクス](<../../PI/index.html>)›[Process Data Analysis](<../../PI/process-data-analysis/index.html>)›Chapter 3

## 3.1 イントロダクション

化学プラントの安全性と生産性を維持するためには、異常の早期検知と故障原因の迅速な診断が不可欠です。 センサーデータから異常パターンを自動的に検出し、故障箇所を特定する技術は、プラント運転の高度化において重要な役割を果たします。 

本章では、統計的手法から機械学習アルゴリズム、深層学習まで、8つの実践的な異常検知・故障診断技術を実装します。 各手法の特性を理解し、プロセス特性に応じた適切なアプローチを選択できるようになります。 

#### 📊 本章で学ぶこと

  * 統計的異常検知（Z-score, Modified Z-score）
  * 機械学習による異常検知（Isolation Forest, One-Class SVM）
  * 深層学習による異常検知（Autoencoder, LSTM）
  * 故障分類とアンサンブル手法
  * 根本原因分析（Granger因果性検定）

## 3.2 統計的異常検知

統計的手法は解釈性が高く、計算コストが低いため、リアルタイム監視に適しています。 正規分布を仮定したZ-scoreベースの手法と、外れ値に頑健なModified Z-scoreを組み合わせることで、高精度な検知を実現します。 

#### Example 1: 統計的異常検知（Z-score & Modified Z-score）

標準的なZ-scoreと外れ値に頑健なModified Z-scoreを組み合わせた異常検知を実装します。
    
    
    # ===================================
    # Example 1: 統計的異常検知
    # ===================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # プロセスデータ生成（反応器温度、異常を含む）
    np.random.seed(42)
    n_samples = 500
    normal_data = np.random.normal(350, 2, n_samples)
    
    # 異常データを追加（3種類）
    anomaly_indices = [50, 150, 250, 350, 450]
    normal_data[anomaly_indices] = [360, 340, 365, 335, 362]  # 異常値
    
    df = pd.DataFrame({'temperature': normal_data})
    
    def detect_anomalies_zscore(data, threshold=3.0):
        """標準Z-score法による異常検知"""
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold, z_scores
    
    def detect_anomalies_modified_zscore(data, threshold=3.5):
        """Modified Z-score法（中央値ベース、外れ値に頑健）"""
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold, modified_z_scores
    
    # 両手法を適用
    is_anomaly_z, z_scores = detect_anomalies_zscore(df['temperature'])
    is_anomaly_mod, mod_z_scores = detect_anomalies_modified_zscore(df['temperature'])
    
    # 結果比較
    print("異常検知結果:")
    print(f"Z-score法: {is_anomaly_z.sum()}個の異常を検出")
    print(f"Modified Z-score法: {is_anomaly_mod.sum()}個の異常を検出")
    
    # 検出精度の評価
    true_anomalies = np.zeros(n_samples, dtype=bool)
    true_anomalies[anomaly_indices] = True
    
    tp_z = np.sum(is_anomaly_z & true_anomalies)
    fp_z = np.sum(is_anomaly_z & ~true_anomalies)
    precision_z = tp_z / is_anomaly_z.sum() if is_anomaly_z.sum() > 0 else 0
    recall_z = tp_z / true_anomalies.sum()
    
    tp_mod = np.sum(is_anomaly_mod & true_anomalies)
    fp_mod = np.sum(is_anomaly_mod & ~true_anomalies)
    precision_mod = tp_mod / is_anomaly_mod.sum() if is_anomaly_mod.sum() > 0 else 0
    recall_mod = tp_mod / true_anomalies.sum()
    
    print(f"\nZ-score法: Precision={precision_z:.2f}, Recall={recall_z:.2f}")
    print(f"Modified Z-score法: Precision={precision_mod:.2f}, Recall={recall_mod:.2f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(df['temperature'], 'b-', alpha=0.6, label='データ')
    axes[0].scatter(np.where(is_anomaly_z)[0], df.loc[is_anomaly_z, 'temperature'],
                    color='red', s=100, zorder=5, label=f'異常 (Z-score)')
    axes[0].axhline(df['temperature'].mean() + 3*df['temperature'].std(),
                    color='r', linestyle='--', alpha=0.5, label='±3σ閾値')
    axes[0].axhline(df['temperature'].mean() - 3*df['temperature'].std(),
                    color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('温度 (°C)')
    axes[0].set_title('Z-score法')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['temperature'], 'b-', alpha=0.6, label='データ')
    axes[1].scatter(np.where(is_anomaly_mod)[0], df.loc[is_anomaly_mod, 'temperature'],
                    color='red', s=100, zorder=5, label='異常 (Modified Z-score)')
    axes[1].set_xlabel('サンプル')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].set_title('Modified Z-score法（外れ値に頑健）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_anomaly_detection.png', dpi=300)
    print("\n結果: Modified Z-scoreが外れ値の影響を受けにくく、より正確")
    

## 3.3 機械学習による異常検知

機械学習アルゴリズムは、高次元データや複雑な非線形パターンの検知に優れています。 教師なし学習により、正常データのみから異常を識別できる点が実用上の大きな利点です。 

#### Example 2: Isolation Forestによる異常検知

ランダムフォレストベースの効率的な異常検知アルゴリズムを実装します。
    
    
    # ===================================
    # Example 2: Isolation Forest
    # ===================================
    from sklearn.ensemble import IsolationForest
    
    # 多変量プロセスデータ生成
    np.random.seed(42)
    n_normal = 450
    n_anomaly = 50
    
    # 正常データ（温度、圧力、流量の相関あり）
    normal_temp = np.random.normal(350, 3, n_normal)
    normal_pressure = 5 + 0.01 * normal_temp + np.random.normal(0, 0.2, n_normal)
    normal_flow = 100 + 0.5 * normal_temp + np.random.normal(0, 5, n_normal)
    
    # 異常データ（相関が崩れる）
    anomaly_temp = np.random.uniform(340, 370, n_anomaly)
    anomaly_pressure = np.random.uniform(4, 7, n_anomaly)
    anomaly_flow = np.random.uniform(80, 150, n_anomaly)
    
    # データ結合
    X = np.column_stack([
        np.concatenate([normal_temp, anomaly_temp]),
        np.concatenate([normal_pressure, anomaly_pressure]),
        np.concatenate([normal_flow, anomaly_flow])
    ])
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    df_process = pd.DataFrame(X, columns=['temperature', 'pressure', 'flow'])
    df_process['is_anomaly_true'] = y_true
    
    # Isolation Forest訓練
    iso_forest = IsolationForest(
        contamination=0.1,  # 期待される異常割合
        random_state=42,
        n_estimators=100
    )
    predictions = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.score_samples(X)
    
    df_process['anomaly_score'] = anomaly_scores
    df_process['is_anomaly_pred'] = predictions == -1
    
    # 評価
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Isolation Forest結果:")
    print(classification_report(y_true, predictions == -1, target_names=['正常', '異常']))
    print("\n混同行列:")
    print(confusion_matrix(y_true, predictions == -1))
    
    # 可視化（3D散布図）
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 真の異常
    scatter1 = ax1.scatter(df_process['temperature'], df_process['pressure'],
                           df_process['flow'], c=df_process['is_anomaly_true'],
                           cmap='RdYlGn_r', s=30, alpha=0.6)
    ax1.set_xlabel('温度 (°C)')
    ax1.set_ylabel('圧力 (MPa)')
    ax1.set_zlabel('流量 (m³/h)')
    ax1.set_title('真の異常ラベル')
    plt.colorbar(scatter1, ax=ax1)
    
    # 検出結果
    scatter2 = ax2.scatter(df_process['temperature'], df_process['pressure'],
                           df_process['flow'], c=df_process['anomaly_score'],
                           cmap='RdYlGn', s=30, alpha=0.6)
    ax2.set_xlabel('温度 (°C)')
    ax2.set_ylabel('圧力 (MPa)')
    ax2.set_zlabel('流量 (m³/h)')
    ax2.set_title('Isolation Forest異常スコア')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('isolation_forest.png', dpi=300)
    print("\n結果: 高次元空間での異常パターンを効率的に検出")
    

#### Example 3: One-Class SVMによる新規性検知

正常データの境界を学習し、未知の異常パターンを検出します。
    
    
    # ===================================
    # Example 3: One-Class SVM
    # ===================================
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    
    # データの標準化（SVMには重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-Class SVM訓練
    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=0.1  # 異常の上限割合
    )
    predictions_svm = oc_svm.fit_predict(X_scaled)
    decision_scores = oc_svm.decision_function(X_scaled)
    
    df_process['decision_score'] = decision_scores
    df_process['is_anomaly_svm'] = predictions_svm == -1
    
    # 評価
    print("\nOne-Class SVM結果:")
    print(classification_report(y_true, predictions_svm == -1, target_names=['正常', '異常']))
    
    # 決定境界の可視化（2D投影）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 温度-圧力平面
    xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100),
                         np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape[0])])
    Z = Z.reshape(xx.shape)
    
    axes[0].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='RdYlGn', alpha=0.3)
    axes[0].contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    axes[0].scatter(X_scaled[y_true==0, 0], X_scaled[y_true==0, 1],
                    c='green', s=20, alpha=0.6, label='正常')
    axes[0].scatter(X_scaled[y_true==1, 0], X_scaled[y_true==1, 1],
                    c='red', s=50, alpha=0.8, marker='x', label='異常')
    axes[0].set_xlabel('温度（標準化）')
    axes[0].set_ylabel('圧力（標準化）')
    axes[0].set_title('One-Class SVM決定境界')
    axes[0].legend()
    
    # 決定スコアの分布
    axes[1].hist(decision_scores[y_true==0], bins=30, alpha=0.6, label='正常', color='green')
    axes[1].hist(decision_scores[y_true==1], bins=30, alpha=0.6, label='異常', color='red')
    axes[1].axvline(0, color='black', linestyle='--', label='決定閾値')
    axes[1].set_xlabel('決定スコア')
    axes[1].set_ylabel('頻度')
    axes[1].set_title('決定スコア分布')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('one_class_svm.png', dpi=300)
    print("\n結果: 正常データの境界を厳密にモデル化、新規異常に対応可能")
    

## 3.4 深層学習による異常検知

深層学習は、複雑な時空間パターンや高次元データの異常検知に威力を発揮します。 AutoencoderとLSTMを用いた2つのアプローチを実装します。 

#### Example 4: Autoencoderによる再構成誤差ベース異常検知

正常データを圧縮・復元し、再構成誤差から異常を検出します。
    
    
    # ===================================
    # Example 4: Autoencoder異常検知
    # ===================================
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    
    # 正常データのみで訓練（教師なし学習）
    X_train = X[y_true == 0]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X)
    
    # Autoencoderモデル構築
    input_dim = X_train_scaled.shape[1]
    encoding_dim = 2  # 圧縮次元
    
    autoencoder = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(encoding_dim, activation='relu', name='encoder'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear', name='decoder')
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 訓練
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # 再構成誤差計算
    X_reconstructed = autoencoder.predict(X_test_scaled)
    reconstruction_errors = np.mean(np.square(X_test_scaled - X_reconstructed), axis=1)
    
    # 異常検知閾値（正常データの95パーセンタイル）
    threshold = np.percentile(reconstruction_errors[y_true==0], 95)
    predictions_ae = reconstruction_errors > threshold
    
    # 評価
    print("\nAutoencoder結果:")
    print(f"閾値: {threshold:.4f}")
    print(classification_report(y_true, predictions_ae, target_names=['正常', '異常']))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 学習曲線
    axes[0, 0].plot(history.history['loss'], label='訓練損失')
    axes[0, 0].plot(history.history['val_loss'], label='検証損失')
    axes[0, 0].set_xlabel('エポック')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Autoencoder学習曲線')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 再構成誤差分布
    axes[0, 1].hist(reconstruction_errors[y_true==0], bins=30, alpha=0.6,
                    label='正常', color='green')
    axes[0, 1].hist(reconstruction_errors[y_true==1], bins=30, alpha=0.6,
                    label='異常', color='red')
    axes[0, 1].axvline(threshold, color='black', linestyle='--', label='閾値')
    axes[0, 1].set_xlabel('再構成誤差')
    axes[0, 1].set_ylabel('頻度')
    axes[0, 1].set_title('再構成誤差分布')
    axes[0, 1].legend()
    
    # 潜在空間（2次元エンコーディング）
    encoder_model = keras.Model(autoencoder.input,
                                autoencoder.get_layer('encoder').output)
    encoded = encoder_model.predict(X_test_scaled)
    
    axes[1, 0].scatter(encoded[y_true==0, 0], encoded[y_true==0, 1],
                       c='green', s=20, alpha=0.6, label='正常')
    axes[1, 0].scatter(encoded[y_true==1, 0], encoded[y_true==1, 1],
                       c='red', s=50, alpha=0.8, marker='x', label='異常')
    axes[1, 0].set_xlabel('潜在次元1')
    axes[1, 0].set_ylabel('潜在次元2')
    axes[1, 0].set_title('潜在空間表現')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC曲線
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='ランダム')
    axes[1, 1].set_xlabel('偽陽性率')
    axes[1, 1].set_ylabel('真陽性率')
    axes[1, 1].set_title('ROC曲線')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_anomaly.png', dpi=300)
    print(f"\n結果: AUC={roc_auc:.3f}、高次元データの非線形パターンを学習")
    

#### Example 5: LSTMによる時系列異常検知

時系列の時間的依存性を学習し、予測誤差から異常を検出します。
    
    
    # ===================================
    # Example 5: LSTM時系列異常検知
    # ===================================
    
    # 時系列データ生成（季節性+トレンド+異常）
    np.random.seed(42)
    n_timesteps = 1000
    t = np.arange(n_timesteps)
    
    # 正常パターン
    normal_series = 100 + 0.01*t + 10*np.sin(2*np.pi*t/50) + np.random.normal(0, 1, n_timesteps)
    
    # 異常を注入
    anomaly_ranges = [(200, 220), (500, 530), (800, 815)]
    for start, end in anomaly_ranges:
        normal_series[start:end] += np.random.uniform(15, 25, end-start)
    
    # シーケンスデータ作成
    def create_sequences(data, seq_length=50):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 50
    X_seq, y_seq = create_sequences(normal_series, seq_length)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    
    # 訓練/テスト分割（異常を含まない部分で訓練）
    split_idx = 150 - seq_length
    X_train_lstm = X_seq[:split_idx]
    y_train_lstm = y_seq[:split_idx]
    X_test_lstm = X_seq
    y_test_lstm = y_seq
    
    # LSTMモデル構築
    lstm_model = keras.Sequential([
        keras.layers.LSTM(32, activation='relu', input_shape=(seq_length, 1)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # 訓練
    history_lstm = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # 予測と誤差計算
    predictions_lstm = lstm_model.predict(X_test_lstm).flatten()
    prediction_errors = np.abs(y_test_lstm - predictions_lstm)
    
    # 異常閾値（訓練データの95パーセンタイル）
    train_errors = prediction_errors[:split_idx]
    threshold_lstm = np.percentile(train_errors, 95)
    anomalies_lstm = prediction_errors > threshold_lstm
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 元データ
    axes[0].plot(normal_series, 'b-', alpha=0.7, label='実測値')
    for start, end in anomaly_ranges:
        axes[0].axvspan(start, end, color='red', alpha=0.2)
    axes[0].set_ylabel('プロセス値')
    axes[0].set_title('時系列データ（異常期間を赤で表示）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 予測 vs 実測
    axes[1].plot(y_test_lstm, 'b-', alpha=0.5, label='実測値')
    axes[1].plot(predictions_lstm, 'g--', alpha=0.7, label='予測値')
    axes[1].set_ylabel('プロセス値')
    axes[1].set_title('LSTM予測結果')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 予測誤差と異常検知
    axes[2].plot(prediction_errors, 'k-', alpha=0.6, label='予測誤差')
    axes[2].axhline(threshold_lstm, color='red', linestyle='--', label=f'閾値={threshold_lstm:.2f}')
    axes[2].fill_between(range(len(anomalies_lstm)), 0, prediction_errors,
                         where=anomalies_lstm, color='red', alpha=0.3, label='検出異常')
    axes[2].set_xlabel('時刻')
    axes[2].set_ylabel('予測誤差')
    axes[2].set_title('LSTM異常検知結果')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_anomaly.png', dpi=300)
    print("\n結果: 時系列パターンを学習、異常期間を高精度で検出")
    

## 3.5 故障分類とアンサンブル手法

異常を検出するだけでなく、故障の種類を分類することで、適切な対処法を迅速に決定できます。 Random Forestとアンサンブル投票による頑健な分類システムを構築します。 

#### Example 6: Random Forestによる故障分類

複数の故障タイプを機械学習で分類します。
    
    
    # ===================================
    # Example 6: 故障分類（Random Forest）
    # ===================================
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # 故障データ生成（4クラス：正常、センサー故障、プロセス異常、制御系異常）
    np.random.seed(42)
    n_per_class = 200
    
    # クラス0: 正常
    normal = np.random.multivariate_normal([350, 5, 100], [[4, 0.1, 5], [0.1, 0.04, 0.5], [5, 0.5, 25]], n_per_class)
    
    # クラス1: センサー故障（ノイズ増大）
    sensor_fault = np.random.multivariate_normal([350, 5, 100], [[16, 0.1, 5], [0.1, 0.16, 0.5], [5, 0.5, 100]], n_per_class)
    
    # クラス2: プロセス異常（温度上昇）
    process_fault = np.random.multivariate_normal([365, 5.5, 120], [[4, 0.2, 8], [0.2, 0.04, 0.6], [8, 0.6, 25]], n_per_class)
    
    # クラス3: 制御系異常（変動パターン変化）
    control_fault = np.random.multivariate_normal([350, 4.5, 90], [[9, -0.3, -10], [-0.3, 0.09, 2], [-10, 2, 64]], n_per_class)
    
    # データ統合
    X_fault = np.vstack([normal, sensor_fault, process_fault, control_fault])
    y_fault = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class + [3]*n_per_class)
    
    # 訓練/テスト分割
    X_train_fault, X_test_fault, y_train_fault, y_test_fault = train_test_split(
        X_fault, y_fault, test_size=0.3, random_state=42, stratify=y_fault
    )
    
    # Random Forest訓練
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_fault, y_train_fault)
    
    # 予測
    y_pred_rf = rf_classifier.predict(X_test_fault)
    y_pred_proba = rf_classifier.predict_proba(X_test_fault)
    
    # 評価
    fault_names = ['正常', 'センサー故障', 'プロセス異常', '制御系異常']
    print("\nRandom Forest故障分類結果:")
    print(classification_report(y_test_fault, y_pred_rf, target_names=fault_names))
    
    # 混同行列
    cm = confusion_matrix(y_test_fault, y_pred_rf)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_names,
                yticklabels=fault_names, ax=axes[0])
    axes[0].set_xlabel('予測')
    axes[0].set_ylabel('真値')
    axes[0].set_title('混同行列')
    
    # 特徴重要度
    importances = rf_classifier.feature_importances_
    feature_names = ['温度', '圧力', '流量']
    axes[1].barh(feature_names, importances, color='teal')
    axes[1].set_xlabel('重要度')
    axes[1].set_title('特徴重要度')
    
    plt.tight_layout()
    plt.savefig('fault_classification.png', dpi=300)
    print("\n結果: 複数の故障タイプを高精度で分類、特徴重要度から診断根拠を提示")
    

#### Example 7: アンサンブル異常検知（投票方式）

複数の異常検知手法を組み合わせ、頑健性を向上させます。
    
    
    # ===================================
    # Example 7: アンサンブル異常検知
    # ===================================
    
    # 複数の検知器を訓練（前述のデータを使用）
    # 1. Isolation Forest
    iso_pred = (iso_forest.predict(X) == -1).astype(int)
    
    # 2. One-Class SVM
    svm_pred = (oc_svm.predict(X_scaled) == -1).astype(int)
    
    # 3. Autoencoder（再構成誤差）
    ae_pred = (reconstruction_errors > threshold).astype(int)
    
    # アンサンブル投票
    ensemble_votes = np.column_stack([iso_pred, svm_pred, ae_pred])
    ensemble_pred = (ensemble_votes.sum(axis=1) >= 2).astype(int)  # 多数決
    
    # 評価
    from sklearn.metrics import accuracy_score, f1_score
    
    methods = {
        'Isolation Forest': iso_pred,
        'One-Class SVM': svm_pred,
        'Autoencoder': ae_pred,
        'アンサンブル（投票）': ensemble_pred
    }
    
    print("\nアンサンブル異常検知結果:")
    print(f"{'手法':<25} {'Accuracy':<10} {'F1-score':<10}")
    print("-" * 45)
    for name, pred in methods.items():
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred)
        print(f"{name:<25} {acc:.4f}     {f1:.4f}")
    
    # 個別手法の一致度分析
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 投票数の分布
    vote_counts = ensemble_votes.sum(axis=1)
    axes[0].hist([vote_counts[y_true==0], vote_counts[y_true==1]],
                 bins=4, label=['正常', '異常'], color=['green', 'red'], alpha=0.6)
    axes[0].set_xlabel('検知器の投票数')
    axes[0].set_ylabel('頻度')
    axes[0].set_title('アンサンブル投票分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 手法間の相関
    method_matrix = np.column_stack([iso_pred, svm_pred, ae_pred, ensemble_pred])
    correlation = np.corrcoef(method_matrix.T)
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=['Iso Forest', 'SVM', 'AE', 'Ensemble'],
                yticklabels=['Iso Forest', 'SVM', 'AE', 'Ensemble'],
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('検知手法間の相関')
    
    plt.tight_layout()
    plt.savefig('ensemble_anomaly.png', dpi=300)
    print("\n結果: アンサンブルにより、個別手法の弱点を補完し精度向上")
    

## 3.6 根本原因分析

異常が検出された際、その原因を特定することが重要です。 Granger因果性検定により、時系列データから変数間の因果関係を推定し、根本原因を絞り込みます。 

#### Example 8: Granger因果性検定による根本原因分析

複数のプロセス変数間の因果関係を分析し、故障の原因を特定します。
    
    
    # ===================================
    # Example 8: Granger因果性分析
    # ===================================
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # 因果関係を持つプロセスデータ生成
    np.random.seed(42)
    n = 500
    
    # 原因変数（触媒温度）
    catalyst_temp = np.zeros(n)
    catalyst_temp[0] = 300
    for i in range(1, n):
        catalyst_temp[i] = 0.95 * catalyst_temp[i-1] + 300 * 0.05 + np.random.normal(0, 1)
    
    # 触媒温度が反応速度に影響（遅延あり）
    reaction_rate = np.zeros(n)
    for i in range(3, n):
        reaction_rate[i] = 50 + 0.3 * catalyst_temp[i-2] + np.random.normal(0, 2)
    
    # 反応速度が製品収率に影響
    product_yield = np.zeros(n)
    for i in range(2, n):
        product_yield[i] = 70 + 0.5 * reaction_rate[i-1] + np.random.normal(0, 1.5)
    
    # データフレーム作成
    df_causal = pd.DataFrame({
        'catalyst_temp': catalyst_temp,
        'reaction_rate': reaction_rate,
        'product_yield': product_yield
    })
    
    # Granger因果性検定関数
    def granger_causality_matrix(data, variables, max_lag=5):
        """変数間のGranger因果性をテストし、行列で表示"""
        df_results = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                  columns=variables, index=variables)
    
        for c in variables:
            for r in variables:
                if c != r:
                    test_result = grangercausalitytests(
                        data[[r, c]], max_lag, verbose=False
                    )
                    # 各ラグでの最小p値を使用
                    p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                    min_p = np.min(p_values)
                    df_results.loc[r, c] = min_p
    
        return df_results
    
    # 因果性検定実施
    variables = ['catalyst_temp', 'reaction_rate', 'product_yield']
    causality_matrix = granger_causality_matrix(df_causal[50:], variables, max_lag=5)
    
    print("\nGranger因果性検定結果（p値）:")
    print("列 → 行への因果関係を示す（p < 0.05で有意）")
    print(causality_matrix.round(4))
    
    # 有意な因果関係の抽出
    print("\n有意な因果関係（p < 0.05）:")
    for cause in variables:
        for effect in variables:
            p_value = causality_matrix.loc[effect, cause]
            if p_value < 0.05 and cause != effect:
                print(f"  {cause} → {effect} (p={p_value:.4f})")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 時系列プロット
    axes[0, 0].plot(df_causal['catalyst_temp'], label='触媒温度')
    axes[0, 0].set_ylabel('温度 (°C)')
    axes[0, 0].set_title('触媒温度（原因変数）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df_causal['reaction_rate'], label='反応速度', color='orange')
    axes[0, 1].set_ylabel('速度 (mol/s)')
    axes[0, 1].set_title('反応速度（中間変数）')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df_causal['product_yield'], label='製品収率', color='green')
    axes[1, 0].set_xlabel('時刻')
    axes[1, 0].set_ylabel('収率 (%)')
    axes[1, 0].set_title('製品収率（結果変数）')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 因果性行列ヒートマップ
    significance_matrix = (causality_matrix < 0.05).astype(int)
    sns.heatmap(significance_matrix, annot=causality_matrix.round(3), fmt='',
                cmap='RdYlGn_r', cbar_kws={'label': '有意性'},
                xticklabels=variables, yticklabels=variables,
                ax=axes[1, 1], vmin=0, vmax=1)
    axes[1, 1].set_title('因果関係マップ（緑=有意）')
    axes[1, 1].set_xlabel('原因 →')
    axes[1, 1].set_ylabel('← 結果')
    
    plt.tight_layout()
    plt.savefig('granger_causality.png', dpi=300)
    
    print("\n結果: 触媒温度 → 反応速度 → 製品収率の因果チェーンを特定")
    print("故障診断: 製品収率低下の根本原因は触媒温度異常の可能性が高い")
    

## 3.7 実プロセスへの適用戦略

#### 💡 実装のベストプラクティス

  * **段階的導入** : 統計手法 → 機械学習 → 深層学習の順で試す
  * **閾値調整** : False Alarmを減らすため、運転データで閾値を最適化
  * **アンサンブル活用** : 複数手法の組み合わせで頑健性を向上
  * **可視化重視** : オペレーターが判断できるよう、検知理由を可視化
  * **継続学習** : プロセス変更に応じてモデルを定期的に再訓練

### 手法選択ガイド

状況 | 推奨手法 | 理由  
---|---|---  
単変量、リアルタイム | Z-score / Modified Z-score | 計算コスト低、解釈容易  
多変量、非線形 | Isolation Forest | 高次元に強く、訓練高速  
新規異常パターン | One-Class SVM | 境界学習により未知異常を検知  
複雑な時空間パターン | Autoencoder / LSTM | 高次元の潜在構造を学習  
故障タイプ分類 | Random Forest | 分類精度高、解釈可能  
根本原因分析 | Granger因果性 | 変数間の因果関係を推定  
  
## 3.8 まとめ

本章では、統計手法から深層学習まで8つの異常検知・故障診断技術を実装しました。 各手法の特性を理解し、プロセス特性や運用条件に応じて適切に選択・組み合わせることで、 高精度な異常検知システムを構築できます。 

### 習得したスキル

  * ✅ 統計的手法（Z-score, Modified Z-score）による基本的な異常検知
  * ✅ Isolation Forestによる高次元データの効率的な異常検知
  * ✅ One-Class SVMによる新規性検知と決定境界の可視化
  * ✅ Autoencoderによる再構成誤差ベースの異常検知
  * ✅ LSTMによる時系列パターン学習と予測誤差ベース検知
  * ✅ Random Forestによる故障タイプの多クラス分類
  * ✅ アンサンブル投票による頑健な異常検知システム
  * ✅ Granger因果性検定による根本原因分析

#### 📚 次のステップ

第4章では、プロセス最適化とソフトセンサー技術を学習します。 機械学習モデルを用いた品質予測、リアルタイム最適化、仮想センサーの構築などを実装します。 

## 3.9 演習問題

#### 演習1（基礎）: 統計的異常検知の比較

Example 1のコードを修正し、正規分布に従わないデータ（対数正規分布など）に対する Z-scoreとModified Z-scoreの性能を比較してください。どちらが頑健か説明してください。 

#### 演習2（中級）: Autoencoderのアーキテクチャ最適化

Example 4のAutoencoderについて、以下の実験を行い、最適なアーキテクチャを決定してください： 

  * エンコーディング次元を1, 2, 4, 8で比較
  * 隠れ層の層数を1, 2, 3で比較
  * 活性化関数（ReLU, tanh, ELU）を比較

#### 演習3（上級）: ハイブリッド異常検知システム

以下の要件を満たす実用的な異常検知システムを設計・実装してください： 

  1. 統計手法で1次スクリーニング（計算コスト削減）
  2. 疑わしいデータに対して機械学習で精密検査
  3. 異常検出時にGranger因果性で原因を推定
  4. False Alarm率を5%以下に抑える閾値設定

#### 💡 ヒント

演習3では、カスケード型の検知システムが効果的です。統計手法で高速スクリーニング後、 疑わしいデータのみを深層学習で検証することで、計算コストと精度のバランスが取れます。 ROC曲線を描いて最適な閾値を決定しましょう。 

[← 第2章へ戻る](<#>) [第4章へ進む →](<#>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
