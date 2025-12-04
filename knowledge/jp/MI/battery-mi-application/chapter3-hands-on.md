---
title: Pythonで実装する電池MI実践ハンズオン
chapter_title: Pythonで実装する電池MI実践ハンズオン
subtitle: 実装コードで学ぶ電池材料設計
reading_time: 60-70分
difficulty: 中級
code_examples: 30
exercises: 5
---

# 第3章：Pythonで実装する電池MI実践ハンズオン

**学習目標:** \- PyBaMMで電池モデルを構築・シミュレーションできる \- 容量・電圧予測モデルを実装し評価できる \- LSTMでサイクル劣化を予測できる \- ベイズ最適化で最適材料を探索できる

**前提知識:** \- Python基礎（NumPy, Pandas, Matplotlib） \- 機械学習基礎（scikit-learn） \- 第1章・第2章の内容理解

**実行環境:**
    
    
    pip install pybamm numpy pandas scikit-learn tensorflow scikit-optimize matplotlib seaborn
    

* * *

## 3.1 電池データの取得と前処理

### 例1: Materials Projectから正極材料データ取得
    
    
    from pymatgen.ext.matproj import MPRester
    import pandas as pd
    
    # Materials Project API
    API_KEY = "YOUR_API_KEY"  # https://materialsproject.org/open から取得
    
    with MPRester(API_KEY) as mpr:
        # Li含有酸化物の検索
        data = mpr.query(
            criteria={
                "elements": {"$all": ["Li"], "$in": ["Co", "Ni", "Mn"]},
                "nelements": {"$lte": 4}
            },
            properties=["material_id", "pretty_formula", "energy_per_atom",
                       "band_gap", "formation_energy_per_atom"]
        )
    
    df = pd.DataFrame(data)
    print(f"取得材料数: {len(df)}")
    print(df.head())
    

### 例2: 充放電曲線の読み込みと可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル充放電データ
    def load_charge_discharge_data():
        """充放電曲線データの生成（実際はファイル読込）"""
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
    
    print(f"平均充電電圧: {V_ch.mean():.2f} V")
    print(f"平均放電電圧: {V_dch.mean():.2f} V")
    

### 例3: 電位プロファイル計算
    
    
    from scipy.integrate import cumtrapz
    
    def calculate_average_voltage(capacity, voltage):
        """平均電圧の計算"""
        energy = cumtrapz(voltage, capacity, initial=0)
        avg_voltage = energy[-1] / capacity[-1] if capacity[-1] > 0 else 0
        return avg_voltage
    
    # 充放電の平均電圧
    V_avg_ch = calculate_average_voltage(cap, V_ch)
    V_avg_dch = calculate_average_voltage(cap, V_dch)
    
    # エネルギー密度
    capacity_max = cap[-1]  # mAh/g
    energy_density = capacity_max * V_avg_dch * 0.001  # Wh/g
    
    print(f"平均充電電圧: {V_avg_ch:.3f} V")
    print(f"平均放電電圧: {V_avg_dch:.3f} V")
    print(f"エネルギー密度: {energy_density:.1f} Wh/g")
    

### 例4: 容量計算とクーロン効率
    
    
    def calculate_coulombic_efficiency(Q_charge, Q_discharge):
        """クーロン効率の計算"""
        CE = (Q_discharge / Q_charge) * 100
        return CE
    
    # サンプルデータ
    Q_charge = 195.0  # mAh/g
    Q_discharge = 190.0  # mAh/g
    
    CE = calculate_coulombic_efficiency(Q_charge, Q_discharge)
    print(f"充電容量: {Q_charge:.1f} mAh/g")
    print(f"放電容量: {Q_discharge:.1f} mAh/g")
    print(f"クーロン効率: {CE:.2f}%")
    
    if CE < 98:
        print("⚠️ 警告: クーロン効率が低い（副反応の可能性）")
    elif CE > 99.5:
        print("✅ 優秀: 高いクーロン効率")
    

### 例5: 記述子の自動計算（matminer）
    
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    
    # 正極材料の組成
    compositions = ["LiCoO2", "LiNi0.8Co0.15Al0.05O2", "LiFePO4"]
    
    # 記述子計算
    ep_feat = ElementProperty.from_preset("magpie")
    descriptors = []
    
    for comp_str in compositions:
        comp = Composition(comp_str)
        desc = ep_feat.featurize(comp)
        descriptors.append(desc)
    
    # DataFrame化
    feature_labels = ep_feat.feature_labels()
    df_desc = pd.DataFrame(descriptors, columns=feature_labels, index=compositions)
    
    print("記述子の例（最初の5列）:")
    print(df_desc.iloc[:, :5])
    print(f"\n総記述子数: {len(feature_labels)}")
    

### 例6: データクリーニングと外れ値除去
    
    
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    
    # サンプルデータ
    np.random.seed(42)
    capacity_data = np.concatenate([
        np.random.normal(180, 10, 95),  # 正常データ
        np.array([250, 280, 300, 310, 50])  # 外れ値
    ])
    
    # Z-scoreによる外れ値検出
    z_scores = np.abs(stats.zscore(capacity_data))
    threshold = 3
    outliers = z_scores > threshold
    
    print(f"データ数: {len(capacity_data)}")
    print(f"外れ値数: {outliers.sum()} ({outliers.sum()/len(capacity_data)*100:.1f}%)")
    print(f"外れ値: {capacity_data[outliers]}")
    
    # クリーニング後
    capacity_clean = capacity_data[~outliers]
    print(f"クリーニング後データ数: {len(capacity_clean)}")
    print(f"平均容量: {capacity_clean.mean():.1f} ± {capacity_clean.std():.1f} mAh/g")
    

### 例7: Train/Testデータ分割
    
    
    from sklearn.model_selection import train_test_split
    
    # サンプルデータセット
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)  # 10個の記述子
    y = 150 + 30 * X[:, 0] - 20 * X[:, 1] + np.random.randn(n_samples) * 5  # 容量
    
    # Train/Test分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    print(f"容量範囲: {y.min():.1f} - {y.max():.1f} mAh/g")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n標準化完了")
    

* * *

## 3.2 容量・電圧予測モデル

### 例8: Random Forest回帰（容量予測）
    
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # モデル訓練
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred_train = model_rf.predict(X_train_scaled)
    y_pred_test = model_rf.predict(X_test_scaled)
    
    # 評価
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"Random Forest 容量予測:")
    print(f"  訓練: MAE={mae_train:.2f} mAh/g, R²={r2_train:.3f}")
    print(f"  テスト: MAE={mae_test:.2f} mAh/g, R²={r2_test:.3f}")
    

### 例9: XGBoost（電圧予測）
    
    
    from xgboost import XGBRegressor
    
    # 電圧データ（サンプル）
    y_voltage = 3.7 + 0.3 * X[:, 0] - 0.2 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    y_v_train, y_v_test = y_voltage[:len(X_train)], y_voltage[len(X_train):]
    
    # XGBoostモデル
    model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model_xgb.fit(X_train_scaled, y_v_train)
    
    # 予測と評価
    y_v_pred = model_xgb.predict(X_test_scaled)
    mae_voltage = mean_absolute_error(y_v_test, y_v_pred)
    r2_voltage = r2_score(y_v_test, y_v_pred)
    
    print(f"XGBoost 電圧予測:")
    print(f"  MAE: {mae_voltage:.3f} V")
    print(f"  R²: {r2_voltage:.3f}")
    

### 例10: Neural Network（Keras）
    
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # モデル構築
    model_nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # 訓練
    history = model_nn.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # 評価
    y_nn_pred = model_nn.predict(X_test_scaled).flatten()
    mae_nn = mean_absolute_error(y_test, y_nn_pred)
    r2_nn = r2_score(y_test, y_nn_pred)
    
    print(f"Neural Network 容量予測:")
    print(f"  MAE: {mae_nn:.2f} mAh/g")
    print(f"  R²: {r2_nn:.3f}")
    

### 例11: Graph Neural Network（概念実装）
    
    
    # PyTorch Geometricを使用（実装の概要）
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
    
    # 訓練・予測（詳細は第4章）
    """
    
    print("Graph Neural Networkの概念:")
    print("  入力: 結晶構造（原子座標、結合情報）")
    print("  処理: Graph Convolution Layers")
    print("  出力: 容量、電圧予測")
    print("  利点: 記述子設計不要、高精度")
    

### 例12: Transfer Learning
    
    
    from tensorflow.keras.models import load_model
    
    # 事前学習済みモデル（仮想）
    def create_pretrained_model():
        """LIB正極材料で訓練済みモデル"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        return model
    
    pretrained = create_pretrained_model()
    pretrained.compile(optimizer=Adam(lr=0.001), loss='mse')
    # 仮想訓練
    pretrained.fit(X_train_scaled[:100], y_train[:100], epochs=50, verbose=0)
    
    # ファインチューニング（全固体電池データ）
    X_target = X_train_scaled[100:120]
    y_target = y_train[100:120]
    
    # 最終層を再訓練
    for layer in pretrained.layers[:-1]:
        layer.trainable = False
    
    pretrained.compile(optimizer=Adam(lr=1e-4), loss='mse')
    pretrained.fit(X_target, y_target, epochs=30, verbose=0)
    
    # 評価
    y_tl_pred = pretrained.predict(X_test_scaled).flatten()
    mae_tl = mean_absolute_error(y_test, y_tl_pred)
    
    print(f"Transfer Learning:")
    print(f"  ソース: LIB正極（100サンプル）")
    print(f"  ターゲット: 全固体電池（20サンプル）")
    print(f"  MAE: {mae_tl:.2f} mAh/g")
    

### 例13: 特徴量重要度分析（SHAP）
    
    
    import shap
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(X_test_scaled[:100])
    
    # 特徴量重要度
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names, show=False)
    
    print("SHAP分析:")
    print("  各特徴量の容量への寄与を定量化")
    print("  正の寄与: 容量増加")
    print("  負の寄与: 容量減少")
    

### 例14: 交差検証
    
    
    from sklearn.model_selection import cross_val_score
    
    # 5-fold交差検証
    cv_scores = cross_val_score(
        model_rf, X_train_scaled, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    
    print(f"5-fold交差検証:")
    print(f"  MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} mAh/g")
    print(f"  各fold: {-cv_scores}")
    

### 例15: Parity Plot
    
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
           'r--', linewidth=2, label='Ideal')
    
    ax.set_xlabel('Actual Capacity (mAh/g)', fontsize=12)
    ax.set_ylabel('Predicted Capacity (mAh/g)', fontsize=12)
    ax.set_title(f'Parity Plot (MAE={mae_test:.2f} mAh/g)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    print("Parity Plot: 予測値 vs 実測値の比較")
    

* * *

## 3.3 サイクル劣化予測

### 例16: 充放電曲線の時系列データ準備
    
    
    def generate_cycle_data(n_cycles=500):
        """サイクルデータの生成"""
        cycles = np.arange(1, n_cycles + 1)
    
        # 容量減衰（指数関数的）
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
    
    print(f"初期容量: {capacity[0]:.1f} mAh/g")
    print(f"最終容量: {capacity[-1]:.1f} mAh/g")
    print(f"容量保持率: {SOH[-1]:.1f}%")
    

### 例17: LSTM（Long Short-Term Memory）モデル
    
    
    from tensorflow.keras.layers import LSTM
    
    # データ準備（時系列ウィンドウ）
    def create_sequences(data, seq_length=50):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 50
    X_lstm, y_lstm = create_sequences(capacity, seq_length)
    X_lstm = X_lstm.reshape(-1, seq_length, 1)
    
    # Train/Test分割
    split = int(0.8 * len(X_lstm))
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
    
    # LSTMモデル
    model_lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_lstm.compile(optimizer=Adam(lr=0.001), loss='mse')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    # 予測
    y_lstm_pred = model_lstm.predict(X_test_lstm).flatten()
    mae_lstm = mean_absolute_error(y_test_lstm, y_lstm_pred)
    
    print(f"LSTM 劣化予測:")
    print(f"  MAE: {mae_lstm:.2f} mAh/g")
    print(f"  シーケンス長: {seq_length}サイクル")
    

### 例18: GRU（Gated Recurrent Unit）モデル
    
    
    from tensorflow.keras.layers import GRU
    
    # GRUモデル（LSTMより軽量）
    model_gru = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
        GRU(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_gru.compile(optimizer=Adam(lr=0.001), loss='mse')
    model_gru.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    # 予測
    y_gru_pred = model_gru.predict(X_test_lstm).flatten()
    mae_gru = mean_absolute_error(y_test_lstm, y_gru_pred)
    
    print(f"GRU 劣化予測:")
    print(f"  MAE: {mae_gru:.2f} mAh/g")
    print(f"  LSTM比較: パラメータ数 {model_gru.count_params()} vs {model_lstm.count_params()}")
    

### 例19: 寿命予測（RUL: Remaining Useful Life）
    
    
    def predict_RUL(capacity_history, threshold=160):
        """80%容量（160 mAh/g）到達までのサイクル数予測"""
        # 初期100サイクルから予測
        early_cycles = capacity_history[:100]
    
        # 線形フィット
        x = np.arange(len(early_cycles))
        coeffs = np.polyfit(x, early_cycles, 1)
        decay_rate = -coeffs[0]
    
        # RUL計算
        current_capacity = early_cycles[-1]
        remaining = current_capacity - threshold
        RUL = int(remaining / decay_rate) if decay_rate > 0 else np.inf
    
        return RUL, decay_rate
    
    RUL, decay = predict_RUL(capacity)
    actual_life = np.where(capacity < 160)[0][0] if np.any(capacity < 160) else len(capacity)
    
    print(f"寿命予測（初期100サイクルから）:")
    print(f"  予測RUL: {RUL}サイクル")
    print(f"  実際の寿命: {actual_life}サイクル")
    print(f"  予測誤差: {abs(RUL - actual_life)}サイクル ({abs(RUL - actual_life)/actual_life*100:.1f}%)")
    print(f"  劣化速度: {decay:.3f} mAh/g/cycle")
    

### 例20: 劣化速度の予測
    
    
    def analyze_degradation_rate(capacity, window=50):
        """移動ウィンドウでの劣化速度解析"""
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
    
    print(f"平均劣化速度: {degradation_rates.mean():.4f} mAh/g/cycle")
    print(f"最大劣化速度: {degradation_rates.max():.4f} mAh/g/cycle (cycle {cycles_rate[degradation_rates.argmax()]})")
    

### 例21: 異常検知（Isolation Forest）
    
    
    from sklearn.ensemble import IsolationForest
    
    # 特徴量: 容量、劣化速度
    features = np.column_stack([capacity[50:], degradation_rates])
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    anomalies = clf.fit_predict(features)
    
    n_anomalies = (anomalies == -1).sum()
    print(f"異常検知:")
    print(f"  異常サイクル数: {n_anomalies}")
    print(f"  異常率: {n_anomalies/len(anomalies)*100:.1f}%")
    print(f"  異常サイクル: {cycles_rate[anomalies == -1]}")
    
    # 可視化
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
    

### 例22: SOH（State of Health）推定
    
    
    def estimate_SOH(current_capacity, initial_capacity=200):
        """SOH推定"""
        SOH = (current_capacity / initial_capacity) * 100
    
        if SOH > 95:
            status = "優秀"
        elif SOH > 80:
            status = "良好"
        elif SOH > 70:
            status = "劣化進行中"
        else:
            status = "要交換"
    
        return SOH, status
    
    # 各サイクルでのSOH推定
    for cycle in [100, 200, 300, 400, 500]:
        if cycle <= len(capacity):
            soh, status = estimate_SOH(capacity[cycle-1])
            print(f"Cycle {cycle:3d}: SOH={soh:5.1f}%, 容量={capacity[cycle-1]:5.1f} mAh/g, 状態={status}")
    

* * *

## 3.4 ベイズ最適化による材料探索

### 例23: Gaussian Process回帰
    
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    
    # サンプルデータ（Ni比率 vs 容量）
    X_gp = np.array([[0.3], [0.5], [0.6], [0.7], [0.9]])
    y_gp = np.array([160, 180, 195, 190, 170])
    
    # GPRモデル
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X_gp, y_gp)
    
    # 予測
    X_pred = np.linspace(0.2, 1.0, 100).reshape(-1, 1)
    y_pred, y_std = gpr.predict(X_pred, return_std=True)
    
    # 可視化
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
    
    print(f"最適Ni比率（予測）: {X_pred[np.argmax(y_pred)][0]:.2f}")
    print(f"最大予測容量: {y_pred.max():.1f} mAh/g")
    

### 例24: ベイズ最適化ループ
    
    
    from skopt import gp_minimize
    from skopt.space import Real
    
    # 目的関数（容量シミュレーション）
    def battery_capacity(x):
        """Ni比率から容量を予測（実際は実験 or DFT計算）"""
        ni_ratio = x[0]
        # 仮想的な容量関数
        capacity = 200 * ni_ratio - 150 * (ni_ratio - 0.65)**2 + np.random.randn() * 3
        return -capacity  # 最小化問題に変換
    
    # 探索空間
    space = [Real(0.3, 1.0, name='Ni_ratio')]
    
    # ベイズ最適化
    result = gp_minimize(
        battery_capacity,
        space,
        n_calls=20,
        random_state=42,
        verbose=False
    )
    
    print(f"ベイズ最適化結果:")
    print(f"  最適Ni比率: {result.x[0]:.3f}")
    print(f"  最大容量: {-result.fun:.1f} mAh/g")
    print(f"  実験回数: {len(result.x_iters)}")
    

### 例25: 多目的最適化（容量 & サイクル寿命）
    
    
    def multi_objective(x):
        """容量とサイクル寿命のトレードオフ"""
        ni_ratio = x[0]
    
        # 容量（高Ni比率で増加）
        capacity = 200 * ni_ratio - 100 * (ni_ratio - 0.7)**2
    
        # サイクル寿命（低Ni比率で向上）
        cycle_life = 2000 - 1000 * ni_ratio + 500 * (ni_ratio - 0.5)**2
    
        # 重み付き和（スカラー化）
        weight_cap = 0.6
        weight_life = 0.4
    
        score = weight_cap * capacity + weight_life * (cycle_life / 10)
        return -score
    
    result_mo = gp_minimize(multi_objective, space, n_calls=25, random_state=42)
    
    print(f"多目的最適化結果:")
    print(f"  最適Ni比率: {result_mo.x[0]:.3f}")
    print(f"  予測容量: {(200 * result_mo.x[0] - 100 * (result_mo.x[0] - 0.7)**2):.1f} mAh/g")
    print(f"  予測寿命: {(2000 - 1000 * result_mo.x[0] + 500 * (result_mo.x[0] - 0.5)**2):.0f} cycles")
    

### 例26: 制約付き最適化
    
    
    def constrained_optimization(x):
        """安全性制約付き容量最適化"""
        ni_ratio = x[0]
    
        # 制約: Ni比率 < 0.85（安全性考慮）
        if ni_ratio > 0.85:
            return 1e6  # ペナルティ
    
        # 容量予測
        capacity = 200 * ni_ratio - 120 * (ni_ratio - 0.7)**2
        return -capacity
    
    result_const = gp_minimize(constrained_optimization, space, n_calls=20, random_state=42)
    
    print(f"制約付き最適化結果:")
    print(f"  最適Ni比率: {result_const.x[0]:.3f} (< 0.85)")
    print(f"  最大容量: {-result_const.fun:.1f} mAh/g")
    

### 例27: パレートフロント可視化
    
    
    # 多目的最適化の結果（容量 vs サイクル寿命）
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
    
    # パレート最適点の検出
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
    
    print(f"パレート最適解の数: {len(pareto_indices)}")
    

* * *

## 3.5 PyBaMMによる電池シミュレーション

### 例28: DFNモデル（Doyle-Fuller-Newman）
    
    
    import pybamm
    
    # DFNモデルの構築
    model = pybamm.lithium_ion.DFN()
    
    # パラメータ設定（Graphite || LCO）
    parameter_values = pybamm.ParameterValues("Chen2020")
    
    # シミュレーション設定
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    
    # 1C放電
    sim.solve([0, 3600])  # 0-3600秒（1時間）
    
    # 結果取得
    time = sim.solution["Time [h]"].entries
    voltage = sim.solution["Terminal voltage [V]"].entries
    current = sim.solution["Current [A]"].entries
    
    print("DFNモデルシミュレーション:")
    print(f"  初期電圧: {voltage[0]:.3f} V")
    print(f"  最終電圧: {voltage[-1]:.3f} V")
    print(f"  放電時間: {time[-1]:.2f} h")
    

### 例29: 充放電曲線シミュレーション
    
    
    # 複数のC-rateでシミュレーション
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
    
    # 可視化
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
    
    print("複数C-rateでの放電シミュレーション完了")
    

### 例30: パラメータ最適化とフィッティング
    
    
    # 実験データ（サンプル）
    experimental_voltage = voltage + np.random.randn(len(voltage)) * 0.05
    
    # パラメータ最適化（簡易版）
    def fit_resistance(R_value):
        """内部抵抗のフィッティング"""
        params = parameter_values.copy()
        params["Electrolyte conductivity [S.m-1]"] = R_value
    
        sim_fit = pybamm.Simulation(model, parameter_values=params)
        sim_fit.solve([0, 3600])
    
        sim_voltage = sim_fit.solution["Terminal voltage [V]"].entries
    
        # 誤差計算
        mse = np.mean((sim_voltage - experimental_voltage)**2)
        return mse
    
    # 最適化
    from scipy.optimize import minimize_scalar
    
    result_fit = minimize_scalar(fit_resistance, bounds=(0.5, 2.0), method='bounded')
    
    print(f"パラメータフィッティング:")
    print(f"  最適電解質伝導度: {result_fit.x:.3f} S/m")
    print(f"  MSE: {result_fit.fun:.6f}")
    
    # 最適パラメータでシミュレーション
    params_opt = parameter_values.copy()
    params_opt["Electrolyte conductivity [S.m-1]"] = result_fit.x
    sim_opt = pybamm.Simulation(model, parameter_values=params_opt)
    sim_opt.solve([0, 3600])
    
    print("最適化完了: 実験データにフィット")
    

* * *

## 3.6 プロジェクトチャレンジ

**課題: 高容量・長寿命正極材料の発見**

以下の手順で、最適なNCM正極材料を設計してください：

  1. **データ収集** : Materials ProjectからNi-Co-Mn酸化物データ取得
  2. **記述子計算** : 組成比、格子定数、バンドギャップなど
  3. **予測モデル構築** : XGBoostで容量予測（目標 > 200 mAh/g）
  4. **ベイズ最適化** : Ni:Co:Mn比の最適化（制約: 安全性）
  5. **サイクル性能評価** : PyBaMMでサイクル寿命シミュレーション（目標 > 2,000サイクル）

**評価基準:** \- 容量 > 200 mAh/g \- サイクル寿命 > 2,000サイクル（80%容量維持） \- 安全性: Ni比率 < 0.85 \- コスト: Co使用量最小化

**提出物:** \- 最適組成（Ni:Co:Mn比率） \- 予測性能（容量、寿命） \- Pythonコード全体

* * *

## 演習問題

**問1:** LiNi₀.₈Co₀.₁Mn₀.₁O₂の理論容量を計算し、実測容量が180 mAh/gの場合のクーロン効率を求めよ。

**問2:** LSTM RNN構造は、Feed-forward Neural Networkと比較してサイクル劣化予測においてなぜ優れているか説明せよ。

**問3:** ベイズ最適化で、獲得関数としてEI（Expected Improvement）を用いる利点を2つ挙げよ。

**問4:** PyBaMMのDFNモデルで、負極材料をグラファイトからシリコンに変更した場合の影響を予測せよ。

**問5:** Transfer Learningを用いて、LIB正極材料の知識をNa-ion電池正極材料に適用する際の課題を論じよ（400字以内）。

* * *

## 参考文献

  1. Sulzer, V. et al. "Python Battery Mathematical Modelling (PyBaMM)." _JOSS_ (2021).
  2. Severson, K. A. et al. "Data-driven prediction of battery cycle life." _Nat. Energy_ (2019).
  3. Chen, C. et al. "A Critical Review of Machine Learning of Energy Materials." _Adv. Energy Mater._ (2020).
  4. Attia, P. M. et al. "Closed-loop optimization of fast-charging protocols." _Nature_ (2020).

* * *

**次章** : [第4章：電池開発の最新事例と産業応用](<chapter4-case-studies.html>)

**ライセンス** : このコンテンツはCC BY 4.0ライセンスの下で提供されています。
