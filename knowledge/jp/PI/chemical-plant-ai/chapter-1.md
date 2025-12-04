---
title: 第1章：プロセス監視とソフトセンサー
chapter_title: 第1章：プロセス監視とソフトセンサー
subtitle: AIベース異常検知と品質予測の実装
reading_time: 30-35分
difficulty: 実践・応用
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 統計的異常検知（PCA、Q統計量、T²統計量）を実装できる
  * ✅ 機械学習ベース異常検知（Isolation Forest、Autoencoder、LSTM）を構築できる
  * ✅ 品質予測モデル（Random Forest）で製品品質を予測できる
  * ✅ ソフトセンサー（GPR、ニューラルネット）で測定困難な変数を推定できる
  * ✅ 統合プロセス監視システムを設計・実装できる

* * *

## 1.1 化学プラント監視の課題とAI技術

### 化学プラント特有の監視課題

化学プラントのプロセス監視は、製品品質、安全性、経済性を確保するための最重要課題です。従来の閾値ベース監視では検出困難な異常が多数存在します：

  * **多変量相関異常** : 個別変数は正常範囲内でも、変数間の相関が異常
  * **緩やかな劣化** : 触媒活性低下、熱交換器汚れなど、数週間～数ヶ月単位の変化
  * **測定困難変数** : 製品品質（純度、粘度）、反応率などのオンライン測定が困難
  * **非線形挙動** : 反応器の非線形動特性、蒸留塔の複雑な相互作用

### AI技術による解決アプローチ
    
    
    ```mermaid
    graph TD
        A[プロセス監視課題] --> B[統計的手法]
        A --> C[機械学習]
        A --> D[深層学習]
    
        B --> B1[PCA異常検知]
        B --> B2[統計的プロセス管理]
    
        C --> C1[Isolation Forest]
        C --> C2[Random Forest品質予測]
        C --> C3[GPRソフトセンサー]
    
        D --> D1[Autoencoder異常検知]
        D --> D2[LSTM時系列予測]
        D --> D3[NN-ソフトセンサー]
    
        style A fill:#11998e,stroke:#0d7a6f,color:#fff
        style B fill:#38ef7d,stroke:#2bc766,color:#333
        style C fill:#38ef7d,stroke:#2bc766,color:#333
        style D fill:#38ef7d,stroke:#2bc766,color:#333
    ```

* * *

## 1.2 統計的異常検知の実装

#### コード例1: PCA法による多変量統計的プロセス監視

**目的** : 主成分分析（PCA）を用いてQ統計量（SPE）とT²統計量でプロセス異常を検出する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy import stats
    
    # 日本語フォント設定
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 化学反応器の正常運転データ生成（訓練データ）
    np.random.seed(42)
    n_samples_normal = 500
    
    # プロセス変数: 温度、圧力、流量、濃度（相関あり）
    temperature = np.random.normal(350, 10, n_samples_normal)  # K
    pressure = 2.0 + 0.01 * (temperature - 350) + np.random.normal(0, 0.1, n_samples_normal)  # bar
    flow_rate = 100 + 0.5 * (temperature - 350) + np.random.normal(0, 5, n_samples_normal)  # L/h
    concentration = 0.8 - 0.001 * (temperature - 350) + np.random.normal(0, 0.05, n_samples_normal)  # mol/L
    
    # 正常運転データ
    X_normal = np.column_stack([temperature, pressure, flow_rate, concentration])
    
    # データ標準化
    mean = X_normal.mean(axis=0)
    std = X_normal.std(axis=0)
    X_scaled = (X_normal - mean) / std
    
    # PCAモデルの構築（主成分数=2）
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    print("=== PCAモデル構築 ===")
    print(f"累積寄与率: {pca.explained_variance_ratio_.cumsum()}")
    print(f"主成分1の寄与率: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"主成分2の寄与率: {pca.explained_variance_ratio_[1]:.3f}")
    
    # Q統計量（SPE: Squared Prediction Error）の計算
    def compute_Q_statistic(X, pca_model):
        """Q統計量（残差空間のノルム）を計算"""
        X_reconstructed = pca_model.inverse_transform(pca_model.transform(X))
        residuals = X - X_reconstructed
        Q = np.sum(residuals**2, axis=1)
        return Q
    
    # T²統計量（Hotelling's T-squared）の計算
    def compute_T2_statistic(X, pca_model):
        """T²統計量（主成分空間の距離）を計算"""
        scores = pca_model.transform(X)
        eigenvalues = pca_model.explained_variance_
        T2 = np.sum((scores**2) / eigenvalues, axis=1)
        return T2
    
    # 正常運転データの統計量
    Q_normal = compute_Q_statistic(X_scaled, pca)
    T2_normal = compute_T2_statistic(X_scaled, pca)
    
    # 管理限界の計算（99%信頼区間）
    Q_limit = np.percentile(Q_normal, 99)
    T2_limit = np.percentile(T2_normal, 99)
    
    print(f"\n管理限界:")
    print(f"Q統計量限界: {Q_limit:.3f}")
    print(f"T²統計量限界: {T2_limit:.3f}")
    
    # 異常データの生成（テストデータ）
    n_samples_test = 100
    
    # ケース1: 温度異常（反応暴走）
    temp_anomaly = np.random.normal(380, 10, 20)  # 高温異常
    press_anomaly = 2.0 + 0.01 * (temp_anomaly - 350) + np.random.normal(0, 0.1, 20)
    flow_anomaly = 100 + 0.5 * (temp_anomaly - 350) + np.random.normal(0, 5, 20)
    conc_anomaly = 0.8 - 0.001 * (temp_anomaly - 350) + np.random.normal(0, 0.05, 20)
    
    # ケース2: 相関異常（センサー故障）
    temp_corr_anomaly = np.random.normal(350, 10, 20)
    press_corr_anomaly = np.random.normal(2.0, 0.5, 20)  # 圧力の相関が崩れる
    flow_corr_anomaly = 100 + 0.5 * (temp_corr_anomaly - 350) + np.random.normal(0, 5, 20)
    conc_corr_anomaly = 0.8 - 0.001 * (temp_corr_anomaly - 350) + np.random.normal(0, 0.05, 20)
    
    # 正常データ（比較用）
    temp_test = np.random.normal(350, 10, 60)
    press_test = 2.0 + 0.01 * (temp_test - 350) + np.random.normal(0, 0.1, 60)
    flow_test = 100 + 0.5 * (temp_test - 350) + np.random.normal(0, 5, 60)
    conc_test = 0.8 - 0.001 * (temp_test - 350) + np.random.normal(0, 0.05, 60)
    
    # テストデータ結合
    X_test = np.vstack([
        np.column_stack([temp_test, press_test, flow_test, conc_test]),
        np.column_stack([temp_anomaly, press_anomaly, flow_anomaly, conc_anomaly]),
        np.column_stack([temp_corr_anomaly, press_corr_anomaly, flow_corr_anomaly, conc_corr_anomaly])
    ])
    
    # ラベル（0: 正常, 1: 温度異常, 2: 相関異常）
    labels = np.array([0]*60 + [1]*20 + [2]*20)
    
    # テストデータの標準化
    X_test_scaled = (X_test - mean) / std
    
    # テストデータの統計量計算
    Q_test = compute_Q_statistic(X_test_scaled, pca)
    T2_test = compute_T2_statistic(X_test_scaled, pca)
    
    # 異常検出
    anomaly_Q = Q_test > Q_limit
    anomaly_T2 = T2_test > T2_limit
    anomaly_combined = anomaly_Q | anomaly_T2
    
    print(f"\n異常検出結果:")
    print(f"Q統計量による検出数: {anomaly_Q.sum()}/{len(Q_test)}")
    print(f"T²統計量による検出数: {anomaly_T2.sum()}/{len(T2_test)}")
    print(f"統合検出数: {anomaly_combined.sum()}/{len(Q_test)}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Q統計量の時系列プロット
    axes[0, 0].plot(Q_test, 'o-', markersize=4, linewidth=0.8, color='#11998e', label='Q統計量')
    axes[0, 0].axhline(y=Q_limit, color='red', linestyle='--', linewidth=2, label=f'管理限界 (99%)')
    axes[0, 0].fill_between(range(len(Q_test)), 0, Q_limit, alpha=0.1, color='green')
    axes[0, 0].scatter(np.where(labels==1)[0], Q_test[labels==1], color='orange', s=80,
                       marker='x', linewidths=3, label='温度異常', zorder=5)
    axes[0, 0].scatter(np.where(labels==2)[0], Q_test[labels==2], color='purple', s=80,
                       marker='^', linewidths=3, label='相関異常', zorder=5)
    axes[0, 0].set_xlabel('サンプル番号', fontsize=11)
    axes[0, 0].set_ylabel('Q統計量', fontsize=11)
    axes[0, 0].set_title('Q統計量による異常検知（残差空間）', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # T²統計量の時系列プロット
    axes[0, 1].plot(T2_test, 'o-', markersize=4, linewidth=0.8, color='#38ef7d', label='T²統計量')
    axes[0, 1].axhline(y=T2_limit, color='red', linestyle='--', linewidth=2, label=f'管理限界 (99%)')
    axes[0, 1].fill_between(range(len(T2_test)), 0, T2_limit, alpha=0.1, color='green')
    axes[0, 1].scatter(np.where(labels==1)[0], T2_test[labels==1], color='orange', s=80,
                       marker='x', linewidths=3, label='温度異常', zorder=5)
    axes[0, 1].scatter(np.where(labels==2)[0], T2_test[labels==2], color='purple', s=80,
                       marker='^', linewidths=3, label='相関異常', zorder=5)
    axes[0, 1].set_xlabel('サンプル番号', fontsize=11)
    axes[0, 1].set_ylabel('T²統計量', fontsize=11)
    axes[0, 1].set_title('T²統計量による異常検知（主成分空間）', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Q-T²プロット
    axes[1, 0].scatter(Q_test[labels==0], T2_test[labels==0], c='blue', s=30, alpha=0.6, label='正常')
    axes[1, 0].scatter(Q_test[labels==1], T2_test[labels==1], c='orange', s=80, marker='x',
                       linewidths=3, label='温度異常')
    axes[1, 0].scatter(Q_test[labels==2], T2_test[labels==2], c='purple', s=80, marker='^',
                       linewidths=3, label='相関異常')
    axes[1, 0].axvline(x=Q_limit, color='red', linestyle='--', alpha=0.5, label='Q限界')
    axes[1, 0].axhline(y=T2_limit, color='red', linestyle='--', alpha=0.5, label='T²限界')
    axes[1, 0].set_xlabel('Q統計量', fontsize=11)
    axes[1, 0].set_ylabel('T²統計量', fontsize=11)
    axes[1, 0].set_title('Q-T²プロット（異常診断）', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 主成分スコアプロット
    scores = pca.transform(X_test_scaled)
    axes[1, 1].scatter(scores[labels==0, 0], scores[labels==0, 1], c='blue', s=30, alpha=0.6, label='正常')
    axes[1, 1].scatter(scores[labels==1, 0], scores[labels==1, 1], c='orange', s=80, marker='x',
                       linewidths=3, label='温度異常')
    axes[1, 1].scatter(scores[labels==2, 0], scores[labels==2, 1], c='purple', s=80, marker='^',
                       linewidths=3, label='相関異常')
    axes[1, 1].set_xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    axes[1, 1].set_ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    axes[1, 1].set_title('主成分スコアプロット', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : PCAベース監視は化学プラントで最も広く使用される統計的手法です。Q統計量は残差空間の異常（センサー故障、相関崩れ）を検出し、T²統計量は主成分空間の異常（プロセス変動）を検出します。両者を組み合わせることで、異なる種類の異常を診断できます。

#### コード例2: Isolation Forestによる多変量異常検知

**目的** : Isolation Forestアルゴリズムでプロセス異常を検出し、異常スコアを可視化する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import classification_report, confusion_matrix
    
    np.random.seed(42)
    
    # 蒸留塔の運転データ生成
    n_normal = 800
    n_anomaly = 50
    
    # 正常運転データ
    塔頂温度_正常 = np.random.normal(85, 2, n_normal)  # °C
    塔底温度_正常 = np.random.normal(155, 3, n_normal)  # °C
    還流比_正常 = np.random.normal(3.5, 0.3, n_normal)
    製品純度_正常 = 0.98 + 0.01 * (塔頂温度_正常 - 85) / 10 + np.random.normal(0, 0.005, n_normal)
    
    # 異常運転データ（複数の異常パターン）
    # パターン1: 塔頂温度異常（冷却器故障）
    塔頂温度_異常1 = np.random.normal(95, 3, 20)
    塔底温度_異常1 = np.random.normal(155, 3, 20)
    還流比_異常1 = np.random.normal(3.5, 0.3, 20)
    製品純度_異常1 = 0.98 + 0.01 * (塔頂温度_異常1 - 85) / 10 + np.random.normal(0, 0.01, 20)
    
    # パターン2: 還流比異常（ポンプ故障）
    塔頂温度_異常2 = np.random.normal(85, 2, 15)
    塔底温度_異常2 = np.random.normal(155, 3, 15)
    還流比_異常2 = np.random.normal(2.0, 0.5, 15)
    製品純度_異常2 = 0.85 + np.random.normal(0, 0.02, 15)
    
    # パターン3: 複合異常（原料組成変動）
    塔頂温度_異常3 = np.random.normal(90, 4, 15)
    塔底温度_異常3 = np.random.normal(165, 5, 15)
    還流比_異常3 = np.random.normal(4.5, 0.5, 15)
    製品純度_異常3 = 0.92 + np.random.normal(0, 0.015, 15)
    
    # データ統合
    X = np.vstack([
        np.column_stack([塔頂温度_正常, 塔底温度_正常, 還流比_正常, 製品純度_正常]),
        np.column_stack([塔頂温度_異常1, 塔底温度_異常1, 還流比_異常1, 製品純度_異常1]),
        np.column_stack([塔頂温度_異常2, 塔底温度_異常2, 還流比_異常2, 製品純度_異常2]),
        np.column_stack([塔頂温度_異常3, 塔底温度_異常3, 還流比_異常3, 製品純度_異常3])
    ])
    
    # ラベル（1: 正常, -1: 異常）
    y_true = np.array([1]*n_normal + [-1]*n_anomaly)
    
    # DataFrameに変換
    df = pd.DataFrame(X, columns=['塔頂温度', '塔底温度', '還流比', '製品純度'])
    df['ラベル'] = y_true
    
    # Isolation Forestモデルの訓練
    iso_forest = IsolationForest(
        contamination=0.05,  # 異常データの割合（5%）
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    # 全データで訓練（実務では正常データのみで訓練）
    iso_forest.fit(X)
    
    # 異常予測
    y_pred = iso_forest.predict(X)
    anomaly_scores = iso_forest.decision_function(X)  # 異常スコア（負の値ほど異常）
    
    # 性能評価
    print("=== Isolation Forest 異常検知性能 ===")
    print("\n混同行列:")
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    print(cm)
    print("\n分類レポート:")
    print(classification_report(y_true, y_pred, target_names=['正常', '異常']))
    
    # 異常スコアの統計
    print(f"\n異常スコア統計:")
    print(f"正常データの平均スコア: {anomaly_scores[y_true==1].mean():.4f}")
    print(f"異常データの平均スコア: {anomaly_scores[y_true==-1].mean():.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 異常スコアの時系列プロット
    axes[0, 0].plot(anomaly_scores, 'o-', markersize=3, linewidth=0.6, color='#11998e')
    axes[0, 0].scatter(np.where(y_true==-1)[0], anomaly_scores[y_true==-1],
                       color='red', s=50, marker='x', linewidths=2, label='真の異常', zorder=5)
    axes[0, 0].axhline(y=0, color='orange', linestyle='--', linewidth=2, label='判定境界')
    axes[0, 0].set_xlabel('サンプル番号', fontsize=11)
    axes[0, 0].set_ylabel('異常スコア', fontsize=11)
    axes[0, 0].set_title('Isolation Forest 異常スコア', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 異常スコアのヒストグラム
    axes[0, 1].hist(anomaly_scores[y_true==1], bins=30, alpha=0.6, color='blue', label='正常', edgecolor='black')
    axes[0, 1].hist(anomaly_scores[y_true==-1], bins=15, alpha=0.8, color='red', label='異常', edgecolor='black')
    axes[0, 1].axvline(x=0, color='orange', linestyle='--', linewidth=2, label='判定境界')
    axes[0, 1].set_xlabel('異常スコア', fontsize=11)
    axes[0, 1].set_ylabel('頻度', fontsize=11)
    axes[0, 1].set_title('異常スコア分布', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 塔頂温度 vs 製品純度（異常パターン可視化）
    colors = ['blue' if label == 1 else 'red' for label in y_pred]
    axes[1, 0].scatter(df['塔頂温度'], df['製品純度'], c=colors, s=30, alpha=0.6)
    axes[1, 0].set_xlabel('塔頂温度 (°C)', fontsize=11)
    axes[1, 0].set_ylabel('製品純度', fontsize=11)
    axes[1, 0].set_title('塔頂温度 vs 製品純度（異常検出結果）', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 還流比 vs 製品純度
    axes[1, 1].scatter(df['還流比'], df['製品純度'], c=colors, s=30, alpha=0.6)
    axes[1, 1].set_xlabel('還流比', fontsize=11)
    axes[1, 1].set_ylabel('製品純度', fontsize=11)
    axes[1, 1].set_title('還流比 vs 製品純度（異常検出結果）', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : Isolation Forestは、異常データが正常データよりも「分離しやすい」という性質を利用した教師なし学習アルゴリズムです。統計的仮定が不要で、非線形な異常パターンも検出でき、化学プラントの多様な異常に対応できます。計算コストが低く、リアルタイム監視に適しています。

#### コード例3: Autoencoderによる非線形異常検知

**目的** : ニューラルネットワークのAutoencoderで再構成誤差に基づく異常検知を実装する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 化学反応器の正常運転データ生成
    n_normal_train = 1000
    n_normal_test = 200
    n_anomaly_test = 100
    
    # 正常運転データ（訓練用）
    def generate_normal_data(n_samples):
        """正常運転データを生成（非線形相関を含む）"""
        temperature = np.random.normal(400, 15, n_samples)  # K
        pressure = 5.0 + 0.02 * (temperature - 400) + 0.0001 * (temperature - 400)**2 + np.random.normal(0, 0.2, n_samples)  # bar
        flow_rate = 200 + 1.5 * np.log(temperature/300) + np.random.normal(0, 10, n_samples)  # L/h
        conversion = 0.85 * (1 - np.exp(-0.01 * (temperature - 350))) + np.random.normal(0, 0.03, n_samples)
        return np.column_stack([temperature, pressure, flow_rate, conversion])
    
    X_train = generate_normal_data(n_normal_train)
    X_test_normal = generate_normal_data(n_normal_test)
    
    # 異常運転データ（テスト用）
    def generate_anomaly_data(n_samples):
        """異常運転データを生成"""
        anomalies = []
    
        # パターン1: 温度暴走
        temp_high = np.random.normal(450, 20, n_samples//3)
        press_high = 5.0 + 0.02 * (temp_high - 400) + np.random.normal(0, 0.3, n_samples//3)
        flow_high = 200 + 1.5 * np.log(temp_high/300) + np.random.normal(0, 15, n_samples//3)
        conv_high = 0.95 + np.random.normal(0, 0.02, n_samples//3)
        anomalies.append(np.column_stack([temp_high, press_high, flow_high, conv_high]))
    
        # パターン2: 圧力異常
        temp_norm = np.random.normal(400, 15, n_samples//3)
        press_low = np.random.normal(3.0, 0.5, n_samples//3)  # 圧力低下
        flow_norm = 200 + 1.5 * np.log(temp_norm/300) + np.random.normal(0, 10, n_samples//3)
        conv_low = 0.60 + np.random.normal(0, 0.05, n_samples//3)  # 転化率低下
        anomalies.append(np.column_stack([temp_norm, press_low, flow_norm, conv_low]))
    
        # パターン3: 流量異常
        temp_norm2 = np.random.normal(400, 15, n_samples - 2*(n_samples//3))
        press_norm2 = 5.0 + 0.02 * (temp_norm2 - 400) + np.random.normal(0, 0.2, n_samples - 2*(n_samples//3))
        flow_low = np.random.normal(100, 20, n_samples - 2*(n_samples//3))  # 流量低下
        conv_norm2 = 0.85 * (1 - np.exp(-0.01 * (temp_norm2 - 350))) + np.random.normal(0, 0.03, n_samples - 2*(n_samples//3))
        anomalies.append(np.column_stack([temp_norm2, press_norm2, flow_low, conv_norm2]))
    
        return np.vstack(anomalies)
    
    X_test_anomaly = generate_anomaly_data(n_anomaly_test)
    
    # データ標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_normal_scaled = scaler.transform(X_test_normal)
    X_test_anomaly_scaled = scaler.transform(X_test_anomaly)
    
    # Autoencoderモデルの定義
    class Autoencoder(nn.Module):
        def __init__(self, input_dim=4, encoding_dim=2):
            super(Autoencoder, self).__init__()
            # エンコーダー
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, encoding_dim),
                nn.ReLU()
            )
            # デコーダー
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 8),
                nn.ReLU(),
                nn.Linear(8, input_dim)
            )
    
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # モデルのインスタンス化
    model = Autoencoder(input_dim=4, encoding_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練データをTensorに変換
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    
    # モデルの訓練
    print("=== Autoencoder訓練開始 ===")
    n_epochs = 100
    batch_size = 32
    losses = []
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            batch = X_train_tensor[i:i+batch_size]
    
            # 順伝播
            outputs = model(batch)
            loss = criterion(outputs, batch)
    
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        avg_loss = epoch_loss / (len(X_train_tensor) / batch_size)
        losses.append(avg_loss)
    
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
    # テストデータで再構成誤差を計算
    model.eval()
    with torch.no_grad():
        # 正常データ
        X_test_normal_tensor = torch.FloatTensor(X_test_normal_scaled)
        reconstructed_normal = model(X_test_normal_tensor)
        reconstruction_error_normal = torch.mean((X_test_normal_tensor - reconstructed_normal)**2, dim=1).numpy()
    
        # 異常データ
        X_test_anomaly_tensor = torch.FloatTensor(X_test_anomaly_scaled)
        reconstructed_anomaly = model(X_test_anomaly_tensor)
        reconstruction_error_anomaly = torch.mean((X_test_anomaly_tensor - reconstructed_anomaly)**2, dim=1).numpy()
    
    # 閾値の設定（訓練データの99パーセンタイル）
    with torch.no_grad():
        reconstructed_train = model(X_train_tensor)
        reconstruction_error_train = torch.mean((X_train_tensor - reconstructed_train)**2, dim=1).numpy()
    threshold = np.percentile(reconstruction_error_train, 99)
    
    print(f"\n再構成誤差閾値（99%): {threshold:.6f}")
    print(f"正常データの平均再構成誤差: {reconstruction_error_normal.mean():.6f}")
    print(f"異常データの平均再構成誤差: {reconstruction_error_anomaly.mean():.6f}")
    
    # ROC-AUC評価
    y_true = np.array([0]*len(reconstruction_error_normal) + [1]*len(reconstruction_error_anomaly))
    y_scores = np.concatenate([reconstruction_error_normal, reconstruction_error_anomaly])
    auc_score = roc_auc_score(y_true, y_scores)
    print(f"\nROC-AUC スコア: {auc_score:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 訓練損失
    axes[0, 0].plot(losses, color='#11998e', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=11)
    axes[0, 0].set_title('訓練損失の推移', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 再構成誤差のヒストグラム
    axes[0, 1].hist(reconstruction_error_normal, bins=30, alpha=0.6, color='blue',
                    label='正常', edgecolor='black')
    axes[0, 1].hist(reconstruction_error_anomaly, bins=30, alpha=0.8, color='red',
                    label='異常', edgecolor='black')
    axes[0, 1].axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label='閾値 (99%)')
    axes[0, 1].set_xlabel('再構成誤差', fontsize=11)
    axes[0, 1].set_ylabel('頻度', fontsize=11)
    axes[0, 1].set_title('再構成誤差分布', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 再構成誤差の時系列（テストデータ）
    all_errors = np.concatenate([reconstruction_error_normal, reconstruction_error_anomaly])
    colors = ['blue']*len(reconstruction_error_normal) + ['red']*len(reconstruction_error_anomaly)
    axes[1, 0].scatter(range(len(all_errors)), all_errors, c=colors, s=30, alpha=0.6)
    axes[1, 0].axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label='閾値')
    axes[1, 0].set_xlabel('サンプル番号', fontsize=11)
    axes[1, 0].set_ylabel('再構成誤差', fontsize=11)
    axes[1, 0].set_title('テストデータの再構成誤差', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # ROC曲線
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    axes[1, 1].plot(fpr, tpr, color='#11998e', linewidth=2, label=f'AUC = {auc_score:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 1].set_title('ROC曲線', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : Autoencoderは、正常データの特徴を低次元表現（潜在変数）に圧縮し、再構成する深層学習モデルです。異常データは正常データとは異なる特徴を持つため、再構成誤差が大きくなります。非線形な変数間関係を学習でき、PCAでは捉えられない複雑な異常パターンを検出できます。

#### コード例4: LSTMによる時系列異常検知

**目的** : LSTM（Long Short-Term Memory）で時系列パターンを学習し、予測誤差に基づく異常検知を実装する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 時系列プロセスデータ生成（バッチ反応器の温度プロファイル）
    def generate_batch_reactor_data(n_batches, batch_length=100, anomaly=False):
        """バッチ反応器の温度プロファイルを生成"""
        data = []
        for _ in range(n_batches):
            t = np.linspace(0, 10, batch_length)  # 時間（時間）
    
            if not anomaly:
                # 正常バッチ: 典型的な発熱反応プロファイル
                temp = 320 + 50 * (1 - np.exp(-0.5 * t)) * np.exp(-0.1 * t) + np.random.normal(0, 2, batch_length)
            else:
                # 異常バッチ: 異常な温度上昇パターン
                if np.random.rand() < 0.5:
                    # パターン1: 過度な発熱
                    temp = 320 + 80 * (1 - np.exp(-0.7 * t)) * np.exp(-0.05 * t) + np.random.normal(0, 3, batch_length)
                else:
                    # パターン2: 不十分な反応
                    temp = 320 + 20 * (1 - np.exp(-0.3 * t)) * np.exp(-0.15 * t) + np.random.normal(0, 2, batch_length)
    
            data.append(temp)
        return np.array(data)
    
    # 訓練データ（正常バッチのみ）
    n_train_batches = 200
    n_test_normal = 50
    n_test_anomaly = 30
    batch_length = 100
    
    X_train = generate_batch_reactor_data(n_train_batches, batch_length, anomaly=False)
    X_test_normal = generate_batch_reactor_data(n_test_normal, batch_length, anomaly=False)
    X_test_anomaly = generate_batch_reactor_data(n_test_anomaly, batch_length, anomaly=True)
    
    print(f"=== データセット ===")
    print(f"訓練バッチ数: {n_train_batches}")
    print(f"テスト正常バッチ数: {n_test_normal}")
    print(f"テスト異常バッチ数: {n_test_anomaly}")
    print(f"バッチ長: {batch_length}")
    
    # データ標準化
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 1)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_normal_scaled = scaler.transform(X_test_normal.reshape(-1, 1)).reshape(X_test_normal.shape)
    X_test_anomaly_scaled = scaler.transform(X_test_anomaly.reshape(-1, 1)).reshape(X_test_anomaly.shape)
    
    # 時系列データをLSTM用に整形（バッチ, シーケンス長, 特徴数）
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(-1)  # (200, 100, 1)
    X_test_normal_tensor = torch.FloatTensor(X_test_normal_scaled).unsqueeze(-1)
    X_test_anomaly_tensor = torch.FloatTensor(X_test_anomaly_scaled).unsqueeze(-1)
    
    # LSTMモデルの定義
    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, num_layers=2):
            super(LSTMPredictor, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
    
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim)
    
        def forward(self, x):
            # LSTM層
            lstm_out, _ = self.lstm(x)
            # 全結合層で各時刻の予測値を出力
            predictions = self.fc(lstm_out)
            return predictions
    
    # モデルのインスタンス化
    model = LSTMPredictor(input_dim=1, hidden_dim=32, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # モデル訓練（1ステップ先予測）
    print("\n=== LSTM訓練開始 ===")
    n_epochs = 50
    batch_size = 16
    losses = []
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            batch = X_train_tensor[i:i+batch_size]
    
            # 入力: t=0~98, ターゲット: t=1~99（1ステップ先予測）
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :]
    
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        avg_loss = epoch_loss / (len(X_train_tensor) / batch_size)
        losses.append(avg_loss)
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
    # テストデータで予測誤差を計算
    model.eval()
    with torch.no_grad():
        # 正常バッチ
        inputs_normal = X_test_normal_tensor[:, :-1, :]
        targets_normal = X_test_normal_tensor[:, 1:, :]
        predictions_normal = model(inputs_normal)
        prediction_error_normal = torch.mean((predictions_normal - targets_normal)**2, dim=(1, 2)).numpy()
    
        # 異常バッチ
        inputs_anomaly = X_test_anomaly_tensor[:, :-1, :]
        targets_anomaly = X_test_anomaly_tensor[:, 1:, :]
        predictions_anomaly = model(inputs_anomaly)
        prediction_error_anomaly = torch.mean((predictions_anomaly - targets_anomaly)**2, dim=(1, 2)).numpy()
    
    # 閾値設定（訓練データの95パーセンタイル）
    with torch.no_grad():
        inputs_train = X_train_tensor[:, :-1, :]
        targets_train = X_train_tensor[:, 1:, :]
        predictions_train = model(inputs_train)
        prediction_error_train = torch.mean((predictions_train - targets_train)**2, dim=(1, 2)).numpy()
    threshold = np.percentile(prediction_error_train, 95)
    
    print(f"\n予測誤差閾値（95%): {threshold:.6f}")
    print(f"正常バッチの平均予測誤差: {prediction_error_normal.mean():.6f}")
    print(f"異常バッチの平均予測誤差: {prediction_error_anomaly.mean():.6f}")
    
    # 異常検出性能
    y_true = np.array([0]*len(prediction_error_normal) + [1]*len(prediction_error_anomaly))
    y_pred = np.array([0 if e < threshold else 1 for e in np.concatenate([prediction_error_normal, prediction_error_anomaly])])
    accuracy = np.mean(y_true == y_pred)
    print(f"\n検出精度: {accuracy:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 訓練損失
    axes[0, 0].plot(losses, color='#11998e', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=11)
    axes[0, 0].set_title('LSTM訓練損失', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 予測誤差のヒストグラム
    axes[0, 1].hist(prediction_error_normal, bins=20, alpha=0.6, color='blue',
                    label='正常', edgecolor='black')
    axes[0, 1].hist(prediction_error_anomaly, bins=20, alpha=0.8, color='red',
                    label='異常', edgecolor='black')
    axes[0, 1].axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label='閾値 (95%)')
    axes[0, 1].set_xlabel('予測誤差（MSE）', fontsize=11)
    axes[0, 1].set_ylabel('頻度', fontsize=11)
    axes[0, 1].set_title('予測誤差分布', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 正常バッチの予測例
    sample_idx = 0
    axes[1, 0].plot(X_test_normal[sample_idx], 'b-', linewidth=2, label='実測値')
    pred_sample = predictions_normal[sample_idx].squeeze().numpy()
    pred_sample_rescaled = scaler.inverse_transform(pred_sample.reshape(-1, 1)).flatten()
    actual_rescaled = scaler.inverse_transform(X_test_normal[sample_idx, 1:].reshape(-1, 1)).flatten()
    axes[1, 0].plot(range(1, 100), pred_sample_rescaled, 'g--', linewidth=2, label='LSTM予測')
    axes[1, 0].set_xlabel('時間ステップ', fontsize=11)
    axes[1, 0].set_ylabel('温度 (K)', fontsize=11)
    axes[1, 0].set_title('正常バッチの予測例', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 異常バッチの予測例
    sample_idx_anomaly = 0
    axes[1, 1].plot(X_test_anomaly[sample_idx_anomaly], 'b-', linewidth=2, label='実測値（異常）')
    pred_sample_anom = predictions_anomaly[sample_idx_anomaly].squeeze().numpy()
    pred_sample_anom_rescaled = scaler.inverse_transform(pred_sample_anom.reshape(-1, 1)).flatten()
    axes[1, 1].plot(range(1, 100), pred_sample_anom_rescaled, 'g--', linewidth=2, label='LSTM予測')
    axes[1, 1].set_xlabel('時間ステップ', fontsize=11)
    axes[1, 1].set_ylabel('温度 (K)', fontsize=11)
    axes[1, 1].set_title('異常バッチの予測例（予測誤差大）', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : LSTMは時系列データの長期依存関係を学習できる再帰型ニューラルネットワークです。バッチプロセスの温度プロファイルなど、時間的パターンが重要な監視対象に有効です。正常な時系列パターンを学習し、異常パターンでは予測誤差が増大するため、異常検知が可能になります。

* * *

## 1.3 品質予測とソフトセンサー

### ソフトセンサーとは

**ソフトセンサー（Soft Sensor）** は、測定困難または測定コストが高い変数（製品品質、反応率、不純物濃度など）を、測定容易なプロセス変数（温度、圧力、流量など）から推定する技術です。

**利点** :

  * リアルタイム品質監視（分析計は数分～数時間の遅れ）
  * コスト削減（高価な分析計の代替）
  * プロセス制御の高度化（品質フィードバック制御）
  * 保全性向上（分析計の故障時のバックアップ）

#### コード例5: Random Forestによる製品品質予測

**目的** : Random Forestで蒸留塔の製品純度を予測する品質予測モデルを構築する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    np.random.seed(42)
    
    # 蒸留塔の運転データと製品純度の関係を生成
    n_samples = 1500
    
    # プロセス変数（入力）
    塔頂温度 = np.random.normal(85, 3, n_samples)  # °C
    塔底温度 = np.random.normal(155, 5, n_samples)  # °C
    還流比 = np.random.normal(3.5, 0.5, n_samples)
    原料流量 = np.random.normal(50, 8, n_samples)  # m³/h
    塔内圧力 = np.random.normal(1.2, 0.15, n_samples)  # bar
    
    # 製品純度（目的変数）- 非線形な関係
    製品純度 = (
        0.95
        - 0.002 * (塔頂温度 - 85)  # 塔頂温度が高いと純度低下
        + 0.0005 * (塔底温度 - 155)  # 塔底温度が高いと純度向上
        + 0.02 * (還流比 - 3.5)  # 還流比が高いと純度向上
        - 0.0003 * (原料流量 - 50)  # 流量が多いと純度低下
        + 0.01 * (塔内圧力 - 1.2)  # 圧力が高いと純度向上
        - 0.0001 * (塔頂温度 - 85)**2  # 非線形効果
        + 0.001 * (還流比 - 3.5) * (塔底温度 - 155) / 10  # 交互作用
        + np.random.normal(0, 0.005, n_samples)  # 測定ノイズ
    )
    
    # DataFrameに格納
    df = pd.DataFrame({
        '塔頂温度': 塔頂温度,
        '塔底温度': 塔底温度,
        '還流比': 還流比,
        '原料流量': 原料流量,
        '塔内圧力': 塔内圧力,
        '製品純度': 製品純度
    })
    
    # データ分割
    X = df.drop('製品純度', axis=1)
    y = df['製品純度']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"=== データセット ===")
    print(f"訓練データ数: {len(X_train)}")
    print(f"テストデータ数: {len(X_test)}")
    print(f"特徴変数数: {X.shape[1]}")
    
    # Random Forestモデルの訓練
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # 予測
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # 性能評価
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n=== モデル性能 ===")
    print(f"訓練データ:")
    print(f"  MAE: {mae_train:.5f}")
    print(f"  RMSE: {rmse_train:.5f}")
    print(f"  R²: {r2_train:.4f}")
    print(f"\nテストデータ:")
    print(f"  MAE: {mae_test:.5f}")
    print(f"  RMSE: {rmse_test:.5f}")
    print(f"  R²: {r2_test:.4f}")
    
    # 特徴重要度
    feature_importance = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': rf_model.feature_importances_
    }).sort_values('重要度', ascending=False)
    
    print(f"\n=== 特徴重要度 ===")
    print(feature_importance)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 予測 vs 実測（テストデータ）
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=30, color='#11998e')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='理想直線')
    axes[0, 0].set_xlabel('実測純度', fontsize=11)
    axes[0, 0].set_ylabel('予測純度', fontsize=11)
    axes[0, 0].set_title(f'予測 vs 実測（R²={r2_test:.4f}）', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 残差プロット
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=30, color='#38ef7d')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].fill_between([y_pred_test.min(), y_pred_test.max()], -2*rmse_test, 2*rmse_test,
                             alpha=0.2, color='orange', label='±2σ範囲')
    axes[0, 1].set_xlabel('予測純度', fontsize=11)
    axes[0, 1].set_ylabel('残差', fontsize=11)
    axes[0, 1].set_title('残差プロット', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 特徴重要度
    axes[1, 0].barh(feature_importance['特徴量'], feature_importance['重要度'], color='#11998e')
    axes[1, 0].set_xlabel('重要度', fontsize=11)
    axes[1, 0].set_title('特徴重要度（Random Forest）', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 時系列予測プロット（最初の100サンプル）
    test_indices = range(100)
    axes[1, 1].plot(test_indices, y_test.iloc[:100].values, 'b-', linewidth=2,
                    label='実測値', marker='o', markersize=3)
    axes[1, 1].plot(test_indices, y_pred_test[:100], 'r--', linewidth=2,
                    label='予測値', marker='s', markersize=3)
    axes[1, 1].fill_between(test_indices, y_pred_test[:100] - 2*rmse_test,
                             y_pred_test[:100] + 2*rmse_test, alpha=0.2, color='red', label='±2σ範囲')
    axes[1, 1].set_xlabel('サンプル番号', fontsize=11)
    axes[1, 1].set_ylabel('製品純度', fontsize=11)
    axes[1, 1].set_title('品質予測の時系列プロット', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : Random Forestは、非線形関係や変数間の交互作用を自動的に学習でき、外れ値に頑健な特性を持ちます。特徴重要度により、品質に影響する主要なプロセス変数を特定できます。化学プラントでは、分析計の測定遅れ（数分～数時間）を補完し、リアルタイム品質監視を実現します。

#### コード例6: ガウス過程回帰（GPR）によるソフトセンサー設計

**目的** : Gaussian Process Regressionで不確実性を含む品質予測ソフトセンサーを構築する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    
    np.random.seed(42)
    
    # 化学反応器の転化率予測ソフトセンサー
    n_samples = 500
    
    # プロセス変数
    温度 = np.random.normal(380, 20, n_samples)  # K
    圧力 = np.random.normal(5.0, 0.5, n_samples)  # bar
    触媒濃度 = np.random.normal(0.1, 0.02, n_samples)  # mol/L
    
    # 転化率（アレニウス型の非線形関係）
    活性化エネルギー = 80000  # J/mol
    R = 8.314  # J/(mol·K)
    反応速度定数 = np.exp(-活性化エネルギー / (R * 温度))
    転化率 = (
        1 - np.exp(-反応速度定数 * 圧力 * 触媒濃度 * 100)
        + np.random.normal(0, 0.02, n_samples)
    )
    転化率 = np.clip(転化率, 0, 1)  # 0-1の範囲にクリップ
    
    # DataFrameに格納
    df = pd.DataFrame({
        '温度': 温度,
        '圧力': 圧力,
        '触媒濃度': 触媒濃度,
        '転化率': 転化率
    })
    
    # データ分割
    X = df[['温度', '圧力', '触媒濃度']].values
    y = df['転化率'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # データ標準化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"=== ソフトセンサー構築 ===")
    print(f"訓練データ数: {len(X_train)}")
    print(f"テストデータ数: {len(X_test)}")
    
    # ガウス過程回帰カーネルの定義
    # RBFカーネル + ホワイトノイズ（測定ノイズを考慮）
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0, 1.0],
                                        length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.01)
    
    # GPRモデルの訓練
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-10,
        random_state=42
    )
    
    gpr.fit(X_train_scaled, y_train)
    
    print(f"\n最適化されたカーネル:")
    print(gpr.kernel_)
    
    # 予測（平均と標準偏差）
    y_pred_test, y_std_test = gpr.predict(X_test_scaled, return_std=True)
    y_pred_train, y_std_train = gpr.predict(X_train_scaled, return_std=True)
    
    # 性能評価
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n=== ソフトセンサー性能 ===")
    print(f"テストデータ MAE: {mae_test:.5f}")
    print(f"テストデータ R²: {r2_test:.4f}")
    print(f"平均予測不確実性（σ）: {y_std_test.mean():.5f}")
    
    # 予測区間内のカバー率（95%信頼区間）
    lower_bound = y_pred_test - 1.96 * y_std_test
    upper_bound = y_pred_test + 1.96 * y_std_test
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    print(f"95%予測区間カバー率: {coverage:.2%}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 予測 vs 実測（不確実性付き）
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=30, c=y_std_test, cmap='viridis')
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('予測標準偏差', fontsize=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='理想直線')
    axes[0, 0].set_xlabel('実測転化率', fontsize=11)
    axes[0, 0].set_ylabel('予測転化率', fontsize=11)
    axes[0, 0].set_title(f'GPRソフトセンサー（R²={r2_test:.4f}）', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 予測不確実性のヒストグラム
    axes[0, 1].hist(y_std_test, bins=30, color='#11998e', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('予測標準偏差（σ）', fontsize=11)
    axes[0, 1].set_ylabel('頻度', fontsize=11)
    axes[0, 1].set_title('予測不確実性の分布', fontsize=13, fontweight='bold')
    axes[0, 1].axvline(x=y_std_test.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'平均σ: {y_std_test.mean():.5f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 時系列予測プロット（95%信頼区間付き）
    sorted_indices = np.argsort(y_test)[:60]
    axes[1, 0].plot(range(60), y_test[sorted_indices], 'bo-', linewidth=2,
                    markersize=4, label='実測値')
    axes[1, 0].plot(range(60), y_pred_test[sorted_indices], 'r--', linewidth=2,
                    marker='s', markersize=4, label='GPR予測')
    axes[1, 0].fill_between(range(60),
                             y_pred_test[sorted_indices] - 1.96*y_std_test[sorted_indices],
                             y_pred_test[sorted_indices] + 1.96*y_std_test[sorted_indices],
                             alpha=0.3, color='red', label='95%信頼区間')
    axes[1, 0].set_xlabel('サンプル番号', fontsize=11)
    axes[1, 0].set_ylabel('転化率', fontsize=11)
    axes[1, 0].set_title('ソフトセンサー予測（不確実性付き）', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 誤差 vs 不確実性の関係
    abs_error = np.abs(y_test - y_pred_test)
    axes[1, 1].scatter(y_std_test, abs_error, alpha=0.5, s=30, color='#38ef7d')
    axes[1, 1].plot([y_std_test.min(), y_std_test.max()],
                    [y_std_test.min(), y_std_test.max()],
                    'r--', linewidth=2, label='完全一致')
    axes[1, 1].set_xlabel('予測標準偏差（σ）', fontsize=11)
    axes[1, 1].set_ylabel('絶対誤差', fontsize=11)
    axes[1, 1].set_title('予測誤差 vs 不確実性', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : ガウス過程回帰（GPR）は、予測値だけでなく予測の不確実性（標準偏差）も出力します。この不確実性情報は、プロセス制御や意思決定に極めて重要です。不確実性が大きい予測値は信頼性が低いと判断し、追加測定や保守的な制御を実施できます。データが少ない領域では自動的に不確実性が大きくなります。

#### コード例7: ニューラルネットワークソフトセンサー（オンライン適応）

**目的** : ニューラルネットワークでソフトセンサーを構築し、オンライン適応（Adaptive Learning）を実装する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # プロセス特性のドリフト（経時変化）を考慮したデータ生成
    def generate_process_data_with_drift(n_samples, drift_factor=0):
        """
        触媒活性の経時劣化を考慮したプロセスデータ生成
        drift_factor: 0（初期）～1（大幅劣化）
        """
        温度 = np.random.normal(370, 15, n_samples)
        圧力 = np.random.normal(4.5, 0.4, n_samples)
        流量 = np.random.normal(180, 20, n_samples)
    
        # 触媒活性の劣化（ドリフト）
        活性係数 = 1.0 - 0.3 * drift_factor  # 最大30%の活性低下
    
        # 製品収率（活性劣化により低下）
        収率 = (
            活性係数 * (0.75 + 0.001 * (温度 - 370) + 0.02 * (圧力 - 4.5) - 0.0002 * (流量 - 180))
            + np.random.normal(0, 0.02, n_samples)
        )
        収率 = np.clip(収率, 0, 1)
    
        return np.column_stack([温度, 圧力, 流量]), 収率
    
    # 初期訓練データ（新品触媒）
    X_train, y_train = generate_process_data_with_drift(1000, drift_factor=0)
    
    # 標準化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # ニューラルネットワークソフトセンサーの定義
    class SoftSensor(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=16):
            super(SoftSensor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
        def forward(self, x):
            return self.network(x)
    
    # モデルのインスタンス化
    model = SoftSensor(input_dim=3, hidden_dim=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 初期訓練
    print("=== ソフトセンサー初期訓練 ===")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).unsqueeze(1)
    
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}")
    
    # オンライン運転シミュレーション（触媒劣化を含む）
    n_online_samples = 500
    drift_schedule = np.linspace(0, 0.8, n_online_samples)  # 徐々に劣化
    
    # 性能追跡用リスト
    mae_history = []
    predictions_history = []
    actuals_history = []
    adaptation_points = []  # 適応更新を行ったポイント
    
    print("\n=== オンライン運転とソフトセンサー適応 ===")
    adaptation_threshold = 0.05  # MAEがこの値を超えたら適応学習
    adaptation_window = 50  # 適応学習用のデータウィンドウ
    
    for t in range(n_online_samples):
        # 現在の触媒劣化状態でデータ生成（1サンプル）
        X_new, y_new = generate_process_data_with_drift(1, drift_factor=drift_schedule[t])
        X_new_scaled = scaler_X.transform(X_new)
    
        # 予測
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_new_scaled)
            y_pred_scaled = model(X_tensor).numpy()[0, 0]
            y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
    
        predictions_history.append(y_pred)
        actuals_history.append(y_new[0])
    
        # 移動窓での性能評価
        if t >= 20:
            recent_mae = mean_absolute_error(
                actuals_history[t-20:t+1],
                predictions_history[t-20:t+1]
            )
            mae_history.append(recent_mae)
    
            # 性能劣化検出とオンライン適応
            if recent_mae > adaptation_threshold and t >= adaptation_window:
                # 最近のデータで適応学習
                X_adapt = np.array([scaler_X.transform([[actuals_history[i],
                                                          4.5 + np.random.normal(0, 0.1),
                                                          180 + np.random.normal(0, 5)]])
                                    for i in range(t-adaptation_window, t)]).squeeze()
                y_adapt = np.array([actuals_history[i] for i in range(t-adaptation_window, t)])
    
                # オンライン適応（少数エポック）
                X_adapt_tensor = torch.FloatTensor(X_adapt)
                y_adapt_scaled = scaler_y.transform(y_adapt.reshape(-1, 1)).flatten()
                y_adapt_tensor = torch.FloatTensor(y_adapt_scaled).unsqueeze(1)
    
                model.train()
                for _ in range(10):  # 少数エポックで微調整
                    optimizer.zero_grad()
                    outputs = model(X_adapt_tensor)
                    loss = criterion(outputs, y_adapt_tensor)
                    loss.backward()
                    optimizer.step()
    
                adaptation_points.append(t)
                print(f"時刻 {t}: オンライン適応実行（MAE={recent_mae:.4f}）")
    
        if (t + 1) % 100 == 0:
            print(f"時刻 {t+1}: 処理完了")
    
    print(f"\n適応学習実行回数: {len(adaptation_points)}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 予測 vs 実測の時系列
    axes[0, 0].plot(actuals_history, 'b-', linewidth=1.5, alpha=0.7, label='実測値')
    axes[0, 0].plot(predictions_history, 'r--', linewidth=1.5, alpha=0.7, label='予測値')
    for ap in adaptation_points:
        axes[0, 0].axvline(x=ap, color='green', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('時刻', fontsize=11)
    axes[0, 0].set_ylabel('製品収率', fontsize=11)
    axes[0, 0].set_title('ソフトセンサー予測（緑線：適応学習実行）', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MAEの時系列（性能劣化の追跡）
    axes[0, 1].plot(range(20, len(actuals_history)), mae_history, color='#11998e', linewidth=2)
    axes[0, 1].axhline(y=adaptation_threshold, color='red', linestyle='--',
                       linewidth=2, label='適応閾値')
    for ap in adaptation_points:
        axes[0, 1].axvline(x=ap, color='green', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('時刻', fontsize=11)
    axes[0, 1].set_ylabel('移動平均MAE（20サンプル）', fontsize=11)
    axes[0, 1].set_title('ソフトセンサー性能モニタリング', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 触媒劣化スケジュール
    axes[1, 0].plot(drift_schedule * 100, color='#38ef7d', linewidth=2)
    axes[1, 0].set_xlabel('時刻', fontsize=11)
    axes[1, 0].set_ylabel('触媒活性低下率 (%)', fontsize=11)
    axes[1, 0].set_title('触媒経時劣化（シミュレーション）', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 予測誤差のヒストグラム（適応前後）
    errors_before = np.array(actuals_history[:adaptation_points[0]]) - np.array(predictions_history[:adaptation_points[0]]) if len(adaptation_points) > 0 else []
    errors_after = np.array(actuals_history[adaptation_points[-1]:]) - np.array(predictions_history[adaptation_points[-1]:]) if len(adaptation_points) > 0 else []
    
    if len(errors_before) > 0 and len(errors_after) > 0:
        axes[1, 1].hist(errors_before, bins=20, alpha=0.6, color='blue', label='適応前', edgecolor='black')
        axes[1, 1].hist(errors_after, bins=20, alpha=0.6, color='green', label='適応後', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('予測誤差', fontsize=11)
        axes[1, 1].set_ylabel('頻度', fontsize=11)
        axes[1, 1].set_title('予測誤差分布（適応効果）', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : 化学プラントでは触媒劣化、原料組成変動、季節変動などによりプロセス特性がドリフトします。固定モデルのソフトセンサーは時間とともに性能が劣化します。オンライン適応学習（Adaptive Learning）により、最新のプロセスデータで定期的にモデルを更新し、長期間の高精度運転を維持できます。

#### コード例8: 統合プロセス監視システム（異常検知+ソフトセンサー）

**目的** : 異常検知とソフトセンサーを統合した総合プロセス監視システムを実装する。
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    
    np.random.seed(42)
    
    # 統合プロセス監視システムクラス
    class IntegratedProcessMonitoring:
        """
        異常検知とソフトセンサーを統合したプロセス監視システム
        """
        def __init__(self):
            self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
            self.soft_sensor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
    
        def train(self, X_process, y_quality):
            """
            システムの訓練
            X_process: プロセス変数（温度、圧力、流量など）
            y_quality: 品質変数（ソフトセンサーのターゲット）
            """
            # データ標準化
            X_scaled = self.scaler.fit_transform(X_process)
    
            # 異常検知モデルの訓練
            self.anomaly_detector.fit(X_scaled)
    
            # ソフトセンサーの訓練
            self.soft_sensor.fit(X_scaled, y_quality)
    
            self.is_trained = True
            print("統合監視システムの訓練完了")
    
        def monitor(self, X_new):
            """
            新しいデータの監視
            Returns: (異常判定, 品質予測, 異常スコア)
            """
            if not self.is_trained:
                raise ValueError("システムが訓練されていません")
    
            X_scaled = self.scaler.transform(X_new.reshape(1, -1))
    
            # 異常検知
            anomaly_pred = self.anomaly_detector.predict(X_scaled)[0]
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
    
            # 品質予測
            quality_pred = self.soft_sensor.predict(X_scaled)[0]
    
            is_anomaly = (anomaly_pred == -1)
    
            return is_anomaly, quality_pred, anomaly_score
    
        def get_alert_message(self, is_anomaly, quality_pred, quality_threshold=0.95):
            """アラートメッセージの生成"""
            alerts = []
    
            if is_anomaly:
                alerts.append("⚠️ プロセス異常検出")
    
            if quality_pred < quality_threshold:
                alerts.append(f"⚠️ 品質低下予測（予測純度: {quality_pred:.3f} < 閾値: {quality_threshold}）")
    
            if len(alerts) == 0:
                return "✓ 正常運転"
            else:
                return " | ".join(alerts)
    
    # 化学プラントの連続運転シミュレーション
    n_normal_train = 800
    n_online = 200
    
    # 訓練データ（正常運転）
    塔頂温度_訓練 = np.random.normal(85, 2, n_normal_train)
    塔底温度_訓練 = np.random.normal(155, 3, n_normal_train)
    還流比_訓練 = np.random.normal(3.5, 0.3, n_normal_train)
    製品純度_訓練 = 0.98 - 0.002 * (塔頂温度_訓練 - 85) + 0.015 * (還流比_訓練 - 3.5) + np.random.normal(0, 0.005, n_normal_train)
    
    X_train = np.column_stack([塔頂温度_訓練, 塔底温度_訓練, 還流比_訓練])
    y_train = 製品純度_訓練
    
    # 統合監視システムの訓練
    monitor_system = IntegratedProcessMonitoring()
    monitor_system.train(X_train, y_train)
    
    # オンライン運転シミュレーション
    print("\n=== オンライン運転監視開始 ===\n")
    
    # 正常運転データ（150サンプル）
    塔頂温度_正常 = np.random.normal(85, 2, 150)
    塔底温度_正常 = np.random.normal(155, 3, 150)
    還流比_正常 = np.random.normal(3.5, 0.3, 150)
    製品純度_正常 = 0.98 - 0.002 * (塔頂温度_正常 - 85) + 0.015 * (還流比_正常 - 3.5) + np.random.normal(0, 0.005, 150)
    
    # 異常運転データ（50サンプル）
    # ケース1: 冷却器故障（塔頂温度上昇）
    塔頂温度_異常1 = np.random.normal(95, 3, 20)
    塔底温度_異常1 = np.random.normal(155, 3, 20)
    還流比_異常1 = np.random.normal(3.5, 0.3, 20)
    製品純度_異常1 = 0.93 - 0.002 * (塔頂温度_異常1 - 85) + 0.015 * (還流比_異常1 - 3.5) + np.random.normal(0, 0.01, 20)
    
    # ケース2: 還流ポンプ故障（還流比低下）
    塔頂温度_異常2 = np.random.normal(85, 2, 15)
    塔底温度_異常2 = np.random.normal(155, 3, 15)
    還流比_異常2 = np.random.normal(2.0, 0.4, 15)
    製品純度_異常2 = 0.88 + 0.015 * (還流比_異常2 - 3.5) + np.random.normal(0, 0.01, 15)
    
    # ケース3: 原料組成異常
    塔頂温度_異常3 = np.random.normal(90, 4, 15)
    塔底温度_異常3 = np.random.normal(165, 6, 15)
    還流比_異常3 = np.random.normal(4.0, 0.5, 15)
    製品純度_異常3 = 0.90 + np.random.normal(0, 0.015, 15)
    
    # データ統合
    X_online = np.vstack([
        np.column_stack([塔頂温度_正常, 塔底温度_正常, 還流比_正常]),
        np.column_stack([塔頂温度_異常1, 塔底温度_異常1, 還流比_異常1]),
        np.column_stack([塔頂温度_異常2, 塔底温度_異常2, 還流比_異常2]),
        np.column_stack([塔頂温度_異常3, 塔底温度_異常3, 還流比_異常3])
    ])
    
    y_online = np.concatenate([製品純度_正常, 製品純度_異常1, 製品純度_異常2, 製品純度_異常3])
    labels = np.array([0]*150 + [1]*20 + [2]*15 + [3]*15)  # 0: 正常, 1-3: 異常パターン
    
    # オンライン監視実行
    anomaly_flags = []
    quality_predictions = []
    anomaly_scores = []
    alert_messages = []
    
    for i in range(len(X_online)):
        is_anomaly, quality_pred, anomaly_score = monitor_system.monitor(X_online[i])
        alert_msg = monitor_system.get_alert_message(is_anomaly, quality_pred, quality_threshold=0.95)
    
        anomaly_flags.append(is_anomaly)
        quality_predictions.append(quality_pred)
        anomaly_scores.append(anomaly_score)
        alert_messages.append(alert_msg)
    
        # 異常検出時のログ出力
        if is_anomaly or quality_pred < 0.95:
            print(f"時刻 {i}: {alert_msg}")
            print(f"  プロセス変数: 塔頂温度={X_online[i, 0]:.1f}°C, 塔底温度={X_online[i, 1]:.1f}°C, 還流比={X_online[i, 2]:.2f}")
            print(f"  品質予測: {quality_pred:.4f}, 異常スコア: {anomaly_score:.4f}\n")
    
    # 性能評価
    true_anomalies = (labels > 0)
    detected_anomalies = np.array(anomaly_flags)
    detection_rate = np.sum(true_anomalies & detected_anomalies) / np.sum(true_anomalies)
    false_alarm_rate = np.sum((~true_anomalies) & detected_anomalies) / np.sum(~true_anomalies)
    
    print(f"\n=== 統合監視システム性能 ===")
    print(f"異常検出率: {detection_rate:.2%}")
    print(f"誤報率: {false_alarm_rate:.2%}")
    
    quality_mae = mean_absolute_error(y_online, quality_predictions)
    quality_r2 = r2_score(y_online, quality_predictions)
    print(f"\nソフトセンサー性能:")
    print(f"  MAE: {quality_mae:.5f}")
    print(f"  R²: {quality_r2:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # プロセス変数の時系列（異常検出結果）
    colors = ['blue' if not flag else 'red' for flag in anomaly_flags]
    axes[0].scatter(range(len(X_online)), X_online[:, 0], c=colors, s=30, alpha=0.6)
    axes[0].set_ylabel('塔頂温度 (°C)', fontsize=11)
    axes[0].set_title('プロセス変数監視（赤: 異常検出）', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 品質予測の時系列
    axes[1].plot(y_online, 'b-', linewidth=1.5, alpha=0.6, label='実測値')
    axes[1].plot(quality_predictions, 'g--', linewidth=1.5, alpha=0.8, label='ソフトセンサー予測')
    axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='品質下限')
    axes[1].scatter(np.where(anomaly_flags)[0], np.array(quality_predictions)[anomaly_flags],
                    color='red', s=80, marker='x', linewidths=3, label='異常検出時', zorder=5)
    axes[1].set_ylabel('製品純度', fontsize=11)
    axes[1].set_title('ソフトセンサーによる品質監視', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 異常スコアの時系列
    axes[2].plot(anomaly_scores, 'o-', markersize=3, linewidth=0.8, color='#11998e')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2, label='異常判定境界')
    axes[2].fill_between(range(len(anomaly_scores)),
                          [min(anomaly_scores)]*len(anomaly_scores), 0,
                          alpha=0.1, color='red', label='異常領域')
    axes[2].set_xlabel('時刻', fontsize=11)
    axes[2].set_ylabel('異常スコア', fontsize=11)
    axes[2].set_title('Isolation Forest 異常スコア', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解説** : 統合プロセス監視システムは、異常検知とソフトセンサーを組み合わせることで、プロセスの異常状態と品質低下の両方をリアルタイムに監視します。異常検知は多変量のプロセスパターン異常を捉え、ソフトセンサーは品質への影響を定量的に予測します。両者を統合することで、包括的なプロセス監視と早期警報が実現できます。

* * *

## 1.4 本章のまとめ

### 学んだこと

  1. **統計的異常検知**
     * PCA法によるQ統計量・T²統計量での多変量監視
     * 主成分空間と残差空間での異常診断
  2. **機械学習ベース異常検知**
     * Isolation Forest: 教師なし学習による柔軟な異常検出
     * Autoencoder: 深層学習による非線形パターン学習
     * LSTM: 時系列パターンの予測誤差に基づく異常検知
  3. **品質予測とソフトセンサー**
     * Random Forest: 非線形品質予測と特徴重要度分析
     * ガウス過程回帰: 不確実性を含む予測
     * ニューラルネットワーク: オンライン適応学習
  4. **統合監視システム**
     * 異常検知とソフトセンサーの組み合わせ
     * 包括的なプロセス監視とアラーム生成

### 実務適用のポイント

手法 | 適用場面 | メリット | 注意点  
---|---|---|---  
**PCA異常検知** | 連続プロセス、多変量相関監視 | 解釈性が高い、計算コスト低 | 線形仮定、正規分布仮定  
**Isolation Forest** | 非線形異常、複雑パターン | 仮定不要、高速、頑健 | パラメータ調整が必要  
**Autoencoder** | 高次元データ、非線形関係 | 柔軟性が高い、特徴抽出 | 訓練時間、ハイパーパラメータ  
**LSTM** | バッチプロセス、時系列パターン | 時間依存性学習 | データ量、計算コスト  
**Random Forest** | 品質予測、特徴選択 | 外れ値に頑健、解釈可能 | 外挿性能に制限  
**GPR** | データ少、不確実性重要 | 予測信頼区間、ベイズ的 | 計算コスト（大規模データ）  
**NN適応学習** | プロセスドリフト対策 | 長期精度維持 | 適応タイミング設計  
  
### 次の章へ

第2章では、**予知保全とRUL推定** を学びます：

  * 振動データ解析とスペクトル特徴抽出
  * LSTM/TCNによる劣化予測
  * 残存耐用寿命（RUL: Remaining Useful Life）推定
  * 故障モード分類と診断
  * 保全計画最適化
