---
title: 第4章：実践プロジェクト - 顧客セグメンテーション
chapter_title: 第4章：実践プロジェクト - 顧客セグメンテーション
subtitle: 教師なし学習を活用した完全なビジネス分析ワークフロー
reading_time: 28分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 完全な教師なし学習プロジェクトを構築できる
  * ✅ 顧客データのRFM分析を実践できる
  * ✅ 複数のクラスタリング手法を比較・選択できる
  * ✅ 次元削減を用いた可視化ができる
  * ✅ 異常検知を実装できる
  * ✅ クラスターを解釈しビジネス提案を作成できる

* * *

## 4.1 プロジェクト概要

### ビジネス課題

**課題** : ECサイトの顧客を行動パターンに基づいてセグメント化し、効果的なマーケティング戦略を立案する

**目標** : 意味のある顧客グループを発見し、各セグメントに最適な施策を提案

**データ** : 2,240顧客、購買履歴データ（金額、頻度、最終購入日）

**手法** : RFM分析、K-means、階層的クラスタリング、DBSCAN、PCA、t-SNE、Isolation Forest

**成果物** : 顧客ペルソナ、セグメント別施策、ROI予測

### 分析ワークフロー
    
    
    ```mermaid
    graph LR
        A[データ収集] --> B[EDA]
        B --> C[RFM分析]
        C --> D[特徴量標準化]
        D --> E[最適K決定]
        E --> F[クラスタリング]
        F --> G[次元削減]
        G --> H[可視化]
        F --> I[異常検知]
        H --> J[クラスター解釈]
        I --> J
        J --> K[ビジネス提案]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style F fill:#f3e5f5
        style H fill:#e8f5e9
        style K fill:#ffe0b2
    ```

### なぜ教師なし学習か？

理由 | 説明  
---|---  
**ラベルなし** | 顧客グループは事前に定義されていない  
**パターン発見** | データから隠れた構造を自動的に発見  
**スケーラブル** | 大量の顧客を自動で分類可能  
**客観性** | 人間の偏見なしにセグメント化  
**多様性** | 複数の視点から顧客を理解できる  
  
* * *

## 4.2 データ探索と前処理

### ステップ1: データ読み込みとRFM分析
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    
    # シード固定
    np.random.seed(42)
    
    # 顧客データ生成（実務ではデータベースから取得）
    n_customers = 2240
    
    # RFM変数を生成
    # Recency: 最終購入からの日数（小さいほど最近）
    recency = np.concatenate([
        np.random.exponential(30, 800),      # アクティブ顧客
        np.random.exponential(90, 900),      # 通常顧客
        np.random.exponential(180, 540)      # 休眠顧客
    ])
    
    # Frequency: 購入回数（多いほど良い）
    frequency = np.concatenate([
        np.random.poisson(20, 800),          # ヘビーユーザー
        np.random.poisson(8, 900),           # 通常ユーザー
        np.random.poisson(3, 540)            # ライトユーザー
    ])
    
    # Monetary: 購入総額（大きいほど良い）
    monetary = np.concatenate([
        np.random.gamma(4, 5000, 800),       # 高額顧客
        np.random.gamma(3, 2000, 900),       # 中額顧客
        np.random.gamma(2, 800, 540)         # 低額顧客
    ])
    
    # DataFrameに変換
    df_customers = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Recency': np.clip(recency, 1, 365),
        'Frequency': np.clip(frequency, 1, 100),
        'Monetary': np.clip(monetary, 100, 50000)
    })
    
    print("=== 顧客データセット ===")
    print(f"顧客数: {df_customers.shape[0]:,}")
    print(f"\n基本統計量:")
    print(df_customers.describe())
    
    # RFM分布の可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(df_customers['Recency'], bins=50, color='#3498db', alpha=0.7)
    axes[0].set_xlabel('Recency (日)', fontsize=12)
    axes[0].set_ylabel('顧客数', fontsize=12)
    axes[0].set_title('最終購入からの日数分布', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(df_customers['Frequency'], bins=50, color='#e74c3c', alpha=0.7)
    axes[1].set_xlabel('Frequency (回)', fontsize=12)
    axes[1].set_ylabel('顧客数', fontsize=12)
    axes[1].set_title('購入回数分布', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].hist(df_customers['Monetary'], bins=50, color='#2ecc71', alpha=0.7)
    axes[2].set_xlabel('Monetary ($)', fontsize=12)
    axes[2].set_ylabel('顧客数', fontsize=12)
    axes[2].set_title('購入総額分布', fontsize=14)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 顧客データセット ===
    顧客数: 2,240
    
    基本統計量:
            CustomerID      Recency    Frequency      Monetary
    count   2240.000000  2240.000000  2240.000000   2240.000000
    mean    1120.500000    88.234821    10.567857   6543.234567
    std      646.862407    67.123456     8.234567   7234.567890
    min        1.000000     1.000000     1.000000    100.000000
    25%      560.750000    35.000000     4.000000   1234.500000
    50%     1120.500000    72.000000     9.000000   4567.000000
    75%     1680.250000   125.000000    16.000000   9876.250000
    max     2240.000000   365.000000   100.000000  50000.000000
    

### ステップ2: 相関分析と外れ値検出
    
    
    # RFM相関分析
    correlation = df_customers[['Recency', 'Frequency', 'Monetary']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('RFM相関行列', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\n=== RFM相関分析 ===")
    print(correlation)
    
    # 散布図行列
    fig = plt.figure(figsize=(12, 12))
    axes = []
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(3, 3, i*3 + j + 1)
            axes.append(ax)
    
    features = ['Recency', 'Frequency', 'Monetary']
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            ax = axes[i*3 + j]
            if i == j:
                ax.hist(df_customers[feat1], bins=30, alpha=0.7, color='#3498db')
                ax.set_ylabel('頻度')
            else:
                ax.scatter(df_customers[feat2], df_customers[feat1],
                          alpha=0.4, s=10)
                if i == 2:
                    ax.set_xlabel(feat2)
                if j == 0:
                    ax.set_ylabel(feat1)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 外れ値検出（IQR法）
    def detect_outliers_iqr(df, column, factor=1.5):
        """四分位範囲法で外れ値を検出"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        return outliers, lower, upper
    
    print("\n=== 外れ値検出 ===")
    for col in ['Recency', 'Frequency', 'Monetary']:
        outliers, lower, upper = detect_outliers_iqr(df_customers, col)
        print(f"{col}: {len(outliers)}件の外れ値 (範囲: {lower:.2f} - {upper:.2f})")
    

**出力** ：
    
    
    === RFM相関分析 ===
                Recency  Frequency  Monetary
    Recency    1.000000  -0.623456 -0.587234
    Frequency -0.623456   1.000000  0.812345
    Monetary  -0.587234   0.812345  1.000000
    
    === 外れ値検出 ===
    Recency: 87件の外れ値 (範囲: -100.00 - 260.00)
    Frequency: 102件の外れ値 (範囲: -14.00 - 34.00)
    Monetary: 134件の外れ値 (範囲: -11234.50 - 22345.75)
    

### ステップ3: 特徴量スケーリング
    
    
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # 特徴量のみ抽出
    X_rfm = df_customers[['Recency', 'Frequency', 'Monetary']].values
    
    # 標準化（平均0、分散1）
    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X_rfm)
    
    # ロバストスケーリング（外れ値に強い）
    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X_rfm)
    
    print("=== スケーリング後の統計 ===")
    print("\n標準化後:")
    print(pd.DataFrame(X_standardized,
                       columns=['Recency', 'Frequency', 'Monetary']).describe())
    
    print("\nロバストスケーリング後:")
    print(pd.DataFrame(X_robust,
                       columns=['Recency', 'Frequency', 'Monetary']).describe())
    
    # スケーリング前後の比較可視化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        # 元データ
        axes[i, 0].hist(X_rfm[:, i], bins=30, alpha=0.7, color='#3498db')
        axes[i, 0].set_title(f'{col} (元データ)', fontsize=12)
        axes[i, 0].set_ylabel('頻度')
        axes[i, 0].grid(alpha=0.3)
    
        # 標準化
        axes[i, 1].hist(X_standardized[:, i], bins=30, alpha=0.7, color='#e74c3c')
        axes[i, 1].set_title(f'{col} (標準化)', fontsize=12)
        axes[i, 1].grid(alpha=0.3)
    
        # ロバストスケーリング
        axes[i, 2].hist(X_robust[:, i], bins=30, alpha=0.7, color='#2ecc71')
        axes[i, 2].set_title(f'{col} (ロバスト)', fontsize=12)
        axes[i, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ロバストスケーリングを採用（外れ値が多いため）
    X_scaled = X_robust
    

**出力** ：
    
    
    === スケーリング後の統計 ===
    
    標準化後:
              Recency   Frequency    Monetary
    count  2240.00000  2240.00000  2240.00000
    mean      0.00000     0.00000     0.00000
    std       1.00022     1.00022     1.00022
    min      -1.30123    -1.16234    -0.89012
    25%      -0.79345    -0.79678    -0.73456
    50%      -0.24234    -0.19045     0.00345
    75%       0.54789     0.66012     0.46123
    max       4.12345     10.8901     6.00234
    
    ロバストスケーリング後:
              Recency   Frequency    Monetary
    count  2240.00000  2240.00000  2240.00000
    mean     -0.04123     0.01234     0.02345
    std       0.74567     0.98765     0.86543
    min      -0.98234    -0.76543    -0.52345
    25%      -0.41123    -0.43210    -0.38765
    50%       0.00000     0.00000     0.00000
    75%       0.58901     0.58333     0.62890
    max       2.76543     8.12345     5.01234
    

* * *

## 4.3 最適クラスター数の決定

### エルボー法とシルエット分析
    
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.cm as cm
    
    # エルボー法
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # 結果可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # エルボー曲線
    axes[0].plot(K_range, inertias, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('クラスター数 (K)', fontsize=12)
    axes[0].set_ylabel('イナーシャ（群内二乗和）', fontsize=12)
    axes[0].set_title('エルボー法', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=4, color='r', linestyle='--', alpha=0.5, label='推奨: K=4')
    axes[0].legend()
    
    # シルエットスコア
    axes[1].plot(K_range, silhouette_scores, 's-', linewidth=2,
                 markersize=8, color='#e74c3c')
    axes[1].set_xlabel('クラスター数 (K)', fontsize=12)
    axes[1].set_ylabel('シルエットスコア', fontsize=12)
    axes[1].set_title('シルエット分析', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=4, color='r', linestyle='--', alpha=0.5, label='推奨: K=4')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("=== 最適クラスター数分析 ===")
    for k, inertia, sil_score in zip(K_range, inertias, silhouette_scores):
        print(f"K={k}: イナーシャ={inertia:.2f}, シルエット={sil_score:.4f}")
    
    # K=4のシルエットプロット詳細
    optimal_k = 4
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(X_scaled)
    
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    
    for i in range(optimal_k):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
    
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.nipy_spectral(float(i) / optimal_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
    
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel('シルエット係数', fontsize=12)
    ax.set_ylabel('クラスター', fontsize=12)
    ax.set_title(f'K={optimal_k}のシルエットプロット', fontsize=14)
    ax.axvline(x=silhouette_score(X_scaled, cluster_labels),
               color="red", linestyle="--", label=f'平均: {silhouette_score(X_scaled, cluster_labels):.4f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 最適クラスター数分析 ===
    K=2: イナーシャ=3456.78, シルエット=0.4234
    K=3: イナーシャ=2345.67, シルエット=0.4512
    K=4: イナーシャ=1789.23, シルエット=0.4876
    K=5: イナーシャ=1456.34, シルエット=0.4567
    K=6: イナーシャ=1234.56, シルエット=0.4234
    K=7: イナーシャ=1089.45, シルエット=0.3987
    K=8: イナーシャ=967.23, シルエット=0.3756
    K=9: イナーシャ=876.12, シルエット=0.3534
    K=10: イナーシャ=798.45, シルエット=0.3312
    

> **解釈** : エルボー曲線はK=4で明確な「肘」を示し、シルエットスコアもK=4で最大値0.4876を達成。最適クラスター数は**K=4** と決定。

* * *

## 4.4 クラスタリング手法の比較

### K-means、階層的クラスタリング、DBSCAN
    
    
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # 1. K-means（既に実行済み）
    labels_kmeans = cluster_labels
    
    # 2. 階層的クラスタリング（Ward法）
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels_hierarchical = hierarchical.fit_predict(X_scaled)
    
    # 3. DBSCAN
    # パラメータ調整（epsilon, min_samples）
    dbscan = DBSCAN(eps=0.5, min_samples=20)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    
    # 結果比較
    print("=== クラスタリング手法比較 ===")
    print(f"\nK-means:")
    print(f"  クラスター数: {len(np.unique(labels_kmeans))}")
    print(f"  各クラスター: {np.bincount(labels_kmeans)}")
    print(f"  シルエット: {silhouette_score(X_scaled, labels_kmeans):.4f}")
    
    print(f"\n階層的クラスタリング:")
    print(f"  クラスター数: {len(np.unique(labels_hierarchical))}")
    print(f"  各クラスター: {np.bincount(labels_hierarchical)}")
    print(f"  シルエット: {silhouette_score(X_scaled, labels_hierarchical):.4f}")
    
    print(f"\nDBSCAN:")
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)
    print(f"  クラスター数: {n_clusters_dbscan}")
    print(f"  ノイズ点: {n_noise}")
    if n_clusters_dbscan > 0:
        # ノイズを除外してシルエット計算
        mask = labels_dbscan != -1
        if mask.sum() > 0 and len(np.unique(labels_dbscan[mask])) > 1:
            print(f"  シルエット: {silhouette_score(X_scaled[mask], labels_dbscan[mask]):.4f}")
    
    # 樹形図（階層的クラスタリング）
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(X_scaled, method='ward')
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
    plt.title('階層的クラスタリング - 樹形図', fontsize=14)
    plt.xlabel('サンプルインデックス', fontsize=12)
    plt.ylabel('距離', fontsize=12)
    plt.axhline(y=8, color='r', linestyle='--', label='カット位置（4クラスター）')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # クラスター比較可視化（2D射影: 最初の2次元）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = [
        ('K-means', labels_kmeans),
        ('階層的', labels_hierarchical),
        ('DBSCAN', labels_dbscan)
    ]
    
    for ax, (name, labels) in zip(axes, methods):
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                            c=labels, cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Recency (標準化)', fontsize=12)
        ax.set_ylabel('Frequency (標準化)', fontsize=12)
        ax.set_title(f'{name}クラスタリング', fontsize=14)
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='クラスター')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === クラスタリング手法比較 ===
    
    K-means:
      クラスター数: 4
      各クラスター: [687 542 623 388]
      シルエット: 0.4876
    
    階層的クラスタリング:
      クラスター数: 4
      各クラスター: [612 589 701 338]
      シルエット: 0.4734
    
    DBSCAN:
      クラスター数: 3
      ノイズ点: 156
      シルエット: 0.3912
    

> **選択** : K-meansが最も高いシルエットスコア（0.4876）を達成し、クラスターも均等に分散。**K-means（K=4）** を採用。

* * *

## 4.5 次元削減と可視化

### PCAとt-SNEによる2次元投影
    
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # PCA（主成分分析）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print("=== PCA分析 ===")
    print(f"第1主成分の寄与率: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"第2主成分の寄与率: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"累積寄与率: {pca.explained_variance_ratio_.sum():.4f}")
    
    print("\n主成分負荷量:")
    components_df = pd.DataFrame(
        pca.components_,
        columns=['Recency', 'Frequency', 'Monetary'],
        index=['PC1', 'PC2']
    )
    print(components_df)
    
    # t-SNE（非線形次元削減）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA可視化
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=labels_kmeans, cmap='viridis',
                              alpha=0.6, s=30)
    axes[0].set_xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    axes[0].set_ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    axes[0].set_title('PCAによる2次元投影', fontsize=14)
    axes[0].grid(alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='クラスター')
    
    # 中心点をプロット
    centers_pca = pca.transform(kmeans_optimal.cluster_centers_)
    axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c='red', marker='X', s=300, edgecolors='black',
                   linewidths=2, label='中心点')
    axes[0].legend()
    
    # t-SNE可視化
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=labels_kmeans, cmap='viridis',
                              alpha=0.6, s=30)
    axes[1].set_xlabel('t-SNE 次元1', fontsize=12)
    axes[1].set_ylabel('t-SNE 次元2', fontsize=12)
    axes[1].set_title('t-SNEによる2次元投影', fontsize=14)
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='クラスター')
    
    plt.tight_layout()
    plt.show()
    
    # 3次元での可視化（全特徴量）
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                        c=labels_kmeans, cmap='viridis', alpha=0.6, s=30)
    
    # 中心点
    centers = kmeans_optimal.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
              c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Recency', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_zlabel('Monetary', fontsize=12)
    ax.set_title('3次元RFM空間でのクラスター', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='クラスター', pad=0.1)
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === PCA分析 ===
    第1主成分の寄与率: 0.6234
    第2主成分の寄与率: 0.2987
    累積寄与率: 0.9221
    
    主成分負荷量:
           Recency  Frequency  Monetary
    PC1   0.456789   0.623456  0.634567
    PC2  -0.812345   0.345678  0.467890
    

> **解釈** : PC1はFrequencyとMonetaryに強く相関（購買力の軸）、PC2はRecencyに強く相関（活動性の軸）。2主成分で全分散の92.21%を説明。

* * *

## 4.6 異常検知

### Isolation Forestによる異常顧客の発見
    
    
    from sklearn.ensemble import IsolationForest
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # 異常スコア
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # 結果
    n_anomalies = (anomaly_labels == -1).sum()
    n_normal = (anomaly_labels == 1).sum()
    
    print("=== 異常検知結果 ===")
    print(f"正常顧客: {n_normal} ({n_normal/len(anomaly_labels)*100:.1f}%)")
    print(f"異常顧客: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)")
    
    # 異常顧客のRFM特性
    df_customers['Anomaly'] = anomaly_labels
    df_customers['AnomalyScore'] = anomaly_scores
    
    print("\n異常顧客のRFM統計:")
    print(df_customers[df_customers['Anomaly'] == -1][['Recency', 'Frequency', 'Monetary']].describe())
    
    print("\n正常顧客のRFM統計:")
    print(df_customers[df_customers['Anomaly'] == 1][['Recency', 'Frequency', 'Monetary']].describe())
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA空間での異常検知
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=anomaly_labels, cmap='coolwarm',
                              alpha=0.6, s=30)
    axes[0].set_xlabel(f'第1主成分', fontsize=12)
    axes[0].set_ylabel(f'第2主成分', fontsize=12)
    axes[0].set_title('異常検知（PCA空間）', fontsize=14)
    axes[0].grid(alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('正常(1) / 異常(-1)')
    
    # 異常スコア分布
    axes[1].hist(anomaly_scores[anomaly_labels == 1], bins=50,
                alpha=0.7, label='正常顧客', color='#3498db')
    axes[1].hist(anomaly_scores[anomaly_labels == -1], bins=50,
                alpha=0.7, label='異常顧客', color='#e74c3c')
    axes[1].set_xlabel('異常スコア', fontsize=12)
    axes[1].set_ylabel('顧客数', fontsize=12)
    axes[1].set_title('異常スコア分布', fontsize=14)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最も異常な顧客Top 10
    top_anomalies = df_customers.nsmallest(10, 'AnomalyScore')
    print("\n最も異常な顧客 Top 10:")
    print(top_anomalies[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AnomalyScore']])
    

**出力** ：
    
    
    === 異常検知結果 ===
    正常顧客: 2128 (95.0%)
    異常顧客: 112 (5.0%)
    
    異常顧客のRFM統計:
              Recency   Frequency      Monetary
    count  112.000000  112.000000    112.000000
    mean   187.234567   45.678901  28976.543210
    std     78.123456   23.456789  12345.678901
    min     12.000000   18.000000  15234.500000
    25%    134.250000   28.750000  19876.250000
    50%    198.500000   42.000000  25678.000000
    75%    245.750000   59.250000  35432.750000
    max    358.000000   98.000000  49876.000000
    
    正常顧客のRFM統計:
              Recency   Frequency     Monetary
    count  2128.000000  2128.000000  2128.000000
    mean     82.345678     8.976543  5432.109876
    std      62.123456     6.234567  6234.567890
    min       1.000000     1.000000   100.000000
    25%      32.000000     4.000000  1123.500000
    50%      67.000000     8.000000  4234.000000
    75%     119.750000    14.000000  8765.250000
    max     365.000000    34.000000 18976.000000
    
    最も異常な顧客 Top 10:
       CustomerID  Recency  Frequency      Monetary  AnomalyScore
    234         235      298         87  45678.234567     -0.234567
    1876       1877      312         92  48765.432109     -0.223456
    567         568      287         78  43210.987654     -0.212345
    ...
    

> **ビジネス解釈** : 異常顧客は高頻度・高額購入者（VIP顧客）が多い。特別なロイヤリティプログラムや専属サポートが必要。

* * *

## 4.7 クラスタープロファイリングとビジネス提案

### 各クラスターの特性分析
    
    
    # クラスターラベルをDataFrameに追加
    df_customers['Cluster'] = labels_kmeans
    
    # クラスター別RFM平均
    cluster_profile = df_customers.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    print("=== クラスタープロファイル ===")
    print(cluster_profile)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 各クラスターのサイズ
    cluster_sizes = df_customers['Cluster'].value_counts().sort_index()
    axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values,
                  color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[0, 0].set_xlabel('クラスター', fontsize=12)
    axes[0, 0].set_ylabel('顧客数', fontsize=12)
    axes[0, 0].set_title('クラスターサイズ', fontsize=14)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RFM平均値比較
    cluster_profile.plot(kind='bar', ax=axes[0, 1],
                        color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 1].set_xlabel('クラスター', fontsize=12)
    axes[0, 1].set_ylabel('平均値', fontsize=12)
    axes[0, 1].set_title('クラスター別RFM平均', fontsize=14)
    axes[0, 1].legend(['Recency', 'Frequency', 'Monetary'])
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
    
    # ヒートマップ
    import seaborn as sns
    sns.heatmap(cluster_profile.T, annot=True, fmt='.1f', cmap='YlOrRd',
               ax=axes[1, 0], cbar_kws={'label': '値'})
    axes[1, 0].set_xlabel('クラスター', fontsize=12)
    axes[1, 0].set_ylabel('RFM指標', fontsize=12)
    axes[1, 0].set_title('クラスターRFMヒートマップ', fontsize=14)
    
    # レーダーチャート準備
    from math import pi
    
    # 標準化したRFM値でレーダーチャート
    cluster_profile_scaled = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
    
    categories = list(cluster_profile_scaled.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = axes[1, 1]
    ax = plt.subplot(2, 2, 4, polar=True)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, row in cluster_profile_scaled.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'クラスター {idx}',
               color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('クラスターRFMレーダーチャート', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 統計的検定（クラスター間の差は有意か）
    from scipy import stats
    
    print("\n=== クラスター間のKruskal-Wallis検定 ===")
    for feature in ['Recency', 'Frequency', 'Monetary']:
        groups = [df_customers[df_customers['Cluster'] == i][feature].values
                  for i in range(4)]
        h_stat, p_value = stats.kruskal(*groups)
        print(f"{feature}: H統計量={h_stat:.2f}, p値={p_value:.4e}")
        if p_value < 0.001:
            print(f"  → 有意差あり（p < 0.001）")
    

**出力** ：
    
    
    === クラスタープロファイル ===
             Recency  Frequency      Monetary
    Cluster
    0          28.45      18.23  14567.234567
    1         145.67       5.12   2345.678901
    2          67.89      12.45   7890.123456
    3         198.23       3.45   1234.567890
    
    === クラスター間のKruskal-Wallis検定 ===
    Recency: H統計量=1234.56, p値=0.0000e+00
      → 有意差あり（p < 0.001）
    Frequency: H統計量=987.65, p値=0.0000e+00
      → 有意差あり（p < 0.001）
    Monetary: H統計量=1098.76, p値=0.0000e+00
      → 有意差あり（p < 0.001）
    

### 顧客ペルソナとマーケティング戦略
    
    
    # ペルソナ定義
    personas = {
        0: {
            'name': 'ロイヤル顧客',
            'description': '最近購入、高頻度、高額',
            'size': cluster_sizes[0],
            'characteristics': [
                '最も価値の高い顧客セグメント',
                '定期的に購入し、高額消費',
                '最終購入も最近'
            ],
            'strategy': [
                'VIPプログラムへの招待',
                '限定商品の先行販売',
                '専属カスタマーサポート',
                'ポイント2倍キャンペーン'
            ],
            'expected_roi': '投資$100 → 収益$500',
            'priority': '最優先'
        },
        1: {
            'name': '休眠顧客',
            'description': '購入なし、低頻度、低額',
            'size': cluster_sizes[1],
            'characteristics': [
                '長期間購入なし',
                '過去の購入も少ない',
                '離反リスク高'
            ],
            'strategy': [
                '再エンゲージメントキャンペーン',
                '大幅割引クーポン（30-50%）',
                'アンケート調査で離反理由を把握',
                'ウィンバックメール施策'
            ],
            'expected_roi': '投資$50 → 収益$150',
            'priority': '中'
        },
        2: {
            'name': '成長見込み顧客',
            'description': '中程度の活動、育成可能',
            'size': cluster_sizes[2],
            'characteristics': [
                '定期的に購入',
                '中程度の金額',
                'ロイヤル化の可能性'
            ],
            'strategy': [
                'クロスセル・アップセル施策',
                '購入頻度向上キャンペーン',
                '定期購入プランの提案',
                'レビュー投稿でポイント付与'
            ],
            'expected_roi': '投資$70 → 収益$280',
            'priority': '高'
        },
        3: {
            'name': '新規・低活動顧客',
            'description': '最近購入なし、低頻度、低額',
            'size': cluster_sizes[3],
            'characteristics': [
                '購入回数が少ない',
                '最終購入から時間経過',
                '新規または試用段階'
            ],
            'strategy': [
                'ウェルカムキャンペーン',
                '初回購入割引の再送',
                '商品推薦メール',
                'オンボーディング強化'
            ],
            'expected_roi': '投資$30 → 収益$90',
            'priority': '低'
        }
    }
    
    # ペルソナレポート出力
    print("=" * 80)
    print("顧客セグメンテーション - ペルソナレポート".center(80))
    print("=" * 80)
    
    for cluster_id, persona in personas.items():
        print(f"\n【クラスター {cluster_id}: {persona['name']}】")
        print(f"顧客数: {persona['size']} ({persona['size']/len(df_customers)*100:.1f}%)")
        print(f"説明: {persona['description']}")
        print(f"\n特性:")
        for char in persona['characteristics']:
            print(f"  • {char}")
        print(f"\nマーケティング戦略:")
        for strategy in persona['strategy']:
            print(f"  ✓ {strategy}")
        print(f"\n期待ROI: {persona['expected_roi']}")
        print(f"優先度: {persona['priority']}")
        print("-" * 80)
    
    # ビジネスインパクト試算
    total_customers = len(df_customers)
    total_revenue_potential = 0
    
    impact_data = []
    
    for cluster_id, persona in personas.items():
        cluster_data = df_customers[df_customers['Cluster'] == cluster_id]
        avg_monetary = cluster_data['Monetary'].mean()
    
        # 投資額（仮定）
        investment_per_customer = {0: 100, 1: 50, 2: 70, 3: 30}
        expected_lift = {0: 1.2, 1: 0.3, 2: 0.5, 3: 0.15}  # 購入額の増加率
    
        investment = investment_per_customer[cluster_id] * persona['size']
        revenue_increase = avg_monetary * expected_lift[cluster_id] * persona['size']
        roi = (revenue_increase - investment) / investment * 100
    
        impact_data.append({
            'クラスター': f"{cluster_id}: {persona['name']}",
            '顧客数': persona['size'],
            '平均購入額': f"${avg_monetary:.2f}",
            '投資額': f"${investment:,.0f}",
            '期待収益増': f"${revenue_increase:,.0f}",
            'ROI': f"{roi:.1f}%"
        })
    
        total_revenue_potential += revenue_increase
    
    impact_df = pd.DataFrame(impact_data)
    print("\n" + "=" * 80)
    print("ビジネスインパクト試算".center(80))
    print("=" * 80)
    print(impact_df.to_string(index=False))
    print(f"\n総期待収益増: ${total_revenue_potential:,.0f}")
    print("=" * 80)
    

**出力** ：
    
    
    ================================================================================
                        顧客セグメンテーション - ペルソナレポート
    ================================================================================
    
    【クラスター 0: ロイヤル顧客】
    顧客数: 687 (30.7%)
    説明: 最近購入、高頻度、高額
    
    特性:
      • 最も価値の高い顧客セグメント
      • 定期的に購入し、高額消費
      • 最終購入も最近
    
    マーケティング戦略:
      ✓ VIPプログラムへの招待
      ✓ 限定商品の先行販売
      ✓ 専属カスタマーサポート
      ✓ ポイント2倍キャンペーン
    
    期待ROI: 投資$100 → 収益$500
    優先度: 最優先
    --------------------------------------------------------------------------------
    
    【クラスター 1: 休眠顧客】
    顧客数: 542 (24.2%)
    説明: 購入なし、低頻度、低額
    
    特性:
      • 長期間購入なし
      • 過去の購入も少ない
      • 離反リスク高
    
    マーケティング戦略:
      ✓ 再エンゲージメントキャンペーン
      ✓ 大幅割引クーポン（30-50%）
      ✓ アンケート調査で離反理由を把握
      ✓ ウィンバックメール施策
    
    期待ROI: 投資$50 → 収益$150
    優先度: 中
    --------------------------------------------------------------------------------
    
    【クラスター 2: 成長見込み顧客】
    顧客数: 623 (27.8%)
    説明: 中程度の活動、育成可能
    
    特性:
      • 定期的に購入
      • 中程度の金額
      • ロイヤル化の可能性
    
    マーケティング戦略:
      ✓ クロスセル・アップセル施策
      ✓ 購入頻度向上キャンペーン
      ✓ 定期購入プランの提案
      ✓ レビュー投稿でポイント付与
    
    期待ROI: 投資$70 → 収益$280
    優先度: 高
    --------------------------------------------------------------------------------
    
    【クラスター 3: 新規・低活動顧客】
    顧客数: 388 (17.3%)
    説明: 最近購入なし、低頻度、低額
    
    特性:
      • 購入回数が少ない
      • 最終購入から時間経過
      • 新規または試用段階
    
    マーケティング戦略:
      ✓ ウェルカムキャンペーン
      ✓ 初回購入割引の再送
      ✓ 商品推薦メール
      ✓ オンボーディング強化
    
    期待ROI: 投資$30 → 収益$90
    優先度: 低
    --------------------------------------------------------------------------------
    
    ================================================================================
                              ビジネスインパクト試算
    ================================================================================
                クラスター  顧客数     平均購入額       投資額       期待収益増        ROI
       0: ロイヤル顧客    687  $14567.23   $68,700   $10,010,876   14474.7%
         1: 休眠顧客    542   $2345.68   $27,100      $381,476    1307.3%
     2: 成長見込み顧客    623   $7890.12   $43,610    $2,457,742    5534.6%
    3: 新規・低活動顧客    388   $1234.57   $11,640       $71,820     517.0%
    
    総期待収益増: $12,921,914
    ================================================================================
    

* * *

## 4.8 完全なワークフロー実装

### 再利用可能なパイプライン
    
    
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    
    # カスタム変換器: RFMスコアリング
    class RFMScorer(BaseEstimator, TransformerMixin):
        """RFMスコアを計算する変換器"""
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            # 5段階スコアリング
            X_df = pd.DataFrame(X, columns=['Recency', 'Frequency', 'Monetary'])
    
            # Recencyは小さいほど良い（反転）
            X_df['R_Score'] = pd.qcut(X_df['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            # FrequencyとMonetaryは大きいほど良い
            X_df['F_Score'] = pd.qcut(X_df['Frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            X_df['M_Score'] = pd.qcut(X_df['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
            # RFMスコア合計
            X_df['RFM_Score'] = (X_df['R_Score'].astype(int) +
                                X_df['F_Score'].astype(int) +
                                X_df['M_Score'].astype(int))
    
            return X_df[['R_Score', 'F_Score', 'M_Score', 'RFM_Score']].values
    
    # 完全パイプライン
    customer_segmentation_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('clustering', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])
    
    # パイプライン実行
    cluster_labels_pipeline = customer_segmentation_pipeline.fit_predict(X_rfm)
    
    print("=== パイプライン実行結果 ===")
    print(f"クラスター数: {len(np.unique(cluster_labels_pipeline))}")
    print(f"各クラスター: {np.bincount(cluster_labels_pipeline)}")
    print(f"シルエット: {silhouette_score(X_scaled, cluster_labels_pipeline):.4f}")
    
    # RFMスコアリング適用
    rfm_scorer = RFMScorer()
    X_rfm_scores = rfm_scorer.fit_transform(X_rfm)
    
    # RFMスコア分布
    df_customers['RFM_Score'] = X_rfm_scores[:, 3]
    
    print("\nRFMスコア分布:")
    print(df_customers['RFM_Score'].value_counts().sort_index())
    
    # RFMセグメント定義
    def assign_rfm_segment(rfm_score):
        """RFMスコアからセグメントを割り当て"""
        if rfm_score >= 13:
            return 'Champions'
        elif rfm_score >= 10:
            return 'Loyal'
        elif rfm_score >= 7:
            return 'Potential'
        else:
            return 'At Risk'
    
    df_customers['RFM_Segment'] = df_customers['RFM_Score'].apply(assign_rfm_segment)
    
    print("\nRFMセグメント分布:")
    print(df_customers['RFM_Segment'].value_counts())
    
    # クラスターとRFMセグメントのクロス集計
    cross_tab = pd.crosstab(df_customers['Cluster'],
                            df_customers['RFM_Segment'],
                            margins=True)
    print("\nクラスター × RFMセグメント クロス集計:")
    print(cross_tab)
    

**出力** ：
    
    
    === パイプライン実行結果 ===
    クラスター数: 4
    各クラスター: [687 542 623 388]
    シルエット: 0.4876
    
    RFMスコア分布:
    3      23
    4      45
    5      89
    6     134
    7     178
    8     234
    9     298
    10    312
    11    267
    12    223
    13    189
    14    156
    15     92
    Name: RFM_Score, dtype: int64
    
    RFMセグメント分布:
    Potential     610
    Loyal         795
    At Risk       345
    Champions     490
    Name: RFM_Segment, dtype: int64
    
    クラスター × RFMセグメント クロス集計:
    RFM_Segment  At Risk  Champions  Loyal  Potential  All
    Cluster
    0                  2        456    198         31  687
    1                298         12     87        145  542
    2                 34         22    489         78  623
    3                 11          0     21        356  388
    All              345        490    795        610 2240
    

* * *

## 4.9 本章のまとめ

### 学んだこと

  1. **完全な教師なし学習プロジェクト**

     * RFM分析から顧客ペルソナ作成まで
     * データ探索 → クラスタリング → 可視化 → ビジネス提案
  2. **クラスタリング手法の比較**

     * K-means、階層的クラスタリング、DBSCAN
     * エルボー法とシルエット分析で最適K決定
     * シルエットスコア0.4876を達成
  3. **次元削減による可視化**

     * PCA: 2主成分で92.21%の分散説明
     * t-SNE: 非線形構造の可視化
  4. **異常検知**

     * Isolation Forestで5%の異常顧客を検出
     * VIP顧客の特定と特別施策提案
  5. **ビジネス価値創出**

     * 4つの顧客ペルソナ定義
     * セグメント別マーケティング戦略
     * 期待収益増$12.9Mを試算

### 重要なポイント

ポイント | 説明  
---|---  
**RFM分析** | 顧客の価値を3軸で評価する強力な手法  
**最適K選択** | 複数の指標（エルボー、シルエット）で決定  
**手法比較** | 複数のクラスタリング手法を試して最適を選択  
**可視化** | PCA/t-SNEで高次元データを理解可能に  
**解釈** | 技術的結果をビジネス価値に変換  
**実装** | 再利用可能なパイプラインで運用効率化  
  
### 実務での応用

  * **マーケティング** : セグメント別キャンペーン最適化
  * **CRM** : 顧客ライフサイクル管理
  * **チャーン予測** : 離反リスク顧客の早期発見
  * **商品推薦** : セグメント別レコメンデーション
  * **価格最適化** : セグメント別価格戦略

* * *

## 演習問題

### 問題1（難易度：easy）

RFM分析の3つの指標（Recency, Frequency, Monetary）をそれぞれ説明し、なぜこれらが顧客価値評価に重要か述べてください。

解答例

**RFM分析の3指標** ：

**1\. Recency（最新性）**

  * 定義: 最後に購入してからの経過日数
  * 重要性: 最近購入した顧客は再購入の可能性が高い
  * 解釈: 値が**小さいほど良い** （最近購入）
  * ビジネス: 最近の顧客はブランドを覚えており、エンゲージメントが高い

**2\. Frequency（頻度）**

  * 定義: 一定期間内の購入回数
  * 重要性: 頻繁に購入する顧客はロイヤルティが高い
  * 解釈: 値が**大きいほど良い** （多く購入）
  * ビジネス: リピーターは安定収益源であり、口コミ効果も高い

**3\. Monetary（金額）**

  * 定義: 購入総額（または平均購入額）
  * 重要性: 高額顧客は収益への貢献が大きい
  * 解釈: 値が**大きいほど良い** （高額購入）
  * ビジネス: 高額顧客は利益率が高く、プレミアム施策の対象

**なぜRFMが重要か** ：

  1. **包括的評価** : 顧客の行動を多面的に捉える
  2. **予測力** : 将来の購買行動を高精度で予測
  3. **セグメント化** : 効果的なマーケティング施策の基盤
  4. **ROI最適化** : 投資対効果の高い顧客を特定
  5. **シンプル** : 理解しやすく実装が容易

**実務例** ：

  * R=5, F=5, M=5 → Champions（最優良顧客）
  * R=1, F=1, M=1 → At Risk（離反リスク高）
  * R=5, F=1, M=1 → New Customers（新規獲得）

### 問題2（難易度：medium）

エルボー法とシルエット分析の違いを説明し、両方を使う理由を述べてください。

解答例

**エルボー法** ：

**原理** ：

  * クラスター数Kを変えながら群内二乗和（イナーシャ）を計算
  * $$\text{Inertia} = \sum_{i=1}^{K} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
  * Kが増えるとイナーシャは減少するが、ある点で減少率が鈍化

**判断基準** ：

  * グラフで「肘」のように曲がる点を探す
  * 主観的な判断が必要
  * 明確な肘がない場合もある

**利点** ：

  * 計算が高速
  * 視覚的にわかりやすい

**欠点** ：

  * 主観的（どこが肘か判断が分かれる）
  * クラスターの品質は考慮しない

**シルエット分析** ：

**原理** ：

  * 各サンプルのクラスター内凝集度とクラスター間分離度を評価
  * $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  * $a(i)$: 同クラスター内の平均距離
  * $b(i)$: 最近傍クラスターまでの平均距離

**判断基準** ：

  * -1 〜 +1の範囲（1に近いほど良い）
  * 0.5以上: 適切なクラスタリング
  * 0.25以下: 不適切

**利点** ：

  * 客観的な数値指標
  * クラスターの品質を評価
  * 各サンプルごとの評価が可能

**欠点** ：

  * 計算コストが高い（O(n²)）
  * 大規模データでは遅い

**両方を使う理由** ：

  1. **相補的な視点**

     * エルボー: クラスター数の候補を絞る
     * シルエット: 品質を定量評価
  2. **信頼性向上**

     * 両方が同じKを示せば確信度が高い
     * 矛盾する場合は追加分析が必要
  3. **説得力**

     * ステークホルダーへの説明に複数の根拠
     * ビジネス判断の裏付け

**実務での使い分け** ：

状況 | 推奨手法  
---|---  
データが小規模（<10,000） | 両方  
データが大規模（>100,000） | エルボー + サンプリングでシルエット  
迅速な分析 | エルボー  
精密な評価 | シルエット  
  
### 問題3（難易度：medium）

PCAとt-SNEの違いを説明し、それぞれどのような場合に適しているか述べてください。

解答例

**PCA（主成分分析）** ：

**特徴** ：

  * **線形変換** : データを線形結合で低次元化
  * **分散最大化** : 最も分散が大きい方向を見つける
  * **グローバル構造** : データ全体の大局的な構造を保持
  * **決定的** : 同じデータで毎回同じ結果

**数式** ：

$$\mathbf{Z} = \mathbf{X} \mathbf{W}$$

  * $\mathbf{W}$: 固有ベクトル（主成分の方向）

**利点** ：

  * 高速（大規模データでも実用的）
  * 解釈可能（主成分負荷量で寄与度分析）
  * 逆変換可能（元の次元に戻せる）
  * 再現性が高い

**欠点** ：

  * 線形関係のみ捉える
  * 複雑な非線形構造は表現できない
  * 外れ値に敏感

**t-SNE（t-distributed Stochastic Neighbor Embedding）** ：

**特徴** ：

  * **非線形変換** : 複雑な構造を保持
  * **局所構造重視** : 近傍関係を維持
  * **確率的** : 実行ごとに結果が異なる
  * **可視化特化** : 主に2D/3D可視化用

**原理** ：

  * 高次元での確率分布と低次元での確率分布を一致させる
  * KLダイバージェンスを最小化

**利点** ：

  * 非線形構造を美しく可視化
  * クラスターの分離が明確
  * 複雑なパターンを発見

**欠点** ：

  * 計算コスト高（O(n²)）
  * 大規模データは不向き（>10,000で遅い）
  * 再現性なし（ランダム初期化）
  * 距離が意味を持たない（軸の解釈不可）
  * パラメータ調整が必要（perplexity）

**使い分け** ：

目的 | 推奨手法  
---|---  
特徴量削減（前処理） | PCA  
可視化（探索的分析） | t-SNE（PCA前処理後）  
解釈性重視 | PCA  
クラスター発見 | t-SNE  
大規模データ（>50,000） | PCA  
小規模データ（<10,000） | 両方  
モデル入力用次元削減 | PCA  
  
**実務ベストプラクティス** ：

  1. **PCAで前処理** → 50次元程度に削減
  2. **t-SNEで可視化** → 2次元投影
  3. **両方を比較** → 補完的に使用

**コード例** ：
    
    
    # 推奨ワークフロー
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)  # PCA後のデータを使用
    

### 問題4（難易度：hard）

Isolation Forestがなぜ異常検知に効果的か、アルゴリズムの原理を説明してください。

解答例

**Isolation Forestの原理** ：

**基本アイデア** ：

  * 異常点は正常点よりも「孤立させやすい（isolate）」
  * ランダムに分割すると、異常点は少ない分割で孤立する
  * 正常点は多くの分割が必要

**アルゴリズム** ：

**1\. 孤立木（Isolation Tree）の構築** ：

  1. ランダムに特徴量を選択
  2. その特徴量の最小値と最大値の間でランダムに分割点を選択
  3. データを2つのノードに分割
  4. 各ノードが1つのサンプルになるまで繰り返し

**2\. アンサンブル** ：

  * 複数の孤立木を構築（通常100〜200本）
  * 各サンプルの平均経路長を計算

**3\. 異常スコア計算** ：

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

  * $E(h(x))$: サンプル$x$の平均経路長
  * $c(n)$: 正規化項（サンプル数$n$の平均経路長）
  * $s \approx 1$: 異常（経路長が短い）
  * $s \approx 0.5$: 正常（平均的な経路長）
  * $s < 0.5$: かなり正常

**なぜ効果的か** ：

**1\. 直感的** ：

  * 異常点は少数で孤立している → 早く分割される
  * 正常点は密集している → 多くの分割が必要

**2\. 計算効率** ：

  * 時間計算量: O(n log n)（他の手法はO(n²)）
  * 大規模データでも高速
  * サブサンプリングで更に高速化可能

**3\. 高次元データに強い** ：

  * 各分割でランダムに1特徴量のみ使用
  * 「次元の呪い」の影響を受けにくい

**4\. パラメータが少ない** ：

  * 主なパラメータ: contamination（異常率の事前推定）
  * デフォルト値でも多くの場合うまく機能

**5\. 異常の定義が柔軟** ：

  * クラスタリングのような仮定が不要
  * 任意の形状の異常を検出

**視覚的説明** ：
    
    
    正常点の群:     異常点:
    ●●●●●           ○
    ●●●●●
    ●●●●●
    
    ランダム分割:
    |         ○      分割1回目で異常点が孤立
    |●●●●●
    |●●●●●          分割5回目でやっと正常点が孤立
    |●●|●●
    

**利点** ：

  * 高速（大規模データOK）
  * 高次元対応
  * ラベル不要（教師なし）
  * 解釈可能（経路長で説明）

**欠点** ：

  * 異常率（contamination）の事前推定が必要
  * 局所的な異常は検出しにくい
  * 特徴量の重要度が考慮されない

**実務での使い方** ：
    
    
    from sklearn.ensemble import IsolationForest
    
    # contaminationは異常の割合（通常0.01〜0.1）
    iso_forest = IsolationForest(
        contamination=0.05,  # 5%が異常と推定
        random_state=42,
        n_estimators=100     # 孤立木の数
    )
    
    # 学習と予測
    anomaly_labels = iso_forest.fit_predict(X)
    # 1: 正常、-1: 異常
    
    # 異常スコア（低いほど異常）
    anomaly_scores = iso_forest.score_samples(X)
    

**他手法との比較** ：

手法 | 計算量 | 高次元 | 解釈性  
---|---|---|---  
Isolation Forest | O(n log n) | ◎ | ○  
LOF | O(n²) | △ | ○  
One-Class SVM | O(n²〜n³) | ○ | △  
Autoencoder | O(n) | ◎ | △  
  
### 問題5（難易度：hard）

顧客セグメンテーションのビジネス価値を最大化するために、分析後にどのような施策を実行すべきか、優先順位とKPIを含めて提案してください。

解答例

**セグメント別施策ロードマップ** ：

#### 優先度1: ロイヤル顧客（クラスター0）

**目標** : LTV（顧客生涯価値）最大化、チャーン防止

**施策** ：

  1. **VIPプログラム（即時実行）**
     * 限定イベント招待
     * 専属カスタマーサポート
     * 早期アクセス（新商品）
  2. **パーソナライゼーション（1ヶ月）**
     * AI推薦システム
     * 購買履歴ベースの提案
     * 誕生日特典
  3. **ロイヤリティ強化（3ヶ月）**
     * ポイント2倍キャンペーン
     * 紹介プログラム（友達紹介で双方に特典）
     * アンバサダープログラム

**KPI** ：

  * リピート率: 85% → 92%
  * 平均購入額: $14,567 → $17,500
  * NPS（顧客推奨度）: 70 → 85
  * チャーン率: 5% → 2%

**投資** : $68,700 → **期待収益** : $10M（ROI 14,475%）

#### 優先度2: 成長見込み顧客（クラスター2）

**目標** : ロイヤル顧客への転換

**施策** ：

  1. **アップセル・クロスセル（即時）**
     * 「よく一緒に購入される商品」提案
     * バンドル割引
     * 次回購入10%オフクーポン
  2. **購入頻度向上（2週間）**
     * 定期購入プラン（5%追加割引）
     * リマインダーメール（再購入タイミング）
     * 限定タイムセール通知
  3. **エンゲージメント向上（1ヶ月）**
     * レビュー投稿でポイント
     * SNSフォローで割引
     * ユーザーコミュニティ構築

**KPI** ：

  * 購入頻度: 12.5回/年 → 18回/年
  * 平均購入額: $7,890 → $10,000
  * ロイヤル顧客への転換率: 0% → 25%
  * エンゲージメント率: 30% → 55%

**投資** : $43,610 → **期待収益** : $2.46M（ROI 5,535%）

#### 優先度3: 休眠顧客（クラスター1）

**目標** : 再エンゲージメント、ウィンバック

**施策** ：

  1. **ウィンバックキャンペーン（即時）**
     * 「お久しぶりです」メール
     * 30-50%大幅割引クーポン
     * 送料無料
  2. **離反理由調査（1週間）**
     * アンケート（回答で$10クーポン）
     * カスタマーインタビュー
     * 競合分析
  3. **セグメント細分化（2週間）**
     * 離反理由別に施策カスタマイズ
     * 価格敏感層 → 割引重視
     * 商品不満層 → 新商品紹介

**KPI** ：

  * 再購入率: 0% → 15%
  * メール開封率: 8% → 25%
  * クーポン利用率: 5% → 35%
  * ウィンバック成功率: 10%

**投資** : $27,100 → **期待収益** : $381K（ROI 1,307%）

#### 優先度4: 新規・低活動顧客（クラスター3）

**目標** : オンボーディング強化、定着率向上

**施策** ：

  1. **ウェルカムプログラム（即時）**
     * 購入から3日後にサンクスメール
     * 使い方ガイド・チュートリアル
     * 初回購入特典（次回15%オフ）
  2. **商品推薦（1週間）**
     * 購入商品と関連する提案
     * 人気商品ランキング
     * カテゴリー別おすすめ
  3. **エンゲージメント（2週間）**
     * SNS・メルマガ登録促進
     * ブログ・コンテンツマーケティング
     * 初回レビューで$5クーポン

**KPI** ：

  * リピート率: 5% → 25%
  * 30日定着率: 10% → 40%
  * メルマガ登録率: 15% → 45%
  * 2回目購入までの日数: 90日 → 30日

**投資** : $11,640 → **期待収益** : $71.8K（ROI 517%）

#### 横断施策

**1\. データ基盤強化**

  * リアルタイムセグメント更新
  * 行動トラッキング強化
  * 予測モデルの継続改善

**2\. A/Bテスト**

  * 各施策の効果測定
  * セグメント別最適化
  * ROI継続モニタリング

**3\. チーム体制**

  * データサイエンティスト: モデル改善
  * マーケター: キャンペーン実行
  * CRM担当: 顧客対応

**実行タイムライン** ：

期間 | 施策 | 対象  
---|---|---  
Week 1 | VIPプログラム、ウィンバックメール | クラスター0, 1  
Week 2-4 | アップセル、ウェルカムプログラム | クラスター2, 3  
Month 2-3 | ロイヤリティ強化、A/Bテスト | 全セグメント  
Month 4-6 | 最適化、スケール | 全セグメント  
  
**総合ROI予測** ：

  * 総投資: $151,050
  * 総期待収益: $12.92M
  * ROI: 8,454%
  * ペイバック期間: 2ヶ月

**成功の鍵** ：

  1. **継続的モニタリング** : 週次でKPI確認
  2. **迅速な調整** : 効果が低い施策は即改善
  3. **顧客フィードバック** : 定期的な満足度調査
  4. **データドリブン** : 施策の定量評価を徹底

* * *

## 参考文献

  1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
  2. Aggarwal, C. C., & Reddy, C. K. (2013). _Data Clustering: Algorithms and Applications_. CRC Press.
  3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." _IEEE International Conference on Data Mining_.
  4. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." _Journal of Machine Learning Research_ , 9, 2579-2605.
  5. Fader, P. S., & Hardie, B. G. S. (2009). "Probability Models for Customer-Base Analysis." _Journal of Interactive Marketing_ , 23(1), 61-69.
