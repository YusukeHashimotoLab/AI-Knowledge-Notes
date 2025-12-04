---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25分
difficulty: 初級
code_examples: 0
exercises: 0
---

# 第2章：次元削減手法による材料空間のマッピング

## 概要

高次元の材料特性データを2次元・3次元に射影することで、材料間の類似性や構造を視覚的に把握できます。本章では、主成分分析（PCA）、t-SNE、UMAPなどの次元削減手法を材料データに適用し、効果的な材料空間マッピングを実現します。

### 学習目標

  * PCA、t-SNE、UMAPの原理と特徴を理解する
  * 各手法を材料データに適用し、結果を比較できる
  * 次元削減結果の品質を評価できる
  * インタラクティブな可視化を実装できる

## 2.1 主成分分析（PCA）

PCAは、データの分散を最大化する方向に新しい軸（主成分）を設定する線形次元削減手法です。材料特性間の相関構造を保持しながら次元を削減できます。

### 2.1.1 PCAの基本実装

### コード例1: PCAによる次元削減
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 材料データの読み込み（第1章で作成したデータ）
    materials_data = pd.read_csv('materials_properties.csv')
    
    # 特性列の抽出
    feature_cols = ['band_gap', 'formation_energy', 'density',
                    'bulk_modulus', 'shear_modulus', 'melting_point']
    X = materials_data[feature_cols].values
    
    # 標準化（PCAは特性のスケールに敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCAの実行（2次元に削減）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 結果をDataFrameに格納
    materials_data['PC1'] = X_pca[:, 0]
    materials_data['PC2'] = X_pca[:, 1]
    
    # 主成分の寄与率
    explained_variance = pca.explained_variance_ratio_
    print("主成分分析の結果:")
    print(f"PC1の寄与率: {explained_variance[0]:.3f}")
    print(f"PC2の寄与率: {explained_variance[1]:.3f}")
    print(f"累積寄与率: {sum(explained_variance):.3f}")
    
    # 主成分の成分（各特性の重み）
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_cols
    )
    print("\n主成分の成分（各特性の寄与）:")
    print(components_df.round(3))
    

**出力例** :
    
    
    主成分分析の結果:
    PC1の寄与率: 0.342
    PC2の寄与率: 0.234
    累積寄与率: 0.576
    
    主成分の成分（各特性の寄与）:
                           PC1     PC2
    band_gap            -0.245   0.512
    formation_energy     0.387  -0.298
    density              0.456   0.321
    bulk_modulus         0.498   0.145
    shear_modulus        0.445   0.087
    melting_point        0.312   -0.687
    

### コード例2: PCA結果の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # PCAスコアプロット
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 安定性でカテゴリ分け
    colors = materials_data['formation_energy'].apply(
        lambda x: 'green' if x < -1.0 else 'orange' if x < 0 else 'red'
    )
    
    scatter = ax.scatter(materials_data['PC1'],
                         materials_data['PC2'],
                         c=colors,
                         s=50,
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=0.5)
    
    # 軸ラベル（寄与率を含む）
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
    ax.set_title('PCA: Materials Space Visualization',
                 fontsize=16, fontweight='bold')
    
    # グリッド
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Stable (E < -1 eV)'),
        Patch(facecolor='orange', edgecolor='black', label='Metastable (-1 < E < 0 eV)'),
        Patch(facecolor='red', edgecolor='black', label='Unstable (E > 0 eV)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('pca_materials_space.png', dpi=300, bbox_inches='tight')
    print("PCAスコアプロットを pca_materials_space.png に保存しました")
    plt.show()
    

### コード例3: PCA寄与率のスクリープロット
    
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # 全主成分を計算
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # スクリープロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左: 各主成分の寄与率
    n_components = len(pca_full.explained_variance_ratio_)
    ax1.bar(range(1, n_components + 1),
            pca_full.explained_variance_ratio_,
            alpha=0.7,
            edgecolor='black',
            color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
    ax1.set_title('Scree Plot: Individual Variance', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右: 累積寄与率
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    ax2.plot(range(1, n_components + 1),
             cumsum_variance,
             marker='o',
             linewidth=2,
             markersize=8,
             color='darkred')
    ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2,
                label='95% variance threshold', alpha=0.7)
    ax2.set_xlabel('Number of Components', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=300, bbox_inches='tight')
    print(f"スクリープロットを pca_scree_plot.png に保存しました")
    print(f"\n95%の分散を説明するために必要な主成分数: {np.argmax(cumsum_variance >= 0.95) + 1}")
    plt.show()
    

### コード例4: PCAローディングプロット（バイプロット）
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # バイプロット
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # スコアプロット（サンプル）
    ax.scatter(materials_data['PC1'],
               materials_data['PC2'],
               alpha=0.3,
               s=20,
               color='lightblue',
               edgecolors='none',
               label='Materials')
    
    # ローディングベクトル（変数）
    scale_factor = 3.0  # ベクトルのスケーリング
    for i, feature in enumerate(feature_cols):
        ax.arrow(0, 0,
                 pca.components_[0, i] * scale_factor,
                 pca.components_[1, i] * scale_factor,
                 head_width=0.15,
                 head_length=0.15,
                 fc='red',
                 ec='darkred',
                 linewidth=2,
                 alpha=0.8)
    
        # ラベル
        ax.text(pca.components_[0, i] * scale_factor * 1.15,
                pca.components_[1, i] * scale_factor * 1.15,
                feature.replace('_', ' ').title(),
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
    ax.set_title('PCA Biplot: Materials and Features',
                 fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
    print("PCAバイプロットを pca_biplot.png に保存しました")
    plt.show()
    

## 2.2 t-SNE（t-Distributed Stochastic Neighbor Embedding）

t-SNEは、高次元データの局所的な構造（近傍関係）を保持しながら低次元に射影する非線形次元削減手法です。クラスタ構造の可視化に優れています。

### 2.2.1 t-SNEの基本実装

### コード例5: t-SNEによる次元削減
    
    
    from sklearn.manifold import TSNE
    import numpy as np
    import time
    
    # t-SNEの実行（複数のperplexityで実験）
    perplexities = [5, 30, 50]
    tsne_results = {}
    
    for perplexity in perplexities:
        print(f"\nt-SNE (perplexity={perplexity}) を実行中...")
        start_time = time.time()
    
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=1000,
                    random_state=42,
                    verbose=0)
    
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_results[perplexity] = X_tsne
    
        elapsed_time = time.time() - start_time
        print(f"完了 (所要時間: {elapsed_time:.2f}秒)")
        print(f"KL divergence: {tsne.kl_divergence_:.3f}")
    
    # 結果の保存（perplexity=30の場合）
    materials_data['tsne1'] = tsne_results[30][:, 0]
    materials_data['tsne2'] = tsne_results[30][:, 1]
    

**出力例** :
    
    
    t-SNE (perplexity=5) を実行中...
    完了 (所要時間: 3.45秒)
    KL divergence: 1.234
    
    t-SNE (perplexity=30) を実行中...
    完了 (所要時間: 3.67秒)
    KL divergence: 0.987
    
    t-SNE (perplexity=50) を実行中...
    完了 (所要時間: 3.89秒)
    KL divergence: 1.056
    

### コード例6: 異なるperplexityでの比較
    
    
    import matplotlib.pyplot as plt
    
    # 3つのperplexityでの結果を並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, perplexity in enumerate(perplexities):
        ax = axes[idx]
        X_tsne = tsne_results[perplexity]
    
        scatter = ax.scatter(X_tsne[:, 0],
                             X_tsne[:, 1],
                             c=materials_data['band_gap'],
                             cmap='viridis',
                             s=50,
                             alpha=0.6,
                             edgecolors='black',
                             linewidth=0.5)
    
        ax.set_title(f't-SNE (perplexity={perplexity})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Band Gap (eV)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tsne_perplexity_comparison.png', dpi=300, bbox_inches='tight')
    print("t-SNE perplexity比較を tsne_perplexity_comparison.png に保存しました")
    plt.show()
    

### コード例7: t-SNEクラスタリング結果の可視化
    
    
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # t-SNE結果に対してクラスタリング
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tsne_results[30])
    
    materials_data['cluster'] = cluster_labels
    
    # クラスタごとの可視化
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(tsne_results[30][mask, 0],
                   tsne_results[30][mask, 1],
                   c=[colors[cluster_id]],
                   label=f'Cluster {cluster_id}',
                   s=60,
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=0.5)
    
    # クラスタ中心
    centers_tsne = kmeans.cluster_centers_
    ax.scatter(centers_tsne[:, 0],
               centers_tsne[:, 1],
               c='red',
               marker='X',
               s=300,
               edgecolors='black',
               linewidth=2,
               label='Cluster Centers',
               zorder=10)
    
    ax.set_xlabel('t-SNE 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=14, fontweight='bold')
    ax.set_title(f't-SNE with K-Means Clustering (k={n_clusters})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('tsne_clustering.png', dpi=300, bbox_inches='tight')
    print(f"t-SNEクラスタリング結果を tsne_clustering.png に保存しました")
    plt.show()
    
    # 各クラスタの特性平均値
    print("\nクラスタごとの特性平均値:")
    cluster_stats = materials_data.groupby('cluster')[feature_cols].mean()
    print(cluster_stats.round(2))
    

## 2.3 UMAP（Uniform Manifold Approximation and Projection）

UMAPは、t-SNEよりも高速で、大域的構造も保持する最新の次元削減手法です。大規模データセットでも効率的に動作します。

### 2.3.1 UMAPのインストールと基本実装

### コード例8: UMAPによる次元削減
    
    
    # UMAPのインストール（初回のみ）
    # !pip install umap-learn
    
    import umap
    import numpy as np
    import time
    
    # UMAPの実行（複数のn_neighborsで実験）
    n_neighbors_list = [5, 15, 50]
    umap_results = {}
    
    for n_neighbors in n_neighbors_list:
        print(f"\nUMAP (n_neighbors={n_neighbors}) を実行中...")
        start_time = time.time()
    
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=0.1,
                            metric='euclidean',
                            random_state=42)
    
        X_umap = reducer.fit_transform(X_scaled)
        umap_results[n_neighbors] = X_umap
    
        elapsed_time = time.time() - start_time
        print(f"完了 (所要時間: {elapsed_time:.2f}秒)")
    
    # 結果の保存（n_neighbors=15の場合）
    materials_data['umap1'] = umap_results[15][:, 0]
    materials_data['umap2'] = umap_results[15][:, 1]
    
    print("\nUMAP実行完了。結果をDataFrameに保存しました。")
    

**出力例** :
    
    
    UMAP (n_neighbors=5) を実行中...
    完了 (所要時間: 1.23秒)
    
    UMAP (n_neighbors=15) を実行中...
    完了 (所要時間: 1.34秒)
    
    UMAP (n_neighbors=50) を実行中...
    完了 (所要時間: 1.45秒)
    
    UMAP実行完了。結果をDataFrameに保存しました。
    

### コード例9: 異なるn_neighborsでの比較
    
    
    import matplotlib.pyplot as plt
    
    # 3つのn_neighborsでの結果を並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, n_neighbors in enumerate(n_neighbors_list):
        ax = axes[idx]
        X_umap = umap_results[n_neighbors]
    
        scatter = ax.scatter(X_umap[:, 0],
                             X_umap[:, 1],
                             c=materials_data['formation_energy'],
                             cmap='RdYlGn_r',
                             s=50,
                             alpha=0.6,
                             edgecolors='black',
                             linewidth=0.5)
    
        ax.set_title(f'UMAP (n_neighbors={n_neighbors})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Formation Energy (eV/atom)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('umap_neighbors_comparison.png', dpi=300, bbox_inches='tight')
    print("UMAP n_neighbors比較を umap_neighbors_comparison.png に保存しました")
    plt.show()
    

### コード例10: UMAP密度マップ
    
    
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # UMAP結果の密度推定
    X_umap = umap_results[15]
    
    # KDE（カーネル密度推定）
    xy = np.vstack([X_umap[:, 0], X_umap[:, 1]])
    density = gaussian_kde(xy)(xy)
    
    # 密度でソート（高密度点を上に描画）
    idx = density.argsort()
    x, y, z = X_umap[idx, 0], X_umap[idx, 1], density[idx]
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 9))
    
    scatter = ax.scatter(x, y, c=z, cmap='hot', s=50, alpha=0.7,
                         edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title('UMAP: Materials Space Density Map',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Point Density', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('umap_density_map.png', dpi=300, bbox_inches='tight')
    print("UMAP密度マップを umap_density_map.png に保存しました")
    plt.show()
    

## 2.4 手法の比較

### コード例11: PCA vs t-SNE vs UMAPの並列比較
    
    
    import matplotlib.pyplot as plt
    
    # 3つの手法の結果を並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 共通のカラーマップ（バンドギャップで色付け）
    vmin = materials_data['band_gap'].min()
    vmax = materials_data['band_gap'].max()
    
    # PCA
    ax = axes[0]
    scatter = ax.scatter(materials_data['PC1'],
                         materials_data['PC2'],
                         c=materials_data['band_gap'],
                         cmap='viridis',
                         s=50,
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=0.5,
                         vmin=vmin,
                         vmax=vmax)
    ax.set_title('PCA', fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # t-SNE
    ax = axes[1]
    scatter = ax.scatter(materials_data['tsne1'],
                         materials_data['tsne2'],
                         c=materials_data['band_gap'],
                         cmap='viridis',
                         s=50,
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=0.5,
                         vmin=vmin,
                         vmax=vmax)
    ax.set_title('t-SNE', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # UMAP
    ax = axes[2]
    scatter = ax.scatter(materials_data['umap1'],
                         materials_data['umap2'],
                         c=materials_data['band_gap'],
                         cmap='viridis',
                         s=50,
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=0.5,
                         vmin=vmin,
                         vmax=vmax)
    ax.set_title('UMAP', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 共通カラーバー
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
    print("次元削減手法比較を dimensionality_reduction_comparison.png に保存しました")
    plt.show()
    

### コード例12: 近傍保存率の評価
    
    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    def calculate_neighborhood_preservation(X_high, X_low, k=10):
        """
        高次元空間と低次元空間での近傍保存率を計算
    
        Parameters:
        -----------
        X_high : array-like
            高次元空間のデータ
        X_low : array-like
            低次元空間のデータ
        k : int
            近傍の数
    
        Returns:
        --------
        preservation_rate : float
            近傍保存率（0-1）
        """
        # 高次元空間でのk近傍
        nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
        _, indices_high = nbrs_high.kneighbors(X_high)
    
        # 低次元空間でのk近傍
        nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
        _, indices_low = nbrs_low.kneighbors(X_low)
    
        # 近傍保存率の計算
        preservation_scores = []
        for i in range(len(X_high)):
            # 自分自身を除く
            neighbors_high = set(indices_high[i, 1:])
            neighbors_low = set(indices_low[i, 1:])
    
            # 共通する近傍の割合
            intersection = len(neighbors_high & neighbors_low)
            preservation_scores.append(intersection / k)
    
        return np.mean(preservation_scores)
    
    # 各手法の近傍保存率を評価
    k_values = [5, 10, 20, 50]
    results = {
        'PCA': [],
        't-SNE': [],
        'UMAP': []
    }
    
    for k in k_values:
        pca_preservation = calculate_neighborhood_preservation(
            X_scaled, X_pca, k=k
        )
        tsne_preservation = calculate_neighborhood_preservation(
            X_scaled, tsne_results[30], k=k
        )
        umap_preservation = calculate_neighborhood_preservation(
            X_scaled, umap_results[15], k=k
        )
    
        results['PCA'].append(pca_preservation)
        results['t-SNE'].append(tsne_preservation)
        results['UMAP'].append(umap_preservation)
    
        print(f"k={k}での近傍保存率:")
        print(f"  PCA:   {pca_preservation:.3f}")
        print(f"  t-SNE: {tsne_preservation:.3f}")
        print(f"  UMAP:  {umap_preservation:.3f}")
        print()
    
    # 結果のプロット
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for method, scores in results.items():
        ax.plot(k_values, scores, marker='o', linewidth=2,
                markersize=8, label=method)
    
    ax.set_xlabel('Number of Neighbors (k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Neighborhood Preservation Rate', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Neighborhood Preservation',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('neighborhood_preservation.png', dpi=300, bbox_inches='tight')
    print("近傍保存率比較を neighborhood_preservation.png に保存しました")
    plt.show()
    

## 2.5 インタラクティブな可視化

### コード例13: Plotlyによる3D UMAP
    
    
    # Plotlyのインストール（初回のみ）
    # !pip install plotly
    
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    
    # 3次元UMAPの実行
    reducer_3d = umap.UMAP(n_components=3,
                           n_neighbors=15,
                           min_dist=0.1,
                           random_state=42)
    
    X_umap_3d = reducer_3d.fit_transform(X_scaled)
    
    materials_data['umap1_3d'] = X_umap_3d[:, 0]
    materials_data['umap2_3d'] = X_umap_3d[:, 1]
    materials_data['umap3_3d'] = X_umap_3d[:, 2]
    
    # インタラクティブ3Dプロット
    fig = px.scatter_3d(materials_data,
                        x='umap1_3d',
                        y='umap2_3d',
                        z='umap3_3d',
                        color='band_gap',
                        size='density',
                        hover_data=['formula', 'formation_energy', 'bulk_modulus'],
                        color_continuous_scale='Viridis',
                        title='Interactive 3D UMAP: Materials Space')
    
    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)",
                       gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)",
                       gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)",
                       gridcolor="white"),
        ),
        width=900,
        height=700,
        font=dict(size=12)
    )
    
    fig.write_html('umap_3d_interactive.html')
    print("インタラクティブ3D UMAPを umap_3d_interactive.html に保存しました")
    fig.show()
    

### コード例14: Bokehによるインタラクティブ散布図
    
    
    # Bokehのインストール（初回のみ）
    # !pip install bokeh
    
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256
    from bokeh.io import show
    
    # カラーマッパー
    color_mapper = LinearColorMapper(palette=Viridis256,
                                     low=materials_data['band_gap'].min(),
                                     high=materials_data['band_gap'].max())
    
    # プロットの作成
    output_file('umap_interactive.html')
    
    p = figure(width=900,
               height=700,
               title='Interactive UMAP: Materials Space',
               tools='pan,wheel_zoom,box_zoom,reset,save')
    
    # データソース
    source_data = dict(
        x=materials_data['umap1'],
        y=materials_data['umap2'],
        formula=materials_data['formula'],
        band_gap=materials_data['band_gap'],
        formation_energy=materials_data['formation_energy'],
        density=materials_data['density'],
        bulk_modulus=materials_data['bulk_modulus']
    )
    
    # 散布図
    circles = p.circle('x', 'y',
                       size=8,
                       source=source_data,
                       fill_color={'field': 'band_gap', 'transform': color_mapper},
                       fill_alpha=0.7,
                       line_color='black',
                       line_width=0.5)
    
    # ホバーツール
    hover = HoverTool(tooltips=[
        ('Formula', '@formula'),
        ('Band Gap', '@band_gap{0.00} eV'),
        ('Formation E', '@formation_energy{0.00} eV/atom'),
        ('Density', '@density{0.00} g/cm³'),
        ('Bulk Modulus', '@bulk_modulus{0.0} GPa')
    ])
    p.add_tools(hover)
    
    # カラーバー
    color_bar = ColorBar(color_mapper=color_mapper,
                         label_standoff=12,
                         title='Band Gap (eV)',
                         location=(0, 0))
    p.add_layout(color_bar, 'right')
    
    # 軸ラベル
    p.xaxis.axis_label = 'UMAP 1'
    p.yaxis.axis_label = 'UMAP 2'
    p.title.text_font_size = '16pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    
    save(p)
    print("インタラクティブUMAPを umap_interactive.html に保存しました")
    show(p)
    

### コード例15: アニメーションによる次元削減プロセスの可視化
    
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from sklearn.decomposition import PCA
    import numpy as np
    
    # 多段階PCAによるアニメーション
    n_frames = 20
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初期3D PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    def update(frame):
        ax.clear()
    
        # 回転角度
        angle = frame * (360 / n_frames)
        angle_rad = np.radians(angle)
    
        # 回転行列を適用
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    
        X_rotated = X_pca_3d @ rotation_matrix
    
        # 2D投影
        scatter = ax.scatter(X_rotated[:, 0],
                             X_rotated[:, 1],
                             c=materials_data['band_gap'],
                             cmap='viridis',
                             s=50,
                             alpha=0.6,
                             edgecolors='black',
                             linewidth=0.5)
    
        ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(f'3D PCA Rotation (angle={angle:.0f}°)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
        # 軸の範囲を固定
        ax.set_xlim(X_rotated[:, 0].min() - 1, X_rotated[:, 0].max() + 1)
        ax.set_ylim(X_rotated[:, 1].min() - 1, X_rotated[:, 1].max() + 1)
    
        return scatter,
    
    # アニメーション作成
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=200, blit=False)
    
    # GIFとして保存
    anim.save('pca_rotation_animation.gif', writer='pillow', fps=5)
    print("PCA回転アニメーションを pca_rotation_animation.gif に保存しました")
    plt.close()
    

## 2.6 まとめ

本章では、次元削減手法を用いた材料空間のマッピングについて学びました：

### 主要な次元削減手法

手法 | 特徴 | 利点 | 欠点  
---|---|---|---  
**PCA** | 線形、分散最大化 | 高速、解釈性高い | 非線形構造に弱い  
**t-SNE** | 非線形、近傍保存 | クラスタ可視化に優れる | 遅い、大域構造損失  
**UMAP** | 非線形、トポロジー保存 | 高速、大域・局所両立 | パラメータ調整必要  
  
### 実装したコード

コード例 | 内容 | 手法  
---|---|---  
例1-4 | PCA基本実装、可視化 | PCA  
例5-7 | t-SNE実装、パラメータ比較 | t-SNE  
例8-10 | UMAP実装、密度マップ | UMAP  
例11-12 | 手法比較、評価指標 | 比較  
例13-15 | インタラクティブ可視化 | Plotly, Bokeh  
  
### ベストプラクティス

  1. **前処理** : 標準化（StandardScaler）は必須
  2. **手法選択** : \- 解釈性重視 → PCA \- クラスタ発見 → t-SNE \- バランス重視 → UMAP
  3. **パラメータ調整** : 複数の設定で実験し、最適値を探索
  4. **評価** : 近傍保存率など定量的指標で品質評価

### 次章への展望

第3章では、Graph Neural Networks（GNN）を用いて材料の構造情報から特徴表現を学習し、より高度な材料空間マッピングを実現します。CGCNN、MEGNet、SchNetなどの最新GNNモデルを実装し、結晶構造を直接入力として次元削減を行います。

* * *

**前章** : [第1章：材料空間可視化の基礎](<chapter-1.html>)

**次章** : [第3章：GNNによる材料表現学習](<chapter-3.html>)

**シリーズトップ** : [材料特性マッピング入門](<index.html>)
