---
title: 第2章：次元削減入門
chapter_title: 第2章：次元削減入門
subtitle: 高次元データの可視化と効率化 - PCAからt-SNE、UMAPまで
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 次元削減の必要性と「次元の呪い」を理解する
  * ✅ 主成分分析（PCA）の理論と実装ができる
  * ✅ 固有値・固有ベクトルの意味を説明できる
  * ✅ t-SNEによる可視化ができる
  * ✅ UMAPの特徴と使い方を理解する
  * ✅ PCA、t-SNE、UMAPを適切に使い分けられる
  * ✅ 次元削減後のデータで機械学習モデルを構築できる

* * *

## 2.1 次元削減とは

### 定義

**次元削減（Dimensionality Reduction）** は、高次元データを低次元空間に射影し、重要な情報を保持しながらデータの次元数を減らす技術です。

> 「$d$ 次元のデータ $\mathbf{X} \in \mathbb{R}^{n \times d}$ を $k$ 次元（$k \ll d$）に変換: $\mathbf{Z} \in \mathbb{R}^{n \times k}$」

### 次元削減の目的

目的 | 説明 | 応用例  
---|---|---  
**可視化** | 高次元データを2D/3Dで可視化 | 探索的データ分析、パターン発見  
**計算効率化** | 特徴量を削減し学習を高速化 | 機械学習の前処理  
**ノイズ除去** | ノイズを含む次元を削除 | 画像処理、信号処理  
**過学習防止** | 特徴量数を減らし汎化性能向上 | モデル学習  
  
### 次元の呪い（Curse of Dimensionality）

次元が増加すると、以下の問題が発生します：

  1. **データの疎性** : データ点間の距離が大きくなり、データが空間に散らばる
  2. **計算量の増加** : 次元 $d$ に対して計算量が $O(d^2)$ や $O(d^3)$ で増加
  3. **過学習のリスク** : パラメータ数に対してサンプル数が不足

    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 次元の呪いの可視化
    np.random.seed(42)
    dimensions = [1, 2, 5, 10, 20, 50, 100]
    n_samples = 1000
    
    # 各次元でのユークリッド距離の平均・分散
    mean_distances = []
    std_distances = []
    
    for d in dimensions:
        # ランダムにサンプル生成
        data = np.random.randn(n_samples, d)
    
        # 原点からの距離を計算
        distances = np.linalg.norm(data, axis=1)
    
        mean_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, mean_distances, 'o-', linewidth=2, markersize=8)
    plt.xlabel('次元数', fontsize=12)
    plt.ylabel('原点からの平均距離', fontsize=12)
    plt.title('次元数と距離の関係', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dimensions, np.array(std_distances) / np.array(mean_distances),
             's-', linewidth=2, markersize=8, color='red')
    plt.xlabel('次元数', fontsize=12)
    plt.ylabel('変動係数 (std/mean)', fontsize=12)
    plt.title('距離の相対的なばらつき', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 次元の呪い ===")
    print(f"1次元: 平均距離 = {mean_distances[0]:.2f}")
    print(f"100次元: 平均距離 = {mean_distances[-1]:.2f}")
    print(f"距離の増加率: {mean_distances[-1] / mean_distances[0]:.2f}倍")
    

### 次元削減手法の分類
    
    
    ```mermaid
    graph TD
        A[次元削減手法] --> B[線形手法]
        A --> C[非線形手法]
    
        B --> B1[PCA 主成分分析]
        B --> B2[LDA 線形判別分析]
        B --> B3[因子分析]
    
        C --> C1[t-SNE]
        C --> C2[UMAP]
        C --> C3[オートエンコーダ]
        C --> C4[Isomap]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

* * *

## 2.2 主成分分析（PCA）

### 概要

**PCA（Principal Component Analysis）** は、データの分散が最大になる方向を見つけ、その方向に射影する線形次元削減手法です。

### 数学的定義

**目的** : データの分散を最大化する直交基底を見つける

データ行列 $\mathbf{X} \in \mathbb{R}^{n \times d}$ に対して：

  1. **中心化** : $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}$
  2. **共分散行列** : $\mathbf{C} = \frac{1}{n-1} \mathbf{X}_c^T \mathbf{X}_c$
  3. **固有値分解** : $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$

第 $k$ 主成分への射影：

$$ \mathbf{Z} = \mathbf{X}_c \mathbf{V}_k $$

ここで $\mathbf{V}_k$ は上位 $k$ 個の固有ベクトル（主成分）

### 固有値と分散

  * **固有値 $\lambda_i$** : 第 $i$ 主成分方向の分散
  * **固有ベクトル $\mathbf{v}_i$** : 第 $i$ 主成分の方向
  * **寄与率** : $\frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$（その主成分が説明する分散の割合）

### PCAの実装（NumPyスクラッチ）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    
    class PCA_Scratch:
        def __init__(self, n_components):
            self.n_components = n_components
            self.components_ = None
            self.mean_ = None
            self.explained_variance_ = None
    
        def fit(self, X):
            # 中心化
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
    
            # 共分散行列
            cov_matrix = np.cov(X_centered.T)
    
            # 固有値・固有ベクトルを計算
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
            # 固有値の降順にソート
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            # 上位k個の主成分を保存
            self.components_ = eigenvectors[:, :self.n_components]
            self.explained_variance_ = eigenvalues[:self.n_components]
    
            return self
    
        def transform(self, X):
            # データを中心化して射影
            X_centered = X - self.mean_
            return np.dot(X_centered, self.components_)
    
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
    
        def inverse_transform(self, Z):
            # 低次元から元の次元に復元
            return np.dot(Z, self.components_.T) + self.mean_
    
    # データ読み込み
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # PCAの適用
    pca = PCA_Scratch(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 結果の表示
    print("=== PCA（スクラッチ実装） ===")
    print(f"元の次元: {X.shape[1]}")
    print(f"削減後の次元: {X_pca.shape[1]}")
    print(f"\n主成分の形状: {pca.components_.shape}")
    print(f"説明された分散: {pca.explained_variance_}")
    print(f"寄与率: {pca.explained_variance_ / np.sum(pca.explained_variance_)}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    target_names = iris.target_names
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color, alpha=0.7, lw=2, label=target_name, s=80)
    
    plt.xlabel('第1主成分', fontsize=12)
    plt.ylabel('第2主成分', fontsize=12)
    plt.title('PCA: Irisデータセット', fontsize=14)
    plt.legend(loc='best', shadow=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    === PCA（スクラッチ実装） ===
    元の次元: 4
    削減後の次元: 2
    
    主成分の形状: (4, 2)
    説明された分散: [4.22824171 0.24267075]
    寄与率: [0.94565341 0.05434659]
    

### scikit-learnによるPCA
    
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # データの標準化（推奨）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCAの適用
    pca_sklearn = PCA(n_components=2)
    X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)
    
    print("\n=== PCA（scikit-learn） ===")
    print(f"寄与率: {pca_sklearn.explained_variance_ratio_}")
    print(f"累積寄与率: {np.cumsum(pca_sklearn.explained_variance_ratio_)}")
    
    # 全主成分の寄与率
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # 寄与率
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue')
    plt.xlabel('主成分番号', fontsize=12)
    plt.ylabel('寄与率', fontsize=12)
    plt.title('各主成分の寄与率', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # 累積寄与率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_),
             'o-', linewidth=2, markersize=8, color='red')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95%ライン')
    plt.xlabel('主成分数', fontsize=12)
    plt.ylabel('累積寄与率', fontsize=12)
    plt.title('累積寄与率', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === PCA（scikit-learn） ===
    寄与率: [0.72962445 0.22850762]
    累積寄与率: [0.72962445 0.95813207]
    

### PCAによる画像圧縮
    
    
    from sklearn.datasets import fetch_olivetti_faces
    
    # 顔画像データセット
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X_faces = faces.data  # (400, 4096) - 64x64ピクセル
    
    print(f"画像データ形状: {X_faces.shape}")
    print(f"元の次元: {X_faces.shape[1]} (64x64ピクセル)")
    
    # 異なる主成分数でPCAを適用
    n_components_list = [10, 50, 100, 200]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    
    # 元の画像
    axes[0, 0].imshow(X_faces[0].reshape(64, 64), cmap='gray')
    axes[0, 0].set_title('元画像', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(X_faces[1].reshape(64, 64), cmap='gray')
    axes[1, 0].set_title('元画像', fontsize=12)
    axes[1, 0].axis('off')
    
    for idx, n_comp in enumerate(n_components_list, 1):
        pca = PCA(n_components=n_comp)
        X_compressed = pca.fit_transform(X_faces)
        X_reconstructed = pca.inverse_transform(X_compressed)
    
        cumulative_var = np.sum(pca.explained_variance_ratio_)
    
        axes[0, idx].imshow(X_reconstructed[0].reshape(64, 64), cmap='gray')
        axes[0, idx].set_title(f'{n_comp}成分\n({cumulative_var:.1%})', fontsize=10)
        axes[0, idx].axis('off')
    
        axes[1, idx].imshow(X_reconstructed[1].reshape(64, 64), cmap='gray')
        axes[1, idx].set_title(f'{n_comp}成分\n({cumulative_var:.1%})', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 圧縮率の計算
    original_size = X_faces.shape[1]
    for n_comp in n_components_list:
        compressed_size = n_comp
        compression_ratio = (1 - compressed_size / original_size) * 100
        print(f"{n_comp}主成分: 圧縮率 {compression_ratio:.1f}%")
    

* * *

## 2.3 t-SNE（t-Distributed Stochastic Neighbor Embedding）

### 概要

**t-SNE** は、高次元データの局所的な構造を保持しながら低次元（通常2D/3D）に可視化する非線形次元削減手法です。

### アルゴリズムの直感

  1. **高次元空間** : データ点間の類似度を確率分布で表現（ガウス分布）
  2. **低次元空間** : データ点間の類似度をt分布で表現
  3. **最適化** : 2つの分布の差（KLダイバージェンス）を最小化

### 数式

**高次元での条件付き確率** ：

$$ p_{j|i} = \frac{\exp(-||\mathbf{x}_i - \mathbf{x}_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||\mathbf{x}_i - \mathbf{x}_k||^2 / 2\sigma_i^2)} $$

**低次元での確率** （t分布を使用）：

$$ q_{ij} = \frac{(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||\mathbf{y}_k - \mathbf{y}_l||^2)^{-1}} $$

**目的関数** （KLダイバージェンス）：

$$ KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$

### 主要なパラメータ: Perplexity

**Perplexity** は、各データ点の近傍点数の目安を決める重要なパラメータです。

  * 推奨範囲: 5〜50（データセットサイズに依存）
  * 小さい値: 局所構造を強調
  * 大きい値: 大域構造を保持

### 実装例: MNISTデータセット
    
    
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    
    # 手書き数字データセット（MNISTの小規模版）
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    print(f"データ形状: {X_digits.shape}")
    print(f"クラス数: {len(np.unique(y_digits))}")
    
    # t-SNEの適用
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_digits)
    
    # 可視化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=y_digits, cmap='tab10',
                         alpha=0.7, s=30, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='数字')
    plt.xlabel('t-SNE 第1成分', fontsize=12)
    plt.ylabel('t-SNE 第2成分', fontsize=12)
    plt.title('t-SNE: 手書き数字データセット', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### Perplexityの影響
    
    
    # 異なるPerplexityでの比較
    perplexities = [5, 30, 50, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_tsne_temp = tsne.fit_transform(X_digits)
    
        scatter = axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                                   c=y_digits, cmap='tab10', alpha=0.7, s=20)
        axes[idx].set_title(f'Perplexity = {perp}', fontsize=14)
        axes[idx].set_xlabel('第1成分')
        axes[idx].set_ylabel('第2成分')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### t-SNEの制約と注意点

制約 | 説明  
---|---  
**計算コスト** | $O(n^2)$ の計算量。大規模データには不向き  
**非決定論的** | ランダム初期化により実行ごとに結果が異なる  
**大域構造** | クラスタ間の距離は意味を持たない  
**新データ対応不可** | 学習後に新しいデータを射影できない  
  
> **重要** : t-SNEは可視化専用。次元削減したデータで機械学習モデルを学習するにはPCAを使用すること。

* * *

## 2.4 UMAP（Uniform Manifold Approximation and Projection）

### 概要

**UMAP** は、t-SNEの欠点を改善した高速な非線形次元削減手法です。多様体学習の理論に基づいています。

### UMAPの利点

特徴 | UMAP | t-SNE  
---|---|---  
**計算速度** | 高速（数分） | 遅い（数十分〜数時間）  
**大域構造** | 保持 | 保持されない  
**新データ対応** | 可能（transform） | 不可能  
**スケーラビリティ** | 100万点以上 | 数千〜数万点  
  
### 主要なパラメータ

  1. **n_neighbors** : 局所近傍のサイズ（デフォルト: 15） 
     * 小さい値: 局所構造を強調
     * 大きい値: 大域構造を保持
  2. **min_dist** : 低次元空間での点間の最小距離（デフォルト: 0.1） 
     * 小さい値: 密なクラスタ
     * 大きい値: 疎なクラスタ

### 実装例
    
    
    import umap
    import matplotlib.pyplot as plt
    
    # UMAPの適用
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(X_digits)
    
    # 可視化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                         c=y_digits, cmap='tab10',
                         alpha=0.7, s=30, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='数字')
    plt.xlabel('UMAP 第1成分', fontsize=12)
    plt.ylabel('UMAP 第2成分', fontsize=12)
    plt.title('UMAP: 手書き数字データセット', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### パラメータの影響
    
    
    # 異なるパラメータでの比較
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    params = [
        {'n_neighbors': 5, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.1},
        {'n_neighbors': 50, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.01},
        {'n_neighbors': 15, 'min_dist': 0.5},
        {'n_neighbors': 15, 'min_dist': 0.99},
    ]
    
    for idx, (ax, param) in enumerate(zip(axes.ravel(), params)):
        umap_temp = umap.UMAP(**param, random_state=42)
        X_umap_temp = umap_temp.fit_transform(X_digits)
    
        scatter = ax.scatter(X_umap_temp[:, 0], X_umap_temp[:, 1],
                            c=y_digits, cmap='tab10', alpha=0.7, s=15)
        ax.set_title(f"n_neighbors={param['n_neighbors']}, min_dist={param['min_dist']}",
                    fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 新データへの適用
    
    
    # 学習データと新データ
    X_train = X_digits[:1500]
    X_test = X_digits[1500:]
    y_train = y_digits[:1500]
    y_test = y_digits[1500:]
    
    # UMAPの学習
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_model.fit(X_train)
    
    # 学習データとテストデータの変換
    X_train_umap = umap_model.transform(X_train)
    X_test_umap = umap_model.transform(X_test)
    
    # 可視化
    plt.figure(figsize=(12, 10))
    plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
               c=y_train, cmap='tab10', alpha=0.5, s=20, label='訓練データ')
    plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
               c=y_test, cmap='tab10', alpha=0.9, s=50,
               marker='s', edgecolors='black', linewidth=1.5, label='新データ')
    plt.xlabel('UMAP 第1成分', fontsize=12)
    plt.ylabel('UMAP 第2成分', fontsize=12)
    plt.title('UMAP: 新データへの適用', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== UMAP ===")
    print(f"訓練データ形状: {X_train_umap.shape}")
    print(f"テストデータ形状: {X_test_umap.shape}")
    

### 3D可視化
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # 3次元UMAPの適用
    umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap_3d = umap_3d.fit_transform(X_digits)
    
    # 3D可視化
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2],
                        c=y_digits, cmap='tab10', alpha=0.6, s=30)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_zlabel('UMAP 3', fontsize=12)
    ax.set_title('UMAP: 3次元可視化', fontsize=14)
    
    plt.colorbar(scatter, label='数字', shrink=0.7)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.5 手法の比較と使い分け

### PCA vs t-SNE vs UMAP
    
    
    from sklearn.datasets import load_digits
    import time
    
    # データ準備
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # 各手法の実行と計測
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PCA
    start_time = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_time = time.time() - start_time
    
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
    axes[0].set_title(f'PCA\n実行時間: {pca_time:.3f}秒', fontsize=14)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    tsne_time = time.time() - start_time
    
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
    axes[1].set_title(f't-SNE\n実行時間: {tsne_time:.3f}秒', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)
    
    # UMAP
    start_time = time.time()
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(X)
    umap_time = time.time() - start_time
    
    scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7, s=30)
    axes[2].set_title(f'UMAP\n実行時間: {umap_time:.3f}秒', fontsize=14)
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes, label='数字', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()
    
    # 比較表
    print("\n=== 実行時間の比較 ===")
    print(f"PCA:   {pca_time:.3f}秒 (1.0x)")
    print(f"t-SNE: {tsne_time:.3f}秒 ({tsne_time/pca_time:.1f}x)")
    print(f"UMAP:  {umap_time:.3f}秒 ({umap_time/pca_time:.1f}x)")
    

### 使い分けのガイドライン

目的 | 推奨手法 | 理由  
---|---|---  
**機械学習の前処理** | PCA | 線形、高速、逆変換可能  
**データの可視化** | t-SNE or UMAP | 非線形、局所構造の保持  
**大規模データ** | UMAP | 高速、スケーラブル  
**クラスタ分析** | UMAP | 大域構造の保持  
**ノイズ除去** | PCA | 分散の小さい次元を削除  
**新データへの適用** | PCA or UMAP | transformメソッド対応  
      
    
    ```mermaid
    graph TD
        A[次元削減の選択] --> B{目的は?}
        B -->|機械学習の前処理| C[PCA]
        B -->|可視化| D{データサイズは?}
        B -->|ノイズ除去| C
    
        D -->|小〜中規模| E{局所 or 大域?}
        D -->|大規模| F[UMAP]
    
        E -->|局所構造重視| G[t-SNE]
        E -->|大域構造重視| F
    
        style A fill:#e3f2fd
        style C fill:#c8e6c9
        style F fill:#fff9c4
        style G fill:#ffccbc
    ```

* * *

## 2.6 実践: 次元削減後の機械学習

### PCAによる特徴量削減とモデル学習
    
    
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    import time
    
    # データ準備
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"元のデータ形状: {X_train.shape}")
    
    # 1. 元のデータで学習
    start_time = time.time()
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    original_time = time.time() - start_time
    
    y_pred_original = rf_original.predict(X_test)
    original_acc = accuracy_score(y_test, y_pred_original)
    
    print("\n=== 元のデータ（64次元） ===")
    print(f"訓練時間: {original_time:.3f}秒")
    print(f"精度: {original_acc:.4f}")
    
    # 2. PCAで次元削減後に学習
    n_components_list = [10, 20, 30, 40]
    results = []
    
    for n_comp in n_components_list:
        # PCA適用
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    
        # モデル学習
        start_time = time.time()
        rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pca.fit(X_train_pca, y_train)
        pca_time = time.time() - start_time
    
        # 評価
        y_pred_pca = rf_pca.predict(X_test_pca)
        pca_acc = accuracy_score(y_test, y_pred_pca)
    
        cumulative_var = np.sum(pca.explained_variance_ratio_)
    
        results.append({
            'n_components': n_comp,
            'accuracy': pca_acc,
            'time': pca_time,
            'variance': cumulative_var
        })
    
        print(f"\n=== PCA（{n_comp}次元） ===")
        print(f"累積寄与率: {cumulative_var:.4f}")
        print(f"訓練時間: {pca_time:.3f}秒 ({pca_time/original_time:.2f}x)")
        print(f"精度: {pca_acc:.4f} ({(pca_acc - original_acc):.4f})")
    
    # 結果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精度の比較
    axes[0].axhline(y=original_acc, color='red', linestyle='--',
                    linewidth=2, label=f'元データ ({original_acc:.4f})')
    axes[0].plot([r['n_components'] for r in results],
                [r['accuracy'] for r in results],
                'o-', linewidth=2, markersize=10, label='PCA後')
    axes[0].set_xlabel('主成分数', fontsize=12)
    axes[0].set_ylabel('精度', fontsize=12)
    axes[0].set_title('主成分数と精度の関係', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 訓練時間の比較
    axes[1].axhline(y=original_time, color='red', linestyle='--',
                    linewidth=2, label=f'元データ ({original_time:.3f}秒)')
    axes[1].plot([r['n_components'] for r in results],
                [r['time'] for r in results],
                's-', linewidth=2, markersize=10, color='green', label='PCA後')
    axes[1].set_xlabel('主成分数', fontsize=12)
    axes[1].set_ylabel('訓練時間（秒）', fontsize=12)
    axes[1].set_title('主成分数と訓練時間の関係', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 2.7 本章のまとめ

### 学んだこと

  1. **次元削減の必要性**

     * 次元の呪い: 高次元データの問題点
     * 可視化、計算効率化、ノイズ除去、過学習防止
  2. **PCA（主成分分析）**

     * 線形次元削減手法
     * 分散を最大化する方向に射影
     * 固有値・固有ベクトルによる実装
     * 寄与率による次元数の選択
  3. **t-SNE**

     * 非線形次元削減手法
     * 局所構造の保持に優れる
     * Perplexityパラメータの重要性
     * 可視化専用（機械学習の前処理には不向き）
  4. **UMAP**

     * t-SNEより高速
     * 大域構造と局所構造の両方を保持
     * 新データへの適用が可能
     * 大規模データに対応
  5. **手法の使い分け**

     * 機械学習の前処理: PCA
     * 可視化: t-SNE or UMAP
     * 大規模データ: UMAP

### 次の章へ

第3章では、**クラスタリング手法** を学びます：

  * k-meansクラスタリング
  * 階層的クラスタリング
  * DBSCANとその他の密度ベース手法
  * クラスタ評価指標

* * *

## 演習問題

### 問題1（難易度：easy）

次元削減が必要な理由を、「次元の呪い」の観点から3つ説明してください。

解答例

**解答** ：

  1. **データの疎性**

     * 次元が増えると、データ点間の距離が大きくなる
     * データが空間に散らばり、密度が低下
     * k-NNなどの距離ベース手法が機能しにくくなる
  2. **計算量の増加**

     * 次元 $d$ に対して計算量が $O(d^2)$ や $O(d^3)$ で増加
     * メモリ使用量も増大
     * 学習時間が現実的でなくなる
  3. **過学習のリスク**

     * パラメータ数（特徴量数）に対してサンプル数が不足
     * 訓練データに過度に適合し、汎化性能が低下
     * 必要なサンプル数が次元に対して指数的に増加

### 問題2（難易度：medium）

PCAの第1主成分と第2主成分は直交（内積が0）することを説明してください。

解答例

**解答** ：

PCAは共分散行列 $\mathbf{C}$ の固有値分解により主成分を求めます：

$$ \mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T $$

ここで $\mathbf{V}$ は固有ベクトル行列、$\mathbf{\Lambda}$ は固有値の対角行列です。

**固有ベクトルの直交性** ：

  * 実対称行列（共分散行列）の異なる固有値に対応する固有ベクトルは直交する
  * 数学的に: $\mathbf{v}_i^T \mathbf{v}_j = 0$ （$i \neq j$）

**意味** ：

  * 各主成分は互いに独立な方向を表す
  * 情報の冗長性がない
  * 直交基底により、元のデータを一意に表現可能

**検証コード** ：
    
    
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # 主成分ベクトル
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    
    # 内積を計算
    dot_product = np.dot(pc1, pc2)
    print(f"内積: {dot_product:.10f}")  # ≈ 0
    

### 問題3（難易度：medium）

t-SNEで可視化したデータを使って機械学習モデルを学習するのが不適切な理由を説明してください。

解答例

**不適切な理由** ：

  1. **大域構造が保存されない**

     * t-SNEは局所的な類似性のみを保持
     * クラスタ間の距離は意味を持たない
     * 全体的なデータ構造が歪む
  2. **非決定論的**

     * ランダム初期化により実行ごとに結果が異なる
     * 再現性がない
     * モデルの性能評価が不安定
  3. **新データへの適用不可**

     * transformメソッドがない
     * テストデータや新規データを同じ空間に射影できない
     * 実運用で使えない
  4. **計算コスト**

     * $O(n^2)$ の計算量
     * 大規模データでは現実的でない

**適切な手法** ：

  * **PCA** : 線形、高速、逆変換可能、transformメソッドあり
  * **UMAP** : 非線形だが新データへの適用が可能（ただし可視化が主目的）

### 問題4（難易度：hard）

PCAの累積寄与率が95%になる主成分数を求め、その次元数でデータを変換するコードを書いてください。Irisデータセットを使用してください。

解答例
    
    
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ読み込み
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 全主成分でPCAを実行
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # 寄与率
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("=== 各主成分の寄与率 ===")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio,
                                            cumulative_variance_ratio)):
        print(f"PC{i+1}: 寄与率 {var:.4f}, 累積寄与率 {cum_var:.4f}")
    
    # 累積寄与率95%以上となる主成分数
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\n累積寄与率95%に必要な主成分数: {n_components_95}")
    
    # その主成分数でPCAを再実行
    pca = PCA(n_components=n_components_95)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\n元の次元: {X.shape[1]}")
    print(f"削減後の次元: {X_pca.shape[1]}")
    print(f"実際の累積寄与率: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio, alpha=0.7, color='steelblue')
    plt.xlabel('主成分番号')
    plt.ylabel('寄与率')
    plt.title('各主成分の寄与率')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95%')
    plt.axvline(x=n_components_95, color='green', linestyle='--',
                linewidth=2, label=f'{n_components_95}主成分')
    plt.xlabel('主成分数')
    plt.ylabel('累積寄与率')
    plt.title('累積寄与率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 各主成分の寄与率 ===
    PC1: 寄与率 0.7296, 累積寄与率 0.7296
    PC2: 寄与率 0.2285, 累積寄与率 0.9581
    PC3: 寄与率 0.0367, 累積寄与率 0.9948
    PC4: 寄与率 0.0052, 累積寄与率 1.0000
    
    累積寄与率95%に必要な主成分数: 2
    
    元の次元: 4
    削減後の次元: 2
    実際の累積寄与率: 0.9581
    

### 問題5（難易度：hard）

PCA、t-SNE、UMAPを同じデータセットに適用し、可視化結果を比較してください。MNISTまたはFashion-MNISTデータセットを使用してください。

解答例
    
    
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import matplotlib.pyplot as plt
    import time
    
    # Fashion-MNISTデータセット（またはMNIST）
    print("データ読み込み中...")
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
    X = fashion_mnist.data.to_numpy()
    y = fashion_mnist.target.astype(int).to_numpy()
    
    # サンプリング（計算時間短縮のため）
    n_samples = 5000
    np.random.seed(42)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    print(f"データ形状: {X_sample.shape}")
    
    # クラス名
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 1. PCA
    print("\nPCA実行中...")
    start_time = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    pca_time = time.time() - start_time
    print(f"PCA完了: {pca_time:.2f}秒")
    
    # 2. t-SNE
    print("t-SNE実行中...")
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    tsne_time = time.time() - start_time
    print(f"t-SNE完了: {tsne_time:.2f}秒")
    
    # 3. UMAP
    print("UMAP実行中...")
    start_time = time.time()
    umap_model = umap.UMAP(n_components=2, n_neighbors=15,
                           min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(X_sample)
    umap_time = time.time() - start_time
    print(f"UMAP完了: {umap_time:.2f}秒")
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # PCA
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=y_sample, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title(f'PCA\n実行時間: {pca_time:.2f}秒\n'
                     f'寄与率: {np.sum(pca.explained_variance_ratio_):.2%}',
                     fontsize=14)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=y_sample, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title(f't-SNE\n実行時間: {tsne_time:.2f}秒',
                     fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)
    
    # UMAP
    scatter3 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1],
                              c=y_sample, cmap='tab10', alpha=0.6, s=10)
    axes[2].set_title(f'UMAP\n実行時間: {umap_time:.2f}秒',
                     fontsize=14)
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.colorbar(scatter3, ax=axes, label='クラス',
                fraction=0.02, pad=0.04, ticks=range(10))
    
    plt.tight_layout()
    plt.show()
    
    # 各手法の特徴
    print("\n=== 比較結果 ===")
    print(f"PCA:   クラスタの分離は弱いが、高速")
    print(f"t-SNE: クラスタが明確に分離、計算時間が長い")
    print(f"UMAP:  クラスタ分離と計算速度のバランスが良い")
    

**観察ポイント** ：

  * **PCA** : 線形なのでクラスタの重なりが多い。高速。
  * **t-SNE** : クラスタが明確に分離。局所構造が強調される。
  * **UMAP** : t-SNEに近い品質で、より高速。大域構造も保持。

* * *

## 参考文献

  1. Jolliffe, I. T. (2002). _Principal Component Analysis_. Springer.
  2. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. _Journal of Machine Learning Research_ , 9, 2579-2605.
  3. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. _arXiv:1802.03426_.
  4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
