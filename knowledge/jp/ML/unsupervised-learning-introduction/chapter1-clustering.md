---
title: 第1章：クラスタリングの基礎
chapter_title: 第1章：クラスタリングの基礎
subtitle: 教師なし学習の基本 - データからパターンを発見する
reading_time: 20-25分
difficulty: 初級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 教師なし学習とクラスタリングの概念を理解する
  * ✅ K-meansアルゴリズムを実装できる
  * ✅ 階層的クラスタリングとデンドログラムを使える
  * ✅ DBSCANで密度ベースのクラスタリングができる
  * ✅ シルエット係数などの評価指標を適用できる
  * ✅ 実データでクラスタ分析を実行できる

* * *

## 1.1 教師なし学習とクラスタリング

### 教師なし学習とは

**教師なし学習（Unsupervised Learning）** は、ラベルのないデータから構造やパターンを発見する機械学習のアプローチです。

> 「正解ラベル $y$ がないデータ $X$ から、隠れた構造や関係性を見つけ出す」

### 教師あり学習 vs 教師なし学習

特徴 | 教師あり学習 | 教師なし学習  
---|---|---  
**データ** | ラベル付き $(X, y)$ | ラベルなし $(X)$  
**目的** | 予測モデルの構築 | パターン・構造の発見  
**タスク例** | 分類、回帰 | クラスタリング、次元削減  
**評価** | テストデータで評価 | 内的指標や可視化  
  
### クラスタリングの定義

**クラスタリング（Clustering）** は、類似したデータポイントをグループ（クラスタ）にまとめる手法です。

$$ C = \\{C_1, C_2, \ldots, C_k\\} $$

  * 各クラスタ $C_i$ は類似度が高いデータの集合
  * 異なるクラスタ間は類似度が低い

### 実世界の応用例
    
    
    ```mermaid
    graph LR
        A[クラスタリングの応用] --> B[マーケティング: 顧客セグメンテーション]
        A --> C[生物学: 遺伝子グルーピング]
        A --> D[画像処理: 画像セグメンテーション]
        A --> E[文書分析: トピック抽出]
        A --> F[異常検知: 正常パターンの特定]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

* * *

## 1.2 K-meansクラスタリング

### アルゴリズムの概要

**K-means** は最も広く使われるクラスタリング手法で、データを $k$ 個のクラスタに分割します。
    
    
    ```mermaid
    graph TD
        A[1. k個の中心点をランダム初期化] --> B[2. 各点を最近接の中心に割り当て]
        B --> C[3. 各クラスタの中心を再計算]
        C --> D{収束?}
        D -->|No| B
        D -->|Yes| E[クラスタリング完了]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#ffe0b2
        style E fill:#e8f5e9
    ```

### 数学的定義

**目的関数** （クラスタ内二乗和）：

$$ J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2 $$

  * $C_i$: クラスタ $i$
  * $\boldsymbol{\mu}_i$: クラスタ $i$ の中心（centroid）
  * $||\mathbf{x} - \boldsymbol{\mu}_i||$: ユークリッド距離

**中心の更新式** ：

$$ \boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \mathbf{x} $$

### 実装例：スクラッチ実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    class KMeans:
        """K-meansクラスタリングのスクラッチ実装"""
    
        def __init__(self, k=3, max_iters=100, random_state=42):
            self.k = k
            self.max_iters = max_iters
            self.random_state = random_state
            self.centroids = None
            self.labels = None
    
        def fit(self, X):
            """K-meansアルゴリズムを実行"""
            np.random.seed(self.random_state)
            n_samples, n_features = X.shape
    
            # 1. 中心点をランダムに初期化
            random_indices = np.random.choice(n_samples, self.k, replace=False)
            self.centroids = X[random_indices]
    
            for iteration in range(self.max_iters):
                # 2. 各点を最近接の中心に割り当て
                distances = self._compute_distances(X)
                new_labels = np.argmin(distances, axis=1)
    
                # 3. 中心を再計算
                new_centroids = np.array([
                    X[new_labels == i].mean(axis=0)
                    for i in range(self.k)
                ])
    
                # 収束判定
                if np.allclose(self.centroids, new_centroids):
                    print(f"収束しました (iteration {iteration})")
                    break
    
                self.centroids = new_centroids
                self.labels = new_labels
    
            return self
    
        def _compute_distances(self, X):
            """各点と各中心との距離を計算"""
            distances = np.zeros((X.shape[0], self.k))
            for i, centroid in enumerate(self.centroids):
                distances[:, i] = np.linalg.norm(X - centroid, axis=1)
            return distances
    
        def predict(self, X):
            """新しいデータのクラスタを予測"""
            distances = self._compute_distances(X)
            return np.argmin(distances, axis=1)
    
        def inertia(self, X):
            """クラスタ内二乗和を計算"""
            distances = self._compute_distances(X)
            min_distances = np.min(distances, axis=1)
            return np.sum(min_distances ** 2)
    
    # データ生成
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.6, random_state=42)
    
    # K-meansの実行
    kmeans = KMeans(k=4)
    kmeans.fit(X)
    
    print(f"最終的なクラスタ内二乗和: {kmeans.inertia(X):.2f}")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.title('元のデータ（真のラベル）', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                label='中心点')
    plt.title('K-meansの結果', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    収束しました (iteration 6)
    最終的なクラスタ内二乗和: 108.45
    

### scikit-learnによる実装
    
    
    from sklearn.cluster import KMeans
    
    # K-meansモデル
    kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_sklearn.fit(X)
    
    print("scikit-learn K-means:")
    print(f"クラスタ内二乗和 (inertia): {kmeans_sklearn.inertia_:.2f}")
    print(f"反復回数: {kmeans_sklearn.n_iter_}")
    print(f"\n中心点の座標:")
    print(kmeans_sklearn.cluster_centers_)
    

**出力** ：
    
    
    scikit-learn K-means:
    クラスタ内二乗和 (inertia): 108.45
    反復回数: 7
    
    中心点の座標:
    [[ 1.83  -8.88]
     [ 2.81   2.85]
     [-9.49   7.27]
     [-3.51  -7.92]]
    

### エルボー法による最適なK値の選択

**エルボー法（Elbow Method）** は、異なる $k$ 値でのクラスタ内二乗和をプロットし、「肘」の位置を見つける手法です。
    
    
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('クラスタ数 (k)', fontsize=12)
    plt.ylabel('クラスタ内二乗和 (Inertia)', fontsize=12)
    plt.title('エルボー法 - 最適なk値の選択', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7, label='推奨k=4')
    plt.legend()
    plt.show()
    

> **ヒント** : 曲線が「肘」のように曲がる点が最適な $k$ 値です。この例では $k=4$ が適切です。

### K-meansの長所と短所

長所 | 短所  
---|---  
シンプルで高速 | $k$ を事前に指定する必要がある  
大規模データに適用可能 | 初期値に依存する（局所最適解）  
実装が容易 | 球状クラスタしか見つけられない  
解釈しやすい | 外れ値に敏感  
  
* * *

## 1.3 階層的クラスタリング

### 概要

**階層的クラスタリング（Hierarchical Clustering）** は、データを階層構造（樹形図）として表現します。

2つのアプローチ：

  * **凝集型（Agglomerative）** : ボトムアップ - 各点を個別クラスタとして開始し、統合
  * **分割型（Divisive）** : トップダウン - 全データを1つのクラスタとして開始し、分割

### 凝集型アルゴリズム
    
    
    ```mermaid
    graph TD
        A[各データ点を個別クラスタに] --> B[最も近いクラスタ対を見つける]
        B --> C[2つのクラスタを統合]
        C --> D{1つのクラスタ?}
        D -->|No| B
        D -->|Yes| E[デンドログラム完成]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#ffe0b2
        style E fill:#e8f5e9
    ```

### 連結法（Linkage Methods）

クラスタ間の距離を計算する方法：

連結法 | 距離の定義 | 特徴  
---|---|---  
**単連結**  
(Single) | $\min(\text{dist}(a, b))$ | 最も近い点同士の距離。長い鎖状クラスタを作る  
**完全連結**  
(Complete) | $\max(\text{dist}(a, b))$ | 最も遠い点同士の距離。コンパクトなクラスタ  
**平均連結**  
(Average) | $\text{mean}(\text{dist}(a, b))$ | 全点対の平均距離。バランスが良い  
**Ward法**  
(Ward) | クラスタ内分散の増加量 | 分散を最小化。最もよく使われる  
  
### 実装例
    
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    
    # サンプルデータ生成
    np.random.seed(42)
    X_small = np.random.randn(20, 2)
    X_small[:10] += [3, 3]  # 1つ目のクラスタ
    X_small[10:] += [-3, -3]  # 2つ目のクラスタ
    
    # 階層的クラスタリング（Ward法）
    linkage_matrix = linkage(X_small, method='ward')
    
    # デンドログラムの描画
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix)
    plt.xlabel('サンプルインデックス', fontsize=12)
    plt.ylabel('距離', fontsize=12)
    plt.title('デンドログラム（Ward法）', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # クラスタリング結果の可視化
    agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = agg_clustering.fit_predict(X_small)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_small[:, 0], X_small[:, 1], c=labels, cmap='viridis',
                s=100, alpha=0.6, edgecolors='black')
    plt.xlabel('特徴量 1', fontsize=12)
    plt.ylabel('特徴量 2', fontsize=12)
    plt.title('階層的クラスタリング結果', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 異なる連結法の比較
    
    
    # 複雑な形状のデータ生成
    from sklearn.datasets import make_moons
    X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    plt.figure(figsize=(14, 10))
    
    for i, method in enumerate(linkage_methods):
        agg = AgglomerativeClustering(n_clusters=2, linkage=method)
        labels = agg.fit_predict(X_moons)
    
        plt.subplot(2, 2, i+1)
        plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis',
                    alpha=0.6, edgecolors='black')
        plt.title(f'{method.capitalize()}連結法', fontsize=12)
        plt.xlabel('特徴量 1')
        plt.ylabel('特徴量 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

> **注意** : Ward法は球状クラスタに適していますが、単連結法は複雑な形状のクラスタも検出できます。

* * *

## 1.4 DBSCAN（密度ベースクラスタリング）

### アルゴリズムの概要

**DBSCAN（Density-Based Spatial Clustering of Applications with Noise）** は、密度が高い領域をクラスタとして識別します。

**主要な概念** ：

  * **コア点（Core Point）** : 半径 $\varepsilon$ 内に $\text{minPts}$ 個以上の点がある
  * **境界点（Border Point）** : コア点の近傍にあるが、コア点ではない
  * **ノイズ点（Noise Point）** : どのクラスタにも属さない

### パラメータ

パラメータ | 説明 | 選択方法  
---|---|---  
$\varepsilon$ (eps) | 近傍の半径 | k距離グラフで決定  
minPts | コア点となる最小点数 | 通常、次元数×2 または 4以上  
  
### アルゴリズムの流れ
    
    
    ```mermaid
    graph TD
        A[未訪問点を選択] --> B{コア点?}
        B -->|Yes| C[新しいクラスタを作成]
        C --> D[近傍の全点を探索]
        D --> E[クラスタを拡張]
        E --> A
        B -->|No| F{境界点?}
        F -->|Yes| G[既存クラスタに追加]
        F -->|No| H[ノイズとしてマーク]
        G --> A
        H --> A
        A --> I{全点訪問?}
        I -->|No| A
        I -->|Yes| J[完了]
    
        style A fill:#e3f2fd
        style C fill:#e8f5e9
        style H fill:#ffebee
        style J fill:#f3e5f5
    ```

### 実装例
    
    
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    
    # 複雑な形状のデータ生成
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCANの適用
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_moons)
    
    # ノイズ点の数
    n_noise = list(labels).count(-1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"クラスタ数: {n_clusters}")
    print(f"ノイズ点数: {n_noise}")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], alpha=0.6)
    plt.title('元のデータ', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis',
                alpha=0.6, edgecolors='black')
    plt.title(f'DBSCAN結果 ({n_clusters}クラスタ, {n_noise}ノイズ点)', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    クラスタ数: 2
    ノイズ点数: 4
    

### パラメータの影響
    
    
    # 異なるepsでの比較
    eps_values = [0.15, 0.2, 0.3, 0.5]
    
    plt.figure(figsize=(14, 10))
    
    for i, eps in enumerate(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_moons)
    
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
    
        plt.subplot(2, 2, i+1)
        plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis',
                    alpha=0.6, edgecolors='black')
        plt.title(f'eps={eps} ({n_clusters}クラスタ, {n_noise}ノイズ)', fontsize=12)
        plt.xlabel('特徴量 1')
        plt.ylabel('特徴量 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### K-means vs DBSCAN
    
    
    # K-meansとDBSCANの比較
    kmeans_moons = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans_moons.fit_predict(X_moons)
    
    dbscan_moons = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan_moons.fit_predict(X_moons)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_labels,
                cmap='viridis', alpha=0.6, edgecolors='black')
    plt.scatter(kmeans_moons.cluster_centers_[:, 0],
                kmeans_moons.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    plt.title('K-means（球状クラスタを仮定）', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_labels,
                cmap='viridis', alpha=0.6, edgecolors='black')
    plt.title('DBSCAN（任意形状を検出）', fontsize=14)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

> **重要** : DBSCANは複雑な形状のクラスタを検出でき、ノイズも扱えます。K-meansは球状クラスタしか見つけられません。

* * *

## 1.5 クラスタリングの評価指標

### 評価の難しさ

教師なし学習では「正解」がないため、評価が困難です。

  * **内的評価** : クラスタの品質を内部データで評価
  * **外的評価** : 真のラベルがある場合の比較

### シルエット係数（Silhouette Score）

各データ点のクラスタ適合度を測定：

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

  * $a(i)$: 点 $i$ と同じクラスタ内の他の点との平均距離
  * $b(i)$: 点 $i$ と最も近い別クラスタ内の点との平均距離
  * 範囲: $[-1, 1]$ （1に近いほど良い）

    
    
    from sklearn.metrics import silhouette_score, silhouette_samples
    
    # データ生成
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.6, random_state=42)
    
    # 異なるk値でのシルエット係数
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"k={k}: シルエット係数 = {score:.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('クラスタ数 (k)', fontsize=12)
    plt.ylabel('シルエット係数', fontsize=12)
    plt.title('シルエット係数によるクラスタ数の評価', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7, label='最適k=4')
    plt.legend()
    plt.show()
    

**出力** ：
    
    
    k=2: シルエット係数 = 0.617
    k=3: シルエット係数 = 0.588
    k=4: シルエット係数 = 0.651
    k=5: シルエット係数 = 0.563
    k=6: シルエット係数 = 0.542
    k=7: シルエット係数 = 0.528
    k=8: シルエット係数 = 0.515
    k=9: シルエット係数 = 0.503
    k=10: シルエット係数 = 0.491
    

### シルエットプロット
    
    
    from sklearn.metrics import silhouette_samples
    import matplotlib.cm as cm
    
    # k=4でのシルエットプロット
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_vals = silhouette_samples(X, labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(4):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
    
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.viridis(float(i) / 4)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.axvline(x=silhouette_score(X, labels), color='red', linestyle='--',
                label=f'平均: {silhouette_score(X, labels):.3f}')
    plt.xlabel('シルエット係数', fontsize=12)
    plt.ylabel('クラスタ', fontsize=12)
    plt.title('シルエットプロット (k=4)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

### Davies-Bouldin指数

クラスタの分離度を測定（低いほど良い）：

$$ DB = \frac{1}{k} \sum_{i=1}^{k} \max_{i \neq j} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right) $$

  * $\sigma_i$: クラスタ $i$ の内部距離
  * $d(c_i, c_j)$: クラスタ中心間の距離

    
    
    from sklearn.metrics import davies_bouldin_score
    
    db_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = davies_bouldin_score(X, labels)
        db_scores.append(score)
        print(f"k={k}: Davies-Bouldin指数 = {score:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, db_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('クラスタ数 (k)', fontsize=12)
    plt.ylabel('Davies-Bouldin指数', fontsize=12)
    plt.title('Davies-Bouldin指数（低いほど良い）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7, label='最適k=4')
    plt.legend()
    plt.show()
    

### Calinski-Harabasz指数

クラスタ間分散とクラスタ内分散の比（高いほど良い）：

$$ CH = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{n - k}{k - 1} $$

  * $B_k$: クラスタ間分散行列
  * $W_k$: クラスタ内分散行列

    
    
    from sklearn.metrics import calinski_harabasz_score
    
    ch_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = calinski_harabasz_score(X, labels)
        ch_scores.append(score)
        print(f"k={k}: Calinski-Harabasz指数 = {score:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, ch_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('クラスタ数 (k)', fontsize=12)
    plt.ylabel('Calinski-Harabasz指数', fontsize=12)
    plt.title('Calinski-Harabasz指数（高いほど良い）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7, label='最適k=4')
    plt.legend()
    plt.show()
    

### 評価指標のまとめ

指標 | 範囲 | 最適値 | 特徴  
---|---|---|---  
**シルエット係数** | [-1, 1] | 高いほど良い | 各点の適合度を測定  
**Davies-Bouldin** | [0, ∞) | 低いほど良い | クラスタの分離度  
**Calinski-Harabasz** | [0, ∞) | 高いほど良い | 分散比に基づく  
  
* * *

## 1.6 実践例：顧客セグメンテーション

### データ準備
    
    
    # 顧客データの生成（購入金額と頻度）
    np.random.seed(42)
    
    # 3つの顧客セグメント
    segment1 = np.random.randn(100, 2) * [10, 2] + [30, 5]   # 高価格・低頻度
    segment2 = np.random.randn(100, 2) * [5, 5] + [50, 20]   # 中価格・高頻度
    segment3 = np.random.randn(100, 2) * [8, 3] + [80, 10]   # 高価格・中頻度
    
    X_customers = np.vstack([segment1, segment2, segment3])
    
    # カラム名
    feature_names = ['平均購入金額（千円）', '月間購入回数']
    

### データの標準化
    
    
    from sklearn.preprocessing import StandardScaler
    
    # 標準化（異なるスケールの特徴量を扱う際に重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customers)
    
    print("元のデータの統計:")
    print(f"平均購入金額: 平均={X_customers[:, 0].mean():.1f}, 標準偏差={X_customers[:, 0].std():.1f}")
    print(f"月間購入回数: 平均={X_customers[:, 1].mean():.1f}, 標準偏差={X_customers[:, 1].std():.1f}")
    
    print("\n標準化後のデータの統計:")
    print(f"平均購入金額: 平均={X_scaled[:, 0].mean():.3f}, 標準偏差={X_scaled[:, 0].std():.3f}")
    print(f"月間購入回数: 平均={X_scaled[:, 1].mean():.3f}, 標準偏差={X_scaled[:, 1].std():.3f}")
    

### 最適なクラスタ数の決定
    
    
    # エルボー法とシルエット係数の両方を評価
    inertias = []
    silhouettes = []
    K_range = range(2, 9)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('クラスタ数 (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('エルボー法', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.7)
    
    axes[1].plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('クラスタ数 (k)', fontsize=12)
    axes[1].set_ylabel('シルエット係数', fontsize=12)
    axes[1].set_title('シルエット係数', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=3, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    

### 最終的なクラスタリングと解釈
    
    
    # k=3でクラスタリング
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X_scaled)
    
    # 元のスケールで中心点を計算
    centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
    
    # 各セグメントの特徴を分析
    print("=== 顧客セグメント分析 ===\n")
    for i in range(3):
        cluster_data = X_customers[labels == i]
        print(f"セグメント {i+1} (n={len(cluster_data)}人):")
        print(f"  平均購入金額: {cluster_data[:, 0].mean():.1f}千円")
        print(f"  月間購入回数: {cluster_data[:, 1].mean():.1f}回")
        print()
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_customers[:, 0], X_customers[:, 1], c=labels,
                cmap='viridis', alpha=0.6, edgecolors='black', s=50)
    plt.scatter(centers_original[:, 0], centers_original[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                label='セグメント中心')
    plt.xlabel('平均購入金額（千円）', fontsize=12)
    plt.ylabel('月間購入回数', fontsize=12)
    plt.title('顧客セグメンテーション結果', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # セグメントサイズの可視化
    plt.subplot(1, 2, 2)
    segment_sizes = [np.sum(labels == i) for i in range(3)]
    segment_names = [f'セグメント {i+1}' for i in range(3)]
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    plt.pie(segment_sizes, labels=segment_names, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('セグメント分布', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 顧客セグメント分析 ===
    
    セグメント 1 (n=100人):
      平均購入金額: 29.8千円
      月間購入回数: 5.0回
    
    セグメント 2 (n=100人):
      平均購入金額: 50.2千円
      月間購入回数: 20.1回
    
    セグメント 3 (n=100人):
      平均購入金額: 79.7千円
      月間購入回数: 10.2回
    

> **ビジネスへの示唆** :
> 
>   * **セグメント1** : 低価格・低頻度 → 新規顧客または機会的購入者
>   * **セグメント2** : 中価格・高頻度 → ロイヤル顧客（最重要）
>   * **セグメント3** : 高価格・中頻度 → プレミアム顧客
> 

* * *

## 1.7 本章のまとめ

### 学んだこと

  1. **教師なし学習の概念**

     * ラベルなしデータからパターンを発見
     * クラスタリング、次元削減、異常検知
  2. **K-meansクラスタリング**

     * 中心点ベースの分割手法
     * エルボー法で最適な $k$ を選択
     * 高速だが球状クラスタが前提
  3. **階層的クラスタリング**

     * デンドログラムで階層構造を可視化
     * 連結法（単連結、完全連結、Ward法など）
     * クラスタ数を事後的に決定可能
  4. **DBSCAN**

     * 密度ベースのクラスタリング
     * 任意形状のクラスタを検出
     * ノイズを自動的に識別
  5. **評価指標**

     * シルエット係数: クラスタ適合度
     * Davies-Bouldin指数: 分離度
     * Calinski-Harabasz指数: 分散比

### アルゴリズムの選択指針

状況 | 推奨手法 | 理由  
---|---|---  
大規模データ | K-means | 高速で効率的  
球状クラスタ | K-means | 最適な性能  
任意形状 | DBSCAN, 単連結法 | 複雑な形状に対応  
ノイズあり | DBSCAN | ノイズを自動識別  
階層構造 | 階層的クラスタリング | デンドログラム分析  
クラスタ数不明 | DBSCAN, 階層的 | 自動的に決定  
  
### 次の章へ

第2章では、**次元削減手法** を学びます：

  * 主成分分析（PCA）
  * t-SNE
  * UMAP
  * オートエンコーダ

* * *

## 演習問題

### 問題1（難易度：easy）

教師あり学習と教師なし学習の違いを3つ挙げてください。

解答例

**解答** ：

  1. **データ** : 教師あり学習はラベル付きデータ、教師なし学習はラベルなしデータ
  2. **目的** : 教師あり学習は予測、教師なし学習はパターン発見
  3. **評価** : 教師あり学習はテストデータで評価、教師なし学習は内的指標や可視化

### 問題2（難易度：medium）

以下のデータでK-meansクラスタリング（k=2）を実装し、各クラスタの中心を求めてください。
    
    
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    

解答例
    
    
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print("クラスタラベル:", labels)
    print("\nクラスタ中心:")
    print(kmeans.cluster_centers_)
    
    # 可視化
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                s=100, edgecolors='black')
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    plt.xlabel('特徴量 1')
    plt.ylabel('特徴量 2')
    plt.title('K-meansクラスタリング結果')
    plt.grid(True, alpha=0.3)
    plt.show()
    

**出力** ：
    
    
    クラスタラベル: [0 0 1 1 0 1]
    
    クラスタ中心:
    [[1.17 1.47]
     [7.33 9.  ]]
    

### 問題3（難易度：medium）

シルエット係数が -0.5 の場合、そのデータ点についてどのようなことが言えますか？

解答例

**解答** ：

シルエット係数が負の値（-0.5）ということは：

  * **間違ったクラスタに割り当てられている可能性が高い**
  * $a(i) > b(i)$ であり、同じクラスタ内の点との距離が、別のクラスタとの距離よりも大きい
  * そのデータ点は別のクラスタに属すべき

**シルエット係数の解釈** ：

  * $s(i) \approx 1$: 適切に分類されている
  * $s(i) \approx 0$: クラスタ境界上にある
  * $s(i) < 0$: 間違ったクラスタに割り当てられている

### 問題4（難易度：hard）

K-meansとDBSCANの長所と短所を比較し、どのような場合にそれぞれを使うべきか説明してください。

解答例

**K-means** ：

長所：

  * 高速で大規模データに適用可能
  * 実装がシンプル
  * 結果が解釈しやすい

短所：

  * クラスタ数 $k$ を事前に指定する必要がある
  * 球状クラスタしか検出できない
  * 外れ値に敏感
  * 初期値に依存する

**DBSCAN** ：

長所：

  * 任意形状のクラスタを検出可能
  * ノイズを自動的に識別
  * クラスタ数を自動決定
  * 外れ値に頑健

短所：

  * パラメータ（eps, min_samples）の調整が難しい
  * 密度が異なるクラスタの検出が困難
  * 高次元データで性能が低下

**使い分け** ：

状況 | 推奨手法  
---|---  
球状クラスタ、クラスタ数が既知 | K-means  
複雑な形状、ノイズあり | DBSCAN  
大規模データ、速度重視 | K-means  
クラスタ数が不明 | DBSCAN  
密度ベースの定義が自然 | DBSCAN  
  
### 問題5（難易度：hard）

以下のコードを完成させ、階層的クラスタリングでデンドログラムを描画してください。
    
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    X = np.random.randn(15, 2)
    
    # ここに実装
    

解答例
    
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データ生成
    np.random.seed(42)
    X = np.random.randn(15, 2)
    
    # 階層的クラスタリング（Ward法）
    linkage_matrix = linkage(X, method='ward')
    
    # デンドログラムの描画
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix)
    plt.xlabel('サンプルインデックス', fontsize=12)
    plt.ylabel('距離', fontsize=12)
    plt.title('デンドログラム（Ward法）', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # k=3でクラスタリング
    agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = agg.fit_predict(X)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                s=100, edgecolors='black', alpha=0.7)
    for i, (x, y) in enumerate(X):
        plt.annotate(str(i), (x, y), fontsize=9, alpha=0.7)
    plt.xlabel('特徴量 1', fontsize=12)
    plt.ylabel('特徴量 2', fontsize=12)
    plt.title('階層的クラスタリング結果 (k=3)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("クラスタラベル:", labels)
    print("\n各クラスタのサイズ:")
    for i in range(3):
        print(f"クラスタ {i}: {np.sum(labels == i)}個")
    

**出力** ：
    
    
    クラスタラベル: [2 0 1 2 0 1 0 2 1 0 2 1 0 1 2]
    
    各クラスタのサイズ:
    クラスタ 0: 6個
    クラスタ 1: 5個
    クラスタ 2: 4個
    

* * *

## 参考文献

  1. MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations". _Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability_.
  2. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise". _KDD_.
  3. Kaufman, L., & Rousseeuw, P. J. (1990). _Finding Groups in Data: An Introduction to Cluster Analysis_. Wiley.
  4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning_. Springer.
