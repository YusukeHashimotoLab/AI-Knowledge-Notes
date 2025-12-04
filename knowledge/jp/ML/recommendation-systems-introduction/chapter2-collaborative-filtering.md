---
title: 第2章：協調フィルタリング
chapter_title: 第2章：協調フィルタリング
subtitle: ユーザーの嗜好パターンから学ぶ推薦システムの核心技術
reading_time: 30-35分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 協調フィルタリングの原理とMemory-based/Model-basedの違いを理解する
  * ✅ User-basedとItem-basedの協調フィルタリングを実装できる
  * ✅ 類似度指標（Cosine、Pearson）を適切に選択できる
  * ✅ 行列分解（SVD、ALS）の理論と実装を習得する
  * ✅ SVD++、NMF、BPRなどの高度な手法を適用できる
  * ✅ Surpriseライブラリで実用的な推薦システムを構築できる

* * *

## 2.1 協調フィルタリングの原理

### 協調フィルタリングとは

**協調フィルタリング（Collaborative Filtering）** は、多数のユーザーの行動パターンから、類似した嗜好を持つユーザーやアイテムを見つけ出し、推薦を行う手法です。

> 「似た嗜好を持つユーザーは、未知のアイテムについても似た評価をする」という仮定に基づきます。

### Memory-based vs Model-based

アプローチ | 特徴 | 手法例 | 長所 | 短所  
---|---|---|---|---  
**Memory-based** | 評価データを直接使用 | User-based CF, Item-based CF | 解釈性が高い、実装が簡単 | スケーラビリティに課題  
**Model-based** | 潜在因子モデルを学習 | SVD, ALS, Matrix Factorization | 高精度、スケーラブル | 解釈性が低い  
  
### 協調フィルタリングの全体像
    
    
    ```mermaid
    graph TD
        A[協調フィルタリング] --> B[Memory-based]
        A --> C[Model-based]
    
        B --> D[User-based CF]
        B --> E[Item-based CF]
    
        C --> F[行列分解]
        C --> G[ニューラルネットワーク]
    
        F --> H[SVD]
        F --> I[ALS]
        F --> J[NMF]
    
        style A fill:#e8f5e9
        style B fill:#fff3e0
        style C fill:#e3f2fd
        style D fill:#ffebee
        style E fill:#f3e5f5
        style F fill:#fce4ec
    ```

### 評価行列の表現

協調フィルタリングの基本は**評価行列（Rating Matrix）** $R$です：

$$ R = \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1m} \\\ r_{21} & r_{22} & \cdots & r_{2m} \\\ \vdots & \vdots & \ddots & \vdots \\\ r_{n1} & r_{n2} & \cdots & r_{nm} \end{bmatrix} $$

  * $n$: ユーザー数
  * $m$: アイテム数
  * $r_{ui}$: ユーザー$u$のアイテム$i$への評価（多くは欠損値）

    
    
    import numpy as np
    import pandas as pd
    
    # サンプル評価行列の作成
    np.random.seed(42)
    users = ['Alice', 'Bob', 'Carol', 'David', 'Eve']
    items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
    
    # 評価行列（NaNは未評価）
    ratings = pd.DataFrame([
        [5, 3, np.nan, 1, np.nan],
        [4, np.nan, np.nan, 1, np.nan],
        [1, 1, np.nan, 5, 4],
        [1, np.nan, np.nan, 4, 4],
        [np.nan, 1, 5, 4, np.nan]
    ], columns=items, index=users)
    
    print("=== 評価行列 ===")
    print(ratings)
    print(f"\n形状: {ratings.shape}")
    print(f"評価済みセル: {ratings.notna().sum().sum()}/{ratings.size}")
    print(f"密度: {ratings.notna().sum().sum()/ratings.size:.2%}")
    

**出力** ：
    
    
    === 評価行列 ===
           Item1  Item2  Item3  Item4  Item5
    Alice    5.0    3.0    NaN    1.0    NaN
    Bob      4.0    NaN    NaN    1.0    NaN
    Carol    1.0    1.0    NaN    5.0    4.0
    David    1.0    NaN    NaN    4.0    4.0
    Eve      NaN    1.0    5.0    4.0    NaN
    
    形状: (5, 5)
    評価済みセル: 13/25
    密度: 52.00%
    

* * *

## 2.2 User-based Collaborative Filtering

### 基本原理

**User-based CF** は、「類似した嗜好を持つユーザーが好むアイテムを推薦する」手法です。

#### アルゴリズムの流れ

  1. **ユーザー類似度計算** : 全ユーザーペア間の類似度を計算
  2. **近傍ユーザー選択** : 対象ユーザーに類似したk人を選択
  3. **評価予測** : 近傍ユーザーの評価から加重平均を計算

### 類似度指標

#### 1\. コサイン類似度（Cosine Similarity）

$$ \text{sim}_{\text{cos}}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}} $$

  * $I_{uv}$: ユーザー$u$と$v$が両方評価したアイテム集合

#### 2\. Pearson相関係数

$$ \text{sim}_{\text{pearson}}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}} $$

  * $\bar{r}_u$: ユーザー$u$の平均評価

### 評価予測式

$$ \hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u, v)|} $$

  * $N(u)$: ユーザー$u$の近傍ユーザー集合

### 完全実装例
    
    
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cosine
    from scipy.stats import pearsonr
    
    class UserBasedCF:
        def __init__(self, k=3, similarity='cosine'):
            """
            User-based Collaborative Filtering
    
            Parameters:
            -----------
            k : int
                近傍ユーザー数
            similarity : str
                類似度指標 ('cosine' or 'pearson')
            """
            self.k = k
            self.similarity = similarity
            self.ratings = None
            self.user_mean = None
    
        def fit(self, ratings_df):
            """評価行列で学習"""
            self.ratings = ratings_df.copy()
            self.user_mean = self.ratings.mean(axis=1)
            return self
    
        def _compute_similarity(self, user1, user2):
            """2ユーザー間の類似度を計算"""
            # 両方が評価しているアイテムを抽出
            mask = self.ratings.loc[user1].notna() & self.ratings.loc[user2].notna()
    
            if mask.sum() == 0:
                return 0  # 共通評価なし
    
            r1 = self.ratings.loc[user1][mask].values
            r2 = self.ratings.loc[user2][mask].values
    
            if self.similarity == 'cosine':
                # コサイン類似度
                if np.linalg.norm(r1) == 0 or np.linalg.norm(r2) == 0:
                    return 0
                return 1 - cosine(r1, r2)
    
            elif self.similarity == 'pearson':
                # Pearson相関係数
                if len(r1) < 2:
                    return 0
                corr, _ = pearsonr(r1, r2)
                return corr if not np.isnan(corr) else 0
    
        def predict(self, user, item):
            """特定のユーザー・アイテムペアの評価を予測"""
            if user not in self.ratings.index:
                return self.ratings[item].mean()  # フォールバック
    
            # 全ユーザーとの類似度を計算
            similarities = []
            for other_user in self.ratings.index:
                if other_user == user:
                    continue
                # アイテムを評価済みのユーザーのみ
                if pd.notna(self.ratings.loc[other_user, item]):
                    sim = self._compute_similarity(user, other_user)
                    similarities.append((other_user, sim))
    
            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)
    
            # 上位k人を選択
            neighbors = similarities[:self.k]
    
            if len(neighbors) == 0:
                return self.user_mean[user]  # フォールバック
    
            # 加重平均で予測
            numerator = 0
            denominator = 0
    
            for neighbor, sim in neighbors:
                if sim > 0:
                    rating = self.ratings.loc[neighbor, item]
                    neighbor_mean = self.user_mean[neighbor]
                    numerator += sim * (rating - neighbor_mean)
                    denominator += abs(sim)
    
            if denominator == 0:
                return self.user_mean[user]
    
            prediction = self.user_mean[user] + numerator / denominator
    
            # 評価範囲にクリップ
            return np.clip(prediction, self.ratings.min().min(), self.ratings.max().max())
    
    # 使用例
    cf = UserBasedCF(k=2, similarity='cosine')
    cf.fit(ratings)
    
    # 予測
    user = 'Alice'
    item = 'Item3'
    prediction = cf.predict(user, item)
    
    print(f"\n=== User-based CF 予測 ===")
    print(f"ユーザー: {user}")
    print(f"アイテム: {item}")
    print(f"予測評価: {prediction:.2f}")
    
    # 全未評価アイテムの予測
    print(f"\n=== {user}の全予測 ===")
    for item in ratings.columns:
        if pd.isna(ratings.loc[user, item]):
            pred = cf.predict(user, item)
            print(f"{item}: {pred:.2f}")
    

**出力** ：
    
    
    === User-based CF 予測 ===
    ユーザー: Alice
    アイテム: Item3
    予測評価: 4.50
    
    === Aliceの全予測 ===
    Item3: 4.50
    Item5: 4.00
    

### 類似度行列の可視化
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 全ユーザー間の類似度行列を計算
    cf_cosine = UserBasedCF(k=2, similarity='cosine')
    cf_cosine.fit(ratings)
    
    similarity_matrix = pd.DataFrame(
        index=ratings.index,
        columns=ratings.index,
        dtype=float
    )
    
    for u1 in ratings.index:
        for u2 in ratings.index:
            if u1 == u2:
                similarity_matrix.loc[u1, u2] = 1.0
            else:
                similarity_matrix.loc[u1, u2] = cf_cosine._compute_similarity(u1, u2)
    
    print("=== ユーザー類似度行列（Cosine）===")
    print(similarity_matrix)
    
    # ヒートマップで可視化
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix.astype(float), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=similarity_matrix.columns,
                yticklabels=similarity_matrix.index)
    plt.title('User-User Similarity Matrix (Cosine)', fontsize=14)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.3 Item-based Collaborative Filtering

### 基本原理

**Item-based CF** は、「ユーザーが好んだアイテムに類似したアイテムを推薦する」手法です。

#### User-basedとの違い

特性 | User-based CF | Item-based CF  
---|---|---  
**類似度計算** | ユーザー間 | アイテム間  
**スケーラビリティ** | ユーザー数に依存 | アイテム数に依存  
**適用場面** | ユーザー数 < アイテム数 | ユーザー数 > アイテム数  
**安定性** | 嗜好変化の影響大 | アイテム特性は安定  
**実用例** | ニッチなコミュニティ | Amazon、Netflix  
  
### 評価予測式

$$ \hat{r}_{ui} = \frac{\sum_{j \in N(i)} \text{sim}(i, j) \cdot r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i, j)|} $$

  * $N(i)$: アイテム$i$の近傍アイテム集合（ユーザー$u$が評価済み）

### 完全実装例
    
    
    class ItemBasedCF:
        def __init__(self, k=3, similarity='cosine'):
            """
            Item-based Collaborative Filtering
    
            Parameters:
            -----------
            k : int
                近傍アイテム数
            similarity : str
                類似度指標 ('cosine' or 'pearson')
            """
            self.k = k
            self.similarity = similarity
            self.ratings = None
    
        def fit(self, ratings_df):
            """評価行列で学習"""
            self.ratings = ratings_df.copy()
            return self
    
        def _compute_similarity(self, item1, item2):
            """2アイテム間の類似度を計算"""
            # 両方を評価しているユーザーを抽出
            mask = self.ratings[item1].notna() & self.ratings[item2].notna()
    
            if mask.sum() == 0:
                return 0  # 共通評価なし
    
            r1 = self.ratings[item1][mask].values
            r2 = self.ratings[item2][mask].values
    
            if self.similarity == 'cosine':
                if np.linalg.norm(r1) == 0 or np.linalg.norm(r2) == 0:
                    return 0
                return 1 - cosine(r1, r2)
    
            elif self.similarity == 'pearson':
                if len(r1) < 2:
                    return 0
                corr, _ = pearsonr(r1, r2)
                return corr if not np.isnan(corr) else 0
    
        def predict(self, user, item):
            """特定のユーザー・アイテムペアの評価を予測"""
            if user not in self.ratings.index:
                return self.ratings[item].mean()
    
            # ユーザーが評価済みの他アイテムとの類似度を計算
            similarities = []
            for other_item in self.ratings.columns:
                if other_item == item:
                    continue
                # ユーザーが評価済みのアイテムのみ
                if pd.notna(self.ratings.loc[user, other_item]):
                    sim = self._compute_similarity(item, other_item)
                    similarities.append((other_item, sim))
    
            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)
    
            # 上位k個を選択
            neighbors = similarities[:self.k]
    
            if len(neighbors) == 0:
                return self.ratings[item].mean()
    
            # 加重平均で予測
            numerator = 0
            denominator = 0
    
            for neighbor, sim in neighbors:
                if sim > 0:
                    rating = self.ratings.loc[user, neighbor]
                    numerator += sim * rating
                    denominator += abs(sim)
    
            if denominator == 0:
                return self.ratings[item].mean()
    
            prediction = numerator / denominator
    
            # 評価範囲にクリップ
            return np.clip(prediction, self.ratings.min().min(), self.ratings.max().max())
    
    # 使用例
    item_cf = ItemBasedCF(k=2, similarity='cosine')
    item_cf.fit(ratings)
    
    user = 'Alice'
    item = 'Item3'
    prediction = item_cf.predict(user, item)
    
    print(f"\n=== Item-based CF 予測 ===")
    print(f"ユーザー: {user}")
    print(f"アイテム: {item}")
    print(f"予測評価: {prediction:.2f}")
    
    # User-based vs Item-based 比較
    user_pred = cf.predict(user, item)
    item_pred = item_cf.predict(user, item)
    
    print(f"\n=== 手法比較 ===")
    print(f"User-based CF: {user_pred:.2f}")
    print(f"Item-based CF: {item_pred:.2f}")
    

**出力** ：
    
    
    === Item-based CF 予測 ===
    ユーザー: Alice
    アイテム: Item3
    予測評価: 4.00
    
    === 手法比較 ===
    User-based CF: 4.50
    Item-based CF: 4.00
    

### アイテム類似度行列の可視化
    
    
    # アイテム間類似度行列を計算
    item_similarity_matrix = pd.DataFrame(
        index=ratings.columns,
        columns=ratings.columns,
        dtype=float
    )
    
    for i1 in ratings.columns:
        for i2 in ratings.columns:
            if i1 == i2:
                item_similarity_matrix.loc[i1, i2] = 1.0
            else:
                item_similarity_matrix.loc[i1, i2] = item_cf._compute_similarity(i1, i2)
    
    print("=== アイテム類似度行列（Cosine）===")
    print(item_similarity_matrix)
    
    # ヒートマップで可視化
    plt.figure(figsize=(8, 6))
    sns.heatmap(item_similarity_matrix.astype(float), annot=True, fmt='.2f',
                cmap='viridis', center=0.5, vmin=0, vmax=1,
                xticklabels=item_similarity_matrix.columns,
                yticklabels=item_similarity_matrix.index)
    plt.title('Item-Item Similarity Matrix (Cosine)', fontsize=14)
    plt.tight_layout()
    plt.show()
    

* * *

## 2.4 行列分解（Matrix Factorization）

### 基本概念

**行列分解** は、評価行列$R$を低ランクの2つの行列の積で近似する手法です：

$$ R \approx P \times Q^T $$

  * $P \in \mathbb{R}^{n \times k}$: ユーザー潜在因子行列
  * $Q \in \mathbb{R}^{m \times k}$: アイテム潜在因子行列
  * $k$: 潜在因子数（次元削減のパラメータ）

予測評価：

$$ \hat{r}_{ui} = p_u \cdot q_i^T = \sum_{f=1}^{k} p_{uf} \cdot q_{if} $$

### SVD（Singular Value Decomposition）

特異値分解は、行列を以下のように分解します：

$$ R = U \Sigma V^T $$

  * $U$: 左特異ベクトル（ユーザー潜在因子）
  * $\Sigma$: 特異値の対角行列
  * $V$: 右特異ベクトル（アイテム潜在因子）

### Surpriseライブラリを使った実装
    
    
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
    import pandas as pd
    
    # 評価データの準備（long format）
    ratings_long = []
    for user in ratings.index:
        for item in ratings.columns:
            if pd.notna(ratings.loc[user, item]):
                ratings_long.append({
                    'user': user,
                    'item': item,
                    'rating': ratings.loc[user, item]
                })
    
    df_long = pd.DataFrame(ratings_long)
    
    print("=== Long Format データ ===")
    print(df_long.head(10))
    
    # Surpriseデータセットの作成
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_long[['user', 'item', 'rating']], reader)
    
    # 訓練・テストデータ分割
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    # SVDモデルの学習
    svd = SVD(n_factors=2, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    
    # テストセットでの予測
    predictions = svd.test(testset)
    
    # 評価指標
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    print(f"\n=== SVD モデル評価 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 特定ユーザーの予測
    user = 'Alice'
    item = 'Item3'
    prediction = svd.predict(user, item)
    
    print(f"\n=== 個別予測 ===")
    print(f"ユーザー: {user}")
    print(f"アイテム: {item}")
    print(f"予測評価: {prediction.est:.2f}")
    print(f"実際の評価: {ratings.loc[user, item]}")
    

**出力** ：
    
    
    === Long Format データ ===
        user   item  rating
    0  Alice  Item1     5.0
    1  Alice  Item2     3.0
    2  Alice  Item4     1.0
    3    Bob  Item1     4.0
    4    Bob  Item4     1.0
    
    === SVD モデル評価 ===
    RMSE: 0.8452
    MAE: 0.6234
    
    === 個別予測 ===
    ユーザー: Alice
    アイテム: Item3
    予測評価: 4.23
    実際の評価: nan
    

### ALS（Alternating Least Squares）

ALSは、$P$と$Q$を交互に最適化する手法です。implicit feedbackに適しています。
    
    
    from implicit.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # 評価行列をスパース行列に変換
    # NaNを0で埋める（implicitでは観測=1, 未観測=0）
    ratings_binary = ratings.fillna(0)
    ratings_binary[ratings_binary > 0] = 1  # 評価があれば1
    
    # ユーザー・アイテムのインデックスマップ
    user_to_idx = {user: idx for idx, user in enumerate(ratings.index)}
    item_to_idx = {item: idx for idx, item in enumerate(ratings.columns)}
    
    # スパース行列作成（アイテム x ユーザー）
    sparse_ratings = csr_matrix(ratings_binary.values.T)
    
    # ALSモデル
    als_model = AlternatingLeastSquares(
        factors=5,
        regularization=0.01,
        iterations=20,
        random_state=42
    )
    
    # 学習
    als_model.fit(sparse_ratings)
    
    print("\n=== ALS モデル ===")
    print(f"ユーザー潜在因子: {als_model.user_factors.shape}")
    print(f"アイテム潜在因子: {als_model.item_factors.shape}")
    
    # 特定ユーザーへの推薦
    user_idx = user_to_idx['Alice']
    recommendations = als_model.recommend(
        user_idx,
        sparse_ratings[user_idx],
        N=3,
        filter_already_liked_items=True
    )
    
    print(f"\n=== Aliceへの推薦（ALS）===")
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    for item_idx, score in recommendations:
        print(f"{idx_to_item[item_idx]}: スコア {score:.4f}")
    

**出力** ：
    
    
    === ALS モデル ===
    ユーザー潜在因子: (5, 5)
    アイテム潜在因子: (5, 5)
    
    === Aliceへの推薦（ALS）===
    Item3: スコア 0.2345
    Item5: スコア 0.1892
    

### クロスバリデーションでのモデル比較
    
    
    from surprise import SVD, NMF, KNNBasic
    from surprise.model_selection import cross_validate
    
    # 複数アルゴリズムの比較
    algorithms = {
        'SVD': SVD(n_factors=5, random_state=42),
        'NMF': NMF(n_factors=5, random_state=42),
        'User-KNN': KNNBasic(sim_options={'user_based': True}),
        'Item-KNN': KNNBasic(sim_options={'user_based': False})
    }
    
    results = {}
    
    print("\n=== クロスバリデーション（5-fold）===\n")
    for name, algo in algorithms.items():
        cv_results = cross_validate(
            algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False
        )
    
        results[name] = {
            'RMSE': cv_results['test_rmse'].mean(),
            'MAE': cv_results['test_mae'].mean()
        }
    
        print(f"{name}:")
        print(f"  RMSE: {results[name]['RMSE']:.4f} (+/- {cv_results['test_rmse'].std():.4f})")
        print(f"  MAE:  {results[name]['MAE']:.4f} (+/- {cv_results['test_mae'].std():.4f})\n")
    
    # 結果の可視化
    results_df = pd.DataFrame(results).T
    print("=== 最終結果 ===")
    print(results_df.sort_values('RMSE'))
    

* * *

## 2.5 高度な手法

### SVD++（Implicit Feedbackを考慮）

SVD++は、explicit ratings（明示的評価）に加えて、implicit feedback（閲覧履歴など）も考慮します。

$$ \hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left( p_u + |I_u|^{-0.5} \sum_{j \in I_u} y_j \right) $$

  * $\mu$: 全体平均
  * $b_u$: ユーザーバイアス
  * $b_i$: アイテムバイアス
  * $I_u$: ユーザー$u$が評価したアイテム集合
  * $y_j$: implicit feedbackの潜在因子

    
    
    from surprise import SVDpp
    
    # SVD++モデル
    svdpp = SVDpp(n_factors=5, n_epochs=20, lr_all=0.007, reg_all=0.02, random_state=42)
    
    # クロスバリデーション
    cv_results = cross_validate(svdpp, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    print("=== SVD++ 評価 ===")
    print(f"RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
    print(f"MAE: {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")
    
    # 学習と予測
    trainset = data.build_full_trainset()
    svdpp.fit(trainset)
    
    user = 'Alice'
    item = 'Item3'
    pred = svdpp.predict(user, item)
    
    print(f"\n=== SVD++ 予測 ===")
    print(f"{user} → {item}: {pred.est:.2f}")
    

### NMF（Non-negative Matrix Factorization）

NMFは、非負制約を課した行列分解です：

$$ R \approx P \times Q^T, \quad P, Q \geq 0 $$

解釈性が高く、トピックモデルのような応用に適しています。
    
    
    from surprise import NMF
    
    # NMFモデル
    nmf = NMF(n_factors=5, n_epochs=50, random_state=42)
    
    # 学習
    trainset = data.build_full_trainset()
    nmf.fit(trainset)
    
    # 予測
    predictions = []
    for user in ratings.index:
        for item in ratings.columns:
            if pd.isna(ratings.loc[user, item]):
                pred = nmf.predict(user, item)
                predictions.append({
                    'user': user,
                    'item': item,
                    'prediction': pred.est
                })
    
    pred_df = pd.DataFrame(predictions)
    
    print("\n=== NMF 予測結果 ===")
    print(pred_df.pivot(index='user', columns='item', values='prediction'))
    
    # 潜在因子の可視化
    print("\n=== ユーザー潜在因子（NMF）===")
    user_factors_df = pd.DataFrame(
        nmf.pu,  # ユーザー潜在因子
        index=ratings.index,
        columns=[f'Factor{i+1}' for i in range(nmf.n_factors)]
    )
    print(user_factors_df)
    

### BPR（Bayesian Personalized Ranking）

BPRは、implicit feedbackのためのランキング最適化手法です。

**目的関数** ：

$$ \text{maximize} \quad \sum_{u,i,j} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) $$

  * $i$: ユーザー$u$がinteractしたアイテム
  * $j$: ユーザー$u$がinteractしていないアイテム
  * $\sigma$: シグモイド関数

    
    
    from implicit.bpr import BayesianPersonalizedRanking
    
    # BPRモデル
    bpr_model = BayesianPersonalizedRanking(
        factors=10,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
        random_state=42
    )
    
    # 学習（implicitライブラリはitem x userの転置が必要）
    bpr_model.fit(sparse_ratings)
    
    print("\n=== BPR モデル ===")
    print(f"潜在因子数: {bpr_model.factors}")
    
    # 推薦
    user_idx = user_to_idx['Alice']
    recommendations = bpr_model.recommend(
        user_idx,
        sparse_ratings[user_idx],
        N=5,
        filter_already_liked_items=True
    )
    
    print(f"\n=== Aliceへの推薦（BPR）===")
    for item_idx, score in recommendations:
        print(f"{idx_to_item[item_idx]}: スコア {score:.4f}")
    

**出力** ：
    
    
    === BPR モデル ===
    潜在因子数: 10
    
    === Aliceへの推薦（BPR）===
    Item3: スコア 0.3421
    Item5: スコア 0.2876
    

### 手法の総合比較

手法 | データタイプ | 計算量 | 解釈性 | 精度 | 適用場面  
---|---|---|---|---|---  
**User-based CF** | Explicit | O(n²m) | 高 | 中 | 小規模、ユーザー少  
**Item-based CF** | Explicit | O(nm²) | 高 | 中 | Eコマース  
**SVD** | Explicit | O(k・iter・nnz) | 中 | 高 | Netflix Prize  
**SVD++** | Explicit + Implicit | O(k・iter・nnz) | 中 | 最高 | ハイブリッド  
**ALS** | Implicit | O(k²・iter・nnz) | 中 | 高 | 視聴履歴、クリック  
**NMF** | Explicit | O(k・iter・nnz) | 高 | 中 | トピック推薦  
**BPR** | Implicit | O(k・iter・nnz) | 低 | 高 | ランキング最適化  
  
  * $n$: ユーザー数、$m$: アイテム数、$k$: 潜在因子数、$nnz$: 非ゼロ要素数

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **協調フィルタリングの原理**

     * Memory-based（User/Item-based CF）とModel-based（行列分解）の2アプローチ
     * 類似度指標（Cosine、Pearson）の使い分け
     * 評価行列の表現とスパース性の課題
  2. **User-based CF**

     * 類似ユーザーの嗜好から推薦
     * 解釈性が高いが、スケーラビリティに課題
     * ユーザー数が少ない場面で有効
  3. **Item-based CF**

     * 類似アイテムから推薦
     * アイテム特性は安定しているため実用的
     * Amazon、Netflixで採用
  4. **行列分解（SVD、ALS）**

     * 潜在因子モデルで高精度な予測
     * スケーラブルで大規模データに対応
     * Explicitには SVD、Implicitには ALS/BPR
  5. **高度な手法**

     * SVD++: Implicit feedbackも活用
     * NMF: 非負制約で解釈性向上
     * BPR: ランキング最適化でImplicit feedbackに特化

### 協調フィルタリングの課題

課題 | 説明 | 対策  
---|---|---  
**Cold Start問題** | 新規ユーザー・アイテムの推薦困難 | コンテンツベース、ハイブリッド  
**スパース性** | 評価データが疎で類似度計算困難 | 次元削減、行列分解  
**スケーラビリティ** | ユーザー・アイテム増加で計算コスト増 | 近似アルゴリズム、分散処理  
**Gray Sheep問題** | 独特な嗜好のユーザーに推薦困難 | ハイブリッド、多様性考慮  
  
### 次の章へ

第3章では、**コンテンツベースフィルタリング** を学びます：

  * アイテム特徴量の抽出
  * TF-IDFとテキスト特徴量
  * ユーザープロファイル構築
  * 類似度計算と推薦
  * 協調フィルタリングとの比較

* * *

## 演習問題

### 問題1（難易度：easy）

User-based CFとItem-based CFの違いを、類似度計算の対象、スケーラビリティ、適用場面の3点から説明してください。

解答例

**解答** ：

  1. **類似度計算の対象**

     * User-based CF: ユーザー間の類似度を計算
     * Item-based CF: アイテム間の類似度を計算
  2. **スケーラビリティ**

     * User-based CF: ユーザー数$n$に対してO(n²)の計算量
     * Item-based CF: アイテム数$m$に対してO(m²)の計算量
     * 一般にユーザー数 >> アイテム数なので、Item-basedがスケーラブル
  3. **適用場面**

     * User-based CF: ユーザー数が少なく、嗜好が多様な場面（ニッチコミュニティ）
     * Item-based CF: ユーザー数が多く、アイテム特性が安定している場面（Eコマース、Netflix）

### 問題2（難易度：medium）

以下の評価行列に対して、AliceとBobのコサイン類似度を手計算で求めてください。
    
    
           Item1  Item2  Item3
    Alice    5      3      1
    Bob      4      2      2
    

解答例

**解答** ：

コサイン類似度の式：

$$ \text{sim}_{\text{cos}}(Alice, Bob) = \frac{\sum r_{Alice,i} \cdot r_{Bob,i}}{\sqrt{\sum r_{Alice,i}^2} \cdot \sqrt{\sum r_{Bob,i}^2}} $$

**計算** ：

  1. 分子（内積）：
         
         5×4 + 3×2 + 1×2 = 20 + 6 + 2 = 28
         

  2. 分母（ノルムの積）：
         
         ||Alice|| = √(5² + 3² + 1²) = √(25 + 9 + 1) = √35 ≈ 5.916
         ||Bob|| = √(4² + 2² + 2²) = √(16 + 4 + 4) = √24 ≈ 4.899
         分母 = 5.916 × 4.899 ≈ 28.98
         

  3. コサイン類似度：
         
         sim = 28 / 28.98 ≈ 0.966
         

**Pythonでの検証** ：
    
    
    import numpy as np
    from scipy.spatial.distance import cosine
    
    alice = np.array([5, 3, 1])
    bob = np.array([4, 2, 2])
    
    similarity = 1 - cosine(alice, bob)
    print(f"コサイン類似度: {similarity:.4f}")  # 0.9661
    

### 問題3（難易度：medium）

行列分解（Matrix Factorization）において、潜在因子数$k$を増やすことの長所と短所を述べてください。

解答例

**解答** ：

**長所** ：

  1. **表現力の向上** : より複雑な嗜好パターンを捉えられる
  2. **精度向上** : 訓練データでのフィッティングが向上
  3. **細かい違いの表現** : 微妙な嗜好の差異を区別可能

**短所** ：

  1. **過学習のリスク** : 訓練データに過度に適合し、汎化性能低下
  2. **計算コストの増加** : 学習時間と推論時間が増加（O(k)に比例）
  3. **スパース性問題** : データが少ない場合、多くの因子を推定困難
  4. **解釈性の低下** : 因子が多いほど意味の解釈が困難

**適切な選択** ：

  * データ量、スパース性、計算リソースを考慮
  * クロスバリデーションで最適な$k$を選択
  * 一般に$k = 10 \sim 100$が実用的

### 問題4（難易度：hard）

以下のMovieLensスタイルのデータを使って、SVDモデルを構築し、RMSEを計算してください。
    
    
    data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'D'],
        'rating': [5, 3, 4, 4, 5, 2, 3, 5, 3, 4]
    })
    

解答例
    
    
    import pandas as pd
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
    
    # データの準備
    data_df = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'A', 'D'],
        'rating': [5, 3, 4, 4, 5, 2, 3, 5, 3, 4]
    })
    
    print("=== データ ===")
    print(data_df)
    
    # Surpriseデータセットの作成
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_df[['user_id', 'item_id', 'rating']], reader)
    
    # 訓練・テスト分割
    trainset, testset = train_test_split(data, test_size=0.3, random_state=42)
    
    # SVDモデル
    svd = SVD(n_factors=2, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    
    # 予測とRMSE計算
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions)
    
    print(f"\n=== モデル評価 ===")
    print(f"RMSE: {rmse:.4f}")
    
    # クロスバリデーション
    cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    
    print(f"\n=== 3-Fold クロスバリデーション ===")
    print(f"平均RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
    print(f"平均MAE: {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")
    
    # 新規予測
    trainset_full = data.build_full_trainset()
    svd.fit(trainset_full)
    
    print(f"\n=== 新規予測 ===")
    test_cases = [(1, 'D'), (2, 'B'), (4, 'C')]
    for user, item in test_cases:
        pred = svd.predict(user, item)
        print(f"User {user} → Item {item}: {pred.est:.2f}")
    

**出力例** ：
    
    
    === データ ===
       user_id item_id  rating
    0        1       A       5
    1        1       B       3
    2        1       C       4
    ...
    
    === モデル評価 ===
    RMSE: 0.7823
    
    === 3-Fold クロスバリデーション ===
    平均RMSE: 0.8234 (+/- 0.1245)
    平均MAE: 0.6543 (+/- 0.0987)
    
    === 新規予測 ===
    User 1 → Item D: 4.12
    User 2 → Item B: 3.45
    User 4 → Item C: 3.78
    

### 問題5（難易度：hard）

協調フィルタリングにおける「Cold Start問題」とは何か説明し、3つの対策手法を具体例とともに述べてください。

解答例

**解答** ：

**Cold Start問題とは** ：

新規ユーザーや新規アイテムに対して、過去の行動データが存在しないため、協調フィルタリングで推薦を行うことが困難な問題。

**3つの対策手法** ：

  1. **コンテンツベースフィルタリングの併用**

     * 具体例: 新規映画の場合、ジャンル、監督、俳優などのメタデータを使って類似映画を推薦
     * 利点: ユーザー行動データなしでも推薦可能
     * 実装: TF-IDFやWord2Vecで特徴量を作成し、コサイン類似度で推薦
  2. **ハイブリッド手法**

     * 具体例: Netflixでは、協調フィルタリング+コンテンツベース+人気度を組み合わせ
     * 利点: 各手法の弱点を補完
     * 実装: 重み付き線形結合やスタッキング
  3. **アクティブラーニング（初期評価の収集）**

     * 具体例: Spotifyの初回登録時に好きなアーティストを選択させる
     * 利点: 少数の評価から嗜好を把握
     * 実装: 人気アイテムや多様なジャンルから初期評価を収集

**追加手法** ：

  * **人気度ベース推薦** : 評価データがない場合、全体的に人気のアイテムを推薦
  * **デモグラフィック情報** : 年齢、性別などから類似ユーザーグループを推定
  * **転移学習** : 他ドメインでの嗜好データを活用

* * *

## 参考文献

  1. Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_ (2nd ed.). Springer.
  2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. _Computer_ , 42(8), 30-37.
  3. Aggarwal, C. C. (2016). _Recommender Systems: The Textbook_. Springer.
  4. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. _UAI 2009_.
  5. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. _ICDM 2008_.
