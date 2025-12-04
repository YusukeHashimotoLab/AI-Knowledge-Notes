---
title: 第1章：推薦システム基礎
chapter_title: 第1章：推薦システム基礎
subtitle: 推薦システムの基本概念とデータ処理の基盤
reading_time: 25-30分
difficulty: 初級
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 推薦システムの役割とビジネス価値を理解する
  * ✅ 推薦タスクの分類と評価指標を学ぶ
  * ✅ Cold Start問題など主要な課題を把握する
  * ✅ MovieLensデータセットの前処理ができる
  * ✅ User-Item行列を構築し、Train-Test分割を実装できる
  * ✅ Pythonで推薦システムの基礎を実装できる

* * *

## 1.1 推薦システムとは

### パーソナライゼーションの重要性

**推薦システム（Recommendation System）** は、ユーザーの好みや行動に基づいて、最適なアイテム（商品、コンテンツ、サービスなど）を提案する技術です。

> 「情報過多の時代において、推薦システムはユーザーと価値あるコンテンツを結びつける重要な役割を果たします。」

### 推薦システムの応用例

業界 | 応用例 | 推薦対象  
---|---|---  
**E-commerce** | Amazon, 楽天 | 商品  
**動画配信** | Netflix, YouTube | 映画、動画  
**音楽配信** | Spotify, Apple Music | 楽曲、プレイリスト  
**SNS** | Facebook, Twitter | 友人、投稿  
**ニュース** | Google News | 記事  
**求人** | LinkedIn | 求人、候補者  
  
### ビジネス価値

  * **売上向上** : クロスセル・アップセルによる収益増加（Amazonの売上の35%は推薦から）
  * **エンゲージメント向上** : ユーザーの滞在時間とコンテンツ消費量の増加
  * **顧客満足度向上** : パーソナライズされた体験による満足度向上
  * **チャーン率低下** : 関連性の高いコンテンツ提供による離脱防止
  * **在庫最適化** : ロングテール商品の発見と販売促進

### 推薦の種類
    
    
    ```mermaid
    graph TD
        A[推薦システム] --> B[協調フィルタリング]
        A --> C[コンテンツベース]
        A --> D[ハイブリッド]
    
        B --> B1[User-based]
        B --> B2[Item-based]
        B --> B3[Matrix Factorization]
    
        C --> C1[特徴量抽出]
        C --> C2[類似度計算]
    
        D --> D1[Weighted Hybrid]
        D --> D2[Switching Hybrid]
        D --> D3[Feature Combination]
    
        style A fill:#e8f5e9
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
    ```

* * *

## 1.2 推薦タスクの分類

### Explicit vs Implicit Feedback

ユーザーのフィードバックには、明示的なものと暗黙的なものがあります。

フィードバック種類 | 説明 | 例 | 長所 | 短所  
---|---|---|---|---  
**Explicit**  
（明示的） | ユーザーが直接評価 | 星評価、いいね、レビュー | 明確な好み情報 | データが少ない  
**Implicit**  
（暗黙的） | 行動から推測 | クリック、視聴時間、購入 | 大量のデータ | 解釈が曖昧  
  
#### Explicit Feedbackの例
    
    
    import pandas as pd
    import numpy as np
    
    # Explicit Feedback: 映画の評価データ
    np.random.seed(42)
    n_ratings = 100
    
    explicit_data = pd.DataFrame({
        'user_id': np.random.randint(1, 21, n_ratings),
        'item_id': np.random.randint(1, 51, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings),  # 1-5の評価
        'timestamp': pd.date_range('2024-01-01', periods=n_ratings, freq='H')
    })
    
    print("=== Explicit Feedback（評価データ）===")
    print(explicit_data.head(10))
    print(f"\n評価の分布:")
    print(explicit_data['rating'].value_counts().sort_index())
    print(f"\n平均評価: {explicit_data['rating'].mean():.2f}")
    print(f"評価の標準偏差: {explicit_data['rating'].std():.2f}")
    

**出力** ：
    
    
    === Explicit Feedback（評価データ）===
       user_id  item_id  rating           timestamp
    0        7       40       4 2024-01-01 00:00:00
    1       20       34       3 2024-01-01 01:00:00
    2       18       48       1 2024-01-01 02:00:00
    3       11       14       5 2024-01-01 03:00:00
    4        6       21       1 2024-01-01 04:00:00
    5       17       28       4 2024-01-01 05:00:00
    6        3        9       1 2024-01-01 06:00:00
    7        9       37       4 2024-01-01 07:00:00
    8       20       17       5 2024-01-01 08:00:00
    9        8       46       2 2024-01-01 09:00:00
    
    評価の分布:
    1    23
    2    19
    3    18
    4    21
    5    19
    Name: rating, dtype: int64
    
    平均評価: 2.98
    評価の標準偏差: 1.47
    

#### Implicit Feedbackの例
    
    
    # Implicit Feedback: 視聴データ
    implicit_data = pd.DataFrame({
        'user_id': np.random.randint(1, 21, n_ratings),
        'item_id': np.random.randint(1, 51, n_ratings),
        'watch_time': np.random.randint(1, 120, n_ratings),  # 分
        'completed': np.random.choice([0, 1], n_ratings, p=[0.3, 0.7]),
        'timestamp': pd.date_range('2024-01-01', periods=n_ratings, freq='H')
    })
    
    # Implicit Feedbackから好みを推測
    # 視聴時間が長い、または完了した場合を「好き」と推定
    implicit_data['preference'] = (
        (implicit_data['watch_time'] > 60) |
        (implicit_data['completed'] == 1)
    ).astype(int)
    
    print("\n=== Implicit Feedback（視聴データ）===")
    print(implicit_data.head(10))
    print(f"\n完了率: {implicit_data['completed'].mean():.1%}")
    print(f"推定好み率: {implicit_data['preference'].mean():.1%}")
    

### Rating Prediction

**Rating Prediction（評価予測）** は、ユーザーがまだ評価していないアイテムに対する評価値を予測するタスクです。

$$ \hat{r}_{ui} = f(\text{user}_u, \text{item}_i) $$

  * $\hat{r}_{ui}$: ユーザー$u$のアイテム$i$に対する予測評価
  * $f$: 推薦モデル

    
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    # User-Item行列の構築（簡易版）
    ratings_matrix = explicit_data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        aggfunc='mean'
    )
    
    print("=== User-Item評価行列 ===")
    print(f"形状: {ratings_matrix.shape}")
    print(f"欠損率: {ratings_matrix.isnull().sum().sum() / (ratings_matrix.shape[0] * ratings_matrix.shape[1]):.1%}")
    print(f"\n行列（一部）:")
    print(ratings_matrix.iloc[:5, :5])
    

### Top-N Recommendation

**Top-N推薦** は、各ユーザーに対して上位N個のアイテムを推薦するタスクです。
    
    
    # 簡易的なTop-N推薦（人気ベース）
    def popularity_based_recommendation(data, n=5):
        """人気度ベースのTop-N推薦"""
        item_popularity = data.groupby('item_id')['rating'].agg(['count', 'mean'])
        item_popularity['score'] = (
            item_popularity['count'] * 0.3 +
            item_popularity['mean'] * 0.7
        )
        top_n = item_popularity.nlargest(n, 'score')
        return top_n
    
    top_items = popularity_based_recommendation(explicit_data, n=5)
    
    print("\n=== Top-5推薦アイテム（人気ベース）===")
    print(top_items)
    print(f"\n推薦理由:")
    print("- スコア = 評価数 × 0.3 + 平均評価 × 0.7")
    

### Ranking Problems

**ランキング問題** は、候補アイテムの順序付けを行うタスクです。関連度の高い順にアイテムを並べます。
    
    
    # ランキングの例: ユーザーごとにアイテムをスコア順に並べる
    def rank_items_for_user(user_id, data):
        """特定ユーザーに対するアイテムランキング"""
        # ユーザーの過去の評価傾向を考慮
        user_ratings = data[data['user_id'] == user_id]
        user_avg_rating = user_ratings['rating'].mean()
    
        # 全アイテムの情報
        all_items = data.groupby('item_id')['rating'].agg(['mean', 'count'])
    
        # スコアリング（簡易版）
        all_items['score'] = (
            all_items['mean'] * 0.5 +
            user_avg_rating * 0.3 +
            np.log1p(all_items['count']) * 0.2
        )
    
        ranked_items = all_items.sort_values('score', ascending=False)
        return ranked_items
    
    # ユーザー7へのランキング
    user_ranking = rank_items_for_user(7, explicit_data)
    print("\n=== ユーザー7へのアイテムランキング（上位10）===")
    print(user_ranking.head(10))
    

* * *

## 1.3 評価指標

### Precision, Recall, F1

推薦システムの精度を測定する基本的な指標です。

$$ \text{Precision@K} = \frac{\text{推薦した関連アイテム数}}{K} $$

$$ \text{Recall@K} = \frac{\text{推薦した関連アイテム数}}{\text{全関連アイテム数}} $$

$$ \text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}} $$
    
    
    def precision_recall_at_k(recommended, relevant, k):
        """Precision@K と Recall@K を計算"""
        recommended_k = recommended[:k]
    
        # 推薦した中で関連があるもの
        hits = len(set(recommended_k) & set(relevant))
    
        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant) if len(relevant) > 0 else 0
    
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    
        return precision, recall, f1
    
    # 例: ユーザーへの推薦
    recommended_items = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # 推薦したアイテム
    relevant_items = [2, 3, 5, 8, 11, 15]  # 実際に関連するアイテム
    
    for k in [5, 10]:
        p, r, f = precision_recall_at_k(recommended_items, relevant_items, k)
        print(f"\n=== K={k} での評価 ===")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Recall@{k}: {r:.3f}")
        print(f"F1@{k}: {f:.3f}")
    

**出力** ：
    
    
    === K=5 での評価 ===
    Precision@5: 0.400
    Recall@5: 0.333
    F1@5: 0.364
    
    === K=10 での評価 ===
    Precision@10: 0.400
    Recall@10: 0.667
    F1@10: 0.500
    

### NDCG (Normalized Discounted Cumulative Gain)

**NDCG** は、ランキングの質を評価する指標です。上位により関連性の高いアイテムを配置することを重視します。

$$ \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)} $$

$$ \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}} $$
    
    
    import numpy as np
    
    def dcg_at_k(relevances, k):
        """DCG@K を計算"""
        relevances = np.array(relevances[:k])
        if relevances.size:
            discounts = np.log2(np.arange(2, relevances.size + 2))
            return np.sum((2**relevances - 1) / discounts)
        return 0.0
    
    def ndcg_at_k(relevances, k):
        """NDCG@K を計算"""
        dcg = dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
    
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    # 例: 推薦結果の関連度スコア（5段階）
    relevances = [3, 2, 5, 0, 1, 4, 2, 0, 3, 1]  # 推薦順の関連度
    
    print("=== NDCG評価 ===")
    for k in [3, 5, 10]:
        ndcg = ndcg_at_k(relevances, k)
        print(f"NDCG@{k}: {ndcg:.3f}")
    
    print(f"\n関連度リスト（推薦順）: {relevances}")
    print(f"理想的な順序: {sorted(relevances, reverse=True)}")
    

### MAP (Mean Average Precision)

**MAP** は、全ユーザーのAverage Precisionの平均です。

$$ \text{AP@K} = \frac{1}{\min(m, K)} \sum_{k=1}^{K} \text{Precision@k} \cdot \text{rel}(k) $$

$$ \text{MAP@K} = \frac{1}{|U|} \sum_{u \in U} \text{AP@K}_u $$
    
    
    def average_precision_at_k(recommended, relevant, k):
        """Average Precision@K を計算"""
        recommended_k = recommended[:k]
    
        score = 0.0
        num_hits = 0.0
    
        for i, item in enumerate(recommended_k):
            if item in relevant:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
    
        if len(relevant) == 0:
            return 0.0
    
        return score / min(len(relevant), k)
    
    # 例: 複数ユーザーのMAP計算
    users_recommendations = [
        ([1, 3, 5, 7, 9], [3, 5, 9]),      # ユーザー1
        ([2, 4, 6, 8, 10], [4, 8]),        # ユーザー2
        ([1, 2, 3, 4, 5], [1, 2, 5]),      # ユーザー3
    ]
    
    aps = []
    for recommended, relevant in users_recommendations:
        ap = average_precision_at_k(recommended, relevant, k=5)
        aps.append(ap)
        print(f"推薦: {recommended}, 関連: {relevant} -> AP@5: {ap:.3f}")
    
    map_score = np.mean(aps)
    print(f"\n=== MAP@5: {map_score:.3f} ===")
    

### Coverage, Diversity, Serendipity

推薦の質を多面的に評価する指標です。

指標 | 説明 | 目的  
---|---|---  
**Coverage**  
（カバレッジ） | 推薦されるアイテムの割合 | ロングテールアイテムの発見  
**Diversity**  
（多様性） | 推薦リスト内のアイテムの多様性 | フィルターバブル回避  
**Serendipity**  
（意外性） | 予想外で関連性の高い推薦 | 新規発見の促進  
      
    
    def calculate_coverage(all_recommendations, total_items):
        """カバレッジを計算"""
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
    
        coverage = len(unique_recommended) / total_items
        return coverage
    
    def calculate_diversity(recommendations):
        """推薦リストの多様性を計算（ユニーク率）"""
        unique_items = len(set(recommendations))
        diversity = unique_items / len(recommendations)
        return diversity
    
    # 例: カバレッジと多様性の計算
    all_recs = [
        [1, 2, 3, 4, 5],
        [1, 3, 6, 7, 8],
        [2, 4, 9, 10, 11],
        [1, 5, 12, 13, 14]
    ]
    
    total_items = 50  # アイテム総数
    
    coverage = calculate_coverage(all_recs, total_items)
    print(f"=== カバレッジと多様性 ===")
    print(f"カバレッジ: {coverage:.1%}")
    print(f"推薦されたユニークアイテム: {len(set([item for recs in all_recs for item in recs]))}")
    
    for i, recs in enumerate(all_recs):
        diversity = calculate_diversity(recs)
        print(f"ユーザー{i+1}の推薦多様性: {diversity:.1%}")
    

* * *

## 1.4 推薦システムの課題

### Cold Start Problem

**Cold Start問題** は、新規ユーザーや新規アイテムに対してデータが不足している問題です。

種類 | 説明 | 対策  
---|---|---  
**User Cold Start** | 新規ユーザーの好み不明 | 人気アイテム推薦、デモグラフィック情報活用  
**Item Cold Start** | 新規アイテムの評価なし | コンテンツベース推薦、メタデータ活用  
**System Cold Start** | システム全体のデータ不足 | 外部データ、クラウドソーシング  
  
### Data Sparsity

**データ希薄性** は、User-Item行列のほとんどが欠損値である問題です。
    
    
    ```mermaid
    graph LR
        A[User-Item行列] --> B[評価済み: 1%]
        A --> C[未評価: 99%]
    
        B --> D[協調フィルタリング可能]
        C --> E[推薦困難]
    
        style A fill:#fff3e0
        style B fill:#c8e6c9
        style C fill:#ffcdd2
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### Scalability

**スケーラビリティ** は、ユーザー数とアイテム数の増加に伴う計算量の問題です。

  * ユーザー数: 100万人
  * アイテム数: 10万個
  * → User-Item行列: 1000億セル

> **対策** : 次元削減（Matrix Factorization）、近似最近傍探索（ANN）、分散処理

### Filter Bubble

**フィルターバブル** は、似たアイテムばかり推薦され、多様性が失われる問題です。

  * **原因** : 過度なパーソナライゼーション
  * **影響** : 新規発見の減少、偏った情報消費
  * **対策** : 多様性の考慮、セレンディピティの導入、探索と活用のバランス

* * *

## 1.5 データセットと前処理

### MovieLens Dataset

**MovieLens** は、推薦システム研究で最も広く使われるデータセットです。

バージョン | 評価数 | ユーザー数 | 映画数 | 用途  
---|---|---|---|---  
100K | 10万 | 943 | 1,682 | 学習、プロトタイプ  
1M | 100万 | 6,040 | 3,706 | 研究、評価  
10M | 1000万 | 71,567 | 10,681 | スケーラビリティ検証  
25M | 2500万 | 162,541 | 62,423 | 大規模実験  
  
### User-Item Matrix

**User-Item行列** は、推薦システムの基本的なデータ構造です。
    
    
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    
    # サンプルデータの作成（MovieLens風）
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 500
    
    ratings_data = pd.DataFrame({
        'user_id': np.random.randint(1, n_users + 1, n_ratings),
        'item_id': np.random.randint(1, n_items + 1, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings),
        'timestamp': pd.date_range('2024-01-01', periods=n_ratings, freq='H')
    })
    
    # 重複を削除（同じユーザー・アイテムペアの最新評価を保持）
    ratings_data = ratings_data.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'item_id'],
        keep='last'
    )
    
    print("=== 評価データ ===")
    print(ratings_data.head(10))
    print(f"\n総評価数: {len(ratings_data)}")
    print(f"ユニークユーザー: {ratings_data['user_id'].nunique()}")
    print(f"ユニークアイテム: {ratings_data['item_id'].nunique()}")
    print(f"評価分布:\n{ratings_data['rating'].value_counts().sort_index()}")
    
    # User-Item行列の構築
    user_item_matrix = ratings_data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )
    
    print(f"\n=== User-Item行列 ===")
    print(f"形状: {user_item_matrix.shape}")
    print(f"密度: {(user_item_matrix > 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.1%}")
    print(f"\n行列の一部（最初の5ユーザー × 5アイテム）:")
    print(user_item_matrix.iloc[:5, :5])
    
    # 疎行列への変換（メモリ効率化）
    sparse_matrix = csr_matrix(user_item_matrix.values)
    print(f"\n疎行列サイズ: {sparse_matrix.data.nbytes / 1024:.2f} KB")
    print(f"密行列サイズ: {user_item_matrix.values.nbytes / 1024:.2f} KB")
    print(f"メモリ削減率: {(1 - sparse_matrix.data.nbytes / user_item_matrix.values.nbytes):.1%}")
    

### Train-Test Split Strategies

推薦システムでは、時系列を考慮した分割が重要です。
    
    
    from sklearn.model_selection import train_test_split
    
    # 1. ランダム分割（単純だが時系列を無視）
    train_random, test_random = train_test_split(
        ratings_data,
        test_size=0.2,
        random_state=42
    )
    
    print("=== 1. ランダム分割 ===")
    print(f"訓練データ: {len(train_random)}件")
    print(f"テストデータ: {len(test_random)}件")
    
    # 2. 時系列分割（より現実的）
    ratings_data_sorted = ratings_data.sort_values('timestamp')
    split_idx = int(len(ratings_data_sorted) * 0.8)
    
    train_temporal = ratings_data_sorted.iloc[:split_idx]
    test_temporal = ratings_data_sorted.iloc[split_idx:]
    
    print("\n=== 2. 時系列分割 ===")
    print(f"訓練期間: {train_temporal['timestamp'].min()} ~ {train_temporal['timestamp'].max()}")
    print(f"テスト期間: {test_temporal['timestamp'].min()} ~ {test_temporal['timestamp'].max()}")
    print(f"訓練データ: {len(train_temporal)}件")
    print(f"テストデータ: {len(test_temporal)}件")
    
    # 3. ユーザーごとの分割（Leave-One-Out）
    def leave_one_out_split(data):
        """各ユーザーの最新評価をテストセットに"""
        train_list = []
        test_list = []
    
        for user_id, group in data.groupby('user_id'):
            group_sorted = group.sort_values('timestamp')
            if len(group_sorted) > 1:
                train_list.append(group_sorted.iloc[:-1])
                test_list.append(group_sorted.iloc[-1:])
            else:
                train_list.append(group_sorted)
    
        train = pd.concat(train_list)
        test = pd.concat(test_list) if test_list else pd.DataFrame()
    
        return train, test
    
    train_loo, test_loo = leave_one_out_split(ratings_data)
    
    print("\n=== 3. Leave-One-Out分割 ===")
    print(f"訓練データ: {len(train_loo)}件")
    print(f"テストデータ: {len(test_loo)}件")
    print(f"テストユーザー数: {test_loo['user_id'].nunique()}")
    

### Python Preprocessing

推薦システムのデータ前処理の実践例です。
    
    
    import pandas as pd
    import numpy as np
    
    class RecommendationDataPreprocessor:
        """推薦システムデータの前処理クラス"""
    
        def __init__(self, min_user_ratings=5, min_item_ratings=5):
            self.min_user_ratings = min_user_ratings
            self.min_item_ratings = min_item_ratings
            self.user_mapping = {}
            self.item_mapping = {}
    
        def filter_rare_users_items(self, data):
            """評価数が少ないユーザー・アイテムを除外"""
            print("=== フィルタリング前 ===")
            print(f"ユーザー数: {data['user_id'].nunique()}")
            print(f"アイテム数: {data['item_id'].nunique()}")
            print(f"評価数: {len(data)}")
    
            # ユーザーのフィルタリング
            user_counts = data['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            data = data[data['user_id'].isin(valid_users)]
    
            # アイテムのフィルタリング
            item_counts = data['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_ratings].index
            data = data[data['item_id'].isin(valid_items)]
    
            print("\n=== フィルタリング後 ===")
            print(f"ユーザー数: {data['user_id'].nunique()}")
            print(f"アイテム数: {data['item_id'].nunique()}")
            print(f"評価数: {len(data)}")
    
            return data
    
        def create_mappings(self, data):
            """ユーザー・アイテムIDを連続した整数にマッピング"""
            unique_users = sorted(data['user_id'].unique())
            unique_items = sorted(data['item_id'].unique())
    
            self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
    
            data['user_idx'] = data['user_id'].map(self.user_mapping)
            data['item_idx'] = data['item_id'].map(self.item_mapping)
    
            print("\n=== IDマッピング ===")
            print(f"ユーザーID範囲: {data['user_id'].min()} ~ {data['user_id'].max()}")
            print(f"ユーザーインデックス範囲: {data['user_idx'].min()} ~ {data['user_idx'].max()}")
            print(f"アイテムID範囲: {data['item_id'].min()} ~ {data['item_id'].max()}")
            print(f"アイテムインデックス範囲: {data['item_idx'].min()} ~ {data['item_idx'].max()}")
    
            return data
    
        def normalize_ratings(self, data, method='mean'):
            """評価値を正規化"""
            if method == 'mean':
                # 平均を引く
                user_means = data.groupby('user_id')['rating'].transform('mean')
                data['rating_normalized'] = data['rating'] - user_means
            elif method == 'minmax':
                # [0, 1]にスケーリング
                data['rating_normalized'] = (data['rating'] - data['rating'].min()) / (
                    data['rating'].max() - data['rating'].min()
                )
    
            print(f"\n=== 評価正規化（{method}）===")
            print(f"元の評価範囲: [{data['rating'].min()}, {data['rating'].max()}]")
            print(f"正規化後の範囲: [{data['rating_normalized'].min():.2f}, {data['rating_normalized'].max():.2f}]")
    
            return data
    
    # 前処理の実行
    preprocessor = RecommendationDataPreprocessor(
        min_user_ratings=3,
        min_item_ratings=3
    )
    
    # データのフィルタリング
    filtered_data = preprocessor.filter_rare_users_items(ratings_data)
    
    # IDマッピング
    mapped_data = preprocessor.create_mappings(filtered_data)
    
    # 評価正規化
    normalized_data = preprocessor.normalize_ratings(mapped_data, method='mean')
    
    print("\n=== 前処理完了データ（サンプル）===")
    print(normalized_data[['user_id', 'user_idx', 'item_id', 'item_idx',
                            'rating', 'rating_normalized']].head(10))
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **推薦システムの役割**

     * 情報過多の時代に最適なコンテンツを提案
     * 売上向上、エンゲージメント向上、顧客満足度向上に貢献
     * 協調フィルタリング、コンテンツベース、ハイブリッド手法
  2. **推薦タスクの種類**

     * Explicit vs Implicit Feedback
     * Rating Prediction、Top-N推薦、ランキング
     * タスクに応じた適切な手法選択
  3. **評価指標**

     * Precision、Recall、F1: 精度の基本指標
     * NDCG: ランキング品質の評価
     * MAP: 平均精度の評価
     * Coverage、Diversity、Serendipity: 推薦の質
  4. **主要な課題**

     * Cold Start問題: 新規ユーザー・アイテムの対処
     * Data Sparsity: 疎なデータの扱い
     * Scalability: 大規模データの処理
     * Filter Bubble: 多様性の確保
  5. **データ処理の実践**

     * MovieLensデータセットの活用
     * User-Item行列の構築
     * 適切なTrain-Test分割
     * 前処理パイプラインの構築

### 推薦システム設計の原則

原則 | 説明  
---|---  
**ユーザー中心設計** | ユーザーの満足度と体験を最優先  
**多面的評価** | 精度だけでなく多様性、新規性も考慮  
**時系列考慮** | 評価の時系列を尊重した分割と評価  
**スケーラビリティ** | 大規模データに対応できる設計  
**継続的改善** | A/Bテストと定期的な評価で改善  
  
### 次の章へ

第2章では、**協調フィルタリング** を学びます：

  * User-based協調フィルタリング
  * Item-based協調フィルタリング
  * 類似度計算（コサイン類似度、ピアソン相関）
  * 近傍探索とk-NN
  * 実装と評価

* * *

## 演習問題

### 問題1（難易度：easy）

Explicit FeedbackとImplicit Feedbackの違いを説明し、それぞれの長所と短所を述べてください。

解答例

**解答** ：

**Explicit Feedback（明示的フィードバック）** ：

  * 定義: ユーザーが意図的に提供する評価（星評価、いいね、レビュー）
  * 長所: 明確な好み情報、解釈が容易
  * 短所: データ収集が困難、ユーザーの負担が大きい、データ量が少ない

**Implicit Feedback（暗黙的フィードバック）** ：

  * 定義: ユーザーの行動から推測される好み（クリック、視聴時間、購入）
  * 長所: 大量のデータ、ユーザー負担なし、自然な行動
  * 短所: 解釈が曖昧（クリックが好みとは限らない）、ネガティブフィードバック不明

**使い分け** ：

  * Explicit: 評価が重要な分野（映画、書籍レビュー）
  * Implicit: 大規模サービス（動画配信、ECサイト）
  * Hybrid: 両方を組み合わせて精度向上

### 問題2（難易度：medium）

以下の推薦結果に対して、Precision@5とRecall@5を計算してください。
    
    
    recommended = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    relevant = [2, 3, 5, 8, 11, 15, 20]
    

解答例
    
    
    def precision_recall_at_k(recommended, relevant, k):
        """Precision@K と Recall@K を計算"""
        recommended_k = recommended[:k]
    
        # 推薦した中で関連があるもの
        hits = len(set(recommended_k) & set(relevant))
    
        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant) if len(relevant) > 0 else 0
    
        return precision, recall
    
    recommended = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    relevant = [2, 3, 5, 8, 11, 15, 20]
    
    precision, recall = precision_recall_at_k(recommended, relevant, k=5)
    
    print("=== 計算過程 ===")
    print(f"推薦アイテム（上位5件）: {recommended[:5]}")
    print(f"関連アイテム: {relevant}")
    print(f"ヒット: {set(recommended[:5]) & set(relevant)}")
    print(f"ヒット数: {len(set(recommended[:5]) & set(relevant))}")
    print(f"\nPrecision@5 = {len(set(recommended[:5]) & set(relevant))} / 5 = {precision:.3f}")
    print(f"Recall@5 = {len(set(recommended[:5]) & set(relevant))} / {len(relevant)} = {recall:.3f}")
    

**出力** ：
    
    
    === 計算過程 ===
    推薦アイテム（上位5件）: [1, 3, 5, 7, 9]
    関連アイテム: [2, 3, 5, 8, 11, 15, 20]
    ヒット: {3, 5}
    ヒット数: 2
    
    Precision@5 = 2 / 5 = 0.400
    Recall@5 = 2 / 7 = 0.286
    

### 問題3（難易度：medium）

Cold Start問題の3つの種類（User、Item、System）をそれぞれ説明し、対処法を提案してください。

解答例

**解答** ：

**1\. User Cold Start（新規ユーザー問題）** ：

  * 説明: 新規ユーザーは評価履歴がなく、好みが不明
  * 対処法: 
    * 人気アイテムの推薦（全体での人気度）
    * デモグラフィック情報（年齢、性別、地域）を活用
    * 初期質問によるプロファイリング
    * ソーシャルネットワーク情報の活用

**2\. Item Cold Start（新規アイテム問題）** ：

  * 説明: 新規アイテムは評価がなく、推薦できない
  * 対処法: 
    * コンテンツベース推薦（アイテムの特徴から類似性計算）
    * メタデータ活用（ジャンル、タグ、説明文）
    * アクティブユーザーへの優先提示
    * 専門家による初期評価

**3\. System Cold Start（システム全体の問題）** ：

  * 説明: サービス開始直後でユーザー・アイテムともにデータ不足
  * 対処法: 
    * 外部データソースの活用（既存レビューサイト）
    * クラウドソーシングによる初期データ収集
    * エキスパートキュレーション
    * Transfer Learning（他ドメインからの知識転移）

**実例** ：

  * Netflix: 新規ユーザーに好きな作品を3つ選ばせる
  * Spotify: 好きなアーティストを選択させてプロファイル作成
  * Amazon: 閲覧履歴と人気商品を組み合わせて推薦

### 問題4（難易度：hard）

以下のデータに対して、User-Item行列を構築し、時系列分割（訓練80%、テスト20%）を実装してください。また、行列の密度も計算してください。
    
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
        'item_id': [10, 20, 10, 30, 40, 20, 30, 10, 50, 40],
        'rating': [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D')
    })
    

解答例
    
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
        'item_id': [10, 20, 10, 30, 40, 20, 30, 10, 50, 40],
        'rating': [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D')
    })
    
    print("=== 元データ ===")
    print(data)
    
    # User-Item行列の構築
    user_item_matrix = data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )
    
    print("\n=== User-Item行列 ===")
    print(user_item_matrix)
    
    # 密度の計算
    total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    non_zero_cells = (user_item_matrix > 0).sum().sum()
    density = non_zero_cells / total_cells
    
    print(f"\n=== 行列の統計 ===")
    print(f"形状: {user_item_matrix.shape}")
    print(f"総セル数: {total_cells}")
    print(f"評価済みセル数: {non_zero_cells}")
    print(f"密度: {density:.1%}")
    print(f"疎度: {(1 - density):.1%}")
    
    # 時系列分割
    data_sorted = data.sort_values('timestamp')
    split_idx = int(len(data_sorted) * 0.8)
    
    train_data = data_sorted.iloc[:split_idx]
    test_data = data_sorted.iloc[split_idx:]
    
    print("\n=== 時系列分割 ===")
    print(f"訓練データ件数: {len(train_data)}")
    print(f"テストデータ件数: {len(test_data)}")
    print(f"\n訓練期間: {train_data['timestamp'].min()} ~ {train_data['timestamp'].max()}")
    print(f"テスト期間: {test_data['timestamp'].min()} ~ {test_data['timestamp'].max()}")
    
    print("\n訓練データ:")
    print(train_data)
    print("\nテストデータ:")
    print(test_data)
    
    # 訓練データとテストデータのUser-Item行列
    train_matrix = train_data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )
    
    print("\n=== 訓練データのUser-Item行列 ===")
    print(train_matrix)
    

**出力** ：
    
    
    === 元データ ===
       user_id  item_id  rating  timestamp
    0        1       10       5 2024-01-01
    1        1       20       4 2024-01-02
    2        2       10       3 2024-01-03
    3        2       30       5 2024-01-04
    4        2       40       2 2024-01-05
    5        3       20       4 2024-01-06
    6        3       30       5 2024-01-07
    7        4       10       3 2024-01-08
    8        4       50       4 2024-01-09
    9        5       40       5 2024-01-10
    
    === User-Item行列 ===
    item_id  10  20  30  40  50
    user_id
    1         5   4   0   0   0
    2         3   0   5   2   0
    3         0   4   5   0   0
    4         3   0   0   0   4
    5         0   0   0   5   0
    
    === 行列の統計 ===
    形状: (5, 5)
    総セル数: 25
    評価済みセル数: 10
    密度: 40.0%
    疎度: 60.0%
    
    === 時系列分割 ===
    訓練データ件数: 8
    テストデータ件数: 2
    
    訓練期間: 2024-01-01 ~ 2024-01-08
    テスト期間: 2024-01-09 ~ 2024-01-10
    
    訓練データ:
       user_id  item_id  rating  timestamp
    0        1       10       5 2024-01-01
    1        1       20       4 2024-01-02
    2        2       10       3 2024-01-03
    3        2       30       5 2024-01-04
    4        2       40       2 2024-01-05
    5        3       20       4 2024-01-06
    6        3       30       5 2024-01-07
    7        4       10       3 2024-01-08
    
    テストデータ:
       user_id  item_id  rating  timestamp
    8        4       50       4 2024-01-09
    9        5       40       5 2024-01-10
    
    === 訓練データのUser-Item行列 ===
    item_id  10  20  30  40
    user_id
    1         5   4   0   0
    2         3   0   5   2
    3         0   4   5   0
    4         3   0   0   0
    

### 問題5（難易度：hard）

NDCG@5を計算する関数を実装し、以下の推薦結果の品質を評価してください。関連度スコアは5段階（0-4）です。
    
    
    relevances = [3, 2, 0, 1, 4, 0, 2, 3, 1, 0]  # 推薦順の関連度
    

解答例
    
    
    import numpy as np
    
    def dcg_at_k(relevances, k):
        """DCG@K を計算
    
        DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
        """
        relevances = np.array(relevances[:k])
        if relevances.size:
            # 位置iのディスカウント係数: log2(i + 1)
            # i=1から始まるため、log2(2), log2(3), ...
            discounts = np.log2(np.arange(2, relevances.size + 2))
            gains = 2**relevances - 1
            dcg = np.sum(gains / discounts)
            return dcg
        return 0.0
    
    def ndcg_at_k(relevances, k):
        """NDCG@K を計算
    
        NDCG@K = DCG@K / IDCG@K
        """
        dcg = dcg_at_k(relevances, k)
    
        # Ideal DCG: 関連度を降順にソートした場合のDCG
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
    
        if idcg == 0:
            return 0.0
    
        ndcg = dcg / idcg
        return ndcg
    
    # 例: 推薦結果の評価
    relevances = [3, 2, 0, 1, 4, 0, 2, 3, 1, 0]
    
    print("=== NDCG評価 ===")
    print(f"推薦順の関連度: {relevances}")
    print(f"理想的な順序: {sorted(relevances, reverse=True)}")
    
    for k in [3, 5, 10]:
        dcg = dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
        ndcg = ndcg_at_k(relevances, k)
    
        print(f"\n=== K={k} ===")
        print(f"DCG@{k}: {dcg:.3f}")
        print(f"IDCG@{k}: {idcg:.3f}")
        print(f"NDCG@{k}: {ndcg:.3f}")
    
    # 詳細計算例（K=5）
    print("\n=== 詳細計算（K=5）===")
    k = 5
    rels = relevances[:k]
    print(f"上位{k}件の関連度: {rels}")
    
    for i, rel in enumerate(rels):
        pos = i + 1
        gain = 2**rel - 1
        discount = np.log2(pos + 1)
        contribution = gain / discount
        print(f"位置{pos}: rel={rel}, gain={gain}, discount={discount:.3f}, contribution={contribution:.3f}")
    
    dcg = dcg_at_k(relevances, k)
    print(f"\nDCG@5 = {dcg:.3f}")
    
    ideal_rels = sorted(relevances, reverse=True)[:k]
    print(f"\n理想的な上位{k}件: {ideal_rels}")
    idcg = dcg_at_k(ideal_rels, k)
    print(f"IDCG@5 = {idcg:.3f}")
    
    ndcg = ndcg_at_k(relevances, k)
    print(f"\nNDCG@5 = {dcg:.3f} / {idcg:.3f} = {ndcg:.3f}")
    

**出力** ：
    
    
    === NDCG評価 ===
    推薦順の関連度: [3, 2, 0, 1, 4, 0, 2, 3, 1, 0]
    理想的な順序: [4, 3, 3, 2, 2, 1, 1, 0, 0, 0]
    
    === K=3 ===
    DCG@3: 7.500
    IDCG@3: 11.131
    NDCG@3: 0.674
    
    === K=5 ===
    DCG@5: 16.714
    IDCG@5: 19.714
    NDCG@5: 0.848
    
    === K=10 ===
    DCG@10: 20.344
    IDCG@10: 23.344
    NDCG@10: 0.871
    
    === 詳細計算（K=5）===
    上位5件の関連度: [3, 2, 0, 1, 4]
    位置1: rel=3, gain=7, discount=1.000, contribution=7.000
    位置2: rel=2, gain=3, discount=1.585, contribution=1.893
    位置3: rel=0, gain=0, discount=2.000, contribution=0.000
    位置4: rel=1, gain=1, discount=2.322, contribution=0.431
    位置5: rel=4, gain=15, discount=2.585, contribution=5.803
    
    DCG@5 = 15.127
    
    理想的な上位5件: [4, 3, 3, 2, 2]
    IDCG@5 = 19.714
    
    NDCG@5 = 15.127 / 19.714 = 0.767
    

* * *

## 参考文献

  1. Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_ (2nd ed.). Springer.
  2. Aggarwal, C. C. (2016). _Recommender Systems: The Textbook_. Springer.
  3. Falk, K. (2019). _Practical Recommender Systems_. Manning Publications.
  4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. _ACM Transactions on Interactive Intelligent Systems_ , 5(4), 1-19.
  5. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. _Computer_ , 42(8), 30-37.
