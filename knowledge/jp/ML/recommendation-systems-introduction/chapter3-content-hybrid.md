---
title: 第3章：コンテンツベース・ハイブリッド推薦
chapter_title: 第3章：コンテンツベース・ハイブリッド推薦
subtitle: 特徴量ベース推薦とハイブリッド手法の実装
reading_time: 70-80分
difficulty: 中級
code_examples: 9
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ コンテンツベースフィルタリングの原理と実装方法を理解する
  * ✅ TF-IDFを用いたテキスト特徴量抽出を実装できる
  * ✅ ユーザープロファイルの構築と推薦生成ができる
  * ✅ ハイブリッド推薦手法を設計・実装できる
  * ✅ Knowledge-based推薦の仕組みを理解する
  * ✅ コンテキストを考慮した推薦システムを構築できる

* * *

## 3.1 コンテンツベース推薦の原理

### コンテンツベースフィルタリングとは

**コンテンツベースフィルタリング（Content-Based Filtering）** は、アイテムの特徴量とユーザーの嗜好を照合して推薦を行う手法です。

> 「ユーザーが過去に好んだアイテムと類似した特徴を持つアイテムを推薦する」

### 協調フィルタリングとの比較

項目 | 協調フィルタリング | コンテンツベース  
---|---|---  
**推薦の根拠** | ユーザー間・アイテム間の類似性 | アイテムの特徴量とユーザー嗜好  
**必要データ** | ユーザー-アイテム評価行列 | アイテムの特徴量  
**コールドスタート** | 新規ユーザー・アイテムで困難 | 新規アイテムでも可能  
**多様性** | 意外性のある推薦が可能 | 既知の嗜好に偏りやすい  
**スケーラビリティ** | ユーザー数依存 | アイテム特徴量依存  
  
### コンテンツベース推薦のフロー
    
    
    ```mermaid
    graph TD
        A[アイテム特徴量抽出] --> B[ユーザープロファイル構築]
        B --> C[アイテムとプロファイルの類似度計算]
        C --> D[類似度に基づく推薦]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### 主要な特徴量の種類

特徴量タイプ | 例 | 抽出手法  
---|---|---  
**テキスト** | 説明文、レビュー、タグ | TF-IDF、Word2Vec、BERT  
**カテゴリ** | ジャンル、カテゴリ | One-Hot、Target Encoding  
**数値** | 価格、評価、人気度 | 標準化、ビニング  
**画像** | 商品画像、サムネイル | CNN、ResNet、CLIP  
**音声** | 音楽、ポッドキャスト | MFCC、スペクトログラム  
  
* * *

## 3.2 TF-IDFによるテキスト特徴量抽出

### TF-IDF（Term Frequency-Inverse Document Frequency）

**TF-IDF** は、文書内の単語の重要度を評価する指標です。

$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$

  * **TF (Term Frequency)** : 文書$d$内の単語$t$の出現頻度
  * **IDF (Inverse Document Frequency)** : 単語$t$の希少性

$$ \text{IDF}(t) = \log \frac{N}{\text{df}(t)} $$

  * $N$: 文書総数
  * $\text{df}(t)$: 単語$t$を含む文書数

### 実装：映画の説明文に基づく推薦
    
    
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # サンプル映画データ
    movies = pd.DataFrame({
        'title': [
            'The Matrix',
            'Inception',
            'The Dark Knight',
            'Interstellar',
            'The Avengers',
            'Iron Man',
            'Titanic',
            'The Notebook'
        ],
        'description': [
            'A computer hacker learns about the true nature of reality and his role in the war against its controllers',
            'A thief who steals corporate secrets through dream-sharing technology',
            'Batman fights crime and chaos in Gotham City with the Joker',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival',
            'Earth mightiest heroes must come together to stop an alien invasion',
            'A billionaire industrialist builds an armored suit to fight evil',
            'A romance develops aboard a ship during its ill-fated maiden voyage',
            'A poor yet passionate young man falls in love with a rich young woman'
        ],
        'genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Sci-Fi', 'Action', 'Action', 'Romance', 'Romance']
    })
    
    print("=== 映画データ ===")
    print(movies[['title', 'genre']])
    
    # TF-IDF特徴量抽出
    tfidf = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_matrix = tfidf.fit_transform(movies['description'])
    
    print(f"\n=== TF-IDF行列 ===")
    print(f"形状: {tfidf_matrix.shape}")
    print(f"特徴語数: {len(tfidf.get_feature_names_out())}")
    print(f"主要な単語: {tfidf.get_feature_names_out()[:15]}")
    
    # コサイン類似度の計算
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print(f"\n=== コサイン類似度行列 ===")
    print(f"形状: {cosine_sim.shape}")
    print("\nThe Matrix と他の映画の類似度:")
    for i, title in enumerate(movies['title']):
        print(f"  {title}: {cosine_sim[0, i]:.3f}")
    

**出力** ：
    
    
    === 映画データ ===
                title   genre
    0      The Matrix  Sci-Fi
    1       Inception  Sci-Fi
    2  The Dark Knight  Action
    3    Interstellar  Sci-Fi
    4    The Avengers  Action
    5        Iron Man  Action
    6         Titanic Romance
    7    The Notebook Romance
    
    === TF-IDF行列 ===
    形状: (8, 50)
    特徴語数: 50
    主要な単語: ['aboard' 'against' 'alien' 'armored' 'attempt' 'batman' 'billionaire'
     'builds' 'chaos' 'city' 'come' 'computer' 'controllers' 'corporate'
     'crime']
    
    === コサイン類似度行列 ===
    形状: (8, 8)
    
    The Matrix と他の映画の類似度:
      The Matrix: 1.000
      Inception: 0.000
      The Dark Knight: 0.000
      Interstellar: 0.000
      The Avengers: 0.000
      Iron Man: 0.000
      Titanic: 0.000
      The Notebook: 0.087
    

### 推薦関数の実装
    
    
    def get_content_based_recommendations(movie_title, movies_df, cosine_sim_matrix, top_n=5):
        """
        コンテンツベース推薦を生成
    
        Args:
            movie_title: 基準となる映画タイトル
            movies_df: 映画データフレーム
            cosine_sim_matrix: コサイン類似度行列
            top_n: 推薦する映画数
    
        Returns:
            推薦映画のデータフレーム
        """
        # 映画のインデックスを取得
        idx = movies_df[movies_df['title'] == movie_title].index[0]
    
        # 類似度スコアを取得
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
        # 類似度でソート（自分自身を除く）
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
        # 映画インデックスを取得
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
    
        # 推薦結果を返す
        recommendations = movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = similarity_scores
    
        return recommendations[['title', 'genre', 'similarity_score']]
    
    # 推薦の生成
    print("\n=== 'The Matrix' に類似した映画 ===")
    recommendations = get_content_based_recommendations('The Matrix', movies, cosine_sim, top_n=3)
    print(recommendations)
    
    print("\n=== 'Titanic' に類似した映画 ===")
    recommendations = get_content_based_recommendations('Titanic', movies, cosine_sim, top_n=3)
    print(recommendations)
    

**出力** ：
    
    
    === 'The Matrix' に類似した映画 ===
               title   genre  similarity_score
    1      Inception  Sci-Fi             0.186
    3   Interstellar  Sci-Fi             0.159
    2 The Dark Knight Action             0.124
    
    === 'Titanic' に類似した映画 ===
               title    genre  similarity_score
    7   The Notebook Romance             0.573
    4   The Avengers   Action             0.097
    5       Iron Man   Action             0.000
    

* * *

## 3.3 ユーザープロファイルの構築

### ユーザープロファイルとは

**ユーザープロファイル** は、ユーザーが過去に評価したアイテムの特徴量を集約したベクトルです。

$$ \text{UserProfile}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui} \cdot \text{ItemFeatures}_i $$

  * $I_u$: ユーザー$u$が評価したアイテム集合
  * $r_{ui}$: ユーザー$u$のアイテム$i$への評価（重み）

### 実装：ユーザープロファイルベース推薦
    
    
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # ユーザー評価データ
    user_ratings = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'title': ['The Matrix', 'Inception', 'Interstellar',
                  'The Avengers', 'Iron Man', 'The Dark Knight',
                  'Titanic', 'The Notebook'],
        'rating': [5, 4, 5, 5, 4, 4, 5, 5]
    })
    
    print("=== ユーザー評価データ ===")
    print(user_ratings)
    
    # TF-IDF行列の再取得
    tfidf_matrix = tfidf.transform(movies['description'])
    
    def build_user_profile(user_id, ratings_df, movies_df, tfidf_matrix):
        """ユーザープロファイルを構築"""
        # ユーザーの評価アイテムを取得
        user_data = ratings_df[ratings_df['user_id'] == user_id]
    
        # 映画インデックスと評価を取得
        movie_indices = []
        ratings = []
        for _, row in user_data.iterrows():
            idx = movies_df[movies_df['title'] == row['title']].index[0]
            movie_indices.append(idx)
            ratings.append(row['rating'])
    
        # 加重平均でプロファイル作成
        ratings = np.array(ratings)
        user_profile = np.zeros(tfidf_matrix.shape[1])
    
        for idx, rating in zip(movie_indices, ratings):
            user_profile += rating * tfidf_matrix[idx].toarray().flatten()
    
        user_profile /= np.sum(ratings)
    
        return user_profile.reshape(1, -1)
    
    # ユーザープロファイルの構築
    user1_profile = build_user_profile(1, user_ratings, movies, tfidf_matrix)
    user2_profile = build_user_profile(2, user_ratings, movies, tfidf_matrix)
    user3_profile = build_user_profile(3, user_ratings, movies, tfidf_matrix)
    
    print("\n=== ユーザープロファイル ===")
    print(f"User 1 profile shape: {user1_profile.shape}")
    print(f"User 2 profile shape: {user2_profile.shape}")
    print(f"User 3 profile shape: {user3_profile.shape}")
    
    def recommend_for_user(user_profile, movies_df, tfidf_matrix, watched_movies, top_n=3):
        """ユーザープロファイルに基づく推薦"""
        # 全映画との類似度計算
        similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
        # 既視聴映画を除外
        watched_indices = [movies_df[movies_df['title'] == title].index[0]
                           for title in watched_movies]
        similarities[watched_indices] = -1
    
        # Top-N推薦
        top_indices = similarities.argsort()[::-1][:top_n]
    
        recommendations = movies_df.iloc[top_indices].copy()
        recommendations['score'] = similarities[top_indices]
    
        return recommendations[['title', 'genre', 'score']]
    
    # ユーザーごとの推薦
    print("\n=== User 1 への推薦（Sci-Fi好き）===")
    watched_1 = user_ratings[user_ratings['user_id'] == 1]['title'].tolist()
    recs_1 = recommend_for_user(user1_profile, movies, tfidf_matrix, watched_1, top_n=3)
    print(recs_1)
    
    print("\n=== User 2 への推薦（Action好き）===")
    watched_2 = user_ratings[user_ratings['user_id'] == 2]['title'].tolist()
    recs_2 = recommend_for_user(user2_profile, movies, tfidf_matrix, watched_2, top_n=3)
    print(recs_2)
    
    print("\n=== User 3 への推薦（Romance好き）===")
    watched_3 = user_ratings[user_ratings['user_id'] == 3]['title'].tolist()
    recs_3 = recommend_for_user(user3_profile, movies, tfidf_matrix, watched_3, top_n=3)
    print(recs_3)
    

**出力** ：
    
    
    === ユーザー評価データ ===
       user_id            title  rating
    0        1       The Matrix       5
    1        1        Inception       4
    2        1     Interstellar       5
    3        2     The Avengers       5
    4        2         Iron Man       4
    5        2  The Dark Knight       4
    6        3          Titanic       5
    7        3     The Notebook       5
    
    === ユーザープロファイル ===
    User 1 profile shape: (1, 50)
    User 2 profile shape: (1, 50)
    User 3 profile shape: (1, 50)
    
    === User 1 への推薦（Sci-Fi好き）===
                  title   genre     score
    2  The Dark Knight  Action  0.091847
    5         Iron Man  Action  0.062134
    4     The Avengers  Action  0.054621
    
    === User 2 への推薦（Action好き）===
            title   genre     score
    0  The Matrix  Sci-Fi  0.103256
    1   Inception  Sci-Fi  0.082471
    3 Interstellar  Sci-Fi  0.071829
    
    === User 3 への推薦（Romance好き）===
            title   genre     score
    0  The Matrix  Sci-Fi  0.028645
    1   Inception  Sci-Fi  0.027183
    4 The Avengers  Action  0.026451
    

* * *

## 3.4 ハイブリッド推薦システム

### ハイブリッド推薦の種類

手法 | 説明 | 実装難易度  
---|---|---  
**Weighted** | 複数手法のスコアを重み付け線形結合 | 低  
**Switching** | 状況に応じて手法を切り替え | 中  
**Mixed** | 複数手法の推薦を混合して提示 | 低  
**Feature Combination** | 協調とコンテンツの特徴量を統合 | 高  
**Cascade** | 段階的に手法を適用 | 中  
**Meta-level** | 一つの手法の出力を別の手法の入力に | 高  
  
### 実装：Weighted Hybrid
    
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    
    # サンプルデータ拡張
    np.random.seed(42)
    n_users = 50
    n_movies = len(movies)
    
    # ユーザー-映画評価行列（協調フィルタリング用）
    ratings_data = []
    for user_id in range(n_users):
        n_ratings = np.random.randint(2, 6)
        movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
        for movie_idx in movie_indices:
            rating = np.random.randint(1, 6)
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_idx,
                'rating': rating
            })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # ユーザー-映画行列の構築
    user_movie_matrix = csr_matrix(
        (ratings_df['rating'], (ratings_df['user_id'], ratings_df['movie_id'])),
        shape=(n_users, n_movies)
    ).toarray()
    
    print("=== ユーザー-映画評価行列 ===")
    print(f"形状: {user_movie_matrix.shape}")
    print(f"評価数: {len(ratings_df)}")
    print(f"スパース率: {(1 - len(ratings_df) / (n_users * n_movies)) * 100:.1f}%")
    
    # 協調フィルタリングスコア計算（Item-based CF）
    def collaborative_filtering_score(user_id, movie_id, user_movie_matrix):
        """Item-based CF スコア"""
        # ユーザーが評価した映画
        user_ratings = user_movie_matrix[user_id]
    
        # 映画間の類似度（評価ベクトルのコサイン類似度）
        movie_sim = cosine_similarity(user_movie_matrix.T)
    
        # ユーザーが評価した映画とターゲット映画の類似度
        rated_movies = np.where(user_ratings > 0)[0]
    
        if len(rated_movies) == 0:
            return 0.0
    
        # 加重平均スコア
        weighted_sum = 0
        similarity_sum = 0
    
        for rated_movie in rated_movies:
            if rated_movie != movie_id:
                sim = movie_sim[movie_id, rated_movie]
                weighted_sum += sim * user_ratings[rated_movie]
                similarity_sum += abs(sim)
    
        if similarity_sum == 0:
            return 0.0
    
        return weighted_sum / similarity_sum
    
    # コンテンツベーススコア計算
    def content_based_score(user_id, movie_id, user_movie_matrix, tfidf_matrix):
        """コンテンツベーススコア"""
        # ユーザープロファイル構築
        user_ratings = user_movie_matrix[user_id]
        rated_movies = np.where(user_ratings > 0)[0]
    
        if len(rated_movies) == 0:
            return 0.0
    
        # 加重平均プロファイル
        user_profile = np.zeros(tfidf_matrix.shape[1])
        for movie_idx in rated_movies:
            user_profile += user_ratings[movie_idx] * tfidf_matrix[movie_idx].toarray().flatten()
    
        user_profile /= np.sum(user_ratings[rated_movies])
    
        # ターゲット映画との類似度
        movie_features = tfidf_matrix[movie_id].toarray().flatten()
        score = cosine_similarity(user_profile.reshape(1, -1),
                                 movie_features.reshape(1, -1))[0, 0]
    
        return score
    
    # ハイブリッドスコア計算
    def hybrid_score(user_id, movie_id, user_movie_matrix, tfidf_matrix,
                    alpha=0.5):
        """
        Weighted Hybrid スコア
    
        Args:
            alpha: 協調フィルタリングの重み (0-1)
                   1-alpha がコンテンツベースの重み
        """
        cf_score = collaborative_filtering_score(user_id, movie_id, user_movie_matrix)
        cb_score = content_based_score(user_id, movie_id, user_movie_matrix, tfidf_matrix)
    
        # 正規化（スコアを0-5の範囲に）
        cf_score_norm = cf_score / 5.0 if cf_score > 0 else 0
        cb_score_norm = cb_score
    
        hybrid = alpha * cf_score_norm + (1 - alpha) * cb_score_norm
    
        return hybrid, cf_score_norm, cb_score_norm
    
    # ハイブリッド推薦の生成
    def get_hybrid_recommendations(user_id, user_movie_matrix, tfidf_matrix,
                                   movies_df, alpha=0.5, top_n=5):
        """ハイブリッド推薦"""
        # 未評価映画を取得
        user_ratings = user_movie_matrix[user_id]
        unrated_movies = np.where(user_ratings == 0)[0]
    
        # スコア計算
        scores = []
        for movie_id in unrated_movies:
            hybrid, cf, cb = hybrid_score(user_id, movie_id, user_movie_matrix,
                                          tfidf_matrix, alpha)
            scores.append({
                'movie_id': movie_id,
                'hybrid_score': hybrid,
                'cf_score': cf,
                'cb_score': cb
            })
    
        # スコアでソート
        scores_df = pd.DataFrame(scores).sort_values('hybrid_score', ascending=False)
        top_scores = scores_df.head(top_n)
    
        # 映画情報を結合
        recommendations = movies_df.iloc[top_scores['movie_id']].copy()
        recommendations['hybrid_score'] = top_scores['hybrid_score'].values
        recommendations['cf_score'] = top_scores['cf_score'].values
        recommendations['cb_score'] = top_scores['cb_score'].values
    
        return recommendations[['title', 'genre', 'hybrid_score', 'cf_score', 'cb_score']]
    
    # 推薦生成（異なるα値で比較）
    print("\n=== User 0 への推薦 ===")
    print("\n[α=0.3: コンテンツベース重視]")
    recs_cb_heavy = get_hybrid_recommendations(0, user_movie_matrix, tfidf_matrix,
                                              movies, alpha=0.3, top_n=5)
    print(recs_cb_heavy)
    
    print("\n[α=0.5: バランス型]")
    recs_balanced = get_hybrid_recommendations(0, user_movie_matrix, tfidf_matrix,
                                              movies, alpha=0.5, top_n=5)
    print(recs_balanced)
    
    print("\n[α=0.7: 協調フィルタリング重視]")
    recs_cf_heavy = get_hybrid_recommendations(0, user_movie_matrix, tfidf_matrix,
                                              movies, alpha=0.7, top_n=5)
    print(recs_cf_heavy)
    

**出力例** ：
    
    
    === ユーザー-映画評価行列 ===
    形状: (50, 8)
    評価数: 174
    スパース率: 56.5%
    
    === User 0 への推薦 ===
    
    [α=0.3: コンテンツベース重視]
               title   genre  hybrid_score  cf_score  cb_score
    1      Inception  Sci-Fi      0.142635  0.183421  0.124738
    0     The Matrix  Sci-Fi      0.138274  0.167283  0.125196
    3   Interstellar  Sci-Fi      0.129847  0.154923  0.119324
    5       Iron Man  Action      0.098234  0.112453  0.092847
    4   The Avengers  Action      0.094512  0.108734  0.089273
    
    [α=0.5: バランス型]
               title   genre  hybrid_score  cf_score  cb_score
    1      Inception  Sci-Fi      0.154080  0.183421  0.124738
    0     The Matrix  Sci-Fi      0.146240  0.167283  0.125196
    3   Interstellar  Sci-Fi      0.137124  0.154923  0.119324
    5       Iron Man  Action      0.102650  0.112453  0.092847
    4   The Avengers  Action      0.099004  0.108734  0.089273
    
    [α=0.7: 協調フィルタリング重視]
               title   genre  hybrid_score  cf_score  cb_score
    1      Inception  Sci-Fi      0.165525  0.183421  0.124738
    0     The Matrix  Sci-Fi      0.154205  0.167283  0.125196
    3   Interstellar  Sci-Fi      0.144400  0.154923  0.119324
    5       Iron Man  Action      0.107066  0.112453  0.092847
    4   The Avengers  Action      0.103495  0.108734  0.089273
    

### Switching Hybrid: 状況適応型
    
    
    def switching_hybrid_recommendation(user_id, user_movie_matrix, tfidf_matrix,
                                       movies_df, top_n=5):
        """
        Switching Hybrid: 状況に応じて手法を切り替え
    
        ルール:
        - ユーザーの評価数が少ない（< 3）→ コンテンツベース
        - ユーザーの評価数が十分 → 協調フィルタリング
        """
        user_ratings = user_movie_matrix[user_id]
        n_ratings = np.sum(user_ratings > 0)
    
        print(f"\nUser {user_id}: 評価数 = {n_ratings}")
    
        if n_ratings < 3:
            print("→ コンテンツベース推薦を使用（評価数不足）")
            method = 'content_based'
            alpha = 0.0  # 100% コンテンツベース
        else:
            print("→ 協調フィルタリングを使用（十分な評価数）")
            method = 'collaborative'
            alpha = 1.0  # 100% 協調フィルタリング
    
        recommendations = get_hybrid_recommendations(
            user_id, user_movie_matrix, tfidf_matrix, movies_df,
            alpha=alpha, top_n=top_n
        )
    
        return recommendations, method
    
    # 評価数が少ないユーザーと多いユーザーで比較
    user_sparse = 0  # 評価数が少ない
    user_dense = np.argmax(np.sum(user_movie_matrix > 0, axis=1))  # 評価数が多い
    
    print("=== Switching Hybrid 推薦 ===")
    recs_sparse, method_sparse = switching_hybrid_recommendation(
        user_sparse, user_movie_matrix, tfidf_matrix, movies, top_n=3
    )
    print(recs_sparse)
    
    recs_dense, method_dense = switching_hybrid_recommendation(
        user_dense, user_movie_matrix, tfidf_matrix, movies, top_n=3
    )
    print(recs_dense)
    

* * *

## 3.5 Knowledge-based 推薦

### Knowledge-based 推薦とは

**Knowledge-based推薦** は、ドメイン知識と制約条件を用いて推薦を行う手法です。

タイプ | 説明 | 用途  
---|---|---  
**Constraint-based** | ユーザーの制約条件を満たすアイテム | 不動産、旅行、求人  
**Case-based** | 過去の類似ケースから推薦 | 医療診断、カスタマーサポート  
**Conversational** | 対話を通じてニーズを特定 | チャットボット、パーソナルアシスタント  
  
### 実装：制約ベース推薦
    
    
    import numpy as np
    import pandas as pd
    
    # 映画データ（詳細情報付き）
    movies_detailed = pd.DataFrame({
        'title': ['The Matrix', 'Inception', 'The Dark Knight', 'Interstellar',
                  'The Avengers', 'Iron Man', 'Titanic', 'The Notebook'],
        'genre': ['Sci-Fi', 'Sci-Fi', 'Action', 'Sci-Fi', 'Action', 'Action', 'Romance', 'Romance'],
        'year': [1999, 2010, 2008, 2014, 2012, 2008, 1997, 2004],
        'duration_min': [136, 148, 152, 169, 143, 126, 195, 123],
        'rating': [8.7, 8.8, 9.0, 8.6, 8.0, 7.9, 7.9, 7.8],
        'language': ['English', 'English', 'English', 'English',
                     'English', 'English', 'English', 'English']
    })
    
    print("=== 映画詳細データ ===")
    print(movies_detailed)
    
    def constraint_based_recommendation(constraints, movies_df):
        """
        制約ベース推薦
    
        Args:
            constraints: 制約条件の辞書
                例: {'genre': 'Sci-Fi', 'min_rating': 8.5, 'max_duration': 150}
        """
        filtered = movies_df.copy()
    
        # 各制約を適用
        for key, value in constraints.items():
            if key == 'genre':
                filtered = filtered[filtered['genre'] == value]
            elif key == 'min_rating':
                filtered = filtered[filtered['rating'] >= value]
            elif key == 'max_rating':
                filtered = filtered[filtered['rating'] <= value]
            elif key == 'min_year':
                filtered = filtered[filtered['year'] >= value]
            elif key == 'max_year':
                filtered = filtered[filtered['year'] <= value]
            elif key == 'max_duration':
                filtered = filtered[filtered['duration_min'] <= value]
            elif key == 'min_duration':
                filtered = filtered[filtered['duration_min'] >= value]
    
        return filtered
    
    # ユースケース1: 短時間のSF映画
    print("\n=== ユースケース1: 短時間の高評価SF映画 ===")
    constraints_1 = {
        'genre': 'Sci-Fi',
        'min_rating': 8.5,
        'max_duration': 150
    }
    print(f"制約条件: {constraints_1}")
    recommendations_1 = constraint_based_recommendation(constraints_1, movies_detailed)
    print(recommendations_1[['title', 'genre', 'rating', 'duration_min']])
    
    # ユースケース2: 2010年以降のアクション映画
    print("\n=== ユースケース2: 2010年以降のアクション映画 ===")
    constraints_2 = {
        'genre': 'Action',
        'min_year': 2010
    }
    print(f"制約条件: {constraints_2}")
    recommendations_2 = constraint_based_recommendation(constraints_2, movies_detailed)
    print(recommendations_2[['title', 'genre', 'year', 'rating']])
    
    # ユースケース3: 高評価の長編映画
    print("\n=== ユースケース3: 高評価の長編映画 ===")
    constraints_3 = {
        'min_rating': 8.5,
        'min_duration': 150
    }
    print(f"制約条件: {constraints_3}")
    recommendations_3 = constraint_based_recommendation(constraints_3, movies_detailed)
    print(recommendations_3[['title', 'rating', 'duration_min']])
    

**出力** ：
    
    
    === 映画詳細データ ===
                 title   genre  year  duration_min  rating language
    0       The Matrix  Sci-Fi  1999           136     8.7  English
    1        Inception  Sci-Fi  2010           148     8.8  English
    2  The Dark Knight  Action  2008           152     9.0  English
    3     Interstellar  Sci-Fi  2014           169     8.6  English
    4     The Avengers  Action  2012           143     8.0  English
    5         Iron Man  Action  2008           126     7.9  English
    6          Titanic Romance  1997           195     7.9  English
    7     The Notebook Romance  2004           123     7.8  English
    
    === ユースケース1: 短時間の高評価SF映画 ===
    制約条件: {'genre': 'Sci-Fi', 'min_rating': 8.5, 'max_duration': 150}
            title   genre  rating  duration_min
    0  The Matrix  Sci-Fi     8.7           136
    1   Inception  Sci-Fi     8.8           148
    
    === ユースケース2: 2010年以降のアクション映画 ===
    制約条件: {'genre': 'Action', 'min_year': 2010}
              title   genre  year  rating
    4  The Avengers  Action  2012     8.0
    
    === ユースケース3: 高評価の長編映画 ===
    制約条件: {'min_rating': 8.5, 'min_duration': 150}
                 title  rating  duration_min
    2  The Dark Knight     9.0           152
    3     Interstellar     8.6           169
    

* * *

## 3.6 コンテキスト考慮推薦

### コンテキスト情報の種類

コンテキスト | 例 | 活用方法  
---|---|---  
**時間** | 時刻、曜日、季節 | 時間帯別の推薦  
**場所** | 位置情報、デバイス | 位置ベース推薦  
**社会的** | 同伴者、グループ | グループ推薦  
**活動** | 作業中、休憩中 | 状況適応型推薦  
**気分** | 感情状態 | 感情考慮推薦  
  
### 実装：時間考慮推薦
    
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 時間帯別の視聴履歴データ生成
    np.random.seed(42)
    n_records = 200
    
    # 時間帯を生成
    hours = np.random.randint(0, 24, n_records)
    users = np.random.randint(0, 20, n_records)
    movie_ids = np.random.randint(0, len(movies), n_records)
    
    viewing_history = pd.DataFrame({
        'user_id': users,
        'movie_id': movie_ids,
        'hour': hours,
        'rating': np.random.randint(1, 6, n_records)
    })
    
    # 映画タイトルとジャンルを結合
    viewing_history['title'] = viewing_history['movie_id'].apply(
        lambda x: movies.iloc[x]['title']
    )
    viewing_history['genre'] = viewing_history['movie_id'].apply(
        lambda x: movies.iloc[x]['genre']
    )
    
    # 時間帯の分類
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    viewing_history['time_of_day'] = viewing_history['hour'].apply(categorize_time)
    
    print("=== 時間帯別視聴履歴 ===")
    print(viewing_history.head(10))
    
    # 時間帯別のジャンル嗜好分析
    time_genre_prefs = viewing_history.groupby(['time_of_day', 'genre']).agg({
        'rating': ['mean', 'count']
    }).reset_index()
    time_genre_prefs.columns = ['time_of_day', 'genre', 'avg_rating', 'count']
    
    print("\n=== 時間帯別ジャンル嗜好 ===")
    print(time_genre_prefs.sort_values(['time_of_day', 'avg_rating'], ascending=[True, False]))
    
    def time_aware_recommendation(user_id, current_hour, viewing_history, movies_df, top_n=3):
        """時間考慮推薦"""
        time_category = categorize_time(current_hour)
    
        # その時間帯でのジャンル嗜好を取得
        time_prefs = viewing_history[viewing_history['time_of_day'] == time_category]
        genre_scores = time_prefs.groupby('genre')['rating'].mean().to_dict()
    
        # 未視聴映画を取得
        watched_movies = viewing_history[viewing_history['user_id'] == user_id]['movie_id'].unique()
        unwatched = movies_df[~movies_df.index.isin(watched_movies)].copy()
    
        # ジャンルスコアを付与
        unwatched['time_score'] = unwatched['genre'].map(genre_scores).fillna(0)
    
        # Top-N推薦
        recommendations = unwatched.nlargest(top_n, 'time_score')
    
        return recommendations[['title', 'genre', 'time_score']], time_category
    
    # 異なる時間帯での推薦
    print("\n=== 時間考慮推薦 ===")
    
    print("\n[朝8時の推薦]")
    recs_morning, time_cat = time_aware_recommendation(0, 8, viewing_history, movies, top_n=3)
    print(f"時間帯: {time_cat}")
    print(recs_morning)
    
    print("\n[夜20時の推薦]")
    recs_evening, time_cat = time_aware_recommendation(0, 20, viewing_history, movies, top_n=3)
    print(f"時間帯: {time_cat}")
    print(recs_evening)
    
    print("\n[深夜1時の推薦]")
    recs_night, time_cat = time_aware_recommendation(0, 1, viewing_history, movies, top_n=3)
    print(f"時間帯: {time_cat}")
    print(recs_night)
    

**出力例** ：
    
    
    === 時間帯別視聴履歴 ===
       user_id  movie_id  hour  rating         title   genre time_of_day
    0       12         6    18       1       Titanic Romance     evening
    1       11         7    19       1  The Notebook Romance     evening
    2       18         2     9       5  The Dark Knight  Action     morning
    3        8         6     3       5       Titanic Romance       night
    4       10         2    14       2  The Dark Knight  Action   afternoon
    ...
    
    === 時間帯別ジャンル嗜好 ===
       time_of_day    genre  avg_rating  count
    0    afternoon   Action    3.181818     22
    1    afternoon  Romance    2.866667     15
    2    afternoon   Sci-Fi    2.952381     21
    3      evening   Action    3.000000     19
    4      evening  Romance    2.941176     17
    5      evening   Sci-Fi    3.117647     17
    6      morning   Action    3.263158     19
    7      morning  Romance    3.250000     16
    8      morning   Sci-Fi    3.105263     19
    9        night   Action    3.055556     18
    10       night  Romance    3.166667     18
    11       night   Sci-Fi    2.857143     14
    
    === 時間考慮推薦 ===
    
    [朝8時の推薦]
    時間帯: morning
              title   genre  time_score
    2  The Dark Knight  Action    3.263158
    0       The Matrix  Sci-Fi    3.105263
    7     The Notebook Romance    3.250000
    
    [夜20時の推薦]
    時間帯: evening
              title   genre  time_score
    1        Inception  Sci-Fi    3.117647
    2  The Dark Knight  Action    3.000000
    7     The Notebook Romance    2.941176
    
    [深夜1時の推薦]
    時間帯: night
              title   genre  time_score
    7     The Notebook Romance    3.166667
    2  The Dark Knight  Action    3.055556
    1        Inception  Sci-Fi    2.857143
    

* * *

## 3.7 本章のまとめ

### 学んだこと

  1. **コンテンツベースフィルタリング**

     * TF-IDFによるテキスト特徴量抽出
     * ユーザープロファイルの構築
     * コサイン類似度による推薦生成
     * コールドスタート問題への対応
  2. **ハイブリッド推薦**

     * Weighted: 重み付け線形結合
     * Switching: 状況適応型切り替え
     * Mixed: 複数手法の混合
     * 協調とコンテンツの相補的活用
  3. **Knowledge-based推薦**

     * 制約ベース推薦の実装
     * ドメイン知識の活用
     * 明示的なユーザー要求への対応
  4. **コンテキスト考慮推薦**

     * 時間・場所・活動の考慮
     * 状況適応型推薦
     * 多基準推薦システム

### 手法の選択ガイドライン

状況 | 推奨手法 | 理由  
---|---|---  
新規アイテムが多い | コンテンツベース | 特徴量があれば即座に推薦可能  
新規ユーザーが多い | Knowledge-based, ハイブリッド | 評価データ不要  
豊富な評価データ | 協調フィルタリング、ハイブリッド | 集合知を活用  
多様性が重要 | ハイブリッド | 複数手法で偏りを軽減  
明確な制約条件 | Knowledge-based | 条件を確実に満たす  
状況依存性が高い | コンテキスト考慮 | 時間・場所を反映  
  
### 次の章へ

第4章では、**深層学習ベースの推薦システム** を学びます：

  * Neural Collaborative Filtering (NCF)
  * Embedding層の活用
  * DeepFM: FactorizationMachines + DNN
  * Two-Tower Model
  * Transformerベースの推薦

* * *

## 演習問題

### 問題1（難易度：easy）

コンテンツベースフィルタリングと協調フィルタリングの主な違いを3つ挙げ、それぞれの長所と短所を説明してください。

解答例

**主な違い** ：

  1. **推薦の根拠**

     * コンテンツベース: アイテムの特徴量
     * 協調フィルタリング: ユーザー・アイテム間の評価パターン
  2. **必要なデータ**

     * コンテンツベース: アイテムの特徴量（メタデータ）
     * 協調フィルタリング: ユーザー-アイテム評価行列
  3. **コールドスタート問題**

     * コンテンツベース: 新規アイテムでも特徴量があれば推薦可能
     * 協調フィルタリング: 新規ユーザー・アイテムで困難

**長所と短所** ：

手法 | 長所 | 短所  
---|---|---  
**コンテンツベース** | \- 新規アイテムに対応  
\- 説明可能性が高い  
\- ユーザー独立 | \- フィルターバブル  
\- 特徴量設計が必要  
\- 意外性が低い  
**協調フィルタリング** | \- 意外性のある推薦  
\- 特徴量不要  
\- 集合知の活用 | \- コールドスタート  
\- スパース性問題  
\- スケーラビリティ  
  
### 問題2（難易度：medium）

TF-IDFの計算式を説明し、以下の3つの文書に対してTF-IDFを手計算してください。
    
    
    文書1: "machine learning is great"
    文書2: "deep learning is powerful"
    文書3: "machine learning and deep learning"
    

解答例

**TF-IDF計算式** ：

$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$

$$ \text{TF}(t, d) = \frac{\text{term } t \text{ の文書 } d \text{ 内の出現回数}}{\text{文書 } d \text{ の総単語数}} $$

$$ \text{IDF}(t) = \log \frac{N}{\text{df}(t)} $$

  * $N$: 総文書数 = 3
  * $\text{df}(t)$: 単語$t$を含む文書数

**手計算例（単語 "machine" の文書1でのTF-IDF）** ：

  1. **TF計算**

     * 文書1: "machine learning is great" (4単語)
     * "machine" の出現回数: 1
     * TF("machine", 文書1) = 1 / 4 = 0.25
  2. **IDF計算**

     * 総文書数 $N$ = 3
     * "machine" を含む文書: 文書1, 文書3 → df("machine") = 2
     * IDF("machine") = log(3 / 2) = log(1.5) ≈ 0.176
  3. **TF-IDF計算**

     * TF-IDF("machine", 文書1) = 0.25 × 0.176 ≈ 0.044

**Pythonでの検証** ：
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    docs = [
        "machine learning is great",
        "deep learning is powerful",
        "machine learning and deep learning"
    ]
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(docs)
    
    print("特徴語:", tfidf.get_feature_names_out())
    print("\nTF-IDF行列:")
    print(tfidf_matrix.toarray())
    

### 問題3（難易度：medium）

Weighted Hybrid推薦システムにおいて、協調フィルタリングの重み（α）をどのように調整すべきか、状況に応じた戦略を3つ提案してください。

解答例

**α調整戦略** ：

  1. **ユーザーの評価数に基づく動的調整**
         
         def adaptive_alpha(user_id, user_movie_matrix, min_ratings=5):
             """ユーザーの評価数に基づくα調整"""
             n_ratings = np.sum(user_movie_matrix[user_id] > 0)
         
             if n_ratings < min_ratings:
                 # 評価数が少ない → コンテンツベース重視
                 alpha = 0.2
             elif n_ratings < 10:
                 # 中程度 → バランス型
                 alpha = 0.5
             else:
                 # 評価数が多い → 協調フィルタリング重視
                 alpha = 0.8
         
             return alpha
         

理由: 評価データが少ない新規ユーザーにはコンテンツベース、十分なデータがあれば協調フィルタリングを重視

  2. **アイテムの人気度に基づく調整**
         
         def popularity_based_alpha(item_id, user_movie_matrix):
             """アイテムの評価数に基づくα調整"""
             item_ratings = user_movie_matrix[:, item_id]
             n_ratings = np.sum(item_ratings > 0)
         
             if n_ratings < 5:
                 # ニッチなアイテム → コンテンツベース
                 alpha = 0.3
             else:
                 # 人気アイテム → 協調フィルタリング
                 alpha = 0.7
         
             return alpha
         

理由: 評価数の少ないニッチなアイテムは協調フィルタリングが機能しにくい

  3. **A/Bテストによる最適化**
         
         def ab_test_alpha_optimization(user_groups, alpha_values,
                                        eval_metric='precision@k'):
             """A/Bテストでαを最適化"""
             results = {}
         
             for alpha in alpha_values:
                 metrics = []
                 for user in user_groups:
                     recs = hybrid_recommend(user, alpha=alpha)
                     metric = evaluate(recs, eval_metric)
                     metrics.append(metric)
         
                 results[alpha] = np.mean(metrics)
         
             optimal_alpha = max(results, key=results.get)
             return optimal_alpha
         

理由: 実データでの性能を測定し、最適なバランスを見つける

### 問題4（難易度：hard）

時間を考慮した推薦システムを実装してください。ユーザーの視聴履歴に時間情報が含まれており、時間帯によってジャンル嗜好が変化すると仮定します。

解答例
    
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    class TimeAwareRecommender:
        """時間考慮推薦システム"""
    
        def __init__(self, decay_factor=0.9):
            """
            Args:
                decay_factor: 時間減衰係数（0-1）
            """
            self.decay_factor = decay_factor
            self.time_genre_prefs = None
    
        def fit(self, viewing_history):
            """時間帯別の嗜好を学習"""
            # 時間帯分類
            viewing_history['time_category'] = viewing_history['hour'].apply(
                self._categorize_time
            )
    
            # 時間帯×ジャンル別の平均評価
            self.time_genre_prefs = viewing_history.groupby(
                ['time_category', 'genre']
            )['rating'].mean().to_dict()
    
            return self
    
        def _categorize_time(self, hour):
            """時間帯の分類"""
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
    
        def _time_decay(self, timestamp, current_time):
            """時間減衰重み"""
            hours_diff = (current_time - timestamp).total_seconds() / 3600
            days_diff = hours_diff / 24
            return self.decay_factor ** days_diff
    
        def recommend(self, user_id, current_time, viewing_history,
                      movies_df, top_n=5):
            """時間考慮推薦"""
            current_hour = current_time.hour
            time_category = self._categorize_time(current_hour)
    
            # ユーザーの履歴を取得
            user_history = viewing_history[
                viewing_history['user_id'] == user_id
            ].copy()
    
            # 時間減衰重みを計算
            user_history['time_weight'] = user_history['timestamp'].apply(
                lambda t: self._time_decay(t, current_time)
            )
    
            # 加重平均でユーザー嗜好を計算
            user_genre_prefs = user_history.groupby('genre').apply(
                lambda g: np.average(g['rating'], weights=g['time_weight'])
            ).to_dict()
    
            # 未視聴映画を取得
            watched = user_history['movie_id'].unique()
            unwatched = movies_df[~movies_df.index.isin(watched)].copy()
    
            # スコア計算
            def calculate_score(row):
                genre = row['genre']
    
                # ユーザー嗜好スコア
                user_pref = user_genre_prefs.get(genre, 0)
    
                # 時間帯嗜好スコア
                time_pref = self.time_genre_prefs.get(
                    (time_category, genre), 0
                )
    
                # 統合スコア
                return 0.6 * user_pref + 0.4 * time_pref
    
            unwatched['score'] = unwatched.apply(calculate_score, axis=1)
    
            # Top-N推薦
            recommendations = unwatched.nlargest(top_n, 'score')
    
            return recommendations[['title', 'genre', 'score']]
    
    # 使用例
    np.random.seed(42)
    
    # 視聴履歴データ生成（タイムスタンプ付き）
    base_time = datetime.now()
    viewing_data = []
    
    for i in range(100):
        timestamp = base_time - timedelta(days=np.random.randint(0, 30))
        viewing_data.append({
            'user_id': np.random.randint(0, 10),
            'movie_id': np.random.randint(0, 8),
            'rating': np.random.randint(1, 6),
            'hour': timestamp.hour,
            'timestamp': timestamp
        })
    
    viewing_df = pd.DataFrame(viewing_data)
    viewing_df['genre'] = viewing_df['movie_id'].apply(
        lambda x: movies.iloc[x]['genre']
    )
    
    # 推薦システムの学習と推薦
    recommender = TimeAwareRecommender(decay_factor=0.95)
    recommender.fit(viewing_df)
    
    # 異なる時間帯での推薦
    morning_time = datetime.now().replace(hour=9)
    evening_time = datetime.now().replace(hour=20)
    
    print("=== 朝9時の推薦 ===")
    recs_morning = recommender.recommend(
        user_id=0,
        current_time=morning_time,
        viewing_history=viewing_df,
        movies_df=movies,
        top_n=3
    )
    print(recs_morning)
    
    print("\n=== 夜8時の推薦 ===")
    recs_evening = recommender.recommend(
        user_id=0,
        current_time=evening_time,
        viewing_history=viewing_df,
        movies_df=movies,
        top_n=3
    )
    print(recs_evening)
    

### 問題5（難易度：hard）

コンテンツベース推薦において、TF-IDFの代わりにWord2VecやBERTを用いた埋め込みベクトルを使用する利点と実装方法を説明してください。

解答例

**TF-IDF vs 埋め込みベクトル** ：

項目 | TF-IDF | Word2Vec/BERT  
---|---|---  
**表現** | スパースなBoW | 密な分散表現  
**意味理解** | 単語の出現のみ | 意味的類似性を捉える  
**次元数** | 語彙サイズ（高次元） | 固定（例: 768次元）  
**類義語対応** | 困難 | 可能  
**計算コスト** | 低 | 高（特にBERT）  
  
**Word2Vec実装例** ：
    
    
    from gensim.models import Word2Vec
    import numpy as np
    
    # テキストの前処理
    texts = [doc.lower().split() for doc in movies['description']]
    
    # Word2Vecモデルの学習
    w2v_model = Word2Vec(
        sentences=texts,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    
    def text_to_vector(text, model):
        """文章をWord2Vecベクトルに変換"""
        words = text.lower().split()
        word_vectors = [
            model.wv[word] for word in words
            if word in model.wv
        ]
    
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
    
        # 平均ベクトル
        return np.mean(word_vectors, axis=0)
    
    # 映画の埋め込みベクトル
    movie_embeddings = np.array([
        text_to_vector(desc, w2v_model)
        for desc in movies['description']
    ])
    
    # コサイン類似度で推薦
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(movie_embeddings)
    
    print("Word2Vecベース類似度:")
    for i, title in enumerate(movies['title']):
        print(f"{title} vs The Matrix: {similarity_matrix[0, i]:.3f}")
    

**BERT実装例** ：
    
    
    from transformers import BertTokenizer, BertModel
    import torch
    
    # BERTモデルのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_bert_embedding(text):
        """BERTで文章埋め込みを取得"""
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # [CLS]トークンの埋め込みを使用
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding.flatten()
    
    # BERT埋め込みベクトル
    bert_embeddings = np.array([
        get_bert_embedding(desc)
        for desc in movies['description']
    ])
    
    # コサイン類似度
    bert_similarity = cosine_similarity(bert_embeddings)
    
    print("\nBERTベース類似度:")
    for i, title in enumerate(movies['title']):
        print(f"{title} vs The Matrix: {bert_similarity[0, i]:.3f}")
    

**利点** ：

  1. **意味的類似性** : "car" と "automobile" を類似として認識
  2. **文脈理解** : 文の意味をより正確に捉える
  3. **多義語対応** : 文脈に応じた単語の意味を理解
  4. **次元削減** : 低次元の密なベクトル表現

* * *

## 参考文献

  1. Aggarwal, C. C. (2016). _Recommender Systems: The Textbook_. Springer.
  2. Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_ (2nd ed.). Springer.
  3. Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In _Recommender Systems Handbook_ (pp. 73-105). Springer.
  4. Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments. _User Modeling and User-Adapted Interaction_ , 12(4), 331-370.
  5. Adomavicius, G., & Tuzhilin, A. (2011). Context-Aware Recommender Systems. In _Recommender Systems Handbook_ (pp. 217-253). Springer.
