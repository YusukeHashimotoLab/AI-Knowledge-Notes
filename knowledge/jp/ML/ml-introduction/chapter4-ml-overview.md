---
title: 第4章：機械学習の概要
chapter_title: 第4章：機械学習の概要
---

**機械学習の世界へようこそ - データから学ぶAIの基礎**

## はじめに

**機械学習（Machine Learning）** は、データからパターンを自動的に学習し、予測や判断を行う技術です。これまで学んだPython、NumPy、Pandasの知識を使って、いよいよ機械学習の世界に踏み出しましょう。

この章では、以下の内容を学びます：

  * 機械学習とは何か - 定義と応用例
  * 教師あり学習（回帰・分類）
  * 教師なし学習（クラスタリング・次元削減）
  * 訓練データとテストデータの分割
  * scikit-learnの基本的な使い方
  * 実践的な機械学習モデルの実装

> **機械学習の定義**  
>  「明示的にプログラムされることなく、経験（データ）から学習する能力をコンピュータに与える研究分野」- Arthur Samuel (1959) 

## 1\. 機械学習とは

### 1.1 機械学習の定義

機械学習は、大量のデータから自動的にパターンを見つけ出し、そのパターンを使って新しいデータに対して予測や判断を行う技術です。

**従来のプログラミング vs 機械学習**

項目 | 従来のプログラミング | 機械学習  
---|---|---  
ルール | 人間が明示的に記述 | データから自動的に学習  
入力 | データ + ルール | データ + 正解（ラベル）  
出力 | 処理結果 | 学習されたモデル  
適用例 | 計算、データ処理 | 予測、認識、推薦  
      
    
    ```mermaid
    graph LR
        A[従来のプログラミング] --> B[データ + ルール]
        B --> C[出力]
    
        D[機械学習] --> E[データ + 正解]
        E --> F[学習]
        F --> G[モデル]
        G --> H[予測]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style G fill:#e8f5e9
    ```

### 1.2 機械学習の応用例

  * **画像認識** : 顔認識、物体検出、医療画像診断
  * **自然言語処理** : 翻訳、チャットボット、感情分析
  * **音声認識** : 音声アシスタント（Siri、Alexa）
  * **推薦システム** : 動画・音楽・商品の推薦（Netflix、Amazon）
  * **金融** : 株価予測、不正検出、信用スコアリング
  * **医療** : 病気の診断、創薬、ゲノム解析
  * **自動運転** : 障害物検出、経路計画

## 2\. 機械学習の種類
    
    
    ```mermaid
    graph TD
        A[機械学習] --> B[教師あり学習]
        A --> C[教師なし学習]
        A --> D[強化学習]
    
        B --> E[分類]
        B --> F[回帰]
    
        C --> G[クラスタリング]
        C --> H[次元削減]
    
        D --> I[エージェント学習]
    
        E --> J["例: スパム判定"]
        F --> K["例: 住宅価格予測"]
        G --> L["例: 顧客セグメンテーション"]
        H --> M["例: データ可視化"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

### 2.1 教師あり学習（Supervised Learning）

正解ラベル付きのデータから学習し、新しいデータに対して予測を行います。

  * **分類（Classification）** : カテゴリを予測（例: スパムか否か、病気の診断）
  * **回帰（Regression）** : 数値を予測（例: 住宅価格、売上予測）

### 2.2 教師なし学習（Unsupervised Learning）

正解ラベルなしのデータから、隠れたパターンや構造を発見します。

  * **クラスタリング（Clustering）** : データをグループに分ける（例: 顧客セグメンテーション）
  * **次元削減（Dimensionality Reduction）** : データの特徴を減らす（例: データ可視化）

### 2.3 強化学習（Reinforcement Learning）

環境との相互作用を通じて、報酬を最大化する行動を学習します（例: ゲームAI、ロボット制御）。

## 3\. 教師あり学習の基礎

### 3.1 分類と回帰の違い

項目 | 分類（Classification） | 回帰（Regression）  
---|---|---  
予測対象 | カテゴリ（離散値） | 数値（連続値）  
例 | 「犬」か「猫」か | 住宅価格は「450万円」  
評価指標 | 正解率、F1スコア | 平均二乗誤差（MSE）  
代表的な手法 | ロジスティック回帰、決定木 | 線形回帰、多項式回帰  
  
#### 例1：分類と回帰の違い
    
    
    import numpy as np
    import pandas as pd
    
    # 分類の例: アヤメの品種分類
    # 入力: 花びらの長さ、幅 → 出力: 品種（Setosa, Versicolor, Virginica）
    classification_data = {
        '花びらの長さ': [1.4, 4.7, 5.1],
        '花びらの幅': [0.2, 1.4, 2.3],
        '品種': ['Setosa', 'Versicolor', 'Virginica']  # カテゴリ
    }
    print("分類データ:")
    print(pd.DataFrame(classification_data))
    
    # 回帰の例: 住宅価格予測
    # 入力: 面積、部屋数 → 出力: 価格（数値）
    regression_data = {
        '面積（㎡）': [50, 70, 90],
        '部屋数': [2, 3, 4],
        '価格（万円）': [3000, 4200, 5500]  # 連続値
    }
    print("\n回帰データ:")
    print(pd.DataFrame(regression_data))
    

## 4\. 訓練データとテストデータ

### 4.1 なぜ分割が必要か？

機械学習モデルは、**訓練データ** で学習し、**テストデータ** で評価します。同じデータで学習と評価を行うと、モデルの真の性能が分かりません。

  * **訓練データ（Training Data）** : モデルの学習に使用（通常70-80%）
  * **テストデータ（Test Data）** : モデルの評価に使用（通常20-30%）

#### 例2：データの分割
    
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # サンプルデータ
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    print("元のデータ:")
    print("X (特徴量):")
    print(X)
    print("y (ラベル):", y)
    
    # 訓練データとテストデータに分割（80%訓練、20%テスト）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n訓練データ:")
    print("X_train:")
    print(X_train)
    print("y_train:", y_train)
    
    print("\nテストデータ:")
    print("X_test:")
    print(X_test)
    print("y_test:", y_test)
    
    print("\nデータサイズ:")
    print(f"訓練データ: {len(X_train)}個, テストデータ: {len(X_test)}個")
    
    
    
    ```mermaid
    graph LR
        A[全データ] --> B[訓練データ 80%]
        A --> C[テストデータ 20%]
    
        B --> D[学習]
        D --> E[モデル]
    
        E --> F[評価]
        C --> F
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style E fill:#e8f5e9
    ```

## 5\. scikit-learnの基本

**scikit-learn** は、Pythonで最も広く使われている機械学習ライブラリです。

### 5.1 scikit-learnの基本的なワークフロー

  1. データの準備
  2. モデルの選択と作成
  3. モデルの訓練（`fit`）
  4. 予測（`predict`）
  5. 評価（`score`）

#### 例3：scikit-learnの基本的な使い方
    
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    # 1. データの準備（Irisデータセット）
    iris = load_iris()
    X = iris.data  # 特徴量（花びらの長さ・幅など）
    y = iris.target  # ラベル（品種）
    
    print("データの形状:")
    print("X:", X.shape)  # (150, 4) = 150サンプル、4特徴量
    print("y:", y.shape)  # (150,) = 150ラベル
    
    # 2. 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\n訓練データ: {}個".format(len(X_train)))
    print("テストデータ: {}個".format(len(X_test)))
    
    # 3. モデルの作成（k-近傍法）
    model = KNeighborsClassifier(n_neighbors=3)
    
    # 4. モデルの訓練
    model.fit(X_train, y_train)
    print("\nモデルの訓練完了")
    
    # 5. 予測
    y_pred = model.predict(X_test)
    print("\n予測結果（最初の10個）:", y_pred[:10])
    print("正解ラベル（最初の10個）:", y_test[:10])
    
    # 6. 評価
    accuracy = accuracy_score(y_test, y_pred)
    print("\n正解率: {:.2f}%".format(accuracy * 100))
    
    # または
    score = model.score(X_test, y_test)
    print("スコア: {:.2f}%".format(score * 100))
    

## 6\. 回帰問題の実装

### 6.1 線形回帰

線形回帰は、入力と出力の関係を直線で近似するモデルです。

数式: \\( y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b \\)

#### 例4：線形回帰の実装
    
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプルデータの作成（住宅価格予測）
    np.random.seed(42)
    n_samples = 100
    
    # 特徴量: 面積（50-150㎡）
    area = np.random.uniform(50, 150, n_samples)
    
    # 目的変数: 価格 = 30 * 面積 + ノイズ
    price = 30 * area + np.random.normal(0, 200, n_samples)
    
    # データを2次元配列に変換
    X = area.reshape(-1, 1)
    y = price
    
    print("データサンプル:")
    for i in range(5):
        print(f"面積: {X[i][0]:.1f}㎡ → 価格: {y[i]:.0f}万円")
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデルの作成と訓練
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("\n学習済みパラメータ:")
    print(f"傾き（係数）: {model.coef_[0]:.2f}")
    print(f"切片: {model.intercept_:.2f}")
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n評価指標:")
    print(f"平均二乗誤差（MSE）: {mse:.2f}")
    print(f"平方根平均二乗誤差（RMSE）: {rmse:.2f}")
    print(f"決定係数（R²）: {r2:.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.5, label='実際の価格')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='予測')
    plt.xlabel('面積（㎡）')
    plt.ylabel('価格（万円）')
    plt.title('住宅価格予測（線形回帰）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('linear_regression.png')
    # plt.show()
    
    print("\nグラフを作成しました。")
    

## 7\. 分類問題の実装

### 7.1 ロジスティック回帰

ロジスティック回帰は、2値分類（0か1か）を行うための基本的な手法です。

#### 例5：ロジスティック回帰の実装
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.datasets import load_iris
    import numpy as np
    
    # データの準備（Irisデータセット、2クラスのみ使用）
    iris = load_iris()
    X = iris.data[:100]  # 最初の100サンプル（2クラス分）
    y = iris.target[:100]
    
    print("データの形状:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("クラス:", np.unique(y))
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # モデルの作成と訓練
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 予測確率
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n予測結果（最初の5個）:")
    for i in range(5):
        print(f"予測: {y_pred[i]}, 正解: {y_test[i]}, 確率: {y_pred_proba[i]}")
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n正解率: {accuracy:.2%}")
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("\n混同行列:")
    print(cm)
    
    # 詳細レポート
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    
    
    ```mermaid
    graph TD
        A[分類評価] --> B[混同行列]
        B --> C[TP: 真陽性]
        B --> D[TN: 真陰性]
        B --> E[FP: 偽陽性]
        B --> F[FN: 偽陰性]
    
        A --> G[評価指標]
        G --> H["正解率 = (TP+TN)/全体"]
        G --> I["精度 = TP/(TP+FP)"]
        G --> J["再現率 = TP/(TP+FN)"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style G fill:#f3e5f5
    ```

## 8\. 実践例：Irisデータセットで多クラス分類

#### 例6：3クラス分類の完全実装
    
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    import pandas as pd
    import numpy as np
    
    # データの読み込み
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("=== Irisデータセット ===")
    print("特徴量名:", iris.feature_names)
    print("クラス名:", iris.target_names)
    print("データ形状:", X.shape)
    
    # データをDataFrameに変換
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({
        0: 'setosa', 1: 'versicolor', 2: 'virginica'
    })
    
    print("\nデータの最初の5行:")
    print(df.head())
    
    print("\nクラスごとのサンプル数:")
    print(df['species_name'].value_counts())
    
    # 統計量
    print("\n特徴量の統計:")
    print(df.describe())
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n訓練データ: {len(X_train)}個")
    print(f"テストデータ: {len(X_test)}個")
    
    # モデルの作成（決定木）
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n正解率: {accuracy:.2%}")
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("\n混同行列:")
    cm_df = pd.DataFrame(
        cm,
        index=['setosa', 'versicolor', 'virginica'],
        columns=['setosa', 'versicolor', 'virginica']
    )
    print(cm_df)
    
    # クラスごとの正解率
    print("\nクラスごとの結果:")
    for i, name in enumerate(iris.target_names):
        class_mask = (y_test == i)
        class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
        print(f"{name}: {class_accuracy:.2%}")
    
    # 特徴量の重要度
    print("\n特徴量の重要度:")
    feature_importance = pd.DataFrame({
        'feature': iris.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    

## 9\. 教師なし学習の基礎

### 9.1 クラスタリング

クラスタリングは、似たデータを自動的にグループ化する手法です。

#### 例7：K-Meansクラスタリング
    
    
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    import numpy as np
    
    # データの準備
    iris = load_iris()
    X = iris.data[:, :2]  # 最初の2特徴量のみ使用（可視化のため）
    
    # K-Meansクラスタリング（3つのクラスタ）
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    print("クラスタリング結果:")
    print("各サンプルが属するクラスタ:", clusters)
    
    # クラスタごとのサンプル数
    unique, counts = np.unique(clusters, return_counts=True)
    print("\nクラスタごとのサンプル数:")
    for cluster, count in zip(unique, counts):
        print(f"クラスタ {cluster}: {count}個")
    
    # クラスタ中心
    print("\nクラスタ中心:")
    print(kmeans.cluster_centers_)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               s=300, c='red', marker='X', edgecolors='black',
               label='中心')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('K-Meansクラスタリング')
    plt.colorbar(scatter)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('kmeans_clustering.png')
    # plt.show()
    
    print("\nクラスタリング完了。")
    

### 9.2 次元削減

次元削減は、データの特徴量を減らして可視化や処理を簡単にする手法です。

#### 例8：PCA（主成分分析）
    
    
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    
    # データの準備
    iris = load_iris()
    X = iris.data  # 4次元
    y = iris.target
    
    print("元のデータ形状:", X.shape)  # (150, 4)
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print("削減後の形状:", X_pca.shape)  # (150, 2)
    
    # 寄与率
    print("\n各主成分の寄与率:")
    print(pca.explained_variance_ratio_)
    print(f"累積寄与率: {sum(pca.explained_variance_ratio_):.2%}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        mask = (y == i)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=iris.target_names[i], alpha=0.6)
    
    plt.xlabel(f'第1主成分 (寄与率: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'第2主成分 (寄与率: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCAによる次元削減')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('pca_visualization.png')
    # plt.show()
    
    print("\nPCA完了。")
    

## 10\. 機械学習プロジェクトの流れ
    
    
    ```mermaid
    graph TD
        A[問題定義] --> B[データ収集]
        B --> C[データ探索・可視化]
        C --> D[データ前処理]
        D --> E[特徴量エンジニアリング]
        E --> F[モデル選択]
        F --> G[訓練・検証]
        G --> H{性能OK?}
        H -->|No| I[ハイパーパラメータ調整]
        I --> G
        H -->|Yes| J[テスト]
        J --> K[デプロイ]
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style G fill:#f3e5f5
        style K fill:#e8f5e9
    ```

#### 例9：完全な機械学習ワークフロー
    
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    import numpy as np
    
    print("=== 機械学習プロジェクトの完全ワークフロー ===\n")
    
    # 1. データ収集
    print("1. データ収集")
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"データサイズ: {X.shape}")
    
    # 2. データ探索
    print("\n2. データ探索")
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    print(df.describe())
    print("\nクラス分布:")
    print(df['target'].value_counts())
    
    # 3. データ前処理
    print("\n3. データ前処理")
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("標準化完了")
    
    # 4. モデル選択
    print("\n4. モデル選択と訓練")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("訓練完了")
    
    # 5. 交差検証
    print("\n5. 交差検証（5-fold）")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"各Foldのスコア: {cv_scores}")
    print(f"平均スコア: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 6. テストデータで評価
    print("\n6. テストデータで評価")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"テスト精度: {accuracy:.2%}")
    
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred,
                              target_names=iris.target_names))
    
    # 7. 特徴量の重要度
    print("7. 特徴量の重要度")
    feature_importance = pd.DataFrame({
        'feature': iris.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    print("\n=== プロジェクト完了 ===")
    

#### 例10：過学習と汎化性能
    
    
    from sklearn.model_selection import learning_curve
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt
    
    # データの準備
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 異なる深さの決定木で比較
    depths = [1, 2, 3, 5, 10, 20]
    
    results = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
        # 学習曲線の計算
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
        )
    
        train_mean = train_scores.mean(axis=1)[-1]
        val_mean = val_scores.mean(axis=1)[-1]
    
        results.append({
            'depth': depth,
            'train_score': train_mean,
            'val_score': val_mean,
            'overfitting': train_mean - val_mean
        })
    
    # 結果を表示
    print("深さと過学習の関係:")
    print("=" * 60)
    for r in results:
        print(f"深さ{r['depth']:2d}: 訓練={r['train_score']:.3f}, "
              f"検証={r['val_score']:.3f}, "
              f"過学習度={r['overfitting']:.3f}")
    
    # 最適な深さを見つける
    best = max(results, key=lambda x: x['val_score'])
    print(f"\n最適な深さ: {best['depth']}")
    print(f"検証スコア: {best['val_score']:.3f}")
    
    # 可視化
    depths_list = [r['depth'] for r in results]
    train_scores = [r['train_score'] for r in results]
    val_scores = [r['val_score'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths_list, train_scores, 'o-', label='訓練スコア')
    plt.plot(depths_list, val_scores, 's-', label='検証スコア')
    plt.xlabel('決定木の深さ')
    plt.ylabel('スコア')
    plt.title('モデルの複雑さと性能の関係')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('overfitting_analysis.png')
    # plt.show()
    
    print("\n過学習分析完了。")
    

## まとめ

この章では、機械学習の基礎を学びました：

  * ✅ **機械学習の定義** : データから学習し予測を行う技術
  * ✅ **種類** : 教師あり学習、教師なし学習、強化学習
  * ✅ **教師あり学習** : 分類（カテゴリ予測）と回帰（数値予測）
  * ✅ **データ分割** : 訓練データとテストデータ
  * ✅ **scikit-learn** : fit, predict, scoreの基本ワークフロー
  * ✅ **実装** : 線形回帰、ロジスティック回帰、決定木
  * ✅ **教師なし学習** : クラスタリング、次元削減
  * ✅ **評価** : 正解率、混同行列、過学習

**次のステップ** : このシリーズを完了したあなたは、より専門的な機械学習シリーズ（教師あり学習入門、ニューラルネットワーク入門）へ進む準備ができています！

## 演習問題

演習1：データ分割の理解

**問題** : 100個のサンプルを訓練80%、テスト20%に分割し、各セットのサイズを確認してください。さらに、`stratify`パラメータの効果を確認してください。
    
    
    # 解答例
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # データの作成（不均衡なクラス）
    X = np.arange(100).reshape(-1, 1)
    y = np.array([0]*30 + [1]*70)  # クラス0: 30個、クラス1: 70個
    
    print("元のクラス分布:", np.bincount(y))
    
    # stratifyなし
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nstratifyなし:")
    print("訓練:", np.bincount(y_train))
    print("テスト:", np.bincount(y_test))
    
    # stratifyあり
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nstratifyあり:")
    print("訓練:", np.bincount(y_train))
    print("テスト:", np.bincount(y_test))
    

演習2：回帰モデルの比較

**問題** : 線形回帰と多項式回帰（2次）を比較し、どちらがデータに適合しているか評価してください。
    
    
    # 解答例
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    # 非線形データの生成
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 0.5 * X**2 + X + 2 + np.random.normal(0, 5, (100, 1)).flatten()
    
    # 線形回帰
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)
    
    # 多項式回帰（2次）
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    
    # 評価
    print("線形回帰:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_linear)):.2f}")
    print(f"R²: {r2_score(y, y_pred_linear):.3f}")
    
    print("\n多項式回帰:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_poly)):.2f}")
    print(f"R²: {r2_score(y, y_pred_poly):.3f}")
    

演習3：分類モデルの評価

**問題** : Irisデータセットで、k-NN（k=3）と決定木を比較し、混同行列と正解率を出力してください。
    
    
    # 解答例
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # データの準備
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    print("k-NN (k=3):")
    print(f"正解率: {accuracy_score(y_test, y_pred_knn):.2%}")
    print("混同行列:")
    print(confusion_matrix(y_test, y_pred_knn))
    
    # 決定木
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    
    print("\n決定木:")
    print(f"正解率: {accuracy_score(y_test, y_pred_tree):.2%}")
    print("混同行列:")
    print(confusion_matrix(y_test, y_pred_tree))
    

演習4：クラスタリングの最適なクラスタ数

**問題** : K-Meansでクラスタ数を2-10まで変化させ、エルボー法で最適なクラスタ数を見つけてください。
    
    
    # 解答例
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    
    # データの準備
    iris = load_iris()
    X = iris.data
    
    # クラスタ数を変化させて慣性を計算
    inertias = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # エルボー法で可視化
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('クラスタ数 k')
    plt.ylabel('慣性（Inertia）')
    plt.title('エルボー法によるクラスタ数の決定')
    plt.grid(True, alpha=0.3)
    # plt.savefig('elbow_method.png')
    # plt.show()
    
    print("慣性の値:")
    for k, inertia in zip(k_range, inertias):
        print(f"k={k}: {inertia:.2f}")
    

演習5：総合問題 - 完全なMLパイプライン

**問題** : Irisデータセットを使って、(1) データ分割、(2) 標準化、(3) モデル訓練、(4) 交差検証、(5) テスト評価の完全なパイプラインを実装してください。
    
    
    # 解答例
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    # (1) データ分割
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("(1) データ分割:")
    print(f"訓練: {X_train.shape}, テスト: {X_test.shape}")
    
    # (2) 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n(2) 標準化:")
    print(f"訓練データ平均: {X_train_scaled.mean(axis=0)}")
    print(f"訓練データ標準偏差: {X_train_scaled.std(axis=0)}")
    
    # (3) モデル訓練
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("\n(3) モデル訓練完了")
    
    # (4) 交差検証
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print("\n(4) 交差検証:")
    print(f"各Foldのスコア: {cv_scores}")
    print(f"平均: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # (5) テスト評価
    y_pred = model.predict(X_test_scaled)
    
    print("\n(5) テスト評価:")
    print(f"テスト精度: {model.score(X_test_scaled, y_test):.2%}")
    print("\n混同行列:")
    print(confusion_matrix(y_test, y_pred))
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred,
                              target_names=iris.target_names))
    

[← 第3章: Pandas基礎](<./chapter3-pandas-basics.html>) [シリーズトップへ](<./index.html>)
