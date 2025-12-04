---
title: 第2章：カテゴリカル変数エンコーディング
chapter_title: 第2章：カテゴリカル変数エンコーディング
subtitle: 機械学習モデルのための質的データの数値変換技術
reading_time: 20-25分
difficulty: 初級〜中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ カテゴリカル変数の種類と特性を理解する
  * ✅ One-Hot Encodingの原理と実装ができる
  * ✅ Label EncodingとOrdinal Encodingの違いを説明できる
  * ✅ Target Encoding（Mean Encoding）を適切に使える
  * ✅ Frequency Encodingの概念と応用を理解する
  * ✅ Binary EncodingとHashing Trickを実装できる
  * ✅ 各エンコーディング手法を適切に使い分けられる

* * *

## 2.1 カテゴリカル変数とは

### 定義

**カテゴリカル変数（Categorical Variable）** は、質的なデータを表す変数で、離散的なカテゴリや水準を持ちます。

> 「数値で表されても、その値自体に数学的な意味（大小関係や加算など）がない変数」

### カテゴリカル変数の分類

種類 | 説明 | 例  
---|---|---  
**名義変数（Nominal）** | 順序関係がないカテゴリ | 色（赤、青、緑）、性別、国名  
**順序変数（Ordinal）** | 順序関係があるカテゴリ | 評価（低、中、高）、学歴、サイズ（S、M、L）  
  
### なぜエンコーディングが必要か

多くの機械学習アルゴリズム（線形回帰、ニューラルネットワーク、SVMなど）は数値データのみを扱います。そのため、カテゴリカル変数を数値に変換する必要があります。
    
    
    import pandas as pd
    import numpy as np
    
    # カテゴリカル変数のサンプルデータ
    data = {
        'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
        'size': ['S', 'M', 'L', 'M', 'S', 'L'],
        'rating': ['low', 'medium', 'high', 'medium', 'low', 'high'],
        'price': [100, 150, 200, 120, 90, 180]
    }
    
    df = pd.DataFrame(data)
    print("=== カテゴリカル変数のサンプルデータ ===")
    print(df)
    print("\nデータ型:")
    print(df.dtypes)
    
    # カテゴリカル変数の確認
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nカテゴリカル変数: {categorical_cols}")
    
    # 各カテゴリカル変数のユニーク値数
    print("\n各変数のカテゴリ数（カーディナリティ）:")
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()}個 -> {df[col].unique()}")
    

**出力** ：
    
    
    === カテゴリカル変数のサンプルデータ ===
       color size rating  price
    0    red    S    low    100
    1   blue    M medium    150
    2  green    L   high    200
    3    red    M medium    120
    4   blue    S    low     90
    5  green    L   high    180
    
    データ型:
    color     object
    size      object
    rating    object
    price      int64
    dtype: object
    
    カテゴリカル変数: ['color', 'size', 'rating']
    
    各変数のカテゴリ数（カーディナリティ）:
      color: 3個 -> ['red' 'blue' 'green']
      size: 3個 -> ['S' 'M' 'L']
      rating: 3個 -> ['low' 'medium' 'high']
    

### カーディナリティの問題

**カーディナリティ（Cardinality）** は、カテゴリカル変数が持つユニークな値の数です。

  * **低カーディナリティ** : 2〜10個程度のカテゴリ → ほとんどの手法が適用可能
  * **高カーディナリティ** : 100個以上のカテゴリ → メモリ効率や過学習に注意が必要

    
    
    ```mermaid
    graph TD
        A[カテゴリカル変数] --> B{カーディナリティは?}
        B -->|低 2-10| C[One-Hot Encoding推奨]
        B -->|中 10-100| D[複数手法の比較検討]
        B -->|高 100+| E[Target/Frequency/Hashing]
    
        C --> F[各手法の適用]
        D --> F
        E --> F
    
        style A fill:#e3f2fd
        style C fill:#c8e6c9
        style D fill:#fff9c4
        style E fill:#ffccbc
    ```

* * *

## 2.2 One-Hot Encoding

### 概要

**One-Hot Encoding** は、カテゴリカル変数の各カテゴリを0と1のバイナリベクトルで表現する手法です。

### 原理

$n$ 個のカテゴリを持つ変数を $n$ 個のバイナリ変数に変換します。各サンプルでは、該当するカテゴリの列が1、それ以外が0になります。

**例** : 色 = {red, blue, green}

元データ | color_red | color_blue | color_green  
---|---|---|---  
red | 1 | 0 | 0  
blue | 0 | 1 | 0  
green | 0 | 0 | 1  
  
### pandasによる実装
    
    
    import pandas as pd
    
    # サンプルデータ
    data = {
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'price': [100, 150, 200, 120, 90]
    }
    
    df = pd.DataFrame(data)
    print("=== 元データ ===")
    print(df)
    
    # pandas get_dummiesによるOne-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['color', 'size'], drop_first=False)
    print("\n=== One-Hot Encoding後 ===")
    print(df_encoded)
    
    # drop_first=Trueで多重共線性を回避
    df_encoded_drop = pd.get_dummies(df, columns=['color', 'size'], drop_first=True)
    print("\n=== drop_first=True（1列削除） ===")
    print(df_encoded_drop)
    

**出力** ：
    
    
    === 元データ ===
       color size  price
    0    red    S    100
    1   blue    M    150
    2  green    L    200
    3    red    M    120
    4   blue    S     90
    
    === One-Hot Encoding後 ===
       price  color_blue  color_green  color_red  size_L  size_M  size_S
    0    100           0            0          1       0       0       1
    1    150           1            0          0       0       1       0
    2    200           0            1          0       1       0       0
    3    120           0            0          1       0       1       0
    4     90           1            0          0       0       0       1
    
    === drop_first=True（1列削除） ===
       price  color_green  color_red  size_M  size_S
    0    100            0          1       0       1
    1    150            0          0       1       0
    2    200            1          0       0       0
    3    120            0          1       1       0
    4     90            0          0       0       1
    

### scikit-learnによる実装
    
    
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    # サンプルデータ
    X = np.array([['red', 'S'],
                  ['blue', 'M'],
                  ['green', 'L'],
                  ['red', 'M'],
                  ['blue', 'S']])
    
    print("=== 元データ ===")
    print(X)
    
    # OneHotEncoderの適用
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_encoded = encoder.fit_transform(X)
    
    print("\n=== One-Hot Encoding後 ===")
    print(X_encoded)
    print(f"\n形状: {X_encoded.shape}")
    
    # カテゴリの確認
    print("\nカテゴリ:")
    for i, categories in enumerate(encoder.categories_):
        print(f"  特徴量{i}: {categories}")
    
    # 新しいデータへの適用
    X_new = np.array([['green', 'S'], ['red', 'L']])
    X_new_encoded = encoder.transform(X_new)
    print("\n=== 新しいデータのエンコーディング ===")
    print(X_new)
    print("↓")
    print(X_new_encoded)
    

### スパース行列の活用

高カーディナリティのカテゴリカル変数では、One-Hot Encodingによって大量の0を含む行列が生成されます。**スパース行列** を使うことでメモリ効率を改善できます。
    
    
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # 高カーディナリティのサンプル
    np.random.seed(42)
    n_samples = 10000
    categories = [f'cat_{i}' for i in range(1000)]
    X = np.random.choice(categories, size=(n_samples, 1))
    
    print(f"サンプル数: {n_samples}")
    print(f"カテゴリ数: {len(categories)}")
    
    # Dense形式
    encoder_dense = OneHotEncoder(sparse_output=False)
    X_dense = encoder_dense.fit_transform(X)
    dense_size = X_dense.nbytes / (1024 ** 2)  # MB
    
    # Sparse形式
    encoder_sparse = OneHotEncoder(sparse_output=True)
    X_sparse = encoder_sparse.fit_transform(X)
    sparse_size = (X_sparse.data.nbytes + X_sparse.indices.nbytes +
                   X_sparse.indptr.nbytes) / (1024 ** 2)  # MB
    
    print("\n=== メモリ使用量の比較 ===")
    print(f"Dense形式: {dense_size:.2f} MB")
    print(f"Sparse形式: {sparse_size:.2f} MB")
    print(f"削減率: {(1 - sparse_size/dense_size) * 100:.1f}%")
    

### One-Hot Encodingの利点と欠点

利点 | 欠点  
---|---  
カテゴリ間に順序を仮定しない | カテゴリ数に比例して次元が増加  
実装が簡単で解釈しやすい | 高カーディナリティで非効率  
線形モデルとの相性が良い | スパース性の問題  
新しいカテゴリの対応が必要 | 多重共線性のリスク  
  
* * *

## 2.3 Label EncodingとOrdinal Encoding

### Label Encoding

**Label Encoding** は、各カテゴリを整数（0, 1, 2, ...）に変換する手法です。
    
    
    from sklearn.preprocessing import LabelEncoder
    
    # サンプルデータ
    colors = ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
    
    # LabelEncoderの適用
    label_encoder = LabelEncoder()
    colors_encoded = label_encoder.fit_transform(colors)
    
    print("=== Label Encoding ===")
    print(f"元データ: {colors}")
    print(f"エンコード後: {colors_encoded}")
    print(f"\nマッピング:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # 逆変換
    colors_decoded = label_encoder.inverse_transform(colors_encoded)
    print(f"\n逆変換: {colors_decoded}")
    

**出力** ：
    
    
    === Label Encoding ===
    元データ: ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
    エンコード後: [2 0 1 2 0 1 2]
    
    マッピング:
      blue -> 0
      green -> 1
      red -> 2
    
    逆変換: ['red' 'blue' 'green' 'red' 'blue' 'green' 'red']
    

### Ordinal Encoding

**Ordinal Encoding** は、順序関係のあるカテゴリに対して、その順序を保持した数値を割り当てる手法です。
    
    
    from sklearn.preprocessing import OrdinalEncoder
    import numpy as np
    
    # 順序付きカテゴリのサンプル
    data = {
        'size': ['S', 'M', 'L', 'XL', 'M', 'S', 'L'],
        'rating': ['low', 'medium', 'high', 'medium', 'low', 'high', 'medium']
    }
    
    df = pd.DataFrame(data)
    print("=== 元データ ===")
    print(df)
    
    # 順序の定義
    size_order = ['S', 'M', 'L', 'XL']
    rating_order = ['low', 'medium', 'high']
    
    # OrdinalEncoderの適用
    ordinal_encoder = OrdinalEncoder(categories=[size_order, rating_order])
    df_encoded = df.copy()
    df_encoded[['size', 'rating']] = ordinal_encoder.fit_transform(df[['size', 'rating']])
    
    print("\n=== Ordinal Encoding後 ===")
    print(df_encoded)
    
    print("\n順序マッピング:")
    print("size: S(0) < M(1) < L(2) < XL(3)")
    print("rating: low(0) < medium(1) < high(2)")
    

**出力** ：
    
    
    === 元データ ===
      size  rating
    0    S     low
    1    M  medium
    2    L    high
    3   XL  medium
    4    M     low
    5    S    high
    6    L  medium
    
    === Ordinal Encoding後 ===
       size  rating
    0   0.0     0.0
    1   1.0     1.0
    2   2.0     2.0
    3   3.0     1.0
    4   1.0     0.0
    5   0.0     2.0
    6   2.0     1.0
    
    順序マッピング:
    size: S(0) < M(1) < L(2) < XL(3)
    rating: low(0) < medium(1) < high(2)
    

### Label EncodingとOrdinal Encodingの違い

特徴 | Label Encoding | Ordinal Encoding  
---|---|---  
**用途** | 目的変数のエンコーディング | 説明変数のエンコーディング  
**順序の考慮** | 考慮しない（アルファベット順など） | 明示的に順序を指定  
**実装** | LabelEncoder（1次元のみ） | OrdinalEncoder（複数列対応）  
**適用対象** | 分類問題のラベル | 順序付きカテゴリ特徴量  
  
### 注意点：誤った順序の仮定
    
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # 名義変数（順序なし）のサンプルデータ
    np.random.seed(42)
    n_samples = 1000
    colors = np.random.choice(['red', 'blue', 'green'], size=n_samples)
    # 'red'のときにyが1になりやすい
    y = (colors == 'red').astype(int)
    
    # 1. Label Encodingで学習（不適切）
    label_encoder = LabelEncoder()
    X_label = label_encoder.fit_transform(colors).reshape(-1, 1)
    
    clf_label = DecisionTreeClassifier(random_state=42)
    score_label = cross_val_score(clf_label, X_label, y, cv=5).mean()
    
    # 2. One-Hot Encodingで学習（適切）
    onehot_encoder = OneHotEncoder(sparse_output=False)
    X_onehot = onehot_encoder.fit_transform(colors.reshape(-1, 1))
    
    clf_onehot = DecisionTreeClassifier(random_state=42)
    score_onehot = cross_val_score(clf_onehot, X_onehot, y, cv=5).mean()
    
    print("=== エンコーディング手法の比較 ===")
    print(f"Label Encoding: {score_label:.4f}")
    print(f"One-Hot Encoding: {score_onehot:.4f}")
    print("\n⚠️ 決定木では差が小さいが、線形モデルでは大きな差が出る")
    

> **重要** : 名義変数にLabel Encodingを適用すると、存在しない順序関係がモデルに学習されます。線形モデルやニューラルネットワークではOne-Hot Encodingを推奨します。

* * *

## 2.4 Target Encoding（Mean Encoding）

### 概要

**Target Encoding** は、各カテゴリを目的変数の平均値（または他の統計量）で置き換える手法です。**Mean Encoding** とも呼ばれます。

### 原理

カテゴリ $c$ のTarget Encoding値：

$$ \text{TE}(c) = \frac{\sum_{i: x_i = c} y_i}{|i: x_i = c|} $$

つまり、そのカテゴリに属するサンプルの目的変数の平均値です。

### 過学習の問題とスムージング

Target Encodingは目的変数を直接使うため、**過学習しやすい** という問題があります。これを防ぐため、**スムージング** を適用します：

$$ \text{TE}_{\text{smooth}}(c) = \frac{n_c \cdot \text{mean}_c + m \cdot \text{global_mean}}{n_c + m} $$

  * $n_c$: カテゴリ $c$ のサンプル数
  * $\text{mean}_c$: カテゴリ $c$ の目的変数平均
  * $\text{global_mean}$: 全体の目的変数平均
  * $m$: スムージングパラメータ（通常1〜100）

### スクラッチ実装
    
    
    import pandas as pd
    import numpy as np
    
    # サンプルデータ
    np.random.seed(42)
    data = {
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] * 10,
        'target': np.random.randint(0, 2, 100)
    }
    
    # カテゴリAのtargetを意図的に高く設定
    data_list = list(zip(data['category'], data['target']))
    modified_data = []
    for cat, target in data_list:
        if cat == 'A':
            target = 1 if np.random.rand() < 0.8 else 0
        modified_data.append((cat, target))
    
    df = pd.DataFrame(modified_data, columns=['category', 'target'])
    
    print("=== サンプルデータ ===")
    print(df.head(10))
    print(f"\n各カテゴリの目的変数平均:")
    print(df.groupby('category')['target'].mean())
    
    # Target Encoding（スムージングなし）
    def target_encoding_simple(df, column, target_col):
        """シンプルなTarget Encoding"""
        mean_encoding = df.groupby(column)[target_col].mean()
        return df[column].map(mean_encoding)
    
    # Target Encoding（スムージングあり）
    def target_encoding_smoothed(df, column, target_col, m=10):
        """スムージング付きTarget Encoding"""
        global_mean = df[target_col].mean()
        category_stats = df.groupby(column)[target_col].agg(['mean', 'count'])
    
        smoothed = (category_stats['count'] * category_stats['mean'] +
                    m * global_mean) / (category_stats['count'] + m)
    
        return df[column].map(smoothed)
    
    # 適用
    df['te_simple'] = target_encoding_simple(df, 'category', 'target')
    df['te_smoothed'] = target_encoding_smoothed(df, 'category', 'target', m=10)
    
    print("\n=== Target Encoding結果 ===")
    print(df.groupby('category')[['target', 'te_simple', 'te_smoothed']].mean())
    

### Cross-Validation戦略

学習データで計算したTarget Encodingを同じ学習データに適用すると、**リーケージ（情報漏洩）** が発生します。これを防ぐため、**Out-of-Fold** 戦略を使います。
    
    
    from sklearn.model_selection import KFold
    
    def target_encoding_cv(X, y, column, n_splits=5, m=10):
        """Cross-ValidationによるTarget Encoding"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoded = np.zeros(len(X))
        global_mean = y.mean()
    
        for train_idx, val_idx in kfold.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
            # 訓練データで統計量を計算
            category_stats = pd.DataFrame({
                'category': X_train[column],
                'target': y_train
            }).groupby('category')['target'].agg(['mean', 'count'])
    
            # スムージング
            smoothed_means = (category_stats['count'] * category_stats['mean'] +
                              m * global_mean) / (category_stats['count'] + m)
    
            # 検証データに適用
            encoded[val_idx] = X.iloc[val_idx][column].map(smoothed_means)
    
            # マッピングされなかった値はglobal_meanで埋める
            encoded[val_idx] = np.nan_to_num(encoded[val_idx], nan=global_mean)
    
        return encoded
    
    # サンプルデータ
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    })
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # カテゴリAのtargetを高く設定
    y[X['category'] == 'A'] = np.random.choice([0, 1], size=(X['category'] == 'A').sum(), p=[0.2, 0.8])
    
    # CV戦略によるTarget Encoding
    X['te_cv'] = target_encoding_cv(X, y, 'category', n_splits=5, m=10)
    
    print("=== Cross-ValidationによるTarget Encoding ===")
    print(X.groupby('category')['te_cv'].agg(['mean', 'std']))
    print(f"\n目的変数の平均:")
    print(y.groupby(X['category']).mean())
    

### category_encodersライブラリの使用
    
    
    import category_encoders as ce
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'category1': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z'], size=n_samples),
        'numeric': np.random.randn(n_samples)
    })
    
    # カテゴリAとXの組み合わせでtargetが1になりやすい
    y = ((X['category1'] == 'A') & (X['category2'] == 'X')).astype(int)
    y = np.where(np.random.rand(n_samples) < 0.3, 1 - y, y)  # ノイズ追加
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. One-Hot Encodingで学習
    X_train_onehot = pd.get_dummies(X_train, columns=['category1', 'category2'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category1', 'category2'])
    
    # カラムを揃える
    missing_cols = set(X_train_onehot.columns) - set(X_test_onehot.columns)
    for col in missing_cols:
        X_test_onehot[col] = 0
    X_test_onehot = X_test_onehot[X_train_onehot.columns]
    
    clf_onehot = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_onehot.fit(X_train_onehot, y_train)
    y_pred_onehot = clf_onehot.predict(X_test_onehot)
    acc_onehot = accuracy_score(y_test, y_pred_onehot)
    
    # 2. Target Encodingで学習
    target_encoder = ce.TargetEncoder(cols=['category1', 'category2'], smoothing=10)
    X_train_te = target_encoder.fit_transform(X_train, y_train)
    X_test_te = target_encoder.transform(X_test)
    
    clf_te = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_te.fit(X_train_te, y_train)
    y_pred_te = clf_te.predict(X_test_te)
    acc_te = accuracy_score(y_test, y_pred_te)
    
    print("=== エンコーディング手法の性能比較 ===")
    print(f"One-Hot Encoding: 精度 = {acc_onehot:.4f}")
    print(f"Target Encoding:  精度 = {acc_te:.4f}")
    

### Target Encodingの利点と欠点

利点 | 欠点  
---|---  
高カーディナリティに対応 | 過学習しやすい  
次元が増加しない | CV戦略が必須  
目的変数との関係を直接捉える | 実装が複雑  
木ベースモデルとの相性が良い | 回帰問題では効果が限定的な場合も  
  
* * *

## 2.5 Frequency Encoding

### 概要

**Frequency Encoding** は、各カテゴリを出現頻度（または出現割合）で置き換える手法です。

### 原理

カテゴリ $c$ のFrequency Encoding値：

$$ \text{FE}(c) = \frac{\text{count}(c)}{N} $$

ここで $N$ は総サンプル数です。

### 実装
    
    
    import pandas as pd
    import numpy as np
    
    # サンプルデータ
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D', 'E']
    # カテゴリAが最も頻繁に出現
    probabilities = [0.5, 0.2, 0.15, 0.1, 0.05]
    
    data = {
        'category': np.random.choice(categories, size=1000, p=probabilities),
        'value': np.random.randn(1000)
    }
    
    df = pd.DataFrame(data)
    
    print("=== カテゴリの出現回数 ===")
    print(df['category'].value_counts().sort_index())
    
    # Frequency Encoding（カウントベース）
    def frequency_encoding_count(df, column):
        """カウントベースのFrequency Encoding"""
        frequency = df[column].value_counts()
        return df[column].map(frequency)
    
    # Frequency Encoding（割合ベース）
    def frequency_encoding_ratio(df, column):
        """割合ベースのFrequency Encoding"""
        frequency = df[column].value_counts(normalize=True)
        return df[column].map(frequency)
    
    # 適用
    df['freq_count'] = frequency_encoding_count(df, 'category')
    df['freq_ratio'] = frequency_encoding_ratio(df, 'category')
    
    print("\n=== Frequency Encoding結果 ===")
    print(df.groupby('category')[['freq_count', 'freq_ratio']].first().sort_index())
    print("\nサンプルデータ:")
    print(df.head(10))
    

**出力** ：
    
    
    === カテゴリの出現回数 ===
    A    492
    B    206
    C    163
    D     95
    E     44
    Name: category, dtype: int64
    
    === Frequency Encoding結果 ===
              freq_count  freq_ratio
    category
    A                492       0.492
    B                206       0.206
    C                163       0.163
    D                 95       0.095
    E                 44       0.044
    
    サンプルデータ:
      category     value  freq_count  freq_ratio
    0        C  0.496714         163       0.163
    1        A -0.138264         492       0.492
    2        A  0.647689         492       0.492
    3        A  1.523030         492       0.492
    4        B -0.234153         206       0.206
    

### Frequency Encodingの応用例
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 2000
    
    # 高頻度カテゴリがtarget=1になりやすい
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'],
                                  size=n_samples,
                                  p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    # 'A'と'B'のときにtargetが1になりやすい
    target = np.where(np.isin(categories, ['A', 'B']),
                      np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                      np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': categories})
    y = target
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Label Encodingで学習
    label_encoder = LabelEncoder()
    X_train_label = label_encoder.fit_transform(X_train['category']).reshape(-1, 1)
    X_test_label = label_encoder.transform(X_test['category']).reshape(-1, 1)
    
    clf_label = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_label.fit(X_train_label, y_train)
    acc_label = accuracy_score(y_test, clf_label.predict(X_test_label))
    
    # 2. Frequency Encodingで学習
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    clf_freq = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_freq.fit(X_train_freq, y_train)
    acc_freq = accuracy_score(y_test, clf_freq.predict(X_test_freq))
    
    # 3. One-Hot Encodingで学習
    X_train_onehot = pd.get_dummies(X_train, columns=['category'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category'])
    X_test_onehot = X_test_onehot.reindex(columns=X_train_onehot.columns, fill_value=0)
    
    clf_onehot = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_onehot.fit(X_train_onehot, y_train)
    acc_onehot = accuracy_score(y_test, clf_onehot.predict(X_test_onehot))
    
    print("=== エンコーディング手法の性能比較 ===")
    print(f"Label Encoding:     精度 = {acc_label:.4f}")
    print(f"Frequency Encoding: 精度 = {acc_freq:.4f}")
    print(f"One-Hot Encoding:   精度 = {acc_onehot:.4f}")
    

### いつFrequency Encodingを使うべきか

  * カテゴリの出現頻度が目的変数と相関がある場合
  * 高カーディナリティのカテゴリ変数
  * 次元削減が必要な場合
  * 新しいカテゴリ（未知のカテゴリ）が出現する可能性がある場合

* * *

## 2.6 Binary EncodingとHashing

### Binary Encoding

**Binary Encoding** は、カテゴリを整数に変換し、その整数を2進数で表現する手法です。One-Hot Encodingより次元を削減できます。

### 原理

$n$ 個のカテゴリを $\lceil \log_2 n \rceil$ 個のバイナリ列で表現します。

**例** : 8個のカテゴリ → 3列（$\lceil \log_2 8 \rceil = 3$）

カテゴリ | 整数 | bit_0 | bit_1 | bit_2  
---|---|---|---|---  
A | 0 | 0 | 0 | 0  
B | 1 | 0 | 0 | 1  
C | 2 | 0 | 1 | 0  
D | 3 | 0 | 1 | 1  
E | 4 | 1 | 0 | 0  
  
### 実装
    
    
    import category_encoders as ce
    import pandas as pd
    import numpy as np
    
    # サンプルデータ
    np.random.seed(42)
    categories = [f'cat_{i}' for i in range(50)]
    data = {
        'category': np.random.choice(categories, size=200)
    }
    
    df = pd.DataFrame(data)
    
    print(f"=== Binary Encoding ===")
    print(f"カテゴリ数: {df['category'].nunique()}")
    print(f"必要なビット数: {int(np.ceil(np.log2(df['category'].nunique())))}")
    
    # Binary Encoderの適用
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    df_encoded = binary_encoder.fit_transform(df)
    
    print(f"\nエンコード後の列数: {df_encoded.shape[1]}")
    print("\nサンプル:")
    print(df_encoded.head(10))
    
    # 次元の比較
    print("\n=== One-Hot vs Binary Encoding ===")
    n_categories = 100
    onehot_dims = n_categories
    binary_dims = int(np.ceil(np.log2(n_categories)))
    
    print(f"カテゴリ数: {n_categories}")
    print(f"One-Hot Encoding: {onehot_dims}次元")
    print(f"Binary Encoding: {binary_dims}次元")
    print(f"削減率: {(1 - binary_dims/onehot_dims) * 100:.1f}%")
    

### Hashing Trick

**Hashing Trick** は、ハッシュ関数を使ってカテゴリを固定次元のベクトルに変換する手法です。

### 原理

  1. ハッシュ関数 $h$ でカテゴリを整数にマッピング: $h(c) \in \\{0, 1, ..., m-1\\}$
  2. その整数に対応する位置を1にする

**利点** :

  * 事前にカテゴリの数を知る必要がない
  * 新しいカテゴリが自動的に処理される
  * メモリ効率が良い

**欠点** :

  * ハッシュの衝突（異なるカテゴリが同じ値にマッピング）
  * 解釈性の低下

### 実装
    
    
    from sklearn.feature_extraction import FeatureHasher
    import pandas as pd
    import numpy as np
    
    # サンプルデータ
    np.random.seed(42)
    categories = [f'cat_{i}' for i in range(1000)]
    data = {'category': np.random.choice(categories, size=5000)}
    df = pd.DataFrame(data)
    
    print("=== Hashing Trick ===")
    print(f"ユニークカテゴリ数: {df['category'].nunique()}")
    
    # FeatureHasherの適用
    n_features = 50  # ハッシュの次元
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    
    # カテゴリをリストのリストに変換
    X_hashed = hasher.transform([[cat] for cat in df['category']])
    
    print(f"ハッシュ後の次元: {X_hashed.shape[1]}")
    print(f"スパース性: {(1 - X_hashed.nnz / (X_hashed.shape[0] * X_hashed.shape[1])) * 100:.1f}%")
    
    # ハッシュ衝突の確認
    unique_hashes = set()
    collisions = 0
    
    for cat in df['category'].unique():
        hash_val = hash(cat) % n_features
        if hash_val in unique_hashes:
            collisions += 1
        unique_hashes.add(hash_val)
    
    print(f"\nハッシュ衝突数: {collisions}")
    print(f"衝突率: {collisions / df['category'].nunique() * 100:.2f}%")
    
    # 次元数と衝突率の関係
    dimensions = [10, 20, 50, 100, 200, 500]
    collision_rates = []
    
    for dim in dimensions:
        unique_hashes = set()
        collisions = 0
        for cat in df['category'].unique():
            hash_val = hash(cat) % dim
            if hash_val in unique_hashes:
                collisions += 1
            unique_hashes.add(hash_val)
        collision_rate = collisions / df['category'].nunique() * 100
        collision_rates.append(collision_rate)
    
    print("\n=== 次元数と衝突率 ===")
    for dim, rate in zip(dimensions, collision_rates):
        print(f"{dim}次元: 衝突率 {rate:.2f}%")
    

* * *

## 2.7 手法の比較と使い分け

### エンコーディング手法の総合比較

手法 | 次元増加 | 高カーディナリティ | 解釈性 | 過学習リスク  
---|---|---|---|---  
**One-Hot** | 大（n列） | 不向き | 高 | 低  
**Label/Ordinal** | なし（1列） | 適用可 | 中 | 低  
**Target** | なし（1列） | 適用可 | 中 | 高（CV必須）  
**Frequency** | なし（1列） | 適用可 | 高 | 低  
**Binary** | 小（log n列） | 適用可 | 低 | 低  
**Hashing** | 固定（m列） | 適用可 | 低 | 低  
  
### 使い分けのフローチャート
    
    
    ```mermaid
    graph TD
        A[カテゴリカル変数] --> B{カーディナリティは?}
        B -->|低 2-10| C{順序あり?}
        B -->|中 10-100| D[複数手法を試す]
        B -->|高 100+| E[Target/Frequency/Hashing]
    
        C -->|あり| F[Ordinal Encoding]
        C -->|なし| G[One-Hot Encoding]
    
        D --> H[One-Hot/Target/Frequency]
    
        E --> I{目的変数との相関?}
        I -->|強い| J[Target Encoding + CV]
        I -->|弱い| K[Frequency/Hashing]
    
        style A fill:#e3f2fd
        style G fill:#c8e6c9
        style F fill:#fff9c4
        style J fill:#ffccbc
    ```

### 実践的な使い分けガイド

状況 | 推奨手法 | 理由  
---|---|---  
線形モデル + 低カーディナリティ | One-Hot | 線形モデルは順序を仮定する  
木ベースモデル + 順序あり | Ordinal | 木の分岐に順序が役立つ  
高カーディナリティ + 分類問題 | Target | 目的変数との関係を直接捉える  
ストリーミングデータ | Hashing | 新カテゴリに自動対応  
メモリ制約 | Binary/Hashing | 次元削減  
解釈性重視 | One-Hot/Frequency | 直感的な理解が可能  
  
### 実例：全手法の性能比較
    
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import category_encoders as ce
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 2000
    
    # 中カーディナリティのカテゴリ（20個）
    categories = [f'cat_{i}' for i in range(20)]
    X_cat = np.random.choice(categories, size=n_samples)
    
    # 一部のカテゴリでtargetが1になりやすい
    high_target_cats = ['cat_0', 'cat_1', 'cat_5', 'cat_10']
    y = np.where(np.isin(X_cat, high_target_cats),
                 np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                 np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': X_cat})
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = []
    
    # 1. One-Hot Encoding
    X_train_onehot = pd.get_dummies(X_train, columns=['category'])
    X_test_onehot = pd.get_dummies(X_test, columns=['category'])
    X_test_onehot = X_test_onehot.reindex(columns=X_train_onehot.columns, fill_value=0)
    
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    score_onehot = cross_val_score(clf_rf, X_train_onehot, y_train, cv=5).mean()
    results.append(('One-Hot', score_onehot, X_train_onehot.shape[1]))
    
    # 2. Label Encoding
    label_encoder = LabelEncoder()
    X_train_label = label_encoder.fit_transform(X_train['category']).reshape(-1, 1)
    X_test_label = label_encoder.transform(X_test['category']).reshape(-1, 1)
    
    score_label = cross_val_score(clf_rf, X_train_label, y_train, cv=5).mean()
    results.append(('Label', score_label, 1))
    
    # 3. Target Encoding
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_train_target = target_encoder.fit_transform(X_train, y_train)
    X_test_target = target_encoder.transform(X_test)
    
    score_target = cross_val_score(clf_rf, X_train_target, y_train, cv=5).mean()
    results.append(('Target', score_target, 1))
    
    # 4. Frequency Encoding
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    score_freq = cross_val_score(clf_rf, X_train_freq, y_train, cv=5).mean()
    results.append(('Frequency', score_freq, 1))
    
    # 5. Binary Encoding
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    X_train_binary = binary_encoder.fit_transform(X_train)
    X_test_binary = binary_encoder.transform(X_test)
    
    score_binary = cross_val_score(clf_rf, X_train_binary, y_train, cv=5).mean()
    results.append(('Binary', score_binary, X_train_binary.shape[1]))
    
    # 結果の表示
    print("=== エンコーディング手法の性能比較（Random Forest） ===")
    print(f"{'手法':<15} {'精度':<10} {'次元数':<10}")
    print("-" * 35)
    for method, score, dims in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{method:<15} {score:.4f}    {dims:<10}")
    

**出力例** ：
    
    
    === エンコーディング手法の性能比較（Random Forest） ===
    手法             精度        次元数
    -----------------------------------
    Target          0.7531    1
    One-Hot         0.7469    20
    Binary          0.7419    5
    Frequency       0.6956    1
    Label           0.6894    1
    

* * *

## 2.8 本章のまとめ

### 学んだこと

  1. **カテゴリカル変数の基礎**

     * 名義変数と順序変数の違い
     * カーディナリティの概念と重要性
     * なぜエンコーディングが必要か
  2. **One-Hot Encoding**

     * バイナリベクトルによる表現
     * pandas get_dummiesとOneHotEncoderの使い分け
     * スパース行列によるメモリ効率化
     * drop_firstによる多重共線性の回避
  3. **Label EncodingとOrdinal Encoding**

     * 整数への変換手法
     * 順序の有無による使い分け
     * 線形モデルでの注意点
  4. **Target Encoding**

     * 目的変数の統計量による変換
     * 過学習対策としてのスムージング
     * Cross-Validation戦略の重要性
     * 高カーディナリティへの対応
  5. **Frequency Encoding**

     * 出現頻度による変換
     * シンプルで効果的な手法
     * 新しいカテゴリへの対応
  6. **Binary EncodingとHashing**

     * 次元削減を実現する手法
     * 高カーディナリティへの対応
     * ハッシュ衝突のトレードオフ
  7. **手法の使い分け**

     * カーディナリティに基づく選択
     * モデルとの相性
     * 計算リソースと精度のバランス

### 次の章へ

第3章では、**数値特徴量の変換とスケーリング** を学びます：

  * 標準化と正規化
  * 対数変換とBox-Cox変換
  * ビニング（離散化）
  * 特徴量の相互作用

* * *

## 演習問題

### 問題1（難易度：easy）

One-Hot Encodingで`drop_first=True`を使う理由を、多重共線性の観点から説明してください。

解答例

**解答** ：

**多重共線性（Multicollinearity）** とは、説明変数間に強い相関がある状態を指します。

One-Hot Encodingでは、$n$ 個のカテゴリを $n$ 個のバイナリ変数に変換します。このとき、以下の関係が成り立ちます：

$$ \sum_{i=1}^{n} x_i = 1 $$

つまり、1つの変数の値が他の $n-1$ 個の変数から完全に予測できます。これが多重共線性を引き起こします。

**問題点** ：

  * 線形回帰の係数が不安定になる
  * 逆行列計算でエラーが発生する可能性
  * 統計的推論が困難になる

**解決策** ：

`drop_first=True`により、$n$ 個のカテゴリを $n-1$ 個の変数で表現します。省略されたカテゴリは「すべての変数が0」で表現されます。

**例** ：
    
    
    色 = {red, blue, green}
    drop_first=False: color_red, color_blue, color_green (3列)
    drop_first=True:  color_blue, color_green (2列)
      - red: [0, 0]
      - blue: [1, 0]
      - green: [0, 1]
    

### 問題2（難易度：medium）

Target Encodingで過学習を防ぐための3つの戦略を説明してください。

解答例

**解答** ：

**1\. スムージング（Smoothing）**

サンプル数が少ないカテゴリの統計量を、全体平均で正則化します：

$$ \text{TE}_{\text{smooth}}(c) = \frac{n_c \cdot \text{mean}_c + m \cdot \text{global_mean}}{n_c + m} $$

  * $m$ が大きいほど全体平均に近づく（保守的）
  * $m$ が小さいほどカテゴリ平均に近づく（過学習リスク）
  * 推奨値: $m = 1 \sim 100$

**2\. Cross-Validation戦略（Out-of-Fold Encoding）**

  1. データをK分割
  2. Fold $k$ の統計量を、他のFoldで計算
  3. 学習データと評価データを分離

これにより、同じデータで統計量を計算して使うという**リーケージ** を防ぎます。

**3\. ノイズ付加（Noise Addition）**

エンコード値に微小なノイズを加えます：

$$ \text{TE}_{\text{noise}}(c) = \text{TE}(c) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

  * 過学習を抑制
  * $\sigma$ は小さい値（0.01〜0.1程度）

**実装例** ：
    
    
    import category_encoders as ce
    
    # スムージング付きTarget Encoding
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_encoded = target_encoder.fit_transform(X_train, y_train)
    

### 問題3（難易度：medium）

以下のカテゴリカル変数に対して、最適なエンコーディング手法を選択し、その理由を説明してください。

  1. 都道府県名（47カテゴリ）
  2. Webサイトの訪問者ID（100万カテゴリ）
  3. 顧客の満足度（1=低、2=中、3=高）
  4. 製品カテゴリ（5カテゴリ）

解答例

**解答** ：

**1\. 都道府県名（47カテゴリ）**

**推奨** : Target Encoding または One-Hot Encoding

**理由** ：

  * カーディナリティ: 中程度（47）
  * One-Hot: 47列に増加するが許容範囲
  * Target: 1列で高い表現力。地域と目的変数の関係を捉えられる
  * 順序関係なし（名義変数）

**選択基準** ：

  * 線形モデル → One-Hot
  * 木ベースモデル + 分類問題 → Target

**2\. Webサイトの訪問者ID（100万カテゴリ）**

**推奨** : Frequency Encoding または Hashing

**理由** ：

  * カーディナリティ: 非常に高い（100万）
  * One-Hot: メモリ不足で実質不可能
  * Frequency: 訪問頻度が有用な特徴になる可能性
  * Hashing: 固定次元で新規IDに自動対応

**3\. 顧客の満足度（1=低、2=中、3=高）**

**推奨** : Ordinal Encoding

**理由** ：

  * 明確な順序関係（順序変数）
  * 低(0) < 中(1) < 高(2)という順序を保持すべき
  * One-Hotは順序情報を失う
  * そのまま整数値として扱える

    
    
    from sklearn.preprocessing import OrdinalEncoder
    
    encoder = OrdinalEncoder(categories=[['低', '中', '高']])
    X_encoded = encoder.fit_transform(X)
    

**4\. 製品カテゴリ（5カテゴリ）**

**推奨** : One-Hot Encoding

**理由** ：

  * カーディナリティ: 低い（5）
  * 順序関係なし（名義変数）
  * One-Hotで5列に増加するが問題なし
  * 解釈性が高い
  * 線形モデルとの相性が良い

### 問題4（難易度：hard）

高カーディナリティのカテゴリカル変数（1000カテゴリ）に対して、One-Hot Encoding、Target Encoding、Frequency Encoding、Binary Encodingを適用し、Random Forestで性能を比較するコードを書いてください。

解答例
    
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import category_encoders as ce
    import time
    
    # 高カーディナリティのサンプルデータ生成
    np.random.seed(42)
    n_samples = 10000
    n_categories = 1000
    
    # カテゴリの生成（べき分布で現実的な頻度分布）
    categories = [f'cat_{i}' for i in range(n_categories)]
    weights = np.array([1/(i+1)**0.8 for i in range(n_categories)])
    weights /= weights.sum()
    
    X_cat = np.random.choice(categories, size=n_samples, p=weights)
    
    # 目的変数: 上位50カテゴリでtarget=1になりやすい
    high_target_cats = [f'cat_{i}' for i in range(50)]
    y = np.where(np.isin(X_cat, high_target_cats),
                 np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                 np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    X = pd.DataFrame({'category': X_cat})
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"=== データ概要 ===")
    print(f"サンプル数: {n_samples}")
    print(f"カテゴリ数: {X['category'].nunique()}")
    print(f"訓練データ: {len(X_train)}, テストデータ: {len(X_test)}")
    
    results = []
    
    # Random Forestモデル
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 1. One-Hot Encoding（スパース行列）
    print("\n1. One-Hot Encoding...")
    start_time = time.time()
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_train_onehot = onehot_encoder.fit_transform(X_train[['category']])
    X_test_onehot = onehot_encoder.transform(X_test[['category']])
    
    score_onehot = cross_val_score(clf, X_train_onehot, y_train, cv=3, n_jobs=-1).mean()
    time_onehot = time.time() - start_time
    results.append(('One-Hot', score_onehot, X_train_onehot.shape[1], time_onehot))
    
    # 2. Target Encoding
    print("2. Target Encoding...")
    start_time = time.time()
    target_encoder = ce.TargetEncoder(cols=['category'], smoothing=10)
    X_train_target = target_encoder.fit_transform(X_train, y_train)
    X_test_target = target_encoder.transform(X_test)
    
    score_target = cross_val_score(clf, X_train_target, y_train, cv=3, n_jobs=-1).mean()
    time_target = time.time() - start_time
    results.append(('Target', score_target, 1, time_target))
    
    # 3. Frequency Encoding
    print("3. Frequency Encoding...")
    start_time = time.time()
    freq_map = X_train['category'].value_counts(normalize=True)
    X_train_freq = X_train['category'].map(freq_map).values.reshape(-1, 1)
    X_test_freq = X_test['category'].map(freq_map).fillna(0).values.reshape(-1, 1)
    
    score_freq = cross_val_score(clf, X_train_freq, y_train, cv=3, n_jobs=-1).mean()
    time_freq = time.time() - start_time
    results.append(('Frequency', score_freq, 1, time_freq))
    
    # 4. Binary Encoding
    print("4. Binary Encoding...")
    start_time = time.time()
    binary_encoder = ce.BinaryEncoder(cols=['category'])
    X_train_binary = binary_encoder.fit_transform(X_train)
    X_test_binary = binary_encoder.transform(X_test)
    
    score_binary = cross_val_score(clf, X_train_binary, y_train, cv=3, n_jobs=-1).mean()
    time_binary = time.time() - start_time
    results.append(('Binary', score_binary, X_train_binary.shape[1], time_binary))
    
    # 結果の表示
    print("\n" + "="*70)
    print("=== エンコーディング手法の性能比較（1000カテゴリ） ===")
    print("="*70)
    print(f"{'手法':<15} {'精度':<10} {'次元数':<10} {'実行時間(秒)':<15}")
    print("-"*70)
    
    for method, score, dims, exec_time in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{method:<15} {score:.4f}    {dims:<10} {exec_time:.2f}")
    
    print("\n" + "="*70)
    print("考察:")
    print("- Target Encoding: 高精度 + 1次元 + 高速")
    print("- One-Hot: 高精度だがメモリ使用量大")
    print("- Binary: 次元削減とバランスの取れた性能")
    print("- Frequency: シンプルだが情報量不足")
    print("="*70)
    

**出力例** ：
    
    
    === データ概要 ===
    サンプル数: 10000
    カテゴリ数: 1000
    訓練データ: 8000, テストデータ: 2000
    
    1. One-Hot Encoding...
    2. Target Encoding...
    3. Frequency Encoding...
    4. Binary Encoding...
    
    ======================================================================
    === エンコーディング手法の性能比較（1000カテゴリ） ===
    ======================================================================
    手法             精度        次元数      実行時間(秒)
    ----------------------------------------------------------------------
    Target          0.8125    1          2.45
    One-Hot         0.8031    1000       5.67
    Binary          0.7794    10         3.12
    Frequency       0.7031    1          1.89
    
    ======================================================================
    考察:
    - Target Encoding: 高精度 + 1次元 + 高速
    - One-Hot: 高精度だがメモリ使用量大
    - Binary: 次元削減とバランスの取れた性能
    - Frequency: シンプルだが情報量不足
    ======================================================================
    

### 問題5（難易度：hard）

新しいカテゴリ（未知のカテゴリ）が出現する可能性がある場合、各エンコーディング手法をどのように対応させるか説明してください。

解答例

**解答** ：

新しいカテゴリへの対応は、実運用で非常に重要です。各手法の対応方法を説明します。

**1\. One-Hot Encoding**

**対応策** ：

  * `handle_unknown='ignore'`: 未知カテゴリをすべて0にする
  * 「その他」カテゴリを追加する

    
    
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train)
    X_test_encoded = encoder.transform(X_test)  # 未知カテゴリは[0,0,0,...]
    

**2\. Label Encoding / Ordinal Encoding**

**対応策** ：

  * 未知カテゴリに特別な値（-1など）を割り当てる
  * 最も頻度の高いカテゴリで代替する

    
    
    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    encoder.fit(X_train)
    
    # 未知カテゴリを-1で処理
    X_test_encoded = []
    for x in X_test:
        if x in encoder.classes_:
            X_test_encoded.append(encoder.transform([x])[0])
        else:
            X_test_encoded.append(-1)  # 未知カテゴリ
    

**3\. Target Encoding**

**対応策** ：

  * 全体平均（global mean）で置き換える
  * スムージングの全体平均と同じ値を使用

    
    
    import category_encoders as ce
    
    target_encoder = ce.TargetEncoder(cols=['category'],
                                      smoothing=10,
                                      handle_unknown='value',
                                      handle_missing='value')
    
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)  # 未知 → global mean
    

**4\. Frequency Encoding**

**対応策** ：

  * 頻度0（または最小頻度）を割り当てる
  * 希少カテゴリとして扱う

    
    
    freq_map = X_train['category'].value_counts(normalize=True)
    min_freq = freq_map.min()
    
    # 未知カテゴリは最小頻度
    X_test_encoded = X_test['category'].map(freq_map).fillna(min_freq)
    

**5\. Binary Encoding**

**対応策** ：

  * 未知カテゴリに特別なコード（すべて0など）を割り当てる

    
    
    import category_encoders as ce
    
    binary_encoder = ce.BinaryEncoder(cols=['category'], handle_unknown='value')
    X_train_encoded = binary_encoder.fit_transform(X_train)
    X_test_encoded = binary_encoder.transform(X_test)
    

**6\. Hashing**

**対応策** ：

  * 自動的に対応（ハッシュ関数で固定次元に変換）
  * 新カテゴリも既存のハッシュ値にマッピングされる

    
    
    from sklearn.feature_extraction import FeatureHasher
    
    hasher = FeatureHasher(n_features=50, input_type='string')
    X_train_hashed = hasher.transform([[cat] for cat in X_train['category']])
    X_test_hashed = hasher.transform([[cat] for cat in X_test['category']])
    # 未知カテゴリも自動的にハッシュされる
    

**推奨戦略** ：

状況 | 推奨手法  
---|---  
未知カテゴリが頻繁 | Hashing  
未知カテゴリが稀 | One-Hot（ignore） or Target（global mean）  
解釈性重視 | Frequency（最小頻度）  
高精度優先 | Target（global mean）  
  
* * *

## 参考文献

  1. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. _ACM SIGKDD Explorations Newsletter_ , 3(1), 27-32.
  2. Weinberger, K., et al. (2009). Feature hashing for large scale multitask learning. _Proceedings of the 26th Annual International Conference on Machine Learning_.
  3. Pargent, F., et al. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. _Computational Statistics_ , 37(5), 2671-2692.
  4. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  5. Kuhn, M., & Johnson, K. (2019). _Feature Engineering and Selection: A Practical Approach for Predictive Models_. CRC Press.
