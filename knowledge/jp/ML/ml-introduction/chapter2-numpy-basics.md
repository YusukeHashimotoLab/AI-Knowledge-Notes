---
title: 第2章：NumPy基礎
chapter_title: 第2章：NumPy基礎
---

**高速数値計算ライブラリNumPyをマスターしよう**

## はじめに

**NumPy (Numerical Python)** は、Pythonで数値計算を高速に行うための基盤ライブラリです。機械学習では、大量のデータを効率的に処理する必要があり、NumPyはそのための必須ツールです。

この章では、以下の内容を学びます：

  * NumPy配列の作成と基本操作
  * 配列の形状操作（reshape, transpose）
  * インデックスとスライシング
  * ユニバーサル関数（数学演算）
  * ブロードキャスティング
  * 統計関数と線形代数

> **なぜNumPyが必要か？**  
>  Pythonのリストに比べて、NumPy配列は10〜100倍高速です。大規模データ処理では、この速度差が重要になります。 

## 1\. NumPy配列の作成

### 1.1 基本的な配列作成

#### 例1：配列の作成方法
    
    
    import numpy as np
    
    # リストから配列を作成
    arr1 = np.array([1, 2, 3, 4, 5])
    print("1次元配列:", arr1)
    print("型:", type(arr1))
    print("データ型:", arr1.dtype)
    # 出力:
    # 1次元配列: [1 2 3 4 5]
    # 型: <class 'numpy.ndarray'>
    # データ型: int64
    
    # 2次元配列（行列）
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n2次元配列:")
    print(arr2)
    print("形状:", arr2.shape)  # (2, 3) = 2行3列
    print("次元数:", arr2.ndim)
    print("要素数:", arr2.size)
    # 出力:
    # 2次元配列:
    # [[1 2 3]
    #  [4 5 6]]
    # 形状: (2, 3)
    # 次元数: 2
    # 要素数: 6
    
    # データ型を指定
    arr3 = np.array([1.5, 2.3, 3.7], dtype=np.float32)
    print("\nfloat32配列:", arr3)
    print("データ型:", arr3.dtype)
    

### 1.2 便利な配列作成関数

#### 例2：特殊な配列の作成
    
    
    import numpy as np
    
    # ゼロで埋めた配列
    zeros = np.zeros((3, 4))  # 3行4列
    print("ゼロ配列:")
    print(zeros)
    
    # 1で埋めた配列
    ones = np.ones((2, 3))
    print("\n1配列:")
    print(ones)
    
    # 連続した数値
    arange = np.arange(0, 10, 2)  # 0から10未満、2刻み
    print("\narange:", arange)  # [0 2 4 6 8]
    
    # 等間隔の数値
    linspace = np.linspace(0, 1, 5)  # 0から1まで5個
    print("linspace:", linspace)  # [0.   0.25 0.5  0.75 1.  ]
    
    # 単位行列
    identity = np.eye(3)  # 3x3の単位行列
    print("\n単位行列:")
    print(identity)
    
    # ランダム配列
    np.random.seed(42)  # 再現性のためのシード
    random = np.random.rand(2, 3)  # 0-1の一様分布
    print("\nランダム配列:")
    print(random)
    
    # 正規分布に従う乱数
    normal = np.random.randn(3, 3)  # 平均0、標準偏差1
    print("\n正規分布:")
    print(normal)
    
    
    
    ```mermaid
    graph LR
        A[配列作成] --> B[np.array]
        A --> C[np.zeros/ones]
        A --> D[np.arange/linspace]
        A --> E[np.random]
    
        B --> F[リストから変換]
        C --> G[特定値で初期化]
        D --> H[数列生成]
        E --> I[乱数生成]
    
        style A fill:#e3f2fd
        style F fill:#fff3e0
        style G fill:#f3e5f5
        style H fill:#e8f5e9
        style I fill:#fce4ec
    ```

## 2\. 配列の形状操作

#### 例3：reshape, flatten, transpose
    
    
    import numpy as np
    
    # 元の配列
    arr = np.arange(12)
    print("元の配列:", arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
    print("形状:", arr.shape)  # (12,)
    
    # reshape: 形状を変更
    reshaped = arr.reshape(3, 4)  # 3行4列に変形
    print("\nreshape (3, 4):")
    print(reshaped)
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]]
    
    # -1を使った自動計算
    reshaped2 = arr.reshape(2, -1)  # 2行、列数は自動計算
    print("\nreshape (2, -1):")
    print(reshaped2)  # 2行6列になる
    
    # flatten: 1次元配列に変換
    flattened = reshaped.flatten()
    print("\nflatten:", flattened)
    # [ 0  1  2  3  4  5  6  7  8  9 10 11]
    
    # transpose: 転置（行と列を入れ替え）
    transposed = reshaped.T
    print("\ntranspose:")
    print(transposed)
    # [[ 0  4  8]
    #  [ 1  5  9]
    #  [ 2  6 10]
    #  [ 3  7 11]]
    
    # 多次元配列の軸入れ替え
    arr3d = np.arange(24).reshape(2, 3, 4)
    print("\n3次元配列の形状:", arr3d.shape)  # (2, 3, 4)
    swapped = np.swapaxes(arr3d, 0, 2)
    print("軸入れ替え後:", swapped.shape)  # (4, 3, 2)
    

## 3\. インデックスとスライシング

#### 例4：配列要素へのアクセス
    
    
    import numpy as np
    
    # 1次元配列
    arr1d = np.array([10, 20, 30, 40, 50])
    print("配列:", arr1d)
    print("arr1d[0]:", arr1d[0])    # 10
    print("arr1d[-1]:", arr1d[-1])  # 50（最後の要素）
    print("arr1d[1:4]:", arr1d[1:4])  # [20 30 40]
    
    # 2次元配列
    arr2d = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    print("\n2次元配列:")
    print(arr2d)
    
    # 要素アクセス
    print("arr2d[0, 0]:", arr2d[0, 0])  # 1
    print("arr2d[1, 2]:", arr2d[1, 2])  # 7（2行目、3列目）
    
    # 行の取得
    print("1行目:", arr2d[0])        # [1 2 3 4]
    print("全行、2列目:", arr2d[:, 1])  # [ 2  6 10]
    
    # スライシング
    print("部分配列:")
    print(arr2d[0:2, 1:3])
    # [[2 3]
    #  [6 7]]
    
    # ブール（真偽値）インデックス
    mask = arr2d > 5
    print("\n5より大きい要素のマスク:")
    print(mask)
    print("5より大きい要素:", arr2d[mask])
    # [ 6  7  8  9 10 11 12]
    
    # 条件を満たす要素の置換
    arr_copy = arr2d.copy()
    arr_copy[arr_copy > 5] = 0
    print("\n5より大きい要素を0に:")
    print(arr_copy)
    
    
    
    ```mermaid
    graph TD
        A[インデックス] --> B[単一要素]
        A --> C[スライス]
        A --> D[ブールインデックス]
    
        B --> E["arr[i, j]"]
        C --> F["arr[1:3, :]"]
        D --> G["arr[arr > 5]"]
    
        style A fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
        style G fill:#e8f5e9
    ```

## 4\. ユニバーサル関数（Universal Functions）

#### 例5：数学演算
    
    
    import numpy as np
    
    # 配列の作成
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])
    
    # 基本演算（要素ごと）
    print("a + b =", a + b)      # [11 22 33 44]
    print("a - b =", a - b)      # [-9 -18 -27 -36]
    print("a * b =", a * b)      # [10 40 90 160]
    print("a / b =", a / b)      # [0.1 0.1 0.1 0.1]
    print("a ** 2 =", a ** 2)    # [ 1  4  9 16]
    
    # 数学関数
    arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print("\n三角関数:")
    print("sin:", np.sin(arr))
    print("cos:", np.cos(arr))
    print("tan:", np.tan(arr))
    
    # 指数・対数関数
    x = np.array([1, 2, 3, 4])
    print("\n指数・対数:")
    print("exp(x):", np.exp(x))       # e^x
    print("log(x):", np.log(x))       # 自然対数
    print("log10(x):", np.log10(x))   # 常用対数
    print("sqrt(x):", np.sqrt(x))     # 平方根
    
    # その他の関数
    y = np.array([-2.5, -1.5, 0.5, 1.5, 2.5])
    print("\nその他:")
    print("abs(y):", np.abs(y))       # 絶対値
    print("ceil(y):", np.ceil(y))     # 切り上げ
    print("floor(y):", np.floor(y))   # 切り捨て
    print("round(y):", np.round(y))   # 四捨五入
    
    # 最大値・最小値
    print("\n最大・最小:")
    print("max:", np.max(x))          # 4
    print("min:", np.min(x))          # 1
    print("argmax:", np.argmax(x))    # 3（最大値のインデックス）
    print("argmin:", np.argmin(x))    # 0（最小値のインデックス）
    

## 5\. ブロードキャスティング

ブロードキャスティングは、形状の異なる配列間で演算を行うNumPyの強力な機能です。

#### 例6：ブロードキャスティングの例
    
    
    import numpy as np
    
    # スカラーとの演算
    arr = np.array([1, 2, 3, 4])
    print("arr + 10 =", arr + 10)  # [11 12 13 14]
    print("arr * 2 =", arr * 2)    # [2 4 6 8]
    
    # 1次元と2次元の演算
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    vector = np.array([10, 20, 30])
    
    result = matrix + vector
    print("\n行列 + ベクトル:")
    print(result)
    # [[11 22 33]
    #  [14 25 36]
    #  [17 28 39]]
    
    # 行ベクトルと列ベクトル
    row = np.array([[1, 2, 3]])  # 形状: (1, 3)
    col = np.array([[10], [20], [30]])  # 形状: (3, 1)
    
    result2 = row + col
    print("\n行ベクトル + 列ベクトル:")
    print(result2)
    # [[11 12 13]
    #  [21 22 23]
    #  [31 32 33]]
    
    # 実用例: 標準化（平均0、標準偏差1に変換）
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=float)
    mean = data.mean(axis=0)  # 列ごとの平均
    std = data.std(axis=0)    # 列ごとの標準偏差
    
    normalized = (data - mean) / std
    print("\n標準化:")
    print("元データ:")
    print(data)
    print("平均:", mean)
    print("標準偏差:", std)
    print("標準化後:")
    print(normalized)
    
    
    
    ```mermaid
    graph TD
        A[ブロードキャスティング] --> B["スカラー (1,) + 配列 (n,)"]
        A --> C["ベクトル (n,) + 行列 (m,n)"]
        A --> D["行 (1,n) + 列 (m,1)"]
    
        B --> E[全要素に適用]
        C --> F[各行に適用]
        D --> G[格子状に展開]
    
        style A fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
        style G fill:#e8f5e9
    ```

## 6\. 統計関数

#### 例7：統計量の計算
    
    
    import numpy as np
    
    # データの作成
    np.random.seed(42)
    data = np.random.randn(100)  # 100個の正規乱数
    
    # 基本統計量
    print("平均:", np.mean(data))
    print("中央値:", np.median(data))
    print("標準偏差:", np.std(data))
    print("分散:", np.var(data))
    print("最小値:", np.min(data))
    print("最大値:", np.max(data))
    print("範囲:", np.ptp(data))  # max - min
    
    # パーセンタイル
    print("\n四分位数:")
    print("25%:", np.percentile(data, 25))
    print("50%:", np.percentile(data, 50))
    print("75%:", np.percentile(data, 75))
    
    # 2次元配列の統計（軸を指定）
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    print("\n2次元配列の統計:")
    print("全体の合計:", np.sum(matrix))  # 45
    print("列ごとの合計:", np.sum(matrix, axis=0))  # [12 15 18]
    print("行ごとの合計:", np.sum(matrix, axis=1))  # [ 6 15 24]
    
    print("\n列ごとの平均:", np.mean(matrix, axis=0))  # [4. 5. 6.]
    print("行ごとの平均:", np.mean(matrix, axis=1))  # [2. 5. 8.]
    
    # 累積統計
    arr = np.array([1, 2, 3, 4, 5])
    print("\n累積和:", np.cumsum(arr))  # [ 1  3  6 10 15]
    print("累積積:", np.cumprod(arr))   # [  1   2   6  24 120]
    

## 7\. 線形代数

#### 例8：行列演算
    
    
    import numpy as np
    
    # 行列の作成
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # 要素ごとの積
    print("要素ごとの積 (A * B):")
    print(A * B)
    # [[ 5 12]
    #  [21 32]]
    
    # 行列の積（内積）
    print("\n行列の積 (A @ B):")
    print(A @ B)  # または np.dot(A, B)
    # [[19 22]
    #  [43 50]]
    
    # ベクトルの内積
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("\nベクトルの内積:", np.dot(v1, v2))  # 32 = 1*4 + 2*5 + 3*6
    
    # 転置
    print("\nAの転置:")
    print(A.T)
    
    # 逆行列
    A_inv = np.linalg.inv(A)
    print("\nAの逆行列:")
    print(A_inv)
    
    # 単位行列になることを確認
    print("\nA @ A_inv（単位行列になる）:")
    print(A @ A_inv)
    
    # 行列式
    det = np.linalg.det(A)
    print("\nAの行列式:", det)
    
    # 固有値と固有ベクトル
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\n固有値:", eigenvalues)
    print("固有ベクトル:")
    print(eigenvectors)
    
    # ノルム（ベクトルの大きさ）
    v = np.array([3, 4])
    print("\nL2ノルム:", np.linalg.norm(v))  # 5.0 = sqrt(3^2 + 4^2)
    

## 8\. 実践例：データ前処理

#### 例9：機械学習のためのデータ前処理
    
    
    import numpy as np
    
    # サンプルデータ（身長、体重、年齢）
    np.random.seed(42)
    data = np.random.randn(100, 3) * 10 + [170, 60, 30]
    print("データの形状:", data.shape)  # (100, 3)
    print("最初の5行:")
    print(data[:5])
    
    # 1. 基本統計量
    print("\n=== 基本統計量 ===")
    print("平均:", data.mean(axis=0))
    print("標準偏差:", data.std(axis=0))
    print("最小値:", data.min(axis=0))
    print("最大値:", data.max(axis=0))
    
    # 2. 欠損値の処理（NaNを含むデータ）
    data_with_nan = data.copy()
    data_with_nan[0, 0] = np.nan
    data_with_nan[5, 1] = np.nan
    
    print("\n=== 欠損値処理 ===")
    print("欠損値の数:", np.isnan(data_with_nan).sum())
    
    # 欠損値を平均で補完
    for col in range(data_with_nan.shape[1]):
        col_mean = np.nanmean(data_with_nan[:, col])
        data_with_nan[np.isnan(data_with_nan[:, col]), col] = col_mean
    
    print("補完後の欠損値の数:", np.isnan(data_with_nan).sum())
    
    # 3. 標準化（Z-score normalization）
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized_data = (data - mean) / std
    
    print("\n=== 標準化後 ===")
    print("平均:", normalized_data.mean(axis=0))  # ほぼ0
    print("標準偏差:", normalized_data.std(axis=0))  # ほぼ1
    
    # 4. 最小-最大正規化（0-1に変換）
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    min_max_scaled = (data - min_vals) / (max_vals - min_vals)
    
    print("\n=== 最小-最大正規化後 ===")
    print("最小値:", min_max_scaled.min(axis=0))  # 0
    print("最大値:", min_max_scaled.max(axis=0))  # 1
    

#### 例10：画像データの操作
    
    
    import numpy as np
    
    # 画像を模したデータ（5x5のグレースケール画像）
    image = np.array([
        [0, 50, 100, 150, 200],
        [50, 100, 150, 200, 250],
        [100, 150, 200, 250, 255],
        [150, 200, 250, 255, 255],
        [200, 250, 255, 255, 255]
    ], dtype=np.uint8)
    
    print("元の画像:")
    print(image)
    print("形状:", image.shape)  # (5, 5)
    
    # 1. 画像の反転
    flipped_h = np.flip(image, axis=1)  # 水平反転
    flipped_v = np.flip(image, axis=0)  # 垂直反転
    
    print("\n水平反転:")
    print(flipped_h)
    
    # 2. 画像の回転（90度）
    rotated = np.rot90(image)
    print("\n90度回転:")
    print(rotated)
    
    # 3. 画像の切り抜き
    cropped = image[1:4, 1:4]  # 中央3x3を切り抜き
    print("\n切り抜き (3x3):")
    print(cropped)
    
    # 4. 明るさ調整
    brightened = np.clip(image + 50, 0, 255).astype(np.uint8)
    print("\n明るさ+50:")
    print(brightened)
    
    # 5. コントラスト調整
    contrast = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    print("\nコントラスト1.5倍:")
    print(contrast)
    
    # 6. RGB画像の模擬（5x5x3）
    rgb_image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    print("\nRGB画像の形状:", rgb_image.shape)  # (5, 5, 3)
    
    # チャンネルごとの平均
    print("Rチャンネル平均:", rgb_image[:, :, 0].mean())
    print("Gチャンネル平均:", rgb_image[:, :, 1].mean())
    print("Bチャンネル平均:", rgb_image[:, :, 2].mean())
    

## まとめ

この章では、NumPyの基礎を学びました：

  * ✅ **配列作成** : np.array, np.zeros, np.ones, np.arange, np.linspace
  * ✅ **形状操作** : reshape, flatten, transpose
  * ✅ **インデックス** : スライス、ブールインデックス
  * ✅ **数学演算** : ユニバーサル関数、三角関数、指数対数
  * ✅ **ブロードキャスティング** : 形状の異なる配列間の演算
  * ✅ **統計関数** : mean, std, min, max, sum
  * ✅ **線形代数** : 内積、行列積、逆行列、固有値

**次のステップ** : 第3章では、これらの知識を使って、Pandasでデータ分析を学びます。

## 演習問題

演習1：配列操作

**問題** : 1から50までの数字を含む配列を作成し、5行10列の行列に変形してください。次に、各行の合計を計算してください。
    
    
    # 解答例
    import numpy as np
    
    # 1から50までの配列
    arr = np.arange(1, 51)
    print("配列:", arr)
    
    # 5行10列に変形
    matrix = arr.reshape(5, 10)
    print("\n5x10行列:")
    print(matrix)
    
    # 各行の合計
    row_sums = matrix.sum(axis=1)
    print("\n各行の合計:", row_sums)
    # 出力: [ 55 155 255 355 455]
    
    # 検証: 最初の行の合計
    print("検証:", sum(range(1, 11)))  # 55
    

演習2：ブールインデックス

**問題** : 0から99までの100個の数字から、3の倍数かつ5の倍数でない数字を抽出してください。
    
    
    # 解答例
    import numpy as np
    
    # 0から99までの配列
    numbers = np.arange(100)
    
    # 条件: 3の倍数かつ5の倍数でない
    condition = (numbers % 3 == 0) & (numbers % 5 != 0)
    result = numbers[condition]
    
    print("結果:", result)
    print("個数:", len(result))
    # 出力: [ 3  6  9 12 18 21 24 27 33 36 39 42 48 51 54 57 63 66 69 72 78 81 84 87 93 96 99]
    # 個数: 27
    

演習3：統計処理

**問題** : 平均50、標準偏差10の正規分布に従う1000個のデータを生成し、ヒストグラムの各区間の度数を計算してください（区間: 0-20, 20-40, 40-60, 60-80, 80-100）。
    
    
    # 解答例
    import numpy as np
    
    # 正規分布データの生成
    np.random.seed(42)
    data = np.random.normal(50, 10, 1000)
    
    # 統計量
    print("平均:", data.mean())
    print("標準偏差:", data.std())
    
    # ヒストグラム（度数計算）
    bins = [0, 20, 40, 60, 80, 100]
    hist, edges = np.histogram(data, bins=bins)
    
    print("\nヒストグラム:")
    for i in range(len(hist)):
        print(f"{edges[i]}-{edges[i+1]}: {hist[i]}個")
    
    # 出力例:
    # 0-20: 22個
    # 20-40: 159個
    # 40-60: 638個
    # 60-80: 175個
    # 80-100: 6個
    

演習4：行列演算

**問題** : 以下の行列Aに対して、(1) 逆行列、(2) 行列式、(3) 固有値を求めてください。また、A @ A_inv が単位行列になることを確認してください。

\\[ A = \begin{bmatrix} 2 & 1 \\\ 1 & 3 \end{bmatrix} \\]
    
    
    # 解答例
    import numpy as np
    
    A = np.array([[2, 1],
                  [1, 3]])
    
    # (1) 逆行列
    A_inv = np.linalg.inv(A)
    print("逆行列:")
    print(A_inv)
    
    # (2) 行列式
    det = np.linalg.det(A)
    print("\n行列式:", det)  # 5.0
    
    # (3) 固有値
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\n固有値:", eigenvalues)
    print("固有ベクトル:")
    print(eigenvectors)
    
    # 検証: A @ A_inv = I (単位行列)
    identity = A @ A_inv
    print("\nA @ A_inv（単位行列）:")
    print(identity)
    print("単位行列との差:", np.allclose(identity, np.eye(2)))
    

演習5：画像データ処理

**問題** : 10x10のランダムな画像データ（0-255）を生成し、(1) 画像全体を2倍明るくし、(2) 中央の5x5領域を抽出してください。ただし、明るさは0-255の範囲に制限してください。
    
    
    # 解答例
    import numpy as np
    
    # ランダム画像の生成
    np.random.seed(42)
    image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    
    print("元の画像:")
    print(image)
    print("平均明るさ:", image.mean())
    
    # (1) 2倍明るく（0-255に制限）
    brightened = np.clip(image * 2, 0, 255).astype(np.uint8)
    print("\n2倍明るく:")
    print(brightened)
    print("平均明るさ:", brightened.mean())
    
    # (2) 中央5x5を抽出（インデックス2:7）
    center = image[2:7, 2:7]
    print("\n中央5x5:")
    print(center)
    print("形状:", center.shape)  # (5, 5)
    

[← 第1章: Python基礎](<./chapter1-python-basics.html>) [第3章: Pandas基礎 →](<./chapter3-pandas-basics.html>)
