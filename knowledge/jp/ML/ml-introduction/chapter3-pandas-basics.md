---
title: 第3章：Pandas基礎
chapter_title: 第3章：Pandas基礎
---

**データ分析の必須ライブラリPandasをマスターしよう**

## はじめに

**Pandas** は、Pythonでデータ分析を行うための最も重要なライブラリです。表形式のデータ（エクセルやCSVファイルのようなデータ）を簡単に扱うことができ、機械学習のデータ前処理には欠かせません。

この章では、以下の内容を学びます：

  * SeriesとDataFrameの基本
  * CSVファイルの読み込みと保存
  * データの選択と抽出
  * データクリーニング（欠損値処理）
  * データ変換と集計
  * データ結合と可視化

> **PandasとNumPyの違い**  
>  NumPyは数値計算に特化していますが、Pandasは表形式のデータ（行と列を持つデータ）を扱うのに特化しています。PandasはNumPyの上に構築されており、内部でNumPy配列を使用しています。 

## 1\. Seriesの基本

**Series** は、1次元のデータ構造です。インデックス（ラベル）付きの配列と考えてください。

#### 例1：Seriesの作成と操作
    
    
    import pandas as pd
    import numpy as np
    
    # リストからSeriesを作成
    s1 = pd.Series([10, 20, 30, 40, 50])
    print("Series 1:")
    print(s1)
    # 出力:
    # 0    10
    # 1    20
    # 2    30
    # 3    40
    # 4    50
    # dtype: int64
    
    # インデックスを指定
    s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print("\nSeries 2:")
    print(s2)
    # 出力:
    # a    10
    # b    20
    # c    30
    # dtype: int64
    
    # 辞書からSeriesを作成
    data_dict = {'東京': 1400, '大阪': 880, '名古屋': 230, '福岡': 155}
    population = pd.Series(data_dict)
    print("\n人口（万人）:")
    print(population)
    
    # 要素へのアクセス
    print("\n大阪の人口:", population['大阪'])  # 880
    print("最初の2つ:")
    print(population[:2])
    
    # 統計量
    print("\n統計量:")
    print("平均:", population.mean())
    print("最大:", population.max())
    print("最小:", population.min())
    print("合計:", population.sum())
    
    # ブール演算
    print("\n人口500万人以上の都市:")
    print(population[population >= 500])
    

## 2\. DataFrameの基本

**DataFrame** は、2次元の表形式データ構造です。エクセルのスプレッドシートに似ています。

#### 例2：DataFrameの作成
    
    
    import pandas as pd
    
    # 辞書からDataFrameを作成
    data = {
        '名前': ['太郎', '花子', '次郎', '桃子'],
        '年齢': [25, 30, 22, 28],
        '都市': ['東京', '大阪', '名古屋', '福岡'],
        '年収': [450, 520, 380, 490]
    }
    df = pd.DataFrame(data)
    print("DataFrame:")
    print(df)
    # 出力:
    #    名前  年齢   都市  年収
    # 0  太郎  25   東京  450
    # 1  花子  30   大阪  520
    # 2  次郎  22  名古屋  380
    # 3  桃子  28   福岡  490
    
    # DataFrameの情報
    print("\n形状:", df.shape)  # (4, 4) = 4行4列
    print("列名:", df.columns.tolist())
    print("インデックス:", df.index.tolist())
    print("データ型:")
    print(df.dtypes)
    
    # 基本統計量
    print("\n統計量:")
    print(df.describe())
    
    # 最初と最後の数行を表示
    print("\n最初の2行:")
    print(df.head(2))
    
    print("\n最後の2行:")
    print(df.tail(2))
    
    # 情報の概要
    print("\nDataFrame情報:")
    print(df.info())
    
    
    
    ```mermaid
    graph LR
        A[Pandas] --> B[Series]
        A --> C[DataFrame]
    
        B --> D[1次元データ]
        C --> E[2次元表形式]
    
        D --> F[インデックス付き配列]
        E --> G[行と列を持つテーブル]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

## 3\. CSVファイルの読み込みと保存

#### 例3：CSV操作
    
    
    import pandas as pd
    
    # サンプルデータを作成
    data = {
        '日付': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
        '商品': ['りんご', 'バナナ', 'オレンジ', 'りんご'],
        '数量': [10, 15, 8, 12],
        '価格': [100, 80, 120, 100]
    }
    df = pd.DataFrame(data)
    
    # CSVファイルに保存
    df.to_csv('sales.csv', index=False, encoding='utf-8')
    print("CSVファイルを保存しました。")
    
    # CSVファイルの読み込み
    df_loaded = pd.read_csv('sales.csv')
    print("\n読み込んだデータ:")
    print(df_loaded)
    
    # 日付型に変換
    df_loaded['日付'] = pd.to_datetime(df_loaded['日付'])
    print("\n日付型変換後:")
    print(df_loaded.dtypes)
    
    # 特定の列だけ読み込み
    df_partial = pd.read_csv('sales.csv', usecols=['商品', '数量'])
    print("\n特定列のみ:")
    print(df_partial)
    
    # ヘッダーがないCSVの読み込み
    # pd.read_csv('data.csv', header=None, names=['col1', 'col2'])
    
    # 区切り文字が異なる場合
    # pd.read_csv('data.tsv', sep='\t')  # TSVファイル
    
    # 大きなファイルを分割して読み込み
    # for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    #     process(chunk)
    

## 4\. データの選択と抽出

#### 例4：loc, iloc, boolean indexing
    
    
    import pandas as pd
    
    # サンプルデータ
    data = {
        '名前': ['太郎', '花子', '次郎', '桃子', '五郎'],
        '年齢': [25, 30, 22, 28, 35],
        '都市': ['東京', '大阪', '名古屋', '福岡', '東京'],
        '年収': [450, 520, 380, 490, 600]
    }
    df = pd.DataFrame(data)
    print("元データ:")
    print(df)
    
    # 列の選択
    print("\n年齢列:")
    print(df['年齢'])
    
    # 複数列の選択
    print("\n名前と年収:")
    print(df[['名前', '年収']])
    
    # loc: ラベルベースのインデックス
    print("\nインデックス0の行:")
    print(df.loc[0])
    
    print("\nインデックス0-2の名前と年齢:")
    print(df.loc[0:2, ['名前', '年齢']])
    
    # iloc: 位置ベースのインデックス
    print("\n最初の2行、最初の2列:")
    print(df.iloc[0:2, 0:2])
    
    # ブールインデックス
    print("\n年齢30歳以上:")
    print(df[df['年齢'] >= 30])
    
    print("\n東京在住:")
    print(df[df['都市'] == '東京'])
    
    # 複数条件（&: AND、|: OR）
    print("\n東京在住かつ年収500万以上:")
    print(df[(df['都市'] == '東京') & (df['年収'] >= 500)])
    
    # isin(): 複数の値のいずれかに一致
    cities = ['東京', '大阪']
    print("\n東京または大阪在住:")
    print(df[df['都市'].isin(cities)])
    
    # 文字列操作
    print("\n名前に「子」が含まれる:")
    print(df[df['名前'].str.contains('子')])
    

## 5\. データクリーニング（欠損値処理）

#### 例5：欠損値の処理
    
    
    import pandas as pd
    import numpy as np
    
    # 欠損値を含むデータ
    data = {
        '名前': ['太郎', '花子', None, '桃子', '五郎'],
        '年齢': [25, 30, 22, np.nan, 35],
        '年収': [450, np.nan, 380, 490, 600]
    }
    df = pd.DataFrame(data)
    print("欠損値を含むデータ:")
    print(df)
    
    # 欠損値の確認
    print("\n欠損値の数:")
    print(df.isnull().sum())
    
    print("\n欠損値があるか（行ごと）:")
    print(df.isnull().any(axis=1))
    
    # 欠損値を含む行を削除
    df_dropped = df.dropna()
    print("\n欠損値を含む行を削除:")
    print(df_dropped)
    
    # 欠損値を含む列を削除
    df_dropped_col = df.dropna(axis=1)
    print("\n欠損値を含む列を削除:")
    print(df_dropped_col)
    
    # 欠損値を特定の値で埋める
    df_filled = df.fillna(0)
    print("\n欠損値を0で埋める:")
    print(df_filled)
    
    # 列ごとに異なる値で埋める
    df_filled2 = df.fillna({'名前': '未記入', '年齢': df['年齢'].mean(), '年収': df['年収'].median()})
    print("\n列ごとに異なる値で埋める:")
    print(df_filled2)
    
    # 前方埋め・後方埋め
    df_ffill = df.fillna(method='ffill')  # 前の値で埋める
    df_bfill = df.fillna(method='bfill')  # 次の値で埋める
    
    # 重複の処理
    data_dup = {
        '名前': ['太郎', '花子', '太郎', '次郎'],
        '年齢': [25, 30, 25, 22]
    }
    df_dup = pd.DataFrame(data_dup)
    print("\n重複を含むデータ:")
    print(df_dup)
    
    print("\n重複の確認:")
    print(df_dup.duplicated())
    
    print("\n重複を削除:")
    print(df_dup.drop_duplicates())
    
    
    
    ```mermaid
    graph TD
        A[欠損値処理] --> B[削除]
        A --> C[補完]
    
        B --> D[dropna 行削除]
        B --> E[dropna 列削除]
    
        C --> F[固定値で埋める]
        C --> G[統計量で埋める]
        C --> H[前後の値で埋める]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

## 6\. データ変換

#### 例6：apply, map, replace
    
    
    import pandas as pd
    
    # サンプルデータ
    data = {
        '名前': ['太郎', '花子', '次郎', '桃子'],
        '年齢': [25, 30, 22, 28],
        '年収': [450, 520, 380, 490]
    }
    df = pd.DataFrame(data)
    print("元データ:")
    print(df)
    
    # apply: 関数を適用
    df['年収（千円）'] = df['年収'].apply(lambda x: x * 1000)
    print("\n年収を千円単位に変換:")
    print(df)
    
    # 年齢カテゴリを追加
    def age_category(age):
        if age < 25:
            return '若手'
        elif age < 30:
            return '中堅'
        else:
            return 'ベテラン'
    
    df['カテゴリ'] = df['年齢'].apply(age_category)
    print("\n年齢カテゴリを追加:")
    print(df)
    
    # map: 辞書でマッピング
    name_english = {'太郎': 'Taro', '花子': 'Hanako', '次郎': 'Jiro', '桃子': 'Momoko'}
    df['英名'] = df['名前'].map(name_english)
    print("\n英名を追加:")
    print(df)
    
    # replace: 値の置換
    df_replaced = df.replace({'カテゴリ': {'若手': 'Junior', '中堅': 'Mid', 'ベテラン': 'Senior'}})
    print("\nカテゴリを英語に:")
    print(df_replaced)
    
    # 新しい列を追加（計算）
    df['税引後年収'] = df['年収'] * 0.8
    print("\n税引後年収を追加:")
    print(df[['名前', '年収', '税引後年収']])
    
    # 列の削除
    df_dropped = df.drop(['年収（千円）', '英名'], axis=1)
    print("\n不要な列を削除:")
    print(df_dropped)
    
    # 列名の変更
    df_renamed = df.rename(columns={'年齢': 'Age', '年収': 'Salary'})
    print("\n列名を変更:")
    print(df_renamed)
    

## 7\. グループ化と集計

#### 例7：groupbyとagg
    
    
    import pandas as pd
    
    # 売上データ
    data = {
        '日付': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-02', '2025-01-03'],
        '商品': ['りんご', 'バナナ', 'りんご', 'オレンジ', 'バナナ'],
        '店舗': ['東京', '東京', '大阪', '東京', '大阪'],
        '数量': [10, 15, 12, 8, 20],
        '売上': [1000, 1200, 1200, 960, 1600]
    }
    df = pd.DataFrame(data)
    print("売上データ:")
    print(df)
    
    # 商品ごとの集計
    print("\n商品ごとの売上合計:")
    print(df.groupby('商品')['売上'].sum())
    
    # 複数の統計量
    print("\n商品ごとの統計:")
    print(df.groupby('商品').agg({
        '数量': 'sum',
        '売上': ['sum', 'mean', 'count']
    }))
    
    # 複数列でグループ化
    print("\n商品×店舗ごとの集計:")
    grouped = df.groupby(['商品', '店舗'])['売上'].sum()
    print(grouped)
    
    # ピボットテーブル
    print("\nピボットテーブル:")
    pivot = df.pivot_table(
        values='売上',
        index='商品',
        columns='店舗',
        aggfunc='sum',
        fill_value=0
    )
    print(pivot)
    
    # カスタム集計関数
    print("\n商品ごとの売上範囲:")
    print(df.groupby('商品')['売上'].agg(lambda x: x.max() - x.min()))
    
    # 集計後のフィルタリング
    print("\n合計売上が2000以上の商品:")
    grouped_filtered = df.groupby('商品')['売上'].sum()
    print(grouped_filtered[grouped_filtered >= 2000])
    

## 8\. データの結合

#### 例8：merge, concat
    
    
    import pandas as pd
    
    # 顧客データ
    customers = pd.DataFrame({
        '顧客ID': [1, 2, 3, 4],
        '名前': ['太郎', '花子', '次郎', '桃子']
    })
    
    # 注文データ
    orders = pd.DataFrame({
        '注文ID': [101, 102, 103, 104],
        '顧客ID': [1, 2, 1, 3],
        '金額': [5000, 3000, 7000, 4500]
    })
    
    print("顧客データ:")
    print(customers)
    print("\n注文データ:")
    print(orders)
    
    # merge: 結合
    merged = pd.merge(customers, orders, on='顧客ID', how='inner')
    print("\n内部結合 (inner join):")
    print(merged)
    
    # 左外部結合
    merged_left = pd.merge(customers, orders, on='顧客ID', how='left')
    print("\n左外部結合 (left join):")
    print(merged_left)
    
    # 右外部結合
    merged_right = pd.merge(customers, orders, on='顧客ID', how='right')
    print("\n右外部結合 (right join):")
    print(merged_right)
    
    # 完全外部結合
    merged_outer = pd.merge(customers, orders, on='顧客ID', how='outer')
    print("\n完全外部結合 (outer join):")
    print(merged_outer)
    
    # concat: 縦方向の連結
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    
    concatenated = pd.concat([df1, df2], ignore_index=True)
    print("\n縦方向の連結:")
    print(concatenated)
    
    # 横方向の連結
    concatenated_h = pd.concat([df1, df2], axis=1)
    print("\n横方向の連結:")
    print(concatenated_h)
    
    
    
    ```mermaid
    graph LR
        A[データ結合] --> B[merge]
        A --> C[concat]
    
        B --> D[inner join]
        B --> E[left join]
        B --> F[right join]
        B --> G[outer join]
    
        C --> H[縦方向 axis=0]
        C --> I[横方向 axis=1]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

## 9\. データ可視化の基礎

#### 例9：matplotlibとの連携
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 月別売上データ
    data = {
        '月': ['1月', '2月', '3月', '4月', '5月', '6月'],
        '売上': [120, 135, 150, 142, 168, 175],
        '利益': [30, 35, 42, 38, 48, 52]
    }
    df = pd.DataFrame(data)
    
    # 折れ線グラフ
    df.plot(x='月', y='売上', kind='line', marker='o', title='月別売上推移')
    plt.xlabel('月')
    plt.ylabel('売上（万円）')
    # plt.savefig('sales_trend.png')
    # plt.show()
    
    # 複数系列
    df.plot(x='月', y=['売上', '利益'], kind='line', marker='o', title='月別売上と利益')
    # plt.show()
    
    # 棒グラフ
    df.plot(x='月', y='売上', kind='bar', title='月別売上（棒グラフ）')
    # plt.show()
    
    # ヒストグラム
    np.random.seed(42)
    scores = pd.DataFrame({'点数': np.random.normal(70, 10, 100)})
    scores.plot(kind='hist', bins=20, title='点数分布', edgecolor='black')
    plt.xlabel('点数')
    plt.ylabel('人数')
    # plt.show()
    
    # 散布図
    data_scatter = {
        '勉強時間': [1, 2, 3, 4, 5, 6, 7, 8],
        '点数': [50, 55, 65, 70, 75, 80, 85, 90]
    }
    df_scatter = pd.DataFrame(data_scatter)
    df_scatter.plot(kind='scatter', x='勉強時間', y='点数', title='勉強時間と点数の関係')
    # plt.show()
    
    # ボックスプロット
    data_box = {
        'クラスA': np.random.normal(75, 10, 30),
        'クラスB': np.random.normal(70, 15, 30),
        'クラスC': np.random.normal(80, 8, 30)
    }
    df_box = pd.DataFrame(data_box)
    df_box.plot(kind='box', title='クラス別点数分布')
    # plt.show()
    
    print("グラフを生成しました。")
    

## 10\. 実践例：データ分析ワークフロー

#### 例10：総合的なデータ分析
    
    
    import pandas as pd
    import numpy as np
    
    # サンプルデータの作成（従業員データ）
    np.random.seed(42)
    n = 100
    
    data = {
        '従業員ID': range(1, n + 1),
        '名前': [f'社員{i}' for i in range(1, n + 1)],
        '部署': np.random.choice(['営業', '開発', '人事', '財務'], n),
        '年齢': np.random.randint(22, 60, n),
        '勤続年数': np.random.randint(1, 30, n),
        '年収': np.random.randint(300, 1000, n),
        '評価': np.random.choice(['A', 'B', 'C', 'D'], n, p=[0.1, 0.3, 0.4, 0.2])
    }
    df = pd.DataFrame(data)
    
    # 欠損値をランダムに追加
    df.loc[np.random.choice(df.index, 5), '年収'] = np.nan
    
    print("=== 1. データの概要 ===")
    print(f"データ形状: {df.shape}")
    print(f"\n最初の5行:")
    print(df.head())
    
    print("\n=== 2. データ型と欠損値 ===")
    print(df.info())
    
    print("\n=== 3. 欠損値処理 ===")
    print(f"欠損値の数: {df.isnull().sum().sum()}")
    df['年収'] = df['年収'].fillna(df['年収'].median())
    print(f"補完後の欠損値: {df.isnull().sum().sum()}")
    
    print("\n=== 4. 基本統計量 ===")
    print(df.describe())
    
    print("\n=== 5. 部署別分析 ===")
    dept_analysis = df.groupby('部署').agg({
        '年齢': 'mean',
        '勤続年数': 'mean',
        '年収': ['mean', 'median', 'count']
    })
    print(dept_analysis)
    
    print("\n=== 6. 評価別分析 ===")
    rating_analysis = df.groupby('評価')['年収'].describe()
    print(rating_analysis)
    
    print("\n=== 7. 相関分析 ===")
    correlation = df[['年齢', '勤続年数', '年収']].corr()
    print(correlation)
    
    print("\n=== 8. 高年収者の抽出 ===")
    high_earners = df[df['年収'] >= 800]
    print(f"年収800万以上: {len(high_earners)}名")
    print(high_earners[['名前', '部署', '年収', '評価']])
    
    print("\n=== 9. 部署×評価のクロス集計 ===")
    crosstab = pd.crosstab(df['部署'], df['評価'])
    print(crosstab)
    
    print("\n=== 10. データのエクスポート ===")
    # 分析結果をCSVに保存
    df.to_csv('employee_data.csv', index=False, encoding='utf-8')
    dept_analysis.to_csv('dept_analysis.csv', encoding='utf-8')
    print("分析結果を保存しました。")
    

## まとめ

この章では、Pandasの基礎を学びました：

  * ✅ **Series/DataFrame** : 基本的なデータ構造
  * ✅ **CSV操作** : read_csv, to_csv
  * ✅ **データ選択** : loc, iloc, boolean indexing
  * ✅ **欠損値処理** : dropna, fillna
  * ✅ **データ変換** : apply, map, replace
  * ✅ **集計** : groupby, agg, pivot_table
  * ✅ **結合** : merge, concat
  * ✅ **可視化** : plot（matplotlib連携）

**次のステップ** : 第4章では、これまで学んだPython、NumPy、Pandasの知識を使って、機械学習の概要を学びます。

## 演習問題

演習1：DataFrame操作

**問題** : 以下のデータからDataFrameを作成し、(1) 平均点が80点以上の学生を抽出、(2) 各科目の平均点を計算してください。
    
    
    # データ
    data = {
        '名前': ['太郎', '花子', '次郎', '桃子', '五郎'],
        '数学': [85, 92, 78, 88, 95],
        '英語': [78, 88, 82, 90, 85],
        '国語': [82, 85, 75, 92, 88]
    }
    
    # 解答例
    import pandas as pd
    
    df = pd.DataFrame(data)
    print("学生データ:")
    print(df)
    
    # (1) 平均点を計算
    df['平均'] = df[['数学', '英語', '国語']].mean(axis=1)
    print("\n平均点追加:")
    print(df)
    
    # 平均80点以上
    high_scorers = df[df['平均'] >= 80]
    print("\n平均80点以上の学生:")
    print(high_scorers)
    
    # (2) 各科目の平均
    subject_avg = df[['数学', '英語', '国語']].mean()
    print("\n各科目の平均点:")
    print(subject_avg)
    

演習2：欠損値処理

**問題** : 以下のデータで、欠損値を(1) 各列の平均値で補完し、(2) 補完前後の統計量を比較してください。
    
    
    import pandas as pd
    import numpy as np
    
    # 欠損値を含むデータ
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, np.nan, 500]
    }
    df = pd.DataFrame(data)
    
    # 解答例
    print("元データ:")
    print(df)
    
    print("\n補完前の統計量:")
    print(df.describe())
    
    # 平均値で補完
    df_filled = df.fillna(df.mean())
    print("\n補完後のデータ:")
    print(df_filled)
    
    print("\n補完後の統計量:")
    print(df_filled.describe())
    
    print("\n変化:")
    for col in df.columns:
        before = df[col].mean()
        after = df_filled[col].mean()
        print(f"{col}列の平均: {before:.2f} → {after:.2f}")
    

演習3：グループ化と集計

**問題** : 以下の売上データを(1) 店舗ごとに集計し、(2) 商品カテゴリごとの売上上位3店舗を抽出してください。
    
    
    import pandas as pd
    
    data = {
        '店舗': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
        'カテゴリ': ['食品', '食品', '衣料', '食品', '衣料', '衣料', '食品', '食品'],
        '売上': [100, 150, 200, 120, 180, 220, 110, 160]
    }
    df = pd.DataFrame(data)
    
    # 解答例
    print("売上データ:")
    print(df)
    
    # (1) 店舗ごとの集計
    store_summary = df.groupby('店舗')['売上'].agg(['sum', 'mean', 'count'])
    print("\n店舗ごとの集計:")
    print(store_summary)
    
    # (2) カテゴリ×店舗のピボット
    pivot = df.pivot_table(values='売上', index='カテゴリ', columns='店舗', aggfunc='sum', fill_value=0)
    print("\nカテゴリ×店舗ピボット:")
    print(pivot)
    
    # 各カテゴリの上位3店舗
    print("\nカテゴリ別上位店舗:")
    for category in df['カテゴリ'].unique():
        cat_data = df[df['カテゴリ'] == category]
        top3 = cat_data.groupby('店舗')['売上'].sum().nlargest(3)
        print(f"\n{category}:")
        print(top3)
    

演習4：データ結合

**問題** : 顧客マスタと注文データを結合し、各顧客の注文回数と合計金額を集計してください。
    
    
    import pandas as pd
    
    # 顧客マスタ
    customers = pd.DataFrame({
        '顧客ID': [1, 2, 3, 4, 5],
        '名前': ['太郎', '花子', '次郎', '桃子', '五郎'],
        '会員ランク': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver']
    })
    
    # 注文データ
    orders = pd.DataFrame({
        '注文ID': [101, 102, 103, 104, 105, 106],
        '顧客ID': [1, 2, 1, 3, 2, 1],
        '金額': [5000, 3000, 7000, 4500, 2000, 6000]
    })
    
    # 解答例
    print("顧客マスタ:")
    print(customers)
    print("\n注文データ:")
    print(orders)
    
    # 結合
    merged = pd.merge(customers, orders, on='顧客ID', how='left')
    print("\n結合後:")
    print(merged)
    
    # 顧客ごとの集計
    customer_summary = merged.groupby(['顧客ID', '名前', '会員ランク'])['金額'].agg(['count', 'sum'])
    customer_summary.columns = ['注文回数', '合計金額']
    customer_summary = customer_summary.reset_index()
    print("\n顧客別集計:")
    print(customer_summary)
    
    # 会員ランク別の集計
    rank_summary = customer_summary.groupby('会員ランク')['合計金額'].mean()
    print("\n会員ランク別平均購入額:")
    print(rank_summary)
    

演習5：総合問題

**問題** : 月別売上データを分析し、(1) 前月比成長率を計算、(2) 3ヶ月移動平均を計算、(3) グラフで可視化してください。
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    data = {
        '月': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'],
        '売上': [120, 135, 150, 142, 168, 175]
    }
    df = pd.DataFrame(data)
    
    # 解答例
    print("月別売上:")
    print(df)
    
    # (1) 前月比成長率
    df['前月比'] = df['売上'].pct_change() * 100
    print("\n前月比成長率:")
    print(df)
    
    # (2) 3ヶ月移動平均
    df['3ヶ月移動平均'] = df['売上'].rolling(window=3).mean()
    print("\n移動平均追加:")
    print(df)
    
    # (3) 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(df['月'], df['売上'], marker='o', label='売上')
    plt.plot(df['月'], df['3ヶ月移動平均'], marker='s', linestyle='--', label='3ヶ月移動平均')
    plt.xlabel('月')
    plt.ylabel('売上（万円）')
    plt.title('月別売上推移')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('sales_analysis.png')
    # plt.show()
    
    print("\nグラフを作成しました。")
    

[← 第2章: NumPy基礎](<./chapter2-numpy-basics.html>) [第4章: 機械学習の概要 →](<./chapter4-ml-overview.html>)
