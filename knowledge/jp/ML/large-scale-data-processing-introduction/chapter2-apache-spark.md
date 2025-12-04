---
title: 第2章：Apache Sparkによる分散機械学習
chapter_title: 第2章：Apache Sparkによる分散機械学習
subtitle: ビッグデータMLの実践基盤 - Sparkで加速する大規模データ処理
reading_time: 35-40分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Sparkアーキテクチャと分散処理の仕組みを理解する
  * ✅ RDD、DataFrame、DatasetのAPIを使い分けられる
  * ✅ Spark SQLで効率的なデータ操作ができる
  * ✅ Spark MLlibで分散機械学習を実装できる
  * ✅ パフォーマンス最適化の手法を適用できる
  * ✅ 実データで大規模ML処理を実行できる

* * *

## 2.1 Sparkアーキテクチャ

### Apache Sparkとは

**Apache Spark** は、大規模データの高速分散処理フレームワークです。MapReduceの100倍以上の速度を実現し、機械学習、ストリーム処理、グラフ処理に対応します。

> 「MapReduceの後継者」- メモリ内処理により、反復処理が圧倒的に高速化します。

### Sparkの主要コンポーネント

コンポーネント | 用途 | 特徴  
---|---|---  
**Spark Core** | 基本処理エンジン | RDD、タスクスケジューリング  
**Spark SQL** | 構造化データ処理 | DataFrame、SQL クエリ  
**Spark MLlib** | 機械学習 | 分散ML、パイプライン  
**Spark Streaming** | ストリーム処理 | リアルタイムデータ処理  
**GraphX** | グラフ処理 | ネットワーク解析  
  
### Driver と Executor の関係
    
    
    ```mermaid
    graph TB
        subgraph "Driver Program"
            A[SparkContext]
            B[DAG Scheduler]
            C[Task Scheduler]
        end
    
        subgraph "Cluster Manager"
            D[YARN / Mesos / K8s]
        end
    
        subgraph "Worker Node 1"
            E1[Executor 1]
            E2[Task]
            E3[Cache]
        end
    
        subgraph "Worker Node 2"
            F1[Executor 2]
            F2[Task]
            F3[Cache]
        end
    
        subgraph "Worker Node N"
            G1[Executor N]
            G2[Task]
            G3[Cache]
        end
    
        A --> B
        B --> C
        C --> D
        D --> E1
        D --> F1
        D --> G1
    
        style A fill:#e3f2fd
        style D fill:#fff3e0
        style E1 fill:#e8f5e9
        style F1 fill:#e8f5e9
        style G1 fill:#e8f5e9
    ```

### Lazy Evaluation（遅延評価）

Sparkは**Transformation** （変換）と**Action** （アクション）を区別します。

タイプ | 説明 | 例  
---|---|---  
**Transformation** | 新しいRDD/DataFrameを返す  
遅延評価（計算は実行されない） | `map()`, `filter()`, `groupBy()`  
**Action** | 結果を返す/保存する  
即時評価（実際の計算を実行） | `count()`, `collect()`, `save()`  
  
### DAG実行モデル
    
    
    ```mermaid
    graph LR
        A[データ読み込み] --> B[filter]
        B --> C[map]
        C --> D[reduceByKey]
        D --> E[collect]
    
        style A fill:#e3f2fd
        style E fill:#ffebee
        style B fill:#f3e5f5
        style C fill:#f3e5f5
        style D fill:#f3e5f5
    
        classDef transformation fill:#f3e5f5
        classDef action fill:#ffebee
    ```

**Transformation** は実行計画（DAG）を構築し、**Action** が呼ばれた時に最適化された計算が実行されます。

### Sparkセッションの初期化
    
    
    from pyspark.sql import SparkSession
    
    # Sparkセッションの作成
    spark = SparkSession.builder \
        .appName("SparkMLExample") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print(f"Spark Version: {spark.version}")
    print(f"Spark Master: {spark.sparkContext.master}")
    print(f"App Name: {spark.sparkContext.appName}")
    
    # Sparkセッションの設定確認
    spark.sparkContext.getConf().getAll()
    

**出力** ：
    
    
    Spark Version: 3.5.0
    Spark Master: local[*]
    App Name: SparkMLExample
    

> **重要** : `local[*]`はローカルモードで全CPUコアを使用します。クラスターモードでは`yarn`や`k8s://`を指定します。

* * *

## 2.2 RDD（Resilient Distributed Datasets）

### RDDとは

**RDD（Resilient Distributed Dataset）** は、Sparkの基本的なデータ抽象化で、分散コレクションの不変オブジェクトです。

#### RDDの3つの特性

  1. **Resilient（耐障害性）** : Lineage（系譜）により、障害時に自動復旧
  2. **Distributed（分散）** : データはクラスター全体に分散
  3. **Dataset（データセット）** : メモリ内の不変コレクション

### RDDの基本操作

#### RDD作成
    
    
    from pyspark import SparkContext
    
    # SparkContextの取得（SparkSessionから）
    sc = spark.sparkContext
    
    # 方法1: Pythonリストから作成
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rdd = sc.parallelize(data, numSlices=4)  # 4パーティションに分割
    
    print(f"パーティション数: {rdd.getNumPartitions()}")
    print(f"最初の5要素: {rdd.take(5)}")
    
    # 方法2: テキストファイルから作成
    # text_rdd = sc.textFile("hdfs://path/to/file.txt")
    
    # 方法3: 複数ファイルから作成
    # multi_rdd = sc.wholeTextFiles("hdfs://path/to/directory/")
    

**出力** ：
    
    
    パーティション数: 4
    最初の5要素: [1, 2, 3, 4, 5]
    

### Transformations（変換）
    
    
    # データ準備
    numbers = sc.parallelize(range(1, 11))
    
    # map: 各要素に関数を適用
    squares = numbers.map(lambda x: x ** 2)
    print(f"二乗: {squares.collect()}")
    
    # filter: 条件に合う要素のみ抽出
    evens = numbers.filter(lambda x: x % 2 == 0)
    print(f"偶数: {evens.collect()}")
    
    # flatMap: 各要素を複数要素に展開
    words = sc.parallelize(["Hello World", "Apache Spark"])
    all_words = words.flatMap(lambda line: line.split(" "))
    print(f"単語: {all_words.collect()}")
    
    # union: 2つのRDDを結合
    rdd1 = sc.parallelize([1, 2, 3])
    rdd2 = sc.parallelize([4, 5, 6])
    combined = rdd1.union(rdd2)
    print(f"結合: {combined.collect()}")
    
    # distinct: 重複を削除
    duplicates = sc.parallelize([1, 2, 2, 3, 3, 3, 4])
    unique = duplicates.distinct()
    print(f"ユニーク: {unique.collect()}")
    

**出力** ：
    
    
    二乗: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    偶数: [2, 4, 6, 8, 10]
    単語: ['Hello', 'World', 'Apache', 'Spark']
    結合: [1, 2, 3, 4, 5, 6]
    ユニーク: [1, 2, 3, 4]
    

### Key-Value RDD操作
    
    
    # ペアRDDの作成
    pairs = sc.parallelize([("apple", 3), ("banana", 2), ("apple", 5), ("orange", 1)])
    
    # reduceByKey: キーごとに値を集約
    total_by_key = pairs.reduceByKey(lambda a, b: a + b)
    print(f"キー別合計: {total_by_key.collect()}")
    
    # groupByKey: キーごとに値をグループ化
    grouped = pairs.groupByKey()
    print(f"グループ化: {[(k, list(v)) for k, v in grouped.collect()]}")
    
    # mapValues: 値のみに関数を適用
    doubled_values = pairs.mapValues(lambda x: x * 2)
    print(f"値を2倍: {doubled_values.collect()}")
    
    # sortByKey: キーでソート
    sorted_pairs = pairs.sortByKey()
    print(f"ソート: {sorted_pairs.collect()}")
    
    # join: 2つのペアRDDを結合
    prices = sc.parallelize([("apple", 100), ("banana", 80), ("orange", 60)])
    joined = pairs.join(prices)
    print(f"結合: {joined.collect()}")
    

**出力** ：
    
    
    キー別合計: [('apple', 8), ('banana', 2), ('orange', 1)]
    グループ化: [('apple', [3, 5]), ('banana', [2]), ('orange', [1])]
    値を2倍: [('apple', 6), ('banana', 4), ('apple', 10), ('orange', 2)]
    ソート: [('apple', 3), ('apple', 5), ('banana', 2), ('orange', 1)]
    結合: [('apple', (3, 100)), ('apple', (5, 100)), ('banana', (2, 80)), ('orange', (1, 60))]
    

### Actions（アクション）
    
    
    numbers = sc.parallelize(range(1, 11))
    
    # count: 要素数をカウント
    print(f"要素数: {numbers.count()}")
    
    # collect: すべての要素を取得（注意: メモリに収まるサイズのみ）
    print(f"全要素: {numbers.collect()}")
    
    # take: 最初のn要素を取得
    print(f"最初の3要素: {numbers.take(3)}")
    
    # first: 最初の要素を取得
    print(f"最初の要素: {numbers.first()}")
    
    # reduce: 全要素を集約
    sum_all = numbers.reduce(lambda a, b: a + b)
    print(f"合計: {sum_all}")
    
    # foreach: 各要素に副作用のある処理を実行
    numbers.foreach(lambda x: print(f"処理中: {x}"))
    
    # saveAsTextFile: ファイルに保存
    # numbers.saveAsTextFile("output/numbers")
    

**出力** ：
    
    
    要素数: 10
    全要素: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    最初の3要素: [1, 2, 3]
    最初の要素: 1
    合計: 55
    

### Lineage（系譜）と耐障害性
    
    
    # RDDの系譜を確認
    numbers = sc.parallelize(range(1, 101))
    squares = numbers.map(lambda x: x ** 2)
    evens = squares.filter(lambda x: x % 2 == 0)
    
    # デバッグ文字列で系譜を表示
    print("RDD系譜:")
    print(evens.toDebugString().decode('utf-8'))
    

**出力** ：
    
    
    RDD系譜:
    (4) PythonRDD[10] at RDD at PythonRDD.scala:53 []
     |  MapPartitionsRDD[9] at mapPartitions at PythonRDD.scala:145 []
     |  MapPartitionsRDD[8] at mapPartitions at PythonRDD.scala:145 []
     |  ParallelCollectionRDD[7] at parallelize at PythonRDD.scala:195 []
    

> **重要** : Sparkは系譜を記録しており、ノード障害時にはこの系譜から自動的にデータを再計算します。

* * *

## 2.3 Spark DataFrame と SQL

### DataFrameとは

**DataFrame** は、名前付き列を持つ分散データセットで、RDDより高速で使いやすいAPIです。

#### DataFrameの利点

  * **Catalyst Optimizer** : クエリ最適化により高速化
  * **Tungsten実行エンジン** : メモリ効率の向上
  * **スキーマ情報** : 型安全性と最適化
  * **SQL互換性** : SQLクエリが使える

### DataFrame作成
    
    
    from pyspark.sql import Row
    import pandas as pd
    
    # 方法1: Pythonリストから作成
    data = [
        ("Alice", 25, "Engineer"),
        ("Bob", 30, "Data Scientist"),
        ("Charlie", 35, "Manager"),
        ("Diana", 28, "Analyst")
    ]
    columns = ["name", "age", "job"]
    df = spark.createDataFrame(data, columns)
    
    # データ確認
    df.show()
    df.printSchema()
    
    # 方法2: Rowオブジェクトから作成
    rows = [
        Row(name="Eve", age=32, job="Developer"),
        Row(name="Frank", age=29, job="Designer")
    ]
    df2 = spark.createDataFrame(rows)
    
    # 方法3: Pandas DataFrameから作成
    pandas_df = pd.DataFrame({
        'name': ['Grace', 'Henry'],
        'age': [27, 31],
        'job': ['Researcher', 'Architect']
    })
    df3 = spark.createDataFrame(pandas_df)
    
    # 方法4: CSVファイルから読み込み
    # df_csv = spark.read.csv("data.csv", header=True, inferSchema=True)
    

**出力** ：
    
    
    +-------+---+---------------+
    |   name|age|            job|
    +-------+---+---------------+
    |  Alice| 25|       Engineer|
    |    Bob| 30|Data Scientist|
    |Charlie| 35|        Manager|
    |  Diana| 28|        Analyst|
    +-------+---+---------------+
    
    root
     |-- name: string (nullable = true)
     |-- age: long (nullable = true)
     |-- job: string (nullable = true)
    

### DataFrame操作

#### 選択とフィルタリング
    
    
    # 列の選択
    df.select("name", "age").show()
    
    # 条件によるフィルタリング
    df.filter(df.age > 28).show()
    
    # where（filterの別名）
    df.where(df.job == "Engineer").show()
    
    # 複数条件
    df.filter((df.age > 25) & (df.age < 32)).show()
    
    # 新しい列の追加
    from pyspark.sql.functions import col, lit
    
    df_with_salary = df.withColumn("salary", col("age") * 1000)
    df_with_salary.show()
    
    # 列名の変更
    df_renamed = df.withColumnRenamed("job", "position")
    df_renamed.show()
    
    # 列の削除
    df_dropped = df.drop("job")
    df_dropped.show()
    

### 集約とグループ化
    
    
    from pyspark.sql.functions import avg, count, max, min, sum
    
    # データ準備
    sales_data = [
        ("Alice", "2024-01", 100),
        ("Alice", "2024-02", 150),
        ("Bob", "2024-01", 200),
        ("Bob", "2024-02", 180),
        ("Charlie", "2024-01", 120),
        ("Charlie", "2024-02", 140)
    ]
    sales_df = spark.createDataFrame(sales_data, ["name", "month", "sales"])
    
    # グループ化と集約
    sales_summary = sales_df.groupBy("name").agg(
        sum("sales").alias("total_sales"),
        avg("sales").alias("avg_sales"),
        count("sales").alias("num_months")
    )
    sales_summary.show()
    
    # 複数列でグループ化
    monthly_stats = sales_df.groupBy("name", "month").agg(
        max("sales").alias("max_sales"),
        min("sales").alias("min_sales")
    )
    monthly_stats.show()
    
    # ピボットテーブル
    pivot_df = sales_df.groupBy("name").pivot("month").sum("sales")
    pivot_df.show()
    

**出力** ：
    
    
    +-------+-----------+---------+----------+
    |   name|total_sales|avg_sales|num_months|
    +-------+-----------+---------+----------+
    |  Alice|        250|    125.0|         2|
    |    Bob|        380|    190.0|         2|
    |Charlie|        260|    130.0|         2|
    +-------+-----------+---------+----------+
    

### Spark SQLの利用
    
    
    # DataFrameをテンポラリビューとして登録
    df.createOrReplaceTempView("employees")
    
    # SQLクエリの実行
    sql_result = spark.sql("""
        SELECT
            job,
            COUNT(*) as num_employees,
            AVG(age) as avg_age,
            MAX(age) as max_age,
            MIN(age) as min_age
        FROM employees
        GROUP BY job
        ORDER BY avg_age DESC
    """)
    
    sql_result.show()
    
    # 複雑なSQLクエリ
    advanced_query = spark.sql("""
        SELECT
            name,
            age,
            job,
            CASE
                WHEN age < 28 THEN 'Junior'
                WHEN age >= 28 AND age < 32 THEN 'Mid-level'
                ELSE 'Senior'
            END as level
        FROM employees
        WHERE age > 25
        ORDER BY age
    """)
    
    advanced_query.show()
    

### 結合操作
    
    
    # データ準備
    employees = spark.createDataFrame([
        (1, "Alice", "Engineering"),
        (2, "Bob", "Data Science"),
        (3, "Charlie", "Management")
    ], ["id", "name", "department"])
    
    salaries = spark.createDataFrame([
        (1, 80000),
        (2, 95000),
        (4, 70000)  # id=4は社員テーブルに存在しない
    ], ["id", "salary"])
    
    # Inner Join（内部結合）
    inner_join = employees.join(salaries, "id", "inner")
    print("Inner Join:")
    inner_join.show()
    
    # Left Outer Join（左外部結合）
    left_join = employees.join(salaries, "id", "left")
    print("Left Outer Join:")
    left_join.show()
    
    # Right Outer Join（右外部結合）
    right_join = employees.join(salaries, "id", "right")
    print("Right Outer Join:")
    right_join.show()
    
    # Full Outer Join（完全外部結合）
    full_join = employees.join(salaries, "id", "outer")
    print("Full Outer Join:")
    full_join.show()
    

**出力（Inner Join）** ：
    
    
    +---+-----+-------------+------+
    | id| name|   department|salary|
    +---+-----+-------------+------+
    |  1|Alice|  Engineering| 80000|
    |  2|  Bob| Data Science| 95000|
    +---+-----+-------------+------+
    

### Catalyst Optimizerの効果
    
    
    # クエリの実行計画を確認
    df_filtered = df.filter(df.age > 25).select("name", "age")
    
    # 物理実行計画
    print("Physical Plan:")
    df_filtered.explain(mode="formatted")
    
    # 最適化前の論理プラン
    print("\nLogical Plan:")
    df_filtered.explain(mode="extended")
    

> **重要** : Catalystは述語プッシュダウン、カラムプルーニング、定数畳み込みなどの最適化を自動的に適用します。

* * *

## 2.4 Spark MLlib（機械学習）

### MLlibとは

**Spark MLlib** は、Sparkの分散機械学習ライブラリで、大規模データに対する学習を効率的に実行します。

#### MLlibの主要機能

  * **分類** : ロジスティック回帰、決定木、ランダムフォレスト、GBT
  * **回帰** : 線形回帰、回帰木、一般化線形モデル
  * **クラスタリング** : K-Means、GMM、LDA
  * **協調フィルタリング** : ALS（交互最小二乗法）
  * **次元削減** : PCA、SVD
  * **特徴量変換** : VectorAssembler、StringIndexer、OneHotEncoder

### ML Pipelineの基本
    
    
    ```mermaid
    graph LR
        A[生データ] --> B[StringIndexer]
        B --> C[VectorAssembler]
        C --> D[StandardScaler]
        D --> E[分類器]
        E --> F[予測結果]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#f3e5f5
        style F fill:#e8f5e9
    ```

### 分類タスクの実装
    
    
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    
    # サンプルデータ生成
    from pyspark.sql.functions import rand, when
    
    # Irisデータセットのような構造のデータを生成
    data = spark.range(0, 1000).select(
        (rand() * 3 + 4).alias("sepal_length"),
        (rand() * 2 + 2).alias("sepal_width"),
        (rand() * 3 + 1).alias("petal_length"),
        (rand() * 2 + 0.1).alias("petal_width")
    )
    
    # ターゲット変数の作成
    data = data.withColumn(
        "species",
        when((data.petal_length < 2), "setosa")
        .when((data.petal_length >= 2) & (data.petal_length < 4), "versicolor")
        .otherwise("virginica")
    )
    
    # データの確認
    data.show(10)
    data.groupBy("species").count().show()
    
    # 訓練・テストデータ分割
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"訓練データ: {train_data.count()}行")
    print(f"テストデータ: {test_data.count()}行")
    

### 特徴量変換パイプライン
    
    
    # ステージ1: カテゴリ変数をインデックスに変換
    label_indexer = StringIndexer(
        inputCol="species",
        outputCol="label"
    )
    
    # ステージ2: 特徴量をベクトルに結合
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    vector_assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    
    # ステージ3: ロジスティック回帰モデル
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01
    )
    
    # パイプラインの構築
    pipeline = Pipeline(stages=[label_indexer, vector_assembler, lr])
    
    # モデルの訓練
    print("モデルの訓練を開始...")
    model = pipeline.fit(train_data)
    print("訓練完了")
    
    # 予測
    predictions = model.transform(test_data)
    
    # 予測結果の確認
    predictions.select("species", "label", "features", "prediction", "probability").show(10, truncate=False)
    

### モデルの評価
    
    
    # 多クラス分類の評価
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )
    
    # 精度（Accuracy）
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    print(f"精度: {accuracy:.4f}")
    
    # F1スコア
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    print(f"F1スコア: {f1:.4f}")
    
    # 加重精度
    weighted_precision = multi_evaluator.evaluate(
        predictions,
        {multi_evaluator.metricName: "weightedPrecision"}
    )
    print(f"加重精度: {weighted_precision:.4f}")
    
    # 加重再現率
    weighted_recall = multi_evaluator.evaluate(
        predictions,
        {multi_evaluator.metricName: "weightedRecall"}
    )
    print(f"加重再現率: {weighted_recall:.4f}")
    
    # 混同行列の計算
    from pyspark.ml.evaluation import MulticlassMetrics
    prediction_and_labels = predictions.select("prediction", "label").rdd
    metrics = MulticlassMetrics(prediction_and_labels)
    
    print("\n混同行列:")
    print(metrics.confusionMatrix().toArray())
    

### 回帰タスクの実装
    
    
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # 回帰用サンプルデータ生成
    regression_data = spark.range(0, 1000).select(
        (rand() * 100).alias("feature1"),
        (rand() * 50).alias("feature2"),
        (rand() * 30).alias("feature3")
    )
    
    # ターゲット変数（線形関係 + ノイズ）
    from pyspark.sql.functions import col
    regression_data = regression_data.withColumn(
        "target",
        col("feature1") * 2 + col("feature2") * 1.5 - col("feature3") * 0.5 + (rand() * 10 - 5)
    )
    
    # 訓練・テスト分割
    train_reg, test_reg = regression_data.randomSplit([0.8, 0.2], seed=42)
    
    # 特徴量ベクトルの作成
    feature_cols = ["feature1", "feature2", "feature3"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # 線形回帰モデル
    lr_regressor = LinearRegression(
        featuresCol="features",
        labelCol="target",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.5  # L1/L2正則化の混合比
    )
    
    # パイプライン構築
    regression_pipeline = Pipeline(stages=[assembler, lr_regressor])
    
    # 訓練
    regression_model = regression_pipeline.fit(train_reg)
    
    # 予測
    regression_predictions = regression_model.transform(test_reg)
    
    # 評価
    reg_evaluator = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction"
    )
    
    rmse = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "rmse"})
    mae = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "mae"})
    r2 = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "r2"})
    
    print(f"\n=== 回帰モデルの評価 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # モデルの係数
    lr_model = regression_model.stages[-1]
    print(f"\n係数: {lr_model.coefficients}")
    print(f"切片: {lr_model.intercept:.4f}")
    

### ランダムフォレストによる分類
    
    
    from pyspark.ml.classification import RandomForestClassifier
    
    # ランダムフォレストモデル
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    # パイプライン（特徴量変換 + RF）
    rf_pipeline = Pipeline(stages=[label_indexer, vector_assembler, rf])
    
    # 訓練
    print("ランダムフォレストの訓練開始...")
    rf_model = rf_pipeline.fit(train_data)
    print("訓練完了")
    
    # 予測
    rf_predictions = rf_model.transform(test_data)
    
    # 評価
    rf_accuracy = multi_evaluator.evaluate(
        rf_predictions,
        {multi_evaluator.metricName: "accuracy"}
    )
    rf_f1 = multi_evaluator.evaluate(
        rf_predictions,
        {multi_evaluator.metricName: "f1"}
    )
    
    print(f"\n=== ランダムフォレストの評価 ===")
    print(f"精度: {rf_accuracy:.4f}")
    print(f"F1スコア: {rf_f1:.4f}")
    
    # 特徴量の重要度
    rf_classifier = rf_model.stages[-1]
    feature_importances = rf_classifier.featureImportances
    
    print("\n特徴量の重要度:")
    for idx, importance in enumerate(feature_importances):
        print(f"{feature_columns[idx]}: {importance:.4f}")
    

### クロスバリデーションとハイパーパラメータ調整
    
    
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    # パラメータグリッドの構築
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.maxIter, [50, 100, 150]) \
        .build()
    
    # クロスバリデーションの設定
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=multi_evaluator,
        numFolds=3,
        seed=42
    )
    
    # クロスバリデーション実行
    print("クロスバリデーション開始...")
    cv_model = cv.fit(train_data)
    print("完了")
    
    # 最良モデルで予測
    cv_predictions = cv_model.transform(test_data)
    
    # 最良パラメータの確認
    best_model = cv_model.bestModel
    print("\n最良パラメータ:")
    print(best_model.stages[-1].extractParamMap())
    
    # 評価
    cv_accuracy = multi_evaluator.evaluate(
        cv_predictions,
        {multi_evaluator.metricName: "accuracy"}
    )
    print(f"\nCV後の精度: {cv_accuracy:.4f}")
    

* * *

## 2.5 パフォーマンス最適化

### パーティショニング戦略

適切なパーティショニングは、Sparkの性能を大きく左右します。

#### パーティション数の決定
    
    
    # デフォルトのパーティション数
    print(f"デフォルトパーティション数: {spark.sparkContext.defaultParallelism}")
    
    # RDDのパーティション数確認
    rdd = sc.parallelize(range(1000))
    print(f"RDDパーティション数: {rdd.getNumPartitions()}")
    
    # DataFrameのパーティション数確認
    df = spark.range(10000)
    print(f"DataFrameパーティション数: {df.rdd.getNumPartitions()}")
    
    # パーティション数の再設定
    rdd_repartitioned = rdd.repartition(8)
    print(f"再パーティション後: {rdd_repartitioned.getNumPartitions()}")
    
    # coalesce: パーティション数を減らす（シャッフルなし）
    rdd_coalesced = rdd.coalesce(4)
    print(f"Coalesce後: {rdd_coalesced.getNumPartitions()}")
    

> **推奨** : パーティション数は（CPUコア数 × 2〜3）が目安です。

#### カスタムパーティショナー
    
    
    # Key-Valueペアでのハッシュパーティショニング
    pairs = sc.parallelize([("A", 1), ("B", 2), ("A", 3), ("C", 4), ("B", 5)])
    
    # ハッシュパーティショニング
    hash_partitioned = pairs.partitionBy(4)
    print(f"ハッシュパーティション数: {hash_partitioned.getNumPartitions()}")
    
    # 各パーティションの内容を確認
    def show_partition_contents(index, iterator):
        yield f"Partition {index}: {list(iterator)}"
    
    partition_contents = hash_partitioned.mapPartitionsWithIndex(show_partition_contents)
    for content in partition_contents.collect():
        print(content)
    

### キャッシングと永続化

#### メモリキャッシング
    
    
    # DataFrameのキャッシュ
    df_large = spark.range(0, 10000000)
    
    # キャッシュ（デフォルト: メモリのみ）
    df_large.cache()
    
    # 初回アクションでキャッシュが作成される
    count1 = df_large.count()
    print(f"初回カウント（キャッシュ作成）: {count1}")
    
    # 2回目以降はキャッシュから取得（高速）
    count2 = df_large.count()
    print(f"2回目カウント（キャッシュ使用）: {count2}")
    
    # キャッシュ解放
    df_large.unpersist()
    

#### 永続化レベルの選択
    
    
    from pyspark import StorageLevel
    
    # RDDの永続化レベル
    rdd = sc.parallelize(range(1000000))
    
    # メモリとディスクの両方を使用
    rdd.persist(StorageLevel.MEMORY_AND_DISK)
    
    # シリアライズしてメモリに保存（メモリ効率向上）
    rdd.persist(StorageLevel.MEMORY_ONLY_SER)
    
    # レプリケーション（耐障害性向上）
    rdd.persist(StorageLevel.MEMORY_AND_DISK_2)
    
    print(f"ストレージレベル: {rdd.getStorageLevel()}")
    

ストレージレベル | 説明 | 用途  
---|---|---  
`MEMORY_ONLY` | メモリのみ（デフォルト） | 十分なメモリがある場合  
`MEMORY_AND_DISK` | メモリ → ディスクにスピル | 大規模データ  
`MEMORY_ONLY_SER` | シリアライズしてメモリ保存 | メモリ効率重視  
`DISK_ONLY` | ディスクのみ | メモリ不足時  
`OFF_HEAP` | オフヒープメモリ | GC回避  
  
### ブロードキャスト変数
    
    
    # 小さなデータセットを全ノードに配布
    lookup_table = {"A": 100, "B": 200, "C": 300, "D": 400}
    
    # ブロードキャスト
    broadcast_lookup = sc.broadcast(lookup_table)
    
    # RDDでブロードキャスト変数を使用
    data = sc.parallelize([("A", 1), ("B", 2), ("C", 3), ("A", 4)])
    
    def enrich_data(pair):
        key, value = pair
        # ブロードキャスト変数を参照
        multiplier = broadcast_lookup.value.get(key, 1)
        return (key, value * multiplier)
    
    enriched = data.map(enrich_data)
    print(enriched.collect())
    
    # ブロードキャスト変数の解放
    broadcast_lookup.unpersist()
    

> **重要** : ブロードキャスト変数は結合操作のパフォーマンスを大幅に向上させます（特に小さいテーブルとの結合）。

### チューニングパラメータ

#### Sparkセッションの設定
    
    
    # パフォーマンス最適化の設定
    spark_optimized = SparkSession.builder \
        .appName("OptimizedSparkApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.default.parallelism", "100") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    # 設定の確認
    for conf in spark_optimized.sparkContext.getConf().getAll():
        print(f"{conf[0]}: {conf[1]}")
    

#### 主要チューニングパラメータ

パラメータ | 説明 | 推奨値  
---|---|---  
`spark.executor.memory` | Executorのメモリサイズ | 利用可能メモリの70%  
`spark.executor.cores` | Executor当たりのコア数 | 4-6コア  
`spark.default.parallelism` | デフォルト並列度 | コア数 × 2-3  
`spark.sql.shuffle.partitions` | シャッフル時のパーティション数 | 100-200（データサイズ依存）  
`spark.sql.adaptive.enabled` | 適応的クエリ実行 | `true`  
`spark.serializer` | シリアライザ | `KryoSerializer`  
  
### 実行計画の最適化
    
    
    # DataFrameの最適化例
    large_df = spark.range(0, 10000000)
    small_df = spark.range(0, 100)
    
    # 最適化前: 大きいテーブルでフィルタ → 結合
    result_unoptimized = large_df.filter(large_df.id % 2 == 0).join(small_df, "id")
    
    # 最適化後: 結合 → フィルタ（述語プッシュダウン）
    result_optimized = large_df.join(small_df, "id").filter(large_df.id % 2 == 0)
    
    # 実行計画の比較
    print("最適化前:")
    result_unoptimized.explain()
    
    print("\n最適化後:")
    result_optimized.explain()
    
    # Catalystが自動的に最適化するため、実際には両方とも同じ実行計画になる
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **Sparkアーキテクチャ**

     * Driver-Executor モデルによる分散処理
     * Lazy EvaluationとDAG実行
     * クラスターマネージャー（YARN、Mesos、K8s）
     * TransformationとActionの区別
  2. **RDD（Resilient Distributed Datasets）**

     * 不変・分散・耐障害性のあるコレクション
     * Lineageによる自動復旧
     * map、filter、reduceByKeyなどの操作
     * Key-Value ペアの処理
  3. **Spark DataFrame と SQL**

     * Catalyst Optimizerによる高速化
     * スキーマ情報による型安全性
     * SQLクエリとDataFrame APIの統合
     * 結合、集約、グループ化の効率的な処理
  4. **Spark MLlib**

     * 分散機械学習のパイプライン
     * 特徴量変換と前処理
     * 分類、回帰、クラスタリング
     * クロスバリデーションとハイパーパラメータ調整
  5. **パフォーマンス最適化**

     * 適切なパーティショニング戦略
     * キャッシングと永続化レベル
     * ブロードキャスト変数による結合最適化
     * チューニングパラメータの設定

### Spark活用のベストプラクティス

項目 | 推奨事項  
---|---  
**API選択** | DataFrame/Dataset > RDD（最適化の恩恵）  
**パーティション** | 適切な数（コア数 × 2-3）、均等な分散  
**キャッシング** | 再利用する中間結果のみキャッシュ  
**シャッフル削減** | 不要なgroupByKey回避、reduceByKey使用  
**ブロードキャスト** | 小さいテーブルとの結合に活用  
**メモリ管理** | Executor メモリの適切な設定  
  
### 次の章へ

第3章では、**分散深層学習フレームワーク** を学びます：

  * Horovod による分散学習
  * TensorFlow と PyTorch の分散戦略
  * Ray による超並列処理
  * MLflow による実験管理
  * 分散ハイパーパラメータ最適化

* * *

## 演習問題

### 問題1（難易度：easy）

SparkにおけるTransformationとActionの違いを説明し、それぞれの例を3つずつ挙げてください。

解答例

**解答** ：

**Transformation（変換）** ：

  * 定義: 新しいRDD/DataFrameを返す操作で、遅延評価される
  * 特徴: 実際の計算は実行されず、実行計画（DAG）が構築される
  * 例: 
    1. `map()` \- 各要素に関数を適用
    2. `filter()` \- 条件に合う要素のみ抽出
    3. `groupBy()` \- キーでグループ化

**Action（アクション）** ：

  * 定義: 結果を返す/保存する操作で、即時評価される
  * 特徴: 実際の計算が実行され、Driverまたはストレージにデータが返される
  * 例: 
    1. `count()` \- 要素数をカウント
    2. `collect()` \- すべての要素を取得
    3. `saveAsTextFile()` \- ファイルに保存

**違いの重要性** ：

TransformationはDAGを構築するだけなので高速です。Actionが呼ばれた時に、Sparkは最適化された実行計画で計算を実行します。

### 問題2（難易度：medium）

以下のコードを実行すると、何が問題になる可能性がありますか？また、どのように修正すべきですか？
    
    
    rdd = sc.parallelize(range(1, 1000000))
    result = rdd.map(lambda x: x ** 2).collect()
    print(result)
    

解答例

**問題点** ：

  1. **メモリ不足** : `collect()`は全データをDriverメモリに集めるため、100万要素だとメモリ不足になる可能性
  2. **パフォーマンス低下** : 分散処理の利点が失われる
  3. **ネットワーク負荷** : 大量のデータをExecutorからDriverに転送

**修正方法** ：
    
    
    # 方法1: 必要な要素のみ取得
    rdd = sc.parallelize(range(1, 1000000))
    result = rdd.map(lambda x: x ** 2).take(10)  # 最初の10要素のみ
    print(result)
    
    # 方法2: ファイルに保存
    rdd.map(lambda x: x ** 2).saveAsTextFile("output/squares")
    
    # 方法3: 集約操作を使用
    total = rdd.map(lambda x: x ** 2).sum()
    print(f"合計: {total}")
    
    # 方法4: サンプリング
    sample = rdd.map(lambda x: x ** 2).sample(False, 0.01).collect()
    print(f"サンプル: {sample[:10]}")
    

**ベストプラクティス** ：

  * `collect()`は小さなデータセット（数千行以下）のみに使用
  * 大規模データは`take(n)`、`sample()`、`saveAsTextFile()`を使用

### 問題3（難易度：medium）

Spark DataFrameで以下のSQLクエリと同等の処理をDataFrame APIで実装してください。
    
    
    SELECT
        department,
        AVG(salary) as avg_salary,
        MAX(salary) as max_salary,
        COUNT(*) as num_employees
    FROM employees
    WHERE age > 25
    GROUP BY department
    HAVING COUNT(*) > 5
    ORDER BY avg_salary DESC
    

解答例
    
    
    from pyspark.sql.functions import avg, max, count, col
    
    # DataFrame API版
    result = employees \
        .filter(col("age") > 25) \
        .groupBy("department") \
        .agg(
            avg("salary").alias("avg_salary"),
            max("salary").alias("max_salary"),
            count("*").alias("num_employees")
        ) \
        .filter(col("num_employees") > 5) \
        .orderBy(col("avg_salary").desc())
    
    result.show()
    
    # 別の書き方（メソッドチェーン）
    result_alt = (employees
        .where("age > 25")
        .groupBy("department")
        .agg(
            {"salary": "avg", "salary": "max", "*": "count"}
        )
        .withColumnRenamed("avg(salary)", "avg_salary")
        .withColumnRenamed("max(salary)", "max_salary")
        .withColumnRenamed("count(1)", "num_employees")
        .filter("num_employees > 5")
        .sort(col("avg_salary").desc())
    )
    

**説明** ：

  * `filter()` / `where()`: WHERE句
  * `groupBy()`: GROUP BY句
  * `agg()`: 集約関数（AVG、MAX、COUNT）
  * `filter()`（2回目）: HAVING句
  * `orderBy()` / `sort()`: ORDER BY句

### 問題4（難易度：hard）

大規模データセット（1億行）でKey-Valueペアの結合を効率的に実行する方法を、以下の3つのシナリオで説明してください：

  1. 両方のデータセットが大きい場合
  2. 片方のデータセットが小さい（メモリに収まる）場合
  3. データが既にソート済みの場合

解答例

**解答** ：

#### シナリオ1: 両方のデータセットが大きい場合
    
    
    # 標準的なjoin（ソートマージ結合またはハッシュ結合）
    large_df1 = spark.read.parquet("large_dataset1.parquet")
    large_df2 = spark.read.parquet("large_dataset2.parquet")
    
    # パーティション数を最適化
    large_df1 = large_df1.repartition(200, "join_key")
    large_df2 = large_df2.repartition(200, "join_key")
    
    # 結合
    result = large_df1.join(large_df2, "join_key", "inner")
    
    # キャッシュ（再利用する場合）
    result.cache()
    result.count()  # キャッシュを実体化
    

**最適化ポイント** ：

  * 適切なパーティション数（データサイズに応じて調整）
  * 結合キーで事前にパーティション化
  * 適応的クエリ実行（AQE）を有効化

#### シナリオ2: 片方のデータセットが小さい場合
    
    
    from pyspark.sql.functions import broadcast
    
    large_df = spark.read.parquet("large_dataset.parquet")
    small_df = spark.read.parquet("small_dataset.parquet")
    
    # ブロードキャスト結合（小さいテーブルを全ノードに配布）
    result = large_df.join(broadcast(small_df), "join_key", "inner")
    
    # または、自動ブロードキャスト閾値を設定
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10485760)  # 10MB
    

**最適化ポイント** ：

  * 小さいテーブル（<10MB）をブロードキャスト
  * シャッフルが不要になり、大幅な高速化
  * メモリ使用量に注意（全Executorに配布される）

#### シナリオ3: データが既にソート済みの場合
    
    
    # データが結合キーでソート済み・パーティション化済みの場合
    sorted_df1 = spark.read.parquet("sorted_dataset1.parquet")
    sorted_df2 = spark.read.parquet("sorted_dataset2.parquet")
    
    # ソートマージ結合を明示的に使用
    result = sorted_df1.join(
        sorted_df2,
        sorted_df1["join_key"] == sorted_df2["join_key"],
        "inner"
    )
    
    # ヒントを使ってソートマージ結合を強制
    from pyspark.sql.functions import expr
    result = sorted_df1.hint("merge").join(sorted_df2, "join_key")
    

**最適化ポイント** ：

  * 既にソート済みならシャッフルが削減される
  * バケッティング（Bucketing）を使って事前にパーティション化
  * Parquet形式で保存時にソート順を維持

#### パフォーマンス比較

シナリオ | 結合タイプ | シャッフル | 速度  
---|---|---|---  
両方大きい | ソートマージ/ハッシュ | あり | 中  
片方小さい | ブロードキャスト | なし | 速い  
ソート済み | ソートマージ | 部分的 | 速い  
  
### 問題5（難易度：hard）

Spark MLlibを使用して、テキスト分類タスク（スパム検出）の完全なパイプラインを構築してください。以下を含めてください：

  * テキストの前処理（トークン化、ストップワード除去）
  * TF-IDF特徴量の作成
  * ロジスティック回帰モデルの訓練
  * クロスバリデーションによる評価

解答例
    
    
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    # サンプルデータ作成
    data = spark.createDataFrame([
        (0, "Free money now click here"),
        (0, "Congratulations you won a prize"),
        (1, "Meeting scheduled for tomorrow"),
        (1, "Please review the attached document"),
        (0, "Claim your free gift today"),
        (1, "Project update for next week"),
        (0, "Urgent account verification required"),
        (1, "Thanks for your help yesterday"),
        (0, "You have been selected winner"),
        (1, "Let's discuss the proposal")
    ] * 100, ["label", "text"])  # データを増やす
    
    print(f"データ数: {data.count()}")
    data.show(5)
    
    # 訓練・テスト分割
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    # ステージ1: トークン化
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # ステージ2: ストップワード除去
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # ステージ3: ハッシング TF
    hashingTF = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=1000
    )
    
    # ステージ4: IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # ステージ5: ロジスティック回帰
    lr = LogisticRegression(maxIter=100, regParam=0.01)
    
    # パイプライン構築
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    
    # パラメータグリッド
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [500, 1000, 2000]) \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.maxIter, [50, 100]) \
        .build()
    
    # クロスバリデーション
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=BinaryClassificationEvaluator(),
        numFolds=3,
        seed=42
    )
    
    # 訓練
    print("\nクロスバリデーション開始...")
    cv_model = cv.fit(train)
    print("訓練完了")
    
    # 予測
    predictions = cv_model.transform(test)
    
    # 予測結果の確認
    predictions.select("text", "label", "prediction", "probability").show(10, truncate=False)
    
    # 評価
    binary_evaluator = BinaryClassificationEvaluator()
    multi_evaluator = MulticlassClassificationEvaluator()
    
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    print("\n=== モデル評価 ===")
    print(f"AUC: {auc:.4f}")
    print(f"精度: {accuracy:.4f}")
    print(f"F1スコア: {f1:.4f}")
    
    # 最良パラメータ
    best_model = cv_model.bestModel
    print("\n最良パラメータ:")
    print(f"numFeatures: {best_model.stages[2].getNumFeatures()}")
    print(f"regParam: {best_model.stages[-1].getRegParam()}")
    print(f"maxIter: {best_model.stages[-1].getMaxIter()}")
    
    # 新しいテキストで予測
    new_data = spark.createDataFrame([
        (0, "Free lottery winner claim now"),
        (1, "Project deadline next Monday")
    ], ["id", "text"])
    
    new_predictions = cv_model.transform(new_data)
    new_predictions.select("text", "prediction", "probability").show(truncate=False)
    

**出力例** ：
    
    
    === モデル評価 ===
    AUC: 0.9850
    精度: 0.9500
    F1スコア: 0.9495
    

**拡張アイデア** ：

  * Word2Vec やGloVe埋め込みを使用
  * N-gram特徴量を追加
  * ランダムフォレストやGBTを試す
  * カスタム特徴量（文の長さ、大文字の割合など）を追加

* * *

## 参考文献

  1. Zaharia, M., et al. (2016). _Apache Spark: A Unified Engine for Big Data Processing_. Communications of the ACM, 59(11), 56-65.
  2. Karau, H., Konwinski, A., Wendell, P., & Zaharia, M. (2015). _Learning Spark: Lightning-Fast Big Data Analysis_. O'Reilly Media.
  3. Chambers, B., & Zaharia, M. (2018). _Spark: The Definitive Guide_. O'Reilly Media.
  4. Meng, X., et al. (2016). _MLlib: Machine Learning in Apache Spark_. Journal of Machine Learning Research, 17(1), 1235-1241.
  5. Apache Spark Documentation. (2024). _Spark SQL, DataFrames and Datasets Guide_. URL: https://spark.apache.org/docs/latest/sql-programming-guide.html
  6. Databricks. (2024). _Apache Spark Performance Tuning Guide_. URL: https://www.databricks.com/blog/performance-tuning
