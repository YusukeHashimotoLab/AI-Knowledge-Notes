---
title: "Chapter 2: Distributed Machine Learning with Apache Spark"
chapter_title: "Chapter 2: Distributed Machine Learning with Apache Spark"
subtitle: Practical Foundation for Big Data ML - Accelerating Large-Scale Data Processing with Spark
reading_time: 35-40 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Distributed Machine Learning with Apache Spark. You will learn Spark architecture, Perform efficient data manipulation with Spark SQL, and distributed machine learning with Spark MLlib.

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Understand Spark architecture and distributed processing mechanisms
  * ✅ Use and differentiate between RDD, DataFrame, and Dataset APIs
  * ✅ Perform efficient data manipulation with Spark SQL
  * ✅ Implement distributed machine learning with Spark MLlib
  * ✅ Apply performance optimization techniques
  * ✅ Execute large-scale ML processing on real data

* * *

## 2.1 Spark Architecture

### What is Apache Spark?

**Apache Spark** is a fast distributed processing framework for large-scale data. It achieves speeds over 100 times faster than MapReduce and supports machine learning, stream processing, and graph processing.

> "The successor to MapReduce" - In-memory processing dramatically accelerates iterative operations.

### Key Spark Components

Component | Purpose | Features  
---|---|---  
**Spark Core** | Basic processing engine | RDD, task scheduling  
**Spark SQL** | Structured data processing | DataFrame, SQL queries  
**Spark MLlib** | Machine learning | Distributed ML, pipelines  
**Spark Streaming** | Stream processing | Real-time data processing  
**GraphX** | Graph processing | Network analysis  
  
### Driver and Executor Relationship
    
    
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

### Lazy Evaluation

Spark distinguishes between **Transformations** and **Actions**.

Type | Description | Examples  
---|---|---  
**Transformation** | Returns new RDD/DataFrame  
Lazy evaluation (computation not executed) | `map()`, `filter()`, `groupBy()`  
**Action** | Returns result or saves  
Eager evaluation (actual computation executed) | `count()`, `collect()`, `save()`  
  
### DAG Execution Model
    
    
    ```mermaid
    graph LR
        A[Load Data] --> B[filter]
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

**Transformations** build an execution plan (DAG), and when an **Action** is called, optimized computation is executed.

### Initializing a Spark Session
    
    
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SparkMLExample") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print(f"Spark Version: {spark.version}")
    print(f"Spark Master: {spark.sparkContext.master}")
    print(f"App Name: {spark.sparkContext.appName}")
    
    # Check Spark session configuration
    spark.sparkContext.getConf().getAll()
    

**Output** :
    
    
    Spark Version: 3.5.0
    Spark Master: local[*]
    App Name: SparkMLExample
    

> **Important** : `local[*]` runs in local mode using all CPU cores. In cluster mode, specify `yarn` or `k8s://`.

* * *

## 2.2 RDD (Resilient Distributed Datasets)

### What are RDDs?

**RDD (Resilient Distributed Dataset)** is Spark's fundamental data abstraction - an immutable object representing a distributed collection.

#### Three Properties of RDDs

  1. **Resilient** : Automatic recovery from failures through lineage
  2. **Distributed** : Data is distributed across the cluster
  3. **Dataset** : Immutable collection in memory

### Basic RDD Operations

#### Creating RDDs
    
    
    # Requirements:
    # - Python 3.9+
    # - pyspark>=3.4.0
    
    """
    Example: Creating RDDs
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from pyspark import SparkContext
    
    # Get SparkContext (from SparkSession)
    sc = spark.sparkContext
    
    # Method 1: Create from Python list
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rdd = sc.parallelize(data, numSlices=4)  # Split into 4 partitions
    
    print(f"Number of partitions: {rdd.getNumPartitions()}")
    print(f"First 5 elements: {rdd.take(5)}")
    
    # Method 2: Create from text file
    # text_rdd = sc.textFile("hdfs://path/to/file.txt")
    
    # Method 3: Create from multiple files
    # multi_rdd = sc.wholeTextFiles("hdfs://path/to/directory/")
    

**Output** :
    
    
    Number of partitions: 4
    First 5 elements: [1, 2, 3, 4, 5]
    

### Transformations
    
    
    # Prepare data
    numbers = sc.parallelize(range(1, 11))
    
    # map: Apply function to each element
    squares = numbers.map(lambda x: x ** 2)
    print(f"Squares: {squares.collect()}")
    
    # filter: Extract elements matching condition
    evens = numbers.filter(lambda x: x % 2 == 0)
    print(f"Evens: {evens.collect()}")
    
    # flatMap: Expand each element to multiple elements
    words = sc.parallelize(["Hello World", "Apache Spark"])
    all_words = words.flatMap(lambda line: line.split(" "))
    print(f"Words: {all_words.collect()}")
    
    # union: Combine two RDDs
    rdd1 = sc.parallelize([1, 2, 3])
    rdd2 = sc.parallelize([4, 5, 6])
    combined = rdd1.union(rdd2)
    print(f"Combined: {combined.collect()}")
    
    # distinct: Remove duplicates
    duplicates = sc.parallelize([1, 2, 2, 3, 3, 3, 4])
    unique = duplicates.distinct()
    print(f"Unique: {unique.collect()}")
    

**Output** :
    
    
    Squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    Evens: [2, 4, 6, 8, 10]
    Words: ['Hello', 'World', 'Apache', 'Spark']
    Combined: [1, 2, 3, 4, 5, 6]
    Unique: [1, 2, 3, 4]
    

### Key-Value RDD Operations
    
    
    # Create pair RDD
    pairs = sc.parallelize([("apple", 3), ("banana", 2), ("apple", 5), ("orange", 1)])
    
    # reduceByKey: Aggregate values by key
    total_by_key = pairs.reduceByKey(lambda a, b: a + b)
    print(f"Total by key: {total_by_key.collect()}")
    
    # groupByKey: Group values by key
    grouped = pairs.groupByKey()
    print(f"Grouped: {[(k, list(v)) for k, v in grouped.collect()]}")
    
    # mapValues: Apply function to values only
    doubled_values = pairs.mapValues(lambda x: x * 2)
    print(f"Doubled values: {doubled_values.collect()}")
    
    # sortByKey: Sort by key
    sorted_pairs = pairs.sortByKey()
    print(f"Sorted: {sorted_pairs.collect()}")
    
    # join: Join two pair RDDs
    prices = sc.parallelize([("apple", 100), ("banana", 80), ("orange", 60)])
    joined = pairs.join(prices)
    print(f"Joined: {joined.collect()}")
    

**Output** :
    
    
    Total by key: [('apple', 8), ('banana', 2), ('orange', 1)]
    Grouped: [('apple', [3, 5]), ('banana', [2]), ('orange', [1])]
    Doubled values: [('apple', 6), ('banana', 4), ('apple', 10), ('orange', 2)]
    Sorted: [('apple', 3), ('apple', 5), ('banana', 2), ('orange', 1)]
    Joined: [('apple', (3, 100)), ('apple', (5, 100)), ('banana', (2, 80)), ('orange', (1, 60))]
    

### Actions
    
    
    numbers = sc.parallelize(range(1, 11))
    
    # count: Count elements
    print(f"Count: {numbers.count()}")
    
    # collect: Get all elements (caution: only for data that fits in memory)
    print(f"All elements: {numbers.collect()}")
    
    # take: Get first n elements
    print(f"First 3 elements: {numbers.take(3)}")
    
    # first: Get first element
    print(f"First element: {numbers.first()}")
    
    # reduce: Aggregate all elements
    sum_all = numbers.reduce(lambda a, b: a + b)
    print(f"Sum: {sum_all}")
    
    # foreach: Execute side-effect operation on each element
    numbers.foreach(lambda x: print(f"Processing: {x}"))
    
    # saveAsTextFile: Save to file
    # numbers.saveAsTextFile("output/numbers")
    

**Output** :
    
    
    Count: 10
    All elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    First 3 elements: [1, 2, 3]
    First element: 1
    Sum: 55
    

### Lineage and Fault Tolerance
    
    
    # Check RDD lineage
    numbers = sc.parallelize(range(1, 101))
    squares = numbers.map(lambda x: x ** 2)
    evens = squares.filter(lambda x: x % 2 == 0)
    
    # Display lineage with debug string
    print("RDD Lineage:")
    print(evens.toDebugString().decode('utf-8'))
    

**Output** :
    
    
    RDD Lineage:
    (4) PythonRDD[10] at RDD at PythonRDD.scala:53 []
     |  MapPartitionsRDD[9] at mapPartitions at PythonRDD.scala:145 []
     |  MapPartitionsRDD[8] at mapPartitions at PythonRDD.scala:145 []
     |  ParallelCollectionRDD[7] at parallelize at PythonRDD.scala:195 []
    

> **Important** : Spark records lineage, and in case of node failure, it automatically recomputes data from this lineage.

* * *

## 2.3 Spark DataFrames and SQL

### What are DataFrames?

**DataFrames** are distributed datasets with named columns, providing faster and more user-friendly APIs than RDDs.

#### Advantages of DataFrames

  * **Catalyst Optimizer** : Query optimization for faster execution
  * **Tungsten execution engine** : Improved memory efficiency
  * **Schema information** : Type safety and optimization
  * **SQL compatibility** : Can use SQL queries

### Creating DataFrames
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Creating DataFrames
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from pyspark.sql import Row
    import pandas as pd
    
    # Method 1: Create from Python list
    data = [
        ("Alice", 25, "Engineer"),
        ("Bob", 30, "Data Scientist"),
        ("Charlie", 35, "Manager"),
        ("Diana", 28, "Analyst")
    ]
    columns = ["name", "age", "job"]
    df = spark.createDataFrame(data, columns)
    
    # Check data
    df.show()
    df.printSchema()
    
    # Method 2: Create from Row objects
    rows = [
        Row(name="Eve", age=32, job="Developer"),
        Row(name="Frank", age=29, job="Designer")
    ]
    df2 = spark.createDataFrame(rows)
    
    # Method 3: Create from Pandas DataFrame
    pandas_df = pd.DataFrame({
        'name': ['Grace', 'Henry'],
        'age': [27, 31],
        'job': ['Researcher', 'Architect']
    })
    df3 = spark.createDataFrame(pandas_df)
    
    # Method 4: Read from CSV file
    # df_csv = spark.read.csv("data.csv", header=True, inferSchema=True)
    

**Output** :
    
    
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
    

### DataFrame Operations

#### Selection and Filtering
    
    
    # Column selection
    df.select("name", "age").show()
    
    # Filtering by condition
    df.filter(df.age > 28).show()
    
    # where (alias for filter)
    df.where(df.job == "Engineer").show()
    
    # Multiple conditions
    df.filter((df.age > 25) & (df.age < 32)).show()
    
    # Add new column
    from pyspark.sql.functions import col, lit
    
    df_with_salary = df.withColumn("salary", col("age") * 1000)
    df_with_salary.show()
    
    # Rename column
    df_renamed = df.withColumnRenamed("job", "position")
    df_renamed.show()
    
    # Drop column
    df_dropped = df.drop("job")
    df_dropped.show()
    

### Aggregation and Grouping
    
    
    from pyspark.sql.functions import avg, count, max, min, sum
    
    # Prepare data
    sales_data = [
        ("Alice", "2024-01", 100),
        ("Alice", "2024-02", 150),
        ("Bob", "2024-01", 200),
        ("Bob", "2024-02", 180),
        ("Charlie", "2024-01", 120),
        ("Charlie", "2024-02", 140)
    ]
    sales_df = spark.createDataFrame(sales_data, ["name", "month", "sales"])
    
    # Group and aggregate
    sales_summary = sales_df.groupBy("name").agg(
        sum("sales").alias("total_sales"),
        avg("sales").alias("avg_sales"),
        count("sales").alias("num_months")
    )
    sales_summary.show()
    
    # Group by multiple columns
    monthly_stats = sales_df.groupBy("name", "month").agg(
        max("sales").alias("max_sales"),
        min("sales").alias("min_sales")
    )
    monthly_stats.show()
    
    # Pivot table
    pivot_df = sales_df.groupBy("name").pivot("month").sum("sales")
    pivot_df.show()
    

**Output** :
    
    
    +-------+-----------+---------+----------+
    |   name|total_sales|avg_sales|num_months|
    +-------+-----------+---------+----------+
    |  Alice|        250|    125.0|         2|
    |    Bob|        380|    190.0|         2|
    |Charlie|        260|    130.0|         2|
    +-------+-----------+---------+----------+
    

### Using Spark SQL
    
    
    # Register DataFrame as temporary view
    df.createOrReplaceTempView("employees")
    
    # Execute SQL query
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
    
    # Complex SQL query
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
    

### Join Operations
    
    
    # Prepare data
    employees = spark.createDataFrame([
        (1, "Alice", "Engineering"),
        (2, "Bob", "Data Science"),
        (3, "Charlie", "Management")
    ], ["id", "name", "department"])
    
    salaries = spark.createDataFrame([
        (1, 80000),
        (2, 95000),
        (4, 70000)  # id=4 doesn't exist in employees table
    ], ["id", "salary"])
    
    # Inner Join
    inner_join = employees.join(salaries, "id", "inner")
    print("Inner Join:")
    inner_join.show()
    
    # Left Outer Join
    left_join = employees.join(salaries, "id", "left")
    print("Left Outer Join:")
    left_join.show()
    
    # Right Outer Join
    right_join = employees.join(salaries, "id", "right")
    print("Right Outer Join:")
    right_join.show()
    
    # Full Outer Join
    full_join = employees.join(salaries, "id", "outer")
    print("Full Outer Join:")
    full_join.show()
    

**Output (Inner Join)** :
    
    
    +---+-----+-------------+------+
    | id| name|   department|salary|
    +---+-----+-------------+------+
    |  1|Alice|  Engineering| 80000|
    |  2|  Bob| Data Science| 95000|
    +---+-----+-------------+------+
    

### Catalyst Optimizer Effects
    
    
    # Check query execution plan
    df_filtered = df.filter(df.age > 25).select("name", "age")
    
    # Physical execution plan
    print("Physical Plan:")
    df_filtered.explain(mode="formatted")
    
    # Logical plan before optimization
    print("\nLogical Plan:")
    df_filtered.explain(mode="extended")
    

> **Important** : Catalyst automatically applies optimizations such as predicate pushdown, column pruning, and constant folding.

* * *

## 2.4 Spark MLlib (Machine Learning)

### What is MLlib?

**Spark MLlib** is Spark's distributed machine learning library that efficiently executes training on large-scale data.

#### Key MLlib Features

  * **Classification** : Logistic regression, decision trees, random forest, GBT
  * **Regression** : Linear regression, regression trees, generalized linear models
  * **Clustering** : K-Means, GMM, LDA
  * **Collaborative Filtering** : ALS (Alternating Least Squares)
  * **Dimensionality Reduction** : PCA, SVD
  * **Feature Transformation** : VectorAssembler, StringIndexer, OneHotEncoder

### ML Pipeline Basics
    
    
    ```mermaid
    graph LR
        A[Raw Data] --> B[StringIndexer]
        B --> C[VectorAssembler]
        C --> D[StandardScaler]
        D --> E[Classifier]
        E --> F[Predictions]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff3e0
        style D fill:#fff3e0
        style E fill:#f3e5f5
        style F fill:#e8f5e9
    ```

### Implementing Classification Tasks
    
    
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    
    # Generate sample data
    from pyspark.sql.functions import rand, when
    
    # Generate data with structure similar to Iris dataset
    data = spark.range(0, 1000).select(
        (rand() * 3 + 4).alias("sepal_length"),
        (rand() * 2 + 2).alias("sepal_width"),
        (rand() * 3 + 1).alias("petal_length"),
        (rand() * 2 + 0.1).alias("petal_width")
    )
    
    # Create target variable
    data = data.withColumn(
        "species",
        when((data.petal_length < 2), "setosa")
        .when((data.petal_length >= 2) & (data.petal_length < 4), "versicolor")
        .otherwise("virginica")
    )
    
    # Check data
    data.show(10)
    data.groupBy("species").count().show()
    
    # Train-test split
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training data: {train_data.count()} rows")
    print(f"Test data: {test_data.count()} rows")
    

### Feature Transformation Pipeline
    
    
    # Stage 1: Convert categorical variable to index
    label_indexer = StringIndexer(
        inputCol="species",
        outputCol="label"
    )
    
    # Stage 2: Combine features into vector
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    vector_assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    
    # Stage 3: Logistic regression model
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01
    )
    
    # Build pipeline
    pipeline = Pipeline(stages=[label_indexer, vector_assembler, lr])
    
    # Train model
    print("Starting model training...")
    model = pipeline.fit(train_data)
    print("Training complete")
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Check prediction results
    predictions.select("species", "label", "features", "prediction", "probability").show(10, truncate=False)
    

### Model Evaluation
    
    
    # Multi-class classification evaluation
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )
    
    # Accuracy
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    print(f"Accuracy: {accuracy:.4f}")
    
    # F1 score
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    print(f"F1 Score: {f1:.4f}")
    
    # Weighted precision
    weighted_precision = multi_evaluator.evaluate(
        predictions,
        {multi_evaluator.metricName: "weightedPrecision"}
    )
    print(f"Weighted Precision: {weighted_precision:.4f}")
    
    # Weighted recall
    weighted_recall = multi_evaluator.evaluate(
        predictions,
        {multi_evaluator.metricName: "weightedRecall"}
    )
    print(f"Weighted Recall: {weighted_recall:.4f}")
    
    # Calculate confusion matrix
    from pyspark.ml.evaluation import MulticlassMetrics
    prediction_and_labels = predictions.select("prediction", "label").rdd
    metrics = MulticlassMetrics(prediction_and_labels)
    
    print("\nConfusion Matrix:")
    print(metrics.confusionMatrix().toArray())
    

### Implementing Regression Tasks
    
    
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Generate regression sample data
    regression_data = spark.range(0, 1000).select(
        (rand() * 100).alias("feature1"),
        (rand() * 50).alias("feature2"),
        (rand() * 30).alias("feature3")
    )
    
    # Target variable (linear relationship + noise)
    from pyspark.sql.functions import col
    regression_data = regression_data.withColumn(
        "target",
        col("feature1") * 2 + col("feature2") * 1.5 - col("feature3") * 0.5 + (rand() * 10 - 5)
    )
    
    # Train-test split
    train_reg, test_reg = regression_data.randomSplit([0.8, 0.2], seed=42)
    
    # Create feature vector
    feature_cols = ["feature1", "feature2", "feature3"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Linear regression model
    lr_regressor = LinearRegression(
        featuresCol="features",
        labelCol="target",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.5  # L1/L2 regularization mix ratio
    )
    
    # Build pipeline
    regression_pipeline = Pipeline(stages=[assembler, lr_regressor])
    
    # Train
    regression_model = regression_pipeline.fit(train_reg)
    
    # Make predictions
    regression_predictions = regression_model.transform(test_reg)
    
    # Evaluation
    reg_evaluator = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction"
    )
    
    rmse = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "rmse"})
    mae = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "mae"})
    r2 = reg_evaluator.evaluate(regression_predictions, {reg_evaluator.metricName: "r2"})
    
    print(f"\n=== Regression Model Evaluation ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Model coefficients
    lr_model = regression_model.stages[-1]
    print(f"\nCoefficients: {lr_model.coefficients}")
    print(f"Intercept: {lr_model.intercept:.4f}")
    

### Random Forest Classification
    
    
    from pyspark.ml.classification import RandomForestClassifier
    
    # Random forest model
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    # Pipeline (feature transformation + RF)
    rf_pipeline = Pipeline(stages=[label_indexer, vector_assembler, rf])
    
    # Train
    print("Starting random forest training...")
    rf_model = rf_pipeline.fit(train_data)
    print("Training complete")
    
    # Make predictions
    rf_predictions = rf_model.transform(test_data)
    
    # Evaluation
    rf_accuracy = multi_evaluator.evaluate(
        rf_predictions,
        {multi_evaluator.metricName: "accuracy"}
    )
    rf_f1 = multi_evaluator.evaluate(
        rf_predictions,
        {multi_evaluator.metricName: "f1"}
    )
    
    print(f"\n=== Random Forest Evaluation ===")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")
    
    # Feature importances
    rf_classifier = rf_model.stages[-1]
    feature_importances = rf_classifier.featureImportances
    
    print("\nFeature Importances:")
    for idx, importance in enumerate(feature_importances):
        print(f"{feature_columns[idx]}: {importance:.4f}")
    

### Cross-Validation and Hyperparameter Tuning
    
    
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    # Build parameter grid
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.maxIter, [50, 100, 150]) \
        .build()
    
    # Configure cross-validation
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=multi_evaluator,
        numFolds=3,
        seed=42
    )
    
    # Execute cross-validation
    print("Starting cross-validation...")
    cv_model = cv.fit(train_data)
    print("Complete")
    
    # Predict with best model
    cv_predictions = cv_model.transform(test_data)
    
    # Check best parameters
    best_model = cv_model.bestModel
    print("\nBest Parameters:")
    print(best_model.stages[-1].extractParamMap())
    
    # Evaluation
    cv_accuracy = multi_evaluator.evaluate(
        cv_predictions,
        {multi_evaluator.metricName: "accuracy"}
    )
    print(f"\nAccuracy after CV: {cv_accuracy:.4f}")
    

* * *

## 2.5 Performance Optimization

### Partitioning Strategies

Proper partitioning significantly affects Spark performance.

#### Determining Partition Count
    
    
    # Default partition count
    print(f"Default partition count: {spark.sparkContext.defaultParallelism}")
    
    # Check RDD partition count
    rdd = sc.parallelize(range(1000))
    print(f"RDD partition count: {rdd.getNumPartitions()}")
    
    # Check DataFrame partition count
    df = spark.range(10000)
    print(f"DataFrame partition count: {df.rdd.getNumPartitions()}")
    
    # Reset partition count
    rdd_repartitioned = rdd.repartition(8)
    print(f"After repartitioning: {rdd_repartitioned.getNumPartitions()}")
    
    # coalesce: Reduce partition count (without shuffle)
    rdd_coalesced = rdd.coalesce(4)
    print(f"After coalesce: {rdd_coalesced.getNumPartitions()}")
    

> **Recommendation** : Partition count guideline is (CPU cores × 2-3).

#### Custom Partitioner
    
    
    # Hash partitioning with key-value pairs
    pairs = sc.parallelize([("A", 1), ("B", 2), ("A", 3), ("C", 4), ("B", 5)])
    
    # Hash partitioning
    hash_partitioned = pairs.partitionBy(4)
    print(f"Hash partition count: {hash_partitioned.getNumPartitions()}")
    
    # Check contents of each partition
    def show_partition_contents(index, iterator):
        yield f"Partition {index}: {list(iterator)}"
    
    partition_contents = hash_partitioned.mapPartitionsWithIndex(show_partition_contents)
    for content in partition_contents.collect():
        print(content)
    

### Caching and Persistence

#### Memory Caching
    
    
    # Cache DataFrame
    df_large = spark.range(0, 10000000)
    
    # Cache (default: memory only)
    df_large.cache()
    
    # First action creates cache
    count1 = df_large.count()
    print(f"First count (creating cache): {count1}")
    
    # Second time onwards uses cache (fast)
    count2 = df_large.count()
    print(f"Second count (using cache): {count2}")
    
    # Release cache
    df_large.unpersist()
    

#### Choosing Persistence Level
    
    
    # Requirements:
    # - Python 3.9+
    # - pyspark>=3.4.0
    
    """
    Example: Choosing Persistence Level
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from pyspark import StorageLevel
    
    # RDD persistence level
    rdd = sc.parallelize(range(1000000))
    
    # Use both memory and disk
    rdd.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Serialize and store in memory (improved memory efficiency)
    rdd.persist(StorageLevel.MEMORY_ONLY_SER)
    
    # Replication (improved fault tolerance)
    rdd.persist(StorageLevel.MEMORY_AND_DISK_2)
    
    print(f"Storage level: {rdd.getStorageLevel()}")
    

Storage Level | Description | Use Case  
---|---|---  
`MEMORY_ONLY` | Memory only (default) | When sufficient memory available  
`MEMORY_AND_DISK` | Memory → spill to disk | Large-scale data  
`MEMORY_ONLY_SER` | Serialize and store in memory | Memory efficiency priority  
`DISK_ONLY` | Disk only | Memory shortage  
`OFF_HEAP` | Off-heap memory | Avoid GC  
  
### Broadcast Variables
    
    
    # Distribute small dataset to all nodes
    lookup_table = {"A": 100, "B": 200, "C": 300, "D": 400}
    
    # Broadcast
    broadcast_lookup = sc.broadcast(lookup_table)
    
    # Use broadcast variable in RDD
    data = sc.parallelize([("A", 1), ("B", 2), ("C", 3), ("A", 4)])
    
    def enrich_data(pair):
        key, value = pair
        # Reference broadcast variable
        multiplier = broadcast_lookup.value.get(key, 1)
        return (key, value * multiplier)
    
    enriched = data.map(enrich_data)
    print(enriched.collect())
    
    # Release broadcast variable
    broadcast_lookup.unpersist()
    

> **Important** : Broadcast variables significantly improve join operation performance (especially joins with small tables).

### Tuning Parameters

#### Spark Session Configuration
    
    
    # Performance optimization settings
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
    
    # Check configuration
    for conf in spark_optimized.sparkContext.getConf().getAll():
        print(f"{conf[0]}: {conf[1]}")
    

#### Key Tuning Parameters

Parameter | Description | Recommended Value  
---|---|---  
`spark.executor.memory` | Executor memory size | 70% of available memory  
`spark.executor.cores` | Cores per executor | 4-6 cores  
`spark.default.parallelism` | Default parallelism | Cores × 2-3  
`spark.sql.shuffle.partitions` | Partitions during shuffle | 100-200 (data size dependent)  
`spark.sql.adaptive.enabled` | Adaptive query execution | `true`  
`spark.serializer` | Serializer | `KryoSerializer`  
  
### Execution Plan Optimization
    
    
    # DataFrame optimization example
    large_df = spark.range(0, 10000000)
    small_df = spark.range(0, 100)
    
    # Before optimization: filter on large table → join
    result_unoptimized = large_df.filter(large_df.id % 2 == 0).join(small_df, "id")
    
    # After optimization: join → filter (predicate pushdown)
    result_optimized = large_df.join(small_df, "id").filter(large_df.id % 2 == 0)
    
    # Compare execution plans
    print("Before optimization:")
    result_unoptimized.explain()
    
    print("\nAfter optimization:")
    result_optimized.explain()
    
    # Catalyst automatically optimizes, so both actually have same execution plan
    

* * *

## 2.6 Chapter Summary

### What We Learned

  1. **Spark Architecture**

     * Distributed processing with Driver-Executor model
     * Lazy Evaluation and DAG execution
     * Cluster managers (YARN, Mesos, K8s)
     * Distinction between Transformations and Actions
  2. **RDD (Resilient Distributed Datasets)**

     * Immutable, distributed, fault-tolerant collections
     * Automatic recovery through lineage
     * Operations like map, filter, reduceByKey
     * Key-Value pair processing
  3. **Spark DataFrames and SQL**

     * Faster execution through Catalyst Optimizer
     * Type safety through schema information
     * Integration of SQL queries and DataFrame API
     * Efficient processing of joins, aggregations, grouping
  4. **Spark MLlib**

     * Distributed machine learning pipelines
     * Feature transformation and preprocessing
     * Classification, regression, clustering
     * Cross-validation and hyperparameter tuning
  5. **Performance Optimization**

     * Appropriate partitioning strategies
     * Caching and persistence levels
     * Join optimization with broadcast variables
     * Tuning parameter configuration

### Spark Best Practices

Item | Recommendation  
---|---  
**API Selection** | DataFrame/Dataset > RDD (optimization benefits)  
**Partitioning** | Appropriate count (cores × 2-3), even distribution  
**Caching** | Cache only reused intermediate results  
**Shuffle Reduction** | Avoid unnecessary groupByKey, use reduceByKey  
**Broadcast** | Utilize for joins with small tables  
**Memory Management** | Appropriate Executor memory configuration  
  
### Next Chapter

In Chapter 3, we'll learn about **Distributed Deep Learning Frameworks** :

  * Distributed training with Horovod
  * TensorFlow and PyTorch distributed strategies
  * Massively parallel processing with Ray
  * Experiment management with MLflow
  * Distributed hyperparameter optimization

* * *

## Exercises

### Exercise 1 (Difficulty: Easy)

Explain the difference between Transformations and Actions in Spark, and provide three examples of each.

Sample Answer

**Answer** :

**Transformation** :

  * Definition: Operations that return a new RDD/DataFrame and are lazily evaluated
  * Characteristics: Actual computation is not executed, an execution plan (DAG) is built
  * Examples: 
    1. `map()` \- Apply function to each element
    2. `filter()` \- Extract elements matching condition
    3. `groupBy()` \- Group by key

**Action** :

  * Definition: Operations that return results or save, eagerly evaluated
  * Characteristics: Actual computation is executed, data is returned to Driver or storage
  * Examples: 
    1. `count()` \- Count elements
    2. `collect()` \- Retrieve all elements
    3. `saveAsTextFile()` \- Save to file

**Importance of Difference** :

Transformations are fast because they only build the DAG. When an Action is called, Spark executes computation with an optimized execution plan.

### Exercise 2 (Difficulty: Medium)

What problems might occur when executing the following code? How should it be fixed?
    
    
    rdd = sc.parallelize(range(1, 1000000))
    result = rdd.map(lambda x: x ** 2).collect()
    print(result)
    

Sample Answer

**Problems** :

  1. **Out of memory** : `collect()` gathers all data into Driver memory, potentially causing out-of-memory with 1 million elements
  2. **Performance degradation** : Loses benefits of distributed processing
  3. **Network load** : Large data transfer from Executors to Driver

**Fixes** :
    
    
    # Method 1: Get only necessary elements
    rdd = sc.parallelize(range(1, 1000000))
    result = rdd.map(lambda x: x ** 2).take(10)  # Only first 10 elements
    print(result)
    
    # Method 2: Save to file
    rdd.map(lambda x: x ** 2).saveAsTextFile("output/squares")
    
    # Method 3: Use aggregation operation
    total = rdd.map(lambda x: x ** 2).sum()
    print(f"Sum: {total}")
    
    # Method 4: Sampling
    sample = rdd.map(lambda x: x ** 2).sample(False, 0.01).collect()
    print(f"Sample: {sample[:10]}")
    

**Best Practices** :

  * Use `collect()` only for small datasets (few thousand rows or less)
  * For large-scale data, use `take(n)`, `sample()`, `saveAsTextFile()`

### Exercise 3 (Difficulty: Medium)

Implement the following SQL query using the DataFrame API in Spark.
    
    
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
    

Sample Answer
    
    
    from pyspark.sql.functions import avg, max, count, col
    
    # DataFrame API version
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
    
    # Alternative notation (method chaining)
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
    

**Explanation** :

  * `filter()` / `where()`: WHERE clause
  * `groupBy()`: GROUP BY clause
  * `agg()`: Aggregation functions (AVG, MAX, COUNT)
  * `filter()` (second time): HAVING clause
  * `orderBy()` / `sort()`: ORDER BY clause

### Exercise 4 (Difficulty: Hard)

Explain how to efficiently perform Key-Value pair joins on a large dataset (100 million rows) for the following three scenarios:

  1. When both datasets are large
  2. When one dataset is small (fits in memory)
  3. When data is already sorted

Sample Answer

**Answer** :

#### Scenario 1: Both datasets are large
    
    
    # Standard join (sort-merge join or hash join)
    large_df1 = spark.read.parquet("large_dataset1.parquet")
    large_df2 = spark.read.parquet("large_dataset2.parquet")
    
    # Optimize partition count
    large_df1 = large_df1.repartition(200, "join_key")
    large_df2 = large_df2.repartition(200, "join_key")
    
    # Join
    result = large_df1.join(large_df2, "join_key", "inner")
    
    # Cache (if reusing)
    result.cache()
    result.count()  # Materialize cache
    

**Optimization points** :

  * Appropriate partition count (adjust based on data size)
  * Pre-partition by join key
  * Enable adaptive query execution (AQE)

#### Scenario 2: One dataset is small
    
    
    from pyspark.sql.functions import broadcast
    
    large_df = spark.read.parquet("large_dataset.parquet")
    small_df = spark.read.parquet("small_dataset.parquet")
    
    # Broadcast join (distribute small table to all nodes)
    result = large_df.join(broadcast(small_df), "join_key", "inner")
    
    # Or set automatic broadcast threshold
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10485760)  # 10MB
    

**Optimization points** :

  * Broadcast small tables (<10MB)
  * No shuffle needed, significant speedup
  * Be careful with memory usage (distributed to all Executors)

#### Scenario 3: Data is already sorted
    
    
    # When data is sorted and partitioned by join key
    sorted_df1 = spark.read.parquet("sorted_dataset1.parquet")
    sorted_df2 = spark.read.parquet("sorted_dataset2.parquet")
    
    # Explicitly use sort-merge join
    result = sorted_df1.join(
        sorted_df2,
        sorted_df1["join_key"] == sorted_df2["join_key"],
        "inner"
    )
    
    # Force sort-merge join with hint
    from pyspark.sql.functions import expr
    result = sorted_df1.hint("merge").join(sorted_df2, "join_key")
    

**Optimization points** :

  * Reduced shuffle if already sorted
  * Pre-partition using bucketing
  * Maintain sort order when saving in Parquet format

#### Performance Comparison

Scenario | Join Type | Shuffle | Speed  
---|---|---|---  
Both large | Sort-merge/Hash | Yes | Medium  
One small | Broadcast | No | Fast  
Already sorted | Sort-merge | Partial | Fast  
  
### Exercise 5 (Difficulty: Hard)

Build a complete pipeline for a text classification task (spam detection) using Spark MLlib. Include:

  * Text preprocessing (tokenization, stopword removal)
  * TF-IDF feature creation
  * Training logistic regression model
  * Evaluation with cross-validation

Sample Answer
    
    
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    # Create sample data
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
    ] * 100, ["label", "text"])  # Increase data
    
    print(f"Data count: {data.count()}")
    data.show(5)
    
    # Train-test split
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    # Stage 1: Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # Stage 2: Stopword removal
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Stage 3: Hashing TF
    hashingTF = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=1000
    )
    
    # Stage 4: IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Stage 5: Logistic regression
    lr = LogisticRegression(maxIter=100, regParam=0.01)
    
    # Build pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    
    # Parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [500, 1000, 2000]) \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.maxIter, [50, 100]) \
        .build()
    
    # Cross-validation
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=BinaryClassificationEvaluator(),
        numFolds=3,
        seed=42
    )
    
    # Train
    print("\nStarting cross-validation...")
    cv_model = cv.fit(train)
    print("Training complete")
    
    # Make predictions
    predictions = cv_model.transform(test)
    
    # Check prediction results
    predictions.select("text", "label", "prediction", "probability").show(10, truncate=False)
    
    # Evaluation
    binary_evaluator = BinaryClassificationEvaluator()
    multi_evaluator = MulticlassClassificationEvaluator()
    
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    print("\n=== Model Evaluation ===")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Best parameters
    best_model = cv_model.bestModel
    print("\nBest Parameters:")
    print(f"numFeatures: {best_model.stages[2].getNumFeatures()}")
    print(f"regParam: {best_model.stages[-1].getRegParam()}")
    print(f"maxIter: {best_model.stages[-1].getMaxIter()}")
    
    # Predict on new text
    new_data = spark.createDataFrame([
        (0, "Free lottery winner claim now"),
        (1, "Project deadline next Monday")
    ], ["id", "text"])
    
    new_predictions = cv_model.transform(new_data)
    new_predictions.select("text", "prediction", "probability").show(truncate=False)
    

**Sample output** :
    
    
    === Model Evaluation ===
    AUC: 0.9850
    Accuracy: 0.9500
    F1 Score: 0.9495
    

**Extension ideas** :

  * Use Word2Vec or GloVe embeddings
  * Add N-gram features
  * Try Random Forest or GBT
  * Add custom features (sentence length, uppercase ratio, etc.)

* * *

## References

  1. Zaharia, M., et al. (2016). _Apache Spark: A Unified Engine for Big Data Processing_. Communications of the ACM, 59(11), 56-65.
  2. Karau, H., Konwinski, A., Wendell, P., & Zaharia, M. (2015). _Learning Spark: Lightning-Fast Big Data Analysis_. O'Reilly Media.
  3. Chambers, B., & Zaharia, M. (2018). _Spark: The Definitive Guide_. O'Reilly Media.
  4. Meng, X., et al. (2016). _MLlib: Machine Learning in Apache Spark_. Journal of Machine Learning Research, 17(1), 1235-1241.
  5. Apache Spark Documentation. (2024). _Spark SQL, DataFrames and Datasets Guide_. URL: https://spark.apache.org/docs/latest/sql-programming-guide.html
  6. Databricks. (2024). _Apache Spark Performance Tuning Guide_. URL: https://www.databricks.com/blog/performance-tuning
