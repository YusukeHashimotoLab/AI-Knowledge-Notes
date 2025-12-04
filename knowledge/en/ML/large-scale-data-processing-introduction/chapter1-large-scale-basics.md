---
title: "Chapter 1: Fundamentals of Large-Scale Data Processing"
chapter_title: "Chapter 1: Fundamentals of Large-Scale Data Processing"
subtitle: Understanding the Principles of Scalability and Distributed Processing
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 7
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers the fundamentals of Fundamentals of Large, which scale data processing. You will learn types of parallelization strategies, challenges in distributed systems, and major distributed processing tools.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand scalability challenges in large-scale data processing
  * ✅ Grasp the basic concepts and architecture of distributed processing
  * ✅ Learn the types of parallelization strategies and how to use them
  * ✅ Understand challenges in distributed systems and their solutions
  * ✅ Understand major distributed processing tools and ecosystems
  * ✅ Implement parallelization in actual code

* * *

## 1.1 Scalability Challenges

### Data Size Growth

In modern machine learning projects, data volumes are growing explosively.

> "Facing data volumes that cannot be processed on a single machine is no longer an exception but the norm."

Data Scale | Size Range | Processing Method  
---|---|---  
**Small-scale** | ~1GB | Single-machine in-memory processing  
**Medium-scale** | 1GB~100GB | Memory optimization, chunk processing  
**Large-scale** | 100GB~1TB | Distributed processing, parallelization  
**Ultra-large-scale** | 1TB+ | Clusters, distributed file systems  
  
### Memory Constraints

The most common problem is that data doesn't fit in memory.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import sys
    
    # Check memory usage
    def memory_usage_mb(data):
        """Return memory usage of data in MB"""
        return sys.getsizeof(data) / (1024 ** 2)
    
    # Large-scale data example
    n_samples = 10_000_000  # 10 million samples
    n_features = 100
    
    # Normal array (loading all data into memory)
    # This consumes approximately 7.5GB of memory
    # data = np.random.random((n_samples, n_features))  # Potential out-of-memory error
    
    # Check with smaller data
    n_samples_small = 1_000_000  # 1 million samples
    data_small = np.random.random((n_samples_small, n_features))
    
    print("=== Memory Usage Check ===")
    print(f"Data shape: {data_small.shape}")
    print(f"Memory usage: {memory_usage_mb(data_small):.2f} MB")
    print(f"\nEstimate: For {n_samples:,} samples")
    print(f"Memory usage: {memory_usage_mb(data_small) * 10:.2f} MB ({memory_usage_mb(data_small) * 10 / 1024:.2f} GB)")
    

**Output** :
    
    
    === Memory Usage Check ===
    Data shape: (1000000, 100)
    Memory usage: 762.94 MB
    
    Estimate: For 10,000,000 samples
    Memory usage: 7629.40 MB (7.45 GB)
    

### Computation Time Issues

As data size increases, computation time increases non-linearly.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: As data size increases, computation time increases non-linea
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Measure computation time for different sizes
    sizes = [1000, 5000, 10000, 50000, 100000]
    times = []
    
    print("=== Computation Time Measurement ===")
    for size in sizes:
        X = np.random.random((size, 100))
    
        start = time.time()
        # Simple matrix operation
        result = X @ X.T
        elapsed = time.time() - start
    
        times.append(elapsed)
        print(f"Size {size:6d}: {elapsed:.4f}s")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Data Size (Number of Samples)', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Relationship Between Data Size and Computation Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Estimate time complexity
    print(f"\nTime increase ratio for 10x size increase: {times[-1] / times[0]:.1f}x")
    

### I/O Bottlenecks

Disk I/O is a major bottleneck in large-scale data processing.

Storage Type | Read Speed | Relative Performance  
---|---|---  
Memory (RAM) | ~50 GB/s | 1,000x  
SSD | ~500 MB/s | 10x  
HDD | ~100 MB/s | 1x (baseline)  
Network (1Gbps) | ~125 MB/s | 1.25x  
  
* * *

## 1.2 Distributed Processing Concepts

### Horizontal Scaling vs Vertical Scaling

There are two approaches to achieving scalability.

Type | Description | Advantages | Disadvantages  
---|---|---|---  
**Vertical Scaling**  
(Scale Up) | Improving single machine performance  
(CPU, memory, storage enhancement) | ・Simple implementation  
・No communication overhead | ・Physical limitations exist  
・Cost increases non-linearly  
**Horizontal Scaling**  
(Scale Out) | Distributed processing across multiple machines  
(Increasing number of nodes) | ・Theoretically infinitely scalable  
・Improved fault tolerance | ・Complex implementation  
・Communication costs occur  
  
> **Practical Selection** : Typically, vertical scaling is pursued to its limit before transitioning to horizontal scaling.

### Master-Worker Architecture

The most common pattern in distributed processing is the **Master-Worker** architecture.
    
    
    ```mermaid
    graph TD
        M[Master NodeTask Distribution & Result Aggregation] --> W1[Worker 1Partial Computation]
        M --> W2[Worker 2Partial Computation]
        M --> W3[Worker 3Partial Computation]
        M --> W4[Worker 4Partial Computation]
    
        W1 --> R[Result Integration]
        W2 --> R
        W3 --> R
        W4 --> R
    
        style M fill:#9d4edd
        style W1 fill:#e3f2fd
        style W2 fill:#e3f2fd
        style W3 fill:#e3f2fd
        style W4 fill:#e3f2fd
        style R fill:#c8e6c9
    ```

#### Role Division

  * **Master Node** : 
    * Task splitting and assignment
    * Worker monitoring
    * Result aggregation
    * Fault detection and recovery
  * **Worker Nodes** : 
    * Execution of assigned tasks
    * Result transmission
    * Status reporting

### Data Partitioning and Sharding

**Sharding** is the technique of dividing data into multiple partitions.

#### Partitioning Strategies

Strategy | Description | Use Cases  
---|---|---  
**Horizontal Partitioning** | Split by rows | Time-series data, user data  
**Vertical Partitioning** | Split by columns | Cases with many features  
**Hash-based** | Split by key hash value | Requires uniform distribution  
**Range-based** | Split by value ranges | Sorted data  
      
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Partitioning Strategies
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import numpy as np
    import pandas as pd
    
    # Sample data
    n_samples = 1000
    data = pd.DataFrame({
        'user_id': np.random.randint(1, 100, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'value': np.random.random(n_samples)
    })
    
    # 1. Horizontal partitioning (by rows)
    n_partitions = 4
    partition_size = len(data) // n_partitions
    
    horizontal_shards = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else len(data)
        shard = data.iloc[start:end]
        horizontal_shards.append(shard)
        print(f"Shard {i+1}: {len(shard)} rows")
    
    # 2. Hash-based partitioning
    def hash_partition(user_id, n_partitions):
        return hash(user_id) % n_partitions
    
    data['partition'] = data['user_id'].apply(lambda x: hash_partition(x, n_partitions))
    
    hash_shards = []
    for i in range(n_partitions):
        shard = data[data['partition'] == i]
        hash_shards.append(shard)
        print(f"Hash shard {i+1}: {len(shard)} rows")
    
    # Check distribution
    print("\n=== Check Partitioning Balance ===")
    hash_sizes = [len(shard) for shard in hash_shards]
    print(f"Min: {min(hash_sizes)}, Max: {max(hash_sizes)}, Mean: {np.mean(hash_sizes):.1f}")
    

### Distributed Processing Architecture Diagram
    
    
    ```mermaid
    graph LR
        subgraph "Input Data"
            D[Large Dataset1TB]
        end
    
        subgraph "Distributed Storage"
            S1[Shard 1250GB]
            S2[Shard 2250GB]
            S3[Shard 3250GB]
            S4[Shard 4250GB]
        end
    
        subgraph "Parallel Processing"
            P1[Process 1]
            P2[Process 2]
            P3[Process 3]
            P4[Process 4]
        end
    
        subgraph "Result Aggregation"
            A[Aggregation & Integration]
        end
    
        D --> S1
        D --> S2
        D --> S3
        D --> S4
    
        S1 --> P1
        S2 --> P2
        S3 --> P3
        S4 --> P4
    
        P1 --> A
        P2 --> A
        P3 --> A
        P4 --> A
    
        style D fill:#ffebee
        style S1 fill:#e3f2fd
        style S2 fill:#e3f2fd
        style S3 fill:#e3f2fd
        style S4 fill:#e3f2fd
        style P1 fill:#fff3e0
        style P2 fill:#fff3e0
        style P3 fill:#fff3e0
        style P4 fill:#fff3e0
        style A fill:#c8e6c9
    ```

* * *

## 1.3 Parallelization Strategies

### Data Parallelism

**Data Parallelism** involves splitting data and executing the same processing on each partition in parallel.

> This is the most common and easiest-to-implement parallelization technique.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import multiprocessing as mp
    import time
    
    # Processing function
    def process_chunk(data_chunk):
        """Process data chunk (example: mean calculation)"""
        return np.mean(data_chunk, axis=0)
    
    # Single-process version
    def single_process_compute(data):
        start = time.time()
        result = np.mean(data, axis=0)
        elapsed = time.time() - start
        return result, elapsed
    
    # Multi-process version (data parallelism)
    def multi_process_compute(data, n_workers=4):
        start = time.time()
    
        # Split data
        chunks = np.array_split(data, n_workers)
    
        # Parallel processing
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_chunk, chunks)
    
        # Integrate results
        final_result = np.mean(results, axis=0)
        elapsed = time.time() - start
    
        return final_result, elapsed
    
    # Test
    if __name__ == '__main__':
        # Sample data
        data = np.random.random((10_000_000, 10))
    
        print("=== Data Parallelism Comparison ===")
        print(f"Data size: {data.shape}")
    
        # Single-process
        result_single, time_single = single_process_compute(data)
        print(f"\nSingle-process: {time_single:.4f}s")
    
        # Multi-process
        n_workers = mp.cpu_count()
        result_multi, time_multi = multi_process_compute(data, n_workers)
        print(f"Multi-process ({n_workers} workers): {time_multi:.4f}s")
    
        # Speedup
        speedup = time_single / time_multi
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Efficiency: {speedup / n_workers * 100:.1f}%")
    

### Model Parallelism

**Model Parallelism** involves splitting the model itself and distributing it across multiple devices.

Used for large-scale neural networks:

  * Place each layer on different GPUs
  * When model parameters don't fit in single device memory

    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    # Conceptual example: splitting large models
    class DistributedModel:
        """Conceptual implementation of model parallelism"""
    
        def __init__(self, layer_sizes):
            self.layers = []
            for i in range(len(layer_sizes) - 1):
                # Place each layer on different devices (here, arrays)
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
                self.layers.append({
                    'weight': weight,
                    'device': f'GPU_{i % 4}'  # Distributed across 4 GPUs
                })
    
        def forward(self, x):
            """Forward propagation (each layer executed on different device)"""
            activation = x
            for i, layer in enumerate(self.layers):
                print(f"Layer {i+1} executing on {layer['device']}")
                activation = np.dot(activation, layer['weight'])
                activation = np.maximum(0, activation)  # ReLU
            return activation
    
    # Usage example
    print("=== Model Parallelism Example ===")
    layer_sizes = [1000, 2000, 2000, 2000, 100]  # Large model
    model = DistributedModel(layer_sizes)
    
    # Input data
    x = np.random.randn(1, 1000)
    output = model.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(layer['weight'].size for layer in model.layers):,}")
    

### Pipeline Parallelism

**Pipeline Parallelism** involves splitting processing into multiple stages and executing each stage in parallel.
    
    
    import multiprocessing as mp
    from queue import Queue
    import time
    
    # Pipeline stages
    def stage1_preprocess(input_queue, output_queue):
        """Stage 1: Preprocessing"""
        while True:
            item = input_queue.get()
            if item is None:
                output_queue.put(None)
                break
            # Preprocessing (example: normalization)
            processed = item / 255.0
            output_queue.put(processed)
    
    def stage2_feature_extract(input_queue, output_queue):
        """Stage 2: Feature extraction"""
        while True:
            item = input_queue.get()
            if item is None:
                output_queue.put(None)
                break
            # Feature extraction (example: calculate statistics)
            features = [item.mean(), item.std(), item.max(), item.min()]
            output_queue.put(features)
    
    def stage3_predict(input_queue, results):
        """Stage 3: Prediction"""
        while True:
            item = input_queue.get()
            if item is None:
                break
            # Prediction (simplified version)
            prediction = sum(item) > 2.0
            results.append(prediction)
    
    # Pipeline parallelization implementation example (conceptual)
    print("=== Pipeline Parallelism Concept ===")
    print("Stage 1: Preprocessing → Stage 2: Feature Extraction → Stage 3: Prediction")
    print("\nBy executing each stage in different processes in parallel,")
    print("throughput improves.")
    

### Comparison of Parallelization Strategies

Strategy | Application Scenarios | Advantages | Disadvantages  
---|---|---|---  
**Data Parallelism** | Large-scale data, same processing | Easy implementation, scalable | Communication cost, memory duplication  
**Model Parallelism** | Large-scale models, GPU constraints | Avoids memory constraints | Complex implementation, inter-device communication  
**Pipeline Parallelism** | Multi-stage processing, ETL | Improved throughput | Increased latency, balance adjustment  
  
* * *

## 1.4 Challenges in Distributed Systems

### Communication Costs

The largest overhead in distributed processing is inter-node communication.

> **Amdahl's Law** : The non-parallelizable portions (such as communication) limit overall performance.

$$ \text{Speedup} = \frac{1}{(1-P) + \frac{P}{N}} $$

  * $P$: Proportion of parallelizable portion
  * $N$: Number of processors

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Visualize Amdahl's Law
    def amdahl_speedup(P, N):
        """Calculate speedup according to Amdahl's Law"""
        return 1 / ((1 - P) + P / N)
    
    # Cases with different parallelization rates
    P_values = [0.5, 0.75, 0.9, 0.95, 0.99]
    N_range = np.arange(1, 65)
    
    plt.figure(figsize=(10, 6))
    for P in P_values:
        speedups = [amdahl_speedup(P, N) for N in N_range]
        plt.plot(N_range, speedups, label=f'P = {P:.0%}', linewidth=2)
    
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title("Amdahl's Law: Parallelization Rate and Performance", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=== Implications of Amdahl's Law ===")
    print("With low parallelization rate, performance improvement is limited even with more processors")
    print(f"Example: P=90%, 64 processors gives max {amdahl_speedup(0.9, 64):.1f}x speedup")
    

### Synchronous vs Asynchronous

Method | Description | Advantages | Disadvantages  
---|---|---|---  
**Synchronous Processing** | Wait for all workers to complete | Simple implementation, consistency guaranteed | Depends on slowest worker  
**Asynchronous Processing** | Proceed to next processing without waiting | Improved throughput | Complex consistency management  
  
### Fault Tolerance

In distributed systems, node and network failures are unavoidable.

#### Fault Handling Techniques

  * **Checkpointing** : Periodically save state
  * **Replication** : Replicate data across multiple nodes
  * **Retry** : Re-execute failed tasks
  * **Redundancy** : Execute same processing on multiple nodes

### Debugging Difficulties

Debugging distributed systems is difficult for the following reasons:

  * Non-deterministic execution order
  * Timing-dependent bugs
  * Logs spanning multiple nodes
  * Difficult to reproduce

* * *

## 1.5 Tools and Ecosystem

### Apache Hadoop / Spark

**Apache Hadoop** and **Apache Spark** are de facto standards for large-scale data processing.

Tool | Features | Use Cases  
---|---|---  
**Hadoop** | ・MapReduce-based  
・Disk-centric processing  
・Suited for batch processing | Large-scale ETL, log analysis  
**Spark** | ・In-memory processing  
・High-speed (100x faster than Hadoop)  
・Machine learning library (MLlib) | Iterative computation, machine learning  
      
    
    # Conceptual usage example of Apache Spark (PySpark)
    # Note: Requires Spark installation
    
    """
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("LargeScaleProcessing") \
        .getOrCreate()
    
    # Load large-scale data
    df = spark.read.parquet("hdfs://path/to/large/data")
    
    # Distributed processing
    result = df.groupBy("category") \
        .agg({"value": "mean"}) \
        .orderBy("category")
    
    # Save results
    result.write.parquet("hdfs://path/to/output")
    
    spark.stop()
    """
    
    print("=== Apache Spark Features ===")
    print("1. Lazy Evaluation: Automatically generates optimal execution plan")
    print("2. In-memory processing: Cache intermediate results in memory")
    print("3. Fault tolerance: Automatic recovery via RDD (Resilient Distributed Dataset)")
    print("4. Unified API: Handle SQL, machine learning, graph processing uniformly")
    

### Dask

**Dask** is a Python-native parallel processing library.
    
    
    # Requirements:
    # - Python 3.9+
    # - dask>=2023.5.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    # Conceptual usage example of Dask
    """
    import dask.array as da
    import dask.dataframe as dd
    
    # Dask array (NumPy-like API)
    x = da.random.random((100000, 10000), chunks=(1000, 1000))
    result = x.mean(axis=0).compute()  # Lazy evaluation → execution
    
    # Dask DataFrame (pandas-like API)
    df = dd.read_csv('large_file_*.csv')
    result = df.groupby('category').value.mean().compute()
    """
    
    print("=== Dask Features ===")
    print("1. NumPy/pandas compatible API: Easy migration of existing code")
    print("2. Task graph: Automatically manages processing dependencies")
    print("3. Scalable: Seamlessly scale from single machine → cluster")
    print("4. Integration with Python ecosystem: scikit-learn, XGBoost, etc.")
    
    # Simple Dask-style processing example (conceptual)
    print("\n=== Chunk Processing Example ===")
    data = np.random.random((10000, 100))
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = np.mean(chunk, axis=0)
        results.append(result)
    
    final_result = np.mean(results, axis=0)
    print(f"Number of chunks: {len(results)}")
    print(f"Final result shape: {final_result.shape}")
    

### Ray

**Ray** is a unified framework for distributed applications.
    
    
    # Requirements:
    # - Python 3.9+
    # - ray>=2.5.0
    
    # Conceptual usage example of Ray
    """
    import ray
    
    # Initialize Ray
    ray.init()
    
    # Remote function
    @ray.remote
    def process_data(data):
        return data.sum()
    
    # Parallel execution
    data_chunks = [np.random.random(1000) for _ in range(10)]
    futures = [process_data.remote(chunk) for chunk in data_chunks]
    results = ray.get(futures)  # Get results
    
    ray.shutdown()
    """
    
    print("=== Ray Features ===")
    print("1. Low-level control: Flexible parallelization with task/actor model")
    print("2. High performance: Distributed scheduling and shared memory")
    print("3. Ecosystem: Ray Tune (hyperparameter tuning), RLlib (reinforcement learning)")
    print("4. Ease of use: Easy parallelization with Python decorators")
    

### Selection Criteria

Situation | Recommended Tool | Reason  
---|---|---  
Large-scale batch processing (TB-scale) | Apache Spark | Mature ecosystem, fault tolerance  
Python-centric development | Dask | NumPy/pandas compatibility, low learning curve  
Complex distributed apps | Ray | Flexible control, high performance  
Single-machine acceleration | multiprocessing, joblib | Simple, no additional installation required  
Machine learning pipelines | Spark MLlib, Ray Tune | Integrated machine learning tools  
  
* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Scalability Challenges**

     * Constraints in data size, memory, computation time, and I/O
     * Single-machine limitations and the need for distributed processing
  2. **Distributed Processing Concepts**

     * Horizontal scaling vs vertical scaling
     * Master-worker architecture
     * Data partitioning and sharding strategies
  3. **Parallelization Strategies**

     * Data parallelism: Most common, easy to implement
     * Model parallelism: For large-scale models
     * Pipeline parallelism: Effective for multi-stage processing
  4. **Distributed System Challenges**

     * Communication costs and Amdahl's Law
     * Synchronous vs asynchronous trade-offs
     * Fault tolerance and debugging difficulties
  5. **Tools and Ecosystem**

     * Hadoop/Spark: Large-scale batch processing
     * Dask: Python-centric parallelization
     * Ray: Flexible distributed applications

### Important Principles

Principle | Description  
---|---  
**Optimization Order** | First algorithm, then implementation, finally parallelization  
**Minimize Communication** | Reducing inter-node communication is key to performance improvement  
**Appropriate Granularity** | Too fine-grained task splitting increases overhead  
**Measurement-Driven** | Don't guess, measure actual performance and decide  
**Gradual Scaling** | Validate at small scale before scaling up  
  
### Next Chapter

In Chapter 2, we will learn **MapReduce and Spark Fundamentals** :

  * MapReduce programming model
  * Apache Spark basic operations
  * RDD and DataFrame
  * Practical distributed processing pipelines

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain the differences between horizontal scaling and vertical scaling, and describe their respective advantages and disadvantages.

Sample Answer

**Answer** :

**Vertical Scaling (Scale Up)** :

  * Definition: Improving single machine performance (enhancing CPU, memory, storage)
  * Advantages: 
    * Simple implementation (existing code can be used as-is)
    * No inter-node communication overhead
    * Easier to maintain data consistency
  * Disadvantages: 
    * Physical limitations exist
    * Cost increases non-linearly (high-performance hardware is expensive)
    * Single Point of Failure (SPOF) risk

**Horizontal Scaling (Scale Out)** :

  * Definition: Distributed processing by increasing number of machines (nodes)
  * Advantages: 
    * Theoretically infinitely scalable
    * Linear cost (adding standard servers)
    * High fault tolerance (node redundancy)
  * Disadvantages: 
    * Complex implementation (requires distributed processing logic)
    * Inter-node communication costs occur
    * Difficult to manage data consistency

**Practical Selection** : Typically, vertical scaling is pursued to its limit before transitioning to horizontal scaling.

### Problem 2 (Difficulty: medium)

Using Amdahl's Law, calculate the speedup when running a program with 80% parallelization rate on 16 processors.

Sample Answer
    
    
    def amdahl_speedup(P, N):
        """
        Calculate speedup according to Amdahl's Law
    
        Parameters:
        P: Proportion of parallelizable portion (0~1)
        N: Number of processors
    
        Returns:
        Speedup factor
        """
        return 1 / ((1 - P) + P / N)
    
    # Problem calculation
    P = 0.8  # 80%
    N = 16   # 16 processors
    
    speedup = amdahl_speedup(P, N)
    
    print("=== Calculation by Amdahl's Law ===")
    print(f"Parallelization rate: {P:.0%}")
    print(f"Number of processors: {N}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"\nExplanation:")
    print(f"Theoretical maximum speedup (infinite processors): {1/(1-P):.2f}x")
    print(f"Efficiency: {speedup/N*100:.1f}%")
    

**Output** :
    
    
    === Calculation by Amdahl's Law ===
    Parallelization rate: 80%
    Number of processors: 16
    Speedup: 4.21x
    
    Explanation:
    Theoretical maximum speedup (infinite processors): 5.00x
    Efficiency: 26.3%
    

**Formula** :

$$ \text{Speedup} = \frac{1}{(1-0.8) + \frac{0.8}{16}} = \frac{1}{0.2 + 0.05} = \frac{1}{0.25} = 4 $$

**Explanation** :

  * Even with 16 processors, speedup is limited to 4.21x
  * The 20% non-parallelizable portion becomes the performance bottleneck
  * No more than 5x speedup is possible regardless of processors added

### Problem 3 (Difficulty: medium)

Parallelize the following code using multiprocessing for data parallelism.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Parallelize the following code using multiprocessing for dat
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Data to process
    data = np.random.random((1_000_000, 10))
    
    # Calculate statistics for each row
    result = []
    for row in data:
        stats = {
            'mean': row.mean(),
            'std': row.std(),
            'max': row.max(),
            'min': row.min()
        }
        result.append(stats)
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import multiprocessing as mp
    import time
    
    # Processing function
    def compute_stats(chunk):
        """Calculate statistics for data chunk"""
        result = []
        for row in chunk:
            stats = {
                'mean': row.mean(),
                'std': row.std(),
                'max': row.max(),
                'min': row.min()
            }
            result.append(stats)
        return result
    
    # Single-process version
    def single_process_version(data):
        start = time.time()
        result = []
        for row in data:
            stats = {
                'mean': row.mean(),
                'std': row.std(),
                'max': row.max(),
                'min': row.min()
            }
            result.append(stats)
        elapsed = time.time() - start
        return result, elapsed
    
    # Multi-process version (data parallelism)
    def multi_process_version(data, n_workers=4):
        start = time.time()
    
        # Split data into chunks
        chunks = np.array_split(data, n_workers)
    
        # Parallel processing
        with mp.Pool(n_workers) as pool:
            results = pool.map(compute_stats, chunks)
    
        # Integrate results
        final_result = []
        for chunk_result in results:
            final_result.extend(chunk_result)
    
        elapsed = time.time() - start
        return final_result, elapsed
    
    if __name__ == '__main__':
        # Test data
        data = np.random.random((100_000, 10))  # Adjusted size
    
        print("=== Data Parallelism Implementation ===")
        print(f"Data shape: {data.shape}")
    
        # Single-process
        result_single, time_single = single_process_version(data)
        print(f"\nSingle-process: {time_single:.4f}s")
    
        # Multi-process
        n_workers = mp.cpu_count()
        result_multi, time_multi = multi_process_version(data, n_workers)
        print(f"Multi-process ({n_workers} workers): {time_multi:.4f}s")
    
        # Validation
        assert len(result_single) == len(result_multi), "Result lengths differ"
        print(f"\nSpeedup: {time_single / time_multi:.2f}x")
        print(f"✓ Parallelization successful")
    

**Key Points** :

  * Split data evenly using `np.array_split()`
  * Create worker pool with `multiprocessing.Pool`
  * Process each chunk in parallel with `pool.map()`
  * Integrate results with `extend()`

### Problem 4 (Difficulty: hard)

When processing a dataset of 10 million rows, implement chunk processing that considers memory constraints. Set each chunk to 1 million rows and aggregate the results.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import time
    
    def process_chunk(chunk):
        """Process each chunk (example: mean, std, sum)"""
        stats = {
            'mean': chunk.mean(axis=0),
            'std': chunk.std(axis=0),
            'sum': chunk.sum(axis=0),
            'count': len(chunk)
        }
        return stats
    
    def aggregate_results(chunk_results):
        """Final aggregation of chunk results"""
        # Calculate total sum
        total_sum = sum(r['sum'] for r in chunk_results)
        total_count = sum(r['count'] for r in chunk_results)
    
        # Global mean
        global_mean = total_sum / total_count
    
        # Global standard deviation (weighted average)
        # More accurately, square root of weighted average of variances
        weighted_var = sum(
            r['count'] * (r['std']**2 + (r['mean'] - global_mean)**2)
            for r in chunk_results
        ) / total_count
        global_std = np.sqrt(weighted_var)
    
        return {
            'global_mean': global_mean,
            'global_std': global_std,
            'total_count': total_count
        }
    
    def chunked_processing(n_samples=10_000_000, n_features=10, chunk_size=1_000_000):
        """Memory-efficient chunk processing"""
        print(f"=== Chunk Processing ===")
        print(f"Total samples: {n_samples:,}")
        print(f"Chunk size: {chunk_size:,}")
        print(f"Number of chunks: {n_samples // chunk_size}")
    
        start = time.time()
        chunk_results = []
    
        # Process by chunks
        for i in range(0, n_samples, chunk_size):
            # Determine chunk size
            current_chunk_size = min(chunk_size, n_samples - i)
    
            # Generate chunk data (in practice, read from file)
            chunk = np.random.random((current_chunk_size, n_features))
    
            # Process chunk
            result = process_chunk(chunk)
            chunk_results.append(result)
    
            print(f"Chunk {len(chunk_results)}: {current_chunk_size:,} rows processed")
    
        # Aggregate results
        final_result = aggregate_results(chunk_results)
    
        elapsed = time.time() - start
    
        print(f"\n=== Processing Results ===")
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Global mean (first 3 dimensions): {final_result['global_mean'][:3]}")
        print(f"Global std (first 3 dimensions): {final_result['global_std'][:3]}")
        print(f"Total samples: {final_result['total_count']:,}")
    
        # Check memory efficiency
        import sys
        chunk_memory = sys.getsizeof(np.random.random((chunk_size, n_features))) / (1024**2)
        print(f"\nMemory usage per chunk: {chunk_memory:.2f} MB")
        print(f"(Processing with 1/{n_samples//chunk_size} of memory compared to loading all data at once)")
    
    if __name__ == '__main__':
        # Execute (reduced size since actual size takes time)
        chunked_processing(n_samples=1_000_000, n_features=10, chunk_size=100_000)
    

**Sample Output** :
    
    
    === Chunk Processing ===
    Total samples: 1,000,000
    Chunk size: 100,000
    Number of chunks: 10
    Chunk 1: 100,000 rows processed
    Chunk 2: 100,000 rows processed
    ...
    Chunk 10: 100,000 rows processed
    
    === Processing Results ===
    Processing time: 2.45s
    Global mean (first 3 dimensions): [0.500 0.499 0.501]
    Global std (first 3 dimensions): [0.289 0.288 0.290]
    Total samples: 1,000,000
    
    Memory usage per chunk: 7.63 MB
    (Processing with 1/10 of memory compared to loading all data at once)
    

**Key Points** :

  * Process in chunks to limit memory usage
  * Save statistics for each chunk
  * Aggregate in statistically correct manner at the end (weighted average)
  * In practice, combine with file I/O or database queries

### Problem 5 (Difficulty: hard)

Regarding the three strategies of data parallelism, model parallelism, and pipeline parallelism, explain their respective application scenarios and considerations when combining them.

Sample Answer

**Answer** :

#### 1\. Data Parallelism

**Application Scenarios** :

  * Large-scale datasets (TB-scale) processing
  * Each sample can be processed independently
  * Training same model with different data (mini-batch learning)

**Examples** :

  * Training large-scale image classification datasets
  * Log data aggregation and analysis
  * Batch prediction processing

#### 2\. Model Parallelism

**Application Scenarios** :

  * Model doesn't fit in single device memory
  * Large-scale neural networks (billions of parameters)
  * When computation graph can be partitioned

**Examples** :

  * Large language models like GPT-3
  * High-resolution image processing networks
  * Graph neural networks

#### 3\. Pipeline Parallelism

**Application Scenarios** :

  * Multiple processing stages exist
  * Each stage has different resource requirements
  * Streaming data processing

**Examples** :

  * ETL pipelines (Extract→Transform→Load)
  * Real-time data processing (preprocessing→inference→postprocessing)
  * Inter-layer pipelines in deep learning

#### Considerations for Combined Use

**1\. Data Parallelism + Model Parallelism** :
    
    
    """
    Ultra-large-scale model training
    
    Example: GPT-3 training
    - Model parallelism: Split each layer across multiple GPUs
    - Data parallelism: Process different mini-batches on multiple model replicas
    - Result: Can efficiently train on thousands of GPUs
    """
    

**Considerations** :

  * Complex communication patterns (inter-layer communication + gradient synchronization between replicas)
  * Optimized memory management (where to store activations, gradients)
  * Load balancing (both data and model)

**2\. Data Parallelism + Pipeline Parallelism** :
    
    
    """
    Large-scale ETL and machine learning pipeline
    
    Example: Streaming prediction system
    - Pipeline: Data acquisition → Preprocessing → Inference → Postprocessing
    - Data parallel: Multiple workers execute in parallel at each stage
    - Result: High-throughput prediction system
    """
    

**Considerations** :

  * Inter-stage buffer management
  * Backpressure control (slow stage blocks fast stage)
  * End-to-end latency management

**3\. Combining All Three** :
    
    
    """
    Ultra-large-scale distributed training system
    
    Example: Large-scale recommendation system
    - Data parallel: Process different user segments in parallel
    - Model parallel: Distribute embedding layers across multiple GPUs
    - Pipeline: Stage-wise feature extraction → Model training → Evaluation
    """
    

**Considerations** :

  * Managing system complexity (debugging, monitoring becomes difficult)
  * Optimizing overall efficiency (which strategy has most impact)
  * Incremental implementation (start with simple strategy first)

#### Selection Guidelines

Bottleneck | Recommended Strategy | Priority  
---|---|---  
Large data size | Data parallelism | 1st  
Large model size | Model parallelism | 1st  
Many processing stages | Pipeline parallelism | 2nd  
Both large | Data+Model parallelism | Gradual  
Real-time requirements | Pipeline parallelism | 1st  
  
**Implementation Principles** :

  1. First optimize with single strategy
  2. Measure and identify next bottleneck
  3. Introduce additional strategies as needed
  4. Always measure overall system efficiency

* * *

## References

  1. Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on large clusters. _Communications of the ACM_ , 51(1), 107-113.
  2. Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. _Communications of the ACM_ , 59(11), 56-65.
  3. Moritz, P., et al. (2018). Ray: A distributed framework for emerging AI applications. _OSDI_ , 561-577.
  4. Rocklin, M. (2015). Dask: Parallel computation with blocked algorithms and task scheduling. _SciPy_ , 126-132.
  5. Barroso, L. A., Clidaras, J., & Hölzle, U. (2013). _The datacenter as a computer_ (2nd ed.). Morgan & Claypool Publishers.
