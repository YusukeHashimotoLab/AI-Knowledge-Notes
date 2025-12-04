---
title: "Chapter 5: Large-Scale Machine Learning Pipelines"
chapter_title: "Chapter 5: Large-Scale Machine Learning Pipelines"
subtitle: Building End-to-End Distributed Machine Learning Systems
reading_time: 35-40 minutes
difficulty: Advanced
code_examples: 8
exercises: 5
---

This chapter covers Large. You will learn Build distributed ETL pipelines integrating Spark and Build end-to-end production-ready ML systems.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand design principles for large-scale machine learning pipelines
  * ✅ Build distributed ETL pipelines integrating Spark and PyTorch
  * ✅ Implement model training and hyperparameter optimization in distributed environments
  * ✅ Apply performance optimization techniques for large-scale pipelines
  * ✅ Build end-to-end production-ready ML systems
  * ✅ Implement monitoring and maintenance strategies

* * *

## 5.1 Pipeline Design Principles

### Requirements for Scalable ML Pipelines

Key considerations when designing large-scale machine learning pipelines:

Requirement | Description | Implementation Technology  
---|---|---  
**Data Scalability** | Processing TB to PB scale data | Spark, Dask, distributed storage  
**Compute Scalability** | Multi-node parallel processing | Ray, Horovod, Kubernetes  
**Fault Tolerance** | Automatic recovery on failure | Checkpointing, redundancy  
**Reproducibility** | Complete reproduction of experiments/models | Version control, seed fixing  
**Monitoring** | Real-time performance monitoring | Prometheus, Grafana, MLflow  
**Cost Efficiency** | Optimization of resource usage | Autoscaling, spot instances  
  
### Pipeline Architecture Patterns
    
    
    graph TB
        subgraph "Data Layer"
            A[Raw Data  
    HDFS/S3] --> B[Data Validation  
    Great Expectations]
            B --> C[Distributed ETL  
    Apache Spark]
        end
    
        subgraph "Feature Layer"
            C --> D[Feature Engineering  
    Spark ML]
            D --> E[Feature Store  
    Feast/Tecton]
        end
    
        subgraph "Training Layer"
            E --> F[Distributed Training  
    Ray/Horovod]
            F --> G[Hyperparameter Optimization  
    Ray Tune]
            G --> H[Model Registry  
    MLflow]
        end
    
        subgraph "Inference Layer"
            H --> I[Model Serving  
    Kubernetes]
            I --> J[Prediction Service  
    TorchServe]
        end
    
        subgraph "Monitoring Layer"
            J --> K[Metrics Collection  
    Prometheus]
            K --> L[Visualization  
    Grafana]
            L --> M[Alerting  
    PagerDuty]
        end
    

#### Code Example 1: Pipeline Configuration Class
    
    
    """
    Configuration management for large-scale ML pipelines
    """
    from dataclasses import dataclass, field
    from typing import Dict, List, Optional
    import yaml
    from pathlib import Path
    
    
    @dataclass
    class DataConfig:
        """Data processing configuration"""
        source_path: str
        output_path: str
        num_partitions: int = 1000
        file_format: str = "parquet"
        compression: str = "snappy"
        validation_rules: Dict[str, any] = field(default_factory=dict)
    
    
    @dataclass
    class TrainingConfig:
        """Training configuration"""
        model_type: str
        num_workers: int = 4
        num_gpus_per_worker: int = 1
        batch_size: int = 256
        max_epochs: int = 100
        learning_rate: float = 0.001
        checkpoint_freq: int = 10
        early_stopping_patience: int = 5
    
    
    @dataclass
    class ResourceConfig:
        """Resource configuration"""
        num_nodes: int = 4
        cpus_per_node: int = 16
        memory_per_node: str = "64GB"
        gpus_per_node: int = 4
        storage_type: str = "ssd"
        network_bandwidth: str = "10Gbps"
    
    
    @dataclass
    class MonitoringConfig:
        """Monitoring configuration"""
        metrics_interval: int = 60  # seconds
        log_level: str = "INFO"
        alert_thresholds: Dict[str, float] = field(default_factory=dict)
        dashboard_url: Optional[str] = None
    
    
    @dataclass
    class PipelineConfig:
        """Integrated pipeline configuration"""
        pipeline_name: str
        version: str
        data: DataConfig
        training: TrainingConfig
        resources: ResourceConfig
        monitoring: MonitoringConfig
    
        @classmethod
        def from_yaml(cls, config_path: Path) -> 'PipelineConfig':
            """Load configuration from YAML file"""
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
    
            return cls(
                pipeline_name=config_dict['pipeline_name'],
                version=config_dict['version'],
                data=DataConfig(**config_dict['data']),
                training=TrainingConfig(**config_dict['training']),
                resources=ResourceConfig(**config_dict['resources']),
                monitoring=MonitoringConfig(**config_dict['monitoring'])
            )
    
        def to_yaml(self, output_path: Path):
            """Save configuration to YAML file"""
            config_dict = {
                'pipeline_name': self.pipeline_name,
                'version': self.version,
                'data': self.data.__dict__,
                'training': self.training.__dict__,
                'resources': self.resources.__dict__,
                'monitoring': self.monitoring.__dict__
            }
    
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
    
    
    # Usage example
    if __name__ == "__main__":
        # Create configuration
        config = PipelineConfig(
            pipeline_name="customer_churn_prediction",
            version="v1.0.0",
            data=DataConfig(
                source_path="s3://my-bucket/raw-data/",
                output_path="s3://my-bucket/processed-data/",
                num_partitions=2000,
                validation_rules={
                    "min_records": 1000000,
                    "required_columns": ["customer_id", "features", "label"]
                }
            ),
            training=TrainingConfig(
                model_type="neural_network",
                num_workers=8,
                num_gpus_per_worker=2,
                batch_size=512,
                max_epochs=50
            ),
            resources=ResourceConfig(
                num_nodes=8,
                cpus_per_node=32,
                memory_per_node="128GB",
                gpus_per_node=4
            ),
            monitoring=MonitoringConfig(
                alert_thresholds={
                    "accuracy": 0.85,
                    "latency_p99": 100.0,  # milliseconds
                    "error_rate": 0.01
                }
            )
        )
    
        # Save configuration
        config.to_yaml(Path("pipeline_config.yaml"))
    
        # Load configuration
        loaded_config = PipelineConfig.from_yaml(Path("pipeline_config.yaml"))
        print(f"Pipeline: {loaded_config.pipeline_name}")
        print(f"Workers: {loaded_config.training.num_workers}")
        print(f"Partitions: {loaded_config.data.num_partitions}")
    

**💡 Design Points:**

  * Externalize configuration in YAML, not embedded in environment variables or code
  * Ensure type safety with dataclasses
  * Track configuration history with version control
  * Separate configuration files for each environment (dev/production)

### Error Handling and Retry Strategies

#### Code Example 2: Robust Pipeline Execution Framework
    
    
    """
    Fault-tolerant pipeline execution
    """
    import time
    import logging
    from typing import Callable, Any, Optional, List
    from functools import wraps
    from dataclasses import dataclass
    from enum import Enum
    
    
    class TaskStatus(Enum):
        """Task execution status"""
        PENDING = "pending"
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"
        RETRYING = "retrying"
    
    
    @dataclass
    class TaskResult:
        """Task execution result"""
        status: TaskStatus
        result: Any = None
        error: Optional[Exception] = None
        execution_time: float = 0.0
        retry_count: int = 0
    
    
    class RetryStrategy:
        """Retry strategy"""
    
        def __init__(
            self,
            max_retries: int = 3,
            initial_delay: float = 1.0,
            backoff_factor: float = 2.0,
            max_delay: float = 60.0,
            retryable_exceptions: List[type] = None
        ):
            self.max_retries = max_retries
            self.initial_delay = initial_delay
            self.backoff_factor = backoff_factor
            self.max_delay = max_delay
            self.retryable_exceptions = retryable_exceptions or [Exception]
    
        def get_delay(self, retry_count: int) -> float:
            """Calculate retry wait time (exponential backoff)"""
            delay = self.initial_delay * (self.backoff_factor ** retry_count)
            return min(delay, self.max_delay)
    
        def should_retry(self, exception: Exception) -> bool:
            """Determine whether to retry"""
            return any(isinstance(exception, exc_type)
                      for exc_type in self.retryable_exceptions)
    
    
    def with_retry(retry_strategy: RetryStrategy):
        """Decorator to add retry functionality"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> TaskResult:
                retry_count = 0
                start_time = time.time()
    
                while retry_count <= retry_strategy.max_retries:
                    try:
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
    
                        return TaskResult(
                            status=TaskStatus.SUCCESS,
                            result=result,
                            execution_time=execution_time,
                            retry_count=retry_count
                        )
    
                    except Exception as e:
                        if (retry_count < retry_strategy.max_retries and
                            retry_strategy.should_retry(e)):
    
                            delay = retry_strategy.get_delay(retry_count)
                            logging.warning(
                                f"Task failed (attempt {retry_count + 1}/"
                                f"{retry_strategy.max_retries + 1}): {str(e)}"
                                f"\nRetrying in {delay:.1f} seconds..."
                            )
    
                            time.sleep(delay)
                            retry_count += 1
                        else:
                            execution_time = time.time() - start_time
                            return TaskResult(
                                status=TaskStatus.FAILED,
                                error=e,
                                execution_time=execution_time,
                                retry_count=retry_count
                            )
    
                return TaskResult(
                    status=TaskStatus.FAILED,
                    error=Exception("Max retries exceeded"),
                    execution_time=time.time() - start_time,
                    retry_count=retry_count
                )
    
            return wrapper
        return decorator
    
    
    class PipelineExecutor:
        """Pipeline execution engine"""
    
        def __init__(self, checkpoint_dir: str = "./checkpoints"):
            self.checkpoint_dir = checkpoint_dir
            self.logger = logging.getLogger(__name__)
    
        def execute_stage(
            self,
            stage_name: str,
            task_func: Callable,
            retry_strategy: Optional[RetryStrategy] = None,
            checkpoint: bool = True
        ) -> TaskResult:
            """Execute pipeline stage"""
            self.logger.info(f"Starting stage: {stage_name}")
    
            # Restore from checkpoint if exists
            if checkpoint and self._checkpoint_exists(stage_name):
                self.logger.info(f"Restoring from checkpoint: {stage_name}")
                return self._load_checkpoint(stage_name)
    
            # Apply retry strategy
            if retry_strategy:
                task_func = with_retry(retry_strategy)(task_func)
    
            # Execute task
            result = task_func()
    
            # Save checkpoint
            if checkpoint and result.status == TaskStatus.SUCCESS:
                self._save_checkpoint(stage_name, result)
    
            self.logger.info(
                f"Stage {stage_name} completed: {result.status.value} "
                f"(time: {result.execution_time:.2f}s, "
                f"retries: {result.retry_count})"
            )
    
            return result
    
        def _checkpoint_exists(self, stage_name: str) -> bool:
            """Check checkpoint existence"""
            from pathlib import Path
            checkpoint_path = Path(self.checkpoint_dir) / f"{stage_name}.ckpt"
            return checkpoint_path.exists()
    
        def _save_checkpoint(self, stage_name: str, result: TaskResult):
            """Save checkpoint"""
            import pickle
            from pathlib import Path
    
            Path(self.checkpoint_dir).mkdir(exist_ok=True)
            checkpoint_path = Path(self.checkpoint_dir) / f"{stage_name}.ckpt"
    
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(result, f)
    
        def _load_checkpoint(self, stage_name: str) -> TaskResult:
            """Load checkpoint"""
            import pickle
            from pathlib import Path
    
            checkpoint_path = Path(self.checkpoint_dir) / f"{stage_name}.ckpt"
    
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
    
    
    # Usage example
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
    
        # Define retry strategy
        retry_strategy = RetryStrategy(
            max_retries=3,
            initial_delay=2.0,
            backoff_factor=2.0,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
    
        # Execute pipeline
        executor = PipelineExecutor(checkpoint_dir="./pipeline_checkpoints")
    
        # Stage 1: Data loading (may fail)
        def load_data():
            import random
            if random.random() < 0.3:  # 30% failure probability
                raise ConnectionError("Failed to connect to data source")
            return {"data": list(range(1000))}
    
        result1 = executor.execute_stage(
            "data_loading",
            load_data,
            retry_strategy=retry_strategy
        )
    
        if result1.status == TaskStatus.SUCCESS:
            print(f"Data loaded: {len(result1.result['data'])} records")
    
            # Stage 2: Data processing
            def process_data():
                data = result1.result['data']
                processed = [x * 2 for x in data]
                return {"processed_data": processed}
    
            result2 = executor.execute_stage(
                "data_processing",
                process_data,
                checkpoint=True
            )
    
            if result2.status == TaskStatus.SUCCESS:
                print(f"Processing completed successfully")
    

**⚠️ Retry Strategy Considerations:**

  * Idempotency: Design so that executing the same operation multiple times doesn't change the result
  * Timeout: Set maximum retry count to prevent infinite loops
  * Backoff: Increase wait time exponentially to prevent service overload
  * Selective retry: Retry only transient errors, fail immediately for persistent errors

* * *

## 5.2 Data Processing Pipeline

### Building Distributed ETL Pipelines

#### Code Example 3: Spark-based Large-Scale Data ETL
    
    
    """
    Large-scale data ETL pipeline with Apache Spark
    """
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from typing import List, Dict
    import logging
    
    
    class DistributedETLPipeline:
        """Distributed ETL pipeline"""
    
        def __init__(
            self,
            app_name: str = "MLDataPipeline",
            master: str = "spark://master:7077",
            num_partitions: int = 1000
        ):
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .master(master) \
                .config("spark.sql.shuffle.partitions", num_partitions) \
                .config("spark.default.parallelism", num_partitions) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
    
            self.logger = logging.getLogger(__name__)
    
        def extract(
            self,
            source_path: str,
            file_format: str = "parquet",
            schema: StructType = None
        ) -> DataFrame:
            """Data extraction"""
            self.logger.info(f"Extracting data from {source_path}")
    
            if schema:
                df = self.spark.read.schema(schema).format(file_format).load(source_path)
            else:
                df = self.spark.read.format(file_format).load(source_path)
    
            self.logger.info(f"Extracted {df.count()} records with {len(df.columns)} columns")
            return df
    
        def validate(self, df: DataFrame, validation_rules: Dict) -> DataFrame:
            """Data validation"""
            self.logger.info("Validating data quality")
    
            # Required columns check
            if "required_columns" in validation_rules:
                missing_cols = set(validation_rules["required_columns"]) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
    
            # Null value check
            if "non_null_columns" in validation_rules:
                for col in validation_rules["non_null_columns"]:
                    null_count = df.filter(F.col(col).isNull()).count()
                    if null_count > 0:
                        self.logger.warning(f"Column {col} has {null_count} null values")
    
            # Value range check
            if "value_ranges" in validation_rules:
                for col, (min_val, max_val) in validation_rules["value_ranges"].items():
                    df = df.filter(
                        (F.col(col) >= min_val) & (F.col(col) <= max_val)
                    )
    
            # Duplicate check
            if "unique_columns" in validation_rules:
                unique_cols = validation_rules["unique_columns"]
                initial_count = df.count()
                df = df.dropDuplicates(unique_cols)
                duplicates_removed = initial_count - df.count()
                if duplicates_removed > 0:
                    self.logger.warning(f"Removed {duplicates_removed} duplicate records")
    
            return df
    
        def transform(
            self,
            df: DataFrame,
            feature_columns: List[str],
            label_column: str = None,
            normalize: bool = True
        ) -> DataFrame:
            """Data transformation and feature engineering"""
            self.logger.info("Transforming data")
    
            # Handle missing values
            df = self._handle_missing_values(df, feature_columns)
    
            # Encode categorical features
            df = self._encode_categorical_features(df, feature_columns)
    
            # Create feature vector
            assembler = VectorAssembler(
                inputCols=feature_columns,
                outputCol="features_raw"
            )
            df = assembler.transform(df)
    
            # Normalization
            if normalize:
                scaler = StandardScaler(
                    inputCol="features_raw",
                    outputCol="features",
                    withMean=True,
                    withStd=True
                )
                scaler_model = scaler.fit(df)
                df = scaler_model.transform(df)
            else:
                df = df.withColumnRenamed("features_raw", "features")
    
            # Select only necessary columns
            select_cols = ["features"]
            if label_column:
                select_cols.append(label_column)
    
            return df.select(select_cols)
    
        def _handle_missing_values(
            self,
            df: DataFrame,
            columns: List[str],
            strategy: str = "mean"
        ) -> DataFrame:
            """Handle missing values"""
            for col in columns:
                if strategy == "mean":
                    mean_val = df.select(F.mean(col)).first()[0]
                    df = df.fillna({col: mean_val})
                elif strategy == "median":
                    median_val = df.approxQuantile(col, [0.5], 0.01)[0]
                    df = df.fillna({col: median_val})
                elif strategy == "drop":
                    df = df.dropna(subset=[col])
    
            return df
    
        def _encode_categorical_features(
            self,
            df: DataFrame,
            feature_columns: List[str]
        ) -> DataFrame:
            """Encode categorical variables"""
            from pyspark.ml.feature import StringIndexer, OneHotEncoder
    
            categorical_cols = [
                col for col in feature_columns
                if dict(df.dtypes)[col] == 'string'
            ]
    
            for col in categorical_cols:
                # StringIndexer
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_index",
                    handleInvalid="keep"
                )
                df = indexer.fit(df).transform(df)
    
                # OneHotEncoder
                encoder = OneHotEncoder(
                    inputCol=f"{col}_index",
                    outputCol=f"{col}_encoded"
                )
                df = encoder.fit(df).transform(df)
    
            return df
    
        def load(
            self,
            df: DataFrame,
            output_path: str,
            file_format: str = "parquet",
            mode: str = "overwrite",
            partition_by: List[str] = None
        ):
            """Data loading"""
            self.logger.info(f"Loading data to {output_path}")
    
            writer = df.write.mode(mode).format(file_format)
    
            if partition_by:
                writer = writer.partitionBy(partition_by)
    
            writer.save(output_path)
            self.logger.info(f"Data successfully loaded to {output_path}")
    
        def run_etl(
            self,
            source_path: str,
            output_path: str,
            feature_columns: List[str],
            label_column: str = None,
            validation_rules: Dict = None,
            partition_by: List[str] = None
        ):
            """Execute complete ETL pipeline"""
            # Extract
            df = self.extract(source_path)
    
            # Validate
            if validation_rules:
                df = self.validate(df, validation_rules)
    
            # Transform
            df = self.transform(df, feature_columns, label_column)
    
            # Load
            self.load(df, output_path, partition_by=partition_by)
    
            return df
    
    
    # Usage example
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
    
        # Initialize ETL pipeline
        pipeline = DistributedETLPipeline(
            app_name="CustomerChurnETL",
            num_partitions=2000
        )
    
        # Define validation rules
        validation_rules = {
            "required_columns": ["customer_id", "age", "balance", "churn"],
            "non_null_columns": ["customer_id", "churn"],
            "value_ranges": {
                "age": (18, 100),
                "balance": (0, 1000000)
            },
            "unique_columns": ["customer_id"]
        }
    
        # Execute ETL
        feature_columns = [
            "age", "balance", "num_products", "credit_score",
            "country", "gender", "is_active_member"
        ]
    
        pipeline.run_etl(
            source_path="s3://my-bucket/raw-data/customers/",
            output_path="s3://my-bucket/processed-data/customers/",
            feature_columns=feature_columns,
            label_column="churn",
            validation_rules=validation_rules,
            partition_by=["country"]
        )
    

**✅ Spark ETL Best Practices:**

  * **Adaptive Query Execution (AQE)** : Dynamically optimize query execution plans
  * **Partition Optimization** : Set appropriate number of partitions based on data size
  * **Caching** : Cache frequently used DataFrames in memory
  * **Avoid Schema Inference** : Use explicit schema definitions for large-scale data

* * *

## 5.3 Distributed Model Training

### Multi-Node Training and Hyperparameter Optimization

#### Code Example 4: Distributed Hyperparameter Optimization with Ray Tune
    
    
    """
    Large-scale hyperparameter optimization using Ray Tune
    """
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from typing import Dict
    
    
    class NeuralNetwork(nn.Module):
        """Simple neural network"""
    
        def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float):
            super().__init__()
    
            layers = []
            prev_dim = input_dim
    
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
    
            layers.append(nn.Linear(prev_dim, output_dim))
    
            self.network = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.network(x)
    
    
    def train_model(config: Dict, checkpoint_dir=None):
        """
        Training function (called by Ray Tune)
    
        Args:
            config: Hyperparameter configuration
            checkpoint_dir: Checkpoint directory
        """
        # Prepare data (in practice, load from Spark)
        np.random.seed(42)
        X_train = np.random.randn(10000, 50).astype(np.float32)
        y_train = np.random.randint(0, 2, 10000).astype(np.int64)
        X_val = np.random.randn(2000, 50).astype(np.float32)
        y_val = np.random.randint(0, 2, 2000).astype(np.int64)
    
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=2
        )
    
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralNetwork(
            input_dim=50,
            hidden_dims=config["hidden_dims"],
            output_dim=2,
            dropout=config["dropout"]
        ).to(device)
    
        # Optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
    
        # Restore from checkpoint
        if checkpoint_dir:
            checkpoint = torch.load(checkpoint_dir + "/checkpoint.pt")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0
    
        # Training loop
        for epoch in range(start_epoch, config["max_epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
    
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
    
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
    
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
    
            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
    
            # Save checkpoint
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, checkpoint_dir + "/checkpoint.pt")
    
            # Report metrics to Ray Tune
            tune.report(
                train_loss=train_loss / len(train_loader),
                train_accuracy=train_acc,
                val_loss=val_loss / len(val_loader),
                val_accuracy=val_acc
            )
    
    
    def run_hyperparameter_optimization():
        """Execute distributed hyperparameter optimization"""
    
        # Initialize Ray
        ray.init(
            address="auto",  # Connect to existing Ray cluster
            _redis_password="password"
        )
    
        # Define search space
        config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([128, 256, 512]),
            "hidden_dims": tune.choice([
                [128, 64],
                [256, 128],
                [512, 256, 128]
            ]),
            "dropout": tune.uniform(0.1, 0.5),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "max_epochs": 50
        }
    
        # Scheduler (early stopping)
        scheduler = ASHAScheduler(
            metric="val_accuracy",
            mode="max",
            max_t=50,
            grace_period=10,
            reduction_factor=2
        )
    
        # Search algorithm (Optuna)
        search_alg = OptunaSearch(
            metric="val_accuracy",
            mode="max"
        )
    
        # Reporter configuration
        reporter = CLIReporter(
            metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"],
            max_progress_rows=20
        )
    
        # Run tuning
        analysis = tune.run(
            train_model,
            config=config,
            num_samples=100,  # Number of trials
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=reporter,
            resources_per_trial={
                "cpu": 4,
                "gpu": 1
            },
            checkpoint_at_end=True,
            checkpoint_freq=10,
            local_dir="./ray_results",
            name="neural_network_hpo"
        )
    
        # Get best configuration
        best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
        best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max")
    
        print("\n" + "="*80)
        print("Best Hyperparameters:")
        print("="*80)
        for key, value in best_config.items():
            print(f"{key:20s}: {value}")
    
        print(f"\nBest Validation Accuracy: {best_trial.last_result['val_accuracy']:.4f}")
    
        # Get results as DataFrame
        df = analysis.dataframe()
        df.to_csv("hpo_results.csv", index=False)
    
        ray.shutdown()
    
        return best_config, analysis
    
    
    # Usage example
    if __name__ == "__main__":
        best_config, analysis = run_hyperparameter_optimization()
    

### Distributed Training Strategies

#### Code Example 5: PyTorch Distributed Data Parallel (DDP)
    
    
    """
    Multi-node distributed training with PyTorch DDP
    """
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    import os
    from typing import Optional
    
    
    def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
        """
        Setup distributed environment
    
        Args:
            rank: Current process rank
            world_size: Total number of processes
            backend: Communication backend (nccl, gloo, mpi)
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        # Initialize process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
        # Set GPU device
        torch.cuda.set_device(rank)
    
    
    def cleanup_distributed():
        """Cleanup distributed environment"""
        dist.destroy_process_group()
    
    
    class DistributedTrainer:
        """Distributed training manager"""
    
        def __init__(
            self,
            model: nn.Module,
            train_dataset,
            val_dataset,
            rank: int,
            world_size: int,
            batch_size: int = 256,
            learning_rate: float = 0.001,
            checkpoint_dir: str = "./checkpoints"
        ):
            self.rank = rank
            self.world_size = world_size
            self.checkpoint_dir = checkpoint_dir
    
            # Device setup
            self.device = torch.device(f"cuda:{rank}")
    
            # Wrap model with DDP
            self.model = model.to(self.device)
            self.model = DDP(self.model, device_ids=[rank])
    
            # Data loaders (using DistributedSampler)
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
    
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
    
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
    
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )
    
            # Optimizer and loss function
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
            self.criterion = nn.CrossEntropyLoss()
    
        def train_epoch(self, epoch: int):
            """Train for one epoch"""
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)  # Set shuffle seed
    
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
    
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
    
                # Backward pass
                loss.backward()
                self.optimizer.step()
    
                # Calculate metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
    
                if batch_idx % 100 == 0 and self.rank == 0:
                    print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {loss.item():.4f}")
    
            # Aggregate metrics across all processes
            avg_loss = self._aggregate_metric(total_loss / len(self.train_loader))
            accuracy = self._aggregate_metric(total_correct / total_samples)
    
            return avg_loss, accuracy
    
        def validate(self):
            """Validation"""
            self.model.eval()
    
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
    
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(labels).sum().item()
                    total_samples += labels.size(0)
    
            # Aggregate metrics across all processes
            avg_loss = self._aggregate_metric(total_loss / len(self.val_loader))
            accuracy = self._aggregate_metric(total_correct / total_samples)
    
            return avg_loss, accuracy
    
        def _aggregate_metric(self, value: float) -> float:
            """Aggregate metric across all processes"""
            tensor = torch.tensor(value).to(self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return (tensor / self.world_size).item()
    
        def save_checkpoint(self, epoch: int, val_accuracy: float):
            """Save checkpoint (rank 0 only)"""
            if self.rank == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pt"
                )
    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, checkpoint_path)
    
                print(f"Checkpoint saved: {checkpoint_path}")
    
        def load_checkpoint(self, checkpoint_path: str):
            """Load checkpoint"""
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
    
        def train(self, num_epochs: int, save_freq: int = 10):
            """Complete training loop"""
            for epoch in range(num_epochs):
                # Training
                train_loss, train_acc = self.train_epoch(epoch)
    
                # Validation
                val_loss, val_acc = self.validate()
    
                # Log output (rank 0 only)
                if self.rank == 0:
                    print(f"\nEpoch {epoch + 1}/{num_epochs}")
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
                # Save checkpoint
                if (epoch + 1) % save_freq == 0:
                    self.save_checkpoint(epoch + 1, val_acc)
    
    
    def distributed_training_worker(
        rank: int,
        world_size: int,
        model_class,
        train_dataset,
        val_dataset
    ):
        """Distributed training worker function"""
        # Setup distributed environment
        setup_distributed(rank, world_size)
    
        # Create model
        model = model_class(input_dim=50, hidden_dims=[256, 128], output_dim=2, dropout=0.3)
    
        # Initialize trainer
        trainer = DistributedTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            rank=rank,
            world_size=world_size,
            batch_size=256,
            learning_rate=0.001
        )
    
        # Execute training
        trainer.train(num_epochs=50, save_freq=10)
    
        # Cleanup
        cleanup_distributed()
    
    
    # Usage example
    if __name__ == "__main__":
        import torch.multiprocessing as mp
        from torch.utils.data import TensorDataset
        import numpy as np
    
        # Create dummy data
        np.random.seed(42)
        X_train = torch.from_numpy(np.random.randn(100000, 50).astype(np.float32))
        y_train = torch.from_numpy(np.random.randint(0, 2, 100000).astype(np.int64))
        X_val = torch.from_numpy(np.random.randn(20000, 50).astype(np.float32))
        y_val = torch.from_numpy(np.random.randint(0, 2, 20000).astype(np.int64))
    
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
    
        # Launch distributed training (4 GPUs)
        world_size = 4
        mp.spawn(
            distributed_training_worker,
            args=(world_size, NeuralNetwork, train_dataset, val_dataset),
            nprocs=world_size,
            join=True
        )
    

**💡 Distributed Training Selection Criteria:**

  * **Data Parallel (DP)** : Single node, multiple GPUs (simple but slower)
  * **Distributed Data Parallel (DDP)** : Multi-node, efficient gradient synchronization
  * **Fully Sharded Data Parallel (FSDP)** : Ultra-large models (GPT-3 class)
  * **Model Parallel** : Huge models that don't fit on a single GPU

* * *

## 5.4 Performance Optimization

### Profiling and Bottleneck Identification

#### Code Example 6: Profiling Distributed Systems
    
    
    """
    Profiling and optimization of distributed machine learning pipelines
    """
    import time
    import psutil
    import torch
    from contextlib import contextmanager
    from typing import Dict, List
    import json
    from dataclasses import dataclass, asdict
    import numpy as np
    
    
    @dataclass
    class ProfileMetrics:
        """Profiling metrics"""
        stage_name: str
        execution_time: float
        cpu_percent: float
        memory_mb: float
        gpu_memory_mb: float = 0.0
        io_read_mb: float = 0.0
        io_write_mb: float = 0.0
        network_sent_mb: float = 0.0
        network_recv_mb: float = 0.0
    
    
    class PerformanceProfiler:
        """Performance profiler"""
    
        def __init__(self, enable_gpu: bool = True):
            self.enable_gpu = enable_gpu and torch.cuda.is_available()
            self.metrics: List[ProfileMetrics] = []
            self.process = psutil.Process()
    
        @contextmanager
        def profile(self, stage_name: str):
            """Profile stage"""
            # Start metrics
            start_time = time.time()
            start_cpu = self.process.cpu_percent()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
            io_start = self.process.io_counters()
            net_start = psutil.net_io_counters()
    
            if self.enable_gpu:
                torch.cuda.reset_peak_memory_stats()
                start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
            try:
                yield
            finally:
                # End metrics
                end_time = time.time()
                end_cpu = self.process.cpu_percent()
                end_memory = self.process.memory_info().rss / 1024 / 1024
    
                io_end = self.process.io_counters()
                net_end = psutil.net_io_counters()
    
                # GPU memory
                if self.enable_gpu:
                    end_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    end_gpu_memory = 0.0
    
                # Record metrics
                metrics = ProfileMetrics(
                    stage_name=stage_name,
                    execution_time=end_time - start_time,
                    cpu_percent=(start_cpu + end_cpu) / 2,
                    memory_mb=end_memory - start_memory,
                    gpu_memory_mb=end_gpu_memory - start_gpu_memory if self.enable_gpu else 0.0,
                    io_read_mb=(io_end.read_bytes - io_start.read_bytes) / 1024 / 1024,
                    io_write_mb=(io_end.write_bytes - io_start.write_bytes) / 1024 / 1024,
                    network_sent_mb=(net_end.bytes_sent - net_start.bytes_sent) / 1024 / 1024,
                    network_recv_mb=(net_end.bytes_recv - net_start.bytes_recv) / 1024 / 1024
                )
    
                self.metrics.append(metrics)
    
        def print_summary(self):
            """Display profiling results summary"""
            print("\n" + "="*100)
            print("Performance Profiling Summary")
            print("="*100)
            print(f"{'Stage':<30} {'Time (s)':<12} {'CPU %':<10} {'Mem (MB)':<12} "
                  f"{'GPU (MB)':<12} {'I/O Read':<12} {'I/O Write':<12}")
            print("-"*100)
    
            total_time = 0.0
            for m in self.metrics:
                print(f"{m.stage_name:<30} {m.execution_time:<12.2f} {m.cpu_percent:<10.1f} "
                      f"{m.memory_mb:<12.1f} {m.gpu_memory_mb:<12.1f} "
                      f"{m.io_read_mb:<12.1f} {m.io_write_mb:<12.1f}")
                total_time += m.execution_time
    
            print("-"*100)
            print(f"{'Total':<30} {total_time:<12.2f}")
            print("="*100)
    
        def get_bottlenecks(self, top_k: int = 3) -> List[ProfileMetrics]:
            """Identify bottlenecks"""
            sorted_metrics = sorted(
                self.metrics,
                key=lambda m: m.execution_time,
                reverse=True
            )
            return sorted_metrics[:top_k]
    
        def export_json(self, output_path: str):
            """Export results to JSON"""
            data = [asdict(m) for m in self.metrics]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    
    class DataLoaderOptimizer:
        """DataLoader optimization helper"""
    
        @staticmethod
        def benchmark_dataloader(
            dataset,
            batch_sizes: List[int],
            num_workers_list: List[int],
            num_iterations: int = 100
        ) -> Dict:
            """Optimize DataLoader configuration"""
            results = []
    
            for batch_size in batch_sizes:
                for num_workers in num_workers_list:
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True
                    )
    
                    start_time = time.time()
                    for i, batch in enumerate(loader):
                        if i >= num_iterations:
                            break
                        _ = batch  # Load data
                    elapsed = time.time() - start_time
    
                    throughput = (batch_size * num_iterations) / elapsed
    
                    results.append({
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'throughput': throughput,
                        'time_per_batch': elapsed / num_iterations
                    })
    
            # Find optimal configuration
            best_config = max(results, key=lambda x: x['throughput'])
    
            print("\nDataLoader Optimization Results:")
            print(f"Best Configuration: batch_size={best_config['batch_size']}, "
                  f"num_workers={best_config['num_workers']}")
            print(f"Throughput: {best_config['throughput']:.2f} samples/sec")
    
            return best_config
    
    
    class GPUOptimizer:
        """GPU optimization helper"""
    
        @staticmethod
        def optimize_memory():
            """Optimize GPU memory"""
            if torch.cuda.is_available():
                # Clear unused cache
                torch.cuda.empty_cache()
    
                # Display memory statistics
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
    
                print(f"\nGPU Memory Status:")
                print(f"Allocated: {allocated:.2f} GB")
                print(f"Reserved: {reserved:.2f} GB")
    
        @staticmethod
        def enable_auto_mixed_precision():
            """Enable automatic mixed precision training"""
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    
        @staticmethod
        def benchmark_precision(model, sample_input, num_iterations: int = 100):
            """Compare performance by precision"""
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            sample_input = sample_input.to(device)
    
            results = {}
    
            # FP32
            model.float()
            start = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(sample_input.float())
            torch.cuda.synchronize()
            results['fp32'] = time.time() - start
    
            # FP16 (if available)
            if GPUOptimizer.enable_auto_mixed_precision():
                model.half()
                start = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(sample_input.half())
                torch.cuda.synchronize()
                results['fp16'] = time.time() - start
    
            print("\nPrecision Benchmark:")
            for precision, elapsed in results.items():
                print(f"{precision.upper()}: {elapsed:.4f}s "
                      f"({num_iterations/elapsed:.2f} iter/s)")
    
            return results
    
    
    # Usage example
    if __name__ == "__main__":
        # Initialize profiler
        profiler = PerformanceProfiler(enable_gpu=True)
    
        # Profile data preparation
        with profiler.profile("Data Loading"):
            data = torch.randn(100000, 50)
            time.sleep(0.5)  # Simulation
    
        # Profile preprocessing
        with profiler.profile("Preprocessing"):
            normalized_data = (data - data.mean()) / data.std()
            time.sleep(0.3)
    
        # Profile model training
        with profiler.profile("Model Training"):
            model = torch.nn.Linear(50, 10)
            optimizer = torch.optim.Adam(model.parameters())
    
            for _ in range(100):
                outputs = model(normalized_data)
                loss = outputs.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
        # Display results
        profiler.print_summary()
    
        # Identify bottlenecks
        bottlenecks = profiler.get_bottlenecks(top_k=2)
        print("\nTop Bottlenecks:")
        for i, m in enumerate(bottlenecks, 1):
            print(f"{i}. {m.stage_name}: {m.execution_time:.2f}s")
    
        # Export results
        profiler.export_json("profiling_results.json")
    
        # Optimize DataLoader
        dataset = torch.utils.data.TensorDataset(data, torch.zeros(len(data)))
        DataLoaderOptimizer.benchmark_dataloader(
            dataset,
            batch_sizes=[128, 256, 512],
            num_workers_list=[2, 4, 8],
            num_iterations=50
        )
    
        # GPU optimization
        GPUOptimizer.optimize_memory()
        if torch.cuda.is_available():
            sample_model = torch.nn.Sequential(
                torch.nn.Linear(50, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )
            sample_input = torch.randn(32, 50)
            GPUOptimizer.benchmark_precision(sample_model, sample_input)
    

### I/O Optimization and Network Efficiency

**💡 I/O Optimization Best Practices:**

  * **Data Format** : Parquet (columnar) is faster than row-oriented formats (CSV)
  * **Compression** : Snappy compression balances read speed and storage efficiency
  * **Prefetching** : DataLoader's `num_workers` for background loading
  * **Memory Mapping** : Efficiently access large files with memory mapping
  * **Sharding** : Split data into multiple files for parallel reading

* * *

## 5.5 End-to-End Implementation Example

### Complete Large-Scale ML Pipeline

#### Code Example 7: Integrated Spark + PyTorch Pipeline
    
    
    """
    Large-scale ML pipeline integrating Spark and PyTorch
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from typing import List, Tuple
    import pickle
    
    
    class SparkDatasetConverter:
        """Convert Spark DataFrame to PyTorch Dataset"""
    
        @staticmethod
        def spark_to_pytorch(
            spark_df,
            feature_column: str = "features",
            label_column: str = "label",
            output_path: str = "/tmp/pytorch_data"
        ):
            """
            Save Spark DataFrame in PyTorch-loadable format
    
            Args:
                spark_df: Spark DataFrame
                feature_column: Feature column name
                label_column: Label column name
                output_path: Output path
            """
            # Convert via Pandas (small to medium data)
            # For large-scale data, process by partition
    
            def convert_partition(iterator):
                """Convert each partition"""
                data_list = []
                for row in iterator:
                    features = row[feature_column].toArray()
                    label = row[label_column]
                    data_list.append((features, label))
    
                # Save file per partition
                import random
                partition_id = random.randint(0, 10000)
                output_file = f"{output_path}/partition_{partition_id}.pkl"
    
                with open(output_file, 'wb') as f:
                    pickle.dump(data_list, f)
    
                yield (output_file, len(data_list))
    
            # Process each partition
            spark_df.rdd.mapPartitions(convert_partition).collect()
    
    
    class DistributedDataset(Dataset):
        """Dataset for loading distributedly stored data"""
    
        def __init__(self, data_dir: str):
            from pathlib import Path
    
            self.data_files = list(Path(data_dir).glob("partition_*.pkl"))
    
            # Build index for all data
            self.file_indices = []
            for file_path in self.data_files:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.file_indices.append((file_path, len(data)))
    
            self.total_samples = sum(count for _, count in self.file_indices)
    
            # Cache (if memory available)
            self.cache = {}
    
        def __len__(self):
            return self.total_samples
    
        def __getitem__(self, idx):
            # Calculate which file and which index
            file_idx = 0
            cumsum = 0
    
            for i, (_, count) in enumerate(self.file_indices):
                if idx < cumsum + count:
                    file_idx = i
                    local_idx = idx - cumsum
                    break
                cumsum += count
    
            # Load data from file
            file_path = self.file_indices[file_idx][0]
    
            if file_path not in self.cache:
                with open(file_path, 'rb') as f:
                    self.cache[file_path] = pickle.load(f)
    
            features, label = self.cache[file_path][local_idx]
    
            return torch.FloatTensor(features), torch.LongTensor([label])[0]
    
    
    class EndToEndMLPipeline:
        """End-to-end ML pipeline"""
    
        def __init__(
            self,
            spark_master: str = "local[*]",
            app_name: str = "EndToEndML"
        ):
            # Initialize Spark
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .master(spark_master) \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
    
            self.profiler = PerformanceProfiler()
    
        def run_pipeline(
            self,
            data_path: str,
            model_class,
            model_params: dict,
            training_config: dict,
            output_dir: str = "./pipeline_output"
        ):
            """Execute complete pipeline"""
    
            # Step 1: Data loading (Spark)
            with self.profiler.profile("1. Data Loading (Spark)"):
                raw_df = self.spark.read.parquet(data_path)
                print(f"Loaded {raw_df.count()} records")
    
            # Step 2: Data validation
            with self.profiler.profile("2. Data Validation"):
                # Basic statistics
                raw_df.describe().show()
    
                # Null value check
                null_counts = raw_df.select([
                    F.count(F.when(F.col(c).isNull(), c)).alias(c)
                    for c in raw_df.columns
                ])
                null_counts.show()
    
            # Step 3: Feature engineering (Spark)
            with self.profiler.profile("3. Feature Engineering (Spark)"):
                from pyspark.ml.feature import VectorAssembler, StandardScaler
    
                feature_cols = [c for c in raw_df.columns if c != 'label']
    
                assembler = VectorAssembler(
                    inputCols=feature_cols,
                    outputCol="features_raw"
                )
                df_assembled = assembler.transform(raw_df)
    
                scaler = StandardScaler(
                    inputCol="features_raw",
                    outputCol="features",
                    withMean=True,
                    withStd=True
                )
                scaler_model = scaler.fit(df_assembled)
                df_scaled = scaler_model.transform(df_assembled)
    
                # Train/test split
                train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
    
            # Step 4: Convert to PyTorch data
            with self.profiler.profile("4. Spark to PyTorch Conversion"):
                train_data_dir = f"{output_dir}/train_data"
                test_data_dir = f"{output_dir}/test_data"
    
                SparkDatasetConverter.spark_to_pytorch(
                    train_df.select("features", "label"),
                    output_path=train_data_dir
                )
                SparkDatasetConverter.spark_to_pytorch(
                    test_df.select("features", "label"),
                    output_path=test_data_dir
                )
    
            # Step 5: Create PyTorch Dataset/DataLoader
            with self.profiler.profile("5. PyTorch DataLoader Setup"):
                train_dataset = DistributedDataset(train_data_dir)
                test_dataset = DistributedDataset(test_data_dir)
    
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=training_config['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=training_config['batch_size'],
                    num_workers=4
                )
    
            # Step 6: Model training
            with self.profiler.profile("6. Model Training"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model_class(**model_params).to(device)
    
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=training_config['learning_rate']
                )
                criterion = nn.CrossEntropyLoss()
    
                # Training loop
                for epoch in range(training_config['num_epochs']):
                    model.train()
                    train_loss = 0.0
    
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
    
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
    
                        train_loss += loss.item()
    
                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.4f}")
    
            # Step 7: Model evaluation
            with self.profiler.profile("7. Model Evaluation"):
                model.eval()
                correct = 0
                total = 0
    
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
    
                accuracy = correct / total
                print(f"\nTest Accuracy: {accuracy:.4f}")
    
            # Step 8: Model saving
            with self.profiler.profile("8. Model Saving"):
                model_path = f"{output_dir}/model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_params': model_params,
                    'accuracy': accuracy
                }, model_path)
                print(f"Model saved to {model_path}")
    
            # Profiling results
            self.profiler.print_summary()
            self.profiler.export_json(f"{output_dir}/profiling.json")
    
            return model, accuracy
    
    
    # Usage example
    if __name__ == "__main__":
        # Execute pipeline
        pipeline = EndToEndMLPipeline(
            spark_master="spark://master:7077",
            app_name="CustomerChurnPrediction"
        )
    
        # Define model
        class ChurnPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, output_dim))
                self.network = nn.Sequential(*layers)
    
            def forward(self, x):
                return self.network(x)
    
        # Execute pipeline
        model, accuracy = pipeline.run_pipeline(
            data_path="s3://my-bucket/customer-data/",
            model_class=ChurnPredictor,
            model_params={
                'input_dim': 50,
                'hidden_dims': [256, 128, 64],
                'output_dim': 2
            },
            training_config={
                'batch_size': 512,
                'learning_rate': 0.001,
                'num_epochs': 50
            },
            output_dir="./churn_prediction_output"
        )
    

### Monitoring and Maintenance

#### Code Example 8: Real-time Monitoring System
    
    
    """
    Monitoring and alerting system for ML pipelines
    """
    import time
    import threading
    from dataclasses import dataclass
    from typing import Dict, List, Callable, Optional
    from datetime import datetime
    import json
    
    
    @dataclass
    class Metric:
        """Metric"""
        name: str
        value: float
        timestamp: datetime
        tags: Dict[str, str] = None
    
    
    class MetricsCollector:
        """Metrics collection"""
    
        def __init__(self):
            self.metrics: List[Metric] = []
            self.lock = threading.Lock()
    
        def record(self, name: str, value: float, tags: Dict[str, str] = None):
            """Record metric"""
            with self.lock:
                metric = Metric(
                    name=name,
                    value=value,
                    timestamp=datetime.now(),
                    tags=tags or {}
                )
                self.metrics.append(metric)
    
        def get_latest(self, name: str, n: int = 1) -> List[Metric]:
            """Get latest metrics"""
            with self.lock:
                filtered = [m for m in self.metrics if m.name == name]
                return sorted(filtered, key=lambda m: m.timestamp, reverse=True)[:n]
    
        def get_average(self, name: str, window_seconds: int = 60) -> Optional[float]:
            """Average value within time window"""
            now = datetime.now()
            with self.lock:
                recent = [
                    m for m in self.metrics
                    if m.name == name and (now - m.timestamp).total_seconds() <= window_seconds
                ]
                if not recent:
                    return None
                return sum(m.value for m in recent) / len(recent)
    
    
    class AlertRule:
        """Alert rule"""
    
        def __init__(
            self,
            name: str,
            metric_name: str,
            threshold: float,
            comparison: str = "greater",  # greater, less, equal
            window_seconds: int = 60,
            callback: Callable = None
        ):
            self.name = name
            self.metric_name = metric_name
            self.threshold = threshold
            self.comparison = comparison
            self.window_seconds = window_seconds
            self.callback = callback or self.default_callback
    
        def check(self, collector: MetricsCollector) -> bool:
            """Check rule"""
            avg_value = collector.get_average(self.metric_name, self.window_seconds)
    
            if avg_value is None:
                return False
    
            if self.comparison == "greater":
                triggered = avg_value > self.threshold
            elif self.comparison == "less":
                triggered = avg_value < self.threshold
            else:  # equal
                triggered = abs(avg_value - self.threshold) < 0.0001
    
            if triggered:
                self.callback(self.name, self.metric_name, avg_value, self.threshold)
    
            return triggered
    
        def default_callback(self, rule_name, metric_name, value, threshold):
            """Default alert callback"""
            print(f"\n⚠️  ALERT: {rule_name}")
            print(f"   Metric: {metric_name} = {value:.4f}")
            print(f"   Threshold: {threshold:.4f}")
            print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    class MonitoringSystem:
        """Integrated monitoring system"""
    
        def __init__(self, check_interval: int = 10):
            self.collector = MetricsCollector()
            self.alert_rules: List[AlertRule] = []
            self.check_interval = check_interval
            self.running = False
            self.monitor_thread = None
    
        def add_alert_rule(self, rule: AlertRule):
            """Add alert rule"""
            self.alert_rules.append(rule)
    
        def start(self):
            """Start monitoring"""
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Monitoring system started")
    
        def stop(self):
            """Stop monitoring"""
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join()
            print("Monitoring system stopped")
    
        def _monitor_loop(self):
            """Monitoring loop"""
            while self.running:
                for rule in self.alert_rules:
                    rule.check(self.collector)
                time.sleep(self.check_interval)
    
        def export_metrics(self, output_path: str):
            """Export metrics"""
            with self.collector.lock:
                data = [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'tags': m.tags
                    }
                    for m in self.collector.metrics
                ]
    
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    
    # Usage example
    if __name__ == "__main__":
        # Initialize monitoring system
        monitor = MonitoringSystem(check_interval=5)
    
        # Setup alert rules
        monitor.add_alert_rule(AlertRule(
            name="High Training Loss",
            metric_name="train_loss",
            threshold=0.5,
            comparison="greater",
            window_seconds=30
        ))
    
        monitor.add_alert_rule(AlertRule(
            name="Low Validation Accuracy",
            metric_name="val_accuracy",
            threshold=0.80,
            comparison="less",
            window_seconds=60
        ))
    
        monitor.add_alert_rule(AlertRule(
            name="High GPU Memory Usage",
            metric_name="gpu_memory_percent",
            threshold=90.0,
            comparison="greater",
            window_seconds=30
        ))
    
        # Start monitoring
        monitor.start()
    
        # Simulation: Record metrics
        try:
            for i in range(50):
                # Training metrics (gradually improving)
                train_loss = 1.0 / (i + 1) + 0.1
                val_acc = min(0.95, 0.5 + i * 0.01)
                gpu_mem = 70 + (i % 5) * 5
    
                monitor.collector.record("train_loss", train_loss)
                monitor.collector.record("val_accuracy", val_acc)
                monitor.collector.record("gpu_memory_percent", gpu_mem)
    
                # Inject anomalous values occasionally
                if i == 20:
                    monitor.collector.record("train_loss", 0.8)  # Trigger alert
                if i == 30:
                    monitor.collector.record("val_accuracy", 0.75)  # Trigger alert
                if i == 40:
                    monitor.collector.record("gpu_memory_percent", 95.0)  # Trigger alert
    
                time.sleep(1)
    
        except KeyboardInterrupt:
            pass
        finally:
            # Stop monitoring and export metrics
            monitor.stop()
            monitor.export_metrics("monitoring_metrics.json")
            print("\nMetrics exported to monitoring_metrics.json")
    

**✅ Production Environment Monitoring Items:**

  * **Model Performance** : Accuracy, F1 score, AUC-ROC, inference latency
  * **Data Quality** : Missing rate, anomaly rate, data drift detection
  * **System Resources** : CPU/GPU utilization, memory usage, disk I/O
  * **Availability** : Uptime, error rate, request throughput
  * **Cost** : Cloud resource usage, inference cost per request

* * *

## Exercises

### Exercise 1: Pipeline Design

**Problem:** Design a recommendation system that generates 10TB of new data per day. Propose a pipeline architecture that meets the following requirements:

  * Data ingestion to recommendation generation within 4 hours
  * 99.9% availability
  * A/B testing capability
  * Automatic model retraining

**Hints:**

  * Consider incremental learning
  * Feature store for data reuse
  * Multi-armed bandit for A/B testing
  * Automatic monitoring and trigger for model performance

### Exercise 2: Distributed Training Optimization

**Problem:** Train an image classification model on a cluster with 8 nodes x 4 GPUs (32 GPUs total). Implement the following optimizations:

  * Efficient gradient synchronization strategy
  * Data loader performance tuning
  * Mixed precision training application
  * Checkpoint strategy

**Expected Improvements:**

  * Reduce training time to 25x faster than single GPU
  * Maintain GPU utilization above 90%
  * Efficient network bandwidth utilization

### Exercise 3: Cost Optimization

**Problem:** An ML pipeline has monthly cloud costs of $50,000. Propose strategies to reduce costs by 30% while maintaining model performance.

**Elements to Consider:**

  * Leveraging spot instances
  * Autoscaling configuration
  * Storage tiering (Hot/Cold data)
  * Compute resource optimization
  * Reducing unnecessary pipeline executions

### Exercise 4: Data Drift Detection

**Problem:** Implement a system to automatically detect data drift in production. Include the following:

  * Statistical tests (KS test, chi-square test)
  * Distribution visualization
  * Alert threshold configuration
  * Automatic retraining trigger

**Implementation Example:**
    
    
    class DataDriftDetector:
        def detect_drift(self, reference_data, current_data):
            # Implement KS test
            pass
    
        def visualize_distributions(self, feature_name):
            # Plot distribution comparison
            pass
    
        def trigger_retraining(self, drift_score, threshold=0.05):
            # Trigger retraining
            pass
    

### Exercise 5: End-to-End Pipeline Implementation

**Problem:** Implement a complete ML pipeline that meets the following requirements:

**Data:** Kaggle "Credit Card Fraud Detection" dataset (284,807 records)

**Requirements:**

  * Data preprocessing with Spark (missing value handling, normalization, imbalanced data handling)
  * Hyperparameter optimization with Ray Tune (100 trials)
  * Distributed training (multi-GPU support)
  * Model evaluation (Precision, Recall, F1, AUC-ROC)
  * Performance profiling
  * Monitoring system integration

**Evaluation Criteria:**

  * F1 score > 0.85
  * Training time < 30 minutes (4 GPU environment)
  * Complete reproducibility (fixed seed)
  * Generate profiling report

* * *

## Summary

In this chapter, we learned about the design and implementation of large-scale machine learning pipelines:

Topic | Key Learning Content | Practical Skills  
---|---|---  
**Pipeline Design** | Scalability, fault tolerance, monitoring | Configuration management, error handling, checkpointing  
**Data Processing** | Spark ETL, data validation, feature engineering | Distributed data transformation, quality checks, optimization  
**Distributed Training** | DDP, Ray Tune, hyperparameter optimization | Multi-node training, efficient HPO  
**Performance Optimization** | Profiling, I/O optimization, GPU utilization | Bottleneck identification, mixed precision training  
**Production Operations** | Monitoring, alerting, cost optimization | Metrics collection, anomaly detection, automation  
  
### Next Steps

  * **Kubernetes Deployment** : Containerize ML pipelines and operate on K8s
  * **Streaming Processing** : Real-time ML with Kafka + Spark Streaming
  * **MLOps Integration** : Integration with MLflow, Kubeflow, SageMaker
  * **Model Serving** : Inference optimization with TorchServe, TensorFlow Serving
  * **AutoML** : Automatic feature engineering, neural architecture search

**💡 Practical Application:**

Large-scale ML pipelines are used in business-critical applications such as recommendation systems, fraud detection, and demand forecasting. The techniques learned in this chapter are essential skills in the career path from data scientist to ML engineer.

* * *

## References

### Books

  * Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  * Ryza, S. et al. (2017). _Advanced Analytics with Spark_ (2nd ed.). O'Reilly Media.
  * Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  * Gift, N. & Deza, A. (2021). _Practical MLOps_. O'Reilly Media.

### Papers

  * Zaharia, M. et al. (2016). "Apache Spark: A Unified Engine for Big Data Processing". _Communications of the ACM_ , 59(11), 56-65.
  * Li, M. et al. (2014). "Scaling Distributed Machine Learning with the Parameter Server". _OSDI_ , 14, 583-598.
  * Goyal, P. et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour". _arXiv:1706.02677_.
  * Liaw, R. et al. (2018). "Tune: A Research Platform for Distributed Model Selection and Training". _arXiv:1807.05118_.

### Official Documentation

  * [Apache Spark Documentation](<https://spark.apache.org/docs/latest/>)
  * [PyTorch Distributed Overview](<https://pytorch.org/tutorials/beginner/dist_overview.html>)
  * [Ray Tune Documentation](<https://docs.ray.io/en/latest/tune/index.html>)
  * [MLflow Documentation](<https://mlflow.org/docs/latest/index.html>)

### Online Resources

  * [Databricks Spark Deep Learning](<https://github.com/databricks/spark-deep-learning>)
  * [TensorFlow Distributed Training Guide](<https://www.tensorflow.org/guide/distributed_training>)
  * [Facebook: Scaling ML Infrastructure](<https://engineering.fb.com/2020/08/06/ml-applications/scaling-machine-learning-infrastructure/>)
  * [Netflix: Distributed Feature Generation](<https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907>)

### Tools & Frameworks

  * **Apache Spark** : <https://spark.apache.org/>
  * **Ray** : <https://www.ray.io/>
  * **Horovod** : <https://horovod.ai/>
  * **Kubeflow** : <https://www.kubeflow.org/>
  * **Feast (Feature Store)** : <https://feast.dev/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
