---
title: "Chapter 3: Pipeline Automation"
chapter_title: "Chapter 3: Pipeline Automation"
subtitle: Automating ML Workflows and Orchestration
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 12
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Pipeline Automation. You will learn ML pipeline design principles, Build workflows with Apache Airflow, and dynamic workflows with Prefect.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand ML pipeline design principles and DAG structure
  * ✅ Build workflows with Apache Airflow
  * ✅ Create containerized pipelines with Kubeflow Pipelines
  * ✅ Implement dynamic workflows with Prefect
  * ✅ Achieve idempotency, error handling, and testability
  * ✅ Design production-ready pipelines

* * *

## 3.1 ML Pipeline Design

### What is a Pipeline?

An **ML Pipeline (Machine Learning Pipeline)** is a workflow that automates the series of processes from data collection to prediction.

> "Manual reproduction is impossible. An automated pipeline is the foundation of a reliable ML system."

### Pipeline Components

Component | Description | Example  
---|---|---  
**Data Acquisition** | Collect data from external sources | API calls, DB extraction  
**Preprocessing** | Data cleaning and transformation | Missing value handling, scaling  
**Feature Engineering** | Create features for model input | Categorical transformation, aggregation  
**Model Training** | Algorithm learning | fit, hyperparameter tuning  
**Evaluation** | Measure model performance | Accuracy, recall, F1 score  
**Deployment** | Deploy to production environment | Model serving, API conversion  
  
### DAG (Directed Acyclic Graph)

A **DAG** is a directed acyclic graph that represents dependencies between tasks. It is widely used as the standard representation method for ML pipelines.
    
    
    ```mermaid
    graph TD
        A[Data Acquisition] --> B[Data Validation]
        B --> C[Preprocessing]
        C --> D[Feature Engineering]
        D --> E[Train/Test Split]
        E --> F[Model Training]
        E --> G[Hyperparameter Tuning]
        F --> H[Model Evaluation]
        G --> H
        H --> I{Performance OK?}
        I -->|Yes| J[Model Registration]
        I -->|No| K[Send Alert]
        J --> L[Deployment]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#fce4ec
        style F fill:#ffe0b2
        style G fill:#ffe0b2
        style H fill:#c8e6c9
        style I fill:#ffccbc
        style J fill:#c5cae9
        style K fill:#ffcdd2
        style L fill:#b2dfdb
    ```

### Orchestration vs Workflow

Aspect | Orchestration | Workflow  
---|---|---  
**Control** | Centralized (orchestrator manages) | Distributed (each task is independent)  
**Examples** | Airflow, Prefect, Dagster | Step Functions, Argo Workflows  
**Use Cases** | Complex dependencies, dynamic tasks | Simple flows, event-driven  
**Visualization** | Full UI, log tracking | Basic status display  
  
### Pipeline Design Principles
    
    
    """
    5 Principles of ML Pipeline Design
    """
    
    # 1. Idempotency
    # Same input produces same output
    def preprocess_data(input_path, output_path):
        """Always generate the same output_path from the same input_path"""
        # Remove existing output before regenerating
        if os.path.exists(output_path):
            os.remove(output_path)
        # Execute processing...
    
    # 2. Rerunability
    # Failed tasks can be safely rerun
    def train_model(data_path, model_path, force=False):
        """Overwrite existing model with force=True"""
        if os.path.exists(model_path) and not force:
            print(f"Model exists: {model_path}")
            return
        # Execute training...
    
    # 3. Loose Coupling
    # Minimize dependencies between tasks
    def extract_features(raw_data):
        """Extract features from raw data (no dependency on previous tasks)"""
        return features
    
    # 4. Parameterization
    # Avoid hardcoding, externalize configuration
    def run_pipeline(config_path):
        """Load all parameters from configuration file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Use config['model_type'], config['batch_size'], etc.
    
    # 5. Observability
    # Record logs, metrics, traces
    import logging
    
    def process_batch(batch_id):
        logging.info(f"Batch {batch_id} processing started")
        try:
            # Processing...
            logging.info(f"Batch {batch_id} succeeded")
        except Exception as e:
            logging.error(f"Batch {batch_id} failed: {e}")
            raise
    

* * *

## 3.2 Apache Airflow

### What is Airflow?

**Apache Airflow** is an open-source platform for defining workflows in Python and executing them on a schedule.

#### Airflow Features

  * **DAG-based** : Represent tasks as DAGs
  * **Rich operators** : Python, Bash, SQL, cloud services, etc.
  * **Scheduler** : Scheduling with Cron expressions
  * **Web UI** : DAG visualization, log viewing
  * **Extensibility** : Custom operators, hooks, sensors

### Airflow Architecture
    
    
    ```mermaid
    graph TB
        A[Web Server] --> B[Scheduler]
        B --> C[Executor]
        C --> D1[Worker 1]
        C --> D2[Worker 2]
        C --> D3[Worker N]
        B --> E[Metadata DB]
        A --> E
        D1 --> F[Task Logs]
        D2 --> F
        D3 --> F
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
        style E fill:#ffe0b2
        style F fill:#ffccbc
    ```

Component | Role  
---|---  
**Scheduler** | Monitor DAGs, schedule tasks  
**Executor** | Manage task execution (Local, Celery, Kubernetes)  
**Worker** | Execute actual tasks  
**Web Server** | Provide UI, DAG visualization  
**Metadata DB** | Store DAGs, tasks, execution history  
  
### Basic DAG Definition
    
    
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    
    # Default arguments
    default_args = {
        'owner': 'mlops-team',
        'depends_on_past': False,  # Don't depend on past runs
        'email': ['alerts@example.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,  # Retry twice on failure
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=1),
    }
    
    # DAG definition
    dag = DAG(
        'ml_pipeline_basic',
        default_args=default_args,
        description='Basic ML pipeline',
        schedule_interval='0 2 * * *',  # Execute daily at 2:00 AM
        start_date=datetime(2025, 1, 1),
        catchup=False,  # Don't execute past unrun instances
        tags=['ml', 'training'],
    )
    
    # Task definitions
    def extract_data(**context):
        """Data extraction"""
        print("Extracting data from DB...")
        # Actual extraction logic
        data = {'records': 1000, 'timestamp': datetime.now().isoformat()}
        # Pass data to next task via XCom
        context['ti'].xcom_push(key='extracted_data', value=data)
        return data
    
    def transform_data(**context):
        """Data transformation"""
        # Get data from previous task
        ti = context['ti']
        data = ti.xcom_pull(key='extracted_data', task_ids='extract')
        print(f"Transforming data: {data['records']} records")
        # Transformation processing...
        transformed = {'records': data['records'], 'features': 50}
        return transformed
    
    def train_model(**context):
        """Model training"""
        ti = context['ti']
        data = ti.xcom_pull(task_ids='transform')
        print(f"Training model: {data['features']} features")
        # Training processing...
        model_metrics = {'accuracy': 0.92, 'f1': 0.89}
        return model_metrics
    
    # Create tasks
    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract_data,
        dag=dag,
    )
    
    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform_data,
        dag=dag,
    )
    
    train_task = PythonOperator(
        task_id='train',
        python_callable=train_model,
        dag=dag,
    )
    
    validate_task = BashOperator(
        task_id='validate',
        bash_command='echo "Model validation complete"',
        dag=dag,
    )
    
    # Define task dependencies
    extract_task >> transform_task >> train_task >> validate_task
    

### Complete ML Pipeline Example
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Complete ML Pipeline Example
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.dummy import DummyOperator
    from airflow.utils.trigger_rule import TriggerRule
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    default_args = {
        'owner': 'data-science',
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
    }
    
    dag = DAG(
        'complete_ml_pipeline',
        default_args=default_args,
        description='Complete ML pipeline (from training to evaluation)',
        schedule_interval='@daily',
        start_date=datetime(2025, 1, 1),
        catchup=False,
    )
    
    # Data collection
    def collect_data(**context):
        """Data collection task"""
        # Generate dummy data (actually retrieve from DB or API)
        import numpy as np
        np.random.seed(42)
    
        n_samples = 1000
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
        # Save data
        data.to_csv('/tmp/raw_data.csv', index=False)
        print(f"Data collection complete: {len(data)} records")
    
        # Share metadata via XCom
        context['ti'].xcom_push(key='data_size', value=len(data))
        return '/tmp/raw_data.csv'
    
    # Data validation
    def validate_data(**context):
        """Data quality validation"""
        data = pd.read_csv('/tmp/raw_data.csv')
    
        # Validation checks
        checks = {
            'no_nulls': data.isnull().sum().sum() == 0,
            'sufficient_size': len(data) >= 500,
            'target_balance': data['target'].value_counts().min() / len(data) >= 0.3
        }
    
        print(f"Data validation results: {checks}")
    
        if not all(checks.values()):
            raise ValueError(f"Data quality check failed: {checks}")
    
        return True
    
    # Preprocessing
    def preprocess_data(**context):
        """Preprocessing task"""
        data = pd.read_csv('/tmp/raw_data.csv')
    
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
    
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Save
        X_train.to_csv('/tmp/X_train.csv', index=False)
        X_test.to_csv('/tmp/X_test.csv', index=False)
        y_train.to_csv('/tmp/y_train.csv', index=False)
        y_test.to_csv('/tmp/y_test.csv', index=False)
    
        print(f"Preprocessing complete: train={len(X_train)}, test={len(X_test)}")
        return True
    
    # Model training
    def train_model(**context):
        """Model training task"""
        # Load data
        X_train = pd.read_csv('/tmp/X_train.csv')
        y_train = pd.read_csv('/tmp/y_train.csv').values.ravel()
    
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # Save model
        joblib.dump(model, '/tmp/model.pkl')
        print("Model training complete")
        return '/tmp/model.pkl'
    
    # Model evaluation
    def evaluate_model(**context):
        """Model evaluation task"""
        # Load data and model
        X_test = pd.read_csv('/tmp/X_test.csv')
        y_test = pd.read_csv('/tmp/y_test.csv').values.ravel()
        model = joblib.load('/tmp/model.pkl')
    
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
    
        print(f"Evaluation complete: {metrics}")
    
        # Share metrics via XCom
        context['ti'].xcom_push(key='metrics', value=metrics)
        return metrics
    
    # Model quality check (branching)
    def check_model_quality(**context):
        """Decide next task based on model quality"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # Accuracy threshold
        threshold = 0.8
    
        if metrics['accuracy'] >= threshold:
            print(f"Model approved: accuracy={metrics['accuracy']:.3f}")
            return 'register_model'
        else:
            print(f"Model rejected: accuracy={metrics['accuracy']:.3f} < {threshold}")
            return 'send_alert'
    
    # Model registration
    def register_model(**context):
        """Register model to registry"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # Actually register to MLflow or similar registry
        print(f"Model registration: accuracy={metrics['accuracy']:.3f}")
    
        # Version management
        import shutil
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_version = f'/tmp/model_{timestamp}.pkl'
        shutil.copy('/tmp/model.pkl', model_version)
    
        print(f"Model version: {model_version}")
        return model_version
    
    # Send alert
    def send_alert(**context):
        """Send alert if quality is insufficient"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # Actually notify via Slack or Email
        print(f"⚠️ Alert: Insufficient model quality - {metrics}")
        return True
    
    # Task definitions
    start = DummyOperator(task_id='start', dag=dag)
    
    collect = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data,
        dag=dag,
    )
    
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        dag=dag,
    )
    
    preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_data,
        dag=dag,
    )
    
    train = PythonOperator(
        task_id='train',
        python_callable=train_model,
        dag=dag,
    )
    
    evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate_model,
        dag=dag,
    )
    
    quality_check = BranchPythonOperator(
        task_id='quality_check',
        python_callable=check_model_quality,
        dag=dag,
    )
    
    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        dag=dag,
    )
    
    alert = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,
        dag=dag,
    )
    
    end = DummyOperator(
        task_id='end',
        trigger_rule=TriggerRule.ONE_SUCCESS,  # Execute if either succeeds
        dag=dag,
    )
    
    # DAG structure
    start >> collect >> validate >> preprocess >> train >> evaluate >> quality_check
    quality_check >> [register, alert]
    register >> end
    alert >> end
    

### Airflow Best Practices
    
    
    """
    Airflow Best Practices
    """
    
    # 1. Ensure task idempotency
    def idempotent_task(output_path):
        """Always generate the same output from the same input"""
        # Remove existing output
        if os.path.exists(output_path):
            os.remove(output_path)
        # Execute processing
        process_data(output_path)
    
    # 2. Use XCom only for small data
    def small_xcom(**context):
        """Pass large data through files"""
        # ❌ Bad example: Pass large DataFrame via XCom
        # context['ti'].xcom_push(key='data', value=large_df)
    
        # ✅ Good example: Pass file path
        large_df.to_parquet('/tmp/data.parquet')
        context['ti'].xcom_push(key='data_path', value='/tmp/data.parquet')
    
    # 3. Appropriate task granularity
    # ❌ Bad example: All processing in one task
    def monolithic_task():
        collect_data()
        preprocess_data()
        train_model()
        evaluate_model()
    
    # ✅ Good example: Separate each step
    collect_task >> preprocess_task >> train_task >> evaluate_task
    
    # 4. Dynamic task generation
    from airflow.operators.python import PythonOperator
    
    def create_dynamic_tasks(dag):
        """Train multiple models in parallel"""
        models = ['rf', 'xgboost', 'lightgbm']
    
        for model_name in models:
            PythonOperator(
                task_id=f'train_{model_name}',
                python_callable=train_specific_model,
                op_kwargs={'model_type': model_name},
                dag=dag,
            )
    
    # 5. Waiting with sensors
    from airflow.sensors.filesystem import FileSensor
    
    wait_for_data = FileSensor(
        task_id='wait_for_data',
        filepath='/data/input.csv',
        poke_interval=60,  # Check every 60 seconds
        timeout=3600,  # Timeout after 1 hour
        dag=dag,
    )
    

* * *

## 3.3 Kubeflow Pipelines

### What is Kubeflow Pipelines?

**Kubeflow Pipelines** is a platform for building, deploying, and managing ML workflows on Kubernetes.

#### Key Features

  * **Container-native** : Each task runs in a Docker container
  * **Reusable components** : Modularize pipeline parts
  * **Scalability** : Leverage Kubernetes auto-scaling
  * **Versioning** : Version management for pipelines and components
  * **Experiment tracking** : Compare and analyze pipeline runs

### Kubeflow Architecture
    
    
    ```mermaid
    graph TB
        A[Pipeline DSL] --> B[Compiler]
        B --> C[Pipeline YAML]
        C --> D[Kubeflow API Server]
        D --> E[Argo Workflows]
        E --> F1[Pod: Data Collection]
        E --> F2[Pod: Preprocessing]
        E --> F3[Pod: Training]
        E --> F4[Pod: Evaluation]
        D --> G[Metadata Store]
        D --> H[Artifact Store]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
        style F1 fill:#ffccbc
        style F2 fill:#ffccbc
        style F3 fill:#ffccbc
        style F4 fill:#ffccbc
        style G fill:#c5cae9
        style H fill:#b2dfdb
    ```

### Basic Pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    
    import kfp
    from kfp import dsl
    from kfp.dsl import component, Input, Output, Dataset, Model
    
    # Component definition (lightweight component)
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def load_data(output_dataset: Output[Dataset]):
        """Data loading component"""
        import pandas as pd
    
        # Generate dummy data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 0, 1, 1, 1]
        })
    
        # Save to output dataset
        data.to_csv(output_dataset.path, index=False)
        print(f"Data saved: {output_dataset.path}")
    
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def train_model(
        input_dataset: Input[Dataset],
        output_model: Output[Model],
        n_estimators: int = 100
    ):
        """Model training component"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    
        # Load data
        data = pd.read_csv(input_dataset.path)
        X = data[['feature1', 'feature2']]
        y = data['target']
    
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
    
        # Save model
        joblib.dump(model, output_model.path)
        print(f"Model saved: {output_model.path}")
    
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def evaluate_model(
        input_dataset: Input[Dataset],
        input_model: Input[Model]
    ) -> float:
        """Model evaluation component"""
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score
    
        # Load data and model
        data = pd.read_csv(input_dataset.path)
        X = data[['feature1', 'feature2']]
        y = data['target']
        model = joblib.load(input_model.path)
    
        # Evaluate
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
    
        print(f"Accuracy: {accuracy:.3f}")
        return accuracy
    
    # Pipeline definition
    @dsl.pipeline(
        name='ML Training Pipeline',
        description='Basic ML training pipeline'
    )
    def ml_pipeline(n_estimators: int = 100):
        """ML pipeline"""
        # Task definitions
        load_task = load_data()
    
        train_task = train_model(
            input_dataset=load_task.outputs['output_dataset'],
            n_estimators=n_estimators
        )
    
        evaluate_task = evaluate_model(
            input_dataset=load_task.outputs['output_dataset'],
            input_model=train_task.outputs['output_model']
        )
    
    # Pipeline compilation
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=ml_pipeline,
            package_path='ml_pipeline.yaml'
        )
        print("Pipeline compilation complete: ml_pipeline.yaml")
    

### Containerized Components
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Docker container-based component definition
    """
    
    from kfp import dsl
    from kfp.dsl import ContainerOp
    
    # Dockerfile
    """
    FROM python:3.9-slim
    
    RUN pip install pandas scikit-learn
    
    COPY train.py /app/train.py
    WORKDIR /app
    
    ENTRYPOINT ["python", "train.py"]
    """
    
    # train.py
    """
    import argparse
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    def main(args):
        # Load data
        data = pd.read_csv(args.input_data)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # Train model
        model = RandomForestClassifier(n_estimators=args.n_estimators)
        model.fit(X, y)
    
        # Save
        joblib.dump(model, args.output_model)
        print(f"Model saved: {args.output_model}")
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-data', required=True)
        parser.add_argument('--output-model', required=True)
        parser.add_argument('--n-estimators', type=int, default=100)
        args = parser.parse_args()
        main(args)
    """
    
    # Using containers in pipeline
    @dsl.pipeline(
        name='Containerized ML Pipeline',
        description='Containerized ML pipeline'
    )
    def containerized_pipeline(n_estimators: int = 100):
        """Container-based pipeline"""
    
        # Data preparation container
        prepare_op = dsl.ContainerOp(
            name='prepare-data',
            image='gcr.io/my-project/data-prep:v1',
            arguments=['--output', '/data/prepared.csv'],
            file_outputs={'data': '/data/prepared.csv'}
        )
    
        # Training container
        train_op = dsl.ContainerOp(
            name='train-model',
            image='gcr.io/my-project/train:v1',
            arguments=[
                '--input-data', prepare_op.outputs['data'],
                '--output-model', '/models/model.pkl',
                '--n-estimators', n_estimators
            ],
            file_outputs={'model': '/models/model.pkl'}
        )
    
        # Evaluation container
        evaluate_op = dsl.ContainerOp(
            name='evaluate-model',
            image='gcr.io/my-project/evaluate:v1',
            arguments=[
                '--input-data', prepare_op.outputs['data'],
                '--input-model', train_op.outputs['model']
            ]
        )
    
        # Specify GPU usage
        train_op.set_gpu_limit(1)
        train_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')
    
    # Compile and execute
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=containerized_pipeline,
            package_path='containerized_pipeline.yaml'
        )
    

### Kubeflow Pipeline Execution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    
    import kfp
    from kfp import dsl
    from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def preprocess_data(
        input_dataset: Input[Dataset],
        output_train: Output[Dataset],
        output_test: Output[Dataset],
        test_size: float = 0.2
    ):
        """Data preprocessing and train/test split"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
    
        data = pd.read_csv(input_dataset.path)
    
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
        # Save
        train_df = X_train.copy()
        train_df['target'] = y_train
        test_df = X_test.copy()
        test_df['target'] = y_test
    
        train_df.to_csv(output_train.path, index=False)
        test_df.to_csv(output_test.path, index=False)
    
        print(f"Training data: {len(train_df)} records")
        print(f"Test data: {len(test_df)} records")
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def hyperparameter_tuning(
        input_train: Input[Dataset],
        output_best_params: Output[Metrics]
    ) -> dict:
        """Hyperparameter tuning"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import json
    
        data = pd.read_csv(input_train.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # Grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
    
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X, y)
    
        best_params = grid_search.best_params_
    
        # Save metrics
        with open(output_best_params.path, 'w') as f:
            json.dump(best_params, f)
    
        print(f"Best parameters: {best_params}")
        print(f"Best score: {grid_search.best_score_:.3f}")
    
        return best_params
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def train_final_model(
        input_train: Input[Dataset],
        best_params: dict,
        output_model: Output[Model]
    ):
        """Train model with optimal parameters"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    
        data = pd.read_csv(input_train.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # Train model with optimal parameters
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X, y)
    
        # Save model
        joblib.dump(model, output_model.path)
        print(f"Model training complete: {best_params}")
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def evaluate_final_model(
        input_test: Input[Dataset],
        input_model: Input[Model],
        output_metrics: Output[Metrics]
    ):
        """Final evaluation"""
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import json
    
        data = pd.read_csv(input_test.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        model = joblib.load(input_model.path)
        y_pred = model.predict(X)
    
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'f1_score': float(f1_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred))
        }
    
        # Save
        with open(output_metrics.path, 'w') as f:
            json.dump(metrics, f)
    
        print(f"Evaluation complete: {metrics}")
    
    @dsl.pipeline(
        name='Complete ML Pipeline with Tuning',
        description='Complete ML pipeline with hyperparameter tuning'
    )
    def complete_ml_pipeline(test_size: float = 0.2):
        """Complete ML pipeline"""
        # Load data (dummy)
        load_task = load_data()
    
        # Preprocessing
        preprocess_task = preprocess_data(
            input_dataset=load_task.outputs['output_dataset'],
            test_size=test_size
        )
    
        # Hyperparameter tuning
        tuning_task = hyperparameter_tuning(
            input_train=preprocess_task.outputs['output_train']
        )
    
        # Final model training
        train_task = train_final_model(
            input_train=preprocess_task.outputs['output_train'],
            best_params=tuning_task.output
        )
    
        # Evaluation
        evaluate_task = evaluate_final_model(
            input_test=preprocess_task.outputs['output_test'],
            input_model=train_task.outputs['output_model']
        )
    
    # Compile
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=complete_ml_pipeline,
            package_path='complete_ml_pipeline.yaml'
        )
        print("Pipeline compilation complete")
    

* * *

## 3.4 Prefect

### What is Prefect?

**Prefect** is a Python-native workflow orchestration tool. It features dynamic task generation and flexible error handling.

#### Prefect Features

  * **Pythonic** : Convert regular Python functions to tasks with decorators
  * **Dynamic workflows** : Generate tasks at runtime
  * **Local execution** : Easy testing in development environment
  * **Cloud UI** : Visualization and management with Prefect Cloud
  * **Flexible scheduling** : Cron, Interval, Event-driven

### Basic Flow
    
    
    from prefect import flow, task
    from datetime import timedelta
    
    @task(retries=3, retry_delay_seconds=60)
    def extract_data():
        """Data extraction task"""
        print("Extracting data...")
        # Extraction processing
        data = {'records': 1000}
        return data
    
    @task
    def transform_data(data):
        """Data transformation task"""
        print(f"Transforming data: {data['records']} records")
        # Transformation processing
        transformed = {'records': data['records'], 'features': 50}
        return transformed
    
    @task(timeout_seconds=3600)
    def load_data(data):
        """Data loading task"""
        print(f"Loading data: {data['records']} records")
        # Loading processing
        return True
    
    @flow(name="ETL Pipeline", log_prints=True)
    def etl_pipeline():
        """ETL pipeline"""
        # Execute tasks
        raw_data = extract_data()
        transformed_data = transform_data(raw_data)
        load_data(transformed_data)
    
        print("ETL pipeline complete")
    
    if __name__ == "__main__":
        etl_pipeline()
    

### Dynamic Task Generation
    
    
    from prefect import flow, task
    from typing import List
    
    @task
    def train_model(model_type: str, data_path: str):
        """Individual model training"""
        print(f"Training {model_type} model...")
        # Training processing
        metrics = {'model': model_type, 'accuracy': 0.85}
        return metrics
    
    @task
    def select_best_model(results: List[dict]):
        """Select best model"""
        best = max(results, key=lambda x: x['accuracy'])
        print(f"Best model: {best['model']} (accuracy={best['accuracy']:.3f})")
        return best
    
    @flow(name="Multi-Model Training")
    def multi_model_training(data_path: str):
        """Train multiple models in parallel"""
        # List of models to train
        model_types = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    
        # Generate tasks dynamically
        results = []
        for model_type in model_types:
            result = train_model(model_type, data_path)
            results.append(result)
    
        # Select best model
        best_model = select_best_model(results)
    
        return best_model
    
    if __name__ == "__main__":
        best = multi_model_training(data_path="/data/train.csv")
        print(f"Selected model: {best}")
    

### Prefect 2.0 Complete Example
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Prefect 2.0 Complete Example
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import ConcurrentTaskRunner
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    from datetime import timedelta
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    
    @task(
        name="Data Collection",
        retries=3,
        retry_delay_seconds=60,
        cache_key_fn=None,  # Disable caching
        timeout_seconds=300
    )
    def collect_data():
        """Data collection task"""
        logger = get_run_logger()
        logger.info("Data collection started")
    
        # Generate dummy data
        import numpy as np
        np.random.seed(42)
    
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
    
        # Save data
        output_path = '/tmp/raw_data.csv'
        data.to_csv(output_path, index=False)
    
        logger.info(f"Data collection complete: {len(data)} records")
        return output_path
    
    @task(name="Data Validation")
    def validate_data(data_path: str):
        """Data quality validation"""
        logger = get_run_logger()
        logger.info("Data validation started")
    
        data = pd.read_csv(data_path)
    
        # Validation
        checks = {
            'no_nulls': data.isnull().sum().sum() == 0,
            'sufficient_size': len(data) >= 500,
            'feature_count': data.shape[1] >= 4
        }
    
        logger.info(f"Validation results: {checks}")
    
        if not all(checks.values()):
            raise ValueError(f"Data validation failed: {checks}")
    
        return True
    
    @task(name="Preprocessing")
    def preprocess_data(data_path: str):
        """Data preprocessing"""
        logger = get_run_logger()
        logger.info("Preprocessing started")
    
        data = pd.read_csv(data_path)
    
        # Split
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Save
        paths = {
            'X_train': '/tmp/X_train.csv',
            'X_test': '/tmp/X_test.csv',
            'y_train': '/tmp/y_train.csv',
            'y_test': '/tmp/y_test.csv'
        }
    
        X_train.to_csv(paths['X_train'], index=False)
        X_test.to_csv(paths['X_test'], index=False)
        y_train.to_csv(paths['y_train'], index=False)
        y_test.to_csv(paths['y_test'], index=False)
    
        logger.info(f"Preprocessing complete: train={len(X_train)}, test={len(X_test)}")
        return paths
    
    @task(name="Model Training", timeout_seconds=1800)
    def train_model(data_paths: dict, n_estimators: int = 100):
        """Model training"""
        logger = get_run_logger()
        logger.info(f"Model training started: n_estimators={n_estimators}")
    
        # Load data
        X_train = pd.read_csv(data_paths['X_train'])
        y_train = pd.read_csv(data_paths['y_train']).values.ravel()
    
        # Training
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    
        # Save
        model_path = '/tmp/model.pkl'
        joblib.dump(model, model_path)
    
        logger.info("Model training complete")
        return model_path
    
    @task(name="Model Evaluation")
    def evaluate_model(model_path: str, data_paths: dict):
        """Model evaluation"""
        logger = get_run_logger()
        logger.info("Model evaluation started")
    
        # Load
        X_test = pd.read_csv(data_paths['X_test'])
        y_test = pd.read_csv(data_paths['y_test']).values.ravel()
        model = joblib.load(model_path)
    
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        metrics = {
            'accuracy': float(accuracy),
            'test_samples': len(y_test)
        }
    
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    @task(name="Model Registration")
    def register_model(model_path: str, metrics: dict):
        """Model registration"""
        logger = get_run_logger()
    
        # Quality check
        if metrics['accuracy'] < 0.7:
            logger.warning(f"Insufficient model quality: accuracy={metrics['accuracy']:.3f}")
            return False
    
        # Registration (actually MLflow, etc.)
        import shutil
        from datetime import datetime
    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        registry_path = f'/tmp/models/model_{timestamp}.pkl'
    
        import os
        os.makedirs('/tmp/models', exist_ok=True)
        shutil.copy(model_path, registry_path)
    
        logger.info(f"Model registration complete: {registry_path}")
        return registry_path
    
    @flow(
        name="ML Training Pipeline",
        description="Complete ML training pipeline",
        task_runner=ConcurrentTaskRunner(),  # Parallel execution
        log_prints=True
    )
    def ml_training_pipeline(n_estimators: int = 100):
        """Main ML pipeline"""
        logger = get_run_logger()
        logger.info("Pipeline started")
    
        # Execute tasks
        data_path = collect_data()
        validate_data(data_path)
        data_paths = preprocess_data(data_path)
        model_path = train_model(data_paths, n_estimators)
        metrics = evaluate_model(model_path, data_paths)
        registry_path = register_model(model_path, metrics)
    
        logger.info(f"Pipeline complete: {registry_path}")
        return {
            'model_path': registry_path,
            'metrics': metrics
        }
    
    # Deployment definition
    if __name__ == "__main__":
        # Local execution
        result = ml_training_pipeline(n_estimators=100)
        print(f"Result: {result}")
    
        # Create deployment (to Prefect Cloud)
        """
        deployment = Deployment.build_from_flow(
            flow=ml_training_pipeline,
            name="daily-ml-training",
            schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2:00 AM
            work_queue_name="ml-training",
            parameters={"n_estimators": 100}
        )
        deployment.apply()
        """
    

### Prefect Cloud Integration
    
    
    """
    Prefect Cloud integration and deployment
    """
    
    from prefect import flow, task
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import IntervalSchedule
    from datetime import timedelta
    
    @flow
    def production_ml_pipeline():
        """Production ML pipeline"""
        # Pipeline processing...
        pass
    
    # Deployment configuration
    deployment = Deployment.build_from_flow(
        flow=production_ml_pipeline,
        name="production-deployment",
        schedule=IntervalSchedule(interval=timedelta(hours=6)),  # Every 6 hours
        work_queue_name="production",
        tags=["ml", "production"],
        parameters={},
        description="Production ML pipeline"
    )
    
    # Deploy
    # deployment.apply()
    
    # CLI deployment
    """
    # Login to Prefect Cloud
    prefect cloud login
    
    # Create deployment
    prefect deployment build ml_pipeline.py:production_ml_pipeline -n production -q production
    
    # Apply deployment
    prefect deployment apply production_ml_pipeline-deployment.yaml
    
    # Start agent
    prefect agent start -q production
    """
    

* * *

## 3.5 Pipeline Design Best Practices

### Ensuring Idempotency

**Idempotency** is the property where running with the same input multiple times produces the same result.
    
    
    """
    Patterns for ensuring idempotency
    """
    
    import os
    import shutil
    from pathlib import Path
    
    # ❌ Non-idempotent processing
    def non_idempotent_process(output_path):
        """Append to existing data (result changes with each run)"""
        with open(output_path, 'a') as f:  # append mode
            f.write("new data\n")
    
    # ✅ Idempotent processing
    def idempotent_process(output_path):
        """Overwrite existing data (always same result)"""
        if os.path.exists(output_path):
            os.remove(output_path)  # Remove existing file
    
        with open(output_path, 'w') as f:  # write mode
            f.write("new data\n")
    
    # Idempotent directory creation
    def create_output_dir(dir_path):
        """Create directory idempotently"""
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Remove existing directory
        os.makedirs(dir_path)
    
    # Idempotent processing with timestamp
    def process_with_version(input_path, output_dir, version):
        """Ensure idempotency with version management"""
        output_path = os.path.join(output_dir, f'output_v{version}.csv')
    
        # Same version always produces same result
        if os.path.exists(output_path):
            os.remove(output_path)
    
        # Execute processing
        process_data(input_path, output_path)
    
    # Idempotent database update
    def upsert_data(data, table_name):
        """UPSERT (update if exists, insert if not)"""
        # SQL example
        query = f"""
        INSERT INTO {table_name} (id, value)
        VALUES (%(id)s, %(value)s)
        ON CONFLICT (id) DO UPDATE
        SET value = EXCLUDED.value
        """
        # Execute...
    

### Error Handling
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Robust error handling
    """
    
    from typing import Optional
    import logging
    import time
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Retry decorator
    def retry_on_failure(max_retries=3, delay=5, backoff=2):
        """Decorator that retries on failure"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                retries = 0
                current_delay = delay
    
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"{func.__name__} failed (max retries reached): {e}")
                            raise
    
                        logger.warning(
                            f"{func.__name__} failed ({retries}/{max_retries}): {e}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff  # Exponential backoff
    
            return wrapper
        return decorator
    
    # Usage example
    @retry_on_failure(max_retries=3, delay=5, backoff=2)
    def fetch_data_from_api(url):
        """Fetch data from API (with retries)"""
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    # Task-level error handling
    def safe_task_execution(task_func, *args, **kwargs):
        """Execute task safely"""
        try:
            logger.info(f"Task started: {task_func.__name__}")
            result = task_func(*args, **kwargs)
            logger.info(f"Task succeeded: {task_func.__name__}")
            return result, None
    
        except Exception as e:
            logger.error(f"Task failed: {task_func.__name__} - {e}", exc_info=True)
            return None, str(e)
    
    # Pipeline-level error handling
    def run_pipeline_with_recovery(tasks):
        """Pipeline execution with recovery"""
        results = {}
        failed_tasks = []
    
        for task_name, task_func in tasks.items():
            result, error = safe_task_execution(task_func)
    
            if error:
                failed_tasks.append({
                    'task': task_name,
                    'error': error
                })
                # Abort on critical task failure
                if is_critical_task(task_name):
                    logger.error(f"Critical task failed: {task_name}. Aborting pipeline")
                    break
            else:
                results[task_name] = result
    
        # Failure summary
        if failed_tasks:
            logger.warning(f"Failed tasks count: {len(failed_tasks)}")
            for failure in failed_tasks:
                logger.warning(f"  - {failure['task']}: {failure['error']}")
    
        return results, failed_tasks
    
    def is_critical_task(task_name):
        """Determine if task is critical"""
        critical_tasks = ['data_validation', 'model_training']
        return task_name in critical_tasks
    

### Parameterization
    
    
    # Requirements:
    # - Python 3.9+
    # - pyyaml>=6.0.0
    
    """
    Configuration parameterization
    """
    
    import yaml
    import json
    from dataclasses import dataclass
    from typing import Dict, Any
    
    # Configuration management with dataclass
    @dataclass
    class PipelineConfig:
        """Pipeline configuration"""
        data_source: str
        output_dir: str
        model_type: str
        n_estimators: int = 100
        test_size: float = 0.2
        random_state: int = 42
    
    # YAML configuration file
    """
    # config.yaml
    pipeline:
      data_source: "s3://bucket/data.csv"
      output_dir: "/tmp/output"
      model_type: "random_forest"
      n_estimators: 100
      test_size: 0.2
      random_state: 42
    
    hyperparameters:
      max_depth: 10
      min_samples_split: 5
    """
    
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run_pipeline_with_config(config_path: str):
        """Pipeline execution using configuration file"""
        config = load_config(config_path)
    
        # Get configuration
        pipeline_config = PipelineConfig(**config['pipeline'])
        hyperparams = config['hyperparameters']
    
        # Execute pipeline
        print(f"Data source: {pipeline_config.data_source}")
        print(f"Model type: {pipeline_config.model_type}")
        print(f"Hyperparameters: {hyperparams}")
    
        # Processing...
    
    # Load configuration from environment variables
    import os
    
    def get_config_from_env():
        """Get configuration from environment variables"""
        config = {
            'data_source': os.getenv('DATA_SOURCE', 'default.csv'),
            'model_type': os.getenv('MODEL_TYPE', 'random_forest'),
            'n_estimators': int(os.getenv('N_ESTIMATORS', '100')),
            'output_dir': os.getenv('OUTPUT_DIR', '/tmp/output')
        }
        return config
    
    # Command-line arguments
    import argparse
    
    def parse_args():
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(description='ML Pipeline')
    
        parser.add_argument('--config', type=str, required=True,
                           help='Configuration file path')
        parser.add_argument('--data-source', type=str,
                           help='Data source (overrides config file)')
        parser.add_argument('--n-estimators', type=int, default=100,
                           help='Number of trees')
    
        return parser.parse_args()
    
    if __name__ == '__main__':
        args = parse_args()
    
        # Load configuration file
        config = load_config(args.config)
    
        # Override with command-line arguments
        if args.data_source:
            config['pipeline']['data_source'] = args.data_source
        if args.n_estimators:
            config['pipeline']['n_estimators'] = args.n_estimators
    
        # Execute pipeline
        run_pipeline_with_config(args.config)
    

### Testability
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Testable pipeline design
    """
    
    import unittest
    from unittest.mock import Mock, patch
    import pandas as pd
    
    # Testable function design
    def load_data(data_source):
        """Data loading (easy to test)"""
        # Implementation...
        pass
    
    def preprocess(data):
        """Preprocessing (pure function)"""
        # No side effects, generate output from input
        processed = data.copy()
        # Processing...
        return processed
    
    def train_model(X, y, model_class, **hyperparams):
        """Model training (dependency injection)"""
        model = model_class(**hyperparams)
        model.fit(X, y)
        return model
    
    # Unit tests
    class TestPreprocessing(unittest.TestCase):
        """Preprocessing tests"""
    
        def test_preprocess_removes_nulls(self):
            """Test that null values are removed"""
            # Test data
            data = pd.DataFrame({
                'feature1': [1, 2, None, 4],
                'feature2': [5, None, 7, 8]
            })
    
            # Execute
            result = preprocess(data)
    
            # Verify
            self.assertEqual(result.isnull().sum().sum(), 0)
    
        def test_preprocess_scales_features(self):
            """Test that features are scaled"""
            data = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [10, 20, 30, 40]
            })
    
            result = preprocess(data)
    
            # Mean is close to 0
            self.assertAlmostEqual(result['feature1'].mean(), 0, places=1)
    
    # Integration tests using mocks
    class TestMLPipeline(unittest.TestCase):
        """Pipeline-wide tests"""
    
        @patch('my_pipeline.load_data')
        @patch('my_pipeline.save_model')
        def test_pipeline_end_to_end(self, mock_save, mock_load):
            """End-to-end pipeline test"""
            # Mock data
            mock_data = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'target': [0, 1, 0]
            })
            mock_load.return_value = mock_data
    
            # Execute pipeline
            # run_pipeline(config)
    
            # Verify
            mock_save.assert_called_once()
    
    # Data validation
    def validate_pipeline_output(output_path):
        """Validate pipeline output"""
        import os
    
        checks = {
            'file_exists': os.path.exists(output_path),
            'file_size': os.path.getsize(output_path) > 0 if os.path.exists(output_path) else False
        }
    
        assert all(checks.values()), f"Output validation failed: {checks}"
        return True
    
    if __name__ == '__main__':
        unittest.main()
    

* * *

## 3.6 Chapter Summary

### What We Learned

  1. **Pipeline Design Principles**

     * Express task dependencies with DAG structure
     * Idempotency, rerunability, loose coupling
     * Parameterization and observability
  2. **Apache Airflow**

     * Python-based workflow orchestration
     * Scheduler and Executor architecture
     * Rich operators and UI
  3. **Kubeflow Pipelines**

     * Kubernetes-native pipelines
     * Containerized components
     * Reusable component design
  4. **Prefect**

     * Pythonic dynamic workflows
     * Flexible error handling
     * Local development and Cloud integration
  5. **Best Practices**

     * Ensure idempotency and safe reruns
     * Robust error handling and retries
     * Configuration parameterization
     * Testable design

### Tool Comparison

Tool | Strengths | Use Cases  
---|---|---  
**Airflow** | Mature ecosystem, rich operators | Complex batch processing, data ETL  
**Kubeflow** | Kubernetes-native, ML-specialized | Large-scale ML, GPU utilization, multi-cloud  
**Prefect** | Pythonic, dynamic tasks, local development | Flexible workflows, event-driven  
  
### To the Next Chapter

In Chapter 4, we will learn about **Model Management and Versioning** :

  * Model tracking with MLflow
  * Model registry
  * Experiment management and metrics logging
  * Model versioning
  * Production deployment

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Explain what a DAG (Directed Acyclic Graph) is and describe why DAGs are important in ML pipelines.

Sample Answer

**Answer** :

A **DAG (Directed Acyclic Graph)** consists of nodes (tasks) and edges (dependencies) with the following properties:

  1. **Directed** : Edges have direction (from task A to task B)
  2. **Acyclic** : No loops (doesn't return to the same task)

**Why DAGs are important in ML pipelines** :

  * **Clarify dependencies** : Execution order between tasks is visually understandable
  * **Optimize parallel execution** : Tasks without dependencies can be executed in parallel
  * **Reproducibility** : Same DAG guarantees same execution order
  * **Easy debugging** : Easy to identify and rerun failed points
  * **Scalability** : Can manage complex workflows

Example:
    
    
    Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation
    

This structure allows each step to be executed independently, and only the failed step can be rerun on failure.

### Problem 2 (Difficulty: medium)

The following Airflow DAG code has errors. Identify the problems and fix them.
    
    
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime
    
    def task_a():
        print("Task A")
    
    def task_b():
        print("Task B")
    
    dag = DAG('example', start_date=datetime(2025, 1, 1))
    
    task1 = PythonOperator(task_id='task_a', python_callable=task_a)
    task2 = PythonOperator(task_id='task_b', python_callable=task_b)
    
    task1 >> task2
    

Sample Answer

**Problems** :

  1. `PythonOperator` doesn't have `dag` parameter specified
  2. `schedule_interval` is not defined
  3. `default_args` is not set (recommended)

**Fixed version** :
    
    
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta
    
    # Define default arguments
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
    
    def task_a():
        print("Task A")
    
    def task_b():
        print("Task B")
    
    # DAG definition (add schedule_interval)
    dag = DAG(
        'example',
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule_interval='@daily',  # Execute daily
        catchup=False
    )
    
    # Add dag argument
    task1 = PythonOperator(
        task_id='task_a',
        python_callable=task_a,
        dag=dag  # ✅ Add dag argument
    )
    
    task2 = PythonOperator(
        task_id='task_b',
        python_callable=task_b,
        dag=dag  # ✅ Add dag argument
    )
    
    # Dependencies
    task1 >> task2
    

**Improvements** :

  * Added retry settings with `default_args`
  * Made execution frequency explicit with `schedule_interval`
  * Skip past unrun instances with `catchup=False`

### Problem 3 (Difficulty: medium)

Explain what idempotency is and modify the following function to be idempotent.
    
    
    def process_data(input_file, output_file):
        data = pd.read_csv(input_file)
        # Processing...
        processed_data = data * 2
    
        # Append to existing file
        with open(output_file, 'a') as f:
            processed_data.to_csv(f, index=False)
    

Sample Answer

**What is idempotency** :

It's the property where running with the same input multiple times always produces the same result. Why it's important in ML pipelines:

  * **Safe reruns** : Can confidently rerun on failure
  * **Predictability** : Results are always consistent
  * **Easy debugging** : Problems can be reliably reproduced

**Problem with original code** :

Using append mode (`'a'`) causes data to accumulate with each run, producing different results.

**Fixed version (idempotent)** :
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import os
    import pandas as pd
    
    def process_data(input_file, output_file):
        """Idempotent data processing"""
        # Load data
        data = pd.read_csv(input_file)
    
        # Processing
        processed_data = data * 2
    
        # ✅ Remove existing file before writing
        if os.path.exists(output_file):
            os.remove(output_file)
    
        # Write new (overwrite mode)
        processed_data.to_csv(output_file, index=False)
    
        print(f"Processing complete: {output_file}")
    
    # Alternative method: temporary file and atomic move
    import shutil
    import tempfile
    
    def process_data_atomic(input_file, output_file):
        """Ensure idempotency with atomic write"""
        data = pd.read_csv(input_file)
        processed_data = data * 2
    
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            processed_data.to_csv(tmp_path, index=False)
    
        # Move atomically (overwrite existing file)
        shutil.move(tmp_path, output_file)
        print(f"Processing complete: {output_file}")
    

**Verification** :
    
    
    # Execute twice with same input
    process_data('input.csv', 'output.csv')  # First time
    process_data('input.csv', 'output.csv')  # Second time
    
    # output.csv always has same content (idempotent)
    

### Problem 4 (Difficulty: hard)

Using Prefect, create a pipeline that trains multiple models in parallel and selects the best model. Use 3 types of models: ['random_forest', 'xgboost', 'lightgbm'].

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - lightgbm>=4.0.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - xgboost>=2.0.0
    
    """
    Example: Using Prefect, create a pipeline that trains multiple models
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import ConcurrentTaskRunner
    from typing import List, Dict
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Dummy data generation
    @task
    def generate_data():
        """Data generation"""
        import numpy as np
        np.random.seed(42)
    
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
    
        return data
    
    @task
    def split_data(data: pd.DataFrame, test_size: float = 0.2):
        """Data splitting"""
        logger = get_run_logger()
    
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
        logger.info(f"Training data: {len(X_train)} records, Test data: {len(X_test)} records")
    
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    @task
    def train_model(model_type: str, data: Dict):
        """Individual model training"""
        logger = get_run_logger()
        logger.info(f"{model_type} model training started")
    
        X_train = data['X_train']
        y_train = data['y_train']
    
        # Model selection
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            # Use RF as alternative if xgboost is not available
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            except ImportError:
                logger.warning("XGBoost not installed. Using RandomForest as alternative")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            # Use RF as alternative if lightgbm is not available
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            except ImportError:
                logger.warning("LightGBM not installed. Using RandomForest as alternative")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
        # Training
        model.fit(X_train, y_train)
    
        logger.info(f"{model_type} model training complete")
    
        return {
            'model_type': model_type,
            'model': model
        }
    
    @task
    def evaluate_model(model_info: Dict, data: Dict):
        """Model evaluation"""
        logger = get_run_logger()
    
        model_type = model_info['model_type']
        model = model_info['model']
    
        X_test = data['X_test']
        y_test = data['y_test']
    
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        result = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'model': model
        }
    
        logger.info(f"{model_type} - Accuracy: {accuracy:.4f}")
    
        return result
    
    @task
    def select_best_model(results: List[Dict]):
        """Best model selection"""
        logger = get_run_logger()
    
        # Select best model by accuracy
        best_result = max(results, key=lambda x: x['accuracy'])
    
        logger.info(f"Best model: {best_result['model_type']} (Accuracy: {best_result['accuracy']:.4f})")
    
        # Compare all models
        logger.info("\n=== Model Comparison ===")
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            logger.info(f"{result['model_type']}: {result['accuracy']:.4f}")
    
        return best_result
    
    @task
    def save_best_model(best_result: Dict, output_path: str = '/tmp/best_model.pkl'):
        """Save best model"""
        logger = get_run_logger()
    
        model_type = best_result['model_type']
        model = best_result['model']
    
        # Save model
        joblib.dump(model, output_path)
    
        logger.info(f"Best model saved: {output_path} ({model_type})")
    
        return {
            'model_type': model_type,
            'accuracy': best_result['accuracy'],
            'path': output_path
        }
    
    @flow(
        name="Multi-Model Training Pipeline",
        description="Multi-model parallel training pipeline",
        task_runner=ConcurrentTaskRunner()  # Parallel execution
    )
    def multi_model_training_pipeline(
        model_types: List[str] = None,
        test_size: float = 0.2
    ):
        """Multi-model parallel training pipeline"""
        logger = get_run_logger()
    
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm']
    
        logger.info(f"Training models: {model_types}")
    
        # Data preparation
        data = generate_data()
        split_result = split_data(data, test_size)
    
        # Parallel training
        trained_models = []
        for model_type in model_types:
            trained_model = train_model(model_type, split_result)
            trained_models.append(trained_model)
    
        # Parallel evaluation
        results = []
        for trained_model in trained_models:
            result = evaluate_model(trained_model, split_result)
            results.append(result)
    
        # Select best model
        best_result = select_best_model(results)
    
        # Save
        saved_info = save_best_model(best_result)
    
        logger.info(f"Pipeline complete: {saved_info}")
    
        return saved_info
    
    # Execute
    if __name__ == "__main__":
        result = multi_model_training_pipeline(
            model_types=['random_forest', 'xgboost', 'lightgbm'],
            test_size=0.2
        )
    
        print(f"\nFinal result: {result}")
    

**Execution result example** :
    
    
    === Model Comparison ===
    lightgbm: 0.9250
    random_forest: 0.9200
    xgboost: 0.9150
    
    Best model: lightgbm (Accuracy: 0.9250)
    Best model saved: /tmp/best_model.pkl (lightgbm)
    

### Problem 5 (Difficulty: hard)

Implement an error handling feature that meets the following requirements: 1\. Retry up to 3 times maximum 2\. Exponential backoff (1 second, 2 seconds, 4 seconds) 3\. Retry only specific exceptions (ValueError, ConnectionError, etc.) 4\. Logging of retry history

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    """
    Example: Implement an error handling feature that meets the following
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 10-20 seconds
    Dependencies: None
    """
    
    import time
    import logging
    from functools import wraps
    from typing import Callable, Tuple, Type
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    def retry_with_backoff(
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (ValueError, ConnectionError)
    ):
        """
        Retry decorator with exponential backoff
    
        Args:
            max_retries: Maximum number of retries
            initial_delay: Wait time before first retry (seconds)
            backoff_factor: Backoff multiplier
            retryable_exceptions: Tuple of exceptions to retry
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                delay = initial_delay
    
                while True:
                    try:
                        # Execute function
                        result = func(*args, **kwargs)
    
                        # Log on success
                        if retries > 0:
                            logger.info(
                                f"{func.__name__} succeeded (after {retries} retries)"
                            )
    
                        return result
    
                    except retryable_exceptions as e:
                        retries += 1
    
                        # Max retries reached
                        if retries > max_retries:
                            logger.error(
                                f"{func.__name__} failed: max retries ({max_retries}) reached"
                            )
                            logger.error(f"Final error: {type(e).__name__}: {e}")
                            raise
    
                        # Retry log
                        logger.warning(
                            f"{func.__name__} failed ({retries}/{max_retries}): "
                            f"{type(e).__name__}: {e}"
                        )
                        logger.info(f"Retrying in {delay:.1f} seconds...")
    
                        # Wait
                        time.sleep(delay)
    
                        # Exponential backoff
                        delay *= backoff_factor
    
                    except Exception as e:
                        # Immediately re-raise non-retryable exceptions
                        logger.error(
                            f"{func.__name__} failed (non-retryable): "
                            f"{type(e).__name__}: {e}"
                        )
                        raise
    
            return wrapper
        return decorator
    
    # Usage example 1: API call
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    def fetch_data_from_api(url: str):
        """Fetch data from API (with retries)"""
        import requests
    
        logger.info(f"API call: {url}")
    
        # Actual API call
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    
        return response.json()
    
    # Usage example 2: Database connection
    @retry_with_backoff(
        max_retries=5,
        initial_delay=2.0,
        backoff_factor=2.0,
        retryable_exceptions=(ConnectionError,)
    )
    def connect_to_database(host: str, port: int):
        """Database connection (with retries)"""
        logger.info(f"DB connection attempt: {host}:{port}")
    
        # Actual DB connection processing
        # connection = psycopg2.connect(host=host, port=port, ...)
    
        return "connection_object"
    
    # Usage example 3: Data validation
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        retryable_exceptions=(ValueError,)
    )
    def validate_and_process_data(data):
        """Data validation and processing (with retries)"""
        logger.info("Data validation started")
    
        # Validation
        if data is None:
            raise ValueError("Data is None")
    
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} records")
    
        # Processing
        processed = data * 2
    
        return processed
    
    # Test function
    def test_retry_mechanism():
        """Test retry mechanism"""
    
        # Test 1: Eventually succeeds
        attempt_count = 0
    
        @retry_with_backoff(max_retries=3, initial_delay=0.5)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
    
            if attempt_count < 3:
                raise ValueError(f"Failed (attempt {attempt_count})")
    
            return "Success"
    
        print("\n=== Test 1: Success after retry ===")
        result = flaky_function()
        print(f"Result: {result}")
        print(f"Attempt count: {attempt_count}")
    
        # Test 2: Fails with max retries
        @retry_with_backoff(max_retries=2, initial_delay=0.5)
        def always_fails():
            raise ValueError("Always fails")
    
        print("\n=== Test 2: Fails with max retries ===")
        try:
            always_fails()
        except ValueError as e:
            print(f"Failed as expected: {e}")
    
        # Test 3: Non-retryable exception
        @retry_with_backoff(
            max_retries=3,
            retryable_exceptions=(ValueError,)
        )
        def non_retryable_error():
            raise RuntimeError("Non-retryable error")
    
        print("\n=== Test 3: Non-retryable exception ===")
        try:
            non_retryable_error()
        except RuntimeError as e:
            print(f"Failed immediately: {e}")
    
    if __name__ == "__main__":
        test_retry_mechanism()
    

**Execution example** :
    
    
    === Test 1: Success after retry ===
    WARNING - flaky_function failed (1/3): ValueError: Failed (attempt 1)
    INFO - Retrying in 0.5 seconds...
    WARNING - flaky_function failed (2/3): ValueError: Failed (attempt 2)
    INFO - Retrying in 1.0 seconds...
    INFO - flaky_function succeeded (after 2 retries)
    Result: Success
    Attempt count: 3
    
    === Test 2: Fails with max retries ===
    WARNING - always_fails failed (1/2): ValueError: Always fails
    INFO - Retrying in 0.5 seconds...
    WARNING - always_fails failed (2/2): ValueError: Always fails
    INFO - Retrying in 1.0 seconds...
    ERROR - always_fails failed: max retries (2) reached
    ERROR - Final error: ValueError: Always fails
    Failed as expected: Always fails
    
    === Test 3: Non-retryable exception ===
    ERROR - non_retryable_error failed (non-retryable): RuntimeError: Non-retryable error
    Failed immediately: Non-retryable error
    

* * *

## References

  1. Apache Airflow Documentation. (2025). _Airflow Concepts_. Retrieved from https://airflow.apache.org/docs/
  2. Kubeflow Pipelines Documentation. (2025). _Building Pipelines_. Retrieved from https://www.kubeflow.org/docs/components/pipelines/
  3. Prefect Documentation. (2025). _Core Concepts_. Retrieved from https://docs.prefect.io/
  4. Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  5. Lakshmanan, V., Robinson, S., & Munn, M. (2020). _Machine Learning Design Patterns_. O'Reilly Media.
