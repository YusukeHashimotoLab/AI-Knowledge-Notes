---
title: "Chapter 1: MLOps Fundamentals"
chapter_title: "Chapter 1: MLOps Fundamentals"
subtitle: Foundation Technologies Supporting Machine Learning System Operations
reading_time: 25-30 minutes
difficulty: Beginner
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers the fundamentals of MLOps Fundamentals, which what is mlops. You will learn entire machine learning lifecycle, main components of MLOps, and about the MLOps tool ecosystem.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the definition and necessity of MLOps
  * ✅ Grasp the entire machine learning lifecycle
  * ✅ Understand the main components of MLOps
  * ✅ Learn about the MLOps tool ecosystem
  * ✅ Understand the MLOps maturity model and evaluate current status
  * ✅ Build the foundation of practical MLOps pipelines

* * *

## 1.1 What is MLOps

### Challenges in Machine Learning Systems

Many machine learning projects end at the PoC (Proof of Concept) stage and fail to deploy to production environments. The main reasons are as follows:

Challenge | Description | Impact  
---|---|---  
**Gap Between Model and Code** | Jupyter Notebook code doesn't work in production | Deployment delays, increased manual work  
**Lack of Reproducibility** | Cannot reproduce the same results (data, code, environment mismatch) | Difficult debugging, quality degradation  
**Model Degradation** | Model performance deteriorates over time | Prediction accuracy decline, business loss  
**Scalability** | Cannot handle large volumes of requests | System downtime, response delays  
**Lack of Governance** | Unclear who deployed which model when | Compliance violations, audit impossibility  
  
> **Statistics** : According to Gartner research, approximately 85% of machine learning projects do not reach production environments.

### Definition and Purpose of MLOps

**MLOps (Machine Learning Operations)** is a set of practices and tools to automate and standardize the development, deployment, and operation of machine learning models.

**Purpose of MLOps** :

  * **Rapid Deployment** : Deploy models to production quickly and reliably
  * **Reproducibility** : Guarantee complete reproduction of experiments and models
  * **Automation** : Reduce manual work and minimize errors
  * **Monitoring** : Continuous monitoring and improvement of model performance
  * **Scalability** : Support numerous models and data
  * **Governance** : Compliance and audit support

### Relationship with DevOps/DataOps

MLOps applies DevOps and DataOps principles to machine learning:

Concept | Focus | Main Practices  
---|---|---  
**DevOps** | Software development and operations | CI/CD, infrastructure automation, monitoring  
**DataOps** | Data pipelines and quality | Data versioning, quality checks, metadata management  
**MLOps** | Machine learning model lifecycle | Experiment management, model versioning, automatic retraining  
      
    
    ```mermaid
    graph LR
        A[DevOps] --> D[MLOps]
        B[DataOps] --> D
        C[Machine Learning] --> D
    
        D --> E[Automated MLLifecycle]
    
        style A fill:#e3f2fd
        style B fill:#f3e5f5
        style C fill:#fff3e0
        style D fill:#c8e6c9
        style E fill:#ffccbc
    ```

### Real Examples of Problems MLOps Solves
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Problem: Models developed in Jupyter Notebook don't work in production
    
    Causes:
    - Different library versions between development and production
    - Data preprocessing steps are not documented
    - Model dependencies are unclear
    """
    
    # ❌ Problematic approach (no reproducibility)
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # Data loading (which version? when?)
    df = pd.read_csv('data.csv')
    
    # Preprocessing (unclear steps)
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Model training (no hyperparameter recording)
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save (no metadata)
    import pickle
    pickle.dump(model, open('model.pkl', 'wb'))
    
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    ✅ MLOps approach (with reproducibility)
    
    Features:
    - Version control (code, data, model)
    - Explicit environment (requirements.txt, Docker)
    - Metadata recording (experiment results, parameters)
    - Pipelined (reproducible processing flow)
    """
    
    import mlflow
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import json
    from datetime import datetime
    
    # Start MLflow experiment
    mlflow.set_experiment("customer_churn_prediction")
    
    with mlflow.start_run():
        # 1. Record data version
        data_version = "v1.2.3"
        mlflow.log_param("data_version", data_version)
    
        # 2. Load data (version-controlled data)
        df = pd.read_csv(f'data/{data_version}/data.csv')
    
        # 3. Preprocessing (explicit steps)
        df = df.dropna()
        X = df.drop('target', axis=1)
        y = df['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # 4. Record hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        mlflow.log_params(params)
    
        # 5. Model training
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
    
        # 6. Evaluation and recording
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
    
        # 7. Save model (with metadata)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn_predictor"
        )
    
        # 8. Additional metadata
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("created_by", "data_science_team")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
    
        print(f"✓ Model training complete - Accuracy: {accuracy:.3f}")
        print(f"✓ Experiment ID: {mlflow.active_run().info.run_id}")
    

* * *

## 1.2 ML Lifecycle

### Overview of Machine Learning Projects

Machine learning projects are an iterative process consisting of the following phases:
    
    
    ```mermaid
    graph TB
        A[Business Understanding] --> B[Data Collection & Preparation]
        B --> C[Model Development & Training]
        C --> D[Model Evaluation]
        D --> E{Performance OK?}
        E -->|No| C
        E -->|Yes| F[Deployment]
        F --> G[Monitoring]
        G --> H{Retraining Needed?}
        H -->|Yes| B
        H -->|No| G
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#fce4ec
        style F fill:#c8e6c9
        style G fill:#ffccbc
    ```

### 1\. Data Collection and Preparation

Building data pipelines and quality assurance:
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Data Collection & Preparation Phase
    
    Purpose:
    - Build high-quality datasets
    - Data validation and versioning
    - Reproducible preprocessing pipeline
    """
    
    import pandas as pd
    import great_expectations as ge
    from sklearn.model_selection import train_test_split
    import hashlib
    import json
    
    class DataPipeline:
        """Data pipeline class"""
    
        def __init__(self, data_path, version):
            self.data_path = data_path
            self.version = version
            self.metadata = {}
    
        def load_data(self):
            """Load data"""
            df = pd.read_csv(self.data_path)
    
            # Calculate data hash (for integrity check)
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(df).values
            ).hexdigest()
    
            self.metadata['data_hash'] = data_hash
            self.metadata['n_samples'] = len(df)
            self.metadata['n_features'] = len(df.columns)
    
            print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"✓ Data hash: {data_hash[:8]}...")
    
            return df
    
        def validate_data(self, df):
            """Data quality validation"""
            # Data validation using Great Expectations
            df_ge = ge.from_pandas(df)
    
            # Define expectations
            expectations = []
    
            # 1. Missing value check
            for col in df.columns:
                missing_pct = df[col].isnull().mean()
                expectations.append({
                    'column': col,
                    'check': 'missing_values',
                    'value': f"{missing_pct:.2%}"
                })
                if missing_pct > 0.3:
                    print(f"⚠️  Warning: {col} has more than 30% missing values")
    
            # 2. Data type check
            expectations.append({
                'check': 'data_types',
                'dtypes': df.dtypes.to_dict()
            })
    
            # 3. Duplicate check
            n_duplicates = df.duplicated().sum()
            expectations.append({
                'check': 'duplicates',
                'value': n_duplicates
            })
    
            self.metadata['validation'] = expectations
    
            print(f"✓ Data validation complete")
            print(f"  - Duplicate rows: {n_duplicates}")
    
            return df
    
        def preprocess_data(self, df):
            """Data preprocessing"""
            # Handle missing values
            df_clean = df.copy()
    
            # Numeric columns: median imputation
            numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
    
            # Categorical columns: mode imputation
            cat_cols = df_clean.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
    
            print(f"✓ Preprocessing complete")
    
            return df_clean
    
        def save_metadata(self, filepath):
            """Save metadata"""
            with open(filepath, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            print(f"✓ Metadata saved: {filepath}")
    
    # Usage example
    pipeline = DataPipeline('customer_data.csv', 'v1.0.0')
    df = pipeline.load_data()
    df = pipeline.validate_data(df)
    df_clean = pipeline.preprocess_data(df)
    pipeline.save_metadata('data_metadata.json')
    

### 2\. Model Development and Training

Experiment management and ensuring reproducibility:
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Model Development & Training Phase
    
    Purpose:
    - Systematic experiment management
    - Hyperparameter optimization
    - Model versioning
    """
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    class ExperimentManager:
        """Experiment management class"""
    
        def __init__(self, experiment_name):
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
    
        def train_and_log(self, model, X_train, y_train, model_name, params):
            """Train and log model"""
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_params(params)
    
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring='accuracy'
                )
    
                # Training
                model.fit(X_train, y_train)
    
                # Log metrics
                mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
                mlflow.log_metric("cv_std_accuracy", cv_scores.std())
    
                # Save model
                mlflow.sklearn.log_model(model, "model")
    
                print(f"✓ {model_name}")
                print(f"  - CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
                return cv_scores.mean()
    
        def compare_models(self, X_train, y_train):
            """Compare multiple models"""
            models = {
                'LogisticRegression': {
                    'model': LogisticRegression(max_iter=1000),
                    'params': {'C': 1.0, 'max_iter': 1000}
                },
                'RandomForest': {
                    'model': RandomForestClassifier(n_estimators=100, random_state=42),
                    'params': {'n_estimators': 100, 'max_depth': 10}
                },
                'GradientBoosting': {
                    'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'params': {'n_estimators': 100, 'learning_rate': 0.1}
                }
            }
    
            results = {}
            for name, config in models.items():
                score = self.train_and_log(
                    config['model'],
                    X_train,
                    y_train,
                    name,
                    config['params']
                )
                results[name] = score
    
            # Select best model
            best_model = max(results, key=results.get)
            print(f"\n🏆 Best model: {best_model} (accuracy: {results[best_model]:.3f})")
    
            return results
    
    # Usage example
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Run experiment
    exp_manager = ExperimentManager("model_comparison")
    results = exp_manager.compare_models(X_train, y_train)
    

### 3\. Deployment and Operations

Deploying models to production environments:
    
    
    # Requirements:
    # - Python 3.9+
    # - flask>=2.3.0
    # - mlflow>=2.4.0
    # - numpy>=1.24.0, <2.0.0
    # - requests>=2.31.0
    
    """
    Deployment Phase
    
    Purpose:
    - Model API-ification
    - Version control
    - A/B testing support
    """
    
    from flask import Flask, request, jsonify
    import mlflow.pyfunc
    import numpy as np
    import logging
    
    class ModelServer:
        """Model server class"""
    
        def __init__(self, model_uri, model_version):
            """
            Args:
                model_uri: MLflow model URI
                model_version: Model version
            """
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = model_version
            self.prediction_count = 0
    
            # Logging setup
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
        def predict(self, features):
            """Execute prediction"""
            try:
                # Input validation
                if not isinstance(features, (list, np.ndarray)):
                    raise ValueError("Input must be an array")
    
                # Prediction
                prediction = self.model.predict(np.array(features).reshape(1, -1))
    
                # Update count
                self.prediction_count += 1
    
                # Log recording
                self.logger.info(f"Prediction executed #{self.prediction_count}")
    
                return {
                    'prediction': int(prediction[0]),
                    'model_version': self.model_version,
                    'prediction_id': self.prediction_count
                }
    
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                return {'error': str(e)}
    
    # Flask API
    app = Flask(__name__)
    
    # Load model (from MLflow Model Registry in production)
    model_server = ModelServer(
        model_uri="models:/churn_predictor/production",
        model_version="1.0.0"
    )
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Prediction endpoint"""
        data = request.get_json()
        features = data.get('features')
    
        if features is None:
            return jsonify({'error': 'features are required'}), 400
    
        result = model_server.predict(features)
    
        if 'error' in result:
            return jsonify(result), 500
    
        return jsonify(result), 200
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_version': model_server.model_version,
            'total_predictions': model_server.prediction_count
        }), 200
    
    # Sample client
    def sample_client():
        """API client usage example"""
        import requests
    
        # Prediction request
        response = requests.post(
            'http://localhost:5000/predict',
            json={'features': [0.5, 1.2, -0.3, 2.1, 0.8]}
        )
    
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction result: {result['prediction']}")
            print(f"Model version: {result['model_version']}")
        else:
            print(f"Error: {response.json()}")
    
    # if __name__ == '__main__':
    #     app.run(host='0.0.0.0', port=5000)
    

### 4\. Monitoring and Improvement

Model performance monitoring in production:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    """
    Monitoring Phase
    
    Purpose:
    - Continuous monitoring of model performance
    - Data drift detection
    - Alerts and automatic retraining triggers
    """
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    from datetime import datetime, timedelta
    
    class ModelMonitor:
        """Model monitoring class"""
    
        def __init__(self, baseline_data, threshold=0.05):
            """
            Args:
                baseline_data: Baseline data (training data)
                threshold: Drift detection threshold
            """
            self.baseline_data = baseline_data
            self.threshold = threshold
            self.drift_history = []
    
        def detect_data_drift(self, new_data, feature_name):
            """Data drift detection (Kolmogorov-Smirnov test)"""
            baseline_feature = self.baseline_data[feature_name]
            new_feature = new_data[feature_name]
    
            # KS test
            statistic, p_value = stats.ks_2samp(baseline_feature, new_feature)
    
            is_drift = p_value < self.threshold
    
            drift_info = {
                'timestamp': datetime.now(),
                'feature': feature_name,
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': is_drift
            }
    
            self.drift_history.append(drift_info)
    
            if is_drift:
                print(f"⚠️  Data drift detected: {feature_name}")
                print(f"   KS statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    
            return is_drift
    
        def monitor_predictions(self, predictions, actuals=None):
            """Prediction monitoring"""
            monitoring_report = {
                'timestamp': datetime.now(),
                'n_predictions': len(predictions),
                'prediction_distribution': {
                    'mean': np.mean(predictions),
                    'std': np.std(predictions),
                    'min': np.min(predictions),
                    'max': np.max(predictions)
                }
            }
    
            # Calculate accuracy if actuals are available
            if actuals is not None:
                accuracy = np.mean(predictions == actuals)
                monitoring_report['accuracy'] = accuracy
    
                if accuracy < 0.7:  # Example threshold
                    print(f"⚠️  Accuracy degradation detected: {accuracy:.3f}")
                    print("   Retraining recommended")
    
            return monitoring_report
    
        def generate_report(self):
            """Generate monitoring report"""
            if not self.drift_history:
                return "No monitoring data available"
    
            df_drift = pd.DataFrame(self.drift_history)
    
            report = f"""
    === Model Monitoring Report ===
    Period: {df_drift['timestamp'].min()} ~ {df_drift['timestamp'].max()}
    Total checks: {len(df_drift)}
    Drift detections: {df_drift['drift_detected'].sum()}
    
    Features with drift detected:
    {df_drift[df_drift['drift_detected']][['feature', 'p_value']].to_string()}
            """
    
            return report
    
    # Usage example
    from sklearn.datasets import make_classification
    
    # Baseline data (training data)
    X_baseline, _ = make_classification(
        n_samples=1000, n_features=5, random_state=42
    )
    df_baseline = pd.DataFrame(
        X_baseline,
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    # New data (production input data)
    # Add shift to simulate drift
    X_new, _ = make_classification(
        n_samples=500, n_features=5, random_state=43
    )
    X_new[:, 0] += 1.5  # Add shift to feature_0
    df_new = pd.DataFrame(
        X_new,
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    # Monitoring
    monitor = ModelMonitor(df_baseline, threshold=0.05)
    
    print("=== Data Drift Detection ===")
    for col in df_baseline.columns:
        monitor.detect_data_drift(df_new, col)
    
    print("\n" + monitor.generate_report())
    

* * *

## 1.3 Main Components of MLOps

### 1\. Data Management

Data versioning, quality management, and lineage tracking:

Component | Purpose | Example Tools  
---|---|---  
**Data Versioning** | Dataset change history management | DVC, LakeFS, Delta Lake  
**Data Quality** | Data validation and anomaly detection | Great Expectations, Deequ  
**Data Lineage** | Data origin and transformation history | Apache Atlas, Marquez  
**Feature Store** | Feature reuse and consistency | Feast, Tecton  
      
    
    """
    Data versioning implementation example (DVC-style)
    """
    
    import os
    import hashlib
    import json
    from pathlib import Path
    
    class SimpleDataVersioning:
        """Simple data versioning system"""
    
        def __init__(self, storage_dir='.data_versions'):
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(exist_ok=True)
            self.manifest_file = self.storage_dir / 'manifest.json'
            self.manifest = self._load_manifest()
    
        def _load_manifest(self):
            """Load manifest file"""
            if self.manifest_file.exists():
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            return {}
    
        def _save_manifest(self):
            """Save manifest file"""
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
    
        def _compute_hash(self, filepath):
            """Compute file hash"""
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
    
        def add(self, filepath, version_tag):
            """Add data file to version control"""
            filepath = Path(filepath)
    
            if not filepath.exists():
                raise FileNotFoundError(f"{filepath} not found")
    
            # Compute hash
            file_hash = self._compute_hash(filepath)
    
            # Copy to storage
            storage_path = self.storage_dir / f"{version_tag}_{file_hash[:8]}"
            import shutil
            shutil.copy(filepath, storage_path)
    
            # Update manifest
            self.manifest[version_tag] = {
                'original_path': str(filepath),
                'storage_path': str(storage_path),
                'hash': file_hash,
                'size': filepath.stat().st_size,
                'timestamp': str(pd.Timestamp.now())
            }
    
            self._save_manifest()
    
            print(f"✓ Added {filepath.name} as version {version_tag}")
            print(f"  Hash: {file_hash[:8]}...")
    
        def checkout(self, version_tag, output_path=None):
            """Retrieve specific version of data"""
            if version_tag not in self.manifest:
                raise ValueError(f"Version {version_tag} not found")
    
            version_info = self.manifest[version_tag]
            storage_path = Path(version_info['storage_path'])
    
            if output_path is None:
                output_path = version_info['original_path']
    
            import shutil
            shutil.copy(storage_path, output_path)
    
            print(f"✓ Checked out version {version_tag}")
            print(f"  Output: {output_path}")
    
        def list_versions(self):
            """List all versions"""
            if not self.manifest:
                print("No versioned data available")
                return
    
            print("=== Data Version List ===")
            for tag, info in self.manifest.items():
                print(f"\nVersion: {tag}")
                print(f"  Path: {info['original_path']}")
                print(f"  Size: {info['size']:,} bytes")
                print(f"  Hash: {info['hash'][:8]}...")
                print(f"  Created: {info['timestamp']}")
    
    # Usage example (demo)
    # dvc = SimpleDataVersioning()
    # dvc.add('data.csv', 'v1.0.0')
    # dvc.add('data.csv', 'v1.1.0')  # After data update
    # dvc.list_versions()
    # dvc.checkout('v1.0.0', 'data_old.csv')
    

### 2\. Model Management

Managing the entire model lifecycle:

Component | Purpose | Features  
---|---|---  
**Experiment Tracking** | Recording and comparing experiment results | Logging parameters, metrics, and artifacts  
**Model Registry** | Centralized model management | Version control, stage management, approval flow  
**Model Packaging** | Converting to deployable format | Dependency resolution, containerization  
  
### 3\. Infrastructure Management

Scalable and reproducible infrastructure:

Component | Purpose | Example Tools  
---|---|---  
**Containerization** | Ensuring environment consistency | Docker, Kubernetes  
**Orchestration** | Workflow automation | Airflow, Kubeflow, Argo  
**Resource Management** | Efficient use of computational resources | Kubernetes, Ray  
  
### 4\. Governance

Compliance and audit support:

Element | Content  
---|---  
**Model Explainability** | Explaining prediction rationale  
**Bias Detection** | Fairness verification  
**Audit Logs** | Recording all change history  
**Access Control** | Permission management and approval flow  
  
* * *

## 1.4 MLOps Tool Ecosystem

### Experiment Management Tools

Tool | Features | Main Use Cases  
---|---|---  
**MLflow** | Open source, multi-functional | Experiment management, model registry, deployment  
**Weights & Biases** | Real-time visualization, collaboration | Experiment comparison, hyperparameter optimization  
**Neptune.ai** | Specialized in metadata management | Long-term experiment management, team collaboration  
  
### Pipeline Orchestration

Tool | Features | Main Use Cases  
---|---|---  
**Kubeflow** | ML on Kubernetes | End-to-end ML pipelines  
**Apache Airflow** | General-purpose workflow | Data pipelines, scheduling  
**Prefect** | Python-native, modern API | Data flow, error handling  
  
### Model Deployment

Tool | Features | Main Use Cases  
---|---|---  
**BentoML** | Specialized in model serving | REST API, batch inference  
**Seldon Core** | Deployment on Kubernetes | Microservices, A/B testing  
**TensorFlow Serving** | TensorFlow-specific | Fast inference, GPU support  
  
### Monitoring Tools

Tool | Features | Main Use Cases  
---|---|---  
**Evidently** | Data drift detection | Model performance monitoring, report generation  
**Prometheus + Grafana** | General-purpose metrics monitoring | System monitoring, alerts  
**Arize AI** | ML-specialized observability | Model monitoring, root cause analysis  
  
### Integrated Platforms

Platform | Features  
---|---  
**AWS SageMaker** | AWS-native, fully managed  
**Azure ML** | Azure ecosystem integration  
**Google Vertex AI** | GCP service integration, AutoML  
**Databricks** | Data + ML integration, Spark foundation  
  
* * *

## 1.5 MLOps Maturity Model

Framework for evaluating organizational MLOps maturity (proposed by Google):

### Level 0: Manual Process

**Characteristics** :

  * All steps are manual
  * Jupyter Notebook-based development
  * Manual model deployment
  * No reproducibility

**Challenges** :

  * Does not scale
  * Frequent errors
  * Time-consuming deployment
  * No monitoring

### Level 1: ML Pipeline Automation

**Characteristics** :

  * Automated training pipeline
  * Continuous Training (CT)
  * Experiment tracking
  * Use of model registry

**Achievements** :

  * Automatic retraining with new data
  * Model versioning
  * Basic monitoring

    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
    """
    Level 1 implementation example: Automated training pipeline
    """
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import mlflow
    import schedule
    import time
    
    class AutoTrainingPipeline:
        """Automated training pipeline"""
    
        def __init__(self, experiment_name):
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
    
        def create_pipeline(self):
            """Build ML pipeline"""
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
    
        def train(self, X_train, y_train, X_val, y_val):
            """Execute training"""
            with mlflow.start_run():
                # Create pipeline
                pipeline = self.create_pipeline()
    
                # Training
                pipeline.fit(X_train, y_train)
    
                # Evaluation
                train_score = pipeline.score(X_train, y_train)
                val_score = pipeline.score(X_val, y_val)
    
                # Log to MLflow
                mlflow.log_metric("train_accuracy", train_score)
                mlflow.log_metric("val_accuracy", val_score)
                mlflow.sklearn.log_model(pipeline, "model")
    
                # Register to model registry
                if val_score > 0.8:  # Only if above threshold
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/model",
                        "production_model"
                    )
                    print(f"✓ Registered new model (validation accuracy: {val_score:.3f})")
                else:
                    print(f"⚠️  Accuracy below threshold ({val_score:.3f} < 0.8)")
    
                return pipeline
    
        def scheduled_training(self, data_loader, schedule_time="00:00"):
            """Scheduled training"""
            def job():
                print(f"=== Automatic training started: {pd.Timestamp.now()} ===")
                X_train, X_val, y_train, y_val = data_loader()
                self.train(X_train, X_val, y_train, y_val)
    
            # Execute daily at specified time
            schedule.every().day.at(schedule_time).do(job)
    
            print(f"✓ Automatic training scheduled: Daily at {schedule_time}")
    
            # Schedule execution (run as background service in actual environment)
            # while True:
            #     schedule.run_pending()
            #     time.sleep(60)
    
    # Usage example (demo)
    # def load_latest_data():
    #     # Latest data loading logic
    #     return X_train, X_val, y_train, y_val
    #
    # pipeline = AutoTrainingPipeline("auto_training")
    # pipeline.scheduled_training(load_latest_data, "02:00")
    

### Level 2: CI/CD Pipeline Automation

**Characteristics** :

  * Complete automation (from code changes to deployment)
  * CI/CD integration
  * Automated testing (data, model, infrastructure)
  * Comprehensive monitoring

**Achievements** :

  * Automatic deployment of code changes
  * A/B testing
  * Canary deployment
  * Automatic rollback

    
    
    ```mermaid
    graph TB
        subgraph "Level 0: Manual"
            A1[Notebook Development] --> A2[Manual Training]
            A2 --> A3[Manual Deployment]
        end
    
        subgraph "Level 1: ML Pipeline"
            B1[Code Development] --> B2[Automated Training Pipeline]
            B2 --> B3[Model Registry]
            B3 --> B4[Manual Deployment Approval]
            B4 --> B5[Deployment]
        end
    
        subgraph "Level 2: CI/CD"
            C1[Code Change] --> C2[CI: Automated Testing]
            C2 --> C3[Automated Training]
            C3 --> C4[Automated Validation]
            C4 --> C5[CD: Automated Deployment]
            C5 --> C6[Monitoring]
            C6 --> C7{Performance OK?}
            C7 -->|No| C8[Automatic Rollback]
            C7 -->|Yes| C6
        end
    
        style A1 fill:#ffebee
        style B2 fill:#fff3e0
        style C5 fill:#e8f5e9
    ```

### Comparison of Maturity Levels

Aspect | Level 0 | Level 1 | Level 2  
---|---|---|---  
**Deployment Frequency** | Monthly to yearly | Weekly to monthly | Daily to weekly  
**Reproducibility** | Low | Medium | High  
**Automation** | None | Training only | End-to-end  
**Monitoring** | None/manual | Basic | Comprehensive  
**Testing** | None | Model only | All components  
**Application Scale** | 1-2 models | Several models | Many models  
  
* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Necessity of MLOps**

     * 85% of machine learning projects don't reach production
     * MLOps bridges the gap from development to operations
     * Integrated approach of DevOps, DataOps, and ML
  2. **ML Lifecycle**

     * Four phases: data collection/preparation, model development, deployment, monitoring
     * Iterative and continuous process
     * Automation and quality assurance important in each phase
  3. **Main Components**

     * Data management: versioning, quality, lineage
     * Model management: experiment tracking, registry
     * Infrastructure management: containerization, orchestration
     * Governance: explainability, audit, access control
  4. **Tool Ecosystem**

     * Experiment management: MLflow, Weights & Biases
     * Pipelines: Kubeflow, Airflow
     * Deployment: BentoML, Seldon
     * Monitoring: Evidently, Prometheus
  5. **Maturity Model**

     * Level 0: Fully manual (does not scale)
     * Level 1: Training pipeline automation
     * Level 2: Complete CI/CD automation (enterprise-ready)

### MLOps Implementation Best Practices

Principle | Description  
---|---  
**Start Small** | Evolve gradually from Level 0 → Level 1 → Level 2  
**Automation First** | Minimize manual work and reduce errors  
**Monitoring Required** | Continuously monitor performance in production  
**Ensure Reproducibility** | Make all experiments and models reproducible  
**Team Collaboration** | Cooperation between data scientists and engineers  
  
### To the Next Chapter

In Chapter 2, we will learn about **Experiment Management and Model Tracking** in detail:

  * Experiment management with MLflow
  * Hyperparameter optimization
  * Utilizing model registry
  * Visualization and comparison of experiment results
  * Sharing experiments in teams

* * *

## Practice Problems

### Problem 1 (Difficulty: easy)

List and explain three differences between MLOps and DevOps. Focus on machine learning-specific challenges.

Sample Answer

**Answer** :

  1. **Handling Data**

     * **DevOps** : Focuses on code version control
     * **MLOps** : Requires version control of code, data, and models
     * In machine learning, the same code with different data produces different results, making data versioning essential
  2. **Testing Complexity**

     * **DevOps** : Deterministic testing (same input → same output)
     * **MLOps** : Probabilistic testing (model performance, data drift, bias, etc.)
     * Model testing must evaluate not only accuracy but also fairness and interpretability
  3. **Continuous Monitoring**

     * **DevOps** : Monitor system uptime and error rates
     * **MLOps** : Monitor model performance degradation, data drift, and prediction distribution changes
     * Model degradation over time is unavoidable, requiring automatic retraining mechanisms

### Problem 2 (Difficulty: medium)

Improve the following code to implement experiment management following MLOps best practices. Use MLflow to record parameters, metrics, and models.
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Improve the following code to implement experiment managemen
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Load data
    df = pd.read_csv('data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import hashlib
    import json
    
    # MLflow experiment setup
    mlflow.set_experiment("customer_classification")
    
    # Data version calculation
    def compute_data_version(df):
        """Calculate data hash to use as version"""
        data_str = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.md5(data_str).hexdigest()[:8]
    
    # Load data
    df = pd.read_csv('data.csv')
    data_version = compute_data_version(df)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Start experiment
    with mlflow.start_run(run_name="rf_baseline"):
    
        # Define hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
    
        # Data split (ensure reproducibility with fixed seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
    
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
    
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
    
        # Test set evaluation
        y_pred = model.predict(X_test)
    
        # Log multiple metrics
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_pred, average='weighted'),
            'test_f1': f1_score(y_test, y_pred, average='weighted')
        }
    
        mlflow.log_metrics(metrics)
    
        # Log feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")
    
        # Save model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="customer_classifier"
        )
    
        # Set tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("environment", "development")
    
        # Display results
        print("=== Experiment Results ===")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Data Version: {data_version}")
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
        # Confirm model registration
        print(f"\n✓ Model logged to MLflow")
        print(f"✓ Experiment name: customer_classification")
    

**Improvements** :

  1. Manage experiments with MLflow
  2. Record data version
  3. Log all hyperparameters
  4. Record multiple evaluation metrics
  5. Perform cross-validation
  6. Save feature importance
  7. Register to model registry
  8. Ensure reproducibility (random_state, stratify)

### Problem 3 (Difficulty: medium)

Implement a data drift detection system. Use the Kolmogorov-Smirnov test to determine if new data is statistically different from baseline data and output alerts.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - scipy>=1.11.0
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    from datetime import datetime
    import json
    
    class DataDriftMonitor:
        """Data drift monitoring system"""
    
        def __init__(self, baseline_data, threshold=0.05, alert_features=None):
            """
            Args:
                baseline_data: Baseline data (DataFrame)
                threshold: p-value threshold (default: 0.05)
                alert_features: List of features to monitor (all features if None)
            """
            self.baseline_data = baseline_data
            self.threshold = threshold
            self.alert_features = alert_features or baseline_data.columns.tolist()
            self.drift_history = []
            self.baseline_stats = self._compute_baseline_stats()
    
        def _compute_baseline_stats(self):
            """Compute baseline statistics"""
            stats_dict = {}
            for col in self.baseline_data.columns:
                if pd.api.types.is_numeric_dtype(self.baseline_data[col]):
                    stats_dict[col] = {
                        'mean': self.baseline_data[col].mean(),
                        'std': self.baseline_data[col].std(),
                        'min': self.baseline_data[col].min(),
                        'max': self.baseline_data[col].max(),
                        'median': self.baseline_data[col].median()
                    }
            return stats_dict
    
        def detect_drift(self, new_data, feature):
            """Detect drift for single feature"""
            if feature not in self.baseline_data.columns:
                raise ValueError(f"Feature {feature} not found")
    
            # Process only numeric types
            if not pd.api.types.is_numeric_dtype(self.baseline_data[feature]):
                return None
    
            # KS test
            baseline_values = self.baseline_data[feature].dropna()
            new_values = new_data[feature].dropna()
    
            statistic, p_value = stats.ks_2samp(baseline_values, new_values)
    
            is_drift = p_value < self.threshold
    
            # Calculate statistical changes
            baseline_mean = baseline_values.mean()
            new_mean = new_values.mean()
            mean_shift = (new_mean - baseline_mean) / baseline_mean * 100
    
            drift_info = {
                'timestamp': datetime.now().isoformat(),
                'feature': feature,
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': bool(is_drift),
                'baseline_mean': float(baseline_mean),
                'new_mean': float(new_mean),
                'mean_shift_pct': float(mean_shift),
                'n_baseline': len(baseline_values),
                'n_new': len(new_values)
            }
    
            return drift_info
    
        def monitor_all_features(self, new_data):
            """Monitor drift for all features"""
            results = []
            alerts = []
    
            print(f"=== Data Drift Monitoring Execution ===")
            print(f"Time: {datetime.now()}")
            print(f"Features monitored: {len(self.alert_features)}")
            print(f"New data samples: {len(new_data)}\n")
    
            for feature in self.alert_features:
                if feature not in new_data.columns:
                    continue
    
                drift_info = self.detect_drift(new_data, feature)
    
                if drift_info is None:
                    continue
    
                results.append(drift_info)
                self.drift_history.append(drift_info)
    
                # Alert on drift detection
                if drift_info['drift_detected']:
                    alert_msg = (
                        f"⚠️  Drift detected: {feature}\n"
                        f"   KS statistic: {drift_info['ks_statistic']:.4f}\n"
                        f"   p-value: {drift_info['p_value']:.4f}\n"
                        f"   Mean shift: {drift_info['mean_shift_pct']:.2f}%"
                    )
                    alerts.append(alert_msg)
                    print(alert_msg + "\n")
    
            # Summary
            n_drift = sum(r['drift_detected'] for r in results)
            print(f"=== Monitoring Results ===")
            print(f"Drift detected: {n_drift}/{len(results)} features")
    
            if n_drift > len(results) * 0.3:  # Alert if >30%
                print("⚠️  Warning: Drift detected in many features")
                print("   Model retraining recommended")
    
            return results, alerts
    
        def generate_report(self):
            """Generate drift report"""
            if not self.drift_history:
                return "No drift history available"
    
            df_history = pd.DataFrame(self.drift_history)
    
            report = f"""
    === Data Drift Monitoring Report ===
    
    Monitoring period: {df_history['timestamp'].min()} ~ {df_history['timestamp'].max()}
    Total monitoring instances: {len(df_history)}
    Unique features: {df_history['feature'].nunique()}
    
    Drift detection summary:
    {df_history.groupby('feature')['drift_detected'].agg(['sum', 'count']).to_string()}
    
    Top 5 drift detection rates:
    {df_history[df_history['drift_detected']].groupby('feature').size().sort_values(ascending=False).head().to_string()}
    
    Top 5 average shift rates (absolute):
    {df_history.groupby('feature')['mean_shift_pct'].apply(lambda x: abs(x).mean()).sort_values(ascending=False).head().to_string()}
            """
    
            return report
    
        def save_report(self, filepath):
            """Save report as JSON"""
            report_data = {
                'baseline_stats': self.baseline_stats,
                'drift_history': self.drift_history,
                'summary': {
                    'total_checks': len(self.drift_history),
                    'total_drifts': sum(d['drift_detected'] for d in self.drift_history)
                }
            }
    
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
    
            print(f"✓ Report saved: {filepath}")
    
    # Usage example
    from sklearn.datasets import make_classification
    
    # Baseline data (training data)
    X_baseline, _ = make_classification(
        n_samples=1000, n_features=10, random_state=42
    )
    df_baseline = pd.DataFrame(
        X_baseline,
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # New data (with drift)
    X_new, _ = make_classification(
        n_samples=500, n_features=10, random_state=43
    )
    # Add shift to some features
    X_new[:, 0] += 2.0  # Large shift to feature_0
    X_new[:, 3] += 0.5  # Small shift to feature_3
    
    df_new = pd.DataFrame(
        X_new,
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Execute drift monitoring
    monitor = DataDriftMonitor(df_baseline, threshold=0.05)
    results, alerts = monitor.monitor_all_features(df_new)
    
    # Generate report
    print("\n" + monitor.generate_report())
    
    # Save report
    monitor.save_report('drift_report.json')
    

**Example output** :
    
    
    === Data Drift Monitoring Execution ===
    Time: 2025-10-21 10:30:45.123456
    Features monitored: 10
    New data samples: 500
    
    ⚠️  Drift detected: feature_0
       KS statistic: 0.8920
       p-value: 0.0000
       Mean shift: 412.34%
    
    ⚠️  Drift detected: feature_3
       KS statistic: 0.2145
       p-value: 0.0023
       Mean shift: 87.56%
    
    === Monitoring Results ===
    Drift detected: 2/10 features
    

### Problem 4 (Difficulty: hard)

Implement an automated training pipeline for MLOps maturity Level 1. Include the following features: 

  * Automatic data acquisition
  * Preprocessing pipeline
  * Model training
  * Performance evaluation and threshold judgment
  * Recording to MLflow
  * Register to model registry only when conditions are met

Sample Answer

Due to length constraints, please refer to the detailed implementation in the Japanese version. The key implementation points include:

  * AutoMLPipeline class with experiment management
  * Automated data loading with validation
  * Preprocessing and model pipeline creation
  * Cross-validation and threshold-based model registration
  * Comprehensive MLflow logging
  * Scheduled training support

### Problem 5 (Difficulty: hard)

Create a checklist to evaluate MLOps maturity level. Make it possible to determine whether an organization is at Level 0, 1, or 2.

Sample Answer

The MLOpsMaturityAssessment class provides:

  * Criteria for each maturity level (Data Management, Model Development, Deployment, Monitoring)
  * Interactive assessment with yes/no questions
  * Scoring system (0-100%) for each level
  * Level determination based on ≥70% threshold
  * Recommendations for next steps based on current level
  * Visual progress bars in assessment report

* * *

## References

  1. Géron, A. (2022). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (3rd ed.). O'Reilly Media.
  2. Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. _IEEE Access_ , 11, 31866-31879.
  3. Google Cloud. (2023). _MLOps: Continuous delivery and automation pipelines in machine learning_. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
  4. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  5. Treveil, M., et al. (2020). _Introducing MLOps_. O'Reilly Media.
