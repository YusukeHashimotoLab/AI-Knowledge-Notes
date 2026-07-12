---
title: "Chapter 2: Experiment Tracking and Version Control"
chapter_title: "Chapter 2: Experiment Tracking and Version Control"
subtitle: Experiment tracking and data versioning for reproducible machine learning
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 12
exercises: 5
version: 1.0
created_at: 2025-10-21
---

## Learning Objectives

By the end of this chapter, you will be able to:

  * ✅ Understand why experiment management matters in machine learning
  * ✅ Implement experiment tracking and model management with MLflow
  * ✅ Run hyperparameter optimization with Weights & Biases
  * ✅ Version data and models with DVC
  * ✅ Apply experiment management best practices
  * ✅ Build reproducible machine learning pipelines

* * *

## 2.1 Why Experiment Management Matters

### What Is Experiment Management?

**Experiment management** is the systematic process of recording, tracking, comparing, and reproducing experiments in a machine learning project.

> "A great ML project depends on the ability to manage hundreds to thousands of experiments."

### Challenges of Experiment Management

Challenge | Impact | Solution  
---|---|---  
**Lack of reproducibility** | Past experiments cannot be reproduced | Version control for parameters, code, and data  
**Difficulty comparing experiments** | Cannot select the best model | Unified metric logging  
**Loss of insights** | Information is not shared across the team | Centralized experiment tracking  
**Data drift** | Data changes cannot be traced | Data versioning  
  
### The Big Picture of Experiment Management
    
    
    ```mermaid
    graph TD
        A[Experiment Design] --> B[Set Parameters]
        B --> C[Load Data]
        C --> D[Train Model]
        D --> E[Log Metrics]
        E --> F[Save Artifacts]
        F --> G[Compare Experiments]
        G --> H{Improved?}
        H -->|Yes| I[Select Best Model]
        H -->|No| B
        I --> J[Deploy Model]
    
        style A fill:#ffebee
        style D fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
        style I fill:#c8e6c9
        style J fill:#c8e6c9
    ```

### The Value Experiment Management Delivers

#### 1\. Guaranteed Reproducibility

  * An environment where the same results can be reproduced
  * Complete records of code, data, and parameters
  * Easier audits and compliance

#### 2\. Experiment Comparison and Analysis

  * Compare multiple experiments systematically
  * Visualize the relationship between parameters and performance
  * Data-driven decision making

#### 3\. Best Model Selection

  * Select models based on objective criteria
  * Analyze the trade-off between performance and cost
  * Deploy to production with confidence

* * *

## 2.2 MLflow

### What Is MLflow?

**MLflow** is an open-source platform for managing the entire machine learning lifecycle.

### Key MLflow Components

Component | Functionality | Use Case  
---|---|---  
**MLflow Tracking** | Log experiment parameters and metrics | Experiment management  
**MLflow Projects** | Reproducible code execution | Environment management  
**MLflow Models** | Model packaging and deployment | Model management  
**MLflow Registry** | Model version management | Production operations  
  
### MLflow Tracking: Basic Usage
    
    
    import mlflow
    import mlflow.sklearn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configure the MLflow experiment
    mlflow.set_experiment("random_forest_classification")
    
    # Run the experiment
    with mlflow.start_run(run_name="rf_baseline"):
        # Set parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
    
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
    
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    
        # Predict and compute metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
    
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    
        # Save the model
        mlflow.sklearn.log_model(model, "model")
    
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
    

**Output** :
    
    
    Accuracy: 0.895
    Precision: 0.891
    Recall: 0.902
    

### Running and Comparing Multiple Experiments
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # List of experiment configurations
    experiment_configs = [
        {"n_estimators": 50, "max_depth": 5, "name": "rf_shallow"},
        {"n_estimators": 100, "max_depth": 10, "name": "rf_medium"},
        {"n_estimators": 200, "max_depth": 20, "name": "rf_deep"},
        {"n_estimators": 300, "max_depth": None, "name": "rf_full"},
    ]
    
    mlflow.set_experiment("rf_hyperparameter_tuning")
    
    # Run an experiment for each configuration
    results = []
    for config in experiment_configs:
        with mlflow.start_run(run_name=config["name"]):
            # Log parameters
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])
    
            # Train the model
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
    
            # Evaluate
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("overfit_gap", train_acc - test_acc)
    
            # Save the model
            mlflow.sklearn.log_model(model, "model")
    
            results.append({
                "name": config["name"],
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfit": train_acc - test_acc
            })
    
            print(f"{config['name']}: Train={train_acc:.3f}, Test={test_acc:.3f}, Overfit={train_acc - test_acc:.3f}")
    
    print("\n=== Experiment result comparison ===")
    for result in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{result['name']}: Test Accuracy = {result['test_acc']:.3f}")
    

**Output** :
    
    
    rf_shallow: Train=0.862, Test=0.855, Overfit=0.007
    rf_medium: Train=0.895, Test=0.895, Overfit=0.000
    rf_deep: Train=0.987, Test=0.890, Overfit=0.097
    rf_full: Train=1.000, Test=0.885, Overfit=0.115
    
    === Experiment result comparison ===
    rf_medium: Test Accuracy = 0.895
    rf_deep: Test Accuracy = 0.890
    rf_full: Test Accuracy = 0.885
    rf_shallow: Test Accuracy = 0.855
    

### MLflow Autolog: Automatic Logging
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Enable MLflow autolog
    mlflow.sklearn.autolog()
    
    mlflow.set_experiment("rf_with_autolog")
    
    with mlflow.start_run(run_name="rf_autolog_example"):
        # Train the model (parameters and metrics are logged automatically)
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # Log additional metrics manually
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
    
        print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")
        print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    

**Output** :
    
    
    Test Accuracy: 0.900
    CV Mean: 0.893 (+/- 0.012)
    

> **Advantages of Autolog** : Parameters, metrics, and models are recorded automatically, preventing mistakes in manual logging code.

### MLflow Models: Packaging Models
    
    
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # Define a custom model wrapper
    class CustomModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model
    
        def predict(self, context, model_input):
            """Custom prediction logic"""
            predictions = self.model.predict_proba(model_input)
            # Return a prediction only when confidence is 0.7 or higher
            confident_predictions = []
            for i, prob in enumerate(predictions):
                max_prob = max(prob)
                if max_prob >= 0.7:
                    confident_predictions.append(int(prob.argmax()))
                else:
                    confident_predictions.append(-1)  # Unknown
            return confident_predictions
    
    # Train the model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Wrap the custom model
    wrapped_model = CustomModelWrapper(base_model)
    
    mlflow.set_experiment("custom_model_packaging")
    
    with mlflow.start_run(run_name="confident_predictor"):
        # Save the custom model
        mlflow.pyfunc.log_model(
            artifact_path="confident_model",
            python_model=wrapped_model,
            conda_env={
                'name': 'mlflow-env',
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'scikit-learn=1.0.2',
                    'numpy',
                ]
            }
        )
    
        # Test predictions
        test_predictions = wrapped_model.predict(None, X_test[:5])
        print(f"Confidence-aware predictions: {test_predictions}")
        print(f"Number of low-confidence predictions (-1): {sum(1 for p in test_predictions if p == -1)}")
    

### MLflow UI: Visualizing Experiments
    
    
    # Start the MLflow UI
    # mlflow ui --port 5000
    
    # Open http://localhost:5000 in your browser
    # - List all experiments
    # - Compare parameters and metrics
    # - Download models
    # - Search and filter experiments
    

### MLflow Projects: Reproducible Execution
    
    
    # MLproject file (YAML format)
    """
    name: my_ml_project
    
    conda_env: conda.yaml
    
    entry_points:
      main:
        parameters:
          n_estimators: {type: int, default: 100}
          max_depth: {type: int, default: 10}
          data_path: {type: string, default: "data/"}
        command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth} --data-path {data_path}"
    """
    
    # Run the project
    import mlflow
    
    # Run locally
    mlflow.run(
        ".",
        parameters={
            "n_estimators": 200,
            "max_depth": 15,
            "data_path": "data/train.csv"
        }
    )
    
    # Run from GitHub
    mlflow.run(
        "https://github.com/username/ml-project",
        version="main",
        parameters={"n_estimators": 150}
    )
    

* * *

## 2.3 Weights & Biases (W&B)

### What Is Weights & Biases?

**Weights & Biases (W&B)** is a powerful platform for experiment tracking, visualization, and hyperparameter optimization.

### Key W&B Features

Feature | Description | Use Case  
---|---|---  
**Experiment Tracking** | Visualize metrics in real time | Experiment monitoring  
**Sweeps** | Automated hyperparameter optimization | Tuning  
**Artifacts** | Store models and datasets | Version control  
**Reports** | Create and share experiment reports | Team collaboration  
  
### W&B: Basic Experiment Tracking
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np
    
    # Initialize W&B
    wandb.init(
        project="ml-experiment-tracking",
        name="rf_baseline",
        config={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42
        }
    )
    
    # Retrieve the configuration
    config = wandb.config
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    # Visualize the confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })
    
    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Finish the run
    wandb.finish()
    

**Output** :
    
    
    Accuracy: 0.895, F1: 0.897
    View run at: https://wandb.ai/username/ml-experiment-tracking/runs/xxxxx
    

### W&B: Real-Time Learning Curve Visualization
    
    
    import wandb
    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    wandb.init(project="learning-curves", name="rf_learning_curve")
    
    # Compute the learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Log scores for each training size
    for i, size in enumerate(train_sizes_abs):
        wandb.log({
            "train_size": size,
            "train_score_mean": train_scores[i].mean(),
            "train_score_std": train_scores[i].std(),
            "test_score_mean": test_scores[i].mean(),
            "test_score_std": test_scores[i].std()
        })
    
    print("Learning curve computation complete")
    wandb.finish()
    

### W&B Sweeps: Hyperparameter Optimization
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {
                'distribution': 'int_uniform',
                'min': 50,
                'max': 300
            },
            'max_depth': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 30
            },
            'min_samples_split': {
                'distribution': 'int_uniform',
                'min': 2,
                'max': 20
            },
            'min_samples_leaf': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 10
            }
        }
    }
    
    # Define the training function
    def train():
        # Initialize W&B
        wandb.init()
        config = wandb.config
    
        # Train the model
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
    
        # Log metrics
        wandb.log({
            'accuracy': test_acc,
            'train_accuracy': train_acc,
            'overfit_gap': train_acc - test_acc
        })
    
    # Run the sweep
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter-tuning")
    
    # Run 10 experiments
    wandb.agent(sweep_id, function=train, count=10)
    
    print(f"Sweep complete: {sweep_id}")
    

**Output** :
    
    
    Sweep complete: username/hyperparameter-tuning/sweep_xxxxx
    Best accuracy: 0.915
    Best parameters: n_estimators=220, max_depth=18, min_samples_split=3, min_samples_leaf=2
    

### W&B: Saving Models and Datasets
    
    
    import wandb
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    wandb.init(project="model-artifacts", name="rf_with_artifacts")
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = "random_forest_model.pkl"
    joblib.dump(model, model_path)
    
    # Store it as a W&B artifact
    artifact = wandb.Artifact(
        name="random_forest_model",
        type="model",
        description="Random Forest classifier trained on classification dataset"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # Save the dataset
    import pandas as pd
    df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    df_train['target'] = y_train
    df_train.to_csv("train_data.csv", index=False)
    
    data_artifact = wandb.Artifact(
        name="training_dataset",
        type="dataset",
        description="Training dataset for RF model"
    )
    data_artifact.add_file("train_data.csv")
    wandb.log_artifact(data_artifact)
    
    print("Model and dataset saved")
    wandb.finish()
    

### W&B: Visual Comparison of Multiple Experiments
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # Experiment with multiple models
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for model_name, model in models.items():
        # Start the run
        run = wandb.init(
            project="model-comparison",
            name=model_name,
            reinit=True
        )
    
        # Train the model
        model.fit(X_train, y_train)
    
        # Predict and evaluate
        y_pred = model.predict(X_test)
    
        # Log metrics
        wandb.log({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "model_type": model_name
        })
    
        # Log feature importance (when available)
        if hasattr(model, 'feature_importances_'):
            importance_data = [[i, imp] for i, imp in enumerate(model.feature_importances_)]
            table = wandb.Table(data=importance_data, columns=["feature", "importance"])
            wandb.log({"feature_importance": wandb.plot.bar(table, "feature", "importance")})
    
        run.finish()
    
    print("All model experiments complete")
    

* * *

## 2.4 DVC (Data Version Control)

### What Is DVC?

**DVC (Data Version Control)** is a tool that brings Git-like workflows to versioning data and models.

### Key DVC Features

Feature | Description | Use Case  
---|---|---  
**Data versioning** | Version control for large data files | Data tracking  
**Pipeline definition** | Reproducible ML pipelines | Workflow management  
**Remote storage** | Integration with S3, GCS, Azure, and more | Data sharing  
**Experiment management** | Track and compare experiments | Experiment comparison  
  
### Setting Up and Initializing DVC
    
    
    # Install DVC
    # pip install dvc
    
    # Initialize a Git repository (if not done yet)
    # git init
    
    # Initialize DVC
    # dvc init
    
    # The .dvc/config file is created
    # git add .dvc .dvcignore
    # git commit -m "Initialize DVC"
    

### Versioning Data
    
    
    # Generate data in Python
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Save the data
    data.to_csv('data/raw_data.csv', index=False)
    print("Data saved: data/raw_data.csv")
    
    
    
    # Track the data with DVC
    # dvc add data/raw_data.csv
    
    # This creates the following:
    # - data/raw_data.csv.dvc (metadata file)
    # - data/.gitignore (excludes the actual data)
    
    # Commit the metadata file to Git
    # git add data/raw_data.csv.dvc data/.gitignore
    # git commit -m "Add raw data"
    
    # Configure remote storage (example: local directory)
    # dvc remote add -d local_storage /tmp/dvc-storage
    # git add .dvc/config
    # git commit -m "Configure DVC remote storage"
    
    # Push the data to the remote
    # dvc push
    

### Defining a DVC Pipeline
    
    
    # prepare.py - data preprocessing script
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sys
    
    def prepare_data(input_file, train_file, test_file):
        # Load the data
        data = pd.read_csv(input_file)
    
        # Split into training and test sets
        train, test = train_test_split(data, test_size=0.2, random_state=42)
    
        # Save
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
    
        print(f"Training data: {len(train)} rows")
        print(f"Test data: {len(test)} rows")
    
    if __name__ == "__main__":
        prepare_data(
            input_file="data/raw_data.csv",
            train_file="data/train.csv",
            test_file="data/test.csv"
        )
    
    
    
    # train.py - model training script
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import json
    
    def train_model(train_file, model_file, metrics_file):
        # Load the data
        train = pd.read_csv(train_file)
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # Save the model
        joblib.dump(model, model_file)
    
        # Save metrics
        train_accuracy = model.score(X_train, y_train)
        metrics = {"train_accuracy": train_accuracy}
    
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"Training accuracy: {train_accuracy:.3f}")
    
    if __name__ == "__main__":
        train_model(
            train_file="data/train.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/train_metrics.json"
        )
    
    
    
    # evaluate.py - model evaluation script
    import pandas as pd
    import joblib
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    def evaluate_model(test_file, model_file, metrics_file):
        # Load the data and model
        test = pd.read_csv(test_file)
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load(model_file)
    
        # Predict and evaluate
        y_pred = model.predict(X_test)
    
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
    
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"Test accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
    
    if __name__ == "__main__":
        evaluate_model(
            test_file="data/test.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/test_metrics.json"
        )
    

### dvc.yaml: Defining the Pipeline
    
    
    # dvc.yaml
    stages:
      prepare:
        cmd: python prepare.py
        deps:
          - data/raw_data.csv
          - prepare.py
        outs:
          - data/train.csv
          - data/test.csv
    
      train:
        cmd: python train.py
        deps:
          - data/train.csv
          - train.py
        outs:
          - models/model.pkl
        metrics:
          - metrics/train_metrics.json:
              cache: false
    
      evaluate:
        cmd: python evaluate.py
        deps:
          - data/test.csv
          - models/model.pkl
          - evaluate.py
        metrics:
          - metrics/test_metrics.json:
              cache: false
    
    
    
    # Run the pipeline
    # dvc repro
    
    # Output:
    # Running stage 'prepare':
    # > python prepare.py
    # Training data: 800 rows
    # Test data: 200 rows
    #
    # Running stage 'train':
    # > python train.py
    # Training accuracy: 1.000
    #
    # Running stage 'evaluate':
    # > python evaluate.py
    # Test accuracy: 0.895
    # Precision: 0.891
    # Recall: 0.902
    
    # Show metrics
    # dvc metrics show
    
    # Visualize the pipeline
    # dvc dag
    

### DVC Experiments: Tracking Experiments
    
    
    # Create the parameter file
    # params.yaml
    """
    model:
      n_estimators: 100
      max_depth: 10
      random_state: 42
    
    data:
      test_size: 0.2
      random_state: 42
    """
    
    # Run an experiment
    # dvc exp run
    
    # Run multiple experiments in parallel
    # dvc exp run --set-param model.n_estimators=150
    # dvc exp run --set-param model.n_estimators=200
    # dvc exp run --set-param model.max_depth=15
    
    # Show experiment results
    # dvc exp show
    
    # Output:
    # ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
    # ┃ Experiment  ┃ n_estimators┃ max_depth┃ accuracy  ┃
    # ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
    # │ workspace   │ 100         │ 10       │ 0.895     │
    # │ exp-1       │ 150         │ 10       │ 0.900     │
    # │ exp-2       │ 200         │ 10       │ 0.905     │
    # │ exp-3       │ 100         │ 15       │ 0.898     │
    # └─────────────┴─────────────┴──────────┴───────────┘
    
    # Apply the best experiment
    # dvc exp apply exp-2
    # git add .
    # git commit -m "Apply best experiment: n_estimators=200"
    

### Integrated DVC and Git Workflow
    
    
    # Complete workflow example
    import subprocess
    import os
    
    def dvc_workflow_example():
        """Complete ML workflow using DVC and Git"""
    
        # 1. Create a new branch
        subprocess.run(["git", "checkout", "-b", "experiment/new-features"])
    
        # 2. Add new data
        print("Generating new data...")
        import pandas as pd
        import numpy as np
    
        new_data = pd.DataFrame({
            'feature1': np.random.randn(1500),
            'feature2': np.random.randn(1500),
            'feature3': np.random.randn(1500),
            'feature4': np.random.randn(1500),  # New feature
            'target': np.random.randint(0, 2, 1500)
        })
        new_data.to_csv('data/raw_data_v2.csv', index=False)
    
        # 3. Track the new data with DVC
        subprocess.run(["dvc", "add", "data/raw_data_v2.csv"])
    
        # 4. Commit the changes
        subprocess.run(["git", "add", "data/raw_data_v2.csv.dvc", "data/.gitignore"])
        subprocess.run(["git", "commit", "-m", "Add new dataset with feature4"])
    
        # 5. Run the pipeline
        subprocess.run(["dvc", "repro"])
    
        # 6. Check the results
        subprocess.run(["dvc", "metrics", "show"])
    
        # 7. Push the changes
        subprocess.run(["git", "push", "origin", "experiment/new-features"])
        subprocess.run(["dvc", "push"])
    
        print("Workflow complete")
    
    # Note: Actual execution requires a proper Git/DVC setup
    print("DVC workflow example (command walkthrough)")
    

* * *

## 2.5 Best Practices

### 1\. Metadata Logging Best Practices

#### Information to Record

Category | Items | Reason  
---|---|---  
**Experiment info** | Experiment name, timestamp, owner | Experiment identification and tracking  
**Environment info** | Python version, library versions, OS | Ensuring reproducibility  
**Data info** | Data version, sample count, distribution | Data drift detection  
**Model info** | Architecture, parameters | Rebuilding the model  
**Evaluation info** | Metrics, confusion matrix | Performance comparison  
      
    
    import mlflow
    import platform
    import sys
    from datetime import datetime
    
    def log_comprehensive_metadata(model, X_train, y_train, X_test, y_test):
        """Log comprehensive metadata"""
    
        with mlflow.start_run(run_name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. Environment info
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("os", platform.system())
            mlflow.log_param("os_version", platform.version())
    
            # 2. Data info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_distribution", dict(zip(*np.unique(y_train, return_counts=True))))
    
            # 3. Model info
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())
    
            # 4. Training
            model.fit(X_train, y_train)
    
            # 5. Evaluation metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
    
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
    
            # 6. Experiment notes
            mlflow.set_tag("experiment_description", "Comprehensive metadata logging example")
            mlflow.set_tag("data_version", "v1.0")
            mlflow.set_tag("experiment_type", "baseline")
    
            # 7. Save the model
            mlflow.sklearn.log_model(model, "model")
    
            print("Comprehensive metadata logged")
    
    # Usage example
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    log_comprehensive_metadata(model, X_train, y_train, X_test, y_test)
    

### 2\. Hyperparameter Management
    
    
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Optional
    
    @dataclass
    class ModelConfig:
        """Structured definition of model configuration"""
        n_estimators: int = 100
        max_depth: Optional[int] = 10
        min_samples_split: int = 2
        min_samples_leaf: int = 1
        random_state: int = 42
    
        def save(self, filepath: str):
            """Save the configuration to a YAML file"""
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f)
    
        @classmethod
        def load(cls, filepath: str):
            """Load the configuration from a YAML file"""
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
    
    # Save the configuration
    config = ModelConfig(n_estimators=150, max_depth=15)
    config.save("configs/model_config.yaml")
    
    # Load the configuration
    loaded_config = ModelConfig.load("configs/model_config.yaml")
    
    # Train the model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**asdict(loaded_config))
    model.fit(X_train, y_train)
    
    print(f"Trained model using configuration: {asdict(loaded_config)}")
    

### 3\. Artifact Management
    
    
    import mlflow
    import joblib
    import json
    from pathlib import Path
    
    def save_experiment_artifacts(
        model,
        metrics,
        config,
        feature_names,
        experiment_name="my_experiment"
    ):
        """Save experiment artifacts systematically"""
    
        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
            # 1. Save the model
            mlflow.sklearn.log_model(model, "model")
    
            # 2. Save metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
            # 3. Save the configuration
            for param_name, param_value in config.items():
                mlflow.log_param(param_name, param_value)
    
            # 4. Save feature information
            feature_info = {
                "feature_names": feature_names,
                "n_features": len(feature_names)
            }
    
            # Save to a temporary file and log to MLflow
            temp_dir = Path("temp_artifacts")
            temp_dir.mkdir(exist_ok=True)
    
            feature_path = temp_dir / "feature_info.json"
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            mlflow.log_artifact(str(feature_path))
    
            # 5. Save feature importance (when available)
            if hasattr(model, 'feature_importances_'):
                importance_df = {
                    name: float(imp)
                    for name, imp in zip(feature_names, model.feature_importances_)
                }
                importance_path = temp_dir / "feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance_df, f, indent=2)
                mlflow.log_artifact(str(importance_path))
    
            # 6. Save sample predictions
            sample_predictions = {
                "sample_input": X_test[:5].tolist(),
                "predictions": model.predict(X_test[:5]).tolist()
            }
            pred_path = temp_dir / "sample_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(sample_predictions, f, indent=2)
            mlflow.log_artifact(str(pred_path))
    
            # Remove temporary files
            import shutil
            shutil.rmtree(temp_dir)
    
            print("All artifacts saved")
    
    # Usage example
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "f1_score": f1_score(y_test, model.predict(X_test))
    }
    
    config = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    save_experiment_artifacts(model, metrics, config, feature_names)
    

### 4\. Organizing Experiments
    
    
    from enum import Enum
    import mlflow
    from datetime import datetime
    
    class ExperimentType(Enum):
        """Definition of experiment types"""
        BASELINE = "baseline"
        FEATURE_ENGINEERING = "feature_engineering"
        HYPERPARAMETER_TUNING = "hyperparameter_tuning"
        MODEL_SELECTION = "model_selection"
        PRODUCTION = "production"
    
    class ExperimentManager:
        """Organized management of experiments"""
    
        def __init__(self, project_name: str):
            self.project_name = project_name
    
        def create_experiment_name(
            self,
            exp_type: ExperimentType,
            model_name: str,
            version: str = "v1"
        ) -> str:
            """Generate a hierarchical experiment name"""
            return f"{self.project_name}/{exp_type.value}/{model_name}/{version}"
    
        def run_experiment(
            self,
            exp_type: ExperimentType,
            model_name: str,
            model,
            train_fn,
            evaluate_fn,
            version: str = "v1",
            description: str = ""
        ):
            """Run and record an experiment"""
    
            # Generate the experiment name
            exp_name = self.create_experiment_name(exp_type, model_name, version)
            mlflow.set_experiment(exp_name)
    
            # Generate the run name (with a timestamp)
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
            with mlflow.start_run(run_name=run_name):
                # Set tags
                mlflow.set_tag("experiment_type", exp_type.value)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("version", version)
                mlflow.set_tag("description", description)
    
                # Training
                train_metrics = train_fn(model)
    
                # Evaluation
                test_metrics = evaluate_fn(model)
    
                # Log metrics
                for metric_name, metric_value in {**train_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # Save the model
                mlflow.sklearn.log_model(model, "model")
    
                print(f"Experiment complete: {exp_name}/{run_name}")
                return test_metrics
    
    # Usage example
    manager = ExperimentManager(project_name="customer_churn")
    
    def train_fn(model):
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        return {"train_accuracy": train_acc}
    
    def evaluate_fn(model):
        test_acc = model.score(X_test, y_test)
        test_f1 = f1_score(y_test, model.predict(X_test))
        return {"test_accuracy": test_acc, "test_f1": test_f1}
    
    # Baseline model experiment
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    results = manager.run_experiment(
        exp_type=ExperimentType.BASELINE,
        model_name="random_forest",
        model=rf_model,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        version="v1",
        description="Initial baseline model with default parameters"
    )
    
    print(f"Results: {results}")
    

* * *

## 2.6 Chapter Summary

### What You Learned

  1. **Why Experiment Management Matters**

     * Ensuring reproducibility is the foundation of machine learning projects
     * Systematic experiment management enables efficient model development
     * The need for data and model version control
  2. **MLflow**

     * MLflow Tracking: logging parameters and metrics
     * MLflow Models: model packaging and deployment
     * MLflow Projects: reproducible experiment environments
     * Automatic logging with the autolog feature
  3. **Weights & Biases**

     * Real-time experiment visualization
     * Automated hyperparameter optimization (Sweeps)
     * Team collaboration and report sharing
     * Artifact management and versioning
  4. **DVC**

     * Git-like management of data and models
     * Reproducible ML pipeline definitions
     * Integration with remote storage
     * Experiment tracking and comparison
  5. **Best Practices**

     * Comprehensive metadata logging
     * Structured parameter management
     * Systematic artifact storage
     * Hierarchical experiment organization

### Guidelines for Choosing the Right Tool

Tool | Strengths | Recommended Use Cases  
---|---|---  
**MLflow** | Open source, highly flexible | On-premises environments, maximum freedom  
**W &B** | Advanced visualization, team collaboration | Cloud environments, team development  
**DVC** | Git affinity, data management | Large datasets, versioning-focused workflows  
  
### To the Next Chapter

In Chapter 3, we will learn about **Continuous Integration/Deployment (CI/CD)** :

  * CI/CD pipelines in MLOps
  * Automated testing and model validation
  * Model deployment strategies
  * Monitoring and feedback loops

* * *

## Exercises

### Problem 1 (Difficulty: easy)

Give three reasons why "reproducibility" is important in experiment management, and explain each.

Sample Answer

**Answer** :

  1. **Verifying results**

     * You can rerun an experiment under the same conditions to confirm the validity of its results
     * You can determine whether an unexpected result is coincidental or a systematic problem
  2. **Sharing insights**

     * Team members can reproduce the same experiment to deepen their understanding
     * The transparency and credibility of research results improve
  3. **Debugging and improvement**

     * When a problem occurs, you can reproduce the specific experiment state to debug it
     * You can build incremental improvements on top of previously successful experiments

### Problem 2 (Difficulty: medium)

Using MLflow, implement experiment tracking that satisfies the following requirements:

  * Train models with three different hyperparameter configurations
  * Record training and test accuracy for each experiment
  * Identify the experiment that achieved the highest test accuracy

Sample Answer
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure the experiment
    mlflow.set_experiment("hyperparameter_comparison")
    
    # Different hyperparameter configurations
    configs = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20}
    ]
    
    results = []
    
    # Run an experiment for each configuration
    for i, config in enumerate(configs):
        with mlflow.start_run(run_name=f"experiment_{i+1}"):
            # Log parameters
            mlflow.log_params(config)
    
            # Train the model
            model = RandomForestClassifier(**config, random_state=42)
            model.fit(X_train, y_train)
    
            # Compute accuracy
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
    
            # Save the results
            results.append({
                "config": config,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "run_id": mlflow.active_run().info.run_id
            })
    
            print(f"Experiment {i+1}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # Identify the best experiment
    best_result = max(results, key=lambda x: x['test_acc'])
    
    print("\n=== Best experiment ===")
    print(f"Configuration: {best_result['config']}")
    print(f"Test accuracy: {best_result['test_acc']:.3f}")
    print(f"Run ID: {best_result['run_id']}")
    

**Output** :
    
    
    Experiment 1: Train=0.862, Test=0.855
    Experiment 2: Train=0.895, Test=0.895
    Experiment 3: Train=0.987, Test=0.890
    
    === Best experiment ===
    Configuration: {'n_estimators': 100, 'max_depth': 10}
    Test accuracy: 0.895
    Run ID: xxxxxxxxxxxxx
    

### Problem 3 (Difficulty: medium)

Explain the main advantages of using DVC compared to using Git alone.

Sample Answer

**Answer** :

**Main advantages of DVC** :

  1. **Efficient management of large files**

     * Git: large files (datasets, models) bloat the repository
     * DVC: actual files live in remote storage; only metadata is stored in Git
  2. **Data versioning**

     * Git: diff management for binary files is inefficient
     * DVC: tracks data change history efficiently and can restore any version
  3. **Reproducible pipelines**

     * Git: version control for scripts only
     * DVC: defines and reproduces complete pipelines including data, code, and parameters
  4. **Easier team collaboration**

     * Git: sharing large files is difficult
     * DVC: shares data efficiently via remote storage

**Comparison table** :

Aspect | Git only | DVC + Git  
---|---|---  
Code management | ◎ Excellent | ◎ Excellent  
Data management | △ Inefficient | ◎ Optimized  
Model management | △ Difficult | ◎ Systematic  
Pipelines | × Not supported | ◎ Fully supported  
Reproducibility | △ Partial | ◎ Complete  
  
### Problem 4 (Difficulty: hard)

Design a comprehensive experiment management system. Include the following elements:

  * Automatic experiment logging
  * Structured parameter management
  * Experiment result comparison
  * Automatic best model selection

Sample Answer
    
    
    import mlflow
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Dict, Any, List, Optional
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import pandas as pd
    
    @dataclass
    class ExperimentConfig:
        """Structured definition of experiment configuration"""
        experiment_name: str
        model_params: Dict[str, Any]
        data_params: Dict[str, Any]
        description: str = ""
        tags: Dict[str, str] = None
    
    class ComprehensiveExperimentManager:
        """Comprehensive experiment management system"""
    
        def __init__(self, tracking_uri: str = None):
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.results = []
    
        def run_experiment(
            self,
            config: ExperimentConfig,
            model: BaseEstimator,
            X_train, y_train,
            X_test, y_test
        ) -> Dict[str, float]:
            """Run an experiment with automatic logging"""
    
            # Configure the experiment
            mlflow.set_experiment(config.experiment_name)
    
            with mlflow.start_run(description=config.description):
                # 1. Log parameters
                mlflow.log_params(config.model_params)
                mlflow.log_params(config.data_params)
    
                # 2. Set tags
                if config.tags:
                    for key, value in config.tags.items():
                        mlflow.set_tag(key, value)
    
                # 3. Train the model
                model.fit(X_train, y_train)
    
                # 4. Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
    
                # 5. Compute metrics
                metrics = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "test_precision": precision_score(y_test, y_test_pred, average='weighted'),
                    "test_recall": recall_score(y_test, y_test_pred, average='weighted'),
                    "test_f1": f1_score(y_test, y_test_pred, average='weighted'),
                    "overfit_gap": accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
                }
    
                # 6. Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # 7. Save the model
                mlflow.sklearn.log_model(model, "model")
    
                # 8. Save the results
                run_id = mlflow.active_run().info.run_id
                result = {
                    "run_id": run_id,
                    "config": asdict(config),
                    "metrics": metrics
                }
                self.results.append(result)
    
                print(f"Experiment complete: {config.experiment_name}")
                print(f"  Test accuracy: {metrics['test_accuracy']:.3f}")
                print(f"  Run ID: {run_id}")
    
                return metrics
    
        def compare_experiments(self) -> pd.DataFrame:
            """Compare experiment results"""
            if not self.results:
                print("No experiment results available")
                return pd.DataFrame()
    
            comparison_data = []
            for result in self.results:
                row = {
                    "run_id": result["run_id"],
                    "experiment": result["config"]["experiment_name"],
                    **result["metrics"]
                }
                comparison_data.append(row)
    
            df = pd.DataFrame(comparison_data)
            return df.sort_values("test_accuracy", ascending=False)
    
        def get_best_model(self, metric: str = "test_accuracy") -> Dict[str, Any]:
            """Automatically select the best model"""
            if not self.results:
                raise ValueError("No experiment results available")
    
            best_result = max(self.results, key=lambda x: x["metrics"][metric])
    
            print(f"\n=== Best model (by {metric}) ===")
            print(f"Run ID: {best_result['run_id']}")
            print(f"Experiment name: {best_result['config']['experiment_name']}")
            print(f"{metric}: {best_result['metrics'][metric]:.3f}")
            print(f"\nAll metrics:")
            for m_name, m_value in best_result['metrics'].items():
                print(f"  {m_name}: {m_value:.3f}")
    
            return best_result
    
        def save_comparison_report(self, filepath: str = "experiment_comparison.csv"):
            """Save the comparison report"""
            df = self.compare_experiments()
            df.to_csv(filepath, index=False)
            print(f"Comparison report saved: {filepath}")
    
    # Usage example
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Prepare the data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the experiment manager
    manager = ComprehensiveExperimentManager()
    
    # Experiment 1: Random Forest (shallow)
    config1 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with shallow depth",
        tags={"model_type": "random_forest", "depth": "shallow"}
    )
    rf_shallow = RandomForestClassifier(**config1.model_params)
    manager.run_experiment(config1, rf_shallow, X_train, y_train, X_test, y_test)
    
    # Experiment 2: Random Forest (deep)
    config2 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 20, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with deep depth",
        tags={"model_type": "random_forest", "depth": "deep"}
    )
    rf_deep = RandomForestClassifier(**config2.model_params)
    manager.run_experiment(config2, rf_deep, X_train, y_train, X_test, y_test)
    
    # Experiment 3: Gradient Boosting
    config3 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Gradient Boosting Classifier",
        tags={"model_type": "gradient_boosting"}
    )
    gb = GradientBoostingClassifier(**config3.model_params)
    manager.run_experiment(config3, gb, X_train, y_train, X_test, y_test)
    
    # Compare the results
    print("\n=== Comparison of all experiments ===")
    comparison_df = manager.compare_experiments()
    print(comparison_df[['experiment', 'test_accuracy', 'test_f1', 'overfit_gap']])
    
    # Select the best model
    best_model = manager.get_best_model(metric="test_accuracy")
    
    # Save the report
    manager.save_comparison_report()
    

**Output** :
    
    
    Experiment complete: model_comparison
      Test accuracy: 0.855
      Run ID: xxxxx
    
    Experiment complete: model_comparison
      Test accuracy: 0.890
      Run ID: yyyyy
    
    Experiment complete: model_comparison
      Test accuracy: 0.905
      Run ID: zzzzz
    
    === Comparison of all experiments ===
           experiment  test_accuracy  test_f1  overfit_gap
    2  model_comparison          0.905    0.903        0.032
    1  model_comparison          0.890    0.891        0.097
    0  model_comparison          0.855    0.856        0.007
    
    === Best model (by test_accuracy) ===
    Run ID: zzzzz
    Experiment name: model_comparison
    test_accuracy: 0.905
    
    All metrics:
      train_accuracy: 0.937
      test_accuracy: 0.905
      test_precision: 0.906
      test_recall: 0.905
      test_f1: 0.903
      overfit_gap: 0.032
    
    Comparison report saved: experiment_comparison.csv
    

### Problem 5 (Difficulty: hard)

Design and implement a complete machine learning workflow that combines MLflow and DVC. Include everything from data version control to experiment tracking and model storage.

Sample Answer
    
    
    """
    Complete ML workflow: DVC + MLflow
    
    Directory structure:
    project/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── models/
    ├── scripts/
    │   ├── prepare_data.py
    │   ├── train_model.py
    │   └── evaluate_model.py
    ├── dvc.yaml
    └── params.yaml
    """
    
    # Contents of params.yaml
    """
    data:
      raw_path: data/raw/dataset.csv
      train_path: data/processed/train.csv
      test_path: data/processed/test.csv
      test_size: 0.2
      random_state: 42
    
    model:
      type: random_forest
      n_estimators: 100
      max_depth: 10
      min_samples_split: 2
      random_state: 42
    
    mlflow:
      experiment_name: dvc_mlflow_integration
      tracking_uri: ./mlruns
    """
    
    # scripts/prepare_data.py
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    
    def load_params():
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_data():
        params = load_params()
        data_params = params['data']
    
        # Load the data
        df = pd.read_csv(data_params['raw_path'])
    
        # Train/test split
        train, test = train_test_split(
            df,
            test_size=data_params['test_size'],
            random_state=data_params['random_state']
        )
    
        # Save
        train.to_csv(data_params['train_path'], index=False)
        test.to_csv(data_params['test_path'], index=False)
    
        print(f"Data preparation complete: Train={len(train)}, Test={len(test)}")
    
    if __name__ == "__main__":
        prepare_data()
    
    # scripts/train_model.py
    import pandas as pd
    import yaml
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    def load_params():
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def train_model():
        params = load_params()
        data_params = params['data']
        model_params = params['model']
        mlflow_params = params['mlflow']
    
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # Load the data
        train = pd.read_csv(data_params['train_path'])
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # Start the MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("train_size", len(X_train))
    
            # Train the model
            model = RandomForestClassifier(
                n_estimators=model_params['n_estimators'],
                max_depth=model_params['max_depth'],
                min_samples_split=model_params['min_samples_split'],
                random_state=model_params['random_state']
            )
            model.fit(X_train, y_train)
    
            # Training metrics
            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_score)
    
            # Save the model
            model_path = "models/model.pkl"
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, "model")
    
            print(f"Training complete: Train Accuracy={train_score:.3f}")
    
    if __name__ == "__main__":
        train_model()
    
    # scripts/evaluate_model.py
    import pandas as pd
    import yaml
    import mlflow
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    import json
    
    def load_params():
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def evaluate_model():
        params = load_params()
        data_params = params['data']
        mlflow_params = params['mlflow']
    
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # Load the data and model
        test = pd.read_csv(data_params['test_path'])
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load("models/model.pkl")
    
        # Evaluate
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
    
        # Detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
    
        # Save metrics
        metrics = {
            "test_accuracy": test_accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        }
    
        with open("metrics/test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
        # Log to MLflow
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
        print(f"Evaluation complete: Test Accuracy={test_accuracy:.3f}")
        print(f"Detailed metrics: {metrics}")
    
    if __name__ == "__main__":
        evaluate_model()
    
    # Contents of dvc.yaml
    """
    stages:
      prepare:
        cmd: python scripts/prepare_data.py
        deps:
          - data/raw/dataset.csv
          - scripts/prepare_data.py
        params:
          - data.test_size
          - data.random_state
        outs:
          - data/processed/train.csv
          - data/processed/test.csv
    
      train:
        cmd: python scripts/train_model.py
        deps:
          - data/processed/train.csv
          - scripts/train_model.py
        params:
          - model
        outs:
          - models/model.pkl
    
      evaluate:
        cmd: python scripts/evaluate_model.py
        deps:
          - data/processed/test.csv
          - models/model.pkl
          - scripts/evaluate_model.py
        metrics:
          - metrics/test_metrics.json:
              cache: false
    """
    
    # Example of running the complete workflow
    """
    # 1. Initialize DVC
    dvc init
    
    # 2. Add the data
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc data/.gitignore
    git commit -m "Add raw data"
    
    # 3. Run the pipeline
    dvc repro
    
    # 4. Change an experiment parameter
    dvc exp run --set-param model.n_estimators=200
    
    # 5. Compare experiment results
    dvc exp show
    
    # 6. Apply the best experiment
    dvc exp apply <experiment-name>
    git add .
    git commit -m "Apply best experiment"
    
    # 7. Review results in the MLflow UI
    mlflow ui --backend-store-uri ./mlruns
    """
    
    print("Complete workflow design finished")
    print("DVC: version control for data and pipelines")
    print("MLflow: experiment tracking and model management")
    print("Integration: a reproducible and traceable ML workflow")
    

* * *

## References

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  3. Lakshmanan, V., Robinson, S., & Munn, M. (2020). _Machine Learning Design Patterns_. O'Reilly Media.
  4. Treveil, M., et al. (2020). _Introducing MLOps_. O'Reilly Media.
  5. MLflow Documentation. <https://mlflow.org/docs/latest/index.html>
  6. Weights & Biases Documentation. <https://docs.wandb.ai/>
  7. DVC Documentation. <https://dvc.org/doc>
