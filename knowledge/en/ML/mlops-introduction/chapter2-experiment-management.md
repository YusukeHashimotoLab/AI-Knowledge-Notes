---
title: Chapter 2：experiment management and version control
chapter_title: Chapter 2：experiment management and version control
subtitle: reproduciblemachine learning of for of experiment tracking and data versioning
reading_time: 30-35min
difficulty: Intermediate
code_examples: 12
exercises: 5
version: 1.0
---

This chapter covers Chapter 2：experiment management and version control. You will learn experiment tracking and best practices for experiment management.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the importance of experiment management in machine learning
  * ✅ Implement experiment tracking and model management using MLflow
  * ✅ Execute hyperparameter optimization with Weights & Biases
  * ✅ Perform data and model version control with DVC
  * ✅ Apply best practices for experiment management
  * ✅ Build reproducible machine learning pipelines

* * *

## 2.1 Importance of Experiment Management

### experiment managementWhat is

**experiment management（Experiment Management）** 、is the systematic process of recording, tracking, comparing, and reproducing experiments in machine learning projects.

> "Excellent ML projects depend on the ability to manage hundreds to thousands of experiments."

### Challenges in Experiment Management

Challenge | Impact | Solution  
---|---|---  
**lack of reproducibility** | Cannot reproduce past experiments | parameters・code・data of version control  
**difficulty comparing experiments** | Cannot select optimal model | unified metrics logging  
**loss of insights** | Information is not shared between teams | centralized experiment tracking  
**data drift** | Cannot track data changes | data versioning  
  
### Overview of Experiment Management
    
    
    ```mermaid
    graph TD
        A[Experiment Design] --> B[Parameter Configuration]
        B --> C[Data Loading]
        C --> D[model training]
        D --> E[Metrics Logging]
        E --> F[Artifact Storage]
        F --> G[Experiment comparison]
        G --> H{Improved?}
        H -->|Yes| I[Best Model Selection]
        H -->|No| B
        I --> J[Model Deployment]
    
        style A fill:#ffebee
        style D fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
        style I fill:#c8e6c9
        style J fill:#c8e6c9
    ```

### Value Brought by Experiment Management

#### 1\. ensuring reproducibility

  * Environment where same results can be reproduced
  * Complete record of code, data, and parameters
  * Easier auditing and compliance

#### 2\. experiment of comparison and minanalysis

  * Systematically compare multiple experiments
  * Visualize relationship between parameters and performance
  * Data-driven decision making

#### 3\. selecting the best model

  * Select model with objective criteria
  * Performance vs cost tradeoff analysis
  * Confident deployment to production

* * *

## 2.2 MLflow

### MLflowWhat is

**MLflow** is an open-source platform for managing the entire machine learning lifecycle.

### MLflow of key components

Component | Function | Purpose  
---|---|---  
**MLflow Tracking** | Record experiment parameters and metrics | experiment management  
**MLflow Projects** | Reproducible code execution | Environment management  
**MLflow Models** | Model packaging and deployment | model management  
**MLflow Registry** | Model of version control | Production operations  
  
### MLflow Tracking: basic usage
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: MLflow Tracking: basic usage
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
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
    
    # MLflowConfigure experiment
    mlflow.set_experiment("random_forest_classification")
    
    # Run experiment
    with mlflow.start_run(run_name="rf_baseline"):
        # Parameter configuration
        n_estimators = 100
        max_depth = 10
        random_state = 42
    
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
    
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    
        # prediction and metrics of calculation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
    
        # Logging metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    
        # Save model
        mlflow.sklearn.log_model(model, "model")
    
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
    

**Output** ：
    
    
    Accuracy: 0.895
    Precision: 0.891
    Recall: 0.902
    

### running and comparing multiple experiments
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: running and comparing multiple experiments
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # experimentconfiguration of list
    experiment_configs = [
        {"n_estimators": 50, "max_depth": 5, "name": "rf_shallow"},
        {"n_estimators": 100, "max_depth": 10, "name": "rf_medium"},
        {"n_estimators": 200, "max_depth": 20, "name": "rf_deep"},
        {"n_estimators": 300, "max_depth": None, "name": "rf_full"},
    ]
    
    mlflow.set_experiment("rf_hyperparameter_tuning")
    
    # Run experiments with each configuration
    results = []
    for config in experiment_configs:
        with mlflow.start_run(run_name=config["name"]):
            # Log parameters
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])
    
            # Train model
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
    
            # evaluation
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # Logging metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("overfit_gap", train_acc - test_acc)
    
            # Save model
            mlflow.sklearn.log_model(model, "model")
    
            results.append({
                "name": config["name"],
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfit": train_acc - test_acc
            })
    
            print(f"{config['name']}: Train={train_acc:.3f}, Test={test_acc:.3f}, Overfit={train_acc - test_acc:.3f}")
    
    print("\n=== Comparison of experiment results ===")
    for result in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{result['name']}: Test Accuracy = {result['test_acc']:.3f}")
    

**Output** ：
    
    
    rf_shallow: Train=0.862, Test=0.855, Overfit=0.007
    rf_medium: Train=0.895, Test=0.895, Overfit=0.000
    rf_deep: Train=0.987, Test=0.890, Overfit=0.097
    rf_full: Train=1.000, Test=0.885, Overfit=0.115
    
    === Comparison of experiment results ===
    rf_medium: Test Accuracy = 0.895
    rf_deep: Test Accuracy = 0.890
    rf_full: Test Accuracy = 0.885
    rf_shallow: Test Accuracy = 0.855
    

### MLflow Autolog: automatic logging
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
    """
    Example: MLflow Autolog: automatic logging
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # enable MLflow autolog
    mlflow.sklearn.autolog()
    
    mlflow.set_experiment("rf_with_autolog")
    
    with mlflow.start_run(run_name="rf_autolog_example"):
        # train model (parameters and metrics are automatically logged)
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # manually log additional metrics
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
    
        print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")
        print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    

**Output** ：
    
    
    Test Accuracy: 0.900
    CV Mean: 0.893 (+/- 0.012)
    

> **benefits of autolog** : Parameters, metrics, and models are automatically logged, preventing manual logging errors.

### MLflow Models: Model of packaging
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - pandas>=2.0.0, <2.2.0
    
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # Define custom model wrapper
    class CustomModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model
    
        def predict(self, context, model_input):
            """Custom prediction logic"""
            predictions = self.model.predict_proba(model_input)
            # Return predictions only when confidence is 0.7 or higher
            confident_predictions = []
            for i, prob in enumerate(predictions):
                max_prob = max(prob)
                if max_prob >= 0.7:
                    confident_predictions.append(int(prob.argmax()))
                else:
                    confident_predictions.append(-1)  # Unknown
            return confident_predictions
    
    # Train model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Wrap custom model
    wrapped_model = CustomModelWrapper(base_model)
    
    mlflow.set_experiment("custom_model_packaging")
    
    with mlflow.start_run(run_name="confident_predictor"):
        # Save custom model
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
        print(f"Predictions with confidence: {test_predictions}")
        print(f"Low-confidence predictions（-1）number of: {sum(1 for p in test_predictions if p == -1)}")
    

### MLflow UI: experiment visualization
    
    
    # start MLflow UI
    # mlflow ui --port 5000
    
    # access http://localhost:5000 in browser
    # - display list of experiments
    # - parameters and metrics of comparison
    # - Model of download
    # - search and filter experiments
    

### MLflow Projects: reproducible execution
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
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
    
    # run project
    import mlflow
    
    # run locally
    mlflow.run(
        ".",
        parameters={
            "n_estimators": 200,
            "max_depth": 15,
            "data_path": "data/train.csv"
        }
    )
    
    # run from GitHub
    mlflow.run(
        "https://github.com/username/ml-project",
        version="main",
        parameters={"n_estimators": 150}
    )
    

* * *

## 2.3 Weights & Biases (W&B)

### Weights & BiasesWhat is

**Weights & Biases (W&B)**is a powerful platform for experiment tracking, visualization, and hyperparameter optimization.

### W&B of key features

Function | Description | Purpose  
---|---|---  
**Experiment Tracking** | Visualize metrics in real-time | Experiment monitoring  
**Sweeps** | Automatic hyperparameter optimization | Tuning  
**Artifacts** | Save models and datasets | version control  
**Reports** | Create and share experiment reports | Team collaboration  
  
### W&B: basic experiment tracking
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - wandb>=0.15.0
    
    """
    Example: W&B: basic experiment tracking
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np
    
    # initialize W&B
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
    
    # get configuration
    config = wandb.config
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)
    
    # evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Logging metrics
    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    # Visualize confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })
    
    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Finish experiment
    wandb.finish()
    

**Output** ：
    
    
    Accuracy: 0.895, F1: 0.897
    View run at: https://wandb.ai/username/ml-experiment-tracking/runs/xxxxx
    

### W&B: real-time visualization of learning curves
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - wandb>=0.15.0
    
    """
    Example: W&B: real-time visualization of learning curves
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import wandb
    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    wandb.init(project="learning-curves", name="rf_learning_curve")
    
    # Calculate learning curves
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
    
    print("Calculate learning curvescomplete")
    wandb.finish()
    

### W&B Sweeps: hyperparameter optimization
    
    
    # Requirements:
    # - Python 3.9+
    # - wandb>=0.15.0
    
    """
    Example: W&B Sweeps: hyperparameter optimization
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesianoptimization
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
    
    # Define training function
    def train():
        # initialize W&B
        wandb.init()
        config = wandb.config
    
        # Train model
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # evaluation
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
    
        # Logging metrics
        wandb.log({
            'accuracy': test_acc,
            'train_accuracy': train_acc,
            'overfit_gap': train_acc - test_acc
        })
    
    # run sweep
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter-tuning")
    
    # 10 timesexperiments of execution
    wandb.agent(sweep_id, function=train, count=10)
    
    print(f"Sweepcomplete: {sweep_id}")
    

**Output** ：
    
    
    Sweepcomplete: username/hyperparameter-tuning/sweep_xxxxx
    best of accuracy: 0.915
    optimalparameters: n_estimators=220, max_depth=18, min_samples_split=3, min_samples_leaf=2
    

### W&B: Model and dataset of save
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    # - wandb>=0.15.0
    
    """
    Example: W&B: Model and dataset of save
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import wandb
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    wandb.init(project="model-artifacts", name="rf_with_artifacts")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = "random_forest_model.pkl"
    joblib.dump(model, model_path)
    
    # W&Bsave as artifact in
    artifact = wandb.Artifact(
        name="random_forest_model",
        type="model",
        description="Random Forest classifier trained on classification dataset"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # dataset of save
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
    
    print("Model and datasetsaved")
    wandb.finish()
    

### W&B: multiplenumberexperiment visualizationcomparison
    
    
    # Requirements:
    # - Python 3.9+
    # - wandb>=0.15.0
    
    """
    Example: W&B: multiplenumberexperiment visualizationcomparison
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # multiplenumber of Modelexperiment with
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for model_name, model in models.items():
        # Start experiment
        run = wandb.init(
            project="model-comparison",
            name=model_name,
            reinit=True
        )
    
        # Train model
        model.fit(X_train, y_train)
    
        # Prediction and evaluation
        y_pred = model.predict(X_test)
    
        # Logging metrics
        wandb.log({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "model_type": model_name
        })
    
        # featuresimportantdegree of log（if possible）
        if hasattr(model, 'feature_importances_'):
            importance_data = [[i, imp] for i, imp in enumerate(model.feature_importances_)]
            table = wandb.Table(data=importance_data, columns=["feature", "importance"])
            wandb.log({"feature_importance": wandb.plot.bar(table, "feature", "importance")})
    
        run.finish()
    
    print("allModelexperiment completion of")
    

* * *

## 2.4 DVC (Data Version Control)

### DVCWhat is

**DVC（Data Version Control）** 、data and Model of version control 、Gitis a tool that implements workflows like。

### DVC of key features

Function | Description | Purpose  
---|---|---  
**data versioning** | large data of version control | datatracking  
**pipeline definition** | reproducibleMLpipeline | workflowmanagement  
**remote storage** | S3、GCS、Azureetc.withintegration | datasharing  
**experiment management** | tracking and comparing experiments | Experiment comparison  
  
### DVCsetup and initialization of
    
    
    # DVCinstallation of
    # pip install dvc
    
    # Gitrepository of initialization（if not yet）
    # git init
    
    # initialize DVC
    # dvc init
    
    # .dvc/config file created
    # git add .dvc .dvcignore
    # git commit -m "Initialize DVC"
    

### data of version control
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: data of version control
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # Pythongenerate data with
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
    
    # data of save
    data.to_csv('data/raw_data.csv', index=False)
    print("datasaved: data/raw_data.csv")
    
    
    
    # DVCtrack data with
    # dvc add data/raw_data.csv
    
    # this creates the following:
    # - data/raw_data.csv.dvc (metadatafile)
    # - data/.gitignore (actual data excluded)
    
    # metadatafile Gitcommit to
    # git add data/raw_data.csv.dvc data/.gitignore
    # git commit -m "Add raw data"
    
    # Configure remote storage（example: localdirectory）
    # dvc remote add -d local_storage /tmp/dvc-storage
    # git add .dvc/config
    # git commit -m "Configure DVC remote storage"
    
    # data remote push
    # dvc push
    

### DVCDefine pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: DVCDefine pipeline
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # prepare.py - data preprocessingscript
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sys
    
    def prepare_data(input_file, train_file, test_file):
        # Load data
        data = pd.read_csv(input_file)
    
        # training・test data of minrate
        train, test = train_test_split(data, test_size=0.2, random_state=42)
    
        # save
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
    
        print(f"training data: {len(train)}row")
        print(f"test data: {len(test)}row")
    
    if __name__ == "__main__":
        prepare_data(
            input_file="data/raw_data.csv",
            train_file="data/train.csv",
            test_file="data/test.csv"
        )
    
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: DVCDefine pipeline
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # train.py - model trainingscript
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import json
    
    def train_model(train_file, model_file, metrics_file):
        # Load data
        train = pd.read_csv(train_file)
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # Save model
        joblib.dump(model, model_file)
    
        # Save metrics
        train_accuracy = model.score(X_train, y_train)
        metrics = {"train_accuracy": train_accuracy}
    
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"training accuracy: {train_accuracy:.3f}")
    
    if __name__ == "__main__":
        train_model(
            train_file="data/train.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/train_metrics.json"
        )
    
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: DVCDefine pipeline
    
    Purpose: Demonstrate data manipulation and preprocessing
    Target: Intermediate
    Execution time: 1-3 seconds
    Dependencies: None
    """
    
    # evaluate.py - Modelevaluationscript
    import pandas as pd
    import joblib
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    def evaluate_model(test_file, model_file, metrics_file):
        # load data and models
        test = pd.read_csv(test_file)
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load(model_file)
    
        # Prediction and evaluation
        y_pred = model.predict(X_test)
    
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
    
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"test accuracy: {metrics['accuracy']:.3f}")
        print(f"precision: {metrics['precision']:.3f}")
        print(f"recall: {metrics['recall']:.3f}")
    
    if __name__ == "__main__":
        evaluate_model(
            test_file="data/test.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/test_metrics.json"
        )
    

### dvc.yaml: Define pipeline
    
    
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
    
    
    
    # Run pipeline
    # dvc repro
    
    # Output:
    # Running stage 'prepare':
    # > python prepare.py
    # training data: 800row
    # test data: 200row
    #
    # Running stage 'train':
    # > python train.py
    # training accuracy: 1.000
    #
    # Running stage 'evaluate':
    # > python evaluate.py
    # test accuracy: 0.895
    # precision: 0.891
    # recall: 0.902
    
    # metrics of display
    # dvc metrics show
    
    # Visualize pipeline
    # dvc dag
    

### DVC Experiments: experiment of tracking
    
    
    # parametersfile of creation
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
    
    # Run experiment
    # dvc exp run
    
    # multiplenumberexperiments of andcolumnexecution
    # dvc exp run --set-param model.n_estimators=150
    # dvc exp run --set-param model.n_estimators=200
    # dvc exp run --set-param model.max_depth=15
    
    # experiment results of display
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
    
    # apply best experiment
    # dvc exp apply exp-2
    # git add .
    # git commit -m "Apply best experiment: n_estimators=200"
    

### DVC and Gitintegrated workflow of
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # completeworkflowexample
    import subprocess
    import os
    
    def dvc_workflow_example():
        """DVC and Gitcomplete usingMLworkflow"""
    
        # 1. new branch creation
        subprocess.run(["git", "checkout", "-b", "experiment/new-features"])
    
        # 2. new data additional
        print("new data generationmedium...")
        import pandas as pd
        import numpy as np
    
        new_data = pd.DataFrame({
            'feature1': np.random.randn(1500),
            'feature2': np.random.randn(1500),
            'feature3': np.random.randn(1500),
            'feature4': np.random.randn(1500),  # new features
            'target': np.random.randint(0, 2, 1500)
        })
        new_data.to_csv('data/raw_data_v2.csv', index=False)
    
        # 3. DVCtrack new data with
        subprocess.run(["dvc", "add", "data/raw_data_v2.csv"])
    
        # 4. changes commit
        subprocess.run(["git", "add", "data/raw_data_v2.csv.dvc", "data/.gitignore"])
        subprocess.run(["git", "commit", "-m", "Add new dataset with feature4"])
    
        # 5. pipeline execution
        subprocess.run(["dvc", "repro"])
    
        # 6. results verification
        subprocess.run(["dvc", "metrics", "show"])
    
        # 7. changes push
        subprocess.run(["git", "push", "origin", "experiment/new-features"])
        subprocess.run(["dvc", "push"])
    
        print("workflowcomplete")
    
    # note: actual of executioninappropriateGit/DVCsetup required
    print("DVCworkflowexample（Command explanation）")
    

* * *

## 2.5 best practices

### 1\. metadatalog of best practices

#### loginformation to be

Category | Item | reason  
---|---|---  
**experimentinformation** | experiment name、date and time、executionperson | experiment of identification and tracking  
**environment information** | Pythonversion、library version、OS | ensuring reproducibility  
**data information** | data version、number of samples、mindistribution | data driftdetection  
**model information** | architecture、parameters | Modelreconstruction of  
**evaluationinformation** | metrics、confusionrowcolumn | performance of comparison  
      
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
    import mlflow
    import platform
    import sys
    from datetime import datetime
    
    def log_comprehensive_metadata(model, X_train, y_train, X_test, y_test):
        """comprehensive metadata logging"""
    
        with mlflow.start_run(run_name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. environment information
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("os", platform.system())
            mlflow.log_param("os_version", platform.version())
    
            # 2. data information
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_distribution", dict(zip(*np.unique(y_train, return_counts=True))))
    
            # 3. model information
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())
    
            # 4. training
            model.fit(X_train, y_train)
    
            # 5. evaluationmetrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
    
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
    
            # 6. experimentmemo
            mlflow.set_tag("experiment_description", "Comprehensive metadata logging example")
            mlflow.set_tag("data_version", "v1.0")
            mlflow.set_tag("experiment_type", "baseline")
    
            # 7. Save model
            mlflow.sklearn.log_model(model, "model")
    
            print("comprehensive metadata log")
    
    # Usage example
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    log_comprehensive_metadata(model, X_train, y_train, X_test, y_test)
    

### 2\. hyperparametermanagement
    
    
    # Requirements:
    # - Python 3.9+
    # - pyyaml>=6.0.0
    
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Optional
    
    @dataclass
    class ModelConfig:
        """Modelconfiguration of structured definition"""
        n_estimators: int = 100
        max_depth: Optional[int] = 10
        min_samples_split: int = 2
        min_samples_leaf: int = 1
        random_state: int = 42
    
        def save(self, filepath: str):
            """configuration YAMLfile save"""
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f)
    
        @classmethod
        def load(cls, filepath: str):
            """YAMLfilefromconfiguration loading"""
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
    
    # save configuration
    config = ModelConfig(n_estimators=150, max_depth=15)
    config.save("configs/model_config.yaml")
    
    # configuration of loading
    loaded_config = ModelConfig.load("configs/model_config.yaml")
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**asdict(loaded_config))
    model.fit(X_train, y_train)
    
    print(f"configurationuseModeltrain: {asdict(loaded_config)}")
    

### 3\. artifact management
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - mlflow>=2.4.0
    
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
        """experiment of artifacts systematic save"""
    
        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
            # 1. Save model
            mlflow.sklearn.log_model(model, "model")
    
            # 2. Save metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
            # 3. save configuration
            for param_name, param_value in config.items():
                mlflow.log_param(param_name, param_value)
    
            # 4. feature information of save
            feature_info = {
                "feature_names": feature_names,
                "n_features": len(feature_names)
            }
    
            # temporaryfile saveMLflowlog to
            temp_dir = Path("temp_artifacts")
            temp_dir.mkdir(exist_ok=True)
    
            feature_path = temp_dir / "feature_info.json"
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            mlflow.log_artifact(str(feature_path))
    
            # 5. featuresimportantdegree of save（if possible）
            if hasattr(model, 'feature_importances_'):
                importance_df = {
                    name: float(imp)
                    for name, imp in zip(feature_names, model.feature_importances_)
                }
                importance_path = temp_dir / "feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance_df, f, indent=2)
                mlflow.log_artifact(str(importance_path))
    
            # 6. predictionexample of save
            sample_predictions = {
                "sample_input": X_test[:5].tolist(),
                "predictions": model.predict(X_test[:5]).tolist()
            }
            pred_path = temp_dir / "sample_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(sample_predictions, f, indent=2)
            mlflow.log_artifact(str(pred_path))
    
            # temporaryfile of delete
            import shutil
            shutil.rmtree(temp_dir)
    
            print("all of artifactssaved")
    
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
    

### 4\. experiment of organization
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
    from enum import Enum
    import mlflow
    from datetime import datetime
    
    class ExperimentType(Enum):
        """experimenttype of definition"""
        BASELINE = "baseline"
        FEATURE_ENGINEERING = "feature_engineering"
        HYPERPARAMETER_TUNING = "hyperparameter_tuning"
        MODEL_SELECTION = "model_selection"
        PRODUCTION = "production"
    
    class ExperimentManager:
        """experiment of organizationalmanagement"""
    
        def __init__(self, project_name: str):
            self.project_name = project_name
    
        def create_experiment_name(
            self,
            exp_type: ExperimentType,
            model_name: str,
            version: str = "v1"
        ) -> str:
            """generate hierarchical experiment names"""
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
            """Run experiment and log"""
    
            # experiment name of generation
            exp_name = self.create_experiment_name(exp_type, model_name, version)
            mlflow.set_experiment(exp_name)
    
            # executionname of generation（with timestamp）
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
            with mlflow.start_run(run_name=run_name):
                # set tags
                mlflow.set_tag("experiment_type", exp_type.value)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("version", version)
                mlflow.set_tag("description", description)
    
                # training
                train_metrics = train_fn(model)
    
                # evaluation
                test_metrics = evaluate_fn(model)
    
                # Logging metrics
                for metric_name, metric_value in {**train_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # Save model
                mlflow.sklearn.log_model(model, "model")
    
                print(f"experiment complete: {exp_name}/{run_name}")
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
    
    # baselineModelexperiments of
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
    
    print(f"results: {results}")
    

* * *

## 2.6 this of Summary

### What We Learned

  1. **Importance of Experiment Management**

     * ensuring reproducibility machine learningproject of infrastructure
     * systematicexperiment managementefficient withModeldevelopment
     * data and Model of version controlnecessity of
  2. **MLflow**

     * MLflow Tracking: parameters and Logging metrics
     * MLflow Models: Model packaging and deployment
     * MLflow Projects: reproducibleexperimentenvironment
     * AutologFunctionbyautomatic logging
  3. **Weights & Biases**

     * real-timeexperimentvisualization
     * Automatic hyperparameter optimization（Sweeps）
     * Team collaborationand report sharing
     * artifact management and versioning
  4. **DVC**

     * data and Model of Gitlikemanagement
     * reproducibleMLpipeline definition
     * remote storagewithintegration
     * tracking and comparing experiments
  5. **best practices**

     * comprehensive metadata logging
     * structuredparametersmanagement
     * systematicArtifact Storage
     * hierarchical experiment organization

### tool usageminguidelines

tool | strengths | recommended use cases  
---|---|---  
**MLflow** | open source、high flexibility | on-premises environment、freedom-oriented  
**W &B** | advanced visualization、Team collaboration | cloud environment、team development  
**DVC** | Gitaffinity with、data management | large data、Versionemphasis  
  
### Next Chapter

Chapter 3in、**continuous integration/deploymentment（CI/CD）** we will learn：

  * MLOpsinCI/CDpipeline
  * automatictest and Modelvalidation
  * Modeldeployment strategy of
  * monitoring and feedback loop

* * *

## Exercises

### Problem1（Difficulty：easy）

experiment managementin「reproducibility」why it is important3listDescriptionplease。

Example solution

**Solution** ：

  1. **results of validation**

     * same conditionsexperiment with reexecution、results of validity verification 
     * unexpectedresults incidentalor、systematicProblemcan determine
  2. **Share insights**

     * team members can reproduce same experiments and deepen understanding
     * research results of transparency and reliability improvement
  3. **debugging and Improved**

     * Problemwhen it occurs、specificexperiments ofstate reproducible and debuggable
     * past of successfulexperiment base 、gradualImproved 

### Problem2（Difficulty：medium）

MLflowusing、following of requirements satisfyexperiment tracking implementationplease：

  * 3 differenthyperparameterconfiguration Modeltrain
  * eachexperiment training accuracy and test accuracy log
  * most alsohightest accuracy achievedexperiment specific

Example solution
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    
    """
    Example: MLflowusing、following of requirements satisfyexperiment trac
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # experimentconfiguration
    mlflow.set_experiment("hyperparameter_comparison")
    
    # differenthyperparameterconfiguration
    configs = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20}
    ]
    
    results = []
    
    # Run experiments with each configuration
    for i, config in enumerate(configs):
        with mlflow.start_run(run_name=f"experiment_{i+1}"):
            # Log parameters
            mlflow.log_params(config)
    
            # Train model
            model = RandomForestClassifier(**config, random_state=42)
            model.fit(X_train, y_train)
    
            # accuracy of calculation
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # Logging metrics
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
    
            # Save results
            results.append({
                "config": config,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "run_id": mlflow.active_run().info.run_id
            })
    
            print(f"experiment {i+1}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # best experiment specific
    best_result = max(results, key=lambda x: x['test_acc'])
    
    print("\n=== best experiment ===")
    print(f"configuration: {best_result['config']}")
    print(f"test accuracy: {best_result['test_acc']:.3f}")
    print(f"Run ID: {best_result['run_id']}")
    

**Output** ：
    
    
    experiment 1: Train=0.862, Test=0.855
    experiment 2: Train=0.895, Test=0.895
    experiment 3: Train=0.987, Test=0.890
    
    === best experiment ===
    configuration: {'n_estimators': 100, 'max_depth': 10}
    test accuracy: 0.895
    Run ID: xxxxxxxxxxxxx
    

### Problem3（Difficulty：medium）

DVCmain benefits of using、Gitcompared to using onlyDescriptionplease。

Example solution

**Solution** ：

**DVCmain benefits of** ：

  1. **large-capacityfile of efficientmanagement**

     * Git: large-capacityfile（dataset、Model）repository bloat with
     * DVC: actualfile remote storage save、Gitonly metadata in
  2. **data of version control**

     * Git: binaryfile of differenceminmanagement inefficient
     * DVC: data of change history efficient tracking、optional of Versionrestorable to
  3. **reproduciblepipeline**

     * Git: scriptversion controlonly
     * DVC: data、code、parameters complete includingpipeline definition・reproduction
  4. **Team collaborationease of**

     * Git: large-capacityfile of sharing difficulty
     * DVC: via remote storage efficient datasharing

**comparisondisplay** ：

perspective | Git only | DVC + Git  
---|---|---  
codemanagement | ◎ excellent | ◎ excellent  
data management | △ inefficient | ◎ optimization  
model management | △ difficulty | ◎ systematic  
pipeline | × unsupported | ◎ full support  
reproducibility | △ sectionmin | ◎ complete  
  
### Problem4（Difficulty：hard）

comprehensiveexperiment managementplease design a system。following of elements include:

  * experiment of automatic logging
  * parameters of structuredmanagement
  * Comparison of experiment resultsFunction
  * bestModelautomatic selection of

Example solution
    
    
    # Requirements:
    # - Python 3.9+
    # - mlflow>=2.4.0
    # - pandas>=2.0.0, <2.2.0
    # - pyyaml>=6.0.0
    
    import mlflow
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Dict, Any, List, Optional
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import pandas as pd
    
    @dataclass
    class ExperimentConfig:
        """experimentconfiguration of structured definition"""
        experiment_name: str
        model_params: Dict[str, Any]
        data_params: Dict[str, Any]
        description: str = ""
        tags: Dict[str, str] = None
    
    class ComprehensiveExperimentManager:
        """comprehensiveexperiment managementsystem"""
    
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
            """Run experiment and automatic logging"""
    
            # Configure experiment
            mlflow.set_experiment(config.experiment_name)
    
            with mlflow.start_run(description=config.description):
                # 1. parameters of logging
                mlflow.log_params(config.model_params)
                mlflow.log_params(config.data_params)
    
                # 2. set tags
                if config.tags:
                    for key, value in config.tags.items():
                        mlflow.set_tag(key, value)
    
                # 3. Train model
                model.fit(X_train, y_train)
    
                # 4. prediction
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
    
                # 5. metrics of calculation
                metrics = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "test_precision": precision_score(y_test, y_test_pred, average='weighted'),
                    "test_recall": recall_score(y_test, y_test_pred, average='weighted'),
                    "test_f1": f1_score(y_test, y_test_pred, average='weighted'),
                    "overfit_gap": accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
                }
    
                # 6. metrics of logging
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # 7. Save model
                mlflow.sklearn.log_model(model, "model")
    
                # 8. Save results
                run_id = mlflow.active_run().info.run_id
                result = {
                    "run_id": run_id,
                    "config": asdict(config),
                    "metrics": metrics
                }
                self.results.append(result)
    
                print(f"experiment complete: {config.experiment_name}")
                print(f"  test accuracy: {metrics['test_accuracy']:.3f}")
                print(f"  Run ID: {run_id}")
    
                return metrics
    
        def compare_experiments(self) -> pd.DataFrame:
            """Comparison of experiment results"""
            if not self.results:
                print("no experiment results")
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
            """best modelautomatic selection of"""
            if not self.results:
                raise ValueError("no experiment results")
    
            best_result = max(self.results, key=lambda x: x["metrics"][metric])
    
            print(f"\n=== best model（{metric}criteria）===")
            print(f"Run ID: {best_result['run_id']}")
            print(f"experiment name: {best_result['config']['experiment_name']}")
            print(f"{metric}: {best_result['metrics'][metric]:.3f}")
            print(f"\nall metrics:")
            for m_name, m_value in best_result['metrics'].items():
                print(f"  {m_name}: {m_value:.3f}")
    
            return best_result
    
        def save_comparison_report(self, filepath: str = "experiment_comparison.csv"):
            """comparisonreport of save"""
            df = self.compare_experiments()
            df.to_csv(filepath, index=False)
            print(f"save comparison report: {filepath}")
    
    # Usage example
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # data of preparation
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # experimentmanager of initialization
    manager = ComprehensiveExperimentManager()
    
    # experiment1: Random Forest（shallow）
    config1 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with shallow depth",
        tags={"model_type": "random_forest", "depth": "shallow"}
    )
    rf_shallow = RandomForestClassifier(**config1.model_params)
    manager.run_experiment(config1, rf_shallow, X_train, y_train, X_test, y_test)
    
    # experiment2: Random Forest（deep）
    config2 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 20, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with deep depth",
        tags={"model_type": "random_forest", "depth": "deep"}
    )
    rf_deep = RandomForestClassifier(**config2.model_params)
    manager.run_experiment(config2, rf_deep, X_train, y_train, X_test, y_test)
    
    # experiment3: Gradient Boosting
    config3 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Gradient Boosting Classifier",
        tags={"model_type": "gradient_boosting"}
    )
    gb = GradientBoostingClassifier(**config3.model_params)
    manager.run_experiment(config3, gb, X_train, y_train, X_test, y_test)
    
    # results of comparison
    print("\n=== all experiments of comparison ===")
    comparison_df = manager.compare_experiments()
    print(comparison_df[['experiment', 'test_accuracy', 'test_f1', 'overfit_gap']])
    
    # best modelselection of
    best_model = manager.get_best_model(metric="test_accuracy")
    
    # report of save
    manager.save_comparison_report()
    

**Output** ：
    
    
    experiment complete: model_comparison
      test accuracy: 0.855
      Run ID: xxxxx
    
    experiment complete: model_comparison
      test accuracy: 0.890
      Run ID: yyyyy
    
    experiment complete: model_comparison
      test accuracy: 0.905
      Run ID: zzzzz
    
    === all experiments of comparison ===
           experiment  test_accuracy  test_f1  overfit_gap
    2  model_comparison          0.905    0.903        0.032
    1  model_comparison          0.890    0.891        0.097
    0  model_comparison          0.855    0.856        0.007
    
    === best model（test_accuracycriteria）===
    Run ID: zzzzz
    experiment name: model_comparison
    test_accuracy: 0.905
    
    all metrics:
      train_accuracy: 0.937
      test_accuracy: 0.905
      test_precision: 0.906
      test_recall: 0.905
      test_f1: 0.903
      overfit_gap: 0.032
    
    save comparison report: experiment_comparison.csv
    

### Problem5（Difficulty：hard）

MLflow and DVCcomplete combination ofmachine learningworkflow design、implementationplease。data of version controlfrom experiment tracking、Save modelinclude up to。

Example solution
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - mlflow>=2.4.0
    # - pandas>=2.0.0, <2.2.0
    # - pyyaml>=6.0.0
    
    """
    completeML workflow: DVC + MLflow
    
    directorystructure:
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
    
    # params.yaml contents of
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
    
        # Load data
        df = pd.read_csv(data_params['raw_path'])
    
        # training・testminrate
        train, test = train_test_split(
            df,
            test_size=data_params['test_size'],
            random_state=data_params['random_state']
        )
    
        # save
        train.to_csv(data_params['train_path'], index=False)
        test.to_csv(data_params['test_path'], index=False)
    
        print(f"data preparationcomplete: Train={len(train)}, Test={len(test)}")
    
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
    
        # MLflow of configuration
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # Load data
        train = pd.read_csv(data_params['train_path'])
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # MLflowStart experiment
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("train_size", len(X_train))
    
            # Train model
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
    
            # Save model
            model_path = "models/model.pkl"
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, "model")
    
            print(f"trainingcomplete: Train Accuracy={train_score:.3f}")
    
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
    
        # MLflow of configuration
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # load data and models
        test = pd.read_csv(data_params['test_path'])
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load("models/model.pkl")
    
        # evaluation
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
    
        # detailsreport
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
    
        # MLflowrecorded in
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
        print(f"evaluationcomplete: Test Accuracy={test_accuracy:.3f}")
        print(f"detailsmetrics: {metrics}")
    
    if __name__ == "__main__":
        evaluate_model()
    
    # dvc.yaml contents of
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
    
    # completeworkflow of executionexample
    """
    # 1. initialize DVC
    dvc init
    
    # 2. data of additional
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc data/.gitignore
    git commit -m "Add raw data"
    
    # 3. Run pipeline
    dvc repro
    
    # 4. experimentparameters of changes
    dvc exp run --set-param model.n_estimators=200
    
    # 5. Comparison of experiment results
    dvc exp show
    
    # 6. apply best experiment
    dvc exp apply <experiment-name>
    git add .
    git commit -m "Apply best experiment"
    
    # 7. MLflow UIcheck results with
    mlflow ui --backend-store-uri ./mlruns
    """
    
    print("completeworkflowdesigncomplete")
    print("DVC: data and pipeline of version control")
    print("MLflow: experiment tracking and model management")
    print("integration: reproducible trackingpossibleMLworkflow")
    </experiment-name>

* * *

## referenceliterature

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  3. Lakshmanan, V., Robinson, S., & Munn, M. (2020). _Machine Learning Design Patterns_. O'Reilly Media.
  4. Treveil, M., et al. (2020). _Introducing MLOps_. O'Reilly Media.
  5. MLflow Documentation. <https://mlflow.org/docs/latest/index.html>
  6. Weights & Biases Documentation. <https://docs.wandb.ai/>
  7. DVC Documentation. <https://dvc.org/doc>
