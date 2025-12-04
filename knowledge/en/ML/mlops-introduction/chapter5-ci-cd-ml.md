---
title: "Chapter 5: Building CI/CD Pipelines"
chapter_title: "Chapter 5: Building CI/CD Pipelines"
subtitle: Automated Testing and Deployment of Machine Learning Models
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 12
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Building CI/CD Pipelines. You will learn characteristics, Design automated testing strategies for ML models, and Build model training.

## Learning Objectives

By reading this chapter, you will learn to:

  * ✅ Understand the characteristics and importance of CI/CD for machine learning
  * ✅ Design automated testing strategies for ML models
  * ✅ Build model training and validation pipelines with GitHub Actions
  * ✅ Implement various deployment strategies (Blue-Green, Canary, etc.)
  * ✅ Apply end-to-end CI/CD pipelines to production environments

* * *

## 5.1 Characteristics of CI/CD for ML

### Differences from Traditional CI/CD

**CI/CD (Continuous Integration/Continuous Delivery)** is a methodology for automating the software development process. In machine learning, data and model management are required in addition to traditional software development.
    
    
    ```mermaid
    graph TD
        A[Code Change] --> B[CI: Automated Testing]
        B --> C[Model Training]
        C --> D[Model Validation]
        D --> E{Performance OK?}
        E -->|Yes| F[CD: Deploy]
        E -->|No| G[Notify & Rollback]
        F --> H[Production]
        G --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#fce4ec
        style E fill:#ffebee
        style F fill:#e8f5e9
        style G fill:#ffccbc
        style H fill:#c8e6c9
    ```

### ML-Specific Considerations

Element | Traditional CI/CD | ML CI/CD  
---|---|---  
**Test Target** | Code | Code + Data + Model  
**Quality Metrics** | Test pass rate | Accuracy, Recall, F1 Score, etc.  
**Reproducibility** | Code version | Code + Data + Hyperparameters  
**Deployment Strategy** | Blue-Green, Canary | A/B Testing, Shadow Mode included  
**Monitoring** | System metrics | Model performance, Data drift  
  
### Responding to Data Changes

> **Important** : ML models require continuous retraining and monitoring because data distributions change over time (data drift).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # Simulate data drift
    np.random.seed(42)
    
    def generate_data(n_samples, drift_level=0.0):
        """drift_level: 0.0 (no change) ~ 1.0 (large change)"""
        X1 = np.random.normal(50 + drift_level * 20, 10, n_samples)
        X2 = np.random.normal(100 + drift_level * 30, 20, n_samples)
        y = ((X1 > 50) & (X2 > 100)).astype(int)
        return pd.DataFrame({'X1': X1, 'X2': X2}), y
    
    # Train initial model
    X_train, y_train = generate_data(1000, drift_level=0.0)
    X_test, y_test = generate_data(200, drift_level=0.0)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    initial_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Data drift over time
    drift_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []
    
    for drift in drift_levels:
        X_test_drift, y_test_drift = generate_data(200, drift_level=drift)
        acc = accuracy_score(y_test_drift, model.predict(X_test_drift))
        accuracies.append(acc)
    
    print("=== Impact of Data Drift ===")
    for drift, acc in zip(drift_levels, accuracies):
        print(f"Drift level {drift:.1f}: Accuracy = {acc:.3f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(drift_levels, accuracies, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', label='Minimum acceptable accuracy')
    plt.xlabel('Data Drift Level', fontsize=12)
    plt.ylabel('Model Accuracy', fontsize=12)
    plt.title('Model Performance Degradation due to Data Drift', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Impact of Data Drift ===
    Drift level 0.0: Accuracy = 0.920
    Drift level 0.2: Accuracy = 0.885
    Drift level 0.4: Accuracy = 0.835
    Drift level 0.6: Accuracy = 0.775
    Drift level 0.8: Accuracy = 0.720
    Drift level 1.0: Accuracy = 0.670
    

* * *

## 5.2 Automated Testing

### Testing Strategies for ML

Machine learning systems require multiple levels of testing:

  1. **Unit Tests** : Testing individual functions and modules
  2. **Data Validation Tests** : Testing data quality
  3. **Model Performance Tests** : Testing model performance
  4. **Integration Tests** : Testing the entire system

### 1\. Unit Tests for ML Code
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - pytest>=7.4.0
    
    # tests/test_preprocessing.py
    import pytest
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    class TestPreprocessing:
        """Unit tests for preprocessing functions"""
    
        def test_handle_missing_values(self):
            """Test missing value handling"""
            # Test data
            data = pd.DataFrame({
                'A': [1, 2, np.nan, 4],
                'B': [5, np.nan, 7, 8]
            })
    
            # Fill missing values with median
            filled = data.fillna(data.median())
    
            # Assertions
            assert filled.isnull().sum().sum() == 0, "Missing values remain"
            assert filled['A'].iloc[2] == 2.0, "Median not calculated correctly"
    
        def test_scaling(self):
            """Test scaling"""
            data = np.array([[1, 2], [3, 4], [5, 6]])
    
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
    
            # Verify mean and standard deviation after standardization
            assert np.allclose(scaled.mean(axis=0), 0, atol=1e-7), "Mean is not 0"
            assert np.allclose(scaled.std(axis=0), 1, atol=1e-7), "Std is not 1"
    
        def test_feature_engineering(self):
            """Test feature engineering"""
            df = pd.DataFrame({
                'height': [170, 180, 160],
                'weight': [65, 80, 55]
            })
    
            # Calculate BMI
            df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    
            # Compare with expected values
            expected_bmi = [22.49, 24.69, 21.48]
            assert np.allclose(df['bmi'].values, expected_bmi, atol=0.01)
    
    # Run with pytest
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 2\. Data Validation Tests
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - pytest>=7.4.0
    
    # tests/test_data_validation.py
    import pytest
    import pandas as pd
    import numpy as np
    
    class TestDataValidation:
        """Data quality validation tests"""
    
        @pytest.fixture
        def sample_data(self):
            """Sample data for testing"""
            return pd.DataFrame({
                'age': [25, 30, 35, 40, 45],
                'income': [50000, 60000, 70000, 80000, 90000],
                'score': [0.7, 0.8, 0.6, 0.9, 0.75]
            })
    
        def test_no_missing_values(self, sample_data):
            """Confirm no missing values"""
            assert sample_data.isnull().sum().sum() == 0, "Missing values exist"
    
        def test_data_types(self, sample_data):
            """Confirm correct data types"""
            assert sample_data['age'].dtype in [np.int64, np.float64]
            assert sample_data['income'].dtype in [np.int64, np.float64]
            assert sample_data['score'].dtype == np.float64
    
        def test_value_ranges(self, sample_data):
            """Confirm appropriate value ranges"""
            assert (sample_data['age'] >= 0).all(), "Age has negative values"
            assert (sample_data['age'] <= 120).all(), "Age is abnormally high"
            assert (sample_data['income'] >= 0).all(), "Income has negative values"
            assert (sample_data['score'] >= 0).all() and (sample_data['score'] <= 1).all(), \
                   "Score is outside 0-1 range"
    
        def test_data_shape(self, sample_data):
            """Confirm data shape is as expected"""
            assert sample_data.shape == (5, 3), f"Expected: (5, 3), Actual: {sample_data.shape}"
    
        def test_no_duplicates(self, sample_data):
            """Confirm no duplicate rows"""
            assert sample_data.duplicated().sum() == 0, "Duplicate rows exist"
    
        def test_statistical_properties(self, sample_data):
            """Confirm statistical properties are within expected range"""
            # Age mean should be in 20-60 range
            assert 20 <= sample_data['age'].mean() <= 60, "Age mean is abnormal"
    
            # Score standard deviation should be reasonable
            assert sample_data['score'].std() < 0.5, "Score variance is too large"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 3\. Model Performance Tests
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pytest>=7.4.0
    
    # tests/test_model_performance.py
    import pytest
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    class TestModelPerformance:
        """Model performance tests"""
    
        @pytest.fixture
        def trained_model(self):
            """Return a trained model"""
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=15,
                n_redundant=5, random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
    
            return model, X_test, y_test
    
        def test_minimum_accuracy(self, trained_model):
            """Confirm minimum accuracy is met"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            accuracy = accuracy_score(y_test, y_pred)
            min_accuracy = 0.80  # Require minimum 80% accuracy
    
            assert accuracy >= min_accuracy, \
                   f"Accuracy below threshold: {accuracy:.3f} < {min_accuracy}"
    
        def test_precision_recall(self, trained_model):
            """Confirm appropriate precision and recall"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
    
            assert precision >= 0.75, f"Precision is low: {precision:.3f}"
            assert recall >= 0.75, f"Recall is low: {recall:.3f}"
    
        def test_f1_score(self, trained_model):
            """Confirm F1 score meets threshold"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            f1 = f1_score(y_test, y_pred)
            min_f1 = 0.78
    
            assert f1 >= min_f1, f"F1 score below threshold: {f1:.3f} < {min_f1}"
    
        def test_no_performance_regression(self, trained_model):
            """Confirm performance hasn't degraded from previous"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            current_accuracy = accuracy_score(y_test, y_pred)
            baseline_accuracy = 0.85  # Previous baseline
            tolerance = 0.02  # Tolerance
    
            assert current_accuracy >= baseline_accuracy - tolerance, \
                   f"Performance degraded: {current_accuracy:.3f} < {baseline_accuracy - tolerance:.3f}"
    
        def test_prediction_distribution(self, trained_model):
            """Confirm prediction distribution is reasonable"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            # Confirm class imbalance is not extreme
            class_0_ratio = (y_pred == 0).sum() / len(y_pred)
    
            assert 0.2 <= class_0_ratio <= 0.8, \
                   f"Prediction distribution is biased: Class 0 ratio = {class_0_ratio:.2%}"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 4\. Integration Tests
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - pytest>=7.4.0
    
    # tests/test_integration.py
    import pytest
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    
    class TestIntegration:
        """Integration tests: End-to-end pipeline"""
    
        @pytest.fixture
        def sample_pipeline(self):
            """Complete machine learning pipeline"""
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
    
        @pytest.fixture
        def sample_data_with_issues(self):
            """Data with issues (missing values, outliers, etc.)"""
            np.random.seed(42)
            data = pd.DataFrame({
                'feature1': [1, 2, np.nan, 4, 5, 100],  # Missing values and outliers
                'feature2': [10, 20, 30, 40, np.nan, 60],
                'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            })
            labels = np.array([0, 0, 1, 1, 0, 1])
            return data, labels
    
        def test_pipeline_handles_missing_values(self, sample_pipeline, sample_data_with_issues):
            """Confirm pipeline can handle missing values"""
            X, y = sample_data_with_issues
    
            # Run pipeline
            sample_pipeline.fit(X, y)
            predictions = sample_pipeline.predict(X)
    
            # Confirm predictions were made for all data
            assert len(predictions) == len(y), "Number of predictions doesn't match input data"
            assert not np.isnan(predictions).any(), "Predictions contain NaN"
    
        def test_pipeline_reproducibility(self, sample_pipeline, sample_data_with_issues):
            """Confirm pipeline reproducibility"""
            X, y = sample_data_with_issues
    
            # First training and prediction
            sample_pipeline.fit(X, y)
            pred1 = sample_pipeline.predict(X)
    
            # Second training and prediction (same data)
            sample_pipeline.fit(X, y)
            pred2 = sample_pipeline.predict(X)
    
            # Confirm results are identical
            assert np.array_equal(pred1, pred2), "Results not reproducible with same data"
    
        def test_pipeline_training_and_inference(self, sample_pipeline):
            """Confirm training and inference flow works correctly"""
            # Training data
            np.random.seed(42)
            X_train = pd.DataFrame(np.random.randn(100, 3))
            y_train = np.random.randint(0, 2, 100)
    
            # Train
            sample_pipeline.fit(X_train, y_train)
    
            # Inference with new data
            X_new = pd.DataFrame(np.random.randn(10, 3))
            predictions = sample_pipeline.predict(X_new)
    
            # Confirm predictions are in appropriate format
            assert predictions.shape == (10,), "Prediction shape is invalid"
            assert set(predictions).issubset({0, 1}), "Predicted values not in expected classes"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### Running pytest Examples
    
    
    # Run all tests
    pytest tests/ -v
    
    # Run specific test file only
    pytest tests/test_model_performance.py -v
    
    # Detailed output with coverage report
    pytest tests/ -v --cov=src --cov-report=html
    

**Output example** :
    
    
    tests/test_preprocessing.py::TestPreprocessing::test_handle_missing_values PASSED
    tests/test_preprocessing.py::TestPreprocessing::test_scaling PASSED
    tests/test_preprocessing.py::TestPreprocessing::test_feature_engineering PASSED
    tests/test_data_validation.py::TestDataValidation::test_no_missing_values PASSED
    tests/test_data_validation.py::TestDataValidation::test_data_types PASSED
    tests/test_data_validation.py::TestDataValidation::test_value_ranges PASSED
    tests/test_model_performance.py::TestModelPerformance::test_minimum_accuracy PASSED
    tests/test_model_performance.py::TestModelPerformance::test_precision_recall PASSED
    tests/test_model_performance.py::TestModelPerformance::test_f1_score PASSED
    tests/test_integration.py::TestIntegration::test_pipeline_handles_missing_values PASSED
    
    ==================== 10 passed in 3.24s ====================
    

* * *

## 5.3 GitHub Actions for ML

### What is GitHub Actions?

**GitHub Actions** is a tool for automating CI/CD workflows on GitHub. It can automatically run tests, train models, and deploy when code changes.

### Basic Workflow Configuration
    
    
    # .github/workflows/ml-ci.yml
    name: ML CI Pipeline
    
    on:
      push:
        branches: [ main, develop ]
      pull_request:
        branches: [ main ]
    
    jobs:
      test:
        runs-on: ubuntu-latest
    
        steps:
        - name: Checkout code
          uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.9'
    
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
    
        - name: Run linting
          run: |
            pip install flake8
            flake8 src/ --max-line-length=100
    
        - name: Run unit tests
          run: |
            pytest tests/ -v --cov=src --cov-report=xml
    
        - name: Upload coverage reports
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml
            flags: unittests
    

### Automating Model Training
    
    
    # .github/workflows/train-model.yml
    name: Train and Validate Model
    
    on:
      schedule:
        # Run daily at 2 AM
        - cron: '0 2 * * *'
      workflow_dispatch:  # Also allow manual execution
    
    jobs:
      train:
        runs-on: ubuntu-latest
    
        steps:
        - name: Checkout code
          uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.9'
    
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
    
        - name: Download training data
          run: |
            python scripts/download_data.py
          env:
            DATA_URL: ${{ secrets.DATA_URL }}
    
        - name: Train model
          run: |
            python src/train.py --config config/train_config.yaml
    
        - name: Validate model
          run: |
            python src/validate.py --model models/latest_model.pkl
    
        - name: Upload model artifact
          uses: actions/upload-artifact@v3
          with:
            name: trained-model
            path: models/latest_model.pkl
    
        - name: Save metrics
          run: |
            python scripts/save_metrics.py --output metrics/metrics.json
    
        - name: Upload metrics
          uses: actions/upload-artifact@v3
          with:
            name: model-metrics
            path: metrics/metrics.json
    

### Performance Regression Testing
    
    
    # .github/workflows/performance-test.yml
    name: Model Performance Regression Test
    
    on:
      pull_request:
        branches: [ main ]
    
    jobs:
      performance-test:
        runs-on: ubuntu-latest
    
        steps:
        - name: Checkout code
          uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.9'
    
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
    
        - name: Download baseline model
          run: |
            # Fetch previous baseline model
            python scripts/download_baseline.py
    
        - name: Train new model
          run: |
            python src/train.py --config config/train_config.yaml
    
        - name: Compare performance
          id: compare
          run: |
            python scripts/compare_models.py \
              --baseline models/baseline_model.pkl \
              --new models/latest_model.pkl \
              --output comparison_result.json
    
        - name: Check performance threshold
          run: |
            python scripts/check_threshold.py \
              --result comparison_result.json \
              --min-accuracy 0.85 \
              --max-regression 0.02
    
        - name: Comment PR with results
          if: always()
          uses: actions/github-script@v6
          with:
            script: |
              const fs = require('fs');
              const result = JSON.parse(fs.readFileSync('comparison_result.json', 'utf8'));
    
              const comment = `
              ## Model Performance Comparison Results
    
              | Metric | Baseline | New Model | Change |
              |--------|----------|-----------|--------|
              | Accuracy  | ${result.baseline.accuracy.toFixed(3)} | ${result.new.accuracy.toFixed(3)} | ${(result.new.accuracy - result.baseline.accuracy).toFixed(3)} |
              | Precision | ${result.baseline.precision.toFixed(3)} | ${result.new.precision.toFixed(3)} | ${(result.new.precision - result.baseline.precision).toFixed(3)} |
              | Recall    | ${result.baseline.recall.toFixed(3)} | ${result.new.recall.toFixed(3)} | ${(result.new.recall - result.baseline.recall).toFixed(3)} |
              | F1 Score  | ${result.baseline.f1.toFixed(3)} | ${result.new.f1.toFixed(3)} | ${(result.new.f1 - result.baseline.f1).toFixed(3)} |
              `;
    
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment
              });
    

### Complete CI/CD YAML Example
    
    
    # .github/workflows/complete-ml-pipeline.yml
    name: Complete ML CI/CD Pipeline
    
    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]
    
    env:
      PYTHON_VERSION: '3.9'
      MODEL_REGISTRY: 's3://my-model-registry'
    
    jobs:
      lint-and-test:
        name: Lint and Test
        runs-on: ubuntu-latest
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Cache dependencies
          uses: actions/cache@v3
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
            pip install flake8 pytest pytest-cov
    
        - name: Lint with flake8
          run: |
            flake8 src/ tests/ --max-line-length=100
    
        - name: Run tests
          run: |
            pytest tests/ -v --cov=src --cov-report=xml
    
        - name: Upload coverage
          uses: codecov/codecov-action@v3
    
      data-validation:
        name: Data Validation
        runs-on: ubuntu-latest
        needs: lint-and-test
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install dependencies
          run: pip install -r requirements.txt
    
        - name: Validate data quality
          run: |
            python scripts/validate_data.py --data data/training_data.csv
    
      train-model:
        name: Train Model
        runs-on: ubuntu-latest
        needs: data-validation
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install dependencies
          run: pip install -r requirements.txt
    
        - name: Train model
          run: |
            python src/train.py \
              --data data/training_data.csv \
              --output models/model.pkl \
              --config config/train_config.yaml
    
        - name: Evaluate model
          run: |
            python src/evaluate.py \
              --model models/model.pkl \
              --data data/test_data.csv \
              --output metrics.json
    
        - name: Upload model
          uses: actions/upload-artifact@v3
          with:
            name: trained-model
            path: models/model.pkl
    
        - name: Upload metrics
          uses: actions/upload-artifact@v3
          with:
            name: metrics
            path: metrics.json
    
      deploy:
        name: Deploy Model
        runs-on: ubuntu-latest
        needs: train-model
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Download model
          uses: actions/download-artifact@v3
          with:
            name: trained-model
            path: models/
    
        - name: Deploy to staging
          run: |
            python scripts/deploy.py \
              --model models/model.pkl \
              --environment staging
          env:
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
        - name: Run smoke tests
          run: |
            python tests/smoke_tests.py --endpoint ${{ secrets.STAGING_ENDPOINT }}
    
        - name: Deploy to production
          if: success()
          run: |
            python scripts/deploy.py \
              --model models/model.pkl \
              --environment production
          env:
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    

* * *

## 5.4 Deployment Strategies

### 1\. Blue-Green Deployment

**Blue-Green deployment** is a method that minimizes downtime by preparing two identical environments (Blue and Green) and instantly switching between them.
    
    
    ```mermaid
    graph LR
        A[Traffic] --> B[Load Balancer]
        B --> C[Blue EnvironmentOld Model v1.0]
        B -.Switch.-> D[Green EnvironmentNew Model v2.0]
    
        style C fill:#a7c7e7
        style D fill:#90ee90
    ```
    
    
    # scripts/blue_green_deploy.py
    import boto3
    import time
    from typing import Dict
    
    class BlueGreenDeployer:
        """Blue-Green deployment implementation"""
    
        def __init__(self, load_balancer_name: str):
            self.elb = boto3.client('elbv2')
            self.lb_name = load_balancer_name
    
        def deploy_green(self, model_path: str, target_group_name: str):
            """Deploy new model to Green environment"""
            print(f"Deploying model to Green environment: {model_path}")
    
            # Deploy model to new target group
            # (Implementation depends on environment)
            self._deploy_model_to_target_group(model_path, target_group_name)
    
            print("Green environment deployment complete")
    
        def run_health_checks(self, target_group_arn: str) -> bool:
            """Run health checks"""
            print("Running health checks on Green environment...")
    
            response = self.elb.describe_target_health(
                TargetGroupArn=target_group_arn
            )
    
            healthy_count = sum(
                1 for target in response['TargetHealthDescriptions']
                if target['TargetHealth']['State'] == 'healthy'
            )
    
            total_count = len(response['TargetHealthDescriptions'])
    
            is_healthy = healthy_count == total_count and total_count > 0
            print(f"Health: {healthy_count}/{total_count} healthy")
    
            return is_healthy
    
        def switch_traffic(self, listener_arn: str, green_target_group_arn: str):
            """Switch traffic to Green environment"""
            print("Switching traffic to Green environment...")
    
            self.elb.modify_listener(
                ListenerArn=listener_arn,
                DefaultActions=[
                    {
                        'Type': 'forward',
                        'TargetGroupArn': green_target_group_arn
                    }
                ]
            )
    
            print("Traffic switch complete")
    
        def rollback(self, listener_arn: str, blue_target_group_arn: str):
            """Rollback to Blue environment"""
            print("Rolling back to Blue environment...")
    
            self.elb.modify_listener(
                ListenerArn=listener_arn,
                DefaultActions=[
                    {
                        'Type': 'forward',
                        'TargetGroupArn': blue_target_group_arn
                    }
                ]
            )
    
            print("Rollback complete")
    
        def _deploy_model_to_target_group(self, model_path: str, target_group: str):
            """Deploy model to specified target group (example implementation)"""
            # Actual implementation depends on infrastructure
            time.sleep(2)  # Simulate deployment
    
    # Usage example
    if __name__ == "__main__":
        deployer = BlueGreenDeployer("my-load-balancer")
    
        # Step 1: Deploy to Green environment
        deployer.deploy_green(
            model_path="s3://models/model_v2.pkl",
            target_group_name="green-target-group"
        )
    
        # Step 2: Health check
        green_tg_arn = "arn:aws:elasticloadbalancing:region:account:targetgroup/green/xxx"
        if deployer.run_health_checks(green_tg_arn):
            # Step 3: Switch traffic
            listener_arn = "arn:aws:elasticloadbalancing:region:account:listener/xxx"
            deployer.switch_traffic(listener_arn, green_tg_arn)
            print("✓ Blue-Green deployment successful")
        else:
            print("✗ Health check failed: Deployment aborted")
    

### 2\. Canary Deployment

**Canary deployment** is a method that gradually releases the new model to a small number of users and expands if there are no issues.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # scripts/canary_deploy.py
    import time
    import random
    from typing import List, Dict
    
    class CanaryDeployer:
        """Canary deployment implementation"""
    
        def __init__(self, old_model, new_model):
            self.old_model = old_model
            self.new_model = new_model
            self.canary_percentage = 0
            self.metrics = {'old': [], 'new': []}
    
        def predict(self, X):
            """Predict based on canary ratio"""
            if random.random() * 100 < self.canary_percentage:
                # Predict with new model
                prediction = self.new_model.predict(X)
                self.metrics['new'].append(prediction)
                return prediction, 'new'
            else:
                # Predict with old model
                prediction = self.old_model.predict(X)
                self.metrics['old'].append(prediction)
                return prediction, 'old'
    
        def increase_canary_traffic(self, increment: int = 10):
            """Increase canary ratio"""
            self.canary_percentage = min(100, self.canary_percentage + increment)
            print(f"Canary ratio: {self.canary_percentage}%")
    
        def rollback(self):
            """Rollback canary"""
            self.canary_percentage = 0
            print("Canary rollback: Using old model only")
    
        def full_rollout(self):
            """Complete transition to new model"""
            self.canary_percentage = 100
            print("Complete transition to new model")
    
        def get_metrics_comparison(self) -> Dict:
            """Compare metrics of old and new models"""
            return {
                'old_model_requests': len(self.metrics['old']),
                'new_model_requests': len(self.metrics['new']),
                'canary_percentage': self.canary_percentage
            }
    
    # Canary deployment simulation
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Prepare data and models
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    old_model = RandomForestClassifier(n_estimators=50, random_state=42)
    old_model.fit(X[:800], y[:800])
    
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X[:800], y[:800])
    
    # Canary deployment
    deployer = CanaryDeployer(old_model, new_model)
    
    # Gradually increase canary ratio
    stages = [10, 25, 50, 75, 100]
    
    for stage in stages:
        print(f"\n=== Stage: Canary {stage}% ===")
        deployer.canary_percentage = stage
    
        # Simulate 100 requests
        for i in range(100):
            idx = random.randint(800, 999)
            prediction, model_used = deployer.predict(X[idx:idx+1])
    
        # Check metrics
        metrics = deployer.get_metrics_comparison()
        print(f"Old model usage: {metrics['old_model_requests']} requests")
        print(f"New model usage: {metrics['new_model_requests']} requests")
    
        # Rollback if there are issues (always succeeds in this example)
        time.sleep(1)
    
    print("\n✓ Canary deployment successful: Complete transition to new model")
    

**Output example** :
    
    
    === Stage: Canary 10% ===
    Canary ratio: 10%
    Old model usage: 91 requests
    New model usage: 9 requests
    
    === Stage: Canary 25% ===
    Canary ratio: 25%
    Old model usage: 166 requests
    New model usage: 34 requests
    
    === Stage: Canary 50% ===
    Canary ratio: 50%
    Old model usage: 216 requests
    New model usage: 84 requests
    
    === Stage: Canary 75% ===
    Canary ratio: 75%
    Old model usage: 241 requests
    New model usage: 159 requests
    
    === Stage: Canary 100% ===
    Canary ratio: 100%
    Old model usage: 241 requests
    New model usage: 259 requests
    
    ✓ Canary deployment successful: Complete transition to new model
    

### 3\. Shadow Deployment

**Shadow deployment** is a method where the new model runs in production but users receive results from the old model. This allows evaluating the new model's performance without risk.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    # scripts/shadow_deploy.py
    import time
    import numpy as np
    from typing import Tuple, Dict
    from sklearn.metrics import accuracy_score
    
    class ShadowDeployer:
        """Shadow deployment implementation"""
    
        def __init__(self, production_model, shadow_model):
            self.production_model = production_model
            self.shadow_model = shadow_model
            self.shadow_predictions = []
            self.production_predictions = []
            self.ground_truth = []
    
        def predict(self, X, y_true=None):
            """Predict with both production and shadow models"""
            # Predict with production model (return to users)
            prod_pred = self.production_model.predict(X)
            self.production_predictions.extend(prod_pred)
    
            # Predict with shadow model (log only, not returned to users)
            shadow_pred = self.shadow_model.predict(X)
            self.shadow_predictions.extend(shadow_pred)
    
            # Ground truth labels (if available)
            if y_true is not None:
                self.ground_truth.extend(y_true)
    
            # Return only production model results to users
            return prod_pred
    
        def compare_models(self) -> Dict:
            """Compare performance of production and shadow models"""
            if not self.ground_truth:
                return {
                    'error': 'No ground truth available for comparison'
                }
    
            prod_accuracy = accuracy_score(
                self.ground_truth,
                self.production_predictions
            )
            shadow_accuracy = accuracy_score(
                self.ground_truth,
                self.shadow_predictions
            )
    
            # Prediction agreement
            agreement = np.mean(
                np.array(self.production_predictions) == np.array(self.shadow_predictions)
            )
    
            return {
                'production_accuracy': prod_accuracy,
                'shadow_accuracy': shadow_accuracy,
                'improvement': shadow_accuracy - prod_accuracy,
                'prediction_agreement': agreement,
                'total_predictions': len(self.production_predictions)
            }
    
        def should_promote_shadow(self, min_improvement: float = 0.02) -> bool:
            """Determine if shadow model should be promoted to production"""
            comparison = self.compare_models()
    
            if 'error' in comparison:
                return False
    
            return comparison['improvement'] >= min_improvement
    
    # Shadow deployment simulation
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Production model (RandomForest)
    production_model = RandomForestClassifier(n_estimators=50, random_state=42)
    production_model.fit(X_train, y_train)
    
    # Shadow model (GradientBoosting - higher performance)
    shadow_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    shadow_model.fit(X_train, y_train)
    
    # Shadow deployment
    deployer = ShadowDeployer(production_model, shadow_model)
    
    print("=== Shadow Deployment Started ===\n")
    
    # Simulate production traffic (process in batches)
    batch_size = 50
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
    
        # Predict with both models (only production results returned to users)
        deployer.predict(X_batch, y_batch)
    
        time.sleep(0.1)  # Simulate production environment
    
    # Model comparison
    comparison = deployer.compare_models()
    
    print("=== Model Performance Comparison ===")
    print(f"Production model accuracy: {comparison['production_accuracy']:.3f}")
    print(f"Shadow model accuracy: {comparison['shadow_accuracy']:.3f}")
    print(f"Improvement: {comparison['improvement']:.3f} ({comparison['improvement']*100:+.1f}%)")
    print(f"Prediction agreement: {comparison['prediction_agreement']:.2%}")
    print(f"Total predictions: {comparison['total_predictions']}")
    
    # Promotion decision
    if deployer.should_promote_shadow(min_improvement=0.02):
        print("\n✓ Promoting shadow model to production")
    else:
        print("\n✗ Insufficient improvement: Shadow model not promoted")
    

**Output example** :
    
    
    === Shadow Deployment Started ===
    
    === Model Performance Comparison ===
    Production model accuracy: 0.883
    Shadow model accuracy: 0.915
    Improvement: 0.032 (+3.2%)
    Prediction agreement: 94.83%
    Total predictions: 600
    
    ✓ Promoting shadow model to production
    

### 4\. A/B Testing
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    # scripts/ab_test.py
    import numpy as np
    from scipy import stats
    from typing import Dict, List
    
    class ABTester:
        """A/B testing implementation"""
    
        def __init__(self, model_a, model_b):
            self.model_a = model_a
            self.model_b = model_b
            self.results_a = []
            self.results_b = []
    
        def assign_and_predict(self, X, y_true, user_id: int):
            """Assign user to group A/B and predict"""
            # Determine group by user ID hash (maintain consistency)
            group = 'A' if hash(user_id) % 2 == 0 else 'B'
    
            if group == 'A':
                prediction = self.model_a.predict(X)
                correct = (prediction == y_true).astype(int)
                self.results_a.extend(correct)
            else:
                prediction = self.model_b.predict(X)
                correct = (prediction == y_true).astype(int)
                self.results_b.extend(correct)
    
            return prediction, group
    
        def calculate_statistics(self) -> Dict:
            """Calculate statistical significance"""
            accuracy_a = np.mean(self.results_a)
            accuracy_b = np.mean(self.results_b)
    
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
    
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.std(self.results_a)**2 + np.std(self.results_b)**2) / 2
            )
            cohens_d = (accuracy_b - accuracy_a) / pooled_std if pooled_std > 0 else 0
    
            return {
                'group_a_accuracy': accuracy_a,
                'group_b_accuracy': accuracy_b,
                'difference': accuracy_b - accuracy_a,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'sample_size_a': len(self.results_a),
                'sample_size_b': len(self.results_b)
            }
    
        def recommend_winner(self) -> str:
            """Recommend winner"""
            stats_result = self.calculate_statistics()
    
            if not stats_result['is_significant']:
                return "No significant difference"
    
            if stats_result['difference'] > 0:
                return "Model B (statistically significant improvement)"
            else:
                return "Model A (statistically significant better)"
    
    # A/B testing simulation
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X, y = make_classification(
        n_samples=3000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Model A (current)
    model_a = RandomForestClassifier(n_estimators=50, random_state=42)
    model_a.fit(X_train, y_train)
    
    # Model B (new)
    model_b = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_b.fit(X_train, y_train)
    
    # Run A/B test
    ab_tester = ABTester(model_a, model_b)
    
    print("=== A/B Testing Started ===\n")
    
    # Predict for each user (simulation)
    for user_id in range(len(X_test)):
        prediction, group = ab_tester.assign_and_predict(
            X_test[user_id:user_id+1],
            y_test[user_id],
            user_id
        )
    
    # Statistical analysis
    stats_result = ab_tester.calculate_statistics()
    
    print("=== A/B Test Results ===")
    print(f"Group A (Current Model):")
    print(f"  Sample size: {stats_result['sample_size_a']}")
    print(f"  Accuracy: {stats_result['group_a_accuracy']:.3f}")
    
    print(f"\nGroup B (New Model):")
    print(f"  Sample size: {stats_result['sample_size_b']}")
    print(f"  Accuracy: {stats_result['group_b_accuracy']:.3f}")
    
    print(f"\nStatistical Analysis:")
    print(f"  Difference: {stats_result['difference']:.3f} ({stats_result['difference']*100:+.1f}%)")
    print(f"  p-value: {stats_result['p_value']:.4f}")
    print(f"  Statistically significant: {'Yes' if stats_result['is_significant'] else 'No'}")
    print(f"  Effect size (Cohen's d): {stats_result['cohens_d']:.3f}")
    
    print(f"\nRecommendation: {ab_tester.recommend_winner()}")
    

### Comparison of Deployment Strategies

Strategy | Risk | Rollback Speed | Use Case  
---|---|---|---  
**Blue-Green** | Medium | Instant | Rapid switching required  
**Canary** | Low | Gradual | Minimize risk  
**Shadow** | Lowest | Not needed (not in production) | Performance evaluation only  
**A/B Testing** | Low-Medium | Gradual | Statistical validation required  
  
* * *

## 5.5 End-to-End Example

### Implementing a Complete CI/CD Pipeline

Here we build a complete pipeline including data validation, model training, deployment, and monitoring.

#### Directory Structure
    
    
    ml-project/
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       ├── train.yml
    │       └── deploy.yml
    ├── src/
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── validation.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── train.py
    │   │   └── evaluate.py
    │   └── deploy/
    │       ├── __init__.py
    │       └── deployment.py
    ├── tests/
    │   ├── test_data_validation.py
    │   ├── test_model.py
    │   └── test_integration.py
    ├── config/
    │   └── model_config.yaml
    ├── requirements.txt
    └── README.md
    

#### Data Validation Module
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    # src/data/validation.py
    import pandas as pd
    import numpy as np
    from typing import Dict, List
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class DataValidator:
        """Data quality validation class"""
    
        def __init__(self, schema: Dict):
            self.schema = schema
            self.validation_errors = []
    
        def validate(self, df: pd.DataFrame) -> bool:
            """Execute all validations"""
            self.validation_errors = []
    
            checks = [
                self._check_schema(df),
                self._check_missing_values(df),
                self._check_value_ranges(df),
                self._check_data_types(df),
                self._check_duplicates(df)
            ]
    
            all_passed = all(checks)
    
            if all_passed:
                logger.info("✓ All data validations passed")
            else:
                logger.error(f"✗ {len(self.validation_errors)} validation errors")
                for error in self.validation_errors:
                    logger.error(f"  - {error}")
    
            return all_passed
    
        def _check_schema(self, df: pd.DataFrame) -> bool:
            """Validate schema (column names)"""
            expected_columns = set(self.schema.keys())
            actual_columns = set(df.columns)
    
            if expected_columns != actual_columns:
                missing = expected_columns - actual_columns
                extra = actual_columns - expected_columns
    
                if missing:
                    self.validation_errors.append(f"Missing columns: {missing}")
                if extra:
                    self.validation_errors.append(f"Extra columns: {extra}")
                return False
    
            return True
    
        def _check_missing_values(self, df: pd.DataFrame) -> bool:
            """Validate missing values"""
            missing_counts = df.isnull().sum()
            total_rows = len(df)
    
            for col, count in missing_counts.items():
                if count > 0:
                    threshold = self.schema.get(col, {}).get('max_missing_rate', 0.05)
                    missing_rate = count / total_rows
    
                    if missing_rate > threshold:
                        self.validation_errors.append(
                            f"{col}: Missing rate {missing_rate:.2%} > {threshold:.2%}"
                        )
                        return False
    
            return True
    
        def _check_value_ranges(self, df: pd.DataFrame) -> bool:
            """Validate value ranges"""
            all_valid = True
    
            for col, col_schema in self.schema.items():
                if 'min' in col_schema:
                    if (df[col] < col_schema['min']).any():
                        self.validation_errors.append(
                            f"{col}: Minimum value violation (< {col_schema['min']})"
                        )
                        all_valid = False
    
                if 'max' in col_schema:
                    if (df[col] > col_schema['max']).any():
                        self.validation_errors.append(
                            f"{col}: Maximum value violation (> {col_schema['max']})"
                        )
                        all_valid = False
    
            return all_valid
    
        def _check_data_types(self, df: pd.DataFrame) -> bool:
            """Validate data types"""
            for col, col_schema in self.schema.items():
                if 'dtype' in col_schema:
                    expected_dtype = col_schema['dtype']
                    actual_dtype = str(df[col].dtype)
    
                    if expected_dtype not in actual_dtype:
                        self.validation_errors.append(
                            f"{col}: Type mismatch (Expected: {expected_dtype}, Actual: {actual_dtype})"
                        )
                        return False
    
            return True
    
        def _check_duplicates(self, df: pd.DataFrame) -> bool:
            """Validate duplicate rows"""
            duplicates = df.duplicated().sum()
    
            if duplicates > 0:
                max_duplicates = self.schema.get('_global', {}).get('max_duplicates', 0)
                if duplicates > max_duplicates:
                    self.validation_errors.append(
                        f"Too many duplicate rows: {duplicates} > {max_duplicates}"
                    )
                    return False
    
            return True
    
    # Usage example
    if __name__ == "__main__":
        # Schema definition
        schema = {
            'age': {'dtype': 'int', 'min': 0, 'max': 120, 'max_missing_rate': 0.05},
            'income': {'dtype': 'float', 'min': 0, 'max_missing_rate': 0.1},
            'score': {'dtype': 'float', 'min': 0, 'max': 1, 'max_missing_rate': 0.02},
            '_global': {'max_duplicates': 10}
        }
    
        # Test data
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'score': [0.7, 0.8, 0.6, 0.9, 0.75]
        })
    
        validator = DataValidator(schema)
        is_valid = validator.validate(df)
    
        if is_valid:
            print("Data validation successful")
        else:
            print("Data validation failed")
    

#### Model Training and Metrics Saving
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Model Training and Metrics Saving
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # src/models/train.py
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class ModelTrainer:
        """Model training class"""
    
        def __init__(self, config: dict):
            self.config = config
            self.model = None
            self.metrics = {}
    
        def train(self, X_train, y_train):
            """Train model"""
            logger.info("Starting model training...")
    
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', None),
                random_state=self.config.get('random_state', 42)
            )
    
            self.model.fit(X_train, y_train)
    
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=self.config.get('cv_folds', 5)
            )
    
            logger.info(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
            self.metrics['cv_accuracy'] = cv_scores.mean()
            self.metrics['cv_std'] = cv_scores.std()
    
        def evaluate(self, X_test, y_test):
            """Evaluate model"""
            logger.info("Evaluating model...")
    
            y_pred = self.model.predict(X_test)
    
            self.metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
            self.metrics['test_precision'] = precision_score(y_test, y_pred, average='weighted')
            self.metrics['test_recall'] = recall_score(y_test, y_pred, average='weighted')
            self.metrics['test_f1'] = f1_score(y_test, y_pred, average='weighted')
    
            logger.info(f"Test accuracy: {self.metrics['test_accuracy']:.3f}")
            logger.info(f"F1 score: {self.metrics['test_f1']:.3f}")
    
            return self.metrics
    
        def save_model(self, path: str):
            """Save model"""
            joblib.dump(self.model, path)
            logger.info(f"Model saved: {path}")
    
        def save_metrics(self, path: str):
            """Save metrics"""
            with open(path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved: {path}")
    
        def check_performance_threshold(self, min_accuracy: float = 0.8) -> bool:
            """Check performance threshold"""
            if self.metrics['test_accuracy'] < min_accuracy:
                logger.error(
                    f"Accuracy below threshold: {self.metrics['test_accuracy']:.3f} < {min_accuracy}"
                )
                return False
    
            logger.info(f"✓ Performance threshold met: {self.metrics['test_accuracy']:.3f}")
            return True
    
    # Usage example
    if __name__ == "__main__":
        from sklearn.datasets import make_classification
    
        # Generate data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Configuration
        config = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'cv_folds': 5
        }
    
        # Training and evaluation
        trainer = ModelTrainer(config)
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)
    
        # Performance check
        if trainer.check_performance_threshold(min_accuracy=0.85):
            trainer.save_model('models/model.pkl')
            trainer.save_metrics('models/metrics.json')
        else:
            logger.error("Model does not meet criteria")
    

#### Complete GitHub Actions Workflow
    
    
    # .github/workflows/complete-ml-cicd.yml
    name: Complete ML CI/CD Pipeline
    
    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]
      schedule:
        # Retrain daily at 2 AM
        - cron: '0 2 * * *'
    
    env:
      PYTHON_VERSION: '3.9'
      MIN_ACCURACY: '0.85'
    
    jobs:
      code-quality:
        name: Code Quality Checks
        runs-on: ubuntu-latest
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install dependencies
          run: |
            pip install flake8 black isort
    
        - name: Check code formatting (black)
          run: black --check src/ tests/
    
        - name: Check import sorting (isort)
          run: isort --check-only src/ tests/
    
        - name: Lint (flake8)
          run: flake8 src/ tests/ --max-line-length=100
    
      test:
        name: Run Tests
        runs-on: ubuntu-latest
        needs: code-quality
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Cache dependencies
          uses: actions/cache@v3
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
            pip install pytest pytest-cov
    
        - name: Run unit tests
          run: pytest tests/ -v --cov=src --cov-report=xml
    
        - name: Upload coverage
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml
    
      data-validation:
        name: Validate Training Data
        runs-on: ubuntu-latest
        needs: test
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install dependencies
          run: pip install -r requirements.txt
    
        - name: Download data
          run: |
            # Download data (replace with actual implementation)
            python scripts/download_data.py
    
        - name: Validate data quality
          run: |
            python src/data/validation.py --data data/training_data.csv
    
      train:
        name: Train Model
        runs-on: ubuntu-latest
        needs: data-validation
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install dependencies
          run: pip install -r requirements.txt
    
        - name: Train model
          run: |
            python src/models/train.py \
              --data data/training_data.csv \
              --config config/model_config.yaml \
              --output models/model.pkl
    
        - name: Evaluate model
          id: evaluate
          run: |
            python src/models/evaluate.py \
              --model models/model.pkl \
              --data data/test_data.csv \
              --metrics-output models/metrics.json
    
        - name: Check performance threshold
          run: |
            python scripts/check_threshold.py \
              --metrics models/metrics.json \
              --min-accuracy ${{ env.MIN_ACCURACY }}
    
        - name: Upload model artifact
          uses: actions/upload-artifact@v3
          with:
            name: trained-model
            path: |
              models/model.pkl
              models/metrics.json
            retention-days: 30
    
      deploy-staging:
        name: Deploy to Staging
        runs-on: ubuntu-latest
        needs: train
        if: github.ref == 'refs/heads/main'
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Download model
          uses: actions/download-artifact@v3
          with:
            name: trained-model
            path: models/
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Install deployment tools
          run: pip install boto3
    
        - name: Deploy to staging (Canary)
          run: |
            python src/deploy/deployment.py \
              --model models/model.pkl \
              --environment staging \
              --strategy canary \
              --canary-percentage 10
          env:
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
        - name: Run smoke tests
          run: |
            python tests/smoke_tests.py \
              --endpoint ${{ secrets.STAGING_ENDPOINT }} \
              --timeout 300
    
      deploy-production:
        name: Deploy to Production
        runs-on: ubuntu-latest
        needs: deploy-staging
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        environment:
          name: production
          url: https://ml-api.example.com
    
        steps:
        - uses: actions/checkout@v3
    
        - name: Download model
          uses: actions/download-artifact@v3
          with:
            name: trained-model
            path: models/
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ env.PYTHON_VERSION }}
    
        - name: Deploy with Blue-Green strategy
          run: |
            python src/deploy/deployment.py \
              --model models/model.pkl \
              --environment production \
              --strategy blue-green
          env:
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
        - name: Monitor deployment
          run: |
            python scripts/monitor_deployment.py \
              --duration 600 \
              --rollback-on-error
    
        - name: Notify deployment success
          if: success()
          uses: slackapi/slack-github-action@v1
          with:
            webhook-url: ${{ secrets.SLACK_WEBHOOK }}
            payload: |
              {
                "text": "✅ ML Model deployed to production successfully"
              }
    

* * *

## 5.6 Chapter Summary

### What We Learned

  1. **ML-Specific CI/CD**

     * Integrated management of code + data + models
     * Responding to data drift
     * Continuous retraining and monitoring
  2. **Automated Testing Strategies**

     * Unit Tests: Code correctness
     * Data Validation: Data quality
     * Model Performance Tests: Model performance
     * Integration Tests: Entire system
  3. **Utilizing GitHub Actions**

     * Automated test execution
     * Model training pipeline
     * Performance regression testing
  4. **Deployment Strategies**

     * Blue-Green: Rapid switching
     * Canary: Risk minimization
     * Shadow: Safe evaluation
     * A/B Testing: Statistical validation
  5. **Production Operations**

     * End-to-end pipeline
     * Automated deployment
     * Monitoring and rollback

### Best Practices

Principle | Description  
---|---  
**Automation First** | Minimize manual operations and ensure reproducibility  
**Gradual Deployment** | Gradually reduce risk with Canary or Shadow  
**Set Performance Thresholds** | Automatically pass/fail based on clear criteria  
**Rapid Rollback** | Mechanism to immediately revert to previous version when issues occur  
**Comprehensive Monitoring** | Continuously monitor both model performance and system metrics  
  
### Next Steps

Based on the CI/CD pipeline learned in this chapter, you can consider the following extensions:

  * Feature store integration
  * Experiment management with MLflow
  * Scalable deployment with Kubernetes
  * Real-time inference systems
  * Integration of model explainability

* * *

## Exercises

### Problem 1 (Difficulty: easy)

List and explain three main differences between traditional software development CI/CD and machine learning CI/CD.

Answer

**Answer** :

  1. **Difference in Test Targets**

     * Traditional: Code only
     * ML: Three elements - Code + Data + Model
     * Reason: In ML, data quality and model performance directly impact results
  2. **Difference in Quality Metrics**

     * Traditional: Test pass rate, code coverage
     * ML: Accuracy, recall, F1 score, data drift
     * Reason: ML is evaluated with statistical performance metrics
  3. **Need for Continuous Updates**

     * Traditional: Updates when adding features or fixing bugs
     * ML: Regular retraining in response to data distribution changes
     * Reason: Model performance degrades due to data drift

### Problem 2 (Difficulty: medium)

Create a test using pytest that confirms the model has a minimum accuracy of 85%.

Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pytest>=7.4.0
    
    import pytest
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    class TestModelAccuracy:
        """Model accuracy test"""
    
        @pytest.fixture
        def trained_model_and_data(self):
            """Prepare trained model and test data"""
            X, y = make_classification(
                n_samples=1000, n_features=20,
                n_informative=15, random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    
            model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            model.fit(X_train, y_train)
    
            return model, X_test, y_test
    
        def test_minimum_accuracy_85_percent(self, trained_model_and_data):
            """Confirm minimum accuracy of 85% is met"""
            model, X_test, y_test = trained_model_and_data
    
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
    
            min_accuracy = 0.85
    
            assert accuracy >= min_accuracy, \
                   f"Accuracy below threshold: {accuracy:.3f} < {min_accuracy}"
    
            print(f"✓ Test passed: Accuracy = {accuracy:.3f}")
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

**Execution** :
    
    
    pytest test_model_accuracy.py -v
    

### Problem 3 (Difficulty: medium)

Explain the differences between Blue-Green deployment and Canary deployment, and describe in what situations each should be used.

Answer

**Answer** :

**Blue-Green Deployment** :

  * Characteristics: Prepare two complete environments and instantly switch
  * Advantages: Instant rollback possible, no downtime
  * Disadvantages: Requires double resources, affects all users at once

**Canary Deployment** :

  * Characteristics: Gradually release new version to subset of users
  * Advantages: Risk minimization, early problem detection
  * Disadvantages: Takes time to switch, complex management

**When to Use Each** :

Situation | Recommended Strategy | Reason  
---|---|---  
Rapid deployment needed | Blue-Green | Can switch instantly  
Want to minimize risk | Canary | Gradually expand impact scope  
Large changes | Canary | Detect problems early  
Small improvements | Blue-Green | Simple and fast  
Resource constraints | Canary | No additional resources needed  
  
### Problem 4 (Difficulty: hard)

Create a Python script that detects data drift and retrains the model when accuracy falls below a threshold.

Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from scipy.stats import ks_2samp
    import joblib
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class DriftDetectorAndRetrainer:
        """Data drift detection and automatic retraining"""
    
        def __init__(self, model_path: str, accuracy_threshold: float = 0.8):
            self.model_path = model_path
            self.accuracy_threshold = accuracy_threshold
            self.model = joblib.load(model_path) if model_path else None
            self.reference_data = None
    
        def set_reference_data(self, X_ref: np.ndarray):
            """Set reference data"""
            self.reference_data = X_ref
            logger.info(f"Reference data set: {X_ref.shape}")
    
        def detect_drift(self, X_new: np.ndarray, alpha: float = 0.05) -> bool:
            """Detect drift using Kolmogorov-Smirnov test"""
            if self.reference_data is None:
                raise ValueError("Reference data not set")
    
            drifts = []
    
            for i in range(X_new.shape[1]):
                statistic, p_value = ks_2samp(
                    self.reference_data[:, i],
                    X_new[:, i]
                )
    
                is_drift = p_value < alpha
                drifts.append(is_drift)
    
                if is_drift:
                    logger.warning(
                        f"Drift detected in feature {i}: p={p_value:.4f} < {alpha}"
                    )
    
            drift_ratio = sum(drifts) / len(drifts)
            logger.info(f"Drift detection rate: {drift_ratio:.2%}")
    
            return drift_ratio > 0.3  # Drift in >30% of features
    
        def check_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
            """Check model performance"""
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
    
            logger.info(f"Current accuracy: {accuracy:.3f}")
    
            return accuracy
    
        def retrain(self, X_train: np.ndarray, y_train: np.ndarray):
            """Retrain model"""
            logger.info("Starting model retraining...")
    
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.model.fit(X_train, y_train)
    
            logger.info("Model retraining complete")
    
        def save_model(self, path: str = None):
            """Save model"""
            save_path = path or self.model_path
            joblib.dump(self.model, save_path)
            logger.info(f"Model saved: {save_path}")
    
        def run_monitoring_cycle(
            self,
            X_new: np.ndarray,
            y_new: np.ndarray,
            X_train: np.ndarray,
            y_train: np.ndarray
        ):
            """Run monitoring cycle"""
            logger.info("=== Monitoring Cycle Started ===")
    
            # 1. Drift detection
            has_drift = self.detect_drift(X_new)
    
            # 2. Performance check
            accuracy = self.check_performance(X_new, y_new)
    
            # 3. Retraining decision
            needs_retrain = (
                has_drift or
                accuracy < self.accuracy_threshold
            )
    
            if needs_retrain:
                logger.warning(
                    f"Retraining needed: drift={has_drift}, "
                    f"accuracy={accuracy:.3f} < {self.accuracy_threshold}"
                )
    
                # 4. Execute retraining
                self.retrain(X_train, y_train)
    
                # 5. Evaluate new model
                new_accuracy = self.check_performance(X_new, y_new)
                logger.info(f"Accuracy after retraining: {new_accuracy:.3f}")
    
                # 6. Save model
                self.save_model()
    
                return True
            else:
                logger.info("Retraining not needed: Model operating normally")
                return False
    
    # Usage example
    if __name__ == "__main__":
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
    
        # Initial data and model
        X_initial, y_initial = make_classification(
            n_samples=1000, n_features=20, random_state=42
        )
        X_train, X_ref, y_train, y_ref = train_test_split(
            X_initial, y_initial, test_size=0.3, random_state=42
        )
    
        # Train initial model
        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train, y_train)
        joblib.dump(initial_model, 'model.pkl')
    
        # Monitoring system
        monitor = DriftDetectorAndRetrainer(
            model_path='model.pkl',
            accuracy_threshold=0.85
        )
        monitor.set_reference_data(X_ref)
    
        # New data (with drift)
        X_new, y_new = make_classification(
            n_samples=200, n_features=20, random_state=100
        )
        # Simulate drift
        X_new = X_new + np.random.normal(0, 1.5, X_new.shape)
    
        # Run monitoring cycle
        was_retrained = monitor.run_monitoring_cycle(
            X_new, y_new, X_train, y_train
        )
    
        if was_retrained:
            print("\n✓ Model was retrained")
        else:
            print("\n✓ Model is operating normally")
    

### Problem 5 (Difficulty: hard)

Create a GitHub Actions workflow that checks whether model performance has not degraded from the previous baseline during a Pull Request, and comments the results on the PR.

Answer
    
    
    # .github/workflows/pr-model-check.yml
    name: PR Model Performance Check
    
    on:
      pull_request:
        branches: [ main ]
    
    jobs:
      compare-model-performance:
        runs-on: ubuntu-latest
    
        steps:
        - name: Checkout PR branch
          uses: actions/checkout@v3
    
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.9'
    
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
            pip install scikit-learn joblib
    
        - name: Download baseline model
          run: |
            # Download previous baseline model
            wget https://storage.example.com/models/baseline_model.pkl \
              -O models/baseline_model.pkl
            wget https://storage.example.com/models/baseline_metrics.json \
              -O models/baseline_metrics.json
    
        - name: Train new model (PR version)
          run: |
            python src/models/train.py \
              --data data/training_data.csv \
              --output models/new_model.pkl \
              --metrics-output models/new_metrics.json
    
        - name: Compare models
          id: compare
          run: |
            python scripts/compare_models.py \
              --baseline models/baseline_model.pkl \
              --new models/new_model.pkl \
              --output comparison.json
    
        - name: Read comparison results
          id: results
          run: |
            echo "comparison<> $GITHUB_OUTPUT
            cat comparison.json >> $GITHUB_OUTPUT
            echo "EOF" >> $GITHUB_OUTPUT
    
        - name: Comment on PR
          uses: actions/github-script@v6
          with:
            script: |
              const fs = require('fs');
              const comparison = JSON.parse(fs.readFileSync('comparison.json', 'utf8'));
    
              const accuracyDiff = comparison.new.accuracy - comparison.baseline.accuracy;
              const f1Diff = comparison.new.f1 - comparison.baseline.f1;
    
              const statusEmoji = accuracyDiff >= -0.01 ? '✅' : '❌';
    
              const comment = `
              ${statusEmoji} **Model Performance Comparison Results**
    
              | Metric | Baseline | New Model (PR) | Change |
              |--------|----------|----------------|--------|
              | **Accuracy** | ${comparison.baseline.accuracy.toFixed(3)} | ${comparison.new.accuracy.toFixed(3)} | ${accuracyDiff >= 0 ? '+' : ''}${accuracyDiff.toFixed(3)} |
              | **Precision** | ${comparison.baseline.precision.toFixed(3)} | ${comparison.new.precision.toFixed(3)} | ${(comparison.new.precision - comparison.baseline.precision).toFixed(3)} |
              | **Recall** | ${comparison.baseline.recall.toFixed(3)} | ${comparison.new.recall.toFixed(3)} | ${(comparison.new.recall - comparison.baseline.recall).toFixed(3)} |
              | **F1 Score** | ${comparison.baseline.f1.toFixed(3)} | ${comparison.new.f1.toFixed(3)} | ${f1Diff >= 0 ? '+' : ''}${f1Diff.toFixed(3)} |
    
              ### Decision
              ${accuracyDiff >= -0.01 ?
                '✅ **Passed**: Performance degradation within acceptable range' :
                '❌ **Failed**: Significant performance degradation (Acceptable: -1%)'}
    
              
              Details
    
              - Training time: ${comparison.training_time} seconds
              - Model size: ${(comparison.model_size_mb).toFixed(2)} MB
              - Test samples: ${comparison.test_samples}
    
              
              `;
    
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment
              });
    
        - name: Check if performance regression
          run: |
            python scripts/check_regression.py \
              --comparison comparison.json \
              --max-regression 0.01
    

**Helper script (scripts/compare_models.py)** :
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    """
    Example: Helper script (scripts/compare_models.py):
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import argparse
    import json
    import joblib
    import time
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    def compare_models(baseline_path, new_path, output_path):
        # Load models
        baseline_model = joblib.load(baseline_path)
        new_model = joblib.load(new_path)
    
        # Test data
        X, y = load_iris(return_X_y=True)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # Baseline evaluation
        baseline_pred = baseline_model.predict(X_test)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, baseline_pred),
            'precision': precision_score(y_test, baseline_pred, average='weighted'),
            'recall': recall_score(y_test, baseline_pred, average='weighted'),
            'f1': f1_score(y_test, baseline_pred, average='weighted')
        }
    
        # New model evaluation
        start = time.time()
        new_pred = new_model.predict(X_test)
        training_time = time.time() - start
    
        new_metrics = {
            'accuracy': accuracy_score(y_test, new_pred),
            'precision': precision_score(y_test, new_pred, average='weighted'),
            'recall': recall_score(y_test, new_pred, average='weighted'),
            'f1': f1_score(y_test, new_pred, average='weighted')
        }
    
        # Comparison result
        result = {
            'baseline': baseline_metrics,
            'new': new_metrics,
            'training_time': training_time,
            'model_size_mb': 1.5,  # Calculate actual size
            'test_samples': len(y_test)
        }
    
        # Save
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
        print(f"Comparison results saved: {output_path}")
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--baseline', required=True)
        parser.add_argument('--new', required=True)
        parser.add_argument('--output', required=True)
        args = parser.parse_args()
    
        compare_models(args.baseline, args.new, args.output)
    

* * *

## References

  1. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  2. Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  3. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  4. Forsgren, N., Humble, J., & Kim, G. (2018). _Accelerate: The Science of Lean Software and DevOps_. IT Revolution Press.
  5. Sato, D., Wider, A., & Windheuser, C. (2019). "Continuous Delivery for Machine Learning." _Martin Fowler's Blog_.
