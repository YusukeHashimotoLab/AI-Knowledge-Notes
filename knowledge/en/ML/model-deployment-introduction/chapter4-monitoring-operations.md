---
title: "Chapter 4: Monitoring and Operations"
chapter_title: "Chapter 4: Monitoring and Operations"
subtitle: Continuous Monitoring and Improvement in Production Environments
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
version: 1.0
created_at: 2025-10-23
---

This chapter covers Monitoring and Operations. You will learn Build structured logging, metrics monitoring with Prometheus + Grafana, and Detect data drift.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Build structured logging and monitoring systems
  * ✅ Implement metrics monitoring with Prometheus + Grafana
  * ✅ Detect data drift and model drift
  * ✅ Implement A/B testing and canary deployments
  * ✅ Design model update and retraining pipelines
  * ✅ Apply production operations best practices

* * *

## 4.1 Logging and Monitoring

### Structured Logging (JSON Logging)

**Structured logging** records logs in a machine-readable format (JSON), making analysis and search easier.
    
    
    import json
    import logging
    from datetime import datetime
    from typing import Dict, Any
    
    class JSONFormatter(logging.Formatter):
        """Custom formatter for structured JSON logs"""
    
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
    
            # Additional custom fields
            if hasattr(record, 'prediction_id'):
                log_data['prediction_id'] = record.prediction_id
            if hasattr(record, 'model_version'):
                log_data['model_version'] = record.model_version
            if hasattr(record, 'latency_ms'):
                log_data['latency_ms'] = record.latency_ms
            if hasattr(record, 'input_features'):
                log_data['input_features'] = record.input_features
    
            # Exception information
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
    
            return json.dumps(log_data)
    
    
    # Logger setup
    def setup_logger(name: str, log_file: str = 'model_predictions.log') -> logging.Logger:
        """Configure a logger using structured logging"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
    
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
    
        # Console handler (for development)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
        return logger
    
    
    # Usage example
    logger = setup_logger('ml_model')
    
    # Regular log
    logger.info('Model service started')
    
    # Prediction log (with additional metadata)
    import time
    start_time = time.time()
    
    # ... prediction processing ...
    
    logger.info(
        'Prediction completed',
        extra={
            'prediction_id': 'pred-12345',
            'model_version': 'v1.2.0',
            'latency_ms': (time.time() - start_time) * 1000,
            'input_features': {'age': 35, 'income': 50000}
        }
    )
    
    print("\n=== Structured Log Example ===")
    print("Logs are saved to model_predictions.log in JSON format")
    

**Output JSON Log** :
    
    
    {
      "timestamp": "2025-10-23T12:34:56.789012",
      "level": "INFO",
      "logger": "ml_model",
      "message": "Prediction completed",
      "module": "main",
      "function": "predict",
      "line": 42,
      "prediction_id": "pred-12345",
      "model_version": "v1.2.0",
      "latency_ms": 123.45,
      "input_features": {"age": 35, "income": 50000}
    }
    

### Prometheus + Grafana Setup

**Prometheus** is a time-series database, and **Grafana** is a visualization tool.

#### Implementing Prometheus Metrics
    
    
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import time
    import random
    
    # Define metrics
    prediction_counter = Counter(
        'model_predictions_total',
        'Total number of predictions',
        ['model_version', 'status']
    )
    
    prediction_latency = Histogram(
        'model_prediction_latency_seconds',
        'Prediction latency in seconds',
        ['model_version'],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    
    active_predictions = Gauge(
        'model_active_predictions',
        'Number of predictions currently being processed'
    )
    
    model_accuracy = Gauge(
        'model_accuracy',
        'Current model accuracy',
        ['model_version']
    )
    
    
    class ModelMonitor:
        """Model prediction monitoring class"""
    
        def __init__(self, model_version: str = 'v1.0.0'):
            self.model_version = model_version
    
        def predict_with_monitoring(self, features: dict) -> dict:
            """Prediction with monitoring"""
            active_predictions.inc()  # Increase active prediction count
    
            try:
                # Measure prediction time
                with prediction_latency.labels(model_version=self.model_version).time():
                    # Actual prediction processing (dummy)
                    time.sleep(random.uniform(0.01, 0.2))
                    prediction = random.choice([0, 1])
                    confidence = random.uniform(0.6, 0.99)
    
                # Success count
                prediction_counter.labels(
                    model_version=self.model_version,
                    status='success'
                ).inc()
    
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_version': self.model_version
                }
    
            except Exception as e:
                # Error count
                prediction_counter.labels(
                    model_version=self.model_version,
                    status='error'
                ).inc()
                raise
    
            finally:
                active_predictions.dec()  # Decrease active prediction count
    
        def update_accuracy(self, accuracy: float):
            """Update model accuracy"""
            model_accuracy.labels(model_version=self.model_version).set(accuracy)
    
    
    # Start metrics server
    print("Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    
    # Usage example
    monitor = ModelMonitor(model_version='v1.2.0')
    
    # Execute multiple predictions
    print("\n=== Executing Predictions and Collecting Metrics ===")
    for i in range(10):
        result = monitor.predict_with_monitoring({'feature1': i})
        print(f"Prediction {i+1}: {result['prediction']} (confidence: {result['confidence']:.2f})")
        time.sleep(0.1)
    
    # Update accuracy
    monitor.update_accuracy(0.92)
    print(f"\nModel accuracy updated: 92%")
    print(f"\nMetrics can be viewed at http://localhost:8000/metrics")
    

#### Prometheus Configuration File (prometheus.yml)
    
    
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'ml_model'
        static_configs:
          - targets: ['localhost:8000']
            labels:
              environment: 'production'
              service: 'recommendation_model'
    

### Custom Metrics
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Custom Metrics
    
    Purpose: Demonstrate simulation and statistical methods
    Target: Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from prometheus_client import Summary, Info
    import numpy as np
    from typing import List
    
    # More detailed metrics
    prediction_score_summary = Summary(
        'model_prediction_score',
        'Distribution of prediction scores',
        ['model_version']
    )
    
    feature_summary = Summary(
        'model_input_feature_value',
        'Distribution of input feature values',
        ['feature_name']
    )
    
    model_info = Info(
        'model_metadata',
        'Model metadata information'
    )
    
    
    class AdvancedModelMonitor:
        """Advanced monitoring features"""
    
        def __init__(self, model_version: str):
            self.model_version = model_version
    
            # Set model metadata
            model_info.info({
                'version': model_version,
                'framework': 'scikit-learn',
                'algorithm': 'RandomForest',
                'trained_date': '2025-10-20'
            })
    
        def track_prediction(self, features: dict, prediction_score: float):
            """Track prediction scores and features"""
            # Prediction score distribution
            prediction_score_summary.labels(
                model_version=self.model_version
            ).observe(prediction_score)
    
            # Distribution of each feature
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    feature_summary.labels(
                        feature_name=feature_name
                    ).observe(value)
    
        def track_batch_predictions(self, batch_features: List[dict],
                                   batch_scores: List[float]):
            """Track batch predictions"""
            for features, score in zip(batch_features, batch_scores):
                self.track_prediction(features, score)
    
    
    # Usage example
    advanced_monitor = AdvancedModelMonitor(model_version='v1.2.0')
    
    # Batch prediction simulation
    print("\n=== Batch Prediction Monitoring ===")
    batch_size = 100
    batch_features = [
        {
            'age': np.random.randint(18, 80),
            'income': np.random.randint(20000, 150000),
            'credit_score': np.random.randint(300, 850)
        }
        for _ in range(batch_size)
    ]
    batch_scores = np.random.beta(5, 2, batch_size).tolist()
    
    advanced_monitor.track_batch_predictions(batch_features, batch_scores)
    print(f"Tracked {batch_size} predictions")
    

### Alert Configuration

#### Prometheus Alert Rules (alerts.yml)
    
    
    groups:
      - name: ml_model_alerts
        interval: 30s
        rules:
          # Latency alert
          - alert: HighPredictionLatency
            expr: histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m])) > 1.0
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High prediction latency detected"
              description: "95th percentile latency is {{ $value }}s (threshold: 1.0s)"
    
          # Error rate alert
          - alert: HighErrorRate
            expr: rate(model_predictions_total{status="error"}[5m]) / rate(model_predictions_total[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
    
          # Accuracy drop alert
          - alert: ModelAccuracyDrop
            expr: model_accuracy < 0.85
            for: 10m
            labels:
              severity: critical
            annotations:
              summary: "Model accuracy dropped below threshold"
              description: "Current accuracy is {{ $value }} (threshold: 0.85)"
    
          # Traffic spike alert
          - alert: TrafficSpike
            expr: rate(model_predictions_total[5m]) > 1000
            for: 2m
            labels:
              severity: warning
            annotations:
              summary: "Unexpected traffic spike"
              description: "Request rate is {{ $value }} req/s (threshold: 1000 req/s)"
    

* * *

## 4.2 Model Performance Tracking

### Data Drift Detection

**Data drift** is a phenomenon where the distribution of input data changes over time.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp, chi2_contingency
    from typing import Tuple, Dict
    import warnings
    
    class DataDriftDetector:
        """Data drift detector"""
    
        def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
            """
            Args:
                reference_data: Reference data (training data)
                threshold: Statistical significance level (default: 0.05)
            """
            self.reference_data = reference_data
            self.threshold = threshold
    
        def detect_drift_numerical(self, current_data: pd.DataFrame,
                                   feature: str) -> Tuple[bool, float]:
            """
            Drift detection for numerical features (Kolmogorov-Smirnov test)
    
            Returns:
                (drift detected, p-value)
            """
            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
    
            # KS test
            statistic, p_value = ks_2samp(ref_values, cur_values)
    
            # If p-value is smaller than threshold, drift is detected
            drift_detected = p_value < self.threshold
    
            return drift_detected, p_value
    
        def detect_drift_categorical(self, current_data: pd.DataFrame,
                                     feature: str) -> Tuple[bool, float]:
            """
            Drift detection for categorical features (Chi-squared test)
    
            Returns:
                (drift detected, p-value)
            """
            ref_counts = self.reference_data[feature].value_counts()
            cur_counts = current_data[feature].value_counts()
    
            # Get common categories
            all_categories = set(ref_counts.index) | set(cur_counts.index)
    
            # Create frequency table
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]
    
            # Chi-squared test
            contingency_table = [ref_freq, cur_freq]
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
            drift_detected = p_value < self.threshold
    
            return drift_detected, p_value
    
        def detect_all_features(self, current_data: pd.DataFrame) -> Dict[str, dict]:
            """Drift detection for all features"""
            results = {}
    
            for feature in self.reference_data.columns:
                try:
                    # Select test method based on data type
                    if pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                        drift, p_value = self.detect_drift_numerical(current_data, feature)
                        method = 'KS-test'
                    else:
                        drift, p_value = self.detect_drift_categorical(current_data, feature)
                        method = 'Chi-squared'
    
                    results[feature] = {
                        'drift_detected': drift,
                        'p_value': p_value,
                        'method': method
                    }
                except Exception as e:
                    results[feature] = {
                        'drift_detected': None,
                        'p_value': None,
                        'error': str(e)
                    }
    
            return results
    
    
    # Demo with sample data
    np.random.seed(42)
    
    # Reference data (training data)
    reference_data = pd.DataFrame({
        'age': np.random.normal(45, 12, 1000),
        'income': np.random.normal(60000, 15000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
    })
    
    # Current data (with drift)
    current_data = pd.DataFrame({
        'age': np.random.normal(40, 12, 1000),  # Mean changed
        'income': np.random.normal(60000, 15000, 1000),  # No change
        'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.3, 0.4, 0.3])  # Distribution changed
    })
    
    # Drift detection
    detector = DataDriftDetector(reference_data, threshold=0.05)
    drift_results = detector.detect_all_features(current_data)
    
    print("=== Data Drift Detection Results ===\n")
    for feature, result in drift_results.items():
        if 'error' not in result:
            status = "Drift Detected" if result['drift_detected'] else "Normal"
            print(f"{feature}:")
            print(f"  Status: {status}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Test method: {result['method']}\n")
    

### Utilizing Evidently AI

**Evidently AI** is a specialized library for ML model monitoring and drift detection.
    
    
    # Drift detection using Evidently AI
    # pip install evidently
    
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    reference = pd.DataFrame({
        'age': np.random.normal(45, 12, n_samples),
        'income': np.random.normal(60000, 15000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Current data with drift
    current = pd.DataFrame({
        'age': np.random.normal(42, 12, n_samples),  # Drift
        'income': np.random.normal(65000, 18000, n_samples),  # Drift
        'credit_score': np.random.normal(700, 50, n_samples),  # Normal
        'target': np.random.binomial(1, 0.35, n_samples)  # Drift
    })
    
    # Generate data drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    data_drift_report.run(reference_data=reference, current_data=current)
    
    # Save HTML report
    data_drift_report.save_html('data_drift_report.html')
    
    print("=== Evidently AI Data Drift Report ===")
    print("Report saved to data_drift_report.html")
    print("\nKey detection results:")
    print(f"- Age: Drift likely detected")
    print(f"- Income: Drift likely detected")
    print(f"- Credit Score: Normal")
    
    # Data drift test
    data_drift_test = TestSuite(tests=[
        DataDriftTestPreset(),
    ])
    
    data_drift_test.run(reference_data=reference, current_data=current)
    
    # Get test results in JSON format
    test_results = data_drift_test.as_dict()
    
    print(f"\nTests passed: {test_results['summary']['success']}")
    print(f"Total tests: {test_results['summary']['total']}")
    print(f"Failed tests: {test_results['summary']['failed']}")
    

### Model Drift Detection

**Model drift** is a phenomenon where the prediction performance of a model degrades over time.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    from collections import deque
    from typing import List, Optional
    import datetime
    
    class ModelPerformanceMonitor:
        """Real-time model performance monitoring"""
    
        def __init__(self, window_size: int = 1000, alert_threshold: float = 0.05):
            """
            Args:
                window_size: Monitoring window size
                alert_threshold: Threshold for alert trigger (performance degradation rate)
            """
            self.window_size = window_size
            self.alert_threshold = alert_threshold
    
            # Historical data
            self.predictions = deque(maxlen=window_size)
            self.actuals = deque(maxlen=window_size)
            self.timestamps = deque(maxlen=window_size)
    
            # Baseline performance (at training time)
            self.baseline_accuracy: Optional[float] = None
            self.baseline_auc: Optional[float] = None
    
        def set_baseline(self, accuracy: float, auc: float):
            """Set baseline performance"""
            self.baseline_accuracy = accuracy
            self.baseline_auc = auc
    
        def add_prediction(self, prediction: int, actual: int,
                           timestamp: Optional[datetime.datetime] = None):
            """Add prediction and actual result"""
            self.predictions.append(prediction)
            self.actuals.append(actual)
            self.timestamps.append(timestamp or datetime.datetime.now())
    
        def get_current_metrics(self) -> dict:
            """Calculate metrics in the current window"""
            if len(self.predictions) < 100:  # Minimum sample size
                return {'status': 'insufficient_data'}
    
            current_accuracy = accuracy_score(
                list(self.actuals),
                list(self.predictions)
            )
    
            try:
                current_auc = roc_auc_score(
                    list(self.actuals),
                    list(self.predictions)
                )
            except:
                current_auc = None
    
            return {
                'accuracy': current_accuracy,
                'auc': current_auc,
                'sample_count': len(self.predictions)
            }
    
        def detect_performance_drift(self) -> dict:
            """Detect performance drift"""
            current_metrics = self.get_current_metrics()
    
            if current_metrics.get('status') == 'insufficient_data':
                return {'drift_detected': False, 'reason': 'insufficient_data'}
    
            if self.baseline_accuracy is None:
                return {'drift_detected': False, 'reason': 'no_baseline'}
    
            # Calculate accuracy degradation rate
            accuracy_drop = self.baseline_accuracy - current_metrics['accuracy']
            accuracy_drop_pct = accuracy_drop / self.baseline_accuracy
    
            drift_detected = accuracy_drop_pct > self.alert_threshold
    
            result = {
                'drift_detected': drift_detected,
                'baseline_accuracy': self.baseline_accuracy,
                'current_accuracy': current_metrics['accuracy'],
                'accuracy_drop': accuracy_drop,
                'accuracy_drop_percentage': accuracy_drop_pct * 100,
                'threshold_percentage': self.alert_threshold * 100
            }
    
            if current_metrics['auc'] and self.baseline_auc:
                auc_drop = self.baseline_auc - current_metrics['auc']
                result['baseline_auc'] = self.baseline_auc
                result['current_auc'] = current_metrics['auc']
                result['auc_drop'] = auc_drop
    
            return result
    
    
    # Usage example
    monitor = ModelPerformanceMonitor(window_size=1000, alert_threshold=0.05)
    
    # Set baseline performance (training time performance)
    monitor.set_baseline(accuracy=0.92, auc=0.95)
    
    print("=== Model Performance Monitoring ===\n")
    
    # Initial normal performance
    print("Phase 1: Normal Operation (1000 samples)")
    for i in range(1000):
        # Simulate predictions with ~92% accuracy
        actual = np.random.binomial(1, 0.3)
        prediction = actual if np.random.random() < 0.92 else 1 - actual
        monitor.add_prediction(prediction, actual)
    
    result1 = monitor.detect_performance_drift()
    print(f"  Drift detected: {result1['drift_detected']}")
    print(f"  Current accuracy: {result1['current_accuracy']:.3f}")
    print(f"  Baseline accuracy: {result1['baseline_accuracy']:.3f}")
    
    # Simulate performance degradation
    print("\nPhase 2: Performance Degradation (1000 samples)")
    for i in range(1000):
        # Accuracy drops to 85%
        actual = np.random.binomial(1, 0.3)
        prediction = actual if np.random.random() < 0.85 else 1 - actual
        monitor.add_prediction(prediction, actual)
    
    result2 = monitor.detect_performance_drift()
    print(f"  Drift detected: {result2['drift_detected']}")
    print(f"  Current accuracy: {result2['current_accuracy']:.3f}")
    print(f"  Accuracy drop: {result2['accuracy_drop']:.3f} ({result2['accuracy_drop_percentage']:.1f}%)")
    
    if result2['drift_detected']:
        print(f"\n  ⚠️ Alert: Accuracy has dropped by more than {result2['threshold_percentage']:.0f}%!")
        print(f"  Recommended action: Consider retraining the model")
    

* * *

## 4.3 A/B Testing and Canary Deployment

### Traffic Splitting
    
    
    import random
    from typing import Callable, Dict, Any
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class ModelVariant:
        """Model variant"""
        name: str
        version: str
        traffic_percentage: float
        predict_fn: Callable
    
    
    class ABTestRouter:
        """Traffic router for A/B testing"""
    
        def __init__(self):
            self.variants: Dict[str, ModelVariant] = {}
            self.prediction_log = []
    
        def add_variant(self, variant: ModelVariant):
            """Add a variant"""
            self.variants[variant.name] = variant
    
        def validate_traffic_split(self) -> bool:
            """Verify that traffic split equals 100%"""
            total = sum(v.traffic_percentage for v in self.variants.values())
            return abs(total - 100.0) < 0.01
    
        def route_prediction(self, request_id: str, features: dict) -> dict:
            """
            Split traffic and route predictions
    
            Args:
                request_id: Request ID (user ID, etc.)
                features: Input features
    
            Returns:
                Prediction result and metadata
            """
            if not self.validate_traffic_split():
                raise ValueError("Traffic split does not sum to 100%")
    
            # Hash-based consistent routing
            # Same user is always routed to the same variant
            hash_value = hash(request_id) % 100
    
            cumulative_percentage = 0
            selected_variant = None
    
            for variant in self.variants.values():
                cumulative_percentage += variant.traffic_percentage
                if hash_value < cumulative_percentage:
                    selected_variant = variant
                    break
    
            # Execute prediction
            prediction = selected_variant.predict_fn(features)
    
            # Log record
            log_entry = {
                'request_id': request_id,
                'variant': selected_variant.name,
                'version': selected_variant.version,
                'prediction': prediction,
                'timestamp': datetime.now()
            }
            self.prediction_log.append(log_entry)
    
            return {
                'prediction': prediction,
                'model_variant': selected_variant.name,
                'model_version': selected_variant.version
            }
    
    
    # Dummy model implementation
    def model_v1_predict(features: dict) -> int:
        """Model v1.0 prediction"""
        # Dummy: random prediction (~80% accuracy)
        return random.choices([0, 1], weights=[0.6, 0.4])[0]
    
    def model_v2_predict(features: dict) -> int:
        """Model v2.0 prediction (improved version)"""
        # Dummy: random prediction (~85% accuracy)
        return random.choices([0, 1], weights=[0.55, 0.45])[0]
    
    
    # A/B test setup
    router = ABTestRouter()
    
    # Control group (existing model): 90%
    router.add_variant(ModelVariant(
        name='control',
        version='v1.0',
        traffic_percentage=90.0,
        predict_fn=model_v1_predict
    ))
    
    # Test group (new model): 10%
    router.add_variant(ModelVariant(
        name='treatment',
        version='v2.0',
        traffic_percentage=10.0,
        predict_fn=model_v2_predict
    ))
    
    print("=== A/B Test Execution ===\n")
    print("Traffic split:")
    print("  Control (v1.0): 90%")
    print("  Test (v2.0): 10%\n")
    
    # Simulate predictions
    n_requests = 1000
    for i in range(n_requests):
        request_id = f"user_{i % 100}"  # Simulate 100 users
        features = {'feature1': random.random()}
        result = router.route_prediction(request_id, features)
    
    # Aggregate results
    variant_counts = {}
    for log in router.prediction_log:
        variant = log['variant']
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
    
    print(f"Total requests: {n_requests}")
    print("\nActual traffic split:")
    for variant, count in variant_counts.items():
        percentage = (count / n_requests) * 100
        print(f"  {variant}: {count} requests ({percentage:.1f}%)")
    

### Statistical Testing
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    
    import numpy as np
    from scipy import stats
    from typing import List, Tuple
    
    class ABTestAnalyzer:
        """Statistical analysis of A/B test results"""
    
        @staticmethod
        def proportion_test(control_successes: int, control_total: int,
                            treatment_successes: int, treatment_total: int,
                            alpha: float = 0.05) -> dict:
            """
            Test of difference in proportions (binomial distribution)
    
            Args:
                control_successes: Number of successes in control group
                control_total: Total size of control group
                treatment_successes: Number of successes in test group
                treatment_total: Total size of test group
                alpha: Significance level
    
            Returns:
                Test results
            """
            # Calculate proportions
            p_control = control_successes / control_total
            p_treatment = treatment_successes / treatment_total
    
            # Pooled proportion
            p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
    
            # Standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    
            # z-statistic
            z_stat = (p_treatment - p_control) / se
    
            # p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
            # Confidence interval
            ci_margin = stats.norm.ppf(1 - alpha/2) * se
            ci_lower = (p_treatment - p_control) - ci_margin
            ci_upper = (p_treatment - p_control) + ci_margin
    
            # Statistical significance
            is_significant = p_value < alpha
    
            # Effect size (relative improvement rate)
            relative_improvement = (p_treatment - p_control) / p_control * 100 if p_control > 0 else 0
    
            return {
                'control_rate': p_control,
                'treatment_rate': p_treatment,
                'absolute_difference': p_treatment - p_control,
                'relative_improvement_pct': relative_improvement,
                'z_statistic': z_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'confidence_interval': (ci_lower, ci_upper),
                'confidence_level': (1 - alpha) * 100
            }
    
        @staticmethod
        def sample_size_calculator(baseline_rate: float,
                                   minimum_detectable_effect: float,
                                   alpha: float = 0.05,
                                   power: float = 0.8) -> int:
            """
            Calculate required sample size
    
            Args:
                baseline_rate: Baseline conversion rate
                minimum_detectable_effect: Minimum detectable effect (relative change rate)
                alpha: Significance level
                power: Statistical power
    
            Returns:
                Required sample size per group
            """
            # Z values
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
    
            # Expected conversion rate for treatment group
            treatment_rate = baseline_rate * (1 + minimum_detectable_effect)
    
            # Sample size calculation
            p_avg = (baseline_rate + treatment_rate) / 2
    
            n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (baseline_rate - treatment_rate)**2
    
            return int(np.ceil(n))
    
    
    # Usage example
    print("=== A/B Test Statistical Analysis ===\n")
    
    # Simulation data
    control_total = 10000
    treatment_total = 1000
    
    # Control: 20% conversion rate
    control_successes = 2000
    
    # Test: 22% conversion rate (10% improvement)
    treatment_successes = 220
    
    # Run statistical test
    analyzer = ABTestAnalyzer()
    results = analyzer.proportion_test(
        control_successes, control_total,
        treatment_successes, treatment_total
    )
    
    print("Test results:")
    print(f"  Control conversion rate: {results['control_rate']:.3f}")
    print(f"  Test conversion rate: {results['treatment_rate']:.3f}")
    print(f"  Absolute difference: {results['absolute_difference']:.3f}")
    print(f"  Relative improvement: {results['relative_improvement_pct']:.1f}%")
    print(f"  p-value: {results['p_value']:.4f}")
    print(f"  Statistically significant: {'Yes' if results['is_significant'] else 'No'}")
    print(f"  95% confidence interval: [{results['confidence_interval'][0]:.3f}, {results['confidence_interval'][1]:.3f}]")
    
    # Calculate required sample size
    print("\n\n=== Required Sample Size Calculation ===\n")
    sample_size = analyzer.sample_size_calculator(
        baseline_rate=0.20,
        minimum_detectable_effect=0.10,  # Detect 10% improvement
        alpha=0.05,
        power=0.8
    )
    
    print(f"Baseline conversion rate: 20%")
    print(f"Improvement to detect: 10% (20% → 22%)")
    print(f"Significance level (α): 5%")
    print(f"Statistical power (1-β): 80%")
    print(f"\nRequired sample size (per group): {sample_size:,} samples")
    

### Canary Deployment Strategy
    
    
    from datetime import datetime, timedelta
    from typing import Optional
    import time
    
    class CanaryDeployment:
        """Canary deployment management"""
    
        def __init__(self,
                     initial_traffic: float = 5.0,
                     max_traffic: float = 100.0,
                     increment_step: float = 10.0,
                     monitoring_window_minutes: int = 30):
            """
            Args:
                initial_traffic: Initial traffic percentage (%)
                max_traffic: Maximum traffic percentage (%)
                increment_step: Traffic increase step (%)
                monitoring_window_minutes: Monitoring time for each step (minutes)
            """
            self.current_traffic = 0.0
            self.initial_traffic = initial_traffic
            self.max_traffic = max_traffic
            self.increment_step = increment_step
            self.monitoring_window = timedelta(minutes=monitoring_window_minutes)
    
            self.deployment_start_time: Optional[datetime] = None
            self.current_stage_start_time: Optional[datetime] = None
            self.stages_completed = []
    
        def start_deployment(self):
            """Start canary deployment"""
            self.deployment_start_time = datetime.now()
            self.current_traffic = self.initial_traffic
            self.current_stage_start_time = datetime.now()
    
            print(f"Canary deployment started: {self.current_traffic}% of traffic to new version")
    
        def should_proceed_to_next_stage(self, health_metrics: dict) -> Tuple[bool, str]:
            """
            Determine whether to proceed to next stage
    
            Args:
                health_metrics: Health metrics
                    - error_rate: Error rate
                    - latency_p95: 95th percentile latency
                    - success_rate: Success rate
    
            Returns:
                (can proceed, reason)
            """
            # Check if monitoring time has elapsed
            elapsed = datetime.now() - self.current_stage_start_time
            if elapsed < self.monitoring_window:
                return False, f"Monitoring time not reached ({elapsed.total_seconds()/60:.1f}/{self.monitoring_window.total_seconds()/60:.0f}min)"
    
            # Health check
            if health_metrics.get('error_rate', 0) > 0.05:
                return False, f"Error rate too high ({health_metrics['error_rate']*100:.1f}%)"
    
            if health_metrics.get('latency_p95', 0) > 2000:
                return False, f"Latency too high ({health_metrics['latency_p95']}ms)"
    
            if health_metrics.get('success_rate', 1.0) < 0.95:
                return False, f"Success rate too low ({health_metrics['success_rate']*100:.1f}%)"
    
            return True, "Health check passed"
    
        def proceed_to_next_stage(self) -> bool:
            """Proceed to next stage"""
            if self.current_traffic >= self.max_traffic:
                print("Canary deployment complete: 100% traffic migrated")
                return False
    
            # Record current stage
            self.stages_completed.append({
                'traffic': self.current_traffic,
                'start_time': self.current_stage_start_time,
                'end_time': datetime.now()
            })
    
            # Increase traffic
            self.current_traffic = min(
                self.current_traffic + self.increment_step,
                self.max_traffic
            )
            self.current_stage_start_time = datetime.now()
    
            print(f"\nProceeding to next stage: {self.current_traffic}% of traffic to new version")
            return True
    
        def rollback(self, reason: str):
            """Execute rollback"""
            print(f"\n🚨 Rollback executed: {reason}")
            print(f"Reverting traffic to old version")
            self.current_traffic = 0.0
    
    
    # Usage example
    print("=== Canary Deployment Execution ===\n")
    
    canary = CanaryDeployment(
        initial_traffic=5.0,
        max_traffic=100.0,
        increment_step=15.0,
        monitoring_window_minutes=1  # 1 minute for demo
    )
    
    canary.start_deployment()
    
    # Simulate deployment process
    stages = []
    while canary.current_traffic < canary.max_traffic:
        print(f"\nCurrent stage: {canary.current_traffic}% traffic")
        print(f"Monitoring... ({canary.monitoring_window.total_seconds()/60:.0f} minutes)")
    
        # Simulate monitoring time (in production: time.sleep(monitoring_window))
        time.sleep(1)
    
        # Simulate health metrics
        health_metrics = {
            'error_rate': np.random.uniform(0, 0.03),  # 0-3% error rate
            'latency_p95': np.random.uniform(100, 500),  # 100-500ms
            'success_rate': np.random.uniform(0.97, 1.0)  # 97-100% success rate
        }
    
        print(f"  Error rate: {health_metrics['error_rate']*100:.2f}%")
        print(f"  Latency P95: {health_metrics['latency_p95']:.0f}ms")
        print(f"  Success rate: {health_metrics['success_rate']*100:.2f}%")
    
        # Determine if we should proceed to next stage
        can_proceed, reason = canary.should_proceed_to_next_stage(health_metrics)
    
        if can_proceed:
            print(f"✓ {reason}")
            if not canary.proceed_to_next_stage():
                break
        else:
            print(f"⏸ {reason}")
            # In production, continue monitoring
    
    print("\n\n=== Deployment Complete ===")
    print(f"Total elapsed time: {len(canary.stages_completed)} stages")
    print(f"Final traffic: {canary.current_traffic}%")
    

* * *

## 4.4 Model Update and Retraining

### Online Learning vs Batch Retraining

Characteristic | Online Learning | Batch Retraining  
---|---|---  
**Update Frequency** | Real-time to minutes | Daily to weekly  
**Computational Cost** | Low (incremental updates) | High (full data retraining)  
**Adaptation Speed** | Fast | Slow  
**Stability** | Low (sensitive to noise) | High  
**Implementation Difficulty** | High | Medium  
**Use Cases** | Recommendation systems, advertising | Credit scoring, fraud detection  
  
### Model Versioning
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    import joblib
    import json
    from pathlib import Path
    from datetime import datetime
    from typing import Any, Dict, Optional
    import hashlib
    
    class ModelRegistry:
        """Model version management registry"""
    
        def __init__(self, registry_path: str = "./model_registry"):
            self.registry_path = Path(registry_path)
            self.registry_path.mkdir(parents=True, exist_ok=True)
            self.metadata_file = self.registry_path / "registry.json"
    
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {'models': {}}
    
        def register_model(self,
                           model: Any,
                           model_name: str,
                           version: str,
                           metrics: Dict[str, float],
                           description: str = "",
                           tags: Dict[str, str] = None) -> str:
            """
            Register a model
    
            Returns:
                Model ID
            """
            # Generate model ID
            model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
            # Model file path
            model_path = self.registry_path / f"{model_id}.pkl"
    
            # Save model
            joblib.dump(model, model_path)
    
            # Calculate model hash
            model_hash = self._calculate_file_hash(model_path)
    
            # Save metadata
            model_metadata = {
                'model_id': model_id,
                'model_name': model_name,
                'version': version,
                'file_path': str(model_path),
                'file_hash': model_hash,
                'metrics': metrics,
                'description': description,
                'tags': tags or {},
                'registered_at': datetime.now().isoformat(),
                'status': 'registered'
            }
    
            self.metadata['models'][model_id] = model_metadata
            self._save_metadata()
    
            return model_id
    
        def load_model(self, model_id: str) -> Any:
            """Load a model"""
            if model_id not in self.metadata['models']:
                raise ValueError(f"Model {model_id} not found in registry")
    
            model_info = self.metadata['models'][model_id]
            model_path = Path(model_info['file_path'])
    
            # Hash verification
            current_hash = self._calculate_file_hash(model_path)
            if current_hash != model_info['file_hash']:
                raise ValueError(f"Model file hash mismatch for {model_id}")
    
            return joblib.load(model_path)
    
        def promote_to_production(self, model_id: str):
            """Promote model to production environment"""
            if model_id not in self.metadata['models']:
                raise ValueError(f"Model {model_id} not found")
    
            # Demote existing production model to staging
            for mid, info in self.metadata['models'].items():
                if info['status'] == 'production':
                    info['status'] = 'staging'
                    info['demoted_at'] = datetime.now().isoformat()
    
            # Promote new model to production
            self.metadata['models'][model_id]['status'] = 'production'
            self.metadata['models'][model_id]['promoted_at'] = datetime.now().isoformat()
    
            self._save_metadata()
    
        def get_production_model(self) -> Optional[str]:
            """Get current production model ID"""
            for model_id, info in self.metadata['models'].items():
                if info['status'] == 'production':
                    return model_id
            return None
    
        def list_models(self, status: Optional[str] = None) -> list:
            """Get list of models"""
            models = list(self.metadata['models'].values())
            if status:
                models = [m for m in models if m['status'] == status]
            return sorted(models, key=lambda x: x['registered_at'], reverse=True)
    
        def _calculate_file_hash(self, file_path: Path) -> str:
            """Calculate file hash"""
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
    
        def _save_metadata(self):
            """Save metadata to file"""
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    
    # Usage example
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("=== Model Versioning ===\n")
    
    # Initialize model registry
    registry = ModelRegistry("./model_registry")
    
    # Train dummy model
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model_v1 = RandomForestClassifier(n_estimators=50, random_state=42)
    model_v1.fit(X, y)
    
    # Register model
    model_id_v1 = registry.register_model(
        model=model_v1,
        model_name="fraud_detector",
        version="1.0.0",
        metrics={
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1': 0.85
        },
        description="Initial production model",
        tags={'framework': 'sklearn', 'algorithm': 'RandomForest'}
    )
    
    print(f"Model registered: {model_id_v1}")
    
    # Promote to production
    registry.promote_to_production(model_id_v1)
    print(f"Promoted to production: {model_id_v1}")
    
    # Train and register new model
    model_v2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_v2.fit(X, y)
    
    model_id_v2 = registry.register_model(
        model=model_v2,
        model_name="fraud_detector",
        version="2.0.0",
        metrics={
            'accuracy': 0.89,
            'precision': 0.88,
            'recall': 0.90,
            'f1': 0.89
        },
        description="Improved model with more estimators",
        tags={'framework': 'sklearn', 'algorithm': 'RandomForest'}
    )
    
    print(f"\nNew model registered: {model_id_v2}")
    
    # Display model list
    print("\n=== Registered Models ===")
    for model_info in registry.list_models():
        print(f"\nModel: {model_info['model_name']} v{model_info['version']}")
        print(f"  ID: {model_info['model_id']}")
        print(f"  Status: {model_info['status']}")
        print(f"  Accuracy: {model_info['metrics']['accuracy']:.3f}")
        print(f"  Registered: {model_info['registered_at']}")
    
    # Get production model
    prod_model_id = registry.get_production_model()
    print(f"\nCurrent production model: {prod_model_id}")
    

### Automated Retraining Pipeline
    
    
    from datetime import datetime, timedelta
    from typing import Optional
    import schedule
    import time
    
    class AutoRetrainingPipeline:
        """Automated retraining pipeline"""
    
        def __init__(self,
                     model_registry: ModelRegistry,
                     performance_monitor: ModelPerformanceMonitor,
                     retrain_threshold: float = 0.05):
            """
            Args:
                model_registry: Model registry
                performance_monitor: Performance monitor
                retrain_threshold: Threshold for retraining trigger (performance degradation rate)
            """
            self.registry = model_registry
            self.monitor = performance_monitor
            self.retrain_threshold = retrain_threshold
            self.last_retrain_time: Optional[datetime] = None
    
        def check_retrain_needed(self) -> bool:
            """Determine if retraining is needed"""
            drift_result = self.monitor.detect_performance_drift()
    
            if drift_result.get('drift_detected'):
                print(f"⚠️ Performance drift detected: {drift_result['accuracy_drop_percentage']:.1f}% drop")
                return True
    
            return False
    
        def retrain_model(self, training_data_path: str) -> str:
            """
            Retrain model
    
            Returns:
                New model ID
            """
            print(f"\n{'='*50}")
            print(f"Automated retraining started: {datetime.now().isoformat()}")
            print(f"{'='*50}\n")
    
            # Load training data (dummy)
            print("1. Loading training data...")
            X, y = make_classification(n_samples=2000, n_features=10, random_state=int(time.time()))
    
            # Train model
            print("2. Training model...")
            new_model = RandomForestClassifier(n_estimators=100, random_state=42)
            new_model.fit(X, y)
    
            # Evaluation
            print("3. Evaluating model...")
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(new_model, X, y, cv=5)
            accuracy = scores.mean()
    
            # Generate version number
            current_prod_id = self.registry.get_production_model()
            if current_prod_id:
                current_version = self.registry.metadata['models'][current_prod_id]['version']
                major, minor, patch = map(int, current_version.split('.'))
                new_version = f"{major}.{minor}.{patch + 1}"
            else:
                new_version = "1.0.0"
    
            # Register to registry
            print("4. Registering model...")
            new_model_id = self.registry.register_model(
                model=new_model,
                model_name="fraud_detector",
                version=new_version,
                metrics={
                    'accuracy': accuracy,
                    'cv_scores_mean': accuracy,
                    'cv_scores_std': scores.std()
                },
                description=f"Auto-retrained model (trigger: performance drift)",
                tags={'retrain_type': 'automatic', 'trigger': 'performance_drift'}
            )
    
            print(f"✓ New model registration complete: {new_model_id}")
            print(f"  Version: {new_version}")
            print(f"  Accuracy: {accuracy:.3f}")
    
            # Promote if performance has improved
            if current_prod_id:
                current_accuracy = self.registry.metadata['models'][current_prod_id]['metrics']['accuracy']
                if accuracy > current_accuracy:
                    print(f"\n5. Promoting to production (accuracy improved: {current_accuracy:.3f} → {accuracy:.3f})")
                    self.registry.promote_to_production(new_model_id)
                else:
                    print(f"\n5. Promotion skipped (no accuracy improvement: {accuracy:.3f} vs {current_accuracy:.3f})")
            else:
                print(f"\n5. Promoting to production (first model)")
                self.registry.promote_to_production(new_model_id)
    
            self.last_retrain_time = datetime.now()
    
            print(f"\n{'='*50}")
            print(f"Retraining complete")
            print(f"{'='*50}\n")
    
            return new_model_id
    
        def scheduled_check(self, training_data_path: str):
            """Scheduled check"""
            print(f"\n[{datetime.now().isoformat()}] Running scheduled check")
    
            if self.check_retrain_needed():
                print("→ Retraining determined necessary")
                self.retrain_model(training_data_path)
            else:
                print("→ Retraining not necessary (performance normal)")
    
    
    # Usage example
    print("=== Automated Retraining Pipeline ===\n")
    
    # Use existing components
    monitor = ModelPerformanceMonitor(window_size=1000, alert_threshold=0.05)
    monitor.set_baseline(accuracy=0.85, auc=0.90)
    
    registry = ModelRegistry("./model_registry")
    
    # Pipeline setup
    pipeline = AutoRetrainingPipeline(
        model_registry=registry,
        performance_monitor=monitor,
        retrain_threshold=0.05
    )
    
    # Simulate performance degradation
    print("Simulating performance degradation...")
    for i in range(1000):
        actual = np.random.binomial(1, 0.3)
        # Accuracy drops to 78%
        prediction = actual if np.random.random() < 0.78 else 1 - actual
        monitor.add_prediction(prediction, actual)
    
    # Retraining check
    pipeline.scheduled_check(training_data_path="data/training.csv")
    
    # Scheduling (production example)
    print("\n\n=== Scheduling Configuration ===")
    print("Automatic checks will run on the following schedule:")
    print("  - Daily at 02:00")
    print("  - Automatic retraining when performance drift detected")
    
    # schedule.every().day.at("02:00").do(
    #     pipeline.scheduled_check,
    #     training_data_path="data/training.csv"
    # )
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
    

* * *

## 4.5 Practice: End-to-End Operations

### Comprehensive Monitoring System
    
    
    from dataclasses import dataclass
    from typing import List, Dict, Any
    import json
    
    @dataclass
    class HealthStatus:
        """System health status"""
        is_healthy: bool
        components: Dict[str, bool]
        alerts: List[str]
        metrics: Dict[str, float]
        timestamp: datetime
    
    
    class ComprehensiveMonitoringSystem:
        """Comprehensive monitoring system"""
    
        def __init__(self):
            self.model_monitor = ModelPerformanceMonitor(window_size=1000)
            self.drift_detector = None  # DataDriftDetector instance
            self.sla_thresholds = {
                'latency_p95_ms': 500,
                'latency_p99_ms': 1000,
                'error_rate': 0.01,
                'availability': 0.999
            }
    
        def check_system_health(self) -> HealthStatus:
            """Check overall system health"""
            components = {}
            alerts = []
            metrics = {}
    
            # 1. Model performance check
            performance_metrics = self.model_monitor.get_current_metrics()
            if performance_metrics.get('status') != 'insufficient_data':
                model_healthy = performance_metrics.get('accuracy', 0) > 0.85
                components['model_performance'] = model_healthy
                metrics['model_accuracy'] = performance_metrics.get('accuracy', 0)
    
                if not model_healthy:
                    alerts.append(f"Model accuracy degraded: {metrics['model_accuracy']:.3f}")
    
            # 2. Latency check
            current_latency_p95 = np.random.uniform(200, 600)  # Dummy
            latency_healthy = current_latency_p95 < self.sla_thresholds['latency_p95_ms']
            components['latency'] = latency_healthy
            metrics['latency_p95_ms'] = current_latency_p95
    
            if not latency_healthy:
                alerts.append(f"Latency SLA violation: {current_latency_p95:.0f}ms")
    
            # 3. Error rate check
            current_error_rate = np.random.uniform(0, 0.02)  # Dummy
            error_rate_healthy = current_error_rate < self.sla_thresholds['error_rate']
            components['error_rate'] = error_rate_healthy
            metrics['error_rate'] = current_error_rate
    
            if not error_rate_healthy:
                alerts.append(f"Error rate SLA violation: {current_error_rate*100:.2f}%")
    
            # 4. Availability check
            current_availability = np.random.uniform(0.995, 1.0)  # Dummy
            availability_healthy = current_availability >= self.sla_thresholds['availability']
            components['availability'] = availability_healthy
            metrics['availability'] = current_availability
    
            if not availability_healthy:
                alerts.append(f"Availability SLA violation: {current_availability*100:.3f}%")
    
            # Overall assessment
            is_healthy = all(components.values())
    
            return HealthStatus(
                is_healthy=is_healthy,
                components=components,
                alerts=alerts,
                metrics=metrics,
                timestamp=datetime.now()
            )
    
        def generate_health_report(self) -> str:
            """Generate health report"""
            status = self.check_system_health()
    
            report = f"""
    {'='*60}
    System Health Report
    {'='*60}
    Timestamp: {status.timestamp.isoformat()}
    Overall Status: {'✓ Normal' if status.is_healthy else '✗ Abnormal'}
    
    Component Status:
    """
            for component, healthy in status.components.items():
                icon = '✓' if healthy else '✗'
                report += f"  {icon} {component}: {'Normal' if healthy else 'Abnormal'}\n"
    
            report += f"\nMetrics:\n"
            for metric, value in status.metrics.items():
                if 'rate' in metric or 'availability' in metric:
                    report += f"  {metric}: {value*100:.2f}%\n"
                else:
                    report += f"  {metric}: {value:.2f}\n"
    
            if status.alerts:
                report += f"\nAlerts ({len(status.alerts)} items):\n"
                for alert in status.alerts:
                    report += f"  ⚠️ {alert}\n"
            else:
                report += f"\nAlerts: None\n"
    
            report += f"{'='*60}\n"
    
            return report
    
    
    # Usage example
    print("=== Comprehensive Monitoring System ===\n")
    
    monitoring = ComprehensiveMonitoringSystem()
    
    # Generate health report
    report = monitoring.generate_health_report()
    print(report)
    

### Production Operations Checklist

Category | Checklist Item | Priority  
---|---|---  
**Pre-Deployment** | Model accuracy exceeds baseline | 🔴 Required  
Statistical significance confirmed in A/B test | 🔴 Required  
Load test meets performance requirements | 🔴 Required  
Rollback procedure documented | 🔴 Required  
**Monitoring** | Structured logs output correctly | 🔴 Required  
Prometheus metrics collected | 🔴 Required  
Grafana dashboard operational | 🟡 Recommended  
Alerts trigger correctly | 🔴 Required  
Data drift detection operational | 🟡 Recommended  
**SLA Definition** | Latency P95 < 500ms | 🔴 Required  
Error rate < 1% | 🔴 Required  
Availability > 99.9% | 🔴 Required  
Model accuracy > 95% of baseline | 🟡 Recommended  
**Incident Response** | On-call system established | 🔴 Required  
Incident response procedures documented | 🔴 Required  
Post-mortem process exists | 🟡 Recommended  
**Data Management** | Prediction logs saved | 🔴 Required  
Actual outcome data collected | 🔴 Required  
Data backup runs regularly | 🟡 Recommended  
  
* * *

## 4.6 Chapter Summary

### What You Learned

  1. **Logging and Monitoring**

     * Machine-readable records with structured JSON logging
     * Time-series metrics monitoring with Prometheus + Grafana
     * Tracking business metrics with custom metrics
     * Early detection of anomalies with alert rules
  2. **Model Performance Tracking**

     * Statistical detection of data drift (KS test, Chi-squared test)
     * Comprehensive drift analysis with Evidently AI
     * Real-time monitoring of model drift
  3. **A/B Testing and Canary Deployment**

     * Safe testing with traffic splitting
     * Effect verification with statistical testing
     * Gradual canary deployment strategy
  4. **Model Update and Retraining**

     * History management with model versioning
     * Building automated retraining pipelines
     * Retraining triggers from performance drift detection
  5. **End-to-End Operations**

     * Comprehensive monitoring system design
     * SLA definition and continuous monitoring
     * Using production operations checklist

### Operations Best Practices

Principle | Description  
---|---  
**Observability First** | Track all important operations with logs and metrics  
**Gradual Deployment** | Limit impact scope with canary deployment  
**Automation First** | Automate retraining, deployment, and alerts  
**Rapid Rollback** | Return to safe state immediately when problems occur  
**Continuous Improvement** | Learn from incidents and improve the system  
  
### Next Steps

You've completed learning about model deployment. To learn more deeply:

  * ML model orchestration on Kubernetes
  * MLOps tools (Kubeflow, MLflow, Vertex AI)
  * Federated learning
  * Deployment to edge devices
  * Real-time inference optimization

* * *

## References

  1. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  2. Ameisen, E. (2020). _Building Machine Learning Powered Applications_. O'Reilly Media.
  3. Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  4. Google. (2023). _Machine Learning Engineering for Production (MLOps)_. Coursera.
  5. Neptune.ai. (2023). _MLOps: Model Monitoring Best Practices_.
