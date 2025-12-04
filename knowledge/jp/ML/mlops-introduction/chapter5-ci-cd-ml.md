---
title: 第5章：CI/CDパイプライン構築
chapter_title: 第5章：CI/CDパイプライン構築
subtitle: 機械学習モデルの自動テストとデプロイメント
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 機械学習におけるCI/CDの特徴と重要性を理解する
  * ✅ MLモデルの自動テスト戦略を設計できる
  * ✅ GitHub Actionsでモデルの学習・検証パイプラインを構築できる
  * ✅ 各種デプロイメント戦略（Blue-Green、Canary等）を実装できる
  * ✅ エンドツーエンドのCI/CDパイプラインを本番環境に適用できる

* * *

## 5.1 CI/CD for MLの特徴

### 従来のCI/CDとの違い

**CI/CD（Continuous Integration/Continuous Delivery）** は、ソフトウェア開発プロセスを自動化する手法です。機械学習では、従来のソフトウェア開発に加えて、データとモデルの管理が必要になります。
    
    
    ```mermaid
    graph TD
        A[コード変更] --> B[CI: 自動テスト]
        B --> C[モデル学習]
        C --> D[モデル検証]
        D --> E{性能基準OK?}
        E -->|Yes| F[CD: デプロイ]
        E -->|No| G[通知・ロールバック]
        F --> H[本番環境]
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

### ML特有の考慮事項

要素 | 従来のCI/CD | ML CI/CD  
---|---|---  
**テスト対象** | コード | コード + データ + モデル  
**品質指標** | テストパス率 | 精度、再現率、F1スコア等  
**再現性** | コードバージョン | コード + データ + ハイパーパラメータ  
**デプロイ戦略** | Blue-Green、Canary | A/Bテスト、Shadow Mode含む  
**モニタリング** | システムメトリクス | モデル性能、データドリフト  
  
### データの変化への対応

> **重要** : MLモデルは時間とともにデータ分布が変化（データドリフト）するため、継続的な再学習とモニタリングが必要です。
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # データドリフトのシミュレーション
    np.random.seed(42)
    
    def generate_data(n_samples, drift_level=0.0):
        """drift_level: 0.0 (変化なし) ~ 1.0 (大きな変化)"""
        X1 = np.random.normal(50 + drift_level * 20, 10, n_samples)
        X2 = np.random.normal(100 + drift_level * 30, 20, n_samples)
        y = ((X1 > 50) & (X2 > 100)).astype(int)
        return pd.DataFrame({'X1': X1, 'X2': X2}), y
    
    # 初期モデルの学習
    X_train, y_train = generate_data(1000, drift_level=0.0)
    X_test, y_test = generate_data(200, drift_level=0.0)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    initial_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # 時間経過によるデータドリフト
    drift_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []
    
    for drift in drift_levels:
        X_test_drift, y_test_drift = generate_data(200, drift_level=drift)
        acc = accuracy_score(y_test_drift, model.predict(X_test_drift))
        accuracies.append(acc)
    
    print("=== データドリフトの影響 ===")
    for drift, acc in zip(drift_levels, accuracies):
        print(f"ドリフトレベル {drift:.1f}: 精度 = {acc:.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(drift_levels, accuracies, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', label='許容精度下限')
    plt.xlabel('データドリフトレベル', fontsize=12)
    plt.ylabel('モデル精度', fontsize=12)
    plt.title('データドリフトによるモデル性能の劣化', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === データドリフトの影響 ===
    ドリフトレベル 0.0: 精度 = 0.920
    ドリフトレベル 0.2: 精度 = 0.885
    ドリフトレベル 0.4: 精度 = 0.835
    ドリフトレベル 0.6: 精度 = 0.775
    ドリフトレベル 0.8: 精度 = 0.720
    ドリフトレベル 1.0: 精度 = 0.670
    

* * *

## 5.2 自動テスト

### MLにおけるテスト戦略

機械学習システムには複数レベルのテストが必要です：

  1. **Unit Tests** : 個別の関数・モジュールのテスト
  2. **Data Validation Tests** : データ品質のテスト
  3. **Model Performance Tests** : モデル性能のテスト
  4. **Integration Tests** : システム全体のテスト

### 1\. Unit Tests for ML Code
    
    
    # tests/test_preprocessing.py
    import pytest
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    class TestPreprocessing:
        """前処理関数のユニットテスト"""
    
        def test_handle_missing_values(self):
            """欠損値処理のテスト"""
            # テストデータ
            data = pd.DataFrame({
                'A': [1, 2, np.nan, 4],
                'B': [5, np.nan, 7, 8]
            })
    
            # 欠損値を中央値で補完
            filled = data.fillna(data.median())
    
            # アサーション
            assert filled.isnull().sum().sum() == 0, "欠損値が残っている"
            assert filled['A'].iloc[2] == 2.0, "中央値が正しく計算されていない"
    
        def test_scaling(self):
            """スケーリングのテスト"""
            data = np.array([[1, 2], [3, 4], [5, 6]])
    
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
    
            # 標準化後の平均と標準偏差を検証
            assert np.allclose(scaled.mean(axis=0), 0, atol=1e-7), "平均が0でない"
            assert np.allclose(scaled.std(axis=0), 1, atol=1e-7), "標準偏差が1でない"
    
        def test_feature_engineering(self):
            """特徴量エンジニアリングのテスト"""
            df = pd.DataFrame({
                'height': [170, 180, 160],
                'weight': [65, 80, 55]
            })
    
            # BMI計算
            df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    
            # 期待値と比較
            expected_bmi = [22.49, 24.69, 21.48]
            assert np.allclose(df['bmi'].values, expected_bmi, atol=0.01)
    
    # pytestで実行
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 2\. Data Validation Tests
    
    
    # tests/test_data_validation.py
    import pytest
    import pandas as pd
    import numpy as np
    
    class TestDataValidation:
        """データ品質検証テスト"""
    
        @pytest.fixture
        def sample_data(self):
            """テスト用サンプルデータ"""
            return pd.DataFrame({
                'age': [25, 30, 35, 40, 45],
                'income': [50000, 60000, 70000, 80000, 90000],
                'score': [0.7, 0.8, 0.6, 0.9, 0.75]
            })
    
        def test_no_missing_values(self, sample_data):
            """欠損値がないことを確認"""
            assert sample_data.isnull().sum().sum() == 0, "欠損値が存在する"
    
        def test_data_types(self, sample_data):
            """データ型が正しいことを確認"""
            assert sample_data['age'].dtype in [np.int64, np.float64]
            assert sample_data['income'].dtype in [np.int64, np.float64]
            assert sample_data['score'].dtype == np.float64
    
        def test_value_ranges(self, sample_data):
            """値の範囲が適切であることを確認"""
            assert (sample_data['age'] >= 0).all(), "年齢に負の値がある"
            assert (sample_data['age'] <= 120).all(), "年齢が異常に高い"
            assert (sample_data['income'] >= 0).all(), "収入に負の値がある"
            assert (sample_data['score'] >= 0).all() and (sample_data['score'] <= 1).all(), \
                   "スコアが0-1の範囲外"
    
        def test_data_shape(self, sample_data):
            """データの形状が期待通りであることを確認"""
            assert sample_data.shape == (5, 3), f"期待: (5, 3), 実際: {sample_data.shape}"
    
        def test_no_duplicates(self, sample_data):
            """重複行がないことを確認"""
            assert sample_data.duplicated().sum() == 0, "重複行が存在する"
    
        def test_statistical_properties(self, sample_data):
            """統計的性質が期待範囲内であることを確認"""
            # 年齢の平均が20-60の範囲内
            assert 20 <= sample_data['age'].mean() <= 60, "年齢の平均が異常"
    
            # スコアの標準偏差が妥当
            assert sample_data['score'].std() < 0.5, "スコアのばらつきが大きすぎる"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 3\. Model Performance Tests
    
    
    # tests/test_model_performance.py
    import pytest
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    class TestModelPerformance:
        """モデル性能テスト"""
    
        @pytest.fixture
        def trained_model(self):
            """学習済みモデルを返す"""
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
            """最低精度を満たすことを確認"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            accuracy = accuracy_score(y_test, y_pred)
            min_accuracy = 0.80  # 最低80%の精度を要求
    
            assert accuracy >= min_accuracy, \
                   f"精度が基準以下: {accuracy:.3f} < {min_accuracy}"
    
        def test_precision_recall(self, trained_model):
            """適切な精度と再現率を確認"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
    
            assert precision >= 0.75, f"適合率が低い: {precision:.3f}"
            assert recall >= 0.75, f"再現率が低い: {recall:.3f}"
    
        def test_f1_score(self, trained_model):
            """F1スコアが基準を満たすことを確認"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            f1 = f1_score(y_test, y_pred)
            min_f1 = 0.78
    
            assert f1 >= min_f1, f"F1スコアが基準以下: {f1:.3f} < {min_f1}"
    
        def test_no_performance_regression(self, trained_model):
            """性能が前回より劣化していないことを確認"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            current_accuracy = accuracy_score(y_test, y_pred)
            baseline_accuracy = 0.85  # 前回のベースライン
            tolerance = 0.02  # 許容誤差
    
            assert current_accuracy >= baseline_accuracy - tolerance, \
                   f"性能が劣化: {current_accuracy:.3f} < {baseline_accuracy - tolerance:.3f}"
    
        def test_prediction_distribution(self, trained_model):
            """予測分布が妥当であることを確認"""
            model, X_test, y_test = trained_model
            y_pred = model.predict(X_test)
    
            # クラス不均衡が極端でないことを確認
            class_0_ratio = (y_pred == 0).sum() / len(y_pred)
    
            assert 0.2 <= class_0_ratio <= 0.8, \
                   f"予測分布が偏っている: クラス0の割合 = {class_0_ratio:.2%}"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### 4\. Integration Tests
    
    
    # tests/test_integration.py
    import pytest
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    
    class TestIntegration:
        """統合テスト：エンドツーエンドのパイプライン"""
    
        @pytest.fixture
        def sample_pipeline(self):
            """完全な機械学習パイプライン"""
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
    
        @pytest.fixture
        def sample_data_with_issues(self):
            """問題を含むデータ（欠損値、外れ値等）"""
            np.random.seed(42)
            data = pd.DataFrame({
                'feature1': [1, 2, np.nan, 4, 5, 100],  # 欠損値と外れ値
                'feature2': [10, 20, 30, 40, np.nan, 60],
                'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            })
            labels = np.array([0, 0, 1, 1, 0, 1])
            return data, labels
    
        def test_pipeline_handles_missing_values(self, sample_pipeline, sample_data_with_issues):
            """パイプラインが欠損値を処理できることを確認"""
            X, y = sample_data_with_issues
    
            # パイプラインの実行
            sample_pipeline.fit(X, y)
            predictions = sample_pipeline.predict(X)
    
            # 予測が全データに対して行われたことを確認
            assert len(predictions) == len(y), "予測数が入力データ数と一致しない"
            assert not np.isnan(predictions).any(), "予測にNaNが含まれている"
    
        def test_pipeline_reproducibility(self, sample_pipeline, sample_data_with_issues):
            """パイプラインの再現性を確認"""
            X, y = sample_data_with_issues
    
            # 1回目の学習と予測
            sample_pipeline.fit(X, y)
            pred1 = sample_pipeline.predict(X)
    
            # 2回目の学習と予測（同じデータ）
            sample_pipeline.fit(X, y)
            pred2 = sample_pipeline.predict(X)
    
            # 結果が同一であることを確認
            assert np.array_equal(pred1, pred2), "同じデータで結果が再現されない"
    
        def test_pipeline_training_and_inference(self, sample_pipeline):
            """学習と推論のフローが正常に動作することを確認"""
            # 訓練データ
            np.random.seed(42)
            X_train = pd.DataFrame(np.random.randn(100, 3))
            y_train = np.random.randint(0, 2, 100)
    
            # 学習
            sample_pipeline.fit(X_train, y_train)
    
            # 新しいデータで推論
            X_new = pd.DataFrame(np.random.randn(10, 3))
            predictions = sample_pipeline.predict(X_new)
    
            # 予測が適切な形式であることを確認
            assert predictions.shape == (10,), "予測の形状が不正"
            assert set(predictions).issubset({0, 1}), "予測値が期待されるクラスでない"
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

### pytest実行例
    
    
    # 全テストを実行
    pytest tests/ -v
    
    # 特定のテストファイルのみ実行
    pytest tests/test_model_performance.py -v
    
    # 詳細な出力とカバレッジレポート
    pytest tests/ -v --cov=src --cov-report=html
    

**出力例** ：
    
    
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

### GitHub Actionsとは

**GitHub Actions** は、GitHub上でCI/CDワークフローを自動化するためのツールです。コード変更時に自動的にテスト、モデル学習、デプロイを実行できます。

### 基本的なワークフロー構成
    
    
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
    

### モデル学習の自動化
    
    
    # .github/workflows/train-model.yml
    name: Train and Validate Model
    
    on:
      schedule:
        # 毎日午前2時に実行
        - cron: '0 2 * * *'
      workflow_dispatch:  # 手動実行も可能
    
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
    

### 性能回帰テスト
    
    
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
            # 前回のベースラインモデルを取得
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
              ## モデル性能比較結果
    
              | メトリクス | ベースライン | 新モデル | 変化 |
              |-----------|-------------|---------|------|
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
    

### 完全なCI/CD YAMLの例
    
    
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

## 5.4 デプロイメント戦略

### 1\. Blue-Green Deployment

**Blue-Greenデプロイメント** は、2つの同一環境（BlueとGreen）を用意し、瞬時に切り替えることでダウンタイムを最小化する手法です。
    
    
    ```mermaid
    graph LR
        A[トラフィック] --> B[ロードバランサー]
        B --> C[Blue環境旧モデル v1.0]
        B -.切替.-> D[Green環境新モデル v2.0]
    
        style C fill:#a7c7e7
        style D fill:#90ee90
    ```
    
    
    # scripts/blue_green_deploy.py
    import boto3
    import time
    from typing import Dict
    
    class BlueGreenDeployer:
        """Blue-Greenデプロイメントの実装"""
    
        def __init__(self, load_balancer_name: str):
            self.elb = boto3.client('elbv2')
            self.lb_name = load_balancer_name
    
        def deploy_green(self, model_path: str, target_group_name: str):
            """Green環境に新モデルをデプロイ"""
            print(f"Green環境にモデルをデプロイ中: {model_path}")
    
            # 新しいターゲットグループにモデルをデプロイ
            # （実装は環境に依存）
            self._deploy_model_to_target_group(model_path, target_group_name)
    
            print("Green環境へのデプロイ完了")
    
        def run_health_checks(self, target_group_arn: str) -> bool:
            """ヘルスチェックを実行"""
            print("Green環境のヘルスチェック中...")
    
            response = self.elb.describe_target_health(
                TargetGroupArn=target_group_arn
            )
    
            healthy_count = sum(
                1 for target in response['TargetHealthDescriptions']
                if target['TargetHealth']['State'] == 'healthy'
            )
    
            total_count = len(response['TargetHealthDescriptions'])
    
            is_healthy = healthy_count == total_count and total_count > 0
            print(f"ヘルス: {healthy_count}/{total_count} 正常")
    
            return is_healthy
    
        def switch_traffic(self, listener_arn: str, green_target_group_arn: str):
            """トラフィックをGreen環境に切り替え"""
            print("トラフィックをGreen環境に切り替え中...")
    
            self.elb.modify_listener(
                ListenerArn=listener_arn,
                DefaultActions=[
                    {
                        'Type': 'forward',
                        'TargetGroupArn': green_target_group_arn
                    }
                ]
            )
    
            print("トラフィック切り替え完了")
    
        def rollback(self, listener_arn: str, blue_target_group_arn: str):
            """Blue環境にロールバック"""
            print("Blue環境にロールバック中...")
    
            self.elb.modify_listener(
                ListenerArn=listener_arn,
                DefaultActions=[
                    {
                        'Type': 'forward',
                        'TargetGroupArn': blue_target_group_arn
                    }
                ]
            )
    
            print("ロールバック完了")
    
        def _deploy_model_to_target_group(self, model_path: str, target_group: str):
            """モデルを指定されたターゲットグループにデプロイ（実装例）"""
            # 実際の実装はインフラに依存
            time.sleep(2)  # デプロイのシミュレーション
    
    # 使用例
    if __name__ == "__main__":
        deployer = BlueGreenDeployer("my-load-balancer")
    
        # Step 1: Green環境にデプロイ
        deployer.deploy_green(
            model_path="s3://models/model_v2.pkl",
            target_group_name="green-target-group"
        )
    
        # Step 2: ヘルスチェック
        green_tg_arn = "arn:aws:elasticloadbalancing:region:account:targetgroup/green/xxx"
        if deployer.run_health_checks(green_tg_arn):
            # Step 3: トラフィック切り替え
            listener_arn = "arn:aws:elasticloadbalancing:region:account:listener/xxx"
            deployer.switch_traffic(listener_arn, green_tg_arn)
            print("✓ Blue-Greenデプロイメント成功")
        else:
            print("✗ ヘルスチェック失敗: デプロイ中止")
    

### 2\. Canary Deployment

**Canaryデプロイメント** は、新モデルを少数のユーザーに段階的に公開し、問題がなければ徐々に拡大する手法です。
    
    
    # scripts/canary_deploy.py
    import time
    import random
    from typing import List, Dict
    
    class CanaryDeployer:
        """Canaryデプロイメントの実装"""
    
        def __init__(self, old_model, new_model):
            self.old_model = old_model
            self.new_model = new_model
            self.canary_percentage = 0
            self.metrics = {'old': [], 'new': []}
    
        def predict(self, X):
            """Canary比率に基づいて予測"""
            if random.random() * 100 < self.canary_percentage:
                # 新モデルで予測
                prediction = self.new_model.predict(X)
                self.metrics['new'].append(prediction)
                return prediction, 'new'
            else:
                # 旧モデルで予測
                prediction = self.old_model.predict(X)
                self.metrics['old'].append(prediction)
                return prediction, 'old'
    
        def increase_canary_traffic(self, increment: int = 10):
            """Canary比率を増やす"""
            self.canary_percentage = min(100, self.canary_percentage + increment)
            print(f"Canary比率: {self.canary_percentage}%")
    
        def rollback(self):
            """Canaryをロールバック"""
            self.canary_percentage = 0
            print("Canaryをロールバック: 旧モデルのみ使用")
    
        def full_rollout(self):
            """新モデルに完全移行"""
            self.canary_percentage = 100
            print("新モデルに完全移行")
    
        def get_metrics_comparison(self) -> Dict:
            """新旧モデルのメトリクスを比較"""
            return {
                'old_model_requests': len(self.metrics['old']),
                'new_model_requests': len(self.metrics['new']),
                'canary_percentage': self.canary_percentage
            }
    
    # Canaryデプロイメントのシミュレーション
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # データとモデルの準備
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    old_model = RandomForestClassifier(n_estimators=50, random_state=42)
    old_model.fit(X[:800], y[:800])
    
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X[:800], y[:800])
    
    # Canaryデプロイメント
    deployer = CanaryDeployer(old_model, new_model)
    
    # 段階的にCanary比率を増やす
    stages = [10, 25, 50, 75, 100]
    
    for stage in stages:
        print(f"\n=== Stage: Canary {stage}% ===")
        deployer.canary_percentage = stage
    
        # 100リクエストをシミュレート
        for i in range(100):
            idx = random.randint(800, 999)
            prediction, model_used = deployer.predict(X[idx:idx+1])
    
        # メトリクスの確認
        metrics = deployer.get_metrics_comparison()
        print(f"旧モデル使用: {metrics['old_model_requests']} リクエスト")
        print(f"新モデル使用: {metrics['new_model_requests']} リクエスト")
    
        # 問題があればロールバック（この例では常に成功）
        time.sleep(1)
    
    print("\n✓ Canaryデプロイメント成功: 新モデルに完全移行")
    

**出力例** ：
    
    
    === Stage: Canary 10% ===
    Canary比率: 10%
    旧モデル使用: 91 リクエスト
    新モデル使用: 9 リクエスト
    
    === Stage: Canary 25% ===
    Canary比率: 25%
    旧モデル使用: 166 リクエスト
    新モデル使用: 34 リクエスト
    
    === Stage: Canary 50% ===
    Canary比率: 50%
    旧モデル使用: 216 リクエスト
    新モデル使用: 84 リクエスト
    
    === Stage: Canary 75% ===
    Canary比率: 75%
    旧モデル使用: 241 リクエスト
    新モデル使用: 159 リクエスト
    
    === Stage: Canary 100% ===
    Canary比率: 100%
    旧モデル使用: 241 リクエスト
    新モデル使用: 259 リクエスト
    
    ✓ Canaryデプロイメント成功: 新モデルに完全移行
    

### 3\. Shadow Deployment

**Shadowデプロイメント** は、新モデルを本番環境で実行するが、実際のユーザーには旧モデルの結果を返す手法です。これにより、新モデルの性能をリスクなく評価できます。
    
    
    # scripts/shadow_deploy.py
    import time
    import numpy as np
    from typing import Tuple, Dict
    from sklearn.metrics import accuracy_score
    
    class ShadowDeployer:
        """Shadowデプロイメントの実装"""
    
        def __init__(self, production_model, shadow_model):
            self.production_model = production_model
            self.shadow_model = shadow_model
            self.shadow_predictions = []
            self.production_predictions = []
            self.ground_truth = []
    
        def predict(self, X, y_true=None):
            """本番モデルとShadowモデルの両方で予測"""
            # 本番モデルで予測（ユーザーに返す）
            prod_pred = self.production_model.predict(X)
            self.production_predictions.extend(prod_pred)
    
            # Shadowモデルで予測（ログのみ、ユーザーには返さない）
            shadow_pred = self.shadow_model.predict(X)
            self.shadow_predictions.extend(shadow_pred)
    
            # 正解ラベル（取得可能な場合）
            if y_true is not None:
                self.ground_truth.extend(y_true)
    
            # ユーザーには本番モデルの結果のみ返す
            return prod_pred
    
        def compare_models(self) -> Dict:
            """本番モデルとShadowモデルのパフォーマンスを比較"""
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
    
            # 予測の一致度
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
            """Shadowモデルを本番に昇格すべきか判定"""
            comparison = self.compare_models()
    
            if 'error' in comparison:
                return False
    
            return comparison['improvement'] >= min_improvement
    
    # Shadowデプロイメントのシミュレーション
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # データ準備
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 本番モデル（RandomForest）
    production_model = RandomForestClassifier(n_estimators=50, random_state=42)
    production_model.fit(X_train, y_train)
    
    # Shadowモデル（GradientBoosting - より高性能）
    shadow_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    shadow_model.fit(X_train, y_train)
    
    # Shadowデプロイメント
    deployer = ShadowDeployer(production_model, shadow_model)
    
    print("=== Shadowデプロイメント開始 ===\n")
    
    # 本番トラフィックをシミュレート（バッチで処理）
    batch_size = 50
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
    
        # 両方のモデルで予測（ユーザーには本番モデルの結果のみ）
        deployer.predict(X_batch, y_batch)
    
        time.sleep(0.1)  # 本番環境のシミュレーション
    
    # モデル比較
    comparison = deployer.compare_models()
    
    print("=== モデル性能比較 ===")
    print(f"本番モデル精度: {comparison['production_accuracy']:.3f}")
    print(f"Shadowモデル精度: {comparison['shadow_accuracy']:.3f}")
    print(f"改善: {comparison['improvement']:.3f} ({comparison['improvement']*100:+.1f}%)")
    print(f"予測一致度: {comparison['prediction_agreement']:.2%}")
    print(f"総予測数: {comparison['total_predictions']}")
    
    # 昇格判定
    if deployer.should_promote_shadow(min_improvement=0.02):
        print("\n✓ Shadowモデルを本番に昇格します")
    else:
        print("\n✗ 改善が不十分: Shadowモデルは昇格しません")
    

**出力例** ：
    
    
    === Shadowデプロイメント開始 ===
    
    === モデル性能比較 ===
    本番モデル精度: 0.883
    Shadowモデル精度: 0.915
    改善: 0.032 (+3.2%)
    予測一致度: 94.83%
    総予測数: 600
    
    ✓ Shadowモデルを本番に昇格します
    

### 4\. A/Bテスト
    
    
    # scripts/ab_test.py
    import numpy as np
    from scipy import stats
    from typing import Dict, List
    
    class ABTester:
        """A/Bテストの実装"""
    
        def __init__(self, model_a, model_b):
            self.model_a = model_a
            self.model_b = model_b
            self.results_a = []
            self.results_b = []
    
        def assign_and_predict(self, X, y_true, user_id: int):
            """ユーザーをグループA/Bに割り当てて予測"""
            # ユーザーIDのハッシュでグループを決定（一貫性を保つ）
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
            """統計的有意性を計算"""
            accuracy_a = np.mean(self.results_a)
            accuracy_b = np.mean(self.results_b)
    
            # 2標本t検定
            t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
    
            # 効果量（Cohen's d）
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
            """勝者を推奨"""
            stats_result = self.calculate_statistics()
    
            if not stats_result['is_significant']:
                return "No significant difference"
    
            if stats_result['difference'] > 0:
                return "Model B (statistically significant improvement)"
            else:
                return "Model A (statistically significant better)"
    
    # A/Bテストのシミュレーション
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # データ準備
    X, y = make_classification(
        n_samples=3000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # モデルA（現行）
    model_a = RandomForestClassifier(n_estimators=50, random_state=42)
    model_a.fit(X_train, y_train)
    
    # モデルB（新規）
    model_b = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_b.fit(X_train, y_train)
    
    # A/Bテスト実行
    ab_tester = ABTester(model_a, model_b)
    
    print("=== A/Bテスト開始 ===\n")
    
    # ユーザーごとに予測（シミュレーション）
    for user_id in range(len(X_test)):
        prediction, group = ab_tester.assign_and_predict(
            X_test[user_id:user_id+1],
            y_test[user_id],
            user_id
        )
    
    # 統計分析
    stats_result = ab_tester.calculate_statistics()
    
    print("=== A/Bテスト結果 ===")
    print(f"グループA（現行モデル）:")
    print(f"  サンプル数: {stats_result['sample_size_a']}")
    print(f"  精度: {stats_result['group_a_accuracy']:.3f}")
    
    print(f"\nグループB（新規モデル）:")
    print(f"  サンプル数: {stats_result['sample_size_b']}")
    print(f"  精度: {stats_result['group_b_accuracy']:.3f}")
    
    print(f"\n統計分析:")
    print(f"  差分: {stats_result['difference']:.3f} ({stats_result['difference']*100:+.1f}%)")
    print(f"  p値: {stats_result['p_value']:.4f}")
    print(f"  統計的有意: {'Yes' if stats_result['is_significant'] else 'No'}")
    print(f"  効果量 (Cohen's d): {stats_result['cohens_d']:.3f}")
    
    print(f"\n推奨: {ab_tester.recommend_winner()}")
    

### デプロイメント戦略の比較

戦略 | リスク | ロールバック速度 | 適用場面  
---|---|---|---  
**Blue-Green** | 中 | 即座 | 迅速な切り替えが必要  
**Canary** | 低 | 段階的 | リスクを最小化したい  
**Shadow** | 最低 | 不要（本番に未反映） | 性能評価のみ  
**A/Bテスト** | 低〜中 | 段階的 | 統計的な検証が必要  
  
* * *

## 5.5 エンドツーエンドの例

### 完全なCI/CDパイプラインの実装

ここでは、データ検証からモデル学習、デプロイメント、モニタリングまでを含む完全なパイプラインを構築します。

#### ディレクトリ構成
    
    
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
    

#### データ検証モジュール
    
    
    # src/data/validation.py
    import pandas as pd
    import numpy as np
    from typing import Dict, List
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class DataValidator:
        """データ品質検証クラス"""
    
        def __init__(self, schema: Dict):
            self.schema = schema
            self.validation_errors = []
    
        def validate(self, df: pd.DataFrame) -> bool:
            """全検証を実行"""
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
                logger.info("✓ 全てのデータ検証が成功しました")
            else:
                logger.error(f"✗ {len(self.validation_errors)} 件の検証エラー")
                for error in self.validation_errors:
                    logger.error(f"  - {error}")
    
            return all_passed
    
        def _check_schema(self, df: pd.DataFrame) -> bool:
            """スキーマ（列名）の検証"""
            expected_columns = set(self.schema.keys())
            actual_columns = set(df.columns)
    
            if expected_columns != actual_columns:
                missing = expected_columns - actual_columns
                extra = actual_columns - expected_columns
    
                if missing:
                    self.validation_errors.append(f"欠落している列: {missing}")
                if extra:
                    self.validation_errors.append(f"余分な列: {extra}")
                return False
    
            return True
    
        def _check_missing_values(self, df: pd.DataFrame) -> bool:
            """欠損値の検証"""
            missing_counts = df.isnull().sum()
            total_rows = len(df)
    
            for col, count in missing_counts.items():
                if count > 0:
                    threshold = self.schema.get(col, {}).get('max_missing_rate', 0.05)
                    missing_rate = count / total_rows
    
                    if missing_rate > threshold:
                        self.validation_errors.append(
                            f"{col}: 欠損率 {missing_rate:.2%} > {threshold:.2%}"
                        )
                        return False
    
            return True
    
        def _check_value_ranges(self, df: pd.DataFrame) -> bool:
            """値の範囲検証"""
            all_valid = True
    
            for col, col_schema in self.schema.items():
                if 'min' in col_schema:
                    if (df[col] < col_schema['min']).any():
                        self.validation_errors.append(
                            f"{col}: 最小値違反 (< {col_schema['min']})"
                        )
                        all_valid = False
    
                if 'max' in col_schema:
                    if (df[col] > col_schema['max']).any():
                        self.validation_errors.append(
                            f"{col}: 最大値違反 (> {col_schema['max']})"
                        )
                        all_valid = False
    
            return all_valid
    
        def _check_data_types(self, df: pd.DataFrame) -> bool:
            """データ型の検証"""
            for col, col_schema in self.schema.items():
                if 'dtype' in col_schema:
                    expected_dtype = col_schema['dtype']
                    actual_dtype = str(df[col].dtype)
    
                    if expected_dtype not in actual_dtype:
                        self.validation_errors.append(
                            f"{col}: 型不一致 (期待: {expected_dtype}, 実際: {actual_dtype})"
                        )
                        return False
    
            return True
    
        def _check_duplicates(self, df: pd.DataFrame) -> bool:
            """重複行の検証"""
            duplicates = df.duplicated().sum()
    
            if duplicates > 0:
                max_duplicates = self.schema.get('_global', {}).get('max_duplicates', 0)
                if duplicates > max_duplicates:
                    self.validation_errors.append(
                        f"重複行が多すぎる: {duplicates} > {max_duplicates}"
                    )
                    return False
    
            return True
    
    # 使用例
    if __name__ == "__main__":
        # スキーマ定義
        schema = {
            'age': {'dtype': 'int', 'min': 0, 'max': 120, 'max_missing_rate': 0.05},
            'income': {'dtype': 'float', 'min': 0, 'max_missing_rate': 0.1},
            'score': {'dtype': 'float', 'min': 0, 'max': 1, 'max_missing_rate': 0.02},
            '_global': {'max_duplicates': 10}
        }
    
        # テストデータ
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'score': [0.7, 0.8, 0.6, 0.9, 0.75]
        })
    
        validator = DataValidator(schema)
        is_valid = validator.validate(df)
    
        if is_valid:
            print("データ検証成功")
        else:
            print("データ検証失敗")
    

#### モデル学習とメトリクス保存
    
    
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
        """モデル学習クラス"""
    
        def __init__(self, config: dict):
            self.config = config
            self.model = None
            self.metrics = {}
    
        def train(self, X_train, y_train):
            """モデルを学習"""
            logger.info("モデル学習開始...")
    
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', None),
                random_state=self.config.get('random_state', 42)
            )
    
            self.model.fit(X_train, y_train)
    
            # クロスバリデーション
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=self.config.get('cv_folds', 5)
            )
    
            logger.info(f"CV精度: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
            self.metrics['cv_accuracy'] = cv_scores.mean()
            self.metrics['cv_std'] = cv_scores.std()
    
        def evaluate(self, X_test, y_test):
            """モデルを評価"""
            logger.info("モデル評価中...")
    
            y_pred = self.model.predict(X_test)
    
            self.metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
            self.metrics['test_precision'] = precision_score(y_test, y_pred, average='weighted')
            self.metrics['test_recall'] = recall_score(y_test, y_pred, average='weighted')
            self.metrics['test_f1'] = f1_score(y_test, y_pred, average='weighted')
    
            logger.info(f"テスト精度: {self.metrics['test_accuracy']:.3f}")
            logger.info(f"F1スコア: {self.metrics['test_f1']:.3f}")
    
            return self.metrics
    
        def save_model(self, path: str):
            """モデルを保存"""
            joblib.dump(self.model, path)
            logger.info(f"モデルを保存: {path}")
    
        def save_metrics(self, path: str):
            """メトリクスを保存"""
            with open(path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"メトリクスを保存: {path}")
    
        def check_performance_threshold(self, min_accuracy: float = 0.8) -> bool:
            """性能閾値をチェック"""
            if self.metrics['test_accuracy'] < min_accuracy:
                logger.error(
                    f"精度が閾値以下: {self.metrics['test_accuracy']:.3f} < {min_accuracy}"
                )
                return False
    
            logger.info(f"✓ 性能閾値をクリア: {self.metrics['test_accuracy']:.3f}")
            return True
    
    # 使用例
    if __name__ == "__main__":
        from sklearn.datasets import make_classification
    
        # データ生成
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # 設定
        config = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'cv_folds': 5
        }
    
        # 学習と評価
        trainer = ModelTrainer(config)
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)
    
        # 性能チェック
        if trainer.check_performance_threshold(min_accuracy=0.85):
            trainer.save_model('models/model.pkl')
            trainer.save_metrics('models/metrics.json')
        else:
            logger.error("モデルが基準を満たしていません")
    

#### 完全なGitHub Actionsワークフロー
    
    
    # .github/workflows/complete-ml-cicd.yml
    name: Complete ML CI/CD Pipeline
    
    on:
      push:
        branches: [ main ]
      pull_request:
        branches: [ main ]
      schedule:
        # 毎日午前2時に再学習
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
            # データのダウンロード（実際の実装に置き換え）
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

## 5.6 本章のまとめ

### 学んだこと

  1. **ML特有のCI/CD**

     * コード + データ + モデルの統合管理
     * データドリフトへの対応
     * 継続的な再学習とモニタリング
  2. **自動テスト戦略**

     * Unit Tests: コードの正確性
     * Data Validation: データ品質
     * Model Performance Tests: モデル性能
     * Integration Tests: システム全体
  3. **GitHub Actionsの活用**

     * 自動化されたテスト実行
     * モデル学習パイプライン
     * 性能回帰テスト
  4. **デプロイメント戦略**

     * Blue-Green: 迅速な切り替え
     * Canary: リスク最小化
     * Shadow: 安全な評価
     * A/Bテスト: 統計的検証
  5. **本番運用**

     * エンドツーエンドのパイプライン
     * 自動デプロイメント
     * モニタリングとロールバック

### ベストプラクティス

原則 | 説明  
---|---  
**自動化優先** | 手動操作を最小化し、再現性を確保  
**段階的デプロイ** | Canaryやshadowで段階的にリスクを低減  
**性能閾値の設定** | 明確な基準で自動的に合否判定  
**迅速なロールバック** | 問題発生時に即座に前バージョンに戻せる仕組み  
**包括的モニタリング** | モデル性能とシステムメトリクスを常時監視  
  
### 次のステップ

本章で学んだCI/CDパイプラインを基盤として、以下の拡張を検討できます：

  * 特徴量ストアの統合
  * MLflowによる実験管理
  * Kubernetesでのスケーラブルなデプロイメント
  * リアルタイム推論システム
  * モデルの説明可能性の統合

* * *

## 演習問題

### 問題1（難易度：easy）

従来のソフトウェア開発のCI/CDと、機械学習のCI/CDの主な違いを3つ挙げて説明してください。

解答例

**解答** ：

  1. **テスト対象の違い**

     * 従来: コードのみ
     * ML: コード + データ + モデルの3要素
     * 理由: MLではデータ品質とモデル性能が結果に直接影響
  2. **品質指標の違い**

     * 従来: テストパス率、コードカバレッジ
     * ML: 精度、再現率、F1スコア、データドリフト
     * 理由: MLは統計的な性能指標で評価
  3. **継続的な更新の必要性**

     * 従来: 機能追加や修正時に更新
     * ML: データ分布の変化に応じて定期的に再学習
     * 理由: データドリフトによりモデル性能が劣化

### 問題2（難易度：medium）

pytestを使って、モデルの最低精度が85%であることを確認するテストを作成してください。

解答例
    
    
    import pytest
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    class TestModelAccuracy:
        """モデル精度のテスト"""
    
        @pytest.fixture
        def trained_model_and_data(self):
            """学習済みモデルとテストデータを準備"""
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
            """最低精度85%を満たすことを確認"""
            model, X_test, y_test = trained_model_and_data
    
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
    
            min_accuracy = 0.85
    
            assert accuracy >= min_accuracy, \
                   f"精度が基準以下: {accuracy:.3f} < {min_accuracy}"
    
            print(f"✓ テスト成功: 精度 = {accuracy:.3f}")
    
    if __name__ == "__main__":
        pytest.main([__file__, '-v'])
    

**実行** ：
    
    
    pytest test_model_accuracy.py -v
    

### 問題3（難易度：medium）

Blue-GreenデプロイメントとCanaryデプロイメントの違いを説明し、それぞれをどのような状況で使うべきか述べてください。

解答例

**解答** ：

**Blue-Greenデプロイメント** ：

  * 特徴: 2つの完全な環境を用意し、瞬時に切り替え
  * メリット: 即座のロールバックが可能、ダウンタイムなし
  * デメリット: 2倍のリソースが必要、全ユーザーに一度に影響

**Canaryデプロイメント** ：

  * 特徴: 新バージョンを一部のユーザーに段階的に公開
  * メリット: リスク最小化、問題の早期発見
  * デメリット: 切り替えに時間がかかる、複雑な管理

**使い分け** ：

状況 | 推奨戦略 | 理由  
---|---|---  
迅速な展開が必要 | Blue-Green | 即座に切り替え可能  
リスクを最小化したい | Canary | 段階的に影響範囲を拡大  
大きな変更 | Canary | 問題を早期に検出  
小さな改善 | Blue-Green | シンプルで迅速  
リソース制約あり | Canary | 追加リソース不要  
  
### 問題4（難易度：hard）

データドリフトを検出し、精度が閾値を下回った場合にモデルを再学習するPythonスクリプトを作成してください。

解答例
    
    
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
        """データドリフト検出と自動再学習"""
    
        def __init__(self, model_path: str, accuracy_threshold: float = 0.8):
            self.model_path = model_path
            self.accuracy_threshold = accuracy_threshold
            self.model = joblib.load(model_path) if model_path else None
            self.reference_data = None
    
        def set_reference_data(self, X_ref: np.ndarray):
            """基準データを設定"""
            self.reference_data = X_ref
            logger.info(f"基準データを設定: {X_ref.shape}")
    
        def detect_drift(self, X_new: np.ndarray, alpha: float = 0.05) -> bool:
            """Kolmogorov-Smirnov検定でドリフトを検出"""
            if self.reference_data is None:
                raise ValueError("基準データが設定されていません")
    
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
                        f"特徴量{i}にドリフト検出: p={p_value:.4f} < {alpha}"
                    )
    
            drift_ratio = sum(drifts) / len(drifts)
            logger.info(f"ドリフト検出率: {drift_ratio:.2%}")
    
            return drift_ratio > 0.3  # 30%以上の特徴量でドリフト
    
        def check_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
            """モデル性能をチェック"""
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
    
            logger.info(f"現在の精度: {accuracy:.3f}")
    
            return accuracy
    
        def retrain(self, X_train: np.ndarray, y_train: np.ndarray):
            """モデルを再学習"""
            logger.info("モデル再学習開始...")
    
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.model.fit(X_train, y_train)
    
            logger.info("モデル再学習完了")
    
        def save_model(self, path: str = None):
            """モデルを保存"""
            save_path = path or self.model_path
            joblib.dump(self.model, save_path)
            logger.info(f"モデルを保存: {save_path}")
    
        def run_monitoring_cycle(
            self,
            X_new: np.ndarray,
            y_new: np.ndarray,
            X_train: np.ndarray,
            y_train: np.ndarray
        ):
            """モニタリングサイクルを実行"""
            logger.info("=== モニタリングサイクル開始 ===")
    
            # 1. ドリフト検出
            has_drift = self.detect_drift(X_new)
    
            # 2. 性能チェック
            accuracy = self.check_performance(X_new, y_new)
    
            # 3. 再学習判定
            needs_retrain = (
                has_drift or
                accuracy < self.accuracy_threshold
            )
    
            if needs_retrain:
                logger.warning(
                    f"再学習が必要: ドリフト={has_drift}, "
                    f"精度={accuracy:.3f} < {self.accuracy_threshold}"
                )
    
                # 4. 再学習実行
                self.retrain(X_train, y_train)
    
                # 5. 新モデルの評価
                new_accuracy = self.check_performance(X_new, y_new)
                logger.info(f"再学習後の精度: {new_accuracy:.3f}")
    
                # 6. モデル保存
                self.save_model()
    
                return True
            else:
                logger.info("再学習不要: モデルは正常に動作中")
                return False
    
    # 使用例
    if __name__ == "__main__":
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
    
        # 初期データとモデル
        X_initial, y_initial = make_classification(
            n_samples=1000, n_features=20, random_state=42
        )
        X_train, X_ref, y_train, y_ref = train_test_split(
            X_initial, y_initial, test_size=0.3, random_state=42
        )
    
        # 初期モデル学習
        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train, y_train)
        joblib.dump(initial_model, 'model.pkl')
    
        # モニタリングシステム
        monitor = DriftDetectorAndRetrainer(
            model_path='model.pkl',
            accuracy_threshold=0.85
        )
        monitor.set_reference_data(X_ref)
    
        # 新しいデータ（ドリフトあり）
        X_new, y_new = make_classification(
            n_samples=200, n_features=20, random_state=100
        )
        # ドリフトをシミュレート
        X_new = X_new + np.random.normal(0, 1.5, X_new.shape)
    
        # モニタリングサイクル実行
        was_retrained = monitor.run_monitoring_cycle(
            X_new, y_new, X_train, y_train
        )
    
        if was_retrained:
            print("\n✓ モデルが再学習されました")
        else:
            print("\n✓ モデルは正常に動作しています")
    

### 問題5（難易度：hard）

GitHub Actionsで、Pull Request時にモデルの性能が前回のベースラインより劣化していないかをチェックし、結果をPRにコメントするワークフローを作成してください。

解答例
    
    
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
            # 前回のベースラインモデルをダウンロード
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
              ${statusEmoji} **モデル性能比較結果**
    
              | メトリクス | ベースライン | 新モデル (PR) | 変化 |
              |-----------|-------------|--------------|------|
              | **Accuracy** | ${comparison.baseline.accuracy.toFixed(3)} | ${comparison.new.accuracy.toFixed(3)} | ${accuracyDiff >= 0 ? '+' : ''}${accuracyDiff.toFixed(3)} |
              | **Precision** | ${comparison.baseline.precision.toFixed(3)} | ${comparison.new.precision.toFixed(3)} | ${(comparison.new.precision - comparison.baseline.precision).toFixed(3)} |
              | **Recall** | ${comparison.baseline.recall.toFixed(3)} | ${comparison.new.recall.toFixed(3)} | ${(comparison.new.recall - comparison.baseline.recall).toFixed(3)} |
              | **F1 Score** | ${comparison.baseline.f1.toFixed(3)} | ${comparison.new.f1.toFixed(3)} | ${f1Diff >= 0 ? '+' : ''}${f1Diff.toFixed(3)} |
    
              ### 判定
              ${accuracyDiff >= -0.01 ?
                '✅ **合格**: 性能劣化は許容範囲内です' :
                '❌ **不合格**: 性能が著しく劣化しています（許容: -1%）'}
    
              
              詳細情報
    
              - 訓練時間: ${comparison.training_time}秒
              - モデルサイズ: ${(comparison.model_size_mb).toFixed(2)} MB
              - テストサンプル数: ${comparison.test_samples}
    
              
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
    

**補助スクリプト（scripts/compare_models.py）** ：
    
    
    import argparse
    import json
    import joblib
    import time
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    def compare_models(baseline_path, new_path, output_path):
        # モデル読み込み
        baseline_model = joblib.load(baseline_path)
        new_model = joblib.load(new_path)
    
        # テストデータ
        X, y = load_iris(return_X_y=True)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # ベースライン評価
        baseline_pred = baseline_model.predict(X_test)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, baseline_pred),
            'precision': precision_score(y_test, baseline_pred, average='weighted'),
            'recall': recall_score(y_test, baseline_pred, average='weighted'),
            'f1': f1_score(y_test, baseline_pred, average='weighted')
        }
    
        # 新モデル評価
        start = time.time()
        new_pred = new_model.predict(X_test)
        training_time = time.time() - start
    
        new_metrics = {
            'accuracy': accuracy_score(y_test, new_pred),
            'precision': precision_score(y_test, new_pred, average='weighted'),
            'recall': recall_score(y_test, new_pred, average='weighted'),
            'f1': f1_score(y_test, new_pred, average='weighted')
        }
    
        # 比較結果
        result = {
            'baseline': baseline_metrics,
            'new': new_metrics,
            'training_time': training_time,
            'model_size_mb': 1.5,  # 実際のサイズを計算
            'test_samples': len(y_test)
        }
    
        # 保存
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
        print(f"比較結果を保存: {output_path}")
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--baseline', required=True)
        parser.add_argument('--new', required=True)
        parser.add_argument('--output', required=True)
        args = parser.parse_args()
    
        compare_models(args.baseline, args.new, args.output)
    

* * *

## 参考文献

  1. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  2. Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  3. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  4. Forsgren, N., Humble, J., & Kim, G. (2018). _Accelerate: The Science of Lean Software and DevOps_. IT Revolution Press.
  5. Sato, D., Wider, A., & Windheuser, C. (2019). "Continuous Delivery for Machine Learning." _Martin Fowler's Blog_.
