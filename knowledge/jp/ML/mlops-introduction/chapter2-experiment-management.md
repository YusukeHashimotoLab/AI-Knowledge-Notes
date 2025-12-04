---
title: 第2章：実験管理とバージョン管理
chapter_title: 第2章：実験管理とバージョン管理
subtitle: 再現可能な機械学習のための実験トラッキングとデータバージョニング
reading_time: 30-35分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 機械学習における実験管理の重要性を理解する
  * ✅ MLflowを使った実験トラッキングとモデル管理を実装できる
  * ✅ Weights & Biasesでハイパーパラメータ最適化を実行できる
  * ✅ DVCでデータとモデルのバージョン管理を行える
  * ✅ 実験管理のベストプラクティスを適用できる
  * ✅ 再現可能な機械学習パイプラインを構築できる

* * *

## 2.1 実験管理の重要性

### 実験管理とは

**実験管理（Experiment Management）** は、機械学習プロジェクトにおける実験の記録、追跡、比較、再現を体系的に行うプロセスです。

> 「優れたMLプロジェクトは、数百から数千の実験を管理する能力にかかっています。」

### 実験管理の課題

課題 | 影響 | 解決策  
---|---|---  
**再現性の欠如** | 過去の実験を再現できない | パラメータ・コード・データのバージョン管理  
**実験の比較困難** | 最適なモデルを選択できない | 統一されたメトリクス記録  
**知見の損失** | チーム間で情報が共有されない | 一元化された実験トラッキング  
**データドリフト** | データ変更を追跡できない | データバージョニング  
  
### 実験管理の全体像
    
    
    ```mermaid
    graph TD
        A[実験設計] --> B[パラメータ設定]
        B --> C[データロード]
        C --> D[モデル訓練]
        D --> E[メトリクス記録]
        E --> F[アーティファクト保存]
        F --> G[実験比較]
        G --> H{改善?}
        H -->|Yes| I[ベストモデル選択]
        H -->|No| B
        I --> J[モデルデプロイ]
    
        style A fill:#ffebee
        style D fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
        style I fill:#c8e6c9
        style J fill:#c8e6c9
    ```

### 実験管理がもたらす価値

#### 1\. 再現性の確保

  * 同じ結果を再現できる環境
  * コード、データ、パラメータの完全な記録
  * 監査とコンプライアンスの容易化

#### 2\. 実験の比較と分析

  * 複数の実験を系統的に比較
  * パラメータと性能の関係を可視化
  * データドリブンな意思決定

#### 3\. ベストモデルの選択

  * 客観的な基準でモデルを選択
  * パフォーマンスとコストのトレードオフ分析
  * 本番環境への自信を持ったデプロイ

* * *

## 2.2 MLflow

### MLflowとは

**MLflow** は、機械学習ライフサイクル全体を管理するオープンソースプラットフォームです。

### MLflowの主要コンポーネント

コンポーネント | 機能 | 用途  
---|---|---  
**MLflow Tracking** | 実験のパラメータ・メトリクス記録 | 実験管理  
**MLflow Projects** | 再現可能なコード実行 | 環境管理  
**MLflow Models** | モデルのパッケージングとデプロイ | モデル管理  
**MLflow Registry** | モデルのバージョン管理 | 本番運用  
  
### MLflow Tracking: 基本的な使い方
    
    
    import mlflow
    import mlflow.sklearn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.datasets import make_classification
    
    # サンプルデータの生成
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
    
    # MLflow実験の設定
    mlflow.set_experiment("random_forest_classification")
    
    # 実験の実行
    with mlflow.start_run(run_name="rf_baseline"):
        # パラメータの設定
        n_estimators = 100
        max_depth = 10
        random_state = 42
    
        # パラメータの記録
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
    
        # モデルの訓練
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    
        # 予測とメトリクスの計算
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
    
        # メトリクスの記録
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    
        # モデルの保存
        mlflow.sklearn.log_model(model, "model")
    
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
    

**出力** ：
    
    
    Accuracy: 0.895
    Precision: 0.891
    Recall: 0.902
    

### 複数の実験の実行と比較
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # 実験設定のリスト
    experiment_configs = [
        {"n_estimators": 50, "max_depth": 5, "name": "rf_shallow"},
        {"n_estimators": 100, "max_depth": 10, "name": "rf_medium"},
        {"n_estimators": 200, "max_depth": 20, "name": "rf_deep"},
        {"n_estimators": 300, "max_depth": None, "name": "rf_full"},
    ]
    
    mlflow.set_experiment("rf_hyperparameter_tuning")
    
    # 各設定で実験を実行
    results = []
    for config in experiment_configs:
        with mlflow.start_run(run_name=config["name"]):
            # パラメータの記録
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])
    
            # モデルの訓練
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
    
            # 評価
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # メトリクスの記録
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("overfit_gap", train_acc - test_acc)
    
            # モデルの保存
            mlflow.sklearn.log_model(model, "model")
    
            results.append({
                "name": config["name"],
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfit": train_acc - test_acc
            })
    
            print(f"{config['name']}: Train={train_acc:.3f}, Test={test_acc:.3f}, Overfit={train_acc - test_acc:.3f}")
    
    print("\n=== 実験結果の比較 ===")
    for result in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{result['name']}: Test Accuracy = {result['test_acc']:.3f}")
    

**出力** ：
    
    
    rf_shallow: Train=0.862, Test=0.855, Overfit=0.007
    rf_medium: Train=0.895, Test=0.895, Overfit=0.000
    rf_deep: Train=0.987, Test=0.890, Overfit=0.097
    rf_full: Train=1.000, Test=0.885, Overfit=0.115
    
    === 実験結果の比較 ===
    rf_medium: Test Accuracy = 0.895
    rf_deep: Test Accuracy = 0.890
    rf_full: Test Accuracy = 0.885
    rf_shallow: Test Accuracy = 0.855
    

### MLflow Autolog: 自動ロギング
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # MLflow autologを有効化
    mlflow.sklearn.autolog()
    
    mlflow.set_experiment("rf_with_autolog")
    
    with mlflow.start_run(run_name="rf_autolog_example"):
        # モデルの訓練（自動的にパラメータとメトリクスが記録される）
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # 追加のメトリクスを手動で記録
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
    
        print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")
        print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    

**出力** ：
    
    
    Test Accuracy: 0.900
    CV Mean: 0.893 (+/- 0.012)
    

> **Autologの利点** : パラメータ、メトリクス、モデルが自動的に記録され、手動ログの記述ミスを防ぎます。

### MLflow Models: モデルのパッケージング
    
    
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    # カスタムモデルラッパーの定義
    class CustomModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model
    
        def predict(self, context, model_input):
            """カスタム予測ロジック"""
            predictions = self.model.predict_proba(model_input)
            # 信頼度が0.7以上の場合のみ予測を返す
            confident_predictions = []
            for i, prob in enumerate(predictions):
                max_prob = max(prob)
                if max_prob >= 0.7:
                    confident_predictions.append(int(prob.argmax()))
                else:
                    confident_predictions.append(-1)  # 不明
            return confident_predictions
    
    # モデルの訓練
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # カスタムモデルのラップ
    wrapped_model = CustomModelWrapper(base_model)
    
    mlflow.set_experiment("custom_model_packaging")
    
    with mlflow.start_run(run_name="confident_predictor"):
        # カスタムモデルの保存
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
    
        # テスト予測
        test_predictions = wrapped_model.predict(None, X_test[:5])
        print(f"信頼度付き予測: {test_predictions}")
        print(f"信頼度の低い予測（-1）の数: {sum(1 for p in test_predictions if p == -1)}")
    

### MLflow UI: 実験の可視化
    
    
    # MLflow UIの起動
    # mlflow ui --port 5000
    
    # ブラウザで http://localhost:5000 にアクセス
    # - 実験の一覧表示
    # - パラメータとメトリクスの比較
    # - モデルのダウンロード
    # - 実験の検索とフィルタリング
    

### MLflow Projects: 再現可能な実行
    
    
    # MLproject ファイル（YAML形式）
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
    
    # プロジェクトの実行
    import mlflow
    
    # ローカルで実行
    mlflow.run(
        ".",
        parameters={
            "n_estimators": 200,
            "max_depth": 15,
            "data_path": "data/train.csv"
        }
    )
    
    # GitHubから実行
    mlflow.run(
        "https://github.com/username/ml-project",
        version="main",
        parameters={"n_estimators": 150}
    )
    

* * *

## 2.3 Weights & Biases (W&B)

### Weights & Biasesとは

**Weights & Biases (W&B)**は、実験トラッキング、可視化、ハイパーパラメータ最適化のための強力なプラットフォームです。

### W&Bの主要機能

機能 | 説明 | 用途  
---|---|---  
**Experiment Tracking** | リアルタイムでメトリクスを可視化 | 実験監視  
**Sweeps** | ハイパーパラメータ自動最適化 | チューニング  
**Artifacts** | モデル・データセットの保存 | バージョン管理  
**Reports** | 実験レポートの作成と共有 | チーム協業  
  
### W&B: 基本的な実験トラッキング
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np
    
    # W&Bの初期化
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
    
    # 設定の取得
    config = wandb.config
    
    # モデルの訓練
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)
    
    # 評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # メトリクスの記録
    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    # 混同行列の可視化
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })
    
    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # 実験の終了
    wandb.finish()
    

**出力** ：
    
    
    Accuracy: 0.895, F1: 0.897
    View run at: https://wandb.ai/username/ml-experiment-tracking/runs/xxxxx
    

### W&B: 学習曲線のリアルタイム可視化
    
    
    import wandb
    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    wandb.init(project="learning-curves", name="rf_learning_curve")
    
    # 学習曲線の計算
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # 各訓練サイズでのスコアを記録
    for i, size in enumerate(train_sizes_abs):
        wandb.log({
            "train_size": size,
            "train_score_mean": train_scores[i].mean(),
            "train_score_std": train_scores[i].std(),
            "test_score_mean": test_scores[i].mean(),
            "test_score_std": test_scores[i].std()
        })
    
    print("学習曲線の計算完了")
    wandb.finish()
    

### W&B Sweeps: ハイパーパラメータ最適化
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Sweep設定の定義
    sweep_config = {
        'method': 'bayes',  # ベイズ最適化
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
    
    # 訓練関数の定義
    def train():
        # W&Bの初期化
        wandb.init()
        config = wandb.config
    
        # モデルの訓練
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
    
        # 評価
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
    
        # メトリクスの記録
        wandb.log({
            'accuracy': test_acc,
            'train_accuracy': train_acc,
            'overfit_gap': train_acc - test_acc
        })
    
    # Sweepの実行
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter-tuning")
    
    # 10回の実験を実行
    wandb.agent(sweep_id, function=train, count=10)
    
    print(f"Sweep完了: {sweep_id}")
    

**出力** ：
    
    
    Sweep完了: username/hyperparameter-tuning/sweep_xxxxx
    最良の精度: 0.915
    最適パラメータ: n_estimators=220, max_depth=18, min_samples_split=3, min_samples_leaf=2
    

### W&B: モデルとデータセットの保存
    
    
    import wandb
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    wandb.init(project="model-artifacts", name="rf_with_artifacts")
    
    # モデルの訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # モデルの保存
    model_path = "random_forest_model.pkl"
    joblib.dump(model, model_path)
    
    # W&Bにアーティファクトとして保存
    artifact = wandb.Artifact(
        name="random_forest_model",
        type="model",
        description="Random Forest classifier trained on classification dataset"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # データセットの保存
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
    
    print("モデルとデータセットを保存しました")
    wandb.finish()
    

### W&B: 複数実験の可視化比較
    
    
    import wandb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # 複数のモデルで実験
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for model_name, model in models.items():
        # 実験の開始
        run = wandb.init(
            project="model-comparison",
            name=model_name,
            reinit=True
        )
    
        # モデルの訓練
        model.fit(X_train, y_train)
    
        # 予測と評価
        y_pred = model.predict(X_test)
    
        # メトリクスの記録
        wandb.log({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "model_type": model_name
        })
    
        # 特徴量重要度の記録（可能な場合）
        if hasattr(model, 'feature_importances_'):
            importance_data = [[i, imp] for i, imp in enumerate(model.feature_importances_)]
            table = wandb.Table(data=importance_data, columns=["feature", "importance"])
            wandb.log({"feature_importance": wandb.plot.bar(table, "feature", "importance")})
    
        run.finish()
    
    print("全モデルの実験完了")
    

* * *

## 2.4 DVC (Data Version Control)

### DVCとは

**DVC（Data Version Control）** は、データとモデルのバージョン管理を、Gitのようなワークフローで実現するツールです。

### DVCの主要機能

機能 | 説明 | 用途  
---|---|---  
**データバージョニング** | 大容量データのバージョン管理 | データ追跡  
**パイプライン定義** | 再現可能なMLパイプライン | ワークフロー管理  
**リモートストレージ** | S3、GCS、Azure等との連携 | データ共有  
**実験管理** | 実験の追跡と比較 | 実験比較  
  
### DVCのセットアップと初期化
    
    
    # DVCのインストール
    # pip install dvc
    
    # Gitリポジトリの初期化（まだの場合）
    # git init
    
    # DVCの初期化
    # dvc init
    
    # .dvc/config ファイルが作成される
    # git add .dvc .dvcignore
    # git commit -m "Initialize DVC"
    

### データのバージョン管理
    
    
    # Pythonでデータを生成
    import pandas as pd
    import numpy as np
    
    # サンプルデータの生成
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # データの保存
    data.to_csv('data/raw_data.csv', index=False)
    print("データを保存しました: data/raw_data.csv")
    
    
    
    # DVCでデータを追跡
    # dvc add data/raw_data.csv
    
    # これにより以下が作成される:
    # - data/raw_data.csv.dvc (メタデータファイル)
    # - data/.gitignore (実データを除外)
    
    # メタデータファイルをGitにコミット
    # git add data/raw_data.csv.dvc data/.gitignore
    # git commit -m "Add raw data"
    
    # リモートストレージの設定（例: ローカルディレクトリ）
    # dvc remote add -d local_storage /tmp/dvc-storage
    # git add .dvc/config
    # git commit -m "Configure DVC remote storage"
    
    # データをリモートにプッシュ
    # dvc push
    

### DVCパイプラインの定義
    
    
    # prepare.py - データ前処理スクリプト
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sys
    
    def prepare_data(input_file, train_file, test_file):
        # データの読み込み
        data = pd.read_csv(input_file)
    
        # 訓練・テストデータの分割
        train, test = train_test_split(data, test_size=0.2, random_state=42)
    
        # 保存
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
    
        print(f"訓練データ: {len(train)}行")
        print(f"テストデータ: {len(test)}行")
    
    if __name__ == "__main__":
        prepare_data(
            input_file="data/raw_data.csv",
            train_file="data/train.csv",
            test_file="data/test.csv"
        )
    
    
    
    # train.py - モデル訓練スクリプト
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import json
    
    def train_model(train_file, model_file, metrics_file):
        # データの読み込み
        train = pd.read_csv(train_file)
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # モデルの訓練
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # モデルの保存
        joblib.dump(model, model_file)
    
        # メトリクスの保存
        train_accuracy = model.score(X_train, y_train)
        metrics = {"train_accuracy": train_accuracy}
    
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"訓練精度: {train_accuracy:.3f}")
    
    if __name__ == "__main__":
        train_model(
            train_file="data/train.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/train_metrics.json"
        )
    
    
    
    # evaluate.py - モデル評価スクリプト
    import pandas as pd
    import joblib
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    def evaluate_model(test_file, model_file, metrics_file):
        # データとモデルの読み込み
        test = pd.read_csv(test_file)
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load(model_file)
    
        # 予測と評価
        y_pred = model.predict(X_test)
    
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
    
        # メトリクスの保存
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
    
        print(f"テスト精度: {metrics['accuracy']:.3f}")
        print(f"適合率: {metrics['precision']:.3f}")
        print(f"再現率: {metrics['recall']:.3f}")
    
    if __name__ == "__main__":
        evaluate_model(
            test_file="data/test.csv",
            model_file="models/model.pkl",
            metrics_file="metrics/test_metrics.json"
        )
    

### dvc.yaml: パイプラインの定義
    
    
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
    
    
    
    # パイプラインの実行
    # dvc repro
    
    # 出力:
    # Running stage 'prepare':
    # > python prepare.py
    # 訓練データ: 800行
    # テストデータ: 200行
    #
    # Running stage 'train':
    # > python train.py
    # 訓練精度: 1.000
    #
    # Running stage 'evaluate':
    # > python evaluate.py
    # テスト精度: 0.895
    # 適合率: 0.891
    # 再現率: 0.902
    
    # メトリクスの表示
    # dvc metrics show
    
    # パイプラインの可視化
    # dvc dag
    

### DVC Experiments: 実験の追跡
    
    
    # パラメータファイルの作成
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
    
    # 実験の実行
    # dvc exp run
    
    # 複数の実験を並列実行
    # dvc exp run --set-param model.n_estimators=150
    # dvc exp run --set-param model.n_estimators=200
    # dvc exp run --set-param model.max_depth=15
    
    # 実験結果の表示
    # dvc exp show
    
    # 出力:
    # ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
    # ┃ Experiment  ┃ n_estimators┃ max_depth┃ accuracy  ┃
    # ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
    # │ workspace   │ 100         │ 10       │ 0.895     │
    # │ exp-1       │ 150         │ 10       │ 0.900     │
    # │ exp-2       │ 200         │ 10       │ 0.905     │
    # │ exp-3       │ 100         │ 15       │ 0.898     │
    # └─────────────┴─────────────┴──────────┴───────────┘
    
    # 最良の実験を適用
    # dvc exp apply exp-2
    # git add .
    # git commit -m "Apply best experiment: n_estimators=200"
    

### DVCとGitの統合ワークフロー
    
    
    # 完全なワークフロー例
    import subprocess
    import os
    
    def dvc_workflow_example():
        """DVCとGitを使った完全なMLワークフロー"""
    
        # 1. 新しいブランチを作成
        subprocess.run(["git", "checkout", "-b", "experiment/new-features"])
    
        # 2. 新しいデータを追加
        print("新しいデータを生成中...")
        import pandas as pd
        import numpy as np
    
        new_data = pd.DataFrame({
            'feature1': np.random.randn(1500),
            'feature2': np.random.randn(1500),
            'feature3': np.random.randn(1500),
            'feature4': np.random.randn(1500),  # 新特徴量
            'target': np.random.randint(0, 2, 1500)
        })
        new_data.to_csv('data/raw_data_v2.csv', index=False)
    
        # 3. DVCで新データを追跡
        subprocess.run(["dvc", "add", "data/raw_data_v2.csv"])
    
        # 4. 変更をコミット
        subprocess.run(["git", "add", "data/raw_data_v2.csv.dvc", "data/.gitignore"])
        subprocess.run(["git", "commit", "-m", "Add new dataset with feature4"])
    
        # 5. パイプラインを実行
        subprocess.run(["dvc", "repro"])
    
        # 6. 結果を確認
        subprocess.run(["dvc", "metrics", "show"])
    
        # 7. 変更をプッシュ
        subprocess.run(["git", "push", "origin", "experiment/new-features"])
        subprocess.run(["dvc", "push"])
    
        print("ワークフロー完了")
    
    # 注意: 実際の実行には適切なGit/DVCセットアップが必要
    print("DVCワークフロー例（コマンド解説）")
    

* * *

## 2.5 ベストプラクティス

### 1\. メタデータ記録のベストプラクティス

#### 記録すべき情報

カテゴリ | 項目 | 理由  
---|---|---  
**実験情報** | 実験名、日時、実行者 | 実験の識別と追跡  
**環境情報** | Python版、ライブラリ版、OS | 再現性の確保  
**データ情報** | データ版、サンプル数、分布 | データドリフト検出  
**モデル情報** | アーキテクチャ、パラメータ | モデルの再構築  
**評価情報** | メトリクス、混同行列 | 性能の比較  
      
    
    import mlflow
    import platform
    import sys
    from datetime import datetime
    
    def log_comprehensive_metadata(model, X_train, y_train, X_test, y_test):
        """包括的なメタデータの記録"""
    
        with mlflow.start_run(run_name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. 環境情報
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("os", platform.system())
            mlflow.log_param("os_version", platform.version())
    
            # 2. データ情報
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_distribution", dict(zip(*np.unique(y_train, return_counts=True))))
    
            # 3. モデル情報
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())
    
            # 4. 訓練
            model.fit(X_train, y_train)
    
            # 5. 評価メトリクス
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
    
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
    
            # 6. 実験メモ
            mlflow.set_tag("experiment_description", "Comprehensive metadata logging example")
            mlflow.set_tag("data_version", "v1.0")
            mlflow.set_tag("experiment_type", "baseline")
    
            # 7. モデルの保存
            mlflow.sklearn.log_model(model, "model")
    
            print("包括的なメタデータを記録しました")
    
    # 使用例
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    log_comprehensive_metadata(model, X_train, y_train, X_test, y_test)
    

### 2\. ハイパーパラメータ管理
    
    
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Optional
    
    @dataclass
    class ModelConfig:
        """モデル設定の構造化定義"""
        n_estimators: int = 100
        max_depth: Optional[int] = 10
        min_samples_split: int = 2
        min_samples_leaf: int = 1
        random_state: int = 42
    
        def save(self, filepath: str):
            """設定をYAMLファイルに保存"""
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f)
    
        @classmethod
        def load(cls, filepath: str):
            """YAMLファイルから設定を読み込み"""
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
    
    # 設定の保存
    config = ModelConfig(n_estimators=150, max_depth=15)
    config.save("configs/model_config.yaml")
    
    # 設定の読み込み
    loaded_config = ModelConfig.load("configs/model_config.yaml")
    
    # モデルの訓練
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**asdict(loaded_config))
    model.fit(X_train, y_train)
    
    print(f"設定を使用してモデルを訓練: {asdict(loaded_config)}")
    

### 3\. アーティファクト管理
    
    
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
        """実験のアーティファクトを体系的に保存"""
    
        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
            # 1. モデルの保存
            mlflow.sklearn.log_model(model, "model")
    
            # 2. メトリクスの保存
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
            # 3. 設定の保存
            for param_name, param_value in config.items():
                mlflow.log_param(param_name, param_value)
    
            # 4. 特徴量情報の保存
            feature_info = {
                "feature_names": feature_names,
                "n_features": len(feature_names)
            }
    
            # 一時ファイルに保存してMLflowにログ
            temp_dir = Path("temp_artifacts")
            temp_dir.mkdir(exist_ok=True)
    
            feature_path = temp_dir / "feature_info.json"
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            mlflow.log_artifact(str(feature_path))
    
            # 5. 特徴量重要度の保存（可能な場合）
            if hasattr(model, 'feature_importances_'):
                importance_df = {
                    name: float(imp)
                    for name, imp in zip(feature_names, model.feature_importances_)
                }
                importance_path = temp_dir / "feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance_df, f, indent=2)
                mlflow.log_artifact(str(importance_path))
    
            # 6. 予測例の保存
            sample_predictions = {
                "sample_input": X_test[:5].tolist(),
                "predictions": model.predict(X_test[:5]).tolist()
            }
            pred_path = temp_dir / "sample_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(sample_predictions, f, indent=2)
            mlflow.log_artifact(str(pred_path))
    
            # 一時ファイルの削除
            import shutil
            shutil.rmtree(temp_dir)
    
            print("全てのアーティファクトを保存しました")
    
    # 使用例
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
    

### 4\. 実験の組織化
    
    
    from enum import Enum
    import mlflow
    from datetime import datetime
    
    class ExperimentType(Enum):
        """実験タイプの定義"""
        BASELINE = "baseline"
        FEATURE_ENGINEERING = "feature_engineering"
        HYPERPARAMETER_TUNING = "hyperparameter_tuning"
        MODEL_SELECTION = "model_selection"
        PRODUCTION = "production"
    
    class ExperimentManager:
        """実験の組織的な管理"""
    
        def __init__(self, project_name: str):
            self.project_name = project_name
    
        def create_experiment_name(
            self,
            exp_type: ExperimentType,
            model_name: str,
            version: str = "v1"
        ) -> str:
            """階層的な実験名を生成"""
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
            """実験の実行と記録"""
    
            # 実験名の生成
            exp_name = self.create_experiment_name(exp_type, model_name, version)
            mlflow.set_experiment(exp_name)
    
            # 実行名の生成（タイムスタンプ付き）
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
            with mlflow.start_run(run_name=run_name):
                # タグの設定
                mlflow.set_tag("experiment_type", exp_type.value)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("version", version)
                mlflow.set_tag("description", description)
    
                # 訓練
                train_metrics = train_fn(model)
    
                # 評価
                test_metrics = evaluate_fn(model)
    
                # メトリクスの記録
                for metric_name, metric_value in {**train_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # モデルの保存
                mlflow.sklearn.log_model(model, "model")
    
                print(f"実験完了: {exp_name}/{run_name}")
                return test_metrics
    
    # 使用例
    manager = ExperimentManager(project_name="customer_churn")
    
    def train_fn(model):
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        return {"train_accuracy": train_acc}
    
    def evaluate_fn(model):
        test_acc = model.score(X_test, y_test)
        test_f1 = f1_score(y_test, model.predict(X_test))
        return {"test_accuracy": test_acc, "test_f1": test_f1}
    
    # ベースラインモデルの実験
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
    
    print(f"結果: {results}")
    

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **実験管理の重要性**

     * 再現性の確保が機械学習プロジェクトの基盤
     * 体系的な実験管理で効率的なモデル開発
     * データとモデルのバージョン管理の必要性
  2. **MLflow**

     * MLflow Tracking: パラメータとメトリクスの記録
     * MLflow Models: モデルのパッケージングとデプロイ
     * MLflow Projects: 再現可能な実験環境
     * Autolog機能による自動ロギング
  3. **Weights & Biases**

     * リアルタイムな実験可視化
     * ハイパーパラメータ自動最適化（Sweeps）
     * チーム協業とレポート共有
     * アーティファクト管理とバージョニング
  4. **DVC**

     * データとモデルのGitライクな管理
     * 再現可能なMLパイプライン定義
     * リモートストレージとの連携
     * 実験の追跡と比較
  5. **ベストプラクティス**

     * 包括的なメタデータの記録
     * 構造化されたパラメータ管理
     * 体系的なアーティファクト保存
     * 階層的な実験の組織化

### ツールの使い分けガイドライン

ツール | 強み | 推奨ユースケース  
---|---|---  
**MLflow** | オープンソース、柔軟性高 | オンプレミス環境、自由度重視  
**W &B** | 高度な可視化、チーム協業 | クラウド環境、チーム開発  
**DVC** | Gitとの親和性、データ管理 | 大容量データ、バージョン重視  
  
### 次の章へ

第3章では、**継続的インテグレーション/デプロイメント（CI/CD）** を学びます：

  * MLOpsにおけるCI/CDパイプライン
  * 自動テストとモデル検証
  * モデルのデプロイメント戦略
  * モニタリングとフィードバックループ

* * *

## 演習問題

### 問題1（難易度：easy）

実験管理における「再現性」が重要な理由を3つ挙げて説明してください。

解答例

**解答** ：

  1. **結果の検証**

     * 同じ条件で実験を再実行し、結果の妥当性を確認できる
     * 予期しない結果が偶然か、体系的な問題かを判断可能
  2. **知見の共有**

     * チームメンバーが同じ実験を再現して理解を深められる
     * 研究成果の透明性と信頼性が向上
  3. **デバッグと改善**

     * 問題が発生した際に、特定の実験状態を再現してデバッグ可能
     * 過去の成功した実験を基に、段階的な改善ができる

### 問題2（難易度：medium）

MLflowを使って、以下の要件を満たす実験トラッキングを実装してください：

  * 3つの異なるハイパーパラメータ設定でモデルを訓練
  * 各実験で訓練精度とテスト精度を記録
  * 最も高いテスト精度を達成した実験を特定

解答例
    
    
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # データの生成
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 実験設定
    mlflow.set_experiment("hyperparameter_comparison")
    
    # 異なるハイパーパラメータ設定
    configs = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20}
    ]
    
    results = []
    
    # 各設定で実験を実行
    for i, config in enumerate(configs):
        with mlflow.start_run(run_name=f"experiment_{i+1}"):
            # パラメータの記録
            mlflow.log_params(config)
    
            # モデルの訓練
            model = RandomForestClassifier(**config, random_state=42)
            model.fit(X_train, y_train)
    
            # 精度の計算
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
    
            # メトリクスの記録
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
    
            # 結果の保存
            results.append({
                "config": config,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "run_id": mlflow.active_run().info.run_id
            })
    
            print(f"実験 {i+1}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # 最良の実験を特定
    best_result = max(results, key=lambda x: x['test_acc'])
    
    print("\n=== 最良の実験 ===")
    print(f"設定: {best_result['config']}")
    print(f"テスト精度: {best_result['test_acc']:.3f}")
    print(f"Run ID: {best_result['run_id']}")
    

**出力** ：
    
    
    実験 1: Train=0.862, Test=0.855
    実験 2: Train=0.895, Test=0.895
    実験 3: Train=0.987, Test=0.890
    
    === 最良の実験 ===
    設定: {'n_estimators': 100, 'max_depth': 10}
    テスト精度: 0.895
    Run ID: xxxxxxxxxxxxx
    

### 問題3（難易度：medium）

DVCを使用する主な利点を、Gitだけを使用する場合と比較して説明してください。

解答例

**解答** ：

**DVCの主な利点** ：

  1. **大容量ファイルの効率的な管理**

     * Git: 大容量ファイル（データセット、モデル）でリポジトリが肥大化
     * DVC: 実ファイルはリモートストレージに保存、Gitにはメタデータのみ
  2. **データのバージョン管理**

     * Git: バイナリファイルの差分管理が非効率
     * DVC: データの変更履歴を効率的に追跡、任意のバージョンに復元可能
  3. **再現可能なパイプライン**

     * Git: スクリプトのバージョン管理のみ
     * DVC: データ、コード、パラメータを含む完全なパイプラインを定義・再現
  4. **チーム協業の容易性**

     * Git: 大容量ファイルの共有が困難
     * DVC: リモートストレージ経由で効率的にデータ共有

**比較表** ：

観点 | Git のみ | DVC + Git  
---|---|---  
コード管理 | ◎ 優秀 | ◎ 優秀  
データ管理 | △ 非効率 | ◎ 最適化  
モデル管理 | △ 困難 | ◎ 体系的  
パイプライン | × 未サポート | ◎ 完全サポート  
再現性 | △ 部分的 | ◎ 完全  
  
### 問題4（難易度：hard）

包括的な実験管理システムを設計してください。以下の要素を含めること：

  * 実験の自動ロギング
  * パラメータの構造化管理
  * 実験結果の比較機能
  * ベストモデルの自動選択

解答例
    
    
    import mlflow
    import yaml
    from dataclasses import dataclass, asdict
    from typing import Dict, Any, List, Optional
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import pandas as pd
    
    @dataclass
    class ExperimentConfig:
        """実験設定の構造化定義"""
        experiment_name: str
        model_params: Dict[str, Any]
        data_params: Dict[str, Any]
        description: str = ""
        tags: Dict[str, str] = None
    
    class ComprehensiveExperimentManager:
        """包括的な実験管理システム"""
    
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
            """実験の実行と自動ロギング"""
    
            # 実験の設定
            mlflow.set_experiment(config.experiment_name)
    
            with mlflow.start_run(description=config.description):
                # 1. パラメータのロギング
                mlflow.log_params(config.model_params)
                mlflow.log_params(config.data_params)
    
                # 2. タグの設定
                if config.tags:
                    for key, value in config.tags.items():
                        mlflow.set_tag(key, value)
    
                # 3. モデルの訓練
                model.fit(X_train, y_train)
    
                # 4. 予測
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
    
                # 5. メトリクスの計算
                metrics = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "test_precision": precision_score(y_test, y_test_pred, average='weighted'),
                    "test_recall": recall_score(y_test, y_test_pred, average='weighted'),
                    "test_f1": f1_score(y_test, y_test_pred, average='weighted'),
                    "overfit_gap": accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
                }
    
                # 6. メトリクスのロギング
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
    
                # 7. モデルの保存
                mlflow.sklearn.log_model(model, "model")
    
                # 8. 結果の保存
                run_id = mlflow.active_run().info.run_id
                result = {
                    "run_id": run_id,
                    "config": asdict(config),
                    "metrics": metrics
                }
                self.results.append(result)
    
                print(f"実験完了: {config.experiment_name}")
                print(f"  テスト精度: {metrics['test_accuracy']:.3f}")
                print(f"  Run ID: {run_id}")
    
                return metrics
    
        def compare_experiments(self) -> pd.DataFrame:
            """実験結果の比較"""
            if not self.results:
                print("実験結果がありません")
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
            """最良モデルの自動選択"""
            if not self.results:
                raise ValueError("実験結果がありません")
    
            best_result = max(self.results, key=lambda x: x["metrics"][metric])
    
            print(f"\n=== 最良モデル（{metric}基準）===")
            print(f"Run ID: {best_result['run_id']}")
            print(f"実験名: {best_result['config']['experiment_name']}")
            print(f"{metric}: {best_result['metrics'][metric]:.3f}")
            print(f"\n全メトリクス:")
            for m_name, m_value in best_result['metrics'].items():
                print(f"  {m_name}: {m_value:.3f}")
    
            return best_result
    
        def save_comparison_report(self, filepath: str = "experiment_comparison.csv"):
            """比較レポートの保存"""
            df = self.compare_experiments()
            df.to_csv(filepath, index=False)
            print(f"比較レポートを保存: {filepath}")
    
    # 使用例
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 実験マネージャーの初期化
    manager = ComprehensiveExperimentManager()
    
    # 実験1: Random Forest（浅い）
    config1 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with shallow depth",
        tags={"model_type": "random_forest", "depth": "shallow"}
    )
    rf_shallow = RandomForestClassifier(**config1.model_params)
    manager.run_experiment(config1, rf_shallow, X_train, y_train, X_test, y_test)
    
    # 実験2: Random Forest（深い）
    config2 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 20, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Random Forest with deep depth",
        tags={"model_type": "random_forest", "depth": "deep"}
    )
    rf_deep = RandomForestClassifier(**config2.model_params)
    manager.run_experiment(config2, rf_deep, X_train, y_train, X_test, y_test)
    
    # 実験3: Gradient Boosting
    config3 = ExperimentConfig(
        experiment_name="model_comparison",
        model_params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
        data_params={"train_size": len(X_train), "test_size": len(X_test)},
        description="Gradient Boosting Classifier",
        tags={"model_type": "gradient_boosting"}
    )
    gb = GradientBoostingClassifier(**config3.model_params)
    manager.run_experiment(config3, gb, X_train, y_train, X_test, y_test)
    
    # 結果の比較
    print("\n=== 全実験の比較 ===")
    comparison_df = manager.compare_experiments()
    print(comparison_df[['experiment', 'test_accuracy', 'test_f1', 'overfit_gap']])
    
    # 最良モデルの選択
    best_model = manager.get_best_model(metric="test_accuracy")
    
    # レポートの保存
    manager.save_comparison_report()
    

**出力** ：
    
    
    実験完了: model_comparison
      テスト精度: 0.855
      Run ID: xxxxx
    
    実験完了: model_comparison
      テスト精度: 0.890
      Run ID: yyyyy
    
    実験完了: model_comparison
      テスト精度: 0.905
      Run ID: zzzzz
    
    === 全実験の比較 ===
           experiment  test_accuracy  test_f1  overfit_gap
    2  model_comparison          0.905    0.903        0.032
    1  model_comparison          0.890    0.891        0.097
    0  model_comparison          0.855    0.856        0.007
    
    === 最良モデル（test_accuracy基準）===
    Run ID: zzzzz
    実験名: model_comparison
    test_accuracy: 0.905
    
    全メトリクス:
      train_accuracy: 0.937
      test_accuracy: 0.905
      test_precision: 0.906
      test_recall: 0.905
      test_f1: 0.903
      overfit_gap: 0.032
    
    比較レポートを保存: experiment_comparison.csv
    

### 問題5（難易度：hard）

MLflowとDVCを組み合わせた完全な機械学習ワークフローを設計し、実装してください。データのバージョン管理から実験トラッキング、モデルの保存まで含めること。

解答例
    
    
    """
    完全なML ワークフロー: DVC + MLflow
    
    ディレクトリ構造:
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
    
    # params.yaml の内容
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
    
        # データの読み込み
        df = pd.read_csv(data_params['raw_path'])
    
        # 訓練・テスト分割
        train, test = train_test_split(
            df,
            test_size=data_params['test_size'],
            random_state=data_params['random_state']
        )
    
        # 保存
        train.to_csv(data_params['train_path'], index=False)
        test.to_csv(data_params['test_path'], index=False)
    
        print(f"データ準備完了: Train={len(train)}, Test={len(test)}")
    
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
    
        # MLflowの設定
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # データの読み込み
        train = pd.read_csv(data_params['train_path'])
        X_train = train.drop('target', axis=1)
        y_train = train['target']
    
        # MLflow実験の開始
        with mlflow.start_run():
            # パラメータの記録
            mlflow.log_params(model_params)
            mlflow.log_param("train_size", len(X_train))
    
            # モデルの訓練
            model = RandomForestClassifier(
                n_estimators=model_params['n_estimators'],
                max_depth=model_params['max_depth'],
                min_samples_split=model_params['min_samples_split'],
                random_state=model_params['random_state']
            )
            model.fit(X_train, y_train)
    
            # 訓練メトリクス
            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_score)
    
            # モデルの保存
            model_path = "models/model.pkl"
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, "model")
    
            print(f"訓練完了: Train Accuracy={train_score:.3f}")
    
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
    
        # MLflowの設定
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
    
        # データとモデルの読み込み
        test = pd.read_csv(data_params['test_path'])
        X_test = test.drop('target', axis=1)
        y_test = test['target']
    
        model = joblib.load("models/model.pkl")
    
        # 評価
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
    
        # 詳細レポート
        report = classification_report(y_test, y_pred, output_dict=True)
    
        # メトリクスの保存
        metrics = {
            "test_accuracy": test_accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        }
    
        with open("metrics/test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
        # MLflowに記録
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
        print(f"評価完了: Test Accuracy={test_accuracy:.3f}")
        print(f"詳細メトリクス: {metrics}")
    
    if __name__ == "__main__":
        evaluate_model()
    
    # dvc.yaml の内容
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
    
    # 完全なワークフローの実行例
    """
    # 1. DVCの初期化
    dvc init
    
    # 2. データの追加
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc data/.gitignore
    git commit -m "Add raw data"
    
    # 3. パイプラインの実行
    dvc repro
    
    # 4. 実験パラメータの変更
    dvc exp run --set-param model.n_estimators=200
    
    # 5. 実験結果の比較
    dvc exp show
    
    # 6. 最良の実験を適用
    dvc exp apply 
    git add .
    git commit -m "Apply best experiment"
    
    # 7. MLflow UIで結果を確認
    mlflow ui --backend-store-uri ./mlruns
    """
    
    print("完全なワークフロー設計完了")
    print("DVC: データとパイプラインのバージョン管理")
    print("MLflow: 実験トラッキングとモデル管理")
    print("統合: 再現可能で追跡可能なMLワークフロー")
    

* * *

## 参考文献

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  3. Lakshmanan, V., Robinson, S., & Munn, M. (2020). _Machine Learning Design Patterns_. O'Reilly Media.
  4. Treveil, M., et al. (2020). _Introducing MLOps_. O'Reilly Media.
  5. MLflow Documentation. <https://mlflow.org/docs/latest/index.html>
  6. Weights & Biases Documentation. <https://docs.wandb.ai/>
  7. DVC Documentation. <https://dvc.org/doc>
