---
title: 第4章：モデル管理とレジストリ
chapter_title: 第4章：モデル管理とレジストリ
subtitle: MLOpsの基盤 - モデルのライフサイクルを管理する
reading_time: 30-35分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ モデル管理の課題と重要性を理解する
  * ✅ MLflow Model Registryを使用してモデルを登録・管理できる
  * ✅ モデルのバージョニングとステージ管理を実装できる
  * ✅ モデルメタデータとスキーマを適切に管理できる
  * ✅ モデルパッケージングと異なるフォーマットを理解する
  * ✅ モデルガバナンスとコンプライアンスを実現できる

* * *

## 4.1 モデル管理の課題

### モデル管理とは

**モデル管理（Model Management）** は、機械学習モデルのライフサイクル全体を体系的に管理するプロセスです。

> 「適切なモデル管理なしに、MLOpsの成功はありえない」- 本番環境でのモデル運用の基盤

### モデル管理の主要課題

#### 1\. モデルバージョニング

課題 | 説明 | 影響  
---|---|---  
**バージョンの追跡** | どのモデルがいつ作成されたか | 再現性の欠如  
**モデルの比較** | 複数バージョンの性能比較 | 最適モデル選択の困難  
**ロールバック** | 問題発生時の旧バージョン復帰 | ダウンタイム増加  
**依存関係管理** | モデルと学習コードの紐付け | 再学習の失敗  
  
#### 2\. メタデータ管理

モデルに関する重要情報の管理：

  * **学習メタデータ** : ハイパーパラメータ、学習データ情報
  * **性能メトリクス** : 精度、再現率、F1スコア
  * **入出力スキーマ** : 期待される入力形式と出力形式
  * **依存ライブラリ** : Python、scikit-learn、PyTorchのバージョン

#### 3\. モデルのライフサイクル
    
    
    ```mermaid
    graph LR
        A[開発] --> B[ステージング]
        B --> C[本番運用]
        C --> D[監視]
        D --> E{性能劣化?}
        E -->|はい| F[アーカイブ]
        E -->|いいえ| C
        F --> A
    
        style A fill:#fff3e0
        style B fill:#e3f2fd
        style C fill:#c8e6c9
        style D fill:#f3e5f5
        style E fill:#ffebee
        style F fill:#e0e0e0
    ```

#### 4\. ガバナンス要件

要件 | 目的 | 実装方法  
---|---|---  
**アクセス制御** | 権限管理 | RBAC、API認証  
**監査ログ** | 変更履歴追跡 | イベントログ、タイムスタンプ  
**コンプライアンス** | 規制遵守 | モデルカード、説明責任  
**承認プロセス** | 品質保証 | レビュー、テスト  
  
### モデル管理の実装課題
    
    
    import os
    import json
    from datetime import datetime
    import numpy as np
    
    # モデル管理における典型的な課題を示す例
    
    class ModelManagementChallenges:
        """モデル管理の課題を示すクラス"""
    
        def __init__(self):
            self.models = {}
            self.challenges = []
    
        def demonstrate_version_chaos(self):
            """バージョン管理の混乱を示す"""
            # 課題1: 一貫性のないバージョン命名
            model_files = [
                "model.pkl",
                "model_v2.pkl",
                "model_final.pkl",
                "model_final_v2.pkl",
                "model_REALLY_final.pkl",
                "model_2024_01_15.pkl"
            ]
    
            print("=== 課題1: バージョン管理の混乱 ===")
            print("非体系的なファイル名:")
            for f in model_files:
                print(f"  - {f}")
            print("\n問題点:")
            print("  - どれが最新か不明")
            print("  - 作成順序が不明")
            print("  - バージョン間の違いが不明")
    
            return model_files
    
        def demonstrate_metadata_loss(self):
            """メタデータの欠落を示す"""
            print("\n=== 課題2: メタデータの欠落 ===")
    
            # モデルファイルだけが保存されている状態
            model_info = {
                "filename": "model.pkl",
                "size_mb": 45.2
            }
    
            print("保存されている情報:")
            print(json.dumps(model_info, indent=2))
    
            print("\n欠落している重要情報:")
            missing_metadata = [
                "学習に使用したデータセット",
                "ハイパーパラメータ",
                "性能メトリクス",
                "入出力スキーマ",
                "依存ライブラリのバージョン",
                "作成者と作成日時",
                "学習環境（GPU、CPU仕様）"
            ]
            for item in missing_metadata:
                print(f"  ❌ {item}")
    
        def demonstrate_deployment_risk(self):
            """デプロイメントリスクを示す"""
            print("\n=== 課題3: デプロイメントリスク ===")
    
            scenarios = [
                {
                    "scenario": "間違ったモデルのデプロイ",
                    "cause": "バージョン管理の欠如",
                    "impact": "性能劣化、ビジネス損失"
                },
                {
                    "scenario": "ロールバック不可",
                    "cause": "旧バージョンの保存不足",
                    "impact": "長時間のダウンタイム"
                },
                {
                    "scenario": "依存性の不整合",
                    "cause": "環境情報の未記録",
                    "impact": "実行時エラー"
                }
            ]
    
            for s in scenarios:
                print(f"\nシナリオ: {s['scenario']}")
                print(f"  原因: {s['cause']}")
                print(f"  影響: {s['impact']}")
    
        def demonstrate_governance_gaps(self):
            """ガバナンスギャップを示す"""
            print("\n=== 課題4: ガバナンスの欠如 ===")
    
            governance_issues = [
                "誰がモデルを本番環境にデプロイしたか不明",
                "モデルの変更が承認プロセスなしで実施",
                "監査ログが存在しない",
                "アクセス制御が未実装",
                "コンプライアンス要件が未対応"
            ]
    
            print("一般的なガバナンス問題:")
            for issue in governance_issues:
                print(f"  ⚠️  {issue}")
    
    # 実行例
    challenges = ModelManagementChallenges()
    challenges.demonstrate_version_chaos()
    challenges.demonstrate_metadata_loss()
    challenges.demonstrate_deployment_risk()
    challenges.demonstrate_governance_gaps()
    
    print("\n" + "="*60)
    print("結論: 体系的なモデル管理システムが必要")
    print("="*60)
    

**出力** ：
    
    
    === 課題1: バージョン管理の混乱 ===
    非体系的なファイル名:
      - model.pkl
      - model_v2.pkl
      - model_final.pkl
      - model_final_v2.pkl
      - model_REALLY_final.pkl
      - model_2024_01_15.pkl
    
    問題点:
      - どれが最新か不明
      - 作成順序が不明
      - バージョン間の違いが不明
    
    === 課題2: メタデータの欠落 ===
    保存されている情報:
    {
      "filename": "model.pkl",
      "size_mb": 45.2
    }
    
    欠落している重要情報:
      ❌ 学習に使用したデータセット
      ❌ ハイパーパラメータ
      ❌ 性能メトリクス
      ❌ 入出力スキーマ
      ❌ 依存ライブラリのバージョン
      ❌ 作成者と作成日時
      ❌ 学習環境（GPU、CPU仕様）
    
    === 課題3: デプロイメントリスク ===
    
    シナリオ: 間違ったモデルのデプロイ
      原因: バージョン管理の欠如
      影響: 性能劣化、ビジネス損失
    
    シナリオ: ロールバック不可
      原因: 旧バージョンの保存不足
      影響: 長時間のダウンタイム
    
    シナリオ: 依存性の不整合
      原因: 環境情報の未記録
      影響: 実行時エラー
    
    === 課題4: ガバナンスの欠如 ===
    一般的なガバナンス問題:
      ⚠️  誰がモデルを本番環境にデプロイしたか不明
      ⚠️  モデルの変更が承認プロセスなしで実施
      ⚠️  監査ログが存在しない
      ⚠️  アクセス制御が未実装
      ⚠️  コンプライアンス要件が未対応
    
    ============================================================
    結論: 体系的なモデル管理システムが必要
    ============================================================
    

* * *

## 4.2 モデルレジストリ

### MLflow Model Registryとは

**MLflow Model Registry** は、機械学習モデルのライフサイクル全体を管理する中央リポジトリです。

### Model Registryの主要機能

機能 | 説明 | 利点  
---|---|---  
**モデル登録** | モデルを名前付きで登録 | 統一的な管理  
**バージョン管理** | 自動的なバージョン番号付与 | 履歴追跡  
**ステージ管理** | Staging/Production/Archive | 環境の明確化  
**メタデータ保存** | 説明、タグ、注釈 | 検索性向上  
**アクセス制御** | 権限ベースの管理 | セキュリティ  
  
### MLflow Model Registryのセットアップ
    
    
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    # MLflowトラッキングサーバーの設定
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model-registry-demo")
    
    # クライアント作成
    client = MlflowClient()
    
    print("=== MLflow Model Registry セットアップ ===")
    print(f"トラッキングURI: {mlflow.get_tracking_uri()}")
    print(f"実験名: {mlflow.get_experiment_by_name('model-registry-demo').name}")
    
    # データ準備
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
    
    print(f"\nデータセット: {X_train.shape[0]} 訓練, {X_test.shape[0]} テスト")
    

### モデルのバージョニング
    
    
    def train_and_register_model(model_name, n_estimators, max_depth):
        """モデルを学習してModel Registryに登録"""
    
        with mlflow.start_run(run_name=f"rf_v{n_estimators}_{max_depth}") as run:
            # モデル学習
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
    
            # 予測と評価
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
    
            # メトリクスのロギング
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1
            })
    
            # モデルのロギング
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
    
            print(f"\n✓ モデル学習完了: {model_name}")
            print(f"  Run ID: {run.info.run_id}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
    
            return run.info.run_id, accuracy, f1
    
    # 複数のモデルバージョンを作成
    model_name = "credit-risk-classifier"
    
    print("\n=== モデルバージョンの作成 ===")
    
    # バージョン1: 小さなモデル
    run_id_v1, acc_v1, f1_v1 = train_and_register_model(
        model_name, n_estimators=10, max_depth=5
    )
    
    # バージョン2: 中規模モデル
    run_id_v2, acc_v2, f1_v2 = train_and_register_model(
        model_name, n_estimators=50, max_depth=10
    )
    
    # バージョン3: 大規模モデル
    run_id_v3, acc_v3, f1_v3 = train_and_register_model(
        model_name, n_estimators=100, max_depth=15
    )
    
    # 登録されたモデルバージョンの確認
    print(f"\n=== {model_name} のバージョン一覧 ===")
    for mv in client.search_model_versions(f"name='{model_name}'"):
        print(f"\nバージョン: {mv.version}")
        print(f"  Run ID: {mv.run_id}")
        print(f"  ステージ: {mv.current_stage}")
        print(f"  作成日時: {mv.creation_timestamp}")
    

### ステージ遷移（Stage Transitions）
    
    
    def transition_model_stage(model_name, version, stage, description=""):
        """モデルを指定のステージに遷移"""
    
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False
        )
    
        # 説明を追加
        if description:
            client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
    
        print(f"✓ {model_name} v{version} を {stage} に遷移")
    
    print("\n=== ステージ管理 ===")
    
    # バージョン1: Staging（開発環境でテスト中）
    transition_model_stage(
        model_name,
        version=1,
        stage="Staging",
        description="初期モデル。軽量だが精度はやや低い。"
    )
    
    # バージョン2: Production（本番環境）
    transition_model_stage(
        model_name,
        version=2,
        stage="Production",
        description="現在の本番モデル。バランスの取れた性能。"
    )
    
    # バージョン3: Staging（評価中）
    transition_model_stage(
        model_name,
        version=3,
        stage="Staging",
        description="最新モデル。高精度だが推論時間が長い可能性。"
    )
    
    # ステージ別のモデル取得
    print("\n=== ステージ別モデル ===")
    
    def get_models_by_stage(model_name, stage):
        """特定ステージのモデルを取得"""
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions
    
    # Production環境のモデル
    prod_models = get_models_by_stage(model_name, "Production")
    for model in prod_models:
        print(f"\nProduction: {model_name} v{model.version}")
        print(f"  説明: {model.description}")
    
    # Staging環境のモデル
    staging_models = get_models_by_stage(model_name, "Staging")
    print(f"\nStaging環境のモデル数: {len(staging_models)}")
    for model in staging_models:
        print(f"  - v{model.version}: {model.description}")
    

### 完全なモデルレジストリの例
    
    
    class ModelRegistry:
        """MLflow Model Registryの包括的な管理クラス"""
    
        def __init__(self, tracking_uri="sqlite:///mlflow.db"):
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
    
        def register_model(self, model, model_name, run_id,
                          params, metrics, tags=None):
            """モデルを登録"""
            # モデルのロギング
            with mlflow.start_run(run_id=run_id):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
    
                if tags:
                    mlflow.set_tags(tags)
    
            # 最新バージョンを取得
            versions = self.client.search_model_versions(
                f"name='{model_name}'"
            )
            latest_version = max([int(v.version) for v in versions])
    
            print(f"✓ モデル登録完了: {model_name} v{latest_version}")
            return latest_version
    
        def promote_to_production(self, model_name, version,
                                 archive_old=True):
            """モデルを本番環境に昇格"""
    
            # 既存のProductionモデルをアーカイブ
            if archive_old:
                prod_models = self.client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                for model in prod_models:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model.version,
                        stage="Archived"
                    )
                    print(f"  古いバージョン v{model.version} をアーカイブ")
    
            # 新しいバージョンをProductionに
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
    
            print(f"✓ {model_name} v{version} を本番環境に昇格")
    
        def compare_versions(self, model_name, version1, version2):
            """2つのバージョンを比較"""
    
            print(f"\n=== {model_name}: v{version1} vs v{version2} ===")
    
            for version in [version1, version2]:
                mv = self.client.get_model_version(model_name, version)
                run = self.client.get_run(mv.run_id)
    
                print(f"\nバージョン {version}:")
                print(f"  ステージ: {mv.current_stage}")
                print(f"  パラメータ: {run.data.params}")
                print(f"  メトリクス: {run.data.metrics}")
    
        def get_production_model(self, model_name):
            """本番環境のモデルを取得"""
            versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
    
            if not versions:
                raise ValueError(f"本番環境に {model_name} が存在しません")
    
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
    
            print(f"✓ 本番モデル読み込み: {model_name} v{versions[0].version}")
            return model
    
        def add_model_alias(self, model_name, version, alias):
            """モデルにエイリアスを追加"""
            self.client.set_registered_model_alias(
                model_name, alias, version
            )
            print(f"✓ エイリアス '{alias}' を v{version} に設定")
    
        def delete_model_version(self, model_name, version):
            """特定バージョンを削除"""
            self.client.delete_model_version(model_name, version)
            print(f"✓ {model_name} v{version} を削除")
    
    # 使用例
    registry = ModelRegistry()
    
    print("\n=== Model Registryの高度な使用例 ===")
    
    # バージョン比較
    registry.compare_versions(model_name, version1=1, version2=3)
    
    # 本番環境へのプロモーション
    registry.promote_to_production(model_name, version=3, archive_old=True)
    
    # 本番モデルの取得と推論
    prod_model = registry.get_production_model(model_name)
    sample_prediction = prod_model.predict(X_test[:5])
    print(f"\nサンプル予測: {sample_prediction}")
    

* * *

## 4.3 モデルメタデータ管理

### モデル署名（Model Signature）

**モデル署名** は、モデルの入出力スキーマを定義し、型安全性を確保します。
    
    
    import mlflow
    from mlflow.models.signature import infer_signature, ModelSignature
    from mlflow.types.schema import Schema, ColSpec
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    
    # データ準備
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)])
    y_train_series = pd.Series(y_train, name="target")
    
    # モデル学習
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_df, y_train_series)
    
    # 予測
    predictions = model.predict(X_train_df[:5])
    predict_proba = model.predict_proba(X_train_df[:5])
    
    print("=== モデル署名の作成 ===\n")
    
    # 方法1: 自動推論
    signature = infer_signature(X_train_df, predictions)
    print("自動推論された署名:")
    print(signature)
    
    # 方法2: 明示的に定義
    from mlflow.types import Schema, ColSpec
    
    input_schema = Schema([
        ColSpec("double", f"feature_{i}") for i in range(20)
    ])
    
    output_schema = Schema([ColSpec("long")])
    
    explicit_signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema
    )
    
    print("\n明示的に定義した署名:")
    print(explicit_signature)
    
    # 署名付きでモデルを保存
    with mlflow.start_run(run_name="model-with-signature"):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_df[:5]
        )
    
        print("\n✓ 署名付きモデルを保存")
    
    # 署名の検証
    print("\n=== 署名の検証 ===")
    
    # 正しい入力
    correct_input = pd.DataFrame(
        np.random.randn(3, 20),
        columns=[f"feature_{i}" for i in range(20)]
    )
    print("✓ 正しい入力形式: OK")
    
    # 間違った入力（列数が異なる）
    try:
        wrong_input = pd.DataFrame(
            np.random.randn(3, 15),  # 列数が少ない
            columns=[f"feature_{i}" for i in range(15)]
        )
        # MLflowが署名をチェック
        print("❌ 間違った入力形式: エラーが検出されるべき")
    except Exception as e:
        print(f"✓ エラー検出: {type(e).__name__}")
    

### 入出力スキーマ管理
    
    
    from mlflow.types.schema import Schema, ColSpec, DataType
    from mlflow.models.signature import ModelSignature
    import json
    
    class SchemaManager:
        """モデルスキーマの管理クラス"""
    
        @staticmethod
        def create_detailed_schema(feature_info):
            """詳細なスキーマを作成"""
    
            col_specs = []
            for name, dtype, description in feature_info:
                col_spec = ColSpec(
                    type=dtype,
                    name=name
                )
                col_specs.append(col_spec)
    
            return Schema(col_specs)
    
        @staticmethod
        def validate_input(data, schema):
            """入力データがスキーマに適合するか検証"""
    
            errors = []
    
            # 列数チェック
            if len(data.columns) != len(schema.inputs):
                errors.append(
                    f"列数が不一致: 期待={len(schema.inputs)}, "
                    f"実際={len(data.columns)}"
                )
    
            # 列名チェック
            expected_cols = [col.name for col in schema.inputs]
            actual_cols = list(data.columns)
    
            if expected_cols != actual_cols:
                errors.append(f"列名が不一致: {set(expected_cols) ^ set(actual_cols)}")
    
            # データ型チェック
            for col_spec in schema.inputs:
                if col_spec.name in data.columns:
                    actual_dtype = data[col_spec.name].dtype
                    # 簡易的な型チェック
                    if col_spec.type == DataType.double:
                        if not np.issubdtype(actual_dtype, np.floating):
                            errors.append(
                                f"{col_spec.name}: 型不一致 "
                                f"(期待=float, 実際={actual_dtype})"
                            )
    
            return len(errors) == 0, errors
    
        @staticmethod
        def export_schema_json(signature, filepath):
            """スキーマをJSON形式でエクスポート"""
    
            schema_dict = {
                "inputs": [
                    {
                        "name": col.name,
                        "type": str(col.type)
                    }
                    for col in signature.inputs.inputs
                ],
                "outputs": [
                    {
                        "name": col.name if hasattr(col, 'name') else "prediction",
                        "type": str(col.type)
                    }
                    for col in signature.outputs.inputs
                ]
            }
    
            with open(filepath, 'w') as f:
                json.dump(schema_dict, f, indent=2)
    
            print(f"✓ スキーマをエクスポート: {filepath}")
    
    # 使用例
    print("\n=== 詳細なスキーマ管理 ===")
    
    # 特徴量情報の定義
    feature_info = [
        ("age", DataType.long, "年齢"),
        ("income", DataType.double, "年収"),
        ("credit_score", DataType.double, "信用スコア"),
        ("loan_amount", DataType.double, "ローン金額"),
    ]
    
    # スキーマ作成
    manager = SchemaManager()
    input_schema = manager.create_detailed_schema(feature_info)
    
    output_schema = Schema([
        ColSpec(DataType.long, "prediction"),
        ColSpec(DataType.double, "probability")
    ])
    
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    print("作成されたスキーマ:")
    print(signature)
    
    # スキーマのエクスポート
    manager.export_schema_json(signature, "model_schema.json")
    
    # 検証例
    test_data_valid = pd.DataFrame({
        "age": [35, 42],
        "income": [50000.0, 75000.0],
        "credit_score": [720.0, 680.0],
        "loan_amount": [25000.0, 40000.0]
    })
    
    test_data_invalid = pd.DataFrame({
        "age": [35, 42],
        "income": [50000.0, 75000.0],
        "credit_score": [720.0, 680.0]
        # loan_amount が欠落
    })
    
    print("\n=== 入力検証 ===")
    
    valid, errors = manager.validate_input(test_data_valid, signature)
    print(f"有効な入力: {valid}")
    
    valid, errors = manager.validate_input(test_data_invalid, signature)
    print(f"無効な入力: {valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    

### 依存関係管理
    
    
    import mlflow
    from mlflow.models import make_metric
    import cloudpickle
    import sys
    
    def log_model_with_dependencies(model, model_name, conda_env=None,
                                    pip_requirements=None):
        """依存関係を含めてモデルを保存"""
    
        with mlflow.start_run(run_name="model-with-deps"):
    
            # 現在の環境情報をロギング
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("mlflow_version", mlflow.__version__)
    
            # Conda環境の定義
            if conda_env is None:
                conda_env = {
                    "name": "model_env",
                    "channels": ["conda-forge"],
                    "dependencies": [
                        f"python={sys.version_info.major}.{sys.version_info.minor}",
                        "pip",
                        {
                            "pip": [
                                f"mlflow=={mlflow.__version__}",
                                "scikit-learn==1.3.0",
                                "pandas==2.0.3",
                                "numpy==1.24.3"
                            ]
                        }
                    ]
                }
    
            # pip requirements
            if pip_requirements is None:
                pip_requirements = [
                    "scikit-learn==1.3.0",
                    "pandas==2.0.3",
                    "numpy==1.24.3"
                ]
    
            # モデルの保存
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                conda_env=conda_env,
                pip_requirements=pip_requirements,
                registered_model_name=model_name
            )
    
            print(f"✓ モデルと依存関係を保存: {model_name}")
            print(f"\nConda環境:")
            print(f"  Python: {conda_env['dependencies'][0]}")
            print(f"  パッケージ: {len(pip_requirements)} 個")
    
            return mlflow.active_run().info.run_id
    
    # 使用例
    print("=== 依存関係を含むモデル保存 ===\n")
    
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    run_id = log_model_with_dependencies(
        model=model,
        model_name="credit-model-with-deps"
    )
    
    print(f"\nRun ID: {run_id}")
    

### パフォーマンスメトリクス管理
    
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    import json
    from datetime import datetime
    
    class PerformanceMetricsManager:
        """モデル性能メトリクスの包括的管理"""
    
        def __init__(self):
            self.metrics_history = []
    
        def compute_classification_metrics(self, y_true, y_pred, y_prob=None):
            """分類問題の包括的なメトリクスを計算"""
    
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='binary'),
                "recall": recall_score(y_true, y_pred, average='binary'),
                "f1_score": f1_score(y_true, y_pred, average='binary')
            }
    
            if y_prob is not None:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    
            # 混同行列
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
    
            # ビジネスメトリクス
            metrics["false_positive_rate"] = cm[0, 1] / (cm[0, 0] + cm[0, 1])
            metrics["false_negative_rate"] = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    
            return metrics
    
        def log_metrics_to_mlflow(self, metrics, model_version=None):
            """メトリクスをMLflowにロギング"""
    
            # 混同行列は別途保存
            cm = metrics.pop("confusion_matrix", None)
    
            # スカラーメトリクスをログ
            mlflow.log_metrics(metrics)
    
            # 混同行列をJSON形式で保存
            if cm:
                mlflow.log_dict(cm, "confusion_matrix.json")
    
            # タイムスタンプ付きで履歴に追加
            metrics_with_time = {
                "timestamp": datetime.now().isoformat(),
                "model_version": model_version,
                **metrics,
                "confusion_matrix": cm
            }
            self.metrics_history.append(metrics_with_time)
    
            print("✓ メトリクスをMLflowにロギング")
    
        def compare_model_performance(self, metrics1, metrics2,
                                      model1_name="Model 1",
                                      model2_name="Model 2"):
            """2つのモデルの性能を比較"""
    
            print(f"\n=== {model1_name} vs {model2_name} ===\n")
    
            comparison = {}
            for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                if metric in metrics1 and metric in metrics2:
                    val1 = metrics1[metric]
                    val2 = metrics2[metric]
                    diff = val2 - val1
                    pct_change = (diff / val1) * 100 if val1 > 0 else 0
    
                    comparison[metric] = {
                        model1_name: val1,
                        model2_name: val2,
                        "difference": diff,
                        "pct_change": pct_change
                    }
    
                    print(f"{metric}:")
                    print(f"  {model1_name}: {val1:.4f}")
                    print(f"  {model2_name}: {val2:.4f}")
                    print(f"  差分: {diff:+.4f} ({pct_change:+.2f}%)")
                    print()
    
            return comparison
    
        def export_metrics_report(self, filepath="metrics_report.json"):
            """メトリクス履歴をレポートとしてエクスポート"""
    
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
            print(f"✓ メトリクスレポートをエクスポート: {filepath}")
    
    # 使用例
    print("\n=== パフォーマンスメトリクス管理 ===")
    
    metrics_manager = PerformanceMetricsManager()
    
    # モデル1の評価
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    y_prob1 = model1.predict_proba(X_test)[:, 1]
    
    metrics1 = metrics_manager.compute_classification_metrics(
        y_test, y_pred1, y_prob1
    )
    
    # モデル2の評価
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    y_prob2 = model2.predict_proba(X_test)[:, 1]
    
    metrics2 = metrics_manager.compute_classification_metrics(
        y_test, y_pred2, y_prob2
    )
    
    # 比較
    comparison = metrics_manager.compare_model_performance(
        metrics1, metrics2,
        model1_name="RF-10",
        model2_name="RF-100"
    )
    
    # MLflowにロギング
    with mlflow.start_run(run_name="rf-10"):
        metrics_manager.log_metrics_to_mlflow(metrics1.copy(), model_version=1)
    
    with mlflow.start_run(run_name="rf-100"):
        metrics_manager.log_metrics_to_mlflow(metrics2.copy(), model_version=2)
    
    # レポートのエクスポート
    metrics_manager.export_metrics_report()
    

* * *

## 4.4 モデルパッケージング

### ONNXフォーマット

**ONNX（Open Neural Network Exchange）** は、異なるフレームワーク間でモデルを交換可能にするオープンフォーマットです。
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt
    import mlflow
    
    print("=== ONNXフォーマットへの変換 ===\n")
    
    # モデル学習
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # ONNX形式に変換
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # ONNX モデルの保存
    onnx_path = "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✓ ONNXモデルを保存: {onnx_path}")
    
    # ONNX Runtimeで推論
    print("\n=== ONNX Runtimeでの推論 ===")
    
    sess = rt.InferenceSession(onnx_path)
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # 推論実行
    X_test_float = X_test.astype(np.float32)
    onnx_pred = sess.run([output_name], {input_name: X_test_float})[0]
    
    # scikit-learnの予測と比較
    sklearn_pred = model.predict(X_test)
    
    print(f"ONNX予測: {onnx_pred[:5]}")
    print(f"sklearn予測: {sklearn_pred[:5]}")
    print(f"一致率: {np.mean(onnx_pred == sklearn_pred):.2%}")
    
    # MLflowに保存
    with mlflow.start_run(run_name="onnx-model"):
        mlflow.onnx.log_model(onnx_model, "onnx_model")
        mlflow.log_metric("accuracy", accuracy_score(y_test, onnx_pred))
        print("\n✓ ONNXモデルをMLflowに保存")
    
    print("\n利点:")
    print("  - フレームワーク非依存")
    print("  - 高速推論")
    print("  - エッジデバイス対応")
    print("  - クロスプラットフォーム")
    

### BentoML

**BentoML** は、MLモデルを本番環境用のAPIサービスとしてパッケージ化するフレームワークです。
    
    
    import bentoml
    from bentoml.io import NumpyNdarray, JSON
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    
    print("=== BentoMLでのモデルパッケージング ===\n")
    
    # モデル学習
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # BentoMLに保存
    saved_model = bentoml.sklearn.save_model(
        "credit_risk_model",
        model,
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        labels={
            "owner": "data-science-team",
            "stage": "production"
        },
        metadata={
            "accuracy": float(accuracy_score(y_test, model.predict(X_test))),
            "model_type": "GradientBoosting",
            "features": X_train.shape[1]
        }
    )
    
    print(f"✓ モデルを保存: {saved_model.tag}")
    print(f"  保存先: {saved_model.path}")
    
    # サービス定義の作成
    service_code = '''
    import bentoml
    import numpy as np
    from bentoml.io import NumpyNdarray, JSON
    
    # モデルの参照を取得
    credit_model_runner = bentoml.sklearn.get("credit_risk_model:latest").to_runner()
    
    # サービスの定義
    svc = bentoml.Service("credit_risk_classifier", runners=[credit_model_runner])
    
    @svc.api(input=NumpyNdarray(), output=JSON())
    async def classify(input_data: np.ndarray) -> dict:
        """信用リスク分類API"""
    
        # 予測実行
        prediction = await credit_model_runner.predict.async_run(input_data)
        probabilities = await credit_model_runner.predict_proba.async_run(input_data)
    
        return {
            "predictions": prediction.tolist(),
            "probabilities": probabilities.tolist()
        }
    '''
    
    # service.pyとして保存
    with open("service.py", "w") as f:
        f.write(service_code)
    
    print("\n✓ サービス定義を作成: service.py")
    
    # Bentoの作成設定
    bentofile_content = '''
    service: "service:svc"
    labels:
      owner: data-science-team
      project: credit-risk
    include:
      - "service.py"
    python:
      packages:
        - scikit-learn==1.3.0
        - pandas==2.0.3
        - numpy==1.24.3
    '''
    
    with open("bentofile.yaml", "w") as f:
        f.write(bentofile_content)
    
    print("✓ Bento設定を作成: bentofile.yaml")
    
    print("\n次のステップ:")
    print("  1. bentoml build  # Bentoをビルド")
    print("  2. bentoml containerize credit_risk_classifier:latest  # Dockerイメージ作成")
    print("  3. bentoml serve service:svc  # ローカルでサービス起動")
    
    print("\nBentoMLの利点:")
    print("  - 簡単なAPI化")
    print("  - 自動スケーリング")
    print("  - バッチ処理対応")
    print("  - モニタリング統合")
    print("  - Dockerコンテナ化")
    

### TorchScript

**TorchScript** は、PyTorchモデルを最適化・シリアライズする形式です。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    
    print("=== TorchScriptでのモデルパッケージング ===\n")
    
    # 簡単なニューラルネットワーク定義
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.2)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # モデルのインスタンス化
    input_size = X_train.shape[1]
    hidden_size = 64
    num_classes = 2
    
    model = SimpleClassifier(input_size, hidden_size, num_classes)
    
    # 簡易学習
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 学習ループ（簡略版）
    model.train()
    for epoch in range(5):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    print("✓ モデル学習完了")
    
    # TorchScriptに変換（Tracing方式）
    model.eval()
    example_input = torch.randn(1, input_size)
    
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存
    traced_model.save("model_traced.pt")
    print("\n✓ TorchScript (traced) を保存: model_traced.pt")
    
    # TorchScriptに変換（Scripting方式）
    scripted_model = torch.jit.script(model)
    scripted_model.save("model_scripted.pt")
    print("✓ TorchScript (scripted) を保存: model_scripted.pt")
    
    # ロードと推論
    print("\n=== TorchScriptモデルのロードと推論 ===")
    
    loaded_model = torch.jit.load("model_traced.pt")
    loaded_model.eval()
    
    # テストデータで推論
    X_test_tensor = torch.FloatTensor(X_test[:5])
    with torch.no_grad():
        outputs = loaded_model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)
    
    print(f"予測結果: {predictions.numpy()}")
    print(f"実際のラベル: {y_test[:5]}")
    
    print("\nTorchScriptの利点:")
    print("  - Python依存なしで実行可能")
    print("  - C++から利用可能")
    print("  - モバイル/エッジデバイス対応")
    print("  - 最適化による高速化")
    print("  - プロダクション環境に最適")
    

### モデルシリアライゼーションの比較
    
    
    import pickle
    import joblib
    import json
    import os
    from datetime import datetime
    import time
    
    class ModelSerializationComparison:
        """異なるシリアライゼーション方法を比較"""
    
        def __init__(self, model):
            self.model = model
            self.results = {}
    
        def compare_formats(self, X_test_sample):
            """各フォーマットを比較"""
    
            print("=== モデルシリアライゼーション比較 ===\n")
    
            # 1. Pickle
            self._test_pickle(X_test_sample)
    
            # 2. Joblib
            self._test_joblib(X_test_sample)
    
            # 3. MLflow
            self._test_mlflow(X_test_sample)
    
            # 4. ONNX
            self._test_onnx(X_test_sample)
    
            # 結果の表示
            self._display_results()
    
        def _test_pickle(self, X_test):
            """Pickle形式のテスト"""
            filepath = "model.pkl"
    
            # 保存
            start = time.time()
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            save_time = time.time() - start
    
            # ロード
            start = time.time()
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            load_time = time.time() - start
    
            # 推論
            start = time.time()
            predictions = loaded_model.predict(X_test)
            inference_time = time.time() - start
    
            self.results['Pickle'] = {
                'size_mb': os.path.getsize(filepath) / 1024 / 1024,
                'save_time': save_time,
                'load_time': load_time,
                'inference_time': inference_time
            }
    
            os.remove(filepath)
    
        def _test_joblib(self, X_test):
            """Joblib形式のテスト"""
            filepath = "model.joblib"
    
            start = time.time()
            joblib.dump(self.model, filepath)
            save_time = time.time() - start
    
            start = time.time()
            loaded_model = joblib.load(filepath)
            load_time = time.time() - start
    
            start = time.time()
            predictions = loaded_model.predict(X_test)
            inference_time = time.time() - start
    
            self.results['Joblib'] = {
                'size_mb': os.path.getsize(filepath) / 1024 / 1024,
                'save_time': save_time,
                'load_time': load_time,
                'inference_time': inference_time
            }
    
            os.remove(filepath)
    
        def _test_mlflow(self, X_test):
            """MLflow形式のテスト"""
            model_path = "mlflow_model"
    
            start = time.time()
            mlflow.sklearn.save_model(self.model, model_path)
            save_time = time.time() - start
    
            start = time.time()
            loaded_model = mlflow.sklearn.load_model(model_path)
            load_time = time.time() - start
    
            start = time.time()
            predictions = loaded_model.predict(X_test)
            inference_time = time.time() - start
    
            # ディレクトリサイズを計算
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            )
    
            self.results['MLflow'] = {
                'size_mb': total_size / 1024 / 1024,
                'save_time': save_time,
                'load_time': load_time,
                'inference_time': inference_time
            }
    
            # クリーンアップ
            import shutil
            shutil.rmtree(model_path)
    
        def _test_onnx(self, X_test):
            """ONNX形式のテスト"""
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                import onnxruntime as rt
    
                filepath = "model.onnx"
                initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]
    
                start = time.time()
                onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                with open(filepath, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                save_time = time.time() - start
    
                start = time.time()
                sess = rt.InferenceSession(filepath)
                load_time = time.time() - start
    
                input_name = sess.get_inputs()[0].name
                output_name = sess.get_outputs()[0].name
    
                start = time.time()
                predictions = sess.run([output_name], {input_name: X_test.astype(np.float32)})[0]
                inference_time = time.time() - start
    
                self.results['ONNX'] = {
                    'size_mb': os.path.getsize(filepath) / 1024 / 1024,
                    'save_time': save_time,
                    'load_time': load_time,
                    'inference_time': inference_time
                }
    
                os.remove(filepath)
            except ImportError:
                print("⚠️  ONNX: ライブラリが未インストール")
    
        def _display_results(self):
            """結果を表形式で表示"""
    
            print("\n" + "="*70)
            print(f"{'フォーマット':<15} {'サイズ(MB)':<12} {'保存時間(s)':<12} {'ロード時間(s)':<12} {'推論時間(s)':<12}")
            print("="*70)
    
            for format_name, metrics in self.results.items():
                print(f"{format_name:<15} "
                      f"{metrics['size_mb']:<12.3f} "
                      f"{metrics['save_time']:<12.4f} "
                      f"{metrics['load_time']:<12.4f} "
                      f"{metrics['inference_time']:<12.4f}")
    
            print("="*70)
    
    # 使用例
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    comparison = ModelSerializationComparison(model)
    comparison.compare_formats(X_test[:100])
    
    print("\n推奨:")
    print("  - 開発/実験: Pickle, Joblib")
    print("  - MLOps: MLflow")
    print("  - プロダクション: ONNX, TorchScript")
    print("  - API化: BentoML")
    

* * *

## 4.5 モデルガバナンス

### アクセス制御

モデルへのアクセスを適切に管理し、セキュリティを確保します。
    
    
    from enum import Enum
    from datetime import datetime
    import hashlib
    
    class UserRole(Enum):
        """ユーザーロール定義"""
        DATA_SCIENTIST = "data_scientist"
        ML_ENGINEER = "ml_engineer"
        ADMIN = "admin"
        VIEWER = "viewer"
    
    class Permission(Enum):
        """権限定義"""
        READ = "read"
        WRITE = "write"
        DEPLOY = "deploy"
        DELETE = "delete"
    
    class AccessControl:
        """モデルアクセス制御システム"""
    
        # ロールと権限のマッピング
        ROLE_PERMISSIONS = {
            UserRole.VIEWER: [Permission.READ],
            UserRole.DATA_SCIENTIST: [Permission.READ, Permission.WRITE],
            UserRole.ML_ENGINEER: [Permission.READ, Permission.WRITE, Permission.DEPLOY],
            UserRole.ADMIN: [Permission.READ, Permission.WRITE, Permission.DEPLOY, Permission.DELETE]
        }
    
        def __init__(self):
            self.users = {}
            self.access_log = []
    
        def add_user(self, username, role):
            """ユーザーを追加"""
            self.users[username] = {
                'role': role,
                'created_at': datetime.now(),
                'api_key': self._generate_api_key(username)
            }
            print(f"✓ ユーザー追加: {username} ({role.value})")
    
        def _generate_api_key(self, username):
            """APIキーを生成"""
            data = f"{username}-{datetime.now().isoformat()}".encode()
            return hashlib.sha256(data).hexdigest()[:32]
    
        def check_permission(self, username, permission):
            """権限チェック"""
            if username not in self.users:
                return False
    
            user_role = self.users[username]['role']
            allowed_permissions = self.ROLE_PERMISSIONS.get(user_role, [])
    
            return permission in allowed_permissions
    
        def access_model(self, username, model_name, action):
            """モデルへのアクセス試行"""
    
            # アクセスログ記録
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'username': username,
                'model_name': model_name,
                'action': action.value,
                'granted': False
            }
    
            # 権限チェック
            if not self.check_permission(username, action):
                log_entry['reason'] = 'Insufficient permissions'
                self.access_log.append(log_entry)
                print(f"❌ アクセス拒否: {username} - {action.value} on {model_name}")
                return False
    
            log_entry['granted'] = True
            self.access_log.append(log_entry)
            print(f"✓ アクセス許可: {username} - {action.value} on {model_name}")
            return True
    
        def get_access_log(self, username=None):
            """アクセスログを取得"""
            if username:
                return [log for log in self.access_log if log['username'] == username]
            return self.access_log
    
        def export_access_log(self, filepath="access_log.json"):
            """アクセスログをエクスポート"""
            import json
            with open(filepath, 'w') as f:
                json.dump(self.access_log, f, indent=2)
            print(f"✓ アクセスログをエクスポート: {filepath}")
    
    # 使用例
    print("=== アクセス制御システム ===\n")
    
    ac = AccessControl()
    
    # ユーザー追加
    ac.add_user("alice", UserRole.DATA_SCIENTIST)
    ac.add_user("bob", UserRole.ML_ENGINEER)
    ac.add_user("charlie", UserRole.VIEWER)
    ac.add_user("admin", UserRole.ADMIN)
    
    print("\n--- アクセステスト ---")
    
    # 各種アクセス試行
    ac.access_model("alice", "credit-model", Permission.READ)      # OK
    ac.access_model("alice", "credit-model", Permission.WRITE)     # OK
    ac.access_model("alice", "credit-model", Permission.DEPLOY)    # NG
    
    ac.access_model("bob", "credit-model", Permission.DEPLOY)      # OK
    ac.access_model("charlie", "credit-model", Permission.READ)    # OK
    ac.access_model("charlie", "credit-model", Permission.WRITE)   # NG
    
    ac.access_model("admin", "credit-model", Permission.DELETE)    # OK
    
    # ログのエクスポート
    ac.export_access_log()
    
    print(f"\n総アクセス数: {len(ac.access_log)}")
    print(f"拒否数: {sum(1 for log in ac.access_log if not log['granted'])}")
    

### 監査ログ
    
    
    import json
    from datetime import datetime
    from enum import Enum
    
    class AuditEventType(Enum):
        """監査イベントタイプ"""
        MODEL_REGISTERED = "model_registered"
        MODEL_UPDATED = "model_updated"
        MODEL_DEPLOYED = "model_deployed"
        MODEL_ARCHIVED = "model_archived"
        MODEL_DELETED = "model_deleted"
        STAGE_TRANSITION = "stage_transition"
        PERMISSION_CHANGED = "permission_changed"
    
    class AuditLogger:
        """包括的な監査ログシステム"""
    
        def __init__(self, log_file="audit_log.json"):
            self.log_file = log_file
            self.events = []
    
        def log_event(self, event_type, model_name, user, details=None):
            """イベントをログに記録"""
    
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type.value,
                'model_name': model_name,
                'user': user,
                'details': details or {}
            }
    
            self.events.append(event)
    
            # ファイルに追記
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
    
            print(f"📝 監査ログ記録: {event_type.value} - {model_name} by {user}")
    
        def get_events(self, model_name=None, user=None, event_type=None):
            """フィルタリングしてイベントを取得"""
    
            filtered = self.events
    
            if model_name:
                filtered = [e for e in filtered if e['model_name'] == model_name]
    
            if user:
                filtered = [e for e in filtered if e['user'] == user]
    
            if event_type:
                filtered = [e for e in filtered if e['event_type'] == event_type.value]
    
            return filtered
    
        def generate_audit_report(self, model_name):
            """モデルの監査レポートを生成"""
    
            events = self.get_events(model_name=model_name)
    
            print(f"\n=== {model_name} 監査レポート ===")
            print(f"総イベント数: {len(events)}\n")
    
            for event in events:
                print(f"{event['timestamp']}")
                print(f"  イベント: {event['event_type']}")
                print(f"  実行者: {event['user']}")
                if event['details']:
                    print(f"  詳細: {event['details']}")
                print()
    
        def check_compliance(self, model_name, required_events):
            """コンプライアンスチェック"""
    
            events = self.get_events(model_name=model_name)
            event_types = set(e['event_type'] for e in events)
    
            compliance_status = {}
            for required in required_events:
                compliance_status[required.value] = required.value in event_types
    
            return compliance_status
    
    # 使用例
    print("=== 監査ログシステム ===\n")
    
    audit = AuditLogger()
    
    # 様々なイベントを記録
    audit.log_event(
        AuditEventType.MODEL_REGISTERED,
        "credit-model",
        "alice",
        {"version": 1, "accuracy": 0.85}
    )
    
    audit.log_event(
        AuditEventType.STAGE_TRANSITION,
        "credit-model",
        "bob",
        {"from_stage": "None", "to_stage": "Staging", "version": 1}
    )
    
    audit.log_event(
        AuditEventType.MODEL_DEPLOYED,
        "credit-model",
        "admin",
        {"version": 1, "environment": "production", "approved_by": "manager"}
    )
    
    # 監査レポート生成
    audit.generate_audit_report("credit-model")
    
    # コンプライアンスチェック
    print("\n=== コンプライアンスチェック ===")
    required = [
        AuditEventType.MODEL_REGISTERED,
        AuditEventType.MODEL_DEPLOYED
    ]
    
    compliance = audit.check_compliance("credit-model", required)
    for req, status in compliance.items():
        symbol = "✓" if status else "❌"
        print(f"{symbol} {req}: {'準拠' if status else '未準拠'}")
    

### モデルカード

**モデルカード** は、モデルの意図、性能、制限を文書化する標準形式です。
    
    
    from dataclasses import dataclass, asdict
    from typing import List, Dict
    import json
    
    @dataclass
    class ModelCard:
        """モデルカード - モデルの包括的な文書化"""
    
        # 基本情報
        model_name: str
        version: str
        date: str
        authors: List[str]
    
        # モデル詳細
        model_type: str
        architecture: str
        training_data: Dict
    
        # 性能
        performance_metrics: Dict
        test_data: Dict
    
        # 使用目的
        intended_use: str
        out_of_scope_use: List[str]
    
        # 制限事項
        limitations: List[str]
        biases: List[str]
    
        # 倫理的考慮事項
        ethical_considerations: List[str]
    
        # 推奨事項
        recommendations: List[str]
    
        def to_dict(self):
            """辞書形式に変換"""
            return asdict(self)
    
        def to_json(self, filepath):
            """JSON形式で保存"""
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"✓ モデルカードを保存: {filepath}")
    
        def to_markdown(self, filepath):
            """Markdown形式で保存"""
    
            md_content = f"""# モデルカード: {self.model_name}
    
    ## 基本情報
    - **モデル名**: {self.model_name}
    - **バージョン**: {self.version}
    - **作成日**: {self.date}
    - **作成者**: {', '.join(self.authors)}
    
    ## モデル詳細
    - **モデルタイプ**: {self.model_type}
    - **アーキテクチャ**: {self.architecture}
    
    ### 学習データ
    """
            for key, value in self.training_data.items():
                md_content += f"- **{key}**: {value}\n"
    
            md_content += f"""
    ## 性能メトリクス
    
    ### テストデータ
    """
            for key, value in self.test_data.items():
                md_content += f"- **{key}**: {value}\n"
    
            md_content += "\n### 性能\n"
            for metric, value in self.performance_metrics.items():
                md_content += f"- **{metric}**: {value}\n"
    
            md_content += f"""
    ## 使用目的
    
    ### 意図された使用方法
    {self.intended_use}
    
    ### 適用範囲外の使用
    """
            for item in self.out_of_scope_use:
                md_content += f"- {item}\n"
    
            md_content += "\n## 制限事項\n"
            for limitation in self.limitations:
                md_content += f"- {limitation}\n"
    
            md_content += "\n## バイアス\n"
            for bias in self.biases:
                md_content += f"- {bias}\n"
    
            md_content += "\n## 倫理的考慮事項\n"
            for consideration in self.ethical_considerations:
                md_content += f"- {consideration}\n"
    
            md_content += "\n## 推奨事項\n"
            for recommendation in self.recommendations:
                md_content += f"- {recommendation}\n"
    
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
    
            print(f"✓ モデルカード(Markdown)を保存: {filepath}")
    
    # 使用例
    print("=== モデルカードの作成 ===\n")
    
    model_card = ModelCard(
        model_name="信用リスク分類モデル",
        version="1.2.0",
        date="2025-10-21",
        authors=["Data Science Team", "ML Engineering Team"],
    
        model_type="Random Forest Classifier",
        architecture="100 estimators, max_depth=10",
        training_data={
            "データセット": "顧客信用データ 2023-2024",
            "サンプル数": "100,000",
            "特徴量数": "20",
            "クラス": "承認/拒否 (バランス済み)"
        },
    
        performance_metrics={
            "Accuracy": "0.892",
            "Precision": "0.885",
            "Recall": "0.901",
            "F1 Score": "0.893",
            "ROC AUC": "0.945"
        },
    
        test_data={
            "データセット": "ホールドアウトテストセット",
            "サンプル数": "20,000",
            "期間": "2024年Q3"
        },
    
        intended_use="個人向けローンの信用リスク評価。融資判断の補助ツールとして使用。",
    
        out_of_scope_use=[
            "企業向け融資の評価",
            "雇用判断への利用",
            "保険料率の設定",
            "人間による審査なしの自動承認"
        ],
    
        limitations=[
            "過去2年間のデータに基づいており、経済環境の急変には対応できない可能性",
            "18歳以上の個人のみを対象としており、未成年者には適用不可",
            "年収データが自己申告ベースであり、検証されていない",
            "地域による信用慣習の違いを十分に考慮していない"
        ],
    
        biases=[
            "学習データに若年層のサンプルが少なく、若年層への予測精度が低い可能性",
            "高所得層に偏ったデータ分布により、低所得層への予測が保守的",
            "都市部のデータが多く、地方部での適用に注意が必要"
        ],
    
        ethical_considerations=[
            "モデルの予測は参考情報であり、最終判断は人間が行うこと",
            "拒否の理由を顧客に説明できる体制を整備すること",
            "定期的にモデルの公平性を監視し、バイアスを検出すること",
            "個人情報保護法に準拠したデータ管理を実施すること"
        ],
    
        recommendations=[
            "3ヶ月ごとにモデルの性能を監視し、劣化が見られた場合は再学習を実施",
            "人間のレビュープロセスと組み合わせて使用すること",
            "モデルの予測に基づく判断は、関連法規制を遵守すること",
            "新しいデータで定期的にバイアス監査を実施すること",
            "ステークホルダーへの透明性を確保し、モデルの動作を説明可能にすること"
        ]
    )
    
    # JSON形式で保存
    model_card.to_json("model_card.json")
    
    # Markdown形式で保存
    model_card.to_markdown("model_card.md")
    
    print("\nモデルカードの利点:")
    print("  - 透明性の向上")
    print("  - 説明責任の確保")
    print("  - 適切な使用の促進")
    print("  - リスクの明確化")
    print("  - コンプライアンス対応")
    

* * *

## 4.6 本章のまとめ

### 学んだこと

  1. **モデル管理の課題**

     * バージョニング、メタデータ、ライフサイクル、ガバナンス
     * 体系的な管理の重要性
  2. **モデルレジストリ**

     * MLflow Model Registryでの中央管理
     * バージョン管理とステージ遷移
     * モデルのプロモーションとロールバック
  3. **モデルメタデータ管理**

     * モデル署名による型安全性
     * 入出力スキーマの定義と検証
     * 依存関係とパフォーマンスメトリクス
  4. **モデルパッケージング**

     * ONNX: フレームワーク非依存
     * BentoML: API化とデプロイ
     * TorchScript: 最適化と高速化
     * 各形式の使い分け
  5. **モデルガバナンス**

     * アクセス制御とRBAC
     * 監査ログとコンプライアンス
     * モデルカードによる文書化

### モデル管理のベストプラクティス

プラクティス | 説明 | 利点  
---|---|---  
**統一的なレジストリ** | すべてのモデルを一箇所で管理 | 可視性、追跡性  
**自動バージョニング** | すべての変更を自動記録 | 再現性、監査  
**ステージ管理** | 開発/Staging/本番の明確化 | リスク管理  
**メタデータの充実** | すべての関連情報を記録 | 検索性、理解  
**アクセス制御** | ロールベースの権限管理 | セキュリティ  
**監査ログ** | すべての操作を記録 | コンプライアンス  
**モデルカード** | 意図、性能、制限を文書化 | 透明性、責任  
  
### 次の章へ

第5章では、**モデルデプロイメント** を学びます：

  * バッチ推論とリアルタイム推論
  * モデルサービング（FastAPI、BentoML）
  * コンテナ化とKubernetes
  * A/Bテストとカナリアデプロイ
  * モニタリングとアラート

* * *

## 演習問題

### 問題1（難易度：easy）

モデルレジストリの主要な機能を3つ挙げ、それぞれの重要性を説明してください。

解答例

**解答** ：

  1. **バージョン管理**

     * 機能: モデルの各バージョンを自動的に追跡
     * 重要性: 再現性の確保、問題発生時のロールバック、モデル間の比較が可能
  2. **ステージ管理**

     * 機能: Staging、Production、Archivedなどのステージを定義
     * 重要性: 環境の明確化、デプロイメントリスクの低減、承認プロセスの実装
  3. **メタデータ保存**

     * 機能: ハイパーパラメータ、メトリクス、説明などを保存
     * 重要性: モデルの検索性向上、意思決定の支援、監査とコンプライアンス

### 問題2（難易度：medium）

MLflow Model Registryを使用して、モデルを登録し、Stagingからプロダクションに昇格させるコードを書いてください。

解答例
    
    
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # セットアップ
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # データ準備
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # モデル学習と登録
    model_name = "my_classifier"
    
    with mlflow.start_run():
        # モデル学習
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # メトリクス計算
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
    
        # モデル登録
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
    
    # 最新バージョンを取得
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max([int(v.version) for v in versions])
    
    # Stagingに遷移
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging"
    )
    print(f"✓ バージョン {latest_version} を Staging に遷移")
    
    # テスト後、Productionに昇格
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True  # 既存のProductionをアーカイブ
    )
    print(f"✓ バージョン {latest_version} を Production に昇格")
    
    # Productionモデルの取得
    prod_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    print(f"✓ Productionモデルを読み込み")
    

### 問題3（難易度：medium）

モデル署名（signature）を作成し、入力データの検証を行うコードを書いてください。

解答例
    
    
    import mlflow
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from mlflow.models.signature import infer_signature
    
    # データ準備
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100)
    })
    y_train = np.random.randint(0, 2, 100)
    
    # モデル学習
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 予測（署名の推論用）
    predictions = model.predict(X_train[:5])
    
    # 署名の作成
    signature = infer_signature(X_train, predictions)
    
    print("=== モデル署名 ===")
    print(signature)
    
    # 署名付きでモデルを保存
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )
        print("\n✓ 署名付きモデルを保存")
    
    # 検証: 正しい入力
    print("\n=== 入力検証 ===")
    correct_input = pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [2.0],
        'feature_3': [3.0]
    })
    print(f"✓ 正しい入力形式（列数: {len(correct_input.columns)}）")
    
    # 検証: 間違った入力（列が不足）
    try:
        wrong_input = pd.DataFrame({
            'feature_1': [1.0],
            'feature_2': [2.0]
            # feature_3 が欠落
        })
        print(f"❌ 間違った入力形式（列数: {len(wrong_input.columns)}）")
        print("   → 実際のデプロイ時にMLflowがエラーを検出")
    except Exception as e:
        print(f"✓ エラー検出: {e}")
    

### 問題4（難易度：hard）

アクセス制御システムを実装し、異なるロールのユーザーがモデルに対して実行できる操作を制限してください。監査ログも含めてください。

解答例
    
    
    from enum import Enum
    from datetime import datetime
    import json
    
    class UserRole(Enum):
        VIEWER = "viewer"
        DATA_SCIENTIST = "data_scientist"
        ML_ENGINEER = "ml_engineer"
        ADMIN = "admin"
    
    class Permission(Enum):
        READ = "read"
        WRITE = "write"
        DEPLOY = "deploy"
        DELETE = "delete"
    
    class ModelAccessControl:
        """モデルアクセス制御と監査ログ"""
    
        ROLE_PERMISSIONS = {
            UserRole.VIEWER: [Permission.READ],
            UserRole.DATA_SCIENTIST: [Permission.READ, Permission.WRITE],
            UserRole.ML_ENGINEER: [Permission.READ, Permission.WRITE, Permission.DEPLOY],
            UserRole.ADMIN: [Permission.READ, Permission.WRITE, Permission.DEPLOY, Permission.DELETE]
        }
    
        def __init__(self):
            self.users = {}
            self.audit_log = []
    
        def add_user(self, username, role):
            """ユーザーを追加"""
            self.users[username] = {'role': role, 'created_at': datetime.now()}
            self._log_audit("USER_ADDED", None, username, {"role": role.value})
    
        def check_permission(self, username, permission):
            """権限をチェック"""
            if username not in self.users:
                return False
            user_role = self.users[username]['role']
            return permission in self.ROLE_PERMISSIONS.get(user_role, [])
    
        def execute_action(self, username, model_name, action):
            """アクションを実行（権限チェック付き）"""
    
            # 権限チェック
            if not self.check_permission(username, action):
                self._log_audit(
                    "ACCESS_DENIED",
                    model_name,
                    username,
                    {"action": action.value, "reason": "insufficient_permissions"}
                )
                print(f"❌ アクセス拒否: {username} - {action.value}")
                return False
    
            # アクション実行
            self._log_audit("ACTION_EXECUTED", model_name, username, {"action": action.value})
            print(f"✓ アクション実行: {username} - {action.value} on {model_name}")
            return True
    
        def _log_audit(self, event_type, model_name, username, details):
            """監査ログに記録"""
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'model_name': model_name,
                'username': username,
                'details': details
            }
            self.audit_log.append(event)
    
        def export_audit_log(self, filepath="audit.json"):
            """監査ログをエクスポート"""
            with open(filepath, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            print(f"\n✓ 監査ログをエクスポート: {filepath}")
    
        def get_user_activity(self, username):
            """ユーザーのアクティビティを取得"""
            return [log for log in self.audit_log if log['username'] == username]
    
        def get_model_activity(self, model_name):
            """モデルのアクティビティを取得"""
            return [log for log in self.audit_log
                    if log['model_name'] == model_name]
    
    # 使用例
    print("=== アクセス制御と監査ログ ===\n")
    
    access_control = ModelAccessControl()
    
    # ユーザー追加
    access_control.add_user("alice", UserRole.DATA_SCIENTIST)
    access_control.add_user("bob", UserRole.ML_ENGINEER)
    access_control.add_user("charlie", UserRole.VIEWER)
    access_control.add_user("admin", UserRole.ADMIN)
    
    print("\n--- アクション実行 ---")
    
    # 各種アクション
    access_control.execute_action("alice", "credit-model", Permission.READ)
    access_control.execute_action("alice", "credit-model", Permission.WRITE)
    access_control.execute_action("alice", "credit-model", Permission.DEPLOY)  # 失敗
    
    access_control.execute_action("bob", "credit-model", Permission.DEPLOY)
    access_control.execute_action("charlie", "credit-model", Permission.READ)
    access_control.execute_action("charlie", "credit-model", Permission.WRITE)  # 失敗
    
    access_control.execute_action("admin", "credit-model", Permission.DELETE)
    
    # 監査ログのエクスポート
    access_control.export_audit_log()
    
    # ユーザーアクティビティ
    print("\n--- Aliceのアクティビティ ---")
    alice_activity = access_control.get_user_activity("alice")
    for activity in alice_activity:
        print(f"{activity['timestamp']}: {activity['event_type']} - {activity.get('details', {})}")
    
    print(f"\n総監査イベント数: {len(access_control.audit_log)}")
    

### 問題5（難易度：hard）

モデルカードを作成し、JSON形式とMarkdown形式の両方でエクスポートしてください。制限事項とバイアスも含めてください。

解答例
    
    
    from dataclasses import dataclass, asdict
    import json
    
    @dataclass
    class ModelCard:
        """包括的なモデルカード"""
        model_name: str
        version: str
        date: str
        authors: list
        model_type: str
        intended_use: str
        performance: dict
        limitations: list
        biases: list
        ethical_considerations: list
    
        def to_json(self, filepath):
            """JSON形式でエクスポート"""
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            print(f"✓ JSON形式で保存: {filepath}")
    
        def to_markdown(self, filepath):
            """Markdown形式でエクスポート"""
            md = f"""# モデルカード: {self.model_name}
    
    ## 基本情報
    - モデル名: {self.model_name}
    - バージョン: {self.version}
    - 作成日: {self.date}
    - 作成者: {', '.join(self.authors)}
    - モデルタイプ: {self.model_type}
    
    ## 使用目的
    {self.intended_use}
    
    ## 性能メトリクス
    """
            for metric, value in self.performance.items():
                md += f"- {metric}: {value}\n"
    
            md += "\n## 制限事項\n"
            for limitation in self.limitations:
                md += f"- {limitation}\n"
    
            md += "\n## バイアス\n"
            for bias in self.biases:
                md += f"- {bias}\n"
    
            md += "\n## 倫理的考慮事項\n"
            for consideration in self.ethical_considerations:
                md += f"- {consideration}\n"
    
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md)
            print(f"✓ Markdown形式で保存: {filepath}")
    
    # モデルカード作成
    card = ModelCard(
        model_name="住宅ローン承認モデル",
        version="2.1.0",
        date="2025-10-21",
        authors=["Data Science Team", "Risk Management Team"],
        model_type="Gradient Boosting Classifier",
        intended_use="住宅ローン申請の初期審査における承認可能性の評価",
        performance={
            "Accuracy": "0.87",
            "Precision": "0.84",
            "Recall": "0.89",
            "F1 Score": "0.865",
            "ROC AUC": "0.92"
        },
        limitations=[
            "学習データは過去3年間に限定されており、長期的な経済変動には対応していない",
            "申請者の年齢が25歳未満の場合、サンプル数が少なく予測精度が低下する",
            "自営業者のデータが不足しており、この層への予測は保守的になる傾向"
        ],
        biases=[
            "都市部のデータが農村部より多く、地域による予測精度に差がある",
            "高所得層のデータが多く、低所得層への予測が厳しくなる傾向",
            "性別による承認率の差異が観察されており、定期的な監視が必要"
        ],
        ethical_considerations=[
            "モデルの予測は参考情報であり、最終判断は人間の審査員が行う",
            "拒否された場合、その理由を説明可能な形で提供する",
            "公平性メトリクスを定期的に監視し、不当な差別がないことを確認",
            "個人情報保護規制に準拠したデータ管理を実施"
        ]
    )
    
    # エクスポート
    print("=== モデルカードのエクスポート ===\n")
    card.to_json("model_card.json")
    card.to_markdown("MODEL_CARD.md")
    
    print("\nモデルカードには以下が含まれます:")
    print("  ✓ 基本情報とメタデータ")
    print("  ✓ 性能メトリクス")
    print("  ✓ 制限事項の明示")
    print("  ✓ バイアスの開示")
    print("  ✓ 倫理的考慮事項")
    

* * *

## 参考文献

  1. Sato, D., Wider, A., & Windheuser, C. (2019). _Continuous Delivery for Machine Learning_. Martin Fowler's Blog.
  2. Polyzotis, N., et al. (2018). _Data Lifecycle Challenges in Production Machine Learning: A Survey_. ACM SIGMOD Record.
  3. Mitchell, M., et al. (2019). _Model Cards for Model Reporting_. Proceedings of FAT* 2019.
  4. Paleyes, A., Urma, R. G., & Lawrence, N. D. (2022). _Challenges in Deploying Machine Learning: A Survey of Case Studies_. ACM Computing Surveys.
  5. Sculley, D., et al. (2015). _Hidden Technical Debt in Machine Learning Systems_. NIPS 2015.
