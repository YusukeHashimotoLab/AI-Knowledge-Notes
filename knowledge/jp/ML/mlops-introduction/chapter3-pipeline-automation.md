---
title: 第3章：パイプライン自動化
chapter_title: 第3章：パイプライン自動化
subtitle: MLワークフローの自動化とオーケストレーション
reading_time: 30-35分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ MLパイプラインの設計原則とDAG構造を理解する
  * ✅ Apache Airflowでワークフローを構築できる
  * ✅ Kubeflow Pipelinesでコンテナ化されたパイプラインを作成できる
  * ✅ Prefectで動的なワークフローを実装できる
  * ✅ 冪等性、エラーハンドリング、テスタビリティを実現できる
  * ✅ 本番環境で運用可能なパイプラインを設計できる

* * *

## 3.1 MLパイプラインの設計

### パイプラインとは

**MLパイプライン（Machine Learning Pipeline）** は、データ収集から予測までの一連のプロセスを自動化するワークフローです。

> 「手作業での再現は不可能。自動化されたパイプラインこそが、信頼できるMLシステムの基盤です。」

### パイプラインの構成要素

要素 | 説明 | 例  
---|---|---  
**データ取得** | 外部ソースからデータを収集 | API呼び出し、DB抽出  
**前処理** | データクリーニング、変換 | 欠損値処理、スケーリング  
**特徴量エンジニアリング** | モデル入力用の特徴量作成 | カテゴリカル変換、集約  
**モデル訓練** | アルゴリズムの学習 | fit、ハイパーパラメータ調整  
**評価** | モデル性能の測定 | 精度、再現率、F1スコア  
**デプロイ** | 本番環境への配置 | モデルサービング、API化  
  
### DAG（Directed Acyclic Graph）

**DAG** は、タスク間の依存関係を表す有向非巡回グラフです。MLパイプラインの標準的な表現方法として広く使用されています。
    
    
    ```mermaid
    graph TD
        A[データ取得] --> B[データ検証]
        B --> C[前処理]
        C --> D[特徴量エンジニアリング]
        D --> E[訓練データ分割]
        E --> F[モデル訓練]
        E --> G[ハイパーパラメータ調整]
        F --> H[モデル評価]
        G --> H
        H --> I{性能OK?}
        I -->|Yes| J[モデル登録]
        I -->|No| K[アラート送信]
        J --> L[デプロイ]
    
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

### オーケストレーション vs ワークフロー

観点 | オーケストレーション | ワークフロー  
---|---|---  
**制御** | 中央集権的（オーケストレータが管理） | 分散的（各タスクが独立）  
**例** | Airflow、Prefect、Dagster | Step Functions、Argo Workflows  
**適用場面** | 複雑な依存関係、動的タスク | シンプルなフロー、イベント駆動  
**可視化** | UI完備、ログ追跡 | 基本的なステータス表示  
  
### パイプライン設計の原則
    
    
    """
    MLパイプライン設計の5つの原則
    """
    
    # 1. 冪等性（Idempotency）
    # 同じ入力から同じ出力が得られる
    def preprocess_data(input_path, output_path):
        """同じinput_pathから常に同じoutput_pathを生成"""
        # 既存のoutputを削除してから再生成
        if os.path.exists(output_path):
            os.remove(output_path)
        # 処理実行...
    
    # 2. 再実行可能性（Rerunability）
    # 失敗したタスクを安全に再実行できる
    def train_model(data_path, model_path, force=False):
        """force=Trueで既存モデルを上書き"""
        if os.path.exists(model_path) and not force:
            print(f"モデル既存: {model_path}")
            return
        # 訓練実行...
    
    # 3. 疎結合（Loose Coupling）
    # タスク間の依存を最小化
    def extract_features(raw_data):
        """rawデータから特徴量を抽出（前のタスクに依存しない）"""
        return features
    
    # 4. パラメータ化（Parameterization）
    # ハードコードを避け、設定を外部化
    def run_pipeline(config_path):
        """設定ファイルから全パラメータを読み込み"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # config['model_type'], config['batch_size']などを使用
    
    # 5. 可観測性（Observability）
    # ログ、メトリクス、トレースを記録
    import logging
    
    def process_batch(batch_id):
        logging.info(f"Batch {batch_id} 処理開始")
        try:
            # 処理...
            logging.info(f"Batch {batch_id} 成功")
        except Exception as e:
            logging.error(f"Batch {batch_id} 失敗: {e}")
            raise
    

* * *

## 3.2 Apache Airflow

### Airflowとは

**Apache Airflow** は、Pythonでワークフローを定義し、スケジュール実行できるオープンソースのプラットフォームです。

#### Airflowの特徴

  * **DAGベース** : タスクをDAGで表現
  * **豊富なオペレータ** : Python、Bash、SQL、クラウドサービスなど
  * **スケジューラ** : Cron式でのスケジューリング
  * **Web UI** : DAGの可視化、ログ確認
  * **拡張性** : カスタムオペレータ、フック、センサー

### Airflow アーキテクチャ
    
    
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

コンポーネント | 役割  
---|---  
**Scheduler** | DAGの監視、タスクのスケジュール  
**Executor** | タスクの実行管理（Local、Celery、Kubernetes）  
**Worker** | 実際のタスク実行  
**Web Server** | UI提供、DAG可視化  
**Metadata DB** | DAG、タスク、実行履歴の保存  
  
### 基本的なDAG定義
    
    
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    
    # デフォルト引数
    default_args = {
        'owner': 'mlops-team',
        'depends_on_past': False,  # 過去の実行に依存しない
        'email': ['alerts@example.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,  # 失敗時に2回再試行
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=1),
    }
    
    # DAG定義
    dag = DAG(
        'ml_pipeline_basic',
        default_args=default_args,
        description='基本的なMLパイプライン',
        schedule_interval='0 2 * * *',  # 毎日2:00 AM実行
        start_date=datetime(2025, 1, 1),
        catchup=False,  # 過去の未実行分を実行しない
        tags=['ml', 'training'],
    )
    
    # タスク定義
    def extract_data(**context):
        """データ抽出"""
        print("データをDBから抽出中...")
        # 実際の抽出ロジック
        data = {'records': 1000, 'timestamp': datetime.now().isoformat()}
        # XComで次のタスクにデータを渡す
        context['ti'].xcom_push(key='extracted_data', value=data)
        return data
    
    def transform_data(**context):
        """データ変換"""
        # 前のタスクからデータを取得
        ti = context['ti']
        data = ti.xcom_pull(key='extracted_data', task_ids='extract')
        print(f"データ変換中: {data['records']}件")
        # 変換処理...
        transformed = {'records': data['records'], 'features': 50}
        return transformed
    
    def train_model(**context):
        """モデル訓練"""
        ti = context['ti']
        data = ti.xcom_pull(task_ids='transform')
        print(f"モデル訓練中: {data['features']}特徴量")
        # 訓練処理...
        model_metrics = {'accuracy': 0.92, 'f1': 0.89}
        return model_metrics
    
    # タスクの作成
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
        bash_command='echo "モデル検証完了"',
        dag=dag,
    )
    
    # タスク依存関係の定義
    extract_task >> transform_task >> train_task >> validate_task
    

### 完全なMLパイプライン例
    
    
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
        description='完全なMLパイプライン（訓練から評価まで）',
        schedule_interval='@daily',
        start_date=datetime(2025, 1, 1),
        catchup=False,
    )
    
    # データ収集
    def collect_data(**context):
        """データ収集タスク"""
        # ダミーデータ生成（実際はDBやAPIから取得）
        import numpy as np
        np.random.seed(42)
    
        n_samples = 1000
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
        # データを保存
        data.to_csv('/tmp/raw_data.csv', index=False)
        print(f"データ収集完了: {len(data)}件")
    
        # メタデータをXComで共有
        context['ti'].xcom_push(key='data_size', value=len(data))
        return '/tmp/raw_data.csv'
    
    # データ検証
    def validate_data(**context):
        """データ品質検証"""
        data = pd.read_csv('/tmp/raw_data.csv')
    
        # 検証チェック
        checks = {
            'no_nulls': data.isnull().sum().sum() == 0,
            'sufficient_size': len(data) >= 500,
            'target_balance': data['target'].value_counts().min() / len(data) >= 0.3
        }
    
        print(f"データ検証結果: {checks}")
    
        if not all(checks.values()):
            raise ValueError(f"データ品質チェック失敗: {checks}")
    
        return True
    
    # 前処理
    def preprocess_data(**context):
        """前処理タスク"""
        data = pd.read_csv('/tmp/raw_data.csv')
    
        # 特徴量とターゲットの分離
        X = data.drop('target', axis=1)
        y = data['target']
    
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # 保存
        X_train.to_csv('/tmp/X_train.csv', index=False)
        X_test.to_csv('/tmp/X_test.csv', index=False)
        y_train.to_csv('/tmp/y_train.csv', index=False)
        y_test.to_csv('/tmp/y_test.csv', index=False)
    
        print(f"前処理完了: 訓練={len(X_train)}, テスト={len(X_test)}")
        return True
    
    # モデル訓練
    def train_model(**context):
        """モデル訓練タスク"""
        # データ読み込み
        X_train = pd.read_csv('/tmp/X_train.csv')
        y_train = pd.read_csv('/tmp/y_train.csv').values.ravel()
    
        # モデル訓練
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # モデル保存
        joblib.dump(model, '/tmp/model.pkl')
        print("モデル訓練完了")
        return '/tmp/model.pkl'
    
    # モデル評価
    def evaluate_model(**context):
        """モデル評価タスク"""
        # データとモデルの読み込み
        X_test = pd.read_csv('/tmp/X_test.csv')
        y_test = pd.read_csv('/tmp/y_test.csv').values.ravel()
        model = joblib.load('/tmp/model.pkl')
    
        # 予測と評価
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
    
        print(f"評価完了: {metrics}")
    
        # メトリクスをXComで共有
        context['ti'].xcom_push(key='metrics', value=metrics)
        return metrics
    
    # モデル品質チェック（分岐）
    def check_model_quality(**context):
        """モデル品質に基づいて次のタスクを決定"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # 精度閾値
        threshold = 0.8
    
        if metrics['accuracy'] >= threshold:
            print(f"モデル承認: accuracy={metrics['accuracy']:.3f}")
            return 'register_model'
        else:
            print(f"モデル却下: accuracy={metrics['accuracy']:.3f} < {threshold}")
            return 'send_alert'
    
    # モデル登録
    def register_model(**context):
        """モデルをレジストリに登録"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # 実際はMLflowなどのレジストリに登録
        print(f"モデル登録: accuracy={metrics['accuracy']:.3f}")
    
        # バージョン管理
        import shutil
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_version = f'/tmp/model_{timestamp}.pkl'
        shutil.copy('/tmp/model.pkl', model_version)
    
        print(f"モデルバージョン: {model_version}")
        return model_version
    
    # アラート送信
    def send_alert(**context):
        """品質不足の場合にアラート送信"""
        ti = context['ti']
        metrics = ti.xcom_pull(key='metrics', task_ids='evaluate')
    
        # 実際はSlackやEmailで通知
        print(f"⚠️ アラート: モデル品質不足 - {metrics}")
        return True
    
    # タスク定義
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
        trigger_rule=TriggerRule.ONE_SUCCESS,  # どちらかが成功すれば実行
        dag=dag,
    )
    
    # DAG構造
    start >> collect >> validate >> preprocess >> train >> evaluate >> quality_check
    quality_check >> [register, alert]
    register >> end
    alert >> end
    

### Airflowのベストプラクティス
    
    
    """
    Airflow ベストプラクティス
    """
    
    # 1. タスクの冪等性を保証
    def idempotent_task(output_path):
        """同じ入力から常に同じ出力を生成"""
        # 既存の出力を削除
        if os.path.exists(output_path):
            os.remove(output_path)
        # 処理実行
        process_data(output_path)
    
    # 2. XComは小さなデータのみ
    def small_xcom(**context):
        """大きなデータはファイルで受け渡し"""
        # ❌ 悪い例: 大きなDataFrameをXComで渡す
        # context['ti'].xcom_push(key='data', value=large_df)
    
        # ✅ 良い例: ファイルパスを渡す
        large_df.to_parquet('/tmp/data.parquet')
        context['ti'].xcom_push(key='data_path', value='/tmp/data.parquet')
    
    # 3. タスクの粒度を適切に
    # ❌ 悪い例: 1つのタスクで全処理
    def monolithic_task():
        collect_data()
        preprocess_data()
        train_model()
        evaluate_model()
    
    # ✅ 良い例: 各ステップを分離
    collect_task >> preprocess_task >> train_task >> evaluate_task
    
    # 4. 動的タスク生成
    from airflow.operators.python import PythonOperator
    
    def create_dynamic_tasks(dag):
        """複数モデルを並列訓練"""
        models = ['rf', 'xgboost', 'lightgbm']
    
        for model_name in models:
            PythonOperator(
                task_id=f'train_{model_name}',
                python_callable=train_specific_model,
                op_kwargs={'model_type': model_name},
                dag=dag,
            )
    
    # 5. センサーでの待機
    from airflow.sensors.filesystem import FileSensor
    
    wait_for_data = FileSensor(
        task_id='wait_for_data',
        filepath='/data/input.csv',
        poke_interval=60,  # 60秒ごとにチェック
        timeout=3600,  # 1時間でタイムアウト
        dag=dag,
    )
    

* * *

## 3.3 Kubeflow Pipelines

### Kubeflow Pipelinesとは

**Kubeflow Pipelines** は、Kubernetes上でMLワークフローを構築、デプロイ、管理するためのプラットフォームです。

#### 主な特徴

  * **コンテナネイティブ** : 各タスクがDockerコンテナで実行
  * **再利用可能なコンポーネント** : パイプラインの部品化
  * **スケーラビリティ** : Kubernetesの自動スケーリング活用
  * **バージョニング** : パイプラインとコンポーネントのバージョン管理
  * **実験追跡** : パイプライン実行の比較と分析

### Kubeflowアーキテクチャ
    
    
    ```mermaid
    graph TB
        A[Pipeline DSL] --> B[Compiler]
        B --> C[Pipeline YAML]
        C --> D[Kubeflow API Server]
        D --> E[Argo Workflows]
        E --> F1[Pod: データ収集]
        E --> F2[Pod: 前処理]
        E --> F3[Pod: 訓練]
        E --> F4[Pod: 評価]
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

### 基本的なパイプライン
    
    
    import kfp
    from kfp import dsl
    from kfp.dsl import component, Input, Output, Dataset, Model
    
    # コンポーネント定義（軽量コンポーネント）
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def load_data(output_dataset: Output[Dataset]):
        """データ読み込みコンポーネント"""
        import pandas as pd
    
        # ダミーデータ生成
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 0, 1, 1, 1]
        })
    
        # 出力データセットに保存
        data.to_csv(output_dataset.path, index=False)
        print(f"データ保存: {output_dataset.path}")
    
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def train_model(
        input_dataset: Input[Dataset],
        output_model: Output[Model],
        n_estimators: int = 100
    ):
        """モデル訓練コンポーネント"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    
        # データ読み込み
        data = pd.read_csv(input_dataset.path)
        X = data[['feature1', 'feature2']]
        y = data['target']
    
        # モデル訓練
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
    
        # モデル保存
        joblib.dump(model, output_model.path)
        print(f"モデル保存: {output_model.path}")
    
    @component(
        base_image='python:3.9',
        packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
    )
    def evaluate_model(
        input_dataset: Input[Dataset],
        input_model: Input[Model]
    ) -> float:
        """モデル評価コンポーネント"""
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score
    
        # データとモデルの読み込み
        data = pd.read_csv(input_dataset.path)
        X = data[['feature1', 'feature2']]
        y = data['target']
        model = joblib.load(input_model.path)
    
        # 評価
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
    
        print(f"精度: {accuracy:.3f}")
        return accuracy
    
    # パイプライン定義
    @dsl.pipeline(
        name='ML Training Pipeline',
        description='基本的なML訓練パイプライン'
    )
    def ml_pipeline(n_estimators: int = 100):
        """MLパイプライン"""
        # タスク定義
        load_task = load_data()
    
        train_task = train_model(
            input_dataset=load_task.outputs['output_dataset'],
            n_estimators=n_estimators
        )
    
        evaluate_task = evaluate_model(
            input_dataset=load_task.outputs['output_dataset'],
            input_model=train_task.outputs['output_model']
        )
    
    # パイプラインのコンパイル
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=ml_pipeline,
            package_path='ml_pipeline.yaml'
        )
        print("パイプラインコンパイル完了: ml_pipeline.yaml")
    

### コンテナ化されたコンポーネント
    
    
    """
    Dockerコンテナベースのコンポーネント定義
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
        # データ読み込み
        data = pd.read_csv(args.input_data)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # モデル訓練
        model = RandomForestClassifier(n_estimators=args.n_estimators)
        model.fit(X, y)
    
        # 保存
        joblib.dump(model, args.output_model)
        print(f"モデル保存: {args.output_model}")
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-data', required=True)
        parser.add_argument('--output-model', required=True)
        parser.add_argument('--n-estimators', type=int, default=100)
        args = parser.parse_args()
        main(args)
    """
    
    # パイプラインでのコンテナ使用
    @dsl.pipeline(
        name='Containerized ML Pipeline',
        description='コンテナ化されたMLパイプライン'
    )
    def containerized_pipeline(n_estimators: int = 100):
        """コンテナベースのパイプライン"""
    
        # データ準備コンテナ
        prepare_op = dsl.ContainerOp(
            name='prepare-data',
            image='gcr.io/my-project/data-prep:v1',
            arguments=['--output', '/data/prepared.csv'],
            file_outputs={'data': '/data/prepared.csv'}
        )
    
        # 訓練コンテナ
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
    
        # 評価コンテナ
        evaluate_op = dsl.ContainerOp(
            name='evaluate-model',
            image='gcr.io/my-project/evaluate:v1',
            arguments=[
                '--input-data', prepare_op.outputs['data'],
                '--input-model', train_op.outputs['model']
            ]
        )
    
        # GPU使用の指定
        train_op.set_gpu_limit(1)
        train_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')
    
    # コンパイルと実行
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=containerized_pipeline,
            package_path='containerized_pipeline.yaml'
        )
    

### Kubeflow パイプライン実行例
    
    
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
        """データ前処理とトレイン・テスト分割"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
    
        data = pd.read_csv(input_dataset.path)
    
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
        # 保存
        train_df = X_train.copy()
        train_df['target'] = y_train
        test_df = X_test.copy()
        test_df['target'] = y_test
    
        train_df.to_csv(output_train.path, index=False)
        test_df.to_csv(output_test.path, index=False)
    
        print(f"訓練データ: {len(train_df)}件")
        print(f"テストデータ: {len(test_df)}件")
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def hyperparameter_tuning(
        input_train: Input[Dataset],
        output_best_params: Output[Metrics]
    ) -> dict:
        """ハイパーパラメータチューニング"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import json
    
        data = pd.read_csv(input_train.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # グリッドサーチ
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
    
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X, y)
    
        best_params = grid_search.best_params_
    
        # メトリクス保存
        with open(output_best_params.path, 'w') as f:
            json.dump(best_params, f)
    
        print(f"最適パラメータ: {best_params}")
        print(f"最高スコア: {grid_search.best_score_:.3f}")
    
        return best_params
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def train_final_model(
        input_train: Input[Dataset],
        best_params: dict,
        output_model: Output[Model]
    ):
        """最適パラメータでモデル訓練"""
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    
        data = pd.read_csv(input_train.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        # 最適パラメータでモデル訓練
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X, y)
    
        # モデル保存
        joblib.dump(model, output_model.path)
        print(f"モデル訓練完了: {best_params}")
    
    @component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn'])
    def evaluate_final_model(
        input_test: Input[Dataset],
        input_model: Input[Model],
        output_metrics: Output[Metrics]
    ):
        """最終評価"""
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import json
    
        data = pd.read_csv(input_test.path)
        X = data.drop('target', axis=1)
        y = data['target']
    
        model = joblib.load(input_model.path)
        y_pred = model.predict(X)
    
        # メトリクス計算
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'f1_score': float(f1_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred))
        }
    
        # 保存
        with open(output_metrics.path, 'w') as f:
            json.dump(metrics, f)
    
        print(f"評価完了: {metrics}")
    
    @dsl.pipeline(
        name='Complete ML Pipeline with Tuning',
        description='ハイパーパラメータチューニング付き完全MLパイプライン'
    )
    def complete_ml_pipeline(test_size: float = 0.2):
        """完全なMLパイプライン"""
        # データ読み込み（ダミー）
        load_task = load_data()
    
        # 前処理
        preprocess_task = preprocess_data(
            input_dataset=load_task.outputs['output_dataset'],
            test_size=test_size
        )
    
        # ハイパーパラメータチューニング
        tuning_task = hyperparameter_tuning(
            input_train=preprocess_task.outputs['output_train']
        )
    
        # 最終モデル訓練
        train_task = train_final_model(
            input_train=preprocess_task.outputs['output_train'],
            best_params=tuning_task.output
        )
    
        # 評価
        evaluate_task = evaluate_final_model(
            input_test=preprocess_task.outputs['output_test'],
            input_model=train_task.outputs['output_model']
        )
    
    # コンパイル
    if __name__ == '__main__':
        kfp.compiler.Compiler().compile(
            pipeline_func=complete_ml_pipeline,
            package_path='complete_ml_pipeline.yaml'
        )
        print("パイプラインコンパイル完了")
    

* * *

## 3.4 Prefect

### Prefectとは

**Prefect** は、Pythonネイティブなワークフローオーケストレーションツールです。動的タスク生成と柔軟なエラーハンドリングが特徴です。

#### Prefectの特徴

  * **Pythonic** : 通常のPython関数をデコレータでタスク化
  * **動的ワークフロー** : 実行時にタスクを生成可能
  * **ローカル実行** : 開発環境で簡単にテスト
  * **クラウドUI** : Prefect Cloudで可視化と管理
  * **柔軟なスケジューリング** : Cron、Interval、Event駆動

### 基本的なFlow
    
    
    from prefect import flow, task
    from datetime import timedelta
    
    @task(retries=3, retry_delay_seconds=60)
    def extract_data():
        """データ抽出タスク"""
        print("データ抽出中...")
        # 抽出処理
        data = {'records': 1000}
        return data
    
    @task
    def transform_data(data):
        """データ変換タスク"""
        print(f"データ変換中: {data['records']}件")
        # 変換処理
        transformed = {'records': data['records'], 'features': 50}
        return transformed
    
    @task(timeout_seconds=3600)
    def load_data(data):
        """データロード タスク"""
        print(f"データロード中: {data['records']}件")
        # ロード処理
        return True
    
    @flow(name="ETL Pipeline", log_prints=True)
    def etl_pipeline():
        """ETLパイプライン"""
        # タスク実行
        raw_data = extract_data()
        transformed_data = transform_data(raw_data)
        load_data(transformed_data)
    
        print("ETLパイプライン完了")
    
    if __name__ == "__main__":
        etl_pipeline()
    

### 動的タスク生成
    
    
    from prefect import flow, task
    from typing import List
    
    @task
    def train_model(model_type: str, data_path: str):
        """個別モデル訓練"""
        print(f"{model_type}モデルを訓練中...")
        # 訓練処理
        metrics = {'model': model_type, 'accuracy': 0.85}
        return metrics
    
    @task
    def select_best_model(results: List[dict]):
        """最良モデルを選択"""
        best = max(results, key=lambda x: x['accuracy'])
        print(f"最良モデル: {best['model']} (accuracy={best['accuracy']:.3f})")
        return best
    
    @flow(name="Multi-Model Training")
    def multi_model_training(data_path: str):
        """複数モデルを並列訓練"""
        # 訓練するモデルのリスト
        model_types = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    
        # 動的にタスクを生成
        results = []
        for model_type in model_types:
            result = train_model(model_type, data_path)
            results.append(result)
    
        # 最良モデルを選択
        best_model = select_best_model(results)
    
        return best_model
    
    if __name__ == "__main__":
        best = multi_model_training(data_path="/data/train.csv")
        print(f"選択されたモデル: {best}")
    

### Prefect 2.0 完全な例
    
    
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
        name="データ収集",
        retries=3,
        retry_delay_seconds=60,
        cache_key_fn=None,  # キャッシュ無効化
        timeout_seconds=300
    )
    def collect_data():
        """データ収集タスク"""
        logger = get_run_logger()
        logger.info("データ収集開始")
    
        # ダミーデータ生成
        import numpy as np
        np.random.seed(42)
    
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
    
        # データ保存
        output_path = '/tmp/raw_data.csv'
        data.to_csv(output_path, index=False)
    
        logger.info(f"データ収集完了: {len(data)}件")
        return output_path
    
    @task(name="データ検証")
    def validate_data(data_path: str):
        """データ品質検証"""
        logger = get_run_logger()
        logger.info("データ検証開始")
    
        data = pd.read_csv(data_path)
    
        # 検証
        checks = {
            'no_nulls': data.isnull().sum().sum() == 0,
            'sufficient_size': len(data) >= 500,
            'feature_count': data.shape[1] >= 4
        }
    
        logger.info(f"検証結果: {checks}")
    
        if not all(checks.values()):
            raise ValueError(f"データ検証失敗: {checks}")
    
        return True
    
    @task(name="前処理")
    def preprocess_data(data_path: str):
        """データ前処理"""
        logger = get_run_logger()
        logger.info("前処理開始")
    
        data = pd.read_csv(data_path)
    
        # 分割
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # 保存
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
    
        logger.info(f"前処理完了: 訓練={len(X_train)}, テスト={len(X_test)}")
        return paths
    
    @task(name="モデル訓練", timeout_seconds=1800)
    def train_model(data_paths: dict, n_estimators: int = 100):
        """モデル訓練"""
        logger = get_run_logger()
        logger.info(f"モデル訓練開始: n_estimators={n_estimators}")
    
        # データ読み込み
        X_train = pd.read_csv(data_paths['X_train'])
        y_train = pd.read_csv(data_paths['y_train']).values.ravel()
    
        # 訓練
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    
        # 保存
        model_path = '/tmp/model.pkl'
        joblib.dump(model, model_path)
    
        logger.info("モデル訓練完了")
        return model_path
    
    @task(name="モデル評価")
    def evaluate_model(model_path: str, data_paths: dict):
        """モデル評価"""
        logger = get_run_logger()
        logger.info("モデル評価開始")
    
        # 読み込み
        X_test = pd.read_csv(data_paths['X_test'])
        y_test = pd.read_csv(data_paths['y_test']).values.ravel()
        model = joblib.load(model_path)
    
        # 評価
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        metrics = {
            'accuracy': float(accuracy),
            'test_samples': len(y_test)
        }
    
        logger.info(f"評価完了: {metrics}")
        return metrics
    
    @task(name="モデル登録")
    def register_model(model_path: str, metrics: dict):
        """モデル登録"""
        logger = get_run_logger()
    
        # 品質チェック
        if metrics['accuracy'] < 0.7:
            logger.warning(f"モデル品質不足: accuracy={metrics['accuracy']:.3f}")
            return False
    
        # 登録（実際はMLflowなど）
        import shutil
        from datetime import datetime
    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        registry_path = f'/tmp/models/model_{timestamp}.pkl'
    
        import os
        os.makedirs('/tmp/models', exist_ok=True)
        shutil.copy(model_path, registry_path)
    
        logger.info(f"モデル登録完了: {registry_path}")
        return registry_path
    
    @flow(
        name="ML Training Pipeline",
        description="完全なML訓練パイプライン",
        task_runner=ConcurrentTaskRunner(),  # 並列実行
        log_prints=True
    )
    def ml_training_pipeline(n_estimators: int = 100):
        """メインのMLパイプライン"""
        logger = get_run_logger()
        logger.info("パイプライン開始")
    
        # タスク実行
        data_path = collect_data()
        validate_data(data_path)
        data_paths = preprocess_data(data_path)
        model_path = train_model(data_paths, n_estimators)
        metrics = evaluate_model(model_path, data_paths)
        registry_path = register_model(model_path, metrics)
    
        logger.info(f"パイプライン完了: {registry_path}")
        return {
            'model_path': registry_path,
            'metrics': metrics
        }
    
    # デプロイメント定義
    if __name__ == "__main__":
        # ローカル実行
        result = ml_training_pipeline(n_estimators=100)
        print(f"結果: {result}")
    
        # デプロイメント作成（Prefect Cloudへ）
        """
        deployment = Deployment.build_from_flow(
            flow=ml_training_pipeline,
            name="daily-ml-training",
            schedule=CronSchedule(cron="0 2 * * *"),  # 毎日2:00 AM
            work_queue_name="ml-training",
            parameters={"n_estimators": 100}
        )
        deployment.apply()
        """
    

### Prefect Cloud統合
    
    
    """
    Prefect Cloud統合とデプロイメント
    """
    
    from prefect import flow, task
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import IntervalSchedule
    from datetime import timedelta
    
    @flow
    def production_ml_pipeline():
        """本番MLパイプライン"""
        # パイプライン処理...
        pass
    
    # デプロイメント設定
    deployment = Deployment.build_from_flow(
        flow=production_ml_pipeline,
        name="production-deployment",
        schedule=IntervalSchedule(interval=timedelta(hours=6)),  # 6時間ごと
        work_queue_name="production",
        tags=["ml", "production"],
        parameters={},
        description="本番環境のMLパイプライン"
    )
    
    # デプロイ
    # deployment.apply()
    
    # CLIでのデプロイ
    """
    # Prefect Cloudにログイン
    prefect cloud login
    
    # デプロイメント作成
    prefect deployment build ml_pipeline.py:production_ml_pipeline -n production -q production
    
    # デプロイメント適用
    prefect deployment apply production_ml_pipeline-deployment.yaml
    
    # エージェント起動
    prefect agent start -q production
    """
    

* * *

## 3.5 パイプライン設計のベストプラクティス

### 冪等性の確保

**冪等性（Idempotency）** とは、同じ入力で何度実行しても同じ結果が得られる性質です。
    
    
    """
    冪等性を確保するパターン
    """
    
    import os
    import shutil
    from pathlib import Path
    
    # ❌ 非冪等的な処理
    def non_idempotent_process(output_path):
        """既存データに追記（実行のたびに結果が変わる）"""
        with open(output_path, 'a') as f:  # append mode
            f.write("new data\n")
    
    # ✅ 冪等的な処理
    def idempotent_process(output_path):
        """既存データを上書き（常に同じ結果）"""
        if os.path.exists(output_path):
            os.remove(output_path)  # 既存ファイルを削除
    
        with open(output_path, 'w') as f:  # write mode
            f.write("new data\n")
    
    # ディレクトリの冪等的な作成
    def create_output_dir(dir_path):
        """ディレクトリを冪等的に作成"""
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # 既存ディレクトリを削除
        os.makedirs(dir_path)
    
    # タイムスタンプを含む冪等的な処理
    def process_with_version(input_path, output_dir, version):
        """バージョン管理で冪等性を確保"""
        output_path = os.path.join(output_dir, f'output_v{version}.csv')
    
        # 同じバージョンは常に同じ結果
        if os.path.exists(output_path):
            os.remove(output_path)
    
        # 処理実行
        process_data(input_path, output_path)
    
    # データベースの冪等的な更新
    def upsert_data(data, table_name):
        """UPSERT（存在すれば更新、なければ挿入）"""
        # SQL例
        query = f"""
        INSERT INTO {table_name} (id, value)
        VALUES (%(id)s, %(value)s)
        ON CONFLICT (id) DO UPDATE
        SET value = EXCLUDED.value
        """
        # 実行...
    

### エラーハンドリング
    
    
    """
    堅牢なエラーハンドリング
    """
    
    from typing import Optional
    import logging
    import time
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # リトライデコレータ
    def retry_on_failure(max_retries=3, delay=5, backoff=2):
        """失敗時にリトライするデコレータ"""
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
                            logger.error(f"{func.__name__} 失敗（最大リトライ到達）: {e}")
                            raise
    
                        logger.warning(
                            f"{func.__name__} 失敗（{retries}/{max_retries}）: {e}. "
                            f"{current_delay}秒後にリトライ..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff  # 指数バックオフ
    
            return wrapper
        return decorator
    
    # 使用例
    @retry_on_failure(max_retries=3, delay=5, backoff=2)
    def fetch_data_from_api(url):
        """APIからデータ取得（リトライあり）"""
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    # タスクレベルのエラーハンドリング
    def safe_task_execution(task_func, *args, **kwargs):
        """タスクを安全に実行"""
        try:
            logger.info(f"タスク開始: {task_func.__name__}")
            result = task_func(*args, **kwargs)
            logger.info(f"タスク成功: {task_func.__name__}")
            return result, None
    
        except Exception as e:
            logger.error(f"タスク失敗: {task_func.__name__} - {e}", exc_info=True)
            return None, str(e)
    
    # パイプラインレベルのエラーハンドリング
    def run_pipeline_with_recovery(tasks):
        """リカバリー機能付きパイプライン実行"""
        results = {}
        failed_tasks = []
    
        for task_name, task_func in tasks.items():
            result, error = safe_task_execution(task_func)
    
            if error:
                failed_tasks.append({
                    'task': task_name,
                    'error': error
                })
                # 重要タスクは失敗時に中断
                if is_critical_task(task_name):
                    logger.error(f"重要タスク失敗: {task_name}. パイプライン中断")
                    break
            else:
                results[task_name] = result
    
        # 失敗サマリー
        if failed_tasks:
            logger.warning(f"失敗タスク数: {len(failed_tasks)}")
            for failure in failed_tasks:
                logger.warning(f"  - {failure['task']}: {failure['error']}")
    
        return results, failed_tasks
    
    def is_critical_task(task_name):
        """重要タスクの判定"""
        critical_tasks = ['data_validation', 'model_training']
        return task_name in critical_tasks
    

### パラメータ化
    
    
    """
    設定のパラメータ化
    """
    
    import yaml
    import json
    from dataclasses import dataclass
    from typing import Dict, Any
    
    # データクラスでの設定管理
    @dataclass
    class PipelineConfig:
        """パイプライン設定"""
        data_source: str
        output_dir: str
        model_type: str
        n_estimators: int = 100
        test_size: float = 0.2
        random_state: int = 42
    
    # YAML設定ファイル
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
        """YAML設定ファイル読み込み"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run_pipeline_with_config(config_path: str):
        """設定ファイルを使用したパイプライン実行"""
        config = load_config(config_path)
    
        # 設定取得
        pipeline_config = PipelineConfig(**config['pipeline'])
        hyperparams = config['hyperparameters']
    
        # パイプライン実行
        print(f"データソース: {pipeline_config.data_source}")
        print(f"モデルタイプ: {pipeline_config.model_type}")
        print(f"ハイパーパラメータ: {hyperparams}")
    
        # 処理...
    
    # 環境変数からの設定読み込み
    import os
    
    def get_config_from_env():
        """環境変数から設定を取得"""
        config = {
            'data_source': os.getenv('DATA_SOURCE', 'default.csv'),
            'model_type': os.getenv('MODEL_TYPE', 'random_forest'),
            'n_estimators': int(os.getenv('N_ESTIMATORS', '100')),
            'output_dir': os.getenv('OUTPUT_DIR', '/tmp/output')
        }
        return config
    
    # コマンドライン引数
    import argparse
    
    def parse_args():
        """コマンドライン引数のパース"""
        parser = argparse.ArgumentParser(description='ML Pipeline')
    
        parser.add_argument('--config', type=str, required=True,
                           help='設定ファイルのパス')
        parser.add_argument('--data-source', type=str,
                           help='データソース（設定ファイルを上書き）')
        parser.add_argument('--n-estimators', type=int, default=100,
                           help='決定木の数')
    
        return parser.parse_args()
    
    if __name__ == '__main__':
        args = parse_args()
    
        # 設定ファイル読み込み
        config = load_config(args.config)
    
        # コマンドライン引数で上書き
        if args.data_source:
            config['pipeline']['data_source'] = args.data_source
        if args.n_estimators:
            config['pipeline']['n_estimators'] = args.n_estimators
    
        # パイプライン実行
        run_pipeline_with_config(args.config)
    

### テスタビリティ
    
    
    """
    テスト可能なパイプライン設計
    """
    
    import unittest
    from unittest.mock import Mock, patch
    import pandas as pd
    
    # テスト可能な関数設計
    def load_data(data_source):
        """データ読み込み（テスト容易）"""
        # 実装...
        pass
    
    def preprocess(data):
        """前処理（pure function）"""
        # 副作用なし、入力から出力を生成
        processed = data.copy()
        # 処理...
        return processed
    
    def train_model(X, y, model_class, **hyperparams):
        """モデル訓練（依存性注入）"""
        model = model_class(**hyperparams)
        model.fit(X, y)
        return model
    
    # ユニットテスト
    class TestPreprocessing(unittest.TestCase):
        """前処理のテスト"""
    
        def test_preprocess_removes_nulls(self):
            """欠損値が削除されるかテスト"""
            # テストデータ
            data = pd.DataFrame({
                'feature1': [1, 2, None, 4],
                'feature2': [5, None, 7, 8]
            })
    
            # 実行
            result = preprocess(data)
    
            # 検証
            self.assertEqual(result.isnull().sum().sum(), 0)
    
        def test_preprocess_scales_features(self):
            """特徴量がスケーリングされるかテスト"""
            data = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [10, 20, 30, 40]
            })
    
            result = preprocess(data)
    
            # 平均が0に近い
            self.assertAlmostEqual(result['feature1'].mean(), 0, places=1)
    
    # モックを使用した統合テスト
    class TestMLPipeline(unittest.TestCase):
        """パイプライン全体のテスト"""
    
        @patch('my_pipeline.load_data')
        @patch('my_pipeline.save_model')
        def test_pipeline_end_to_end(self, mock_save, mock_load):
            """エンドツーエンドのパイプラインテスト"""
            # モックデータ
            mock_data = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'target': [0, 1, 0]
            })
            mock_load.return_value = mock_data
    
            # パイプライン実行
            # run_pipeline(config)
    
            # 検証
            mock_save.assert_called_once()
    
    # データバリデーション
    def validate_pipeline_output(output_path):
        """パイプライン出力の検証"""
        import os
    
        checks = {
            'file_exists': os.path.exists(output_path),
            'file_size': os.path.getsize(output_path) > 0 if os.path.exists(output_path) else False
        }
    
        assert all(checks.values()), f"出力検証失敗: {checks}"
        return True
    
    if __name__ == '__main__':
        unittest.main()
    

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **パイプライン設計の原則**

     * DAG構造でタスク依存を表現
     * 冪等性、再実行可能性、疎結合
     * パラメータ化と可観測性
  2. **Apache Airflow**

     * Pythonベースのワークフローオーケストレーション
     * スケジューラとExecutorアーキテクチャ
     * 豊富なオペレータとUI
  3. **Kubeflow Pipelines**

     * Kubernetesネイティブなパイプライン
     * コンテナ化されたコンポーネント
     * 再利用可能な部品設計
  4. **Prefect**

     * Pythonicな動的ワークフロー
     * 柔軟なエラーハンドリング
     * ローカル開発とCloud統合
  5. **ベストプラクティス**

     * 冪等性の確保と安全な再実行
     * 堅牢なエラーハンドリングとリトライ
     * 設定のパラメータ化
     * テスト可能な設計

### ツール比較

ツール | 強み | 適用場面  
---|---|---  
**Airflow** | 成熟したエコシステム、豊富なオペレータ | 複雑なバッチ処理、データETL  
**Kubeflow** | Kubernetesネイティブ、MLに特化 | 大規模ML、GPU活用、マルチクラウド  
**Prefect** | Pythonic、動的タスク、ローカル開発 | 柔軟なワークフロー、イベント駆動  
  
### 次の章へ

第4章では、**モデル管理とバージョニング** を学びます：

  * MLflowでのモデル追跡
  * モデルレジストリ
  * 実験管理とメトリクス記録
  * モデルのバージョン管理
  * 本番デプロイメント

* * *

## 演習問題

### 問題1（難易度：easy）

DAG（Directed Acyclic Graph）とは何か説明し、MLパイプラインでDAGが重要な理由を述べてください。

解答例

**解答** ：

**DAG（有向非巡環グラフ）** は、ノード（タスク）とエッジ（依存関係）から構成され、以下の特性を持ちます：

  1. **有向（Directed）** : エッジに方向がある（タスクAからタスクBへ）
  2. **非巡環（Acyclic）** : ループがない（同じタスクに戻らない）

**MLパイプラインでDAGが重要な理由** ：

  * **依存関係の明確化** : タスク間の実行順序が視覚的に理解できる
  * **並列実行の最適化** : 依存のないタスクを並列実行可能
  * **再現性** : 同じDAGから常に同じ実行順序が保証される
  * **デバッグの容易さ** : 失敗箇所の特定と再実行が簡単
  * **スケーラビリティ** : 複雑なワークフローを管理可能

例：
    
    
    データ収集 → 前処理 → 特徴量エンジニアリング → モデル訓練 → 評価
    

この構造により、各ステップが独立して実行可能で、失敗時は該当ステップのみ再実行できます。

### 問題2（難易度：medium）

以下のAirflow DAGのコードに誤りがあります。問題点を指摘し、修正してください。
    
    
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
    

解答例

**問題点** ：

  1. `PythonOperator`に`dag`パラメータが指定されていない
  2. `schedule_interval`が定義されていない
  3. `default_args`が設定されていない（推奨）

**修正版** ：
    
    
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta
    
    # デフォルト引数を定義
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
    
    # DAG定義（schedule_intervalを追加）
    dag = DAG(
        'example',
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule_interval='@daily',  # 毎日実行
        catchup=False
    )
    
    # dag引数を追加
    task1 = PythonOperator(
        task_id='task_a',
        python_callable=task_a,
        dag=dag  # ✅ dag引数を追加
    )
    
    task2 = PythonOperator(
        task_id='task_b',
        python_callable=task_b,
        dag=dag  # ✅ dag引数を追加
    )
    
    # 依存関係
    task1 >> task2
    

**改善点** ：

  * `default_args`でリトライ設定を追加
  * `schedule_interval`で実行頻度を明示
  * `catchup=False`で過去の未実行分をスキップ

### 問題3（難易度：medium）

冪等性（Idempotency）とは何か説明し、以下の関数を冪等的に修正してください。
    
    
    def process_data(input_file, output_file):
        data = pd.read_csv(input_file)
        # 処理...
        processed_data = data * 2
    
        # 既存ファイルに追記
        with open(output_file, 'a') as f:
            processed_data.to_csv(f, index=False)
    

解答例

**冪等性とは** ：

同じ入力で何度実行しても、常に同じ結果が得られる性質です。MLパイプラインにおいて重要な理由：

  * **再実行の安全性** : 失敗時に安心して再実行できる
  * **予測可能性** : 結果が常に一貫している
  * **デバッグの容易さ** : 問題の再現が確実

**元のコードの問題** ：

追記モード（`'a'`）を使用しているため、実行するたびにデータが追加され、異なる結果になります。

**修正版（冪等的）** ：
    
    
    import os
    import pandas as pd
    
    def process_data(input_file, output_file):
        """冪等的なデータ処理"""
        # データ読み込み
        data = pd.read_csv(input_file)
    
        # 処理
        processed_data = data * 2
    
        # ✅ 既存ファイルを削除してから書き込み
        if os.path.exists(output_file):
            os.remove(output_file)
    
        # 新規書き込み（上書きモード）
        processed_data.to_csv(output_file, index=False)
    
        print(f"処理完了: {output_file}")
    
    # 別の方法: 一時ファイルとアトミックな移動
    import shutil
    import tempfile
    
    def process_data_atomic(input_file, output_file):
        """アトミックな書き込みで冪等性を確保"""
        data = pd.read_csv(input_file)
        processed_data = data * 2
    
        # 一時ファイルに書き込み
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            processed_data.to_csv(tmp_path, index=False)
    
        # アトミックに移動（既存ファイルを上書き）
        shutil.move(tmp_path, output_file)
        print(f"処理完了: {output_file}")
    

**検証** ：
    
    
    # 同じ入力で2回実行
    process_data('input.csv', 'output.csv')  # 1回目
    process_data('input.csv', 'output.csv')  # 2回目
    
    # output.csvは常に同じ内容（冪等性あり）
    

### 問題4（難易度：hard）

Prefectを使用して、複数のモデルを並列訓練し、最良のモデルを選択するパイプラインを作成してください。モデルは['random_forest', 'xgboost', 'lightgbm']の3種類とします。

解答例
    
    
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import ConcurrentTaskRunner
    from typing import List, Dict
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # ダミーデータ生成
    @task
    def generate_data():
        """データ生成"""
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
        """データ分割"""
        logger = get_run_logger()
    
        X = data.drop('target', axis=1)
        y = data['target']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
        logger.info(f"訓練データ: {len(X_train)}件, テストデータ: {len(X_test)}件")
    
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    @task
    def train_model(model_type: str, data: Dict):
        """個別モデル訓練"""
        logger = get_run_logger()
        logger.info(f"{model_type}モデル訓練開始")
    
        X_train = data['X_train']
        y_train = data['y_train']
    
        # モデル選択
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            # xgboostがない場合はRFで代替
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            except ImportError:
                logger.warning("XGBoost未インストール。RandomForestで代替")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            # lightgbmがない場合はRFで代替
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            except ImportError:
                logger.warning("LightGBM未インストール。RandomForestで代替")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")
    
        # 訓練
        model.fit(X_train, y_train)
    
        logger.info(f"{model_type}モデル訓練完了")
    
        return {
            'model_type': model_type,
            'model': model
        }
    
    @task
    def evaluate_model(model_info: Dict, data: Dict):
        """モデル評価"""
        logger = get_run_logger()
    
        model_type = model_info['model_type']
        model = model_info['model']
    
        X_test = data['X_test']
        y_test = data['y_test']
    
        # 予測と評価
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        result = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'model': model
        }
    
        logger.info(f"{model_type} - 精度: {accuracy:.4f}")
    
        return result
    
    @task
    def select_best_model(results: List[Dict]):
        """最良モデル選択"""
        logger = get_run_logger()
    
        # 精度で最良モデルを選択
        best_result = max(results, key=lambda x: x['accuracy'])
    
        logger.info(f"最良モデル: {best_result['model_type']} (精度: {best_result['accuracy']:.4f})")
    
        # 全モデルの比較
        logger.info("\n=== モデル比較 ===")
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            logger.info(f"{result['model_type']}: {result['accuracy']:.4f}")
    
        return best_result
    
    @task
    def save_best_model(best_result: Dict, output_path: str = '/tmp/best_model.pkl'):
        """最良モデル保存"""
        logger = get_run_logger()
    
        model_type = best_result['model_type']
        model = best_result['model']
    
        # モデル保存
        joblib.dump(model, output_path)
    
        logger.info(f"最良モデル保存: {output_path} ({model_type})")
    
        return {
            'model_type': model_type,
            'accuracy': best_result['accuracy'],
            'path': output_path
        }
    
    @flow(
        name="Multi-Model Training Pipeline",
        description="複数モデル並列訓練パイプライン",
        task_runner=ConcurrentTaskRunner()  # 並列実行
    )
    def multi_model_training_pipeline(
        model_types: List[str] = None,
        test_size: float = 0.2
    ):
        """複数モデル並列訓練パイプライン"""
        logger = get_run_logger()
    
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm']
    
        logger.info(f"訓練モデル: {model_types}")
    
        # データ準備
        data = generate_data()
        split_result = split_data(data, test_size)
    
        # 並列訓練
        trained_models = []
        for model_type in model_types:
            trained_model = train_model(model_type, split_result)
            trained_models.append(trained_model)
    
        # 並列評価
        results = []
        for trained_model in trained_models:
            result = evaluate_model(trained_model, split_result)
            results.append(result)
    
        # 最良モデル選択
        best_result = select_best_model(results)
    
        # 保存
        saved_info = save_best_model(best_result)
    
        logger.info(f"パイプライン完了: {saved_info}")
    
        return saved_info
    
    # 実行
    if __name__ == "__main__":
        result = multi_model_training_pipeline(
            model_types=['random_forest', 'xgboost', 'lightgbm'],
            test_size=0.2
        )
    
        print(f"\n最終結果: {result}")
    

**実行結果例** ：
    
    
    === モデル比較 ===
    lightgbm: 0.9250
    random_forest: 0.9200
    xgboost: 0.9150
    
    最良モデル: lightgbm (精度: 0.9250)
    最良モデル保存: /tmp/best_model.pkl (lightgbm)
    

### 問題5（難易度：hard）

以下の要件を満たすエラーハンドリング機能を実装してください： 1\. 最大3回までリトライ 2\. 指数バックオフ（1秒、2秒、4秒） 3\. 特定の例外のみリトライ（ValueError、ConnectionErrorなど） 4\. リトライ履歴のロギング

解答例
    
    
    import time
    import logging
    from functools import wraps
    from typing import Callable, Tuple, Type
    
    # ロギング設定
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
        リトライ機能付きデコレータ（指数バックオフ）
    
        Args:
            max_retries: 最大リトライ回数
            initial_delay: 初回リトライまでの待機時間（秒）
            backoff_factor: バックオフの倍率
            retryable_exceptions: リトライ対象の例外タプル
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                delay = initial_delay
    
                while True:
                    try:
                        # 関数実行
                        result = func(*args, **kwargs)
    
                        # 成功時のログ
                        if retries > 0:
                            logger.info(
                                f"{func.__name__} 成功（{retries}回リトライ後）"
                            )
    
                        return result
    
                    except retryable_exceptions as e:
                        retries += 1
    
                        # 最大リトライ到達
                        if retries > max_retries:
                            logger.error(
                                f"{func.__name__} 失敗: 最大リトライ回数（{max_retries}）到達"
                            )
                            logger.error(f"最終エラー: {type(e).__name__}: {e}")
                            raise
    
                        # リトライログ
                        logger.warning(
                            f"{func.__name__} 失敗（{retries}/{max_retries}）: "
                            f"{type(e).__name__}: {e}"
                        )
                        logger.info(f"{delay:.1f}秒後にリトライ...")
    
                        # 待機
                        time.sleep(delay)
    
                        # 指数バックオフ
                        delay *= backoff_factor
    
                    except Exception as e:
                        # リトライ対象外の例外は即座に再送出
                        logger.error(
                            f"{func.__name__} 失敗（リトライ対象外）: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise
    
            return wrapper
        return decorator
    
    # 使用例1: API呼び出し
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    def fetch_data_from_api(url: str):
        """APIからデータ取得（リトライあり）"""
        import requests
    
        logger.info(f"API呼び出し: {url}")
    
        # 実際のAPI呼び出し
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    
        return response.json()
    
    # 使用例2: データベース接続
    @retry_with_backoff(
        max_retries=5,
        initial_delay=2.0,
        backoff_factor=2.0,
        retryable_exceptions=(ConnectionError,)
    )
    def connect_to_database(host: str, port: int):
        """データベース接続（リトライあり）"""
        logger.info(f"DB接続試行: {host}:{port}")
    
        # 実際のDB接続処理
        # connection = psycopg2.connect(host=host, port=port, ...)
    
        return "connection_object"
    
    # 使用例3: データ検証
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        retryable_exceptions=(ValueError,)
    )
    def validate_and_process_data(data):
        """データ検証と処理（リトライあり）"""
        logger.info("データ検証開始")
    
        # 検証
        if data is None:
            raise ValueError("データがNone")
    
        if len(data) < 100:
            raise ValueError(f"データ不足: {len(data)}件")
    
        # 処理
        processed = data * 2
    
        return processed
    
    # テスト関数
    def test_retry_mechanism():
        """リトライメカニズムのテスト"""
    
        # テスト1: 最終的に成功するケース
        attempt_count = 0
    
        @retry_with_backoff(max_retries=3, initial_delay=0.5)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
    
            if attempt_count < 3:
                raise ValueError(f"失敗 ({attempt_count}回目)")
    
            return "成功"
    
        print("\n=== テスト1: リトライ後に成功 ===")
        result = flaky_function()
        print(f"結果: {result}")
        print(f"試行回数: {attempt_count}")
    
        # テスト2: 最大リトライで失敗
        @retry_with_backoff(max_retries=2, initial_delay=0.5)
        def always_fails():
            raise ValueError("常に失敗")
    
        print("\n=== テスト2: 最大リトライで失敗 ===")
        try:
            always_fails()
        except ValueError as e:
            print(f"予期通り失敗: {e}")
    
        # テスト3: リトライ対象外の例外
        @retry_with_backoff(
            max_retries=3,
            retryable_exceptions=(ValueError,)
        )
        def non_retryable_error():
            raise RuntimeError("リトライ対象外のエラー")
    
        print("\n=== テスト3: リトライ対象外の例外 ===")
        try:
            non_retryable_error()
        except RuntimeError as e:
            print(f"即座に失敗: {e}")
    
    if __name__ == "__main__":
        test_retry_mechanism()
    

**実行例** ：
    
    
    === テスト1: リトライ後に成功 ===
    WARNING - flaky_function 失敗（1/3）: ValueError: 失敗 (1回目)
    INFO - 0.5秒後にリトライ...
    WARNING - flaky_function 失敗（2/3）: ValueError: 失敗 (2回目)
    INFO - 1.0秒後にリトライ...
    INFO - flaky_function 成功（2回リトライ後）
    結果: 成功
    試行回数: 3
    
    === テスト2: 最大リトライで失敗 ===
    WARNING - always_fails 失敗（1/2）: ValueError: 常に失敗
    INFO - 0.5秒後にリトライ...
    WARNING - always_fails 失敗（2/2）: ValueError: 常に失敗
    INFO - 1.0秒後にリトライ...
    ERROR - always_fails 失敗: 最大リトライ回数（2）到達
    ERROR - 最終エラー: ValueError: 常に失敗
    予期通り失敗: 常に失敗
    
    === テスト3: リトライ対象外の例外 ===
    ERROR - non_retryable_error 失敗（リトライ対象外）: RuntimeError: リトライ対象外のエラー
    即座に失敗: リトライ対象外のエラー
    

* * *

## 参考文献

  1. Apache Airflow Documentation. (2025). _Airflow Concepts_. Retrieved from https://airflow.apache.org/docs/
  2. Kubeflow Pipelines Documentation. (2025). _Building Pipelines_. Retrieved from https://www.kubeflow.org/docs/components/pipelines/
  3. Prefect Documentation. (2025). _Core Concepts_. Retrieved from https://docs.prefect.io/
  4. Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly Media.
  5. Lakshmanan, V., Robinson, S., & Munn, M. (2020). _Machine Learning Design Patterns_. O'Reilly Media.
