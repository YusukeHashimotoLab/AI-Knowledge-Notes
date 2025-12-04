---
title: 第3章：クラウドデプロイメント
chapter_title: 第3章：クラウドデプロイメント
subtitle: AWS、GCP、Azureで実現するスケーラブルなMLシステム
reading_time: 30-35分
difficulty: 中級
code_examples: 8
exercises: 3
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 主要クラウドプラットフォーム（AWS、GCP、Azure）の特徴を理解する
  * ✅ AWS SageMakerでモデルをデプロイできる
  * ✅ AWS Lambdaでサーバーレス推論環境を構築できる
  * ✅ GCP Vertex AIとAzure MLの基本的な使い方を理解する
  * ✅ TerraformとCI/CDでマルチクラウド戦略を実装できる

* * *

## 3.1 クラウドデプロイメントの選択肢

### 主要クラウドプラットフォーム比較

機械学習モデルのデプロイメントには、主に3つの主要クラウドプラットフォームが使用されます。

プラットフォーム | MLサービス | 強み | ユースケース  
---|---|---|---  
**AWS** | SageMaker, Lambda, ECS | 最大のシェア、豊富なサービス | エンタープライズ、大規模システム  
**GCP** | Vertex AI, Cloud Run | TensorFlow統合、BigQuery連携 | データ分析重視、スタートアップ  
**Azure** | Azure ML, Functions | Microsoft製品統合 | エンタープライズ（Microsoft環境）  
  
### デプロイメントサービスの種類

種類 | 説明 | AWS | GCP | Azure  
---|---|---|---|---  
**マネージド** | フルマネージドMLプラットフォーム | SageMaker | Vertex AI | Azure ML  
**サーバーレス** | イベント駆動、自動スケール | Lambda | Cloud Functions | Azure Functions  
**コンテナ** | Docker/Kubernetes基盤 | ECS/EKS | Cloud Run/GKE | AKS  
  
### コスト考慮

クラウドデプロイメントのコスト要因：

  * **コンピューティング** : インスタンスタイプ、稼働時間
  * **ストレージ** : モデルファイル、ログ保存
  * **ネットワーク** : データ転送量
  * **推論リクエスト** : API呼び出し回数

> **コスト最適化のポイント** : オートスケーリング、スポットインスタンス、適切なインスタンスサイズ選択が重要です。
    
    
    ```mermaid
    graph TD
        A[デプロイメント戦略] --> B[トラフィックパターン]
        B --> C{リクエスト頻度}
        C -->|高頻度・安定| D[マネージドSageMaker/Vertex AI]
        C -->|低頻度・不規則| E[サーバーレスLambda/Cloud Functions]
        C -->|バースト対応| F[オートスケーリングECS/Cloud Run]
    
        A --> G[コスト制約]
        G --> H{予算}
        H -->|低予算| I[サーバーレス]
        H -->|中予算| J[コンテナ]
        H -->|高予算| K[専用マネージド]
    
        style D fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
    ```

* * *

## 3.2 AWS SageMakerデプロイメント

### SageMaker Endpointとは

**Amazon SageMaker** は、機械学習モデルの構築、訓練、デプロイを統合的に行うマネージドサービスです。

### モデルのパッケージング
    
    
    # model_package.py
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # モデルの訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    model.fit(X_train, y_train)
    
    # モデルの保存
    joblib.dump(model, 'model.joblib')
    print("✓ モデルを保存しました: model.joblib")
    

### カスタム推論スクリプト
    
    
    # inference.py
    import joblib
    import json
    import numpy as np
    
    def model_fn(model_dir):
        """モデルをロード"""
        model = joblib.load(f"{model_dir}/model.joblib")
        return model
    
    def input_fn(request_body, content_type):
        """入力データをパース"""
        if content_type == 'application/json':
            data = json.loads(request_body)
            return np.array(data['instances'])
        raise ValueError(f"Unsupported content type: {content_type}")
    
    def predict_fn(input_data, model):
        """推論実行"""
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    def output_fn(prediction, accept):
        """レスポンスをフォーマット"""
        if accept == 'application/json':
            return json.dumps(prediction), accept
        raise ValueError(f"Unsupported accept type: {accept}")
    

### SageMakerへのデプロイ
    
    
    import boto3
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel
    from datetime import datetime
    
    # セッション設定
    session = sagemaker.Session()
    role = 'arn:aws:iam::123456789012:role/SageMakerRole'
    bucket = session.default_bucket()
    
    # モデルをS3にアップロード
    model_data = session.upload_data(
        path='model.joblib',
        bucket=bucket,
        key_prefix='models/sklearn-model'
    )
    
    # SageMakerモデルの作成
    sklearn_model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point='inference.py',
        framework_version='1.0-1',
        py_version='py3'
    )
    
    # エンドポイントのデプロイ
    endpoint_name = f'sklearn-endpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name
    )
    
    print(f"✓ エンドポイントをデプロイしました: {endpoint_name}")
    print(f"✓ インスタンスタイプ: ml.m5.large")
    print(f"✓ インスタンス数: 1")
    

### 推論リクエストの実行
    
    
    import boto3
    import json
    import numpy as np
    
    # SageMaker Runtimeクライアント
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # テストデータ
    test_data = {
        'instances': np.random.randn(5, 10).tolist()
    }
    
    # 推論リクエスト
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(test_data)
    )
    
    # レスポンスのパース
    result = json.loads(response['Body'].read().decode())
    print("\n=== 推論結果 ===")
    print(f"予測: {result['predictions']}")
    print(f"確率: {result['probabilities']}")
    
    # パフォーマンス情報
    print(f"\n推論時間: {response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-invocation-timestamp', 'N/A')}")
    

### オートスケーリングの設定
    
    
    import boto3
    
    # Auto Scalingクライアント
    autoscaling = boto3.client('application-autoscaling', region_name='us-east-1')
    
    # スケーラブルターゲットの登録
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    autoscaling.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=5
    )
    
    # スケーリングポリシーの設定
    autoscaling.put_scaling_policy(
        PolicyName=f'{endpoint_name}-scaling-policy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,  # 目標CPU使用率70%
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,   # スケールイン待機時間（秒）
            'ScaleOutCooldown': 60    # スケールアウト待機時間（秒）
        }
    )
    
    print("✓ オートスケーリングを設定しました")
    print(f"  最小インスタンス数: 1")
    print(f"  最大インスタンス数: 5")
    print(f"  目標メトリック: リクエスト/インスタンス")
    

> **ベストプラクティス** : 本番環境では、最小2インスタンスで可用性を確保し、トラフィックパターンに応じてスケールアウト閾値を調整します。

* * *

## 3.3 AWS Lambdaサーバーレスデプロイメント

### サーバーレスアーキテクチャの利点

  * **コスト効率** : 実行時間分のみ課金
  * **自動スケーリング** : 同時実行数に応じて自動調整
  * **運用負荷軽減** : インフラ管理不要
  * **高可用性** : マルチAZ自動配置

### Lambda関数の作成
    
    
    # lambda_function.py
    import json
    import joblib
    import numpy as np
    import base64
    import io
    
    # グローバルスコープでモデルをロード（コールドスタート最適化）
    model = None
    
    def load_model():
        """モデルのロード（初回のみ実行）"""
        global model
        if model is None:
            # S3からモデルをロード、またはレイヤーに含める
            model = joblib.load('/opt/model.joblib')
        return model
    
    def lambda_handler(event, context):
        """Lambda関数のメインハンドラー"""
        try:
            # モデルのロード
            ml_model = load_model()
    
            # リクエストボディのパース
            body = json.loads(event.get('body', '{}'))
            instances = body.get('instances', [])
    
            if not instances:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No instances provided'})
                }
    
            # 推論実行
            input_data = np.array(instances)
            predictions = ml_model.predict(input_data)
            probabilities = ml_model.predict_proba(input_data)
    
            # レスポンス
            response = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'model_version': '1.0'
            }
    
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(response)
            }
    
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    

### コンテナイメージでのデプロイ
    
    
    # Dockerfile
    FROM public.ecr.aws/lambda/python:3.9
    
    # 依存関係のインストール
    COPY requirements.txt .
    RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
    
    # モデルファイルのコピー
    COPY model.joblib ${LAMBDA_TASK_ROOT}/opt/
    
    # Lambda関数コードのコピー
    COPY lambda_function.py ${LAMBDA_TASK_ROOT}
    
    # ハンドラーの指定
    CMD ["lambda_function.lambda_handler"]
    
    
    
    #!/bin/bash
    # deploy.sh - Lambdaコンテナイメージのビルドとデプロイ
    
    # 変数設定
    AWS_REGION="us-east-1"
    AWS_ACCOUNT_ID="123456789012"
    ECR_REPO="ml-inference-lambda"
    IMAGE_TAG="latest"
    
    # ECRリポジトリの作成（初回のみ）
    aws ecr create-repository \
        --repository-name ${ECR_REPO} \
        --region ${AWS_REGION} 2>/dev/null || true
    
    # ECRにログイン
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin \
        ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Dockerイメージのビルド
    docker build -t ${ECR_REPO}:${IMAGE_TAG} .
    
    # イメージのタグ付け
    docker tag ${ECR_REPO}:${IMAGE_TAG} \
        ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
    
    # ECRにプッシュ
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
    
    echo "✓ イメージをECRにプッシュしました"
    

### API Gatewayとの統合
    
    
    import boto3
    import json
    
    # API Gateway作成
    apigateway = boto3.client('apigateway', region_name='us-east-1')
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # REST APIの作成
    api = apigateway.create_rest_api(
        name='ML-Inference-API',
        description='Machine Learning Inference API',
        endpointConfiguration={'types': ['REGIONAL']}
    )
    api_id = api['id']
    
    # リソースの取得
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # /predictリソースの作成
    predict_resource = apigateway.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='predict'
    )
    
    # POSTメソッドの作成
    apigateway.put_method(
        restApiId=api_id,
        resourceId=predict_resource['id'],
        httpMethod='POST',
        authorizationType='NONE'
    )
    
    # Lambda統合の設定
    lambda_arn = f"arn:aws:lambda:us-east-1:123456789012:function:ml-inference"
    apigateway.put_integration(
        restApiId=api_id,
        resourceId=predict_resource['id'],
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=f'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
    )
    
    # デプロイ
    deployment = apigateway.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    
    endpoint_url = f"https://{api_id}.execute-api.us-east-1.amazonaws.com/prod/predict"
    print(f"✓ API Gatewayをデプロイしました")
    print(f"✓ エンドポイント: {endpoint_url}")
    

### コールドスタート対策

Lambdaの**コールドスタート** （初回起動の遅延）を軽減する方法：

手法 | 説明 | 効果  
---|---|---  
**プロビジョニング同時実行** | 常時起動インスタンスを確保 | コールドスタート完全回避  
**モデル最適化** | 軽量モデル、量子化 | ロード時間短縮  
**レイヤー活用** | 依存関係を別レイヤーに分離 | デプロイパッケージ削減  
**定期ウォームアップ** | EventBridgeで定期実行 | アイドル状態回避  
  
* * *

## 3.4 GCP Vertex AIとAzure ML

### GCP Vertex AI Endpoints

**Vertex AI** は、GoogleのマネージドMLプラットフォームで、TensorFlowとの深い統合が特徴です。
    
    
    # vertex_ai_deploy.py
    from google.cloud import aiplatform
    
    # Vertex AIの初期化
    aiplatform.init(
        project='my-gcp-project',
        location='us-central1',
        staging_bucket='gs://my-ml-models'
    )
    
    # モデルのアップロード
    model = aiplatform.Model.upload(
        display_name='sklearn-classifier',
        artifact_uri='gs://my-ml-models/sklearn-model',
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
    )
    
    # エンドポイントの作成
    endpoint = aiplatform.Endpoint.create(display_name='sklearn-endpoint')
    
    # モデルのデプロイ
    endpoint.deploy(
        model=model,
        deployed_model_display_name='sklearn-v1',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=5,
        traffic_percentage=100
    )
    
    print(f"✓ エンドポイントをデプロイしました: {endpoint.resource_name}")
    
    # 推論リクエスト
    instances = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    prediction = endpoint.predict(instances=instances)
    print(f"予測結果: {prediction.predictions}")
    

### Azure ML Managed Endpoints

**Azure Machine Learning** は、Microsoft Azureのマネージド機械学習サービスです。
    
    
    # azure_ml_deploy.py
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        Model,
        Environment,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    
    # Azure ML Clientの初期化
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id='subscription-id',
        resource_group_name='ml-resources',
        workspace_name='ml-workspace'
    )
    
    # モデルの登録
    model = Model(
        path='./model',
        name='sklearn-classifier',
        description='Scikit-learn classification model'
    )
    registered_model = ml_client.models.create_or_update(model)
    
    # エンドポイントの作成
    endpoint = ManagedOnlineEndpoint(
        name='sklearn-endpoint',
        description='Sklearn classification endpoint',
        auth_mode='key'
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # デプロイメントの作成
    deployment = ManagedOnlineDeployment(
        name='blue',
        endpoint_name='sklearn-endpoint',
        model=registered_model,
        environment='AzureML-sklearn-1.0-ubuntu20.04-py38-cpu',
        code_configuration=CodeConfiguration(
            code='./src',
            scoring_script='score.py'
        ),
        instance_type='Standard_DS3_v2',
        instance_count=1
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # トラフィックの割り当て
    endpoint.traffic = {'blue': 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print(f"✓ Azure MLエンドポイントをデプロイしました: {endpoint.name}")
    

### クラウドプラットフォーム比較

機能 | AWS SageMaker | GCP Vertex AI | Azure ML  
---|---|---|---  
**デプロイ方法** | Endpoint, Lambda | Endpoint, Cloud Run | Managed Endpoint  
**オートスケール** | ◎（柔軟） | ◎（自動） | ○（設定必要）  
**モデル管理** | Model Registry | Model Registry | Model Registry  
**モニタリング** | CloudWatch | Cloud Monitoring | Application Insights  
**料金体系** | インスタンス時間 | インスタンス時間 | インスタンス時間  
**学習コスト** | 中 | 低（GCP経験者） | 低（Azure経験者）  
  
* * *

## 3.5 実践：マルチクラウド戦略とCI/CD

### Terraformによるインフラ管理

**Infrastructure as Code（IaC）** で、再現可能なデプロイメントを実現します。
    
    
    # terraform/main.tf - AWS SageMakerエンドポイント
    terraform {
      required_providers {
        aws = {
          source  = "hashicorp/aws"
          version = "~> 5.0"
        }
      }
    }
    
    provider "aws" {
      region = var.aws_region
    }
    
    # SageMaker実行ロール
    resource "aws_iam_role" "sagemaker_role" {
      name = "sagemaker-execution-role"
    
      assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [{
          Action = "sts:AssumeRole"
          Effect = "Allow"
          Principal = {
            Service = "sagemaker.amazonaws.com"
          }
        }]
      })
    }
    
    # SageMakerモデル
    resource "aws_sagemaker_model" "ml_model" {
      name               = "sklearn-model-${var.environment}"
      execution_role_arn = aws_iam_role.sagemaker_role.arn
    
      primary_container {
        image          = var.container_image
        model_data_url = var.model_data_s3_uri
      }
    
      tags = {
        Environment = var.environment
        ManagedBy   = "Terraform"
      }
    }
    
    # SageMakerエンドポイント設定
    resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
      name = "sklearn-endpoint-config-${var.environment}"
    
      production_variants {
        variant_name           = "AllTraffic"
        model_name             = aws_sagemaker_model.ml_model.name
        initial_instance_count = var.initial_instance_count
        instance_type          = var.instance_type
      }
    
      tags = {
        Environment = var.environment
        ManagedBy   = "Terraform"
      }
    }
    
    # SageMakerエンドポイント
    resource "aws_sagemaker_endpoint" "endpoint" {
      name                 = "sklearn-endpoint-${var.environment}"
      endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config.name
    
      tags = {
        Environment = var.environment
        ManagedBy   = "Terraform"
      }
    }
    
    # 変数定義
    variable "aws_region" {
      default = "us-east-1"
    }
    
    variable "environment" {
      description = "Environment name (dev, staging, prod)"
      type        = string
    }
    
    variable "container_image" {
      description = "SageMaker container image URI"
      type        = string
    }
    
    variable "model_data_s3_uri" {
      description = "S3 URI of model artifacts"
      type        = string
    }
    
    variable "instance_type" {
      default = "ml.m5.large"
    }
    
    variable "initial_instance_count" {
      default = 1
    }
    
    # 出力
    output "endpoint_name" {
      value = aws_sagemaker_endpoint.endpoint.name
    }
    
    output "endpoint_arn" {
      value = aws_sagemaker_endpoint.endpoint.arn
    }
    

### GitHub ActionsによるCI/CD
    
    
    # .github/workflows/deploy-ml-model.yml
    name: Deploy ML Model
    
    on:
      push:
        branches:
          - main
          - develop
      pull_request:
        branches:
          - main
    
    env:
      AWS_REGION: us-east-1
      MODEL_BUCKET: ml-models-artifacts
    
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
    
          - name: Set up Python
            uses: actions/setup-python@v4
            with:
              python-version: '3.9'
    
          - name: Install dependencies
            run: |
              pip install -r requirements.txt
              pip install pytest pytest-cov
    
          - name: Run tests
            run: |
              pytest tests/ --cov=src --cov-report=xml
    
          - name: Upload coverage
            uses: codecov/codecov-action@v3
    
      build-and-deploy:
        needs: test
        runs-on: ubuntu-latest
        if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    
        steps:
          - uses: actions/checkout@v3
    
          - name: Set environment
            run: |
              if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
                echo "ENVIRONMENT=prod" >> $GITHUB_ENV
              else
                echo "ENVIRONMENT=dev" >> $GITHUB_ENV
              fi
    
          - name: Configure AWS credentials
            uses: aws-actions/configure-aws-credentials@v2
            with:
              aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
              aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
              aws-region: ${{ env.AWS_REGION }}
    
          - name: Train and package model
            run: |
              python src/train.py
              tar -czf model.tar.gz model.joblib
    
          - name: Upload model to S3
            run: |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)
              aws s3 cp model.tar.gz \
                s3://${{ env.MODEL_BUCKET }}/${{ env.ENVIRONMENT }}/model-${TIMESTAMP}.tar.gz
              echo "MODEL_S3_URI=s3://${{ env.MODEL_BUCKET }}/${{ env.ENVIRONMENT }}/model-${TIMESTAMP}.tar.gz" >> $GITHUB_ENV
    
          - name: Build Docker image
            run: |
              docker build -t ml-inference:${{ github.sha }} .
    
          - name: Push to ECR
            run: |
              aws ecr get-login-password --region ${{ env.AWS_REGION }} | \
                docker login --username AWS --password-stdin \
                ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com
    
              docker tag ml-inference:${{ github.sha }} \
                ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/ml-inference:${{ github.sha }}
    
              docker push \
                ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/ml-inference:${{ github.sha }}
    
          - name: Setup Terraform
            uses: hashicorp/setup-terraform@v2
    
          - name: Terraform Init
            working-directory: ./terraform
            run: terraform init
    
          - name: Terraform Plan
            working-directory: ./terraform
            run: |
              terraform plan \
                -var="environment=${{ env.ENVIRONMENT }}" \
                -var="model_data_s3_uri=${{ env.MODEL_S3_URI }}" \
                -var="container_image=${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/ml-inference:${{ github.sha }}" \
                -out=tfplan
    
          - name: Terraform Apply
            if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
            working-directory: ./terraform
            run: terraform apply -auto-approve tfplan
    
          - name: Smoke test
            run: |
              ENDPOINT_NAME=$(terraform -chdir=./terraform output -raw endpoint_name)
              python scripts/smoke_test.py --endpoint-name $ENDPOINT_NAME
    
      notify:
        needs: build-and-deploy
        runs-on: ubuntu-latest
        if: always()
        steps:
          - name: Send Slack notification
            uses: 8398a7/action-slack@v3
            with:
              status: ${{ job.status }}
              text: 'ML Model deployment ${{ job.status }}'
              webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    

### 環境別デプロイメント戦略
    
    
    ```mermaid
    graph LR
        A[コード変更] --> B[GitHub Push]
        B --> C{ブランチ}
        C -->|develop| D[Dev環境]
        C -->|staging| E[Staging環境]
        C -->|main| F[Production環境]
    
        D --> G[自動テスト]
        E --> H[統合テスト]
        F --> I[スモークテスト]
    
        G --> J[自動デプロイ]
        H --> K[手動承認]
        K --> L[デプロイ]
        I --> M[ヘルスチェック]
    
        style D fill:#e8f5e9
        style E fill:#fff3e0
        style F fill:#ffebee
    ```

環境 | 用途 | インスタンス数 | デプロイ方法  
---|---|---|---  
**Dev** | 開発・テスト | 1 | 自動（developブランチ）  
**Staging** | 統合テスト・QA | 2 | 自動（stagingブランチ）  
**Production** | 本番運用 | 3+（オートスケール） | 手動承認後  
  
* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **クラウドプラットフォームの選択**

     * AWS、GCP、Azureの特徴と使い分け
     * マネージド、サーバーレス、コンテナの比較
     * コスト最適化の考慮事項
  2. **AWS SageMaker**

     * エンドポイントの作成とデプロイ
     * カスタム推論スクリプトの実装
     * オートスケーリングの設定
  3. **AWS Lambdaサーバーレス**

     * Lambda関数の作成とコンテナデプロイ
     * API Gatewayとの統合
     * コールドスタート対策
  4. **GCP Vertex AIとAzure ML**

     * Vertex AI Endpointsのデプロイ
     * Azure ML Managed Endpointsの活用
     * プラットフォーム間の比較
  5. **マルチクラウド戦略**

     * Terraformによるインフラコード化
     * GitHub ActionsでのCI/CDパイプライン
     * 環境別デプロイメント管理

### 選択ガイドライン

要件 | 推奨ソリューション | 理由  
---|---|---  
高頻度リクエスト | SageMaker/Vertex AI | 専用インスタンスで低レイテンシ  
低頻度・不規則 | Lambda/Cloud Functions | コスト効率が高い  
バースト対応 | ECS/Cloud Run | 柔軟なスケーリング  
マルチモデル | Kubernetes (EKS/GKE) | 統一管理とリソース効率  
TensorFlow中心 | GCP Vertex AI | ネイティブ統合  
Microsoft環境 | Azure ML | 既存システムとの親和性  
  
### 次の章へ

第4章では、**モニタリングと運用管理** を学びます：

  * パフォーマンスモニタリング
  * ログ管理とトレーシング
  * モデルドリフト検出
  * A/Bテストとカナリアデプロイ
  * インシデント対応

* * *

## 演習問題

### 問題1（難易度：medium）

AWS SageMakerとAWS Lambdaのどちらを選ぶべきか、以下のシナリオで判断し、理由を説明してください。

**シナリオA** : ECサイトの商品推薦システム（1日10万リクエスト、レスポンス時間100ms以内）  
**シナリオB** : バッチ処理による月次レポート生成（月1回、処理時間1時間）

解答例

**解答** ：

**シナリオA：AWS SageMaker推奨**

  * **理由** : 
    * 高頻度リクエスト（1日10万 = 1秒あたり約1.2リクエスト）で安定したトラフィック
    * レスポンス時間100ms以内が求められ、コールドスタートは許容できない
    * 専用インスタンスで常時稼働により、低レイテンシを保証
    * オートスケーリングでトラフィックのピークに対応可能
  * **構成** : ml.m5.large × 2インスタンス（最小）、オートスケール最大5インスタンス
  * **コスト** : 月額約$300-500（インスタンス稼働時間ベース）

**シナリオB：AWS Lambda推奨**

  * **理由** : 
    * 低頻度（月1回）の実行で、常時稼働は不要
    * 処理時間1時間でも、Lambda（最大15分）× 4回の分割実行で対応可能
    * 実行時間のみ課金で、大幅なコスト削減
    * レスポンス時間の厳しい要件がない
  * **構成** : Lambda（メモリ3008MB）、Step Functionsで処理オーケストレーション
  * **コスト** : 月額約$5-10（実行時間のみ）

**判断基準まとめ** ：

要因 | SageMaker | Lambda  
---|---|---  
リクエスト頻度 | 高頻度・安定 | 低頻度・不規則  
レイテンシ要件 | 厳しい（< 100ms） | 緩い（> 1秒OK）  
コスト特性 | 固定コスト高 | 従量課金  
運用負荷 | 中（スケール管理） | 低（フルマネージド）  
  
### 問題2（難易度：hard）

TerraformとGitHub Actionsを使用して、開発環境と本番環境で異なる設定（インスタンス数、タイプ）を持つSageMakerエンドポイントをデプロイする構成を設計してください。

解答例

**解答** ：

**1\. Terraform変数ファイル（環境別）**
    
    
    # terraform/environments/dev.tfvars
    environment            = "dev"
    instance_type          = "ml.t3.medium"
    initial_instance_count = 1
    min_capacity           = 1
    max_capacity           = 2
    enable_autoscaling     = false
    
    # terraform/environments/prod.tfvars
    environment            = "prod"
    instance_type          = "ml.m5.xlarge"
    initial_instance_count = 2
    min_capacity           = 2
    max_capacity           = 10
    enable_autoscaling     = true
    

**2\. Terraformメインファイル**
    
    
    # terraform/main.tf
    resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
      name = "sklearn-endpoint-config-${var.environment}"
    
      production_variants {
        variant_name           = "AllTraffic"
        model_name             = aws_sagemaker_model.ml_model.name
        initial_instance_count = var.initial_instance_count
        instance_type          = var.instance_type
      }
    }
    
    resource "aws_appautoscaling_target" "sagemaker_target" {
      count              = var.enable_autoscaling ? 1 : 0
      service_namespace  = "sagemaker"
      resource_id        = "endpoint/${aws_sagemaker_endpoint.endpoint.name}/variant/AllTraffic"
      scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
      min_capacity       = var.min_capacity
      max_capacity       = var.max_capacity
    }
    
    resource "aws_appautoscaling_policy" "sagemaker_policy" {
      count              = var.enable_autoscaling ? 1 : 0
      name               = "sagemaker-scaling-policy-${var.environment}"
      service_namespace  = "sagemaker"
      resource_id        = aws_appautoscaling_target.sagemaker_target[0].resource_id
      scalable_dimension = aws_appautoscaling_target.sagemaker_target[0].scalable_dimension
      policy_type        = "TargetTrackingScaling"
    
      target_tracking_scaling_policy_configuration {
        predefined_metric_specification {
          predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
        }
        target_value = 1000.0
      }
    }
    

**3\. GitHub Actions Workflow**
    
    
    # .github/workflows/deploy.yml
    jobs:
      deploy:
        steps:
          - name: Set environment variables
            run: |
              if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
                echo "TFVARS_FILE=prod.tfvars" >> $GITHUB_ENV
                echo "REQUIRE_APPROVAL=true" >> $GITHUB_ENV
              else
                echo "TFVARS_FILE=dev.tfvars" >> $GITHUB_ENV
                echo "REQUIRE_APPROVAL=false" >> $GITHUB_ENV
              fi
    
          - name: Terraform Plan
            working-directory: ./terraform
            run: |
              terraform plan \
                -var-file="environments/${{ env.TFVARS_FILE }}" \
                -out=tfplan
    
          - name: Wait for approval (prod only)
            if: env.REQUIRE_APPROVAL == 'true'
            uses: trstringer/manual-approval@v1
            with:
              approvers: platform-team
              minimum-approvals: 2
    
          - name: Terraform Apply
            working-directory: ./terraform
            run: terraform apply -auto-approve tfplan
    

**4\. デプロイフロー**
    
    
    ```mermaid
    graph TD
        A[Git Push] --> B{ブランチ判定}
        B -->|develop| C[Dev環境設定ml.t3.medium×1]
        B -->|main| D[Prod環境設定ml.m5.xlarge×2]
        C --> E[自動デプロイ]
        D --> F[承認待機]
        F --> G[手動承認]
        G --> H[デプロイ実行]
        E --> I[スモークテスト]
        H --> I
    ```

**構成のポイント** ：

  * 環境ごとに異なるインスタンスタイプ・数を定義
  * 本番環境のみオートスケーリング有効化
  * 本番デプロイは手動承認ゲート追加
  * Terraform Workspaceで環境分離

### 問題3（難易度：hard）

AWS Lambdaのコールドスタート問題を軽減するために、どのような手法を組み合わせるべきか、具体的な実装を含めて説明してください。

解答例

**解答** ：

**1\. プロビジョニング同時実行の設定**
    
    
    import boto3
    
    lambda_client = boto3.client('lambda')
    
    # プロビジョニング同時実行の設定
    lambda_client.put_provisioned_concurrency_config(
        FunctionName='ml-inference',
        Qualifier='$LATEST',  # またはバージョン/エイリアス
        ProvisionedConcurrentExecutions=5  # 常時5インスタンスを起動
    )
    
    print("✓ プロビジョニング同時実行を設定しました（5インスタンス）")
    

**効果** : 常時起動インスタンスでコールドスタート完全回避  
**コスト** : 通常実行の約2倍（常時課金）

**2\. 軽量モデルとレイヤー分離**
    
    
    # モデルの軽量化
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # 元のモデル
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    # サイズ: 約50MB
    
    # 軽量化（木の数を削減）
    model_light = RandomForestClassifier(n_estimators=20, max_depth=8)
    # サイズ: 約10MB（80%削減）
    
    # 量子化（オプション）
    import onnx
    import onnxruntime
    # ONNX形式で量子化してサイズ削減
    

**Lambda Layerの活用** :
    
    
    # 依存関係をレイヤーに分離
    mkdir -p layer/python/lib/python3.9/site-packages
    pip install scikit-learn numpy -t layer/python/lib/python3.9/site-packages
    cd layer
    zip -r layer.zip .
    
    # レイヤーの公開
    aws lambda publish-layer-version \
        --layer-name ml-dependencies \
        --zip-file fileb://layer.zip \
        --compatible-runtimes python3.9
    

**効果** : デプロイパッケージ削減でコールドスタート時間短縮（10MB → 1-2秒、50MB → 5-10秒）

**3\. EventBridgeによる定期ウォームアップ**
    
    
    import boto3
    
    events = boto3.client('events')
    lambda_arn = 'arn:aws:lambda:us-east-1:123456789012:function:ml-inference'
    
    # CloudWatch Eventsルールの作成
    rule_response = events.put_rule(
        Name='lambda-warmup-rule',
        ScheduleExpression='rate(5 minutes)',  # 5分ごとに実行
        State='ENABLED',
        Description='Keep Lambda warm to avoid cold starts'
    )
    
    # Lambda関数をターゲットに設定
    events.put_targets(
        Rule='lambda-warmup-rule',
        Targets=[
            {
                'Id': '1',
                'Arn': lambda_arn,
                'Input': json.dumps({'warmup': True})  # ウォームアップフラグ
            }
        ]
    )
    
    print("✓ 5分ごとのウォームアップを設定しました")
    

**Lambda関数の修正** :
    
    
    def lambda_handler(event, context):
        # ウォームアップリクエストの判定
        if event.get('warmup'):
            print("Warmup request - keeping instance alive")
            return {'statusCode': 200, 'body': 'warmed up'}
    
        # 通常の推論処理
        # ...
    

**効果** : アイドル状態回避（コスト増加: 月額$1-5程度）

**4\. 最適な組み合わせ戦略**

トラフィックパターン | 推奨手法 | 期待効果  
---|---|---  
常時高頻度（>10 req/s） | プロビジョニング同時実行 | コールドスタート0%  
中頻度（1-10 req/s） | 軽量化 + 定期ウォームアップ | コールドスタート<5%  
低頻度（<1 req/s） | 軽量化のみ | 起動時間1-2秒  
バースト対応 | 全手法組み合わせ | 最大パフォーマンス  
  
**実装例（全手法統合）** :
    
    
    # 統合戦略
    # 1. 軽量モデル（10MB以下）
    # 2. Lambda Layer活用
    # 3. プロビジョニング同時実行（ピーク時間のみ）
    # 4. EventBridgeウォームアップ（5分間隔）
    
    # コスト試算（月間10万リクエスト想定）
    # - 通常Lambda: $5
    # - プロビジョニング: $30（ピーク8時間/日）
    # - ウォームアップ: $3
    # - 合計: 約$40/月（SageMaker比70%削減）
    

* * *

## 参考文献

  1. Amazon Web Services. (2024). _Amazon SageMaker Developer Guide_. AWS Documentation.
  2. Google Cloud. (2024). _Vertex AI Documentation_. Google Cloud Documentation.
  3. Microsoft Azure. (2024). _Azure Machine Learning Documentation_. Microsoft Learn.
  4. HashiCorp. (2024). _Terraform AWS Provider Documentation_. Terraform Registry.
  5. Géron, A. (2022). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (3rd ed.). O'Reilly Media.
