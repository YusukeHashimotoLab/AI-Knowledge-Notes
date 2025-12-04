---
title: "Chapter 3: Cloud Deployment"
chapter_title: "Chapter 3: Cloud Deployment"
subtitle: Building Scalable ML Systems with AWS, GCP, and Azure
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 8
exercises: 3
version: 1.0
created_at: 2025-10-23
---

This chapter covers Cloud Deployment. You will learn characteristics of major cloud platforms (AWS, Deploy models with AWS SageMaker, and basics of GCP Vertex AI.

## Learning Objectives

After reading this chapter, you will be able to:

  * ✅ Understand the characteristics of major cloud platforms (AWS, GCP, Azure)
  * ✅ Deploy models with AWS SageMaker
  * ✅ Build serverless inference environments with AWS Lambda
  * ✅ Understand the basics of GCP Vertex AI and Azure ML
  * ✅ Implement multi-cloud strategies with Terraform and CI/CD

* * *

## 3.1 Cloud Deployment Options

### Comparison of Major Cloud Platforms

Three major cloud platforms are primarily used for deploying machine learning models.

Platform | ML Services | Strengths | Use Cases  
---|---|---|---  
**AWS** | SageMaker, Lambda, ECS | Largest market share, extensive services | Enterprise, large-scale systems  
**GCP** | Vertex AI, Cloud Run | TensorFlow integration, BigQuery connectivity | Data analytics focus, startups  
**Azure** | Azure ML, Functions | Microsoft product integration | Enterprise (Microsoft environments)  
  
### Types of Deployment Services

Type | Description | AWS | GCP | Azure  
---|---|---|---|---  
**Managed** | Fully managed ML platform | SageMaker | Vertex AI | Azure ML  
**Serverless** | Event-driven, auto-scaling | Lambda | Cloud Functions | Azure Functions  
**Container** | Docker/Kubernetes-based | ECS/EKS | Cloud Run/GKE | AKS  
  
### Cost Considerations

Cloud deployment cost factors:

  * **Compute** : Instance type, uptime
  * **Storage** : Model files, log storage
  * **Network** : Data transfer volume
  * **Inference requests** : API call count

> **Cost Optimization Tips** : Auto-scaling, spot instances, and appropriate instance sizing are key.
    
    
    ```mermaid
    graph TD
        A[Deployment Strategy] --> B[Traffic Pattern]
        B --> C{Request Frequency}
        C -->|High frequency, stable| D[ManagedSageMaker/Vertex AI]
        C -->|Low frequency, irregular| E[ServerlessLambda/Cloud Functions]
        C -->|Burst handling| F[Auto-scalingECS/Cloud Run]
    
        A --> G[Cost Constraints]
        G --> H{Budget}
        H -->|Low budget| I[Serverless]
        H -->|Medium budget| J[Container]
        H -->|High budget| K[Dedicated Managed]
    
        style D fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#f3e5f5
    ```

* * *

## 3.2 AWS SageMaker Deployment

### What is a SageMaker Endpoint?

**Amazon SageMaker** is a managed service that integrates building, training, and deploying machine learning models.

### Model Packaging
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Model Packaging
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # model_package.py
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'model.joblib')
    print("✓ Model saved: model.joblib")
    

### Custom Inference Script
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    # inference.py
    import joblib
    import json
    import numpy as np
    
    def model_fn(model_dir):
        """Load the model"""
        model = joblib.load(f"{model_dir}/model.joblib")
        return model
    
    def input_fn(request_body, content_type):
        """Parse input data"""
        if content_type == 'application/json':
            data = json.loads(request_body)
            return np.array(data['instances'])
        raise ValueError(f"Unsupported content type: {content_type}")
    
    def predict_fn(input_data, model):
        """Execute inference"""
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    def output_fn(prediction, accept):
        """Format response"""
        if accept == 'application/json':
            return json.dumps(prediction), accept
        raise ValueError(f"Unsupported accept type: {accept}")
    

### Deploying to SageMaker
    
    
    import boto3
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel
    from datetime import datetime
    
    # Session configuration
    session = sagemaker.Session()
    role = 'arn:aws:iam::123456789012:role/SageMakerRole'
    bucket = session.default_bucket()
    
    # Upload model to S3
    model_data = session.upload_data(
        path='model.joblib',
        bucket=bucket,
        key_prefix='models/sklearn-model'
    )
    
    # Create SageMaker model
    sklearn_model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point='inference.py',
        framework_version='1.0-1',
        py_version='py3'
    )
    
    # Deploy endpoint
    endpoint_name = f'sklearn-endpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name
    )
    
    print(f"✓ Endpoint deployed: {endpoint_name}")
    print(f"✓ Instance type: ml.m5.large")
    print(f"✓ Instance count: 1")
    

### Executing Inference Requests
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Executing Inference Requests
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import boto3
    import json
    import numpy as np
    
    # SageMaker Runtime client
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Test data
    test_data = {
        'instances': np.random.randn(5, 10).tolist()
    }
    
    # Inference request
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(test_data)
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    print("\n=== Inference Results ===")
    print(f"Predictions: {result['predictions']}")
    print(f"Probabilities: {result['probabilities']}")
    
    # Performance information
    print(f"\nInference time: {response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-invocation-timestamp', 'N/A')}")
    

### Configuring Auto-Scaling
    
    
    import boto3
    
    # Auto Scaling client
    autoscaling = boto3.client('application-autoscaling', region_name='us-east-1')
    
    # Register scalable target
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    autoscaling.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=5
    )
    
    # Configure scaling policy
    autoscaling.put_scaling_policy(
        PolicyName=f'{endpoint_name}-scaling-policy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,  # Target CPU utilization 70%
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,   # Scale-in cooldown (seconds)
            'ScaleOutCooldown': 60    # Scale-out cooldown (seconds)
        }
    )
    
    print("✓ Auto-scaling configured")
    print(f"  Minimum instances: 1")
    print(f"  Maximum instances: 5")
    print(f"  Target metric: Requests/instance")
    

> **Best Practice** : In production, ensure availability with a minimum of 2 instances and adjust scale-out thresholds according to traffic patterns.

* * *

## 3.3 AWS Lambda Serverless Deployment

### Advantages of Serverless Architecture

  * **Cost efficiency** : Charged only for execution time
  * **Auto-scaling** : Automatically adjusts to concurrent executions
  * **Reduced operational burden** : No infrastructure management required
  * **High availability** : Automatic multi-AZ deployment

### Creating a Lambda Function
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    # lambda_function.py
    import json
    import joblib
    import numpy as np
    import base64
    import io
    
    # Load model in global scope (cold start optimization)
    model = None
    
    def load_model():
        """Load model (executed only once)"""
        global model
        if model is None:
            # Load model from S3 or include in layer
            model = joblib.load('/opt/model.joblib')
        return model
    
    def lambda_handler(event, context):
        """Lambda function main handler"""
        try:
            # Load model
            ml_model = load_model()
    
            # Parse request body
            body = json.loads(event.get('body', '{}'))
            instances = body.get('instances', [])
    
            if not instances:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No instances provided'})
                }
    
            # Execute inference
            input_data = np.array(instances)
            predictions = ml_model.predict(input_data)
            probabilities = ml_model.predict_proba(input_data)
    
            # Response
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
    

### Deploying with Container Image
    
    
    # Dockerfile
    FROM public.ecr.aws/lambda/python:3.9
    
    # Install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
    
    # Copy model file
    COPY model.joblib ${LAMBDA_TASK_ROOT}/opt/
    
    # Copy Lambda function code
    COPY lambda_function.py ${LAMBDA_TASK_ROOT}
    
    # Specify handler
    CMD ["lambda_function.lambda_handler"]
    
    
    
    #!/bin/bash
    # deploy.sh - Build and deploy Lambda container image
    
    # Variable configuration
    AWS_REGION="us-east-1"
    AWS_ACCOUNT_ID="123456789012"
    ECR_REPO="ml-inference-lambda"
    IMAGE_TAG="latest"
    
    # Create ECR repository (first time only)
    aws ecr create-repository \
        --repository-name ${ECR_REPO} \
        --region ${AWS_REGION} 2>/dev/null || true
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin \
        ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Build Docker image
    docker build -t ${ECR_REPO}:${IMAGE_TAG} .
    
    # Tag image
    docker tag ${ECR_REPO}:${IMAGE_TAG} \
        ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
    
    # Push to ECR
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
    
    echo "✓ Image pushed to ECR"
    

### Integration with API Gateway
    
    
    import boto3
    import json
    
    # Create API Gateway
    apigateway = boto3.client('apigateway', region_name='us-east-1')
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # Create REST API
    api = apigateway.create_rest_api(
        name='ML-Inference-API',
        description='Machine Learning Inference API',
        endpointConfiguration={'types': ['REGIONAL']}
    )
    api_id = api['id']
    
    # Get resources
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # Create /predict resource
    predict_resource = apigateway.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='predict'
    )
    
    # Create POST method
    apigateway.put_method(
        restApiId=api_id,
        resourceId=predict_resource['id'],
        httpMethod='POST',
        authorizationType='NONE'
    )
    
    # Configure Lambda integration
    lambda_arn = f"arn:aws:lambda:us-east-1:123456789012:function:ml-inference"
    apigateway.put_integration(
        restApiId=api_id,
        resourceId=predict_resource['id'],
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=f'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
    )
    
    # Deploy
    deployment = apigateway.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    
    endpoint_url = f"https://{api_id}.execute-api.us-east-1.amazonaws.com/prod/predict"
    print(f"✓ API Gateway deployed")
    print(f"✓ Endpoint: {endpoint_url}")
    

### Cold Start Mitigation

Methods to mitigate Lambda **cold start** (initial startup delay):

Method | Description | Effect  
---|---|---  
**Provisioned Concurrency** | Reserve always-on instances | Complete cold start avoidance  
**Model Optimization** | Lightweight models, quantization | Reduced load time  
**Layer Utilization** | Separate dependencies into layers | Reduced deployment package  
**Periodic Warmup** | Scheduled execution with EventBridge | Avoid idle state  
  
* * *

## 3.4 GCP Vertex AI and Azure ML

### GCP Vertex AI Endpoints

**Vertex AI** is Google's managed ML platform, featuring deep integration with TensorFlow.
    
    
    # vertex_ai_deploy.py
    from google.cloud import aiplatform
    
    # Initialize Vertex AI
    aiplatform.init(
        project='my-gcp-project',
        location='us-central1',
        staging_bucket='gs://my-ml-models'
    )
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name='sklearn-classifier',
        artifact_uri='gs://my-ml-models/sklearn-model',
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
    )
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name='sklearn-endpoint')
    
    # Deploy model
    endpoint.deploy(
        model=model,
        deployed_model_display_name='sklearn-v1',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=5,
        traffic_percentage=100
    )
    
    print(f"✓ Endpoint deployed: {endpoint.resource_name}")
    
    # Inference request
    instances = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    prediction = endpoint.predict(instances=instances)
    print(f"Prediction result: {prediction.predictions}")
    

### Azure ML Managed Endpoints

**Azure Machine Learning** is Microsoft Azure's managed machine learning service.
    
    
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
    
    # Initialize Azure ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id='subscription-id',
        resource_group_name='ml-resources',
        workspace_name='ml-workspace'
    )
    
    # Register model
    model = Model(
        path='./model',
        name='sklearn-classifier',
        description='Scikit-learn classification model'
    )
    registered_model = ml_client.models.create_or_update(model)
    
    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name='sklearn-endpoint',
        description='Sklearn classification endpoint',
        auth_mode='key'
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Create deployment
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
    
    # Allocate traffic
    endpoint.traffic = {'blue': 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print(f"✓ Azure ML endpoint deployed: {endpoint.name}")
    

### Cloud Platform Comparison

Feature | AWS SageMaker | GCP Vertex AI | Azure ML  
---|---|---|---  
**Deployment Method** | Endpoint, Lambda | Endpoint, Cloud Run | Managed Endpoint  
**Auto-scaling** | ◎ (Flexible) | ◎ (Automatic) | ○ (Configuration required)  
**Model Management** | Model Registry | Model Registry | Model Registry  
**Monitoring** | CloudWatch | Cloud Monitoring | Application Insights  
**Pricing Model** | Instance hours | Instance hours | Instance hours  
**Learning Curve** | Medium | Low (GCP experienced users) | Low (Azure experienced users)  
  
* * *

## 3.5 Practice: Multi-Cloud Strategy and CI/CD

### Infrastructure Management with Terraform

**Infrastructure as Code (IaC)** enables reproducible deployments.
    
    
    # terraform/main.tf - AWS SageMaker endpoint
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
    
    # SageMaker execution role
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
    
    # SageMaker model
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
    
    # SageMaker endpoint configuration
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
    
    # SageMaker endpoint
    resource "aws_sagemaker_endpoint" "endpoint" {
      name                 = "sklearn-endpoint-${var.environment}"
      endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config.name
    
      tags = {
        Environment = var.environment
        ManagedBy   = "Terraform"
      }
    }
    
    # Variable definitions
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
    
    # Outputs
    output "endpoint_name" {
      value = aws_sagemaker_endpoint.endpoint.name
    }
    
    output "endpoint_arn" {
      value = aws_sagemaker_endpoint.endpoint.arn
    }
    

### CI/CD with GitHub Actions
    
    
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
    

### Environment-Specific Deployment Strategy
    
    
    ```mermaid
    graph LR
        A[Code Changes] --> B[GitHub Push]
        B --> C{Branch}
        C -->|develop| D[Dev Environment]
        C -->|staging| E[Staging Environment]
        C -->|main| F[Production Environment]
    
        D --> G[Automated Tests]
        E --> H[Integration Tests]
        F --> I[Smoke Tests]
    
        G --> J[Auto Deploy]
        H --> K[Manual Approval]
        K --> L[Deploy]
        I --> M[Health Check]
    
        style D fill:#e8f5e9
        style E fill:#fff3e0
        style F fill:#ffebee
    ```

Environment | Purpose | Instance Count | Deployment Method  
---|---|---|---  
**Dev** | Development, testing | 1 | Automatic (develop branch)  
**Staging** | Integration testing, QA | 2 | Automatic (staging branch)  
**Production** | Production operation | 3+ (auto-scaling) | After manual approval  
  
* * *

## 3.6 Chapter Summary

### What We Learned

  1. **Choosing Cloud Platforms**

     * Characteristics and use cases of AWS, GCP, and Azure
     * Comparison of managed, serverless, and container options
     * Cost optimization considerations
  2. **AWS SageMaker**

     * Creating and deploying endpoints
     * Implementing custom inference scripts
     * Configuring auto-scaling
  3. **AWS Lambda Serverless**

     * Creating Lambda functions and container deployment
     * Integration with API Gateway
     * Cold start mitigation strategies
  4. **GCP Vertex AI and Azure ML**

     * Deploying Vertex AI Endpoints
     * Utilizing Azure ML Managed Endpoints
     * Cross-platform comparison
  5. **Multi-Cloud Strategy**

     * Infrastructure as Code with Terraform
     * CI/CD pipelines with GitHub Actions
     * Environment-specific deployment management

### Selection Guidelines

Requirements | Recommended Solution | Reason  
---|---|---  
High-frequency requests | SageMaker/Vertex AI | Low latency with dedicated instances  
Low-frequency, irregular | Lambda/Cloud Functions | Highly cost-efficient  
Burst handling | ECS/Cloud Run | Flexible scaling  
Multi-model | Kubernetes (EKS/GKE) | Unified management and resource efficiency  
TensorFlow-centric | GCP Vertex AI | Native integration  
Microsoft environment | Azure ML | Compatibility with existing systems  
  
### Next Chapter

In Chapter 4, we'll learn about **Monitoring and Operations Management** :

  * Performance monitoring
  * Log management and tracing
  * Model drift detection
  * A/B testing and canary deployment
  * Incident response

* * *

## Exercises

### Exercise 1 (Difficulty: medium)

For the following scenarios, determine whether to choose AWS SageMaker or AWS Lambda, and explain your reasoning.

**Scenario A** : E-commerce product recommendation system (100,000 requests/day, response time within 100ms)  
**Scenario B** : Batch processing for monthly report generation (once per month, 1-hour processing time)

Sample Answer

**Answer** :

**Scenario A: AWS SageMaker recommended**

  * **Reasoning** : 
    * High-frequency requests (100,000/day = approximately 1.2 requests/second) with stable traffic
    * Response time within 100ms is required, cold start is not acceptable
    * Dedicated instances with constant availability guarantee low latency
    * Auto-scaling can handle traffic peaks
  * **Configuration** : ml.m5.large × 2 instances (minimum), auto-scale up to 5 instances
  * **Cost** : Approximately $300-500/month (based on instance uptime)

**Scenario B: AWS Lambda recommended**

  * **Reasoning** : 
    * Low frequency (once per month), constant availability not required
    * Even with 1-hour processing time, can be addressed by dividing into Lambda (maximum 15 minutes) × 4 executions
    * Significant cost reduction with charges only for execution time
    * No strict response time requirements
  * **Configuration** : Lambda (3008MB memory), orchestration with Step Functions
  * **Cost** : Approximately $5-10/month (execution time only)

**Decision Criteria Summary** :

Factor | SageMaker | Lambda  
---|---|---  
Request frequency | High frequency, stable | Low frequency, irregular  
Latency requirements | Strict (< 100ms) | Relaxed (> 1s OK)  
Cost characteristics | High fixed cost | Pay-per-use  
Operational burden | Medium (scale management) | Low (fully managed)  
  
### Exercise 2 (Difficulty: hard)

Using Terraform and GitHub Actions, design a configuration to deploy SageMaker endpoints with different settings (instance count, type) for development and production environments.

Sample Answer

**Answer** :

**1\. Terraform variable files (by environment)**
    
    
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
    

**2\. Terraform main file**
    
    
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
    

**4\. Deployment Flow**
    
    
    ```mermaid
    graph TD
        A[Git Push] --> B{Branch Detection}
        B -->|develop| C[Dev Environmentml.t3.medium×1]
        B -->|main| D[Prod Environmentml.m5.xlarge×2]
        C --> E[Auto Deploy]
        D --> F[Awaiting Approval]
        F --> G[Manual Approval]
        G --> H[Execute Deploy]
        E --> I[Smoke Test]
        H --> I
    ```

**Configuration Key Points** :

  * Define different instance types and counts for each environment
  * Enable auto-scaling only for production environment
  * Add manual approval gate for production deployments
  * Environment separation with Terraform Workspaces

### Exercise 3 (Difficulty: hard)

Explain what methods should be combined to mitigate AWS Lambda's cold start problem, including specific implementations.

Sample Answer

**Answer** :

**1\. Configuring Provisioned Concurrency**
    
    
    import boto3
    
    lambda_client = boto3.client('lambda')
    
    # Configure provisioned concurrency
    lambda_client.put_provisioned_concurrency_config(
        FunctionName='ml-inference',
        Qualifier='$LATEST',  # Or version/alias
        ProvisionedConcurrentExecutions=5  # Always keep 5 instances running
    )
    
    print("✓ Provisioned concurrency configured (5 instances)")
    

**Effect** : Complete cold start avoidance with always-on instances  
**Cost** : Approximately 2x normal execution (constant charges)

**2\. Lightweight Model and Layer Separation**
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    """
    Example: 2. Lightweight Model and Layer Separation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # Model lightweighting
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Original model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    # Size: approximately 50MB
    
    # Lightweight (reduce number of trees)
    model_light = RandomForestClassifier(n_estimators=20, max_depth=8)
    # Size: approximately 10MB (80% reduction)
    
    # Quantization (optional)
    import onnx
    import onnxruntime
    # Reduce size with ONNX format quantization
    

**Utilizing Lambda Layers** :
    
    
    # Separate dependencies into layers
    mkdir -p layer/python/lib/python3.9/site-packages
    pip install scikit-learn numpy -t layer/python/lib/python3.9/site-packages
    cd layer
    zip -r layer.zip .
    
    # Publish layer
    aws lambda publish-layer-version \
        --layer-name ml-dependencies \
        --zip-file fileb://layer.zip \
        --compatible-runtimes python3.9
    

**Effect** : Reduced deployment package shortens cold start time (10MB → 1-2 seconds, 50MB → 5-10 seconds)

**3\. Periodic Warmup with EventBridge**
    
    
    import boto3
    
    events = boto3.client('events')
    lambda_arn = 'arn:aws:lambda:us-east-1:123456789012:function:ml-inference'
    
    # Create CloudWatch Events rule
    rule_response = events.put_rule(
        Name='lambda-warmup-rule',
        ScheduleExpression='rate(5 minutes)',  # Execute every 5 minutes
        State='ENABLED',
        Description='Keep Lambda warm to avoid cold starts'
    )
    
    # Set Lambda function as target
    events.put_targets(
        Rule='lambda-warmup-rule',
        Targets=[
            {
                'Id': '1',
                'Arn': lambda_arn,
                'Input': json.dumps({'warmup': True})  # Warmup flag
            }
        ]
    )
    
    print("✓ Warmup every 5 minutes configured")
    

**Lambda function modification** :
    
    
    def lambda_handler(event, context):
        # Detect warmup request
        if event.get('warmup'):
            print("Warmup request - keeping instance alive")
            return {'statusCode': 200, 'body': 'warmed up'}
    
        # Normal inference processing
        # ...
    

**Effect** : Avoid idle state (cost increase: approximately $1-5/month)

**4\. Optimal Combination Strategy**

Traffic Pattern | Recommended Method | Expected Effect  
---|---|---  
Constant high frequency (>10 req/s) | Provisioned concurrency | 0% cold starts  
Medium frequency (1-10 req/s) | Lightweighting + periodic warmup | <5% cold starts  
Low frequency (<1 req/s) | Lightweighting only | 1-2 second startup time  
Burst handling | All methods combined | Maximum performance  
  
**Implementation example (all methods integrated)** :
    
    
    # Integrated strategy
    # 1. Lightweight model (under 10MB)
    # 2. Lambda Layer utilization
    # 3. Provisioned concurrency (peak hours only)
    # 4. EventBridge warmup (5-minute intervals)
    
    # Cost estimate (assuming 100,000 requests/month)
    # - Regular Lambda: $5
    # - Provisioned: $30 (8 hours/day peak)
    # - Warmup: $3
    # - Total: approximately $40/month (70% reduction vs SageMaker)
    

* * *

## References

  1. Amazon Web Services. (2024). _Amazon SageMaker Developer Guide_. AWS Documentation.
  2. Google Cloud. (2024). _Vertex AI Documentation_. Google Cloud Documentation.
  3. Microsoft Azure. (2024). _Azure Machine Learning Documentation_. Microsoft Learn.
  4. HashiCorp. (2024). _Terraform AWS Provider Documentation_. Terraform Registry.
  5. Géron, A. (2022). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (3rd ed.). O'Reilly Media.
