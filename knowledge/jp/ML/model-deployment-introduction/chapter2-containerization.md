---
title: 第2章：コンテナ化技術
chapter_title: 第2章：コンテナ化技術
subtitle: Dockerによる機械学習モデルの可搬性と再現性
reading_time: 25-30分
difficulty: 初級-中級
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Dockerコンテナの基礎概念と仮想マシンとの違いを理解する
  * ✅ 機械学習モデル用のDockerfileを作成できる
  * ✅ MLモデルを効率的にコンテナ化できる
  * ✅ Docker Composeで複数サービスを管理できる
  * ✅ GPU対応のMLコンテナを構築できる

* * *

## 2.1 Dockerの基礎

### コンテナとは

**コンテナ（Container）** は、アプリケーションとその依存関係を独立した環境にパッケージ化する技術です。

> 「Build once, Run anywhere」- 一度ビルドすれば、どこでも同じように実行できる

### Docker vs 仮想マシン

特徴 | Docker コンテナ | 仮想マシン (VM)  
---|---|---  
**起動時間** | 秒単位 | 分単位  
**リソース** | 軽量（MB単位） | 重い（GB単位）  
**分離レベル** | プロセスレベル | 完全な OS 分離  
**性能** | ネイティブに近い | オーバーヘッドあり  
**可搬性** | 高い | 中程度  
      
    
    ```mermaid
    graph TD
        subgraph "仮想マシン"
            A1[アプリ1] --> B1[ゲストOS1]
            A2[アプリ2] --> B2[ゲストOS2]
            B1 --> C[ハイパーバイザー]
            B2 --> C
            C --> D[ホストOS]
            D --> E[物理サーバー]
        end
    
        subgraph "Docker コンテナ"
            F1[アプリ1] --> G[Docker Engine]
            F2[アプリ2] --> G
            G --> H[ホストOS]
            H --> I[物理サーバー]
        end
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style F1 fill:#c8e6c9
        style F2 fill:#c8e6c9
    ```

### Docker基本コマンド
    
    
    # Dockerバージョン確認
    docker --version
    
    # イメージ一覧表示
    docker images
    
    # コンテナ一覧表示（実行中）
    docker ps
    
    # コンテナ一覧表示（全て）
    docker ps -a
    
    # イメージのダウンロード
    docker pull python:3.9-slim
    
    # コンテナの実行
    docker run -it python:3.9-slim bash
    
    # コンテナの停止
    docker stop <container_id>
    
    # コンテナの削除
    docker rm <container_id>
    
    # イメージの削除
    docker rmi <image_id>
    
    # システム全体のクリーンアップ
    docker system prune -a
    

### イメージとコンテナの関係

**イメージ（Image）** ：アプリケーションの設計図（読み取り専用）

**コンテナ（Container）** ：イメージから作成された実行可能なインスタンス
    
    
    ```mermaid
    graph LR
        A[Dockerfile] -->|docker build| B[Docker Image]
        B -->|docker run| C[Container 1]
        B -->|docker run| D[Container 2]
        B -->|docker run| E[Container 3]
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
    ```

> **重要** : 1つのイメージから複数のコンテナを起動できます。各コンテナは独立した環境です。

* * *

## 2.2 Dockerfileの作成

### ベースイメージ選択

機械学習モデル用の代表的なベースイメージ：

イメージ | サイズ | 用途  
---|---|---  
`python:3.9-slim` | 約120MB | 軽量なPython環境  
`python:3.9` | 約900MB | フル機能のPython環境  
`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` | 約2GB | GPU推論用  
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | 約4GB | GPU開発・学習用  
  
### 基本的なDockerfile構造
    
    
    # ベースイメージの指定
    FROM python:3.9-slim
    
    # 作業ディレクトリの設定
    WORKDIR /app
    
    # システムパッケージの更新とインストール
    RUN apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Pythonパッケージのインストール
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # アプリケーションコードのコピー
    COPY . .
    
    # ポート公開
    EXPOSE 8000
    
    # 起動コマンド
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### マルチステージビルド

イメージサイズを削減し、セキュリティを向上させる技術：
    
    
    # ステージ1: ビルド環境
    FROM python:3.9 as builder
    
    WORKDIR /build
    
    # 依存関係のインストール
    COPY requirements.txt .
    RUN pip install --user --no-cache-dir -r requirements.txt
    
    # ステージ2: 実行環境（軽量）
    FROM python:3.9-slim
    
    WORKDIR /app
    
    # ビルドステージから必要なファイルのみコピー
    COPY --from=builder /root/.local /root/.local
    COPY . .
    
    # PATHの設定
    ENV PATH=/root/.local/bin:$PATH
    
    EXPOSE 8000
    
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    

> **効果** : マルチステージビルドにより、イメージサイズを50-70%削減できることがあります。

### 最適化テクニック

#### レイヤーキャッシュの活用
    
    
    # ❌ 非効率: コードが変更されるたびに依存関係を再インストール
    FROM python:3.9-slim
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    
    # ✅ 効率的: 依存関係が変更されない限りキャッシュを利用
    FROM python:3.9-slim
    WORKDIR /app
    
    # 先に依存関係をインストール（変更頻度が低い）
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # 後からコードをコピー（変更頻度が高い）
    COPY . .
    

#### 不要なファイルの除外

.dockerignoreファイルの例：
    
    
    # .dockerignore
    __pycache__
    *.pyc
    *.pyo
    *.pyd
    .Python
    *.so
    *.egg
    *.egg-info
    dist
    build
    .git
    .gitignore
    .env
    .venv
    venv/
    data/
    notebooks/
    tests/
    *.md
    Dockerfile
    docker-compose.yml
    

* * *

## 2.3 MLモデルのコンテナ化

### FastAPI + PyTorchのDockerfile
    
    
    # マルチステージビルド
    FROM python:3.9 as builder
    
    WORKDIR /build
    
    # 依存関係ファイルのコピー
    COPY requirements.txt .
    
    # 依存関係のインストール
    RUN pip install --user --no-cache-dir \
        torch==2.0.0 \
        torchvision==0.15.0 \
        fastapi==0.104.0 \
        uvicorn[standard]==0.24.0 \
        pydantic==2.5.0 \
        pillow==10.1.0
    
    # 実行環境
    FROM python:3.9-slim
    
    WORKDIR /app
    
    # ビルドステージから依存関係をコピー
    COPY --from=builder /root/.local /root/.local
    
    # アプリケーションコードとモデルをコピー
    COPY app/ ./app/
    COPY models/ ./models/
    
    # 環境変数の設定
    ENV PATH=/root/.local/bin:$PATH \
        PYTHONUNBUFFERED=1 \
        MODEL_PATH=/app/models/model.pth
    
    # 非rootユーザーの作成（セキュリティ向上）
    RUN useradd -m -u 1000 appuser && \
        chown -R appuser:appuser /app
    
    USER appuser
    
    # ヘルスチェック
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD python -c "import requests; requests.get('http://localhost:8000/health')"
    
    EXPOSE 8000
    
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### requirements.txtの例
    
    
    # requirements.txt
    torch==2.0.0
    torchvision==0.15.0
    fastapi==0.104.0
    uvicorn[standard]==0.24.0
    pydantic==2.5.0
    pillow==10.1.0
    numpy==1.24.3
    python-multipart==0.0.6
    

### イメージビルドと実行
    
    
    # イメージのビルド
    docker build -t ml-api:v1.0 .
    
    # ビルドログの詳細表示
    docker build -t ml-api:v1.0 --progress=plain .
    
    # キャッシュを使わずにビルド
    docker build -t ml-api:v1.0 --no-cache .
    
    # コンテナの実行
    docker run -d \
        --name ml-api \
        -p 8000:8000 \
        -v $(pwd)/models:/app/models \
        ml-api:v1.0
    
    # ログの確認
    docker logs ml-api
    
    # リアルタイムログ表示
    docker logs -f ml-api
    
    # コンテナ内でコマンド実行
    docker exec -it ml-api bash
    
    # コンテナの停止と削除
    docker stop ml-api
    docker rm ml-api
    

### ポートマッピング

オプション | 説明 | 例  
---|---|---  
`-p 8000:8000` | ホスト:コンテナ | ホストの8000番をコンテナの8000番に  
`-p 8080:8000` | 異なるポート | ホストの8080番をコンテナの8000番に  
`-p 127.0.0.1:8000:8000` | ローカルのみ | ローカルホストからのみアクセス可能  
  
* * *

## 2.4 Docker Composeによるオーケストレーション

### docker-compose.yml構成

複数のサービスを統合管理するための設定ファイル：
    
    
    # docker-compose.yml
    version: '3.8'
    
    services:
      # FastAPI アプリケーション
      api:
        build:
          context: .
          dockerfile: Dockerfile
        container_name: ml-api
        ports:
          - "8000:8000"
        environment:
          - MODEL_PATH=/app/models/model.pth
          - REDIS_HOST=redis
          - REDIS_PORT=6379
        volumes:
          - ./models:/app/models:ro
          - ./logs:/app/logs
        depends_on:
          - redis
        restart: unless-stopped
        networks:
          - ml-network
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 40s
    
      # Redis キャッシュ
      redis:
        image: redis:7-alpine
        container_name: ml-redis
        ports:
          - "6379:6379"
        volumes:
          - redis-data:/data
        restart: unless-stopped
        networks:
          - ml-network
        command: redis-server --appendonly yes
    
    networks:
      ml-network:
        driver: bridge
    
    volumes:
      redis-data:
    

### 複数サービスの統合例
    
    
    # docker-compose.yml (拡張版)
    version: '3.8'
    
    services:
      # MLモデル推論API
      ml-api:
        build: ./api
        ports:
          - "8000:8000"
        environment:
          - REDIS_HOST=redis
          - DB_HOST=postgres
        volumes:
          - ./models:/app/models:ro
        depends_on:
          - redis
          - postgres
        networks:
          - ml-network
    
      # キャッシュ層
      redis:
        image: redis:7-alpine
        volumes:
          - redis-data:/data
        networks:
          - ml-network
    
      # データベース
      postgres:
        image: postgres:15-alpine
        environment:
          - POSTGRES_USER=mluser
          - POSTGRES_PASSWORD=mlpass
          - POSTGRES_DB=mldb
        volumes:
          - postgres-data:/var/lib/postgresql/data
        networks:
          - ml-network
    
      # モニタリング
      prometheus:
        image: prom/prometheus:latest
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
          - prometheus-data:/prometheus
        networks:
          - ml-network
    
    networks:
      ml-network:
        driver: bridge
    
    volumes:
      redis-data:
      postgres-data:
      prometheus-data:
    

### ボリュームマウント

タイプ | 構文 | 用途  
---|---|---  
**バインドマウント** | `./host/path:/container/path` | 開発時のコード同期  
**名前付きボリューム** | `volume-name:/container/path` | 永続的なデータ保存  
**読み取り専用** | `./path:/path:ro` | モデルファイルなど  
  
### 環境変数管理

.envファイルの例：
    
    
    # .env
    MODEL_PATH=/app/models/resnet50.pth
    REDIS_HOST=redis
    REDIS_PORT=6379
    LOG_LEVEL=INFO
    MAX_WORKERS=4
    

docker-compose.ymlでの使用：
    
    
    services:
      api:
        env_file:
          - .env
        # または個別に指定
        environment:
          - MODEL_PATH=${MODEL_PATH}
          - REDIS_HOST=${REDIS_HOST}
    

### Docker Compose コマンド
    
    
    # サービスの起動（バックグラウンド）
    docker-compose up -d
    
    # サービスの起動（ログ表示）
    docker-compose up
    
    # サービスのビルドと起動
    docker-compose up -d --build
    
    # 特定のサービスのみ起動
    docker-compose up -d api redis
    
    # サービスの停止
    docker-compose stop
    
    # サービスの停止と削除
    docker-compose down
    
    # ボリュームも含めて削除
    docker-compose down -v
    
    # ログの確認
    docker-compose logs -f
    
    # 特定のサービスのログ
    docker-compose logs -f api
    
    # サービスの状態確認
    docker-compose ps
    
    # サービスの再起動
    docker-compose restart api
    

* * *

## 2.5 実践: GPU対応MLコンテナ

### NVIDIA Dockerセットアップ

前提条件：

  * NVIDIA GPU搭載マシン
  * NVIDIAドライバーのインストール
  * NVIDIA Container Toolkitのインストール

    
    
    # NVIDIA Container Toolkit のインストール
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Dockerの再起動
    sudo systemctl restart docker
    
    # GPU の確認
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    

### CUDAイメージ使用のDockerfile
    
    
    # GPU推論用 Dockerfile
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
    
    # Python のインストール
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # PyTorch GPU版のインストール
    COPY requirements-gpu.txt .
    RUN pip3 install --no-cache-dir -r requirements-gpu.txt
    
    # アプリケーションコードとモデル
    COPY app/ ./app/
    COPY models/ ./models/
    
    ENV PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES=0
    
    EXPOSE 8000
    
    CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    

### requirements-gpu.txt
    
    
    # requirements-gpu.txt
    torch==2.0.0+cu118
    torchvision==0.15.0+cu118
    --extra-index-url https://download.pytorch.org/whl/cu118
    fastapi==0.104.0
    uvicorn[standard]==0.24.0
    pydantic==2.5.0
    pillow==10.1.0
    numpy==1.24.3
    

### GPU推論の実装

app/main.pyの例：
    
    
    # app/main.py
    import torch
    from fastapi import FastAPI, File, UploadFile
    from PIL import Image
    import io
    
    app = FastAPI()
    
    # GPU使用可否の確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルのロード
    model = torch.load("/app/models/model.pth", map_location=device)
    model.eval()
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        # 画像の読み込み
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    
        # 前処理（省略）
        # tensor = preprocess(image)
    
        # GPU推論
        with torch.no_grad():
            # tensor = tensor.to(device)
            # output = model(tensor)
            pass
    
        return {"prediction": "result"}
    

### Docker ComposeでGPU使用
    
    
    # docker-compose-gpu.yml
    version: '3.8'
    
    services:
      ml-api-gpu:
        build:
          context: .
          dockerfile: Dockerfile.gpu
        container_name: ml-api-gpu
        ports:
          - "8000:8000"
        volumes:
          - ./models:/app/models:ro
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]
        environment:
          - CUDA_VISIBLE_DEVICES=0
        restart: unless-stopped
    

起動コマンド：
    
    
    # GPU対応コンテナの起動
    docker-compose -f docker-compose-gpu.yml up -d
    
    # GPU使用状況の確認
    docker exec ml-api-gpu nvidia-smi
    
    # ログの確認
    docker-compose -f docker-compose-gpu.yml logs -f
    

### パフォーマンス比較

環境 | 推論時間（1画像） | スループット（画像/秒） | 備考  
---|---|---|---  
**CPU (8コア)** | 150ms | 6.7 | python:3.9-slim  
**GPU (RTX 3090)** | 15ms | 66.7 | nvidia/cuda:11.8.0  
**高速化比** | 10倍 | 10倍 | バッチサイズ1  
  
> **注意** : バッチサイズを増やすことで、GPUのスループットをさらに向上できます。

* * *

## 2.6 本章のまとめ

### 学んだこと

  1. **Dockerの基礎**

     * コンテナと仮想マシンの違い
     * 基本的なDockerコマンド
     * イメージとコンテナの関係
  2. **Dockerfileの作成**

     * 適切なベースイメージの選択
     * マルチステージビルドによる最適化
     * レイヤーキャッシュの活用
  3. **MLモデルのコンテナ化**

     * FastAPI + PyTorchのDocker化
     * .dockerignoreによる効率化
     * セキュリティとヘルスチェック
  4. **Docker Composeオーケストレーション**

     * 複数サービスの統合管理
     * ボリュームと環境変数の管理
     * サービス間の依存関係
  5. **GPU対応MLコンテナ**

     * NVIDIA Dockerのセットアップ
     * CUDAイメージの使用
     * CPU比10倍のパフォーマンス

### ベストプラクティス

原則 | 説明  
---|---  
**軽量イメージ** | slimやalpineベースを優先  
**レイヤー最適化** | 変更頻度の低いものを先に  
**マルチステージビルド** | ビルドと実行環境を分離  
**非rootユーザー** | セキュリティ向上のため  
**.dockerignore** | 不要なファイルを除外  
**ヘルスチェック** | サービスの健全性監視  
**環境変数** | 設定の外部化  
  
### 次の章へ

第3章では、**Kubernetes によるオーケストレーション** を学びます：

  * Kubernetesの基本概念
  * Pod、Service、Deploymentの作成
  * スケーリングと負荷分散
  * ConfigMapとSecretの管理
  * 本番環境へのデプロイ
