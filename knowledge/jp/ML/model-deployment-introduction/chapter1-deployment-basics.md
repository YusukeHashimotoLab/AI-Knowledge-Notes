---
title: 第1章：デプロイメントの基礎
chapter_title: 第1章：デプロイメントの基礎
subtitle: 機械学習モデルを本番環境で動かす技術
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 9
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 機械学習モデルのライフサイクル全体を理解する
  * ✅ デプロイメント方式（バッチ、リアルタイム、エッジ）を使い分けられる
  * ✅ FlaskとFastAPIで推論APIを構築できる
  * ✅ モデルのシリアライゼーション形式を適切に選択できる
  * ✅ 実践的な画像分類APIを実装できる

* * *

## 1.1 デプロイメントの基礎

### 機械学習モデルのライフサイクル

機械学習モデルは、開発から本番運用まで以下のステージを経ます。
    
    
    ```mermaid
    graph LR
        A[問題定義] --> B[データ収集]
        B --> C[特徴量エンジニアリング]
        C --> D[モデル開発]
        D --> E[モデル評価]
        E --> F[デプロイメント]
        F --> G[監視・運用]
        G --> H[再学習]
        H --> D
    
        style A fill:#ffebee
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e3f2fd
        style E fill:#e8f5e9
        style F fill:#fce4ec
        style G fill:#ede7f6
        style H fill:#e0f2f1
    ```

> **デプロイメント** は、訓練済みモデルを実際のユーザーが利用できる環境に配置し、予測を提供するプロセスです。

### デプロイメント方式の比較

方式 | 特徴 | レイテンシ | 適用例  
---|---|---|---  
**バッチ推論** | 定期的に大量データを一括処理 | 数分〜数時間 | レコメンデーション、レポート生成  
**リアルタイム推論** | リクエストごとに即座に応答 | 数十ms〜数秒 | Web API、チャットボット  
**エッジ推論** | デバイス上でローカル実行 | 数ms | スマホアプリ、IoTデバイス  
  
### REST APIの基本

リアルタイム推論では、RESTful APIが最も一般的です。
    
    
    # 基本的なAPIリクエストの流れ
    """
    1. クライアント → サーバー: POST /predict
       {
           "features": [5.1, 3.5, 1.4, 0.2]
       }
    
    2. サーバー処理:
       - データ検証
       - 前処理
       - モデル推論
       - 結果フォーマット
    
    3. サーバー → クライアント: 200 OK
       {
           "prediction": "setosa",
           "confidence": 0.98
       }
    """
    

* * *

## 1.2 Flaskによる推論API

### Flask基本セットアップ

Flaskは軽量で学習コストが低いPythonウェブフレームワークです。
    
    
    # app.py - 基本的なFlaskアプリケーション
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """ヘルスチェックエンドポイント"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0'
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """予測エンドポイント"""
        data = request.get_json()
    
        # 簡易的な応答（後でモデル推論に置き換え）
        return jsonify({
            'prediction': 'sample',
            'input_received': data
        })
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)
    

**起動方法** ：
    
    
    python app.py
    # → http://localhost:5000 でサーバー起動
    
    # 別のターミナルでテスト
    curl http://localhost:5000/health
    

### scikit-learnモデルのデプロイ
    
    
    # train_model.py - モデル訓練とシリアライゼーション
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # データ準備
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # モデル訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # モデル保存
    joblib.dump(model, 'iris_model.pkl')
    print(f"モデル精度: {model.score(X_test, y_test):.3f}")
    print("モデル保存完了: iris_model.pkl")
    

### POSTリクエストでの予測
    
    
    # flask_app.py - 完全な推論API
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    
    # モデル読み込み（起動時に1回だけ）
    model = joblib.load('iris_model.pkl')
    class_names = ['setosa', 'versicolor', 'virginica']
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Iris分類の予測
    
        リクエスト例:
        {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        """
        try:
            # リクエストデータ取得
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
    
            # 予測
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
    
            # 結果フォーマット
            return jsonify({
                'prediction': class_names[prediction],
                'class_id': int(prediction),
                'probabilities': {
                    class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            })
    
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)
    

**テスト** ：
    
    
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
    
    # 出力:
    # {
    #   "prediction": "setosa",
    #   "class_id": 0,
    #   "probabilities": {
    #     "setosa": 0.98,
    #     "versicolor": 0.02,
    #     "virginica": 0.0
    #   }
    # }
    

### エラーハンドリング
    
    
    # error_handling.py - 堅牢なエラーハンドリング
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    model = joblib.load('iris_model.pkl')
    class_names = ['setosa', 'versicolor', 'virginica']
    
    def validate_input(data):
        """入力データの検証"""
        if 'features' not in data:
            raise ValueError("'features' キーが必要です")
    
        features = data['features']
        if not isinstance(features, list):
            raise ValueError("features はリストである必要があります")
    
        if len(features) != 4:
            raise ValueError(f"features は4要素必要です（受信: {len(features)}）")
    
        return np.array(features).reshape(1, -1)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
    
            # 入力検証
            features = validate_input(data)
    
            # 予測
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
    
            return jsonify({
                'prediction': class_names[prediction],
                'class_id': int(prediction),
                'probabilities': {
                    class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            }), 200
    
        except ValueError as e:
            return jsonify({'error': f'入力エラー: {str(e)}'}), 400
    
        except Exception as e:
            return jsonify({'error': f'サーバーエラー: {str(e)}'}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'エンドポイントが見つかりません'}), 404
    
    if __name__ == '__main__':
        app.run(debug=False, host='0.0.0.0', port=5000)
    

* * *

## 1.3 FastAPIによる高速推論

### FastAPIの利点

FastAPIは、Flaskより高速で、自動ドキュメント生成や型検証を提供します。

特徴 | Flask | FastAPI  
---|---|---  
**速度** | 中程度 | 高速（非同期対応）  
**型検証** | 手動 | Pydanticで自動  
**ドキュメント** | 手動 | 自動生成（Swagger UI）  
**学習コスト** | 低い | やや高い  
  
### Pydanticモデル定義
    
    
    # models.py - Pydanticモデル定義
    from pydantic import BaseModel, Field, validator
    from typing import List
    
    class IrisFeatures(BaseModel):
        """Iris特徴量の入力スキーマ"""
        features: List[float] = Field(
            ...,
            description="4つの特徴量 [sepal_length, sepal_width, petal_length, petal_width]",
            min_items=4,
            max_items=4
        )
    
        @validator('features')
        def check_positive(cls, v):
            """特徴量が正の値であることを検証"""
            if any(x < 0 for x in v):
                raise ValueError('すべての特徴量は正の値である必要があります')
            return v
    
    class PredictionResponse(BaseModel):
        """予測結果のレスポンススキーマ"""
        prediction: str = Field(..., description="予測クラス名")
        class_id: int = Field(..., description="クラスID (0-2)")
        probabilities: dict = Field(..., description="各クラスの確率")
    
    # 使用例
    sample_input = IrisFeatures(features=[5.1, 3.5, 1.4, 0.2])
    print(sample_input.json())
    # → {"features": [5.1, 3.5, 1.4, 0.2]}
    

### PyTorchモデルのデプロイ
    
    
    # fastapi_pytorch.py - FastAPI + PyTorch推論API
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import torch.nn as nn
    from typing import List
    import uvicorn
    
    # Pydanticモデル
    class InputData(BaseModel):
        features: List[float]
    
    class PredictionOutput(BaseModel):
        prediction: int
        confidence: float
    
    # PyTorchモデル定義
    class SimpleNN(nn.Module):
        def __init__(self, input_size=4, hidden_size=16, num_classes=3):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # FastAPIアプリケーション
    app = FastAPI(
        title="Iris Classification API",
        description="PyTorchモデルによるIris分類API",
        version="1.0.0"
    )
    
    # モデル読み込み（起動時）
    model = SimpleNN()
    model.load_state_dict(torch.load('iris_pytorch_model.pth'))
    model.eval()
    
    @app.post("/predict", response_model=PredictionOutput)
    async def predict(data: InputData):
        """
        Iris分類の予測
    
        - **features**: 4つの特徴量のリスト
        """
        try:
            # テンソル変換
            features_tensor = torch.tensor([data.features], dtype=torch.float32)
    
            # 推論
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
    
            return PredictionOutput(
                prediction=prediction,
                confidence=confidence
            )
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    

### Swagger UI自動生成

FastAPIを起動すると、自動的に対話的なAPIドキュメントが生成されます。
    
    
    # 起動
    python fastapi_pytorch.py
    
    # ブラウザで以下にアクセス:
    # - Swagger UI: http://localhost:8000/docs
    # - ReDoc: http://localhost:8000/redoc
    # - OpenAPIスキーマ: http://localhost:8000/openapi.json
    

> **利点** : ブラウザから直接APIをテストでき、開発効率が大幅に向上します。

* * *

## 1.4 モデルのシリアライゼーション

### pickle / joblib

Pythonオブジェクトを保存する標準的な方法です。
    
    
    # serialization_comparison.py
    import pickle
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    # モデル訓練
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    
    # pickle保存
    with open('model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # joblib保存（より効率的）
    joblib.dump(model, 'model_joblib.pkl')
    
    # 読み込み
    model_pickle = pickle.load(open('model_pickle.pkl', 'rb'))
    model_joblib = joblib.load('model_joblib.pkl')
    
    # ファイルサイズ比較
    import os
    print(f"pickle: {os.path.getsize('model_pickle.pkl')} bytes")
    print(f"joblib: {os.path.getsize('model_joblib.pkl')} bytes")
    # → joblibの方が効率的（特に大きなnumpy配列を含む場合）
    

### ONNX形式

ONNX（Open Neural Network Exchange）は、異なるフレームワーク間で互換性のある形式です。
    
    
    # onnx_export.py - PyTorchモデルをONNXに変換
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import numpy as np
    
    # PyTorchモデル定義
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 3)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # モデルのONNXエクスポート
    model = SimpleNN()
    model.eval()
    
    dummy_input = torch.randn(1, 4)
    torch.onnx.export(
        model,
        dummy_input,
        "iris_model.onnx",
        input_names=['features'],
        output_names=['logits'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # ONNX Runtimeで推論
    ort_session = ort.InferenceSession("iris_model.onnx")
    
    def predict_onnx(features):
        ort_inputs = {'features': features.astype(np.float32)}
        ort_outputs = ort_session.run(None, ort_inputs)
        return ort_outputs[0]
    
    # テスト
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    output = predict_onnx(test_input)
    print(f"ONNX推論結果: {output}")
    

### TorchScript

PyTorchモデルを本番環境用に最適化する方式です。
    
    
    # torchscript_export.py
    import torch
    import torch.nn as nn
    
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 3)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleNN()
    model.eval()
    
    # TorchScript変換（トレーシング）
    example_input = torch.randn(1, 4)
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存
    traced_model.save("iris_torchscript.pt")
    
    # 読み込みと推論
    loaded_model = torch.jit.load("iris_torchscript.pt")
    test_input = torch.tensor([[5.1, 3.5, 1.4, 0.2]])
    
    with torch.no_grad():
        output = loaded_model(test_input)
        print(f"TorchScript推論結果: {output}")
    

### 形式の比較と選択

形式 | 対象 | 長所 | 短所 | 推奨用途  
---|---|---|---|---  
**pickle/joblib** | scikit-learn | 簡単、軽量 | Python依存、セキュリティリスク | 開発、プロトタイピング  
**ONNX** | 全般 | フレームワーク非依存、高速 | 変換の手間 | 本番環境、マルチ言語  
**TorchScript** | PyTorch | 最適化、C++実行可能 | PyTorch専用 | 本番環境（PyTorch）  
  
* * *

## 1.5 実践: 画像分類APIの構築

### ResNet推論API
    
    
    # image_classification_api.py - ResNetによる画像分類API
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from pydantic import BaseModel
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    import io
    import time
    
    app = FastAPI(title="Image Classification API")
    
    # ResNet50モデル読み込み（起動時）
    print("モデル読み込み中...")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    
    # 前処理パイプライン
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ImageNetクラス名
    categories = weights.meta["categories"]
    
    class PredictionResult(BaseModel):
        top_class: str
        confidence: float
        top_5: dict
        inference_time_ms: float
    
    @app.post("/predict", response_model=PredictionResult)
    async def predict_image(file: UploadFile = File(...)):
        """
        画像分類
    
        - **file**: 画像ファイル（JPEG, PNG等）
        """
        try:
            # 画像読み込み
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
            # 前処理
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
    
            # 推論
            start_time = time.time()
            with torch.no_grad():
                output = model(input_batch)
            inference_time = (time.time() - start_time) * 1000
    
            # 確率計算
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
            # Top-5予測
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            top5_results = {
                categories[idx]: float(prob)
                for idx, prob in zip(top5_idx, top5_prob)
            }
    
            return PredictionResult(
                top_class=categories[top5_idx[0]],
                confidence=float(top5_prob[0]),
                top_5=top5_results,
                inference_time_ms=inference_time
            )
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    

### Base64画像処理
    
    
    # base64_image_api.py - Base64エンコード画像の処理
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import base64
    import io
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    
    app = FastAPI()
    
    # モデル設定
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class Base64ImageInput(BaseModel):
        image: str  # Base64エンコード文字列
    
    class PredictionOutput(BaseModel):
        prediction: str
        confidence: float
    
    @app.post("/predict", response_model=PredictionOutput)
    async def predict_base64(data: Base64ImageInput):
        """
        Base64エンコード画像の分類
    
        リクエスト例:
        {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        }
        """
        try:
            # Base64デコード
            if ',' in data.image:
                image_data = data.image.split(',')[1]  # "data:image/jpeg;base64," を除去
            else:
                image_data = data.image
    
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
            # 推論
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
    
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probabilities, dim=0)
    
            return PredictionOutput(
                prediction=categories[top_idx],
                confidence=float(top_prob)
            )
    
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"画像処理エラー: {str(e)}")
    

### パフォーマンス測定
    
    
    # benchmark.py - APIパフォーマンス測定
    import requests
    import time
    import numpy as np
    from PIL import Image
    import io
    
    API_URL = "http://localhost:8000/predict"
    
    def create_test_image():
        """テスト用ダミー画像作成"""
        img = Image.new('RGB', (224, 224), color='red')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return buf
    
    def benchmark_api(num_requests=100):
        """API性能測定"""
        latencies = []
    
        print(f"{num_requests}回のリクエストを送信中...")
    
        for i in range(num_requests):
            image_file = create_test_image()
            files = {'file': ('test.jpg', image_file, 'image/jpeg')}
    
            start = time.time()
            response = requests.post(API_URL, files=files)
            latency = (time.time() - start) * 1000
    
            if response.status_code == 200:
                latencies.append(latency)
            else:
                print(f"エラー: {response.status_code}")
    
            if (i + 1) % 10 == 0:
                print(f"  進捗: {i + 1}/{num_requests}")
    
        # 統計
        latencies = np.array(latencies)
        print("\n=== パフォーマンス統計 ===")
        print(f"リクエスト数: {len(latencies)}")
        print(f"平均レイテンシ: {latencies.mean():.2f} ms")
        print(f"中央値: {np.median(latencies):.2f} ms")
        print(f"最小: {latencies.min():.2f} ms")
        print(f"最大: {latencies.max():.2f} ms")
        print(f"標準偏差: {latencies.std():.2f} ms")
        print(f"P95: {np.percentile(latencies, 95):.2f} ms")
        print(f"P99: {np.percentile(latencies, 99):.2f} ms")
    
    if __name__ == "__main__":
        benchmark_api(num_requests=100)
    

**実行例** ：
    
    
    === パフォーマンス統計 ===
    リクエスト数: 100
    平均レイテンシ: 125.34 ms
    中央値: 120.12 ms
    最小: 98.45 ms
    最大: 210.67 ms
    標準偏差: 18.92 ms
    P95: 155.23 ms
    P99: 180.45 ms
    

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **デプロイメントの基礎**

     * 機械学習ライフサイクル全体の理解
     * バッチ、リアルタイム、エッジ推論の使い分け
  2. **Flaskによる推論API**

     * 軽量で学習コストが低い
     * エラーハンドリングの実装
  3. **FastAPIによる高速推論**

     * Pydanticによる型検証
     * 自動ドキュメント生成（Swagger UI）
     * 非同期処理による高速化
  4. **モデルシリアライゼーション**

     * pickle/joblib: 開発・プロトタイピング
     * ONNX: マルチフレームワーク対応
     * TorchScript: PyTorch最適化
  5. **実践的な画像分類API**

     * ResNetによるリアルタイム推論
     * Base64画像処理
     * パフォーマンス測定と最適化

### 次の章へ

第2章では、**Dockerコンテナ化とデプロイメント** を学びます：

  * Dockerの基礎とコンテナ化
  * マルチステージビルド
  * Docker Composeによる環境構築
  * クラウドへのデプロイ（AWS、GCP）

* * *

## 参考文献

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  3. FastAPI公式ドキュメント: <https://fastapi.tiangolo.com/>
  4. ONNX公式サイト: <https://onnx.ai/>
