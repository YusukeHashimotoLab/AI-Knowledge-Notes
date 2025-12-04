---
title: "Chapter 1: Deployment Basics"
chapter_title: "Chapter 1: Deployment Basics"
subtitle: Techniques for Running Machine Learning Models in Production
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 9
version: 1.0
created_at: 2025-10-23
---

This chapter introduces the basics of Deployment Basics. You will learn complete machine learning model lifecycle, Distinguish between deployment methods (batch, and Build inference APIs with Flask.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the complete machine learning model lifecycle
  * ✅ Distinguish between deployment methods (batch, real-time, edge)
  * ✅ Build inference APIs with Flask and FastAPI
  * ✅ Select appropriate model serialization formats
  * ✅ Implement practical image classification APIs

* * *

## 1.1 Deployment Fundamentals

### Machine Learning Model Lifecycle

Machine learning models go through the following stages from development to production operation.
    
    
    ```mermaid
    graph LR
        A[Problem Definition] --> B[Data Collection]
        B --> C[Feature Engineering]
        C --> D[Model Development]
        D --> E[Model Evaluation]
        E --> F[Deployment]
        F --> G[Monitoring & Operations]
        G --> H[Retraining]
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

> **Deployment** is the process of placing a trained model in an environment where actual users can access it and providing predictions.

### Comparison of Deployment Methods

Method | Characteristics | Latency | Use Cases  
---|---|---|---  
**Batch Inference** | Process large amounts of data periodically in bulk | Minutes to hours | Recommendations, report generation  
**Real-time Inference** | Respond immediately to each request | Tens of ms to seconds | Web APIs, chatbots  
**Edge Inference** | Execute locally on devices | A few ms | Mobile apps, IoT devices  
  
### REST API Basics

For real-time inference, RESTful APIs are the most common approach.
    
    
    # Basic API request flow
    """
    1. Client → Server: POST /predict
       {
           "features": [5.1, 3.5, 1.4, 0.2]
       }
    
    2. Server Processing:
       - Data validation
       - Preprocessing
       - Model inference
       - Result formatting
    
    3. Server → Client: 200 OK
       {
           "prediction": "setosa",
           "confidence": 0.98
       }
    """
    

* * *

## 1.2 Inference API with Flask

### Flask Basic Setup

Flask is a lightweight Python web framework with a low learning curve.
    
    
    # Requirements:
    # - Python 3.9+
    # - flask>=2.3.0
    
    # app.py - Basic Flask application
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0'
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Prediction endpoint"""
        data = request.get_json()
    
        # Simple response (will be replaced with model inference later)
        return jsonify({
            'prediction': 'sample',
            'input_received': data
        })
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)
    

**How to Run** :
    
    
    python app.py
    # → Server starts at http://localhost:5000
    
    # Test from another terminal
    curl http://localhost:5000/health
    

### Deploying scikit-learn Models
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    
    """
    Example: Deploying scikit-learn Models
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # train_model.py - Model training and serialization
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Data preparation
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'iris_model.pkl')
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    print("Model saved: iris_model.pkl")
    

### Prediction with POST Requests
    
    
    # Requirements:
    # - Python 3.9+
    # - flask>=2.3.0
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    # flask_app.py - Complete inference API
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    
    # Load model (only once at startup)
    model = joblib.load('iris_model.pkl')
    class_names = ['setosa', 'versicolor', 'virginica']
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Iris classification prediction
    
        Request example:
        {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        """
        try:
            # Get request data
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
    
            # Prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
    
            # Format result
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
    

**Testing** :
    
    
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
    
    # Output:
    # {
    #   "prediction": "setosa",
    #   "class_id": 0,
    #   "probabilities": {
    #     "setosa": 0.98,
    #     "versicolor": 0.02,
    #     "virginica": 0.0
    #   }
    # }
    

### Error Handling
    
    
    # Requirements:
    # - Python 3.9+
    # - flask>=2.3.0
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    # error_handling.py - Robust error handling
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    model = joblib.load('iris_model.pkl')
    class_names = ['setosa', 'versicolor', 'virginica']
    
    def validate_input(data):
        """Validate input data"""
        if 'features' not in data:
            raise ValueError("'features' key is required")
    
        features = data['features']
        if not isinstance(features, list):
            raise ValueError("features must be a list")
    
        if len(features) != 4:
            raise ValueError(f"features must have 4 elements (received: {len(features)})")
    
        return np.array(features).reshape(1, -1)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
    
            # Input validation
            features = validate_input(data)
    
            # Prediction
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
            return jsonify({'error': f'Input error: {str(e)}'}), 400
    
        except Exception as e:
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    if __name__ == '__main__':
        app.run(debug=False, host='0.0.0.0', port=5000)
    

* * *

## 1.3 High-Speed Inference with FastAPI

### Advantages of FastAPI

FastAPI is faster than Flask and provides automatic documentation generation and type validation.

Feature | Flask | FastAPI  
---|---|---  
**Speed** | Moderate | High (async support)  
**Type Validation** | Manual | Automatic with Pydantic  
**Documentation** | Manual | Auto-generated (Swagger UI)  
**Learning Curve** | Low | Somewhat higher  
  
### Pydantic Model Definition
    
    
    # models.py - Pydantic model definition
    from pydantic import BaseModel, Field, validator
    from typing import List
    
    class IrisFeatures(BaseModel):
        """Input schema for Iris features"""
        features: List[float] = Field(
            ...,
            description="4 features [sepal_length, sepal_width, petal_length, petal_width]",
            min_items=4,
            max_items=4
        )
    
        @validator('features')
        def check_positive(cls, v):
            """Validate that features are positive values"""
            if any(x < 0 for x in v):
                raise ValueError('All features must be positive values')
            return v
    
    class PredictionResponse(BaseModel):
        """Response schema for predictions"""
        prediction: str = Field(..., description="Predicted class name")
        class_id: int = Field(..., description="Class ID (0-2)")
        probabilities: dict = Field(..., description="Probabilities for each class")
    
    # Usage example
    sample_input = IrisFeatures(features=[5.1, 3.5, 1.4, 0.2])
    print(sample_input.json())
    # → {"features": [5.1, 3.5, 1.4, 0.2]}
    

### Deploying PyTorch Models
    
    
    # Requirements:
    # - Python 3.9+
    # - fastapi>=0.100.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Deploying PyTorch Models
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # fastapi_pytorch.py - FastAPI + PyTorch inference API
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import torch.nn as nn
    from typing import List
    import uvicorn
    
    # Pydantic models
    class InputData(BaseModel):
        features: List[float]
    
    class PredictionOutput(BaseModel):
        prediction: int
        confidence: float
    
    # PyTorch model definition
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
    
    # FastAPI application
    app = FastAPI(
        title="Iris Classification API",
        description="Iris classification API using PyTorch model",
        version="1.0.0"
    )
    
    # Load model (at startup)
    model = SimpleNN()
    model.load_state_dict(torch.load('iris_pytorch_model.pth'))
    model.eval()
    
    @app.post("/predict", response_model=PredictionOutput)
    async def predict(data: InputData):
        """
        Iris classification prediction
    
        - **features**: List of 4 features
        """
        try:
            # Convert to tensor
            features_tensor = torch.tensor([data.features], dtype=torch.float32)
    
            # Inference
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
    

### Automatic Swagger UI Generation

When you start FastAPI, interactive API documentation is automatically generated.
    
    
    # Start the server
    python fastapi_pytorch.py
    
    # Access the following in your browser:
    # - Swagger UI: http://localhost:8000/docs
    # - ReDoc: http://localhost:8000/redoc
    # - OpenAPI schema: http://localhost:8000/openapi.json
    

> **Benefit** : You can test the API directly from your browser, significantly improving development efficiency.

* * *

## 1.4 Model Serialization

### pickle / joblib

The standard way to save Python objects.
    
    
    # Requirements:
    # - Python 3.9+
    # - joblib>=1.3.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: The standard way to save Python objects.
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 30-60 seconds
    Dependencies: None
    """
    
    # serialization_comparison.py
    import pickle
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    # Train model
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    
    # Save with pickle
    with open('model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save with joblib (more efficient)
    joblib.dump(model, 'model_joblib.pkl')
    
    # Load
    model_pickle = pickle.load(open('model_pickle.pkl', 'rb'))
    model_joblib = joblib.load('model_joblib.pkl')
    
    # Compare file sizes
    import os
    print(f"pickle: {os.path.getsize('model_pickle.pkl')} bytes")
    print(f"joblib: {os.path.getsize('model_joblib.pkl')} bytes")
    # → joblib is more efficient (especially with large numpy arrays)
    

### ONNX Format

ONNX (Open Neural Network Exchange) is a format compatible across different frameworks.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: ONNX (Open Neural Network Exchange) is a format compatible a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    # onnx_export.py - Convert PyTorch model to ONNX
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    import numpy as np
    
    # PyTorch model definition
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
    
    # Export model to ONNX
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
    
    # Inference with ONNX Runtime
    ort_session = ort.InferenceSession("iris_model.onnx")
    
    def predict_onnx(features):
        ort_inputs = {'features': features.astype(np.float32)}
        ort_outputs = ort_session.run(None, ort_inputs)
        return ort_outputs[0]
    
    # Test
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    output = predict_onnx(test_input)
    print(f"ONNX inference result: {output}")
    

### TorchScript

A method to optimize PyTorch models for production environments.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: A method to optimize PyTorch models for production environme
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
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
    
    # TorchScript conversion (tracing)
    example_input = torch.randn(1, 4)
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save("iris_torchscript.pt")
    
    # Load and inference
    loaded_model = torch.jit.load("iris_torchscript.pt")
    test_input = torch.tensor([[5.1, 3.5, 1.4, 0.2]])
    
    with torch.no_grad():
        output = loaded_model(test_input)
        print(f"TorchScript inference result: {output}")
    

### Format Comparison and Selection

Format | Target | Advantages | Disadvantages | Recommended Use  
---|---|---|---|---  
**pickle/joblib** | scikit-learn | Simple, lightweight | Python dependency, security risks | Development, prototyping  
**ONNX** | General | Framework-agnostic, fast | Conversion overhead | Production, multi-language  
**TorchScript** | PyTorch | Optimized, C++ executable | PyTorch-specific | Production (PyTorch)  
  
* * *

## 1.5 Practical Example: Building an Image Classification API

### ResNet Inference API
    
    
    # Requirements:
    # - Python 3.9+
    # - fastapi>=0.100.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: ResNet Inference API
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # image_classification_api.py - Image classification API with ResNet
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from pydantic import BaseModel
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    import io
    import time
    
    app = FastAPI(title="Image Classification API")
    
    # Load ResNet50 model (at startup)
    print("Loading model...")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ImageNet class names
    categories = weights.meta["categories"]
    
    class PredictionResult(BaseModel):
        top_class: str
        confidence: float
        top_5: dict
        inference_time_ms: float
    
    @app.post("/predict", response_model=PredictionResult)
    async def predict_image(file: UploadFile = File(...)):
        """
        Image classification
    
        - **file**: Image file (JPEG, PNG, etc.)
        """
        try:
            # Load image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
            # Preprocessing
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
    
            # Inference
            start_time = time.time()
            with torch.no_grad():
                output = model(input_batch)
            inference_time = (time.time() - start_time) * 1000
    
            # Calculate probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
            # Top-5 predictions
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
    

### Base64 Image Processing
    
    
    # Requirements:
    # - Python 3.9+
    # - fastapi>=0.100.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Base64 Image Processing
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    # base64_image_api.py - Processing Base64-encoded images
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import base64
    import io
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    
    app = FastAPI()
    
    # Model setup
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
        image: str  # Base64-encoded string
    
    class PredictionOutput(BaseModel):
        prediction: str
        confidence: float
    
    @app.post("/predict", response_model=PredictionOutput)
    async def predict_base64(data: Base64ImageInput):
        """
        Classification of Base64-encoded images
    
        Request example:
        {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        }
        """
        try:
            # Base64 decoding
            if ',' in data.image:
                image_data = data.image.split(',')[1]  # Remove "data:image/jpeg;base64,"
            else:
                image_data = data.image
    
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
            # Inference
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
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
    

### Performance Measurement
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - requests>=2.31.0
    
    # benchmark.py - API performance measurement
    import requests
    import time
    import numpy as np
    from PIL import Image
    import io
    
    API_URL = "http://localhost:8000/predict"
    
    def create_test_image():
        """Create dummy test image"""
        img = Image.new('RGB', (224, 224), color='red')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return buf
    
    def benchmark_api(num_requests=100):
        """Measure API performance"""
        latencies = []
    
        print(f"Sending {num_requests} requests...")
    
        for i in range(num_requests):
            image_file = create_test_image()
            files = {'file': ('test.jpg', image_file, 'image/jpeg')}
    
            start = time.time()
            response = requests.post(API_URL, files=files)
            latency = (time.time() - start) * 1000
    
            if response.status_code == 200:
                latencies.append(latency)
            else:
                print(f"Error: {response.status_code}")
    
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_requests}")
    
        # Statistics
        latencies = np.array(latencies)
        print("\n=== Performance Statistics ===")
        print(f"Number of requests: {len(latencies)}")
        print(f"Average latency: {latencies.mean():.2f} ms")
        print(f"Median: {np.median(latencies):.2f} ms")
        print(f"Minimum: {latencies.min():.2f} ms")
        print(f"Maximum: {latencies.max():.2f} ms")
        print(f"Standard deviation: {latencies.std():.2f} ms")
        print(f"P95: {np.percentile(latencies, 95):.2f} ms")
        print(f"P99: {np.percentile(latencies, 99):.2f} ms")
    
    if __name__ == "__main__":
        benchmark_api(num_requests=100)
    

**Execution Example** :
    
    
    === Performance Statistics ===
    Number of requests: 100
    Average latency: 125.34 ms
    Median: 120.12 ms
    Minimum: 98.45 ms
    Maximum: 210.67 ms
    Standard deviation: 18.92 ms
    P95: 155.23 ms
    P99: 180.45 ms
    

* * *

## 1.6 Chapter Summary

### What We Learned

  1. **Deployment Fundamentals**

     * Understanding the complete machine learning lifecycle
     * Distinguishing between batch, real-time, and edge inference
  2. **Inference API with Flask**

     * Lightweight with low learning curve
     * Implementing error handling
  3. **High-Speed Inference with FastAPI**

     * Type validation with Pydantic
     * Automatic documentation generation (Swagger UI)
     * Speed improvements through async processing
  4. **Model Serialization**

     * pickle/joblib: Development and prototyping
     * ONNX: Multi-framework compatibility
     * TorchScript: PyTorch optimization
  5. **Practical Image Classification API**

     * Real-time inference with ResNet
     * Base64 image processing
     * Performance measurement and optimization

### Next Chapter

In Chapter 2, we will learn about **Docker Containerization and Deployment** :

  * Docker basics and containerization
  * Multi-stage builds
  * Environment setup with Docker Compose
  * Cloud deployment (AWS, GCP)

* * *

## References

  1. Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.
  2. Huyen, C. (2022). _Designing Machine Learning Systems_. O'Reilly Media.
  3. FastAPI Official Documentation: <https://fastapi.tiangolo.com/>
  4. ONNX Official Website: <https://onnx.ai/>
