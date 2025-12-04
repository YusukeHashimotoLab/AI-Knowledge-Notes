---
title: "Chapter 2: Image Classification and Deep Learning"
chapter_title: "Chapter 2: Image Classification and Deep Learning"
subtitle: Building High-Accuracy Image Classification Systems with CNN Architectures and Transfer Learning
reading_time: 30-35 minutes
difficulty: Intermediate
code_examples: 10
exercises: 5
---

This chapter covers Image Classification and Deep Learning. You will learn efficient design principles of Inception and EfficientNet's Compound Scaling.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the characteristics of major CNN architectures such as LeNet, AlexNet, VGG, and ResNet
  * ✅ Explain the efficient design principles of Inception and MobileNet
  * ✅ Understand EfficientNet's Compound Scaling
  * ✅ Master the differences and applications of Transfer Learning and Fine-tuning
  * ✅ Learn how to utilize pre-trained models using torchvision.models
  * ✅ Implement techniques to improve generalization performance through Data Augmentation
  * ✅ Utilize training techniques such as Learning Rate Scheduling, TTA, and Model Ensemble
  * ✅ Complete practical image classification projects

* * *

## 2.1 Evolution of CNN Architectures

### Historical Development of Image Classification

Image classification is one of the most fundamental and important tasks in computer vision. With the advent of deep learning, the accuracy of image classification has dramatically improved.
    
    
    ```mermaid
    graph LR
        A[LeNet-51998MNIST] --> B[AlexNet2012ImageNet]
        B --> C[VGG201419 layers deep]
        C --> D[GoogLeNet2014Inception]
        D --> E[ResNet2015Residual connections]
        E --> F[Inception-v42016Hybrid]
        F --> G[MobileNet2017Lightweight]
        G --> H[EfficientNet2019Optimization]
        H --> I[Vision Transformer2020+Attention]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
        style F fill:#03a9f4
        style G fill:#039be5
        style H fill:#0288d1
        style I fill:#0277bd
    ```

### LeNet-5 (1998): The Origin of CNNs

**LeNet-5** was developed by Yann LeCun for handwritten digit recognition and became the foundation of modern CNNs.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LeNet5(nn.Module):
        """LeNet-5: Classical CNN for handwritten digit recognition"""
        def __init__(self, num_classes=10):
            super(LeNet5, self).__init__()
    
            # Feature extraction layers
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28×28 → 24×24
            self.pool1 = nn.AvgPool2d(kernel_size=2)       # 24×24 → 12×12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12×12 → 8×8
            self.pool2 = nn.AvgPool2d(kernel_size=2)       # 8×8 → 4×4
    
            # Classification layers
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
    
            x = x.view(x.size(0), -1)  # Flatten
    
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
    
            return x
    
    # Model instantiation and testing
    model = LeNet5(num_classes=10)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    
    print(f"LeNet-5")
    print(f"Input: {x.shape} → Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    

### AlexNet (2012): The Deep Learning Revolution

**AlexNet** won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and sparked the deep learning boom.

Main innovations:

  * **ReLU activation function** : Faster learning than Sigmoid
  * **Dropout** : Preventing overfitting
  * **Data Augmentation** : Improving generalization performance
  * **GPU parallel processing** : Enabling training of large-scale models

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class AlexNet(nn.Module):
        """AlexNet: ImageNet 2012 winning model"""
        def __init__(self, num_classes=1000):
            super(AlexNet, self).__init__()
    
            # Feature extraction layers
            self.features = nn.Sequential(
                # Conv1: 96 filters, 11×11, stride=4
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
    
                # Conv2: 256 filters, 5×5
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
    
                # Conv3: 384 filters, 3×3
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
    
                # Conv4: 384 filters, 3×3
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
    
                # Conv5: 256 filters, 3×3
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
    
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Check model size
    model = AlexNet(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nAlexNet Total Parameters: {total_params:,}")
    print(f"Memory Usage: approx {total_params * 4 / (1024**2):.1f} MB")
    

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
