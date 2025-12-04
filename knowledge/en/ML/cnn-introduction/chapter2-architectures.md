---
title: "Chapter 2: Pooling Layers and CNN Architectures"
chapter_title: "Chapter 2: Pooling Layers and CNN Architectures"
subtitle: Understanding the Evolution and Design Principles of Representative CNN Models
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 10
exercises: 5
---

This chapter covers Pooling Layers and CNN Architectures. You will learn dimensionality reduction and overfitting prevention techniques using Dropout.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the role and types of pooling layers (Max Pooling, Average Pooling)
  * ✅ Explain dimensionality reduction and acquisition of translation invariance through pooling
  * ✅ Grasp the evolution of representative CNN architectures from LeNet-5 to ResNet
  * ✅ Understand the principles and effects of Batch Normalization
  * ✅ Implement overfitting prevention techniques using Dropout
  * ✅ Understand the importance of Skip Connections (Residual Connections)
  * ✅ Build a practical image classification model on the CIFAR-10 dataset

* * *

## 2.1 The Role of Pooling Layers

### What is Pooling?

A **Pooling Layer** is a layer that performs **spatial downsampling** of the output from convolutional layers. Its main purposes are the following three:

  * **Dimensionality reduction** : Reduces the size of feature maps, decreasing computational cost and memory usage
  * **Translation invariance** : Acquires robustness to minor positional changes in features
  * **Expanding receptive field** : Integrates information from a wider range

> "Pooling is an operation that preserves important features of an image while discarding unnecessary details"

### Max Pooling vs Average Pooling

There are mainly two types of pooling:

Type | Operation | Characteristics | Use Cases  
---|---|---|---  
**Max Pooling** | Takes the maximum value in the region | Preserves the strongest features | Object detection, general image classification  
**Average Pooling** | Takes the average value in the region | Preserves overall features | Global Average Pooling, segmentation  
      
    
    ```mermaid
    graph LR
        A["Input Feature Map4×4"] --> B["Max Pooling2×2, stride=2"]
        A --> C["Average Pooling2×2, stride=2"]
    
        B --> D["Output2×2Max values retained"]
        C --> E["Output2×2Average values retained"]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
    ```

### Example of Max Pooling Operation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Example of Max Pooling Operation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import torch
    import torch.nn as nn
    
    # Input data (1 channel, 4×4)
    input_data = torch.tensor([[
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 6.0, 1.0, 2.0],
        [7.0, 2.0, 8.0, 3.0],
        [1.0, 4.0, 6.0, 9.0]
    ]], dtype=torch.float32).unsqueeze(0)  # (1, 1, 4, 4)
    
    print("Input Feature Map:")
    print(input_data.squeeze().numpy())
    
    # Max Pooling (2×2, stride=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output_max = max_pool(input_data)
    
    print("\nMax Pooling (2×2) Output:")
    print(output_max.squeeze().numpy())
    
    # Average Pooling (2×2, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    output_avg = avg_pool(input_data)
    
    print("\nAverage Pooling (2×2) Output:")
    print(output_avg.squeeze().numpy())
    
    # Manual calculation verification (top-left region)
    print("\nManual calculation (top-left 2×2 region):")
    region = input_data[0, 0, 0:2, 0:2].numpy()
    print(f"Region: \n{region}")
    print(f"Max: {region.max()}")
    print(f"Average: {region.mean()}")
    

### Effect of Pooling: Translation Invariance
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Effect of Pooling: Translation Invariance
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Original feature map
    original = torch.tensor([[
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    
    # Slightly shifted feature map
    shifted = torch.tensor([[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    print("Max Pooling of original feature map:")
    print(max_pool(original).squeeze().numpy())
    
    print("\nMax Pooling of shifted feature map:")
    print(max_pool(shifted).squeeze().numpy())
    
    print("\n→ Even with minor positional changes, Max Pooling output has 1 appearing in the same region")
    print("  This is 'translation invariance'")
    

### Pooling Parameters

  * **kernel_size** : Size of the pooling region (typically 2×2 or 3×3)
  * **stride** : Sliding width (typically the same as kernel_size, with no overlap)
  * **padding** : Zero padding (typically 0)

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Pooling Parameters
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Comparison of different pooling configurations
    input_data = torch.randn(1, 1, 8, 8)  # (batch, channels, height, width)
    
    # Configuration 1: 2×2, stride=2 (standard)
    pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    output1 = pool1(input_data)
    
    # Configuration 2: 3×3, stride=2 (with overlap)
    pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    output2 = pool2(input_data)
    
    # Configuration 3: 2×2, stride=1 (high overlap)
    pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
    output3 = pool3(input_data)
    
    print(f"Input size: {input_data.shape}")
    print(f"2×2, stride=2 output size: {output1.shape}")
    print(f"3×3, stride=2 output size: {output2.shape}")
    print(f"2×2, stride=1 output size: {output3.shape}")
    
    # Calculate dimensionality reduction rate
    reduction1 = (input_data.numel() - output1.numel()) / input_data.numel() * 100
    reduction2 = (input_data.numel() - output2.numel()) / input_data.numel() * 100
    
    print(f"\n2×2, stride=2 dimensionality reduction rate: {reduction1:.1f}%")
    print(f"3×3, stride=2 dimensionality reduction rate: {reduction2:.1f}%")
    

### Global Average Pooling

**Global Average Pooling (GAP)** is a special type of pooling that takes the average of the entire feature map. In modern CNNs, it is increasingly used as a replacement for the final fully connected layer.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Global Average Pooling (GAP)is a special type of pooling tha
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Input: (batch_size, channels, height, width)
    input_features = torch.randn(2, 512, 7, 7)  # 2 samples, 512 channels, 7×7
    
    # Global Average Pooling
    gap = nn.AdaptiveAvgPool2d((1, 1))  # Specify output size as (1, 1)
    output = gap(input_features)
    
    print(f"Input size: {input_features.shape}")
    print(f"GAP output size: {output.shape}")
    
    # Flatten
    output_flat = output.view(output.size(0), -1)
    print(f"After flattening: {output_flat.shape}")
    
    # Benefits of GAP
    print("\nAdvantages of Global Average Pooling:")
    print("1. Zero parameters (compared to fully connected layers)")
    print("2. Independent of input size (works with any size)")
    print("3. Reduced risk of overfitting")
    print("4. Spatial average of each channel = intensity of the concept that channel represents")
    

* * *

## 2.2 Representative CNN Architectures

### Evolution of CNNs: Historical Overview
    
    
    ```mermaid
    graph LR
        A[LeNet-51998] --> B[AlexNet2012]
        B --> C[VGGNet2014]
        C --> D[GoogLeNet2014]
        D --> E[ResNet2015]
        E --> F[DenseNet2017]
        F --> G[EfficientNet2019]
        G --> H[Vision Transformer2020+]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
        style F fill:#03a9f4
        style G fill:#039be5
        style H fill:#0288d1
    ```

### LeNet-5 (1998): The Origin of CNNs

**LeNet-5** was developed by Yann LeCun for handwritten digit recognition (MNIST). It is the foundational architecture of modern CNNs.

Layer | Output Size | Parameters  
---|---|---  
Input | 1×28×28 | -  
Conv1 (5×5, 6ch) | 6×24×24 | 156  
AvgPool (2×2) | 6×12×12 | 0  
Conv2 (5×5, 16ch) | 16×8×8 | 2,416  
AvgPool (2×2) | 16×4×4 | 0  
FC1 (120) | 120 | 30,840  
FC2 (84) | 84 | 10,164  
FC3 (10) | 10 | 850  
      
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: LeNet-5was developed by Yann LeCun for handwritten digit rec
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LeNet5(nn.Module):
        def __init__(self, num_classes=10):
            super(LeNet5, self).__init__()
    
            # Feature extraction layers
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28×28 → 24×24
            self.pool1 = nn.AvgPool2d(kernel_size=2)        # 24×24 → 12×12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12×12 → 8×8
            self.pool2 = nn.AvgPool2d(kernel_size=2)        # 8×8 → 4×4
    
            # Classification layers
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
    
        def forward(self, x):
            # Feature extraction
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
    
            # Flatten
            x = x.view(x.size(0), -1)
    
            # Classification
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
    
            return x
    
    # Create model and summary
    model = LeNet5(num_classes=10)
    print(model)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test run
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"\nInput size: {x.shape}")
    print(f"Output size: {output.shape}")
    

### AlexNet (2012): The Dawn of Deep Learning

**AlexNet** demonstrated overwhelming performance in the 2012 ImageNet competition and sparked the deep learning boom.

Key features:

  * Use of **ReLU activation function** (learns faster than Sigmoid)
  * **Dropout** for overfitting prevention
  * Utilization of **Data Augmentation**
  * Leveraging **GPU parallel processing**
  * **Local Response Normalization** (not commonly used today)

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Key features:
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    class AlexNet(nn.Module):
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
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            return x
    
    # Create model
    model = AlexNet(num_classes=1000)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AlexNet total parameters: {total_params:,}")
    
    # Check layer sizes
    x = torch.randn(1, 3, 224, 224)
    print(f"\nInput: {x.shape}")
    
    for i, layer in enumerate(model.features):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"Layer {i} ({layer.__class__.__name__}): {x.shape}")
    

### VGGNet (2014): The Aesthetics of Simplicity

**VGGNet** demonstrated the effectiveness of deep networks with a simple design that repeatedly uses small 3×3 filters.

Design principles:

  * Uses **only 3×3 filters** (stacking small filters is more efficient)
  * Gradually halves size with **2×2 Max Pooling**
  * **Doubles channel count** while going deeper (64 → 128 → 256 → 512)
  * VGG-16 (16 layers) and VGG-19 (19 layers) are famous

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class VGGBlock(nn.Module):
        """Basic VGG block: repeat Conv → ReLU"""
        def __init__(self, in_channels, out_channels, num_convs):
            super(VGGBlock, self).__init__()
    
            layers = []
            for i in range(num_convs):
                layers.append(nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ))
                layers.append(nn.ReLU(inplace=True))
    
            self.block = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.block(x)
    
    class VGG16(nn.Module):
        def __init__(self, num_classes=1000):
            super(VGG16, self).__init__()
    
            # Feature extraction part
            self.features = nn.Sequential(
                VGGBlock(3, 64, 2),      # Block 1: 64 channels, 2 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(64, 128, 2),    # Block 2: 128 channels, 2 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(128, 256, 3),   # Block 3: 256 channels, 3 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(256, 512, 3),   # Block 4: 512 channels, 3 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(512, 512, 3),   # Block 5: 512 channels, 3 convs
                nn.MaxPool2d(2, 2),
            )
    
            # Classification part
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Create model
    model = VGG16(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VGG-16 total parameters: {total_params:,}")
    
    # Why stack 3×3 filters twice?
    print("\nTwo 3×3 filters vs one 5×5 filter:")
    print("Receptive field: Same 5×5")
    print("Parameters: 3×3×2 = 18 < 5×5 = 25")
    print("Non-linearity: 2 ReLUs > 1 ReLU (higher expressiveness)")
    

### ResNet (2015): The Revolution of Skip Connections

**ResNet (Residual Network)** introduced **Skip Connections (residual connections)** , enabling the training of very deep networks (over 100 layers).

Problem: Deeper networks should perform better, but in practice the **vanishing gradient problem** makes training difficult.

Solution: Introduce **Residual Blocks**
    
    
    ```mermaid
    graph TD
        A["Input x"] --> B["Conv + ReLU"]
        B --> C["Conv"]
        A --> D["Identity(as is)"]
        C --> E["Addition +"]
        D --> E
        E --> F["ReLU"]
        F --> G["Output"]
    
        style A fill:#e1f5ff
        style D fill:#fff9c4
        style E fill:#c8e6c9
        style G fill:#4fc3f7
    ```

Mathematical expression:

$$ \mathbf{y} = F(\mathbf{x}) + \mathbf{x} $$ 

Where $F(\mathbf{x})$ is the residual function (the part to be learned) and $\mathbf{x}$ is the shortcut connection.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ResidualBlock(nn.Module):
        """Basic ResNet block"""
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
    
            # Main path
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
    
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
            # Shortcut path (adjustment when input/output channels differ)
            self.downsample = downsample
    
        def forward(self, x):
            identity = x
    
            # Main path
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
    
            # Shortcut connection
            if self.downsample is not None:
                identity = self.downsample(x)
    
            # Addition
            out += identity
            out = F.relu(out)
    
            return out
    
    class SimpleResNet(nn.Module):
        """Simplified ResNet (for CIFAR-10)"""
        def __init__(self, num_classes=10):
            super(SimpleResNet, self).__init__()
    
            # Initial layer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
    
            # Residual blocks
            self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
            self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
    
            # Classification layer
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)
    
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
    
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
    
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
    
            return nn.Sequential(*layers)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
    
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
            return x
    
    # Create model
    model = SimpleResNet(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet total parameters: {total_params:,}")
    
    # Visualize Skip Connection effect
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"\nInput: {x.shape} → Output: {output.shape}")
    

### Why Skip Connections are Effective

**Theoretical Background of Skip Connections**

In conventional networks, deepening causes the vanishing gradient problem:

$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} $$ 

With deep layers, $\frac{\partial y}{\partial x}$ is multiplied many times, causing gradients to vanish.

With Skip Connections:

$$ \frac{\partial}{\partial x}(F(x) + x) = \frac{\partial F(x)}{\partial x} + 1 $$ 

Because of the "+1" term, gradients always flow!

Furthermore, the network only needs to "learn the identity mapping" → learning becomes easier.

* * *

## 2.3 Batch Normalization

### What is Batch Normalization?

**Batch Normalization (BN)** is a technique that normalizes the output of each mini-batch to stabilize learning.

Normalize the output of each layer to mean 0 and variance 1 across the mini-batch:

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$ $$ y_i = \gamma \hat{x}_i + \beta $$ 

Where:

  * $\mu_B, \sigma_B^2$: Mean and variance of the mini-batch
  * $\gamma, \beta$: Learnable parameters (scale and shift)
  * $\epsilon$: Small value for numerical stability (e.g., 1e-5)

### Effects of Batch Normalization

  * **Faster learning** : Can use larger learning rates
  * **Gradient stabilization** : Suppresses Internal Covariate Shift
  * **Regularization effect** : Reduces need for Dropout
  * **Reduced dependency on initialization** : Weight initialization becomes easier

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConvBlockWithoutBN(nn.Module):
        """Without Batch Normalization"""
        def __init__(self, in_channels, out_channels):
            super(ConvBlockWithoutBN, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
        def forward(self, x):
            return F.relu(self.conv(x))
    
    class ConvBlockWithBN(nn.Module):
        """With Batch Normalization"""
        def __init__(self, in_channels, out_channels):
            super(ConvBlockWithBN, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
    
        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))
    
    # Comparison experiment
    x = torch.randn(32, 3, 32, 32)  # Batch size 32
    
    # Without BN
    block_without_bn = ConvBlockWithoutBN(3, 64)
    output_without_bn = block_without_bn(x)
    
    # With BN
    block_with_bn = ConvBlockWithBN(3, 64)
    output_with_bn = block_with_bn(x)
    
    print("=== Effect of Batch Normalization ===")
    print(f"Without BN - Mean: {output_without_bn.mean():.4f}, Std: {output_without_bn.std():.4f}")
    print(f"With BN - Mean: {output_with_bn.mean():.4f}, Std: {output_with_bn.std():.4f}")
    
    # Visualize distribution
    print("\nStatistics per channel (with BN):")
    for i in range(min(5, output_with_bn.size(1))):
        channel_data = output_with_bn[:, i, :, :]
        print(f"  Channel {i}: Mean={channel_data.mean():.4f}, Std={channel_data.std():.4f}")
    

### Placement of Batch Normalization

BN is typically placed in the order: **Conv → BN → Activation** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: BN is typically placed in the order:Conv → BN → Activation:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch.nn as nn
    
    # Recommended order
    class StandardConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(StandardConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
    
        def forward(self, x):
            x = self.conv(x)    # 1. Convolution
            x = self.bn(x)      # 2. Batch Normalization
            x = self.relu(x)    # 3. Activation function
            return x
    
    # Note: Specify bias=False in Conv
    # Reason: BN makes the mean 0, so bias is unnecessary
    block = StandardConvBlock(3, 64)
    print("Conv → BN → ReLU order")
    print(block)
    

* * *

## 2.4 Overfitting Prevention with Dropout

### What is Dropout?

**Dropout** is a regularization technique that prevents overfitting by randomly disabling (dropping out) neurons during training.

  * During training: Randomly set neurons to 0 with probability $p$
  * During testing: Use all neurons (with scaling)

    
    
    ```mermaid
    graph TD
        A["During Training"] --> B["All Neurons"]
        B --> C["Drop 50% Randomly"]
        C --> D["Learn with Remaining 50%"]
    
        E["During Testing"] --> F["Use All Neurons"]
        F --> G["Scale Weights by 0.5"]
    
        style A fill:#e1f5ff
        style E fill:#c8e6c9
        style D fill:#b3e5fc
        style G fill:#81d4fa
    ```

### Why is Dropout Effective?

  * **Ensemble effect** : Train different sub-networks each time → similar to averaging multiple models
  * **Prevent co-adaptation** : Learn robust representations that don't depend on specific neurons
  * **Regularization** : Suppresses model complexity

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Why is Dropout Effective?
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Verify Dropout behavior
    x = torch.ones(4, 10)  # Tensor of all 1s
    
    dropout = nn.Dropout(p=0.5)  # 50% probability of dropping
    
    # Training mode
    dropout.train()
    print("=== Training Mode (Dropout Enabled) ===")
    for i in range(3):
        output = dropout(x)
        print(f"Trial {i+1}: {output[0, :5].numpy()}")  # Display first 5 elements
    
    # Evaluation mode
    dropout.eval()
    print("\n=== Evaluation Mode (Dropout Disabled) ===")
    output = dropout(x)
    print(f"Output: {output[0, :5].numpy()}")
    

### How to Use Dropout in CNNs

In CNNs, Dropout is typically placed **before fully connected layers**. It's not commonly used in convolutional layers.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: In CNNs, Dropout is typically placedbefore fully connected l
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch.nn as nn
    
    class CNNWithDropout(nn.Module):
        def __init__(self, num_classes=10):
            super(CNNWithDropout, self).__init__()
    
            # Convolutional layers (without Dropout)
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
    
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
    
            # Fully connected layers (with Dropout)
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),  # Dropout
    
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),  # Dropout
    
                nn.Linear(256, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = CNNWithDropout(num_classes=10)
    print(model)
    
    # Experiment with Dropout effect
    model.train()
    x = torch.randn(2, 3, 32, 32)
    output1 = model(x)
    output2 = model(x)
    print(f"\nTraining mode: Same input produces different outputs = {not torch.allclose(output1, output2)}")
    
    model.eval()
    output3 = model(x)
    output4 = model(x)
    print(f"Evaluation mode: Same input produces same output = {torch.allclose(output3, output4)}")
    

### Dropout vs Batch Normalization

Item | Dropout | Batch Normalization  
---|---|---  
**Main Purpose** | Overfitting prevention | Learning stabilization & acceleration  
**Where Used** | Fully connected layers | Convolutional layers  
**Train/Test** | Different behavior | Different behavior  
**Combined Use** | Possible (though BN may make it unnecessary) | -  
  
> **Modern Best Practice** : Use Batch Normalization in convolutional layers and Dropout in fully connected layers (as needed). However, due to BN's regularization effect, Dropout often becomes unnecessary.

* * *

## 2.5 Practical: CIFAR-10 Image Classification

### CIFAR-10 Dataset

**CIFAR-10** is a dataset consisting of 32×32 color images (60,000 total) in 10 classes:

  * Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  * Training data: 50,000 images
  * Test data: 10,000 images

### Complete CNN Classifier Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Complete CNN Classifier Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"Training data: {len(train_dataset)} images")
    print(f"Test data: {len(test_dataset)} images")
    
    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    

### Modern CNN Architecture
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CIFAR10Net(nn.Module):
        """Modern CNN for CIFAR-10"""
        def __init__(self, num_classes=10):
            super(CIFAR10Net, self).__init__()
    
            # Block 1
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
    
            # Block 2
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
    
            # Block 3
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(256)
    
            # Block 4
            self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
            self.bn5 = nn.BatchNorm2d(512)
            self.conv6 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
            self.bn6 = nn.BatchNorm2d(512)
    
            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
    
            # Classification layer
            self.fc = nn.Linear(512, num_classes)
    
            # Dropout
            self.dropout = nn.Dropout(0.5)
    
        def forward(self, x):
            # Block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 2
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 3
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 4
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
    
            # Global Average Pooling
            x = self.gap(x)
            x = x.view(x.size(0), -1)
    
            # Dropout + classification
            x = self.dropout(x)
            x = self.fc(x)
    
            return x
    
    # Create model
    model = CIFAR10Net(num_classes=10).to(device)
    print(model)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    

### Training Loop
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Training Loop
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch.optim as optim
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Backward
            loss.backward()
            optimizer.step()
    
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def test_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                outputs = model(inputs)
                loss = criterion(outputs, labels)
    
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    # Execute training
    num_epochs = 50
    best_acc = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
        scheduler.step()
    
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    

### Prediction and Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def imshow(img, title=None):
        """Helper function for image display"""
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
    
    # Get samples from test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # Display first 8 images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        imshow(images[i].cpu(), title=f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
    
    plt.tight_layout()
    plt.savefig('cifar10_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved prediction results: cifar10_predictions.png")
    

* * *

## 2.6 Overview of Modern Architectures

### EfficientNet (2019): Efficient Scaling

**EfficientNet** proposed a method to **scale network depth, width, and resolution in a balanced way**.

Compound Scaling:

$$ \text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi $$ 

Constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

  * Achieves high accuracy with fewer parameters
  * Uses Mobile Inverted Bottleneck Convolution (MBConv)
  * Variations from EfficientNet-B0 to B7 for accuracy and size trade-offs

### Vision Transformer (2020+): Surpassing CNNs

**Vision Transformer (ViT)** is a new approach that **divides images into patches** and processes them with Transformers.
    
    
    ```mermaid
    graph LR
        A["Image224×224"] --> B["Patch Division16×16 patches"]
        B --> C["Linear Projection"]
        C --> D["Transformer Encoder"]
        D --> E["Classification Head"]
        E --> F["Class Prediction"]
    
        style A fill:#e1f5ff
        style D fill:#b3e5fc
        style F fill:#4fc3f7
    ```

Features:

  * Abandons CNN's inductive biases (locality, translation invariance)
  * Surpasses CNNs on large-scale data
  * Captures relationships across the entire image with Self-Attention
  * Likely to become mainstream in the future

### Architecture Selection Guidelines

Architecture | Features | Recommended Cases  
---|---|---  
**LeNet-5** | Simple, lightweight | MNIST, educational purposes  
**VGGNet** | Easy-to-understand structure | Transfer learning base, education  
**ResNet** | Deep network, stable | General image classification, standard choice  
**EfficientNet** | Efficient, high accuracy | Resource constraints, mobile  
**Vision Transformer** | State-of-the-art, large-scale data | Large datasets, research  
  
* * *

## Exercises

**Exercise 1: Effects of Pooling Layers**

Apply Max Pooling and Average Pooling to the same input and verify the differences in output. For what types of image features is each advantageous?
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Apply Max Pooling and Average Pooling to the same input and 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Create feature map with edges
    edge_feature = torch.tensor([[
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Exercise: Apply Max Pooling and Average Pooling and compare results
    # Hint: Use nn.MaxPool2d and nn.AvgPool2d
    

**Exercise 2: Skip Connection in Residual Block**

Compare Residual blocks with and without Skip Connections to verify differences in gradient flow.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare Residual blocks with and without Skip Connections to
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Implement blocks with and without Skip Connection
    # Hint: Same structure, only difference is presence of Skip Connection
    # To compare gradients, check grad attribute after backward()
    

**Exercise 3: Effect of Batch Normalization**

Train the same network with and without Batch Normalization and compare convergence speed and final accuracy.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Train the same network with and without Batch Normalization 
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Exercise: Create models with and without BN
    # Exercise: Train on MNIST or CIFAR-10
    # Exercise: Compare training curves
    

**Exercise 4: Regularization Effect of Dropout**

Vary Dropout probability (p=0.0, 0.3, 0.5, 0.7) and investigate the impact on overfitting.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Vary Dropout probability (p=0.0, 0.3, 0.5, 0.7) and investig
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Train models with different Dropout probabilities
    # Exercise: Compare difference between training and test errors (degree of overfitting)
    # Which Dropout probability is optimal?
    

**Exercise 5: Architecture Comparison on CIFAR-10**

Compare three architectures on CIFAR-10: LeNet-5, VGG-style, and ResNet-style.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare three architectures on CIFAR-10: LeNet-5, VGG-style,
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Implement three different architectures
    # Exercise: Compare performance with same training settings
    # Exercise: Record parameter count, training time, and accuracy
    
    # Evaluation metrics:
    # - Test accuracy
    # - Parameter count
    # - Training time (per epoch)
    # - Epochs until convergence
    

* * *

## Summary

In this chapter, we learned about pooling layers and representative CNN architectures.

### Key Points

  * **Pooling layers** : Provide dimensionality reduction and translation invariance. Choosing between Max Pooling and Average Pooling
  * **LeNet-5** : Foundation of CNNs. Basic structure of Conv → Pool → FC
  * **AlexNet** : Utilization of ReLU, Dropout, and Data Augmentation
  * **VGGNet** : Simple design with repeated 3×3 filters
  * **ResNet** : Solved vanishing gradients with Skip Connections, enabling networks over 100 layers
  * **Batch Normalization** : Stabilizes and accelerates learning. Conv → BN → ReLU order
  * **Dropout** : Prevents overfitting. Used in fully connected layers
  * **Global Average Pooling** : Reduces parameters, independent of input size

### Next Steps

In the next chapter, we will learn about **Transfer Learning** and **Fine-tuning**. We will master practical techniques for building high-accuracy models with limited data by leveraging pre-trained models.
