---
title: "Chapter 1: Fundamentals of CNN and Convolutional Layers"
chapter_title: "Chapter 1: Fundamentals of CNN and Convolutional Layers"
subtitle: Revolution in Image Recognition - Understanding the Basic Principles of Convolutional Neural Networks
reading_time: 25-30 minutes
difficulty: Beginner to Intermediate
code_examples: 11
exercises: 5
---

This chapter covers the fundamentals of Fundamentals of CNN and Convolutional Layers, which forms the foundation of this area. You will learn mathematical definition, roles of stride, and concepts of feature maps.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the challenges of traditional image recognition methods and the advantages of CNNs
  * ✅ Master the mathematical definition and computational process of convolution operations
  * ✅ Understand the roles of stride, padding, and kernel size
  * ✅ Explain the concepts of feature maps and receptive fields
  * ✅ Implement Conv2d layers in PyTorch and calculate parameters
  * ✅ Understand filter visualization and feature extraction mechanisms

* * *

## 1.1 Challenges in Image Recognition and the Emergence of CNNs

### Limitations of Traditional Image Recognition Methods

When using **Fully Connected Networks** for image recognition, serious problems arise.

> "Images are two-dimensional data with spatial structure. Ignoring this structure leads to an explosion of parameters and overfitting."

#### Problem 1: Explosion of Parameter Count

For example, when inputting a 224×224 pixel color image (RGB) to a fully connected layer:

  * Input dimensions: $224 \times 224 \times 3 = 150,528$
  * If hidden layer has 1,000 neurons: $150,528 \times 1,000 = 150,528,000$ parameters
  * That's over 150 million parameters in just the first layer!

#### Problem 2: Lack of Translation Invariance

In fully connected layers, even a slight change in object position within an image is treated as a completely different input.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: In fully connected layers, even a slight change in object po
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Simple example: Representing "cat" features (ears) in a 5×5 image
    original = np.zeros((5, 5))
    original[0, 1] = 1  # Left ear
    original[0, 3] = 1  # Right ear
    original[2, 2] = 1  # Nose
    
    # Shifted 1 pixel to the right
    shifted = np.zeros((5, 5))
    shifted[0, 2] = 1  # Left ear
    shifted[0, 4] = 1  # Right ear
    shifted[2, 3] = 1  # Nose
    
    print("Original image flattened:", original.flatten())
    print("Shifted image flattened:", shifted.flatten())
    print(f"Euclidean distance: {np.linalg.norm(original.flatten() - shifted.flatten()):.2f}")
    
    # In fully connected layers, these two are treated as completely different inputs
    print("\nConclusion: Fully connected layers cannot handle minor positional changes")
    

**Output** :
    
    
    Original image flattened: [0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    Shifted image flattened: [0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    Euclidean distance: 2.45
    
    Conclusion: Fully connected layers cannot handle minor positional changes
    

### Three Important Properties of CNNs

**Convolutional Neural Networks (CNNs)** leverage the spatial structure of images with the following properties:

Property | Explanation | Effect  
---|---|---  
**Local Connectivity** | Each neuron connects only to a small region of the input | Reduction in parameter count  
**Weight Sharing** | Same filter used across the entire image | Acquisition of translation invariance  
**Hierarchical Feature Learning** | Progressively extracts low-level to high-level features | Complex pattern recognition  
  
### Overall Structure of CNNs
    
    
    ```mermaid
    graph LR
        A[Input Image28×28×1] --> B[Conv Layer26×26×32]
        B --> C[ActivationReLU]
        C --> D[Pooling13×13×32]
        D --> E[Conv Layer11×11×64]
        E --> F[ActivationReLU]
        F --> G[Pooling5×5×64]
        G --> H[Flatten1600]
        H --> I[FC Layer128]
        I --> J[Output Layer10 classes]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#fce4ec
        style E fill:#fff3e0
        style F fill:#e8f5e9
        style G fill:#fce4ec
        style H fill:#f3e5f5
        style I fill:#fff9c4
        style J fill:#ffebee
    ```

* * *

## 1.2 Fundamentals of Convolution Operations

### What is Convolution?

**Convolution** is an operation where a filter (kernel) is slid across an image while performing element-wise multiplication and summation.

#### Mathematical Definition

Two-dimensional discrete convolution is defined as follows:

$$ (I * K)(i, j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m, n) $$ 

Where:

  * $I$: Input image
  * $K$: Kernel or Filter
  * $(i, j)$: Output position
  * $(m, n)$: Position within kernel

#### Concrete Example: Convolution with 3×3 Kernel
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Concrete Example: Convolution with 3×3 Kernel
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Input image (5×5)
    image = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6]
    ])
    
    # Edge detection kernel (3×3)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    def manual_convolution(image, kernel):
        """
        Manually execute convolution operation
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
    
        # Calculate output size
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
    
        output = np.zeros((out_h, out_w))
    
        # Convolution operation
        for i in range(out_h):
            for j in range(out_w):
                # Extract image region
                region = image[i:i+ker_h, j:j+ker_w]
                # Sum of element-wise products
                output[i, j] = np.sum(region * kernel)
    
        return output
    
    # Execute convolution
    result = manual_convolution(image, kernel)
    
    print("Input image (5×5):")
    print(image)
    print("\nKernel (3×3, edge detection):")
    print(kernel)
    print("\nOutput (3×3):")
    print(result)
    
    # Detailed calculation example (top-left position)
    print("\n=== Calculation Example (Position [0, 0]) ===")
    region = image[0:3, 0:3]
    print("Image region:")
    print(region)
    print("\nKernel:")
    print(kernel)
    print("\nElement-wise product:")
    print(region * kernel)
    print(f"\nSum: {np.sum(region * kernel)}")
    

**Output** :
    
    
    Input image (5×5):
    [[1 2 3 0 1]
     [4 5 6 1 2]
     [7 8 9 2 3]
     [1 2 3 4 5]
     [2 3 4 5 6]]
    
    Kernel (3×3, edge detection):
    [[-1 -1 -1]
     [-1  8 -1]
     [-1 -1 -1]]
    
    Output (3×3):
    [[-13. -15. -12.]
     [ -8.  -9.  -6.]
     [ -5.  -6.   0.]]
    
    === Calculation Example (Position [0, 0]) ===
    Image region:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    Kernel:
    [[-1 -1 -1]
     [-1  8 -1]
     [-1 -1 -1]]
    
    Element-wise product:
    [[-1 -2 -3]
     [-4 40 -6]
     [-7 -8 -9]]
    
    Sum: -13
    

### Filters and Kernels

**Kernel** and **Filter** are often used interchangeably, but strictly speaking:

  * **Kernel** : A 2D weight array (e.g., 3×3 matrix)
  * **Filter** : A collection of kernels across all channels (e.g., 3×3×3 filter for RGB images)

#### Examples of Representative Kernels
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - scipy>=1.11.0
    
    """
    Example: Examples of Representative Kernels
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # Define various kernels
    kernels = {
        "Identity": np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),
    
        "Edge Detection (Vertical)": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),  # Sobel filter
    
        "Edge Detection (Horizontal)": np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]]),
    
        "Smoothing (blur)": np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]) / 9,
    
        "Sharpening": np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]])
    }
    
    # Test image (simple pattern)
    test_image = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    # Visualize effects of each kernel
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for idx, (name, kernel) in enumerate(kernels.items(), 1):
        result = signal.correlate2d(test_image, kernel, mode='same', boundary='symm')
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    print("Visualized kernel effects")
    

### Stride and Padding

#### Stride

**Stride** is the step size when moving the kernel.

  * Stride = 1: Kernel moves 1 pixel at a time (standard)
  * Stride = 2: Kernel moves 2 pixels at a time (output size halved)

Output size calculation formula:

$$ \text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size}}{\text{Stride}} \right\rfloor + 1 $$ 

#### Padding

**Padding** is the operation of adding values (typically 0) around the input image.

Padding Type | Explanation | Use Case  
---|---|---  
**Valid** | No padding | When reducing output size  
**Same** | Adjusted so output size = input size | When maintaining spatial size  
**Full** | So that entire kernel overlaps with image | When maximizing use of boundary information  
  
Padding amount calculation for Same padding:

$$ \text{Padding} = \frac{\text{Kernel Size} - 1}{2} $$ 
    
    
    def calculate_output_size(input_size, kernel_size, stride, padding):
        """
        Calculate output size after convolution operation
    
        Parameters:
        -----------
        input_size : int
            Input height or width
        kernel_size : int
            Kernel height or width
        stride : int
            Stride
        padding : int
            Padding amount
    
        Returns:
        --------
        int : Output size
        """
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    # Calculate output sizes for various configurations
    print("=== Output Size Calculation Examples ===\n")
    
    configurations = [
        (28, 3, 1, 0, "Valid (no padding)"),
        (28, 3, 1, 1, "Same (maintain size)"),
        (28, 5, 2, 2, "Stride 2, Padding 2"),
        (32, 3, 1, 1, "32×32 image, 3×3 kernel"),
    ]
    
    for input_size, kernel_size, stride, padding, description in configurations:
        output_size = calculate_output_size(input_size, kernel_size, stride, padding)
        print(f"{description}")
        print(f"  Input: {input_size}×{input_size}")
        print(f"  Kernel: {kernel_size}×{kernel_size}, Stride: {stride}, Padding: {padding}")
        print(f"  → Output: {output_size}×{output_size}\n")
    

**Output** :
    
    
    === Output Size Calculation Examples ===
    
    Valid (no padding)
      Input: 28×28
      Kernel: 3×3, Stride: 1, Padding: 0
      → Output: 26×26
    
    Same (maintain size)
      Input: 28×28
      Kernel: 3×3, Stride: 1, Padding: 1
      → Output: 28×28
    
    Stride 2, Padding 2
      Input: 28×28
      Kernel: 5×5, Stride: 2, Padding: 2
      → Output: 14×14
    
    32×32 image, 3×3 kernel
      Input: 32×32
      Kernel: 3×3, Stride: 1, Padding: 1
      → Output: 32×32
    

* * *

## 1.3 Feature Maps and Receptive Fields

### Feature Maps

**Feature maps** are the output results of convolution operations. Each filter detects different features (edges, textures, etc.) and generates respective feature maps.

  * Number of input channels = $C_{in}$
  * Number of output channels = $C_{out}$ (number of filters)
  * Size of each filter = $K \times K \times C_{in}$

#### Multi-Channel Convolution Calculation

For color images (RGB, 3 channels):

$$ \text{Output}(i, j) = \sum_{c=1}^{3} \sum_{m}\sum_{n} I_c(i+m, j+n) \cdot K_c(m, n) + b $$ 

Where $b$ is the bias term.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Where $b$ is the bias term.
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # RGB image (batch size 1, 3 channels, 28×28)
    input_image = torch.randn(1, 3, 28, 28)
    
    # Define convolutional layer
    # Input: 3 channels (RGB)
    # Output: 16 channels (16 feature maps)
    # Kernel size: 3×3
    conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Forward pass
    output = conv_layer(input_image)
    
    print(f"Input size: {input_image.shape}")
    print(f"  → [Batch, Channel, Height, Width] = [1, 3, 28, 28]")
    print(f"\nConvolutional layer parameters:")
    print(f"  Input channels: 3")
    print(f"  Output channels: 16")
    print(f"  Kernel size: 3×3")
    print(f"  Padding: 1 (Same padding)")
    print(f"\nOutput size: {output.shape}")
    print(f"  → [Batch, Channel, Height, Width] = [1, 16, 28, 28]")
    
    # Calculate parameter count
    weight_params = 3 * 16 * 3 * 3  # in_ch × out_ch × k_h × k_w
    bias_params = 16  # One per output channel
    total_params = weight_params + bias_params
    
    print(f"\nParameter count:")
    print(f"  Weights: {weight_params:,} (= 3 × 16 × 3 × 3)")
    print(f"  Bias: {bias_params}")
    print(f"  Total: {total_params:,}")
    

**Output** :
    
    
    Input size: torch.Size([1, 3, 28, 28])
      → [Batch, Channel, Height, Width] = [1, 3, 28, 28]
    
    Convolutional layer parameters:
      Input channels: 3
      Output channels: 16
      Kernel size: 3×3
      Padding: 1 (Same padding)
    
    Output size: torch.Size([1, 16, 28, 28])
      → [Batch, Channel, Height, Width] = [1, 16, 28, 28]
    
    Parameter count:
      Weights: 432 (= 3 × 16 × 3 × 3)
      Bias: 16
      Total: 448
    

### Receptive Field

The **receptive field** is the region of the input image that a particular output neuron "sees". In CNNs, the receptive field expands as layers are stacked.

#### Receptive Field Size Calculation

Receptive field size $R$ calculation formula (with stride 1 and padding):

$$ R_l = R_{l-1} + (K_l - 1) $$ 

Where:

  * $R_l$: Receptive field size at layer $l$
  * $K_l$: Kernel size at layer $l$
  * $R_0 = 1$ (input layer)

    
    
    def calculate_receptive_field(layers_info):
        """
        Calculate CNN receptive field size
    
        Parameters:
        -----------
        layers_info : list of tuples
            List of (kernel_size, stride) for each layer
    
        Returns:
        --------
        list : Receptive field size for each layer
        """
        receptive_fields = [1]  # Input layer
    
        for kernel_size, stride in layers_info:
            # Simplified calculation (for stride 1)
            rf = receptive_fields[-1] + (kernel_size - 1)
            receptive_fields.append(rf)
    
        return receptive_fields
    
    # VGG-style network configuration
    vgg_layers = [
        (3, 1),  # Conv1
        (3, 1),  # Conv2
        (2, 2),  # MaxPool
        (3, 1),  # Conv3
        (3, 1),  # Conv4
        (2, 2),  # MaxPool
    ]
    
    receptive_fields = calculate_receptive_field(vgg_layers)
    
    print("=== Receptive Field Expansion Process ===\n")
    print("Layer                Receptive Field Size")
    print("-" * 35)
    print(f"Input layer            {receptive_fields[0]}×{receptive_fields[0]}")
    
    layer_names = ["Conv1 (3×3)", "Conv2 (3×3)", "MaxPool (2×2)",
                   "Conv3 (3×3)", "Conv4 (3×3)", "MaxPool (2×2)"]
    
    for i, name in enumerate(layer_names, 1):
        print(f"{name:18}{receptive_fields[i]:2}×{receptive_fields[i]:2}")
    
    print(f"\nFinal receptive field: {receptive_fields[-1]}×{receptive_fields[-1]} pixels")
    

**Output** :
    
    
    === Receptive Field Expansion Process ===
    
    Layer                Receptive Field Size
    -----------------------------------
    Input layer             1×1
    Conv1 (3×3)        3×3
    Conv2 (3×3)        5×5
    MaxPool (2×2)      6×6
    Conv3 (3×3)        8×8
    Conv4 (3×3)       10×10
    MaxPool (2×2)     11×11
    
    Final receptive field: 11×11 pixels
    

#### Receptive Field Visualization
    
    
    ```mermaid
    graph TD
        subgraph "Input Image"
        A1[" "]
        A2[" "]
        A3[" "]
        A4[" "]
        A5[" "]
        end
    
        subgraph "Conv1: 3×3 Kernel"
        B1[Receptive field: 3×3]
        end
    
        subgraph "Conv2: 3×3 Kernel"
        C1[Receptive field: 5×5]
        end
    
        subgraph "Conv3: 3×3 Kernel"
        D1[Receptive field: 7×7]
        end
    
        A1 --> B1
        A2 --> B1
        A3 --> B1
        B1 --> C1
        C1 --> D1
    
        style B1 fill:#fff3e0
        style C1 fill:#ffe0b2
        style D1 fill:#ffcc80
    ```

> **Important** : Deeper networks have larger receptive fields and can integrate information from wider areas. This is the source of deep learning's powerful feature extraction capability.

* * *

## 1.4 Implementing Convolutional Layers in PyTorch

### Basic Usage of Conv2d

In PyTorch, we use the `torch.nn.Conv2d` class to define convolutional layers.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: In PyTorch, we use thetorch.nn.Conv2dclass to define convolu
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Basic Conv2d syntax
    conv = nn.Conv2d(
        in_channels=3,      # Number of input channels (3 for RGB)
        out_channels=64,    # Number of output channels (number of filters)
        kernel_size=3,      # Kernel size (3×3)
        stride=1,           # Stride
        padding=1,          # Padding
        bias=True           # Whether to use bias term
    )
    
    # Dummy input (batch size 8, RGB image, 224×224)
    x = torch.randn(8, 3, 224, 224)
    
    # Forward pass
    output = conv(x)
    
    print("=== Conv2d Operation Verification ===\n")
    print(f"Input size: {x.shape}")
    print(f"  [Batch, Channel, Height, Width] = [{x.shape[0]}, {x.shape[1]}, {x.shape[2]}, {x.shape[3]}]")
    print(f"\nOutput size: {output.shape}")
    print(f"  [Batch, Channel, Height, Width] = [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}, {output.shape[3]}]")
    
    # Parameter details
    print(f"\nParameter details:")
    print(f"  Weight size: {conv.weight.shape}")
    print(f"  → [Output ch, Input ch, Height, Width] = [{conv.weight.shape[0]}, {conv.weight.shape[1]}, {conv.weight.shape[2]}, {conv.weight.shape[3]}]")
    print(f"  Bias size: {conv.bias.shape}")
    print(f"  → [Output ch] = [{conv.bias.shape[0]}]")
    
    # Parameter count
    total_params = conv.weight.numel() + conv.bias.numel()
    print(f"\nTotal parameter count: {total_params:,}")
    print(f"  Formula: (3 × 64 × 3 × 3) + 64 = {total_params:,}")
    

**Output** :
    
    
    === Conv2d Operation Verification ===
    
    Input size: torch.Size([8, 3, 224, 224])
      [Batch, Channel, Height, Width] = [8, 3, 224, 224]
    
    Output size: torch.Size([8, 64, 224, 224])
      [Batch, Channel, Height, Width] = [8, 64, 224, 224]
    
    Parameter details:
      Weight size: torch.Size([64, 3, 3, 3])
      → [Output ch, Input ch, Height, Width] = [64, 3, 3, 3]
      Bias size: torch.Size([64])
      → [Output ch] = [64]
    
    Total parameter count: 1,792
      Formula: (3 × 64 × 3 × 3) + 64 = 1,792
    

### Parameter Count Calculation Formula

The parameter count for convolutional layers is calculated by the following formula:

$$ \text{Parameter Count} = (C_{in} \times K_h \times K_w \times C_{out}) + C_{out} $$ 

Where:

  * $C_{in}$: Number of input channels
  * $C_{out}$: Number of output channels (number of filters)
  * $K_h, K_w$: Kernel height and width
  * The last $C_{out}$ is for bias terms

    
    
    def calculate_conv_params(in_channels, out_channels, kernel_size, bias=True):
        """
        Calculate parameter count for convolutional layer
        """
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
    
        weight_params = in_channels * out_channels * kernel_h * kernel_w
        bias_params = out_channels if bias else 0
    
        return weight_params + bias_params
    
    # Calculate parameter counts for various configurations
    print("=== Convolutional Layer Parameter Count Comparison ===\n")
    
    configs = [
        (3, 32, 3, "Layer 1 (RGB → 32 channels)"),
        (32, 64, 3, "Layer 2 (32 → 64 channels)"),
        (64, 128, 3, "Layer 3 (64 → 128 channels)"),
        (128, 256, 3, "Layer 4 (128 → 256 channels)"),
        (3, 64, 7, "Large kernel (7×7)"),
        (512, 512, 3, "Deep layer (512 → 512 channels)"),
    ]
    
    for in_ch, out_ch, k_size, description in configs:
        params = calculate_conv_params(in_ch, out_ch, k_size)
        print(f"{description}")
        print(f"  Configuration: {in_ch}ch → {out_ch}ch, Kernel{k_size}×{k_size}")
        print(f"  Parameter count: {params:,}\n")
    
    # Comparison with fully connected layer
    print("=== Comparison with Fully Connected Layer ===\n")
    fc_input = 224 * 224 * 3
    fc_output = 1000
    fc_params = fc_input * fc_output + fc_output
    
    print(f"Fully connected layer (224×224×3 → 1000):")
    print(f"  Parameter count: {fc_params:,}")
    
    conv_params = calculate_conv_params(3, 64, 3)
    print(f"\nConvolutional layer (3ch → 64ch, 3×3):")
    print(f"  Parameter count: {conv_params:,}")
    print(f"\nReduction rate: {(1 - conv_params/fc_params)*100:.2f}%")
    

**Output** :
    
    
    === Convolutional Layer Parameter Count Comparison ===
    
    Layer 1 (RGB → 32 channels)
      Configuration: 3ch → 32ch, Kernel3×3
      Parameter count: 896
    
    Layer 2 (32 → 64 channels)
      Configuration: 32ch → 64ch, Kernel3×3
      Parameter count: 18,496
    
    Layer 3 (64 → 128 channels)
      Configuration: 64ch → 128ch, Kernel3×3
      Parameter count: 73,856
    
    Layer 4 (128 → 256 channels)
      Configuration: 128ch → 256ch, Kernel3×3
      Parameter count: 295,168
    
    Large kernel (7×7)
      Configuration: 3ch → 64ch, Kernel7×7
      Parameter count: 9,472
    
    Deep layer (512 → 512 channels)
      Configuration: 512ch → 512ch, Kernel3×3
      Parameter count: 2,359,808
    
    === Comparison with Fully Connected Layer ===
    
    Fully connected layer (224×224×3 → 1000):
      Parameter count: 150,529,000
    
    Convolutional layer (3ch → 64ch, 3×3):
      Parameter count: 1,792
    
    Reduction rate: 100.00%
    

### Convolutional Filter Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Convolutional Filter Visualization
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import torch.nn as nn
    
    # Define convolutional layer
    conv_layer = nn.Conv2d(1, 8, kernel_size=3, padding=1)
    
    # Visualize trained filters (here using random initialization)
    filters = conv_layer.weight.data.cpu().numpy()
    
    # Display 8 filters in 2 rows × 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(8):
        # filters[i, 0] is the i-th filter (1st channel)
        axes[i].imshow(filters[i, 0], cmap='gray')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Convolutional Filter Visualization (3×3 kernel)', fontsize=16)
    plt.tight_layout()
    print("Visualized filters (random initialization)")
    print("After training, filters evolve to have features like edge detection and texture detection")
    

* * *

## 1.5 Activation Function: ReLU

### Why Activation Functions Are Needed

Convolution operations are linear transformations. Without activation functions, stacking multiple layers would simply be a combination of linear transformations, unable to learn complex patterns.

> Activation functions introduce **non-linearity** , giving the network the ability to approximate complex functions.

### ReLU (Rectified Linear Unit)

The most commonly used activation function in CNNs is **ReLU**.

$$ \text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\\ 0 & \text{if } x \leq 0 \end{cases} $$ 

#### Advantages of ReLU

Advantage | Explanation  
---|---  
**Computational Efficiency** | Only simple max operation  
**Mitigates Vanishing Gradients** | Gradient is 1 in positive region (better than Sigmoid or Tanh)  
**Sparsity** | Setting negative values to 0 creates sparse representations  
**Biological Plausibility** | Similar to neuron firing patterns  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Advantages of ReLU
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Comparison of various activation functions
    x = np.linspace(-3, 3, 100)
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(x, relu, 'b-', linewidth=2)
    axes[0, 0].set_title('ReLU: max(0, x)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    axes[0, 1].plot(x, sigmoid, 'r-', linewidth=2)
    axes[0, 1].set_title('Sigmoid: 1/(1+exp(-x))', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    axes[1, 0].plot(x, tanh, 'g-', linewidth=2)
    axes[1, 0].set_title('Tanh: tanh(x)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    axes[1, 1].plot(x, leaky_relu, 'm-', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU: max(0.1x, x)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    print("Compared activation function shapes")
    
    # Usage example in PyTorch
    print("\n=== Using Activation Functions in PyTorch ===\n")
    
    x_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    relu_layer = nn.ReLU()
    print(f"Input: {x_tensor.numpy()}")
    print(f"ReLU: {relu_layer(x_tensor).numpy()}")
    

**Output** :
    
    
    Compared activation function shapes
    
    === Using Activation Functions in PyTorch ===
    
    Input: [-2. -1.  0.  1.  2.]
    ReLU: [0. 0. 0. 1. 2.]
    

### Conv + ReLU Pattern
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Conv + ReLU Pattern
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Standard Conv-ReLU block
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x
    
    # Usage example
    block = ConvBlock(3, 64)
    x = torch.randn(1, 3, 224, 224)
    output = block(x)
    
    print(f"Input size: {x.shape}")
    print(f"Output size: {output.shape}")
    print(f"\nProcessing flow:")
    print(f"  1. Conv2d(3 → 64, 3×3) filtering")
    print(f"  2. ReLU() non-linear transformation")
    print(f"  → Negative values in feature map become 0")
    

**Output** :
    
    
    Input size: torch.Size([1, 3, 224, 224])
    Output size: torch.Size([1, 64, 224, 224])
    
    Processing flow:
      1. Conv2d(3 → 64, 3×3) filtering
      2. ReLU() non-linear transformation
      → Negative values in feature map become 0
    

* * *

## 1.6 Practice: Handwritten Digit Recognition (MNIST)

### Building a Simple CNN

We will implement a basic CNN to classify the MNIST dataset (28×28 grayscale handwritten digit images).
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: We will implement a basic CNN to classify the MNIST dataset 
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Convolutional layer 1: 1ch → 32ch, 3×3 kernel
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            # Convolutional layer 2: 32ch → 64ch, 3×3 kernel
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            # Others
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
    
        def forward(self, x):
            # Conv1 → ReLU → MaxPool
            x = self.pool(F.relu(self.conv1(x)))  # 28×28 → 14×14
            # Conv2 → ReLU → MaxPool
            x = self.pool(F.relu(self.conv2(x)))  # 14×14 → 7×7
            # Flatten
            x = x.view(-1, 64 * 7 * 7)
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Model instantiation
    model = SimpleCNN()
    
    # Display model structure
    print("=== SimpleCNN Architecture ===\n")
    print(model)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Parameter count details for each layer
    print("\n=== Parameter Count per Layer ===")
    for name, param in model.named_parameters():
        print(f"{name:20} {str(list(param.shape)):30} {param.numel():>10,} params")
    

**Output** :
    
    
    === SimpleCNN Architecture ===
    
    SimpleCNN(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fc1): Linear(in_features=3136, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (dropout): Dropout(p=0.25, inplace=False)
    )
    
    Total parameters: 421,066
    Trainable parameters: 421,066
    
    === Parameter Count per Layer ===
    conv1.weight         [32, 1, 3, 3]                         288 params
    conv1.bias           [32]                                   32 params
    conv2.weight         [64, 32, 3, 3]                     18,432 params
    conv2.bias           [64]                                   64 params
    fc1.weight           [128, 3136]                       401,408 params
    fc1.bias             [128]                                 128 params
    fc2.weight           [10, 128]                           1,280 params
    fc2.bias             [10]                                   10 params
    

### Data Preparation and Training
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Data Preparation and Training
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch.optim as optim
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets (download only on first run)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training function
    def train_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    # Evaluation function
    def evaluate(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0
        correct = 0
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy
    
    # Execute training (simplified version: 3 epochs)
    print("\n=== Training Started ===\n")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%\n")
    
    print("Training complete!")
    

**Expected Output** :
    
    
    === Training Started ===
    
    Epoch 1/3
      Train Loss: 0.2145, Train Acc: 93.52%
      Test Loss:  0.0789, Test Acc:  97.56%
    
    Epoch 2/3
      Train Loss: 0.0701, Train Acc: 97.89%
      Test Loss:  0.0512, Test Acc:  98.34%
    
    Epoch 3/3
      Train Loss: 0.0512, Train Acc: 98.42%
      Test Loss:  0.0401, Test Acc:  98.67%
    
    Training complete!
    

### Visualizing Trained Filters
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Visualizing Trained Filters
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Visualize first layer convolutional filters
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    
    # Display first 16 filters
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(32, len(axes))):
        axes[i].imshow(conv1_weights[i, 0], cmap='viridis')
        axes[i].set_title(f'Filter {i+1}', fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('Trained Convolutional Filters (Layer 1, 32 out of 32)', fontsize=16)
    plt.tight_layout()
    print("Visualized trained filters")
    print("Each filter has learned to detect different features such as edges, curves, and corners")
    

* * *

## Summary

In this chapter, we learned the fundamentals of CNNs and convolutional layers.

### Key Points

**1\. Local connectivity and weight sharing** — Enable CNNs to drastically reduce parameter count compared to fully connected networks.

**2\. Convolution operations** — Extract features by sliding filters across images, detecting edges, textures, and patterns.

**3\. Stride and padding** — Control output size and spatial dimension management.

**4\. Receptive field** — Expands with each layer, integrating information from progressively wider areas.

**5\. ReLU activation function** — Introduces non-linearity, enabling the network to learn complex patterns.

### Preview of Next Chapter

Chapter 2 will cover pooling layers (MaxPooling and AveragePooling), Batch Normalization techniques, regularization with Dropout, and representative CNN architectures including VGG and ResNet.

* * *

## Exercises

**Exercise 1: Output Size Calculation**

**Problem** : Calculate the output size for the following convolutional layer.

  * Input: 64×64×3
  * Kernel: 5×5
  * Stride: 2
  * Padding: 2
  * Output channels: 128

**Solution** :
    
    
    # Calculate output size
    output_h = (64 + 2*2 - 5) // 2 + 1 = 32
    output_w = (64 + 2*2 - 5) // 2 + 1 = 32
    
    # Answer: 32×32×128
    

**Exercise 2: Parameter Count Calculation**

**Problem** : Calculate the parameter count for the following CNN.

  * Conv1: 3ch → 64ch, 7×7 kernel
  * Conv2: 64ch → 128ch, 3×3 kernel
  * Conv3: 128ch → 256ch, 3×3 kernel

**Solution** :
    
    
    # Conv1 parameters
    conv1_params = (3 * 64 * 7 * 7) + 64 = 9,472
    
    # Conv2 parameters
    conv2_params = (64 * 128 * 3 * 3) + 128 = 73,856
    
    # Conv3 parameters
    conv3_params = (128 * 256 * 3 * 3) + 256 = 295,168
    
    # Total
    total_params = 9,472 + 73,856 + 295,168 = 378,496
    

**Exercise 3: Receptive Field Calculation**

**Problem** : Calculate the final receptive field size for a CNN with the following configuration (all with stride 1 and padding).

  * Conv1: 3×3 kernel
  * Conv2: 3×3 kernel
  * Conv3: 3×3 kernel
  * Conv4: 3×3 kernel

**Solution** :
    
    
    # Receptive field calculation
    # R_0 = 1 (input)
    # R_1 = 1 + (3-1) = 3
    # R_2 = 3 + (3-1) = 5
    # R_3 = 5 + (3-1) = 7
    # R_4 = 7 + (3-1) = 9
    
    # Answer: 9×9 pixels
    

**Exercise 4: Custom CNN Implementation**

**Problem** : Implement a CNN in PyTorch with the following specifications.

  * Input: 32×32×3 (CIFAR-10 format)
  * Conv1: 32 filters, 3×3 kernel, ReLU
  * MaxPool: 2×2
  * Conv2: 64 filters, 3×3 kernel, ReLU
  * MaxPool: 2×2
  * Fully connected: 10 class classification

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Solution Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32×32 → 16×16
            x = self.pool(F.relu(self.conv2(x)))  # 16×16 → 8×8
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc(x)
            return x
    

**Exercise 5: Comparison of Fully Connected Layers and CNNs**

**Problem** : For a 224×224×3 image input, compare parameter counts for the following two approaches.

  * Approach 1: Fully connected layer (input → 1000 units)
  * Approach 2: 3 Conv layers (3ch→64ch, 3×3 kernel)

**Solution** :
    
    
    # Approach 1: Fully connected layer
    fc_params = (224 * 224 * 3 * 1000) + 1000 = 150,529,000
    
    # Approach 2: CNN (3 layers)
    conv1_params = (3 * 64 * 3 * 3) + 64 = 1,792
    conv2_params = (64 * 64 * 3 * 3) + 64 = 36,928
    conv3_params = (64 * 64 * 3 * 3) + 64 = 36,928
    cnn_total = 1,792 + 36,928 + 36,928 = 75,648
    
    # Reduction rate
    reduction = (1 - 75,648/150,529,000) * 100 = 99.95%
    
    # CNNs require only 0.05% of the parameters of fully connected layers!
    

* * *
