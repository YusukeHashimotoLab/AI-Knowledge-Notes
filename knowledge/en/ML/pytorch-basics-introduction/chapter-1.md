---
title: "Chapter 1: PyTorch Fundamentals"
chapter_title: "Chapter 1: PyTorch Fundamentals"
subtitle: PyTorch as a Deep Learning Framework
reading_time: 25-30 min
difficulty: Beginner
code_examples: 10
exercises: 5
---

In this chapter, you'll understand what PyTorch is and why it's widely used in deep learning. You'll learn about comparisons with TensorFlow, installation methods, Tensor fundamentals, and automatic differentiation mechanisms. Through practical code examples, you'll become familiar with the PyTorch programming style.

## Learning Objectives

  * ✅ Explain PyTorch's features and differences from TensorFlow
  * ✅ Install PyTorch and verify it works
  * ✅ Create and manipulate basic Tensors
  * ✅ Understand the concept of dynamic computation graphs
  * ✅ Practice basic usage of automatic differentiation (autograd)

## 1\. What is PyTorch

**PyTorch** is an open-source deep learning framework developed by Facebook's (now Meta) AI Research Lab (FAIR). Since its release in 2016, it has been widely used by researchers and practitioners.

### Main Features of PyTorch

  * **Dynamic Computation Graph** : Computation graphs are built at runtime, enabling flexible Python-like coding
  * **Pythonic API** : Intuitive API design similar to NumPy
  * **Powerful automatic differentiation** : Automatic gradient computation via autograd functionality
  * **GPU support** : High-speed parallel computation using CUDA
  * **Rich ecosystem** : torchvision (images), torchtext (text), torchaudio (audio), etc.

### PyTorch vs TensorFlow

Feature | PyTorch | TensorFlow  
---|---|---  
Computation Graph | Dynamic (Define-by-Run) | Static (Define-and-Run) *Dynamic available in TF2.0+  
Learning Curve | Intuitive, Python-like | Somewhat complex (especially TF1.x)  
Debugging | Possible with standard Python debugger | Somewhat difficult (TF1.x), improved in TF2.0  
Research Adoption | Very high (mainstream for paper implementations) | Moderate  
Production | Deployable with TorchServe | Strong with TensorFlow Serving  
Mobile Support | PyTorch Mobile | TensorFlow Lite (mature)  
  
**💡 Which Should You Choose?**

PyTorch is popular for research and prototyping. On the other hand, TensorFlow is a strong choice for large-scale production environments or when existing TensorFlow infrastructure is available. Learning both allows you to use the optimal tool for each situation.

## 2\. Installation and Environment Setup

### Method 1: Installation via pip (Recommended)
    
    
    # CPU version
    pip install torch torchvision torchaudio
    
    # GPU version (for CUDA 11.8)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # GPU version (for CUDA 12.1)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    

### Method 2: Installation via conda
    
    
    # CPU version
    conda install pytorch torchvision torchaudio -c pytorch
    
    # GPU version (for CUDA 11.8)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    

### Method 3: Google Colab (No environment setup required)

Using Google Colab, you can use PyTorch with just a browser. GPU is also available for free.
    
    
    import torch
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    

### Installation Verification

Let's verify that PyTorch is correctly installed with the following code:
    
    
    import torch
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA (GPU) is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # CUDA version
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Simple Tensor creation
    x = torch.tensor([1, 2, 3])
    print(f"Sample tensor: {x}")
    

**Example output:**
    
    
    PyTorch version: 2.1.0
    CUDA available: True
    CUDA version: 11.8
    GPU device: Tesla T4
    Sample tensor: tensor([1, 2, 3])
    

## 3\. Tensor Fundamentals

**Tensor** is the central data structure in PyTorch. Similar to NumPy's ndarray, but supports GPU computation and automatic differentiation.

### Tensor Creation
    
    
    import torch
    
    # Create Tensor from list
    x = torch.tensor([1, 2, 3, 4, 5])
    print(x)
    # Output: tensor([1, 2, 3, 4, 5])
    
    # 2D Tensor (matrix)
    matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(matrix)
    # Output: tensor([[1, 2],
    #               [3, 4],
    #               [5, 6]])
    
    # Specify data type
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(float_tensor)
    # Output: tensor([1., 2., 3.])
    
    # Special Tensors
    zeros = torch.zeros(3, 4)  # 3x4 matrix of zeros
    ones = torch.ones(2, 3)    # 2x3 matrix of ones
    random = torch.rand(2, 2)  # 2x2 random matrix (uniform distribution 0-1)
    randn = torch.randn(2, 2)  # 2x2 random matrix (standard normal distribution)
    
    print("Zeros:\n", zeros)
    print("Ones:\n", ones)
    print("Random:\n", random)
    print("Randn:\n", randn)
    

### Tensor Attributes
    
    
    x = torch.randn(3, 4)
    
    print(f"Shape: {x.shape}")           # Shape: torch.Size([3, 4])
    print(f"Size: {x.size()}")           # Shape (method version)
    print(f"Data type: {x.dtype}")       # Data type: torch.float32
    print(f"Device: {x.device}")         # Device: cpu
    print(f"Requires grad: {x.requires_grad}")  # Gradient computation: False
    print(f"Number of dimensions: {x.ndim}")    # Number of dimensions: 2
    print(f"Number of elements: {x.numel()}")   # Number of elements: 12
    

### Interconversion with NumPy
    
    
    import numpy as np
    import torch
    
    # NumPy array → PyTorch Tensor
    numpy_array = np.array([1, 2, 3, 4, 5])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print(f"From NumPy: {tensor_from_numpy}")
    
    # PyTorch Tensor → NumPy array
    tensor = torch.tensor([10, 20, 30])
    numpy_from_tensor = tensor.numpy()
    print(f"To NumPy: {numpy_from_tensor}")
    
    # Important: Memory is shared, so modifying one changes the other
    numpy_array[0] = 100
    print(f"Changed tensor: {tensor_from_numpy}")  # First element becomes 100
    

**⚠️ Note**

`from_numpy()` and `numpy()` share memory. If you need a copy of the data, use `tensor.clone()` or `numpy_array.copy()`.

## 4\. Basic Tensor Operations

### Arithmetic Operations
    
    
    import torch
    
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    
    # Element-wise operations
    print(x + y)      # tensor([5, 7, 9])
    print(x - y)      # tensor([-3, -3, -3])
    print(x * y)      # tensor([4, 10, 18])
    print(x / y)      # tensor([0.25, 0.4, 0.5])
    
    # Operator vs method versions
    print(torch.add(x, y))      # Same as x + y
    print(torch.mul(x, y))      # Same as x * y
    
    # In-place operations (modify original Tensor)
    x.add_(y)  # Same as x = x + y (with underscore)
    print(x)   # tensor([5, 7, 9])
    

### Matrix Operations
    
    
    import torch
    
    # Matrix multiplication
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    
    # @ operator or torch.matmul()
    C = A @ B
    print(C)
    # Output: tensor([[19, 22],
    #               [43, 50]])
    
    # Transpose
    print(A.T)
    # Output: tensor([[1, 3],
    #               [2, 4]])
    
    # Vector dot product
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5, 6])
    dot_product = torch.dot(v1, v2)
    print(f"Dot product: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
    

### Shape Operations
    
    
    import torch
    
    x = torch.randn(2, 3, 4)  # 2x3x4 Tensor
    print(f"Original shape: {x.shape}")
    
    # reshape: Convert to new shape (copies if not contiguous)
    y = x.reshape(2, 12)
    print(f"Reshaped: {y.shape}")  # torch.Size([2, 12])
    
    # view: Convert to new shape (must be contiguous)
    z = x.view(-1)  # Flatten to 1D (-1 means auto-calculate)
    print(f"Flattened: {z.shape}")  # torch.Size([24])
    
    # unsqueeze: Add dimension
    a = torch.tensor([1, 2, 3])
    b = a.unsqueeze(0)  # Add at 0th dimension
    print(f"Original: {a.shape}, After unsqueeze(0): {b.shape}")
    # Original: torch.Size([3]), After unsqueeze(0): torch.Size([1, 3])
    
    # squeeze: Remove dimension (remove dimensions with size 1)
    c = torch.randn(1, 3, 1, 4)
    d = c.squeeze()
    print(f"Before squeeze: {c.shape}, After squeeze: {d.shape}")
    # Before squeeze: torch.Size([1, 3, 1, 4]), After squeeze: torch.Size([3, 4])
    

## 5\. Automatic Differentiation (Autograd) Fundamentals

PyTorch's **autograd** feature provides automatic differentiation, which is key to deep learning. This allows automatic calculation of gradients for complex computations.

### requires_grad and backward()
    
    
    import torch
    
    # Enable gradient tracking with requires_grad=True
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x: {x}")
    print(f"x.requires_grad: {x.requires_grad}")
    
    # Perform computation
    y = x ** 2 + 3 * x + 1
    print(f"y: {y}")
    
    # Gradient computation (dy/dx)
    y.backward()  # Calculate gradient
    print(f"x.grad: {x.grad}")  # dy/dx = 2x + 3 = 2*2 + 3 = 7
    

**Mathematical background:**

$$y = x^2 + 3x + 1$$

$$\frac{dy}{dx} = 2x + 3$$

$$x = 2 \text{ when }, \frac{dy}{dx} = 2(2) + 3 = 7$$

### Complex Computation Graphs
    
    
    import torch
    
    # Build computation graph with multiple variables
    a = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([4.0], requires_grad=True)
    
    # Complex computation
    c = a * b           # c = 3 * 4 = 12
    d = c ** 2          # d = 12^2 = 144
    e = torch.sin(d)    # e = sin(144)
    f = e + c           # f = sin(144) + 12
    
    print(f"f: {f}")
    
    # Gradient computation
    f.backward()
    
    print(f"df/da: {a.grad}")
    print(f"df/db: {b.grad}")
    
    
    
    ```mermaid
    graph LR
      A[a=3] --> C[c=a*b]
      B[b=4] --> C
      C --> D[d=c^2]
      D --> E[e=sin d]
      E --> F[f=e+c]
      C --> F
    ```

### Gradient Accumulation and Initialization
    
    
    import torch
    
    x = torch.tensor([1.0], requires_grad=True)
    
    # First computation
    y = x ** 2
    y.backward()
    print(f"1st gradient: {x.grad}")  # 2.0
    
    # Second computation (gradients accumulate!)
    y = x ** 3
    y.backward()
    print(f"2nd gradient (accumulated): {x.grad}")  # 2.0 + 3.0 = 5.0
    
    # Reset gradient to zero
    x.grad.zero_()
    y = x ** 3
    y.backward()
    print(f"3rd gradient (after reset): {x.grad}")  # 3.0
    

**⚠️ Important**

PyTorch **accumulates** gradients by default. In training loops, you need to call `optimizer.zero_grad()` or `tensor.grad.zero_()` to reset gradients at each iteration.

## 6\. Using GPUs

In PyTorch, you can easily move Tensors to GPUs for high-speed computation.
    
    
    import torch
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move Tensor to GPU
    x = torch.randn(3, 3)
    x = x.to(device)
    print(f"x device: {x.device}")
    
    # Computation on GPU
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(f"z device: {z.device}")
    
    # Move back to CPU
    z_cpu = z.to("cpu")
    print(f"z_cpu device: {z_cpu.device}")
    
    # Or use .cpu() method
    z_cpu = z.cpu()
    

**💡 Best Practice**

It's important to place models and data on the same device. Operations between different devices will result in errors.

## 7\. Your First Complete PyTorch Program

Let's integrate what we've learned so far and write a simple program to learn a linear function.
    
    
    import torch
    
    # Data generation: y = 3x + 2 with noise
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5
    
    # Parameter initialization (to be learned)
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    # Learning rate
    learning_rate = 0.01
    
    # Training loop
    for epoch in range(100):
        # Prediction
        y_pred = w * X + b
    
        # Loss computation (mean squared error)
        loss = ((y_pred - y_true) ** 2).mean()
    
        # Gradient computation
        loss.backward()
    
        # Parameter update (gradient descent)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
    
        # Reset gradients
        w.grad.zero_()
        b.grad.zero_()
    
        # Display results every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    print(f"\nTraining complete!")
    print(f"Final parameters: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"True values: w = 3.0, b = 2.0")
    

**Example output:**
    
    
    Epoch 10: Loss = 0.3245, w = 2.7234, b = 1.8456
    Epoch 20: Loss = 0.2456, w = 2.8567, b = 1.9123
    Epoch 30: Loss = 0.2123, w = 2.9123, b = 1.9567
    ...
    Epoch 100: Loss = 0.1234, w = 2.9876, b = 1.9934
    
    Training complete!
    Final parameters: w = 2.9876, b = 1.9934
    True values: w = 3.0, b = 2.0
    

## Exercises

**Exercise 1: Tensor Creation and Manipulation**

Implement the following operations:

  1. Create a 5x5 random Tensor (standard normal distribution)
  2. Calculate the mean, standard deviation, maximum, and minimum of that Tensor
  3. Flatten the Tensor to 1D

    
    
    # Write your code here
    

**Exercise 2: NumPy Conversion**

Convert NumPy array `np.array([[1, 2], [3, 4], [5, 6]])` to a PyTorch Tensor, multiply each element by 2, and convert back to a NumPy array.

**Exercise 3: Automatic Differentiation**

For function $f(x, y) = x^2 + y^2 + 2xy$, calculate partial derivatives $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ at point $(x=1, y=2)$ using autograd.

Hint: Mathematically $\frac{\partial f}{\partial x} = 2x + 2y$, $\frac{\partial f}{\partial y} = 2y + 2x$

**Exercise 4: GPU Transfer**

Create a 3x3 random Tensor, transfer it to GPU (if available), calculate the square of that Tensor, and move it back to CPU.

**Exercise 5: Simple Optimization Problem**

Find the value of x that minimizes function $f(x) = (x - 5)^2$ using gradient descent. Initial value is $x=0$, learning rate is 0.1, execute for 100 steps.

Expected answer: $x \approx 5$

## Summary

In this chapter, you learned PyTorch fundamentals:

  * ✅ PyTorch is a flexible deep learning framework adopting dynamic computation graphs
  * ✅ Compared to TensorFlow, it enables intuitive Python-like coding
  * ✅ Tensor creation, manipulation, and interconversion with NumPy
  * ✅ Gradient computation using automatic differentiation (autograd)
  * ✅ High-speed computation using GPUs

**🎉 Next Steps**

In the next chapter, you'll learn more advanced Tensor operations (indexing, slicing, broadcasting) and acquire skills needed for real data processing.

* * *

**Reference Resources**

  * [PyTorch Official Documentation](<https://pytorch.org/docs/stable/index.html>)
  * [PyTorch Official Tutorials](<https://pytorch.org/tutorials/>)
  * [PyTorch GitHub](<https://github.com/pytorch/pytorch>)
