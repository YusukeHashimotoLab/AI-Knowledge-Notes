---
title: "Chapter 1: Basic Concepts of Deep Learning"
chapter_title: "Chapter 1: Basic Concepts of Deep Learning"
subtitle: Understanding Definition, History, and Basic Structure of Neural Networks
reading_time: 30-35 minutes
difficulty: Beginner
code_examples: 8
exercises: 5
---

In this chapter, we will learn about the definition and history of deep learning, the basic structure of neural networks, and activation functions. We will trace the development from perceptrons to modern deep learning and verify theory with implementation code. 

## Learning Objectives

  * Understand the definition of deep learning and its relationship with machine learning
  * Grasp the historical development of deep learning
  * Understand the basic structure of neural networks (layers, neurons, weights)
  * Learn the characteristics of major activation functions (Sigmoid, ReLU, Softmax)
  * Be able to implement simple neural networks in NumPy and PyTorch

## 1\. Definition of Deep Learning

### 1.1 Relationship with Machine Learning

**Deep Learning** is a branch of **Machine Learning** that uses multi-layered neural networks to automatically learn features from data.
    
    
    ```mermaid
    graph TB
        A[Artificial Intelligence AI] --> B[Machine Learning ML]
        B --> C[Deep Learning DL]
    
        A --> A1[Rule-based Systemsif-then rules]
        B --> B1[Traditional Machine LearningDecision Tree, SVM]
        C --> C1[Deep Neural NetworksCNN, RNN, Transformer]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

Item | Traditional Machine Learning | Deep Learning  
---|---|---  
**Feature Extraction** | Manually designed by humans | Automatically learned  
**Model Structure** | Shallow (1-2 layers) | Deep (3+ layers to hundreds)  
**Data Amount** | Effective with small to medium data | Shows true potential with large data  
**Computing Resources** | CPU sufficient | GPU recommended (parallel computing)  
**Accuracy** | Task dependent | High accuracy in images, audio, text  
  
### 1.2 What Does "Deep" Mean

"Deep" refers to the **number of layers** in neural networks:

  * **Shallow Network** : 1-2 hidden layers
  * **Deep Network** : 3 or more hidden layers
  * **Very Deep Network** : Dozens to hundreds of layers (ResNet-152, GPT-3, etc.)

**Important Concept** : As networks become deeper, **hierarchical feature learning** becomes possible.

  * **Lower Layers (initial layers)** : Simple features (edges, colors, textures)
  * **Middle Layers** : Complex patterns (parts like eyes, nose, ears)
  * **Higher Layers (deep layers)** : Abstract concepts (entire face, entire object)

### 1.3 Representation Learning and Feature Extraction

The essence of deep learning is **representation learning** :

  * **Traditional Approach** : Engineers manually design features (SIFT, HOG, etc.)
  * **Deep Learning** : Networks automatically learn optimal features from data

For example, in image classification tasks:

  * Traditional: Engineers design features like "edge detection", "color histogram"
  * DL: Networks automatically discover useful features from training data

## 2\. History of Deep Learning

### 2.1 Period 1: Birth (1943-1969)

#### Perceptron (1958)

The first neural network with a learning algorithm, invented by Frank Rosenblatt.

  * **Single-layer structure** : Input and output layers only
  * **Linearly separable** : Can only solve problems separable by a straight line
  * **Limitation** : Cannot solve XOR problem (Minsky and Papert's criticism, 1969)

### 2.2 Period 2: AI Winter (1969-1986)

Limitations of perceptrons became clear, reducing funding for AI research. However, important theoretical developments occurred:

  * **Multi-Layer Perceptron (MLP)** : Introduced hidden layers to handle non-linear problems
  * **Problem** : Effective learning algorithm not yet discovered

### 2.3 Period 3: Revival (1986-2006)

#### Backpropagation (1986)

Rumelhart, Hinton, and Williams proposed an efficient learning method for multi-layer neural networks.

  * **Chain Rule** for gradient computation
  * **Multi-layer networks** became trainable
  * **Limitation** : Vanishing gradient problem in deep networks

### 2.4 Period 4: Dawn of Deep Learning (2006-2012)

Geoffrey Hinton and others proposed Deep Belief Networks (DBN):

  * **Pre-training** : Initialize weights with unsupervised learning
  * **Fine-tuning** : Adjust with supervised learning
  * **Achievement** : Training deep networks became practical

### 2.5 Period 5: Deep Learning Revolution (2012-Present)

#### ImageNet 2012: AlexNet

AlexNet by Alex Krizhevsky et al. dominated image recognition competition:

  * **5-layer Convolutional Neural Network (CNN)**
  * **ReLU activation function** adoption
  * **GPU parallel computing** utilization
  * **Dropout** for regularization
  * **Achievement** : Reduced error rate from 26% to 15%

#### Major Subsequent Milestones

  * **2014** : Proposal of GAN (Generative Adversarial Networks)
  * **2015** : ResNet (152 layers) achieves human-level accuracy on ImageNet
  * **2017** : Proposal of Transformer (attention mechanism)
  * **2018** : BERT (revolution in natural language processing)
  * **2020** : GPT-3 (large language model with 175 billion parameters)
  * **2022** : ChatGPT, Stable Diffusion (practical generative AI)

    
    
    ```mermaid
    timeline
        title History of Deep Learning
        1958 : Perceptron
        1969 : AI Winter Begins
        1986 : Backpropagation
        2006 : Dawn of Deep Learning
        2012 : AlexNet Revolution
        2017 : Transformer
        2022 : ChatGPT
    ```

## 3\. Basic Structure of Neural Networks

### 3.1 Types of Layers

Neural networks consist of multiple **layers** :

  * **Input Layer** : Layer that receives data
  * **Hidden Layer(s)** : Layer(s) that transform and process data (1 or more)
  * **Output Layer** : Layer that outputs final prediction results

    
    
    ```mermaid
    graph LR
        I1((Input 1)) --> H1((Hidden 1-1))
        I2((Input 2)) --> H1
        I3((Input 3)) --> H1
    
        I1 --> H2((Hidden 1-2))
        I2 --> H2
        I3 --> H2
    
        I1 --> H3((Hidden 1-3))
        I2 --> H3
        I3 --> H3
    
        H1 --> H4((Hidden 2-1))
        H2 --> H4
        H3 --> H4
    
        H1 --> H5((Hidden 2-2))
        H2 --> H5
        H3 --> H5
    
        H4 --> O1((Output 1))
        H5 --> O1
        H4 --> O2((Output 2))
        H5 --> O2
    
        style I1 fill:#e3f2fd
        style I2 fill:#e3f2fd
        style I3 fill:#e3f2fd
        style H1 fill:#fff3e0
        style H2 fill:#fff3e0
        style H3 fill:#fff3e0
        style H4 fill:#f3e5f5
        style H5 fill:#f3e5f5
        style O1 fill:#e8f5e9
        style O2 fill:#e8f5e9
    ```

### 3.2 Neurons and Activation

**Neurons** in each layer perform the following processing:

  1. **Weighted sum calculation** : input × weight + bias
  2. **Apply activation function** : non-linear transformation

Expressed as equations:

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = f(z)$$

Where:

  * $\mathbf{x}$: input vector
  * $\mathbf{w}$: weight vector
  * $b$: bias (intercept)
  * $z$: linear combination (weighted sum)
  * $f$: activation function
  * $a$: activation output

### 3.3 Weights and Biases

**Weights** and **biases** are parameters that neural networks learn:

  * **Weight $w$** : Coefficient representing the importance of each input (optimized by learning)
  * **Bias $b$** : Constant term that adjusts the threshold of activation function

**Intuitive Understanding** : Analogy to linear regression $y = wx + b$

  * $w$: slope (magnitude of influence input has on output)
  * $b$: intercept (shift of output baseline)

## 4\. Activation Functions

An **activation function** is a function that introduces **non-linearity** into neural networks. Without activation functions, no matter how deep the layers, it would only be a combination of linear transformations, unable to learn complex patterns.

### 4.1 Sigmoid Function

One of the most classical activation functions, compresses output to range from 0 to 1.

**Equation** :

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Characteristics** :

  * Output range: $(0, 1)$
  * Differentiable and smooth
  * Can be interpreted as probability (used in output layer for binary classification)

**Problems** :

  * **Vanishing gradient problem** : When $x$ is large or small, gradient becomes nearly 0
  * **Output not centered** : Always positive values (reduces learning efficiency)
  * **Computational cost** : Expensive exponential function calculation

**NumPy Implementation** :
    
    
    import numpy as np
    
    def sigmoid(x):
        """
        Sigmoid activation function
    
        Parameters:
        -----------
        x : array-like
            Input
    
        Returns:
        --------
        array-like
            Output in range 0 to 1
        """
        return 1 / (1 + np.exp(-x))
    
    # Usage example
    x = np.array([-2, -1, 0, 1, 2])
    print("Input:", x)
    print("Sigmoid output:", sigmoid(x))
    # Example output: [0.119, 0.269, 0.5, 0.731, 0.881]
    

### 4.2 ReLU (Rectified Linear Unit)

The most widely used activation function in modern deep learning.

**Equation** :

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\\ 0 & \text{if } x \leq 0 \end{cases}$$

**Characteristics** :

  * Very fast computation (max operation only)
  * Mitigates vanishing gradient problem (gradient is always 1 when $x > 0$)
  * Sparse activation (about 50% of neurons become 0)

**Problems** :

  * **Dying ReLU problem** : Gradient becomes 0 for negative inputs, causing neurons to "die"

**NumPy Implementation** :
    
    
    def relu(x):
        """
        ReLU activation function
    
        Parameters:
        -----------
        x : array-like
            Input
    
        Returns:
        --------
        array-like
            Negative values become 0, positive values unchanged
        """
        return np.maximum(0, x)
    
    # Usage example
    x = np.array([-2, -1, 0, 1, 2])
    print("Input:", x)
    print("ReLU output:", relu(x))
    # Example output: [0, 0, 0, 1, 2]
    

### 4.3 Softmax Function

Activation function used in output layer for multi-class classification, converts output to probability distribution.

**Equation** :

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Characteristics** :

  * Sum of outputs equals 1 (probability distribution)
  * All outputs in range 0 to 1
  * Largest input value has largest probability

**NumPy Implementation** :
    
    
    def softmax(x):
        """
        Softmax activation function
    
        Parameters:
        -----------
        x : array-like
            Input (logits)
    
        Returns:
        --------
        array-like
            Probability distribution (sum = 1)
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    # Usage example
    x = np.array([2.0, 1.0, 0.1])
    print("Input:", x)
    print("Softmax output:", softmax(x))
    print("Sum:", softmax(x).sum())
    # Example output: [0.659, 0.242, 0.099]
    # Sum: 1.0
    

### 4.4 tanh (Hyperbolic Tangent)

Improved version of Sigmoid, compresses output to range from -1 to 1.

**Equation** :

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**Characteristics** :

  * Output range: $(-1, 1)$
  * Output is centered (better learning efficiency than Sigmoid)
  * Often used in RNN (Recurrent Neural Networks)

**NumPy Implementation** :
    
    
    def tanh(x):
        """
        tanh activation function
    
        Parameters:
        -----------
        x : array-like
            Input
    
        Returns:
        --------
        array-like
            Output in range -1 to 1
        """
        return np.tanh(x)
    
    # Usage example
    x = np.array([-2, -1, 0, 1, 2])
    print("Input:", x)
    print("tanh output:", tanh(x))
    # Example output: [-0.964, -0.762, 0.0, 0.762, 0.964]
    

### 4.5 Comparison of Activation Functions

Function | Output Range | Computation Speed | Vanishing Gradient | Main Use  
---|---|---|---|---  
**Sigmoid** | (0, 1) | Slow | Yes | Binary classification output layer  
**tanh** | (-1, 1) | Slow | Yes | RNN hidden layers  
**ReLU** | [0, ∞) | Fast | No (in positive region) | CNN hidden layers (most common)  
**Softmax** | (0, 1), sum=1 | Medium | N/A | Multi-class classification output layer  
  
## 5\. Implementing Simple Neural Networks

### 5.1 NumPy Implementation

First, we'll implement from basics using NumPy. This allows us to understand the internal workings of neural networks.
    
    
    import numpy as np
    
    class SimpleNN:
        """
        Simple 3-layer neural network
    
        Structure: Input layer → Hidden layer → Output layer
        """
    
        def __init__(self, input_size, hidden_size, output_size):
            """
            Parameters:
            -----------
            input_size : int
                Number of neurons in input layer
            hidden_size : int
                Number of neurons in hidden layer
            output_size : int
                Number of neurons in output layer
            """
            # Initialize weights (small random values)
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
    
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))
    
        def forward(self, X):
            """
            Forward Propagation
    
            Parameters:
            -----------
            X : array-like, shape (n_samples, input_size)
                Input data
    
            Returns:
            --------
            array-like, shape (n_samples, output_size)
                Output (probability distribution)
            """
            # Layer 1: Input → Hidden layer
            self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
            self.a1 = relu(self.z1)                 # ReLU activation
    
            # Layer 2: Hidden layer → Output layer
            self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation
            self.a2 = softmax(self.z2)                     # Softmax activation
    
            return self.a2
    
    # Usage example
    # Input: 4 dimensions, Hidden layer: 10 neurons, Output: 3 classes
    model = SimpleNN(input_size=4, hidden_size=10, output_size=3)
    
    # Dummy data (5 samples, 4 dimensions)
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 2.9, 5.6, 1.8],
        [5.8, 2.7, 5.1, 1.9],
        [5.0, 3.4, 1.5, 0.2],
        [6.7, 3.1, 5.6, 2.4]
    ])
    
    # Prediction
    predictions = model.forward(X)
    print("Prediction probabilities:\n", predictions)
    print("\nPredicted classes:", np.argmax(predictions, axis=1))
    

### 5.2 PyTorch Implementation

Next, we'll implement the same network in PyTorch. PyTorch has built-in automatic differentiation and GPU support, allowing more concise code.
    
    
    import torch
    import torch.nn as nn
    
    class SimpleNNPyTorch(nn.Module):
        """
        3-layer neural network in PyTorch
        """
    
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNNPyTorch, self).__init__()
    
            # Define layers
            self.fc1 = nn.Linear(input_size, hidden_size)  # Input → Hidden layer
            self.relu = nn.ReLU()                          # ReLU activation
            self.fc2 = nn.Linear(hidden_size, output_size) # Hidden layer → Output
            self.softmax = nn.Softmax(dim=1)               # Softmax activation
    
        def forward(self, x):
            """
            Forward propagation
    
            Parameters:
            -----------
            x : torch.Tensor, shape (n_samples, input_size)
                Input data
    
            Returns:
            --------
            torch.Tensor, shape (n_samples, output_size)
                Output (probability distribution)
            """
            x = self.fc1(x)      # Linear transformation
            x = self.relu(x)     # ReLU activation
            x = self.fc2(x)      # Linear transformation
            x = self.softmax(x)  # Softmax activation
            return x
    
    # Usage example
    model_pytorch = SimpleNNPyTorch(input_size=4, hidden_size=10, output_size=3)
    
    # Convert dummy data to Tensor
    X_torch = torch.tensor(X, dtype=torch.float32)
    
    # Prediction
    with torch.no_grad():  # Disable gradient calculation (for inference)
        predictions_pytorch = model_pytorch(X_torch)
    
    print("PyTorch prediction probabilities:\n", predictions_pytorch.numpy())
    print("\nPredicted classes:", torch.argmax(predictions_pytorch, dim=1).numpy())
    

### 5.3 NumPy Implementation vs PyTorch Implementation

Item | NumPy Implementation | PyTorch Implementation  
---|---|---  
**Code Amount** | More (manual implementation) | Less (high-level API)  
**Learning Purpose** | Ideal for understanding internals | Ideal for practical development  
**Automatic Differentiation** | Manual implementation required | Automatic computation  
**GPU Support** | Difficult | Easy (just .cuda())  
**Performance** | Slow | Fast (optimized)  
  
## Exercises

Exercise 1: Implementing and Visualizing Activation Functions

**Problem** : Implement Sigmoid, ReLU, and tanh activation functions, and draw graphs in the range x = -5 to 5. Compare the characteristics of each function.

**Hints** :

  * Use matplotlib.pyplot
  * Generate equally spaced points with np.linspace(-5, 5, 100)
  * Plot each function with plt.plot()

Exercise 2: Softmax Temperature Parameter

**Problem** : Implement the following formula with temperature parameter T introduced to the Softmax function:

$$\text{softmax}(\mathbf{x}, T)_i = \frac{e^{x_i/T}}{\sum_{j=1}^{n} e^{x_j/T}}$$

For input x = [2.0, 1.0, 0.1], compare outputs for T = 0.5, 1.0, 2.0. How does the output change as temperature increases?

Exercise 3: XOR Problem

**Problem** : Design a neural network to solve the XOR problem (exclusive OR).

  * Input: 2-dimensional (x1, x2), each element is 0 or 1
  * Output: x1 XOR x2 (1 if x1 and x2 differ, 0 if same)
  * Network structure: Input layer(2) → Hidden layer(2) → Output layer(1)

Modify the SimpleNN class to handle the XOR problem (learning will be covered in the next chapter).

Exercise 4: Weight Initialization

**Problem** : Implement weight initialization in the SimpleNN class using the following 3 methods and compare the weight distributions after initialization:

  1. **All zeros** : W = np.zeros(...)
  2. **Normal distribution** : W = np.random.randn(...) * 0.01
  3. **Xavier initialization** : W = np.random.randn(...) * np.sqrt(1/n_in)

Calculate the mean and standard deviation of weights for each initialization, and consider which method is most appropriate.

Exercise 5: Multi-layer Network

**Problem** : Extend the SimpleNN class to have 2 hidden layers:

  * Input layer → Hidden layer 1 (10 neurons, ReLU) → Hidden layer 2 (5 neurons, ReLU) → Output layer (3 classes, Softmax)

Verify the output shape of each layer for 4-dimensional input data.

## Summary

In this chapter, we learned the basic concepts of deep learning:

  * **Definition** : Deep learning is representation learning using multi-layered neural networks
  * **History** : Development trajectory of about 70 years from perceptrons to the modern era
  * **Structure** : Composed of input layer, hidden layers, and output layer, each layer is a collection of neurons
  * **Activation Functions** : Introduce non-linearity with Sigmoid, ReLU, Softmax, tanh, etc.
  * **Implementation** : Understand principles with NumPy, implement efficiently with PyTorch

**Next Chapter Preview** : In Chapter 2, we will learn about detailed forward propagation calculations, the role of weight matrices, and loss functions, and deeply understand how neural networks make predictions.
