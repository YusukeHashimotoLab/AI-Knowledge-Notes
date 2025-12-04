---
title: "Chapter 4: Interpreting Deep Learning Models"
chapter_title: "Chapter 4: Interpreting Deep Learning Models"
subtitle: Visualization Techniques for CNNs and Transformers - Saliency Maps, Grad-CAM, Attention Visualization
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
version: 1.0
created_at: 2025-10-21
---

This chapter covers Interpreting Deep Learning Models. You will learn Visualize CNN attention regions using Grad-CAM, Calculate attributions with Integrated Gradients, and Visualize Transformer attention mechanisms.

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Understand gradient-based visualization methods (Saliency Maps, Gradient × Input, SmoothGrad)
  * ✅ Visualize CNN attention regions using Grad-CAM
  * ✅ Calculate attributions with Integrated Gradients
  * ✅ Visualize Transformer attention mechanisms
  * ✅ Interpret deep learning models using PyTorch and Captum
  * ✅ Debug image classification and text classification models

* * *

## 4.1 Saliency Maps and Gradient-Based Methods

### Overview

**Saliency Maps** are techniques for visualizing the importance of each pixel in the input with respect to the neural network's prediction.

> "Calculate from gradients which parts of the input image most influence the model's prediction"

### Classification of Gradient-Based Methods
    
    
    ```mermaid
    graph TD
        A[Gradient-Based Visualization] --> B[Vanilla Gradients]
        A --> C[Gradient × Input]
        A --> D[SmoothGrad]
        A --> E[Integrated Gradients]
    
        B --> B1[Simplest∂y/∂x]
        C --> C1[Product of gradient and inputClearer]
        D --> D1[Add noise and averageDenoising]
        E --> E1[Path integralTheoretical guarantees]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### Vanilla Gradients

Compute the gradient of the output $y_c$ for class $c$ with respect to input $x$.

$$ S_c(x) = \frac{\partial y_c}{\partial x} $$

#### Implementation with PyTorch
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Implementation with PyTorch
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    def vanilla_gradients(image_path, model, target_class=None):
        """
        Generate saliency map with Vanilla Gradients
    
        Args:
            image_path: Image file path
            model: PyTorch model
            target_class: Target class (if None, use highest probability class)
    
        Returns:
            saliency: Saliency map
            pred_class: Predicted class
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    
        # Forward pass
        output = model(img_tensor)
    
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
    
        # Compute gradients
        model.zero_grad()
        output[0, target_class].backward()
    
        # Generate saliency map (max absolute value of gradients)
        saliency = img_tensor.grad.data.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # Usage example
    saliency, pred = vanilla_gradients('cat.jpg', model)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f'Vanilla Gradients (Class: {pred})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### Gradient × Input

Take element-wise product of gradients and input values for more interpretable visualization.

$$ S_c(x) = x \odot \frac{\partial y_c}{\partial x} $$
    
    
    def gradient_input(image_path, model, target_class=None):
        """
        Generate saliency map with Gradient × Input
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    
        # Forward pass
        output = model(img_tensor)
    
        if target_class is None:
            target_class = output.argmax(dim=1).item()
    
        # Compute gradients
        model.zero_grad()
        output[0, target_class].backward()
    
        # Gradient × Input
        saliency = (img_tensor.grad.data * img_tensor.data).abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # Comparison
    saliency_vanilla, _ = vanilla_gradients('cat.jpg', model)
    saliency_gi, pred = gradient_input('cat.jpg', model)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_vanilla, cmap='hot')
    axes[1].set_title('Vanilla Gradients')
    axes[1].axis('off')
    
    axes[2].imshow(saliency_gi, cmap='hot')
    axes[2].set_title('Gradient × Input')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### SmoothGrad

Remove noise by averaging gradients over multiple samples with added noise.

$$ \hat{S}_c(x) = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial y_c}{\partial (x + \mathcal{N}(0, \sigma^2))} $$
    
    
    def smooth_grad(image_path, model, target_class=None,
                    n_samples=50, noise_level=0.15):
        """
        Generate saliency map with SmoothGrad
    
        Args:
            n_samples: Number of noise samples
            noise_level: Standard deviation of noise
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    
        # Determine target class
        with torch.no_grad():
            output = model(img_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
        # Compute gradients for noisy samples
        gradients = []
        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(img_tensor) * noise_level
            noisy_img = img_tensor + noise
            noisy_img.requires_grad = True
    
            # Compute gradients
            output = model(noisy_img)
            model.zero_grad()
            output[0, target_class].backward()
    
            gradients.append(noisy_img.grad.data)
    
        # Average
        avg_gradient = torch.stack(gradients).mean(dim=0)
        saliency = avg_gradient.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # Usage example
    saliency_smooth, pred = smooth_grad('cat.jpg', model, n_samples=50)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(Image.open('cat.jpg'))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_vanilla, cmap='hot')
    axes[1].set_title('Vanilla Gradients')
    axes[1].axis('off')
    
    axes[2].imshow(saliency_smooth, cmap='hot')
    axes[2].set_title('SmoothGrad')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 4.2 Grad-CAM

### Overview

**Grad-CAM (Gradient-weighted Class Activation Mapping)** uses the final convolutional layer of a CNN to visualize class-specific attention regions.

> "Identify regions important for class discrimination by weighting feature maps of the convolutional layer with gradients"

### Algorithm
    
    
    ```mermaid
    graph LR
        A[Input Image] --> B[CNN]
        B --> C[Final Convolutional LayerFeature Maps A^k]
        C --> D[Global Average Pooling]
        D --> E[Fully Connected Layer]
        E --> F[Class Score y^c]
    
        F --> G[Gradient Computation∂y^c/∂A^k]
        G --> H[Global Average Poolingα_k^c]
        H --> I[Weighted SumL = ReLU Σ α_k^c A^k]
        I --> J[Grad-CAM]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style J fill:#e8f5e9
    ```

1\. Obtain feature maps $A^k$ from the final convolutional layer

2\. Compute gradients of $A^k$ with respect to class $c$ score $y^c$

$$ \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} $$

3\. Apply weighted sum and ReLU

$$ L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right) $$

### Implementation Example
    
    
    class GradCAM:
        """
        Grad-CAM implementation
        """
        def __init__(self, model, target_layer):
            """
            Args:
                model: PyTorch model
                target_layer: Layer to visualize (final convolutional layer)
            """
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
    
            # Register hooks
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_backward_hook(self.save_gradient)
    
        def save_activation(self, module, input, output):
            """Save activations during forward pass"""
            self.activations = output.detach()
    
        def save_gradient(self, module, grad_input, grad_output):
            """Save gradients during backward pass"""
            self.gradients = grad_output[0].detach()
    
        def generate_cam(self, image_tensor, target_class=None):
            """
            Generate Grad-CAM
    
            Args:
                image_tensor: Input image tensor
                target_class: Target class
    
            Returns:
                cam: Grad-CAM
                pred_class: Predicted class
            """
            # Forward pass
            output = self.model(image_tensor)
    
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
    
            # Compute weights (global average pooling)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
    
            # Weighted sum
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
    
            # ReLU
            cam = torch.relu(cam)
    
            # Normalize
            cam = cam - cam.min()
            cam = cam / cam.max()
    
            # Resize
            cam = torch.nn.functional.interpolate(
                cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False
            )
    
            return cam.squeeze().cpu().numpy(), target_class
    
    # Use Grad-CAM with ResNet50
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # Specify final convolutional layer
    target_layer = model.layer4[-1].conv3
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    # Load image
    img = Image.open('cat.jpg').convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True
    
    # Generate Grad-CAM
    cam, pred_class = gradcam.generate_cam(img_tensor)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'Grad-CAM (Class: {pred_class})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### Grad-CAM++

An improved version of Grad-CAM that provides more accurate visualization for multiple objects or small objects.

$$ \alpha_k^c = \sum_i \sum_j \left( \frac{\partial^2 y^c}{(\partial A_{ij}^k)^2} \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{ij}^k}\right) \right) $$
    
    
    class GradCAMPlusPlus(GradCAM):
        """
        Grad-CAM++ implementation
        """
        def generate_cam(self, image_tensor, target_class=None):
            """Generate Grad-CAM++"""
            # Forward pass
            output = self.model(image_tensor)
    
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
            # Compute first and second order gradients
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
    
            grad_1 = self.gradients.clone()
    
            # Second order gradient
            self.model.zero_grad()
            grad_1.backward(torch.ones_like(grad_1), retain_graph=True)
            grad_2 = self.gradients.clone()
    
            # Third order gradient
            self.model.zero_grad()
            grad_2.backward(torch.ones_like(grad_2))
            grad_3 = self.gradients.clone()
    
            # Compute weights
            alpha_num = grad_2
            alpha_denom = 2.0 * grad_2 + (grad_3 * self.activations).sum(dim=(2, 3), keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    
            alpha = alpha_num / alpha_denom
            weights = (alpha * torch.relu(grad_1)).sum(dim=(2, 3), keepdim=True)
    
            # Compute CAM
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
    
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
    
            # Resize
            cam = torch.nn.functional.interpolate(
                cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False
            )
    
            return cam.squeeze().cpu().numpy(), target_class
    

* * *

## 4.3 Integrated Gradients

### Overview

**Integrated Gradients** computes the contribution of each feature by integrating gradients along the path from a baseline (e.g., black image) to the input image.

> "Path integration provides theoretical guarantees that the sum of attributions equals the difference in model output"

### Formula

Given a path $\gamma(\alpha) = x' + \alpha \cdot (x - x')$ from baseline $x'$ to input $x$:

$$ \text{IntegratedGrad}_i(x) = (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial F(\gamma(\alpha))}{\partial x_i} d\alpha $$

In implementation, the integral is approximated with Riemann sum:

$$ \text{IntegratedGrad}_i(x) \approx (x_i - x'_i) \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i} \cdot \frac{1}{m} $$

### Implementation with Captum Library
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation with Captum Library
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from captum.attr import IntegratedGradients, visualization as viz
    import torch.nn.functional as F
    
    # Prepare model and data
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # Load image
    img = Image.open('cat.jpg').convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Create Integrated Gradients instance
    ig = IntegratedGradients(model)
    
    # Get target class
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Set baseline (black image)
    baseline = torch.zeros_like(img_tensor)
    
    # Compute Integrated Gradients
    attributions = ig.attribute(img_tensor, baseline, target=pred_class, n_steps=50)
    
    # Visualization
    def visualize_attributions(img, attributions, pred_class):
        """
        Visualize Integrated Gradients
        """
        # Convert tensor to numpy array
        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    
        # Process attributions
        attr = attributions.squeeze().cpu().permute(1, 2, 0).numpy()
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        # Attributions (heatmap)
        attr_sum = np.abs(attr).sum(axis=2)
        im = axes[1].imshow(attr_sum, cmap='hot')
        axes[1].set_title(f'Integrated Gradients (Class: {pred_class})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
        # Overlay
        axes[2].imshow(img_np)
        axes[2].imshow(attr_sum, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    visualize_attributions(img, attributions, pred_class)
    

### Effect of Different Baselines
    
    
    # Comparison with different baselines
    baselines = {
        'Black': torch.zeros_like(img_tensor),
        'White': torch.ones_like(img_tensor),
        'Random': torch.randn_like(img_tensor),
        'Blur': None  # Gaussian blur image
    }
    
    # Gaussian blur baseline
    from torchvision.transforms import GaussianBlur
    blur_transform = GaussianBlur(kernel_size=51, sigma=50)
    img_blur = blur_transform(img)
    baselines['Blur'] = transform(img_blur).unsqueeze(0).to(device)
    
    # Compute for each baseline
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, (name, baseline) in enumerate(baselines.items(), 1):
        # Compute Integrated Gradients
        attr = ig.attribute(img_tensor, baseline, target=pred_class, n_steps=50)
    
        # Visualization
        attr_sum = attr.squeeze().cpu().abs().sum(dim=0).numpy()
        im = axes[idx].imshow(attr_sum, cmap='hot')
        axes[idx].set_title(f'Baseline: {name}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    # Hide last empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### Method Comparison

Method | Computational Cost | Theoretical Guarantee | Noise | Use Case  
---|---|---|---|---  
Vanilla Gradients | Low | None | High | Quick analysis  
SmoothGrad | Medium | None | Low | Denoising  
Grad-CAM | Low | None | Low | CNN visualization  
Integrated Gradients | High | Yes | Low | Precise attribution  
  
* * *

## 4.4 Attention Visualization

### Overview

**Attention mechanisms** are at the core of Transformer models, learning relationships between different parts of the input. By visualizing attention weights, we can understand what the model is focusing on.

### Self-Attention Formula

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

  * $Q$: Query
  * $K$: Key
  * $V$: Value
  * $d_k$: Dimension of keys

### BERT Attention Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    from transformers import BertTokenizer, BertModel
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    
    def visualize_attention(text, layer=0, head=0):
        """
        Visualize BERT attention weights
    
        Args:
            text: Input text
            layer: Layer to visualize (0-11)
            head: Head to visualize (0-11)
        """
        # Tokenization
        inputs = tokenizer(text, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Get attention weights
        # attentions: (layers, batch, heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0, head].cpu().numpy()
    
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                    cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
        ax.set_title(f'BERT Attention (Layer {layer}, Head {head})')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # Usage example
    text = "The cat sat on the mat"
    visualize_attention(text, layer=0, head=0)
    

### Multi-Head Attention Visualization
    
    
    def visualize_multi_head_attention(text, layer=0):
        """
        Visualize multiple attention heads simultaneously
    
        Args:
            text: Input text
            layer: Layer to visualize
        """
        # Tokenization
        inputs = tokenizer(text, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Visualize 12 heads
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
    
        for head in range(12):
            attention = outputs.attentions[layer][0, head].cpu().numpy()
    
            sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                       cmap='viridis', ax=axes[head], cbar=False)
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('')
            axes[head].set_ylabel('')
    
            if head % 4 != 0:
                axes[head].set_yticklabels([])
            if head < 8:
                axes[head].set_xticklabels([])
            else:
                axes[head].set_xticklabels(tokens, rotation=45, ha='right')
    
        plt.suptitle(f'BERT Multi-Head Attention (Layer {layer})', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Usage example
    visualize_multi_head_attention("The quick brown fox jumps over the lazy dog", layer=5)
    

### Interactive Visualization with BertViz
    
    
    # Requirements:
    # - Python 3.9+
    # - transformers>=4.30.0
    
    """
    Example: Interactive Visualization with BertViz
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    # Install BertViz: pip install bertviz
    
    from bertviz import head_view, model_view
    from transformers import AutoTokenizer, AutoModel
    
    # Load model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    # Text
    text = "The cat sat on the mat because it was tired"
    
    # Tokenization
    inputs = tokenizer(text, return_tensors='pt')
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Head View (attention for each head)
    head_view(outputs.attentions, tokens)
    
    # Model View (attention for all layers)
    model_view(outputs.attentions, tokens)
    

### Vision Transformer Attention Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - pillow>=10.0.0
    # - requests>=2.31.0
    # - transformers>=4.30.0
    
    from transformers import ViTModel, ViTFeatureExtractor
    from PIL import Image
    import requests
    
    # Load Vision Transformer
    model_name = 'google/vit-base-patch16-224'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    vit_model = ViTModel.from_pretrained(model_name, output_attentions=True)
    vit_model.eval()
    
    def visualize_vit_attention(image_path, layer=-1, head=0):
        """
        Visualize Vision Transformer attention
    
        Args:
            image_path: Image path
            layer: Layer index (-1 for last layer)
            head: Head index
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors='pt')
    
        # Forward pass
        with torch.no_grad():
            outputs = vit_model(**inputs)
    
        # Get attention weights
        attention = outputs.attentions[layer][0, head].cpu().numpy()
    
        # Get CLS token attention (first token)
        cls_attention = attention[0, 1:]  # From CLS token to image patches
    
        # Reshape to 14x14 grid (for ViT-Base-Patch16-224)
        num_patches = int(cls_attention.shape[0] ** 0.5)
        cls_attention = cls_attention.reshape(num_patches, num_patches)
    
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        # Attention map
        im = axes[1].imshow(cls_attention, cmap='hot')
        axes[1].set_title(f'CLS Attention (Layer {layer}, Head {head})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
        # Overlay
        from scipy.ndimage import zoom
        attention_resized = zoom(cls_attention, 224/num_patches, order=1)
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    # Usage example
    visualize_vit_attention('cat.jpg', layer=-1, head=0)
    

* * *

## 4.5 End-to-End Practical Examples

### Image Classification Model Interpretation

Combine multiple visualization techniques in real-world use cases.
    
    
    class ImageClassifierInterpreter:
        """
        Comprehensive interpretation tool for image classification models
        """
        def __init__(self, model, device='cuda'):
            self.model = model.to(device)
            self.device = device
            self.model.eval()
    
            # Prepare Grad-CAM
            if hasattr(model, 'layer4'):  # ResNet family
                target_layer = model.layer4[-1].conv3
            else:
                target_layer = list(model.children())[-2]
    
            self.gradcam = GradCAM(model, target_layer)
    
            # Prepare Integrated Gradients
            self.ig = IntegratedGradients(model)
    
        def interpret(self, image_path, methods=['gradcam', 'ig', 'smoothgrad']):
            """
            Interpret model using multiple methods
    
            Args:
                image_path: Image path
                methods: List of methods to use
    
            Returns:
                results: Dictionary of interpretation results
            """
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
    
            # Prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)
                top5_probs, top5_idx = probs.topk(5)
    
            results = {
                'image': img,
                'predictions': {
                    'classes': top5_idx[0].cpu().numpy(),
                    'probabilities': top5_probs[0].cpu().numpy()
                }
            }
    
            # Grad-CAM
            if 'gradcam' in methods:
                cam, _ = self.gradcam.generate_cam(img_tensor, target_class=top5_idx[0, 0].item())
                results['gradcam'] = cam
    
            # Integrated Gradients
            if 'ig' in methods:
                baseline = torch.zeros_like(img_tensor)
                attr = self.ig.attribute(img_tensor, baseline, target=top5_idx[0, 0].item())
                results['integrated_gradients'] = attr.squeeze().cpu().abs().sum(dim=0).numpy()
    
            # SmoothGrad
            if 'smoothgrad' in methods:
                saliency, _ = smooth_grad(image_path, self.model, target_class=top5_idx[0, 0].item())
                results['smoothgrad'] = saliency
    
            return results
    
        def visualize(self, results):
            """Visualize interpretation results"""
            n_methods = len([k for k in results.keys() if k not in ['image', 'predictions']])
    
            fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
    
            # Original image and predictions
            axes[0].imshow(results['image'])
            pred_text = f"Top predictions:\n"
            for idx, (cls, prob) in enumerate(zip(results['predictions']['classes'][:3],
                                                  results['predictions']['probabilities'][:3])):
                pred_text += f"{idx+1}. Class {cls}: {prob:.2%}\n"
            axes[0].set_title(pred_text, fontsize=10)
            axes[0].axis('off')
    
            # Results for each method
            idx = 1
            if 'gradcam' in results:
                axes[idx].imshow(results['image'])
                axes[idx].imshow(results['gradcam'], cmap='jet', alpha=0.5)
                axes[idx].set_title('Grad-CAM')
                axes[idx].axis('off')
                idx += 1
    
            if 'integrated_gradients' in results:
                im = axes[idx].imshow(results['integrated_gradients'], cmap='hot')
                axes[idx].set_title('Integrated Gradients')
                axes[idx].axis('off')
                plt.colorbar(im, ax=axes[idx], fraction=0.046)
                idx += 1
    
            if 'smoothgrad' in results:
                axes[idx].imshow(results['smoothgrad'], cmap='hot')
                axes[idx].set_title('SmoothGrad')
                axes[idx].axis('off')
                idx += 1
    
            plt.tight_layout()
            plt.show()
    
    # Usage example
    model = models.resnet50(pretrained=True)
    interpreter = ImageClassifierInterpreter(model)
    
    # Run interpretation
    results = interpreter.interpret('cat.jpg', methods=['gradcam', 'ig', 'smoothgrad'])
    interpreter.visualize(results)
    

### Text Classification Model Interpretation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - transformers>=4.30.0
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from captum.attr import LayerIntegratedGradients
    
    class TextClassifierInterpreter:
        """
        Text classification model interpretation tool
        """
        def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
    
            # Prepare Layer Integrated Gradients
            self.lig = LayerIntegratedGradients(self.forward_func,
                                               self.model.distilbert.embeddings)
    
        def forward_func(self, inputs):
            """Model forward function"""
            return self.model(inputs_embeds=inputs).logits
    
        def interpret(self, text, target_class=None):
            """
            Interpret text
    
            Args:
                text: Input text
                target_class: Target class (None for predicted class)
    
            Returns:
                attributions: Importance of each token
                tokens: Token list
                prediction: Prediction result
            """
            # Tokenization
            inputs = self.tokenizer(text, return_tensors='pt')
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
            # Prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_class = outputs.logits.argmax(dim=1).item()
                pred_prob = probs[0, pred_class].item()
    
            if target_class is None:
                target_class = pred_class
    
            # Compute Integrated Gradients
            input_embeds = self.model.distilbert.embeddings(inputs['input_ids'])
            baseline = torch.zeros_like(input_embeds)
    
            attributions = self.lig.attribute(
                input_embeds,
                baseline,
                target=target_class,
                n_steps=50
            )
    
            # Aggregate attributions per token
            attributions_sum = attributions.sum(dim=-1).squeeze(0)
            attributions_sum = attributions_sum / torch.norm(attributions_sum)
            attributions_sum = attributions_sum.cpu().detach().numpy()
    
            return {
                'tokens': tokens,
                'attributions': attributions_sum,
                'prediction': {
                    'class': pred_class,
                    'probability': pred_prob,
                    'label': self.model.config.id2label[pred_class]
                }
            }
    
        def visualize(self, text, target_class=None):
            """Visualize interpretation results"""
            results = self.interpret(text, target_class)
    
            tokens = results['tokens']
            attributions = results['attributions']
    
            # Normalize for visualization
            attr_min, attr_max = attributions.min(), attributions.max()
            attributions_norm = (attributions - attr_min) / (attr_max - attr_min + 1e-8)
    
            # Color map
            import matplotlib.cm as cm
            colors = cm.RdYlGn(attributions_norm)
    
            # Text display
            fig, ax = plt.subplots(figsize=(15, 3))
            ax.axis('off')
    
            # Prediction result
            pred_text = f"Prediction: {results['prediction']['label']} ({results['prediction']['probability']:.2%})"
            ax.text(0.5, 0.9, pred_text, ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=ax.transAxes)
    
            # Tokens and importance
            x_pos = 0.05
            for token, attr, color in zip(tokens, attributions_norm, colors):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
    
                # Background color
                bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none')
                ax.text(x_pos, 0.5, token, ha='left', va='center', fontsize=12,
                       bbox=bbox_props, transform=ax.transAxes)
    
                # Importance score
                ax.text(x_pos, 0.2, f'{attr:.3f}', ha='left', va='center', fontsize=8,
                       transform=ax.transAxes)
    
                x_pos += len(token) * 0.015 + 0.02
    
                if x_pos > 0.95:
                    break
    
            # Color bar
            sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn, norm=plt.Normalize(vmin=attr_min, vmax=attr_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
            cbar.set_label('Attribution Score', fontsize=10)
    
            plt.tight_layout()
            plt.show()
    
    # Usage example
    text_interpreter = TextClassifierInterpreter()
    
    # Positive sentiment
    text_interpreter.visualize("This movie is absolutely fantastic and amazing!")
    
    # Negative sentiment
    text_interpreter.visualize("This is the worst film I have ever seen.")
    

### Practical Model Debugging
    
    
    def debug_model_prediction(model, image_path, true_label, expected_label):
        """
        Debug misclassification
    
        Args:
            model: Classification model
            image_path: Image path
            true_label: Ground truth label
            expected_label: Expected label
        """
        interpreter = ImageClassifierInterpreter(model)
    
        # Interpretation
        results = interpreter.interpret(image_path, methods=['gradcam', 'ig'])
    
        pred_class = results['predictions']['classes'][0]
        pred_prob = results['predictions']['probabilities'][0]
    
        print(f"=== Model Debug Report ===")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_class} (Probability: {pred_prob:.2%})")
        print(f"Expected Label: {expected_label}")
    
        if pred_class != expected_label:
            print(f"\n❌ Misclassification detected")
            print(f"\nTop-5 Predictions:")
            for idx, (cls, prob) in enumerate(zip(results['predictions']['classes'],
                                                  results['predictions']['probabilities'])):
                marker = "✓" if cls == true_label else " "
                print(f"  {marker} {idx+1}. Class {cls}: {prob:.2%}")
    
            # Visualization
            interpreter.visualize(results)
    
            # Interpretation
            print("\n=== Interpretation ===")
            print("Check Grad-CAM:")
            print("- Which regions of the image is the model focusing on?")
            print("- Are the attention regions appropriate for the true label?")
            print("- Is the model focusing on background or noise?")
        else:
            print(f"\n✓ Correctly classified")
            interpreter.visualize(results)
    
    # Usage example
    model = models.resnet50(pretrained=True)
    debug_model_prediction(model, 'dog.jpg', true_label=254, expected_label=254)
    

* * *

## 4.6 Chapter Summary

### What We Learned

  1. **Gradient-Based Methods**

     * Vanilla Gradients: Simple gradient visualization
     * Gradient × Input: Clearer visualization
     * SmoothGrad: Denoising
  2. **Grad-CAM**

     * Visualize CNN attention regions
     * Utilize final convolutional layer
     * Improvements with Grad-CAM++
  3. **Integrated Gradients**

     * Attribution computation via path integration
     * Method with theoretical guarantees
     * Importance of baseline selection
  4. **Attention Visualization**

     * Transformer Self-Attention
     * Multi-Head Attention interpretation
     * Interactive visualization with BertViz
  5. **Practical Applications**

     * Image classification interpretation
     * Text classification interpretation
     * Model debugging techniques

### Next Chapter

In Chapter 5, we will learn about **Practical Applications of Model Interpretation** :

  * Model auditing and bias detection
  * Regulatory compliance (GDPR, AI Act)
  * Explanation to stakeholders
  * Continuous monitoring

* * *

## Exercises

### Problem 1 (Difficulty: Easy)

List three main differences between Vanilla Gradients and Grad-CAM.

Sample Answer

**Answer** :

  1. **Information Used** : Vanilla Gradients uses only gradients with respect to input, while Grad-CAM uses feature maps and gradients from the final convolutional layer
  2. **Resolution** : Vanilla Gradients has the same resolution as the input image, while Grad-CAM interpolates from lower resolution
  3. **Noise** : Vanilla Gradients has more noise, while Grad-CAM is smoother and more interpretable

### Problem 2 (Difficulty: Medium)

Implement SmoothGrad and investigate how the number of noise samples (n_samples) affects the results.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implement SmoothGrad and investigate how the number of noise
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Compare different sample numbers
    n_samples_list = [10, 25, 50, 100]
    
    fig, axes = plt.subplots(1, len(n_samples_list) + 1, figsize=(20, 4))
    
    # Original image
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, n_samples in enumerate(n_samples_list, 1):
        saliency, _ = smooth_grad('cat.jpg', model, n_samples=n_samples)
    
        axes[idx].imshow(saliency, cmap='hot')
        axes[idx].set_title(f'n_samples={n_samples}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    

**Observations** :

  * Small n_samples (10): Noise remains
  * Moderate n_samples (50): Smooth and clear
  * Large n_samples (100): Higher computational cost but smoother
  * **Recommendation** : Around 50 provides a good balance in practice

### Problem 3 (Difficulty: Medium)

Explain how the results change when using different baselines (black, white, blur) with Integrated Gradients.

Sample Answer

**Answer** :

Baseline | Characteristics | Application  
---|---|---  
Black Image | All pixels are 0  
Most common | Regular image classification  
White Image | All pixels are 1  
Effective for black backgrounds | Medical images, etc.  
Blur Image | Preserves structure, loses detail  
More realistic | When texture is important  
Random Noise | Chaotic image  
For reference | Comparative verification  
  
**Baseline Selection Guidelines** :

  * Black image is generally recommended
  * Select based on domain knowledge
  * Compare results with multiple baselines
  * Verify consistency of results

### Problem 4 (Difficulty: Hard)

Visualize BERT attention weights and analyze which words each word attends to in the sentence "The cat sat on the mat".

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    # - torch>=2.0.0, <2.3.0
    # - transformers>=4.30.0
    
    """
    Example: Visualize BERT attention weights and analyze which words eac
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from transformers import BertTokenizer, BertModel
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    
    text = "The cat sat on the mat"
    
    # Tokenization
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Average attention across all layers
    all_attentions = torch.stack([att.squeeze(0) for att in outputs.attentions])
    avg_attention = all_attentions.mean(dim=[0, 1]).cpu().numpy()  # Average over layers and heads
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', ax=ax, cbar_kws={'label': 'Average Attention'})
    ax.set_title('BERT Average Attention Across All Layers and Heads')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("=== Attention Analysis ===\n")
    
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
    
        # Word most attended to by each token (excluding itself)
        attention_weights = avg_attention[i].copy()
        attention_weights[i] = -1  # Exclude itself
        max_idx = np.argmax(attention_weights)
    
        print(f"'{token}' attends most to: '{tokens[max_idx]}' (weight: {attention_weights[max_idx]:.3f})")
    
    print("\nObservations:")
    print("- 'cat' attends to 'sat' (subject-verb relationship)")
    print("- 'sat' attends to 'cat' and 'on' (verb attends to subject and preposition)")
    print("- 'mat' attends to 'the' and 'on' (noun attends to article and preposition)")
    

**Expected Observations** :

  * **Syntactic Relationships** : Subject-verb, modifier-modified relationships show mutual attention
  * **Locality** : Strong attention to neighboring tokens
  * **Grammatical Patterns** : Prepositions attend to following nouns

### Problem 5 (Difficulty: Hard)

Write code to apply Grad-CAM and Integrated Gradients to a misclassified image and analyze the cause of misclassification.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    import torch
    import torchvision.models as models
    from captum.attr import IntegratedGradients
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    class MisclassificationAnalyzer:
        """
        Misclassification analysis tool
        """
        def __init__(self, model, device='cuda'):
            self.model = model.to(device)
            self.device = device
            self.model.eval()
    
            # Prepare Grad-CAM
            self.gradcam = GradCAM(model, model.layer4[-1].conv3)
    
            # Prepare Integrated Gradients
            self.ig = IntegratedGradients(model)
    
        def analyze_misclassification(self, image_path, true_label, imagenet_labels):
            """
            Detailed analysis of misclassification
    
            Args:
                image_path: Image path
                true_label: Ground truth label
                imagenet_labels: ImageNet label dictionary
            """
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
    
            # Prediction
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
    
            print(f"{'='*60}")
            print(f"Misclassification Analysis Report")
            print(f"{'='*60}")
            print(f"\nTrue: {imagenet_labels[true_label]}")
            print(f"Predicted: {imagenet_labels[pred_label]} (Probability: {probs[0, pred_label]:.2%})")
    
            # Top-5 predictions
            top5_probs, top5_idx = probs.topk(5)
            print(f"\nTop-5 Predictions:")
            for i, (idx, prob) in enumerate(zip(top5_idx[0], top5_probs[0])):
                marker = "✓" if idx == true_label else " "
                print(f"  {marker} {i+1}. {imagenet_labels[idx.item()]}: {prob:.2%}")
    
            # True class score
            true_prob = probs[0, true_label]
            true_rank = (probs[0] > true_prob).sum().item() + 1
            print(f"\nTrue Class Rank: {true_rank} (Probability: {true_prob:.2%})")
    
            # Grad-CAM (predicted class)
            cam_pred, _ = self.gradcam.generate_cam(img_tensor, target_class=pred_label)
    
            # Grad-CAM (true class)
            cam_true, _ = self.gradcam.generate_cam(img_tensor, target_class=true_label)
    
            # Integrated Gradients
            baseline = torch.zeros_like(img_tensor)
            attr_pred = self.ig.attribute(img_tensor, baseline, target=pred_label, n_steps=50)
            attr_true = self.ig.attribute(img_tensor, baseline, target=true_label, n_steps=50)
    
            # Visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
            # Original image
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
    
            # Grad-CAM for predicted class
            axes[0, 1].imshow(img)
            axes[0, 1].imshow(cam_pred, cmap='jet', alpha=0.5)
            axes[0, 1].set_title(f'Grad-CAM: Predicted\n{imagenet_labels[pred_label]}')
            axes[0, 1].axis('off')
    
            # Grad-CAM for true class
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(cam_true, cmap='jet', alpha=0.5)
            axes[0, 2].set_title(f'Grad-CAM: True\n{imagenet_labels[true_label]}')
            axes[0, 2].axis('off')
    
            # Spacer
            axes[1, 0].axis('off')
    
            # IG for predicted class
            attr_pred_sum = attr_pred.squeeze().cpu().abs().sum(dim=0).numpy()
            axes[1, 1].imshow(attr_pred_sum, cmap='hot')
            axes[1, 1].set_title(f'IG: Predicted\n{imagenet_labels[pred_label]}')
            axes[1, 1].axis('off')
    
            # IG for true class
            attr_true_sum = attr_true.squeeze().cpu().abs().sum(dim=0).numpy()
            axes[1, 2].imshow(attr_true_sum, cmap='hot')
            axes[1, 2].set_title(f'IG: True\n{imagenet_labels[true_label]}')
            axes[1, 2].axis('off')
    
            plt.tight_layout()
            plt.show()
    
            # Diagnosis
            print(f"\n{'='*60}")
            print(f"Diagnosis")
            print(f"{'='*60}")
            print("1. Compare Grad-CAM:")
            print("   - Do attention regions differ between predicted and true class?")
            print("   - Is the predicted class focusing on background or noise?")
            print("\n2. Check Integrated Gradients:")
            print("   - Do features necessary for the true class exist in the image?")
            print("   - Are incorrect features for the predicted class strongly present?")
            print("\n3. Possible Causes:")
            if true_prob < 0.01:
                print("   - Model barely considers the true class")
                print("   - Dataset may lack similar examples")
            elif true_rank <= 5:
                print("   - True class is in top ranks (boundary case)")
                print("   - High similarity between classes possible")
            else:
                print("   - Possible issues with image quality or preprocessing")
                print("   - Object may be partially occluded")
    
    # Load ImageNet labels (simplified version)
    imagenet_labels = {
        254: 'Pug',
        281: 'Tabby Cat',
        # ... other labels
    }
    
    # Usage example
    model = models.resnet50(pretrained=True)
    analyzer = MisclassificationAnalyzer(model)
    
    # Analyze misclassified image
    analyzer.analyze_misclassification('pug.jpg', true_label=254, imagenet_labels=imagenet_labels)
    

**Example Output** :
    
    
    ============================================================
    Misclassification Analysis Report
    ============================================================
    
    True: Pug
    Predicted: Tabby Cat (Probability: 45.23%)
    
    Top-5 Predictions:
       1. Tabby Cat: 45.23%
       2. Egyptian Cat: 23.45%
      ✓ 3. Pug: 12.34%
       4. Bulldog: 8.90%
       5. Chihuahua: 5.67%
    
    True Class Rank: 3 (Probability: 12.34%)
    
    ============================================================
    Diagnosis
    ============================================================
    1. Compare Grad-CAM:
       - Do attention regions differ between predicted and true class?
       - Is the predicted class focusing on background or noise?
    
    2. Check Integrated Gradients:
       - Do features necessary for the true class exist in the image?
       - Are incorrect features for the predicted class strongly present?
    
    3. Possible Causes:
       - True class is in top ranks (boundary case)
       - High similarity between classes possible
    

* * *

## References

  1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). _Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps_. arXiv:1312.6034.
  2. Selvaraju, R. R., et al. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_. ICCV 2017.
  3. Sundararajan, M., Taly, A., & Yan, Q. (2017). _Axiomatic Attribution for Deep Networks_. ICML 2017.
  4. Smilkov, D., et al. (2017). _SmoothGrad: removing noise by adding noise_. arXiv:1706.03825.
  5. Chattopadhay, A., et al. (2018). _Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks_. WACV 2018.
  6. Vaswani, A., et al. (2017). _Attention is All You Need_. NeurIPS 2017.
  7. Vig, J. (2019). _A Multiscale Visualization of Attention in the Transformer Model_. ACL 2019.
  8. Natekar, P., & Sharma, M. (2020). _Captum: A unified and generic model interpretability library for PyTorch_. arXiv:2009.07896.
