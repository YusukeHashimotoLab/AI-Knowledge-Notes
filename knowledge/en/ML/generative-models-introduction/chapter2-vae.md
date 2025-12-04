---
title: "Chapter 2: VAE (Variational Autoencoder)"
chapter_title: "Chapter 2: VAE (Variational Autoencoder)"
subtitle: Complete Understanding of Probabilistic Latent Variable Models and ELBO
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
---

This chapter covers VAE (Variational Autoencoder). You will learn limitations of standard Autoencoders, theory of ELBO (Evidence Lower Bound), and VAE's Encoder (Recognition network).

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the limitations of standard Autoencoders and the motivation for VAE
  * ✅ Explain the theory of ELBO (Evidence Lower Bound) and KL divergence
  * ✅ Understand the mechanism and necessity of the Reparameterization Trick
  * ✅ Implement VAE's Encoder (Recognition network) and Decoder (Generative network)
  * ✅ Build MNIST/CelebA image generation systems in PyTorch
  * ✅ Visualize latent space and perform interpolation

* * *

## 2.1 Review and Limitations of Autoencoders

### Structure of Standard Autoencoders

An **Autoencoder (AE)** is an unsupervised learning model that compresses input data into low-dimensional latent representations and then reconstructs them.
    
    
    ```mermaid
    graph LR
        X["Input x(784-dim)"] --> E["Encoderq(z|x)"]
        E --> Z["Latent variable z(2-20 dim)"]
        Z --> D["Decoderp(x|z)"]
        D --> X2["Reconstruction x̂(784-dim)"]
    
        style E fill:#b3e5fc
        style Z fill:#fff9c4
        style D fill:#ffab91
    ```

### Objective Function of Autoencoders

Standard Autoencoders minimize the **reconstruction error** :

$$ \mathcal{L}_{\text{AE}} = \|x - \hat{x}\|^2 = \|x - \text{Decoder}(\text{Encoder}(x))\|^2 $$ 

### Basic Autoencoder Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Autoencoder(nn.Module):
        """Standard Autoencoder (deterministic)"""
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super(Autoencoder, self).__init__()
    
            # Encoder
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)
    
            # Decoder
            self.fc3 = nn.Linear(latent_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """Encode input to latent representation"""
            h = F.relu(self.fc1(x))
            z = self.fc2(h)
            return z
    
        def decode(self, z):
            """Reconstruct input from latent representation"""
            h = F.relu(self.fc3(z))
            x_recon = torch.sigmoid(self.fc4(h))
            return x_recon
    
        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z
    
    
    # Test the model
    print("=== Standard Autoencoder Operation ===")
    ae = Autoencoder(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # Dummy data (28x28 MNIST images)
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    x_recon, z = ae(x)
    
    print(f"Input: {x.shape}")
    print(f"Latent variable: {z.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    
    # Reconstruction error
    recon_loss = F.mse_loss(x_recon, x)
    print(f"Reconstruction error: {recon_loss.item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in ae.parameters())
    print(f"Total parameters: {total_params:,}")
    

### Limitations of Autoencoders

Issue | Description | Impact  
---|---|---  
**Deterministic** | Same input always produces same latent variable | Cannot generate diverse samples  
**Unstructured latent space** | No clear probability distribution in latent space | Difficult to perform random sampling  
**Prone to overfitting** | Tends to memorize training data | Weak at generating novel data  
**Interpolation quality** | Latent space interpolation may not be meaningful | Cannot perform smooth transformations  
  
> "Standard Autoencoders can compress and reconstruct, but are insufficient as generative models for creating new data"

### Visualizing Latent Space Problems
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Visualizing Latent Space Problems
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Problems with standard Autoencoder latent space
    print("\n=== Latent Space Problems in Autoencoders ===")
    
    # Generate multiple random inputs
    num_samples = 100
    x_samples = torch.randn(num_samples, 784)
    
    # Get latent representations with Encoder
    ae.eval()
    with torch.no_grad():
        z_samples = ae.encode(x_samples)
    
    print(f"Number of latent variable samples: {z_samples.shape[0]}")
    print(f"Latent dimensions: {z_samples.shape[1]}")
    
    # Latent space statistics
    z_mean = z_samples.mean(dim=0)
    z_std = z_samples.std(dim=0)
    
    print(f"\nLatent variable means (partial): {z_mean[:5].numpy()}")
    print(f"Latent variable standard deviations (partial): {z_std[:5].numpy()}")
    print("→ Means and variances are scattered and unstructured")
    
    # Attempt reconstruction from random latent space sampling
    z_random = torch.randn(10, 20)  # Random latent variables
    with torch.no_grad():
        x_from_random = ae.decode(z_random)
    
    print(f"\nReconstruction from random sampling: {x_from_random.shape}")
    print("→ Unlikely to generate meaningful images")
    print("→ This is why VAE is needed!")
    

* * *

## 2.2 Motivation and Theory of VAE

### Basic Idea of VAE

**Variational Autoencoder (VAE)** introduces a **probabilistic framework** to Autoencoders, achieving the following:

  1. **Structured latent space** : Regularize latent variables to follow a normal distribution $\mathcal{N}(0, I)$
  2. **Probabilistic generation** : Can generate new data through sampling from latent space
  3. **Smooth interpolation** : Interpolating in latent space yields meaningful intermediate data

    
    
    ```mermaid
    graph TB
        X["Input x"] --> E["Encoderq_φ(z|x)"]
        E --> Mu["Mean μ"]
        E --> Logvar["Log variance log σ²"]
        Mu --> Sample["Samplingz ~ N(μ, σ²)"]
        Logvar --> Sample
        Sample --> Z["Latent variable z"]
        Z --> D["Decoderp_θ(x|z)"]
        D --> X2["Reconstruction x̂"]
    
        Prior["Priorp(z) = N(0,I)"] -.->|KL regularization| Sample
    
        style E fill:#b3e5fc
        style Sample fill:#fff59d
        style D fill:#ffab91
        style Prior fill:#c5e1a5
    ```

### Probabilistic Formulation

VAE assumes the following probabilistic model:

  * **Generative process** : $$ \begin{align} z &\sim p(z) = \mathcal{N}(0, I) \quad \text{(Prior)} \\\ x &\sim p_\theta(x|z) \quad \text{(Likelihood)} \end{align} $$ 
  * **Inference process** : $$ q_\phi(z|x) \approx p(z|x) \quad \text{(Variational approximation)} $$ 

### ELBO (Evidence Lower Bound)

The goal of VAE is to maximize the log-likelihood $\log p_\theta(x)$, but this is computationally intractable. Instead, we maximize the **ELBO** :

$$ \begin{align} \log p_\theta(x) &\geq \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \| p(z)) \\\ &= \text{ELBO}(\theta, \phi; x) \end{align} $$ 

Where:

  * **First term** : Reconstruction term - Decoder's reconstruction performance
  * **Second term** : KL regularization term - Brings Encoder's distribution closer to the prior

### Analytical Solution of KL Divergence

When both Encoder and Decoder use Gaussian distributions, the KL divergence can be computed analytically:

$$ D_{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{J}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right) $$ 

Where $J$ is the latent dimension, and $\mu_j$ and $\sigma_j^2$ are the mean and variance output by the Encoder.

### ELBO Derivation and Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    
    def kl_divergence_gaussian(mu, logvar):
        """
        Calculate KL divergence between Gaussian distributions
    
        Args:
            mu: (batch, latent_dim) - Mean output by Encoder
            logvar: (batch, latent_dim) - Log variance output by Encoder
    
        Returns:
            kl_loss: (batch,) - KL divergence for each sample
        """
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl
    
    
    def vae_loss(x_recon, x, mu, logvar):
        """
        VAE loss function (ELBO)
    
        Args:
            x_recon: (batch, input_dim) - Reconstructed input
            x: (batch, input_dim) - Original input
            mu: (batch, latent_dim) - Mean
            logvar: (batch, latent_dim) - Log variance
    
        Returns:
            total_loss: Scalar - ELBO (negative value)
            recon_loss: Scalar - Reconstruction error
            kl_loss: Scalar - KL divergence
        """
        # Reconstruction error (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
        # KL divergence
        kl_loss = kl_divergence_gaussian(mu, logvar).sum()
    
        # ELBO = Reconstruction term - KL term (maximize = minimize negative)
        total_loss = recon_loss + kl_loss
    
        return total_loss, recon_loss, kl_loss
    
    
    # Numerical example for ELBO calculation
    print("=== Numerical Example of ELBO ===")
    
    batch_size = 32
    input_dim = 784
    latent_dim = 20
    
    # Dummy data
    x = torch.rand(batch_size, input_dim)
    x_recon = torch.rand(batch_size, input_dim)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    
    print(f"Total loss (ELBO): {total_loss.item():.2f}")
    print(f"Reconstruction error: {recon_loss.item():.2f}")
    print(f"KL divergence: {kl_loss.item():.2f}")
    print(f"\nBatch average:")
    print(f"  Reconstruction error: {recon_loss.item()/batch_size:.2f}")
    print(f"  KL divergence: {kl_loss.item()/batch_size:.2f}")
    print("\n→ Balance between the two terms is important!")
    

### Comparison of Autoencoder and VAE

Item | Autoencoder | VAE  
---|---|---  
**Latent variable** | Deterministic $z = f(x)$ | Probabilistic $z \sim q_\phi(z|x)$  
**Objective function** | Reconstruction error only | ELBO (Reconstruction + KL)  
**Latent space** | Unstructured | Regularized to normal distribution  
**Generation capability** | Weak | Strong (sampling possible)  
**Interpolation** | Unstable | Smooth  
**Training** | Simple | Requires Reparameterization Trick  
  
> "VAE adds a probabilistic framework to Autoencoders, significantly improving generative model capabilities"

* * *

## 2.3 Reparameterization Trick

### Why the Reparameterization Trick is Needed

Training VAE requires sampling from $z \sim q_\phi(z|x)$. However, **probabilistic sampling is non-differentiable** , so backpropagation cannot be performed directly.
    
    
    ```mermaid
    graph LR
        X["x"] --> E["Encoder"]
        E --> Mu["μ"]
        E --> Sigma["σ"]
        Mu --> Sample["z ~ N(μ, σ²)"]
        Sigma --> Sample
        Sample -.->|Gradient doesn't flow!| D["Decoder"]
    
        style Sample fill:#ffccbc
    ```

### Mechanism of Reparameterization Trick

Rewrite sampling as a **deterministic transformation** as follows:

$$ \begin{align} z &\sim \mathcal{N}(\mu, \sigma^2) \\\ &\Downarrow \text{(Reparameterization)} \\\ z &= \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) \end{align} $$ 

Here, $\epsilon$ is **noise** independent of the parameters. This transformation allows gradients to flow through $\mu$ and $\sigma$.
    
    
    ```mermaid
    graph LR
        X["x"] --> E["Encoder"]
        E --> Mu["μ"]
        E --> Sigma["σ"]
        Epsilon["ϵ ~ N(0,1)"] --> Reparam["z = μ + σ·ϵ"]
        Mu --> Reparam
        Sigma --> Reparam
        Reparam -->|Gradient flows!| D["Decoder"]
    
        style Reparam fill:#c5e1a5
        style Epsilon fill:#fff59d
    ```

### Implementation of Reparameterization Trick
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class VAEEncoder(nn.Module):
        """VAE Encoder (Recognition network)"""
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super(VAEEncoder, self).__init__()
    
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
        def forward(self, x):
            """
            Args:
                x: (batch, input_dim)
    
            Returns:
                mu: (batch, latent_dim) - Mean
                logvar: (batch, latent_dim) - Log variance
            """
            h = torch.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
    
    def reparameterize(mu, logvar):
        """
        Reparameterization Trick
    
        Args:
            mu: (batch, latent_dim) - Mean
            logvar: (batch, latent_dim) - Log variance
    
        Returns:
            z: (batch, latent_dim) - Sampled latent variable
        """
        # Calculate standard deviation (use log variance for numerical stability)
        std = torch.exp(0.5 * logvar)
    
        # Sample noise from standard normal distribution
        epsilon = torch.randn_like(std)
    
        # z = μ + σ·ϵ
        z = mu + std * epsilon
    
        return z
    
    
    # Test the implementation
    print("=== Reparameterization Trick Operation ===")
    
    encoder = VAEEncoder(input_dim=784, hidden_dim=400, latent_dim=20)
    x = torch.randn(32, 784)
    
    # Get mean and variance from Encoder
    mu, logvar = encoder(x)
    print(f"Mean: {mu.shape}")
    print(f"Log variance: {logvar.shape}")
    
    # Sampling with Reparameterization Trick
    z = reparameterize(mu, logvar)
    print(f"Sampled latent variable: {z.shape}")
    
    # Check gradient flow
    print("\n=== Verify Gradient Flow ===")
    z.sum().backward()
    print(f"Gradient for μ: {encoder.fc_mu.weight.grad is not None}")
    print(f"Gradient for log(σ²): {encoder.fc_logvar.weight.grad is not None}")
    print("→ Gradients flow thanks to Reparameterization Trick!")
    
    # Sample multiple times from same input (probabilistic)
    print("\n=== Verify Probabilistic Sampling ===")
    z1 = reparameterize(mu, logvar)
    z2 = reparameterize(mu, logvar)
    z3 = reparameterize(mu, logvar)
    
    print(f"Sample 1 (first 5 dims): {z1[0, :5].detach().numpy()}")
    print(f"Sample 2 (first 5 dims): {z2[0, :5].detach().numpy()}")
    print(f"Sample 3 (first 5 dims): {z3[0, :5].detach().numpy()}")
    print("→ Different samples obtained from same input (diversity)")
    

### Numerical Stability Considerations
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Numerical Stability Considerations
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Why use log variance
    print("\n=== Importance of Numerical Stability ===")
    
    # Bad example: directly using variance
    sigma_bad = torch.tensor([0.001, 1.0, 100.0])
    print(f"Variance (direct): {sigma_bad}")
    print(f"→ Extreme values exist, numerically unstable")
    
    # Good example: using log variance
    logvar_good = torch.log(sigma_bad)
    print(f"\nLog variance: {logvar_good}")
    print(f"→ Numerically stable range")
    
    # Recovery
    sigma_recovered = torch.exp(0.5 * logvar_good)
    print(f"\nStandard deviation (recovered): {sigma_recovered}")
    print(f"→ Original values are accurately recovered")
    
    # Further stabilization with clipping
    logvar_clipped = torch.clamp(logvar_good, min=-10, max=10)
    print(f"\nAfter clipping: {logvar_clipped}")
    print("→ Prevents extreme values")
    

* * *

## 2.4 Complete Implementation of VAE Architecture

### Implementation of Encoder and Decoder
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class VAE(nn.Module):
        """Complete VAE model"""
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super(VAE, self).__init__()
    
            self.input_dim = input_dim
            self.latent_dim = latent_dim
    
            # ===== Encoder (Recognition network) =====
            self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
            self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
            # ===== Decoder (Generative network) =====
            self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
            self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """
            Encoder: x -> (μ, log σ²)
    
            Args:
                x: (batch, input_dim)
    
            Returns:
                mu: (batch, latent_dim)
                logvar: (batch, latent_dim)
            """
            h = F.relu(self.encoder_fc1(x))
            mu = self.encoder_fc_mu(h)
            logvar = self.encoder_fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """
            Reparameterization Trick: z = μ + σ·ϵ
    
            Args:
                mu: (batch, latent_dim)
                logvar: (batch, latent_dim)
    
            Returns:
                z: (batch, latent_dim)
            """
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
            return z
    
        def decode(self, z):
            """
            Decoder: z -> x̂
    
            Args:
                z: (batch, latent_dim)
    
            Returns:
                x_recon: (batch, input_dim)
            """
            h = F.relu(self.decoder_fc1(z))
            x_recon = torch.sigmoid(self.decoder_fc2(h))
            return x_recon
    
        def forward(self, x):
            """
            Forward pass: x -> z -> x̂
    
            Args:
                x: (batch, input_dim)
    
            Returns:
                x_recon: (batch, input_dim) - Reconstructed input
                mu: (batch, latent_dim) - Mean
                logvar: (batch, latent_dim) - Log variance
            """
            # Encode
            mu, logvar = self.encode(x)
    
            # Reparameterize
            z = self.reparameterize(mu, logvar)
    
            # Decode
            x_recon = self.decode(z)
    
            return x_recon, mu, logvar
    
        def sample(self, num_samples, device='cpu'):
            """
            Generate images by sampling from latent space
    
            Args:
                num_samples: Number of samples to generate
                device: Device
    
            Returns:
                samples: (num_samples, input_dim)
            """
            # Sample latent variables from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
    
            # Generate with Decoder
            samples = self.decode(z)
    
            return samples
    
    
    # Create model
    print("=== Creating VAE Model ===")
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # Test
    batch_size = 32
    x = torch.rand(batch_size, 784)  # MNIST images normalized to 0-1
    
    x_recon, mu, logvar = vae(x)
    
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"Mean: {mu.shape}")
    print(f"Log variance: {logvar.shape}")
    
    # Calculate loss
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    print(f"\nLoss:")
    print(f"  Total: {total_loss.item():.2f}")
    print(f"  Reconstruction: {recon_loss.item():.2f}")
    print(f"  KL divergence: {kl_loss.item():.2f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Random sampling
    print("\n=== Random Sampling ===")
    samples = vae.sample(num_samples=10)
    print(f"Generated samples: {samples.shape}")
    print("→ VAE can generate new images from latent space!")
    

### Convolutional VAE
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConvVAE(nn.Module):
        """Convolutional VAE (for images)"""
        def __init__(self, input_channels=1, latent_dim=128):
            super(ConvVAE, self).__init__()
    
            self.latent_dim = latent_dim
    
            # ===== Encoder =====
            self.encoder_conv = nn.Sequential(
                # 28x28 -> 14x14
                nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # 14x14 -> 7x7
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # 7x7 -> 3x3
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )
    
            # Latent variable parameters
            self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
            self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)
    
            # ===== Decoder =====
            self.decoder_fc = nn.Linear(latent_dim, 128 * 3 * 3)
    
            self.decoder_conv = nn.Sequential(
                # 3x3 -> 7x7
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # 7x7 -> 14x14
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # 14x14 -> 28x28
                nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
    
        def encode(self, x):
            """
            Args:
                x: (batch, channels, 28, 28)
            Returns:
                mu, logvar: (batch, latent_dim)
            """
            h = self.encoder_conv(x)  # (batch, 128, 3, 3)
            h = h.view(h.size(0), -1)  # (batch, 128*3*3)
    
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
    
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
            return z
    
        def decode(self, z):
            """
            Args:
                z: (batch, latent_dim)
            Returns:
                x_recon: (batch, channels, 28, 28)
            """
            h = self.decoder_fc(z)  # (batch, 128*3*3)
            h = h.view(h.size(0), 128, 3, 3)  # (batch, 128, 3, 3)
            x_recon = self.decoder_conv(h)  # (batch, channels, 28, 28)
            return x_recon
    
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
    
        def sample(self, num_samples, device='cpu'):
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples
    
    
    # Create model
    print("\n=== Creating Convolutional VAE ===")
    conv_vae = ConvVAE(input_channels=1, latent_dim=128)
    
    # Test
    batch_size = 16
    x_img = torch.rand(batch_size, 1, 28, 28)  # MNIST images
    
    x_recon, mu, logvar = conv_vae(x_img)
    
    print(f"Input image: {x_img.shape}")
    print(f"Reconstructed image: {x_recon.shape}")
    print(f"Latent variable: {mu.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in conv_vae.parameters())
    print(f"Total parameters: {total_params:,}")
    print("→ Convolutional layers preserve spatial structure of images")
    

* * *

## 2.5 Training in PyTorch and MNIST Image Generation

### Dataset Preparation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Dataset Preparation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Prepare MNIST dataset
    print("=== Preparing MNIST Dataset ===")
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Normalize to 0-1
    ])
    
    # Download (first time only)
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # DataLoader
    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training data: {len(train_dataset)}")
    print(f"Test data: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    

### Training Loop Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.optim as optim
    
    def train_epoch(model, train_loader, optimizer, device):
        """Train for one epoch"""
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
    
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
    
            # Flatten images (for fully-connected VAE)
            data_flat = data.view(data.size(0), -1)
    
            # Forward
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data_flat)
    
            # Calculate loss
            total_loss, recon_loss, kl_loss = vae_loss(x_recon, data_flat, mu, logvar)
    
            # Backward
            total_loss.backward()
            optimizer.step()
    
            # Accumulate
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
    
        # Average loss
        num_samples = len(train_loader.dataset)
        avg_loss = train_loss / num_samples
        avg_recon = train_recon_loss / num_samples
        avg_kl = train_kl_loss / num_samples
    
        return avg_loss, avg_recon, avg_kl
    
    
    def test_epoch(model, test_loader, device):
        """Evaluate on test data"""
        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
    
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)
    
                x_recon, mu, logvar = model(data_flat)
                total_loss, recon_loss, kl_loss = vae_loss(x_recon, data_flat, mu, logvar)
    
                test_loss += total_loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
    
        num_samples = len(test_loader.dataset)
        avg_loss = test_loss / num_samples
        avg_recon = test_recon_loss / num_samples
        avg_kl = test_kl_loss / num_samples
    
        return avg_loss, avg_recon, avg_kl
    
    
    # Training setup
    print("\n=== Training VAE ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    num_epochs = 10
    print(f"Number of epochs: {num_epochs}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
    
        # Test
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device)
    
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
    
    print("\nTraining complete!")
    

### Image Generation and Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_reconstruction(model, test_loader, device, num_samples=10):
        """Visualize reconstruction results"""
        model.eval()
    
        # Get one batch from test data
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        data_flat = data.view(data.size(0), -1)
    
        # Reconstruct
        with torch.no_grad():
            x_recon, mu, logvar = model(data_flat)
    
        # Move to CPU
        data = data.cpu().numpy()
        x_recon = x_recon.view(-1, 1, 28, 28).cpu().numpy()
    
        # Visualize
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples, 2))
    
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(data[i, 0], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
    
            # Reconstructed image
            axes[1, i].imshow(x_recon[i, 0], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
    
        plt.tight_layout()
        plt.savefig('vae_reconstruction.png', dpi=150, bbox_inches='tight')
        print("Saved reconstruction results: vae_reconstruction.png")
        plt.close()
    
    
    def visualize_samples(model, device, num_samples=16):
        """Visualize images generated by random sampling"""
        model.eval()
    
        # Random sampling
        with torch.no_grad():
            samples = model.sample(num_samples, device)
    
        samples = samples.view(-1, 1, 28, 28).cpu().numpy()
    
        # Visualize
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
    
        plt.suptitle('Randomly Generated Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig('vae_samples.png', dpi=150, bbox_inches='tight')
        print("Saved generated samples: vae_samples.png")
        plt.close()
    
    
    # Execute visualization
    print("\n=== Visualizing Results ===")
    visualize_reconstruction(model, test_loader, device, num_samples=10)
    visualize_samples(model, device, num_samples=16)
    print("→ VAE can reconstruct original images and generate new images!")
    

* * *

## 2.6 Latent Space Visualization and Interpolation

### 2D Latent Space Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_latent_space_2d(model, test_loader, device):
        """Visualize 2D latent space (for latent_dim=2)"""
        model.eval()
    
        z_list = []
        label_list = []
    
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)
    
                mu, logvar = model.encode(data_flat)
                z_list.append(mu.cpu())
                label_list.append(labels)
    
        z = torch.cat(z_list, dim=0).numpy()
        labels = torch.cat(label_list, dim=0).numpy()
    
        # Scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('2D Latent Space Visualization (MNIST)')
        plt.grid(True, alpha=0.3)
        plt.savefig('vae_latent_2d.png', dpi=150, bbox_inches='tight')
        print("Saved 2D latent space: vae_latent_2d.png")
        plt.close()
    
    
    # Experiment with 2D VAE
    print("\n=== Visualizing 2D Latent Space ===")
    model_2d = VAE(input_dim=784, hidden_dim=400, latent_dim=2).to(device)
    
    # Simple training (optional - use pre-trained model if available)
    optimizer_2d = optim.Adam(model_2d.parameters(), lr=1e-3)
    for epoch in range(3):
        train_loss, _, _ = train_epoch(model_2d, train_loader, optimizer_2d, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
    
    # Visualize
    visualize_latent_space_2d(model_2d, test_loader, device)
    print("→ Verify that different digits form clusters")
    

### Latent Space Interpolation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    
    def interpolate_latent_space(model, z_start, z_end, num_steps=10):
        """
        Interpolate two points in latent space
    
        Args:
            model: VAE model
            z_start: (latent_dim,) - Start point
            z_end: (latent_dim,) - End point
            num_steps: Number of interpolation steps
    
        Returns:
            interpolated_images: (num_steps, 1, 28, 28)
        """
        model.eval()
    
        # Linear interpolation
        alphas = torch.linspace(0, 1, num_steps)
        z_interp = torch.stack([
            (1 - alpha) * z_start + alpha * z_end
            for alpha in alphas
        ])
    
        # Generate with Decoder
        with torch.no_grad():
            images = model.decode(z_interp)
    
        images = images.view(-1, 1, 28, 28)
    
        return images
    
    
    def visualize_interpolation(model, test_loader, device, num_pairs=3, num_steps=10):
        """Visualize interpolation results"""
        model.eval()
    
        # Select images from test data
        data, _ = next(iter(test_loader))
        data = data[:num_pairs*2].to(device)
        data_flat = data.view(data.size(0), -1)
    
        # Encode to get latent variables
        with torch.no_grad():
            mu, _ = model.encode(data_flat)
    
        # Visualize
        fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps, num_pairs))
    
        for i in range(num_pairs):
            z_start = mu[i*2]
            z_end = mu[i*2 + 1]
    
            # Interpolate
            images = interpolate_latent_space(model, z_start, z_end, num_steps)
            images = images.cpu().numpy()
    
            for j in range(num_steps):
                ax = axes[i, j] if num_pairs > 1 else axes[j]
                ax.imshow(images[j, 0], cmap='gray')
                ax.axis('off')
    
        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
        print("Saved interpolation results: vae_interpolation.png")
        plt.close()
    
    
    # Visualize interpolation
    print("\n=== Latent Space Interpolation ===")
    visualize_interpolation(model, test_loader, device, num_pairs=3, num_steps=10)
    print("→ Smoothly changing intermediate images are generated")
    

### Exploring Latent Space Structure
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_latent_manifold(model, device, n=20, digit_size=28):
        """
        Visualize latent space manifold in 2D
    
        Args:
            model: VAE with 2D latent variables
            device: Device
            n: Grid size
            digit_size: Image size
        """
        model.eval()
    
        # Grid range (covers 99% of normal distribution)
        grid_range = 3
        grid_x = np.linspace(-grid_range, grid_range, n)
        grid_y = np.linspace(-grid_range, grid_range, n)
    
        # Prepare full image
        figure = np.zeros((digit_size * n, digit_size * n))
    
        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    # Latent variable
                    z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
    
                    # Generate with Decoder
                    x_recon = model.decode(z)
                    digit = x_recon.view(digit_size, digit_size).cpu().numpy()
    
                    # Place
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit
    
        # Visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.axis('off')
        plt.title('Latent Space Manifold (2D)', fontsize=14)
        plt.savefig('vae_manifold.png', dpi=150, bbox_inches='tight')
        print("Saved latent space manifold: vae_manifold.png")
        plt.close()
    
    
    # Visualize manifold
    print("\n=== Latent Space Manifold ===")
    visualize_latent_manifold(model_2d, device, n=20, digit_size=28)
    print("→ Verify how digits are distributed in 2D space")
    

* * *

## 2.7 Practical Application: CelebA Face Image Generation

### Convolutional VAE for CelebA
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CelebAVAE(nn.Module):
        """VAE for CelebA face images (64x64 resolution)"""
        def __init__(self, input_channels=3, latent_dim=256):
            super(CelebAVAE, self).__init__()
    
            self.latent_dim = latent_dim
    
            # ===== Encoder =====
            self.encoder = nn.Sequential(
                # 64x64 -> 32x32
                nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 32x32 -> 16x16
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 16x16 -> 8x8
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 8x8 -> 4x4
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
    
            self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    
            # ===== Decoder =====
            self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
    
            self.decoder = nn.Sequential(
                # 4x4 -> 8x8
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 8x8 -> 16x16
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 16x16 -> 32x32
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 32x32 -> 64x64
                nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
    
        def encode(self, x):
            h = self.encoder(x)
            h = h.view(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
            return z
    
        def decode(self, z):
            h = self.decoder_fc(z)
            h = h.view(h.size(0), 256, 4, 4)
            x_recon = self.decoder(h)
            return x_recon
    
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
    
        def sample(self, num_samples, device='cpu'):
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples
    
    
    # Create model
    print("\n=== Creating CelebA VAE ===")
    celeba_vae = CelebAVAE(input_channels=3, latent_dim=256)
    
    # Test
    batch_size = 16
    x_celeba = torch.rand(batch_size, 3, 64, 64)  # RGB 64x64 images
    
    x_recon, mu, logvar = celeba_vae(x_celeba)
    
    print(f"Input image: {x_celeba.shape}")
    print(f"Reconstructed image: {x_recon.shape}")
    print(f"Latent variable: {mu.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in celeba_vae.parameters())
    print(f"Total parameters: {total_params:,}")
    print("→ Deep architecture for high-resolution images")
    

* * *

## Exercises

**Exercise 1: Implementing β-VAE**

Implement **β-VAE** which introduces a weight $\beta$ to the KL term, and analyze the differences in latent space for various values of $\beta$.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementβ-VAEwhich introduces a weight $\beta$ to the KL te
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    
    # Exercise: Implement β-VAE loss function
    # Loss = Reconstruction + β * KL divergence
    
    # Exercise: Train with β = 0.5, 1.0, 2.0, 4.0
    # Exercise: Evaluate disentanglement with latent space visualization
    # Hint: Larger β leads to more independent latent variables
    

**Exercise 2: Investigating the Effect of Latent Dimensions**

Vary the number of latent dimensions (2, 10, 20, 50, 100) and investigate the tradeoff between reconstruction quality and generation diversity.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Vary the number of latent dimensions (2, 10, 20, 50, 100) an
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    
    # Exercise: Train models with different latent dimensions
    # Exercise: Compare reconstruction error, KL divergence, and quality of generated images
    # Exercise: Create a graph of dimensions vs performance
    # Expected: More dimensions improve reconstruction but reduce generation diversity
    

**Exercise 3: Implementing Conditional VAE (CVAE)**

Implement **Conditional VAE** that takes class labels as conditions.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: ImplementConditional VAEthat takes class labels as condition
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    
    # Exercise: Verify that images of specified classes can be generated
    # Exercise: Visualize interpolation between different classes
    

**Exercise 4: Comparing Reconstruction Errors (MSE vs BCE)**

Compare MSE (Mean Squared Error) and BCE (Binary Cross Entropy) as reconstruction errors.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Compare MSE (Mean Squared Error) and BCE (Binary Cross Entro
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.nn.functional as F
    
    # Exercise: Train models with both loss functions
    # Exercise: Visually compare quality of reconstructed images
    # Exercise: Record convergence speed and final values of losses
    # Analysis: BCE has probabilistic interpretation and is suitable for binary images like MNIST
    

**Exercise 5: Implementing KL Annealing**

Implement **KL Annealing** which starts with a small weight for the KL term and gradually increases it during training.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: ImplementKL Annealingwhich starts with a small weight for th
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    
    # weight = min(1.0, epoch / num_annealing_epochs)
    
    # Exercise: Compare convergence speed with and without Annealing
    # Exercise: Evaluate final latent space quality
    # Expected: Annealing stabilizes training and yields better latent space
    

* * *

## Summary

In this chapter, we learned the theory and implementation of VAE (Variational Autoencoder).

### Key Points

  * **Limitations of Autoencoders** : Deterministic and unstructured latent space
  * **VAE Motivation** : Acquire generative model capabilities with probabilistic framework
  * **ELBO** : Objective function composed of reconstruction term and KL regularization term
  * **KL Divergence** : Regularizes latent variables to standard normal distribution
  * **Reparameterization Trick** : Technique to make probabilistic sampling differentiable
  * **Encoder/Decoder** : Output parameters (mean and variance) of Gaussian distribution
  * **Latent Space** : Structured and enables interpolation and exploration
  * **Implementation** : Complete implementation of MNIST/CelebA image generation systems
  * **Applications** : Extension techniques such as β-VAE, CVAE, and KL Annealing

### Next Steps

In the next chapter, we will learn about **GAN (Generative Adversarial Networks)**. We will master higher-quality image generation through adversarial learning, Generator/Discriminator architectures, training stabilization techniques, and other generative model approaches that differ from VAE.
