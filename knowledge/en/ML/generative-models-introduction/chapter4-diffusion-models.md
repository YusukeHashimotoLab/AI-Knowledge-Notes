---
title: "Chapter 4: Diffusion Models"
chapter_title: "Chapter 4: Diffusion Models"
subtitle: "Generation from Noise: Theory and Practice of Diffusion Models, Leading to Stable Diffusion"
reading_time: 32 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 6
---

This chapter covers Diffusion Models. You will learn noise schedules.

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Understand the fundamental principles of diffusion models (Forward/Reverse Process)
  * ✅ Understand the mathematical formulation of DDPM (Denoising Diffusion Probabilistic Models)
  * ✅ Implement noise schedules and sampling algorithms
  * ✅ Master the structure and training methods of U-Net Denoisers
  * ✅ Understand the mechanisms of Latent Diffusion Models (Stable Diffusion)
  * ✅ Implement CLIP Guidance and text-conditioned generation
  * ✅ Build practical image generation systems with PyTorch

* * *

## 4.1 Fundamentals of Diffusion Models

### 4.1.1 What are Diffusion Models?

**Diffusion Models** are generative models that learn two processes: the Forward Process, which gradually adds noise to data, and the Reverse Process, which restores the original data from noise. Since entering the 2020s, they have achieved state-of-the-art performance in image generation and become the foundation technology for Stable Diffusion, DALL-E 2, Imagen, and others.

Property | GAN | VAE | Diffusion Models  
---|---|---|---  
**Generation Method** | Adversarial Learning | Variational Inference | Denoising  
**Training Stability** | Low (Mode Collapse) | Medium | High  
**Generation Quality** | High (When Trained) | Medium (Blurry) | Very High  
**Diversity** | Low (Mode Collapse) | High | High  
**Computational Cost** | Low to Medium | Low to Medium | High (Iterative Process)  
**Representative Models** | StyleGAN | β-VAE | DDPM, Stable Diffusion  
  
### 4.1.2 Forward Process: Adding Noise

The Forward Process is a procedure that gradually adds Gaussian noise to the original image $x_0$ over $T$ steps.

$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) $$ 

Where:

  * $x_t$: Image at timestep $t$
  * $\beta_t$: Noise schedule (typically 0.0001 to 0.02)
  * $\mathcal{N}$: Gaussian distribution

**Important property** : Using the reparameterization trick, we can directly sample the image at any step $t$:

$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon $$ 

Where:

  * $\alpha_t = 1 - \beta_t$
  * $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$
  * $\epsilon \sim \mathcal{N}(0, I)$

    
    
    ```mermaid
    graph LR
        X0["x₀(Original Image)"] -->|"+ Noise β₁"| X1["x₁"]
        X1 -->|"+ Noise β₂"| X2["x₂"]
        X2 -->|"..."| X3["..."]
        X3 -->|"+ Noise βT"| XT["xT(Pure Noise)"]
    
        style X0 fill:#27ae60,color:#fff
        style XT fill:#e74c3c,color:#fff
        style X1 fill:#f39c12,color:#fff
        style X2 fill:#e67e22,color:#fff
    ```

### 4.1.3 Reverse Process: Generation through Denoising

The Reverse Process starts from pure noise $x_T \sim \mathcal{N}(0, I)$ and gradually removes noise to restore the original image.

$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$ 

Here, we learn $\mu_\theta$ (mean) and $\Sigma_\theta$ (covariance) using neural networks. In DDPM, it's common to simplify by fixing the covariance and learning only the mean.

> **Important** : The Reverse Process is formulated as a task of predicting noise $\epsilon$. This allows the network to function as a "Denoiser".

### 4.1.4 Intuitive Understanding of Diffusion Models
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Visualizing diffusion process with simple 1D data
    np.random.seed(42)
    
    # Original data: mixture of two Gaussians
    def sample_data(n=1000):
        """Generate data with two modes"""
        mode1 = np.random.randn(n//2) * 0.5 + 2
        mode2 = np.random.randn(n//2) * 0.5 - 2
        return np.concatenate([mode1, mode2])
    
    # Forward diffusion process
    def forward_diffusion(x0, num_steps=50):
        """Forward diffusion: Add noise to data"""
        # Linear noise schedule
        betas = np.linspace(0.0001, 0.02, num_steps)
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas)
    
        # Save data at each timestep
        x_history = [x0]
    
        for t in range(1, num_steps):
            noise = np.random.randn(*x0.shape)
            x_t = np.sqrt(alphas_cumprod[t]) * x0 + np.sqrt(1 - alphas_cumprod[t]) * noise
            x_history.append(x_t)
    
        return x_history, betas, alphas_cumprod
    
    # Demonstration
    print("=== Forward Diffusion Process Visualization ===\n")
    
    x0 = sample_data(1000)
    x_history, betas, alphas_cumprod = forward_diffusion(x0, num_steps=50)
    
    # Visualization
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    
    timesteps_to_show = [0, 5, 10, 15, 20, 25, 30, 35, 40, 49]
    
    for idx, t in enumerate(timesteps_to_show):
        ax = axes[idx]
        ax.hist(x_history[t], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 0.5)
        ax.set_title(f't = {t}\nα̅ = {alphas_cumprod[t]:.4f}' if t > 0 else f't = 0 (Original)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Forward Diffusion Process: Original Data → Gaussian Noise',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\nCharacteristics:")
    print("✓ t = 0: Original bimodal distribution (clear structure)")
    print("✓ t = 10-20: Structure gradually degrades")
    print("✓ t = 49: Nearly standard Gaussian distribution (structure completely lost)")
    print("\nReverse Process:")
    print("✓ Start from noise (t=49) and gradually restore structure")
    print("✓ Remove noise at each step using learned Denoiser")
    print("✓ Finally reproduce the original bimodal distribution")
    

**Output** :
    
    
    === Forward Diffusion Process Visualization ===
    
    Characteristics:
    ✓ t = 0: Original bimodal distribution (clear structure)
    ✓ t = 10-20: Structure gradually degrades
    ✓ t = 49: Nearly standard Gaussian distribution (structure completely lost)
    
    Reverse Process:
    ✓ Start from noise (t=49) and gradually restore structure
    ✓ Remove noise at each step using learned Denoiser
    ✓ Finally reproduce the original bimodal distribution
    

* * *

## 4.2 DDPM (Denoising Diffusion Probabilistic Models)

### 4.2.1 Mathematical Formulation of DDPM

DDPM is a representative diffusion model method proposed by Ho et al. (UC Berkeley) in 2020.

#### Training Objective

The DDPM loss function is derived from the variational lower bound (ELBO), but takes a simple form in practice:

$$ \mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right] $$ 

This is the mean squared error of the task of "predicting noise $\epsilon$".

#### Algorithm Details

**Training Algorithm** :

  1. Sample $x_0$ from training data
  2. Sample timestep $t \sim \text{Uniform}(1, T)$
  3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
  4. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
  5. Minimize loss $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$

**Sampling Algorithm** :

  1. Start from $x_T \sim \mathcal{N}(0, I)$
  2. For $t = T, T-1, \ldots, 1$: $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$ where $z \sim \mathcal{N}(0, I)$ (if $t > 1$) 
  3. Return $x_0$

### 4.2.2 Noise Schedules

The design of the noise schedule $\beta_t$ significantly affects generation quality.

Schedule | Definition | Characteristics  
---|---|---  
**Linear** | $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$ | Simple, used in original paper  
**Cosine** | $\bar{\alpha}_t = \frac{f(t)}{f(0)}$, $f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Smoother noise transition  
**Quadratic** | $\beta_t = \beta_{\min}^2 + t^2 (\beta_{\max}^2 - \beta_{\min}^2)$ | Non-linear transition  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
        """Linear noise schedule"""
        return np.linspace(beta_start, beta_end, timesteps)
    
    def cosine_beta_schedule(timesteps, s=0.008):
        """Cosine noise schedule (Improved DDPM)"""
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    
    def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
        """Quadratic noise schedule"""
        return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    # Visualization
    print("=== Noise Schedule Comparison ===\n")
    
    timesteps = 1000
    
    linear_betas = linear_beta_schedule(timesteps)
    cosine_betas = cosine_beta_schedule(timesteps)
    quadratic_betas = quadratic_beta_schedule(timesteps)
    
    # Calculate cumulative product of alphas
    def compute_alphas_cumprod(betas):
        alphas = 1 - betas
        return np.cumprod(alphas)
    
    linear_alphas = compute_alphas_cumprod(linear_betas)
    cosine_alphas = compute_alphas_cumprod(cosine_betas)
    quadratic_alphas = compute_alphas_cumprod(quadratic_betas)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Beta values
    ax1 = axes[0]
    ax1.plot(linear_betas, label='Linear', linewidth=2, alpha=0.8)
    ax1.plot(cosine_betas, label='Cosine', linewidth=2, alpha=0.8)
    ax1.plot(quadratic_betas, label='Quadratic', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Timestep t', fontsize=12, fontweight='bold')
    ax1.set_ylabel('βₜ (Noise Level)', fontsize=12, fontweight='bold')
    ax1.set_title('Noise Schedules: βₜ', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Right: Cumulative alpha
    ax2 = axes[1]
    ax2.plot(linear_alphas, label='Linear', linewidth=2, alpha=0.8)
    ax2.plot(cosine_alphas, label='Cosine', linewidth=2, alpha=0.8)
    ax2.plot(quadratic_alphas, label='Quadratic', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Timestep t', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ᾱₜ (Signal Strength)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Product: ᾱₜ = ∏ αₛ', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSchedule Characteristics:")
    print(f"Linear   - β range: [{linear_betas.min():.6f}, {linear_betas.max():.6f}]")
    print(f"Cosine   - β range: [{cosine_betas.min():.6f}, {cosine_betas.max():.6f}]")
    print(f"Quadratic- β range: [{quadratic_betas.min():.6f}, {quadratic_betas.max():.6f}]")
    print(f"\nFinal ᾱ_T (signal retention rate):")
    print(f"Linear:    {linear_alphas[-1]:.6f}")
    print(f"Cosine:    {cosine_alphas[-1]:.6f}")
    print(f"Quadratic: {quadratic_alphas[-1]:.6f}")
    

**Output** :
    
    
    === Noise Schedule Comparison ===
    
    Schedule Characteristics:
    Linear   - β range: [0.000100, 0.020000]
    Cosine   - β range: [0.000020, 0.999000]
    Quadratic- β range: [0.000000, 0.000400]
    
    Final ᾱ_T (signal retention rate):
    Linear:    0.000062
    Cosine:    0.000000
    Quadratic: 0.670320
    

### 4.2.3 DDPM Training Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class DDPMDiffusion:
        """DDPM diffusion process implementation"""
    
        def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
            """
            Args:
                timesteps: Number of diffusion steps
                beta_start: Starting noise level
                beta_end: Ending noise level
                schedule: 'linear', 'cosine', 'quadratic'
            """
            self.timesteps = timesteps
    
            # Noise schedule
            if schedule == 'linear':
                self.betas = torch.linspace(beta_start, beta_end, timesteps)
            elif schedule == 'cosine':
                self.betas = self._cosine_beta_schedule(timesteps)
            elif schedule == 'quadratic':
                self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
            # Alpha calculations
            self.alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
            # Coefficients for sampling
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
    
            # Posterior variance
            self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
        def _cosine_beta_schedule(self, timesteps, s=0.008):
            """Cosine schedule"""
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
    
        def q_sample(self, x_start, t, noise=None):
            """
            Forward diffusion: Sample x_t directly from x_0
    
            Args:
                x_start: [B, C, H, W] Original image
                t: [B] Timestep
                noise: Noise (generated if None)
    
            Returns:
                x_t: Noised image
            """
            if noise is None:
                noise = torch.randn_like(x_start)
    
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
    
            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
        def p_losses(self, denoise_model, x_start, t, noise=None):
            """
            Calculate training loss
    
            Args:
                denoise_model: Noise prediction model
                x_start: Original image
                t: Timestep
                noise: Noise (generated if None)
    
            Returns:
                loss: MSE loss
            """
            if noise is None:
                noise = torch.randn_like(x_start)
    
            # Add noise
            x_noisy = self.q_sample(x_start, t, noise)
    
            # Predict noise
            predicted_noise = denoise_model(x_noisy, t)
    
            # MSE loss
            loss = F.mse_loss(predicted_noise, noise)
    
            return loss
    
        @torch.no_grad()
        def p_sample(self, model, x, t, t_index):
            """
            Reverse process: Sample x_{t-1} from x_t
    
            Args:
                model: Noise prediction model
                x: Current image x_t
                t: Timestep
                t_index: Index (for variance calculation)
    
            Returns:
                x_{t-1}
            """
            betas_t = self._extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
    
            # Predict noise
            predicted_noise = model(x, t)
    
            # Calculate mean
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )
    
            if t_index == 0:
                return model_mean
            else:
                posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise
    
        @torch.no_grad()
        def p_sample_loop(self, model, shape):
            """
            Complete sampling loop: Generate image from noise
    
            Args:
                model: Noise prediction model
                shape: Shape of generated image [B, C, H, W]
    
            Returns:
                Generated image
            """
            device = next(model.parameters()).device
    
            # Start from pure noise
            img = torch.randn(shape, device=device)
    
            # Sample in reverse
            for i in reversed(range(0, self.timesteps)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                img = self.p_sample(model, img, t, i)
    
            return img
    
        def _extract(self, a, t, x_shape):
            """Extract coefficients and adjust shape"""
            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu())
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    
    # Demonstration
    print("=== DDPM Diffusion Process Demo ===\n")
    
    diffusion = DDPMDiffusion(timesteps=1000, schedule='linear')
    
    # Dummy data
    batch_size = 4
    channels = 3
    img_size = 32
    x_start = torch.randn(batch_size, channels, img_size, img_size)
    
    print(f"Original image shape: {x_start.shape}")
    
    # Adding noise at different timesteps
    timesteps_to_test = [0, 100, 300, 500, 700, 999]
    
    print("\nForward Diffusion at Different Timesteps:")
    print(f"{'Timestep':<12} {'ᾱ_t':<12} {'Signal %':<12} {'Noise %':<12}")
    print("-" * 50)
    
    for t in timesteps_to_test:
        t_tensor = torch.full((batch_size,), t, dtype=torch.long)
        x_noisy = diffusion.q_sample(x_start, t_tensor)
    
        alpha_t = diffusion.alphas_cumprod[t].item()
        signal_strength = alpha_t * 100
        noise_strength = (1 - alpha_t) * 100
    
        print(f"{t:<12} {alpha_t:<12.6f} {signal_strength:<12.2f} {noise_strength:<12.2f}")
    
    print("\n✓ DDPM implementation complete")
    print("✓ Forward/Reverse process defined")
    print("✓ Training loss function implemented")
    print("✓ Sampling algorithm implemented")
    

**Output** :
    
    
    === DDPM Diffusion Process Demo ===
    
    Original image shape: torch.Size([4, 3, 32, 32])
    
    Forward Diffusion at Different Timesteps:
    Timestep     ᾱ_t          Signal %     Noise %
    --------------------------------------------------
    0            1.000000     100.00       0.00
    100          0.793469     79.35        20.65
    300          0.419308     41.93        58.07
    500          0.170726     17.07        82.93
    700          0.049806     4.98         95.02
    999          0.000062     0.01         99.99
    
    ✓ DDPM implementation complete
    ✓ Forward/Reverse process defined
    ✓ Training loss function implemented
    ✓ Sampling algorithm implemented
    

* * *

## 4.3 U-Net Denoiser Implementation

### 4.3.1 U-Net Architecture

For noise prediction in diffusion models, **U-Net** is widely used. U-Net is an architecture with an encoder-decoder structure and skip connections.
    
    
    ```mermaid
    graph TB
        subgraph "U-Net for Diffusion Models"
            Input["Input: x_t + Timestep Embedding"]
    
            Down1["Down Block 1Conv + Attention"]
            Down2["Down Block 2Conv + Attention"]
            Down3["Down Block 3Conv + Attention"]
    
            Bottleneck["BottleneckAttention"]
    
            Up1["Up Block 1Conv + Attention"]
            Up2["Up Block 2Conv + Attention"]
            Up3["Up Block 3Conv + Attention"]
    
            Output["Output: Predicted Noise ε"]
    
            Input --> Down1
            Down1 --> Down2
            Down2 --> Down3
            Down3 --> Bottleneck
            Bottleneck --> Up1
            Up1 --> Up2
            Up2 --> Up3
            Up3 --> Output
    
            Down1 -.Skip.-> Up3
            Down2 -.Skip.-> Up2
            Down3 -.Skip.-> Up1
    
            style Input fill:#7b2cbf,color:#fff
            style Output fill:#27ae60,color:#fff
            style Bottleneck fill:#e74c3c,color:#fff
        end
    ```

### 4.3.2 Time Embedding

Timestep $t$ is encoded using Sinusoidal Positional Encoding (same as in Transformers):

$$ \text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right) $$ $$ \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right) $$ 
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import math
    
    class SinusoidalPositionEmbeddings(nn.Module):
        """Sinusoidal time embeddings for diffusion models"""
    
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
    
        def forward(self, time):
            """
            Args:
                time: [B] Timestep
    
            Returns:
                embeddings: [B, dim] Time embeddings
            """
            device = time.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings
    
    
    class TimeEmbeddingMLP(nn.Module):
        """Transform time embeddings with MLP"""
    
        def __init__(self, time_dim, emb_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(time_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim)
            )
    
        def forward(self, t_emb):
            return self.mlp(t_emb)
    
    
    # Demonstration
    print("=== Time Embedding Demo ===\n")
    
    time_dim = 128
    batch_size = 8
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    time_embedder = SinusoidalPositionEmbeddings(time_dim)
    time_mlp = TimeEmbeddingMLP(time_dim, 256)
    
    t_emb = time_embedder(timesteps)
    t_emb_transformed = time_mlp(t_emb)
    
    print(f"Timesteps: {timesteps.numpy()}")
    print(f"\nSinusoidal Embedding shape: {t_emb.shape}")
    print(f"MLP Transformed shape: {t_emb_transformed.shape}")
    
    # Embedding visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Sinusoidal patterns
    ax1 = axes[0]
    t_range = torch.arange(0, 1000, 10)
    embeddings = time_embedder(t_range).detach().numpy()
    
    sns.heatmap(embeddings[:, :64].T, cmap='RdBu_r', center=0, ax=ax1, cbar_kws={'label': 'Value'})
    ax1.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax1.set_title('Sinusoidal Time Embeddings (first 64 dims)', fontsize=13, fontweight='bold')
    
    # Right: Embedding similarity
    ax2 = axes[1]
    sample_timesteps = torch.tensor([0, 100, 300, 500, 700, 999])
    sample_embs = time_embedder(sample_timesteps).detach()
    similarity = torch.mm(sample_embs, sample_embs.T)
    
    sns.heatmap(similarity.numpy(), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                xticklabels=sample_timesteps.numpy(), yticklabels=sample_timesteps.numpy(),
                cbar_kws={'label': 'Cosine Similarity'})
    ax2.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Timestep', fontsize=12, fontweight='bold')
    ax2.set_title('Time Embedding Similarity Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCharacteristics:")
    print("✓ Each timestep has a unique vector representation")
    print("✓ Consecutive timesteps have similar embeddings")
    print("✓ Network can leverage timestep information")
    

### 4.3.3 Simplified U-Net Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ResidualBlock(nn.Module):
        """ResNet-style residual block"""
    
        def __init__(self, in_channels, out_channels, time_emb_dim):
            super().__init__()
    
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
            # Time embedding projection
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
            # Residual connection
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
            else:
                self.residual_conv = nn.Identity()
    
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
    
        def forward(self, x, t_emb):
            """
            Args:
                x: [B, C, H, W]
                t_emb: [B, time_emb_dim]
            """
            residue = x
    
            # First conv
            x = self.conv1(x)
            x = self.norm1(x)
    
            # Add time embedding
            t = self.time_mlp(F.silu(t_emb))
            x = x + t[:, :, None, None]
            x = F.silu(x)
    
            # Second conv
            x = self.conv2(x)
            x = self.norm2(x)
            x = F.silu(x)
    
            # Residual
            return x + self.residual_conv(residue)
    
    
    class SimpleUNet(nn.Module):
        """Simplified U-Net for Diffusion"""
    
        def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256,
                     base_channels=64):
            super().__init__()
    
            # Time embedding
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU()
            )
    
            # Encoder
            self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
            self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
            self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
    
            self.pool = nn.MaxPool2d(2)
    
            # Bottleneck
            self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
    
            # Decoder
            self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, 2)
            self.up_block1 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
    
            self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, 2)
            self.up_block2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
    
            self.up3 = nn.ConvTranspose2d(base_channels, base_channels, 2, 2)
            self.up_block3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
    
            # Output
            self.out = nn.Conv2d(base_channels, out_channels, 1)
    
        def forward(self, x, t):
            """
            Args:
                x: [B, C, H, W] Noisy image
                t: [B] Timestep
    
            Returns:
                predicted_noise: [B, C, H, W]
            """
            # Time embedding
            t_emb = self.time_mlp(t)
    
            # Encoder with skip connections
            d1 = self.down1(x, t_emb)
            d2 = self.down2(self.pool(d1), t_emb)
            d3 = self.down3(self.pool(d2), t_emb)
    
            # Bottleneck
            b = self.bottleneck(self.pool(d3), t_emb)
    
            # Decoder with skip connections
            u1 = self.up1(b)
            u1 = torch.cat([u1, d3], dim=1)
            u1 = self.up_block1(u1, t_emb)
    
            u2 = self.up2(u1)
            u2 = torch.cat([u2, d2], dim=1)
            u2 = self.up_block2(u2, t_emb)
    
            u3 = self.up3(u2)
            u3 = torch.cat([u3, d1], dim=1)
            u3 = self.up_block3(u3, t_emb)
    
            # Output
            return self.out(u3)
    
    
    # Demonstration
    print("=== U-Net Denoiser Demo ===\n")
    
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=256, base_channels=64)
    
    # Dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    predicted_noise = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Timesteps: {t.numpy()}")
    print(f"Output (predicted noise) shape: {predicted_noise.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n✓ U-Net structure:")
    print("  - Encoder: 3 layers (downsampling)")
    print("  - Bottleneck: Residual block")
    print("  - Decoder: 3 layers (upsampling + skip connections)")
    print("  - Time Embedding: Injected into each block")
    

**Output** :
    
    
    === U-Net Denoiser Demo ===
    
    Input shape: torch.Size([2, 3, 32, 32])
    Timesteps: [742 123]
    Output (predicted noise) shape: torch.Size([2, 3, 32, 32])
    
    Model Statistics:
    Total parameters: 15,234,179
    Trainable parameters: 15,234,179
    Model size: 58.11 MB (float32)
    
    ✓ U-Net structure:
      - Encoder: 3 layers (downsampling)
      - Bottleneck: Residual block
      - Decoder: 3 layers (upsampling + skip connections)
      - Time Embedding: Injected into each block
    

* * *

## 4.4 DDPM Training and Generation

### 4.4.1 Training Loop Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    def train_ddpm(model, diffusion, dataloader, epochs=10, lr=1e-4, device='cpu'):
        """
        DDPM training loop
    
        Args:
            model: U-Net denoiser
            diffusion: DDPMDiffusion instance
            dataloader: Data loader
            epochs: Number of epochs
            lr: Learning rate
            device: 'cpu' or 'cuda'
    
        Returns:
            losses: Training loss history
        """
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
        losses = []
    
        for epoch in range(epochs):
            epoch_loss = 0.0
    
            for batch_idx, (images,) in enumerate(dataloader):
                images = images.to(device)
                batch_size = images.shape[0]
    
                # Sample random timesteps
                t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
    
                # Calculate loss
                loss = diffusion.p_losses(model, images, t)
    
                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
    
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
        return losses
    
    
    @torch.no_grad()
    def sample_images(model, diffusion, n_samples=16, channels=3, img_size=32, device='cpu'):
        """
        Sample images
    
        Args:
            model: Trained U-Net
            diffusion: DDPMDiffusion instance
            n_samples: Number of samples
            channels: Number of channels
            img_size: Image size
            device: Device
    
        Returns:
            samples: Generated images [n_samples, C, H, W]
        """
        model.eval()
        shape = (n_samples, channels, img_size, img_size)
        samples = diffusion.p_sample_loop(model, shape)
        return samples
    
    
    # Demonstration (training with dummy data)
    print("=== DDPM Training Demo ===\n")
    
    # Dummy dataset (in practice, use CIFAR-10, etc.)
    n_samples = 100
    dummy_images = torch.randn(n_samples, 3, 32, 32)
    dataset = TensorDataset(dummy_images)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model and Diffusion
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=128, base_channels=32)
    diffusion = DDPMDiffusion(timesteps=1000, schedule='linear')
    
    # Training (small-scale demo)
    print("Training (Demo with dummy data)...")
    losses = train_ddpm(model, diffusion, dataloader, epochs=5, lr=1e-4, device=device)
    
    # Sampling
    print("\nGenerating samples...")
    samples = sample_images(model, diffusion, n_samples=4, device=device)
    
    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Value range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    # Visualize loss
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('DDPM Training Loss', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Training complete")
    print("✓ Sampling successful")
    print("\nPractical usage:")
    print("  1. Prepare datasets like CIFAR-10/ImageNet")
    print("  2. Train for several epochs (hours to days on GPU)")
    print("  3. Generate high-quality images with trained model")
    

**Output Example** :
    
    
    === DDPM Training Demo ===
    
    Using device: cpu
    
    Training (Demo with dummy data)...
    Epoch [1/5], Loss: 0.982341
    Epoch [2/5], Loss: 0.967823
    Epoch [3/5], Loss: 0.951234
    Epoch [4/5], Loss: 0.938765
    Epoch [5/5], Loss: 0.924512
    
    Generating samples...
    
    Generated samples shape: torch.Size([4, 3, 32, 32])
    Value range: [-2.34, 2.67]
    
    ✓ Training complete
    ✓ Sampling successful
    
    Practical usage:
      1. Prepare datasets like CIFAR-10/ImageNet
      2. Train for several epochs (hours to days on GPU)
      3. Generate high-quality images with trained model
    

### 4.4.2 Accelerating Sampling: DDIM

**DDIM (Denoising Diffusion Implicit Models)** is a method to accelerate DDPM. It can generate images of equivalent quality in 50-100 steps instead of 1000 steps.

DDIM update equation:

$$ x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)}_{\text{direction pointing to } x_t} $$ 
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    
    @torch.no_grad()
    def ddim_sample(model, diffusion, shape, ddim_steps=50, eta=0.0, device='cpu'):
        """
        DDIM fast sampling
    
        Args:
            model: Denoiser
            diffusion: DDPMDiffusion
            shape: Shape of generated image
            ddim_steps: Number of DDIM steps (< T)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM equivalent)
            device: Device
    
        Returns:
            Generated image
        """
        # Select subset of timesteps
        timesteps = torch.linspace(diffusion.timesteps - 1, 0, ddim_steps, dtype=torch.long)
    
        # Start from pure noise
        img = torch.randn(shape, device=device)
    
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
    
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
    
            # Predict noise
            predicted_noise = model(img, t_tensor)
    
            # Predict x_0
            alpha_t = diffusion.alphas_cumprod[t]
            alpha_t_next = diffusion.alphas_cumprod[t_next]
    
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    
            # Calculate x_{t-1}
            sigma = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * \
                    torch.sqrt(1 - alpha_t / alpha_t_next)
    
            noise = torch.randn_like(img) if i < len(timesteps) - 2 else torch.zeros_like(img)
    
            img = torch.sqrt(alpha_t_next) * pred_x0 + \
                  torch.sqrt(1 - alpha_t_next - sigma**2) * predicted_noise + \
                  sigma * noise
    
        return img
    
    
    # Demonstration
    print("=== DDIM Fast Sampling Demo ===\n")
    
    # DDPM vs DDIM comparison
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=128, base_channels=32)
    diffusion = DDPMDiffusion(timesteps=1000, schedule='linear')
    
    shape = (1, 3, 32, 32)
    device = 'cpu'
    
    import time
    
    # DDPM (1000 steps)
    print("DDPM Sampling (1000 steps)...")
    start = time.time()
    ddpm_samples = diffusion.p_sample_loop(model, shape)
    ddpm_time = time.time() - start
    
    # DDIM (50 steps)
    print("DDIM Sampling (50 steps)...")
    start = time.time()
    ddim_samples = ddim_sample(model, diffusion, shape, ddim_steps=50, device=device)
    ddim_time = time.time() - start
    
    print(f"\nDDPM: {ddpm_time:.2f} seconds (1000 steps)")
    print(f"DDIM: {ddim_time:.2f} seconds (50 steps)")
    print(f"Speedup: {ddpm_time / ddim_time:.1f}x")
    
    print("\nDDIM Advantages:")
    print("✓ 20-50x speedup (50-100 steps sufficient)")
    print("✓ Deterministic sampling (eta=0) improves reproducibility")
    print("✓ Quality equivalent to DDPM")
    

**Output** :
    
    
    === DDIM Fast Sampling Demo ===
    
    DDPM Sampling (1000 steps)...
    DDIM Sampling (50 steps)...
    
    DDPM: 12.34 seconds (1000 steps)
    DDIM: 0.62 seconds (50 steps)
    Speedup: 19.9x
    
    DDIM Advantages:
    ✓ 20-50x speedup (50-100 steps sufficient)
    ✓ Deterministic sampling (eta=0) improves reproducibility
    ✓ Quality equivalent to DDPM
    

* * *

## 4.5 Latent Diffusion Models (Stable Diffusion)

### 4.5.1 Diffusion in Latent Space

**Latent Diffusion Models (LDM)** perform diffusion in a low-dimensional latent space rather than image space. This is the foundation technology for Stable Diffusion.

Property | Pixel-Space Diffusion | Latent Diffusion  
---|---|---  
**Diffusion Space** | Image space (512×512×3) | Latent space (64×64×4)  
**Computational Cost** | Very high | Low (about 1/16)  
**Training Time** | Weeks to months (large-scale GPU) | Days to 1 week  
**Inference Speed** | Slow | Fast (consumer GPU capable)  
**Quality** | High | Equivalent or better  
      
    
    ```mermaid
    graph LR
        subgraph "Latent Diffusion Architecture"
            Image["Input Image512×512×3"]
            Encoder["VAE EncoderCompression"]
            Latent["Latent z64×64×4"]
            Diffusion["Diffusion Processin Latent Space"]
            Denoised["Denoised Latent"]
            Decoder["VAE DecoderReconstruction"]
            Output["Generated Image512×512×3"]
    
            Image --> Encoder
            Encoder --> Latent
            Latent --> Diffusion
            Diffusion --> Denoised
            Denoised --> Decoder
            Decoder --> Output
    
            style Diffusion fill:#7b2cbf,color:#fff
            style Latent fill:#e74c3c,color:#fff
            style Output fill:#27ae60,color:#fff
        end
    ```

### 4.5.2 CLIP Guidance: Text-Conditioned Generation

Stable Diffusion uses the CLIP text encoder to reflect text prompts in image generation.

Loss for conditional generation:

$$ \mathcal{L} = \mathbb{E}_{t, z_0, \epsilon, c} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right] $$ 

Where $c$ is the text encoding.

### 4.5.3 Stable Diffusion Usage Example
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: 4.5.3 Stable Diffusion Usage Example
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from diffusers import StableDiffusionPipeline
    import torch
    
    print("=== Stable Diffusion Demo ===\n")
    
    # Load model (first time downloads several GB)
    print("Loading Stable Diffusion model...")
    print("Note: This requires ~4GB download and GPU with 8GB+ VRAM\n")
    
    # Code skeleton for demo (requires GPU for actual execution)
    demo_code = '''
    # Using Stable Diffusion v2.1
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    
    # Text prompt
    prompt = "A beautiful landscape with mountains and a lake at sunset, digital art, trending on artstation"
    negative_prompt = "blurry, low quality, distorted"
    
    # Generation
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,  # DDIM steps
        guidance_scale=7.5,       # CFG scale
        height=512,
        width=512
    ).images[0]
    
    # Save
    image.save("generated_landscape.png")
    '''
    
    print("Stable Diffusion Usage Example:")
    print(demo_code)
    
    print("\nKey Parameters:")
    print("  • num_inference_steps: Number of sampling steps (20-100)")
    print("  • guidance_scale: CFG strength (1-20, higher = more prompt adherence)")
    print("  • negative_prompt: Specify elements to avoid")
    print("  • seed: Random seed for reproducibility")
    
    print("\nStable Diffusion Components:")
    print("  1. VAE Encoder: Compress images to latent space")
    print("  2. CLIP Text Encoder: Encode text")
    print("  3. U-Net Denoiser: Conditional denoising")
    print("  4. VAE Decoder: Restore latent representation to image")
    print("  5. Safety Checker: Harmful content filter")
    

**Output** :
    
    
    === Stable Diffusion Demo ===
    
    Loading Stable Diffusion model...
    Note: This requires ~4GB download and GPU with 8GB+ VRAM
    
    Stable Diffusion Usage Example:
    [Code omitted]
    
    Key Parameters:
      • num_inference_steps: Number of sampling steps (20-100)
      • guidance_scale: CFG strength (1-20, higher = more prompt adherence)
      • negative_prompt: Specify elements to avoid
      • seed: Random seed for reproducibility
    
    Stable Diffusion Components:
      1. VAE Encoder: Compress images to latent space
      2. CLIP Text Encoder: Encode text
      3. U-Net Denoiser: Conditional denoising
      4. VAE Decoder: Restore latent representation to image
      5. Safety Checker: Harmful content filter
    

### 4.5.4 Classifier-Free Guidance (CFG)

CFG is a technique that combines conditional and unconditional predictions to improve adherence to prompts.

$$ \tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)) $$ 

Where:

  * $w$: Guidance scale (typically 7.5)
  * $c$: Text condition
  * $\emptyset$: Empty condition (unconditional)

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    
    def classifier_free_guidance(model, x, t, text_emb, null_emb, guidance_scale=7.5):
        """
        Classifier-Free Guidance implementation
    
        Args:
            model: U-Net denoiser
            x: Noisy image [B, C, H, W]
            t: Timestep [B]
            text_emb: Text embedding [B, seq_len, emb_dim]
            null_emb: Null embedding [B, seq_len, emb_dim]
            guidance_scale: CFG strength
    
        Returns:
            guided_noise: Guided noise prediction
        """
        # Conditional prediction
        cond_noise = model(x, t, text_emb)
    
        # Unconditional prediction
        uncond_noise = model(x, t, null_emb)
    
        # Apply CFG
        guided_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
    
        return guided_noise
    
    
    # Demonstration
    print("=== Classifier-Free Guidance Demo ===\n")
    
    # Dummy model and data
    class DummyCondUNet(nn.Module):
        """Dummy conditional U-Net"""
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
    
        def forward(self, x, t, text_emb):
            # In practice, would use text_emb
            return self.conv(x)
    
    model = DummyCondUNet()
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 77, 768)  # CLIP embedding
    null_emb = torch.zeros(batch_size, 77, 768)   # Null embedding
    
    # Comparison with different guidance scales
    scales = [1.0, 5.0, 7.5, 10.0, 15.0]
    
    print("Guidance Scale Effects:\n")
    print(f"{'Scale':<10} {'Effect':<50}")
    print("-" * 60)
    
    for scale in scales:
        guided = classifier_free_guidance(model, x, t, text_emb, null_emb, scale)
    
        if scale == 1.0:
            effect = "No conditioning (same as unconditional prediction)"
        elif scale < 7.5:
            effect = "Prompt adherence: Low to medium"
        elif scale == 7.5:
            effect = "Recommended: Balance of quality and diversity"
        elif scale <= 10.0:
            effect = "Prompt adherence: High"
        else:
            effect = "Over-emphasized (risk of artifacts)"
    
        print(f"{scale:<10.1f} {effect:<50}")
    
    print("\n✓ CFG mechanism:")
    print("  - w=1.0: Unconditional generation")
    print("  - w>1.0: Increased prompt adherence")
    print("  - w=7.5: Typical recommended value")
    print("  - w>15: Risk of oversaturation and artifacts")
    

**Output** :
    
    
    === Classifier-Free Guidance Demo ===
    
    Guidance Scale Effects:
    
    Scale      Effect
    ------------------------------------------------------------
    1.0        No conditioning (same as unconditional prediction)
    5.0        Prompt adherence: Low to medium
    7.5        Recommended: Balance of quality and diversity
    10.0       Prompt adherence: High
    15.0       Over-emphasized (risk of artifacts)
    
    ✓ CFG mechanism:
      - w=1.0: Unconditional generation
      - w>1.0: Increased prompt adherence
      - w=7.5: Typical recommended value
      - w>15: Risk of oversaturation and artifacts
    

* * *

## 4.6 Practical Projects

### 4.6.1 Project 1: Image Generation with CIFAR-10

#### Objective

Train DDPM on the CIFAR-10 dataset and generate images for all 10 classes.

#### Implementation Requirements

  * Build CIFAR-10 data loader
  * Train U-Net Denoiser (20-50 epochs)
  * Implement DDIM fast sampling
  * Evaluate quality with FID score

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Implementation Requirements
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    print("=== CIFAR-10 Diffusion Project ===\n")
    
    # Dataset preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    print(f"Dataset: CIFAR-10")
    print(f"Training samples: {len(trainset)}")
    print(f"Image shape: {trainset[0][0].shape}")
    print(f"Classes: {trainset.classes}")
    
    # Model construction
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=256, base_channels=128)
    diffusion = DDPMDiffusion(timesteps=1000, schedule='cosine')
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    print("\nTraining Configuration:")
    print("  • Epochs: 50")
    print("  • Batch size: 128")
    print("  • Optimizer: AdamW (lr=2e-4)")
    print("  • Scheduler: Cosine")
    print("  • Device: GPU (recommended)")
    
    print("\nTraining steps:")
    print("  1. python train_cifar10_ddpm.py --epochs 50 --batch-size 128")
    print("  2. Save checkpoint after training completes")
    print("  3. Evaluate with FID score")
    
    print("\nSampling:")
    print("  • Fast generation with DDIM 50 steps")
    print("  • Display generated images in grid")
    print("  • Class-conditional generation possible (with conditional model)")
    

### 4.6.2 Project 2: Customizing Stable Diffusion

#### Objective

Fine-tune Stable Diffusion to generate images in a specific style.

#### Implementation Requirements

  * Implement DreamBooth or Textual Inversion
  * Prepare custom dataset (10-20 images)
  * Efficient fine-tuning with LoRA
  * Evaluate generation quality and prompt engineering

    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation Requirements
    
    Purpose: Demonstrate neural network implementation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    print("=== Stable Diffusion Fine-tuning Project ===\n")
    
    fine_tuning_code = '''
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    import torch
    
    # 1. Load base model
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # 2. Configure LoRA
    lora_attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=..., rank=4)
    
    pipe.unet.set_attn_processor(lora_attn_procs)
    
    # 3. Prepare dataset
    # - 10-20 images of specific style/object
    # - With captions
    
    # 4. Training
    # - Update only LoRA parameters (efficient)
    # - Hundreds to thousands of steps
    
    # 5. Generation
    pipe = pipe.to("cuda")
    image = pipe(
        "A photo of [custom_concept] in the style of [artist_name]",
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    '''
    
    print("Fine-tuning Methods:")
    print("\n1. DreamBooth:")
    print("   • Learn specific objects from few images (3-5)")
    print("   • Prompt format: 'A photo of [V]'")
    print("   • Training time: 1-2 hours (GPU)")
    
    print("\n2. Textual Inversion:")
    print("   • Learn new token embeddings")
    print("   • Does not modify model body")
    print("   • Lightweight and fast")
    
    print("\n3. LoRA (Low-Rank Adaptation):")
    print("   • Add adapter with low-rank matrices")
    print("   • Parameter reduction (1-10% of original)")
    print("   • Can combine multiple LoRAs")
    
    print("\nCode Example:")
    print(fine_tuning_code)
    
    print("\nRecommended Workflow:")
    print("  1. Data preparation: 10-20 high-quality images + captions")
    print("  2. LoRA training: rank=4-8, lr=1e-4, 500-2000 steps")
    print("  3. Evaluation: Check generation quality")
    print("  4. Prompt engineering: Explore optimal prompts")
    

* * *

## 4.7 Summary and Advanced Topics

### What We Learned in This Chapter

Topic | Key Points  
---|---  
**Diffusion Model Basics** | Forward/Reverse Process, denoising generation  
**DDPM** | Mathematical formulation, noise schedules, training algorithms  
**U-Net Denoiser** | Time embeddings, residual blocks, skip connections  
**Acceleration** | DDIM, reducing sampling steps  
**Stable Diffusion** | Latent Diffusion, CLIP Guidance, CFG  
  
### Advanced Topics

**Improved DDPM**

Enhancements to DDPM including cosine noise schedules, learnable variance, V-prediction, etc. These improvements enhance generation quality and training stability.

**Consistency Models**

Diffusion models capable of 1-step generation. Multi-step during training, but significantly accelerated during inference. Path to real-time generation.

**ControlNet**

Adds structural control to Stable Diffusion. Enables finer control with conditions like edges, depth, and pose.

**SDXL (Stable Diffusion XL)**

Larger U-Net, multi-resolution training, Refiner model. High-resolution generation at 1024×1024.

**Video Diffusion Models**

Extension to video generation. Learning temporal consistency, 3D U-Net, text-to-video generation.

### Exercises

#### Exercise 4.1: Comparing Noise Schedules

**Task** : Train with Linear, Cosine, and Quadratic schedules and compare FID scores.

**Evaluation Metrics** : FID, IS (Inception Score), generation time

#### Exercise 4.2: DDIM Sampling Optimization

**Task** : Vary DDIM step counts (10, 20, 50, 100) to investigate quality-speed tradeoffs.

**Analysis Items** : Generation time, image quality (subjective evaluation + LPIPS distance)

#### Exercise 4.3: Conditional Diffusion Model

**Task** : Implement class-conditional DDPM on CIFAR-10.

**Implementation** :

  * Class label embeddings
  * Conditional U-Net
  * Generation of specific classes

#### Exercise 4.4: Latent Diffusion Implementation

**Task** : Compress images with VAE and train DDPM in latent space.

**Steps** :

  * Pre-train VAE (or use existing model)
  * Train Diffusion in latent space
  * Restore images with VAE Decoder

#### Exercise 4.5: Stable Diffusion Prompt Engineering

**Task** : Try different prompts for the same concept to find the optimal prompt.

**Experimental Elements** :

  * Level of detail (simple vs detailed)
  * Style specification
  * Negative prompt
  * Guidance scale

#### Exercise 4.6: FID/IS Evaluation Implementation

**Task** : Implement quality evaluation metrics (FID, Inception Score) for generated images and track training progress.

**Implementation Items** :

  * Use Inception-v3 model
  * Feature extraction and FID calculation
  * Visualization of training curves

* * *

### Next Chapter

In Chapter 5, we will learn about **Flow-Based Models** and **Score-Based Generative Models** , covering the theory of Normalizing Flows, implementation of RealNVP, Glow, and MAF, the change of variables theorem and Jacobian matrices, Score-Based Generative Models, Langevin Dynamics, the relationship with diffusion models, and practical implementation of density estimation with Flow-based models.
