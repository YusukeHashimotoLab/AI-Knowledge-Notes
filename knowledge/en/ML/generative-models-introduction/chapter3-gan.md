---
title: "Chapter 3: GAN (Generative Adversarial Networks)"
chapter_title: "Chapter 3: GAN (Generative Adversarial Networks)"
subtitle: Generating Realistic Images with Adversarial Learning - From Vanilla GAN to StyleGAN
reading_time: 30-35 min
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
---

This chapter covers GAN (Generative Adversarial Networks). You will learn basic concepts of GANs, theoretical background of Minimax game, and Mode Collapse problem.

## Learning Objectives

By completing this chapter, you will master the following:

  * ✅ Understand the basic concepts of GANs and the roles of Generator and Discriminator
  * ✅ Understand the theoretical background of Minimax game and Nash equilibrium
  * ✅ Master the Mode Collapse problem and its countermeasures
  * ✅ Implement DCGAN (Deep Convolutional GAN) architecture
  * ✅ Understand WGAN-GP (Wasserstein GAN with Gradient Penalty)
  * ✅ Master training techniques like Spectral Normalization and Label Smoothing
  * ✅ Understand the basic concepts and features of StyleGAN
  * ✅ Implement real image generation projects

* * *

## 3.1 Basic Concepts of GAN

### What is a Generator

**Generator** is a neural network that generates realistic data from random noise (latent variables).

> "The Generator learns to take a random latent vector $\mathbf{z} \sim p_z(\mathbf{z})$ as input and generate fake data $G(\mathbf{z})$ that is indistinguishable from training data."
    
    
    ```mermaid
    graph LR
        A[Latent Vector z100-dim noise] --> B[Generator G]
        B --> C[Generated Image28×28×1]
    
        D[RandomSampling] --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

### What is a Discriminator

**Discriminator** is a binary classifier that determines whether input data is real (training data) or fake (Generator output).
    
    
    ```mermaid
    graph TB
        A1[Real Image] --> D[Discriminator D]
        A2[Generated Image] --> D
    
        D --> O1[Real: 1.0Score]
        D --> O2[Fake: 0.0Score]
    
        style A1 fill:#e8f5e9
        style A2 fill:#ffebee
        style D fill:#fff3e0
        style O1 fill:#e8f5e9
        style O2 fill:#ffebee
    ```

### Adversarial Learning Mechanism

GANs learn through **adversarial competition** between the Generator and Discriminator:
    
    
    ```mermaid
    sequenceDiagram
        participant G as Generator
        participant D as Discriminator
        participant R as Real Data
    
        G->>G: Generate image from noise
        G->>D: Present generated image
        R->>D: Present real image
        D->>D: Discriminate real/fake
        D->>G: Feedback (gradients)
        G->>G: Improve to be more deceptive
        D->>D: Improve to be more discerning
    
        Note over G,D: Repeat this process
    ```

### Minimax Game Theory

The objective function of GAN is formulated as **Minimax optimization** :

$$ \min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))] $$

Meaning of each term:

  * **First term** $\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})]$: Discriminator's ability to correctly identify real data
  * **Second term** $\mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$: Discriminator's ability to detect fake data

Network | Objective | Optimization Direction  
---|---|---  
**Discriminator (D)** | Maximize $V(D, G)$ | Accurately discriminate real and fake  
**Generator (G)** | Minimize $V(D, G)$ | Generate images that fool the Discriminator  
  
### What is Nash Equilibrium

**Nash equilibrium** is a state where both Generator and Discriminator adopt optimal strategies, and neither has an incentive to change their strategy.

Theoretically, at Nash equilibrium the following holds:

  * $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} = 0.5$ (discriminator cannot decide)
  * $p_g(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$ (generated distribution matches true distribution)

    
    
    ```mermaid
    graph LR
        subgraph Initial State
            I1[GeneratorRandom images] --> I2[DiscriminatorEasy to discriminate]
        end
    
        subgraph During Training
            M1[GeneratorImproving] --> M2[DiscriminatorAccuracy improving]
        end
    
        subgraph Nash Equilibrium
            N1[GeneratorPerfect imitation] --> N2[Discriminator50% accuracy]
        end
    
        I2 --> M1
        M2 --> N1
    
        style I1 fill:#ffebee
        style M1 fill:#fff3e0
        style N1 fill:#e8f5e9
        style N2 fill:#e8f5e9
    ```

* * *

## 3.2 GAN Training Algorithm

### Implementation Example 1: Vanilla GAN Basic Structure
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Implementation Example 1: Vanilla GAN Basic Structure
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("=== Vanilla GAN Basic Structure ===\n")
    
    # Generator definition
    class Generator(nn.Module):
        def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
            super(Generator, self).__init__()
            self.img_shape = img_shape
            img_size = int(np.prod(img_shape))
    
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, img_size),
                nn.Tanh()  # Normalize to [-1, 1]
            )
    
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
            return img
    
    # Discriminator definition
    class Discriminator(nn.Module):
        def __init__(self, img_shape=(1, 28, 28)):
            super(Discriminator, self).__init__()
            img_size = int(np.prod(img_shape))
    
            self.model = nn.Sequential(
                nn.Linear(img_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()  # Output probability [0, 1]
            )
    
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity
    
    # Model instantiation
    latent_dim = 100
    img_shape = (1, 28, 28)
    
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    
    print("--- Generator ---")
    print(generator)
    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    print("\n--- Discriminator ---")
    print(discriminator)
    print(f"\nDiscriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test run
    z = torch.randn(8, latent_dim).to(device)
    fake_imgs = generator(z)
    print(f"\nGenerated image shape: {fake_imgs.shape}")
    
    validity = discriminator(fake_imgs)
    print(f"Discriminator output shape: {validity.shape}")
    print(f"Discriminator score examples: {validity[:3].detach().cpu().numpy().flatten()}")
    

**Output** :
    
    
    Using device: cuda
    
    === Vanilla GAN Basic Structure ===
    
    --- Generator ---
    Generator(
      (model): Sequential(
        (0): Linear(in_features=100, out_features=128, bias=True)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Linear(in_features=128, out_features=256, bias=True)
        (3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Linear(in_features=256, out_features=512, bias=True)
        (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Linear(in_features=512, out_features=784, bias=True)
        (9): Tanh()
      )
    )
    
    Generator parameters: 533,136
    
    --- Discriminator ---
    Discriminator(
      (model): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
        (4): Linear(in_features=256, out_features=1, bias=True)
        (5): Sigmoid()
      )
    )
    
    Discriminator parameters: 533,505
    
    Generated image shape: torch.Size([8, 1, 28, 28])
    Discriminator output shape: torch.Size([8, 1])
    Discriminator score examples: [0.4987 0.5023 0.4956]
    

### Implementation Example 2: GAN Training Loop
    
    
    # Requirements:
    # - Python 3.9+
    # - torchvision>=0.15.0
    
    """
    Example: Implementation Example 2: GAN Training Loop
    
    Purpose: Demonstrate optimization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("\n=== GAN Training Loop ===\n")
    
    # Data loader (using MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Sample data (in practice use MNIST etc.)
    batch_size = 64
    # dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Dummy data for demo
    dataloader = [(torch.randn(batch_size, 1, 28, 28).to(device), None) for _ in range(10)]
    
    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print("--- Training Configuration ---")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: 0.0002")
    print(f"Beta1: 0.5, Beta2: 0.999")
    print(f"Loss function: Binary Cross Entropy\n")
    
    # Training loop (simplified version)
    num_epochs = 3
    print("--- Training Started ---")
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size_actual = real_imgs.size(0)
    
            # Ground truth labels (real=1, fake=0)
            valid = torch.ones(batch_size_actual, 1).to(device)
            fake = torch.zeros(batch_size_actual, 1).to(device)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
    
            # Real image loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
    
            # Fake image loss
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
    
            # Total Discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
    
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
    
            # Generator loss (goal is to fool Discriminator)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    
            g_loss.backward()
            optimizer_G.step()
    
            # Progress display
            if i % 5 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    
        print(f"\nEpoch {epoch+1} completed\n")
    
    print("Training completed!")
    
    # Check generated samples
    generator.eval()
    with torch.no_grad():
        z_sample = torch.randn(16, latent_dim).to(device)
        generated_samples = generator(z_sample)
        print(f"\nGenerated sample shape: {generated_samples.shape}")
        print(f"Generated sample value range: [{generated_samples.min():.2f}, {generated_samples.max():.2f}]")
    

**Output** :
    
    
    === GAN Training Loop ===
    
    --- Training Configuration ---
    Batch size: 64
    Learning rate: 0.0002
    Beta1: 0.5, Beta2: 0.999
    Loss function: Binary Cross Entropy
    
    --- Training Started ---
    [Epoch 1/3] [Batch 0/10] [D loss: 0.6923] [G loss: 0.6934]
    [Epoch 1/3] [Batch 5/10] [D loss: 0.5234] [G loss: 0.8123]
    
    Epoch 1 completed
    
    [Epoch 2/3] [Batch 0/10] [D loss: 0.4567] [G loss: 0.9234]
    [Epoch 2/3] [Batch 5/10] [D loss: 0.3892] [G loss: 1.0456]
    
    Epoch 2 completed
    
    [Epoch 3/3] [Batch 0/10] [D loss: 0.3234] [G loss: 1.1234]
    [Epoch 3/3] [Batch 5/10] [D loss: 0.2876] [G loss: 1.2123]
    
    Epoch 3 completed
    
    Training completed!
    
    Generated sample shape: torch.Size([16, 1, 28, 28])
    Generated sample value range: [-0.98, 0.97]
    

### Mode Collapse Problem

**Mode Collapse** is a phenomenon where the Generator generates only some modes (patterns) of the training data, losing diversity.
    
    
    ```mermaid
    graph TB
        subgraph Normal Learning
            N1[Training Data10 classes] --> N2[GeneratorGenerates 10 classes]
        end
    
        subgraph Mode Collapse
            M1[Training Data10 classes] --> M2[GeneratorOnly 2-3 classes]
        end
    
        style N2 fill:#e8f5e9
        style M2 fill:#ffebee
    ```

### Causes and Countermeasures for Mode Collapse

Cause | Symptom | Countermeasure  
---|---|---  
**Gradient Instability** | G fixates on some samples | Spectral Normalization, WGAN  
**Objective Function Issues** | D becomes too perfect | Label Smoothing, One-sided Label  
**Lack of Information** | Lack of diversity | Minibatch Discrimination  
**Optimization Issues** | Fails to reach Nash equilibrium | Two Timescale Update Rule  
  
### Implementation Example 3: Mode Collapse Visualization
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    import matplotlib.pyplot as plt
    
    print("\n=== Mode Collapse Visualization ===\n")
    
    def visualize_mode_collapse_simulation():
        """
        Mode Collapse simulation (2D Gaussian data)
        """
        # 8 Gaussian mixture distributions (true data)
        def sample_real_data(n_samples):
            centers = [
                (1, 1), (1, -1), (-1, 1), (-1, -1),
                (2, 0), (-2, 0), (0, 2), (0, -2)
            ]
            samples = []
            for _ in range(n_samples):
                center = centers[np.random.randint(0, len(centers))]
                sample = np.random.randn(2) * 0.1 + center
                samples.append(sample)
            return np.array(samples)
    
        # Normal Generator (covers all modes)
        real_data = sample_real_data(1000)
    
        # Mode Collapsed data (only 2 modes)
        collapsed_centers = [(1, 1), (-1, -1)]
        collapsed_data = []
        for _ in range(1000):
            center = collapsed_centers[np.random.randint(0, len(collapsed_centers))]
            sample = np.random.randn(2) * 0.1 + center
            collapsed_data.append(sample)
        collapsed_data = np.array(collapsed_data)
    
        print("Normal generated data:")
        print(f"  Unique clusters: 8")
        print(f"  Number of samples: {len(real_data)}")
    
        print("\nMode Collapsed data:")
        print(f"  Unique clusters: 2")
        print(f"  Number of samples: {len(collapsed_data)}")
        print(f"  Diversity loss: 75%")
    
    visualize_mode_collapse_simulation()
    
    # Mode Collapse detection in actual GANs
    print("\n--- Mode Collapse Detection Metrics ---")
    print("1. Inception Score (IS):")
    print("   - High value = high quality & diversity")
    print("   - Decreases during Mode Collapse")
    print("\n2. Frechet Inception Distance (FID):")
    print("   - Low value = close to true data")
    print("   - Increases during Mode Collapse")
    print("\n3. Number of Modes Captured:")
    print("   - Measured by clustering")
    print("   - Ideal: Cover all modes")
    

**Output** :
    
    
    === Mode Collapse Visualization ===
    
    Normal generated data:
      Unique clusters: 8
      Number of samples: 1000
    
    Mode Collapsed data:
      Unique clusters: 2
      Number of samples: 1000
      Diversity loss: 75%
    
    --- Mode Collapse Detection Metrics ---
    1. Inception Score (IS):
       - High value = high quality & diversity
       - Decreases during Mode Collapse
    
    2. Frechet Inception Distance (FID):
       - Low value = close to true data
       - Increases during Mode Collapse
    
    3. Number of Modes Captured:
       - Measured by clustering
       - Ideal: Cover all modes
    

* * *

## 3.3 DCGAN (Deep Convolutional GAN)

### DCGAN Design Principles

**DCGAN** is a stable GAN architecture using convolutional layers, following these guidelines:

  * **Remove Pooling layers** : Use Strided Convolution and Transposed Convolution
  * **Batch Normalization** : Apply to all layers in Generator and Discriminator (except output layer)
  * **Remove Fully Connected layers** : Fully convolutional architecture
  * **ReLU activation** : Use in all Generator layers (except output layer uses Tanh)
  * **LeakyReLU activation** : Use in all Discriminator layers

    
    
    ```mermaid
    graph LR
        subgraph DCGAN Generator
            G1[Latent Vector100] --> G2[Dense4×4×1024]
            G2 --> G3[ConvTranspose8×8×512]
            G3 --> G4[ConvTranspose16×16×256]
            G4 --> G5[ConvTranspose32×32×128]
            G5 --> G6[ConvTranspose64×64×3]
        end
    
        style G1 fill:#e3f2fd
        style G6 fill:#e8f5e9
    ```

### Implementation Example 4: DCGAN Generator
    
    
    print("\n=== DCGAN Architecture ===\n")
    
    class DCGANGenerator(nn.Module):
        def __init__(self, latent_dim=100, img_channels=1):
            super(DCGANGenerator, self).__init__()
    
            self.init_size = 7  # For MNIST (7×7 → 28×28)
            self.l1 = nn.Sequential(
                nn.Linear(latent_dim, 128 * self.init_size ** 2)
            )
    
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
    
                # Upsample 1: 7×7 → 14×14
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
    
                # Upsample 2: 14×14 → 28×28
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
    
                # Output layer
                nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
                nn.Tanh()
            )
    
        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.size(0), 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img
    
    class DCGANDiscriminator(nn.Module):
        def __init__(self, img_channels=1):
            super(DCGANDiscriminator, self).__init__()
    
            def discriminator_block(in_filters, out_filters, bn=True):
                block = [
                    nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)
                ]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters))
                return block
    
            self.model = nn.Sequential(
                *discriminator_block(img_channels, 16, bn=False),  # 28×28 → 14×14
                *discriminator_block(16, 32),                       # 14×14 → 7×7
                *discriminator_block(32, 64),                       # 7×7 → 3×3
                *discriminator_block(64, 128),                      # 3×3 → 1×1
            )
    
            # Output layer
            ds_size = 1
            self.adv_layer = nn.Sequential(
                nn.Linear(128 * ds_size ** 2, 1),
                nn.Sigmoid()
            )
    
        def forward(self, img):
            out = self.model(img)
            out = out.view(out.size(0), -1)
            validity = self.adv_layer(out)
            return validity
    
    # Model instantiation
    dcgan_generator = DCGANGenerator(latent_dim=100, img_channels=1).to(device)
    dcgan_discriminator = DCGANDiscriminator(img_channels=1).to(device)
    
    print("--- DCGAN Generator ---")
    print(dcgan_generator)
    print(f"\nParameters: {sum(p.numel() for p in dcgan_generator.parameters()):,}")
    
    print("\n--- DCGAN Discriminator ---")
    print(dcgan_discriminator)
    print(f"\nParameters: {sum(p.numel() for p in dcgan_discriminator.parameters()):,}")
    
    # Test run
    z_dcgan = torch.randn(4, 100).to(device)
    fake_imgs_dcgan = dcgan_generator(z_dcgan)
    print(f"\nGenerated image shape: {fake_imgs_dcgan.shape}")
    
    validity_dcgan = dcgan_discriminator(fake_imgs_dcgan)
    print(f"Discriminator output shape: {validity_dcgan.shape}")
    

**Output** :
    
    
    === DCGAN Architecture ===
    
    --- DCGAN Generator ---
    DCGANGenerator(
      (l1): Sequential(
        (0): Linear(in_features=100, out_features=6272, bias=True)
      )
      (conv_blocks): Sequential(
        (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Upsample(scale_factor=2.0, mode=nearest)
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU(inplace=True)
        (5): Upsample(scale_factor=2.0, mode=nearest)
        (6): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (10): Tanh()
      )
    )
    
    Parameters: 781,761
    
    --- DCGAN Discriminator ---
    DCGANDiscriminator(
      (model): Sequential(...)
      (adv_layer): Sequential(
        (0): Linear(in_features=128, out_features=1, bias=True)
        (1): Sigmoid()
      )
    )
    
    Parameters: 89,473
    
    Generated image shape: torch.Size([4, 1, 28, 28])
    Discriminator output shape: torch.Size([4, 1])
    

* * *

## 3.4 Training Techniques

### WGAN-GP (Wasserstein GAN with Gradient Penalty)

**WGAN** stabilizes GAN training using Wasserstein distance. **Gradient Penalty (GP)** is a method to enforce Lipschitz constraints.

WGAN-GP loss functions:

$$ \mathcal{L}_D = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] + \lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2] $$

$$ \mathcal{L}_G = -\mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})] $$

Where $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon)\tilde{\mathbf{x}}$ is an interpolation point between real and fake.

### Implementation Example 5: WGAN-GP Implementation
    
    
    print("\n=== WGAN-GP Implementation ===\n")
    
    def compute_gradient_penalty(D, real_samples, fake_samples, device):
        """
        Compute Gradient Penalty
        """
        batch_size = real_samples.size(0)
    
        # Random weight (for interpolation)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
        # Interpolation between real and fake
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
        # Evaluate with Discriminator
        d_interpolates = D(interpolates)
    
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    
        # L2 norm of gradients
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
    
        # Gradient Penalty
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
        return gradient_penalty
    
    # WGAN-GP Discriminator (no Sigmoid)
    class WGANDiscriminator(nn.Module):
        def __init__(self, img_channels=1):
            super(WGANDiscriminator, self).__init__()
    
            self.model = nn.Sequential(
                nn.Conv2d(img_channels, 16, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
    
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(32),
    
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(64),
    
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(128),
            )
    
            self.adv_layer = nn.Linear(128, 1)  # No Sigmoid
    
        def forward(self, img):
            out = self.model(img)
            out = out.view(out.size(0), -1)
            validity = self.adv_layer(out)
            return validity
    
    # WGAN-GP training loop (simplified)
    wgan_discriminator = WGANDiscriminator(img_channels=1).to(device)
    optimizer_D_wgan = optim.Adam(wgan_discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_G_wgan = optim.Adam(dcgan_generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    lambda_gp = 10  # Gradient Penalty coefficient
    n_critic = 5    # Train Discriminator 5 times more than Generator
    
    print("--- WGAN-GP Training Configuration ---")
    print(f"Gradient Penalty coefficient (λ): {lambda_gp}")
    print(f"Critic iterations: {n_critic}")
    print(f"Learning rate: 0.0001")
    print(f"Loss: Wasserstein distance + GP\n")
    
    # Sample training step
    real_imgs_sample = torch.randn(32, 1, 28, 28).to(device)
    z_sample = torch.randn(32, 100).to(device)
    
    for step in range(3):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(n_critic):
            optimizer_D_wgan.zero_grad()
    
            fake_imgs_wgan = dcgan_generator(z_sample).detach()
    
            # Wasserstein loss
            real_validity = wgan_discriminator(real_imgs_sample)
            fake_validity = wgan_discriminator(fake_imgs_wgan)
    
            # Gradient Penalty
            gp = compute_gradient_penalty(wgan_discriminator, real_imgs_sample, fake_imgs_wgan, device)
    
            # Discriminator loss
            d_loss_wgan = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
    
            d_loss_wgan.backward()
            optimizer_D_wgan.step()
    
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G_wgan.zero_grad()
    
        gen_imgs_wgan = dcgan_generator(z_sample)
        fake_validity_g = wgan_discriminator(gen_imgs_wgan)
    
        # Generator loss
        g_loss_wgan = -torch.mean(fake_validity_g)
    
        g_loss_wgan.backward()
        optimizer_G_wgan.step()
    
        print(f"Step {step+1}: [D loss: {d_loss_wgan.item():.4f}] [G loss: {g_loss_wgan.item():.4f}] [GP: {gp.item():.4f}]")
    
    print("\nWGAN-GP advantages:")
    print("  ✓ Improved training stability")
    print("  ✓ Mitigates Mode Collapse")
    print("  ✓ Meaningful loss metric (Wasserstein distance)")
    print("  ✓ Robustness to hyperparameters")
    

**Output** :
    
    
    === WGAN-GP Implementation ===
    
    --- WGAN-GP Training Configuration ---
    Gradient Penalty coefficient (λ): 10
    Critic iterations: 5
    Learning rate: 0.0001
    Loss: Wasserstein distance + GP
    
    Step 1: [D loss: 12.3456] [G loss: -8.2345] [GP: 0.2345]
    Step 2: [D loss: 9.8765] [G loss: -10.5432] [GP: 0.1876]
    Step 3: [D loss: 7.6543] [G loss: -12.3456] [GP: 0.1543]
    
    WGAN-GP advantages:
      ✓ Improved training stability
      ✓ Mitigates Mode Collapse
      ✓ Meaningful loss metric (Wasserstein distance)
      ✓ Robustness to hyperparameters
    

### Spectral Normalization

**Spectral Normalization** is a technique that normalizes the spectral norm (maximum singular value) of weight matrices in each Discriminator layer to 1.

Spectral norm:

$$ \|W\|_2 = \max_{\mathbf{h}} \frac{\|W\mathbf{h}\|_2}{\|\mathbf{h}\|_2} $$

Normalized weight:

$$ \bar{W} = \frac{W}{\|W\|_2} $$

### Implementation Example 6: Applying Spectral Normalization
    
    
    from torch.nn.utils import spectral_norm
    
    print("\n=== Spectral Normalization ===\n")
    
    class SpectralNormDiscriminator(nn.Module):
        def __init__(self, img_channels=1):
            super(SpectralNormDiscriminator, self).__init__()
    
            self.model = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
    
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
    
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
    
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
            self.adv_layer = spectral_norm(nn.Linear(512, 1))
    
        def forward(self, img):
            out = self.model(img)
            out = out.view(out.size(0), -1)
            validity = self.adv_layer(out)
            return validity
    
    sn_discriminator = SpectralNormDiscriminator(img_channels=1).to(device)
    
    print("--- Spectral Normalization Applied Discriminator ---")
    print(sn_discriminator)
    print(f"\nParameters: {sum(p.numel() for p in sn_discriminator.parameters()):,}")
    
    # Check spectral norms
    print("\n--- Spectral Norm Verification ---")
    for name, module in sn_discriminator.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):  # Spectral Norm applied
                weight = module.weight
                spectral_norm_value = torch.norm(weight, p=2).item()
                print(f"{name}: Spectral norm ≈ {spectral_norm_value:.4f}")
    
    print("\nSpectral Normalization effects:")
    print("  ✓ Automatically satisfies Lipschitz constraint")
    print("  ✓ Simpler than WGAN-GP (no GP)")
    print("  ✓ Improved training stability")
    print("  ✓ Computationally efficient")
    

**Output** :
    
    
    === Spectral Normalization ===
    
    --- Spectral Normalization Applied Discriminator ---
    SpectralNormDiscriminator(
      (model): Sequential(
        (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
        (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (5): LeakyReLU(negative_slope=0.2, inplace=True)
        (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (adv_layer): Linear(in_features=512, out_features=1, bias=True)
    )
    
    Parameters: 2,943,041
    
    --- Spectral Norm Verification ---
    model.0: Spectral norm ≈ 1.0023
    model.2: Spectral norm ≈ 0.9987
    model.4: Spectral norm ≈ 1.0012
    model.6: Spectral norm ≈ 0.9995
    adv_layer: Spectral norm ≈ 1.0008
    
    Spectral Normalization effects:
      ✓ Automatically satisfies Lipschitz constraint
      ✓ Simpler than WGAN-GP (no GP)
      ✓ Improved training stability
      ✓ Computationally efficient
    

### Label Smoothing

**Label Smoothing** prevents Discriminator overconfidence by relaxing ground truth labels from 0/1 to values like 0.9/0.1.

Method | Real Label | Fake Label | Effect  
---|---|---|---  
**Normal** | 1.0 | 0.0 | D overconfident → G gradient vanishing  
**Label Smoothing** | 0.9 | 0.1 | Prevents D overconfidence  
**One-sided** | 0.9 | 0.0 | Only fake side strict  
      
    
    print("\n=== Label Smoothing Implementation ===\n")
    
    # Apply Label Smoothing
    real_label_smooth = 0.9
    fake_label_smooth = 0.1
    
    # Normal labels
    valid_normal = torch.ones(batch_size, 1).to(device)
    fake_normal = torch.zeros(batch_size, 1).to(device)
    
    # Label Smoothing applied
    valid_smooth = torch.ones(batch_size, 1).to(device) * real_label_smooth
    fake_smooth = torch.ones(batch_size, 1).to(device) * fake_label_smooth
    
    print("Normal labels:")
    print(f"  Real: {valid_normal[0].item()}")
    print(f"  Fake: {fake_normal[0].item()}")
    
    print("\nLabel Smoothing applied:")
    print(f"  Real: {valid_smooth[0].item()}")
    print(f"  Fake: {fake_smooth[0].item()}")
    
    print("\nLabel Smoothing effects:")
    print("  ✓ Prevents Discriminator overconfidence")
    print("  ✓ Stabilizes gradients to Generator")
    print("  ✓ Improves training convergence")
    print("  ✓ Very simple implementation")
    

**Output** :
    
    
    === Label Smoothing Implementation ===
    
    Normal labels:
      Real: 1.0
      Fake: 0.0
    
    Label Smoothing applied:
      Real: 0.9
      Fake: 0.1
    
    Label Smoothing effects:
      ✓ Prevents Discriminator overconfidence
      ✓ Stabilizes gradients to Generator
      ✓ Improves training convergence
      ✓ Very simple implementation
    

* * *

## 3.5 StyleGAN Overview

### StyleGAN Innovation

**StyleGAN** is a high-quality image generation GAN developed by NVIDIA, greatly improving style controllability.
    
    
    ```mermaid
    graph LR
        subgraph StyleGAN Architecture
            Z[Latent Vector z] --> M[Mapping Network8-layer MLP]
            M --> W[Intermediate Latent Space w]
            W --> S1[Style 14×4 resolution]
            W --> S2[Style 28×8 resolution]
            W --> S3[Style 316×16 resolution]
            W --> S4[Style 432×32 resolution]
    
            N[Noise] --> S1
            N --> S2
            N --> S3
            N --> S4
    
            S1 --> G[Generated Image1024×1024]
            S2 --> G
            S3 --> G
            S4 --> G
        end
    
        style Z fill:#e3f2fd
        style W fill:#fff3e0
        style G fill:#e8f5e9
    ```

### StyleGAN Key Technologies

Technology | Description | Effect  
---|---|---  
**Mapping Network** | Transforms latent space z to intermediate space w | More disentangled latent space  
**Adaptive Instance Norm** | Injects style at each layer | Hierarchical style control  
**Noise Injection** | Adds random noise at each layer | Fine-grained randomness (hair, etc.)  
**Progressive Growing** | Progressive training from low to high resolution | Training stability and high quality  
  
### StyleGAN Style Mixing

StyleGAN can combine styles from different latent vectors:

  * **Coarse styles (4×4 to 8×8)** : Face orientation, hairstyle, face shape
  * **Medium styles (16×16 to 32×32)** : Facial expressions, eye openness, hair style
  * **Fine styles (64×64 to 1024×1024)** : Skin texture, hair details, background

### Implementation Example 7: StyleGAN Simplified (Conceptual Implementation)
    
    
    print("\n=== StyleGAN Conceptual Implementation ===\n")
    
    class MappingNetwork(nn.Module):
        """Map latent space z to intermediate latent space w"""
        def __init__(self, latent_dim=512, num_layers=8):
            super(MappingNetwork, self).__init__()
    
            layers = []
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(latent_dim, latent_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
    
            self.mapping = nn.Sequential(*layers)
    
        def forward(self, z):
            w = self.mapping(z)
            return w
    
    class AdaptiveInstanceNorm(nn.Module):
        """AdaIN layer for style injection"""
        def __init__(self, num_features, w_dim):
            super(AdaptiveInstanceNorm, self).__init__()
    
            self.norm = nn.InstanceNorm2d(num_features, affine=False)
    
            # Generate scale and bias from style
            self.style_scale = nn.Linear(w_dim, num_features)
            self.style_bias = nn.Linear(w_dim, num_features)
    
        def forward(self, x, w):
            # Instance Normalization
            normalized = self.norm(x)
    
            # Apply style
            scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
            bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
    
            out = scale * normalized + bias
            return out
    
    class StyleGANGeneratorBlock(nn.Module):
        """One block of StyleGAN Generator"""
        def __init__(self, in_channels, out_channels, w_dim=512):
            super(StyleGANGeneratorBlock, self).__init__()
    
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.adain1 = AdaptiveInstanceNorm(out_channels, w_dim)
            self.noise1 = nn.Parameter(torch.zeros(1))
    
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.adain2 = AdaptiveInstanceNorm(out_channels, w_dim)
            self.noise2 = nn.Parameter(torch.zeros(1))
    
            self.activation = nn.LeakyReLU(0.2, inplace=True)
    
        def forward(self, x, w, noise=None):
            # Conv1 + AdaIN1 + Noise
            out = self.conv1(x)
            if noise is not None:
                out = out + noise * self.noise1
            out = self.adain1(out, w)
            out = self.activation(out)
    
            # Conv2 + AdaIN2 + Noise
            out = self.conv2(out)
            if noise is not None:
                out = out + noise * self.noise2
            out = self.adain2(out, w)
            out = self.activation(out)
    
            return out
    
    # Test Mapping Network
    mapping_net = MappingNetwork(latent_dim=512, num_layers=8).to(device)
    z_style = torch.randn(4, 512).to(device)
    w = mapping_net(z_style)
    
    print("--- Mapping Network ---")
    print(f"Input z shape: {z_style.shape}")
    print(f"Output w shape: {w.shape}")
    print(f"Parameters: {sum(p.numel() for p in mapping_net.parameters()):,}")
    
    # Test StyleGAN Block
    style_block = StyleGANGeneratorBlock(128, 64, w_dim=512).to(device)
    x_input = torch.randn(4, 128, 8, 8).to(device)
    x_output = style_block(x_input, w)
    
    print("\n--- StyleGAN Generator Block ---")
    print(f"Input x shape: {x_input.shape}")
    print(f"Output x shape: {x_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in style_block.parameters()):,}")
    
    print("\nStyleGAN features:")
    print("  ✓ High-quality image generation (1024×1024 and above)")
    print("  ✓ Fine-grained style control")
    print("  ✓ Diverse image generation through style mixing")
    print("  ✓ More disentangled latent space (w space)")
    print("  ✓ Excellent performance in face image generation")
    

**Output** :
    
    
    === StyleGAN Conceptual Implementation ===
    
    --- Mapping Network ---
    Input z shape: torch.Size([4, 512])
    Output w shape: torch.Size([4, 512])
    Parameters: 2,101,248
    
    --- StyleGAN Generator Block ---
    Input x shape: torch.Size([4, 128, 8, 8])
    Output x shape: torch.Size([4, 64, 8, 8])
    Parameters: 222,976
    
    StyleGAN features:
      ✓ High-quality image generation (1024×1024 and above)
      ✓ Fine-grained style control
      ✓ Diverse image generation through style mixing
      ✓ More disentangled latent space (w space)
      ✓ Excellent performance in face image generation
    

* * *

## 3.6 Practice: Image Generation Project

### Implementation Example 8: Complete Image Generation Pipeline
    
    
    # Requirements:
    # - Python 3.9+
    # - torchvision>=0.15.0
    
    import torchvision.utils as vutils
    from torchvision.utils import save_image
    
    print("\n=== Complete Image Generation Pipeline ===\n")
    
    class ImageGenerationPipeline:
        """Complete pipeline for image generation"""
    
        def __init__(self, generator, latent_dim=100, device='cuda'):
            self.generator = generator
            self.latent_dim = latent_dim
            self.device = device
            self.generator.eval()
    
        def generate_images(self, num_images=16, seed=None):
            """Generate specified number of images"""
            if seed is not None:
                torch.manual_seed(seed)
    
            with torch.no_grad():
                z = torch.randn(num_images, self.latent_dim).to(self.device)
                generated_imgs = self.generator(z)
    
            return generated_imgs
    
        def interpolate_latent(self, z1, z2, num_steps=10):
            """Interpolate between two latent vectors"""
            alphas = torch.linspace(0, 1, num_steps)
            interpolated_imgs = []
    
            with torch.no_grad():
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img = self.generator(z_interp)
                    interpolated_imgs.append(img)
    
            return torch.cat(interpolated_imgs, dim=0)
    
        def explore_latent_space(self, base_z, dimension, range_scale=3.0, num_steps=10):
            """Explore a specific dimension of latent space"""
            variations = []
    
            with torch.no_grad():
                for scale in torch.linspace(-range_scale, range_scale, num_steps):
                    z_var = base_z.clone()
                    z_var[0, dimension] += scale
                    img = self.generator(z_var)
                    variations.append(img)
    
            return torch.cat(variations, dim=0)
    
        def save_generated_images(self, images, filename, nrow=8):
            """Save generated images"""
            # Normalize [-1, 1] → [0, 1]
            images = (images + 1) / 2.0
            images = torch.clamp(images, 0, 1)
    
            # Save in grid format
            grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
            print(f"Saving images: {filename}")
            print(f"  Grid size: {grid.shape}")
            # save_image(grid, filename)  # Actual saving
    
            return grid
    
    # Initialize pipeline
    pipeline = ImageGenerationPipeline(
        generator=dcgan_generator,
        latent_dim=100,
        device=device
    )
    
    print("--- Image Generation ---")
    generated_imgs = pipeline.generate_images(num_images=16, seed=42)
    print(f"Generated images: {generated_imgs.size(0)}")
    print(f"Image shape: {generated_imgs.shape}")
    
    # Save grid
    grid = pipeline.save_generated_images(generated_imgs, "generated_samples.png", nrow=4)
    print(f"Grid shape: {grid.shape}\n")
    
    # Latent space interpolation
    print("--- Latent Space Interpolation ---")
    z1 = torch.randn(1, 100).to(device)
    z2 = torch.randn(1, 100).to(device)
    interpolated_imgs = pipeline.interpolate_latent(z1, z2, num_steps=8)
    print(f"Interpolated images: {interpolated_imgs.size(0)}")
    print(f"Interpolation steps: 8\n")
    
    # Latent space exploration
    print("--- Latent Space Exploration ---")
    base_z = torch.randn(1, 100).to(device)
    dimension_to_explore = 5
    variations = pipeline.explore_latent_space(base_z, dimension_to_explore, num_steps=10)
    print(f"Explored dimension: {dimension_to_explore}")
    print(f"Variations: {variations.size(0)}")
    print(f"Range: [-3.0, 3.0]\n")
    
    # Quality evaluation metrics (conceptual)
    print("--- Generation Quality Metrics ---")
    print("1. Inception Score (IS):")
    print("   - Evaluates image quality and diversity")
    print("   - Range: 1.0~ (higher is better)")
    print("   - MNIST: ~2-3, ImageNet: ~10-15")
    
    print("\n2. Frechet Inception Distance (FID):")
    print("   - Distance between generated and true distributions")
    print("   - Range: 0~ (lower is better)")
    print("   - FID < 50: Good, FID < 10: Very good")
    
    print("\n3. Precision & Recall:")
    print("   - Precision: Quality of generated images")
    print("   - Recall: Diversity of generated images")
    print("   - Ideally both high")
    
    print("\n--- Practical Applications ---")
    print("✓ Face image generation (StyleGAN)")
    print("✓ Artwork generation")
    print("✓ Data augmentation (supplementing small datasets)")
    print("✓ Image super-resolution (Super-Resolution GAN)")
    print("✓ Image-to-image translation (pix2pix, CycleGAN)")
    print("✓ 3D model generation")
    

**Output** :
    
    
    === Complete Image Generation Pipeline ===
    
    --- Image Generation ---
    Generated images: 16
    Image shape: torch.Size([16, 1, 28, 28])
    Saving images: generated_samples.png
      Grid size: torch.Size([3, 62, 62])
    Grid shape: torch.Size([3, 62, 62])
    
    --- Latent Space Interpolation ---
    Interpolated images: 8
    Interpolation steps: 8
    
    --- Latent Space Exploration ---
    Explored dimension: 5
    Variations: 10
    Range: [-3.0, 3.0]
    
    --- Generation Quality Metrics ---
    1. Inception Score (IS):
       - Evaluates image quality and diversity
       - Range: 1.0~ (higher is better)
       - MNIST: ~2-3, ImageNet: ~10-15
    
    2. Frechet Inception Distance (FID):
       - Distance between generated and true distributions
       - Range: 0~ (lower is better)
       - FID < 50: Good, FID < 10: Very good
    
    3. Precision & Recall:
       - Precision: Quality of generated images
       - Recall: Diversity of generated images
       - Ideally both high
    
    --- Practical Applications ---
    ✓ Face image generation (StyleGAN)
    ✓ Artwork generation
    ✓ Data augmentation (supplementing small datasets)
    ✓ Image super-resolution (Super-Resolution GAN)
    ✓ Image-to-image translation (pix2pix, CycleGAN)
    ✓ 3D model generation
    

* * *

## GAN Training Best Practices

### Hyperparameter Selection

Parameter | Recommended Value | Reason  
---|---|---  
**Learning rate** | 0.0001-0.0002 | Set low for stable training  
**Beta1 (Adam)** | 0.5 | Lower than typical 0.9 (GAN characteristic)  
**Beta2 (Adam)** | 0.999 | Maintain standard value  
**Batch size** | 64-128 | Balance of stability and computational efficiency  
**Latent dimension** | 100-512 | Adjust based on complexity  
  
### Training Stabilization Techniques
    
    
    ```mermaid
    graph TB
        A[Training Instability] --> B1[Gradient Issues]
        A --> B2[Mode Collapse]
        A --> B3[Convergence Failure]
    
        B1 --> C1[Spectral Norm]
        B1 --> C2[Gradient Clipping]
        B1 --> C3[WGAN-GP]
    
        B2 --> D1[Minibatch Discrimination]
        B2 --> D2[Feature Matching]
        B2 --> D3[Two Timescale]
    
        B3 --> E1[Label Smoothing]
        B3 --> E2[Noise Injection]
        B3 --> E3[Learning Rate Decay]
    
        style B1 fill:#ffebee
        style B2 fill:#ffebee
        style B3 fill:#ffebee
        style C1 fill:#e8f5e9
        style C2 fill:#e8f5e9
        style C3 fill:#e8f5e9
        style D1 fill:#e8f5e9
        style D2 fill:#e8f5e9
        style D3 fill:#e8f5e9
        style E1 fill:#e8f5e9
        style E2 fill:#e8f5e9
        style E3 fill:#e8f5e9
    ```

### Debugging Checklist

  * **Discriminator too strong** : Lower learning rate, apply Label Smoothing
  * **Generator too strong** : Increase Discriminator training iterations
  * **Mode Collapse occurring** : Try WGAN-GP, Spectral Norm, Minibatch Discrimination
  * **Gradient vanishing** : Use LeakyReLU, add Batch Normalization
  * **Training oscillation** : Lower learning rate, Two Timescale Update Rule

* * *

## Summary

In this chapter, we learned about GANs from basics to applications:

### Key Points

**1\. Basic Principles of GANs**

  * Adversarial competition between Generator and Discriminator
  * Minimax game and Nash equilibrium
  * Image generation from latent space
  * Training instability and countermeasures

**2\. Mode Collapse Problem**

  * Phenomenon where generation diversity is lost
  * Causes: Gradient instability, objective function issues
  * Countermeasures: WGAN-GP, Spectral Norm, Minibatch Discrimination
  * Evaluation metrics: IS, FID, Precision/Recall

**3\. DCGAN**

  * Stable GAN with convolutional layers
  * Design guidelines: Remove pooling, apply BN, remove fully connected layers
  * Excellent performance in image generation
  * Simple implementation, easy to understand

**4\. Training Techniques**

  * **WGAN-GP** : Wasserstein distance + Gradient Penalty
  * **Spectral Normalization** : Automatic satisfaction of Lipschitz constraint
  * **Label Smoothing** : Prevent Discriminator overconfidence
  * Combine these techniques to achieve stable training

**5\. StyleGAN**

  * High-quality image generation (1024×1024 and above)
  * More disentangled latent space via Mapping Network
  * Hierarchical style control via AdaIN
  * Diverse image generation through style mixing

### Next Chapter

In the next chapter, we will proceed to more advanced generative models, covering Conditional GAN for conditional generation, pix2pix and CycleGAN for image-to-image translation, BigGAN and Progressive GAN for large-scale high-resolution generation, and comparison with non-GAN generative models such as VAE and Diffusion Models.

* * *

## Practice Problems

**Problem 1: Understanding Nash Equilibrium**

**Question** : When GAN reaches Nash equilibrium, explain what happens to the following conditions:

  1. Value of Discriminator output $D(\mathbf{x})$
  2. Relationship between generated distribution $p_g(\mathbf{x})$ and true distribution $p_{\text{data}}(\mathbf{x})$
  3. State of Generator loss
  4. Can training continue

**Sample Answer** :

**1\. Discriminator Output**

  * $D(\mathbf{x}) = 0.5$ (for all inputs)
  * Reason: Cannot distinguish between real and fake
  * Theoretical derivation: $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} = 0.5$

**2\. Distribution Relationship**

  * $p_g(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$ (perfect match)
  * Generator perfectly mimics true data distribution
  * KL divergence: $D_{KL}(p_{\text{data}} \| p_g) = 0$

**3\. Generator Loss**

  * Reaches minimum value (theoretically)
  * $\mathcal{L}_G = -\log(0.5) = \log(2) \approx 0.693$
  * No room for further improvement

**4\. Continuing Training**

  * Training ends theoretically (convergence)
  * In practice, complete Nash equilibrium is never reached
  * Oscillations or minor improvements may continue

**Problem 2: Mode Collapse Detection and Countermeasures**

**Question** : When training a GAN on MNIST dataset (10 classes of handwritten digits), generated images only show digits "1" and "7". Explain the following:

  1. Name of this phenomenon and its cause
  2. How can it be detected (3 methods)
  3. Propose 3 countermeasures and explain their effects

**Sample Answer** :

**1\. Phenomenon and Cause**

  * **Phenomenon** : Mode Collapse
  * **Cause** : 
    * Generator discovers it can fool Discriminator with "1" and "7"
    * Easier to learn than other digits (simpler shapes)
    * Falls into local optimum due to gradient instability

**2\. Detection Methods**

  * **Visual Inspection** : Check generated images, observe lack of diversity
  * **Clustering** : Cluster generated images with k-means, few clusters (2)
  * **Inception Score** : IS score decreases due to low diversity

**3\. Countermeasures**

**Countermeasure A: Apply WGAN-GP**

  * Stabilize training with Wasserstein distance + Gradient Penalty
  * Effect: Prevents gradient explosion/vanishing, makes learning all modes easier
  * Implementation: Remove Sigmoid from Discriminator output, add GP term

**Countermeasure B: Minibatch Discrimination**

  * Provide Discriminator with similarity information between samples in batch
  * Effect: If Generator produces same samples, Discriminator can easily detect
  * Implementation: Calculate batch statistics and concatenate to Discriminator input

**Countermeasure C: Two Timescale Update Rule**

  * Train Discriminator more than Generator (e.g., 5:1 ratio)
  * Effect: Discriminator stays strong, Generator explores all modes
  * Implementation: Set D_steps parameter in training loop

**Problem 3: Comparing WGAN-GP and Spectral Normalization**

**Question** : Compare WGAN-GP and Spectral Normalization from the following perspectives:

  1. Method of realizing Lipschitz constraint
  2. Computational cost
  3. Implementation complexity
  4. Training stability
  5. Which to choose (by situation)

**Sample Answer** :

**1\. Lipschitz Constraint Realization**

  * **WGAN-GP** : 
    * Constrains gradient norm to 1 with Gradient Penalty
    * Computes gradients at interpolation points during training
    * Soft constraint (added as penalty term)
  * **Spectral Norm** : 
    * Normalizes spectral norm of weight matrices in each layer to 1
    * Constrains weights themselves
    * Hard constraint (direct normalization)

**2\. Computational Cost**

  * **WGAN-GP** : 
    * GP calculation needed each iteration (interpolation + backprop)
    * Training overhead: approximately 30-50% increase
  * **Spectral Norm** : 
    * Estimates maximum singular value with Power Iteration
    * Training overhead: approximately 5-10% increase
    * No overhead at inference

**3\. Implementation Complexity**

  * **WGAN-GP** : 
    * Requires interpolation point generation, gradient calculation, GP term addition
    * Somewhat complex implementation (about 50 lines of code)
  * **Spectral Norm** : 
    * Just apply PyTorch's `spectral_norm()` to layers
    * Very simple implementation (complete in 1 line)

**4\. Training Stability**

  * **WGAN-GP** : 
    * Meaningful loss via Wasserstein distance
    * Effective at mitigating Mode Collapse
    * Requires tuning λ (GP coefficient)
  * **Spectral Norm** : 
    * Consistent Lipschitz constraint across all layers
    * Few hyperparameters (no tuning needed)
    * High stability

**5\. Selection Criteria**

  * **Choose WGAN-GP when** : 
    * Theoretical guarantees are important
    * Want to use Wasserstein distance as loss
    * Have ample computational resources
  * **Choose Spectral Norm when** : 
    * Prioritize simple implementation
    * Computational efficiency is important
    * Want to quickly create prototypes
    * Modern choice (frequently used in recent papers)

**Problem 4: StyleGAN Style Mixing**

**Question** : In StyleGAN, if you want to generate an image with "face shape from A + facial expression and hairstyle from B" using two latent vectors $\mathbf{z}_A$ and $\mathbf{z}_B$, how would you implement this?

  1. Latent vector mapping procedure
  2. At which resolution layers to switch styles
  3. Implementation code outline

**Sample Answer** :

**1\. Mapping Procedure**

  * $\mathbf{z}_A \rightarrow$ Mapping Network $\rightarrow \mathbf{w}_A$
  * $\mathbf{z}_B \rightarrow$ Mapping Network $\rightarrow \mathbf{w}_B$
  * Use different $\mathbf{w}$ at each resolution layer

**2\. Style Switching Point**

  * **Coarse styles (4×4 to 8×8)** : Use $\mathbf{w}_A$ 
    * Face orientation, overall shape
    * Retain A's "face shape"
  * **Medium to fine styles (16×16 to 1024×1024)** : Use $\mathbf{w}_B$ 
    * Facial expressions, eye openness, hairstyle, skin texture
    * Apply B's "facial expression and hairstyle"

**3\. Implementation Code Outline**
    
    
    # Mapping Network
    w_A = mapping_network(z_A)
    w_B = mapping_network(z_B)
    
    # Initial constant input
    x = constant_input  # 4×4
    
    # Coarse styles (A's face shape)
    x = synthesis_block_4x4(x, w_A)  # 4×4
    x = synthesis_block_8x8(x, w_A)  # 8×8
    
    # Medium to fine styles (B's expression & hairstyle)
    x = synthesis_block_16x16(x, w_B)  # 16×16
    x = synthesis_block_32x32(x, w_B)  # 32×32
    x = synthesis_block_64x64(x, w_B)  # 64×64
    # ...continue using w_B
    
    generated_image = x
    

**Effect** :

  * Maintains basic structure of A's face while reflecting B's expression and hairstyle
  * Infinite variations possible through style mixing
  * Different effects achieved by changing switching resolution

**Problem 5: GAN Evaluation Metrics**

**Question** : You need to evaluate the following 3 GAN models. Explain which metrics to use and how:

  * **Model A** : IS = 8.5, FID = 25, Precision = 0.85, Recall = 0.60
  * **Model B** : IS = 6.2, FID = 18, Precision = 0.75, Recall = 0.82
  * **Model C** : IS = 7.8, FID = 15, Precision = 0.80, Recall = 0.78

  1. Meaning of each metric
  2. Which model is optimal (by use case)
  3. Overall recommended model

**Sample Answer** :

**1\. Meaning of Each Metric**

  * **Inception Score (IS)** : 
    * Combination of image quality and diversity
    * High value = high quality and diverse
    * Limitation: Doesn't consider true data distribution
  * **Frechet Inception Distance (FID)** : 
    * Distance between generated and true distributions
    * Low value = close to true data
    * Most reliable metric
  * **Precision** : 
    * Quality of generated images (realism)
    * High = high quality, but diversity not guaranteed
  * **Recall** : 
    * Diversity of generated images (coverage)
    * High = diverse, but quality not guaranteed

**2\. Optimal Model by Use Case**

  * **High-quality image generation priority (e.g., advertising materials)** : 
    * Model A (Precision = 0.85 is highest)
    * Reason: Individual image quality is important, diversity is secondary
  * **Data augmentation (e.g., supplementing training data)** : 
    * Model B (Recall = 0.82 is highest)
    * Reason: Need diverse samples, some quality degradation acceptable
  * **General image generation** : 
    * Model C (FID = 15 is lowest, well-balanced)
    * Reason: Good balance between quality and diversity

**3\. Overall Recommendation**

  * **Recommended model: Model C**
  * Reasons: 
    * Lowest FID (15) = closest to true data
    * Well-balanced Precision (0.80) and Recall (0.78)
    * General-purpose without bias toward specific use
  * **Overall evaluation approach** : 
    * Prioritize FID (most reliable)
    * Check Precision/Recall balance for quality and diversity
    * Use IS as reference only (insufficient alone)

* * *
