---
title: "Chapter 4: Deep Learning for Anomaly Detection"
chapter_title: "Chapter 4: Deep Learning for Anomaly Detection"
subtitle: Autoencoder, VAE, GAN, Time Series Anomaly Detection
reading_time: 80-90 minutes
difficulty: Intermediate to Advanced
code_examples: 9
exercises: 6
version: 1.0
created_at: 2025-10-21
---

This chapter covers Deep Learning for Anomaly Detection. You will learn mechanisms of GAN-based anomaly detection (AnoGAN) and Build end-to-end anomaly detection pipelines.

## Learning Objectives

  * Understand the principles and implementation of Autoencoder-based anomaly detection
  * Learn the probabilistic approach of Variational Autoencoder (VAE)
  * Understand the mechanisms of GAN-based anomaly detection (AnoGAN)
  * Implement time series anomaly detection with LSTM Autoencoder
  * Build end-to-end anomaly detection pipelines

## 4.1 Autoencoder-based Anomaly Detection

### 4.1.1 Autoencoder Fundamentals

An **Autoencoder** is an unsupervised learning model that compresses input data and reconstructs it. By training on normal data, it can detect anomalies based on the principle that reconstruction errors are larger for anomalous data.

**Architecture:**
    
    
    Input (x)
        ↓
    Encoder: x → z (latent representation)
        ↓
    Latent Space (z)
        ↓
    Decoder: z → x̂ (reconstruction)
        ↓
    Reconstruction Error: ||x - x̂||²
    

**Principles of Anomaly Detection:**

  * Train on normal data: Learn normal patterns
  * Small reconstruction error: Normal data
  * Large reconstruction error: Anomalous data (patterns not learned)

**Mathematical Expression:**

$$ \text{Anomaly Score} = \|x - \text{Decoder}(\text{Encoder}(x))\|^2 $$

### 4.1.2 Reconstruction Error and Threshold Selection

For anomaly determination, a threshold is set for the reconstruction error.

**Threshold Setting Methods:**

Method | Description | Application Scenario  
---|---|---  
Percentile Method | 95th percentile of training data reconstruction error | Learning with normal data only  
Statistical Method | Mean + 3σ | Assumes normal distribution  
ROC Curve | Maximize AUC on validation data | Small amount of anomaly labels available  
Business Requirements | Specify False Positive rate | Emphasis on production operations  
  
### 4.1.3 PyTorch Implementation (Complete Version)
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # Autoencoder model definition
    class Autoencoder(nn.Module):
        """Simple Autoencoder"""
        def __init__(self, input_dim=784, hidden_dims=[128, 64, 32]):
            super(Autoencoder, self).__init__()
    
            # Encoder
            encoder_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = hidden_dim
    
            self.encoder = nn.Sequential(*encoder_layers)
    
            # Decoder (reverse order of Encoder)
            decoder_layers = []
            for i in range(len(hidden_dims) - 1, 0, -1):
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
    
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            decoder_layers.append(nn.Sigmoid())  # Normalize output to [0,1]
    
            self.decoder = nn.Sequential(*decoder_layers)
    
        def forward(self, x):
            """Forward pass"""
            z = self.encoder(x)  # Encode
            x_reconstructed = self.decoder(z)  # Decode
            return x_reconstructed
    
        def encode(self, x):
            """Get latent representation"""
            return self.encoder(x)
    
    
    def train_autoencoder(model, train_loader, n_epochs=50, lr=0.001):
        """Train Autoencoder"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()  # Reconstruction error (Mean Squared Error)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        train_losses = []
    
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
    
            for batch_x, in train_loader:
                batch_x = batch_x.to(device)
    
                # Forward pass
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
        return model, train_losses
    
    
    def compute_reconstruction_errors(model, data_loader):
        """Compute reconstruction errors"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    
        errors = []
    
        with torch.no_grad():
            for batch_x, in data_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
    
                # Reconstruction error per sample (MSE)
                batch_errors = torch.mean((batch_x - reconstructed) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())
    
        return np.array(errors)
    
    
    def detect_anomalies(model, test_loader, threshold):
        """Perform anomaly detection"""
        errors = compute_reconstruction_errors(model, test_loader)
        predictions = (errors > threshold).astype(int)
        return predictions, errors
    
    
    # Usage example
    if __name__ == "__main__":
        # Generate sample data (normal data from normal distribution)
        np.random.seed(42)
        torch.manual_seed(42)
    
        # Normal data (28x28 = 784 dimensions)
        n_normal = 1000
        normal_data = np.random.randn(n_normal, 784) * 0.5 + 0.5
        normal_data = np.clip(normal_data, 0, 1)
    
        # Anomalous data (different distribution from normal)
        n_anomaly = 50
        anomaly_data = np.random.uniform(0, 1, (n_anomaly, 784))
    
        # PyTorch Dataset
        train_dataset = TensorDataset(torch.FloatTensor(normal_data))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
        test_data = np.vstack([normal_data[:100], anomaly_data])
        test_labels = np.array([0] * 100 + [1] * n_anomaly)  # 0: Normal, 1: Anomaly
    
        test_dataset = TensorDataset(torch.FloatTensor(test_data))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
        # Model training
        print("=== Autoencoder Training Started ===")
        model = Autoencoder(input_dim=784, hidden_dims=[256, 128, 64])
        trained_model, losses = train_autoencoder(model, train_loader, n_epochs=50, lr=0.001)
    
        # Threshold setting (95th percentile of training data)
        train_errors = compute_reconstruction_errors(trained_model, train_loader)
        threshold = np.percentile(train_errors, 95)
        print(f"\nThreshold (95th percentile): {threshold:.6f}")
    
        # Anomaly detection on test data
        predictions, test_errors = detect_anomalies(trained_model, test_loader, threshold)
    
        # Evaluation
        from sklearn.metrics import classification_report, roc_auc_score
    
        print("\n=== Anomaly Detection Results ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
    
        auc_score = roc_auc_score(test_labels, test_errors)
        print(f"ROC-AUC: {auc_score:.3f}")
    
        # Visualization
        plt.figure(figsize=(12, 4))
    
        # Training curve
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
    
        # Reconstruction error distribution
        plt.subplot(1, 2, 2)
        plt.hist(test_errors[test_labels == 0], bins=30, alpha=0.6, label='Normal')
        plt.hist(test_errors[test_labels == 1], bins=30, alpha=0.6, label='Anomaly')
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig('autoencoder_anomaly_detection.png', dpi=150)
        print("\nGraph saved: autoencoder_anomaly_detection.png")
    

### 4.1.4 Network Architecture Selection

**Architecture Design Considerations:**

Component | Recommended Value | Reason  
---|---|---  
Latent Dimension | 10-30% of input dimension | Excessive compression causes information loss, too large leads to identity mapping  
Number of Hidden Layers | 2-4 layers | Too deep is difficult to train, too shallow lacks expressiveness  
Activation Function | ReLU (hidden layers), Sigmoid (output) | Prevents vanishing gradients, constrains output range  
Dropout | 0.2-0.3 | Prevents overfitting (use cautiously for anomaly detection)  
  
* * *

## 4.2 Variational Autoencoder (VAE)

### 4.2.1 Motivation for VAE

**Limitations of Standard Autoencoders:**

  * Latent space is discontinuous and lacks meaningful structure
  * Prone to overfitting to training data
  * Limited generative capability

**Features of VAE:**

  * Models latent variables as probability distributions
  * Learns smooth latent space through regularization
  * Functions as a generative model

### 4.2.2 Mathematical Foundation of VAE

**Probabilistic Encoder:**

$$ q_\phi(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)) $$

The encoder outputs mean $\mu(x)$ and variance $\sigma^2(x)$.

**Decoder:**

$$ p_\theta(x|z) = \mathcal{N}(x; \mu_{\text{dec}}(z), \sigma^2_{\text{dec}}) $$

**Loss Function (ELBO):**

$$ \mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}} $$

  * First term: Reconstruction loss (same as Autoencoder)
  * Second term: KL divergence (regularization term), assuming $p(z) = \mathcal{N}(0, I)$

**Closed-form KL Divergence:**

$$ D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2) $$

### 4.2.3 Anomaly Detection with VAE

In VAE, the anomaly score is calculated by combining reconstruction error and KL divergence.

$$ \text{Anomaly Score} = \text{Reconstruction Error} + \beta \cdot D_{KL} $$

### 4.2.4 PyTorch Implementation
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class VAE(nn.Module):
        """Variational Autoencoder"""
        def __init__(self, input_dim=784, latent_dim=32, hidden_dims=[256, 128]):
            super(VAE, self).__init__()
    
            self.latent_dim = latent_dim
    
            # Encoder
            encoder_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = hidden_dim
    
            self.encoder = nn.Sequential(*encoder_layers)
    
            # Latent distribution parameters (mean and variance)
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
            # Decoder
            decoder_layers = []
            decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
            decoder_layers.append(nn.ReLU())
    
            for i in range(len(hidden_dims) - 1, 0, -1):
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
    
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            decoder_layers.append(nn.Sigmoid())
    
            self.decoder = nn.Sequential(*decoder_layers)
    
        def encode(self, x):
            """Encode: output mean and log variance"""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """Reparameterization trick"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # Sample from N(0, 1)
            z = mu + eps * std
            return z
    
        def decode(self, z):
            """Decode"""
            return self.decoder(z)
    
        def forward(self, x):
            """Forward pass"""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_reconstructed = self.decode(z)
            return x_reconstructed, mu, logvar
    
    
    def vae_loss(x, x_reconstructed, mu, logvar, beta=1.0):
        """VAE loss function
    
        Args:
            beta: Weight for KL divergence (β-VAE)
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        # Total loss
        total_loss = recon_loss + beta * kl_div
    
        return total_loss, recon_loss, kl_div
    
    
    def train_vae(model, train_loader, n_epochs=50, lr=0.001, beta=1.0):
        """Train VAE"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
    
            for batch_x, in train_loader:
                batch_x = batch_x.to(device)
    
                # Forward pass
                x_recon, mu, logvar = model(batch_x)
                loss, recon, kl = vae_loss(batch_x, x_recon, mu, logvar, beta)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
    
            avg_loss = train_loss / len(train_loader.dataset)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
        return model
    
    
    def vae_anomaly_score(model, x, beta=1.0):
        """Compute anomaly score with VAE"""
        model.eval()
        device = next(model.parameters()).device
    
        with torch.no_grad():
            x = x.to(device)
            x_recon, mu, logvar = model(x)
    
            # Reconstruction error (per sample)
            recon_error = F.binary_cross_entropy(x_recon, x, reduction='none').sum(dim=1)
    
            # KL divergence (per sample)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
            # Anomaly score
            anomaly_scores = recon_error + beta * kl_div
    
        return anomaly_scores.cpu().numpy()
    
    
    # Usage example
    if __name__ == "__main__":
        # Data preparation (same as before)
        train_dataset = TensorDataset(torch.FloatTensor(normal_data))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
        # VAE model
        print("=== VAE Training Started ===")
        vae_model = VAE(input_dim=784, latent_dim=32, hidden_dims=[256, 128])
        trained_vae = train_vae(vae_model, train_loader, n_epochs=50, lr=0.001, beta=1.0)
    
        # Compute anomaly scores
        test_tensor = torch.FloatTensor(test_data)
        anomaly_scores = vae_anomaly_score(trained_vae, test_tensor, beta=1.0)
    
        # Threshold setting and evaluation
        threshold = np.percentile(anomaly_scores[:100], 95)  # 95th percentile of normal data
        predictions = (anomaly_scores > threshold).astype(int)
    
        print("\n=== VAE Anomaly Detection Results ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
        print(f"ROC-AUC: {roc_auc_score(test_labels, anomaly_scores):.3f}")
    

### 4.2.5 Latent Space Analysis

The latent space of VAE has a smooth distribution of normal data. Anomalous data is expected to be located in outlying regions of the latent space.
    
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    def visualize_latent_space(model, data, labels):
        """Visualize latent space (2D projection)"""
        model.eval()
        device = next(model.parameters()).device
    
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(device)
            mu, _ = model.encode(data_tensor)
            latent_codes = mu.cpu().numpy()
    
        # Compress to 2D (if latent dimension is greater than 2)
        if latent_codes.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_codes)
        else:
            latent_2d = latent_codes
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_2d[labels == 0, 0], latent_2d[labels == 0, 1],
                    c='blue', alpha=0.5, label='Normal')
        plt.scatter(latent_2d[labels == 1, 0], latent_2d[labels == 1, 1],
                    c='red', alpha=0.5, label='Anomaly')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('VAE Latent Space Visualization')
        plt.legend()
        plt.grid(True)
        plt.savefig('vae_latent_space.png', dpi=150)
        print("Latent space visualization saved: vae_latent_space.png")
    
    # Usage example
    visualize_latent_space(trained_vae, test_data, test_labels)
    

* * *

## 4.3 GAN-based Anomaly Detection

### 4.3.1 AnoGAN (Anomaly Detection with GAN)

**AnoGAN** learns a generative model of normal data using GANs and detects anomalies based on how much the test data deviates from this generative distribution.

**Training Phase:**

  * Train GAN on normal data
  * Generator G learns the distribution of normal data

**Testing Phase:**

  1. For test sample $x$, optimize latent variable $z$: $G(z) \approx x$
  2. Calculate anomaly score: Residual Loss + Discrimination Loss

### 4.3.2 Definition of Anomaly Score

$$ A(x) = (1 - \lambda) \cdot L_R(x) + \lambda \cdot L_D(x) $$

  * $L_R(x) = \|x - G(z^*)\|_1$: Residual Loss (reconstruction error)
  * $L_D(x) = \|f(x) - f(G(z^*))\|_1$: Discrimination Loss (distance in feature space)
  * $f(\cdot)$: Features from intermediate layer of Discriminator

### 4.3.3 Latent Variable Optimization

For test sample $x$, search for $z$ such that $G(z) \approx x$ using gradient descent:

$$ z^* = \arg\min_z \|x - G(z)\|_1 + \lambda \|f(x) - f(G(z))\|_1 $$

### 4.3.4 Implementation Overview
    
    
    import torch
    import torch.nn as nn
    
    class Generator(nn.Module):
        """GAN Generator"""
        def __init__(self, latent_dim=100, output_dim=784):
            super(Generator, self).__init__()
    
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, output_dim),
                nn.Sigmoid()
            )
    
        def forward(self, z):
            return self.model(z)
    
    
    class Discriminator(nn.Module):
        """GAN Discriminator (also extracts intermediate layer features)"""
        def __init__(self, input_dim=784):
            super(Discriminator, self).__init__()
    
            self.features = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2)
            )
    
            self.classifier = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
    
        def forward(self, x, return_features=False):
            feat = self.features(x)
            output = self.classifier(feat)
    
            if return_features:
                return output, feat
            return output
    
    
    def find_latent_code(generator, discriminator, x, n_iterations=500, lr=0.01, lambda_weight=0.1):
        """Search for optimal latent variable z for test sample x"""
        device = next(generator.parameters()).device
    
        # Initialize
        z = torch.randn(x.size(0), generator.model[0].in_features, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)
    
        for i in range(n_iterations):
            optimizer.zero_grad()
    
            # Generate
            G_z = generator(z)
    
            # Residual Loss
            residual_loss = torch.mean(torch.abs(x - G_z))
    
            # Discrimination Loss (distance in feature space)
            _, feat_real = discriminator(x, return_features=True)
            _, feat_fake = discriminator(G_z, return_features=True)
            discrimination_loss = torch.mean(torch.abs(feat_real - feat_fake))
    
            # Total loss
            loss = (1 - lambda_weight) * residual_loss + lambda_weight * discrimination_loss
    
            loss.backward()
            optimizer.step()
    
        # Anomaly score
        with torch.no_grad():
            G_z_final = generator(z)
            residual = torch.mean(torch.abs(x - G_z_final), dim=1)
    
            _, feat_real = discriminator(x, return_features=True)
            _, feat_fake = discriminator(G_z_final, return_features=True)
            discrimination = torch.mean(torch.abs(feat_real - feat_fake), dim=1)
    
            anomaly_scores = (1 - lambda_weight) * residual + lambda_weight * discrimination
    
        return anomaly_scores.cpu().numpy()
    
    
    # Note: GAN training code is omitted (perform standard GAN training)
    # In actual use, first train the GAN on normal data, then use the above function for anomaly detection
    

> **Note** : AnoGAN is time-consuming due to latent variable optimization, making it unsuitable for real-time anomaly detection. To address this issue, improved methods such as Fast-AnoGAN and EGBAd have been proposed.

* * *

## 4.4 Time Series Anomaly Detection

### 4.4.1 Characteristics of Time Series Data

Time series anomaly detection requires consideration of the following characteristics:

  * **Temporal Dependencies** : Past values influence the future
  * **Seasonality and Periodicity** : Daily, weekly, and yearly patterns
  * **Trends** : Long-term upward or downward tendencies
  * **Multivariate Nature** : Multiple sensor values are interrelated

### 4.4.2 LSTM Autoencoder

**LSTM Autoencoder** learns temporal patterns in time series using LSTM and detects anomalies through reconstruction error.

**Architecture:**
    
    
    Input: (batch, seq_len, features)
        ↓
    LSTM Encoder: Compress time series to fixed-length vector
        ↓
    Latent Vector: (batch, latent_dim)
        ↓
    LSTM Decoder: Reconstruct time series from latent vector
        ↓
    Output: (batch, seq_len, features)
    

### 4.4.3 PyTorch Implementation
    
    
    import torch
    import torch.nn as nn
    
    class LSTMAutoencoder(nn.Module):
        """LSTM-based Autoencoder for time series"""
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, latent_dim=32):
            super(LSTMAutoencoder, self).__init__()
    
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.latent_dim = latent_dim
    
            # Encoder LSTM
            self.encoder_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
    
            # Compress to latent representation
            self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
    
            # FC for Decoder (latent representation to LSTM initial state)
            self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
    
            # Decoder LSTM
            self.decoder_lstm = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
    
            # Output layer
            self.output_fc = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """Encode: time series → latent vector"""
            # x: (batch, seq_len, input_dim)
            lstm_out, (hidden, cell) = self.encoder_lstm(x)
    
            # Use last hidden state
            last_hidden = hidden[-1]  # (batch, hidden_dim)
    
            # Compress to latent vector
            z = self.encoder_fc(last_hidden)  # (batch, latent_dim)
    
            return z
    
        def decode(self, z, seq_len):
            """Decode: latent vector → time series"""
            batch_size = z.size(0)
    
            # Decoder LSTM initial state
            hidden = self.decoder_fc(z).unsqueeze(0)  # (1, batch, hidden_dim)
            hidden = hidden.repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
            cell = torch.zeros_like(hidden)
    
            # Decoder input (repeat latent vector seq_len times)
            decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
    
            # LSTM Decoder
            lstm_out, _ = self.decoder_lstm(decoder_input, (hidden, cell))
            # lstm_out: (batch, seq_len, hidden_dim)
    
            # Output layer
            output = self.output_fc(lstm_out)  # (batch, seq_len, input_dim)
    
            return output
    
        def forward(self, x):
            """Forward pass"""
            seq_len = x.size(1)
    
            z = self.encode(x)
            x_reconstructed = self.decode(z, seq_len)
    
            return x_reconstructed
    
    
    def train_lstm_autoencoder(model, train_loader, n_epochs=50, lr=0.001):
        """Train LSTM Autoencoder"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
    
            for batch_x, in train_loader:
                batch_x = batch_x.to(device)
    
                # Forward pass
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(train_loader)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
    
        return model
    
    
    def detect_ts_anomalies(model, data_loader, threshold):
        """Time series anomaly detection"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    
        all_errors = []
    
        with torch.no_grad():
            for batch_x, in data_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
    
                # Reconstruction error for entire sequence (average)
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
    
        all_errors = np.array(all_errors)
        predictions = (all_errors > threshold).astype(int)
    
        return predictions, all_errors
    
    
    # Usage example
    if __name__ == "__main__":
        # Generate sample time series data (normal: sine wave, anomaly: noise)
        np.random.seed(42)
        torch.manual_seed(42)
    
        seq_len = 50
        input_dim = 5  # 5 sensors
    
        # Normal data (sine wave based)
        n_normal_sequences = 500
        t = np.linspace(0, 4*np.pi, seq_len)
        normal_sequences = []
        for _ in range(n_normal_sequences):
            seq = np.array([np.sin(t + np.random.randn() * 0.1) for _ in range(input_dim)]).T
            seq += np.random.randn(seq_len, input_dim) * 0.1
            normal_sequences.append(seq)
    
        normal_sequences = np.array(normal_sequences)  # (n_normal, seq_len, input_dim)
    
        # Anomalous data (random noise)
        n_anomaly_sequences = 50
        anomaly_sequences = np.random.randn(n_anomaly_sequences, seq_len, input_dim)
    
        # Dataset
        train_dataset = TensorDataset(torch.FloatTensor(normal_sequences))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
        test_sequences = np.vstack([normal_sequences[:50], anomaly_sequences])
        test_labels = np.array([0] * 50 + [1] * n_anomaly_sequences)
    
        test_dataset = TensorDataset(torch.FloatTensor(test_sequences))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
        # Model training
        print("=== LSTM Autoencoder Training Started ===")
        lstm_ae = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64, num_layers=2, latent_dim=32)
        trained_lstm_ae = train_lstm_autoencoder(lstm_ae, train_loader, n_epochs=50, lr=0.001)
    
        # Threshold setting
        train_errors = []
        trained_lstm_ae.eval()
        with torch.no_grad():
            for batch_x, in train_loader:
                reconstructed = trained_lstm_ae(batch_x)
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                train_errors.extend(errors.cpu().numpy())
    
        threshold = np.percentile(train_errors, 95)
        print(f"\nThreshold (95th percentile): {threshold:.6f}")
    
        # Anomaly detection
        predictions, test_errors = detect_ts_anomalies(trained_lstm_ae, test_loader, threshold)
    
        print("\n=== LSTM Autoencoder Anomaly Detection Results ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
        print(f"ROC-AUC: {roc_auc_score(test_labels, test_errors):.3f}")
    

### 4.4.4 Multivariate Time Series Anomaly Detection

When handling data from multiple sensors simultaneously, correlations between variables must also be considered.

**Methods:**

  * **LSTM Autoencoder** : Already supported in the above implementation
  * **Attention Mechanism** : Interpret which variables contribute to anomalies
  * **Transformer** : Learn long-term dependencies

* * *

## 4.5 End-to-End Practice

### 4.5.1 Data Preparation

In real-world anomaly detection, data is prepared through the following steps.
    
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    class AnomalyDetectionPipeline:
        """Anomaly detection pipeline"""
        def __init__(self, model_type='autoencoder'):
            self.model_type = model_type
            self.scaler = StandardScaler()
            self.model = None
            self.threshold = None
    
        def preprocess(self, data, fit_scaler=False):
            """Preprocessing: normalization, missing value handling, etc."""
            # Missing value imputation (mean)
            data = data.fillna(data.mean())
    
            # Standardization
            if fit_scaler:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = self.scaler.transform(data)
    
            return data_scaled
    
        def create_sequences(self, data, seq_len=50):
            """Split time series data into sequences"""
            sequences = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i+seq_len])
    
            return np.array(sequences)
    
        def train(self, normal_data, seq_len=50, n_epochs=50):
            """Model training"""
            # Preprocessing
            normal_scaled = self.preprocess(normal_data, fit_scaler=True)
    
            # Sequence creation
            if self.model_type in ['lstm_ae', 'transformer']:
                sequences = self.create_sequences(normal_scaled, seq_len)
                train_dataset = TensorDataset(torch.FloatTensor(sequences))
            else:
                # For Autoencoder
                train_dataset = TensorDataset(torch.FloatTensor(normal_scaled))
    
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
            # Model selection and training
            if self.model_type == 'autoencoder':
                self.model = Autoencoder(input_dim=normal_scaled.shape[1])
                self.model, _ = train_autoencoder(self.model, train_loader, n_epochs)
            elif self.model_type == 'vae':
                self.model = VAE(input_dim=normal_scaled.shape[1])
                self.model = train_vae(self.model, train_loader, n_epochs)
            elif self.model_type == 'lstm_ae':
                self.model = LSTMAutoencoder(input_dim=normal_scaled.shape[1])
                self.model = train_lstm_autoencoder(self.model, train_loader, n_epochs)
    
            # Threshold setting (95th percentile of training data)
            if self.model_type == 'vae':
                scores = vae_anomaly_score(self.model, torch.FloatTensor(normal_scaled))
            else:
                scores = compute_reconstruction_errors(self.model, train_loader)
    
            self.threshold = np.percentile(scores, 95)
            print(f"Threshold set: {self.threshold:.6f}")
    
        def predict(self, test_data, seq_len=50):
            """Anomaly prediction"""
            # Preprocessing
            test_scaled = self.preprocess(test_data, fit_scaler=False)
    
            # Sequence creation
            if self.model_type in ['lstm_ae', 'transformer']:
                sequences = self.create_sequences(test_scaled, seq_len)
                test_dataset = TensorDataset(torch.FloatTensor(sequences))
            else:
                test_dataset = TensorDataset(torch.FloatTensor(test_scaled))
    
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
            # Anomaly score calculation
            if self.model_type == 'vae':
                scores = vae_anomaly_score(self.model, torch.FloatTensor(test_scaled))
            else:
                scores = compute_reconstruction_errors(self.model, test_loader)
    
            # Anomaly determination
            predictions = (scores > self.threshold).astype(int)
    
            return predictions, scores
    
    
    # Usage example
    if __name__ == "__main__":
        # Dummy dataframe
        normal_df = pd.DataFrame(np.random.randn(1000, 10))
        test_df = pd.DataFrame(np.random.randn(100, 10))
    
        # Pipeline
        pipeline = AnomalyDetectionPipeline(model_type='autoencoder')
        pipeline.train(normal_df, n_epochs=30)
    
        predictions, scores = pipeline.predict(test_df)
        print(f"\nNumber of anomalies detected: {predictions.sum()} / {len(predictions)}")
    

### 4.5.2 Model Selection

Select an appropriate model based on data characteristics.

Data Type | Recommended Model | Reason  
---|---|---  
Image Data | Convolutional AE, VAE | Preserves spatial structure  
Time Series Data | LSTM AE, Transformer | Captures temporal dependencies  
Tabular Data | Autoencoder, VAE | Simple and effective  
High-dimensional Sparse | Sparse AE, VAE | Dimensionality reduction and regularization  
  
### 4.5.3 Threshold Adjustment

In production operations, thresholds are adjusted according to business requirements.

  * **Emphasize False Positive Rate** : Set threshold high (reduce false alarms)
  * **Emphasize Recall** : Set threshold low (don't miss anomalies)
  * **Maximize F1** : Select point that maximizes F1 score on validation data

### 4.5.4 Production Deployment

**Real-time Anomaly Detection System Architecture:**
    
    
    Data Collection (Sensors, Logs)
        ↓
    Preprocessing Pipeline (Normalization, Sequencing)
        ↓
    Anomaly Detection Model (PyTorch → ONNX → TorchScript)
        ↓
    Threshold Determination
        ↓
    Alerts and Visualization (Grafana, Slack Notifications)
    

**Deployment Considerations:**

  * **Model Optimization** : Speed up inference with TorchScript, ONNX conversion
  * **Batch Processing** : Use batching for efficiency when real-time is not required
  * **Regular Retraining** : Adapt to changes in data distribution (Concept Drift)
  * **Automatic Threshold Adjustment** : Adaptively adjust from operational data

### 4.5.5 Monitoring and Alerts
    
    
    import logging
    from datetime import datetime
    
    class AnomalyMonitor:
        """Anomaly detection monitoring"""
        def __init__(self, alert_threshold=0.9):
            self.alert_threshold = alert_threshold
            self.logger = self._setup_logger()
    
        def _setup_logger(self):
            logger = logging.getLogger('AnomalyDetection')
            logger.setLevel(logging.INFO)
    
            # File handler
            fh = logging.FileHandler('anomaly_detection.log')
            fh.setLevel(logging.INFO)
    
            # Format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
    
            logger.addHandler(fh)
            return logger
    
        def log_anomaly(self, timestamp, anomaly_score, features):
            """Log anomaly"""
            self.logger.info(f"Anomaly detected - Time: {timestamp}, Score: {anomaly_score:.4f}")
            self.logger.info(f"Features: {features}")
    
        def send_alert(self, anomaly_score, message):
            """Send alert (example implementation)"""
            if anomaly_score > self.alert_threshold:
                # Send to Slack, Email, PagerDuty, etc.
                print(f"[ALERT] High anomaly detected: {message}")
                self.logger.warning(f"High severity alert: {message}")
    
        def monitor(self, pipeline, data_stream):
            """Real-time monitoring"""
            for timestamp, data in data_stream:
                predictions, scores = pipeline.predict(data)
    
                if predictions.any():
                    self.log_anomaly(timestamp, scores.max(), data)
                    self.send_alert(scores.max(), f"Anomaly at {timestamp}")
    
    
    # Usage example (virtual data stream)
    monitor = AnomalyMonitor(alert_threshold=0.9)
    
    # Virtual data stream
    def data_stream_generator():
        for i in range(10):
            timestamp = datetime.now()
            data = pd.DataFrame(np.random.randn(1, 10))
            yield timestamp, data
    
    # Run monitoring
    # monitor.monitor(pipeline, data_stream_generator())
    

* * *

## Summary

What we learned in this chapter:

  1. **Autoencoder-based Anomaly Detection:**
     * Detecting anomalies with reconstruction error
     * Network architecture design
     * Threshold selection methods
     * Complete PyTorch implementation
  2. **Variational Autoencoder (VAE):**
     * Anomaly detection with probabilistic latent representations
     * Reconstruction error + KL divergence
     * Latent space visualization and analysis
     * Adjustment with β-VAE
  3. **GAN-based Anomaly Detection:**
     * Principles and implementation of AnoGAN
     * Latent variable optimization
     * Utilizing Discriminator features
     * Improved methods such as Fast-AnoGAN
  4. **Time Series Anomaly Detection:**
     * LSTM Autoencoder implementation
     * Learning temporal patterns
     * Handling multivariate time series
     * Processing sequence data
  5. **End-to-End Practice:**
     * Data preprocessing pipelines
     * Guidelines for model selection
     * Threshold adjustment methods
     * Production deployment design
     * Monitoring and alerts

* * *

## Exercises

**Question 1:** Explain why the latent dimension in Autoencoder-based anomaly detection is set to 10% of the input dimension.

**Question 2:** Explain the role of the KL divergence term in the VAE loss function from the perspective of the latent space.

**Question 3:** List three main differences between AnoGAN and standard Autoencoder in anomaly detection.

**Question 4:** Discuss how to determine the sequence length when performing time series anomaly detection with LSTM Autoencoder from three perspectives.

**Question 5:** If there is a business requirement that the False Positive rate be 5% or less when setting the threshold for an anomaly detection model, explain specifically how the threshold should be determined.

**Question 6:** List five technical challenges to consider when building a real-time anomaly detection system and propose solutions for each.

* * *

## References

  1. Goodfellow, I. et al. "Deep Learning." MIT Press (2016).
  2. Kingma, D. P., & Welling, M. "Auto-Encoding Variational Bayes." _ICLR_ (2014).
  3. Schlegl, T. et al. "Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery." _IPMI_ (2017). [AnoGAN]
  4. Malhotra, P. et al. "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection." _ICML Anomaly Detection Workshop_ (2016).
  5. Park, D. et al. "A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder." _IEEE Robotics and Automation Letters_ (2018).
  6. Su, Y. et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." _KDD_ (2019).
  7. Vaswani, A. et al. "Attention is All You Need." _NeurIPS_ (2017). [Transformer]
  8. Audibert, J. et al. "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." _KDD_ (2020).

* * *
