---
title: 第2章：VAE（Variational Autoencoder）
chapter_title: 第2章：VAE（Variational Autoencoder）
subtitle: 確率的潜在変数モデルとELBOによる完全理解
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 通常のAutoencoderの限界とVAEの動機を理解する
  * ✅ ELBO（Evidence Lower Bound）とKL divergenceの理論を説明できる
  * ✅ Reparameterization Trickの仕組みと必要性を理解する
  * ✅ VAEのEncoder（Recognition network）とDecoder（Generative network）を実装できる
  * ✅ PyTorchでMNIST/CelebA画像生成システムを構築できる
  * ✅ 潜在空間を可視化し、補間（interpolation）を実行できる

* * *

## 2.1 Autoencoderの復習と限界

### 通常のAutoencoderの構造

**Autoencoder（AE）** は、入力データを低次元の潜在表現に圧縮し、それを復元する教師なし学習モデルです。
    
    
    ```mermaid
    graph LR
        X["入力 x(784次元)"] --> E["Encoderq(z|x)"]
        E --> Z["潜在変数 z(2-20次元)"]
        Z --> D["Decoderp(x|z)"]
        D --> X2["復元 x̂(784次元)"]
    
        style E fill:#b3e5fc
        style Z fill:#fff9c4
        style D fill:#ffab91
    ```

### Autoencoderの目的関数

通常のAutoencoderは、**再構成誤差** を最小化します：

$$ \mathcal{L}_{\text{AE}} = \|x - \hat{x}\|^2 = \|x - \text{Decoder}(\text{Encoder}(x))\|^2 $$ 

### 基本的なAutoencoderの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Autoencoder(nn.Module):
        """通常のAutoencoder（決定論的）"""
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super(Autoencoder, self).__init__()
    
            # Encoder
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)
    
            # Decoder
            self.fc3 = nn.Linear(latent_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """入力を潜在表現にエンコード"""
            h = F.relu(self.fc1(x))
            z = self.fc2(h)
            return z
    
        def decode(self, z):
            """潜在表現から入力を復元"""
            h = F.relu(self.fc3(z))
            x_recon = torch.sigmoid(self.fc4(h))
            return x_recon
    
        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z
    
    
    # 動作確認
    print("=== 通常のAutoencoderの動作 ===")
    ae = Autoencoder(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # ダミーデータ（28x28のMNIST画像）
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    x_recon, z = ae(x)
    
    print(f"入力: {x.shape}")
    print(f"潜在変数: {z.shape}")
    print(f"復元: {x_recon.shape}")
    
    # 再構成誤差
    recon_loss = F.mse_loss(x_recon, x)
    print(f"再構成誤差: {recon_loss.item():.4f}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in ae.parameters())
    print(f"総パラメータ数: {total_params:,}")
    

### Autoencoderの限界

問題点 | 説明 | 影響  
---|---|---  
**決定論的** | 同じ入力は常に同じ潜在変数を生成 | 多様性のあるサンプル生成ができない  
**構造化されていない潜在空間** | 潜在空間に明確な確率分布がない | ランダムサンプリングが困難  
**過学習しやすい** | 訓練データを記憶してしまう | 新規データ生成に弱い  
**補間の品質** | 潜在空間の補間が意味を持たない場合がある | スムーズな変換ができない  
  
> 「通常のAutoencoderは圧縮・復元はできるが、新しいデータを生成する生成モデルとしては不十分です」

### 潜在空間の問題の可視化
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 通常のAutoencoderの潜在空間の問題
    print("\n=== Autoencoderの潜在空間の問題 ===")
    
    # ランダムな入力を複数生成
    num_samples = 100
    x_samples = torch.randn(num_samples, 784)
    
    # Encoderで潜在表現を取得
    ae.eval()
    with torch.no_grad():
        z_samples = ae.encode(x_samples)
    
    print(f"潜在変数のサンプル数: {z_samples.shape[0]}")
    print(f"潜在次元: {z_samples.shape[1]}")
    
    # 潜在空間の統計
    z_mean = z_samples.mean(dim=0)
    z_std = z_samples.std(dim=0)
    
    print(f"\n潜在変数の平均（一部）: {z_mean[:5].numpy()}")
    print(f"潜在変数の標準偏差（一部）: {z_std[:5].numpy()}")
    print("→ 平均・分散がバラバラで構造化されていない")
    
    # ランダムに潜在空間からサンプリングして復元を試みる
    z_random = torch.randn(10, 20)  # ランダムな潜在変数
    with torch.no_grad():
        x_from_random = ae.decode(z_random)
    
    print(f"\nランダムサンプリングからの復元: {x_from_random.shape}")
    print("→ 意味のある画像が生成されない可能性が高い")
    print("→ これがVAEが必要な理由！")
    

* * *

## 2.2 VAEの動機と理論

### VAEの基本アイデア

**Variational Autoencoder（VAE）** は、Autoencoderに**確率的な枠組み** を導入することで、以下を実現します：

  1. **構造化された潜在空間** ：潜在変数を正規分布 $\mathcal{N}(0, I)$ に従うよう正則化
  2. **確率的生成** ：潜在空間からのサンプリングで新しいデータを生成可能
  3. **滑らかな補間** ：潜在空間で補間すると意味のある中間データが得られる

    
    
    ```mermaid
    graph TB
        X["入力 x"] --> E["Encoderq_φ(z|x)"]
        E --> Mu["平均 μ"]
        E --> Logvar["対数分散 log σ²"]
        Mu --> Sample["サンプリングz ~ N(μ, σ²)"]
        Logvar --> Sample
        Sample --> Z["潜在変数 z"]
        Z --> D["Decoderp_θ(x|z)"]
        D --> X2["復元 x̂"]
    
        Prior["事前分布p(z) = N(0,I)"] -.->|KL正則化| Sample
    
        style E fill:#b3e5fc
        style Sample fill:#fff59d
        style D fill:#ffab91
        style Prior fill:#c5e1a5
    ```

### 確率的定式化

VAEは以下の確率モデルを仮定します：

  * **生成過程（Generative process）** ： $$ \begin{align} z &\sim p(z) = \mathcal{N}(0, I) \quad \text{（事前分布）} \\\ x &\sim p_\theta(x|z) \quad \text{（尤度）} \end{align} $$ 
  * **推論過程（Inference）** ： $$ q_\phi(z|x) \approx p(z|x) \quad \text{（変分近似）} $$ 

### ELBO（Evidence Lower Bound）

VAEの目的は、データの対数尤度 $\log p_\theta(x)$ を最大化することですが、これは計算困難です。そこで**ELBO** を最大化します：

$$ \begin{align} \log p_\theta(x) &\geq \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \| p(z)) \\\ &= \text{ELBO}(\theta, \phi; x) \end{align} $$ 

ここで：

  * **第1項** ：再構成項（Reconstruction term）- Decoderの復元性能
  * **第2項** ：KL正則化項（KL divergence）- Encoderの分布を事前分布に近づける

### KL Divergenceの解析解

EncoderとDecoderがガウス分布の場合、KL divergenceは解析的に計算できます：

$$ D_{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{J}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right) $$ 

ここで、$J$は潜在次元数、$\mu_j$と$\sigma_j^2$はEncoderが出力する平均と分散です。

### ELBOの導出と可視化
    
    
    import torch
    import torch.nn.functional as F
    
    def kl_divergence_gaussian(mu, logvar):
        """
        ガウス分布間のKL divergenceを計算
    
        Args:
            mu: (batch, latent_dim) - Encoderが出力する平均
            logvar: (batch, latent_dim) - Encoderが出力する対数分散
    
        Returns:
            kl_loss: (batch,) - 各サンプルのKL divergence
        """
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl
    
    
    def vae_loss(x_recon, x, mu, logvar):
        """
        VAEの損失関数（ELBO）
    
        Args:
            x_recon: (batch, input_dim) - 復元された入力
            x: (batch, input_dim) - 元の入力
            mu: (batch, latent_dim) - 平均
            logvar: (batch, latent_dim) - 対数分散
    
        Returns:
            total_loss: スカラー - ELBO（負の値）
            recon_loss: スカラー - 再構成誤差
            kl_loss: スカラー - KL divergence
        """
        # 再構成誤差（Binary Cross Entropy）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
        # KL divergence
        kl_loss = kl_divergence_gaussian(mu, logvar).sum()
    
        # ELBO = 再構成項 - KL項（最大化 = 負の最小化）
        total_loss = recon_loss + kl_loss
    
        return total_loss, recon_loss, kl_loss
    
    
    # 数値例でELBOを計算
    print("=== ELBOの数値例 ===")
    
    batch_size = 32
    input_dim = 784
    latent_dim = 20
    
    # ダミーデータ
    x = torch.rand(batch_size, input_dim)
    x_recon = torch.rand(batch_size, input_dim)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    
    print(f"総損失（ELBO）: {total_loss.item():.2f}")
    print(f"再構成誤差: {recon_loss.item():.2f}")
    print(f"KL divergence: {kl_loss.item():.2f}")
    print(f"\nバッチ平均:")
    print(f"  再構成誤差: {recon_loss.item()/batch_size:.2f}")
    print(f"  KL divergence: {kl_loss.item()/batch_size:.2f}")
    print("\n→ 2つの項のバランスが重要！")
    

### AutoencoderとVAEの比較

項目 | Autoencoder | VAE  
---|---|---  
**潜在変数** | 決定論的 $z = f(x)$ | 確率的 $z \sim q_\phi(z|x)$  
**目的関数** | 再構成誤差のみ | ELBO（再構成 + KL）  
**潜在空間** | 構造なし | 正規分布に正則化  
**生成能力** | 弱い | 強い（サンプリング可能）  
**補間** | 不安定 | 滑らか  
**訓練** | シンプル | Reparameterization Trick必要  
  
> 「VAEはAutoencoderに確率的な枠組みを加えることで、生成モデルとしての能力を大幅に向上させます」

* * *

## 2.3 Reparameterization Trick

### なぜReparameterization Trickが必要か

VAEの訓練では、$z \sim q_\phi(z|x)$ からサンプリングする必要があります。しかし、**確率的サンプリングは微分不可能** なため、そのままでは誤差逆伝播できません。
    
    
    ```mermaid
    graph LR
        X["x"] --> E["Encoder"]
        E --> Mu["μ"]
        E --> Sigma["σ"]
        Mu --> Sample["z ~ N(μ, σ²)"]
        Sigma --> Sample
        Sample -.->|勾配が流れない!| D["Decoder"]
    
        style Sample fill:#ffccbc
    ```

### Reparameterization Trickの仕組み

サンプリングを以下のように**決定論的変換** に書き換えます：

$$ \begin{align} z &\sim \mathcal{N}(\mu, \sigma^2) \\\ &\Downarrow \text{（Reparameterization）} \\\ z &= \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) \end{align} $$ 

ここで、$\epsilon$はパラメータに依存しない**ノイズ** です。この変換により、$\mu$と$\sigma$を通じて勾配が流れます。
    
    
    ```mermaid
    graph LR
        X["x"] --> E["Encoder"]
        E --> Mu["μ"]
        E --> Sigma["σ"]
        Epsilon["ϵ ~ N(0,1)"] --> Reparam["z = μ + σ·ϵ"]
        Mu --> Reparam
        Sigma --> Reparam
        Reparam -->|勾配が流れる!| D["Decoder"]
    
        style Reparam fill:#c5e1a5
        style Epsilon fill:#fff59d
    ```

### Reparameterization Trickの実装
    
    
    import torch
    import torch.nn as nn
    
    class VAEEncoder(nn.Module):
        """VAEのEncoder（Recognition network）"""
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
                mu: (batch, latent_dim) - 平均
                logvar: (batch, latent_dim) - 対数分散
            """
            h = torch.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
    
    def reparameterize(mu, logvar):
        """
        Reparameterization Trick
    
        Args:
            mu: (batch, latent_dim) - 平均
            logvar: (batch, latent_dim) - 対数分散
    
        Returns:
            z: (batch, latent_dim) - サンプリングされた潜在変数
        """
        # 標準偏差を計算（数値安定性のため対数分散を使用）
        std = torch.exp(0.5 * logvar)
    
        # 標準正規分布からノイズをサンプリング
        epsilon = torch.randn_like(std)
    
        # z = μ + σ·ϵ
        z = mu + std * epsilon
    
        return z
    
    
    # 動作確認
    print("=== Reparameterization Trickの動作 ===")
    
    encoder = VAEEncoder(input_dim=784, hidden_dim=400, latent_dim=20)
    x = torch.randn(32, 784)
    
    # Encoderで平均と分散を取得
    mu, logvar = encoder(x)
    print(f"平均: {mu.shape}")
    print(f"対数分散: {logvar.shape}")
    
    # Reparameterization Trickでサンプリング
    z = reparameterize(mu, logvar)
    print(f"サンプリングされた潜在変数: {z.shape}")
    
    # 勾配の確認
    print("\n=== 勾配の流れを確認 ===")
    z.sum().backward()
    print(f"μの勾配: {encoder.fc_mu.weight.grad is not None}")
    print(f"log(σ²)の勾配: {encoder.fc_logvar.weight.grad is not None}")
    print("→ Reparameterization Trickにより勾配が流れる！")
    
    # 同じ入力から複数回サンプリング（確率的）
    print("\n=== 確率的サンプリングの確認 ===")
    z1 = reparameterize(mu, logvar)
    z2 = reparameterize(mu, logvar)
    z3 = reparameterize(mu, logvar)
    
    print(f"サンプル1（最初の5次元）: {z1[0, :5].detach().numpy()}")
    print(f"サンプル2（最初の5次元）: {z2[0, :5].detach().numpy()}")
    print(f"サンプル3（最初の5次元）: {z3[0, :5].detach().numpy()}")
    print("→ 同じ入力でも異なるサンプルが得られる（多様性）")
    

### 数値安定性の考慮
    
    
    import torch
    import torch.nn as nn
    
    # 対数分散を使う理由
    print("\n=== 数値安定性の重要性 ===")
    
    # 悪い例：直接分散を使う
    sigma_bad = torch.tensor([0.001, 1.0, 100.0])
    print(f"分散（直接）: {sigma_bad}")
    print(f"→ 極端な値が存在し、数値不安定")
    
    # 良い例：対数分散を使う
    logvar_good = torch.log(sigma_bad)
    print(f"\n対数分散: {logvar_good}")
    print(f"→ 数値的に安定した範囲")
    
    # 復元
    sigma_recovered = torch.exp(0.5 * logvar_good)
    print(f"\n標準偏差（復元）: {sigma_recovered}")
    print(f"→ 元の値が正確に復元される")
    
    # クリッピングでさらに安定化
    logvar_clipped = torch.clamp(logvar_good, min=-10, max=10)
    print(f"\nクリッピング後: {logvar_clipped}")
    print("→ 極端な値を防ぐ")
    

* * *

## 2.4 VAEアーキテクチャの完全実装

### EncoderとDecoderの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class VAE(nn.Module):
        """完全なVAEモデル"""
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super(VAE, self).__init__()
    
            self.input_dim = input_dim
            self.latent_dim = latent_dim
    
            # ===== Encoder（Recognition network） =====
            self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
            self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
            # ===== Decoder（Generative network） =====
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
                x_recon: (batch, input_dim) - 復元された入力
                mu: (batch, latent_dim) - 平均
                logvar: (batch, latent_dim) - 対数分散
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
            潜在空間からサンプリングして画像を生成
    
            Args:
                num_samples: 生成するサンプル数
                device: デバイス
    
            Returns:
                samples: (num_samples, input_dim)
            """
            # 標準正規分布から潜在変数をサンプリング
            z = torch.randn(num_samples, self.latent_dim).to(device)
    
            # Decoderで生成
            samples = self.decode(z)
    
            return samples
    
    
    # モデル作成
    print("=== VAEモデルの作成 ===")
    vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    # 動作確認
    batch_size = 32
    x = torch.rand(batch_size, 784)  # 0-1に正規化されたMNIST画像
    
    x_recon, mu, logvar = vae(x)
    
    print(f"入力: {x.shape}")
    print(f"復元: {x_recon.shape}")
    print(f"平均: {mu.shape}")
    print(f"対数分散: {logvar.shape}")
    
    # 損失計算
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    print(f"\n損失:")
    print(f"  総損失: {total_loss.item():.2f}")
    print(f"  再構成: {recon_loss.item():.2f}")
    print(f"  KL divergence: {kl_loss.item():.2f}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    
    # ランダムサンプリング
    print("\n=== ランダムサンプリング ===")
    samples = vae.sample(num_samples=10)
    print(f"生成されたサンプル: {samples.shape}")
    print("→ VAEは潜在空間から新しい画像を生成可能！")
    

### 畳み込みVAE（Convolutional VAE）
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConvVAE(nn.Module):
        """畳み込みVAE（画像用）"""
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
    
            # 潜在変数のパラメータ
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
    
    
    # モデル作成
    print("\n=== 畳み込みVAEの作成 ===")
    conv_vae = ConvVAE(input_channels=1, latent_dim=128)
    
    # 動作確認
    batch_size = 16
    x_img = torch.rand(batch_size, 1, 28, 28)  # MNIST画像
    
    x_recon, mu, logvar = conv_vae(x_img)
    
    print(f"入力画像: {x_img.shape}")
    print(f"復元画像: {x_recon.shape}")
    print(f"潜在変数: {mu.shape}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in conv_vae.parameters())
    print(f"総パラメータ数: {total_params:,}")
    print("→ 畳み込み層で画像の空間構造を保持")
    

* * *

## 2.5 PyTorchでの訓練とMNIST画像生成

### データセットの準備
    
    
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # MNISTデータセットの準備
    print("=== MNISTデータセットの準備 ===")
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # 0-1に正規化
    ])
    
    # ダウンロード（初回のみ）
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
    
    print(f"訓練データ数: {len(train_dataset)}")
    print(f"テストデータ数: {len(test_dataset)}")
    print(f"バッチサイズ: {batch_size}")
    print(f"訓練バッチ数: {len(train_loader)}")
    

### 訓練ループの実装
    
    
    import torch
    import torch.optim as optim
    
    def train_epoch(model, train_loader, optimizer, device):
        """1エポックの訓練"""
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
    
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
    
            # 画像を平坦化（Fully-connected VAEの場合）
            data_flat = data.view(data.size(0), -1)
    
            # Forward
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data_flat)
    
            # 損失計算
            total_loss, recon_loss, kl_loss = vae_loss(x_recon, data_flat, mu, logvar)
    
            # Backward
            total_loss.backward()
            optimizer.step()
    
            # 累積
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
    
        # 平均損失
        num_samples = len(train_loader.dataset)
        avg_loss = train_loss / num_samples
        avg_recon = train_recon_loss / num_samples
        avg_kl = train_kl_loss / num_samples
    
        return avg_loss, avg_recon, avg_kl
    
    
    def test_epoch(model, test_loader, device):
        """テストデータでの評価"""
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
    
    
    # 訓練設定
    print("\n=== VAEの訓練 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")
    
    # モデル作成
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    
    # オプティマイザ
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 訓練
    num_epochs = 10
    print(f"エポック数: {num_epochs}\n")
    
    for epoch in range(1, num_epochs + 1):
        # 訓練
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
    
        # テスト
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device)
    
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
    
    print("\n訓練完了！")
    

### 画像生成と可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_reconstruction(model, test_loader, device, num_samples=10):
        """再構成結果を可視化"""
        model.eval()
    
        # テストデータから1バッチ取得
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        data_flat = data.view(data.size(0), -1)
    
        # 再構成
        with torch.no_grad():
            x_recon, mu, logvar = model(data_flat)
    
        # CPUに移動
        data = data.cpu().numpy()
        x_recon = x_recon.view(-1, 1, 28, 28).cpu().numpy()
    
        # 可視化
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples, 2))
    
        for i in range(num_samples):
            # 元画像
            axes[0, i].imshow(data[i, 0], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
    
            # 再構成画像
            axes[1, i].imshow(x_recon[i, 0], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
    
        plt.tight_layout()
        plt.savefig('vae_reconstruction.png', dpi=150, bbox_inches='tight')
        print("再構成結果を保存: vae_reconstruction.png")
        plt.close()
    
    
    def visualize_samples(model, device, num_samples=16):
        """ランダムサンプリングで生成した画像を可視化"""
        model.eval()
    
        # ランダムサンプリング
        with torch.no_grad():
            samples = model.sample(num_samples, device)
    
        samples = samples.view(-1, 1, 28, 28).cpu().numpy()
    
        # 可視化
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
    
        plt.suptitle('Randomly Generated Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig('vae_samples.png', dpi=150, bbox_inches='tight')
        print("生成サンプルを保存: vae_samples.png")
        plt.close()
    
    
    # 可視化実行
    print("\n=== 結果の可視化 ===")
    visualize_reconstruction(model, test_loader, device, num_samples=10)
    visualize_samples(model, device, num_samples=16)
    print("→ VAEは元画像を復元し、新しい画像を生成できる！")
    

* * *

## 2.6 潜在空間の可視化と補間

### 2次元潜在空間の可視化
    
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_latent_space_2d(model, test_loader, device):
        """2次元潜在空間を可視化（latent_dim=2の場合）"""
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
    
        # 散布図
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('2D Latent Space Visualization (MNIST)')
        plt.grid(True, alpha=0.3)
        plt.savefig('vae_latent_2d.png', dpi=150, bbox_inches='tight')
        print("2D潜在空間を保存: vae_latent_2d.png")
        plt.close()
    
    
    # 2次元VAEで実験
    print("\n=== 2次元潜在空間の可視化 ===")
    model_2d = VAE(input_dim=784, hidden_dim=400, latent_dim=2).to(device)
    
    # 簡易訓練（省略可能 - 事前訓練済みモデルを使用）
    optimizer_2d = optim.Adam(model_2d.parameters(), lr=1e-3)
    for epoch in range(3):
        train_loss, _, _ = train_epoch(model_2d, train_loader, optimizer_2d, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
    
    # 可視化
    visualize_latent_space_2d(model_2d, test_loader, device)
    print("→ 異なる数字がクラスタを形成していることを確認")
    

### 潜在空間の補間（Interpolation）
    
    
    import torch
    import matplotlib.pyplot as plt
    
    def interpolate_latent_space(model, z_start, z_end, num_steps=10):
        """
        潜在空間で2点を補間
    
        Args:
            model: VAEモデル
            z_start: (latent_dim,) - 開始点
            z_end: (latent_dim,) - 終了点
            num_steps: 補間ステップ数
    
        Returns:
            interpolated_images: (num_steps, 1, 28, 28)
        """
        model.eval()
    
        # 線形補間
        alphas = torch.linspace(0, 1, num_steps)
        z_interp = torch.stack([
            (1 - alpha) * z_start + alpha * z_end
            for alpha in alphas
        ])
    
        # Decoderで生成
        with torch.no_grad():
            images = model.decode(z_interp)
    
        images = images.view(-1, 1, 28, 28)
    
        return images
    
    
    def visualize_interpolation(model, test_loader, device, num_pairs=3, num_steps=10):
        """補間結果を可視化"""
        model.eval()
    
        # テストデータから画像を選択
        data, _ = next(iter(test_loader))
        data = data[:num_pairs*2].to(device)
        data_flat = data.view(data.size(0), -1)
    
        # Encodeして潜在変数を取得
        with torch.no_grad():
            mu, _ = model.encode(data_flat)
    
        # 可視化
        fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps, num_pairs))
    
        for i in range(num_pairs):
            z_start = mu[i*2]
            z_end = mu[i*2 + 1]
    
            # 補間
            images = interpolate_latent_space(model, z_start, z_end, num_steps)
            images = images.cpu().numpy()
    
            for j in range(num_steps):
                ax = axes[i, j] if num_pairs > 1 else axes[j]
                ax.imshow(images[j, 0], cmap='gray')
                ax.axis('off')
    
        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
        print("補間結果を保存: vae_interpolation.png")
        plt.close()
    
    
    # 補間の可視化
    print("\n=== 潜在空間の補間 ===")
    visualize_interpolation(model, test_loader, device, num_pairs=3, num_steps=10)
    print("→ 滑らかに変化する中間画像が生成される")
    

### 潜在空間の構造を探索
    
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    def visualize_latent_manifold(model, device, n=20, digit_size=28):
        """
        2次元潜在空間のマニフォールドを可視化
    
        Args:
            model: 2次元潜在変数を持つVAE
            device: デバイス
            n: グリッドサイズ
            digit_size: 画像サイズ
        """
        model.eval()
    
        # グリッド範囲（正規分布の99%をカバー）
        grid_range = 3
        grid_x = np.linspace(-grid_range, grid_range, n)
        grid_y = np.linspace(-grid_range, grid_range, n)
    
        # 全体画像の準備
        figure = np.zeros((digit_size * n, digit_size * n))
    
        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    # 潜在変数
                    z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
    
                    # Decoderで生成
                    x_recon = model.decode(z)
                    digit = x_recon.view(digit_size, digit_size).cpu().numpy()
    
                    # 配置
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit
    
        # 可視化
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.axis('off')
        plt.title('Latent Space Manifold (2D)', fontsize=14)
        plt.savefig('vae_manifold.png', dpi=150, bbox_inches='tight')
        print("潜在空間マニフォールドを保存: vae_manifold.png")
        plt.close()
    
    
    # マニフォールドの可視化
    print("\n=== 潜在空間マニフォールド ===")
    visualize_latent_manifold(model_2d, device, n=20, digit_size=28)
    print("→ 2次元空間で数字がどのように分布しているかを確認")
    

* * *

## 2.7 実践：CelebA顔画像生成

### CelebA用畳み込みVAE
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CelebAVAE(nn.Module):
        """CelebA顔画像用のVAE（64x64解像度）"""
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
    
    
    # モデル作成
    print("\n=== CelebA VAEの作成 ===")
    celeba_vae = CelebAVAE(input_channels=3, latent_dim=256)
    
    # 動作確認
    batch_size = 16
    x_celeba = torch.rand(batch_size, 3, 64, 64)  # RGB 64x64画像
    
    x_recon, mu, logvar = celeba_vae(x_celeba)
    
    print(f"入力画像: {x_celeba.shape}")
    print(f"復元画像: {x_recon.shape}")
    print(f"潜在変数: {mu.shape}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in celeba_vae.parameters())
    print(f"総パラメータ数: {total_params:,}")
    print("→ 高解像度画像用の深いアーキテクチャ")
    

* * *

## 演習問題

**演習1：β-VAEの実装**

KL項に重み$\beta$を導入した**β-VAE** を実装し、$\beta$の値による潜在空間の違いを分析してください。
    
    
    import torch
    
    # TODO: β-VAEの損失関数を実装
    # Loss = Reconstruction + β * KL divergence
    
    # TODO: β = 0.5, 1.0, 2.0, 4.0 で訓練
    # TODO: 潜在空間の可視化で disentanglement（分離性）を評価
    # ヒント: βが大きいほど潜在変数が独立になる
    

**演習2：潜在次元数の影響を調査**

潜在次元数（2, 10, 20, 50, 100）を変えて、再構成品質と生成多様性のトレードオフを調査してください。
    
    
    import torch
    
    # TODO: 異なる潜在次元数でモデルを訓練
    # TODO: 再構成誤差、KL divergence、生成画像の品質を比較
    # TODO: 次元数 vs 性能のグラフを作成
    # 期待: 次元数が多いほど再構成は良いが、生成の多様性が低下
    

**演習3：条件付きVAE（CVAE）の実装**

クラスラベルを条件として与える**Conditional VAE** を実装してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: EncoderとDecoderにクラスラベルを入力
    # TODO: 指定したクラスの画像を生成できることを確認
    # TODO: 異なるクラス間の補間を可視化
    

**演習4：再構成誤差の比較（MSE vs BCE）**

再構成誤差として、MSE（平均二乗誤差）とBCE（Binary Cross Entropy）を比較してください。
    
    
    import torch
    import torch.nn.functional as F
    
    # TODO: 2つの損失関数でモデルを訓練
    # TODO: 再構成画像の品質を視覚的に比較
    # TODO: 損失の収束速度と最終値を記録
    # 分析: BCEは確率的解釈があり、MNISTのような2値画像に適している
    

**演習5：KL Annealingの実装**

訓練初期はKL項の重みを小さくし、徐々に増やす**KL Annealing** を実装してください。
    
    
    import torch
    
    # TODO: KL項の重みをエポックごとに線形増加
    # weight = min(1.0, epoch / num_annealing_epochs)
    
    # TODO: Annealingあり・なしで収束速度を比較
    # TODO: 最終的な潜在空間の品質を評価
    # 期待: Annealingにより訓練が安定し、より良い潜在空間が得られる
    

* * *

## まとめ

この章では、VAE（Variational Autoencoder）の理論と実装を学びました。

### 重要ポイント

  * **Autoencoderの限界** ：決定論的で構造化されていない潜在空間
  * **VAEの動機** ：確率的枠組みで生成モデルとしての能力を獲得
  * **ELBO** ：再構成項とKL正則化項の2つで構成される目的関数
  * **KL Divergence** ：潜在変数を標準正規分布に正則化
  * **Reparameterization Trick** ：確率的サンプリングを微分可能にする技術
  * **Encoder/Decoder** ：ガウス分布のパラメータ（平均・分散）を出力
  * **潜在空間** ：構造化されており、補間や探索が可能
  * **実装** ：MNIST/CelebA画像生成システムの完全実装
  * **応用** ：β-VAE、CVAE、KL Annealingなどの拡張技術

### 次のステップ

次章では、**GAN（Generative Adversarial Networks）** について学びます。敵対的学習によるより高品質な画像生成、Generator/Discriminatorのアーキテクチャ、訓練の安定化技術など、VAEとは異なるアプローチの生成モデルを習得します。
