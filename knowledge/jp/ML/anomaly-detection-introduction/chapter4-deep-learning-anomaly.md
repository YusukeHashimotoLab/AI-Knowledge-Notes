---
title: 第4章：深層学習による異常検知
chapter_title: 第4章：深層学習による異常検知
subtitle: Autoencoder、VAE、GAN、時系列異常検知
reading_time: 80-90分
difficulty: 中級〜上級
code_examples: 9
exercises: 6
---

## 学習目標

  * Autoencoderによる異常検知の原理と実装を理解する
  * Variational Autoencoder (VAE) の確率的アプローチを学ぶ
  * GAN-based異常検知（AnoGAN）の仕組みを理解する
  * LSTM Autoencoderで時系列異常検知を実装する
  * エンドツーエンドの異常検知パイプラインを構築できる

## 4.1 Autoencoderによる異常検知

### 4.1.1 Autoencoderの基礎

**Autoencoder（自己符号化器）** は、入力データを圧縮し、再構成する教師なし学習モデルです。正常データで訓練することで、異常データの再構成誤差が大きくなることを利用して異常検知を行います。

**アーキテクチャ：**
    
    
    Input (x)
        ↓
    Encoder: x → z (潜在表現)
        ↓
    Latent Space (z)
        ↓
    Decoder: z → x̂ (再構成)
        ↓
    Reconstruction Error: ||x - x̂||²
    

**異常検知の原理：**

  * 正常データで訓練: 正常パターンを学習
  * 再構成誤差が小さい: 正常データ
  * 再構成誤差が大きい: 異常データ（学習していないパターン）

**数式表現：**

$$ \text{Anomaly Score} = \|x - \text{Decoder}(\text{Encoder}(x))\|^2 $$

### 4.1.2 再構成誤差とThreshold選択

異常判定には、再構成誤差に対するthresholdを設定します。

**Threshold設定手法：**

手法 | 説明 | 適用場面  
---|---|---  
百分位数法 | 訓練データの再構成誤差の95%点 | 正常データのみで学習  
統計的手法 | 平均 + 3σ | 正規分布を仮定  
ROC曲線 | 検証データでAUC最大化 | 少量の異常ラベルあり  
業務要件 | False Positive率を指定 | 実運用重視  
  
### 4.1.3 PyTorch実装（完全版）
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # Autoencoderモデル定義
    class Autoencoder(nn.Module):
        """シンプルなAutoencoder"""
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
    
            # Decoder（Encoderの逆順）
            decoder_layers = []
            for i in range(len(hidden_dims) - 1, 0, -1):
                decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
    
            decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
            decoder_layers.append(nn.Sigmoid())  # 出力を[0,1]に正規化
    
            self.decoder = nn.Sequential(*decoder_layers)
    
        def forward(self, x):
            """順伝播"""
            z = self.encoder(x)  # エンコード
            x_reconstructed = self.decoder(z)  # デコード
            return x_reconstructed
    
        def encode(self, x):
            """潜在表現の取得"""
            return self.encoder(x)
    
    
    def train_autoencoder(model, train_loader, n_epochs=50, lr=0.001):
        """Autoencoderの訓練"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()  # 再構成誤差（Mean Squared Error）
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
        """再構成誤差の計算"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    
        errors = []
    
        with torch.no_grad():
            for batch_x, in data_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
    
                # サンプルごとの再構成誤差（MSE）
                batch_errors = torch.mean((batch_x - reconstructed) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())
    
        return np.array(errors)
    
    
    def detect_anomalies(model, test_loader, threshold):
        """異常検知の実行"""
        errors = compute_reconstruction_errors(model, test_loader)
        predictions = (errors > threshold).astype(int)
        return predictions, errors
    
    
    # 使用例
    if __name__ == "__main__":
        # サンプルデータ生成（正規分布の正常データ）
        np.random.seed(42)
        torch.manual_seed(42)
    
        # 正常データ（28x28 = 784次元）
        n_normal = 1000
        normal_data = np.random.randn(n_normal, 784) * 0.5 + 0.5
        normal_data = np.clip(normal_data, 0, 1)
    
        # 異常データ（正常とは異なる分布）
        n_anomaly = 50
        anomaly_data = np.random.uniform(0, 1, (n_anomaly, 784))
    
        # PyTorch Dataset
        train_dataset = TensorDataset(torch.FloatTensor(normal_data))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
        test_data = np.vstack([normal_data[:100], anomaly_data])
        test_labels = np.array([0] * 100 + [1] * n_anomaly)  # 0: 正常, 1: 異常
    
        test_dataset = TensorDataset(torch.FloatTensor(test_data))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
        # モデル訓練
        print("=== Autoencoder訓練開始 ===")
        model = Autoencoder(input_dim=784, hidden_dims=[256, 128, 64])
        trained_model, losses = train_autoencoder(model, train_loader, n_epochs=50, lr=0.001)
    
        # Threshold設定（訓練データの95%点）
        train_errors = compute_reconstruction_errors(trained_model, train_loader)
        threshold = np.percentile(train_errors, 95)
        print(f"\n閾値（95%点）: {threshold:.6f}")
    
        # テストデータで異常検知
        predictions, test_errors = detect_anomalies(trained_model, test_loader, threshold)
    
        # 評価
        from sklearn.metrics import classification_report, roc_auc_score
    
        print("\n=== 異常検知結果 ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
    
        auc_score = roc_auc_score(test_labels, test_errors)
        print(f"ROC-AUC: {auc_score:.3f}")
    
        # 可視化
        plt.figure(figsize=(12, 4))
    
        # 学習曲線
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
    
        # 再構成誤差分布
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
        print("\nグラフを保存しました: autoencoder_anomaly_detection.png")
    

### 4.1.4 ネットワークアーキテクチャの選択

**アーキテクチャ設計のポイント：**

要素 | 推奨値 | 理由  
---|---|---  
潜在次元 | 入力次元の10-30% | 過度な圧縮は情報損失、大きすぎると恒等写像  
隠れ層数 | 2-4層 | 深すぎると訓練困難、浅すぎると表現力不足  
活性化関数 | ReLU（隠れ層）、Sigmoid（出力） | 勾配消失を防ぐ、出力範囲を制限  
Dropout | 0.2-0.3 | 過学習防止（ただし異常検知では慎重に）  
  
* * *

## 4.2 Variational Autoencoder (VAE)

### 4.2.1 VAEの動機

**通常のAutoencoderの課題：**

  * 潜在空間が不連続で、意味のある構造を持たない
  * 学習データに過剰適合しやすい
  * 生成能力が限定的

**VAEの特徴：**

  * 潜在変数を確率分布としてモデル化
  * 正則化により滑らかな潜在空間を学習
  * 生成モデルとしても機能

### 4.2.2 VAEの数理

**確率的エンコーダ：**

$$ q_\phi(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)) $$

エンコーダは平均$\mu(x)$と分散$\sigma^2(x)$を出力します。

**デコーダ：**

$$ p_\theta(x|z) = \mathcal{N}(x; \mu_{\text{dec}}(z), \sigma^2_{\text{dec}}) $$

**損失関数（ELBO）：**

$$ \mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}} $$

  * 第1項: 再構成損失（Autoencoderと同じ）
  * 第2項: KLダイバージェンス（正則化項）、$p(z) = \mathcal{N}(0, I)$を仮定

**KLダイバージェンスの解析解：**

$$ D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2) $$

### 4.2.3 VAEによる異常検知

VAEでは、再構成誤差とKLダイバージェンスを組み合わせて異常スコアを計算します。

$$ \text{Anomaly Score} = \text{Reconstruction Error} + \beta \cdot D_{KL} $$

### 4.2.4 PyTorch実装
    
    
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
    
            # 潜在分布のパラメータ（平均と分散）
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
            """エンコード: 平均と対数分散を出力"""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
        def reparameterize(self, mu, logvar):
            """再パラメータ化トリック"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # N(0, 1)からサンプリング
            z = mu + eps * std
            return z
    
        def decode(self, z):
            """デコード"""
            return self.decoder(z)
    
        def forward(self, x):
            """順伝播"""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_reconstructed = self.decode(z)
            return x_reconstructed, mu, logvar
    
    
    def vae_loss(x, x_reconstructed, mu, logvar, beta=1.0):
        """VAE損失関数
    
        Args:
            beta: KLダイバージェンスの重み（β-VAE）
        """
        # Reconstruction loss（バイナリクロスエントロピー）
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        # Total loss
        total_loss = recon_loss + beta * kl_div
    
        return total_loss, recon_loss, kl_div
    
    
    def train_vae(model, train_loader, n_epochs=50, lr=0.001, beta=1.0):
        """VAEの訓練"""
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
        """VAEによる異常スコア計算"""
        model.eval()
        device = next(model.parameters()).device
    
        with torch.no_grad():
            x = x.to(device)
            x_recon, mu, logvar = model(x)
    
            # Reconstruction error（サンプルごと）
            recon_error = F.binary_cross_entropy(x_recon, x, reduction='none').sum(dim=1)
    
            # KL divergence（サンプルごと）
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
            # Anomaly score
            anomaly_scores = recon_error + beta * kl_div
    
        return anomaly_scores.cpu().numpy()
    
    
    # 使用例
    if __name__ == "__main__":
        # データ準備（前述と同じ）
        train_dataset = TensorDataset(torch.FloatTensor(normal_data))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
        # VAEモデル
        print("=== VAE訓練開始 ===")
        vae_model = VAE(input_dim=784, latent_dim=32, hidden_dims=[256, 128])
        trained_vae = train_vae(vae_model, train_loader, n_epochs=50, lr=0.001, beta=1.0)
    
        # 異常スコア計算
        test_tensor = torch.FloatTensor(test_data)
        anomaly_scores = vae_anomaly_score(trained_vae, test_tensor, beta=1.0)
    
        # Threshold設定と評価
        threshold = np.percentile(anomaly_scores[:100], 95)  # 正常データの95%点
        predictions = (anomaly_scores > threshold).astype(int)
    
        print("\n=== VAE異常検知結果 ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
        print(f"ROC-AUC: {roc_auc_score(test_labels, anomaly_scores):.3f}")
    

### 4.2.5 潜在空間の分析

VAEの潜在空間は、正常データが滑らかに分布しています。異常データは潜在空間の外れた領域に配置されることが期待されます。
    
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    def visualize_latent_space(model, data, labels):
        """潜在空間の可視化（2次元投影）"""
        model.eval()
        device = next(model.parameters()).device
    
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(device)
            mu, _ = model.encode(data_tensor)
            latent_codes = mu.cpu().numpy()
    
        # 2次元に圧縮（潜在次元が2より大きい場合）
        if latent_codes.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_codes)
        else:
            latent_2d = latent_codes
    
        # プロット
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
        print("潜在空間の可視化を保存しました: vae_latent_space.png")
    
    # 使用例
    visualize_latent_space(trained_vae, test_data, test_labels)
    

* * *

## 4.3 GAN-based異常検知

### 4.3.1 AnoGAN（Anomaly Detection with GAN）

**AnoGAN** は、GANを用いて正常データの生成モデルを学習し、テストデータがどの程度その生成分布から逸脱しているかで異常を検知します。

**訓練フェーズ：**

  * 正常データでGANを訓練
  * Generator Gが正常データの分布を学習

**テストフェーズ：**

  1. テストサンプル$x$に対して、潜在変数$z$を最適化: $G(z) \approx x$
  2. 異常スコアを計算: Residual Loss + Discrimination Loss

### 4.3.2 異常スコアの定義

$$ A(x) = (1 - \lambda) \cdot L_R(x) + \lambda \cdot L_D(x) $$

  * $L_R(x) = \|x - G(z^*)\|_1$: Residual Loss（再構成誤差）
  * $L_D(x) = \|f(x) - f(G(z^*))\|_1$: Discrimination Loss（特徴空間での距離）
  * $f(\cdot)$: Discriminatorの中間層の特徴

### 4.3.3 潜在変数の最適化

テストサンプル$x$に対して、$G(z) \approx x$となる$z$を勾配降下法で探索：

$$ z^* = \arg\min_z \|x - G(z)\|_1 + \lambda \|f(x) - f(G(z))\|_1 $$

### 4.3.4 実装概要
    
    
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
        """GAN Discriminator（中間層の特徴も取得）"""
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
        """テストサンプルxに対する最適な潜在変数zを探索"""
        device = next(generator.parameters()).device
    
        # 初期化
        z = torch.randn(x.size(0), generator.model[0].in_features, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)
    
        for i in range(n_iterations):
            optimizer.zero_grad()
    
            # 生成
            G_z = generator(z)
    
            # Residual Loss
            residual_loss = torch.mean(torch.abs(x - G_z))
    
            # Discrimination Loss（特徴空間での距離）
            _, feat_real = discriminator(x, return_features=True)
            _, feat_fake = discriminator(G_z, return_features=True)
            discrimination_loss = torch.mean(torch.abs(feat_real - feat_fake))
    
            # Total loss
            loss = (1 - lambda_weight) * residual_loss + lambda_weight * discrimination_loss
    
            loss.backward()
            optimizer.step()
    
        # 異常スコア
        with torch.no_grad():
            G_z_final = generator(z)
            residual = torch.mean(torch.abs(x - G_z_final), dim=1)
    
            _, feat_real = discriminator(x, return_features=True)
            _, feat_fake = discriminator(G_z_final, return_features=True)
            discrimination = torch.mean(torch.abs(feat_real - feat_fake), dim=1)
    
            anomaly_scores = (1 - lambda_weight) * residual + lambda_weight * discrimination
    
        return anomaly_scores.cpu().numpy()
    
    
    # 注意: GANの訓練コードは省略（標準的なGAN訓練を実施）
    # 実際の使用では、まずGANを正常データで訓練し、その後上記の関数で異常検知を行う
    

> **注意** : AnoGANは潜在変数の最適化に時間がかかるため、リアルタイム異常検知には不向きです。この問題を解決するために、Fast-AnoGANやEGBAdなどの改良手法が提案されています。

* * *

## 4.4 時系列異常検知

### 4.4.1 時系列データの特徴

時系列データの異常検知では、以下の特性を考慮する必要があります：

  * **時間的依存性** : 過去の値が未来に影響
  * **季節性・周期性** : 日次、週次、年次パターン
  * **トレンド** : 長期的な上昇・下降傾向
  * **多変量性** : 複数のセンサー値が相互に関連

### 4.4.2 LSTM Autoencoder

**LSTM Autoencoder** は、LSTMを用いて時系列の時間的パターンを学習し、再構成誤差で異常を検知します。

**アーキテクチャ：**
    
    
    Input: (batch, seq_len, features)
        ↓
    LSTM Encoder: 時系列を固定長ベクトルに圧縮
        ↓
    Latent Vector: (batch, latent_dim)
        ↓
    LSTM Decoder: 潜在ベクトルから時系列を再構成
        ↓
    Output: (batch, seq_len, features)
    

### 4.4.3 PyTorch実装
    
    
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
    
            # 潜在表現への圧縮
            self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
    
            # Decoder用のFC（潜在表現からLSTM初期状態へ）
            self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
    
            # Decoder LSTM
            self.decoder_lstm = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
    
            # 出力層
            self.output_fc = nn.Linear(hidden_dim, input_dim)
    
        def encode(self, x):
            """エンコード: 時系列 → 潜在ベクトル"""
            # x: (batch, seq_len, input_dim)
            lstm_out, (hidden, cell) = self.encoder_lstm(x)
    
            # 最後の隠れ状態を使用
            last_hidden = hidden[-1]  # (batch, hidden_dim)
    
            # 潜在ベクトルに圧縮
            z = self.encoder_fc(last_hidden)  # (batch, latent_dim)
    
            return z
    
        def decode(self, z, seq_len):
            """デコード: 潜在ベクトル → 時系列"""
            batch_size = z.size(0)
    
            # デコーダのLSTM初期状態
            hidden = self.decoder_fc(z).unsqueeze(0)  # (1, batch, hidden_dim)
            hidden = hidden.repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
            cell = torch.zeros_like(hidden)
    
            # デコーダの入力（潜在ベクトルをseq_len回繰り返し）
            decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
    
            # LSTM Decoder
            lstm_out, _ = self.decoder_lstm(decoder_input, (hidden, cell))
            # lstm_out: (batch, seq_len, hidden_dim)
    
            # 出力層
            output = self.output_fc(lstm_out)  # (batch, seq_len, input_dim)
    
            return output
    
        def forward(self, x):
            """順伝播"""
            seq_len = x.size(1)
    
            z = self.encode(x)
            x_reconstructed = self.decode(z, seq_len)
    
            return x_reconstructed
    
    
    def train_lstm_autoencoder(model, train_loader, n_epochs=50, lr=0.001):
        """LSTM Autoencoderの訓練"""
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
        """時系列異常検知"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    
        all_errors = []
    
        with torch.no_grad():
            for batch_x, in data_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
    
                # 系列全体の再構成誤差（平均）
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
    
        all_errors = np.array(all_errors)
        predictions = (all_errors > threshold).astype(int)
    
        return predictions, all_errors
    
    
    # 使用例
    if __name__ == "__main__":
        # サンプル時系列データ生成（正常：正弦波、異常：ノイズ）
        np.random.seed(42)
        torch.manual_seed(42)
    
        seq_len = 50
        input_dim = 5  # 5つのセンサー
    
        # 正常データ（正弦波ベース）
        n_normal_sequences = 500
        t = np.linspace(0, 4*np.pi, seq_len)
        normal_sequences = []
        for _ in range(n_normal_sequences):
            seq = np.array([np.sin(t + np.random.randn() * 0.1) for _ in range(input_dim)]).T
            seq += np.random.randn(seq_len, input_dim) * 0.1
            normal_sequences.append(seq)
    
        normal_sequences = np.array(normal_sequences)  # (n_normal, seq_len, input_dim)
    
        # 異常データ（ランダムノイズ）
        n_anomaly_sequences = 50
        anomaly_sequences = np.random.randn(n_anomaly_sequences, seq_len, input_dim)
    
        # Dataset
        train_dataset = TensorDataset(torch.FloatTensor(normal_sequences))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
        test_sequences = np.vstack([normal_sequences[:50], anomaly_sequences])
        test_labels = np.array([0] * 50 + [1] * n_anomaly_sequences)
    
        test_dataset = TensorDataset(torch.FloatTensor(test_sequences))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
        # モデル訓練
        print("=== LSTM Autoencoder訓練開始 ===")
        lstm_ae = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64, num_layers=2, latent_dim=32)
        trained_lstm_ae = train_lstm_autoencoder(lstm_ae, train_loader, n_epochs=50, lr=0.001)
    
        # Threshold設定
        train_errors = []
        trained_lstm_ae.eval()
        with torch.no_grad():
            for batch_x, in train_loader:
                reconstructed = trained_lstm_ae(batch_x)
                errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
                train_errors.extend(errors.cpu().numpy())
    
        threshold = np.percentile(train_errors, 95)
        print(f"\n閾値（95%点）: {threshold:.6f}")
    
        # 異常検知
        predictions, test_errors = detect_ts_anomalies(trained_lstm_ae, test_loader, threshold)
    
        print("\n=== LSTM Autoencoder異常検知結果 ===")
        print(classification_report(test_labels, predictions,
                                    target_names=['Normal', 'Anomaly']))
        print(f"ROC-AUC: {roc_auc_score(test_labels, test_errors):.3f}")
    

### 4.4.4 多変量時系列異常検知

複数のセンサーからのデータを同時に扱う場合、各変数間の相関関係も考慮する必要があります。

**手法：**

  * **LSTM Autoencoder** : 上記の実装で対応済み
  * **Attention機構** : どの変数が異常に寄与しているかを解釈
  * **Transformer** : 長期依存関係の学習

* * *

## 4.5 エンドツーエンド実践

### 4.5.1 データ準備

実世界の異常検知では、以下のステップでデータを準備します。
    
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    class AnomalyDetectionPipeline:
        """異常検知パイプライン"""
        def __init__(self, model_type='autoencoder'):
            self.model_type = model_type
            self.scaler = StandardScaler()
            self.model = None
            self.threshold = None
    
        def preprocess(self, data, fit_scaler=False):
            """前処理: 正規化、欠損値処理など"""
            # 欠損値補完（平均値）
            data = data.fillna(data.mean())
    
            # 標準化
            if fit_scaler:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = self.scaler.transform(data)
    
            return data_scaled
    
        def create_sequences(self, data, seq_len=50):
            """時系列データをシーケンスに分割"""
            sequences = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i+seq_len])
    
            return np.array(sequences)
    
        def train(self, normal_data, seq_len=50, n_epochs=50):
            """モデル訓練"""
            # 前処理
            normal_scaled = self.preprocess(normal_data, fit_scaler=True)
    
            # シーケンス作成
            if self.model_type in ['lstm_ae', 'transformer']:
                sequences = self.create_sequences(normal_scaled, seq_len)
                train_dataset = TensorDataset(torch.FloatTensor(sequences))
            else:
                # Autoencoderの場合
                train_dataset = TensorDataset(torch.FloatTensor(normal_scaled))
    
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
            # モデル選択と訓練
            if self.model_type == 'autoencoder':
                self.model = Autoencoder(input_dim=normal_scaled.shape[1])
                self.model, _ = train_autoencoder(self.model, train_loader, n_epochs)
            elif self.model_type == 'vae':
                self.model = VAE(input_dim=normal_scaled.shape[1])
                self.model = train_vae(self.model, train_loader, n_epochs)
            elif self.model_type == 'lstm_ae':
                self.model = LSTMAutoencoder(input_dim=normal_scaled.shape[1])
                self.model = train_lstm_autoencoder(self.model, train_loader, n_epochs)
    
            # Threshold設定（訓練データの95%点）
            if self.model_type == 'vae':
                scores = vae_anomaly_score(self.model, torch.FloatTensor(normal_scaled))
            else:
                scores = compute_reconstruction_errors(self.model, train_loader)
    
            self.threshold = np.percentile(scores, 95)
            print(f"閾値設定: {self.threshold:.6f}")
    
        def predict(self, test_data, seq_len=50):
            """異常予測"""
            # 前処理
            test_scaled = self.preprocess(test_data, fit_scaler=False)
    
            # シーケンス作成
            if self.model_type in ['lstm_ae', 'transformer']:
                sequences = self.create_sequences(test_scaled, seq_len)
                test_dataset = TensorDataset(torch.FloatTensor(sequences))
            else:
                test_dataset = TensorDataset(torch.FloatTensor(test_scaled))
    
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
            # 異常スコア計算
            if self.model_type == 'vae':
                scores = vae_anomaly_score(self.model, torch.FloatTensor(test_scaled))
            else:
                scores = compute_reconstruction_errors(self.model, test_loader)
    
            # 異常判定
            predictions = (scores > self.threshold).astype(int)
    
            return predictions, scores
    
    
    # 使用例
    if __name__ == "__main__":
        # 仮のデータフレーム
        normal_df = pd.DataFrame(np.random.randn(1000, 10))
        test_df = pd.DataFrame(np.random.randn(100, 10))
    
        # パイプライン
        pipeline = AnomalyDetectionPipeline(model_type='autoencoder')
        pipeline.train(normal_df, n_epochs=30)
    
        predictions, scores = pipeline.predict(test_df)
        print(f"\n異常検出数: {predictions.sum()} / {len(predictions)}")
    

### 4.5.2 モデル選択

データ特性に応じた適切なモデルを選択します。

データ種類 | 推奨モデル | 理由  
---|---|---  
画像データ | Convolutional AE、VAE | 空間構造を保持  
時系列データ | LSTM AE、Transformer | 時間的依存性を捉える  
表形式データ | Autoencoder、VAE | シンプルで効果的  
高次元スパース | Sparse AE、VAE | 次元削減と正則化  
  
### 4.5.3 Threshold調整

実運用では、業務要件に応じてThresholdを調整します。

  * **False Positive率を重視** : Thresholdを高く設定（誤検知を減らす）
  * **Recall重視** : Thresholdを低く設定（異常を見逃さない）
  * **F1最大化** : 検証データでF1スコアが最大となる点を選択

### 4.5.4 Production Deployment

**リアルタイム異常検知システムの構成：**
    
    
    データ収集（センサー、ログ）
        ↓
    前処理パイプライン（正規化、シーケンス化）
        ↓
    異常検知モデル（PyTorch → ONNX → TorchScript）
        ↓
    Threshold判定
        ↓
    アラート・可視化（Grafana、Slack通知）
    

**デプロイメントのポイント：**

  * **モデルの軽量化** : TorchScript、ONNX変換で推論高速化
  * **バッチ処理** : リアルタイム性が求められない場合はバッチで効率化
  * **定期的な再学習** : データ分布の変化に対応（Concept Drift）
  * **閾値の自動調整** : 運用データから適応的に調整

### 4.5.5 モニタリングとアラート
    
    
    import logging
    from datetime import datetime
    
    class AnomalyMonitor:
        """異常検知モニタリング"""
        def __init__(self, alert_threshold=0.9):
            self.alert_threshold = alert_threshold
            self.logger = self._setup_logger()
    
        def _setup_logger(self):
            logger = logging.getLogger('AnomalyDetection')
            logger.setLevel(logging.INFO)
    
            # ファイルハンドラ
            fh = logging.FileHandler('anomaly_detection.log')
            fh.setLevel(logging.INFO)
    
            # フォーマット
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
    
            logger.addHandler(fh)
            return logger
    
        def log_anomaly(self, timestamp, anomaly_score, features):
            """異常をログに記録"""
            self.logger.info(f"Anomaly detected - Time: {timestamp}, Score: {anomaly_score:.4f}")
            self.logger.info(f"Features: {features}")
    
        def send_alert(self, anomaly_score, message):
            """アラート送信（実装例）"""
            if anomaly_score > self.alert_threshold:
                # Slack、Email、PagerDutyなどに通知
                print(f"[ALERT] High anomaly detected: {message}")
                self.logger.warning(f"High severity alert: {message}")
    
        def monitor(self, pipeline, data_stream):
            """リアルタイムモニタリング"""
            for timestamp, data in data_stream:
                predictions, scores = pipeline.predict(data)
    
                if predictions.any():
                    self.log_anomaly(timestamp, scores.max(), data)
                    self.send_alert(scores.max(), f"Anomaly at {timestamp}")
    
    
    # 使用例（仮想データストリーム）
    monitor = AnomalyMonitor(alert_threshold=0.9)
    
    # 仮想データストリーム
    def data_stream_generator():
        for i in range(10):
            timestamp = datetime.now()
            data = pd.DataFrame(np.random.randn(1, 10))
            yield timestamp, data
    
    # モニタリング実行
    # monitor.monitor(pipeline, data_stream_generator())
    

* * *

## まとめ

本章で学んだこと：

  1. **Autoencoderによる異常検知:**
     * 再構成誤差で異常を検出
     * ネットワークアーキテクチャの設計
     * Threshold選択手法
     * PyTorchでの完全実装
  2. **Variational Autoencoder (VAE):**
     * 確率的潜在表現による異常検知
     * 再構成誤差 + KLダイバージェンス
     * 潜在空間の可視化と分析
     * β-VAEによる調整
  3. **GAN-based異常検知:**
     * AnoGANの原理と実装
     * 潜在変数の最適化
     * Discriminatorの特徴を利用
     * Fast-AnoGANなどの改良手法
  4. **時系列異常検知:**
     * LSTM Autoencoderの実装
     * 時間的パターンの学習
     * 多変量時系列への対応
     * シーケンスデータの処理
  5. **エンドツーエンド実践:**
     * データ前処理パイプライン
     * モデル選択の指針
     * Threshold調整手法
     * Production deploymentの設計
     * モニタリングとアラート

* * *

## 演習問題

**問1:** Autoencoderによる異常検知で、潜在次元を入力次元の10%に設定する理由を説明せよ。

**問2:** VAEの損失関数におけるKLダイバージェンス項の役割を、潜在空間の観点から説明せよ。

**問3:** AnoGANと通常のAutoencoderの異常検知における主な違いを3つ挙げよ。

**問4:** LSTM Autoencoderで時系列異常検知を行う際、シーケンス長をどのように決定すべきか、3つの観点から論じよ。

**問5:** 異常検知モデルのThreshold設定において、False Positive率が5%以下という業務要件がある場合、どのようにThresholdを決定すべきか具体的に説明せよ。

**問6:** リアルタイム異常検知システムを構築する際、考慮すべき技術的課題を5つ挙げ、それぞれの対処法を提案せよ。

* * *

## 参考文献

  1. Goodfellow, I. et al. "Deep Learning." MIT Press (2016).
  2. Kingma, D. P., & Welling, M. "Auto-Encoding Variational Bayes." _ICLR_ (2014).
  3. Schlegl, T. et al. "Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery." _IPMI_ (2017). [AnoGAN]
  4. Malhotra, P. et al. "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection." _ICML Anomaly Detection Workshop_ (2016).
  5. Park, D. et al. "A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder." _IEEE Robotics and Automation Letters_ (2018).
  6. Su, Y. et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." _KDD_ (2019).
  7. Vaswani, A. et al. "Attention is All You Need." _NeurIPS_ (2017). [Transformer]
  8. Audibert, J. et al. "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." _KDD_ (2020).

* * *
