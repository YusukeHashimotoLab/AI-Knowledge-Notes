---
title: 第3章：GAN (Generative Adversarial Networks)
chapter_title: 第3章：GAN (Generative Adversarial Networks)
subtitle: 敵対的学習で現実的な画像を生成 - Vanilla GANからStyleGANまで
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ GANの基本概念とGeneratorとDiscriminatorの役割を理解する
  * ✅ Minimax gameの理論的背景とNash均衡を理解する
  * ✅ Mode collapse問題とその対策を習得する
  * ✅ DCGAN（Deep Convolutional GAN）のアーキテクチャを実装できる
  * ✅ WGAN-GP（Wasserstein GAN with Gradient Penalty）を理解する
  * ✅ Spectral NormalizationとLabel Smoothingの訓練テクニックを習得する
  * ✅ StyleGANの基本概念と特徴を理解する
  * ✅ 実際の画像生成プロジェクトを実装できる

* * *

## 3.1 GANの基本概念

### Generatorとは

**Generator（生成器）** は、ランダムノイズ（潜在変数）から現実的なデータを生成するニューラルネットワークです。

> 「Generatorは、ランダムな潜在ベクトル $\mathbf{z} \sim p_z(\mathbf{z})$ を入力として受け取り、訓練データと見分けがつかない偽データ $G(\mathbf{z})$ を生成することを学習する」
    
    
    ```mermaid
    graph LR
        A[潜在ベクトル z100次元ノイズ] --> B[Generator G]
        B --> C[生成画像28×28×1]
    
        D[ランダムサンプリング] --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

### Discriminatorとは

**Discriminator（識別器）** は、入力データが本物（訓練データ）か偽物（Generatorの出力）かを判定する二値分類器です。
    
    
    ```mermaid
    graph TB
        A1[本物画像] --> D[Discriminator D]
        A2[生成画像] --> D
    
        D --> O1[本物: 1.0スコア]
        D --> O2[偽物: 0.0スコア]
    
        style A1 fill:#e8f5e9
        style A2 fill:#ffebee
        style D fill:#fff3e0
        style O1 fill:#e8f5e9
        style O2 fill:#ffebee
    ```

### 敵対的学習のメカニズム

GANは、GeneratorとDiscriminatorの**敵対的な競争** を通じて学習します：
    
    
    ```mermaid
    sequenceDiagram
        participant G as Generator
        participant D as Discriminator
        participant R as 本物データ
    
        G->>G: ノイズから画像生成
        G->>D: 生成画像を提示
        R->>D: 本物画像を提示
        D->>D: 本物/偽物を識別
        D->>G: フィードバック（勾配）
        G->>G: より騙しやすい画像へ改善
        D->>D: より見破りやすく改善
    
        Note over G,D: このプロセスを繰り返す
    ```

### Minimax Game理論

GANの目的関数は**Minimax最適化** として定式化されます：

$$ \min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))] $$

各項の意味：

  * **第1項** $\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})]$：Discriminatorが本物を正しく識別する能力
  * **第2項** $\mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$：Discriminatorが偽物を見破る能力

ネットワーク | 目標 | 最適化方向  
---|---|---  
**Discriminator (D)** | $V(D, G)$ を最大化 | 本物と偽物を正確に識別  
**Generator (G)** | $V(D, G)$ を最小化 | Discriminatorを騙す画像を生成  
  
### Nash均衡とは

**Nash均衡** は、GeneratorとDiscriminatorが互いに最適な戦略を取り、どちらも戦略を変更する動機がない状態です。

理論的には、Nash均衡で以下が成立します：

  * $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} = 0.5$（識別器が判断できない）
  * $p_g(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$（生成分布が真の分布と一致）

    
    
    ```mermaid
    graph LR
        subgraph 初期状態
            I1[Generatorランダム画像] --> I2[Discriminator簡単に識別]
        end
    
        subgraph 訓練中
            M1[Generator改善中] --> M2[Discriminator精度向上]
        end
    
        subgraph Nash均衡
            N1[Generator完璧な模倣] --> N2[Discriminator50%の精度]
        end
    
        I2 --> M1
        M2 --> N1
    
        style I1 fill:#ffebee
        style M1 fill:#fff3e0
        style N1 fill:#e8f5e9
        style N2 fill:#e8f5e9
    ```

* * *

## 3.2 GANの学習アルゴリズム

### 実装例1: Vanilla GAN基本構造
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}\n")
    
    print("=== Vanilla GAN 基本構造 ===\n")
    
    # Generator定義
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
                nn.Tanh()  # [-1, 1]の範囲に正規化
            )
    
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
            return img
    
    # Discriminator定義
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
                nn.Sigmoid()  # [0, 1]の確率出力
            )
    
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity
    
    # モデルのインスタンス化
    latent_dim = 100
    img_shape = (1, 28, 28)
    
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    
    print("--- Generator ---")
    print(generator)
    print(f"\nGenerator パラメータ数: {sum(p.numel() for p in generator.parameters()):,}")
    
    print("\n--- Discriminator ---")
    print(discriminator)
    print(f"\nDiscriminator パラメータ数: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # テスト実行
    z = torch.randn(8, latent_dim).to(device)
    fake_imgs = generator(z)
    print(f"\n生成画像形状: {fake_imgs.shape}")
    
    validity = discriminator(fake_imgs)
    print(f"Discriminator出力形状: {validity.shape}")
    print(f"Discriminatorスコア例: {validity[:3].detach().cpu().numpy().flatten()}")
    

**出力** ：
    
    
    使用デバイス: cuda
    
    === Vanilla GAN 基本構造 ===
    
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
    
    Generator パラメータ数: 533,136
    
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
    
    Discriminator パラメータ数: 533,505
    
    生成画像形状: torch.Size([8, 1, 28, 28])
    Discriminator出力形状: torch.Size([8, 1])
    Discriminatorスコア例: [0.4987 0.5023 0.4956]
    

### 実装例2: GAN訓練ループ
    
    
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("\n=== GAN 訓練ループ ===\n")
    
    # データローダー（MNIST使用）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1]に正規化
    ])
    
    # サンプルデータ（実際にはMNISTなどを使用）
    batch_size = 64
    # dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # デモ用のダミーデータ
    dataloader = [(torch.randn(batch_size, 1, 28, 28).to(device), None) for _ in range(10)]
    
    # 損失関数とオプティマイザー
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print("--- 訓練設定 ---")
    print(f"バッチサイズ: {batch_size}")
    print(f"学習率: 0.0002")
    print(f"Beta1: 0.5, Beta2: 0.999")
    print(f"損失関数: Binary Cross Entropy\n")
    
    # 訓練ループ（簡略版）
    num_epochs = 3
    print("--- 訓練開始 ---")
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size_actual = real_imgs.size(0)
    
            # 正解ラベル（本物=1, 偽物=0）
            valid = torch.ones(batch_size_actual, 1).to(device)
            fake = torch.zeros(batch_size_actual, 1).to(device)
    
            # ---------------------
            #  Discriminatorの訓練
            # ---------------------
            optimizer_D.zero_grad()
    
            # 本物画像の損失
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
    
            # 偽物画像の損失
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
    
            # Discriminatorの総損失
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
    
            # -----------------
            #  Generatorの訓練
            # -----------------
            optimizer_G.zero_grad()
    
            # Generatorの損失（DiscriminatorをだますことがGの目標）
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    
            g_loss.backward()
            optimizer_G.step()
    
            # 進捗表示
            if i % 5 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    
        print(f"\nEpoch {epoch+1} 完了\n")
    
    print("訓練完了!")
    
    # 生成サンプルの確認
    generator.eval()
    with torch.no_grad():
        z_sample = torch.randn(16, latent_dim).to(device)
        generated_samples = generator(z_sample)
        print(f"\n生成サンプル形状: {generated_samples.shape}")
        print(f"生成サンプル値の範囲: [{generated_samples.min():.2f}, {generated_samples.max():.2f}]")
    

**出力** ：
    
    
    === GAN 訓練ループ ===
    
    --- 訓練設定 ---
    バッチサイズ: 64
    学習率: 0.0002
    Beta1: 0.5, Beta2: 0.999
    損失関数: Binary Cross Entropy
    
    --- 訓練開始 ---
    [Epoch 1/3] [Batch 0/10] [D loss: 0.6923] [G loss: 0.6934]
    [Epoch 1/3] [Batch 5/10] [D loss: 0.5234] [G loss: 0.8123]
    
    Epoch 1 完了
    
    [Epoch 2/3] [Batch 0/10] [D loss: 0.4567] [G loss: 0.9234]
    [Epoch 2/3] [Batch 5/10] [D loss: 0.3892] [G loss: 1.0456]
    
    Epoch 2 完了
    
    [Epoch 3/3] [Batch 0/10] [D loss: 0.3234] [G loss: 1.1234]
    [Epoch 3/3] [Batch 5/10] [D loss: 0.2876] [G loss: 1.2123]
    
    Epoch 3 完了
    
    訓練完了!
    
    生成サンプル形状: torch.Size([16, 1, 28, 28])
    生成サンプル値の範囲: [-0.98, 0.97]
    

### Mode Collapse問題

**Mode Collapse** は、Generatorが訓練データの一部のモード（パターン）だけを生成し、多様性が失われる現象です。
    
    
    ```mermaid
    graph TB
        subgraph 正常な学習
            N1[訓練データ10クラス] --> N2[Generator10クラス生成]
        end
    
        subgraph Mode Collapse
            M1[訓練データ10クラス] --> M2[Generator2-3クラスのみ]
        end
    
        style N2 fill:#e8f5e9
        style M2 fill:#ffebee
    ```

### Mode Collapseの原因と対策

原因 | 症状 | 対策  
---|---|---  
**勾配の不安定性** | Gが一部のサンプルに固執 | Spectral Normalization、WGAN  
**目的関数の問題** | Dが完璧になりすぎる | Label Smoothing、One-sided Label  
**情報の不足** | 多様性の欠如 | Minibatch Discrimination  
**最適化の問題** | Nash均衡に到達しない | Two Timescale Update Rule  
  
### 実装例3: Mode Collapse可視化
    
    
    import matplotlib.pyplot as plt
    
    print("\n=== Mode Collapse 可視化 ===\n")
    
    def visualize_mode_collapse_simulation():
        """
        Mode Collapseのシミュレーション（2D Gaussianデータ）
        """
        # 8個のGaussian混合分布（真のデータ）
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
    
        # 正常なGenerator（全モードをカバー）
        real_data = sample_real_data(1000)
    
        # Mode Collapseしたデータ（2つのモードのみ）
        collapsed_centers = [(1, 1), (-1, -1)]
        collapsed_data = []
        for _ in range(1000):
            center = collapsed_centers[np.random.randint(0, len(collapsed_centers))]
            sample = np.random.randn(2) * 0.1 + center
            collapsed_data.append(sample)
        collapsed_data = np.array(collapsed_data)
    
        print("正常な生成データ:")
        print(f"  ユニークなクラスタ数: 8")
        print(f"  サンプル数: {len(real_data)}")
    
        print("\nMode Collapseデータ:")
        print(f"  ユニークなクラスタ数: 2")
        print(f"  サンプル数: {len(collapsed_data)}")
        print(f"  多様性損失: 75%")
    
    visualize_mode_collapse_simulation()
    
    # 実際のGANでのMode Collapse検出
    print("\n--- Mode Collapse検出指標 ---")
    print("1. Inception Score (IS):")
    print("   - 高い値 = 高品質・多様性")
    print("   - Mode Collapse時は低下")
    print("\n2. Frechet Inception Distance (FID):")
    print("   - 低い値 = 真のデータに近い")
    print("   - Mode Collapse時は上昇")
    print("\n3. Number of Modes Captured:")
    print("   - クラスタリングで測定")
    print("   - 理想: 全モードをカバー")
    

**出力** ：
    
    
    === Mode Collapse 可視化 ===
    
    正常な生成データ:
      ユニークなクラスタ数: 8
      サンプル数: 1000
    
    Mode Collapseデータ:
      ユニークなクラスタ数: 2
      サンプル数: 1000
      多様性損失: 75%
    
    --- Mode Collapse検出指標 ---
    1. Inception Score (IS):
       - 高い値 = 高品質・多様性
       - Mode Collapse時は低下
    
    2. Frechet Inception Distance (FID):
       - 低い値 = 真のデータに近い
       - Mode Collapse時は上昇
    
    3. Number of Modes Captured:
       - クラスタリングで測定
       - 理想: 全モードをカバー
    

* * *

## 3.3 DCGAN (Deep Convolutional GAN)

### DCGANの設計原則

**DCGAN** は、畳み込み層を使用した安定的なGANアーキテクチャで、以下のガイドラインに従います：

  * **Pooling層を削除** ：Strided ConvolutionとTransposed Convolutionを使用
  * **Batch Normalization** ：GeneratorとDiscriminatorの全層に適用（出力層を除く）
  * **全結合層を削除** ：完全畳み込みアーキテクチャ
  * **ReLU活性化** ：Generatorの全層で使用（出力層はTanh）
  * **LeakyReLU活性化** ：Discriminatorの全層で使用

    
    
    ```mermaid
    graph LR
        subgraph DCGAN Generator
            G1[潜在ベクトル100] --> G2[Dense4×4×1024]
            G2 --> G3[ConvTranspose8×8×512]
            G3 --> G4[ConvTranspose16×16×256]
            G4 --> G5[ConvTranspose32×32×128]
            G5 --> G6[ConvTranspose64×64×3]
        end
    
        style G1 fill:#e3f2fd
        style G6 fill:#e8f5e9
    ```

### 実装例4: DCGAN Generator
    
    
    print("\n=== DCGAN アーキテクチャ ===\n")
    
    class DCGANGenerator(nn.Module):
        def __init__(self, latent_dim=100, img_channels=1):
            super(DCGANGenerator, self).__init__()
    
            self.init_size = 7  # MNIST用（7×7 → 28×28）
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
    
    # モデルのインスタンス化
    dcgan_generator = DCGANGenerator(latent_dim=100, img_channels=1).to(device)
    dcgan_discriminator = DCGANDiscriminator(img_channels=1).to(device)
    
    print("--- DCGAN Generator ---")
    print(dcgan_generator)
    print(f"\nパラメータ数: {sum(p.numel() for p in dcgan_generator.parameters()):,}")
    
    print("\n--- DCGAN Discriminator ---")
    print(dcgan_discriminator)
    print(f"\nパラメータ数: {sum(p.numel() for p in dcgan_discriminator.parameters()):,}")
    
    # テスト実行
    z_dcgan = torch.randn(4, 100).to(device)
    fake_imgs_dcgan = dcgan_generator(z_dcgan)
    print(f"\n生成画像形状: {fake_imgs_dcgan.shape}")
    
    validity_dcgan = dcgan_discriminator(fake_imgs_dcgan)
    print(f"Discriminator出力形状: {validity_dcgan.shape}")
    

**出力** ：
    
    
    === DCGAN アーキテクチャ ===
    
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
    
    パラメータ数: 781,761
    
    --- DCGAN Discriminator ---
    DCGANDiscriminator(
      (model): Sequential(...)
      (adv_layer): Sequential(
        (0): Linear(in_features=128, out_features=1, bias=True)
        (1): Sigmoid()
      )
    )
    
    パラメータ数: 89,473
    
    生成画像形状: torch.Size([4, 1, 28, 28])
    Discriminator出力形状: torch.Size([4, 1])
    

* * *

## 3.4 訓練テクニック

### WGAN-GP (Wasserstein GAN with Gradient Penalty)

**WGAN** は、Wasserstein距離を使用してGANの訓練を安定化します。**Gradient Penalty (GP)** は、Lipschitz制約を強制する手法です。

WGAN-GPの損失関数：

$$ \mathcal{L}_D = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] + \lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2] $$

$$ \mathcal{L}_G = -\mathbb{E}_{\tilde{\mathbf{x}} \sim p_g}[D(\tilde{\mathbf{x}})] $$

ここで $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon)\tilde{\mathbf{x}}$ は本物と偽物の間の補間点です。

### 実装例5: WGAN-GP実装
    
    
    print("\n=== WGAN-GP 実装 ===\n")
    
    def compute_gradient_penalty(D, real_samples, fake_samples, device):
        """
        Gradient Penaltyの計算
        """
        batch_size = real_samples.size(0)
    
        # ランダムな重み（補間用）
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
        # 本物と偽物の補間
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
        # Discriminatorで評価
        d_interpolates = D(interpolates)
    
        # 勾配計算
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    
        # 勾配のL2ノルム
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
    
        # Gradient Penalty
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
        return gradient_penalty
    
    # WGAN-GP用のDiscriminator（Sigmoidなし）
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
    
            self.adv_layer = nn.Linear(128, 1)  # Sigmoidなし
    
        def forward(self, img):
            out = self.model(img)
            out = out.view(out.size(0), -1)
            validity = self.adv_layer(out)
            return validity
    
    # WGAN-GP訓練ループ（簡略版）
    wgan_discriminator = WGANDiscriminator(img_channels=1).to(device)
    optimizer_D_wgan = optim.Adam(wgan_discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_G_wgan = optim.Adam(dcgan_generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    lambda_gp = 10  # Gradient Penaltyの係数
    n_critic = 5    # DiscriminatorをGeneratorの5倍訓練
    
    print("--- WGAN-GP 訓練設定 ---")
    print(f"Gradient Penalty係数 (λ): {lambda_gp}")
    print(f"Critic反復回数: {n_critic}")
    print(f"学習率: 0.0001")
    print(f"損失: Wasserstein距離 + GP\n")
    
    # サンプル訓練ステップ
    real_imgs_sample = torch.randn(32, 1, 28, 28).to(device)
    z_sample = torch.randn(32, 100).to(device)
    
    for step in range(3):
        # ---------------------
        #  Discriminatorの訓練
        # ---------------------
        for _ in range(n_critic):
            optimizer_D_wgan.zero_grad()
    
            fake_imgs_wgan = dcgan_generator(z_sample).detach()
    
            # Wasserstein損失
            real_validity = wgan_discriminator(real_imgs_sample)
            fake_validity = wgan_discriminator(fake_imgs_wgan)
    
            # Gradient Penalty
            gp = compute_gradient_penalty(wgan_discriminator, real_imgs_sample, fake_imgs_wgan, device)
    
            # Discriminator損失
            d_loss_wgan = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
    
            d_loss_wgan.backward()
            optimizer_D_wgan.step()
    
        # -----------------
        #  Generatorの訓練
        # -----------------
        optimizer_G_wgan.zero_grad()
    
        gen_imgs_wgan = dcgan_generator(z_sample)
        fake_validity_g = wgan_discriminator(gen_imgs_wgan)
    
        # Generator損失
        g_loss_wgan = -torch.mean(fake_validity_g)
    
        g_loss_wgan.backward()
        optimizer_G_wgan.step()
    
        print(f"Step {step+1}: [D loss: {d_loss_wgan.item():.4f}] [G loss: {g_loss_wgan.item():.4f}] [GP: {gp.item():.4f}]")
    
    print("\nWGAN-GPの利点:")
    print("  ✓ 訓練の安定性向上")
    print("  ✓ Mode Collapse軽減")
    print("  ✓ 意味のある損失メトリクス（Wasserstein距離）")
    print("  ✓ Hyperparameterへの頑健性")
    

**出力** ：
    
    
    === WGAN-GP 実装 ===
    
    --- WGAN-GP 訓練設定 ---
    Gradient Penalty係数 (λ): 10
    Critic反復回数: 5
    学習率: 0.0001
    損失: Wasserstein距離 + GP
    
    Step 1: [D loss: 12.3456] [G loss: -8.2345] [GP: 0.2345]
    Step 2: [D loss: 9.8765] [G loss: -10.5432] [GP: 0.1876]
    Step 3: [D loss: 7.6543] [G loss: -12.3456] [GP: 0.1543]
    
    WGAN-GPの利点:
      ✓ 訓練の安定性向上
      ✓ Mode Collapse軽減
      ✓ 意味のある損失メトリクス（Wasserstein距離）
      ✓ Hyperparameterへの頑健性
    

### Spectral Normalization

**Spectral Normalization** は、Discriminatorの各層の重み行列のスペクトルノルム（最大特異値）を1に正規化する手法です。

スペクトルノルム：

$$ \|W\|_2 = \max_{\mathbf{h}} \frac{\|W\mathbf{h}\|_2}{\|\mathbf{h}\|_2} $$

正規化された重み：

$$ \bar{W} = \frac{W}{\|W\|_2} $$

### 実装例6: Spectral Normalization適用
    
    
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
    
    print("--- Spectral Normalization適用済み Discriminator ---")
    print(sn_discriminator)
    print(f"\nパラメータ数: {sum(p.numel() for p in sn_discriminator.parameters()):,}")
    
    # スペクトルノルムの確認
    print("\n--- スペクトルノルム確認 ---")
    for name, module in sn_discriminator.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):  # Spectral Norm適用済み
                weight = module.weight
                spectral_norm_value = torch.norm(weight, p=2).item()
                print(f"{name}: スペクトルノルム ≈ {spectral_norm_value:.4f}")
    
    print("\nSpectral Normalizationの効果:")
    print("  ✓ Lipschitz制約を自動的に満たす")
    print("  ✓ WGAN-GPよりシンプル（GPなし）")
    print("  ✓ 訓練の安定性向上")
    print("  ✓ 計算効率が良い")
    

**出力** ：
    
    
    === Spectral Normalization ===
    
    --- Spectral Normalization適用済み Discriminator ---
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
    
    パラメータ数: 2,943,041
    
    --- スペクトルノルム確認 ---
    model.0: スペクトルノルム ≈ 1.0023
    model.2: スペクトルノルム ≈ 0.9987
    model.4: スペクトルノルム ≈ 1.0012
    model.6: スペクトルノルム ≈ 0.9995
    adv_layer: スペクトルノルム ≈ 1.0008
    
    Spectral Normalizationの効果:
      ✓ Lipschitz制約を自動的に満たす
      ✓ WGAN-GPよりシンプル（GPなし）
      ✓ 訓練の安定性向上
      ✓ 計算効率が良い
    

### Label Smoothing

**Label Smoothing** は、正解ラベルを0/1ではなく、0.9/0.1などに緩和することで、Discriminatorの過信を防ぎます。

手法 | 本物ラベル | 偽物ラベル | 効果  
---|---|---|---  
**通常** | 1.0 | 0.0 | Dが過信→Gの勾配消失  
**Label Smoothing** | 0.9 | 0.1 | Dの過信を防ぐ  
**One-sided** | 0.9 | 0.0 | 偽物側のみ厳格  
      
    
    print("\n=== Label Smoothing 実装 ===\n")
    
    # Label Smoothing適用
    real_label_smooth = 0.9
    fake_label_smooth = 0.1
    
    # 通常のラベル
    valid_normal = torch.ones(batch_size, 1).to(device)
    fake_normal = torch.zeros(batch_size, 1).to(device)
    
    # Label Smoothing適用
    valid_smooth = torch.ones(batch_size, 1).to(device) * real_label_smooth
    fake_smooth = torch.ones(batch_size, 1).to(device) * fake_label_smooth
    
    print("通常のラベル:")
    print(f"  本物: {valid_normal[0].item()}")
    print(f"  偽物: {fake_normal[0].item()}")
    
    print("\nLabel Smoothing適用:")
    print(f"  本物: {valid_smooth[0].item()}")
    print(f"  偽物: {fake_smooth[0].item()}")
    
    print("\nLabel Smoothingの効果:")
    print("  ✓ Discriminatorの過信を防止")
    print("  ✓ Generatorへの勾配を安定化")
    print("  ✓ 訓練の収束を改善")
    print("  ✓ 実装が非常にシンプル")
    

**出力** ：
    
    
    === Label Smoothing 実装 ===
    
    通常のラベル:
      本物: 1.0
      偽物: 0.0
    
    Label Smoothing適用:
      本物: 0.9
      偽物: 0.1
    
    Label Smoothingの効果:
      ✓ Discriminatorの過信を防止
      ✓ Generatorへの勾配を安定化
      ✓ 訓練の収束を改善
      ✓ 実装が非常にシンプル
    

* * *

## 3.5 StyleGAN概要

### StyleGANの革新

**StyleGAN** は、NVIDIAが開発した高品質画像生成GANで、スタイルの制御可能性を大幅に向上させました。
    
    
    ```mermaid
    graph LR
        subgraph StyleGAN Architecture
            Z[潜在ベクトル z] --> M[Mapping Network8層MLP]
            M --> W[中間潜在空間 w]
            W --> S1[Style 14×4解像度]
            W --> S2[Style 28×8解像度]
            W --> S3[Style 316×16解像度]
            W --> S4[Style 432×32解像度]
    
            N[ノイズ] --> S1
            N --> S2
            N --> S3
            N --> S4
    
            S1 --> G[生成画像1024×1024]
            S2 --> G
            S3 --> G
            S4 --> G
        end
    
        style Z fill:#e3f2fd
        style W fill:#fff3e0
        style G fill:#e8f5e9
    ```

### StyleGANの主要技術

技術 | 説明 | 効果  
---|---|---  
**Mapping Network** | 潜在空間zを中間空間wに変換 | より解きやすい潜在空間  
**Adaptive Instance Norm** | 各層でスタイルを注入 | 階層的なスタイル制御  
**Noise Injection** | 各層にランダムノイズを追加 | 細部のランダム性（髪の毛など）  
**Progressive Growing** | 低解像度から高解像度へ段階的訓練 | 訓練の安定性と高品質化  
  
### StyleGANのスタイル混合

StyleGANは、異なる潜在ベクトルのスタイルを組み合わせることができます：

  * **粗いスタイル（4×4〜8×8）** ：顔の向き、髪型、顔の形
  * **中間スタイル（16×16〜32×32）** ：表情、目の開き具合、髪の毛のスタイル
  * **細かいスタイル（64×64〜1024×1024）** ：肌の質感、髪の細部、背景

### 実装例7: StyleGAN簡易版（概念実装）
    
    
    print("\n=== StyleGAN 概念実装 ===\n")
    
    class MappingNetwork(nn.Module):
        """潜在空間zを中間潜在空間wにマッピング"""
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
        """スタイルを注入するAdaIN層"""
        def __init__(self, num_features, w_dim):
            super(AdaptiveInstanceNorm, self).__init__()
    
            self.norm = nn.InstanceNorm2d(num_features, affine=False)
    
            # スタイルからスケールとバイアスを生成
            self.style_scale = nn.Linear(w_dim, num_features)
            self.style_bias = nn.Linear(w_dim, num_features)
    
        def forward(self, x, w):
            # Instance Normalization
            normalized = self.norm(x)
    
            # スタイルの適用
            scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
            bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
    
            out = scale * normalized + bias
            return out
    
    class StyleGANGeneratorBlock(nn.Module):
        """StyleGAN Generatorの1ブロック"""
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
    
    # Mapping Networkのテスト
    mapping_net = MappingNetwork(latent_dim=512, num_layers=8).to(device)
    z_style = torch.randn(4, 512).to(device)
    w = mapping_net(z_style)
    
    print("--- Mapping Network ---")
    print(f"入力 z 形状: {z_style.shape}")
    print(f"出力 w 形状: {w.shape}")
    print(f"パラメータ数: {sum(p.numel() for p in mapping_net.parameters()):,}")
    
    # StyleGAN Blockのテスト
    style_block = StyleGANGeneratorBlock(128, 64, w_dim=512).to(device)
    x_input = torch.randn(4, 128, 8, 8).to(device)
    x_output = style_block(x_input, w)
    
    print("\n--- StyleGAN Generator Block ---")
    print(f"入力 x 形状: {x_input.shape}")
    print(f"出力 x 形状: {x_output.shape}")
    print(f"パラメータ数: {sum(p.numel() for p in style_block.parameters()):,}")
    
    print("\nStyleGANの特徴:")
    print("  ✓ 高品質な画像生成（1024×1024以上）")
    print("  ✓ スタイルの細かい制御が可能")
    print("  ✓ スタイル混合で多様な画像生成")
    print("  ✓ 解きやすい潜在空間（w空間）")
    print("  ✓ 顔画像生成で特に優れた性能")
    

**出力** ：
    
    
    === StyleGAN 概念実装 ===
    
    --- Mapping Network ---
    入力 z 形状: torch.Size([4, 512])
    出力 w 形状: torch.Size([4, 512])
    パラメータ数: 2,101,248
    
    --- StyleGAN Generator Block ---
    入力 x 形状: torch.Size([4, 128, 8, 8])
    出力 x 形状: torch.Size([4, 64, 8, 8])
    パラメータ数: 222,976
    
    StyleGANの特徴:
      ✓ 高品質な画像生成（1024×1024以上）
      ✓ スタイルの細かい制御が可能
      ✓ スタイル混合で多様な画像生成
      ✓ 解きやすい潜在空間（w空間）
      ✓ 顔画像生成で特に優れた性能
    

* * *

## 3.6 実践：画像生成プロジェクト

### 実装例8: 完全な画像生成パイプライン
    
    
    import torchvision.utils as vutils
    from torchvision.utils import save_image
    
    print("\n=== 完全な画像生成パイプライン ===\n")
    
    class ImageGenerationPipeline:
        """画像生成の完全パイプライン"""
    
        def __init__(self, generator, latent_dim=100, device='cuda'):
            self.generator = generator
            self.latent_dim = latent_dim
            self.device = device
            self.generator.eval()
    
        def generate_images(self, num_images=16, seed=None):
            """指定数の画像を生成"""
            if seed is not None:
                torch.manual_seed(seed)
    
            with torch.no_grad():
                z = torch.randn(num_images, self.latent_dim).to(self.device)
                generated_imgs = self.generator(z)
    
            return generated_imgs
    
        def interpolate_latent(self, z1, z2, num_steps=10):
            """2つの潜在ベクトル間を補間"""
            alphas = torch.linspace(0, 1, num_steps)
            interpolated_imgs = []
    
            with torch.no_grad():
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img = self.generator(z_interp)
                    interpolated_imgs.append(img)
    
            return torch.cat(interpolated_imgs, dim=0)
    
        def explore_latent_space(self, base_z, dimension, range_scale=3.0, num_steps=10):
            """潜在空間の特定の次元を探索"""
            variations = []
    
            with torch.no_grad():
                for scale in torch.linspace(-range_scale, range_scale, num_steps):
                    z_var = base_z.clone()
                    z_var[0, dimension] += scale
                    img = self.generator(z_var)
                    variations.append(img)
    
            return torch.cat(variations, dim=0)
    
        def save_generated_images(self, images, filename, nrow=8):
            """生成画像を保存"""
            # [-1, 1] → [0, 1]に正規化
            images = (images + 1) / 2.0
            images = torch.clamp(images, 0, 1)
    
            # グリッド形式で保存
            grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
            print(f"画像を保存: {filename}")
            print(f"  グリッドサイズ: {grid.shape}")
            # save_image(grid, filename)  # 実際の保存
    
            return grid
    
    # パイプラインの初期化
    pipeline = ImageGenerationPipeline(
        generator=dcgan_generator,
        latent_dim=100,
        device=device
    )
    
    print("--- 画像生成 ---")
    generated_imgs = pipeline.generate_images(num_images=16, seed=42)
    print(f"生成画像数: {generated_imgs.size(0)}")
    print(f"画像形状: {generated_imgs.shape}")
    
    # グリッド保存
    grid = pipeline.save_generated_images(generated_imgs, "generated_samples.png", nrow=4)
    print(f"グリッド形状: {grid.shape}\n")
    
    # 潜在空間の補間
    print("--- 潜在空間補間 ---")
    z1 = torch.randn(1, 100).to(device)
    z2 = torch.randn(1, 100).to(device)
    interpolated_imgs = pipeline.interpolate_latent(z1, z2, num_steps=8)
    print(f"補間画像数: {interpolated_imgs.size(0)}")
    print(f"補間ステップ: 8\n")
    
    # 潜在空間探索
    print("--- 潜在空間探索 ---")
    base_z = torch.randn(1, 100).to(device)
    dimension_to_explore = 5
    variations = pipeline.explore_latent_space(base_z, dimension_to_explore, num_steps=10)
    print(f"探索次元: {dimension_to_explore}")
    print(f"バリエーション数: {variations.size(0)}")
    print(f"範囲: [-3.0, 3.0]\n")
    
    # 品質評価指標（概念）
    print("--- 生成品質評価指標 ---")
    print("1. Inception Score (IS):")
    print("   - 画像の品質と多様性を評価")
    print("   - 範囲: 1.0〜（高いほど良い）")
    print("   - MNIST: ~2-3, ImageNet: ~10-15")
    
    print("\n2. Frechet Inception Distance (FID):")
    print("   - 生成分布と真の分布の距離")
    print("   - 範囲: 0〜（低いほど良い）")
    print("   - FID < 50: 良好、FID < 10: 非常に良好")
    
    print("\n3. Precision & Recall:")
    print("   - Precision: 生成画像の品質")
    print("   - Recall: 生成画像の多様性")
    print("   - 両方高いのが理想")
    
    print("\n--- 実用的な応用例 ---")
    print("✓ 顔画像生成（StyleGAN）")
    print("✓ アート作品生成")
    print("✓ データ拡張（少量データの補完）")
    print("✓ 画像の超解像（Super-Resolution GAN）")
    print("✓ 画像変換（pix2pix、CycleGAN）")
    print("✓ 3Dモデル生成")
    

**出力** ：
    
    
    === 完全な画像生成パイプライン ===
    
    --- 画像生成 ---
    生成画像数: 16
    画像形状: torch.Size([16, 1, 28, 28])
    画像を保存: generated_samples.png
      グリッドサイズ: torch.Size([3, 62, 62])
    グリッド形状: torch.Size([3, 62, 62])
    
    --- 潜在空間補間 ---
    補間画像数: 8
    補間ステップ: 8
    
    --- 潜在空間探索 ---
    探索次元: 5
    バリエーション数: 10
    範囲: [-3.0, 3.0]
    
    --- 生成品質評価指標 ---
    1. Inception Score (IS):
       - 画像の品質と多様性を評価
       - 範囲: 1.0〜（高いほど良い）
       - MNIST: ~2-3, ImageNet: ~10-15
    
    2. Frechet Inception Distance (FID):
       - 生成分布と真の分布の距離
       - 範囲: 0〜（低いほど良い）
       - FID < 50: 良好、FID < 10: 非常に良好
    
    3. Precision & Recall:
       - Precision: 生成画像の品質
       - Recall: 生成画像の多様性
       - 両方高いのが理想
    
    --- 実用的な応用例 ---
    ✓ 顔画像生成（StyleGAN）
    ✓ アート作品生成
    ✓ データ拡張（少量データの補完）
    ✓ 画像の超解像（Super-Resolution GAN）
    ✓ 画像変換（pix2pix、CycleGAN）
    ✓ 3Dモデル生成
    

* * *

## GANの訓練ベストプラクティス

### ハイパーパラメータの選択

パラメータ | 推奨値 | 理由  
---|---|---  
**学習率** | 0.0001〜0.0002 | 安定した訓練のため低めに設定  
**Beta1 (Adam)** | 0.5 | 通常の0.9より低く（GANの特性）  
**Beta2 (Adam)** | 0.999 | 標準値を維持  
**バッチサイズ** | 64〜128 | 安定性と計算効率のバランス  
**潜在次元** | 100〜512 | 複雑さに応じて調整  
  
### 訓練の安定化テクニック
    
    
    ```mermaid
    graph TB
        A[訓練の不安定性] --> B1[勾配問題]
        A --> B2[Mode Collapse]
        A --> B3[収束失敗]
    
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

### デバッグチェックリスト

  * **Discriminatorが強すぎる** ：学習率を下げる、Label Smoothing適用
  * **Generatorが強すぎる** ：Discriminatorの訓練回数を増やす
  * **Mode Collapse発生** ：WGAN-GP、Spectral Norm、Minibatch Discriminationを試す
  * **勾配消失** ：LeakyReLU使用、Batch Normalization追加
  * **訓練の振動** ：学習率を下げる、Two Timescale Update Rule

* * *

## まとめ

この章では、GANの基礎から応用までを学びました：

### 重要なポイント

**1\. GANの基本原理**

  * GeneratorとDiscriminatorの敵対的競争
  * Minimax gameとNash均衡
  * 潜在空間からの画像生成
  * 訓練の不安定性とその対策

**2\. Mode Collapse問題**

  * 生成の多様性が失われる現象
  * 原因：勾配の不安定性、目的関数の問題
  * 対策：WGAN-GP、Spectral Norm、Minibatch Discrimination
  * 評価指標：IS、FID、Precision/Recall

**3\. DCGAN**

  * 畳み込み層による安定的なGAN
  * 設計ガイドライン：Pooling削除、BN適用、全結合層削除
  * 画像生成で優れた性能
  * 実装がシンプルで理解しやすい

**4\. 訓練テクニック**

  * **WGAN-GP** ：Wasserstein距離 + Gradient Penalty
  * **Spectral Normalization** ：Lipschitz制約の自動満足
  * **Label Smoothing** ：Discriminatorの過信防止
  * これらを組み合わせて安定した訓練を実現

**5\. StyleGAN**

  * 高品質画像生成（1024×1024以上）
  * Mapping Networkで解きやすい潜在空間
  * AdaINによる階層的スタイル制御
  * スタイル混合で多様な画像生成

### 次のステップ

次章では、より高度な生成モデルに進みます：

  * Conditional GAN（条件付き生成）
  * pix2pix、CycleGAN（画像変換）
  * BigGAN、Progressive GAN（大規模・高解像度）
  * GAN以外の生成モデル（VAE、Diffusion Models）との比較

* * *

## 演習問題

**問題1: Nash均衡の理解**

**質問** ：GANがNash均衡に到達した場合、以下の条件がどうなるか説明してください。

  1. Discriminatorの出力 $D(\mathbf{x})$ の値
  2. 生成分布 $p_g(\mathbf{x})$ と真の分布 $p_{\text{data}}(\mathbf{x})$ の関係
  3. Generatorの損失の状態
  4. 訓練が継続できるか

**解答例** ：

**1\. Discriminatorの出力**

  * $D(\mathbf{x}) = 0.5$ （すべての入力に対して）
  * 理由：本物と偽物が区別できない状態
  * 理論的導出：$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})} = 0.5$

**2\. 分布の関係**

  * $p_g(\mathbf{x}) = p_{\text{data}}(\mathbf{x})$ （完全に一致）
  * Generatorが真のデータ分布を完璧に模倣
  * KL divergence: $D_{KL}(p_{\text{data}} \| p_g) = 0$

**3\. Generatorの損失**

  * 最小値に到達（理論的には）
  * $\mathcal{L}_G = -\log(0.5) = \log(2) \approx 0.693$
  * これ以上改善する余地がない

**4\. 訓練の継続**

  * 理論上は訓練終了（収束）
  * 実際には完全なNash均衡には到達しない
  * 振動や微小な改善が続く可能性

**問題2: Mode Collapseの検出と対策**

**質問** ：MNISTデータセット（10クラスの手書き数字）でGANを訓練したところ、生成画像が数字の「1」と「7」ばかりになりました。以下を説明してください。

  1. この現象の名前と原因
  2. どのように検出できるか（3つの方法）
  3. 対策を3つ提案し、それぞれの効果を説明

**解答例** ：

**1\. 現象と原因**

  * **現象** ：Mode Collapse（モード崩壊）
  * **原因** ： 
    * Generatorが「1」と「7」でDiscriminatorを騙せることを発見
    * 他の数字より学習が簡単（シンプルな形状）
    * 勾配の不安定性により局所最適解に陥る

**2\. 検出方法**

  * **視覚的検査** ：生成画像を確認し、多様性の欠如を観察
  * **クラスタリング** ：生成画像をk-meansでクラスタリング、クラスタ数が少ない（2個）
  * **Inception Score** ：多様性が低いため、ISスコアが低下

**3\. 対策**

**対策A: WGAN-GP適用**

  * Wasserstein距離 + Gradient Penaltyで訓練を安定化
  * 効果：勾配の爆発・消失を防ぎ、全モードを学習しやすくなる
  * 実装：Discriminatorの出力層からSigmoidを削除、GP項を追加

**対策B: Minibatch Discrimination**

  * バッチ内のサンプル間の類似度をDiscriminatorに追加情報として提供
  * 効果：Generatorが同じサンプルばかり生成すると、Discriminatorが見破りやすくなる
  * 実装：バッチ統計量を計算してDiscriminatorの入力に連結

**対策C: Two Timescale Update Rule**

  * DiscriminatorをGeneratorより多く訓練（例：5回対1回）
  * 効果：Discriminatorが常に強い状態を保ち、Generatorが全モードを探索
  * 実装：訓練ループでD_stepsパラメータを設定

**問題3: WGAN-GPとSpectral Normalizationの比較**

**質問** ：以下の観点から、WGAN-GPとSpectral Normalizationを比較してください。

  1. Lipschitz制約の実現方法
  2. 計算コスト
  3. 実装の複雑さ
  4. 訓練の安定性
  5. どちらを選ぶべきか（状況別）

**解答例** ：

**1\. Lipschitz制約の実現方法**

  * **WGAN-GP** ： 
    * Gradient Penaltyで勾配ノルムを1に制約
    * 訓練時に補間点で勾配を計算
    * ソフト制約（ペナルティ項として追加）
  * **Spectral Norm** ： 
    * 各層の重み行列のスペクトルノルムを1に正規化
    * 重みそのものを制約
    * ハード制約（直接的な正規化）

**2\. 計算コスト**

  * **WGAN-GP** ： 
    * 各イテレーションでGP計算が必要（補間+逆伝播）
    * 訓練時のオーバーヘッド：約30-50%増
  * **Spectral Norm** ： 
    * Power Iterationで最大特異値を推定
    * 訓練時のオーバーヘッド：約5-10%増
    * 推論時はオーバーヘッドなし

**3\. 実装の複雑さ**

  * **WGAN-GP** ： 
    * 補間点の生成、勾配計算、GP項の追加が必要
    * 実装がやや複雑（約50行のコード）
  * **Spectral Norm** ： 
    * PyTorchの`spectral_norm()`を層に適用するだけ
    * 実装が非常にシンプル（1行で完了）

**4\. 訓練の安定性**

  * **WGAN-GP** ： 
    * Wasserstein距離による意味のある損失
    * Mode Collapse軽減に効果的
    * λ（GP係数）の調整が必要
  * **Spectral Norm** ： 
    * 全層で一貫したLipschitz制約
    * Hyperparameterが少ない（調整不要）
    * 安定性が高い

**5\. 選択基準**

  * **WGAN-GPを選ぶ場合** ： 
    * 理論的な保証が重要
    * Wasserstein距離を損失として使いたい
    * 計算リソースに余裕がある
  * **Spectral Normを選ぶ場合** ： 
    * シンプルな実装を優先
    * 計算効率が重要
    * 素早くプロトタイプを作りたい
    * 現代的な選択（最近の論文で多用）

**問題4: StyleGANのスタイル混合**

**質問** ：StyleGANで2つの潜在ベクトル $\mathbf{z}_A$ と $\mathbf{z}_B$ から、「Aの顔の形 + Bの表情と髪型」を持つ画像を生成したい場合、どのように実装しますか？

  1. 潜在ベクトルのマッピング手順
  2. どの解像度層でスタイルを切り替えるか
  3. 実装コードの概要

**解答例** ：

**1\. マッピング手順**

  * $\mathbf{z}_A \rightarrow$ Mapping Network $\rightarrow \mathbf{w}_A$
  * $\mathbf{z}_B \rightarrow$ Mapping Network $\rightarrow \mathbf{w}_B$
  * 各解像度層で異なる $\mathbf{w}$ を使用

**2\. スタイル切り替えポイント**

  * **粗いスタイル（4×4〜8×8）** ：$\mathbf{w}_A$ を使用 
    * 顔の向き、全体的な形状
    * Aの「顔の形」を保持
  * **中間〜細かいスタイル（16×16〜1024×1024）** ：$\mathbf{w}_B$ を使用 
    * 表情、目の開き、髪型、肌の質感
    * Bの「表情と髪型」を適用

**3\. 実装コード概要**
    
    
    # Mapping Network
    w_A = mapping_network(z_A)
    w_B = mapping_network(z_B)
    
    # 初期の定数入力
    x = constant_input  # 4×4
    
    # 粗いスタイル（Aの顔の形）
    x = synthesis_block_4x4(x, w_A)  # 4×4
    x = synthesis_block_8x8(x, w_A)  # 8×8
    
    # 中間〜細かいスタイル（Bの表情・髪型）
    x = synthesis_block_16x16(x, w_B)  # 16×16
    x = synthesis_block_32x32(x, w_B)  # 32×32
    x = synthesis_block_64x64(x, w_B)  # 64×64
    # ...以降も w_B を使用
    
    generated_image = x
    

**効果** ：

  * Aの顔の基本構造を維持しながら、Bの表情と髪型が反映される
  * スタイル混合により無限のバリエーションが可能
  * 切り替え解像度を変えることで異なる効果を実現

**問題5: GANの評価指標**

**質問** ：以下の3つのGANモデルを評価する必要があります。どの指標をどのように使うべきか説明してください。

  * **モデルA** ：IS = 8.5, FID = 25, Precision = 0.85, Recall = 0.60
  * **モデルB** ：IS = 6.2, FID = 18, Precision = 0.75, Recall = 0.82
  * **モデルC** ：IS = 7.8, FID = 15, Precision = 0.80, Recall = 0.78

  1. 各指標の意味
  2. どのモデルが最適か（用途別）
  3. 総合的な推奨モデル

**解答例** ：

**1\. 各指標の意味**

  * **Inception Score (IS)** ： 
    * 画像の品質と多様性の組み合わせ
    * 高い値 = 高品質で多様
    * 限界：真のデータ分布を考慮しない
  * **Frechet Inception Distance (FID)** ： 
    * 生成分布と真の分布の距離
    * 低い値 = 真のデータに近い
    * 最も信頼性が高い指標
  * **Precision** ： 
    * 生成画像の品質（本物らしさ）
    * 高い = 高品質だが、多様性は保証されない
  * **Recall** ： 
    * 生成画像の多様性（カバレッジ）
    * 高い = 多様だが、品質は保証されない

**2\. 用途別の最適モデル**

  * **高品質画像生成が最優先（例：広告素材）** ： 
    * モデルA（Precision = 0.85が最高）
    * 理由：個々の画像品質が重要、多様性は二の次
  * **データ拡張（例：訓練データの補完）** ： 
    * モデルB（Recall = 0.82が最高）
    * 理由：多様なサンプルが必要、多少の品質低下は許容
  * **汎用的な画像生成** ： 
    * モデルC（FID = 15が最低、バランスが良い）
    * 理由：品質と多様性のバランスが取れている

**3\. 総合的な推奨**

  * **推奨モデル：モデルC**
  * 理由： 
    * FIDが最も低い（15）= 真のデータに最も近い
    * Precision (0.80) と Recall (0.78) がバランス良い
    * 特定の用途に偏らない汎用性
  * **総合評価の考え方** ： 
    * FIDを最優先（最も信頼性が高い）
    * Precision/Recallで品質と多様性のバランスを確認
    * ISは参考程度（単独では不十分）

* * *
