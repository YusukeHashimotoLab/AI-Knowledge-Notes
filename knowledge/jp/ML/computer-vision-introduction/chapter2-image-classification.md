---
title: 第2章：画像分類とディープラーニング
chapter_title: 第2章：画像分類とディープラーニング
subtitle: CNNアーキテクチャと転移学習による高精度な画像分類システムの構築
reading_time: 30-35分
difficulty: 中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ LeNet、AlexNet、VGG、ResNetなどの主要CNNアーキテクチャの特徴を理解する
  * ✅ InceptionとMobileNetの効率的な設計原理を説明できる
  * ✅ EfficientNetのCompound Scalingを理解する
  * ✅ 転移学習とFine-tuningの違いと使い分けを習得する
  * ✅ torchvision.modelsを使った事前学習済みモデルの活用方法を学ぶ
  * ✅ Data Augmentationによる汎化性能の向上手法を実装できる
  * ✅ Learning Rate Scheduling、TTA、Model Ensembleなどの訓練テクニックを活用できる
  * ✅ 実践的な画像分類プロジェクトを完成させられる

* * *

## 2.1 CNNアーキテクチャの進化

### 画像分類の歴史的発展

画像分類は、コンピュータビジョンの最も基本的かつ重要なタスクの一つです。ディープラーニングの登場により、画像分類の精度は飛躍的に向上しました。
    
    
    ```mermaid
    graph LR
        A[LeNet-51998MNIST] --> B[AlexNet2012ImageNet]
        B --> C[VGG2014深さ19層]
        C --> D[GoogLeNet2014Inception]
        D --> E[ResNet2015残差接続]
        E --> F[Inception-v42016Hybrid]
        F --> G[MobileNet2017軽量化]
        G --> H[EfficientNet2019最適化]
        H --> I[Vision Transformer2020+Attention]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
        style F fill:#03a9f4
        style G fill:#039be5
        style H fill:#0288d1
        style I fill:#0277bd
    ```

### LeNet-5 (1998): CNNの原点

**LeNet-5** は、Yann LeCunが開発した手書き数字認識のためのネットワークで、現代のCNNの基礎となりました。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LeNet5(nn.Module):
        """LeNet-5: 手書き数字認識のための古典的CNN"""
        def __init__(self, num_classes=10):
            super(LeNet5, self).__init__()
    
            # 特徴抽出層
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28×28 → 24×24
            self.pool1 = nn.AvgPool2d(kernel_size=2)       # 24×24 → 12×12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12×12 → 8×8
            self.pool2 = nn.AvgPool2d(kernel_size=2)       # 8×8 → 4×4
    
            # 分類層
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
    
            x = x.view(x.size(0), -1)  # 平坦化
    
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
    
            return x
    
    # モデルのインスタンス化とテスト
    model = LeNet5(num_classes=10)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    
    print(f"LeNet-5")
    print(f"入力: {x.shape} → 出力: {output.shape}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    

### AlexNet (2012): ディープラーニング革命

**AlexNet** は、2012年のImageNet Large Scale Visual Recognition Challenge (ILSVRC)で優勝し、ディープラーニングブームの火付け役となりました。

主な革新：

  * **ReLU活性化関数** : Sigmoidより高速な学習
  * **Dropout** : 過学習の防止
  * **Data Augmentation** : 汎化性能の向上
  * **GPU並列処理** : 大規模モデルの訓練を可能に

    
    
    import torch
    import torch.nn as nn
    
    class AlexNet(nn.Module):
        """AlexNet: ImageNet 2012優勝モデル"""
        def __init__(self, num_classes=1000):
            super(AlexNet, self).__init__()
    
            # 特徴抽出層
            self.features = nn.Sequential(
                # Conv1: 96 filters, 11×11, stride=4
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
    
                # Conv2: 256 filters, 5×5
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
    
                # Conv3: 384 filters, 3×3
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
    
                # Conv4: 384 filters, 3×3
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
    
                # Conv5: 256 filters, 3×3
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
    
            # 分類層
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # モデルサイズの確認
    model = AlexNet(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nAlexNet 総パラメータ数: {total_params:,}")
    print(f"メモリ使用量: 約{total_params * 4 / (1024**2):.1f} MB")
    

### VGGNet (2014): シンプルさの美学

**VGGNet** は、3×3の小さなフィルタを繰り返し使うシンプルな設計により、深いネットワークの有効性を示しました。

設計原則：

  * **3×3フィルタのみ** 使用（小さいフィルタを重ねる方が効率的）
  * **2×2 Max Pooling** で段階的にサイズを半減
  * **チャンネル数を倍増** （64 → 128 → 256 → 512）

> なぜ3×3フィルタを2回重ねるのか？  
>  受容野: 5×5と同じ  
>  パラメータ数: 3×3×2 = 18 < 5×5 = 25  
>  非線形性: ReLUが2回 → より強い表現力
    
    
    import torch
    import torch.nn as nn
    
    class VGGBlock(nn.Module):
        """VGGの基本ブロック: Conv → ReLU を繰り返す"""
        def __init__(self, in_channels, out_channels, num_convs):
            super(VGGBlock, self).__init__()
    
            layers = []
            for i in range(num_convs):
                layers.append(nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ))
                layers.append(nn.ReLU(inplace=True))
    
            self.block = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.block(x)
    
    class VGG16(nn.Module):
        """VGG-16: 16層の深いネットワーク"""
        def __init__(self, num_classes=1000):
            super(VGG16, self).__init__()
    
            # 特徴抽出部分
            self.features = nn.Sequential(
                VGGBlock(3, 64, 2),      # Block 1
                nn.MaxPool2d(2, 2),
    
                VGGBlock(64, 128, 2),    # Block 2
                nn.MaxPool2d(2, 2),
    
                VGGBlock(128, 256, 3),   # Block 3
                nn.MaxPool2d(2, 2),
    
                VGGBlock(256, 512, 3),   # Block 4
                nn.MaxPool2d(2, 2),
    
                VGGBlock(512, 512, 3),   # Block 5
                nn.MaxPool2d(2, 2),
            )
    
            # 分類部分
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # VGG-16のアーキテクチャ確認
    model = VGG16(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    
    print("VGG-16 各層の出力サイズ:")
    for name, module in model.features.named_children():
        x = module(x)
        if isinstance(module, (VGGBlock, nn.MaxPool2d)):
            print(f"  {name}: {x.shape}")
    

### ResNet (2015): 残差接続の革命

**ResNet** は、**Skip Connections（残差接続）** を導入し、非常に深いネットワーク（100層以上）の学習を可能にしました。

問題：ネットワークを深くすると勾配消失問題が発生  
解決：Residual Blockによる恒等写像の学習

$$ \mathbf{y} = F(\mathbf{x}) + \mathbf{x} $$ 

ここで、$F(\mathbf{x})$は残差関数、$\mathbf{x}$はショートカット接続。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ResidualBlock(nn.Module):
        """ResNetの基本ブロック（Residual Block）"""
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
    
            # Main path
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
    
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
            # Shortcut path
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
        def forward(self, x):
            identity = x
    
            # Main path
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
    
            # Shortcut connection
            out += self.shortcut(identity)
            out = F.relu(out)
    
            return out
    
    class ResNet18(nn.Module):
        """ResNet-18: 18層の残差ネットワーク"""
        def __init__(self, num_classes=1000):
            super(ResNet18, self).__init__()
    
            # 初期層
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
            # Residual blocks
            self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
            self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
            self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
    
            # 分類層
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
    
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, 1))
            return nn.Sequential(*layers)
    
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
    
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
            return x
    
    # ResNet-18の構造確認
    model = ResNet18(num_classes=1000)
    print(f"ResNet-18 パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # Skip Connectionの効果を確認
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"入力: {x.shape} → 出力: {output.shape}")
    

### Inception (GoogLeNet, 2014): 効率的なマルチスケール特徴抽出

**Inception** モジュールは、異なるサイズのフィルタを並列に適用し、マルチスケールな特徴を効率的に抽出します。
    
    
    import torch
    import torch.nn as nn
    
    class InceptionModule(nn.Module):
        """Inception Module: 複数のフィルタサイズを並列処理"""
        def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
            super(InceptionModule, self).__init__()
    
            # 1x1 convolution branch
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, ch1x1, kernel_size=1),
                nn.ReLU(inplace=True)
            )
    
            # 1x1 → 3x3 convolution branch
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
    
            # 1x1 → 5x5 convolution branch
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
                nn.ReLU(inplace=True)
            )
    
            # 3x3 pooling → 1x1 convolution branch
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, pool_proj, kernel_size=1),
                nn.ReLU(inplace=True)
            )
    
        def forward(self, x):
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
    
            # Concatenate along channel dimension
            outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
            return outputs
    
    # Inception Moduleのテスト
    x = torch.randn(1, 256, 28, 28)
    inception = InceptionModule(256, ch1x1=64, ch3x3red=96, ch3x3=128,
                                ch5x5red=16, ch5x5=32, pool_proj=32)
    output = inception(x)
    
    print(f"Inception Module")
    print(f"入力: {x.shape}")
    print(f"出力: {output.shape}")
    print(f"出力チャンネル数: {64 + 128 + 32 + 32} = {output.size(1)}")
    

### MobileNet (2017): 軽量化とモバイル最適化

**MobileNet** は、**Depthwise Separable Convolution** を使用し、計算量とパラメータ数を大幅に削減します。

標準的な畳み込み：$D_K \times D_K \times M \times N$のコスト  
Depthwise Separable: $D_K \times D_K \times M + M \times N$のコスト  
削減率：約8〜9倍
    
    
    import torch
    import torch.nn as nn
    
    class DepthwiseSeparableConv(nn.Module):
        """Depthwise Separable Convolution"""
        def __init__(self, in_channels, out_channels, stride=1):
            super(DepthwiseSeparableConv, self).__init__()
    
            # Depthwise: 各チャンネルごとに畳み込み
            self.depthwise = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3,
                         stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
    
            # Pointwise: 1x1畳み込みでチャンネル方向の結合
            self.pointwise = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x
    
    # 計算量の比較
    def count_operations(in_channels, out_channels, kernel_size, input_size):
        # 標準的な畳み込み
        standard_ops = kernel_size * kernel_size * in_channels * out_channels * input_size * input_size
    
        # Depthwise Separable
        depthwise_ops = kernel_size * kernel_size * in_channels * input_size * input_size
        pointwise_ops = in_channels * out_channels * input_size * input_size
        separable_ops = depthwise_ops + pointwise_ops
    
        reduction = standard_ops / separable_ops
    
        return standard_ops, separable_ops, reduction
    
    standard, separable, reduction = count_operations(128, 256, 3, 56)
    print(f"計算量の比較（128→256チャンネル、3×3フィルタ、56×56入力）:")
    print(f"  標準的な畳み込み: {standard:,} operations")
    print(f"  Depthwise Separable: {separable:,} operations")
    print(f"  削減率: {reduction:.2f}x")
    

### EfficientNet (2019): 最適なスケーリング

**EfficientNet** は、ネットワークの深さ・幅・解像度をバランスよくスケーリングする**Compound Scaling** を提案しました。

Compound Scaling:

$$ \text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi $$ 

制約条件: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$

モデル | Top-1精度 | パラメータ数 | FLOPs  
---|---|---|---  
EfficientNet-B0 | 77.1% | 5.3M | 0.39B  
EfficientNet-B1 | 79.1% | 7.8M | 0.70B  
EfficientNet-B7 | 84.4% | 66M | 37B  
ResNet-50 | 76.0% | 26M | 4.1B  
  
### アーキテクチャ比較まとめ

アーキテクチャ | 主な特徴 | 利点 | 欠点  
---|---|---|---  
LeNet-5 | 基本的なCNN | シンプル、理解しやすい | 現代的タスクには性能不足  
AlexNet | ReLU、Dropout | 実用的な性能 | パラメータが多い  
VGG | 3×3フィルタの繰り返し | 構造が単純 | 非常に重い  
ResNet | Skip Connections | 深いネットワークが可能 | メモリ消費大  
Inception | マルチスケール並列処理 | 効率的な特徴抽出 | 複雑な構造  
MobileNet | Depthwise Separable Conv | 軽量、高速 | 精度がやや低い  
EfficientNet | Compound Scaling | 最高の効率性 | 訓練に時間がかかる  
  
* * *

## 2.2 転移学習とFine-tuning

### 転移学習とは

**転移学習（Transfer Learning）** は、大規模データセット（ImageNetなど）で事前学習されたモデルを、別のタスクに適用する手法です。

メリット：

  * **少ないデータで高精度** : 数百〜数千枚の画像で実用的な性能
  * **訓練時間の短縮** : ゼロから訓練するより大幅に高速
  * **汎化性能の向上** : 事前学習で得た豊富な特徴表現を活用

    
    
    ```mermaid
    graph LR
        A[ImageNetで事前学習] --> B[重みを読み込み]
        B --> C{転移学習の戦略}
        C --> D[Feature Extraction畳み込み層を凍結]
        C --> E[Fine-tuning全体または一部を再訓練]
    
        D --> F[新しいタスクで高精度]
        E --> F
    
        style A fill:#e1f5ff
        style C fill:#fff9c4
        style D fill:#c8e6c9
        style E fill:#ffccbc
        style F fill:#b3e5fc
    ```

### Feature Extraction vs Fine-tuning

手法 | 畳み込み層 | 分類層 | データ量 | 類似度  
---|---|---|---|---  
**Feature Extraction** | 凍結 | 訓練 | 少（100〜1000枚） | 高  
**Fine-tuning（全層）** | 訓練 | 訓練 | 多（10000枚以上） | 低  
**Fine-tuning（部分）** | 上層のみ訓練 | 訓練 | 中（1000〜10000枚） | 中  
  
### torchvision.modelsによる事前学習済みモデルの活用
    
    
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # 1. 事前学習済みモデルのロード
    print("=== 事前学習済みResNet-18のロード ===")
    model = models.resnet18(pretrained=True)
    
    # モデル構造の確認
    print(f"\nオリジナルの分類層:")
    print(model.fc)
    
    # 2. Feature Extraction: 畳み込み層を凍結
    print("\n=== Feature Extraction（特徴抽出） ===")
    
    # すべてのパラメータを凍結
    for param in model.parameters():
        param.requires_grad = False
    
    # 最後の分類層を置き換え（新しいタスク用）
    num_classes = 10  # CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 訓練可能なパラメータの確認
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    print(f"凍結パラメータ数: {total_params - trainable_params:,}")
    
    # 3. Fine-tuning: 上位層のみ訓練
    print("\n=== Fine-tuning（部分的な再訓練） ===")
    
    # 新しくモデルをロード
    model_ft = models.resnet18(pretrained=True)
    
    # すべてを一旦凍結
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # 最後の2つのブロックと分類層を解凍
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    
    model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
    
    trainable_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"Fine-tuning訓練可能パラメータ数: {trainable_ft:,}")
    
    # 4. 学習率の設定（層ごとに異なる学習率）
    print("\n=== 層ごとの学習率設定 ===")
    optimizer = torch.optim.Adam([
        {'params': model_ft.layer4.parameters(), 'lr': 1e-4},  # 上位層: 小さい学習率
        {'params': model_ft.fc.parameters(), 'lr': 1e-3}       # 分類層: 大きい学習率
    ])
    
    print("層ごとの学習率:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  グループ {i}: lr = {param_group['lr']}")
    

### 実践：CIFAR-10での転移学習
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データ準備
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ResNetの入力サイズに合わせる
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet統計
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # CIFAR-10データセット
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 事前学習済みResNet-18をロード
    model = models.resnet18(pretrained=True)
    
    # Feature Extraction: 畳み込み層を凍結
    for param in model.parameters():
        param.requires_grad = False
    
    # 分類層を置き換え
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    # 訓練関数
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        return running_loss / len(loader), 100. * correct / total
    
    def test_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
    
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        return running_loss / len(loader), 100. * correct / total
    
    # 訓練実行（Feature Extraction）
    print("\n=== Feature Extraction訓練開始 ===")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\n訓練完了！")
    

### Fine-tuning戦略
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models
    
    def create_finetuning_model(num_classes, freeze_layers='none'):
        """
        Fine-tuningモデルを作成
    
        freeze_layers:
            - 'none': すべて訓練
            - 'early': 初期層のみ凍結
            - 'most': 最終層のみ訓練
        """
        model = models.resnet18(pretrained=True)
    
        if freeze_layers == 'early':
            # 初期層（layer1, layer2）を凍結
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
    
            print("凍結: conv1, bn1, layer1, layer2")
            print("訓練: layer3, layer4, fc")
    
        elif freeze_layers == 'most':
            # ほとんどを凍結、layer4とfcのみ訓練
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
    
            print("凍結: conv1〜layer3")
            print("訓練: layer4, fc")
    
        else:
            print("訓練: すべての層")
    
        # 分類層を置き換え
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
        return model
    
    # 異なる戦略でモデルを作成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 戦略1: すべての層を訓練 ===")
    model_all = create_finetuning_model(num_classes=10, freeze_layers='none')
    trainable_all = sum(p.numel() for p in model_all.parameters() if p.requires_grad)
    print(f"訓練可能パラメータ: {trainable_all:,}\n")
    
    print("=== 戦略2: 初期層を凍結 ===")
    model_early = create_finetuning_model(num_classes=10, freeze_layers='early')
    trainable_early = sum(p.numel() for p in model_early.parameters() if p.requires_grad)
    print(f"訓練可能パラメータ: {trainable_early:,}\n")
    
    print("=== 戦略3: 最終層のみ訓練 ===")
    model_most = create_finetuning_model(num_classes=10, freeze_layers='most')
    trainable_most = sum(p.numel() for p in model_most.parameters() if p.requires_grad)
    print(f"訓練可能パラメータ: {trainable_most:,}\n")
    
    # 層ごとに異なる学習率を設定
    def get_optimizer_with_layer_lr(model, base_lr=1e-3):
        """層の深さに応じて学習率を変える"""
        params = []
    
        # 浅い層: 小さい学習率
        params.append({'params': model.conv1.parameters(), 'lr': base_lr * 0.1})
        params.append({'params': model.layer1.parameters(), 'lr': base_lr * 0.2})
        params.append({'params': model.layer2.parameters(), 'lr': base_lr * 0.4})
    
        # 深い層: 大きい学習率
        params.append({'params': model.layer3.parameters(), 'lr': base_lr * 0.7})
        params.append({'params': model.layer4.parameters(), 'lr': base_lr})
    
        # 分類層: 最も大きい学習率
        params.append({'params': model.fc.parameters(), 'lr': base_lr * 2})
    
        return optim.Adam(params)
    
    optimizer = get_optimizer_with_layer_lr(model_all, base_lr=1e-3)
    print("=== 層ごとの学習率 ===")
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for name, param_group in zip(layer_names, optimizer.param_groups):
        print(f"{name:10s}: lr = {param_group['lr']:.4f}")
    

* * *

## 2.3 Data Augmentation（データ拡張）

### Data Augmentationとは

**Data Augmentation** は、訓練データに様々な変換を適用し、データセットを拡張する手法です。過学習を防ぎ、汎化性能を向上させます。

### 幾何学的変換
    
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプル画像の生成（実際にはデータセットから読み込む）
    # ここでは32×32のランダム画像を使用
    np.random.seed(42)
    sample_image = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    
    # 幾何学的変換の定義
    geometric_transforms = {
        'Original': transforms.ToTensor(),
        'Random Crop': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ]),
        'Horizontal Flip': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),
        'Rotation': transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ]),
        'Affine': transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ]),
    }
    
    print("=== 幾何学的変換の例 ===")
    for name, transform in geometric_transforms.items():
        augmented = transform(sample_image)
        print(f"{name:20s}: {augmented.shape}")
    
    # 標準的な訓練用Data Augmentation
    standard_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("\n標準的な訓練用変換:")
    print(standard_train_transform)
    

### 色変換（Color Jittering）
    
    
    from torchvision import transforms
    
    # 色変換の種類
    color_transforms = {
        'Color Jitter': transforms.ColorJitter(
            brightness=0.2,    # 明るさを±20%変更
            contrast=0.2,      # コントラストを±20%変更
            saturation=0.2,    # 彩度を±20%変更
            hue=0.1           # 色相を±10%変更
        ),
        'Grayscale': transforms.RandomGrayscale(p=0.2),  # 20%の確率でグレースケール化
        'Random Erasing': transforms.RandomErasing(
            p=0.5,             # 50%の確率で適用
            scale=(0.02, 0.33),  # 消去領域のサイズ
            ratio=(0.3, 3.3)   # アスペクト比
        )
    }
    
    # 強力なData Augmentation（CIFAR-10用）
    strong_augmentation = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5)
    ])
    
    print("=== 強力なData Augmentation ===")
    print(strong_augmentation)
    

### Mixup と CutMix

**Mixup** と**CutMix** は、2つの画像を混ぜ合わせて新しい訓練サンプルを生成する高度な手法です。
    
    
    import torch
    import numpy as np
    
    def mixup_data(x, y, alpha=1.0):
        """
        Mixup: 2つの画像を線形補間
    
        Args:
            x: 入力画像バッチ (B, C, H, W)
            y: ラベル (B,)
            alpha: Beta分布のパラメータ
    
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
    
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
    
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(x, y, alpha=1.0):
        """
        CutMix: 2つの画像の一部を切り貼り
    
        Args:
            x: 入力画像バッチ (B, C, H, W)
            y: ラベル (B,)
            alpha: Beta分布のパラメータ
    
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
    
        # ランダムな矩形領域を切り取る
        _, _, H, W = x.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
    
        # 中心座標をランダムに選択
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    
        # 画像を混ぜる
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
        # 混合比を調整
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
    
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Mixup用の損失関数"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    # 使用例
    x = torch.randn(4, 3, 32, 32)  # 4枚の画像
    y = torch.tensor([0, 1, 2, 3])  # ラベル
    
    # Mixup
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
    print(f"Mixup:")
    print(f"  元の画像: {x.shape}")
    print(f"  混合画像: {mixed_x.shape}")
    print(f"  混合比 λ: {lam:.3f}")
    print(f"  ラベルA: {y_a.tolist()}, ラベルB: {y_b.tolist()}")
    
    # CutMix
    cutmix_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
    print(f"\nCutMix:")
    print(f"  元の画像: {x.shape}")
    print(f"  混合画像: {cutmix_x.shape}")
    print(f"  混合比 λ: {lam:.3f}")
    

### albumentationsライブラリの活用

**albumentations** は、高速で豊富なData Augmentation機能を提供するライブラリです。
    
    
    # albumentationsのインストール: pip install albumentations
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    import numpy as np
    
    # albumentationsによる強力なData Augmentation
    album_transform = A.Compose([
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])
    
    print("=== albumentations Data Augmentation ===")
    print("変換一覧:")
    for i, transform in enumerate(album_transform.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    # PyTorchのDatasetと組み合わせる例
    class AlbumentationsDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
    
            # PIL Image → numpy array
            image = np.array(image)
    
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
    
            return image, label
    
    print("\nalbumentationsとPyTorchの統合:")
    print("AlbumentationsDatasetクラスを使用して、")
    print("torchvision.datasetsと組み合わせることができます。")
    

* * *

## 2.4 訓練テクニック

### Learning Rate Scheduling

**Learning Rate Scheduling** は、訓練の進行に応じて学習率を調整し、収束を改善する手法です。
    
    
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
    import matplotlib.pyplot as plt
    
    # ダミーモデル
    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 1. StepLR: 固定ステップで学習率を減衰
    scheduler_step = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 2. CosineAnnealingLR: コサイン関数で学習率を減衰
    optimizer2 = optim.SGD(model.parameters(), lr=0.1)
    scheduler_cosine = CosineAnnealingLR(optimizer2, T_max=50, eta_min=1e-5)
    
    # 3. ReduceLROnPlateau: 検証損失が改善しない場合に減衰
    optimizer3 = optim.SGD(model.parameters(), lr=0.1)
    scheduler_plateau = ReduceLROnPlateau(optimizer3, mode='min', factor=0.5,
                                          patience=5, verbose=True)
    
    # 学習率の推移を可視化
    lrs_step = []
    lrs_cosine = []
    
    for epoch in range(50):
        # StepLR
        lrs_step.append(optimizer.param_groups[0]['lr'])
        scheduler_step.step()
    
        # CosineAnnealingLR
        lrs_cosine.append(optimizer2.param_groups[0]['lr'])
        scheduler_cosine.step()
    
    print("=== Learning Rate Schedulingの比較 ===")
    print(f"初期学習率: {lrs_step[0]}")
    print(f"StepLR (epoch 50): {lrs_step[-1]:.6f}")
    print(f"CosineAnnealingLR (epoch 50): {lrs_cosine[-1]:.6f}")
    
    # 実践的な使用例
    def train_with_scheduler(model, train_loader, val_loader, epochs=50):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
        # Cosine Annealing with Warm Restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
        for epoch in range(epochs):
            # 訓練ループ
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
            # 学習率の更新
            scheduler.step()
    
            current_lr = optimizer.param_groups[0]['lr']
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}")
    
    print("\n実践的なScheduler: CosineAnnealingWarmRestarts")
    print("  T_0=10: 最初の再起動周期")
    print("  T_mult=2: 周期を2倍ずつ増やす")
    print("  eta_min=1e-6: 最小学習率")
    

### Progressive Resizing

**Progressive Resizing** は、訓練の初期に小さい画像で学習し、徐々に大きい画像に移行する手法です。
    
    
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    
    class ProgressiveResizingTrainer:
        """Progressive Resizingを実装した訓練クラス"""
    
        def __init__(self, model, dataset_path, device):
            self.model = model
            self.dataset_path = dataset_path
            self.device = device
    
        def get_dataloader(self, image_size, batch_size):
            """指定サイズのDataLoaderを作成"""
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
            dataset = datasets.ImageFolder(self.dataset_path, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
            return loader
    
        def train_phase(self, image_size, epochs, lr):
            """特定の画像サイズで訓練"""
            print(f"\n=== Training with image size: {image_size}×{image_size} ===")
    
            loader = self.get_dataloader(image_size, batch_size=32)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
    
            for epoch in range(epochs):
                self.model.train()
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
    
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
        def progressive_train(self):
            """Progressive Resizingで訓練"""
            # フェーズ1: 64×64で高速に学習
            self.train_phase(image_size=64, epochs=10, lr=1e-3)
    
            # フェーズ2: 128×128で中程度の学習
            self.train_phase(image_size=128, epochs=10, lr=5e-4)
    
            # フェーズ3: 224×224で最終調整
            self.train_phase(image_size=224, epochs=20, lr=1e-4)
    
    print("=== Progressive Resizing戦略 ===")
    print("フェーズ1: 64×64   (10 epochs, lr=1e-3)  - 高速な初期学習")
    print("フェーズ2: 128×128 (10 epochs, lr=5e-4)  - 中間的な調整")
    print("フェーズ3: 224×224 (20 epochs, lr=1e-4)  - 高解像度での最終調整")
    print("\nメリット:")
    print("  - 訓練時間の短縮（初期段階）")
    print("  - メモリ使用量の削減")
    print("  - 段階的な精度向上")
    

### Test-Time Augmentation (TTA)

**Test-Time Augmentation** は、テスト時に複数の拡張画像で予測し、その平均を取ることで精度を向上させる手法です。
    
    
    import torch
    import torch.nn as nn
    from torchvision import transforms
    
    class TTAWrapper(nn.Module):
        """Test-Time Augmentationを実装したラッパー"""
    
        def __init__(self, model, num_augmentations=5):
            super(TTAWrapper, self).__init__()
            self.model = model
            self.num_augmentations = num_augmentations
    
            # TTA用の変換
            self.tta_transforms = [
                transforms.Compose([]),  # オリジナル
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),  # 左右反転
                transforms.Compose([transforms.RandomRotation(degrees=5)]),    # 5度回転
                transforms.Compose([transforms.RandomRotation(degrees=-5)]),   # -5度回転
                transforms.Compose([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1)
                ]),  # 色調整
            ]
    
        def forward(self, x):
            """複数の変換で予測し、平均を取る"""
            predictions = []
    
            for transform in self.tta_transforms[:self.num_augmentations]:
                # 画像を変換
                # 注意: transformsはTensorに適用できないため、実際にはカスタム実装が必要
                augmented = x  # 簡略化のため、ここではオリジナルのまま
    
                # 予測
                with torch.no_grad():
                    pred = self.model(augmented)
                    predictions.append(pred)
    
            # 平均を取る
            avg_prediction = torch.stack(predictions).mean(dim=0)
            return avg_prediction
    
    # TTAの使用例
    def predict_with_tta(model, image, num_augmentations=5):
        """
        TTAを使った予測
    
        Args:
            model: 訓練済みモデル
            image: 入力画像 (C, H, W)
            num_augmentations: 拡張の数
    
        Returns:
            averaged prediction
        """
        model.eval()
        predictions = []
    
        # オリジナル
        with torch.no_grad():
            pred = model(image.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
        # 左右反転
        flipped = torch.flip(image, dims=[2])
        with torch.no_grad():
            pred = model(flipped.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
        # 上下反転
        vflipped = torch.flip(image, dims=[1])
        with torch.no_grad():
            pred = model(vflipped.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
        # 左右+上下反転
        hvflipped = torch.flip(torch.flip(image, dims=[1]), dims=[2])
        with torch.no_grad():
            pred = model(hvflipped.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
        # 平均を計算
        avg_pred = torch.stack(predictions).mean(dim=0)
    
        return avg_pred
    
    print("=== Test-Time Augmentation (TTA) ===")
    print("変換の種類:")
    print("  1. オリジナル")
    print("  2. 左右反転")
    print("  3. 上下反転")
    print("  4. 左右+上下反転")
    print("\n精度向上の目安: 1-2%")
    print("計算コスト: 変換数に比例")
    

### Model Ensemble

**Model Ensemble** は、複数のモデルの予測を組み合わせることで、精度と頑健性を向上させる手法です。
    
    
    import torch
    import torch.nn as nn
    from torchvision import models
    
    class ModelEnsemble(nn.Module):
        """複数モデルのアンサンブル"""
    
        def __init__(self, models_list):
            super(ModelEnsemble, self).__init__()
            self.models = nn.ModuleList(models_list)
    
        def forward(self, x):
            """各モデルの予測を平均"""
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(torch.softmax(pred, dim=1))
    
            # 平均を取る
            avg_prediction = torch.stack(predictions).mean(dim=0)
            return avg_prediction
    
    # アンサンブルの作成例
    def create_ensemble(num_classes=10, num_models=3):
        """異なるアーキテクチャのアンサンブル"""
        models_list = []
    
        # ResNet-18
        model1 = models.resnet18(pretrained=False)
        model1.fc = nn.Linear(model1.fc.in_features, num_classes)
        models_list.append(model1)
    
        # ResNet-34
        model2 = models.resnet34(pretrained=False)
        model2.fc = nn.Linear(model2.fc.in_features, num_classes)
        models_list.append(model2)
    
        # MobileNetV2
        model3 = models.mobilenet_v2(pretrained=False)
        model3.classifier[1] = nn.Linear(model3.last_channel, num_classes)
        models_list.append(model3)
    
        ensemble = ModelEnsemble(models_list)
        return ensemble
    
    # 重み付きアンサンブル
    class WeightedEnsemble(nn.Module):
        """重み付きアンサンブル"""
    
        def __init__(self, models_list, weights=None):
            super(WeightedEnsemble, self).__init__()
            self.models = nn.ModuleList(models_list)
    
            if weights is None:
                weights = [1.0 / len(models_list)] * len(models_list)
            self.weights = torch.tensor(weights)
    
        def forward(self, x):
            """重み付き平均"""
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(torch.softmax(pred, dim=1))
    
            # 重み付き平均
            predictions = torch.stack(predictions)
            weighted_pred = (predictions * self.weights.view(-1, 1, 1)).sum(dim=0)
    
            return weighted_pred
    
    print("=== Model Ensemble ===")
    print("戦略:")
    print("  1. 単純平均: すべてのモデルの予測を均等に平均")
    print("  2. 重み付き平均: 性能に応じた重みで平均")
    print("  3. Voting: 多数決による分類")
    print("\nアンサンブルの効果:")
    print("  - 精度向上: 1-3%")
    print("  - 頑健性向上: 個々のモデルのエラーを補完")
    print("  - 推論コスト: モデル数に比例")
    
    # 使用例
    ensemble = create_ensemble(num_classes=10, num_models=3)
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\nアンサンブル総パラメータ数: {total_params:,}")
    

* * *

## 2.5 実践プロジェクト：完全な画像分類システム

### プロジェクト概要

実際のプロジェクトを想定し、カスタムデータセットでの画像分類システムを構築します。
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    import os
    from pathlib import Path
    
    # カスタムデータセット
    class CustomImageDataset(Dataset):
        """カスタム画像データセット"""
    
        def __init__(self, root_dir, transform=None):
            """
            Args:
                root_dir: データディレクトリ
                          root_dir/
                            class1/
                              img1.jpg
                              img2.jpg
                            class2/
                              img1.jpg
            """
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
            # すべての画像パスとラベルを取得
            self.samples = []
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
        def __len__(self):
            return len(self.samples)
    
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
    
            if self.transform:
                image = self.transform(image)
    
            return image, label
    
    # データ準備
    def get_data_transforms(image_size=224):
        """訓練とテスト用の変換を取得"""
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        return train_transform, test_transform
    
    # モデル構築
    def build_model(num_classes, architecture='resnet18', pretrained=True):
        """
        モデルを構築
    
        Args:
            num_classes: クラス数
            architecture: 'resnet18', 'resnet50', 'efficientnet_b0'
            pretrained: 事前学習済み重みを使用するか
        """
        if architecture == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
        elif architecture == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
        elif architecture == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
        return model
    
    # 訓練クラス
    class ImageClassifier:
        """完全な画像分類パイプライン"""
    
        def __init__(self, model, device):
            self.model = model.to(device)
            self.device = device
            self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
        def train_epoch(self, loader, criterion, optimizer):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
    
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
            epoch_loss = running_loss / len(loader)
            epoch_acc = 100. * correct / total
    
            return epoch_loss, epoch_acc
    
        def validate(self, loader, criterion):
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
    
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
    
            epoch_loss = running_loss / len(loader)
            epoch_acc = 100. * correct / total
    
            return epoch_loss, epoch_acc
    
        def fit(self, train_loader, val_loader, epochs=50, lr=1e-3):
            """訓練を実行"""
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
            best_val_acc = 0
    
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc = self.validate(val_loader, criterion)
    
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
    
                scheduler.step()
    
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model('best_model.pth')
    
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}]")
                    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                    print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
            print(f"\n訓練完了！ベスト検証精度: {best_val_acc:.2f}%")
    
        def save_model(self, path):
            """モデルを保存"""
            torch.save(self.model.state_dict(), path)
    
        def load_model(self, path):
            """モデルを読み込み"""
            self.model.load_state_dict(torch.load(path))
    
    # 使用例
    print("=== 完全な画像分類パイプライン ===")
    print("\n使用方法:")
    print("1. データセットを準備（root_dir/class1/, root_dir/class2/, ...）")
    print("2. モデルを構築")
    print("3. ImageClassifierで訓練")
    print("4. ベストモデルを保存")
    print("\nサンプルコード:")
    print("""
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # データローダー
    train_transform, val_transform = get_data_transforms()
    train_dataset = CustomImageDataset('data/train', transform=train_transform)
    val_dataset = CustomImageDataset('data/val', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # モデル構築
    model = build_model(num_classes=10, architecture='resnet18', pretrained=True)
    
    # 訓練
    classifier = ImageClassifier(model, device)
    classifier.fit(train_loader, val_loader, epochs=50, lr=1e-3)
    """)
    

### 評価と可視化
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    def evaluate_model(model, test_loader, device, class_names):
        """
        モデルを詳細に評価
    
        Returns:
            accuracy, confusion matrix, classification report
        """
        model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
    
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
        # 精度
        accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
        # 混同行列
        cm = confusion_matrix(all_labels, all_preds)
    
        # 分類レポート
        report = classification_report(all_labels, all_preds, target_names=class_names)
    
        return accuracy, cm, report
    
    def plot_confusion_matrix(cm, class_names):
        """混同行列を可視化"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("混同行列を保存しました: confusion_matrix.png")
    
    def plot_training_history(history):
        """訓練履歴を可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
    
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("訓練履歴を保存しました: training_history.png")
    
    print("=== モデルの評価と可視化 ===")
    print("\n評価指標:")
    print("  1. 精度（Accuracy）")
    print("  2. 混同行列（Confusion Matrix）")
    print("  3. クラスごとの Precision, Recall, F1-score")
    print("\n可視化:")
    print("  1. 訓練曲線（Loss, Accuracy）")
    print("  2. 混同行列のヒートマップ")
    

* * *

## 演習問題

**演習1: アーキテクチャの比較**

ResNet-18、ResNet-50、EfficientNet-B0の3つのアーキテクチャをCIFAR-10で比較してください。パラメータ数、訓練時間、精度を記録し、どのモデルが最も効率的か考察してください。
    
    
    # TODO: 3つのモデルを実装
    # TODO: 同じ訓練設定で性能を比較
    # TODO: 結果を表にまとめる
    
    # ヒント:
    # - torchvision.modelsを使用
    # - 訓練時間を計測
    # - メモリ使用量も考慮
    

**演習2: 転移学習の効果**

事前学習済みモデルとランダム初期化モデルの性能を比較してください。データ量を変えて（100枚、500枚、全データ）、転移学習の効果がどう変化するか調べてください。
    
    
    # TODO: pretrained=TrueとFalseで比較
    # TODO: データ量を変えて実験
    # TODO: 訓練曲線を比較
    
    # 評価指標:
    # - 最終精度
    # - 収束までのエポック数
    # - データ効率性
    

**演習3: Data Augmentationの効果**

異なるData Augmentation戦略を比較してください：(1) 変換なし、(2) 標準的な変換、(3) 強力な変換（Mixup/CutMix含む）。過学習への影響を調べてください。
    
    
    # TODO: 3つの変換戦略を実装
    # TODO: 訓練誤差とテスト誤差の差を比較
    # TODO: 最適な変換の組み合わせを見つける
    
    # ヒント:
    # - 過学習度 = 訓練精度 - テスト精度
    # - 訓練曲線を可視化
    

**演習4: Learning Rate Schedulingの最適化**

StepLR、CosineAnnealingLR、ReduceLROnPlateauの3つのSchedulerを比較してください。どのSchedulerが最も効果的か、データセットに応じて変わるか調べてください。
    
    
    # TODO: 3つのSchedulerを実装
    # TODO: 学習率の推移を可視化
    # TODO: 最終精度を比較
    
    # 評価:
    # - 収束速度
    # - 最終精度
    # - 安定性
    

**演習5: 完全なプロジェクトの構築**

カスタムデータセットで完全な画像分類システムを構築してください。データ準備、モデル選択、訓練、評価、推論までのパイプラインを実装してください。
    
    
    # TODO: データセットの準備
    # TODO: 最適なアーキテクチャの選択
    # TODO: Data Augmentationの設計
    # TODO: 訓練と評価
    # TODO: 推論用のAPIを作成
    
    # 要件:
    # 1. テスト精度 > 90%
    # 2. 推論時間 < 100ms/画像
    # 3. モデルサイズ < 100MB
    # 4. 可視化と詳細なレポート
    

* * *

## 参考文献

  * LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
  * Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks." NeurIPS.
  * Simonyan, K., & Zisserman, A. (2014). "Very deep convolutional networks for large-scale image recognition." arXiv preprint.
  * He, K., et al. (2016). "Deep residual learning for image recognition." CVPR.
  * Szegedy, C., et al. (2015). "Going deeper with convolutions." CVPR.
  * Howard, A. G., et al. (2017). "MobileNets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint.
  * Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML.
  * Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" NeurIPS.
  * Zhang, H., et al. (2018). "mixup: Beyond empirical risk minimization." ICLR.
  * Yun, S., et al. (2019). "CutMix: Regularization strategy to train strong classifiers with localizable features." ICCV.

* * *

## まとめ

この章では、画像分類とディープラーニングについて学びました。

### 重要ポイント

  * **CNNアーキテクチャの進化** : LeNet → AlexNet → VGG → ResNet → EfficientNet
  * **ResNetの革新** : Skip Connectionsによる深いネットワークの学習
  * **転移学習** : 事前学習済みモデルで少ないデータで高精度を実現
  * **Feature Extraction vs Fine-tuning** : データ量とタスクの類似度に応じた使い分け
  * **Data Augmentation** : 幾何学的変換、色変換、Mixup/CutMixで汎化性能を向上
  * **訓練テクニック** : Learning Rate Scheduling、Progressive Resizing、TTA、Ensembleで精度を向上
  * **実践的パイプライン** : データ準備から評価まで一貫したワークフロー

### 次のステップ

次章では、**物体検出とセマンティックセグメンテーション** について学びます。画像内の物体の位置特定やピクセル単位の分類など、より高度なコンピュータビジョンタスクに挑戦します。
