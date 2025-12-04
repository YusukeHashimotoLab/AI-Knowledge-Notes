---
title: 第2章：プーリング層とCNNアーキテクチャ
chapter_title: 第2章：プーリング層とCNNアーキテクチャ
subtitle: 代表的なCNNモデルの進化と設計原理を理解する
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ プーリング層の役割と種類（Max Pooling, Average Pooling）を理解する
  * ✅ プーリングによる次元削減と位置不変性の獲得を説明できる
  * ✅ LeNet-5からResNetまでの代表的なCNNアーキテクチャの進化を把握する
  * ✅ Batch Normalizationの原理と効果を理解する
  * ✅ Dropoutを用いた過学習防止手法を実装できる
  * ✅ Skip Connections（Residual Connections）の重要性を理解する
  * ✅ CIFAR-10データセットで実践的な画像分類モデルを構築できる

* * *

## 2.1 プーリング層の役割

### プーリングとは

**プーリング層（Pooling Layer）** は、畳み込み層の出力を**空間的にダウンサンプリング** する層です。主な目的は以下の3つです：

  * **次元削減** ：特徴マップのサイズを減らし、計算量とメモリ使用量を削減
  * **位置不変性** ：特徴の微小な位置変化に対して頑健性を獲得
  * **受容野の拡大** ：より広い範囲の情報を統合

> 「プーリングは、画像の重要な特徴を保ちながら、不要な詳細を捨てる操作」

### Max Pooling vs Average Pooling

プーリングには主に2つの種類があります：

種類 | 動作 | 特徴 | 使用場面  
---|---|---|---  
**Max Pooling** | 領域内の最大値を取る | 最も強い特徴を保持 | 物体検出、一般的な画像分類  
**Average Pooling** | 領域内の平均値を取る | 全体的な特徴を保持 | Global Average Pooling、セグメンテーション  
      
    
    ```mermaid
    graph LR
        A["入力特徴マップ4×4"] --> B["Max Pooling2×2, stride=2"]
        A --> C["Average Pooling2×2, stride=2"]
    
        B --> D["出力2×2最大値を保持"]
        C --> E["出力2×2平均値を保持"]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
    ```

### Max Poolingの動作例
    
    
    import numpy as np
    import torch
    import torch.nn as nn
    
    # 入力データ（1チャンネル、4×4）
    input_data = torch.tensor([[
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 6.0, 1.0, 2.0],
        [7.0, 2.0, 8.0, 3.0],
        [1.0, 4.0, 6.0, 9.0]
    ]], dtype=torch.float32).unsqueeze(0)  # (1, 1, 4, 4)
    
    print("入力特徴マップ:")
    print(input_data.squeeze().numpy())
    
    # Max Pooling (2×2, stride=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output_max = max_pool(input_data)
    
    print("\nMax Pooling (2×2) の出力:")
    print(output_max.squeeze().numpy())
    
    # Average Pooling (2×2, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    output_avg = avg_pool(input_data)
    
    print("\nAverage Pooling (2×2) の出力:")
    print(output_avg.squeeze().numpy())
    
    # 手動計算による確認（左上の領域）
    print("\n手動計算（左上の2×2領域）:")
    region = input_data[0, 0, 0:2, 0:2].numpy()
    print(f"領域: \n{region}")
    print(f"Max: {region.max()}")
    print(f"Average: {region.mean()}")
    

### プーリングの効果：位置不変性
    
    
    import torch
    import torch.nn as nn
    
    # 元の特徴マップ
    original = torch.tensor([[
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    
    # 微小にシフトした特徴マップ
    shifted = torch.tensor([[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    print("元の特徴マップのMax Pooling:")
    print(max_pool(original).squeeze().numpy())
    
    print("\nシフトした特徴マップのMax Pooling:")
    print(max_pool(shifted).squeeze().numpy())
    
    print("\n→ 位置が微小に変化しても、Max Poolingの出力は同じ領域に1が出現")
    print("  これが「位置不変性」です")
    

### プーリングのパラメータ

  * **kernel_size** ：プーリング領域のサイズ（通常2×2または3×3）
  * **stride** ：スライド幅（通常kernel_sizeと同じで、重複なし）
  * **padding** ：ゼロパディング（通常0）

    
    
    import torch
    import torch.nn as nn
    
    # 異なるプーリング設定の比較
    input_data = torch.randn(1, 1, 8, 8)  # (batch, channels, height, width)
    
    # 設定1: 2×2, stride=2（標準）
    pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    output1 = pool1(input_data)
    
    # 設定2: 3×3, stride=2（重複あり）
    pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    output2 = pool2(input_data)
    
    # 設定3: 2×2, stride=1（高重複）
    pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
    output3 = pool3(input_data)
    
    print(f"入力サイズ: {input_data.shape}")
    print(f"2×2, stride=2 出力サイズ: {output1.shape}")
    print(f"3×3, stride=2 出力サイズ: {output2.shape}")
    print(f"2×2, stride=1 出力サイズ: {output3.shape}")
    
    # 次元削減率の計算
    reduction1 = (input_data.numel() - output1.numel()) / input_data.numel() * 100
    reduction2 = (input_data.numel() - output2.numel()) / input_data.numel() * 100
    
    print(f"\n2×2, stride=2 の次元削減率: {reduction1:.1f}%")
    print(f"3×3, stride=2 の次元削減率: {reduction2:.1f}%")
    

### Global Average Pooling

**Global Average Pooling (GAP)** は、特徴マップ全体の平均を取る特殊なプーリングです。現代のCNNでは、最終層の全結合層の代わりに使われることが多くなっています。
    
    
    import torch
    import torch.nn as nn
    
    # 入力: (batch_size, channels, height, width)
    input_features = torch.randn(2, 512, 7, 7)  # 2サンプル、512チャンネル、7×7
    
    # Global Average Pooling
    gap = nn.AdaptiveAvgPool2d((1, 1))  # 出力サイズを(1, 1)に指定
    output = gap(input_features)
    
    print(f"入力サイズ: {input_features.shape}")
    print(f"GAP出力サイズ: {output.shape}")
    
    # 平坦化
    output_flat = output.view(output.size(0), -1)
    print(f"平坦化後: {output_flat.shape}")
    
    # GAPのメリット
    print("\nGlobal Average Poolingの利点:")
    print("1. パラメータ数ゼロ（全結合層と比較）")
    print("2. 入力サイズに依存しない（任意のサイズに対応）")
    print("3. 過学習のリスク低減")
    print("4. 各チャンネルの空間的平均＝そのチャンネルが表す概念の強度")
    

* * *

## 2.2 代表的なCNNアーキテクチャ

### CNNの進化：歴史的概観
    
    
    ```mermaid
    graph LR
        A[LeNet-51998] --> B[AlexNet2012]
        B --> C[VGGNet2014]
        C --> D[GoogLeNet2014]
        D --> E[ResNet2015]
        E --> F[DenseNet2017]
        F --> G[EfficientNet2019]
        G --> H[Vision Transformer2020+]
    
        style A fill:#e1f5ff
        style B fill:#b3e5fc
        style C fill:#81d4fa
        style D fill:#4fc3f7
        style E fill:#29b6f6
        style F fill:#03a9f4
        style G fill:#039be5
        style H fill:#0288d1
    ```

### LeNet-5 (1998): CNNの原点

**LeNet-5** は、Yann LeCunによって開発された、手書き数字認識（MNIST）のためのネットワークです。現代のCNNの基礎となるアーキテクチャです。

層 | 出力サイズ | パラメータ数  
---|---|---  
入力 | 1×28×28 | -  
Conv1 (5×5, 6ch) | 6×24×24 | 156  
AvgPool (2×2) | 6×12×12 | 0  
Conv2 (5×5, 16ch) | 16×8×8 | 2,416  
AvgPool (2×2) | 16×4×4 | 0  
FC1 (120) | 120 | 30,840  
FC2 (84) | 84 | 10,164  
FC3 (10) | 10 | 850  
      
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LeNet5(nn.Module):
        def __init__(self, num_classes=10):
            super(LeNet5, self).__init__()
    
            # 特徴抽出層
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28×28 → 24×24
            self.pool1 = nn.AvgPool2d(kernel_size=2)        # 24×24 → 12×12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12×12 → 8×8
            self.pool2 = nn.AvgPool2d(kernel_size=2)        # 8×8 → 4×4
    
            # 分類層
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
    
        def forward(self, x):
            # 特徴抽出
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
    
            # 平坦化
            x = x.view(x.size(0), -1)
    
            # 分類
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
    
            return x
    
    # モデル作成とサマリー
    model = LeNet5(num_classes=10)
    print(model)
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # テスト実行
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"\n入力サイズ: {x.shape}")
    print(f"出力サイズ: {output.shape}")
    

### AlexNet (2012): ディープラーニングの幕開け

**AlexNet** は、2012年のImageNet競技会で圧倒的な性能を示し、ディープラーニングブームの火付け役となりました。

主な特徴：

  * **ReLU活性化関数** の使用（Sigmoidより高速に学習）
  * **Dropout** による過学習防止
  * **Data Augmentation** の活用
  * **GPU並列処理** の活用
  * **Local Response Normalization** （現在はあまり使われない）

    
    
    import torch
    import torch.nn as nn
    
    class AlexNet(nn.Module):
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
            x = x.view(x.size(0), -1)  # 平坦化
            x = self.classifier(x)
            return x
    
    # モデル作成
    model = AlexNet(num_classes=1000)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AlexNet 総パラメータ数: {total_params:,}")
    
    # 各層のサイズ確認
    x = torch.randn(1, 3, 224, 224)
    print(f"\n入力: {x.shape}")
    
    for i, layer in enumerate(model.features):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"Layer {i} ({layer.__class__.__name__}): {x.shape}")
    

### VGGNet (2014): シンプルさの美学

**VGGNet** は、3×3の小さなフィルタを繰り返し使うシンプルな設計で、深いネットワークの有効性を示しました。

設計原則：

  * **3×3フィルタのみ** 使用（小さいフィルタを重ねる方が効率的）
  * **2×2 Max Pooling** で段階的にサイズを半減
  * **チャンネル数を倍増** させながら深くする（64 → 128 → 256 → 512）
  * VGG-16（16層）とVGG-19（19層）が有名

    
    
    import torch
    import torch.nn as nn
    
    class VGGBlock(nn.Module):
        """VGGの基本ブロック：Conv → ReLU を繰り返す"""
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
        def __init__(self, num_classes=1000):
            super(VGG16, self).__init__()
    
            # 特徴抽出部分
            self.features = nn.Sequential(
                VGGBlock(3, 64, 2),      # Block 1: 64 channels, 2 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(64, 128, 2),    # Block 2: 128 channels, 2 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(128, 256, 3),   # Block 3: 256 channels, 3 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(256, 512, 3),   # Block 4: 512 channels, 3 convs
                nn.MaxPool2d(2, 2),
    
                VGGBlock(512, 512, 3),   # Block 5: 512 channels, 3 convs
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
    
    # モデル作成
    model = VGG16(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VGG-16 総パラメータ数: {total_params:,}")
    
    # なぜ3×3フィルタを2回重ねるのか？
    print("\n3×3フィルタを2回 vs 5×5フィルタを1回:")
    print("受容野: 同じ5×5")
    print("パラメータ数: 3×3×2 = 18 < 5×5 = 25")
    print("非線形性: 2回のReLU > 1回のReLU（表現力が高い）")
    

### ResNet (2015): Skip Connectionsの革命

**ResNet（Residual Network）** は、**Skip Connections（残差接続）** を導入し、非常に深いネットワーク（100層以上）の学習を可能にしました。

問題：深いネットワークほど性能が向上するはずが、実際には**勾配消失問題** で学習が困難に。

解決策：**Residual Block** を導入
    
    
    ```mermaid
    graph TD
        A["入力 x"] --> B["Conv + ReLU"]
        B --> C["Conv"]
        A --> D["Identity（そのまま）"]
        C --> E["加算 +"]
        D --> E
        E --> F["ReLU"]
        F --> G["出力"]
    
        style A fill:#e1f5ff
        style D fill:#fff9c4
        style E fill:#c8e6c9
        style G fill:#4fc3f7
    ```

数式表現：

$$ \mathbf{y} = F(\mathbf{x}) + \mathbf{x} $$ 

ここで、$F(\mathbf{x})$は残差関数（学習する部分）、$\mathbf{x}$はショートカット接続。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ResidualBlock(nn.Module):
        """ResNetの基本ブロック"""
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
    
            # Main path
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
    
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
            # Shortcut path（入出力のチャンネル数が違う場合の調整）
            self.downsample = downsample
    
        def forward(self, x):
            identity = x
    
            # Main path
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
    
            # Shortcut connection
            if self.downsample is not None:
                identity = self.downsample(x)
    
            # 加算
            out += identity
            out = F.relu(out)
    
            return out
    
    class SimpleResNet(nn.Module):
        """簡易版ResNet（CIFAR-10用）"""
        def __init__(self, num_classes=10):
            super(SimpleResNet, self).__init__()
    
            # 初期層
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
    
            # Residual blocks
            self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
            self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
    
            # 分類層
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)
    
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
    
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
    
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
    
            return nn.Sequential(*layers)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
    
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
            return x
    
    # モデル作成
    model = SimpleResNet(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet 総パラメータ数: {total_params:,}")
    
    # Skip Connectionの効果を可視化
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"\n入力: {x.shape} → 出力: {output.shape}")
    

### なぜSkip Connectionsが有効なのか

**Skip Connectionsの理論的背景**

従来のネットワークでは、深くすると勾配消失問題が発生：

$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} $$ 

層が深いと、$\frac{\partial y}{\partial x}$が何度も掛け算され、勾配が消失。

Skip Connectionsでは：

$$ \frac{\partial}{\partial x}(F(x) + x) = \frac{\partial F(x)}{\partial x} + 1 $$ 

「+1」の項があるため、勾配が必ず流れる！

さらに、ネットワークは「恒等写像を学習するだけでよい」→ 学習が容易に。

* * *

## 2.3 Batch Normalization

### Batch Normalizationとは

**Batch Normalization (BN)** は、各ミニバッチの出力を正規化し、学習を安定化させる手法です。

各層の出力を、ミニバッチ全体で平均0、分散1に正規化：

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$ $$ y_i = \gamma \hat{x}_i + \beta $$ 

ここで：

  * $\mu_B, \sigma_B^2$: ミニバッチの平均と分散
  * $\gamma, \beta$: 学習可能なパラメータ（スケールとシフト）
  * $\epsilon$: 数値安定性のための小さな値（例：1e-5）

### Batch Normalizationの効果

  * **学習の高速化** ：より大きな学習率を使える
  * **勾配の安定化** ：Internal Covariate Shiftを抑制
  * **正則化効果** ：Dropoutの必要性が減る
  * **初期値への依存低減** ：重み初期化が簡単に

    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConvBlockWithoutBN(nn.Module):
        """Batch Normalizationなし"""
        def __init__(self, in_channels, out_channels):
            super(ConvBlockWithoutBN, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
        def forward(self, x):
            return F.relu(self.conv(x))
    
    class ConvBlockWithBN(nn.Module):
        """Batch Normalizationあり"""
        def __init__(self, in_channels, out_channels):
            super(ConvBlockWithBN, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
    
        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))
    
    # 比較実験
    x = torch.randn(32, 3, 32, 32)  # バッチサイズ32
    
    # BNなし
    block_without_bn = ConvBlockWithoutBN(3, 64)
    output_without_bn = block_without_bn(x)
    
    # BNあり
    block_with_bn = ConvBlockWithBN(3, 64)
    output_with_bn = block_with_bn(x)
    
    print("=== Batch Normalizationの効果 ===")
    print(f"BNなし - 平均: {output_without_bn.mean():.4f}, 標準偏差: {output_without_bn.std():.4f}")
    print(f"BNあり - 平均: {output_with_bn.mean():.4f}, 標準偏差: {output_with_bn.std():.4f}")
    
    # 分布の可視化
    print("\n各チャンネルの統計量（BNあり）:")
    for i in range(min(5, output_with_bn.size(1))):
        channel_data = output_with_bn[:, i, :, :]
        print(f"  Channel {i}: 平均={channel_data.mean():.4f}, 標準偏差={channel_data.std():.4f}")
    

### Batch Normalizationの配置

BNは通常、**Conv → BN → Activation** の順序で配置されます：
    
    
    import torch.nn as nn
    
    # 推奨される順序
    class StandardConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(StandardConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
    
        def forward(self, x):
            x = self.conv(x)    # 1. 畳み込み
            x = self.bn(x)      # 2. Batch Normalization
            x = self.relu(x)    # 3. 活性化関数
            return x
    
    # 注意: Convにbias=Falseを指定
    # 理由: BNが平均を0にするため、biasは不要
    block = StandardConvBlock(3, 64)
    print("Conv → BN → ReLU の順序")
    print(block)
    

* * *

## 2.4 Dropoutによる過学習防止

### Dropoutとは

**Dropout** は、訓練時にランダムにニューロンを無効化（dropout）することで、過学習を防ぐ正則化手法です。

  * 訓練時：確率$p$でランダムにニューロンを0にする
  * テスト時：すべてのニューロンを使用（スケーリングあり）

    
    
    ```mermaid
    graph TD
        A["訓練時"] --> B["全ニューロン"]
        B --> C["50%をランダムにドロップ"]
        C --> D["残り50%で学習"]
    
        E["テスト時"] --> F["全ニューロンを使用"]
        F --> G["重みを0.5倍にスケール"]
    
        style A fill:#e1f5ff
        style E fill:#c8e6c9
        style D fill:#b3e5fc
        style G fill:#81d4fa
    ```

### なぜDropoutが効果的なのか

  * **アンサンブル効果** ：毎回異なるサブネットワークを訓練→複数モデルの平均に近い
  * **共適応の防止** ：特定のニューロンに依存しない頑健な表現を学習
  * **正則化** ：モデルの複雑さを抑制

    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Dropoutの動作を確認
    x = torch.ones(4, 10)  # 全て1のテンソル
    
    dropout = nn.Dropout(p=0.5)  # 50%の確率でドロップ
    
    # 訓練モード
    dropout.train()
    print("=== 訓練モード（Dropout有効） ===")
    for i in range(3):
        output = dropout(x)
        print(f"試行 {i+1}: {output[0, :5].numpy()}")  # 最初の5要素を表示
    
    # 評価モード
    dropout.eval()
    print("\n=== 評価モード（Dropout無効） ===")
    output = dropout(x)
    print(f"出力: {output[0, :5].numpy()}")
    

### CNNでのDropoutの使い方

CNNでは、通常**全結合層の前** にDropoutを配置します。畳み込み層にはあまり使いません。
    
    
    import torch.nn as nn
    
    class CNNWithDropout(nn.Module):
        def __init__(self, num_classes=10):
            super(CNNWithDropout, self).__init__()
    
            # 畳み込み層（Dropoutなし）
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
    
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
    
            # 全結合層（Dropoutあり）
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),  # Dropout
    
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),  # Dropout
    
                nn.Linear(256, num_classes),
            )
    
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = CNNWithDropout(num_classes=10)
    print(model)
    
    # Dropoutの効果を実験
    model.train()
    x = torch.randn(2, 3, 32, 32)
    output1 = model(x)
    output2 = model(x)
    print(f"\n訓練モード: 同じ入力でも出力が異なる = {not torch.allclose(output1, output2)}")
    
    model.eval()
    output3 = model(x)
    output4 = model(x)
    print(f"評価モード: 同じ入力で同じ出力 = {torch.allclose(output3, output4)}")
    

### Dropout vs Batch Normalization

項目 | Dropout | Batch Normalization  
---|---|---  
**主な目的** | 過学習防止 | 学習の安定化・高速化  
**使用箇所** | 全結合層 | 畳み込み層  
**訓練/テスト** | 動作が異なる | 動作が異なる  
**併用** | 可能（ただしBNがあれば不要な場合も） | -  
  
> **現代のベストプラクティス** ：畳み込み層にはBatch Normalization、全結合層にはDropout（必要に応じて）を使用します。ただし、BNの正則化効果により、Dropoutが不要になるケースも多いです。

* * *

## 2.5 実践：CIFAR-10画像分類

### CIFAR-10データセット

**CIFAR-10** は、10クラスの32×32カラー画像（60,000枚）からなるデータセットです：

  * クラス: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  * 訓練データ: 50,000枚
  * テストデータ: 10,000枚

### 完全なCNN分類器の実装
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データ拡張と正規化
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # データセット読み込み
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"訓練データ: {len(train_dataset)}枚")
    print(f"テストデータ: {len(test_dataset)}枚")
    
    # クラス名
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    

### モダンなCNNアーキテクチャ
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CIFAR10Net(nn.Module):
        """CIFAR-10用のモダンなCNN"""
        def __init__(self, num_classes=10):
            super(CIFAR10Net, self).__init__()
    
            # Block 1
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
    
            # Block 2
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
    
            # Block 3
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(256)
    
            # Block 4
            self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
            self.bn5 = nn.BatchNorm2d(512)
            self.conv6 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
            self.bn6 = nn.BatchNorm2d(512)
    
            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
    
            # 分類層
            self.fc = nn.Linear(512, num_classes)
    
            # Dropout
            self.dropout = nn.Dropout(0.5)
    
        def forward(self, x):
            # Block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 2
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 3
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, 2)
    
            # Block 4
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
    
            # Global Average Pooling
            x = self.gap(x)
            x = x.view(x.size(0), -1)
    
            # Dropout + 分類
            x = self.dropout(x)
            x = self.fc(x)
    
            return x
    
    # モデル作成
    model = CIFAR10Net(num_classes=10).to(device)
    print(model)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    

### 訓練ループ
    
    
    import torch.optim as optim
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Backward
            loss.backward()
            optimizer.step()
    
            # 統計
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
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
    
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    # 訓練実行
    num_epochs = 50
    best_acc = 0
    
    print("\n訓練開始...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
        scheduler.step()
    
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
        # ベストモデルの保存
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    print(f"\n訓練完了！ベスト精度: {best_acc:.2f}%")
    

### 予測と可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def imshow(img, title=None):
        """画像表示用のヘルパー関数"""
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
    
    # テストデータからサンプル取得
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 予測
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # 最初の8枚を表示
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        imshow(images[i].cpu(), title=f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
    
    plt.tight_layout()
    plt.savefig('cifar10_predictions.png', dpi=150, bbox_inches='tight')
    print("予測結果を保存しました: cifar10_predictions.png")
    

* * *

## 2.6 モダンなアーキテクチャの概観

### EfficientNet (2019): 効率的なスケーリング

**EfficientNet** は、ネットワークの深さ・幅・解像度を**バランスよくスケーリング** する手法を提案しました。

Compound Scaling:

$$ \text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi $$ 

制約条件：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

  * 少ないパラメータで高精度を達成
  * Mobile Inverted Bottleneck Convolution (MBConv)を使用
  * EfficientNet-B0からB7まで、精度とサイズのバリエーション

### Vision Transformer (2020+): CNNを超える

**Vision Transformer (ViT)** は、画像を**パッチに分割** し、Transformerで処理する新しいアプローチです。
    
    
    ```mermaid
    graph LR
        A["画像224×224"] --> B["パッチ分割16×16パッチ"]
        B --> C["Linear Projection"]
        C --> D["Transformer Encoder"]
        D --> E["分類ヘッド"]
        E --> F["クラス予測"]
    
        style A fill:#e1f5ff
        style D fill:#b3e5fc
        style F fill:#4fc3f7
    ```

特徴：

  * CNNの帰納的バイアス（局所性、平行移動不変性）を捨てる
  * 大規模データでCNNを上回る性能
  * Self-Attentionで画像全体の関係を捉える
  * 今後の主流になる可能性

### アーキテクチャ選択のガイドライン

アーキテクチャ | 特徴 | 推奨ケース  
---|---|---  
**LeNet-5** | シンプル、軽量 | MNIST、学習目的  
**VGGNet** | 理解しやすい構造 | 転移学習のベース、教育  
**ResNet** | 深いネットワーク、安定 | 一般的な画像分類、標準選択  
**EfficientNet** | 効率的、高精度 | リソース制約、モバイル  
**Vision Transformer** | 最先端、大規模データ | 大規模データセット、研究  
  
* * *

## 演習問題

**演習1：プーリング層の効果**

Max PoolingとAverage Poolingを同じ入力に適用し、出力の違いを確認してください。どのような画像特徴に対して、それぞれが有利でしょうか？
    
    
    import torch
    import torch.nn as nn
    
    # エッジを含む特徴マップを作成
    edge_feature = torch.tensor([[
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # TODO: Max PoolingとAverage Poolingを適用し、結果を比較
    # ヒント: nn.MaxPool2d と nn.AvgPool2d を使用
    

**演習2：ResidualブロックのSkip Connection**

ResidualブロックでSkip Connectionありとなしを比較し、勾配の流れにどのような違いがあるか確認してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: Skip Connectionありとなしのブロックを実装
    # ヒント: 同じ構造で、Skip Connectionの有無のみ変える
    # 勾配を比較するには、backward()後にgrad属性を確認
    

**演習3：Batch Normalizationの効果**

同じネットワークでBatch Normalizationありとなしを訓練し、収束速度と最終精度を比較してください。
    
    
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # TODO: BNありとなしのモデルを作成
    # TODO: MNIST または CIFAR-10 で訓練
    # TODO: 訓練曲線を比較
    

**演習4：Dropoutの正則化効果**

Dropoutの確率を変えて（p=0.0, 0.3, 0.5, 0.7）、過学習への影響を調べてください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 異なるDropout確率でモデルを訓練
    # TODO: 訓練誤差とテスト誤差の差（過学習の度合い）を比較
    # どのDropout確率が最適ですか？
    

**演習5：CIFAR-10でアーキテクチャ比較**

LeNet-5、VGG-style、ResNet-styleの3つのアーキテクチャをCIFAR-10で比較してください。
    
    
    import torch
    import torch.nn as nn
    
    # TODO: 3つの異なるアーキテクチャを実装
    # TODO: 同じ訓練設定で性能を比較
    # TODO: パラメータ数、訓練時間、精度を記録
    
    # 評価指標:
    # - テスト精度
    # - パラメータ数
    # - 訓練時間（1エポックあたり）
    # - 収束までのエポック数
    

* * *

## まとめ

この章では、プーリング層とCNNの代表的なアーキテクチャについて学びました。

### 重要ポイント

  * **プーリング層** ：次元削減と位置不変性を提供。Max PoolingとAverage Poolingの使い分け
  * **LeNet-5** ：CNNの基礎。Conv → Pool → FC の基本構造
  * **AlexNet** ：ReLU、Dropout、Data Augmentationの活用
  * **VGGNet** ：3×3フィルタの繰り返しによるシンプルな設計
  * **ResNet** ：Skip Connectionsで勾配消失を解決、100層以上のネットワークが可能に
  * **Batch Normalization** ：学習の安定化と高速化。Conv → BN → ReLU の順序
  * **Dropout** ：過学習防止。全結合層に使用
  * **Global Average Pooling** ：パラメータ削減、入力サイズ非依存

### 次のステップ

次章では、**転移学習** と**Fine-tuning** について学びます。事前学習済みモデルを活用し、少ないデータで高精度なモデルを構築する実践的な手法を習得します。
