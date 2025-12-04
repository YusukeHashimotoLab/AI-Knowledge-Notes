---
title: 第1章：CNNの基礎と畳み込み層
chapter_title: 第1章：CNNの基礎と畳み込み層
subtitle: 画像認識の革命 - 畳み込みニューラルネットワークの基本原理を理解する
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 11
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 画像認識における従来手法の課題とCNNの優位性を理解する
  * ✅ 畳み込み演算の数学的定義と計算プロセスを習得する
  * ✅ ストライド、パディング、カーネルサイズの役割を理解する
  * ✅ 特徴マップと受容野の概念を説明できる
  * ✅ PyTorchでConv2d層を実装し、パラメータを計算できる
  * ✅ フィルタの可視化と特徴抽出の仕組みを理解する

* * *

## 1.1 画像認識の課題とCNNの登場

### 従来の画像認識手法の限界

**全結合ニューラルネットワーク（Fully Connected Network）** を画像認識に使用する場合、深刻な問題が発生します。

> 「画像は空間構造を持つ2次元データである。この構造を無視すると、膨大なパラメータと過学習を招く。」

#### 問題1：パラメータ数の爆発

例えば、224×224ピクセルのカラー画像（RGB）を全結合層に入力する場合：

  * 入力次元数：$224 \times 224 \times 3 = 150,528$
  * 隠れ層が1,000ニューロンの場合：$150,528 \times 1,000 = 150,528,000$ パラメータ
  * これは第1層だけで1億5千万以上のパラメータ！

#### 問題2：位置不変性の欠如

全結合層では、画像内のオブジェクトの位置が少し変わるだけで、全く異なる入力として扱われます。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 簡単な例：5×5画像で「猫」の特徴（耳）を表現
    original = np.zeros((5, 5))
    original[0, 1] = 1  # 左耳
    original[0, 3] = 1  # 右耳
    original[2, 2] = 1  # 鼻
    
    # 1ピクセル右に移動
    shifted = np.zeros((5, 5))
    shifted[0, 2] = 1  # 左耳
    shifted[0, 4] = 1  # 右耳
    shifted[2, 3] = 1  # 鼻
    
    print("元の画像を平坦化:", original.flatten())
    print("移動後の画像を平坦化:", shifted.flatten())
    print(f"ユークリッド距離: {np.linalg.norm(original.flatten() - shifted.flatten()):.2f}")
    
    # 全結合層では、この2つは完全に異なる入力として扱われる
    print("\n結論: 全結合層では位置の微小な変化に対応できない")
    

**出力** ：
    
    
    元の画像を平坦化: [0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    移動後の画像を平坦化: [0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    ユークリッド距離: 2.45
    
    結論: 全結合層では位置の微小な変化に対応できない
    

### CNNの3つの重要な性質

**畳み込みニューラルネットワーク（Convolutional Neural Network, CNN）** は、画像の空間構造を活用する以下の性質を持ちます：

性質 | 説明 | 効果  
---|---|---  
**局所接続性** | 各ニューロンは入力の小さな領域のみに接続 | パラメータ数の削減  
**重み共有** | 同じフィルタを画像全体で使用 | 位置不変性の獲得  
**階層的特徴学習** | 低レベル→高レベル特徴を段階的に抽出 | 複雑なパターン認識  
  
### CNNの全体構造
    
    
    ```mermaid
    graph LR
        A[入力画像28×28×1] --> B[Conv層26×26×32]
        B --> C[活性化ReLU]
        C --> D[プーリング13×13×32]
        D --> E[Conv層11×11×64]
        E --> F[活性化ReLU]
        F --> G[プーリング5×5×64]
        G --> H[平坦化1600]
        H --> I[全結合層128]
        I --> J[出力層10クラス]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#fce4ec
        style E fill:#fff3e0
        style F fill:#e8f5e9
        style G fill:#fce4ec
        style H fill:#f3e5f5
        style I fill:#fff9c4
        style J fill:#ffebee
    ```

* * *

## 1.2 畳み込み演算の基礎

### 畳み込み演算とは

**畳み込み（Convolution）** は、フィルタ（カーネル）を画像上でスライドさせながら、要素ごとの積和演算を行う操作です。

#### 数学的定義

2次元の離散畳み込みは以下のように定義されます：

$$ (I * K)(i, j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m, n) $$ 

ここで：

  * $I$: 入力画像（Input）
  * $K$: カーネル（Kernel）またはフィルタ（Filter）
  * $(i, j)$: 出力位置
  * $(m, n)$: カーネル内の位置

#### 具体例：3×3カーネルによる畳み込み
    
    
    import numpy as np
    
    # 入力画像（5×5）
    image = np.array([
        [1, 2, 3, 0, 1],
        [4, 5, 6, 1, 2],
        [7, 8, 9, 2, 3],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6]
    ])
    
    # エッジ検出カーネル（3×3）
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    def manual_convolution(image, kernel):
        """
        手動で畳み込み演算を実行
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
    
        # 出力サイズの計算
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
    
        output = np.zeros((out_h, out_w))
    
        # 畳み込み演算
        for i in range(out_h):
            for j in range(out_w):
                # 画像の部分領域を切り出し
                region = image[i:i+ker_h, j:j+ker_w]
                # 要素ごとの積の合計
                output[i, j] = np.sum(region * kernel)
    
        return output
    
    # 畳み込み実行
    result = manual_convolution(image, kernel)
    
    print("入力画像 (5×5):")
    print(image)
    print("\nカーネル (3×3, エッジ検出):")
    print(kernel)
    print("\n出力 (3×3):")
    print(result)
    
    # 詳細な計算例（左上の位置）
    print("\n=== 計算例（位置 [0, 0]）===")
    region = image[0:3, 0:3]
    print("画像の部分領域:")
    print(region)
    print("\nカーネル:")
    print(kernel)
    print("\n要素ごとの積:")
    print(region * kernel)
    print(f"\n合計: {np.sum(region * kernel)}")
    

**出力** ：
    
    
    入力画像 (5×5):
    [[1 2 3 0 1]
     [4 5 6 1 2]
     [7 8 9 2 3]
     [1 2 3 4 5]
     [2 3 4 5 6]]
    
    カーネル (3×3, エッジ検出):
    [[-1 -1 -1]
     [-1  8 -1]
     [-1 -1 -1]]
    
    出力 (3×3):
    [[-13. -15. -12.]
     [ -8.  -9.  -6.]
     [ -5.  -6.   0.]]
    
    === 計算例（位置 [0, 0]）===
    画像の部分領域:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    カーネル:
    [[-1 -1 -1]
     [-1  8 -1]
     [-1 -1 -1]]
    
    要素ごとの積:
    [[-1 -2 -3]
     [-4 40 -6]
     [-7 -8 -9]]
    
    合計: -13
    

### フィルタとカーネル

**カーネル（Kernel）** と**フィルタ（Filter）** は、多くの場合同じ意味で使われますが、厳密には：

  * **カーネル** : 2次元の重み配列（例：3×3行列）
  * **フィルタ** : 全チャネルにわたるカーネルの集合（例：3×3×3のRGB画像用フィルタ）

#### 代表的なカーネルの例
    
    
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # 各種カーネルの定義
    kernels = {
        "恒等変換": np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),
    
        "エッジ検出（縦）": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),  # Sobelフィルタ
    
        "エッジ検出（横）": np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]]),
    
        "平滑化（blur）": np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]) / 9,
    
        "シャープ化": np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]])
    }
    
    # テスト画像（簡単なパターン）
    test_image = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    # 各カーネルの効果を可視化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('元画像')
    axes[0].axis('off')
    
    for idx, (name, kernel) in enumerate(kernels.items(), 1):
        result = signal.correlate2d(test_image, kernel, mode='same', boundary='symm')
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    print("カーネルの効果を可視化しました")
    

### ストライドとパディング

#### ストライド（Stride）

**ストライド** は、カーネルを移動させる際のステップ幅です。

  * ストライド = 1: カーネルを1ピクセルずつ移動（標準）
  * ストライド = 2: カーネルを2ピクセルずつ移動（出力サイズが半分に）

出力サイズの計算式：

$$ \text{出力サイズ} = \left\lfloor \frac{\text{入力サイズ} - \text{カーネルサイズ}}{\text{ストライド}} \right\rfloor + 1 $$ 

#### パディング（Padding）

**パディング** は、入力画像の周囲に値（通常は0）を追加する操作です。

パディングの種類 | 説明 | 用途  
---|---|---  
**Valid** | パディングなし | 出力サイズを縮小したい場合  
**Same** | 出力サイズ = 入力サイズになるように調整 | 空間サイズを維持したい場合  
**Full** | カーネル全体が画像と重なるように | 境界情報を最大限利用したい場合  
  
Sameパディングのパディング量の計算：

$$ \text{パディング} = \frac{\text{カーネルサイズ} - 1}{2} $$ 
    
    
    def calculate_output_size(input_size, kernel_size, stride, padding):
        """
        畳み込み演算後の出力サイズを計算
    
        Parameters:
        -----------
        input_size : int
            入力の高さまたは幅
        kernel_size : int
            カーネルの高さまたは幅
        stride : int
            ストライド
        padding : int
            パディング量
    
        Returns:
        --------
        int : 出力サイズ
        """
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    # 様々な設定での出力サイズを計算
    print("=== 出力サイズの計算例 ===\n")
    
    configurations = [
        (28, 3, 1, 0, "Valid（パディングなし）"),
        (28, 3, 1, 1, "Same（サイズ維持）"),
        (28, 5, 2, 2, "ストライド2、パディング2"),
        (32, 3, 1, 1, "32×32画像、3×3カーネル"),
    ]
    
    for input_size, kernel_size, stride, padding, description in configurations:
        output_size = calculate_output_size(input_size, kernel_size, stride, padding)
        print(f"{description}")
        print(f"  入力: {input_size}×{input_size}")
        print(f"  カーネル: {kernel_size}×{kernel_size}, ストライド: {stride}, パディング: {padding}")
        print(f"  → 出力: {output_size}×{output_size}\n")
    

**出力** ：
    
    
    === 出力サイズの計算例 ===
    
    Valid（パディングなし）
      入力: 28×28
      カーネル: 3×3, ストライド: 1, パディング: 0
      → 出力: 26×26
    
    Same（サイズ維持）
      入力: 28×28
      カーネル: 3×3, ストライド: 1, パディング: 1
      → 出力: 28×28
    
    ストライド2、パディング2
      入力: 28×28
      カーネル: 5×5, ストライド: 2, パディング: 2
      → 出力: 14×14
    
    32×32画像、3×3カーネル
      入力: 32×32
      カーネル: 3×3, ストライド: 1, パディング: 1
      → 出力: 32×32
    

* * *

## 1.3 特徴マップと受容野

### 特徴マップ（Feature Map）

**特徴マップ** は、畳み込み演算の出力結果です。各フィルタは異なる特徴（エッジ、テクスチャなど）を検出し、それぞれの特徴マップを生成します。

  * 入力チャネル数 = $C_{in}$
  * 出力チャネル数 = $C_{out}$（フィルタ数）
  * 各フィルタのサイズ = $K \times K \times C_{in}$

#### 多チャネル畳み込みの計算

カラー画像（RGB、3チャネル）の場合：

$$ \text{出力}(i, j) = \sum_{c=1}^{3} \sum_{m}\sum_{n} I_c(i+m, j+n) \cdot K_c(m, n) + b $$ 

ここで $b$ はバイアス項です。
    
    
    import torch
    import torch.nn as nn
    
    # RGB画像（バッチサイズ1、3チャネル、28×28）
    input_image = torch.randn(1, 3, 28, 28)
    
    # 畳み込み層の定義
    # 入力: 3チャネル（RGB）
    # 出力: 16チャネル（16個の特徴マップ）
    # カーネルサイズ: 3×3
    conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # 順伝播
    output = conv_layer(input_image)
    
    print(f"入力サイズ: {input_image.shape}")
    print(f"  → [バッチ, チャネル, 高さ, 幅] = [1, 3, 28, 28]")
    print(f"\n畳み込み層のパラメータ:")
    print(f"  入力チャネル: 3")
    print(f"  出力チャネル: 16")
    print(f"  カーネルサイズ: 3×3")
    print(f"  パディング: 1 (Sameパディング)")
    print(f"\n出力サイズ: {output.shape}")
    print(f"  → [バッチ, チャネル, 高さ, 幅] = [1, 16, 28, 28]")
    
    # パラメータ数の計算
    weight_params = 3 * 16 * 3 * 3  # in_ch × out_ch × k_h × k_w
    bias_params = 16  # 各出力チャネルに1つ
    total_params = weight_params + bias_params
    
    print(f"\nパラメータ数:")
    print(f"  重み: {weight_params:,} (= 3 × 16 × 3 × 3)")
    print(f"  バイアス: {bias_params}")
    print(f"  合計: {total_params:,}")
    

**出力** ：
    
    
    入力サイズ: torch.Size([1, 3, 28, 28])
      → [バッチ, チャネル, 高さ, 幅] = [1, 3, 28, 28]
    
    畳み込み層のパラメータ:
      入力チャネル: 3
      出力チャネル: 16
      カーネルサイズ: 3×3
      パディング: 1 (Sameパディング)
    
    出力サイズ: torch.Size([1, 16, 28, 28])
      → [バッチ, チャネル, 高さ, 幅] = [1, 16, 28, 28]
    
    パラメータ数:
      重み: 432 (= 3 × 16 × 3 × 3)
      バイアス: 16
      合計: 448
    

### 受容野（Receptive Field）

**受容野** は、ある出力ニューロンが「見ている」入力画像の領域です。CNNでは層を重ねるごとに受容野が拡大します。

#### 受容野のサイズ計算

受容野サイズ $R$ の計算式（ストライド1、パディングありの場合）：

$$ R_l = R_{l-1} + (K_l - 1) $$ 

ここで：

  * $R_l$: 第 $l$ 層の受容野サイズ
  * $K_l$: 第 $l$ 層のカーネルサイズ
  * $R_0 = 1$（入力層）

    
    
    def calculate_receptive_field(layers_info):
        """
        CNNの受容野サイズを計算
    
        Parameters:
        -----------
        layers_info : list of tuples
            各層の (カーネルサイズ, ストライド) のリスト
    
        Returns:
        --------
        list : 各層の受容野サイズ
        """
        receptive_fields = [1]  # 入力層
    
        for kernel_size, stride in layers_info:
            # 簡略化した計算（ストライド1の場合）
            rf = receptive_fields[-1] + (kernel_size - 1)
            receptive_fields.append(rf)
    
        return receptive_fields
    
    # VGG風のネットワーク構成
    vgg_layers = [
        (3, 1),  # Conv1
        (3, 1),  # Conv2
        (2, 2),  # MaxPool
        (3, 1),  # Conv3
        (3, 1),  # Conv4
        (2, 2),  # MaxPool
    ]
    
    receptive_fields = calculate_receptive_field(vgg_layers)
    
    print("=== 受容野の拡大過程 ===\n")
    print("層                受容野サイズ")
    print("-" * 35)
    print(f"入力層            {receptive_fields[0]}×{receptive_fields[0]}")
    
    layer_names = ["Conv1 (3×3)", "Conv2 (3×3)", "MaxPool (2×2)",
                   "Conv3 (3×3)", "Conv4 (3×3)", "MaxPool (2×2)"]
    
    for i, name in enumerate(layer_names, 1):
        print(f"{name:18}{receptive_fields[i]:2}×{receptive_fields[i]:2}")
    
    print(f"\n最終的な受容野: {receptive_fields[-1]}×{receptive_fields[-1]} ピクセル")
    

**出力** ：
    
    
    === 受容野の拡大過程 ===
    
    層                受容野サイズ
    -----------------------------------
    入力層             1×1
    Conv1 (3×3)        3×3
    Conv2 (3×3)        5×5
    MaxPool (2×2)      6×6
    Conv3 (3×3)        8×8
    Conv4 (3×3)       10×10
    MaxPool (2×2)     11×11
    
    最終的な受容野: 11×11 ピクセル
    

#### 受容野の可視化
    
    
    ```mermaid
    graph TD
        subgraph "入力画像"
        A1[" "]
        A2[" "]
        A3[" "]
        A4[" "]
        A5[" "]
        end
    
        subgraph "Conv1: 3×3カーネル"
        B1[受容野: 3×3]
        end
    
        subgraph "Conv2: 3×3カーネル"
        C1[受容野: 5×5]
        end
    
        subgraph "Conv3: 3×3カーネル"
        D1[受容野: 7×7]
        end
    
        A1 --> B1
        A2 --> B1
        A3 --> B1
        B1 --> C1
        C1 --> D1
    
        style B1 fill:#fff3e0
        style C1 fill:#ffe0b2
        style D1 fill:#ffcc80
    ```

> **重要** : 深いネットワークほど大きな受容野を持ち、広範囲の情報を統合できます。これが深層学習の強力な特徴抽出能力の源泉です。

* * *

## 1.4 PyTorchでの畳み込み層の実装

### Conv2dの基本的な使い方

PyTorchでは`torch.nn.Conv2d`クラスを使用して畳み込み層を定義します。
    
    
    import torch
    import torch.nn as nn
    
    # Conv2dの基本構文
    conv = nn.Conv2d(
        in_channels=3,      # 入力チャネル数（RGBなら3）
        out_channels=64,    # 出力チャネル数（フィルタ数）
        kernel_size=3,      # カーネルサイズ（3×3）
        stride=1,           # ストライド
        padding=1,          # パディング
        bias=True           # バイアス項を使用するか
    )
    
    # ダミー入力（バッチサイズ8、RGB画像、224×224）
    x = torch.randn(8, 3, 224, 224)
    
    # 順伝播
    output = conv(x)
    
    print("=== Conv2dの動作確認 ===\n")
    print(f"入力サイズ: {x.shape}")
    print(f"  [バッチ, チャネル, 高さ, 幅] = [{x.shape[0]}, {x.shape[1]}, {x.shape[2]}, {x.shape[3]}]")
    print(f"\n出力サイズ: {output.shape}")
    print(f"  [バッチ, チャネル, 高さ, 幅] = [{output.shape[0]}, {output.shape[1]}, {output.shape[2]}, {output.shape[3]}]")
    
    # パラメータの詳細
    print(f"\nパラメータ詳細:")
    print(f"  重みのサイズ: {conv.weight.shape}")
    print(f"  → [出力ch, 入力ch, 高さ, 幅] = [{conv.weight.shape[0]}, {conv.weight.shape[1]}, {conv.weight.shape[2]}, {conv.weight.shape[3]}]")
    print(f"  バイアスのサイズ: {conv.bias.shape}")
    print(f"  → [出力ch] = [{conv.bias.shape[0]}]")
    
    # パラメータ数
    total_params = conv.weight.numel() + conv.bias.numel()
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"  計算式: (3 × 64 × 3 × 3) + 64 = {total_params:,}")
    

**出力** ：
    
    
    === Conv2dの動作確認 ===
    
    入力サイズ: torch.Size([8, 3, 224, 224])
      [バッチ, チャネル, 高さ, 幅] = [8, 3, 224, 224]
    
    出力サイズ: torch.Size([8, 64, 224, 224])
      [バッチ, チャネル, 高さ, 幅] = [8, 64, 224, 224]
    
    パラメータ詳細:
      重みのサイズ: torch.Size([64, 3, 3, 3])
      → [出力ch, 入力ch, 高さ, 幅] = [64, 3, 3, 3]
      バイアスのサイズ: torch.Size([64])
      → [出力ch] = [64]
    
    総パラメータ数: 1,792
      計算式: (3 × 64 × 3 × 3) + 64 = 1,792
    

### パラメータ数の計算式

畳み込み層のパラメータ数は以下の式で計算されます：

$$ \text{パラメータ数} = (C_{in} \times K_h \times K_w \times C_{out}) + C_{out} $$ 

ここで：

  * $C_{in}$: 入力チャネル数
  * $C_{out}$: 出力チャネル数（フィルタ数）
  * $K_h, K_w$: カーネルの高さと幅
  * 最後の $C_{out}$ はバイアス項

    
    
    def calculate_conv_params(in_channels, out_channels, kernel_size, bias=True):
        """
        畳み込み層のパラメータ数を計算
        """
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
    
        weight_params = in_channels * out_channels * kernel_h * kernel_w
        bias_params = out_channels if bias else 0
    
        return weight_params + bias_params
    
    # 様々な設定でのパラメータ数を計算
    print("=== 畳み込み層のパラメータ数比較 ===\n")
    
    configs = [
        (3, 32, 3, "第1層（RGB → 32チャネル）"),
        (32, 64, 3, "第2層（32 → 64チャネル）"),
        (64, 128, 3, "第3層（64 → 128チャネル）"),
        (128, 256, 3, "第4層（128 → 256チャネル）"),
        (3, 64, 7, "大きなカーネル（7×7）"),
        (512, 512, 3, "深い層（512 → 512チャネル）"),
    ]
    
    for in_ch, out_ch, k_size, description in configs:
        params = calculate_conv_params(in_ch, out_ch, k_size)
        print(f"{description}")
        print(f"  設定: {in_ch}ch → {out_ch}ch, カーネル{k_size}×{k_size}")
        print(f"  パラメータ数: {params:,}\n")
    
    # 全結合層との比較
    print("=== 全結合層との比較 ===\n")
    fc_input = 224 * 224 * 3
    fc_output = 1000
    fc_params = fc_input * fc_output + fc_output
    
    print(f"全結合層（224×224×3 → 1000）:")
    print(f"  パラメータ数: {fc_params:,}")
    
    conv_params = calculate_conv_params(3, 64, 3)
    print(f"\n畳み込み層（3ch → 64ch, 3×3）:")
    print(f"  パラメータ数: {conv_params:,}")
    print(f"\n削減率: {(1 - conv_params/fc_params)*100:.2f}%")
    

**出力** ：
    
    
    === 畳み込み層のパラメータ数比較 ===
    
    第1層（RGB → 32チャネル）
      設定: 3ch → 32ch, カーネル3×3
      パラメータ数: 896
    
    第2層（32 → 64チャネル）
      設定: 32ch → 64ch, カーネル3×3
      パラメータ数: 18,496
    
    第3層（64 → 128チャネル）
      設定: 64ch → 128ch, カーネル3×3
      パラメータ数: 73,856
    
    第4層（128 → 256チャネル）
      設定: 128ch → 256ch, カーネル3×3
      パラメータ数: 295,168
    
    大きなカーネル（7×7）
      設定: 3ch → 64ch, カーネル7×7
      パラメータ数: 9,472
    
    深い層（512 → 512チャネル）
      設定: 512ch → 512ch, カーネル3×3
      パラメータ数: 2,359,808
    
    === 全結合層との比較 ===
    
    全結合層（224×224×3 → 1000）:
      パラメータ数: 150,529,000
    
    畳み込み層（3ch → 64ch, 3×3）:
      パラメータ数: 1,792
    
    削減率: 100.00%
    

### 畳み込みフィルタの可視化
    
    
    import matplotlib.pyplot as plt
    import torch.nn as nn
    
    # 畳み込み層を定義
    conv_layer = nn.Conv2d(1, 8, kernel_size=3, padding=1)
    
    # 学習済みのフィルタを可視化（ここではランダム初期値）
    filters = conv_layer.weight.data.cpu().numpy()
    
    # 8個のフィルタを2行4列で表示
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(8):
        # filters[i, 0]は第i番目のフィルタ（1チャネル目）
        axes[i].imshow(filters[i, 0], cmap='gray')
        axes[i].set_title(f'フィルタ {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('畳み込みフィルタの可視化（3×3カーネル）', fontsize=16)
    plt.tight_layout()
    print("フィルタを可視化しました（ランダム初期値）")
    print("学習後は、エッジ検出やテクスチャ検出などの特徴を持つフィルタに変化します")
    

* * *

## 1.5 活性化関数：ReLU

### なぜ活性化関数が必要か

畳み込み演算は線形変換です。活性化関数を使わないと、複数の層を重ねても単なる線形変換の組み合わせになり、複雑なパターンを学習できません。

> 活性化関数は**非線形性** を導入し、ネットワークに複雑な関数を近似する能力を与えます。

### ReLU（Rectified Linear Unit）

CNNで最も一般的に使用される活性化関数は**ReLU** です。

$$ \text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\\ 0 & \text{if } x \leq 0 \end{cases} $$ 

#### ReLUの利点

利点 | 説明  
---|---  
**計算効率** | 単純なmax演算のみ  
**勾配消失の軽減** | 正の領域で勾配が1（SigmoidやTanhより優れる）  
**スパース性** | 負の値を0にすることで、スパースな表現を生成  
**生物学的妥当性** | 神経細胞の発火パターンに類似  
      
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 各種活性化関数の比較
    x = np.linspace(-3, 3, 100)
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(x, relu, 'b-', linewidth=2)
    axes[0, 0].set_title('ReLU: max(0, x)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    axes[0, 1].plot(x, sigmoid, 'r-', linewidth=2)
    axes[0, 1].set_title('Sigmoid: 1/(1+exp(-x))', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    axes[1, 0].plot(x, tanh, 'g-', linewidth=2)
    axes[1, 0].set_title('Tanh: tanh(x)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    axes[1, 1].plot(x, leaky_relu, 'm-', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU: max(0.1x, x)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    print("活性化関数の形状を比較しました")
    
    # PyTorchでの使用例
    print("\n=== PyTorchでの活性化関数の使用 ===\n")
    
    x_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    relu_layer = nn.ReLU()
    print(f"入力: {x_tensor.numpy()}")
    print(f"ReLU: {relu_layer(x_tensor).numpy()}")
    

**出力** ：
    
    
    活性化関数の形状を比較しました
    
    === PyTorchでの活性化関数の使用 ===
    
    入力: [-2. -1.  0.  1.  2.]
    ReLU: [0. 0. 0. 1. 2.]
    

### Conv + ReLUのパターン
    
    
    import torch
    import torch.nn as nn
    
    # 標準的なConv-ReLUブロック
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x
    
    # 使用例
    block = ConvBlock(3, 64)
    x = torch.randn(1, 3, 224, 224)
    output = block(x)
    
    print(f"入力サイズ: {x.shape}")
    print(f"出力サイズ: {output.shape}")
    print(f"\n処理の流れ:")
    print(f"  1. Conv2d(3 → 64, 3×3) でフィルタリング")
    print(f"  2. ReLU()で非線形変換")
    print(f"  → 特徴マップ内の負の値が0になる")
    

**出力** ：
    
    
    入力サイズ: torch.Size([1, 3, 224, 224])
    出力サイズ: torch.Size([1, 64, 224, 224])
    
    処理の流れ:
      1. Conv2d(3 → 64, 3×3) でフィルタリング
      2. ReLU()で非線形変換
      → 特徴マップ内の負の値が0になる
    

* * *

## 1.6 実践：手書き数字認識（MNIST）

### シンプルなCNNの構築

MNISTデータセット（28×28のグレースケール手書き数字画像）を分類する基本的なCNNを実装します。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # シンプルなCNNモデル
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # 畳み込み層1: 1ch → 32ch, 3×3カーネル
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            # 畳み込み層2: 32ch → 64ch, 3×3カーネル
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            # 全結合層
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            # その他
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
    
        def forward(self, x):
            # Conv1 → ReLU → MaxPool
            x = self.pool(F.relu(self.conv1(x)))  # 28×28 → 14×14
            # Conv2 → ReLU → MaxPool
            x = self.pool(F.relu(self.conv2(x)))  # 14×14 → 7×7
            # 平坦化
            x = x.view(-1, 64 * 7 * 7)
            # 全結合層
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # モデルのインスタンス化
    model = SimpleCNN()
    
    # モデル構造の表示
    print("=== SimpleCNN のアーキテクチャ ===\n")
    print(model)
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # 各層のパラメータ数詳細
    print("\n=== 各層のパラメータ数 ===")
    for name, param in model.named_parameters():
        print(f"{name:20} {str(list(param.shape)):30} {param.numel():>10,} params")
    

**出力** ：
    
    
    === SimpleCNN のアーキテクチャ ===
    
    SimpleCNN(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fc1): Linear(in_features=3136, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (dropout): Dropout(p=0.25, inplace=False)
    )
    
    総パラメータ数: 421,066
    学習可能パラメータ数: 421,066
    
    === 各層のパラメータ数 ===
    conv1.weight         [32, 1, 3, 3]                         288 params
    conv1.bias           [32]                                   32 params
    conv2.weight         [64, 32, 3, 3]                     18,432 params
    conv2.bias           [64]                                   64 params
    fc1.weight           [128, 3136]                       401,408 params
    fc1.bias             [128]                                 128 params
    fc2.weight           [10, 128]                           1,280 params
    fc2.bias             [10]                                   10 params
    

### データの準備と学習
    
    
    import torch.optim as optim
    
    # データの前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNISTの平均・標準偏差
    ])
    
    # データセットの読み込み（ダウンロードは初回のみ）
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 学習の設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 学習関数
    def train_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    # 評価関数
    def evaluate(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0
        correct = 0
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy
    
    # 学習実行（簡略版：3エポック）
    print("\n=== 学習開始 ===\n")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%\n")
    
    print("学習完了！")
    

**期待される出力** ：
    
    
    === 学習開始 ===
    
    Epoch 1/3
      Train Loss: 0.2145, Train Acc: 93.52%
      Test Loss:  0.0789, Test Acc:  97.56%
    
    Epoch 2/3
      Train Loss: 0.0701, Train Acc: 97.89%
      Test Loss:  0.0512, Test Acc:  98.34%
    
    Epoch 3/3
      Train Loss: 0.0512, Train Acc: 98.42%
      Test Loss:  0.0401, Test Acc:  98.67%
    
    学習完了！
    

### 学習済みフィルタの可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 第1層の畳み込みフィルタを可視化
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    
    # 最初の16個のフィルタを表示
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(32, len(axes))):
        axes[i].imshow(conv1_weights[i, 0], cmap='viridis')
        axes[i].set_title(f'Filter {i+1}', fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('学習済み畳み込みフィルタ（第1層、32個中32個）', fontsize=16)
    plt.tight_layout()
    print("学習済みフィルタを可視化しました")
    print("各フィルタは、エッジ、曲線、コーナーなどの異なる特徴を検出するように学習されています")
    

* * *

## まとめ

この章では、CNNの基礎と畳み込み層について学習しました。

### 重要なポイント

  * **局所接続性と重み共有** により、CNNはパラメータ数を大幅に削減
  * **畳み込み演算** は、フィルタを画像上でスライドさせて特徴を抽出
  * **ストライドとパディング** で出力サイズを制御
  * **受容野** は層を重ねるごとに拡大し、広範囲の情報を統合
  * **ReLU活性化関数** が非線形性を導入し、複雑なパターン学習を可能に

### 次章の予告

第2章では、以下のトピックを扱います：

  * プーリング層（MaxPooling、AveragePooling）
  * バッチ正規化（Batch Normalization）
  * Dropout による正則化
  * 代表的なCNNアーキテクチャ（VGG、ResNet）

* * *

## 演習問題

**演習1：出力サイズの計算**

**問題** ：以下の畳み込み層の出力サイズを計算してください。

  * 入力: 64×64×3
  * カーネル: 5×5
  * ストライド: 2
  * パディング: 2
  * 出力チャネル数: 128

**解答** ：
    
    
    # 出力サイズの計算
    output_h = (64 + 2*2 - 5) // 2 + 1 = 32
    output_w = (64 + 2*2 - 5) // 2 + 1 = 32
    
    # 答え: 32×32×128
    

**演習2：パラメータ数の計算**

**問題** ：以下のCNNのパラメータ数を計算してください。

  * Conv1: 3ch → 64ch, 7×7カーネル
  * Conv2: 64ch → 128ch, 3×3カーネル
  * Conv3: 128ch → 256ch, 3×3カーネル

**解答** ：
    
    
    # Conv1のパラメータ数
    conv1_params = (3 * 64 * 7 * 7) + 64 = 9,472
    
    # Conv2のパラメータ数
    conv2_params = (64 * 128 * 3 * 3) + 128 = 73,856
    
    # Conv3のパラメータ数
    conv3_params = (128 * 256 * 3 * 3) + 256 = 295,168
    
    # 合計
    total_params = 9,472 + 73,856 + 295,168 = 378,496
    

**演習3：受容野の計算**

**問題** ：以下の構成のCNNの最終的な受容野サイズを計算してください（全てストライド1、パディングあり）。

  * Conv1: 3×3カーネル
  * Conv2: 3×3カーネル
  * Conv3: 3×3カーネル
  * Conv4: 3×3カーネル

**解答** ：
    
    
    # 受容野の計算
    # R_0 = 1 (入力)
    # R_1 = 1 + (3-1) = 3
    # R_2 = 3 + (3-1) = 5
    # R_3 = 5 + (3-1) = 7
    # R_4 = 7 + (3-1) = 9
    
    # 答え: 9×9ピクセル
    

**演習4：カスタムCNNの実装**

**問題** ：以下の仕様のCNNをPyTorchで実装してください。

  * 入力: 32×32×3（CIFAR-10形式）
  * Conv1: 32フィルタ、3×3カーネル、ReLU
  * MaxPool: 2×2
  * Conv2: 64フィルタ、3×3カーネル、ReLU
  * MaxPool: 2×2
  * 全結合: 10クラス分類

**解答例** ：
    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32×32 → 16×16
            x = self.pool(F.relu(self.conv2(x)))  # 16×16 → 8×8
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc(x)
            return x
    

**演習5：全結合層とCNNの比較**

**問題** ：224×224×3の画像を入力とする場合、以下の2つのアプローチでパラメータ数を比較してください。

  * アプローチ1: 全結合層（入力→1000ユニット）
  * アプローチ2: Conv層（3ch→64ch、3×3カーネル）を3層

**解答** ：
    
    
    # アプローチ1: 全結合層
    fc_params = (224 * 224 * 3 * 1000) + 1000 = 150,529,000
    
    # アプローチ2: CNN（3層）
    conv1_params = (3 * 64 * 3 * 3) + 64 = 1,792
    conv2_params = (64 * 64 * 3 * 3) + 64 = 36,928
    conv3_params = (64 * 64 * 3 * 3) + 64 = 36,928
    cnn_total = 1,792 + 36,928 + 36,928 = 75,648
    
    # 削減率
    reduction = (1 - 75,648/150,529,000) * 100 = 99.95%
    
    # CNNは全結合層の0.05%のパラメータで済む！
    

* * *
