---
title: 第1章：画像処理の基礎
chapter_title: 第1章：画像処理の基礎
subtitle: コンピュータビジョンの第一歩 - デジタル画像の表現と基本処理を理解する
reading_time: 30-35分
difficulty: 初級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ デジタル画像の表現方法（ピクセル、チャンネル）を理解する
  * ✅ RGB、HSV、LABなどのカラー空間の違いと用途を説明できる
  * ✅ OpenCVとPILを使った画像の読み込み・保存・表示ができる
  * ✅ リサイズ、回転、色変換などの基本的な画像処理を実装できる
  * ✅ 平滑化、エッジ検出などのフィルタリング手法を適用できる
  * ✅ 機械学習のための画像データ前処理パイプラインを構築できる

* * *

## 1.1 画像の表現

### デジタル画像とピクセル

**デジタル画像** は、離散的な点（ピクセル）の集合として表現されます。各ピクセルは色情報（強度値）を持ちます。

> 「画像はピクセルの2次元配列として扱われ、数値計算可能なデータ構造になります。」

#### 画像の基本構造

  * **高さ（Height）** : 縦方向のピクセル数
  * **幅（Width）** : 横方向のピクセル数
  * **チャンネル（Channel）** : 色成分の数（グレースケール=1、RGB=3）

画像の形状は通常、以下のいずれかの形式で表現されます：

形式 | 次元順序 | 使用ライブラリ  
---|---|---  
**HWC** | (Height, Width, Channels) | OpenCV, PIL, matplotlib  
**CHW** | (Channels, Height, Width) | PyTorch, Caffe  
**NHWC** | (Batch, Height, Width, Channels) | TensorFlow, Keras  
**NCHW** | (Batch, Channels, Height, Width) | PyTorch  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # グレースケール画像の作成（高さ5、幅5）
    gray_image = np.array([
        [0, 50, 100, 150, 200],
        [50, 100, 150, 200, 250],
        [100, 150, 200, 250, 255],
        [150, 200, 250, 255, 200],
        [200, 250, 255, 200, 150]
    ], dtype=np.uint8)
    
    # RGB画像の作成（高さ3、幅3、チャンネル3）
    rgb_image = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb_image[0, 0] = [255, 0, 0]      # 赤
    rgb_image[0, 1] = [0, 255, 0]      # 緑
    rgb_image[0, 2] = [0, 0, 255]      # 青
    rgb_image[1, 1] = [255, 255, 0]    # 黄
    rgb_image[2, 2] = [255, 255, 255]  # 白
    
    print("=== グレースケール画像 ===")
    print(f"形状: {gray_image.shape}")
    print(f"データ型: {gray_image.dtype}")
    print(f"最小値: {gray_image.min()}, 最大値: {gray_image.max()}")
    print(f"\n画像データ:\n{gray_image}")
    
    print("\n=== RGB画像 ===")
    print(f"形状: {rgb_image.shape} (Height, Width, Channels)")
    print(f"左上のピクセル値 (R,G,B): {rgb_image[0, 0]}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('グレースケール画像 (5×5)')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_image)
    axes[1].set_title('RGB画像 (3×3)')
    axes[1].axis('off')
    
    plt.tight_layout()
    print("\n画像を可視化しました")
    

### カラー空間（Color Space）

カラー空間は、色を数値で表現する方法です。用途に応じて適切なカラー空間を選択することが重要です。

#### RGB（Red, Green, Blue）

最も一般的なカラー空間。加法混色（光の三原色）に基づきます。

  * 各チャンネルの値域: 0〜255（8bit整数）または 0.0〜1.0（浮動小数点）
  * 用途: デジタルカメラ、ディスプレイ、画像保存
  * 特徴: 直感的だが、人間の色知覚とは一致しない

#### HSV（Hue, Saturation, Value）

色相、彩度、明度で色を表現します。

  * **Hue（色相）** : 0〜179度（OpenCVでは0〜180）、色の種類
  * **Saturation（彩度）** : 0〜255、色の鮮やかさ
  * **Value（明度）** : 0〜255、色の明るさ
  * 用途: 色ベースの物体検出、画像セグメンテーション

#### LAB（L*a*b*）

人間の視覚に近い色空間です。

  * **L** : 輝度（0〜100）
  * **a** : 緑〜赤の軸（-128〜127）
  * **b** : 青〜黄の軸（-128〜127）
  * 用途: 色差計算、画像補正

    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル画像の作成（赤、緑、青のブロック）
    sample = np.zeros((100, 300, 3), dtype=np.uint8)
    sample[:, 0:100] = [255, 0, 0]     # 赤
    sample[:, 100:200] = [0, 255, 0]   # 緑
    sample[:, 200:300] = [0, 0, 255]   # 青
    
    # BGRからRGBへ変換（OpenCVはBGR順序を使用）
    sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    
    # 各種カラー空間への変換
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 元画像（RGB）
    axes[0, 0].imshow(sample_rgb)
    axes[0, 0].set_title('Original (RGB)')
    axes[0, 0].axis('off')
    
    # HSV（各チャンネルを個別表示）
    axes[0, 1].imshow(sample_hsv[:, :, 0], cmap='hsv')
    axes[0, 1].set_title('HSV - Hue（色相）')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sample_hsv[:, :, 1], cmap='gray')
    axes[0, 2].set_title('HSV - Saturation（彩度）')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sample_hsv[:, :, 2], cmap='gray')
    axes[0, 3].set_title('HSV - Value（明度）')
    axes[0, 3].axis('off')
    
    # LAB（各チャンネルを個別表示）
    axes[1, 0].imshow(sample_lab[:, :, 0], cmap='gray')
    axes[1, 0].set_title('LAB - L（輝度）')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sample_lab[:, :, 1], cmap='RdYlGn_r')
    axes[1, 1].set_title('LAB - a（緑〜赤）')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sample_lab[:, :, 2], cmap='RdYlBu_r')
    axes[1, 2].set_title('LAB - b（青〜黄）')
    axes[1, 2].axis('off')
    
    # グレースケール
    axes[1, 3].imshow(sample_gray, cmap='gray')
    axes[1, 3].set_title('Grayscale')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    print("各種カラー空間への変換を可視化しました")
    
    # 数値の確認
    print(f"\n中央の赤ブロックのピクセル値（位置 [50, 50]）:")
    print(f"  RGB: {sample_rgb[50, 50]}")
    print(f"  HSV: {sample_hsv[50, 50]}")
    print(f"  LAB: {sample_lab[50, 50]}")
    print(f"  Gray: {sample_gray[50, 50]}")
    

### 画像の読み込みと保存

#### OpenCVを使用
    
    
    import cv2
    import numpy as np
    
    # 画像の読み込み（存在しない場合はダミー画像を作成）
    try:
        image = cv2.imread('sample.jpg')
        if image is None:
            raise FileNotFoundError
    except:
        # ダミー画像の作成（グラデーション）
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                image[i, j] = [i * 255 // 480, j * 255 // 640, 128]
        print("ダミー画像を作成しました")
    
    print(f"画像の形状: {image.shape}")
    print(f"データ型: {image.dtype}")
    
    # グレースケールとして読み込み
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 画像の保存
    cv2.imwrite('output_color.jpg', image)
    cv2.imwrite('output_gray.jpg', gray)
    
    print("\n画像を保存しました:")
    print("  - output_color.jpg (カラー)")
    print("  - output_gray.jpg (グレースケール)")
    
    # 画像情報の表示
    print(f"\nカラー画像:")
    print(f"  形状: {image.shape}")
    print(f"  メモリサイズ: {image.nbytes / 1024:.2f} KB")
    
    print(f"\nグレースケール画像:")
    print(f"  形状: {gray.shape}")
    print(f"  メモリサイズ: {gray.nbytes / 1024:.2f} KB")
    

#### PILを使用
    
    
    from PIL import Image
    import numpy as np
    
    # PIL Imageの作成（グラデーション）
    width, height = 640, 480
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            array[i, j] = [i * 255 // height, j * 255 // width, 128]
    
    pil_image = Image.fromarray(array)
    
    print(f"PIL画像のサイズ: {pil_image.size}")  # (width, height)
    print(f"PIL画像のモード: {pil_image.mode}")
    
    # グレースケール変換
    pil_gray = pil_image.convert('L')
    
    # 保存
    pil_image.save('output_pil_color.png')
    pil_gray.save('output_pil_gray.png')
    
    # PIL → NumPy配列
    np_array = np.array(pil_image)
    print(f"\nNumPy配列に変換:")
    print(f"  形状: {np_array.shape}")
    
    # NumPy配列 → PIL
    pil_from_numpy = Image.fromarray(np_array)
    print(f"\nPIL画像に変換:")
    print(f"  サイズ: {pil_from_numpy.size}")
    
    print("\nPILを使用した画像を保存しました:")
    print("  - output_pil_color.png")
    print("  - output_pil_gray.png")
    

* * *

## 1.2 基本的な画像処理

### リサイズとクロップ

**リサイズ** は画像のサイズを変更する操作です。補間方法によって品質が変わります。

補間方法 | 特徴 | 用途  
---|---|---  
**NEAREST** | 最近傍補間、高速だが品質低い | 整数倍の拡大・縮小  
**LINEAR** | 線形補間、バランスが良い | 一般的な縮小  
**CUBIC** | 3次補間、高品質だが低速 | 拡大処理  
**LANCZOS** | 最高品質、最も低速 | 高品質が必要な場合  
      
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル画像の作成（チェッカーボード）
    def create_checkerboard(size=200, square_size=20):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = [255, 255, 255]
        return image
    
    original = create_checkerboard(200, 20)
    
    # 各種補間方法でリサイズ
    resized_nearest = cv2.resize(original, (400, 400), interpolation=cv2.INTER_NEAREST)
    resized_linear = cv2.resize(original, (400, 400), interpolation=cv2.INTER_LINEAR)
    resized_cubic = cv2.resize(original, (400, 400), interpolation=cv2.INTER_CUBIC)
    resized_lanczos = cv2.resize(original, (400, 400), interpolation=cv2.INTER_LANCZOS4)
    
    # 縮小
    resized_small = cv2.resize(original, (100, 100), interpolation=cv2.INTER_AREA)
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original (200×200)"),
        (cv2.cvtColor(resized_nearest, cv2.COLOR_BGR2RGB), "NEAREST (400×400)"),
        (cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB), "LINEAR (400×400)"),
        (cv2.cvtColor(resized_cubic, cv2.COLOR_BGR2RGB), "CUBIC (400×400)"),
        (cv2.cvtColor(resized_lanczos, cv2.COLOR_BGR2RGB), "LANCZOS (400×400)"),
        (cv2.cvtColor(resized_small, cv2.COLOR_BGR2RGB), "AREA (100×100 縮小)"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("各種補間方法によるリサイズを比較しました")
    
    # クロップ（トリミング）
    print("\n=== クロップの例 ===")
    height, width = original.shape[:2]
    x, y, w, h = 50, 50, 100, 100  # (x, y, width, height)
    
    cropped = original[y:y+h, x:x+w]
    print(f"元画像: {original.shape}")
    print(f"クロップ後: {cropped.shape}")
    print(f"クロップ範囲: x={x}, y={y}, width={w}, height={h}")
    

### 回転と反転
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # テスト画像の作成（矢印パターン）
    def create_arrow(size=200):
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        # 矢印を描画
        pts = np.array([[100, 50], [150, 100], [125, 100], [125, 150],
                        [75, 150], [75, 100], [50, 100]], np.int32)
        cv2.fillPoly(image, [pts], (0, 0, 255))
        return image
    
    original = create_arrow()
    
    # 回転（90度、180度、270度）
    rotated_90 = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(original, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 任意角度の回転（45度）
    height, width = original.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_45 = cv2.warpAffine(original, rotation_matrix, (width, height))
    
    # 反転
    flipped_horizontal = cv2.flip(original, 1)  # 左右反転
    flipped_vertical = cv2.flip(original, 0)    # 上下反転
    flipped_both = cv2.flip(original, -1)       # 両方向反転
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB), "回転 90°"),
        (cv2.cvtColor(rotated_180, cv2.COLOR_BGR2RGB), "回転 180°"),
        (cv2.cvtColor(rotated_270, cv2.COLOR_BGR2RGB), "回転 270°"),
        (cv2.cvtColor(rotated_45, cv2.COLOR_BGR2RGB), "回転 45°"),
        (cv2.cvtColor(flipped_horizontal, cv2.COLOR_BGR2RGB), "左右反転"),
        (cv2.cvtColor(flipped_vertical, cv2.COLOR_BGR2RGB), "上下反転"),
        (cv2.cvtColor(flipped_both, cv2.COLOR_BGR2RGB), "両方向反転"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("回転と反転の操作を可視化しました")
    
    # 回転行列の詳細
    print("\n=== 45度回転の変換行列 ===")
    print(rotation_matrix)
    

### 色変換とヒストグラム
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル画像の作成（グラデーション + ノイズ）
    def create_sample_image():
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(400):
                image[i, j] = [
                    int(i * 255 / 300),
                    int(j * 255 / 400),
                    128
                ]
        # ノイズを追加
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image
    
    image = create_sample_image()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ヒストグラムの計算
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元画像
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('元画像')
    axes[0, 0].axis('off')
    
    # RGBヒストグラム
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        axes[0, 1].plot(hist, color=color, label=f'{color.upper()} channel')
    axes[0, 1].set_title('RGBヒストグラム')
    axes[0, 1].set_xlabel('ピクセル値')
    axes[0, 1].set_ylabel('頻度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ヒストグラム均等化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('ヒストグラム均等化後')
    axes[1, 0].axis('off')
    
    # 均等化前後のヒストグラム比較
    hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    axes[1, 1].plot(hist_before, 'b-', label='均等化前', alpha=0.7)
    axes[1, 1].plot(hist_after, 'r-', label='均等化後', alpha=0.7)
    axes[1, 1].set_title('グレースケールヒストグラム')
    axes[1, 1].set_xlabel('ピクセル値')
    axes[1, 1].set_ylabel('頻度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("ヒストグラムと均等化を可視化しました")
    
    # 統計情報
    print("\n=== 画像の統計情報 ===")
    print(f"平均値（均等化前）: {gray.mean():.2f}")
    print(f"平均値（均等化後）: {equalized.mean():.2f}")
    print(f"標準偏差（均等化前）: {gray.std():.2f}")
    print(f"標準偏差（均等化後）: {equalized.std():.2f}")
    

* * *

## 1.3 フィルタリング

### 平滑化フィルタ

**平滑化** は、画像のノイズを除去したり、ぼかし効果を与える処理です。
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ノイズを含む画像の作成
    def create_noisy_image(size=200):
        # クリーンな画像（円）
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(image, (size//2, size//2), size//3, 255, -1)
    
        # ガウシアンノイズを追加
        noise = np.random.normal(0, 25, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    
        # ソルト&ペッパーノイズを追加
        salt_pepper = noisy.copy()
        salt = np.random.random(image.shape) < 0.02
        pepper = np.random.random(image.shape) < 0.02
        salt_pepper[salt] = 255
        salt_pepper[pepper] = 0
    
        return image, noisy, salt_pepper
    
    clean, gaussian_noisy, sp_noisy = create_noisy_image()
    
    # 各種平滑化フィルタの適用
    # 平均フィルタ（Mean Filter）
    mean_blur = cv2.blur(gaussian_noisy, (5, 5))
    
    # ガウシアンフィルタ（Gaussian Filter）
    gaussian_blur = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
    
    # メディアンフィルタ（Median Filter）- ソルト&ペッパーノイズに効果的
    median_blur = cv2.medianBlur(sp_noisy, 5)
    
    # バイラテラルフィルタ（Bilateral Filter）- エッジを保持しながら平滑化
    bilateral = cv2.bilateralFilter(gaussian_noisy, 9, 75, 75)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (clean, "クリーン画像"),
        (gaussian_noisy, "ガウシアンノイズ"),
        (mean_blur, "平均フィルタ"),
        (gaussian_blur, "ガウシアンフィルタ"),
        (sp_noisy, "ソルト&ペッパーノイズ"),
        (median_blur, "メディアンフィルタ"),
        (bilateral, "バイラテラルフィルタ"),
        (gaussian_noisy - gaussian_blur, "ノイズ成分（差分）"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("各種平滑化フィルタの効果を比較しました")
    
    # PSNR（ピーク信号対雑音比）の計算
    def calculate_psnr(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    print("\n=== ノイズ除去性能（PSNR、高いほど良い）===")
    print(f"ノイズ画像: {calculate_psnr(clean, gaussian_noisy):.2f} dB")
    print(f"平均フィルタ: {calculate_psnr(clean, mean_blur):.2f} dB")
    print(f"ガウシアンフィルタ: {calculate_psnr(clean, gaussian_blur):.2f} dB")
    print(f"バイラテラルフィルタ: {calculate_psnr(clean, bilateral):.2f} dB")
    

### エッジ検出

**エッジ検出** は、画像内の急激な輝度変化を検出する処理です。
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # テスト画像の作成（複数の図形）
    def create_shapes_image(size=300):
        image = np.ones((size, size), dtype=np.uint8) * 200
        # 四角形
        cv2.rectangle(image, (50, 50), (120, 120), 50, -1)
        # 円
        cv2.circle(image, (220, 80), 40, 100, -1)
        # 三角形
        pts = np.array([[150, 200], [100, 280], [200, 280]], np.int32)
        cv2.fillPoly(image, [pts], 150)
        return image
    
    image = create_shapes_image()
    
    # Sobelフィルタ（X方向、Y方向）
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # Laplacianフィルタ
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Cannyエッジ検出
    canny = cv2.Canny(image, 50, 150)
    
    # Scharrフィルタ（Sobelより精度が高い）
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(scharr_combined / scharr_combined.max() * 255)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (image, "元画像"),
        (sobel_x, "Sobel X"),
        (sobel_y, "Sobel Y"),
        (sobel_combined, "Sobel 結合"),
        (laplacian, "Laplacian"),
        (canny, "Canny"),
        (scharr_combined, "Scharr 結合"),
        (image, "元画像（参考）"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("各種エッジ検出フィルタを比較しました")
    
    # エッジ強度の分析
    print("\n=== エッジ検出結果の統計 ===")
    print(f"Sobel: 平均強度 = {sobel_combined.mean():.2f}")
    print(f"Laplacian: 平均強度 = {laplacian.mean():.2f}")
    print(f"Canny: エッジピクセル数 = {np.sum(canny > 0)}")
    print(f"Scharr: 平均強度 = {scharr_combined.mean():.2f}")
    

### モルフォロジー演算

**モルフォロジー演算** は、二値画像に対する形状処理です。

演算 | 効果 | 用途  
---|---|---  
**膨張（Dilation）** | 白領域を拡大 | 穴を埋める、途切れを接続  
**収縮（Erosion）** | 白領域を縮小 | ノイズ除去、細い線を除去  
**オープニング** | 収縮→膨張 | 小さなノイズ除去  
**クロージング** | 膨張→収縮 | 穴や隙間を埋める  
      
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ノイズを含む二値画像の作成
    def create_noisy_binary_image(size=200):
        image = np.zeros((size, size), dtype=np.uint8)
        # 主要な形状（長方形）
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        # 小さな穴を追加
        for _ in range(10):
            x, y = np.random.randint(60, 140, 2)
            cv2.circle(image, (x, y), 3, 0, -1)
        # 小さなノイズを追加
        for _ in range(20):
            x, y = np.random.randint(0, size, 2)
            cv2.circle(image, (x, y), 2, 255, -1)
        return image
    
    binary = create_noisy_binary_image()
    
    # カーネル（構造要素）の作成
    kernel = np.ones((5, 5), np.uint8)
    
    # モルフォロジー演算
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (binary, "元画像（ノイズ含む）"),
        (erosion, "収縮（Erosion）"),
        (dilation, "膨張（Dilation）"),
        (opening, "オープニング"),
        (closing, "クロージング"),
        (gradient, "勾配（Gradient）"),
        (tophat, "トップハット"),
        (blackhat, "ブラックハット"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("モルフォロジー演算を可視化しました")
    
    # ピクセル数の比較
    print("\n=== 白ピクセル数の変化 ===")
    print(f"元画像: {np.sum(binary == 255):,} pixels")
    print(f"収縮後: {np.sum(erosion == 255):,} pixels ({np.sum(erosion == 255) / np.sum(binary == 255) * 100:.1f}%)")
    print(f"膨張後: {np.sum(dilation == 255):,} pixels ({np.sum(dilation == 255) / np.sum(binary == 255) * 100:.1f}%)")
    print(f"オープニング後: {np.sum(opening == 255):,} pixels")
    print(f"クロージング後: {np.sum(closing == 255):,} pixels")
    

* * *

## 1.4 特徴抽出

### コーナー検出

**コーナー** は、画像内の重要な特徴点です。物体認識や追跡に使用されます。

#### Harris Corner Detection
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # チェッカーボードパターンの作成
    def create_checkerboard_complex(size=400):
        image = np.zeros((size, size), dtype=np.uint8)
        square_size = 40
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
        # 追加の形状
        cv2.circle(image, (300, 300), 50, 128, -1)
        return image
    
    image = create_checkerboard_complex()
    
    # Harrisコーナー検出
    harris = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)  # 結果を強調
    
    # 閾値処理でコーナーを検出
    image_harris = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    
    # Shi-Tomasiコーナー検出（Good Features to Track）
    corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
    image_shi = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_shi, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('元画像')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image_harris, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Harris Corner Detection')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(image_shi, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Shi-Tomasi ({len(corners) if corners is not None else 0} corners)')
    axes[2].axis('off')
    
    plt.tight_layout()
    print("コーナー検出アルゴリズムを比較しました")
    
    if corners is not None:
        print(f"\n検出されたコーナー数: {len(corners)}")
        print(f"最初の5つのコーナー座標:")
        for i, corner in enumerate(corners[:5]):
            x, y = corner.ravel()
            print(f"  コーナー{i+1}: ({x:.1f}, {y:.1f})")
    

### SIFT / ORB特徴量

**SIFT（Scale-Invariant Feature Transform）** とは、スケールや回転に不変な特徴量です。

> 注意: OpenCVの一部のバージョンでは、SIFTは特許の関係でopencv-contribに含まれています。ここではORB（Oriented FAST and Rotated BRIEF）を主に使用します。
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 特徴的な画像の作成
    def create_feature_image(size=400):
        image = np.ones((size, size), dtype=np.uint8) * 200
        # 複数の図形
        cv2.rectangle(image, (50, 50), (150, 150), 50, -1)
        cv2.circle(image, (300, 100), 50, 100, -1)
        cv2.rectangle(image, (100, 250), (200, 350), 150, 3)
        pts = np.array([[250, 250], [350, 280], [320, 350]], np.int32)
        cv2.fillPoly(image, [pts], 80)
        return image
    
    image = create_feature_image()
    
    # ORB特徴量検出器の作成
    orb = cv2.ORB_create(nfeatures=100)
    
    # キーポイントと記述子の検出
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # キーポイントを描画
    image_keypoints = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('元画像')
    axes[0].axis('off')
    
    axes[1].imshow(image_keypoints)
    axes[1].set_title(f'ORB特徴点 ({len(keypoints)} points)')
    axes[1].axis('off')
    
    plt.tight_layout()
    print(f"ORB特徴量を検出しました: {len(keypoints)} 個のキーポイント")
    
    # 特徴量の詳細
    print("\n=== ORB特徴量の詳細 ===")
    print(f"キーポイント数: {len(keypoints)}")
    if descriptors is not None:
        print(f"記述子の形状: {descriptors.shape}")
        print(f"  各キーポイントは {descriptors.shape[1]} 次元のベクトルで記述される")
    
    # 最初の5つのキーポイント情報
    print("\n最初の5つのキーポイント:")
    for i, kp in enumerate(keypoints[:5]):
        print(f"  Point {i+1}: 位置=({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), "
              f"サイズ={kp.size:.1f}, 角度={kp.angle:.1f}°")
    

### HOG（Histogram of Oriented Gradients）

**HOG** は、勾配方向のヒストグラムを特徴量として使用します。歩行者検出などに広く使用されます。
    
    
    from skimage.feature import hog
    from skimage import exposure
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 人型シルエットの簡易作成
    def create_person_silhouette(size=128):
        image = np.zeros((size, size), dtype=np.uint8)
        # 頭
        cv2.circle(image, (size//2, size//4), size//8, 255, -1)
        # 体
        cv2.rectangle(image, (size//2 - size//10, size//4 + size//10),
                      (size//2 + size//10, size//2 + size//6), 255, -1)
        # 腕
        cv2.line(image, (size//2 - size//10, size//4 + size//6),
                 (size//2 - size//4, size//2), 255, size//20)
        cv2.line(image, (size//2 + size//10, size//4 + size//6),
                 (size//2 + size//4, size//2), 255, size//20)
        # 脚
        cv2.line(image, (size//2 - size//20, size//2 + size//6),
                 (size//2 - size//10, size - size//8), 255, size//20)
        cv2.line(image, (size//2 + size//20, size//2 + size//6),
                 (size//2 + size//10, size - size//8), 255, size//20)
        return image
    
    image = create_person_silhouette()
    
    # HOG特徴量の計算
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    
    # HOG画像のコントラストを強調
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('元画像（人型シルエット）')
    axes[0].axis('off')
    
    axes[1].imshow(hog_image_rescaled, cmap='gray')
    axes[1].set_title('HOG特徴量の可視化')
    axes[1].axis('off')
    
    # HOG特徴量ベクトルの一部を表示
    axes[2].bar(range(min(100, len(fd))), fd[:100])
    axes[2].set_title('HOG特徴量ベクトル（最初の100次元）')
    axes[2].set_xlabel('次元')
    axes[2].set_ylabel('値')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("HOG特徴量を計算しました")
    
    print(f"\n=== HOG特徴量の詳細 ===")
    print(f"特徴量ベクトルの次元数: {len(fd)}")
    print(f"平均値: {fd.mean():.4f}")
    print(f"標準偏差: {fd.std():.4f}")
    print(f"最大値: {fd.max():.4f}")
    print(f"最小値: {fd.min():.4f}")
    

* * *

## 1.5 画像データの前処理

### Normalization（正規化）とStandardization（標準化）

機械学習では、画像データを適切にスケーリングすることが重要です。

手法 | 計算式 | 用途  
---|---|---  
**Min-Max正規化** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | 値域を[0, 1]に変換  
**標準化（Z-score）** | $x' = \frac{x - \mu}{\sigma}$ | 平均0、分散1に変換  
**ImageNet正規化** | チャンネルごとに標準化 | 事前学習モデル利用時  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # サンプル画像の作成
    np.random.seed(42)
    image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    print("=== 元画像の統計 ===")
    print(f"形状: {image.shape}")
    print(f"データ型: {image.dtype}")
    print(f"値域: [{image.min()}, {image.max()}]")
    print(f"平均値: {image.mean():.2f}")
    print(f"標準偏差: {image.std():.2f}")
    
    # Min-Max正規化 [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    print("\n=== Min-Max正規化後 ===")
    print(f"データ型: {normalized.dtype}")
    print(f"値域: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"平均値: {normalized.mean():.3f}")
    
    # 標準化（Z-score）
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    standardized = (image.astype(np.float32) - mean) / (std + 1e-7)
    
    print("\n=== 標準化（Z-score）後 ===")
    print(f"平均値: {standardized.mean():.6f} (≈ 0)")
    print(f"標準偏差: {standardized.std():.6f} (≈ 1)")
    print(f"値域: [{standardized.min():.3f}, {standardized.max():.3f}]")
    
    # ImageNet標準化（事前学習モデル用）
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_normalized = (normalized - imagenet_mean) / imagenet_std
    
    print("\n=== ImageNet標準化後 ===")
    print(f"Rチャンネル平均: {imagenet_normalized[:,:,0].mean():.3f}")
    print(f"Gチャンネル平均: {imagenet_normalized[:,:,1].mean():.3f}")
    print(f"Bチャンネル平均: {imagenet_normalized[:,:,2].mean():.3f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('元画像\n[50, 200]')
    axes[0].axis('off')
    
    axes[1].imshow(normalized)
    axes[1].set_title('Min-Max正規化\n[0, 1]')
    axes[1].axis('off')
    
    # 標準化画像は可視化のために調整
    standardized_vis = (standardized - standardized.min()) / (standardized.max() - standardized.min())
    axes[2].imshow(standardized_vis)
    axes[2].set_title('標準化\n平均≈0, 分散≈1')
    axes[2].axis('off')
    
    # ImageNet正規化も調整
    imagenet_vis = (imagenet_normalized - imagenet_normalized.min()) / \
                   (imagenet_normalized.max() - imagenet_normalized.min())
    axes[3].imshow(imagenet_vis)
    axes[3].set_title('ImageNet標準化')
    axes[3].axis('off')
    
    plt.tight_layout()
    print("\n正規化・標準化の効果を可視化しました")
    

### Data Augmentation（データ拡張）

**データ拡張** は、限られた訓練データから多様なバリエーションを生成し、モデルの汎化性能を向上させる手法です。
    
    
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image, ImageEnhance
    
    # サンプル画像の作成
    def create_sample_object(size=128):
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        # 矢印型のオブジェクト
        pts = np.array([[size//2, size//4], [3*size//4, size//2],
                        [5*size//8, size//2], [5*size//8, 3*size//4],
                        [3*size//8, 3*size//4], [3*size//8, size//2],
                        [size//4, size//2]], np.int32)
        cv2.fillPoly(image, [pts], (30, 144, 255))
        return image
    
    original = create_sample_object()
    
    # 各種データ拡張の適用
    # 1. 回転
    rotated = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    
    # 2. 左右反転
    flipped = cv2.flip(original, 1)
    
    # 3. ランダムクロップ&リサイズ
    h, w = original.shape[:2]
    crop_size = 96
    x, y = np.random.randint(0, w - crop_size), np.random.randint(0, h - crop_size)
    cropped = original[y:y+crop_size, x:x+crop_size]
    cropped_resized = cv2.resize(cropped, (128, 128))
    
    # 4. 明度変更
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    brightness = ImageEnhance.Brightness(pil_img).enhance(1.5)
    brightness = cv2.cvtColor(np.array(brightness), cv2.COLOR_RGB2BGR)
    
    # 5. コントラスト変更
    contrast = ImageEnhance.Contrast(pil_img).enhance(1.5)
    contrast = cv2.cvtColor(np.array(contrast), cv2.COLOR_RGB2BGR)
    
    # 6. ガウシアンノイズ
    noisy = original.copy().astype(np.float32)
    noise = np.random.normal(0, 10, original.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    
    # 7. ぼかし
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    
    # 8. 色調変更（HSV空間で操作）
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # 色相をシフト
    hue_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 可視化
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "元画像"),
        (cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), "回転（90°）"),
        (cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB), "左右反転"),
        (cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB), "ランダムクロップ"),
        (cv2.cvtColor(brightness, cv2.COLOR_BGR2RGB), "明度変更（1.5倍）"),
        (cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB), "コントラスト変更"),
        (cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB), "ガウシアンノイズ"),
        (cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), "ぼかし"),
        (cv2.cvtColor(hue_shifted, cv2.COLOR_BGR2RGB), "色調変更"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("データ拡張の各種手法を可視化しました")
    print("\nこれらの手法を組み合わせることで、1枚の画像から数百〜数千のバリエーションを生成できます")
    

### PyTorch Transformsによる前処理パイプライン
    
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    # サンプル画像の作成
    sample_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_pil = Image.fromarray(sample_np)
    
    print("=== PyTorch Transforms による前処理パイプライン ===\n")
    
    # 訓練用の変換パイプライン
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 検証・テスト用の変換パイプライン
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 変換の適用
    train_tensor = train_transform(sample_pil)
    val_tensor = val_transform(sample_pil)
    
    print("訓練用変換:")
    print(f"  入力: PIL Image {sample_pil.size}")
    print(f"  出力: Tensor {train_tensor.shape}")
    print(f"  データ型: {train_tensor.dtype}")
    print(f"  値域: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    print("\n検証用変換:")
    print(f"  入力: PIL Image {sample_pil.size}")
    print(f"  出力: Tensor {val_tensor.shape}")
    print(f"  値域: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
    
    # バッチ処理のシミュレーション
    batch_size = 4
    batch_tensors = [train_transform(sample_pil) for _ in range(batch_size)]
    batch = torch.stack(batch_tensors)
    
    print(f"\nバッチ処理:")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  バッチテンソルの形状: {batch.shape}")
    print(f"  → [Batch, Channels, Height, Width]")
    
    # 個別の変換例
    print("\n=== 個別の変換の詳細 ===")
    
    # ToTensorのみ
    to_tensor = transforms.ToTensor()
    tensor_only = to_tensor(sample_pil)
    print(f"\n1. ToTensor:")
    print(f"   PIL (H, W, C) → Tensor (C, H, W)")
    print(f"   値域: [0, 255] → [0.0, 1.0]")
    print(f"   形状変化: {sample_pil.size} → {tensor_only.shape}")
    
    # Normalizeの効果
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized = normalize(tensor_only)
    print(f"\n2. Normalize (mean=0.5, std=0.5):")
    print(f"   変換前の平均: {tensor_only.mean():.3f}")
    print(f"   変換後の平均: {normalized.mean():.3f}")
    print(f"   値域: [0.0, 1.0] → [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("\n前処理パイプラインの構築が完了しました")
    

* * *

## まとめ

この章では、画像処理の基礎について学習しました。

### 重要なポイント

  * **デジタル画像** はピクセルの配列として表現され、HWCまたはCHW形式で扱われる
  * **カラー空間** （RGB、HSV、LAB）は用途に応じて使い分ける
  * **基本処理** （リサイズ、回転、色変換）は画像解析の前処理として重要
  * **フィルタリング** （平滑化、エッジ検出）で画像の特徴を抽出
  * **特徴量** （コーナー、SIFT、HOG）は物体認識の基礎
  * **前処理とデータ拡張** は機械学習モデルの性能向上に不可欠

### 次章の予告

第2章では、以下のトピックを扱います：

  * 畳み込みニューラルネットワーク（CNN）の基礎
  * 畳み込み層とプーリング層の仕組み
  * 代表的なCNNアーキテクチャ（LeNet、AlexNet）
  * PyTorchによるCNNの実装と画像分類

* * *

## 演習問題

**演習1：カラー空間の理解**

**問題** ：RGB画像をHSV空間に変換し、特定の色（例：赤色）の領域を抽出してください。

**ヒント** ：

  * HSV空間では色相（Hue）で色を指定しやすい
  * 赤色の色相範囲: 0-10 および 170-180（OpenCV）

**解答例** ：
    
    
    import cv2
    import numpy as np
    
    # RGB画像の作成（赤、緑、青の領域）
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, 0:100] = [0, 0, 255]    # 赤（BGR）
    image[:, 100:200] = [0, 255, 0]  # 緑
    image[:, 200:300] = [255, 0, 0]  # 青
    
    # HSVに変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 赤色の範囲を定義
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # マスクの作成
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # マスクを適用
    red_only = cv2.bitwise_and(image, image, mask=red_mask)
    
    print(f"赤色領域のピクセル数: {np.sum(red_mask > 0)}")
    

**演習2：補間方法の比較**

**問題** ：100×100の画像を500×500に拡大する際、NEAREST、LINEAR、CUBIC、LANCZOSの4つの補間方法で処理時間と視覚的品質を比較してください。

**解答例** ：
    
    
    import cv2
    import numpy as np
    import time
    
    # テスト画像
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    methods = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'LANCZOS': cv2.INTER_LANCZOS4
    }
    
    print("補間方法の処理時間比較:")
    for name, method in methods.items():
        start = time.time()
        resized = cv2.resize(image, (500, 500), interpolation=method)
        elapsed = time.time() - start
        print(f"  {name:10s}: {elapsed*1000:.2f} ms")
    

**演習3：カスタムフィルタの実装**

**問題** ：以下のカーネルを使用して、手動で畳み込み演算を実装してください。
    
    
    シャープ化カーネル:
     0  -1   0
    -1   5  -1
     0  -1   0
    

**解答例** ：
    
    
    import numpy as np
    import cv2
    
    def custom_convolution(image, kernel):
        """カスタム畳み込み関数"""
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        pad_h, pad_w = ker_h // 2, ker_w // 2
    
        # パディング
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
        # 出力画像
        output = np.zeros_like(image)
    
        # 畳み込み
        for i in range(img_h):
            for j in range(img_w):
                region = padded[i:i+ker_h, j:j+ker_w]
                output[i, j] = np.sum(region * kernel)
    
        return np.clip(output, 0, 255).astype(np.uint8)
    
    # シャープ化カーネル
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    
    # テスト
    test_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    sharpened = custom_convolution(test_image, sharpen_kernel)
    
    print("カスタム畳み込みを実装しました")
    print(f"入力: {test_image.shape}")
    print(f"出力: {sharpened.shape}")
    

**演習4：PyTorch Transformsの応用**

**問題** ：以下の要件を満たすデータ拡張パイプラインを作成してください：

  * 80%の確率で左右反転
  * 明度とコントラストをランダムに±20%変更
  * ±10度のランダム回転
  * 最終的に224×224にリサイズ
  * ImageNet標準化を適用

**解答例** ：
    
    
    from torchvision import transforms
    
    augmentation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("データ拡張パイプラインを作成しました")
    print(f"変換数: {len(augmentation_pipeline.transforms)}")
    

**演習5：エッジ検出の応用**

**問題** ：Cannyエッジ検出を使用して、画像内の長方形領域を検出し、その面積を計算してください。

**解答例** ：
    
    
    import cv2
    import numpy as np
    
    # 長方形を含む画像を作成
    image = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 150), 255, -1)
    cv2.rectangle(image, (250, 100), (350, 250), 255, -1)
    
    # Cannyエッジ検出
    edges = cv2.Canny(image, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"検出された輪郭数: {len(contours)}")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(f"輪郭{i+1}: 面積={area:.0f}, 周囲長={perimeter:.2f}")
    

* * *

## 参考文献

  * [OpenCV Documentation](<https://docs.opencv.org/>) \- 公式ドキュメント
  * [Pillow (PIL) Documentation](<https://pillow.readthedocs.io/>) \- Python Imaging Library
  * [torchvision.transforms](<https://pytorch.org/vision/stable/transforms.html>) \- PyTorch画像変換
  * Szeliski, R. (2010). _Computer Vision: Algorithms and Applications_. Springer.
  * Gonzalez, R. C., & Woods, R. E. (2018). _Digital Image Processing_ (4th ed.). Pearson.
  * Bradski, G., & Kaehler, A. (2008). _Learning OpenCV_. O'Reilly Media.

* * *
