---
title: 第4章：PyTorchとTensorFlow実践
chapter_title: 第4章：PyTorchとTensorFlow実践
subtitle: 現代的なディープラーニングフレームワークの実装とベストプラクティス
reading_time: 25-30分
difficulty: 中級
code_examples: 12
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ PyTorchのテンソル操作と自動微分の仕組みを理解する
  * ✅ nn.Moduleを使った本格的なニューラルネットワーク構築
  * ✅ DataLoaderを活用した効率的なバッチ処理
  * ✅ TensorFlow/Kerasの実装パターンとPyTorchとの比較
  * ✅ モデルの保存・読み込み、GPU活用などの実用テクニック

* * *

## 4.1 PyTorch入門

### PyTorchとは

**PyTorch** は、Meta（旧Facebook）が開発したオープンソースの機械学習フレームワークです。研究者や実務家に広く使われており、以下の特徴があります：

  * **Define-by-Run** : 動的な計算グラフ（実行しながらグラフを構築）
  * **Pythonic** : NumPyライクな直感的なAPI
  * **自動微分** : Autogradによる自動的な勾配計算
  * **GPU対応** : CUDAによる高速化が容易

> 「PyTorchは研究から本番環境まで、シームレスに使えるフレームワークです。柔軟性と高速性を両立しています。」

### テンソル操作の基礎

PyTorchの基本データ構造は**テンソル（Tensor）** です。NumPyのndarrayに似ていますが、GPU上で計算できます。
    
    
    import torch
    import numpy as np
    
    # テンソルの作成
    print("=== テンソルの作成 ===")
    
    # リストから作成
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"x = {x}")
    print(f"x.shape = {x.shape}")  # torch.Size([3])
    
    # ゼロで初期化
    zeros = torch.zeros(2, 3)
    print(f"\nzeros:\n{zeros}")
    
    # 正規分布からランダムに初期化
    randn = torch.randn(2, 3)
    print(f"\nrandn:\n{randn}")
    
    # NumPy配列から変換
    np_array = np.array([[1, 2], [3, 4]])
    torch_tensor = torch.from_numpy(np_array)
    print(f"\nNumPyから変換:\n{torch_tensor}")
    
    # テンソルの演算
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"\n=== テンソル演算 ===")
    print(f"a + b:\n{a + b}")
    print(f"a * b（要素ごと）:\n{a * b}")
    print(f"a @ b（行列積）:\n{a @ b}")
    print(f"a.T（転置）:\n{a.T}")
    

**出力** ：
    
    
    === テンソルの作成 ===
    x = tensor([1., 2., 3.])
    x.shape = torch.Size([3])
    
    zeros:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    
    randn:
    tensor([[ 0.3367, -1.2312,  0.5414],
            [-0.8485,  1.1234, -0.3421]])
    
    NumPyから変換:
    tensor([[1, 2],
            [3, 4]])
    
    === テンソル演算 ===
    a + b:
    tensor([[ 6.,  8.],
            [10., 12.]])
    a * b（要素ごと）:
    tensor([[ 5., 12.],
            [21., 32.]])
    a @ b（行列積）:
    tensor([[19., 22.],
            [43., 50.]])
    a.T（転置）:
    tensor([[1., 3.],
            [2., 4.]])
    

### GPU対応コード
    
    
    import torch
    
    # GPUが利用可能かチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # テンソルをGPUに転送
    x = torch.randn(1000, 1000)
    x_gpu = x.to(device)  # GPUに転送
    print(f"x_gpu.device: {x_gpu.device}")
    
    # GPUで計算
    y_gpu = x_gpu @ x_gpu.T
    print(f"GPU計算結果のshape: {y_gpu.shape}")
    
    # CPUに戻す
    y_cpu = y_gpu.cpu()
    print(f"CPUに転送: {y_cpu.device}")
    
    # デバイスを明示的に指定して作成
    z = torch.randn(10, 10, device=device)
    print(f"デバイス指定で作成: {z.device}")
    

### Autograd（自動微分）

PyTorchの**Autograd** は、テンソル演算を追跡して自動的に勾配を計算します。
    
    
    import torch
    
    # 勾配を計算したいテンソルは requires_grad=True を設定
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x = {x}")
    
    # 順伝播: y = x^2 + 3x + 1
    y = x**2 + 3*x + 1
    print(f"y = {y}")
    
    # 逆伝播: dy/dx を計算
    y.backward()  # 自動微分実行
    
    # 勾配を取得
    print(f"dy/dx = {x.grad}")  # dy/dx = 2x + 3 = 2*2 + 3 = 7
    
    # より複雑な例
    print("\n=== 複雑な計算グラフ ===")
    x = torch.randn(3, requires_grad=True)
    print(f"x = {x}")
    
    # 複数の演算を組み合わせ
    y = x * 2
    z = y ** 2
    out = z.mean()  # スカラー値
    
    print(f"out = {out}")
    
    # 勾配計算
    out.backward()
    print(f"x.grad = {x.grad}")
    
    # 勾配の検証: d(mean((x*2)^2))/dx = d(mean(4x^2))/dx = 8x/3
    print(f"期待値（8x/3）: {8 * x.data / 3}")
    

**出力** ：
    
    
    x = tensor([2.], requires_grad=True)
    y = tensor([11.], grad_fn=<AddBackward0>)
    dy/dx = tensor([7.])
    
    === 複雑な計算グラフ ===
    x = tensor([-0.5234,  1.2156, -0.8945], requires_grad=True)
    out = tensor(2.0394, grad_fn=<MeanBackward0>)
    x.grad = tensor([-1.3957,  3.2416, -2.3853])
    期待値（8x/3）: tensor([-1.3957,  3.2416, -2.3853])
    
    
    
    ```mermaid
    graph TD
        x[x: requires_grad=True] --> mul[y = x * 2]
        mul --> pow[z = y ** 2]
        pow --> mean[out = z.mean]
        mean --> backward[backward]
        backward --> grad[x.grad に勾配が保存]
    
        style x fill:#e3f2fd
        style mul fill:#fff3e0
        style pow fill:#fff3e0
        style mean fill:#fff3e0
        style backward fill:#f3e5f5
        style grad fill:#e8f5e9
    ```

### nn.Moduleでモデル定義

**nn.Module** はPyTorchのニューラルネットワーク構築の基本クラスです。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleNet(nn.Module):
        """シンプルな3層ニューラルネットワーク"""
    
        def __init__(self, input_size, hidden_size, output_size):
            """
            Args:
                input_size: 入力層のサイズ
                hidden_size: 隠れ層のサイズ
                output_size: 出力層のサイズ
            """
            super(SimpleNet, self).__init__()
    
            # レイヤーの定義
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            """順伝播"""
            x = F.relu(self.fc1(x))  # 1層目 + ReLU
            x = F.relu(self.fc2(x))  # 2層目 + ReLU
            x = self.fc3(x)           # 出力層（活性化なし）
            return x
    
    # モデルのインスタンス化
    model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
    print(model)
    
    # パラメータの確認
    print(f"\n総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 推論の実行
    x = torch.randn(32, 784)  # バッチサイズ32
    output = model(x)
    print(f"\n入力shape: {x.shape}")
    print(f"出力shape: {output.shape}")  # (32, 10)
    

**出力** ：
    
    
    SimpleNet(
      (fc1): Linear(in_features=784, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=128, bias=True)
      (fc3): Linear(in_features=128, out_features=10, bias=True)
    )
    
    総パラメータ数: 118,026
    
    入力shape: torch.Size([32, 784])
    出力shape: torch.Size([32, 10])
    

### 学習ループの実装
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # ダミーデータの作成
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))  # 0-9のラベル
    
    # モデル、損失関数、オプティマイザの準備
    model = SimpleNet(784, 128, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学習ループ
    num_epochs = 10
    batch_size = 32
    
    print("=== 学習開始 ===")
    for epoch in range(num_epochs):
        # エポックごとの損失
        epoch_loss = 0.0
        num_batches = 0
    
        # ミニバッチに分割
        for i in range(0, len(X_train), batch_size):
            # バッチデータの取得
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
    
            # 勾配をゼロにリセット（重要！）
            optimizer.zero_grad()
    
            # 順伝播
            outputs = model(batch_X)
    
            # 損失計算
            loss = criterion(outputs, batch_y)
    
            # 逆伝播
            loss.backward()
    
            # パラメータ更新
            optimizer.step()
    
            epoch_loss += loss.item()
            num_batches += 1
    
        # エポックごとの平均損失
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("\n学習完了！")
    

**出力** ：
    
    
    === 学習開始 ===
    Epoch [1/10], Loss: 2.3124
    Epoch [2/10], Loss: 2.2845
    Epoch [3/10], Loss: 2.2567
    Epoch [4/10], Loss: 2.2301
    Epoch [5/10], Loss: 2.2048
    Epoch [6/10], Loss: 2.1807
    Epoch [7/10], Loss: 2.1577
    Epoch [8/10], Loss: 2.1357
    Epoch [9/10], Loss: 2.1146
    Epoch [10/10], Loss: 2.0944
    
    学習完了！
    

* * *

## 4.2 実践的なモデル構築

### カスタムレイヤーの作成

独自の処理を行うレイヤーをnn.Moduleを継承して作成できます。
    
    
    import torch
    import torch.nn as nn
    import math
    
    class GaussianNoise(nn.Module):
        """学習時にガウシアンノイズを追加するレイヤー"""
    
        def __init__(self, sigma=0.1):
            """
            Args:
                sigma: ノイズの標準偏差
            """
            super(GaussianNoise, self).__init__()
            self.sigma = sigma
    
        def forward(self, x):
            """
            学習時のみノイズを追加
            """
            if self.training:  # 学習モードの場合のみ
                noise = torch.randn_like(x) * self.sigma
                return x + noise
            return x
    
    class ResidualBlock(nn.Module):
        """残差接続（Residual Connection）を持つブロック"""
    
        def __init__(self, size):
            super(ResidualBlock, self).__init__()
            self.fc1 = nn.Linear(size, size)
            self.fc2 = nn.Linear(size, size)
            self.bn1 = nn.BatchNorm1d(size)
            self.bn2 = nn.BatchNorm1d(size)
    
        def forward(self, x):
            """
            y = F(x) + x  （残差接続）
            """
            residual = x  # ショートカット接続
    
            out = F.relu(self.bn1(self.fc1(x)))
            out = self.bn2(self.fc2(out))
    
            out = out + residual  # 残差を追加
            out = F.relu(out)
    
            return out
    
    # カスタムレイヤーを組み込んだモデル
    class CustomNet(nn.Module):
        def __init__(self):
            super(CustomNet, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.noise = GaussianNoise(sigma=0.05)
            self.res1 = ResidualBlock(256)
            self.res2 = ResidualBlock(256)
            self.fc_out = nn.Linear(256, 10)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.noise(x)  # ノイズ追加（正則化）
            x = self.res1(x)   # 残差ブロック1
            x = self.res2(x)   # 残差ブロック2
            x = self.fc_out(x)
            return x
    
    # テスト
    model = CustomNet()
    x = torch.randn(16, 784)
    
    # 学習モード
    model.train()
    out_train = model(x)
    print(f"学習モード出力: {out_train.shape}")
    
    # 評価モード
    model.eval()
    out_eval = model(x)
    print(f"評価モード出力: {out_eval.shape}")
    
    
    
    ```mermaid
    graph LR
        input[入力 x] --> fc1[全結合層]
        fc1 --> noise[ガウシアンノイズ]
        noise --> res1[残差ブロック1]
        res1 --> res2[残差ブロック2]
        res2 --> fc_out[出力層]
        fc_out --> output[出力]
    
        input -.ショートカット.-> res1
        res1 -.ショートカット.-> res2
    
        style input fill:#e3f2fd
        style noise fill:#fff9c4
        style res1 fill:#f3e5f5
        style res2 fill:#f3e5f5
        style output fill:#e8f5e9
    ```

### 損失関数とオプティマイザ

PyTorchは多様な損失関数とオプティマイザを提供しています。
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # モデルの準備
    model = SimpleNet(784, 128, 10)
    
    print("=== 損失関数の選択 ===")
    
    # 分類問題の損失関数
    criterion_ce = nn.CrossEntropyLoss()  # 多クラス分類
    criterion_bce = nn.BCEWithLogitsLoss()  # 二値分類
    criterion_nll = nn.NLLLoss()  # 負の対数尤度
    
    # 回帰問題の損失関数
    criterion_mse = nn.MSELoss()  # 平均二乗誤差
    criterion_mae = nn.L1Loss()   # 平均絶対誤差
    criterion_huber = nn.SmoothL1Loss()  # Huber損失
    
    print("分類: CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss")
    print("回帰: MSELoss, L1Loss, SmoothL1Loss")
    
    print("\n=== オプティマイザの選択 ===")
    
    # SGD（確率的勾配降下法）
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Adam（適応的学習率）
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    # AdamW（Weight Decayを改善したAdam）
    optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # RMSprop
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    
    print("SGD: 古典的、モメンタムで安定化")
    print("Adam: 適応的学習率、デフォルトで良好")
    print("AdamW: Adamの改良版、正則化が改善")
    print("RMSprop: RNNで効果的")
    
    # オプティマイザの使用例
    print("\n=== 学習ステップの例 ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # ダミーデータ
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    # 1ステップの学習
    optimizer.zero_grad()      # 勾配リセット
    output = model(x)          # 順伝播
    loss = criterion(output, y)  # 損失計算
    loss.backward()            # 逆伝播
    optimizer.step()           # パラメータ更新
    
    print(f"損失: {loss.item():.4f}")
    

### DataLoaderとバッチ処理

**DataLoader** は、データの自動バッチ化、シャッフル、並列読み込みを提供します。
    
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    
    # カスタムデータセットクラス
    class CustomDataset(Dataset):
        """カスタムデータセットの実装例"""
    
        def __init__(self, data, labels, transform=None):
            """
            Args:
                data: 入力データ（NumPy配列またはテンソル）
                labels: ラベルデータ
                transform: データ変換関数（オプション）
            """
            self.data = data
            self.labels = labels
            self.transform = transform
    
        def __len__(self):
            """データセットのサイズを返す"""
            return len(self.data)
    
        def __getitem__(self, idx):
            """インデックスに対応するデータを返す"""
            x = self.data[idx]
            y = self.labels[idx]
    
            # 変換を適用
            if self.transform:
                x = self.transform(x)
    
            return x, y
    
    # ダミーデータの作成
    np.random.seed(42)
    data = np.random.randn(1000, 784).astype(np.float32)
    labels = np.random.randint(0, 10, size=1000)
    
    # データセットの作成
    dataset = CustomDataset(data, labels)
    
    # DataLoaderの作成
    train_loader = DataLoader(
        dataset,
        batch_size=32,      # バッチサイズ
        shuffle=True,       # エポックごとにシャッフル
        num_workers=0,      # データ読み込みの並列数（0=メインプロセスのみ）
        drop_last=True      # 最後の不完全なバッチを破棄
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    print(f"バッチ数: {len(train_loader)}")
    
    # DataLoaderの使用例
    print("\n=== DataLoaderによるバッチ処理 ===")
    for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):
        if batch_idx >= 3:  # 最初の3バッチのみ表示
            break
        print(f"バッチ {batch_idx+1}: data shape={data_batch.shape}, labels shape={labels_batch.shape}")
    
    # 学習ループでの使用例
    print("\n=== 学習ループ例 ===")
    model = SimpleNet(784, 128, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        model.train()
        total_loss = 0
    
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: 平均損失 = {avg_loss:.4f}")
    

### 学習の可視化
    
    
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    class TrainingMonitor:
        """学習過程を可視化するクラス"""
    
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
    
        def update(self, train_loss, val_loss, train_acc, val_acc):
            """メトリクスを追加"""
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
    
        def plot(self):
            """リアルタイムプロット"""
            clear_output(wait=True)
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
            # 損失のプロット
            ax1.plot(self.train_losses, label='Train Loss', marker='o')
            ax1.plot(self.val_losses, label='Val Loss', marker='s')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
            # 精度のプロット
            ax2.plot(self.train_accs, label='Train Acc', marker='o')
            ax2.plot(self.val_accs, label='Val Acc', marker='s')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # 使用例
    monitor = TrainingMonitor()
    
    # ダミーデータで学習シミュレーション
    for epoch in range(10):
        # 損失は徐々に減少
        train_loss = 2.0 - epoch * 0.15 + np.random.randn() * 0.05
        val_loss = 2.1 - epoch * 0.13 + np.random.randn() * 0.05
    
        # 精度は徐々に向上
        train_acc = 20 + epoch * 7 + np.random.randn() * 2
        val_acc = 18 + epoch * 6.5 + np.random.randn() * 2
    
        monitor.update(train_loss, val_loss, train_acc, val_acc)
        monitor.plot()
    
    print("学習曲線の可視化完了")
    

* * *

## 4.3 TensorFlow/Keras入門

### TensorFlowとKerasの概要

**TensorFlow** はGoogleが開発したディープラーニングフレームワークで、**Keras** は高レベルAPIとして統合されています。

  * **Define-and-Run** : 静的な計算グラフ（高速化に有利）
  * **Eager Execution** : TensorFlow 2.x以降は動的実行もサポート
  * **Production Ready** : TensorFlow Serving、TF Lite、TF.jsで本番展開が容易
  * **Keras API** : シンプルで直感的なAPI

### Sequential APIとFunctional API
    
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # ===== Sequential API =====
    print("\n=== Sequential API ===")
    # 層を順番に積み重ねるシンプルなモデル
    
    model_seq = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model_seq.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model_seq.summary())
    
    # ===== Functional API =====
    print("\n=== Functional API ===")
    # より複雑な構造（分岐、マルチ入出力）が可能
    
    # 入力の定義
    inputs = keras.Input(shape=(784,))
    
    # 隠れ層の定義
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 出力層
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # モデルの構築
    model_func = keras.Model(inputs=inputs, outputs=outputs)
    
    model_func.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model_func.summary())
    
    # ダミーデータで学習
    X_train = np.random.randn(1000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=1000)
    
    print("\n=== 学習実行 ===")
    history = model_seq.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    

### カスタムモデルの作成
    
    
    import tensorflow as tf
    from tensorflow import keras
    
    class CustomModel(keras.Model):
        """カスタムモデルクラス（PyTorchのnn.Moduleに相当）"""
    
        def __init__(self, num_classes=10):
            super(CustomModel, self).__init__()
            self.dense1 = layers.Dense(128, activation='relu')
            self.dropout1 = layers.Dropout(0.2)
            self.dense2 = layers.Dense(128, activation='relu')
            self.dropout2 = layers.Dropout(0.2)
            self.dense3 = layers.Dense(num_classes, activation='softmax')
    
        def call(self, inputs, training=False):
            """順伝播（PyTorchのforwardに相当）"""
            x = self.dense1(inputs)
            x = self.dropout1(x, training=training)  # 学習時のみDropout
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            return self.dense3(x)
    
    # カスタムモデルのインスタンス化
    custom_model = CustomModel(num_classes=10)
    
    # コンパイル
    custom_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # ダミーデータで学習
    X_train = np.random.randn(1000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=1000)
    
    history = custom_model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    
    print("\nカスタムモデル学習完了")
    

### コールバックの活用

**コールバック** は、学習中の特定のタイミングで実行される処理です。
    
    
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard
    )
    import datetime
    
    # Early Stopping: 検証損失が改善しなくなったら学習を停止
    early_stop = EarlyStopping(
        monitor='val_loss',     # 監視する指標
        patience=5,              # 改善しないエポック数
        restore_best_weights=True  # 最良の重みを復元
    )
    
    # Model Checkpoint: ベストモデルを保存
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Learning Rate Scheduler: 学習率を動的に調整
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,              # 学習率を半分に
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard: 学習過程の可視化
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # コールバックを使った学習
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # ダミーデータ
    X_train = np.random.randn(1000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=1000)
    X_val = np.random.randn(200, 784).astype(np.float32)
    y_val = np.random.randint(0, 10, size=200)
    
    # コールバックを指定して学習
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )
    
    print("\nコールバックを活用した学習完了")
    print(f"学習終了エポック: {len(history.history['loss'])}")
    

### PyTorchとの比較

機能 | PyTorch | TensorFlow/Keras  
---|---|---  
**計算グラフ** | 動的（Define-by-Run） | 静的+動的（Eager Execution）  
**モデル定義** | nn.Module継承 | Sequential/Functional/Model  
**学習ループ** | 手動で記述 | model.fit()で自動化  
**データ読み込み** | DataLoader | tf.data.Dataset  
**オプティマイザ** | torch.optim | keras.optimizers  
**デバイス管理** | .to(device)明示的 | 自動（デフォルトでGPU使用）  
**コミュニティ** | 研究者に人気 | 産業界に人気  
**本番展開** | TorchServe | TF Serving（成熟）  
  
> **選択のポイント** :  
>  \- **PyTorch** : 研究、柔軟性重視、細かい制御が必要  
>  \- **TensorFlow/Keras** : 本番展開、高速化重視、シンプルなAPI
    
    
    ```mermaid
    graph TB
        subgraph PyTorch
            A1[nn.Module] --> A2[forward]
            A2 --> A3[手動学習ループ]
            A3 --> A4[optimizer.step]
        end
    
        subgraph TensorFlow
            B1[Sequential/Model] --> B2[call/forward]
            B2 --> B3[model.fit]
            B3 --> B4[自動学習]
        end
    
        style A1 fill:#ffe0b2
        style A2 fill:#fff9c4
        style A3 fill:#f0f4c3
        style A4 fill:#c8e6c9
    
        style B1 fill:#e1bee7
        style B2 fill:#f3e5f5
        style B3 fill:#e8eaf6
        style B4 fill:#c5cae9
    ```

* * *

## 4.4 実用的なテクニック

### GPU使用とメモリ管理
    
    
    import torch
    import torch.nn as nn
    
    # === GPU使用の基本 ===
    print("=== GPU情報 ===")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数: {torch.cuda.device_count()}")
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"メモリ割り当て: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"メモリキャッシュ: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # デバイスの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用デバイス: {device}")
    
    # モデルとデータをGPUに転送
    model = SimpleNet(784, 128, 10).to(device)
    x = torch.randn(32, 784, device=device)  # 直接GPUに作成
    
    # 推論
    with torch.no_grad():  # 勾配計算を無効化（メモリ節約）
        output = model(x)
    print(f"出力デバイス: {output.device}")
    
    # === メモリ管理のベストプラクティス ===
    print("\n=== メモリ管理 ===")
    
    # 1. 不要なテンソルは削除
    x = torch.randn(1000, 1000, device=device)
    del x
    torch.cuda.empty_cache()  # キャッシュをクリア
    
    # 2. Gradient Checkpointing（メモリと計算のトレードオフ）
    from torch.utils.checkpoint import checkpoint
    
    class MemoryEfficientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(784, 1024)
            self.layer2 = nn.Linear(1024, 1024)
            self.layer3 = nn.Linear(1024, 10)
    
        def forward(self, x):
            # checkpointを使うと、メモリ使用量が減る（計算は遅くなる）
            x = checkpoint(lambda x: F.relu(self.layer1(x)), x)
            x = checkpoint(lambda x: F.relu(self.layer2(x)), x)
            x = self.layer3(x)
            return x
    
    # 3. Mixed Precision Training（メモリ削減＋高速化）
    from torch.cuda.amp import autocast, GradScaler
    
    model = SimpleNet(784, 128, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()  # 自動スケーリング
    
    for epoch in range(3):
        for data, target in [(torch.randn(32, 784, device=device),
                              torch.randint(0, 10, (32,), device=device))]:
            optimizer.zero_grad()
    
            # Mixed Precision
            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
    
            # スケーリング付き逆伝播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        print(f"Epoch {epoch+1} 完了")
    
    print("\nGPU最適化学習完了")
    

### モデルの保存と読み込み
    
    
    import torch
    import torch.nn as nn
    
    # モデルの準備
    model = SimpleNet(784, 128, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # ===== PyTorchでの保存・読み込み =====
    print("=== PyTorch: モデル保存 ===")
    
    # 方法1: state_dictのみ保存（推奨）
    torch.save(model.state_dict(), 'model_weights.pth')
    print("重みを保存: model_weights.pth")
    
    # 方法2: モデル全体を保存
    torch.save(model, 'model_full.pth')
    print("モデル全体を保存: model_full.pth")
    
    # 方法3: チェックポイント（学習途中の状態）
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5,
    }
    torch.save(checkpoint, 'checkpoint.pth')
    print("チェックポイント保存: checkpoint.pth")
    
    print("\n=== PyTorch: モデル読み込み ===")
    
    # 方法1: state_dictの読み込み
    model_new = SimpleNet(784, 128, 10)
    model_new.load_state_dict(torch.load('model_weights.pth'))
    model_new.eval()  # 評価モードに設定
    print("重みを読み込み完了")
    
    # 方法2: モデル全体の読み込み
    model_loaded = torch.load('model_full.pth')
    model_loaded.eval()
    print("モデル全体読み込み完了")
    
    # 方法3: チェックポイントから再開
    checkpoint = torch.load('checkpoint.pth')
    model_new.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"チェックポイント読み込み: Epoch={epoch}, Loss={loss}")
    
    # ===== TensorFlow/Kerasでの保存・読み込み =====
    print("\n=== TensorFlow/Keras: モデル保存 ===")
    
    # Kerasモデルの準備
    keras_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # 方法1: SavedFormat（推奨）
    keras_model.save('saved_model/')
    print("SavedFormat保存: saved_model/")
    
    # 方法2: HDF5形式
    keras_model.save('model.h5')
    print("HDF5保存: model.h5")
    
    # 方法3: 重みのみ
    keras_model.save_weights('weights.h5')
    print("重みのみ保存: weights.h5")
    
    print("\n=== TensorFlow/Keras: モデル読み込み ===")
    
    # SavedFormatから読み込み
    loaded_model = keras.models.load_model('saved_model/')
    print("SavedFormat読み込み完了")
    
    # HDF5から読み込み
    loaded_model_h5 = keras.models.load_model('model.h5')
    print("HDF5読み込み完了")
    
    # 重みのみ読み込み
    new_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    new_model.load_weights('weights.h5')
    print("重みのみ読み込み完了")
    

### Early StoppingとCheckpointing
    
    
    import torch
    import numpy as np
    
    class EarlyStopping:
        """Early Stoppingを実装するクラス（PyTorch用）"""
    
        def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
            """
            Args:
                patience: 改善しないエポック数
                min_delta: 改善と判定する最小変化量
                restore_best_weights: 最良の重みを復元するか
            """
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
    
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.best_weights = None
    
        def __call__(self, val_loss, model):
            """
            Args:
                val_loss: 検証損失
                model: モデル
    
            Returns:
                早期停止すべきかどうか
            """
            if self.best_loss is None:
                self.best_loss = val_loss
                self.save_checkpoint(model)
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights:
                        print('最良の重みを復元します')
                        model.load_state_dict(self.best_weights)
            else:
                self.best_loss = val_loss
                self.save_checkpoint(model)
                self.counter = 0
    
            return self.early_stop
    
        def save_checkpoint(self, model):
            """ベストモデルを保存"""
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
    
    # 使用例
    print("=== Early Stopping 実装例 ===")
    model = SimpleNet(784, 128, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    
    # ダミーデータ
    X_train = torch.randn(800, 784)
    y_train = torch.randint(0, 10, (800,))
    X_val = torch.randn(200, 784)
    y_val = torch.randint(0, 10, (200,))
    
    for epoch in range(50):
        # 学習
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
        # 検証
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
    
        print(f'Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}')
    
        # Early Stopping判定
        if early_stopping(val_loss.item(), model):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print("\n学習完了")
    

### 学習率スケジューリング
    
    
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        StepLR,
        ExponentialLR,
        CosineAnnealingLR,
        ReduceLROnPlateau
    )
    import matplotlib.pyplot as plt
    
    # モデルとオプティマイザ
    model = SimpleNet(784, 128, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    print("=== 学習率スケジューラの種類 ===\n")
    
    # 1. StepLR: 一定エポックごとに学習率を減衰
    scheduler1 = StepLR(optimizer, step_size=10, gamma=0.5)
    print("StepLR: 10エポックごとに学習率を半減")
    
    # 2. ExponentialLR: 指数関数的に減衰
    scheduler2 = ExponentialLR(optimizer, gamma=0.95)
    print("ExponentialLR: 毎エポック5%減衰")
    
    # 3. CosineAnnealingLR: コサイン関数で減衰
    scheduler3 = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    print("CosineAnnealingLR: コサイン関数で滑らかに減衰")
    
    # 4. ReduceLROnPlateau: 損失が改善しない場合に減衰
    scheduler4 = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print("ReduceLROnPlateau: 損失が改善しない場合に減衰")
    
    # 学習率の推移を可視化
    print("\n=== 学習率の推移 ===")
    
    schedulers = {
        'StepLR': StepLR(optim.Adam(model.parameters(), lr=0.1), step_size=10, gamma=0.5),
        'ExponentialLR': ExponentialLR(optim.Adam(model.parameters(), lr=0.1), gamma=0.95),
        'CosineAnnealingLR': CosineAnnealingLR(optim.Adam(model.parameters(), lr=0.1), T_max=50)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        optimizer = scheduler.optimizer
    
        for epoch in range(50):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
    
        ax.plot(lrs, label=name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('学習率スケジューラの比較', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    
    # 実際の使用例
    print("\n=== スケジューラ使用例 ===")
    model = SimpleNet(784, 128, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    for epoch in range(10):
        # 学習処理（省略）
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")
    
        # エポック終了時にスケジューラを更新
        scheduler.step()
    
    print("\n学習率スケジューリング完了")
    
    
    
    ```mermaid
    graph TB
        A[学習開始] --> B{検証損失改善?}
        B -->|Yes| C[学習継続]
        B -->|No| D[カウンター+1]
        D --> E{patience超過?}
        E -->|No| C
        E -->|Yes| F[Early Stopping]
        F --> G[ベストモデル復元]
    
        C --> H{最大エポック?}
        H -->|No| I[学習率更新]
        I --> B
        H -->|Yes| J[学習完了]
    
        style A fill:#e3f2fd
        style F fill:#ffebee
        style G fill:#fff9c4
        style J fill:#e8f5e9
    ```

* * *

## 4.5 本章のまとめ

### 学んだこと

  1. **PyTorch基礎**

     * テンソル操作とGPU活用
     * Autogradによる自動微分
     * nn.Moduleでのモデル定義
     * 学習ループの実装パターン
  2. **実践的なモデル構築**

     * カスタムレイヤー（残差接続、ノイズ層）
     * 損失関数とオプティマイザの選択
     * DataLoaderによる効率的なバッチ処理
     * 学習過程の可視化
  3. **TensorFlow/Keras**

     * Sequential/Functional/Custom Modelの3つのAPI
     * コールバックによる学習制御
     * PyTorchとの違いと使い分け
  4. **実用テクニック**

     * GPUメモリ管理とMixed Precision
     * モデル保存・読み込みのベストプラクティス
     * Early StoppingとCheckpointing
     * 学習率スケジューリング

### フレームワーク選択ガイド

用途 | 推奨フレームワーク | 理由  
---|---|---  
**研究・論文実装** | PyTorch | 柔軟性、デバッグしやすさ  
**プロトタイピング** | Keras | シンプル、高速開発  
**本番展開** | TensorFlow | TF Serving、成熟したエコシステム  
**カスタム実装** | PyTorch | 細かい制御が可能  
**モバイル/組み込み** | TensorFlow | TF Lite、最適化ツール  
  
### ベストプラクティス

  1. **デバイス管理** : GPUの有無に関わらず動作するコードを書く
  2. **再現性** : 乱数シードを固定（`torch.manual_seed(42)`）
  3. **メモリ効率** : `torch.no_grad()`で推論時のメモリを節約
  4. **学習監視** : TensorBoardやWandBで可視化
  5. **チェックポイント** : 定期的に保存して学習の中断に備える

### 次の章へ

第5章では、**畳み込みニューラルネットワーク（CNN）** を学びます：

  * 画像認識のための畳み込み層
  * プーリング層と特徴抽出
  * 代表的なCNNアーキテクチャ（LeNet、VGG、ResNet）
  * 実データ（CIFAR-10、ImageNet）での実装

* * *

## 演習問題

### 問題1（難易度：easy）

以下の文章の正誤を判定してください。

  1. PyTorchはDefine-by-Run方式を採用している
  2. TensorFlow 2.xではEager Executionがデフォルトで有効
  3. nn.Moduleのforwardメソッドは手動で呼び出す必要がある
  4. DataLoaderのshuffle=Trueは毎エポックでデータ順をシャッフルする

解答例

**解答** ：

  1. **正** \- PyTorchは動的な計算グラフを構築
  2. **正** \- TF 2.xではデフォルトで動的実行
  3. **誤** \- `model(x)`で自動的に呼ばれる
  4. **正** \- エポックごとにシャッフルされる

### 問題2（難易度：medium）

以下のPyTorchコードのバグを修正してください。
    
    
    import torch
    import torch.nn as nn
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
    
        output = model(x)
        loss = nn.MSELoss(output, y)
        loss.backward()
        optimizer.step()
    

ヒント

  * 損失関数の使い方を確認
  * 勾配のリセットが必要

解答例
    
    
    import torch
    import torch.nn as nn
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # 損失関数をインスタンス化
    
    for epoch in range(10):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
    
        optimizer.zero_grad()  # 勾配をゼロにリセット（追加）
        output = model(x)
        loss = criterion(output, y)  # 修正
        loss.backward()
        optimizer.step()
    

**修正点** ：

  1. `nn.MSELoss()`でインスタンス化
  2. `optimizer.zero_grad()`で勾配リセット

### 問題3（難易度：medium）

以下の要件を満たすカスタムデータセットクラスを実装してください：

  * NumPy配列のデータとラベルを受け取る
  * データを正規化（平均0、標準偏差1）
  * DataLoaderで使用可能

解答例
    
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    
    class NormalizedDataset(Dataset):
        """正規化を適用するカスタムデータセット"""
    
        def __init__(self, data, labels):
            """
            Args:
                data: NumPy配列 (N, features)
                labels: NumPy配列 (N,)
            """
            # データを正規化
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8  # ゼロ除算回避
            self.data = (data - self.mean) / self.std
            self.labels = labels
    
            # PyTorchテンソルに変換
            self.data = torch.FloatTensor(self.data)
            self.labels = torch.LongTensor(self.labels)
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # テスト
    np.random.seed(42)
    data = np.random.randn(100, 10) * 5 + 10  # 平均10、標準偏差5
    labels = np.random.randint(0, 3, size=100)
    
    dataset = NormalizedDataset(data, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 確認
    sample_data, sample_label = dataset[0]
    print(f"正規化後のデータ（最初のサンプル）: {sample_data}")
    print(f"データの平均（正規化後）: {dataset.data.mean():.4f}")
    print(f"データの標準偏差（正規化後）: {dataset.data.std():.4f}")
    

### 問題4（難易度：hard）

PyTorchとTensorFlow/Kerasの両方で、同じアーキテクチャのモデルを実装してください：

  * 入力: 28x28の画像（784次元）
  * 隠れ層1: 256ユニット、ReLU、Dropout(0.3)
  * 隠れ層2: 128ユニット、ReLU、Dropout(0.3)
  * 出力層: 10クラス分類

解答例

**PyTorch実装** ：
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class PyTorchModel(nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, 10)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    
    # 使用例
    pytorch_model = PyTorchModel()
    x = torch.randn(32, 784)
    output = pytorch_model(x)
    print(f"PyTorch output shape: {output.shape}")
    

**TensorFlow/Keras実装** ：
    
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Sequential API
    keras_model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10)
    ])
    
    # 使用例
    x = tf.random.normal((32, 784))
    output = keras_model(x)
    print(f"Keras output shape: {output.shape}")
    

**パラメータ数の確認** ：
    
    
    # PyTorch
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"PyTorch parameters: {pytorch_params:,}")
    
    # Keras
    keras_model.summary()
    

### 問題5（難易度：hard）

Early StoppingとModel Checkpointingを組み合わせた学習コードを実装してください。以下の条件を満たすこと：

  * 検証損失が5エポック改善しない場合に停止
  * ベストモデルを自動保存
  * 学習終了時にベストモデルを復元

解答例
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    class EarlyStoppingWithCheckpoint:
        """Early StoppingとCheckpointingを統合したクラス"""
    
        def __init__(self, patience=5, checkpoint_path='best_model.pth', verbose=True):
            self.patience = patience
            self.checkpoint_path = checkpoint_path
            self.verbose = verbose
    
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
    
        def __call__(self, val_loss, model):
            if self.best_loss is None or val_loss < self.best_loss:
                # 改善した場合
                if self.verbose:
                    if self.best_loss is not None:
                        improvement = self.best_loss - val_loss
                        print(f'検証損失が改善: {self.best_loss:.4f} → {val_loss:.4f} ({improvement:.4f})')
                    else:
                        print(f'初期ベストモデル: {val_loss:.4f}')
    
                self.best_loss = val_loss
                self.counter = 0
    
                # モデルを保存
                torch.save(model.state_dict(), self.checkpoint_path)
                if self.verbose:
                    print(f'モデルを保存: {self.checkpoint_path}')
            else:
                # 改善しなかった場合
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter}/{self.patience}')
    
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print('Early Stopping発動！')
    
            return self.early_stop
    
        def load_best_model(self, model):
            """ベストモデルを読み込む"""
            model.load_state_dict(torch.load(self.checkpoint_path))
            if self.verbose:
                print(f'ベストモデルを復元: {self.checkpoint_path} (loss={self.best_loss:.4f})')
    
    # 使用例
    print("=== Early Stopping + Checkpoint 実装例 ===\n")
    
    # ダミーデータ
    X_train = torch.randn(800, 784)
    y_train = torch.randint(0, 10, (800,))
    X_val = torch.randn(200, 784)
    y_val = torch.randint(0, 10, (200,))
    
    # モデル、損失関数、オプティマイザ
    model = SimpleNet(784, 128, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early Stoppingの初期化
    early_stopping = EarlyStoppingWithCheckpoint(
        patience=5,
        checkpoint_path='best_model.pth',
        verbose=True
    )
    
    # 学習ループ
    for epoch in range(50):
        # 学習
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()
    
        # 検証
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
    
        print(f'\nEpoch {epoch+1}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss.item():.4f}')
    
        # Early Stopping判定
        if early_stopping(val_loss.item(), model):
            print(f'\n学習を{epoch+1}エポックで終了')
            break
    
    # ベストモデルを復元
    print("\n" + "="*50)
    early_stopping.load_best_model(model)
    print("="*50)
    

* * *

## 参考文献

  1. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." _NeurIPS_.
  2. Abadi, M., et al. (2016). "TensorFlow: A System for Large-Scale Machine Learning." _OSDI_.
  3. Chollet, F. (2017). _Deep Learning with Python_. Manning Publications.
  4. Stevens, E., Antiga, L., & Viehmann, T. (2020). _Deep Learning with PyTorch_. Manning Publications.
