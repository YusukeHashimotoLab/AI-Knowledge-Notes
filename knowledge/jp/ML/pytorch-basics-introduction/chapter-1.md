---
title: 第1章：PyTorchの基礎
chapter_title: 第1章：PyTorchの基礎
subtitle: ディープラーニングフレームワークとしてのPyTorch
reading_time: 25-30分
difficulty: 初級
code_examples: 10
exercises: 5
---

この章では、PyTorchとは何か、なぜディープラーニングで広く使われているのかを理解します。TensorFlowとの比較、インストール方法、Tensorの基本概念、そして自動微分の仕組みを学びます。実践的なコード例を通じて、PyTorchでのプログラミングスタイルに慣れていきましょう。

## 学習目標

  * ✅ PyTorchの特徴とTensorFlowとの違いを説明できる
  * ✅ PyTorchをインストールし、動作確認ができる
  * ✅ Tensorの基本的な作成と操作ができる
  * ✅ 動的計算グラフの概念を理解する
  * ✅ 自動微分（autograd）の基本的な使い方を実践できる

## 1\. PyTorchとは何か

**PyTorch** は、Facebook（現Meta）のAI Research Lab（FAIR）が開発したオープンソースのディープラーニングフレームワークです。2016年にリリースされて以来、研究者や実務家に広く使われています。

### PyTorchの主な特徴

  * **動的計算グラフ（Dynamic Computation Graph）** : 実行時に計算グラフが構築されるため、Pythonライクな柔軟なコーディングが可能
  * **Pythonic API** : NumPyに似た直感的なAPI設計
  * **強力な自動微分** : autograd機能による自動的な勾配計算
  * **GPU対応** : CUDAを使った高速な並列計算
  * **豊富なエコシステム** : torchvision（画像）、torchtext（テキスト）、torchaudio（音声）など

### PyTorch vs TensorFlow

特徴 | PyTorch | TensorFlow  
---|---|---  
計算グラフ | 動的（Define-by-Run） | 静的（Define-and-Run）※TF2.0以降は動的も可  
学習曲線 | Pythonライクで直感的 | やや複雑（特にTF1.x）  
デバッグ | Python標準デバッガで可能 | やや困難（TF1.x）、TF2.0で改善  
研究での採用 | 非常に高い（論文実装の主流） | 中程度  
プロダクション | TorchServeでデプロイ可能 | TensorFlow Servingで強力  
モバイル対応 | PyTorch Mobile | TensorFlow Lite（成熟）  
  
**💡 どちらを選ぶべき？**

研究やプロトタイピングではPyTorchが人気です。一方、大規模なプロダクション環境や既存のTensorFlowインフラがある場合はTensorFlowも有力な選択肢です。両方を学ぶことで、状況に応じた最適なツールを使えるようになります。

## 2\. インストールと環境セットアップ

### 方法1: pip経由のインストール（推奨）
    
    
    # CPU版
    pip install torch torchvision torchaudio
    
    # GPU版（CUDA 11.8の場合）
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # GPU版（CUDA 12.1の場合）
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    

### 方法2: conda経由のインストール
    
    
    # CPU版
    conda install pytorch torchvision torchaudio -c pytorch
    
    # GPU版（CUDA 11.8の場合）
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    

### 方法3: Google Colab（環境構築不要）

Google Colabを使えば、ブラウザだけでPyTorchを使えます。無料でGPUも利用可能です。
    
    
    import torch
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    

### インストール確認

以下のコードでPyTorchが正しくインストールされているか確認しましょう：
    
    
    import torch
    
    # PyTorchのバージョン
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA（GPU）が利用可能か
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # CUDA version
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 簡単なTensor作成
    x = torch.tensor([1, 2, 3])
    print(f"Sample tensor: {x}")
    

**出力例：**
    
    
    PyTorch version: 2.1.0
    CUDA available: True
    CUDA version: 11.8
    GPU device: Tesla T4
    Sample tensor: tensor([1, 2, 3])
    

## 3\. Tensorの基礎

**Tensor** は、PyTorchの中心的なデータ構造です。NumPyのndarrayに似ていますが、GPU上での計算や自動微分に対応しています。

### Tensorの作成
    
    
    import torch
    
    # リストからTensorを作成
    x = torch.tensor([1, 2, 3, 4, 5])
    print(x)
    # 出力: tensor([1, 2, 3, 4, 5])
    
    # 2次元Tensor（行列）
    matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(matrix)
    # 出力: tensor([[1, 2],
    #               [3, 4],
    #               [5, 6]])
    
    # データ型を指定
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(float_tensor)
    # 出力: tensor([1., 2., 3.])
    
    # 特殊なTensor
    zeros = torch.zeros(3, 4)  # 3x4の0行列
    ones = torch.ones(2, 3)    # 2x3の1行列
    random = torch.rand(2, 2)  # 2x2のランダム行列（0-1の一様分布）
    randn = torch.randn(2, 2)  # 2x2のランダム行列（標準正規分布）
    
    print("Zeros:\n", zeros)
    print("Ones:\n", ones)
    print("Random:\n", random)
    print("Randn:\n", randn)
    

### Tensorの属性
    
    
    x = torch.randn(3, 4)
    
    print(f"Shape: {x.shape}")           # 形状: torch.Size([3, 4])
    print(f"Size: {x.size()}")           # 形状（メソッド版）
    print(f"Data type: {x.dtype}")       # データ型: torch.float32
    print(f"Device: {x.device}")         # デバイス: cpu
    print(f"Requires grad: {x.requires_grad}")  # 勾配計算: False
    print(f"Number of dimensions: {x.ndim}")    # 次元数: 2
    print(f"Number of elements: {x.numel()}")   # 要素数: 12
    

### NumPyとの相互変換
    
    
    import numpy as np
    import torch
    
    # NumPy array → PyTorch Tensor
    numpy_array = np.array([1, 2, 3, 4, 5])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print(f"From NumPy: {tensor_from_numpy}")
    
    # PyTorch Tensor → NumPy array
    tensor = torch.tensor([10, 20, 30])
    numpy_from_tensor = tensor.numpy()
    print(f"To NumPy: {numpy_from_tensor}")
    
    # 重要: メモリを共有するため、一方を変更するともう一方も変わる
    numpy_array[0] = 100
    print(f"Changed tensor: {tensor_from_numpy}")  # 最初の要素が100に変わる
    

**⚠️ 注意**

`from_numpy()`と`numpy()`は、メモリを共有します。データのコピーが必要な場合は`tensor.clone()`や`numpy_array.copy()`を使いましょう。

## 4\. Tensorの基本操作

### 算術演算
    
    
    import torch
    
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    
    # 要素ごとの演算
    print(x + y)      # tensor([5, 7, 9])
    print(x - y)      # tensor([-3, -3, -3])
    print(x * y)      # tensor([4, 10, 18])
    print(x / y)      # tensor([0.25, 0.4, 0.5])
    
    # 演算子版とメソッド版
    print(torch.add(x, y))      # x + y と同じ
    print(torch.mul(x, y))      # x * y と同じ
    
    # インプレース演算（元のTensorを変更）
    x.add_(y)  # x = x + y と同じ（アンダースコア付き）
    print(x)   # tensor([5, 7, 9])
    

### 行列演算
    
    
    import torch
    
    # 行列積
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    
    # @ 演算子または torch.matmul()
    C = A @ B
    print(C)
    # 出力: tensor([[19, 22],
    #               [43, 50]])
    
    # 転置
    print(A.T)
    # 出力: tensor([[1, 3],
    #               [2, 4]])
    
    # ベクトルの内積
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5, 6])
    dot_product = torch.dot(v1, v2)
    print(f"Dot product: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
    

### 形状操作
    
    
    import torch
    
    x = torch.randn(2, 3, 4)  # 2x3x4のTensor
    print(f"Original shape: {x.shape}")
    
    # reshape: 新しい形状に変換（連続していない場合はコピー）
    y = x.reshape(2, 12)
    print(f"Reshaped: {y.shape}")  # torch.Size([2, 12])
    
    # view: 新しい形状に変換（連続している必要がある）
    z = x.view(-1)  # 1次元に平坦化（-1は自動計算）
    print(f"Flattened: {z.shape}")  # torch.Size([24])
    
    # unsqueeze: 次元を追加
    a = torch.tensor([1, 2, 3])
    b = a.unsqueeze(0)  # 0番目の次元に追加
    print(f"Original: {a.shape}, After unsqueeze(0): {b.shape}")
    # Original: torch.Size([3]), After unsqueeze(0): torch.Size([1, 3])
    
    # squeeze: 次元を削除（サイズが1の次元を削除）
    c = torch.randn(1, 3, 1, 4)
    d = c.squeeze()
    print(f"Before squeeze: {c.shape}, After squeeze: {d.shape}")
    # Before squeeze: torch.Size([1, 3, 1, 4]), After squeeze: torch.Size([3, 4])
    

## 5\. 自動微分（Autograd）の基礎

PyTorchの**autograd** 機能は、ディープラーニングの鍵となる自動微分を提供します。これにより、複雑な計算の勾配を自動的に計算できます。

### requires_gradとbackward()
    
    
    import torch
    
    # requires_grad=True で勾配追跡を有効化
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x: {x}")
    print(f"x.requires_grad: {x.requires_grad}")
    
    # 計算を実行
    y = x ** 2 + 3 * x + 1
    print(f"y: {y}")
    
    # 勾配計算（dy/dx）
    y.backward()  # 勾配を計算
    print(f"x.grad: {x.grad}")  # dy/dx = 2x + 3 = 2*2 + 3 = 7
    

**数学的背景：**

$$y = x^2 + 3x + 1$$

$$\frac{dy}{dx} = 2x + 3$$

$$x = 2 のとき、\frac{dy}{dx} = 2(2) + 3 = 7$$

### 複雑な計算グラフ
    
    
    import torch
    
    # 複数の変数で計算グラフを構築
    a = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([4.0], requires_grad=True)
    
    # 複雑な計算
    c = a * b           # c = 3 * 4 = 12
    d = c ** 2          # d = 12^2 = 144
    e = torch.sin(d)    # e = sin(144)
    f = e + c           # f = sin(144) + 12
    
    print(f"f: {f}")
    
    # 勾配計算
    f.backward()
    
    print(f"df/da: {a.grad}")
    print(f"df/db: {b.grad}")
    
    
    
    ```mermaid
    graph LR
        A[a=3] --> C[c=a*b]
        B[b=4] --> C
        C --> D[d=c^2]
        D --> E[e=sin d]
        E --> F[f=e+c]
        C --> F
    ```

### 勾配の蓄積と初期化
    
    
    import torch
    
    x = torch.tensor([1.0], requires_grad=True)
    
    # 1回目の計算
    y = x ** 2
    y.backward()
    print(f"1st gradient: {x.grad}")  # 2.0
    
    # 2回目の計算（勾配が蓄積される！）
    y = x ** 3
    y.backward()
    print(f"2nd gradient (accumulated): {x.grad}")  # 2.0 + 3.0 = 5.0
    
    # 勾配を0にリセット
    x.grad.zero_()
    y = x ** 3
    y.backward()
    print(f"3rd gradient (after reset): {x.grad}")  # 3.0
    

**⚠️ 重要**

PyTorchはデフォルトで勾配を**蓄積** します。学習ループでは、各イテレーションで`optimizer.zero_grad()`または`tensor.grad.zero_()`を呼んで勾配をリセットする必要があります。

## 6\. GPUの使用

PyTorchでは、Tensorを簡単にGPUに移動して高速計算ができます。
    
    
    import torch
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # TensorをGPUに移動
    x = torch.randn(3, 3)
    x = x.to(device)
    print(f"x device: {x.device}")
    
    # GPU上での計算
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(f"z device: {z.device}")
    
    # CPUに戻す
    z_cpu = z.to("cpu")
    print(f"z_cpu device: {z_cpu.device}")
    
    # または .cpu() メソッド
    z_cpu = z.cpu()
    

**💡 ベストプラクティス**

モデルとデータを同じデバイス上に配置することが重要です。異なるデバイス間での演算はエラーになります。

## 7\. 最初の完全なPyTorchプログラム

ここまで学んだ内容を統合して、線形関数を学習する簡単なプログラムを書いてみましょう。
    
    
    import torch
    
    # データ生成: y = 3x + 2 にノイズを加える
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 3 * X + 2 + torch.randn(100, 1) * 0.5
    
    # パラメータ初期化（学習対象）
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    # 学習率
    learning_rate = 0.01
    
    # 学習ループ
    for epoch in range(100):
        # 予測
        y_pred = w * X + b
    
        # 損失計算（平均二乗誤差）
        loss = ((y_pred - y_true) ** 2).mean()
    
        # 勾配計算
        loss.backward()
    
        # パラメータ更新（勾配降下法）
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
    
        # 勾配をリセット
        w.grad.zero_()
        b.grad.zero_()
    
        # 10エポックごとに結果を表示
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    print(f"\n学習完了！")
    print(f"最終的なパラメータ: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"真の値: w = 3.0, b = 2.0")
    

**出力例：**
    
    
    Epoch 10: Loss = 0.3245, w = 2.7234, b = 1.8456
    Epoch 20: Loss = 0.2456, w = 2.8567, b = 1.9123
    Epoch 30: Loss = 0.2123, w = 2.9123, b = 1.9567
    ...
    Epoch 100: Loss = 0.1234, w = 2.9876, b = 1.9934
    
    学習完了！
    最終的なパラメータ: w = 2.9876, b = 1.9934
    真の値: w = 3.0, b = 2.0
    

## 演習問題

**演習1：Tensorの作成と操作**

以下の操作を実装してください：

  1. 5x5のランダムTensor（標準正規分布）を作成
  2. そのTensorの平均、標準偏差、最大値、最小値を計算
  3. Tensorを1次元に平坦化

    
    
    # ここにコードを書く
    

**演習2：NumPy変換**

NumPy配列`np.array([[1, 2], [3, 4], [5, 6]])`をPyTorch Tensorに変換し、各要素を2倍にして、再度NumPy配列に戻してください。

**演習3：自動微分**

関数 $f(x, y) = x^2 + y^2 + 2xy$ について、点 $(x=1, y=2)$ における偏微分 $\frac{\partial f}{\partial x}$ と $\frac{\partial f}{\partial y}$ をautogradで計算してください。

ヒント: 数学的には $\frac{\partial f}{\partial x} = 2x + 2y$、$\frac{\partial f}{\partial y} = 2y + 2x$

**演習4：GPU転送**

3x3のランダムTensorを作成し、GPU（利用可能な場合）に転送し、そのTensorの2乗を計算して、CPUに戻してください。

**演習5：簡単な最適化問題**

関数 $f(x) = (x - 5)^2$ を最小化するxの値を勾配降下法で求めてください。初期値は $x=0$、学習率は0.1、100ステップ実行してください。

期待される答え: $x \approx 5$

## まとめ

この章では、PyTorchの基礎を学びました：

  * ✅ PyTorchは動的計算グラフを採用した柔軟なディープラーニングフレームワーク
  * ✅ TensorFlowと比較して、Pythonライクな直感的なコーディングが可能
  * ✅ Tensorの作成、操作、NumPyとの相互変換
  * ✅ 自動微分（autograd）による勾配計算
  * ✅ GPUを使った高速計算

**🎉 次のステップ**

次章では、Tensorのより高度な操作（インデックス、スライス、ブロードキャスティング）を学び、実際のデータ処理に必要なスキルを身につけます。

* * *

**参考リソース**

  * [PyTorch公式ドキュメント](<https://pytorch.org/docs/stable/index.html>)
  * [PyTorch公式チュートリアル](<https://pytorch.org/tutorials/>)
  * [PyTorch GitHub](<https://github.com/pytorch/pytorch>)
