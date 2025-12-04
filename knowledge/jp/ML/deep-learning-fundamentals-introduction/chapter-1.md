---
title: 第1章：ディープラーニングの基礎概念
chapter_title: 第1章：ディープラーニングの基礎概念
subtitle: 定義、歴史、ニューラルネットワークの基本構造を理解する
reading_time: 30-35分
difficulty: 初級
code_examples: 8
exercises: 5
---

この章では、ディープラーニングの定義と歴史、ニューラルネットワークの基本構造、活性化関数について学びます。パーセプトロンから現代のディープラーニングまでの発展を追い、理論を実装コードで確認します。 

## 学習目標

  * ディープラーニングの定義と機械学習との関係を理解する
  * ディープラーニングの歴史的発展を把握する
  * ニューラルネットワークの基本構造（層、ニューロン、重み）を理解する
  * 主要な活性化関数（Sigmoid、ReLU、Softmax）の特性を学ぶ
  * NumPyとPyTorchで簡単なニューラルネットワークを実装できる

## 1\. ディープラーニングの定義

### 1.1 機械学習との関係

**ディープラーニング（Deep Learning、深層学習）** は、**機械学習（Machine Learning）** の一分野であり、多層のニューラルネットワークを用いてデータから自動的に特徴を学習する技術です。
    
    
    ```mermaid
    graph TB
        A[人工知能 AIArtificial Intelligence] --> B[機械学習 MLMachine Learning]
        B --> C[ディープラーニング DLDeep Learning]
    
        A --> A1[ルールベースシステムif-then rules]
        B --> B1[従来の機械学習Decision Tree, SVM]
        C --> C1[深層ニューラルネットワークCNN, RNN, Transformer]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
    ```

項目 | 従来の機械学習 | ディープラーニング  
---|---|---  
**特徴抽出** | 人間が手動で設計 | 自動的に学習  
**モデル構造** | 浅い（1-2層） | 深い（3層以上〜数百層）  
**データ量** | 少量〜中量でも効果的 | 大量データで真価を発揮  
**計算リソース** | CPUで十分 | GPU推奨（並列計算）  
**精度** | タスク依存 | 画像・音声・テキストで高精度  
  
### 1.2 「深い」とは何か

「深層（Deep）」とは、ニューラルネットワークの**層の数** を指します：

  * **浅いネットワーク（Shallow Network）** : 1-2層の隠れ層
  * **深いネットワーク（Deep Network）** : 3層以上の隠れ層
  * **超深層ネットワーク** : 数十〜数百層（ResNet-152、GPT-3など）

**重要な概念** : 層が深くなると、**階層的特徴表現（Hierarchical Feature Learning）** が可能になります。

  * **低層（初期層）** : 単純な特徴（エッジ、色、テクスチャ）
  * **中層** : 複雑なパターン（目、鼻、耳などの部品）
  * **高層（深い層）** : 抽象的概念（顔全体、オブジェクト全体）

### 1.3 表現学習と特徴抽出

ディープラーニングの本質は**表現学習（Representation Learning）** です：

  * **従来のアプローチ** : エンジニアが手動で特徴量を設計（SIFT、HOGなど）
  * **ディープラーニング** : ネットワークがデータから自動的に最適な特徴を学習

例えば、画像分類タスクでは：

  * 従来: エンジニアが「エッジ検出」「色ヒストグラム」などの特徴を設計
  * DL: ネットワークが学習データから自動的に有用な特徴を発見

## 2\. ディープラーニングの歴史

### 2.1 第1期：誕生期（1943-1969）

#### パーセプトロン（Perceptron, 1958）

フランク・ローゼンブラット（Frank Rosenblatt）が発明した、最初の学習アルゴリズムを持つニューラルネットワーク。

  * **単層構造** : 入力層と出力層のみ
  * **線形分離可能** : 直線で分けられる問題のみ解ける
  * **限界** : XOR問題が解けない（ミンスキーとパパートの批判、1969）

### 2.2 第2期：冬の時代（1969-1986）

パーセプトロンの限界が明らかになり、AI研究への資金が減少。しかし、重要な理論的進展がありました：

  * **多層パーセプトロン（MLP）** : 隠れ層を導入することで非線形問題に対応
  * **問題** : 効果的な学習アルゴリズムが未発見

### 2.3 第3期：復活期（1986-2006）

#### 誤差逆伝播法（Backpropagation, 1986）

ラメルハート、ヒントン、ウィリアムズが多層ニューラルネットワークの効率的な学習方法を提案。

  * **連鎖律（Chain Rule）** を用いた勾配計算
  * **多層ネットワーク** の学習が可能に
  * **限界** : 深いネットワークでは勾配消失問題（Vanishing Gradient Problem）

### 2.4 第4期：深層学習の夜明け（2006-2012）

ジェフリー・ヒントン（Geoffrey Hinton）らによる深層信念ネットワーク（Deep Belief Network, DBN）の提案：

  * **事前学習（Pre-training）** : 教師なし学習で重みを初期化
  * **ファインチューニング（Fine-tuning）** : 教師あり学習で調整
  * **成果** : 深いネットワークの学習が実用的に

### 2.5 第5期：ディープラーニング革命（2012-現在）

#### ImageNet 2012: AlexNet

アレックス・クリジェフスキー（Alex Krizhevsky）らのAlexNetが画像認識コンペティションで圧勝：

  * **5層の畳み込みニューラルネットワーク（CNN）**
  * **ReLU活性化関数** の採用
  * **GPU並列計算** の活用
  * **Dropout** による正則化
  * **成果** : エラー率を従来手法の26%から15%に削減

#### その後の主要なマイルストーン

  * **2014** : GAN（Generative Adversarial Networks）の提案
  * **2015** : ResNet（152層）がImageNetで人間を超える精度を達成
  * **2017** : Transformer（注意機構）の提案
  * **2018** : BERT（自然言語処理の革命）
  * **2020** : GPT-3（1750億パラメータの大規模言語モデル）
  * **2022** : ChatGPT、Stable Diffusion（生成AIの実用化）

    
    
    ```mermaid
    timeline
        title ディープラーニングの歴史
        1958 : パーセプトロン
        1969 : 冬の時代開始
        1986 : 誤差逆伝播法
        2006 : 深層学習の夜明け
        2012 : AlexNet革命
        2017 : Transformer
        2022 : ChatGPT
    ```

## 3\. ニューラルネットワークの基本構造

### 3.1 層の種類

ニューラルネットワークは、複数の**層（Layer）** から構成されます：

  * **入力層（Input Layer）** : データを受け取る層
  * **隠れ層（Hidden Layer）** : データを変換・処理する層（1層以上）
  * **出力層（Output Layer）** : 最終的な予測結果を出力する層

    
    
    ```mermaid
    graph LR
        I1((入力1)) --> H1((隠れ1-1))
        I2((入力2)) --> H1
        I3((入力3)) --> H1
    
        I1 --> H2((隠れ1-2))
        I2 --> H2
        I3 --> H2
    
        I1 --> H3((隠れ1-3))
        I2 --> H3
        I3 --> H3
    
        H1 --> H4((隠れ2-1))
        H2 --> H4
        H3 --> H4
    
        H1 --> H5((隠れ2-2))
        H2 --> H5
        H3 --> H5
    
        H4 --> O1((出力1))
        H5 --> O1
        H4 --> O2((出力2))
        H5 --> O2
    
        style I1 fill:#e3f2fd
        style I2 fill:#e3f2fd
        style I3 fill:#e3f2fd
        style H1 fill:#fff3e0
        style H2 fill:#fff3e0
        style H3 fill:#fff3e0
        style H4 fill:#f3e5f5
        style H5 fill:#f3e5f5
        style O1 fill:#e8f5e9
        style O2 fill:#e8f5e9
    ```

### 3.2 ニューロンと活性化

各層の**ニューロン（Neuron）** は、以下の処理を行います：

  1. **加重和の計算** : 入力 × 重み + バイアス
  2. **活性化関数の適用** : 非線形変換

数式で表すと：

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = f(z)$$

ここで：

  * $\mathbf{x}$: 入力ベクトル
  * $\mathbf{w}$: 重みベクトル
  * $b$: バイアス（切片）
  * $z$: 線形結合（加重和）
  * $f$: 活性化関数
  * $a$: 活性化出力

### 3.3 重みとバイアス

**重み（Weight）** と**バイアス（Bias）** は、ニューラルネットワークが学習するパラメータです：

  * **重み $w$** : 各入力の重要度を表す係数（学習によって最適化）
  * **バイアス $b$** : 活性化関数の閾値を調整する定数項

**直感的理解** : 線形回帰 $y = wx + b$ との類推

  * $w$: 傾き（入力が出力に与える影響の大きさ）
  * $b$: 切片（出力の基準値のシフト）

## 4\. 活性化関数

**活性化関数（Activation Function）** は、ニューラルネットワークに**非線形性** を導入する関数です。活性化関数がないと、どれだけ層を深くしても線形変換の組み合わせにしかならず、複雑なパターンを学習できません。

### 4.1 Sigmoid関数

最も古典的な活性化関数の一つで、出力を0から1の範囲に圧縮します。

**数式** :

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**特徴** :

  * 出力範囲: $(0, 1)$
  * 微分可能で滑らか
  * 確率として解釈可能（二値分類の出力層で使用）

**問題点** :

  * **勾配消失問題** : $x$が大きいまたは小さいとき、勾配がほぼ0になる
  * **出力が中心化されていない** : 常に正の値（学習効率が低下）
  * **計算コスト** : 指数関数の計算が重い

**NumPy実装** :
    
    
    import numpy as np
    
    def sigmoid(x):
        """
        Sigmoid活性化関数
    
        Parameters:
        -----------
        x : array-like
            入力
    
        Returns:
        --------
        array-like
            0から1の範囲の出力
        """
        return 1 / (1 + np.exp(-x))
    
    # 使用例
    x = np.array([-2, -1, 0, 1, 2])
    print("入力:", x)
    print("Sigmoid出力:", sigmoid(x))
    # 出力例: [0.119, 0.269, 0.5, 0.731, 0.881]
    

### 4.2 ReLU（Rectified Linear Unit）

現代のディープラーニングで最も広く使われる活性化関数です。

**数式** :

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\\ 0 & \text{if } x \leq 0 \end{cases}$$

**特徴** :

  * 計算が非常に高速（max演算のみ）
  * 勾配消失問題を軽減（$x > 0$のとき勾配は常に1）
  * 疎な活性化（約50%のニューロンが0になる）

**問題点** :

  * **Dying ReLU問題** : 負の入力で勾配が0になり、ニューロンが「死ぬ」

**NumPy実装** :
    
    
    def relu(x):
        """
        ReLU活性化関数
    
        Parameters:
        -----------
        x : array-like
            入力
    
        Returns:
        --------
        array-like
            負の値は0、正の値はそのまま
        """
        return np.maximum(0, x)
    
    # 使用例
    x = np.array([-2, -1, 0, 1, 2])
    print("入力:", x)
    print("ReLU出力:", relu(x))
    # 出力例: [0, 0, 0, 1, 2]
    

### 4.3 Softmax関数

多クラス分類の出力層で使用される活性化関数で、出力を確率分布に変換します。

**数式** :

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**特徴** :

  * 出力の合計が1になる（確率分布）
  * すべての出力が0から1の範囲
  * 最も大きい入力値が最も大きい確率を持つ

**NumPy実装** :
    
    
    def softmax(x):
        """
        Softmax活性化関数
    
        Parameters:
        -----------
        x : array-like
            入力（ロジット）
    
        Returns:
        --------
        array-like
            確率分布（合計が1）
        """
        # 数値安定性のため、最大値を引く
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    # 使用例
    x = np.array([2.0, 1.0, 0.1])
    print("入力:", x)
    print("Softmax出力:", softmax(x))
    print("合計:", softmax(x).sum())
    # 出力例: [0.659, 0.242, 0.099]
    # 合計: 1.0
    

### 4.4 tanh（双曲線正接）

Sigmoidの改良版で、出力を-1から1の範囲に圧縮します。

**数式** :

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

**特徴** :

  * 出力範囲: $(-1, 1)$
  * 出力が中心化されている（Sigmoidより学習効率が良い）
  * RNN（再帰型ニューラルネットワーク）でよく使用

**NumPy実装** :
    
    
    def tanh(x):
        """
        tanh活性化関数
    
        Parameters:
        -----------
        x : array-like
            入力
    
        Returns:
        --------
        array-like
            -1から1の範囲の出力
        """
        return np.tanh(x)
    
    # 使用例
    x = np.array([-2, -1, 0, 1, 2])
    print("入力:", x)
    print("tanh出力:", tanh(x))
    # 出力例: [-0.964, -0.762, 0.0, 0.762, 0.964]
    

### 4.5 活性化関数の比較

関数 | 出力範囲 | 計算速度 | 勾配消失 | 主な用途  
---|---|---|---|---  
**Sigmoid** | (0, 1) | 遅い | あり | 二値分類の出力層  
**tanh** | (-1, 1) | 遅い | あり | RNNの隠れ層  
**ReLU** | [0, ∞) | 速い | なし（正の領域） | CNNの隠れ層（最も一般的）  
**Softmax** | (0, 1)、合計=1 | 中程度 | N/A | 多クラス分類の出力層  
  
## 5\. 簡単なニューラルネットワークの実装

### 5.1 NumPyでの実装

まず、NumPyを使って基礎から実装します。これにより、ニューラルネットワークの内部動作を理解できます。
    
    
    import numpy as np
    
    class SimpleNN:
        """
        簡単な3層ニューラルネットワーク
    
        構造: 入力層 → 隠れ層 → 出力層
        """
    
        def __init__(self, input_size, hidden_size, output_size):
            """
            Parameters:
            -----------
            input_size : int
                入力層のニューロン数
            hidden_size : int
                隠れ層のニューロン数
            output_size : int
                出力層のニューロン数
            """
            # 重みの初期化（小さなランダム値）
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
    
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))
    
        def forward(self, X):
            """
            順伝播（Forward Propagation）
    
            Parameters:
            -----------
            X : array-like, shape (n_samples, input_size)
                入力データ
    
            Returns:
            --------
            array-like, shape (n_samples, output_size)
                出力（確率分布）
            """
            # 第1層: 入力 → 隠れ層
            self.z1 = np.dot(X, self.W1) + self.b1  # 線形変換
            self.a1 = relu(self.z1)                 # ReLU活性化
    
            # 第2層: 隠れ層 → 出力層
            self.z2 = np.dot(self.a1, self.W2) + self.b2  # 線形変換
            self.a2 = softmax(self.z2)                     # Softmax活性化
    
            return self.a2
    
    # 使用例
    # 入力: 4次元、隠れ層: 10ニューロン、出力: 3クラス
    model = SimpleNN(input_size=4, hidden_size=10, output_size=3)
    
    # ダミーデータ（5サンプル、4次元）
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 2.9, 5.6, 1.8],
        [5.8, 2.7, 5.1, 1.9],
        [5.0, 3.4, 1.5, 0.2],
        [6.7, 3.1, 5.6, 2.4]
    ])
    
    # 予測
    predictions = model.forward(X)
    print("予測確率:\n", predictions)
    print("\n予測クラス:", np.argmax(predictions, axis=1))
    

### 5.2 PyTorchでの実装

次に、同じネットワークをPyTorchで実装します。PyTorchは自動微分やGPU対応が組み込まれており、より簡潔に書けます。
    
    
    import torch
    import torch.nn as nn
    
    class SimpleNNPyTorch(nn.Module):
        """
        PyTorchによる3層ニューラルネットワーク
        """
    
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNNPyTorch, self).__init__()
    
            # 層の定義
            self.fc1 = nn.Linear(input_size, hidden_size)  # 入力 → 隠れ層
            self.relu = nn.ReLU()                          # ReLU活性化
            self.fc2 = nn.Linear(hidden_size, output_size) # 隠れ層 → 出力
            self.softmax = nn.Softmax(dim=1)               # Softmax活性化
    
        def forward(self, x):
            """
            順伝播
    
            Parameters:
            -----------
            x : torch.Tensor, shape (n_samples, input_size)
                入力データ
    
            Returns:
            --------
            torch.Tensor, shape (n_samples, output_size)
                出力（確率分布）
            """
            x = self.fc1(x)      # 線形変換
            x = self.relu(x)     # ReLU活性化
            x = self.fc2(x)      # 線形変換
            x = self.softmax(x)  # Softmax活性化
            return x
    
    # 使用例
    model_pytorch = SimpleNNPyTorch(input_size=4, hidden_size=10, output_size=3)
    
    # ダミーデータをTensorに変換
    X_torch = torch.tensor(X, dtype=torch.float32)
    
    # 予測
    with torch.no_grad():  # 勾配計算を無効化（推論時）
        predictions_pytorch = model_pytorch(X_torch)
    
    print("PyTorch予測確率:\n", predictions_pytorch.numpy())
    print("\n予測クラス:", torch.argmax(predictions_pytorch, dim=1).numpy())
    

### 5.3 NumPy実装 vs PyTorch実装

項目 | NumPy実装 | PyTorch実装  
---|---|---  
**コード量** | 多い（全て手動実装） | 少ない（高レベルAPI）  
**学習目的** | 内部動作の理解に最適 | 実用的な開発に最適  
**自動微分** | 手動実装が必要 | 自動的に計算  
**GPU対応** | 困難 | 簡単（.cuda()のみ）  
**パフォーマンス** | 遅い | 速い（最適化済み）  
  
## 演習問題

演習1: 活性化関数の実装と可視化

**問題** : Sigmoid、ReLU、tanhの3つの活性化関数を実装し、x = -5から5までの範囲でグラフを描画してください。各関数の特性を比較してください。

**ヒント** :

  * matplotlib.pyplot を使用
  * np.linspace(-5, 5, 100) で等間隔の点を生成
  * plt.plot() で各関数をプロット

演習2: Softmaxの温度パラメータ

**問題** : Softmax関数に温度パラメータ T を導入した以下の式を実装してください：

$$\text{softmax}(\mathbf{x}, T)_i = \frac{e^{x_i/T}}{\sum_{j=1}^{n} e^{x_j/T}}$$

入力 x = [2.0, 1.0, 0.1] に対して、T = 0.5, 1.0, 2.0 の場合の出力を比較してください。温度が高いと出力がどう変化しますか？

演習3: XOR問題

**問題** : XOR問題（排他的論理和）を解くニューラルネットワークを設計してください。

  * 入力: 2次元 (x1, x2)、各要素は0または1
  * 出力: x1 XOR x2（x1とx2が異なれば1、同じなら0）
  * ネットワーク構造: 入力層(2) → 隠れ層(2) → 出力層(1)

SimpleNNクラスを修正して、XOR問題に対応させてください（学習は次章で扱います）。

演習4: 重みの初期化

**問題** : SimpleNNクラスの重み初期化を以下の3つの方法で実装し、初期化後の重みの分布を比較してください：

  1. **全て0** : W = np.zeros(...)
  2. **正規分布** : W = np.random.randn(...) * 0.01
  3. **Xavier初期化** : W = np.random.randn(...) * np.sqrt(1/n_in)

各初期化の重みの平均と標準偏差を計算し、どの方法が最も適切か考察してください。

演習5: 多層ネットワーク

**問題** : SimpleNNクラスを拡張して、隠れ層を2層に増やしてください：

  * 入力層 → 隠れ層1（10ニューロン、ReLU） → 隠れ層2（5ニューロン、ReLU） → 出力層（3クラス、Softmax）

4次元の入力データに対して、各層の出力形状を確認してください。

## まとめ

この章では、ディープラーニングの基礎概念を学びました：

  * **定義** : ディープラーニングは多層ニューラルネットワークによる表現学習
  * **歴史** : パーセプトロンから現代まで、約70年の発展の軌跡
  * **構造** : 入力層、隠れ層、出力層から構成され、各層はニューロンの集合
  * **活性化関数** : Sigmoid、ReLU、Softmax、tanhなどで非線形性を導入
  * **実装** : NumPyで原理を理解し、PyTorchで効率的に実装

**次章の予告** : 第2章では、順伝播の詳細な計算、重み行列の役割、損失関数について学び、ニューラルネットワークがどのように予測を行うかを深く理解します。
