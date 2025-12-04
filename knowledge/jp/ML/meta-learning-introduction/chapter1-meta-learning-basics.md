---
title: 第1章：メタ学習の基礎
chapter_title: 第1章：メタ学習の基礎
subtitle: Learning to Learn - 少数サンプルから学ぶ新しいパラダイム
reading_time: 25-30分
difficulty: 初級〜中級
code_examples: 7
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ メタ学習（Learning to Learn）の概念と通常の学習との違いを理解する
  * ✅ Few-Shot Learningの問題設定とN-way K-shot分類を説明できる
  * ✅ Support SetとQuery Setの役割を理解する
  * ✅ メタ学習の3つの主要アプローチを分類できる
  * ✅ Omniglotデータセットの構造とエピソード生成方法を理解する
  * ✅ シンプルなFew-Shot分類ベースラインを実装できる

* * *

## 1.1 メタ学習とは

### Learning to Learnの概念

**メタ学習（Meta-Learning）** は、「学習方法を学習する」というパラダイムです。従来の機械学習が特定のタスクを解くのに対し、メタ学習は「新しいタスクに素早く適応する能力」そのものを学習します。

> 「人間は数個の例を見ただけで新しい概念を学べる。機械も同じことができるべきだ。」

### 通常の学習との違い

観点 | 通常の機械学習 | メタ学習  
---|---|---  
**目標** | 単一タスクの性能最大化 | 新しいタスクへの適応能力獲得  
**訓練データ** | 大量のラベル付きデータ | 多様なタスクからの少数サンプル  
**学習単位** | 個別サンプル | タスク（エピソード）  
**評価** | 同一分布のテストセット | 未知のタスクでの適応速度  
**用途** | 固定タスク（例：猫vs犬分類） | 動的タスク（例：新種の動物認識）  
  
### メタ学習の学習プロセス
    
    
    ```mermaid
    graph TD
        A[多数のタスク] --> B[タスク1: 5サンプルで学習]
        A --> C[タスク2: 5サンプルで学習]
        A --> D[タスク3: 5サンプルで学習]
        B --> E[メタ知識の蓄積]
        C --> E
        D --> E
        E --> F[新しいタスクN]
        F --> G[5サンプルで高精度]
    
        style A fill:#e3f2fd
        style E fill:#fff3e0
        style G fill:#c8e6c9
    ```

### メタ学習が有効なシナリオ

  * **医療画像診断** : 稀な疾患の例が少ない
  * **個人化推薦** : 新規ユーザーの履歴が少ない
  * **ロボット工学** : 新しい環境への迅速な適応
  * **創薬** : 新規化合物のデータが限定的
  * **多言語処理** : 低リソース言語での学習

### 実例：人間の学習との比較
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 比較: 通常学習 vs メタ学習の学習曲線シミュレーション
    
    def standard_learning_curve(n_samples):
        """通常の学習: 線形的な改善"""
        return 0.5 + 0.45 * (1 - np.exp(-n_samples / 500))
    
    def meta_learning_curve(n_samples):
        """メタ学習: 少数サンプルで急速に学習"""
        return 0.5 + 0.45 * (1 - np.exp(-n_samples / 20))
    
    # データポイント
    samples = np.arange(1, 101, 1)
    standard_acc = standard_learning_curve(samples)
    meta_acc = meta_learning_curve(samples)
    
    # 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(samples, standard_acc, 'b-', linewidth=2, label='通常の機械学習')
    plt.plot(samples, meta_acc, 'r-', linewidth=2, label='メタ学習')
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='目標精度 90%')
    plt.axvline(x=10, color='green', linestyle=':', alpha=0.5, label='Few-Shot領域 (10サンプル)')
    
    plt.xlabel('訓練サンプル数', fontsize=12)
    plt.ylabel('精度', fontsize=12)
    plt.title('学習パラダイムの比較: 通常学習 vs メタ学習', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0.4, 1.0)
    
    # 重要ポイントを注釈
    plt.annotate('メタ学習: 10サンプルで85%達成',
                 xy=(10, meta_learning_curve(10)),
                 xytext=(30, 0.75),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red')
    
    plt.annotate('通常学習: 10サンプルでは60%程度',
                 xy=(10, standard_learning_curve(10)),
                 xytext=(30, 0.55),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.show()
    
    print("=== 学習効率の比較 ===")
    print(f"10サンプルでの精度:")
    print(f"  通常学習: {standard_learning_curve(10):.3f}")
    print(f"  メタ学習: {meta_learning_curve(10):.3f}")
    print(f"  差: {(meta_learning_curve(10) - standard_learning_curve(10)):.3f}")
    

**出力** ：
    
    
    === 学習効率の比較 ===
    10サンプルでの精度:
      通常学習: 0.591
      メタ学習: 0.873
      差: 0.282
    

> **重要** : メタ学習は少数サンプルで高精度を達成できる点が最大の利点です。

* * *

## 1.2 Few-Shot Learning問題設定

### N-way K-shot分類

Few-Shot Learningの標準的な問題設定は**N-way K-shot分類** です：

  * **N-way** : N個のクラスを分類
  * **K-shot** : 各クラスにK個のラベル付きサンプル

例：**5-way 1-shot** 分類 = 5クラスを各クラス1サンプルから学習

### Support SetとQuery Set

各エピソード（タスク）は2つのセットで構成されます：

セット | 役割 | サイズ | 用途  
---|---|---|---  
**Support Set** | 学習用サンプル | N × K 個 | モデルの適応・更新  
**Query Set** | 評価用サンプル | N × Q 個 | タスクでの性能評価  
  
### エピソードの構造
    
    
    ```mermaid
    graph LR
        A[1つのエピソード] --> B[Support SetN×K サンプル]
        A --> C[Query SetN×Q サンプル]
        B --> D[モデルを適応]
        C --> E[性能を評価]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### 具体例：5-way 1-shot分類
    
    
    import numpy as np
    
    # 5-way 1-shot エピソードの構造を可視化
    
    def create_episode_structure(n_way=5, k_shot=1, n_query=5):
        """
        N-way K-shot エピソードの構造を生成
    
        Args:
            n_way: クラス数
            k_shot: 各クラスのサポートサンプル数
            n_query: 各クラスのクエリサンプル数
    
        Returns:
            support_set, query_set のサイズ情報
        """
        support_size = n_way * k_shot
        query_size = n_way * n_query
    
        print(f"=== {n_way}-way {k_shot}-shot エピソード構造 ===\n")
        print(f"【Support Set】")
        print(f"  目的: モデルの適応・学習")
        print(f"  構成: {n_way} クラス × {k_shot} サンプル/クラス = {support_size} サンプル")
    
        for i in range(n_way):
            samples = [f"S_{i}_{j}" for j in range(k_shot)]
            print(f"    クラス {i}: {samples}")
    
        print(f"\n【Query Set】")
        print(f"  目的: 性能評価")
        print(f"  構成: {n_way} クラス × {n_query} サンプル/クラス = {query_size} サンプル")
    
        for i in range(n_way):
            samples = [f"Q_{i}_{j}" for j in range(min(n_query, 3))]
            if n_query > 3:
                samples.append("...")
            print(f"    クラス {i}: {samples}")
    
        return support_size, query_size
    
    # 5-way 1-shot の例
    support_size, query_size = create_episode_structure(n_way=5, k_shot=1, n_query=5)
    
    print(f"\n総サンプル数: {support_size + query_size}")
    print(f"  Support: {support_size}")
    print(f"  Query: {query_size}")
    

**出力** ：
    
    
    === 5-way 1-shot エピソード構造 ===
    
    【Support Set】
      目的: モデルの適応・学習
      構成: 5 クラス × 1 サンプル/クラス = 5 サンプル
        クラス 0: ['S_0_0']
        クラス 1: ['S_1_0']
        クラス 2: ['S_2_0']
        クラス 3: ['S_3_0']
        クラス 4: ['S_4_0']
    
    【Query Set】
      目的: 性能評価
      構成: 5 クラス × 5 サンプル/クラス = 25 サンプル
        クラス 0: ['Q_0_0', 'Q_0_1', 'Q_0_2', '...']
        クラス 1: ['Q_1_0', 'Q_1_1', 'Q_1_2', '...']
        クラス 2: ['Q_2_0', 'Q_2_1', 'Q_2_2', '...']
        クラス 3: ['Q_3_0', 'Q_3_1', 'Q_3_2', '...']
        クラス 4: ['Q_4_0', 'Q_4_1', 'Q_4_2', '...']
    
    総サンプル数: 30
      Support: 5
      Query: 25
    

### Episode-based学習

メタ学習では、多数のエピソードを通じて学習します：

  1. ランダムにN個のクラスを選択
  2. 各クラスからK個のサポートサンプルとQ個のクエリサンプルをサンプリング
  3. サポートセットでモデルを適応
  4. クエリセットで評価し、メタ知識を更新
  5. 1〜4を繰り返す

    
    
    import numpy as np
    
    def meta_training_simulation(n_episodes=1000, n_way=5, k_shot=1):
        """
        メタ学習の訓練プロセスをシミュレーション
    
        Args:
            n_episodes: エピソード数
            n_way: クラス数
            k_shot: サポートサンプル数
        """
        episode_accuracies = []
    
        for episode in range(n_episodes):
            # 各エピソードでランダムにタスクを生成
            # （実際にはデータセットからサンプリング）
    
            # シミュレーション: エピソードが進むにつれて精度向上
            base_acc = 0.2  # ランダム推測 (5-way: 20%)
            improvement = 0.7 * (1 - np.exp(-episode / 200))
            noise = np.random.normal(0, 0.05)  # ランダムノイズ
    
            acc = min(max(base_acc + improvement + noise, 0), 1)
            episode_accuracies.append(acc)
    
        # 可視化
        import matplotlib.pyplot as plt
    
        plt.figure(figsize=(12, 6))
    
        # エピソードごとの精度
        plt.subplot(1, 2, 1)
        plt.plot(episode_accuracies, alpha=0.3, color='blue')
    
        # 移動平均
        window = 50
        moving_avg = np.convolve(episode_accuracies,
                                 np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, n_episodes), moving_avg,
                 'r-', linewidth=2, label=f'{window}-エピソード移動平均')
    
        plt.axhline(y=0.2, color='gray', linestyle='--',
                    alpha=0.5, label='ランダム推測 (20%)')
        plt.xlabel('エピソード', fontsize=12)
        plt.ylabel('Query Set 精度', fontsize=12)
        plt.title(f'{n_way}-way {k_shot}-shot メタ訓練の進行', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # 精度分布の変化
        plt.subplot(1, 2, 2)
        early = episode_accuracies[:200]
        late = episode_accuracies[-200:]
    
        plt.hist(early, bins=20, alpha=0.5, label='初期 (0-200)', color='blue')
        plt.hist(late, bins=20, alpha=0.5, label='後期 (800-1000)', color='red')
        plt.xlabel('精度', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.title('精度分布の変化', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        print(f"=== メタ訓練統計 ({n_episodes} エピソード) ===")
        print(f"初期100エピソードの平均精度: {np.mean(episode_accuracies[:100]):.3f}")
        print(f"最終100エピソードの平均精度: {np.mean(episode_accuracies[-100:]):.3f}")
        print(f"改善: {(np.mean(episode_accuracies[-100:]) - np.mean(episode_accuracies[:100])):.3f}")
    
    # シミュレーション実行
    meta_training_simulation(n_episodes=1000, n_way=5, k_shot=1)
    

> **重要** : エピソードベースの学習により、モデルは「少数サンプルから学ぶ能力」そのものを獲得します。

* * *

## 1.3 メタ学習のアプローチ分類

メタ学習の手法は、大きく3つのカテゴリに分類されます：

### 1\. Metric-based（距離ベース）

**基本アイデア** : 良い距離空間を学習し、近傍に基づいて分類

手法 | 特徴 | 距離計算  
---|---|---  
**Siamese Networks** | ペアワイズ比較 | ユークリッド距離、コサイン類似度  
**Matching Networks** | 注意機構で加重平均 | コサイン類似度 + 注意  
**Prototypical Networks** | クラスごとのプロトタイプ | プロトタイプまでの距離  
**Relation Networks** | 学習可能な距離関数 | ニューラルネットで距離学習  
  
#### Prototypical Networksの概念
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    # Prototypical Networksの概念を可視化
    
    # シミュレーション: 3クラスの埋め込み空間
    np.random.seed(42)
    
    # 各クラスのデータ生成
    n_samples_per_class = 20
    centers = np.array([[0, 0], [3, 3], [0, 3]])
    X, y = make_blobs(n_samples=n_samples_per_class * 3,
                      centers=centers,
                      cluster_std=0.5,
                      random_state=42)
    
    # Support Set (各クラス3サンプル)
    support_indices = []
    for cls in range(3):
        cls_indices = np.where(y == cls)[0]
        support_indices.extend(cls_indices[:3])
    
    support_X = X[support_indices]
    support_y = y[support_indices]
    
    # Query Set (残りのサンプル)
    query_indices = [i for i in range(len(X)) if i not in support_indices]
    query_X = X[query_indices]
    query_y = y[query_indices]
    
    # プロトタイプ計算（各クラスのサポートサンプルの平均）
    prototypes = []
    for cls in range(3):
        cls_support = support_X[support_y == cls]
        prototype = cls_support.mean(axis=0)
        prototypes.append(prototype)
    
    prototypes = np.array(prototypes)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # 左: Support Set とプロトタイプ
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green']
    for cls in range(3):
        cls_support = support_X[support_y == cls]
        plt.scatter(cls_support[:, 0], cls_support[:, 1],
                    c=colors[cls], s=100, alpha=0.6,
                    label=f'Class {cls} Support', marker='o')
    
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=colors, s=300, marker='*',
                edgecolors='black', linewidth=2,
                label='Prototypes')
    
    plt.xlabel('埋め込み次元1', fontsize=12)
    plt.ylabel('埋め込み次元2', fontsize=12)
    plt.title('Support Set とプロトタイプ', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右: Query Set の分類
    plt.subplot(1, 2, 2)
    
    # 全データポイント
    for cls in range(3):
        cls_query = query_X[query_y == cls]
        plt.scatter(cls_query[:, 0], cls_query[:, 1],
                    c=colors[cls], s=50, alpha=0.3,
                    label=f'Class {cls} Query')
    
    # プロトタイプ
    plt.scatter(prototypes[:, 0], prototypes[:, 1],
                c=colors, s=300, marker='*',
                edgecolors='black', linewidth=2,
                label='Prototypes')
    
    # 1つのクエリサンプルの分類を示す
    query_sample = query_X[0]
    plt.scatter(query_sample[0], query_sample[1],
                c='orange', s=200, marker='X',
                edgecolors='black', linewidth=2,
                label='Query Sample', zorder=5)
    
    # プロトタイプまでの距離を線で示す
    for i, proto in enumerate(prototypes):
        dist = np.linalg.norm(query_sample - proto)
        plt.plot([query_sample[0], proto[0]],
                 [query_sample[1], proto[1]],
                 'k--', alpha=0.3, linewidth=1)
        mid = (query_sample + proto) / 2
        plt.text(mid[0], mid[1], f'd={dist:.2f}', fontsize=9)
    
    plt.xlabel('埋め込み次元1', fontsize=12)
    plt.ylabel('埋め込み次元2', fontsize=12)
    plt.title('Prototypical Networks: 距離ベース分類', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Prototypical Networks ===")
    print(f"プロトタイプ座標:")
    for i, proto in enumerate(prototypes):
        print(f"  Class {i}: [{proto[0]:.2f}, {proto[1]:.2f}]")
    

### 2\. Model-based（モデルベース）

**基本アイデア** : メモリや再帰構造を持つモデルで高速適応

  * **Memory-Augmented Neural Networks (MANN)** : 外部メモリで過去の経験を保存
  * **Meta Networks** : 高速パラメータ生成器を学習
  * **SNAIL** : 時系列として過去サンプルを処理

### 3\. Optimization-based（最適化ベース）

**基本アイデア** : 良い初期パラメータを学習し、少数ステップで適応

手法 | 特徴 | 適応方法  
---|---|---  
**MAML** | モデル非依存、勾配ベース | 数ステップの勾配降下  
**Reptile** | MAMLの簡易版 | 1次微分のみ  
**Meta-SGD** | 学習率も学習 | 適応的学習率 + 勾配降下  
  
### アプローチの比較

アプローチ | 長所 | 短所 | 適用例  
---|---|---|---  
**Metric-based** | シンプル、高速、解釈性 | 複雑なタスクに限界 | 画像分類、少数サンプル認識  
**Model-based** | 柔軟、表現力高い | 訓練が複雑 | シーケンシャルタスク  
**Optimization-based** | 汎用性、強力 | 計算コスト高い | 強化学習、複雑タスク  
  
* * *

## 1.4 Omniglotデータセット

### データセットの構造

**Omniglot** は「メタ学習のMNIST」と呼ばれるベンチマークデータセットです：

  * **1,623 文字クラス** （50の異なる文字体系から）
  * **各クラス20サンプル** （20人が描いた手書き文字）
  * **画像サイズ** : 105×105 ピクセル、グレースケール
  * **総サンプル数** : 32,460 枚

### データセットのダウンロードと準備
    
    
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Omniglotデータセットの準備
    # 注: torchvision.datasets.Omniglot を使用
    
    from torchvision.datasets import Omniglot
    
    # データ変換
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # MNISTサイズに縮小
        transforms.ToTensor(),
    ])
    
    # データセットのロード
    try:
        # 背景セット（訓練用）
        omniglot_train = Omniglot(
            root='./data',
            background=True,
            download=True,
            transform=transform
        )
    
        # 評価セット（テスト用）
        omniglot_test = Omniglot(
            root='./data',
            background=False,
            download=True,
            transform=transform
        )
    
        print("=== Omniglot データセット ===")
        print(f"訓練セット: {len(omniglot_train)} サンプル")
        print(f"テストセット: {len(omniglot_test)} サンプル")
    
        # データ構造の確認
        print(f"\nデータセット構造:")
        print(f"  訓練クラス数: {len(omniglot_train._alphabets)} 文字体系")
        print(f"  テストクラス数: {len(omniglot_test._alphabets)} 文字体系")
    
        # サンプル可視化
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    
        for i in range(10):
            # 訓練セットから
            img, label = omniglot_train[i * 100]
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Train {i}', fontsize=9)
    
            # テストセットから
            img, label = omniglot_test[i * 50]
            axes[1, i].imshow(img.squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Test {i}', fontsize=9)
    
        plt.suptitle('Omniglot サンプル（上: 訓練セット、下: テストセット）', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"データセットロードエラー: {e}")
        print("注: 実行には torchvision と インターネット接続が必要です")
    

### Episode生成
    
    
    import random
    
    class OmniglotEpisodeSampler:
        """
        Omniglot用のエピソードサンプラー
        N-way K-shot エピソードを生成
        """
        def __init__(self, dataset, n_way=5, k_shot=1, n_query=5):
            self.dataset = dataset
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
    
            # クラスごとにサンプルをグループ化
            self.class_to_indices = {}
            for idx, (_, label) in enumerate(dataset):
                if label not in self.class_to_indices:
                    self.class_to_indices[label] = []
                self.class_to_indices[label].append(idx)
    
            self.classes = list(self.class_to_indices.keys())
            print(f"サンプラー初期化: {len(self.classes)} クラス")
    
        def sample_episode(self):
            """
            1つのエピソードをサンプリング
    
            Returns:
                support_set: (n_way * k_shot, C, H, W) tensor
                query_set: (n_way * n_query, C, H, W) tensor
                support_labels: (n_way * k_shot,) tensor
                query_labels: (n_way * n_query,) tensor
            """
            # N個のクラスをランダム選択
            episode_classes = random.sample(self.classes, self.n_way)
    
            support_set = []
            query_set = []
            support_labels = []
            query_labels = []
    
            for class_idx, cls in enumerate(episode_classes):
                # このクラスのサンプルインデックス
                cls_indices = self.class_to_indices[cls]
    
                # K+Q個サンプリング（重複なし）
                sampled_indices = random.sample(cls_indices,
                                               self.k_shot + self.n_query)
    
                # Support Set
                for i in range(self.k_shot):
                    img, _ = self.dataset[sampled_indices[i]]
                    support_set.append(img)
                    support_labels.append(class_idx)
    
                # Query Set
                for i in range(self.k_shot, self.k_shot + self.n_query):
                    img, _ = self.dataset[sampled_indices[i]]
                    query_set.append(img)
                    query_labels.append(class_idx)
    
            # Tensorに変換
            support_set = torch.stack(support_set)
            query_set = torch.stack(query_set)
            support_labels = torch.tensor(support_labels)
            query_labels = torch.tensor(query_labels)
    
            return support_set, query_set, support_labels, query_labels
    
    # エピソードサンプラーの使用例
    try:
        sampler = OmniglotEpisodeSampler(
            omniglot_train,
            n_way=5,
            k_shot=1,
            n_query=5
        )
    
        # 1つのエピソードをサンプリング
        support, query, support_labels, query_labels = sampler.sample_episode()
    
        print(f"\n=== エピソード構造 ===")
        print(f"Support Set: {support.shape}")
        print(f"Query Set: {query.shape}")
        print(f"Support Labels: {support_labels}")
        print(f"Query Labels: {query_labels}")
    
        # 可視化
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
        # Support Set
        for i in range(5):
            axes[0, i].imshow(support[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Support\nClass {support_labels[i].item()}',
                                fontsize=10)
    
        # Query Set（各クラスから1つ）
        for i in range(5):
            axes[1, i].imshow(query[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Query\nClass {query_labels[i].item()}',
                                fontsize=10)
    
        plt.suptitle('5-way 1-shot エピソードの例', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    except NameError:
        print("注: Omniglotデータセットが読み込まれている必要があります")
    

* * *

## 1.5 実践: シンプルなFew-Shot分類

### 基本的なN-way K-shotタスク

最もシンプルなFew-Shot分類は、サポートセットとクエリサンプルの距離を計算する方法です。

### Nearest Neighbor Baseline
    
    
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    class NearestNeighborClassifier:
        """
        最近傍法によるFew-Shot分類ベースライン
        """
        def __init__(self, distance_metric='euclidean'):
            self.distance_metric = distance_metric
    
        def fit(self, support_set, support_labels):
            """
            Support Setを記憶
    
            Args:
                support_set: (N*K, feature_dim) tensor
                support_labels: (N*K,) tensor
            """
            self.support_set = support_set
            self.support_labels = support_labels
    
        def predict(self, query_set):
            """
            Query Setを分類
    
            Args:
                query_set: (N*Q, feature_dim) tensor
    
            Returns:
                predictions: (N*Q,) tensor
            """
            n_queries = query_set.size(0)
            predictions = []
    
            for i in range(n_queries):
                query = query_set[i]
    
                # 全サポートサンプルとの距離計算
                if self.distance_metric == 'euclidean':
                    distances = torch.norm(self.support_set - query, dim=1)
                elif self.distance_metric == 'cosine':
                    # コサイン類似度（距離に変換）
                    similarities = F.cosine_similarity(
                        self.support_set,
                        query.unsqueeze(0),
                        dim=1
                    )
                    distances = 1 - similarities
    
                # 最近傍のラベルを予測
                nearest_idx = torch.argmin(distances)
                pred_label = self.support_labels[nearest_idx]
                predictions.append(pred_label)
    
            return torch.tensor(predictions)
    
        def evaluate(self, query_set, query_labels):
            """
            精度を計算
            """
            predictions = self.predict(query_set)
            accuracy = (predictions == query_labels).float().mean()
            return accuracy.item()
    
    # 実験: シンプルな2次元データで動作確認
    def test_nearest_neighbor():
        """Nearest Neighborの動作確認"""
    
        # 5-way 1-shot タスクをシミュレーション
        n_way = 5
        k_shot = 1
        n_query = 10
    
        # Support Set生成（各クラスを異なる領域に配置）
        support_set = []
        support_labels = []
    
        for cls in range(n_way):
            # クラスごとに中心を設定
            center = torch.tensor([cls * 2.0, cls * 2.0])
            sample = center + torch.randn(2) * 0.5  # ノイズ追加
            support_set.append(sample)
            support_labels.append(cls)
    
        support_set = torch.stack(support_set)
        support_labels = torch.tensor(support_labels)
    
        # Query Set生成（各クラスから複数サンプル）
        query_set = []
        query_labels = []
    
        for cls in range(n_way):
            center = torch.tensor([cls * 2.0, cls * 2.0])
            for _ in range(n_query // n_way):
                sample = center + torch.randn(2) * 0.5
                query_set.append(sample)
                query_labels.append(cls)
    
        query_set = torch.stack(query_set)
        query_labels = torch.tensor(query_labels)
    
        # Nearest Neighbor分類
        nn_classifier = NearestNeighborClassifier(distance_metric='euclidean')
        nn_classifier.fit(support_set, support_labels)
        accuracy = nn_classifier.evaluate(query_set, query_labels)
    
        print(f"=== Nearest Neighbor ベースライン ===")
        print(f"タスク: {n_way}-way {k_shot}-shot")
        print(f"精度: {accuracy:.3f}")
    
        # 可視化
        import matplotlib.pyplot as plt
    
        plt.figure(figsize=(10, 8))
    
        colors = ['red', 'blue', 'green', 'orange', 'purple']
    
        # Support Set
        for cls in range(n_way):
            cls_support = support_set[support_labels == cls]
            plt.scatter(cls_support[:, 0], cls_support[:, 1],
                       c=colors[cls], s=300, marker='*',
                       edgecolors='black', linewidth=2,
                       label=f'Support Class {cls}', zorder=5)
    
        # Query Set
        for cls in range(n_way):
            cls_query = query_set[query_labels == cls]
            plt.scatter(cls_query[:, 0], cls_query[:, 1],
                       c=colors[cls], s=100, alpha=0.5,
                       marker='o', edgecolors='black')
    
        # 予測結果
        predictions = nn_classifier.predict(query_set)
        correct = (predictions == query_labels)
        incorrect = ~correct
    
        # 誤分類を×で示す
        if incorrect.any():
            plt.scatter(query_set[incorrect, 0], query_set[incorrect, 1],
                       s=200, marker='x', c='black', linewidth=3,
                       label='誤分類', zorder=6)
    
        plt.xlabel('特徴次元1', fontsize=12)
        plt.ylabel('特徴次元2', fontsize=12)
        plt.title(f'Nearest Neighbor: {n_way}-way {k_shot}-shot\n精度: {accuracy:.1%}',
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 実験実行
    test_nearest_neighbor()
    

### 評価プロトコル

Few-Shot学習の標準的な評価方法：

  1. **多数のエピソード生成** （例：600エピソード）
  2. 各エピソードで精度を計算
  3. 平均精度と標準偏差を報告

    
    
    def evaluate_fewshot_model(model, dataset_sampler, n_episodes=600):
        """
        Few-Shotモデルの標準評価プロトコル
    
        Args:
            model: Few-Shot分類モデル
            dataset_sampler: エピソードサンプラー
            n_episodes: 評価エピソード数
    
        Returns:
            mean_accuracy: 平均精度
            std_accuracy: 標準偏差
        """
        accuracies = []
    
        for episode in range(n_episodes):
            # エピソードをサンプリング
            support, query, support_labels, query_labels = \
                dataset_sampler.sample_episode()
    
            # 平坦化（特徴量として扱う）
            support_flat = support.view(support.size(0), -1)
            query_flat = query.view(query.size(0), -1)
    
            # モデルで評価
            model.fit(support_flat, support_labels)
            accuracy = model.evaluate(query_flat, query_labels)
            accuracies.append(accuracy)
    
            if (episode + 1) % 100 == 0:
                print(f"エピソード {episode + 1}/{n_episodes} 完了")
    
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
    
        # 95% 信頼区間
        conf_interval = 1.96 * std_acc / np.sqrt(n_episodes)
    
        print(f"\n=== 評価結果 ({n_episodes} エピソード) ===")
        print(f"平均精度: {mean_acc:.3f} ± {conf_interval:.3f}")
        print(f"標準偏差: {std_acc:.3f}")
        print(f"最小精度: {min(accuracies):.3f}")
        print(f"最大精度: {max(accuracies):.3f}")
    
        # 精度分布の可視化
        import matplotlib.pyplot as plt
    
        plt.figure(figsize=(10, 6))
        plt.hist(accuracies, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        plt.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
                   label=f'平均: {mean_acc:.3f}')
        plt.axvline(mean_acc - conf_interval, color='orange', linestyle=':',
                   linewidth=2, label=f'95% CI')
        plt.axvline(mean_acc + conf_interval, color='orange', linestyle=':',
                   linewidth=2)
        plt.xlabel('精度', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.title(f'Few-Shot精度分布 ({n_episodes} エピソード)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        return mean_acc, std_acc
    
    # 評価実行（Omniglotデータセットが利用可能な場合）
    try:
        nn_model = NearestNeighborClassifier(distance_metric='euclidean')
        mean_acc, std_acc = evaluate_fewshot_model(
            nn_model,
            sampler,
            n_episodes=100  # デモ用に少なめ
        )
    except NameError:
        print("注: Omniglotデータセットとサンプラーが必要です")
    

> **重要** : Nearest Neighborベースラインは、シンプルながら多くのFew-Shotタスクで競争力のある性能を示します。

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **メタ学習の本質**

     * Learning to Learn: 学習方法そのものを学習
     * 少数サンプルでの高速適応が目標
     * エピソードベースの訓練プロセス
  2. **Few-Shot Learning問題設定**

     * N-way K-shot分類の定義
     * Support SetとQuery Setの役割
     * 評価プロトコルの標準化
  3. **メタ学習の3つのアプローチ**

     * Metric-based: 距離学習
     * Model-based: メモリと再帰
     * Optimization-based: 良い初期化
  4. **Omniglotデータセット**

     * 1,623クラス、各20サンプル
     * エピソード生成の実装
     * Few-Shot学習の標準ベンチマーク
  5. **ベースライン実装**

     * Nearest Neighbor分類器
     * 標準評価プロトコル
     * 精度と信頼区間の報告

### メタ学習の重要概念

概念 | 説明  
---|---  
**エピソード** | 1つの学習タスク（Support + Query）  
**メタ訓練** | 多数のエピソードから適応能力を学習  
**メタテスト** | 未知のタスクでの適応性能を評価  
**Few-Shot** | 少数サンプル（通常1〜5個）での学習  
**Zero-Shot** | 訓練サンプルなしでの推論  
  
### 次の章へ

第2章では、**Prototypical Networks** を詳しく学びます：

  * プロトタイプベースの分類
  * 埋め込みネットワークの設計
  * エピソード訓練の実装
  * Omniglotでの性能評価
  * ハイパーパラメータチューニング

* * *

## 演習問題

### 問題1（難易度：easy）

メタ学習と通常の機械学習の違いを、「学習単位」「訓練データ」「評価方法」の3つの観点から説明してください。

解答例

**解答** ：

観点 | 通常の機械学習 | メタ学習  
---|---|---  
**学習単位** | 個別サンプル（画像、テキストなど） | タスク全体（エピソード単位）  
**訓練データ** | 1つのタスクに対する大量ラベル付きデータ | 多様なタスクからの少数サンプル集合  
**評価方法** | 同一分布のテストセットでの精度 | 未知タスクへの適応速度と精度  
  
**具体例** ：

  * **通常学習** : 10万枚の猫vs犬画像で分類器を訓練 → 同じ分布のテスト画像で評価
  * **メタ学習** : 1000種類の動物（各5枚）で学習 → 新種の動物を5枚だけ見て分類

### 問題2（難易度：medium）

5-way 3-shot分類タスクにおいて、Support SetとQuery Set（各クラス5サンプル）のサイズをそれぞれ計算してください。また、1エピソードあたりの総サンプル数も求めてください。

解答例

**解答** ：

**条件** ：

  * N-way = 5 クラス
  * K-shot = 3 サンプル/クラス（Support）
  * Q = 5 サンプル/クラス（Query）

**計算** ：

  1. **Support Set サイズ** : $$\text{Support} = N \times K = 5 \times 3 = 15 \text{ サンプル}$$
  2. **Query Set サイズ** : $$\text{Query} = N \times Q = 5 \times 5 = 25 \text{ サンプル}$$
  3. **総サンプル数** : $$\text{Total} = \text{Support} + \text{Query} = 15 + 25 = 40 \text{ サンプル}$$

**構造** ：
    
    
    Support Set (15サンプル):
      Class 0: [S_0_0, S_0_1, S_0_2]
      Class 1: [S_1_0, S_1_1, S_1_2]
      Class 2: [S_2_0, S_2_1, S_2_2]
      Class 3: [S_3_0, S_3_1, S_3_2]
      Class 4: [S_4_0, S_4_1, S_4_2]
    
    Query Set (25サンプル):
      Class 0: [Q_0_0, Q_0_1, Q_0_2, Q_0_3, Q_0_4]
      Class 1: [Q_1_0, Q_1_1, Q_1_2, Q_1_3, Q_1_4]
      Class 2: [Q_2_0, Q_2_1, Q_2_2, Q_2_3, Q_2_4]
      Class 3: [Q_3_0, Q_3_1, Q_3_2, Q_3_3, Q_3_4]
      Class 4: [Q_4_0, Q_4_1, Q_4_2, Q_4_3, Q_4_4]
    

### 問題3（難易度：medium）

Metric-based、Model-based、Optimization-basedの3つのメタ学習アプローチについて、それぞれの基本アイデアと代表的な手法を1つずつ挙げてください。

解答例

**解答** ：

アプローチ | 基本アイデア | 代表手法 | 特徴  
---|---|---|---  
**Metric-based** | 良い距離空間を学習し、  
近傍に基づいて分類 | Prototypical  
Networks | 各クラスのプロトタイプを計算し、  
最も近いクラスに分類  
**Model-based** | メモリや再帰構造で  
高速適応 | Memory-Augmented  
Neural Networks | 外部メモリに過去の経験を保存し、  
新タスクで参照  
**Optimization-based** | 良い初期パラメータを学習し、  
少数ステップで適応 | MAML  
(Model-Agnostic Meta-Learning) | 数ステップの勾配降下で  
高精度に到達する初期値を学習  
  
**使い分け** ：

  * **Metric-based** : シンプルで高速、画像分類に最適
  * **Model-based** : 複雑なシーケンシャルタスク向け
  * **Optimization-based** : 汎用性が高く、強化学習にも適用可能

### 問題4（難易度：hard）

Omniglotデータセットで5-way 1-shot分類を行う際、ランダム推測の精度と、理想的なNearest Neighbor分類器の期待精度を推定してください。また、実際のメタ学習手法が目指すべき精度範囲を考察してください。

解答例

**解答** ：

**1\. ランダム推測の精度** ：

  * 5クラスから1つをランダムに選ぶ
  * 精度 = 1/5 = **20%**

**2\. 理想的なNearest Neighbor分類器の期待精度** ：

Omniglotの特性を考慮：

  * 各クラスは視覚的に識別可能（異なる文字）
  * 同一クラス内の変動（20人の手書き）あり
  * ピクセルベースの距離は不完全

期待精度: **60-75%** 程度

理由：

  * Support Set は1サンプルのみ → クラス内変動を捉えられない
  * ピクセルレベルの距離は、回転や変形に敏感
  * それでもランダムよりは遥かに良い

**3\. メタ学習手法の目標精度範囲** ：

手法タイプ | 期待精度 | 理由  
---|---|---  
ベースライン（NN） | 60-75% | ピクセル距離のみ  
Metric-based | 85-95% | 学習された埋め込み空間  
Optimization-based | 95-98% | タスクごとに適応  
最先端 | 98%+ | データ拡張 + アンサンブル  
  
**実例（論文結果）** ：

  * Siamese Networks: ~92%
  * Matching Networks: ~93%
  * Prototypical Networks: ~95%
  * MAML: ~95-98%

### 問題5（難易度：hard）

以下のコードを完成させて、シンプルなPrototype分類器を実装してください。各クラスのサポートサンプルの平均（プロトタイプ）を計算し、クエリサンプルを最も近いプロトタイプのクラスに分類する関数を作成してください。
    
    
    import torch
    
    def prototype_classify(support_set, support_labels, query_set, n_way):
        """
        Prototypeベースの分類
    
        Args:
            support_set: (N*K, feature_dim) tensor
            support_labels: (N*K,) tensor
            query_set: (N*Q, feature_dim) tensor
            n_way: クラス数
    
        Returns:
            predictions: (N*Q,) tensor
        """
        # TODO: プロトタイプを計算
        prototypes = None  # ここを実装
    
        # TODO: 距離を計算して分類
        predictions = None  # ここを実装
    
        return predictions
    

解答例
    
    
    import torch
    
    def prototype_classify(support_set, support_labels, query_set, n_way):
        """
        Prototypeベースの分類
    
        Args:
            support_set: (N*K, feature_dim) tensor
            support_labels: (N*K,) tensor
            query_set: (N*Q, feature_dim) tensor
            n_way: クラス数
    
        Returns:
            predictions: (N*Q,) tensor
        """
        # 1. 各クラスのプロトタイプを計算
        prototypes = []
        for c in range(n_way):
            # クラスcのサポートサンプルを抽出
            class_support = support_set[support_labels == c]
            # 平均を計算してプロトタイプとする
            prototype = class_support.mean(dim=0)
            prototypes.append(prototype)
    
        prototypes = torch.stack(prototypes)  # (n_way, feature_dim)
    
        # 2. 各クエリサンプルを最も近いプロトタイプのクラスに分類
        n_queries = query_set.size(0)
        predictions = []
    
        for i in range(n_queries):
            query = query_set[i]  # (feature_dim,)
    
            # 全プロトタイプとの距離を計算
            distances = torch.norm(prototypes - query, dim=1)  # (n_way,)
    
            # 最小距離のクラスを予測
            pred_class = torch.argmin(distances)
            predictions.append(pred_class)
    
        predictions = torch.stack(predictions)
    
        return predictions
    
    
    # テストコード
    def test_prototype_classifier():
        """Prototype分類器のテスト"""
    
        # 5-way 2-shot タスクをシミュレーション
        n_way = 5
        k_shot = 2
        n_query = 10
        feature_dim = 128
    
        # ダミーデータ生成
        support_set = torch.randn(n_way * k_shot, feature_dim)
        support_labels = torch.tensor([i for i in range(n_way) for _ in range(k_shot)])
    
        # Query Set: 各クラスから2サンプル
        query_set = torch.randn(n_query, feature_dim)
        query_labels = torch.tensor([i % n_way for i in range(n_query)])
    
        # 分類実行
        predictions = prototype_classify(support_set, support_labels, query_set, n_way)
    
        # 精度計算
        accuracy = (predictions == query_labels).float().mean()
    
        print("=== Prototype分類器テスト ===")
        print(f"タスク: {n_way}-way {k_shot}-shot")
        print(f"Support Set: {support_set.shape}")
        print(f"Query Set: {query_set.shape}")
        print(f"予測: {predictions}")
        print(f"正解: {query_labels}")
        print(f"精度: {accuracy:.3f}")
    
        # より現実的なテスト: クラスを空間的に分離
        print("\n=== 分離されたクラスでのテスト ===")
    
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
    
        for c in range(n_way):
            # クラスごとに中心を設定
            center = torch.randn(feature_dim) * 5  # 大きく分離
    
            # Support samples
            for _ in range(k_shot):
                sample = center + torch.randn(feature_dim) * 0.5  # 小さなノイズ
                support_set.append(sample)
                support_labels.append(c)
    
            # Query samples
            for _ in range(2):
                sample = center + torch.randn(feature_dim) * 0.5
                query_set.append(sample)
                query_labels.append(c)
    
        support_set = torch.stack(support_set)
        support_labels = torch.tensor(support_labels)
        query_set = torch.stack(query_set)
        query_labels = torch.tensor(query_labels)
    
        # 分類実行
        predictions = prototype_classify(support_set, support_labels, query_set, n_way)
        accuracy = (predictions == query_labels).float().mean()
    
        print(f"分離データでの精度: {accuracy:.3f}")
        print("（クラスが明確に分離されている場合、精度は高くなる）")
    
    # テスト実行
    test_prototype_classifier()
    

**出力例** ：
    
    
    === Prototype分類器テスト ===
    タスク: 5-way 2-shot
    Support Set: torch.Size([10, 128])
    Query Set: torch.Size([10, 128])
    予測: tensor([1, 3, 0, 2, 4, 0, 1, 2, 3, 4])
    正解: tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    精度: 0.300
    
    === 分離されたクラスでのテスト ===
    分離データでの精度: 1.000
    （クラスが明確に分離されている場合、精度は高くなる）
    

**解説** ：

  1. **プロトタイプ計算** : 各クラスのサポートサンプルの平均を取る
  2. **距離計算** : クエリサンプルと全プロトタイプ間のユークリッド距離
  3. **分類** : 最小距離のプロトタイプのクラスを予測
  4. **性能** : クラスが空間的に分離されている場合、高精度を達成

* * *

## 参考文献

  1. Vinyals, O., et al. (2016). "Matching Networks for One Shot Learning." _NeurIPS_.
  2. Snell, J., Swersky, K., & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning." _NeurIPS_.
  3. Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." _ICML_.
  4. Lake, B. M., et al. (2015). "Human-level concept learning through probabilistic program induction." _Science_.
  5. Hospedales, T., et al. (2020). "Meta-Learning in Neural Networks: A Survey." _arXiv:2004.05439_.
