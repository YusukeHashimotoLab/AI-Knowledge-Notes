---
title: 第3章：Neural Architecture Search
chapter_title: 第3章：Neural Architecture Search
subtitle: ニューラルネットワークの自動設計 - DARTSとAutoKerasによる最適アーキテクチャの探索
reading_time: 35-40分
difficulty: 中級-上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Neural Architecture Search（NAS）の探索空間設計を理解する
  * ✅ NASの主要な探索戦略（強化学習、進化アルゴリズム、勾配ベース）を理解する
  * ✅ AutoKerasを使った自動モデル探索を実装できる
  * ✅ DARTS（微分可能アーキテクチャ探索）の原理と実装を理解する
  * ✅ NASの効率化手法（Weight Sharing、Proxy Tasks）を理解する
  * ✅ 実データでAutoKerasとDARTSを活用できる

* * *

## 3.1 NASの探索空間

### Neural Architecture Searchとは

**Neural Architecture Search（NAS）** は、ニューラルネットワークのアーキテクチャを自動的に設計する技術です。

> 「手動設計 vs 自動設計」- NASは人間の専門知識を超えるアーキテクチャを発見できます。

### NASの3要素

要素 | 説明 | 例  
---|---|---  
**探索空間** | 探索可能なアーキテクチャの集合 | Cell-based、Macro、Micro  
**探索戦略** | アーキテクチャの探索方法 | 強化学習、進化、勾配ベース  
**性能評価** | アーキテクチャの良し悪しの判定 | 精度、FLOPs、レイテンシ  
  
### Cell-based Search Space

**Cell-based探索空間** では、繰り返し使用される「Cell」を探索します。
    
    
    ```mermaid
    graph TD
        A[入力画像] --> B[Stem Convolution]
        B --> C[Normal Cell 1]
        C --> D[Normal Cell 2]
        D --> E[Reduction Cell]
        E --> F[Normal Cell 3]
        F --> G[Normal Cell 4]
        G --> H[Reduction Cell]
        H --> I[Normal Cell 5]
        I --> J[Global Pool]
        J --> K[Softmax]
    
        style C fill:#e3f2fd
        style D fill:#e3f2fd
        style E fill:#ffebee
        style F fill:#e3f2fd
        style G fill:#e3f2fd
        style H fill:#ffebee
        style I fill:#e3f2fd
    ```

#### Cellの種類

Cell種類 | 役割 | 空間解像度  
---|---|---  
**Normal Cell** | 特徴抽出 | 維持  
**Reduction Cell** | ダウンサンプリング | 1/2に削減  
  
### Macro vs Micro Architecture

アーキテクチャ | 探索対象 | 利点 | 欠点  
---|---|---|---  
**Macro** | 全体構造（層数、結合） | 柔軟性が高い | 探索空間が巨大  
**Micro** | Cell内部の構造 | 効率的、転移可能 | 制約が多い  
  
### 探索空間のサイズ

Cell-based探索空間のサイズは膨大です：

$$ \text{Search Space Size} \approx O^{E} $$

  * $O$: 操作の種類（例: 8種類）
  * $E$: エッジの数（例: 14個）
  * 例: $8^{14} \approx 4.4 \times 10^{12}$ 通り

### 探索空間設計の実例
    
    
    import numpy as np
    
    # NAS探索空間の定義
    class SearchSpace:
        def __init__(self):
            # 利用可能な操作
            self.operations = [
                'conv_3x3',
                'conv_5x5',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'max_pool_3x3',
                'avg_pool_3x3',
                'identity',
                'zero'
            ]
    
            # Cellの構造パラメータ
            self.num_nodes = 4  # Cell内のノード数
            self.num_edges_per_node = 2  # 各ノードへの入力エッジ数
    
        def calculate_space_size(self):
            """探索空間のサイズを計算"""
            num_ops = len(self.operations)
    
            # 各ノードごとに選択肢を計算
            total_choices = 1
            for node_id in range(2, self.num_nodes + 2):
                # エッジ元の選択（前のノードから選ぶ）
                edge_choices = node_id
                # 操作の選択
                op_choices = num_ops
                # このノードの選択肢
                node_choices = (edge_choices * op_choices) ** self.num_edges_per_node
                total_choices *= node_choices
    
            return total_choices
    
        def sample_architecture(self):
            """ランダムにアーキテクチャをサンプリング"""
            architecture = []
    
            for node_id in range(2, self.num_nodes + 2):
                # このノードへの入力を選択
                node_config = []
                for _ in range(self.num_edges_per_node):
                    # 入力元ノード
                    input_node = np.random.randint(0, node_id)
                    # 操作
                    operation = np.random.choice(self.operations)
                    node_config.append((input_node, operation))
    
                architecture.append(node_config)
    
            return architecture
    
    # 探索空間のサイズを計算
    search_space = SearchSpace()
    space_size = search_space.calculate_space_size()
    
    print("=== NAS探索空間の分析 ===")
    print(f"操作の種類: {len(search_space.operations)}")
    print(f"Cell内のノード数: {search_space.num_nodes}")
    print(f"探索空間のサイズ: {space_size:,}")
    print(f"科学的記法: {space_size:.2e}")
    
    # サンプルアーキテクチャ
    sample = search_space.sample_architecture()
    print(f"\n=== サンプルアーキテクチャ ===")
    for i, node in enumerate(sample, start=2):
        print(f"ノード {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  入力{j}: ノード{input_node} → {op}")
    

**出力** ：
    
    
    === NAS探索空間の分析 ===
    操作の種類: 8
    Cell内のノード数: 4
    探索空間のサイズ: 17,179,869,184
    科学的記法: 1.72e+10
    
    === サンプルアーキテクチャ ===
    ノード 2:
      入力0: ノード0 → sep_conv_3x3
      入力1: ノード1 → max_pool_3x3
    ノード 3:
      入力0: ノード2 → conv_5x5
      入力1: ノード0 → identity
    ノード 4:
      入力0: ノード3 → avg_pool_3x3
      入力1: ノード1 → sep_conv_5x5
    ノード 5:
      入力0: ノード2 → conv_3x3
      入力1: ノード4 → zero
    

* * *

## 3.2 NASの探索戦略

### 主要な探索戦略の比較

探索戦略 | 原理 | 計算コスト | 代表手法  
---|---|---|---  
**強化学習** | コントローラがアーキテクチャを生成 | 非常に高い | NASNet  
**進化アルゴリズム** | 突然変異と選択を繰り返す | 高い | AmoebaNet  
**勾配ベース** | 連続緩和で微分可能に | 低い | DARTS  
**One-shot** | スーパーネットを一度学習 | 中程度 | ENAS  
  
### 1\. 強化学習ベース（NASNet）

NASNetは、RNNコントローラがアーキテクチャを生成し、精度を報酬として学習します。
    
    
    ```mermaid
    graph LR
        A[RNNコントローラ] -->|アーキテクチャ生成| B[子ネットワーク]
        B -->|学習・評価| C[検証精度]
        C -->|報酬| A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### 強化学習NASの疑似コード
    
    
    # NASNetスタイルの強化学習探索（概念的実装）
    import numpy as np
    
    class RLController:
        """強化学習ベースのNASコントローラ"""
    
        def __init__(self, search_space):
            self.search_space = search_space
            self.history = []
    
        def sample_architecture(self, epsilon=0.1):
            """ε-greedy戦略でアーキテクチャをサンプリング"""
            if np.random.random() < epsilon:
                # 探索: ランダムサンプリング
                return self.search_space.sample_architecture()
            else:
                # 活用: 過去の良いアーキテクチャから変異
                if self.history:
                    best_arch = max(self.history, key=lambda x: x['reward'])
                    return self.mutate_architecture(best_arch['architecture'])
                else:
                    return self.search_space.sample_architecture()
    
        def mutate_architecture(self, architecture):
            """アーキテクチャに小さな変異を加える"""
            mutated = [node[:] for node in architecture]
    
            # ランダムに1つのノードを変異
            node_idx = np.random.randint(len(mutated))
            edge_idx = np.random.randint(len(mutated[node_idx]))
    
            input_node, _ = mutated[node_idx][edge_idx]
            new_op = np.random.choice(self.search_space.operations)
            mutated[node_idx][edge_idx] = (input_node, new_op)
    
            return mutated
    
        def update(self, architecture, reward):
            """報酬を受け取って履歴を更新"""
            self.history.append({
                'architecture': architecture,
                'reward': reward
            })
    
    # シミュレーション
    search_space = SearchSpace()
    controller = RLController(search_space)
    
    print("=== 強化学習NASのシミュレーション ===")
    for iteration in range(10):
        # アーキテクチャのサンプリング
        arch = controller.sample_architecture(epsilon=0.3)
    
        # 報酬をシミュレート（実際には学習して精度を取得）
        # ここでは操作の多様性に基づくダミー報酬
        ops_used = set()
        for node in arch:
            for _, op in node:
                ops_used.add(op)
        reward = len(ops_used) / len(search_space.operations) + np.random.normal(0, 0.1)
    
        # コントローラを更新
        controller.update(arch, reward)
    
        print(f"Iteration {iteration + 1}: 報酬 = {reward:.3f}")
    
    # 最良のアーキテクチャを表示
    best = max(controller.history, key=lambda x: x['reward'])
    print(f"\n=== 最良のアーキテクチャ（報酬: {best['reward']:.3f}）===")
    for i, node in enumerate(best['architecture'], start=2):
        print(f"ノード {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  入力{j}: ノード{input_node} → {op}")
    

### 2\. 進化アルゴリズム

進化アルゴリズムは、生物進化を模倣してアーキテクチャを最適化します。
    
    
    # 進化アルゴリズムによるNAS（簡易版）
    import random
    import copy
    
    class EvolutionaryNAS:
        """進化アルゴリズムベースのNAS"""
    
        def __init__(self, search_space, population_size=20, num_generations=10):
            self.search_space = search_space
            self.population_size = population_size
            self.num_generations = num_generations
    
        def initialize_population(self):
            """初期集団を生成"""
            return [self.search_space.sample_architecture()
                    for _ in range(self.population_size)]
    
        def evaluate_fitness(self, architecture):
            """適応度を評価（ダミー実装）"""
            # 実際にはネットワークを学習して精度を測定
            # ここでは操作の多様性をスコアとする
            ops_used = set()
            for node in architecture:
                for _, op in node:
                    ops_used.add(op)
            return len(ops_used) + random.gauss(0, 1)
    
        def select_parents(self, population, fitness_scores, k=2):
            """トーナメント選択"""
            selected = []
            for _ in range(2):
                candidates_idx = random.sample(range(len(population)), k)
                best_idx = max(candidates_idx, key=lambda i: fitness_scores[i])
                selected.append(copy.deepcopy(population[best_idx]))
            return selected
    
        def crossover(self, parent1, parent2):
            """交叉（一点交叉）"""
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
    
        def mutate(self, architecture, mutation_rate=0.1):
            """突然変異"""
            mutated = copy.deepcopy(architecture)
    
            for node_idx in range(len(mutated)):
                for edge_idx in range(len(mutated[node_idx])):
                    if random.random() < mutation_rate:
                        input_node, _ = mutated[node_idx][edge_idx]
                        new_op = random.choice(self.search_space.operations)
                        mutated[node_idx][edge_idx] = (input_node, new_op)
    
            return mutated
    
        def run(self):
            """進化アルゴリズムを実行"""
            # 初期集団
            population = self.initialize_population()
    
            best_history = []
    
            for generation in range(self.num_generations):
                # 適応度評価
                fitness_scores = [self.evaluate_fitness(arch) for arch in population]
    
                # 統計
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                best_history.append(best_fitness)
    
                print(f"世代 {generation + 1}: 最良={best_fitness:.3f}, 平均={avg_fitness:.3f}")
    
                # 新世代の生成
                new_population = []
    
                # エリート保存
                elite_idx = fitness_scores.index(max(fitness_scores))
                new_population.append(copy.deepcopy(population[elite_idx]))
    
                # 選択、交叉、突然変異
                while len(new_population) < self.population_size:
                    parents = self.select_parents(population, fitness_scores)
                    offspring1, offspring2 = self.crossover(parents[0], parents[1])
                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)
    
                    new_population.extend([offspring1, offspring2])
    
                population = new_population[:self.population_size]
    
            # 最良の個体を返す
            fitness_scores = [self.evaluate_fitness(arch) for arch in population]
            best_idx = fitness_scores.index(max(fitness_scores))
    
            return population[best_idx], best_history
    
    # 実行
    search_space = SearchSpace()
    evo_nas = EvolutionaryNAS(search_space, population_size=20, num_generations=10)
    
    print("=== 進化アルゴリズムによるNAS ===")
    best_arch, history = evo_nas.run()
    
    print(f"\n=== 最良のアーキテクチャ ===")
    for i, node in enumerate(best_arch, start=2):
        print(f"ノード {i}:")
        for j, (input_node, op) in enumerate(node):
            print(f"  入力{j}: ノード{input_node} → {op}")
    

### 3\. 勾配ベース（DARTS概要）

DARTSは探索空間を連続緩和し、勾配降下法で最適化します（詳細は3.4節）。

> **重要** : 勾配ベース手法は、強化学習や進化アルゴリズムと比べて1000倍以上高速です。

* * *

## 3.3 AutoKeras

### AutoKerasとは

**AutoKeras** は、KerasベースのAutoMLライブラリで、NASを簡単に利用できます。

### AutoKerasのインストール
    
    
    pip install autokeras
    

### AutoKerasの基本的な使い方
    
    
    # AutoKerasの基本例：画像分類
    import numpy as np
    import autokeras as ak
    from tensorflow.keras.datasets import mnist
    
    # データの準備
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 正規化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 訓練データを削減（デモ用）
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    print("=== AutoKerasによる画像分類 ===")
    print(f"訓練データ: {x_train.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    # AutoKerasのImageClassifier
    clf = ak.ImageClassifier(
        max_trials=5,  # 試行するモデル数
        overwrite=True,
        directory='autokeras_results',
        project_name='mnist_classification'
    )
    
    # モデルの探索と学習
    print("\n探索を開始...")
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=1
    )
    
    # 評価
    print("\n=== モデルの評価 ===")
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"テスト精度: {test_acc:.4f}")
    print(f"テスト損失: {test_loss:.4f}")
    
    # 最良モデルの取得
    best_model = clf.export_model()
    print("\n=== 最良モデルの構造 ===")
    best_model.summary()
    

### AutoKerasの様々なタスク

#### 1\. 構造化データの分類
    
    
    # AutoKerasで構造化データを扱う
    import numpy as np
    import pandas as pd
    import autokeras as ak
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # データの準備
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== 構造化データ分類 ===")
    print(f"特徴量: {X.shape[1]}個")
    print(f"訓練サンプル: {len(X_train)}")
    
    # AutoKerasのStructuredDataClassifier
    clf = ak.StructuredDataClassifier(
        max_trials=3,
        overwrite=True,
        directory='autokeras_structured',
        project_name='breast_cancer'
    )
    
    # 探索と学習
    clf.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        verbose=0
    )
    
    # 評価
    test_loss, test_acc = clf.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== 評価結果 ===")
    print(f"テスト精度: {test_acc:.4f}")
    
    # 予測
    predictions = clf.predict(X_test[:5])
    print(f"\n=== サンプル予測 ===")
    for i, pred in enumerate(predictions[:5]):
        true_label = y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]
        print(f"サンプル {i+1}: 予測={pred[0][0]:.3f}, 真値={true_label}")
    

#### 2\. テキスト分類
    
    
    # AutoKerasでテキスト分類
    import numpy as np
    import autokeras as ak
    from tensorflow.keras.datasets import imdb
    
    # IMDBデータセット（映画レビューの感情分析）
    max_features = 10000
    maxlen = 200
    
    print("=== テキスト分類（IMDB）===")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # データを削減（デモ用）
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]
    
    # パディング
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    print(f"訓練データ: {x_train.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    # AutoKerasのTextClassifier
    clf = ak.TextClassifier(
        max_trials=3,
        overwrite=True,
        directory='autokeras_text',
        project_name='imdb_sentiment'
    )
    
    # 探索と学習
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=0
    )
    
    # 評価
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"\n=== 評価結果 ===")
    print(f"テスト精度: {test_acc:.4f}")
    

### AutoKerasのカスタム探索空間
    
    
    # AutoKerasで探索空間をカスタマイズ
    import autokeras as ak
    from tensorflow.keras.datasets import mnist
    
    # データ準備
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:5000].astype('float32') / 255.0
    y_train = y_train[:5000]
    x_test = x_test[:1000].astype('float32') / 255.0
    y_test = y_test[:1000]
    
    print("=== カスタム探索空間 ===")
    
    # 入力ノード
    input_node = ak.ImageInput()
    
    # 正規化ブロック
    output = ak.Normalization()(input_node)
    
    # ConvBlockの探索空間をカスタマイズ
    output = ak.ConvBlock(
        num_blocks=2,  # 畳み込みブロックの数
        num_layers=2,  # ブロック内の層数
        max_pooling=True,
        dropout=0.25
    )(output)
    
    # 分類ヘッド
    output = ak.ClassificationHead(
        num_classes=10,
        dropout=0.5
    )(output)
    
    # モデル構築
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output,
        max_trials=3,
        overwrite=True,
        directory='autokeras_custom',
        project_name='mnist_custom'
    )
    
    # 学習
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=3,
        verbose=0
    )
    
    # 評価
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"\nテスト精度: {test_acc:.4f}")
    
    # 最良モデルの取得
    best_model = clf.export_model()
    print("\n=== 発見されたアーキテクチャ ===")
    best_model.summary()
    

* * *

## 3.4 DARTS（Differentiable Architecture Search）

### DARTSの原理

**DARTS** は、離散的な探索空間を連続緩和することで、勾配降下法を適用可能にします。

### 連続緩和（Continuous Relaxation）

各エッジの操作を、全操作の重み付き和として表現します：

$$ \bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x) $$

  * $\mathcal{O}$: 操作の集合
  * $\alpha_o^{(i,j)}$: エッジ$(i,j)$での操作$o$の重み（アーキテクチャパラメータ）
  * ソフトマックスで正規化し、微分可能に

### Bi-level Optimization

DARTSは2つのパラメータを交互に最適化します：

パラメータ | 説明 | 最適化  
---|---|---  
**重み $w$** | ネットワークの重み | 訓練データで最小化  
**アーキテクチャ $\alpha$** | 操作の重み | 検証データで最小化  
  
最適化問題：

$$ \begin{aligned} \min_{\alpha} \quad & \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \\\ \text{s.t.} \quad & w^*(\alpha) = \arg\min_{w} \mathcal{L}_{\text{train}}(w, \alpha) \end{aligned} $$

### DARTSの実装（簡易版）
    
    
    # DARTSの概念的実装（簡易版）
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MixedOp(nn.Module):
        """複数の操作の重み付き和"""
    
        def __init__(self, C, stride):
            super(MixedOp, self).__init__()
            self._ops = nn.ModuleList()
    
            # 利用可能な操作
            self.operations = [
                ('sep_conv_3x3', lambda C, stride: SepConv(C, C, 3, stride, 1)),
                ('sep_conv_5x5', lambda C, stride: SepConv(C, C, 5, stride, 2)),
                ('avg_pool_3x3', lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1)),
                ('max_pool_3x3', lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1)),
                ('skip_connect', lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C)),
            ]
    
            for name, op in self.operations:
                self._ops.append(op(C, stride))
    
        def forward(self, x, weights):
            """重み付き和を計算"""
            return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    class Cell(nn.Module):
        """DARTSのCell"""
    
        def __init__(self, num_nodes, C_prev, C, reduction):
            super(Cell, self).__init__()
            self.num_nodes = num_nodes
    
            # 各エッジに対する操作
            self._ops = nn.ModuleList()
            for i in range(num_nodes):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
    
        def forward(self, s0, s1, weights):
            """順伝播"""
            states = [s0, s1]
            offset = 0
    
            for i in range(self.num_nodes):
                s = sum(self._ops[offset + j](h, weights[offset + j])
                       for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
    
            return torch.cat(states[-self.num_nodes:], dim=1)
    
    class DARTSNetwork(nn.Module):
        """DARTS探索ネットワーク"""
    
        def __init__(self, C=16, num_cells=8, num_nodes=4, num_classes=10):
            super(DARTSNetwork, self).__init__()
            self.num_cells = num_cells
            self.num_nodes = num_nodes
    
            # アーキテクチャパラメータ（α）
            num_ops = 5  # 操作の種類
            num_edges = sum(2 + i for i in range(num_nodes))
            self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops))
            self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops))
    
            # ネットワークの重み（w）
            self.stem = nn.Sequential(
                nn.Conv2d(3, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )
    
            # Cellsの構築は簡略化
            self.cells = nn.ModuleList()
            # ... (実際の実装では複数のCellを追加)
    
            self.classifier = nn.Linear(C, num_classes)
    
        def arch_parameters(self):
            """アーキテクチャパラメータを返す"""
            return [self.alphas_normal, self.alphas_reduce]
    
        def weights_parameters(self):
            """ネットワークの重みを返す"""
            return [p for n, p in self.named_parameters()
                    if 'alpha' not in n]
    
    # DARTSの使用例
    print("=== DARTS概念モデル ===")
    model = DARTSNetwork(C=16, num_cells=8, num_nodes=4, num_classes=10)
    
    print(f"アーキテクチャパラメータ数: {sum(p.numel() for p in model.arch_parameters())}")
    print(f"ネットワーク重みパラメータ数: {sum(p.numel() for p in model.weights_parameters())}")
    
    # アーキテクチャパラメータの形状
    print(f"\nNormal cell α: {model.alphas_normal.shape}")
    print(f"Reduction cell α: {model.alphas_reduce.shape}")
    
    # ソフトマックスで正規化
    weights_normal = F.softmax(model.alphas_normal, dim=-1)
    print(f"\n正規化後の重み（Normal cell, 最初のエッジ）:")
    print(weights_normal[0].detach().numpy())
    

### DARTSの学習アルゴリズム
    
    
    # DARTSの学習手順（疑似コード）
    import torch
    import torch.optim as optim
    
    class DARTSTrainer:
        """DARTSの学習クラス"""
    
        def __init__(self, model, train_loader, val_loader):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
    
            # 2つのオプティマイザ
            self.optimizer_w = optim.SGD(
                model.weights_parameters(),
                lr=0.025,
                momentum=0.9,
                weight_decay=3e-4
            )
    
            self.optimizer_alpha = optim.Adam(
                model.arch_parameters(),
                lr=3e-4,
                betas=(0.5, 0.999),
                weight_decay=1e-3
            )
    
            self.criterion = nn.CrossEntropyLoss()
    
        def train_step(self, train_data, val_data):
            """1ステップの学習"""
            # 1. アーキテクチャパラメータ（α）の更新
            self.model.train()
            x_val, y_val = val_data
    
            self.optimizer_alpha.zero_grad()
            logits = self.model(x_val)
            loss_alpha = self.criterion(logits, y_val)
            loss_alpha.backward()
            self.optimizer_alpha.step()
    
            # 2. ネットワーク重み（w）の更新
            x_train, y_train = train_data
    
            self.optimizer_w.zero_grad()
            logits = self.model(x_train)
            loss_w = self.criterion(logits, y_train)
            loss_w.backward()
            self.optimizer_w.step()
    
            return loss_w.item(), loss_alpha.item()
    
        def derive_architecture(self):
            """最終的なアーキテクチャを導出"""
            # 各エッジで最も重みの大きい操作を選択
            def parse_alpha(alpha):
                gene = []
                n = 2
                start = 0
                for i in range(self.model.num_nodes):
                    end = start + n
                    W = alpha[start:end].copy()
    
                    # 各エッジで最良の操作を2つ選択
                    edges = sorted(range(W.shape[0]),
                                  key=lambda x: -max(W[x]))[:2]
    
                    for j in edges:
                        k_best = W[j].argmax()
                        gene.append((j, k_best))
    
                    start = end
                    n += 1
    
                return gene
    
            # ソフトマックスで正規化
            weights_normal = F.softmax(self.model.alphas_normal, dim=-1)
            weights_reduce = F.softmax(self.model.alphas_reduce, dim=-1)
    
            gene_normal = parse_alpha(weights_normal.data.cpu().numpy())
            gene_reduce = parse_alpha(weights_reduce.data.cpu().numpy())
    
            return gene_normal, gene_reduce
    
    # シミュレーション例
    print("=== DARTS学習手順 ===")
    print("1. モデルの初期化")
    print("2. 各エポックで:")
    print("   a. 検証データでαを更新（アーキテクチャ最適化）")
    print("   b. 訓練データでwを更新（重み最適化）")
    print("3. 学習終了後、各エッジで最も重みの大きい操作を選択")
    print("4. 選択された操作でネットワークを再構築し、最終学習")
    

### DARTSの実用例（PyTorch）
    
    
    # 実際のDARTS実装を使った例（pt-dartsライブラリ使用）
    # 注: pt-dartsは外部ライブラリ（pip install pt-darts）
    
    # 以下は概念的なコード例
    """
    import torch
    from darts import DARTS
    from darts.api import spaces
    from darts.trainer import DARTSTrainer
    
    # 探索空間の定義
    search_space = spaces.get_search_space('darts', 'cifar10')
    
    # DARTSモデルの構築
    model = DARTS(
        C=16,
        num_classes=10,
        layers=8,
        criterion=nn.CrossEntropyLoss(),
        steps=4,
        multiplier=4,
        stem_multiplier=3
    )
    
    # トレーナーの初期化
    trainer = DARTSTrainer(
        model,
        optimizer_config={
            'w_lr': 0.025,
            'w_momentum': 0.9,
            'w_weight_decay': 3e-4,
            'alpha_lr': 3e-4,
            'alpha_weight_decay': 1e-3
        }
    )
    
    # 探索の実行
    trainer.search(
        train_loader,
        val_loader,
        epochs=50
    )
    
    # 最良アーキテクチャの取得
    best_architecture = model.genotype()
    print(f"発見されたアーキテクチャ: {best_architecture}")
    """
    
    print("=== DARTSの実用的な使い方 ===")
    print("1. pt-dartsなどのライブラリをインストール")
    print("2. 探索空間とモデルを定義")
    print("3. Bi-level最適化で探索")
    print("4. 発見されたアーキテクチャで再学習")
    print("\nDARTSの利点:")
    print("- 探索時間: 4 GPU日（NASNetは1800 GPU日）")
    print("- 高精度: CIFAR-10で97%以上")
    print("- 転移可能: ImageNetなどにも適用可能")
    

* * *

## 3.5 NASの効率化

### 効率化手法の比較

手法 | 原理 | 高速化率 | 精度への影響  
---|---|---|---  
**Weight Sharing** | 候補アーキテクチャ間で重みを共有 | 1000倍 | 小  
**Proxy Tasks** | 簡易タスクで評価 | 10-100倍 | 中  
**Early Stopping** | 低性能モデルを早期に打ち切り | 2-5倍 | 小  
**Transfer Learning** | 類似タスクから知識転移 | 5-10倍 | 小  
  
### 1\. Weight Sharing（ENAS）

**Weight Sharing** は、全ての候補アーキテクチャが重みを共有するスーパーネットを構築します。
    
    
    # Weight Sharingの概念（ENAS風）
    import torch
    import torch.nn as nn
    
    class SharedWeightSuperNet(nn.Module):
        """重み共有スーパーネットワーク"""
    
        def __init__(self, num_nodes=4, C=16):
            super(SharedWeightSuperNet, self).__init__()
            self.num_nodes = num_nodes
    
            # 全ての可能な操作を事前に構築（重みを共有）
            self.ops = nn.ModuleDict({
                'conv_3x3': nn.Conv2d(C, C, 3, padding=1),
                'conv_5x5': nn.Conv2d(C, C, 5, padding=2),
                'max_pool': nn.MaxPool2d(3, stride=1, padding=1),
                'avg_pool': nn.AvgPool2d(3, stride=1, padding=1),
                'identity': nn.Identity()
            })
    
        def forward(self, x, architecture):
            """
            architecture: 各ノードの操作を指定
            例: [('conv_3x3', 0), ('max_pool', 1), ...]
            """
            states = [x, x]  # 初期状態
    
            for node_id, (op_name, input_id) in enumerate(architecture):
                # 指定された操作と入力で計算
                s = self.ops[op_name](states[input_id])
                states.append(s)
    
            # 最後のノードの出力を返す
            return states[-1]
    
    # スーパーネットの構築
    supernet = SharedWeightSuperNet(num_nodes=4, C=16)
    
    print("=== Weight Sharing（ENAS風）===")
    print(f"スーパーネットのパラメータ数: {sum(p.numel() for p in supernet.parameters()):,}")
    
    # 異なるアーキテクチャで同じ重みを使用
    arch1 = [('conv_3x3', 0), ('max_pool', 1), ('identity', 2), ('avg_pool', 1)]
    arch2 = [('conv_5x5', 1), ('identity', 0), ('max_pool', 2), ('conv_3x3', 3)]
    
    # ダミー入力
    x = torch.randn(1, 16, 32, 32)
    
    output1 = supernet(x, arch1)
    output2 = supernet(x, arch2)
    
    print(f"\nアーキテクチャ1の出力形状: {output1.shape}")
    print(f"アーキテクチャ2の出力形状: {output2.shape}")
    print("\n→ 同じ重みを共有しながら、異なるアーキテクチャを評価可能")
    

### 2\. Proxy Tasksによる高速化

Proxy Tasksでは、以下のような簡易化でコストを削減します：

簡易化 | 例 | 高速化  
---|---|---  
**データサイズ削減** | CIFAR-10の一部のみ使用 | 2-5倍  
**エポック数削減** | 10エポックで評価 | 5-10倍  
**モデルサイズ削減** | チャネル数を1/4に | 4-8倍  
**解像度削減** | 32x32の代わりに16x16 | 4倍  
  
### 3\. NAS-Benchデータセット

**NAS-Bench** は、事前計算されたアーキテクチャの性能データベースです。
    
    
    # NAS-Benchの概念的使用例
    # 注: 実際にはnasbenchiライブラリを使用（pip install nasbench）
    
    class NASBenchSimulator:
        """NAS-Benchのシミュレータ"""
    
        def __init__(self):
            # 事前計算された性能データ（ダミー）
            self.benchmark_data = {}
            self._populate_dummy_data()
    
        def _populate_dummy_data(self):
            """ダミーベンチマークデータの生成"""
            import random
            random.seed(42)
    
            # 100個のアーキテクチャの性能を事前計算
            for i in range(100):
                arch_hash = f"arch_{i:03d}"
                self.benchmark_data[arch_hash] = {
                    'val_accuracy': random.uniform(0.88, 0.95),
                    'test_accuracy': random.uniform(0.87, 0.94),
                    'training_time': random.uniform(100, 500),
                    'params': random.randint(1_000_000, 10_000_000),
                    'flops': random.randint(50_000_000, 500_000_000)
                }
    
        def query(self, architecture):
            """アーキテクチャの性能をクエリ（即座に返る）"""
            # アーキテクチャをハッシュ化
            arch_hash = self._hash_architecture(architecture)
    
            if arch_hash in self.benchmark_data:
                return self.benchmark_data[arch_hash]
            else:
                # 未知のアーキテクチャは推定
                return {
                    'val_accuracy': 0.90,
                    'test_accuracy': 0.89,
                    'training_time': 300,
                    'params': 5_000_000,
                    'flops': 250_000_000
                }
    
        def _hash_architecture(self, architecture):
            """アーキテクチャをハッシュ化"""
            # 簡易ハッシュ（実際はもっと複雑）
            arch_str = str(architecture)
            hash_val = sum(ord(c) for c in arch_str) % 100
            return f"arch_{hash_val:03d}"
    
    # NAS-Benchの使用
    bench = NASBenchSimulator()
    
    print("=== NAS-Benchによる高速評価 ===")
    
    # アーキテクチャの探索
    import time
    
    architectures = [
        [('conv_3x3', 0), ('max_pool', 1)],
        [('conv_5x5', 0), ('identity', 1)],
        [('avg_pool', 0), ('conv_3x3', 1)]
    ]
    
    start_time = time.time()
    results = []
    
    for arch in architectures:
        result = bench.query(arch)
        results.append((arch, result))
    
    end_time = time.time()
    
    print(f"探索時間: {end_time - start_time:.4f}秒")
    print(f"\n=== 探索結果 ===")
    for arch, result in results:
        print(f"アーキテクチャ: {arch}")
        print(f"  検証精度: {result['val_accuracy']:.3f}")
        print(f"  テスト精度: {result['test_accuracy']:.3f}")
        print(f"  学習時間: {result['training_time']:.1f}秒")
        print(f"  パラメータ数: {result['params']:,}")
        print()
    
    print("→ 実際の学習なしで性能を即座に取得可能")
    

### 効率化手法の組み合わせ
    
    
    # 複数の効率化手法を組み合わせた探索
    import numpy as np
    
    class EfficientNAS:
        """効率化されたNAS"""
    
        def __init__(self, use_weight_sharing=True, use_proxy=True,
                     use_early_stopping=True):
            self.use_weight_sharing = use_weight_sharing
            self.use_proxy = use_proxy
            self.use_early_stopping = use_early_stopping
    
            if use_weight_sharing:
                self.supernet = SharedWeightSuperNet()
    
            if use_proxy:
                self.proxy_epochs = 10  # 完全学習の代わりに10エポック
                self.proxy_data_fraction = 0.2  # データの20%のみ使用
    
        def evaluate_architecture(self, architecture, full_evaluation=False):
            """アーキテクチャを評価"""
            if full_evaluation:
                # 完全評価（最終候補のみ）
                epochs = 50
                data_fraction = 1.0
            else:
                # Proxy評価
                epochs = self.proxy_epochs if self.use_proxy else 50
                data_fraction = self.proxy_data_fraction if self.use_proxy else 1.0
    
            # Early stoppingのシミュレーション
            if self.use_early_stopping:
                # 最初の数エポックで性能が悪ければ打ち切り
                early_acc = np.random.random()
                if early_acc < 0.5:  # 閾値
                    return {'accuracy': early_acc, 'stopped_early': True}
    
            # 評価（ダミー）
            accuracy = np.random.uniform(0.85, 0.95)
    
            return {
                'accuracy': accuracy,
                'epochs': epochs,
                'data_fraction': data_fraction,
                'stopped_early': False
            }
    
        def search(self, num_candidates=100, top_k=5):
            """NAS探索の実行"""
            print("=== 効率的NAS探索 ===")
            print(f"Weight Sharing: {self.use_weight_sharing}")
            print(f"Proxy Tasks: {self.use_proxy}")
            print(f"Early Stopping: {self.use_early_stopping}")
            print()
    
            candidates = []
    
            # 1. 大規模なProxy評価
            for i in range(num_candidates):
                arch = [('conv_3x3', 0), ('max_pool', 1)]  # ダミー
                result = self.evaluate_architecture(arch, full_evaluation=False)
                candidates.append((arch, result))
    
            # Early stoppingで打ち切られなかった候補
            valid_candidates = [c for c in candidates if not c[1]['stopped_early']]
    
            print(f"初期候補: {num_candidates}")
            print(f"Early stoppingで削減: {num_candidates - len(valid_candidates)}")
    
            # 2. トップKを完全評価
            top_candidates = sorted(valid_candidates,
                                   key=lambda x: x[1]['accuracy'],
                                   reverse=True)[:top_k]
    
            print(f"完全評価する候補: {top_k}")
            print()
    
            final_results = []
            for arch, proxy_result in top_candidates:
                full_result = self.evaluate_architecture(arch, full_evaluation=True)
                final_results.append((arch, full_result))
    
            # 最良の候補を返す
            best = max(final_results, key=lambda x: x[1]['accuracy'])
    
            return best, final_results
    
    # 実行
    nas = EfficientNAS(
        use_weight_sharing=True,
        use_proxy=True,
        use_early_stopping=True
    )
    
    best_arch, all_results = nas.search(num_candidates=100, top_k=5)
    
    print("=== 探索結果 ===")
    print(f"最良アーキテクチャ: {best_arch[0]}")
    print(f"最良精度: {best_arch[1]['accuracy']:.3f}")
    print(f"\nトップ5の精度:")
    for i, (arch, result) in enumerate(all_results, 1):
        print(f"{i}. 精度={result['accuracy']:.3f}")
    

* * *

## 3.6 本章のまとめ

### 学んだこと

  1. **NASの探索空間**

     * Cell-based探索空間の設計
     * Macro vs Micro architecture
     * 探索空間のサイズと複雑性
  2. **NASの探索戦略**

     * 強化学習（NASNet）: RNNコントローラで生成
     * 進化アルゴリズム: 突然変異と選択
     * 勾配ベース（DARTS）: 連続緩和で高速化
     * One-shot（ENAS）: Weight Sharingで効率化
  3. **AutoKeras**

     * 画像、テキスト、構造化データの自動学習
     * カスタム探索空間の定義
     * 簡単なAPIで高度なNASを利用
  4. **DARTS**

     * 連続緩和による微分可能なNAS
     * Bi-level optimization（w と α）
     * 1000倍以上の高速化を実現
  5. **NASの効率化**

     * Weight Sharing: スーパーネットで重みを共有
     * Proxy Tasks: 簡易タスクで評価
     * Early Stopping: 低性能を早期打ち切り
     * NAS-Bench: 事前計算データベース

### 探索戦略の選択ガイドライン

状況 | 推奨手法 | 理由  
---|---|---  
計算資源が豊富 | 強化学習、進化 | 高精度が期待できる  
計算資源が限定的 | DARTS、ENAS | 高速で実用的  
初めてのNAS | AutoKeras | 簡単で使いやすい  
カスタマイズが必要 | DARTSの実装 | 柔軟性が高い  
ベンチマーク研究 | NAS-Bench | 再現性と公平性  
  
### 次の章へ

第4章では、**Feature Engineering Automation** を学びます：

  * 自動特徴量生成
  * 特徴選択の自動化
  * 特徴量重要度の可視化
  * AutoMLパイプラインの統合
  * 実践的なFeature Engineering

* * *

## 演習問題

### 問題1（難易度：easy）

NASの3要素（探索空間、探索戦略、性能評価）について、それぞれ説明してください。

解答例

**解答** ：

  1. **探索空間（Search Space）**

     * 説明: 探索可能なアーキテクチャの集合
     * 例: Cell-based（Normal CellとReduction Cell）、層の種類（Conv、Pooling）、結合パターン
     * 重要性: 探索空間が大きすぎると計算コストが高く、小さすぎると最適解を逃す
  2. **探索戦略（Search Strategy）**

     * 説明: アーキテクチャをどのように探索するか
     * 例: 強化学習（NASNet）、進化アルゴリズム（AmoebaNet）、勾配ベース（DARTS）
     * トレードオフ: 精度 vs 計算コスト
  3. **性能評価（Performance Estimation）**

     * 説明: アーキテクチャの良し悪しを判定する方法
     * 指標: 精度、FLOPs、パラメータ数、レイテンシ、メモリ使用量
     * 効率化: Proxy tasks、Weight sharing、Early stopping

### 問題2（難易度：medium）

DARTSが強化学習ベースのNASに比べて高速な理由を、連続緩和の観点から説明してください。

解答例

**解答** ：

**強化学習ベースのNAS（例: NASNet）** ：

  * 離散的探索: アーキテクチャをサンプリング → 学習 → 評価を繰り返す
  * 各候補を個別に学習する必要がある
  * 計算コスト: 数千のアーキテクチャ × 完全学習 = 非常に高い（1800 GPU日）

**DARTS（勾配ベース）** ：

  * 連続緩和: 離散的な選択（どの操作を使うか）を連続的な重み付き和に変換
  * 式: $\bar{o}(x) = \sum_o \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} \cdot o(x)$
  * 勾配降下法が適用可能: αを勾配で最適化できる
  * Weight sharing: 全ての候補が同じスーパーネットを共有
  * 計算コスト: 1回のスーパーネット学習 = 大幅削減（4 GPU日）

**高速化の理由** ：

  1. 離散→連続: 微分可能になり、効率的な勾配最適化が可能
  2. Weight sharing: 候補間で重みを共有し、個別学習を回避
  3. Bi-level optimization: wとαを交互に更新し、効率的に探索

**結果** ：

  * NASNet: 1800 GPU日
  * DARTS: 4 GPU日
  * 高速化率: 約450倍

### 問題3（難易度：medium）

以下のコードを完成させて、AutoKerasで画像分類モデルを探索してください。
    
    
    import autokeras as ak
    from tensorflow.keras.datasets import fashion_mnist
    
    # データの準備
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # TODO: データの正規化
    # x_train = ...
    # x_test = ...
    
    # TODO: AutoKerasのImageClassifierを構築
    # clf = ak.ImageClassifier(...)
    
    # TODO: モデルの学習
    # clf.fit(...)
    
    # TODO: 評価
    # test_acc = ...
    # print(f"テスト精度: {test_acc:.4f}")
    

解答例
    
    
    import autokeras as ak
    from tensorflow.keras.datasets import fashion_mnist
    
    # データの準備
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # データの正規化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # データサイズ削減（デモ用）
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    
    print("=== Fashion-MNIST分類 ===")
    print(f"訓練データ: {x_train.shape}")
    print(f"テストデータ: {x_test.shape}")
    
    # AutoKerasのImageClassifier
    clf = ak.ImageClassifier(
        max_trials=5,  # 探索する候補数
        epochs=10,     # 各候補の学習エポック数
        overwrite=True,
        directory='autokeras_fashion',
        project_name='fashion_mnist'
    )
    
    # モデルの学習
    print("\n探索を開始...")
    clf.fit(
        x_train, y_train,
        validation_split=0.2,
        verbose=1
    )
    
    # 評価
    print("\n=== 評価 ===")
    test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
    print(f"テスト精度: {test_acc:.4f}")
    print(f"テスト損失: {test_loss:.4f}")
    
    # 最良モデルの取得
    best_model = clf.export_model()
    print("\n=== 発見されたモデル ===")
    best_model.summary()
    
    # 予測例
    import numpy as np
    predictions = clf.predict(x_test[:5])
    print("\n=== 予測例 ===")
    for i in range(5):
        print(f"サンプル {i+1}: 予測={np.argmax(predictions[i])}, 真値={y_test[i]}")
    

### 問題4（難易度：hard）

Weight Sharingを使ったスーパーネットワークを実装し、異なる2つのアーキテクチャで重みが共有されることを確認してください。

解答例
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SharedOperations(nn.Module):
        """重みを共有する操作のプール"""
    
        def __init__(self, C):
            super(SharedOperations, self).__init__()
    
            # 全ての可能な操作（重みは1度だけ定義）
            self.ops = nn.ModuleDict({
                'conv_3x3': nn.Conv2d(C, C, 3, padding=1, bias=False),
                'conv_5x5': nn.Conv2d(C, C, 5, padding=2, bias=False),
                'sep_conv_3x3': nn.Sequential(
                    nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                    nn.Conv2d(C, C, 1, bias=False)
                ),
                'max_pool_3x3': nn.MaxPool2d(3, stride=1, padding=1),
                'avg_pool_3x3': nn.AvgPool2d(3, stride=1, padding=1),
                'identity': nn.Identity()
            })
    
        def forward(self, x, op_name):
            """指定された操作を適用"""
            return self.ops[op_name](x)
    
    class SuperNet(nn.Module):
        """スーパーネットワーク"""
    
        def __init__(self, C=16):
            super(SuperNet, self).__init__()
            self.shared_ops = SharedOperations(C)
    
        def forward(self, x, architecture):
            """
            architecture: [(op_name, input_id), ...]の形式
            """
            # 初期状態
            states = [x]
    
            for op_name, input_id in architecture:
                s = self.shared_ops(states[input_id], op_name)
                states.append(s)
    
            # 最後の状態を返す
            return states[-1]
    
    # スーパーネットの構築
    supernet = SuperNet(C=16)
    
    print("=== Weight Sharingの検証 ===")
    print(f"スーパーネットのパラメータ数: {sum(p.numel() for p in supernet.parameters()):,}")
    
    # 操作ごとのパラメータ数
    print("\n各操作のパラメータ数:")
    for name, op in supernet.shared_ops.ops.items():
        num_params = sum(p.numel() for p in op.parameters())
        print(f"  {name}: {num_params:,}")
    
    # 2つの異なるアーキテクチャ
    arch1 = [('conv_3x3', 0), ('max_pool_3x3', 0), ('identity', 1)]
    arch2 = [('conv_5x5', 0), ('avg_pool_3x3', 0), ('conv_3x3', 1)]
    
    # 同じ入力
    x = torch.randn(2, 16, 32, 32)
    
    # アーキテクチャ1で順伝播
    output1 = supernet(x, arch1)
    
    # アーキテクチャ2で順伝播
    output2 = supernet(x, arch2)
    
    print(f"\n=== 順伝播の検証 ===")
    print(f"アーキテクチャ1の出力形状: {output1.shape}")
    print(f"アーキテクチャ2の出力形状: {output2.shape}")
    
    # 重みが共有されていることを確認
    print("\n=== 重み共有の確認 ===")
    conv_3x3_params_before = list(supernet.shared_ops.ops['conv_3x3'].parameters())[0].clone()
    
    # アーキテクチャ1で逆伝播（conv_3x3を使用）
    loss1 = output1.sum()
    loss1.backward()
    
    conv_3x3_params_after = list(supernet.shared_ops.ops['conv_3x3'].parameters())[0]
    
    # 勾配が蓄積されているか確認
    has_gradient = conv_3x3_params_after.grad is not None
    print(f"conv_3x3に勾配が蓄積: {has_gradient}")
    
    # 重みの共有を視覚的に確認
    print("\n=== 重み共有の利点 ===")
    print("1. メモリ効率: 全アーキテクチャで同じ重みを使用")
    print("2. 学習効率: 1つのスーパーネットで全候補を評価")
    print("3. 高速化: 個別学習の代わりに共有学習")
    
    # 異なるアーキテクチャを試す
    print("\n=== 複数アーキテクチャの評価 ===")
    architectures = [
        [('conv_3x3', 0), ('max_pool_3x3', 0)],
        [('conv_5x5', 0), ('identity', 0)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
    ]
    
    for i, arch in enumerate(architectures, 1):
        output = supernet(x, arch)
        print(f"アーキテクチャ {i}: 出力形状 = {output.shape}, 平均値 = {output.mean():.4f}")
    
    print("\n→ 全てのアーキテクチャが同じ重みを共有しながら評価されました")
    

### 問題5（難易度：hard）

DARTSのBi-level最適化において、なぜ訓練データと検証データを分けて最適化する必要があるのか説明してください。また、同じデータで最適化した場合に何が起こるか予想してください。

解答例

**解答** ：

**Bi-level最適化の目的** ：

DARTSは2種類のパラメータを最適化します：

  1. **ネットワークの重み（w）** : 訓練データで最小化
  2. **アーキテクチャパラメータ（α）** : 検証データで最小化

最適化問題：

$$ \begin{aligned} \min_{\alpha} \quad & \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \\\ \text{s.t.} \quad & w^*(\alpha) = \arg\min_{w} \mathcal{L}_{\text{train}}(w, \alpha) \end{aligned} $$

**訓練・検証データを分ける理由** ：

  1. **過学習の防止**

     * αを訓練データで最適化すると、訓練データに過適合したアーキテクチャを選択
     * 検証データで最適化することで、汎化性能の高いアーキテクチャを選択
  2. **役割の分離**

     * w: 与えられたアーキテクチャで最良の重みを学習（訓練データ）
     * α: 検証性能が最も高いアーキテクチャを選択（検証データ）
  3. **公平な評価**

     * 訓練データで学習したwを、独立した検証データで評価
     * 真の汎化性能を反映したアーキテクチャ選択

**同じデータで最適化した場合の問題** ：
    
    
    # 誤った方法（同じデータでwとαを最適化）
    # ❌ 問題のあるコード例
    for epoch in range(num_epochs):
        # 訓練データでwを更新
        loss_w = train_loss(w, alpha, train_data)
        w.update(-lr * grad(loss_w, w))
    
        # 同じ訓練データでαを更新 ← 問題！
        loss_alpha = train_loss(w, alpha, train_data)
        alpha.update(-lr * grad(loss_alpha, alpha))
    

**起こる問題** ：

  1. **過学習** : 訓練データに特化したアーキテクチャを選択
  2. **Identity操作の優先** : 計算コストなしで訓練損失を下げられるため、skip connectionばかり選択
  3. **汎化性能の低下** : テストデータでの性能が悪化
  4. **意味のある探索の失敗** : 真に有用なアーキテクチャを発見できない

**正しい方法** ：
    
    
    # ✅ 正しい方法
    for epoch in range(num_epochs):
        # 訓練データでwを更新
        loss_w = train_loss(w, alpha, train_data)
        w.update(-lr * grad(loss_w, w))
    
        # 検証データでαを更新 ← 正しい！
        loss_alpha = val_loss(w, alpha, val_data)
        alpha.update(-lr * grad(loss_alpha, alpha))
    

**まとめ** ：

側面 | 訓練・検証分離 | 同じデータ使用  
---|---|---  
**過学習** | 防止できる | 過学習しやすい  
**汎化性能** | 高い | 低い  
**アーキテクチャ** | 有用 | trivial（identity偏重）  
**実用性** | 高い | 低い  
  
* * *

## 参考文献

  1. Zoph, B., & Le, Q. V. (2017). _Neural Architecture Search with Reinforcement Learning_. ICLR 2017.
  2. Liu, H., Simonyan, K., & Yang, Y. (2019). _DARTS: Differentiable Architecture Search_. ICLR 2019.
  3. Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). _Efficient Neural Architecture Search via Parameter Sharing_. ICML 2018.
  4. Real, E., et al. (2019). _Regularized Evolution for Image Classifier Architecture Search_. AAAI 2019.
  5. Jin, H., Song, Q., & Hu, X. (2019). _Auto-Keras: An Efficient Neural Architecture Search System_. KDD 2019.
  6. Elsken, T., Metzen, J. H., & Hutter, F. (2019). _Neural Architecture Search: A Survey_. JMLR 2019.
