---
title: 第1章：PyTorch Geometric入門とグラフデータの基礎
chapter_title: 第1章：PyTorch Geometric入門とグラフデータの基礎
subtitle: グラフ構造データとGNNの第一歩
reading_time: 30-35分
difficulty: 中級
code_examples: 12
exercises: 5
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/JWRfVp9wjPE"
    title="PyTorch Geometric 第1章: PyTorch Geometric入門とグラフデータの基礎"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

---

この章では、グラフニューラルネットワーク（GNN）の基礎となるグラフデータの概念と、PyTorch Geometric（PyG）ライブラリの使い方を学びます。グラフの基本構造、PyGのインストール、Dataオブジェクトの扱い方、組み込みデータセット、そして簡単なGCNレイヤーの実装を通じて、GNN開発の基盤を固めていきましょう。

## 学習目標

  * ✅ グラフデータの基本概念（ノード、エッジ、隣接行列）を理解する
  * ✅ PyTorch Geometricをインストールし、動作確認ができる
  * ✅ PyGのDataオブジェクトを作成・操作できる
  * ✅ 組み込みデータセットを読み込み、探索できる
  * ✅ 簡単なGCNレイヤーを使ったノード分類を実装できる

## 1\. グラフデータの基礎概念

**グラフ（Graph）** は、ノード（頂点）とエッジ（辺）から構成されるデータ構造です。実世界の多くの複雑な関係性をグラフで表現できます。

### グラフの基本要素

  * **ノード（Node/Vertex）** : グラフの要素を表す点。例：人、分子の原子、論文
  * **エッジ（Edge/Link）** : ノード間の関係を表す線。例：友人関係、化学結合、引用関係
  * **特徴量（Features）** : ノードやエッジに付随する属性情報

    
    
    ```mermaid
    graph LR
      A[ノード A] -->|エッジ| B[ノード B]
      B --> C[ノード C]
      A --> C
      C --> D[ノード D]
      B --> D
    ```

### グラフの種類

分類 | 種類 | 説明 | 例  
---|---|---|---  
方向性 | 有向グラフ | エッジに方向がある | Twitterのフォロー関係、引用ネットワーク  
| 無向グラフ | エッジに方向がない | Facebookの友人関係、分子構造  
ノード種類 | 同種グラフ | 1種類のノード | ソーシャルネットワーク（人のみ）  
| 異種グラフ | 複数種類のノード | ユーザーと商品を含むレコメンデーショングラフ  
重み | 重み付きグラフ | エッジに重み（強度）がある | 道路ネットワーク（距離）、類似度グラフ  
  
### グラフの表現方法

グラフをコンピュータで扱うための主な表現方法：

#### 1\. 隣接行列（Adjacency Matrix）

ノード数を \\(N\\) とすると、\\(N \times N\\) の行列 \\(A\\) で表現：

$$A_{ij} = \begin{cases} 1 & \text{if ノード } i \text{ から } j \text{ へエッジがある} \\\ 0 & \text{otherwise} \end{cases}$$
    
    
    import numpy as np
    
    # 4ノードのグラフの隣接行列
    # エッジ: 0→1, 1→2, 0→2, 2→3, 1→3
    adjacency_matrix = np.array([
        [0, 1, 1, 0],  # ノード0からの接続
        [0, 0, 1, 1],  # ノード1からの接続
        [0, 0, 0, 1],  # ノード2からの接続
        [0, 0, 0, 0]   # ノード3からの接続
    ])
    
    print("隣接行列:\n", adjacency_matrix)
    

#### 2\. エッジインデックス（Edge Index）

PyTorch Geometricで採用されている効率的な表現方法。スパース（疎）なグラフに適しています。
    
    
    import torch
    
    # 同じグラフをエッジインデックスで表現
    # 形状: [2, num_edges]
    # 1行目: 始点ノード、2行目: 終点ノード
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1],  # 始点ノード
        [1, 2, 2, 3, 3]   # 終点ノード
    ], dtype=torch.long)
    
    print("エッジインデックス:\n", edge_index)
    

**💡 なぜエッジインデックス？**

隣接行列は \\(O(N^2)\\) のメモリが必要ですが、実世界のグラフは疎（スパース）なことが多く、エッジインデックスは \\(O(E)\\)（\\(E\\)はエッジ数）で済みます。例えば、1万ノードで平均次数10のグラフでは、隣接行列は100MB必要ですが、エッジインデックスは約800KBで済みます。

## 2\. PyTorch Geometricのインストールと環境構築

**PyTorch Geometric** は、PyTorchをベースにしたグラフニューラルネットワーク専用ライブラリです。

### インストール方法

PyTorch Geometricは、PyTorchとCUDAのバージョンに依存します。まず、使用環境を確認しましょう。
    
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    

#### 方法1: pip経由のインストール（推奨）
    
    
    # PyTorch 2.0以降の場合（CPU版）
    pip install torch-geometric
    
    # 追加の依存パッケージ
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    
    # GPU版（CUDA 11.8の場合）
    pip install torch-geometric
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    

#### 方法2: conda経由のインストール
    
    
    # condaの場合
    conda install pyg -c pyg
    

#### 方法3: Google Colab（環境構築不要）

Google Colabでは以下のコマンドで簡単にインストールできます：
    
    
    !pip install torch-geometric
    !pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    

### インストール確認
    
    
    import torch
    import torch_geometric
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    # サンプルデータで動作確認
    from torch_geometric.data import Data
    
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    print(f"\nSample Data object created successfully!")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    

**出力例：**
    
    
    PyTorch version: 2.1.0
    PyTorch Geometric version: 2.4.0
    
    Sample Data object created successfully!
    Number of nodes: 3
    Number of edges: 4
    

## 3\. PyGのDataオブジェクト

PyTorch Geometricの中心的なデータ構造が**Dataオブジェクト** です。グラフの構造と特徴量を効率的に格納します。

### Dataオブジェクトの構造

属性 | 形状 | 説明  
---|---|---  
`x` | [num_nodes, num_features] | ノード特徴量行列  
`edge_index` | [2, num_edges] | エッジの接続情報（COO形式）  
`edge_attr` | [num_edges, num_edge_features] | エッジ特徴量行列（オプション）  
`y` | 任意 | ターゲットラベル（ノードまたはグラフ）  
`pos` | [num_nodes, num_dimensions] | ノードの位置座標（オプション）  
  
### Dataオブジェクトの作成
    
    
    import torch
    from torch_geometric.data import Data
    
    # ノード特徴量（3ノード、各ノード2次元特徴）
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=torch.float)
    
    # エッジインデックス（4つのエッジ）
    # 0→1, 1→0, 1→2, 2→1
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    # エッジ特徴量（各エッジ1次元特徴）
    edge_attr = torch.tensor([[1.0], [1.0], [2.0], [2.0]], dtype=torch.float)
    
    # ノードラベル（ノード分類タスクの場合）
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    
    # Dataオブジェクト作成
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(data)
    print(f"\nNumber of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_node_features}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    

**出力：**
    
    
    Data(x=[3, 2], edge_index=[2, 4], edge_attr=[4, 1], y=[3])
    
    Number of nodes: 3
    Number of edges: 4
    Number of features: 2
    Has isolated nodes: False
    Has self-loops: False
    Is undirected: True
    

### Dataオブジェクトの操作
    
    
    import torch
    from torch_geometric.data import Data
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # 特定ノードの特徴量取得
    print("Node 0 features:", data.x[0])
    
    # 特定エッジの情報取得
    print("Edge 0:", data.edge_index[:, 0])
    print("Edge 0 attribute:", data.edge_attr[0])
    
    # データをGPUに転送
    if torch.cuda.is_available():
        data = data.to('cuda')
        print(f"Data moved to: {data.x.device}")
    
    # CPUに戻す
    data = data.to('cpu')
    
    # データの検証
    print(f"\nIs valid: {data.validate()}")
    

## 4\. 基本的なデータ操作と組み込みデータセット

PyTorch Geometricには、研究・学習用の組み込みデータセットが多数用意されています。

### 主要な組み込みデータセット

データセット | 種類 | ノード数 | 説明  
---|---|---|---  
Cora | 引用ネットワーク | 2,708 | 論文の引用関係、7クラス分類  
Citeseer | 引用ネットワーク | 3,327 | 論文の引用関係、6クラス分類  
PubMed | 引用ネットワーク | 19,717 | 医学論文の引用関係、3クラス分類  
PPI | 生物ネットワーク | 14,755 | タンパク質相互作用、マルチラベル分類  
QM9 | 分子グラフ | 約13万分子 | 分子特性予測、回帰タスク  
  
### Coraデータセットの読み込み
    
    
    from torch_geometric.datasets import Planetoid
    
    # Coraデータセットをダウンロード・読み込み
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # 最初のグラフ（Coraは単一グラフ）
    data = dataset[0]
    
    print(f"\nGraph structure:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")
    
    # ノード特徴量とラベルの確認
    print(f"\nNode features shape: {data.x.shape}")
    print(f"Node labels shape: {data.y.shape}")
    print(f"First node features: {data.x[0][:10]}...")
    print(f"First node label: {data.y[0].item()}")
    

**出力例：**
    
    
    Dataset: Cora()
    Number of graphs: 1
    Number of features: 1433
    Number of classes: 7
    
    Graph structure:
    Number of nodes: 2708
    Number of edges: 10556
    Average node degree: 3.90
    Training nodes: 140
    Validation nodes: 500
    Test nodes: 1000
    
    Node features shape: torch.Size([2708, 1433])
    Node labels shape: torch.Size([2708])
    First node features: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])...
    First node label: 3
    

### DataLoaderの使い方

複数のグラフを含むデータセットでは、DataLoaderを使ってバッチ処理します。
    
    
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    # ENZYMES データセット（タンパク質のグラフ分類）
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of features: {dataset.num_features}")
    
    # DataLoader作成
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # バッチの確認
    for batch in loader:
        print(f"\nBatch:")
        print(f"Number of graphs in batch: {batch.num_graphs}")
        print(f"Total nodes in batch: {batch.num_nodes}")
        print(f"Total edges in batch: {batch.num_edges}")
        print(f"Batch shape: {batch.batch.shape}")
        break  # 最初のバッチのみ表示
    

**💡 バッチ処理の仕組み**

PyGのDataLoaderは、複数のグラフを1つの大きなグラフとして結合します。各ノードがどのグラフに属するかは`batch`属性で管理されます。これにより、異なるサイズのグラフを効率的にバッチ処理できます。

## 5\. 簡単なGNNの実装例

最も基本的なグラフニューラルネットワーク層である**GCNConv（Graph Convolutional Network）** を使って、ノード分類モデルを実装します。

### GCNの基本原理

GCNは各ノードの特徴量を、隣接ノードの特徴量を集約して更新します：

$$\mathbf{x}_i^{(k+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \\{i\\}} \frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(k)} \mathbf{x}_j^{(k)}\right)$$

ここで：

  * \\(\mathbf{x}_i^{(k)}\\): 層 \\(k\\) におけるノード \\(i\\) の特徴量
  * \\(\mathcal{N}(i)\\): ノード \\(i\\) の隣接ノード集合
  * \\(d_i\\): ノード \\(i\\) の次数
  * \\(\mathbf{W}^{(k)}\\): 学習可能な重み行列
  * \\(\sigma\\): 活性化関数（ReLU等）

    
    
    ```mermaid
    graph LR
      A[ノード A特徴量] --> AGG[集約]
      B[隣接ノード B特徴量] --> AGG
      C[隣接ノード C特徴量] --> AGG
      AGG --> UPDATE[更新]
      UPDATE --> A2[新しい特徴量]
    ```

### GCNモデルの実装
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class GCN(torch.nn.Module):
        def __init__(self, num_features, num_classes):
            super(GCN, self).__init__()
            # 2層のGCN
            self.conv1 = GCNConv(num_features, 16)
            self.conv2 = GCNConv(16, num_classes)
    
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
    
            # 第1層: 入力 → 16次元
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
    
            # 第2層: 16次元 → クラス数
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
    # モデル作成
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    model = GCN(num_features=dataset.num_features,
                num_classes=dataset.num_classes)
    
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    

### 学習ループの実装
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import Planetoid
    
    # データとモデルの準備
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=dataset.num_features,
                num_classes=dataset.num_classes).to(device)
    data = data.to(device)
    
    # オプティマイザ
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 学習ループ
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
    
        # 順伝播
        out = model(data)
    
        # 損失計算（訓練データのみ）
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
        # 逆伝播
        loss.backward()
        optimizer.step()
    
        # 10エポックごとに結果表示
        if (epoch + 1) % 10 == 0:
            model.eval()
            _, pred = model(data).max(dim=1)
            correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            accuracy = correct / data.train_mask.sum().item()
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {accuracy:.4f}')
            model.train()
    

### モデルの評価
    
    
    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            _, pred = out.max(dim=1)
    
            # 訓練データ精度
            correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            train_acc = correct / data.train_mask.sum().item()
    
            # 検証データ精度
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            val_acc = correct / data.val_mask.sum().item()
    
            # テストデータ精度
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            test_acc = correct / data.test_mask.sum().item()
    
        return train_acc, val_acc, test_acc
    
    train_acc, val_acc, test_acc = test(model, data)
    print(f'\nFinal Results:')
    print(f'Train Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    

**出力例：**
    
    
    Epoch 010, Loss: 1.9234, Train Acc: 0.3143
    Epoch 020, Loss: 1.7845, Train Acc: 0.4357
    Epoch 030, Loss: 1.5234, Train Acc: 0.6000
    ...
    Epoch 200, Loss: 0.5123, Train Acc: 0.9714
    
    Final Results:
    Train Accuracy: 0.9714
    Validation Accuracy: 0.7540
    Test Accuracy: 0.8130
    

**🎉 初めてのGNN実装完了！**

Coraデータセットでテスト精度81%を達成しました。これはグラフ構造を考慮しないMLPモデル（約60%）と比べて大幅に改善されています。GNNはノード間の関係性を学習することで、より高い精度を実現しています。

## 演習問題

**演習1：カスタムグラフの作成**

以下の条件でグラフを作成してください：

  1. 5つのノード（各ノード3次元特徴量）
  2. 無向グラフ（双方向エッジ）
  3. エッジ: 0-1, 1-2, 2-3, 3-4, 4-0
  4. 各ノードにランダムなラベル（0,1,2のいずれか）を付与

    
    
    # ここにコードを書く
    

**演習2：データセット探索**

Citeseerデータセットを読み込み、以下の情報を出力してください：

  * ノード数、エッジ数
  * 平均ノード次数
  * 特徴量次元数
  * クラス数
  * 訓練/検証/テストノード数

**演習3：3層GCNの実装**

2層GCNを拡張して、3層のGCNモデルを実装してください。中間層の次元数は32と16にしてください。Coraデータセットで学習し、精度を比較してください。

ヒント: 層を増やすと過学習しやすくなるため、Dropoutの調整が必要かもしれません。

**演習4：エッジ特徴量の活用**

エッジに重み（特徴量）を持つグラフを作成し、`edge_attr`属性を設定してください。エッジの重みはランダムな値（0.1〜1.0の範囲）としてください。

**演習5：グラフの可視化**

NetworkXとMatplotlibを使って、作成したグラフを可視化してください。ノードの色をラベルで分けて表示してください。
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx
    
    # PyGのDataオブジェクトをNetworkXグラフに変換
    # ここにコードを書く
    

## まとめ

この章では、グラフニューラルネットワークの基礎を学びました：

  * ✅ グラフデータの基本概念（ノード、エッジ、隣接行列、エッジインデックス）
  * ✅ PyTorch Geometricのインストールと環境構築
  * ✅ Dataオブジェクトの構造と操作方法
  * ✅ 組み込みデータセット（Cora、ENZYMES等）の使い方
  * ✅ GCNConvレイヤーを使ったノード分類の実装

**🎉 次のステップ**

次章では、グラフ畳み込みネットワーク（GCN）のメッセージパッシングの仕組みを詳しく学び、ノード分類タスクを完全に理解します。過学習対策やハイパーパラメータ調整についても実践的に学びます。

* * *

**参考リソース**

  * [PyTorch Geometric公式ドキュメント](<https://pytorch-geometric.readthedocs.io/>)
  * [PyTorch Geometric GitHub](<https://github.com/pyg-team/pytorch_geometric>)
  * [GCN論文: Semi-Supervised Classification with Graph Convolutional Networks](<https://arxiv.org/abs/1609.02907>)
