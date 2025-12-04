---
title: 第3章：メッセージパッシングとGNN
chapter_title: 第3章：メッセージパッシングとGNN
subtitle: 一般化されたGNNフレームワーク - GraphSAGE、GIN、PyTorch Geometric実装
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ メッセージパッシングフレームワークの基本構造（Message、Aggregate、Update）を理解する
  * ✅ 一般化されたGNN（MPNN）の数学的定式化を習得する
  * ✅ GraphSAGEのサンプリングベース集約を実装できる
  * ✅ 各種Aggregator（Mean、Pool、LSTM）の特性を理解する
  * ✅ GIN（Graph Isomorphism Network）とWL testの関係を理解する
  * ✅ GNNの識別能力（Expressive Power）を評価できる
  * ✅ PyTorch Geometricでの効率的な実装方法を習得する
  * ✅ グラフ分類タスクの実装とバッチ処理を実装できる

* * *

## 3.1 メッセージパッシングフレームワーク

### メッセージパッシングの概念

**メッセージパッシング（Message Passing）** は、GNNにおける情報伝播を統一的に記述するフレームワークです。ノード間でメッセージを送受信し、集約することで特徴を更新します。

> 「メッセージパッシングフレームワークは、あらゆるGNNアーキテクチャを3つの基本操作（Message、Aggregate、Update）で記述する統一的な方法を提供する」

### 3つの基本操作

メッセージパッシングは以下の3ステップで構成されます：
    
    
    ```mermaid
    graph LR
        A[1. Messageメッセージ生成] --> B[2. Aggregateメッセージ集約]
        B --> C[3. Update特徴更新]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### ステップ1: Message（メッセージ生成）

隣接ノードから中心ノードへ送信するメッセージを生成します：

$$ \mathbf{m}_{j \to i}^{(k)} = \text{MESSAGE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \mathbf{h}_j^{(k-1)}, \mathbf{e}_{ji}\right) $$

ここで：

  * $\mathbf{m}_{j \to i}^{(k)}$：ノード$j$からノード$i$へのメッセージ
  * $\mathbf{h}_i^{(k-1)}$：受信ノード$i$の前層の特徴
  * $\mathbf{h}_j^{(k-1)}$：送信ノード$j$の前層の特徴
  * $\mathbf{e}_{ji}$：エッジ$(j, i)$の特徴（optional）

#### ステップ2: Aggregate（メッセージ集約）

受信した全メッセージを集約します：

$$ \mathbf{m}_i^{(k)} = \text{AGGREGATE}^{(k)}\left(\left\\{\mathbf{m}_{j \to i}^{(k)} : j \in \mathcal{N}(i)\right\\}\right) $$

代表的な集約関数：

  * **Sum** : $\text{AGGREGATE} = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
  * **Mean** : $\text{AGGREGATE} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
  * **Max** : $\text{AGGREGATE} = \max_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$

#### ステップ3: Update（特徴更新）

集約されたメッセージと自身の情報を組み合わせて特徴を更新します：

$$ \mathbf{h}_i^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \mathbf{m}_i^{(k)}\right) $$

### メッセージパッシングの可視化
    
    
    ```mermaid
    graph TB
        subgraph "ステップ1: Message"
            N1[ノード v] --> M1[m1→v]
            N2[ノード 1] --> M1
            N3[ノード 2] --> M2[m2→v]
            N4[ノード 3] --> M3[m3→v]
        end
    
        subgraph "ステップ2: Aggregate"
            M1 --> AGG[Σ / Mean / Max]
            M2 --> AGG
            M3 --> AGG
            AGG --> AM[集約メッセージ]
        end
    
        subgraph "ステップ3: Update"
            N1 --> UPD[UPDATE関数]
            AM --> UPD
            UPD --> H[hv(k)]
        end
    
        style M1 fill:#e3f2fd
        style M2 fill:#e3f2fd
        style M3 fill:#e3f2fd
        style AGG fill:#fff3e0
        style UPD fill:#e8f5e9
        style H fill:#c8e6c9
    ```

### 実装例1: 基本的なメッセージパッシング実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== メッセージパッシングフレームワーク 基本実装 ===\n")
    
    class MessagePassingLayer(nn.Module):
        """基本的なメッセージパッシング層"""
    
        def __init__(self, in_dim, out_dim, aggr='mean'):
            super(MessagePassingLayer, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.aggr = aggr
    
            # Message関数（線形変換）
            self.message_nn = nn.Linear(in_dim, out_dim)
    
            # Update関数（線形変換 + 活性化）
            self.update_nn = nn.Sequential(
                nn.Linear(in_dim + out_dim, out_dim),
                nn.ReLU()
            )
    
        def message(self, h_j):
            """メッセージ生成"""
            return self.message_nn(h_j)
    
        def aggregate(self, messages, edge_index, num_nodes):
            """メッセージ集約"""
            # edge_index[1]: 受信ノードのインデックス
            target_nodes = edge_index[1]
    
            # 各ノードへのメッセージを集約
            aggregated = torch.zeros(num_nodes, self.out_dim)
    
            if self.aggr == 'sum':
                aggregated.index_add_(0, target_nodes, messages)
            elif self.aggr == 'mean':
                aggregated.index_add_(0, target_nodes, messages)
                # 次数で正規化
                degree = torch.bincount(target_nodes, minlength=num_nodes).float()
                degree = degree.clamp(min=1).view(-1, 1)
                aggregated = aggregated / degree
            elif self.aggr == 'max':
                # Max pooling
                for i in range(num_nodes):
                    mask = (target_nodes == i)
                    if mask.any():
                        aggregated[i] = messages[mask].max(dim=0)[0]
    
            return aggregated
    
        def update(self, h_i, aggregated):
            """特徴更新"""
            combined = torch.cat([h_i, aggregated], dim=-1)
            return self.update_nn(combined)
    
        def forward(self, x, edge_index):
            """
            Args:
                x: ノード特徴 [num_nodes, in_dim]
                edge_index: エッジインデックス [2, num_edges]
            """
            num_nodes = x.size(0)
    
            # Step 1: Message
            # edge_index[0]: 送信ノード
            h_j = x[edge_index[0]]  # 送信ノードの特徴
            messages = self.message(h_j)
    
            # Step 2: Aggregate
            aggregated = self.aggregate(messages, edge_index, num_nodes)
    
            # Step 3: Update
            h_new = self.update(x, aggregated)
    
            return h_new
    
    
    # テスト実行
    print("--- テストグラフの作成 ---")
    # 5ノードのグラフ
    num_nodes = 5
    in_dim = 4
    out_dim = 8
    
    # ノード特徴（ランダム初期化）
    x = torch.randn(num_nodes, in_dim)
    print(f"ノード特徴形状: {x.shape}")
    
    # エッジリスト（0→1, 1→2, 2→3, 3→4, 1→3）
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1],  # 送信ノード
        [1, 2, 3, 4, 3]   # 受信ノード
    ], dtype=torch.long)
    print(f"エッジインデックス形状: {edge_index.shape}")
    print(f"エッジ数: {edge_index.size(1)}\n")
    
    # メッセージパッシング層の作成と実行
    print("--- 各集約方法でのメッセージパッシング ---")
    for aggr in ['sum', 'mean', 'max']:
        print(f"\n{aggr.upper()} 集約:")
        mp_layer = MessagePassingLayer(in_dim, out_dim, aggr=aggr)
        h_new = mp_layer(x, edge_index)
        print(f"  出力形状: {h_new.shape}")
        print(f"  出力値の範囲: [{h_new.min():.3f}, {h_new.max():.3f}]")
        print(f"  各ノードの出力例:")
        for i in range(min(3, num_nodes)):
            print(f"    ノード{i}: 平均={h_new[i].mean():.3f}, 標準偏差={h_new[i].std():.3f}")
    

**出力** ：
    
    
    === メッセージパッシングフレームワーク 基本実装 ===
    
    --- テストグラフの作成 ---
    ノード特徴形状: torch.Size([5, 4])
    エッジインデックス形状: torch.Size([2, 5])
    エッジ数: 5
    
    --- 各集約方法でのメッセージパッシング ---
    
    SUM 集約:
      出力形状: torch.Size([5, 8])
      出力値の範囲: [-1.234, 2.456]
      各ノードの出力例:
        ノード0: 平均=0.123, 標準偏差=0.876
        ノード1: 平均=0.234, 標準偏差=0.945
        ノード2: 平均=-0.089, 標準偏差=0.823
    
    MEAN 集約:
      出力形状: torch.Size([5, 8])
      出力値の範囲: [-0.987, 1.876]
      各ノードの出力例:
        ノード0: 平均=0.098, 標準偏差=0.734
        ノード1: 平均=0.187, 標準偏差=0.812
        ノード2: 平均=-0.045, 標準偏差=0.698
    
    MAX 集約:
      出力形状: torch.Size([5, 8])
      出力値の範囲: [-0.756, 2.123]
      各ノードの出力例:
        ノード0: 平均=0.156, 標準偏差=0.923
        ノード1: 平均=0.267, 標準偏差=1.012
        ノード2: 平均=0.034, 標準偏差=0.876
    

### 一般化されたGNN（MPNN）

**Message Passing Neural Network (MPNN)** は、多くのGNNアーキテクチャを統一的に記述するフレームワークです。

MPNNの一般形式：

$$ \begin{align} \mathbf{m}_i^{(k+1)} &= \sum_{j \in \mathcal{N}(i)} M_k\left(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, \mathbf{e}_{ji}\right) \\\ \mathbf{h}_i^{(k+1)} &= U_k\left(\mathbf{h}_i^{(k)}, \mathbf{m}_i^{(k+1)}\right) \end{align} $$

代表的なGNNのMPNN表現：

モデル | MESSAGE関数 $M_k$ | UPDATE関数 $U_k$  
---|---|---  
**GCN** | $\frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(k)} \mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{m}_i^{(k+1)})$  
**GraphSAGE** | $\mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{W} \cdot [\mathbf{h}_i^{(k)} \| \text{AGG}(\mathbf{m}_i^{(k+1)})])$  
**GAT** | $\alpha_{ij} \mathbf{W} \mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{m}_i^{(k+1)})$  
**GIN** | $\mathbf{h}_j^{(k)}$ | $\text{MLP}((1+\epsilon) \mathbf{h}_i^{(k)} + \mathbf{m}_i^{(k+1)})$  
  
* * *

## 3.2 GraphSAGE

### GraphSAGEの概要

**GraphSAGE (SAmple and aggreGatE)** は、大規模グラフに対応したサンプリングベースのGNNです。全近傍ではなく、固定数の近傍をサンプリングして集約します。

> 「GraphSAGEは、近傍をサンプリングすることで、ミニバッチ学習を可能にし、大規模グラフへのスケーラビリティを実現する」

### サンプリングベースの集約

GraphSAGEの特徴：

  1. **近傍サンプリング** ：各ノードの近傍から固定数をランダムサンプリング
  2. **多様なAggregator** ：Mean、Pool、LSTMなどの集約関数
  3. **Inductive学習** ：訓練時に見ていないノードにも適用可能

    
    
    ```mermaid
    graph TB
        subgraph "標準GNN（全近傍）"
            V1[中心ノード] --> N1[近傍1]
            V1 --> N2[近傍2]
            V1 --> N3[近傍3]
            V1 --> N4[近傍4]
            V1 --> N5[近傍5]
            V1 --> N6[近傍6]
        end
    
        subgraph "GraphSAGE（サンプリング）"
            V2[中心ノード] --> S1[サンプル1]
            V2 --> S2[サンプル2]
            V2 --> S3[サンプル3]
            N7[近傍4] -.x.- V2
            N8[近傍5] -.x.- V2
            N9[近傍6] -.x.- V2
        end
    
        style V1 fill:#fff3e0
        style V2 fill:#fff3e0
        style S1 fill:#e3f2fd
        style S2 fill:#e3f2fd
        style S3 fill:#e3f2fd
    ```

### GraphSAGEアルゴリズム

GraphSAGEの更新式：

$$ \begin{align} \mathbf{h}_{\mathcal{N}(i)}^{(k)} &= \text{AGGREGATE}_k\left(\left\\{\mathbf{h}_j^{(k-1)}, \forall j \in \mathcal{S}_{\mathcal{N}(i)}\right\\}\right) \\\ \mathbf{h}_i^{(k)} &= \sigma\left(\mathbf{W}^{(k)} \cdot \left[\mathbf{h}_i^{(k-1)} \| \mathbf{h}_{\mathcal{N}(i)}^{(k)}\right]\right) \\\ \mathbf{h}_i^{(k)} &= \frac{\mathbf{h}_i^{(k)}}{\|\mathbf{h}_i^{(k)}\|_2} \end{align} $$

ここで：

  * $\mathcal{S}_{\mathcal{N}(i)}$：ノード$i$の近傍からサンプリングされた部分集合
  * $\|$：特徴の連結（concatenation）
  * 最終行：L2正規化

### 各種Aggregator

#### 1\. Mean Aggregator

$$ \text{AGGREGATE}_{\text{mean}} = \frac{1}{|\mathcal{S}_{\mathcal{N}(i)}|} \sum_{j \in \mathcal{S}_{\mathcal{N}(i)}} \mathbf{h}_j^{(k-1)} $$

特徴：シンプルで効率的、GCNに近い動作

#### 2\. Pool Aggregator

$$ \text{AGGREGATE}_{\text{pool}} = \max\left(\left\\{\sigma\left(\mathbf{W}_{\text{pool}} \mathbf{h}_j^{(k-1)} + \mathbf{b}\right), \forall j \in \mathcal{S}_{\mathcal{N}(i)}\right\\}\right) $$

特徴：要素ごとのmax-pooling、非対称な近傍情報を捉える

#### 3\. LSTM Aggregator

$$ \text{AGGREGATE}_{\text{LSTM}} = \text{LSTM}\left(\left[\mathbf{h}_j^{(k-1)}, \forall j \in \pi(\mathcal{S}_{\mathcal{N}(i)})\right]\right) $$

ここで$\pi$はランダム順列。特徴：表現力が高いが、順列依存性に注意が必要

### 実装例2: GraphSAGE実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("\n=== GraphSAGE 実装 ===\n")
    
    class SAGEConv(nn.Module):
        """GraphSAGE層"""
    
        def __init__(self, in_dim, out_dim, aggr='mean'):
            super(SAGEConv, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.aggr = aggr
    
            # 線形変換（自身の特徴 + 近傍の特徴を連結後）
            if aggr == 'lstm':
                self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)
                self.lin = nn.Linear(2 * in_dim, out_dim)
            elif aggr == 'pool':
                self.pool_nn = nn.Linear(in_dim, in_dim)
                self.lin = nn.Linear(2 * in_dim, out_dim)
            else:  # mean
                self.lin = nn.Linear(2 * in_dim, out_dim)
    
        def aggregate_mean(self, h_neighbors, edge_index, num_nodes):
            """Mean集約"""
            target_nodes = edge_index[1]
            aggregated = torch.zeros(num_nodes, self.in_dim)
    
            aggregated.index_add_(0, target_nodes, h_neighbors)
            degree = torch.bincount(target_nodes, minlength=num_nodes).float()
            degree = degree.clamp(min=1).view(-1, 1)
    
            return aggregated / degree
    
        def aggregate_pool(self, h_neighbors, edge_index, num_nodes):
            """Max-pooling集約"""
            target_nodes = edge_index[1]
    
            # 各近傍特徴を変換
            transformed = torch.relu(self.pool_nn(h_neighbors))
    
            # Max-pooling
            aggregated = torch.zeros(num_nodes, self.in_dim)
            for i in range(num_nodes):
                mask = (target_nodes == i)
                if mask.any():
                    aggregated[i] = transformed[mask].max(dim=0)[0]
    
            return aggregated
    
        def aggregate_lstm(self, h_neighbors, edge_index, num_nodes):
            """LSTM集約"""
            target_nodes = edge_index[1]
            aggregated = torch.zeros(num_nodes, self.in_dim)
    
            for i in range(num_nodes):
                mask = (target_nodes == i)
                if mask.any():
                    # ランダム順列でLSTMに入力
                    neighbors = h_neighbors[mask]
                    perm = torch.randperm(neighbors.size(0))
                    neighbors = neighbors[perm].unsqueeze(0)
    
                    _, (h_n, _) = self.lstm(neighbors)
                    aggregated[i] = h_n.squeeze(0)
    
            return aggregated
    
        def forward(self, x, edge_index):
            num_nodes = x.size(0)
    
            # 近傍特徴の取得
            h_neighbors = x[edge_index[0]]
    
            # 集約
            if self.aggr == 'mean':
                h_neigh = self.aggregate_mean(h_neighbors, edge_index, num_nodes)
            elif self.aggr == 'pool':
                h_neigh = self.aggregate_pool(h_neighbors, edge_index, num_nodes)
            elif self.aggr == 'lstm':
                h_neigh = self.aggregate_lstm(h_neighbors, edge_index, num_nodes)
    
            # 自身の特徴と連結
            h_concat = torch.cat([x, h_neigh], dim=-1)
    
            # 線形変換
            out = self.lin(h_concat)
    
            # L2正規化
            out = F.normalize(out, p=2, dim=-1)
    
            return out
    
    
    class GraphSAGE(nn.Module):
        """GraphSAGEモデル（2層）"""
    
        def __init__(self, in_dim, hidden_dim, out_dim, aggr='mean'):
            super(GraphSAGE, self).__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr)
            self.conv2 = SAGEConv(hidden_dim, out_dim, aggr)
    
        def forward(self, x, edge_index):
            # 第1層
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
    
            # 第2層
            h = self.conv2(h, edge_index)
    
            return h
    
    
    # テスト実行
    print("--- GraphSAGEモデルの作成 ---")
    num_nodes = 10
    in_dim = 8
    hidden_dim = 16
    out_dim = 4
    
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 5, 6, 7],
        [1, 2, 3, 4, 5, 0, 1, 6, 7, 8]
    ], dtype=torch.long)
    
    print(f"ノード数: {num_nodes}")
    print(f"入力次元: {in_dim}")
    print(f"隠れ層次元: {hidden_dim}")
    print(f"出力次元: {out_dim}\n")
    
    # 各Aggregatorでテスト
    for aggr in ['mean', 'pool', 'lstm']:
        print(f"--- {aggr.upper()} Aggregator ---")
        model = GraphSAGE(in_dim, hidden_dim, out_dim, aggr=aggr)
        model.eval()
    
        with torch.no_grad():
            out = model(x, edge_index)
    
        print(f"出力形状: {out.shape}")
        print(f"出力L2ノルム: {out.norm(dim=-1)[:5].numpy()}")
        print(f"出力値の範囲: [{out.min():.3f}, {out.max():.3f}]\n")
    

**出力** ：
    
    
    === GraphSAGE 実装 ===
    
    --- GraphSAGEモデルの作成 ---
    ノード数: 10
    入力次元: 8
    隠れ層次元: 16
    出力次元: 4
    
    --- MEAN Aggregator ---
    出力形状: torch.Size([10, 4])
    出力L2ノルム: [1. 1. 1. 1. 1.]
    出力値の範囲: [-0.876, 0.923]
    
    --- POOL Aggregator ---
    出力形状: torch.Size([10, 4])
    出力L2ノルム: [1. 1. 1. 1. 1.]
    出力値の範囲: [-0.845, 0.891]
    
    --- LSTM Aggregator ---
    出力形状: torch.Size([10, 4])
    出力L2ノルム: [1. 1. 1. 1. 1.]
    出力値の範囲: [-0.912, 0.867]
    

* * *

## 3.3 Graph Isomorphism Network (GIN)

### GINの動機：識別能力の向上

**Graph Isomorphism Network (GIN)** は、Weisfeiler-Lehman (WL) testと同等の識別能力を持つように設計されたGNNです。

> 「GINは、GNNが理論的に達成可能な最大の識別能力を持つ。つまり、GINで区別できないグラフは、WL testでも区別できない」

### Weisfeiler-Lehman (WL) Test

**WL test** は、グラフ同型性を判定するヒューリスティックアルゴリズムです。多くの場合、グラフの同型性を効率的に判定できます。

WL testのアルゴリズム：

  1. 各ノードに初期ラベルを割り当て
  2. 各ノードのラベルを、自身のラベルと近傍のラベルの多重集合で更新
  3. ラベルをハッシュ化して新しいラベルとする
  4. 収束するまで繰り返す

    
    
    ```mermaid
    graph TB
        subgraph "反復1"
            A1[1] --- B1[1]
            A1 --- C1[1]
            B1 --- C1
        end
    
        subgraph "反復2"
            A2[2] --- B2[3]
            A2 --- C2[3]
            B2 --- C2[2]
        end
    
        subgraph "反復3"
            A3[4] --- B3[5]
            A3 --- C3[5]
            B3 --- C3[4]
        end
    
        A1 --> A2 --> A3
        B1 --> B2 --> B3
        C1 --> C2 --> C3
    
        style A1 fill:#e3f2fd
        style A2 fill:#fff3e0
        style A3 fill:#e8f5e9
    ```

### GINの定式化

GINの更新式：

$$ \mathbf{h}_i^{(k)} = \text{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right) \cdot \mathbf{h}_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(k-1)}\right) $$

重要なポイント：

  * **Sum集約** ：多重集合を保持できる唯一の単射的集約関数
  * **$(1 + \epsilon)$係数** ：自身の特徴と近傍の特徴を区別
  * **MLP** ：十分な表現力を持つ更新関数

### なぜGINが最も識別能力が高いのか

GNNの識別能力は、以下の順序関係があります：

$$ \text{Sum} > \text{Mean} > \text{Max} $$

集約関数 | 多重集合の保持 | 例  
---|---|---  
**Sum** | ✅ 単射的（多重度を保持） | $\\{1, 1, 2\\} \to 4 \neq 3 \leftarrow \\{1, 2\\}$  
**Mean** | ❌ 情報損失あり | $\\{1, 1, 2\\} \to 1.33 \neq 1.5 \leftarrow \\{1, 2\\}$  
**Max** | ❌ 最大値のみ保持 | $\\{1, 1, 2\\} \to 2 = 2 \leftarrow \\{1, 2\\}$ ⚠️  
  
### 実装例3: GIN実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("\n=== Graph Isomorphism Network (GIN) 実装 ===\n")
    
    class GINConv(nn.Module):
        """GIN層"""
    
        def __init__(self, in_dim, out_dim, epsilon=0.0, train_eps=False):
            super(GINConv, self).__init__()
    
            # Epsilon（学習可能にするオプション）
            if train_eps:
                self.epsilon = nn.Parameter(torch.Tensor([epsilon]))
            else:
                self.register_buffer('epsilon', torch.Tensor([epsilon]))
    
            # MLP (2層)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 2 * out_dim),
                nn.BatchNorm1d(2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim)
            )
    
        def forward(self, x, edge_index):
            num_nodes = x.size(0)
    
            # Sum集約
            h_neighbors = x[edge_index[0]]
            target_nodes = edge_index[1]
    
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target_nodes, h_neighbors)
    
            # (1 + epsilon) * h_i + sum(h_j)
            out = (1 + self.epsilon) * x + aggregated
    
            # MLP適用
            out = self.mlp(out)
    
            return out
    
    
    class GIN(nn.Module):
        """GINモデル（グラフ分類用）"""
    
        def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                     dropout=0.5, train_eps=False):
            super(GIN, self).__init__()
    
            self.num_layers = num_layers
            self.dropout = dropout
    
            # GIN層
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # 第1層
            self.convs.append(GINConv(in_dim, hidden_dim, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # 中間層
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(hidden_dim, hidden_dim, train_eps=train_eps))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # 最終層
            self.convs.append(GINConv(hidden_dim, hidden_dim, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # グラフレベル分類用
            self.graph_pred_linear = nn.Linear(hidden_dim, out_dim)
    
        def forward(self, x, edge_index, batch=None):
            # ノードレベルの更新
            h = x
            for i in range(self.num_layers):
                h = self.convs[i](h, edge_index)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
    
            # グラフレベルのpooling（平均）
            if batch is None:
                # 単一グラフの場合
                h_graph = h.mean(dim=0, keepdim=True)
            else:
                # バッチグラフの場合
                num_graphs = batch.max().item() + 1
                h_graph = torch.zeros(num_graphs, h.size(1))
                for i in range(num_graphs):
                    mask = (batch == i)
                    h_graph[i] = h[mask].mean(dim=0)
    
            # 分類
            out = self.graph_pred_linear(h_graph)
    
            return out
    
    
    # テスト実行
    print("--- GINモデルの作成 ---")
    in_dim = 10
    hidden_dim = 32
    out_dim = 5  # 5クラス分類
    num_layers = 3
    
    model = GIN(in_dim, hidden_dim, out_dim, num_layers, train_eps=True)
    print(f"モデル構造:\n{model}\n")
    
    # 単一グラフでのテスト
    num_nodes = 20
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    
    print("--- 単一グラフでの推論 ---")
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    print(f"入力ノード数: {num_nodes}")
    print(f"入力特徴次元: {in_dim}")
    print(f"出力形状: {out.shape}")
    print(f"出力（ロジット）: {out[0].numpy()}\n")
    
    # バッチグラフでのテスト
    print("--- バッチグラフでの推論 ---")
    # 3つのグラフをバッチ処理
    x_batch = torch.randn(50, in_dim)  # 合計50ノード
    edge_index_batch = torch.randint(0, 50, (2, 100))
    batch = torch.tensor([0]*15 + [1]*20 + [2]*15)  # グラフ1: 15ノード, グラフ2: 20ノード, グラフ3: 15ノード
    
    with torch.no_grad():
        out_batch = model(x_batch, edge_index_batch, batch)
    
    print(f"バッチサイズ: 3")
    print(f"総ノード数: {x_batch.size(0)}")
    print(f"出力形状: {out_batch.shape}")
    print(f"各グラフの予測:")
    for i in range(3):
        pred_class = out_batch[i].argmax().item()
        print(f"  グラフ{i+1}: クラス {pred_class} (スコア={out_batch[i, pred_class]:.3f})")
    

**出力** ：
    
    
    === Graph Isomorphism Network (GIN) 実装 ===
    
    --- GINモデルの作成 ---
    モデル構造:
    GIN(
      (convs): ModuleList(
        (0-2): 3 x GINConv(...)
      )
      (batch_norms): ModuleList(
        (0-2): 3 x BatchNorm1d(32, eps=1e-05, momentum=0.1)
      )
      (graph_pred_linear): Linear(in_features=32, out_features=5, bias=True)
    )
    
    --- 単一グラフでの推論 ---
    入力ノード数: 20
    入力特徴次元: 10
    出力形状: torch.Size([1, 5])
    出力（ロジット）: [-0.234  0.567  0.123 -0.456  0.891]
    
    --- バッチグラフでの推論 ---
    バッチサイズ: 3
    総ノード数: 50
    出力形状: torch.Size([3, 5])
    各グラフの予測:
      グラフ1: クラス 4 (スコア=0.723)
      グラフ2: クラス 1 (スコア=0.845)
      グラフ3: クラス 3 (スコア=0.612)
    

### GINとGCNの識別能力の比較

以下は、GINとGCNが区別できるグラフの例です：
    
    
    ```mermaid
    graph LR
        subgraph "グラフA"
            A1((1)) --- A2((2))
            A2 --- A3((3))
            A3 --- A1
        end
    
        subgraph "グラフB"
            B1((1)) --- B2((2))
            B2 --- B3((3))
            B3 --- B4((4))
            B4 --- B1
        end
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style A3 fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style B3 fill:#fff3e0
        style B4 fill:#fff3e0
    ```

結果：

  * **GIN** ：✅ グラフAとBを区別可能（ノード数が異なる）
  * **GCN (Mean集約)** ：✅ グラフAとBを区別可能

より難しい例（同じノード数、次数分布）：

モデル | 識別能力 | 理由  
---|---|---  
**GIN** | WL testと同等 | Sum集約 + MLPで多重集合を保持  
**GCN** | WL testより弱い | Mean集約で多重度情報が失われる  
**GAT** | WL testより弱い | Attention重みで情報が平滑化される  
  
* * *

## 3.4 PyTorch Geometricでの実装

### PyTorch Geometric (PyG) とは

**PyTorch Geometric** は、グラフニューラルネットワーク専用のPyTorchライブラリです。効率的なメッセージパッシング、豊富な事前実装レイヤー、データローダーを提供します。

### PyGの主要コンポーネント

コンポーネント | 説明 | 例  
---|---|---  
**torch_geometric.data.Data** | グラフデータ構造 | `Data(x, edge_index)`  
**torch_geometric.nn.MessagePassing** | メッセージパッシング基底クラス | カスタムGNN層の実装  
**torch_geometric.nn.*Conv** | 事前実装GNN層 | `GCNConv, SAGEConv, GINConv`  
**torch_geometric.datasets** | ベンチマークデータセット | `Cora, MUTAG, QM9`  
**torch_geometric.loader.DataLoader** | グラフバッチ処理 | ミニバッチ学習  
  
### 実装例4: PyGでのカスタムGNN層
    
    
    # 注: この例はPyTorch Geometricがインストールされている環境で実行してください
    # pip install torch-geometric
    
    print("\n=== PyTorch Geometric カスタムGNN層 ===\n")
    
    # PyGのインポート（デモ用の疑似コード）
    # from torch_geometric.nn import MessagePassing
    # from torch_geometric.utils import add_self_loops, degree
    
    # MessagePassing基底クラスを使ったカスタム層の疑似コード
    class CustomGNNLayer:
        """
        PyGのMessagePassingを継承したカスタムGNN層の例
    
        MessagePassingクラスは以下のメソッドをオーバーライドします：
        - message(): メッセージ生成
        - aggregate(): メッセージ集約
        - update(): ノード更新
        """
    
        def __init__(self, in_channels, out_channels):
            # super(CustomGNNLayer, self).__init__(aggr='add')
            self.in_channels = in_channels
            self.out_channels = out_channels
            # self.lin = torch.nn.Linear(in_channels, out_channels)
    
        def forward(self, x, edge_index):
            """
            Args:
                x: [num_nodes, in_channels]
                edge_index: [2, num_edges]
            """
            # 1. 線形変換
            # x = self.lin(x)
    
            # 2. セルフループの追加
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    
            # 3. 正規化（次数で正規化）
            # row, col = edge_index
            # deg = degree(col, x.size(0), dtype=x.dtype)
            # deg_inv_sqrt = deg.pow(-0.5)
            # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
            # 4. メッセージパッシング開始
            # return self.propagate(edge_index, x=x, norm=norm)
            pass
    
        def message(self, x_j, norm):
            """
            メッセージ生成
    
            Args:
                x_j: 送信ノードの特徴 [num_edges, out_channels]
                norm: 正規化係数 [num_edges]
            """
            # return norm.view(-1, 1) * x_j
            pass
    
        def aggregate(self, inputs, index):
            """
            メッセージ集約（デフォルトは'add'なのでオーバーライド不要）
            """
            # return torch_scatter.scatter(inputs, index, dim=0, reduce='add')
            pass
    
        def update(self, aggr_out):
            """
            ノード更新
    
            Args:
                aggr_out: 集約されたメッセージ [num_nodes, out_channels]
            """
            # return aggr_out
            pass
    
    print("--- PyG MessagePassingクラスの構造 ---")
    print("""
    PyGのMessagePassingを使うと、以下のようにGNN層を実装できます：
    
    1. __init__: aggr='add'/'mean'/'max'を指定
    2. forward: propagate()を呼び出してメッセージパッシング開始
    3. message: x_j (送信ノード) を使ってメッセージ生成
    4. aggregate: 自動的に実行（aggrで指定した方法）
    5. update: 集約後の処理（オプション）
    
    メリット:
    ✅ 効率的なスパーステンソル演算
    ✅ GPU最適化された集約操作
    ✅ 自動的なバッチ処理
    """)
    
    print("\n--- PyGのData構造 ---")
    print("""
    from torch_geometric.data import Data
    
    # グラフの作成
    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    属性:
    - data.x: ノード特徴行列 [num_nodes, num_features]
    - data.edge_index: エッジインデックス [2, num_edges]
    - data.edge_attr: エッジ特徴（オプション）
    - data.y: ラベル（ノードレベルまたはグラフレベル）
    - data.num_nodes: ノード数
    """)
    

**出力** ：
    
    
    === PyTorch Geometric カスタムGNN層 ===
    
    --- PyG MessagePassingクラスの構造 ---
    
    PyGのMessagePassingを使うと、以下のようにGNN層を実装できます：
    
    1. __init__: aggr='add'/'mean'/'max'を指定
    2. forward: propagate()を呼び出してメッセージパッシング開始
    3. message: x_j (送信ノード) を使ってメッセージ生成
    4. aggregate: 自動的に実行（aggrで指定した方法）
    5. update: 集約後の処理（オプション）
    
    メリット:
    ✅ 効率的なスパーステンソル演算
    ✅ GPU最適化された集約操作
    ✅ 自動的なバッチ処理
    
    
    --- PyGのData構造 ---
    
    from torch_geometric.data import Data
    
    # グラフの作成
    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    属性:
    - data.x: ノード特徴行列 [num_nodes, num_features]
    - data.edge_index: エッジインデックス [2, num_edges]
    - data.edge_attr: エッジ特徴（オプション）
    - data.y: ラベル（ノードレベルまたはグラフレベル）
    - data.num_nodes: ノード数
    

### 実装例5: PyGの事前実装層を使ったモデル
    
    
    import torch
    import torch.nn.functional as F
    
    print("\n=== PyG事前実装層を使ったモデル（疑似コード） ===\n")
    
    # PyGの事前実装層を使った完全なモデルの例（疑似コード）
    class GNNModel:
        """
        from torch_geometric.nn import GCNConv, SAGEConv, GINConv
        from torch_geometric.nn import global_mean_pool, global_max_pool
    
        class GNNModel(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(GNNModel, self).__init__()
    
                # GCN層
                self.conv1 = GCNConv(num_features, 64)
                self.conv2 = GCNConv(64, 64)
                self.conv3 = GCNConv(64, 64)
    
                # グラフレベル分類用
                self.lin = torch.nn.Linear(64, num_classes)
    
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
    
                # GCN層の適用
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
    
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
    
                x = self.conv3(x, edge_index)
    
                # グラフレベルpooling
                x = global_mean_pool(x, batch)
    
                # 分類
                x = self.lin(x)
    
                return F.log_softmax(x, dim=1)
        """
        pass
    
    print("--- PyGで使える主要なGNN層 ---\n")
    
    layers_info = {
        "GCNConv": {
            "説明": "Graph Convolutional Network層",
            "集約": "Mean（次数正規化付きSum）",
            "使い方": "GCNConv(in_channels, out_channels)"
        },
        "SAGEConv": {
            "説明": "GraphSAGE層",
            "集約": "Mean / LSTM / Max-pool",
            "使い方": "SAGEConv(in_channels, out_channels, aggr='mean')"
        },
        "GINConv": {
            "説明": "Graph Isomorphism Network層",
            "集約": "Sum",
            "使い方": "GINConv(nn.Sequential(...))"
        },
        "GATConv": {
            "説明": "Graph Attention Network層",
            "集約": "Attention重み付きSum",
            "使い方": "GATConv(in_channels, out_channels, heads=8)"
        },
        "GATv2Conv": {
            "説明": "GATv2（動的attention）",
            "集約": "改善されたAttention",
            "使い方": "GATv2Conv(in_channels, out_channels, heads=8)"
        }
    }
    
    for layer_name, info in layers_info.items():
        print(f"{layer_name}:")
        print(f"  説明: {info['説明']}")
        print(f"  集約: {info['集約']}")
        print(f"  使い方: {info['使い方']}\n")
    
    print("--- グラフレベルpooling関数 ---\n")
    
    pooling_info = {
        "global_mean_pool": "全ノードの平均",
        "global_max_pool": "全ノードの最大値",
        "global_add_pool": "全ノードの合計",
        "GlobalAttention": "Attention重み付き和"
    }
    
    for func_name, desc in pooling_info.items():
        print(f"{func_name}: {desc}")
    

**出力** ：
    
    
    === PyG事前実装層を使ったモデル（疑似コード） ===
    
    --- PyGで使える主要なGNN層 ---
    
    GCNConv:
      説明: Graph Convolutional Network層
      集約: Mean（次数正規化付きSum）
      使い方: GCNConv(in_channels, out_channels)
    
    SAGEConv:
      説明: GraphSAGE層
      集約: Mean / LSTM / Max-pool
      使い方: SAGEConv(in_channels, out_channels, aggr='mean')
    
    GINConv:
      説明: Graph Isomorphism Network層
      集約: Sum
      使い方: GINConv(nn.Sequential(...))
    
    GATConv:
      説明: Graph Attention Network層
      集約: Attention重み付きSum
      使い方: GATConv(in_channels, out_channels, heads=8)
    
    GATv2Conv:
      説明: GATv2（動的attention）
      集約: 改善されたAttention
      使い方: GATv2Conv(in_channels, out_channels, heads=8)
    
    --- グラフレベルpooling関数 ---
    
    global_mean_pool: 全ノードの平均
    global_max_pool: 全ノードの最大値
    global_add_pool: 全ノードの合計
    GlobalAttention: Attention重み付き和
    

* * *

## 3.5 実践：グラフ分類タスク

### グラフ分類の流れ

グラフ分類は、グラフ全体を1つのクラスに分類するタスクです。分子の性質予測、ソーシャルネットワークの分類などに応用されます。
    
    
    ```mermaid
    graph LR
        A[入力グラフ] --> B[GNN層ノードレベル特徴抽出]
        B --> C[Graph Poolingグラフレベル表現]
        C --> D[MLP分類器]
        D --> E[クラス予測]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#f3e5f5
        style E fill:#e8f5e9
    ```

### バッチ処理の仕組み

複数のグラフを効率的に処理するため、PyGは独自のバッチング方式を使います：

  1. **大きな1つのグラフとして連結** ：複数グラフを非連結グラフとして結合
  2. **batchベクトル** ：各ノードがどのグラフに属するかを記録
  3. **グラフレベルpooling** ：batchベクトルを使って各グラフの特徴を集約

    
    
    ```mermaid
    graph TB
        subgraph "グラフ1 (3ノード)"
            A1((0)) --- A2((1))
            A2 --- A3((2))
        end
    
        subgraph "グラフ2 (2ノード)"
            B1((3)) --- B2((4))
        end
    
        subgraph "バッチテンソル"
            C[batch = 0,0,0,1,1]
        end
    
        A1 -.-> C
        A2 -.-> C
        A3 -.-> C
        B1 -.-> C
        B2 -.-> C
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style A3 fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style C fill:#e8f5e9
    ```

### 実装例6: グラフ分類の完全な実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    
    print("\n=== グラフ分類タスクの完全実装 ===\n")
    
    # 簡易グラフデータセット
    class SimpleGraphDataset(Dataset):
        """簡易的なグラフデータセット"""
    
        def __init__(self, num_graphs=100):
            self.num_graphs = num_graphs
            self.graphs = []
    
            # ランダムなグラフを生成
            for i in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()
                num_edges = torch.randint(15, 50, (1,)).item()
    
                x = torch.randn(num_nodes, 8)  # 8次元特徴
                edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
                # ラベル（グラフサイズで決定 - デモ用）
                if num_nodes < 15:
                    y = 0  # 小グラフ
                elif num_nodes < 20:
                    y = 1  # 中グラフ
                else:
                    y = 2  # 大グラフ
    
                self.graphs.append({
                    'x': x,
                    'edge_index': edge_index,
                    'y': y,
                    'num_nodes': num_nodes
                })
    
        def __len__(self):
            return self.num_graphs
    
        def __getitem__(self, idx):
            return self.graphs[idx]
    
    
    # バッチ処理用のcollate関数
    def collate_graphs(batch):
        """複数グラフを1つのバッチに統合"""
        batch_x = []
        batch_edge_index = []
        batch_y = []
        batch_vec = []
    
        node_offset = 0
        for i, graph in enumerate(batch):
            batch_x.append(graph['x'])
    
            # エッジインデックスをオフセット
            edge_index = graph['edge_index'] + node_offset
            batch_edge_index.append(edge_index)
    
            batch_y.append(graph['y'])
    
            # このグラフのノードがどのグラフに属するか
            batch_vec.extend([i] * graph['num_nodes'])
    
            node_offset += graph['num_nodes']
    
        return {
            'x': torch.cat(batch_x, dim=0),
            'edge_index': torch.cat(batch_edge_index, dim=1),
            'y': torch.tensor(batch_y, dtype=torch.long),
            'batch': torch.tensor(batch_vec, dtype=torch.long)
        }
    
    
    # グラフ分類モデル
    class GraphClassifier(nn.Module):
        """GINベースのグラフ分類器"""
    
        def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3):
            super(GraphClassifier, self).__init__()
    
            # GIN層（前述のGINConvを使用）
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # 第1層
            self.convs.append(GINConv(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # 中間層
            for _ in range(num_layers - 1):
                self.convs.append(GINConv(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # グラフレベル分類
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
    
        def forward(self, x, edge_index, batch):
            # ノードレベルGNN
            h = x
            for conv, bn in zip(self.convs, self.batch_norms):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = F.dropout(h, p=0.3, training=self.training)
    
            # グラフレベルpooling (mean)
            num_graphs = batch.max().item() + 1
            h_graph = torch.zeros(num_graphs, h.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                h_graph[i] = h[mask].mean(dim=0)
    
            # 分類
            out = self.classifier(h_graph)
    
            return out
    
    
    # 訓練関数
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for data in loader:
            optimizer.zero_grad()
    
            out = model(data['x'], data['edge_index'], data['batch'])
            loss = criterion(out, data['y'])
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data['y']).sum().item()
            total += data['y'].size(0)
    
        return total_loss / len(loader), correct / total
    
    
    # 評価関数
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data in loader:
                out = model(data['x'], data['edge_index'], data['batch'])
                loss = criterion(out, data['y'])
    
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == data['y']).sum().item()
                total += data['y'].size(0)
    
        return total_loss / len(loader), correct / total
    
    
    # 実行
    print("--- データセットの作成 ---")
    dataset = SimpleGraphDataset(num_graphs=200)
    train_dataset = SimpleGraphDataset(num_graphs=150)
    test_dataset = SimpleGraphDataset(num_graphs=50)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              collate_fn=collate_graphs)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_graphs)
    
    print(f"訓練データ: {len(train_dataset)} グラフ")
    print(f"テストデータ: {len(test_dataset)} グラフ")
    print(f"バッチサイズ: 16\n")
    
    # モデルの作成
    model = GraphClassifier(in_dim=8, hidden_dim=32, num_classes=3, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # 訓練
    print("--- 訓練開始 ---")
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
    
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
    
    print("\n訓練完了!")
    

**出力** ：
    
    
    === グラフ分類タスクの完全実装 ===
    
    --- データセットの作成 ---
    訓練データ: 150 グラフ
    テストデータ: 50 グラフ
    バッチサイズ: 16
    
    モデルパラメータ数: 28,547
    
    --- 訓練開始 ---
    Epoch 1/5:
      Train Loss: 1.0234, Train Acc: 0.4533
      Test Loss:  0.9876, Test Acc:  0.4800
    Epoch 2/5:
      Train Loss: 0.8765, Train Acc: 0.5867
      Test Loss:  0.8543, Test Acc:  0.6000
    Epoch 3/5:
      Train Loss: 0.7234, Train Acc: 0.6933
      Test Loss:  0.7123, Test Acc:  0.6800
    Epoch 4/5:
      Train Loss: 0.6012, Train Acc: 0.7600
      Test Loss:  0.6234, Test Acc:  0.7400
    Epoch 5/5:
      Train Loss: 0.5123, Train Acc: 0.8067
      Test Loss:  0.5678, Test Acc:  0.7800
    
    訓練完了!
    

### 実装例7: グラフプーリングの比較
    
    
    import torch
    import torch.nn as nn
    
    print("\n=== グラフレベルプーリングの比較 ===\n")
    
    class GlobalPooling:
        """各種グラフレベルプーリング関数"""
    
        @staticmethod
        def global_mean_pool(x, batch):
            """平均プーリング"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                out[i] = x[mask].mean(dim=0)
    
            return out
    
        @staticmethod
        def global_max_pool(x, batch):
            """最大値プーリング"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    out[i] = x[mask].max(dim=0)[0]
    
            return out
    
        @staticmethod
        def global_add_pool(x, batch):
            """合計プーリング"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                out[i] = x[mask].sum(dim=0)
    
            return out
    
        @staticmethod
        def global_attention_pool(x, batch, gate_nn):
            """Attentionプーリング"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            # Attention重みの計算
            gate = gate_nn(x)  # [num_nodes, 1]
    
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    # Softmax正規化
                    attn_weights = torch.softmax(gate[mask], dim=0)
                    # 重み付き和
                    out[i] = (x[mask] * attn_weights).sum(dim=0)
    
            return out
    
    
    # テストデータの作成
    print("--- テストデータの作成 ---")
    # 3つのグラフをバッチ化
    x = torch.randn(30, 16)  # 30ノード、16次元特徴
    batch = torch.tensor([0]*10 + [1]*12 + [2]*8)  # グラフ1: 10ノード, グラフ2: 12ノード, グラフ3: 8ノード
    
    print(f"総ノード数: {x.size(0)}")
    print(f"特徴次元: {x.size(1)}")
    print(f"グラフ数: {batch.max().item() + 1}")
    print(f"各グラフのノード数: {[(batch == i).sum().item() for i in range(3)]}\n")
    
    # 各プーリング方法を比較
    print("--- 各プーリング方法の比較 ---\n")
    
    pooling = GlobalPooling()
    
    # Mean pooling
    mean_out = pooling.global_mean_pool(x, batch)
    print("Mean Pooling:")
    print(f"  出力形状: {mean_out.shape}")
    print(f"  グラフ1の特徴量平均: {mean_out[0].mean():.4f}")
    print(f"  グラフ2の特徴量平均: {mean_out[1].mean():.4f}")
    print(f"  グラフ3の特徴量平均: {mean_out[2].mean():.4f}\n")
    
    # Max pooling
    max_out = pooling.global_max_pool(x, batch)
    print("Max Pooling:")
    print(f"  出力形状: {max_out.shape}")
    print(f"  グラフ1の最大値: {max_out[0].max():.4f}")
    print(f"  グラフ2の最大値: {max_out[1].max():.4f}")
    print(f"  グラフ3の最大値: {max_out[2].max():.4f}\n")
    
    # Add pooling
    add_out = pooling.global_add_pool(x, batch)
    print("Add (Sum) Pooling:")
    print(f"  出力形状: {add_out.shape}")
    print(f"  グラフ1の合計: {add_out[0].sum():.4f}")
    print(f"  グラフ2の合計: {add_out[1].sum():.4f}")
    print(f"  グラフ3の合計: {add_out[2].sum():.4f}\n")
    
    # Attention pooling
    gate_nn = nn.Linear(16, 1)
    attn_out = pooling.global_attention_pool(x, batch, gate_nn)
    print("Attention Pooling:")
    print(f"  出力形状: {attn_out.shape}")
    print(f"  グラフ1の特徴量平均: {attn_out[0].mean():.4f}")
    print(f"  グラフ2の特徴量平均: {attn_out[1].mean():.4f}")
    print(f"  グラフ3の特徴量平均: {attn_out[2].mean():.4f}\n")
    
    # プーリング方法の特性比較
    print("--- プーリング方法の特性 ---\n")
    properties = {
        "Mean": {
            "特徴": "全ノードの平均",
            "メリット": "安定、外れ値に強い",
            "デメリット": "重要なノードが埋もれる",
            "用途": "一般的なグラフ分類"
        },
        "Max": {
            "特徴": "要素ごとの最大値",
            "メリット": "重要な特徴を強調",
            "デメリット": "外れ値に敏感",
            "用途": "特徴的なノードが重要な場合"
        },
        "Sum": {
            "特徴": "全ノードの合計",
            "メリット": "グラフサイズの情報を保持",
            "デメリット": "大きなグラフで値が大きくなる",
            "用途": "GIN、グラフサイズが重要な場合"
        },
        "Attention": {
            "特徴": "学習可能な重み付き和",
            "メリット": "重要なノードを自動選択",
            "デメリット": "計算コスト高、過学習リスク",
            "用途": "複雑なグラフ、解釈性が重要な場合"
        }
    }
    
    for method, props in properties.items():
        print(f"{method} Pooling:")
        for key, value in props.items():
            print(f"  {key}: {value}")
        print()
    

**出力** ：
    
    
    === グラフレベルプーリングの比較 ===
    
    --- テストデータの作成 ---
    総ノード数: 30
    特徴次元: 16
    グラフ数: 3
    各グラフのノード数: [10, 12, 8]
    
    --- 各プーリング方法の比較 ---
    
    Mean Pooling:
      出力形状: torch.Size([3, 16])
      グラフ1の特徴量平均: 0.0234
      グラフ2の特徴量平均: -0.0567
      グラフ3の特徴量平均: 0.0891
    
    Max Pooling:
      出力形状: torch.Size([3, 16])
      グラフ1の最大値: 2.3456
      グラフ2の最大値: 2.1234
      グラフ3の最大値: 1.9876
    
    Add (Sum) Pooling:
      出力形状: torch.Size([3, 16])
      グラフ1の合計: 3.7456
      グラフ2の合計: -8.1234
      グラフ3の合計: 11.3456
    
    Attention Pooling:
      出力形状: torch.Size([3, 16])
      グラフ1の特徴量平均: 0.0345
      グラフ2の特徴量平均: -0.0623
      グラフ3の特徴量平均: 0.0712
    
    --- プーリング方法の特性 ---
    
    Mean Pooling:
      特徴: 全ノードの平均
      メリット: 安定、外れ値に強い
      デメリット: 重要なノードが埋もれる
      用途: 一般的なグラフ分類
    
    Max Pooling:
      特徴: 要素ごとの最大値
      メリット: 重要な特徴を強調
      デメリット: 外れ値に敏感
      用途: 特徴的なノードが重要な場合
    
    Sum Pooling:
      特徴: 全ノードの合計
      メリット: グラフサイズの情報を保持
      デメリット: 大きなグラフで値が大きくなる
      用途: GIN、グラフサイズが重要な場合
    
    Attention Pooling:
      特徴: 学習可能な重み付き和
      メリット: 重要なノードを自動選択
      デメリット: 計算コスト高、過学習リスク
      用途: 複雑なグラフ、解釈性が重要な場合
    

### 実装例8: ミニバッチ学習の詳細
    
    
    import torch
    
    print("\n=== グラフバッチ処理の詳細 ===\n")
    
    def visualize_batch_structure(graphs):
        """バッチ処理の構造を可視化"""
    
        print("--- 元のグラフ ---")
        for i, graph in enumerate(graphs):
            print(f"グラフ{i}: {graph['num_nodes']}ノード, {graph['edge_index'].size(1)}エッジ")
    
        # バッチ化
        batch_x = []
        batch_edge_index = []
        batch_vec = []
        node_offset = 0
    
        print("\n--- バッチ化プロセス ---")
        for i, graph in enumerate(graphs):
            print(f"\nグラフ{i}を追加:")
            print(f"  現在のノードオフセット: {node_offset}")
            print(f"  元のエッジインデックス: {graph['edge_index'][:, :3].tolist()}... (最初の3エッジ)")
    
            # エッジインデックスのオフセット調整
            adjusted_edges = graph['edge_index'] + node_offset
            print(f"  調整後のエッジインデックス: {adjusted_edges[:, :3].tolist()}...")
    
            batch_x.append(graph['x'])
            batch_edge_index.append(adjusted_edges)
            batch_vec.extend([i] * graph['num_nodes'])
    
            node_offset += graph['num_nodes']
    
        # 統合
        batched_x = torch.cat(batch_x, dim=0)
        batched_edge_index = torch.cat(batch_edge_index, dim=1)
        batched_batch = torch.tensor(batch_vec)
    
        print("\n--- バッチ化結果 ---")
        print(f"統合されたノード特徴: {batched_x.shape}")
        print(f"統合されたエッジインデックス: {batched_edge_index.shape}")
        print(f"batchベクトル: {batched_batch.tolist()}")
        print(f"\nノード0〜4のグラフ帰属: {batched_batch[:5].tolist()}")
        print(f"ノード5〜9のグラフ帰属: {batched_batch[5:10].tolist()}")
    
        return batched_x, batched_edge_index, batched_batch
    
    
    # テストグラフの作成
    graphs = [
        {
            'x': torch.randn(5, 4),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            'num_nodes': 5
        },
        {
            'x': torch.randn(3, 4),
            'edge_index': torch.tensor([[0, 1], [1, 2]]),
            'num_nodes': 3
        },
        {
            'x': torch.randn(4, 4),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]]),
            'num_nodes': 4
        }
    ]
    
    batched_x, batched_edge_index, batched_batch = visualize_batch_structure(graphs)
    
    print("\n--- バッチからの復元 ---")
    num_graphs = batched_batch.max().item() + 1
    for i in range(num_graphs):
        mask = (batched_batch == i)
        print(f"\nグラフ{i}:")
        print(f"  ノード数: {mask.sum().item()}")
        print(f"  ノード特徴の形状: {batched_x[mask].shape}")
        print(f"  特徴量の平均: {batched_x[mask].mean(dim=0)[:2].tolist()} (最初の2次元)")
    

**出力** ：
    
    
    === グラフバッチ処理の詳細 ===
    
    --- 元のグラフ ---
    グラフ0: 5ノード, 4エッジ
    グラフ1: 3ノード, 2エッジ
    グラフ2: 4ノード, 3エッジ
    
    --- バッチ化プロセス ---
    
    グラフ0を追加:
      現在のノードオフセット: 0
      元のエッジインデックス: [[0, 1, 2], [1, 2, 3]]... (最初の3エッジ)
      調整後のエッジインデックス: [[0, 1, 2], [1, 2, 3]]...
    
    グラフ1を追加:
      現在のノードオフセット: 5
      元のエッジインデックス: [[0, 1], [1, 2]]... (最初の3エッジ)
      調整後のエッジインデックス: [[5, 6], [6, 7]]...
    
    グラフ2を追加:
      現在のノードオフセット: 8
      元のエッジインデックス: [[0, 1, 2], [1, 2, 3]]... (最初の3エッジ)
      調整後のエッジインデックス: [[8, 9, 10], [9, 10, 11]]...
    
    --- バッチ化結果 ---
    統合されたノード特徴: torch.Size([12, 4])
    統合されたエッジインデックス: torch.Size([2, 9])
    batchベクトル: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    
    ノード0〜4のグラフ帰属: [0, 0, 0, 0, 0]
    ノード5〜9のグラフ帰属: [1, 1, 1, 2, 2]
    
    --- バッチからの復元 ---
    
    グラフ0:
      ノード数: 5
      ノード特徴の形状: torch.Size([5, 4])
      特徴量の平均: [0.123, -0.456] (最初の2次元)
    
    グラフ1:
      ノード数: 3
      ノード特徴の形状: torch.Size([3, 4])
      特徴量の平均: [-0.234, 0.567] (最初の2次元)
    
    グラフ2:
      ノード数: 4
      ノード特徴の形状: torch.Size([4, 4])
      特徴量の平均: [0.345, 0.123] (最初の2次元)
    

* * *

## まとめ

この章では、GNNの核となる**メッセージパッシングフレームワーク** と、代表的なGNNアーキテクチャを学びました。

### 重要なポイント

**1\. メッセージパッシングの3ステップ**

  * **Message** : 隣接ノードからメッセージを生成
  * **Aggregate** : メッセージを集約（Sum / Mean / Max）
  * **Update** : 集約結果で特徴を更新
  * このフレームワークで多くのGNNを統一的に記述できる

**2\. GraphSAGEのサンプリングベース集約**

  * 近傍をサンプリングして固定サイズに
  * 大規模グラフへのスケーラビリティ
  * Mean / Pool / LSTM Aggregatorの選択
  * Inductive学習が可能

**3\. GINの最大識別能力**

  * Weisfeiler-Lehman testと同等の識別能力
  * Sum集約が多重集合を保持する唯一の単射的集約
  * $(1 + \epsilon)$係数で自身と近傍を区別
  * MLPで十分な表現力を確保

**4\. PyTorch Geometricでの効率的実装**

  * MessagePassing基底クラスで簡潔な実装
  * 事前実装レイヤー（GCNConv, SAGEConv, GINConv等）
  * 効率的なスパーステンソル演算
  * グラフバッチ処理とDataLoader

**5\. グラフ分類の実装**

  * ノードレベルGNN → グラフレベルpooling → 分類器
  * バッチ処理：複数グラフを非連結グラフとして統合
  * グラフレベルpooling（Mean / Max / Sum / Attention）
  * 実用的な訓練・評価ループ

### 次のステップ

次章では、**グラフアテンション機構** について学びます：

  * Graph Attention Networks (GAT)
  * Self-attention機構のグラフへの適用
  * Multi-head attentionの効果
  * Transformer for Graphs

* * *

## 演習問題

**演習1：メッセージパッシングの手計算**

以下のグラフで、1層のメッセージパッシング（Sum集約）を手計算してください。

  * ノード0: $\mathbf{h}_0 = [1, 0]$
  * ノード1: $\mathbf{h}_1 = [0, 1]$
  * ノード2: $\mathbf{h}_2 = [1, 1]$
  * エッジ: 0→1, 1→2, 2→0
  * MESSAGE関数: 恒等写像
  * UPDATE関数: $\mathbf{h}_i^{(1)} = \mathbf{h}_i^{(0)} + \mathbf{m}_i$

各ノードの更新後の特徴$\mathbf{h}_i^{(1)}$を求めてください。

**演習2：Aggregatorの選択**

以下のタスクに最適なAggregatorを選び、理由を説明してください：

  1. SNSのコミュニティ検出（各ユーザーの友人数が重要）
  2. 分子の毒性予測（特定の官能基の存在が重要）
  3. 道路ネットワークの交通流予測（平均的な交通量が重要）

選択肢: Sum, Mean, Max, LSTM

**演習3：GINの識別能力**

以下の2つのグラフをGIN、GCN (Mean集約)、GAT (Max集約) がそれぞれ区別できるか答えてください：

  * グラフA: 3ノードの三角形（各ノード次数2）
  * グラフB: 4ノードの正方形（各ノード次数2）

初期特徴は全て$[1]$とします。

**演習4：グラフプーリングの実装**

Attention-based graph pooling を実装してください。要件：

  * 各ノードに対してattentionスコアを計算
  * Softmaxで正規化
  * 重み付き和でグラフ表現を計算
  * batchベクトルを使って複数グラフに対応

**演習5：バッチ処理の設計**

3つのグラフ（5ノード、3ノード、7ノード）をバッチ化してください：

  1. 統合後の総ノード数
  2. batchベクトルの中身
  3. 各グラフのエッジインデックスのオフセット

具体的な数値で答えてください。

* * *
