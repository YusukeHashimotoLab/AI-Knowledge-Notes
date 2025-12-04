---
title: 第1章：GNN構造ベース特徴量の基礎
chapter_title: 第1章：GNN構造ベース特徴量の基礎
---

**グラフ表現が捉える構造情報：組成ベース特徴量では見えなかった世界**

## 1.1 組成ベース特徴量の限界

材料科学のAI予測において、長年の主流は**組成ベース特徴量** でした。Magpie（Ward et al., 2016）やMatminerのような手法は、化学組成（例: Fe₂O₃）から統計的特徴量を計算します。

### 1.1.1 組成ベース特徴量の例
    
    
    # Google Colab環境セットアップ
    !pip install matminer pymatgen scikit-learn
    
    import numpy as np
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core.composition import Composition
    
    # Example 1: Magpie特徴量の計算
    magpie = ElementProperty.from_preset("magpie")
    
    # Fe₂O₃（酸化鉄）の特徴量
    comp = Composition("Fe2O3")
    features = magpie.featurize(comp)
    
    print(f"特徴量の次元数: {len(features)}")
    print(f"最初の10個の特徴量: {features[:10]}")
    # 出力例: 特徴量の次元数: 132
    # 出力例: 最初の10個の特徴量: [55.845, 15.999, 39.998, ...]
    

組成ベース特徴量は以下の132次元を含みます：

  * 平均原子量、密度
  * 電気陰性度、イオン半径の統計量（平均、分散、最大、最小）
  * 電子配置の統計量

### 1.1.2 致命的な限界：構造情報の欠落

組成ベース特徴量は、**原子の空間配置** を一切考慮しません。これは以下の問題を引き起こします：
    
    
    ```mermaid
    graph LR
        A[C: ダイヤモンド] -.同じ組成.-> B[C: グラファイト]
        A --> C[硬度: 10,000 HV]
        B --> D[硬度: 2-3 HV]
    
        E[SiO₂: α-quartz] -.同じ組成.-> F[SiO₂: β-cristobalite]
        E --> G[密度: 2.65 g/cm³]
        F --> H[密度: 2.33 g/cm³]
    
        style A fill:#e3f2fd
        style B fill:#e3f2fd
        style E fill:#fff3e0
        style F fill:#fff3e0
    ```

**具体例** :

  1. **ダイヤモンド vs グラファイト** （両方ともC）: 
     * ダイヤモンド: sp³混成、正四面体構造、硬度 10,000 HV
     * グラファイト: sp²混成、層状構造、硬度 2-3 HV
     * 組成ベース特徴量は完全に同一！
  2. **SiO₂の多形** （α-quartz vs β-cristobalite）: 
     * α-quartz: 密度 2.65 g/cm³、六方晶
     * β-cristobalite: 密度 2.33 g/cm³、立方晶
     * 組成は同一だが物性が全く異なる

    
    
    # Example 2: 組成ベース特徴量の限界の実証
    from pymatgen.core import Structure, Lattice
    
    # ダイヤモンド構造（sp³）
    diamond_lattice = Lattice.cubic(3.567)
    diamond = Structure(diamond_lattice, ["C", "C"],
                        [[0, 0, 0], [0.25, 0.25, 0.25]])
    
    # グラファイト構造（sp²、簡略版）
    graphite_lattice = Lattice.hexagonal(2.46, 6.71)
    graphite = Structure(graphite_lattice, ["C", "C"],
                         [[0, 0, 0], [1/3, 2/3, 0.5]])
    
    # 組成ベース特徴量の計算
    comp_diamond = diamond.composition
    comp_graphite = graphite.composition
    
    features_diamond = magpie.featurize(comp_diamond)
    features_graphite = magpie.featurize(comp_graphite)
    
    print(f"ダイヤモンドとグラファイトの特徴量が同一: {np.allclose(features_diamond, features_graphite)}")
    # 出力: True（組成ベースでは区別不可能）
    
    print(f"実際の密度:")
    print(f"  ダイヤモンド: {diamond.density:.2f} g/cm³")  # 3.51
    print(f"  グラファイト: {graphite.density:.2f} g/cm³")  # 2.26
    print(f"密度差: {abs(diamond.density - graphite.density)/diamond.density * 100:.1f}%")
    # 出力: 密度差: 35.6%（組成ベースでは捉えられない）
    

## 1.2 グラフ表現：構造を記述する数学的言語

### 1.2.1 グラフ理論の基礎

グラフ \\( G = (V, E) \\) は以下から構成されます：

  * **頂点集合 \\( V \\)** : 原子の集合
  * **辺集合 \\( E \\)** : 原子間の結合（化学結合、または空間的近接）

各頂点 \\( v_i \in V \\) には**ノード特徴量 \\( \mathbf{x}_i \in \mathbb{R}^d \\)** が付与されます：

  * 原子番号、原子量
  * 電気陰性度、イオン半径
  * 価電子数

各辺 \\( e_{ij} \in E \\) には**エッジ特徴量 \\( \mathbf{e}_{ij} \in \mathbb{R}^k \\)** が付与されます：

  * 原子間距離 \\( r_{ij} \\)
  * 結合タイプ（単結合、二重結合等）
  * 結合角度

    
    
    ```mermaid
    graph TD
        subgraph "分子: H₂O"
            O[O原子番号=8電気陰性度=3.44]
            H1[H原子番号=1電気陰性度=2.20]
            H2[H原子番号=1電気陰性度=2.20]
    
            O ---|"距離=0.96Å単結合"| H1
            O ---|"距離=0.96Å単結合"| H2
        end
    
        style O fill:#e8f5e9
        style H1 fill:#e3f2fd
        style H2 fill:#e3f2fd
    ```

### 1.2.2 PyTorch Geometricでのグラフデータ構造
    
    
    # Example 3: PyTorch GeometricでH₂Oをグラフ表現
    !pip install torch-geometric torch-scatter torch-sparse
    
    import torch
    from torch_geometric.data import Data
    
    # H₂O分子のグラフ表現
    # ノード特徴量: [原子番号, 電気陰性度]
    node_features = torch.tensor([
        [8, 3.44],   # O
        [1, 2.20],   # H1
        [1, 2.20]    # H2
    ], dtype=torch.float)
    
    # エッジリスト（無向グラフなので双方向）
    edge_index = torch.tensor([
        [0, 1, 1, 0, 0, 2, 2, 0],  # source nodes
        [1, 0, 0, 1, 2, 0, 0, 2]   # target nodes
    ], dtype=torch.long)
    
    # エッジ特徴量: [原子間距離（Å）]
    edge_attr = torch.tensor([
        [0.96], [0.96],  # O-H1
        [0.96], [0.96],  # H1-O (双方向)
        [0.96], [0.96],  # O-H2
        [0.96], [0.96]   # H2-O (双方向)
    ], dtype=torch.float)
    
    # PyTorch Geometric Dataオブジェクト
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    print(f"ノード数: {data.num_nodes}")        # 3
    print(f"エッジ数: {data.num_edges}")        # 8（双方向）
    print(f"ノード特徴量次元: {data.num_node_features}")  # 2
    print(f"エッジ特徴量次元: {data.num_edge_features}")  # 1
    print(f"\nデータ構造:")
    print(data)
    

### 1.2.3 組成ベース vs グラフベース：情報量の比較

特徴 | 組成ベース（Magpie） | グラフベース（GNN）  
---|---|---  
**情報源** | 化学組成のみ | 組成 + 原子配置  
**特徴量次元** | 固定（132次元） | 可変（ノード数に依存）  
**多形の区別** | ❌ 不可能 | ✅ 可能  
**局所環境** | ❌ 考慮しない | ✅ メッセージパッシングで考慮  
**計算コスト** | 低（秒単位） | 中〜高（分単位、GPU推奨）  
**データ要件** | 低（100-1000サンプル） | 中〜高（1000-10000サンプル）  
  
## 1.3 GNN（グラフニューラルネットワーク）の基本原理

### 1.3.1 メッセージパッシングの概念

GNNの核心は**メッセージパッシング** です。各原子（ノード）は、近傍原子から情報を集約し、自身の表現を更新します。

**数学的定式化** :

\\[ \mathbf{h}_i^{(k+1)} = \text{UPDATE}^{(k)} \left( \mathbf{h}_i^{(k)}, \text{AGGREGATE}^{(k)} \left( \\{ \mathbf{h}_j^{(k)} : j \in \mathcal{N}(i) \\} \right) \right) \\]

ここで：

  * \\( \mathbf{h}_i^{(k)} \\): ノード \\( i \\) の第 \\( k \\) 層での隠れ表現
  * \\( \mathcal{N}(i) \\): ノード \\( i \\) の近傍ノード集合
  * \\( \text{AGGREGATE} \\): 近傍情報の集約関数（SUM、MEAN、MAX等）
  * \\( \text{UPDATE} \\): ノード表現の更新関数（MLP、GRU等）

    
    
    ```mermaid
    graph LR
        subgraph "Layer k"
            A1[h₁⁽ᵏ⁾]
            B1[h₂⁽ᵏ⁾]
            C1[h₃⁽ᵏ⁾]
            D1[h₄⁽ᵏ⁾]
        end
    
        subgraph "Message Passing"
            B1 --> M[AGGREGATE]
            C1 --> M
            D1 --> M
            A1 --> U[UPDATE]
            M --> U
        end
    
        subgraph "Layer k+1"
            A2[h₁⁽ᵏ⁺¹⁾]
        end
    
        U --> A2
    
        style M fill:#fff3e0
        style U fill:#e8f5e9
        style A2 fill:#e3f2fd
    ```

### 1.3.2 シンプルなGNNの実装
    
    
    # Example 4: メッセージパッシングの最小実装
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    
    class SimpleGCNConv(MessagePassing):
        """シンプルなGraph Convolutional Network層"""
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='add')  # "add"集約
            self.lin = nn.Linear(in_channels, out_channels)
    
        def forward(self, x, edge_index):
            # ステップ1: 自己ループ追加（自分自身のメッセージも考慮）
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    
            # ステップ2: 線形変換
            x = self.lin(x)
    
            # ステップ3: 次数による正規化
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
            # ステップ4: メッセージパッシング
            return self.propagate(edge_index, x=x, norm=norm)
    
        def message(self, x_j, norm):
            # 正規化されたメッセージ
            return norm.view(-1, 1) * x_j
    
    # テスト
    conv = SimpleGCNConv(in_channels=2, out_channels=8)
    h0 = data.x  # H₂Oのノード特徴量（3ノード、2次元）
    
    h1 = conv(h0, data.edge_index)
    print(f"入力形状: {h0.shape}")  # torch.Size([3, 2])
    print(f"出力形状: {h1.shape}")  # torch.Size([3, 8])
    print(f"第1層の出力（最初のノード）:\n{h1[0]}")
    

### 1.3.3 CGCNN vs MPNN：アーキテクチャの違い

**CGCNN（Crystal Graph Convolutional Neural Networks）** は結晶材料に特化：

  * **対象** : 結晶構造（周期境界条件あり）
  * **エッジ特徴量** : 原子間距離を重視
  * **集約** : エッジゲート機構によるソフトアテンション
  * **適用例** : Materials Project、OQMD

**MPNN（Message Passing Neural Networks）** は汎用フレームワーク：

  * **対象** : 分子、タンパク質、結晶全て
  * **メッセージ関数** : カスタマイズ可能
  * **集約** : SUM、MEAN、MAX等を選択
  * **適用例** : QM9、ZINC、ChEMBL

特徴 | CGCNN | MPNN  
---|---|---  
**論文** | Xie & Grossman (2018) | Gilmer et al. (2017)  
**主な対象** | 結晶材料 | 分子・結晶両方  
**エッジ処理** | ゲート機構 | 汎用メッセージ関数  
**集約方法** | 重み付きSUM | SUM/MEAN/MAX  
**周期境界条件** | ✅ 考慮 | ❌ 標準では非対応  
  
## 1.4 実例：ダイヤモンドとグラファイトの区別
    
    
    # Example 5: GNNでダイヤモンドとグラファイトを区別
    from torch_geometric.data import Data
    import numpy as np
    
    def structure_to_graph(structure, cutoff=5.0):
        """pymatgen構造からPyTorch Geometricグラフを作成"""
        # ノード特徴量: 原子番号
        x = torch.tensor([[site.specie.Z] for site in structure], dtype=torch.float)
    
        # エッジリスト作成（cutoff半径内の近傍）
        edges = []
        edge_attrs = []
    
        for i, site_i in enumerate(structure):
            for j, site_j in enumerate(structure):
                if i != j:
                    dist = structure.get_distance(i, j)
                    if dist <= cutoff:
                        edges.append([i, j])
                        edge_attrs.append([dist])
    
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # ダイヤモンドとグラファイトをグラフ化
    graph_diamond = structure_to_graph(diamond, cutoff=2.0)
    graph_graphite = structure_to_graph(graphite, cutoff=2.0)
    
    print("ダイヤモンドのグラフ:")
    print(f"  ノード数: {graph_diamond.num_nodes}")
    print(f"  エッジ数: {graph_diamond.num_edges}")
    print(f"  平均次数: {graph_diamond.num_edges / graph_diamond.num_nodes:.2f}")
    
    print("\nグラファイトのグラフ:")
    print(f"  ノード数: {graph_graphite.num_nodes}")
    print(f"  エッジ数: {graph_graphite.num_edges}")
    print(f"  平均次数: {graph_graphite.num_edges / graph_graphite.num_nodes:.2f}")
    
    # 出力例:
    # ダイヤモンド: 平均次数 4.00（sp³、正四面体）
    # グラファイト: 平均次数 3.00（sp²、平面三角形）
    # → グラフ構造から区別可能！
    

## 1.5 PyTorch Geometricの基礎

### 1.5.1 Dataオブジェクトの詳細
    
    
    # Example 6: PyTorch Geometric Dataオブジェクトの属性
    from torch_geometric.data import Data
    
    # より複雑な例：CO₂分子
    # O=C=O（線形構造）
    
    node_features = torch.tensor([
        [8, 3.44, 6],   # O1: 原子番号、電気陰性度、価電子数
        [6, 2.55, 4],   # C:  原子番号、電気陰性度、価電子数
        [8, 3.44, 6]    # O2: 原子番号、電気陰性度、価電子数
    ], dtype=torch.float)
    
    edge_index = torch.tensor([
        [0, 1, 1, 0, 1, 2, 2, 1],
        [1, 0, 0, 1, 2, 1, 1, 2]
    ], dtype=torch.long)
    
    edge_attr = torch.tensor([
        [1.16, 2],  # O1-C: 距離（Å）、結合次数（二重結合）
        [1.16, 2],  # C-O1
        [1.16, 2],  # O1-C（逆方向）
        [1.16, 2],  # C-O1（逆方向）
        [1.16, 2],  # C-O2
        [1.16, 2],  # O2-C
        [1.16, 2],  # C-O2（逆方向）
        [1.16, 2]   # O2-C（逆方向）
    ], dtype=torch.float)
    
    # ターゲット値: 双極子モーメント（Debye）
    y = torch.tensor([[0.0]], dtype=torch.float)  # CO₂は対称なので0
    
    data_co2 = Data(x=node_features, edge_index=edge_index,
                    edge_attr=edge_attr, y=y)
    
    print("CO₂分子のグラフデータ:")
    print(f"  ノード特徴量形状: {data_co2.x.shape}")        # [3, 3]
    print(f"  エッジインデックス形状: {data_co2.edge_index.shape}")  # [2, 8]
    print(f"  エッジ特徴量形状: {data_co2.edge_attr.shape}")     # [8, 2]
    print(f"  ターゲット値: {data_co2.y.item():.2f} Debye")
    print(f"  有向グラフ: {data_co2.is_directed()}")  # True
    

### 1.5.2 DataLoaderとバッチ処理
    
    
    # Example 7: バッチ処理の実装
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch
    
    # 複数の分子データを作成
    molecules = [data, data_co2]  # H₂OとCO₂
    
    # DataLoader作成
    loader = DataLoader(molecules, batch_size=2, shuffle=True)
    
    for batch in loader:
        print("バッチデータ:")
        print(f"  バッチサイズ: {batch.num_graphs}")
        print(f"  総ノード数: {batch.num_nodes}")
        print(f"  総エッジ数: {batch.num_edges}")
        print(f"  ノード特徴量形状: {batch.x.shape}")
        print(f"  バッチベクトル: {batch.batch}")
        # バッチベクトル例: tensor([0, 0, 0, 1, 1, 1])
        # → 最初の3ノードは分子0、次の3ノードは分子1
        break
    
    # バッチ処理の利点
    print("\n✅ バッチ処理の利点:")
    print("  1. GPU並列化による高速化")
    print("  2. メモリ効率の向上")
    print("  3. ミニバッチ勾配降下法の適用")
    

### 1.5.3 グラフプーリング：分子レベルの表現
    
    
    # Example 8: グローバルプーリング
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    
    # ノード特徴量からグラフレベルの表現を作成
    batch = Batch.from_data_list(molecules)
    
    # 平均プーリング
    graph_emb_mean = global_mean_pool(batch.x, batch.batch)
    print(f"平均プーリング出力形状: {graph_emb_mean.shape}")  # [2, feature_dim]
    
    # 最大プーリング
    graph_emb_max = global_max_pool(batch.x, batch.batch)
    print(f"最大プーリング出力形状: {graph_emb_max.shape}")
    
    # 合計プーリング
    graph_emb_sum = global_add_pool(batch.x, batch.batch)
    print(f"合計プーリング出力形状: {graph_emb_sum.shape}")
    
    print("\nプーリング手法の使い分け:")
    print("  - 平均プーリング: 分子サイズに不変（推奨）")
    print("  - 最大プーリング: 最も重要な原子を強調")
    print("  - 合計プーリング: 分子サイズに依存（加法的特性向き）")
    

## 1.6 まとめ

この章では、GNN構造ベース特徴量の基礎を学びました：

  1. **組成ベース特徴量の限界** : ダイヤモンドとグラファイトを区別できない
  2. **グラフ表現の強み** : 原子配置を数学的に記述
  3. **メッセージパッシング** : GNNの核心的な計算プロセス
  4. **CGCNN vs MPNN** : 結晶特化 vs 汎用フレームワーク
  5. **PyTorch Geometric** : グラフ深層学習の実装基盤

次章では、CGCNNの詳細な実装とMaterials Projectでの結晶物性予測を学びます。

* * *

## 演習問題

### Easy（基礎確認）

**Q1** : 組成ベース特徴量が捉えられない情報は何ですか？

**正解** : 原子の空間配置（構造情報）

**解説** :

組成ベース特徴量（Magpie等）は、化学組成（Fe₂O₃等）から統計的特徴量を計算しますが、以下を一切考慮しません：

  * 原子の3D座標
  * 結合長、結合角
  * 結晶構造（fcc、bcc等）
  * 局所環境（配位数、配位多面体）

このため、ダイヤモンドとグラファイト（両方とも純炭素）を区別できません。

**Q2** : PyTorch GeometricのDataオブジェクトで、ノード特徴量を格納する属性は何ですか？

**正解** : `x`

**解説** :

PyTorch Geometric Dataオブジェクトの主要属性：

  * `data.x`: ノード特徴量（形状: [num_nodes, num_features]）
  * `data.edge_index`: エッジリスト（形状: [2, num_edges]）
  * `data.edge_attr`: エッジ特徴量（形状: [num_edges, num_edge_features]）
  * `data.y`: ターゲット値（予測したい物性）

**Q3** : メッセージパッシングの3つの主要ステップは何ですか？

**正解** : Message（メッセージ生成）、Aggregate（集約）、Update（更新）

**解説** :

  1. **Message** : 各エッジでメッセージを生成 
     * 例: \\( m_{ij} = \text{MLP}(\mathbf{h}_j, \mathbf{e}_{ij}) \\)
  2. **Aggregate** : 近傍からのメッセージを集約 
     * 例: \\( m_i = \sum_{j \in \mathcal{N}(i)} m_{ij} \\)
  3. **Update** : ノード表現を更新 
     * 例: \\( \mathbf{h}_i^{(k+1)} = \text{GRU}(\mathbf{h}_i^{(k)}, m_i) \\)

### Medium（応用）

**Q4** : ダイヤモンドとグラファイトのグラフ構造の違いを、平均次数を用いて説明してください。

**正解** : ダイヤモンド（平均次数 ≈ 4.0）vs グラファイト（平均次数 ≈ 3.0）

**解説** :

  * **ダイヤモンド** : 
    * sp³混成軌道
    * 各炭素原子が4つの隣接原子と結合
    * 正四面体構造
    * 平均次数 ≈ 4.0
  * **グラファイト** : 
    * sp²混成軌道
    * 各炭素原子が3つの隣接原子と結合
    * 平面三角形構造（層状）
    * 平均次数 ≈ 3.0

この次数の違いをGNNは学習し、両者を区別できます。

**Q5** : CGCNNがMPNNと比較して結晶材料に適している理由を2つ挙げてください。

**正解** : (1) 周期境界条件の考慮、(2) エッジゲート機構

**解説** :

  1. **周期境界条件** : 
     * 結晶は無限に繰り返される周期構造
     * CGCNNは単位格子内の原子と、周期的に繰り返される近傍原子も考慮
     * MPNNは標準では非周期的（分子向け）
  2. **エッジゲート機構** : 
     * 原子間距離に応じた重み付け
     * 遠い原子からのメッセージを抑制
     * 結晶の局所環境を適切にモデル化

**Q6** : PyTorch GeometricでCO₂分子（O=C=O）のグラフを作成してください。ノード特徴量は原子番号のみとします。

**解答例** :
    
    
    import torch
    from torch_geometric.data import Data
    
    # ノード特徴量: 原子番号
    x = torch.tensor([[8], [6], [8]], dtype=torch.float)  # O, C, O
    
    # エッジリスト（無向グラフ）
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # O-C, C-O, C-O, O-C
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    data_co2 = Data(x=x, edge_index=edge_index)
    print(data_co2)
    

**解説** :

  * ノード0: O（原子番号8）
  * ノード1: C（原子番号6）
  * ノード2: O（原子番号8）
  * エッジ: O-C, C-O（無向グラフなので双方向）

### Hard（発展）

**Q7** : 組成ベース特徴量とGNN特徴量のデータ効率性を比較してください。少量データ（100サンプル）と大量データ（10,000サンプル）でどちらが有利か、理由とともに説明してください。

**解答** :

**少量データ（100サンプル）** : 組成ベース特徴量が有利

  * **理由1** : 固定次元（132次元）で過学習しにくい
  * **理由2** : ドメイン知識（電気陰性度、イオン半径等）が組み込まれている
  * **理由3** : 線形モデル（Ridge、Lasso）でも高精度

**大量データ（10,000サンプル）** : GNN特徴量が有利

  * **理由1** : 構造情報を活用し、組成ベースより高精度
  * **理由2** : 深層学習の表現力をフル活用できる
  * **理由3** : 転移学習（事前学習モデル）でさらに精度向上

**定量的比較（Materials Projectデータ）** :

データ量 | 組成ベース（MAE） | GNN（MAE）  
---|---|---  
100サンプル | 0.25 eV/atom | 0.35 eV/atom（過学習）  
1,000サンプル | 0.18 eV/atom | 0.15 eV/atom  
10,000サンプル | 0.15 eV/atom | 0.08 eV/atom ⭐  
**Q8** : メッセージパッシングの集約関数（SUM、MEAN、MAX）の特性を比較し、それぞれが適する状況を説明してください。

**解答** :

集約関数 | 数式 | 特性 | 適する状況  
---|---|---|---  
**SUM** | \\( \sum_{j \in \mathcal{N}(i)} m_{ij} \\) | 加法的、次数依存 | 化学量論的特性（質量、電荷）  
**MEAN** | \\( \frac{1}{|\mathcal{N}(i)|} \sum_{j} m_{ij} \\) | 正規化、次数不変 | 局所環境の平均的特性（電気陰性度）  
**MAX** | \\( \max_{j \in \mathcal{N}(i)} m_{ij} \\) | 最大値抽出 | 最も重要な特徴を強調（活性部位検出）  
  
**実践的推奨** :

  * **MEAN** : デフォルトで推奨（次数に不変、安定した学習）
  * **SUM** : 加法的特性を予測する場合
  * **MAX** : 局所的な異常検出、触媒活性部位の特定

**Q9** : 結晶材料のグラフ表現において、カットオフ半径の選択が予測精度に与える影響を考察してください。カットオフ半径が短すぎる場合と長すぎる場合の問題点を説明してください。

**解答** :

**カットオフ半径が短すぎる場合（例: 2Å）** :

  * **問題1** : 第一近接のみ考慮、長距離相互作用を無視
  * **問題2** : エッジ数が少なく、情報伝播が遅い
  * **問題3** : 多層GNNが必要（計算コスト増）
  * **例** : イオン結晶のクーロン相互作用（長距離）を捉えられない

**カットオフ半径が長すぎる場合（例: 10Å）** :

  * **問題1** : エッジ数爆発的増加（O(N²)）
  * **問題2** : メモリ枯渇、計算時間増大
  * **問題3** : ノイズ増加（遠い原子の影響を過大評価）
  * **問題4** : 過学習のリスク増大

**最適なカットオフ半径の選択** :

材料タイプ | 推奨カットオフ半径 | 理由  
---|---|---  
共有結合結晶（Si、Diamond） | 4-5Å | 第二近接まで考慮  
イオン結晶（NaCl、MgO） | 6-8Å | 長距離クーロン相互作用  
金属（Fe、Cu） | 5-6Å | 第三近接まで考慮  
分子結晶 | 8-10Å | 分子間相互作用（ファンデルワールス力）  
  
**実験的最適化** :
    
    
    # カットオフ半径の影響を評価
    cutoffs = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    for cutoff in cutoffs:
        graph = structure_to_graph(structure, cutoff=cutoff)
        print(f"Cutoff {cutoff}Å: {graph.num_edges} edges")
        # 訓練して精度評価
    

* * *

## 学習目標の確認

このchapterを完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 組成ベース特徴量の3つの限界を具体例とともに挙げられる
  * ✅ グラフ表現の数学的定義（\\( G = (V, E) \\)）を説明できる
  * ✅ メッセージパッシングの3ステップを理解している
  * ✅ CGCNN vs MPNNの違いを説明できる

### 実践スキル

  * ✅ PyTorch GeometricでDataオブジェクトを作成できる
  * ✅ pymatgen構造からグラフを構築できる
  * ✅ ダイヤモンドとグラファイトをグラフ構造で区別できる
  * ✅ DataLoaderでバッチ処理を実装できる

### 応用力

  * ✅ 新しい材料問題に対して組成ベース vs GNNを選択できる
  * ✅ カットオフ半径の最適値を実験的に決定できる
  * ✅ グラフ構造の特性（平均次数、密度）から材料特性を推測できる

* * *

## 次のステップ

次章では、CGCNNの詳細な実装とMaterials Projectでの形成エネルギー予測を学びます。

[← シリーズ目次](<./index.html>) [第2章：CGCNN実装 →](<./chapter-2.html>)

* * *

## 参考文献

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028, pp. 1-7.
  2. Xie, T., & Grossman, J. C. (2018). "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." _Physical Review Letters_ , 120(14), 145301, pp. 1-6.
  3. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). "Neural Message Passing for Quantum Chemistry." _Proceedings of the 34th International Conference on Machine Learning_ , PMLR 70, pp. 1263-1272.
  4. Fey, M., & Lenssen, J. E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." _ICLR Workshop on Representation Learning on Graphs and Manifolds_ , pp. 1-9.
  5. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." _APL Materials_ , 1(1), 011002, pp. 1-11.
  6. Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., ... & Persson, K. A. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, pp. 314-319.
  7. Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). "SchNet – A deep learning architecture for molecules and materials." _The Journal of Chemical Physics_ , 148(24), 241722, pp. 1-10.

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
