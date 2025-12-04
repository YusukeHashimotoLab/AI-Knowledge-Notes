---
title: Chapter
chapter_title: Chapter
subtitle: 
reading_time: 20-25分
difficulty: 初級
code_examples: 0
exercises: 0
---

# 第4章：実践編 - GNN + 次元削減による材料マッピング

## 概要

本章では、これまで学んだGNNによる表現学習（第3章）と次元削減手法（第2章）を組み合わせた、実践的な材料空間マッピングシステムを構築します。Materials Project APIから実データを取得し、エンドツーエンドのパイプラインを実装します。

### 学習目標

  * Materials Project APIを用いた実データ収集ができる
  * GNN学習パイプラインを構築できる
  * 学習済みGNN埋め込みを次元削減で可視化できる
  * インタラクティブな材料探索システムを実装できる
  * 実際の材料設計タスクに応用できる

## 4.1 環境構築とデータ収集

### コード例1: 必要なライブラリのインストール
    
    
    # 必要なライブラリのインストール（初回のみ実行）
    """
    !pip install pymatgen
    !pip install mp-api
    !pip install torch torchvision torchaudio
    !pip install torch-geometric
    !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    !pip install umap-learn
    !pip install plotly
    !pip install bokeh
    !pip install scikit-learn
    !pip install pandas matplotlib seaborn
    !pip install tqdm
    """
    
    # インポート
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Dataset, DataLoader
    
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    
    from tqdm import tqdm
    import json
    import pickle
    from pathlib import Path
    
    print("All libraries imported successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    

### コード例2: Materials Project APIの設定とデータ取得
    
    
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    import warnings
    warnings.filterwarnings('ignore')
    
    # APIキーの設定（環境変数または直接指定）
    # MP_API_KEY = "your_api_key_here"  # https://next-gen.materialsproject.org/api で取得
    # 注: 実際のプロジェクトでは環境変数から読み込むことを推奨
    
    def fetch_materials_data(api_key, criteria, properties, max_materials=1000):
        """
        Materials Project APIから材料データを取得
    
        Parameters:
        -----------
        api_key : str
            Materials Project APIキー
        criteria : dict
            検索条件
        properties : list
            取得する特性のリスト
        max_materials : int
            取得する最大材料数
    
        Returns:
        --------
        materials_df : pd.DataFrame
            材料データ
        structures : list
            結晶構造のリスト
        """
        with MPRester(api_key) as mpr:
            # データの取得
            docs = mpr.materials.summary.search(
                **criteria,
                fields=properties,
                num_chunks=10,
                chunk_size=100
            )
    
            # 最大数に制限
            docs = docs[:max_materials]
    
            print(f"取得した材料数: {len(docs)}")
    
            # DataFrameに変換
            data_dict = {prop: [] for prop in properties}
            data_dict['material_id'] = []
            data_dict['formula'] = []
            structures = []
    
            for doc in tqdm(docs, desc="データ処理中"):
                try:
                    data_dict['material_id'].append(doc.material_id)
                    data_dict['formula'].append(str(doc.formula_pretty))
    
                    for prop in properties:
                        if prop == 'structure':
                            structures.append(doc.structure)
                        else:
                            value = getattr(doc, prop, None)
                            data_dict[prop].append(value)
    
                except Exception as e:
                    print(f"Error processing {doc.material_id}: {e}")
                    continue
    
            materials_df = pd.DataFrame(data_dict)
            materials_df = materials_df.dropna()  # 欠損値を含む行を削除
    
            print(f"有効な材料数: {len(materials_df)}")
    
            return materials_df, structures
    
    
    # 使用例（ダミーデータで代替 - 実際のAPIキーがある場合はコメント解除）
    # criteria = {
    #     "band_gap": (0.5, 5.0),  # バンドギャップ 0.5-5.0 eV
    #     "is_stable": True,       # 安定な材料のみ
    #     "nelements": (2, 3)      # 2-3元素系
    # }
    #
    # properties = [
    #     "material_id", "formula_pretty", "structure",
    #     "band_gap", "formation_energy_per_atom", "density",
    #     "energy_above_hull", "volume"
    # ]
    #
    # materials_df, structures = fetch_materials_data(
    #     api_key=MP_API_KEY,
    #     criteria=criteria,
    #     properties=properties,
    #     max_materials=1000
    # )
    
    # ダミーデータの生成（APIキーがない場合）
    print("\nダミーデータを生成します...")
    from pymatgen.core import Lattice, Structure
    
    np.random.seed(42)
    n_materials = 1000
    
    materials_df = pd.DataFrame({
        'material_id': [f'mp-{1000+i}' for i in range(n_materials)],
        'formula': [f'Material_{i}' for i in range(n_materials)],
        'band_gap': np.random.exponential(2.0, n_materials),
        'formation_energy_per_atom': np.random.normal(-1.5, 0.8, n_materials),
        'density': np.random.normal(5.0, 1.5, n_materials).clip(0.1),
        'energy_above_hull': np.random.exponential(0.05, n_materials),
        'volume': np.random.normal(50, 15, n_materials).clip(10)
    })
    
    # ダミー結晶構造の生成
    structures = []
    for i in range(n_materials):
        lattice = Lattice.cubic(4.0 + np.random.rand())
        n_atoms = np.random.randint(2, 8)
        species = np.random.choice(['Li', 'Na', 'K', 'O', 'S', 'Cl', 'F'], n_atoms)
        coords = np.random.rand(n_atoms, 3)
        structure = Structure(lattice, species, coords)
        structures.append(structure)
    
    print(f"ダミー材料データを {n_materials}個 生成しました")
    print("\n材料データの最初の5行:")
    print(materials_df.head())
    

**出力例** :
    
    
    ダミー材料データを 1000個 生成しました
    
    材料データの最初の5行:
      material_id      formula  band_gap  formation_energy_per_atom  density  energy_above_hull  volume
    0     mp-1000  Material_0  1.234567                  -1.678901    4.567             0.012345  45.678
    1     mp-1001  Material_1  2.345678                  -1.234567    5.678             0.023456  50.789
    2     mp-1002  Material_2  0.987654                  -2.012345    3.456             0.001234  48.901
    3     mp-1003  Material_3  3.456789                  -0.987654    6.789             0.034567  52.345
    4     mp-1004  Material_4  1.876543                  -1.456789    4.890             0.009876  47.234
    

### コード例3: データの探索的分析
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 基本統計量
    print("材料データの基本統計量:")
    print(materials_df.describe())
    
    # 特性の分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    properties_to_plot = [
        'band_gap', 'formation_energy_per_atom', 'density',
        'energy_above_hull', 'volume'
    ]
    
    for idx, prop in enumerate(properties_to_plot):
        axes[idx].hist(materials_df[prop], bins=50, alpha=0.7,
                       edgecolor='black', color='steelblue')
        axes[idx].set_xlabel(prop.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'Distribution of {prop.replace("_", " ").title()}',
                            fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # 最後のサブプロットを非表示
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('materials_data_distributions.png', dpi=300, bbox_inches='tight')
    print("\n特性分布を materials_data_distributions.png に保存しました")
    plt.show()
    
    # 相関行列
    correlation_matrix = materials_df[properties_to_plot].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Material Properties',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('materials_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("相関行列を materials_correlation_matrix.png に保存しました")
    plt.show()
    

## 4.2 グラフデータセットの構築

### コード例4: 結晶構造からグラフへの変換（最適化版）
    
    
    import torch
    from torch_geometric.data import Data
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import Element
    import numpy as np
    
    class MaterialGraphConverter:
        """結晶構造をグラフに変換するクラス"""
    
        def __init__(self, cutoff=5.0, max_neighbors=12):
            self.cutoff = cutoff
            self.max_neighbors = max_neighbors
    
            # 原子特性の定義（元素記号 → 特徴ベクトル）
            self._build_atom_features()
    
        def _build_atom_features(self):
            """原子特性辞書を構築"""
            # 主要な元素の特性を事前計算
            common_elements = [
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr'
            ]
    
            self.atom_features = {}
    
            for symbol in common_elements:
                try:
                    element = Element(symbol)
    
                    features = [
                        element.Z / 100.0,  # 正規化された原子番号
                        element.atomic_mass / 200.0,  # 正規化された原子量
                        element.X if element.X is not None else 0.0,  # 電気陰性度
                        element.atomic_radius if element.atomic_radius is not None else 0.0,  # 原子半径
                        element.group if element.group is not None else 0.0,  # 族
                        element.row if element.row is not None else 0.0,  # 周期
                    ]
    
                    self.atom_features[symbol] = features
    
                except Exception as e:
                    print(f"Warning: Could not process element {symbol}: {e}")
                    self.atom_features[symbol] = [0.0] * 6
    
        def get_atom_features(self, species):
            """元素記号から特徴ベクトルを取得"""
            symbol = str(species)
            if symbol in self.atom_features:
                return self.atom_features[symbol]
            else:
                # 未知の元素はゼロベクトル
                return [0.0] * 6
    
        def structure_to_graph(self, structure, target=None):
            """
            pymatgen StructureをPyTorch Geometric Dataに変換
    
            Parameters:
            -----------
            structure : pymatgen.Structure
                結晶構造
            target : float, optional
                ターゲット値（特性予測用）
    
            Returns:
            --------
            data : torch_geometric.data.Data
                グラフデータ
            """
            # ノード特徴
            node_features = []
            for site in structure:
                features = self.get_atom_features(site.specie)
                node_features.append(features)
    
            node_features = torch.tensor(node_features, dtype=torch.float)
    
            # エッジの構築
            all_neighbors = structure.get_all_neighbors(self.cutoff)
            edge_index = []
            edge_attr = []
    
            for i, neighbors in enumerate(all_neighbors):
                # 距離でソートして近い順にmax_neighbors個まで
                neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:self.max_neighbors]
    
                for neighbor in neighbors:
                    j = neighbor.index
                    distance = neighbor.nn_distance
    
                    edge_index.append([i, j])
                    edge_attr.append([distance])
    
            # エッジがない場合の処理
            if len(edge_index) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
            # Dataオブジェクトの作成
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
            if target is not None:
                data.y = torch.tensor([target], dtype=torch.float)
    
            return data
    
    
    # インスタンス化とテスト
    converter = MaterialGraphConverter(cutoff=5.0, max_neighbors=12)
    
    # 最初の材料でテスト
    test_structure = structures[0]
    test_target = materials_df.iloc[0]['band_gap']
    test_graph = converter.structure_to_graph(test_structure, test_target)
    
    print("グラフ変換のテスト:")
    print(f"ノード数: {test_graph.num_nodes}")
    print(f"エッジ数: {test_graph.num_edges}")
    print(f"ノード特徴次元: {test_graph.x.shape[1]}")
    print(f"ターゲット値: {test_graph.y.item():.3f}")
    

### コード例5: カスタムデータセットクラス
    
    
    from torch_geometric.data import Dataset, Data
    import torch
    from tqdm import tqdm
    import pickle
    
    class MaterialsDataset(Dataset):
        """
        材料データセット
    
        Parameters:
        -----------
        structures : list
            pymatgen.Structure のリスト
        targets : array-like
            ターゲット値
        converter : MaterialGraphConverter
            グラフ変換器
        root : str
            データ保存先ディレクトリ
        transform : callable, optional
            データ変換関数
        pre_transform : callable, optional
            前処理関数
        """
    
        def __init__(self, structures, targets, converter,
                     root='data/materials', transform=None, pre_transform=None):
            self.structures = structures
            self.targets = targets
            self.converter = converter
    
            super().__init__(root, transform, pre_transform)
    
        @property
        def raw_file_names(self):
            return []
    
        @property
        def processed_file_names(self):
            return [f'data_{i}.pt' for i in range(len(self.structures))]
    
        def download(self):
            pass
    
        def process(self):
            """構造をグラフに変換して保存"""
            for idx in tqdm(range(len(self.structures)), desc="グラフ変換中"):
                structure = self.structures[idx]
                target = self.targets[idx]
    
                # グラフへの変換
                data = self.converter.structure_to_graph(structure, target)
    
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
    
                # 保存
                torch.save(data, self.processed_paths[idx])
    
        def len(self):
            return len(self.structures)
    
        def get(self, idx):
            data = torch.load(self.processed_paths[idx])
            return data
    
    
    # データセットの作成
    print("データセットを作成中...")
    
    # ターゲット特性（バンドギャップ）
    targets = materials_df['band_gap'].values
    
    # データセットのインスタンス化
    dataset = MaterialsDataset(
        structures=structures,
        targets=targets,
        converter=converter,
        root='data/materials_dataset'
    )
    
    print(f"\nデータセット作成完了！")
    print(f"データセットサイズ: {len(dataset)}")
    print(f"サンプルデータ:")
    print(dataset[0])
    

**出力例** :
    
    
    グラフ変換中: 100%|██████████| 1000/1000 [00:15<00:00, 63.42it/s]
    
    データセット作成完了！
    データセットサイズ: 1000
    サンプルデータ:
    Data(x=[5, 6], edge_index=[2, 48], edge_attr=[48, 1], y=[1])
    

### コード例6: データセットの分割とDataLoader
    
    
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    import torch
    
    # データセットのインデックス分割
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42
    )
    
    # サブセットの作成
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    print(f"データ分割:")
    print(f"  学習データ: {len(train_dataset)}")
    print(f"  検証データ: {len(val_dataset)}")
    print(f"  テストデータ: {len(test_dataset)}")
    
    # DataLoaderの作成
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # DataLoaderのテスト
    batch = next(iter(train_loader))
    print(f"\nバッチデータ:")
    print(f"  バッチサイズ: {batch.num_graphs}")
    print(f"  総ノード数: {batch.num_nodes}")
    print(f"  総エッジ数: {batch.num_edges}")
    print(f"  ノード特徴形状: {batch.x.shape}")
    print(f"  ターゲット形状: {batch.y.shape}")
    

**出力例** :
    
    
    データ分割:
      学習データ: 700
      検証データ: 150
      テストデータ: 150
    
    バッチデータ:
      バッチサイズ: 32
      総ノード数: 168
      総エッジ数: 1568
      ノード特徴形状: torch.Size([168, 6])
      ターゲット形状: torch.Size([32, 1])
    

## 4.3 GNNモデルの学習

### コード例7: 改良版CGCNNモデル
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
    
    class ImprovedCGConv(MessagePassing):
        """改良版CGCNN畳み込み層"""
    
        def __init__(self, node_dim, edge_dim, hidden_dim=128):
            super().__init__(aggr='add')
    
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim
    
            # エッジネットワーク
            self.edge_network = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus()
            )
    
            # ゲートネットワーク
            self.gate_network = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.Sigmoid()
            )
    
            # ノード更新ネットワーク
            self.node_update = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
        def message(self, x_i, x_j, edge_attr):
            """メッセージ関数"""
            # 送信元、受信先、エッジ特徴を結合
            z = torch.cat([x_i, x_j, edge_attr], dim=1)
    
            # ゲーティング機構
            gate = self.gate_network(z)
            message = self.edge_network(z)
    
            return gate * message
    
        def update(self, aggr_out, x):
            """ノード更新"""
            combined = torch.cat([x, aggr_out], dim=1)
            return self.node_update(combined)
    
    
    class ImprovedCGCNN(nn.Module):
        """改良版CGCNNモデル"""
    
        def __init__(self, node_dim, edge_dim, hidden_dim=128,
                     num_conv=4, dropout=0.2, pooling='mean'):
            super().__init__()
    
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim
            self.num_conv = num_conv
            self.pooling = pooling
    
            # 入力埋め込み
            self.node_embedding = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus()
            )
    
            # CGConv層
            self.conv_layers = nn.ModuleList([
                ImprovedCGConv(hidden_dim, edge_dim, hidden_dim)
                for _ in range(num_conv)
            ])
    
            # Batch Normalization
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_conv)
            ])
    
            # Dropout
            self.dropout = nn.Dropout(dropout)
    
            # プーリング
            if pooling == 'mean':
                self.pool = global_mean_pool
            elif pooling == 'add':
                self.pool = global_add_pool
            elif pooling == 'max':
                self.pool = global_max_pool
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
    
            # 出力ネットワーク
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Softplus(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.Softplus(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
    
        def forward(self, data, return_embedding=False):
            """
            順伝播
    
            Parameters:
            -----------
            data : torch_geometric.data.Batch
                バッチデータ
            return_embedding : bool
                埋め込みを返すか
    
            Returns:
            --------
            out : Tensor
                予測値
            embedding : Tensor (optional)
                グラフ埋め込み
            """
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # 入力埋め込み
            x = self.node_embedding(x)
    
            # CGConv層の適用
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x_new = conv(x, edge_index, edge_attr)
                x_new = bn(x_new)
                x_new = F.softplus(x_new)
                x_new = self.dropout(x_new)
    
                # 残差接続
                x = x + x_new
    
            # グラフレベルのプーリング
            graph_embedding = self.pool(x, batch)
    
            # 出力
            out = self.fc(graph_embedding)
    
            if return_embedding:
                return out, graph_embedding
            else:
                return out
    
    
    # モデルのインスタンス化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedCGCNN(
        node_dim=6,
        edge_dim=1,
        hidden_dim=128,
        num_conv=4,
        dropout=0.2,
        pooling='mean'
    ).to(device)
    
    print(f"モデルの総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # テスト
    batch = next(iter(train_loader)).to(device)
    predictions, embeddings = model(batch, return_embedding=True)
    
    print(f"\n予測値の形状: {predictions.shape}")
    print(f"埋め込みの形状: {embeddings.shape}")
    

### コード例8: 学習ループ（Early Stopping付き）
    
    
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    from tqdm import tqdm
    import time
    
    class EarlyStopping:
        """Early Stoppingクラス"""
    
        def __init__(self, patience=20, delta=0.001, path='best_model.pt'):
            self.patience = patience
            self.delta = delta
            self.path = path
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
    
        def __call__(self, val_loss, model):
            score = -val_loss
    
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
    
        def save_checkpoint(self, val_loss, model):
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
    
    
    def train_epoch(model, loader, criterion, optimizer, device):
        """1エポックの学習"""
        model.train()
        total_loss = 0
        total_samples = 0
    
        for batch in loader:
            batch = batch.to(device)
    
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
    
        return total_loss / total_samples
    
    
    def validate_epoch(model, loader, criterion, device):
        """1エポックの検証"""
        model.eval()
        total_loss = 0
        total_samples = 0
    
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
    
                predictions = model(batch)
                loss = criterion(predictions, batch.y)
    
                total_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs
    
        return total_loss / total_samples
    
    
    # 学習設定
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=30, delta=0.001,
                                   path='best_cgcnn_model.pt')
    
    # 学習ループ
    num_epochs = 200
    train_losses = []
    val_losses = []
    
    print("学習開始...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 学習
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
    
        # 検証
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
    
        # 学習率スケジューラ
        scheduler.step(val_loss)
    
        # Early Stopping
        early_stopping(val_loss, model)
    
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                  f"Time: {elapsed_time:.1f}s")
    
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n学習完了！ 総時間: {total_time/60:.1f}分")
    
    # ベストモデルのロード
    model.load_state_dict(torch.load('best_cgcnn_model.pt'))
    print(f"ベストモデルをロードしました (Val Loss: {early_stopping.val_loss_min:.4f})")
    

**出力例** :
    
    
    学習開始...
    Epoch [10/200] - Train Loss: 1.2345, Val Loss: 1.3456 - Time: 15.3s
    Epoch [20/200] - Train Loss: 0.9876, Val Loss: 1.1234 - Time: 30.7s
    Epoch [30/200] - Train Loss: 0.7654, Val Loss: 1.0123 - Time: 46.1s
    Epoch [40/200] - Train Loss: 0.6543, Val Loss: 0.9876 - Time: 61.5s
    Epoch [50/200] - Train Loss: 0.5678, Val Loss: 0.9654 - Time: 76.9s
    Early stopping at epoch 85
    
    学習完了！ 総時間: 2.3分
    ベストモデルをロードしました (Val Loss: 0.9234)
    

### コード例9: 学習曲線の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 学習曲線
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss曲線
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, color='red')
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 対数スケール
    ax2.semilogy(epochs, train_losses, label='Train Loss', linewidth=2, color='blue')
    ax2.semilogy(epochs, val_losses, label='Validation Loss', linewidth=2, color='red')
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MSE Loss (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Training Curve (Log Scale)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("学習曲線を training_curve.png に保存しました")
    plt.show()
    

### コード例10: テストデータでの評価
    
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    
    def evaluate_model(model, loader, device):
        """モデルの評価"""
        model.eval()
    
        all_predictions = []
        all_targets = []
    
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
    
                predictions = model(batch)
    
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
    
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
    
        return predictions, targets
    
    
    # テストデータでの評価
    test_predictions, test_targets = evaluate_model(model, test_loader, device)
    
    # 評価指標の計算
    mae = mean_absolute_error(test_targets, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    r2 = r2_score(test_targets, test_predictions)
    
    print("テストデータでの評価:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # 予測値 vs 実測値のプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散布図
    ax1.scatter(test_targets, test_predictions, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5, color='steelblue')
    
    # 理想線
    min_val = min(test_targets.min(), test_predictions.min())
    max_val = max(test_targets.max(), test_predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Ideal')
    
    ax1.set_xlabel('True Band Gap (eV)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Band Gap (eV)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Predictions vs True Values\nR²={r2:.3f}, MAE={mae:.3f}',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 残差プロット
    residuals = test_predictions - test_targets
    ax2.scatter(test_targets, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5, color='coral')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('True Band Gap (eV)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Residuals (eV)', fontsize=14, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n評価結果を test_evaluation.png に保存しました")
    plt.show()
    

## 4.4 埋め込み抽出と次元削減

### コード例11: 全データからの埋め込み抽出
    
    
    def extract_all_embeddings(model, dataset, device, batch_size=64):
        """
        全データセットから埋め込みを抽出
    
        Parameters:
        -----------
        model : nn.Module
            学習済みモデル
        dataset : Dataset
            データセット
        device : torch.device
            デバイス
        batch_size : int
            バッチサイズ
    
        Returns:
        --------
        embeddings : np.ndarray
            埋め込みベクトル
        targets : np.ndarray
            ターゲット値
        """
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        all_embeddings = []
        all_targets = []
    
        with torch.no_grad():
            for batch in tqdm(loader, desc="埋め込み抽出中"):
                batch = batch.to(device)
    
                _, embeddings = model(batch, return_embedding=True)
    
                all_embeddings.append(embeddings.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
    
        embeddings = np.concatenate(all_embeddings, axis=0)
        targets = np.concatenate(all_targets, axis=0).flatten()
    
        return embeddings, targets
    
    
    # 全データから埋め込みを抽出
    embeddings, targets = extract_all_embeddings(model, dataset, device)
    
    print(f"埋め込みの形状: {embeddings.shape}")
    print(f"ターゲットの形状: {targets.shape}")
    
    # 保存
    np.save('cgcnn_embeddings.npy', embeddings)
    np.save('cgcnn_targets.npy', targets)
    print("\n埋め込みを保存しました")
    

### コード例12: PCAによる次元削減
    
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    # PCAの実行
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    
    # 寄与率の分析
    explained_variance_ratio = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance_ratio)
    
    # スクリープロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 個別寄与率
    ax1.bar(range(1, 21), explained_variance_ratio[:20],
            alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
    ax1.set_title('PCA Scree Plot (Top 20 Components)',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 累積寄与率
    ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance,
             marker='o', linewidth=2, markersize=4, color='darkred')
    ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2,
                label='95% threshold', alpha=0.7)
    ax2.set_xlabel('Number of Components', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    print(f"PCA分析を pca_analysis.png に保存しました")
    print(f"\n95%の分散を説明する主成分数: {np.argmax(cumsum_variance >= 0.95) + 1}")
    plt.show()
    
    # 2次元PCA可視化
    pca_2d = PCA(n_components=2)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1],
                          c=targets, cmap='viridis', s=50, alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)',
               fontsize=14, fontweight='bold')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)',
               fontsize=14, fontweight='bold')
    plt.title('PCA: 2D Projection of CGCNN Embeddings',
              fontsize=16, fontweight='bold')
    plt.colorbar(scatter, label='Band Gap (eV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_2d_projection.png', dpi=300, bbox_inches='tight')
    print("PCA 2D射影を pca_2d_projection.png に保存しました")
    plt.show()
    

### コード例13: UMAPによる次元削減
    
    
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    
    # UMAP実行（複数パラメータで実験）
    n_neighbors_list = [5, 15, 30, 50]
    min_dist_list = [0.0, 0.1, 0.3]
    
    # 最適なパラメータでの実行
    print("UMAPを実行中...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        verbose=True
    )
    
    embeddings_umap = reducer.fit_transform(embeddings)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ターゲット値で色付け
    scatter1 = axes[0].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                               c=targets, cmap='viridis', s=50, alpha=0.6,
                               edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    axes[0].set_title('UMAP: Colored by Band Gap',
                      fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    # カテゴリで色付け
    band_gap_categories = pd.cut(targets, bins=[0, 1, 2, 3, 10],
                                  labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    colors_map = {'Small': 'red', 'Medium': 'orange', 'Large': 'yellow', 'Very Large': 'green'}
    colors = [colors_map[cat] for cat in band_gap_categories]
    
    for category in ['Small', 'Medium', 'Large', 'Very Large']:
        mask = band_gap_categories == category
        axes[1].scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                        c=colors_map[category], label=category, s=50, alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    axes[1].set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    axes[1].set_title('UMAP: Colored by Band Gap Category',
                      fontsize=16, fontweight='bold')
    axes[1].legend(title='Band Gap', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('umap_2d_projection.png', dpi=300, bbox_inches='tight')
    print("\nUMAP 2D射影を umap_2d_projection.png に保存しました")
    plt.show()
    
    # UMAP結果の保存
    np.save('umap_embeddings_2d.npy', embeddings_umap)
    

### コード例14: t-SNEによる次元削減
    
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import time
    
    # t-SNE実行（サブセットで高速化）
    subset_size = min(1000, len(embeddings))
    subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
    
    embeddings_subset = embeddings[subset_indices]
    targets_subset = targets[subset_indices]
    
    print(f"t-SNEを実行中 (サブセットサイズ: {subset_size})...")
    start_time = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    
    embeddings_tsne = tsne.fit_transform(embeddings_subset)
    
    elapsed_time = time.time() - start_time
    print(f"t-SNE完了 (所要時間: {elapsed_time:.1f}秒)")
    
    # 可視化
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                          c=targets_subset, cmap='plasma', s=50, alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    plt.xlabel('t-SNE 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14, fontweight='bold')
    plt.title('t-SNE: 2D Projection of CGCNN Embeddings',
              fontsize=16, fontweight='bold')
    plt.colorbar(scatter, label='Band Gap (eV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_2d_projection.png', dpi=300, bbox_inches='tight')
    print("t-SNE 2D射影を tsne_2d_projection.png に保存しました")
    plt.show()
    

### コード例15: 次元削減手法の比較
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 3つの手法の結果を並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # PCA
    scatter = axes[0].scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1],
                              c=targets, cmap='viridis', s=30, alpha=0.6,
                              edgecolors='none')
    axes[0].set_xlabel('PC1', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PC2', fontsize=12, fontweight='bold')
    axes[0].set_title('PCA', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # UMAP
    scatter = axes[1].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                              c=targets, cmap='viridis', s=30, alpha=0.6,
                              edgecolors='none')
    axes[1].set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    axes[1].set_title('UMAP', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # t-SNE
    scatter = axes[2].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                              c=targets_subset, cmap='viridis', s=30, alpha=0.6,
                              edgecolors='none')
    axes[2].set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    axes[2].set_title('t-SNE', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 共通カラーバー
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
    print("次元削減手法比較を dimensionality_reduction_comparison.png に保存しました")
    plt.show()
    

## 4.5 材料空間の分析

### コード例16: クラスタリングと特性分析
    
    
    from sklearn.cluster import KMeans
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # K-Meansクラスタリング（UMAP空間上）
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(embeddings_umap)
    
    # クラスタごとの可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # クラスタラベルで色付け
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax1.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                    c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # クラスタ中心
    umap_reducer_for_centers = umap.UMAP(n_components=2, n_neighbors=15,
                                         min_dist=0.1, random_state=42)
    umap_reducer_for_centers.fit(embeddings)
    centers_umap = umap_reducer_for_centers.transform(kmeans.cluster_centers_)
    
    ax1.scatter(centers_umap[:, 0], centers_umap[:, 1],
                c='red', marker='X', s=300, edgecolors='black',
                linewidth=2, label='Centroids', zorder=10)
    
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax1.set_title(f'K-Means Clustering (k={n_clusters})',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # クラスタごとの特性分布
    cluster_df = pd.DataFrame({
        'cluster': cluster_labels,
        'band_gap': targets
    })
    
    sns.boxplot(data=cluster_df, x='cluster', y='band_gap',
                ax=ax2, palette='Set3')
    sns.swarmplot(data=cluster_df, x='cluster', y='band_gap',
                  ax=ax2, color='black', alpha=0.3, size=2)
    
    ax2.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Band Gap (eV)', fontsize=14, fontweight='bold')
    ax2.set_title('Band Gap Distribution by Cluster',
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("クラスタリング分析を clustering_analysis.png に保存しました")
    plt.show()
    
    # クラスタ統計
    print("\nクラスタごとのバンドギャップ統計:")
    cluster_stats = cluster_df.groupby('cluster')['band_gap'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(cluster_stats.round(3))
    

### コード例17: 材料特性とクラスタの関係分析
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 追加の材料特性をDataFrameに統合
    materials_analysis_df = materials_df.copy()
    materials_analysis_df['cluster'] = cluster_labels
    materials_analysis_df['umap1'] = embeddings_umap[:, 0]
    materials_analysis_df['umap2'] = embeddings_umap[:, 1]
    
    # クラスタごとの複数特性の統計
    properties_to_analyze = [
        'band_gap', 'formation_energy_per_atom',
        'density', 'energy_above_hull', 'volume'
    ]
    
    cluster_property_stats = materials_analysis_df.groupby('cluster')[
        properties_to_analyze
    ].mean()
    
    # ヒートマップで可視化
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_property_stats.T, annot=True, fmt='.2f',
                cmap='RdYlGn_r', center=0, linewidths=1,
                cbar_kws={"label": "Normalized Value"})
    plt.title('Average Material Properties by Cluster',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cluster ID', fontsize=14, fontweight='bold')
    plt.ylabel('Property', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_properties_heatmap.png', dpi=300, bbox_inches='tight')
    print("クラスタ特性ヒートマップを cluster_properties_heatmap.png に保存しました")
    plt.show()
    

### コード例18: 密度マップの作成
    
    
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    import numpy as np
    
    # UMAP空間での密度推定
    xy = np.vstack([embeddings_umap[:, 0], embeddings_umap[:, 1]])
    density = gaussian_kde(xy)(xy)
    
    # 密度でソート（高密度点を上に描画）
    idx = density.argsort()
    x, y, z = embeddings_umap[idx, 0], embeddings_umap[idx, 1], density[idx]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 密度マップ
    scatter1 = ax1.scatter(x, y, c=z, cmap='hot', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax1.set_title('Materials Space Density Map',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Point Density', fontsize=12, fontweight='bold')
    
    # 密度マップ + バンドギャップ
    scatter2 = ax2.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                           c=targets, s=50 + density*1000, alpha=0.6,
                           cmap='viridis', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax2.set_title('Band Gap with Density (size = density)',
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('density_map.png', dpi=300, bbox_inches='tight')
    print("密度マップを density_map.png に保存しました")
    plt.show()
    

### コード例19: 近傍材料の探索
    
    
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    
    def find_similar_materials_in_embedding_space(
        query_idx, embeddings, materials_df, k=10
    ):
        """
        埋め込み空間で類似材料を検索
    
        Parameters:
        -----------
        query_idx : int
            クエリ材料のインデックス
        embeddings : np.ndarray
            埋め込みベクトル
        materials_df : pd.DataFrame
            材料データ
        k : int
            検索する近傍数
    
        Returns:
        --------
        results_df : pd.DataFrame
            近傍材料の情報
        """
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings[query_idx:query_idx+1])
    
        # 結果のDataFrame作成
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if i == 0:  # クエリ自身をスキップ
                continue
    
            result = {
                'rank': i,
                'material_id': materials_df.iloc[idx]['material_id'],
                'formula': materials_df.iloc[idx]['formula'],
                'band_gap': materials_df.iloc[idx]['band_gap'],
                'formation_energy': materials_df.iloc[idx]['formation_energy_per_atom'],
                'density': materials_df.iloc[idx]['density'],
                'distance': dist
            }
            results.append(result)
    
        results_df = pd.DataFrame(results)
        return results_df
    
    
    # 使用例: ランダムに材料を選んで類似材料を検索
    np.random.seed(42)
    query_indices = np.random.choice(len(materials_df), 3, replace=False)
    
    print("類似材料検索の例:\n")
    for query_idx in query_indices:
        query_material = materials_df.iloc[query_idx]
    
        print(f"クエリ材料:")
        print(f"  Material ID: {query_material['material_id']}")
        print(f"  Formula: {query_material['formula']}")
        print(f"  Band Gap: {query_material['band_gap']:.3f} eV")
        print(f"\n類似材料 (Top 5):")
    
        similar_materials = find_similar_materials_in_embedding_space(
            query_idx, embeddings, materials_df, k=5
        )
    
        print(similar_materials[['rank', 'material_id', 'formula', 'band_gap', 'distance']].to_string(index=False))
        print("\n" + "="*80 + "\n")
    

### コード例20: 材料推薦システム
    
    
    def recommend_materials_by_property(
        target_property_value,
        property_name,
        embeddings,
        materials_df,
        n_recommendations=10,
        property_weight=0.7,
        embedding_weight=0.3
    ):
        """
        目標特性値に基づいて材料を推薦
    
        Parameters:
        -----------
        target_property_value : float
            目標特性値
        property_name : str
            特性名
        embeddings : np.ndarray
            埋め込みベクトル
        materials_df : pd.DataFrame
            材料データ
        n_recommendations : int
            推薦数
        property_weight : float
            特性値の重み
        embedding_weight : float
            埋め込み距離の重み
    
        Returns:
        --------
        recommendations_df : pd.DataFrame
            推薦材料のリスト
        """
        # 特性値の差
        property_diff = np.abs(materials_df[property_name].values - target_property_value)
        property_diff_normalized = property_diff / property_diff.max()
    
        # 目標特性値に最も近い材料のインデックス
        closest_idx = property_diff.argmin()
    
        # 埋め込み空間での距離
        from sklearn.metrics.pairwise import cosine_distances
        embedding_distances = cosine_distances(
            embeddings[closest_idx:closest_idx+1],
            embeddings
        ).flatten()
        embedding_distances_normalized = embedding_distances / embedding_distances.max()
    
        # 統合スコア（小さいほど良い）
        combined_score = (
            property_weight * property_diff_normalized +
            embedding_weight * embedding_distances_normalized
        )
    
        # 上位n_recommendations個を選択
        top_indices = combined_score.argsort()[:n_recommendations]
    
        # 結果のDataFrame作成
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            rec = {
                'rank': rank,
                'material_id': materials_df.iloc[idx]['material_id'],
                'formula': materials_df.iloc[idx]['formula'],
                property_name: materials_df.iloc[idx][property_name],
                'property_diff': property_diff[idx],
                'embedding_distance': embedding_distances[idx],
                'combined_score': combined_score[idx]
            }
            recommendations.append(rec)
    
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df
    
    
    # 使用例: バンドギャップ2.0 eVに近い材料を推薦
    target_bandgap = 2.0
    
    print(f"目標バンドギャップ: {target_bandgap} eV")
    print("\n推薦材料（Top 10）:\n")
    
    recommendations = recommend_materials_by_property(
        target_property_value=target_bandgap,
        property_name='band_gap',
        embeddings=embeddings,
        materials_df=materials_df,
        n_recommendations=10,
        property_weight=0.7,
        embedding_weight=0.3
    )
    
    print(recommendations[['rank', 'material_id', 'formula', 'band_gap',
                           'property_diff', 'combined_score']].to_string(index=False))
    

## 4.6 インタラクティブ可視化

### コード例21: Plotlyによる3D UMAP
    
    
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    import pandas as pd
    
    # 3次元UMAP
    print("3次元UMAPを実行中...")
    reducer_3d = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False
    )
    
    embeddings_umap_3d = reducer_3d.fit_transform(embeddings)
    
    # DataFrameの作成
    df_3d = pd.DataFrame({
        'UMAP1': embeddings_umap_3d[:, 0],
        'UMAP2': embeddings_umap_3d[:, 1],
        'UMAP3': embeddings_umap_3d[:, 2],
        'Band_Gap': targets,
        'Material_ID': materials_df['material_id'].values,
        'Formula': materials_df['formula'].values,
        'Formation_Energy': materials_df['formation_energy_per_atom'].values,
        'Density': materials_df['density'].values,
        'Cluster': cluster_labels
    })
    
    # インタラクティブ3Dプロット
    fig = px.scatter_3d(
        df_3d,
        x='UMAP1', y='UMAP2', z='UMAP3',
        color='Band_Gap',
        size='Density',
        hover_data=['Material_ID', 'Formula', 'Formation_Energy', 'Cluster'],
        color_continuous_scale='Viridis',
        title='Interactive 3D UMAP: Materials Space Explorer'
    )
    
    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        ),
        width=1000,
        height=800,
        font=dict(size=12)
    )
    
    fig.write_html('materials_3d_interactive.html')
    print("インタラクティブ3D可視化を materials_3d_interactive.html に保存しました")
    fig.show()
    

### コード例22: Bokehによるインタラクティブ散布図
    
    
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper, ColumnDataSource
    from bokeh.palettes import Viridis256
    from bokeh.io import show
    from bokeh.layouts import column, row
    from bokeh.models.widgets import Select
    
    # カラーマッパー
    color_mapper = LinearColorMapper(
        palette=Viridis256,
        low=targets.min(),
        high=targets.max()
    )
    
    # データソース
    source = ColumnDataSource(data=dict(
        x=embeddings_umap[:, 0],
        y=embeddings_umap[:, 1],
        material_id=materials_df['material_id'].values,
        formula=materials_df['formula'].values,
        band_gap=targets,
        formation_energy=materials_df['formation_energy_per_atom'].values,
        density=materials_df['density'].values,
        volume=materials_df['volume'].values,
        cluster=cluster_labels
    ))
    
    # プロットの作成
    output_file('materials_interactive.html')
    
    p = figure(
        width=1000,
        height=800,
        title='Interactive Materials Space Explorer (UMAP)',
        tools='pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save'
    )
    
    # 散布図
    circles = p.circle(
        'x', 'y',
        size=8,
        source=source,
        fill_color={'field': 'band_gap', 'transform': color_mapper},
        fill_alpha=0.7,
        line_color='black',
        line_width=0.5
    )
    
    # ホバーツール
    hover = HoverTool(tooltips=[
        ('Material ID', '@material_id'),
        ('Formula', '@formula'),
        ('Band Gap', '@band_gap{0.00} eV'),
        ('Formation E', '@formation_energy{0.00} eV/atom'),
        ('Density', '@density{0.00} g/cm³'),
        ('Volume', '@volume{0.0} Ų'),
        ('Cluster', '@cluster')
    ])
    p.add_tools(hover)
    
    # カラーバー
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        title='Band Gap (eV)',
        location=(0, 0)
    )
    p.add_layout(color_bar, 'right')
    
    # 軸ラベル
    p.xaxis.axis_label = 'UMAP 1'
    p.yaxis.axis_label = 'UMAP 2'
    p.title.text_font_size = '16pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    
    save(p)
    print("インタラクティブUMAPを materials_interactive.html に保存しました")
    show(p)
    

### コード例23: Dashによるダッシュボード（オプション）
    
    
    # Dashのインストール（初回のみ）
    # !pip install dash
    
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    # Dashアプリの作成
    app = dash.Dash(__name__)
    
    # データフレームの準備
    df_dash = pd.DataFrame({
        'umap1': embeddings_umap[:, 0],
        'umap2': embeddings_umap[:, 1],
        'material_id': materials_df['material_id'].values,
        'formula': materials_df['formula'].values,
        'band_gap': targets,
        'formation_energy': materials_df['formation_energy_per_atom'].values,
        'density': materials_df['density'].values,
        'cluster': cluster_labels
    })
    
    # レイアウト
    app.layout = html.Div([
        html.H1('Materials Space Explorer Dashboard'),
    
        html.Div([
            html.Label('Color by:'),
            dcc.Dropdown(
                id='color-dropdown',
                options=[
                    {'label': 'Band Gap', 'value': 'band_gap'},
                    {'label': 'Formation Energy', 'value': 'formation_energy'},
                    {'label': 'Density', 'value': 'density'},
                    {'label': 'Cluster', 'value': 'cluster'}
                ],
                value='band_gap'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    
        dcc.Graph(id='umap-scatter'),
    
        html.Div(id='material-info')
    ])
    
    # コールバック
    @app.callback(
        Output('umap-scatter', 'figure'),
        Input('color-dropdown', 'value')
    )
    def update_scatter(color_by):
        fig = px.scatter(
            df_dash,
            x='umap1',
            y='umap2',
            color=color_by,
            hover_data=['material_id', 'formula', 'band_gap', 'formation_energy'],
            color_continuous_scale='Viridis',
            title=f'UMAP colored by {color_by}'
        )
    
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')))
    
        return fig
    
    # アプリの実行
    if __name__ == '__main__':
        app.run_server(debug=True)
    """
    
    print("Dashダッシュボードのコード例を表示しました")
    print("コメントを外して実行してください")
    

## 4.7 高度な分析と応用

### コード例24: 材料空間のボロノイ分割
    
    
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サブセットでボロノイ図を作成（計算量削減）
    subset_size = 100
    subset_indices = np.random.choice(len(embeddings_umap), subset_size, replace=False)
    
    points = embeddings_umap[subset_indices]
    targets_subset = targets[subset_indices]
    
    # ボロノイ図の計算
    vor = Voronoi(points)
    
    # プロット
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # ボロノイセル
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                    line_width=1, line_alpha=0.6, point_size=0)
    
    # 材料点
    scatter = ax.scatter(points[:, 0], points[:, 1],
                         c=targets_subset, cmap='viridis',
                         s=100, alpha=0.8, edgecolors='black', linewidth=1.5,
                         zorder=5)
    
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title('Voronoi Tessellation of Materials Space',
                 fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    # 軸の範囲を制限（無限遠点の除外）
    ax.set_xlim([points[:, 0].min() - 1, points[:, 0].max() + 1])
    ax.set_ylim([points[:, 1].min() - 1, points[:, 1].max() + 1])
    
    plt.tight_layout()
    plt.savefig('voronoi_tessellation.png', dpi=300, bbox_inches='tight')
    print("ボロノイ分割を voronoi_tessellation.png に保存しました")
    plt.show()
    

### コード例25: 特性勾配の可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    
    # グリッドの作成
    grid_x, grid_y = np.mgrid[
        embeddings_umap[:, 0].min():embeddings_umap[:, 0].max():100j,
        embeddings_umap[:, 1].min():embeddings_umap[:, 1].max():100j
    ]
    
    # 補間
    grid_z = griddata(
        embeddings_umap,
        targets,
        (grid_x, grid_y),
        method='cubic'
    )
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 等高線図
    contour = ax1.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.8)
    ax1.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                c='white', s=5, alpha=0.5, edgecolors='none')
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax1.set_title('Band Gap Contour Map', fontsize=16, fontweight='bold')
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    # 勾配ベクトル
    gradient_y, gradient_x = np.gradient(grid_z)
    
    # サブサンプリング（矢印の密度調整）
    skip = 5
    ax2.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.6)
    ax2.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
               gradient_x[::skip, ::skip], gradient_y[::skip, ::skip],
               color='white', alpha=0.8, scale=50)
    ax2.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                c='black', s=5, alpha=0.3, edgecolors='none')
    ax2.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax2.set_title('Band Gap Gradient Field', fontsize=16, fontweight='bold')
    cbar2 = plt.colorbar(contour, ax=ax2)
    cbar2.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('property_gradient.png', dpi=300, bbox_inches='tight')
    print("特性勾配を property_gradient.png に保存しました")
    plt.show()
    

### コード例26: 埋め込みの安定性評価
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    
    def evaluate_embedding_stability(model, dataset, device, n_splits=5):
        """
        クロスバリデーションで埋め込みの安定性を評価
    
        Parameters:
        -----------
        model : nn.Module
            モデル（再学習用）
        dataset : Dataset
            データセット
        device : torch.device
            デバイス
        n_splits : int
            分割数
    
        Returns:
        --------
        stability_scores : list
            安定性スコア
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        embeddings_list = []
    
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
            print(f"Fold {fold+1}/{n_splits}")
    
            # 省略: ここで各foldでモデルを再学習
            # 実際には新しいモデルインスタンスを作成し、train_idx で学習
    
            # デモ用: 既存の埋め込みを使用
            embeddings_list.append(embeddings)
    
        # 埋め込み間の類似度を計算
        from sklearn.metrics.pairwise import cosine_similarity
    
        similarities = []
        for i in range(len(embeddings_list)):
            for j in range(i+1, len(embeddings_list)):
                sim_matrix = cosine_similarity(embeddings_list[i], embeddings_list[j])
                # 対角成分の平均（各点の対応する埋め込み間の類似度）
                sim_score = np.mean(np.diag(sim_matrix))
                similarities.append(sim_score)
    
        return similarities
    
    
    # 実行（デモ）
    print("埋め込み安定性評価（デモ）:")
    stability_scores = evaluate_embedding_stability(model, dataset, device, n_splits=3)
    print(f"平均類似度: {np.mean(stability_scores):.3f}")
    print(f"標準偏差: {np.std(stability_scores):.3f}")
    

### コード例27: アンサンブルモデルの埋め込み統合
    
    
    def create_ensemble_embeddings(models_list, dataset, device):
        """
        複数モデルの埋め込みを統合
    
        Parameters:
        -----------
        models_list : list
            学習済みモデルのリスト
        dataset : Dataset
            データセット
        device : torch.device
            デバイス
    
        Returns:
        --------
        ensemble_embeddings : np.ndarray
            統合された埋め込み
        """
        all_embeddings = []
    
        for model in models_list:
            emb, _ = extract_all_embeddings(model, dataset, device)
            all_embeddings.append(emb)
    
        # 平均を取る
        ensemble_embeddings = np.mean(all_embeddings, axis=0)
    
        return ensemble_embeddings
    
    
    # デモ: 単一モデルを3回使用（実際には異なるモデルを使用）
    print("アンサンブル埋め込みのデモ:")
    models_list = [model, model, model]  # 実際には異なるモデル
    
    ensemble_emb = create_ensemble_embeddings(models_list, dataset, device)
    print(f"アンサンブル埋め込みの形状: {ensemble_emb.shape}")
    

### コード例28: 時系列的な材料空間の変化（拡張性）
    
    
    """
    時系列データや複数バージョンのモデルで材料空間の変化を追跡
    
    def visualize_materials_space_evolution(
        embeddings_timeline,
        targets_timeline,
        timestamps
    ):
        '''
        時系列的な材料空間の変化を可視化
    
        Parameters:
        -----------
        embeddings_timeline : list of np.ndarray
            各時点での埋め込み
        targets_timeline : list of np.ndarray
            各時点でのターゲット
        timestamps : list
            タイムスタンプ
    
        Returns:
        --------
        animation or multiple plots
        '''
        import matplotlib.animation as animation
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        def update(frame):
            ax.clear()
            embeddings = embeddings_timeline[frame]
            targets = targets_timeline[frame]
    
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                c=targets, cmap='viridis', s=50, alpha=0.6)
            ax.set_title(f'Materials Space at {timestamps[frame]}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
    
            return scatter,
    
        anim = animation.FuncAnimation(fig, update, frames=len(timestamps),
                                      interval=500, blit=False)
    
        return anim
    '''
    
    print("時系列可視化の拡張性コード例を表示しました")
    """
    

### コード例29: 外挿領域の検出
    
    
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 学習データの埋め込み範囲を学習
    train_embeddings = embeddings[train_idx]
    
    # Isolation Forestで外れ値（外挿領域）を検出
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(train_embeddings)
    
    # 全データで予測
    outlier_labels = iso_forest.predict(embeddings)
    outlier_scores = iso_forest.score_samples(embeddings)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 外れ値ラベル
    colors_outlier = ['red' if label == -1 else 'blue' for label in outlier_labels]
    ax1.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                c=colors_outlier, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax1.set_title('Extrapolation Region Detection (Red = Outlier)',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 外れ値スコア
    scatter = ax2.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                          c=outlier_scores, cmap='RdYlGn', s=50, alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax2.set_title('Anomaly Score (Green = Normal, Red = Outlier)',
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Anomaly Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('extrapolation_detection.png', dpi=300, bbox_inches='tight')
    print("外挿領域検出を extrapolation_detection.png に保存しました")
    plt.show()
    
    print(f"\n外れ値数: {np.sum(outlier_labels == -1)} / {len(outlier_labels)}")
    

### コード例30: 総合レポートの生成
    
    
    import json
    from datetime import datetime
    
    def generate_comprehensive_report(
        model, dataset, embeddings, targets,
        materials_df, cluster_labels, test_predictions, test_targets
    ):
        """
        総合的な分析レポートを生成
    
        Parameters:
        -----------
        model : nn.Module
            学習済みモデル
        dataset : Dataset
            データセット
        embeddings : np.ndarray
            埋め込み
        targets : np.ndarray
            ターゲット
        materials_df : pd.DataFrame
            材料データ
        cluster_labels : np.ndarray
            クラスタラベル
        test_predictions : np.ndarray
            テスト予測値
        test_targets : np.ndarray
            テスト真値
    
        Returns:
        --------
        report : dict
            レポート辞書
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
        # モデル性能
        mae = mean_absolute_error(test_targets, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        r2 = r2_score(test_targets, test_predictions)
    
        # 埋め込み統計
        embedding_stats = {
            'dimension': embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        }
    
        # クラスタ統計
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(mask)),
                'mean_band_gap': float(np.mean(targets[mask])),
                'std_band_gap': float(np.std(targets[mask]))
            }
    
        # レポート
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(dataset),
                'model_parameters': sum(p.numel() for p in model.parameters())
            },
            'model_performance': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2)
            },
            'embedding_statistics': embedding_stats,
            'cluster_statistics': cluster_stats,
            'target_statistics': {
                'mean': float(np.mean(targets)),
                'std': float(np.std(targets)),
                'min': float(np.min(targets)),
                'max': float(np.max(targets))
            }
        }
    
        return report
    
    
    # レポート生成
    report = generate_comprehensive_report(
        model, dataset, embeddings, targets,
        materials_df, cluster_labels,
        test_predictions, test_targets
    )
    
    # JSON形式で保存
    with open('comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("総合レポートを comprehensive_report.json に保存しました\n")
    print("レポートサマリー:")
    print(json.dumps(report, indent=2)[:1000] + "...")
    
    # Markdownレポート生成
    markdown_report = f"""
    # 材料空間マッピング - 総合レポート
    
    生成日時: {report['metadata']['timestamp']}
    
    ## 1. データセット情報
    
    - データセットサイズ: {report['metadata']['dataset_size']}
    - モデルパラメータ数: {report['metadata']['model_parameters']:,}
    
    ## 2. モデル性能
    
    | 指標 | 値 |
    |------|-----|
    | MAE  | {report['model_performance']['MAE']:.4f} |
    | RMSE | {report['model_performance']['RMSE']:.4f} |
    | R²   | {report['model_performance']['R2']:.4f} |
    
    ## 3. 埋め込み統計
    
    - 次元数: {report['embedding_statistics']['dimension']}
    - 平均ノルム: {report['embedding_statistics']['mean_norm']:.3f}
    - ノルム標準偏差: {report['embedding_statistics']['std_norm']:.3f}
    
    ## 4. ターゲット特性統計
    
    | 統計量 | 値 (eV) |
    |--------|---------|
    | 平均   | {report['target_statistics']['mean']:.3f} |
    | 標準偏差 | {report['target_statistics']['std']:.3f} |
    | 最小値 | {report['target_statistics']['min']:.3f} |
    | 最大値 | {report['target_statistics']['max']:.3f} |
    
    ## 5. クラスタ分析
    
    """
    
    for cluster_id, stats in report['cluster_statistics'].items():
        markdown_report += f"- **{cluster_id}**: サイズ={stats['size']}, 平均バンドギャップ={stats['mean_band_gap']:.3f} eV\n"
    
    markdown_report += """
    
    ## 6. 生成された可視化
    
    - `training_curve.png`: 学習曲線
    - `test_evaluation.png`: テスト評価
    - `pca_2d_projection.png`: PCA 2D射影
    - `umap_2d_projection.png`: UMAP 2D射影
    - `tsne_2d_projection.png`: t-SNE 2D射影
    - `clustering_analysis.png`: クラスタリング分析
    - `materials_3d_interactive.html`: インタラクティブ3D可視化
    - `materials_interactive.html`: インタラクティブ2D可視化
    
    ## 7. まとめ
    
    本分析では、GNNによる材料表現学習と次元削減を組み合わせることで、
    高次元の材料空間を効果的に可視化し、材料間の類似性や特性トレンドを
    明らかにすることができました。
    
    """
    
    # Markdownレポートの保存
    with open('comprehensive_report.md', 'w') as f:
        f.write(markdown_report)
    
    print("\nMarkdownレポートを comprehensive_report.md に保存しました")
    

## 4.8 まとめ

本章では、GNNと次元削減を組み合わせた実践的な材料マッピングシステムを構築しました。

### 実装した機能

機能 | コード例数 | 主な内容  
---|---|---  
データ収集・準備 | 例1-6 | MP API、グラフ変換、データセット  
モデル学習 | 例7-10 | CGCNN、学習ループ、評価  
埋め込み抽出・次元削減 | 例11-15 | PCA、UMAP、t-SNE、比較  
材料空間分析 | 例16-20 | クラスタリング、近傍探索、推薦  
インタラクティブ可視化 | 例21-23 | Plotly 3D、Bokeh、Dash  
高度な分析 | 例24-30 | ボロノイ、勾配、外挿、レポート  
  
### ベストプラクティス

  1. **データ前処理** : 標準化、欠損値処理、外れ値除去
  2. **モデル学習** : Early Stopping、学習率スケジューリング、正則化
  3. **次元削減** : 複数手法の比較、パラメータチューニング
  4. **可視化** : インタラクティブ性、複数視点、説明性
  5. **再現性** : 乱数シード固定、設定保存、バージョン管理

### 応用可能性

  * **材料探索** : 目標特性を持つ材料の推薦
  * **特性予測** : 新規材料の特性予測
  * **構造-特性関係** : 材料空間での構造と特性の関係解明
  * **実験計画** : 次に合成すべき材料の提案

### 今後の発展

  * **マルチタスク学習** : 複数特性の同時予測
  * **転移学習** : 小規模データセットへの適用
  * **能動学習** : 効率的なデータ収集戦略
  * **説明可能AI** : GNN予測の解釈性向上

* * *

**前章** : [第3章：GNNによる材料表現学習](<chapter-3.html>)

**シリーズトップ** : [材料特性マッピング入門](<index.html>)

## さらに学ぶために

### 推薦文献

  1. **GNN for Materials** : "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals" (Xie & Grossman, 2018)
  2. **UMAP** : "UMAP: Uniform Manifold Approximation and Projection" (McInnes et al., 2018)
  3. **Materials Informatics** : "Materials Informatics" (Ramprasad et al., 2017)

### 関連リソース

  * [Materials Project](<https://materialsproject.org/>)
  * [PyTorch Geometric](<https://pytorch-geometric.readthedocs.io/>)
  * [UMAP Documentation](<https://umap-learn.readthedocs.io/>)

お疲れ様でした！本シリーズを通じて、材料特性マッピングの基礎から実践までを学びました。
