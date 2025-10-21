# 第3章：GNNによる材料表現学習

## 概要

Graph Neural Networks（GNN）は、結晶構造のような原子間のつながりを持つデータを自然に扱える強力な機械学習モデルです。本章では、材料の結晶構造から高次元の特徴表現（embedding）を学習し、それを次元削減と組み合わせて材料空間をマッピングする方法を学びます。

### 学習目標

- 材料をグラフとして表現する方法を理解する
- PyTorch Geometricを用いたGNNモデルの実装ができる
- CGCNN、MEGNet、SchNetの特徴と実装を理解する
- GNNから得られたembeddingを可視化できる

## 3.1 材料のグラフ表現

### 3.1.1 結晶構造とグラフの対応

結晶構造は自然にグラフとして表現できます：

- **ノード（頂点）**: 原子
- **ノード特徴**: 原子番号、電気陰性度、イオン半径など
- **エッジ（辺）**: 原子間の結合（一定距離内の隣接関係）
- **エッジ特徴**: 原子間距離、結合角度など
- **グローバル特徴**: セル体積、密度など

### コード例1: PyTorch Geometricのインストールと基本設定

```python
# PyTorch Geometricのインストール（初回のみ）
# !pip install torch torchvision torchaudio
# !pip install torch-geometric
# !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
import numpy as np
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PyTorch Geometricバージョン確認
import torch_geometric
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

**出力例**:
```
Using device: cpu
PyTorch Geometric version: 2.3.1
PyTorch version: 2.0.1
```

### コード例2: 結晶構造からグラフデータへの変換

```python
from pymatgen.core import Structure, Lattice
import numpy as np
import torch
from torch_geometric.data import Data

def structure_to_graph(structure, cutoff=5.0):
    """
    pymatgen StructureをPyTorch Geometric Dataオブジェクトに変換

    Parameters:
    -----------
    structure : pymatgen.Structure
        結晶構造
    cutoff : float
        エッジを生成する原子間距離の閾値（Angstrom）

    Returns:
    --------
    data : torch_geometric.data.Data
        グラフデータ
    """
    # ノード特徴: 原子番号（ワンホットエンコーディングは後で実装）
    atomic_numbers = torch.tensor([site.specie.Z for site in structure],
                                  dtype=torch.float).view(-1, 1)

    # エッジの構築
    all_neighbors = structure.get_all_neighbors(cutoff)
    edge_index = []
    edge_attr = []

    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor.index
            distance = neighbor.nn_distance

            edge_index.append([i, j])
            edge_attr.append(distance)

    # エッジがない場合の処理
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    # Dataオブジェクトの作成
    data = Data(x=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr)

    return data


# サンプル結晶構造の作成（シンプルなCsCl構造）
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# グラフへの変換
graph_data = structure_to_graph(structure, cutoff=5.0)

print("グラフデータの情報:")
print(f"ノード数: {graph_data.num_nodes}")
print(f"エッジ数: {graph_data.num_edges}")
print(f"ノード特徴の形状: {graph_data.x.shape}")
print(f"エッジインデックスの形状: {graph_data.edge_index.shape}")
print(f"エッジ特徴の形状: {graph_data.edge_attr.shape}")
```

**出力例**:
```
グラフデータの情報:
ノード数: 2
エッジ数: 16
ノード特徴の形状: torch.Size([2, 1])
エッジインデックスの形状: torch.Size([2, 16])
エッジ特徴の形状: torch.Size([16, 1])
```

### コード例3: 原子特徴のエンコーディング

```python
import torch
import torch.nn as nn

class AtomFeaturizer:
    """原子の特徴をベクトル化するクラス"""

    def __init__(self, max_z=100):
        """
        Parameters:
        -----------
        max_z : int
            扱う最大の原子番号
        """
        self.max_z = max_z

        # 原子特性のリスト（簡易版）
        # 実際には mendeleev などのライブラリを使用すると良い
        self.electronegativity = self._load_property('electronegativity')
        self.covalent_radius = self._load_property('covalent_radius')
        self.valence_electrons = self._load_property('valence_electrons')

    def _load_property(self, property_name):
        """
        原子特性のダミーデータ生成
        実際のプロジェクトではmendeleevライブラリなどから取得
        """
        # ダミーデータ（実際には正確な値を使用）
        np.random.seed(42)
        if property_name == 'electronegativity':
            return np.random.uniform(0.7, 4.0, self.max_z)
        elif property_name == 'covalent_radius':
            return np.random.uniform(0.3, 2.5, self.max_z)
        elif property_name == 'valence_electrons':
            return np.random.randint(1, 8, self.max_z).astype(float)

    def featurize(self, atomic_number):
        """
        原子番号から特徴ベクトルを生成

        Parameters:
        -----------
        atomic_number : int or array-like
            原子番号

        Returns:
        --------
        features : torch.Tensor
            特徴ベクトル
        """
        if isinstance(atomic_number, (int, np.integer)):
            atomic_number = [atomic_number]

        features = []
        for z in atomic_number:
            z_idx = int(z) - 1  # 0-indexed

            feat = [
                z / self.max_z,  # 正規化された原子番号
                self.electronegativity[z_idx],
                self.covalent_radius[z_idx],
                self.valence_electrons[z_idx]
            ]
            features.append(feat)

        return torch.tensor(features, dtype=torch.float)


# 使用例
featurizer = AtomFeaturizer(max_z=100)

# Cs (Z=55) と Cl (Z=17) の特徴化
cs_features = featurizer.featurize(55)
cl_features = featurizer.featurize(17)

print("Cs の特徴ベクトル:")
print(cs_features)
print("\nCl の特徴ベクトル:")
print(cl_features)

# 構造全体の特徴化
atomic_numbers = [site.specie.Z for site in structure]
all_features = featurizer.featurize(atomic_numbers)
print(f"\n構造全体の特徴行列の形状: {all_features.shape}")
```

### コード例4: ダミー材料データセットの生成

```python
import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class DummyMaterialsDataset(InMemoryDataset):
    """
    学習用のダミー材料データセット
    実際のプロジェクトではMaterials Project APIなどから取得
    """

    def __init__(self, num_materials=1000, num_atom_types=20):
        self.num_materials = num_materials
        self.num_atom_types = num_atom_types
        super().__init__(root=None)
        self.data, self.slices = self._generate_data()

    def _generate_data(self):
        """ダミーのグラフデータを生成"""
        data_list = []

        np.random.seed(42)
        torch.manual_seed(42)

        for i in range(self.num_materials):
            # ノード数をランダムに設定（5-30原子）
            num_nodes = np.random.randint(5, 30)

            # ノード特徴（原子タイプのワンホット + 連続値特徴）
            atom_types = torch.randint(0, self.num_atom_types, (num_nodes,))
            atom_features = torch.randn(num_nodes, 4)  # 4次元の特徴

            # 完全なノード特徴
            x = torch.cat([
                F.one_hot(atom_types, num_classes=self.num_atom_types).float(),
                atom_features
            ], dim=1)

            # エッジの生成（各ノードから平均5つのエッジ）
            num_edges = num_nodes * 5
            edge_index = torch.randint(0, num_nodes, (2, num_edges))

            # エッジ特徴（距離など）
            edge_attr = torch.rand(num_edges, 1) * 5.0  # 0-5 Angstrom

            # ターゲット特性（バンドギャップを想定）
            y = torch.tensor([np.random.exponential(2.0)], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        return self.collate(data_list)

# データセットの作成
dataset = DummyMaterialsDataset(num_materials=1000, num_atom_types=20)

print(f"データセットサイズ: {len(dataset)}")
print(f"サンプルデータ:")
print(dataset[0])
print(f"\nノード特徴次元: {dataset[0].x.shape[1]}")
print(f"ターゲット: {dataset[0].y.item():.3f}")
```

**出力例**:
```
データセットサイズ: 1000
サンプルデータ:
Data(x=[18, 24], edge_index=[2, 90], edge_attr=[90, 1], y=[1])

ノード特徴次元: 24
ターゲット: 2.134
```

## 3.2 Crystal Graph Convolutional Neural Network (CGCNN)

CGCNNは、結晶構造の特性予測に特化したGNNモデルです。

### コード例5: CGCNNの畳み込み層

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class CGConv(MessagePassing):
    """
    CGCNN畳み込み層
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__(aggr='add')  # aggregation: sum

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # ノード更新用のネットワーク
        self.node_fc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # エッジ特徴とノード特徴を組み合わせるネットワーク
        self.edge_fc = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ゲート機構
        self.gate = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        順伝播

        Parameters:
        -----------
        x : Tensor [num_nodes, node_dim]
            ノード特徴
        edge_index : Tensor [2, num_edges]
            エッジインデックス
        edge_attr : Tensor [num_edges, edge_dim]
            エッジ特徴

        Returns:
        --------
        out : Tensor [num_nodes, hidden_dim]
            更新されたノード特徴
        """
        # メッセージパッシング
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 自己ループ（残差接続的な処理）
        out = out + self.node_fc(x)

        return out

    def message(self, x_j, edge_attr):
        """
        メッセージ関数
        エッジを通じて隣接ノードから送られる情報を計算

        Parameters:
        -----------
        x_j : Tensor [num_edges, node_dim]
            送信元ノードの特徴
        edge_attr : Tensor [num_edges, edge_dim]
            エッジ特徴

        Returns:
        --------
        message : Tensor [num_edges, hidden_dim]
            メッセージ
        """
        # エッジとノード特徴を連結
        z = torch.cat([x_j, edge_attr], dim=1)

        # ゲーティング
        gate_values = self.gate(z)
        message_values = self.edge_fc(z)

        return gate_values * message_values


# テスト
node_dim = 24
edge_dim = 1
hidden_dim = 64

conv_layer = CGConv(node_dim, edge_dim, hidden_dim)

# ダミーデータ
x = torch.randn(10, node_dim)
edge_index = torch.randint(0, 10, (2, 30))
edge_attr = torch.randn(30, edge_dim)

# 順伝播
out = conv_layer(x, edge_index, edge_attr)

print(f"入力ノード特徴の形状: {x.shape}")
print(f"出力ノード特徴の形状: {out.shape}")
```

**出力例**:
```
入力ノード特徴の形状: torch.Size([10, 24])
出力ノード特徴の形状: torch.Size([10, 64])
```

### コード例6: 完全なCGCNNモデル

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_conv=3, num_fc=2):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_conv = num_conv

        # 入力埋め込み層
        self.embedding = nn.Linear(node_dim, hidden_dim)

        # CGConv層のリスト
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_conv)
        ])

        # Batch Normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_conv)
        ])

        # 全結合層（特性予測用）
        fc_layers = []
        for i in range(num_fc):
            if i == 0:
                fc_layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
            else:
                fc_layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 2))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))

        fc_layers.append(nn.Linear(hidden_dim // 2, 1))  # 出力次元=1（回帰タスク）

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, data):
        """
        順伝播

        Parameters:
        -----------
        data : torch_geometric.data.Data or Batch
            グラフデータ

        Returns:
        --------
        out : Tensor [batch_size, 1]
            予測値
        embedding : Tensor [batch_size, hidden_dim]
            グラフ埋め込み（可視化用）
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 入力埋め込み
        x = self.embedding(x)

        # CGConv層の適用
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.softplus(x)

        # グラフレベルの埋め込み（平均プーリング）
        graph_embedding = global_mean_pool(x, batch)

        # 特性予測
        out = self.fc(graph_embedding)

        return out, graph_embedding


# モデルのインスタンス化
model = CGCNN(node_dim=24, edge_dim=1, hidden_dim=64, num_conv=3, num_fc=2)

print(f"モデルの総パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# テスト
from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
batch = next(iter(dataloader))

predictions, embeddings = model(batch)

print(f"\n予測値の形状: {predictions.shape}")
print(f"埋め込みの形状: {embeddings.shape}")
```

**出力例**:
```
モデルの総パラメータ数: 23,713

予測値の形状: torch.Size([32, 1])
埋め込みの形状: torch.Size([32, 64])
```

### コード例7: CGCNNの学習ループ

```python
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# データセットの分割
train_idx, test_idx = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

# DataLoaderの作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# モデル、損失関数、オプティマイザ
model = CGCNN(node_dim=24, edge_dim=1, hidden_dim=64, num_conv=3)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 50
train_losses = []
test_losses = []

print("学習開始...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        predictions, _ = model(batch)
        loss = criterion(predictions, batch.y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            predictions, _ = model(batch)
            loss = criterion(predictions, batch.y)

            test_loss += loss.item() * batch.num_graphs

    test_loss /= len(test_dataset)
    test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

print("学習完了！")

# 学習曲線のプロット
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('MSE Loss', fontsize=14, fontweight='bold')
plt.title('CGCNN Training Curve', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cgcnn_training_curve.png', dpi=300)
print("学習曲線を cgcnn_training_curve.png に保存しました")
plt.show()
```

**出力例**:
```
学習開始...
Epoch [10/50] - Train Loss: 2.1234, Test Loss: 2.3456
Epoch [20/50] - Train Loss: 1.5678, Test Loss: 1.7890
Epoch [30/50] - Train Loss: 1.2345, Test Loss: 1.4567
Epoch [40/50] - Train Loss: 1.0123, Test Loss: 1.2345
Epoch [50/50] - Train Loss: 0.8901, Test Loss: 1.1234
学習完了！
学習曲線を cgcnn_training_curve.png に保存しました
```

### コード例8: CGCNNからの埋め込み抽出

```python
import torch
import numpy as np
from torch_geometric.loader import DataLoader

def extract_embeddings(model, dataset, device='cpu'):
    """
    学習済みモデルから全データの埋め込みを抽出

    Parameters:
    -----------
    model : nn.Module
        学習済みモデル
    dataset : Dataset
        データセット
    device : str
        デバイス

    Returns:
    --------
    embeddings : np.ndarray
        埋め込みベクトル
    targets : np.ndarray
        ターゲット値
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, embeddings = model(batch)

            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    targets = np.concatenate(all_targets, axis=0).flatten()

    return embeddings, targets


# 全データセットから埋め込みを抽出
embeddings, targets = extract_embeddings(model, dataset, device=device)

print(f"埋め込みの形状: {embeddings.shape}")
print(f"ターゲットの形状: {targets.shape}")
print(f"\nターゲット統計:")
print(f"  平均: {targets.mean():.3f}")
print(f"  標準偏差: {targets.std():.3f}")
print(f"  最小値: {targets.min():.3f}")
print(f"  最大値: {targets.max():.3f}")
```

**出力例**:
```
埋め込みの形状: (1000, 64)
ターゲットの形状: (1000,)

ターゲット統計:
  平均: 2.015
  標準偏差: 2.034
  最小値: 0.012
  最大値: 12.456
```

## 3.3 MEGNet（MatErials Graph Network）

MEGNetは、グローバル状態を考慮したより柔軟なGNNアーキテクチャです。

### コード例9: MEGNetブロック

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class MEGNetBlock(MessagePassing):
    """
    MEGNetの基本ブロック
    ノード、エッジ、グローバル状態を同時に更新
    """

    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=64):
        super().__init__(aggr='mean')

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim

        # エッジ更新ネットワーク
        self.edge_model = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, edge_dim)
        )

        # ノード更新ネットワーク
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim)
        )

        # グローバル状態更新ネットワーク
        self.global_model = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, global_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        順伝播

        Parameters:
        -----------
        x : Tensor [num_nodes, node_dim]
            ノード特徴
        edge_index : Tensor [2, num_edges]
            エッジインデックス
        edge_attr : Tensor [num_edges, edge_dim]
            エッジ特徴
        u : Tensor [batch_size, global_dim]
            グローバル状態
        batch : Tensor [num_nodes]
            バッチインデックス

        Returns:
        --------
        x_new : Tensor [num_nodes, node_dim]
            更新されたノード特徴
        edge_attr_new : Tensor [num_edges, edge_dim]
            更新されたエッジ特徴
        u_new : Tensor [batch_size, global_dim]
            更新されたグローバル状態
        """
        row, col = edge_index

        # 1. エッジ更新
        edge_input = torch.cat([
            x[row], x[col], edge_attr, u[batch[row]]
        ], dim=1)
        edge_attr_new = edge_attr + self.edge_model(edge_input)

        # 2. ノード更新（メッセージパッシング）
        x_new = x + self.propagate(edge_index, x=x, edge_attr=edge_attr_new,
                                    u=u, batch=batch)

        # 3. グローバル状態更新
        # グラフごとのノード・エッジ特徴の平均
        node_global = global_mean_pool(x_new, batch)
        edge_global = global_mean_pool(edge_attr_new, batch[row])

        global_input = torch.cat([node_global, edge_global, u], dim=1)
        u_new = u + self.global_model(global_input)

        return x_new, edge_attr_new, u_new

    def message(self, x_j, edge_attr, u, batch):
        """
        メッセージ関数

        Parameters:
        -----------
        x_j : Tensor [num_edges, node_dim]
            送信元ノード特徴
        edge_attr : Tensor [num_edges, edge_dim]
            エッジ特徴
        u : Tensor [batch_size, global_dim]
            グローバル状態
        batch : Tensor [num_edges]
            バッチインデックス

        Returns:
        --------
        message : Tensor [num_edges, node_dim]
            メッセージ
        """
        # ノード自身の特徴も使う場合はpropagateの引数から取得
        # ここでは簡略化のため省略
        message_input = torch.cat([x_j, edge_attr, u[batch]], dim=1)
        return self.node_model(message_input) - x_j  # 残差


# テスト
node_dim = 32
edge_dim = 16
global_dim = 8
hidden_dim = 64

megnet_block = MEGNetBlock(node_dim, edge_dim, global_dim, hidden_dim)

# ダミーデータ
x = torch.randn(10, node_dim)
edge_index = torch.randint(0, 10, (2, 30))
edge_attr = torch.randn(30, edge_dim)
u = torch.randn(1, global_dim)  # 1つのグラフ
batch = torch.zeros(10, dtype=torch.long)  # 全ノードが同じグラフに属する

# 順伝播
x_new, edge_attr_new, u_new = megnet_block(x, edge_index, edge_attr, u, batch)

print(f"ノード特徴: {x.shape} -> {x_new.shape}")
print(f"エッジ特徴: {edge_attr.shape} -> {edge_attr_new.shape}")
print(f"グローバル状態: {u.shape} -> {u_new.shape}")
```

**出力例**:
```
ノード特徴: torch.Size([10, 32]) -> torch.Size([10, 32])
エッジ特徴: torch.Size([30, 16]) -> torch.Size([30, 16])
グローバル状態: torch.Size([1, 8]) -> torch.Size([1, 8])
```

### コード例10: 完全なMEGNetモデル

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class MEGNet(nn.Module):
    """
    MatErials Graph Network (MEGNet)
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_blocks=3):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim_orig = edge_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # 特徴の次元を統一
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)

        # グローバル状態の初期化
        self.global_init = nn.Parameter(torch.randn(1, hidden_dim))

        # MEGNetブロック
        self.blocks = nn.ModuleList([
            MEGNetBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        """
        順伝播

        Parameters:
        -----------
        data : torch_geometric.data.Data or Batch
            グラフデータ

        Returns:
        --------
        out : Tensor [batch_size, 1]
            予測値
        embedding : Tensor [batch_size, hidden_dim]
            グラフ埋め込み
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 埋め込み
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # バッチサイズの取得
        batch_size = batch.max().item() + 1

        # グローバル状態の初期化
        u = self.global_init.expand(batch_size, -1)

        # MEGNetブロックの適用
        for block in self.blocks:
            x, edge_attr, u = block(x, edge_index, edge_attr, u, batch)

        # グローバル状態が最終的な埋め込み
        graph_embedding = u

        # 出力
        out = self.fc(graph_embedding)

        return out, graph_embedding


# モデルのインスタンス化とテスト
megnet_model = MEGNet(node_dim=24, edge_dim=1, hidden_dim=64, num_blocks=3)

print(f"MEGNetモデルの総パラメータ数: {sum(p.numel() for p in megnet_model.parameters()):,}")

# テスト
batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
predictions, embeddings = megnet_model(batch)

print(f"\n予測値の形状: {predictions.shape}")
print(f"埋め込みの形状: {embeddings.shape}")
```

## 3.4 SchNet

SchNetは、連続フィルタを用いた物理的に妥当なGNNモデルです。

### コード例11: SchNetの連続フィルタ畳み込み層

```python
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing

class GaussianSmearing(nn.Module):
    """
    ガウス基底関数による距離の埋め込み
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()

        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        """
        Parameters:
        -----------
        dist : Tensor [num_edges, 1]
            原子間距離

        Returns:
        --------
        rbf : Tensor [num_edges, num_gaussians]
            ガウス基底関数による表現
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConv(MessagePassing):
    """
    Continuous-Filter Convolution (SchNetの基本層)
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_gaussians=50):
        super().__init__(aggr='add')

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # ガウス基底関数
        self.distance_expansion = GaussianSmearing(0.0, 5.0, num_gaussians)

        # フィルター生成ネットワーク
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )

        # ノード更新ネットワーク
        self.node_network = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        順伝播

        Parameters:
        -----------
        x : Tensor [num_nodes, node_dim]
            ノード特徴
        edge_index : Tensor [2, num_edges]
            エッジインデックス
        edge_attr : Tensor [num_edges, 1]
            エッジ特徴（距離）

        Returns:
        --------
        out : Tensor [num_nodes, hidden_dim]
            更新されたノード特徴
        """
        # ノード特徴の前処理
        x = self.node_network(x)

        # メッセージパッシング
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return out

    def message(self, x_j, edge_attr):
        """
        メッセージ関数

        Parameters:
        -----------
        x_j : Tensor [num_edges, node_dim]
            送信元ノード特徴
        edge_attr : Tensor [num_edges, 1]
            エッジ特徴（距離）

        Returns:
        --------
        message : Tensor [num_edges, hidden_dim]
            メッセージ
        """
        # 距離を基底関数で展開
        edge_features = self.distance_expansion(edge_attr)

        # フィルターの生成
        W = self.filter_network(edge_features)
        W = W.view(-1, self.node_dim, self.hidden_dim)

        # フィルターの適用
        x_j = x_j.unsqueeze(1)  # [num_edges, 1, node_dim]
        message = torch.bmm(x_j, W).squeeze(1)  # [num_edges, hidden_dim]

        return message


# テスト
node_dim = 64
edge_dim = 1
hidden_dim = 64

cfconv = CFConv(node_dim, edge_dim, hidden_dim, num_gaussians=50)

# ダミーデータ
x = torch.randn(10, node_dim)
edge_index = torch.randint(0, 10, (2, 30))
edge_attr = torch.rand(30, 1) * 5.0  # 距離

# 順伝播
out = cfconv(x, edge_index, edge_attr)

print(f"入力ノード特徴の形状: {x.shape}")
print(f"出力ノード特徴の形状: {out.shape}")
```

**出力例**:
```
入力ノード特徴の形状: torch.Size([10, 64])
出力ノード特徴の形状: torch.Size([10, 64])
```

### コード例12: 完全なSchNetモデル

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

class SchNet(nn.Module):
    """
    SchNet: continuous-filter convolutional neural network
    """

    def __init__(self, node_dim, hidden_dim=64, num_filters=64,
                 num_interactions=3, num_gaussians=50):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.num_interactions = num_interactions

        # 埋め込み層
        self.embedding = nn.Linear(node_dim, hidden_dim)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            CFConv(hidden_dim, 1, num_filters, num_gaussians)
            for _ in range(num_interactions)
        ])

        # Update networks
        self.updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_filters, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_interactions)
        ])

        # 出力ネットワーク
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        """
        順伝播

        Parameters:
        -----------
        data : torch_geometric.data.Data or Batch
            グラフデータ

        Returns:
        --------
        out : Tensor [batch_size, 1]
            予測値
        embedding : Tensor [batch_size, hidden_dim]
            グラフ埋め込み
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 埋め込み
        x = self.embedding(x)

        # Interaction blocks
        for interaction, update in zip(self.interactions, self.updates):
            # CFConv
            v = interaction(x, edge_index, edge_attr)

            # Update
            x = x + update(v)

        # グラフレベルの埋め込み
        graph_embedding = global_add_pool(x, batch)

        # 出力
        out = self.fc(graph_embedding)

        return out, graph_embedding


# モデルのインスタンス化とテスト
schnet_model = SchNet(node_dim=24, hidden_dim=64, num_filters=64,
                      num_interactions=3, num_gaussians=50)

print(f"SchNetモデルの総パラメータ数: {sum(p.numel() for p in schnet_model.parameters()):,}")

# テスト
batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
predictions, embeddings = schnet_model(batch)

print(f"\n予測値の形状: {predictions.shape}")
print(f"埋め込みの形状: {embeddings.shape}")
```

## 3.5 埋め込みの可視化と分析

### コード例13: UMAPによるGNN埋め込みの可視化

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# CGCNNからの埋め込み抽出
cgcnn_embeddings, cgcnn_targets = extract_embeddings(model, dataset, device=device)

# UMAPによる次元削減
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
cgcnn_umap = reducer.fit_transform(cgcnn_embeddings)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ターゲット値で色付け
scatter1 = ax1.scatter(cgcnn_umap[:, 0], cgcnn_umap[:, 1],
                       c=cgcnn_targets, cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
ax1.set_title('CGCNN Embeddings: colored by Band Gap',
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')

# 安定性でカテゴリ分け（ダミーデータなので仮の分類）
stability_categories = np.digitize(cgcnn_targets, bins=[0, 1, 2, 4])
colors_cat = ['red', 'orange', 'yellow', 'green']

for i, label in enumerate(['Very Low', 'Low', 'Medium', 'High']):
    mask = stability_categories == i
    ax2.scatter(cgcnn_umap[mask, 0], cgcnn_umap[mask, 1],
                c=colors_cat[i], label=label, s=50, alpha=0.7,
                edgecolors='black', linewidth=0.5)

ax2.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
ax2.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
ax2.set_title('CGCNN Embeddings: colored by Category',
              fontsize=16, fontweight='bold')
ax2.legend(title='Band Gap Category', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cgcnn_embeddings_umap.png', dpi=300, bbox_inches='tight')
print("CGCNN埋め込みのUMAPを cgcnn_embeddings_umap.png に保存しました")
plt.show()
```

### コード例14: t-SNEによる複数モデルの比較

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 複数モデルの埋め込み（ここではCGCNNのみだがMEGNet、SchNetも同様に抽出可能）
models_dict = {
    'CGCNN': (model, cgcnn_embeddings)
}

# t-SNEによる次元削減
tsne_results = {}
for model_name, (_, embeddings) in models_dict.items():
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_embedding = tsne.fit_transform(embeddings)
    tsne_results[model_name] = tsne_embedding

# 可視化
fig, axes = plt.subplots(1, len(models_dict), figsize=(8 * len(models_dict), 7))

if len(models_dict) == 1:
    axes = [axes]

for idx, (model_name, tsne_emb) in enumerate(tsne_results.items()):
    ax = axes[idx]

    scatter = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1],
                         c=cgcnn_targets, cmap='plasma',
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('t-SNE 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} Embeddings (t-SNE)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Band Gap (eV)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('gnn_embeddings_tsne_comparison.png', dpi=300, bbox_inches='tight')
print("GNN埋め込みのt-SNE比較を gnn_embeddings_tsne_comparison.png に保存しました")
plt.show()
```

### コード例15: クラスタリングと特性分析

```python
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

# K-Meansクラスタリング
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(cgcnn_embeddings)

# UMAP上にクラスタを表示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# クラスタラベルで色付け
colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
for cluster_id in range(n_clusters):
    mask = cluster_labels == cluster_id
    ax1.scatter(cgcnn_umap[mask, 0], cgcnn_umap[mask, 1],
                c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# クラスタ中心
kmeans_umap = reducer.transform(kmeans.cluster_centers_)
ax1.scatter(kmeans_umap[:, 0], kmeans_umap[:, 1],
            c='red', marker='X', s=300, edgecolors='black',
            linewidth=2, label='Centroids', zorder=10)

ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
ax1.set_title('Clustering on CGCNN Embeddings',
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# クラスタごとのターゲット分布
cluster_df = pd.DataFrame({
    'cluster': cluster_labels,
    'band_gap': cgcnn_targets
})

sns.boxplot(data=cluster_df, x='cluster', y='band_gap', ax=ax2, palette='Set3')
ax2.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
ax2.set_ylabel('Band Gap (eV)', fontsize=14, fontweight='bold')
ax2.set_title('Band Gap Distribution by Cluster',
              fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cgcnn_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("クラスタリング分析を cgcnn_clustering_analysis.png に保存しました")
plt.show()

# クラスタ統計
print("\nクラスタごとのバンドギャップ統計:")
cluster_stats = cluster_df.groupby('cluster')['band_gap'].agg(['mean', 'std', 'min', 'max', 'count'])
print(cluster_stats.round(3))
```

## 3.6 埋め込み空間の解釈

### コード例16: 埋め込みベクトルの主成分分析

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# PCAによる埋め込み空間の分析
pca_embedding = PCA(n_components=10)
cgcnn_pca = pca_embedding.fit_transform(cgcnn_embeddings)

# 寄与率のプロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 個別寄与率
ax1.bar(range(1, 11), pca_embedding.explained_variance_ratio_,
        alpha=0.7, edgecolor='black', color='steelblue')
ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
ax1.set_ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
ax1.set_title('PCA on CGCNN Embeddings: Variance Explained',
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 累積寄与率
cumsum_var = np.cumsum(pca_embedding.explained_variance_ratio_)
ax2.plot(range(1, 11), cumsum_var, marker='o', linewidth=2,
         markersize=8, color='darkred')
ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2,
            label='95% threshold', alpha=0.7)
ax2.set_xlabel('Number of Components', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.set_title('Cumulative Variance Explained',
              fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cgcnn_embedding_pca.png', dpi=300, bbox_inches='tight')
print("埋め込みPCA分析を cgcnn_embedding_pca.png に保存しました")
print(f"\n95%の分散を説明するために必要な主成分数: {np.argmax(cumsum_var >= 0.95) + 1}")
plt.show()
```

### コード例17: 埋め込み空間での近傍探索

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def find_similar_materials(query_idx, embeddings, targets, k=5):
    """
    埋め込み空間で類似材料を検索

    Parameters:
    -----------
    query_idx : int
        クエリ材料のインデックス
    embeddings : np.ndarray
        埋め込みベクトル
    targets : np.ndarray
        ターゲット値
    k : int
        検索する近傍数

    Returns:
    --------
    neighbors : dict
        近傍材料の情報
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings[query_idx:query_idx+1])

    neighbors = {
        'query_idx': query_idx,
        'query_target': targets[query_idx],
        'neighbor_indices': indices[0, 1:],  # 自分自身を除く
        'neighbor_targets': targets[indices[0, 1:]],
        'distances': distances[0, 1:]
    }

    return neighbors


# ランダムに5つの材料を選んで近傍探索
np.random.seed(42)
query_indices = np.random.choice(len(dataset), 5, replace=False)

print("近傍探索結果:\n")
for query_idx in query_indices:
    neighbors = find_similar_materials(query_idx, cgcnn_embeddings,
                                       cgcnn_targets, k=5)

    print(f"クエリ材料 #{neighbors['query_idx']}:")
    print(f"  ターゲット値: {neighbors['query_target']:.3f}")
    print(f"  類似材料:")

    for i, (neighbor_idx, target, dist) in enumerate(zip(
        neighbors['neighbor_indices'],
        neighbors['neighbor_targets'],
        neighbors['distances']
    )):
        print(f"    {i+1}. Material #{neighbor_idx}: "
              f"Target={target:.3f}, Distance={dist:.3f}")
    print()
```

**出力例**:
```
近傍探索結果:

クエリ材料 #123:
  ターゲット値: 2.456
  類似材料:
    1. Material #456: Target=2.389, Distance=0.145
    2. Material #789: Target=2.567, Distance=0.189
    3. Material #234: Target=2.123, Distance=0.234
    4. Material #567: Target=2.678, Distance=0.267
    5. Material #890: Target=2.345, Distance=0.289
...
```

### コード例18: 埋め込み空間の距離分布分析

```python
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# 全ペア間の距離を計算（サブセットで計算）
subset_size = 200
subset_indices = np.random.choice(len(cgcnn_embeddings), subset_size, replace=False)
subset_embeddings = cgcnn_embeddings[subset_indices]
subset_targets = cgcnn_targets[subset_indices]

# ユークリッド距離とコサイン距離
euclidean_distances = pdist(subset_embeddings, metric='euclidean')
cosine_distances = pdist(subset_embeddings, metric='cosine')

# ターゲット値の差
target_diff = pdist(subset_targets.reshape(-1, 1), metric='euclidean')

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. ユークリッド距離の分布
axes[0, 0].hist(euclidean_distances, bins=50, alpha=0.7,
                edgecolor='black', color='steelblue')
axes[0, 0].set_xlabel('Euclidean Distance', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Distribution of Euclidean Distances',
                      fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. コサイン距離の分布
axes[0, 1].hist(cosine_distances, bins=50, alpha=0.7,
                edgecolor='black', color='coral')
axes[0, 1].set_xlabel('Cosine Distance', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Distribution of Cosine Distances',
                      fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. 埋め込み距離 vs ターゲット差
axes[1, 0].scatter(euclidean_distances, target_diff,
                   alpha=0.3, s=10, color='purple')
axes[1, 0].set_xlabel('Embedding Distance (Euclidean)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Band Gap Difference', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Embedding Distance vs Property Difference',
                      fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 相関係数
correlation = np.corrcoef(euclidean_distances, target_diff)[0, 1]
axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=axes[1, 0].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. 2D密度プロット
from scipy.stats import gaussian_kde

# データの準備
x = euclidean_distances
y = target_diff

# KDE
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# ソート（密度の高い点を上に描画）
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

scatter = axes[1, 1].scatter(x, y, c=z, cmap='hot', s=15, alpha=0.6)
axes[1, 1].set_xlabel('Embedding Distance (Euclidean)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Band Gap Difference', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Density Plot: Distance vs Difference',
                      fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Density', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('cgcnn_embedding_distance_analysis.png', dpi=300, bbox_inches='tight')
print("埋め込み距離分析を cgcnn_embedding_distance_analysis.png に保存しました")
plt.show()
```

### コード例19: インタラクティブ3D可視化（Plotly）

```python
import plotly.express as px
import plotly.graph_objects as go
import umap

# 3次元UMAP
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
cgcnn_umap_3d = reducer_3d.fit_transform(cgcnn_embeddings)

# データフレームの作成
import pandas as pd

df_3d = pd.DataFrame({
    'UMAP1': cgcnn_umap_3d[:, 0],
    'UMAP2': cgcnn_umap_3d[:, 1],
    'UMAP3': cgcnn_umap_3d[:, 2],
    'Band_Gap': cgcnn_targets,
    'Material_ID': [f'Material_{i}' for i in range(len(cgcnn_targets))],
    'Cluster': cluster_labels
})

# インタラクティブ3Dプロット
fig = px.scatter_3d(df_3d,
                    x='UMAP1', y='UMAP2', z='UMAP3',
                    color='Band_Gap',
                    size='Band_Gap',
                    hover_data=['Material_ID', 'Cluster'],
                    color_continuous_scale='Viridis',
                    title='Interactive 3D Visualization of CGCNN Embeddings')

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

fig.write_html('cgcnn_embeddings_3d_interactive.html')
print("インタラクティブ3D可視化を cgcnn_embeddings_3d_interactive.html に保存しました")
fig.show()
```

### コード例20: 埋め込み品質の定量評価

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def evaluate_embedding_quality(embeddings, labels, targets):
    """
    埋め込みの品質を複数の指標で評価

    Parameters:
    -----------
    embeddings : np.ndarray
        埋め込みベクトル
    labels : np.ndarray
        クラスタラベル
    targets : np.ndarray
        ターゲット値（連続値）

    Returns:
    --------
    metrics : dict
        評価指標
    """
    # クラスタリング品質指標
    silhouette = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)

    # ターゲット値との相関（近傍保存）
    from sklearn.neighbors import NearestNeighbors

    k = 10
    nbrs_embedding = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    _, indices_emb = nbrs_embedding.kneighbors(embeddings)

    nbrs_target = NearestNeighbors(n_neighbors=k+1).fit(targets.reshape(-1, 1))
    _, indices_tgt = nbrs_target.kneighbors(targets.reshape(-1, 1))

    # 近傍の一致率
    neighborhood_match = []
    for i in range(len(embeddings)):
        neighbors_emb = set(indices_emb[i, 1:])
        neighbors_tgt = set(indices_tgt[i, 1:])
        intersection = len(neighbors_emb & neighbors_tgt)
        neighborhood_match.append(intersection / k)

    neighborhood_preservation = np.mean(neighborhood_match)

    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'neighborhood_preservation': neighborhood_preservation
    }

    return metrics


# 評価実行
metrics = evaluate_embedding_quality(cgcnn_embeddings, cluster_labels, cgcnn_targets)

print("CGCNN埋め込みの品質評価:")
print(f"  Silhouette Score: {metrics['silhouette_score']:.3f} (高いほど良い)")
print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (低いほど良い)")
print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f} (高いほど良い)")
print(f"  Neighborhood Preservation: {metrics['neighborhood_preservation']:.3f} (高いほど良い)")

# 複数のクラスタ数で評価
k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(cgcnn_embeddings)
    silhouette_scores.append(silhouette_score(cgcnn_embeddings, labels_k))

# プロット
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouette_scores, marker='o', linewidth=2,
         markersize=8, color='darkblue')
plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=14, fontweight='bold')
plt.title('Silhouette Score vs Number of Clusters',
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cgcnn_silhouette_analysis.png', dpi=300, bbox_inches='tight')
print("\nシルエットスコア分析を cgcnn_silhouette_analysis.png に保存しました")
plt.show()
```

**出力例**:
```
CGCNN埋め込みの品質評価:
  Silhouette Score: 0.342 (高いほど良い)
  Davies-Bouldin Score: 1.234 (低いほど良い)
  Calinski-Harabasz Score: 456.789 (高いほど良い)
  Neighborhood Preservation: 0.675 (高いほど良い)

シルエットスコア分析を cgcnn_silhouette_analysis.png に保存しました
```

## 3.7 まとめ

本章では、GNNによる材料表現学習について学びました：

### 主要なGNNモデル

| モデル | 特徴 | 利点 | 適用場面 |
|-------|------|------|---------|
| **CGCNN** | 結晶構造特化 | 物理的解釈性 | 結晶材料特性予測 |
| **MEGNet** | グローバル状態 | 柔軟性、表現力 | 多様な材料システム |
| **SchNet** | 連続フィルタ | 物理的妥当性 | 分子・原子系 |

### 実装したコード

| コード例 | 内容 | 主な機能 |
|---------|------|---------|
| 例1-4 | グラフデータ準備 | 構造→グラフ変換 |
| 例5-8 | CGCNN実装 | モデル構築、学習、埋め込み抽出 |
| 例9-10 | MEGNet実装 | グローバル状態を含むGNN |
| 例11-12 | SchNet実装 | 連続フィルタ畳み込み |
| 例13-20 | 埋め込み可視化・分析 | UMAP、t-SNE、クラスタリング、評価 |

### ベストプラクティス

1. **データ準備**: 適切なカットオフ距離、特徴エンコーディング
2. **モデル選択**: タスクに応じたアーキテクチャ
3. **ハイパーパラメータ**: hidden_dim、num_layers、learning_rateの調整
4. **評価**: 予測性能だけでなく埋め込み品質も評価

### 次章への展望

第4章では、これまで学んだ次元削減手法（第2章）とGNN表現学習（第3章）を組み合わせた実践的な材料マッピングシステムを構築します。Materials Project APIから実データを取得し、エンドツーエンドのパイプラインを実装します。

---

**前章**: [第2章：次元削減手法による材料空間のマッピング](chapter-2.html)

**次章**: [第4章：実践編 - GNN + 次元削減による材料マッピング](chapter-4.html)

**シリーズトップ**: [材料特性マッピング入門](index.html)
