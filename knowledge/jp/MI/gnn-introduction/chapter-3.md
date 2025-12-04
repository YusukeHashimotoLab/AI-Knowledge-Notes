---
title: 第3章：PyTorch Geometric実践 - 分子・材料特性予測の実装
chapter_title: 第3章：PyTorch Geometric実践 - 分子・材料特性予測の実装
subtitle: 実データで学ぶグラフニューラルネットワークの構築と評価
reading_time: 20-25分
difficulty: 中級
code_examples: 0
exercises: 0
---

# 第3章：PyTorch Geometric実践 - 分子・材料特性予測の実装

PyGでデータ準備→モデル学習→評価までを最短ルートで体験します。学習が安定しない時のチェックリストも示します。

**💡 補足:** まずは小グラフ・小エポックで動作確認→バッチ・層数を増やす順番で拡張。学習率は小さめから。

## 学習目標

この章を読むことで、以下を習得できます： \- PyTorch Geometric環境を構築し、GNNライブラリを使いこなせる \- QM9データセットで分子特性予測モデルを実装できる \- Materials Projectデータで結晶特性予測を実行できる \- モデル訓練のベストプラクティスを適用できる \- 予測結果を可視化し、性能を評価できる

**読了時間** : 25-30分 **コード例** : 10個 **演習問題** : 3問

* * *

## 3.1 環境構築：PyTorch Geometricのインストール

### 3.1.1 PyTorch Geometricとは

**PyTorch Geometric (PyG)** は、PyTorch上で動作するグラフニューラルネットワーク専用ライブラリです。

**主な特徴** : \- 🚀 **高速** : GPUによる効率的なグラフ処理 \- 📦 **豊富なモデル** : GCN、GAT、GraphSAGE、SchNetなど30種類以上 \- 🧪 **データセット** : QM9、ZINC、OGB（Open Graph Benchmark）が組み込み済み \- 🛠️ **柔軟性** : カスタムレイヤーやモデルを簡単に実装可能

### 3.1.2 インストール手順

**Option 1: Conda環境（推奨）**
    
    
    # 1. Python 3.9以上の環境を作成
    conda create -n gnn-env python=3.10
    conda activate gnn-env
    
    # 2. PyTorchをインストール（CUDA版推奨）
    # CPU版の場合:
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    
    # GPU版の場合（CUDA 11.8）:
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # 3. PyTorch Geometricをインストール
    conda install pyg -c pyg
    
    # 4. 追加ライブラリ
    pip install rdkit matplotlib seaborn pandas scikit-learn
    

**Option 2: pipでのインストール**
    
    
    # 1. 仮想環境を作成
    python -m venv gnn-env
    source gnn-env/bin/activate  # macOS/Linux
    # gnn-env\Scripts\activate  # Windows
    
    # 2. PyTorchをインストール
    pip install torch torchvision torchaudio
    
    # 3. PyTorch Geometricをインストール
    pip install torch-geometric
    
    # 4. 依存ライブラリ
    pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    
    # 5. 追加ライブラリ
    pip install rdkit matplotlib seaborn pandas scikit-learn
    

**Option 3: Google Colab（インストール不要）**
    
    
    # Google Colabでは以下を実行
    !pip install torch-geometric
    !pip install rdkit
    

### 3.1.3 インストール確認
    
    
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    from rdkit import Chem
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    print("===== インストール確認 =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 簡単なグラフを作成してテスト
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    print(f"\nテストグラフ作成成功!")
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.num_edges}")
    print("✅ PyTorch Geometric環境の構築完了!")
    

**期待される出力** :
    
    
    ===== インストール確認 =====
    PyTorch version: 2.0.0
    PyTorch Geometric version: 2.3.0
    CUDA available: True
    CUDA version: 11.8
    GPU: NVIDIA GeForce RTX 3090
    
    テストグラフ作成成功!
    ノード数: 3
    エッジ数: 4
    ✅ PyTorch Geometric環境の構築完了!
    

### 3.1.4 トラブルシューティング

エラー | 原因 | 解決方法  
---|---|---  
`ImportError: No module named 'torch_geometric'` | PyG未インストール | `pip install torch-geometric`  
`OSError: [WinError 126] DLL読み込みエラー` (Windows) | C++再頒布可能パッケージ不足 | Microsoft Visual C++ Redistributableをインストール  
`RuntimeError: CUDA out of memory` | GPU メモリ不足 | バッチサイズを削減、CPU版PyTorch使用  
`ImportError: cannot import name 'Data'` | バージョン不一致 | PyTorchとPyGのバージョンを確認  
  
* * *

## 3.2 PyTorch Geometricの基本：データ構造とDataLoader

### 3.2.1 Dataオブジェクトの構造

PyTorch Geometricでは、グラフを`Data`オブジェクトで表現します。
    
    
    from torch_geometric.data import Data
    import torch
    
    # エタノール分子 (C2H5OH) をグラフで表現
    # C: 炭素（ノード0, 1）
    # O: 酸素（ノード2）
    # H: 水素（ノード3-7）
    
    # ノード特徴量（原子番号を使用）
    x = torch.tensor([
        [6],   # C (炭素)
        [6],   # C (炭素)
        [8],   # O (酸素)
        [1],   # H (水素)
        [1],   # H (水素)
        [1],   # H (水素)
        [1],   # H (水素)
        [1],   # H (水素)
    ], dtype=torch.float)
    
    # エッジインデックス（結合関係）
    # 各結合は双方向（無向グラフ）
    edge_index = torch.tensor([
        [0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 1, 4, 4, 1, 1, 5, 5, 1, 2, 6, 6, 2],
        [1, 0, 2, 2, 3, 0, 0, 3, 4, 1, 1, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 2, 2, 7]
    ], dtype=torch.long)
    
    # エッジ特徴量（結合タイプ: 1=単結合）
    edge_attr = torch.ones(edge_index.size(1), 1)
    
    # 分子レベルの特徴（目的変数）
    y = torch.tensor([[156.0]], dtype=torch.float)  # 沸点 (°C)
    
    # Dataオブジェクトを作成
    ethanol = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print("===== エタノール分子のグラフ表現 =====")
    print(f"ノード数（原子数）: {ethanol.num_nodes}")
    print(f"エッジ数（結合数×2）: {ethanol.num_edges}")
    print(f"ノード特徴量の形状: {ethanol.x.shape}")
    print(f"エッジインデックスの形状: {ethanol.edge_index.shape}")
    print(f"目的変数（沸点）: {ethanol.y.item()} °C")
    
    # グラフの基本統計
    print(f"\n===== グラフの統計情報 =====")
    print(f"平均次数（結合数）: {ethanol.num_edges / ethanol.num_nodes:.2f}")
    print(f"孤立ノード: {ethanol.contains_isolated_nodes()}")
    print(f"自己ループ: {ethanol.contains_self_loops()}")
    

**出力** :
    
    
    ===== エタノール分子のグラフ表現 =====
    ノード数（原子数）: 8
    エッジ数（結合数×2）: 24
    ノード特徴量の形状: torch.Size([8, 1])
    エッジインデックスの形状: torch.Size([2, 24])
    目的変数（沸点）: 156.0 °C
    
    ===== グラフの統計情報 =====
    平均次数（結合数）: 3.00
    孤立ノード: False
    自己ループ: False
    

### 3.2.2 RDKitからグラフへの変換

RDKitはSMILES（分子の文字列表現）から分子オブジェクトを作成できます。
    
    
    from rdkit import Chem
    from rdkit.Chem import Draw
    from torch_geometric.data import Data
    import torch
    
    def mol_to_graph(smiles):
        """
        SMILES文字列からPyTorch GeometricのDataオブジェクトを作成
    
        Parameters:
        -----------
        smiles : str
            分子のSMILES表現
    
        Returns:
        --------
        data : torch_geometric.data.Data
            グラフデータ
        """
        # SMILESから分子オブジェクトを作成
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    
        # ノード特徴量（原子の特性）
        atom_features = []
        for atom in mol.GetAtoms():
            # 原子番号をワンホットエンコーディング（C, N, O, F, その他）
            atom_type = [0] * 5
            if atom.GetAtomicNum() == 6:    # C
                atom_type[0] = 1
            elif atom.GetAtomicNum() == 7:  # N
                atom_type[1] = 1
            elif atom.GetAtomicNum() == 8:  # O
                atom_type[2] = 1
            elif atom.GetAtomicNum() == 9:  # F
                atom_type[3] = 1
            else:
                atom_type[4] = 1
    
            # 形式電荷と芳香族性を追加
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
    
            atom_features.append(atom_type + [formal_charge, is_aromatic])
    
        x = torch.tensor(atom_features, dtype=torch.float)
    
        # エッジインデックス（結合関係）
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]  # 無向グラフなので双方向
    
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
        data = Data(x=x, edge_index=edge_index)
        return data, mol
    
    # テスト: いくつかの分子をグラフに変換
    smiles_list = [
        ("C", "メタン"),
        ("CCO", "エタノール"),
        ("c1ccccc1", "ベンゼン"),
        ("CC(=O)O", "酢酸"),
    ]
    
    print("===== SMILESからグラフへの変換 =====")
    for smiles, name in smiles_list:
        data, mol = mol_to_graph(smiles)
        print(f"\n{name} ({smiles}):")
        print(f"  ノード数: {data.num_nodes}")
        print(f"  エッジ数: {data.num_edges}")
        print(f"  ノード特徴量次元: {data.x.shape[1]}")
    
    # 分子構造の可視化
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (smiles, name) in enumerate(smiles_list):
        _, mol = mol_to_graph(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        axes[i].imshow(img)
        axes[i].set_title(f"{name}\n{smiles}", fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### 3.2.3 DataLoaderの使用

複数のグラフをバッチ処理するには`DataLoader`を使用します。
    
    
    from torch_geometric.data import Data, DataLoader
    import torch
    
    # サンプルデータセットを作成（10個の分子）
    dataset = []
    for i in range(10):
        num_nodes = torch.randint(5, 15, (1,)).item()  # 5-14原子
        x = torch.randn(num_nodes, 7)  # ノード特徴量（7次元）
    
        # ランダムなエッジを生成
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
        # 目的変数（例: HOMO-LUMOギャップ）
        y = torch.randn(1)
    
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    # DataLoaderを作成（バッチサイズ=4）
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("===== DataLoaderの使用 =====")
    print(f"データセットサイズ: {len(dataset)}")
    print(f"バッチ数: {len(loader)}")
    
    # 最初のバッチを確認
    for batch in loader:
        print(f"\n最初のバッチ:")
        print(f"  バッチ内の分子数: {batch.num_graphs}")
        print(f"  総ノード数: {batch.num_nodes}")
        print(f"  総エッジ数: {batch.num_edges}")
        print(f"  ノード特徴量の形状: {batch.x.shape}")
        print(f"  バッチインデックス: {batch.batch}")
        print(f"  目的変数の形状: {batch.y.shape}")
        break
    

**出力例** :
    
    
    ===== DataLoaderの使用 =====
    データセットサイズ: 10
    バッチ数: 3
    
    最初のバッチ:
      バッチ内の分子数: 4
      総ノード数: 38
      総エッジ数: 76
      ノード特徴量の形状: torch.Size([38, 7])
      バッチインデックス: tensor([0, 0, 0, ..., 3, 3, 3])
      目的変数の形状: torch.Size([4, 1])
    

**重要** : `batch`テンソルは各ノードがどの分子に属するかを示します（0, 0, 0, 1, 1, 2, 2, 2, 3, ...）。

* * *

## 3.3 QM9データセットで分子特性予測

### 3.3.1 QM9データセットの概要

**QM9** は134,000個の有機小分子の量子化学計算データセットです。

**含まれる特性** : \- HOMO（最高被占軌道エネルギー） \- LUMO（最低非占軌道エネルギー） \- バンドギャップ（HOMO-LUMOギャップ） \- 双極子モーメント \- 内部エネルギー \- エンタルピー、自由エネルギー、熱容量など

### 3.3.2 QM9データセットの読み込み
    
    
    from torch_geometric.datasets import QM9
    import torch
    
    # データセットをダウンロード（初回のみ、約1GB）
    dataset = QM9(root='./data/QM9')
    
    print("===== QM9データセット =====")
    print(f"分子数: {len(dataset)}")
    print(f"ノード特徴量次元: {dataset.num_node_features}")
    print(f"エッジ特徴量次元: {dataset.num_edge_features}")
    print(f"目的変数数: {dataset.num_classes}")
    
    # 最初の分子を確認
    data = dataset[0]
    print(f"\n最初の分子:")
    print(f"  原子数: {data.num_nodes}")
    print(f"  結合数: {data.num_edges // 2}")
    print(f"  ノード特徴量: {data.x.shape}")
    print(f"  エッジ特徴量: {data.edge_attr.shape}")
    print(f"  目的変数（19種類）: {data.y.shape}")
    
    # 目的変数の一部を表示
    target_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                    'U0', 'U', 'H', 'G', 'Cv']
    print(f"\n主要な特性値:")
    for i, name in enumerate(target_names):
        print(f"  {name}: {data.y[0, i].item():.4f}")
    

### 3.3.3 Graph Convolutional Network（GCN）の実装
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    
    class GCN_QM9(torch.nn.Module):
        """
        QM9分子特性予測用のGraph Convolutional Network
    
        Architecture:
        - 3層のGCNConv
        - Global mean pooling
        - 2層の全結合層
        """
        def __init__(self, num_node_features, num_classes, hidden_channels=64):
            super(GCN_QM9, self).__init__()
    
            # GCN層
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
    
            # 全結合層
            self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
            self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)
    
        def forward(self, x, edge_index, batch):
            """
            Parameters:
            -----------
            x : torch.Tensor (num_nodes, num_node_features)
                ノード特徴量
            edge_index : torch.Tensor (2, num_edges)
                エッジインデックス
            batch : torch.Tensor (num_nodes,)
                バッチインデックス
    
            Returns:
            --------
            out : torch.Tensor (batch_size, num_classes)
                予測値
            """
            # GCN層1（畳み込み + 活性化 + ドロップアウト）
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
    
            # GCN層2
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
    
            # GCN層3
            x = self.conv3(x, edge_index)
            x = F.relu(x)
    
            # グローバルプーリング（ノード特徴量を分子レベルに集約）
            x = global_mean_pool(x, batch)
    
            # 全結合層
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
    
            x = self.lin2(x)
            return x
    
    # モデルのインスタンス化
    model = GCN_QM9(
        num_node_features=dataset.num_node_features,
        num_classes=1,  # HOMO-LUMOギャップのみを予測
        hidden_channels=64
    )
    
    print("===== GCNモデル =====")
    print(model)
    print(f"\nパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    

### 3.3.4 モデルの訓練
    
    
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    import time
    
    # データセットを小さくする（高速化のため、実際には全データを使用）
    dataset = dataset[:10000]
    
    # HOMO-LUMOギャップ（index=4）のみを目的変数に設定
    for data in dataset:
        data.y = data.y[:, 4:5]  # shape: (1, 1)
    
    # データ分割（80% train, 10% val, 10% test）
    train_dataset = dataset[:8000]
    val_dataset = dataset[8000:9000]
    test_dataset = dataset[9000:]
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # デバイス設定（GPU利用可能ならGPU、そうでなければCPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 損失関数と最適化アルゴリズム
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 訓練関数
    def train(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
    
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
    
            # 順伝播
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
    
            # 逆伝播
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * data.num_graphs
    
        return total_loss / len(loader.dataset)
    
    # 検証関数
    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
    
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
    
        return total_loss / len(loader.dataset)
    
    # 訓練ループ
    epochs = 50
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("===== 訓練開始 =====")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_qm9.pt')
    
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"\n訓練完了! 所要時間: {training_time:.2f}秒")
    
    # 最良モデルをロード
    model.load_state_dict(torch.load('best_model_qm9.pt'))
    
    # テストデータで評価
    test_loss = evaluate(model, test_loader, criterion, device)
    test_mae = test_loss ** 0.5  # RMSEをMAEの近似として使用
    
    print(f"\n===== テスト性能 =====")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE (approx): {test_mae:.4f} eV")
    

### 3.3.5 学習曲線の可視化
    
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('GCN学習曲線（QM9 HOMO-LUMOギャップ予測）', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 予測 vs 実測のプロット
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_targets, all_preds, alpha=0.6, s=10)
    ax.plot([all_targets.min(), all_targets.max()],
            [all_targets.min(), all_targets.max()],
            'r--', lw=2, label='完全な予測')
    ax.set_xlabel('実測値 (eV)', fontsize=12)
    ax.set_ylabel('予測値 (eV)', fontsize=12)
    ax.set_title('HOMO-LUMOギャップ予測結果', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # R²スコアを計算
    from sklearn.metrics import r2_score
    r2 = r2_score(all_targets, all_preds)
    mae = np.mean(np.abs(all_targets - all_preds))
    
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f} eV',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"===== 最終性能 =====")
    print(f"R² score: {r2:.4f}")
    print(f"MAE: {mae:.4f} eV")
    

* * *

## 3.4 Materials Projectデータで結晶特性予測

### 3.4.1 結晶構造のグラフ表現

結晶は周期的な構造を持つため、分子とは異なる扱いが必要です。
    
    
    from pymatgen.core import Structure
    from pymatgen.ext.matproj import MPRester
    import torch
    from torch_geometric.data import Data
    
    def structure_to_graph(structure, cutoff=5.0):
        """
        pymatgen Structureオブジェクトをグラフに変換
    
        Parameters:
        -----------
        structure : pymatgen.core.Structure
            結晶構造
        cutoff : float
            エッジを作成する距離のカットオフ（Å）
    
        Returns:
        --------
        data : torch_geometric.data.Data
            グラフデータ
        """
        # ノード特徴量（原子番号）
        atomic_numbers = [site.specie.Z for site in structure]
        x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)
    
        # エッジインデックスとエッジ特徴量（原子間距離）
        edge_indices = []
        edge_attrs = []
    
        for i, site_i in enumerate(structure):
            for j, site_j in enumerate(structure):
                if i != j:
                    distance = structure.get_distance(i, j)
                    if distance < cutoff:
                        edge_indices.append([i, j])
                        edge_attrs.append([distance])
    
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    
    # Materials ProjectからLi化合物を取得（サンプル）
    # 注意: 実際にはAPIキーが必要
    # API_KEY = "your_api_key_here"
    # with MPRester(API_KEY) as mpr:
    #     entries = mpr.query(
    #         criteria={"elements": {"$all": ["Li"]}, "nelements": 2},
    #         properties=["structure", "band_gap"]
    #     )
    
    # サンプルデータ（LiCl結晶）
    from pymatgen.core import Lattice, Structure
    
    # LiCl 岩塩型構造
    lattice = Lattice.cubic(5.14)  # 格子定数
    species = ["Li", "Li", "Li", "Li", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]
    ]
    structure = Structure(lattice, species, coords)
    
    # グラフに変換
    data = structure_to_graph(structure, cutoff=4.0)
    
    print("===== LiCl結晶のグラフ表現 =====")
    print(f"ノード数（原子数）: {data.num_nodes}")
    print(f"エッジ数（距離 < 4.0Åの原子ペア）: {data.num_edges}")
    print(f"ノード特徴量: {data.x}")
    print(f"\nエッジ特徴量（距離）の統計:")
    print(f"  最小距離: {data.edge_attr.min().item():.2f} Å")
    print(f"  最大距離: {data.edge_attr.max().item():.2f} Å")
    print(f"  平均距離: {data.edge_attr.mean().item():.2f} Å")
    

### 3.4.2 結晶特性予測モデル（Crystal Graph Convolutional Network）
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_add_pool
    
    class CGCN(torch.nn.Module):
        """
        Crystal Graph Convolutional Network
        結晶のバンドギャップを予測
        """
        def __init__(self, num_node_features=1, hidden_channels=64):
            super(CGCN, self).__init__()
    
            # ノード埋め込み層
            self.embedding = torch.nn.Linear(num_node_features, hidden_channels)
    
            # GCN層（エッジ特徴量を考慮する場合はSchNetなどを使用）
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
    
            # 全結合層
            self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
            self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)
    
        def forward(self, x, edge_index, edge_attr, batch):
            # ノード埋め込み
            x = self.embedding(x)
            x = F.relu(x)
    
            # GCN層
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
    
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
    
            x = self.conv3(x, edge_index)
            x = F.relu(x)
    
            # グローバルプーリング（結晶レベルに集約）
            x = global_add_pool(x, batch)
    
            # 全結合層
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
    
            return x
    
    # モデルのインスタンス化
    model_crystal = CGCN(num_node_features=1, hidden_channels=128)
    
    print("===== Crystal Graph Convolutional Network =====")
    print(model_crystal)
    print(f"\nパラメータ数: {sum(p.numel() for p in model_crystal.parameters()):,}")
    

### 3.4.3 模擬データでの訓練デモ
    
    
    # 模擬データセット作成（実際はMaterials Projectデータを使用）
    crystal_dataset = []
    
    for i in range(200):
        num_atoms = torch.randint(4, 12, (1,)).item()
        x = torch.randint(1, 20, (num_atoms, 1)).float()  # 原子番号
    
        # ランダムなエッジ（距離でフィルタリングしたと仮定）
        edge_index = torch.randint(0, num_atoms, (2, num_atoms * 4))
        edge_attr = torch.rand(num_atoms * 4, 1) * 5.0  # 距離 (0-5Å)
    
        # バンドギャップ（原子番号の関数として模擬）
        y = (x.mean() / 10.0 + torch.randn(1) * 0.5).clamp(0, 10)
    
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        crystal_dataset.append(data)
    
    # データ分割
    train_crystals = crystal_dataset[:160]
    test_crystals = crystal_dataset[160:]
    
    train_loader_crystal = DataLoader(train_crystals, batch_size=16, shuffle=True)
    test_loader_crystal = DataLoader(test_crystals, batch_size=16, shuffle=False)
    
    # 訓練（簡略版）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_crystal = model_crystal.to(device)
    optimizer = torch.optim.Adam(model_crystal.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("===== 結晶バンドギャップ予測訓練 =====")
    for epoch in range(1, 51):
        model_crystal.train()
        total_loss = 0
    
        for data in train_loader_crystal:
            data = data.to(device)
            optimizer.zero_grad()
            out = model_crystal(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
    
        if epoch % 10 == 0:
            train_loss = total_loss / len(train_crystals)
            print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}")
    
    # テスト評価
    model_crystal.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for data in test_loader_crystal:
            data = data.to(device)
            out = model_crystal(data.x, data.edge_index, data.edge_attr, data.batch)
            test_preds.append(out.cpu().numpy())
            test_targets.append(data.y.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    
    test_mae = np.mean(np.abs(test_targets - test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\n===== テスト性能 =====")
    print(f"MAE: {test_mae:.4f} eV")
    print(f"R²: {test_r2:.4f}")
    

* * *

## 3.5 トレーニングのベストプラクティス

### 3.5.1 学習率スケジューリング
    
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    # 学習率を動的に調整
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,     # 学習率を半分に
        patience=10,    # 10エポック改善なしで調整
        verbose=True
    )
    
    # 訓練ループ内で使用
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
    
        # 検証損失に基づいて学習率を調整
        scheduler.step(val_loss)
    

### 3.5.2 Early Stopping
    
    
    class EarlyStopping:
        """
        Early Stoppingクラス
        検証損失が改善しなくなったら訓練を停止
        """
        def __init__(self, patience=20, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
    
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
    
    # 使用例
    early_stopping = EarlyStopping(patience=20)
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
    
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    

### 3.5.3 データ拡張（グラフの摂動）
    
    
    import torch
    from torch_geometric.utils import dropout_edge
    
    def augment_graph(data, drop_edge_prob=0.1, noise_scale=0.01):
        """
        グラフのデータ拡張
    
        Parameters:
        -----------
        data : Data
            元のグラフ
        drop_edge_prob : float
            エッジをドロップする確率
        noise_scale : float
            ノード特徴量に加えるノイズのスケール
    
        Returns:
        --------
        augmented_data : Data
            拡張されたグラフ
        """
        # エッジのドロップアウト
        edge_index, edge_mask = dropout_edge(data.edge_index, p=drop_edge_prob)
    
        # ノード特徴量にノイズを追加
        noise = torch.randn_like(data.x) * noise_scale
        x = data.x + noise
    
        augmented_data = Data(x=x, edge_index=edge_index, y=data.y)
        return augmented_data
    
    # 使用例
    original = dataset[0]
    augmented = augment_graph(original, drop_edge_prob=0.15)
    
    print(f"元のエッジ数: {original.num_edges}")
    print(f"拡張後のエッジ数: {augmented.num_edges}")
    

* * *

## 3.6 モデル性能の評価と可視化

### 3.6.1 評価指標の計算
    
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    def evaluate_regression(y_true, y_pred):
        """
        回帰モデルの評価指標を計算
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
    
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    
    # 使用例
    metrics = evaluate_regression(test_targets, test_preds)
    
    print("===== 評価指標 =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    

### 3.6.2 残差プロット
    
    
    import matplotlib.pyplot as plt
    
    def plot_residuals(y_true, y_pred):
        """
        残差プロット
        """
        residuals = y_true - y_pred
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # 残差 vs 予測値
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('予測値', fontsize=12)
        axes[0].set_ylabel('残差（実測 - 予測）', fontsize=12)
        axes[0].set_title('残差プロット', fontsize=14)
        axes[0].grid(True, alpha=0.3)
    
        # 残差のヒストグラム
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('残差', fontsize=12)
        axes[1].set_ylabel('頻度', fontsize=12)
        axes[1].set_title('残差分布', fontsize=14)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].grid(True, alpha=0.3, axis='y')
    
        plt.tight_layout()
        plt.show()
    
    # 使用例
    plot_residuals(test_targets, test_preds)
    

### 3.6.3 モデル比較
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 複数モデルの性能を比較
    models_performance = {
        'GCN (3層)': {'MAE': 0.32, 'R²': 0.88, 'Time': 45.2},
        'GAT (2層)': {'MAE': 0.28, 'R²': 0.91, 'Time': 62.8},
        'SchNet': {'MAE': 0.25, 'R²': 0.93, 'Time': 89.5},
        'MPNN': {'MAE': 0.30, 'R²': 0.90, 'Time': 55.1},
    }
    
    df = pd.DataFrame(models_performance).T
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # MAE比較
    df['MAE'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_ylabel('MAE (eV)', fontsize=12)
    axes[0].set_title('平均絶対誤差（低いほど良い）', fontsize=13)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # R²比較
    df['R²'].plot(kind='bar', ax=axes[1], color='forestgreen')
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('決定係数（高いほど良い）', fontsize=13)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0.8, 1.0)
    
    # 訓練時間比較
    df['Time'].plot(kind='bar', ax=axes[2], color='coral')
    axes[2].set_ylabel('訓練時間 (秒)', fontsize=12)
    axes[2].set_title('計算コスト', fontsize=13)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 3.7 トラブルシューティング

### 3.7.1 よくあるエラーと解決策

エラー | 原因 | 解決方法  
---|---|---  
`RuntimeError: CUDA out of memory` | GPU メモリ不足 | バッチサイズ削減、モデルの小型化、CPU使用  
`AssertionError: edge_index not contiguous` | エッジインデックスのメモリ配置 | `edge_index = edge_index.t().contiguous()`  
`ValueError: too many values to unpack` | Dataオブジェクトの属性不足 | `x`, `edge_index`, `batch`が正しく設定されているか確認  
`RuntimeError: Expected all tensors on same device` | テンソルのデバイス不一致 | `data = data.to(device)`を確認  
  
### 3.7.2 デバッグのチェックリスト
    
    
    # データの確認
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.num_edges}")
    print(f"孤立ノード: {data.contains_isolated_nodes()}")
    print(f"自己ループ: {data.contains_self_loops()}")
    
    # テンソルの形状確認
    print(f"x.shape: {data.x.shape}")
    print(f"edge_index.shape: {data.edge_index.shape}")
    print(f"y.shape: {data.y.shape}")
    
    # デバイスの確認
    print(f"x device: {data.x.device}")
    print(f"edge_index device: {data.edge_index.device}")
    
    # エッジインデックスの範囲確認
    print(f"max edge index: {data.edge_index.max().item()}")
    print(f"num_nodes: {data.num_nodes}")
    assert data.edge_index.max().item() < data.num_nodes, "エッジインデックスがノード数を超えています"
    

* * *

## 3.8 本章のまとめ

### 学んだこと

  1. **PyTorch Geometric環境構築** \- Conda、pip、Google Colabの3つの方法 \- バージョン互換性の確認とトラブルシューティング

  2. **データ構造の理解** \- Dataオブジェクトの構造（x, edge_index, batch） \- RDKitからのグラフ変換 \- DataLoaderによるバッチ処理

  3. **QM9データセットでの実践** \- 134,000分子の量子化学データセット \- GCNモデルの実装と訓練 \- HOMO-LUMOギャップ予測（MAE < 0.5 eV目標）

  4. **結晶特性予測** \- Materials Project結晶データのグラフ表現 \- Crystal Graph Convolutional Network \- バンドギャップ予測

  5. **訓練のベストプラクティス** \- 学習率スケジューリング \- Early Stopping \- データ拡張（グラフの摂動）

  6. **評価と可視化** \- MAE、MSE、R²などの指標 \- 残差プロット \- モデル間の性能比較

### 重要なポイント

  * ✅ PyTorch Geometricは材料・分子のGNN実装に最適
  * ✅ QM9は初学者に最適な分子特性予測ベンチマーク
  * ✅ グラフの前処理（孤立ノード、自己ループの確認）が重要
  * ✅ バッチ処理では`batch`テンソルが各ノードの所属を示す
  * ✅ 学習率スケジューリングとEarly Stoppingで過学習を防止

### 次の章へ

第4章では、高度なGNN技術を学びます： \- グラフプーリング（階層的表現） \- エッジ特徴量の活用 \- 3D幾何情報の組込み（SchNet、DimeNet） \- 等変GNN（E(3)-equivariant） \- GNNExplainerによる解釈可能性

**[第4章：高度なGNN技術 →](<./chapter-4.html>)**

* * *

## 演習問題

### 問題1（難易度：easy）

PyTorch GeometricのDataオブジェクトに含まれる主要な属性を3つ挙げ、それぞれの役割を説明してください。

ヒント ノード、エッジ、バッチに関する情報を格納する属性を考えましょう。  解答例 **主要な3つの属性**: 1\. **`x` (ノード特徴量)** \- 形状: `(num_nodes, num_node_features)` \- 役割: 各ノード（原子）の特徴量を格納 \- 例: 原子番号、電気陰性度、形式電荷など 2\. **`edge_index` (エッジインデックス)** \- 形状: `(2, num_edges)` \- 役割: グラフの接続関係（隣接リスト形式） \- 例: `[[0, 1], [1, 2]]` → ノード0とノード1が接続 3\. **`batch` (バッチインデックス)** \- 形状: `(num_nodes,)` \- 役割: 各ノードがどのグラフに属するかを示す \- 例: `[0, 0, 1, 1, 2]` → ノード0,1はグラフ0、ノード2,3はグラフ1 **追加の重要な属性**: \- `edge_attr`: エッジ特徴量（結合タイプ、距離など） \- `y`: 目的変数（分子特性、結晶特性） 

* * *

### 問題2（難易度：medium）

QM9データセットで訓練したGCNモデルのMAEが0.8 eVでした。性能を向上させるための3つの具体的なアプローチを提案してください。

ヒント モデルアーキテクチャ、ハイパーパラメータ、データ前処理の3つの観点から考えましょう。  解答例 **アプローチ1: モデルアーキテクチャの改善** 
    
    
    # GATレイヤーを使用（注意機構で重要な結合を学習）
    from torch_geometric.nn import GATConv
    
    class ImprovedGNN(torch.nn.Module):
        def __init__(self, num_node_features, hidden_channels=128):
            super().__init__()
            # GATレイヤー（ヘッド数=8）
            self.conv1 = GATConv(num_node_features, hidden_channels, heads=8)
            self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8)
            self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=1)
            # 層を増やす（3層 → 4層）
            self.conv4 = GCNConv(hidden_channels, hidden_channels)
    

**期待される改善**: MAE 0.8 eV → 0.5-0.6 eV \--- **アプローチ2: エッジ特徴量の活用** 
    
    
    # エッジ特徴量（結合タイプ）を組み込む
    from torch_geometric.nn import NNConv
    
    class EdgeFeaturesGNN(torch.nn.Module):
        def __init__(self, num_node_features, num_edge_features, hidden_channels=64):
            super().__init__()
            # NNConv: エッジ特徴量を考慮
            nn = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features, hidden_channels * hidden_channels),
                torch.nn.ReLU()
            )
            self.conv1 = NNConv(num_node_features, hidden_channels, nn, aggr='mean')
    

**期待される改善**: MAE 0.8 eV → 0.6-0.7 eV \--- **アプローチ3: データ正規化と拡張** 
    
    
    # 目的変数を標準化
    y_mean = train_dataset.data.y.mean(dim=0)
    y_std = train_dataset.data.y.std(dim=0)
    
    for data in train_dataset:
        data.y = (data.y - y_mean) / y_std
    
    # データ拡張（グラフの摂動）
    def augment_graph(data):
        # エッジのドロップアウト
        edge_index, _ = dropout_edge(data.edge_index, p=0.1)
        # ノイズ追加
        x = data.x + torch.randn_like(data.x) * 0.01
        return Data(x=x, edge_index=edge_index, y=data.y)
    
    # 訓練データを2倍に
    augmented_train = [augment_graph(data) for data in train_dataset]
    train_dataset = train_dataset + augmented_train
    

**期待される改善**: MAE 0.8 eV → 0.7 eV \--- **最適な戦略**: アプローチ1（モデル改善）とアプローチ2（エッジ特徴量）を組み合わせ、MAE 0.4-0.5 eVを目指す。 

* * *

### 問題3（難易度：hard）

以下のコードでエラーが発生しました。原因を特定し、修正してください。
    
    
    # エラーが発生するコード
    model = GCN_QM9(num_node_features=11, num_classes=1)
    device = torch.device('cuda')
    model = model.to(device)
    
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    

**エラーメッセージ** :
    
    
    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    

ヒント モデルとデータのデバイスが一致していません。  解答例 **原因**: モデルは`cuda`デバイスに移動されていますが、`data`オブジェクトは`cpu`のままです。PyTorchでは、すべてのテンソルが同じデバイス上にある必要があります。 **修正コード**: 
    
    
    model = GCN_QM9(num_node_features=11, num_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for data in train_loader:
        # データをGPUに移動（重要！）
        data = data.to(device)
    
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    

**重要なポイント**: 1\. `data = data.to(device)`で、`data`内のすべてのテンソル（`x`, `edge_index`, `batch`, `y`）を一度にGPUに移動 2\. `torch.cuda.is_available()`でGPUが利用可能か確認（CPUのみの環境でもエラーを回避） 3\. 訓練ループの**最初**でデータをデバイスに移動 **デバッグのチェック**: 
    
    
    # デバイスの確認
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data x device: {data.x.device}")
    print(f"Data edge_index device: {data.edge_index.device}")
    

* * *

## 参考文献

  1. Fey, M., & Lenssen, J. E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." _ICLR Workshop on Representation Learning on Graphs and Manifolds_. GitHub: https://github.com/pyg-team/pytorch_geometric _PyTorch Geometric公式論文。ライブラリの設計思想と実装の詳細。_

  2. Ramakrishnan, R., et al. (2014). "Quantum chemistry structures and properties of 134 kilo molecules." _Scientific Data_ , 1, 140022. DOI: [10.1038/sdata.2014.22](<https://doi.org/10.1038/sdata.2014.22>) _QM9データセット公式論文。134,000分子の量子化学計算データ。_

  3. Xie, T., & Grossman, J. C. (2018). "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." _Physical Review Letters_ , 120(14), 145301. DOI: [10.1103/PhysRevLett.120.145301](<https://doi.org/10.1103/PhysRevLett.120.145301>) _Crystal Graph Convolutional Networks（CGCN）の原論文。結晶特性予測への応用。_

  4. Gilmer, J., et al. (2017). "Neural Message Passing for Quantum Chemistry." _ICML 2017_. URL: https://arxiv.org/abs/1704.01212 _Message Passing Neural Networks（MPNN）の理論。QM9での高精度予測を達成。_

  5. PyTorch Geometric Documentation. (2024). "Introduction by Example." URL: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html _PyTorch Geometric公式チュートリアル。基本的な使い方を例示。_

  6. RDKit Documentation. (2024). "Getting Started with the RDKit in Python." URL: https://www.rdkit.org/docs/GettingStartedInPython.html _RDKitの公式ドキュメント。SMILESから分子オブジェクトを作成する方法。_

* * *

**作成日** : 2025-10-17 **バージョン** : 1.0 **テンプレート** : chapter-template-v2.0 **著者** : GNN入門シリーズプロジェクト
