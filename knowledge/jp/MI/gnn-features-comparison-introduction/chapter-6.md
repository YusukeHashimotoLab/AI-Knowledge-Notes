---
title: "Chapter 6: PyTorch Geometricワークフロー"
chapter_title: "Chapter 6: PyTorch Geometricワークフロー"
---

🌐 JP | [🇬🇧 EN](<../../../en/MI/gnn-features-comparison-introduction/chapter-6.html>) | Last sync: 2025-11-16

# Chapter 6: PyTorch Geometricワークフロー

本章では、PyTorch GeometricとMaterials Project APIを用いた実践的なワークフローを学びます。カスタムデータセットの作成、分散学習、GPU最適化、そして本番環境へのデプロイメントまで、実際のプロジェクトで必要となる技術を包括的に習得します。

## 6.1 Materials Project APIによるデータ取得

Materials Projectは材料科学における最大級のオープンデータベースであり、148,000以上の結晶構造と物性データを提供しています。`pymatgen`ライブラリと`mp-api`を用いて、これらのデータを効率的に取得・処理する方法を学びます。

### 6.1.1 Materials Project API認証

Materials Project APIを使用するには、無料のAPIキーが必要です。[Materials Projectウェブサイト](<https://materialsproject.org/>)でアカウントを作成し、APIキーを取得してください。
    
    
    # コード例1: Materials Project API認証と基本的なデータ取得
    # Google Colabで実行可能
    
    # 必要なライブラリのインストール
    !pip install mp-api pymatgen -q
    
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    import pandas as pd
    
    # APIキーを設定（ご自身のAPIキーに置き換えてください）
    API_KEY = "your_api_key_here"
    
    # MPResterを初期化
    with MPRester(API_KEY) as mpr:
        # ペロブスカイト構造（ABX3）の材料を検索
        # 形成エネルギーが負（安定）で、バンドギャップが1-3 eVの材料
        docs = mpr.materials.summary.search(
            formula="*3",  # ABX3形式
            num_elements=(3, 3),  # 3元素系
            energy_above_hull=(0, 0.01),  # ほぼ安定相
            band_gap=(1.0, 3.0),  # 半導体領域
            fields=["material_id", "formula_pretty", "band_gap",
                    "energy_above_hull", "formation_energy_per_atom"]
        )
    
    # 結果をDataFrameに変換
    data = []
    for doc in docs:
        data.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap,
            "e_hull": doc.energy_above_hull,
            "formation_energy": doc.formation_energy_per_atom
        })
    
    df = pd.DataFrame(data)
    print(f"検索結果: {len(df)}件の材料")
    print(df.head())
    
    # 統計情報
    print("\n=== 統計情報 ===")
    print(f"バンドギャップ範囲: {df['band_gap'].min():.3f} - {df['band_gap'].max():.3f} eV")
    print(f"形成エネルギー範囲: {df['formation_energy'].min():.3f} - {df['formation_energy'].max():.3f} eV/atom")
    

**実行結果の例:**  
検索結果: 247件の材料  
バンドギャップ範囲: 1.012 - 2.987 eV  
形成エネルギー範囲: -2.345 - -0.128 eV/atom 

### 6.1.2 結晶構造データの取得とCIF形式保存

Materials Projectから取得した結晶構造は、pymatgenの`Structure`オブジェクトとして扱われます。これをCIF（Crystallographic Information File）形式で保存し、可視化や機械学習モデルへの入力として活用できます。
    
    
    # コード例2: 結晶構造の取得とCIF形式での保存
    # Google Colabで実行可能
    
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter
    import os
    
    API_KEY = "your_api_key_here"
    
    # 保存ディレクトリの作成
    os.makedirs("structures", exist_ok=True)
    
    with MPRester(API_KEY) as mpr:
        # 例: mp-1234（仮のMaterial ID）の結晶構造を取得
        # 実際のMaterial IDに置き換えてください
        structure = mpr.get_structure_by_material_id("mp-1234")
    
        # 構造情報の表示
        print("=== 結晶構造情報 ===")
        print(f"化学式: {structure.composition.reduced_formula}")
        print(f"空間群: {structure.get_space_group_info()}")
        print(f"格子定数: {structure.lattice.abc}")
        print(f"格子角度: {structure.lattice.angles}")
        print(f"原子数: {len(structure)}")
        print(f"体積: {structure.volume:.3f} Å³")
    
        # 原子サイト情報
        print("\n=== 原子サイト ===")
        for i, site in enumerate(structure):
            print(f"Site {i+1}: {site.species_string} at {site.frac_coords}")
    
        # CIF形式で保存
        cif_writer = CifWriter(structure)
        cif_writer.write_file(f"structures/mp-1234.cif")
        print("\nCIFファイルを保存しました: structures/mp-1234.cif")
    
    # 複数の材料をバッチ取得
    material_ids = ["mp-1234", "mp-5678", "mp-9012"]  # 実際のIDに置き換え
    
    with MPRester(API_KEY) as mpr:
        for mat_id in material_ids:
            try:
                structure = mpr.get_structure_by_material_id(mat_id)
                cif_writer = CifWriter(structure)
                cif_writer.write_file(f"structures/{mat_id}.cif")
                print(f"✓ {mat_id}: {structure.composition.reduced_formula}")
            except Exception as e:
                print(f"✗ {mat_id}: エラー - {e}")
    

**API制限について:** Materials Project APIには1日あたりのリクエスト数制限があります。大規模なデータ取得を行う場合は、`time.sleep()`でレート制限を遵守し、バッチ処理を検討してください。 

## 6.2 PyTorch Geometricカスタムデータセット

Materials Projectから取得したデータをPyTorch Geometricの`Data`オブジェクトに変換し、訓練用データセットを作成します。`InMemoryDataset`クラスを継承して、効率的なデータローディングを実現します。

### 6.2.1 Materials ProjectからPyG Dataへの変換
    
    
    # コード例3: Materials Project構造をPyTorch Geometric Dataに変換
    # Google Colabで実行可能（GPU推奨）
    
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    from pymatgen.core import Structure
    from mp_api.client import MPRester
    import numpy as np
    from typing import List, Tuple
    
    class StructureToGraph:
        """
        pymatgen Structureオブジェクトをグラフ表現に変換
        """
        def __init__(self, cutoff: float = 5.0):
            """
            Args:
                cutoff: 原子間距離のカットオフ半径（Å）
            """
            self.cutoff = cutoff
    
        def convert(self, structure: Structure) -> Data:
            """
            Structure → PyG Data変換
    
            Args:
                structure: pymatgen Structureオブジェクト
    
            Returns:
                PyG Dataオブジェクト
            """
            # ノード特徴量: 原子番号のワンホット表現（最大原子番号92: U）
            atom_numbers = [site.specie.Z for site in structure]
            x = torch.zeros((len(atom_numbers), 92))
            for i, z in enumerate(atom_numbers):
                x[i, z-1] = 1.0  # インデックスは0始まり
    
            # エッジ構築: カットオフ半径内の原子ペア
            edge_index = []
            edge_attr = []
    
            for i, site_i in enumerate(structure):
                # 周期境界条件を考慮した近傍探索
                neighbors = structure.get_neighbors(site_i, self.cutoff)
    
                for neighbor in neighbors:
                    j = neighbor.index
                    distance = neighbor.nn_distance
    
                    # エッジ追加（無向グラフなので双方向）
                    edge_index.append([i, j])
    
                    # エッジ特徴量: 距離のガウス展開
                    edge_feature = self._gaussian_expansion(distance)
                    edge_attr.append(edge_feature)
    
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
            # グラフデータ作成
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
    
            return data
    
        def _gaussian_expansion(self, distance: float, num_centers: int = 41) -> np.ndarray:
            """
            距離をガウス基底関数で展開
    
            Args:
                distance: 原子間距離（Å）
                num_centers: ガウス基底の数
    
            Returns:
                展開係数ベクトル
            """
            centers = np.linspace(0, self.cutoff, num_centers)
            width = 0.5  # ガウス幅
    
            gamma = -0.5 / (width ** 2)
            return np.exp(gamma * (distance - centers) ** 2)
    
    # 使用例
    API_KEY = "your_api_key_here"
    converter = StructureToGraph(cutoff=5.0)
    
    with MPRester(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id("mp-1234")
        data = converter.convert(structure)
    
        print("=== グラフ表現 ===")
        print(f"ノード数: {data.x.size(0)}")
        print(f"ノード特徴量次元: {data.x.size(1)}")
        print(f"エッジ数: {data.edge_index.size(1)}")
        print(f"エッジ特徴量次元: {data.edge_attr.size(1)}")
    

### 6.2.2 カスタムInMemoryDataset実装

`InMemoryDataset`を継承して、Materials Projectデータの前処理とキャッシングを自動化します。これにより、再実行時のデータ取得時間を大幅に削減できます。
    
    
    # コード例4: Materials Project用カスタムInMemoryDataset
    # Google Colabで実行可能（GPU推奨）
    
    import os
    import torch
    from torch_geometric.data import InMemoryDataset, Data
    from mp_api.client import MPRester
    import pickle
    
    class MaterialsProjectDataset(InMemoryDataset):
        """
        Materials Projectから材料物性予測用データセットを作成
        """
        def __init__(self, root, api_key, material_ids=None,
                     property_name="band_gap", cutoff=5.0,
                     transform=None, pre_transform=None, pre_filter=None):
            """
            Args:
                root: データセット保存ディレクトリ
                api_key: Materials Project APIキー
                material_ids: 取得するMaterial IDのリスト（Noneの場合は検索）
                property_name: 予測対象物性（'band_gap', 'formation_energy_per_atom'等）
                cutoff: グラフ構築カットオフ半径（Å）
            """
            self.api_key = api_key
            self.material_ids = material_ids
            self.property_name = property_name
            self.converter = StructureToGraph(cutoff=cutoff)
    
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
        @property
        def raw_file_names(self):
            return ['materials.pkl']
    
        @property
        def processed_file_names(self):
            return ['data.pt']
    
        def download(self):
            """
            Materials Project APIからデータをダウンロード
            """
            with MPRester(self.api_key) as mpr:
                if self.material_ids is None:
                    # Material IDが指定されていない場合は検索
                    docs = mpr.materials.summary.search(
                        energy_above_hull=(0, 0.05),  # 準安定相まで含む
                        num_elements=(1, 5),  # 1-5元素系
                        fields=["material_id", self.property_name]
                    )
                    self.material_ids = [doc.material_id for doc in docs
                                         if getattr(doc, self.property_name) is not None]
                    print(f"検索結果: {len(self.material_ids)}件の材料")
    
                # 構造と物性データを取得
                materials_data = []
                for i, mat_id in enumerate(self.material_ids):
                    try:
                        structure = mpr.get_structure_by_material_id(mat_id)
                        doc = mpr.materials.summary.search(
                            material_ids=[mat_id],
                            fields=[self.property_name]
                        )[0]
    
                        property_value = getattr(doc, self.property_name)
    
                        materials_data.append({
                            'material_id': mat_id,
                            'structure': structure,
                            'property': property_value
                        })
    
                        if (i + 1) % 100 == 0:
                            print(f"ダウンロード進捗: {i+1}/{len(self.material_ids)}")
    
                    except Exception as e:
                        print(f"エラー ({mat_id}): {e}")
    
                # 保存
                os.makedirs(self.raw_dir, exist_ok=True)
                with open(self.raw_paths[0], 'wb') as f:
                    pickle.dump(materials_data, f)
    
                print(f"✓ ダウンロード完了: {len(materials_data)}件")
    
        def process(self):
            """
            Raw dataをPyG Data形式に変換
            """
            # Raw dataの読み込み
            with open(self.raw_paths[0], 'rb') as f:
                materials_data = pickle.load(f)
    
            # PyG Data形式に変換
            data_list = []
            for item in materials_data:
                # グラフ変換
                data = self.converter.convert(item['structure'])
    
                # ラベル（物性値）を追加
                data.y = torch.tensor([item['property']], dtype=torch.float)
                data.material_id = item['material_id']
    
                data_list.append(data)
    
            # フィルタリング（オプション）
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
    
            # 前処理（オプション）
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
    
            # 保存
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print(f"✓ 処理完了: {len(data_list)}件")
    
    # 使用例
    API_KEY = "your_api_key_here"
    
    # データセット作成（初回は自動ダウンロード & 処理）
    dataset = MaterialsProjectDataset(
        root='./data/mp_band_gap',
        api_key=API_KEY,
        property_name='band_gap',
        cutoff=5.0
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    print(f"サンプル: {dataset[0]}")
    

**キャッシング機能:** `InMemoryDataset`は処理済みデータを自動保存します。2回目以降の実行では、保存されたデータを読み込むだけで高速に起動します。 

## 6.3 分散学習とGPU最適化

大規模データセットや複雑なGNNモデルを効率的に訓練するため、PyTorchの分散学習機能とGPU最適化テクニックを活用します。

### 6.3.1 DataParallelによるマルチGPU学習
    
    
    # コード例5: DataParallelによるマルチGPU並列学習
    # Google Colab Pro/Pro+（複数GPU環境）で実行可能
    
    import torch
    import torch.nn as nn
    from torch_geometric.nn import CGConv, global_mean_pool
    from torch_geometric.loader import DataLoader
    import time
    
    class CGCNNModel(nn.Module):
        """
        CGCNN（Crystal Graph Convolutional Neural Network）
        """
        def __init__(self, atom_fea_len=92, nbr_fea_len=41,
                     hidden_dim=128, n_conv=3):
            super(CGCNNModel, self).__init__()
    
            # 原子埋め込み
            self.atom_embedding = nn.Linear(atom_fea_len, hidden_dim)
    
            # CGConv層
            self.conv_layers = nn.ModuleList([
                CGConv(hidden_dim, nbr_fea_len) for _ in range(n_conv)
            ])
    
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
            ])
    
            # 予測ヘッド
            self.fc1 = nn.Linear(hidden_dim, 64)
            self.fc2 = nn.Linear(64, 1)
    
            self.activation = nn.Softplus()
    
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # 原子埋め込み
            x = self.atom_embedding(x)
    
            # CGConv層（残差接続付き）
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x_new = conv(x, edge_index, edge_attr)
                x_new = bn(x_new)
                x_new = self.activation(x_new)
                x = x + x_new  # 残差接続
    
            # グラフレベルプーリング
            x = global_mean_pool(x, batch)
    
            # 予測
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
    
            return x.squeeze()
    
    # マルチGPU学習
    def train_multigpu(dataset, epochs=100, batch_size=64, lr=0.001):
        """
        DataParallelによるマルチGPU並列学習
        """
        # データローダー
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
        # モデル構築
        model = CGCNNModel()
    
        # GPUデバイス確認
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_count = torch.cuda.device_count()
        print(f"使用可能GPU数: {gpu_count}")
    
        if gpu_count > 1:
            # マルチGPU並列化
            model = nn.DataParallel(model)
            print(f"DataParallelモード: {gpu_count}個のGPUを使用")
    
        model = model.to(device)
    
        # 最適化設定
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
        # 学習ループ
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            start_time = time.time()
    
            for batch in train_loader:
                batch = batch.to(device)
    
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item() * batch.num_graphs
    
            avg_loss = total_loss / len(dataset)
            epoch_time = time.time() - start_time
    
            # 学習率調整
            scheduler.step(avg_loss)
    
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
        return model
    
    # 実行例
    # dataset = MaterialsProjectDataset(...) で作成したデータセット
    # model = train_multigpu(dataset, epochs=200, batch_size=64)
    

**DataParallelの動作:** バッチをGPU間で分割し、各GPUで並列に順伝播・逆伝播を実行します。勾配はGPU 0に集約され、パラメータ更新が行われます。 

### 6.3.2 Mixed Precision Training（混合精度学習）

PyTorchの`torch.cuda.amp`を用いて、FP16（半精度浮動小数点）とFP32（単精度）を混合して学習します。メモリ使用量を削減し、学習速度を最大2倍高速化できます。
    
    
    # コード例6: Mixed Precision Training（混合精度学習）
    # Google Colab（GPU環境）で実行可能
    
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from torch_geometric.loader import DataLoader
    import time
    
    def train_mixed_precision(model, dataset, epochs=100, batch_size=64, lr=0.001):
        """
        Mixed Precision Trainingによる高速学習
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        # データローダー
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
        # 最適化設定
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        # Gradient Scaler（勾配スケーリング）
        scaler = GradScaler()
    
        print("=== Mixed Precision Training開始 ===")
    
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            start_time = time.time()
    
            for batch in train_loader:
                batch = batch.to(device)
    
                optimizer.zero_grad()
    
                # Mixed Precision: FP16で順伝播
                with autocast():
                    output = model(batch)
                    loss = criterion(output, batch.y)
    
                # スケーリング付き逆伝播
                scaler.scale(loss).backward()
    
                # 勾配クリッピング（オプション）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
                # パラメータ更新
                scaler.step(optimizer)
                scaler.update()
    
                total_loss += loss.item() * batch.num_graphs
    
            avg_loss = total_loss / len(dataset)
            epoch_time = time.time() - start_time
    
            if (epoch + 1) % 10 == 0:
                # メモリ使用量表示
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, "
                          f"Memory={memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                else:
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
    
        return model
    
    # 使用例
    model = CGCNNModel()
    # model = train_mixed_precision(model, dataset, epochs=200)
    

**Mixed Precisionの効果:** V100 GPUで約1.5-2倍の学習速度向上、メモリ使用量を約40%削減可能です。精度への影響はほとんどありません（MAE差0.001以下）。 

## 6.4 モデルの保存とロード

訓練済みモデルを保存し、後で推論に使用するためのベストプラクティスを学びます。

### 6.4.1 チェックポイント保存
    
    
    # コード例7: モデルチェックポイントの保存とロード
    # Google Colabで実行可能
    
    import torch
    import os
    from datetime import datetime
    
    class ModelCheckpoint:
        """
        モデルチェックポイント管理
        """
        def __init__(self, save_dir='checkpoints', monitor='val_loss', mode='min'):
            """
            Args:
                save_dir: 保存ディレクトリ
                monitor: 監視する指標（'val_loss', 'val_mae'等）
                mode: 'min'（最小化）または'max'（最大化）
            """
            self.save_dir = save_dir
            self.monitor = monitor
            self.mode = mode
            self.best_score = float('inf') if mode == 'min' else float('-inf')
    
            os.makedirs(save_dir, exist_ok=True)
    
        def save(self, model, optimizer, epoch, metrics, filename=None):
            """
            チェックポイントを保存
    
            Args:
                model: PyTorchモデル
                optimizer: オプティマイザー
                epoch: 現在のエポック
                metrics: メトリクス辞書（例: {'val_loss': 0.025, 'val_mae': 0.18}）
                filename: 保存ファイル名（Noneの場合は自動生成）
            """
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"checkpoint_epoch{epoch}_{timestamp}.pt"
    
            filepath = os.path.join(self.save_dir, filename)
    
            # チェックポイントデータ
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
    
            torch.save(checkpoint, filepath)
            print(f"✓ Checkpoint saved: {filepath}")
    
            # ベストモデルの場合は別名でも保存
            current_score = metrics.get(self.monitor)
            if current_score is not None:
                is_best = (self.mode == 'min' and current_score < self.best_score) or \
                          (self.mode == 'max' and current_score > self.best_score)
    
                if is_best:
                    self.best_score = current_score
                    best_path = os.path.join(self.save_dir, 'best_model.pt')
                    torch.save(checkpoint, best_path)
                    print(f"✓ Best model updated: {self.monitor}={current_score:.4f}")
    
        @staticmethod
        def load(filepath, model, optimizer=None):
            """
            チェックポイントをロード
    
            Args:
                filepath: チェックポイントファイルパス
                model: ロード先モデル
                optimizer: ロード先オプティマイザー（オプション）
    
            Returns:
                epoch, metrics
            """
            checkpoint = torch.load(filepath)
    
            model.load_state_dict(checkpoint['model_state_dict'])
    
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', {})
    
            print(f"✓ Checkpoint loaded: {filepath}")
            print(f"  Epoch: {epoch}, Metrics: {metrics}")
    
            return epoch, metrics
    
    # 使用例: 学習ループ内でのチェックポイント保存
    checkpoint_manager = ModelCheckpoint(save_dir='./checkpoints', monitor='val_mae', mode='min')
    
    for epoch in range(100):
        # 学習処理
        train_loss = 0.0  # 実際の学習で計算
    
        # 検証処理
        val_loss = 0.0  # 実際の検証で計算
        val_mae = 0.0
    
        # 10エポックごとにチェックポイント保存
        if (epoch + 1) % 10 == 0:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae
            }
            checkpoint_manager.save(model, optimizer, epoch + 1, metrics)
    
    # ベストモデルのロード
    model_new = CGCNNModel()
    checkpoint_manager.load('./checkpoints/best_model.pt', model_new)
    

### 6.4.2 ONNX形式エクスポート（推論最適化）

ONNX（Open Neural Network Exchange）形式にエクスポートすることで、推論速度を最大化し、異なるフレームワーク（TensorFlow、C++等）でモデルを使用できます。
    
    
    # コード例8: ONNXエクスポートと推論
    # Google Colabで実行可能
    
    import torch
    import torch.onnx
    from torch_geometric.data import Batch
    
    def export_to_onnx(model, sample_data, onnx_path='model.onnx'):
        """
        PyTorch GeometricモデルをONNX形式にエクスポート
    
        Args:
            model: PyTorchモデル
            sample_data: サンプル入力データ（Data型）
            onnx_path: 保存パス
        """
        model.eval()
    
        # サンプルデータをバッチ形式に変換
        batch = Batch.from_data_list([sample_data])
    
        # ダミー入力作成（ONNXエクスポートに必要）
        dummy_input = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )
    
        # ONNXエクスポート
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['x', 'edge_index', 'edge_attr', 'batch'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'edge_attr': {0: 'num_edges'},
                'batch': {0: 'num_nodes'}
            }
        )
    
        print(f"✓ ONNX export completed: {onnx_path}")
    
        # ONNXモデルの検証
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    
    # ONNX Runtime推論（高速推論）
    def inference_onnx(onnx_path, data):
        """
        ONNX Runtimeを用いた高速推論
    
        Args:
            onnx_path: ONNXモデルパス
            data: 入力Data
    
        Returns:
            予測値
        """
        import onnxruntime as ort
        import numpy as np
    
        # ONNX Runtimeセッション作成
        ort_session = ort.InferenceSession(onnx_path)
    
        # バッチ化
        batch = Batch.from_data_list([data])
    
        # NumPy配列に変換
        ort_inputs = {
            'x': batch.x.numpy(),
            'edge_index': batch.edge_index.numpy(),
            'edge_attr': batch.edge_attr.numpy(),
            'batch': batch.batch.numpy()
        }
    
        # 推論
        ort_outputs = ort_session.run(None, ort_inputs)
        prediction = ort_outputs[0]
    
        return prediction[0]
    
    # 使用例
    # model = CGCNNModel()  # 訓練済みモデル
    # sample_data = dataset[0]  # サンプルデータ
    
    # export_to_onnx(model, sample_data, 'cgcnn_model.onnx')
    # prediction = inference_onnx('cgcnn_model.onnx', sample_data)
    # print(f"ONNX予測: {prediction:.4f}")
    

**ONNX Runtimeの利点:** PyTorchネイティブ推論と比較して、1.5-3倍の推論速度向上が期待できます。特にCPU環境での効果が顕著です。 

## 6.5 本番環境デプロイメント

訓練済みモデルをREST APIとして公開し、Webアプリケーションや他のシステムから利用可能にします。FastAPIを用いた実装例を示します。
    
    
    ```mermaid
    graph LR
        A[Webクライアント] -->|POST /predict| B[FastAPI Server]
        B --> C[モデルロード]
        C --> D[PyG Data変換]
        D --> E[推論実行]
        E --> F[結果JSON]
        F -->|Response| A
    
        style B fill:#667eea,color:#fff
        style E fill:#764ba2,color:#fff
    ```
    
    
    # コード例9: FastAPI REST APIデプロイメント
    # ローカル環境またはサーバーで実行
    
    # requirements.txt:
    # fastapi
    # uvicorn
    # torch
    # torch-geometric
    # pymatgen
    
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    from pymatgen.core import Structure
    import json
    
    app = FastAPI(title="Materials Property Prediction API")
    
    # グローバル変数としてモデルをロード
    MODEL = None
    DEVICE = None
    
    class CrystalInput(BaseModel):
        """
        入力データスキーマ
        """
        structure: dict  # pymatgen Structure辞書表現
        # または
        cif_string: str = None  # CIF文字列
    
    class PredictionResponse(BaseModel):
        """
        予測結果スキーマ
        """
        prediction: float
        uncertainty: float = None
        material_id: str = None
    
    @app.on_event("startup")
    async def load_model():
        """
        サーバー起動時にモデルをロード
        """
        global MODEL, DEVICE
    
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # モデルロード
        MODEL = CGCNNModel()
        checkpoint = torch.load('checkpoints/best_model.pt', map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
    
        print(f"✓ Model loaded on {DEVICE}")
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_property(input_data: CrystalInput):
        """
        材料物性予測エンドポイント
    
        Args:
            input_data: 結晶構造データ
    
        Returns:
            予測結果
        """
        try:
            # 構造データのパース
            if input_data.cif_string:
                structure = Structure.from_str(input_data.cif_string, fmt='cif')
            else:
                structure = Structure.from_dict(input_data.structure)
    
            # グラフ変換
            converter = StructureToGraph(cutoff=5.0)
            data = converter.convert(structure)
            data = data.to(DEVICE)
    
            # 推論
            with torch.no_grad():
                prediction = MODEL(data).item()
    
            return PredictionResponse(
                prediction=prediction,
                material_id=input_data.structure.get('material_id', 'unknown')
            )
    
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """
        ヘルスチェックエンドポイント
        """
        return {
            "status": "healthy",
            "model_loaded": MODEL is not None,
            "device": str(DEVICE)
        }
    
    @app.get("/")
    async def root():
        """
        APIルート
        """
        return {
            "message": "Materials Property Prediction API",
            "endpoints": {
                "POST /predict": "Predict material property from structure",
                "GET /health": "Health check",
                "GET /docs": "API documentation (Swagger UI)"
            }
        }
    
    # サーバー起動コマンド:
    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    
    # クライアント使用例（Python）:
    """
    import requests
    import json
    
    # CIF文字列（例）
    cif_string = '''
    data_mp-1234
    _cell_length_a    3.905
    _cell_length_b    3.905
    _cell_length_c    3.905
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _symmetry_space_group_name_H-M 'P 1'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ti1 Ti 0.0 0.0 0.0
    O1  O  0.5 0.5 0.0
    O2  O  0.5 0.0 0.5
    O3  O  0.0 0.5 0.5
    '''
    
    # API呼び出し
    response = requests.post(
        'http://localhost:8000/predict',
        json={'cif_string': cif_string}
    )
    
    result = response.json()
    print(f"予測バンドギャップ: {result['prediction']:.3f} eV")
    """
    

**FastAPIの利点:** 自動API文書生成（Swagger UI）、高速な非同期処理、型チェック、簡潔なコード記述が可能です。`http://localhost:8000/docs`でインタラクティブなAPI文書を確認できます。 

## 6.6 本章のまとめ

本章では、PyTorch GeometricとMaterials Project APIを用いた実践的なワークフローを学びました。カスタムデータセットの作成から本番デプロイメントまで、実際のプロジェクトで必要となる技術を包括的に習得しました。

### 重要なポイント

  * **Materials Project API** : 148,000以上の結晶構造と物性データへのアクセス、pymatgenによる構造解析
  * **カスタムデータセット** : `InMemoryDataset`継承による効率的なデータローディング、キャッシング機能
  * **分散学習** : `DataParallel`によるマルチGPU学習、Mixed Precision Training（1.5-2倍高速化、40%メモリ削減）
  * **モデル保存** : チェックポイント管理、ONNX形式エクスポート（1.5-3倍推論高速化）
  * **本番デプロイ** : FastAPIによるREST APIサーバー、Swagger UI自動生成

### 実践的なワークフロー
    
    
    ```mermaid
    graph TD
        A[Materials Project API] --> B[データ取得 & CIF保存]
        B --> C[PyG Dataset作成]
        C --> D[分散学習 / GPU最適化]
        D --> E[チェックポイント保存]
        E --> F[ONNX変換]
        F --> G[FastAPI デプロイ]
        G --> H[REST API公開]
    
        style A fill:#667eea,color:#fff
        style D fill:#764ba2,color:#fff
        style H fill:#28a745,color:#fff
    ```

### 次のステップ

本シリーズで学んだ知識を活用して、以下のような実践プロジェクトに挑戦してみましょう：

  1. **カスタム物性予測モデル** : Materials Projectの別の物性（形成エネルギー、弾性率等）を予測するモデルを構築
  2. **アンサンブルモデル** : 組成ベースモデルとGNNモデルのアンサンブルによる高精度予測
  3. **能動学習パイプライン** : 不確実性推定を用いた効率的なデータ収集戦略
  4. **Webアプリケーション** : Streamlit等を用いたインタラクティブな材料探索ツール

## 演習問題

#### 演習 6.1（Easy）: Materials Project APIによる基本的なデータ取得

Materials Project APIを用いて、形成エネルギーが-2.0 eV/atom以下の安定な酸化物材料（O含有）を100件取得し、以下の統計情報を表示するコードを作成してください。

  * 平均形成エネルギー
  * 含まれる元素の種類と頻度
  * 空間群の分布

**解答例:**
    
    
    from mp_api.client import MPRester
    import pandas as pd
    from collections import Counter
    
    API_KEY = "your_api_key_here"
    
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["O"],  # 酸素含有
            formation_energy_per_atom=(None, -2.0),  # -2.0 eV/atom以下
            num_elements=(2, 5),  # 2-5元素系
            fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                    "elements", "symmetry"]
        )
    
    # 統計情報計算
    formation_energies = [doc.formation_energy_per_atom for doc in docs]
    all_elements = []
    space_groups = []
    
    for doc in docs:
        all_elements.extend([str(el) for el in doc.elements])
        space_groups.append(doc.symmetry.symbol)
    
    print(f"=== 統計情報（{len(docs)}件） ===")
    print(f"平均形成エネルギー: {sum(formation_energies)/len(formation_energies):.3f} eV/atom")
    print(f"\n元素頻度（上位10）:")
    for elem, count in Counter(all_elements).most_common(10):
        print(f"  {elem}: {count}回")
    print(f"\n空間群分布（上位5）:")
    for sg, count in Counter(space_groups).most_common(5):
        print(f"  {sg}: {count}件")

#### 演習 6.2（Easy）: CIF形式でのデータ保存とロード

Materials Projectから任意の材料（material_id指定）の結晶構造を取得し、CIF形式で保存後、再度ロードして原子サイト情報を表示するコードを作成してください。

**解答例:**
    
    
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter, CifParser
    
    API_KEY = "your_api_key_here"
    material_id = "mp-1234"  # 実際のIDに置き換え
    
    # 1. データ取得とCIF保存
    with MPRester(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(material_id)
    
        cif_writer = CifWriter(structure)
        cif_writer.write_file(f"{material_id}.cif")
        print(f"✓ CIF保存: {material_id}.cif")
    
    # 2. CIFロード
    parser = CifParser(f"{material_id}.cif")
    structure_loaded = parser.get_structures()[0]
    
    # 3. 原子サイト情報表示
    print(f"\n=== 原子サイト情報 ===")
    print(f"化学式: {structure_loaded.composition.reduced_formula}")
    for i, site in enumerate(structure_loaded):
        print(f"Site {i+1}: {site.species_string} at fractional coords {site.frac_coords}")

#### 演習 6.3（Medium）: カスタムInMemoryDatasetの拡張

コード例4の`MaterialsProjectDataset`を拡張し、以下の機能を追加してください：

  1. 複数の物性（band_gap, formation_energy_per_atom）を同時に取得
  2. `__len__()`と`__getitem__()`メソッドの明示的実装
  3. データセットの統計情報を返す`statistics()`メソッドの追加

**解答例:**
    
    
    class MultiPropertyDataset(InMemoryDataset):
        def __init__(self, root, api_key, property_names=['band_gap', 'formation_energy_per_atom'],
                     cutoff=5.0, transform=None, pre_transform=None, pre_filter=None):
            self.api_key = api_key
            self.property_names = property_names
            self.converter = StructureToGraph(cutoff=cutoff)
    
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
        @property
        def raw_file_names(self):
            return ['materials.pkl']
    
        @property
        def processed_file_names(self):
            return ['data.pt']
    
        def download(self):
            with MPRester(self.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    energy_above_hull=(0, 0.05),
                    fields=["material_id"] + self.property_names
                )
    
                materials_data = []
                for doc in docs:
                    try:
                        structure = mpr.get_structure_by_material_id(doc.material_id)
                        properties = {prop: getattr(doc, prop) for prop in self.property_names}
    
                        materials_data.append({
                            'material_id': doc.material_id,
                            'structure': structure,
                            'properties': properties
                        })
                    except:
                        pass
    
                with open(self.raw_paths[0], 'wb') as f:
                    pickle.dump(materials_data, f)
    
        def process(self):
            with open(self.raw_paths[0], 'rb') as f:
                materials_data = pickle.load(f)
    
            data_list = []
            for item in materials_data:
                data = self.converter.convert(item['structure'])
    
                # 複数物性をテンソル化
                y = torch.tensor([item['properties'][prop] for prop in self.property_names],
                                 dtype=torch.float)
                data.y = y
                data.material_id = item['material_id']
    
                data_list.append(data)
    
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
    
        def __len__(self):
            return len(self.slices['x']) - 1
    
        def __getitem__(self, idx):
            data = self.get(idx)
            return data
    
        def statistics(self):
            """データセットの統計情報を返す"""
            stats = {
                'num_samples': len(self),
                'properties': {}
            }
    
            for i, prop in enumerate(self.property_names):
                values = [self[j].y[i].item() for j in range(len(self))]
                stats['properties'][prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
            return stats

#### 演習 6.4（Medium）: Mixed Precision Trainingの効果検証

通常のFP32学習とMixed Precision Training（FP16）を比較し、以下を検証するコードを作成してください：

  * 学習時間の差
  * GPU メモリ使用量の差
  * 最終的なMAEの差（精度への影響）

**解答例:**
    
    
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    
    def compare_training_precision(model, dataset, epochs=50):
        device = torch.device('cuda')
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
        results = {}
    
        # 1. FP32学習
        print("=== FP32学習 ===")
        model_fp32 = model.to(device)
        optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
    
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
    
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model_fp32(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
    
        fp32_time = time.time() - start_time
        fp32_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
        results['fp32'] = {'time': fp32_time, 'memory': fp32_memory}
    
        # 2. Mixed Precision学習
        print("\n=== Mixed Precision学習 ===")
        model_fp16 = model.to(device)
        optimizer = torch.optim.Adam(model_fp16.parameters(), lr=0.001)
        scaler = GradScaler()
    
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
    
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
    
                with autocast():
                    output = model_fp16(batch)
                    loss = criterion(output, batch.y)
    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
        fp16_time = time.time() - start_time
        fp16_memory = torch.cuda.max_memory_allocated() / 1024**3
    
        results['fp16'] = {'time': fp16_time, 'memory': fp16_memory}
    
        # 結果表示
        print("\n=== 比較結果 ===")
        print(f"学習時間: FP32={fp32_time:.2f}s, FP16={fp16_time:.2f}s (高速化率: {fp32_time/fp16_time:.2f}x)")
        print(f"GPU メモリ: FP32={fp32_memory:.2f}GB, FP16={fp16_memory:.2f}GB (削減率: {(1-fp16_memory/fp32_memory)*100:.1f}%)")
    
        return results

#### 演習 6.5（Medium）: チェックポイントからの学習再開

学習を途中で中断し、保存されたチェックポイントから学習を再開するコードを作成してください。エポック番号、loss、optimizer状態を正しく復元してください。

**解答例:**
    
    
    import torch
    
    def train_with_resume(model, dataset, total_epochs=100, checkpoint_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
        start_epoch = 0
    
        # チェックポイントから再開
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"✓ チェックポイントから再開: Epoch {start_epoch}")
    
        # 学習ループ
        for epoch in range(start_epoch, total_epochs):
            model.train()
            total_loss = 0
    
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.4f}")
    
            # 10エポックごとにチェックポイント保存
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(checkpoint, f'checkpoint_epoch{epoch+1}.pt')
                print(f"✓ Checkpoint saved")
    
        return model
    
    # 使用例
    # model = train_with_resume(CGCNNModel(), dataset, total_epochs=100, checkpoint_path='checkpoint_epoch50.pt')

#### 演習 6.6（Hard）: バッチ予測とONNX推論速度ベンチマーク

PyTorchネイティブ推論とONNX Runtime推論の速度を比較するベンチマークコードを作成してください。以下の条件で測定してください：

  * バッチサイズ: 1, 32, 64, 128
  * 各バッチサイズで100回推論を実行
  * 平均推論時間（ms）とスループット（samples/sec）を計算

**解答例:**
    
    
    import torch
    import onnxruntime as ort
    import time
    import numpy as np
    from torch_geometric.data import DataLoader, Batch
    
    def benchmark_inference(model, dataset, onnx_path, batch_sizes=[1, 32, 64, 128], n_iterations=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
    
        # ONNX Runtime セッション
        ort_session = ort.InferenceSession(onnx_path)
    
        results = []
    
        for batch_size in batch_sizes:
            print(f"\n=== Batch Size: {batch_size} ===")
    
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
            # PyTorch推論
            torch_times = []
            for _ in range(n_iterations):
                batch = next(iter(loader))
                batch = batch.to(device)
    
                start = time.time()
                with torch.no_grad():
                    _ = model(batch)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch_times.append((time.time() - start) * 1000)  # ms
    
            torch_avg = np.mean(torch_times)
            torch_throughput = batch_size * 1000 / torch_avg
    
            # ONNX Runtime推論
            onnx_times = []
            for _ in range(n_iterations):
                batch = next(iter(loader))
    
                ort_inputs = {
                    'x': batch.x.numpy(),
                    'edge_index': batch.edge_index.numpy(),
                    'edge_attr': batch.edge_attr.numpy(),
                    'batch': batch.batch.numpy()
                }
    
                start = time.time()
                _ = ort_session.run(None, ort_inputs)
                onnx_times.append((time.time() - start) * 1000)
    
            onnx_avg = np.mean(onnx_times)
            onnx_throughput = batch_size * 1000 / onnx_avg
    
            # 結果保存
            results.append({
                'batch_size': batch_size,
                'pytorch_ms': torch_avg,
                'onnx_ms': onnx_avg,
                'speedup': torch_avg / onnx_avg,
                'pytorch_throughput': torch_throughput,
                'onnx_throughput': onnx_throughput
            })
    
            print(f"PyTorch: {torch_avg:.2f} ms/batch ({torch_throughput:.1f} samples/sec)")
            print(f"ONNX: {onnx_avg:.2f} ms/batch ({onnx_throughput:.1f} samples/sec)")
            print(f"高速化率: {torch_avg/onnx_avg:.2f}x")
    
        return results
    
    # 使用例
    # results = benchmark_inference(model, dataset, 'cgcnn_model.onnx')

#### 演習 6.7（Hard）: FastAPI非同期バッチ推論

FastAPIのバックグラウンドタスク機能を用いて、複数の予測リクエストをバッチ処理する非同期APIを実装してください。以下の要件を満たしてください：

  * リクエストを一定時間（例: 1秒）バッファリング
  * バッファされたリクエストをバッチとして一括推論
  * 各リクエストに対してユニークなジョブIDを発行
  * `/result/{job_id}`エンドポイントで結果を取得

**解答例:**
    
    
    from fastapi import FastAPI, BackgroundTasks
    from pydantic import BaseModel
    import asyncio
    import uuid
    from collections import defaultdict
    import torch
    
    app = FastAPI()
    
    # グローバル状態
    pending_requests = []
    results_store = {}
    MODEL = None
    
    class PredictionRequest(BaseModel):
        structure: dict
    
    class JobResponse(BaseModel):
        job_id: str
        status: str
    
    class ResultResponse(BaseModel):
        job_id: str
        prediction: float = None
        status: str
    
    async def batch_processor():
        """バックグラウンドでバッチ処理を実行"""
        while True:
            await asyncio.sleep(1.0)  # 1秒ごとにバッチ処理
    
            if len(pending_requests) == 0:
                continue
    
            # バッファされたリクエストを取得
            batch_requests = pending_requests.copy()
            pending_requests.clear()
    
            # バッチ推論
            job_ids = [req['job_id'] for req in batch_requests]
            structures = [req['structure'] for req in batch_requests]
    
            # グラフ変換（並列処理）
            data_list = []
            for structure in structures:
                converter = StructureToGraph(cutoff=5.0)
                data = converter.convert(Structure.from_dict(structure))
                data_list.append(data)
    
            # バッチ化
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(data_list)
            batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
    
            # 推論
            with torch.no_grad():
                predictions = MODEL(batch).cpu().numpy()
    
            # 結果保存
            for job_id, pred in zip(job_ids, predictions):
                results_store[job_id] = {
                    'status': 'completed',
                    'prediction': float(pred)
                }
    
    @app.on_event("startup")
    async def startup_event():
        global MODEL
        MODEL = CGCNNModel()
        checkpoint = torch.load('checkpoints/best_model.pt')
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
    
        # バックグラウンドタスク開始
        asyncio.create_task(batch_processor())
    
    @app.post("/predict/async", response_model=JobResponse)
    async def predict_async(request: PredictionRequest):
        """非同期予測リクエスト"""
        job_id = str(uuid.uuid4())
    
        # リクエストをバッファに追加
        pending_requests.append({
            'job_id': job_id,
            'structure': request.structure
        })
    
        # 結果ストアに初期状態を保存
        results_store[job_id] = {'status': 'pending'}
    
        return JobResponse(job_id=job_id, status='pending')
    
    @app.get("/result/{job_id}", response_model=ResultResponse)
    async def get_result(job_id: str):
        """結果取得"""
        if job_id not in results_store:
            return ResultResponse(job_id=job_id, status='not_found')
    
        result = results_store[job_id]
    
        return ResultResponse(
            job_id=job_id,
            prediction=result.get('prediction'),
            status=result['status']
        )

#### 演習 6.8（Hard）: 不確実性推定付き予測API

Monte Carlo Dropout（MCドロップアウト）を用いて、予測値の不確実性を推定するAPIを実装してください。以下を含めてください：

  * ドロップアウト層を持つモデル定義
  * 推論時にドロップアウトを有効化し、複数回（例: 30回）推論
  * 予測値の平均と標準偏差を返す

**解答例:**
    
    
    import torch
    import torch.nn as nn
    from torch_geometric.nn import CGConv, global_mean_pool
    import numpy as np
    
    class CGCNNWithDropout(nn.Module):
        """ドロップアウト層を持つCGCNN"""
        def __init__(self, atom_fea_len=92, nbr_fea_len=41,
                     hidden_dim=128, n_conv=3, dropout=0.1):
            super().__init__()
    
            self.atom_embedding = nn.Linear(atom_fea_len, hidden_dim)
    
            self.conv_layers = nn.ModuleList([
                CGConv(hidden_dim, nbr_fea_len) for _ in range(n_conv)
            ])
    
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
            ])
    
            self.dropout = nn.Dropout(p=dropout)
    
            self.fc1 = nn.Linear(hidden_dim, 64)
            self.fc2 = nn.Linear(64, 1)
            self.activation = nn.Softplus()
    
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            x = self.atom_embedding(x)
    
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x_new = conv(x, edge_index, edge_attr)
                x_new = bn(x_new)
                x_new = self.activation(x_new)
                x_new = self.dropout(x_new)  # ドロップアウト
                x = x + x_new
    
            x = global_mean_pool(x, batch)
    
            x = self.dropout(self.activation(self.fc1(x)))  # ドロップアウト
            x = self.fc2(x)
    
            return x.squeeze()
    
        def predict_with_uncertainty(self, data, n_samples=30):
            """
            MC Dropoutによる不確実性推定
    
            Returns:
                mean, std (予測値の平均と標準偏差)
            """
            self.train()  # ドロップアウトを有効化
    
            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.forward(data).item()
                    predictions.append(pred)
    
            mean = np.mean(predictions)
            std = np.std(predictions)
    
            return mean, std
    
    # FastAPI統合
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    MODEL = None
    
    class UncertaintyResponse(BaseModel):
        prediction: float
        uncertainty: float
        confidence_interval_95: tuple
    
    @app.post("/predict/uncertainty", response_model=UncertaintyResponse)
    async def predict_with_uncertainty(request: CrystalInput):
        """不確実性推定付き予測"""
        # 構造データのパース
        structure = Structure.from_dict(request.structure)
    
        # グラフ変換
        converter = StructureToGraph(cutoff=5.0)
        data = converter.convert(structure)
        data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
    
        # MC Dropout推論
        mean, std = MODEL.predict_with_uncertainty(data, n_samples=30)
    
        # 95%信頼区間
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
    
        return UncertaintyResponse(
            prediction=mean,
            uncertainty=std,
            confidence_interval_95=(ci_lower, ci_upper)
        )

## 参考文献

  1. Jain, A., Ong, S. P., Hautier, G., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. _APL Materials_ , 1(1), 011002. DOI: 10.1063/1.4812323, pp. 1-11. (Materials Project APIの基礎文献)
  2. Ong, S. P., Richards, W. D., Jain, A., et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. _Computational Materials Science_ , 68, 314-319. DOI: 10.1016/j.commatsci.2012.10.028, pp. 314-319. (pymatgenライブラリの公式論文)
  3. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. In _ICLR Workshop on Representation Learning on Graphs and Manifolds_. arXiv:1903.02428, pp. 1-5. (PyTorch Geometricの公式論文)
  4. Micikevicius, P., Narang, S., Alben, J., et al. (2018). Mixed Precision Training. In _International Conference on Learning Representations (ICLR)_. arXiv:1710.03740, pp. 1-12. (Mixed Precision Trainingの提案論文)
  5. Bingham, E., Chen, J. P., Jankowiak, M., et al. (2019). Pyro: Deep Universal Probabilistic Programming. _Journal of Machine Learning Research_ , 20(28), 1-6. (不確実性推定の理論的背景)
  6. Ramírez, S. (2021). _FastAPI: Modern Python Web Development_. O'Reilly Media, pp. 1-350. (FastAPIの包括的なガイド、特にChapter 5-7が本番デプロイメントに有用)
  7. ONNX Runtime Development Team. (2020). ONNX Runtime Performance Tuning. Microsoft Technical Report. https://onnxruntime.ai/docs/performance/ (ONNX Runtime最適化の公式ドキュメント)

[← シリーズトップに戻る](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
