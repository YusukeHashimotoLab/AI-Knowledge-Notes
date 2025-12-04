---
title: 第5章：GNNの応用 (GNN Applications)
chapter_title: 第5章：GNNの応用 (GNN Applications)
subtitle: 分子予測から推薦システムまでの実践的応用
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ノード分類タスクを半教師あり学習で実装できる
  * ✅ グラフ分類による分子特性予測を実装できる
  * ✅ リンク予測による推薦システムを構築できる
  * ✅ 創薬応用における薬剤-標的相互作用予測を理解できる
  * ✅ 知識グラフと交通ネットワークへの応用を理解できる
  * ✅ 実践的なGNNプロジェクトを設計・実装できる
  * ✅ GNNの限界と対処法を理解できる

* * *

## 5.1 ノード分類 (Node Classification)

### 半教師あり学習によるノード分類

**ノード分類** は、グラフ内の各ノードにラベルを割り当てるタスクです。典型的な応用として、ソーシャルネットワークにおけるユーザーの興味分類、論文引用ネットワークにおける研究分野分類などがあります。
    
    
    ```mermaid
    graph LR
        A[入力グラフ] --> B[GNN Layers特徴伝播]
        B --> C[ノード埋め込み]
        C --> D[分類器Linear + Softmax]
        D --> E[ノードラベル予測]
    
        F[訓練ノードラベル付き] --> G[損失計算Cross-Entropy]
        E --> G
        G --> H[バックプロパゲーション]
    
        style A fill:#e3f2fd
        style E fill:#c8e6c9
        style F fill:#fff9c4
        style G fill:#ffccbc
    ```

#### 半教師あり学習の利点

特徴 | 説明 | 利点  
---|---|---  
**少量のラベルデータ** | 一部のノードのみラベル付き | アノテーションコスト削減  
**グラフ構造活用** | 隣接ノードから情報伝播 | 精度向上、汎化性能改善  
**Transductive学習** | テストデータもグラフに含まれる | グラフ全体の構造を利用可能  
**Homophily活用** | 類似ノードは繋がりやすい | ラベル伝播が効果的  
  
### Coraデータセットによるノード分類実装

**Coraデータセット** は、機械学習論文の引用ネットワークです。各ノード（論文）は7つの研究分野のいずれかに分類されます。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.utils import to_networkx
    import matplotlib.pyplot as plt
    import networkx as nx
    from sklearn.manifold import TSNE
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import numpy as np
    
    class NodeClassificationGCN(nn.Module):
        """
        Graph Convolutional Networkによるノード分類モデル
    
        Architecture:
        - 2-layer GCN
        - Dropout for regularization
        - Log-softmax output for multi-class classification
        """
    
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            """
            Args:
                num_features: 入力特徴量次元
                hidden_dim: 隠れ層の次元
                num_classes: クラス数
                dropout: ドロップアウト率
            """
            super(NodeClassificationGCN, self).__init__()
    
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
            self.dropout = dropout
    
        def forward(self, data):
            """
            Args:
                data: PyG Data object (x, edge_index)
    
            Returns:
                log_probs: ノードごとのクラス確率（log-space）
            """
            x, edge_index = data.x, data.edge_index
    
            # 第1層: GCN + ReLU + Dropout
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
            # 第2層: GCN
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
        def get_embeddings(self, data):
            """
            ノード埋め込みを取得（可視化用）
    
            Returns:
                第1層のGCN出力（隠れ表現）
            """
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            return x.detach()
    
    
    class NodeClassificationGAT(nn.Module):
        """
        Graph Attention Networkによるノード分類モデル
    
        Features:
        - Multi-head attention mechanism
        - Learnable attention weights
        - Better handling of heterophilic graphs
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     heads=8, dropout=0.6):
            """
            Args:
                num_features: 入力特徴量次元
                hidden_dim: 隠れ層の次元（各ヘッドあたり）
                num_classes: クラス数
                heads: アテンションヘッド数
                dropout: ドロップアウト率
            """
            super(NodeClassificationGAT, self).__init__()
    
            # 第1層: Multi-head GAT
            self.conv1 = GATConv(
                num_features,
                hidden_dim,
                heads=heads,
                dropout=dropout
            )
    
            # 第2層: Single-head GAT for classification
            self.conv2 = GATConv(
                hidden_dim * heads,  # 第1層の全ヘッド出力を結合
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout
            )
    
            self.dropout = dropout
    
        def forward(self, data):
            """
            Args:
                data: PyG Data object
    
            Returns:
                log_probs: クラス確率
            """
            x, edge_index = data.x, data.edge_index
    
            # 第1層: GAT + ELU + Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
    
            # 第2層: GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
    
    class NodeClassificationTrainer:
        """
        ノード分類モデルの訓練・評価クラス
        """
    
        def __init__(self, model, data, device='cuda'):
            """
            Args:
                model: GNNモデル（GCN or GAT）
                data: PyG Data object
                device: 計算デバイス
            """
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)
            self.data = data.to(self.device)
    
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.01,
                weight_decay=5e-4
            )
    
            self.train_losses = []
            self.val_accs = []
    
        def train_epoch(self):
            """
            1エポックの訓練
    
            Returns:
                loss: 訓練損失
            """
            self.model.train()
            self.optimizer.zero_grad()
    
            # 順伝播
            out = self.model(self.data)
    
            # 訓練ノードのみで損失計算（半教師あり学習）
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
    
            # 逆伝播
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        @torch.no_grad()
        def evaluate(self, mask):
            """
            評価
    
            Args:
                mask: 評価対象のノードマスク（train/val/test）
    
            Returns:
                accuracy: 精度
            """
            self.model.eval()
            out = self.model(self.data)
            pred = out.argmax(dim=1)
    
            correct = (pred[mask] == self.data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
    
            return acc
    
        def train(self, epochs=200, early_stopping_patience=20, verbose=True):
            """
            訓練ループ
    
            Args:
                epochs: エポック数
                early_stopping_patience: Early stoppingの許容エポック数
                verbose: ログ表示
    
            Returns:
                best_val_acc: 最良のValidation精度
            """
            best_val_acc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # 訓練
                loss = self.train_epoch()
                self.train_losses.append(loss)
    
                # 評価
                train_acc = self.evaluate(self.data.train_mask)
                val_acc = self.evaluate(self.data.val_mask)
                self.val_accs.append(val_acc)
    
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # ベストモデルを保存
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
                # ログ出力
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
            # ベストモデルをロード
            self.model.load_state_dict(self.best_model_state)
    
            return best_val_acc
    
        @torch.no_grad()
        def test(self):
            """
            テストセットで最終評価
    
            Returns:
                test_acc: テスト精度
                predictions: 全ノードの予測ラベル
                embeddings: ノード埋め込み
            """
            self.model.eval()
    
            # 予測
            out = self.model(self.data)
            pred = out.argmax(dim=1)
    
            # テスト精度
            test_acc = self.evaluate(self.data.test_mask)
    
            # 埋め込み取得（GCNの場合）
            embeddings = None
            if hasattr(self.model, 'get_embeddings'):
                embeddings = self.model.get_embeddings(self.data).cpu().numpy()
    
            return test_acc, pred.cpu().numpy(), embeddings
    
        def plot_training_history(self):
            """
            訓練履歴をプロット
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # 損失
            axes[0].plot(self.train_losses, label='Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
            # Validation精度
            axes[1].plot(self.val_accs, label='Validation Accuracy', color='orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def visualize_embeddings(self, embeddings, labels, title="Node Embeddings"):
            """
            t-SNEによる埋め込み可視化
    
            Args:
                embeddings: ノード埋め込み (N, D)
                labels: ノードラベル (N,)
                title: プロットタイトル
            """
            # t-SNEで2次元に削減
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
    
            # プロット
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter, label='Class')
            plt.title(title)
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    
    # 使用例
    if __name__ == "__main__":
        # データ読み込み
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    
        print(f"Dataset: {dataset.name}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Training nodes: {data.train_mask.sum().item()}")
        print(f"Validation nodes: {data.val_mask.sum().item()}")
        print(f"Test nodes: {data.test_mask.sum().item()}")
    
        # GCNモデル
        print("\n=== Training GCN ===")
        gcn_model = NodeClassificationGCN(
            num_features=dataset.num_features,
            hidden_dim=16,
            num_classes=dataset.num_classes,
            dropout=0.5
        )
    
        gcn_trainer = NodeClassificationTrainer(gcn_model, data)
        best_val_acc = gcn_trainer.train(epochs=200, early_stopping_patience=20)
    
        # テスト評価
        test_acc, predictions, embeddings = gcn_trainer.test()
        print(f"\nGCN Test Accuracy: {test_acc:.4f}")
    
        # 訓練履歴プロット
        gcn_trainer.plot_training_history()
    
        # 埋め込み可視化
        if embeddings is not None:
            gcn_trainer.visualize_embeddings(
                embeddings,
                data.y.cpu().numpy(),
                title="GCN Node Embeddings (t-SNE)"
            )
    
        # GATモデル
        print("\n=== Training GAT ===")
        gat_model = NodeClassificationGAT(
            num_features=dataset.num_features,
            hidden_dim=8,
            num_classes=dataset.num_classes,
            heads=8,
            dropout=0.6
        )
    
        gat_trainer = NodeClassificationTrainer(gat_model, data)
        best_val_acc = gat_trainer.train(epochs=200, early_stopping_patience=20)
    
        # テスト評価
        test_acc, predictions, _ = gat_trainer.test()
        print(f"\nGAT Test Accuracy: {test_acc:.4f}")
    
        # Confusion matrix
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = predictions[data.test_mask.cpu().numpy()]
    
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (GAT on Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    

> **Transductive vs Inductive学習** :
> 
>   * **Transductive** : テストノードもグラフに含まれる。訓練時に全グラフ構造を利用可能。
>   * **Inductive** : テストノードは未知。新しいグラフへの汎化が必要。GraphSAGEなどが対応。
> 

* * *

## 5.2 グラフ分類 (Graph Classification)

### 分子特性予測への応用

**グラフ分類** は、グラフ全体を分類するタスクです。典型的な応用として、分子の生物活性予測、タンパク質の機能分類、ソーシャルネットワークのコミュニティ分類などがあります。
    
    
    ```mermaid
    graph TB
        A[分子グラフ] --> B[GNN Layers原子特徴伝播]
        B --> C[ノード埋め込み]
        C --> D[Graph Poolingグラフレベル表現]
        D --> E[分類器MLP]
        E --> F[分子特性予測毒性/活性など]
    
        style A fill:#e3f2fd
        style D fill:#fff9c4
        style F fill:#c8e6c9
    ```

#### Graph Poolingの種類

手法 | 説明 | 計算式  
---|---|---  
**Global Mean Pooling** | 全ノード埋め込みの平均 | $\mathbf{h}_G = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{h}_v$  
**Global Max Pooling** | 各次元の最大値 | $\mathbf{h}_G = \max_{v \in \mathcal{V}} \mathbf{h}_v$  
**Global Sum Pooling** | 全ノード埋め込みの和 | $\mathbf{h}_G = \sum_{v \in \mathcal{V}} \mathbf{h}_v$  
**Attention Pooling** | アテンション重み付き和 | $\mathbf{h}_G = \sum_{v \in \mathcal{V}} \alpha_v \mathbf{h}_v$  
  
### MUTAG分子データセットによる実装

**MUTAGデータセット** は、188個の化合物の変異原性（DNA変異を引き起こす性質）を分類するデータセットです。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
    import matplotlib.pyplot as plt
    import numpy as np
    
    class MolecularGCN(nn.Module):
        """
        分子グラフ分類のためのGCNモデル
    
        Architecture:
        - Multiple GCN layers for feature propagation
        - Global pooling for graph-level representation
        - MLP classifier for prediction
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     num_layers=3, dropout=0.5, pooling='mean'):
            """
            Args:
                num_features: ノード特徴量次元
                hidden_dim: 隠れ層次元
                num_classes: クラス数
                num_layers: GCN層の数
                dropout: ドロップアウト率
                pooling: Pooling方式 ('mean', 'max', 'sum')
            """
            super(MolecularGCN, self).__init__()
    
            # GCN層
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(num_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
            # 分類器
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    
            self.dropout = dropout
            self.pooling = pooling
    
        def forward(self, data):
            """
            Args:
                data: PyG Batch object (x, edge_index, batch)
    
            Returns:
                logits: グラフ分類ロジット
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # GCN層で特徴伝播
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
    
            # Graph pooling
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                x = global_max_pool(x, batch)
            elif self.pooling == 'sum':
                x = global_add_pool(x, batch)
    
            # 分類器
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
    
            return x
    
    
    class MolecularGIN(nn.Module):
        """
        Graph Isomorphism Network (GIN) for molecular graphs
    
        GINはグラフの同型性を識別できる表現力を持つ
        WL testと同等の識別能力
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     num_layers=5, dropout=0.5):
            """
            Args:
                num_features: ノード特徴量次元
                hidden_dim: 隠れ層次元
                num_classes: クラス数
                num_layers: GIN層の数
                dropout: ドロップアウト率
            """
            super(MolecularGIN, self).__init__()
    
            # GIN層
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # 初期層
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # 中間層
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # 分類器
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    
            self.dropout = dropout
    
        def forward(self, data):
            """
            Args:
                data: PyG Batch object
    
            Returns:
                logits: グラフ分類ロジット
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # GIN層
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
    
            # Global sum pooling (GINの標準)
            x = global_add_pool(x, batch)
    
            # 分類器
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
    
            return x
    
    
    class GraphClassificationTrainer:
        """
        グラフ分類モデルの訓練・評価クラス
        """
    
        def __init__(self, model, train_loader, val_loader, test_loader,
                     num_classes, device='cuda'):
            """
            Args:
                model: GNNモデル
                train_loader, val_loader, test_loader: DataLoader
                num_classes: クラス数
                device: 計算デバイス
            """
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.num_classes = num_classes
    
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                weight_decay=5e-4
            )
    
            # 損失関数
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
    
            self.train_losses = []
            self.val_accs = []
    
        def train_epoch(self):
            """
            1エポックの訓練
    
            Returns:
                avg_loss: 平均損失
            """
            self.model.train()
            total_loss = 0
    
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
    
                # 順伝播
                out = self.model(data)
    
                # 損失計算
                if self.num_classes == 2:
                    loss = self.criterion(out.squeeze(), data.y.float())
                else:
                    loss = self.criterion(out, data.y)
    
                # 逆伝播
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item() * data.num_graphs
    
            return total_loss / len(self.train_loader.dataset)
    
        @torch.no_grad()
        def evaluate(self, loader):
            """
            評価
    
            Args:
                loader: DataLoader
    
            Returns:
                accuracy: 精度
                all_preds, all_labels: 予測とラベル（ROC AUC計算用）
            """
            self.model.eval()
            correct = 0
            all_preds = []
            all_labels = []
    
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
    
                if self.num_classes == 2:
                    pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
                    all_preds.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
                else:
                    pred = out.argmax(dim=1)
                    all_preds.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    
                correct += (pred == data.y).sum().item()
                all_labels.extend(data.y.cpu().numpy())
    
            acc = correct / len(loader.dataset)
            return acc, np.array(all_preds), np.array(all_labels)
    
        def train(self, epochs=100, early_stopping_patience=10, verbose=True):
            """
            訓練ループ
    
            Args:
                epochs: エポック数
                early_stopping_patience: Early stopping許容エポック数
                verbose: ログ表示
    
            Returns:
                best_val_acc: 最良のValidation精度
            """
            best_val_acc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # 訓練
                loss = self.train_epoch()
                self.train_losses.append(loss)
    
                # 評価
                train_acc, _, _ = self.evaluate(self.train_loader)
                val_acc, _, _ = self.evaluate(self.val_loader)
                self.val_accs.append(val_acc)
    
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
                # ログ
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
            # ベストモデルをロード
            self.model.load_state_dict(self.best_model_state)
            return best_val_acc
    
        def test(self):
            """
            テストセットで最終評価
    
            Returns:
                test_acc: テスト精度
                test_auc: ROC AUC（二値分類の場合）
            """
            test_acc, preds, labels = self.evaluate(self.test_loader)
    
            # ROC AUC（二値分類のみ）
            test_auc = None
            if self.num_classes == 2:
                test_auc = roc_auc_score(labels, preds)
    
            return test_acc, test_auc
    
    
    # 使用例
    if __name__ == "__main__":
        # データセット読み込み
        dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    
        print(f"Dataset: {dataset.name}")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
    
        # データ分割
        indices = list(range(len(dataset)))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=[data.y.item() for data in dataset]
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, random_state=42, stratify=[dataset[i].y.item() for i in train_indices]
        )
    
        # DataLoader作成
        train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=32, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_indices], batch_size=32)
        test_loader = DataLoader([dataset[i] for i in test_indices], batch_size=32)
    
        # GCNモデル
        print("\n=== Training GCN ===")
        gcn_model = MolecularGCN(
            num_features=dataset.num_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            num_layers=3,
            dropout=0.5,
            pooling='mean'
        )
    
        gcn_trainer = GraphClassificationTrainer(
            gcn_model, train_loader, val_loader, test_loader, dataset.num_classes
        )
        gcn_trainer.train(epochs=100, early_stopping_patience=10)
    
        # テスト評価
        test_acc, test_auc = gcn_trainer.test()
        print(f"\nGCN Test Accuracy: {test_acc:.4f}")
        if test_auc:
            print(f"GCN Test ROC AUC: {test_auc:.4f}")
    
        # GINモデル
        print("\n=== Training GIN ===")
        gin_model = MolecularGIN(
            num_features=dataset.num_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            num_layers=5,
            dropout=0.5
        )
    
        gin_trainer = GraphClassificationTrainer(
            gin_model, train_loader, val_loader, test_loader, dataset.num_classes
        )
        gin_trainer.train(epochs=100, early_stopping_patience=10)
    
        # テスト評価
        test_acc, test_auc = gin_trainer.test()
        print(f"\nGIN Test Accuracy: {test_acc:.4f}")
        if test_auc:
            print(f"GIN Test ROC AUC: {test_auc:.4f}")
    
        # 訓練履歴比較
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        axes[0].plot(gcn_trainer.train_losses, label='GCN', alpha=0.7)
        axes[0].plot(gin_trainer.train_losses, label='GIN', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
        axes[1].plot(gcn_trainer.val_accs, label='GCN', alpha=0.7)
        axes[1].plot(gin_trainer.val_accs, label='GIN', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    

* * *

## 5.3 リンク予測 (Link Prediction)

### 推薦システムへの応用

**リンク予測** は、グラフ内の2ノード間にエッジが存在する確率を予測するタスクです。推薦システム（User-Itemグラフ）、知識グラフ補完、ソーシャルネットワーク分析などに応用されます。
    
    
    ```mermaid
    graph TB
        A[グラフ一部エッジを隠す] --> B[GNN Encoderノード埋め込み学習]
        B --> C[ノード埋め込みu, v]
        C --> D[Link Decoderスコア計算]
        D --> E[リンク確率p(u,v)]
    
        F[正例エッジ] --> G[損失計算BCE Loss]
        H[負例エッジサンプリング] --> G
        E --> G
        G --> I[パラメータ更新]
    
        style A fill:#e3f2fd
        style C fill:#fff9c4
        style E fill:#c8e6c9
        style G fill:#ffccbc
    ```

#### Link Decoderの種類

手法 | 計算式 | 特徴  
---|---|---  
**Inner Product** | $s(u,v) = \mathbf{z}_u^\top \mathbf{z}_v$ | シンプル、計算効率良い  
**Cosine Similarity** | $s(u,v) = \frac{\mathbf{z}_u^\top \mathbf{z}_v}{\|\mathbf{z}_u\| \|\mathbf{z}_v\|}$ | 正規化済み、スケール不変  
**MLP Decoder** | $s(u,v) = \text{MLP}([\mathbf{z}_u; \mathbf{z}_v])$ | 高表現力、非線形関係モデル化  
**DistMult** | $s(u,r,v) = \mathbf{z}_u^\top \mathbf{R}_r \mathbf{z}_v$ | 関係タイプ考慮（知識グラフ）  
  
### 推薦システムの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling, train_test_split_edges
    from sklearn.metrics import roc_auc_score, average_precision_score
    import matplotlib.pyplot as plt
    import numpy as np
    
    class LinkPredictionGNN(nn.Module):
        """
        リンク予測のためのGNNエンコーダー
    
        GraphSAGEを使用してノード埋め込みを学習
        """
    
        def __init__(self, num_features, hidden_dim, embedding_dim, num_layers=2, dropout=0.5):
            """
            Args:
                num_features: ノード特徴量次元
                hidden_dim: 隠れ層次元
                embedding_dim: 最終埋め込み次元
                num_layers: GraphSAGE層の数
                dropout: ドロップアウト率
            """
            super(LinkPredictionGNN, self).__init__()
    
            # GraphSAGE層
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(num_features, hidden_dim))
    
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
    
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))
    
            self.dropout = dropout
    
        def encode(self, x, edge_index):
            """
            ノード埋め込みをエンコード
    
            Args:
                x: ノード特徴量
                edge_index: エッジインデックス
    
            Returns:
                z: ノード埋め込み
            """
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
    
            return x
    
        def decode(self, z, edge_index, method='inner_product'):
            """
            エッジスコアをデコード
    
            Args:
                z: ノード埋め込み
                edge_index: エッジインデックス
                method: デコード方式 ('inner_product', 'cosine', 'mlp')
    
            Returns:
                scores: エッジスコア
            """
            src, dst = edge_index
    
            if method == 'inner_product':
                # 内積
                scores = (z[src] * z[dst]).sum(dim=1)
            elif method == 'cosine':
                # コサイン類似度
                scores = F.cosine_similarity(z[src], z[dst])
            else:
                raise ValueError(f"Unknown decode method: {method}")
    
            return scores
    
        def forward(self, x, edge_index, decode_edge_index=None):
            """
            順伝播
    
            Args:
                x: ノード特徴量
                edge_index: メッセージパッシング用エッジ
                decode_edge_index: スコア計算対象エッジ（Noneの場合はedge_index）
    
            Returns:
                scores: エッジスコア
            """
            z = self.encode(x, edge_index)
    
            if decode_edge_index is None:
                decode_edge_index = edge_index
    
            scores = self.decode(z, decode_edge_index)
            return scores
    
    
    class MLPDecoder(nn.Module):
        """
        MLPベースのリンクデコーダー
    
        より複雑な非線形関係をモデル化
        """
    
        def __init__(self, embedding_dim, hidden_dim=128):
            """
            Args:
                embedding_dim: ノード埋め込み次元
                hidden_dim: MLP隠れ層次元
            """
            super(MLPDecoder, self).__init__()
    
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, 1)
            )
    
        def forward(self, z, edge_index):
            """
            Args:
                z: ノード埋め込み
                edge_index: エッジインデックス
    
            Returns:
                scores: エッジスコア
            """
            src, dst = edge_index
    
            # ノード埋め込みを連結
            edge_features = torch.cat([z[src], z[dst]], dim=1)
    
            # MLPでスコア計算
            scores = self.mlp(edge_features).squeeze()
    
            return scores
    
    
    class RecommendationSystem:
        """
        GNNベースの推薦システム
    
        User-Itemグラフでリンク予測を行い、アイテム推薦を実現
        """
    
        def __init__(self, model, decoder=None, device='cuda'):
            """
            Args:
                model: GNNエンコーダー
                decoder: リンクデコーダー（Noneの場合はinner product）
                device: 計算デバイス
            """
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)
            self.decoder = decoder.to(self.device) if decoder else None
    
            # オプティマイザ
            params = list(model.parameters())
            if decoder:
                params += list(decoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)
    
            self.train_losses = []
            self.val_aucs = []
    
        def train_epoch(self, data, train_pos_edge_index):
            """
            1エポックの訓練
    
            Args:
                data: PyG Data object
                train_pos_edge_index: 訓練用正例エッジ
    
            Returns:
                loss: 損失値
            """
            self.model.train()
            if self.decoder:
                self.decoder.train()
    
            self.optimizer.zero_grad()
    
            # ノード埋め込み取得
            z = self.model.encode(data.x, train_pos_edge_index)
    
            # 負例サンプリング
            neg_edge_index = negative_sampling(
                edge_index=train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=train_pos_edge_index.size(1)
            )
    
            # 正例スコア
            if self.decoder:
                pos_scores = self.decoder(z, train_pos_edge_index)
                neg_scores = self.decoder(z, neg_edge_index)
            else:
                pos_scores = self.model.decode(z, train_pos_edge_index)
                neg_scores = self.model.decode(z, neg_edge_index)
    
            # 損失計算（BCE Loss）
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss
    
            # 逆伝播
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        @torch.no_grad()
        def evaluate(self, data, pos_edge_index, neg_edge_index):
            """
            評価
    
            Args:
                data: PyG Data object
                pos_edge_index: 正例エッジ
                neg_edge_index: 負例エッジ
    
            Returns:
                auc: ROC AUC
                ap: Average Precision
            """
            self.model.eval()
            if self.decoder:
                self.decoder.eval()
    
            # 全エッジでエンコード
            z = self.model.encode(data.x, data.train_pos_edge_index)
    
            # スコア計算
            if self.decoder:
                pos_scores = self.decoder(z, pos_edge_index).cpu().numpy()
                neg_scores = self.decoder(z, neg_edge_index).cpu().numpy()
            else:
                pos_scores = self.model.decode(z, pos_edge_index).cpu().numpy()
                neg_scores = self.model.decode(z, neg_edge_index).cpu().numpy()
    
            # ラベルとスコア
            scores = np.concatenate([pos_scores, neg_scores])
            labels = np.concatenate([
                np.ones(pos_scores.shape[0]),
                np.zeros(neg_scores.shape[0])
            ])
    
            # メトリクス計算
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
    
            return auc, ap
    
        def train(self, data, epochs=100, early_stopping_patience=10, verbose=True):
            """
            訓練ループ
    
            Args:
                data: PyG Data object (train_test_split_edges済み)
                epochs: エポック数
                early_stopping_patience: Early stopping許容エポック数
                verbose: ログ表示
    
            Returns:
                best_val_auc: 最良のValidation AUC
            """
            best_val_auc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # 訓練
                loss = self.train_epoch(data, data.train_pos_edge_index)
                self.train_losses.append(loss)
    
                # 評価
                val_auc, val_ap = self.evaluate(
                    data, data.val_pos_edge_index, data.val_neg_edge_index
                )
                self.val_aucs.append(val_auc)
    
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                    if self.decoder:
                        self.best_decoder_state = self.decoder.state_dict()
                else:
                    patience_counter += 1
    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
                # ログ
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
    
            # ベストモデルをロード
            self.model.load_state_dict(self.best_model_state)
            if self.decoder:
                self.decoder.load_state_dict(self.best_decoder_state)
    
            return best_val_auc
    
        def test(self, data):
            """
            テストセットで最終評価
    
            Returns:
                test_auc: ROC AUC
                test_ap: Average Precision
            """
            test_auc, test_ap = self.evaluate(
                data, data.test_pos_edge_index, data.test_neg_edge_index
            )
            return test_auc, test_ap
    
        @torch.no_grad()
        def recommend_items(self, data, user_id, k=10, exclude_known=True):
            """
            ユーザーにアイテムを推薦
    
            Args:
                data: PyG Data object
                user_id: ユーザーノードID
                k: 推薦数
                exclude_known: 既知のアイテムを除外するか
    
            Returns:
                recommended_items: 推薦アイテムIDのリスト
                scores: 対応するスコア
            """
            self.model.eval()
            if self.decoder:
                self.decoder.eval()
    
            # ノード埋め込み取得
            z = self.model.encode(data.x, data.train_pos_edge_index)
    
            # 全アイテムとのスコア計算
            num_nodes = data.num_nodes
            user_embedding = z[user_id].unsqueeze(0).repeat(num_nodes, 1)
    
            if self.decoder:
                edge_index = torch.stack([
                    torch.full((num_nodes,), user_id, dtype=torch.long),
                    torch.arange(num_nodes)
                ]).to(self.device)
                scores = self.decoder(z, edge_index)
            else:
                scores = (user_embedding * z).sum(dim=1)
    
            scores = torch.sigmoid(scores).cpu().numpy()
    
            # 既知アイテムを除外
            if exclude_known:
                known_items = data.train_pos_edge_index[1][
                    data.train_pos_edge_index[0] == user_id
                ].cpu().numpy()
                scores[known_items] = -1
    
            # Top-k選択
            top_k_indices = np.argsort(scores)[-k:][::-1]
            top_k_scores = scores[top_k_indices]
    
            return top_k_indices, top_k_scores
    
    
    # 使用例
    if __name__ == "__main__":
        # データ読み込み（例：Coraデータセット）
        from torch_geometric.datasets import Planetoid
    
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    
        # リンク予測用にエッジ分割
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
    
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Training edges: {data.train_pos_edge_index.size(1)}")
        print(f"Validation edges: {data.val_pos_edge_index.size(1)}")
        print(f"Test edges: {data.test_pos_edge_index.size(1)}")
    
        # モデル初期化（Inner Product Decoder）
        print("\n=== Training with Inner Product Decoder ===")
        model_ip = LinkPredictionGNN(
            num_features=dataset.num_features,
            hidden_dim=128,
            embedding_dim=64,
            num_layers=2,
            dropout=0.5
        )
    
        rec_system_ip = RecommendationSystem(model_ip)
        rec_system_ip.train(data, epochs=100, early_stopping_patience=10)
    
        # テスト評価
        test_auc, test_ap = rec_system_ip.test(data)
        print(f"\nInner Product - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
        # モデル初期化（MLP Decoder）
        print("\n=== Training with MLP Decoder ===")
        model_mlp = LinkPredictionGNN(
            num_features=dataset.num_features,
            hidden_dim=128,
            embedding_dim=64,
            num_layers=2,
            dropout=0.5
        )
    
        mlp_decoder = MLPDecoder(embedding_dim=64, hidden_dim=128)
        rec_system_mlp = RecommendationSystem(model_mlp, mlp_decoder)
        rec_system_mlp.train(data, epochs=100, early_stopping_patience=10)
    
        # テスト評価
        test_auc, test_ap = rec_system_mlp.test(data)
        print(f"\nMLP Decoder - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
        # 推薦例
        user_id = 0
        recommended_items, scores = rec_system_mlp.recommend_items(
            data, user_id, k=10, exclude_known=True
        )
    
        print(f"\nTop-10 Recommendations for User {user_id}:")
        for idx, (item, score) in enumerate(zip(recommended_items, scores), 1):
            print(f"{idx}. Item {item}: Score {score:.4f}")
    
        # 訓練履歴比較
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        axes[0].plot(rec_system_ip.train_losses, label='Inner Product', alpha=0.7)
        axes[0].plot(rec_system_mlp.train_losses, label='MLP Decoder', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
        axes[1].plot(rec_system_ip.val_aucs, label='Inner Product', alpha=0.7)
        axes[1].plot(rec_system_mlp.val_aucs, label='MLP Decoder', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROC AUC')
        axes[1].set_title('Validation AUC Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    

* * *

## 5.4 創薬応用 (Drug Discovery)

### 薬剤-標的相互作用予測

**薬剤-標的相互作用（Drug-Target Interaction, DTI）予測** は、薬剤分子がどのタンパク質標的と相互作用するかを予測するタスクです。創薬において重要な応用であり、候補化合物のスクリーニング時間とコストを大幅に削減できます。
    
    
    ```mermaid
    graph TB
        A[薬剤分子グラフ] --> B[Molecular GNN薬剤埋め込み]
        C[標的タンパク質配列/構造] --> D[Protein Encoder標的埋め込み]
    
        B --> E[薬剤特徴z_drug]
        D --> F[標的特徴z_target]
    
        E --> G[Interaction PredictorMLP/Bilinear]
        F --> G
    
        G --> H[相互作用スコアBinding Affinity]
    
        style A fill:#e3f2fd
        style C fill:#fff9c4
        style H fill:#c8e6c9
    ```

#### DTI予測のアプローチ

コンポーネント | 手法 | 説明  
---|---|---  
**薬剤エンコーダー** | GCN, GIN, AttentiveFP | 分子グラフから特徴抽出  
**標的エンコーダー** | CNN, LSTM, Transformer | アミノ酸配列/3D構造から特徴抽出  
**相互作用予測** | MLP, Bilinear, Attention | 薬剤-標的ペアのスコア計算  
**損失関数** | BCE, MSE, Ranking Loss | 結合親和性または分類ラベル  
  
### DTI予測システムの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    class DrugEncoder(nn.Module):
        """
        薬剤分子グラフのエンコーダー
    
        GCNを使って分子の特徴ベクトルを学習
        """
    
        def __init__(self, num_atom_features, hidden_dim, embedding_dim, num_layers=3):
            """
            Args:
                num_atom_features: 原子特徴量次元
                hidden_dim: 隠れ層次元
                embedding_dim: 最終埋め込み次元
                num_layers: GCN層の数
            """
            super(DrugEncoder, self).__init__()
    
            # GCN層
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(num_atom_features, hidden_dim))
    
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
    
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else embedding_dim)
                for i in range(num_layers)
            ])
    
        def forward(self, data):
            """
            Args:
                data: PyG Batch object（複数分子グラフ）
    
            Returns:
                drug_embeddings: 分子埋め込み (batch_size, embedding_dim)
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # GCN層で特徴伝播
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index)
                x = bn(x)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
    
            # Graph poolingでグラフレベル表現
            drug_embeddings = global_mean_pool(x, batch)
    
            return drug_embeddings
    
    
    class ProteinEncoder(nn.Module):
        """
        タンパク質配列のエンコーダー
    
        1D CNNを使ってアミノ酸配列から特徴抽出
        """
    
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_filters=128, kernel_sizes=[3, 5, 7]):
            """
            Args:
                vocab_size: アミノ酸の種類数（通常20 + パディング）
                embedding_dim: アミノ酸埋め込み次元
                hidden_dim: 最終隠れ層次元
                num_filters: CNNフィルター数
                kernel_sizes: カーネルサイズのリスト
            """
            super(ProteinEncoder, self).__init__()
    
            # アミノ酸埋め込み
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
            # Multi-scale 1D CNN
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
                for k in kernel_sizes
            ])
    
            # 全結合層
            self.fc = nn.Sequential(
                nn.Linear(num_filters * len(kernel_sizes), hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
        def forward(self, protein_sequences):
            """
            Args:
                protein_sequences: アミノ酸配列 (batch_size, seq_len)
    
            Returns:
                protein_embeddings: タンパク質埋め込み (batch_size, hidden_dim)
            """
            # アミノ酸埋め込み
            x = self.embedding(protein_sequences)  # (batch, seq_len, emb_dim)
            x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)
    
            # Multi-scale CNN
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
                # Global max pooling
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_outputs.append(pooled)
    
            # 結合
            x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))
    
            # 全結合層
            protein_embeddings = self.fc(x)
    
            return protein_embeddings
    
    
    class DTIPredictor(nn.Module):
        """
        薬剤-標的相互作用予測モデル
    
        薬剤エンコーダーと標的エンコーダーを組み合わせて
        相互作用スコアを予測
        """
    
        def __init__(self, drug_encoder, protein_encoder, hidden_dim):
            """
            Args:
                drug_encoder: DrugEncoder instance
                protein_encoder: ProteinEncoder instance
                hidden_dim: 相互作用予測用隠れ層次元
            """
            super(DTIPredictor, self).__init__()
    
            self.drug_encoder = drug_encoder
            self.protein_encoder = protein_encoder
    
            # 相互作用予測MLP
            combined_dim = drug_encoder.convs[-1].out_channels + hidden_dim
            self.interaction_mlp = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            )
    
        def forward(self, drug_data, protein_sequences):
            """
            Args:
                drug_data: PyG Batch（薬剤分子グラフ）
                protein_sequences: タンパク質配列 (batch_size, seq_len)
    
            Returns:
                interaction_scores: 相互作用スコア (batch_size,)
            """
            # 薬剤埋め込み
            drug_emb = self.drug_encoder(drug_data)
    
            # 標的埋め込み
            protein_emb = self.protein_encoder(protein_sequences)
    
            # 結合
            combined = torch.cat([drug_emb, protein_emb], dim=1)
    
            # 相互作用スコア
            scores = self.interaction_mlp(combined).squeeze()
    
            return scores
    
    
    class DTIPredictionSystem:
        """
        薬剤-標的相互作用予測システム
        """
    
        def __init__(self, model, device='cuda'):
            """
            Args:
                model: DTIPredictor instance
                device: 計算デバイス
            """
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)
    
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.0001,
                weight_decay=1e-5
            )
    
            self.criterion = nn.BCEWithLogitsLoss()
    
            self.train_losses = []
            self.val_aucs = []
    
        def train_epoch(self, train_loader):
            """
            1エポックの訓練
    
            Args:
                train_loader: DataLoader (drug_data, protein_seq, label)
    
            Returns:
                avg_loss: 平均損失
            """
            self.model.train()
            total_loss = 0
    
            for drug_batch, protein_batch, labels in train_loader:
                # デバイスに転送
                protein_batch = protein_batch.to(self.device)
                labels = labels.to(self.device).float()
    
                self.optimizer.zero_grad()
    
                # 順伝播
                scores = self.model(drug_batch, protein_batch)
    
                # 損失計算
                loss = self.criterion(scores, labels)
    
                # 逆伝播
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
    
            return total_loss / len(train_loader)
    
        @torch.no_grad()
        def evaluate(self, loader):
            """
            評価
    
            Args:
                loader: DataLoader
    
            Returns:
                auc: ROC AUC
                auprc: Area Under Precision-Recall Curve
            """
            self.model.eval()
            all_scores = []
            all_labels = []
    
            for drug_batch, protein_batch, labels in loader:
                protein_batch = protein_batch.to(self.device)
                labels = labels.to(self.device).float()
    
                scores = self.model(drug_batch, protein_batch)
                scores = torch.sigmoid(scores)
    
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
    
            # ROC AUC
            auc_score = roc_auc_score(all_labels, all_scores)
    
            # AUPRC
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            auprc = auc(recall, precision)
    
            return auc_score, auprc
    
        def train(self, train_loader, val_loader, epochs=50,
                  early_stopping_patience=10, verbose=True):
            """
            訓練ループ
    
            Args:
                train_loader, val_loader: DataLoader
                epochs: エポック数
                early_stopping_patience: Early stopping許容エポック数
                verbose: ログ表示
    
            Returns:
                best_val_auc: 最良のValidation AUC
            """
            best_val_auc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # 訓練
                loss = self.train_epoch(train_loader)
                self.train_losses.append(loss)
    
                # 評価
                val_auc, val_auprc = self.evaluate(val_loader)
                self.val_aucs.append(val_auc)
    
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
                # ログ
                if verbose and epoch % 5 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}')
    
            # ベストモデルをロード
            self.model.load_state_dict(self.best_model_state)
            return best_val_auc
    
        def test(self, test_loader):
            """
            テストセットで最終評価
    
            Returns:
                test_auc: ROC AUC
                test_auprc: AUPRC
            """
            return self.evaluate(test_loader)
    
        def predict_interaction(self, drug_graph, protein_sequence):
            """
            単一の薬剤-標的ペアの相互作用を予測
    
            Args:
                drug_graph: PyG Data object（分子グラフ）
                protein_sequence: アミノ酸配列テンソル
    
            Returns:
                interaction_prob: 相互作用確率
            """
            self.model.eval()
    
            with torch.no_grad():
                # バッチ化
                drug_batch = Batch.from_data_list([drug_graph]).to(self.device)
                protein_batch = protein_sequence.unsqueeze(0).to(self.device)
    
                # 予測
                score = self.model(drug_batch, protein_batch)
                prob = torch.sigmoid(score).item()
    
            return prob
    
    
    # 使用例（ダミーデータ）
    if __name__ == "__main__":
        # ダミーデータ生成
        def create_dummy_molecule():
            """ダミー分子グラフ生成"""
            num_atoms = np.random.randint(10, 30)
            x = torch.randn(num_atoms, 9)  # 原子特徴量
            edge_index = torch.randint(0, num_atoms, (2, num_atoms * 2))
            return Data(x=x, edge_index=edge_index)
    
        def create_dummy_protein(max_len=1000):
            """ダミータンパク質配列生成"""
            seq_len = np.random.randint(500, max_len)
            # アミノ酸ID (1-20, 0はパディング)
            seq = torch.randint(1, 21, (seq_len,))
            # パディング
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, dtype=torch.long)
                seq = torch.cat([seq, padding])
            return seq
    
        # ダミーデータセット作成
        num_samples = 1000
        drug_graphs = [create_dummy_molecule() for _ in range(num_samples)]
        protein_seqs = torch.stack([create_dummy_protein() for _ in range(num_samples)])
        labels = torch.randint(0, 2, (num_samples,))  # ランダムラベル
    
        # DataLoader作成（簡略版）
        class DTIDataset(torch.utils.data.Dataset):
            def __init__(self, drug_graphs, protein_seqs, labels):
                self.drug_graphs = drug_graphs
                self.protein_seqs = protein_seqs
                self.labels = labels
    
            def __len__(self):
                return len(self.labels)
    
            def __getitem__(self, idx):
                return self.drug_graphs[idx], self.protein_seqs[idx], self.labels[idx]
    
        def collate_fn(batch):
            drugs, proteins, labels = zip(*batch)
            drug_batch = Batch.from_data_list(drugs)
            protein_batch = torch.stack(proteins)
            label_batch = torch.tensor(labels)
            return drug_batch, protein_batch, label_batch
    
        # データ分割
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
    
        train_dataset = DTIDataset(
            drug_graphs[:train_size],
            protein_seqs[:train_size],
            labels[:train_size]
        )
        val_dataset = DTIDataset(
            drug_graphs[train_size:train_size+val_size],
            protein_seqs[train_size:train_size+val_size],
            labels[train_size:train_size+val_size]
        )
        test_dataset = DTIDataset(
            drug_graphs[train_size+val_size:],
            protein_seqs[train_size+val_size:],
            labels[train_size+val_size:]
        )
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, collate_fn=collate_fn
        )
    
        # モデル初期化
        drug_encoder = DrugEncoder(
            num_atom_features=9,
            hidden_dim=128,
            embedding_dim=128,
            num_layers=3
        )
    
        protein_encoder = ProteinEncoder(
            vocab_size=21,  # 20 amino acids + padding
            embedding_dim=128,
            hidden_dim=128,
            num_filters=128,
            kernel_sizes=[3, 5, 7, 9]
        )
    
        dti_model = DTIPredictor(
            drug_encoder=drug_encoder,
            protein_encoder=protein_encoder,
            hidden_dim=128
        )
    
        # 訓練
        print("=== Training DTI Prediction Model ===")
        dti_system = DTIPredictionSystem(dti_model)
        best_val_auc = dti_system.train(
            train_loader, val_loader, epochs=50, early_stopping_patience=10
        )
    
        # テスト評価
        test_auc, test_auprc = dti_system.test(test_loader)
        print(f"\nTest AUC: {test_auc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")
    
        # 訓練履歴プロット
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        axes[0].plot(dti_system.train_losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
    
        axes[1].plot(dti_system.val_aucs)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROC AUC')
        axes[1].set_title('Validation AUC')
        axes[1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # 単一予測例
        test_drug = create_dummy_molecule()
        test_protein = create_dummy_protein()
    
        interaction_prob = dti_system.predict_interaction(test_drug, test_protein)
        print(f"\nPredicted interaction probability: {interaction_prob:.4f}")
    

> **創薬におけるGNNの利点** :
> 
>   * **分子構造の直接モデル化** : グラフ表現により原子間の結合関係を自然に扱える
>   * **転移学習** : 大規模データで事前学習し、小規模データでファインチューニング
>   * **解釈可能性** : アテンション機構により重要な部分構造を特定可能
>   * **高速スクリーニング** : 数百万の化合物を短時間で評価可能
> 

* * *

## 5.5 その他の応用

### 知識グラフ (Knowledge Graphs)

**知識グラフ** は、エンティティ（ノード）とその関係（エッジ）を表現したグラフです。GNNは知識グラフ補完（missing link prediction）、質問応答、推論タスクに応用されます。
    
    
    ```mermaid
    graph LR
        A[Entity: Barack Obama] -->|born_in| B[Entity: Hawaii]
        A -->|president_of| C[Entity: USA]
        C -->|capital| D[Entity: Washington D.C.]
        A -->|married_to| E[Entity: Michelle Obama]
        E -->|born_in| F[Entity: Chicago]
    
        style A fill:#e3f2fd
        style B fill:#c8e6c9
        style C fill:#fff9c4
        style D fill:#ffccbc
        style E fill:#f8bbd0
        style F fill:#c8e6c9
    ```

#### 知識グラフ補完の実装例
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import RGCNConv
    
    class KnowledgeGraphCompletion(nn.Module):
        """
        Relational GCN (R-GCN) による知識グラフ補完
    
        異なる関係タイプごとに異なる重み行列を使用
        """
    
        def __init__(self, num_entities, num_relations, hidden_dim, num_layers=2):
            """
            Args:
                num_entities: エンティティ数
                num_relations: 関係タイプ数
                hidden_dim: 隠れ層次元
                num_layers: R-GCN層の数
            """
            super(KnowledgeGraphCompletion, self).__init__()
    
            # エンティティ埋め込み
            self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
    
            # R-GCN層
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
                )
    
            # 関係埋め込み（リンク予測用）
            self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
    
        def forward(self, edge_index, edge_type):
            """
            Args:
                edge_index: エッジインデックス
                edge_type: エッジタイプ（関係ID）
    
            Returns:
                entity_embeddings: エンティティ埋め込み
            """
            x = self.entity_embedding.weight
    
            # R-GCN層で特徴伝播
            for conv in self.convs:
                x = conv(x, edge_index, edge_type)
                x = F.relu(x)
    
            return x
    
        def score_triple(self, head, relation, tail, entity_embeddings):
            """
            トリプル (head, relation, tail) のスコア計算
    
            DistMultスコア関数を使用:
            score(h, r, t) = h^T R_r t
    
            Args:
                head: ヘッドエンティティID
                relation: 関係ID
                tail: テールエンティティID
                entity_embeddings: エンティティ埋め込み
    
            Returns:
                scores: トリプルスコア
            """
            h_emb = entity_embeddings[head]
            r_emb = self.relation_embedding(relation)
            t_emb = entity_embeddings[tail]
    
            # DistMult: element-wise product
            scores = (h_emb * r_emb * t_emb).sum(dim=1)
    
            return scores
    
    # 使用例
    if __name__ == "__main__":
        # ダミー知識グラフ
        num_entities = 100
        num_relations = 10
    
        # ダミーエッジ
        edge_index = torch.randint(0, num_entities, (2, 500))
        edge_type = torch.randint(0, num_relations, (500,))
    
        # モデル
        kg_model = KnowledgeGraphCompletion(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=64,
            num_layers=2
        )
    
        # エンティティ埋め込み取得
        entity_emb = kg_model(edge_index, edge_type)
    
        # トリプルスコア計算例
        head = torch.tensor([0, 1, 2])
        relation = torch.tensor([0, 1, 2])
        tail = torch.tensor([10, 11, 12])
    
        scores = kg_model.score_triple(head, relation, tail, entity_emb)
        print(f"Triple scores: {scores}")
    

### 交通ネットワーク (Traffic Networks)

**交通ネットワーク** では、道路や交差点をノード、道路をエッジとしてモデル化します。GNNは交通流予測、渋滞予測、最適経路探索に応用されます。

#### 時空間グラフニューラルネットワーク (Spatial-Temporal GNN)

交通予測では空間的依存関係（道路ネットワーク）と時間的依存関係（時系列）の両方を考慮する必要があります。
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class SpatialTemporalGNN(nn.Module):
        """
        時空間グラフニューラルネットワーク
    
        GCN（空間）+ LSTM（時間）のハイブリッド
        """
    
        def __init__(self, num_nodes, node_features, hidden_dim, output_dim, num_timesteps):
            """
            Args:
                num_nodes: ノード数（交差点/センサー数）
                node_features: ノード特徴量次元
                hidden_dim: 隠れ層次元
                output_dim: 出力次元（予測対象、例: 交通量）
                num_timesteps: 時系列長
            """
            super(SpatialTemporalGNN, self).__init__()
    
            # 空間モジュール（GCN）
            self.gcn1 = GCNConv(node_features, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
    
            # 時間モジュール（LSTM）
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
    
            # 出力層
            self.fc = nn.Linear(hidden_dim, output_dim)
    
        def forward(self, x_seq, edge_index):
            """
            Args:
                x_seq: 時系列ノード特徴量 (batch, num_timesteps, num_nodes, features)
                edge_index: エッジインデックス（静的グラフ構造）
    
            Returns:
                predictions: 予測値 (batch, num_nodes, output_dim)
            """
            batch_size, num_timesteps, num_nodes, _ = x_seq.size()
    
            # 各時刻でGCNを適用
            spatial_features = []
            for t in range(num_timesteps):
                x_t = x_seq[:, t, :, :].reshape(-1, x_seq.size(-1))  # (batch*nodes, features)
    
                # GCN層
                x_t = self.gcn1(x_t, edge_index)
                x_t = F.relu(x_t)
                x_t = self.gcn2(x_t, edge_index)
                x_t = F.relu(x_t)
    
                # (batch, nodes, hidden_dim)に復元
                x_t = x_t.view(batch_size, num_nodes, -1)
                spatial_features.append(x_t)
    
            # (batch, num_timesteps, nodes, hidden_dim)
            spatial_features = torch.stack(spatial_features, dim=1)
    
            # ノードごとにLSTMを適用
            predictions = []
            for node_idx in range(num_nodes):
                node_seq = spatial_features[:, :, node_idx, :]  # (batch, timesteps, hidden)
    
                # LSTM
                lstm_out, _ = self.lstm(node_seq)  # (batch, timesteps, hidden)
    
                # 最後の時刻の出力
                last_output = lstm_out[:, -1, :]  # (batch, hidden)
    
                # 予測
                pred = self.fc(last_output)  # (batch, output_dim)
                predictions.append(pred)
    
            # (batch, nodes, output_dim)
            predictions = torch.stack(predictions, dim=1)
    
            return predictions
    
    # 使用例
    if __name__ == "__main__":
        # ダミー交通ネットワーク
        num_nodes = 50  # 交差点数
        node_features = 4  # 特徴量（速度、密度、流量など）
        num_timesteps = 12  # 過去12時間
    
        # ダミーデータ
        batch_size = 8
        x_seq = torch.randn(batch_size, num_timesteps, num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
    
        # モデル
        st_gnn = SpatialTemporalGNN(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=64,
            output_dim=1,  # 交通量予測
            num_timesteps=num_timesteps
        )
    
        # 予測
        predictions = st_gnn(x_seq, edge_index)
        print(f"Predictions shape: {predictions.shape}")  # (batch, nodes, 1)
    

* * *

## 演習問題

**演習1: ノード分類の改善**

**課題** : NodeClassificationGCNモデルを改善し、より高い精度を達成してください。

**改善案** :

  1. **アーキテクチャ変更** : より深い層、Residual接続、Attention機構の追加
  2. **正則化強化** : Dropoutの調整、L2正則化、DropEdge
  3. **ハイパーパラメータ最適化** : 学習率、隠れ層次元、層数の調整
  4. **データ拡張** : ノード特徴量の正規化、グラフ拡張技術

**評価基準** : Coraデータセットでテスト精度85%以上を目指す

**演習2: 分子特性予測の拡張**

**課題** : MolecularGCNモデルを拡張し、複数の分子特性を同時に予測するマルチタスク学習を実装してください。

**仕様** :

  * 複数の出力ヘッド（毒性、溶解度、活性など）
  * タスク間の共有表現学習
  * タスク別の損失重み付け
  * 各タスクの評価メトリクス（AUC, RMSE）

**ヒント** : 共有GCN層 + タスク別MLP分類器の構成を検討

**演習3: 推薦システムのCold-Start問題**

**課題** : リンク予測ベースの推薦システムにおけるCold-Start問題（新規ユーザー/アイテム）への対処法を実装してください。

**実装項目** :

  1. **Side Information活用** : ユーザー/アイテムの属性情報を特徴量として使用
  2. **Inductive学習** : GraphSAGEなどで新規ノードに対応
  3. **Content-based Filtering統合** : GNN推薦とContent-basedのハイブリッド
  4. **Meta-learning** : Few-shot learningで新規ユーザーに素早く適応

**評価** : 新規ユーザー/アイテムでの推薦精度を測定

**演習4: DTI予測の解釈可能性**

**課題** : DTIPredictorモデルに解釈可能性機能を追加してください。

**実装内容** :

  * **アテンション可視化** : 重要な原子/部分構造を特定
  * **GradCAM適用** : 予測に重要な領域をヒートマップで表示
  * **部分構造分析** : 相互作用に寄与する分子断片を抽出
  * **アミノ酸重要度** : 結合部位の特定

**出力** : 分子グラフ上の重要度スコアを可視化

**演習5: 時空間グラフ予測の実装**

**課題** : 実際の交通データセット（例: METR-LA）を使って時空間GNNによる交通量予測を実装してください。

**要件** :

  1. **データ前処理** : 時系列データの正規化、欠損値処理
  2. **モデル構築** : GCN + LSTM/GRU/Transformerのハイブリッド
  3. **マルチステップ予測** : 1時間先、3時間先、6時間先を予測
  4. **評価** : MAE, RMSE, MAPEで性能評価

**発展課題** : アテンション機構による動的グラフ構造学習を実装

* * *

## まとめ

この章では、グラフニューラルネットワークの実践的応用について学習しました：

  * **ノード分類** : 半教師あり学習による効率的なノードラベル予測
  * **グラフ分類** : 分子特性予測などグラフ全体の分類タスク
  * **リンク予測** : 推薦システムなどエッジ存在確率の予測
  * **創薬応用** : 薬剤-標的相互作用予測による創薬支援
  * **その他応用** : 知識グラフ、交通ネットワークなど多様な領域

GNNは構造化データを扱う強力なツールであり、様々な実世界の問題に応用できます。各タスクの特性を理解し、適切なアーキテクチャと学習戦略を選択することが成功の鍵です。
