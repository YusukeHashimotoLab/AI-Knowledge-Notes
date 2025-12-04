---
title: "Chapter 5: GNN Applications"
chapter_title: "Chapter 5: GNN Applications"
subtitle: Practical Applications from Molecular Prediction to Recommendation Systems
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
---

This chapter focuses on practical applications of GNN Applications. You will learn Build recommendation systems using link prediction and applications to knowledge graphs.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Implement node classification tasks with semi-supervised learning
  * ✅ Implement molecular property prediction through graph classification
  * ✅ Build recommendation systems using link prediction
  * ✅ Understand drug-target interaction prediction in drug discovery applications
  * ✅ Understand applications to knowledge graphs and traffic networks
  * ✅ Design and implement practical GNN projects
  * ✅ Understand GNN limitations and countermeasures

* * *

## 5.1 Node Classification

### Node Classification with Semi-Supervised Learning

**Node classification** is the task of assigning labels to each node in a graph.Typical applications include user interest classification in social networks and research field classification in paper citation networks.
    
    
    ```mermaid
    graph LR
        A[Input Graph] --> B[GNN LayersFeature Propagation]
        B --> C[Node Embeddings]
        C --> D[ClassifierLinear + Softmax]
        D --> E[Node Label Prediction]
    
        F[Training NodesLabeled] --> G[Loss CalculationCross-Entropy]
        E --> G
        G --> H[Backpropagation]
    
        style A fill:#e3f2fd
        style E fill:#c8e6c9
        style F fill:#fff9c4
        style G fill:#ffccbc
    ```

#### Advantages of Semi-Supervised Learning

Feature | Description | Advantage  
---|---|---  
**Small Amount of Labeled Data** | Only some nodes are labeled | Reduced annotation cost  
**Graph Structure Utilization** | Information propagation from neighboring nodes | Improved accuracy and generalization  
**Transductive Learning** | Test data is included in the graph | Can utilize entire graph structure  
**Homophily Exploitation** | Similar nodes tend to be connected | Effective label propagation  
  
### Node Classification Implementation with Cora Dataset

**The Cora dataset** is a citation network of machine learning papers. Each node (paper) is classified into one of seven research fields.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    # - seaborn>=0.12.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: The Cora datasetis a citation network of machine learning pa
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
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
        Node classification model using Graph Convolutional Network
    
        Architecture:
        - 2-layer GCN
        - Dropout for regularization
        - Log-softmax output for multi-class classification
        """
    
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            """
            Args:
                num_features: Input feature dimension
                hidden_dim: Hidden layer dimension
                num_classes: Number of classes
                dropout: Dropout rate
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
                log_probs: Class probability for each node (log-space)
            """
            x, edge_index = data.x, data.edge_index
    
            # Layer 1: GCN + ReLU + Dropout
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
            # Layer 2: GCN
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
        def get_embeddings(self, data):
            """
            Get node embeddings (for visualization)
    
            Returns:
                GCN output from first layer (hidden representation)
            """
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            return x.detach()
    
    
    class NodeClassificationGAT(nn.Module):
        """
        Node classification model using Graph Attention Network
    
        Features:
        - Multi-head attention mechanism
        - Learnable attention weights
        - Better handling of heterophilic graphs
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     heads=8, dropout=0.6):
            """
            Args:
                num_features: Input feature dimension
                hidden_dim: Hidden layer dimension (per head)
                num_classes: Number of classes
                heads: Number of attention heads
                dropout: Dropout rate
            """
            super(NodeClassificationGAT, self).__init__()
    
            # Layer 1: Multi-head GAT
            self.conv1 = GATConv(
                num_features,
                hidden_dim,
                heads=heads,
                dropout=dropout
            )
    
            # Layer 2: Single-head GAT for classification
            self.conv2 = GATConv(
                hidden_dim * heads,  # Concatenate all head outputs from layer 1
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
                log_probs: Class probability
            """
            x, edge_index = data.x, data.edge_index
    
            # Layer 1: GAT + ELU + Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
    
            # Layer 2: GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
    
    class NodeClassificationTrainer:
        """
        Training and evaluation class for node classification models
        """
    
        def __init__(self, model, data, device='cuda'):
            """
            Args:
                model: GNNModel（GCN or GAT）
                data: PyG Data object
                device: Computation device
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
            Train for one epoch
    
            Returns:
                loss: Training loss
            """
            self.model.train()
            self.optimizer.zero_grad()
    
            # Forward pass
            out = self.model(self.data)
    
            # Calculate loss only for training nodes (semi-supervised learning)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
    
            # Backward pass
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        @torch.no_grad()
        def evaluate(self, mask):
            """
            Evaluation
    
            Args:
                mask: Node mask for evaluation targets (train/val/test)
    
            Returns:
                accuracy: Accuracy
            """
            self.model.eval()
            out = self.model(self.data)
            pred = out.argmax(dim=1)
    
            correct = (pred[mask] == self.data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
    
            return acc
    
        def train(self, epochs=200, early_stopping_patience=20, verbose=True):
            """
            Training loop
    
            Args:
                epochs: Number of epochs
                early_stopping_patience: Tolerance epochs for early stopping
                verbose: Log display
    
            Returns:
                best_val_acc: Best validation accuracy
            """
            best_val_acc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # Training
                loss = self.train_epoch()
                self.train_losses.append(loss)
    
                # Evaluation
                train_acc = self.evaluate(self.data.train_mask)
                val_acc = self.evaluate(self.data.val_mask)
                self.val_accs.append(val_acc)
    
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
    
                # Log output
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
            # Load best model
            self.model.load_state_dict(self.best_model_state)
    
            return best_val_acc
    
        @torch.no_grad()
        def test(self):
            """
            Final evaluation on test set
    
            Returns:
                test_acc: Test accuracy
                predictions: Predicted labels for all nodes
                embeddings: Node Embeddings
            """
            self.model.eval()
    
            # Prediction
            out = self.model(self.data)
            pred = out.argmax(dim=1)
    
            # Test accuracy
            test_acc = self.evaluate(self.data.test_mask)
    
            # Get embeddings (for GCN)
            embeddings = None
            if hasattr(self.model, 'get_embeddings'):
                embeddings = self.model.get_embeddings(self.data).cpu().numpy()
    
            return test_acc, pred.cpu().numpy(), embeddings
    
        def plot_training_history(self):
            """
            Plot training history
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # Loss
            axes[0].plot(self.train_losses, label='Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
            # ValidationAccuracy
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
            Visualize embeddings using t-SNE
    
            Args:
                embeddings: Node Embeddings (N, D)
                labels: Node labels (N,)
                title: Plot title
            """
            # Reduce to 2D with t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
    
            # Plot
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
    
    
    # Usage example
    if __name__ == "__main__":
        # Load data
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
    
        # GCN model
        print("\n=== Training GCN ===")
        gcn_model = NodeClassificationGCN(
            num_features=dataset.num_features,
            hidden_dim=16,
            num_classes=dataset.num_classes,
            dropout=0.5
        )
    
        gcn_trainer = NodeClassificationTrainer(gcn_model, data)
        best_val_acc = gcn_trainer.train(epochs=200, early_stopping_patience=20)
    
        # Test evaluation
        test_acc, predictions, embeddings = gcn_trainer.test()
        print(f"\nGCN Test Accuracy: {test_acc:.4f}")
    
        # Plot training history
        gcn_trainer.plot_training_history()
    
        # Embedding visualization
        if embeddings is not None:
            gcn_trainer.visualize_embeddings(
                embeddings,
                data.y.cpu().numpy(),
                title="GCN Node Embeddings (t-SNE)"
            )
    
        # GAT model
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
    
        # Test evaluation
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
    

> **Transductive vs Inductive Learning** :
> 
>   * **Transductive** : Test nodes are also included in the graph. The entire graph structure can be utilized during training.
>   * **Inductive** : Test nodes are unknown. Generalization to new graphs is required. GraphSAGE and similar models support this.
> 

* * *

## 5.2 Graph Classification

### Application to Molecular Property Prediction

**Graph classification** is the task of classifying entire graphs.Typical applications include molecular bioactivity prediction, protein function classification, and social network community classification.
    
    
    ```mermaid
    graph TB
        A[Molecular Graph] --> B[GNN LayersAtom Feature Propagation]
        B --> C[Node Embeddings]
        C --> D[Graph PoolingGraph-level Representation]
        D --> E[ClassifierMLP]
        E --> F[Molecular Property PredictionToxicity/Activity etc.]
    
        style A fill:#e3f2fd
        style D fill:#fff9c4
        style F fill:#c8e6c9
    ```

#### Types of Graph Pooling

Method | Description | Formula  
---|---|---  
**Global Mean Pooling** | Average of all node embeddings | $\mathbf{h}_G = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{h}_v$  
**Global max pooling** | Maximum value per dimension | $\mathbf{h}_G = \max_{v \in \mathcal{V}} \mathbf{h}_v$  
**Global Sum Pooling** | Sum of all node embeddings | $\mathbf{h}_G = \sum_{v \in \mathcal{V}} \mathbf{h}_v$  
**Attention Pooling** | Attention-weighted sum | $\mathbf{h}_G = \sum_{v \in \mathcal{V}} \alpha_v \mathbf{h}_v$  
  
### Implementation with MUTAG Molecular Dataset

**The MUTAG dataset** is a dataset for classifying the mutagenicity (property of causing DNA mutations) of 188 compounds.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
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
        GCN model for molecular graph classification
    
        Architecture:
        - Multiple GCN layers for feature propagation
        - Global pooling for graph-level representation
        - MLP classifier for prediction
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     num_layers=3, dropout=0.5, pooling='mean'):
            """
            Args:
                num_features: Node feature dimension
                hidden_dim: Hidden layer dimension
                num_classes: Number of classes
                num_layers: Number of GCN layers
                dropout: Dropout rate
                pooling: Pooling method ('mean', 'max', 'sum')
            """
            super(MolecularGCN, self).__init__()
    
            # GCN layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(num_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
            # Classifier
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    
            self.dropout = dropout
            self.pooling = pooling
    
        def forward(self, data):
            """
            Args:
                data: PyG Batch object (x, edge_index, batch)
    
            Returns:
                logits: Graph classification logits
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # Feature propagation with GCN layers
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
    
            # Classifier
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
    
            return x
    
    
    class MolecularGIN(nn.Module):
        """
        Graph Isomorphism Network (GIN) for molecular graphs
    
        GIN has expressive power to distinguish graph isomorphisms
        Equivalent discriminative power to WL test
        """
    
        def __init__(self, num_features, hidden_dim, num_classes,
                     num_layers=5, dropout=0.5):
            """
            Args:
                num_features: Node feature dimension
                hidden_dim: Hidden layer dimension
                num_classes: Number of classes
                num_layers: Number of GIN layers
                dropout: Dropout rate
            """
            super(MolecularGIN, self).__init__()
    
            # GIN layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # Initial layer
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # Intermediate layers
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # Classifier
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    
            self.dropout = dropout
    
        def forward(self, data):
            """
            Args:
                data: PyG Batch object
    
            Returns:
                logits: Graph classification logits
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # GIN layers
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
    
            # Global sum pooling (standard for GIN)
            x = global_add_pool(x, batch)
    
            # Classifier
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
    
            return x
    
    
    class GraphClassificationTrainer:
        """
        Training and evaluation class for graph classification model
        """
    
        def __init__(self, model, train_loader, val_loader, test_loader,
                     num_classes, device='cuda'):
            """
            Args:
                model: GNNModel
                train_loader, val_loader, test_loader: DataLoader
                num_classes: Number of classes
                device: Computation device
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
    
            # Loss Function
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
    
            self.train_losses = []
            self.val_accs = []
    
        def train_epoch(self):
            """
            Train for one epoch
    
            Returns:
                avg_loss: Average loss
            """
            self.model.train()
            total_loss = 0
    
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
    
                # Forward pass
                out = self.model(data)
    
                # Loss calculation
                if self.num_classes == 2:
                    loss = self.criterion(out.squeeze(), data.y.float())
                else:
                    loss = self.criterion(out, data.y)
    
                # Backward pass
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item() * data.num_graphs
    
            return total_loss / len(self.train_loader.dataset)
    
        @torch.no_grad()
        def evaluate(self, loader):
            """
            Evaluation
    
            Args:
                loader: DataLoader
    
            Returns:
                accuracy: Accuracy
                all_preds, all_labels: Predictions and labels (for ROC AUC calculation)
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
            Training loop
    
            Args:
                epochs: Number of epochs
                early_stopping_patience: Tolerance epochs for early stopping
                verbose: Log display
    
            Returns:
                best_val_acc: Best validation accuracy
            """
            best_val_acc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # Training
                loss = self.train_epoch()
                self.train_losses.append(loss)
    
                # Evaluation
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
    
                # Log
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
            # Load best model
            self.model.load_state_dict(self.best_model_state)
            return best_val_acc
    
        def test(self):
            """
            Final evaluation on test set
    
            Returns:
                test_acc: Test accuracy
                test_auc: ROC AUC (for binary classification)
            """
            test_acc, preds, labels = self.evaluate(self.test_loader)
    
            # ROC AUC (binary classification only)
            test_auc = None
            if self.num_classes == 2:
                test_auc = roc_auc_score(labels, preds)
    
            return test_acc, test_auc
    
    
    # Usage example
    if __name__ == "__main__":
        # Load dataset
        dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    
        print(f"Dataset: {dataset.name}")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
    
        # Data split
        indices = list(range(len(dataset)))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=[data.y.item() for data in dataset]
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, random_state=42, stratify=[dataset[i].y.item() for i in train_indices]
        )
    
        # Create DataLoaders
        train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=32, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_indices], batch_size=32)
        test_loader = DataLoader([dataset[i] for i in test_indices], batch_size=32)
    
        # GCN model
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
    
        # Test evaluation
        test_acc, test_auc = gcn_trainer.test()
        print(f"\nGCN Test Accuracy: {test_acc:.4f}")
        if test_auc:
            print(f"GCN Test ROC AUC: {test_auc:.4f}")
    
        # GINModel
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
    
        # Test evaluation
        test_acc, test_auc = gin_trainer.test()
        print(f"\nGIN Test Accuracy: {test_acc:.4f}")
        if test_auc:
            print(f"GIN Test ROC AUC: {test_auc:.4f}")
    
        # Compare training history
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

## 5.3 Link Prediction

### Application to Recommendation Systems

**Link prediction** is the task of predicting the probability that an edge exists between two nodes in a graph. It is applied to recommendation systems (User-Item graphs), knowledge graph completion, social network analysis, and more.
    
    
    ```mermaid
    graph TB
        A[GraphHide Some Edges] --> B[GNN EncoderNode Embedding Learning]
        B --> C[Node Embeddingsu, v]
        C --> D[Link DecoderScore Calculation]
        D --> E[Link Probabilityp(u,v)]
    
        F[Positive Edges] --> G[Loss CalculationBCE Loss]
        H[Negative EdgesSampling] --> G
        E --> G
        G --> I[Parameter Update]
    
        style A fill:#e3f2fd
        style C fill:#fff9c4
        style E fill:#c8e6c9
        style G fill:#ffccbc
    ```

#### Types of Link Decoders

Method | Formula | Feature  
---|---|---  
**Inner Product** | $s(u,v) = \mathbf{z}_u^\top \mathbf{z}_v$ | Simple, computationally efficient  
**Cosine Similarity** | $s(u,v) = \frac{\mathbf{z}_u^\top \mathbf{z}_v}{\|\mathbf{z}_u\| \|\mathbf{z}_v\|}$ | Normalized, scale-invariant  
**MLP Decoder** | $s(u,v) = \text{MLP}([\mathbf{z}_u; \mathbf{z}_v])$ | High expressiveness, models nonlinear relationships  
**DistMult** | $s(u,r,v) = \mathbf{z}_u^\top \mathbf{R}_r \mathbf{z}_v$ | Considering relationship types (knowledge graphs)  
  
### Implementation of Recommendation System
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
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
        GNN encoder for link prediction
    
        Learn node embeddings using GraphSAGE
        """
    
        def __init__(self, num_features, hidden_dim, embedding_dim, num_layers=2, dropout=0.5):
            """
            Args:
                num_features: Node feature dimension
                hidden_dim: Hidden layer dimension
                embedding_dim: Final embedding dimension
                num_layers: Number of GraphSAGE layers
                dropout: Dropout rate
            """
            super(LinkPredictionGNN, self).__init__()
    
            # GraphSAGE layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(num_features, hidden_dim))
    
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
    
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))
    
            self.dropout = dropout
    
        def encode(self, x, edge_index):
            """
            Encode node embeddings
    
            Args:
                x: Node features
                edge_index: Edge index
    
            Returns:
                z: Node Embeddings
            """
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
    
            return x
    
        def decode(self, z, edge_index, method='inner_product'):
            """
            Decode edge scores
    
            Args:
                z: Node Embeddings
                edge_index: Edge index
                method: Decoding method ('inner_product', 'cosine', 'mlp')
    
            Returns:
                scores: Edge scores
            """
            src, dst = edge_index
    
            if method == 'inner_product':
                # Inner product
                scores = (z[src] * z[dst]).sum(dim=1)
            elif method == 'cosine':
                # Cosine similarity
                scores = F.cosine_similarity(z[src], z[dst])
            else:
                raise ValueError(f"Unknown decode method: {method}")
    
            return scores
    
        def forward(self, x, edge_index, decode_edge_index=None):
            """
            Forward pass
    
            Args:
                x: Node features
                edge_index: Edge for message passing
                decode_edge_index: Edges for score calculation (edge_index if None)
    
            Returns:
                scores: Edge scores
            """
            z = self.encode(x, edge_index)
    
            if decode_edge_index is None:
                decode_edge_index = edge_index
    
            scores = self.decode(z, decode_edge_index)
            return scores
    
    
    class MLPDecoder(nn.Module):
        """
        MLP-based link decoder
    
        Models more complex nonlinear relationships
        """
    
        def __init__(self, embedding_dim, hidden_dim=128):
            """
            Args:
                embedding_dim: Node embedding dimension
                hidden_dim: MLP hidden layer dimension
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
                z: Node Embeddings
                edge_index: Edge index
    
            Returns:
                scores: Edge scores
            """
            src, dst = edge_index
    
            # Concatenate node embeddings
            edge_features = torch.cat([z[src], z[dst]], dim=1)
    
            # Calculate score with MLP
            scores = self.mlp(edge_features).squeeze()
    
            return scores
    
    
    class RecommendationSystem:
        """
        GNN-based recommendation system
    
        Perform link prediction on User-Item graph to achieve item recommendation
        """
    
        def __init__(self, model, decoder=None, device='cuda'):
            """
            Args:
                model: GNN encoder
                decoder: Link decoder (inner product if None)
                device: Computation device
            """
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)
            self.decoder = decoder.to(self.device) if decoder else None
    
            # Optimizer
            params = list(model.parameters())
            if decoder:
                params += list(decoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)
    
            self.train_losses = []
            self.val_aucs = []
    
        def train_epoch(self, data, train_pos_edge_index):
            """
            Train for one epoch
    
            Args:
                data: PyG Data object
                train_pos_edge_index: Positive edges for training
    
            Returns:
                loss: Loss value
            """
            self.model.train()
            if self.decoder:
                self.decoder.train()
    
            self.optimizer.zero_grad()
    
            # Get node embeddings
            z = self.model.encode(data.x, train_pos_edge_index)
    
            # Negative sampling
            neg_edge_index = negative_sampling(
                edge_index=train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=train_pos_edge_index.size(1)
            )
    
            # Positive scores
            if self.decoder:
                pos_scores = self.decoder(z, train_pos_edge_index)
                neg_scores = self.decoder(z, neg_edge_index)
            else:
                pos_scores = self.model.decode(z, train_pos_edge_index)
                neg_scores = self.model.decode(z, neg_edge_index)
    
            # Loss calculation (BCE Loss)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss
    
            # Backward pass
            loss.backward()
            self.optimizer.step()
    
            return loss.item()
    
        @torch.no_grad()
        def evaluate(self, data, pos_edge_index, neg_edge_index):
            """
            Evaluation
    
            Args:
                data: PyG Data object
                pos_edge_index: Positive Edges
                neg_edge_index: Negative edges
    
            Returns:
                auc: ROC AUC
                ap: Average Precision
            """
            self.model.eval()
            if self.decoder:
                self.decoder.eval()
    
            # Encode with all edges
            z = self.model.encode(data.x, data.train_pos_edge_index)
    
            # Score calculation
            if self.decoder:
                pos_scores = self.decoder(z, pos_edge_index).cpu().numpy()
                neg_scores = self.decoder(z, neg_edge_index).cpu().numpy()
            else:
                pos_scores = self.model.decode(z, pos_edge_index).cpu().numpy()
                neg_scores = self.model.decode(z, neg_edge_index).cpu().numpy()
    
            # Labels and scores
            scores = np.concatenate([pos_scores, neg_scores])
            labels = np.concatenate([
                np.ones(pos_scores.shape[0]),
                np.zeros(neg_scores.shape[0])
            ])
    
            # Calculate metrics
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
    
            return auc, ap
    
        def train(self, data, epochs=100, early_stopping_patience=10, verbose=True):
            """
            Training loop
    
            Args:
                data: PyG Data object (train_test_split_edges applied)
                epochs: Number of epochs
                early_stopping_patience: Tolerance epochs for early stopping
                verbose: Log display
    
            Returns:
                best_val_auc: Best validation AUC
            """
            best_val_auc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # Training
                loss = self.train_epoch(data, data.train_pos_edge_index)
                self.train_losses.append(loss)
    
                # Evaluation
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
    
                # Log
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
    
            # Load best model
            self.model.load_state_dict(self.best_model_state)
            if self.decoder:
                self.decoder.load_state_dict(self.best_decoder_state)
    
            return best_val_auc
    
        def test(self, data):
            """
            Final evaluation on test set
    
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
            Recommend items to a user
    
            Args:
                data: PyG Data object
                user_id: User node ID
                k: Number of recommendations
                exclude_known: Whether to exclude known items
    
            Returns:
                recommended_items: List of recommended item IDs
                scores: Corresponding scores
            """
            self.model.eval()
            if self.decoder:
                self.decoder.eval()
    
            # Get node embeddings
            z = self.model.encode(data.x, data.train_pos_edge_index)
    
            # Calculate scores with all items
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
    
            # Exclude known items
            if exclude_known:
                known_items = data.train_pos_edge_index[1][
                    data.train_pos_edge_index[0] == user_id
                ].cpu().numpy()
                scores[known_items] = -1
    
            # Select top-k
            top_k_indices = np.argsort(scores)[-k:][::-1]
            top_k_scores = scores[top_k_indices]
    
            return top_k_indices, top_k_scores
    
    
    # Usage example
    if __name__ == "__main__":
        # Load data (example: Cora dataset)
        from torch_geometric.datasets import Planetoid
    
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    
        # Split edges for link prediction
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
    
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Training edges: {data.train_pos_edge_index.size(1)}")
        print(f"Validation edges: {data.val_pos_edge_index.size(1)}")
        print(f"Test edges: {data.test_pos_edge_index.size(1)}")
    
        # Model initialization（Inner Product Decoder）
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
    
        # Test evaluation
        test_auc, test_ap = rec_system_ip.test(data)
        print(f"\nInner Product - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
        # Model initialization（MLP Decoder）
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
    
        # Test evaluation
        test_auc, test_ap = rec_system_mlp.test(data)
        print(f"\nMLP Decoder - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
        # Recommendation example
        user_id = 0
        recommended_items, scores = rec_system_mlp.recommend_items(
            data, user_id, k=10, exclude_known=True
        )
    
        print(f"\nTop-10 Recommendations for User {user_id}:")
        for idx, (item, score) in enumerate(zip(recommended_items, scores), 1):
            print(f"{idx}. Item {item}: Score {score:.4f}")
    
        # Compare training history
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

## 5.4 Drug Discovery

### Drug-Target Interaction Prediction

**Drug-Target Interaction (DTI) Prediction** is the task of predicting which protein targets a drug molecule will interact with. This is an important application in drug discovery that can significantly reduce the time and cost of screening candidate compounds.
    
    
    ```mermaid
    graph TB
        A[Drug Molecular Graph] --> B[Molecular GNNDrug Embedding]
        C[Target ProteinSequence/Structure] --> D[Protein EncoderTarget Embedding]
    
        B --> E[Drug Featuresz_drug]
        D --> F[Target Featuresz_target]
    
        E --> G[Interaction PredictorMLP/Bilinear]
        F --> G
    
        G --> H[Interaction ScoreBinding Affinity]
    
        style A fill:#e3f2fd
        style C fill:#fff9c4
        style H fill:#c8e6c9
    ```

#### DTI Prediction Approaches

Component | Method | Description  
---|---|---  
**Drug Encoder** | GCN, GIN, AttentiveFP | Feature extraction from molecular graph  
**Target Encoder** | CNN, LSTM, Transformer | Feature extraction from amino acid sequences/3D structure  
**Interaction Prediction** | MLP, Bilinear, Attention | Score calculation for drug-target pairs  
**Loss Function** | BCE, MSE, Ranking Loss | Binding affinity or classification labels  
  
### Implementation of DTI Prediction System
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    class DrugEncoder(nn.Module):
        """
        Encoder for drug molecular graph
    
        Learn molecular feature vectors using GCN
        """
    
        def __init__(self, num_atom_features, hidden_dim, embedding_dim, num_layers=3):
            """
            Args:
                num_atom_features: Atom feature dimension
                hidden_dim: Hidden layer dimension
                embedding_dim: Final embedding dimension
                num_layers: Number of GCN layers
            """
            super(DrugEncoder, self).__init__()
    
            # GCN layers
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
                data: PyG Batch object (multiple molecular graphs)
    
            Returns:
                drug_embeddings: Molecular embeddings (batch_size, embedding_dim)
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
    
            # Feature propagation with GCN layers
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index)
                x = bn(x)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
    
            # Graph-level representation with graph pooling
            drug_embeddings = global_mean_pool(x, batch)
    
            return drug_embeddings
    
    
    class ProteinEncoder(nn.Module):
        """
        Encoder for protein sequences
    
        Feature extraction from amino acid sequences using 1D CNN
        """
    
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_filters=128, kernel_sizes=[3, 5, 7]):
            """
            Args:
                vocab_size: Number of amino acid types (typically 20 + padding)
                embedding_dim: Amino acid embedding dimension
                hidden_dim: Final hidden layer dimension
                num_filters: Number of CNN filters
                kernel_sizes: List of kernel sizes
            """
            super(ProteinEncoder, self).__init__()
    
            # Amino acid embedding
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
            # Multi-scale 1D CNN
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
                for k in kernel_sizes
            ])
    
            # Fully connected layer
            self.fc = nn.Sequential(
                nn.Linear(num_filters * len(kernel_sizes), hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
        def forward(self, protein_sequences):
            """
            Args:
                protein_sequences: Amino acid sequences (batch_size, seq_len)
    
            Returns:
                protein_embeddings: Protein embeddings (batch_size, hidden_dim)
            """
            # Amino acid embedding
            x = self.embedding(protein_sequences)  # (batch, seq_len, emb_dim)
            x = x.transpose(1, 2)  # (batch, emb_dim, seq_len)
    
            # Multi-scale CNN
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
                # Global max pooling
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_outputs.append(pooled)
    
            # Concatenate
            x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))
    
            # Fully connected layer
            protein_embeddings = self.fc(x)
    
            return protein_embeddings
    
    
    class DTIPredictor(nn.Module):
        """
        Drug-target interaction prediction model
    
        Combine drug encoder and target encoder to
        predict interaction scores
        """
    
        def __init__(self, drug_encoder, protein_encoder, hidden_dim):
            """
            Args:
                drug_encoder: DrugEncoder instance
                protein_encoder: ProteinEncoder instance
                hidden_dim: Hidden layer dimension for interaction prediction
            """
            super(DTIPredictor, self).__init__()
    
            self.drug_encoder = drug_encoder
            self.protein_encoder = protein_encoder
    
            # Interaction PredictionMLP
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
                drug_data: PyG Batch (drug molecular graph)
                protein_sequences: Protein sequences (batch_size, seq_len)
    
            Returns:
                interaction_scores: Interaction scores (batch_size,)
            """
            # Drug embedding
            drug_emb = self.drug_encoder(drug_data)
    
            # Target embedding
            protein_emb = self.protein_encoder(protein_sequences)
    
            # Concatenate
            combined = torch.cat([drug_emb, protein_emb], dim=1)
    
            # Interaction score
            scores = self.interaction_mlp(combined).squeeze()
    
            return scores
    
    
    class DTIPredictionSystem:
        """
        Drug-target interaction prediction system
        """
    
        def __init__(self, model, device='cuda'):
            """
            Args:
                model: DTIPredictor instance
                device: Computation device
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
            Train for one epoch
    
            Args:
                train_loader: DataLoader (drug_data, protein_seq, label)
    
            Returns:
                avg_loss: Average loss
            """
            self.model.train()
            total_loss = 0
    
            for drug_batch, protein_batch, labels in train_loader:
                # Transfer to device
                protein_batch = protein_batch.to(self.device)
                labels = labels.to(self.device).float()
    
                self.optimizer.zero_grad()
    
                # Forward pass
                scores = self.model(drug_batch, protein_batch)
    
                # Loss calculation
                loss = self.criterion(scores, labels)
    
                # Backward pass
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
    
            return total_loss / len(train_loader)
    
        @torch.no_grad()
        def evaluate(self, loader):
            """
            Evaluation
    
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
            Training loop
    
            Args:
                train_loader, val_loader: DataLoader
                epochs: Number of epochs
                early_stopping_patience: Tolerance epochs for early stopping
                verbose: Log display
    
            Returns:
                best_val_auc: Best validation AUC
            """
            best_val_auc = 0
            patience_counter = 0
    
            for epoch in range(1, epochs + 1):
                # Training
                loss = self.train_epoch(train_loader)
                self.train_losses.append(loss)
    
                # Evaluation
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
    
                # Log
                if verbose and epoch % 5 == 0:
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}')
    
            # Load best model
            self.model.load_state_dict(self.best_model_state)
            return best_val_auc
    
        def test(self, test_loader):
            """
            Final evaluation on test set
    
            Returns:
                test_auc: ROC AUC
                test_auprc: AUPRC
            """
            return self.evaluate(test_loader)
    
        def predict_interaction(self, drug_graph, protein_sequence):
            """
            Predict interaction for a single drug-target pair
    
            Args:
                drug_graph: PyG Data object（Molecular Graph）
                protein_sequence: Amino acid sequence tensor
    
            Returns:
                interaction_prob: Interaction probability
            """
            self.model.eval()
    
            with torch.no_grad():
                # Batchify
                drug_batch = Batch.from_data_list([drug_graph]).to(self.device)
                protein_batch = protein_sequence.unsqueeze(0).to(self.device)
    
                # Prediction
                score = self.model(drug_batch, protein_batch)
                prob = torch.sigmoid(score).item()
    
            return prob
    
    
    # Usage example（Dummy data）
    if __name__ == "__main__":
        # Generate dummy data
        def create_dummy_molecule():
            """Generate dummy molecular graph"""
            num_atoms = np.random.randint(10, 30)
            x = torch.randn(num_atoms, 9)  # Atom features
            edge_index = torch.randint(0, num_atoms, (2, num_atoms * 2))
            return Data(x=x, edge_index=edge_index)
    
        def create_dummy_protein(max_len=1000):
            """Generate dummy protein sequence"""
            seq_len = np.random.randint(500, max_len)
            # Amino acid IDs (1-20, 0 is padding)
            seq = torch.randint(1, 21, (seq_len,))
            # Padding
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, dtype=torch.long)
                seq = torch.cat([seq, padding])
            return seq
    
        # Create dummy dataset
        num_samples = 1000
        drug_graphs = [create_dummy_molecule() for _ in range(num_samples)]
        protein_seqs = torch.stack([create_dummy_protein() for _ in range(num_samples)])
        labels = torch.randint(0, 2, (num_samples,))  # Random labels
    
        # Create DataLoaders（Simplified version）
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
    
        # Data split
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
    
        # Model initialization
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
    
        # Training
        print("=== Training DTI Prediction Model ===")
        dti_system = DTIPredictionSystem(dti_model)
        best_val_auc = dti_system.train(
            train_loader, val_loader, epochs=50, early_stopping_patience=10
        )
    
        # Test evaluation
        test_auc, test_auprc = dti_system.test(test_loader)
        print(f"\nTest AUC: {test_auc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")
    
        # Plot training history
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
    
        # Single prediction example
        test_drug = create_dummy_molecule()
        test_protein = create_dummy_protein()
    
        interaction_prob = dti_system.predict_interaction(test_drug, test_protein)
        print(f"\nPredicted interaction probability: {interaction_prob:.4f}")
    

> **GNN Advantages in Drug Discovery** :
> 
>   * **Direct modeling of molecular structure** : Graph representation naturally handles inter-atomic connection relationships
>   * **Transfer learning** : Pre-train on large-scale data, fine-tune on small-scale data
>   * **Interpretability** : Attention mechanisms can identify important substructures
>   * **High-speed screening** : Millions of compounds can be evaluated in a short time
> 

* * *

## 5.5 Other Applications

### Knowledge Graphs

**Knowledge graphs** are graphs that represent entities (nodes) and their relationships (edges). GNNs are applied to knowledge graph completion (missing link prediction), question answering, and reasoning tasks.
    
    
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

#### Knowledge Graph Completion Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import RGCNConv
    
    class KnowledgeGraphCompletion(nn.Module):
        """
        Knowledge graph completion using Relational GCN (R-GCN)
    
        Uses different weight matrices for different relation types
        """
    
        def __init__(self, num_entities, num_relations, hidden_dim, num_layers=2):
            """
            Args:
                num_entities: Number of entities
                num_relations: Number of relation types
                hidden_dim: Hidden layer dimension
                num_layers: R-Number of GCN layers
            """
            super(KnowledgeGraphCompletion, self).__init__()
    
            # Entity embedding
            self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
    
            # R-GCN layers
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
                )
    
            # Relation embeddings (for link prediction)
            self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
    
        def forward(self, edge_index, edge_type):
            """
            Args:
                edge_index: Edge index
                edge_type: Edge type (relation ID)
    
            Returns:
                entity_embeddings: Entity embedding
            """
            x = self.entity_embedding.weight
    
            # Feature propagation with R-GCN layers
            for conv in self.convs:
                x = conv(x, edge_index, edge_type)
                x = F.relu(x)
    
            return x
    
        def score_triple(self, head, relation, tail, entity_embeddings):
            """
            Calculate score for triple (head, relation, tail)
    
            Uses DistMult scoring function:
            score(h, r, t) = h^T R_r t
    
            Args:
                head: Head entity ID
                relation: Relation ID
                tail: Tail entity ID
                entity_embeddings: Entity embedding
    
            Returns:
                scores: Triple scores
            """
            h_emb = entity_embeddings[head]
            r_emb = self.relation_embedding(relation)
            t_emb = entity_embeddings[tail]
    
            # DistMult: element-wise product
            scores = (h_emb * r_emb * t_emb).sum(dim=1)
    
            return scores
    
    # Usage example
    if __name__ == "__main__":
        # Dummy knowledge graph
        num_entities = 100
        num_relations = 10
    
        # Dummy edges
        edge_index = torch.randint(0, num_entities, (2, 500))
        edge_type = torch.randint(0, num_relations, (500,))
    
        # Model
        kg_model = KnowledgeGraphCompletion(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=64,
            num_layers=2
        )
    
        # Get entity embeddings
        entity_emb = kg_model(edge_index, edge_type)
    
        # Example of triple score calculation
        head = torch.tensor([0, 1, 2])
        relation = torch.tensor([0, 1, 2])
        tail = torch.tensor([10, 11, 12])
    
        scores = kg_model.score_triple(head, relation, tail, entity_emb)
        print(f"Triple scores: {scores}")
    

### Traffic Networks

**Traffic networks** model roads and intersections as nodes and roads as edges.GNNs are applied to traffic flow prediction, congestion prediction, and optimal route search.

#### Spatial-Temporal Graph Neural Network

In traffic prediction, both spatial dependencies (road networks) and temporal dependencies (time series) must be considered.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class SpatialTemporalGNN(nn.Module):
        """
        Spatial-temporal graph neural network
    
        Hybrid of GCN (spatial) + LSTM (temporal)
        """
    
        def __init__(self, num_nodes, node_features, hidden_dim, output_dim, num_timesteps):
            """
            Args:
                num_nodes: Number of nodes (intersections/sensors)
                node_features: Node feature dimension
                hidden_dim: Hidden layer dimension
                output_dim: Output dimension (prediction target, e.g., traffic volume)
                num_timesteps: Time series length
            """
            super(SpatialTemporalGNN, self).__init__()
    
            # Spatial module (GCN)
            self.gcn1 = GCNConv(node_features, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
    
            # Temporal module (LSTM)
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
    
            # Output layer
            self.fc = nn.Linear(hidden_dim, output_dim)
    
        def forward(self, x_seq, edge_index):
            """
            Args:
                x_seq: Time series node features (batch, num_timesteps, num_nodes, features)
                edge_index: Edge index（Static graph structure）
    
            Returns:
                predictions: Prediction values (batch, num_nodes, output_dim)
            """
            batch_size, num_timesteps, num_nodes, _ = x_seq.size()
    
            # Apply GCN at each time step
            spatial_features = []
            for t in range(num_timesteps):
                x_t = x_seq[:, t, :, :].reshape(-1, x_seq.size(-1))  # (batch*nodes, features)
    
                # GCN layers
                x_t = self.gcn1(x_t, edge_index)
                x_t = F.relu(x_t)
                x_t = self.gcn2(x_t, edge_index)
                x_t = F.relu(x_t)
    
                # (batch, nodes, hidden_dim)Restore to
                x_t = x_t.view(batch_size, num_nodes, -1)
                spatial_features.append(x_t)
    
            # (batch, num_timesteps, nodes, hidden_dim)
            spatial_features = torch.stack(spatial_features, dim=1)
    
            # Apply LSTM for each node
            predictions = []
            for node_idx in range(num_nodes):
                node_seq = spatial_features[:, :, node_idx, :]  # (batch, timesteps, hidden)
    
                # LSTM
                lstm_out, _ = self.lstm(node_seq)  # (batch, timesteps, hidden)
    
                # Last timestep output
                last_output = lstm_out[:, -1, :]  # (batch, hidden)
    
                # Prediction
                pred = self.fc(last_output)  # (batch, output_dim)
                predictions.append(pred)
    
            # (batch, nodes, output_dim)
            predictions = torch.stack(predictions, dim=1)
    
            return predictions
    
    # Usage example
    if __name__ == "__main__":
        # Dummy traffic network
        num_nodes = 50  # Number of intersections
        node_features = 4  # Features (speed, density, flow, etc.)
        num_timesteps = 12  # Past 12 hours
    
        # Dummy data
        batch_size = 8
        x_seq = torch.randn(batch_size, num_timesteps, num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
    
        # Model
        st_gnn = SpatialTemporalGNN(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=64,
            output_dim=1,  # Traffic volume prediction
            num_timesteps=num_timesteps
        )
    
        # Prediction
        predictions = st_gnn(x_seq, edge_index)
        print(f"Predictions shape: {predictions.shape}")  # (batch, nodes, 1)
    

* * *

## Exercises

**Exercise 1: Improving Node Classification**

**Task** : Improve the NodeClassificationGCN model to achieve higher accuracy.

**Improvement Ideas** :

  1. **Architecture changes** : Deeper layers, Residual connections, add Attention mechanism
  2. **Enhanced regularization** : Adjust Dropout, L2 regularization, DropEdge
  3. **Hyperparameter optimization** : Adjust learning rate, hidden layer dimension, number of layers
  4. **Data augmentation** : Node feature normalization, graph augmentation techniques

**Evaluation criteria** : Aim for test accuracy of 85% or higher on Cora dataset

**Exercise 2: Extending Molecular Property Prediction**

**Task** : Extend the MolecularGCN model to implement multi-task learning that simultaneously predicts multiple molecular properties.

**Specifications** :

  * Multiple output heads (toxicity, solubility, activity, etc.)
  * Shared representation learning between tasks
  * Task-specific loss weighting
  * Evaluation metrics for each task (AUC, RMSE)

**Hint** : Consider a configuration of shared GCN layers + task-specific MLP classifiers

**Exercise 3: Cold-Start Problem in Recommendation Systems**

**Task** : Implement solutions for the cold-start problem (new users/items) in link prediction-based recommendation systems.

**Implementation items** :

  1. **Utilize side information** : Use user/item attribute information as features
  2. **Inductive learning** : Support new nodes with GraphSAGE, etc.
  3. **Integrate content-based filtering** : Hybrid of GNN recommendation and content-based
  4. **Meta-learning** : Quickly adapt to new users with few-shot learning

**Evaluation** : Measure recommendation accuracy for new users/items

**Exercise 4: Interpretability of DTI Prediction**

**Task** : Add interpretability features to the DTIPredictor model.

**Implementation content** :

  * **Attention visualization** : Identify important atoms/substructures
  * **Apply GradCAM** : Display important regions for prediction as heatmap
  * **Substructure analysis** : Extract molecular fragments contributing to interaction
  * **Amino acid importance** : Identify binding sites

**Output** : Visualize importance scores on molecular graph

**Exercise 5: Implementing Spatial-Temporal Graph Prediction**

**Task** : Implement traffic volume prediction using spatiotemporal GNN with actual traffic datasets (e.g., METR-LA).

**Requirements** :

  1. **Data preprocessing** : Normalize time series data, handle missing values
  2. **Model building** : Hybrid of GCN + LSTM/GRU/Transformer
  3. **Multi-step prediction** : Predict 1 hour, 3 hours, and 6 hours ahead
  4. **Evaluation** : Evaluate performance with MAE, RMSE, MAPE

**Advanced task** : Implement dynamic graph structure learning using attention mechanism

* * *

## Summary

In this chapter, we learned about practical applications of graph neural networks:

  * **Node classification** : Efficient node label prediction using semi-supervised learning
  * **Graph classification** : Classification of entire graphs such as molecular property prediction
  * **Link prediction** : Prediction of edge existence probability for recommendation systems
  * **Drug discovery applications** : Drug discovery support through drug-target interaction prediction
  * **Other applications** : Diverse domains including knowledge graphs and traffic networks

GNNs are powerful tools for handling structured data and can be applied to various real-world problems. Understanding the characteristics of each task and selecting appropriate architectures and learning strategies is the key to success.
