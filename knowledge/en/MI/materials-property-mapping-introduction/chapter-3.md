---
title: "Chapter 3: Materials Representation Learning with GNN"
chapter_title: "Chapter 3: Materials Representation Learning with GNN"
subtitle: 
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 20
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Materials Representation Learning with GNN

This chapter covers Materials Representation Learning with GNN. You will learn essential concepts and techniques.

## Overview

Graph Neural Networks (GNN) are powerful machine learning models that can naturally handle data with atomic connections, such as crystal structures. In this chapter, you will learn how to extract high-dimensional feature representations (embeddings) from crystal structures of materials and combine them with dimensionality reduction techniques to map material space.

### Learning Objectives

  * Understand how to represent materials as graphs
  * Implement GNN models using PyTorch Geometric
  * Understand the characteristics and implementation of CGCNN, MEGNet, and SchNet
  * Visualize embeddings obtained from GNN

## 3.1 Graph Representation of Materials

### 3.1.1 Correspondence between Crystal Structures and Graphs

Crystal structures can be naturally represented as graphs:

  * **Nodes (vertices)** : Atoms
  * **Node features** : Atomic number, electronegativity, ionic radius, etc.
  * **Edges (links)** : Bonds between atoms (adjacency within a certain distance)
  * **Edge features** : Interatomic distance, bond angle, etc.
  * **Global features** : Cell volume, density, etc.

### Code Example 1: PyTorch Geometric Installation and Basic Setup
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Code Example 1: PyTorch Geometric Installation and Basic Set
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    # PyTorch Geometric installation (first time only)
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
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # PyTorch Geometric version check
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    

**Example Output** :
    
    
    Using device: cpu
    PyTorch Geometric version: 2.3.1
    PyTorch version: 2.0.1
    

### Code Example 2: Converting Crystal Structures to Graph Data
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    import torch
    from torch_geometric.data import Data
    
    def structure_to_graph(structure, cutoff=5.0):
        """
        Convert pymatgen Structure to PyTorch Geometric Data object
    
        Parameters:
        -----------
        structure : pymatgen.Structure
            Crystal structure
        cutoff : float
            Distance threshold for edge generation (Angstrom)
    
        Returns:
        --------
        data : torch_geometric.data.Data
            Graph data
        """
        # Node features: atomic numbers (one-hot encoding implemented later)
        atomic_numbers = torch.tensor([site.specie.Z for site in structure],
                                      dtype=torch.float).view(-1, 1)
    
        # Edge construction
        all_neighbors = structure.get_all_neighbors(cutoff)
        edge_index = []
        edge_attr = []
    
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                distance = neighbor.nn_distance
    
                edge_index.append([i, j])
                edge_attr.append(distance)
    
        # Handle case with no edges
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    
        # Create Data object
        data = Data(x=atomic_numbers,
                    edge_index=edge_index,
                    edge_attr=edge_attr)
    
        return data
    
    
    # Create sample crystal structure (simple CsCl structure)
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    # Convert to graph
    graph_data = structure_to_graph(structure, cutoff=5.0)
    
    print("Graph data information:")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Node feature shape: {graph_data.x.shape}")
    print(f"Edge index shape: {graph_data.edge_index.shape}")
    print(f"Edge attribute shape: {graph_data.edge_attr.shape}")
    

**Example Output** :
    
    
    Graph data information:
    Number of nodes: 2
    Number of edges: 16
    Node feature shape: torch.Size([2, 1])
    Edge index shape: torch.Size([2, 16])
    Edge attribute shape: torch.Size([16, 1])
    

### Code Example 3: Encoding Atomic Features
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    class AtomFeaturizer:
        """Class for vectorizing atomic features"""
    
        def __init__(self, max_z=100):
            """
            Parameters:
            -----------
            max_z : int
                Maximum atomic number to handle
            """
            self.max_z = max_z
    
            # List of atomic properties (simplified version)
            # In practice, use libraries like mendeleev
            self.electronegativity = self._load_property('electronegativity')
            self.covalent_radius = self._load_property('covalent_radius')
            self.valence_electrons = self._load_property('valence_electrons')
    
        def _load_property(self, property_name):
            """
            Generate dummy atomic property data
            In real projects, retrieve from mendeleev library
            """
            # Dummy data (use accurate values in practice)
            np.random.seed(42)
            if property_name == 'electronegativity':
                return np.random.uniform(0.7, 4.0, self.max_z)
            elif property_name == 'covalent_radius':
                return np.random.uniform(0.3, 2.5, self.max_z)
            elif property_name == 'valence_electrons':
                return np.random.randint(1, 8, self.max_z).astype(float)
    
        def featurize(self, atomic_number):
            """
            Generate feature vector from atomic number
    
            Parameters:
            -----------
            atomic_number : int or array-like
                Atomic number
    
            Returns:
            --------
            features : torch.Tensor
                Feature vector
            """
            if isinstance(atomic_number, (int, np.integer)):
                atomic_number = [atomic_number]
    
            features = []
            for z in atomic_number:
                z_idx = int(z) - 1  # 0-indexed
    
                feat = [
                    z / self.max_z,  # Normalized atomic number
                    self.electronegativity[z_idx],
                    self.covalent_radius[z_idx],
                    self.valence_electrons[z_idx]
                ]
                features.append(feat)
    
            return torch.tensor(features, dtype=torch.float)
    
    
    # Usage example
    featurizer = AtomFeaturizer(max_z=100)
    
    # Featurize Cs (Z=55) and Cl (Z=17)
    cs_features = featurizer.featurize(55)
    cl_features = featurizer.featurize(17)
    
    print("Cs feature vector:")
    print(cs_features)
    print("\nCl feature vector:")
    print(cl_features)
    
    # Featurize entire structure
    atomic_numbers = [site.specie.Z for site in structure]
    all_features = featurizer.featurize(atomic_numbers)
    print(f"\nFeature matrix shape for entire structure: {all_features.shape}")
    

### Code Example 4: Generating Dummy Materials Dataset
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    import numpy as np
    
    class DummyMaterialsDataset(InMemoryDataset):
        """
        Dummy materials dataset for training
        In real projects, retrieve from Materials Project API
        """
    
        def __init__(self, num_materials=1000, num_atom_types=20):
            self.num_materials = num_materials
            self.num_atom_types = num_atom_types
            super().__init__(root=None)
            self.data, self.slices = self._generate_data()
    
        def _generate_data(self):
            """Generate dummy graph data"""
            data_list = []
    
            np.random.seed(42)
            torch.manual_seed(42)
    
            for i in range(self.num_materials):
                # Randomly set number of nodes (5-30 atoms)
                num_nodes = np.random.randint(5, 30)
    
                # Node features (one-hot of atom types + continuous features)
                atom_types = torch.randint(0, self.num_atom_types, (num_nodes,))
                atom_features = torch.randn(num_nodes, 4)  # 4-dimensional features
    
                # Complete node features
                x = torch.cat([
                    F.one_hot(atom_types, num_classes=self.num_atom_types).float(),
                    atom_features
                ], dim=1)
    
                # Edge generation (average 5 edges per node)
                num_edges = num_nodes * 5
                edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
                # Edge features (distance, etc.)
                edge_attr = torch.rand(num_edges, 1) * 5.0  # 0-5 Angstrom
    
                # Target property (assuming band gap)
                y = torch.tensor([np.random.exponential(2.0)], dtype=torch.float)
    
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data_list.append(data)
    
            return self.collate(data_list)
    
    # Create dataset
    dataset = DummyMaterialsDataset(num_materials=1000, num_atom_types=20)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample data:")
    print(dataset[0])
    print(f"\nNode feature dimension: {dataset[0].x.shape[1]}")
    print(f"Target: {dataset[0].y.item():.3f}")
    

**Example Output** :
    
    
    Dataset size: 1000
    Sample data:
    Data(x=[18, 24], edge_index=[2, 90], edge_attr=[90, 1], y=[1])
    
    Node feature dimension: 24
    Target: 2.134
    

## 3.2 Crystal Graph Convolutional Neural Network (CGCNN)

CGCNN is a GNN model specialized for crystal structure property prediction.

### Code Example 5: CGCNN Convolution Layer
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    from torch_geometric.nn import MessagePassing
    
    class CGConv(MessagePassing):
        """
        CGCNN convolution layer
        """
    
        def __init__(self, node_dim, edge_dim, hidden_dim=128):
            super().__init__(aggr='add')  # aggregation: sum
    
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim
    
            # Network for node update
            self.node_fc = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
            # Network to combine edge and node features
            self.edge_fc = nn.Sequential(
                nn.Linear(node_dim + edge_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
            # Gate mechanism
            self.gate = nn.Sequential(
                nn.Linear(node_dim + edge_dim, hidden_dim),
                nn.Sigmoid()
            )
    
        def forward(self, x, edge_index, edge_attr):
            """
            Forward propagation
    
            Parameters:
            -----------
            x : Tensor [num_nodes, node_dim]
                Node features
            edge_index : Tensor [2, num_edges]
                Edge index
            edge_attr : Tensor [num_edges, edge_dim]
                Edge features
    
            Returns:
            --------
            out : Tensor [num_nodes, hidden_dim]
                Updated node features
            """
            # Message passing
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
            # Self-loop (residual connection-like processing)
            out = out + self.node_fc(x)
    
            return out
    
        def message(self, x_j, edge_attr):
            """
            Message function
            Compute information sent from neighboring nodes via edges
    
            Parameters:
            -----------
            x_j : Tensor [num_edges, node_dim]
                Source node features
            edge_attr : Tensor [num_edges, edge_dim]
                Edge features
    
            Returns:
            --------
            message : Tensor [num_edges, hidden_dim]
                Message
            """
            # Concatenate edge and node features
            z = torch.cat([x_j, edge_attr], dim=1)
    
            # Gating
            gate_values = self.gate(z)
            message_values = self.edge_fc(z)
    
            return gate_values * message_values
    
    
    # Test
    node_dim = 24
    edge_dim = 1
    hidden_dim = 64
    
    conv_layer = CGConv(node_dim, edge_dim, hidden_dim)
    
    # Dummy data
    x = torch.randn(10, node_dim)
    edge_index = torch.randint(0, 10, (2, 30))
    edge_attr = torch.randn(30, edge_dim)
    
    # Forward pass
    out = conv_layer(x, edge_index, edge_attr)
    
    print(f"Input node feature shape: {x.shape}")
    print(f"Output node feature shape: {out.shape}")
    

**Example Output** :
    
    
    Input node feature shape: torch.Size([10, 24])
    Output node feature shape: torch.Size([10, 64])
    

### Code Example 6: Complete CGCNN Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
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
    
            # Input embedding layer
            self.embedding = nn.Linear(node_dim, hidden_dim)
    
            # List of CGConv layers
            self.conv_layers = nn.ModuleList([
                CGConv(hidden_dim, edge_dim, hidden_dim)
                for _ in range(num_conv)
            ])
    
            # Batch Normalization
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_conv)
            ])
    
            # Fully connected layers (for property prediction)
            fc_layers = []
            for i in range(num_fc):
                if i == 0:
                    fc_layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
                else:
                    fc_layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 2))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(0.2))
    
            fc_layers.append(nn.Linear(hidden_dim // 2, 1))  # Output dimension=1 (regression task)
    
            self.fc = nn.Sequential(*fc_layers)
    
        def forward(self, data):
            """
            Forward propagation
    
            Parameters:
            -----------
            data : torch_geometric.data.Data or Batch
                Graph data
    
            Returns:
            --------
            out : Tensor [batch_size, 1]
                Prediction
            embedding : Tensor [batch_size, hidden_dim]
                Graph embedding (for visualization)
            """
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # Input embedding
            x = self.embedding(x)
    
            # Apply CGConv layers
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = F.softplus(x)
    
            # Graph-level embedding (average pooling)
            graph_embedding = global_mean_pool(x, batch)
    
            # Property prediction
            out = self.fc(graph_embedding)
    
            return out, graph_embedding
    
    
    # Model instantiation
    model = CGCNN(node_dim=24, edge_dim=1, hidden_dim=64, num_conv=3, num_fc=2)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    from torch_geometric.loader import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(dataloader))
    
    predictions, embeddings = model(batch)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    

**Example Output** :
    
    
    Total model parameters: 23,713
    
    Prediction shape: torch.Size([32, 1])
    Embedding shape: torch.Size([32, 64])
    

### Code Example 7: CGCNN Training Loop
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Code Example 7: CGCNN Training Loop
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torch.optim as optim
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Dataset split
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model, loss function, optimizer
    model = CGCNN(node_dim=24, edge_dim=1, hidden_dim=64, num_conv=3)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    train_losses = []
    test_losses = []
    
    print("Starting training...")
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
    
    print("Training complete!")
    
    # Plot training curves
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
    print("Saved training curve to cgcnn_training_curve.png")
    plt.show()
    

**Example Output** :
    
    
    Starting training...
    Epoch [10/50] - Train Loss: 2.1234, Test Loss: 2.3456
    Epoch [20/50] - Train Loss: 1.5678, Test Loss: 1.7890
    Epoch [30/50] - Train Loss: 1.2345, Test Loss: 1.4567
    Epoch [40/50] - Train Loss: 1.0123, Test Loss: 1.2345
    Epoch [50/50] - Train Loss: 0.8901, Test Loss: 1.1234
    Training complete!
    Saved training curve to cgcnn_training_curve.png
    

### Code Example 8: Extracting Embeddings from CGCNN
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import numpy as np
    from torch_geometric.loader import DataLoader
    
    def extract_embeddings(model, dataset, device='cpu'):
        """
        Extract embeddings from trained model for all data
    
        Parameters:
        -----------
        model : nn.Module
            Trained model
        dataset : Dataset
            Dataset
        device : str
            Device
    
        Returns:
        --------
        embeddings : np.ndarray
            Embedding vectors
        targets : np.ndarray
            Target values
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
    
    
    # Extract embeddings from entire dataset
    embeddings, targets = extract_embeddings(model, dataset, device=device)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"\nTarget statistics:")
    print(f"  Mean: {targets.mean():.3f}")
    print(f"  Std: {targets.std():.3f}")
    print(f"  Min: {targets.min():.3f}")
    print(f"  Max: {targets.max():.3f}")
    

**Example Output** :
    
    
    Embedding shape: (1000, 64)
    Target shape: (1000,)
    
    Target statistics:
      Mean: 2.015
      Std: 2.034
      Min: 0.012
      Max: 12.456
    

## 3.3 MEGNet (MatErials Graph Network)

MEGNet is a more flexible GNN architecture that considers global state.

### Code Example 9: MEGNet Block
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    from torch_geometric.nn import MessagePassing, global_mean_pool
    
    class MEGNetBlock(MessagePassing):
        """
        Basic MEGNet block
        Updates nodes, edges, and global state simultaneously
        """
    
        def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=64):
            super().__init__(aggr='mean')
    
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.global_dim = global_dim
            self.hidden_dim = hidden_dim
    
            # Edge update network
            self.edge_model = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim + global_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, edge_dim)
            )
    
            # Node update network
            self.node_model = nn.Sequential(
                nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, node_dim)
            )
    
            # Global state update network
            self.global_model = nn.Sequential(
                nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, global_dim)
            )
    
        def forward(self, x, edge_index, edge_attr, u, batch):
            """
            Forward propagation
    
            Parameters:
            -----------
            x : Tensor [num_nodes, node_dim]
                Node features
            edge_index : Tensor [2, num_edges]
                Edge index
            edge_attr : Tensor [num_edges, edge_dim]
                Edge features
            u : Tensor [batch_size, global_dim]
                Global state
            batch : Tensor [num_nodes]
                Batch index
    
            Returns:
            --------
            x_new : Tensor [num_nodes, node_dim]
                Updated node features
            edge_attr_new : Tensor [num_edges, edge_dim]
                Updated edge features
            u_new : Tensor [batch_size, global_dim]
                Updated global state
            """
            row, col = edge_index
    
            # 1. Edge update
            edge_input = torch.cat([
                x[row], x[col], edge_attr, u[batch[row]]
            ], dim=1)
            edge_attr_new = edge_attr + self.edge_model(edge_input)
    
            # 2. Node update (message passing)
            x_new = x + self.propagate(edge_index, x=x, edge_attr=edge_attr_new,
                                        u=u, batch=batch)
    
            # 3. Global state update
            # Average of node/edge features per graph
            node_global = global_mean_pool(x_new, batch)
            edge_global = global_mean_pool(edge_attr_new, batch[row])
    
            global_input = torch.cat([node_global, edge_global, u], dim=1)
            u_new = u + self.global_model(global_input)
    
            return x_new, edge_attr_new, u_new
    
        def message(self, x_j, edge_attr, u, batch):
            """
            Message function
    
            Parameters:
            -----------
            x_j : Tensor [num_edges, node_dim]
                Source node features
            edge_attr : Tensor [num_edges, edge_dim]
                Edge features
            u : Tensor [batch_size, global_dim]
                Global state
            batch : Tensor [num_edges]
                Batch index
    
            Returns:
            --------
            message : Tensor [num_edges, node_dim]
                Message
            """
            # If using node's own features, get from propagate arguments
            # Simplified here for brevity
            message_input = torch.cat([x_j, edge_attr, u[batch]], dim=1)
            return self.node_model(message_input) - x_j  # Residual
    
    
    # Test
    node_dim = 32
    edge_dim = 16
    global_dim = 8
    hidden_dim = 64
    
    megnet_block = MEGNetBlock(node_dim, edge_dim, global_dim, hidden_dim)
    
    # Dummy data
    x = torch.randn(10, node_dim)
    edge_index = torch.randint(0, 10, (2, 30))
    edge_attr = torch.randn(30, edge_dim)
    u = torch.randn(1, global_dim)  # One graph
    batch = torch.zeros(10, dtype=torch.long)  # All nodes belong to same graph
    
    # Forward pass
    x_new, edge_attr_new, u_new = megnet_block(x, edge_index, edge_attr, u, batch)
    
    print(f"Node features: {x.shape} -> {x_new.shape}")
    print(f"Edge features: {edge_attr.shape} -> {edge_attr_new.shape}")
    print(f"Global state: {u.shape} -> {u_new.shape}")
    

**Example Output** :
    
    
    Node features: torch.Size([10, 32]) -> torch.Size([10, 32])
    Edge features: torch.Size([30, 16]) -> torch.Size([30, 16])
    Global state: torch.Size([1, 8]) -> torch.Size([1, 8])
    

### Code Example 10: Complete MEGNet Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
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
    
            # Unify feature dimensions
            self.node_embedding = nn.Linear(node_dim, hidden_dim)
            self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
    
            # Global state initialization
            self.global_init = nn.Parameter(torch.randn(1, hidden_dim))
    
            # MEGNet blocks
            self.blocks = nn.ModuleList([
                MEGNetBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_blocks)
            ])
    
            # Output layer
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Softplus(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )
    
        def forward(self, data):
            """
            Forward propagation
    
            Parameters:
            -----------
            data : torch_geometric.data.Data or Batch
                Graph data
    
            Returns:
            --------
            out : Tensor [batch_size, 1]
                Prediction
            embedding : Tensor [batch_size, hidden_dim]
                Graph embedding
            """
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # Embedding
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)
    
            # Get batch size
            batch_size = batch.max().item() + 1
    
            # Initialize global state
            u = self.global_init.expand(batch_size, -1)
    
            # Apply MEGNet blocks
            for block in self.blocks:
                x, edge_attr, u = block(x, edge_index, edge_attr, u, batch)
    
            # Global state is final embedding
            graph_embedding = u
    
            # Output
            out = self.fc(graph_embedding)
    
            return out, graph_embedding
    
    
    # Model instantiation and test
    megnet_model = MEGNet(node_dim=24, edge_dim=1, hidden_dim=64, num_blocks=3)
    
    print(f"Total MEGNet model parameters: {sum(p.numel() for p in megnet_model.parameters()):,}")
    
    # Test
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    predictions, embeddings = megnet_model(batch)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    

## 3.4 SchNet

SchNet is a physically plausible GNN model using continuous filters.

### Code Example 11: SchNet Continuous-Filter Convolution Layer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import numpy as np
    from torch_geometric.nn import MessagePassing
    
    class GaussianSmearing(nn.Module):
        """
        Distance embedding using Gaussian basis functions
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
                Interatomic distance
    
            Returns:
            --------
            rbf : Tensor [num_edges, num_gaussians]
                Representation using Gaussian basis functions
            """
            dist = dist.view(-1, 1) - self.offset.view(1, -1)
            return torch.exp(self.coeff * torch.pow(dist, 2))
    
    
    class CFConv(MessagePassing):
        """
        Continuous-Filter Convolution (SchNet basic layer)
        """
    
        def __init__(self, node_dim, edge_dim, hidden_dim=64, num_gaussians=50):
            super().__init__(aggr='add')
    
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim
    
            # Gaussian basis functions
            self.distance_expansion = GaussianSmearing(0.0, 5.0, num_gaussians)
    
            # Filter generation network
            self.filter_network = nn.Sequential(
                nn.Linear(num_gaussians, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, node_dim * hidden_dim)
            )
    
            # Node update network
            self.node_network = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
        def forward(self, x, edge_index, edge_attr):
            """
            Forward propagation
    
            Parameters:
            -----------
            x : Tensor [num_nodes, node_dim]
                Node features
            edge_index : Tensor [2, num_edges]
                Edge index
            edge_attr : Tensor [num_edges, 1]
                Edge features (distance)
    
            Returns:
            --------
            out : Tensor [num_nodes, hidden_dim]
                Updated node features
            """
            # Node feature preprocessing
            x = self.node_network(x)
    
            # Message passing
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
            return out
    
        def message(self, x_j, edge_attr):
            """
            Message function
    
            Parameters:
            -----------
            x_j : Tensor [num_edges, node_dim]
                Source node features
            edge_attr : Tensor [num_edges, 1]
                Edge features (distance)
    
            Returns:
            --------
            message : Tensor [num_edges, hidden_dim]
                Message
            """
            # Expand distance with basis functions
            edge_features = self.distance_expansion(edge_attr)
    
            # Generate filter
            W = self.filter_network(edge_features)
            W = W.view(-1, self.node_dim, self.hidden_dim)
    
            # Apply filter
            x_j = x_j.unsqueeze(1)  # [num_edges, 1, node_dim]
            message = torch.bmm(x_j, W).squeeze(1)  # [num_edges, hidden_dim]
    
            return message
    
    
    # Test
    node_dim = 64
    edge_dim = 1
    hidden_dim = 64
    
    cfconv = CFConv(node_dim, edge_dim, hidden_dim, num_gaussians=50)
    
    # Dummy data
    x = torch.randn(10, node_dim)
    edge_index = torch.randint(0, 10, (2, 30))
    edge_attr = torch.rand(30, 1) * 5.0  # Distance
    
    # Forward pass
    out = cfconv(x, edge_index, edge_attr)
    
    print(f"Input node feature shape: {x.shape}")
    print(f"Output node feature shape: {out.shape}")
    

**Example Output** :
    
    
    Input node feature shape: torch.Size([10, 64])
    Output node feature shape: torch.Size([10, 64])
    

### Code Example 12: Complete SchNet Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
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
    
            # Embedding layer
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
    
            # Output network
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Softplus(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )
    
        def forward(self, data):
            """
            Forward propagation
    
            Parameters:
            -----------
            data : torch_geometric.data.Data or Batch
                Graph data
    
            Returns:
            --------
            out : Tensor [batch_size, 1]
                Prediction
            embedding : Tensor [batch_size, hidden_dim]
                Graph embedding
            """
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
            # Embedding
            x = self.embedding(x)
    
            # Interaction blocks
            for interaction, update in zip(self.interactions, self.updates):
                # CFConv
                v = interaction(x, edge_index, edge_attr)
    
                # Update
                x = x + update(v)
    
            # Graph-level embedding
            graph_embedding = global_add_pool(x, batch)
    
            # Output
            out = self.fc(graph_embedding)
    
            return out, graph_embedding
    
    
    # Model instantiation and test
    schnet_model = SchNet(node_dim=24, hidden_dim=64, num_filters=64,
                          num_interactions=3, num_gaussians=50)
    
    print(f"Total SchNet model parameters: {sum(p.numel() for p in schnet_model.parameters()):,}")
    
    # Test
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    predictions, embeddings = schnet_model(batch)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    

## 3.5 Embedding Visualization and Analysis

### Code Example 13: Visualizing GNN Embeddings with UMAP
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 13: Visualizing GNN Embeddings with UMAP
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract embeddings from CGCNN
    cgcnn_embeddings, cgcnn_targets = extract_embeddings(model, dataset, device=device)
    
    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    cgcnn_umap = reducer.fit_transform(cgcnn_embeddings)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Color by target value
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
    
    # Categorize by stability (dummy categorization for dummy data)
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
    print("Saved CGCNN embedding UMAP to cgcnn_embeddings_umap.png")
    plt.show()
    

### Code Example 14: Comparing Multiple Models with t-SNE
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    """
    Example: Code Example 14: Comparing Multiple Models with t-SNE
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Multiple model embeddings (only CGCNN here, but MEGNet and SchNet can be extracted similarly)
    models_dict = {
        'CGCNN': (model, cgcnn_embeddings)
    }
    
    # Dimensionality reduction with t-SNE
    tsne_results = {}
    for model_name, (_, embeddings) in models_dict.items():
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_embedding = tsne.fit_transform(embeddings)
        tsne_results[model_name] = tsne_embedding
    
    # Visualization
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
    print("Saved GNN embedding t-SNE comparison to gnn_embeddings_tsne_comparison.png")
    plt.show()
    

### Code Example 15: Clustering and Property Analysis
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 15: Clustering and Property Analysis
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.cluster import KMeans
    import pandas as pd
    import seaborn as sns
    
    # K-Means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cgcnn_embeddings)
    
    # Display clusters on UMAP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Color by cluster label
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax1.scatter(cgcnn_umap[mask, 0], cgcnn_umap[mask, 1],
                    c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Cluster centers
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
    
    # Target distribution by cluster
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
    print("Saved clustering analysis to cgcnn_clustering_analysis.png")
    plt.show()
    
    # Cluster statistics
    print("\nBand gap statistics by cluster:")
    cluster_stats = cluster_df.groupby('cluster')['band_gap'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(cluster_stats.round(3))
    

## 3.6 Interpreting Embedding Space

### Code Example 16: Principal Component Analysis of Embedding Vectors
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Code Example 16: Principal Component Analysis of Embedding V
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Analyze embedding space with PCA
    pca_embedding = PCA(n_components=10)
    cgcnn_pca = pca_embedding.fit_transform(cgcnn_embeddings)
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual variance
    ax1.bar(range(1, 11), pca_embedding.explained_variance_ratio_,
            alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
    ax1.set_title('PCA on CGCNN Embeddings: Variance Explained',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
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
    print("Saved embedding PCA analysis to cgcnn_embedding_pca.png")
    print(f"\nNumber of principal components needed to explain 95% of variance: {np.argmax(cumsum_var >= 0.95) + 1}")
    plt.show()
    

### Code Example 17: Nearest Neighbor Search in Embedding Space
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    def find_similar_materials(query_idx, embeddings, targets, k=5):
        """
        Search for similar materials in embedding space
    
        Parameters:
        -----------
        query_idx : int
            Index of query material
        embeddings : np.ndarray
            Embedding vectors
        targets : np.ndarray
            Target values
        k : int
            Number of neighbors to search
    
        Returns:
        --------
        neighbors : dict
            Information about neighboring materials
        """
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings[query_idx:query_idx+1])
    
        neighbors = {
            'query_idx': query_idx,
            'query_target': targets[query_idx],
            'neighbor_indices': indices[0, 1:],  # Exclude self
            'neighbor_targets': targets[indices[0, 1:]],
            'distances': distances[0, 1:]
        }
    
        return neighbors
    
    
    # Randomly select 5 materials for neighbor search
    np.random.seed(42)
    query_indices = np.random.choice(len(dataset), 5, replace=False)
    
    print("Nearest neighbor search results:\n")
    for query_idx in query_indices:
        neighbors = find_similar_materials(query_idx, cgcnn_embeddings,
                                           cgcnn_targets, k=5)
    
        print(f"Query material #{neighbors['query_idx']}:")
        print(f"  Target value: {neighbors['query_target']:.3f}")
        print(f"  Similar materials:")
    
        for i, (neighbor_idx, target, dist) in enumerate(zip(
            neighbors['neighbor_indices'],
            neighbors['neighbor_targets'],
            neighbors['distances']
        )):
            print(f"    {i+1}. Material #{neighbor_idx}: "
                  f"Target={target:.3f}, Distance={dist:.3f}")
        print()
    

**Example Output** :
    
    
    Nearest neighbor search results:
    
    Query material #123:
      Target value: 2.456
      Similar materials:
        1. Material #456: Target=2.389, Distance=0.145
        2. Material #789: Target=2.567, Distance=0.189
        3. Material #234: Target=2.123, Distance=0.234
        4. Material #567: Target=2.678, Distance=0.267
        5. Material #890: Target=2.345, Distance=0.289
    ...
    

### Code Example 18: Distance Distribution Analysis in Embedding Space
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - seaborn>=0.12.0
    
    """
    Example: Code Example 18: Distance Distribution Analysis in Embedding
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate pairwise distances (compute on subset)
    subset_size = 200
    subset_indices = np.random.choice(len(cgcnn_embeddings), subset_size, replace=False)
    subset_embeddings = cgcnn_embeddings[subset_indices]
    subset_targets = cgcnn_targets[subset_indices]
    
    # Euclidean and cosine distances
    euclidean_distances = pdist(subset_embeddings, metric='euclidean')
    cosine_distances = pdist(subset_embeddings, metric='cosine')
    
    # Target value differences
    target_diff = pdist(subset_targets.reshape(-1, 1), metric='euclidean')
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Euclidean distance distribution
    axes[0, 0].hist(euclidean_distances, bins=50, alpha=0.7,
                    edgecolor='black', color='steelblue')
    axes[0, 0].set_xlabel('Euclidean Distance', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Distribution of Euclidean Distances',
                          fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cosine distance distribution
    axes[0, 1].hist(cosine_distances, bins=50, alpha=0.7,
                    edgecolor='black', color='coral')
    axes[0, 1].set_xlabel('Cosine Distance', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distribution of Cosine Distances',
                          fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Embedding distance vs target difference
    axes[1, 0].scatter(euclidean_distances, target_diff,
                       alpha=0.3, s=10, color='purple')
    axes[1, 0].set_xlabel('Embedding Distance (Euclidean)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Band Gap Difference', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Embedding Distance vs Property Difference',
                          fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation coefficient
    correlation = np.corrcoef(euclidean_distances, target_diff)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=axes[1, 0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 2D density plot
    from scipy.stats import gaussian_kde
    
    # Prepare data
    x = euclidean_distances
    y = target_diff
    
    # KDE
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort (draw high-density points on top)
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
    print("Saved embedding distance analysis to cgcnn_embedding_distance_analysis.png")
    plt.show()
    

### Code Example 19: Interactive 3D Visualization (Plotly)
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    # - plotly>=5.14.0
    
    """
    Example: Code Example 19: Interactive 3D Visualization (Plotly)
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    
    # 3D UMAP
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    cgcnn_umap_3d = reducer_3d.fit_transform(cgcnn_embeddings)
    
    # Create DataFrame
    import pandas as pd
    
    df_3d = pd.DataFrame({
        'UMAP1': cgcnn_umap_3d[:, 0],
        'UMAP2': cgcnn_umap_3d[:, 1],
        'UMAP3': cgcnn_umap_3d[:, 2],
        'Band_Gap': cgcnn_targets,
        'Material_ID': [f'Material_{i}' for i in range(len(cgcnn_targets))],
        'Cluster': cluster_labels
    })
    
    # Interactive 3D plot
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
    print("Saved interactive 3D visualization to cgcnn_embeddings_3d_interactive.html")
    fig.show()
    

### Code Example 20: Quantitative Evaluation of Embedding Quality
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import numpy as np
    
    def evaluate_embedding_quality(embeddings, labels, targets):
        """
        Evaluate embedding quality using multiple metrics
    
        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding vectors
        labels : np.ndarray
            Cluster labels
        targets : np.ndarray
            Target values (continuous)
    
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        # Clustering quality metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
    
        # Correlation with target values (neighborhood preservation)
        from sklearn.neighbors import NearestNeighbors
    
        k = 10
        nbrs_embedding = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
        _, indices_emb = nbrs_embedding.kneighbors(embeddings)
    
        nbrs_target = NearestNeighbors(n_neighbors=k+1).fit(targets.reshape(-1, 1))
        _, indices_tgt = nbrs_target.kneighbors(targets.reshape(-1, 1))
    
        # Neighborhood match rate
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
    
    
    # Execute evaluation
    metrics = evaluate_embedding_quality(cgcnn_embeddings, cluster_labels, cgcnn_targets)
    
    print("CGCNN embedding quality evaluation:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better)")
    print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f} (higher is better)")
    print(f"  Neighborhood Preservation: {metrics['neighborhood_preservation']:.3f} (higher is better)")
    
    # Evaluate for multiple cluster numbers
    k_range = range(2, 11)
    silhouette_scores = []
    
    for k in k_range:
        kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = kmeans_k.fit_predict(cgcnn_embeddings)
        silhouette_scores.append(silhouette_score(cgcnn_embeddings, labels_k))
    
    # Plot
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
    print("\nSaved silhouette score analysis to cgcnn_silhouette_analysis.png")
    plt.show()
    

**Example Output** :
    
    
    CGCNN embedding quality evaluation:
      Silhouette Score: 0.342 (higher is better)
      Davies-Bouldin Score: 1.234 (lower is better)
      Calinski-Harabasz Score: 456.789 (higher is better)
      Neighborhood Preservation: 0.675 (higher is better)
    
    Saved silhouette score analysis to cgcnn_silhouette_analysis.png
    

## 3.7 Summary

In this chapter, we learned about materials representation learning with GNN:

### Major GNN Models

Model | Features | Advantages | Application Scenarios  
---|---|---|---  
**CGCNN** | Crystal structure specialized | Physical interpretability | Crystalline material property prediction  
**MEGNet** | Global state | Flexibility, expressiveness | Diverse material systems  
**SchNet** | Continuous filters | Physical plausibility | Molecular and atomic systems  
  
### Implemented Code

Code Example | Content | Main Functions  
---|---|---  
Examples 1-4 | Graph data preparation | Structure to graph conversion  
Examples 5-8 | CGCNN implementation | Model building, training, embedding extraction  
Examples 9-10 | MEGNet implementation | GNN with global state  
Examples 11-12 | SchNet implementation | Continuous-filter convolution  
Examples 13-20 | Embedding visualization and analysis | UMAP, t-SNE, clustering, evaluation  
  
### Best Practices

  1. **Data Preparation** : Appropriate cutoff distance, feature encoding
  2. **Model Selection** : Architecture suited to the task
  3. **Hyperparameters** : Tuning hidden_dim, num_layers, learning_rate
  4. **Evaluation** : Assess not only prediction performance but also embedding quality

### Outlook to Next Chapter

In Chapter 4, we will build a practical materials mapping system combining the dimensionality reduction methods (Chapter 2) and GNN representation learning (Chapter 3) that we have learned so far. We will retrieve real data from the Materials Project API and implement an end-to-end pipeline.

* * *

**Previous Chapter** : [Chapter 2: Mapping Material Space with Dimensionality Reduction Methods](<chapter-2.html>)

**Next Chapter** : [Chapter 4: Practical Application - Materials Mapping with GNN + Dimensionality Reduction](<chapter-4.html>)

**Series Home** : [Introduction to Materials Property Mapping](<index.html>)
