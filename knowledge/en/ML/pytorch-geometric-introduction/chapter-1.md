---
title: "Chapter 1: PyTorch Geometric Introduction and Graph Data Fundamentals"
chapter_title: "Chapter 1: PyTorch Geometric Introduction and Graph Data Fundamentals"
subtitle: First Steps with Graph-Structured Data and GNN
reading_time: 30-35 min
difficulty: Intermediate
code_examples: 12
exercises: 5
---

In this chapter, you'll learn the fundamental concepts of graph data that underpin Graph Neural Networks (GNN) and how to use the PyTorch Geometric (PyG) library. Through the basic graph structure, PyG installation, working with Data objects, built-in datasets, and implementing a simple GCN layer, you'll build a solid foundation for GNN development.

## Learning Objectives

  * ✅ Understand basic graph data concepts (nodes, edges, adjacency matrices)
  * ✅ Install PyTorch Geometric and verify it works
  * ✅ Create and manipulate PyG Data objects
  * ✅ Load and explore built-in datasets
  * ✅ Implement node classification with a simple GCN layer

## 1\. Graph Data Fundamentals

A **Graph** is a data structure composed of nodes (vertices) and edges (links). Many complex relationships in the real world can be represented as graphs.

### Basic Graph Elements

  * **Node (Node/Vertex)** : Points representing elements in a graph. Examples: people, molecular atoms, papers
  * **Edge (Edge/Link)** : Lines representing relationships between nodes. Examples: friendships, chemical bonds, citation relationships
  * **Features** : Attribute information associated with nodes or edges

    
    
    ```mermaid
    graph LR
      A[Node A] -->|Edge| B[Node B]
      B --> C[Node C]
      A --> C
      C --> D[Node D]
      B --> D
    ```

### Types of Graphs

Category | Type | Description | Examples  
---|---|---|---  
Directionality | Directed Graph | Edges have direction | Twitter follow relationships, citation networks  
| Undirected Graph | Edges have no direction | Facebook friendships, molecular structures  
Node Types | Homogeneous Graph | Single type of node | Social networks (people only)  
| Heterogeneous Graph | Multiple types of nodes | Recommendation graphs with users and products  
Weight | Weighted Graph | Edges have weights (strength) | Road networks (distance), similarity graphs  
  
### Graph Representation Methods

Main methods for representing graphs in computers:

#### 1\. Adjacency Matrix

With \\(N\\) nodes, represented as an \\(N \times N\\) matrix \\(A\\):

$$A_{ij} = \begin{cases} 1 & \text{if there is an edge from node } i \text{ to } j \\\ 0 & \text{otherwise} \end{cases}$$
    
    
    import numpy as np
    
    # Adjacency matrix for a 4-node graph
    # Edges: 0→1, 1→2, 0→2, 2→3, 1→3
    adjacency_matrix = np.array([
        [0, 1, 1, 0],  # Connections from node 0
        [0, 0, 1, 1],  # Connections from node 1
        [0, 0, 0, 1],  # Connections from node 2
        [0, 0, 0, 0]   # Connections from node 3
    ])
    
    print("Adjacency Matrix:\n", adjacency_matrix)
    

#### 2\. Edge Index

An efficient representation method used in PyTorch Geometric. Suitable for sparse graphs.
    
    
    import torch
    
    # Same graph represented as edge index
    # Shape: [2, num_edges]
    # Row 0: source nodes, Row 1: target nodes
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1],  # Source nodes
        [1, 2, 2, 3, 3]   # Target nodes
    ], dtype=torch.long)
    
    print("Edge Index:\n", edge_index)
    

**💡 Why Edge Index?**

The adjacency matrix requires \\(O(N^2)\\) memory, but real-world graphs are often sparse, so edge index only requires \\(O(E)\\) (where \\(E\\) is the number of edges). For example, for a graph with 10,000 nodes and average degree 10, the adjacency matrix requires 100MB, but edge index only needs about 800KB.

## 2\. PyTorch Geometric Installation and Environment Setup

**PyTorch Geometric** is a graph neural network library built on PyTorch.

### Installation Methods

PyTorch Geometric depends on the PyTorch and CUDA versions. First, check your environment.
    
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    

#### Method 1: Installation via pip (Recommended)
    
    
    # For PyTorch 2.0+ (CPU version)
    pip install torch-geometric
    
    # Additional dependencies
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    
    # GPU version (for CUDA 11.8)
    pip install torch-geometric
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    

#### Method 2: Installation via conda
    
    
    # For conda
    conda install pyg -c pyg
    

#### Method 3: Google Colab (No Environment Setup Required)

In Google Colab, you can easily install with the following commands:
    
    
    !pip install torch-geometric
    !pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    

### Installation Verification
    
    
    import torch
    import torch_geometric
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Verify with sample data
    from torch_geometric.data import Data
    
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    print(f"\nSample Data object created successfully!")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    

**Sample Output:**
    
    
    PyTorch version: 2.1.0
    PyTorch Geometric version: 2.4.0
    
    Sample Data object created successfully!
    Number of nodes: 3
    Number of edges: 4
    

## 3\. PyG Data Object

The central data structure in PyTorch Geometric is the **Data object**. It efficiently stores graph structure and features.

### Data Object Structure

Attribute | Shape | Description  
---|---|---  
`x` | [num_nodes, num_features] | Node feature matrix  
`edge_index` | [2, num_edges] | Edge connectivity information (COO format)  
`edge_attr` | [num_edges, num_edge_features] | Edge feature matrix (optional)  
`y` | Arbitrary | Target labels (node or graph)  
`pos` | [num_nodes, num_dimensions] | Node position coordinates (optional)  
  
### Creating Data Objects
    
    
    import torch
    from torch_geometric.data import Data
    
    # Node features (3 nodes, each with 2-dimensional features)
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=torch.float)
    
    # Edge index (4 edges)
    # 0→1, 1→0, 1→2, 2→1
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    # Edge features (each edge has 1-dimensional features)
    edge_attr = torch.tensor([[1.0], [1.0], [2.0], [2.0]], dtype=torch.float)
    
    # Node labels (for node classification tasks)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(data)
    print(f"\nNumber of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_node_features}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    

**Output:**
    
    
    Data(x=[3, 2], edge_index=[2, 4], edge_attr=[4, 1], y=[3])
    
    Number of nodes: 3
    Number of edges: 4
    Number of features: 2
    Has isolated nodes: False
    Has self-loops: False
    Is undirected: True
    

### Data Object Operations
    
    
    import torch
    from torch_geometric.data import Data
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Get features of a specific node
    print("Node 0 features:", data.x[0])
    
    # Get information of a specific edge
    print("Edge 0:", data.edge_index[:, 0])
    print("Edge 0 attribute:", data.edge_attr[0])
    
    # Transfer data to GPU
    if torch.cuda.is_available():
        data = data.to('cuda')
        print(f"Data moved to: {data.x.device}")
    
    # Move back to CPU
    data = data.to('cpu')
    
    # Validate data
    print(f"\nIs valid: {data.validate()}")
    

## 4\. Basic Data Operations and Built-in Datasets

PyTorch Geometric provides many built-in datasets for research and learning.

### Major Built-in Datasets

Dataset | Type | Nodes | Description  
---|---|---|---  
Cora | Citation Network | 2,708 | Paper citation relationships, 7-class classification  
Citeseer | Citation Network | 3,327 | Paper citation relationships, 6-class classification  
PubMed | Citation Network | 19,717 | Medical paper citations, 3-class classification  
PPI | Biological Network | 14,755 | Protein interactions, multi-label classification  
QM9 | Molecular Graph | ~130k molecules | Molecular property prediction, regression tasks  
  
### Loading the Cora Dataset
    
    
    from torch_geometric.datasets import Planetoid
    
    # Download and load Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # First graph (Cora is a single graph)
    data = dataset[0]
    
    print(f"\nGraph structure:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")
    
    # Check node features and labels
    print(f"\nNode features shape: {data.x.shape}")
    print(f"Node labels shape: {data.y.shape}")
    print(f"First node features: {data.x[0][:10]}...")
    print(f"First node label: {data.y[0].item()}")
    

**Sample Output:**
    
    
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
    

### Using DataLoader

For datasets with multiple graphs, use DataLoader for batch processing.
    
    
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    # ENZYMES dataset (protein graph classification)
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of features: {dataset.num_features}")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Check batch
    for batch in loader:
        print(f"\nBatch:")
        print(f"Number of graphs in batch: {batch.num_graphs}")
        print(f"Total nodes in batch: {batch.num_nodes}")
        print(f"Total edges in batch: {batch.num_edges}")
        print(f"Batch shape: {batch.batch.shape}")
        break  # Display first batch only
    

**💡 Batch Processing Mechanism**

PyG's DataLoader combines multiple graphs into one large graph. Which graph each node belongs to is managed by the `batch` attribute. This allows efficient batch processing of graphs of different sizes.

## 5\. Simple GNN Implementation Example

Let's implement a node classification model using the most basic graph neural network layer, **GCNConv (Graph Convolutional Network)**.

### Basic Principles of GCN

GCN updates each node's features by aggregating features from neighboring nodes:

$$\mathbf{x}_i^{(k+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \\{i\\}} \frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(k)} \mathbf{x}_j^{(k)}\right)$$

Where:

  * \\(\mathbf{x}_i^{(k)}\\): Features of node \\(i\\) at layer \\(k\\)
  * \\(\mathcal{N}(i)\\): Set of neighbors of node \\(i\\)
  * \\(d_i\\): Degree of node \\(i\\)
  * \\(\mathbf{W}^{(k)}\\): Learnable weight matrix
  * \\(\sigma\\): Activation function (ReLU, etc.)

    
    
    ```mermaid
    graph LR
      A[Node AFeatures] --> AGG[Aggregate]
      B[Neighbor BFeatures] --> AGG
      C[Neighbor CFeatures] --> AGG
      AGG --> UPDATE[Update]
      UPDATE --> A2[New Features]
    ```

### GCN Model Implementation
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class GCN(torch.nn.Module):
        def __init__(self, num_features, num_classes):
            super(GCN, self).__init__()
            # 2-layer GCN
            self.conv1 = GCNConv(num_features, 16)
            self.conv2 = GCNConv(16, num_classes)
    
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
    
            # Layer 1: input → 16 dimensions
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
    
            # Layer 2: 16 dimensions → number of classes
            x = self.conv2(x, edge_index)
    
            return F.log_softmax(x, dim=1)
    
    # Create model
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    model = GCN(num_features=dataset.num_features,
                num_classes=dataset.num_classes)
    
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    

### Training Loop Implementation
    
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import Planetoid
    
    # Prepare data and model
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=dataset.num_features,
                num_classes=dataset.num_classes).to(device)
    data = data.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
    
        # Forward pass
        out = model(data)
    
        # Compute loss (training data only)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
        # Backward pass
        loss.backward()
        optimizer.step()
    
        # Display results every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            _, pred = model(data).max(dim=1)
            correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            accuracy = correct / data.train_mask.sum().item()
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {accuracy:.4f}')
            model.train()
    

### Model Evaluation
    
    
    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            _, pred = out.max(dim=1)
    
            # Training data accuracy
            correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            train_acc = correct / data.train_mask.sum().item()
    
            # Validation data accuracy
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            val_acc = correct / data.val_mask.sum().item()
    
            # Test data accuracy
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            test_acc = correct / data.test_mask.sum().item()
    
        return train_acc, val_acc, test_acc
    
    train_acc, val_acc, test_acc = test(model, data)
    print(f'\nFinal Results:')
    print(f'Train Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    

**Sample Output:**
    
    
    Epoch 010, Loss: 1.9234, Train Acc: 0.3143
    Epoch 020, Loss: 1.7845, Train Acc: 0.4357
    Epoch 030, Loss: 1.5234, Train Acc: 0.6000
    ...
    Epoch 200, Loss: 0.5123, Train Acc: 0.9714
    
    Final Results:
    Train Accuracy: 0.9714
    Validation Accuracy: 0.7540
    Test Accuracy: 0.8130
    

**🎉 First GNN Implementation Complete!**

We achieved 81% test accuracy on the Cora dataset. This is a significant improvement over MLP models that don't consider graph structure (around 60%). GNNs achieve higher accuracy by learning relationships between nodes.

## Exercises

**Exercise 1: Creating Custom Graphs**

Create a graph with the following conditions:

  1. 5 nodes (each node with 3-dimensional features)
  2. Undirected graph (bidirectional edges)
  3. Edges: 0-1, 1-2, 2-3, 3-4, 4-0
  4. Assign random labels (0, 1, or 2) to each node

    
    
    # Write your code here
    

**Exercise 2: Dataset Exploration**

Load the Citeseer dataset and output the following information:

  * Number of nodes, number of edges
  * Average node degree
  * Number of feature dimensions
  * Number of classes
  * Number of training/validation/test nodes

**Exercise 3: 3-Layer GCN Implementation**

Extend the 2-layer GCN to implement a 3-layer GCN model. Set the intermediate layer dimensions to 32 and 16. Train on the Cora dataset and compare accuracy.

Hint: Adding more layers can make overfitting easier, so you may need to adjust Dropout.

**Exercise 4: Using Edge Features**

Create a graph with weighted edges (features) and set the `edge_attr` attribute. Use random values (range 0.1-1.0) for edge weights.

**Exercise 5: Graph Visualization**

Use NetworkX and Matplotlib to visualize the created graph. Display nodes with different colors based on labels.
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx
    
    # Convert PyG Data object to NetworkX graph
    # Write your code here
    

## Summary

In this chapter, we learned the fundamentals of graph neural networks:

  * ✅ Basic graph data concepts (nodes, edges, adjacency matrices, edge indices)
  * ✅ PyTorch Geometric installation and environment setup
  * ✅ Data object structure and operation methods
  * ✅ How to use built-in datasets (Cora, ENZYMES, etc.)
  * ✅ Node classification implementation with GCNConv layers

**🎉 Next Steps**

In the next chapter, you'll learn in detail about the message passing mechanism of Graph Convolutional Networks (GCN) and fully understand node classification tasks. You'll also learn practical overfitting prevention and hyperparameter tuning techniques.

* * *

**Reference Resources**

  * [PyTorch Geometric Official Documentation](<https://pytorch-geometric.readthedocs.io/>)
  * [PyTorch Geometric GitHub](<https://github.com/pyg-team/pytorch_geometric>)
  * [GCN Paper: Semi-Supervised Classification with Graph Convolutional Networks](<https://arxiv.org/abs/1609.02907>)
