---
title: "Chapter 2: Graph Convolutional Networks (GCN)"
chapter_title: "Chapter 2: Graph Convolutional Networks (GCN)"
subtitle: Complete Understanding from Spectral Graph Theory to Implementation
reading_time: 30-35 minutes
difficulty: Intermediate to Advanced
code_examples: 12
exercises: 5
---

This chapter covers Graph Convolutional Networks (GCN). You will learn mathematical formulation of GCN layers, GCN layers from scratch in PyTorch, and Build GCN models with PyTorch Geometric library.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the fundamentals of spectral graph theory (graph Laplacian, eigenvalue decomposition)
  * ✅ Explain the motivation for theoretical extension from CNNs to GCNs
  * ✅ Understand the mathematical formulation of GCN layers and the meaning of symmetric normalization
  * ✅ Implement GCN layers from scratch in PyTorch
  * ✅ Build GCN models with PyTorch Geometric library
  * ✅ Execute node classification tasks on the Cora dataset

* * *

## 2.1 Fundamentals of Spectral Graph Theory

### Graph Laplacian

The **Graph Laplacian** is a fundamental method for representing graph structure as a matrix. For a graph $G = (V, E)$, it is defined as follows:

$$ L = D - A $$ 

where:

  * $A$: Adjacency matrix
  * $D$: Degree matrix, with diagonal elements $D_{ii} = \sum_j A_{ij}$
  * $L$: Laplacian matrix

### Normalized Laplacian

In GCN, the **symmetric normalized Laplacian** is used:

$$ L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2} $$ 

There is also a random walk normalized Laplacian:

$$ L_{\text{rw}} = D^{-1} L = I - D^{-1} A $$ 
    
    
    ```mermaid
    graph LR
        A["Adjacency Matrix A"] --> L["Laplacian L = D - A"]
        D["Degree Matrix D"] --> L
        L --> Lsym["Symmetric Normalized L_sym"]
        L --> Lrw["Random Walk L_rw"]
    
        style A fill:#b3e5fc
        style D fill:#c5e1a5
        style L fill:#fff9c4
        style Lsym fill:#ffab91
    ```

### Graph Laplacian Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import numpy as np
    import torch
    import networkx as nx
    import matplotlib.pyplot as plt
    
    def compute_graph_laplacian(A):
        """
        Compute graph Laplacian
    
        Args:
            A: (N, N) adjacency matrix
    
        Returns:
            L: (N, N) Laplacian matrix
            D: (N, N) degree matrix
        """
        # Degree matrix (diagonal matrix)
        D = np.diag(A.sum(axis=1))
    
        # Laplacian L = D - A
        L = D - A
    
        return L, D
    
    
    def compute_normalized_laplacian(A, method='symmetric'):
        """
        Compute normalized Laplacian
    
        Args:
            A: (N, N) adjacency matrix
            method: 'symmetric' or 'random_walk'
    
        Returns:
            L_norm: (N, N) normalized Laplacian
        """
        # Degree matrix
        D = np.diag(A.sum(axis=1))
    
        # D^{-1/2} (for symmetric normalization)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    
        if method == 'symmetric':
            # L_sym = I - D^{-1/2} A D^{-1/2}
            L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
        elif method == 'random_walk':
            # L_rw = I - D^{-1} A
            D_inv = np.diag(1.0 / (np.diag(D) + 1e-10))
            L_norm = np.eye(len(A)) - D_inv @ A
        else:
            raise ValueError(f"Unknown method: {method}")
    
        return L_norm
    
    
    # Simple graph example
    print("=== Graph Laplacian Calculation Example ===")
    
    # Create a graph with 5 nodes
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    
    # Adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    print(f"Adjacency matrix A:\n{A}\n")
    
    # Laplacian matrix
    L, D = compute_graph_laplacian(A)
    print(f"Degree matrix D:\n{D}\n")
    print(f"Laplacian L = D - A:\n{L}\n")
    
    # Normalized Laplacian
    L_sym = compute_normalized_laplacian(A, method='symmetric')
    print(f"Symmetric normalized Laplacian L_sym:\n{L_sym}\n")
    
    # Verify properties of Laplacian
    print("=== Properties of Laplacian ===")
    print(f"Row sums of L: {L.sum(axis=1)}")
    print("→ All zeros (important property of Laplacian)")
    
    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=800, font_size=16, font_weight='bold')
    plt.title("Sample Graph (5 nodes)")
    plt.savefig('graph_example.png', dpi=150, bbox_inches='tight')
    print("\nGraph saved: graph_example.png")
    plt.close()
    

### Eigenvalue Decomposition and Spectrum

The **eigenvalue decomposition** of the Laplacian matrix represents the frequency components of the graph:

$$ L = U \Lambda U^T $$ 

where:

  * $U$: Matrix of eigenvectors (graph Fourier basis)
  * $\Lambda$: Diagonal matrix of eigenvalues (corresponding to frequencies)

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def spectral_analysis(L):
        """
        Spectral analysis of Laplacian
    
        Args:
            L: (N, N) Laplacian matrix
    
        Returns:
            eigenvalues: (N,) eigenvalues (in ascending order)
            eigenvectors: (N, N) eigenvectors
        """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    
        return eigenvalues, eigenvectors
    
    
    print("\n=== Spectral Analysis ===")
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = spectral_analysis(L)
    
    print(f"Eigenvalues (ascending order):\n{eigenvalues}\n")
    print(f"Eigenvectors (first 3):\n{eigenvectors[:, :3]}\n")
    
    # Visualize eigenvalues
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Laplacian Eigenvalues (Spectrum)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(eigenvectors[:, 0], 'o-', label='1st eigenvector', markersize=8)
    plt.plot(eigenvectors[:, 1], 's-', label='2nd eigenvector', markersize=8)
    plt.plot(eigenvectors[:, 2], '^-', label='3rd eigenvector', markersize=8)
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title('First 3 Eigenvectors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplacian_spectrum.png', dpi=150, bbox_inches='tight')
    print("Spectrum saved: laplacian_spectrum.png")
    plt.close()
    
    # Important properties
    print("=== Important Properties of Laplacian ===")
    print(f"Minimum eigenvalue: {eigenvalues[0]:.6f}")
    print("→ Always 0 for connected graphs")
    print(f"Eigenvector of minimum eigenvalue: {eigenvectors[:, 0]}")
    print("→ Constant vector (all same values)")
    

### Spectral Convolution

Convolution on graphs is defined in the **Fourier domain** :

$$ x \star_G g = U \left( (U^T g) \odot (U^T x) \right) $$ 

where:

  * $x$: Node feature vector
  * $g$: Filter (defined in frequency domain)
  * $\odot$: Element-wise product (Hadamard product)

> "Spectral convolution processes signals on graphs in the frequency domain using graph Fourier transform"

* * *

## 2.2 Extension from CNN to GCN

### Convolution Operation in CNN

In conventional CNNs (Convolutional Neural Networks), convolution is performed on the **grid structure of images** :

$$ h_i^{(\ell+1)} = \sigma \left( W^{(\ell)} \sum_{j \in \mathcal{N}(i)} h_j^{(\ell)} + b^{(\ell)} \right) $$ 

where $\mathcal{N}(i)$ is the neighborhood of pixel $i$ (typically a 3×3 kernel).

### Challenges in Generalizing to Graphs

Challenge | Images (Grid) | Graphs  
---|---|---  
**Neighborhood Size** | Fixed (e.g., 3×3) | Varies per node  
**Order** | Fixed (up, down, left, right) | No definition  
**Distance** | Euclidean distance | Distance on graph  
**Symmetry** | Translation symmetry | Permutation symmetry  
      
    
    ```mermaid
    graph TB
        CNN["CNNGrid Structure"] --> Challenge["ChallengesIrregular Graph Structure"]
        Challenge --> Spectral["Approach 1Spectral Method"]
        Challenge --> Spatial["Approach 2Spatial Method"]
        Spectral --> GCN["GCNFirst-order Approximation"]
    
        style CNN fill:#b3e5fc
        style Challenge fill:#fff9c4
        style Spectral fill:#c5e1a5
        style GCN fill:#ffab91
    ```

### Basic Idea of GCN

**Graph Convolutional Networks (GCN)** achieve efficient graph convolution by using a **first-order approximation** of spectral convolution:

$$ H^{(\ell+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(\ell)} W^{(\ell)} \right) $$ 

where:

  * $\tilde{A} = A + I$: Adjacency matrix with self-loops added
  * $\tilde{D}$: Degree matrix of $\tilde{A}$
  * $H^{(\ell)}$: Node feature matrix at layer $\ell$
  * $W^{(\ell)}$: Learnable weight matrix
  * $\sigma$: Activation function

### Adding Self-Loops and Its Significance
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import numpy as np
    import torch
    
    def add_self_loops(A):
        """
        Add self-loops to adjacency matrix
    
        Args:
            A: (N, N) adjacency matrix
    
        Returns:
            A_tilde: (N, N) adjacency matrix with self-loops
        """
        # Add identity matrix
        A_tilde = A + np.eye(len(A))
        return A_tilde
    
    
    print("\n=== Adding Self-Loops ===")
    
    # Original adjacency matrix
    print(f"Original adjacency matrix A:\n{A}\n")
    
    # Add self-loops
    A_tilde = add_self_loops(A)
    print(f"With self-loops Ã = A + I:\n{A_tilde}\n")
    
    print("=== Significance of Self-Loops ===")
    print("1. Retain node's own features")
    print("2. Allow nodes with degree 0 to have information")
    print("3. Improve numerical stability")
    

### Derivation of Symmetric Normalization

Symmetric normalization **normalizes the influence of node degree** :

$$ \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} $$ 

This results in:

  1. Normalizing contributions from high-degree nodes
  2. Symmetric matrix, numerically stable
  3. Eigenvalues confined to the range [-1, 1]

    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def symmetric_normalization(A):
        """
        Compute symmetric normalized adjacency matrix
    
        Args:
            A: (N, N) adjacency matrix
    
        Returns:
            A_hat: (N, N) normalized adjacency matrix
        """
        # Add self-loops
        A_tilde = A + np.eye(len(A))
    
        # Degree matrix
        D_tilde = np.diag(A_tilde.sum(axis=1))
    
        # D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    
        # Â = D^{-1/2} Ã D^{-1/2}
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
        return A_hat
    
    
    print("\n=== Symmetric Normalization ===")
    
    A_hat = symmetric_normalization(A)
    print(f"Normalized adjacency matrix Â:\n{A_hat}\n")
    
    # Check row sums
    print(f"Row sums of Â:\n{A_hat.sum(axis=1)}\n")
    print("→ Not necessarily 1, but well balanced")
    
    # Check symmetry
    is_symmetric = np.allclose(A_hat, A_hat.T)
    print(f"Â is symmetric: {is_symmetric}")
    

* * *

## 2.3 Mathematical Formulation of GCN Layers

### Definition of a Single GCN Layer

A single GCN layer performs the following operation:

$$ H^{(\ell+1)} = \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} \right) $$ 

where:

  * $H^{(\ell)} \in \mathbb{R}^{N \times d_\ell}$: Node features at layer $\ell$ ($N$ is the number of nodes, $d_\ell$ is feature dimension)
  * $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$: Learnable weight matrix
  * $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$: Normalized adjacency matrix
  * $\sigma$: Activation function (ReLU, Tanh, etc.)

### Computation Flow
    
    
    ```mermaid
    graph LR
        H0["H^(ℓ)(N × d_ℓ)"] --> Mult1["H^(ℓ) W^(ℓ)"]
        W["W^(ℓ)(d_ℓ × d_ℓ+1)"] --> Mult1
        Mult1 --> Aggregate["Â × (H^(ℓ) W^(ℓ))"]
        Ahat["Â(N × N)"] --> Aggregate
        Aggregate --> Activation["σ(...)"]
        Activation --> H1["H^(ℓ+1)(N × d_ℓ+1)"]
    
        style H0 fill:#b3e5fc
        style W fill:#c5e1a5
        style Aggregate fill:#fff59d
        style H1 fill:#ffab91
    ```

### Node-Level Interpretation

For each node $i$:

$$ h_i^{(\ell+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \\{i\\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} h_j^{(\ell)} W^{(\ell)} \right) $$ 

This means computing a **weighted sum of neighboring node features**.

### Adding Bias Term

In practice, a bias term is also added:

$$ H^{(\ell+1)} = \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} + b^{(\ell)} \right) $$ 

### Multi-Layer GCN Formulation

An $L$-layer GCN is defined recursively:

$$ \begin{align} H^{(0)} &= X \quad \text{(input features)} \\\ H^{(\ell+1)} &= \sigma \left( \hat{A} H^{(\ell)} W^{(\ell)} \right), \quad \ell = 0, 1, \ldots, L-1 \\\ Z &= H^{(L)} \quad \text{(output)} \end{align} $$ 
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    def gcn_layer_forward(A_hat, H, W, bias=None, activation=None):
        """
        Forward pass of GCN layer
    
        Args:
            A_hat: (N, N) normalized adjacency matrix
            H: (N, d_in) node features
            W: (d_in, d_out) weight matrix
            bias: (d_out,) bias (optional)
            activation: activation function (optional)
    
        Returns:
            H_next: (N, d_out) features of next layer
        """
        # 1. Feature transformation: H @ W
        H_transformed = H @ W
    
        # 2. Neighborhood aggregation: Â @ (H @ W)
        H_aggregated = A_hat @ H_transformed
    
        # 3. Add bias
        if bias is not None:
            H_aggregated = H_aggregated + bias
    
        # 4. Activation
        if activation is not None:
            H_next = activation(H_aggregated)
        else:
            H_next = H_aggregated
    
        return H_next
    
    
    # Numerical example
    print("\n=== Forward Pass of GCN Layer ===")
    
    N = 5  # Number of nodes
    d_in = 4  # Input feature dimension
    d_out = 8  # Output feature dimension
    
    # Dummy data
    A_hat_torch = torch.FloatTensor(A_hat)
    H = torch.randn(N, d_in)
    W = torch.randn(d_in, d_out)
    b = torch.randn(d_out)
    
    print(f"Input features H: {H.shape}")
    print(f"Weights W: {W.shape}")
    print(f"Normalized adjacency matrix Â: {A_hat_torch.shape}")
    
    # GCN layer computation
    H_next = gcn_layer_forward(A_hat_torch, H, W, bias=b, activation=torch.relu)
    
    print(f"Output features H^(ℓ+1): {H_next.shape}")
    print(f"\nSample output (node 0 features):\n{H_next[0]}")
    

* * *

## 2.4 GCN Layer Implementation in PyTorch

### Implementing GCNLayer Class
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GCNLayer(nn.Module):
        """
        Single GCN layer
        """
        def __init__(self, in_features, out_features, bias=True):
            """
            Args:
                in_features: input feature dimension
                out_features: output feature dimension
                bias: whether to use bias
            """
            super(GCNLayer, self).__init__()
    
            # Weight matrix
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
    
            # Bias
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
    
            # Initialize parameters
            self.reset_parameters()
    
        def reset_parameters(self):
            """Xavier initialization"""
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
        def forward(self, x, adj):
            """
            Forward pass
    
            Args:
                x: (N, in_features) node features
                adj: (N, N) normalized adjacency matrix
    
            Returns:
                output: (N, out_features) output features
            """
            # Feature transformation: X @ W
            support = torch.mm(x, self.weight)
    
            # Neighborhood aggregation: Â @ (X @ W)
            output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
    
            # Add bias
            if self.bias is not None:
                output = output + self.bias
    
            return output
    
    
    # Verify operation
    print("\n=== Verifying GCNLayer Class ===")
    
    N = 10  # Number of nodes
    in_features = 16
    out_features = 32
    
    # Create GCN layer
    gcn_layer = GCNLayer(in_features, out_features)
    
    # Dummy data
    x = torch.randn(N, in_features)
    adj = torch.randn(N, N)
    # Symmetric normalization (simplified)
    adj = (adj + adj.T) / 2
    adj = adj / adj.sum(dim=1, keepdim=True)
    
    # Forward
    output = gcn_layer(x, adj)
    
    print(f"Input: {x.shape}")
    print(f"Adjacency matrix: {adj.shape}")
    print(f"Output: {output.shape}")
    
    # Number of parameters
    num_params = sum(p.numel() for p in gcn_layer.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(f"  Weights: {gcn_layer.weight.numel():,}")
    print(f"  Bias: {gcn_layer.bias.numel() if gcn_layer.bias is not None else 0}")
    

### Implementing Complete GCN Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class GCN(nn.Module):
        """
        Multi-layer Graph Convolutional Network
        """
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
            """
            Args:
                input_dim: input feature dimension
                hidden_dim: hidden layer dimension
                output_dim: output dimension (number of classes)
                num_layers: number of GCN layers
                dropout: dropout rate
            """
            super(GCN, self).__init__()
    
            self.num_layers = num_layers
            self.dropout = dropout
    
            # List of GCN layers
            self.gcn_layers = nn.ModuleList()
    
            # First layer
            self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
    
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
    
            # Last layer
            if num_layers > 1:
                self.gcn_layers.append(GCNLayer(hidden_dim, output_dim))
            else:
                # For single layer case
                self.gcn_layers[0] = GCNLayer(input_dim, output_dim)
    
        def forward(self, x, adj):
            """
            Forward pass
    
            Args:
                x: (N, input_dim) node features
                adj: (N, N) normalized adjacency matrix
    
            Returns:
                output: (N, output_dim) output (logits)
            """
            h = x
    
            # Intermediate layers
            for i in range(self.num_layers - 1):
                h = self.gcn_layers[i](h, adj)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
    
            # Last layer (no activation)
            output = self.gcn_layers[-1](h, adj)
    
            return output
    
    
    # Create model
    print("\n=== Creating GCN Model ===")
    
    input_dim = 1433  # Feature dimension of Cora dataset
    hidden_dim = 16
    output_dim = 7  # Number of classes
    num_layers = 2
    
    model = GCN(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=0.5)
    
    # Dummy data
    N = 100
    x = torch.randn(N, input_dim)
    adj = torch.randn(N, N)
    adj = (adj + adj.T) / 2
    adj = adj / adj.sum(dim=1, keepdim=True)
    
    # Forward
    output = model(x, adj)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Parameters per layer
    for i, layer in enumerate(model.gcn_layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  GCN layer {i+1}: {layer_params:,}")
    

### Preprocessing Normalized Adjacency Matrix
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - scipy>=1.11.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import scipy.sparse as sp
    import numpy as np
    
    def preprocess_adjacency(adj):
        """
        Symmetric normalization of adjacency matrix
    
        Args:
            adj: (N, N) adjacency matrix (NumPy array or SciPy sparse)
    
        Returns:
            adj_normalized: (N, N) normalized adjacency matrix (Tensor)
        """
        # Convert to NumPy array
        if sp.issparse(adj):
            adj = adj.toarray()
    
        # Add self-loops
        adj_tilde = adj + np.eye(adj.shape[0])
    
        # Degree matrix
        degree = np.array(adj_tilde.sum(1))
    
        # D^{-1/2}
        degree_inv_sqrt = np.power(degree, -0.5).flatten()
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(degree_inv_sqrt)
    
        # Â = D^{-1/2} Ã D^{-1/2}
        if sp.issparse(adj):
            adj_normalized = D_inv_sqrt @ sp.csr_matrix(adj_tilde) @ D_inv_sqrt
            adj_normalized = torch.FloatTensor(adj_normalized.toarray())
        else:
            adj_normalized = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
            adj_normalized = torch.FloatTensor(adj_normalized)
    
        return adj_normalized
    
    
    # Usage example
    print("\n=== Adjacency Matrix Preprocessing ===")
    
    # Sample adjacency matrix
    N = 5
    adj_raw = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float32)
    
    print(f"Original adjacency matrix:\n{adj_raw}\n")
    
    # Normalization
    adj_norm = preprocess_adjacency(adj_raw)
    
    print(f"Normalized adjacency matrix:\n{adj_norm}\n")
    
    # Verify properties
    print("=== Effect of Normalization ===")
    print(f"Maximum value: {adj_norm.max().item():.4f}")
    print(f"Minimum value: {adj_norm.min().item():.4f}")
    print(f"Diagonal elements (self-loops): {adj_norm.diag()}")
    

* * *

## 2.5 Using PyTorch Geometric

### What is PyTorch Geometric

**PyTorch Geometric (PyG)** is a powerful library for graph neural networks. It provides the following features:

  * Efficient graph data structures (edge lists in COO format)
  * Rich GNN layers (GCN, GAT, GraphSAGE, etc.)
  * Graph datasets (Cora, PubMed, CiteSeer, etc.)
  * Mini-batch processing support

### GCN Implementation with PyTorch Geometric
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    
    class PyGGCN(torch.nn.Module):
        """
        Model using PyTorch Geometric's GCNConv
        """
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
            super(PyGGCN, self).__init__()
    
            # GCN layers
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
    
            self.dropout = dropout
    
        def forward(self, x, edge_index):
            """
            Args:
                x: (N, input_dim) node features
                edge_index: (2, E) edge index (COO format)
    
            Returns:
                output: (N, output_dim) output
            """
            # First layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
            # Second layer
            x = self.conv2(x, edge_index)
    
            return x
    
    
    # Create model
    print("\n=== PyTorch Geometric GCN ===")
    
    input_dim = 1433
    hidden_dim = 16
    output_dim = 7
    
    pyg_model = PyGGCN(input_dim, hidden_dim, output_dim)
    
    # Dummy data (PyG format)
    N = 100
    E = 200
    
    x = torch.randn(N, input_dim)
    edge_index = torch.randint(0, N, (2, E))  # (2, E)
    
    # Forward
    output = pyg_model(x, edge_index)
    
    print(f"Input: {x.shape}")
    print(f"Edge index: {edge_index.shape}")
    print(f"Output: {output.shape}")
    
    # Number of parameters
    total_params = sum(p.numel() for p in pyg_model.parameters())
    print(f"Total parameters: {total_params:,}")
    

### Building Graph Data
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    from torch_geometric.data import Data
    import networkx as nx
    
    def networkx_to_pyg(G, node_features=None, labels=None):
        """
        Convert NetworkX graph to PyTorch Geometric format
    
        Args:
            G: NetworkX graph
            node_features: (N, d) node features (optional)
            labels: (N,) node labels (optional)
    
        Returns:
            data: PyTorch Geometric Data object
        """
        # Edge index (COO format)
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
        # For undirected graphs, add reverse edges
        if not G.is_directed():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
        # Node features
        if node_features is None:
            # Dummy features (identity matrix)
            x = torch.eye(G.number_of_nodes())
        else:
            x = torch.FloatTensor(node_features)
    
        # Labels
        y = torch.LongTensor(labels) if labels is not None else None
    
        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
    
        return data
    
    
    # Example: simple graph
    print("\n=== NetworkX → PyTorch Geometric ===")
    
    G = nx.karate_club_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to PyG format
    data = networkx_to_pyg(G)
    
    print(f"\nPyG data:")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Number of nodes: {data.num_nodes}")
    

* * *

## 2.6 Practice: Node Classification on Cora Dataset

### Overview of Cora Dataset

The **Cora dataset** is a citation network of papers:

  * **Nodes** : 2,708 papers
  * **Edges** : 5,429 citation relationships
  * **Features** : 1,433-dimensional bag-of-words features (word presence/absence)
  * **Classes** : 7 research areas (Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)

### Loading and Visualizing Dataset
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Loading and Visualizing Dataset
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from torch_geometric.datasets import Planetoid
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Load Cora dataset
    print("=== Loading Cora Dataset ===")
    
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"\nGraph information:")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Node feature dimension: {data.num_node_features}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"\nData split:")
    print(f"  Training nodes: {data.train_mask.sum().item()}")
    print(f"  Validation nodes: {data.val_mask.sum().item()}")
    print(f"  Test nodes: {data.test_mask.sum().item()}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for i in range(dataset.num_classes):
        count = (data.y == i).sum().item()
        print(f"  Class {i}: {count} nodes")
    
    # Check part of data
    print(f"\nFirst 5 nodes' features (non-zero elements only):")
    for i in range(5):
        nonzero = data.x[i].nonzero().squeeze()
        print(f"  Node {i}: {len(nonzero)} words, label={data.y[i].item()}")
    

### Training GCN Model
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    
    class CoraGCN(torch.nn.Module):
        """GCN model for Cora"""
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            super(CoraGCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
            self.dropout = dropout
    
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    
    def train(model, data, optimizer):
        """Training for one epoch"""
        model.train()
        optimizer.zero_grad()
    
        # Forward
        out = model(data.x, data.edge_index)
    
        # Calculate loss only on training nodes
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
        # Backward
        loss.backward()
        optimizer.step()
    
        return loss.item()
    
    
    def evaluate(model, data, mask):
        """Evaluation"""
        model.eval()
    
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
    
            # Calculate accuracy
            correct = (pred[mask] == data.y[mask]).sum().item()
            accuracy = correct / mask.sum().item()
    
        return accuracy
    
    
    # Create model
    print("\n=== Training GCN Model ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CoraGCN(
        num_features=dataset.num_node_features,
        hidden_dim=16,
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)
    
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    num_epochs = 200
    best_val_acc = 0
    best_test_acc = 0
    
    print("\nEpoch | Loss   | Train Acc | Val Acc  | Test Acc")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, optimizer)
    
        if epoch % 10 == 0:
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)
    
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
            print(f"{epoch:7d} | {loss:.4f} | {train_acc:.4f}   | {val_acc:.4f}   | {test_acc:.4f}")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Corresponding test accuracy: {best_test_acc:.4f}")
    

### Visualizing Learning Curves
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def train_with_history(model, data, optimizer, num_epochs=200):
        """Training with history recording"""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }
    
        for epoch in range(1, num_epochs + 1):
            # Training
            loss = train(model, data, optimizer)
    
            # Evaluation
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            test_acc = evaluate(model, data, data.test_mask)
    
            history['train_loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
    
        return history
    
    
    # Train with new model
    print("\n=== Recording Learning Curves ===")
    
    model_new = CoraGCN(
        num_features=dataset.num_node_features,
        hidden_dim=16,
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)
    
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.01, weight_decay=5e-4)
    
    history = train_with_history(model_new, data, optimizer_new, num_epochs=200)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].plot(history['test_acc'], label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cora_training_curves.png', dpi=150, bbox_inches='tight')
    print("Learning curves saved: cora_training_curves.png")
    plt.close()
    
    print(f"\nFinal accuracy:")
    print(f"  Training: {history['train_acc'][-1]:.4f}")
    print(f"  Validation: {history['val_acc'][-1]:.4f}")
    print(f"  Test: {history['test_acc'][-1]:.4f}")
    

### Visualizing Node Embeddings
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    def visualize_embeddings(model, data, layer='layer1'):
        """Visualize node embeddings with t-SNE"""
        model.eval()
    
        with torch.no_grad():
            # Get output of first layer
            x = model.conv1(data.x, data.edge_index)
            x = F.relu(x)
    
            if layer == 'layer2':
                x = model.conv2(x, data.edge_index)
    
            embeddings = x.cpu().numpy()
    
        # t-SNE
        print(f"\n=== t-SNE Embedding ({embeddings.shape[1]}D → 2D) ===")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    
        # Visualization
        plt.figure(figsize=(12, 10))
    
        # Color by class
        labels = data.y.cpu().numpy()
        for i in range(dataset.num_classes):
            mask = labels == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       label=f'Class {i}', alpha=0.6, s=30)
    
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f'Node Embeddings Visualization ({layer})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        filename = f'cora_embeddings_{layer}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Embeddings saved: {filename}")
        plt.close()
    
    
    # Visualize embeddings from first and second layers
    visualize_embeddings(model_new, data, layer='layer1')
    visualize_embeddings(model_new, data, layer='layer2')
    
    print("\n→ Verify that nodes of the same class are placed close together")
    

* * *

## Exercises

**Exercise 1: Investigating Effect of Number of GCN Layers**

Train GCNs with different numbers of layers (1, 2, 3, 4 layers) on the Cora dataset and compare performance. Can you observe the over-smoothing problem?
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Train GCNs with different numbers of layers (1, 2, 3, 4 laye
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    
    # Exercise: Train GCN models with different numbers of layers
    # Exercise: Plot test accuracy
    # Exercise: Analyze why performance degrades with more layers
    # Hint: Too many layers cause node representations to become similar (over-smoothing)
    

**Exercise 2: Optimizing Dropout Rate**

Try different dropout rates (0.0, 0.2, 0.5, 0.7, 0.9) and investigate changes in training and validation accuracy.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Try different dropout rates (0.0, 0.2, 0.5, 0.7, 0.9) and in
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    
    # Exercise: Train models with different dropout rates
    # Exercise: Create graph of training accuracy vs validation accuracy
    # Expectation: Moderate dropout prevents overfitting
    

**Exercise 3: Effect of Hidden Layer Dimension**

Try different hidden layer dimensions (4, 8, 16, 32, 64, 128) and investigate the tradeoff between performance and computation time.
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Try different hidden layer dimensions (4, 8, 16, 32, 64, 128
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import time
    
    # Exercise: Train models with different hidden dimensions
    # Exercise: Record test accuracy and time per epoch
    # Exercise: Create graph of performance vs computational cost
    # Analysis: Larger dimensions have higher representation power but are prone to overfitting
    

**Exercise 4: Comparing Normalization Methods**

Train GCNs with three methods: symmetric normalization, random walk normalization, and no normalization, and compare performance.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Train GCNs with three methods: symmetric normalization, rand
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import numpy as np
    
    # Exercise: Implement 3 normalization methods
    # Exercise: Train GCN with each
    # Exercise: Compare test accuracy
    # Expectation: Symmetric normalization is most stable and high-performing
    

**Exercise 5: Experiments on Other Datasets**

Train GCN on the CiteSeer or PubMed dataset and analyze differences from Cora.
    
    
    from torch_geometric.datasets import Planetoid
    
    # Exercise: Load CiteSeer or PubMed dataset
    # Exercise: Train with same GCN model
    # Exercise: Analyze performance differences between datasets
    # Exercise: Investigate differences in graph structure (density, degree distribution, etc.)
    

* * *

## Summary

In this chapter, we learned the theory and implementation of Graph Convolutional Networks (GCN).

### Key Points

  * **Graph Laplacian** : Represent graph structure as a matrix, spectral analysis via eigenvalue decomposition
  * **Spectral Convolution** : Graph signal processing in Fourier domain
  * **GCN Motivation** : Extending CNN convolution to graph structures
  * **Symmetric Normalization** : Normalize degree influence, improve numerical stability
  * **GCN Layer** : Aggregate neighboring node features to learn new representations
  * **Implementation** : From-scratch implementation in PyTorch and using PyTorch Geometric
  * **Node Classification** : Achieved high-accuracy paper classification on Cora dataset
  * **Over-smoothing** : Problem where node representations become similar when layers are too deep

### Advantages and Limitations of GCN

Item | Advantages | Limitations  
---|---|---  
**Computational Efficiency** | Linear complexity $O(|E|)$ | Memory constraints on large graphs  
**Representation Power** | Leverages graph structure | Over-smoothing problem  
**Generalizability** | Applicable to various graph tasks | Not suitable for dynamic graphs  
**Interpretability** | Intuitive understanding of neighborhood aggregation | No attention mechanism  
  
### Next Chapter

In the next chapter, we will learn about **Graph Attention Networks (GAT)** , covering attention mechanisms for more flexible neighborhood aggregation, multi-head attention for richer representations, and comparisons with GCN to understand the improvements in GNN representation power.
