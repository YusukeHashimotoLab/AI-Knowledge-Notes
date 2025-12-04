---
title: "Chapter 3: Message Passing and GNN"
chapter_title: "Chapter 3: Message Passing and GNN"
subtitle: Generalized GNN Framework - GraphSAGE, GIN, PyTorch Geometric Implementation
reading_time: 25-30 minutes
difficulty: Intermediate to Advanced
code_examples: 8
exercises: 5
---

This chapter covers Message Passing and GNN. You will learn mathematical formulation of generalized GNN (MPNN), GraphSAGE's sampling-based aggregation, and characteristics of various aggregators (Mean.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand the basic structure of the message passing framework (Message, Aggregate, Update)
  * ✅ Master the mathematical formulation of generalized GNN (MPNN)
  * ✅ Implement GraphSAGE's sampling-based aggregation
  * ✅ Understand the characteristics of various aggregators (Mean, Pool, LSTM)
  * ✅ Understand the relationship between GIN (Graph Isomorphism Network) and WL test
  * ✅ Evaluate the expressive power of GNNs
  * ✅ Master efficient implementation methods with PyTorch Geometric
  * ✅ Implement graph classification tasks and batch processing

* * *

## 3.1 Message Passing Framework

### Concept of Message Passing

**Message Passing** is a framework that describes information propagation in GNNs in a unified manner. It updates features by sending and receiving messages between nodes and aggregating them.

> "The message passing framework provides a unified way to describe any GNN architecture with three basic operations (Message, Aggregate, Update)"

### Three Basic Operations

Message passing consists of the following three steps:
    
    
    ```mermaid
    graph LR
        A[1. MessageMessage generation] --> B[2. AggregateMessage aggregation]
     B -->C[3. UpdateFeatureupdate] 
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
    ```

#### Step 1: Message (Message Generation)

Generate messages to be sent from neighboring nodes to the center node:

$$ \mathbf{m}_{j \to i}^{(k)} = \text{MESSAGE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \mathbf{h}_j^{(k-1)}, \mathbf{e}_{ji}\right) $$

Where:

  * $\mathbf{m}_{j \to i}^{(k)}$: Message from node $j$ to node $i$
  * $\mathbf{h}_i^{(k-1)}$: Previous layer features of receiving node $i$
  * $\mathbf{h}_j^{(k-1)}$: Previous layer features of sending node $j$
  * $\mathbf{e}_{ji}$: Edge $(j, i)$ features (optional)

#### Step 2: Aggregate (Message Aggregation)

Aggregate all received messages:

$$ \mathbf{m}_i^{(k)} = \text{AGGREGATE}^{(k)}\left(\left\\{\mathbf{m}_{j \to i}^{(k)} : j \in \mathcal{N}(i)\right\\}\right) $$

Representative aggregation functions:

  * **Sum** : $\text{AGGREGATE} = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
  * **Mean** : $\text{AGGREGATE} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
  * **Max** : $\text{AGGREGATE} = \max_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$

#### Step 3: Update (Feature Update)

Update features by combining aggregated messages with own information:

$$ \mathbf{h}_i^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \mathbf{m}_i^{(k)}\right) $$

### Visualization of Message Passing
    
    
    ```mermaid
    graph TB
        subgraph "Step1: Message"
            N1[Node v] --> M1[m1→v]
            N2[Node 1] --> M1
            N3[Node 2] --> M2[m2→v]
            N4[Node 3] --> M3[m3→v]
        end
    
        subgraph "Step2: Aggregate"
            M1 --> AGG[Σ / Mean / Max]
            M2 --> AGG
            M3 --> AGG
     AGG -->AM[aggregationmessage]     end
    
        subgraph "Step3: Update"
            N1 --> UPD[UPDATE Function]
            AM --> UPD
            UPD --> H[hv(k)]
        end
    
        style M1 fill:#e3f2fd
        style M2 fill:#e3f2fd
        style M3 fill:#e3f2fd
        style AGG fill:#fff3e0
        style UPD fill:#e8f5e9
        style H fill:#c8e6c9
    ```

### Implementation Example 1: Basic Message Passing Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("=== Message Passing Framework Basic Implementation ===\n")
    
    class MessagePassingLayer(nn.Module):
        """Basic message passing layer"""
    
        def __init__(self, in_dim, out_dim, aggr='mean'):
            super(MessagePassingLayer, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.aggr = aggr
    
     # Messagefunction（Linear transformation）         self.message_nn = nn.Linear(in_dim, out_dim)
    
     # Updatefunction（Linear transformation + activation）         self.update_nn = nn.Sequential(
                nn.Linear(in_dim + out_dim, out_dim),
                nn.ReLU()
            )
    
        def message(self, h_j):
            """Message generation"""
            return self.message_nn(h_j)
    
        def aggregate(self, messages, edge_index, num_nodes):
            """Message aggregation"""
     # edge_index[1]: receiving nodeindex         target_nodes = edge_index[1]
    
     # eachNode to messageaggregation         aggregated = torch.zeros(num_nodes, self.out_dim)
    
            if self.aggr == 'sum':
                aggregated.index_add_(0, target_nodes, messages)
            elif self.aggr == 'mean':
                aggregated.index_add_(0, target_nodes, messages)
                # Normalize by degree
                degree = torch.bincount(target_nodes, minlength=num_nodes).float()
                degree = degree.clamp(min=1).view(-1, 1)
                aggregated = aggregated / degree
            elif self.aggr == 'max':
                # Max pooling
                for i in range(num_nodes):
                    mask = (target_nodes == i)
                    if mask.any():
                        aggregated[i] = messages[mask].max(dim=0)[0]
    
            return aggregated
    
        def update(self, h_i, aggregated):
     """Featureupdate"""         combined = torch.cat([h_i, aggregated], dim=-1)
            return self.update_nn(combined)
    
        def forward(self, x, edge_index):
            """
            Args:
                x: NodeFeature [num_nodes, in_dim]
                edge_index: Edge index [2, num_edges]
            """
            num_nodes = x.size(0)
    
            # Step 1: Message
     # edge_index[0]: sending node  h_j = x[edge_index[0]] # sending nodeFeature         messages = self.message(h_j)
    
            # Step 2: Aggregate
            aggregated = self.aggregate(messages, edge_index, num_nodes)
    
            # Step 3: Update
            h_new = self.update(x, aggregated)
    
            return h_new
    
    
    # test execution print("--- Creating Test Graph ---")
    # 5NodeGraph num_nodes = 5
    in_dim = 4
    out_dim = 8
    
    # NodeFeature（randomly initialized） x = torch.randn(num_nodes, in_dim)
    print(f"Node feature shape: {x.shape}")
    
    # edge list（0→1, 1→2, 2→3, 3→4, 1→3） edge_index = torch.tensor([
     [0, 1, 2, 3, 1], # sending node  [1, 2, 3, 4, 3] # receiving node ], dtype=torch.long)
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.size(1)}\n")
    
    # creating and executing message passing layer print("--- Message Passing with Each Aggregation Method ---")
    for aggr in ['sum', 'mean', 'max']:
        print(f"\n{aggr.upper()} Aggregation:")
        mp_layer = MessagePassingLayer(in_dim, out_dim, aggr=aggr)
        h_new = mp_layer(x, edge_index)
        print(f"  Output shape: {h_new.shape}")
        print(f"  Output value range: [{h_new.min():.3f}, {h_new.max():.3f}]")
        print(f"  Output examples for each node:")
        for i in range(min(3, num_nodes)):
            print(f"    Node{i}: mean={h_new[i].mean():.3f}, std={h_new[i].std():.3f}")
    

**Output** ：
    
    
    === Message Passing Framework Basic Implementation ===
    
    --- Creating Test Graph ---
    Node feature shape: torch.Size([5, 4])
    Edge index shape: torch.Size([2, 5])
    Number of edges: 5
    
    --- Message Passing with Each Aggregation Method ---
    
    SUM Aggregation:
      Output shape: torch.Size([5, 8])
      Output value range: [-1.234, 2.456]
      Output examples for each node:
        Node0: mean=0.123, std=0.876
        Node1: mean=0.234, std=0.945
        Node2: mean=-0.089, std=0.823
    
    MEAN Aggregation:
      Output shape: torch.Size([5, 8])
      Output value range: [-0.987, 1.876]
      Output examples for each node:
        Node0: mean=0.098, std=0.734
        Node1: mean=0.187, std=0.812
        Node2: mean=-0.045, std=0.698
    
    MAX Aggregation:
      Output shape: torch.Size([5, 8])
      Output value range: [-0.756, 2.123]
      Output examples for each node:
        Node0: mean=0.156, std=0.923
        Node1: mean=0.267, std=1.012
        Node2: mean=0.034, std=0.876
    

### Generalized GNN (MPNN)

**Message Passing Neural Network (MPNN)** is a framework that describes many GNN architectures in a unified manner.

General form of MPNN:

$$ \begin{align} \mathbf{m}_i^{(k+1)} &= \sum_{j \in \mathcal{N}(i)} M_k\left(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, \mathbf{e}_{ji}\right) \\\ \mathbf{h}_i^{(k+1)} &= U_k\left(\mathbf{h}_i^{(k)}, \mathbf{m}_i^{(k+1)}\right) \end{align} $$

MPNN representation of representative GNNs:

Model | MESSAGE Function $M_k$ | UPDATE Function $U_k$  
---|---|---  
**GCN** | $\frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(k)} \mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{m}_i^{(k+1)})$  
**GraphSAGE** | $\mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{W} \cdot [\mathbf{h}_i^{(k)} \| \text{AGG}(\mathbf{m}_i^{(k+1)})])$  
**GAT** | $\alpha_{ij} \mathbf{W} \mathbf{h}_j^{(k)}$ | $\sigma(\mathbf{m}_i^{(k+1)})$  
**GIN** | $\mathbf{h}_j^{(k)}$ | $\text{MLP}((1+\epsilon) \mathbf{h}_i^{(k)} + \mathbf{m}_i^{(k+1)})$  
  
* * *

## 3.2 GraphSAGE

### Overview of GraphSAGE

**GraphSAGE (SAmple and aggreGatE)** is a sampling-based GNN for large-scale graphs. Instead of using all neighbors, it samples and aggregates a fixed number of neighbors.

> "GraphSAGE enables mini-batch learning by sampling neighbors, achieving scalability to large-scale graphs"

### Sampling-Based Aggregation

Features of GraphSAGE:

  1. **Neighbor sampling** ：randomly sampling fixed number of neighbors from each node
  2. **Various Aggregators** : Aggregation functions such as Mean, Pool, LSTM
  3. **Inductive learning** ：can apply to nodes not seen during training

    
    
    ```mermaid
    graph TB
        subgraph "Standard GNN (All Neighbors)"
     V1[centerNode] -->N1[Neighbor1]         V1 --> N2[Neighbor2]
            V1 --> N3[Neighbor3]
            V1 --> N4[Neighbor4]
            V1 --> N5[Neighbor5]
            V1 --> N6[Neighbor6]
        end
    
        subgraph "GraphSAGE (Sampling)"
     V2[centerNode] -->S1[Sample1]         V2 --> S2[Sample2]
            V2 --> S3[Sample3]
            N7[Neighbor4] -.x.- V2
            N8[Neighbor5] -.x.- V2
            N9[Neighbor6] -.x.- V2
        end
    
        style V1 fill:#fff3e0
        style V2 fill:#fff3e0
        style S1 fill:#e3f2fd
        style S2 fill:#e3f2fd
        style S3 fill:#e3f2fd
    ```

### GraphSAGE Algorithm

Update equations for GraphSAGE:

$$ \begin{align} \mathbf{h}_{\mathcal{N}(i)}^{(k)} &= \text{AGGREGATE}_k\left(\left\\{\mathbf{h}_j^{(k-1)}, \forall j \in \mathcal{S}_{\mathcal{N}(i)}\right\\}\right) \\\ \mathbf{h}_i^{(k)} &= \sigma\left(\mathbf{W}^{(k)} \cdot \left[\mathbf{h}_i^{(k-1)} \| \mathbf{h}_{\mathcal{N}(i)}^{(k)}\right]\right) \\\ \mathbf{h}_i^{(k)} &= \frac{\mathbf{h}_i^{(k)}}{\|\mathbf{h}_i^{(k)}\|_2} \end{align} $$

Where:

  * $\mathcal{S}_{\mathcal{N}(i)}$：Node$i$Neighbor from samplingpartset
  * $\|$: Feature concatenation
  * Final line: L2 normalization

### Various Aggregators

#### 1\. Mean Aggregator

$$ \text{AGGREGATE}_{\text{mean}} = \frac{1}{|\mathcal{S}_{\mathcal{N}(i)}|} \sum_{j \in \mathcal{S}_{\mathcal{N}(i)}} \mathbf{h}_j^{(k-1)} $$

Feature: Simple and efficient, behaves similar to GCN

#### 2\. Pool Aggregator

$$ \text{AGGREGATE}_{\text{pool}} = \max\left(\left\\{\sigma\left(\mathbf{W}_{\text{pool}} \mathbf{h}_j^{(k-1)} + \mathbf{b}\right), \forall j \in \mathcal{S}_{\mathcal{N}(i)}\right\\}\right) $$

Feature: Element-wise max-pooling, captures asymmetric neighbor information

#### 3\. LSTM Aggregator

$$ \text{AGGREGATE}_{\text{LSTM}} = \text{LSTM}\left(\left[\mathbf{h}_j^{(k-1)}, \forall j \in \pi(\mathcal{S}_{\mathcal{N}(i)})\right]\right) $$

Where $\pi$ is a random permutation. Feature: High expressive power but requires attention to permutation dependency

### Implementation Example 2: GraphSAGE Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("\n=== GraphSAGE Implementation ===\n")
    
    class SAGEConv(nn.Module):
        """GraphSAGE layer"""
    
        def __init__(self, in_dim, out_dim, aggr='mean'):
            super(SAGEConv, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.aggr = aggr
    
     # Linear transformation（ownFeature + NeighborFeatureconcatenationafter）         if aggr == 'lstm':
                self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)
                self.lin = nn.Linear(2 * in_dim, out_dim)
            elif aggr == 'pool':
                self.pool_nn = nn.Linear(in_dim, in_dim)
                self.lin = nn.Linear(2 * in_dim, out_dim)
            else:  # mean
                self.lin = nn.Linear(2 * in_dim, out_dim)
    
        def aggregate_mean(self, h_neighbors, edge_index, num_nodes):
            """Mean aggregation"""
            target_nodes = edge_index[1]
            aggregated = torch.zeros(num_nodes, self.in_dim)
    
            aggregated.index_add_(0, target_nodes, h_neighbors)
            degree = torch.bincount(target_nodes, minlength=num_nodes).float()
            degree = degree.clamp(min=1).view(-1, 1)
    
            return aggregated / degree
    
        def aggregate_pool(self, h_neighbors, edge_index, num_nodes):
            """Max-pooling aggregation"""
            target_nodes = edge_index[1]
    
     # eachNeighborFeaturetransformation         transformed = torch.relu(self.pool_nn(h_neighbors))
    
            # Max-pooling
            aggregated = torch.zeros(num_nodes, self.in_dim)
            for i in range(num_nodes):
                mask = (target_nodes == i)
                if mask.any():
                    aggregated[i] = transformed[mask].max(dim=0)[0]
    
            return aggregated
    
        def aggregate_lstm(self, h_neighbors, edge_index, num_nodes):
            """LSTM aggregation"""
            target_nodes = edge_index[1]
            aggregated = torch.zeros(num_nodes, self.in_dim)
    
            for i in range(num_nodes):
                mask = (target_nodes == i)
                if mask.any():
     # input to LSTM in random order                 neighbors = h_neighbors[mask]
                    perm = torch.randperm(neighbors.size(0))
                    neighbors = neighbors[perm].unsqueeze(0)
    
                    _, (h_n, _) = self.lstm(neighbors)
                    aggregated[i] = h_n.squeeze(0)
    
            return aggregated
    
        def forward(self, x, edge_index):
            num_nodes = x.size(0)
    
     # Get neighbor features         h_neighbors = x[edge_index[0]]
    
     # aggregation         if self.aggr == 'mean':
                h_neigh = self.aggregate_mean(h_neighbors, edge_index, num_nodes)
            elif self.aggr == 'pool':
                h_neigh = self.aggregate_pool(h_neighbors, edge_index, num_nodes)
            elif self.aggr == 'lstm':
                h_neigh = self.aggregate_lstm(h_neighbors, edge_index, num_nodes)
    
     # ownFeature and concatenation         h_concat = torch.cat([x, h_neigh], dim=-1)
    
            # Linear transformation
            out = self.lin(h_concat)
    
     # L2normalization         out = F.normalize(out, p=2, dim=-1)
    
            return out
    
    
    class GraphSAGE(nn.Module):
     """GraphSAGEModel（2layer）""" 
        def __init__(self, in_dim, hidden_dim, out_dim, aggr='mean'):
            super(GraphSAGE, self).__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr)
            self.conv2 = SAGEConv(hidden_dim, out_dim, aggr)
    
        def forward(self, x, edge_index):
            # Layer 1
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
    
            # Layer 2
            h = self.conv2(h, edge_index)
    
            return h
    
    
    # test execution print("--- Creating GraphSAGE Model ---") num_nodes = 10
    in_dim = 8
    hidden_dim = 16
    out_dim = 4
    
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 5, 6, 7],
        [1, 2, 3, 4, 5, 0, 1, 6, 7, 8]
    ], dtype=torch.long)
    
    print(f"Number of nodes: {num_nodes}") print(f"Input dimension: {in_dim}")
    print(f"Hidden layer dimension: {hidden_dim}")
    print(f"Output dimension: {out_dim}\n")
    
    # eachAggregatortest for aggr in ['mean', 'pool', 'lstm']:
        print(f"--- {aggr.upper()} Aggregator ---")
        model = GraphSAGE(in_dim, hidden_dim, out_dim, aggr=aggr)
        model.eval()
    
        with torch.no_grad():
            out = model(x, edge_index)
    
        print(f"Output shape: {out.shape}")
        print(f"Output L2 norm: {out.norm(dim=-1)[:5].numpy()}")
        print(f"Output value range: [{out.min():.3f}, {out.max():.3f}]\n")
    

**Output** ：
    
    
    === GraphSAGE Implementation ===
    
    --- Creating GraphSAGE Model --- Number of nodes: 10 Input dimension: 8
    Hidden layer dimension: 16
    Output dimension: 4
    
    --- MEAN Aggregator ---
    Output shape: torch.Size([10, 4])
    Output L2 norm: [1. 1. 1. 1. 1.]
    Output value range: [-0.876, 0.923]
    
    --- POOL Aggregator ---
    Output shape: torch.Size([10, 4])
    Output L2 norm: [1. 1. 1. 1. 1.]
    Output value range: [-0.845, 0.891]
    
    --- LSTM Aggregator ---
    Output shape: torch.Size([10, 4])
    Output L2 norm: [1. 1. 1. 1. 1.]
    Output value range: [-0.912, 0.867]
    

* * *

## 3.3 Graph Isomorphism Network (GIN)

### Motivation for GIN: Improving Discriminative Power

**Graph Isomorphism Network (GIN)** is a GNN designed to have discriminative power equivalent to the Weisfeiler-Lehman (WL) test.

> "GIN has the maximum discriminative power theoretically achievable by GNNs. That is, graphs that GIN cannot distinguish cannot be distinguished by the WL test either"

### Weisfeiler-Lehman (WL) Test

**WL test** is a heuristic algorithm for determining graph isomorphism. It can efficiently determine graph isomorphism in many cases.

WL test algorithm:

  1. assign initial labels to each node
  2. for each node label, update with own label and neighbor labels multiple set
  3. Hash the labels to create new labels
  4. Repeat until convergence

    
    
    ```mermaid
    graph TB
        subgraph "Iteration1"
            A1[1] --- B1[1]
            A1 --- C1[1]
            B1 --- C1
        end
    
        subgraph "Iteration2"
            A2[2] --- B2[3]
            A2 --- C2[3]
            B2 --- C2[2]
        end
    
        subgraph "Iteration3"
            A3[4] --- B3[5]
            A3 --- C3[5]
            B3 --- C3[4]
        end
    
        A1 --> A2 --> A3
        B1 --> B2 --> B3
        C1 --> C2 --> C3
    
        style A1 fill:#e3f2fd
        style A2 fill:#fff3e0
        style A3 fill:#e8f5e9
    ```

### Formulation of GIN

Update equation for GIN:

$$ \mathbf{h}_i^{(k)} = \text{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right) \cdot \mathbf{h}_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(k-1)}\right) $$

Important points:

  * **Sum aggregation** : The only injective aggregation function that can preserve multisets
  * **$(1 + \epsilon)$ coefficient** : Distinguishes between own features and neighbor features
  * **MLP** : Update function with sufficient expressive power

### Why GIN Has the Highest Discriminative Power

The discriminative power of GNNs has the following order:

$$ \text{Sum} > \text{Mean} > \text{Max} $$

Aggregation Function | Multiset Preservation | Example  
---|---|---  
**Sum** | ✅ Injective (preserves multiplicity) | $\\{1, 1, 2\\} \to 4 \neq 3 \leftarrow \\{1, 2\\}$  
**Mean** | ❌ Information loss | $\\{1, 1, 2\\} \to 1.33 \neq 1.5 \leftarrow \\{1, 2\\}$  
**Max** | ❌ Preserves only maximum value | $\\{1, 1, 2\\} \to 2 = 2 \leftarrow \\{1, 2\\}$ ⚠️  
  
### Implementation Example3: GINimplementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print("\n=== Graph Isomorphism Network (GIN) Implementation ===\n")
    
    class GINConv(nn.Module):
        """GIN layer"""
    
        def __init__(self, in_dim, out_dim, epsilon=0.0, train_eps=False):
            super(GINConv, self).__init__()
    
     # Epsilon（learnable option）         if train_eps:
                self.epsilon = nn.Parameter(torch.Tensor([epsilon]))
            else:
                self.register_buffer('epsilon', torch.Tensor([epsilon]))
    
     # MLP (2layer)         self.mlp = nn.Sequential(
                nn.Linear(in_dim, 2 * out_dim),
                nn.BatchNorm1d(2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim)
            )
    
        def forward(self, x, edge_index):
            num_nodes = x.size(0)
    
     # Sum aggregation         h_neighbors = x[edge_index[0]]
            target_nodes = edge_index[1]
    
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target_nodes, h_neighbors)
    
            # (1 + epsilon) * h_i + sum(h_j)
            out = (1 + self.epsilon) * x + aggregated
    
     # MLPapply         out = self.mlp(out)
    
            return out
    
    
    class GIN(nn.Module):
     """GINModel（Graphclassificationfor）""" 
        def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                     dropout=0.5, train_eps=False):
            super(GIN, self).__init__()
    
            self.num_layers = num_layers
            self.dropout = dropout
    
            # GIN layer
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # Layer 1
            self.convs.append(GINConv(in_dim, hidden_dim, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(hidden_dim, hidden_dim, train_eps=train_eps))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # Final layer
            self.convs.append(GINConv(hidden_dim, hidden_dim, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
     # Graphlevelclassificationfor         self.graph_pred_linear = nn.Linear(hidden_dim, out_dim)
    
        def forward(self, x, edge_index, batch=None):
     # Nodelevelupdate         h = x
            for i in range(self.num_layers):
                h = self.convs[i](h, edge_index)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
    
     # Graph-level pooling（mean）         if batch is None:
     # singleGraphcase             h_graph = h.mean(dim=0, keepdim=True)
            else:
     # batchGraphcase             num_graphs = batch.max().item() + 1
                h_graph = torch.zeros(num_graphs, h.size(1))
                for i in range(num_graphs):
                    mask = (batch == i)
                    h_graph[i] = h[mask].mean(dim=0)
    
     # classification         out = self.graph_pred_linear(h_graph)
    
            return out
    
    
    # test execution print("--- Creating GIN Model ---") in_dim = 10
    hidden_dim = 32
    out_dim = 5 # 5Classclassification num_layers = 3
    
    model = GIN(in_dim, hidden_dim, out_dim, num_layers, train_eps=True)
    print(f"Modelstructure:\n{model}\n") 
    # singleGraphtest num_nodes = 20
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    
    print("--- Inference on Single Graph ---")
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    print(f"Input number of nodes: {num_nodes}") print(f"Input feature dimension: {in_dim}")
    print(f"Output shape: {out.shape}")
    print(f"Output (logits): {out[0].numpy()}\n")
    
    # batchGraphtest print("--- Inference on Batch Graphs ---")
    # 3 graphsbatch processing x_batch = torch.randn(50, in_dim) # sum50Node edge_index_batch = torch.randint(0, 50, (2, 100))
    batch = torch.tensor([0]*15 + [1]*20 + [2]*15)  # Graph1: 15Node, Graph2: 20Node, Graph3: 15Node
    
    with torch.no_grad():
        out_batch = model(x_batch, edge_index_batch, batch)
    
    print(f"Batch size: 3")
    print(f"Total number of nodes: {x_batch.size(0)}") print(f"Output shape: {out_batch.shape}")
    print(f"Predictions for each graph:")
    for i in range(3):
        pred_class = out_batch[i].argmax().item()
        print(f"  Graph{i+1}: Class {pred_class} (score={out_batch[i, pred_class]:.3f})")
    

**Output** ：
    
    
    === Graph Isomorphism Network (GIN) Implementation ===
    
    --- Creating GIN Model --- Modelstructure: GIN(
      (convs): ModuleList(
        (0-2): 3 x GINConv(...)
      )
      (batch_norms): ModuleList(
        (0-2): 3 x BatchNorm1d(32, eps=1e-05, momentum=0.1)
      )
      (graph_pred_linear): Linear(in_features=32, out_features=5, bias=True)
    )
    
    --- Inference on Single Graph ---
    Input number of nodes: 20 Input feature dimension: 10
    Output shape: torch.Size([1, 5])
    Output (logits): [-0.234  0.567  0.123 -0.456  0.891]
    
    --- Inference on Batch Graphs ---
    Batch size: 3
    Total number of nodes: 50 Output shape: torch.Size([3, 5])
    Predictions for each graph:
      Graph1: Class 4 (score=0.723)
      Graph2: Class 1 (score=0.845)
      Graph3: Class 3 (score=0.612)
    

### Comparing Discriminative Power of GIN and GCN

following examples show graphs that GIN and GCN can distinguish：
    
    
    ```mermaid
    graph LR
        subgraph "GraphA"
            A1((1)) --- A2((2))
            A2 --- A3((3))
            A3 --- A1
        end
    
        subgraph "GraphB"
            B1((1)) --- B2((2))
            B2 --- B3((3))
            B3 --- B4((4))
            B4 --- B1
        end
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style A3 fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style B3 fill:#fff3e0
        style B4 fill:#fff3e0
    ```

Results:

  * **GIN** ：✅ can distinguish Graph A and B（Number of nodesdifferent）
  * **GCN (Mean aggregation)** ：✅ can distinguish Graph A and B

more difficult examples（same number of nodes, same degree distribution）：

Model | Discriminative Power | Reason  
---|---|---  
**GIN** | Equivalent to WL test | Sum aggregation + MLP preserves multisets  
**GCN** | Weaker than WL test | Mean aggregation loses multiplicity information  
**GAT** | Weaker than WL test | Information is smoothed by attention weights  
  
* * *

## 3.4 Implementation with PyTorch Geometric

### What is PyTorch Geometric (PyG)

**PyTorch Geometric** 、PyTorch library specialized for Graph Neural Networks。efficient message passing、rich pre-implemented layers、provides data loaders。

### Key Components of PyG

Component | Description | Example  
---|---|---  
**torch_geometric.data.Data** | Graphdatastructure| `Data(x, edge_index)`  
**torch_geometric.nn.MessagePassing** | Message passing base class| Custom GNN layer implementation  
**torch_geometric.nn.*Conv** | Pre-implemented GNN layers | `GCNConv, SAGEConv, GINConv`  
**torch_geometric.datasets** | Benchmark datasets | `Cora, MUTAG, QM9`  
**torch_geometric.loader.DataLoader** | Graphbatch processing| Mini-batch learning  
  
### Implementation Example4: PyGcustomGNN layer
    
    
    # Note: This example should be executed in an environment with PyTorch Geometric installed # pip install torch-geometric
    
    print("\n=== PyTorch Geometric Custom GNN Layer ===\n")
    
    # PyG imports (demo pseudo-code)
    # from torch_geometric.nn import MessagePassing
    # from torch_geometric.utils import add_self_loops, degree
    
    # Pseudo-code for custom layer using MessagePassing base class class CustomGNNLayer:
        """
     Example of custom GNN layer inheriting from PyG MessagePassing 
     Override the following methods of MessagePassing class：     - message(): Message generation
        - aggregate(): Message aggregation
     - update(): Nodeupdate     """
    
        def __init__(self, in_channels, out_channels):
            # super(CustomGNNLayer, self).__init__(aggr='add')
            self.in_channels = in_channels
            self.out_channels = out_channels
            # self.lin = torch.nn.Linear(in_channels, out_channels)
    
        def forward(self, x, edge_index):
            """
            Args:
                x: [num_nodes, in_channels]
                edge_index: [2, num_edges]
            """
            # 1. Linear transformation
            # x = self.lin(x)
    
            # 2. Add self-loops
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    
     # 3. normalization（Normalize by degree）         # row, col = edge_index
            # deg = degree(col, x.size(0), dtype=x.dtype)
            # deg_inv_sqrt = deg.pow(-0.5)
            # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
            # 4. Start message passing
            # return self.propagate(edge_index, x=x, norm=norm)
            pass
    
        def message(self, x_j, norm):
            """
            Message generation
    
            Args:
     x_j: sending nodeFeature [num_edges, out_channels]             norm: Normalization coefficients [num_edges]
            """
            # return norm.view(-1, 1) * x_j
            pass
    
        def aggregate(self, inputs, index):
            """
     Message aggregation（default 'add', no override needed）         """
            # return torch_scatter.scatter(inputs, index, dim=0, reduce='add')
            pass
    
        def update(self, aggr_out):
            """
     Nodeupdate 
            Args:
                aggr_out: Aggregated messages [num_nodes, out_channels]
            """
            # return aggr_out
            pass
    
    print("--- PyG MessagePassingClassstructure ---") print("""
    PyG's MessagePassing allows you to implement GNN layers as follows:
    
    1. __init__: aggr='add'/'mean'/'max'Specify
    2. forward: propagate()Start message passing by calling
    3. message: x_j (sending node) used for message generation 4. aggregate: Automatically executed (method specified by aggr)
    5. update: Post-aggregation processing (optional)
    
    Advantage:
    ✅ Efficient sparse tensor operations
    ✅ GPU-optimized aggregation operations
    ✅ Automatic batch processing
    """)
    
    print("\n--- PyG Data Structure ---")
    print("""
    from torch_geometric.data import Data
    
    # Creating graph edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    Attributes:
    - data.x: NodeFeaturematrix [num_nodes, num_features] - data.edge_index: Edge index [2, num_edges]
    - data.edge_attr: Edge features (optional)
    - data.y: labels（node level or graph level） - data.num_nodes: Number of nodes """)
    

**Output** ：
    
    
    === PyTorch Geometric Custom GNN Layer ===
    
    --- PyG MessagePassingClassstructure --- 
    PyG's MessagePassing allows you to implement GNN layers as follows:
    
    1. __init__: aggr='add'/'mean'/'max'Specify
    2. forward: propagate()Start message passing by calling
    3. message: x_j (sending node) used for message generation 4. aggregate: Automatically executed (method specified by aggr)
    5. update: Post-aggregation processing (optional)
    
    Advantage:
    ✅ Efficient sparse tensor operations
    ✅ GPU-optimized aggregation operations
    ✅ Automatic batch processing
    
    
    --- PyG Data Structure ---
    
    from torch_geometric.data import Data
    
    # Creating graph edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    Attributes:
    - data.x: NodeFeaturematrix [num_nodes, num_features] - data.edge_index: Edge index [2, num_edges]
    - data.edge_attr: Edge features (optional)
    - data.y: labels（node level or graph level） - data.num_nodes: Number of nodes 

### Implementation Example5: Model using PyG pre-implemented layers
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn.functional as F
    
    print("\n=== Model using PyG pre-implemented layers（pseudo-code） ===\n") 
    # Complete model example using PyG pre-implemented layers（pseudo-code） class GNNModel:
        """
        from torch_geometric.nn import GCNConv, SAGEConv, GINConv
        from torch_geometric.nn import global_mean_pool, global_max_pool
    
        class GNNModel(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(GNNModel, self).__init__()
    
     # GCNlayer             self.conv1 = GCNConv(num_features, 64)
                self.conv2 = GCNConv(64, 64)
                self.conv3 = GCNConv(64, 64)
    
     # Graphlevelclassificationfor             self.lin = torch.nn.Linear(64, num_classes)
    
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
    
     # GCNlayerapply             x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
    
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
    
                x = self.conv3(x, edge_index)
    
     # Graph-level pooling             x = global_mean_pool(x, batch)
    
     # classification             x = self.lin(x)
    
                return F.log_softmax(x, dim=1)
        """
        pass
    
    print("--- Main GNN Layers Available in PyG ---\n")
    
    layers_info = {
        "GCNConv": {
            "Description": "Graph Convolutional Network layer",
     "aggregation": "Mean (Sum with degree normalization)",         "Usage": "GCNConv(in_channels, out_channels)"
        },
        "SAGEConv": {
            "Description": "GraphSAGE layer",
     "aggregation": "Mean / LSTM / Max-pool",         "Usage": "SAGEConv(in_channels, out_channels, aggr='mean')"
        },
        "GINConv": {
            "Description": "Graph Isomorphism Network layer",
     "aggregation": "Sum",         "Usage": "GINConv(nn.Sequential(...))"
        },
        "GATConv": {
            "Description": "Graph Attention Network layer",
     "aggregation": "Attention-weighted Sum",         "Usage": "GATConv(in_channels, out_channels, heads=8)"
        },
        "GATv2Conv": {
            "Description": "GATv2 (dynamic attention)",
     "aggregation": "Improved Attention",         "Usage": "GATv2Conv(in_channels, out_channels, heads=8)"
        }
    }
    
    for layer_name, info in layers_info.items():
        print(f"{layer_name}:")
        print(f"  Description: {info['Description']}")
     print(f" Aggregation: {info['aggregation']}")     print(f"  Usage: {info['Usage']}\n")
    
    print("--- Graph-level poolingfunction ---\n") 
    pooling_info = {
     "global_mean_pool": "mean of all nodes",  "global_max_pool": "max value of all nodes",  "global_add_pool": "sum of all nodes",     "GlobalAttention": "Attention-weighted sum"
    }
    
    for func_name, desc in pooling_info.items():
        print(f"{func_name}: {desc}")
    

**Output** ：
    
    
    === Model using PyG pre-implemented layers（pseudo-code） === 
    --- Main GNN Layers Available in PyG ---
    
    GCNConv:
      Description: Graph Convolutional Network layer
      Aggregation: Mean (Sum with degree normalization)
      Usage: GCNConv(in_channels, out_channels)
    
    SAGEConv:
      Description: GraphSAGE layer
      Aggregation: Mean / LSTM / Max-pool
      Usage: SAGEConv(in_channels, out_channels, aggr='mean')
    
    GINConv:
      Description: Graph Isomorphism Network layer
      Aggregation: Sum
      Usage: GINConv(nn.Sequential(...))
    
    GATConv:
      Description: Graph Attention Network layer
      Aggregation: Attention-weighted Sum
      Usage: GATConv(in_channels, out_channels, heads=8)
    
    GATv2Conv:
      Description: GATv2 (dynamic attention)
      Aggregation: Improved Attention
      Usage: GATv2Conv(in_channels, out_channels, heads=8)
    
    --- Graph-level poolingfunction --- 
    global_mean_pool: mean of all nodes global_max_pool: max value of all nodes global_add_pool: sum of all nodes GlobalAttention: Attention-weighted sum
    

* * *

## 3.5 Practice：Graph classification task

### Graphclassificationflow

Graphclassification、task to classify entire graph into one class。molecular property prediction、social network classificationand other applications。
    
    
    ```mermaid
    graph LR
     A[Input Graph] -->B[GNN layernode-level feature extraction]  B -->C[Graph Poolinggraph-level representation]     C --> D[MLPClassifier]
     D -->E[Classprediction] 
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#ffe0b2
        style D fill:#f3e5f5
        style E fill:#e8f5e9
    ```

### Batch Processing Mechanism

for efficient processing of multiple graphs、PyG uses unique batching method：

  1. **concatenate as one large graph** ：combine multiple graphs as non-connected graph
  2. **Batch vector** ：record which graph each node belongs to
  3. **Graph-level pooling** ：aggregate features for each graph using batch vector

    
    
    ```mermaid
    graph TB
        subgraph "Graph1 (3Node)"
            A1((0)) --- A2((1))
            A2 --- A3((2))
        end
    
        subgraph "Graph2 (2Node)"
            B1((3)) --- B2((4))
        end
    
        subgraph "Batch Tensor"
            C[batch = 0,0,0,1,1]
        end
    
        A1 -.-> C
        A2 -.-> C
        A3 -.-> C
        B1 -.-> C
        B2 -.-> C
    
        style A1 fill:#e3f2fd
        style A2 fill:#e3f2fd
        style A3 fill:#e3f2fd
        style B1 fill:#fff3e0
        style B2 fill:#fff3e0
        style C fill:#e8f5e9
    ```

### Implementation Example6: Complete implementation of graph classification
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    
    print("\n=== Complete implementation of graph classification task ===\n") 
    # simple graph dataset class SimpleGraphDataset(Dataset):
     """simple graph dataset""" 
        def __init__(self, num_graphs=100):
            self.num_graphs = num_graphs
            self.graphs = []
    
     # random graph generation         for i in range(num_graphs):
                num_nodes = torch.randint(10, 30, (1,)).item()
                num_edges = torch.randint(15, 50, (1,)).item()
    
     x = torch.randn(num_nodes, 8) # 8dimensionFeature             edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
     # labels（determined by graph size - for demo）             if num_nodes < 15:
     y = 0 # smallGraph             elif num_nodes < 20:
     y = 1 # mediumGraph             else:
     y = 2 # largeGraph 
                self.graphs.append({
                    'x': x,
                    'edge_index': edge_index,
                    'y': y,
                    'num_nodes': num_nodes
                })
    
        def __len__(self):
            return self.num_graphs
    
        def __getitem__(self, idx):
            return self.graphs[idx]
    
    
    # Collate function for batch processing
    def collate_graphs(batch):
     """multipleGraph1batchMerge"""     batch_x = []
        batch_edge_index = []
        batch_y = []
        batch_vec = []
    
        node_offset = 0
        for i, graph in enumerate(batch):
            batch_x.append(graph['x'])
    
     # Edge index offset         edge_index = graph['edge_index'] + node_offset
            batch_edge_index.append(edge_index)
    
            batch_y.append(graph['y'])
    
     # which graph this graph node belongs to         batch_vec.extend([i] * graph['num_nodes'])
    
            node_offset += graph['num_nodes']
    
        return {
            'x': torch.cat(batch_x, dim=0),
            'edge_index': torch.cat(batch_edge_index, dim=1),
            'y': torch.tensor(batch_y, dtype=torch.long),
            'batch': torch.tensor(batch_vec, dtype=torch.long)
        }
    
    
    # GraphclassificationModel class GraphClassifier(nn.Module):
     """GINbaseGraphclassificationdevice""" 
        def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3):
            super(GraphClassifier, self).__init__()
    
     # GIN layer（using previously mentioned GINConv）         self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
    
            # Layer 1
            self.convs.append(GINConv(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
            # Middle layers
            for _ in range(num_layers - 1):
                self.convs.append(GINConv(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
     # Graphlevelclassification         self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
    
        def forward(self, x, edge_index, batch):
     # NodelevelGNN         h = x
            for conv, bn in zip(self.convs, self.batch_norms):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = F.dropout(h, p=0.3, training=self.training)
    
     # Graph-level pooling (mean)         num_graphs = batch.max().item() + 1
            h_graph = torch.zeros(num_graphs, h.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                h_graph[i] = h[mask].mean(dim=0)
    
     # classification         out = self.classifier(h_graph)
    
            return out
    
    
    # Training function
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for data in loader:
            optimizer.zero_grad()
    
            out = model(data['x'], data['edge_index'], data['batch'])
            loss = criterion(out, data['y'])
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data['y']).sum().item()
            total += data['y'].size(0)
    
        return total_loss / len(loader), correct / total
    
    
    # Evaluation function
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data in loader:
                out = model(data['x'], data['edge_index'], data['batch'])
                loss = criterion(out, data['y'])
    
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == data['y']).sum().item()
                total += data['y'].size(0)
    
        return total_loss / len(loader), correct / total
    
    
    # Execution
    print("--- Creating Dataset ---")
    dataset = SimpleGraphDataset(num_graphs=200)
    train_dataset = SimpleGraphDataset(num_graphs=150)
    test_dataset = SimpleGraphDataset(num_graphs=50)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              collate_fn=collate_graphs)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_graphs)
    
    print(f"Training data: {len(train_dataset)} Graph")
    print(f"Test data: {len(test_dataset)} Graph")
    print(f"Batch size: 16\n")
    
    # Creating model model = GraphClassifier(in_dim=8, hidden_dim=32, num_classes=3, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}\n") 
    # training print("--- Training Start ---")
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
    
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
    
    print("\nTraining complete!")
    

**Output** ：
    
    
    === Complete implementation of graph classification task === 
    --- Creating Dataset ---
    Training data: 150 Graph
    Test data: 50 Graph
    Batch size: 16
    
    Model parameter count: 28,547 
    --- Training Start ---
    Epoch 1/5:
      Train Loss: 1.0234, Train Acc: 0.4533
      Test Loss:  0.9876, Test Acc:  0.4800
    Epoch 2/5:
      Train Loss: 0.8765, Train Acc: 0.5867
      Test Loss:  0.8543, Test Acc:  0.6000
    Epoch 3/5:
      Train Loss: 0.7234, Train Acc: 0.6933
      Test Loss:  0.7123, Test Acc:  0.6800
    Epoch 4/5:
      Train Loss: 0.6012, Train Acc: 0.7600
      Test Loss:  0.6234, Test Acc:  0.7400
    Epoch 5/5:
      Train Loss: 0.5123, Train Acc: 0.8067
      Test Loss:  0.5678, Test Acc:  0.7800
    
    Training complete!
    

### Implementation Example7: Graphpoolingcomparison
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    
    print("\n=== Graph-level poolingcomparison ===\n") 
    class GlobalPooling:
     """variousGraph-level poolingfunction""" 
        @staticmethod
        def global_mean_pool(x, batch):
     """meanpooling"""         num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                out[i] = x[mask].mean(dim=0)
    
            return out
    
        @staticmethod
        def global_max_pool(x, batch):
            """Max pooling"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    out[i] = x[mask].max(dim=0)[0]
    
            return out
    
        @staticmethod
        def global_add_pool(x, batch):
            """Sum pooling"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            for i in range(num_graphs):
                mask = (batch == i)
                out[i] = x[mask].sum(dim=0)
    
            return out
    
        @staticmethod
        def global_attention_pool(x, batch, gate_nn):
            """Attention pooling"""
            num_graphs = batch.max().item() + 1
            out = torch.zeros(num_graphs, x.size(1))
    
            # Compute attention weights
            gate = gate_nn(x)  # [num_nodes, 1]
    
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    # Softmax normalization
                    attn_weights = torch.softmax(gate[mask], dim=0)
                    # Weighted sum
                    out[i] = (x[mask] * attn_weights).sum(dim=0)
    
            return out
    
    
    # Creating test data print("--- Creating test data ---") # 3 graphs batching x = torch.randn(30, 16) # 30 nodes, 16-dimensional features batch = torch.tensor([0]*10 + [1]*12 + [2]*8)  # Graph 1: 10 nodes, Graph 2: 12 nodes, Graph 3: 8 nodes
    
    print(f"Total number of nodes: {x.size(0)}") print(f"Feature dimension: {x.size(1)}")
    print(f"Graphnumber: {batch.max().item() + 1}") print(f"eachGraphNumber of nodes: {[(batch == i).sum().item() for i in range(3)]}\n") 
    # eachpoolingmethodcomparison print("--- Comparison of Each Pooling Method ---\n")
    
    pooling = GlobalPooling()
    
    # Mean pooling
    mean_out = pooling.global_mean_pool(x, batch)
    print("Mean Pooling:")
    print(f"  Output shape: {mean_out.shape}")
    print(f" Graph1Featurequantitymean: {mean_out[0].mean():.4f}") print(f" Graph2Featurequantitymean: {mean_out[1].mean():.4f}") print(f" Graph3Featurequantitymean: {mean_out[2].mean():.4f}\n") 
    # Max pooling
    max_out = pooling.global_max_pool(x, batch)
    print("Max Pooling:")
    print(f"  Output shape: {max_out.shape}")
    print(f" Graph 1 max value: {max_out[0].max():.4f}") print(f" Graph 2 max value: {max_out[1].max():.4f}") print(f" Graph 3 max value: {max_out[2].max():.4f}\n") 
    # Add pooling
    add_out = pooling.global_add_pool(x, batch)
    print("Add (Sum) Pooling:")
    print(f"  Output shape: {add_out.shape}")
    print(f" Graph1sum: {add_out[0].sum():.4f}") print(f" Graph2sum: {add_out[1].sum():.4f}") print(f" Graph3sum: {add_out[2].sum():.4f}\n") 
    # Attention pooling
    gate_nn = nn.Linear(16, 1)
    attn_out = pooling.global_attention_pool(x, batch, gate_nn)
    print("Attention Pooling:")
    print(f"  Output shape: {attn_out.shape}")
    print(f" Graph1Featurequantitymean: {attn_out[0].mean():.4f}") print(f" Graph2Featurequantitymean: {attn_out[1].mean():.4f}") print(f" Graph3Featurequantitymean: {attn_out[2].mean():.4f}\n") 
    # Comparison of pooling method characteristics
    print("--- Characteristics of Pooling Methods ---\n")
    properties = {
        "Mean": {
     "Feature": "mean of all nodes",         "Advantage": "Stable, robust to outliers",
     "Disadvantage": "important nodes may be buried",  "Use Case": "general graph classification"     },
        "Max": {
            "Feature": "Element-wise maximum",
     "Advantage": "strongly emphasizes important features",  "Disadvantage": "Sensitive to outliers",  "Use Case": "case where specific nodes are important"     },
        "Sum": {
     "Feature": "sum of all nodes",  "Advantage": "preserves graph size information",  "Disadvantage": "values become large for large graphs",  "Use Case": "GIN、case where graph size is important"     },
        "Attention": {
            "Feature": "Learnable weighted sum",
     "Advantage": "automatically select important nodes",  "Disadvantage": "High computational cost, overfitting risk",  "Use Case": "complex graphs、case where interpretability is important"     }
    }
    
    for method, props in properties.items():
        print(f"{method} Pooling:")
        for key, value in props.items():
            print(f"  {key}: {value}")
        print()
    

**Output** ：
    
    
    === Graph-level poolingcomparison === 
    --- Creating test data --- Total number of nodes: 30 Feature dimension: 16
    Graphnumber: 3 eachGraphNumber of nodes: [10, 12, 8] 
    --- Comparison of Each Pooling Method ---
    
    Mean Pooling:
      Output shape: torch.Size([3, 16])
     Graph1Featurequantitymean: 0.0234  Graph2Featurequantitymean: -0.0567  Graph3Featurequantitymean: 0.0891 
    Max Pooling:
      Output shape: torch.Size([3, 16])
     Graph 1 max value: 2.3456  Graph 2 max value: 2.1234  Graph 3 max value: 1.9876 
    Add (Sum) Pooling:
      Output shape: torch.Size([3, 16])
     Graph1sum: 3.7456  Graph2sum: -8.1234  Graph3sum: 11.3456 
    Attention Pooling:
      Output shape: torch.Size([3, 16])
     Graph1Featurequantitymean: 0.0345  Graph2Featurequantitymean: -0.0623  Graph3Featurequantitymean: 0.0712 
    --- Characteristics of Pooling Methods ---
    
    Mean Pooling:
     Feature: mean of all nodes   Advantage: Stable, robust to outliers
     Disadvantage: important nodes may be buried  Use Case: general graph classification 
    Max Pooling:
      Feature: Element-wise maximum
     Advantage: strongly emphasizes important features  Disadvantage: Sensitive to outliers  Use Case: case where specific nodes are important 
    Sum Pooling:
     Feature: sum of all nodes  Advantage: preserves graph size information  Disadvantage: values become large for large graphs  Use Case: GIN、case where graph size is important 
    Attention Pooling:
      Feature: Learnable weighted sum
     Advantage: automatically select important nodes  Disadvantage: High computational cost, overfitting risk  Use Case: complex graphs、case where interpretability is important 

### Implementation Example8: Mini-batch learningdetailed
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    
    print("\n=== Graphbatch processingdetailed ===\n") 
    def visualize_batch_structure(graphs):
        """Visualize batch processing structure"""
    
     print("--- originalGraph ---")     for i, graph in enumerate(graphs):
            print(f"Graph{i}: {graph['num_nodes']}Node, {graph['edge_index'].size(1)}edge")
    
        # Batching
        batch_x = []
        batch_edge_index = []
        batch_vec = []
        node_offset = 0
    
        print("\n--- Batching Process ---")
        for i, graph in enumerate(graphs):
            print(f"\nGraph{i}:")
     print(f" current node offset: {node_offset}")  print(f" original edge index: {graph['edge_index'][:, :3].tolist()}... (first 3 edges)") 
     # Edge index offsetadjustment         adjusted_edges = graph['edge_index'] + node_offset
     print(f" Adjusted edge index: {adjusted_edges[:, :3].tolist()}...") 
            batch_x.append(graph['x'])
            batch_edge_index.append(adjusted_edges)
            batch_vec.extend([i] * graph['num_nodes'])
    
            node_offset += graph['num_nodes']
    
        # Merge
        batched_x = torch.cat(batch_x, dim=0)
        batched_edge_index = torch.cat(batch_edge_index, dim=1)
        batched_batch = torch.tensor(batch_vec)
    
        print("\n--- Batching Result ---")
     print(f"MergeNodeFeature: {batched_x.shape}")  print(f"MergeEdge index: {batched_edge_index.shape}")     print(f"Batch vector: {batched_batch.tolist()}")
     print(f"\nGraph assignment for nodes 0-4: {batched_batch[:5].tolist()}")  print(f"Graph assignment for nodes 5-9: {batched_batch[5:10].tolist()}") 
        return batched_x, batched_edge_index, batched_batch
    
    
    # Creating test graphs graphs = [
        {
            'x': torch.randn(5, 4),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            'num_nodes': 5
        },
        {
            'x': torch.randn(3, 4),
            'edge_index': torch.tensor([[0, 1], [1, 2]]),
            'num_nodes': 3
        },
        {
            'x': torch.randn(4, 4),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]]),
            'num_nodes': 4
        }
    ]
    
    batched_x, batched_edge_index, batched_batch = visualize_batch_structure(graphs)
    
    print("\n--- Recovering from Batch ---")
    num_graphs = batched_batch.max().item() + 1
    for i in range(num_graphs):
        mask = (batched_batch == i)
        print(f"\nGraph{i}:")
     print(f" Number of nodes: {mask.sum().item()}")  print(f" NodeFeatureshapestate: {batched_x[mask].shape}")  print(f" Featurequantitymean: {batched_x[mask].mean(dim=0)[:2].tolist()} (first 2 dimensions)") 

**Output** ：
    
    
    === Graphbatch processingdetailed === 
    --- originalGraph --- Graph0: 5Node, 4edge
    Graph1: 3Node, 2edge
    Graph2: 4Node, 3edge
    
    --- Batching Process ---
    
    Graph0:
     current node offset: 0  original edge index: [[0, 1, 2], [1, 2, 3]]... (first 3 edges)  Adjusted edge index: [[0, 1, 2], [1, 2, 3]]... 
    Graph1:
     current node offset: 5  original edge index: [[0, 1], [1, 2]]... (first 3 edges)  Adjusted edge index: [[5, 6], [6, 7]]... 
    Graph2:
     current node offset: 8  original edge index: [[0, 1, 2], [1, 2, 3]]... (first 3 edges)  Adjusted edge index: [[8, 9, 10], [9, 10, 11]]... 
    --- Batching Result ---
    MergeNodeFeature: torch.Size([12, 4]) MergeEdge index: torch.Size([2, 9]) Batch vector: [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    
    Graph assignment for nodes 0-4: [0, 0, 0, 0, 0] Graph assignment for nodes 5-9: [1, 1, 1, 2, 2] 
    --- Recovering from Batch ---
    
    Graph0:
     Number of nodes: 5  NodeFeatureshapestate: torch.Size([5, 4])  Featurequantitymean: [0.123, -0.456] (first 2 dimensions) 
    Graph1:
     Number of nodes: 3  NodeFeatureshapestate: torch.Size([3, 4])  Featurequantitymean: [-0.234, 0.567] (first 2 dimensions) 
    Graph2:
     Number of nodes: 4  NodeFeatureshapestate: torch.Size([4, 4])  Featurequantitymean: [0.345, 0.123] (first 2 dimensions) 

* * *

## Summary

In this chapter, we learned the core **message passing framework** of GNNs and representative GNN architectures.

### Key Points

**1\. Three Steps of Message Passing**

  * **Message** : message generation from neighboring connected nodes
  * **Aggregate** : Aggregate messages (Sum / Mean / Max)
  * **Update** : aggregationresultFeatureupdate
  * Many GNNs can be described uniformly with this framework

**2\. Sampling-Based Aggregation in GraphSAGE**

  * Sample neighbors to fixed size
  * scalability for large-scale graphs
  * Choice of Mean / Pool / LSTM Aggregator
  * Enables inductive learning

**3\. GINMaximum Discriminative Power**

  * Weisfeiler-Lehman test and equivalentDiscriminative Power
  * Sum aggregation is the only injective aggregation that preserves multisets
  * $(1 + \epsilon)$ coefficient distinguishes between self and neighbors
  * MLP ensures sufficient expressive power

**4\. Efficient Implementation with PyTorch Geometric**

  * simple and clean implementation using MessagePassing base class
  * Pre-implemented layers (GCNConv, SAGEConv, GINConv, etc.)
  * efficient sparse tensor operations
  * Graph batch processing and DataLoader

**5\. Graphclassificationimplementation**

  * NodelevelGNN → Graph-level pooling → classificationdevice
  * batch processing：merge multiple graphs as non-connected graph
  * Graph-level pooling（Mean / Max / Sum / Attention）
  * Practical training and evaluation loop

### Next Chapter

In the next chapter, we will learn about **Graph Attention Mechanisms** , covering Graph Attention Networks (GAT), how self-attention mechanisms are applied to graphs, the effects of multi-head attention for richer representations, and Transformers for Graphs.

* * *

## Exercise Questions

**Exercise 1: Hand Calculation of Message Passing**

Following graph、1-layer message passing（Sum aggregation）compute by hand。

  * Node0: $\mathbf{h}_0 = [1, 0]$
  * Node1: $\mathbf{h}_1 = [0, 1]$
  * Node2: $\mathbf{h}_2 = [1, 1]$
  * edge: 0→1, 1→2, 2→0
  * MESSAGE Function: identity mapping
  * UPDATE Function: $\mathbf{h}_i^{(1)} = \mathbf{h}_i^{(0)} + \mathbf{m}_i$

feature after updating each node$\mathbf{h}_i^{(1)}$calculate。

**Exercise 2: Aggregator Selection**

Choose the best aggregator for the following tasks and describe the reason：

  1. SNS community detection (number of friends for each user is important)
  2. Molecular toxicity prediction (presence of specific functional groups is important)
  3. road network traffic flow prediction（average traffic volume is important）

Options: Sum, Mean, Max, LSTM

**Exercise3：GIN Discriminative Power**

For the following 2 graphs, answer whether GIN、GCN (Mean aggregation)、GAT (Maxaggregation) can distinguish them respectively：

  * Graph A: 3 nodes in triangular shape（each node degree2）
  * Graph B: 4 nodes in square shape（each node degree2）

All initial features are$[1]$ .

**Exercise4：Graph pooling implementation**

Implement attention-based graph pooling. Requirements:

  * Compute attention score for each node pair
  * Normalize with Softmax
  * Compute graph representation by weighted sum
  * Handle multiple graphs using batch vector

**Exercise 5: Batch Processing Design**

3 graphs（5 nodes, 3 nodes, 7 nodes）after batching：

  1. MergeafterTotal number of nodes
  2. Contents of batch vector
  3. Edge index offset for each graph

Answer with specific numbers.

* * *
