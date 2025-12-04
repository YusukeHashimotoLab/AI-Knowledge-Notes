---
title: "Chapter 1: Fundamentals of Network Analysis"
chapter_title: "Chapter 1: Fundamentals of Network Analysis"
subtitle: From Graph Theory to Practical Network Data Analysis
reading_time: 20-25 minutes
difficulty: Beginner
code_examples: 8
version: 1.0
created_at: 2025-10-23
---

This chapter covers the fundamentals of Fundamentals of Network Analysis, which fundamentals of graph theory. You will learn fundamental concepts of graph theory (nodes, different network representation methods, and special graph structures (random.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand fundamental concepts of graph theory (nodes, edges, directed/undirected graphs)
  * ✅ Learn different network representation methods and their conversion techniques
  * ✅ Calculate and interpret basic network metrics
  * ✅ Understand special graph structures (random, small-world, scale-free)
  * ✅ Practice loading real data and performing basic analysis using NetworkX

* * *

## 1.1 Fundamentals of Graph Theory

### Graph Definition

A **graph** is a collection of **nodes (vertices)** and **edges (links)**. Mathematically, it is represented as $G = (V, E)$.

  * **Nodes ($V$)** : Components of the system (people, web pages, proteins, etc.)
  * **Edges ($E$)** : Relationships between nodes (friendships, links, interactions, etc.)

    
    
    ```mermaid
    graph LR
        A((Node A)) --- B((Node B))
        B --- C((Node C))
        C --- A
        C --- D((Node D))
    
        style A fill:#e3f2fd
        style B fill:#e3f2fd
        style C fill:#e3f2fd
        style D fill:#e3f2fd
    ```

### Directed vs Undirected Graphs

Type | Description | Examples  
---|---|---  
**Undirected Graph** | Edges have no direction (symmetric relationships) | Friendship networks, co-authorship, road networks  
**Directed Graph** | Edges have direction (asymmetric relationships) | Twitter follows, citation networks, web links  
**Weighted Graph** | Edges have weights (strength, distance, etc.) | Transportation networks, communication frequency, transaction amounts  
  
### Basic NetworkX Operations
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    
    """
    Example: Basic NetworkX Operations
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create an undirected graph
    G = nx.Graph()
    
    # Add nodes
    G.add_node(1)
    G.add_nodes_from([2, 3, 4, 5])
    
    # Add edges
    G.add_edge(1, 2)
    G.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5)])
    
    # Basic information
    print("=== Basic Graph Information ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"List of nodes: {list(G.nodes())}")
    print(f"List of edges: {list(G.edges())}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue',
            node_size=800, font_size=16, font_weight='bold')
    plt.title('Basic Undirected Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Basic Graph Information ===
    Number of nodes: 5
    Number of edges: 5
    List of nodes: [1, 2, 3, 4, 5]
    List of edges: [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    

* * *

## 1.2 Network Representations

### Three Main Representation Methods

Networks can be represented in multiple ways. Each has its own advantages and disadvantages, and they are used according to the application.

#### 1\. Adjacency Matrix

An $n \times n$ matrix $A$ where $A_{ij} = 1$ if nodes $i$ and $j$ are connected.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: An $n \times n$ matrix $A$ where $A_{ij} = 1$ if nodes $i$ a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import numpy as np
    
    # Create a graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    print("=== Adjacency Matrix ===")
    print(adj_matrix)
    
    # Visualize the matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Adjacency Matrix Visualization')
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Adjacency Matrix ===
    [[0 1 1 0]
     [1 0 1 0]
     [1 1 0 1]
     [0 0 1 0]]
    

#### 2\. Adjacency List

Maintains a list of connected nodes for each node. Efficient for sparse graphs.
    
    
    # Get adjacency list
    adj_list = dict(G.adjacency())
    
    print("=== Adjacency List ===")
    for node, neighbors in adj_list.items():
        neighbor_list = list(neighbors.keys())
        print(f"Node {node}: {neighbor_list}")
    
    # Memory efficiency comparison
    print(f"\nAdjacency matrix size: {adj_matrix.nbytes} bytes")
    print(f"Adjacency list size (estimated): {len(str(adj_list))} bytes")
    

**Output** :
    
    
    === Adjacency List ===
    Node 0: [1, 2]
    Node 1: [0, 2]
    Node 2: [0, 1, 3]
    Node 3: [2]
    
    Adjacency matrix size: 128 bytes
    Adjacency list size (estimated): 71 bytes
    

#### 3\. Edge List

Records all edges as (source, target) pairs. Convenient for data storage.
    
    
    # Create edge list
    edge_list = list(G.edges())
    
    print("=== Edge List ===")
    for i, (u, v) in enumerate(edge_list):
        print(f"Edge {i}: {u} -- {v}")
    
    # Weighted edge list
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([
        (0, 1, 2.5), (0, 2, 1.8), (1, 2, 3.2), (2, 3, 1.5)
    ])
    
    print("\n=== Weighted Edge List ===")
    for u, v, weight in G_weighted.edges(data='weight'):
        print(f"{u} -- {v}: weight = {weight}")
    

**Output** :
    
    
    === Edge List ===
    Edge 0: 0 -- 1
    Edge 1: 0 -- 2
    Edge 2: 1 -- 2
    Edge 3: 2 -- 3
    
    === Weighted Edge List ===
    0 -- 1: weight = 2.5
    0 -- 2: weight = 1.8
    1 -- 2: weight = 3.2
    2 -- 3: weight = 1.5
    

### Comparison and Conversion of Representation Methods

Representation | Memory Efficiency | Adjacency Check | Edge Traversal | Use Cases  
---|---|---|---|---  
**Adjacency Matrix** | $O(n^2)$ | $O(1)$ | $O(n^2)$ | Dense graphs, matrix operations  
**Adjacency List** | $O(n + m)$ | $O(d)$ | $O(m)$ | Sparse graphs, general analysis  
**Edge List** | $O(m)$ | $O(m)$ | $O(m)$ | Data storage, I/O operations  
  
> $n$ = number of nodes, $m$ = number of edges, $d$ = degree (average connections)

* * *

## 1.3 Basic Network Metrics

### Degree

The number of edges a node has. Represents "connectivity" within the network.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: The number of edges a node has. Represents "connectivity" wi
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sample graph
    G = nx.karate_club_graph()
    
    # Calculate degrees
    degrees = dict(G.degree())
    
    print("=== Degree Statistics ===")
    print(f"Average degree: {np.mean(list(degrees.values())):.2f}")
    print(f"Maximum degree: {max(degrees.values())}")
    print(f"Minimum degree: {min(degrees.values())}")
    
    # Visualize degree distribution
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(list(degrees.values()), bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.grid(True, alpha=0.3)
    
    # Network visualization (node size by degree)
    plt.subplot(1, 2, 2)
    node_sizes = [v * 50 for v in degrees.values()]
    nx.draw_spring(G, node_size=node_sizes, node_color='lightblue',
                   with_labels=True, font_size=8)
    plt.title('Node Size by Degree')
    
    plt.tight_layout()
    plt.show()
    

### Density

The ratio of actual edges to possible edges. Indicates network "compactness".

$$\text{Density} = \frac{2m}{n(n-1)}$$ (for undirected graphs)
    
    
    # Calculate density
    density = nx.density(G)
    
    print(f"\n=== Network Density ===")
    print(f"Density: {density:.4f}")
    print(f"Actual number of edges: {G.number_of_edges()}")
    
    n = G.number_of_nodes()
    max_edges = n * (n - 1) // 2
    print(f"Maximum possible edges: {max_edges}")
    print(f"Connection rate: {(G.number_of_edges() / max_edges) * 100:.2f}%")
    

### Diameter and Clustering Coefficient
    
    
    # Diameter (longest shortest path)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"\n=== Path Metrics ===")
        print(f"Diameter: {diameter}")
        print(f"Average shortest path length: {avg_path_length:.2f}")
    
    # Clustering coefficient (triangle density)
    clustering = nx.clustering(G)
    avg_clustering = nx.average_clustering(G)
    
    print(f"\n=== Clustering Coefficient ===")
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    print(f"Top 5 nodes by clustering coefficient:")
    sorted_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
    for node, coef in sorted_clustering[:5]:
        print(f"  Node {node}: {coef:.4f}")
    

**Example Output** :
    
    
    === Degree Statistics ===
    Average degree: 4.59
    Maximum degree: 17
    Minimum degree: 1
    
    === Network Density ===
    Density: 0.1390
    Actual number of edges: 78
    Maximum possible edges: 561
    Connection rate: 13.90%
    
    === Path Metrics ===
    Diameter: 5
    Average shortest path length: 2.41
    
    === Clustering Coefficient ===
    Average clustering coefficient: 0.5706
    Top 5 nodes by clustering coefficient:
      Node 4: 1.0000
      Node 6: 1.0000
      Node 7: 1.0000
      Node 10: 1.0000
      Node 11: 1.0000
    

* * *

## 1.4 Special Graph Structures

### Random Graphs (Erdős-Rényi Model)

A model where edges exist independently between each node pair with probability $p$.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    
    """
    Example: A model where edges exist independently between each node pa
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Erdős-Rényi random graph
    n = 30  # Number of nodes
    p = 0.1  # Edge probability
    
    G_random = nx.erdos_renyi_graph(n, p, seed=42)
    
    print("=== Erdős-Rényi Random Graph ===")
    print(f"Number of nodes: {G_random.number_of_nodes()}")
    print(f"Number of edges: {G_random.number_of_edges()}")
    print(f"Average degree: {np.mean([d for n, d in G_random.degree()]):.2f}")
    print(f"Clustering coefficient: {nx.average_clustering(G_random):.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 8))
    nx.draw_spring(G_random, node_color='lightcoral', node_size=300,
                   with_labels=True, font_size=8)
    plt.title(f'Erdős-Rényi Random Graph (n={n}, p={p})')
    plt.tight_layout()
    plt.show()
    

### Small-World Networks (Watts-Strogatz Model)

A model that achieves both high clustering coefficient and short average path length. Commonly observed in real-world networks.
    
    
    # Watts-Strogatz small-world network
    n = 30
    k = 4    # Number of neighboring nodes each node connects to
    p = 0.3  # Edge rewiring probability
    
    G_small_world = nx.watts_strogatz_graph(n, k, p, seed=42)
    
    print("\n=== Watts-Strogatz Small-World Network ===")
    print(f"Number of nodes: {G_small_world.number_of_nodes()}")
    print(f"Number of edges: {G_small_world.number_of_edges()}")
    print(f"Average degree: {np.mean([d for n, d in G_small_world.degree()]):.2f}")
    print(f"Clustering coefficient: {nx.average_clustering(G_small_world):.4f}")
    if nx.is_connected(G_small_world):
        print(f"Average shortest path length: {nx.average_shortest_path_length(G_small_world):.2f}")
    
    # Visualization
    plt.figure(figsize=(8, 8))
    nx.draw_circular(G_small_world, node_color='lightgreen', node_size=300,
                     with_labels=True, font_size=8)
    plt.title(f'Watts-Strogatz Small-World (n={n}, k={k}, p={p})')
    plt.tight_layout()
    plt.show()
    

### Scale-Free Networks (Barabási-Albert Model)

A network generated by "the rich get richer" (preferential attachment). The degree distribution follows a power law.
    
    
    # Barabási-Albert scale-free network
    n = 30
    m = 2  # Number of edges a new node connects to
    
    G_scale_free = nx.barabasi_albert_graph(n, m, seed=42)
    
    print("\n=== Barabási-Albert Scale-Free Network ===")
    print(f"Number of nodes: {G_scale_free.number_of_nodes()}")
    print(f"Number of edges: {G_scale_free.number_of_edges()}")
    print(f"Average degree: {np.mean([d for n, d in G_scale_free.degree()]):.2f}")
    print(f"Maximum degree: {max([d for n, d in G_scale_free.degree()])}")
    
    degrees = [d for n, d in G_scale_free.degree()]
    
    # Visualize degree distribution (logarithmic scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Network structure
    node_sizes = [d * 100 for d in degrees]
    nx.draw_spring(G_scale_free, ax=axes[0], node_size=node_sizes,
                   node_color='lightyellow', with_labels=True, font_size=8)
    axes[0].set_title(f'Barabási-Albert Scale-Free (n={n}, m={m})')
    
    # Degree distribution (log plot)
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    axes[1].loglog(list(degree_counts.keys()), list(degree_counts.values()),
                   'bo-', markersize=8)
    axes[1].set_xlabel('Degree (log scale)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Degree Distribution (Power Law)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Comparison of Three Models** :

Property | Random Graph | Small-World | Scale-Free  
---|---|---|---  
**Clustering** | Low | High | Moderate  
**Average Path Length** | Short | Short | Short  
**Degree Distribution** | Poisson | Nearly uniform | Power law  
**Real-World Examples** | Theoretical model | Social networks, neural networks | WWW, citation networks  
  
* * *

## 1.5 Practice: Loading Network Data and Basic Analysis

### Building Networks from CSV
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Building Networks from CSV
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create edge list CSV (sample)
    edge_data = {
        'source': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'target': ['Bob', 'Charlie', 'Charlie', 'David', 'David', 'Eve'],
        'weight': [3, 2, 5, 1, 4, 2]
    }
    df = pd.DataFrame(edge_data)
    
    print("=== Edge List Data ===")
    print(df)
    
    # Convert to NetworkX graph
    G = nx.from_pandas_edgelist(df, source='source', target='target',
                                edge_attr='weight', create_using=nx.Graph())
    
    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"List of nodes: {list(G.nodes())}")
    
    # Visualization (weighted)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Change edge thickness by weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=1000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=[w * 0.8 for w in weights],
                           alpha=0.6, edge_color='gray')
    
    # Edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title('Weighted Network Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

### Calculating Basic Statistics
    
    
    # Comprehensive network analysis
    print("\n=== Network Basic Statistics ===")
    
    # Degree statistics
    degrees = dict(G.degree())
    print(f"Average degree: {np.mean(list(degrees.values())):.2f}")
    
    # Weight statistics
    total_weight = sum([d['weight'] for u, v, d in G.edges(data=True)])
    avg_weight = total_weight / G.number_of_edges()
    print(f"Total weight: {total_weight}")
    print(f"Average edge weight: {avg_weight:.2f}")
    
    # Connectivity
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    print(f"Network density: {nx.density(G):.4f}")
    
    if nx.is_connected(G):
        print(f"Diameter: {nx.diameter(G)}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    
    # Centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    print("\n=== Centrality Rankings ===")
    print("Degree centrality (top 3):")
    for node, cent in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {node}: {cent:.4f}")
    
    print("\nBetweenness centrality (top 3):")
    for node, cent in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {node}: {cent:.4f}")
    

**Example Output** :
    
    
    === Edge List Data ===
        source   target  weight
    0    Alice      Bob       3
    1    Alice  Charlie       2
    2      Bob  Charlie       5
    3      Bob    David       1
    4  Charlie    David       4
    5    David      Eve       2
    
    Number of nodes: 5
    Number of edges: 6
    
    === Network Basic Statistics ===
    Average degree: 2.40
    Total weight: 17
    Average edge weight: 2.83
    Number of connected components: 1
    Network density: 0.6000
    Diameter: 3
    Average shortest path length: 1.60
    
    === Centrality Rankings ===
    Degree centrality (top 3):
      Bob: 0.7500
      Charlie: 0.7500
      David: 0.7500
    
    Betweenness centrality (top 3):
      Charlie: 0.4000
      Bob: 0.4000
      David: 0.2667
    

### Saving and Loading in GraphML Format
    
    
    # Save network
    output_file = 'sample_network.graphml'
    nx.write_graphml(G, output_file)
    print(f"\nNetwork saved to {output_file}")
    
    # Load saved network
    G_loaded = nx.read_graphml(output_file)
    print(f"Loading complete: {G_loaded.number_of_nodes()} nodes, {G_loaded.number_of_edges()} edges")
    
    # Other formats
    # nx.write_edgelist(G, 'network.edgelist')  # Edge list
    # nx.write_gexf(G, 'network.gexf')  # GEXF (for Gephi)
    # nx.write_pajek(G, 'network.net')  # Pajek format
    

* * *

## Chapter Summary

### What We Learned

  1. **Fundamentals of Graph Theory**

     * Network representation using nodes and edges
     * Differences between directed, undirected, and weighted graphs
     * Basic operations with NetworkX
  2. **Network Representations**

     * Characteristics of adjacency matrices, adjacency lists, and edge lists
     * Choosing representation methods according to use case
     * Trade-offs between computational complexity and memory efficiency
  3. **Basic Network Metrics**

     * Degree, density, diameter, clustering coefficient
     * Centrality metrics (degree centrality, betweenness centrality)
     * Quantitative evaluation of network characteristics
  4. **Special Graph Structures**

     * Random graphs, small-world, scale-free networks
     * Modeling real-world networks
     * Characteristics and application examples of each model
  5. **Practical Data Analysis**

     * Loading data from CSV and GraphML
     * Calculating and interpreting basic statistics
     * Gaining insights through visualization

### Next Chapter

In Chapter 2, we will learn about **centrality metrics and community detection** :

  * Advanced centrality metrics (eigenvector centrality, PageRank)
  * Community detection algorithms
  * Modularity optimization
  * Hierarchical clustering
  * Community analysis on real data

* * *
