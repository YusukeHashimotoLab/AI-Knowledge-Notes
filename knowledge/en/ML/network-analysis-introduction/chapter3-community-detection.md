---
title: "Chapter 3: Community Detection"
chapter_title: "Chapter 3: Community Detection"
subtitle: Modularity Optimization and Label Propagation - Theory and Implementation of Louvain Method and Label Propagation
reading_time: 25-30 minutes
difficulty: Intermediate
code_examples: 8
exercises: 5
---

This chapter covers Community Detection. You will learn basic concepts of community detection, principles of Label Propagation, and Practice community analysis.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Understand the basic concepts of community detection and modularity
  * ✅ Implement the hierarchical optimization algorithm of the Louvain method
  * ✅ Learn the principles of Label Propagation and the secrets of its speed
  * ✅ Compare major community detection methods (Girvan-Newman, Infomap, Spectral Clustering)
  * ✅ Practice community analysis and visualization with real data

## 1\. Fundamentals of Community Detection

### 1.1 What is a Community?

A **community** in a network is a group of nodes with dense internal connections and sparse external connections. It represents important structures in various fields, such as friend groups in social networks, functional modules in biological networks, and topic groups of web pages.
    
    
    ```mermaid
    graph LR
        subgraph C1["Community 1"]
            A1((A))---A2((B))
            A2---A3((C))
            A3---A1
        end
        subgraph C2["Community 2"]
            B1((D))---B2((E))
            B2---B3((F))
            B3---B1
        end
        A2-.Weak connection.-B1
    ```

### 1.2 Modularity

**Modularity** is the most important metric for measuring the quality of community structure. It quantifies how clearly a network has community structure for a given community partition:

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$ 

Where:

  * $A_{ij}$: Element of the adjacency matrix (edge existence)
  * $k_i, k_j$: Degrees of nodes $i, j$
  * $m$: Total number of edges
  * $c_i, c_j$: Communities of nodes $i, j$
  * $\delta(c_i, c_j)$: 1 if in the same community, 0 otherwise

> **Intuitive Understanding:** Modularity measures the difference between the "actual number of edges" and the "expected value in a random network." The value range is -0.5 to 1.0, and a value of 0.3 or higher indicates a clear community structure. 

### 1.3 Evaluation Metrics
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: 1.3 Evaluation Metrics
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Intermediate
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import numpy as np
    from networkx.algorithms import community
    
    # Create sample network (Karate Club dataset)
    G = nx.karate_club_graph()
    
    # Ground truth communities (actual club split)
    ground_truth = {
        frozenset([n for n in G.nodes() if G.nodes[n]['club'] == 'Mr. Hi']),
        frozenset([n for n in G.nodes() if G.nodes[n]['club'] == 'Officer'])
    }
    
    # Community detection with Louvain method
    detected_communities = community.louvain_communities(G, seed=42)
    
    # Calculate modularity
    modularity = community.modularity(G, detected_communities)
    print(f"Modularity: {modularity:.4f}")
    
    # Calculate NMI (Normalized Mutual Information)
    from sklearn.metrics import normalized_mutual_info_score
    
    # Convert communities to label arrays
    def communities_to_labels(communities, n_nodes):
        labels = np.zeros(n_nodes, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                labels[node] = i
        return labels
    
    gt_labels = communities_to_labels(ground_truth, len(G))
    detected_labels = communities_to_labels(detected_communities, len(G))
    
    nmi = normalized_mutual_info_score(gt_labels, detected_labels)
    print(f"NMI: {nmi:.4f}")
    
    # Coverage and performance
    coverage = community.coverage(G, detected_communities)
    performance = community.performance(G, detected_communities)
    print(f"Coverage: {coverage:.4f}")
    print(f"Performance: {performance:.4f}")
    

Explanation of Major Evaluation Metrics Metric | Description | Range | Advantages/Disadvantages  
---|---|---|---  
**Modularity (Q)** | Density of intra-community edges | -0.5~1.0 | Most common / Resolution limit problem  
**NMI** | Agreement with ground truth (information content) | 0~1 | Requires ground truth  
**Coverage** | Proportion of intra-community edges | 0~1 | Intuitive / Ignores inter-community edges  
**Performance** | Proportion of correctly classified pairs | 0~1 | Balanced  
  
## 2\. Louvain Method

### 2.1 How the Algorithm Works

The Louvain method is a hierarchical algorithm that greedily optimizes modularity. It consists of a two-phase iterative process:
    
    
    ```mermaid
    graph TD
        A[Initialization: Each node is its own community] --> B[Phase 1: Local optimization]
        B --> C{Modularityimprovement?}
        C -->|Yes| B
        C -->|No| D[Phase 2: Network aggregation]
        D --> E{Singlecommunity?}
        E -->|No| B
        E -->|Yes| F[Complete]
    ```

**Phase 1 (Local Optimization):**

  1. Process each node in sequence
  2. Try moving to neighboring communities
  3. Adopt the move that maximizes modularity
  4. Repeat until no improvement

The modularity change $\Delta Q$ can be computed efficiently:

$$\Delta Q = \left[ \frac{\Sigma_{in} + k_{i,in}}{2m} - \left( \frac{\Sigma_{tot} + k_i}{2m} \right)^2 \right] - \left[ \frac{\Sigma_{in}}{2m} - \left( \frac{\Sigma_{tot}}{2m} \right)^2 - \left( \frac{k_i}{2m} \right)^2 \right]$$ 

### 2.2 Hierarchical Community Detection

**Phase 2 (Network Aggregation):**

  * Aggregate each community into a single supernode
  * Aggregate edges between communities
  * Execute Phase 1 on the new network

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    def visualize_louvain_hierarchy(G):
        """Visualize the hierarchical structure of Louvain method"""
        # Community detection at each level
        communities_level0 = [{i} for i in G.nodes()]  # Level 0: Individual nodes
        communities_level1 = community.louvain_communities(G, seed=42, resolution=1.0)
    
        # Coarser communities (lower resolution)
        communities_level2 = community.louvain_communities(G, seed=42, resolution=0.5)
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        levels = [communities_level0, communities_level1, communities_level2]
        titles = ['Level 0\n(Individual Nodes)', 'Level 1\n(Fine Communities)', 'Level 2\n(Coarse Communities)']
    
        for ax, comms, title in zip(axes, levels, titles):
            pos = nx.spring_layout(G, seed=42)
    
            # Color by community
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            node_colors = []
            for node in G.nodes():
                for i, comm in enumerate(comms):
                    if node in comm:
                        node_colors.append(colors[i % len(colors)])
                        break
    
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=300, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
    
        plt.tight_layout()
        plt.savefig('louvain_hierarchy.png', dpi=150, bbox_inches='tight')
        print(f"Number of levels: {len(levels)}")
        for i, comms in enumerate(levels):
            print(f"Level {i}: {len(comms)} communities")
    
    # Execute
    G = nx.karate_club_graph()
    visualize_louvain_hierarchy(G)
    

### 2.3 NetworkX/python-louvain Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    
    """
    Example: 2.3 NetworkX/python-louvain Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import community as community_louvain  # python-louvain
    import networkx as nx
    
    # Sample network (Les Miserables co-occurrence network)
    G = nx.les_miserables_graph()
    
    # --- NetworkX built-in Louvain ---
    communities_nx = community.louvain_communities(G, seed=42)
    modularity_nx = community.modularity(G, communities_nx)
    
    print(f"NetworkX Louvain:")
    print(f"  Number of communities: {len(communities_nx)}")
    print(f"  Modularity: {modularity_nx:.4f}")
    
    # --- python-louvain package (get more detailed information) ---
    # Get the best partition
    partition = community_louvain.best_partition(G)
    
    # Get the entire hierarchical structure
    dendro = community_louvain.generate_dendrogram(G)
    print(f"\npython-louvain:")
    print(f"  Number of hierarchical levels: {len(dendro)}")
    
    # Number of communities at each level
    for level in range(len(dendro)):
        partition_at_level = community_louvain.partition_at_level(dendro, level)
        num_communities = len(set(partition_at_level.values()))
        mod = community_louvain.modularity(partition_at_level, G)
        print(f"  Level {level}: {num_communities} communities, Q={mod:.4f}")
    
    # Effect of resolution parameter
    resolutions = [0.5, 1.0, 1.5, 2.0]
    print(f"\nEffect of resolution parameter:")
    for res in resolutions:
        partition_res = community_louvain.best_partition(G, resolution=res)
        num_comm = len(set(partition_res.values()))
        mod = community_louvain.modularity(partition_res, G)
        print(f"  resolution={res}: {num_comm} communities, Q={mod:.4f}")
    

> **Computational Complexity:** The time complexity of the Louvain method is $O(n \log n)$, making it applicable to large-scale networks (millions of nodes). The python-louvain package provides a particularly fast implementation. 

## 3\. Label Propagation

### 3.1 Label Propagation Algorithm

Label Propagation is an extremely simple and fast community detection method. The basic idea is "majority vote":

  1. **Initialization:** Assign a unique label to each node
  2. **Propagation:** Each node adopts the majority label from its neighbors
  3. **Convergence:** Repeat until labels no longer change

    
    
    ```mermaid
    graph LR
        subgraph "Step 0: Initialization"
            A0((A:1))
            B0((B:2))
            C0((C:3))
            D0((D:4))
            A0---B0---C0---D0
        end
    ```
    
    
    ```mermaid
    graph LR
        subgraph "Step 1: Propagation"
            A1((A:1))
            B1((B:1))
            C1((C:2))
            D1((D:3))
            A1---B1---C1---D1
        end
    ```
    
    
    ```mermaid
    graph LR
        subgraph "Step 2: Convergence"
            A2((A:1))
            B2((B:1))
            C2((C:1))
            D2((D:1))
            A2---B2---C2---D2
        end
    ```

### 3.2 Trade-off Between Speed and Accuracy

The main advantage of Label Propagation is its **linear time complexity $O(m)$** (where $m$ is the number of edges). However, there are some challenges:

Feature | Advantage | Disadvantage  
---|---|---  
**Computational speed** | Very fast (linear time) | -  
**Scalability** | Can handle networks with tens of millions of nodes | -  
**Stability** | - | Results vary between runs (non-deterministic)  
**Quality** | - | Lower modularity than Louvain method  
**Convergence** | - | May oscillate  
  
### 3.3 Implementation Example
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    # - numpy>=1.24.0, <2.0.0
    
    import networkx as nx
    import numpy as np
    from collections import Counter
    
    def label_propagation_manual(G, max_iter=100, seed=None):
        """Manual implementation of Label Propagation (for educational purposes)"""
        if seed is not None:
            np.random.seed(seed)
    
        # Initialization: unique label for each node
        labels = {node: i for i, node in enumerate(G.nodes())}
        nodes = list(G.nodes())
    
        for iteration in range(max_iter):
            # Randomize node processing order
            np.random.shuffle(nodes)
            changed = False
    
            for node in nodes:
                # Collect labels from neighbor nodes
                neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
    
                if not neighbor_labels:
                    continue
    
                # Adopt the most frequent label (random choice if tied)
                label_counts = Counter(neighbor_labels)
                max_count = max(label_counts.values())
                most_common = [label for label, count in label_counts.items()
                              if count == max_count]
                new_label = np.random.choice(most_common)
    
                if labels[node] != new_label:
                    labels[node] = new_label
                    changed = True
    
            if not changed:
                print(f"Converged: {iteration + 1} iterations")
                break
    
        # Convert labels to community sets
        communities = {}
        for node, label in labels.items():
            if label not in communities:
                communities[label] = set()
            communities[label].add(node)
    
        return list(communities.values())
    
    # Test and comparison
    G = nx.karate_club_graph()
    
    # Manual implementation
    communities_manual = label_propagation_manual(G, seed=42)
    mod_manual = community.modularity(G, communities_manual)
    
    # NetworkX built-in
    communities_nx = list(community.label_propagation_communities(G))
    mod_nx = community.modularity(G, communities_nx)
    
    print("Label Propagation result comparison:")
    print(f"Manual implementation: {len(communities_manual)} communities, Q={mod_manual:.4f}")
    print(f"NetworkX: {len(communities_nx)} communities, Q={mod_nx:.4f}")
    
    # Run multiple times to check stability
    print("\nStability test (10 runs):")
    modularities = []
    for i in range(10):
        comms = label_propagation_manual(G, seed=i)
        mod = community.modularity(G, comms)
        modularities.append(mod)
        print(f"  Run {i+1}: Q={mod:.4f}, Number of communities={len(comms)}")
    
    print(f"Average modularity: {np.mean(modularities):.4f} ± {np.std(modularities):.4f}")
    

> **Practical Advice:** Label Propagation is effective for initial exploration and ultra-large-scale networks. When higher quality results are needed, a "hybrid approach" that uses Label Propagation results as initial values for the Louvain method is effective. 

## 4\. Other Community Detection Methods

### 4.1 Girvan-Newman Method (Edge Betweenness-Based)

The Girvan-Newman method is a hierarchical approach that removes edges connecting communities:

  1. Calculate edge betweenness centrality for all edges
  2. Remove the edge with maximum betweenness centrality
  3. Calculate modularity
  4. Repeat until all edges are removed
  5. Adopt the partition with maximum modularity

    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    
    """
    Example: The Girvan-Newman method is a hierarchical approach that rem
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    from networkx.algorithms.community import girvan_newman
    
    G = nx.karate_club_graph()
    
    # Girvan-Newman method (generate entire hierarchy)
    communities_generator = girvan_newman(G)
    
    # Get different partition levels
    modularities = []
    all_partitions = []
    
    for i, communities in enumerate(communities_generator):
        partition = tuple(sorted(communities, key=len, reverse=True))
        all_partitions.append(partition)
    
        # Calculate modularity
        mod = community.modularity(G, partition)
        modularities.append(mod)
    
        print(f"Partition {i+1}: {len(partition)} communities, Q={mod:.4f}")
    
        # Check up to 10 partitions
        if i >= 9:
            break
    
    # Select optimal partition
    best_idx = np.argmax(modularities)
    best_partition = all_partitions[best_idx]
    print(f"\nOptimal partition: Level {best_idx+1}, Q={modularities[best_idx]:.4f}")
    

**Computational Complexity:** $O(m^2 n)$ - Not suitable for large-scale networks, but provides interpretable hierarchical structure for small-scale networks.

### 4.2 Infomap

Infomap formulates community detection as a random walk encoding problem. It seeks partitions that can efficiently describe random walks that stay within communities for long periods.
    
    
    try:
        import infomap
        has_infomap = True
    except ImportError:
        has_infomap = False
        print("infomap not installed: pip install infomap")
    
    if has_infomap:
        # Run Infomap
        im = infomap.Infomap("--two-level --directed")
    
        # Add network (for undirected graphs, add both directions)
        for u, v in G.edges():
            im.add_link(u, v)
            im.add_link(v, u)
    
        # Run clustering
        im.run()
    
        # Get results
        communities_infomap = {}
        for node in im.tree:
            if node.is_leaf:
                module_id = node.module_id
                node_id = node.node_id
                if module_id not in communities_infomap:
                    communities_infomap[module_id] = set()
                communities_infomap[module_id].add(node_id)
    
        communities_infomap = list(communities_infomap.values())
        mod_infomap = community.modularity(G, communities_infomap)
    
        print(f"\nInfomap:")
        print(f"  Number of communities: {len(communities_infomap)}")
        print(f"  Modularity: {mod_infomap:.4f}")
        print(f"  Codelength: {im.codelength:.4f}")
    

### 4.3 Spectral Clustering

Spectral clustering detects communities using eigenvectors of the graph Laplacian:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Spectral clustering detects communities using eigenvectors o
    
    Purpose: Demonstrate machine learning model training and evaluation
    Target: Advanced
    Execution time: 10-30 seconds
    Dependencies: None
    """
    
    from sklearn.cluster import SpectralClustering
    import numpy as np
    
    # Get adjacency matrix
    A = nx.to_numpy_array(G)
    
    # Try with different numbers of communities
    for n_clusters in [2, 3, 4, 5]:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
    
        labels = sc.fit_predict(A)
    
        # Convert labels to community sets
        communities_spectral = [set() for _ in range(n_clusters)]
        for node, label in enumerate(labels):
            communities_spectral[label].add(node)
    
        mod = community.modularity(G, communities_spectral)
        print(f"Spectral Clustering (k={n_clusters}): Q={mod:.4f}")
    

Detailed Method Comparison Method | Time Complexity | Quality | Deterministic | Application Scenario  
---|---|---|---|---  
**Louvain** | $O(n \log n)$ | High | Semi-deterministic | General purpose, large-scale networks  
**Label Propagation** | $O(m)$ | Medium | Non-deterministic | Ultra-large-scale networks, initial exploration  
**Girvan-Newman** | $O(m^2 n)$ | Medium~High | Deterministic | Small-scale networks, hierarchical structure visualization  
**Infomap** | $O(m)$ | High | Semi-deterministic | When flow information is important  
**Spectral** | $O(n^3)$ | Medium | Deterministic | When number of communities is known  
  
## 5\. Practice: Community Analysis of Social Networks

### 5.1 Facebook Network Example
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - networkx>=3.1.0
    
    """
    Example: 5.1 Facebook Network Example
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Alternative to Facebook SNAP dataset (ego-Facebook): Detailed analysis with Zachary's Karate Club
    G = nx.karate_club_graph()
    
    print(f"Network information:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  Density: {nx.density(G):.4f}")
    
    # Ground truth (actual club split)
    ground_truth = []
    for club in ['Mr. Hi', 'Officer']:
        comm = {n for n in G.nodes() if G.nodes[n]['club'] == club}
        ground_truth.append(comm)
    
    print(f"\nGround truth: {len(ground_truth)} groups")
    
    # Community detection with each method
    results = {}
    
    # 1. Louvain
    communities_louvain = community.louvain_communities(G, seed=42)
    results['Louvain'] = communities_louvain
    
    # 2. Label Propagation
    communities_lp = list(community.label_propagation_communities(G))
    results['Label Propagation'] = communities_lp
    
    # 3. Greedy Modularity (fast alternative method)
    communities_greedy = community.greedy_modularity_communities(G)
    results['Greedy Modularity'] = communities_greedy
    
    # 4. Girvan-Newman (optimal partition only)
    gn_generator = girvan_newman(G)
    gn_modularities = []
    gn_partitions = []
    for partition in gn_generator:
        gn_partitions.append(partition)
        gn_modularities.append(community.modularity(G, partition))
        if len(gn_partitions) >= 10:  # Evaluate only first 10 partitions
            break
    best_gn = gn_partitions[np.argmax(gn_modularities)]
    results['Girvan-Newman'] = best_gn
    
    print("\nCommunity detection results:")
    for method, comms in results.items():
        mod = community.modularity(G, comms)
    
        # Calculate NMI with ground truth
        gt_labels = communities_to_labels(ground_truth, len(G))
        detected_labels = communities_to_labels(comms, len(G))
        nmi = normalized_mutual_info_score(gt_labels, detected_labels)
    
        print(f"{method:20s}: {len(comms):2d} communities, Q={mod:.4f}, NMI={nmi:.4f}")
    

### 5.2 Method Comparison
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    
    # Comprehensive comparison analysis
    def comprehensive_comparison(G, ground_truth, results):
        """Compare methods with multiple evaluation metrics"""
        metrics = {
            'Modularity': [],
            'NMI': [],
            'ARI': [],  # Adjusted Rand Index
            'Coverage': [],
            'Communities': [],
            'Runtime': []
        }
    
        methods = list(results.keys())
        gt_labels = communities_to_labels(ground_truth, len(G))
    
        import time
    
        for method in methods:
            comms = results[method]
            detected_labels = communities_to_labels(comms, len(G))
    
            # Calculate metrics
            metrics['Modularity'].append(community.modularity(G, comms))
            metrics['NMI'].append(normalized_mutual_info_score(gt_labels, detected_labels))
            metrics['ARI'].append(adjusted_rand_score(gt_labels, detected_labels))
            metrics['Coverage'].append(community.coverage(G, comms))
            metrics['Communities'].append(len(comms))
    
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
        for i, (metric, values) in enumerate(list(metrics.items())[:5]):
            ax = axes[i]
            bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
    
            # Display values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
        # Remove last subplot
        fig.delaxes(axes[5])
    
        plt.tight_layout()
        plt.savefig('community_comparison.png', dpi=150, bbox_inches='tight')
        print("Comparison plot saved: community_comparison.png")
    
    # Execute
    comprehensive_comparison(G, ground_truth, results)
    

### 5.3 Community Visualization and Interpretation
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    def visualize_communities(G, communities, title, ground_truth=None):
        """Beautifully visualize communities"""
        fig, ax = plt.subplots(figsize=(12, 10))
    
        # Calculate layout
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
        # Color map
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84']
    
        # Color nodes
        node_colors = []
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_colors.append(colors[i % len(colors)])
                node_to_comm[node] = i
    
        # Draw edges (distinguish intra/inter-community)
        edges_within = []
        edges_between = []
        for u, v in G.edges():
            if node_to_comm[u] == node_to_comm[v]:
                edges_within.append((u, v))
            else:
                edges_between.append((u, v))
    
        # Intra-community edges (dark color)
        nx.draw_networkx_edges(G, pos, edgelist=edges_within,
                               width=2, alpha=0.6, edge_color='#2C3E50', ax=ax)
    
        # Inter-community edges (light color, dashed)
        nx.draw_networkx_edges(G, pos, edgelist=edges_between,
                               width=1, alpha=0.3, edge_color='#95A5A6',
                               style='dashed', ax=ax)
    
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=500, alpha=0.9,
                               edgecolors='white', linewidths=2, ax=ax)
    
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
        # Legend
        legend_elements = [mpatches.Patch(facecolor=colors[i % len(colors)],
                                          label=f'Community {i+1} ({len(comm)} nodes)')
                          for i, comm in enumerate(communities)]
        ax.legend(handles=legend_elements, loc='upper left',
                 framealpha=0.9, fontsize=10)
    
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
    
        # Display metrics
        mod = community.modularity(G, communities)
        info_text = f"Modularity: {mod:.4f}\n"
        info_text += f"Communities: {len(communities)}\n"
        info_text += f"Intra-edges: {len(edges_within)}\n"
        info_text += f"Inter-edges: {len(edges_between)}"
    
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        return fig
    
    # Visualize results of each method
    for method, comms in results.items():
        fig = visualize_communities(G, comms, f"Community Detection: {method}")
        plt.savefig(f'community_{method.replace(" ", "_").lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("All visualizations saved")
    

> **Key Points for Interpretation:**
> 
>   * **Modularity:** 0.3 or higher indicates clear community structure
>   * **Community size:** Be cautious of extremely small/large communities
>   * **Edge ratio:** Many inter-community edges indicate ambiguous boundaries
>   * **Agreement with ground truth:** Use NMI/ARI to check correspondence with known structure
> 

## Exercises

Exercise 1: Understanding Modularity

**Problem:** For the following network, calculate the modularity of community partition C1 = {A, B} and C2 = {C, D} by hand.
    
    
    A -- B
    |    |
    C -- D
    

**Hint:** Use $Q = \frac{1}{2m} \sum_{ij} [A_{ij} - \frac{k_i k_j}{2m}] \delta(c_i, c_j)$

Exercise 2: Implementing Louvain Method

**Problem:** Implement a function that efficiently computes $\Delta Q$ when moving a node in Phase 1 local optimization.
    
    
    def compute_delta_Q(G, node, current_comm, new_comm, m):
        """
        Calculate the modularity change when moving a node
        from current_comm to new_comm
    
        Parameters:
        -----------
        G : NetworkX graph
        node : int
            Node to move
        current_comm : set
            Current community
        new_comm : set
            Destination community
        m : int
            Total number of edges
    
        Returns:
        --------
        delta_Q : float
            Modularity change
        """
        # Implement here
        pass
    

Exercise 3: Convergence of Label Propagation

**Problem:** Construct a case where Label Propagation does not converge (oscillates) and explain the reason. Also, propose an improvement strategy to guarantee convergence.

Exercise 4: Resolution Limit Problem

**Problem:** Modularity optimization has a "resolution limit" problem. Create an example where small communities cannot be detected, and show that it can be improved with the resolution parameter.
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    
    """
    Example: Problem:Modularity optimization has a "resolution limit" pro
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    from networkx.algorithms import community
    
    # Create a network with multiple small cliques
    # Compare results by changing resolution parameter
    

Exercise 5: Community Detection in Weighted Networks

**Problem:** Apply the Louvain method that considers weights to a network with weighted edges. Analyze how results differ with and without weights.
    
    
    # Requirements:
    # - Python 3.9+
    # - networkx>=3.1.0
    
    """
    Example: Problem:Apply the Louvain method that considers weights to a
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import networkx as nx
    
    # Create weighted network
    G = nx.karate_club_graph()
    
    # Assign random weights
    import random
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 2.0)
    
    # Compare community detection with and without weights
    

## Summary

In this chapter, we learned the theory and practice of community detection in networks:

  * ✅ **Modularity:** Standard metric for quantifying the quality of community structure
  * ✅ **Louvain Method:** High-quality, fast detection through hierarchical greedy optimization
  * ✅ **Label Propagation:** Ultra-fast algorithm with linear time complexity (with trade-offs)
  * ✅ **Other Methods:** Girvan-Newman, Infomap, Spectral Clustering
  * ✅ **Practice:** Methods for comparing, visualizing, and interpreting approaches

> **Next Steps:** In the next chapter, we will handle the dynamic nature of networks. We will learn about analyzing time-evolving networks, link prediction, and network growth models.
