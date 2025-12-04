---
title: 第1章：グラフとグラフ表現学習の基礎
chapter_title: 第1章：グラフとグラフ表現学習の基礎
subtitle: グラフ理論の基礎、グラフ表現、特徴量抽出、グラフ埋め込み手法の理解
reading_time: 30-35分
difficulty: 初級〜中級
code_examples: 12
exercises: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ グラフの基本概念（ノード、エッジ、有向/無向グラフ）を理解する
  * ✅ グラフの種類（木、DAG、完全グラフ、二部グラフ）を説明できる
  * ✅ グラフの表現方法（隣接行列、隣接リスト、エッジリスト）を使い分けられる
  * ✅ NetworkXを使ってグラフを作成・可視化できる
  * ✅ グラフの特徴量（次数、クラスタリング係数、中心性指標）を計算できる
  * ✅ PageRankアルゴリズムの原理を理解し実装できる
  * ✅ Random Walk based 埋め込み手法（DeepWalk、Node2Vec）を理解する
  * ✅ グラフ埋め込みを用いてノード分類やリンク予測を実装できる
  * ✅ コミュニティ検出アルゴリズムを適用できる
  * ✅ ソーシャルネットワーク分析を実践できる

* * *

## 1.1 グラフ理論の基礎

### グラフとは何か

グラフは、オブジェクト間の関係を表現する数学的構造です。ソーシャルネットワーク、分子構造、道路網、知識グラフなど、現実世界の多くの問題をグラフで表現できます。

> 「グラフ $G$ は、ノード（頂点）の集合 $V$ とエッジ（辺）の集合 $E$ で定義される：$G = (V, E)$」

#### 基本用語

  * **ノード（Node/Vertex）** ：エンティティを表す点（例：人、Webページ、原子）
  * **エッジ（Edge/Link）** ：ノード間の関係を表す線（例：友人関係、リンク、化学結合）
  * **有向グラフ（Directed Graph）** ：エッジに方向性がある（例：Twitter のフォロー関係）
  * **無向グラフ（Undirected Graph）** ：エッジに方向性がない（例：Facebook の友人関係）
  * **重み付きグラフ（Weighted Graph）** ：エッジに重み（数値）が付与される

    
    
    ```mermaid
    graph LR
        subgraph "無向グラフ"
        A1((A)) --- B1((B))
        B1 --- C1((C))
        C1 --- A1
        A1 --- D1((D))
        end
    
        subgraph "有向グラフ"
        A2((A)) --> B2((B))
        B2 --> C2((C))
        C2 --> A2
        A2 --> D2((D))
        D2 --> B2
        end
    
        style A1 fill:#e3f2fd
        style B1 fill:#e3f2fd
        style C1 fill:#e3f2fd
        style D1 fill:#e3f2fd
        style A2 fill:#fff3e0
        style B2 fill:#fff3e0
        style C2 fill:#fff3e0
        style D2 fill:#fff3e0
    ```

### グラフの種類

グラフの種類 | 定義 | 具体例  
---|---|---  
**木（Tree）** | 閉路を持たない連結グラフ | ファイルシステム、組織図  
**DAG** | 有向閉路を持たない有向グラフ | タスクの依存関係、因果グラフ  
**完全グラフ** | 全ノード間にエッジが存在 | 完全に接続されたネットワーク  
**二部グラフ** | ノードが2つのグループに分割可能 | 推薦システム（ユーザー-アイテム）  
**サイクルグラフ** | 単一の閉路を形成 | 循環参照、リング構造  
**正則グラフ** | 全ノードの次数が等しい | 結晶格子、トーラスグラフ  
      
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("=== NetworkX によるグラフの基本操作 ===\n")
    
    # 1. 無向グラフの作成
    G_undirected = nx.Graph()
    G_undirected.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'C'), ('C', 'D')
    ])
    
    print("無向グラフ:")
    print(f"  ノード数: {G_undirected.number_of_nodes()}")
    print(f"  エッジ数: {G_undirected.number_of_edges()}")
    print(f"  ノード: {list(G_undirected.nodes())}")
    print(f"  エッジ: {list(G_undirected.edges())}\n")
    
    # 2. 有向グラフの作成
    G_directed = nx.DiGraph()
    G_directed.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'A'),
        ('A', 'D'), ('D', 'B')
    ])
    
    print("有向グラフ:")
    print(f"  ノード数: {G_directed.number_of_nodes()}")
    print(f"  エッジ数: {G_directed.number_of_edges()}")
    print(f"  ノードAの出次数: {G_directed.out_degree('A')}")
    print(f"  ノードAの入次数: {G_directed.in_degree('A')}\n")
    
    # 3. 重み付きグラフの作成
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([
        ('Tokyo', 'Osaka', 400),
        ('Tokyo', 'Nagoya', 350),
        ('Osaka', 'Nagoya', 180),
        ('Osaka', 'Fukuoka', 500)
    ])
    
    print("重み付きグラフ（都市間距離）:")
    for u, v, weight in G_weighted.edges(data='weight'):
        print(f"  {u} - {v}: {weight}km")
    
    # 4. 特殊なグラフの生成
    print("\n=== 特殊なグラフの生成 ===\n")
    
    # 完全グラフ（K5: 5ノードの完全グラフ）
    G_complete = nx.complete_graph(5)
    print(f"完全グラフ K5:")
    print(f"  ノード数: {G_complete.number_of_nodes()}")
    print(f"  エッジ数: {G_complete.number_of_edges()} (理論値: n(n-1)/2 = 10)\n")
    
    # 木（二分木）
    G_tree = nx.balanced_tree(r=2, h=3)  # r=分岐数, h=深さ
    print(f"二分木 (深さ3):")
    print(f"  ノード数: {G_tree.number_of_nodes()}")
    print(f"  エッジ数: {G_tree.number_of_edges()}")
    print(f"  木構造か: {nx.is_tree(G_tree)}\n")
    
    # 二部グラフ
    G_bipartite = nx.complete_bipartite_graph(3, 4)
    print(f"完全二部グラフ K(3,4):")
    print(f"  ノード数: {G_bipartite.number_of_nodes()}")
    print(f"  エッジ数: {G_bipartite.number_of_edges()}")
    print(f"  二部グラフか: {nx.is_bipartite(G_bipartite)}\n")
    
    # グラフの可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 無向グラフ
    nx.draw(G_undirected, ax=axes[0, 0], with_labels=True,
            node_color='lightblue', node_size=800, font_size=12)
    axes[0, 0].set_title('無向グラフ', fontsize=14)
    
    # 有向グラフ
    nx.draw(G_directed, ax=axes[0, 1], with_labels=True,
            node_color='lightcoral', node_size=800, font_size=12,
            arrows=True, arrowsize=20)
    axes[0, 1].set_title('有向グラフ', fontsize=14)
    
    # 重み付きグラフ
    pos = nx.spring_layout(G_weighted, seed=42)
    nx.draw(G_weighted, pos, ax=axes[0, 2], with_labels=True,
            node_color='lightgreen', node_size=1000, font_size=10)
    edge_labels = nx.get_edge_attributes(G_weighted, 'weight')
    nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels, ax=axes[0, 2])
    axes[0, 2].set_title('重み付きグラフ', fontsize=14)
    
    # 完全グラフ
    nx.draw(G_complete, ax=axes[1, 0], with_labels=True,
            node_color='lightyellow', node_size=800)
    axes[1, 0].set_title('完全グラフ K5', fontsize=14)
    
    # 木
    nx.draw(G_tree, ax=axes[1, 1], with_labels=True,
            node_color='lavender', node_size=600, font_size=8)
    axes[1, 1].set_title('二分木 (深さ3)', fontsize=14)
    
    # 二部グラフ
    nx.draw(G_bipartite, ax=axes[1, 2], with_labels=True,
            node_color='peachpuff', node_size=600)
    axes[1, 2].set_title('完全二部グラフ K(3,4)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('graph_types.png', dpi=150, bbox_inches='tight')
    print("グラフを 'graph_types.png' に保存しました。")
    

実行結果の例
    
    
    === NetworkX によるグラフの基本操作 ===
    
    無向グラフ:
      ノード数: 4
      エッジ数: 5
      ノード: ['A', 'B', 'C', 'D']
      エッジ: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    
    有向グラフ:
      ノード数: 4
      エッジ数: 5
      ノードAの出次数: 2
      ノードAの入次数: 1
    
    重み付きグラフ（都市間距離）:
      Tokyo - Osaka: 400km
      Tokyo - Nagoya: 350km
      Osaka - Nagoya: 180km
      Osaka - Fukuoka: 500km
    
    === 特殊なグラフの生成 ===
    
    完全グラフ K5:
      ノード数: 5
      エッジ数: 10 (理論値: n(n-1)/2 = 10)
    
    二分木 (深さ3):
      ノード数: 15
      エッジ数: 14
      木構造か: True
    
    完全二部グラフ K(3,4):
      ノード数: 7
      エッジ数: 12
      二部グラフか: True
    

* * *

## 1.2 グラフの表現方法

### 3つの主要な表現形式

グラフをコンピュータで扱うには、適切なデータ構造で表現する必要があります。主な表現方法は以下の3つです：

#### 1\. 隣接行列（Adjacency Matrix）

$n$ 個のノードを持つグラフに対して、$n \times n$ の行列 $A$ で表現します：

$$ A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\\ 0 & \text{otherwise} \end{cases} $$ 

**特徴** ：

  * エッジの存在確認が $O(1)$ で可能
  * 密グラフに適している
  * メモリ使用量：$O(n^2)$
  * 無向グラフの場合、行列は対称

#### 2\. 隣接リスト（Adjacency List）

各ノードについて、隣接するノードのリストを保持します：

**特徴** ：

  * 疎グラフに適している
  * メモリ使用量：$O(n + m)$（$m$ はエッジ数）
  * 近傍ノードの走査が効率的

#### 3\. エッジリスト（Edge List）

単純にエッジの集合として表現：$E = \\{(u_1, v_1), (u_2, v_2), \ldots\\}$

**特徴** ：

  * 最もシンプルな表現
  * グラフ全体の走査に適している
  * 特定エッジの検索は非効率（$O(m)$）

操作 | 隣接行列 | 隣接リスト | エッジリスト  
---|---|---|---  
エッジの存在確認 | $O(1)$ | $O(d)$ | $O(m)$  
隣接ノードの取得 | $O(n)$ | $O(d)$ | $O(m)$  
全エッジの走査 | $O(n^2)$ | $O(n+m)$ | $O(m)$  
メモリ使用量 | $O(n^2)$ | $O(n+m)$ | $O(m)$  
  
※ $d$ はノードの次数、$m$ はエッジ数、$n$ はノード数
    
    
    import networkx as nx
    import numpy as np
    import pandas as pd
    
    print("=== グラフの表現方法の比較 ===\n")
    
    # サンプルグラフの作成
    G = nx.Graph()
    edges = [
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'C'), ('C', 'D')
    ]
    G.add_edges_from(edges)
    
    print("グラフ構造:")
    print(f"  ノード: {list(G.nodes())}")
    print(f"  エッジ: {list(G.edges())}\n")
    
    # 1. 隣接行列（Adjacency Matrix）
    print("1. 隣接行列（Adjacency Matrix）")
    print("-" * 40)
    adj_matrix = nx.adjacency_matrix(G).todense()
    node_list = sorted(G.nodes())
    
    print("行列形式:")
    df_adj = pd.DataFrame(adj_matrix, index=node_list, columns=node_list)
    print(df_adj)
    
    print("\nNumPy配列形式:")
    print(adj_matrix)
    
    # 隣接行列の性質
    print(f"\n対称行列か: {np.allclose(adj_matrix, adj_matrix.T)}")
    print(f"メモリ使用量: O(n²) = O({len(G.nodes())}²) = {len(G.nodes())**2} 要素")
    
    # 2. 隣接リスト（Adjacency List）
    print("\n2. 隣接リスト（Adjacency List）")
    print("-" * 40)
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}
    for node, neighbors in sorted(adj_list.items()):
        print(f"  {node}: {neighbors}")
    
    print(f"\nメモリ使用量: O(n+m) = O({len(G.nodes())}+{len(G.edges())}) "
          f"= {len(G.nodes()) + len(G.edges())} 要素")
    
    # 3. エッジリスト（Edge List）
    print("\n3. エッジリスト（Edge List）")
    print("-" * 40)
    edge_list = list(G.edges())
    for edge in edge_list:
        print(f"  {edge}")
    
    print(f"\nメモリ使用量: O(m) = O({len(G.edges())}) = {len(G.edges())} エッジ")
    
    # 4. 各表現での操作の比較
    print("\n" + "=" * 50)
    print("各表現での基本操作")
    print("=" * 50)
    
    # エッジの存在確認
    print("\n【エッジの存在確認】")
    u, v = 'A', 'C'
    print(f"エッジ ({u}, {v}) は存在するか?\n")
    
    # 隣接行列での確認
    u_idx = node_list.index(u)
    v_idx = node_list.index(v)
    exists_matrix = adj_matrix[u_idx, v_idx] == 1
    print(f"  隣接行列: {exists_matrix} (O(1) 時間)")
    
    # 隣接リストでの確認
    exists_list = v in adj_list[u]
    print(f"  隣接リスト: {exists_list} (O(d) 時間)")
    
    # エッジリストでの確認
    exists_edges = (u, v) in edge_list or (v, u) in edge_list
    print(f"  エッジリスト: {exists_edges} (O(m) 時間)")
    
    # 隣接ノードの取得
    print(f"\n【ノード {u} の隣接ノード】\n")
    
    # 隣接行列から取得
    neighbors_matrix = [node_list[i] for i in range(len(node_list))
                       if adj_matrix[u_idx, i] == 1]
    print(f"  隣接行列: {neighbors_matrix} (O(n) 時間)")
    
    # 隣接リストから取得
    neighbors_list = adj_list[u]
    print(f"  隣接リスト: {neighbors_list} (O(d) 時間)")
    
    # エッジリストから取得
    neighbors_edges = list(set([v for (s, t) in edge_list
                               for v in [s, t] if (s == u or t == u) and v != u]))
    print(f"  エッジリスト: {neighbors_edges} (O(m) 時間)")
    
    # 5. 有向グラフと重み付きグラフの表現
    print("\n" + "=" * 50)
    print("有向グラフと重み付きグラフの表現")
    print("=" * 50)
    
    # 有向グラフ
    G_directed = nx.DiGraph()
    G_directed.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    
    print("\n有向グラフの隣接行列:")
    adj_directed = nx.adjacency_matrix(G_directed).todense()
    print(pd.DataFrame(adj_directed,
                      index=sorted(G_directed.nodes()),
                      columns=sorted(G_directed.nodes())))
    print(f"対称行列か: {np.allclose(adj_directed, adj_directed.T)}")
    
    # 重み付きグラフ
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([
        ('A', 'B', 0.5),
        ('A', 'C', 1.2),
        ('B', 'C', 0.8)
    ])
    
    print("\n重み付きグラフの隣接行列:")
    adj_weighted = nx.adjacency_matrix(G_weighted).todense()
    print(pd.DataFrame(adj_weighted,
                      index=sorted(G_weighted.nodes()),
                      columns=sorted(G_weighted.nodes())))
    
    print("\n重みを含む隣接リスト:")
    for node in sorted(G_weighted.nodes()):
        neighbors_with_weight = [(neighbor, G_weighted[node][neighbor]['weight'])
                                for neighbor in G_weighted.neighbors(node)]
        print(f"  {node}: {neighbors_with_weight}")
    

実行結果の例
    
    
    === グラフの表現方法の比較 ===
    
    グラフ構造:
      ノード: ['A', 'B', 'C', 'D']
      エッジ: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    
    1. 隣接行列（Adjacency Matrix）
    ----------------------------------------
    行列形式:
       A  B  C  D
    A  0  1  1  1
    B  1  0  1  0
    C  1  1  0  1
    D  1  0  1  0
    
    対称行列か: True
    メモリ使用量: O(n²) = O(4²) = 16 要素
    
    2. 隣接リスト（Adjacency List）
    ----------------------------------------
      A: ['B', 'C', 'D']
      B: ['A', 'C']
      C: ['A', 'B', 'D']
      D: ['A', 'C']
    
    メモリ使用量: O(n+m) = O(4+5) = 9 要素
    
    3. エッジリスト（Edge List）
    ----------------------------------------
      ('A', 'B')
      ('A', 'C')
      ('A', 'D')
      ('B', 'C')
      ('C', 'D')
    
    メモリ使用量: O(m) = O(5) = 5 エッジ
    

* * *

## 1.3 グラフの特徴量

### ノードレベルの特徴量

グラフニューラルネットワークでは、ノードやエッジの特徴量が重要です。以下は代表的な特徴量です：

#### 1\. 次数（Degree）

ノード $v$ に接続されているエッジの数：

$$ d(v) = |\\{u \in V : (v, u) \in E\\}| $$ 

  * **無向グラフ** ：単純に接続エッジ数
  * **有向グラフ** ：入次数（in-degree）と出次数（out-degree）

#### 2\. クラスタリング係数（Clustering Coefficient）

ノードの近傍がどれだけ密に接続されているかを示す指標：

$$ C(v) = \frac{2 \cdot |\\{(u, w) : u, w \in N(v), (u, w) \in E\\}|}{d(v) \cdot (d(v) - 1)} $$ 

$N(v)$ はノード $v$ の近傍ノード集合。値は $[0, 1]$ の範囲で、1に近いほど近傍が密に接続されています。

#### 3\. 中心性指標（Centrality Measures）

ノードの「重要度」を測る指標です：

中心性 | 定義 | 意味  
---|---|---  
**次数中心性** | $C_D(v) = d(v)$ | 接続数が多い = 重要  
**近接中心性** | $C_C(v) = \frac{n-1}{\sum_{u} d(v,u)}$ | 他ノードへ近い = 重要  
**媒介中心性** | $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ | 最短経路上にある = 重要  
**固有ベクトル中心性** | $A \mathbf{x} = \lambda \mathbf{x}$ | 重要なノードと接続 = 重要  
**PageRank** | 反復計算による重要度 | Google検索アルゴリズム  
      
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("=== グラフの特徴量計算 ===\n")
    
    # Zachary Karate Club グラフ（有名な社会ネットワーク）
    G = nx.karate_club_graph()
    print(f"Zachary Karate Club ネットワーク:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}\n")
    
    # 1. 次数（Degree）
    print("=" * 60)
    print("1. 次数（Degree）")
    print("=" * 60)
    
    degrees = dict(G.degree())
    print("\n各ノードの次数（上位5ノード）:")
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    for node, degree in sorted_degrees[:5]:
        print(f"  ノード {node:2d}: 次数 {degree}")
    
    avg_degree = np.mean(list(degrees.values()))
    print(f"\n平均次数: {avg_degree:.2f}")
    
    # 2. クラスタリング係数（Clustering Coefficient）
    print("\n" + "=" * 60)
    print("2. クラスタリング係数（Clustering Coefficient）")
    print("=" * 60)
    
    clustering_coeffs = nx.clustering(G)
    print("\nクラスタリング係数（上位5ノード）:")
    sorted_clustering = sorted(clustering_coeffs.items(),
                              key=lambda x: x[1], reverse=True)
    for node, coeff in sorted_clustering[:5]:
        print(f"  ノード {node:2d}: {coeff:.3f}")
    
    avg_clustering = nx.average_clustering(G)
    print(f"\n平均クラスタリング係数: {avg_clustering:.3f}")
    
    # 3. 中心性指標（Centrality Measures）
    print("\n" + "=" * 60)
    print("3. 中心性指標（Centrality Measures）")
    print("=" * 60)
    
    # 次数中心性
    degree_centrality = nx.degree_centrality(G)
    print("\n次数中心性（上位5ノード）:")
    sorted_dc = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, centrality in sorted_dc[:5]:
        print(f"  ノード {node:2d}: {centrality:.3f}")
    
    # 近接中心性
    closeness_centrality = nx.closeness_centrality(G)
    print("\n近接中心性（上位5ノード）:")
    sorted_cc = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, centrality in sorted_cc[:5]:
        print(f"  ノード {node:2d}: {centrality:.3f}")
    
    # 媒介中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    print("\n媒介中心性（上位5ノード）:")
    sorted_bc = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, centrality in sorted_bc[:5]:
        print(f"  ノード {node:2d}: {centrality:.3f}")
    
    # 固有ベクトル中心性
    eigenvector_centrality = nx.eigenvector_centrality(G)
    print("\n固有ベクトル中心性（上位5ノード）:")
    sorted_ec = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    for node, centrality in sorted_ec[:5]:
        print(f"  ノード {node:2d}: {centrality:.3f}")
    
    # PageRank
    pagerank = nx.pagerank(G)
    print("\nPageRank（上位5ノード）:")
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    for node, rank in sorted_pr[:5]:
        print(f"  ノード {node:2d}: {rank:.3f}")
    
    # 4. グラフレベルの特徴量
    print("\n" + "=" * 60)
    print("4. グラフレベルの特徴量")
    print("=" * 60)
    
    # 直径（Diameter）：最長の最短経路
    diameter = nx.diameter(G)
    print(f"\n直径（Diameter）: {diameter}")
    
    # 平均最短経路長
    avg_shortest_path = nx.average_shortest_path_length(G)
    print(f"平均最短経路長: {avg_shortest_path:.3f}")
    
    # 密度（Density）
    density = nx.density(G)
    print(f"密度（Density）: {density:.3f}")
    
    # 推移性（Transitivity）：グラフ全体のクラスタリング係数
    transitivity = nx.transitivity(G)
    print(f"推移性（Transitivity）: {transitivity:.3f}")
    
    # 連結成分数
    num_components = nx.number_connected_components(G)
    print(f"連結成分数: {num_components}")
    
    # 5. 可視化：中心性指標の比較
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pos = nx.spring_layout(G, seed=42)
    
    # ノードサイズを中心性で調整
    def draw_with_centrality(ax, centrality, title):
        node_sizes = [v * 3000 for v in centrality.values()]
        nx.draw(G, pos, ax=ax, node_size=node_sizes,
                node_color=list(centrality.values()),
                cmap='YlOrRd', with_labels=True, font_size=8,
                edge_color='gray', alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    draw_with_centrality(axes[0, 0], degree_centrality, '次数中心性')
    draw_with_centrality(axes[0, 1], closeness_centrality, '近接中心性')
    draw_with_centrality(axes[0, 2], betweenness_centrality, '媒介中心性')
    draw_with_centrality(axes[1, 0], eigenvector_centrality, '固有ベクトル中心性')
    draw_with_centrality(axes[1, 1], pagerank, 'PageRank')
    
    # 次数分布
    axes[1, 2].hist(list(degrees.values()), bins=15, color='skyblue', edgecolor='black')
    axes[1, 2].set_xlabel('次数', fontsize=12)
    axes[1, 2].set_ylabel('頻度', fontsize=12)
    axes[1, 2].set_title('次数分布', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graph_features.png', dpi=150, bbox_inches='tight')
    print("\n可視化を 'graph_features.png' に保存しました。")
    

実行結果の例
    
    
    === グラフの特徴量計算 ===
    
    Zachary Karate Club ネットワーク:
      ノード数: 34
      エッジ数: 78
    
    ============================================================
    1. 次数（Degree）
    ============================================================
    
    各ノードの次数（上位5ノード）:
      ノード 33: 次数 17
      ノード  0: 次数 16
      ノード 32: 次数 12
      ノード  2: 次数 10
      ノード  1: 次数 9
    
    平均次数: 4.59
    
    ============================================================
    2. クラスタリング係数（Clustering Coefficient）
    ============================================================
    
    クラスタリング係数（上位5ノード）:
      ノード  5: 0.667
      ノード  6: 0.600
      ノード 11: 0.545
      ノード  4: 0.500
      ノード 10: 0.467
    
    平均クラスタリング係数: 0.571
    
    ============================================================
    3. 中心性指標（Centrality Measures）
    ============================================================
    
    次数中心性（上位5ノード）:
      ノード 33: 0.515
      ノード  0: 0.485
      ノード 32: 0.364
      ノード  2: 0.303
      ノード  1: 0.273
    
    媒介中心性（上位5ノード）:
      ノード  0: 0.438
      ノード 33: 0.304
      ノード 32: 0.145
      ノード  2: 0.143
      ノード 31: 0.138
    
    PageRank（上位5ノード）:
      ノード 33: 0.101
      ノード  0: 0.097
      ノード 32: 0.071
      ノード  2: 0.057
      ノード  1: 0.053
    

### PageRankアルゴリズムの実装

PageRankは、Webページの重要度を計算するGoogleの検索アルゴリズムです。基本的なアイデアは：

> 「重要なページからリンクされているページは重要である」

PageRankは反復計算により求められます：

$$ PR(v) = \frac{1-d}{N} + d \sum_{u \in N_{in}(v)} \frac{PR(u)}{d_{out}(u)} $$ 

ここで：

  * $d$: ダンピング係数（通常0.85）
  * $N$: グラフ内のノード総数
  * $N_{in}(v)$: ノード $v$ への入リンクを持つノード集合
  * $d_{out}(u)$: ノード $u$ の出次数

    
    
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    
    print("=== PageRank アルゴリズムの実装 ===\n")
    
    def pagerank_custom(G, d=0.85, max_iter=100, tol=1e-6):
        """
        PageRankの自作実装
    
        Parameters:
        -----------
        G : NetworkX graph
            有向グラフ
        d : float
            ダンピング係数（デフォルト: 0.85）
        max_iter : int
            最大反復回数
        tol : float
            収束判定の閾値
    
        Returns:
        --------
        dict : ノードごとのPageRankスコア
        """
        N = len(G.nodes())
        nodes = list(G.nodes())
    
        # PageRankの初期化（全ノードに均等に分配）
        pr = {node: 1.0 / N for node in nodes}
    
        print(f"パラメータ:")
        print(f"  ノード数: {N}")
        print(f"  ダンピング係数: {d}")
        print(f"  最大反復回数: {max_iter}")
        print(f"  収束閾値: {tol}\n")
    
        for iteration in range(max_iter):
            pr_new = {}
    
            for node in nodes:
                # ランダムジャンプの寄与
                rank = (1 - d) / N
    
                # 入リンクからの寄与
                for neighbor in G.predecessors(node):
                    out_degree = G.out_degree(neighbor)
                    if out_degree > 0:
                        rank += d * pr[neighbor] / out_degree
    
                pr_new[node] = rank
    
            # 収束判定
            diff = sum(abs(pr_new[node] - pr[node]) for node in nodes)
    
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"反復 {iteration + 1:3d}: 差分 = {diff:.6f}")
    
            if diff < tol:
                print(f"\n収束しました（反復回数: {iteration + 1}）\n")
                break
    
            pr = pr_new
    
        # 正規化（合計が1になるように）
        total = sum(pr.values())
        pr = {node: score / total for node, score in pr.items()}
    
        return pr
    
    # テストグラフの作成
    print("=" * 60)
    print("簡単な有向グラフでのテスト")
    print("=" * 60 + "\n")
    
    G_simple = nx.DiGraph()
    G_simple.add_edges_from([
        ('A', 'B'), ('A', 'C'),
        ('B', 'C'), ('B', 'D'),
        ('C', 'A'), ('D', 'C')
    ])
    
    print("グラフ構造:")
    for node in G_simple.nodes():
        out_neighbors = list(G_simple.successors(node))
        in_neighbors = list(G_simple.predecessors(node))
        print(f"  {node}: 出リンク → {out_neighbors}, 入リンク ← {in_neighbors}")
    
    print("\n")
    
    # 自作PageRankの実行
    pr_custom = pagerank_custom(G_simple, d=0.85)
    
    print("自作PageRank結果:")
    for node, score in sorted(pr_custom.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {score:.4f}")
    
    # NetworkXのPageRankと比較
    pr_nx = nx.pagerank(G_simple, alpha=0.85)
    print("\nNetworkX PageRank結果:")
    for node, score in sorted(pr_nx.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {score:.4f}")
    
    print("\n差分:")
    for node in G_simple.nodes():
        diff = abs(pr_custom[node] - pr_nx[node])
        print(f"  {node}: {diff:.6f}")
    
    # より大きなグラフでのテスト
    print("\n" + "=" * 60)
    print("スケールフリーネットワークでのテスト")
    print("=" * 60 + "\n")
    
    # Barabási-Albert モデル（スケールフリーネットワーク）
    G_large = nx.barabasi_albert_graph(n=100, m=3, seed=42)
    G_large_directed = G_large.to_directed()
    
    pr_large = pagerank_custom(G_large_directed, d=0.85, max_iter=50)
    
    print("上位10ノードのPageRank:")
    sorted_pr = sorted(pr_large.items(), key=lambda x: x[1], reverse=True)
    for i, (node, score) in enumerate(sorted_pr[:10], 1):
        degree = G_large_directed.degree(node)
        print(f"  {i:2d}. ノード {node:3d}: PageRank = {score:.5f}, 次数 = {degree}")
    
    # PageRankと次数の関係を可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左：グラフ可視化（PageRankでノードサイズ調整）
    pos = nx.spring_layout(G_simple, seed=42)
    node_sizes = [pr_custom[node] * 5000 for node in G_simple.nodes()]
    node_colors = [pr_custom[node] for node in G_simple.nodes()]
    
    nx.draw(G_simple, pos, ax=axes[0],
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            with_labels=True,
            font_size=14,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2)
    axes[0].set_title('PageRankによるノードの重要度\n（大きいほど重要）',
                      fontsize=14, fontweight='bold')
    
    # 右：PageRankと次数の散布図
    degrees = [G_large_directed.degree(node) for node in G_large_directed.nodes()]
    pageranks = [pr_large[node] for node in G_large_directed.nodes()]
    
    axes[1].scatter(degrees, pageranks, alpha=0.6, s=50, color='steelblue')
    axes[1].set_xlabel('次数', fontsize=12)
    axes[1].set_ylabel('PageRank', fontsize=12)
    axes[1].set_title('次数とPageRankの関係\n（スケールフリーネットワーク）',
                      fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # 相関係数を計算
    correlation = np.corrcoef(degrees, pageranks)[0, 1]
    axes[1].text(0.05, 0.95, f'相関係数: {correlation:.3f}',
                transform=axes[1].transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pagerank_implementation.png', dpi=150, bbox_inches='tight')
    print("\n可視化を 'pagerank_implementation.png' に保存しました。")
    

* * *

## 1.4 グラフ埋め込み（Graph Embedding）

### なぜグラフ埋め込みが必要か

グラフ構造データを機械学習モデルで扱うには、ノードやグラフ全体を低次元の連続ベクトル空間に埋め込む必要があります：

> 「グラフ埋め込みは、グラフの構造情報を保持しながら、ノードを低次元ベクトル表現に変換する技術である。」

#### グラフ埋め込みの目標

  * グラフの構造的性質を保持
  * 近傍ノードが埋め込み空間でも近くに配置される
  * 類似した役割を持つノードが近くに配置される
  * 計算効率が高い

手法 | アプローチ | 特徴  
---|---|---  
**DeepWalk** | ランダムウォーク + Skip-gram | 無向グラフ、構造の保持  
**Node2Vec** | バイアス付きランダムウォーク | BFS/DFS の制御可能  
**LINE** | 1次・2次近接性の保持 | 大規模グラフに効率的  
**GCN** | グラフ畳み込み | ノード特徴量を活用  
**GraphSAGE** | 近傍サンプリング + 集約 | 帰納的学習が可能  
  
### DeepWalkとNode2Vec

#### DeepWalkの基本アイデア

  1. グラフ上でランダムウォークを実行してノード系列を生成
  2. この系列を「文」と見なし、Word2Vec（Skip-gram）を適用
  3. 近傍ノードが埋め込み空間でも近くになるように学習

目的関数：

$$ \max_f \sum_{v \in V} \log P(N(v) | f(v)) $$ 

ここで $N(v)$ はランダムウォークで訪れたノード $v$ の近傍、$f(v)$ はノード $v$ の埋め込みベクトル。

#### Node2Vecの改良点

Node2Vecは、ランダムウォークにバイアスを導入します：

  * **Return parameter ($p$)** ：直前のノードに戻る確率を制御
  * **In-out parameter ($q$)** ：BFS（幅優先探索）的 vs DFS（深さ優先探索）的な探索を制御

パラメータ設定 | 探索の特性 | 捉える構造  
---|---|---  
$p$ 低, $q$ 高 | BFS的（広い探索） | 局所的なコミュニティ構造  
$p$ 高, $q$ 低 | DFS的（深い探索） | グローバルな構造、役割  
      
    
    import networkx as nx
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    print("=== DeepWalk / Node2Vec 風の実装 ===\n")
    
    class SimpleNode2Vec:
        """
        Node2Vecの簡易実装
        """
        def __init__(self, G, embedding_dim=128, walk_length=80,
                     num_walks=10, p=1.0, q=1.0, window_size=10):
            """
            Parameters:
            -----------
            G : NetworkX graph
            embedding_dim : int
                埋め込み次元数
            walk_length : int
                1回のランダムウォークの長さ
            num_walks : int
                各ノードから開始するウォーク数
            p : float
                Return parameter（直前のノードに戻る確率の逆数）
            q : float
                In-out parameter（BFS vs DFS）
            window_size : int
                Skip-gramのウィンドウサイズ
            """
            self.G = G
            self.embedding_dim = embedding_dim
            self.walk_length = walk_length
            self.num_walks = num_walks
            self.p = p
            self.q = q
            self.window_size = window_size
            self.embeddings = None
    
        def _get_alias_edge(self, src, dst):
            """
            エッジの遷移確率を計算（Node2Vecのバイアス付き）
            """
            G = self.G
            p = self.p
            q = self.q
    
            unnormalized_probs = []
            for dst_nbr in G.neighbors(dst):
                if dst_nbr == src:
                    # 直前のノードに戻る
                    unnormalized_probs.append(1.0 / p)
                elif G.has_edge(dst_nbr, src):
                    # 距離1のノード（共通の隣接ノード）
                    unnormalized_probs.append(1.0)
                else:
                    # 距離2のノード
                    unnormalized_probs.append(1.0 / q)
    
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const
                              for u_prob in unnormalized_probs]
    
            return list(G.neighbors(dst)), normalized_probs
    
        def _node2vec_walk(self, start_node):
            """
            1回のNode2Vecランダムウォークを実行
            """
            G = self.G
            walk = [start_node]
    
            while len(walk) < self.walk_length:
                cur = walk[-1]
                cur_nbrs = list(G.neighbors(cur))
    
                if len(cur_nbrs) == 0:
                    break
    
                if len(walk) == 1:
                    # 最初のステップ：一様ランダムに選択
                    walk.append(np.random.choice(cur_nbrs))
                else:
                    # 2ステップ目以降：バイアス付き選択
                    prev = walk[-2]
                    neighbors, probs = self._get_alias_edge(prev, cur)
                    walk.append(np.random.choice(neighbors, p=probs))
    
            return walk
    
        def _generate_walks(self):
            """
            全ノードからランダムウォークを生成
            """
            walks = []
            nodes = list(self.G.nodes())
    
            print(f"ランダムウォークを生成中...")
            print(f"  各ノードから {self.num_walks} 回のウォーク")
            print(f"  ウォーク長: {self.walk_length}")
            print(f"  p={self.p}, q={self.q}\n")
    
            for walk_iter in range(self.num_walks):
                np.random.shuffle(nodes)
                for node in nodes:
                    walks.append(self._node2vec_walk(node))
    
            print(f"生成完了: {len(walks)} 個のウォーク\n")
            return walks
    
        def _skipgram_training(self, walks):
            """
            Skip-gram風の学習（簡易版）
            実際にはWord2VecやGensimを使用するが、ここでは説明用に簡略化
            """
            nodes = list(self.G.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
            # ランダムに初期化
            embeddings = np.random.randn(len(nodes), self.embedding_dim) * 0.01
    
            print("Skip-gram学習中（簡易版）...")
    
            # 共起カウント
            cooccurrence = defaultdict(lambda: defaultdict(int))
    
            for walk in walks:
                for i, node in enumerate(walk):
                    # ウィンドウ内のノードをカウント
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            cooccurrence[node][context_node] += 1
    
            # 簡易的な埋め込み学習（実際はネガティブサンプリングなど使用）
            # ここでは共起頻度の高いノード対が近くなるように調整
            learning_rate = 0.01
            epochs = 5
    
            for epoch in range(epochs):
                for node in nodes:
                    node_idx = node_to_idx[node]
    
                    for context_node, count in cooccurrence[node].items():
                        context_idx = node_to_idx[context_node]
    
                        # 簡易的な勾配更新（実際はもっと複雑）
                        diff = embeddings[context_idx] - embeddings[node_idx]
                        embeddings[node_idx] += learning_rate * count * diff * 0.01
    
                if (epoch + 1) % 2 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs} 完了")
    
            print("学習完了\n")
            return embeddings
    
        def fit(self):
            """
            Node2Vec埋め込みを学習
            """
            walks = self._generate_walks()
            self.embeddings = self._skipgram_training(walks)
            return self
    
        def get_embeddings(self):
            """
            学習済み埋め込みを取得
            """
            if self.embeddings is None:
                raise ValueError("まず fit() を実行してください")
    
            nodes = list(self.G.nodes())
            return {node: self.embeddings[i] for i, node in enumerate(nodes)}
    
    # テスト：Karate Clubグラフ
    print("=" * 60)
    print("Zachary Karate Club での Node2Vec")
    print("=" * 60 + "\n")
    
    G = nx.karate_club_graph()
    print(f"グラフ情報:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}\n")
    
    # Node2Vec学習
    model = SimpleNode2Vec(
        G,
        embedding_dim=64,
        walk_length=30,
        num_walks=10,
        p=1.0,  # DeepWalk相当
        q=1.0,
        window_size=5
    )
    
    model.fit()
    embeddings_dict = model.get_embeddings()
    
    # 埋め込みベクトルを配列に変換
    nodes = list(G.nodes())
    embeddings = np.array([embeddings_dict[node] for node in nodes])
    
    print(f"埋め込み結果:")
    print(f"  形状: {embeddings.shape}")
    print(f"  最初のノードの埋め込み（最初の5次元）: {embeddings[0, :5]}\n")
    
    # t-SNEで2次元に可視化
    print("t-SNEで2次元に削減中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings)
    print("完了\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左：元のグラフ
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[node]['club'] for node in G.nodes()]
    nx.draw(G, pos, ax=axes[0],
            node_color=node_colors,
            cmap='Set1',
            with_labels=True,
            node_size=400,
            font_size=8,
            edge_color='gray',
            alpha=0.7)
    axes[0].set_title('元のグラフ構造\n（色 = 所属クラブ）',
                      fontsize=14, fontweight='bold')
    
    # 右：埋め込み空間での可視化
    scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=node_colors, cmap='Set1',
                             s=200, alpha=0.7, edgecolors='black')
    
    for i, node in enumerate(nodes):
        axes[1].annotate(str(node),
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, ha='center', va='center')
    
    axes[1].set_xlabel('t-SNE 次元 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 次元 2', fontsize=12)
    axes[1].set_title('Node2Vec 埋め込み空間\n（t-SNE可視化）',
                      fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('node2vec_embedding.png', dpi=150, bbox_inches='tight')
    print("可視化を 'node2vec_embedding.png' に保存しました。")
    

* * *

## 1.5 実践：ソーシャルネットワーク分析

### コミュニティ検出

ソーシャルネットワークには、密に接続されたノードの集団（コミュニティ）が存在します。コミュニティ検出は、このような構造を自動的に発見する手法です。

#### 主なアルゴリズム

  * **Louvain法** ：モジュラリティを最大化
  * **Label Propagation** ：ラベルの伝播により分類
  * **Girvan-Newman法** ：媒介中心性の高いエッジを除去
  * **スペクトラルクラスタリング** ：ラプラシアン行列の固有値分解

#### モジュラリティ（Modularity）

コミュニティ構造の品質を評価する指標：

$$ Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j) $$ 

ここで：

  * $m$: エッジ数
  * $A_{ij}$: 隣接行列
  * $k_i$: ノード $i$ の次数
  * $c_i$: ノード $i$ の所属コミュニティ
  * $\delta(c_i, c_j)$: 同じコミュニティなら1、そうでなければ0

    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from networkx.algorithms import community
    
    print("=== ソーシャルネットワーク分析：コミュニティ検出 ===\n")
    
    # Zachary Karate Clubグラフ
    G = nx.karate_club_graph()
    
    print("Zachary Karate Club ネットワーク:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}")
    print("\n背景：空手クラブのメンバー間の友人関係ネットワーク")
    print("   後に指導者の対立により2つのグループに分裂\n")
    
    # 実際の分裂情報
    true_labels = [G.nodes[i]['club'] for i in G.nodes()]
    print(f"実際の分裂:")
    print(f"  Officer's club: {true_labels.count('Officer')} 人")
    print(f"  Mr. Hi's club: {true_labels.count('Mr. Hi')} 人\n")
    
    print("=" * 60)
    print("コミュニティ検出アルゴリズムの適用")
    print("=" * 60 + "\n")
    
    # 1. Louvain法
    print("1. Louvain法（モジュラリティ最大化）")
    print("-" * 40)
    communities_louvain = community.louvain_communities(G, seed=42)
    modularity_louvain = community.modularity(G, communities_louvain)
    
    print(f"検出されたコミュニティ数: {len(communities_louvain)}")
    for i, comm in enumerate(communities_louvain):
        print(f"  コミュニティ {i+1}: {len(comm)} ノード - {sorted(comm)}")
    print(f"モジュラリティ: {modularity_louvain:.4f}\n")
    
    # 2. Label Propagation
    print("2. Label Propagation（ラベル伝播）")
    print("-" * 40)
    communities_label_prop = community.label_propagation_communities(G)
    communities_label_prop = list(communities_label_prop)
    modularity_lp = community.modularity(G, communities_label_prop)
    
    print(f"検出されたコミュニティ数: {len(communities_label_prop)}")
    for i, comm in enumerate(communities_label_prop):
        print(f"  コミュニティ {i+1}: {len(comm)} ノード - {sorted(comm)}")
    print(f"モジュラリティ: {modularity_lp:.4f}\n")
    
    # 3. Girvan-Newman法
    print("3. Girvan-Newman法（媒介中心性ベース）")
    print("-" * 40)
    communities_gn_generator = community.girvan_newman(G)
    # 2つのコミュニティに分割
    communities_gn = next(communities_gn_generator)
    communities_gn = [set(c) for c in communities_gn]
    modularity_gn = community.modularity(G, communities_gn)
    
    print(f"検出されたコミュニティ数: {len(communities_gn)}")
    for i, comm in enumerate(communities_gn):
        print(f"  コミュニティ {i+1}: {len(comm)} ノード - {sorted(comm)}")
    print(f"モジュラリティ: {modularity_gn:.4f}\n")
    
    # 4. 実際の分裂との比較
    print("=" * 60)
    print("実際の分裂との比較")
    print("=" * 60 + "\n")
    
    def compare_with_ground_truth(communities, true_labels, G):
        """
        検出されたコミュニティと実際のラベルを比較
        """
        # コミュニティ番号を各ノードに割り当て
        node_to_community = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = comm_id
    
        # 混同行列的な分析
        from collections import Counter
    
        for comm_id, comm in enumerate(communities):
            labels_in_comm = [G.nodes[node]['club'] for node in comm]
            counter = Counter(labels_in_comm)
            print(f"検出コミュニティ {comm_id + 1}:")
            for club, count in counter.items():
                print(f"  {club}: {count} 人")
    
        # 正解率（多数決）
        correct = 0
        for node in G.nodes():
            comm_id = node_to_community[node]
            # このコミュニティの多数派ラベル
            labels_in_comm = [G.nodes[n]['club'] for n in communities[comm_id]]
            majority_label = Counter(labels_in_comm).most_common(1)[0][0]
    
            if G.nodes[node]['club'] == majority_label:
                correct += 1
    
        accuracy = correct / len(G.nodes())
        return accuracy
    
    print("Louvain法:")
    acc_louvain = compare_with_ground_truth(communities_louvain, true_labels, G)
    print(f"正解率: {acc_louvain:.2%}\n")
    
    print("Girvan-Newman法:")
    acc_gn = compare_with_ground_truth(communities_gn, true_labels, G)
    print(f"正解率: {acc_gn:.2%}\n")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    pos = nx.spring_layout(G, seed=42)
    
    # 実際の分裂
    true_community_map = {node: 0 if G.nodes[node]['club'] == 'Mr. Hi' else 1
                         for node in G.nodes()}
    nx.draw(G, pos, ax=axes[0, 0],
            node_color=[true_community_map[node] for node in G.nodes()],
            cmap='Set1',
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color='gray',
            alpha=0.7)
    axes[0, 0].set_title(f'実際の分裂\n（基準となるグランドトゥルース）',
                        fontsize=14, fontweight='bold')
    
    # Louvain法
    louvain_map = {}
    for comm_id, comm in enumerate(communities_louvain):
        for node in comm:
            louvain_map[node] = comm_id
    
    nx.draw(G, pos, ax=axes[0, 1],
            node_color=[louvain_map[node] for node in G.nodes()],
            cmap='Set2',
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color='gray',
            alpha=0.7)
    axes[0, 1].set_title(f'Louvain法\nモジュラリティ: {modularity_louvain:.3f}, 正解率: {acc_louvain:.1%}',
                        fontsize=14, fontweight='bold')
    
    # Label Propagation
    lp_map = {}
    for comm_id, comm in enumerate(communities_label_prop):
        for node in comm:
            lp_map[node] = comm_id
    
    nx.draw(G, pos, ax=axes[1, 0],
            node_color=[lp_map[node] for node in G.nodes()],
            cmap='Set3',
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color='gray',
            alpha=0.7)
    axes[1, 0].set_title(f'Label Propagation\nモジュラリティ: {modularity_lp:.3f}',
                        fontsize=14, fontweight='bold')
    
    # Girvan-Newman
    gn_map = {}
    for comm_id, comm in enumerate(communities_gn):
        for node in comm:
            gn_map[node] = comm_id
    
    nx.draw(G, pos, ax=axes[1, 1],
            node_color=[gn_map[node] for node in G.nodes()],
            cmap='Pastel1',
            with_labels=True,
            node_size=500,
            font_size=9,
            edge_color='gray',
            alpha=0.7)
    axes[1, 1].set_title(f'Girvan-Newman法\nモジュラリティ: {modularity_gn:.3f}, 正解率: {acc_gn:.1%}',
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('community_detection.png', dpi=150, bbox_inches='tight')
    print("可視化を 'community_detection.png' に保存しました。")
    

### リンク予測（Link Prediction）

リンク予測は、グラフ内で将来形成される可能性のあるエッジを予測するタスクです。推薦システムや知識グラフの補完に応用されます。

#### 基本的なアプローチ

  1. **共通隣接数** ：共通の友人が多いほど繋がりやすい
  2. **Jaccard係数** ：隣接ノード集合の類似度
  3. **Adamic-Adar指数** ：共通隣接の希少性を考慮
  4. **Preferential Attachment** ：次数の積（人気者同士が繋がる）

    
    
    import networkx as nx
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    import matplotlib.pyplot as plt
    
    print("=== リンク予測（Link Prediction）===\n")
    
    # グラフの準備
    G_original = nx.karate_club_graph()
    print(f"元のグラフ:")
    print(f"  ノード数: {G_original.number_of_nodes()}")
    print(f"  エッジ数: {G_original.number_of_edges()}\n")
    
    # エッジを訓練用とテスト用に分割
    print("エッジ分割:")
    edges = list(G_original.edges())
    np.random.seed(42)
    np.random.shuffle(edges)
    
    # 80%を訓練、20%をテスト
    split_idx = int(0.8 * len(edges))
    train_edges = edges[:split_idx]
    test_edges = edges[split_idx:]
    
    print(f"  訓練エッジ: {len(train_edges)}")
    print(f"  テストエッジ: {len(test_edges)}\n")
    
    # 訓練用グラフの作成
    G_train = nx.Graph()
    G_train.add_nodes_from(G_original.nodes())
    G_train.add_edges_from(train_edges)
    
    print(f"訓練グラフ:")
    print(f"  ノード数: {G_train.number_of_nodes()}")
    print(f"  エッジ数: {G_train.number_of_edges()}\n")
    
    # 負例の生成（存在しないエッジ）
    def generate_negative_edges(G, num_samples):
        """存在しないエッジをサンプリング"""
        non_edges = list(nx.non_edges(G))
        return [non_edges[i] for i in np.random.choice(
            len(non_edges), size=min(num_samples, len(non_edges)), replace=False
        )]
    
    negative_edges = generate_negative_edges(G_train, len(test_edges))
    print(f"負例エッジ: {len(negative_edges)}\n")
    
    # リンク予測スコアの計算
    print("=" * 60)
    print("リンク予測手法の評価")
    print("=" * 60 + "\n")
    
    def evaluate_link_prediction(scores, test_edges, negative_edges):
        """
        リンク予測の評価
        """
        # テストエッジ（正例）のスコア
        positive_scores = [scores.get((u, v), scores.get((v, u), 0))
                          for u, v in test_edges]
    
        # 負例エッジのスコア
        negative_scores = [scores.get((u, v), scores.get((v, u), 0))
                          for u, v in negative_edges]
    
        # ラベルとスコアを結合
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = positive_scores + negative_scores
    
        # 評価指標
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    
        return auc, ap, y_scores, y_true
    
    # 1. 共通隣接数（Common Neighbors）
    print("1. 共通隣接数（Common Neighbors）")
    print("-" * 40)
    cn_scores = {}
    for u, v in test_edges + negative_edges:
        common_neighbors = len(list(nx.common_neighbors(G_train, u, v)))
        cn_scores[(u, v)] = common_neighbors
    
    auc_cn, ap_cn, _, _ = evaluate_link_prediction(cn_scores, test_edges, negative_edges)
    print(f"AUC: {auc_cn:.4f}")
    print(f"Average Precision: {ap_cn:.4f}\n")
    
    # 2. Jaccard係数
    print("2. Jaccard係数")
    print("-" * 40)
    jaccard_scores = {}
    for u, v in test_edges + negative_edges:
        preds = list(nx.jaccard_coefficient(G_train, [(u, v)]))
        jaccard_scores[(u, v)] = preds[0][2] if preds else 0
    
    auc_jc, ap_jc, _, _ = evaluate_link_prediction(jaccard_scores, test_edges, negative_edges)
    print(f"AUC: {auc_jc:.4f}")
    print(f"Average Precision: {ap_jc:.4f}\n")
    
    # 3. Adamic-Adar指数
    print("3. Adamic-Adar指数")
    print("-" * 40)
    aa_scores = {}
    for u, v in test_edges + negative_edges:
        preds = list(nx.adamic_adar_index(G_train, [(u, v)]))
        aa_scores[(u, v)] = preds[0][2] if preds else 0
    
    auc_aa, ap_aa, _, _ = evaluate_link_prediction(aa_scores, test_edges, negative_edges)
    print(f"AUC: {auc_aa:.4f}")
    print(f"Average Precision: {ap_aa:.4f}\n")
    
    # 4. Preferential Attachment
    print("4. Preferential Attachment")
    print("-" * 40)
    pa_scores = {}
    for u, v in test_edges + negative_edges:
        preds = list(nx.preferential_attachment(G_train, [(u, v)]))
        pa_scores[(u, v)] = preds[0][2] if preds else 0
    
    auc_pa, ap_pa, _, _ = evaluate_link_prediction(pa_scores, test_edges, negative_edges)
    print(f"AUC: {auc_pa:.4f}")
    print(f"Average Precision: {ap_pa:.4f}\n")
    
    # 結果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 手法の比較
    methods = ['Common\nNeighbors', 'Jaccard', 'Adamic-\nAdar', 'Preferential\nAttachment']
    aucs = [auc_cn, auc_jc, auc_aa, auc_pa]
    aps = [ap_cn, ap_jc, ap_aa, ap_pa]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, aucs, width, label='AUC', color='steelblue')
    bars2 = axes[0].bar(x + width/2, aps, width, label='Average Precision', color='coral')
    
    axes[0].set_xlabel('手法', fontsize=12)
    axes[0].set_ylabel('スコア', fontsize=12)
    axes[0].set_title('リンク予測手法の性能比較', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=10)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    
    # 値をバーに表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    # グラフ可視化（訓練グラフとテストエッジ）
    pos = nx.spring_layout(G_original, seed=42)
    
    # 訓練エッジ
    nx.draw_networkx_edges(G_train, pos, ax=axes[1],
                           edge_color='gray', alpha=0.3, width=1)
    
    # テストエッジ（正例）
    nx.draw_networkx_edges(G_original, pos, ax=axes[1],
                           edgelist=test_edges,
                           edge_color='green', width=2, alpha=0.7,
                           label='テストエッジ（正例）')
    
    # ノード
    nx.draw_networkx_nodes(G_original, pos, ax=axes[1],
                          node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G_original, pos, ax=axes[1], font_size=8)
    
    axes[1].set_title('訓練グラフとテストエッジ', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('link_prediction.png', dpi=150, bbox_inches='tight')
    print("可視化を 'link_prediction.png' に保存しました。")
    

* * *

## 演習問題

演習1：グラフの基本操作

**問題** ：以下の条件を満たすグラフを作成し、その特徴を分析してください。

  * 10個のノードを持つランダムグラフ（Erdős-Rényi モデル、$p=0.3$）
  * グラフの次数分布をヒストグラムで可視化
  * 平均クラスタリング係数を計算
  * 最も中心性の高いノードを特定（次数中心性、媒介中心性、PageRank）

    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # ヒント
    G = nx.erdos_renyi_graph(n=10, p=0.3, seed=42)
    
    # 1. 次数分布の可視化
    degrees = [G.degree(node) for node in G.nodes()]
    # plt.hist(degrees, ...)
    
    # 2. クラスタリング係数
    avg_clustering = nx.average_clustering(G)
    
    # 3. 中心性指標
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    

演習2：グラフ表現の変換

**問題** ：以下のエッジリストで定義されるグラフを、隣接行列と隣接リストの両方で表現してください。
    
    
    エッジリスト: [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    

  * 隣接行列をNumPy配列として出力
  * 隣接リストを辞書として出力
  * どちらの表現がメモリ効率が良いか考察

演習3：PageRankの実装と検証

**問題** ：簡単な有向グラフに対してPageRankを手動計算し、NetworkXの結果と比較してください。
    
    
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('B', 'D')])
    
    # 1. 手動でPageRankを3回反復計算
    # 2. NetworkXのPageRankと比較
    # 3. 収束までに何回の反復が必要か調査
    

演習4：コミュニティ検出の比較

**問題** ：Louvain法とLabel Propagationで検出されたコミュニティの違いを分析してください。

  * 任意のグラフ（例：nx.karate_club_graph()）を使用
  * 各手法で検出されたコミュニティ数を比較
  * モジュラリティスコアを計算
  * 両手法の結果を可視化して違いを説明

演習5：Node2Vecのパラメータ調整

**問題** ：Node2Vecの $p$ と $q$ パラメータを変えて、埋め込み結果がどう変わるか観察してください。

  * $(p=0.5, q=2.0)$（BFS的）と $(p=2.0, q=0.5)$（DFS的）を試す
  * 各設定で生成された埋め込みをt-SNEで可視化
  * どちらがコミュニティ構造をより良く捉えているか評価

演習6：リンク予測の実装

**問題** ：独自のグラフに対してリンク予測を実装し、異なる手法の性能を比較してください。

  * 任意のグラフを選択（実データまたは生成）
  * エッジを訓練/テストに分割（80/20）
  * Common Neighbors、Jaccard、Adamic-Adarを実装
  * AUCとAverage Precisionで評価
  * どの手法が最も効果的か分析

* * *

## まとめ

この章では、グラフとグラフ表現学習の基礎を学びました：

  * ✅ **グラフ理論の基礎** ：ノード、エッジ、有向/無向グラフ、グラフの種類（木、DAG、完全グラフ）
  * ✅ **グラフの表現** ：隣接行列、隣接リスト、エッジリストの特徴と使い分け
  * ✅ **グラフの特徴量** ：次数、クラスタリング係数、中心性指標（次数、近接、媒介、固有ベクトル、PageRank）
  * ✅ **グラフ埋め込み** ：DeepWalk、Node2Vecによるランダムウォークベースの埋め込み手法
  * ✅ **実践的応用** ：コミュニティ検出、リンク予測、ソーシャルネットワーク分析

次章では、これらの基礎知識を踏まえて、**グラフニューラルネットワーク（GNN）** の基本原理を学びます。グラフ畳み込みの仕組み、メッセージパッシング、近傍集約など、GNNの核心的な概念を理解していきます。

> **重要なポイント** ：グラフ構造データは、画像や自然言語とは異なる特性を持ちます。可変サイズ、非ユークリッド構造、置換不変性などの性質を理解することが、GNNを効果的に使う鍵となります。
