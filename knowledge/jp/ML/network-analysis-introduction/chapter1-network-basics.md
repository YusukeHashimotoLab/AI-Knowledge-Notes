---
title: 第1章：ネットワーク分析の基礎
chapter_title: 第1章：ネットワーク分析の基礎
subtitle: グラフ理論から実践的なネットワークデータ分析まで
reading_time: 20-25分
difficulty: 初級
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ グラフ理論の基本概念（ノード、エッジ、有向・無向グラフ）を理解する
  * ✅ ネットワークの異なる表現方法とその変換方法を学ぶ
  * ✅ 基本的なネットワーク指標の計算と解釈ができる
  * ✅ 特殊なグラフ構造（ランダム、スモールワールド、スケールフリー）を理解する
  * ✅ NetworkXを使った実データの読み込みと基本分析を実践する

* * *

## 1.1 グラフ理論の基礎

### グラフの定義

**グラフ（Graph）** は、**ノード（Node/Vertex）** と**エッジ（Edge/Link）** の集合です。数学的には $G = (V, E)$ と表されます。

  * **ノード（$V$）** ：システムの構成要素（人、ウェブページ、タンパク質など）
  * **エッジ（$E$）** ：ノード間の関係（友人関係、リンク、相互作用など）

    
    
    ```mermaid
    graph LR
        A((ノード A)) --- B((ノード B))
        B --- C((ノード C))
        C --- A
        C --- D((ノード D))
    
        style A fill:#e3f2fd
        style B fill:#e3f2fd
        style C fill:#e3f2fd
        style D fill:#e3f2fd
    ```

### 有向グラフ vs 無向グラフ

種類 | 説明 | 例  
---|---|---  
**無向グラフ** | エッジに方向がない（対称的な関係） | 友人関係、共著関係、道路ネットワーク  
**有向グラフ** | エッジに方向がある（非対称的な関係） | Twitterフォロー、引用ネットワーク、Webリンク  
**重み付きグラフ** | エッジに重み（強度、距離など）がある | 交通ネットワーク、通信頻度、取引額  
  
### NetworkX基本操作
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 無向グラフの作成
    G = nx.Graph()
    
    # ノードの追加
    G.add_node(1)
    G.add_nodes_from([2, 3, 4, 5])
    
    # エッジの追加
    G.add_edge(1, 2)
    G.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5)])
    
    # 基本情報
    print("=== グラフの基本情報 ===")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
    print(f"ノード一覧: {list(G.nodes())}")
    print(f"エッジ一覧: {list(G.edges())}")
    
    # 可視化
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue',
            node_size=800, font_size=16, font_weight='bold')
    plt.title('基本的な無向グラフ')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === グラフの基本情報 ===
    ノード数: 5
    エッジ数: 5
    ノード一覧: [1, 2, 3, 4, 5]
    エッジ一覧: [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    

* * *

## 1.2 ネットワーク表現

### 3つの主要表現方法

ネットワークは複数の方法で表現できます。それぞれに長所・短所があり、用途に応じて使い分けます。

#### 1\. 隣接行列（Adjacency Matrix）

$n \times n$ の行列 $A$ で、$A_{ij} = 1$ ならノード $i$ と $j$ が接続されています。
    
    
    import numpy as np
    
    # グラフの作成
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    
    # 隣接行列の取得
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    print("=== 隣接行列 ===")
    print(adj_matrix)
    
    # 行列の可視化
    plt.figure(figsize=(6, 5))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('隣接行列の可視化')
    plt.xlabel('ノード')
    plt.ylabel('ノード')
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 隣接行列 ===
    [[0 1 1 0]
     [1 0 1 0]
     [1 1 0 1]
     [0 0 1 0]]
    

#### 2\. 隣接リスト（Adjacency List）

各ノードに接続されているノードのリストを保持します。疎なグラフで効率的です。
    
    
    # 隣接リストの取得
    adj_list = dict(G.adjacency())
    
    print("=== 隣接リスト ===")
    for node, neighbors in adj_list.items():
        neighbor_list = list(neighbors.keys())
        print(f"ノード {node}: {neighbor_list}")
    
    # メモリ効率の比較
    print(f"\n隣接行列のサイズ: {adj_matrix.nbytes} bytes")
    print(f"隣接リストのサイズ（推定）: {len(str(adj_list))} bytes")
    

**出力** ：
    
    
    === 隣接リスト ===
    ノード 0: [1, 2]
    ノード 1: [0, 2]
    ノード 2: [0, 1, 3]
    ノード 3: [2]
    
    隣接行列のサイズ: 128 bytes
    隣接リストのサイズ（推定）: 71 bytes
    

#### 3\. エッジリスト（Edge List）

すべてのエッジを (始点, 終点) のペアで記録します。データ保存に便利です。
    
    
    # エッジリストの作成
    edge_list = list(G.edges())
    
    print("=== エッジリスト ===")
    for i, (u, v) in enumerate(edge_list):
        print(f"エッジ {i}: {u} -- {v}")
    
    # 重み付きエッジリスト
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([
        (0, 1, 2.5), (0, 2, 1.8), (1, 2, 3.2), (2, 3, 1.5)
    ])
    
    print("\n=== 重み付きエッジリスト ===")
    for u, v, weight in G_weighted.edges(data='weight'):
        print(f"{u} -- {v}: 重み = {weight}")
    

**出力** ：
    
    
    === エッジリスト ===
    エッジ 0: 0 -- 1
    エッジ 1: 0 -- 2
    エッジ 2: 1 -- 2
    エッジ 3: 2 -- 3
    
    === 重み付きエッジリスト ===
    0 -- 1: 重み = 2.5
    0 -- 2: 重み = 1.8
    1 -- 2: 重み = 3.2
    2 -- 3: 重み = 1.5
    

### 表現方法の比較と変換

表現方法 | メモリ効率 | 隣接確認 | 全エッジ走査 | 適用場面  
---|---|---|---|---  
**隣接行列** | $O(n^2)$ | $O(1)$ | $O(n^2)$ | 密なグラフ、行列演算  
**隣接リスト** | $O(n + m)$ | $O(d)$ | $O(m)$ | 疎なグラフ、一般的な分析  
**エッジリスト** | $O(m)$ | $O(m)$ | $O(m)$ | データ保存、I/O操作  
  
> $n$ = ノード数、$m$ = エッジ数、$d$ = 次数（平均接続数）

* * *

## 1.3 基本的なネットワーク指標

### 次数（Degree）

ノードが持つエッジの数。ネットワーク内での「接続性」を表します。
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # サンプルグラフ
    G = nx.karate_club_graph()
    
    # 次数の計算
    degrees = dict(G.degree())
    
    print("=== 次数の統計 ===")
    print(f"平均次数: {np.mean(list(degrees.values())):.2f}")
    print(f"最大次数: {max(degrees.values())}")
    print(f"最小次数: {min(degrees.values())}")
    
    # 次数分布の可視化
    plt.figure(figsize=(12, 5))
    
    # ヒストグラム
    plt.subplot(1, 2, 1)
    plt.hist(list(degrees.values()), bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('次数')
    plt.ylabel('ノード数')
    plt.title('次数分布')
    plt.grid(True, alpha=0.3)
    
    # ネットワーク可視化（次数でノードサイズを変更）
    plt.subplot(1, 2, 2)
    node_sizes = [v * 50 for v in degrees.values()]
    nx.draw_spring(G, node_size=node_sizes, node_color='lightblue',
                   with_labels=True, font_size=8)
    plt.title('次数に応じたノードサイズ')
    
    plt.tight_layout()
    plt.show()
    

### 密度（Density）

実際のエッジ数と可能なエッジ数の比率。ネットワークの「密集度」を示します。

$$\text{Density} = \frac{2m}{n(n-1)}$$ （無向グラフの場合）
    
    
    # 密度の計算
    density = nx.density(G)
    
    print(f"\n=== ネットワーク密度 ===")
    print(f"密度: {density:.4f}")
    print(f"実際のエッジ数: {G.number_of_edges()}")
    
    n = G.number_of_nodes()
    max_edges = n * (n - 1) // 2
    print(f"最大可能エッジ数: {max_edges}")
    print(f"接続率: {(G.number_of_edges() / max_edges) * 100:.2f}%")
    

### 直径（Diameter）とクラスタリング係数
    
    
    # 直径（最長最短経路）
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"\n=== 経路指標 ===")
        print(f"直径: {diameter}")
        print(f"平均最短経路長: {avg_path_length:.2f}")
    
    # クラスタリング係数（三角形の密度）
    clustering = nx.clustering(G)
    avg_clustering = nx.average_clustering(G)
    
    print(f"\n=== クラスタリング係数 ===")
    print(f"平均クラスタリング係数: {avg_clustering:.4f}")
    print(f"上位5ノードのクラスタリング係数:")
    sorted_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
    for node, coef in sorted_clustering[:5]:
        print(f"  ノード {node}: {coef:.4f}")
    

**出力例** ：
    
    
    === 次数の統計 ===
    平均次数: 4.59
    最大次数: 17
    最小次数: 1
    
    === ネットワーク密度 ===
    密度: 0.1390
    実際のエッジ数: 78
    最大可能エッジ数: 561
    接続率: 13.90%
    
    === 経路指標 ===
    直径: 5
    平均最短経路長: 2.41
    
    === クラスタリング係数 ===
    平均クラスタリング係数: 0.5706
    上位5ノードのクラスタリング係数:
      ノード 4: 1.0000
      ノード 6: 1.0000
      ノード 7: 1.0000
      ノード 10: 1.0000
      ノード 11: 1.0000
    

* * *

## 1.4 特殊なグラフ構造

### ランダムグラフ（Erdős–Rényi モデル）

各ノードペア間にエッジが確率 $p$ で独立に存在するモデル。
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Erdős–Rényiランダムグラフ
    n = 30  # ノード数
    p = 0.1  # エッジ確率
    
    G_random = nx.erdos_renyi_graph(n, p, seed=42)
    
    print("=== Erdős–Rényiランダムグラフ ===")
    print(f"ノード数: {G_random.number_of_nodes()}")
    print(f"エッジ数: {G_random.number_of_edges()}")
    print(f"平均次数: {np.mean([d for n, d in G_random.degree()]):.2f}")
    print(f"クラスタリング係数: {nx.average_clustering(G_random):.4f}")
    
    # 可視化
    plt.figure(figsize=(8, 8))
    nx.draw_spring(G_random, node_color='lightcoral', node_size=300,
                   with_labels=True, font_size=8)
    plt.title(f'Erdős–Rényiランダムグラフ (n={n}, p={p})')
    plt.tight_layout()
    plt.show()
    

### スモールワールドネットワーク（Watts-Strogatz モデル）

高いクラスタリング係数と短い平均経路長を両立するモデル。実世界のネットワークで多く観察されます。
    
    
    # Watts-Strogatzスモールワールドネットワーク
    n = 30
    k = 4    # 各ノードが接続する近隣ノード数
    p = 0.3  # エッジの張り替え確率
    
    G_small_world = nx.watts_strogatz_graph(n, k, p, seed=42)
    
    print("\n=== Watts-Strogatzスモールワールドネットワーク ===")
    print(f"ノード数: {G_small_world.number_of_nodes()}")
    print(f"エッジ数: {G_small_world.number_of_edges()}")
    print(f"平均次数: {np.mean([d for n, d in G_small_world.degree()]):.2f}")
    print(f"クラスタリング係数: {nx.average_clustering(G_small_world):.4f}")
    if nx.is_connected(G_small_world):
        print(f"平均最短経路長: {nx.average_shortest_path_length(G_small_world):.2f}")
    
    # 可視化
    plt.figure(figsize=(8, 8))
    nx.draw_circular(G_small_world, node_color='lightgreen', node_size=300,
                     with_labels=True, font_size=8)
    plt.title(f'Watts-Strogatzスモールワールド (n={n}, k={k}, p={p})')
    plt.tight_layout()
    plt.show()
    

### スケールフリーネットワーク（Barabási–Albert モデル）

「富める者はより富む」（優先的選択）により生成されるネットワーク。次数分布がべき乗則に従います。
    
    
    # Barabási–Albertスケールフリーネットワーク
    n = 30
    m = 2  # 新ノードが接続するエッジ数
    
    G_scale_free = nx.barabasi_albert_graph(n, m, seed=42)
    
    print("\n=== Barabási–Albertスケールフリーネットワーク ===")
    print(f"ノード数: {G_scale_free.number_of_nodes()}")
    print(f"エッジ数: {G_scale_free.number_of_edges()}")
    print(f"平均次数: {np.mean([d for n, d in G_scale_free.degree()]):.2f}")
    print(f"最大次数: {max([d for n, d in G_scale_free.degree()])}")
    
    degrees = [d for n, d in G_scale_free.degree()]
    
    # 次数分布の可視化（対数スケール）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ネットワーク構造
    node_sizes = [d * 100 for d in degrees]
    nx.draw_spring(G_scale_free, ax=axes[0], node_size=node_sizes,
                   node_color='lightyellow', with_labels=True, font_size=8)
    axes[0].set_title(f'Barabási–Albertスケールフリー (n={n}, m={m})')
    
    # 次数分布（対数プロット）
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    axes[1].loglog(list(degree_counts.keys()), list(degree_counts.values()),
                   'bo-', markersize=8)
    axes[1].set_xlabel('次数 (log scale)')
    axes[1].set_ylabel('頻度 (log scale)')
    axes[1].set_title('次数分布（べき乗則）')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**3つのモデルの比較** ：

特性 | ランダムグラフ | スモールワールド | スケールフリー  
---|---|---|---  
**クラスタリング** | 低い | 高い | 中程度  
**平均経路長** | 短い | 短い | 短い  
**次数分布** | ポアソン分布 | ほぼ均一 | べき乗則  
**実世界の例** | 理論モデル | SNS、神経ネットワーク | WWW、引用ネットワーク  
  
* * *

## 1.5 実践: ネットワークデータの読み込みと基本分析

### CSVからのネットワーク構築
    
    
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # エッジリストCSVの作成（サンプル）
    edge_data = {
        'source': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'target': ['Bob', 'Charlie', 'Charlie', 'David', 'David', 'Eve'],
        'weight': [3, 2, 5, 1, 4, 2]
    }
    df = pd.DataFrame(edge_data)
    
    print("=== エッジリストデータ ===")
    print(df)
    
    # NetworkXグラフへの変換
    G = nx.from_pandas_edgelist(df, source='source', target='target',
                                edge_attr='weight', create_using=nx.Graph())
    
    print(f"\nノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
    print(f"ノード一覧: {list(G.nodes())}")
    
    # 可視化（重み付き）
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # エッジの重みで太さを変更
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=1000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=[w * 0.8 for w in weights],
                           alpha=0.6, edge_color='gray')
    
    # エッジのラベル（重み）
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title('重み付きネットワークの可視化')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

### 基本統計量の計算
    
    
    # 総合的なネットワーク分析
    print("\n=== ネットワーク基本統計 ===")
    
    # 次数統計
    degrees = dict(G.degree())
    print(f"平均次数: {np.mean(list(degrees.values())):.2f}")
    
    # 重み統計
    total_weight = sum([d['weight'] for u, v, d in G.edges(data=True)])
    avg_weight = total_weight / G.number_of_edges()
    print(f"総重み: {total_weight}")
    print(f"平均エッジ重み: {avg_weight:.2f}")
    
    # 接続性
    print(f"連結成分数: {nx.number_connected_components(G)}")
    print(f"ネットワーク密度: {nx.density(G):.4f}")
    
    if nx.is_connected(G):
        print(f"直径: {nx.diameter(G)}")
        print(f"平均最短経路長: {nx.average_shortest_path_length(G):.2f}")
    
    # 中心性指標
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    print("\n=== 中心性ランキング ===")
    print("次数中心性（上位3）:")
    for node, cent in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {node}: {cent:.4f}")
    
    print("\n媒介中心性（上位3）:")
    for node, cent in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {node}: {cent:.4f}")
    

**出力例** ：
    
    
    === エッジリストデータ ===
        source   target  weight
    0    Alice      Bob       3
    1    Alice  Charlie       2
    2      Bob  Charlie       5
    3      Bob    David       1
    4  Charlie    David       4
    5    David      Eve       2
    
    ノード数: 5
    エッジ数: 6
    
    === ネットワーク基本統計 ===
    平均次数: 2.40
    総重み: 17
    平均エッジ重み: 2.83
    連結成分数: 1
    ネットワーク密度: 0.6000
    直径: 3
    平均最短経路長: 1.60
    
    === 中心性ランキング ===
    次数中心性（上位3）:
      Bob: 0.7500
      Charlie: 0.7500
      David: 0.7500
    
    媒介中心性（上位3）:
      Charlie: 0.4000
      Bob: 0.4000
      David: 0.2667
    

### GraphMLフォーマットでの保存と読み込み
    
    
    # ネットワークの保存
    output_file = 'sample_network.graphml'
    nx.write_graphml(G, output_file)
    print(f"\nネットワークを {output_file} に保存しました")
    
    # 保存したネットワークの読み込み
    G_loaded = nx.read_graphml(output_file)
    print(f"読み込み完了: {G_loaded.number_of_nodes()} ノード, {G_loaded.number_of_edges()} エッジ")
    
    # 他のフォーマット
    # nx.write_edgelist(G, 'network.edgelist')  # エッジリスト
    # nx.write_gexf(G, 'network.gexf')  # GEXF（Gephi用）
    # nx.write_pajek(G, 'network.net')  # Pajek形式
    

* * *

## 本章のまとめ

### 学んだこと

  1. **グラフ理論の基礎**

     * ノードとエッジによるネットワーク表現
     * 有向・無向・重み付きグラフの違い
     * NetworkXによる基本操作
  2. **ネットワーク表現**

     * 隣接行列、隣接リスト、エッジリストの特徴
     * 用途に応じた表現方法の選択
     * 計算量とメモリ効率のトレードオフ
  3. **基本的なネットワーク指標**

     * 次数、密度、直径、クラスタリング係数
     * 中心性指標（次数中心性、媒介中心性）
     * ネットワーク特性の定量的評価
  4. **特殊なグラフ構造**

     * ランダムグラフ、スモールワールド、スケールフリー
     * 実世界ネットワークのモデル化
     * 各モデルの特徴と応用例
  5. **実践的なデータ分析**

     * CSVやGraphMLからのデータ読み込み
     * 基本統計量の計算と解釈
     * 可視化による洞察の獲得

### 次の章へ

第2章では、**中心性指標とコミュニティ検出** を学びます：

  * 高度な中心性指標（固有ベクトル中心性、PageRank）
  * コミュニティ検出アルゴリズム
  * モジュラリティ最適化
  * 階層的クラスタリング
  * 実データでのコミュニティ分析

* * *
