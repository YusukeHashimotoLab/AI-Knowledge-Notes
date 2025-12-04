---
title: 第4章：ネットワーク可視化ツール
chapter_title: 第4章：ネットワーク可視化ツール
subtitle: NetworkX、PyVis、igraph、Gephiを活用した効果的なグラフ可視化
---

🌐 JP | [🇬🇧 EN](<../../../en/ML/network-analysis-introduction/chapter4-visualization-tools.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[機械学習](<../../ML/index.html>)›[Network Analysis](<../../ML/network-analysis-introduction/index.html>)›Chapter 4

## 1\. NetworkX可視化

### 1.1 Matplotlibとの統合

NetworkXはmatplotlibと完全に統合されており、静的なグラフ可視化に適しています。
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # グラフ作成
    G = nx.karate_club_graph()
    
    # 基本的な可視化
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    plt.title('空手クラブネットワーク')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('karate_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 1.2 レイアウトアルゴリズム

適切なレイアウト選択はネットワーク構造の理解に不可欠です。
    
    
    import numpy as np
    
    # 各種レイアウトの比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    layouts = {
        'Spring': nx.spring_layout(G, k=0.3, iterations=50),
        'Circular': nx.circular_layout(G),
        'Kamada-Kawai': nx.kamada_kawai_layout(G),
        'Spectral': nx.spectral_layout(G)
    }
    
    for ax, (name, pos) in zip(axes.flat, layouts.items()):
        nx.draw(G, pos, ax=ax, node_color='lightblue',
                node_size=300, with_labels=True, font_size=8)
        ax.set_title(f'{name} Layout', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('layout_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # レイアウトアルゴリズムの選択基準
    # - Spring: 汎用的、力学的バランス（O(n²)）
    # - Circular: 対称性の可視化（O(n)）
    # - Kamada-Kawai: より正確な距離表現（O(n³)）
    # - Spectral: コミュニティ構造の強調（O(n²)）
    

### 1.3 ノード・エッジのカスタマイズ

ネットワーク属性を視覚的に表現することで、データの洞察を深めます。
    
    
    # 次数中心性に基づくカスタマイズ
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # ノードサイズ: 次数中心性
    node_sizes = [v * 3000 for v in degree_centrality.values()]
    
    # ノード色: 媒介中心性
    node_colors = list(betweenness_centrality.values())
    
    # エッジ幅: 重み（この例では次数の積）
    edge_weights = [G.degree(u) * G.degree(v) * 0.1
                    for u, v in G.edges()]
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # 描画
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, cmap='YlOrRd',
                           alpha=0.9, edgecolors='black', linewidths=1.5)
    
    nx.draw_networkx_edges(G, pos, width=edge_weights,
                           alpha=0.5, edge_color='gray')
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title('中心性メトリクスによるネットワーク可視化',
              fontsize=16, fontweight='bold')
    plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'),
                 label='媒介中心性', ax=plt.gca())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('customized_network.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可視化のベストプラクティス
    # 1. ノードサイズ: 重要度（中心性）を表現
    # 2. ノード色: カテゴリまたは連続値を表現
    # 3. エッジ幅: 関係の強さを表現
    # 4. レイアウト: データの性質に応じて選択
    

## 2\. 高度な可視化ライブラリ

### 2.1 PyVisによるインタラクティブ可視化

PyVisはインタラクティブなネットワーク可視化をHTML形式で生成します。
    
    
    from pyvis.network import Network
    import networkx as nx
    
    # PyVisネットワークの作成
    net = Network(height='750px', width='100%', bgcolor='#222222',
                  font_color='white', notebook=True)
    
    # NetworkXグラフからインポート
    G = nx.karate_club_graph()
    
    # コミュニティ検出
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    
    # ノードに色とサイズを設定
    for node in G.nodes():
        # コミュニティごとに色を変更
        color = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][community_map[node]]
        # 次数に応じてサイズを変更
        size = G.degree(node) * 3
        net.add_node(node, label=str(node), color=color, size=size,
                     title=f'Node {node}  
    Degree: {G.degree(node)}')
    
    # エッジを追加
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # 物理シミュレーション設定
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    # HTML出力
    net.save_graph('interactive_network.html')
    print("インタラクティブグラフをinteractive_network.htmlに保存しました")
    

### 2.2 Plotlyによるインタラクティブグラフ

Plotlyは高度なカスタマイズが可能なインタラクティブグラフを提供します。
    
    
    import plotly.graph_objects as go
    
    # レイアウト計算
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # エッジのトレース作成
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # ノードのトレース作成
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'ノード {node}  
    次数: {G.degree(node)}')
        node_sizes.append(G.degree(node) * 5)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=node_sizes,
            color=[G.degree(node) for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='ノード次数',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2, color='white')))
    
    # 図の作成
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Plotlyインタラクティブネットワーク',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(240,240,240,0.9)'))
    
    fig.write_html('plotly_network.html')
    fig.show()
    

### 2.3 大規模ネットワークの可視化

10,000ノード以上の大規模グラフには特殊なアプローチが必要です。
    
    
    # 大規模グラフの効率的な可視化
    def visualize_large_network(G, max_nodes=5000, sample_method='degree'):
        """
        大規模ネットワークをサンプリングして可視化
    
        Parameters:
        - G: NetworkXグラフ
        - max_nodes: 表示する最大ノード数
        - sample_method: 'degree', 'random', 'pagerank'
        """
        if len(G.nodes()) > max_nodes:
            print(f"ノード数 {len(G.nodes())} → {max_nodes} にサンプリング")
    
            if sample_method == 'degree':
                # 次数の高いノードを優先的に選択
                top_nodes = sorted(G.degree(), key=lambda x: x[1],
                                 reverse=True)[:max_nodes]
                nodes_to_keep = [n for n, d in top_nodes]
            elif sample_method == 'pagerank':
                # PageRankの高いノードを選択
                pr = nx.pagerank(G)
                top_nodes = sorted(pr.items(), key=lambda x: x[1],
                                 reverse=True)[:max_nodes]
                nodes_to_keep = [n for n, p in top_nodes]
            else:  # random
                import random
                nodes_to_keep = random.sample(list(G.nodes()), max_nodes)
    
            G_sample = G.subgraph(nodes_to_keep).copy()
        else:
            G_sample = G
    
        # 可視化
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G_sample, k=1/np.sqrt(len(G_sample.nodes())),
                              iterations=20)
    
        degree_centrality = nx.degree_centrality(G_sample)
        node_sizes = [v * 1000 for v in degree_centrality.values()]
    
        nx.draw_networkx(G_sample, pos,
                         node_size=node_sizes,
                         node_color=list(degree_centrality.values()),
                         cmap='viridis',
                         with_labels=False,
                         alpha=0.7,
                         edge_color='gray',
                         width=0.5)
    
        plt.title(f'サンプリングされたネットワーク ({len(G_sample.nodes())} ノード)',
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('large_network_sampled.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 使用例
    G_large = nx.barabasi_albert_graph(10000, 3, seed=42)
    visualize_large_network(G_large, max_nodes=500, sample_method='pagerank')
    

## 3\. igraphによる高速分析

### 3.1 igraph vs NetworkX

igraphはC言語で実装されており、大規模グラフの高速処理に優れています。

特徴 | NetworkX | igraph  
---|---|---  
実装言語 | Python | C (Pythonバインディング)  
速度 | 中程度 | 高速（10-100倍）  
メモリ効率 | 標準 | 効率的  
学習曲線 | 緩やか | やや急  
エコシステム | 豊富（matplotlib等） | 独自の可視化  
適用範囲 | 中小規模（~10K ノード） | 大規模（100K+ ノード）  
  
### 3.2 高速アルゴリズム実装
    
    
    import igraph as ig
    import time
    
    # NetworkXグラフをigraphに変換
    def nx_to_igraph(G_nx):
        """NetworkXグラフをigraphに変換"""
        G_ig = ig.Graph()
        G_ig.add_vertices(list(G_nx.nodes()))
        G_ig.add_edges(list(G_nx.edges()))
        return G_ig
    
    # パフォーマンス比較
    G_nx = nx.barabasi_albert_graph(5000, 3, seed=42)
    G_ig = nx_to_igraph(G_nx)
    
    # NetworkX: PageRank
    start = time.time()
    pr_nx = nx.pagerank(G_nx)
    time_nx = time.time() - start
    
    # igraph: PageRank
    start = time.time()
    pr_ig = G_ig.pagerank()
    time_ig = time.time() - start
    
    print(f"NetworkX PageRank: {time_nx:.4f}秒")
    print(f"igraph PageRank: {time_ig:.4f}秒")
    print(f"高速化率: {time_nx/time_ig:.2f}x")
    
    # igraphによるコミュニティ検出
    start = time.time()
    communities = G_ig.community_multilevel()
    time_community = time.time() - start
    
    print(f"\nigraphコミュニティ検出: {time_community:.4f}秒")
    print(f"検出されたコミュニティ数: {len(communities)}")
    print(f"モジュラリティ: {communities.modularity:.4f}")
    

### 3.3 大規模グラフ処理
    
    
    # igraphを使った大規模グラフの効率的な処理
    def analyze_large_graph_igraph(n_nodes=100000, m_edges=3):
        """大規模グラフの効率的な分析"""
        print(f"グラフ生成中: {n_nodes}ノード...")
        G = ig.Graph.Barabasi(n_nodes, m_edges)
    
        print("中心性メトリクスの計算中...")
        start = time.time()
    
        # 各種中心性の計算
        degree = G.degree()
        betweenness = G.betweenness()
        closeness = G.closeness()
        pagerank = G.pagerank()
    
        calc_time = time.time() - start
        print(f"計算時間: {calc_time:.2f}秒")
    
        # コミュニティ検出
        print("コミュニティ検出中...")
        start = time.time()
        communities = G.community_multilevel()
        comm_time = time.time() - start
        print(f"検出時間: {comm_time:.2f}秒")
        print(f"コミュニティ数: {len(communities)}")
    
        # 可視化（サンプリング）
        print("可視化用サンプリング中...")
        # 上位500ノードを選択
        top_nodes = sorted(range(len(pagerank)),
                          key=lambda i: pagerank[i], reverse=True)[:500]
        G_sample = G.subgraph(top_nodes)
    
        # igraph可視化
        visual_style = {
            "vertex_size": [pagerank[i] * 1000 for i in top_nodes],
            "vertex_color": [communities.membership[i] for i in top_nodes],
            "vertex_label": None,
            "edge_width": 0.5,
            "edge_color": "#cccccc",
            "layout": G_sample.layout_fruchterman_reingold()
        }
    
        ig.plot(G_sample,
                "large_graph_igraph.png",
                bbox=(1200, 1200),
                **visual_style)
    
        print("可視化完了: large_graph_igraph.png")
    
        return {
            'nodes': n_nodes,
            'edges': G.ecount(),
            'calc_time': calc_time,
            'comm_time': comm_time,
            'communities': len(communities),
            'modularity': communities.modularity
        }
    
    # 実行
    results = analyze_large_graph_igraph(n_nodes=100000, m_edges=3)
    print(f"\n結果サマリー: {results}")
    

## 4\. Gephi入門

### 4.1 Gephiの特徴

Gephiはインタラクティブな可視化とネットワーク探索のための強力なデスクトップアプリケーションです。

> **Gephiの主な利点：**
> 
>   * リアルタイムの視覚的フィードバック
>   * 高度なレイアウトアルゴリズム（ForceAtlas2等）
>   * 統計分析とフィルタリング機能
>   * 高品質な出力（出版用グラフィックス）
>   * プラグインエコシステム
> 

### 4.2 データのエクスポート/インポート
    
    
    # NetworkXからGephi形式へのエクスポート
    import networkx as nx
    
    # サンプルグラフの作成と属性追加
    G = nx.karate_club_graph()
    
    # ノード属性の追加
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    communities = nx.community.greedy_modularity_communities(G)
    
    # コミュニティIDをノードに追加
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_centrality[node]
        G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
        G.nodes[node]['community'] = community_map[node]
        G.nodes[node]['label'] = f'Node_{node}'
    
    # エッジ属性の追加
    for u, v in G.edges():
        G[u][v]['weight'] = G.degree(u) + G.degree(v)
    
    # GEXF形式でエクスポート（Gephi推奨形式）
    nx.write_gexf(G, 'network_for_gephi.gexf')
    print("GEXFファイル作成完了: network_for_gephi.gexf")
    
    # GraphML形式でもエクスポート可能
    nx.write_graphml(G, 'network_for_gephi.graphml')
    print("GraphMLファイル作成完了: network_for_gephi.graphml")
    
    # CSVエッジリスト形式（シンプルな方法）
    import pandas as pd
    
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'Source': u,
            'Target': v,
            'Weight': data.get('weight', 1)
        })
    
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv('edges.csv', index=False)
    
    # ノードリストCSV
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            'Id': node,
            'Label': data.get('label', str(node)),
            'Community': data.get('community', 0),
            'Degree_Centrality': data.get('degree_centrality', 0),
            'Betweenness_Centrality': data.get('betweenness_centrality', 0)
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv('nodes.csv', index=False)
    
    print("CSVファイル作成完了: edges.csv, nodes.csv")
    

### 4.3 可視化のベストプラクティス

Gephiワークフロー推奨手順

  1. **データインポート** : File → Open → GEXF/GraphMLファイルを選択
  2. **統計計算** : Statistics パネルで以下を実行 
     * Average Degree
     * Network Diameter
     * Modularity（コミュニティ検出）
     * PageRank
  3. **レイアウト適用** : Layout パネルで ForceAtlas2 を選択 
     * Scaling: 2.0-10.0（グラフサイズに応じて）
     * Gravity: 1.0
     * Prevent Overlap: チェック
  4. **視覚的調整** : Appearance パネルで設定 
     * ノードサイズ: Ranking → Degree/PageRank
     * ノード色: Partition → Modularity Class
     * ラベル: Size = ノードサイズに比例
  5. **エクスポート** : Preview → Export → PNG/PDF (300+ DPI推奨)

## 5\. 実践：大規模ネットワークの可視化

### 5.1 サンプリング手法
    
    
    # 様々なサンプリング手法の実装
    class NetworkSampler:
        """大規模ネットワークのサンプリングクラス"""
    
        @staticmethod
        def random_node_sampling(G, sample_size):
            """ランダムノードサンプリング"""
            import random
            nodes = random.sample(list(G.nodes()),
                                min(sample_size, len(G.nodes())))
            return G.subgraph(nodes).copy()
    
        @staticmethod
        def random_edge_sampling(G, sample_ratio=0.1):
            """ランダムエッジサンプリング"""
            import random
            n_edges = int(len(G.edges()) * sample_ratio)
            edges = random.sample(list(G.edges()), n_edges)
            H = nx.Graph()
            H.add_edges_from(edges)
            return H
    
        @staticmethod
        def induced_subgraph_sampling(G, sample_size):
            """誘導部分グラフサンプリング（重要ノード優先）"""
            # PageRankで重要ノードを選択
            pr = nx.pagerank(G)
            top_nodes = sorted(pr.items(), key=lambda x: x[1],
                              reverse=True)[:sample_size]
            nodes = [n for n, _ in top_nodes]
            return G.subgraph(nodes).copy()
    
        @staticmethod
        def snowball_sampling(G, seed_nodes, k=2):
            """スノーボールサンプリング（k-hopネイバーフッド）"""
            sampled_nodes = set(seed_nodes)
            for _ in range(k):
                new_nodes = set()
                for node in sampled_nodes:
                    new_nodes.update(G.neighbors(node))
                sampled_nodes.update(new_nodes)
            return G.subgraph(sampled_nodes).copy()
    
        @staticmethod
        def forest_fire_sampling(G, sample_size, p=0.4):
            """Forest Fireサンプリング"""
            import random
            sampled_nodes = set()
            queue = [random.choice(list(G.nodes()))]
    
            while len(sampled_nodes) < sample_size and queue:
                current = queue.pop(0)
                if current not in sampled_nodes:
                    sampled_nodes.add(current)
                    neighbors = list(G.neighbors(current))
                    # 確率pで隣接ノードを追加
                    n_select = int(len(neighbors) * p)
                    queue.extend(random.sample(neighbors,
                                             min(n_select, len(neighbors))))
    
            return G.subgraph(sampled_nodes).copy()
    
    # サンプリング手法の比較
    G_large = nx.barabasi_albert_graph(10000, 3, seed=42)
    sample_size = 500
    
    samplers = {
        'Random Node': NetworkSampler.random_node_sampling(G_large, sample_size),
        'Induced (PageRank)': NetworkSampler.induced_subgraph_sampling(G_large, sample_size),
        'Snowball (k=2)': NetworkSampler.snowball_sampling(
            G_large, [0, 1, 2], k=2),
        'Forest Fire': NetworkSampler.forest_fire_sampling(
            G_large, sample_size, p=0.4)
    }
    
    # 各サンプリング手法の特性を比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for ax, (name, G_sample) in zip(axes.flat, samplers.items()):
        pos = nx.spring_layout(G_sample, k=0.5, iterations=20)
        degree_centrality = nx.degree_centrality(G_sample)
        node_sizes = [v * 500 for v in degree_centrality.values()]
    
        nx.draw_networkx(G_sample, pos, ax=ax,
                         node_size=node_sizes,
                         node_color=list(degree_centrality.values()),
                         cmap='viridis',
                         with_labels=False,
                         alpha=0.7,
                         edge_color='gray',
                         width=0.5)
    
        # 統計情報
        density = nx.density(G_sample)
        avg_degree = sum(dict(G_sample.degree()).values()) / len(G_sample.nodes())
    
        ax.set_title(f'{name}\nノード: {len(G_sample.nodes())}, '
                    f'密度: {density:.4f}, 平均次数: {avg_degree:.2f}',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 5.2 階層的可視化
    
    
    # 階層的ネットワーク可視化
    def hierarchical_visualization(G, threshold_degree=10):
        """
        階層的可視化：重要ノードとその周辺を段階的に表示
        """
        # レベル1: 高次数ノード（ハブ）
        high_degree_nodes = [n for n, d in G.degree() if d >= threshold_degree]
    
        # レベル2: ハブの直接的な隣接ノード
        level2_nodes = set()
        for hub in high_degree_nodes:
            level2_nodes.update(G.neighbors(hub))
        level2_nodes = list(level2_nodes - set(high_degree_nodes))
    
        # レベル3: その他のノード（サンプリング）
        remaining = set(G.nodes()) - set(high_degree_nodes) - set(level2_nodes)
        import random
        level3_nodes = random.sample(list(remaining),
                                    min(100, len(remaining)))
    
        # 階層的レイアウト
        fig = plt.figure(figsize=(18, 6))
    
        # レベル1可視化
        ax1 = plt.subplot(131)
        G1 = G.subgraph(high_degree_nodes).copy()
        pos1 = nx.spring_layout(G1, k=1, seed=42)
        nx.draw_networkx(G1, pos1, ax=ax1,
                         node_color='red', node_size=500,
                         with_labels=True, font_size=8)
        ax1.set_title(f'レベル1: ハブノード ({len(high_degree_nodes)})',
                      fontsize=14, fontweight='bold')
        ax1.axis('off')
    
        # レベル2可視化
        ax2 = plt.subplot(132)
        G2 = G.subgraph(high_degree_nodes + level2_nodes).copy()
        pos2 = nx.spring_layout(G2, k=0.5, seed=42)
        node_colors = ['red' if n in high_degree_nodes else 'lightblue'
                       for n in G2.nodes()]
        node_sizes = [500 if n in high_degree_nodes else 200
                      for n in G2.nodes()]
        nx.draw_networkx(G2, pos2, ax=ax2,
                         node_color=node_colors, node_size=node_sizes,
                         with_labels=False)
        ax2.set_title(f'レベル2: +直接隣接 ({len(G2.nodes())})',
                      fontsize=14, fontweight='bold')
        ax2.axis('off')
    
        # レベル3可視化
        ax3 = plt.subplot(133)
        all_nodes = high_degree_nodes + level2_nodes + level3_nodes
        G3 = G.subgraph(all_nodes).copy()
        pos3 = nx.spring_layout(G3, k=0.3, seed=42)
        node_colors = ['red' if n in high_degree_nodes
                       else 'lightblue' if n in level2_nodes
                       else 'lightgreen' for n in G3.nodes()]
        node_sizes = [500 if n in high_degree_nodes
                      else 200 if n in level2_nodes
                      else 100 for n in G3.nodes()]
        nx.draw_networkx(G3, pos3, ax=ax3,
                         node_color=node_colors, node_size=node_sizes,
                         with_labels=False, alpha=0.8)
        ax3.set_title(f'レベル3: +その他 ({len(G3.nodes())})',
                      fontsize=14, fontweight='bold')
        ax3.axis('off')
    
        plt.tight_layout()
        plt.savefig('hierarchical_viz.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 使用例
    G = nx.barabasi_albert_graph(1000, 3, seed=42)
    hierarchical_visualization(G, threshold_degree=20)
    

### 5.3 インタラクティブダッシュボード作成
    
    
    # Plotly Dashを使ったインタラクティブダッシュボード
    from dash import Dash, dcc, html, Input, Output
    import plotly.graph_objects as go
    import networkx as nx
    
    # グラフ作成
    G = nx.karate_club_graph()
    
    # ダッシュボードアプリケーション
    app = Dash(__name__)
    
    # レイアウト
    app.layout = html.Div([
        html.H1("ネットワーク可視化ダッシュボード",
                style={'textAlign': 'center'}),
    
        html.Div([
            html.Label("レイアウトアルゴリズム:"),
            dcc.Dropdown(
                id='layout-dropdown',
                options=[
                    {'label': 'Spring Layout', 'value': 'spring'},
                    {'label': 'Circular Layout', 'value': 'circular'},
                    {'label': 'Kamada-Kawai', 'value': 'kamada_kawai'},
                    {'label': 'Spectral Layout', 'value': 'spectral'}
                ],
                value='spring'
            )
        ], style={'width': '300px', 'margin': '20px'}),
    
        html.Div([
            html.Label("ノード色メトリクス:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': '次数中心性', 'value': 'degree'},
                    {'label': '媒介中心性', 'value': 'betweenness'},
                    {'label': '固有ベクトル中心性', 'value': 'eigenvector'},
                    {'label': 'PageRank', 'value': 'pagerank'}
                ],
                value='degree'
            )
        ], style={'width': '300px', 'margin': '20px'}),
    
        dcc.Graph(id='network-graph', style={'height': '800px'})
    ])
    
    # コールバック
    @app.callback(
        Output('network-graph', 'figure'),
        [Input('layout-dropdown', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_graph(layout_type, metric_type):
        # レイアウト計算
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=0.5, seed=42)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:  # spectral
            pos = nx.spectral_layout(G)
    
        # メトリクス計算
        if metric_type == 'degree':
            metric = nx.degree_centrality(G)
        elif metric_type == 'betweenness':
            metric = nx.betweenness_centrality(G)
        elif metric_type == 'eigenvector':
            metric = nx.eigenvector_centrality(G)
        else:  # pagerank
            metric = nx.pagerank(G)
    
        # エッジトレース
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
        # ノードトレース
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [f'ノード {node}  
    {metric_type}: {metric[node]:.4f}'
                     for node in G.nodes()]
    
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(n) for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=[metric[node] * 50 for node in G.nodes()],
                color=[metric[node] for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title=metric_type,
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
    
        # 図の作成
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'{layout_type.title()} Layout - {metric_type.title()}',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
    
        return fig
    
    # アプリケーション実行
    # if __name__ == '__main__':
    #     app.run_server(debug=True, port=8050)
    
    print("ダッシュボードコード準備完了")
    print("実行するには、最後の2行のコメントを外してください")
    

> **可視化ツール選択ガイドライン：**
> 
>   * **NetworkX + Matplotlib** : 静的可視化、論文・レポート用、小規模（~1K ノード）
>   * **PyVis** : 素早いインタラクティブ探索、プレゼンテーション用
>   * **Plotly** : カスタマイズ性の高いインタラクティブ可視化、ダッシュボード
>   * **igraph** : 大規模グラフの高速処理と分析（10K+ ノード）
>   * **Gephi** : 出版品質の可視化、詳細な視覚的探索、大規模グラフ（100K+ ノード）
> 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
