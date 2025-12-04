---
title: 第5章：ネットワーク分析の実践応用
chapter_title: 第5章：ネットワーク分析の実践応用
subtitle: 実世界の問題をネットワークで解決する - 総合プロジェクト
reading_time: 35-40分
difficulty: 中級〜上級
code_examples: 8
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ソーシャルネットワークでの影響力伝播と情報拡散を分析できる
  * ✅ 知識グラフとセマンティックネットワークを構築できる
  * ✅ 生物学的ネットワークを解析し、パスウェイを特定できる
  * ✅ グラフベースの推薦システムを実装できる
  * ✅ 実践的なネットワーク分析プロジェクトを設計・実行できる

* * *

## 5.1 ソーシャルネットワーク分析

### 影響力伝播モデル

**影響力伝播（Influence Propagation）** は、ソーシャルネットワーク上で情報や行動がどのように広がるかを研究する分野です。

#### Linear Threshold Model（線形閾値モデル）

各ノードには閾値があり、影響を受けた隣接ノードの割合が閾値を超えると活性化されます。
    
    
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    def linear_threshold_model(G, seed_nodes, thresholds=None, iterations=20):
        """
        Linear Threshold Modelによる影響力伝播シミュレーション
    
        Parameters:
        -----------
        G : NetworkX graph
            ソーシャルネットワーク
        seed_nodes : list
            初期の影響を受けたノード（シード）
        thresholds : dict
            各ノードの閾値（Noneの場合はランダム）
        iterations : int
            シミュレーションの最大イテレーション数
    
        Returns:
        --------
        history : list
            各イテレーションでの活性化ノード集合
        """
        # 閾値の初期化
        if thresholds is None:
            thresholds = {node: np.random.uniform(0.3, 0.7) for node in G.nodes()}
    
        # 初期状態
        active = set(seed_nodes)
        history = [active.copy()]
    
        for iteration in range(iterations):
            new_active = set()
    
            for node in G.nodes():
                if node in active:
                    continue
    
                # 隣接ノードからの影響を計算
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 0:
                    continue
    
                active_neighbors = sum(1 for n in neighbors if n in active)
                influence = active_neighbors / len(neighbors)
    
                # 閾値を超えたら活性化
                if influence >= thresholds[node]:
                    new_active.add(node)
    
            if len(new_active) == 0:
                break
    
            active.update(new_active)
            history.append(active.copy())
    
        return history
    
    # サンプルネットワーク作成
    np.random.seed(42)
    G = nx.watts_strogatz_graph(50, 6, 0.3)
    
    # シードノードの選択（中心性の高いノード）
    centrality = nx.degree_centrality(G)
    seed_nodes = sorted(centrality, key=centrality.get, reverse=True)[:3]
    
    # シミュレーション実行
    history = linear_threshold_model(G, seed_nodes)
    
    print("=== Linear Threshold Model シミュレーション ===")
    print(f"シードノード数: {len(seed_nodes)}")
    print(f"総ノード数: {G.number_of_nodes()}")
    print(f"影響拡散のイテレーション数: {len(history)}")
    print(f"\n各イテレーションでの活性化ノード数:")
    for i, active_set in enumerate(history):
        print(f"  イテレーション {i}: {len(active_set)}ノード ({len(active_set)/G.number_of_nodes()*100:.1f}%)")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    pos = nx.spring_layout(G, seed=42)
    
    snapshots = [0, 1, 2, 3, 4, len(history)-1]
    for idx, (ax, snap_idx) in enumerate(zip(axes.flat, snapshots)):
        if snap_idx >= len(history):
            snap_idx = len(history) - 1
    
        active_nodes = history[snap_idx]
        node_colors = ['red' if n in active_nodes else 'lightblue' for n in G.nodes()]
    
        nx.draw(G, pos, node_color=node_colors, node_size=100,
                edge_color='gray', alpha=0.6, ax=ax)
        ax.set_title(f'イテレーション {snap_idx}\n活性化: {len(active_nodes)}ノード',
                     fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

#### Independent Cascade Model（独立カスケードモデル）

活性化されたノードは、各隣接ノードを確率的に活性化しようと試みます。
    
    
    def independent_cascade_model(G, seed_nodes, prob=0.1, iterations=20):
        """
        Independent Cascade Modelによる影響力伝播シミュレーション
    
        Parameters:
        -----------
        G : NetworkX graph
            ソーシャルネットワーク
        seed_nodes : list
            初期の影響を受けたノード
        prob : float
            各エッジでの活性化確率
        iterations : int
            最大イテレーション数
    
        Returns:
        --------
        history : list
            各イテレーションでの活性化ノード集合
        """
        active = set(seed_nodes)
        newly_active = set(seed_nodes)
        history = [active.copy()]
    
        for iteration in range(iterations):
            next_active = set()
    
            for node in newly_active:
                for neighbor in G.neighbors(node):
                    if neighbor not in active:
                        # 確率的に活性化
                        if np.random.random() < prob:
                            next_active.add(neighbor)
    
            if len(next_active) == 0:
                break
    
            active.update(next_active)
            newly_active = next_active
            history.append(active.copy())
    
        return history
    
    # シミュレーション実行（複数回の平均）
    num_simulations = 100
    all_spreads = []
    
    for _ in range(num_simulations):
        history = independent_cascade_model(G, seed_nodes, prob=0.15)
        all_spreads.append(len(history[-1]))
    
    print("\n=== Independent Cascade Model シミュレーション ===")
    print(f"シミュレーション回数: {num_simulations}")
    print(f"影響拡散の統計:")
    print(f"  平均: {np.mean(all_spreads):.2f}ノード ({np.mean(all_spreads)/G.number_of_nodes()*100:.1f}%)")
    print(f"  標準偏差: {np.std(all_spreads):.2f}")
    print(f"  最小: {np.min(all_spreads)}ノード")
    print(f"  最大: {np.max(all_spreads)}ノード")
    
    # 分布の可視化
    plt.figure(figsize=(10, 6))
    plt.hist(all_spreads, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(all_spreads), color='red', linestyle='--',
                linewidth=2, label=f'平均: {np.mean(all_spreads):.2f}')
    plt.xlabel('影響を受けたノード数', fontsize=12)
    plt.ylabel('頻度', fontsize=12)
    plt.title('Independent Cascade Model: 影響拡散の分布', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

### インフルエンサー特定

効果的な情報拡散のため、影響力の高いノード（インフルエンサー）を特定します。
    
    
    def identify_influencers(G, k=5, method='degree'):
        """
        インフルエンサーの特定
    
        Parameters:
        -----------
        G : NetworkX graph
            ソーシャルネットワーク
        k : int
            特定するインフルエンサーの数
        method : str
            使用する中心性指標 ('degree', 'betweenness', 'closeness', 'pagerank')
    
        Returns:
        --------
        influencers : list
            上位kノードのリスト
        scores : dict
            各ノードのスコア
        """
        if method == 'degree':
            scores = nx.degree_centrality(G)
        elif method == 'betweenness':
            scores = nx.betweenness_centrality(G)
        elif method == 'closeness':
            scores = nx.closeness_centrality(G)
        elif method == 'pagerank':
            scores = nx.pagerank(G)
        else:
            raise ValueError(f"Unknown method: {method}")
    
        influencers = sorted(scores, key=scores.get, reverse=True)[:k]
        return influencers, scores
    
    # 実際のソーシャルネットワークで実験（Karate Clubデータ）
    G_karate = nx.karate_club_graph()
    
    # 異なる方法でインフルエンサーを特定
    methods = ['degree', 'betweenness', 'closeness', 'pagerank']
    k = 5
    
    print("\n=== インフルエンサー特定の比較 ===")
    print(f"ネットワーク: Karate Club ({G_karate.number_of_nodes()}ノード)")
    print(f"上位{k}のインフルエンサーを特定:\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    pos = nx.spring_layout(G_karate, seed=42)
    
    for ax, method in zip(axes.flat, methods):
        influencers, scores = identify_influencers(G_karate, k=k, method=method)
    
        print(f"{method.capitalize()}:")
        for i, node in enumerate(influencers):
            print(f"  {i+1}. ノード {node}: スコア = {scores[node]:.4f}")
        print()
    
        # 可視化
        node_colors = ['red' if n in influencers else 'lightblue' for n in G_karate.nodes()]
        node_sizes = [scores[n] * 2000 for n in G_karate.nodes()]
    
        nx.draw(G_karate, pos, node_color=node_colors, node_size=node_sizes,
                edge_color='gray', alpha=0.6, with_labels=True, ax=ax)
        ax.set_title(f'{method.capitalize()} Centrality\n(赤 = 上位{k}のインフルエンサー)',
                     fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 各方法での影響拡散を比較
    print("\n=== 各インフルエンサー選択方法の効果比較 ===")
    results = {}
    
    for method in methods:
        influencers, _ = identify_influencers(G_karate, k=3, method=method)
    
        # 100回シミュレーションして平均
        spreads = []
        for _ in range(100):
            history = independent_cascade_model(G_karate, influencers, prob=0.2)
            spreads.append(len(history[-1]))
    
        results[method] = {
            'mean': np.mean(spreads),
            'std': np.std(spreads)
        }
    
        print(f"{method.capitalize()}:")
        print(f"  平均影響ノード数: {results[method]['mean']:.2f} ± {results[method]['std']:.2f}")
        print(f"  カバレッジ: {results[method]['mean']/G_karate.number_of_nodes()*100:.1f}%")
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    methods_list = list(results.keys())
    means = [results[m]['mean'] for m in methods_list]
    stds = [results[m]['std'] for m in methods_list]
    
    plt.bar(methods_list, means, yerr=stds, alpha=0.7, capsize=5, edgecolor='black')
    plt.xlabel('中心性指標', fontsize=12)
    plt.ylabel('平均影響ノード数', fontsize=12)
    plt.title('インフルエンサー選択方法の効果比較', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    

* * *

## 5.2 知識グラフとセマンティックネットワーク

### 知識グラフの構築

**知識グラフ（Knowledge Graph）** は、エンティティとその関係を表現するネットワークです。
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    
    class KnowledgeGraph:
        """知識グラフの構築と管理"""
    
        def __init__(self):
            self.G = nx.DiGraph()
    
        def add_triple(self, subject, predicate, obj):
            """
            トリプル（主語-述語-目的語）を追加
    
            Parameters:
            -----------
            subject : str
                主語（エンティティ）
            predicate : str
                述語（関係）
            obj : str
                目的語（エンティティまたは値）
            """
            self.G.add_edge(subject, obj, relation=predicate)
    
        def query_relations(self, entity):
            """エンティティに関連する全ての関係を取得"""
            outgoing = [(entity, self.G[entity][neighbor]['relation'], neighbor)
                        for neighbor in self.G.successors(entity)]
            incoming = [(source, self.G[source][entity]['relation'], entity)
                        for source in self.G.predecessors(entity)]
            return {'outgoing': outgoing, 'incoming': incoming}
    
        def find_path(self, start, end, max_length=3):
            """2つのエンティティ間のパスを検索"""
            try:
                paths = list(nx.all_simple_paths(self.G, start, end, cutoff=max_length))
                return paths
            except nx.NetworkXNoPath:
                return []
    
        def visualize(self, figsize=(12, 8)):
            """知識グラフの可視化"""
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(self.G, k=2, seed=42)
    
            # ノード描画
            nx.draw_networkx_nodes(self.G, pos, node_color='lightblue',
                                   node_size=2000, alpha=0.9)
    
            # エッジ描画
            nx.draw_networkx_edges(self.G, pos, edge_color='gray',
                                   arrows=True, arrowsize=20, alpha=0.6,
                                   connectionstyle='arc3,rad=0.1')
    
            # ラベル描画
            nx.draw_networkx_labels(self.G, pos, font_size=10, font_weight='bold')
    
            # エッジラベル（関係）描画
            edge_labels = nx.get_edge_attributes(self.G, 'relation')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels, font_size=8)
    
            plt.title('Knowledge Graph', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # 映画に関する知識グラフの構築
    kg = KnowledgeGraph()
    
    # トリプルの追加
    triples = [
        # 映画とジャンル
        ('The Matrix', 'genre', 'Sci-Fi'),
        ('The Matrix', 'genre', 'Action'),
        ('Inception', 'genre', 'Sci-Fi'),
        ('Inception', 'genre', 'Thriller'),
    
        # 監督
        ('The Matrix', 'directed_by', 'Wachowski'),
        ('Inception', 'directed_by', 'Nolan'),
        ('Interstellar', 'directed_by', 'Nolan'),
    
        # 俳優
        ('The Matrix', 'starring', 'Keanu Reeves'),
        ('Inception', 'starring', 'Leonardo DiCaprio'),
        ('Interstellar', 'starring', 'Matthew McConaughey'),
    
        # 年代
        ('The Matrix', 'released_in', '1999'),
        ('Inception', 'released_in', '2010'),
        ('Interstellar', 'released_in', '2014'),
    ]
    
    for subject, predicate, obj in triples:
        kg.add_triple(subject, predicate, obj)
    
    print("=== 知識グラフの構築 ===")
    print(f"エンティティ数: {kg.G.number_of_nodes()}")
    print(f"関係数: {kg.G.number_of_edges()}")
    
    # エンティティのクエリ
    print("\n=== 'Inception'に関する関係 ===")
    relations = kg.query_relations('Inception')
    print("Outgoing relations:")
    for s, p, o in relations['outgoing']:
        print(f"  {s} --[{p}]--> {o}")
    
    # パス検索
    print("\n=== 'The Matrix'と'Interstellar'の関連性 ===")
    paths = kg.find_path('The Matrix', 'Interstellar')
    if paths:
        print(f"{len(paths)}個のパスを発見:")
        for i, path in enumerate(paths):
            print(f"  パス {i+1}: {' -> '.join(path)}")
    else:
        print("直接的なパスは見つかりませんでした")
    
    # 可視化
    kg.visualize()
    

### エンティティ関係抽出とNeo4j基礎

実際のテキストからエンティティと関係を抽出し、グラフデータベースに格納する例です。
    
    
    import spacy
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 簡易的なエンティティ関係抽出（実際はspaCyなどのNLPライブラリを使用）
    def extract_triples_simple(text):
        """
        簡易的なトリプル抽出
        （実際のプロジェクトではspaCyやBERTベースのモデルを使用）
        """
        # サンプルとして手動でトリプルを定義
        triples = [
            ('Albert Einstein', 'born_in', 'Germany'),
            ('Albert Einstein', 'developed', 'Theory of Relativity'),
            ('Albert Einstein', 'won', 'Nobel Prize'),
            ('Nobel Prize', 'awarded_for', 'Physics'),
            ('Theory of Relativity', 'is_type_of', 'Physics Theory'),
        ]
        return triples
    
    # Neo4j風のCypherクエリシミュレーション
    class SimpleGraphDB:
        """グラフデータベースの簡易実装（Neo4j風）"""
    
        def __init__(self):
            self.kg = KnowledgeGraph()
    
        def create_node(self, label, properties=None):
            """ノード作成"""
            node_id = f"{label}_{len(self.kg.G.nodes())}"
            self.kg.G.add_node(node_id, label=label, **properties if properties else {})
            return node_id
    
        def create_relationship(self, node1, node2, rel_type):
            """関係作成"""
            self.kg.add_triple(node1, rel_type, node2)
    
        def match(self, pattern):
            """
            パターンマッチング（簡易版）
            例: MATCH (a)-[r]->(b) WHERE a.label = 'Person'
            """
            results = []
            for u, v, data in self.kg.G.edges(data=True):
                u_label = self.kg.G.nodes[u].get('label', '')
                v_label = self.kg.G.nodes[v].get('label', '')
    
                if pattern.get('source_label') and u_label != pattern['source_label']:
                    continue
                if pattern.get('relation') and data['relation'] != pattern['relation']:
                    continue
                if pattern.get('target_label') and v_label != pattern['target_label']:
                    continue
    
                results.append((u, data['relation'], v))
    
            return results
    
    # グラフDBの使用例
    print("\n=== グラフデータベースの使用例 ===")
    db = SimpleGraphDB()
    
    # テキストからトリプル抽出
    text = "Albert Einstein was born in Germany and developed the Theory of Relativity."
    triples = extract_triples_simple(text)
    
    # グラフDBに格納
    for subject, predicate, obj in triples:
        db.create_relationship(subject, predicate, obj)
    
    print(f"格納されたトリプル数: {len(triples)}")
    print("\n全トリプル:")
    for s, p, o in triples:
        print(f"  ({s}) -[{p}]-> ({o})")
    
    # クエリ実行
    print("\n=== クエリ: 'Albert Einstein'の全関係 ===")
    pattern = {'source_label': None}  # 全ての関係
    einstein_relations = db.kg.query_relations('Albert Einstein')
    for s, p, o in einstein_relations['outgoing']:
        print(f"  {s} -[{p}]-> {o}")
    
    # 可視化
    db.kg.visualize(figsize=(14, 10))
    

> **実践のヒント** : 本番環境では、Neo4j、Amazon Neptune、またはTigerGraphなどの専用グラフデータベースを使用することで、大規模な知識グラフの効率的な管理とクエリが可能になります。

* * *

## 5.3 生物学的ネットワーク

### タンパク質相互作用ネットワーク（PPI）

**タンパク質相互作用ネットワーク（Protein-Protein Interaction Network）** は、細胞内のタンパク質間の相互作用を表現します。
    
    
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    def create_ppi_network(n_proteins=100, interaction_prob=0.05):
        """
        タンパク質相互作用ネットワークのシミュレーション
    
        Parameters:
        -----------
        n_proteins : int
            タンパク質の数
        interaction_prob : float
            相互作用の確率
    
        Returns:
        --------
        G : NetworkX graph
            PPIネットワーク
        """
        G = nx.erdos_renyi_graph(n_proteins, interaction_prob)
    
        # タンパク質の機能をランダムに割り当て
        functions = ['Metabolism', 'Signaling', 'Transport', 'Regulation', 'Structure']
        for node in G.nodes():
            G.nodes[node]['function'] = np.random.choice(functions)
            G.nodes[node]['expression'] = np.random.uniform(0, 10)  # 発現レベル
    
        return G
    
    def identify_protein_complexes(G, min_size=3):
        """
        タンパク質複合体の特定（クリーク検出）
    
        Parameters:
        -----------
        G : NetworkX graph
            PPIネットワーク
        min_size : int
            最小の複合体サイズ
    
        Returns:
        --------
        complexes : list
            検出されたタンパク質複合体
        """
        cliques = list(nx.find_cliques(G))
        complexes = [c for c in cliques if len(c) >= min_size]
        return complexes
    
    def analyze_hub_proteins(G, top_k=10):
        """
        ハブタンパク質（高次数ノード）の分析
    
        Parameters:
        -----------
        G : NetworkX graph
            PPIネットワーク
        top_k : int
            上位k個のハブ
    
        Returns:
        --------
        hubs : list
            ハブタンパク質のリスト
        """
        degrees = dict(G.degree())
        hubs = sorted(degrees, key=degrees.get, reverse=True)[:top_k]
        return hubs, degrees
    
    # PPIネットワークの作成
    np.random.seed(42)
    ppi = create_ppi_network(n_proteins=80, interaction_prob=0.08)
    
    print("=== タンパク質相互作用ネットワーク（PPI）===")
    print(f"タンパク質数: {ppi.number_of_nodes()}")
    print(f"相互作用数: {ppi.number_of_edges()}")
    print(f"平均次数: {np.mean([d for _, d in ppi.degree()]):.2f}")
    
    # タンパク質複合体の特定
    complexes = identify_protein_complexes(ppi, min_size=3)
    print(f"\n検出されたタンパク質複合体: {len(complexes)}個")
    print(f"複合体サイズ分布:")
    complex_sizes = [len(c) for c in complexes]
    for size in sorted(set(complex_sizes)):
        count = complex_sizes.count(size)
        print(f"  サイズ {size}: {count}個")
    
    # ハブタンパク質の分析
    hubs, degrees = analyze_hub_proteins(ppi, top_k=5)
    print(f"\n上位5つのハブタンパク質:")
    for i, hub in enumerate(hubs):
        func = ppi.nodes[hub]['function']
        deg = degrees[hub]
        print(f"  {i+1}. タンパク質 {hub}: 次数={deg}, 機能={func}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 全体ネットワーク
    pos = nx.spring_layout(ppi, seed=42)
    node_colors = [ppi.nodes[n]['expression'] for n in ppi.nodes()]
    node_sizes = [degrees[n] * 30 for n in ppi.nodes()]
    
    nx.draw(ppi, pos, node_color=node_colors, node_size=node_sizes,
            cmap='YlOrRd', edge_color='gray', alpha=0.7, ax=axes[0])
    axes[0].set_title('PPI Network (色 = 発現レベル, サイズ = 次数)', fontsize=12)
    
    # 最大のタンパク質複合体を強調
    if complexes:
        largest_complex = max(complexes, key=len)
        subgraph = ppi.subgraph(largest_complex)
        sub_pos = nx.spring_layout(subgraph, seed=42)
    
        nx.draw(subgraph, sub_pos, node_color='lightcoral', node_size=500,
                edge_color='black', width=2, with_labels=True, ax=axes[1])
        axes[1].set_title(f'最大のタンパク質複合体 (サイズ: {len(largest_complex)})',
                          fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

### 遺伝子制御ネットワークとパスウェイ解析

遺伝子制御ネットワークは、遺伝子間の制御関係を表現する有向グラフです。
    
    
    def create_gene_regulatory_network(n_genes=50, regulation_prob=0.1):
        """
        遺伝子制御ネットワークの生成
    
        Parameters:
        -----------
        n_genes : int
            遺伝子の数
        regulation_prob : float
            制御関係の確率
    
        Returns:
        --------
        G : NetworkX DiGraph
            遺伝子制御ネットワーク
        """
        G = nx.DiGraph()
    
        # 遺伝子ノードの追加
        for i in range(n_genes):
            G.add_node(f'Gene_{i}',
                       expression=np.random.uniform(0, 1),
                       type=np.random.choice(['TF', 'target']))  # TF = 転写因子
    
        # 制御関係の追加
        for i in range(n_genes):
            for j in range(n_genes):
                if i != j and np.random.random() < regulation_prob:
                    # activation (+1) or repression (-1)
                    regulation_type = np.random.choice([1, -1])
                    G.add_edge(f'Gene_{i}', f'Gene_{j}',
                              weight=regulation_type)
    
        return G
    
    def find_regulatory_motifs(G):
        """
        制御モチーフ（フィードバックループなど）の検出
    
        Parameters:
        -----------
        G : NetworkX DiGraph
            遺伝子制御ネットワーク
    
        Returns:
        --------
        motifs : dict
            検出されたモチーフ
        """
        motifs = {
            'feedback_loops': [],
            'feedforward_loops': [],
            'cascades': []
        }
    
        # フィードバックループ（サイクル）
        try:
            cycles = list(nx.simple_cycles(G))
            motifs['feedback_loops'] = [c for c in cycles if len(c) <= 4]
        except:
            pass
    
        # カスケード（長いパス）
        for node in G.nodes():
            descendants = nx.descendants(G, node)
            if len(descendants) >= 3:
                motifs['cascades'].append((node, len(descendants)))
    
        return motifs
    
    def pathway_enrichment_analysis(G, target_genes):
        """
        パスウェイ濃縮解析（簡易版）
    
        Parameters:
        -----------
        G : NetworkX DiGraph
            遺伝子制御ネットワーク
        target_genes : list
            注目している遺伝子のリスト
    
        Returns:
        --------
        enriched_regulators : list
            濃縮された上流制御因子
        """
        # 各遺伝子から標的遺伝子への到達可能性
        regulators = defaultdict(int)
    
        for target in target_genes:
            if target in G:
                # 上流の全ての制御因子を取得
                ancestors = nx.ancestors(G, target)
                for anc in ancestors:
                    regulators[anc] += 1
    
        # スコアでソート
        enriched = sorted(regulators.items(), key=lambda x: x[1], reverse=True)
        return enriched
    
    # 遺伝子制御ネットワークの作成
    np.random.seed(42)
    grn = create_gene_regulatory_network(n_genes=40, regulation_prob=0.12)
    
    print("\n=== 遺伝子制御ネットワーク（GRN）===")
    print(f"遺伝子数: {grn.number_of_nodes()}")
    print(f"制御関係数: {grn.number_of_edges()}")
    
    # 転写因子の特定
    tfs = [n for n in grn.nodes() if grn.nodes[n]['type'] == 'TF']
    print(f"転写因子数: {len(tfs)}")
    
    # モチーフ検出
    motifs = find_regulatory_motifs(grn)
    print(f"\n=== 制御モチーフ ===")
    print(f"フィードバックループ: {len(motifs['feedback_loops'])}個")
    if motifs['feedback_loops']:
        print(f"  例: {motifs['feedback_loops'][0]}")
    print(f"制御カスケード: {len(motifs['cascades'])}個")
    if motifs['cascades']:
        top_cascade = max(motifs['cascades'], key=lambda x: x[1])
        print(f"  最大カスケード: {top_cascade[0]} → {top_cascade[1]}個の下流遺伝子")
    
    # パスウェイ濃縮解析
    target_genes = list(grn.nodes())[:10]  # 最初の10遺伝子を標的とする
    enriched = pathway_enrichment_analysis(grn, target_genes)
    
    print(f"\n=== パスウェイ濃縮解析 ===")
    print(f"標的遺伝子数: {len(target_genes)}")
    print(f"上位5つの濃縮された制御因子:")
    for i, (reg, score) in enumerate(enriched[:5]):
        print(f"  {i+1}. {reg}: スコア={score}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 全体ネットワーク
    pos = nx.spring_layout(grn, seed=42, k=0.5)
    node_colors = ['red' if grn.nodes[n]['type'] == 'TF' else 'lightblue'
                   for n in grn.nodes()]
    
    # エッジの色（活性化=緑、抑制=赤）
    edge_colors = ['green' if grn[u][v]['weight'] > 0 else 'red'
                   for u, v in grn.edges()]
    
    nx.draw(grn, pos, node_color=node_colors, edge_color=edge_colors,
            node_size=300, alpha=0.7, arrows=True, arrowsize=10, ax=axes[0])
    axes[0].set_title('遺伝子制御ネットワーク\n(赤=転写因子, 緑=活性化, 赤=抑制)',
                      fontsize=12)
    
    # フィードバックループの強調表示
    if motifs['feedback_loops']:
        loop = motifs['feedback_loops'][0]
        subgraph = grn.subgraph(loop)
        sub_pos = nx.circular_layout(subgraph)
    
        edge_colors_sub = ['green' if subgraph[u][v]['weight'] > 0 else 'red'
                           for u, v in subgraph.edges()]
    
        nx.draw(subgraph, sub_pos, node_color='yellow', edge_color=edge_colors_sub,
                node_size=800, width=2, with_labels=True, arrows=True,
                arrowsize=20, ax=axes[1])
        axes[1].set_title(f'フィードバックループの例\n({len(loop)}遺伝子)',
                          fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

* * *

## 5.4 推薦システムへの応用

### グラフベース協調フィルタリング

ユーザーとアイテムを二部グラフとして表現し、推薦を行います。
    
    
    import networkx as nx
    from networkx.algorithms import bipartite
    import numpy as np
    import matplotlib.pyplot as plt
    
    class GraphBasedRecommender:
        """グラフベースの推薦システム"""
    
        def __init__(self):
            self.G = nx.Graph()
            self.users = set()
            self.items = set()
    
        def add_interaction(self, user, item, rating=1.0):
            """ユーザーとアイテムの相互作用を追加"""
            self.G.add_node(user, bipartite=0)  # ユーザー側
            self.G.add_node(item, bipartite=1)  # アイテム側
            self.G.add_edge(user, item, weight=rating)
            self.users.add(user)
            self.items.add(item)
    
        def recommend_by_neighbors(self, user, top_k=5):
            """
            隣接ユーザーベースの推薦
    
            1. ユーザーの隣接アイテムを取得
            2. それらのアイテムに接続している他のユーザーを見つける
            3. それらのユーザーが好むアイテムを推薦
            """
            if user not in self.G:
                return []
    
            # ユーザーが既に評価したアイテム
            user_items = set(self.G.neighbors(user))
    
            # 候補アイテムのスコア計算
            candidate_scores = {}
    
            for item in user_items:
                # このアイテムを好む他のユーザー
                for other_user in self.G.neighbors(item):
                    if other_user == user:
                        continue
    
                    # 他のユーザーが好むアイテム
                    for candidate in self.G.neighbors(other_user):
                        if candidate not in user_items:
                            # スコア = 経路の重みの積
                            score = (self.G[user][item]['weight'] *
                                    self.G[other_user][candidate]['weight'])
                            candidate_scores[candidate] = candidate_scores.get(candidate, 0) + score
    
            # スコアでソート
            recommendations = sorted(candidate_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:top_k]
            return recommendations
    
        def recommend_by_random_walk(self, user, walk_length=10, n_walks=100, top_k=5):
            """
            ランダムウォークベースの推薦
    
            Parameters:
            -----------
            user : str
                対象ユーザー
            walk_length : int
                各ウォークの長さ
            n_walks : int
                実行するウォーク数
            top_k : int
                推薦するアイテム数
            """
            if user not in self.G:
                return []
    
            user_items = set(self.G.neighbors(user))
            visit_counts = {}
    
            for _ in range(n_walks):
                current = user
                for step in range(walk_length):
                    neighbors = list(self.G.neighbors(current))
                    if not neighbors:
                        break
    
                    # 重み付きランダム選択
                    weights = [self.G[current][n]['weight'] for n in neighbors]
                    weights = np.array(weights) / sum(weights)
                    current = np.random.choice(neighbors, p=weights)
    
                    # アイテムの訪問をカウント
                    if current in self.items and current not in user_items:
                        visit_counts[current] = visit_counts.get(current, 0) + 1
    
            # 訪問回数でソート
            recommendations = sorted(visit_counts.items(),
                                    key=lambda x: x[1], reverse=True)[:top_k]
            return recommendations
    
    # サンプルデータの作成（MovieLens風）
    np.random.seed(42)
    recommender = GraphBasedRecommender()
    
    # ユーザーとアイテム
    users = [f'User_{i}' for i in range(20)]
    items = [f'Movie_{i}' for i in range(15)]
    
    # ランダムな評価データ生成
    for user in users:
        n_ratings = np.random.randint(3, 8)
        rated_items = np.random.choice(items, n_ratings, replace=False)
        for item in rated_items:
            rating = np.random.uniform(3, 5)
            recommender.add_interaction(user, item, rating)
    
    print("=== グラフベース推薦システム ===")
    print(f"ユーザー数: {len(recommender.users)}")
    print(f"アイテム数: {len(recommender.items)}")
    print(f"評価数: {recommender.G.number_of_edges()}")
    
    # 推薦の実行
    test_user = 'User_0'
    print(f"\n=== '{test_user}'への推薦 ===")
    
    # 既に評価したアイテム
    user_items = list(recommender.G.neighbors(test_user))
    print(f"既に評価したアイテム: {user_items}")
    
    # 隣接ベースの推薦
    neighbor_recs = recommender.recommend_by_neighbors(test_user, top_k=5)
    print(f"\n隣接ベースの推薦:")
    for i, (item, score) in enumerate(neighbor_recs):
        print(f"  {i+1}. {item}: スコア={score:.3f}")
    
    # ランダムウォークベースの推薦
    rw_recs = recommender.recommend_by_random_walk(test_user, top_k=5)
    print(f"\nランダムウォークベースの推薦:")
    for i, (item, count) in enumerate(rw_recs):
        print(f"  {i+1}. {item}: 訪問回数={count}")
    
    # 可視化
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(recommender.G, seed=42, k=2)
    
    # ユーザーとアイテムで色分け
    node_colors = ['lightblue' if n in recommender.users else 'lightcoral'
                   for n in recommender.G.nodes()]
    
    # 対象ユーザーを強調
    node_colors = ['yellow' if n == test_user else c
                   for n, c in zip(recommender.G.nodes(), node_colors)]
    
    nx.draw(recommender.G, pos, node_color=node_colors, node_size=300,
            alpha=0.7, edge_color='gray', with_labels=True, font_size=8)
    plt.title('二部グラフ推薦システム\n(青=ユーザー, 赤=アイテム, 黄=対象ユーザー)',
              fontsize=14)
    plt.tight_layout()
    plt.show()
    

### ネットワーク埋め込み（Node2Vec）による推薦

Node2Vecを使ってノードをベクトル空間に埋め込み、類似度ベースの推薦を行います。
    
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    def simple_node2vec(G, dimensions=64, walk_length=10, num_walks=80, p=1, q=1):
        """
        Node2Vecの簡易実装（概念実証用）
    
        Parameters:
        -----------
        G : NetworkX graph
            ネットワーク
        dimensions : int
            埋め込みの次元数
        walk_length : int
            ランダムウォークの長さ
        num_walks : int
            各ノードからのウォーク数
        p : float
            Return parameter
        q : float
            In-out parameter
    
        Returns:
        --------
        embeddings : dict
            ノードの埋め込みベクトル
        """
        # ランダムウォークの生成
        walks = []
        nodes = list(G.nodes())
    
        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                for _ in range(walk_length - 1):
                    current = walk[-1]
                    neighbors = list(G.neighbors(current))
                    if neighbors:
                        walk.append(np.random.choice(neighbors))
                    else:
                        break
                walks.append(walk)
    
        # 簡易的な埋め込み（実際はSkip-gramなどを使用）
        # ここでは共起行列ベースの簡易実装
        node_to_id = {node: i for i, node in enumerate(G.nodes())}
        cooccurrence = np.zeros((len(nodes), len(nodes)))
    
        for walk in walks:
            for i, node in enumerate(walk):
                for j in range(max(0, i-2), min(len(walk), i+3)):
                    if i != j:
                        cooccurrence[node_to_id[node]][node_to_id[walk[j]]] += 1
    
        # SVDで次元削減
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(dimensions, len(nodes)-1))
        embeddings_matrix = svd.fit_transform(cooccurrence)
    
        embeddings = {node: embeddings_matrix[node_to_id[node]]
                      for node in G.nodes()}
    
        return embeddings
    
    # Node2Vecの適用
    print("\n=== Node2Vec埋め込み ===")
    embeddings = simple_node2vec(recommender.G, dimensions=32, walk_length=10, num_walks=50)
    print(f"埋め込み次元: {len(list(embeddings.values())[0])}")
    
    # 埋め込みベースの推薦
    def recommend_by_embedding(embeddings, user, items, user_items, top_k=5):
        """埋め込みベースの推薦"""
        user_emb = embeddings[user].reshape(1, -1)
    
        # 未評価アイテムのみを候補とする
        candidate_items = [item for item in items if item not in user_items]
    
        if not candidate_items:
            return []
    
        # 類似度計算
        similarities = {}
        for item in candidate_items:
            item_emb = embeddings[item].reshape(1, -1)
            sim = cosine_similarity(user_emb, item_emb)[0][0]
            similarities[item] = sim
    
        # スコアでソート
        recommendations = sorted(similarities.items(),
                                key=lambda x: x[1], reverse=True)[:top_k]
        return recommendations
    
    # 推薦実行
    user_items_set = set(recommender.G.neighbors(test_user))
    emb_recs = recommend_by_embedding(embeddings, test_user,
                                      recommender.items, user_items_set, top_k=5)
    
    print(f"\n=== Node2Vec埋め込みベースの推薦 ===")
    for i, (item, sim) in enumerate(emb_recs):
        print(f"  {i+1}. {item}: 類似度={sim:.3f}")
    
    # 埋め込みの可視化（t-SNE）
    from sklearn.manifold import TSNE
    
    print("\n=== 埋め込みの可視化 ===")
    node_list = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[n] for n in node_list])
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    plt.figure(figsize=(12, 8))
    user_indices = [i for i, n in enumerate(node_list) if n in recommender.users]
    item_indices = [i for i, n in enumerate(node_list) if n in recommender.items]
    
    plt.scatter(embeddings_2d[user_indices, 0], embeddings_2d[user_indices, 1],
                c='blue', label='Users', alpha=0.6, s=100)
    plt.scatter(embeddings_2d[item_indices, 0], embeddings_2d[item_indices, 1],
                c='red', label='Items', alpha=0.6, s=100)
    
    # 対象ユーザーを強調
    test_user_idx = node_list.index(test_user)
    plt.scatter(embeddings_2d[test_user_idx, 0], embeddings_2d[test_user_idx, 1],
                c='yellow', s=300, marker='*', edgecolors='black', linewidths=2,
                label='Target User')
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Node2Vec Embeddings (t-SNE可視化)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

* * *

## 5.5 実践プロジェクト: 総合ネットワーク分析

### プロジェクト: Twitterライクなソーシャルネットワークの分析

実践的なプロジェクトとして、ソーシャルメディアネットワークの包括的な分析を行います。

#### プロジェクト設計

**目標** ：ソーシャルネットワークを分析し、以下を実現する

  1. コミュニティ検出とユーザークラスタリング
  2. インフルエンサーの特定
  3. 情報拡散のシミュレーション
  4. 推薦システムの構築
  5. レポート作成とインサイト抽出

    
    
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    import pandas as pd
    
    class SocialNetworkAnalyzer:
        """ソーシャルネットワークの総合分析システム"""
    
        def __init__(self, name="Social Network"):
            self.name = name
            self.G = nx.DiGraph()  # 有向グラフ（フォロー関係）
            self.metrics = {}
            self.communities = None
            self.influencers = None
    
        def load_network(self, edges, user_attributes=None):
            """
            ネットワークデータの読み込み
    
            Parameters:
            -----------
            edges : list of tuples
                (follower, followee) のリスト
            user_attributes : dict
                ユーザー属性（オプション）
            """
            self.G.add_edges_from(edges)
    
            if user_attributes:
                for user, attrs in user_attributes.items():
                    if user in self.G:
                        for key, value in attrs.items():
                            self.G.nodes[user][key] = value
    
            print(f"✓ ネットワーク読み込み完了")
            print(f"  ユーザー数: {self.G.number_of_nodes()}")
            print(f"  フォロー関係数: {self.G.number_of_edges()}")
    
        def compute_basic_metrics(self):
            """基本的なネットワーク指標を計算"""
            # 次数（フォロワー数・フォロー数）
            in_degree = dict(self.G.in_degree())  # フォロワー数
            out_degree = dict(self.G.out_degree())  # フォロー数
    
            # 中心性指標
            pagerank = nx.pagerank(self.G)
    
            # 弱連結成分
            undirected = self.G.to_undirected()
            n_components = nx.number_connected_components(undirected)
    
            # クラスタリング係数
            clustering = nx.average_clustering(undirected)
    
            self.metrics = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'pagerank': pagerank,
                'n_components': n_components,
                'avg_clustering': clustering,
                'density': nx.density(self.G)
            }
    
            print(f"\n✓ 基本指標計算完了")
            print(f"  連結成分数: {n_components}")
            print(f"  平均クラスタリング係数: {clustering:.3f}")
            print(f"  密度: {self.metrics['density']:.4f}")
            print(f"  平均フォロワー数: {np.mean(list(in_degree.values())):.2f}")
            print(f"  平均フォロー数: {np.mean(list(out_degree.values())):.2f}")
    
        def detect_communities(self, method='louvain'):
            """コミュニティ検出"""
            undirected = self.G.to_undirected()
    
            if method == 'louvain':
                # Louvainアルゴリズム（NetworkXの標準機能）
                import networkx.algorithms.community as nx_comm
                self.communities = list(nx_comm.greedy_modularity_communities(undirected))
    
            print(f"\n✓ コミュニティ検出完了")
            print(f"  検出されたコミュニティ数: {len(self.communities)}")
            print(f"  コミュニティサイズ分布:")
            sizes = sorted([len(c) for c in self.communities], reverse=True)
            for i, size in enumerate(sizes[:5]):
                print(f"    コミュニティ {i+1}: {size}ユーザー")
    
        def identify_influencers(self, top_k=10):
            """インフルエンサーの特定"""
            # 複数の指標を組み合わせてスコアリング
            scores = {}
    
            for user in self.G.nodes():
                score = (
                    0.4 * self.metrics['in_degree'][user] +  # フォロワー数
                    0.3 * self.metrics['pagerank'][user] * 1000 +  # PageRank
                    0.3 * self.metrics['out_degree'][user]  # エンゲージメント
                )
                scores[user] = score
    
            self.influencers = sorted(scores, key=scores.get, reverse=True)[:top_k]
    
            print(f"\n✓ インフルエンサー特定完了")
            print(f"  上位{top_k}のインフルエンサー:")
            for i, user in enumerate(self.influencers):
                print(f"    {i+1}. {user}:")
                print(f"       フォロワー: {self.metrics['in_degree'][user]}")
                print(f"       フォロー: {self.metrics['out_degree'][user]}")
                print(f"       PageRank: {self.metrics['pagerank'][user]:.4f}")
    
        def simulate_information_diffusion(self, seed_users, prob=0.1, iterations=10):
            """情報拡散シミュレーション"""
            active = set(seed_users)
            history = [len(active)]
    
            for _ in range(iterations):
                new_active = set()
                for user in active:
                    for follower in self.G.predecessors(user):  # フォロワー
                        if follower not in active and np.random.random() < prob:
                            new_active.add(follower)
    
                if not new_active:
                    break
    
                active.update(new_active)
                history.append(len(active))
    
            print(f"\n✓ 情報拡散シミュレーション完了")
            print(f"  初期ユーザー数: {len(seed_users)}")
            print(f"  最終到達数: {history[-1]}ユーザー ({history[-1]/self.G.number_of_nodes()*100:.1f}%)")
    
            return history
    
        def generate_report(self):
            """分析レポートの生成"""
            report = f"""
    {'='*60}
    ソーシャルネットワーク分析レポート: {self.name}
    {'='*60}
    
    1. ネットワーク概要
       - ユーザー数: {self.G.number_of_nodes():,}
       - フォロー関係数: {self.G.number_of_edges():,}
       - ネットワーク密度: {self.metrics['density']:.4f}
       - 連結成分数: {self.metrics['n_components']}
    
    2. ユーザー行動
       - 平均フォロワー数: {np.mean(list(self.metrics['in_degree'].values())):.2f}
       - 平均フォロー数: {np.mean(list(self.metrics['out_degree'].values())):.2f}
       - 平均クラスタリング係数: {self.metrics['avg_clustering']:.3f}
    
    3. コミュニティ構造
       - コミュニティ数: {len(self.communities) if self.communities else 'N/A'}
       - 最大コミュニティサイズ: {max(len(c) for c in self.communities) if self.communities else 'N/A'}
    
    4. インフルエンサー
       - 上位インフルエンサー: {', '.join(self.influencers[:5]) if self.influencers else 'N/A'}
    
    5. 推奨事項
       - マーケティング対象: トップ5のインフルエンサーを活用
       - コミュニティ戦略: 各コミュニティに特化したコンテンツ作成
       - エンゲージメント向上: クラスタリング係数の高いユーザーをハブとして活用
    
    {'='*60}
            """
            return report
    
    # サンプルネットワークの生成とプロジェクト実行
    print("=== ソーシャルネットワーク分析プロジェクト ===\n")
    
    # データ生成（実際はTwitter APIなどから取得）
    np.random.seed(42)
    n_users = 100
    
    # スケールフリーネットワーク（現実的なソーシャルネットワーク）
    G_sample = nx.scale_free_graph(n_users, seed=42)
    edges = [(u, v) for u, v in G_sample.edges()]
    
    # ユーザー属性
    user_attributes = {
        i: {
            'join_date': f'2020-{np.random.randint(1, 13):02d}',
            'posts': np.random.randint(10, 1000),
            'active': np.random.choice([True, False], p=[0.7, 0.3])
        }
        for i in range(n_users)
    }
    
    # 分析実行
    analyzer = SocialNetworkAnalyzer(name="Twitter-like Network")
    analyzer.load_network(edges, user_attributes)
    analyzer.compute_basic_metrics()
    analyzer.detect_communities()
    analyzer.identify_influencers(top_k=10)
    
    # 情報拡散シミュレーション（トップインフルエンサーから開始）
    diffusion_history = analyzer.simulate_information_diffusion(
        seed_users=analyzer.influencers[:3],
        prob=0.15,
        iterations=10
    )
    
    # レポート生成
    report = analyzer.generate_report()
    print(report)
    
    # 可視化
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. ネットワーク全体図
    ax1 = fig.add_subplot(gs[0, :])
    pos = nx.spring_layout(analyzer.G, seed=42, k=0.5)
    node_colors = [analyzer.metrics['pagerank'][n] * 1000 for n in analyzer.G.nodes()]
    node_sizes = [analyzer.metrics['in_degree'][n] * 20 + 50 for n in analyzer.G.nodes()]
    nx.draw(analyzer.G, pos, node_color=node_colors, node_size=node_sizes,
            cmap='YlOrRd', edge_color='gray', alpha=0.6, arrows=False, ax=ax1)
    ax1.set_title('ソーシャルネットワーク全体図\n(色=PageRank, サイズ=フォロワー数)',
                  fontsize=14, fontweight='bold')
    
    # 2. 次数分布
    ax2 = fig.add_subplot(gs[1, 0])
    in_degrees = list(analyzer.metrics['in_degree'].values())
    ax2.hist(in_degrees, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('フォロワー数', fontsize=11)
    ax2.set_ylabel('ユーザー数', fontsize=11)
    ax2.set_title('フォロワー数の分布', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. コミュニティ可視化
    ax3 = fig.add_subplot(gs[1, 1])
    if analyzer.communities:
        community_map = {}
        for i, comm in enumerate(analyzer.communities):
            for node in comm:
                community_map[node] = i
    
        comm_colors = [community_map.get(n, 0) for n in analyzer.G.nodes()]
        nx.draw(analyzer.G, pos, node_color=comm_colors, node_size=100,
                cmap='tab10', edge_color='gray', alpha=0.6, arrows=False, ax=ax3)
        ax3.set_title(f'コミュニティ構造 ({len(analyzer.communities)}コミュニティ)',
                      fontsize=12)
    
    # 4. 情報拡散
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(range(len(diffusion_history)), diffusion_history,
             marker='o', linewidth=2, markersize=8)
    ax4.set_xlabel('イテレーション', fontsize=11)
    ax4.set_ylabel('到達ユーザー数', fontsize=11)
    ax4.set_title('情報拡散シミュレーション', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. インフルエンサー比較
    ax5 = fig.add_subplot(gs[2, 1])
    influencer_data = {
        'User': analyzer.influencers[:5],
        'Followers': [analyzer.metrics['in_degree'][u] for u in analyzer.influencers[:5]],
    }
    df_inf = pd.DataFrame(influencer_data)
    ax5.barh(df_inf['User'], df_inf['Followers'], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('フォロワー数', fontsize=11)
    ax5.set_title('トップ5インフルエンサー', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ プロジェクト完了!")
    print("  可視化、分析レポート、インサイトを生成しました。")
    

### ストーリーテリングとレポート作成

> **実践のヒント** : 分析結果を効果的に伝えるためのストーリーテリング
> 
>   1. **背景と目的** : なぜこの分析が必要か明確にする
>   2. **データの特徴** : ネットワークの基本統計を提示
>   3. **主要な発見** : 重要なインサイトを3-5個に絞る
>   4. **可視化** : 直感的で分かりやすいグラフを使用
>   5. **推奨事項** : 具体的なアクションプランを提示
>   6. **次のステップ** : さらなる分析の方向性を示す
> 

* * *

## 本章のまとめ

### 学んだこと

  1. **ソーシャルネットワーク分析**

     * Linear ThresholdモデルとIndependent Cascadeモデルによる影響力伝播
     * 複数の中心性指標を用いたインフルエンサー特定
     * 情報拡散のシミュレーションと予測
  2. **知識グラフ**

     * エンティティと関係のモデリング（トリプル構造）
     * グラフデータベースの基礎（Neo4j風のクエリ）
     * セマンティックネットワークの構築と活用
  3. **生物学的ネットワーク**

     * タンパク質相互作用ネットワークの解析
     * 遺伝子制御ネットワークとモチーフ検出
     * パスウェイ濃縮解析の基礎
  4. **推薦システム**

     * グラフベース協調フィルタリング
     * ランダムウォークを用いた推薦
     * Node2Vec埋め込みによる類似度推薦
  5. **実践プロジェクト**

     * 包括的なネットワーク分析パイプラインの設計
     * 複数の分析手法の統合
     * 効果的なレポーティングとストーリーテリング

### 実世界への応用

分野 | 応用例 | 主要技術  
---|---|---  
**マーケティング** | インフルエンサーマーケティング、バイラル戦略 | 影響力伝播、中心性分析  
**eコマース** | 商品推薦、クロスセル | グラフベース推薦、Node2Vec  
**製薬** | 創薬、疾患メカニズム解明 | PPI分析、パスウェイ解析  
**金融** | 不正検出、リスク分析 | 異常検出、コミュニティ検出  
**AI/NLP** | 知識ベース構築、質問応答システム | 知識グラフ、セマンティック検索  
  
### 次のステップ

ネットワーク分析をさらに深めるために：

  * **深層学習との統合** : Graph Neural Networks (GNN)、Graph Attention Networks (GAT)
  * **動的ネットワーク** : 時間発展するネットワークの分析
  * **大規模ネットワーク** : 分散処理、近似アルゴリズム
  * **因果推論** : ネットワークにおける因果関係の特定
  * **実データでの実践** : Kaggleコンペ、研究プロジェクト

* * *

## 参考文献

  1. Barabási, A. L. (2016). _Network Science_. Cambridge University Press.
  2. Newman, M. (2018). _Networks: An Introduction_ (2nd ed.). Oxford University Press.
  3. Easley, D., & Kleinberg, J. (2010). _Networks, Crowds, and Markets_. Cambridge University Press.
  4. Hamilton, W. L. (2020). _Graph Representation Learning_. Morgan & Claypool.
  5. Grover, A., & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. _KDD_.
