---
title: 第3章：コミュニティ検出
chapter_title: 第3章：コミュニティ検出
subtitle: モジュラリティ最適化とラベル伝播 - Louvain法、Label Propagationの理論と実装
reading_time: 25-30分
difficulty: 中級
code_examples: 8
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ コミュニティ検出の基本概念とモジュラリティを理解する
  * ✅ Louvain法の階層的最適化アルゴリズムを実装する
  * ✅ Label Propagationの原理と高速性の秘密を学ぶ
  * ✅ 主要なコミュニティ検出手法（Girvan-Newman、Infomap、スペクトラルクラスタリング）を比較する
  * ✅ 実データでのコミュニティ分析と可視化を実践する

## 1\. コミュニティ検出の基礎

### 1.1 コミュニティとは

ネットワークにおける**コミュニティ（Community）** とは、内部の結合が密で、外部との結合が疎なノードの集まりです。ソーシャルネットワークでの友人グループ、生物学的ネットワークでの機能モジュール、Webページのトピックグループなど、様々な分野で重要な構造を表します。
    
    
    ```mermaid
    graph LR
        subgraph C1["コミュニティ1"]
            A1((A))---A2((B))
            A2---A3((C))
            A3---A1
        end
        subgraph C2["コミュニティ2"]
            B1((D))---B2((E))
            B2---B3((F))
            B3---B1
        end
        A2-.弱い結合.-B1
    ```

### 1.2 モジュラリティ（Modularity）

**モジュラリティ** は、コミュニティ構造の品質を測る最も重要な指標です。ネットワークが与えられたコミュニティ分割に対して、どれだけ明確なコミュニティ構造を持つかを定量化します：

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$ 

ここで：

  * $A_{ij}$: 隣接行列の要素（エッジの存在）
  * $k_i, k_j$: ノード$i, j$の次数
  * $m$: エッジ総数
  * $c_i, c_j$: ノード$i, j$のコミュニティ
  * $\delta(c_i, c_j)$: 同じコミュニティなら1、異なれば0

> **直感的理解：** モジュラリティは「実際のエッジ数」と「ランダムネットワークでの期待値」の差を測ります。値の範囲は-0.5〜1.0で、0.3以上なら明確なコミュニティ構造を持つとされます。 

### 1.3 評価指標
    
    
    import networkx as nx
    import numpy as np
    from networkx.algorithms import community
    
    # サンプルネットワークの作成（Karate Clubデータセット）
    G = nx.karate_club_graph()
    
    # Ground truthのコミュニティ（実際のクラブ分裂）
    ground_truth = {
        frozenset([n for n in G.nodes() if G.nodes[n]['club'] == 'Mr. Hi']),
        frozenset([n for n in G.nodes() if G.nodes[n]['club'] == 'Officer'])
    }
    
    # Louvain法でコミュニティ検出
    detected_communities = community.louvain_communities(G, seed=42)
    
    # モジュラリティの計算
    modularity = community.modularity(G, detected_communities)
    print(f"モジュラリティ: {modularity:.4f}")
    
    # NMI（Normalized Mutual Information）の計算
    from sklearn.metrics import normalized_mutual_info_score
    
    # コミュニティをラベル配列に変換
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
    
    # カバレッジとパフォーマンス
    coverage = community.coverage(G, detected_communities)
    performance = community.performance(G, detected_communities)
    print(f"カバレッジ: {coverage:.4f}")
    print(f"パフォーマンス: {performance:.4f}")
    

主要な評価指標の解説 指標 | 説明 | 範囲 | 利点/欠点  
---|---|---|---  
**モジュラリティ(Q)** | コミュニティ内エッジの密度 | -0.5〜1.0 | 最も一般的／解像度限界問題  
**NMI** | 正解との一致度（情報量） | 0〜1 | Ground truth必要  
**カバレッジ** | コミュニティ内エッジの割合 | 0〜1 | 直感的／コミュニティ間エッジ無視  
**パフォーマンス** | 正確に分類されたペアの割合 | 0〜1 | バランス良い  
  
## 2\. Louvain法

### 2.1 アルゴリズムの仕組み

Louvain法は、モジュラリティを貪欲に最適化する階層的アルゴリズムです。2段階の反復プロセスで構成されます：
    
    
    ```mermaid
    graph TD
        A[初期化: 各ノードが独自コミュニティ] --> B[フェーズ1: ローカル最適化]
        B --> C{モジュラリティ改善あり?}
        C -->|Yes| B
        C -->|No| D[フェーズ2: ネットワーク縮約]
        D --> E{1つのコミュニティ?}
        E -->|No| B
        E -->|Yes| F[完了]
    ```

**フェーズ1（ローカル最適化）：**

  1. 各ノードを順番に処理
  2. 隣接コミュニティへの移動を試行
  3. モジュラリティが最大になる移動を採用
  4. 改善がなくなるまで繰り返す

モジュラリティの変化量 $\Delta Q$ は効率的に計算できます：

$$\Delta Q = \left[ \frac{\Sigma_{in} + k_{i,in}}{2m} - \left( \frac{\Sigma_{tot} + k_i}{2m} \right)^2 \right] - \left[ \frac{\Sigma_{in}}{2m} - \left( \frac{\Sigma_{tot}}{2m} \right)^2 - \left( \frac{k_i}{2m} \right)^2 \right]$$ 

### 2.2 階層的コミュニティ検出

**フェーズ2（ネットワーク縮約）：**

  * 各コミュニティを1つのスーパーノードに縮約
  * コミュニティ間のエッジを集約
  * 新しいネットワークでフェーズ1を実行

    
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    def visualize_louvain_hierarchy(G):
        """Louvain法の階層構造を可視化"""
        # 各階層でのコミュニティ検出
        communities_level0 = [{i} for i in G.nodes()]  # レベル0: 個別ノード
        communities_level1 = community.louvain_communities(G, seed=42, resolution=1.0)
    
        # より粗いコミュニティ（解像度を下げる）
        communities_level2 = community.louvain_communities(G, seed=42, resolution=0.5)
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        levels = [communities_level0, communities_level1, communities_level2]
        titles = ['レベル0\n(個別ノード)', 'レベル1\n(細かいコミュニティ)', 'レベル2\n(粗いコミュニティ)']
    
        for ax, comms, title in zip(axes, levels, titles):
            pos = nx.spring_layout(G, seed=42)
    
            # コミュニティごとに色分け
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
        print(f"階層数: {len(levels)}")
        for i, comms in enumerate(levels):
            print(f"レベル{i}: {len(comms)}個のコミュニティ")
    
    # 実行
    G = nx.karate_club_graph()
    visualize_louvain_hierarchy(G)
    

### 2.3 NetworkX/python-louvain実装
    
    
    import community as community_louvain  # python-louvain
    import networkx as nx
    
    # サンプルネットワーク（Les Miserables共起ネットワーク）
    G = nx.les_miserables_graph()
    
    # --- NetworkX組み込みのLouvain ---
    communities_nx = community.louvain_communities(G, seed=42)
    modularity_nx = community.modularity(G, communities_nx)
    
    print(f"NetworkX Louvain:")
    print(f"  コミュニティ数: {len(communities_nx)}")
    print(f"  モジュラリティ: {modularity_nx:.4f}")
    
    # --- python-louvainパッケージ（より詳細な情報を取得） ---
    # 最良の分割を取得
    partition = community_louvain.best_partition(G)
    
    # 階層構造全体を取得
    dendro = community_louvain.generate_dendrogram(G)
    print(f"\npython-louvain:")
    print(f"  階層レベル数: {len(dendro)}")
    
    # 各レベルのコミュニティ数
    for level in range(len(dendro)):
        partition_at_level = community_louvain.partition_at_level(dendro, level)
        num_communities = len(set(partition_at_level.values()))
        mod = community_louvain.modularity(partition_at_level, G)
        print(f"  レベル{level}: {num_communities}個のコミュニティ, Q={mod:.4f}")
    
    # 解像度パラメータの効果
    resolutions = [0.5, 1.0, 1.5, 2.0]
    print(f"\n解像度パラメータの影響:")
    for res in resolutions:
        partition_res = community_louvain.best_partition(G, resolution=res)
        num_comm = len(set(partition_res.values()))
        mod = community_louvain.modularity(partition_res, G)
        print(f"  resolution={res}: {num_comm}個のコミュニティ, Q={mod:.4f}")
    

> **計算量：** Louvain法の時間計算量は$O(n \log n)$で、大規模ネットワーク（数百万ノード）にも適用可能です。python-louvainパッケージは特に高速な実装を提供しています。 

## 3\. Label Propagation

### 3.1 ラベル伝播アルゴリズム

Label Propagation（ラベル伝播法）は、極めてシンプルで高速なコミュニティ検出手法です。基本的なアイデアは「多数決」：

  1. **初期化：** 各ノードに一意のラベルを割り当て
  2. **伝播：** 各ノードが隣接ノードの多数派ラベルを採用
  3. **収束：** ラベルが変化しなくなるまで繰り返す

    
    
    ```mermaid
    graph LR
        subgraph "ステップ0: 初期化"
            A0((A:1))
            B0((B:2))
            C0((C:3))
            D0((D:4))
            A0---B0---C0---D0
        end
    ```
    
    
    ```mermaid
    graph LR
        subgraph "ステップ1: 伝播"
            A1((A:1))
            B1((B:1))
            C1((C:2))
            D1((D:3))
            A1---B1---C1---D1
        end
    ```
    
    
    ```mermaid
    graph LR
        subgraph "ステップ2: 収束"
            A2((A:1))
            B2((B:1))
            C2((C:1))
            D2((D:1))
            A2---B2---C2---D2
        end
    ```

### 3.2 高速性と精度のトレードオフ

Label Propagationの最大の利点は**線形時間計算量 $O(m)$** （$m$はエッジ数）です。しかし、いくつかの課題もあります：

特徴 | 利点 | 欠点  
---|---|---  
**計算速度** | 非常に高速（線形時間） | -  
**スケーラビリティ** | 数千万ノードのネットワークも処理可能 | -  
**安定性** | - | 結果が実行ごとに変わる（非決定的）  
**品質** | - | Louvain法より低いモジュラリティ  
**収束性** | - | 振動する場合がある  
  
### 3.3 実装例
    
    
    import networkx as nx
    import numpy as np
    from collections import Counter
    
    def label_propagation_manual(G, max_iter=100, seed=None):
        """Label Propagationの手動実装（教育目的）"""
        if seed is not None:
            np.random.seed(seed)
    
        # 初期化: 各ノードに一意のラベル
        labels = {node: i for i, node in enumerate(G.nodes())}
        nodes = list(G.nodes())
    
        for iteration in range(max_iter):
            # ノードの処理順序をランダム化
            np.random.shuffle(nodes)
            changed = False
    
            for node in nodes:
                # 隣接ノードのラベルを収集
                neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
    
                if not neighbor_labels:
                    continue
    
                # 最頻ラベルを採用（同数の場合はランダム選択）
                label_counts = Counter(neighbor_labels)
                max_count = max(label_counts.values())
                most_common = [label for label, count in label_counts.items()
                              if count == max_count]
                new_label = np.random.choice(most_common)
    
                if labels[node] != new_label:
                    labels[node] = new_label
                    changed = True
    
            if not changed:
                print(f"収束: {iteration + 1}回の反復")
                break
    
        # ラベルをコミュニティセットに変換
        communities = {}
        for node, label in labels.items():
            if label not in communities:
                communities[label] = set()
            communities[label].add(node)
    
        return list(communities.values())
    
    # テストと比較
    G = nx.karate_club_graph()
    
    # 手動実装
    communities_manual = label_propagation_manual(G, seed=42)
    mod_manual = community.modularity(G, communities_manual)
    
    # NetworkX組み込み
    communities_nx = list(community.label_propagation_communities(G))
    mod_nx = community.modularity(G, communities_nx)
    
    print("Label Propagation結果比較:")
    print(f"手動実装: {len(communities_manual)}個のコミュニティ, Q={mod_manual:.4f}")
    print(f"NetworkX: {len(communities_nx)}個のコミュニティ, Q={mod_nx:.4f}")
    
    # 複数回実行して安定性を確認
    print("\n安定性テスト（10回実行）:")
    modularities = []
    for i in range(10):
        comms = label_propagation_manual(G, seed=i)
        mod = community.modularity(G, comms)
        modularities.append(mod)
        print(f"  実行{i+1}: Q={mod:.4f}, コミュニティ数={len(comms)}")
    
    print(f"平均モジュラリティ: {np.mean(modularities):.4f} ± {np.std(modularities):.4f}")
    

> **実践的アドバイス：** Label Propagationは初期探索や超大規模ネットワークに有効です。より高品質な結果が必要な場合は、Label Propagationの結果をLouvain法の初期値として使用する「ハイブリッドアプローチ」が効果的です。 

## 4\. その他のコミュニティ検出手法

### 4.1 Girvan-Newman法（エッジ媒介性ベース）

Girvan-Newman法は、コミュニティ間を結ぶエッジを除去していく階層的手法です：

  1. 全エッジのエッジ媒介中心性を計算
  2. 最大媒介中心性のエッジを除去
  3. モジュラリティを計算
  4. 全てのエッジが除去されるまで繰り返す
  5. 最大モジュラリティの分割を採用

    
    
    import networkx as nx
    from networkx.algorithms.community import girvan_newman
    
    G = nx.karate_club_graph()
    
    # Girvan-Newman法（階層全体を生成）
    communities_generator = girvan_newman(G)
    
    # 異なる分割レベルを取得
    modularities = []
    all_partitions = []
    
    for i, communities in enumerate(communities_generator):
        partition = tuple(sorted(communities, key=len, reverse=True))
        all_partitions.append(partition)
    
        # モジュラリティを計算
        mod = community.modularity(G, partition)
        modularities.append(mod)
    
        print(f"分割{i+1}: {len(partition)}個のコミュニティ, Q={mod:.4f}")
    
        # 10分割まで確認
        if i >= 9:
            break
    
    # 最適分割を選択
    best_idx = np.argmax(modularities)
    best_partition = all_partitions[best_idx]
    print(f"\n最適分割: レベル{best_idx+1}, Q={modularities[best_idx]:.4f}")
    

**計算量：** $O(m^2 n)$ - 大規模ネットワークには不向きですが、小規模ネットワークで解釈可能な階層構造を提供します。

### 4.2 Infomap

Infomapは、ランダムウォークの符号化問題としてコミュニティ検出を定式化します。コミュニティ内を長く滞在するランダムウォークを効率的に記述できる分割を探します。
    
    
    try:
        import infomap
        has_infomap = True
    except ImportError:
        has_infomap = False
        print("infomap未インストール: pip install infomap")
    
    if has_infomap:
        # Infomapの実行
        im = infomap.Infomap("--two-level --directed")
    
        # ネットワークを追加（無向グラフの場合は両方向に追加）
        for u, v in G.edges():
            im.add_link(u, v)
            im.add_link(v, u)
    
        # クラスタリング実行
        im.run()
    
        # 結果を取得
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
        print(f"  コミュニティ数: {len(communities_infomap)}")
        print(f"  モジュラリティ: {mod_infomap:.4f}")
        print(f"  Codelength: {im.codelength:.4f}")
    

### 4.3 スペクトラルクラスタリング

スペクトラルクラスタリングは、グラフラプラシアンの固有ベクトルを用いてコミュニティを検出します：
    
    
    from sklearn.cluster import SpectralClustering
    import numpy as np
    
    # 隣接行列を取得
    A = nx.to_numpy_array(G)
    
    # 異なるコミュニティ数で試行
    for n_clusters in [2, 3, 4, 5]:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
    
        labels = sc.fit_predict(A)
    
        # ラベルをコミュニティセットに変換
        communities_spectral = [set() for _ in range(n_clusters)]
        for node, label in enumerate(labels):
            communities_spectral[label].add(node)
    
        mod = community.modularity(G, communities_spectral)
        print(f"スペクトラルクラスタリング (k={n_clusters}): Q={mod:.4f}")
    

手法比較の詳細 手法 | 時間計算量 | 品質 | 決定性 | 適用場面  
---|---|---|---|---  
**Louvain** | $O(n \log n)$ | 高 | 準決定的 | 一般的な用途、大規模ネットワーク  
**Label Propagation** | $O(m)$ | 中 | 非決定的 | 超大規模ネットワーク、初期探索  
**Girvan-Newman** | $O(m^2 n)$ | 中〜高 | 決定的 | 小規模ネットワーク、階層構造の可視化  
**Infomap** | $O(m)$ | 高 | 準決定的 | フロー情報が重要な場合  
**スペクトラル** | $O(n^3)$ | 中 | 決定的 | コミュニティ数が既知の場合  
  
## 5\. 実践: ソーシャルネットワークのコミュニティ分析

### 5.1 Facebookネットワーク例
    
    
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Facebook SNAPデータセット（ego-Facebook）の代替: Zachary's Karate Clubで詳細分析
    G = nx.karate_club_graph()
    
    print(f"ネットワーク情報:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}")
    print(f"  平均次数: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  密度: {nx.density(G):.4f}")
    
    # Ground truth（実際のクラブ分裂）
    ground_truth = []
    for club in ['Mr. Hi', 'Officer']:
        comm = {n for n in G.nodes() if G.nodes[n]['club'] == club}
        ground_truth.append(comm)
    
    print(f"\nGround truth: {len(ground_truth)}個のグループ")
    
    # 各手法でコミュニティ検出
    results = {}
    
    # 1. Louvain
    communities_louvain = community.louvain_communities(G, seed=42)
    results['Louvain'] = communities_louvain
    
    # 2. Label Propagation
    communities_lp = list(community.label_propagation_communities(G))
    results['Label Propagation'] = communities_lp
    
    # 3. Greedy Modularity（高速な代替手法）
    communities_greedy = community.greedy_modularity_communities(G)
    results['Greedy Modularity'] = communities_greedy
    
    # 4. Girvan-Newman（最適分割のみ）
    gn_generator = girvan_newman(G)
    gn_modularities = []
    gn_partitions = []
    for partition in gn_generator:
        gn_partitions.append(partition)
        gn_modularities.append(community.modularity(G, partition))
        if len(gn_partitions) >= 10:  # 最初の10分割のみ評価
            break
    best_gn = gn_partitions[np.argmax(gn_modularities)]
    results['Girvan-Newman'] = best_gn
    
    print("\nコミュニティ検出結果:")
    for method, comms in results.items():
        mod = community.modularity(G, comms)
    
        # Ground truthとのNMIを計算
        gt_labels = communities_to_labels(ground_truth, len(G))
        detected_labels = communities_to_labels(comms, len(G))
        nmi = normalized_mutual_info_score(gt_labels, detected_labels)
    
        print(f"{method:20s}: {len(comms):2d}個, Q={mod:.4f}, NMI={nmi:.4f}")
    

### 5.2 手法の比較
    
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    
    # 包括的な比較分析
    def comprehensive_comparison(G, ground_truth, results):
        """複数の評価指標で手法を比較"""
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
    
            # メトリクスを計算
            metrics['Modularity'].append(community.modularity(G, comms))
            metrics['NMI'].append(normalized_mutual_info_score(gt_labels, detected_labels))
            metrics['ARI'].append(adjusted_rand_score(gt_labels, detected_labels))
            metrics['Coverage'].append(community.coverage(G, comms))
            metrics['Communities'].append(len(comms))
    
        # 可視化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
        for i, (metric, values) in enumerate(list(metrics.items())[:5]):
            ax = axes[i]
            bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
    
            # 値をバーの上に表示
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
        # 最後のサブプロットを削除
        fig.delaxes(axes[5])
    
        plt.tight_layout()
        plt.savefig('community_comparison.png', dpi=150, bbox_inches='tight')
        print("比較図を保存: community_comparison.png")
    
    # 実行
    comprehensive_comparison(G, ground_truth, results)
    

### 5.3 コミュニティの可視化と解釈
    
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    def visualize_communities(G, communities, title, ground_truth=None):
        """コミュニティを美しく可視化"""
        fig, ax = plt.subplots(figsize=(12, 10))
    
        # レイアウト計算
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
        # カラーマップ
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84']
    
        # ノードの色付け
        node_colors = []
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_colors.append(colors[i % len(colors)])
                node_to_comm[node] = i
    
        # エッジの描画（コミュニティ内/間で色分け）
        edges_within = []
        edges_between = []
        for u, v in G.edges():
            if node_to_comm[u] == node_to_comm[v]:
                edges_within.append((u, v))
            else:
                edges_between.append((u, v))
    
        # コミュニティ内エッジ（濃い色）
        nx.draw_networkx_edges(G, pos, edgelist=edges_within,
                               width=2, alpha=0.6, edge_color='#2C3E50', ax=ax)
    
        # コミュニティ間エッジ（薄い色、点線）
        nx.draw_networkx_edges(G, pos, edgelist=edges_between,
                               width=1, alpha=0.3, edge_color='#95A5A6',
                               style='dashed', ax=ax)
    
        # ノード描画
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=500, alpha=0.9,
                               edgecolors='white', linewidths=2, ax=ax)
    
        # ラベル
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
        # 凡例
        legend_elements = [mpatches.Patch(facecolor=colors[i % len(colors)],
                                          label=f'Community {i+1} ({len(comm)} nodes)')
                          for i, comm in enumerate(communities)]
        ax.legend(handles=legend_elements, loc='upper left',
                 framealpha=0.9, fontsize=10)
    
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
    
        # メトリクス表示
        mod = community.modularity(G, communities)
        info_text = f"Modularity: {mod:.4f}\n"
        info_text += f"Communities: {len(communities)}\n"
        info_text += f"Intra-edges: {len(edges_within)}\n"
        info_text += f"Inter-edges: {len(edges_between)}"
    
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        return fig
    
    # 各手法の結果を可視化
    for method, comms in results.items():
        fig = visualize_communities(G, comms, f"Community Detection: {method}")
        plt.savefig(f'community_{method.replace(" ", "_").lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("全ての可視化を保存しました")
    

> **解釈のポイント：**
> 
>   * **モジュラリティ：** 0.3以上なら明確なコミュニティ構造
>   * **コミュニティサイズ：** 極端に小さい/大きいコミュニティは要注意
>   * **エッジ比率：** コミュニティ間エッジが多い場合、境界が曖昧
>   * **Ground truthとの一致：** NMI/ARIで既知の構造との対応を確認
> 

## 演習問題

問題1: モジュラリティの理解

**問題：** 以下のネットワークで、コミュニティ分割 C1 = {A, B} と C2 = {C, D} のモジュラリティを手計算してください。
    
    
    A -- B
    |    |
    C -- D
    

**ヒント：** $Q = \frac{1}{2m} \sum_{ij} [A_{ij} - \frac{k_i k_j}{2m}] \delta(c_i, c_j)$ を使用

問題2: Louvain法の実装

**問題：** フェーズ1のローカル最適化において、ノード移動による $\Delta Q$ を効率的に計算する関数を実装してください。
    
    
    def compute_delta_Q(G, node, current_comm, new_comm, m):
        """
        ノードを current_comm から new_comm に移動した時の
        モジュラリティ変化量を計算
    
        Parameters:
        -----------
        G : NetworkX graph
        node : int
            移動するノード
        current_comm : set
            現在のコミュニティ
        new_comm : set
            移動先のコミュニティ
        m : int
            エッジ総数
    
        Returns:
        --------
        delta_Q : float
            モジュラリティの変化量
        """
        # ここに実装
        pass
    

問題3: Label Propagationの収束性

**問題：** Label Propagationが収束しない（振動する）ケースを構築し、その理由を説明してください。また、収束を保証する改善策を提案してください。

問題4: 解像度限界問題

**問題：** モジュラリティ最適化には「解像度限界（resolution limit）」問題があります。小さなコミュニティが検出できない例を作成し、解像度パラメータで改善できることを示してください。
    
    
    import networkx as nx
    from networkx.algorithms import community
    
    # 複数の小さなクリークを持つネットワークを作成
    # 解像度パラメータを変えて結果を比較
    

問題5: 重み付きネットワークのコミュニティ検出

**問題：** エッジに重みがあるネットワークで、重みを考慮したLouvain法を適用してください。重みの有無で結果がどう変わるか分析してください。
    
    
    import networkx as nx
    
    # 重み付きネットワークの作成
    G = nx.karate_club_graph()
    
    # ランダムに重みを付与
    import random
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 2.0)
    
    # 重みありとなしでコミュニティ検出を比較
    

## まとめ

この章では、ネットワークにおけるコミュニティ検出の理論と実践を学びました：

  * ✅ **モジュラリティ：** コミュニティ構造の品質を定量化する標準的指標
  * ✅ **Louvain法：** 階層的な貪欲最適化による高品質・高速な検出
  * ✅ **Label Propagation：** 線形時間の超高速アルゴリズム（トレードオフあり）
  * ✅ **その他の手法：** Girvan-Newman、Infomap、スペクトラルクラスタリング
  * ✅ **実践：** 手法の比較、可視化、解釈の方法

> **次のステップ：** 次章では、ネットワークの動的な性質を扱います。時間発展するネットワークの分析、リンク予測、ネットワークの成長モデルについて学びます。
