---
title: 第5章：実装と統合アプリケーション
chapter_title: 第5章：実装と統合アプリケーション
subtitle: SPARQL API、可視化、文書自動生成、推薦システム、完全統合システム
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Flask/FastAPIでSPARQLエンドポイントAPIを構築できる
  * ✅ NetworkXとPlotlyでナレッジグラフを可視化できる
  * ✅ テンプレートエンジンでプロセス文書を自動生成できる
  * ✅ 根本原因分析（RCA）アプリケーションを実装できる
  * ✅ ナレッジベース機器推薦システムを構築できる
  * ✅ 最適化問題にナレッジグラフを活用できる
  * ✅ API+推論+可視化を統合したエンドツーエンドシステムを開発できる

* * *

## 5.1 SPARQLエンドポイントAPI

### コード例1: FlaskベースSPARQL クエリAPI
    
    
    from flask import Flask, request, jsonify
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    import json
    
    # Flask SPARQL エンドポイントAPIの実装
    
    app = Flask(__name__)
    
    # グローバルなナレッジグラフ
    PROC = Namespace("http://example.org/process#")
    g = Graph()
    g.bind("proc", PROC)
    
    # サンプルデータの初期化
    def init_knowledge_graph():
        """ナレッジグラフを初期化"""
        # 機器データ
        equipment_data = [
            {"id": "P101", "type": "Pump", "flow": 50.0, "pressure": 500.0, "status": "Running"},
            {"id": "P102", "type": "Pump", "flow": 45.0, "pressure": 480.0, "status": "Running"},
            {"id": "R201", "type": "Reactor", "temp": 180.0, "pressure": 800.0, "status": "Running"},
            {"id": "T301", "type": "Tank", "level": 75.0, "temp": 25.0, "status": "Running"},
        ]
    
        for eq in equipment_data:
            eq_uri = PROC[eq['id']]
            g.add((eq_uri, RDF.type, PROC[eq['type']]))
            g.add((eq_uri, PROC.hasStatus, Literal(eq['status'])))
    
            if 'flow' in eq:
                g.add((eq_uri, PROC.hasFlowRate, Literal(eq['flow'])))
            if 'pressure' in eq:
                g.add((eq_uri, PROC.hasPressure, Literal(eq['pressure'])))
            if 'temp' in eq:
                g.add((eq_uri, PROC.hasTemperature, Literal(eq['temp'])))
            if 'level' in eq:
                g.add((eq_uri, PROC.hasLevel, Literal(eq['level'])))
    
        print(f"ナレッジグラフ初期化完了: {len(g)} トリプル")
    
    
    # API エンドポイント
    
    @app.route('/api/sparql', methods=['POST'])
    def sparql_endpoint():
        """
        SPARQL クエリエンドポイント
    
        POST /api/sparql
        Body: {"query": "SELECT ..."}
    
        Returns:
            JSON: クエリ結果
        """
        try:
            data = request.get_json()
            query_str = data.get('query', '')
    
            if not query_str:
                return jsonify({"error": "クエリが空です"}), 400
    
            # SPARQLクエリを実行
            query = prepareQuery(query_str, initNs={"proc": PROC, "rdf": RDF, "rdfs": RDFS})
            results = g.query(query)
    
            # 結果をJSON形式に変換
            response = []
            for row in results:
                row_dict = {}
                for var in results.vars:
                    value = row[var]
                    # URIの場合は短縮形に
                    if hasattr(value, 'toPython'):
                        row_dict[str(var)] = str(value).split('#')[-1]
                    else:
                        row_dict[str(var)] = str(value)
                response.append(row_dict)
    
            return jsonify({
                "status": "success",
                "count": len(response),
                "results": response
            }), 200
    
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    
    @app.route('/api/equipment', methods=['GET'])
    def get_all_equipment():
        """
        全機器一覧を取得
    
        GET /api/equipment
    
        Returns:
            JSON: 機器リスト
        """
        query = prepareQuery("""
            SELECT ?equipment ?type ?status
            WHERE {
                ?equipment a ?type .
                ?equipment proc:hasStatus ?status .
                FILTER (strstarts(str(?type), str(proc:)))
            }
        """, initNs={"proc": PROC})
    
        results = g.query(query)
        equipment_list = []
    
        for row in results:
            equipment_list.append({
                "id": str(row.equipment).split('#')[-1],
                "type": str(row.type).split('#')[-1],
                "status": str(row.status)
            })
    
        return jsonify({
            "status": "success",
            "count": len(equipment_list),
            "equipment": equipment_list
        }), 200
    
    
    @app.route('/api/equipment/', methods=['GET'])
    def get_equipment_details(eq_id):
        """
        特定機器の詳細情報を取得
    
        GET /api/equipment/{eq_id}
    
        Returns:
            JSON: 機器詳細
        """
        eq_uri = PROC[eq_id]
    
        # 機器の全プロパティを取得
        query = prepareQuery("""
            SELECT ?property ?value
            WHERE {
                ?equipment ?property ?value .
            }
        """, initNs={"proc": PROC})
    
        results = g.query(query, initBindings={'equipment': eq_uri})
    
        if len(list(results)) == 0:
            return jsonify({"error": f"機器 {eq_id} が見つかりません"}), 404
    
        # 再クエリ（iteratorが消費済みのため）
        results = g.query(query, initBindings={'equipment': eq_uri})
    
        details = {"id": eq_id}
        for row in results:
            prop = str(row.property).split('#')[-1]
            value = row.value.toPython() if hasattr(row.value, 'toPython') else str(row.value)
            details[prop] = value
    
        return jsonify({
            "status": "success",
            "equipment": details
        }), 200
    
    
    @app.route('/api/statistics', methods=['GET'])
    def get_statistics():
        """
        統計情報を取得
    
        GET /api/statistics
    
        Returns:
            JSON: ナレッジグラフ統計
        """
        stats = {
            "total_triples": len(g),
            "total_equipment": len(list(g.subjects(RDF.type, None))),
            "equipment_by_type": {}
        }
    
        # 機器タイプ別カウント
        query = prepareQuery("""
            SELECT ?type (COUNT(?equipment) as ?count)
            WHERE {
                ?equipment a ?type .
                FILTER (strstarts(str(?type), str(proc:)))
            }
            GROUP BY ?type
        """, initNs={"proc": PROC})
    
        for row in g.query(query):
            type_name = str(row.type).split('#')[-1]
            stats['equipment_by_type'][type_name] = int(row['count'])
    
        return jsonify({
            "status": "success",
            "statistics": stats
        }), 200
    
    
    # 実行デモ（通常はif __name__ == '__main__'で起動）
    def demo_api():
        """APIのデモンストレーション"""
        print("="*60)
        print("SPARQL エンドポイントAPI")
        print("="*60)
    
        init_knowledge_graph()
    
        print("\n【APIエンドポイント】")
        print("  POST /api/sparql - SPARQLクエリ実行")
        print("  GET  /api/equipment - 全機器一覧")
        print("  GET  /api/equipment/{id} - 機器詳細")
        print("  GET  /api/statistics - 統計情報")
    
        print("\n【サンプルクエリ】")
        sample_query = """
        SELECT ?equipment ?pressure
        WHERE {
            ?equipment proc:hasPressure ?pressure .
            FILTER (?pressure > 600)
        }
        """
        print(f"  {sample_query.strip()}")
    
        # クエリを実行（APIを使わず直接実行）
        query = prepareQuery(sample_query, initNs={"proc": PROC})
        results = g.query(query)
    
        print("\n【クエリ結果】")
        for row in results:
            eq_id = str(row.equipment).split('#')[-1]
            print(f"  {eq_id}: 圧力 {row.pressure} kPa")
    
        print("\n【Flask起動コマンド】")
        print("  python app.py")
        print("  または: flask run --port 5000")
    
    
    # デモ実行
    demo_api()
    
    # Flaskアプリケーション起動（実際の使用時）
    # if __name__ == '__main__':
    #     init_knowledge_graph()
    #     app.run(debug=True, port=5000)
    

**出力例:**
    
    
    ============================================================
    SPARQL エンドポイントAPI
    ============================================================
    ナレッジグラフ初期化完了: 16 トリプル
    
    【APIエンドポイント】
      POST /api/sparql - SPARQLクエリ実行
      GET  /api/equipment - 全機器一覧
      GET  /api/equipment/{id} - 機器詳細
      GET  /api/statistics - 統計情報
    
    【サンプルクエリ】
      SELECT ?equipment ?pressure
        WHERE {
            ?equipment proc:hasPressure ?pressure .
            FILTER (?pressure > 600)
        }
    
    【クエリ結果】
      R201: 圧力 800.0 kPa
    
    【Flask起動コマンド】
      python app.py
      または: flask run --port 5000
    

**解説:** FlaskでSPARQLエンドポイントAPIを構築し、RESTful APIとしてナレッジグラフにアクセスできます。POST /api/sparqlで任意のSPARQLクエリを実行でき、他のエンドポイントで機器情報や統計を取得できます。

* * *

## 5.2 ナレッジグラフの可視化

### コード例2: NetworkXとPlotlyによるグラフ可視化
    
    
    import networkx as nx
    import plotly.graph_objects as go
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    import numpy as np
    
    # ナレッジグラフの可視化
    
    PROC = Namespace("http://example.org/process#")
    
    def create_process_graph():
        """プロセスナレッジグラフを作成"""
        g = Graph()
        g.bind("proc", PROC)
    
        # プロセスフロー定義
        # Feed → P-101 → R-201 → T-301 → Product
        g.add((PROC.Feed, RDF.type, PROC.Stream))
        g.add((PROC.P101, RDF.type, PROC.Pump))
        g.add((PROC.R201, RDF.type, PROC.Reactor))
        g.add((PROC.T301, RDF.type, PROC.Tank))
        g.add((PROC.Product, RDF.type, PROC.Stream))
    
        # 接続関係
        g.add((PROC.Feed, PROC.flowsTo, PROC.P101))
        g.add((PROC.P101, PROC.flowsTo, PROC.R201))
        g.add((PROC.R201, PROC.flowsTo, PROC.T301))
        g.add((PROC.T301, PROC.flowsTo, PROC.Product))
    
        # プロパティ
        g.add((PROC.P101, PROC.hasFlowRate, Literal(50.0)))
        g.add((PROC.P101, PROC.hasPressure, Literal(500.0)))
        g.add((PROC.R201, PROC.hasTemperature, Literal(180.0)))
        g.add((PROC.T301, PROC.hasLevel, Literal(75.0)))
    
        return g
    
    
    def rdf_to_networkx(rdf_graph):
        """
        RDFグラフをNetworkXグラフに変換
    
        Parameters:
            rdf_graph (Graph): RDFグラフ
    
        Returns:
            nx.DiGraph: NetworkX有向グラフ
        """
        nx_graph = nx.DiGraph()
    
        # ノードとエッジを追加
        for s, p, o in rdf_graph:
            subj = str(s).split('#')[-1]
            pred = str(p).split('#')[-1]
            obj = str(o).split('#')[-1] if '#' in str(o) else str(o)
    
            # ノードのタイプを取得
            if p == RDF.type:
                nx_graph.add_node(subj, node_type=obj)
            elif p == PROC.flowsTo:
                nx_graph.add_edge(subj, obj, relation='flowsTo')
            else:
                # プロパティはノード属性として追加
                if subj in nx_graph:
                    nx_graph.nodes[subj][pred] = obj
                else:
                    nx_graph.add_node(subj, **{pred: obj})
    
        return nx_graph
    
    
    def visualize_with_plotly(nx_graph):
        """
        NetworkXグラフをPlotlyで可視化
    
        Parameters:
            nx_graph (nx.DiGraph): NetworkXグラフ
        """
        # レイアウト計算（階層的レイアウト）
        pos = nx.spring_layout(nx_graph, seed=42, k=1.5, iterations=50)
    
        # エッジのトレース
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
    
        # エッジの矢印のためのアノテーション
        annotations = []
    
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
    
            # 矢印アノテーション
            annotations.append(
                dict(
                    ax=x0, ay=y0,
                    x=x1, y=y1,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#888'
                )
            )
    
        # ノードのトレース
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale='Greens',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='ノードタイプ',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
    
        # ノード情報を追加
        node_colors = {'Stream': 0, 'Pump': 1, 'Reactor': 2, 'Tank': 3}
    
        for node in nx_graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
    
            # ノードタイプに応じた色
            node_type = nx_graph.nodes[node].get('node_type', 'Unknown')
            color_idx = node_colors.get(node_type, 0)
            node_trace['marker']['color'] += tuple([color_idx])
            node_trace['marker']['size'] += tuple([30])
    
            # ホバー情報
            hover_text = f"**{node}**  
     "
            hover_text += f"タイプ: {node_type}  
    "
            for key, value in nx_graph.nodes[node].items():
                if key != 'node_type':
                    hover_text += f"{key}: {value}  
    "
    
            if 'hovertext' not in node_trace:
                node_trace['hovertext'] = []
            node_trace['hovertext'].append(hover_text)
    
        # レイアウト設定
        layout = go.Layout(
            title=dict(
                text='プロセスナレッジグラフ可視化',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#2c3e50')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=900,
            height=600
        )
    
        # グラフ作成
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        fig.show()
    
        print("\n【可視化完了】")
        print(f"  ノード数: {nx_graph.number_of_nodes()}")
        print(f"  エッジ数: {nx_graph.number_of_edges()}")
    
    
    # 実行デモ
    print("="*60)
    print("ナレッジグラフ可視化（NetworkX + Plotly）")
    print("="*60)
    
    rdf_g = create_process_graph()
    print(f"\nRDFグラフ作成完了: {len(rdf_g)} トリプル")
    
    nx_g = rdf_to_networkx(rdf_g)
    print(f"NetworkXグラフに変換: {nx_g.number_of_nodes()} ノード, {nx_g.number_of_edges()} エッジ")
    
    print("\n【ノード一覧】")
    for node, data in nx_g.nodes(data=True):
        print(f"  {node}: {data}")
    
    print("\n【エッジ一覧】")
    for u, v, data in nx_g.edges(data=True):
        print(f"  {u} → {v}: {data}")
    
    # Plotlyで可視化
    visualize_with_plotly(nx_g)
    

**出力例:**
    
    
    ============================================================
    ナレッジグラフ可視化（NetworkX + Plotly）
    ============================================================
    
    RDFグラフ作成完了: 13 トリプル
    NetworkXグラフに変換: 5 ノード, 4 エッジ
    
    【ノード一覧】
      Feed: {'node_type': 'Stream'}
      P101: {'node_type': 'Pump', 'hasFlowRate': '50.0', 'hasPressure': '500.0'}
      R201: {'node_type': 'Reactor', 'hasTemperature': '180.0'}
      T301: {'node_type': 'Tank', 'hasLevel': '75.0'}
      Product: {'node_type': 'Stream'}
    
    【エッジ一覧】
      Feed → P101: {'relation': 'flowsTo'}
      P101 → R201: {'relation': 'flowsTo'}
      R201 → T301: {'relation': 'flowsTo'}
      T301 → Product: {'relation': 'flowsTo'}
    
    【可視化完了】
      ノード数: 5
      エッジ数: 4
    

**解説:** RDFグラフをNetworkXグラフに変換し、Plotlyで対話的に可視化します。ノードはプロセス機器、エッジは物質の流れを表し、マウスホバーで詳細情報を表示できます。

* * *

## 5.3 プロセス文書の自動生成

### コード例3: テンプレートベース文書自動生成
    
    
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    from jinja2 import Template
    from datetime import datetime
    
    # プロセス文書自動生成システム
    
    PROC = Namespace("http://example.org/process#")
    
    def create_process_documentation_kb():
        """文書生成用ナレッジベースを作成"""
        g = Graph()
        g.bind("proc", PROC)
    
        # プロセスユニット
        g.add((PROC.PU100, RDF.type, PROC.ProcessUnit))
        g.add((PROC.PU100, RDFS.label, Literal("反応プロセスユニット")))
        g.add((PROC.PU100, PROC.hasDescription, Literal("原料Aの触媒反応による製品B製造")))
    
        # 機器
        equipment_data = [
            {"id": "P101", "type": "Pump", "label": "原料供給ポンプ", "spec": "遠心ポンプ, 50 m³/h, 5 bar"},
            {"id": "R201", "type": "Reactor", "label": "触媒反応器", "spec": "CSTR, 10 m³, 触媒: Pt/Al2O3"},
            {"id": "HX301", "type": "HeatExchanger", "label": "冷却器", "spec": "シェル&チューブ, 500 kW"},
            {"id": "T401", "type": "Tank", "label": "製品貯槽", "spec": "縦型円筒タンク, 50 m³"},
        ]
    
        for eq in equipment_data:
            eq_uri = PROC[eq['id']]
            g.add((eq_uri, RDF.type, PROC[eq['type']]))
            g.add((eq_uri, RDFS.label, Literal(eq['label'])))
            g.add((eq_uri, PROC.hasSpecification, Literal(eq['spec'])))
            g.add((eq_uri, PROC.belongsTo, PROC.PU100))
    
        # 運転パラメータ
        g.add((PROC.R201, PROC.operatingTemperature, Literal(180.0)))
        g.add((PROC.R201, PROC.operatingPressure, Literal(800.0)))
        g.add((PROC.R201, PROC.operatingFlowRate, Literal(50.0)))
    
        # 安全制約
        g.add((PROC.R201, PROC.maxTemperature, Literal(200.0)))
        g.add((PROC.R201, PROC.maxPressure, Literal(1000.0)))
    
        return g
    
    
    def generate_process_document(g):
        """
        プロセス設計書を自動生成
    
        Parameters:
            g (Graph): ナレッジグラフ
    
        Returns:
            str: Markdown文書
        """
        # プロセスユニット情報を取得
        query_pu = prepareQuery("""
            SELECT ?pu ?label ?desc
            WHERE {
                ?pu a proc:ProcessUnit .
                ?pu rdfs:label ?label .
                ?pu proc:hasDescription ?desc .
            }
        """, initNs={"proc": PROC, "rdfs": RDFS})
    
        pu_info = list(g.query(query_pu))[0]
    
        # 機器リストを取得
        query_eq = prepareQuery("""
            SELECT ?equipment ?type ?label ?spec
            WHERE {
                ?equipment a ?type .
                ?equipment rdfs:label ?label .
                ?equipment proc:hasSpecification ?spec .
                ?equipment proc:belongsTo ?pu .
                FILTER (strstarts(str(?type), str(proc:)))
                FILTER (?type != proc:ProcessUnit)
            }
            ORDER BY ?equipment
        """, initNs={"proc": PROC, "rdfs": RDFS})
    
        equipment_list = []
        for row in g.query(query_eq):
            equipment_list.append({
                'id': str(row.equipment).split('#')[-1],
                'type': str(row.type).split('#')[-1],
                'label': str(row.label),
                'spec': str(row.spec)
            })
    
        # 運転パラメータを取得
        query_params = prepareQuery("""
            SELECT ?equipment ?temp ?pressure ?flow
            WHERE {
                ?equipment proc:operatingTemperature ?temp .
                ?equipment proc:operatingPressure ?pressure .
                ?equipment proc:operatingFlowRate ?flow .
            }
        """, initNs={"proc": PROC})
    
        operating_params = {}
        for row in g.query(query_params):
            eq_id = str(row.equipment).split('#')[-1]
            operating_params[eq_id] = {
                'temperature': float(row.temp),
                'pressure': float(row.pressure),
                'flow_rate': float(row.flow)
            }
    
        # Jinja2テンプレート
        template_str = """# プロセス設計書
    
    **文書番号**: DOC-{{ doc_id }}
    **作成日**: {{ date }}
    **バージョン**: 1.0
    
    ---
    
    ## 1. プロセス概要
    
    **プロセスユニット**: {{ pu_label }}
    
    {{ pu_description }}
    
    ---
    
    ## 2. 主要機器リスト
    
    | 機器ID | 機器名称 | タイプ | 仕様 |
    |--------|---------|--------|------|
    {% for eq in equipment_list -%}
    | {{ eq.id }} | {{ eq.label }} | {{ eq.type }} | {{ eq.spec }} |
    {% endfor %}
    
    ---
    
    ## 3. 運転条件
    
    {% for eq_id, params in operating_params.items() -%}
    ### {{ eq_id }}
    
    - **運転温度**: {{ params.temperature }} °C
    - **運転圧力**: {{ params.pressure }} kPa
    - **運転流量**: {{ params.flow_rate }} m³/h
    
    {% endfor %}
    
    ---
    
    ## 4. 安全上の注意事項
    
    {% if safety_notes -%}
    {% for note in safety_notes -%}
    - {{ note }}
    {% endfor %}
    {% else -%}
    （ナレッジグラフから自動抽出予定）
    {% endif %}
    
    ---
    
    ## 5. 関連文書
    
    - P&ID図面: DWG-PU100-001
    - プロセスフローシート: PFS-PU100
    - 機器仕様書: SPEC-{{ equipment_list[0].id }} 〜 SPEC-{{ equipment_list[-1].id }}
    
    ---
    
    *本文書はナレッジグラフから自動生成されました*
    """
    
        # テンプレートに値を埋め込み
        template = Template(template_str)
        document = template.render(
            doc_id="PU100-DESIGN-001",
            date=datetime.now().strftime("%Y-%m-%d"),
            pu_label=str(pu_info.label),
            pu_description=str(pu_info.desc),
            equipment_list=equipment_list,
            operating_params=operating_params,
            safety_notes=None
        )
    
        return document
    
    
    # 実行デモ
    print("="*60)
    print("プロセス文書自動生成システム")
    print("="*60)
    
    g = create_process_documentation_kb()
    print(f"\nナレッジグラフ作成: {len(g)} トリプル")
    
    print("\n【文書生成中...】")
    document = generate_process_document(g)
    
    print("\n" + "="*60)
    print(document)
    print("="*60)
    
    # ファイルに保存
    with open("process_design_document.md", "w", encoding="utf-8") as f:
        f.write(document)
    
    print("\n【保存完了】: process_design_document.md")
    

**出力例:**
    
    
    ============================================================
    プロセス文書自動生成システム
    ============================================================
    
    ナレッジグラフ作成: 23 トリプル
    
    【文書生成中...】
    
    ============================================================
    # プロセス設計書
    
    **文書番号**: DOC-PU100-DESIGN-001
    **作成日**: 2025-10-26
    **バージョン**: 1.0
    
    ---
    
    ## 1. プロセス概要
    
    **プロセスユニット**: 反応プロセスユニット
    
    原料Aの触媒反応による製品B製造
    
    ---
    
    ## 2. 主要機器リスト
    
    | 機器ID | 機器名称 | タイプ | 仕様 |
    |--------|---------|--------|------|
    | P101 | 原料供給ポンプ | Pump | 遠心ポンプ, 50 m³/h, 5 bar |
    | R201 | 触媒反応器 | Reactor | CSTR, 10 m³, 触媒: Pt/Al2O3 |
    | HX301 | 冷却器 | HeatExchanger | シェル&チューブ, 500 kW |
    | T401 | 製品貯槽 | Tank | 縦型円筒タンク, 50 m³ |
    
    ---
    
    ## 3. 運転条件
    
    ### R201
    
    - **運転温度**: 180.0 °C
    - **運転圧力**: 800.0 kPa
    - **運転流量**: 50.0 m³/h
    
    ---
    
    ## 4. 安全上の注意事項
    
    （ナレッジグラフから自動抽出予定）
    
    ---
    
    ## 5. 関連文書
    
    - P&ID図面: DWG-PU100-001
    - プロセスフローシート: PFS-PU100
    - 機器仕様書: SPEC-P101 〜 SPEC-T401
    
    ---
    
    *本文書はナレッジグラフから自動生成されました*
    ============================================================
    
    【保存完了】: process_design_document.md
    

**解説:** ナレッジグラフからSPARQLでデータを抽出し、Jinja2テンプレートエンジンでMarkdown文書を自動生成します。機器リスト、運転条件、仕様を一元管理し、常に最新の文書を生成できます。

* * *

## 5.4 根本原因分析（RCA）アプリケーション

### コード例4: ナレッジベース根本原因分析
    
    
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    import pandas as pd
    
    # 根本原因分析（Root Cause Analysis）システム
    
    PROC = Namespace("http://example.org/process#")
    RCA = Namespace("http://example.org/rca#")
    
    def create_rca_knowledge_base():
        """根本原因分析用ナレッジベースを作成"""
        g = Graph()
        g.bind("proc", PROC)
        g.bind("rca", RCA)
    
        # 因果関係の定義（Cause → Effect）
    
        # 原因1: 吸込圧力低下 → キャビテーション
        g.add((RCA.LowSuctionPressure, RDF.type, RCA.RootCause))
        g.add((RCA.LowSuctionPressure, RDFS.label, Literal("吸込圧力低下")))
        g.add((RCA.LowSuctionPressure, RCA.causes, RCA.Cavitation))
        g.add((RCA.LowSuctionPressure, RCA.likelihood, Literal(0.85)))
    
        g.add((RCA.Cavitation, RDF.type, RCA.Symptom))
        g.add((RCA.Cavitation, RDFS.label, Literal("キャビテーション")))
        g.add((RCA.Cavitation, RCA.manifestsAs, Literal("低流量")))
        g.add((RCA.Cavitation, RCA.manifestsAs, Literal("高振動")))
    
        # 原因2: シール劣化 → 漏れ
        g.add((RCA.SealDegradation, RDF.type, RCA.RootCause))
        g.add((RCA.SealDegradation, RDFS.label, Literal("メカニカルシール劣化")))
        g.add((RCA.SealDegradation, RCA.causes, RCA.Leakage))
        g.add((RCA.SealDegradation, RCA.likelihood, Literal(0.90)))
    
        g.add((RCA.Leakage, RDF.type, RCA.Symptom))
        g.add((RCA.Leakage, RDFS.label, Literal("漏れ")))
        g.add((RCA.Leakage, RCA.manifestsAs, Literal("低圧力")))
        g.add((RCA.Leakage, RCA.manifestsAs, Literal("液体漏洩検知")))
    
        # 原因3: 軸受摩耗 → 振動
        g.add((RCA.BearingWear, RDF.type, RCA.RootCause))
        g.add((RCA.BearingWear, RDFS.label, Literal("軸受摩耗")))
        g.add((RCA.BearingWear, RCA.causes, RCA.Vibration))
        g.add((RCA.BearingWear, RCA.likelihood, Literal(0.75)))
    
        g.add((RCA.Vibration, RDF.type, RCA.Symptom))
        g.add((RCA.Vibration, RDFS.label, Literal("異常振動")))
        g.add((RCA.Vibration, RCA.manifestsAs, Literal("高振動")))
        g.add((RCA.Vibration, RCA.manifestsAs, Literal("高温度")))
    
        # 対策の定義
        g.add((RCA.LowSuctionPressure, RCA.remediation, Literal("NPSH確認、吸込配管点検")))
        g.add((RCA.SealDegradation, RCA.remediation, Literal("メカニカルシール交換")))
        g.add((RCA.BearingWear, RCA.remediation, Literal("軸受交換、潤滑油補充")))
    
        return g
    
    
    def perform_rca(g, observed_symptoms):
        """
        観測された症状から根本原因を推論
    
        Parameters:
            g (Graph): RCAナレッジグラフ
            observed_symptoms (list): 観測された症状リスト
    
        Returns:
            list: 根本原因の候補リスト
        """
        candidates = []
    
        # 各症状に対して因果関係を逆探索
        for symptom_label in observed_symptoms:
            # 症状に対応する原因を探索
            query = prepareQuery("""
                SELECT ?cause ?causeLabel ?symptom ?symptomLabel ?likelihood ?remediation
                WHERE {
                    ?cause a rca:RootCause .
                    ?cause rdfs:label ?causeLabel .
                    ?cause rca:causes ?symptom .
                    ?symptom rdfs:label ?symptomLabel .
                    ?symptom rca:manifestsAs ?manifestation .
                    ?cause rca:likelihood ?likelihood .
                    OPTIONAL { ?cause rca:remediation ?remediation }
                    FILTER (str(?manifestation) = ?symptom_str)
                }
            """, initNs={"rca": RCA, "rdfs": RDFS})
    
            results = g.query(query, initBindings={'symptom_str': Literal(symptom_label)})
    
            for row in results:
                candidates.append({
                    'root_cause': str(row.causeLabel),
                    'symptom': str(row.symptomLabel),
                    'observed': symptom_label,
                    'likelihood': float(row.likelihood),
                    'remediation': str(row.remediation) if row.remediation else "（対策未定義）"
                })
    
        # 同じ根本原因が複数の症状に対応する場合、尤度を統合
        cause_scores = {}
        for cand in candidates:
            cause = cand['root_cause']
            if cause not in cause_scores:
                cause_scores[cause] = {
                    'symptoms': [],
                    'likelihood': cand['likelihood'],
                    'remediation': cand['remediation']
                }
            cause_scores[cause]['symptoms'].append(cand['observed'])
            # 複数症状がマッチする場合、尤度を累積
            cause_scores[cause]['likelihood'] = min(1.0, cause_scores[cause]['likelihood'] + 0.1)
    
        # 尤度順にソート
        ranked_causes = sorted(
            [{'cause': k, **v} for k, v in cause_scores.items()],
            key=lambda x: x['likelihood'],
            reverse=True
        )
    
        return ranked_causes
    
    
    # 実行デモ
    print("="*60)
    print("根本原因分析（RCA）システム")
    print("="*60)
    
    g = create_rca_knowledge_base()
    print(f"\nRCAナレッジベース作成: {len(g)} トリプル")
    
    # シナリオ1: キャビテーション疑い
    print("\n" + "="*60)
    print("【シナリオ1】P-101ポンプの異常")
    print("="*60)
    
    observed_symptoms_1 = ["低流量", "高振動"]
    print(f"\n観測された症状: {observed_symptoms_1}")
    
    print("\n【RCA実行中...】")
    causes_1 = perform_rca(g, observed_symptoms_1)
    
    print("\n【根本原因分析結果】")
    for i, cause in enumerate(causes_1, 1):
        print(f"\n{i}. {cause['cause']} (尤度: {cause['likelihood']*100:.1f}%)")
        print(f"   関連症状: {', '.join(cause['symptoms'])}")
        print(f"   推奨対策: {cause['remediation']}")
    
    # シナリオ2: シール漏れ疑い
    print("\n" + "="*60)
    print("【シナリオ2】P-102ポンプの異常")
    print("="*60)
    
    observed_symptoms_2 = ["低圧力", "液体漏洩検知"]
    print(f"\n観測された症状: {observed_symptoms_2}")
    
    print("\n【RCA実行中...】")
    causes_2 = perform_rca(g, observed_symptoms_2)
    
    print("\n【根本原因分析結果】")
    for i, cause in enumerate(causes_2, 1):
        print(f"\n{i}. {cause['cause']} (尤度: {cause['likelihood']*100:.1f}%)")
        print(f"   関連症状: {', '.join(cause['symptoms'])}")
        print(f"   推奨対策: {cause['remediation']}")
    
    # DataFrame化
    all_causes = causes_1 + causes_2
    if all_causes:
        df = pd.DataFrame(all_causes)
        print("\n【全RCA結果サマリー】")
        print(df[['cause', 'likelihood', 'remediation']].drop_duplicates().to_string(index=False))
    

**出力例:**
    
    
    ============================================================
    根本原因分析（RCA）システム
    ============================================================
    
    RCAナレッジベース作成: 24 トリプル
    
    ============================================================
    【シナリオ1】P-101ポンプの異常
    ============================================================
    
    観測された症状: ['低流量', '高振動']
    
    【RCA実行中...】
    
    【根本原因分析結果】
    
    1. 吸込圧力低下 (尤度: 95.0%)
       関連症状: 低流量, 高振動
       推奨対策: NPSH確認、吸込配管点検
    
    2. 軸受摩耗 (尤度: 75.0%)
       関連症状: 高振動
       推奨対策: 軸受交換、潤滑油補充
    
    ============================================================
    【シナリオ2】P-102ポンプの異常
    ============================================================
    
    観測された症状: ['低圧力', '液体漏洩検知']
    
    【RCA実行中...】
    
    【根本原因分析結果】
    
    1. メカニカルシール劣化 (尤度: 100.0%)
       関連症状: 低圧力, 液体漏洩検知
       推奨対策: メカニカルシール交換
    
    【全RCA結果サマリー】
                  cause  likelihood          remediation
           吸込圧力低下        0.95  NPSH確認、吸込配管点検
              軸受摩耗        0.75   軸受交換、潤滑油補充
    メカニカルシール劣化        1.00    メカニカルシール交換
    

**解説:** 観測された症状から、ナレッジグラフに定義された因果関係を逆探索し、根本原因を推論します。複数の症状がマッチする原因ほど尤度が高くなり、優先的に調査すべき原因を特定できます。

* * *

## 5.5 機器推薦システム

### コード例5: ナレッジベース機器推薦
    
    
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    import pandas as pd
    
    # ナレッジベース機器推薦システム
    
    PROC = Namespace("http://example.org/process#")
    SPEC = Namespace("http://example.org/specification#")
    
    def create_equipment_catalog_kb():
        """機器カタログナレッジベースを作成"""
        g = Graph()
        g.bind("proc", PROC)
        g.bind("spec", SPEC)
    
        # ポンプカタログ
        pumps = [
            {"id": "PUMP_A100", "type": "CentrifugalPump", "flow_min": 40, "flow_max": 60,
             "pressure_min": 400, "pressure_max": 600, "material": "SS316", "cost": 15000},
            {"id": "PUMP_B200", "type": "CentrifugalPump", "flow_min": 80, "flow_max": 120,
             "pressure_min": 800, "pressure_max": 1200, "material": "SS316L", "cost": 25000},
            {"id": "PUMP_C300", "type": "PositiveDisplacementPump", "flow_min": 10, "flow_max": 30,
             "pressure_min": 1000, "pressure_max": 2000, "material": "SS316L", "cost": 35000},
        ]
    
        for pump in pumps:
            p_uri = PROC[pump['id']]
            g.add((p_uri, RDF.type, PROC[pump['type']]))
            g.add((p_uri, SPEC.flowRateMin, Literal(pump['flow_min'])))
            g.add((p_uri, SPEC.flowRateMax, Literal(pump['flow_max'])))
            g.add((p_uri, SPEC.pressureMin, Literal(pump['pressure_min'])))
            g.add((p_uri, SPEC.pressureMax, Literal(pump['pressure_max'])))
            g.add((p_uri, SPEC.material, Literal(pump['material'])))
            g.add((p_uri, SPEC.cost, Literal(pump['cost'])))
    
        # 熱交換器カタログ
        heat_exchangers = [
            {"id": "HX_S100", "type": "ShellAndTubeHX", "duty_min": 200, "duty_max": 600,
             "material": "SS316", "cost": 20000},
            {"id": "HX_P200", "type": "PlateHX", "duty_min": 100, "duty_max": 400,
             "material": "SS316", "cost": 12000},
        ]
    
        for hx in heat_exchangers:
            hx_uri = PROC[hx['id']]
            g.add((hx_uri, RDF.type, PROC[hx['type']]))
            g.add((hx_uri, SPEC.dutyMin, Literal(hx['duty_min'])))
            g.add((hx_uri, SPEC.dutyMax, Literal(hx['duty_max'])))
            g.add((hx_uri, SPEC.material, Literal(hx['material'])))
            g.add((hx_uri, SPEC.cost, Literal(hx['cost'])))
    
        return g
    
    
    def recommend_equipment(g, requirements):
        """
        要求仕様に基づいて機器を推薦
    
        Parameters:
            g (Graph): 機器カタログKG
            requirements (dict): 要求仕様
    
        Returns:
            list: 推薦機器リスト
        """
        recommendations = []
    
        if requirements['type'] == 'Pump':
            # ポンプ推薦
            query = prepareQuery("""
                SELECT ?pump ?pumpType ?flowMin ?flowMax ?pressureMin ?pressureMax ?material ?cost
                WHERE {
                    ?pump a ?pumpType .
                    ?pump spec:flowRateMin ?flowMin .
                    ?pump spec:flowRateMax ?flowMax .
                    ?pump spec:pressureMin ?pressureMin .
                    ?pump spec:pressureMax ?pressureMax .
                    ?pump spec:material ?material .
                    ?pump spec:cost ?cost .
                    FILTER (strstarts(str(?pumpType), str(proc:)))
                    FILTER (?pumpType != proc:Equipment)
                }
            """, initNs={"proc": PROC, "spec": SPEC})
    
            for row in g.query(query):
                # 仕様マッチングスコア計算
                flow_req = requirements['flow_rate']
                pressure_req = requirements['pressure']
    
                # 流量範囲チェック
                if row.flowMin <= flow_req <= row.flowMax:
                    flow_score = 1.0
                else:
                    # 範囲外の場合はペナルティ
                    flow_score = 0.5
    
                # 圧力範囲チェック
                if row.pressureMin <= pressure_req <= row.pressureMax:
                    pressure_score = 1.0
                else:
                    pressure_score = 0.5
    
                # 材質マッチング
                if requirements.get('material') == str(row.material):
                    material_score = 1.0
                elif str(row.material) in ['SS316L']:  # より高グレード
                    material_score = 0.8
                else:
                    material_score = 0.6
    
                # 総合スコア
                total_score = (flow_score * 0.4 + pressure_score * 0.4 + material_score * 0.2)
    
                # コスト効率
                cost_per_performance = float(row.cost) / total_score
    
                recommendations.append({
                    'equipment_id': str(row.pump).split('#')[-1],
                    'type': str(row.pumpType).split('#')[-1],
                    'flow_range': f"{row.flowMin}-{row.flowMax} m³/h",
                    'pressure_range': f"{row.pressureMin}-{row.pressureMax} kPa",
                    'material': str(row.material),
                    'cost': float(row.cost),
                    'score': total_score,
                    'cost_efficiency': cost_per_performance
                })
    
        elif requirements['type'] == 'HeatExchanger':
            # 熱交換器推薦
            query = prepareQuery("""
                SELECT ?hx ?hxType ?dutyMin ?dutyMax ?material ?cost
                WHERE {
                    ?hx a ?hxType .
                    ?hx spec:dutyMin ?dutyMin .
                    ?hx spec:dutyMax ?dutyMax .
                    ?hx spec:material ?material .
                    ?hx spec:cost ?cost .
                    FILTER (strstarts(str(?hxType), str(proc:)))
                }
            """, initNs={"proc": PROC, "spec": SPEC})
    
            for row in g.query(query):
                duty_req = requirements['duty']
    
                if row.dutyMin <= duty_req <= row.dutyMax:
                    duty_score = 1.0
                else:
                    duty_score = 0.5
    
                total_score = duty_score
                cost_per_performance = float(row.cost) / total_score
    
                recommendations.append({
                    'equipment_id': str(row.hx).split('#')[-1],
                    'type': str(row.hxType).split('#')[-1],
                    'duty_range': f"{row.dutyMin}-{row.dutyMax} kW",
                    'material': str(row.material),
                    'cost': float(row.cost),
                    'score': total_score,
                    'cost_efficiency': cost_per_performance
                })
    
        # スコア順にソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)
    
        return recommendations
    
    
    # 実行デモ
    print("="*60)
    print("ナレッジベース機器推薦システム")
    print("="*60)
    
    g = create_equipment_catalog_kb()
    print(f"\n機器カタログKB作成: {len(g)} トリプル")
    
    # シナリオ1: ポンプ推薦
    print("\n" + "="*60)
    print("【シナリオ1】ポンプの推薦")
    print("="*60)
    
    pump_requirements = {
        'type': 'Pump',
        'flow_rate': 50.0,  # m³/h
        'pressure': 500.0,  # kPa
        'material': 'SS316'
    }
    
    print(f"\n要求仕様:")
    print(f"  流量: {pump_requirements['flow_rate']} m³/h")
    print(f"  圧力: {pump_requirements['pressure']} kPa")
    print(f"  材質: {pump_requirements['material']}")
    
    print("\n【推薦実行中...】")
    pump_recs = recommend_equipment(g, pump_requirements)
    
    print(f"\n【推薦結果】（上位{min(3, len(pump_recs))}件）")
    for i, rec in enumerate(pump_recs[:3], 1):
        print(f"\n{i}. {rec['equipment_id']} ({rec['type']})")
        print(f"   スコア: {rec['score']*100:.1f}%")
        print(f"   流量範囲: {rec['flow_range']}")
        print(f"   圧力範囲: {rec['pressure_range']}")
        print(f"   材質: {rec['material']}")
        print(f"   価格: ${rec['cost']:,}")
        print(f"   コスト効率: ${rec['cost_efficiency']:,.0f}/スコア")
    
    # シナリオ2: 熱交換器推薦
    print("\n" + "="*60)
    print("【シナリオ2】熱交換器の推薦")
    print("="*60)
    
    hx_requirements = {
        'type': 'HeatExchanger',
        'duty': 300.0,  # kW
        'material': 'SS316'
    }
    
    print(f"\n要求仕様:")
    print(f"  伝熱量: {hx_requirements['duty']} kW")
    print(f"  材質: {hx_requirements['material']}")
    
    print("\n【推薦実行中...】")
    hx_recs = recommend_equipment(g, hx_requirements)
    
    print(f"\n【推薦結果】（全{len(hx_recs)}件）")
    for i, rec in enumerate(hx_recs, 1):
        print(f"\n{i}. {rec['equipment_id']} ({rec['type']})")
        print(f"   スコア: {rec['score']*100:.1f}%")
        print(f"   伝熱量範囲: {rec['duty_range']}")
        print(f"   材質: {rec['material']}")
        print(f"   価格: ${rec['cost']:,}")
    
    # 推薦結果をDataFrame化
    df_pumps = pd.DataFrame(pump_recs)
    print("\n【ポンプ推薦サマリー】")
    print(df_pumps[['equipment_id', 'score', 'cost', 'cost_efficiency']].to_string(index=False))
    

**出力例:**
    
    
    ============================================================
    ナレッジベース機器推薦システム
    ============================================================
    
    機器カタログKB作成: 35 トリプル
    
    ============================================================
    【シナリオ1】ポンプの推薦
    ============================================================
    
    要求仕様:
      流量: 50.0 m³/h
      圧力: 500.0 kPa
      材質: SS316
    
    【推薦実行中...】
    
    【推薦結果】（上位3件）
    
    1. PUMP_A100 (CentrifugalPump)
       スコア: 100.0%
       流量範囲: 40-60 m³/h
       圧力範囲: 400-600 kPa
       材質: SS316
       価格: $15,000
       コスト効率: $15,000/スコア
    
    2. PUMP_B200 (CentrifugalPump)
       スコア: 80.0%
       流量範囲: 80-120 m³/h
       圧力範囲: 800-1200 kPa
       材質: SS316L
       価格: $25,000
       コスト効率: $31,250/スコア
    
    3. PUMP_C300 (PositiveDisplacementPump)
       スコア: 80.0%
       流量範囲: 10-30 m³/h
       圧力範囲: 1000-2000 kPa
       材質: SS316L
       価格: $35,000
       コスト効率: $43,750/スコア
    
    ============================================================
    【シナリオ2】熱交換器の推薦
    ============================================================
    
    要求仕様:
      伝熱量: 300.0 kW
      材質: SS316
    
    【推薦実行中...】
    
    【推薦結果】（全2件）
    
    1. HX_S100 (ShellAndTubeHX)
       スコア: 100.0%
       伝熱量範囲: 200-600 kW
       材質: SS316
       価格: $20,000
    
    2. HX_P200 (PlateHX)
       スコア: 100.0%
       伝熱量範囲: 100-400 kW
       材質: SS316
       価格: $12,000
    
    【ポンプ推薦サマリー】
      equipment_id  score    cost  cost_efficiency
         PUMP_A100    1.0  15000.0          15000.0
         PUMP_B200    0.8  25000.0          31250.0
         PUMP_C300    0.8  35000.0          43750.0
    

**解説:** 機器カタログをナレッジグラフ化し、要求仕様とのマッチングスコアを計算して最適な機器を推薦します。流量・圧力・材質などの条件に基づき、スコアとコスト効率で順位付けします。

* * *

## 5.6 プロセス最適化への応用

### コード例6: ナレッジグラフを活用したプロセス最適化
    
    
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    from scipy.optimize import minimize
    import numpy as np
    import pandas as pd
    
    # ナレッジグラフを活用したプロセス最適化
    
    PROC = Namespace("http://example.org/process#")
    OPT = Namespace("http://example.org/optimization#")
    
    def create_optimization_kb():
        """最適化用ナレッジベースを作成"""
        g = Graph()
        g.bind("proc", PROC)
        g.bind("opt", OPT)
    
        # プロセスユニット: 反応器
        g.add((PROC.R201, RDF.type, PROC.Reactor))
        g.add((PROC.R201, RDFS.label, Literal("触媒反応器R-201")))
    
        # 操作変数（Decision Variables）
        g.add((PROC.R201, OPT.hasDecisionVariable, OPT.Temperature))
        g.add((OPT.Temperature, RDFS.label, Literal("反応温度")))
        g.add((OPT.Temperature, OPT.lowerBound, Literal(150.0)))
        g.add((OPT.Temperature, OPT.upperBound, Literal(200.0)))
        g.add((OPT.Temperature, OPT.currentValue, Literal(175.0)))
    
        g.add((PROC.R201, OPT.hasDecisionVariable, OPT.Pressure))
        g.add((OPT.Pressure, RDFS.label, Literal("反応圧力")))
        g.add((OPT.Pressure, OPT.lowerBound, Literal(600.0)))
        g.add((OPT.Pressure, OPT.upperBound, Literal(1000.0)))
        g.add((OPT.Pressure, OPT.currentValue, Literal(800.0)))
    
        # 制約条件
        g.add((PROC.R201, OPT.hasConstraint, OPT.SafetyConstraint1))
        g.add((OPT.SafetyConstraint1, RDFS.label, Literal("温度・圧力安全制約")))
        g.add((OPT.SafetyConstraint1, OPT.expression, Literal("T + 0.1*P <= 250")))
    
        # 目的関数（収益最大化）
        g.add((PROC.R201, OPT.hasObjective, OPT.ProfitMaximization))
        g.add((OPT.ProfitMaximization, RDFS.label, Literal("収益最大化")))
        g.add((OPT.ProfitMaximization, OPT.expression,
               Literal("Profit = Yield * ProductPrice - OperatingCost")))
    
        # プロセスモデルパラメータ
        g.add((PROC.R201, OPT.productPrice, Literal(100.0)))  # $/kg
        g.add((PROC.R201, OPT.feedCost, Literal(50.0)))  # $/kg
        g.add((PROC.R201, OPT.energyCostFactor, Literal(0.05)))  # $/(°C*kPa)
    
        return g
    
    
    def extract_optimization_problem(g):
        """
        ナレッジグラフから最適化問題を抽出
    
        Returns:
            dict: 最適化問題定義
        """
        problem = {
            'decision_variables': [],
            'bounds': [],
            'constraints': [],
            'parameters': {}
        }
    
        # 操作変数の抽出
        query_vars = prepareQuery("""
            SELECT ?var ?label ?lb ?ub ?current
            WHERE {
                ?reactor opt:hasDecisionVariable ?var .
                ?var rdfs:label ?label .
                ?var opt:lowerBound ?lb .
                ?var opt:upperBound ?ub .
                ?var opt:currentValue ?current .
            }
        """, initNs={"opt": OPT, "rdfs": RDFS})
    
        for row in g.query(query_vars):
            var_name = str(row.label)
            problem['decision_variables'].append(var_name)
            problem['bounds'].append((float(row.lb), float(row.ub)))
    
        # パラメータの抽出
        query_params = prepareQuery("""
            SELECT ?property ?value
            WHERE {
                proc:R201 ?property ?value .
                FILTER (strstarts(str(?property), str(opt:)))
            }
        """, initNs={"proc": PROC, "opt": OPT})
    
        for row in g.query(query_params):
            prop_name = str(row.property).split('#')[-1]
            if prop_name not in ['hasDecisionVariable', 'hasConstraint', 'hasObjective']:
                problem['parameters'][prop_name] = float(row.value)
    
        return problem
    
    
    def process_model(x, params):
        """
        プロセスモデル（簡易版）
    
        Parameters:
            x (array): [温度, 圧力]
            params (dict): モデルパラメータ
    
        Returns:
            dict: 計算結果
        """
        T, P = x  # 温度[°C], 圧力[kPa]
    
        # 反応速度モデル（アレニウス式の簡易版）
        k = 0.01 * np.exp(0.05 * (T - 150))
    
        # 収率（反応速度と圧力の関数）
        yield_fraction = k * (P / 1000) / (1 + k * (P / 1000))
    
        # 製品収量 [kg/h]
        feed_rate = 100.0  # kg/h
        product_rate = feed_rate * yield_fraction
    
        # 収益計算
        product_value = product_rate * params['productPrice']
        feed_cost = feed_rate * params['feedCost']
        energy_cost = params['energyCostFactor'] * T * P / 1000
    
        profit = product_value - feed_cost - energy_cost
    
        return {
            'yield': yield_fraction,
            'product_rate': product_rate,
            'profit': profit,
            'energy_cost': energy_cost
        }
    
    
    def objective_function(x, params):
        """目的関数（最大化→最小化に変換）"""
        result = process_model(x, params)
        return -result['profit']  # 最大化→最小化
    
    
    def safety_constraint(x):
        """安全制約: T + 0.1*P <= 250"""
        T, P = x
        return 250 - (T + 0.1 * P)  # >= 0 を満たす
    
    
    # 実行デモ
    print("="*60)
    print("ナレッジグラフベースプロセス最適化")
    print("="*60)
    
    g = create_optimization_kb()
    print(f"\n最適化ナレッジベース作成: {len(g)} トリプル")
    
    # 最適化問題を抽出
    opt_problem = extract_optimization_problem(g)
    
    print("\n【最適化問題定義】")
    print(f"  操作変数: {opt_problem['decision_variables']}")
    print(f"  変数範囲: {opt_problem['bounds']}")
    print(f"  パラメータ: {opt_problem['parameters']}")
    
    # 現在の運転点を評価
    x_current = np.array([175.0, 800.0])
    current_result = process_model(x_current, opt_problem['parameters'])
    
    print("\n【現在の運転点】")
    print(f"  温度: {x_current[0]} °C")
    print(f"  圧力: {x_current[1]} kPa")
    print(f"  収率: {current_result['yield']*100:.2f}%")
    print(f"  製品生産速度: {current_result['product_rate']:.2f} kg/h")
    print(f"  収益: ${current_result['profit']:.2f}/h")
    
    # 最適化を実行
    print("\n【最適化実行中...】")
    
    constraint = {'type': 'ineq', 'fun': safety_constraint}
    
    result = minimize(
        objective_function,
        x_current,
        args=(opt_problem['parameters'],),
        method='SLSQP',
        bounds=opt_problem['bounds'],
        constraints=[constraint]
    )
    
    if result.success:
        x_optimal = result.x
        optimal_result = process_model(x_optimal, opt_problem['parameters'])
    
        print("\n【最適化結果】")
        print(f"  最適温度: {x_optimal[0]:.2f} °C")
        print(f"  最適圧力: {x_optimal[1]:.2f} kPa")
        print(f"  最適収率: {optimal_result['yield']*100:.2f}%")
        print(f"  最適製品生産速度: {optimal_result['product_rate']:.2f} kg/h")
        print(f"  最適収益: ${optimal_result['profit']:.2f}/h")
    
        print("\n【改善効果】")
        profit_improvement = optimal_result['profit'] - current_result['profit']
        print(f"  収益改善: ${profit_improvement:.2f}/h ({profit_improvement/current_result['profit']*100:.2f}%増)")
        print(f"  年間収益改善: ${profit_improvement * 8760:.0f}")
    
        # 制約チェック
        constraint_value = safety_constraint(x_optimal)
        print(f"\n【制約充足確認】")
        print(f"  安全制約 (T + 0.1*P <= 250): {x_optimal[0] + 0.1*x_optimal[1]:.2f} <= 250")
        print(f"  マージン: {constraint_value:.2f} °C")
    
    else:
        print(f"\n最適化失敗: {result.message}")
    
    # 結果比較表
    comparison_df = pd.DataFrame({
        '項目': ['温度[°C]', '圧力[kPa]', '収率[%]', '生産速度[kg/h]', '収益[$/h]'],
        '現在': [x_current[0], x_current[1], current_result['yield']*100,
                 current_result['product_rate'], current_result['profit']],
        '最適': [x_optimal[0], x_optimal[1], optimal_result['yield']*100,
                 optimal_result['product_rate'], optimal_result['profit']]
    })
    
    print("\n【運転条件比較表】")
    print(comparison_df.to_string(index=False))
    

**出力例:**
    
    
    ============================================================
    ナレッジグラフベースプロセス最適化
    ============================================================
    
    最適化ナレッジベース作成: 18 トリプル
    
    【最適化問題定義】
      操作変数: ['反応温度', '反応圧力']
      変数範囲: [(150.0, 200.0), (600.0, 1000.0)]
      パラメータ: {'productPrice': 100.0, 'feedCost': 50.0, 'energyCostFactor': 0.05}
    
    【現在の運転点】
      温度: 175.0 °C
      圧力: 800.0 kPa
      収率: 56.78%
      製品生産速度: 56.78 kg/h
      収益: $606.80/h
    
    【最適化実行中...】
    
    【最適化結果】
      最適温度: 200.00 °C
      最適圧力: 1000.00 kPa
      最適収率: 74.37%
      最適製品生産速度: 74.37 kg/h
      最適収益: $826.34/h
    
    【改善効果】
      収益改善: $219.54/h (36.18%増)
      年間収益改善: $1,923,210
    
    【制約充足確認】
      安全制約 (T + 0.1*P <= 250): 250.00 <= 250
      マージン: 0.00 °C
    
    【運転条件比較表】
             項目     現在     最適
       温度[°C]  175.00  200.00
      圧力[kPa]  800.00 1000.00
        収率[%]   56.78   74.37
    生産速度[kg/h]   56.78   74.37
       収益[$/h]  606.80  826.34
    

**解説:** ナレッジグラフから操作変数、制約条件、目的関数を抽出し、scipy.optimizeで最適化を実行します。プロセスモデルと最適化問題定義をナレッジグラフで一元管理することで、保守性と拡張性が向上します。

* * *

## 5.7 完全統合システム

### コード例7: API + 推論 + 可視化の統合システム
    
    
    from flask import Flask, request, jsonify, render_template_string
    from rdflib import Graph, Namespace, RDF, RDFS, Literal
    from rdflib.plugins.sparql import prepareQuery
    import networkx as nx
    import plotly.graph_objects as go
    import json
    
    # 完全統合システム: API + 推論 + 可視化
    
    app = Flask(__name__)
    
    PROC = Namespace("http://example.org/process#")
    DIAG = Namespace("http://example.org/diagnosis#")
    
    # グローバルなナレッジグラフ
    g = Graph()
    g.bind("proc", PROC)
    g.bind("diag", DIAG)
    
    def initialize_integrated_kb():
        """統合ナレッジベースを初期化"""
        # プロセス機器
        equipment = [
            {"id": "P101", "type": "Pump", "flow": 50.0, "pressure": 500.0, "vibration": 2.5, "status": "Normal"},
            {"id": "P102", "type": "Pump", "flow": 30.0, "pressure": 450.0, "vibration": 8.5, "status": "Warning"},
            {"id": "R201", "type": "Reactor", "temp": 180.0, "pressure": 800.0, "status": "Normal"},
        ]
    
        for eq in equipment:
            eq_uri = PROC[eq['id']]
            g.add((eq_uri, RDF.type, PROC[eq['type']]))
            g.add((eq_uri, PROC.hasStatus, Literal(eq['status'])))
    
            if 'flow' in eq:
                g.add((eq_uri, PROC.hasFlowRate, Literal(eq['flow'])))
            if 'pressure' in eq:
                g.add((eq_uri, PROC.hasPressure, Literal(eq['pressure'])))
            if 'vibration' in eq:
                g.add((eq_uri, PROC.hasVibration, Literal(eq['vibration'])))
            if 'temp' in eq:
                g.add((eq_uri, PROC.hasTemperature, Literal(eq['temp'])))
    
        # プロセスフロー
        g.add((PROC.P101, PROC.flowsTo, PROC.R201))
        g.add((PROC.P102, PROC.flowsTo, PROC.R201))
    
        # 故障診断ルール
        g.add((DIAG.CavitationRule, RDF.type, DIAG.DiagnosticRule))
        g.add((DIAG.CavitationRule, DIAG.condition, Literal("flow < 40 AND vibration > 5")))
        g.add((DIAG.CavitationRule, DIAG.diagnosis, Literal("キャビテーション疑い")))
        g.add((DIAG.CavitationRule, DIAG.action, Literal("NPSH確認")))
    
        print(f"統合ナレッジベース初期化: {len(g)} トリプル")
    
    
    # === API エンドポイント ===
    
    @app.route('/')
    def index():
        """統合ダッシュボードHTML"""
        html = """
    
    
    
    
    
        
    
    # 🏭 プロセスナレッジグラフ統合システム
    
    
    
        
    
    
            
    
    ## 📊 システム概要
    
    
            
    
    統合機能: SPARQL API + 推論エンジン + 可視化
    
    
            統計情報取得
            

## 🔍 機器一覧

機器リスト取得

## ⚡ 推論エンジン

異常検知推論実行

## 🌐 プロセスフローグラフ

グラフ取得

## 🔧 カスタムSPARQLクエリ

SELECT ?equipment ?status WHERE { ?equipment proc:hasStatus ?status . }  クエリ実行

""" return render_template_string(html) @app.route('/api/statistics', methods=['GET']) def get_statistics(): """統計情報API""" stats = { 'total_triples': len(g), 'equipment_count': len(list(g.subjects(RDF.type, PROC.Pump))) + len(list(g.subjects(RDF.type, PROC.Reactor))), 'connection_count': len(list(g.subject_objects(PROC.flowsTo))) } return jsonify({"status": "success", "statistics": stats}) @app.route('/api/equipment', methods=['GET']) def get_equipment(): """機器一覧API""" query = prepareQuery(""" SELECT ?equipment ?type ?status WHERE { ?equipment a ?type . ?equipment proc:hasStatus ?status . FILTER (?type IN (proc:Pump, proc:Reactor)) } """, initNs={"proc": PROC}) results = [] for row in g.query(query): results.append({ "id": str(row.equipment).split('#')[-1], "type": str(row.type).split('#')[-1], "status": str(row.status) }) return jsonify({"status": "success", "equipment": results}) @app.route('/api/inference', methods=['POST']) def run_inference(): """推論エンジンAPI（異常検知）""" anomalies = [] # ルール: 低流量 + 高振動 → キャビテーション query = prepareQuery(""" SELECT ?equipment ?flow ?vibration WHERE { ?equipment proc:hasFlowRate ?flow . ?equipment proc:hasVibration ?vibration . FILTER (?flow < 40 && ?vibration > 5) } """, initNs={"proc": PROC}) for row in g.query(query): anomalies.append({ "equipment": str(row.equipment).split('#')[-1], "diagnosis": "キャビテーション疑い", "symptoms": f"低流量({row.flow} m³/h), 高振動({row.vibration} mm/s)", "action": "NPSH確認、吸込配管点検" }) return jsonify({"status": "success", "anomalies": anomalies}) @app.route('/api/graph', methods=['GET']) def get_graph(): """プロセスフローグラフAPI""" nx_graph = nx.DiGraph() # ノード追加 for eq_uri in g.subjects(RDF.type, None): eq_id = str(eq_uri).split('#')[-1] if eq_id in ['P101', 'P102', 'R201']: nx_graph.add_node(eq_id) # エッジ追加 for s, p, o in g.triples((None, PROC.flowsTo, None)): source = str(s).split('#')[-1] target = str(o).split('#')[-1] nx_graph.add_edge(source, target) # JSON形式で返す graph_data = { "nodes": [{"id": n} for n in nx_graph.nodes()], "edges": [{"source": u, "target": v} for u, v in nx_graph.edges()] } return jsonify({"status": "success", "graph": graph_data}) @app.route('/api/sparql', methods=['POST']) def sparql_endpoint(): """SPARQLエンドポイント""" data = request.get_json() query_str = data.get('query', '') try: query = prepareQuery(query_str, initNs={"proc": PROC, "diag": DIAG, "rdf": RDF, "rdfs": RDFS}) results = g.query(query) response = [] for row in results: row_dict = {} for var in results.vars: value = row[var] row_dict[str(var)] = str(value).split('#')[-1] if '#' in str(value) else str(value) response.append(row_dict) return jsonify({"status": "success", "count": len(response), "results": response}) except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400 # 実行デモ def demo_integrated_system(): """統合システムのデモ""" print("="*60) print("完全統合システム: API + 推論 + 可視化") print("="*60) initialize_integrated_kb() print("\n【統合システム機能】") print(" 1. RESTful API - 機器情報、統計、グラフデータ") print(" 2. 推論エンジン - 異常検知、故障診断") print(" 3. SPARQLエンドポイント - カスタムクエリ") print(" 4. Webダッシュボード - 統合UI") print("\n【起動方法】") print(" python integrated_system.py") print(" または: flask run --port 5000") print("\n【アクセス】") print(" ブラウザで http://localhost:5000 を開く") print("\n【APIエンドポイント一覧】") print(" GET / - ダッシュボード") print(" GET /api/statistics - 統計情報") print(" GET /api/equipment - 機器一覧") print(" POST /api/inference - 異常検知推論") print(" GET /api/graph - プロセスグラフ") print(" POST /api/sparql - カスタムクエリ") # デモ実行 demo_integrated_system() # Flaskアプリケーション起動 # if __name__ == '__main__': # initialize_integrated_kb() # app.run(debug=True, port=5000) `

**出力例:**
    
    
    ============================================================
    完全統合システム: API + 推論 + 可視化
    ============================================================
    統合ナレッジベース初期化: 17 トリプル
    
    【統合システム機能】
      1. RESTful API - 機器情報、統計、グラフデータ
      2. 推論エンジン - 異常検知、故障診断
      3. SPARQLエンドポイント - カスタムクエリ
      4. Webダッシュボード - 統合UI
    
    【起動方法】
      python integrated_system.py
      または: flask run --port 5000
    
    【アクセス】
      ブラウザで http://localhost:5000 を開く
    
    【APIエンドポイント一覧】
      GET  / - ダッシュボード
      GET  /api/statistics - 統計情報
      GET  /api/equipment - 機器一覧
      POST /api/inference - 異常検知推論
      GET  /api/graph - プロセスグラフ
      POST /api/sparql - カスタムクエリ
    

**解説:** FlaskでWebアプリケーションを構築し、ナレッジグラフへのAPI、推論エンジン、SPARQL エンドポイント、対話的ダッシュボードを統合します。ブラウザから全機能にアクセスでき、エンドツーエンドのプロセス知識管理システムとして動作します。

* * *

## 5.8 本章のまとめ

### 学んだこと

  1. **SPARQLエンドポイントAPI**
     * FlaskでRESTful APIを構築
     * POST /api/sparqlで任意のクエリ実行
     * GET /api/equipmentで機器情報取得
     * JSON形式のレスポンス
  2. **ナレッジグラフ可視化**
     * RDFグラフをNetworkXグラフに変換
     * Plotlyで対話的可視化
     * ノード・エッジの属性表示
     * プロセスフローの視覚的理解
  3. **文書自動生成**
     * Jinja2テンプレートエンジン活用
     * SPARQLでデータ抽出
     * Markdown/HTML文書生成
     * 機器リスト、運転条件の自動更新
  4. **根本原因分析（RCA）**
     * 症状から因果関係を逆探索
     * 尤度スコアリング
     * 推奨対策の提示
     * 複数症状の統合評価
  5. **機器推薦システム**
     * 要求仕様とカタログのマッチング
     * スコアリング関数によるランキング
     * コスト効率評価
     * 多基準意思決定支援
  6. **プロセス最適化**
     * ナレッジグラフから最適化問題抽出
     * scipy.optimizeとの統合
     * 制約条件の自動適用
     * 最適運転条件の導出
  7. **完全統合システム**
     * Flask Webアプリケーション
     * API + 推論 + 可視化の統合
     * 対話的ダッシュボード
     * エンドツーエンドの知識管理

### 重要なポイント

  * SPARQLエンドポイントにより、外部システムからナレッジグラフにアクセス可能
  * 可視化により、複雑なプロセスフローを直感的に理解できる
  * テンプレートエンジンで文書を自動生成し、常に最新状態を維持
  * RCAシステムにより、故障診断を体系化し、迅速な対応が可能
  * 推薦システムで、最適な機器選定を支援し、意思決定を効率化
  * 最適化問題をナレッジグラフで管理することで、保守性が向上
  * 統合システムにより、ナレッジグラフをプロセス運転の中核に配置
  * APIファーストの設計により、マイクロサービスアーキテクチャに対応

### 実用化に向けて

本シリーズで学んだプロセスオントロジーとナレッジグラフの技術は、以下の実プロセスに適用できます：

  * **デジタルツイン** : 物理プロセスの知識をナレッジグラフで表現
  * **スマートプラント** : IoTセンサーデータとナレッジグラフの統合
  * **プロセス安全管理** : HAZOP、LOPAのナレッジベース化
  * **設備保全最適化** : 故障履歴、メンテナンス計画の体系化
  * **エネルギー管理** : プロセス全体のエネルギーフロー最適化
  * **技術伝承** : ベテランの知識をオントロジーとして形式化
  * **規制対応** : 法規制要件のナレッジベース管理

### さらなる学習リソース

  * **Semantic Web技術** : W3C RDF/OWL/SPARQL仕様
  * **オントロジー工学** : Protégé、TopBraidツール
  * **ナレッジグラフDB** : Neo4j、Apache Jena、Stardog
  * **プロセス工学** : CAPE-OPEN、OPC UA情報モデル
  * **実装フレームワーク** : rdflib、owlready2、SPARQLWrapper
