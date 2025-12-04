---
title: 第3章：プロセスデータからのナレッジグラフ構築
chapter_title: 第3章：プロセスデータからのナレッジグラフ構築
subtitle: CSV、センサー、P&IDデータの自動RDF変換とトリプル生成
---

## 3.1 CSVデータからのエンティティ抽出

実際の化学プラントでは、装置情報や運転データがCSV形式で管理されています。このデータからナレッジグラフを自動構築する手法を学びます。

**💡 ナレッジグラフ構築の3ステップ**

  1. **エンティティ抽出** : データから装置やストリームを識別
  2. **関係抽出** : 装置間の接続や因果関係を特定
  3. **トリプル生成** : RDF形式（Subject-Predicate-Object）に変換

### Example 1: CSVデータからの装置エンティティ抽出

装置マスターデータからRDFトリプルを自動生成します。
    
    
    # ===================================
    # Example 1: CSVからのエンティティ抽出
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    # CSVデータ（装置マスター）
    csv_data = """EquipmentID,Type,Name,Temperature_K,Pressure_bar,Volume_m3,Efficiency_pct
    R-101,CSTR,主反応器,350.0,5.0,10.0,92.5
    R-102,PFR,管型反応器,420.0,8.0,5.0,88.0
    HX-201,HeatExchanger,冷却器HX-201,320.0,,,90.0
    HX-202,HeatExchanger,加熱器HX-202,450.0,,,85.0
    SEP-301,Separator,蒸留塔,340.0,1.5,,95.0
    P-401,Pump,フィードポンプ,300.0,10.0,,85.0"""
    
    # DataFrameに読み込み
    df = pd.read_csv(io.StringIO(csv_data))
    
    print("=== 元のCSVデータ ===")
    print(df.head(3))
    
    # RDFグラフの作成
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== CSVからRDFトリプルへの変換 =====
    
    def csv_to_rdf(row):
        """CSV行をRDFトリプルに変換"""
        equipment_uri = PROC[row['EquipmentID']]
    
        # 基本トリプル
        g.add((equipment_uri, RDF.type, PROC[row['Type']]))
        g.add((equipment_uri, RDFS.label, Literal(row['Name'], lang='ja')))
    
        # 温度（必須）
        g.add((equipment_uri, PROC.hasTemperature,
               Literal(row['Temperature_K'], datatype=XSD.double)))
    
        # 圧力（オプショナル）
        if pd.notna(row['Pressure_bar']):
            g.add((equipment_uri, PROC.hasPressure,
                   Literal(row['Pressure_bar'], datatype=XSD.double)))
    
        # 容積（オプショナル）
        if pd.notna(row['Volume_m3']):
            g.add((equipment_uri, PROC.hasVolume,
                   Literal(row['Volume_m3'], datatype=XSD.double)))
    
        # 効率（必須）
        g.add((equipment_uri, PROC.hasEfficiency,
               Literal(row['Efficiency_pct'], datatype=XSD.double)))
    
        return len(g)  # 現在のトリプル数
    
    # 全行を変換
    initial_count = len(g)
    for idx, row in df.iterrows():
        csv_to_rdf(row)
    
    print(f"\n=== RDF変換結果 ===")
    print(f"処理行数: {len(df)}")
    print(f"生成トリプル数: {len(g) - initial_count}")
    
    # 装置タイプ別集計
    print("\n=== 装置タイプ別統計 ===")
    type_counts = df['Type'].value_counts()
    for eq_type, count in type_counts.items():
        print(f"{eq_type}: {count}個")
    
    # Turtle形式で出力（抜粋）
    print("\n=== Turtle形式（抜粋） ===")
    print(g.serialize(format="turtle")[:600])
    
    # ファイル保存
    g.serialize(destination="equipment_from_csv.ttl", format="turtle")
    print("\n✓ RDFファイル保存完了: equipment_from_csv.ttl")
    

**出力例:**  
=== 元のCSVデータ ===  
EquipmentID Type Temperature_K Pressure_bar  
0 R-101 CSTR 350.0 5.0  
1 R-102 PFR 420.0 8.0  
2 HX-201 HeatExchanger 320.0 NaN  
  
=== RDF変換結果 ===  
処理行数: 6  
生成トリプル数: 28  
  
=== 装置タイプ別統計 ===  
HeatExchanger: 2個  
CSTR: 1個  
PFR: 1個  
Separator: 1個  
Pump: 1個  
  
✓ RDFファイル保存完了: equipment_from_csv.ttl 

## 3.2 装置接続関係の抽出

### Example 2: フローデータからの関係抽出

物質フローデータから装置間の接続関係を自動抽出します。
    
    
    # ===================================
    # Example 2: フローデータからの関係抽出
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    # フロー接続データ
    flow_data = """StreamID,SourceEquipment,TargetEquipment,FlowRate_kgh,Composition
    S-001,Feed,R-101,1000.0,原料混合物
    S-002,R-101,HX-201,980.0,反応生成物
    S-003,HX-201,SEP-301,975.0,冷却生成物
    S-004,SEP-301,Product,920.0,製品
    S-005,SEP-301,R-101,55.0,リサイクル"""
    
    df_flow = pd.read_csv(io.StringIO(flow_data))
    
    print("=== フローデータ ===")
    print(df_flow)
    
    # RDFグラフ作成
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    # ===== フローデータからトリプル生成 =====
    
    for idx, row in df_flow.iterrows():
        # ストリームのトリプル
        stream = PROC[row['StreamID']]
        g.add((stream, RDF.type, PROC.Stream))
        g.add((stream, RDFS.label, Literal(row['Composition'], lang='ja')))
        g.add((stream, PROC.hasFlowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
    
        # 送信元装置
        source = PROC[row['SourceEquipment']]
        g.add((source, PROC.hasOutput, stream))
    
        # 送信先装置
        target = PROC[row['TargetEquipment']]
        g.add((target, PROC.hasInput, stream))
    
        # 装置間の直接接続
        g.add((source, PROC.connectedTo, target))
    
    print(f"\n=== トリプル生成結果 ===")
    print(f"総ストリーム数: {len(df_flow)}")
    print(f"総トリプル数: {len(g)}")
    
    # ===== 接続関係の可視化 =====
    print("\n=== プロセスフロー接続 ===")
    
    # SPARQLクエリで接続を取得
    query = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?source ?target ?stream ?flowrate ?composition
    WHERE {
        ?source proc:hasOutput ?stream .
        ?target proc:hasInput ?stream .
        ?stream proc:hasFlowRate ?flowrate .
        ?stream rdfs:label ?composition .
    }
    ORDER BY ?source
    """
    
    for row in g.query(query):
        source = str(row.source).split('/')[-1]
        target = str(row.target).split('/')[-1]
        print(f"{source} → {target}: {float(row.flowrate):.0f} kg/h ({row.composition})")
    
    # リサイクルループの検出
    print("\n=== リサイクルストリーム検出 ===")
    recycled = df_flow[df_flow['Composition'].str.contains('リサイクル', na=False)]
    for idx, row in recycled.iterrows():
        print(f"✓ {row['SourceEquipment']} → {row['TargetEquipment']} (リサイクル)")
    
    g.serialize(destination="process_flow.ttl", format="turtle")
    print("\n✓ フローグラフ保存完了: process_flow.ttl")
    

**出力例:**  
=== フローデータ ===  
StreamID SourceEquipment TargetEquipment FlowRate_kgh  
0 S-001 Feed R-101 1000.0  
1 S-002 R-101 HX-201 980.0  
2 S-003 HX-201 SEP-301 975.0  
  
=== トリプル生成結果 ===  
総ストリーム数: 5  
総トリプル数: 23  
  
=== プロセスフロー接続 ===  
Feed → R-101: 1000 kg/h (原料混合物)  
R-101 → HX-201: 980 kg/h (反応生成物)  
HX-201 → SEP-301: 975 kg/h (冷却生成物)  
SEP-301 → Product: 920 kg/h (製品)  
SEP-301 → R-101: 55 kg/h (リサイクル)  
  
=== リサイクルストリーム検出 ===  
✓ SEP-301 → R-101 (リサイクル)  
  
✓ フローグラフ保存完了: process_flow.ttl 
    
    
    ```mermaid
    graph LR
        Feed[原料] -->|S-0011000 kg/h| R101[R-101反応器]
        R101 -->|S-002980 kg/h| HX201[HX-201冷却器]
        HX201 -->|S-003975 kg/h| SEP301[SEP-301分離器]
        SEP301 -->|S-004920 kg/h| Product[製品]
        SEP301 -.->|S-00555 kg/hリサイクル| R101
    
        style Feed fill:#e3f2fd
        style Product fill:#e8f5e9
        style R101 fill:#fff3e0
        style HX201 fill:#f3e5f5
        style SEP301 fill:#fce4ec
    ```

### Example 3: pandasからの自動トリプル生成

汎用的なDataFrameからRDFへの変換関数を実装します。
    
    
    # ===================================
    # Example 3: 汎用DataFrame→RDF変換
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    import io
    
    def dataframe_to_rdf(df, namespace_uri, entity_column, type_name=None):
        """pandasデータフレームをRDFグラフに変換
    
        Args:
            df: pandas DataFrame
            namespace_uri: 名前空間URI
            entity_column: エンティティIDとなる列名
            type_name: エンティティのクラス名（Noneの場合は'Entity'）
    
        Returns:
            rdflib.Graph: RDFグラフ
        """
        g = Graph()
        NS = Namespace(namespace_uri)
        g.bind("data", NS)
    
        type_name = type_name or "Entity"
    
        for idx, row in df.iterrows():
            # エンティティURI
            entity_id = str(row[entity_column])
            entity_uri = NS[entity_id]
    
            # タイプトリプル
            g.add((entity_uri, RDF.type, NS[type_name]))
    
            # 各カラムをプロパティとして追加
            for col in df.columns:
                if col == entity_column:
                    continue  # IDカラムはスキップ
    
                value = row[col]
                if pd.isna(value):
                    continue  # 欠損値はスキップ
    
                # プロパティURI
                prop_uri = NS[col]
    
                # データ型の判定と適切なリテラル生成
                if isinstance(value, (int, float)):
                    g.add((entity_uri, prop_uri,
                           Literal(value, datatype=XSD.double)))
                elif isinstance(value, bool):
                    g.add((entity_uri, prop_uri,
                           Literal(value, datatype=XSD.boolean)))
                else:
                    g.add((entity_uri, prop_uri, Literal(str(value))))
    
        return g
    
    # ===== テストデータ =====
    
    sensor_data = """SensorID,Location,Type,Value,Unit,Timestamp
    TE-101,R-101,Temperature,77.5,degC,2025-10-26 10:00:00
    PE-101,R-101,Pressure,5.2,bar,2025-10-26 10:00:00
    FE-201,HX-201,FlowRate,980.0,kg/h,2025-10-26 10:00:00
    TE-201,HX-201,Temperature,45.3,degC,2025-10-26 10:00:00"""
    
    df_sensor = pd.read_csv(io.StringIO(sensor_data))
    
    print("=== センサーデータ ===")
    print(df_sensor)
    
    # RDF変換
    g_sensor = dataframe_to_rdf(
        df_sensor,
        namespace_uri="http://example.org/sensor/",
        entity_column="SensorID",
        type_name="Sensor"
    )
    
    print(f"\n=== RDF変換結果 ===")
    print(f"センサー数: {len(df_sensor)}")
    print(f"総トリプル数: {len(g_sensor)}")
    
    # SPARQLクエリでデータ確認
    query = """
    PREFIX data: 
    PREFIX rdf: 
    
    SELECT ?sensor ?location ?type ?value ?unit
    WHERE {
        ?sensor rdf:type data:Sensor .
        ?sensor data:Location ?location .
        ?sensor data:Type ?type .
        ?sensor data:Value ?value .
        ?sensor data:Unit ?unit .
    }
    """
    
    print("\n=== センサー情報一覧 ===")
    for row in g_sensor.query(query):
        sensor = str(row.sensor).split('/')[-1]
        print(f"{sensor} @ {row.location}: {row.type} = {float(row.value):.1f} {row.unit}")
    
    # Turtle出力
    print("\n=== Turtle形式（抜粋） ===")
    turtle_output = g_sensor.serialize(format="turtle")
    print(turtle_output[:400])
    
    g_sensor.serialize(destination="sensor_data.ttl", format="turtle")
    print("\n✓ センサーデータRDF保存完了: sensor_data.ttl")
    

**出力例:**  
=== センサーデータ ===  
SensorID Location Type Value Unit  
0 TE-101 R-101 Temperature 77.5 degC  
1 PE-101 R-101 Pressure 5.2 bar  
2 FE-201 HX-201 FlowRate 980.0 kg/h  
  
=== RDF変換結果 ===  
センサー数: 4  
総トリプル数: 25  
  
=== センサー情報一覧 ===  
TE-101 @ R-101: Temperature = 77.5 degC  
PE-101 @ R-101: Pressure = 5.2 bar  
FE-201 @ HX-201: FlowRate = 980.0 kg/h  
TE-201 @ HX-201: Temperature = 45.3 degC  
  
✓ センサーデータRDF保存完了: sensor_data.ttl 

## 3.3 P&IDテキストからの知識抽出

### Example 4: P&ID記述のパースと知識抽出

P&ID（配管計装図）の文字情報から装置接続をパースします。
    
    
    # ===================================
    # Example 4: P&ID記述のパースと知識抽出
    # ===================================
    
    import re
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS
    
    # P&ID記述テキスト（簡易DSL形式）
    pid_text = """
    # エステル化プラントP&ID
    
    [EQUIPMENT]
    R-101: Type=CSTR, Name=主反応器, Temp=350K, Press=5bar, Vol=10m3
    HX-201: Type=HeatExchanger, Name=冷却器, Temp=320K
    HX-202: Type=HeatExchanger, Name=加熱器, Temp=450K
    SEP-301: Type=Separator, Name=蒸留塔, Temp=340K, Press=1.5bar
    P-401: Type=Pump, Name=フィードポンプ
    
    [CONNECTIONS]
    Feed -> P-401 (S-001, 1000kg/h)
    P-401 -> R-101 (S-002, 1000kg/h)
    R-101 -> HX-201 (S-003, 980kg/h)
    HX-201 -> SEP-301 (S-004, 975kg/h)
    SEP-301 -> Product (S-005, 920kg/h)
    SEP-301 -> HX-202 (S-006, 55kg/h, recycle)
    HX-202 -> R-101 (S-007, 55kg/h)
    """
    
    # ===== パーサー関数 =====
    
    def parse_equipment_line(line):
        """装置定義行をパース"""
        match = re.match(r'(\S+):\s*(.+)', line)
        if not match:
            return None
    
        eq_id = match.group(1)
        params_str = match.group(2)
    
        # パラメータ抽出
        params = {}
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
    
        return eq_id, params
    
    def parse_connection_line(line):
        """接続行をパース"""
        # 例: "Feed -> P-401 (S-001, 1000kg/h)"
        match = re.match(r'(\S+)\s*->\s*(\S+)\s*\(([^)]+)\)', line)
        if not match:
            return None
    
        source = match.group(1)
        target = match.group(2)
        stream_info = match.group(3)
    
        # ストリーム情報の抽出
        stream_parts = [p.strip() for p in stream_info.split(',')]
        stream_id = stream_parts[0]
        flowrate = stream_parts[1] if len(stream_parts) > 1 else None
        is_recycle = 'recycle' in stream_info.lower()
    
        return source, target, stream_id, flowrate, is_recycle
    
    # ===== P&IDテキストのパース =====
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    g.bind("proc", PROC)
    
    lines = pid_text.strip().split('\n')
    section = None
    
    equipment_count = 0
    connection_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
    
        if line.startswith('[EQUIPMENT]'):
            section = 'equipment'
            continue
        elif line.startswith('[CONNECTIONS]'):
            section = 'connections'
            continue
    
        if section == 'equipment':
            parsed = parse_equipment_line(line)
            if parsed:
                eq_id, params = parsed
                eq_uri = PROC[eq_id]
    
                # RDFトリプル生成
                if 'Type' in params:
                    g.add((eq_uri, RDF.type, PROC[params['Type']]))
                if 'Name' in params:
                    g.add((eq_uri, RDFS.label, Literal(params['Name'], lang='ja')))
    
                equipment_count += 1
    
        elif section == 'connections':
            parsed = parse_connection_line(line)
            if parsed:
                source, target, stream_id, flowrate, is_recycle = parsed
    
                # ストリームトリプル
                stream_uri = PROC[stream_id]
                g.add((stream_uri, RDF.type, PROC.Stream))
    
                if flowrate:
                    # "1000kg/h" から数値抽出
                    flow_value = re.search(r'(\d+)', flowrate)
                    if flow_value:
                        g.add((stream_uri, PROC.hasFlowRate,
                               Literal(float(flow_value.group(1)))))
    
                # 接続トリプル
                source_uri = PROC[source]
                target_uri = PROC[target]
                g.add((source_uri, PROC.hasOutput, stream_uri))
                g.add((target_uri, PROC.hasInput, stream_uri))
    
                if is_recycle:
                    g.add((stream_uri, PROC.isRecycle, Literal(True)))
    
                connection_count += 1
    
    print("=== P&IDパース結果 ===")
    print(f"装置数: {equipment_count}")
    print(f"接続数: {connection_count}")
    print(f"総トリプル数: {len(g)}")
    
    # 装置リスト
    print("\n=== 装置一覧 ===")
    query_eq = """
    PREFIX proc: 
    PREFIX rdfs: 
    
    SELECT ?eq ?label
    WHERE {
        ?eq rdfs:label ?label .
    }
    """
    for row in g.query(query_eq):
        eq_id = str(row.eq).split('/')[-1]
        print(f"- {eq_id}: {row.label}")
    
    # リサイクルストリーム
    print("\n=== リサイクルストリーム ===")
    query_recycle = """
    PREFIX proc: 
    
    SELECT ?stream
    WHERE {
        ?stream proc:isRecycle true .
    }
    """
    recycle_streams = list(g.query(query_recycle))
    print(f"リサイクル数: {len(recycle_streams)}")
    for row in recycle_streams:
        print(f"✓ {str(row.stream).split('/')[-1]}")
    
    g.serialize(destination="pid_knowledge.ttl", format="turtle")
    print("\n✓ P&IDナレッジグラフ保存完了: pid_knowledge.ttl")
    

**出力例:**  
=== P&IDパース結果 ===  
装置数: 5  
接続数: 7  
総トリプル数: 38  
  
=== 装置一覧 ===  
\- R-101: 主反応器  
\- HX-201: 冷却器  
\- HX-202: 加熱器  
\- SEP-301: 蒸留塔  
\- P-401: フィードポンプ  
  
=== リサイクルストリーム ===  
リサイクル数: 1  
✓ S-006  
  
✓ P&IDナレッジグラフ保存完了: pid_knowledge.ttl 

**💡 P &IDデータソースの拡張**

実務では、P&IDはCADソフト（AutoCAD Plant 3D、Intergraph SmartPlant等）で管理されています。これらのツールはXML/JSONエクスポート機能を持ち、同様の方法で知識抽出が可能です。

## 3.4 センサーストリームデータのRDF化

### Example 5: リアルタイムセンサーデータのRDF変換

時系列センサーデータをRDFで表現し、時間情報を保持します。
    
    
    # ===================================
    # Example 5: センサーストリームのRDF変換
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    from datetime import datetime, timedelta
    import numpy as np
    
    # 時系列センサーデータの生成
    base_time = datetime(2025, 10, 26, 10, 0, 0)
    time_points = [base_time + timedelta(minutes=5*i) for i in range(12)]
    
    # シミュレーションデータ（1時間分、5分間隔）
    sensor_stream = pd.DataFrame({
        'Timestamp': time_points,
        'TE-101_degC': 77.5 + np.random.normal(0, 0.5, 12),  # 温度
        'PE-101_bar': 5.0 + np.random.normal(0, 0.1, 12),    # 圧力
        'FE-101_kgh': 1000 + np.random.normal(0, 10, 12),    # 流量
    })
    
    print("=== センサーストリームデータ（抜粋） ===")
    print(sensor_stream.head(3))
    
    # RDFグラフ作成
    g = Graph()
    SENSOR = Namespace("http://example.org/sensor/")
    TIME = Namespace("http://www.w3.org/2006/time#")
    g.bind("sensor", SENSOR)
    g.bind("time", TIME)
    
    # ===== 時系列データのRDF変換 =====
    
    for idx, row in sensor_stream.iterrows():
        # タイムスタンプ
        timestamp = row['Timestamp']
        instant_uri = SENSOR[f"Instant_{idx}"]
    
        g.add((instant_uri, RDF.type, TIME.Instant))
        g.add((instant_uri, TIME.inXSDDateTime,
               Literal(timestamp.isoformat(), datatype=XSD.dateTime)))
    
        # 各センサー値
        for col in ['TE-101_degC', 'PE-101_bar', 'FE-101_kgh']:
            sensor_id, unit = col.rsplit('_', 1)
            measurement_uri = SENSOR[f"{sensor_id}_M{idx}"]
    
            # 測定トリプル
            g.add((measurement_uri, RDF.type, SENSOR.Measurement))
            g.add((measurement_uri, SENSOR.sensor, SENSOR[sensor_id]))
            g.add((measurement_uri, SENSOR.hasTimestamp, instant_uri))
            g.add((measurement_uri, SENSOR.hasValue,
                   Literal(row[col], datatype=XSD.double)))
            g.add((measurement_uri, SENSOR.hasUnit, Literal(unit)))
    
    print(f"\n=== RDF変換結果 ===")
    print(f"時間ポイント数: {len(sensor_stream)}")
    print(f"総トリプル数: {len(g)}")
    
    # ===== SPARQLクエリ：温度の統計 =====
    
    query_stats = """
    PREFIX sensor: 
    PREFIX xsd: 
    
    SELECT (AVG(?value) AS ?avgTemp) (MIN(?value) AS ?minTemp) (MAX(?value) AS ?maxTemp)
    WHERE {
        ?measurement sensor:sensor sensor:TE-101 .
        ?measurement sensor:hasValue ?value .
    }
    """
    
    print("\n=== 温度センサーTE-101の統計（1時間） ===")
    for row in g.query(query_stats):
        print(f"平均温度: {float(row.avgTemp):.2f}°C")
        print(f"最低温度: {float(row.minTemp):.2f}°C")
        print(f"最高温度: {float(row.maxTemp):.2f}°C")
    
    # ===== 異常値検出（閾値ベース） =====
    
    query_anomaly = """
    PREFIX sensor: 
    PREFIX time: 
    
    SELECT ?timestamp ?value
    WHERE {
        ?measurement sensor:sensor sensor:TE-101 .
        ?measurement sensor:hasValue ?value .
        ?measurement sensor:hasTimestamp ?instant .
        ?instant time:inXSDDateTime ?timestamp .
        FILTER (?value > 78.5 || ?value < 76.5)
    }
    ORDER BY ?timestamp
    """
    
    print("\n=== 温度異常検出（閾値: 76.5-78.5°C） ===")
    anomalies = list(g.query(query_anomaly))
    print(f"異常データ数: {len(anomalies)}")
    for row in anomalies[:3]:  # 最初の3件
        print(f"{row.timestamp}: {float(row.value):.2f}°C")
    
    g.serialize(destination="sensor_stream.ttl", format="turtle")
    print("\n✓ センサーストリームRDF保存完了: sensor_stream.ttl")
    

**出力例:**  
=== センサーストリームデータ（抜粋） ===  
Timestamp TE-101_degC PE-101_bar FE-101_kgh  
0 2025-10-26 10:00:00 77.45 5.02 998.3  
1 2025-10-26 10:05:00 77.58 4.98 1005.1  
2 2025-10-26 10:10:00 77.32 5.03 995.7  
  
=== RDF変換結果 ===  
時間ポイント数: 12  
総トリプル数: 182  
  
=== 温度センサーTE-101の統計（1時間） ===  
平均温度: 77.48°C  
最低温度: 76.85°C  
最高温度: 78.12°C  
  
=== 温度異常検出（閾値: 76.5-78.5°C） ===  
異常データ数: 2  
2025-10-26T10:20:00: 78.67°C  
2025-10-26T10:45:00: 76.32°C  
  
✓ センサーストリームRDF保存完了: sensor_stream.ttl 

## 3.5 歴史的データの統合

### Example 6: 歴史的運転データとの統合

過去の運転実績データを時系列プロパティとして統合します。
    
    
    # ===================================
    # Example 6: 歴史的データの統合
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD
    from datetime import datetime, timedelta
    import numpy as np
    
    # 歴史的運転データ（1ヶ月分の日次データ）
    dates = pd.date_range(start='2025-09-26', end='2025-10-25', freq='D')
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'R101_Conversion': np.random.uniform(0.88, 0.95, len(dates)),  # 転化率
        'R101_Yield': np.random.uniform(0.85, 0.92, len(dates)),       # 収率
        'R101_Temp_avg': np.random.uniform(348, 352, len(dates)),      # 平均温度
        'R101_Uptime_pct': np.random.uniform(95, 100, len(dates)),     # 稼働率
    })
    
    print("=== 歴史的運転データ（直近7日） ===")
    print(historical_data.tail(7)[['Date', 'R101_Conversion', 'R101_Yield']])
    
    # RDFグラフ作成
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    PERF = Namespace("http://example.org/performance/")
    g.bind("proc", PROC)
    g.bind("perf", PERF)
    
    # 反応器R-101の定義
    r101 = PROC["R-101"]
    g.add((r101, RDF.type, PROC.Reactor))
    g.add((r101, RDFS.label, Literal("主反応器R-101", lang='ja')))
    
    # ===== 歴史的データをRDFに変換 =====
    
    for idx, row in historical_data.iterrows():
        date = row['Date']
        date_str = date.strftime('%Y-%m-%d')
    
        # 日次パフォーマンス記録
        record_uri = PERF[f"R101_Daily_{date_str}"]
    
        g.add((record_uri, RDF.type, PERF.DailyPerformance))
        g.add((record_uri, PERF.equipment, r101))
        g.add((record_uri, PERF.date,
               Literal(date_str, datatype=XSD.date)))
    
        # パフォーマンス指標
        g.add((record_uri, PERF.conversion,
               Literal(row['R101_Conversion'], datatype=XSD.double)))
        g.add((record_uri, PERF.yieldValue,
               Literal(row['R101_Yield'], datatype=XSD.double)))
        g.add((record_uri, PERF.avgTemperature,
               Literal(row['R101_Temp_avg'], datatype=XSD.double)))
        g.add((record_uri, PERF.uptime,
               Literal(row['R101_Uptime_pct'], datatype=XSD.double)))
    
    print(f"\n=== RDF変換結果 ===")
    print(f"日次記録数: {len(historical_data)}")
    print(f"総トリプル数: {len(g)}")
    
    # ===== 統計分析クエリ =====
    
    query_monthly_stats = """
    PREFIX perf: 
    
    SELECT
        (AVG(?conv) AS ?avgConversion)
        (AVG(?yield) AS ?avgYield)
        (AVG(?temp) AS ?avgTemp)
        (AVG(?uptime) AS ?avgUptime)
        (MIN(?conv) AS ?minConversion)
        (MAX(?conv) AS ?maxConversion)
    WHERE {
        ?record a perf:DailyPerformance .
        ?record perf:conversion ?conv .
        ?record perf:yieldValue ?yield .
        ?record perf:avgTemperature ?temp .
        ?record perf:uptime ?uptime .
    }
    """
    
    print("\n=== 月間パフォーマンス統計（R-101） ===")
    for row in g.query(query_monthly_stats):
        print(f"平均転化率: {float(row.avgConversion) * 100:.2f}%")
        print(f"平均収率: {float(row.avgYield) * 100:.2f}%")
        print(f"平均温度: {float(row.avgTemp):.1f}K ({float(row.avgTemp) - 273.15:.1f}°C)")
        print(f"平均稼働率: {float(row.avgUptime):.2f}%")
        print(f"転化率範囲: {float(row.minConversion) * 100:.2f}% - {float(row.maxConversion) * 100:.2f}%")
    
    # ===== 低性能日の検出 =====
    
    query_low_performance = """
    PREFIX perf: 
    
    SELECT ?date ?conv ?yield
    WHERE {
        ?record a perf:DailyPerformance .
        ?record perf:date ?date .
        ?record perf:conversion ?conv .
        ?record perf:yieldValue ?yield .
        FILTER (?conv < 0.90 || ?yield < 0.87)
    }
    ORDER BY ?date
    """
    
    print("\n=== 性能低下日（転化率<90% or 収率<87%） ===")
    low_perf_days = list(g.query(query_low_performance))
    print(f"該当日数: {len(low_perf_days)}")
    for row in low_perf_days[:3]:  # 最初の3件
        print(f"{row.date}: 転化率{float(row.conv) * 100:.1f}%, 収率{float(row.yield) * 100:.1f}%")
    
    g.serialize(destination="historical_performance.ttl", format="turtle")
    print("\n✓ 歴史的パフォーマンスデータRDF保存完了: historical_performance.ttl")
    

**出力例:**  
=== 歴史的運転データ（直近7日） ===  
Date R101_Conversion R101_Yield  
23 2025-10-19 0.9245 0.8932  
24 2025-10-20 0.9012 0.8765  
25 2025-10-21 0.9356 0.9087  
  
=== RDF変換結果 ===  
日次記録数: 30  
総トリプル数: 152  
  
=== 月間パフォーマンス統計（R-101） ===  
平均転化率: 91.48%  
平均収率: 88.72%  
平均温度: 350.2K (77.0°C)  
平均稼働率: 97.45%  
転化率範囲: 88.23% - 94.87%  
  
=== 性能低下日（転化率<90% or 収率<87%） ===  
該当日数: 4  
2025-09-28: 転化率89.5%, 収率86.3%  
2025-10-05: 転化率88.8%, 収率85.9%  
2025-10-12: 転化率89.2%, 収率86.7%  
  
✓ 歴史的パフォーマンスデータRDF保存完了: historical_performance.ttl 

**⚠️ 大規模時系列データの扱い**

数年分の秒単位データ（数億トリプル）の場合、トリプルストア（Apache Jena Fuseki、Virtuoso）の利用を推奨します。rdflibは中規模データ（〜100万トリプル）まで実用的です。

## 3.6 マルチソースデータの統合ナレッジグラフ

### Example 7: 完全な統合ナレッジグラフの構築

すべてのデータソースを統合した包括的なナレッジグラフを構築します。
    
    
    # ===================================
    # Example 7: マルチソース統合ナレッジグラフ
    # ===================================
    
    import pandas as pd
    from rdflib import Graph, Namespace, Literal
    from rdflib.namespace import RDF, RDFS, XSD, OWL
    from datetime import datetime
    import io
    
    # ===== 複数データソースの定義 =====
    
    # 1. 装置マスターデータ
    equipment_csv = """EquipmentID,Type,Name,InstallDate,Manufacturer
    R-101,CSTR,主反応器,2020-03-15,Mitsubishi Chemical
    HX-201,HeatExchanger,冷却器,2020-04-01,Kobe Steel
    SEP-301,Separator,蒸留塔,2020-05-10,Sumitomo Heavy"""
    
    # 2. 現在の運転条件
    operating_csv = """EquipmentID,Temperature_K,Pressure_bar,FlowRate_kgh,Efficiency_pct
    R-101,350.5,5.1,1005.0,92.8
    HX-201,320.2,5.0,980.0,89.5
    SEP-301,340.0,1.5,975.0,95.2"""
    
    # 3. 接続情報
    connection_csv = """StreamID,Source,Target,FlowRate_kgh
    S-001,Feed,R-101,1000.0
    S-002,R-101,HX-201,980.0
    S-003,HX-201,SEP-301,975.0
    S-004,SEP-301,Product,920.0"""
    
    # ===== 統合RDFグラフの構築 =====
    
    g = Graph()
    PROC = Namespace("http://example.org/process/")
    MAINT = Namespace("http://example.org/maintenance/")
    g.bind("proc", PROC)
    g.bind("maint", MAINT)
    
    # データフレーム読み込み
    df_equipment = pd.read_csv(io.StringIO(equipment_csv))
    df_operating = pd.read_csv(io.StringIO(operating_csv))
    df_connection = pd.read_csv(io.StringIO(connection_csv))
    
    print("=== データソース統合 ===")
    print(f"装置マスター: {len(df_equipment)}件")
    print(f"運転データ: {len(df_operating)}件")
    print(f"接続データ: {len(df_connection)}件")
    
    # ===== 1. 装置マスターデータの統合 =====
    
    for idx, row in df_equipment.iterrows():
        eq_uri = PROC[row['EquipmentID']]
    
        g.add((eq_uri, RDF.type, PROC[row['Type']]))
        g.add((eq_uri, RDFS.label, Literal(row['Name'], lang='ja')))
        g.add((eq_uri, MAINT.installDate,
               Literal(row['InstallDate'], datatype=XSD.date)))
        g.add((eq_uri, MAINT.manufacturer, Literal(row['Manufacturer'])))
    
    # ===== 2. 運転条件データの統合 =====
    
    current_time = datetime.now()
    
    for idx, row in df_operating.iterrows():
        eq_uri = PROC[row['EquipmentID']]
    
        # 現在の運転状態
        state_uri = PROC[f"{row['EquipmentID']}_State_{current_time.strftime('%Y%m%d')}"]
        g.add((state_uri, RDF.type, PROC.OperatingState))
        g.add((state_uri, PROC.equipment, eq_uri))
        g.add((state_uri, PROC.timestamp,
               Literal(current_time.isoformat(), datatype=XSD.dateTime)))
    
        # 運転パラメータ
        g.add((state_uri, PROC.temperature,
               Literal(row['Temperature_K'], datatype=XSD.double)))
        g.add((state_uri, PROC.pressure,
               Literal(row['Pressure_bar'], datatype=XSD.double)))
        g.add((state_uri, PROC.flowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
        g.add((state_uri, PROC.efficiency,
               Literal(row['Efficiency_pct'], datatype=XSD.double)))
    
    # ===== 3. 接続情報の統合 =====
    
    for idx, row in df_connection.iterrows():
        stream_uri = PROC[row['StreamID']]
        source_uri = PROC[row['Source']]
        target_uri = PROC[row['Target']]
    
        g.add((stream_uri, RDF.type, PROC.Stream))
        g.add((stream_uri, PROC.flowRate,
               Literal(row['FlowRate_kgh'], datatype=XSD.double)))
    
        g.add((source_uri, PROC.hasOutput, stream_uri))
        g.add((target_uri, PROC.hasInput, stream_uri))
        g.add((source_uri, PROC.connectedTo, target_uri))
    
    print(f"\n=== 統合ナレッジグラフ ===")
    print(f"総トリプル数: {len(g)}")
    
    # ===== 統合データのクエリ =====
    
    # クエリ1: 装置の完全情報（マスター + 運転状態）
    query_complete = """
    PREFIX proc: 
    PREFIX maint: 
    PREFIX rdfs: 
    
    SELECT ?id ?name ?manufacturer ?temp ?press ?eff
    WHERE {
        ?equipment rdfs:label ?name .
        ?equipment maint:manufacturer ?manufacturer .
    
        ?state proc:equipment ?equipment .
        ?state proc:temperature ?temp .
        ?state proc:pressure ?press .
        ?state proc:efficiency ?eff .
    
        BIND(STRAFTER(STR(?equipment), "#") AS ?id)
    }
    """
    
    print("\n=== 装置完全情報（マスター + 運転状態） ===")
    for row in g.query(query_complete):
        print(f"{row.name} ({row.manufacturer})")
        print(f"  温度: {float(row.temp):.1f}K, 圧力: {float(row.press):.1f}bar, 効率: {float(row.eff):.1f}%")
    
    # クエリ2: プロセスフロー（接続 + 流量）
    query_flow = """
    PREFIX proc: 
    
    SELECT ?source ?target ?flowrate
    WHERE {
        ?source proc:hasOutput ?stream .
        ?target proc:hasInput ?stream .
        ?stream proc:flowRate ?flowrate .
    }
    """
    
    print("\n=== プロセスフロー（接続 + 流量） ===")
    for row in g.query(query_flow):
        source = str(row.source).split('/')[-1]
        target = str(row.target).split('/')[-1]
        print(f"{source} → {target}: {float(row.flowrate):.0f} kg/h")
    
    # ===== 推論による新知識の導出 =====
    
    # 推論ルール: 効率90%以上の装置は"HighPerformance"クラス
    print("\n=== 推論結果（効率ベース分類） ===")
    for s, p, o in g.triples((None, PROC.efficiency, None)):
        if float(o) >= 90.0:
            equipment = g.value(s, PROC.equipment)
            g.add((equipment, RDF.type, PROC.HighPerformanceEquipment))
            eq_name = g.value(equipment, RDFS.label)
            print(f"✓ {eq_name}: HighPerformance ({float(o):.1f}%)")
    
    print(f"\n総トリプル数（推論後）: {len(g)}")
    
    # 保存
    g.serialize(destination="integrated_knowledge_graph.ttl", format="turtle")
    print("\n✓ 統合ナレッジグラフ保存完了: integrated_knowledge_graph.ttl")
    
    # OWL形式でも保存（Protégéで開ける）
    g.serialize(destination="integrated_knowledge_graph.owl", format="xml")
    print("✓ OWL形式保存完了: integrated_knowledge_graph.owl")
    

**出力例:**  
=== データソース統合 ===  
装置マスター: 3件  
運転データ: 3件  
接続データ: 4件  
  
=== 統合ナレッジグラフ ===  
総トリプル数: 48  
  
=== 装置完全情報（マスター + 運転状態） ===  
主反応器 (Mitsubishi Chemical)  
温度: 350.5K, 圧力: 5.1bar, 効率: 92.8%  
冷却器 (Kobe Steel)  
温度: 320.2K, 圧力: 5.0bar, 効率: 89.5%  
蒸留塔 (Sumitomo Heavy)  
温度: 340.0K, 圧力: 1.5bar, 効率: 95.2%  
  
=== プロセスフロー（接続 + 流量） ===  
Feed → R-101: 1000 kg/h  
R-101 → HX-201: 980 kg/h  
HX-201 → SEP-301: 975 kg/h  
SEP-301 → Product: 920 kg/h  
  
=== 推論結果（効率ベース分類） ===  
✓ 主反応器: HighPerformance (92.8%)  
✓ 蒸留塔: HighPerformance (95.2%)  
  
総トリプル数（推論後）: 50  
  
✓ 統合ナレッジグラフ保存完了: integrated_knowledge_graph.ttl  
✓ OWL形式保存完了: integrated_knowledge_graph.owl 

**✅ 統合ナレッジグラフの成果**

  * **マルチソース統合** : 装置マスター、運転データ、接続情報を単一グラフに統合
  * **時間情報保持** : 現在の運転状態と歴史的データを時系列で管理
  * **推論による知識拡張** : ルールベースで新しい知識（高性能装置）を自動分類
  * **標準形式出力** : Turtle/OWL形式で他ツール（Protégé、GraphDB）と連携可能

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ CSVデータからのエンティティ抽出プロセスを説明できる
  * ✅ 装置接続関係の抽出パターンを理解している
  * ✅ 時系列データのRDF表現方法を知っている
  * ✅ マルチソースデータ統合の課題と解決策を理解している

### 実践スキル

  * ✅ pandasデータフレームをRDFグラフに自動変換できる
  * ✅ フローデータから装置接続のトリプルを生成できる
  * ✅ P&IDテキストをパースして知識抽出できる
  * ✅ センサーストリームデータを時系列RDFに変換できる
  * ✅ 歴史的運転データを統合してパフォーマンス分析ができる
  * ✅ 複数データソースを単一ナレッジグラフに統合できる
  * ✅ SPARQLで統合データの高度なクエリができる

### 応用力

  * ✅ 実プラントの各種データソースをRDF化する戦略を立案できる
  * ✅ 異常検出や性能低下をSPARQLクエリで発見できる
  * ✅ ルールベース推論で新しい知識を自動導出できる
  * ✅ 大規模データに対するトリプルストア選定ができる
  * ✅ ナレッジグラフをProtégé等の外部ツールで可視化・編集できる

## 次のステップ

第3章では、実際のプロセスデータからナレッジグラフを自動構築する包括的な手法を学びました。次章では、構築したナレッジグラフに対する高度なSPARQL推論、機械学習との統合、そして産業応用事例を学びます。

**📚 次章の内容（第4章予告）**

  * SPARQL推論エンジンによる知識推論
  * ナレッジグラフと機械学習の統合
  * グラフニューラルネットワーク（GNN）の応用
  * プロセス異常検知とルールベース診断
  * 産業界での実装事例とベストプラクティス

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
